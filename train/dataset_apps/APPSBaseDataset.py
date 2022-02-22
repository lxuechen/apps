"""
Dataset to be used for APPS Training

formats:
    call-based
    standard

TODO for converting to mercury
    - token limit:
        - throw away examples with long q_str
        - throw away examples with long combined q_str + a_str
    - avoid the in-context learning jazz
    - data loading is complicated!
        - need to generate paths after unzipping the folder!
"""

import gc
import io
import logging
import os
import random

from dataset_lm.reindent import run as run_reindent
import torch
from tqdm import tqdm
import transformers


class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, problem_dirs, mode, max_tokens, sample_mode):
        """

        Args:
            sample_mode: `uniform_prob` samples over problem. `uniform_sol` samples over problem-solution pair..
                `uniform_sol` isn't actually sampling; just plain sequential processing!
        """
        self.data_root = data_root
        self.problem_dirs = problem_dirs  # Loaded from train/test split json files

        self.mode = mode
        self.sample_mode = sample_mode  # Either "uniform_sol" or "uniform_prob"
        self.max_tokens = max_tokens

        self.samples = None  # Set in `initialize`.
        self.samples_dict = None  # Set in `initialize`. Maps problem fname to example. Multi-solution for one problem.
        self.initialize()

        if 'EleutherAI' in mode or '2700' in mode:
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        elif 'gpt' in self.mode:  # Should handle GPT-2 and GPT-Neo
            self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(mode)
        elif self.mode in {'codebert'}:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        else:
            raise NotImplementedError()

    def initialize(self):
        """Populate `self.samples` and `self.samples_dict`."""
        all_samples = []
        skipped_problems = []

        all_samples_dict = {}  # Mapping from question_fname to list of samples

        print(f"Loading {len(self.problem_dirs)} problems from {self.data_root}.")
        for problem_name in tqdm(self.problem_dirs):
            question_fname = os.path.join(self.data_root, problem_name, "question.txt")
            sols_fname = os.path.join(self.data_root, problem_name, "solutions.json")
            starter_code = os.path.join(self.data_root, problem_name, "starter_code.py")

            if os.path.exists(starter_code):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue

            # Load starter code if there's any.
            if os.path.isfile(starter_code):
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Read the question description.
            with open(question_fname, 'r') as f:
                question_str = f.read()

            # Read all the solutions.
            with open(sols_fname, 'r') as f:
                sols_str_list = json.load(f)
                for sol_str in sols_str_list:
                    sol_str = reindent_code(sol_str)
                    sample = (question_str, starter_code, sol_str, answer_type)
                    all_samples.append(sample)
                    if question_str in all_samples_dict:
                        all_samples_dict[question_str].append(sample)
                    else:
                        all_samples_dict[question_str] = [sample]

        logging.warning(f"Loaded {len(all_samples)} samples from {self.data_root}.")
        logging.warning(f"Skipped {len(skipped_problems)} problems from {self.data_root}.")

        self.samples = all_samples
        self.samples_dict = all_samples_dict

    def __len__(self):
        return len(self.samples)

    def pack_samples(self, idx: int) -> list:
        """Repeatedly pick question, answer pairs from self.data_root until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        SINGLE_STR_LIMIT = 150000  # Ad-hoc constant from original codebase.

        curr_num_tokens = 0
        curr_samples = []

        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = self.samples[idx]
        elif self.sample_mode == 'uniform_prob':
            curr_q = random.choice(list(self.samples_dict.keys()))
            curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples_dict[curr_q])
        else:
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:
            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:SINGLE_STR_LIMIT]
            curr_s = curr_s[:SINGLE_STR_LIMIT]
            curr_a = curr_a[:SINGLE_STR_LIMIT]

            if self.mode in {'codebert'}:
                curr_q = curr_q.replace('\t', '\0')
                curr_s = curr_s.replace('\t', '\0')
                curr_a = curr_a.replace('\t', '\0')

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))

            # Prepare for next round, if you can't reach the budget!
            if self.sample_mode == 'uniform_sol':
                curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples)
            elif self.sample_mode == 'uniform_prob':
                curr_q = random.choice(list(self.samples_dict.keys()))
                curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples_dict[curr_q])
            else:
                raise NotImplementedError()

        return curr_samples

    def __getitem__(self, idx):
        raw_samples = self.pack_samples(idx)
        if 'gpt' in self.mode:
            retval = self.sample_gpt_task(raw_samples)
        else:
            raise NotImplementedError()

        gc.collect()
        return retval

    def sample_gpt_task(self, raw_samples):
        """Create the true sample used for the GPT task."""
        input_ids = []
        label_ids = []

        for q_str, s_str, a_str, answer_type in raw_samples:
            # Loss is not calculated on this
            q_str = "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            answer_token_ids.append(self.tokenizer.eos_token_id)

            input_ids.extend(question_token_ids)
            input_ids.extend(answer_token_ids)

            label_ids.extend([-100] * len(question_token_ids))
            label_ids.extend(answer_token_ids)

        # Cut off the excess; ad-hoc but from the original codebase!
        input_ids = input_ids[:self.max_tokens]
        label_ids = label_ids[:self.max_tokens]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "labels": torch.LongTensor(label_ids),
        }


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr,
        ret,
        config={
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()


if __name__ == '__main__':
    import json

    data_root = os.path.join(os.path.expanduser('~'), "data/apps")

    # --- lxuechen ---
    with open(os.path.join(data_root, "data_split/train.json")) as f:
        fnames = json.load(f)
    # ---

    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    dataset = APPSBaseDataset(
        data_root=data_root,
        problem_dirs=fnames,
        mode='gpt2',
        max_tokens=1024,
        sample_mode="uniform_sol"
    )

    print_limit = 10
    for i, e in enumerate(dataset):
        if i >= print_limit:
            break

        print(e.keys())
        import pdb;

        pdb.set_trace()
        print("------- input_ids ------------------------------------------------------------------------------------")
        print(tokenizer.decode(e['input_ids']))
        print("------- labels ------------------------------------------------------------------------------------")
        labels = e['labels']
        labels[labels == -100] = tokenizer.eos_token_id
        labels_str = tokenizer.decode(labels)
        print(labels_str)
