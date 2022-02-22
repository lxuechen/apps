import json
import os
import pathlib

# TODO: insert path to train and test
#   path should be relative to root directory or absolute paths
dataroot = "/Users/xuechenli/data/apps/"

paths_to_probs = [os.path.join(dataroot, "APPS/train"), os.path.join(dataroot, "APPS/test")]
names = ["train", "test"]

data_split_dir = os.path.join(dataroot, 'data_split')
os.makedirs(data_split_dir, exist_ok=True)

def create_split(split="train", name="train"):
    paths = []
    roots = sorted(os.listdir(split))
    for folder in roots:
        root_path = os.path.join(split, folder)
        paths.append(root_path)

    target_path = os.path.join(data_split_dir, name + ".json")
    with open(target_path, "w") as f:
        json.dump(paths, f)

    return paths


all_paths = []
for index in range(len(paths_to_probs)):
    all_paths.extend(create_split(split=paths_to_probs[index], name=names[index]))

target_path = os.path.join(data_split_dir, "train_and_test.json")
with open(target_path, "w") as f:
    print(f"Writing all paths. Length = {len(all_paths)}")
    json.dump(all_paths, f)
