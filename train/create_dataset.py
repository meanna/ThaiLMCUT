# -*- coding: utf-8 -*-
import glob
import os
import random
import math
import shutil

random.seed(4)


def create_datasets(source_dir, output_dir, train_size=0.8, dev_size=0.1, test_size=0.1):
    domains = os.listdir(source_dir)

    file_path_all_domains = []
    for domain in domains:
        path = os.path.join(source_dir, domain, "*.txt")
        file_paths = glob.glob(path)
        file_path_all_domains += file_paths

    random.shuffle(file_path_all_domains)

    train_split = math.ceil(len(file_path_all_domains) * train_size)
    train_files = file_path_all_domains[:train_split]
    print("number of files in train split : ", len(train_files))

    dev_split = math.ceil(len(file_path_all_domains) * dev_size)
    dev_files = file_path_all_domains[train_split:train_split + dev_split]
    print("number of files in dev split : ", len(dev_files))

    test_files = file_path_all_domains[train_split + dev_split:]
    print("number of files in test split : ", len(test_files))

    assert len(file_path_all_domains) == len(train_files) + len(dev_files) + len(test_files)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    split_to_dir = {}
    for split in ["train", "dev", "test"]:
        split_path = os.path.join(output_dir, split)

        split_to_dir[split] = split_path
        if os.path.exists(split_path):
            # print("remove ",split_path)
            shutil.rmtree(split_path)
        # print("create ", split_path)
        os.makedirs(split_path)

    split_to_files = {"train": train_files, "dev": dev_files, "test": test_files}

    for split, files in split_to_files.items():
        for file_path in files:
            file_name = (os.path.basename(file_path))

            target_path = os.path.join(split_to_dir[split], file_name)
            # print(target_path)
            shutil.copyfile(file_path, target_path)
    print("done..")
    print("split data is in", output_dir)


BEST_dataset_dir = os.path.join("..", "data", "best")
# BEST_dataset_dir contains folders of txt files
output_dir = os.path.join("..", "data", "best_dataset_split")
create_datasets(BEST_dataset_dir, output_dir)
