# -*- coding: utf-8 -*-
import random
import os

import re
from itertools import chain

from set_path import DATA

"""
some part of the code is from attacut
"""

random.seed(4)


# define the dataset paths for the language model

def get_path_data_LM(dataset="default"):
    # please change train/dev/test dataset before training
    if dataset == "default":
        train_path = os.path.join(DATA, "TEST_100K.txt")
        dev_path = os.path.join(DATA, "TEST_100K.txt")
        test_path = os.path.join(DATA, "TEST_100K.txt")
    elif dataset == "ty":
        train_path = os.path.join(DATA, "ty", "trustyou_dev_0.1.txt")  # trustyou_train_0.9.txt
        dev_path = os.path.join(DATA, "ty", "trustyou_dev_0.1.txt")
        test_path = os.path.join(DATA, "ty", "trustyou_dev_0.1.txt")

    else:
        raise AssertionError("the given dataset name is wrong :", dataset)
    return train_path, dev_path, test_path


def get_path_data_tokenizer(dataset="default"):
    # please change train/dev/test dataset before training
    if dataset == "default":
        train_path = os.path.join(DATA, "toy", "train")
        dev_path = os.path.join(DATA, "toy", "dev")
        test_path = os.path.join(DATA, "toy", "test")
    elif dataset == "best":
        train_path = os.path.join(DATA, "best_dataset_split", "train")
        dev_path = os.path.join(DATA, "best_dataset_split", "dev")
        test_path = os.path.join(DATA, "best_dataset_split", "test")
    else:
        raise AssertionError("the given dataset name is wrong :", dataset)
    return train_path, dev_path, test_path


# following code for text preprocessing is modified from AttaCut

# you can train the LM using any text. If you use BEST2009, preprocessing is needed
# the tags must be removed
preprocessing_LM = ["remove_url", "remove_newline"]

preprocessing_tokenizer = ["remove_poem", "remove_tags", "remove_url", "remove_newline"]


# preprocessing_tokenizer = ["remove_tags", "remove_url", "remove_newline"]


def keep_only_thai_chars(text):
    pattern = re.compile(u'[^|\nกขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ ]')
    text = re.sub(pattern, "", text)
    return text


def keep_thai_chars_and_fullstops(text):  # new line included

    pattern = re.compile(u'[^|\.\nกขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ ]')
    text = re.sub(pattern, "", text)
    return text


def remove_spaces(text):
    text = re.sub(" +", "", text)
    text = re.sub("(\|){2,}", "|", text)
    return text


def remove_poem(text):
    text = re.sub(r'^\<POEM>.*[\r\n]*\</POEM>\|', '', text, flags=re.MULTILINE)
    return text


def remove_url(text):
    text = text.lower()
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    all_urls = re.compile(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*')
    text = re.sub(all_urls, '', text)
    return text


def remove_tags(txt):
    return re.sub(r"<\/?[A-Z]+>", "", txt)


def remove_newline(txt):
    return re.sub(r"\n", "", txt)


def preprocess(text, steps):
    for step in steps:
        if isinstance(step, str):
            text = globals()[step](text)
        elif callable(step):
            text = step(text)
    return text


# the code below is modified from https://github.com/m-hahn/tabula-rasa-rnns

# input: a text file (if using BEST2009, you can adjust the preprocessing step to remove the tags)
# output: a string containing "len_chunk" number of lines
def load_data_LM(path_corpus, doShuffling=False, len_chunk=100):
    chunks = []
    with open(path_corpus, "r") as inFile:
        for line in inFile:
            line = line.strip()
            line = preprocess(line, preprocessing_LM)
            chunks.append(line)
            if len(chunks) > len_chunk:
                if doShuffling:
                    random.shuffle(chunks)
                # print("".join(chunks))
                yield "".join(chunks)
                chunks = []
    yield "".join(chunks)


# input: BEST2009
# output: a generator of lists of words
def load_data_tokenizer(dir_path, doShuffling=True, len_chunk=100):
    # path is a dir path
    chunks = []
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, "r") as inFile:
            for line in inFile:
                line = preprocess(line, preprocessing_tokenizer)

                words = line.split("|")
                words = [word for word in words if word != ""]

                chunks.append(words)

                if len(chunks) == len_chunk:
                    if doShuffling:
                        random.shuffle(chunks)
                    chunks_ = list(chain.from_iterable(chunks))
                    # print(chunks_)
                    yield chunks_
                    chunks = []
            chunks_ = list(chain.from_iterable(chunks))
            # print(chunks_)
            yield chunks_
