import random
import os

#from .set_path import HOME
from set_path import HOME

random.seed(4)

# define the dataset paths for the language model

def get_path_data(dataset="default"):

    if dataset == "default":
        train_path = os.path.join(HOME,"data", "news_00001.txt")
        dev_path =os.path.join(HOME,"data", "news_00001.txt")
        test_path = ""

    else:
        raise AssertionError("the given dataset name is wrong :",dataset )
    return train_path, dev_path, test_path

def load(path_corpus, doShuffling=False, len_chunk=100):
    chunks = []
    with open(path_corpus, "r") as inFile:
        for line in inFile:
            chunks.append(line.strip())
            if len(chunks) > len_chunk:
                if doShuffling:
                    random.shuffle(chunks)
                yield "".join(chunks)
                chunks = []
    yield "".join(chunks)
