import os

from set_path import HOME

# define the dataset paths for the tokenizer
path = HOME
def get_path_data(dataset_size="default"):

    if dataset_size == "default":
        print("train on sample set")
        train_path = os.path.join(HOME,"data", "news_00001.txt")
        dev_path =os.path.join(HOME,"data", "news_00001.txt")
        test_path = ""

    else:
        print(dataset_size)
        raise AssertionError("the given dataset name is wrong")
    return train_path, dev_path, test_path