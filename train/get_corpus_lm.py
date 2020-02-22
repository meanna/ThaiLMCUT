import random
random.seed(4)
import set_path
# define the dataset paths for the language model

path = set_path.HOME
def get_path_data(dataset_size="small"):

    # datasets for out-of-domain experiment
    # E.g. --dataset_size ty_without_novel,  ty_without_news
    # ty_without_novel means the corpus consists of trustyou + 3 domains from BEST2009 without novel domain
    if dataset_size[:11] == "ty_without_":
        path_domain = path+"/data/TY_dataset/domain_exp/TY_without_each_domain_9-1/"
        dom =  dataset_size[11:]
        print(" train on ty_without_"+dom)
        train_path = path_domain + "TY_without_"+dom+"_train_0.9.txt"
        dev_path = path_domain + "TY_without_"+dom+"_dev_0.1.txt"
        test_path = ""

    elif dataset_size == "small":
        print("train on small dataset")
        train_path = path+"/data/TY_dataset/trustyou_toy_train_2000.txt"
        dev_path = path+"/data/TY_dataset/trustyou_toy_dev_1000.txt"
        test_path = ""

    elif dataset_size == "sample":
        print("train on sample set")
        train_path = path+"/sample_data/trustyou_toy_train_2000.txt"
        dev_path = path+"/sample_data/trustyou_toy_dev_1000.txt"
        test_path = ""

    elif dataset_size == "big":
        print("train on big dataset")
        train_path = path+"/data/TY_dataset/trustyou_toy_train_4000.txt"
        dev_path = path+"/data/TY_dataset/trustyou_toy_dev_2000.txt"
        test_path = ""

    elif dataset_size == "trustyou":
        print("train on full trustyou dataset")
        train_path = path+"/data/TY_dataset/trustyou_train_0.9.txt"
        dev_path = path+"/data/TY_dataset/trustyou_dev_0.1.txt"
        test_path = ""

    elif dataset_size == "trustyou_best":
        print(" train on trustyou+ BEST(train, dev) dataset")
        train_path = path+"/data/TY_dataset/TY_BEST_train_0.9.txt"
        dev_path = path+"/data/TY_dataset/TY_BEST_dev_0.1.txt"
        test_path = ""

    else:
        print(dataset_size)
        raise AssertionError("the given dataset name is wrong")
    return train_path, dev_path, test_path


def load(path_corpus, doShuffling=False, len_chunk=100):
    chunks = []
    with open(path_corpus, "r") as inFile:
        print("data from :", path_corpus)
        for line in inFile:
            chunks.append(line.strip())
            if len(chunks) > len_chunk:
                if doShuffling:
                    random.shuffle(chunks)
                yield "".join(chunks)
                chunks = []
    yield "".join(chunks)
