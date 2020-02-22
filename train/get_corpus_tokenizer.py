# define the dataset paths for the tokenizer
import set_path

path = set_path.PATH
def get_path_data(dataset_size="small"):

    # out-of-domain
    if dataset_size[:6] == "domain":
        name= dataset_size[7:].split("-")
        print(dataset_size)
        train_path = path+"/data/BEST_data/BEST_dot_cleaned_full/split_9-1/"+ name[0]+ "_train_0.9.txt"
        dev_path = path+"/data/BEST_data/BEST_dot_cleaned_full/split_9-1/"+name[0]+"_dev_0.1.txt"
        test_path = path+"/data/BEST_data/BEST_dot_cleaned_full/" + name[1] + "_cleaned_merged.txt"


    elif dataset_size == "small":
        print("train on small dataset")
        train_path = "../data/news_00001.txt"
        dev_path = "../data/news_00001.txt"
        test_path = "../data/news_00001.txt"

        #train_path = path+"/data/BEST_dataset/BEST_toy_train_3000.txt"
        #dev_path = path+"/data/BEST_dataset/BEST_toy_dev_2000.txt"
        #test_path = path+"/data/BEST_dataset/BEST_toy_test_2000.txt"

    elif dataset_size == "sample":
        print("train on sample set")
        train_path = path+"sample_data/BEST_toy_train_3000.txt"
        dev_path = path+"/sample_data/BEST_toy_dev_2000.txt"
        test_path = path+"/sample_data/BEST_toy_test_2000.txt"

    elif dataset_size == "fast_test":
        print("1000 lines data set")
        train_path = path+"/data/BEST_dataset/BEST_small_test_1000.txt"
        dev_path = path+"/data/BEST_dataset/BEST_small_test_1000.txt"
        test_path = path+"/data/BEST_dataset/BEST_small_test_1000.txt"

    elif dataset_size == "test_100_line":
        print(dataset_size)
        train_path = path+"/data/BEST_dataset/train_100.txt"
        dev_path = path+"/data/BEST_dataset/dev_100.txt"
        test_path = path+"/data/BEST_dataset/test_100.txt"

    elif dataset_size == "big":
        print("train on big dataset")
        train_path = path+"/data/BEST_dataset/BEST_toy_train_8000.txt"
        dev_path = path+"/data/BEST_dataset/BEST_toy_dev_4000.txt"
        test_path = path+"/data/BEST_dataset/BEST_toy_test_4000.txt"

    elif dataset_size == "best_full":
        print("train on 80% of BEST data ")
        train_path = path+"/data/BEST_dataset/BEST_dot_full_train_0.8.txt"
        dev_path = path+"/data/BEST_dataset/BEST_dot_full_dev_0.1.txt"
        test_path = path+"/data/BEST_dataset/BEST_dot_full_test_0.1.txt"

    elif dataset_size == "BEST_40":
        print("train on 40% of BEST data")

        train_path = path+"/data/BEST_dataset/BEST_dot_train_0.4.txt"
        dev_path = path+"/data/BEST_dataset/BEST_dot_full_dev_0.1.txt"
        test_path = path+"/data/BEST_dataset/BEST_dot_full_test_0.1.txt"
    elif dataset_size == "BEST_20":
        print("train on 20% of BEST data")

        train_path = path+"/data/BEST_dataset/BEST_dot_train_0.2.txt"
        dev_path = path+"/data/BEST_dataset/BEST_dot_full_dev_0.1.txt"
        test_path = path+"/data/BEST_dataset/BEST_dot_full_test_0.1.txt"
    elif dataset_size == "BEST_10":
        print("train on 10% of BEST data")

        train_path = path+"/data/BEST_dataset/BEST_dot_train_0.1.txt"
        dev_path = path+"/data/BEST_dataset/BEST_dot_full_dev_0.1.txt"
        test_path = path+"/data/BEST_dataset/BEST_dot_full_test_0.1.txt"
    elif dataset_size == "BEST_5":
        print("train on 5% of BEST data")

        train_path = path+"/data/BEST_dataset/BEST_dot_train_0.05.txt"
        dev_path = path+"/data/BEST_dataset/BEST_dot_full_dev_0.1.txt"
        test_path = path+"/data/BEST_dataset/BEST_dot_full_test_0.1.txt"

    else:
        raise AssertionError("the given dataset name is wrong")
    return train_path, dev_path, test_path
