import re
import json


# args_path = CHECKPOINTS_TOKENIZER + weights_file_name + ".json"

def load_args(args_path):
    with open(args_path + ".json") as f:
        args_dict = json.load(f)
    return args_dict


def export_args(args_dict, save_path):
    with open(save_path + ".json", "w") as config:
        json.dump(args_dict, config)
        print("json")


def get_command(param):
    param = re.sub(r'Namespace\(', "--", param)
    param = re.sub(r'\)', "", param)
    param = re.sub(r'\n', " ", param)

    param = re.sub(r' --save_to \S*', "", param)
    param = re.sub(r'=', " ", param)
    param = re.sub(r', ', " --", param)

    param = re.sub(r'--verbose \w* ', "", param)
    param = re.sub(r'--load_from None', "", param)
    long = param
    # --load_from None

    param = re.sub(r'\'', "", param)

    pattern8 = re.compile(
        r'--hidden_dim \d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--len_lines_per_chunk \d* |--optim \w* |--lstm_num_direction \d |--epoch \d+ |--with_dot \d |--dataset_size \w+ |--load_from \w+ ')

    short = list(set(re.findall(pattern8, param)))
    short.sort()

    short = "".join(short)
    # p = re.sub(r'--save_to', "--load_from", p)

    return long, short


def get_param(param):
    param = re.sub(r'Namespace\(', "--", param)
    param = re.sub(r'\)', "", param)
    param = re.sub(r'\n', " ", param)

    param = re.sub(r' --save_to \S*', "", param)
    param = re.sub(r'=', " ", param)
    param = re.sub(r', ', " --", param)

    param = re.sub(r'--verbose \w* ', "", param)
    param = re.sub(r'\'', "", param)

    # print(param)

    pattern8 = re.compile(
        r'--hidden_dim \d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--save_to \S* |--len_lines_per_chunk \d* |--optim \w* |--lstm_num_direction \d ')

    # pattern8 = re.compile(
    # r'--hidden_dim \d* |--weight_dropout_hidden \d\.\d* |--char_noise_prob \d\.\d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--weight_dropout_in \d\.\d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--save_to \S* |--len_lines_per_chunk \d* ')

    # print("-------------")
    p = list(set(re.findall(pattern8, param)))
    p.sort()

    p = "".join(p)
    p = re.sub(r'--save_to', "--load_from", p)
    return p


def get_param2(param):
    pattern1 = re.compile(r'Namespace\(')
    pattern2 = re.compile(r'\)')
    pattern3 = re.compile(r'\n')
    pattern4 = re.compile(r' --save_to \S*')
    pattern5 = re.compile(r'=')
    pattern6 = re.compile(r', ')
    pattern7 = re.compile(r'--verbose \w* ')
    pattern9 = re.compile(r'\'')
    pattern10 = re.compile(r'--save_to')

    param = re.sub(pattern1, " --", param)
    param = re.sub(pattern2, "", param)
    param = re.sub(pattern3, " ", param)
    # print(re.findall(pattern4, param))
    param = re.sub(pattern4, "", param)
    param = re.sub(pattern5, " ", param)
    param = re.sub(pattern6, " --", param)
    # print(re.findall(pattern7, param))
    param = re.sub(pattern7, "", param)
    param = re.sub(pattern9, "", param)
    # param = re.sub(pattern10,"--load_from",param)
    # print(param)

    pattern8 = re.compile(
        r'--language \w* |--hidden_dim \d* |--weight_dropout_hidden \d\.\d* |--char_noise_prob \d\.\d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--weight_dropout_in \d\.\d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--save_to \S* ')

    # print("-------------")

    p = list(set(re.findall(pattern8, param)))
    p.sort()
    # print(p)
    # print("-----------", len(p))
    p = " ".join(p)
    p = re.sub(pattern10, "--load_from", p)
    return p


def print_param(param):
    pattern1 = re.compile(r'Namespace\(')
    pattern2 = re.compile(r'\)')
    pattern3 = re.compile(r'\n')
    pattern4 = re.compile(r' --save_to \S*')
    pattern5 = re.compile(r'=')
    pattern6 = re.compile(r', ')
    pattern7 = re.compile(r'--verbose \w* ')
    pattern9 = re.compile(r'\'')
    pattern10 = re.compile(r'--save_to')

    param = re.sub(pattern1, " --", param)
    param = re.sub(pattern2, "", param)
    param = re.sub(pattern3, " ", param)
    # print(re.findall(pattern4, param))
    param = re.sub(pattern4, "", param)
    param = re.sub(pattern5, " ", param)
    param = re.sub(pattern6, " --", param)

    param = re.sub(pattern7, "", param)
    param = re.sub(pattern9, "", param)

    pattern8 = re.compile(
        r'--language \w* |--hidden_dim \d* |--weight_dropout_hidden \d\.\d* |--char_noise_prob \d\.\d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--weight_dropout_in \d\.\d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--save_to \S* ')

    # print("-------------")

    p = list(set(re.findall(pattern8, param)))
    p.sort()
    # print(p)
    # print("-----------", len(p))
    p = " ".join(p)
    p = re.sub(pattern10, "--load_from", p)
    print(p)
