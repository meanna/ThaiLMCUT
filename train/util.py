import re
import json

def load_args(args_path):
    with open(args_path + ".json") as f:
        args_dict = json.load(f)
    return args_dict

def export_args(args_dict, save_path):
    with open(save_path + ".json", "w") as config:
        json.dump(args_dict, config)

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
    param = re.sub(r'\'', "", param)

    pattern8 = re.compile(
        r'--hidden_dim \d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--len_lines_per_chunk \d* |--optim \w* |--lstm_num_direction \d |--epoch \d+ |--with_dot \d |--dataset_size \w+ |--load_from \w+ ')

    short = list(set(re.findall(pattern8, param)))
    short.sort()

    short = "".join(short)
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
    pattern8 = re.compile(
        r'--hidden_dim \d* |--char_embedding_size \d* |--layer_num \d* |--learning_rate \d\.\d* |--sequence_length \d* |--char_dropout_prob \d\.\d* |--batchSize \d* |--save_to \S* |--len_lines_per_chunk \d* |--optim \w* |--lstm_num_direction \d ')
    p = list(set(re.findall(pattern8, param)))
    p.sort()
    p = "".join(p)
    p = re.sub(r'--save_to', "--load_from", p)
    return p
