# -*- coding: utf-8 -*-
# credit: the code below is modified from https://github.com/m-hahn/tabula-rasa-rnns
import errno
import glob
import json
import logging
import os.path

import torch

from .model import Model

# set weights path
file_path = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(file_path, "weight")

# find model weight in lmcut/weight
tar_paths = os.path.join(WEIGHTS_DIR, "*.pth.tar")
weight_paths = glob.glob(tar_paths)

# if there are many models in the directory, pick the first one
if weight_paths:
    weight_path = os.path.basename(weight_paths[0])

else:
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "Model weight is not found")

# best model (supports only Thai language) : "Tokenizer_2019-11-11_04.24.19"
# best model (supports Thai + English) : "Tokenizer_2022-05-04_10.22.18"
WEIGHT_FILE_NAME = weight_path[:-8]
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHT_FILE_NAME + ".pth.tar")
PARAMS = os.path.join(WEIGHTS_DIR, WEIGHT_FILE_NAME + ".json")

if WEIGHT_FILE_NAME == "Tokenizer_2019-11-11_04.24.19":
    # define vocabulary
    thai_chars = 'กขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ'
    chars = thai_chars + "."
    itos = ['<START>'] + ['<END>'] + ['<UNK>'] + list(chars)
    stoi = {v: k for k, v in enumerate(itos)}
else:
    from train.data_utils import stoi, itos

cuda = torch.cuda.is_available()
logging.info("GPU available", torch.cuda.is_available())

tokenizer = None


def _load_args(args_path):
    with open(args_path) as f:
        args_dict = json.load(f)
    return args_dict


def tokenize(text):
    global tokenizer
    if not tokenizer:
        tokenizer = LM_CUT()
    return tokenizer.tokenize(text)


# not used
def _remove_white_space(text):
    # remove white space before tokenize the text
    text = text.replace(" ", "")
    text = text.replace("\t", "")
    text = text.replace("\n", "")
    return text


class LM_CUT:

    def __init__(self):
        args_dict = _load_args(PARAMS)
        char_embedding_size = args_dict["char_embedding_size"]
        hidden_dim = args_dict["hidden_dim"]
        layer_num = args_dict["layer_num"]
        lstm_num_direction = args_dict["lstm_num_direction"]
        self.batchSize = 1

        self.model = Model(char_embedding_size, hidden_dim, layer_num, lstm_num_direction, cuda, len(itos))
        self._load_weight(self.model, WEIGHTS_PATH)
        self.model.rnn.train(False)

    def _load_weight(self, model, weights_path):
        named_modules = {"rnn": model.rnn, "char_embeddings": model.char_embeddings,
                         "output_classifier": model.output_classifier}

        if weights_path is not None:
            if cuda:
                checkpoint = torch.load(weights_path)
            else:
                checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

            for name, module in named_modules.items():
                module.load_state_dict(checkpoint[name])

    def _create_tensor_classifier(self, text):
        input_list = []
        for char in text:
            input_list.append(stoi[char] if char in stoi else 2)
        sequence_length = len(text)
        input_tensor = input_list

        if cuda:
            input_tensor = torch.LongTensor(input_tensor).view(self.batchSize, -1, sequence_length).transpose(0,
                                                                                                              1).transpose(
                1, 2).cuda()

        else:
            input_tensor = torch.LongTensor(input_tensor).view(self.batchSize, -1, sequence_length).transpose(0,
                                                                                                              1).transpose(
                1, 2)
        yield input_tensor[0]

    # for tokenize only Thai characters
    def tokenize(self, text):
        if not text:
            return [""]

        ฃtext = _remove_white_space(text)
        out_put = []
        sequence_length = len(text)
        test_chars = self._create_tensor_classifier(text)

        while True:
            try:
                numeric = next(test_chars)
            except StopIteration:
                break
            log_probs_cal, input_tensor_cal = self.model._forward(numeric)

            tag_score = log_probs_cal
            flat_input = input_tensor_cal.transpose(1, 0)
            tag_score = tag_score.transpose(0, 1)
            val, argmax = tag_score.max(dim=2)

            chars = []
            output = flat_input[0]
            for i, elem in enumerate(output):
                # if char = <UNK>, then output the original character(not <UNK> token)
                if elem.item() == 2:
                    chars.append(text[i])
                else:
                    chars.append(itos[elem.item()])

            pred_seq = chars.copy()
            pred = argmax[0]

            for i in range(sequence_length):
                if pred[i] == 1:
                    pred_seq[i] = " " + pred_seq[i]
            out_put_line = "".join(pred_seq).split()
            out_put = out_put + out_put_line
        return out_put
