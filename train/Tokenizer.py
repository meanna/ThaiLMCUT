import argparse
import re
import torch
import math
import time
import random
from torch.autograd import Variable
import data_LM
import get_corpus_tokenizer, util
from itertools import chain
from timeit import default_timer as timer
from datetime import timedelta
from set_path import CHECKPOINTS_LM, CHECKPOINTS_TOKENIZER
import logging
logging.basicConfig(filename='myapp.log', level=logging.INFO)

timestr = time.strftime("%Y-%m-%d_%H.%M.%S")

start = timer()

parser = argparse.ArgumentParser()
parser.add_argument("--load_from", type=str)
parser.add_argument("--save_to", type=str, default="Tokenizer_" + timestr)

# model parameters
parser.add_argument("--lstm_num_direction", type=int, default=1)
parser.add_argument("--batchSize", type=int, default=30)
parser.add_argument("--sequence_length", type=int, default=100)

# layer and dimensions
parser.add_argument("--char_embedding_size", type=int, default=10)
parser.add_argument("--hidden_dim", type=int, default=10)
parser.add_argument("--layer_num", type=int, default=1)

# dropout
parser.add_argument("--char_dropout_prob", type=float, default=random.choice([0.01]))
parser.add_argument("--clip_grad", type=float, default=0.5)

# training parameters
parser.add_argument("--learning_rate", type=float, default=1)
parser.add_argument("--optim", type=str, default="adam")  # sgd or adam
parser.add_argument("--sgd_momentum", type=float, default=0.02)  # 0.02, 0.9
parser.add_argument("--adam_lr_decay", type=float, default=0.00)
parser.add_argument("--lr_decay", type=float, default=0.01)
parser.add_argument("--epoch", type=int, default=1)

# dataset parameters
parser.add_argument("--dataset", type=str, default="small")  # small, big, full
parser.add_argument("--len_lines_per_chunk", type=int, default=100)

# log file parameters
parser.add_argument("--save_weight", type=int, default=0) # 1 for saving weight, 0 otherwise
# if resume the training, 1 for overwrite the weights to the same model, 0 for saving a new model
parser.add_argument("--over_write", type=int, default=0)
parser.add_argument("--add_note", type=str)

# 1 if include fullstops(dot) in the text input, 0 otherwise
parser.add_argument("--with_dot", type=int, default=0)
# 1 for printing the output during test time
parser.add_argument("--printPrediction", type=int, default=0)

args = parser.parse_args()
args_dict = vars(args)


start_training = True

# if resume the training
if args.load_from is not None:
    if str(args.load_from)[:9] == "Tokenizer":
        logging.info("resume training the tokenizer..." + args.load_from)
        CHECKPOINTS = CHECKPOINTS_TOKENIZER  # path to the folder that store the trained model, if any
        if args.over_write == 1:
            args.save_to = args.load_from  # overwrite the weights
            logging.info("overwrite the weights to ", args.save_to)

    # if start training a new model (with and without downloading LM)
    elif str(args.load_from)[:2] == "LM":
        CHECKPOINTS = CHECKPOINTS_LM
        logging.info("download the language model from " + args.load_from)

    # get the network structure from the loaded model
    if True:
        args_dict = util._load_args(CHECKPOINTS + args.load_from)
        args.char_embedding_size = args_dict["char_embedding_size"]
        args.hidden_dim = args_dict["hidden_dim"]
        args.layer_num = args_dict["layer_num"]
        # learning_rate = args_dict["learning_rate"]
        args.clip_grad = args_dict["clip_grad"]
        args.sequence_length = args_dict["sequence_length"]
        args.batchSize = args_dict["batchSize"]
        args.lstm_num_direction = args_dict["lstm_num_direction"]
        # sgd_momentum = args_dict["sgd_momentum"]
        args.len_lines_per_chunk = args_dict["len_lines_per_chunk"]
        args.optim = args_dict["optim"]
        logging.info("get the network structure from the loaded model...")

# set the default note
if args.add_note is None:
    args.add_note = "load from "+ str(args.load_from)+" , "+str(args.dataset)+ " , lr "+ str(args.learning_rate)+ ", epoch "+ str(args.epoch)

logging.info(args)
logging.info("save to ", args.save_to)

no_dot = args.with_dot == 0
logging.info("remove fullstops from the input data ", no_dot)
save_weights = args.save_weight == 1
bi_lstm = args.lstm_num_direction == 2
adam_with_lr_decay = args.adam_lr_decay != 0

cuda = torch.cuda.is_available()
logging.info("cuda....,", torch.cuda.is_available())

# print the last tokenization output for every batch
printPrediction = args.printPrediction == 1
printAllPrediction = False

logging.info("train for ", args.epoch, " epoch")
util.export_args(args_dict, CHECKPOINTS_TOKENIZER + args.save_to)
logging.info("export parameters as json")

# ------------------ define vocabulary----------------
stoi = data_LM.stoi  # dictionary for character-id mapping
itos = data_LM.itos  # list of Thai characters

train_path, dev_path, test_path = get_corpus_tokenizer.get_path_data(args.dataset)

class Model:
    """
    define the model and download weights if available.
    define forward and backward function
    """

    def __init__(self, bi_lstm):

        if cuda:
            if bi_lstm:
                self.rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num,
                                         bidirectional=True).cuda()
                self.output_classifier = torch.nn.Linear(args.hidden_dim * 2, 2).cuda()
            else:
                self.rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num).cuda()
                self.output_classifier = torch.nn.Linear(args.hidden_dim, 2).cuda()
            self.char_embeddings = torch.nn.Embedding(num_embeddings=len(itos),
                                                      embedding_dim=args.char_embedding_size).cuda()
        else:
            if bi_lstm:
                self.rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num, bidirectional=True)
                self.output_classifier = torch.nn.Linear(args.hidden_dim * 2, 2)
            else:
                self.rnn = torch.nn.LSTM(args.char_embedding_size, args.hidden_dim, args.layer_num)
                self.output_classifier = torch.nn.Linear(args.hidden_dim, 2)

            self.char_embeddings = torch.nn.Embedding(num_embeddings=len(itos), embedding_dim=args.char_embedding_size)

        self.modules = [self.rnn, self.output_classifier, self.char_embeddings]
        self.parameters_cached = [x for x in self.parameters(self.modules)]

        learning_rate = args.learning_rate
        if args.optim == "adam":
            self.optim = torch.optim.Adam(self.parameters(self.modules), lr=learning_rate)
        else:
            self.optim = torch.optim.SGD(self.parameters(self.modules), lr=learning_rate,
                                         momentum=args.sgd_momentum)  # 0.02, 0.9
        # if the language model is imported, do not download the output layer and optim
        if str(args.load_from)[:2] == "LM":
            self.named_modules = {"rnn": self.rnn, "char_embeddings": self.char_embeddings}
            logging.info("get the embedding and rnn layer from the pretrained language model")
        # if a tokenizer is imported, also download optim
        else:
            self.named_modules = {"rnn": self.rnn, "char_embeddings": self.char_embeddings, "optim": self.optim,
                                  "output_classifier": self.output_classifier}
            logging.info("get the parameters of the imported tokenizer")

        # load the model
        if args.load_from is not None:
            checkpoint = torch.load(CHECKPOINTS + args.load_from + ".pth.tar")
            for name, module in self.named_modules.items():
                module.load_state_dict(checkpoint[name])
                logging.info("load ", name)
            logging.info("load parameters and weights....")

        # after loading parameters from the language model, set the dictionary to save all parameters
        self.named_modules = {"rnn": self.rnn, "char_embeddings": self.char_embeddings, "optim": self.optim,
                              "output_classifier": self.output_classifier}

    def parameters(self, modules):
        for module in modules:
            for param in module.parameters(modules):
                yield param

    # define the forward function
    def forward_cl(self, numeric_pair, train=True):
        global hidden
        global beginning
        if hidden is None or (train and random.random() > 0.9):
            hidden = None
        elif hidden is not None:
            hidden = tuple([Variable(x.data).detach() for x in hidden])

        input_tensor, target_tensor = numeric_pair[0], numeric_pair[1]
        input_tensor = Variable(input_tensor, requires_grad=False)
        target_tensor = Variable(target_tensor, requires_grad=False)

        embedded = self.char_embeddings(input_tensor)
        if train:
            embedded = char_dropout(embedded)

        out, hidden = self.rnn(embedded, hidden)  # ------------train the tokenizer

        logits = self.output_classifier(out)
        log_probs = logsoftmax(logits)
        loss = classifier_loss(logits.view(-1, 2), target_tensor.view(-1))
        return loss, target_tensor.view(-1).size()[0], log_probs, input_tensor, target_tensor

    def backward(self, loss):
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters_cached, clip_value=args.clip_grad)
        self.optim.step()

# create batches
def create_tensor_classifier(generator_chunks):
    target_list = []
    input_list = []
    for chunks in generator_chunks:
        for word in chunks:
            target_list.append(1)
            for char in word[:-1]:
                target_list.append(0)
        word_input = "".join(chunks)

        for char in word_input:
            input_list.append(stoi[char] if char in stoi else 2)
        seq_len = args.sequence_length

        cutoff = int(len(input_list) / (args.batchSize * seq_len)) * (
                args.batchSize * seq_len)

        input_tensor = input_list[:cutoff]
        input_list = input_list[cutoff:]
        target_tensor = target_list[:cutoff]
        target_list = target_list[cutoff:]
        if cuda:
            input_tensor = torch.LongTensor(input_tensor).view(args.batchSize, -1, args.sequence_length).transpose(0,
                                                                                                                   1).transpose(
                1,
                2).cuda()
            target_tensor = torch.LongTensor(target_tensor).view(args.batchSize, -1, args.sequence_length).transpose(0,
                                                                                                                     1).transpose(
                1, 2).cuda()

        else:
            input_tensor = torch.LongTensor(input_tensor).view(args.batchSize, -1, args.sequence_length).transpose(0,
                                                                                                                   1).transpose(
                1,
                2)
            target_tensor = torch.LongTensor(target_tensor).view(args.batchSize, -1, args.sequence_length).transpose(0,
                                                                                                                     1).transpose(
                1, 2)

        numberOfSequences = input_tensor.size()[0]

        for i in range(numberOfSequences):
            yield (input_tensor[i], target_tensor[i])

# save log file
def save_log(mode="w"):
    with open(CHECKPOINTS_TOKENIZER + args.save_to, mode) as outFile:
        if mode == "a":
            print("----------resume training/early stopping ---------", file=outFile)
        else:
            print("-----------Tokenizer---------", file=outFile)

        # commands for later training
        long, short = util.get_command(str(args))
        p = "python Tokenizer.py "
        long = p + long
        short = p + short
        print(">>> command with full parameters", file=outFile)
        print(long, file=outFile)
        print("\n>>> command with short parameters", file=outFile)
        print(short, file=outFile)
        print("", file=outFile)

        # model parameters and losses
        print("file name = ", args.save_to, file=outFile)
        print("", file=outFile)
        print(">>> ", args.add_note, file=outFile)
        print("bi_lstm ", bi_lstm, file=outFile)
        print("train classifier for " + str(args.epoch) + " epoch", file=outFile)
        print("trainLosses ", trainLosses, file=outFile)
        print("devLosses ", devLosses, file=outFile)
        print("", file=outFile)
        print("count train sample ", count_train_samples, file=outFile)
        print("count dev sample ", count_dev_samples, file=outFile)

        # model parameters
        print("", file=outFile)
        l = str(args)[10:].strip(")").split(",")

        for i in l:
            print(i, file=outFile)
        print("", file=outFile)

        # data set
        print("train set: ", train_path, file=outFile)
        print("dev set: ", dev_path, file=outFile)
        print("test set: ", test_path, file=outFile)
        print("", file=outFile)

        # number of samples
        print("count train sample ", count_train_samples, file=outFile)
        print("count dev sample ", count_dev_samples, file=outFile)

        # parameter flags (used when later import this model to another model)
        print("config for later download : ", file=outFile)
        p = util.get_param(str(args))
        print(p, file=outFile)
        print("", file=outFile)
        print("save log file to ", args.save_to)


import preprocess

# load and preprocess original BEST corpus
def load_data_classifier(path, num_lines_per_chunk, doShuffling=True, not_include_dot=False):
    chunks = []
    with open(path, "r") as inFile:
        for line in inFile:
            line = line.strip()
            line = preprocess.preprocess(line)
            words = line.split("|")
            logging.info(words)
            chunks.append(words)
            if len(chunks) >= num_lines_per_chunk:
                if doShuffling:
                    random.shuffle(chunks)
                chunks = list(chain(*chunks))
                yield chunks
                chunks = []
    chunks = list(chain(*chunks))
    yield chunks

# train the tokenizer
num_epoch = 1
if start_training:
    model = Model(bi_lstm)
    train_path, dev_path, test_path = get_corpus_tokenizer.get_path_data(args.dataset)
    logsoftmax = torch.nn.LogSoftmax(dim=2)
    classifier_loss = torch.nn.CrossEntropyLoss()
    char_dropout = torch.nn.Dropout2d(p=args.char_dropout_prob)

    count_train_samples = 0
    count_dev_samples = 0
    count_test_samples = 0

    trainLosses = []
    devLosses = []
    for epoch in range(args.epoch):
        logging.info("epoch: ", epoch + 1)
        training_data_CL = load_data_classifier(train_path, num_lines_per_chunk=args.len_lines_per_chunk,
                                                doShuffling=True, not_include_dot=no_dot)
        logging.info("Got the training data")
        logging.info("train: ", train_path)
        training_chars = create_tensor_classifier(training_data_CL)

        model.rnn.train(True)
        startTime = time.time()
        trainChars = 0
        train_loss_ = 0
        counter = 0
        hidden, beginning = None, None
        while True:
            counter += 1
            try:
                numeric = next(training_chars)
            except StopIteration:
                logging.info("end of the batch")
                break
            loss, charCounts, _, _, _ = model.forward_cl(numeric, train=True)  # training
            if epoch == 0:
                count_train_samples += args.batchSize

            model.backward(loss)
            train_loss_ += charCounts * loss.cpu().data.numpy()
            trainChars += charCounts

        trainLosses.append(train_loss_ / trainChars)
        logging.info("trainLosses ", trainLosses)

        if save_weights:
            logging.info("save model parameters... ", args.save_to)
            torch.save(dict([(name, module.state_dict()) for name, module in model.named_modules.items()]),
                       CHECKPOINTS_TOKENIZER + args.save_to + ".pth.tar")

        model.rnn.train(False)
        dev_data_CL = load_data_classifier(dev_path, num_lines_per_chunk=args.len_lines_per_chunk, doShuffling=False,
                                          not_include_dot=no_dot)

        logging.info("Got dev data")
        logging.info("dev path: ", dev_path)
        dev_chars = create_tensor_classifier(dev_data_CL)
        dev_loss = 0
        dev_char_count = 0
        counter = 0
        hidden, beginning = None, None
        while True:
            counter += 1
            try:
                numeric = next(dev_chars)
            except StopIteration:
                break
            loss, numberOfCharacters, _, _, _ = model.forward_cl(numeric, train=False)
            dev_loss += numberOfCharacters * loss.cpu().data.numpy()
            dev_char_count += numberOfCharacters
            if epoch == 0:
                count_dev_samples += args.batchSize

        devLosses.append(dev_loss / dev_char_count)
        logging.info("dev losses ", devLosses)
        num_epoch = epoch

        if args.load_from is None or str(args.load_from)[:2] == "LM":
            save_log("w")

        # if resume the training, append log data only for the last iteration
        elif str(args.load_from)[:2] == "To" and epoch >= args.epoch - 1:
            save_log("a")
        if len(devLosses) > 1 and devLosses[-1] >= devLosses[-2]:
            logging.info("early stopping")
            save_log("a")
            break

        if args.optim == "adam" and adam_with_lr_decay:
            learning_rate = args.learning_rate * math.pow(args.adam_lr_decay, len(devLosses))
            optim = torch.optim.Adam(model.parameters(model.modules), lr=learning_rate)
        elif args.optim == "sgd":
            learning_rate = args.learning_rate * math.pow(args.lr_decay, len(devLosses))
            optim = torch.optim.SGD(model.parameters(model.modules), lr=learning_rate,

                                    momentum=args.sgd_momentum)  # 0.02, 0.9


def test():

    global model
    count_test_samples = 0

    correct = 0
    falsePositives = 0
    falseNegatives = 0
    count_real_word = 0
    count_predict_word = 0
    word_label = []
    word_pred = []
    word_correct = 0

    model.rnn.train(False)
    logging.info("testing......")

    test_data_CL = load_data_classifier(test_path, num_lines_per_chunk=args.len_lines_per_chunk, doShuffling=False,
                                        not_include_dot=no_dot)

    logging.info("Got test data")
    logging.info("test path: ", test_path)

    test_chars = create_tensor_classifier(test_data_CL)

    counter = 0
    while True:
        counter += 1
        try:
            numeric = next(test_chars)
        except StopIteration:
            break

        loss, numberOfCharacters, log_probs_cal, input_tensor_cal, target_tensor_cal = model._forward(numeric,
                                                                                                      train=False)
        count_test_samples += args.batchSize
        tag_score = log_probs_cal

        flat_input = input_tensor_cal.transpose(1, 0)
        flat_label = target_tensor_cal.transpose(1, 0)
        tag_score = tag_score.transpose(0, 1)
        val, argmax = tag_score.max(dim=2)

        for b in range(args.batchSize):
            chars = [itos[element.item()] if element.item() != 0 else "-" for element in flat_input[b]]
            pred_seq = chars.copy()
            label = flat_label[b]
            pred = argmax[b]

            for i in range(args.sequence_length):

                if label[i] == 1:
                    chars[i] = " " + chars[i]

                if pred[i] == 1:
                    pred_seq[i] = " " + pred_seq[i]

                    if pred[i] == label[i]:
                        correct += 1

                if pred[i] != label[i] and pred[i] == 0:
                    falseNegatives += 1

                if pred[i] != label[i] and pred[i] == 1:
                    falsePositives += 1
                # word level evaluation
                word_label.append(label[i])
                word_pred.append(pred[i])
                if pred[i] == 1:
                    count_predict_word += 1
                if label[i] == 1:
                    if word_label == word_pred:
                        word_correct += 1
                    word_pred = []
                    word_label = []
                    count_real_word += 1

            if printAllPrediction and "".join(chars) != "".join(pred_seq):
                logging.info("".join(chars))
                logging.info("".join(pred_seq))

        if printPrediction and "".join(chars) != "".join(pred_seq):
            logging.info("".join(chars))
            logging.info("".join(pred_seq))

    logging.info("train losses ", trainLosses)
    logging.info("dev losses ", devLosses)

    if (correct + falsePositives) == 0:

        logging.info("correct ", correct)
        logging.info(total_time)

    else:
        logging.info(total_time)

        precision = correct / (correct + falsePositives)
        recall = correct / (correct + falseNegatives)

        f1 = 2 * (precision * recall) / (precision + recall)
        logging.info("falseNegatives ", falseNegatives)
        logging.info("falsePositives ", falsePositives)
        logging.info("correct ", correct)

        logging.info("Boundary measures", "Precision", round(precision * 100, 2), "Recall", round(recall * 100, 2), "F1",
              round(f1 * 100, 2))

        word_precision = word_correct / (count_predict_word)
        word_recall = word_correct / (count_real_word)
        word_f1 = 2 * (word_precision * word_recall) / (word_precision + word_recall)
        logging.info("word_correct :", word_correct)
        logging.info("count_predict_word: ", count_predict_word)
        logging.info("count_real_word: ", count_real_word)
        logging.info("Word measures", "Precision", round(word_precision * 100, 2), "Recall", round(word_recall * 100, 2), "F1",
              round(word_f1 * 100, 2))

        logging.info("".join(chars))
        logging.info("".join(pred_seq))

        with open(CHECKPOINTS_TOKENIZER + args.save_to, "a") as outFile:

            logging.info("falseNegatives ", falseNegatives, file=outFile)
            logging.info("falsePositives ", falsePositives, file=outFile)
            logging.info("correct ", correct, file=outFile)

            logging.info("Boundary measures: ", file=outFile)
            logging.info("Precision", precision, "Recall", recall, "F1",
                  2 * (precision * recall) / (precision + recall), file=outFile)
            logging.info("\nWord measures", "Precision", word_precision, "Recall", word_recall, "F1",
                  2 * (word_precision * word_recall) / (word_precision + word_recall), file=outFile)
            logging.info("word_correct :", word_correct, file=outFile)
            logging.info("count_predict_word: ", count_predict_word, file=outFile)
            logging.info("count_real_word: ", count_real_word, file=outFile)

            logging.info("", file=outFile)
            logging.info("".join(chars), file=outFile)
            logging.info("".join(pred_seq), file=outFile)
            logging.info(total_time, file=outFile)

    with open(CHECKPOINTS_TOKENIZER + "tokenizer_result.csv", "a") as table:
        logging.info("---------save results------")
        logging.info("-", file=table, end=';')
        if str(args.load_from)[:2] == "LM":
            logging.info("LM", file=table, end=';')
        else:
            logging.info("-", file=table, end=';')

        logging.info(args.save_to, file=table, end=';')
        logging.info(args.dataset, file=table, end=';')

        logging.info("boundary", file=table, end=';')
        precision = round(precision * 100, 2)
        recall = round(recall * 100, 2)
        f1 = round(f1 * 100, 2)

        logging.info(precision, file=table, end=';')
        logging.info(recall, file=table, end=';')
        logging.info(f1, file=table, end=';')

        logging.info("word", file=table, end=';')
        word_precision = round(word_precision * 100, 2)
        word_recall = round(word_recall * 100, 2)
        word_f1 = round(word_f1 * 100, 2)

        logging.info(word_precision, file=table, end=';')
        logging.info(word_recall, file=table, end=';')
        logging.info(word_f1, file=table, end=';')

        logging.info(num_epoch+1, file=table, end=';')
        logging.info(total_time, file=table, end=';')

        logging.info("trainLosses ", trainLosses, file=table, end=';')
        logging.info("devLosses ", devLosses, file=table, end=';')
        logging.info(args.add_note, file=table, end=';')
        p = util.get_param(str(args))
        logging.info(p, file=table, end=';')
        long, short = util.get_command(str(args))
        p = "python Tokenizer.py "
        long = p + long
        logging.info(long, file=table, end='\n')


end = timer()
total_time = timedelta(seconds=end - start)
logging.info(timedelta(seconds=end - start))
test()
logging.info(args.save_to)
logging.info(args.add_note)
