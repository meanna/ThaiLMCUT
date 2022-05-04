# -*- coding: utf-8 -*-
# credit: the code below is modified from https://github.com/m-hahn/tabula-rasa-rnns
import torch
from torch.autograd import Variable

hidden = None


class Model:
    """
    define the model and download weights
    """

    def __init__(self, char_embedding_size, hidden_dim, layer_num, lstm_num_direction, cuda, len_id_to_char):

        bi_lstm = lstm_num_direction == 2

        self.logsoftmax = torch.nn.LogSoftmax(dim=2)

        self.classifier_loss = torch.nn.CrossEntropyLoss()

        if cuda:
            if bi_lstm:
                self.rnn = torch.nn.LSTM(char_embedding_size, hidden_dim, layer_num, bidirectional=True).cuda()
                self.output_classifier = torch.nn.Linear(hidden_dim * 2, 2).cuda()
            else:
                self.rnn = torch.nn.LSTM(char_embedding_size, hidden_dim, layer_num).cuda()
                self.output_classifier = torch.nn.Linear(hidden_dim, 2).cuda()
            self.char_embeddings = torch.nn.Embedding(num_embeddings=len_id_to_char,
                                                      embedding_dim=char_embedding_size).cuda()
        else:
            if bi_lstm:
                self.rnn = torch.nn.LSTM(char_embedding_size, hidden_dim, layer_num, bidirectional=True)
                self.output_classifier = torch.nn.Linear(hidden_dim * 2, 2)
            else:
                self.rnn = torch.nn.LSTM(char_embedding_size, hidden_dim, layer_num)
                self.output_classifier = torch.nn.Linear(hidden_dim, 2)

            self.char_embeddings = torch.nn.Embedding(num_embeddings=len_id_to_char, embedding_dim=char_embedding_size)

    def _forward(self, numeric):
        global hidden
        if hidden is not None:
            hidden = tuple([Variable(x.data).detach() for x in hidden])
        input_tensor = numeric
        input_tensor = Variable(input_tensor, requires_grad=False)
        embedded = self.char_embeddings(input_tensor)
        out, hidden = self.rnn(embedded, hidden)  # train the tokenizer
        logits = self.output_classifier(out)
        log_probs = self.logsoftmax(logits)
        return log_probs, input_tensor
