# models.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random
from sentiment_data import *


def form_input(x) -> torch.Tensor:
    return torch.from_numpy(x).float()

def get_average_vector(words, word_embeddings):
    return form_input(np.sum(np.array(
            [word_embeddings.get_embedding(w) for w in words]), axis=0)
            / len(words))

class DAN(nn.Module):

    def __init__(self, inp, hid, out, exs, word_embeddings):
        super(DAN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.ReLU6()
        self.W = nn.Linear(hid, out)
        self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        nn.init.kaiming_uniform_(self.V.weight)
        nn.init.kaiming_uniform_(self.W.weight)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.word_embeddings = word_embeddings
        self.ex_averages = self.__computeAverages(exs)

    def forward(self, x):
        return self.log_softmax(self.W(self.dropout2(self.g(self.V(self.dropout1(x))))))

    def __computeAverages(self, exs: List[SentimentExample]):
        avgs = []
        for ex in exs:
            avgs.append(get_average_vector(ex.words, self.word_embeddings))
        return avgs

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex: SentimentExample):
        """
        Makes a prediction on the given SentimentExample
        :param ex: example to predict
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex: SentimentExample):
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

BATCH_SIZE = 20
HIDDEN_SIZE = 20
LEARNING_RATE = 0.001
EPOCHS = 40
NUM_CLASSES = 2

class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """
    def __init__(self, dan):
        self.dan = dan

    def predict(self, ex: SentimentExample):
        self.dan.eval()
        avg_vector = get_average_vector(ex.words, self.dan.word_embeddings)
        probs = self.dan.forward(avg_vector)
        # print(torch.argmax(probs).item(), "\n", ex.words)
        return torch.argmax(probs).item()

NEG_LABEL = torch.zeros(NUM_CLASSES)
POS_LABEL = torch.zeros(NUM_CLASSES)
NEG_LABEL.scatter_(0, torch.from_numpy(np.asarray(0, dtype=np.int64)), 1)
POS_LABEL.scatter_(0, torch.from_numpy(np.asarray(1, dtype=np.int64)), 1)

GOLD_LABELS = {
    0: NEG_LABEL,
    1: POS_LABEL
}

def train_deep_averaging_network(args, train_exs, dev_exs, word_embeddings):
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return:
    """
    input_size = int(word_embeddings.get_embedding_length())
    dan = DAN(input_size, HIDDEN_SIZE, NUM_CLASSES, train_exs, word_embeddings)
    dan.train()
    optimizer = optim.Adam(dan.parameters(), lr=LEARNING_RATE)
    ex_idxs = [i for i in range(len(train_exs))]
    for epoch in range(EPOCHS):
        random.shuffle(ex_idxs)
        idxs = list(ex_idxs)
        while idxs:
            curr_idxs = []
            i = 0
            while idxs and i < BATCH_SIZE:
                curr_idxs.append(idxs.pop(0))
                i += 1
            exs = [train_exs[i] for i in curr_idxs]
            labels = torch.tensor([GOLD_LABELS[ex.label].tolist() for ex in exs])
            avg_vectors = torch.tensor([dan.ex_averages[i].tolist() for i in curr_idxs])
            if len(labels) == 1:
                labels = labels[0]
                avg_vectors = avg_vectors[0]
            dan.zero_grad()
            probs = dan.forward(avg_vectors)
            loss = torch.neg(probs).mul(labels)
            loss.sum().backward()
            optimizer.step()
    return NeuralSentimentClassifier(dan)
