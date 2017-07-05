# coding: utf8
"""Conll training algorithm"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from torch.autograd import Variable
from torch import from_numpy

class NeuralCoref(nn.Module):
    def __init__(self, vocab_size, embedding_dim, H1, H2, H3, D_pair_in, D_single_in):
        super(NeuralCoref, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.pair_top_layer1 = nn.Linear(D_pair_in, H1)
        self.pair_top_layer2 = nn.Linear(H1, H2)
        self.pair_top_layer3 = nn.Linear(H2, H3)
        self.pair_out_score = nn.Linear(H3, 1)
        self.single_top_layer1 = nn.Linear(D_single_in, H1)
        self.single_top_layer2 = nn.Linear(H1, H2)
        self.single_top_layer3 = nn.Linear(H2, H3)
        self.single_out_score = nn.Linear(H3, 1)

    @staticmethod
    def BCELogit(input, target):
        r"""Function that measures Binary Cross Entropy between target and output logits:
        See :class:`~torch.nn.BCEWithLogitsLoss` for details.
        """
        if not target.is_same_size(input):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.sum()

    def all_pair_loss(self, pair_in, pair_labels, single_in, single_label):
        """
        All pairs and single mentions probabilistic loss
        """
        pair_scores = self.forward(pair_in)
        pair_loss = self.BCELogit(pair_scores, pair_labels)

        single_score = self.single_forward(single_in)
        single_loss = self.BCELogit(single_score, single_label)

        return pair_loss + single_loss

    def top_pair_loss(self, pair_in, true_ant, false_ant, single_in):
        """
        Top pairs (best true and best mistaken) and single mention probabilistic loss
        """
        pair_scores = self.forward(pair_in)
        single_score = self.single_forward(single_in)
        all_score = torch.cat([pair_scores, single_score], 0)
        print("all_score", all_score.size())
        print("true_ant", true_ant.size())
        true_pairs = torch.gather(all_score, 0, true_ant)
        false_pairs = torch.gather(all_score, 0, false_ant)

        m = nn.LogSigmoid()
        top_true = m(true_pairs).max()
        top_false = m(false_pairs.neg()).min()
        
        return top_true + top_false
    
    def ranking_loss(self, pair_in, single_in, true_ant, costs):
        """
        Slack-rescaled max margin loss
        """
        pair_scores = self.forward(pair_in)
        single_score = self.single_forward(single_in)
        all_score = torch.cat([pair_scores, single_score], 0)

        true_ant_score = torch.gather(all_score, 0, true_ant)
        top_true = true_ant_score.max()
        print("costs", costs.size())
        print("all_score", all_score.size())
        print("top_true", top_true.size())
        loss = costs * (1 + all_score - top_true.expand_as(all_score))
        return loss.max()

    def forward(self, pair_in):
        """
        Forward pass of the model to score a pair of mentions or a list of pairs
        """
        ant_spans, ant_words, ana_spans, ana_words, pair_features = pair_in
        print("ant_words", ant_words.size())
        ant_embed_words = self.word_embeds(ant_words).view(ant_words.size()[0], -1)
        print("ant_embed_words", ant_embed_words.size())
        ana_embed_words = self.word_embeds(ana_words).view(ana_words.size()[0], -1)
        pair_input = torch.cat([ant_spans, ant_embed_words, ana_spans, ana_embed_words, pair_features], 1)
        
        l1_out = self.pair_top_layer1(pair_input).clamp(min=0)
        l2_out = self.pair_top_layer2(l1_out).clamp(min=0)
        l3_out = self.pair_top_layer3(l2_out).clamp(min=0)
        return self.pair_out_score(l3_out)

    def single_forward(self, single_in):
        """
        Forward pass of the model to score a single mention
        """
        spans, words, single_features = single_in
        embed_words = self.word_embeds(words).view(1, -1)
        print("spans", spans.size())
        print("embed_words", embed_words.size())
        print("single_features", embed_words.size())
        single_input = torch.cat([spans, embed_words, single_features], 1)
        
        l1_out = self.single_top_layer1(single_input).clamp(min=0)
        l2_out = self.single_top_layer2(l1_out).clamp(min=0)
        l3_out = self.single_top_layer3(l2_out).clamp(min=0)
        return self.single_out_score(l3_out)

class CorefDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, FN, FL, WL):
        self.FN = FN
        self.FL = FL
        self.WL = WL
        self.datas = {}
        if not os.listdir(data_path):
            raise ValueError("Empty data_path")
        for file in os.listdir(data_path):
            print("Reading", file)
            self.datas[file.split(sep='.')[0]] = np.load(data_path + file)
        
    def get_input_dim(self, embedding_dim):
        # return D_pair_in, D_single_in
        vocab_size = np.asscalar(self.datas["vocab_size"])
        D_pair_in = self.datas["pair_span"].shape[1] + self.datas["pair_words"].shape[1] * embedding_dim \
                    + self.datas["mention_span"].shape[1] + self.datas["mention_words"].shape[1] * embedding_dim \
                    + self.datas["pair_features"].shape[1]
        D_single_in = self.datas["mention_span"].shape[1] + self.datas["mention_words"].shape[1] * embedding_dim \
                      + self.datas["mention_features"].shape[1]
        return (vocab_size, D_pair_in, D_single_in)

    def __len__(self):
        return len(self.datas["single_labels"])

    def __getitem__(self, idx):
        start, end = self.datas["mention_pair_indices"][idx, :]
        ant_spans = Variable(from_numpy(self.datas["pair_span"][start:end, :]).float())
        ant_words = Variable(from_numpy(self.datas["pair_words"][start:end, :]))
        pair_features = Variable(from_numpy(self.datas["pair_features"][start:end, :]).float())
        spans = Variable(from_numpy(self.datas["mention_span"][idx, :]).unsqueeze(0).float())
        words = Variable(from_numpy(self.datas["mention_words"][idx, :]).unsqueeze(0))
        single_features = Variable(from_numpy(self.datas["mention_features"][idx, :]).unsqueeze(0).float())
        ana_spans = Variable(from_numpy(self.datas["mention_span"][idx, :]).expand_as(ant_spans).float())
        ana_words = Variable(from_numpy(self.datas["mention_words"][idx, :]).expand_as(ant_words))
        pair_labels = Variable(from_numpy(self.datas["pair_labels"][start:end, :]).float(), requires_grad=False)
        single_labels = Variable(from_numpy(self.datas["single_labels"][idx, :]).unsqueeze(0).float(), requires_grad=False)
        labels_stack = np.vstack([self.datas["pair_labels"][start:end, :], self.datas["single_labels"][idx, :]])
        true_ant = Variable(from_numpy(labels_stack))
        false_ant = Variable(from_numpy(1 - labels_stack))
        if self.datas["single_labels"][idx, :] == 1:
            costs = Variable(from_numpy(np.vstack([self.WL * (1 - self.datas["pair_labels"][start:end, :]), self.FN])).float(), requires_grad=False)  # Inverse labels: 1=>0, 0=>1
        else:
            costs = Variable(from_numpy(np.vstack([self.FL * np.ones_like(self.datas["pair_labels"][start:end, :]), 0])).float(), requires_grad=False)

        pair_in = (ant_spans, ant_words, ana_spans, ana_words, pair_features)
        single_in = (spans, words, single_features)
        for t in pair_in + single_in + (pair_labels, single_labels, true_ant, false_ant, costs):
            print(t.size())

        return (pair_in, pair_labels, true_ant, false_ant, single_in, single_labels, costs)

def run_model(DATA_PATH):
    ####################
    # Hyper-parameters #
    H1, H2, H3 = 1000, 500, 500
    embedding_dim = 50
    # error penalties for heuristic ranking objective
    # WL: Wrong link, FN: False new (indicated as new but is not), FL: False link (new but indicated as not new)
    FN, FL, WL = 0.8, 0.4, 1.0
    # learning rates
    all_pairs_lr, top_pairs_lr, ranking_lr = 0.002, 0.0002, 0.000002

    dataset = CorefDataset(DATA_PATH, FN, FL, WL)
    vocab_size, D_pair_in, D_single_in = dataset.get_input_dim(embedding_dim)
    print(vocab_size, D_pair_in, D_single_in)

    # Construct our model
    model = NeuralCoref(vocab_size, embedding_dim, H1, H2, H3, D_pair_in, D_single_in)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    for i in range(len(dataset)):
        pair_in, pair_labels, true_ant, false_ant, single_in, single_labels, costs = dataset[i]
        # Compute and print loss
        all_pair_loss = model.all_pair_loss(pair_in, pair_labels, single_in, single_labels)
        print("all_pair_loss", i, all_pair_loss.data[0])

        top_pair_loss = model.top_pair_loss(pair_in, true_ant, false_ant, single_in)
        print("top_pair_loss", i, top_pair_loss.data[0])
        
        ranking_loss = model.ranking_loss(pair_in, single_in, true_ant, costs)
        print("ranking_loss", i, ranking_loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        all_pair_loss.backward()
        optimizer.step()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
        run_model(DATA_PATH)
