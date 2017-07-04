# coding: utf8
"""Conll training algorithm"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import re
import sys
import os
import spacy

import torch
import torch.autograd as autograd
import torch.nn as nn

from neuralcoref.data import Mention, Data, Speaker

class NeuralCoref(nn.Module):
    def __init__(self, vocab_size, embedding_dim, H1, H2, H3, D_pair_in, D_single_in):
        super(NeuralCoref, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
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

    def all_pair_loss(self, pair_in, pair_labels, single_in, single_label):
        """
        All pair and signle mention probabilistic loss
        """
        pair_scores = self.pair_forward(pair_in)
        pair_loss = nn.BCEWithLogitsLoss(size_average=False)(pair_scores, pair_labels)

        single_score = self.single_forward(single_in)
        single_loss = nn.BCEWithLogitsLoss(size_average=False)(single_score, single_label)

        return torch.add(pair_loss, single_loss)

    def top_pair_loss(self, pair_in, true_ant, false_ant, single_in, single_label):
        """
        Top pairs (best true and best mistaken) and single mention probabilistic loss
        """
        pair_scores = self.pair_forward(pair_in)
        
        true_pairs = torch.gather(pair_scores, 0, torch.LongTensor(true_ant))
        top_true = torch.max(nn.LogSigmoid(true_pairs))
        
        false_pairs = torch.gather(pair_scores, 0, torch.LongTensor(false_ant))
        top_false = torch.min(nn.LogSigmoid(torch.neg(false_pairs)))
        
        pair_loss = torch.add(top_true, top_false)
        
        single_score = self.single_forward(single_in)
        single_loss = nn.BCEWithLogitsLoss(size_average=False)(single_score, single_label)

        return torch.add(pair_loss, single_loss)

    def ranking_loss(self, pair_in, single_in, true_ant, costs):
        """
        Slack-rescaled max margin loss
        """
        pair_scores = self.pair_forward(seq_in)
        single_score = self.single_forward(single_in)
        all_score = torch.cat([pair_scores, single_score], 0)
        
        true_ant_score = torch.gather(all_score, 0, torch.LongTensor(true_ant))
        top_true = torch.max(true_ant_score)
        
        loss = torch.mul(costs, torch.add(torch.add(all_score, -1, top_true), 1))
        return torch.max(loss)

    def pair_forward(self, pair_in):
        """
        Forward pass of the model to score a pair of mentions
        """
        ant_spans, ant_words, ana_spans, ana_words, pair_features = pair_in
        ant_embed_words = self.word_embeds(ant_words).view(ant_words.size()[0], ant_words.size()[1] * self.embedding_dim)
        ana_embed_words = self.word_embeds(ana_words).view(ana_words.size()[0], ana_words.size()[1] * self.embedding_dim)
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
        embed_words = self.word_embeds(words).view(words.size()[0], words.size()[1] * self.embedding_dim)
        single_input = torch.cat([spans, embed_words, single_features], 1)
        
        l1_out = self.single_top_layer1(single_input).clamp(min=0)
        l2_out = self.single_top_layer2(l1_out).clamp(min=0)
        l3_out = self.single_top_layer3(l2_out).clamp(min=0)
        return self.single_out_score(l3_out)

def prepare_pair_embedding(ant_spans, ant_words, ana_spans, ana_words, pair_features, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def prepare_pair_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
H1, H2, H3 = 1000, 500, 500

# Create random Tensors to hold inputs and outputs, and wrap them in Variables
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Construct our model by instantiating the class defined above
model = NeuralCoref(vocab_size, embedding_dim, H1, H2, H3)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    all_pair_loss = criterion(y_pred, y)
    print(t, all_pair_loss.data[0])

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    all_pair_loss.backward()
    optimizer.step()
  
class CorefDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.mentions = []
    
    def __len__(self):
        return len(self.mentions)
    
    def __getitem__(self, idx):
        return self.mention[idx]

