"""Conll training algorithm"""

import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data


class Model(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, H1, H2, H3, D_pair_in, D_single_in, dropout=0.5
    ):
        super(Model, self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.pair_top = nn.Sequential(
            nn.Linear(D_pair_in, H1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H2, H3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H3, 1),
            nn.Linear(1, 1),
        )
        self.single_top = nn.Sequential(
            nn.Linear(D_single_in, H1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H2, H3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H3, 1),
            nn.Linear(1, 1),
        )
        self.init_weights()

    def init_weights(self):
        w = (param.data for name, param in self.named_parameters() if "weight" in name)
        b = (param.data for name, param in self.named_parameters() if "bias" in name)
        nn.init.uniform_(self.word_embeds.weight.data, a=-0.5, b=0.5)
        for t in w:
            nn.init.xavier_uniform_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def load_embeddings(self, preloaded_weights):
        self.word_embeds.weight = nn.Parameter(preloaded_weights)

    def load_weights(self, weights_path):
        print("Loading weights")
        single_layers_weights, single_layers_biases = [], []
        for f in sorted(os.listdir(weights_path)):
            if f.startswith("single_mention_weights"):
                single_layers_weights.append(np.load(os.path.join(weights_path, f)))
            if f.startswith("single_mention_bias"):
                single_layers_biases.append(np.load(os.path.join(weights_path, f)))
        top_single_linear = (
            layer for layer in self.single_top if isinstance(layer, nn.Linear)
        )
        for w, b, layer in zip(
            single_layers_weights, single_layers_biases, top_single_linear
        ):
            layer.weight = nn.Parameter(torch.from_numpy(w).float())
            layer.bias = nn.Parameter(torch.from_numpy(b).float().squeeze())
        pair_layers_weights, pair_layers_biases = [], []
        for f in sorted(os.listdir(weights_path)):
            if f.startswith("pair_mentions_weights"):
                pair_layers_weights.append(np.load(os.path.join(weights_path, f)))
            if f.startswith("pair_mentions_bias"):
                pair_layers_biases.append(np.load(os.path.join(weights_path, f)))
        top_pair_linear = (
            layer for layer in self.pair_top if isinstance(layer, nn.Linear)
        )
        for w, b, layer in zip(
            pair_layers_weights, pair_layers_biases, top_pair_linear
        ):
            layer.weight = nn.Parameter(torch.from_numpy(w).float())
            layer.bias = nn.Parameter(torch.from_numpy(b).float().squeeze())

    def forward(self, inputs, concat_axis=1):
        pairs = len(inputs) == 8
        if pairs:
            spans, words, single_features, ant_spans, ant_words, ana_spans, ana_words, pair_features = (
                inputs
            )
        else:
            spans, words, single_features = inputs
        words = words.type(torch.LongTensor)
        if torch.cuda.is_available():
            words = words.cuda()
        embed_words = self.drop(self.word_embeds(words).view(words.size()[0], -1))
        single_input = torch.cat([spans, embed_words, single_features], 1)
        single_scores = self.single_top(single_input)
        if pairs:
            batchsize, pairs_num, _ = ana_spans.size()
            ant_words_long = ant_words.view(batchsize, -1).type(torch.LongTensor)
            ana_words_long = ana_words.view(batchsize, -1).type(torch.LongTensor)
            if torch.cuda.is_available():
                ant_words_long = ant_words_long.cuda()
                ana_words_long = ana_words_long.cuda()
            ant_embed_words = self.drop(
                self.word_embeds(ant_words_long).view(batchsize, pairs_num, -1)
            )
            ana_embed_words = self.drop(
                self.word_embeds(ana_words_long).view(batchsize, pairs_num, -1)
            )
            pair_input = torch.cat(
                [ant_spans, ant_embed_words, ana_spans, ana_embed_words, pair_features],
                2,
            )
            pair_scores = self.pair_top(pair_input).squeeze(dim=2)
            total_scores = torch.cat([pair_scores, single_scores], concat_axis)
        return total_scores if pairs else single_scores
