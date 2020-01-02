"""Conll training algorithm"""

import os
import io
import numpy as np

import torch
import torch.utils.data

from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset

from neuralcoref.train.utils import (
    encode_distance,
    BATCH_SIZE_PATH,
    SIZE_FP,
    SIZE_FP_COMPRESSED,
    SIZE_FS,
    SIZE_FS_COMPRESSED,
    SIZE_GENRE,
    SIZE_PAIR_IN,
    SIZE_SINGLE_IN,
)
from neuralcoref.train.conllparser import FEATURES_NAMES


def load_embeddings_from_file(name):
    print("loading", name + "_embeddings.npy")
    embed = torch.from_numpy(np.load(name + "_embeddings.npy")).float()
    print(embed.size())
    print("loading", name + "_vocabulary.txt")
    with io.open(name + "_vocabulary.txt", "r", encoding="utf-8") as f:
        voc = [line.strip() for line in f]
    return embed, voc


class _DictionaryDataLoader(object):
    def __init__(self, dict_object, order):
        self.dict_object = dict_object
        self.order = order

    def __len__(self):
        return len(self.dict_object[self.order[0]])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            data = []
            for i in range(
                idx.start, idx.stop, idx.step if idx.step is not None else 1
            ):
                temp_data = []
                for key in self.order:
                    temp_data.append(self.dict_object[key][i])
                data.append(temp_data)

        else:
            data = []
            for key in self.order:
                data.append(self.dict_object[key][idx])

        return data


class NCDataset(Dataset):
    def __init__(self, data_path, params, no_targets=False):
        print("🏝 Loading Dataset at", data_path)
        self.costs = params.costs
        self.no_targets = no_targets
        # Load files
        datas = {}
        if not os.listdir(data_path):
            raise ValueError("Empty data_path")
        numpy_files_found = False
        print("Reading ", end="")
        for file_name in os.listdir(data_path):
            if not ".npy" in file_name:
                continue
            numpy_files_found = True
            print(file_name, end=", ")
            datas[file_name.split(".")[0]] = np.load(
                data_path + file_name, mmap_mode="r" if params.lazy else None
            )
        if not numpy_files_found:
            raise ValueError(f"Can't find numpy files in {data_path}")

        # Gather arrays in two lists of tuples for mention and pairs
        if not params.lazy:
            self.mentions = list(
                zip(
                    *(
                        arr
                        for key, arr in sorted(datas.items())
                        if key.startswith("mentions")
                    )
                )
            )
            self.pairs = list(
                zip(
                    *(
                        arr
                        for key, arr in sorted(datas.items())
                        if key.startswith("pairs")
                    )
                )
            )
        else:
            self.mentions = _DictionaryDataLoader(
                datas,
                order=(
                    "mentions_features",
                    "mentions_labels",
                    "mentions_pairs_length",
                    "mentions_pairs_start_index",
                    "mentions_spans",
                    "mentions_words",
                ),
            )
            self.pairs = _DictionaryDataLoader(
                datas, order=("pairs_ant_index", "pairs_features", "pairs_labels")
            )

        self.mentions_pair_length = datas[FEATURES_NAMES[2]]
        assert [arr.shape[0] for arr in self.mentions[0]] == [
            6,
            1,
            1,
            1,
            250,
            8,
        ]  # Cf order of FEATURES_NAMES in conllparser.py
        assert [arr.shape[0] for arr in self.pairs[0]] == [
            1,
            9,
            1,
        ]  # Cf order of FEATURES_NAMES in conllparser.py

    def __len__(self):
        return len(self.mentions)

    def __getitem__(self, mention_idx, debug=False):
        """
        Return:
            Definitions:
                P is the number of antecedent per mention (number of pairs for the mention)
                S = 250 is the size of the span vector (averaged word embeddings)
                W = 8 is the number of words in a mention (tuned embeddings)
                Fp = 70 is the number of features for a pair of mention
                Fs = 24 is the number of features of a single mention

            if there are some pairs:
                inputs = (spans, words, features, ant_spans, ant_words, ana_spans, ana_words, pairs_features)
                targets = (labels, costs, true_ants, false_ants)
            else:
                inputs = (spans, words, features)
                targets = (labels, costs, true_ants)

            inputs: Tuple of
                spans => (S,)
                words => (W,)
                features => (Fs,)
                + if there are potential antecedents (P > 0):
                    ant_spans => (P, S) or nothing if no pairs
                    ant_words => (P, W) or nothing if no pairs
                    ana_spans => (P, S) or nothing if no pairs
                    ana_words => (P, W) or nothing if no pairs
                    pair_features => (P, Fp) or nothing if no pairs

            targets: Tuple of
                labels => (P+1,)
                costs => (P+1,)
                true_ant => (P+1,)
                + if there are potential antecedents (P > 0):
                    false_ant => (P+1,)

        """
        features_raw, label, pairs_length, pairs_start_index, spans, words = self.mentions[
            mention_idx
        ]
        pairs_start_index = pairs_start_index.item()
        pairs_length = pairs_length.item()

        # Build features array (float) from raw features (int)
        assert features_raw.shape[0] == SIZE_FS_COMPRESSED
        features = np.zeros((SIZE_FS,))
        features[features_raw[0]] = 1
        features[4:15] = encode_distance(features_raw[1])
        features[15] = features_raw[2].astype(float) / features_raw[3].astype(float)
        features[16] = features_raw[4]
        features[features_raw[5] + 17] = 1

        if pairs_length == 0:
            spans = torch.from_numpy(spans).float()
            words = torch.from_numpy(words)
            features = torch.from_numpy(features).float()
            inputs = (spans, words, features)
            if self.no_targets:
                return inputs
            true_ant = torch.zeros(1).long()  # zeros = indices of true ant
            costs = torch.from_numpy((1 - label) * self.costs["FN"]).float()
            label = torch.from_numpy(label).float()
            targets = (label, costs, true_ant)
            if debug:
                print("inputs shapes: ", [a.size() for a in inputs])
                print("targets shapes: ", [a.size() for a in targets])
            return inputs, targets

        start = pairs_start_index
        end = pairs_start_index + pairs_length
        pairs = self.pairs[start:end]
        assert len(pairs) == pairs_length
        assert (
            len(pairs[0]) == 3
        )  # pair[i] = (pairs_ant_index, pairs_features, pairs_labels)
        pairs_ant_index, pairs_features_raw, pairs_labels = list(zip(*pairs))

        pairs_features_raw = np.stack(pairs_features_raw)
        pairs_labels = np.squeeze(np.stack(pairs_labels), axis=1)

        # Build pair features array (float) from raw features (int)
        assert pairs_features_raw[0, :].shape[0] == SIZE_FP_COMPRESSED
        pairs_features = np.zeros((len(pairs_ant_index), SIZE_FP))
        pairs_features[:, 0:6] = pairs_features_raw[:, 0:6]
        pairs_features[:, 6:17] = encode_distance(pairs_features_raw[:, 6])
        pairs_features[:, 17:28] = encode_distance(pairs_features_raw[:, 7])
        pairs_features[:, 28] = pairs_features_raw[:, 8]
        # prepare antecent features
        ant_features_raw = np.concatenate(
            [self.mentions[idx.item()][0][np.newaxis, :] for idx in pairs_ant_index]
        )
        ant_features = np.zeros((pairs_length, SIZE_FS - SIZE_GENRE))
        ant_features[:, ant_features_raw[:, 0]] = 1
        ant_features[:, 4:15] = encode_distance(ant_features_raw[:, 1])
        ant_features[:, 15] = ant_features_raw[:, 2].astype(float) / ant_features_raw[
            :, 3
        ].astype(float)
        ant_features[:, 16] = ant_features_raw[:, 4]
        pairs_features[:, 29:46] = ant_features
        # Here we keep the genre
        ana_features = np.tile(features, (pairs_length, 1))
        pairs_features[:, 46:] = ana_features

        ant_spans = np.concatenate(
            [self.mentions[idx.item()][4][np.newaxis, :] for idx in pairs_ant_index]
        )
        ant_words = np.concatenate(
            [self.mentions[idx.item()][5][np.newaxis, :] for idx in pairs_ant_index]
        )
        ana_spans = np.tile(spans, (pairs_length, 1))
        ana_words = np.tile(words, (pairs_length, 1))
        ant_spans = torch.from_numpy(ant_spans).float()
        ant_words = torch.from_numpy(ant_words)
        ana_spans = torch.from_numpy(ana_spans).float()
        ana_words = torch.from_numpy(ana_words)
        pairs_features = torch.from_numpy(pairs_features).float()

        labels_stack = np.concatenate((pairs_labels, label), axis=0)
        assert labels_stack.shape == (pairs_length + 1,)
        labels = torch.from_numpy(labels_stack).float()

        spans = torch.from_numpy(spans).float()
        words = torch.from_numpy(words)
        features = torch.from_numpy(features).float()

        inputs = (
            spans,
            words,
            features,
            ant_spans,
            ant_words,
            ana_spans,
            ana_words,
            pairs_features,
        )

        if self.no_targets:
            return inputs

        if label == 0:
            costs = np.concatenate(
                (self.costs["WL"] * (1 - pairs_labels), [self.costs["FN"]])
            )  # Inverse labels: 1=>0, 0=>1
        else:
            costs = np.concatenate((self.costs["FL"] * np.ones_like(pairs_labels), [0]))
        assert costs.shape == (pairs_length + 1,)
        costs = torch.from_numpy(costs).float()

        true_ants_unpad = np.flatnonzero(labels_stack)
        if len(true_ants_unpad) == 0:
            raise ValueError("Error: no True antecedent for mention")
        true_ants = np.pad(
            true_ants_unpad, (0, len(pairs_labels) + 1 - len(true_ants_unpad)), "edge"
        )
        assert true_ants.shape == (pairs_length + 1,)
        true_ants = torch.from_numpy(true_ants).long()

        false_ants_unpad = np.flatnonzero(1 - labels_stack)
        assert len(false_ants_unpad) != 0
        false_ants = np.pad(
            false_ants_unpad, (0, len(pairs_labels) + 1 - len(false_ants_unpad)), "edge"
        )
        assert false_ants.shape == (pairs_length + 1,)
        false_ants = torch.from_numpy(false_ants).long()

        targets = (labels, costs, true_ants, false_ants)
        if debug:
            print("Mention", mention_idx)
            print("inputs shapes: ", [a.size() for a in inputs])
            print("targets shapes: ", [a.size() for a in targets])
        return inputs, targets


class NCBatchSampler(Sampler):
    """A Batch sampler to group mentions in batches with close number of pairs to be padded together
    """

    def __init__(
        self, mentions_pairs_length, batchsize=600, shuffle=False, debug=False
    ):
        """ Create and feed batches of mentions having close number of antecedents
            The batch are padded and collated by the padder_collate function

        # Arguments:
            mentions_pairs_length array of shape (N, 1): list/array of the number of pairs for each mention
            batchsize: Number of pairs of each batch will be capped at this
        """
        self.shuffle = shuffle
        num_mentions = len(mentions_pairs_length)
        mentions_lengths = np.concatenate(
            [
                mentions_pairs_length,
                np.arange(0, num_mentions, 1, dtype=int)[:, np.newaxis],
            ],
            axis=1,
        )
        sorted_lengths = mentions_lengths[mentions_lengths[:, 0].argsort()]
        print("Preparing batches 📚")

        self.batches = []
        self.batches_pairs = []
        self.batches_size = []
        batch = []
        n_pairs = []
        num = 0
        for length, mention_idx in sorted_lengths:
            if num > batchsize or (
                num == len(batch) and length != 0
            ):  # We keep the no_pairs batches pure
                if debug:
                    print(
                        "Added batch number",
                        len(self.batches),
                        "with",
                        len(batch),
                        "mentions and",
                        num,
                        "pairs",
                    )
                self.batches.append(batch)
                self.batches_size.append(
                    num
                )  # We don't count the max 7 additional mentions that are repeated
                self.batches_pairs.append(n_pairs)

                # Start a new batch
                batch = [mention_idx]
                n_pairs = [length]
                num = (
                    length + 1
                )  # +1 since we also have the single mention to add to the number of pairs
            else:
                num += length + 1
                batch.append(mention_idx)
                n_pairs.append(length)

        # Complete and store the last batch
        if debug:
            print(
                "Added batch number",
                len(self.batches),
                "with",
                len(batch),
                "mentions and",
                num,
                "pairs",
            )
        self.batches.append(batch)
        self.batches_size.append(num)
        self.batches_pairs.append(n_pairs)
        self.n_pairs = sum(sum(p) for p in self.batches_pairs)
        self.n_mentions = sum(len(b) for b in self.batches)
        self.n_batches = len(self.batches)
        self.pairs_per_batch = float(self.n_pairs) / self.n_batches
        self.mentions_per_batch = float(self.n_mentions) / self.n_batches
        print(
            "Dataset has:",
            self.n_batches,
            "batches,",
            self.n_mentions,
            "mentions,",
            self.n_pairs,
            "pairs",
        )

    def get_batch_info(self):
        return self.batches, self.batches_pairs

    def save_batch_sizes(self, save_file=BATCH_SIZE_PATH, debug=False):
        print("🌋 Saving sizes of batches")
        with io.open(save_file, "w", encoding="utf-8") as f:
            if debug:
                print("Batch sizes saved in", save_file)
            for batch, size in zip(self.batches, self.batches_size):
                out_str = str(len(batch)) + "\t" + str(size) + "\n"
                f.write(out_str)

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return self.n_batches


def padder_collate(batch, debug=False):
    """ Puts each data field into a tensor with outer dimension batch size
        Pad variable length input tensors and add a weight tensor to the target
    """
    transposed_inputs = tuple(zip(*batch))
    if len(transposed_inputs) == 2:
        inputs, targets = transposed_inputs
        transposed_inputs = tuple(zip(*inputs))
        transposed_targets = tuple(zip(*targets))
    else:
        transposed_targets = None

    max_pairs = (
        max(len(t) for t in transposed_inputs[3]) if len(transposed_inputs) == 8 else 0
    )  # Get max nb of pairs (batch are sorted by nb of pairs)
    if max_pairs > 0:
        out_inputs = []
        out_targets = []
        for t_inp in transposed_inputs:
            if len(t_inp[0].shape) == 2:
                out_inputs.append(
                    torch.stack(
                        [
                            torch.cat([t, t.new(max_pairs - len(t), len(t[0])).zero_()])
                            if len(t) != max_pairs
                            else t
                            for t in t_inp
                        ],
                        0,
                    )
                )
            else:
                out_inputs.append(torch.stack(t_inp, 0))
        if transposed_targets is not None:
            for i, t_targ in enumerate(
                transposed_targets
            ):  # 0:labels, 1:costs, 2:true_ants, 3:false_ants
                if i == 2 or i == 3:
                    if debug:
                        print("collate before", t_targ)
                    # shift the antecedent index associated to single anaphores (last)
                    t_targ = tuple(
                        t.masked_fill_(torch.eq(t, len(t) - 1), max_pairs)
                        for t in t_targ
                    )
                    if debug:
                        print("collate after", t_targ)
                out_targets.append(
                    torch.stack(
                        [
                            torch.cat(
                                [
                                    t[:-1] if len(t) > 2 else t.new(1).fill_(t[0]),
                                    t.new(max_pairs + 1 - len(t)).fill_(t[0]),
                                    t.new(1).fill_(t[-1]),
                                ]
                            )
                            if len(t) != max_pairs + 1
                            else t
                            for t in t_targ
                        ],
                        0,
                    )
                )

            t_costs = transposed_targets[
                1
            ]  # We build the weights from the costs to have a float Tensor
            out_targets.append(
                torch.stack(
                    [
                        torch.cat(
                            [
                                t.new(len(t) - 1).fill_(1),
                                t.new(max_pairs + 1 - len(t)).zero_(),
                                t.new(1).fill_(1),
                            ]
                        )
                        if len(t) != max_pairs + 1
                        else t.new(max_pairs + 1).fill_(1)
                        for t in t_costs
                    ],
                    0,
                )
            )
        else:
            # Remark this mask is the inverse of the weights in the above target (used for evaluation masking)
            t_base = transposed_inputs[3]
            out_targets = torch.stack(
                [
                    torch.cat(
                        [
                            t.new(len(t) - 1).zero_().bool(),
                            t.new(max_pairs + 1 - len(t)).fill_(1).bool(),
                            t.new(1).zero_().bool(),
                        ]
                    )
                    if len(t) != max_pairs + 1
                    else t.new(max_pairs + 1).zero_().bool()
                    for t in t_base
                ],
                0,
            )
    else:
        out_inputs = [torch.stack(t_inp, 0) for t_inp in transposed_inputs]
        if transposed_targets is not None:
            out_targets = [torch.stack(t_targ, 0) for t_targ in transposed_targets]
            out_targets.append(out_targets[1].new(len(out_targets[1]), 1).fill_(1))
        else:
            out_targets = out_inputs[0].new(len(out_inputs[0]), 1).zero_().bool()
    return (out_inputs, out_targets)
