# coding: utf8
# cython: profile=True
# cython: infer_types=True
# distutils: language=c++
"""Coref resolution"""

from __future__ import unicode_literals
from __future__ import print_function

import sys
import os
from cpython cimport array
import array
from libc.stdint cimport uint16_t, uint32_t, uint64_t, uintptr_t, int32_t

import spacy
cimport numpy as np
np.import_array()
import numpy

from neuralcoref.utils import (PACKAGE_DIRECTORY)
from neuralcoref.compat import unicode_
from neuralcoref.document import Document, MENTION_TYPE, NO_COREF_LIST

#######################
##### UTILITIES #######
DEF SIZE_SPAN = 250 # size of the span vector (averaged word embeddings)
DEF SIZE_WORD = 8 # number of words in a mention (tuned embeddings)
DEF SIZE_EMBEDDING = 50 # size of the words embeddings
DEF SIZE_FP = 70 # number of features for a pair of mention
DEF SIZE_FP_COMPRESSED = 9 # size of the features for a pair of mentions as stored in numpy arrays
DEF SIZE_FS = 24 # number of features of a single mention
DEF SIZE_FS_COMPRESSED = 6 # size of the features for a mention as stored in numpy arrays
DEF SIZE_GENRE = 7 # Size of the genre one-hot array
DEF SIZE_MENTION_EMBEDDING = SIZE_SPAN + SIZE_WORD * SIZE_EMBEDDING # A mention embeddings (span + words vectors)
DEF SIZE_SNGL_FEATS = SIZE_FS - SIZE_GENRE
DEF SIZE_PAIR_FEATS = SIZE_FP - SIZE_GENRE
DEF SIZE_SNGL_IN_NO_GENRE = SIZE_MENTION_EMBEDDING + SIZE_SNGL_FEATS
DEF SIZE_PAIR_IN_NO_GENRE = 2 * SIZE_MENTION_EMBEDDING + SIZE_PAIR_FEATS

DEF SIZE_PAIR_IN = 2 * SIZE_MENTION_EMBEDDING + SIZE_FP # Input to the mentions pair neural network
DEF SIZE_SINGLE_IN = SIZE_MENTION_EMBEDDING + SIZE_FS  # Input to the single mention neural network

DISTANCE_BINS_PY = array.array('i', list(range(5)) + [5]*3 + [6]*8 + [7]*16 + [8]*32)
cdef int[:] DISTANCE_BINS = DISTANCE_BINS_PY
cdef float BINS_NUM = float(len(DISTANCE_BINS))
cdef int MAX_BINS = DISTANCE_BINS[-1] + 1

MAX_FOLLOW_UP = 50

cdef index_distance(int d):
    ''' Return index and value encoding to encode an integer as a (bined) one-hot array '''
    cdef float float_val = min(float(d), BINS_NUM) / BINS_NUM
    if d < 64:
        return DISTANCE_BINS[d], float_val
    return DISTANCE_BINS[-1] + 1, float_val

#######################
###### CLASSES ########

cdef class Model(object):
    '''
    Coreference neural model
    '''
    def __init__(self, model_path):
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("single_mention_weights"):
                w = numpy.load(os.path.join(model_path, file)).astype(dtype='float32')
                weights.append(w)
            if file.startswith("single_mention_bias"):
                w = numpy.load(os.path.join(model_path, file)).astype(dtype='float32')
                biases.append(w)
        self.single_mention_model = list(zip(weights, biases))
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("pair_mentions_weights"):
                w = numpy.load(os.path.join(model_path, file)).astype(dtype='float32')
                weights.append(w)
            if file.startswith("pair_mentions_bias"):
                w = numpy.load(os.path.join(model_path, file)).astype(dtype='float32')
                biases.append(w)
        self.pair_mentions_model = list(zip(weights, biases))
        self.n_layers = len(self.pair_mentions_model)

    cdef _score(self, float[:,:] features, bint single):
        layers = self.single_mention_model if single else self.pair_mentions_model
        for weights, bias in layers:
            features = numpy.matmul(weights, features) + bias
            if weights.shape[0] > 1:
                features = numpy.maximum(features, 0) # ReLU
        return numpy.sum(features, axis=0)

    def get_multiple_single_score(self, float[:,:] first_layer_input):
        return self._score(first_layer_input, True)

    def get_multiple_pair_score(self, float[:,:] first_layer_input):
        return self._score(first_layer_input, False)


class Coref(object):
    '''
    Main coreference resolution algorithm
    '''
    def __init__(self, nlp=None, greedyness=0.5, max_dist=50, max_dist_match=500, conll=None,
                 use_no_coref_list=True, debug=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.debug = debug
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/conll/" if conll is not None else "weights/")
        trained_embed_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        print("Loading neuralcoref model from", model_path)
        self.coref_model = Model(model_path)
        if nlp is None:
            print("Loading spacy model")
            try:
                spacy.info('en_core_web_sm')
                model = 'en_core_web_sm'
            except IOError:
                print("No spacy 2 model detected, using spacy1 'en' model")
                spacy.info('en')
                model = 'en'
            nlp = spacy.load(model)
        self.data = Document(nlp, conll=conll, use_no_coref_list=use_no_coref_list, trained_embed_path=trained_embed_path)
        self.clusters = {}
        self.mention_to_cluster = []
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}

    ###################################
    #### ENTITY CLUSTERS FUNCTIONS ####
    ###################################

    def _prepare_clusters(self):
        '''
        Clean up and prepare one cluster for each mention
        '''
        self.mention_to_cluster = list(range(len(self.data.mentions)))
        self.clusters = dict((i, [i]) for i in self.mention_to_cluster)
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}
        for mention in self.mention_to_cluster:
            self.mentions_single_scores[mention] = None
            self.mentions_pairs_scores[mention] = {}

    def _merge_coreference_clusters(self, ant_idx, mention_idx):
        '''
        Merge two clusters together
        '''
        if self.mention_to_cluster[ant_idx] == self.mention_to_cluster[mention_idx]:
            return

        remove_id = self.mention_to_cluster[ant_idx]
        keep_id = self.mention_to_cluster[mention_idx]
        for idx in self.clusters[remove_id]:
            self.mention_to_cluster[idx] = keep_id
            self.clusters[keep_id].append(idx)

        del self.clusters[remove_id]

    def remove_singletons_clusters(self):
        remove_id = []
        for key, mentions in self.clusters.items():
            if len(mentions) == 1:
                remove_id.append(key)
                self.mention_to_cluster[key] = None
        for rem in remove_id:
            del self.clusters[rem]

    def display_clusters(self):
        '''
        Print clusters informations
        '''
        print(self.clusters)
        for key, mentions in self.clusters.items():
            print("cluster", key, "(", ", ".join(unicode_(self.data[m]) for m in mentions), ")")

    ###################################
    ####### MAIN COREF FUNCTIONS ######
    ###################################

    def run_coref_on_utterances(self):
        '''
        Run the coreference model on a mentions list
        '''
        self._prepare_clusters()

        cdef int i, j1, j2, b_idx
        cdef float val
        cdef int n_ant = 0
        cdef np.ndarray genre = self.data.genre
        cdef int SZ_GENRE = genre.shape[0]
        cdef np.ndarray[uint64_t, ndim=1] p_ant = self.data.pairs_ant
        cdef np.ndarray[uint64_t, ndim=1] p_men = self.data.pairs_men
        cdef int ant_idx
        cdef int men_idx
        cdef float [:] score, embed, feats, embed2, feats2
        cdef int n_mentions = self.data.n_mentions
        cdef int n_pairs = self.data.n_pairs
        inp_ar = numpy.empty((SIZE_SNGL_IN_NO_GENRE + SZ_GENRE, n_mentions), dtype='float32')
        cdef float [:, :] inp = inp_ar
        best_score_ar = numpy.empty((n_mentions), dtype='float32')
        cdef float [:] best_score = best_score_ar
        inp2_ar = numpy.empty((SIZE_PAIR_IN_NO_GENRE + SZ_GENRE, n_pairs), dtype='float32')
        cdef float [:, :] inp2 = inp2_ar
        best_ant_ar = numpy.empty((n_mentions), dtype=numpy.uint64)
        cdef uint64_t [:] best_ant = best_ant_ar
        for i in range(n_mentions):
            mention = self.data.mentions[i]
            embed = mention.embeddings
            feats = mention.features
            inp[:SIZE_MENTION_EMBEDDING, i] = embed
            inp[SIZE_MENTION_EMBEDDING:SIZE_MENTION_EMBEDDING+SIZE_SNGL_FEATS, i] = feats
            inp[SIZE_MENTION_EMBEDDING+SIZE_SNGL_FEATS:, i] = self.data.genre
        score = self.coref_model.get_multiple_single_score(inp)
        for i in range(n_mentions):
            best_score[i] = score[i] - 50 * (self.greedyness - 0.5)
            best_ant[i] = i

        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            m1 = self.data.mentions[ant_idx]
            embed = m1.embeddings
            feats = m1.features
            m2 = self.data.mentions[men_idx]
            embed2 = m2.embeddings
            feats2 = m2.features
            j1 = SIZE_MENTION_EMBEDDING
            inp2[:j1, i] = embed
            j2 = j1 + SIZE_MENTION_EMBEDDING
            inp2[j1:j2, i] = embed2
            j1 = j2
            inp2[j1, i] = 1
            j1 += 1
            inp2[j1, i] = 0
            j1 += 1
            inp2[j1, i] = 0
            j1 += 1
            inp2[j1, i] = m1.heads_agree(m2)
            j1 += 1
            inp2[j1, i] = m1.exact_match(m2)
            j1 += 1
            inp2[j1, i] = m1.relaxed_match(m2)
            j1 += 1
            b_idx, val = index_distance(m2.utterances_sent - m1.utterances_sent)
            inp2[j1:j1 + MAX_BINS + 1, i] = 0
            inp2[j1 + b_idx, i] = 1
            j1 += MAX_BINS + 1;
            inp2[j1, i] = val
            j1 += 1
            b_idx, val = index_distance(m2.index - m1.index - 1)
            inp2[j1:j1 + MAX_BINS + 1, i] = 0
            inp2[j1+b_idx, i] = 1
            j1 += MAX_BINS + 1;
            inp2[j1, i] = val
            j1 += 1
            inp2[j1, i] = m1.overlapping(m2)
            j1 += 1
            j2 = j1 + SIZE_SNGL_FEATS
            inp2[j1:j2, i] = feats
            j1 = j2
            j2 = j1 + SIZE_SNGL_FEATS
            inp2[j1:j2, i] = feats2
            j1 = j2
            inp2[j1:, i] = self.data.genre
        score = self.coref_model.get_multiple_pair_score(inp2)
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            if score[i] > best_score[men_idx]:
                best_score[men_idx] = score[i]
                best_ant[men_idx] = ant_idx
        for i in range(n_mentions):
            if best_ant[i] != i:
                n_ant += 1
                self._merge_coreference_clusters(best_ant[i], i)
        return (n_ant, best_ant)

    def one_shot_coref(self, utterances):
        ''' Clear history, load a list of utterances and run the coreference model on them

        Arg:
        - `utterances` : iterator or list of string corresponding to successive utterances (in a dialogue) or sentences.
            Can be a single string for non-dialogue text.
        Return:
            clusters of entities with coreference resolved
        '''
        self.continuous_coref(utterances)
        return self.get_clusters()

    def continuous_coref(self, utterances):
        '''
        Only resolve coreferences for the mentions in the utterances
        (but use the mentions in previously loaded utterances as possible antecedents)
        Arg:
            utterances : iterator or list of string corresponding to successive utterances
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.add_utterances(utterances)
        self.run_coref_on_utterances()
        return self.get_clusters()

    ###################################
    ###### INFORMATION RETRIEVAL ######
    ###################################

    def get_utterances(self):
        ''' Retrieve the list of parsed uterrances'''
        return self.data.utterances

    def get_resolved_utterances(self, use_no_coref_list=True):
        ''' Return a list of utterrances text where the coref are resolved to the most representative mention'''
        coreferences = self.get_most_representative(use_no_coref_list)
        resolved_utterances = []
        for utt in self.get_utterances():
            resolved_utt = ""
            in_coref = None
            for token in utt:
                if in_coref is None:
                    for coref_original, coref_replace in coreferences.items():
                        if coref_original[0] == token:
                            in_coref = coref_original
                            resolved_utt += coref_replace.span.lower_
                            break
                    if in_coref is None:
                        resolved_utt += token.text_with_ws
                if in_coref is not None and token == in_coref[-1]:
                    resolved_utt += ' ' if token.whitespace_ and resolved_utt[-1] is not ' ' else ''
                    in_coref = None
            resolved_utterances.append(resolved_utt)
        return resolved_utterances

    def get_mentions(self):
        ''' Retrieve the list of mentions'''
        return self.data.mentions

    def get_scores(self):
        ''' Retrieve scores for single mentions and pair of mentions'''
        return {"single_scores": self.mentions_single_scores,
                "pair_scores": self.mentions_pairs_scores}

    def get_clusters(self, remove_singletons=False, use_no_coref_list=False):
        ''' Retrieve cleaned clusters'''
        clusters = self.clusters
        mention_to_cluster = self.mention_to_cluster
        remove_id = []
        if use_no_coref_list:
            for key, mentions in clusters.items():
                cleaned_list = []
                for mention_idx in mentions:
                    mention = self.data.mentions[mention_idx]
                    if mention.span.lower_ not in NO_COREF_LIST:
                        cleaned_list.append(mention_idx)
                clusters[key] = cleaned_list
            # Also clean up keys so we can build coref chains in self.get_most_representative
            added = {}
            for key, mentions in clusters.items():
                if self.data.mentions[key].span.lower_ in NO_COREF_LIST:
                    remove_id.append(key)
                    mention_to_cluster[key] = None
                    if mentions:
                        added[mentions[0]] = mentions
            for rem in remove_id:
                del clusters[rem]
            clusters.update(added)

        if remove_singletons:
            remove_id = []
            for key, mentions in clusters.items():
                if len(mentions) == 1:
                    remove_id.append(key)
                    mention_to_cluster[key] = None
            for rem in remove_id:
                del clusters[rem]

        return clusters, mention_to_cluster

    def get_most_representative(self, use_no_coref_list=True):
        '''
        Find a most representative mention for each cluster

        Return:
            Dictionnary of {original_mention: most_representative_resolved_mention, ...}
        '''
        clusters, _ = self.get_clusters(remove_singletons=True, use_no_coref_list=use_no_coref_list)
        coreferences = {}
        cdef int key
        for key in range(self.data.n_mentions):
            if self.mention_to_cluster[key] is None:
                continue
            mentions = clusters.get(self.mention_to_cluster[key], None)
            if mentions is None:
                continue
            representative = self.data.mentions[key]
            for mention_idx in mentions[1:]:
                mention = self.data.mentions[mention_idx]
                if mention.mention_type is not representative.mention_type:
                    if mention.mention_type == MENTION_TYPE["PROPER"] \
                        or (mention.mention_type == MENTION_TYPE["NOMINAL"] and
                                representative.mention_type == MENTION_TYPE["PRONOMINAL"]):
                        coreferences[self.data.mentions[key]] = mention
                        representative = mention

        return coreferences
