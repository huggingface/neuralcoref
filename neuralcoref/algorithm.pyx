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
from spacy.typedefs cimport hash_t

from neuralcoref.utils import PACKAGE_DIRECTORY
from neuralcoref.compat import unicode_
from neuralcoref.document import Document, MENTION_TYPE, NO_COREF_LIST
from neuralcoref.document cimport Mention_C, Hashes

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

DEF MAX_BINS = 9

DEF PAIR_FEATS_00 = SIZE_MENTION_EMBEDDING
DEF PAIR_FEATS_01 = 2*SIZE_MENTION_EMBEDDING
DEF PAIR_FEATS_02 = PAIR_FEATS_01+6
DEF PAIR_FEATS_03 = PAIR_FEATS_02 + MAX_BINS + 1
DEF PAIR_FEATS_04 = PAIR_FEATS_03 + 1
DEF PAIR_FEATS_05 = PAIR_FEATS_04 + MAX_BINS + 1
DEF PAIR_FEATS_06 = PAIR_FEATS_05 + 2
DEF PAIR_FEATS_07 = PAIR_FEATS_06 + SIZE_SNGL_FEATS
DEF PAIR_FEATS_08 = PAIR_FEATS_07 + SIZE_SNGL_FEATS

DEF MAX_FOLLOW_UP = 50

DISTANCE_BINS_PY = array.array('i', list(range(5)) + [5]*3 + [6]*8 + [7]*16 + [8]*32)
cdef:
    int[:] DISTANCE_BINS = DISTANCE_BINS_PY
    float BINS_NUM = float(len(DISTANCE_BINS))
    int MAX_BINS = DISTANCE_BINS[-1] + 1

cdef bint inside(hash_t element, Hashes hashes):
    cdef int i
    cdef hash_t* arr = hashes.arr
    cdef int length = hashes.length
    for i in range(length):
        if arr[i] == element:
            return True
    return False

cdef index_distance(int d):
    ''' Return index and value encoding to encode an integer as a (bined) one-hot array '''
    cdef float float_val = min(float(d), BINS_NUM) / BINS_NUM
    if d < 64:
        # print('DISTANCE BINS LEN', len(DISTANCE_BINS), d, float_val)
        return DISTANCE_BINS[d], float_val
    return DISTANCE_BINS[-1] + 1, float_val

cdef heads_agree(Mention_C m1, Mention_C m2):
    ''' Does the root of the Mention match the root of another Mention/Span'''
    # In CoreNLP: they allow same-type NEs to not match perfectly
    # but rather one could be included in the other, e.g., "George" -> "George Bush"
    # => See the python variant in the Mention class
    # In this cython C function, we take the simpler approach of directly comparing the roots
    return 1 if m1.root_lower == m2.root_lower else 0

cdef exact_match(Mention_C m1, Mention_C m2):
    ''' Does the Mention lowercase text matches another Mention/Span lowercase text'''
    return 1 if m1.span_lower == m2.span_lower else 0

cdef relaxed_match(Mention_C m1, Mention_C m2):
    ''' Does the content words in m1 have at least one element in commmon
        with m2 content words'''
    cdef int i
    for i in range(m1.content_words.length):
        if inside(m1.content_words.arr[i], m2.content_words):
            return True
    return False

cdef overlapping(Mention_C m1, Mention_C m2):
    if (m1.sent_idx == m2.sent_idx and m1.span_end > m2.span_start):
        return 1
    else:
        return 0

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


cdef class Coref(object):
    '''
    Main coreference resolution algorithm
    '''
    def __cinit__(self, nlp=None, spacy_model='en', greedyness=0.5, max_dist=50, max_dist_match=500, conll=None,
                  blacklist=False, debug=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.debug = debug
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/conll/" if conll is not None else "weights/")
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        print("Loading neuralcoref model from", model_path)
        self.coref_model = Model(model_path)
        if nlp is None:
            print("Loading spacy model")
            spacy.info(spacy_model)
            nlp = spacy.load(spacy_model)
        else:
            print("Using provided spacy model")
        self.data = Document(nlp, conll=conll, blacklist=blacklist, model_path=model_path)
        self.clusters = {}
        self.mention_to_cluster = []
        self.mentions_single_scores = {}
        self.mentions_pairs_scores = {}
        self.inp = None
        self.inp2 = None

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

        keep_id = self.mention_to_cluster[ant_idx]
        remove_id = self.mention_to_cluster[mention_idx]
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

    ###################################
    ####### MAIN COREF FUNCTIONS ######
    ###################################

    def run_coref_on_utterances(self):
        '''
        Run the coreference model on a mentions list
        '''
        self._prepare_clusters()
        n_corefs = 0
        (best_ant, score, scorep, inp, inp2) = self._run_coref_on_utterances()
        self.inp = numpy.transpose(numpy.asarray(inp))
        self.inp2 = numpy.transpose(numpy.asarray(inp2))
        for i in range(self.data.n_mentions):
            self.mentions_single_scores[i] = score[i]
            if best_ant[i] != i:
                n_corefs += 1
                self._merge_coreference_clusters(best_ant[i], i)
        for i in range(self.data.n_pairs):
            ant_idx = self.data.pairs_ant[i]
            men_idx = self.data.pairs_men[i]
            self.mentions_pairs_scores[men_idx][ant_idx] = scorep[i]
        return n_corefs

    def get_pair_features(self, idx_m1, idx_m2):
        ''' Retrieve the features of a pair of mentions'''
        for i in range(self.data.n_pairs):
            if idx_m1 == self.data.pairs_ant[i] and idx_m2 == self.data.pairs_men[i]:
                m1 = self.data.mentions[idx_m1]
                m2 = self.data.mentions[idx_m2]
                return (m1, m2, self.inp2[i], self.data.get_pair_mentions_features(m1, m2))
        return None

    def get_single_features(self, m_idx):
        ''' Retrieve the features of a mentions'''
        m = self.data.mentions[m_idx]
        return (m, self.inp[m_idx], self.data.get_single_mention_features(m))

    cdef _run_coref_on_utterances(self):
        cdef:
            int i, j1, j2, b_idx
            float val
            np.ndarray genre = self.data.genre
            int SZ_GENRE = genre.shape[0]
            np.ndarray[uint64_t, ndim=1] p_ant = self.data.pairs_ant
            np.ndarray[uint64_t, ndim=1] p_men = self.data.pairs_men
            int ant_idx
            int men_idx
            float [:] score, embed, feats, embed2, feats2
            Mention_C mention
            int n_mentions = self.data.n_mentions
            int n_pairs = self.data.n_pairs
        inp_ar = numpy.empty((SIZE_SNGL_IN_NO_GENRE + SZ_GENRE, n_mentions), dtype='float32')
        best_score_ar = numpy.empty((n_mentions), dtype='float32')
        inp2_ar = numpy.empty((SIZE_PAIR_IN_NO_GENRE + SZ_GENRE, n_pairs), dtype='float32')
        best_ant_ar = numpy.empty((n_mentions), dtype=numpy.uint64)
        cdef:
            float [:, :] inp = inp_ar
            float [:] best_score = best_score_ar
            float [:, :] inp2 = inp2_ar
            uint64_t [:] best_ant = best_ant_ar
        # Single mention scores
        # print('Build feature list for Single mention scores')
        for i in range(n_mentions):
            mention = (<Mention_C*>self.data.c)[i]
            embed = <float[:SIZE_MENTION_EMBEDDING]>mention.embeddings
            feats = <float[:SIZE_SNGL_FEATS]>mention.features
            inp[:SIZE_MENTION_EMBEDDING, i] = embed
            inp[SIZE_MENTION_EMBEDDING:SIZE_MENTION_EMBEDDING+SIZE_SNGL_FEATS, i] = feats
            inp[SIZE_MENTION_EMBEDDING+SIZE_SNGL_FEATS:, i] = self.data.genre
        # print('Computing Single mention scores')
        score = self.coref_model.get_multiple_single_score(inp)
        for i in range(n_mentions):
            best_score[i] = score[i] - 50 * (self.greedyness - 0.5)
            best_ant[i] = i
        # Pairs of mentions scores
        # print('Build feature list for pair mention scores')
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            m1 = (<Mention_C*>self.data.c)[ant_idx]
            embed = <float[:SIZE_MENTION_EMBEDDING]>m1.embeddings
            feats = <float[:SIZE_SNGL_FEATS]>m1.features
            m2 = (<Mention_C*>self.data.c)[men_idx]
            embed2 = <float[:SIZE_MENTION_EMBEDDING]>m2.embeddings
            feats2 = <float[:SIZE_SNGL_FEATS]>m2.features
            inp2[:PAIR_FEATS_00, i] = embed
            inp2[PAIR_FEATS_00:PAIR_FEATS_01, i] = embed2
            inp2[PAIR_FEATS_01, i] = 1
            inp2[PAIR_FEATS_01 + 1, i] = 0
            inp2[PAIR_FEATS_01 + 2, i] = 0
            inp2[PAIR_FEATS_01 + 3, i] = heads_agree(m1, m2)
            inp2[PAIR_FEATS_01 + 4, i] = exact_match(m1, m2)
            inp2[PAIR_FEATS_01 + 5, i] = relaxed_match(m1, m2)
            b_idx, val = index_distance(m2.sent_idx - m1.sent_idx)
            inp2[PAIR_FEATS_02:PAIR_FEATS_03, i] = 0
            inp2[PAIR_FEATS_02 + b_idx, i] = 1
            inp2[PAIR_FEATS_03, i] = val
            b_idx, val = index_distance(men_idx - ant_idx - 1)
            inp2[PAIR_FEATS_04:PAIR_FEATS_05, i] = 0
            inp2[PAIR_FEATS_04 + b_idx, i] = 1
            inp2[PAIR_FEATS_05, i] = val
            inp2[PAIR_FEATS_05 + 1, i] = overlapping(m1, m2)
            inp2[PAIR_FEATS_06:PAIR_FEATS_07, i] = feats
            inp2[PAIR_FEATS_07:PAIR_FEATS_08, i] = feats2
            inp2[PAIR_FEATS_08:, i] = self.data.genre
        # print('Computing pair mention scores')
        scorep = self.coref_model.get_multiple_pair_score(inp2)
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            if scorep[i] > best_score[men_idx]:
                best_score[men_idx] = scorep[i]
                best_ant[men_idx] = ant_idx
        return (best_ant, score, scorep, inp, inp2)

    def one_shot_coref(self, utterances, debug=False):
        ''' Clear history, load a list of utterances and run the coreference model on them

        Arg:
        - `utterances` : iterator or list of string corresponding to successive utterances (in a dialogue) or sentences.
            Can be a single string for non-dialogue text.
        Return:
            clusters of entities with coreference resolved
        '''
        self.data.set_utterances(utterances)
        n_corefs = self.run_coref_on_utterances()
        if debug: print("Found corefs:", n_corefs)
        return self.get_clusters()

    ###################################
    ###### INFORMATION RETRIEVAL ######
    ###################################

    def get_utterances(self):
        ''' Retrieve the list of parsed uterrances'''
        return self.data.utterances

    def get_resolved_utterances(self, blacklist=False):
        ''' Return a list of utterrances text where the coref are resolved to the most representative mention'''
        coreferences = self.get_most_representative(blacklist)
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

    def get_scores(self, strings=False):
        ''' Retrieve scores for single mentions and pair of mentions'''
        if strings:
            def st(m):
                return "{:03d}".format(m) + '_' + str(self.data.mentions[m])
            out_s = dict((st(m), s) for m,s in self.mentions_single_scores.items())
            out_p = dict((st(m), dict((st(m2), s2) for m2, s2 in ds.items())) \
                         for m, ds in self.mentions_pairs_scores.items())
            return {"single_scores": out_s,
                    "pair_scores": out_p}
        return {"single_scores": self.mentions_single_scores,
                "pair_scores": self.mentions_pairs_scores}

    def get_clusters(self, remove_singletons=False, blacklist=False, strings=False):
        ''' Retrieve cleaned clusters'''
        clusters = self.clusters
        mention_to_cluster = self.mention_to_cluster
        remove_id = []
        if blacklist:
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

        if strings:
            def st(m):
                return "{:03d}".format(m) + '_' + str(self.data.mentions[m])
            out_c = dict((st(m), list(st(a) for a in a_l)) for m, a_l in clusters.items())
            out_m = dict((st(m), st(a)) for m, a in enumerate(mention_to_cluster))
            return {"clusters": out_c,
                    "mention_to_cluster": out_m}
        return {"clusters": clusters,
                "mention_to_cluster": mention_to_cluster}

    def get_most_representative(self, blacklist=False):
        '''
        Find a most representative mention for each cluster

        Return:
            Dictionnary of {original_mention: most_representative_resolved_mention, ...}
        '''
        clusters, _ = self.get_clusters(remove_singletons=True, blacklist=blacklist)
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
