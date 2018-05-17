cimport numpy as np
from neuralcoref.document cimport Document

cdef class Model(object):
    cdef object single_mention_model
    cdef object pair_mentions_model
    cdef int n_layers

    cdef _score(self, float[:,:] features, bint single)

cdef class Coref(object):
    cdef float greedyness
    cdef int max_dist
    cdef int max_dist_match
    cdef bint debug
    cdef Model coref_model
    cdef Document data
    cdef object clusters
    cdef object mention_to_cluster
    cdef object mentions_single_scores
    cdef object mentions_pairs_scores
    cdef object inp
    cdef object inp2

    cdef _run_coref_on_utterances(self)