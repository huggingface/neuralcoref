cimport numpy as np

cdef class Model(object):
    cdef object single_mention_model
    cdef object pair_mentions_model
    cdef int n_layers

    cdef _score(self, float[:,:] features, bint single)