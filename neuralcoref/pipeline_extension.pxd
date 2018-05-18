from spacy.tokens.doc cimport Doc
from spacy.tokens.span cimport Span
from spacy.typedefs cimport flags_t, attr_t, hash_t

from spacy.structs cimport TokenC
cimport numpy as np
from cymem.cymem cimport Pool

cdef struct SpanC:
    int start
    int end

cdef struct SentSpans:
    SpanC* spans
    int max_spans
    int num

cdef struct Hashes:
    hash_t* arr
    int length

cdef struct HashesList:
    Hashes no_coref_list
    Hashes keep_tags
    Hashes PRP_tags
    Hashes leave_dep
    Hashes keep_dep
    Hashes nsubj_or_dep
    Hashes conj_or_prep
    Hashes remove_pos
    Hashes lower_not_end
    hash_t POSSESSIVE_MARK
    hash_t NSUBJ_MARK
    hash_t IN_TAG
    hash_t MARK_DEP

cdef struct Mention_C:
    long entity_label
    int span_start
    int span_end
    int utt_idx
    int sent_idx
    int mention_type
    hash_t root_lower
    hash_t span_lower
    Hashes content_words
    float* embeddings
    float* features

cdef class Mention:
    cdef readonly Span span
    cdef public int utt_idx
    cdef public int sent_idx
    cdef public object spans_embeddings_
    cdef public object words_embeddings_
    cdef public object features_
    cdef public object embeddings
    cdef public object features
    cdef public object content_words

cdef class CorefComponent(object):
    cdef readonly Pool mem
    cdef public float greedyness
    cdef public int max_dist
    cdef public int max_dist_match
    cdef public bint blacklist
    cdef public object coref_model
    cdef public object embed_extractor
    cdef public object name
    cdef public object label

    cdef build_clusters(self, doc)