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

    cpdef int heads_agree(self, Mention mention2)
    cpdef int exact_match(self, Mention mention2)
    cpdef int relaxed_match(self, Mention mention2)
    cpdef int overlapping(self, Mention m2)

cdef class Document:
    cdef Mention_C* c
    cdef readonly Pool mem
    cdef readonly object nlp
    cdef readonly bint use_no_coref_list
    cdef readonly bint debug
    cdef readonly object utterances
    cdef readonly object mentions
    cdef readonly int n_sents
    cdef readonly int n_mentions
    cdef readonly int n_pairs
    cdef readonly object pairs_ant
    cdef readonly object pairs_men
    cdef readonly object genre_
    cdef readonly object genre
    cdef readonly object embed_extractor

    cdef add_utterances(self, utterances)
    cdef set_mentions_features(self)
