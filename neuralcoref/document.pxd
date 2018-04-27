from spacy.tokens.doc cimport Doc
from spacy.tokens.span cimport Span
from spacy.typedefs cimport flags_t, attr_t, hash_t

from spacy.structs cimport TokenC

cdef struct SpanC:
    int start
    int end

cdef struct SentSpans:
    SpanC* spans
    int max_spans
    int num

cdef struct Hashes:
    attr_t* arr
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
    attr_t POSSESSIVE_MARK
    attr_t NSUBJ_MARK
    attr_t IN_TAG
    attr_t MARK_DEP

cdef struct DocSpan:
    const TokenC* c
    const int start
    const int end

cdef class Mention:
    cdef readonly Span span
    cdef readonly long entity_label
    cdef readonly int in_ent
    cdef readonly int index
    cdef readonly int utterance_index
    cdef readonly int utterances_sent
    cdef readonly int mention_type
    cdef readonly propers

    cdef public object spans_embeddings
    cdef public object words_embeddings
    cdef public object embeddings
    cdef public object features
    cdef public object spans_embeddings_
    cdef public object words_embeddings_
    cdef public object features_

    cpdef int heads_agree(self, Mention mention2)
    cpdef int exact_match(self, Mention mention2)
    cpdef int relaxed_match(self, Mention mention2)
    cpdef int overlapping(self, Mention m2)

cdef class Document:
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
