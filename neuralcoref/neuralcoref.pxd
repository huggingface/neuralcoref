from spacy.tokens.doc cimport Doc
from spacy.tokens.span cimport Span
from spacy.typedefs cimport flags_t, attr_t, hash_t
from spacy.vectors import Vectors
from spacy.vocab cimport Vocab
from spacy.structs cimport TokenC, LexemeC
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
    Hashes conj_tags
    Hashes proper_tags
    Hashes puncts
    hash_t POSSESSIVE_MARK
    hash_t NSUBJ_MARK
    hash_t IN_TAG
    hash_t MARK_DEP
    hash_t missing_word
    hash_t digit_word
    hash_t unknown_word

cdef struct Mention_C:
    hash_t entity_label
    int span_root
    int span_start
    int span_end
    int sent_idx
    int sent_start
    int sent_end
    int mention_type
    hash_t root_lower
    hash_t span_lower
    Hashes content_words

cdef class NeuralCoref(object):
    cdef HashesList hashes
    cdef readonly Vocab vocab
    cdef readonly object cfg
    cdef public object model
    cdef public object static_vectors
    cdef public object tuned_vectors
    cdef public object conv_dict
