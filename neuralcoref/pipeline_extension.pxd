from spacy.tokens.doc cimport Doc
from spacy.tokens.span cimport Span
from spacy.typedefs cimport flags_t, attr_t, hash_t
from spacy.vectors import Vectors
from spacy.strings cimport StringStore
from spacy.structs cimport TokenC, LexemeC
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
    Hashes conj_tags
    Hashes proper_tags
    Hashes puncts
    hash_t POSSESSIVE_MARK
    hash_t NSUBJ_MARK
    hash_t IN_TAG
    hash_t MARK_DEP

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

cdef class Model:
    cdef int n_layers
    cdef object s_weights
    cdef object s_biases
    cdef object p_weights
    cdef object p_biases

    cpdef float [:] get_score(self, float [:,:] features, bint single)

cdef class EmbeddingExtractor(object):
    cdef hash_t missing_word
    cdef hash_t digit_word
    cdef hash_t unknown_word
    cdef object static
    cdef object tuned
    cdef object shape
    cdef object fallback
    cdef object conv_dict

    cdef hash_t normalize(self, const LexemeC* c)
    cdef float [:] get_static(self, hash_t word)
    cdef float [:] get_word_embedding(self, const LexemeC* c, bint static=*)
    cdef float [:] get_word_in_sentence(self, int word_idx, TokenC* doc, int sent_start, int sent_end)
    cdef float [:] get_average_embedding(self, TokenC* doc, int start, int end, Hashes puncts, StringStore strings)
    cdef float [:] get_mention_embeddings(self, TokenC* doc, Mention_C m, Hashes puncts, StringStore strings, float [:] doc_embedding)


cdef class CorefComponent(object):
    cdef Pool mem
    cdef HashesList hashes
    cdef float greedyness
    cdef int max_dist
    cdef int max_dist_match
    cdef bint blacklist
    cdef Model coref_model
    cdef EmbeddingExtractor embed_extractor
    cdef object name
    cdef object label

    cdef build_clusters(self, Doc doc)