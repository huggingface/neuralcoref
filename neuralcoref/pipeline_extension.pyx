# coding: utf8
# cython: profile=True, infer_types=True, boundscheck=False
# distutils: language=c++
"""Coref resolution spaCy v2.0 pipeline component 
Custom pipeline components: https://spacy.io//usage/processing-pipelines#custom-components
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

import plac
import re
import os
import io
from collections import OrderedDict
from cpython cimport array
cimport cython
import array
from libc.stdint cimport uint16_t, uint32_t, uint64_t, uintptr_t, int32_t

cimport numpy as np
np.import_array()
import numpy

from cymem.cymem cimport Pool
from cpython.exc cimport PyErr_CheckSignals

import spacy
from spacy.typedefs cimport hash_t
from spacy.structs cimport LexemeC, TokenC
from spacy.lang.en import English
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token
from spacy.tokens.token cimport Token
from spacy.strings cimport StringStore
from spacy.vocab cimport Vocab
from spacy.lexeme cimport Lexeme
from spacy.attrs cimport IS_DIGIT
from spacy.vectors import Vectors

from neuralcoref.utils import PACKAGE_DIRECTORY, encode_distance
from neuralcoref.compat import unicode_

##############################
##### A BUNCH OF SIZES #######
DEF SIZE_WORD = 8 # number of words in a mention (tuned embeddings)
DEF SIZE_EMBEDDING = 50 # size of the words embeddings
DEF SIZE_SPAN = 5 * SIZE_EMBEDDING # size of the span vector (averaged word embeddings)
DEF SIZE_PAIR_FEATS = 63 # number of features for a pair of mention
DEF SIZE_FP_COMPRESSED = 9 # size of the features for a pair of mentions as stored in numpy arrays
DEF SIZE_SNGL_FEATS = 17 # number of features of a single mention
DEF SIZE_FS_COMPRESSED = 6 # size of the features for a mention as stored in numpy arrays
DEF SIZE_GENRE = 1 # Size of the genre one-hot array when no conll is used
DEF SIZE_MENTION_EMBEDDING = SIZE_SPAN + SIZE_WORD * SIZE_EMBEDDING # A mention embeddings (span + words vectors)
DEF SIZE_FS = SIZE_SNGL_FEATS + SIZE_GENRE
DEF SIZE_FP = SIZE_PAIR_FEATS + SIZE_GENRE
DEF SIZE_SNGL_IN_NO_GENRE = SIZE_MENTION_EMBEDDING + SIZE_SNGL_FEATS
DEF SIZE_PAIR_IN_NO_GENRE = 2 * SIZE_MENTION_EMBEDDING + SIZE_PAIR_FEATS
DEF SIZE_PAIR_IN = 2 * SIZE_MENTION_EMBEDDING + SIZE_FP # Input to the mentions pair neural network
DEF SIZE_SINGLE_IN = SIZE_MENTION_EMBEDDING + SIZE_FS  # Input to the single mention neural network

DEF PAIR_FEATS_0 = SIZE_MENTION_EMBEDDING
DEF PAIR_FEATS_1 = 2 * SIZE_MENTION_EMBEDDING
DEF PAIR_FEATS_2 = PAIR_FEATS_1 + 6
DEF PAIR_FEATS_3 = PAIR_FEATS_2 + MAX_BINS + 1
DEF PAIR_FEATS_4 = PAIR_FEATS_3 + 1
DEF PAIR_FEATS_5 = PAIR_FEATS_4 + MAX_BINS + 1
DEF PAIR_FEATS_6 = PAIR_FEATS_5 + 2
DEF PAIR_FEATS_7 = PAIR_FEATS_6 + SIZE_SNGL_FEATS
DEF PAIR_FEATS_8 = PAIR_FEATS_7 + SIZE_SNGL_FEATS

DEF SGNL_FEATS_0 = SIZE_MENTION_EMBEDDING
DEF SGNL_FEATS_1 = SGNL_FEATS_0 + 4
DEF SGNL_FEATS_2 = SGNL_FEATS_1 + MAX_BINS + 1
DEF SGNL_FEATS_3 = SGNL_FEATS_2 + 1
DEF SGNL_FEATS_4 = SGNL_FEATS_3 + 1
DEF SGNL_FEATS_5 = SGNL_FEATS_4 + 1

DEF EMBED_01 = SIZE_EMBEDDING
DEF EMBED_02 = 2 * SIZE_EMBEDDING
DEF EMBED_03 = 3 * SIZE_EMBEDDING
DEF EMBED_04 = 4 * SIZE_EMBEDDING
DEF EMBED_05 = 5 * SIZE_EMBEDDING
DEF EMBED_06 = 6 * SIZE_EMBEDDING
DEF EMBED_07 = 7 * SIZE_EMBEDDING
DEF EMBED_08 = 8 * SIZE_EMBEDDING
DEF EMBED_09 = 9 * SIZE_EMBEDDING
DEF EMBED_10 = 10 * SIZE_EMBEDDING
DEF EMBED_11 = 11 * SIZE_EMBEDDING
DEF EMBED_12 = 12 * SIZE_EMBEDDING
DEF EMBED_13 = 13 * SIZE_EMBEDDING

DEF MAX_BINS = 9
DEF MAX_FOLLOW_UP = 50
DEF MAX_ITER = 100
DEF SPAN_FACTOR = 4

DISTANCE_BINS_PY = array.array('i', list(range(5)) + [5]*3 + [6]*8 + [7]*16 + [8]*32)

cdef:
    int [:] DISTANCE_BINS = DISTANCE_BINS_PY
    int BINS_NUM = len(DISTANCE_BINS)

##########################################################
##### STRINGS USED IN RULE_BASED MENTION DETECTION #######

NO_COREF_LIST = ["i", "me", "my", "you", "your"]
MENTION_TYPE = {"PRONOMINAL": 0, "NOMINAL": 1, "PROPER": 2, "LIST": 3}
MENTION_LABEL = {0: "PRONOMINAL", 1: "NOMINAL", 2: "PROPER", 3: "LIST"}
KEEP_TAGS = ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "DT", "IN"]
CONTENT_TAGS = ["NN", "NNS", "NNP", "NNPS"]
PRP_TAGS = ["PRP", "PRP$"]
CONJ_TAGS = ["CC", ","]
PROPER_TAGS = ["NNP", "NNPS"]
NSUBJ_OR_DEP = ["nsubj", "dep"]
CONJ_OR_PREP = ["conj", "prep"]
LEAVE_DEP = ["det", "compound", "appos"]
KEEP_DEP = ["nsubj", "dobj", "iobj", "pobj"]
REMOVE_POS = ["CCONJ", "INTJ", "ADP"]
LOWER_NOT_END = ["'s", ',', '.', '!', '?', ':', ';']
PUNCTS = [".", "!", "?"]
ACCEPTED_ENTS = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"]
WHITESPACE_PATTERN = r"\s+|_+"
UNKNOWN_WORD = "*UNK*"
MISSING_WORD = "<missing>"

###############################################################
##### UTILITIES TO CONVERT SAID STRINGS IN SPACY HASHES #######

cdef set_hashes_list(Hashes* hashes, py_list, StringStore store, Pool mem):
    hashes.length = len(py_list)
    hashes.arr = <hash_t*>mem.alloc(hashes.length, sizeof(hash_t))
    for i, st in enumerate(py_list):
        hashes.arr[i] = store.add(st)

cdef HashesList get_hash_lookups(StringStore store, Pool mem):
    cdef HashesList hashes
    set_hashes_list(&hashes.no_coref_list, NO_COREF_LIST, store, mem)
    set_hashes_list(&hashes.keep_tags, KEEP_TAGS, store, mem)
    set_hashes_list(&hashes.PRP_tags, PRP_TAGS, store, mem)
    set_hashes_list(&hashes.leave_dep, LEAVE_DEP, store, mem)
    set_hashes_list(&hashes.keep_dep, KEEP_DEP, store, mem)
    set_hashes_list(&hashes.nsubj_or_dep, NSUBJ_OR_DEP, store, mem)
    set_hashes_list(&hashes.conj_or_prep, CONJ_OR_PREP, store, mem)
    set_hashes_list(&hashes.remove_pos, REMOVE_POS, store, mem)
    set_hashes_list(&hashes.lower_not_end, LOWER_NOT_END, store, mem)
    set_hashes_list(&hashes.conj_tags, CONJ_TAGS, store, mem)
    set_hashes_list(&hashes.proper_tags, PROPER_TAGS, store, mem)
    set_hashes_list(&hashes.proper_tags, PROPER_TAGS, store, mem)
    set_hashes_list(&hashes.puncts, PUNCTS, store, mem)
    hashes.POSSESSIVE_MARK = store.add("'s")
    hashes.NSUBJ_MARK = store.add("nsubj")
    hashes.IN_TAG = store.add('IN')
    hashes.MARK_DEP = store.add("mark")
    return hashes

@cython.profile(False)
cdef inline bint inside(hash_t element, Hashes hashes) nogil:
    cdef int i
    cdef hash_t* arr = hashes.arr
    cdef int length = hashes.length
    for i in range(length):
        if arr[i] == element:
            return True
    return False

#########################################
##### A BUNCH OF CYTHON UTILITIES #######

@cython.profile(False)
cdef inline int is_nested(Mention_C* c, int n_mentions, int m_idx):
    for i in range(n_mentions):
        if i == m_idx:
            continue
        if c[i].sent_idx == c[m_idx].sent_idx \
            and c[i].span_start <= c[m_idx].span_start \
            and c[i].span_end >= c[m_idx].span_end:
            return 1
    return 0

@cython.profile(False)
cdef inline (int, float) index_distance(int d) nogil:
    ''' Return index and value encoding to encode an integer as a (bined) one-hot array '''
    global BINS_NUM, DISTANCE_BINS
    cdef float float_val
    cdef int bin_d
    float_val = min(float(d), float(BINS_NUM))
    if BINS_NUM != 0:
        float_val = float_val/ BINS_NUM
    bin_d = DISTANCE_BINS[d] if d < 64 else DISTANCE_BINS[BINS_NUM-1] + 1
    return bin_d, float_val

@cython.profile(False)
cdef inline int heads_agree(Mention_C m1, Mention_C m2) nogil:
    ''' Does the root of the Mention match the root of another Mention/Span'''
    # In CoreNLP: they allow same-type NEs to not match perfectly
    # but rather one could be included in the other, e.g., "George" -> "George Bush"
    # => See the python variant in the Mention class
    # In this cython C function, we take the simpler approach of directly comparing the roots hashes
    return 1 if m1.root_lower == m2.root_lower else 0

@cython.profile(False)
cdef inline int exact_match(Mention_C m1, Mention_C m2) nogil:
    ''' Does the Mention lowercase text matches another Mention/Span lowercase text'''
    return 1 if m1.span_lower == m2.span_lower else 0

@cython.profile(False)
cdef inline int relaxed_match(Mention_C m1, Mention_C m2) nogil:
    ''' Does the content words in m1 have at least one element in commmon
        with m2 content words'''
    cdef int i
    for i in range(m1.content_words.length):
        if inside(m1.content_words.arr[i], m2.content_words):
            return True
    return False

@cython.profile(False)
cdef inline int overlapping(Mention_C m1, Mention_C m2) nogil:
    if (m1.sent_idx == m2.sent_idx and m1.span_end > m2.span_start):
        return 1
    else:
        return 0

cdef (int, int, int) get_span_sent(Span span):
    ''' return index (of the sentence), start and end of the sentence of a Span in its Doc'''
    cdef:
        int n = 0
        int i
        const TokenC* root = &span.doc.c[span.start]
    while root.head != 0: # find left edge
        root += root.head
        n += 1
        if n >= span.doc.length:
            raise RuntimeError("Error while getting Mention sentence index. Infinite loop detected.")
    n = 0
    for i in range(root.l_edge+1):
        if span.doc.c[i].sent_start == 1:
            n += 1
    return n, root.l_edge, root.r_edge + 1

cdef hash_t get_span_entity_label(Span span):
    ''' Label of a detected named entity the Mention is nested in if any'''
    cdef int i
    cdef const TokenC* token
    cdef hash_t label
    cdef bint has_label = False
    for i in range(span.start, span.end):
        token = &span.doc.c[i]
        if token.ent_iob == 1: # Inside
            if not has_label:
                label = token.ent_id
                has_label = True
        elif token.ent_iob == 2 or token.ent_iob == 0: # Outside
            return -1 # Not nested in entity
        elif token.ent_iob == 3: # Beggining
            if has_label:
                return -1 # Not nested in entity
            has_label = True
            label = token.ent_id
    return label

cdef get_span_type(Span span):
    ''' Find the type of a Span '''
    if any(t.tag_ in CONJ_TAGS and t.ent_type_ not in ACCEPTED_ENTS for t in span):
        mention_type = MENTION_TYPE["LIST"]
    elif span.root.tag_ in PRP_TAGS:
        mention_type = MENTION_TYPE["PRONOMINAL"]
    elif span.root.ent_type_ in ACCEPTED_ENTS or span.root.tag_ in PROPER_TAGS:
        mention_type = MENTION_TYPE["PROPER"]
    else:
        mention_type = MENTION_TYPE["NOMINAL"]
    return mention_type

def get_resolved(doc, coreferences):
    ''' Return a list of utterrances text where the coref are resolved to the most representative mention'''
    resolved = list(tok.text_with_ws for tok in doc)
    for main, cluster in coreferences.items():
        for coref in cluster:
            if coref != main:
                resolved[coref.start] = main.text + doc[coref.end-1].whitespace_
                for i in range(coref.start+1, coref.end):
                    resolved[i] = ""
    return ''.join(resolved)

#########################
## MENTION EXTRACTION ###
#########################
# Utility to remove bad endings
cdef (int, int) enlarge_span(TokenC* doc_c, int i, int sent_start, int sent_end, int test,
                  HashesList hashes, StringStore store, bint debug=False):
    cdef int j
    cdef uint32_t minchild_idx
    cdef uint32_t maxchild_idx
    # if debug: print("⚜️ Enlarge span")
    minchild_idx = i
    maxchild_idx = i
    for j in range(sent_start, sent_end):
        c = doc_c[j]
        c_head = j + c.head
        # if debug: print("minchild c & c.head:", store[doc_c[j].lex.lower], store[doc_c[c_head].lex.lower])
        if c_head != i:
            continue
        if c.l_edge >= minchild_idx:
            continue
        if test == 0 \
                or (test == 1 and inside(c.dep, hashes.nsubj_or_dep)) \
                or (test == 2 and c.head == i and not inside(c.dep, hashes.conj_or_prep)):
            minchild_idx = c.l_edge
            # if debug: print("keep as minchild", store[doc_c[minchild_idx].lex.lower])
    for j in range(sent_start, sent_end):
        c = doc_c[j]
        c_head = j + c.head
        # if debug: print("maxchild c & c.head:", store[doc_c[j].lex.lower], store[doc_c[c_head].lex.lower])
        if c_head != i:
            continue
        if c.r_edge <= maxchild_idx:
            continue
        if test == 0 \
                or (test == 1 and inside(c.dep, hashes.nsubj_or_dep)) \
                or (test == 2 and c.head == i and not inside(c.dep, hashes.conj_or_prep)):
            maxchild_idx = c.r_edge
            # if debug: print("keep as maxchild", store[doc_c[maxchild_idx].lex.lower])
    # if debug: print("left side before cleaning:", store[doc_c[minchild_idx].lex.lower])
    # if debug: print("right side before cleaning:", store[doc_c[maxchild_idx].lex.lower])
    # Clean up endings and begginging
    while maxchild_idx >= minchild_idx and (inside(doc_c[maxchild_idx].pos, hashes.remove_pos)
                                        or inside(doc_c[maxchild_idx].lex.lower, hashes.lower_not_end)):
        # PyErr_CheckSignals()
        # if debug: print("Removing last token", store[doc_c[maxchild_idx].lex.lower], store[doc_c[maxchild_idx].tag])
        maxchild_idx -= 1 # We don't want mentions finishing with 's or conjunctions/punctuation
    while minchild_idx <= maxchild_idx and (inside(doc_c[minchild_idx].pos, hashes.remove_pos) 
                                        or inside(doc_c[minchild_idx].lex.lower, hashes.lower_not_end)):
        # PyErr_CheckSignals()
        # if debug: print("Removing first token", store[doc_c[minchild_idx].lex.lower], store[doc_c[minchild_idx].tag])
        minchild_idx += 1 # We don't want mentions starting with 's or conjunctions/punctuation
    # if debug: print("left side after cleaning:", store[doc_c[minchild_idx].lex.lower])
    # if debug: print("right side after cleaning:", store[doc_c[maxchild_idx].lex.lower])
    return minchild_idx, maxchild_idx + 1

cdef bint add_span(int start, int end, SentSpans* mentions_spans, TokenC* doc_c,
              StringStore store, bint debug=False):
    cdef int num = mentions_spans.num
    # if debug: print("🔥 Add span: " + ' '.join(store[doc_c[i].lex.lower] for i in range(start, end)))
    mentions_spans.spans[num].start = start
    mentions_spans.spans[num].end = end
    mentions_spans.num += 1
    # if debug: print("🔥 Add span: " + ' '.join(store[doc_c[i].lex.lower] for i in range(start, end)))
    # if debug: print("🔥 mentions_spans.num: ", mentions_spans.num)
    return mentions_spans.num >= mentions_spans.max_spans

cdef void _extract_from_sent(TokenC* doc_c, int sent_start, int sent_end, SentSpans* mentions_spans,
                        HashesList hashes, StringStore store, bint blacklist=False,
                        bint debug=False):
    ''' Extract Pronouns and Noun phrases mentions from a spacy Span '''
    cdef int i, j, c_head, k, endIdx, minchild_idx, maxchild_idx, n_spans
    cdef bint test
    # if debug: print("😎 Extract sents start, end:", sent_start, sent_end)

    for i in range(sent_start, sent_end):
        # PyErr_CheckSignals()
        token = doc_c[i]
        # if debug: print("🚀 tok:", store[token.lex.lower], "tok.tag:", store[token.tag], "tok.pos:", store[token.pos], "tok.dep:", store[token.dep])
        if blacklist and inside(token.lex.lower, hashes.no_coref_list):
            # if debug: print("token in no_coref_list")
            continue
        if (not inside(token.tag, hashes.keep_tags) or inside(token.dep, hashes.leave_dep) \
            and not inside(token.dep, hashes.keep_dep)):
            # if debug: print("not pronoun or no right dependency")
            continue
        # pronoun
        if inside(token.tag, hashes.PRP_tags): #re.match(r"PRP.*", token.tag_):
            # if debug: print("PRP")
            endIdx = i + 1
            #span = doc_c[i: endIdx]
            ## if debug: print("==-- PRP store:", span)
            test = add_span(i, i+1, mentions_spans, doc_c, store)
            if test: return
            # when pronoun is a part of conjunction (e.g., you and I)
            if token.r_kids > 0 or token.l_kids > 0:
                #span = doc[token.l_edge : token.r_edge+1]
                ## if debug: print("==-- in conj store:", span)
                test = add_span(token.l_edge, token.r_edge+1, mentions_spans, doc_c, store)
                if test: return
            continue
        # Add NP mention
        # if debug: print("NP or IN:", store[token.lex.lower])
        # Take care of 's
        if token.lex.lower == hashes.POSSESSIVE_MARK:
            # if debug: print("'s detected")
            c_head = i + token.head
            j = 0
            while c_head != 0 and j < MAX_ITER:
                # if debug: print("token head:", c_head, doc_c[c_head].dep, "head:", c_head + doc_c[c_head].head)
                if doc_c[c_head].dep == hashes.NSUBJ_MARK:
                    start, end = enlarge_span(doc_c, c_head, sent_start, sent_end, 1, hashes, store)
                    # if debug: print("'s', i1:", store[doc_c[start].lex.lower], " i2:", store[doc_c[end].lex.lower])
                    ## if debug: print("==-- 's' store:", span)
                    test = add_span(start, end+1, mentions_spans, doc_c, store)
                    if test: return
                    break
                c_head += doc_c[c_head].head
                j += 1
            continue

        for j in range(sent_start, sent_end):
            c = doc_c[j]
            #if debug and j + c.head == i: print("🚧 token in span:", store[c.lex.lower])#, "- head & dep:", c.head, c.dep)
        start, end = enlarge_span(doc_c, i, sent_start, sent_end, 0, hashes, store)
        if token.tag == hashes.IN_TAG and token.dep == hashes.MARK_DEP and start == end:
            start, end = enlarge_span(doc_c, i + token.head, sent_start, sent_end, 0, hashes, store)
        #if debug:
        #    print("left side:", left)
        #    print("right side:", right)
        #    minchild_idx = min(left) if left else token.i
        #    maxchild_idx = max(right) if right else token.i
        #    print("full span:", doc[minchild_idx:maxchild_idx+1])
        if start == end:
            continue
        if doc_c[start].lex.lower == hashes.POSSESSIVE_MARK:
            continue # we probably already have stored this mention
        #span = doc_c[start:end]
        test = add_span(start, end, mentions_spans, doc_c, store)
        if test: return
        #if debug:
        #    print("cleaned endings span:", doc_c[start:end])
        #    print("==-- full span store:", span)
        # if debug and token.tag == hashes.IN_TAG: print("IN tag")
        test = False
        for tok in doc_c[sent_start:sent_end]:
            if inside(tok.dep, hashes.conj_or_prep):
                test = True
                break
        if test:
            # if debug: print("Conjunction found, storing first element separately")
            #for c in doc:
            #    if c.head.i == i and inside(c.dep, hashes.conj_or_prep):
            #        if debug: print("left no conj:", c, 'dep & edge:', c.dep, c.left_edge)
            #        if debug: print("right no conj:", c, 'dep & edge:', c.dep, c.right_edge)
            #left_no_conj = list(c.left_edge.i for c in doc if c.head.i == i and not inside(c.dep, hashes.conj_or_prep))
            #right_no_conj = list(c.right_edge.i for c in doc if c.head.i == i and not inside(c.dep, hashes.conj_or_prep))
            #if debug: print("left side no conj:", [doc[i] for i in left_no_conj])
            #if debug: print("right side no conj:", [doc[i] for i in right_no_conj])
            start, end = enlarge_span(doc_c, i, sent_start, sent_end, 0, hashes, store)
            if start == end:
                continue
            test = add_span(start, end, mentions_spans, doc_c, store)
            if test: return
            #if debug: print("==-- full span store:", span)
    #if debug: print("mentions_spans inside", mentions_spans)
    return

cdef extract_mentions_spans(Doc doc, HashesList hashes, bint blacklist=False, bint debug=False):
    ''' Extract potential mentions from a spacy parsed Doc '''
    cdef:
        int i, max_spans
        int n_sents
        SpanC spans_c
        int n_spans = 0
        Pool mem = Pool()

    # if debug: print('===== doc ====:', doc)
    # for c in doc:
        # if debug: print("🚧 span search:", c, "head:", c.head, "tag:", c.tag_, "pos:", c.pos_, "dep:", c.dep_)
    # Named entities
    mentions_spans = list(ent for ent in doc.ents if ent.label_ in ACCEPTED_ENTS)

    # Setup for fast scanning
    n_sents = len(list(doc.sents))
    sent_spans = <SentSpans*>mem.alloc(n_sents, sizeof(SentSpans))
    for i, sent in enumerate(doc.sents):
        max_spans = len(sent)*SPAN_FACTOR
        sent_spans[i].spans = <SpanC*>mem.alloc(max_spans, sizeof(SpanC))
        sent_spans[i].max_spans = max_spans
        sent_spans[i].num = 0

    # if debug: print("==-- ents:", list(((ent, ent.label_) for ent in mentions_spans)))
    for i, sent in enumerate(doc.sents):
        _extract_from_sent(doc.c, sent.start, sent.end, &sent_spans[i],
                           hashes, doc.vocab.strings,
                           blacklist=blacklist)
    #for spans in parallel_process([{'span': sent,
    #                                'blacklist': blacklist} for sent in doc.sents],
    #                            _extract_from_sent, use_kwargs=True, n_jobs=4, front_num=0):
    #    mentions_spans = mentions_spans + spans
    spans_set = set()
    for m in mentions_spans:
        if m.end > m.start and (m.start, m.end) not in spans_set:
            spans_set.add((m.start, m.end))
            n_spans += 1
    for i in range(n_sents):
        for j in range(sent_spans[i].num):
            spans_c = sent_spans[i].spans[j]
            if spans_c.end > spans_c.start and (spans_c.start, spans_c.end) not in spans_set:
                spans_set.add((spans_c.start, spans_c.end))
                n_spans += 1
    sorted_spans = sorted(spans_set)
    cleaned_mentions_spans = [doc[s[0]:s[1]] for s in sorted_spans]
    # if debug: print("cleaned_mentions_spans", cleaned_mentions_spans)
    return cleaned_mentions_spans, n_spans

#######################
###### CLASSES ########

cdef class Model(object):
    ''' Coreference neural net model '''
    def __cinit__(self, model_path):
        self.n_layers = 0

    def __init__(self, model_path):
        self.s_weights, self.s_biases, self.p_weights, self.p_biases = [], [], [], []
        for file in sorted(os.listdir(model_path)):
            if not file.endswith('.npy'):
                continue
            w = numpy.load(os.path.join(model_path, file)).astype(dtype='float32')
            if file.startswith("single_mention_weights"):
                self.s_weights.append(w)
            if file.startswith("single_mention_bias"):
                self.s_biases.append(w)
            if file.startswith("pair_mentions_weights"):
                self.p_weights.append(w)
            if file.startswith("pair_mentions_bias"):
                self.p_biases.append(w)
        assert len(self.s_weights) == len(self.s_biases) == len(self.p_weights) == len(self.p_biases)
        self.n_layers = len(self.s_weights)

    cpdef float [:] get_score(self, float [:,:] features, bint single):
        cdef:
            int i
            float [:] out
            float [:, :] weights, bias
        for i in range(self.n_layers):
            weights = self.s_weights[i] if single else self.p_weights[i]
            bias = self.s_biases[i] if single else self.p_biases[i]
            features = numpy.matmul(weights, features) + bias
            if weights.shape[0] > 1:
                features = numpy.maximum(features, 0) # ReLU
        out = numpy.sum(features, axis=0)
        return out

cdef class EmbeddingExtractor(object):
    ''' Compute words embedding features for mentions '''
    def __cinit__(self, pretrained_model_path, Vocab vocab, conv_dict=None):
        # First add our strings in the vocab to get hashes
        self.unknown_word = vocab[UNKNOWN_WORD].orth
        self.missing_word = vocab[MISSING_WORD].orth
        self.digit_word = vocab[u"0"].orth
        self.conv_dict = None

    def __init__(self, pretrained_model_path, Vocab vocab, conv_dict=None):
        keys, mat = self.load_embeddings_from_file(pretrained_model_path + 'static_word', vocab.strings)
        self.static = Vectors(shape=mat.shape, data=mat, keys=keys, name='coref_static')
        keys, mat = self.load_embeddings_from_file(pretrained_model_path + 'tuned_word', vocab.strings)
        self.tuned = Vectors(shape=mat.shape, data=mat, keys=keys, name='coref_tuned')
        assert self.static.shape[1] == self.tuned.shape[1] == SIZE_EMBEDDING
        self.shape = self.static.shape[1]
        self.fallback = self.static[self.unknown_word]
        if conv_dict is not None:
            self.conv_dict = Vectors()
            for key, words in conv_dict.items():
                norm_k = self.normalize(vocab.get(key, vocab.mem))
                norm_w = list(self.normalize(vocab.get(w, vocab.mem)) for w in words)
                vect = self.average_list(norm_w)
                self.conv_dict.add(norm_k, vector=vect)

    @staticmethod
    def load_embeddings_from_file(name, StringStore store):
        # print("Loading embeddings from", name)
        keys = []
        mat = numpy.load(name+"_embeddings.npy").astype(dtype='float32')
        with io.open(name+"_vocabulary.txt", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                keys.append(store.add(line.strip()))
        return keys, mat

    def average_list(self, hash_list):
        ''' Embed a list of words '''
        cdef int i
        embed_vector = numpy.zeros(self.shape, dtype='float32') #We could also use numpy.copy(self.average_mean)
        for hash_w in hash_list:
            embed = self.tuned[hash_w] if hash_w in self.tuned else self.get_static(hash_w)
            embed_vector = embed_vector + embed
        return embed_vector/max(len(hash_list), 1)

    cdef hash_t normalize(self, const LexemeC* c):
        if c is NULL:
            return self.missing_word
        if Lexeme.c_check_flag(c, IS_DIGIT):
            return self.digit_word
        return c.lower

    cdef float [:] get_static(self, hash_t word):
        return self.static[word] if word in self.static else self.fallback

    cdef float [:] get_word_embedding(self, const LexemeC* c, bint static=False):
        ''' Embedding for a single word hash (tuned if possible, otherwise static) '''
        hash_w = self.normalize(c)
        if self.conv_dict is not None:
            word = self.conv_dict.get(hash_w, None)
            if word is not None:
                return word
        if static:
            return self.get_static(hash_w)
        else:
            if hash_w in self.tuned:
                return self.tuned[hash_w]
            else:
                return self.get_static(hash_w)
 
    cdef float [:] get_word_in_sentence(self, int word_idx, TokenC* doc, int sent_start, int sent_end):
        ''' Embedding for a word hash in a sentence '''
        if word_idx < sent_start or word_idx >= sent_end:
            return self.get_word_embedding(NULL)
        return self.get_word_embedding(doc[word_idx].lex)

    cdef float [:] get_average_embedding(self, TokenC* doc, int start, int end, Hashes puncts, StringStore strings):
        ''' Embedding for a list of word hashes '''
        cdef:
            int i
            int n = 0
            float [:] embed_vector, embed
        embed_arr = numpy.zeros(self.shape, dtype='float32') #We could also use numpy.copy(self.average_mean)
        for i in range(start, end):
            t_lex = doc[i].lex
            if not inside(t_lex.lower, puncts):
                n += 1
                embed = self.get_word_embedding(t_lex, static=True)
                embed_arr = embed_arr + embed
        embed_vector = numpy.divide(embed_arr, float(max(n, 1)))
        return embed_vector

    cdef float [:] get_mention_embeddings(self, TokenC* doc, Mention_C m, Hashes puncts, StringStore strings, float [:] doc_embedding):
        ''' Get span (averaged) and word (single) embeddings of a mention '''
        cdef:
            float [:] embed
            int head = m.span_root + doc[m.span_root].head
        # doc_embedding = 0
        embeddings = numpy.zeros((EMBED_13, ), dtype='float32')
        embed = embeddings
        embed[:EMBED_01]         = self.get_average_embedding(doc, m.span_start, m.span_end, puncts, strings)
        embed[EMBED_01:EMBED_02] = self.get_average_embedding(doc, max(m.span_start-5, m.sent_start), m.span_start, puncts, strings)
        embed[EMBED_02:EMBED_03] = self.get_average_embedding(doc, m.span_end, min(m.span_end + 5, m.sent_end), puncts, strings)
        embed[EMBED_03:EMBED_04] = self.get_average_embedding(doc, m.sent_start, m.sent_end, puncts, strings)
        embed[EMBED_04:EMBED_05] = doc_embedding
        embed[EMBED_05:EMBED_06] = self.get_word_embedding(doc[m.span_root].lex)
        embed[EMBED_06:EMBED_07] = self.get_word_embedding(doc[m.span_start].lex)
        embed[EMBED_07:EMBED_08] = self.get_word_embedding(doc[m.span_end-1].lex)
        embed[EMBED_08:EMBED_09] = self.get_word_in_sentence(m.span_start-1, doc, m.sent_start, m.sent_end)
        embed[EMBED_09:EMBED_10] = self.get_word_in_sentence(m.span_end, doc, m.sent_start, m.sent_end)
        embed[EMBED_10:EMBED_11] = self.get_word_in_sentence(m.span_start-2, doc, m.sent_start, m.sent_end)
        embed[EMBED_11:EMBED_12] = self.get_word_in_sentence(m.span_end+1, doc, m.sent_start, m.sent_end)
        embed[EMBED_12:        ] = self.get_word_embedding(doc[head].lex)
        return embed


cdef class CorefComponent(object):
    """spaCy v2.0 Coref pipeline component. """
    def __cinit__(self, nlp, label='coref',
                  greedyness=0.5, max_dist=50, max_dist_match=500,
                  conv_dict=None, blacklist=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.blacklist = blacklist
        self.coref_model = None
        self.embed_extractor = None
        self.name = None
        self.label = None
        self.mem = Pool() # We use this for the allocations shared between all the docs (mainly the hashes)
        self.hashes = get_hash_lookups(nlp.vocab.strings, self.mem)

    def __init__(self, nlp, label='coref', greedyness=0.5, max_dist=50,
                 max_dist_match=500, conv_dict=None, blacklist=False):
        """Initialise the pipeline component.
        """
        self.name = 'coref' # component name, will show up in the pipeline
        self.label = nlp.vocab.strings[label]  # get entity label ID
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        # print("Loading neuralcoref model from", model_path)
        self.coref_model = Model(model_path)
        self.embed_extractor = EmbeddingExtractor(model_path, nlp.vocab, conv_dict)

        # Register attributes on Doc and Span
        Doc.set_extension('has_coref', default=False)
        Doc.set_extension('coref_mentions', default=None)
        Doc.set_extension('coref_clusters', default=None)
        Doc.set_extension('coref_resolved', default="")
        Span.set_extension('is_coref', default=False)
        Span.set_extension('coref_cluster', default=None)
        Span.set_extension('coref_main', default=None)

    def __call__(self, doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.
        """
        self.build_clusters(doc)
        return doc  # don't forget to return the Doc!

    #######################################
    ###### FEATURE BUILDING FUNCTIONS #####
    #######################################

    cdef build_clusters(self, Doc doc):
        ''' Build coreference clusters '''
        cdef:
            Mention_C* c
            Mention_C m1, m2
            TokenC* doc_c
            uint64_t i, ant_idx, men_idx, b_idx, n_mentions, n_pairs
            uint64_t [:] p_ant, p_men, best_ant
            float [:] embed, feats, doc_embed
            float [:, :] s_inp, p_inp
            float [:] score, best_score
            Pool mem = Pool() # We use this for doc specific allocation

            timespec ts
            double curr_t0, curr_t1, curr_t2, curr_t3, curr_t4, curr_t5, curr_t6

        # print("Build coreference clusters")
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t0 = ts.tv_sec + (ts.tv_nsec / 1000000000.)
        strings = doc.vocab.strings

        # print("Extract mentions")
        # ''' Extract mentions '''
        mentions, n_mentions = extract_mentions_spans(doc, self.hashes, blacklist=self.blacklist)
        n_sents = len(list(doc.sents))
        mentions = sorted((m for m in mentions), key=lambda m: (m.root.i, m.start))
        c = <Mention_C*>mem.alloc(n_mentions, sizeof(Mention_C))
        content_words = []
        for i, m in enumerate(mentions):
            c[i].entity_label = get_span_entity_label(m)
            c[i].span_start = m.start
            c[i].span_end = m.end
            c[i].span_root = m.root.i
            idx, sent_start, sent_end = get_span_sent(m)
            c[i].sent_idx = idx
            c[i].sent_start = sent_start
            c[i].sent_end = sent_end
            c[i].mention_type = get_span_type(m)
            c[i].root_lower = (<Token>m.root).c.lex.lower
            c[i].span_lower = strings.add(m.text.lower())
            content_words.append(set(tok.lower_ for tok in m if tok.tag_ in CONTENT_TAGS))
            c[i].content_words.length = len(content_words[-1])
            c[i].content_words.arr = <hash_t*>mem.alloc(len(content_words[-1]), sizeof(hash_t))
            for j, w in enumerate(content_words[-1]):
                c[i].content_words.arr[j] = strings.add(w)

        # print("Prepare arrays")
        # ''' Prepare arrays of pairs indices and features for feeding the model '''
        # pairs_ant, pairs_mem, n_pairs = self.get_pairs(n_mentions, content_words)
        pairs_ant = []
        pairs_men = []
        n_pairs = 0
        if self.max_dist_match is not None:
            word_to_mentions = {}
            for i in range(n_mentions):
                for tok in content_words[i]:
                    if not tok in word_to_mentions:
                        word_to_mentions[tok] = [i]
                    else:
                        word_to_mentions[tok].append(i)
        for i in range(n_mentions):
            if self.max_dist is None:
                antecedents = set(range(<object>i))
            else:
                antecedents = set(range(max(0, <object>i - self.max_dist), <object>i))
            if self.max_dist_match is not None:
                for tok in content_words[i]:
                    with_string_match = word_to_mentions.get(tok, None)
                    for match_idx in with_string_match:
                        if match_idx < i and match_idx >= i - self.max_dist_match:
                            antecedents.add(match_idx)
            pairs_ant += list(antecedents)
            pairs_men += [i]*len(antecedents)
            n_pairs += len(antecedents)
        p_ant_arr = numpy.asarray(pairs_ant, dtype=numpy.uint64)
        p_men_arr = numpy.asarray(pairs_men, dtype=numpy.uint64) #Should probably update these to uint64 but then all index
        p_ant = p_ant_arr
        p_men = p_men_arr
        s_inp_arr = numpy.zeros((SIZE_SNGL_IN_NO_GENRE + SIZE_GENRE, n_mentions), dtype='float32')
        p_inp_arr = numpy.zeros((SIZE_PAIR_IN_NO_GENRE + SIZE_GENRE, n_pairs), dtype='float32')
        s_inp = s_inp_arr
        p_inp = p_inp_arr
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t1 = ts.tv_sec + (ts.tv_nsec / 1000000000.) - curr_t0
        #print("n_mentions", n_mentions)
        #print("mentions", mentions)

        # ''' Build single features and pair features arrays '''
        # print("Build single features")
        doc_c = doc.c
        doc_embedding = numpy.zeros(SIZE_EMBEDDING, dtype='float32')# self.embed_extractor.get_average_embedding(doc.c, 0, doc.length + 1, self.hashes.puncts)
        doc_embed = doc_embedding
        # print("Build mentions features")
        for i in range(n_mentions):
            #print("Get embed for mention", i)
            # struct_print(c[i], doc.vocab.strings)
            embeddings = (<EmbeddingExtractor>self.embed_extractor).get_mention_embeddings(doc.c, c[i], self.hashes.puncts, doc.vocab.strings, doc_embed)
            embed = embeddings
            # print("Prepare array for mention", i)
            s_inp[:SGNL_FEATS_0, i] = embed
            s_inp[SGNL_FEATS_0 + c[i].mention_type, i] = 1 # 01_MentionType
            b_idx, val = index_distance(c[i].span_end - c[i].span_start - 1) # 02_MentionLength
            s_inp[SGNL_FEATS_1 + b_idx, i] = 1
            s_inp[SGNL_FEATS_2, i] = val
            val = float(i)/float(n_mentions) # 03_MentionNormLocation
            s_inp[SGNL_FEATS_3, i] = val
            s_inp[SGNL_FEATS_4, i] = is_nested(c, n_mentions, i) # 04_IsMentionNested
            #print('features:', numpy.array(s_inp[SGNL_FEATS_0:SGNL_FEATS_1]), "\n",
            #      numpy.array(s_inp[SGNL_FEATS_1:SGNL_FEATS_2]), "\n",
            #      numpy.array(s_inp[SGNL_FEATS_2:SGNL_FEATS_3]), "\n",
            #      numpy.array(s_inp[SGNL_FEATS_3:SGNL_FEATS_4]), "\n",
            #      numpy.array(s_inp[SGNL_FEATS_4:SGNL_FEATS_5]))
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t2 = ts.tv_sec + (ts.tv_nsec / 1000000000.) - curr_t0
        # print("Build pair features")
        #this = 0
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            #if ant_idx == 1 and men_idx == 6:
            #    this = i
            #    print("this", i)
            m1 = c[ant_idx]
            m2 = c[men_idx]
            p_inp[:PAIR_FEATS_0, i] = s_inp[:SGNL_FEATS_0, ant_idx]
            p_inp[PAIR_FEATS_0:PAIR_FEATS_1, i] = s_inp[:SGNL_FEATS_0, men_idx]
            p_inp[PAIR_FEATS_1, i] = 1       # 00_SameSpeaker
            # p_inp[PAIR_FEATS_1 + 1, i] = 0 # 01_AntMatchMentionSpeaker # arrays are initialized to zero
            # p_inp[PAIR_FEATS_1 + 2, i] = 0 # 02_MentionMatchSpeaker
            p_inp[PAIR_FEATS_1 + 3, i] = heads_agree(m1, m2) # 03_HeadsAgree
            p_inp[PAIR_FEATS_1 + 4, i] = exact_match(m1, m2) # 04_ExactStringMatch
            p_inp[PAIR_FEATS_1 + 5, i] = relaxed_match(m1, m2) # 05_RelaxedStringMatch
            b_idx, val = index_distance(m2.sent_idx - m1.sent_idx) # 06_SentenceDistance
            # p_inp[PAIR_FEATS_2:PAIR_FEATS_3, i] = 0
            p_inp[PAIR_FEATS_2 + b_idx, i] = 1
            p_inp[PAIR_FEATS_3, i] = val
            b_idx, val = index_distance(men_idx - ant_idx - 1) # 07_MentionDistance
            # p_inp[PAIR_FEATS_4:PAIR_FEATS_5, i] = 0
            p_inp[PAIR_FEATS_4 + b_idx, i] = 1
            p_inp[PAIR_FEATS_5, i] = val
            p_inp[PAIR_FEATS_5 + 1, i] = overlapping(m1, m2) # 08_Overlapping
            p_inp[PAIR_FEATS_6:PAIR_FEATS_7, i] = s_inp[SGNL_FEATS_0:SGNL_FEATS_5, ant_idx] # 09_M1Features
            p_inp[PAIR_FEATS_7:PAIR_FEATS_8, i] = s_inp[SGNL_FEATS_0:SGNL_FEATS_5, men_idx] # 10_M2Features
            # 11_DocGenre is zero currently
        # numpy.save('61_dump', numpy.array(p_inp[:, this]))
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t3 = ts.tv_sec + (ts.tv_nsec / 1000000000.) - curr_t0

        # print("Compute scores")
        # ''' Compute scores '''
        best_score_ar = numpy.empty((n_mentions), dtype='float32')
        best_ant_ar = numpy.empty((n_mentions), dtype=numpy.uint64)
        best_score = best_score_ar
        best_ant = best_ant_ar
        score = self.coref_model.get_score(s_inp, single=True)
        # print('single score:', numpy.array(score))
        for i in range(n_mentions):
            best_score[i] = score[i] - 50 * (self.greedyness - 0.5)
            best_ant[i] = i
        score = self.coref_model.get_score(p_inp, single=False)
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            # print("score for", ant_idx, men_idx, score[i])
            if score[i] > best_score[men_idx]:
                best_score[men_idx] = score[i]
                best_ant[men_idx] = ant_idx
        # print('pair score:', numpy.array(score))
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t4 = ts.tv_sec + (ts.tv_nsec / 1000000000.) - curr_t0

        # for i in range(n_pairs):
        #    ant_idx = p_ant[i]
        #    men_idx = p_men[i]
        #    print("pair features for", ant_idx, men_idx, numpy.array(p_inp[PAIR_FEATS_1:, i]))

        # print("Build clusters")
        # ''' Build clusters '''
        mention_to_cluster = list(range(n_mentions))
        cluster_to_main = list(range(n_mentions))
        clusters = dict((i, [i]) for i in mention_to_cluster)
        for mention_idx, ant_idx in enumerate(best_ant):
            if ant_idx != mention_idx:
                if mention_to_cluster[ant_idx] == mention_to_cluster[mention_idx]:
                    continue
                keep_id = mention_to_cluster[ant_idx]
                remove_id = mention_to_cluster[mention_idx]
                for idx in clusters[remove_id]:
                    mention_to_cluster[idx] = keep_id
                    clusters[keep_id].append(idx)
                del clusters[remove_id]
                if c[ant_idx].mention_type != c[mention_idx].mention_type:
                    if c[mention_idx].mention_type == MENTION_TYPE["PROPER"] \
                        or (c[mention_idx].mention_type == MENTION_TYPE["NOMINAL"] and
                                c[ant_idx].mention_type == MENTION_TYPE["PRONOMINAL"]):
                        cluster_to_main[ant_idx] = mention_idx
        remove_id = []
        main = list(mentions)
        mentions_list = []
        clusters_list = {}
        for key, m_idx_list in clusters.items():
            if len(m_idx_list) != 1:
                m_list = list(mentions[i] for i in m_idx_list)
                main = mentions[cluster_to_main[key]]
                mentions_list += m_list
                clusters_list[main] = m_list
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t5 = ts.tv_sec + (ts.tv_nsec / 1000000000.) - curr_t0

        # print("Update doc")
        # ''' Update doc '''
        if len(clusters) != 0:
            doc._.set('has_coref', True)
            doc._.set('coref_mentions', mentions_list)
            doc._.set('coref_clusters', clusters_list)
            doc._.set('coref_resolved', get_resolved(doc, clusters_list))
            for main, m_list in clusters_list.items():
                for mention in m_list:
                    mention._.set('is_coref', True)
                    mention._.set('coref_cluster', m_list)
                    mention._.set('coref_main', main)
        clock_gettime(CLOCK_REALTIME, &ts)
        curr_t6 = ts.tv_sec + (ts.tv_nsec / 1000000000.) - curr_t0
        print("mentions", mentions)
        print("timing",
              "\nExtract mentions and Prepare arrays", <object>curr_t1,
              "\nBuild single features", <object>curr_t2,
              "\nBuild pair features", <object>curr_t3,
              "\nCompute scores", <object>curr_t4,
              "\nBuild clusters", <object>curr_t5,
              "\nUpdate doc", <object>curr_t6)
        return doc

cdef struct_print(Mention_C mention, StringStore strings):
    print("== span_lower", <object>strings[mention.span_lower])
    print("entity_label", <object>mention.entity_label)
    print("span_root", <object>mention.span_root)
    print("span_start", <object>mention.span_start)
    print("span_end", <object>mention.span_end)
    print("sent_idx", <object>mention.sent_idx)
    print("sent_start", <object>mention.sent_start)
    print("sent_end", <object>mention.sent_end)
    print("mention_type", <object>mention.mention_type)
    print("root_lower", <object>strings[mention.root_lower])
    print("- content_words --")
