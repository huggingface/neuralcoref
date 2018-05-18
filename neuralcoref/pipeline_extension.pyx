# coding: utf8
# cython: profile=True
# cython: infer_types=True
# distutils: language=c++
"""Coref resolution spaCy v2.0 pipeline component 
Custom pipeline components: https://spacy.io//usage/processing-pipelines#custom-components
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function

import plac
import re
import os
import io
from cpython cimport array
import array
from libc.stdint cimport uint16_t, uint32_t, uint64_t, uintptr_t, int32_t

import spacy
from spacy.typedefs cimport hash_t
from cymem.cymem cimport Pool
from cpython.exc cimport PyErr_CheckSignals

from spacy.lang.en import English
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token
from spacy.tokens.token cimport Token
from spacy.strings cimport StringStore
cimport numpy as np
np.import_array()
import numpy

from neuralcoref.utils import PACKAGE_DIRECTORY, encode_distance
from neuralcoref.compat import unicode_

#######################
##### UTILITIES #######
DEF SIZE_SPAN = 250 # size of the span vector (averaged word embeddings)
DEF SIZE_WORD = 8 # number of words in a mention (tuned embeddings)
DEF SIZE_EMBEDDING = 50 # size of the words embeddings
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

DEF MAX_BINS = 9
DEF MAX_FOLLOW_UP = 50

DEF PAIR_FEATS_0 = SIZE_MENTION_EMBEDDING
DEF PAIR_FEATS_1 = 2*SIZE_MENTION_EMBEDDING
DEF PAIR_FEATS_2 = PAIR_FEATS_1+6
DEF PAIR_FEATS_3 = PAIR_FEATS_2 + MAX_BINS + 1
DEF PAIR_FEATS_4 = PAIR_FEATS_3 + 1
DEF PAIR_FEATS_5 = PAIR_FEATS_4 + MAX_BINS + 1
DEF PAIR_FEATS_6 = PAIR_FEATS_5 + 2
DEF PAIR_FEATS_7 = PAIR_FEATS_6 + SIZE_SNGL_FEATS
DEF PAIR_FEATS_8 = PAIR_FEATS_7 + SIZE_SNGL_FEATS

DEF SGNL_FEATS_0 = SIZE_MENTION_EMBEDDING
DEF SGNL_FEATS_1 = SGNL_FEATS_0 + SIZE_SNGL_FEATS

DISTANCE_BINS_PY = array.array('i', list(range(5)) + [5]*3 + [6]*8 + [7]*16 + [8]*32)

cdef:
    int[:] DISTANCE_BINS = DISTANCE_BINS_PY
    float BINS_NUM = float(len(DISTANCE_BINS))

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
    # In this cython C function, we take the simpler approach of directly comparing the roots hashes
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

cdef get_span_sent_number(Span span):
    ''' Index of the sentence of a Span in its Doc'''
    cdef int n = 0
    cdef int i
    cdef const TokenC* root = &span.doc.c[span.start]
    while root.head != 0: # find left edge
        root += root.head
        n += 1
        if n >= span.doc.length:
            raise RuntimeError("Error while getting Mention sentence index")
    n = -1
    for i in range(root.l_edge+1):
        if span.doc.c[i].sent_start == 1:
            n += 1
    return n

cdef get_span_type(Span span):
    ''' Find the type of a Span '''
    conj = ["CC", ","]
    prp = ["PRP", "PRP$"]
    proper = ["NNP", "NNPS"]
    if any(t.tag_ in conj and t.ent_type_ not in ACCEPTED_ENTS for t in span):
        mention_type = MENTION_TYPE["LIST"]
    elif span.root.tag_ in prp:
        mention_type = MENTION_TYPE["PRONOMINAL"]
    elif span.root.ent_type_ in ACCEPTED_ENTS or span.root.tag_ in proper:
        mention_type = MENTION_TYPE["PROPER"]
    else:
        mention_type = MENTION_TYPE["NOMINAL"]
    return mention_type

cdef get_span_entity_label(Span span):
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

def get_resolved(doc, coreferences):
    ''' Return a list of utterrances text where the coref are resolved to the most representative mention'''
    resolved = ""
    in_coref = None
    for tok in doc:
        if in_coref is None:
            for coref_original, coref_replace in coreferences:
                if coref_original[0] == tok:
                    in_coref = coref_original
                    resolved += coref_replace.span.lower_
                    break
            if in_coref is None:
                resolved += tok.text_with_ws
        if in_coref is not None and tok == in_coref[-1]:
            resolved += ' ' if tok.whitespace_ and resolved[-1] is not ' ' else ''
            in_coref = None
    return resolved

#########################
## MENTION EXTRACTION ###
#########################
MENTION_TYPE = {"PRONOMINAL": 0, "NOMINAL": 1, "PROPER": 2, "LIST": 3}
MENTION_LABEL = {0: "PRONOMINAL", 1: "NOMINAL", 2: "PROPER", 3: "LIST"}
NO_COREF_LIST = ["i", "me", "my", "you", "your"]
KEEP_TAGS = ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "DT", "IN"]
CONTENT_TAGS = ["NN", "NNS", "NNP", "NNPS"]
PRP_TAGS = ["PRP", "PRP$"]
NSUBJ_OR_DEP = ["nsubj", "dep"]
CONJ_OR_PREP = ["conj", "prep"]
LEAVE_DEP = ["det", "compound", "appos"]
KEEP_DEP = ["nsubj", "dobj", "iobj", "pobj"]
REMOVE_POS = ["CCONJ", "INTJ", "ADP"]
LOWER_NOT_END = ["'s", ',', '.', '!', '?', ':', ';']
ACCEPTED_ENTS = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"]
WHITESPACE_PATTERN = r"\s+|_+"
UNKNOWN_WORD = "*UNK*"
MISSING_WORD = "<missing>"
DEF MAX_ITER = 100
DEF SPAN_FACTOR = 4

cdef HashesList get_hash_lookups(StringStore store, Pool mem):
    cdef HashesList hashes
    hashes.no_coref_list.length = len(NO_COREF_LIST)
    hashes.no_coref_list.arr = <hash_t*>mem.alloc(hashes.no_coref_list.length, sizeof(hash_t))
    for i, st in enumerate(NO_COREF_LIST):
        hashes.no_coref_list.arr[i] = store.add(st)
    hashes.keep_tags.length = len(KEEP_TAGS)
    hashes.keep_tags.arr = <hash_t*>mem.alloc(hashes.keep_tags.length, sizeof(hash_t))
    for i, st in enumerate(KEEP_TAGS):
        hashes.keep_tags.arr[i] = store.add(st)
    hashes.PRP_tags.length = len(PRP_TAGS)
    hashes.PRP_tags.arr = <hash_t*>mem.alloc(hashes.PRP_tags.length, sizeof(hash_t))
    for i, st in enumerate(PRP_TAGS):
        hashes.PRP_tags.arr[i] = store.add(st)
    hashes.leave_dep.length = len(LEAVE_DEP)
    hashes.leave_dep.arr = <hash_t*>mem.alloc(hashes.leave_dep.length, sizeof(hash_t))
    for i, st in enumerate(LEAVE_DEP):
        hashes.leave_dep.arr[i] = store.add(st)
    hashes.keep_dep.length = len(KEEP_DEP)
    hashes.keep_dep.arr = <hash_t*>mem.alloc(hashes.keep_dep.length, sizeof(hash_t))
    for i, st in enumerate(KEEP_DEP):
        hashes.keep_dep.arr[i] = store.add(st)
    hashes.nsubj_or_dep.length = len(NSUBJ_OR_DEP)
    hashes.nsubj_or_dep.arr = <hash_t*>mem.alloc(hashes.nsubj_or_dep.length, sizeof(hash_t))
    for i, st in enumerate(NSUBJ_OR_DEP):
        hashes.nsubj_or_dep.arr[i] = store.add(st)
    hashes.conj_or_prep.length = len(CONJ_OR_PREP)
    hashes.conj_or_prep.arr = <hash_t*>mem.alloc(hashes.conj_or_prep.length, sizeof(hash_t))
    for i, st in enumerate(CONJ_OR_PREP):
        hashes.conj_or_prep.arr[i] = store.add(st)
    hashes.remove_pos.length = len(REMOVE_POS)
    hashes.remove_pos.arr = <hash_t*>mem.alloc(hashes.remove_pos.length, sizeof(hash_t))
    for i, st in enumerate(REMOVE_POS):
        hashes.remove_pos.arr[i] = store.add(st)
    hashes.lower_not_end.length = len(LOWER_NOT_END)
    hashes.lower_not_end.arr = <hash_t*>mem.alloc(hashes.lower_not_end.length, sizeof(hash_t))
    for i, st in enumerate(LOWER_NOT_END):
        hashes.lower_not_end.arr[i] = store.add(st)
    hashes.POSSESSIVE_MARK = store.add("'s")
    hashes.NSUBJ_MARK = store.add("nsubj")
    hashes.IN_TAG = store.add('IN')
    hashes.MARK_DEP = store.add("mark")
    return hashes

# Utility to remove bad endings
cdef enlarge_span(TokenC* doc_c, int i, int sent_start, int sent_end, int test,
                  HashesList hashes, StringStore store, bint debug=False):
    cdef int j
    cdef uint32_t minchild_idx
    cdef uint32_t maxchild_idx
    if debug: print("⚜️ Enlarge span")
    minchild_idx = i
    maxchild_idx = i
    for j in range(sent_start, sent_end):
        c = doc_c[j]
        c_head = j + c.head
        if debug: print("minchild c & c.head:", store[doc_c[j].lex.lower], store[doc_c[c_head].lex.lower])
        if c_head != i:
            continue
        if c.l_edge >= minchild_idx:
            continue
        if test == 0 \
                or (test == 1 and inside(c.dep, hashes.nsubj_or_dep)) \
                or (test == 2 and c.head == i and not inside(c.dep, hashes.conj_or_prep)):
            minchild_idx = c.l_edge
            if debug: print("keep as minchild", store[doc_c[minchild_idx].lex.lower])
    for j in range(sent_start, sent_end):
        c = doc_c[j]
        c_head = j + c.head
        if debug: print("maxchild c & c.head:", store[doc_c[j].lex.lower], store[doc_c[c_head].lex.lower])
        if c_head != i:
            continue
        if c.r_edge <= maxchild_idx:
            continue
        if test == 0 \
                or (test == 1 and inside(c.dep, hashes.nsubj_or_dep)) \
                or (test == 2 and c.head == i and not inside(c.dep, hashes.conj_or_prep)):
            maxchild_idx = c.r_edge
            if debug: print("keep as maxchild", store[doc_c[maxchild_idx].lex.lower])
    if debug:
        print("left side before cleaning:", store[doc_c[minchild_idx].lex.lower])
        print("right side before cleaning:", store[doc_c[maxchild_idx].lex.lower])
    # Clean up endings and begginging
    while maxchild_idx >= minchild_idx and (inside(doc_c[maxchild_idx].pos, hashes.remove_pos)
                                        or inside(doc_c[maxchild_idx].lex.lower, hashes.lower_not_end)):
        PyErr_CheckSignals()
        if debug: print("Removing last token", store[doc_c[maxchild_idx].lex.lower], store[doc_c[maxchild_idx].tag])
        maxchild_idx -= 1 # We don't want mentions finishing with 's or conjunctions/punctuation
    while minchild_idx <= maxchild_idx and (inside(doc_c[minchild_idx].pos, hashes.remove_pos) 
                                        or inside(doc_c[minchild_idx].lex.lower, hashes.lower_not_end)):
        PyErr_CheckSignals()
        if debug: print("Removing first token", store[doc_c[minchild_idx].lex.lower], store[doc_c[minchild_idx].tag])
        minchild_idx += 1 # We don't want mentions starting with 's or conjunctions/punctuation
    if debug:
        print("left side after cleaning:", store[doc_c[minchild_idx].lex.lower])
        print("right side after cleaning:", store[doc_c[maxchild_idx].lex.lower])
    return minchild_idx, maxchild_idx + 1

cdef add_span(int start, int end, SentSpans* mentions_spans, TokenC* doc_c,
              StringStore store, bint debug=False):
    cdef int num = mentions_spans.num
    if debug:
        print("🔥 Add span: " + ' '.join(store[doc_c[i].lex.lower] for i in range(start, end)))
    mentions_spans.spans[num].start = start
    mentions_spans.spans[num].end = end
    mentions_spans.num += 1
    if debug:
        print("🔥 Add span: " + ' '.join(store[doc_c[i].lex.lower] for i in range(start, end)))
        print("🔥 mentions_spans.num: ", mentions_spans.num)
    return mentions_spans.num >= mentions_spans.max_spans

cdef _extract_from_sent(TokenC* doc_c, int sent_start, int sent_end, SentSpans* mentions_spans,
                        HashesList hashes, StringStore store, bint blacklist=False,
                        bint debug=False):
    '''
    Extract Pronouns and Noun phrases mentions from a spacy Span
    '''
    cdef int i, j, c_head, k, endIdx, minchild_idx, maxchild_idx, n_spans
    cdef bint test
    if debug:
        print("😎 Extract sents start, end:", sent_start, sent_end)

    for i in range(sent_start, sent_end):
        PyErr_CheckSignals()
        token = doc_c[i]
        if debug: print("🚀 tok:", store[token.lex.lower], "tok.tag:", store[token.tag],
                        "tok.pos:", store[token.pos], "tok.dep:", store[token.dep])
        if blacklist and inside(token.lex.lower, hashes.no_coref_list):
            if debug: print("token in no_coref_list")
            continue
        if (not inside(token.tag, hashes.keep_tags) or inside(token.dep, hashes.leave_dep) \
            and not inside(token.dep, hashes.keep_dep)):
            if debug: print("not pronoun or no right dependency")
            continue
        # pronoun
        if inside(token.tag, hashes.PRP_tags): #re.match(r"PRP.*", token.tag_):
            if debug: print("PRP")
            endIdx = i + 1
            #span = doc_c[i: endIdx]
            #if debug: print("==-- PRP store:", span)
            test = add_span(i, i+1, mentions_spans, doc_c, store)
            if test: return
            # when pronoun is a part of conjunction (e.g., you and I)
            if token.r_kids > 0 or token.l_kids > 0:
                #span = doc[token.l_edge : token.r_edge+1]
                #if debug: print("==-- in conj store:", span)
                test = add_span(token.l_edge, token.r_edge+1, mentions_spans, doc_c, store)
                if test: return
            continue
        # Add NP mention
        if debug:
            print("NP or IN:", store[token.lex.lower])
            if store[token.tag] == 'IN':
                print("IN tag")
        # Take care of 's
        if token.lex.lower == hashes.POSSESSIVE_MARK:
            if debug: print("'s detected")
            c_head = i + token.head
            j = 0
            while c_head != 0 and j < MAX_ITER:
                if debug:
                    print("token head:", c_head, doc_c[c_head].dep, "head:", c_head + doc_c[c_head].head)
                if doc_c[c_head].dep == hashes.NSUBJ_MARK:
                    start, end = enlarge_span(doc_c, c_head, sent_start, sent_end, 1, hashes, store)
                    if debug: print("'s', i1:", store[doc_c[start].lex.lower], " i2:", store[doc_c[end].lex.lower])
                    #if debug: print("==-- 's' store:", span)
                    test = add_span(start, end+1, mentions_spans, doc_c, store)
                    if test: return
                    break
                c_head += doc_c[c_head].head
                j += 1
            assert j != MAX_ITER
            continue

        for j in range(sent_start, sent_end):
            c = doc_c[j]
            if debug and j + c.head == i:
                print("🚧 token in span:", store[c.lex.lower])#, "- head & dep:", c.head, c.dep)
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
        if debug and token.tag == hashes.IN_TAG:
            print("IN tag")
        test = False
        for tok in doc_c[sent_start:sent_end]:
            if inside(tok.dep, hashes.conj_or_prep):
                test = True
                break
        if test:
            if debug: print("Conjunction found, storing first element separately")
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

cdef extract_mentions_spans(Doc doc, bint blacklist=False, bint debug=True):
    '''
    Extract potential mentions from a spacy parsed Doc
    '''
    cdef:
        int i, max_spans
        int n_sents
        HashesList hashes
        SpanC spans_c
        int n_spans = 0
        Pool mem = Pool()

    if debug: print('===== doc ====:', doc)
    for c in doc:
        if debug: print("🚧 span search:", c, "head:", c.head, "tag:", c.tag_, "pos:", c.pos_, "dep:", c.dep_)
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

    hashes = get_hash_lookups(doc.vocab.strings, mem)

    if debug: print("==-- ents:", list(((ent, ent.label_) for ent in mentions_spans)))
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
    if debug: print("cleaned_mentions_spans", cleaned_mentions_spans)
    return cleaned_mentions_spans, n_spans

#######################
###### CLASSES ########

class Model(object):
    ''' Coreference neural net model '''
    def __init__(self, model_path):
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("single_mention_weights"):
                w = numpy.load(os.path.join(model_path, file))
                weights.append(w)
            if file.startswith("single_mention_bias"):
                w = numpy.load(os.path.join(model_path, file))
                biases.append(w)
        self.single_mention_model = list(zip(weights, biases))
        weights, biases = [], []
        for file in sorted(os.listdir(model_path)):
            if file.startswith("pair_mentions_weights"):
                w = numpy.load(os.path.join(model_path, file))
                weights.append(w)
            if file.startswith("pair_mentions_bias"):
                w = numpy.load(os.path.join(model_path, file))
                biases.append(w)
        self.pair_mentions_model = list(zip(weights, biases))

    def _score(self, features, layers):
        for weights, bias in layers:
            features = numpy.matmul(weights, features) + bias
            if weights.shape[0] > 1:
                features = numpy.maximum(features, 0) # ReLU
        return numpy.sum(features, axis=0)

    def get_multiple_single_score(self, first_layer_input):
        return self._score(first_layer_input, self.single_mention_model)

    def get_multiple_pair_score(self, first_layer_input):
        return self._score(first_layer_input, self.pair_mentions_model)


class EmbeddingExtractor:
    ''' Compute words embedding features for mentions '''
    def __init__(self, pretrained_model_path):
        _, self.static_embeddings, self.stat_idx, self.stat_voc = self.load_embeddings_from_file(pretrained_model_path + "static_word")
        _, self.tuned_embeddings, self.tun_idx, self.tun_voc = self.load_embeddings_from_file(pretrained_model_path + "tuned_word")
        self.fallback = self.static_embeddings.get(UNKNOWN_WORD)

        self.shape = self.static_embeddings[UNKNOWN_WORD].shape
        shape2 = self.tuned_embeddings[UNKNOWN_WORD].shape
        assert self.shape == shape2

    @staticmethod
    def load_embeddings_from_file(name):
        print("Loading embeddings from", name)
        embeddings = {}
        voc_to_idx = {}
        idx_to_voc = []
        mat = numpy.load(name+"_embeddings.npy").astype(dtype='float32')
        average_mean = numpy.average(mat, axis=0, weights=numpy.sum(mat, axis=1))
        with io.open(name+"_vocabulary.txt", 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                embeddings[line.strip()] = mat[i, :]
                voc_to_idx[line.strip()] = i
                idx_to_voc.append(line.strip())
        return average_mean, embeddings, voc_to_idx, idx_to_voc

    @staticmethod
    def normalize_word(w):
        if w is None:
            return MISSING_WORD
        return re.sub(r"\d", u"0", w.lower_)

    def get_stat_word(self, word):
        if word in self.static_embeddings:
            return word, self.static_embeddings.get(word)
        else:
            return UNKNOWN_WORD, self.fallback

    def get_word_embedding(self, word, static=False):
        ''' Embedding for a single word (tuned if possible, otherwise static) '''
        norm_word = self.normalize_word(word)
        if static:
            return self.get_stat_word(norm_word)
        else:
            if norm_word in self.tuned_embeddings:
                return norm_word, self.tuned_embeddings.get(norm_word)
            else:
                return self.get_stat_word(norm_word)
 
    def get_word_in_sentence(self, word_idx, sentence):
        ''' Embedding for a word in a sentence '''
        if word_idx < sentence.start or word_idx >= sentence.end:
            return self.get_word_embedding(None)
        return self.get_word_embedding(sentence.doc[word_idx])

    def get_average_embedding(self, token_list):
        ''' Embedding for a list of words '''
        embed_vector = numpy.zeros(self.shape, dtype='float32') #We could also use numpy.copy(self.average_mean)
        word_list = []
        for tok in token_list:
            if tok.lower_ not in [".", "!", "?"]:
                word, embed = self.get_word_embedding(tok, static=True)
                embed_vector += embed
                word_list.append(word)
        return word_list, (embed_vector/max(len(word_list), 1))

    def get_mention_embeddings(self, ms, doc_embedding):
        ''' Get span (averaged) and word (single) embeddings of a mention '''
        st = ms.sent
        ms_lefts = ms.doc[max(ms.start-5, st.start):ms.start]
        ms_rights = ms.doc[ms.end:min(ms.end+5, st.end)]
        head = ms.root.head
        spans = [self.get_average_embedding(ms),
                 self.get_average_embedding(ms_lefts),
                 self.get_average_embedding(ms_rights),
                 self.get_average_embedding(st),
                 (unicode_(doc_embedding[0:8]) + "...", doc_embedding)]
        words = [self.get_word_embedding(ms.root),
                 self.get_word_embedding(ms[0]),
                 self.get_word_embedding(ms[-1]),
                 self.get_word_in_sentence(ms.start-1, st),
                 self.get_word_in_sentence(ms.end, st),
                 self.get_word_in_sentence(ms.start-2, st),
                 self.get_word_in_sentence(ms.end+1, st),
                 self.get_word_embedding(head)]
        spans_embeddings_ = {"00_Mention": spans[0][0],
                             "01_MentionLeft": spans[1][0],
                             "02_MentionRight": spans[2][0],
                             "03_Sentence": spans[3][0],
                             "04_Doc": spans[4][0]}
        words_embeddings_ = {"00_MentionHead": words[0][0],
                             "01_MentionFirstWord": words[1][0],
                             "02_MentionLastWord": words[2][0],
                             "03_PreviousWord": words[3][0],
                             "04_NextWord": words[4][0],
                             "05_SecondPreviousWord": words[5][0],
                             "06_SecondNextWord": words[6][0],
                             "07_MentionRootHead": words[7][0]}
        return (spans_embeddings_,
                words_embeddings_,
                numpy.concatenate([em[1] for em in spans], axis=0),
                numpy.concatenate([em[1] for em in words], axis=0))


cdef class CorefComponent(object):
    """spaCy v2.0 Coref pipeline component. """
    def __cinit__(self, nlp, label='coref',
                 greedyness=0.5, max_dist=50, max_dist_match=500, blacklist=False):
        self.greedyness = greedyness
        self.max_dist = max_dist
        self.max_dist_match = max_dist_match
        self.blacklist = blacklist
        self.coref_model = None
        self.embed_extractor = None
        self.name = None
        self.label = None

    def __init__(self, nlp, label='coref', greedyness=0.5, max_dist=50,
                 max_dist_match=500, blacklist=False):
        """Initialise the pipeline component.
        """
        self.name = 'coref' # component name, will show up in the pipeline
        self.label = nlp.vocab.strings[label]  # get entity label ID
        model_path = os.path.join(PACKAGE_DIRECTORY, "weights/")
        print("Loading neuralcoref model from", model_path)
        self.coref_model = Model(model_path)
        self.embed_extractor = EmbeddingExtractor(model_path)

        # Register attributes on Doc and Span
        Doc.set_extension('has_coref', default=False)
        Doc.set_extension('coref_mentions', default=None)
        Doc.set_extension('coref_clusters', default=None)
        Doc.set_extension('coref_resolved', default="")
        Span.set_extension('is_coref', default=False)
        Span.set_extension('coref_cluster', default=None)
        Span.set_extension('coref_main', default=None)

        print('Coref Initialized')
        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        # If no default value is set, it defaults to None.
        # Token.set_extension('in_coref', default=False)
        # Token.set_extension('coref_clusters')
        # Token.set_extension('coref_mentions', getter=self.coref_clusters)

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

    cdef build_clusters(self, doc):
        ''' Build coreference clusters '''
        cdef:
            Mention_C* c
            Mention_C m1, m2
            uint64_t i, ant_idx, men_idx, b_idx # Should maybe be uint64_t (but we get a set of cython warnings)
            uint64_t [:] p_ant, p_men, best_ant
            float [:] embed, feats
            float [:, :] s_inp, p_inp
            double [:] score, best_score
            Pool mem = Pool()
        strings = doc.vocab.strings

        print('Coref Extract mentions')
        # ''' Extract mentions '''
        mentions, n_mentions = extract_mentions_spans(doc, blacklist=self.blacklist)

        print('Coref Sorte and prepare mentions')
        n_sents = len(list(doc.sents))
        mentions = sorted((m for m in mentions), key=lambda m: (m.root.i, m.start))
        c = <Mention_C*>mem.alloc(n_mentions, sizeof(Mention_C))
        print('Coref Fill up corefs C struct')
        for i, m in enumerate(mentions):
            print('Coref Fill up corefs C struct step', i)
            c[i].entity_label = get_span_entity_label(m)
            c[i].span_start = m.start
            c[i].span_end = m.end
            c[i].sent_idx = get_span_sent_number(m)
            c[i].mention_type = get_span_type(m)
            c[i].root_lower = (<Token>m.root).c.lex.lower
            c[i].span_lower = strings.add(m.text.lower())
            content_words = set(tok.lower_ for tok in m if tok.tag_ in CONTENT_TAGS)
            print('Coref Fill up corefs C struct set content words', content_words)
            c[i].content_words.length = len(content_words)
            c[i].content_words.arr = <hash_t*>mem.alloc(len(content_words), sizeof(hash_t))
            for j, w in enumerate(content_words):
                c[i].content_words.arr[j] = strings.add(w)

        print('Coref Prepare arrays')
        # ''' Prepare arrays of pairs indices and features for feeding the model '''
        pairs_ant = []
        pairs_men = []
        n_pairs = 0
        if self.max_dist_match is not None:
            word_to_mentions = {}
            for i in range(n_mentions):
                c_w = c[i].content_words
                for j in range(c_w.length):
                    tok = strings[c_w.arr[j]]
                    if not tok in word_to_mentions:
                        word_to_mentions[tok] = [i]
                    else:
                        word_to_mentions[tok].append(i)
        for i in range(n_mentions):
            antecedents = set(range(i)) if self.max_dist is None else set(range(max(0, i - self.max_dist), i))
            # if debug: print("antecedents", antecedents)
            if self.max_dist_match is not None:
                c_w = c[i].content_words
                for j in range(c_w.length):
                    tok = strings[c_w.arr[j]]
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

        # ''' Build single features and pair features arrays '''
        _, doc_embedding = self.embed_extractor.get_average_embedding(doc)
        print('build single features')
        for i, mention in enumerate(mentions):
            (_, _, spans_embeddings, words_embeddings) = self.embed_extractor.get_mention_embeddings(mention, doc_embedding)
            print('set feats i', i)
            one_hot_type = numpy.zeros((4,), dtype='float32')
            one_hot_type[c[i].mention_type] = 1
            features_ = {"01_MentionType": c[i].mention_type,
                         "02_MentionLength": len(mention)-1,
                         "03_MentionNormLocation": (i/n_mentions),
                         "04_IsMentionNested": 1 if any((mentions[j] is not mention
                                                          and c[j].sent_idx == c[i].sent_idx
                                                          and c[j].span_start <= c[i].span_start
                                                          and c[j].span_end <= c[i].span_end)
                                                         for j in range(n_mentions)) else 0}
            print('single features ok')
            features = numpy.concatenate([one_hot_type,
                                          encode_distance(features_["02_MentionLength"]),
                                          numpy.array(features_["03_MentionNormLocation"], ndmin=1, copy=False, dtype='float32'),
                                          numpy.array(features_["04_IsMentionNested"], ndmin=1, copy=False, dtype='float32')
                                         ], axis=0)
            embed_cont = numpy.concatenate([spans_embeddings, words_embeddings], axis=0)
            embed = embed_cont
            feats = features
            s_inp[:SGNL_FEATS_0, i] = embed
            s_inp[SGNL_FEATS_0:SGNL_FEATS_1, i] = feats

        print('build pair features')
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            m1 = c[ant_idx]
            m2 = c[men_idx]
            p_inp[:PAIR_FEATS_0, i] = s_inp[:SGNL_FEATS_0, ant_idx]
            p_inp[PAIR_FEATS_0:PAIR_FEATS_1, i] = s_inp[:SGNL_FEATS_0, men_idx]
            p_inp[PAIR_FEATS_1, i] = 1
            p_inp[PAIR_FEATS_1 + 1, i] = 0
            p_inp[PAIR_FEATS_1 + 2, i] = 0
            p_inp[PAIR_FEATS_1 + 3, i] = heads_agree(m1, m2)
            p_inp[PAIR_FEATS_1 + 4, i] = exact_match(m1, m2)
            p_inp[PAIR_FEATS_1 + 5, i] = relaxed_match(m1, m2)
            b_idx, val = index_distance(m2.sent_idx - m1.sent_idx)
            p_inp[PAIR_FEATS_2:PAIR_FEATS_3, i] = 0
            p_inp[PAIR_FEATS_2 + b_idx, i] = 1
            p_inp[PAIR_FEATS_3, i] = val
            b_idx, val = index_distance(men_idx - ant_idx - 1)
            p_inp[PAIR_FEATS_4:PAIR_FEATS_5, i] = 0
            p_inp[PAIR_FEATS_4 + b_idx, i] = 1
            p_inp[PAIR_FEATS_5, i] = val
            p_inp[PAIR_FEATS_5 + 1, i] = overlapping(m1, m2)
            p_inp[PAIR_FEATS_6:PAIR_FEATS_7, i] = s_inp[SGNL_FEATS_0:SGNL_FEATS_1, ant_idx]
            p_inp[PAIR_FEATS_7:PAIR_FEATS_8, i] = s_inp[SGNL_FEATS_0:SGNL_FEATS_1, men_idx]

        # ''' Compute scores '''
        best_score_ar = numpy.empty((n_mentions), dtype='float64')
        best_ant_ar = numpy.empty((n_mentions), dtype=numpy.uint64)
        best_score = best_score_ar
        best_ant = best_ant_ar
        print('Computing Single mention scores')
        score = self.coref_model.get_multiple_single_score(s_inp)
        for i in range(n_mentions):
            best_score[i] = score[i] - 50 * (self.greedyness - 0.5)
            best_ant[i] = i
        print('Computing pair mention scores')
        score = self.coref_model.get_multiple_pair_score(p_inp)
        for i in range(n_pairs):
            ant_idx = p_ant[i]
            men_idx = p_men[i]
            if score[i] > best_score[men_idx]:
                best_score[men_idx] = score[i]
                best_ant[men_idx] = ant_idx

        # ''' Build clusters '''
        mention_to_cluster = list(range(n_mentions))
        cluster_to_main = list(range(n_mentions))
        clusters = dict((i, [i]) for i in mention_to_cluster)
        print("merge clusters")
        for mention_idx, ant_idx in enumerate(best_ant):
            if ant_idx != mention_idx:
                if mention_to_cluster[ant_idx] == mention_to_cluster[mention_idx]:
                    continue
                print("Merge clusters", ant_idx, mention_idx)
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
                        print("Update representative mention of cluster")
                        cluster_to_main[ant_idx] = mention_idx
        print("Remove singleton clusters and convert rest to spans")
        remove_id = []
        main = list(mentions)
        mentions_list = []
        coreferences = []
        clusters_list = {}
        for key, m_idx_list in clusters.items():
            if len(m_idx_list) != 1:
                m_list = list(mentions[i] for i in m_idx_list)
                main = mentions[cluster_to_main[key]]
                mentions_list += m_list
                clusters_list[main] = m_list
                coreferences += list((main, m) for m in m_list)

        # ''' Update doc '''
        if len(clusters) != 0:
            doc._.set('has_coref', True)
            doc._.set('coref_mentions', mentions_list)
            doc._.set('coref_clusters', clusters_list)
            doc._.set('coref_resolved', get_resolved(doc, coreferences))
            for main, m_list in clusters_list.items():
                for mention in m_list:
                    mention._.set('is_coref', True)
                    mention._.set('coref_cluster', m_list)
                    mention._.set('coref_main', main)
        return doc
