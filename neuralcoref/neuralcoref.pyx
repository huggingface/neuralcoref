# coding: utf8
# cython: infer_types=True, boundscheck=False
# distutils: language=c++
""" NeuralCoref resolution spaCy v2.0 pipeline component 
Custom pipeline components: https://spacy.io//usage/processing-pipelines#custom-components
Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
# from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME

import plac
import re
import os
import io
from collections import OrderedDict
import json
cimport cython
from cpython cimport array
import array
from libc.stdint cimport uint16_t, uint32_t, uint64_t, uintptr_t, int32_t
import cytoolz

import numpy
from cymem.cymem cimport Pool
import spacy
from spacy.compat import json_dumps
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
from spacy import util
from spacy.compat import is_config

from thinc.v2v import Model, ReLu, Affine
from thinc.api import chain, clone
# from thinc.neural.util import get_array_module

##############################
##### A BUNCH OF SIZES #######

DEF MAX_BINS = 9
DEF MAX_FOLLOW_UP = 50
DEF MAX_ITER = 100
DEF SPAN_FACTOR = 4

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

DISTANCE_BINS_PY = array.array('i', list(range(5)) + [5]*3 + [6]*8 + [7]*16 + [8]*32)

cdef:
    int [::1] DISTANCE_BINS = DISTANCE_BINS_PY
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

##########################################################
##### UTILITIES TO CONVERT STRINGS IN SPACY HASHES #######

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
    hashes.unknown_word = store.add(u"*UNK*")
    hashes.missing_word = store.add(u"<missing>")
    hashes.digit_word = store.add(u"0")
    return hashes

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

cdef inline int is_nested(Mention_C* c, int n_mentions, int m_idx):
    for i in range(n_mentions):
        if i == m_idx:
            continue
        if c[i].sent_idx == c[m_idx].sent_idx \
            and c[i].span_start <= c[m_idx].span_start \
            and c[i].span_end >= c[m_idx].span_end:
            return 1
    return 0

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

cdef inline int heads_agree(Mention_C m1, Mention_C m2) nogil:
    ''' Does the root of the Mention match the root of another Mention/Span'''
    # In CoreNLP: they allow same-type NEs to not match perfectly
    # but one could be included in the other, e.g., "George" -> "George Bush"
    # In this cython C function, we take the simpler approach of directly comparing the roots hashes
    return 1 if m1.root_lower == m2.root_lower else 0

cdef inline int exact_match(Mention_C m1, Mention_C m2) nogil:
    return 1 if m1.span_lower == m2.span_lower else 0

cdef inline int relaxed_match(Mention_C m1, Mention_C m2) nogil:
    for i in range(m1.content_words.length):
        if inside(m1.content_words.arr[i], m2.content_words):
            return True
    return False

cdef inline int overlapping(Mention_C m1, Mention_C m2) nogil:
    return 1 if (m1.sent_idx == m2.sent_idx and m1.span_end > m2.span_start) else 0

cdef (int, int, int) get_span_sent(Span span):
    ''' return index, start and end of the sentence of a Span in its Doc'''
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

def get_resolved(doc, clusters):
    ''' Return a list of utterrances text where the coref are resolved to the most representative mention'''
    resolved = list(tok.text_with_ws for tok in doc)
    for cluster in clusters:
        for coref in cluster:
            if coref != cluster.main:
                resolved[coref.start] = cluster.main.text + doc[coref.end-1].whitespace_
                for i in range(coref.start+1, coref.end):
                    resolved[i] = ""
    return ''.join(resolved)

#################################################
##### RULE_BASED MENTION EXTRACTION LOGIC #######

cdef (int, int) enlarge_span(TokenC* doc_c, int i, int sent_start, int sent_end, int test,
                  HashesList hashes) nogil:
    ''' Utility function to remove bad detected mention endings '''
    cdef int j
    cdef uint32_t minchild_idx
    cdef uint32_t maxchild_idx
    minchild_idx = i
    maxchild_idx = i
    # if debug: print("enlarge_span")
    # if debug: print("test", test)
    # if debug: print("sent_start", sent_start)
    # if debug: print("sent_end", sent_end)
    for j in range(sent_start, sent_end):
        # if debug: print("j", j)
        c = doc_c[j]
        c_head = j + c.head
        if c_head != i:
            continue
        if c.l_edge >= minchild_idx:
            continue
        if test == 0 \
                or (test == 1 and inside(c.dep, hashes.nsubj_or_dep)) \
                or (test == 2 and c.head == i and not inside(c.dep, hashes.conj_or_prep)):
            minchild_idx = c.l_edge
    for j in range(sent_start, sent_end):
        # if debug: print("j", j)
        c = doc_c[j]
        c_head = j + c.head
        if c_head != i:
            continue
        if c.r_edge <= maxchild_idx:
            continue
        if test == 0 \
                or (test == 1 and inside(c.dep, hashes.nsubj_or_dep)) \
                or (test == 2 and c.head == i and not inside(c.dep, hashes.conj_or_prep)):
            maxchild_idx = c.r_edge
    # if debug: print("minchild_idx", minchild_idx)
    # if debug: print("maxchild_idx", maxchild_idx)
    # if debug: print("Clean up endings and begginging")
    # Clean up endings and begginging
    while maxchild_idx >= minchild_idx and maxchild_idx > sent_start \
          and (inside(doc_c[maxchild_idx].pos, hashes.remove_pos)
               or inside(doc_c[maxchild_idx].lex.lower, hashes.lower_not_end)):
        # if debug: print("maxchild_idx", maxchild_idx)
        maxchild_idx -= 1 # We don't want mentions finishing with 's or conjunctions/punctuation
    # if debug: print("maxchild_idx", maxchild_idx)
    while minchild_idx <= maxchild_idx and minchild_idx < sent_end - 1 \
          and (inside(doc_c[minchild_idx].pos, hashes.remove_pos) 
               or inside(doc_c[minchild_idx].lex.lower, hashes.lower_not_end)):
        minchild_idx += 1 # We don't want mentions starting with 's or conjunctions/punctuation
        # if debug: print("minchild_idx", minchild_idx)
    # if debug: print("minchild_idx", minchild_idx)
    return minchild_idx, maxchild_idx + 1

cdef bint add_span(int start, int end, SentSpans* mentions_spans, TokenC* doc_c) nogil:
    ''' Utility function to add a detected mention to our SentSpans structure '''
    cdef int num = mentions_spans.num
    # if debug: print("add_span")
    mentions_spans.spans[num].start = start
    mentions_spans.spans[num].end = end
    mentions_spans.num += 1
    return mentions_spans.num >= mentions_spans.max_spans # True when the max memory available to store spans is reached

cdef void _extract_from_sent(TokenC* doc_c, int sent_start, int sent_end, SentSpans* mentions_spans,
                        HashesList hashes, bint blacklist=False) nogil:
    ''' Main function to extract Pronouns and Noun phrases mentions from a spacy Span '''
    cdef int i, j, c_head, k, endIdx, minchild_idx, maxchild_idx, n_spans
    cdef bint test
    for i in range(sent_start, sent_end):
        # if debug: print("token", i)
        token = doc_c[i]
        if blacklist and inside(token.lex.lower, hashes.no_coref_list):
            # if debug: print("blacklist")
            continue
        if (not inside(token.tag, hashes.keep_tags) or inside(token.dep, hashes.leave_dep) \
            and not inside(token.dep, hashes.keep_dep)):
            # if debug: print("not in keep tags or deps")
            continue
        if inside(token.tag, hashes.PRP_tags): # pronoun
            # if debug: print("pronoun")
            endIdx = i + 1
            test = add_span(i, i+1, mentions_spans, doc_c)
            if test: return
            # when pronoun is a part of conjunction (e.g., you and I)
            if token.r_kids > 0 or token.l_kids > 0:
                test = add_span(token.l_edge, token.r_edge+1, mentions_spans, doc_c)
                if test: return
            continue
        # Add NP mention
        # if debug: print("NP mention")
        if token.lex.lower == hashes.POSSESSIVE_MARK: # Take care of 's
            # if debug: print("Take care of 's")
            c_head = i + token.head
            j = 0
            while c_head != 0 and j < MAX_ITER:
                if doc_c[c_head].dep == hashes.NSUBJ_MARK:
                    start, end = enlarge_span(doc_c, c_head, sent_start, sent_end, 1, hashes)
                    test = add_span(start, end+1, mentions_spans, doc_c)
                    if test: return
                    break
                c_head += doc_c[c_head].head
                j += 1
            continue
        # if debug: print("Enlarge span")
        for j in range(sent_start, sent_end):
            c = doc_c[j]
        start, end = enlarge_span(doc_c, i, sent_start, sent_end, 0, hashes)
        if token.tag == hashes.IN_TAG and token.dep == hashes.MARK_DEP and start == end:
            start, end = enlarge_span(doc_c, i + token.head, sent_start, sent_end, 0, hashes)
        if start == end:
            # if debug: print("Empty span")
            continue
        if doc_c[start].lex.lower == hashes.POSSESSIVE_MARK:
            # if debug: print("we probably already have stored this mention")
            continue # we probably already have stored this mention
        test = add_span(start, end, mentions_spans, doc_c)
        if test: return
        test = False
        for tok in doc_c[sent_start:sent_end]:
            if inside(tok.dep, hashes.conj_or_prep):
                test = True
                break
        if test:
            # if debug: print("conj_or_prep")
            start, end = enlarge_span(doc_c, i, sent_start, sent_end, 0, hashes)
            if start == end:
                continue
            test = add_span(start, end, mentions_spans, doc_c)
            if test: return
    return

cdef extract_mentions_spans(Doc doc, HashesList hashes, bint blacklist=False):
    ''' Extract potential mentions from a spacy parsed Doc '''
    cdef:
        int i, max_spans
        int n_sents
        SpanC spans_c
        int n_spans = 0
        Pool mem = Pool()
    mentions_spans = list(ent for ent in doc.ents if ent.label_ in ACCEPTED_ENTS) # Named entities
    n_sents = len(list(doc.sents))
    # if debug: print("n_sents", n_sents)
    sent_spans = <SentSpans*>mem.alloc(n_sents, sizeof(SentSpans))
    for i, sent in enumerate(doc.sents):
        max_spans = len(sent)*SPAN_FACTOR
        sent_spans[i].spans = <SpanC*>mem.alloc(max_spans, sizeof(SpanC))
        sent_spans[i].max_spans = max_spans
        sent_spans[i].num = 0
        # if debug: print("sent", i, "max_spans", max_spans)
    for i, sent in enumerate(doc.sents): # Extract spans from each sentence in the doc (nogil so could be parallelized)
        # if debug: print("extact from", i)
        _extract_from_sent(doc.c, sent.start, sent.end, &sent_spans[i], hashes, blacklist=blacklist)
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
    return cleaned_mentions_spans, n_spans


##########################
###### MAIN CLASSES ########

class Cluster:
    """ A utility class to store our annotations in the spaCy Doc """
    def __init__(self, i, main, mentions):
        self.i = i
        self.main = main
        self.mentions = mentions

    def __getitem__(self, i):
        return self.mentions[i]

    def __iter__(self):
        for mention in self.mentions:
            yield mention

    def __len__(self):
        return len(self.mentions)

    def __unicode__(self):
        return unicode(self.main) + u': ' + unicode(self.mentions)

    def __bytes__(self):
        return unicode(self).encode('utf-8')

    def __str__(self):
        if is_config(python3=True):
            return self.__unicode__()
        return self.__bytes__()

    def __repr__(self):
        return self.__str__()

cdef class NeuralCoref(object):
    """ spaCy v2.0 Coref pipeline component """
    name = 'coref'

    @classmethod
    def Model(cls, **cfg):
        """Initialize a model for the pipe."""
        h1 = util.env_opt('h1', cfg.get('h1', 1000))
        h2 = util.env_opt('h2', cfg.get('h2', 500))
        h3 = util.env_opt('h3', cfg.get('h3', 500))
        with Model.define_operators({'**': clone, '>>': chain}):
            single_model = ReLu(h1, SIZE_SINGLE_IN) >> ReLu(h2, h1) >> ReLu(h3, h2) >> Affine(1, h3) >> Affine(1, 1)
            pairs_model = ReLu(h1, SIZE_PAIR_IN) >> ReLu(h2, h1) >> ReLu(h3, h2) >> Affine(1, h3) >> Affine(1, 1)
        cfg = {
            'h1': h1,
            'h2': h2,
            'h3': h3,
        }
        return (single_model, pairs_model), cfg

    def __init__(self, Vocab vocab, model=True, **cfg):
        """Create a Coref pipeline component.
        vocab (Vocab): The vocabulary object. Must be shared with documents
            to be processed. The value is set to the `.vocab` attribute.
        model (object): Neural net model. The value is set to the .model attribute. If set to True
            (default), a new instance will be created with `NeuralCoref.Model()`
            in NeuralCoref.from_disk() or NeuralCoref.from_bytes().
        **cfg: Arbitrary configuration parameters. Set to the `.cfg` attribute
        """
        self.vocab = vocab
        self.model = model
        if 'greedyness' not in cfg:
            cfg['greedyness'] = util.env_opt('greedyness', 0.5)
        if 'max_dist' not in cfg:
            cfg['max_dist'] = util.env_opt('max_dist', 50)
        if 'max_dist_match' not in cfg:
            cfg['max_dist_match'] = util.env_opt('max_dist_match', 500)
        if 'blacklist' not in cfg:
            cfg['blacklist'] = util.env_opt('blacklist', True)
        if 'conv_dict' not in cfg:
            cfg['conv_dict'] = util.env_opt('conv_dict', None)
        self.cfg = cfg
        self.hashes = get_hash_lookups(vocab.strings, vocab.mem)
        self.static_vectors = Vectors()
        self.tuned_vectors = Vectors()
        self.conv_dict = None

        # Register attributes on Doc and Span
        if not Doc.has_extension('huggingface_neuralcoref'):
            Doc.set_extension('huggingface_neuralcoref', default=True)
            Doc.set_extension('has_coref', default=False)
            Doc.set_extension('coref_clusters', default=None)
            Doc.set_extension('coref_resolved', default="")
            Span.set_extension('is_coref', default=False)
            Span.set_extension('coref_cluster', default=None)
            Token.set_extension('in_coref', getter=self.token_in_coref)
            Token.set_extension('coref_clusters', getter=self.token_clusters)

    def __reduce__(self):
        return (NeuralCoref, (self.vocab, self.model), None, None)

    def set_conv_dict(self, conv_dict):
        self.conv_dict = Vectors()
        for key, words in conv_dict.items():
            norm_k = self.normalize(key)
            norm_w = list(self.normalize(w) for w in words)
            embed_vector = numpy.zeros(self.static_vectors.shape[1], dtype='float32')
            for hash_w in norm_w:
                embed_vector += self.tuned_vectors[hash_w] if hash_w in self.tuned_vectors else self.get_static(hash_w)
            self.conv_dict.add(key=norm_k, vector=embed_vector/max(len(norm_w), 1))

    def __call__(self, doc, greedyness=None, max_dist=None, max_dist_match=None,
             conv_dict=None, blacklist=None):
        """Apply the pipeline component on a Doc object. """
        if greedyness is None:
            greedyness = self.cfg.get('greedyness', 0.5)
        if max_dist is None:
            max_dist = self.cfg.get('max_dist', 50)
        if max_dist_match is None:
            max_dist_match = self.cfg.get('max_dist_match', 500)
        if conv_dict is None:
            conv_dict = self.cfg.get('conv_dict', None)
        if blacklist is None:
            blacklist = self.cfg.get('blacklist', True)
        if conv_dict is not None:
            self.set_conv_dict(conv_dict)
        annotations = self.predict([doc], greedyness=greedyness, max_dist=max_dist,
                                  max_dist_match=max_dist_match, blacklist=blacklist)
        self.set_annotations([doc], annotations)
        return doc

    def pipe(self, stream, batch_size=128, n_threads=1,
             greedyness=None, max_dist=None, max_dist_match=None,
             conv_dict=None, blacklist=None):
        """Process a stream of documents. Currently not optimized.
        stream: The sequence of documents to process.
        batch_size (int): Number of documents to accumulate into a working set.
        n_threads (int): The number of threads with which to work on the buffer
            in parallel.
        YIELDS (Doc): Documents, in order.
        """
        if greedyness is None:
            greedyness = self.cfg.get('greedyness', 0.5)
        if max_dist is None:
            max_dist = self.cfg.get('max_dist', 50)
        if max_dist_match is None:
            max_dist_match = self.cfg.get('max_dist_match', 500)
        if conv_dict is None:
            conv_dict = self.cfg.get('conv_dict', None)
        if blacklist is None:
            blacklist = self.cfg.get('blacklist', True)
        for docs in cytoolz.partition_all(batch_size, stream):
            docs = list(docs)
            annotations = self.predict(docs, greedyness=greedyness, max_dist=max_dist,
                                    max_dist_match=max_dist_match, blacklist=blacklist)
            self.set_annotations(docs, annotations)
            yield from docs

    def predict(self, docs, float greedyness=0.5, int max_dist=50, int max_dist_match=500,
                conv_dict=None, bint blacklist=False):
        ''' Predict coreference clusters
        docs (iterable): A sequence of `Doc` objects.
        RETURNS (iterable): List of (lists of mentions, lists of clusters, lists of main mentions per cluster) for each doc.
        '''
        cdef:
            Mention_C* c
            Mention_C m1, m2
            TokenC* doc_c
            Doc doc
            uint64_t i, ant_idx, men_idx, b_idx, n_mentions, n_pairs
            uint64_t [::1] p_ant, p_men, best_ant
            float [::1] embed, feats, doc_embed, mention_embed, best_score
            float [:, ::1] s_inp, p_inp
            float [:, ::1] score
            Pool mem
            StringStore strings
        #    timespec ts
        #    double timing0, timing1, timing2, timing3, timing4
        #    clock_gettime(CLOCK_REALTIME, &ts)
        #    timing0 = ts.tv_sec + (ts.tv_nsec / 1000000000.)

        annotations = []
        # if debug: print("Extract mentions")
        for doc in docs:
            mem = Pool() # We use this for doc specific allocation
            strings = doc.vocab.strings
            # ''' Extract mentions '''
            mentions, n_mentions = extract_mentions_spans(doc, self.hashes, blacklist=blacklist)
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

            # if debug: print("Prepare arrays of pairs indices and features for feeding the model")
            # ''' Prepare arrays of pairs indices and features for feeding the model '''
            pairs_ant = []
            pairs_men = []
            n_pairs = 0
            if max_dist_match is not None:
                word_to_mentions = {}
                for i in range(n_mentions):
                    for tok in content_words[i]:
                        if not tok in word_to_mentions:
                            word_to_mentions[tok] = [i]
                        else:
                            word_to_mentions[tok].append(i)
            for i in range(n_mentions):
                if max_dist is None:
                    antecedents = set(range(<object>i))
                else:
                    antecedents = set(range(max(0, <object>i - max_dist), <object>i))
                if max_dist_match is not None:
                    for tok in content_words[i]:
                        with_string_match = word_to_mentions.get(tok, None)
                        for match_idx in with_string_match:
                            if match_idx < i and match_idx >= i - max_dist_match:
                                antecedents.add(match_idx)
                pairs_ant += list(antecedents)
                pairs_men += [i]*len(antecedents)
                n_pairs += len(antecedents)
            p_ant_arr = numpy.asarray(pairs_ant, dtype=numpy.uint64)
            p_men_arr = numpy.asarray(pairs_men, dtype=numpy.uint64)
            p_ant = p_ant_arr
            p_men = p_men_arr
            s_inp_arr = numpy.zeros((n_mentions, SIZE_SNGL_IN_NO_GENRE + SIZE_GENRE), dtype='float32')
            s_inp = s_inp_arr
            p_inp_arr = numpy.zeros((n_pairs, SIZE_PAIR_IN_NO_GENRE + SIZE_GENRE), dtype='float32')
            p_inp = p_inp_arr

            # if debug: print("Build single features and pair features arrays")
            # ''' Build single features and pair features arrays '''
            doc_c = doc.c
            doc_embedding = numpy.zeros(SIZE_EMBEDDING, dtype='float32') # self.embeds.get_average_embedding(doc.c, 0, doc.length + 1, self.hashes.puncts)
            doc_embed = doc_embedding
            for i in range(n_mentions):
                s_inp_arr[i, :SGNL_FEATS_0] = self.get_mention_embeddings(mentions[i], doc_embedding) # Set embeddings
                s_inp_arr[i, SGNL_FEATS_0 + c[i].mention_type] = 1                      # 01_MentionType
                b_idx, val = index_distance(c[i].span_end - c[i].span_start - 1)    # 02_MentionLength
                s_inp_arr[i, SGNL_FEATS_1 + b_idx] = 1
                s_inp_arr[i, SGNL_FEATS_2] = val
                val = float(i)/float(n_mentions)                                    # 03_MentionNormLocation
                s_inp_arr[i, SGNL_FEATS_3] = val
                s_inp_arr[i, SGNL_FEATS_4] = is_nested(c, n_mentions, i)                # 04_IsMentionNested
            for i in range(n_pairs):
                ant_idx = p_ant[i]
                men_idx = p_men[i]
                m1 = c[ant_idx]
                m2 = c[men_idx]
                p_inp[i, :PAIR_FEATS_0] = s_inp[ant_idx, :SGNL_FEATS_0]
                p_inp[i, PAIR_FEATS_0:PAIR_FEATS_1] = s_inp[men_idx, :SGNL_FEATS_0]
                p_inp[i, PAIR_FEATS_1] = 1                                          # 00_SameSpeaker
                # p_inp[i, PAIR_FEATS_1 + 1] = 0                                    # 01_AntMatchMentionSpeaker # arrays are initialized to zero
                # p_inp[i, PAIR_FEATS_1 + 2] = 0                                    # 02_MentionMatchSpeaker
                p_inp[i, PAIR_FEATS_1 + 3] = heads_agree(m1, m2)                    # 03_HeadsAgree
                p_inp[i, PAIR_FEATS_1 + 4] = exact_match(m1, m2)                    # 04_ExactStringMatch
                p_inp[i, PAIR_FEATS_1 + 5] = relaxed_match(m1, m2)                  # 05_RelaxedStringMatch
                b_idx, val = index_distance(m2.sent_idx - m1.sent_idx)              # 06_SentenceDistance
                # p_inp[i, PAIR_FEATS_2:PAIR_FEATS_3] = 0
                p_inp[i, PAIR_FEATS_2 + b_idx] = 1
                p_inp[i, PAIR_FEATS_3] = val
                b_idx, val = index_distance(men_idx - ant_idx - 1)                  # 07_MentionDistance
                # p_inp[i, PAIR_FEATS_4:PAIR_FEATS_5] = 0
                p_inp[i, PAIR_FEATS_4 + b_idx] = 1
                p_inp[i, PAIR_FEATS_5] = val
                p_inp[i, PAIR_FEATS_5 + 1] = overlapping(m1, m2)                    # 08_Overlapping
                p_inp[i, PAIR_FEATS_6:PAIR_FEATS_7] = s_inp[ant_idx, SGNL_FEATS_0:SGNL_FEATS_5] # 09_M1Features
                p_inp[i, PAIR_FEATS_7:PAIR_FEATS_8] = s_inp[men_idx, SGNL_FEATS_0:SGNL_FEATS_5] # 10_M2Features
                # 11_DocGenre is zero currently

            # if debug: print("Compute scores")
            # ''' Compute scores '''
            best_score_ar = numpy.empty((n_mentions), dtype='float32')
            best_ant_ar = numpy.empty((n_mentions), dtype=numpy.uint64)
            best_score = best_score_ar
            best_ant = best_ant_ar
            score = self.model[0](s_inp_arr)
            for i in range(n_mentions):
                best_score[i] = score[i, 0] - 50 * (greedyness - 0.5)
                best_ant[i] = i
            score = self.model[1](p_inp_arr)
            for i in range(n_pairs):
                ant_idx = p_ant[i]
                men_idx = p_men[i]
                if score[i, 0] > best_score[men_idx]:
                    best_score[men_idx] = score[i, 0]
                    best_ant[men_idx] = ant_idx

            # if debug: print("Build clusters")
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
            clusters_list = []
            i = 0
            for key, m_idx_list in clusters.items():
                if len(m_idx_list) != 1:
                    m_list = list(mentions[i] for i in m_idx_list)
                    main = mentions[cluster_to_main[key]]
                    clusters_list.append(Cluster(i, main, m_list))
                    i += 1
            annotations.append(clusters_list)

        return annotations

    def set_annotations(self, docs, annotations):
        """Set the tensor attribute for a batch of documents.
        docs (iterable): A sequence of `Doc` objects.
        tensors (object): Vector representation for each token in the docs.
        """
        if isinstance(docs, Doc):
            docs = [docs]
        cdef Doc doc
        for doc, clusters in zip(docs, annotations):
            if len(clusters) != 0:
                doc._.set('has_coref', True)
                doc._.set('coref_clusters', clusters)
                doc._.set('coref_resolved', get_resolved(doc, clusters))
                for cluster in clusters:
                    for mention in cluster:
                        mention._.set('is_coref', True)
                        mention._.set('coref_cluster', cluster)

    def token_in_coref(self, token):
        """Getter for Token attributes. Returns True if the token
        is in a cluster."""
        return any([token in mention for cluster in token.doc._.coref_clusters
                    for mention in cluster])

    def token_clusters(self, token):
        """Getter for Token attributes. Returns a list of the cluster in which the token
        is."""
        clusters = []
        for cluster in token.doc._.coref_clusters:
            for mention in cluster:
                if token in mention:
                    clusters.append(cluster)
                    break
        return clusters

    def normalize(self, Token token):
        return self.hashes.digit_word if token.is_digit else token.lower

    def get_static(self, hash_t word):
        return self.static_vectors[word] if word in self.static_vectors else self.static_vectors[self.hashes.unknown_word]

    def get_word_embedding(self, Token token, bint tuned=True):
        hash_w = self.normalize(token)
        if self.conv_dict is not None and hash_w in self.conv_dict:
            return self.conv_dict[hash_w]
        if tuned and hash_w in self.tuned_vectors:
            return self.tuned_vectors[hash_w]
        return self.get_static(hash_w)
 
    def get_word_in_sentence(self, int i, Span sent):
        if i < sent.start or i >= sent.end:
            return self.tuned_vectors[self.hashes.missing_word]
        return self.get_word_embedding(sent.doc[i])

    def get_average_embedding(self, Span span):
        cdef int i
        cdef int n = 0
        embed_arr = numpy.zeros(self.static_vectors.shape[1], dtype='float32')
        for token in span:
            if token.lower not in PUNCTS:
                n += 1
                embed_vector = self.get_word_embedding(token, tuned=False)
                embed_arr = embed_arr + embed_vector
        embed_arr = numpy.divide(embed_arr, float(max(n, 1)))
        return embed_arr

    def get_mention_embeddings(self, Span span, doc_embedding):
        ''' Create a mention embedding with span (averaged) and word (single) embeddings '''
        doc = span.doc
        sent = span.sent
        embeddings = numpy.zeros((EMBED_13, ), dtype='float32')
        embeddings[        :EMBED_01] = self.get_average_embedding(span)
        embeddings[EMBED_01:EMBED_02] = self.get_average_embedding(doc[max(span.start-5, sent.start):span.start])
        embeddings[EMBED_02:EMBED_03] = self.get_average_embedding(doc[span.end:min(span.end + 5, sent.end)])
        embeddings[EMBED_03:EMBED_04] = self.get_average_embedding(sent)
        embeddings[EMBED_04:EMBED_05] = doc_embedding
        embeddings[EMBED_05:EMBED_06] = self.get_word_embedding(span.root)
        embeddings[EMBED_06:EMBED_07] = self.get_word_embedding(span[0])
        embeddings[EMBED_07:EMBED_08] = self.get_word_embedding(span[-1])
        embeddings[EMBED_08:EMBED_09] = self.get_word_in_sentence(span.start-1, sent)
        embeddings[EMBED_09:EMBED_10] = self.get_word_in_sentence(span.end, sent)
        embeddings[EMBED_10:EMBED_11] = self.get_word_in_sentence(span.start-2, sent)
        embeddings[EMBED_11:EMBED_12] = self.get_word_in_sentence(span.end+1, sent)
        embeddings[EMBED_12:        ] = self.get_word_embedding(span.root.head)
        return embeddings

    def to_disk(self, path, **exclude):
        serializers = {
            'single_model': lambda p: p.open('wb').write(self.model[0].to_bytes()),
            'pairs_model': lambda p: p.open('wb').write(self.model[1].to_bytes()),
            'static_vectors': lambda p: self.static_vectors.to_disk(p),
            'tuned_vectors': lambda p: self.tuned_vectors.to_disk(p),
            'cfg': lambda p: p.open('w').write(json_dumps(self.cfg))
        }
        util.to_disk(path, serializers, exclude)

    def from_disk(self, path, **exclude):
        deserializers = {
            'cfg': lambda p: self.cfg.update(util.read_json(p)),
            'model': lambda p: None
        }
        util.from_disk(path, deserializers, exclude)
        if 'model' not in exclude:
            path = util.ensure_path(path)
            if self.model is True:
                self.model, cfg = self.Model(**self.cfg)
            else:
                cfg = {}
            with (path / 'single_model').open('rb') as file_:
                bytes_data = file_.read()
            self.model[0].from_bytes(bytes_data)
            with (path / 'pairs_model').open('rb') as file_:
                bytes_data = file_.read()
            self.model[1].from_bytes(bytes_data)
            self.static_vectors.from_disk(path / 'static_vectors')
            self.tuned_vectors.from_disk(path / 'tuned_vectors')
            self.cfg.update(cfg)
        return self

    def to_bytes(self, **exclude):
        serializers = OrderedDict((
            ('static_vectors', lambda: self.static_vectors.to_bytes()),
            ('tuned_vectors', lambda: self.tuned_vectors.to_bytes())
            ('single_model', lambda: self.model[0].to_bytes()),
            ('pairs_model', lambda: self.model[1].to_bytes()),
            ('cfg', lambda: json.dumps(self.cfg, indent=2, sort_keys=True))
        ))
        if 'model' in exclude:
            exclude['static_vectors'] = True
            exclude['tuned_vectors'] = True
            exclude['single_model'] = True
            exclude['pairs_model'] = True
            exclude.pop('model')
        return util.to_bytes(serializers, exclude)

    def from_bytes(self, bytes_data, **exclude):
        deserializers = OrderedDict((
            ('cfg', lambda b: self.cfg.update(json.loads(b))),
            ('static_vectors', lambda b: None),
            ('tuned_vectors', lambda b: None),
            ('single_model', lambda b: None),
            ('pairs_model', lambda b: None)
        ))
        msg = util.from_bytes(bytes_data, deserializers, exclude)
        if 'model' not in exclude:
            if self.model is True:
                self.model, cfg = self.Model(**self.cfg)
            else:
                cfg = {}
            if 'static_vectors' in msg:
                self.static_vectors.from_bytes(msg['static_vectors'])
            if 'tuned_vectors' in msg:
                self.tuned_vectors.from_bytes(msg['tuned_vectors'])
            if 'single_model' in msg:
                self.model[0].from_bytes(msg['single_model'])
            if 'pairs_model' in msg:
                self.model[1].from_bytes(msg['pairs_model'])
            self.cfg.update(cfg)
        return self
