# coding: utf8
# cython: profile=True
# cython: infer_types=True
# distutils: language=c++
"""data models and pre-processing for the coref algorithm"""

from __future__ import unicode_literals
from __future__ import print_function

import re
import io
from six import string_types, integer_types
from libc.stdint cimport uint8_t, uint32_t, int32_t, uint64_t

from neuralcoref.compat import unicode_
from neuralcoref.utils import encode_distance, parallel_process

try:
    from itertools import izip_longest as zip_longest
except ImportError: # will be 3.x series
    from itertools import zip_longest

from cymem.cymem cimport Pool
from cpython.exc cimport PyErr_CheckSignals

import spacy
from spacy.strings cimport StringStore
from spacy.tokens.span cimport Span
from spacy.typedefs cimport flags_t, attr_t, hash_t
from spacy.structs cimport TokenC
cimport numpy as np
np.import_array()
import numpy

#########################
####### UTILITIES #######
#########################

MENTION_TYPE = {"PRONOMINAL": 0, "NOMINAL": 1, "PROPER": 2, "LIST": 3}
MENTION_LABEL = {0: "PRONOMINAL", 1: "NOMINAL", 2: "PROPER", 3: "LIST"}

NO_COREF_LIST = ["i", "me", "my", "you", "your"]
KEEP_TAGS = ["NN", "NNP", "NNPS", "NNS", "PRP", "PRP$", "DT", "IN"]
PROPERS_TAGS = ["NN", "NNS", "NNP", "NNPS"]
PRP_TAGS = ["PRP", "PRP$"]
LEAVE_DEP = ["det", "compound", "appos"]
KEEP_DEP = ["nsubj", "dobj", "iobj", "pobj"]
REMOVE_POS = ["CCONJ", "INTJ", "ADP"]
LOWER_NOT_END = ["'s", ',', '.', '!', '?', ':', ';']
ACCEPTED_ENTS = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"]
WHITESPACE_PATTERN = r"\s+|_+"
UNKNOWN_WORD = "*UNK*"
MISSING_WORD = "<missing>"
MAX_ITER = 100
SPAN_FACTOR = 4

#########################
## MENTION EXTRACTION ###
#########################

cdef bint inside(attr_t element, Hashes hashes):
    cdef int i
    cdef attr_t* arr = hashes.arr
    cdef int length = hashes.length
    for i in range(length):
        if arr[i] == element:
            return True
    return False

cdef HashesList get_hash_lookups(StringStore store):
    cdef HashesList hashes
    cdef np.ndarray[attr_t, ndim=1] no_coref_list = numpy.asarray(list(store.add(st) for st in NO_COREF_LIST), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] keep_tags = numpy.asarray(list(store.add(st) for st in KEEP_TAGS), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] PRP_tags = numpy.asarray(list(store.add(st) for st in PRP_TAGS), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] leave_dep = numpy.asarray(list(store.add(st) for st in LEAVE_DEP), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] keep_dep = numpy.asarray(list(store.add(st) for st in KEEP_DEP), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] nsubj_or_dep = numpy.asarray(list(store.add(st) for st in ["nsubj", "dep"]), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] conj_or_prep = numpy.asarray(list(store.add(st) for st in ["conj", "prep"]), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] remove_pos = numpy.asarray(list(store.add(st) for st in REMOVE_POS), dtype=numpy.uint64)
    cdef np.ndarray[attr_t, ndim=1] lower_not_end = numpy.asarray(list(store.add(st) for st in LOWER_NOT_END), dtype=numpy.uint64)
    cdef attr_t POSSESSIVE_MARK = store.add("'s")
    cdef attr_t NSUBJ_MARK = store.add("nsubj")
    cdef attr_t IN_TAG = store.add('IN')
    cdef attr_t MARK_DEP = store.add("mark")
    hashes.no_coref_list.arr, hashes.no_coref_list.length = <hash_t*>no_coref_list.data, no_coref_list.shape[0]
    hashes.keep_tags.arr, hashes.keep_tags.length = <hash_t*>keep_tags.data, keep_tags.shape[0]
    hashes.PRP_tags.arr, hashes.PRP_tags.length = <hash_t*>PRP_tags.data, PRP_tags.shape[0]
    hashes.leave_dep.arr, hashes.leave_dep.length = <hash_t*>leave_dep.data, leave_dep.shape[0]
    hashes.keep_dep.arr, hashes.keep_dep.length = <hash_t*>keep_dep.data, keep_dep.shape[0]
    hashes.nsubj_or_dep.arr, hashes.nsubj_or_dep.length = <hash_t*>nsubj_or_dep.data, nsubj_or_dep.shape[0]
    hashes.conj_or_prep.arr, hashes.conj_or_prep.length = <hash_t*>conj_or_prep.data, conj_or_prep.shape[0]
    hashes.remove_pos.arr, hashes.remove_pos.length = <hash_t*>remove_pos.data, remove_pos.shape[0]
    hashes.lower_not_end.arr, hashes.lower_not_end.length = <hash_t*>lower_not_end.data, lower_not_end.shape[0]
    hashes.POSSESSIVE_MARK = POSSESSIVE_MARK
    hashes.NSUBJ_MARK = NSUBJ_MARK
    hashes.IN_TAG = IN_TAG
    hashes.MARK_DEP = MARK_DEP
    return hashes

# Utility to remove bad endings
cdef enlarge_span(TokenC* doc_c, int i, int sent_start, int sent_end, int test,
                  HashesList hashes, StringStore store, bint debug=True):
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

cdef add_span(int start, int end, SentSpans mentions_spans, TokenC* doc_c, StringStore store, bint debug=True):
    cdef int num = mentions_spans.num
    if debug:
        print("🔥 Add span: " + ' '.join(store[doc_c[i].lex.lower] for i in range(start, end)))
    mentions_spans.spans[num].start = start
    mentions_spans.spans[num].end = end
    mentions_spans.num += 1
    return mentions_spans.num >= mentions_spans.max_spans

cdef _extract_from_sent(TokenC* doc_c, int sent_start, int sent_end, SentSpans mentions_spans,
                        HashesList hashes, StringStore store, bint use_no_coref_list=True, bint debug=True):
    '''
    Extract Pronouns and Noun phrases mentions from a spacy Span
    '''
    cdef int i, j, k, endIdx, minchild_idx, maxchild_idx, n_spans
    cdef bint test
    if debug:
        print("😎 Extract sents start, end:", sent_start, sent_end)

    for i in range(sent_start, sent_end):
        PyErr_CheckSignals()
        token = doc_c[i]
        if debug: print("🚀 tok:", store[token.lex.lower], "tok.tag:", store[token.tag],
                        "tok.pos:", store[token.pos], "tok.dep:", store[token.dep])
        if use_no_coref_list and inside(token.lex.lower, hashes.no_coref_list):
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
            h = i + token.head
            j = 0
            while h != 0 and j < MAX_ITER:
                if debug:
                    print("token head:", h, h.dep, "head:", h + h.head)
                    print(id(h.head), id(h))
                if h.dep == hashes.NSUBJ_MARK:
                    start, end = enlarge_span(doc_c, h, sent_start, sent_end, 1, hashes, store)
                    if debug: print("'s', i1:", store[doc_c[start].lex.lower], " i2:", store[doc_c[end].lex.lower])
                    #if debug: print("==-- 's' store:", span)
                    test = add_span(start, end+1, mentions_spans, doc_c, store)
                    if test: return
                    break
                h += h.head
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

cdef extract_mentions_spans(Doc doc, bint use_no_coref_list=True, bint debug=True):
    '''
    Extract potential mentions from a spacy parsed Doc
    '''
    if debug: print('===== doc ====:', doc)
    for c in doc:
        if debug: print("🚧 span search:", c, "head:", c.head, "tag:", c.tag_, "pos:", c.pos_, "dep:", c.dep_)
    # Named entities
    mentions_spans = list(ent for ent in doc.ents if ent.label_ in ACCEPTED_ENTS)

    # Setup for fast scanning
    cdef Pool mem = Pool()
    cdef int n_sents = len(list(doc.sents))
    sent_spans = <SentSpans*>mem.alloc(n_sents, sizeof(SentSpans))
    cdef int max_spans
    for i, sent in enumerate(doc.sents):
        max_spans = len(sent)*SPAN_FACTOR
        sent_spans[i].spans = <SpanC*>mem.alloc(max_spans, sizeof(SpanC))
        sent_spans[i].max_spans = max_spans
        sent_spans[i].num = 0

    cdef HashesList hashes
    hashes = get_hash_lookups(doc.vocab.strings)

    if debug: print("==-- ents:", list(((ent, ent.label_) for ent in mentions_spans)))
    for i, sent in enumerate(doc.sents):
        _extract_from_sent(doc.c, sent.start, sent.end, sent_spans[i], hashes, doc.vocab.strings)
    #for spans in parallel_process([{'span': sent,
    #                                'use_no_coref_list': use_no_coref_list} for sent in doc.sents],
    #                            _extract_from_sent, use_kwargs=True, n_jobs=4, front_num=0):
    #    mentions_spans = mentions_spans + spans
    spans_set = set()
    cleaned_mentions_spans = []
    for spans in mentions_spans:
        if spans.end > spans.start and (spans.start, spans.end) not in spans_set:
            cleaned_mentions_spans.append(spans)
            spans_set.add((spans.start, spans.end))

    return cleaned_mentions_spans

#########################
####### CLASSES #########

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

cdef class Mention():
    '''
    A mention (possible anaphor) comprise a spacy Span class with additional informations
    '''
    def __cinit__(self, Span span, int mention_index, int utterance_index, int utterances_start_sent):
        '''
        Arguments:
            span (spaCy Span): the spaCy span from which creating the Mention object
            mention_index (int): index of the Mention in the Document
            utterance_index (int): index of the utterance of the Mention in the Document
            utterances_start_sent (int): index of the first sentence of the utterance of the Mention in the Document
                (an utterance can comprise several sentences)
        '''
        self.span = span
        self.index = mention_index
        self.utterance_index = utterance_index

        self.utterances_sent = utterances_start_sent + get_span_sent_number(span)
        self.spans_embeddings = None
        self.words_embeddings = None
        self.embeddings = None
        self.features = None

        self.spans_embeddings_ = None
        self.words_embeddings_ = None
        self.features_ = None

        self.mention_type = get_span_type(span)
        self.propers = set(self.content_words)
        self.entity_label = get_span_entity_label(span)

    property content_words:
        ''' Returns an iterator of nouns/proper nouns in the Mention '''
        def __get__(self):
            return (tok.lower_ for tok in self.span if tok.tag_ in PROPERS_TAGS)

    def __repr__(self):
        return self.span.__repr__()

    def __len__(self):
        return len(self.span)

    def __getitem__(self, x):
        return self.span[x]

    def __contains__(self, x):
        return x in self.span

    cpdef int heads_agree(self, Mention mention2):
        ''' Does the root of the Mention match the root of another Mention/Span'''
        # we allow same-type NEs to not match perfectly,
        # but rather one could be included in the other, e.g., "George" -> "George Bush"
        if (self.entity_label != -1 and mention2.entity_label != -1 and
                self.entity_label == mention2.entity_label and
                (self.span.root.lower_ in mention2.span.lower_ \
                 or mention2.span.root.lower_ in self.span.lower_)):
            return 1
        return 1 if self.span.root.lower_ == mention2.span.root.lower_ else 0

    cpdef int exact_match(self, Mention mention2):
        ''' Does the Mention lowercase text matches another Mention/Span lowercase text'''
        return 1 if self.span.lower_ == mention2.span.lower_ else 0

    cpdef int relaxed_match(self, Mention mention2):
        ''' Does the nouns/proper nous in the Mention match another Mention/Span nouns/propers'''
        return 1 if not self.propers.isdisjoint(mention2.propers) else 0

    cpdef int overlapping(self, Mention m2):
        return 1 if (self.utterances_sent == m2.utterances_sent \
                     and self.span.end > m2.span.start) else 0

class EmbeddingExtractor:
    '''
    Compute words embedding features for mentions
    '''
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

    def get_document_embedding(self, utterances_list):
        ''' Embedding for the document '''
    #    We could also use this: embed_vector = numpy.copy(self.average_mean)#numpy.zeros(self.shape)
    #    return embed_vector
        embed_vector = numpy.zeros(self.shape, dtype='float32')
        for utt in utterances_list:
            _, utt_embed = self.get_average_embedding(utt)
            embed_vector += utt_embed
        return embed_vector/max(len(utterances_list), 1)

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

    def get_mention_embeddings(self, mention, doc_embedding):
        ''' Get span (averaged) and word (single) embeddings of a mention '''
        ms = mention.span
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

cdef class Document:
    '''
    Main data class: encapsulate list of utterances, mentions
    Process utterances to extract mentions and pre-compute mentions features
    '''
    def __cinit__(self, nlp, utterances=None,
                 use_no_coref_list=False,
                 trained_embed_path=None, embedding_extractor=None,
                 conll=None, debug=False):
        '''
        Arguments:
            nlp (spaCy Language Class): A spaCy Language Class for processing the text input
            utterances: utterance(s) to load already see self.add_utterances()
            use_no_coref_list (boolean): use a list of term for which coreference is not preformed
            pretrained_model_path (string): Path to a folder with pretrained word embeddings
            embedding_extractor (EmbeddingExtractor): Use a pre-loaded word embeddings extractor
            conll (string): If training on coNLL data: identifier of the document type
            debug (boolean): print debug informations
        '''
        self.nlp = nlp
        self.use_no_coref_list = use_no_coref_list
        self.debug = debug

        self.utterances = []
        self.mentions = []
        self.n_sents = 0
        self.n_mentions = 0
        self.n_pairs = 0
        self.pairs_ant = None
        self.pairs_men = None

        self.genre_, self.genre = self.set_genre(conll)

        if trained_embed_path is not None and embedding_extractor is None:
            self.embed_extractor = EmbeddingExtractor(trained_embed_path)
        elif embedding_extractor is not None:
            self.embed_extractor = embedding_extractor
        else:
            self.embed_extractor = None

        if utterances:
            self.add_utterances(utterances)

    def set_genre(self, conll):
        if conll is not None:
            genre = numpy.zeros((7,), dtype='float32')
            genre[conll] = 1
        else:
            genre = numpy.array(0, ndmin=1, copy=False)
        return conll, genre

    def __str__(self):
        return '<utterances> \n {}\n<mentions> \n {}' \
                .format('\n '.join(unicode_(i) for i in self.utterances),
                        '\n '.join(unicode_(i) for i in self.mentions))

    def __len__(self):
        ''' Return the number of mentions (not utterances) since it is what we really care about '''
        return len(self.mentions)

    def __getitem__(self, key):
        ''' Return a specific mention (not utterance) '''
        return self.mentions[key]

    def __iter__(self):
        ''' Iterate over mentions (not utterances) '''
        for mention in self.mentions:
            yield mention

    #######################################
    ###### UTERANCE LOADING FUNCTIONS #####
    #######################################
    
    def set_utterances(self, utterances):
        self.utterances = []
        self.mentions = []
        self.n_sents = 0
        self.n_mentions = 0
        if utterances:
            self.add_utterances(utterances)

    def add_utterances(self, utterances):
        '''
        Add utterances to the utterance list and build mention list for these utterances

        Arg:
            utterances : iterator or list of string corresponding to successive utterances
        Return:
            List of indexes of added utterances in the docs
        '''
        if self.debug: print("Adding utterances", utterances)
        if isinstance(utterances, string_types):
            utterances = [utterances]
        utterances_index = []
        utt_start = len(self.utterances)
        docs = self.nlp.pipe(utterances)
        for utt_index, doc in enumerate(docs):
            m_span = extract_mentions_spans(doc, use_no_coref_list=self.use_no_coref_list)
            self._process_mentions(m_span, utt_start + utt_index, self.n_sents)
            utterances_index.append(utt_start + utt_index)
            self.utterances.append(doc)
            self.n_sents += len(list(doc.sents))

        self.set_mentions_features()
        self.set_candidate_pairs()

    ###################################
    ## FEATURES MENTIONS EXTRACTION ###
    ###################################

    def _process_mentions(self, mentions_spans, utterance_index, n_sents):
        '''
        Process mentions in a spacy doc (an utterance)
        '''
        processed_spans = sorted((m for m in mentions_spans), key=lambda m: (m.root.i, m.start))
        for mention_index, span in enumerate(processed_spans):
            self.mentions.append(Mention(span, mention_index + self.n_mentions,
                                             utterance_index, n_sents))
            self.n_mentions += 1

    def set_mentions_features(self):
        '''
        Compute features for the extracted mentions
        '''
        doc_embedding = self.embed_extractor.get_document_embedding(self.utterances) if self.embed_extractor is not None else None
        for mention in self.mentions:
            one_hot_type = numpy.zeros((4,), dtype='float32')
            one_hot_type[mention.mention_type] = 1
            features_ = {"01_MentionType": mention.mention_type,
                         "02_MentionLength": len(mention)-1,
                         "03_MentionNormLocation": (mention.index)/len(self.mentions),
                         "04_IsMentionNested": 1 if any((m is not mention
                                                          and m.utterances_sent == mention.utterances_sent
                                                          and m.span.start <= mention.span.start
                                                          and mention.span.end <= m.span.end)
                                                         for m in self.mentions) else 0}
            features = numpy.concatenate([one_hot_type,
                                       encode_distance(features_["02_MentionLength"]),
                                       numpy.array(features_["03_MentionNormLocation"], ndmin=1, copy=False, dtype='float32'),
                                       numpy.array(features_["04_IsMentionNested"], ndmin=1, copy=False, dtype='float32')
                                      ], axis=0)
            spans_embeddings_, words_embeddings_, spans_embeddings, words_embeddings = self.embed_extractor.get_mention_embeddings(mention, doc_embedding)
            mention.features_ = features_
            mention.features = features
            mention.spans_embeddings = spans_embeddings
            mention.spans_embeddings_ = spans_embeddings_
            mention.words_embeddings = words_embeddings
            mention.words_embeddings_ = words_embeddings_
            mention.embeddings = numpy.concatenate([spans_embeddings, words_embeddings], axis=0)

    def get_single_mention_features(self, mention):
        ''' Features for anaphoricity test (signle mention features + genre if conll)'''
        features_ = mention.features_
        features_["DocGenre"] = self.genre_
        return (features_, numpy.concatenate([mention.features, self.genre], axis=0))

    def get_raw_pair_features(self, m1, m2):
        ''' Features for pair of mentions (string match)'''
        return {"00_SameSpeaker": 1,
                "01_AntMatchMentionSpeaker": 0,
                "02_MentionMatchSpeaker": 0,
                "03_HeadsAgree": 1 if m1.heads_agree(m2) else 0,
                "04_ExactStringMatch": 1 if m1.exact_match(m2) else 0,
                "05_RelaxedStringMatch": 1 if m1.relaxed_match(m2) else 0,
                "06_SentenceDistance": m2.utterances_sent - m1.utterances_sent,
                "07_MentionDistance": m2.index - m1.index - 1,
                "08_Overlapping": 1 if (m1.utterances_sent == m2.utterances_sent and m1.span.end > m2.span.start) else 0,
                "09_M1Features": m1.features_,
                "10_M2Features": m2.features_,
                "11_DocGenre": self.genre_}

    def get_pair_mentions_features(self, m1, m2):
        features_ = self.get_raw_pair_features(m1, m2)
        pairwise_features = [m1.embeddings,
                             m2.embeddings,
                             numpy.array([features_["00_SameSpeaker"],
                                       features_["01_AntMatchMentionSpeaker"],
                                       features_["02_MentionMatchSpeaker"],
                                       features_["03_HeadsAgree"],
                                       features_["04_ExactStringMatch"],
                                       features_["05_RelaxedStringMatch"]]),
                             encode_distance(features_["06_SentenceDistance"]),
                             encode_distance(features_["07_MentionDistance"]),
                             numpy.array(features_["08_Overlapping"], ndmin=1),
                             m1.features,
                             m2.features,
                             self.genre]
        return (features_, numpy.concatenate(pairwise_features, axis=0))

    ###################################
    ###### ITERATOR OVER MENTIONS #####
    ###################################

    def set_candidate_pairs(self, max_distance=50, max_distance_with_match=500, debug=False):
        '''
        Yield tuples of mentions, dictionnary of candidate antecedents for the mention

        Arg:
            mentions: an iterator over mention indexes (as returned by get_candidate_mentions)
            max_mention_distance : max distance between a mention and its antecedent
            max_mention_distance_string_match : max distance between a mention and
                its antecedent when there is a proper noun match
        '''
        cdef int i
        pairs_ant = []
        pairs_men = []
        if max_distance_with_match is not None:
            word_to_mentions = {}
            for i in range(self.n_mentions):
                for tok in self.mentions[i].content_words:
                    if not tok in word_to_mentions:
                        word_to_mentions[tok] = [i]
                    else:
                        word_to_mentions[tok].append(i)
        for i in range(self.n_mentions):
            antecedents = set(range(i)) if max_distance is None else set(range(max(0, i - max_distance), i))
            if debug: print("antecedents", antecedents)
            if max_distance_with_match is not None:
                for tok in self.mentions[i].content_words:
                    with_string_match = word_to_mentions.get(tok, None)
                    for match_idx in with_string_match:
                        if match_idx < i and match_idx >= i - max_distance_with_match:
                            antecedents.add(match_idx)
            pairs_ant += list(antecedents)
            pairs_men += [i]*len(antecedents)
            self.n_pairs += len(antecedents)
        self.pairs_ant = numpy.asarray(pairs_ant, dtype=numpy.uint64)
        self.pairs_men = numpy.asarray(pairs_men, dtype=numpy.uint64)

def mention_detection_debug(sentence):
    print(u"🌋 Loading spacy model")
    try:
        spacy.info('en_core_web_sm')
        model = 'en_core_web_sm'
    except IOError:
        print("No spacy 2 model detected, using spacy1 'en' model")
        spacy.info('en')
        model = 'en'
    nlp = spacy.load(model)
    doc = nlp(sentence.decode('utf-8'))
    mentions = extract_mentions_spans(doc, use_no_coref_list=False, debug=True)
    for mention in mentions:
        print(mention)
