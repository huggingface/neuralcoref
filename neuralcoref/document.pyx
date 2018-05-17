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
from spacy.tokens.token cimport Token
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

#########################
## MENTION EXTRACTION ###
#########################

cdef bint inside(hash_t element, Hashes hashes):
    cdef int i
    cdef hash_t* arr = hashes.arr
    cdef int length = hashes.length
    for i in range(length):
        if arr[i] == element:
            return True
    return False

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
                        HashesList hashes, StringStore store, bint use_no_coref_list=False,
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

cdef extract_mentions_spans(Doc doc, bint use_no_coref_list=False, bint debug=False):
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
                           use_no_coref_list=use_no_coref_list)
    #for spans in parallel_process([{'span': sent,
    #                                'use_no_coref_list': use_no_coref_list} for sent in doc.sents],
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
    def __cinit__(self, Span span, int utt_idx, int utt_start_sent_idx):
        '''
        Arguments:
            span (spaCy Span): the spaCy span from which creating the Mention object
            utterance_index (int): index of the utterance of the Mention in the Document
            utterances_start_sent (int): index of the first sentence of the utterance of the Mention in the Document
                (an utterance can comprise several sentences)
        '''
        self.span = span
        self.utt_idx = utt_idx
        self.sent_idx = utt_start_sent_idx + get_span_sent_number(span)
        self.spans_embeddings_ = None
        self.words_embeddings_ = None
        self.features_ = None
        self.content_words = set(tok.lower_ for tok in self.span if tok.tag_ in CONTENT_TAGS)

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
        if (self.c.entity_label != -1 and mention2.c.entity_label != -1 and
                self.c.entity_label == mention2.c.entity_label and
                (self.span.root.lower_ in mention2.span.lower_ \
                 or mention2.span.root.lower_ in self.span.lower_)):
            return 1
        return 1 if self.span.root.lower_ == mention2.span.root.lower_ else 0

    cpdef int exact_match(self, Mention mention2):
        ''' Does the Mention lowercase text matches another Mention/Span lowercase text'''
        return 1 if self.span.lower_ == mention2.span.lower_ else 0

    cpdef int relaxed_match(self, Mention mention2):
        ''' Does the content words in the Mention match another Mention/Span content words'''
        return 1 if not self.content_words.isdisjoint(mention2.content_words) else 0

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
                  bint use_no_coref_list=False,
                  trained_embed_path=None, embedding_extractor=None,
                  conll=None,
                  bint debug=False):
        '''
        '''
        self.nlp = None
        self.mem = Pool()
        self.use_no_coref_list = use_no_coref_list
        self.debug = debug
        self.genre_ = None
        self.genre = None
        self.embed_extractor = None

    def __init__(self, nlp, utterances=None,
                 use_no_coref_list=False,
                 trained_embed_path=None, embedding_extractor=None,
                 conll=None, debug=False):
        '''
        Arguments:
            nlp (spaCy Language Class): A spaCy Language Class for processing the text input
            utterances: utterance(s) to load already see self.set_utterances()
            use_no_coref_list (boolean): use a list of term for which coreference is not preformed
            pretrained_model_path (string): Path to a folder with pretrained word embeddings
            embedding_extractor (EmbeddingExtractor): Use a pre-loaded word embeddings extractor
            conll (string): If training on coNLL data: identifier of the document type
            debug (boolean): print debug informations
        '''
        self.nlp = nlp
        self.mem = Pool()
        self.use_no_coref_list = use_no_coref_list
        self.debug = debug
        self.genre_, self.genre = self.set_genre(conll)
        if trained_embed_path is not None and embedding_extractor is None:
            self.embed_extractor = EmbeddingExtractor(trained_embed_path)
        elif embedding_extractor is not None:
            self.embed_extractor = embedding_extractor
        else:
            self.embed_extractor = None
        self.set_utterances(utterances)

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
        '''
        Clean up add utterances
        Arg:
            utterances : iterator or list of string corresponding to successive utterances
        '''
        self.utterances = []
        self.mentions = []
        self.n_sents = 0
        self.n_mentions = 0
        self.n_pairs = 0
        self.pairs_ant = None
        self.pairs_men = None
        try:
            self.mem.free(self.c)
        except KeyError:
            pass # we have not yet allocated the Mention_C array
        if utterances is not None:
            self.add_utterances(utterances)

    cdef add_utterances(self, utterances):
        '''
        Add utterances and process mentions in a spacy doc (an utterance)
        '''
        cdef:
            Pool mem = self.mem
            Mention_C* c

        if self.debug: print("Adding utterances", utterances)
        if isinstance(utterances, string_types):
            utterances = [utterances]
        mentions = []
        docs = self.nlp.pipe(utterances)
        for utt_idx, doc in enumerate(docs):
            m_span, n_spans = extract_mentions_spans(doc, use_no_coref_list=self.use_no_coref_list)
            mentions += list(Mention(span, utt_idx, self.n_sents) for span in m_span)
            self.n_mentions += n_spans
            self.utterances.append(doc)
            self.n_sents += len(list(doc.sents))
        self.mentions = sorted((m for m in mentions), key=lambda m: (m.span.root.i, m.span.start))
        c = <Mention_C*>mem.alloc(self.n_mentions, sizeof(Mention_C))
        for i, m in enumerate(self.mentions):
            c[i].entity_label = get_span_entity_label(m.span)
            c[i].span_start = m.span.start
            c[i].span_end = m.span.end
            c[i].utt_idx = m.utt_idx
            c[i].sent_idx = m.sent_idx
            c[i].mention_type = get_span_type(m.span)
            c[i].root_lower = (<Token>m.span.root).c.lex.lower
            c[i].span_lower = m.span.vocab.strings.add(m.span.text.lower())
            c[i].content_words.length = len(m.content_words)
            c[i].content_words.arr = <hash_t*>mem.alloc(len(m.content_words), sizeof(hash_t))
            for i, w in enumerate(m.content_words):
                c[i].content_words.arr[i] = m.span.vocab.strings.add(w)
        self.c = c
        self.set_mentions_features()
        self.set_candidate_pairs()

    cdef set_mentions_features(self):
        '''
        Compute features for the extracted mentions
        '''
        cdef:
            float [:] feat
            float [:] embed
        doc_embedding = self.embed_extractor.get_document_embedding(self.utterances) if self.embed_extractor is not None else None
        for i, mention in enumerate(self.mentions):
            print('set feats i', i)
            one_hot_type = numpy.zeros((4,), dtype='float32')
            one_hot_type[self.c[i].mention_type] = 1
            features_ = {"01_MentionType": self.c[i].mention_type,
                         "02_MentionLength": len(mention)-1,
                         "03_MentionNormLocation": (i/self.n_mentions),
                         "04_IsMentionNested": 1 if any((self.mentions[j] is not mention
                                                          and self.c[j].sent_idx == self.c[i].sent_idx
                                                          and self.c[j].span_start <= self.c[i].span_start
                                                          and self.c[j].span_end <= self.c[i].span_end)
                                                         for j in range(self.n_mentions)) else 0}
            print('features ok')
            features = numpy.concatenate([one_hot_type,
                                          encode_distance(features_["02_MentionLength"]),
                                          numpy.array(features_["03_MentionNormLocation"], ndmin=1, copy=False, dtype='float32'),
                                          numpy.array(features_["04_IsMentionNested"], ndmin=1, copy=False, dtype='float32')
                                         ], axis=0)
            (spans_embeddings_, words_embeddings_,
             spans_embeddings, words_embeddings) = self.embed_extractor.get_mention_embeddings(mention, doc_embedding)
            print('numpy arrays ok')
            mention.features_ = features_
            mention.features = features
            mention.spans_embeddings_ = spans_embeddings_
            mention.words_embeddings_ = words_embeddings_
            mention.embeddings = numpy.concatenate([spans_embeddings, words_embeddings], axis=0)
            feat = mention.features
            embed = mention.embeddings
            print('Storing pointers', i)
            self.c[i].features = <float*>&feat[0]    # Storing the pointer to the memoryview in C struct - Check https://github.com/nipy/dipy/issues/1435
            self.c[i].embeddings = <float*>&embed[0] # (not 100% clean, should be carefull to keep C struct and Python mentions in sync)
                                                     # Check also https://github.com/cython/cython/issues/1453

    def set_candidate_pairs(self, max_distance=50, max_distance_with_match=500, debug=False):
        '''
        Prepare numpy arrays of candidate antecedents for quickly iterating over pairs of mentions

        Arg:
            max_mention_distance : max distance between a mention and its antecedent
            max_mention_distance_string_match : max distance between a mention and
                its antecedent when there is a content word match
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

    ###################################
    ### DEBUGGING MENTION FEATURES ####
    ###################################

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
