# coding: utf8
"""data models and pre-processing for the coref algorithm"""

from __future__ import unicode_literals
from __future__ import print_function

import re
from six import string_types, integer_types

try:
    from itertools import izip_longest as zip_longest
except ImportError: # will be 3.x series
    from itertools import zip_longest

import spacy
import numpy as np

#########################
####### UTILITIES #######
#########################

NO_COREF_LIST = ["i", "me", "my", "you", "your"]

MENTION_TYPE = {"PRONOMINAL": 0, "NOMINAL": 1, "PROPER": 2, "LIST": 3}
MENTION_LABEL = {0: "PRONOMINAL", 1: "NOMINAL", 2: "PROPER", 3: "LIST"}

PROPERS_TAGS = ["NN", "NNS", "NNP", "NNPS"]
ACCEPTED_ENTS = ["PERSON", "NORP", "FACILITY", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LANGUAGE"]
WHITESPACE_PATTERN = r"\s+|_+"
UNKNOWN_WORD = "*UNK*"
NORMALIZE_DICT = {"/.": ".", "/?": "?", "-LRB-": "(", "-RRB-": ")",
                  "-LCB-": "{", "-RCB-": "}", "-LSB-": "[", "-RSB-": "]"}
DISTANCE_BINS = list(range(5)) + [5]*3 + [6]*8 + [7]*16 +[8]*32

def encode_distance(d):
    ''' Encode an integer as a (bined) one-hot numpy array '''
    dist_vect = np.zeros((11,))
    if d < 64:
        dist_vect[DISTANCE_BINS[d]] = 1
    else:
        dist_vect[9] = 1
    dist_vect[10] = min(float(d), 64.0) / 64.0
    return dist_vect

#########################
## MENTION EXTRACTION ###
#########################

def extract_mentions_spans(doc, use_no_coref_list=True, debug=False):
    '''
    Extract potential mentions from a spacy parsed Doc
    '''
    nouns_or_prp = re.compile(r"N.*|PRP.*|DT")
    det_or_comp = ["det", "compound", "appos"]
    nsubj_or_dep = ["nsubj", "dep"]
    conj_punct_pos = ["CCONJ", "PUNCT"]

    # Utility to remove bad endings
    def cleanup_endings(left, right, token):
        minchild_idx = min(left) if left else token.i
        maxchild_idx = max(right) if right else token.i
        # Clean up endings and begginging
        while maxchild_idx >= token.i and (doc[maxchild_idx].pos_ in conj_punct_pos or doc[maxchild_idx].lower_ == "'s"):
            if debug: print("Removing last token", doc[maxchild_idx].lower_, doc[maxchild_idx].tag_)
            maxchild_idx -= 1 # We don't want mentions finishing with 's or conjunctions/punctuation
        return min(minchild_idx, token.i), max(maxchild_idx, token.i)+1

    if debug: print('===== doc ====:', doc)
    for c in doc:
        if debug: print("🚧 span search:", c, "head:", c.head, "tag:", c.tag_, "pos:", c.pos_, "dep:", c.dep_)
    # Named entities
    mentions_spans = list(ent for ent in doc.ents if ent.label_ in ACCEPTED_ENTS)
    if debug: print("==-- ents:", list(((ent, ent.label_) for ent in mentions_spans)))

    # Pronouns and Noun phrases
    for token in doc:
        if debug: print("🚀 tok:", token, "tok.tag_", token.tag_)

        if use_no_coref_list and token.lower_ in NO_COREF_LIST:
            if debug: print("token in no_coref_list")
            continue
        if not nouns_or_prp.match(token.tag_) or token.dep_ in det_or_comp:
            if debug: print("not pronoun or no right dependency")
            continue

        # pronoun
        if re.match(r"PRP.*", token.tag_):
            if debug: print("PRP")
            endIdx = token.i + 1

            span = doc[token.i: endIdx]
            if not any((ent.start <= span.start and span.end <= ent.end for ent in doc.ents)):
                if debug: print("==-- not in entity store:", span)
                mentions_spans.append(span)

            # when pronoun is a part of conjunction (e.g., you and I)
            if token.n_rights > 0 or token.n_lefts > 0:
                span = doc[token.left_edge.i : token.right_edge.i+1]
                if debug: print("==-- in conj store:", span)
                mentions_spans.append(span)
            continue

        # Add NP mention
        if debug: print("NP", token.lower_)
        # Take care of 's
        if token.lower_ == "'s":
            if debug: print("'s detected")
            h = token.head
            while h.head is not h:
                if debug: print("token head", h, h.dep_, "head:", h.head)
                if h.dep_ == "nsubj":
                    minchild_idx = min((c.left_edge.i for c in doc if c.head == h.head and c.dep_ in nsubj_or_dep),
                                       default=token.i)
                    maxchild_idx = max((c.right_edge.i for c in doc if c.head == h.head and c.dep_ in nsubj_or_dep),
                                       default=token.i)
                    if debug: print("'s', i1:", doc[minchild_idx], " i2:", doc[maxchild_idx])
                    span = doc[minchild_idx : maxchild_idx+1]
                    if debug: print("==-- 's' store:", span)
                    mentions_spans.append(span)
                    break
                h = h.head
            continue

        # clean up
        for c in doc:
            if debug and c.head == token: print("🚧 token in span", c, c.head, c.dep_)
        left = list(c.left_edge.i for c in doc if c.head == token)
        if debug: print("left side", left)
        right = list(c.right_edge.i for c in doc if c.head == token)
        if debug: print("right side", right)
        start, end = cleanup_endings(left, right, token)
        if doc[start].lower_ == "'s":
            continue # we probably already have stored this mention
        span = doc[start:end]
        if debug: print("==-- full span store:", span)
        mentions_spans.append(span)
        if any(tok.dep_ == "conj" for tok in span):
            if debug: print("Conjunction found, storing first element separately")
            for c in doc:
                if c.head == token and c.dep_ != "conj":
                    if debug: print("left no conj", c, c.dep_, c.left_edge)
                    if debug: print("right no conj", c, c.dep_, c.right_edge)
            left_no_conj = list(c.left_edge.i for c in doc if c.head == token and c.dep_ != "conj")
            right_no_conj = list(c.right_edge.i for c in doc if c.head == token and c.dep_ != "conj")
            if debug: print("left side no conj", [doc[i] for i in left_no_conj])
            if debug: print("right side no conj", [doc[i] for i in right_no_conj])
            start, end = cleanup_endings(left_no_conj, right_no_conj, token)
            span = doc[start:end]
            if debug: print("==-- full span store:", span)
            mentions_spans.append(span)

    spans_set = set()
    cleaned_mentions_spans = []
    for spans in mentions_spans:
        if (spans.start, spans.end) not in spans_set:
            cleaned_mentions_spans.append(spans)
            spans_set.add((spans.start, spans.end))

    return cleaned_mentions_spans

#########################
####### CLASSES #########

class Mention(spacy.tokens.Span):
    '''
    A mention (possible anaphor) inherite from spacy Span class with additional informations
    '''
    def __new__(cls, span, mention_index, utterance_index, utterance_start_sent, speaker=None, gold_label=None, *args, **kwargs):
        # We need to override __new__ see http://cython.readthedocs.io/en/latest/src/userguide/special_methods.html
        obj = spacy.tokens.Span.__new__(cls, span.doc, span.start, span.end, *args, **kwargs)
        return obj

    def __init__(self, span, mention_index, utterance_index, utterances_start_sent, speaker=None, gold_label=None):
        self.index = mention_index
        self.utterance_index = utterance_index
        self.utterances_sent = utterances_start_sent + self._doc_sent_number()
        self.speaker = speaker
        self.gold_label = gold_label
        self.mention_type = self.find_type()

        self.spans_embeddings = None
        self.words_embeddings = None
        self.features = None

        self.features_ = None
        self.spans_embeddings_ = None
        self.words_embeddings_ = None

    @property
    def propers(self):
        ''' Set of nouns and proper nouns in the Mention'''
        return set(self.content_words)

    @property
    def entity_label(self):
        ''' Label of a detected named entity the Mention is nested in if any'''
        for ent in self.doc.ents:
            if ent.start <= self.start and self.end <= ent.end:
                return ent.label
        return None

    @property
    def in_entities(self):
        ''' Is the Mention nested in a detected named entity'''
        return self.entity_label is not None

    @property
    def content_words(self):
        ''' Returns an iterator of nouns/proper nouns in the Mention '''
        return (tok.lower_ for tok in self if tok.tag_ in PROPERS_TAGS)

    @property
    def embedding(self):
        return np.concatenate([self.spans_embeddings, self.words_embeddings], axis=0)

    def find_type(self):
        ''' Find the type of the Span '''
        conj = ["CC", ","]
        prp = ["PRP", "PRP$"]
        proper = ["NNP", "NNPS"]
        if any(t.tag_ in conj and t.ent_type_ not in ACCEPTED_ENTS for t in self):
            mention_type = MENTION_TYPE["LIST"]
        elif self.root.tag_ in prp:
            mention_type = MENTION_TYPE["PRONOMINAL"]
        elif self.root.ent_type_ in ACCEPTED_ENTS or self.root.tag_ in proper:
            mention_type = MENTION_TYPE["PROPER"]
        else:
            mention_type = MENTION_TYPE["NOMINAL"]
        return mention_type

    def _doc_sent_number(self):
        ''' Index of the sentence of the Mention in the current utterance'''
        for i, sent in enumerate(self.doc.sents):
            if sent == self.sent:
                return i
        return None

    def heads_agree(self, mention2):
        ''' Does the root of the Mention match the root of another Mention/Span'''
        # we allow same-type NEs to not match perfectly,
        # but rather one could be included in the other, e.g., "George" -> "George Bush"
        if (self.in_entities and mention2.in_entities and
                self.entity_label == mention2.entity_label and
                (self.root.lower_ in mention2.lower_ or mention2.root.lower_ in self.lower_)):
            return True
        return self.root.lower_ == mention2.root.lower_

    def exact_match(self, mention2):
        ''' Does the Mention lowercase text matches another Mention/Span lowercase text'''
        return self.lower_ == mention2.lower_

    def relaxed_match(self, mention2):
        ''' Does the nouns/proper nous in the Mention match another Mention/Span nouns/propers'''
        return not self.propers.isdisjoint(mention2.propers)

    def speaker_match_mention(self, mention2):
        # To take care of sentences like 'Larry said, "San Francisco is a city."': (id(Larry), id(San Francisco))
        # if document.speakerPairs.contains(new Pair<>(mention.mentionID, ant.mentionID)):
        #    return True
        if self.speaker is not None:
            return self.speaker.speaker_matches_mention(mention2, strict_match=False)
        return False

class Speaker:
    '''
    A speaker with its names, list of mentions and matching test functions
    '''
    def __init__(self, speaker_id, speaker_names=None):
        self.mentions = []
        self.speaker_id = speaker_id
        if speaker_names is None:
            self.speaker_names = [str(speaker_id)]
        elif isinstance(speaker_names, string_types):
            self.speaker_names = [speaker_names]
        elif len(speaker_names) > 1:
            self.speaker_names = speaker_names
        else:
            self.speaker_names = str(speaker_names)
        self.speaker_tokens = [tok.lower() for s in self.speaker_names for tok in re.split(WHITESPACE_PATTERN, s)]

    def __str__(self):
        return '{} <names> {}'.format(self.speaker_id, self.speaker_names)

    def add_mention(self, mention):
        ''' Add a Mention of the Speaker'''
        self.mentions.append(mention)

    def contain_mention(self, mention):
        ''' Does the Speaker contains a Mention'''
        return mention in self.mentions

    def contain_string(self, string):
        ''' Does the Speaker names contains a string'''
        return any(re.sub(WHITESPACE_PATTERN, "", string.lower()) == re.sub(WHITESPACE_PATTERN, "", s.lower())
                   for s in self.speaker_names)

    def contain_token(self, token):
        ''' Does the Speaker names contains a token (word)'''
        return any(token.lower() == tok.lower() for tok in self.speaker_tokens)

    def speaker_matches_mention(self, mention, strict_match=False):
        ''' Does a Mention matches the speaker names'''
         # Got info about this speaker
        if self.contain_mention(mention):
            return True
        if strict_match:
            if self.contain_string(mention):
                self.mentions.append(mention)
                return True
        else:
            # test each token in the speaker names
            if not mention.root.tag_.startswith("NNP"):
                return False
            if self.contain_token(mention.root.lower_):
                self.mentions.append(mention)
                return True
        return False

class EmbeddingExtractor:
    '''
    Compute words embedding features for mentions
    '''
    def __init__(self, model_path):
        self.average_mean, self.static_embeddings, self.static_voc = self.load_embeddings_from_file(model_path + "static_word")
        _, self.tuned_embeddings, self.tuned_voc = self.load_embeddings_from_file(model_path + "tuned_word")
        self.fallback = self.static_embeddings.get(UNKNOWN_WORD)

        self.shape = self.static_embeddings[UNKNOWN_WORD].shape
        shape2 = self.tuned_embeddings[UNKNOWN_WORD].shape
        assert self.shape == shape2

    @staticmethod
    def load_embeddings_from_file(name):
        embeddings = {}
        voc = {}
        mat = np.load(name+"_embeddings.npy")
        average_mean = np.average(mat, axis=0, weights=np.sum(mat, axis=1))
        with open(name+"_vocabulary.txt") as f:
            for i, line in enumerate(f):
                embeddings[line.strip()] = mat[i, :]
                voc[line.strip()] = i
        return average_mean, embeddings, voc

    @staticmethod
    def normalize_word(w):
        if w is None:
            return "<missing>"
        elif w.lower_ in NORMALIZE_DICT:
            return NORMALIZE_DICT[w.lower_]
        return w.lower_.replace("\\d", "0")

    def get_document_embedding(self, utterances_list):
        ''' Embedding for the document '''
    #    We could also use this: embed_vector = np.copy(self.average_mean)#np.zeros(self.shape)
    #    return embed_vector
        embed_vector = np.zeros(self.shape)
        for utt in utterances_list:
            embed_vector += self.get_average_embedding(utt)
        return embed_vector/max(len(utterances_list), 1)

    def get_word_embedding(self, word, static=False):
        ''' Embedding for a single word (tuned if possible, otherwise static) '''
        norm_word = self.normalize_word(word)
        if static:
            if norm_word in self.static_embeddings:
                word = norm_word
                embed = self.static_embeddings.get(norm_word)
            else:
                word = UNKNOWN_WORD
                embed = self.fallback
        else:
            if norm_word in self.tuned_embeddings:
                word = norm_word
                embed = self.tuned_embeddings.get(norm_word)
            elif norm_word in self.static_embeddings:
                word = norm_word
                embed = self.static_embeddings.get(norm_word)
            else:
                word = UNKNOWN_WORD
                embed = self.fallback
        return word, embed

    def get_word_in_sentence(self, word_idx, sentence):
        ''' Embedding for a word in a sentence '''
        if word_idx < sentence.start or word_idx >= sentence.end:
            return self.get_word_embedding(None)
        return self.get_word_embedding(sentence.doc[word_idx])

    def get_average_embedding(self, token_list):
        ''' Embedding for a list of words '''
        embed_vector = np.zeros(self.shape) #We could also use np.copy(self.average_mean)
        word_list = []
        '''for tok in token_list:
            if tok.lower_ not in [".", "!", "?"]:
                word, embed = self.get_word_embedding(tok, static=True)
                embed_vector += embed
                word_list.append(word)
        return word_list, (embed_vector/max(len(word_list), 1))
        '''
    def get_mention_embeddings(self, mention, doc_embedding):
        ''' Embedding for a mention '''
        sent = mention.sent
        mention_lefts = mention.doc[max(mention.start-5, sent.start):mention.start]
        mention_rights = mention.doc[mention.end:min(mention.end+5, sent.end)]
        head = mention.root.head
        spans = [self.get_average_embedding(mention),
                 self.get_average_embedding(mention_lefts),
                 self.get_average_embedding(mention_rights),
                 self.get_average_embedding(sent),
                 (str(doc_embedding[0:8]) + "...", doc_embedding)]
        words = [self.get_word_embedding(mention.root),
                 self.get_word_embedding(mention[0]),
                 self.get_word_embedding(mention[-1]),
                 self.get_word_in_sentence(mention.start-1, sent),
                 self.get_word_in_sentence(mention.end, sent),
                 self.get_word_in_sentence(mention.start-2, sent),
                 self.get_word_in_sentence(mention.end+1, sent),
                 self.get_word_embedding(head)]
        spans_embeddings_ = {"Mention": spans[0][0],
                             "MentionLeft": spans[1][0],
                             "MentionRight": spans[2][0],
                             "Sentence": spans[3][0],
                             "Doc": spans[4][0]}
        words_embeddings_ = {"MentionHead": words[0][0],
                             "MentionFirstWord": words[1][0],
                             "MentionLastWord": words[2][0],
                             "PreviousWord": words[3][0],
                             "NextWord": words[4][0],
                             "SecondPreviousWord": words[5][0],
                             "SecondNextWord": words[6][0],
                             "MentionRootHead": words[7][0]}
        return (spans_embeddings_,
                words_embeddings_,
                np.concatenate(list(em[1] for em in spans), axis=0)[: np.newaxis],
                np.concatenate(list(em[1] for em in words), axis=0)[: np.newaxis])

    def lda_algi(self, tokens):
        embed_vector = np.zeros(self.shape)
        mention = []
        for i in mention.enumerate():
            mention.append(tokens)
        word_list = []
        for tok in tokens:
            if tok.lower_ not in [".", "!", "?"]:
                new_word, embed = self.get_word_embedding(tok, static=True)
                embed_vector += embed_vector
                word_list.append(new_word)
                #words = [self.get_word_embedding(extract_mentions_spans(mention.start-1))]
        return word_list, (embed_vector/max(len(word_list), 1))




class Data:
    '''
    Main data class: encapsulate list of utterances, mentions and speakers
    Process utterances to extract mentions and pre-compute mentions features
    '''
    def __init__(self, nlp, model_path=None, conll=None, utterances=None, utterances_speaker=None,
                 speakers_names=None, use_no_coref_list=True, consider_speakers=False ,debug=False):
        self.nlp = nlp
        self.use_no_coref_list = use_no_coref_list
        self.utterances = []
        self.utterances_speaker = []
        self.last_utterances_loaded = None
        self.mentions = []
        self.speakers = {}
        self.n_sents = 0
        self.debug = debug
        self.consider_speakers = consider_speakers

        self.genre_ = conll
        if conll is not None:
            self.genre = np.zeros((7,))
            genres = {"bc": 0, "bn": 1, "mz": 2, "nw": 3, "pt": 4, "tc": 5, "wb": 6}
            #. We take broadcast conversations to use speaker infos
            self.genre[genres[conll]] = 1
        else:
            self.genre = np.array(0, ndmin=1, copy=False)
        
        if model_path is not None:
            self.embed_extractor = EmbeddingExtractor(model_path)
            assert self.embed_extractor.shape is not None
            self.doc_embedding = np.zeros(self.embed_extractor.shape)
        else:
            self.embed_extractor = None
            self.doc_embedding = None

        if utterances:
            self.add_utterances(utterances, utterances_speaker, speakers_names)

    def __str__(self):
        return '<utterances, speakers> \n {}\n<mentions> \n {}' \
                .format('\n '.join(str(i) + " " + str(s) for i, s in zip(self.utterances, self.utterances_speaker)),
                        '\n '.join(str(i) + " " + str(i.speaker) for i in self.mentions))

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
    
    def set_utterances(self, utterances, utterances_speaker=None, speakers_names=None):
        self.utterances = []
        self.utterances_speaker = []
        self.mentions = []
        self.speakers = {}
        self.n_sents = 0
        if utterances:
            self.add_utterances(utterances, utterances_speaker, speakers_names)

    def add_utterances(self, utterances, utterances_speaker=None, speakers_names=None):
        '''
        Add utterances to the utterance list and build mention list for these utterances

        Arg:
            utterances : iterator or list of string corresponding to successive utterances
            utterances_speaker : iterator or list of speaker id for each utterance.
                If not provided, assume two speakers speaking alternatively.
                if utterances and utterances_speaker are not of the same length padded with None
            speakers_names : dictionnary of list of acceptable speaker names for each speaker id
        Return:
            List of indexes of added utterances in the docs
        '''
        if self.debug: print("Adding utterances", utterances)
        if isinstance(utterances, string_types):
            utterances = [utterances]
        if utterances_speaker is None:
            if self.debug: print("No utterance speaker indication")
            a = -1
            if self.utterances_speaker and isinstance(self.utterances_speaker[-1].speaker_id, integer_types):
                a = self.utterances_speaker[-1].speaker_id
            utterances_speaker = ((i + a + 1) % 2 for i in range(len(utterances)))
        utterances_index = []
        utt_start = len(self.utterances)
        for utt_index, (utterance, speaker_id) in enumerate(zip_longest(utterances, utterances_speaker)):
            if utterance is None:
                break
            # Pipe currently broken in spacy 2 alpha
            # Also, spacy 2 currently throws an exception on empty strings
            try:
                doc = self.nlp(utterance)
            except IndexError:
                doc = self.nlp(u" ")
                if self.debug: print("Empty string")
            if speaker_id not in self.speakers:
                speaker_name = speakers_names.get(speaker_id, None) if speakers_names else None
                self.speakers[speaker_id] = Speaker(speaker_id, speaker_name)
            self._extract_mentions(doc, utt_start + utt_index, self.n_sents, self.speakers[speaker_id])
            utterances_index.append(utt_start + utt_index)
            self.utterances.append(doc)
            self.utterances_speaker.append(self.speakers[speaker_id])
            self.n_sents += len(list(doc.sents))

        self.set_mentions_features()
        self.last_utterances_loaded = utterances_index

    ###################################
    ## FEATURES MENTIONS EXTRACTION ###
    ###################################

    def _extract_mentions(self, doc, utterance_index, n_sents, speaker):
        '''
        Extract mentions in a spacy doc (an utterance)
        '''
        mentions_spans = extract_mentions_spans(doc, use_no_coref_list=self.use_no_coref_list)
        processed_spans = sorted((m for m in mentions_spans), key=lambda m: (m.root.i, m.start))
        n_mentions = len(self.mentions)
        for mention_index, span in enumerate(processed_spans):
            self.mentions.append(Mention(span, mention_index + n_mentions,
                                             utterance_index, n_sents, speaker))

    def set_mentions_features(self):
        '''
        Compute features for the extracted mentions
        '''
        #TODO : we should probably update doc embedding here (not used currently)
        for mention in self.mentions:
            one_hot_type = np.zeros((4,))
            one_hot_type[mention.mention_type] = 1
            features_ = {"MentionType": mention.mention_type,
                         "MentionLength": len(mention)-1,
                         "MentionNormLocation": (mention.index)/len(self.mentions),
                         "IsMentionNested": 1 if any((m is not mention
                                                          and m.utterances_sent == mention.utterances_sent
                                                          and m.start <= mention.start
                                                          and mention.end <= m.end)
                                                         for m in self.mentions) else 0}
            features = np.concatenate([one_hot_type,
                                       encode_distance(features_["MentionLength"]),
                                       np.array(features_["MentionNormLocation"], ndmin=1, copy=False),
                                       np.array(features_["IsMentionNested"], ndmin=1, copy=False)
                                      ], axis=0)
            spans_embeddings_, words_embeddings_, spans_embeddings, words_embeddings = self.embed_extractor.get_mention_embeddings(mention, self.doc_embedding)
            mention.features_ = features_
            mention.features = features
            mention.spans_embeddings = spans_embeddings
            mention.spans_embeddings_ = spans_embeddings_
            mention.words_embeddings = words_embeddings
            mention.words_embeddings_ = words_embeddings_

    def get_single_mention_features(self, mention):
        ''' Features for anaphoricity test (signle mention features + genre if conll)'''
        features_ = mention.features_
        features_["DocGenre"] = self.genre_
        return (features_, np.concatenate([mention.features, self.genre], axis=0))

    def get_pair_mentions_features(self, m1, m2):
        ''' Features for pair of mentions (same speakers, speaker mentioned, string match)'''
        features_ = {"SameSpeaker": 1 if self.consider_speakers and m1.speaker == m2.speaker else 0,
                     "AntMatchMentionSpeaker": 1 if self.consider_speakers and m2.speaker_match_mention(m1) else 0,
                     "MentionMatchSpeaker": 1 if self.consider_speakers and m1.speaker_match_mention(m2) else 0,
                     "HeadsAgree": 1 if m1.heads_agree(m2) else 0,
                     "ExactStringMatch": 1 if m1.exact_match(m2) else 0,
                     "RelaxedStringMatch": 1 if m1.relaxed_match(m2) else 0,
                     "SentenceDistance": m2.utterances_sent - m1.utterances_sent,
                     "MentionDistance": 1, #m2.index - m1.index - 1,
                     "Overlapping": 1 if (m1.utterances_sent == m2.utterances_sent and m1.end > m2.start) else 0,
                     "M1Features": m1.features_,
                     "M2Features": m2.features_,
                     "DocGenre": self.genre_}
        pairwise_features = [np.array([features_["SameSpeaker"],
                                       features_["AntMatchMentionSpeaker"],
                                       features_["MentionMatchSpeaker"],
                                       features_["HeadsAgree"],
                                       features_["ExactStringMatch"],
                                       features_["RelaxedStringMatch"]]),
                             encode_distance(features_["SentenceDistance"]),
                             encode_distance(features_["MentionDistance"]),
                             np.array(features_["Overlapping"], ndmin=1),
                             m1.features,
                             m2.features,
                             self.genre]
        return (features_, np.concatenate(pairwise_features, axis=0))

    ###################################
    ###### ITERATOR OVER MENTIONS #####
    ###################################

    def get_candidate_mentions(self, last_utterances_added=False):
        '''
        Return iterator over indexes of mentions in a list of utterances if specified
        '''
        if last_utterances_added:
            for i, mention in enumerate(self.mentions):
                if self.debug: print("🤣", i, mention, "utterance index", mention.utterance_index)
                if mention.utterance_index in self.last_utterances_loaded:
                    yield i
        else:
            iterator = range(len(self.mentions))
            for i in iterator:
                yield i

    def get_candidate_pairs(self, mentions=None, max_distance=50, max_distance_with_match=500):
        '''
        Yield tuples of mentions, dictionnary of candidate antecedents for the mention

        Arg:
            mentions: an iterator over mention indexes (as returned by get_candidate_mentions)
            max_mention_distance : max distance between a mention and its antecedent
            max_mention_distance_string_match : max distance between a mention and
                its antecedent when there is a proper noun match
        '''
        if mentions is None:
            mentions = range(len(self.mentions))

        word_to_mentions = {}
        for i in mentions:
            for tok in self.mentions[i].content_words:
                if not tok in word_to_mentions:
                    word_to_mentions[tok] = [i]
                else:
                    word_to_mentions[tok].append(i)

        for i in mentions:
            antecedents = set(range(max(0, i - max_distance), i))
            for tok in self.mentions[i].content_words:
                with_string_match = word_to_mentions.get(tok, None)
                for match_idx in with_string_match:
                    if match_idx < i and match_idx >= i - max_distance_with_match:
                        antecedents.add(match_idx)
            if antecedents:
                yield i, antecedents
