"""data models and pre-processing for the coref algorithm"""

import re
import io
from six import string_types, integer_types
from spacy.tokens import Span, Token

from neuralcoref.train.compat import unicode_
from neuralcoref.train.utils import encode_distance, parallel_process

try:
    from itertools import izip_longest as zip_longest
except ImportError:  # will be 3.x series
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
ACCEPTED_ENTS = [
    "PERSON",
    "NORP",
    "FACILITY",
    "ORG",
    "GPE",
    "LOC",
    "PRODUCT",
    "EVENT",
    "WORK_OF_ART",
    "LANGUAGE",
]
WHITESPACE_PATTERN = r"\s+|_+"
UNKNOWN_WORD = "*UNK*"
MISSING_WORD = "<missing>"
MAX_ITER = 100

#########################
## MENTION EXTRACTION ###
#########################


def extract_mentions_spans(doc, blacklist, debug=False):
    """
    Extract potential mentions from a spacy parsed Doc
    """
    if debug:
        print("===== doc ====:", doc)
    for c in doc:
        if debug:
            print(
                "🚧 span search:",
                c,
                "head:",
                c.head,
                "tag:",
                c.tag_,
                "pos:",
                c.pos_,
                "dep:",
                c.dep_,
            )
    # Named entities
    mentions_spans = list(ent for ent in doc.ents if ent.label_ in ACCEPTED_ENTS)

    if debug:
        print("==-- ents:", list(((ent, ent.label_) for ent in mentions_spans)))
    for spans in parallel_process(
        [{"doc": doc, "span": sent, "blacklist": blacklist} for sent in doc.sents],
        _extract_from_sent,
        use_kwargs=True,
        front_num=0,
    ):
        mentions_spans = mentions_spans + spans
    spans_set = set()
    cleaned_mentions_spans = []
    for spans in mentions_spans:
        if spans.end > spans.start and (spans.start, spans.end) not in spans_set:
            cleaned_mentions_spans.append(spans)
            spans_set.add((spans.start, spans.end))

    return cleaned_mentions_spans


def _extract_from_sent(doc, span, blacklist=True, debug=False):
    """
    Extract Pronouns and Noun phrases mentions from a spacy Span
    """
    keep_tags = re.compile(r"N.*|PRP.*|DT|IN")
    leave_dep = ["det", "compound", "appos"]
    keep_dep = ["nsubj", "dobj", "iobj", "pobj"]
    nsubj_or_dep = ["nsubj", "dep"]
    conj_or_prep = ["conj", "prep"]
    remove_pos = ["CCONJ", "SCONJ", "INTJ", "ADP"]
    lower_not_end = ["'s", ",", ".", "!", "?", ":", ";"]

    # Utility to remove bad endings
    def cleanup_endings(left, right, token):
        minchild_idx = min(left + [token.i]) if left else token.i
        maxchild_idx = max(right + [token.i]) if right else token.i
        # Clean up endings and begginging
        while maxchild_idx >= minchild_idx and (
            doc[maxchild_idx].pos_ in remove_pos
            or doc[maxchild_idx].lower_ in lower_not_end
        ):
            if debug:
                print(
                    "Removing last token",
                    doc[maxchild_idx].lower_,
                    doc[maxchild_idx].tag_,
                )
            maxchild_idx -= (
                1
            )  # We don't want mentions finishing with 's or conjunctions/punctuation
        while minchild_idx <= maxchild_idx and (
            doc[minchild_idx].pos_ in remove_pos
            or doc[minchild_idx].lower_ in lower_not_end
        ):
            if debug:
                print(
                    "Removing first token",
                    doc[minchild_idx].lower_,
                    doc[minchild_idx].tag_,
                )
            minchild_idx += (
                1
            )  # We don't want mentions starting with 's or conjunctions/punctuation
        return minchild_idx, maxchild_idx + 1

    mentions_spans = []
    for token in span:
        if debug:
            print(
                "🚀 tok:",
                token,
                "tok.tag_:",
                token.tag_,
                "tok.pos_:",
                token.pos_,
                "tok.dep_:",
                token.dep_,
            )

        if blacklist and token.lower_ in NO_COREF_LIST:
            if debug:
                print("token in no_coref_list")
            continue
        if (
            not keep_tags.match(token.tag_) or token.dep_ in leave_dep
        ) and not token.dep_ in keep_dep:
            if debug:
                print("not pronoun or no right dependency")
            continue

        # pronoun
        if re.match(r"PRP.*", token.tag_):
            if debug:
                print("PRP")
            endIdx = token.i + 1

            span = doc[token.i : endIdx]
            if debug:
                print("==-- PRP store:", span)
            mentions_spans.append(span)

            # when pronoun is a part of conjunction (e.g., you and I)
            if token.n_rights > 0 or token.n_lefts > 0:
                span = doc[token.left_edge.i : token.right_edge.i + 1]
                if debug:
                    print("==-- in conj store:", span)
                mentions_spans.append(span)
            continue

        # Add NP mention
        if debug:
            print("NP or IN:", token.lower_)
            if token.tag_ == "IN":
                print("IN tag")
        # Take care of 's
        if token.lower_ == "'s":
            if debug:
                print("'s detected")
            h = token.head
            j = 0
            while h.head.i != h.i and j < MAX_ITER:
                if debug:
                    print("token head:", h, h.dep_, "head:", h.head)
                    print(id(h.head), id(h))
                if h.dep_ == "nsubj":
                    minchild_idx = min(
                        (
                            c.left_edge.i
                            for c in doc
                            if c.head.i == h.head.i and c.dep_ in nsubj_or_dep
                        ),
                        default=token.i,
                    )
                    maxchild_idx = max(
                        (
                            c.right_edge.i
                            for c in doc
                            if c.head.i == h.head.i and c.dep_ in nsubj_or_dep
                        ),
                        default=token.i,
                    )
                    if debug:
                        print("'s', i1:", doc[minchild_idx], " i2:", doc[maxchild_idx])
                    span = doc[minchild_idx : maxchild_idx + 1]
                    if debug:
                        print("==-- 's' store:", span)
                    mentions_spans.append(span)
                    break
                h = h.head
                j += 1
            assert j != MAX_ITER
            continue

        # clean up
        for c in doc:
            if debug and c.head.i == token.i:
                print("🚧 token in span:", c, "- head & dep:", c.head, c.dep_)
        left = list(c.left_edge.i for c in doc if c.head.i == token.i)
        right = list(c.right_edge.i for c in doc if c.head.i == token.i)
        if (
            token.tag_ == "IN"
            and token.dep_ == "mark"
            and len(left) == 0
            and len(right) == 0
        ):
            left = list(c.left_edge.i for c in doc if c.head.i == token.head.i)
            right = list(c.right_edge.i for c in doc if c.head.i == token.head.i)
        if debug:
            print("left side:", left)
            print("right side:", right)
            minchild_idx = min(left) if left else token.i
            maxchild_idx = max(right) if right else token.i
            print("full span:", doc[minchild_idx : maxchild_idx + 1])
        start, end = cleanup_endings(left, right, token)
        if start == end:
            continue
        if doc[start].lower_ == "'s":
            continue  # we probably already have stored this mention
        span = doc[start:end]
        if debug:
            print("cleaned endings span:", doc[start:end])
            print("==-- full span store:", span)
        mentions_spans.append(span)
        if debug and token.tag_ == "IN":
            print("IN tag")
        if any(tok.dep_ in conj_or_prep for tok in span):
            if debug:
                print("Conjunction found, storing first element separately")
            for c in doc:
                if c.head.i == token.i and c.dep_ not in conj_or_prep:
                    if debug:
                        print("left no conj:", c, "dep & edge:", c.dep_, c.left_edge)
                    if debug:
                        print("right no conj:", c, "dep & edge:", c.dep_, c.right_edge)
            left_no_conj = list(
                c.left_edge.i
                for c in doc
                if c.head.i == token.i and c.dep_ not in conj_or_prep
            )
            right_no_conj = list(
                c.right_edge.i
                for c in doc
                if c.head.i == token.i and c.dep_ not in conj_or_prep
            )
            if debug:
                print("left side no conj:", [doc[i] for i in left_no_conj])
            if debug:
                print("right side no conj:", [doc[i] for i in right_no_conj])
            start, end = cleanup_endings(left_no_conj, right_no_conj, token)
            if start == end:
                continue
            span = doc[start:end]
            if debug:
                print("==-- full span store:", span)
            mentions_spans.append(span)
    if debug:
        print("mentions_spans inside", mentions_spans)
    return mentions_spans


#########################
####### CLASSES #########


class Mention(spacy.tokens.Span):
    """
    A mention (possible anaphor) inherite from spacy Span class with additional informations
    """

    def __new__(
        cls,
        span,
        mention_index,
        utterance_index,
        utterance_start_sent,
        speaker=None,
        gold_label=None,
        *args,
        **kwargs,
    ):
        # We need to override __new__ see http://cython.readthedocs.io/en/latest/src/userguide/special_methods.html
        obj = spacy.tokens.Span.__new__(
            cls, span.doc, span.start, span.end, *args, **kwargs
        )
        return obj

    def __init__(
        self,
        span,
        mention_index,
        utterance_index,
        utterances_start_sent,
        speaker=None,
        gold_label=None,
    ):
        """
        Arguments:
            span (spaCy Span): the spaCy span from which creating the Mention object
            mention_index (int): index of the Mention in the Document
            utterance_index (int): index of the utterance of the Mention in the Document
            utterances_start_sent (int): index of the first sentence of the utterance of the Mention in the Document
                (an utterance can comprise several sentences)
            speaker (Speaker): the speaker of the mention
            gold_label (anything): a gold label associated to the Mention (for training)
        """
        self.index = mention_index
        self.utterance_index = utterance_index
        self.utterances_sent = utterances_start_sent + self._get_doc_sent_number()
        self.speaker = speaker
        self.gold_label = gold_label
        self.spans_embeddings = None
        self.words_embeddings = None
        self.features = None

        self.features_ = None
        self.spans_embeddings_ = None
        self.words_embeddings_ = None

        self.mention_type = self._get_type()
        self.propers = set(self.content_words)
        self.entity_label = self._get_entity_label()
        self.in_entities = self._get_in_entities()

    def _get_entity_label(self):
        """ Label of a detected named entity the Mention is nested in if any"""
        for ent in self.doc.ents:
            if ent.start <= self.start and self.end <= ent.end:
                return ent.label
        return None

    def _get_in_entities(self):
        """ Is the Mention nested in a detected named entity"""
        return self.entity_label is not None

    def _get_type(self):
        """ Find the type of the Span """
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

    def _get_doc_sent_number(self):
        """ Index of the sentence of the Mention in the current utterance"""
        for i, s in enumerate(self.doc.sents):
            if s == self.sent:
                return i
        return None

    @property
    def content_words(self):
        """ Returns an iterator of nouns/proper nouns in the Mention """
        return (tok.lower_ for tok in self if tok.tag_ in PROPERS_TAGS)

    @property
    def embedding(self):
        return np.concatenate([self.spans_embeddings, self.words_embeddings], axis=0)

    def heads_agree(self, mention2):
        """ Does the root of the Mention match the root of another Mention/Span"""
        # we allow same-type NEs to not match perfectly,
        # but rather one could be included in the other, e.g., "George" -> "George Bush"
        if (
            self.in_entities
            and mention2.in_entities
            and self.entity_label == mention2.entity_label
            and (
                self.root.lower_ in mention2.lower_
                or mention2.root.lower_ in self.lower_
            )
        ):
            return True
        return self.root.lower_ == mention2.root.lower_

    def exact_match(self, mention2):
        """ Does the Mention lowercase text matches another Mention/Span lowercase text"""
        return self.lower_ == mention2.lower_

    def relaxed_match(self, mention2):
        """ Does the nouns/proper nous in the Mention match another Mention/Span nouns/propers"""
        return not self.propers.isdisjoint(mention2.propers)

    def speaker_match_mention(self, mention2):
        # To take care of sentences like 'Larry said, "San Francisco is a city."': (id(Larry), id(San Francisco))
        # if document.speakerPairs.contains(new Pair<>(mention.mentionID, ant.mentionID)):
        #    return True
        if self.speaker is not None:
            return self.speaker.speaker_matches_mention(mention2, strict_match=False)
        return False


class Speaker(object):
    """
    A speaker with its names, list of mentions and matching test functions
    """

    def __init__(self, speaker_id, speaker_names=None):
        self.mentions = []
        self.speaker_id = speaker_id
        if speaker_names is None:
            self.speaker_names = [unicode_(speaker_id)]
        elif isinstance(speaker_names, string_types):
            self.speaker_names = [speaker_names]
        elif len(speaker_names) > 1:
            self.speaker_names = speaker_names
        else:
            self.speaker_names = unicode_(speaker_names)
        self.speaker_tokens = [
            tok.lower()
            for s in self.speaker_names
            for tok in re.split(WHITESPACE_PATTERN, s)
        ]

    def __str__(self):
        return f"{self.speaker_id} <names> {self.speaker_names}"

    def add_mention(self, mention):
        """ Add a Mention of the Speaker"""
        self.mentions.append(mention)

    def contain_mention(self, mention):
        """ Does the Speaker contains a Mention"""
        return mention in self.mentions

    def contain_string(self, string):
        """ Does the Speaker names contains a string"""
        return any(
            re.sub(WHITESPACE_PATTERN, "", string.lower())
            == re.sub(WHITESPACE_PATTERN, "", s.lower())
            for s in self.speaker_names
        )

    def contain_token(self, token):
        """ Does the Speaker names contains a token (word)"""
        return any(token.lower() == tok.lower() for tok in self.speaker_tokens)

    def speaker_matches_mention(self, mention, strict_match=False):
        """ Does a Mention matches the speaker names"""
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
    """
    Compute words embedding features for mentions
    """

    def __init__(self, pretrained_model_path):
        _, self.static_embeddings, self.stat_idx, self.stat_voc = self.load_embeddings_from_file(
            pretrained_model_path + "static_word"
        )
        _, self.tuned_embeddings, self.tun_idx, self.tun_voc = self.load_embeddings_from_file(
            pretrained_model_path + "tuned_word"
        )
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
        mat = np.load(name + "_embeddings.npy")
        average_mean = np.average(mat, axis=0, weights=np.sum(mat, axis=1))
        with io.open(name + "_vocabulary.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                embeddings[line.strip()] = mat[i, :]
                voc_to_idx[line.strip()] = i
                idx_to_voc.append(line.strip())
        return average_mean, embeddings, voc_to_idx, idx_to_voc

    @staticmethod
    def normalize_word(w):
        if w is None:
            return MISSING_WORD
        return re.sub(r"\d", "0", w.lower_)

    def get_document_embedding(self, utterances_list):
        """ Embedding for the document """
        #    We could also use this: embed_vector = np.copy(self.average_mean)#np.zeros(self.shape)
        #    return embed_vector
        embed_vector = np.zeros(self.shape)
        for utt in utterances_list:
            _, utt_embed = self.get_average_embedding(utt)
            embed_vector += utt_embed
        return embed_vector / max(len(utterances_list), 1)

    def get_stat_word(self, word):
        if word in self.static_embeddings:
            return word, self.static_embeddings.get(word)
        else:
            return UNKNOWN_WORD, self.fallback

    def get_word_embedding(self, word, static=False):
        """ Embedding for a single word (tuned if possible, otherwise static) """
        norm_word = self.normalize_word(word)
        if static:
            return self.get_stat_word(norm_word)
        else:
            if norm_word in self.tuned_embeddings:
                return norm_word, self.tuned_embeddings.get(norm_word)
            else:
                return self.get_stat_word(norm_word)

    def get_word_in_sentence(self, word_idx, sentence):
        """ Embedding for a word in a sentence """
        if word_idx < sentence.start or word_idx >= sentence.end:
            return self.get_word_embedding(None)
        return self.get_word_embedding(sentence.doc[word_idx])

    def get_average_embedding(self, token_list):
        """ Embedding for a list of words """
        embed_vector = np.zeros(
            self.shape
        )  # We could also use np.copy(self.average_mean)
        word_list = []
        for tok in token_list:
            if tok.lower_ not in [".", "!", "?"]:
                word, embed = self.get_word_embedding(tok, static=True)
                embed_vector += embed
                word_list.append(word)
        return word_list, (embed_vector / max(len(word_list), 1))

    def get_mention_embeddings(self, mention, doc_embedding):
        """ Get span (averaged) and word (single) embeddings of a mention """
        st = mention.sent
        mention_lefts = mention.doc[max(mention.start - 5, st.start) : mention.start]
        mention_rights = mention.doc[mention.end : min(mention.end + 5, st.end)]
        head = mention.root.head
        spans = [
            self.get_average_embedding(mention),
            self.get_average_embedding(mention_lefts),
            self.get_average_embedding(mention_rights),
            self.get_average_embedding(st),
            (unicode_(doc_embedding[0:8]) + "...", doc_embedding),
        ]
        words = [
            self.get_word_embedding(mention.root),
            self.get_word_embedding(mention[0]),
            self.get_word_embedding(mention[-1]),
            self.get_word_in_sentence(mention.start - 1, st),
            self.get_word_in_sentence(mention.end, st),
            self.get_word_in_sentence(mention.start - 2, st),
            self.get_word_in_sentence(mention.end + 1, st),
            self.get_word_embedding(head),
        ]
        spans_embeddings_ = {
            "00_Mention": spans[0][0],
            "01_MentionLeft": spans[1][0],
            "02_MentionRight": spans[2][0],
            "03_Sentence": spans[3][0],
            "04_Doc": spans[4][0],
        }
        words_embeddings_ = {
            "00_MentionHead": words[0][0],
            "01_MentionFirstWord": words[1][0],
            "02_MentionLastWord": words[2][0],
            "03_PreviousWord": words[3][0],
            "04_NextWord": words[4][0],
            "05_SecondPreviousWord": words[5][0],
            "06_SecondNextWord": words[6][0],
            "07_MentionRootHead": words[7][0],
        }
        return (
            spans_embeddings_,
            words_embeddings_,
            np.concatenate([em[1] for em in spans], axis=0),
            np.concatenate([em[1] for em in words], axis=0),
        )


class Document(object):
    """
    Main data class: encapsulate list of utterances, mentions and speakers
    Process utterances to extract mentions and pre-compute mentions features
    """

    def __init__(
        self,
        nlp,
        utterances=None,
        utterances_speaker=None,
        speakers_names=None,
        blacklist=False,
        consider_speakers=False,
        model_path=None,
        embedding_extractor=None,
        conll=None,
        debug=False,
    ):
        """
        Arguments:
            nlp (spaCy Language Class): A spaCy Language Class for processing the text input
            utterances: utterance(s) to load already see self.add_utterances()
            utterances_speaker: speaker(s) of utterance(s) to load already see self.add_utterances()
            speakers_names: speaker(s) of utterance(s) to load already see self.add_utterances()
            blacklist (boolean): use a list of term for which coreference is not preformed
            consider_speakers (boolean): consider speakers informations
            pretrained_model_path (string): Path to a folder with pretrained word embeddings
            embedding_extractor (EmbeddingExtractor): Use a pre-loaded word embeddings extractor
            conll (string): If training on coNLL data: identifier of the document type
            debug (boolean): print debug informations
        """
        self.nlp = nlp
        self.blacklist = blacklist
        self.utterances = []
        self.utterances_speaker = []
        self.last_utterances_loaded = []
        self.mentions = []
        self.speakers = {}
        self.n_sents = 0
        self.debug = debug
        self.consider_speakers = consider_speakers or conll is not None

        self.genre_, self.genre = self.set_genre(conll)

        if model_path is not None and embedding_extractor is None:
            self.embed_extractor = EmbeddingExtractor(model_path)
        elif embedding_extractor is not None:
            self.embed_extractor = embedding_extractor
        else:
            self.embed_extractor = None

        if utterances:
            self.add_utterances(utterances, utterances_speaker, speakers_names)

    def set_genre(self, conll):
        if conll is not None:
            genre = np.zeros((7,))
            genre[conll] = 1
        else:
            genre = np.array(0, ndmin=1, copy=False)
        return conll, genre

    def __str__(self):
        formatted = "\n ".join(
            unicode_(i) + " " + unicode_(s)
            for i, s in zip(self.utterances, self.utterances_speaker)
        )
        mentions = "\n ".join(
            unicode_(i) + " " + unicode_(i.speaker) for i in self.mentions
        )
        return f"<utterances, speakers> \n {formatted}\n<mentions> \n {mentions}"

    def __len__(self):
        """ Return the number of mentions (not utterances) since it is what we really care about """
        return len(self.mentions)

    def __getitem__(self, key):
        """ Return a specific mention (not utterance) """
        return self.mentions[key]

    def __iter__(self):
        """ Iterate over mentions (not utterances) """
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
        """
        Add utterances to the utterance list and build mention list for these utterances

        Arg:
            utterances : iterator or list of string corresponding to successive utterances
            utterances_speaker : iterator or list of speaker id for each utterance.
                If not provided, assume two speakers speaking alternatively.
                if utterances and utterances_speaker are not of the same length padded with None
            speakers_names : dictionnary of list of acceptable speaker names for each speaker id
        Return:
            List of indexes of added utterances in the docs
        """
        if self.debug:
            print("Adding utterances", utterances)
        if isinstance(utterances, string_types):
            utterances = [utterances]
        if utterances_speaker is None:
            if self.debug:
                print("No utterance speaker indication")
            a = -1
            if self.utterances_speaker and isinstance(
                self.utterances_speaker[-1].speaker_id, integer_types
            ):
                a = self.utterances_speaker[-1].speaker_id
            utterances_speaker = ((i + a + 1) % 2 for i in range(len(utterances)))
        utterances_index = []
        utt_start = len(self.utterances)
        docs = list(self.nlp.pipe(utterances))
        m_spans = list(
            extract_mentions_spans(doc, blacklist=self.blacklist) for doc in docs
        )
        for utt_index, (doc, m_spans, speaker_id) in enumerate(
            zip_longest(docs, m_spans, utterances_speaker)
        ):
            if speaker_id not in self.speakers:
                speaker_name = (
                    speakers_names.get(speaker_id, None) if speakers_names else None
                )
                self.speakers[speaker_id] = Speaker(speaker_id, speaker_name)
            self._process_mentions(
                m_spans, utt_start + utt_index, self.n_sents, self.speakers[speaker_id]
            )
            utterances_index.append(utt_start + utt_index)
            self.utterances.append(doc)
            self.utterances_speaker.append(self.speakers[speaker_id])
            self.n_sents += len(list(doc.sents))

        self.set_mentions_features()
        self.last_utterances_loaded = utterances_index

    ###################################
    ## FEATURES MENTIONS EXTRACTION ###
    ###################################

    def _process_mentions(self, mentions_spans, utterance_index, n_sents, speaker):
        """
        Process mentions in a spacy doc (an utterance)
        """
        processed_spans = sorted(
            (m for m in mentions_spans), key=lambda m: (m.root.i, m.start)
        )
        n_mentions = len(self.mentions)
        for mention_index, span in enumerate(processed_spans):
            self.mentions.append(
                Mention(
                    span, mention_index + n_mentions, utterance_index, n_sents, speaker
                )
            )

    def set_mentions_features(self):
        """
        Compute features for the extracted mentions
        """
        doc_embedding = (
            self.embed_extractor.get_document_embedding(self.utterances)
            if self.embed_extractor is not None
            else None
        )
        for mention in self.mentions:
            one_hot_type = np.zeros((4,))
            one_hot_type[mention.mention_type] = 1
            features_ = {
                "01_MentionType": mention.mention_type,
                "02_MentionLength": len(mention) - 1,
                "03_MentionNormLocation": (mention.index) / len(self.mentions),
                "04_IsMentionNested": 1
                if any(
                    (
                        m is not mention
                        and m.utterances_sent == mention.utterances_sent
                        and m.start <= mention.start
                        and mention.end <= m.end
                    )
                    for m in self.mentions
                )
                else 0,
            }
            features = np.concatenate(
                [
                    one_hot_type,
                    encode_distance(features_["02_MentionLength"]),
                    np.array(features_["03_MentionNormLocation"], ndmin=1, copy=False),
                    np.array(features_["04_IsMentionNested"], ndmin=1, copy=False),
                ],
                axis=0,
            )
            (
                spans_embeddings_,
                words_embeddings_,
                spans_embeddings,
                words_embeddings,
            ) = self.embed_extractor.get_mention_embeddings(mention, doc_embedding)
            mention.features_ = features_
            mention.features = features
            mention.spans_embeddings = spans_embeddings
            mention.spans_embeddings_ = spans_embeddings_
            mention.words_embeddings = words_embeddings
            mention.words_embeddings_ = words_embeddings_

    def get_single_mention_features(self, mention):
        """ Features for anaphoricity test (single mention features + genre if conll)"""
        features_ = mention.features_
        features_["DocGenre"] = self.genre_
        return (features_, np.concatenate([mention.features, self.genre], axis=0))

    def get_pair_mentions_features(self, m1, m2):
        """ Features for pair of mentions (same speakers, speaker mentioned, string match)"""
        features_ = {
            "00_SameSpeaker": 1
            if self.consider_speakers and m1.speaker == m2.speaker
            else 0,
            "01_AntMatchMentionSpeaker": 1
            if self.consider_speakers and m2.speaker_match_mention(m1)
            else 0,
            "02_MentionMatchSpeaker": 1
            if self.consider_speakers and m1.speaker_match_mention(m2)
            else 0,
            "03_HeadsAgree": 1 if m1.heads_agree(m2) else 0,
            "04_ExactStringMatch": 1 if m1.exact_match(m2) else 0,
            "05_RelaxedStringMatch": 1 if m1.relaxed_match(m2) else 0,
            "06_SentenceDistance": m2.utterances_sent - m1.utterances_sent,
            "07_MentionDistance": m2.index - m1.index - 1,
            "08_Overlapping": 1
            if (m1.utterances_sent == m2.utterances_sent and m1.end > m2.start)
            else 0,
            "09_M1Features": m1.features_,
            "10_M2Features": m2.features_,
            "11_DocGenre": self.genre_,
        }
        pairwise_features = [
            np.array(
                [
                    features_["00_SameSpeaker"],
                    features_["01_AntMatchMentionSpeaker"],
                    features_["02_MentionMatchSpeaker"],
                    features_["03_HeadsAgree"],
                    features_["04_ExactStringMatch"],
                    features_["05_RelaxedStringMatch"],
                ]
            ),
            encode_distance(features_["06_SentenceDistance"]),
            encode_distance(features_["07_MentionDistance"]),
            np.array(features_["08_Overlapping"], ndmin=1),
            m1.features,
            m2.features,
            self.genre,
        ]
        return (features_, np.concatenate(pairwise_features, axis=0))

    ###################################
    ###### ITERATOR OVER MENTIONS #####
    ###################################

    def get_candidate_mentions(self, last_utterances_added=False):
        """
        Return iterator over indexes of mentions in a list of utterances if specified
        """
        if last_utterances_added:
            for i, mention in enumerate(self.mentions):
                if self.debug:
                    print("🤣", i, mention, "utterance index", mention.utterance_index)
                if mention.utterance_index in self.last_utterances_loaded:
                    yield i
        else:
            iterator = range(len(self.mentions))
            for i in iterator:
                yield i

    def get_candidate_pairs(
        self, mentions=None, max_distance=50, max_distance_with_match=500, debug=False
    ):
        """
        Yield tuples of mentions, dictionnary of candidate antecedents for the mention

        Arg:
            mentions: an iterator over mention indexes (as returned by get_candidate_mentions)
            max_mention_distance : max distance between a mention and its antecedent
            max_mention_distance_string_match : max distance between a mention and
                its antecedent when there is a proper noun match
        """
        if mentions is None:
            mentions = range(len(self.mentions))
        if debug:
            print("get_candidate_pairs: mentions", mentions)

        if max_distance_with_match is not None:
            word_to_mentions = {}
            for i in mentions:
                for tok in self.mentions[i].content_words:
                    if not tok in word_to_mentions:
                        word_to_mentions[tok] = [i]
                    else:
                        word_to_mentions[tok].append(i)

        for i in mentions:
            antecedents = (
                set(range(i))
                if max_distance is None
                else set(range(max(0, i - max_distance), i))
            )
            if debug:
                print("antecedents", antecedents)
            if max_distance_with_match is not None:
                for tok in self.mentions[i].content_words:
                    with_string_match = word_to_mentions.get(tok, None)
                    for match_idx in with_string_match:
                        if match_idx < i and match_idx >= i - max_distance_with_match:
                            antecedents.add(match_idx)
            yield i, antecedents


def mention_detection_debug(sentence):
    print("🌋 Loading spacy model")
    try:
        spacy.info("en_core_web_sm")
        model = "en_core_web_sm"
    except IOError:
        print("No spacy 2 model detected, using spacy1 'en' model")
        spacy.info("en")
        model = "en"
    nlp = spacy.load(model)
    doc = nlp(sentence.decode("utf-8"))
    mentions = extract_mentions_spans(doc, blacklist=False, debug=True)
    for mention in mentions:
        print(mention)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        sent = sys.argv[1]
        mention_detection_debug(sent)
    else:
        mention_detection_debug("My sister has a dog. She loves him.")
