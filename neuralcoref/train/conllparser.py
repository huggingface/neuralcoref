"""Conll parser"""

import re
import argparse
import time
import os
import io
import pickle

import spacy

import numpy as np

from tqdm import tqdm

from neuralcoref.train.compat import unicode_
from neuralcoref.train.document import (
    Mention,
    Document,
    Speaker,
    EmbeddingExtractor,
    MISSING_WORD,
    extract_mentions_spans,
)
from neuralcoref.train.utils import parallel_process

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REMOVED_CHAR = ["/", "%", "*"]
NORMALIZE_DICT = {
    "/.": ".",
    "/?": "?",
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "-LSB-": "[",
    "-RSB-": "]",
}

CONLL_GENRES = {"bc": 0, "bn": 1, "mz": 2, "nw": 3, "pt": 4, "tc": 5, "wb": 6}

FEATURES_NAMES = [
    "mentions_features",  # 0
    "mentions_labels",  # 1
    "mentions_pairs_length",  # 2
    "mentions_pairs_start_index",  # 3
    "mentions_spans",  # 4
    "mentions_words",  # 5
    "pairs_ant_index",  # 6
    "pairs_features",  # 7
    "pairs_labels",  # 8
    "locations",  # 9
    "conll_tokens",  # 10
    "spacy_lookup",  # 11
    "doc",  # 12
]

MISSED_MENTIONS_FILE = os.path.join(
    PACKAGE_DIRECTORY, "test_mentions_identification.txt"
)
SENTENCES_PATH = os.path.join(PACKAGE_DIRECTORY, "test_sentences.txt")

###################
### UTILITIES #####


def clean_token(token):
    cleaned_token = token
    if cleaned_token in NORMALIZE_DICT:
        cleaned_token = NORMALIZE_DICT[cleaned_token]
    if cleaned_token not in REMOVED_CHAR:
        for char in REMOVED_CHAR:
            cleaned_token = cleaned_token.replace(char, "")
    if len(cleaned_token) == 0:
        cleaned_token = ","
    return cleaned_token


def mention_words_idx(embed_extractor, mention, debug=False):
    # index of the word in the tuned embeddings no need for normalizing,
    # it is already performed in set_mentions_features()
    # We take them in the tuned vocabulary which is a smaller voc tailored from conll
    words = []
    for _, w in sorted(mention.words_embeddings_.items()):
        if w not in embed_extractor.tun_idx:
            if debug:
                print(
                    "No matching tokens in tuned voc for word ",
                    w,
                    "surrounding or inside mention",
                    mention,
                )
            words.append(MISSING_WORD)
        else:
            words.append(w)
    return [embed_extractor.tun_idx[w] for w in words]


def check_numpy_array(feature, array, n_mentions_list, compressed=True):
    for n_mentions in n_mentions_list:
        if feature == FEATURES_NAMES[0]:
            assert array.shape[0] == len(n_mentions)
            if compressed:
                assert np.array_equiv(
                    array[:, 3], np.array([len(n_mentions)] * len(n_mentions))
                )
                assert np.max(array[:, 2]) == len(n_mentions) - 1
                assert np.min(array[:, 2]) == 0
        elif feature == FEATURES_NAMES[1]:
            assert array.shape[0] == len(n_mentions)
        elif feature == FEATURES_NAMES[2]:
            assert array.shape[0] == len(n_mentions)
            assert np.array_equiv(array[:, 0], np.array(list(range(len(n_mentions)))))
        elif feature == FEATURES_NAMES[3]:
            assert array.shape[0] == len(n_mentions)
            assert np.array_equiv(
                array[:, 0], np.array([p * (p - 1) / 2 for p in range(len(n_mentions))])
            )
        elif feature == FEATURES_NAMES[4]:
            assert array.shape[0] == len(n_mentions)
        elif feature == FEATURES_NAMES[5]:
            assert array.shape[0] == len(n_mentions)
        elif feature == FEATURES_NAMES[6]:
            assert array.shape[0] == len(n_mentions) * (len(n_mentions) - 1) / 2
            assert np.max(array) == len(n_mentions) - 2
        elif feature == FEATURES_NAMES[7]:
            if compressed:
                assert array.shape[0] == len(n_mentions) * (len(n_mentions) - 1) / 2
                assert np.max(array[:, 7]) == len(n_mentions) - 2
                assert np.min(array[:, 7]) == 0
        elif feature == FEATURES_NAMES[8]:
            assert array.shape[0] == len(n_mentions) * (len(n_mentions) - 1) / 2


###############################################################################################
### PARALLEL FCT (has to be at top-level of the module to be pickled for multiprocessing) #####
def load_file(full_name, debug=False):
    """
    load a *._conll file
    Input: full_name: path to the file
    Output: list of tuples for each conll doc in the file, where the tuple contains:
        (utts_text ([str]): list of the utterances in the document
         utts_tokens ([[str]]): list of the tokens (conll words) in the document
         utts_corefs: list of coref objects (dicts) with the following properties:
            coref['label']: id of the coreference cluster,
            coref['start']: start index (index of first token in the utterance),
            coref['end': end index (index of last token in the utterance).
         utts_speakers ([str]): list of the speaker associated to each utterances in the document
         name (str): name of the document
         part (str): part of the document
        )
    """
    docs = []
    with io.open(full_name, "rt", encoding="utf-8", errors="strict") as f:
        lines = list(f)  # .readlines()
        utts_text = []
        utts_tokens = []
        utts_corefs = []
        utts_speakers = []
        tokens = []
        corefs = []
        index = 0
        speaker = ""
        name = ""
        part = ""
        for li, line in enumerate(lines):
            cols = line.split()
            if debug:
                print("line", li, "cols:", cols)
            # End of utterance
            if len(cols) == 0:
                if tokens:
                    if debug:
                        print("End of utterance")
                    utts_text.append("".join(t + " " for t in tokens))
                    utts_tokens.append(tokens)
                    utts_speakers.append(speaker)
                    utts_corefs.append(corefs)
                    tokens = []
                    corefs = []
                    index = 0
                    speaker = ""
                    continue
            # End of doc
            elif len(cols) == 2:
                if debug:
                    print("End of doc")
                if cols[0] == "#end":
                    if debug:
                        print("Saving doc")
                    docs.append(
                        (utts_text, utts_tokens, utts_corefs, utts_speakers, name, part)
                    )
                    utts_text = []
                    utts_tokens = []
                    utts_corefs = []
                    utts_speakers = []
                else:
                    raise ValueError("Error on end line " + line)
            # New doc
            elif len(cols) == 5:
                if debug:
                    print("New doc")
                if cols[0] == "#begin":
                    name = re.match(r"\((.*)\);", cols[2]).group(1)
                    try:
                        part = cols[4]
                    except ValueError:
                        print("Error parsing document part " + line)
                    if debug:
                        print("New doc", name, part, name[:2])
                    tokens = []
                    corefs = []
                    index = 0
                else:
                    raise ValueError("Error on begin line " + line)
            # Inside utterance
            elif len(cols) > 7:
                if debug:
                    print("Inside utterance")
                assert cols[0] == name and int(cols[1]) == int(part), (
                    "Doc name or part error " + line
                )
                assert int(cols[2]) == index, "Index error on " + line
                if speaker:
                    assert cols[9] == speaker, "Speaker changed in " + line + speaker
                else:
                    speaker = cols[9]
                    if debug:
                        print("speaker", speaker)
                if cols[-1] != "-":
                    coref_expr = cols[-1].split("|")
                    if debug:
                        print("coref_expr", coref_expr)
                    if not coref_expr:
                        raise ValueError("Coref expression empty " + line)
                    for tok in coref_expr:
                        if debug:
                            print("coref tok", tok)
                        try:
                            match = re.match(r"^(\(?)(\d+)(\)?)$", tok)
                        except:
                            print("error getting coreferences for line " + line)
                        assert match is not None, (
                            "Error parsing coref " + tok + " in " + line
                        )
                        num = match.group(2)
                        assert num is not "", (
                            "Error parsing coref " + tok + " in " + line
                        )
                        if match.group(1) == "(":
                            if debug:
                                print("New coref", num)
                            corefs.append({"label": num, "start": index, "end": None})
                        if match.group(3) == ")":
                            j = None
                            for i in range(len(corefs) - 1, -1, -1):
                                if debug:
                                    print("i", i)
                                if (
                                    corefs[i]["label"] == num
                                    and corefs[i]["end"] is None
                                ):
                                    j = i
                                    break
                            assert j is not None, "coref closing error " + line
                            if debug:
                                print("End coref", num)
                            corefs[j]["end"] = index
                tokens.append(clean_token(cols[3]))
                index += 1
            else:
                raise ValueError("Line not standard " + line)
    return docs


def set_feats(doc):
    doc.set_mentions_features()


def get_feats(doc, i):
    return doc.get_feature_array(doc_id=i)


def gather_feats(gathering_array, array, feat_name, pairs_ant_index, pairs_start_index):
    if gathering_array is None:
        gathering_array = array
    else:
        if feat_name == FEATURES_NAMES[6]:
            array = [a + pairs_ant_index for a in array]
        elif feat_name == FEATURES_NAMES[3]:
            array = [a + pairs_start_index for a in array]
        gathering_array += array
    return feat_name, gathering_array


def read_file(full_name):
    doc = ""
    with io.open(full_name, "rt", encoding="utf-8", errors="strict") as f:
        doc = f.read()
    return doc


###################
### ConllDoc #####


class ConllDoc(Document):
    def __init__(self, name, part, *args, **kwargs):
        self.name = name
        self.part = part
        self.feature_matrix = {}
        self.conll_tokens = []
        self.conll_lookup = []
        self.gold_corefs = []
        self.missed_gold = []
        super(ConllDoc, self).__init__(*args, **kwargs)

    def get_conll_spacy_lookup(self, conll_tokens, spacy_tokens, debug=False):
        """
        Compute a look up table between spacy tokens (from spacy tokenizer)
        and conll pre-tokenized tokens
        Output: list[conll_index] => list of associated spacy tokens (assume spacy tokenizer has a finer granularity)
        """
        lookup = []
        c_iter = (t for t in conll_tokens)
        s_iter = enumerate(t for t in spacy_tokens)
        i, s_tok = next(s_iter)
        for c_tok in c_iter:
            # if debug: print("conll", c_tok, "spacy", s_tok, "index", i)
            c_lookup = []
            while i is not None and len(c_tok) and c_tok.startswith(s_tok.text):
                c_lookup.append(i)
                c_tok = c_tok[len(s_tok) :]
                i, s_tok = next(s_iter, (None, None))
                if debug and len(c_tok):
                    print("eating token: conll", c_tok, "spacy", s_tok, "index", i)
            assert len(c_lookup), "Unmatched conll and spacy tokens"
            lookup.append(c_lookup)
        return lookup

    def add_conll_utterance(
        self, parsed, tokens, corefs, speaker_id, use_gold_mentions, debug=False
    ):
        conll_lookup = self.get_conll_spacy_lookup(tokens, parsed)
        self.conll_tokens.append(tokens)
        self.conll_lookup.append(conll_lookup)
        # Convert conll tokens coref index in spacy tokens indexes
        identified_gold = [False] * len(corefs)
        for coref in corefs:
            missing_values = [key for key in ['label', 'start', 'end', ] if coref.get(key, None) is None]
            if missing_values:
                found_values = {key: coref[key] for key in ['label', 'start', 'end'] if coref.get(key, None) is not None}
                raise Exception(f"Coref {self.name} with fields {found_values} has empty values for the keys {missing_values}.")

            coref["start"] = conll_lookup[coref["start"]][0]
            coref["end"] = conll_lookup[coref["end"]][-1]

        if speaker_id not in self.speakers:
            speaker_name = speaker_id.split("_")
            if debug:
                print("New speaker: ", speaker_id, "name: ", speaker_name)
            self.speakers[speaker_id] = Speaker(speaker_id, speaker_name)
        if use_gold_mentions:
            for coref in corefs:
                # print("coref['label']", coref['label'])
                # print("coref text",parsed[coref['start']:coref['end']+1])
                mention = Mention(
                    parsed[coref["start"] : coref["end"] + 1],
                    len(self.mentions),
                    len(self.utterances),
                    self.n_sents,
                    speaker=self.speakers[speaker_id],
                    gold_label=coref["label"],
                )
                self.mentions.append(mention)
                # print("mention: ", mention, "label", mention.gold_label)
        else:
            mentions_spans = extract_mentions_spans(
                doc=parsed, blacklist=self.blacklist
            )
            self._process_mentions(
                mentions_spans,
                len(self.utterances),
                self.n_sents,
                self.speakers[speaker_id],
            )

            # Assign a gold label to mentions which have one
            if debug:
                print("Check corefs", corefs)
            for i, coref in enumerate(corefs):
                for m in self.mentions:
                    if m.utterance_index != len(self.utterances):
                        continue
                    # if debug: print("Checking mention", m, m.utterance_index, m.start, m.end)
                    if coref["start"] == m.start and coref["end"] == m.end - 1:
                        m.gold_label = coref["label"]
                        identified_gold[i] = True
                        # if debug: print("Gold mention found:", m, coref['label'])
            for found, coref in zip(identified_gold, corefs):
                if not found:
                    self.missed_gold.append(
                        [
                            self.name,
                            self.part,
                            str(len(self.utterances)),
                            parsed.text,
                            parsed[coref["start"] : coref["end"] + 1].text,
                        ]
                    )
                    if debug:
                        print(
                            "❄️ gold mention not in predicted mentions",
                            coref,
                            parsed[coref["start"] : coref["end"] + 1],
                        )
        self.utterances.append(parsed)
        self.gold_corefs.append(corefs)
        self.utterances_speaker.append(self.speakers[speaker_id])
        self.n_sents += len(list(parsed.sents))

    def get_single_mention_features_conll(self, mention, compressed=True):
        """ Compressed or not single mention features"""
        if not compressed:
            _, features = self.get_single_mention_features(mention)
            return features[np.newaxis, :]
        feat_l = [
            mention.features_["01_MentionType"],
            mention.features_["02_MentionLength"],
            mention.index,
            len(self.mentions),
            mention.features_["04_IsMentionNested"],
            self.genre_,
        ]
        return feat_l

    def get_pair_mentions_features_conll(self, m1, m2, compressed=True):
        """ Compressed or not single mention features"""
        if not compressed:
            _, features = self.get_pair_mentions_features(m1, m2)
            return features[np.newaxis, :]
        features_, _ = self.get_pair_mentions_features(m1, m2)
        feat_l = [
            features_["00_SameSpeaker"],
            features_["01_AntMatchMentionSpeaker"],
            features_["02_MentionMatchSpeaker"],
            features_["03_HeadsAgree"],
            features_["04_ExactStringMatch"],
            features_["05_RelaxedStringMatch"],
            features_["06_SentenceDistance"],
            features_["07_MentionDistance"],
            features_["08_Overlapping"],
        ]
        return feat_l

    def get_feature_array(self, doc_id, feature=None, compressed=True, debug=False):
        """
        Prepare feature array:
            mentions_spans: (N, S)
            mentions_words: (N, W)
            mentions_features: (N, Fs)
            mentions_labels: (N, 1)
            mentions_pairs_start_index: (N, 1) index of beggining of pair list in pair_labels
            mentions_pairs_length: (N, 1) number of pairs (i.e. nb of antecedents) for each mention
            pairs_features: (P, Fp)
            pairs_labels: (P, 1)
            pairs_ant_idx: (P, 1) => indexes of antecedents mention for each pair (mention index in doc)
        """
        if not self.mentions:
            if debug:
                print("No mention in this doc !")
            return {}
        if debug:
            print("🛎 features matrices")
        mentions_spans = []
        mentions_words = []
        mentions_features = []
        pairs_ant_idx = []
        pairs_features = []
        pairs_labels = []
        mentions_labels = []
        mentions_pairs_start = []
        mentions_pairs_length = []
        mentions_location = []
        n_mentions = 0
        total_pairs = 0
        if debug:
            print("mentions", self.mentions, str([m.gold_label for m in self.mentions]))
        for mention_idx, antecedents_idx in list(
            self.get_candidate_pairs(max_distance=None, max_distance_with_match=None)
        ):
            n_mentions += 1
            mention = self.mentions[mention_idx]
            mentions_spans.append(mention.spans_embeddings)
            w_idx = mention_words_idx(self.embed_extractor, mention)
            if w_idx is None:
                print("error in", self.name, self.part, mention.utterance_index)
            mentions_words.append(w_idx)
            mentions_features.append(
                self.get_single_mention_features_conll(mention, compressed)
            )
            mentions_location.append(
                [
                    mention.start,
                    mention.end,
                    mention.utterance_index,
                    mention_idx,
                    doc_id,
                ]
            )
            ants = [self.mentions[ant_idx] for ant_idx in antecedents_idx]
            no_antecedent = (
                not any(ant.gold_label == mention.gold_label for ant in ants)
                or mention.gold_label is None
            )
            if antecedents_idx:
                pairs_ant_idx += [idx for idx in antecedents_idx]
                pairs_features += [
                    self.get_pair_mentions_features_conll(ant, mention, compressed)
                    for ant in ants
                ]
                ant_labels = (
                    [0 for ant in ants]
                    if no_antecedent
                    else [
                        1 if ant.gold_label == mention.gold_label else 0 for ant in ants
                    ]
                )
                pairs_labels += ant_labels
            mentions_labels.append(1 if no_antecedent else 0)
            mentions_pairs_start.append(total_pairs)
            total_pairs += len(ants)
            mentions_pairs_length.append(len(ants))

        out_dict = {
            FEATURES_NAMES[0]: mentions_features,
            FEATURES_NAMES[1]: mentions_labels,
            FEATURES_NAMES[2]: mentions_pairs_length,
            FEATURES_NAMES[3]: mentions_pairs_start,
            FEATURES_NAMES[4]: mentions_spans,
            FEATURES_NAMES[5]: mentions_words,
            FEATURES_NAMES[6]: pairs_ant_idx if pairs_ant_idx else None,
            FEATURES_NAMES[7]: pairs_features if pairs_features else None,
            FEATURES_NAMES[8]: pairs_labels if pairs_labels else None,
            FEATURES_NAMES[9]: [mentions_location],
            FEATURES_NAMES[10]: [self.conll_tokens],
            FEATURES_NAMES[11]: [self.conll_lookup],
            FEATURES_NAMES[12]: [
                {
                    "name": self.name,
                    "part": self.part,
                    "utterances": list(str(u) for u in self.utterances),
                    "mentions": list(str(m) for m in self.mentions),
                }
            ],
        }
        if debug:
            print("🚘 Summary")
            for k, v in out_dict.items():
                print(k, len(v))
        return n_mentions, total_pairs, out_dict


###################
### ConllCorpus #####
class ConllCorpus(object):
    def __init__(
        self,
        n_jobs=4,
        embed_path=PACKAGE_DIRECTORY + "/weights/",
        gold_mentions=False,
        blacklist=False,
    ):
        self.n_jobs = n_jobs
        self.features = {}
        self.utts_text = []
        self.utts_tokens = []
        self.utts_corefs = []
        self.utts_speakers = []
        self.utts_doc_idx = []
        self.docs_names = []
        self.docs = []
        if embed_path is not None:
            self.embed_extractor = EmbeddingExtractor(embed_path)
        self.trainable_embed = []
        self.trainable_voc = []
        self.gold_mentions = gold_mentions
        self.blacklist = blacklist

    def check_words_in_embeddings_voc(self, embedding, tuned=True, debug=False):
        print("🌋 Checking if words are in embedding voc")
        if tuned:
            embed_voc = embedding.tun_idx
        else:
            embed_voc = embedding.stat_idx
        missing_words = []
        missing_words_sents = []
        missing_words_doc = []
        for doc in self.docs:
            # if debug: print("Checking doc", doc.name, doc.part)
            for sent in doc.utterances:
                # if debug: print(sent.text)
                for word in sent:
                    w = embedding.normalize_word(word)
                    # if debug: print(w)
                    if w not in embed_voc:
                        missing_words.append(w)
                        missing_words_sents.append(sent.text)
                        missing_words_doc.append(doc.name + doc.part)
                        if debug:
                            out_str = (
                                "No matching tokens in tuned voc for "
                                + w
                                + " in sentence "
                                + sent.text
                                + " in doc "
                                + doc.name
                                + doc.part
                            )
                            print(out_str)
        return missing_words, missing_words_sents, missing_words_doc

    def test_sentences_words(self, save_file, debug=False):
        print("🌋 Saving sentence list")
        with io.open(save_file, "w", encoding="utf-8") as f:
            if debug:
                print("Sentences saved in", save_file)
            for doc in self.docs:
                out_str = "#begin document (" + doc.name + "); part " + doc.part + "\n"
                f.write(out_str)
                for sent in doc.utterances:
                    f.write(sent.text + "\n")
                out_str = "#end document\n\n"
                f.write(out_str)

    def save_sentences(self, save_file, debug=False):
        print("🌋 Saving sentence list")
        with io.open(save_file, "w", encoding="utf-8") as f:
            if debug:
                print("Sentences saved in", save_file)
            for doc in self.docs:
                out_str = "#begin document (" + doc.name + "); part " + doc.part + "\n"
                f.write(out_str)
                for sent in doc.utterances:
                    f.write(sent.text + "\n")
                out_str = "#end document\n\n"
                f.write(out_str)

    def build_key_file(self, data_path, key_file, debug=False):
        print("🌋 Building key file from corpus")
        print("Saving in", key_file)
        # Create a pool of processes. By default, one is created for each CPU in your machine.
        with io.open(key_file, "w", encoding="utf-8") as kf:
            if debug:
                print("Key file saved in", key_file)
            for dirpath, _, filenames in os.walk(data_path):
                print("In", dirpath)
                file_list = [
                    os.path.join(dirpath, f)
                    for f in filenames
                    if f.endswith(".v4_auto_conll") or f.endswith(".v4_gold_conll")
                ]
                cleaned_file_list = []
                for f in file_list:
                    fn = f.split(".")
                    if fn[1] == "v4_auto_conll":
                        gold = fn[0] + "." + "v4_gold_conll"
                        if gold not in file_list:
                            cleaned_file_list.append(f)
                    else:
                        cleaned_file_list.append(f)
                # self.load_file(file_list[0])
                doc_list = parallel_process(cleaned_file_list, read_file)
                for doc in doc_list:
                    kf.write(doc)

    def list_undetected_mentions(self, data_path, save_file, debug=True):
        self.read_corpus(data_path)
        print("🌋 Listing undetected mentions")
        with io.open(save_file, "w", encoding="utf-8") as out_file:
            for doc in tqdm(self.docs):
                for name, part, utt_i, utt, coref in doc.missed_gold:
                    out_str = name + "\t" + part + "\t" + utt_i + '\t"' + utt + '"\n'
                    out_str += coref + "\n"
                    out_file.write(out_str)
                    if debug:
                        print(out_str)

    def read_corpus(self, data_path, model=None, debug=False):
        print("🌋 Reading files")
        for dirpath, _, filenames in os.walk(data_path):
            print("In", dirpath, os.path.abspath(dirpath))
            file_list = [
                os.path.join(dirpath, f)
                for f in filenames
                if f.endswith(".v4_auto_conll") or f.endswith(".v4_gold_conll")
            ]
            cleaned_file_list = []
            for f in file_list:
                fn = f.split(".")
                if fn[1] == "v4_auto_conll":
                    gold = fn[0] + "." + "v4_gold_conll"
                    if gold not in file_list:
                        cleaned_file_list.append(f)
                else:
                    cleaned_file_list.append(f)
            doc_list = parallel_process(cleaned_file_list, load_file)
            for docs in doc_list:  # executor.map(self.load_file, cleaned_file_list):
                for (
                    utts_text,
                    utt_tokens,
                    utts_corefs,
                    utts_speakers,
                    name,
                    part,
                ) in docs:
                    if debug:
                        print("Imported", name)
                        print("utts_text", utts_text)
                        print("utt_tokens", utt_tokens)
                        print("utts_corefs", utts_corefs)
                        print("utts_speakers", utts_speakers)
                        print("name, part", name, part)
                    self.utts_text += utts_text
                    self.utts_tokens += utt_tokens
                    self.utts_corefs += utts_corefs
                    self.utts_speakers += utts_speakers
                    self.utts_doc_idx += [len(self.docs_names)] * len(utts_text)
                    self.docs_names.append((name, part))
        print("utts_text size", len(self.utts_text))
        print("utts_tokens size", len(self.utts_tokens))
        print("utts_corefs size", len(self.utts_corefs))
        print("utts_speakers size", len(self.utts_speakers))
        print("utts_doc_idx size", len(self.utts_doc_idx))
        print("🌋 Building docs")
        for name, part in self.docs_names:
            self.docs.append(
                ConllDoc(
                    name=name,
                    part=part,
                    nlp=None,
                    blacklist=self.blacklist,
                    consider_speakers=True,
                    embedding_extractor=self.embed_extractor,
                    conll=CONLL_GENRES[name[:2]],
                )
            )
        print("🌋 Loading spacy model")

        if model is None:
            model_options = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm", "en"]
            for model_option in model_options:
                if not model:
                    try:
                        spacy.info(model_option)
                        model = model_option
                        print("Loading model", model_option)
                    except:
                        print("Could not detect model", model_option)
            if not model:
                print("Could not detect any suitable English model")
                return
        else:
            spacy.info(model)
            print("Loading model", model)
        nlp = spacy.load(model)
        print(
            "🌋 Parsing utterances and filling docs with use_gold_mentions="
            + (str(bool(self.gold_mentions)))
        )
        doc_iter = (s for s in self.utts_text)
        for utt_tuple in tqdm(
            zip(
                nlp.pipe(doc_iter),
                self.utts_tokens,
                self.utts_corefs,
                self.utts_speakers,
                self.utts_doc_idx,
            )
        ):
            spacy_tokens, conll_tokens, corefs, speaker, doc_id = utt_tuple
            if debug:
                print(unicode_(self.docs_names[doc_id]), "-", spacy_tokens)
            doc = spacy_tokens
            if debug:
                out_str = (
                    "utterance "
                    + unicode_(doc)
                    + " corefs "
                    + unicode_(corefs)
                    + " speaker "
                    + unicode_(speaker)
                    + "doc_id"
                    + unicode_(doc_id)
                )
                print(out_str.encode("utf-8"))
            self.docs[doc_id].add_conll_utterance(
                doc, conll_tokens, corefs, speaker, use_gold_mentions=self.gold_mentions
            )

    def build_and_gather_multiple_arrays(self, save_path):
        print(f"🌋 Extracting mentions features with {self.n_jobs} job(s)")
        parallel_process(self.docs, set_feats, n_jobs=self.n_jobs)

        print(f"🌋 Building and gathering array with {self.n_jobs} job(s)")
        arr = [{"doc": doc, "i": i} for i, doc in enumerate(self.docs)]
        arrays_dicts = parallel_process(
            arr, get_feats, use_kwargs=True, n_jobs=self.n_jobs
        )
        gathering_dict = dict((feat, None) for feat in FEATURES_NAMES)
        n_mentions_list = []
        pairs_ant_index = 0
        pairs_start_index = 0
        for npaidx in tqdm(range(len(arrays_dicts))):
            try:
                n, p, arrays_dict = arrays_dicts[npaidx]
            except:
                # empty array dict, cannot extract the dict values for this doc
                continue

            for f in FEATURES_NAMES:
                if gathering_dict[f] is None:
                    gathering_dict[f] = arrays_dict[f]
                else:
                    if f == FEATURES_NAMES[6]:
                        array = [a + pairs_ant_index for a in arrays_dict[f]]
                    elif f == FEATURES_NAMES[3]:
                        array = [a + pairs_start_index for a in arrays_dict[f]]
                    else:
                        array = arrays_dict[f]
                    gathering_dict[f] += array
            pairs_ant_index += n
            pairs_start_index += p
            n_mentions_list.append(n)

        for feature in FEATURES_NAMES[:9]:
            feature_data = gathering_dict[feature]
            if not feature_data:
                print("No data for", feature)
                continue
            print("Building numpy array for", feature, "length", len(feature_data))
            if feature != "mentions_spans":
                array = np.array(feature_data)
                if array.ndim == 1:
                    array = np.expand_dims(array, axis=1)
            else:
                array = np.stack(feature_data)
            # check_numpy_array(feature, array, n_mentions_list)
            print("Saving numpy", feature, "size", array.shape)
            np.save(save_path + feature, array)
        for feature in FEATURES_NAMES[9:]:
            feature_data = gathering_dict[feature]
            if feature_data:
                print("Saving pickle", feature, "size", len(feature_data))
                with open(save_path + feature + ".bin", "wb") as fp:
                    pickle.dump(feature_data, fp)

    def save_vocabulary(self, save_path, debug=False):
        def _vocabulary_to_file(path, vocabulary):
            print("🌋 Saving vocabulary")
            with io.open(path, "w", encoding="utf-8") as f:
                if debug:
                    print(f"voc saved in {path}, length: {len(vocabulary)}")
                for w in tunable_voc:
                    f.write(w + "\n")

        print("🌋 Building tunable vocabulary matrix from static vocabulary")
        tunable_voc = self.embed_extractor.tun_voc
        _vocabulary_to_file(
            path=save_path + "tuned_word_vocabulary.txt", vocabulary=tunable_voc
        )

        static_voc = self.embed_extractor.stat_voc
        _vocabulary_to_file(
            path=save_path + "static_word_vocabulary.txt", vocabulary=static_voc
        )

        tuned_word_embeddings = np.vstack(
            [self.embed_extractor.get_stat_word(w)[1] for w in tunable_voc]
        )
        print("Saving tunable voc, size:", tuned_word_embeddings.shape)
        np.save(save_path + "tuned_word_embeddings", tuned_word_embeddings)

        static_word_embeddings = np.vstack(
            [self.embed_extractor.static_embeddings[w] for w in static_voc]
        )
        print("Saving static voc, size:", static_word_embeddings.shape)
        np.save(save_path + "static_word_embeddings", static_word_embeddings)


if __name__ == "__main__":
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(
        description="Training the neural coreference model"
    )
    parser.add_argument(
        "--function",
        type=str,
        default="all",
        help='Function ("all", "key", "parse", "find_undetected")',
    )
    parser.add_argument(
        "--path", type=str, default=DIR_PATH + "/data/", help="Path to the dataset"
    )
    parser.add_argument(
        "--key", type=str, help="Path to an optional key file for scoring"
    )
    parser.add_argument(
        "--n_jobs", type=int, default=1, help="Number of parallel jobs (default 1)"
    )
    parser.add_argument(
        "--gold_mentions",
        type=int,
        default=0,
        help="Use gold mentions (1) or not (0, default)",
    )
    parser.add_argument(
        "--blacklist", type=int, default=0, help="Use blacklist (1) or not (0, default)"
    )
    parser.add_argument("--spacy_model", type=str, default=None, help="model name")
    args = parser.parse_args()
    if args.key is None:
        args.key = args.path + "/key.txt"
    CORPUS = ConllCorpus(
        n_jobs=args.n_jobs, gold_mentions=args.gold_mentions, blacklist=args.blacklist
    )
    if args.function == "parse" or args.function == "all":
        SAVE_DIR = args.path + "/numpy/"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        else:
            if os.listdir(SAVE_DIR):
                print("There are already data in", SAVE_DIR)
                print("Erasing")
                for file in os.listdir(SAVE_DIR):
                    print(file)
                    os.remove(SAVE_DIR + file)
        start_time = time.time()
        CORPUS.read_corpus(args.path, model=args.spacy_model)
        print("=> read_corpus time elapsed", time.time() - start_time)
        if not CORPUS.docs:
            print("Could not parse any valid docs")
        else:
            start_time2 = time.time()
            CORPUS.build_and_gather_multiple_arrays(SAVE_DIR)
            print(
                "=> build_and_gather_multiple_arrays time elapsed",
                time.time() - start_time2,
            )
            start_time2 = time.time()
            CORPUS.save_vocabulary(SAVE_DIR)
            print("=> save_vocabulary time elapsed", time.time() - start_time2)
            print("=> total time elapsed", time.time() - start_time)
    if args.function == "key" or args.function == "all":
        CORPUS.build_key_file(args.path, args.key)
    if args.function == "find_undetected":
        CORPUS.list_undetected_mentions(
            args.path, args.path + "/undetected_mentions.txt"
        )
