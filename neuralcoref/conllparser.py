# coding: utf8
"""Conll parser"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import re
import sys
import os
import spacy
import numpy as np

from data import Mention, Data, Speaker

class ConllDoc(Data):
    def __init__(self, name, part, *args, **kwargs):
        self.name = name
        self.part = part
        super(ConllDoc, self).__init__(*args, **kwargs)

    def add_conll_utterance(self, doc, corefs, speaker_id):
        if speaker_id not in self.speakers:
            speaker_name = speaker_id.split(u'_')
#            print("New speaker: ", speaker_id, "name: ", speaker_name)
            self.speakers[speaker_id] = Speaker(speaker_id, speaker_name)
#        print("doc: ", doc)
        for coref in corefs:
            assert (coref['label'] is not None and coref['start'] is not None and coref['end'] is not None), \
                   ("Error in coreference " + coref + " in " + doc)
#            print("coref['label']", coref['label'])
#            print("coref text",doc[coref['start']:coref['end']])
            mention = Mention(doc[coref['start']:coref['end']], len(self.mentions), len(self.utterances),
                                  self.n_sents, speaker=self.speakers[speaker_id], gold_label=coref['label'])
            self.mentions.append(mention)
#            print("mention: ", mention, "label", mention.gold_label)
        self.utterances.append(doc)
        self.utterances_speaker.append(self.speakers[speaker_id])
        self.n_sents += len(list(doc.sents))

    def w2idx(self, word):
        # index of the word in the static embeddings
        return self.embed_extractor.static_voc[word]

    def get_features_matrices(self, debug=False):
        if not self.mentions:
            raise ValueError("No mention in this doc !")

        self.set_mentions_features()

        mention_span = []
        mention_words = []
        mention_features = []
        pair_span = []
        pair_words = []
        pair_features = []
        pair_labels = []
        single_labels = []
        mention_pair_indices = []
        total_pairs = 0
        for mention_idx, antecedents_idx in self.get_candidate_pairs():
            mention = self.mentions[mention_idx]
            ants = [self.mentions[ant_idx] for ant_idx in antecedents_idx]
            mention_span.append(mention.spans_embeddings)
            mention_words.append(np.array([self.w2idx(w) for _, w in sorted(mention.words_embeddings_.items())]))
            mention_features.append(self.get_anaphoricity_features(mention)[1])
            pair_span.append(np.vstack([ant.spans_embeddings for ant in ants]))
            pair_words.append(np.array([[self.w2idx(w) for _, w in sorted(ant.words_embeddings_.items())] \
                                        for ant in ants]))
            pair_features.append(np.array([self.get_pair_features(ant, mention)[1] for ant in ants]))
            ant_labels = [[1 if ant.gold_label == mention.gold_label else 0] for ant in ants]
            no_antecedent = not any(ant.gold_label == mention.gold_label for ant in ants)
            pair_labels += ant_labels
            single_labels.append([1 if no_antecedent else 0])
            mention_pair_indices.append([total_pairs, total_pairs + len(ants)])
            total_pairs += len(ants)

            if debug:
                print(mention, ants)
                print("mention.spans_embeddings", mention.spans_embeddings.shape)
                print("mention_words", np.array([self.w2idx(w) for _, w in sorted(mention.words_embeddings_.items())]).shape)
                print("mention_features", self.get_anaphoricity_features(mention)[1].shape)
                print("pair_span", np.vstack([ant.spans_embeddings for ant in ants]).shape)
                print("pair_words", np.array([[self.w2idx(w) for _, w in sorted(ant.words_embeddings_.items())] \
                                        for ant in ants]).shape)
                print("pair_features", np.array([self.get_pair_features(ant, mention)[1] for ant in ants]).shape)
                print("pair_labels", np.array([1 if ant.gold_label == mention.gold_label else 0 for ant in ants]))
                print("single_labels", 0 if any(ant.gold_label == mention.gold_label for ant in ants) else 1)

        if debug:
            print("mention_span", np.vstack(mention_span).shape)
            print("mention_words", np.vstack(mention_words).shape)
            print("mention_features", np.vstack(mention_features).shape)
            print("pair_span", np.vstack(pair_span).shape)
            print("pair_words", np.vstack(pair_words).shape)
            print("pair_features", np.vstack(pair_features).shape)
            print("pair_labels", np.array(pair_labels).shape)
            print("single_labels", np.array(single_labels).shape)
            print("mention_pair_indices", np.array(mention_pair_indices).shape)
        return {"mention_span": np.vstack(mention_span),
                "mention_words": np.vstack(mention_words),
                "mention_features": np.vstack(mention_features),
                "pair_span": np.vstack(pair_span),
                "pair_words": np.vstack(pair_words),
                "pair_features": np.vstack(pair_features),
                "pair_labels": np.array(pair_labels),
                "single_labels": np.array(single_labels),
                "mention_pair_indices": np.array(mention_pair_indices),
                "vocab_size": np.array(len(self.embed_extractor.static_voc))}

class ConllDataset:
    def __init__(self, data_path, save_path, embed_path="./weights/", nlp=None):
        if nlp is None:
            print("Loading spacy model")
            try:
                spacy.info('en_core_web_sm')
                model = 'en_core_web_sm'
            except IOError:
                print("No spacy 2 model detected, using spacy1 'en' model")
                model = 'en'
            nlp = spacy.load(model)
        self.nlp = nlp
        self.embed_path = embed_path
        self.save_path = save_path
        self.features = None
        self.load_dataset(data_path)

    def update_arrays(self, doc, debug=True):
        update_dict = doc.get_features_matrices(debug=debug)
        if debug: print("🎃 updating array")
        if self.features is not None:
            for k, v in update_dict.items():
                self.features[k] = np.vstack([self.features[k], v])
                if debug: print(k, self.features[k].shape)
        else:
            self.features = update_dict

    def save_arrays(self, save_name):
        for k, v in self.features.items():
            np.save(save_name + k, v)

    def load_dataset(self, path):
        for dirpath, _, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(".v4_auto_conll")]:
                full_name = os.path.join(dirpath, filename)
                print("Importing", full_name)
                self.features = None
                self.load_file(full_name)
                self.save_arrays(self.save_path + full_name.split(sep='.')[0])

    def load_file(self, full_name, debug=False):
        with open(full_name, 'r') as f:
            lines = f.readlines()
            doc = None
            tokens = []
            corefs = []
            index = 0
            speaker = ""
            for line in lines:
                cols = line.split()
                if debug: print("cols:", cols)
                # End of utterance
                if len(cols) == 0:
                    if tokens:
                        if debug: print("End of utterance")
                        try:
                            parsed = self.nlp(u''.join(t + u' ' for t in tokens))
                        except IndexError:
                            parsed = self.nlp(u" ")
                            print("Empty utterance")
                        doc.add_conll_utterance(parsed, corefs, speaker)
                        tokens = []
                        corefs = []
                        index = 0
                        speaker = ""
                # End of doc
                elif len(cols) == 2:
                    if debug: print("End of doc")
                    if cols[0] == u"#end" and doc is not None:
                        print("Doc", doc.name, doc.part,
                              "utterances", len(doc.utterances), #"\n", doc.utterances,
                              "mentions", len(doc.mentions), #"\n", doc.mentions,
                              "speakers", len(doc.speakers))#, "\n", doc.speakers)
                        self.update_arrays(doc)
                        doc = None
                    else:
                        raise ValueError("Error on end line " + line)
                # New doc
                elif len(cols) == 5:
                    if debug: print("New doc")
                    if cols[0] == u"#begin" and doc is None:
                        name = re.match(r"\((.*)\);", cols[2]).group(1)
                        try:
                            part = int(cols[4])
                        except ValueError:
                            print("Error parsing document part " + line)
                        if debug: print("New doc", name, part, name[:2])
                        doc = ConllDoc(name, part, model_path=self.embed_path, nlp=self.nlp, conll=name[:2])
                        tokens = []
                        corefs = []
                        index = 0
                    else:
                        raise ValueError("Error on begin line " + line)
                # Inside utterance
                elif len(cols) > 7:
                    if debug: print("Inside utterance")
                    assert (doc is not None), "In utterance but doc is None " + line
                    assert (cols[0] == doc.name and int(cols[1]) == doc.part), "Doc name or part error " + line
                    assert (int(cols[2]) == index), "Index error on " + line
                    if speaker:
                        assert (cols[9] == speaker), "Speaker changed in " + line + speaker
                    else:
                        speaker = cols[9]
                        if debug: print("speaker", speaker)
                    if cols[-1] is not '-':
                        coref_expr = cols[-1].split('|')
                        if debug: print("coref_expr", coref_expr)
                        if not coref_expr:
                            raise ValueError("Coref expression empty " + line)
                        for tok in coref_expr:
                            if debug: print("coref tok", tok)
                            try:
                                match = re.match(r"^(\(?)(\d+)(\)?)$", tok)
                            except:
                                print("error getting coreferences for line " + line)
                            num = match.group(2)
                            assert (num is not ''), "Error parsing coref " + tok + " in " + line
                            if match.group(1) == '(':
                                if debug: print("New coref", num)
                                corefs.append({'label': num, 'start': index, 'end': None})
                            if match.group(3) == ')':
                                j = None
                                for i in range(len(corefs)-1, -1, -1):
                                    if debug: print("i", i)
                                    if corefs[i]['label'] == num and corefs[i]['end'] is None:
                                        j = i
                                        break
                                assert (j is not None), "coref closing error " + line
                                if debug: print("End coref", num)
                                corefs[j]['end'] = index + 1
                    tokens.append(cols[3].replace(u'/', ''))
                    index += 1
                else:
                    raise ValueError("Line not standard " + line)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATA_PATH = sys.argv[1]
        SAVE_DIR = DATA_PATH + "/numpy/"
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        else:
            if os.listdir(SAVE_DIR):
                print("There are already data in", SAVE_DIR)
                print("Erasing")
                for file in os.listdir(SAVE_DIR):
                    print(file)
                    os.remove(SAVE_DIR + file)
            ConllDataset(DATA_PATH, SAVE_DIR)
