# coding: utf8
"""Conll parser"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import re
import sys
import os
import spacy

from neuralcoref.docs import Mention, Docs, Speaker

class ConllDoc(Docs):
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
                                  self.n_sents, self.speakers[speaker_id], coref['label'])
            self.mentions.append(mention)
#            print("mention: ", mention, "label", mention.gold_label)
        self.utterances.append(doc)
        self.utterances_speaker.append(self.speakers[speaker_id])
        self.n_sents += len(list(doc.sents))

    def set_conll_features(self):
        self.set_mentions_features()

class ConllDataset:
    def __init__(self, data_path, nlp=None):
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
        self.conll_docs = []
        self.load_dataset(data_path)

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
                              "\n utterances", len(doc.utterances), "\n", doc.utterances,
                              "\n mentions", len(doc.mentions), "\n", doc.mentions,
                              "\n speakers", len(doc.speakers), "\n", doc.speakers)
                        self.conll_docs.append(doc)
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
                        if debug: print("New doc", name, part)
                        doc = ConllDoc(name, part, nlp=self.nlp)
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
                                corefs.append({'label': num,'start': index, 'end': None})
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

    def load_dataset(self, path):
        for dirpath, _, filenames in os.walk(path):
            for filename in [f for f in filenames if f.endswith(".v4_auto_conll")]:
                full_name = os.path.join(dirpath, filename)
                print("Importing", full_name)
                self.load_file(full_name)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DATASET = ConllDataset(sys.argv[1])
