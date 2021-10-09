# coding=utf-8
# Copyright, 2021 Ontocord, LLC, All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from time import time
import numpy as np
from collections import Counter
from itertools import chain
import os
import re
import glob
import math
import difflib
import random
import nltk
from random import choice
import spacy, neuralcoref, itertools#, blackstone
from collections import Counter, OrderedDict
trannum = str.maketrans("0123456789", "1111111111")
import torch
from transformers import pipeline, XLMRobertaForTokenClassification, BertForTokenClassification, AutoTokenizer
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from pii_pro.pii.round_trip_trans import RoundTripTranslate
from pii_pro.ontology.ontology_manager import OntologyManager

class Processor (OntologyManager):
  """
  A multi-lingual PII/NER processor.  
  
  Pre-processes text into chunks of NER/PII words and coreference tagged words. 

  Uses a multilingual onology and regex rules to obtain PII/NER labels and coreference labels.

  Note that Spacy parsing (and Neuralcoref https://github.com/huggingface/neuralcoref) are performed in English. 
  We use round-trip translation to perform NER in English, and then translate back to target lang.

  Provides basic multi-lingual functionality using a multilingual ontology and regex rules, round-trip-translation and HF ner transformer pipelines. 

  See OntologyManager for the defintion of the upper_ontology.

  Spacy's NER Tag:

      PERSON - People, including fictional.
      NORP - Nationalities or religious or political groups.
      FAC - Buildings, airports, highways, bridges, etc.
      ORG - Companies, agencies, institutions, etc.
      GPE - Countries, cities, states.
      LOC - Non-GPE locations, mountain ranges, bodies of water.
      PRODUCT - Objects, vehicles, foods, etc. (Not services.)
      EVENT - Named hurricanes, battles, wars, sports events, etc.
      WORK_OF_ART - Titles of books, songs, etc.
      LAW - Named documents made into laws.
      LANGUAGE - Any named language.
      DATE - Absolute or relative dates or periods.
      TIME - Times smaller than a day.
      PERCENT - Percentage, including ”%“.
      MONEY - Monetary values, including unit.
      QUANTITY - Measurements, as of weight or distance.
      ORDINAL - “first”, “second”, etc.
      CARDINAL - Numerals that do not fall under another type.

  """
  # base en spacy models
  nlp = None

  model_loaded = {}

  #which one to use?
  #jplu/tf-xlm-r-ner-40-lang
  #gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner
  default_ner_model = ("wietsedv/bert-base-multilingual-cased-finetuned-conll2002-ner", BertForTokenClassification) 
  
  lang2ner_model = {
      "sw": ("Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification ), # consider using one of the smaller models
      "yo": ("Davlan/xlm-roberta-large-masakhaner", XLMRobertaForTokenClassification ), 
      "ar": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "de": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "en": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "es": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "it": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "nl": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "la": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "pt": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "fr": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
      "zh": ("Davlan/xlm-roberta-base-ner-hrl", XLMRobertaForTokenClassification ),
  }

  def __init__(self, target_lang="en", data_dir="./",  max_onto_len=4, max_compound_word =3, ner_regexes=None,  strip_chars=None,  \
              domain_ontology_paths=None, upper_ontology=None, ontology=None, ontology_compound_word_start=None, \
              connector = "_", en_spacy_models=[]):
    super().__init__(target_lang=target_lang, data_dir=os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                                                          os.path.pardir, 'data')),  \
                       max_onto_len=max_onto_len, max_compound_word =max_compound_word, ner_regexes=ner_regexes,  strip_chars=strip_chars,  \
                       domain_ontology_paths=domain_ontology_paths, upper_ontology=upper_ontology, ontology=ontology, ontology_compound_word_start=ontology_compound_word_start, \
                       connector = connector)
    self.target_lang = target_lang

    if False:
      #hf stuff. we assume we are working only in CPU mode.
      if target_lang in self.lang2ner_model:
        model_name, cls = self.lang2ner_model[target_lang]
        if model_name in self.model_loaded:
          model = self.model_loaded[model_name]
        else:
          model = cls.from_pretrained(model_name)
          model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
          self.model_loaded[model_name] = model
        self.ner_model_pipeline = pipeline("ner", model=model, tokenizer=AutoTokenizer.from_pretrained(model_name))
      else:
        model_name, cls = self.default_ner_model
        model = cls.from_pretrained(model_name)
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.ner_model_pipeline = pipeline("ner", model=model, tokenizer=AutoTokenizer.from_pretrained(model_name))

    #spacy stuff
    #we are storing the nlp object as a class variable to save on loading time. 
    #we can abstract this to permit configuring a pipeline of en spacy models.
    self.en_spacy_models = []
    if Processor.nlp is None:
      Processor.nlp = spacy.load('en_core_web_lg')
      #we could increase the ontology by adding more words here
      #e.g., conv_dict={"Angela": ["woman", "girl"]} 
      coref = neuralcoref.NeuralCoref(Processor.nlp.vocab) #, conv_dict
      Processor.nlp.add_pipe(coref, name='neuralcoref')
    if en_spacy_models: 
      self.en_spacy_models = en_spacy_models
    else:
      self.en_spacy_models.append(Processor.nlp)
       

  def add_chunks_span(self, chunks, new_mention, old_mention, label, coref, chunk2ner, chunk2ref, ref2chunk):
    """ add a span to the chunks sequence and update the various ref and NER hashes """
    if old_mention in chunk2ner:
      del chunk2ner[old_mention]
    if label:
      chunk2ner[new_mention] = label
    if old_mention in chunk2ref:
      old_ref = chunk2ref[old_mention]
      ref2chunk[old_ref].remove(old_mention)
      if not ref2chunk[old_ref]:
        del ref2chunk[old_ref]
      del chunk2ref[old_mention]
    if new_mention in chunk2ref and coref != chunk2ref[new_mention]:
      old_ref = chunk2ref[new_mention]
      ref2chunk[old_ref].remove(new_mention)
      if not ref2chunk[old_ref]:
        del ref2chunk[old_ref]
      del chunk2ref[new_mention]
    if coref:
      chunk2ref[new_mention] = coref
      lst = ref2chunk.get(coref, [])
      if new_mention not in lst:
        ref2chunk[coref] = lst + [new_mention]
    chunks.append(new_mention)

  def del_ner_coref(self, old_mention, chunk2ner, chunk2ref, ref2chunk):
    """ remove an old_mention from the various NER and ref hashes """
    if old_mention in chunk2ner:
      del chunk2ner[old_mention]
    if old_mention in chunk2ref:
      old_ref = chunk2ref[old_mention]
      ref2chunk[old_ref].remove(old_mention)
      if not ref2chunk[old_ref]:
        del ref2chunk[old_ref]
      del chunk2ref[old_mention]

  def _spacy_ner_coref(self, text, nlp, chunk2ner, chunk2ref, ref2chunk):
    """ 
    Use the spacy English models to create chunks for English text
    and gather NER and coreference information
    """
    connector = self.connector
    pronouns = self.pronouns
    person_pronouns = self.person_pronouns
    other_pronouns = self.other_pronouns
    # spacy is not as high accuracy, but we use the spacey neuralcoref model so we can get pronoun coreference groups
    # to be able to do proper gender, race, job, etc. swapping. and we could potentially add new items to the vocabulary to improve coref.
    doc = nlp(text)

    #store away NOUNs for potential label and coref reference
    #rule for promotig a noun span into one considered for further processing:
    # - length of the number of words > 2 or length of span > 2 and the span is all uppercase (for abbreviations)
    # coref candidates:
    # - create an abbreviation from noun phrases as a candidate coref.
    # - use either the last two words a span as a candidate coref.
    # - use the abbreviation as a candidate coref
    for entity in list(doc.noun_chunks) + list(doc.ents) :
      chunk2ner[(entity.text, entity.start, entity.end, row_id, doc_id)]= "NOUN"
      mention_lower = entity.text.lower()
      textArr = mention_lower.split()
      if len(textArr) > 2:
        short_span = " ".join(textArr[-2:])
        ref2chunk[short_span] = ref2chunk.get(short_span, []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
        non_stopwords = [a for a in textArr if a not in self.stopwords]
        if len(non_stopwords) > 2:
          abrev = "".join([a[0] for a in non_stopwords])
          ref2chunk[abrev] = ref2chunk.get(abrev, []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]
      elif (len(entity.text) >=2 and entity.text == entity.text.upper()):
        ref2chunk[entity.text.lower()] = ref2chunk.get(entity.text.lower(), []) + [(entity.text, entity.start, entity.end, row_id, doc_id)]

    #store away coref NOUNs for potential label and coref reference
    #same rule as above for promoting a noun span into one considered for further processing.
    for cl in list(doc._.coref_clusters) :
      mentions = [(entity.text, entity.start, entity.end, row_id, doc_id) for entity in cl.mentions]
      mentions.sort(key=lambda e: len(e[0]), reverse=True)
      textArr = mentions[0][0].lower().split()
      for key in mentions:
        chunk2ner[key]= "NOUN"
      for mention in mentions:
        mention_lower = mention[0].lower()
        textArr = mention_lower.split()
        if mention_lower not in self.stopwords:
          if len(textArr) > 1:
            short_span = " ".join(textArr[-2:])
          else:
            short_span = textArr[0]
          ref2chunk[short_span] = ref2chunk.get(short_span, []) + mentions
          non_stopwords = [a for a in textArr if a not in self.stopwords]
          if len(non_stopwords) > 2:
            abrev = "".join([a[0] for a in non_stopwords])
            ref2chunk[abrev] = ref2chunk.get(abrev, []) + mentions

    #cleanup the chunk2ref, favoring large clusters with coref labels that are longer
    seen = {}
    corefs = [(a, list(set(b))) for a, b in ref2chunk.items()]
    corefs.sort(key=lambda a: a[0].count(" ")+len(a[1]), reverse=True)
    for coref, spans in corefs:
      new_spans = []
      spans = list(set(spans))
      spans.sort(key=lambda a: a[1]+(1.0/(1.0+a[2]-a[1])))
      spans2 = []
      for span in spans:
        if spans2 and spans2[-1][1] >= span[1]:
          continue
        spans2.append(span)
      for span in spans2:
        if span in seen: continue
        seen[span] = 1
        new_spans.append(span)
      del ref2chunk[coref]
      if new_spans:
        new_coref = [s[0] for s in new_spans]
        new_coref.sort(key=lambda a: len(a), reverse=True)
        ref2chunk[new_coref[0].lower()] = list(set(list(ref2chunk.get(new_coref[0].lower(), [])) + new_spans))

    chunk2ref.clear()
    for a, b1 in ref2chunk.items():
      for b in b1:
        chunk2ref[b] = a

    # expand coref information by using the most common coref label in a cluster
    if True:
      for cl in list(doc._.coref_clusters) :
        mentions = [(entity.text, entity.start, entity.end, row_id, doc_id) for entity in cl.mentions]
        all_mentions = list(set(itertools.chain(*[ref2chunk[chunk2ref[mention]] for mention in mentions if mention in chunk2ref])))
        corefs = [chunk2ref[mention] for mention in mentions if mention in chunk2ref]
        if corefs:
          coref = Counter(corefs).most_common()[0][0]
        else:
          coref = cl.main.text.lower()
        for mention in all_mentions:
          if mention not in chunk2ner:
            chunk2ner[mention] = 'NOUN'
          old_ref = chunk2ref.get(mention)
          if old_ref and mention in ref2chunk[old_ref]:
            ref2chunk[old_ref].remove(mention)
            if not ref2chunk[old_ref]:
              del ref2chunk[old_ref]
          chunk2ref[mention] = coref
          if mention not in ref2chunk[coref]:
            ref2chunk[coref].append(mention)

    #expand ner labels based on coref matches 
    for entity in list(doc.ents) :
      mention = (entity.text, entity.start, entity.end, row_id, doc_id)
      chunk2ner[mention]= entity.label_  
      if mention in chunk2ref:
        coref = chunk2ref[mention]
        for mention in ref2chunk[coref]:
          chunk2ner[mention] = entity.label_  

    # overwrite all ner labels in the coref cluster to PERSON if there is a person pronoun
    if True:
      for cl in list(doc._.coref_clusters):
        cluster_text_list = set([m.text.lower() for m in cl.mentions])
        # I don't use "us" because that is sometimes the U.S.
        # TODO: fix to check for upper case only US as an exception
        if "you" in cluster_text_list or "your"  in cluster_text_list  or "yours"  in cluster_text_list  or  "we" in cluster_text_list  or 'i' in cluster_text_list  or 'my' in cluster_text_list  or 'mine' in cluster_text_list or 'me' in cluster_text_list or 'he' in cluster_text_list or "she" in cluster_text_list or "his" in cluster_text_list or "her" in cluster_text_list or "him" in cluster_text_list or "hers" in cluster_text_list:
          label = "PERSON"
          for m in cl.mentions:
            chunk2ner[(m.text, m.start, m.end, row_id, doc_id)] = label
    return self._cleanup_chunks(text, chunks, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id, doc, mention_access_fn=lambda a: a.txt)

  def _hf_ner(self, text, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id):
    """
    run the text through a Huggingface ner pipeline for the target_lang text. 
    any tags found by this method that contradicts what was already tagged will take precedence. 
    For example, if we previously had categorized a noun as an ORG (which spacy sometimes does), but 
    the HF ner pipeline categorized it as a PERSON, then the tag will be changed to PERSON.
    """
    connector = self.connector
    pronouns = self.pronouns
    person_pronouns = self.person_pronouns
    other_pronouns = self.other_pronouns
    results = self.ner_model_pipeline(text)
    results.sort(lambda a: a['start'])
    prev_word = []
    prev_label = None
    prev_start = None
    for ner_result in results:
      if ner_result['entity'].startswith('I-'):
        if prev_label and ner_result['entity'].split("-")[-1].upper() != prev_label:
          start = ner_result['start']
          if not is_cjk and text[start] != ' ':
            for j in range(1, start):
              if start - j == -1 or text[start-j] == ' ':
                start = max(start -j, 0)
                break
          if text[start] == ' ': start += 1
          prev_word.append(text[start:].strip().split(" ",1)[0])
      if ner_result['entity'].startswith('B-'):
        start = ner_result['start']
        if not is_cjk and text[start] != ' ':
          for j in range(1, start):
            if start - j == -1 or text[start-j] == ' ':
              start = max(start -j, 0)
              break
        if text[start] == ' ': start += 1
        if prev_word:
          ner_word = " ".join(prev_word)
          mention = (ner_word, prev_start, prev_start+len(ner_word), row_id, doc_id)
          if mention not in chunk2ner or prev_label not in chunk2ner[mention]:
            if prev_label == 'PER': prev_label = 'PERSON'
            chunk2ner[mention] = prev_label
          prev_word = []
          prev_label = None
          prev_start = None
        prev_word.append(text[start:].strip().split(" ",1)[0])
        prev_label = ner_result['entity'].split("-")[-1].upper() 
        if prev_label == "MISC":
          prev_label = "NOUN"

        prev_start = ner_result['start']
    if prev_word:
        ner_word = " ".join(prev_word)
        mention = (ner_word, prev_start, prev_start+len(ner_word), row_id, doc_id)
        if mention not in chunk2ner or prev_label not in chunk2ner[mention]:
          if prev_label == 'PER': prev_label = 'PERSON'
          chunk2ner[mention] = prev_label

    return self._cleanup_chunks(text, chunks, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)

  def _rule_based_ner_coref(self, text, chunks2, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id, ner_regexes, number_parse_level=1, do_ontology=True):
    """
    do rule based ner and coref using regex, ontology and connect coreference using pronouns around a noun.
    regex matches can be word based.
    
    TODO: rule based matching.  e.g., rule chaining:
      <PUBLIC_FIGURE1> (Rd.|Street|Ave.) => <STREET_ADDRESS2>
      President <PERSON1> => <PUBLIC_FIGURE2>
      <LANGUAGE1> <GENDER2> => <RACE3> 
      *the numbers after the NER labels represents a unique id for the tag in the document. 
  
    """
    connector = self.connector
    pronouns = self.pronouns
    person_pronouns = self.person_pronouns
    other_pronouns = self.other_pronouns
    print (number_parse_level)
    for rng in range(number_parse_level):
      chunks = []
      len_chunks2 = len(chunks2)
      for spanIdx, mention in enumerate(chunks2):
        label = chunk2ner.get(mention)
        coref = chunk2ref.get(mention)
        self.del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
        span_str = mention[0]
        prev_words = []
        label2word = {}

        #regex matching
        #check surronding words for additional contexts when doing regexes.
        #prioritize those rules first that matching surronding words
        idx = 0
        wordsDict = dict([(s,1) for s in span_str.lower().split()])
        ner_regex_list1 = []
        ner_regex_list2 = []
        print (ner_regexes)
        for label2, regex_rule in ner_regexes:
          found=False
          for surronding_word in regex_rule[-1]:
            if surronding_word in wordsDict:
              ner_regex_list1.append((label2, regex_rule))
              found=True
              break
          if not found:
            ner_regex_list2.append((label2, regex_rule))
        for label2, regex_rule in ner_regex_list1 + ner_regex_list2:
            regex0 = regex_rule[0]
            for x in regex0.findall(span_str):
              if type(x) != str: continue
              span_str = span_str.replace(x.strip(), "<"+label2.upper()+">"+str(idx))
              if "<" in x and ">" in x:
                # expand out embedded tags
                x = ' '.join([a if a not in label2word else label2word[a] for a in x.strip().split()])
              label2word["<"+label2.upper()+">"+str(idx)] = (x.strip(), label2)
              idx += 1
        if do_ontology:
          span_str, word2label = self.tokenize(span_str)
        else:
          word2label = {}
        wordArr = [w2.strip(self.strip_chars) for w2 in span_str.split()]
        for word in wordArr:
          #first do poor man's coref
          #let's look at the previous nouns/ner we found
          found_chunks = []
          if not coref and word in pronouns:
            for idx in [i for i in self.coref_window if i <0]: 
              if chunk2ner.get(chunks[idx]) and chunks[idx][0].lower() not in pronouns:
                if word in person_pronouns and 'PERSON' not in self.ontology.get(chunk2ner.get(chunks[idx]), []): 
                  continue
                coref = chunk2ref.get(chunks[idx])
                found_chunks.append(chunks)
                break
          #now do the window between spans both prior and next
          if not coref and word in pronouns:
            for idx in [spanIdx + i for i in self.coref_window]:
              if idx >= 0 and idx < len_chunks2  and chunk2ner.get(chunks2[idx]) and chunks2[idx][0].lower() not in pronouns:
                if word in person_pronouns and 'PERSON' not in self.ontology.get(chunk2ner.get(chunks2[idx]), []): 
                  continue
                coref = chunk2ref.get(chunks2[idx])
                found_chunks.append(chunks)
                break
          #create a new coref group if there isn't one
          if not coref and found_chunks:
            coref = found_chunks[0][0].lower()
          
          #now see if the word is already regex labeled or is in the ontology and label accordingly 
          if word in label2word or word in word2label:
            if prev_words:
              new_word = " ".join(prev_words)
              len_new_word = len(new_word)
              self.add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, label, coref, chunk2ner, chunk2ref, ref2chunk)
            new_word = label2word[word][0] if word in label2word else word.replace(connector, " ")
            len_new_word = len(new_word)
            new_label = label2word[word][1] if word in label2word else (word2label[word] if word in word2label else label)
            self.add_chunks_span(chunks, (word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, new_label, coref, chunk2ner, chunk2ref, ref2chunk)
            prev_words = []
            continue
          prev_words.append(word)

        if prev_words:
          new_word = " ".join(prev_words)
          len_new_word = len(new_word)
          self.add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1, len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, label, coref, chunk2ner, chunk2ref, ref2chunk)
      
      #let's prepare for the next loop
      chunks2 = chunks
      do_ontology=False # save processing because we don't need ontology matching in subsequent loops of the parsing.

    text = " ".join([c[0] for c in chunks])
    
    return self._cleanup_chunks(text, chunks, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)


  def _cleanup_chunks(self, text, chunks, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id, doc=None, mention_access_fn=None):
    connector = self.connector
    pronouns = self.pronouns
    person_pronouns = self.person_pronouns
    other_pronouns = self.other_pronouns
    if doc is None: doc = text
    if mention_access_fn is None:
      mention_access_fn = lambda a: a
    
    # propogate the ner label to everything in the same coref group
    for coref, seq in ref2chunk.items():
      labels = [chunk2ner[mention]  for mention in seq if mention in chunk2ner and chunk2ner[mention] != 'NOUN']
      if labels:
        label = Counter(labels).most_common()[0][0]
        for mention in seq:
          if mention in chunk2ner and  label not in self.ontology.get(chunk2ner[mention], []): chunk2ner[mention] = label

    #add other words from the document into a sequence of the form (word, start_idx, end_idx, row_id, doc_id)
    #add in coref label into the sequence
    #clear duplicates and subsumed mentions 
    chunks = [a for a in chunk2ner.items() if a[0][-2] == row_id and a[0][-1] == doc_id] 
    chunks.sort(key=lambda a: a[0][1]+(1.0/(1.0+a[0][2]-a[0][1])))
    chunks2 = []
    # we may need to do this by doc and doc2. 
    for mention, label in chunks:
      if not chunks2 or (chunks2[-1][2] <= mention[1]):
        if not chunks2 or chunks2[-1][2] < mention[1]: 
          self.add_chunks_span(chunks2, (mention_access_fn(doc[0 if not chunks2 else chunks2[-1][2]: mention[1]]), 0 if not chunks2 else chunks2[-1][2], mention[1], row_id, doc_id), None, None, None, chunk2ner, chunk2ref, ref2chunk)
        self.add_chunks_span(chunks2, mention, None, label, chunk2ref.get(mention), chunk2ner, chunk2ref, ref2chunk)
      elif chunks2[-1][2] > mention[1] and chunks2[-1][1] <= mention[1]:
        if chunk2ner.get(chunks2[-1]) not in (None, '', 'NOUN'):
          self.del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
          continue
        elif label in  (None, '', 'NOUN'):
          self.del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
          continue
        old_mention = chunks2.pop()
        oldSpan = old_mention[0]
        oldLabel = chunk2ner.get(old_mention)
        oldAnaphore = chunk2ref.get(old_mention)
        sArr = oldSpan.split(mention[0], 1)
        self.del_ner_coref(old_mention, chunk2ner, chunk2ref, ref2chunk)
        s0 = sArr[0].strip()
        if s0:
          self.add_chunks_span(chunks2, (s0, old_mention[1], mention[1], row_id, doc_id), None, oldLabel if s0 in pronouns or (len(s0) > 1 and s0 not in self.stopwords) else None, oldAnaphore  if s0 in pronouns or (len(s0) > 1 and s0 not in self.stopwords) else None, chunk2ner, chunk2ref, ref2chunk)
        self.add_chunks_span(chunks2,  mention, None, label, oldAnaphore if not chunk2ref.get(mention) else chunk2ref.get(mention), chunk2ner, chunk2ref, ref2chunk)
        if len(sArr) > 1:
          s1 = sArr[1].strip()
          if s1:
            add_chunks_span(chunks2, (s1, mention[2], old_mention[2], row_id, doc_id), None,  oldLabel if s1 in pronouns or (len(s1) > 1 and s1 not in self.stopwords) else None, oldAnaphore  if s1 in pronouns or (len(s1) > 1 and s1 not in self.stopwords) else None, chunk2ner, chunk2ref, ref2chunk)
    len_doc = len(doc) 
    print (chunks2)
    if chunks2[-1][2] < len_doc:
      self.add_chunks_span(chunks2, (mention_access_fn(doc[chunks2[-1][2]:]), chunks2[-1][2], len_doc, row_id, doc_id), None, None, None, chunk2ner, chunk2ref, ref2chunk)

    #reset the indexes for chunks to be per character index.
    chunks = []
    len_chunks2 = len(chunks2)
    for spanIdx, mention in enumerate(chunks2):
      label = chunk2ner.get(mention)
      coref = chunk2ref.get(mention)
      self.del_ner_coref(mention, chunk2ner, chunk2ref, ref2chunk)
      new_word = mention[0]
      len_new_word = len(new_word)
      self.add_chunks_span(chunks, (new_word, 0 if not chunks else chunks[-1][2]+1,  len_new_word if not chunks else chunks[-1][2]+1+len_new_word, row_id, doc_id), None, label, coref, chunk2ner, chunk2ref, ref2chunk)
    text = " ".join([c[0] for c in chunks])

    # propogate the ner label to everything in the same coref group
    for coref, seq in ref2chunk.items():
      labels = [chunk2ner[mention]  for mention in seq if mention in chunk2ner and chunk2ner[mention] != 'NOUN']
      if labels:
        label = Counter(labels).most_common()[0][0]
        for mention in seq:
          if mention in chunk2ner and  label not in self.ontology.get(chunk2ner[mention], []): chunk2ner[mention] = label
    
    return text, chunks, chunk2ner, chunk2ref, ref2chunk

  def analyze_with_ner_coref(self, text,  target_lang=None, row_id=0, doc_id=0, chunk2ner=None, ref2chunk=None, chunk2ref=None):
    """
    Process multilingual NER on spans of text. Perform some coreference 
    matching (anaphoric matching). Use rules to expand and cleanup the coref and ner
    labeling.

    :arg text: The plaint text input or a JSON representation or a dict.
      If in JSON or dict form, the data will include row_id, doc_id, chunk2ner, ref2chunk and chunk2ref as described below.
    :arg row_id: Some unique id representing the sentence in a document or dataset.
    :arg doc_id: some unique document or dataset # 
    :arg chunk2ner: a dict that maps the chunk to an ner label, such as PERSON, ORG, etc. 
    :arg ref2chunk: a coreference group label that maps to the applicable chunk or chunks
    :arg chunk2ref: a dict that maps a chunk to a coreference group.
    
    Return a dict of form:
      {'text': text, 
      'chunks':chunks, 
      'chunk2ner': chunk2ner, 
      'ref2chunk': ref2chunk, 
      'chunk2ref': chunk2ref}. 

    Each chunks is in the form of a list of tuples:
      [(text_span, start_id, end_id, doc_id, row_id), ...]. 
    
    A note on terminology: A span is a segment of text of one or more words.  
    A mention is a chunk that is recognized by some processor.
    """
    connector = self.connector
    pronouns = self.pronouns
    person_pronouns = self.person_pronouns
    other_pronouns = self.other_pronouns
    ret={}
    if ref2chunk is None:
      ref2chunk = {}
    if chunk2ref is None:
      chunk2ref = {}
    if chunk2ner is None:
      chunk2ner = {}
    ontology = self.ontology
    ner_regexes = self.ner_regexes
    if not text: return ret
    if type(text) is dict:
      ret = text
    elif text[0] == '{' and text[-1] == '}':
      ret = json.loads(text)
    if ret:
      text = ret['text']
      doc_id = ret.get('doc_id', doc_id)
      row_id = ret.get('id', row_id)
      chunk2ner = ret.get('chunk2ner', chunk2ner)
      chunk2ref = ret.get('chunk2ref', chunk2ref)
      ref2chunk = ret.get('ref2chunk', ref2chunk)
    text = text.strip()
    if target_lang is None:
      target_lang = self.target_lang
    assert target_lang == self.target_lang or target_lang == 'en', "Can only processes in English or the target_lang"
    is_cjk = target_lang in ("zh", "ja", "ko")
    #we need to be able to find potential words broken on spaces for cjk langauges so we tokenize using mt5
    if is_cjk and self.t5_tokenizer is not None and self.detect_cjk(text):
        text = " ".join(self.mt5_tokenize(text)).replace(self.mt5_underscore+" ", self.mt5_underscore).strip()
    chunks = []
    chunks2 = []
    chunks3 = []
    #TODO: We could rank decisions by the spacy model, HF ner pipeline and the regex, ontology matcher and choose the max scoring decision.

    #first do rule based ner and coref processing using regex, ontology, etc. on the whole text
    text, chunks, chunk2ner, chunk2ref, ref2chunk = self._rule_based_ner_coref(text, [(text, 0, len(text), row_id, doc_id)], chunk2ner, chunk2ref, ref2chunk, row_id, doc_id, ner_regexes)

    #run the spacy model if this is English to get coreference and NER info.
    if target_lang == "en":
      for nlp in self.en_spacy_models:
        text, chunks2, chunk2ner, chunk2ref, ref2chunk = self._spacy_ner_coref(text, nlp, chunk2ner, chunk2ref, ref2chunk)

    #now let's run a HF transformer ner pipeline model to refine the tags
    if self.ner_model_pipeline is not None:
      text, chunks3, chunk2ner, chunk2ref, ref2chunk = self._hf_ner(text, chunk2ner, chunk2ref, ref2chunk)

    # do one final cleanup and resolve any conflicts
    text, chunks, chunk2ner, chunk2ref, ref2chunk = self._cleanup_chunks(text, chunks+chunks2+chunks3, chunk2ner, chunk2ref, ref2chunk, row_id, doc_id)

    ret['doc_id'] = doc_id
    ret['id'] = row_id
    ret['text'] = " ".join([c[0] for c in chunks])
    ret['chunks'] = chunks
    ret['chunk2ner'] = chunk2ner
    ret['chunk2ref'] = chunk2ref
    ret['ref2chunk'] = ref2chunk
    return ret

  def process(self, text="", batch=None, *args, **argv):
    """
    Process a single row of text or a batch. Performs basic chunking into roughly interesting phrases, and corefrence and PII/NER identification.
    """
    # need paramater to decide if we want to reset the various *2* hashes for each example in a batch?
    row_id=argv.get('row_id',0)
    doc_id=argv.get('doc_id',0)
    chunk2ner=argv.get('chunk2ner')
    ref2chunk=argv.get('ref2chunk')
    chunk2ref=argv.get('chunk2ref')
    target_lang = argv.get('target_lang', self.target_lang)
    if batch is None:
      return_single = True
      batch = [text]
    else:
      return_single = False
    ret = None
    if type(batch[0]) is str:
      if batch[0][0] == '{' and batch[0][-1] == '}':
        batch = [json.loads(text) for text in batch]
      else:
        batch = [{'id': _id+row_id, 'doc_id': doc_id, 'text': text} for _id, text in enumerate(batch)]
    if 'do_round_trip_trans' in argv and target_lang != 'en':
      if not hasattr(self, 'trans'):
        self.trans = Translate(target_lang=target_lang)
      batch_en = [self.trans.translate(text, target_lang='en') for text in batch]
      batch_en = [self.analyze_with_ner_coref(text, target_lang='en')  for text in batch_en]
      batch_trans = [self.trans.translate(text, target_lang=target_lang) for text in batch_en]
      batch= [self.analyze_with_ner_coref(text, target_lang=target_lang, chunk2ner=chunk2ner, ref2chunk=ref2chunk, chunk2ref=chunk2ref, )  for text in batch]
      ret = self.merge_ner_coref_predictions([batch, batch_trans])
    if ret is None:
      ret = [self.analyze_with_ner_coref(text, target_lang=target_lang, chunk2ner=chunk2ner, ref2chunk=ref2chunk, chunk2ref=chunk2ref, ) for text in batch]
    if return_single: 
      return ret[0]
    else:
      return ret

if __name__ == "__main__":  
  if "-t" in sys.argv:
    target_lang = sys.argv[sys.argv.index("-t")+1]    
    sentence = sys.argv[sys.argv.index("-t")+2]   
    pii_processor = Processor(target_lang=target_lang) 
    print (pii_processor.process(sentence))
