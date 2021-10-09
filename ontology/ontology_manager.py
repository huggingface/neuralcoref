from random import sample
import glob, os, re
import multiprocessing

import gzip
import  os, argparse
import itertools
from collections import Counter, OrderedDict
import os
import json
import threading
import numpy as np
import os
import time
import json
import copy

from time import time
import numpy as np
from collections import Counter
from itertools import chain
import glob
import json
import math, os
import random
import transformers
import sys, os
import json
import faker
import gzip
from faker.providers import person, job

from collections import Counter
import re
import gzip
import urllib
import re
from transformers import AutoTokenizer
from nltk.corpus import stopwords
mt5_underscore= "▁"
trannum = str.maketrans("0123456789", "1111111111")

class OntologyManager:
  """ 
  Basic ontology manager. Stores the upper ontology and lexicon that
  maps to the leaves of the ontology.  Has functions to determine
  whether a word is in the ontology, and to tokenize a sentence with
  words from the ontology.
  """

  default_strip_chars="-,~`.?!@#$%^&*(){}[]|\\/-_+=<>;'\""
  stopwords = set(stopwords.words())
  x_lingual_onto_name = "yago_cn_wn"

  def __init__(self, target_lang="en", data_dir="./pii_pro/data/",  shared_dir=None, max_word_len=4, compound_word_step =3,  strip_chars=None,  \
                 upper_ontology=None,  x_lingual_lexicon_by_prefix_file="lexicon_by_prefix.json.gz", target_lang_config_file=None, x_lingual2ner_file=None, \
                 connector = "_"):
    self._max_lexicon = 0
    if data_dir is None:
      data_dir = "./"
    if shared_dir is None: shared_dir=data_dir
    self.shared_dir = shared_dir
    self.data_dir = data_dir
    if strip_chars is None:
      strip_chars = self.default_strip_chars
    self.strip_chars = strip_chars
    self.connector = connector
    self.max_word_len = max_word_len
    self.compound_word_step = compound_word_step
    self.ontology = OrderedDict()
    self.load_upper_ontology(upper_ontology)
    self.load_x_lingual_lexicon_from_prefix_file(x_lingual_lexicon_by_prefix_file)
    self.load_x_lingual_lexicon_from_x_lingual2ner_file(x_lingual2ner_file)
    self.load_target_lang_config(target_lang_config_file, target_lang=target_lang)
    #used for cjk processing
    self.mt5_tokenizer = None
    if target_lang in ("zh","ja", "ko"):
      self.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

  def load_upper_ontology(self, upper_ontology):
    # TODO: load and save from json file
    if upper_ontology is None: upper_ontology =  {}

    self.upper_ontology = {}

    for key, val in upper_ontology.items():
      key = key.upper()
      if key not in self.upper_ontology:
        self.upper_ontology[key] = [val, len(self.upper_ontology), None]
      else:
        self.upper_ontology[key] = [val, self.upper_ontology[key][1]]
  
  def load_x_lingual_lexicon_from_x_lingual2ner_file(self, x_lingual2ner_file):
    data_dir = self.data_dir
    shared_dir = self.shared_dir
    if x_lingual2ner_file is None: return
    if os.path.exists(x_lingual2ner_file):
      word2ner = json.load(open(x_lingual2ner_file, "rb"))
      self.add_to_ontology(word2ner, onto_name="yago_cn_wn")
    elif os.path.exists(os.path.join(data_dir, x_lingual2ner_file)):
      word2ner = json.load(open(os.path.join(data_dir, x_lingual2ner_file), "rb"))
      self.add_to_ontology(word2ner, onto_name=self.x_lingual_onto_name)
    else:
      print ("warning: could not find x_lingual2ner_file")

  def load_x_lingual_lexicon_from_prefix_file(self, x_lingual_lexicon_by_prefix_file="lexicon_by_prefix.json.gz"):
    data_dir = self.data_dir
    shared_dir = self.shared_dir
    if x_lingual_lexicon_by_prefix_file is not None:
      if not os.path.exists(x_lingual_lexicon_by_prefix_file):
        x_lingual_lexicon_by_prefix_file = f"{data_dir}/{x_lingual_lexicon_by_prefix_file}"
      if not os.path.exists(x_lingual_lexicon_by_prefix_file): 
        self.x_lingual_lexicon_by_prefix = {}
        self.ontology[self.x_lingual_onto_name] = self.x_lingual_lexicon_by_prefix
        return
      if x_lingual_lexicon_by_prefix_file.endswith(".gz"):
        with gzip.open(x_lingual_lexicon_by_prefix_file, 'r') as fin:  
          json_bytes = fin.read()                     
          json_str = json_bytes.decode('utf-8')            
          self.x_lingual_lexicon_by_prefix = json.loads(json_str)
      else:
        self.x_lingual_lexicon_by_prefix = json.load(open(x_lingual_lexicon_by_prefix_file, "rb"))
      for lexicon in self.x_lingual_lexicon_by_prefix.values():
        for val in lexicon[-1].values():
          label = val[0][0]
          if label not in self.upper_ontology:
            self.upper_ontology[label] = [val[0], len(self.upper_ontology)]
          label = self.upper_ontology[label][0]
          val[0] = label
          self._max_lexicon  = max(self._max_lexicon, val[1])
    else:
      self.x_lingual_lexicon_by_prefix = {}
    self.ontology[self.x_lingual_onto_name] = self.x_lingual_lexicon_by_prefix

  def save_x_lingual_lexicon_prefix_file(self, x_lingual_lexicon_by_prefix_file="lexicon_by_prefix.json.gz"):
    """ saves the base cross lingual leixcon """
    data_dir = self.data_dir
    shared_dir = self.shared_dir
    print (data_dir, x_lingual_lexicon_by_prefix_file)
    x_lingual_lexicon_by_prefix_file = x_lingual_lexicon_by_prefix_file.replace(".gz", "")
    if not x_lingual_lexicon_by_prefix_file.startswith(data_dir): 
      x_lingual_lexicon_by_prefix_file=f"{data_dir}/{x_lingual_lexicon_by_prefix_file}"  
    json.dump(self.x_lingual_lexicon_by_prefix,open(x_lingual_lexicon_by_prefix_file, "w", encoding="utf8"), indent=1)
    os.system(f"gzip {x_lingual_lexicon_by_prefix_file}")
    if shared_dir is not None and data_dir != shared_dir: os.system(f"cp {x_lingual_lexicon_by_prefix_file}.gz {shared_dir}")
    os.system(f"rm {x_lingual_lexicon_by_prefix_file}")

  def load_target_lang_config(self,  target_lang_config_file=None, target_lang=None):
    data_dir = self.data_dir
    shared_dir = self.shared_dir
    if target_lang_config_file is None:
      if os.path.exists(os.path.join(data_dir, f'{target_lang}.json')): 
        target_lang_config_file=  os.path.join(data_dir, f'{target_lang}.json')
    if target_lang_config_file is None: return
    if os.path.exists(target_lang_config_file):
      self.target_lang_config = json.load(open(target_lang_config_file, "rb"))
    else:
      self.target_lang_config = {}
    ner_regexes = {}
    if 'ner_regexes' in self.target_lang_config.items():
      ner_regexes = self.target_lang_config['ner_regexes']
      for regex in ner_regexes:
        if regex[1]:
          regex[1] = re.compile(regex[1], re.IGNORECASE)
        else:
          regex[1] = re.compile(regex[1])
    self.ner_regexes = ner_regexes

    #pronouns used for basic coref
    self.other_pronouns = set(self.target_lang_config.get('OTHER_PRONOUNS',[]))
    self.person_pronouns = set(self.target_lang_config.get('PERSON_PRONOUNS',[]))
    self.pronouns = set(list(self.other_pronouns) + list(self.person_pronouns))

    #these are used for aonymizing and de-biasing swapping. 
    #TODO: consider whether we want to create shorter/stemmed versions of these.
    self.binary_gender_swap = self.target_lang_config.get('binary_gender_swap', {})
    self.other_gender_swap = self.target_lang_config.get('other_gender_swap', {})
    self.en_pronoun2gender = self.target_lang_config.get('en_pronoun2gender', {})
    self.en_pronoun2pronoun = self.target_lang_config.get('en_pronoun2pronoun', {}) 
    self.en_pronoun2title = self.target_lang_config.get('en_pronoun2title', {})
    self.person2religion = self.target_lang_config.get('person2religion', {})  
    self.gender2en_pronoun = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.en_pronoun2gender.items()]))
    self.pronoun2en_pronoun = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.en_pronoun2pronoun.items()]))
    self.title2en_pronoun = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.en_pronoun2title.items()]))
    self.religion2person = dict(itertools.chain(*[[(b,a) for b in lst] for a, lst in self.person2religion.items()]))
    self.coref_window = self.target_lang_config.get('coref_window', [-1, -2, 1, 2])  #maybe this should be a parameter and not in the ontology
 
    #now specialize the ontology for target_lang and have no limit on the size of the words
    target_lang_ontology = {}
    for label, words in self.target_lang_config if type(self.target_lang_config) is list else self.target_lang_config.items():
      if label == label.upper():
        for word in words:
          target_lang_ontology[word] = label
    #print (target_lang_ontology)
    self.add_to_ontology(target_lang_ontology, max_word_len=100000, onto_name=os.path.split(target_lang_config_file)[-1].split(".")[0])

  def save_target_lang_config(self, target_lang_config_file):
    if target_lang_config_file is None: return
    data_dir = self.data_dir
    shared_dir = self.shared_dir
    json.dump(self.target_lang_config,open(f"{data_dir}/{target_lang_config_file}", "w", encoding="utf8"), indent=1)
    #os.system(f"gzip {data_dir}/{target_lang_config_file}")
    if shared_dir is not None and data_dir != shared_dir: os.system(f"cp {data_dir}/{target_lang_config_file} {shared_dir}")

  def _has_nonstopword(self, wordArr):
    for word in wordArr:
      if word.strip(self.strip_chars) not in self.stopwords:
        return True
    return False

  def _get_all_word_shingles(self, wordArr, max_word_len=None, create_suffix_end=True):
    """  create patterned variations (prefix and suffix based shingles) """
    lenWordArr = len(wordArr)
    if max_word_len is None: max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    wordArr1 = wordArr2 = wordArr3 = wordArr4 = None
    ret = []
    if lenWordArr > compound_word_step:
        # we add some randomness in how we create patterns
        wordArr1 = wordArr[:compound_word_step-1] + [wordArr[-1]]
        wordArr2 = [wordArr[0]] + wordArr[1-compound_word_step:] 
        wordArr1 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr1]
        wordArr2 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr2]
        ret.extend([tuple(wordArr1),tuple(wordArr2)])
        if create_suffix_end:
          wordArr3 = copy.copy(wordArr1)
          wordArr3[-1] = wordArr3[-1] if len(wordArr3[-1]) <=max_word_len else '*'+wordArr3[-1][len(wordArr3[-1])-max_word_len+1:]
          wordArr4 = copy.copy(wordArr2)
          wordArr4[-1] = wordArr4[-1] if len(wordArr4[-1]) <=max_word_len else '*'+wordArr4[-1][len(wordArr4[-1])-max_word_len+1:]
          wordArr3 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr3]
          wordArr4 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr4]
          ret.extend([tuple(wordArr3),tuple(wordArr4)])
    else: # lenWordArr <= compound_word_step
        wordArr1 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr]
        ret.append(tuple(wordArr1))
        if lenWordArr > 1 and create_suffix_end:
          wordArr2 = copy.copy(wordArr)
          wordArr2[-1] = wordArr2[-1] if len(wordArr2[-1]) <=max_word_len else '*'+wordArr2[-1][len(wordArr2[-1])-max_word_len+1:]
          wordArr2 = [w if len(w) <=max_word_len else w[:max_word_len] for w in wordArr2]
          ret.append(tuple(wordArr2))
    return [list(a) for a in set(ret)]

  def add_to_ontology(self, word2ner, max_word_len=None, onto_name=None):
    """
    Add words to the ontology. The ontology is stored in compressed
    form, using an upper ontology and several subsuming prefix based lexicon
    mappings. We try to generalize the lexicon by using subsequences of
    the words and compound words.  Each word is shortened to
    max_word_len. Compound words are connected by a connector.
    Compound words longer than compound_word_step are shortened to
    that length for storage purposes.  All words except upper ontology
    labels are lower cased.  Assumes cjk tokens have already been
    parsed by mt5 tokenizer.
    """
    if onto_name is None:
      onto_name = self.x_lingual_onto_name
    if onto_name == self.x_lingual_onto_name:
      self.x_lingual_lexicon_by_prefix = ontology = self.ontology[onto_name] = self.ontology.get(onto_name, self.x_lingual_lexicon_by_prefix)
    else:
      ontology = self.ontology[onto_name] = self.ontology.get(onto_name, {})
    if max_word_len is None: max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    connector = self.connector
    if type(word2ner) is dict:
      word2ner = list(word2ner.items())
    lexicon = {}
    _max_lexicon = self._max_lexicon 
    for _idx, word_label in enumerate(word2ner):
      _idx += _max_lexicon
      word, label = word_label
      #if word.startswith('geor'): print (word, label)
      label = label.upper()
      if label not in self.upper_ontology:
        self.upper_ontology[label] = [[label], len(self.upper_ontology)]
      word = word.strip().lower().translate(trannum).replace(" ",connector)
      wordArr = word.split(connector)
      wordArr = [w2.strip(self.strip_chars) for w2 in wordArr if w2.strip(self.strip_chars)]
      #print (word)
      while wordArr:
        if wordArr[0] in self.stopwords:
          wordArr= wordArr[1:]
        else:
          break
      #while wordArr:
      #  if wordArr[-1] in self.stopwords:
      #    wordArr= wordArr[:-1]
      #  else:
      #    break
      if not wordArr:
        continue
      word = connector.join(wordArr)
      #we don't have an actual count of the word in the corpus, so we create a weight based 
      #on the length, assuming shorter words with less compound parts are more frequent
      weight = 1/(1.0+math.sqrt(len(word) + len(wordArr)))
      lenWordArr = len(wordArr)
      if lenWordArr == 0: 
        continue
      # add some randomness and only do suffix ends in some cases. TODO: we can use a config var.
      for wordArr in self._get_all_word_shingles(wordArr, max_word_len=max_word_len, create_suffix_end = _idx % 5 == 0):
        if not wordArr: continue
        word = connector.join(wordArr)
        key = (word, lenWordArr//(compound_word_step+1))
        #print (word0, word, weight)
        if type(label) in (list, tuple, set):
          if type(label) != list:
            label = list(label)
          _label, _idx, _cnt = lexicon.get(key, [label, _idx, {}])
          if _cnt is None: _cnt = {}
          _cnt[label[0]] = _cnt.get(label[0], 0.0) + weight
          lexicon[key] = [_label, _idx, _cnt]
        else:
          _label, _idx, _cnt = lexicon.get(key, [[label], _idx, {}])
          if _cnt is None: _cnt = {}
          _cnt[label] = _cnt.get(label, 0.0) + weight
          lexicon[key] = [_label, _idx, _cnt]
        prev_val= ontology.get(wordArr[0], [1, 100])
        ontology[wordArr[0]] = [max(lenWordArr, prev_val[0]), 2 if lenWordArr == 2 else min(max(lenWordArr-1,1), prev_val[1])]
    for key in lexicon:
      _cnt = lexicon[key][2]
      if _cnt:
        label = Counter(_cnt).most_common(1)[0][0]
        lexicon[key][0] = lexicon.get(label, [[label]])[0]
        lexicon[key] = lexicon[key][:-1]
    for word, slot in lexicon:
      prefix = word.split(connector,1)[0]
      if prefix in ontology:
        rec = ontology[prefix]
        if len(rec) == 2:
          rec.append({})
          rec.append({})
          rec.append({})
          rec.append({})
        lexicon2 = rec[2+min(3,slot)]
        if connector in word:
          word2 = '*'+connector+word.split(connector,1)[1]
        else:
          word2 = word
        lexicon2[word2] = lexicon[(word, slot)]
    self._max_lexicon += len(word2ner)

  def cjk_pre_tokenize(self, text, connector=None):
    """ tokenize using mt5. meant for cjk languages"""
    if connector is None:
      connector = self.connector
    if self.mt5_tokenizer is None:
      self.mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
    text = " ".join(self.mt5_tokenizer.tokenize(text.strip())).replace(mt5_underscore," ").replace("  ", " ").replace("  ", " ").strip()
    return text

  def in_ontology(self, word, connector=None, do_mt5_tokenize=False):
    """ find whether a word is in the ontology. """
    orig_word = word
    max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    if connector is None:
      connector = self.connector
    if do_mt5_tokenize:
      word = self.cjk_pre_tokenize(word, connector)
    word = word.strip().lower().translate(trannum) 
    wordArr = word.replace(" ",connector).split(connector) 
    wordArr = [w2.strip(self.strip_chars) for w2 in wordArr if w2.strip(self.strip_chars)]
    if not wordArr:
      return word, None
    lenWordArr = len(wordArr)
    all_shingles = self._get_all_word_shingles(wordArr, max_word_len=max_word_len)
    long_shingles= self._get_all_word_shingles(wordArr, max_word_len=100000, create_suffix_end=False)
    for ontology in reversed(list(self.ontology.values())):
      #find patterned variations (shingles)
      for wordArr in long_shingles + all_shingles: # we can probably dedup to make it faster
        if wordArr and wordArr[0] in ontology:
          lexicon2 = ontology[wordArr[0]][2+min(3,lenWordArr//(compound_word_step+1))]
          if len(wordArr) > 1:
            word = '*'+connector+connector.join((wordArr[1:]))
          else:
            word = wordArr[0]
          label, _ = lexicon2.get(word, (None, None))
          if label is not None:
            label = label[0]
            return word, label
    return orig_word, None

  def _get_ngram_start_end(self, start_word):
    """ find the possible range of a compound word that starts with start_word """
    ngram_start = -1
    ngram_end = 100000
    for ontology in self.ontology.values():
      rec = ontology.get(start_word, [ngram_start, ngram_end])
      ngram_start, ngram_end = max(ngram_start,rec[0]), min(ngram_end,rec[1])
    return ngram_start, ngram_end
        

  def tokenize(self, text, connector=None, do_mt5_tokenize=False, return_dict=False):
    """
    Parse text for words in the ontology.  For compound words,
    transform into single word sequence, with a word potentially
    having a connector seperator.  Optionally, use the mt5 tokenizer
    to separate the words into subtokens first, and then do multi-word
    parsing.  Used for mapping a word back to an item in an ontology.
    Returns the tokenized text along with word to ner label mapping
    for words in this text.
    """
    max_word_len =self.max_word_len
    compound_word_step = self.compound_word_step
    labels = []
    if connector is None:
      connector = self.connector
    if do_mt5_tokenize:
      text = self.cjk_pre_tokenize(text, connector)
    sent = text.strip().split()
    len_sent = len(sent)
    pos = 0
    for i in range(len_sent-1):
      if sent[i] is None: continue
      start_word =  sent[i].lower().strip(self.strip_chars) 
      if start_word in self.stopwords: 
        pos += len(sent[i])+1
        continue
      start_word = start_word.translate(trannum).split(connector)[0]
      start_word = start_word if len(start_word) <=  max_word_len else start_word[:max_word_len]
      ngram_start, ngram_end = self._get_ngram_start_end(start_word)
      if ngram_start > 0:
        for j in range(ngram_start-1, ngram_end-2, -1):
          if len_sent - i  > j:
            wordArr = sent[i:i+1+j]
            new_word = " ".join(wordArr)
            if not self._has_nonstopword(wordArr): break
            _, label = self.in_ontology(new_word, connector=connector, do_mt5_tokenize=do_mt5_tokenize)
            if label is not None:
              new_word = new_word.replace(" ", connector)
              #print ('found', new_word)
              sent[i] = new_word
              labels.append(((pos, pos + len(new_word)), (label, new_word,)))
              for k in range(i+1, i+j+1):
                sent[k] = None  
              break
      pos += len(sent[i])+1
    if return_dict:
      return {'text': " ".join([s for s in sent if s]), 'span2ner': dict(labels)}   
    else:
      return " ".join([s for s in sent if s]) 

  def cjk_detect(self, texts):
    # korean
    if re.search("[\uac00-\ud7a3]", texts):
        return "ko"
    # japanese
    if re.search("[\u3040-\u30ff]", texts):
        return "ja"
    # chinese
    if re.search("[\u4e00-\u9FFF]", texts):
        return "zh"
    return None

if __name__ == "__main__":  
  try:
    data_dir = shared_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
  except:
    data_dir = shared_dir = "./"
  if "-s" in sys.argv:
    shared_dir = sys.argv[sys.argv.indexof("-s")+1]
  if "-t" in sys.argv:
    sentence = sys.argv[sys.argv.indexof("-t")+1]
    manager = OntologyManager(data_dir=data_dir, shared_dir=shared_dir)
    txt = manager.tokenize(sentence)
    print(txt)
  
