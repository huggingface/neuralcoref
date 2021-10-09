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
#from sklearn.cluster import AgglomerativeClustering
from time import time
import numpy as np
from nltk.corpus import wordnet as wn
from collections import Counter
from itertools import chain
import os
#from joblib import dump, load
#from joblib import Parallel, delayed
import glob
import os
import json
import math, os
import random
import transformers
mt5_underscore= "▁"
trannum = str.maketrans("0123456789", "1111111111")
import sys, os
import json
import faker

from faker.providers import person, job

from collections import Counter
import re
import gzip
import urllib
import re
from nltk.corpus import wordnet as wn
import transformers

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from pii_pro.ontology.ontology_manager import OntologyManager
from pii_pro.ontology.ontology_builder_data import OntologyBuilderData

class OntologyBuilder (OntologyManager, OntologyBuilderData):

  def __init__(self, data_dir="./pii_pro/data", shared_dir=None):
    OntologyManager.__init__(self, data_dir=data_dir, shared_dir=shared_dir)
    self.word2en = {}

  def load_cn_data(self):
    shared_dir = self.shared_dir
    data_dir = self.data_dir
    if not os.path.exists(f"{data_dir}/conceptnet-assertions-5.7.0.csv"):
      if not os.path.exists(f"{shared_dir}/conceptnet-assertions-5.7.0.csv.gz"):
        os.system(f"wget https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz")
        os.system(f"mv ./conceptnet-assertions-5.7.0.csv.gz {data_dir}")
        if shared_dir != data_dir: os.system(f"cp {data_dir}/conceptnet-assertions-5.7.0.csv.gz {shared_dir}")
      else:
        os.system(f"cp {shared_dir}/conceptnet-assertions-5.7.0.csv.gz {data_dir}")
      os.system(f"gzip -d {data_dir}/conceptnet-assertions-5.7.0.csv.gz")
    
  def create_wn_cat(self, keep_percentage=.01):
    """
    extract linkage from conceptnet words and wordnet category words
    """
    self.load_cn_data()
    shared_dir = self.shared_dir
    data_dir = self.data_dir
    if not os.path.exists(f"{shared_dir}/wn.csv"):
      os.system(f"grep '\/wn\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/wn.csv")
      os.system(f"mv {data_dir}/wn.csv {shared_dir}")
    wn =  open(f"{shared_dir}/wn.csv", "rb").read().decode().split('\n')
    wn = [s.split('\t')[0] for s in wn]
    wn = [s.strip(']').split(',/c/')[1:] for s in wn]

    wn_list = itertools.chain(*[[w.split("/")[1] if w.split("/")[-2] == "wn" else w.split("/")[-2] for w in awn if "_" in w] for awn in wn])
    items = list(Counter([w for w in wn_list if w[0] not in "0123456789"]).items())
    items.sort(key=lambda a:a[1], reverse=True)
    items = [i for i in items if i[1] != 1]
    typeHash = dict( items[:int(len(items)*keep_percentage)])
    wn_list2 = itertools.chain(*[[w for w in awn if (w.split("/")[2] == 'n') and (w.split("/")[0] != 'en') and ("_" in  w.split("/")[1]) and (w.split("/")[-2] in typeHash)] for awn in wn])
    items2 = list(Counter([w for w in wn_list2 if w[0] not in "0123456789"]).items())
    items2.sort(key=lambda a:a[1], reverse=True)
    return items2, typeHash

  def create_rel(self):
    """
    extract words that are related to each other based on conceptnet.
    We need to expand the connections between words to find more related nouns.
    Conceptnet maps words too broadly so we need to restrict the relationships.
    It's unclear if we should expand this much though. 
    """
    self.load_cn_data()
    shared_dir = self.shared_dir
    data_dir =  self.data_dir
    print (shared_dir, data_dir)
    if not os.path.exists(f"{shared_dir}/syn.csv"):
        os.system(f"grep '\/r\/Synonym\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/syn.csv")
        os.system(f"grep 'SimilarTo\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/sim.csv")
        #os.system(f"grep 'MannerOf\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/manner.csv")
        #os.system(f"grep 'DistinctFrom\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/dest.csv")
        os.system(f"grep 'DerivedFrom\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/deri.csv")
        #os.system(f"grep 'Antonym\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/anti.csv")
        os.system(f"grep 'EtymologicallyRelatedTo\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/erel.csv")
        os.system(f"grep 'EtymologicallyDerivedFrom\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/ederi.csv")
        os.system(f"grep 'RelatedTo\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/rel.csv")
        os.system(f"grep 'FormOf\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/formof.csv")
        os.system(f"grep 'IsA\/' {data_dir}/conceptnet-assertions-5.7.0.csv > {data_dir}/isa.csv")
        os.system(f"mv {data_dir}/syn.csv {data_dir}/sim.csv  {data_dir}/deri.csv  {data_dir}/erel.csv {data_dir}/ederi.csv {data_dir}/rel.csv {data_dir}/formof.csv {data_dir}/isa.csv {shared_dir}")
    rel2 = OrderedDict()
    for rel_type in ('syn', 'sim', 'deri', 'erel', 'ederi', 'rel', 'formof','isa') : #'dest', 'anti', 'manner', 
      i = 0
      with open(f"{shared_dir}/{rel_type}.csv", "rb") as f:
        while True:
          rel = f.readline()
          if not rel: break
          rel = rel.decode()
          rel = rel.split('\t')[0]
          rel = rel.strip(']').split(',/c/')[1:]
          if not rel:
            continue
          else:
            if len(rel) < 2:
              continue
            a = rel[0]
            b = rel[1]
            a = a.split('/')
            lang1 = a[0]
            a = a[1]
            b = b.split('/')
            lang2 = b[0]
            b = b[1]
            if a == b:
              continue
            val = [a,b]
            if len(a) > len(b):
              if a in rel2:
                val = rel2[a] + [a,b]
                del rel2[a]
              rel2[b]  =  rel2.get(b, []) + val
            else:
              if b in rel2:
                val = rel2[b] + [a,b]
                del rel2[b]
              rel2[a]  =  rel2.get(a, []) +  val
          #i+= 1
          #if i > 10000:
          #  break
      for key in list(rel2.keys()):
        rel2[key] = list(set(rel2[key]))
        
    return rel2

  def create_cn_ontology(self):
    shared_dir = self.shared_dir
    data_dir = self.data_dir
    if os.path.exists(f"{shared_dir}/conceptnet_ontology.json"):
      return json.load(open(f"{shared_dir}/conceptnet_ontology.json", "rb")), json.load(open(f"{shared_dir}/conceptnet_ontology_cat2word.json", "rb"))
    categories, typeHash = self.create_wn_cat(1.0)
    rel = self.create_rel()
    os.system(f"rm {data_dir}/conceptnet-assertions-5.7.0.csv")
    word2wncat = OrderedDict()
    #get a word to category mapping
    for concept, ct in categories:
      conceptArr = concept.split("/")
      word, wncat = conceptArr[1], conceptArr[-2]
      if wncat == 'linkdef':
        continue
      if word in word2wncat:
        if wncat != word2wncat[word]:
          word2wncat[word]='*'
      else:
        word2wncat[word] = wncat

    rel2=OrderedDict()
    rel3=OrderedDict()
    #group by categories, while expanding the categories
    for key, val in rel.items():
      if len(val) < 6:
        val2 = [(word2wncat[v] if v in word2wncat else (word2wncat[v.split("_")[0]] if v.split("_")[0] in word2wncat else word2wncat[v.split("_")[-1]]),v) for v in val if (v in word2wncat) or (v.split("_")[0] in word2wncat) or (v.split("_")[-1] in word2wncat)]
        if  val2: 
          val3 = [v for v in val if v not in val2]
          itr = itertools.groupby(val2, lambda x : x[0])
          groups = [(key, list(group)) for key, group in itr]
          groups.sort(key=lambda a: len(a[1]))
          max_cat = groups[-1][0]

          if max_cat != '*':
            # infer category of other words in this group if majority of labels is max_cat
            if len(groups[-1][1])*2 >= len(val):
              for word in val3:
                if word in word2wncat:
                  if wncat != word2wncat[word]:
                    word2wncat[word]='*'
                  else:
                    groups[-1][1].append((max_cat, word))
                else:
                  word2wncat[word] = max_cat
                  groups[-1][1].append((max_cat, word))
          all = {}
          for key, group in groups:
            if key == '*': continue
            group= list(group)
            if len(group) == 1:
              continue
            group = [g[1] for g in group]
            group.sort(key=lambda s: len(s))
            rel2[group[0]] = list(set(rel2.get(group[0],[]) + group))
            for g in group:
              all[g]=1
          val = [v for v in val if v not in all]
      if val:
        val.sort(key=lambda a: len(a))
        rel3[val[0]] = list(set(rel3.get(val[0],[])+val))

    #group by common prefix, infix, or suffix

    for key, val in rel3.items():
      val.sort(key=lambda a: len(a))
      len_val = len(val)
      for rng in range(0, len_val, 5):
          all = {}
          max_rng = min(rng+5, len_val)
          val2 = val[rng:max_rng]
          len_val2=len(val2)
          val2 = copy.deepcopy(val2)
          copy_val2 = copy.deepcopy(val2)
          for idx2, word in enumerate(copy_val2):
            if len(word) <= 4:
              continue
            for idx in range(idx2+1, len_val2):
              if type(val2[idx]) is tuple: 
                continue
              if type(val2[idx]) is str and (word in val2[idx] or val2[idx].startswith(word[:-1]) or val2[idx].startswith(word[:-2]) or val2[idx].startswith(word[:-3]) or val2[idx].endswith(word[1:]) or val2[idx].endswith(word[2:]) or val2[idx].endswith(word[2:])):
                val2[idx] = (word, val2[idx])
          val2 = [v for v in val2 if type(v) is tuple]
          itr = itertools.groupby(val2, lambda x : x[0])
          for key, group in itr:
            rel2[key] = list(set(rel2.get(key,[key]) + [v[1] for v in group]))
            all[key] = 1
            for v in group:
              all[v[1]] = 1
          #val3 = [v for v in val[rng:max_rng] if v not in all]
          #if val3:
          #  rel2[val3[0]] = val3

    print ('rel', len(rel), 'rel2', len(rel2), 'word2wncat', len(word2wncat))
    cat2word={}
    for key, value in word2wncat.items():
      cat2word[value]=cat2word.get(value,[])+[key]
    json.dump(rel2, open(f"{shared_dir}/conceptnet_ontology.json", "w", encoding="utf8"), indent=1)
    json.dump(cat2word, open(f"{shared_dir}/conceptnet_ontology_cat2word.json", "w", encoding="utf8"), indent=1)
    return rel2, cat2word
    
  def create_eng2multilang_dict(self):
      shared_dir = self.shared_dir
      if hasattr(self, 'word2lang') and self.word2lang: return
      if os.path.exists(f"{shared_dir}/conceptnet_en.json"):
        self.en = json.load(open(f"{shared_dir}/conceptnet_en.json", "rb"))
        self.word2en = json.load(open(f"{shared_dir}/conceptnet_word2en.json", "rb"))
        self.word2lang= json.load(open(f"{shared_dir}/conceptnet_word2lang.json", "rb"))
        return
      self.load_cn_data()
      if not os.path.exists(f"{shared_dir}/syn.csv"):
          os.system(f"grep '\/r\/Synonym\/' conceptnet-assertions-5.7.0.csv > {shared_dir}/syn.csv")
          
      mt5_tok = transformers.AutoTokenizer.from_pretrained("google/mt5-small")
      rel2 = OrderedDict()
      word2lang = {}
      for rel_type in ('syn', ) :
        i = 0
        rel =  open(f"{shared_dir}/{rel_type}.csv", "rb").read().decode().split('\n')
        rel = [s.split('\t')[0] for s in rel]
        rel = [s.strip(']').split(',/c/')[1:] for s in rel]
        for s in rel:
          if len(s) < 2:
            continue
          a = s[0]
          b = s[1]
          a = a.split('/')
          lang1 = a[0]
          
          a = a[1]
          b = b.split('/')
          lang2 = b[0]
          b = b[1]
          word2lang[a.replace(" ", "_").replace("-", "_").lower().strip(".")] = list(set(word2lang.get(a.replace(" ", "_").replace("-", "_").lower().strip("."), [])+[lang1]))
          word2lang[b.replace(" ", "_").replace("-", "_").lower().strip(".")] = list(set(word2lang.get(b.replace(" ", "_").replace("-", "_").lower().strip("."), [])+[lang2]))
          if a == b:
            continue
          if lang2 == 'en':
            tmp = b
            b = a
            a = tmp
          if lang1 in ('zh', 'ja', 'ko') and self.cjk_detect(a):
            a = "_".join(mt5_tok.tokenize(a)).replace(mt5_underscore,"_").replace("__", "_").replace("__", "_").strip("_")
            #print (a)
          if lang2 in ('zh', 'ja', 'ko') and self.cjk_detect(b):
            b = "_".join(mt5_tok.tokenize(b)).replace(mt5_underscore,"_").replace("__", "_").replace("__", "_").strip("_")
          val = [a,b]
          if lang1 != 'en' and lang2 != 'en': continue
          if  lang1 == 'en' and lang2 == 'en': continue
          if True:
            #if b in rel2:
            #  val = rel2[b] + [a,b]
            #  del rel2[b]
            rel2[a]  =  rel2.get(a, []) +  val
          i+= 1
          #if i > 10000:
          #  break

        for key in list(rel2.keys()):
          rel2[key] = list(set(rel2[key]))
      self.en = rel2
      word2en = {}
      for r, words in rel2.items():
        for word in words:
          word2en[word] = set(list(word2en.get(word, []))+ [r])
      self.word2en = word2en
      self.word2lang = word2lang
      for key in list(self.word2en.keys()):
        self.word2en[key] = dict([(a,1) for a in self.word2en[key]])
      json.dump(self.en, open(f"{shared_dir}/conceptnet_en.json", "w", encoding="utf8"), indent=1)
      json.dump(self.word2en, open(f"{shared_dir}/conceptnet_word2en.json", "w", encoding="utf8"), indent=1)
      json.dump(self.word2lang, open(f"{shared_dir}/conceptnet_word2lang.json", "w", encoding="utf8"), indent=1)


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

  def yago_step0(self):
    shared_dir = self.shared_dir
    data_dir = self.data_dir
    aHash = {}
    onto = {}
    catAll = {}
    if  os.path.exists(f"{data_dir}/yago0.tsv"):
      return
    if  not os.path.exists(f"{data_dir}/yago0.tsv") and os.path.exists(f"{shared_dir}/yago0.tsv"):
      os.system(f"cp {shared_dir}/yago0.tsv {data_dir}")
      return 
    if not os.path.exists(f"{shared_dir}/yago-wd-simple-types.nt.gz"):
      os.system("wget https://yago-knowledge.org/data/yago4/full/2020-02-24/yago-wd-simple-types.nt.gz")
      os.system(f"mv ./yago-wd-simple-types.nt.gz {data_dir}")
      if shared_dir != data_dir: os.system(f"cp {data_dir}/yago-wd-simple-types.nt.gz {shared_dir}")
    else:
      if shared_dir != data_dir: os.system(f"cp {shared_dir}/yago-wd-simple-types.nt.gz {data_dir}")
    with gzip.open(f"{data_dir}/yago-wd-simple-types.nt.gz") as f: #gzip.
      with open(f"{data_dir}/yago0.tsv", "w", encoding="utf8") as o:
        while True:
          line = f.readline()
          if not line: break
          line = line.decode()
          entity, rel, cat, _ = [l.split("/")[-1].strip(" .>") for l in line.split(">")]
          if "_" in entity:
            entityArr = entity.split("_")
            _id = entityArr[-1]
            entity = aHash[_id]= urllib.parse.unquote("_".join(entityArr[:-1]).split("(")[0].lower().replace("-","_").replace("__","_").strip("_")).strip(self.strip_chars)
          else:
            entity = aHash.get(entity, urllib.parse.unquote(entity.split("(")[0].lower().replace("-","_").replace("__","_").strip("_")).strip(self.strip_chars))
          if rel=="22-rdf-syntax-ns#type":
            if entity:
              cat=re.sub(r'(?<!^)(?=[A-Z])', '_', cat).upper()
              catAll[cat]=catAll.get(cat,0) + 1
              o.write(f"{entity}\t{cat}\n")
          else:
            print (rel)
    self.yago_cat_stats = catAll                                    
    os.system(f"sort --parallel=32 {data_dir}/yago0.tsv --o {data_dir}/yago0.tsv")
    os.system(f"cp {data_dir}/yago0.tsv {shared_dir}")

  def yago_step1(self):
    word2en = {}
    shared_dir = self.shared_dir
    data_dir = self.data_dir
    if  os.path.exists(f"{data_dir}/yago_ontology.tsv") or os.path.exists(f"{shared_dir}/yago_ontology.tsv"):
      if not os.path.exists(f"{data_dir}/yago_ontology.tsv"): os.system(f"cp {shared_dir}/yago_ontology.tsv {data_dir}")
      #word2en = json.load(open(f"{shared_dir}/yago_word2en.json", "rb"))
      #for word, en in word2en.items():
      #  if word.count("_") < 5 and en.count("_") < 5 and word not in self.word2en:
      #    self.word2en[word] = en
      return 
    if not os.path.exists(f"{shared_dir}/yago0.tsv"):
      self.yago_step0()
    elif not os.path.exists(f"{data_dir}/yago0.tsv"):
      os.system(f"cp {shared_dir}/yago0.tsv {data_dir}")

    _idx= 0
    with open(f"{data_dir}/yago_work_of_art.tsv", "w", encoding="utf8") as a:
      with open(f"{data_dir}/yago_problem.tsv", "w", encoding="utf8") as p:
        for _idx in range(2):
          prev_entity =""
          cats = []
          with open(f"{data_dir}/yago{_idx}.tsv", "rb") as f:
            _idx +=1
            with open(f"{data_dir}/yago{_idx}.tsv", "w", encoding="utf8") as o:
              while True:
                line = f.readline()
                if not line: break
                line = line.decode().strip()
                entity, cat = line.split("\t")
                entity=entity.strip(".")
                entityArr = entity.split("_")
                if entityArr[-1] in {"a", "and", "the", "or", "the", "of", "to", "in", "on", "from"}:
                  entity = "_".join(entityArr[:-1])
                cat = self.yago_upper_ontology.get(cat, cat)
                if "|" in entity:
                  entityArr = entity.split("|")
                  entity, en = entityArr[0], entityArr[1]
                  #print (entityArr)
                  #if len(entityArr) > 2:
                    #print (entityArr)
                  entity=entity.strip(".")
                  entityArr = entity.split("_")
                  if entityArr[-1] in {"a", "and", "the", "or", "the", "of", "to", "in", "on", "from"}:
                    entity = "_".join(entityArr[:-1])
                  word2en[entity] = en
                if prev_entity and entity != prev_entity:
                  if ('DOMAIN_NAME' in cats and not prev_entity.endswith(".com")) or ('WORK_OF_ART' in cats and not (prev_entity.count("_") <= 4 and (":" in prev_entity or prev_entity[-1] == "1"))):
                    a.write (prev_entity.translate(trannum)+"\n") 
                  else:
                    cats = [a for a in Counter(cats).most_common() if a[0] !='THING']
                    if cats and prev_entity.count("_") <= 4:
                      o.write (prev_entity.translate(trannum) +"\t" +cats[0][0]+"\n")
                    else:
                      p.write (prev_entity.translate(trannum)+"\n")
                  cats = [cat]
                  prev_entity = entity
                else:
                  cats.append(cat)
                  prev_entity = entity
              if cats:
                  if ('DOMAIN_NAME' in cats and not prev_entity.endswith(".com")) or ('WORK_OF_ART' in cats and not (prev_entity.count("_") <= 4 and (":" in prev_entity or prev_entity[-1] == "1"))):
                    a.write (prev_entity.translate(trannum)+"\n") 
                  else:
                    cats = [a for a in Counter(cats).most_common() if a[0] !='THING']
                    if cats and prev_entity.count("_") <= 4:
                      o.write (prev_entity.translate(trannum) +"\t" +cats[0][0]+"\n")
                    else:
                      p.write (prev_entity.translate(trannum)+"\n")
          os.system(f"sort --parallel=32 {data_dir}/yago{_idx}.tsv --o {data_dir}/yago{_idx}.tsv")
                                            
    os.system(f"cp {data_dir}/yago{_idx}.tsv {data_dir}/yago_ontology.tsv")
    os.system(f"cp {data_dir}/yago_ontology.tsv {shared_dir}")
    os.system(f"cp {data_dir}/yago_work_of_art.tsv {shared_dir}")
    os.system(f"cp {data_dir}/yago_problem.tsv {shared_dir}")
    #json.dump(word2en, open(f"{shared_dir}/yago_word2en.json", "w", encoding="utf8"), indent=1)
    #for word, en in word2en.items():
    #  if word.count("_") < 5 and en.count("_") < 5 and word not in self.word2en:
    #    self.word2en[word] = en

  def yago_step2(self):
    shared_dir = self.shared_dir
    data_dir = self.data_dir
    if os.path.exists(f"{data_dir}/yago_ontology.json") or os.path.exists(f"{shared_dir}/yago_ontology.json"):
      if not os.path.exists(f"{data_dir}/yago_ontology.json"): os.system(f"cp {shared_dir}/yago_ontology.json {data_dir}")
      return json.load(open(f"{data_dir}/yago_ontology.json"))

    if not os.path.exists(f"{data_dir}/yago_ontology.tsv"):
      self.yago_step1()

    Synset = wn.synset
    person = Synset('person.n.01') #person or job
    commodity = Synset('commodity.n.01') #product
    vehicle =  Synset('conveyance.n.03') #product
    artifact = Synset('artifact.n.01')
    plant = Synset('plant.n.02')
    molecule = Synset('molecule.n.01') #substance
    compound = Synset('compound.n.02')#substance
    scientist = Synset('scientist.n.01')
    leader = Synset('leader.n.01')
    capitalist = Synset('capitalist.n.02') # job
    event = Synset('event.n.01')
    animal = Synset('biological_group.n.01') #animal
    structure = Synset('structure.n.01')
    fac = Synset('facility.n.01')
    group = Synset('group.n.01')
    symptom = Synset('symptom.n.01')
    location = Synset('location.n.01')
    condition = Synset('condition.n.01') # disease
    body_part = Synset('body_part.n.01') #anat
    substance = Synset('substance.n.07')#sub, BIO_CHEM_ENTITY
    food = Synset('food.n.01') #product
    act = Synset('act.n.02') #medical_therapy
    process  = Synset('process.n.06')

    new_yago_ontology = {}
    mt5_tok = transformers.AutoTokenizer.from_pretrained("google/mt5-small")
    yago_dict = dict([a.split("\t") for a in open(f"{data_dir}/yago_ontology.tsv", "rb").read().decode().split("\n") if len(a.split("\t"))== 2 ])
    for word, label in yago_dict.items():
      if self.cjk_detect(word):
        word = word.replace("_","")
        word = "_".join(mt5_tok.tokenize(word)).replace(mt5_underscore,"_").replace("__", "_").replace("__", "_").strip("_")
      if label == 'MEDICAL_CONDITION':
          label='DISEASE'
      if label in ('WORK_OF_ART',):
        if ":" in word or word.count("_") > 1:
          new_yago_ontology[word] = label
        continue
      if self.cjk_detect(word):
        if len(word) > 1:
          new_yago_ontology[word] = label
        continue
      elif "_" not in word:
        continue

      synset= None
      try:
        synset= wn.synset(word+'.n.01')
      except:
        if label != 'PERSON':
          try:
            synset = wn.synset(word.split("_")[-1]+'.n.01')
          except:
            synset = None
      if synset is not None:
        hype =  (list(synset.closure(lambda a: a.hypernyms())))
        if label in ('MEDICAL_THERAPY', ):
          if act in hype or process in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('DISEASE', ):
          if condition in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('ANAT',):
          if body_part in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('PRODUCT',):
          if fac in hype or structure in hype:
            label = 'FAC' 
            new_yago_ontology[word] = label
            continue
          elif food in hype:
            label = 'FOOD'
            new_yago_ontology[word] = label
            continue
          elif commodity in hype or vehicle in hype or artifact in hype:
            new_yago_ontology[word] = label
        elif label in ('ANIMAL',):
          if plant in hype:
            label = 'PLANT'
            new_yago_ontology[word] = label
            continue
          elif animal in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('ORG',):
          if group in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('PERSON', 'JOB', ):
          if label == 'JOB' and (scientist in hype or leader in hype or capitalist in hype):
            new_yago_ontology[word] = label
            continue
          if person in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('SUBSTANCE', 'BIO_CHEM_ENTITY'):
          if substance in hype or molecule in hype or compound in hype:
            new_yago_ontology[word] = label
            continue
        elif label in ('GPE', 'LOCATION'):
          if location in hype:
            new_yago_ontology[word] = label
            continue
          elif fac in hype or structure in hype:
            label = 'FAC' 
            new_yago_ontology[word] = label
            continue
        elif label in ('FAC',):
          if location in hype and fac not in hype:
            label = 'LOCATION'
            new_yago_ontology[word] = label
            continue
        elif label in ('EVENT',):
          if event in hype:
            new_yago_ontology[word] = label
            continue
        if commodity in hype or vehicle in hype or artifact in hype or \
            plant in hype or molecule in hype or compound in hype or event in hype or \
            animal in hype or fac in hype or group in hype or symptom in hype or location in hype or \
            condition in hype or body_part in hype or substance in hype or food in hype or act in hype or \
            process in hype:
            continue    

      if label in ('PERSON',) or synset is None:
          new_yago_ontology[word] = label

    lst = list(new_yago_ontology.items())
    json.dump(lst, open(f"{data_dir}/yago_ontology.json", "w", encoding="utf8"), indent=1)
    os.system(f"cp {data_dir}/yago_ontology.json {shared_dir}")
    return lst

  def create_yago_ontology(self):
    return self.yago_step2()


  def create_combined_cn_yago(self):
    self.create_cn_ontology()
    self.create_eng2multilang_dict()
    shared_dir = self.shared_dir
    if hasattr(self, 'ner2word') and self.ner2word: return
    if os.path.exists(f"{shared_dir}/conceptnet_yago_combined.json"):
      self.ner2word = json.load(open(f"{shared_dir}/conceptnet_yago_combined.json", "rb"))
      self.yago2ner = [tuple(a) for a in json.load(open(f"{shared_dir}/yago2ner.json", "rb"))]
      return
    
    mt5_tok = transformers.AutoTokenizer.from_pretrained("google/mt5-small")

    cat2word_list = list(json.load(open(f"{shared_dir}/conceptnet_ontology_cat2word.json")).items())
    word2ner = {}
    ner2word = {}
    #cat2ner_map = dict([ ('person', 'PUBLIC_FIGURE'), ('body', 'ANAT'), ('artifact', 'PRODUCT'), ('medicine', 'MEDICAL_THERAPY',), ('plant', 'PLANT'), ('animal', 'ANIMAL'),  ('location', 'LOCATION'),  ('language', 'LANGUAGE'), ('substance', 'SUBSTANCE'), ('state', 'DISEASE'), ('group', 'ORG'), ('time', 'DATE'), ('law', 'LAW'), ('food', 'FOOD'), ('quantity', 'QUANTITY')])
    cat2ner_map = dict([ ('person', 'PUBLIC_FIGURE'),  ('medicine', 'MEDICAL_THERAPY',), ('plant', 'PLANT'), ('animal', 'ANIMAL'),   ('language', 'LANGUAGE'), ('substance', 'SUBSTANCE'),  ('group', 'ORG'), ('time', 'DATE'), ('law', 'LAW'), ('food', 'FOOD'), ('quantity', 'QUANTITY')])
    
    for cat, words in cat2word_list:
      label = cat2ner_map.get(cat)
      if label:
        for word in words:
          if self.cjk_detect(word):
            word = "_".join(mt5_tok.tokenize(word.replace("_", ""))).replace(mt5_underscore,"_").replace("__", "_").replace("__", "_").strip("_")   
            #if len(word) <= 1:
            #  continue
          if word in self.word2en:
            word2ner[word]= label

    yago0 = self.create_yago_ontology()
    yago2ner = []
    for word, label in yago0:
      if word in word2ner or word in self.word2en:
        if label == 'PERSON' and word  in word2ner  and word2ner[word] == 'PUBLIC_FIGURE': continue
        word2ner[word] = label
      else:
        yago2ner.append((word, label))
    yago0 =  None
    for word, label in word2ner.items():
      ner2word[label] = ner2word.get(label,[])+list(self.word2en[word])
    word2ner=None
    json.dump(ner2word, open(f"{shared_dir}/conceptnet_yago_combined.json", "w", encoding="utf8"), indent=1)
    json.dump(yago2ner, open(f"{shared_dir}/yago2ner.json", "w", encoding="utf8"), indent=1)
    self.ner2word = ner2word
    self.yago2ner = yago2ner

  def save_cross_lingual_ontology(self, word2ner_file = "word2ner.json"):
    shared_dir=self.shared_dir
    data_dir=self.data_dir
    self.create_eng2multilang_dict()
    self.create_combined_cn_yago()
    ner2word = self.ner2word
   
    yago2ner = [a for a in self.yago2ner if a[1] !='WORK_OF_ART']
    mt5_tok = transformers.AutoTokenizer.from_pretrained("google/mt5-small")

    # except for domain names, anat/body part and language we want to only consider compound words
    anat_en = list(set(ner2word['ANAT']))
    product_en = list(set(a for a in ner2word['PRODUCT'] if a.count("_") > 0))
    medical_therapy_en = list(set(a for a in ner2word['MEDICAL_THERAPY']if a.count("_") > 0))
    plant_en = list(set(a for a in ner2word['PLANT'] if a.count("_") > 0))
    animal_en = list(set(a for a in ner2word['ANIMAL'] if a.count("_") > 0))
    work_of_art_en = [] # list(set(a for a in ner2word['WORK_OF_ART'] if a.split("_")[0].lower() not in self.stopwords and a.count("_") > 0 and  a.count("_") < 5))
    gpe_en = list(set(a for a in ner2word['GPE'] if a.count("_") > 0))
    event_en = list(set(a for a in ner2word['EVENT'] if a.count("_") > 0))
    language_en = list(set(ner2word['LANGUAGE']))
    disease_en = list(set(a for a in ner2word['DISEASE'] if a.count("_") > 0))
    org_en = list(set(a for a in ner2word['ORG'] if a.count("_") > 0))
    date_en = list(set(a for a in ner2word['DATE'] if a.count("_") > 0))
    law_en = list(set(a for a in ner2word['LAW'] if a.count("_") > 0))
    food_en = list(set(a for a in ner2word['FOOD'] if a.count("_") > 0))
    domain_name_en = list(set(a for a in ner2word.get('DOMAIN_NAME',[]) if a.count("_")))
    quantity_en = list(set(a for a in ner2word['QUANTITY'] if a.count("_") > 0))
    union_en = list(set(a for a in ner2word.get('UNION',[]) if a.count("_") > 0))
    fac_en = list(set(a for a in ner2word['FAC'] if a.count("_") > 0))
    location_en = list(set(a for a in ner2word['LOCATION'] if a.count("_") > 0))
    bio_chem_en = list(set(a for a in ner2word['BIO_CHEM_ENTITY'] if a.count("_") > 0))

    # handle public figure and person specially
    block_list = set(['handyman', 'redeemer', 'attorney', 'kroeber', 'anthropologist', 'harte', 'bartholin', 'eames', 'snow', 'crocetti', 'dissident', 'donor', 'adrian', 'otis', 'arouet', 'lachaise', 'danton', 'casanova', 'mantell', 'bernini', 'dame', 'duchess', 'fechner', 'hovick', 'oersted', 'heinz', 'laurens', 'historian', 'didion', 'pro', 'ernst', 'positivist', 'rembrandt', 'clive', 'island', 'crouse', 'dali', 'arrhenius', 'bowdler', 'domitian', 'trader', 'nernst', 'blok', 'markov', 'bontemps', 'berliner', 'gilman', 'geologist', 'hirschsprung', 'bridges', 'lauder', 'hoagland', 'marcher', 'boundaries', 'giver', 'trilling', 'livermore', 'romancer', 'gardiner', 'benet', 'shawn', 'weld', 'brooks', 'enlai', 'apparent', 'samaritan', 'charmer', 'sukarno', 'hirschfeld', 'apes', 'dodger', 'avila', 'eustachio', 'rabbit', 'bullfinch', 'talleyrand', 'bakey', 'broglie', 'bardi', 'pitar', 'egypt', 'culbertson', 'haeckel', 'schrödinger', 'townsend', 'jacob', 'vinson', 'fahrenheit', 'arbiter', 'pickett', 'guess', 'vargas', 'piper', 'gram', 'heart', 'hoffmannsthal', 'robusti', 'purkinje', 'gris', 'millet', 'garnier', 'miro', 'schiller', 'carrere', 'feifer', 'maccabaeus', 'berzelius', 'jacobi', 'wolff', 'parr', 'muck', 'neel', 'xi', 'xii', 'xiii', 'xv', 'xvi', 'argote', 'beyle', 'african', 'lobachevsky', 'jumper', 'glendower', 'vernier', 'broca', 'hoax', 'ovid', 'terence', 'dolby', 'cavelier', 'lion', 'raskolnikov', 'laban', 'trooper', 'bolivar', 'mackenzie', 'wilkinson', 'bessemer', 'maxim', 'dewar', 'paget', 'reynolds', 'khama', 'haworth', 'girard', 'carew', 'vespasianus', 'susumu', 'christian', 'havel', 'balboa', 'columnist', 'ang', 'heisenberg', 'micawber', 'architect', 'cow', 'diver', 'banker', 'loci', 'clothing', 'bleu', 'bearer', 'journalist', 'gillette', 'government', 'widow', 'manufacturer', 'miró', 'hokusai', 'felix', 'weight', 'expert', 'mata', 'kaer', 'mariner', 'trajan', 'leather', 'vinci', 'communicator', 'type', 'toast', 'stockholder', 'shrine', 'salad', 'korea', 'acquaintance', 'montgomerie', 'dosè', 'machinis', 'european', 'hell', 'china', 'inflection', 'islands', 'admirals', 'good', 'fatales', 'monkey', 'university', 'chiannoort', 'bishop', 'puta', 'rivers', 'philosophers', 'dragons', 'colleges', 'grandparents', 'linguists', 'aunts', 'firsts', 'technicians', 'committees', 'eggs', 'test', 'guns', 'pornography', 'nests', 'associations', 'association', 'companions', 'sons', 'homme', 'intellectuals', 'market', 'cell', 'dating', 'composer', 'data', 'colors', 'request', 'pollutant', 'ibt', 'status', 'mexico', 'predicate', 'vulnerability', 'loire', 'today', 'weapons', 'series', 'analysis', 'sector', 'tract', 'game', 'creation', 'missile', 'odor', 'pea', 'district', 'sandwich', 'trades', 'realm', 'worshiper', 'eiffel', 'tzu', 'builder', 'buster', 'marshal', 'cohn', 'hueffer', 'ohm', 'alger', 'hogg', 'jensen', 'drew', 'po', 'langtry', 'flory', 'senator', 'pamby', 'fan', 'hakim', 'robber', 'devi', 'digger', 'chaser', 'holley', 'jerk', 'fallot', 'coca', 'tenens', 'kahn', 'niner', 'fly', 'dipper', 'relation', 'recruit', 'duddy', 'domo', 'witch', 'banneret', 'masoud', 'alexander', 'omikami', 'correggio', 'laurel', 'claudius', 'brinton', 'juvenalis', 'sider', 'know', 'lil', 'baptist', 'julian', 'bodoni', 'houghton', 'devries', 'alhazen', 'gloriosus', 'worcester', 'billings', 'middleweight', 'libertarian', 'din', 'depressive', 'madonna', 'ciccone', 'hershey', 'moto', 'source', 'thumb', 'presumptive', 'hohenheim', 'blonde', 'dakotan', 'hadrianus', 'virgil', 'quarters', 'tertullian', 'rolfe', 'winkle', 'goodfellow', 'langley', 'falstaff', 'sidney', 'daula', 'catcher', 'sepulcher', 'hastings', 'lucretius', 'voter', 'turk', 'kettle', 'protestant', 'resistant', 'anatomist', 'mitty', 'opel', 'speaker', 'bojaxhiu', 'corporal', 'cimabue', 'brontë', 'tussauds', 'nostradamus', 'ship', 'thales', 'climber', 'mercenary', 'xiaoping', 'dostoevsky', 'blimp', 'style', 'saxon', 'leonean', 'snowman', 'ruler', 'velazquez', 'wrestler', 'shaker', 'panther', 'sanzio', 'symbol', 'turkish', 'vecellio', 'petrarca', 'livius', 'hawk', 'nothing', 'phrase', 'accelerator', 'multiplier', 'potyokin', 'half', 'tom', 'bridge', 'priest', 'grandniece',  'financiers', 'эне', 'kozh', 'orchestra',  'centre', 'eminence', 'mauldin', 'donatello', 'steuben', 'compatible', 'dialogue', 'reference', 'subtree', 'investor', 'schüler', 'undertaking', 'fossil', 'shopkeeper', 'benefactor', 'calligrapher', 'storyteller', 'fireman', 'sized', 'ape', 'cobra', 'kong', 'jiom', 'xuan', 'mu', 'shang', 'soong', 'sasser', 'herod', 'freeman', 'japan', 'jin', 'mushroom', 'jeffords', 'jie', 'arabia', 'andronicus', 'compatriot', 'wang', 'legge', 'incompetent', 'qin', 'cake', 'beloved', 'cheerful', 'seal', 'diamond', 'force', 'bastard', 'ina', 'diamonds', 'voting', 'sinister', 'bar', 'mouse', 'blue', 'villains', 'points', 'point', 'professors', 'sib', 'maps', 'machine', 'integrity', 'galaxy', 'store', 'slavic', 'saladine', 'comedy', 'policemen', 'spuriae', 'women', 'five', 'goods', 'part', 'mayores', 'sauce', 'renaissance', 'nièces', 'interviewai', 'interviewée', 'aventure', 'saladines', 'employa', 'highness',  'negócios', 'couple', 'monarch', 'grade', 'pairs', 'effects', 'files', 'eigne', 'neighbor', 'relative', 'relatives', 'caesars', 'objects', 'things', 'bastards', 'administrators', 'brethren', 'wives', 'bulb', 'religionist', 'religionists', 'scholars', 'farmers', 'gab', 'granddaughters', 'porn', 'politicians', 'heads', 'cheetahs', 'emperors', 'grandchildren', 'riches', 'like', 'interest', 'swear', 'structures', 'ideals', 'ideal', 'chancellors', 'address', 'complement', 'complements', 'water', 'error', 'citizens', 'partners', 'dress', 'familiares', 'hommes', 'pères', 'maître', 'maternels', 'filles', 'sparrow', 'généraux', 'série', 'arricchiti', 'corte', 'lavoro', 'uranium', 'cinema',  'quadrinhos', 'бабушки', 'variante', 'competition', 'players', 'sex', 'w', 'living', 'love', 'pawn', 'queen', 'rook', 'newspaper', '1', '3', '4', '5', 'feature', 'check', 'news', 'computers', 'night', 'set', 'roads', 'locations', 'film', 'murder', 'spindle', 'else', 'shop', 'attack', 'oil', 'torino', 'designing', 'performer', 'village', 'expression', 'assignment', 'classical', 'weapon', 'human', 'category', '240', 'tool', 'minor', 'buoy', 'plants', 'murders', 'industry', 'care', 'gene', 'bible', 'movement', 'wisconsin', 'gynecologist', 'environment', 'habakkuk', 'noise', 'dc', 'cousin', 'book', 'shoe', 'vector', 'procedure', 'coat', 'molecule', 'material', 'statement', 'site', 'literature', 'rescue', 'negro', 'warbler', 'cartoon', 'skiing', 'pooh', 'duke', 'investigator', 'hays', 'dick', 'welterweight', 'thanh', 'laureate', 'order', 'kin', 'grise', 'borbon', 'handy', 'sugar', 's', 'city', 'holly', 'cher', 'rule', 'rules', 'musics', 'commissioners', 'secretaries', 'languages', 'tests', 'lights', 'residence', 'schools', 'fair', 'bank', 'tea', 'guy', 'dad', 'bomber', 'foyer', 'nageurs', 'telephone', 'brand', 'crimes', 'compound', 'region', 'composition', 'job', 'name', 'penn', 'bone', 'cars', 'jokes', 'place', 'song', 'report', 'artaxerxes', 'negroid', 'busybody', 'none', 'washer', 'cleaner', 'conservative', 'adviser', 'alene', 'enemy', 'x', 'ballerina', 'lord', 'abed', 'slaver', 'timer', 'stapler', 'buyer', 'slave', 'schoolman', 'saki', 'tripa', 'fellow', 'follower', 'may', 'pachanoi', 'treasurer', 'vespasian', 'thumper', 'nude', 'jaybird', 'betrothed', 'lifer', 'nester', 'saxophonist', 'freak', 'dvořák', 'fool', 'spouse', 'socratic', 'fröbel', 'illiterate', 'germans', 'giotto', 'hawaiian', 'dropper', 'kite', 'hyde', 'disorderly', 'savant', 'carlos', 'poster', 'featherweight', 'impressionist', 'grandmaster', 'ringer', 'watcher', 'hog', 'neuroscientist', 'rater', 'kluxer', 'cannon', 'puller', 'gyffes', 'nessie', 'sulla', 'gardener', 'kopernik', 'tatar', 'goer', 'aristocrat', 'offspring', 'crawler', 'esprit', 'thinker', 'pet', 'absconder', 'wearer', 'auditor', 'highlander', 'logician', 'campaigner', 'beneficiary', 'discoverer', 'faddist', 'separatist', 'supremacist', 'tout', 'reveler', 'bomb', 'kitten', 'goldbrick', 'licker', 'polisher', 'noser', 'daddy', 'wall', 'cane', 'pontiff', 'realtor', 'matron', 'hygienist', 'breed', 'caste', 'ax', 'editor', 'stendhal', 'arabian', 'marinese', 'hero', 'burner', 'virginian', 'founder', 'face', 'squeak', 'gautama', 'commander', 'actor', 'pincher', 'saxons', 'blower', 'guide', 'turks', 'prize', 'preacher', 'ag', 'dumb', 'dear',  'ocean', 'racer', 'lockspitzel', 'cäsar', 'persóna', 'testamentsexekutor', 'prosecutor', 'figurões', 'archangel', 'd.j', 'diaghilevian', 'farrells', 'greenbergian', 'reaper', 'grim', 'anger', 'guarnieris', 'harrimans', 'apostate',  'hurler', 'estate', 'menningerian', 'श्वश्रू', 'alaskan', 'australian', 'heaghra', 'hack', 'schlesingers', 'proud', 'probeta', 'vargasllosista', 'vargasllosismo', 'veblenian', 'heeler', 'encephalopathy', 'brain',  'abuelas', 'entreprise', 'never', 'firstborn', 'ill', 'rap',  'confort','mac','formula', 'intelectual', 'senegalesa', 'warfare', 'beads', 'babes', 'blairian', 'bloorian', 'labour', 'bondlike', 'boodlerism', 'boomerish', 'bradleyan', 'brainly', 'brentanian', 'manor', 'burrograss', 'caputoan', 'cattellian', 'cavalery', 'cavellian', 'least', 'chumpish', 'tibetan', 'cohenian', 'anarchists', 'correalism', 'coutts', 'crusoesque', 'asylum',  'cybersavvy', 'equation', 'dragonfruit', 'dragonfruits', 'dufferish', 'dummettian', 'pita', 'prithvi', 'dysexecutive', 'eldship', 'etonians', 'evonomics', 'eyrean', 'farmgate', 'f.p.s', 'dam', 'sire', 'frankensteinish', 'map', 'gemman', 'gilliganian', 'grahamesque', 'humour', 'parrot', 'greenfinch', 'crew', 'oxide', 'shake', 'flag', 'springs', 'greensick', 'guardage', 'gulliverian', 'haietlik', 'handbaggy', 'hartmanian', 'hartmannian', 'heiderian', 'hiawathan', 'sound', 'hookerian', 'negotiator', 'negotiators', 'virus', 'right', 'housecleaner', 'housetrain', 'boating', 'boat', 'illegit', 'names', 'infobesity', 'infoglut', 'informatical', 'engine', 'infotrash', 'infortuned', 'intellected', 'disco', 'jekyllesque', 'jekyllian', 'jesuslike', 'kautskyan', 'counts', 'count', 'hit', 'hits', 'hitting', 'thornbill', 'em', 'township', 'prussia', 'eryngii', 'moon', 'tide', 'tides', 'six', 'trade', 'kohlbergian', 'kojac', 'kristevan', 'lakao', 'lewinian', 'lightermen', 'vendor', 'dyehouse', 'dyehouses', 'lowndes', 'hayneville', 'lowville', 'macdonaldian', 'maidish', 'mamaroneck', 'mammygate', 'mauston', 'mcluhanian', 'merrill', 'metrodorian', 'meyerian', 'micropub', 'micropubs', 'midtier', 'militiamen', 'mirrory', 'philopedia', 'bands', 'monona', 'madisonville', 'stroudsburg', 'woodsfield', 'montfortian', 'christiansburg', 'conroe', 'crawfordsville', 'ida', 'sterling', 'norristown', 'pity', 'falls', 'morrisseyesque', 'muscicapine', 'nagelian', 'nahunta', 'together', 'niebuhrian', 'maréchalian', 'nussbaumian', 'swing', 'danish', 'caucasian', 'scandinavian', 'string', 'orbisonian', 'farm', 'farms', 'palmerstonian', 'pambasileia', 'ferryhouse', 'perihelial', 'perihelic', 'perkinsian', 'petersonian', 'phillipsburg', 'pikeville', 'australis', 'polypragmon', 'portalian', 'pastorpreneur', 'postmove', 'predentistry', 'prepharmacy', 'dicks', 'argument', 'problem', 'thesis', 'czech', 'shrink', 'pubwards', 'rahnerian', 'rambam', 'endangerment', 'magentaish', 'rat', 'reinholdian', 'nonstockholder', 'riceroot', 'hon', 'ruesome', 'reindeer', 'saintless', 'tithe', 'santalike', 'schleicherian', 'schubertiade', 'sconnie', 'screven', 'awakening', 'palatalization', 'sundays', 'picture', 'sermonium', 'adventism', 'it', 'sherlockish', 'eyes', 'possum', 'silvercloth', 'script', 'si', 'snapefic', 'snapefics', 'spokespeople', 'wing', 'spuria', 'squatchy', "james's", 'stearns', 'fou', 'supravision', 'identifier', 'sylvania', 'tailtiu', 'taisch', 'silk', 'tarrant', 'technorganic', 'thatcheresque', 'therapese', 'seamounts', 'toadyish', 'odham', 'schumacher', 'joneses', 'sheet', 'spin', 'traditionise', 'traditionism', 'trochiform', 'quark', 'ttbar', 'tullian', 'accountants', 'pips', 'tzutujil', 'unacquaint', 'samboism', 'year', 'under', 'understair', 'unsensible', 'goghs', 'veepstakes', 'auction', 'riche', 'vizenorian', 'walpolean', 'walworth', 'royal', 'queensbury', 'warrenton', 'honesdale', 'jesup', 'waynesboro', 'waynesville', 'corridor', 'generation', 'wooster', 'end', 'wowserism', 'yarnspinner', 'yarnspinners', 'narrator', 'epiparasitic', 'yorkville', 'yoshke', 'creationism', 'congenita', 'aiel', 'avetrol', 'gentyl', 'kingpleie', 'leofman', 'leofmon', 'markisesse', 'philosophre', 'dobles','list', 'lists', 'abuelos', 'dressing', 'vest','tape','prawns', 'prawn', 'block', 'surgery', 'theologian',  'oven', 'sotatekninen', 'revival', 'library',  'setting', 'hospital',  'salary',  'assemblyperson', 'système', 'interviewa', 'interviewer', 'interviewez', 'stoppeuses', 'dos', 'bgén', 'papas', 'mamans', 'chabiha', 'emphysémateux', 'surface', 'bombardiers', 'cueilleurs', 'privés', 'artificielles', 'orne', 'rogatory', 'rogatoires', 'crépin', 'crépins', 'tibetaine', 'alata', 'finalistes', 'feuillus','statesmen', 'immergleich', 'anmalen', 'equitys', 'equity', 'squaws', 'staatsfrau', 'ownership', 'ownerships', 'voters', 'vote', 'votes', 'rapes', 'rape', 'ahuramazda', 'morocco', 'assemblypeople', 'assemblypersons', 'bent', 'degrees', 'shakespeareans', 'shakespearean', 'file', 'gemsbok', 'gemsboks', 'gumwoods', 'gumwood', 'halibuts', 'halibut', 'hartebeest', 'hartebeests', 'ironwood', 'ironwoods', 'mahoganies', 'mahogany', 'manchineels', 'manchineel', 'sandalwood', 'sandalwoods', 'titles', 'title', 'trumpeter', 'trumpeters', 'thorn', 'thorns', 'swords', 'bastardswords', 'sword', 'bastardsword', 'mum', 'neighbour', "belov'd", 'arnolds', 'overlord', 'overlords', 'cheeses', 'lid', 'sheets', 'side', 'sides', 'sparks', 'brightfield', 'field', 'privates', 'private', 'burkean', 'burkian', 'caligrapher', 'calligraphers', 'accessories', 'cannonfodder', 'chus', 'chu', 'childbrides', 'childbride', 'extraordinaire', 'collegegoers', 'collegegoer', 'blimps', 'esse', 'concessionaires', 'bachelors', 'consumptives', 'triplet', 'triplets', 'customshouses', 'customshouse', 'washingtons', 'presidents', 'bulbs', 'screws', 'screw', 'geese', 'palm', 'palms', 'bunnies', 'bunny', 'extraordinaires', 'fatherlashers', 'fatherlasher', 'management', 'episode', 'firstlight', 'equals', 'lasts', 'officials', 'rate', 'monsters', 'freind', 'scientists', 'door', 'leisure', 'trotters', 'atheism', 'nieuw', 'grandnieces', 'monkeys', 'greatgranddaughter', 'grandsons', 'greatgrandson', 'ears', 'travelers', 'moms', 'mom', 'coil', 'coils', 'resonator', 'resonators', 'heman', 'hitlists', 'hitlist', 'fatals', 'fatal', 'hormazd', 'work', 'hourmazd', 'houseguest', 'houseguests', 'houselike', 'housecalls', 'housecall', 'housecommune', 'housecommunes', 'housedoor', 'housedoors', 'housefloor', 'housefloors', 'househunting', 'housemakers', 'housemaker', 'houseproud', 'housesitters', 'housesitter', 'houseslaves', 'houseslave', 'housewalls', 'housewall', 'hycsos', 'hyksos', 'hypergreen', 'illegitimates', 'inhouse', 'cases', 'henrys', 'хозяйничать', 'browns', 'cabs', 'cab', 'cards', 'cheetah', 'harps', 'harp', 'mushrooms', 'skins', 'skin', 'clubs', 'mountains', 'kingsize', 'kingsized', 'legionaire', 'hats', 'lithouse', 'lockmasters', 'longino', 'breakfasts', 'breakfast', 'shirts', 'shirt', 'cups', 'cup', 'manorhouse', 'manorhouses', 'mariners', 'darling', 'darlings', 'mantles', 'ships', 'suns', 'pie', 'pies', 'shelter', 'shelters', 'tables', 'basket', 'baskets', 'soles', 'sole', 'pictures', 'multigreen', 'informations', 'tones', 'tone', 'newfashioned', 'newfounded', 'newmown', 'correspondents', 'watchmen', 'nixons', 'acquaintances', 'nonstockholders', 'nosefirst', 'noticers', 'pauvres', 'offical', 'ohrmazd', 'mice', 'oldtown', 'oldtowns', 'olds', 'ontop', 'dinosaurs', 'dinosaur', 'ostrichlike', 'outfriend', 'pastorpreneurs', 'toms', 'get', 'perseverings', 'colour', 'pregnancies', 'pregnancy', 'pillowtop', 'plasterers', 'posthouse', 'posthouses', 'creditors', 'stockholders', 'presidiums', 'alcohols', 'alcohol', 'cilia', 'cilium', 'energies', 'energy', 'immunodeficiencies', 'immunodeficiency', 'markets', 'myelofibroses', 'myelofibrosis', 'producer', 'producers', 'interface', 'interfaces', 'reinforcement', 'reinforcements', 'residences', 'primaariaineisto', 'sources', 'structure', 'valences', 'valence', 'bishops', 'jackets', 'islanders', 'spur', 'spurs', 'regents', 'drop', 'drops', 'tear', 'tears', 'valiant', 'valiants', 'exchanges', 'candidates', 'costs', 'cost', 'docents', 'docent', 'investigators', 'bankers', 'keys', 'arguments', 'problems', 'theses', 'lives', 'bills', 'sectors', 'siding', 'sidings', 'stocks', 'stock', 'characters', 'wiki', 'wikis', 'cells', 'butterfly', 'democrats', 'democrat', 'redhaired', 'rudokožec', 'redbaiter', 'redbaiters', 'redbay', 'redbays', 'redflower', 'redhanded', 'redhandedly', 'redhandedness', 'redlink', 'redlinks', 'redrimmed', 'redshort', 'redtapism', 'addresses', 'adjectives', 'adjective', 'datings', 'tense', 'tenses', 'fairs', 'mission', 'missions', 'basins', 'basin', 'beds', 'bed', 'birches', 'birch', 'crab', 'crabs', 'runners', 'turtles', 'riverwater', 'way', 'ways', 'tithes', 'errors', 'knot', 'knots', 'pin', 'pins', 'secondhandedly', 'secondhanded', 'secondhandedness', 'embarrassments', 'embarrassment', 'smoke', 'seminew', 'captains', 'livery', 'shakspearean', 'shakspearian', 'superintendents', 'superintendent', 'skinflints', 'mountaintop', 'rooftop', 'stew', 'jackal', 'morning', 'goats', 'privilege', 'whores', 'caucasians', 'coquettes', 'coquette', 'squawberry', 'berries', 'squawberries', 'carpets', 'carpet', 'dresses', 'roots', 'root', 'teas', 'winter', 'winters', 'ends', 'parts', 'stowaways', 'traits', 'trait', 'sunlike', 'sundrenched', 'sunline', 'sunlines', 'leaders', 'deal', 'deals', 'neckline', 'necklines', 'brokers', 'gender', 'virgins', 'singularity', 'singularities', 'gets', 'theoreticians', 'toastmasters', 'comfort', 'show', 'toplevel', 'toplevels', 'topnotch', 'operators', 'identifiers', 'entrepreneurs', 'entrepreneur', 'auctions', 'mittys', 'mistresses', 'wedlocks', 'dads', 'weekold', 'europeans', 'corridors', 'around', 'momma', 'brothur',  'cnavechild', 'colege', 'colegg', 'concubyne', 'web',  'reunion', 'patronas', 'feudales', 'sociales', 'angels', 'autointerviewer',  'maternel', 'ducs', 'médecine', 'air', 'hôtel', 'chefs', 'femmes',  'topmodels',  'vertes', 'effant', 'tube', 'atpatruus', 'atpatrue', 'atpatrui', 'atpatruis', 'atpatruo', 'atpatruum', 'catule', 'catuli', 'catulis', 'catulli', 'catullis', 'catullum', 'catulum', 'maimonidae', 'maimoniden', 'mosen', 'mosis', 'plotus', 'spuri', 'spuriam', 'spuriis', 'tacitae', 'tacitam', 'tacitis', 'tacitum', 'titum', 'vitrice', 'vitrici', 'vitricis', 'vitrico', 'vitricum', 'kleindochtertjes', 'seriemoordenaars', 'mitaines', 'affaithes', 'enfan', 'bożym', 'lali', 'lalą', 'lalę', 'nakrył', 'ogonem', 'tasmański', 'chryste', 'turkawce', 'turkawek', 'turkawko', 'turkawkę', 'neves', 'diabo', 'leite', 'correspondência', 'papões', 'expiatórios', 'recompensas', 'estado', 'família', 'classe', 'quarto', 'correspondentes', 'honra', 'premiadas', 'mentais', 'domésticos', 'familiar', 'defence', 'run', 'skiiing', 'tactician', 'moqrin', 'avenue', 'ridgeway', 'position', 'athlete', 'mcbeal', 'cleanser', 'lyrics', 'mars', 'argyre', 'there', 'astronomers', 'fcc', 'receipt', 'mechanism', 'dumbunny', 'hpanduro', 'capable', 'systems', 'babysittings', 'girls', 'codes', 'measure', 'barcodes', 'tall', 'artifact', 'palmers', 'wage', 'signal', 'ocd', 'factors', 'grotesque', 'synonyms', 'books', 'africa', 'clues', 'heelers', '30', '318', '320i', '325', '524', '525', '528', '530', '533', '535', '630', '633', '635', '728', '732', '733', '735', '750', '850', 'm5', 'cleaned', 'junkie', 'holbach', 'bottle', 'bearings', 'bad', 'spot', 'busdriver', 'butterfinger', 'c3h4n2', 'ca2', 'ca3', 'ca4', 'camry', 'porcupine', 'anesthesiologist', 'cataloguing', 'goals', 'havasi', 'pennsylvania', 'catwalks', 'executives', 'exchequer', 'superba', 'astro', 'caprice', 'cavalier', 'chevelle', 'chevette', 'cheyenne', 'citation', 'corvair', 'camino', 'greenbrier', 'apv', 'malibu', 'pickup', 'tahoe', 'volt', 'blazer', 'rank', 'toxin', 'popular', 'cirrus', 'cordoba', 'eagle', 'lighter', 'concert', 'guitar', 'guitars', 'dancing', 'us', 'clopilet', 'clortermine', 'when', 'sets', 'events', 'drive', 'resources', 'schedule', 'deed', 'ammonis', 'fraud', 'stitching', 'interglacial', 'moi', 'matthews', 'db6', 'pantera', 'graveyard', 'dentists', 'apparel', 'network', 'costumes', 'items', 'median', 'cheney', 'puzzle', 'division', 'subtraction', 'twirler', 'smallpox', 'duckman', 'south', 'bandleader', 'kitt', 'meat', 'wands', 'elantra', 'midicare', 'fudd', 'eds', 'race', 'necessity', 'balanced', 'wiring', 'eyore', 'excellence', '6', 'fairlane', '20', 'piaggio', 'ferns', 'figureskating', 'mim', 'seat', 'accommodation', 'impression', 'moment', 'shepherd', 'apparatus', 'peg', 'expenses', 'z', 'ad', 'hooks', 'contour', 'cortina', 'fairmont', 'futura', 'maverick', 'mustang', 'tempo', 'apology', 'graduate', 'wilma', 'flintstones', 'embassy', 'tuck', 'buddy', 'planet', 'tradesperson', 'bedroom', 'doorway', 'garifuna', 'kasparow', 'chessplayer', 'behavior', 'collection', 'locorum', 'metro', 'prizm', 'windgpresident', 'occasion', 'grammies', 'cherokee', 'cellist', 'looking', 'sephanoides', 'carcinus', 'fir', 'lawns', 'jealousy', 'gt6', 'bissauan', 'hackers', 'karzai', 'marvin', 'after', 'holt', 'emotion', 'hc2o4', 'label', 'wave', 'brewing', 'sense', 'balloning', 'quean', 'jintao', 'parenting', 'rbc', 'accent', 'scoupe', 'sonata', 'tiburon', 'do', 'coring', 'light', 'format', 'wait', 'vegetarians', 'specification', 'row', 'transfer', 'topic', 'cavity', 'tension', 'location', 'designers', 'question', 'kurdish', 'insurgent', 'jails', 'uprising', 'janusz', 'argonauts', 'cj7', 'comanche', 'wrangler', 'zealand', 'shipley', 'saviour', 'pearse', 'austria', 'botanist', 'emperior', 'carnell', 'fault', 'clam', 'police', 'cruiser', 'lessig', 'notebook', 'ho', 'story', 'hollow', 'claypool', 'messi', 'hacking', 'reader', 'weather', 'roll', 'earth', 'coordinates', 'march', 'emotions', 'fear', 'decision', 'scheme', 'tag', 'linebacker', 'telecommuters', 'jewelry', 'lace', 'teenager', 'dresser', 'species', 'formation', 'window', 'churches', 'marauders', 'addition', 'diagnosing', 'crude', 'experience', 'step', 'melodies', 'operation', '180', '190', '200', '218', '219', '230', '250', '280', '300', '300sd', '350', '380', '400', '450', '500', '600', 'wagon', 'mestico', 'conversion', 'pediphile', 'micky', 'dance', 'abbreviation', 'mindpixel', 'mingw', 'launch', '4150', '7961', 'downline', 'mondeo', 'morrocan', 'note', 'blackboards', 'scene', 'rating', 'mp3s', 'mr2', 'holding', 'cleric', 'pleasure', 'holiday', 'kaitlyn', 'intelligence', 'crunch', 'nethanethiol', 'purpose', 'lactamase', 'highway', 'theaters', 'restaurant', 'york', 'press', 'liberal', 'recyclable', 'bottles', 'clipping', 'cheerleader', 'headquarters', 'building', 'virginia', 'nigerois', 'dealership', 'depositor', 'context', 'notepads', 'nouns', 'novicodin', 'season', 'safety', 'pashtoo', 'no', 'gundaroo', 'indeed', 'iranian', 'timepiece', 'font', 'remembered', 'expereinced', 'display', 'again', 'commiseration', 'quartet', 'valley', 'friendly', 'dinner', 'administration', 'spears', 'gardening', 'mossberg', 'firearms', 'birds', 'emu', 'feather', 'feathers', 'untouchables', 'glyndwr', 'osbourne', 'schema', 'ss', 'guinean', 'paracelsus', 'paracodin', 'families', 'reach', 'fuselage', 'compartment', 'gondola', 'amvs', 'paws', 'pinful', 'cardiologist', 'enlargement', 'warmer', 'long', 'comfortable', 'pricks', 'issue', 'patterns', 'will', 'to', 'scarry', 'come', 'rest', 'intolerant', 'alcoholics', 'hurry', 'pharmacologists', 'ills', 'indicating', 'cries', 'coa', 'philippino', 'phillipino', 'philospher', 'checkup', 'filled', 'perfect', 'kill', 'rangers', 'deviously', 'four', 'barrack', 'encapsulation', 'family', 'protocol', 'pralidoxime', 'meal', 'terms', 'controller', 'server', 'singular', 'industries', 'wafer', 'substance', 'handguns', 'teeth', 'toothpaste', 'coats', 'alberts', 'guards', 'club', 'suspects', 'insurer', 'aircraft', 'airplane', 'container', 'corporation', 'truck', 'entertainer', 'physicist', 'judges', 'transcript', 'sites', 'stressor', 'derivative', 'hospitals', 'prison', 'crown', 'solvers', 'rapacodin', 'm', 'fowl', 'passion', 'delight', 'teams', 'squirrel', 'hair', 'leafs', 'bean', 'confinement', 'remedeine', 'attempt', 'revelations', 'cowgirl', 'rhianna', 'munian', 'traditionalists', 'infestation', 'blindness', 'otters', 'cover', 'yesterday', 'rochelle', 'twist', 'stamping', 'mates', 'saddles', 'comfortably', 'switch', 'subject', 'americans', 'golfer', 'christians', 'herriott', 'creator', 'b', 'movies', 'alaska', 'gametocyte', 'spermatocyte', 'starter', 'tanners', 'pack', 'bloke', 'fireworks', 'where', 'isolation', 'kit', 'mosque', 'shimming', 'sibs', 'syracuse', 'yorktown', 'carol', 'silkbamboo', 'complicated', 'sisterfucker', 'temples', 'if', 'sleeve', 'privacy', 'picrate', 'demonstration', 'programs', 'latino', 'army', 'airspace', 'gonzales', 'stadium', 'sportwear', 'international', 'california', 'sport', 'stacey', 'mitosis', 'comptroller', 'bicycle', 'ejector', 'stepfathers', 'stepparents', 'stepmothers', 'stepsisters', 'plaster', 'record', 'tennessee', 'organizer', 'stovetop', 'knowledge', 'power', 'structuralhypothesistypebystructuretype', 'emission', 'harper', 'fighter', 'g', 'nintendo', 'minx', 'casing', 'forsa', 't100', 'act', 'bath', 'halonen', 'reading', 'tausug', 'cleaning', 'adapter', 'connector', 'listing', 'code', 'pizza', 'creatures', 'mangusta', 'not', 'combination', 'floors', '2.2', 'motion', 'townhouses', 'tradeswoman', 'turret', 'coaches', 'lesbian', 'transporation', 'helicopter', 'materials', 'website', 'tredia', '1500', '2000', 'tr250', 'destinations', 'aikman', 'democracy', 'consequences', 'truths', '138', '238tnb', '238tnbf', '238tnf', 'ufs', 'society', 'restrooms', 'election', 'bus', 'usb', 'florida', 'ussecretaryofcommerce', 'butter', 'valgrind', 'plas', 'stent', 'vegans', 'cart', 'facility', 'humans', 'veterinarians', 'archive', 'vindicator', 'viscidity', 'quantity', 'ifa', 'religion', 'roof', 'roses', 'wasserman', 'thrush', 'gretzy', 'business', 'bells', 'warrior', 'samoan', 'dealers', 'brush', 'lessons', 'stitch', 'rebellion', 'juice', 'piece', 'sepulchers', 'sepulchre', 'babylon', 'winches', 'blade', 'textured', 'trace', 'hosiery', 'swimwear', 'windbreaker', 'chair', 'theory', 'token', 'adult', 'workaholic', 'graffiti', 'pen', 'score', 'wulver', 'xs', 'avoid', 'ev', 'afafine', 'québec', 'hispanoaméricain', 'gens', 'p.d.g', 'orthant', 'résédacées', 'impure', 'rams', 'csapat', 'buccaneers', 'titans', 'redskins', 'cardinals', 'falcons', 'colts', 'raiders', 'afabarn', 'ash', 'informática',  'surrogate', 'russian', 'military', 'commissaire', 'poet', 'persona', 'organ', 'left', 'anti', 'nouveau', 'in', 'average', 'eager','u.s', 'green',  'new', 'top', 'house', 'lightweight', 'sea', 'drug', 'hat', 'shade','sol', 'package', 'package', 'spots', 'spiders', 'rooms', 'steads', 'product', 'outages', 'gear', 'beetle', 'bitterns', 'command', 'judge', 'objector', 'suspect', 'mayor', 'goddess', 'politique', 'center', 'intellectual', 'silent', 'rights', 'fruit', 'life', 'grandmothers', 'ministers', 'office', 'activity', 'task', 'user', 'authority', 'arab', 'provocateur', 'machina', 'all','out','ego','infant','arms','guest', 'breaker', 'town', 'smoker', 'policeman', 'politics', 'time', 'animal', 'group', 'turtle', 'first', 'level', 'hunter', 'nephew',  'britain', 'first', 'level',   'consultant',   'beater', 'surgeon', 'assistant',   'egg',  'goose',  'fodder',  'liberator', 'granddaughter',  'jewel',  'hitter',  'statesman',  'boys',  'palace',  'cipher',  'removed',  'politiques',  'card',  'brothers',  'students',  'organization',  'software',  'orange',  'means',  'niece',  'principle',  'duck',  'rider', 'mate' 'dealer',  'sprite',  'captain',  'wise',  'traveler',  'link',  'cause', 'link',  'cause',  'street', 'down',  'up',  'world',  'plant',  'country',  'elimination',  'call',  'unconscious',  'area',  'school',  'room',  'unit',  'commission',  'siblings',  'vehicle',  'doves',  'sun',  'equipment',  'fact', 'aborigine', 'attache', 'pair', 'beauty', 'scout', 'errant', 'worth', 'almighty', 'guard', 'announcer', 'boss', 'baker', 'pilot', 'cop', 'gatherer', 'piston', 'board', 'unionist', 'reporter', 'bachelor', 'cyclops', 'guest', 'infant', 'breaker', 'arms', 'demon', 'coward', 'boomer', 'household', 'gun', 'korean', 'eater', 'department', 'figure', 'bag', 'conjunction', 'numbers', 'paper','ambassador', 'system', 'gods', 'sibling', 'politician', 'duckling', 'correspondent','syndrome', 'wife', 'male', 'waiting','driver', 'governor', 'german', 'lieutenant', 'writer', 'painter', 'critic', 'citizen', 'program', 'clause', 'process', 'parent', 'fathers', 'reds', 'day', 'representative', 'advocate', 'cheese','grandson', 'collector', 'professional', 'musician', 'shot', 'indian', 'men','miner', 'canadian', 'number', 'sisters', 'gentleman', 'grandmother', 'tops', 'husband', 'scientist',  'accountant', 'official', 'clerk', 'heavyweight', 'party', 'bird', 'partner', 'working', 'kings', 'hand', 'princess', 'detective', 'witness', 'twin','agreement', 'property', 'accessory', 'companion','accountant', 'official', 'clerk', 'dog', 'bachelor, ''car', 'parents', 'seconds', 'mothers', 'children', 'parliament', 'coach', 'affaires', 'grandfather', 'solider', 'woman', 'line', 'county', 'effect', 'dove', 'law', 'person', 'house', 'people', 'man', 'houses','player','brother', 'student', 'officer', 'green', 'agent', 'child', 'president', 'minister', 'mother', 'music', 'information', 'uncle', 'leader', 'war', 'uncle', 'secretary', 'engineer', 'second','jockey', 'friend', 'chief', 'being', 'boy', 'live', 'professor', 'american', 'dragon', 'college', 'girl', 'catholic', 'states','grandchild','sister', 'father','prisoner','worker', 'top', 'committee', 'administrator', 'thing', 'god', 'class','automobile','spirit', 'admiral', 'ceremonies', 'russia', 'songwriter', 'executive', 'commissioner', 'technician', 'doctrine', 'therapist', 'operator', 'old', 'greens', 'deity', 'aunt', 'language', 'nest', 'red', 'cat', 'consort', 'member', 'lady', 'killer', 'baby', 'manager', 'disease', 'specialist', 'friends', 'son', 'practitioner', 'event', 'designer', 'you', 'trinity', 'doctor', 'attaché', 'servant', 'designer', 'nurse'])
    block_list2 = set(['walk', 'commercial','deems', 'puerto', 'french',  'r',  'recruiting', 'plautus', 'lighthorse', 'kay',  'confederate',  'angus', 'anicius', 'barthold', "frankenstein's", 'honoré', 'front', 'keystone', 'cold',  'talking', 'knight', 'broker',  'financial', 'cy',  'gypsy', 'ostrich', 'knee', 'subordinate', 'lead', 'key', 'real', 'his', 'surname', 'instrumental', 'criminal', 'dual', 'philosopher', 'write', 'milne', 'tap', 'security', 'constant', 'costa', 'research', 'opera', 'riding', 'unknown', "printer's", 'kitchen', 'belt',  "photographer's", 'gandy', 'solicitor',  'bay',  'zoo', 'granite', 'garden', 'major', 'rhodes', 'senior', 'quartermaster', 'desert', 'iron', 'barrel', 'dark', 'running', 'granville', 'big', 'faisal',  'soldier',  'render', 'vice', 'melchior',  'illegal',  'crafty',  'last', 'death', 'your', 'dodge',  'yellow', 'tribes', 'general', 'biological', 'topological', 'barracks', 'virgin', 'bright', 'chinese', 'cognitive', 'developmental', 'differential', 'economic', 'euro', 'false', 'fore', 'ground', 'jungle', "li'l", 'natural', 'river', 'some', 'squaw', 'titan', 'undocumented', 'role', 'males', 'at', 'chevrolet', 'making', 'existentialist', 'for', 'state', 'vanuatuan', 'langue',  'piano', 'opium', 'acid', 'lieder', 'coureur', 'saloon', 'edsel', 'desk', 'past', 'hablot', 'grey', 'hooray', 'ivy', 'caffeine', 'cocaine', "king's", 'pledge', 'morris', 'latter', 'hired', 'federal', "artist's", 'odds', 'addle', 'profit', 'lighthouse', 'able', 'mixed',  'rhode', 'credit', 'staff',  'confessor',  'mae', 'amen', 'breughel', 'fetid',  'sparring', 'imaginary', 'seneca', 'repas', 'creditor', 'devoted', 'diocletian','sailing', 'lasso', 'wet',  'musical', 'chandler',  'comics', 'popper', 'firth', 'hertz',  'biographic',  'armchair', 'ögey', 'quantitativer', 'agony', 'april', 'bond', 'cylinder', 'analytic', 'future', 'hiv',  'deer', 'revlon', 'caesar', 'wild', 'youthful', 'aires', 'architecture', 'architectures', 'colleville', 'entre', 'faire', 'grosses', 'les', 'orteil', 'orteils', 'petites', 'fiscal', 'monetary', 'primary',  'imperial', 'нағашы', 'қайын', 'къарт', 'уллу', 'fait',  'mestresses', 'alta', 'assigned', 'bloody', 'comparative', "could've", 'could', 'cunning', 'dead', 'domestic', 'feistel', 'indentured', 'independent', 'internally',  'liquor', 'low', 'maid', 'massage', 'missionary', 'passes', 'passed', 'passing', 'pass', 'pinky', 'playfair', 'potentially', 'preferred', 'princes', 'purple', 'puss', 're', 'reasonable', 'rice', 'she', 'snail', 'sneaker', 'submarine', 'substitution', 'shooting', 'suppressive', 'taxi', 'temporary', 'those', 'trailing', 'vigenère', 'wardrobe', 'flamenco', 'flamencos', 'chevaliers', 'her', 'lupi', 'padroni', 'tasmanian', 'donas', 'gêngis', 'abe', 'bistrița', 'full', 'bqdp', 'bqnd', 'bronco', 'buck', 'bypass', 'mid', 'character', 'chase', 'closing', 'compact', 'luxury', 'premium', 'conditioning', 'important', 'cosmetics', 'cuban', 'determining', 'duplex', 'dutch', 'econoline', 'electoral', 'elton', 'female', 'flight', 'females', 'chin', 'named', 'suwannee', 'ford', 'free', 'grammy', 'guyanese', 'hainan', 'hip', 'immigrants', 'inanimate', 'indus', 'israeli', 'iteration', 'jazz', 'jews', 'jordanian', 'kazakh', 'lebanese', 'ficticious', 'lisa', 'literary', 'luxemburgian',  'more', 'marathon',  'mariachi', 'measuring', 'milk', 'passenger', 'motorcycle', 'road', 'move', 'movie', 'sports', 'entertainment', 'narrative', 'neck', 'political', 'norfolk', 'north', 'orderly', 'parabolic', 'path', 'knows', 'one', 'playstation', 'polish', 'prescription', 'presidential', 'prester', 'pristine', 'radio', 'ally', 'editing', 'trying', 'retail', 'ros', 'saudi', 'seventh', 'shannon', 'sierra', 'sinhalese', 'sneak', 'soeren', 'speed', 'stones', 'sturdy', 'non', 'mainstream', 'surinamese', 'japanese', 'tajik', 'teen', 'bidirectional', 'thai', 'tour', 'trajectory', 'treasure', 'trivial', 'cord', 'elected', 'uruguayan', 'venezuelan', 'video', 'vegetation', 'voice', 'waist', 'while', 'bébé', 'second', 'first','mazda', 'mitsubishi',])
    lst = [(word, (1.0+ sum([a.count('_') for a in self.word2en.get(word, [])]))/(1.0 + len(self.word2en.get(word, []))) - 1.0) for word in ner2word['PUBLIC_FIGURE']]
    lst = [l for l in lst if (l[1] >= 0.0) and (l[0].count("_") < 5) and  not [a for a in self.word2en.get(l[0], {l[0]}) if a.split("_")[-1] in block_list] and  not [a for a in self.word2en.get(l[0], {l[0]}) if a.split("_")[0] in set(list(block_list)+list(block_list2))]]
    lst = lst + [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('person.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms()])  if a[0] == a[0].upper()]
    public_figure_en = list(set(OntologyBuilder.public_figure_list + list(itertools.chain(*[self.word2en.get(l[0], {l[0]}) for l in lst]))))
    lst = [(word, (1.0+ sum([a.count('_') for a in self.word2en.get(word, [])]))/(1.0 + len(self.word2en.get(word, []))) - 1.0) for word in ner2word['PERSON']]
    lst = [l for l in lst if (l[1] >= 0.0) and (l[0].count("_") < 5) and  not [a for a in self.word2en.get(l[0], {l[0]}) if a.split("_")[-1] in block_list] and  not [a for a in self.word2en.get(l[0], {l[0]}) if a.split("_")[0] in set(list(block_list)+list(block_list2))]]
    person_en = list(itertools.chain(*[self.word2en.get(l[0], {l[0]}) for l in lst]))
    
    union_en = union_en + [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in  OntologyBuilder.union_list] + [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('union.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms()]) ]
    social_economic_class_en = OntologyBuilder.soc_eco_class_list +  [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('class.n.03').closure(lambda s: s.hyponyms()) if not d.hyponyms()]) ] 
    professional_en = [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('professional.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms()])  if a[0] == a[0].lower()]
    worker_en = [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('worker.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms() if not d.hyponyms()])  if a[0] == a[0].lower()]
    poltical_party_member_en = self.political_party_member_list + [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('politician.n.02').closure(lambda s: s.hyponyms()) if not d.hyponyms()]) if a[0] == a[0].upper() and "_" not in a]
    politcal_party_en = OntologyBuilder.political_party_list + [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('party.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms()]) if a[0] == a[0].upper()]
    organization_en =  org_en + OntologyBuilder.org_list + [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('organization.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms()]) if a[0] == a[0].upper()]
    religion_list_en =  [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in  OntologyBuilder.person2religion.values()] +  [a for a in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('religion.n.01').closure(lambda s: s.hyponyms()) if not d.hyponyms()]) if a[0] == a[0].upper()]
    religion_list_en = religion_list_en + OntologyBuilder.religion_list
    gender_person_en = OntologyBuilder.gender_list + [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in  itertools.chain(*OntologyBuilder.pronoun2gender.values())]

    language_list_en = list(set(language_en + [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in person.Provider.language_names]))
    language_list_en.remove('interlingua')
    race_list_en = list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in OntologyBuilder.race_list]))
    race_list2 = []

    for word in race_list_en + language_list_en:
      for gender in ['person', 'people', 'females', 'women', 'ladies', 'gays', 'males', 'men', 'lesbians', 'boys', 'girls', 'adults', 'female', 'woman', 'lady', 'gay', 'male', 'man', 'lesbian', 'boy', 'girl', ]:
        for word2 in [gender+"_"+word, word+"_"+gender, "older_"+gender+"_"+word, "old_"+word+"_"+gender, "younger_"+gender+"_"+word, "young_"+word+"_"+gender, "adult_"+word+"_"+gender, "senior_"+word+"_"+gender,]: 
          if word2 in self.en:
            race_list2.append(word2)

    religious_person_en = list(set(OntologyBuilder.religious_member_list + [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in [word.split(",")[0].strip()  for word in  OntologyBuilder.person2religion.keys()]]))
    religious_person_en = religious_person_en +  [a for a  in itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('religious_person.n.01').closure(lambda s: s.hyponyms())]) if a[0] == a[0].upper() and a not in ('WASP',)]
    religious_person2 = []
    for word in religious_person_en:
      for gender in ['person', 'people', 'females', 'women', 'ladies', 'gays', 'males', 'men', 'lesbians', 'boys', 'girls', 'adults', 'female', 'woman', 'lady', 'gay', 'male', 'man', 'lesbian', 'boy', 'girl', ]:
        for word2 in [gender+"_"+word, word+"_"+gender, "older_"+gender+"_"+word, "old_"+word+"_"+gender, "younger_"+gender+"_"+word, "young_"+word+"_"+gender, "adult_"+word+"_"+gender, "senior_"+word+"_"+gender,]: 
          if word2 in self.en:
            religious_person2.append(word2)
    religious_person_en = list(set(religious_person_en + religious_person2))

    jobs_en = professional_en + worker_en + OntologyBuilder.jobs + list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in [word.split(",")[0].strip()  for word in  job.Provider.jobs]]))
    jobs_list2 = []
    for word in jobs_en :
      for gender in ['female', 'woman', 'lady', 'gay', 'male', 'man', 'lesbian', 'boy', 'girl', ]:
        for word2 in [gender+"_"+word, word+"_"+gender]: 
          if word2 in self.en:
            jobs_list2.append(word2)    
    jobs_en = list(set(jobs_en + jobs_list2))
    jobs_en = list(set(jobs_en + jobs_list2))

    gender_list2 = []
    for word in gender_person_en :
      for age in ['old', 'older', 'young', 'younger', 'adult', 'senior',]:
        for word2 in [age+"_"+word, word+"_"+gender]: 
          if word2 in self.en:
            gender_list2.append(word2) 
    gender_person_en = list(set(gender_person_en+gender_list2))
    
    race_list_en = list(set(self.race_list + race_list2))
    disease_list_en =  disease_en + self.disease_list + list(itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('physical_condition.n.01').closure(lambda s: s.hyponyms())]))
    disease_list_en = disease_list_en + list(itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('infectious_agent.n.01').closure(lambda s: s.hyponyms())]))
    disease_list_en = disease_list_en + list(itertools.chain(*[[str(a.name()) for a in d.lemmas()] for d in wn.synset('symptom.n.01').closure(lambda s: s.hyponyms())]))
    
    #now create the x-lingual word2ner mapping
    # TODO: need to do LANGUAGE, GPE (country, city) and block lists
    block_list= []
    word2ner = yago2ner
    print ('num of items', len(word2ner))
    gender_person_list, word2ner, block_list = self.create_multilingual_examples(gender_person_en , 'GENDER', word2ner, block_list= block_list + ['maidservant', 'slip', 'kiddo', 'alma', 'heiress', 'pussy', 'pucelle', 'miss', \
                                                                                                     'handmaid', 'doll', 'noblewoman', 'damselfly',  "lady's_maid", 'ponce', 'bachelorette', 'lesser_black_backed_gull',  'leopardess', \
                                                                                                     'housemaid', 'zygoptera', 'damselfish', 'demoiselle', 'zygopteran', 'silver_perch', 'sister', 'maiden_over', 'brevier', 'big_sister', 'spinster', 'brat', \
                                                                                                     'shojo', 'girlfriend', 'wench', 'daughterling', 'petite', 'girlish', 'wife',  'daughter', 'chinese_girl', 'paternal_aunt', 'son', 'spouse', 'husband', 'fellow', 'bloke', 'covey', 'stud', 'cove', 'youngster', 'cub', 'sex', 'madam', 'rocky', 'youth', 'stripling', 'knave', 'child', 'ferrule', 'groom', 'lord', 'master', 'edging', 'person', \
                                                                                                     'casanova', 'missus', 'grandpa', 'geezer', 'boyfriend','adulteress', 'foot', 'leg', 'mutually', 'with_one_another', 'lazy', 'from', 'jack', 'cock', 'crevice', 'messiah', 'slit', 'fissure', 'young_buck','pair', 'young', 'virgin', 'motel', 'bread', 'dam', 'dam', 'isle_of_man','virgo', 'i_translations', 'prostitute', 'broad', 'man_translations', 'retroflex_final', 'son_translations', 'virile', 'waiter', 'ready',  'vagina',  'wow', 'dungeon', 'wow', 'guild', 'servant' ])
    print ('num of items', len(word2ner))
    title_list, word2ner, block_list =  self.create_multilingual_examples(self.title_list , 'TITLE', word2ner, cut_off_per=0.2, block_list= block_list+ ['possessor', 'virgin', 'girl', 'neglect', 'overlook','fail', 'fail_to_catch', 'lose', 'want'])
    person_pronoun_list, word2ner,  block_list = self.create_multilingual_examples(self.person_pronoun_list, 'PERSON_PRONOUN', word2ner, cut_off_per=0.2, block_list= block_list + ['interjection_expressing_anger_or_chagrin', 'i_translations', 'information_technology', 'here']+self.other_pronoun_list)
    other_pronoun_list, word2ner, block_list =  self.create_multilingual_examples( self.other_pronoun_list, 'OTHER_PRONOUN', word2ner,  cut_off_per=0.2, block_list= block_list+ ['interjection_expressing_anger_or_chagrin', 'i_translations', 'information_technology', 'here']+self.person_pronoun_list )
    domain_name_list, word2ner, block_list = self.create_multilingual_examples(domain_name_en , 'DOMAIN_NAME', word2ner, block_list=block_list)
    quantity_list, word2ner, block_list = self.create_multilingual_examples(quantity_en  , 'QUANTITY',  word2ner, block_list=block_list +  ['mimosa', 'attic', 'hay', 'flytrap', 'rudd', 'hop', 'ginger', 'sunray', 'lupine', 'draughts','angelica', 'orange', 'daisy', 'buoy',  'parking_space', 'bin', 'standard', 'long_weekend','rubber', 'won',  'real', 'shield', 'parking',  'short_for',  'snot',  'compositeness', 'mark','decimal', '3d', 'irl', 'three_mountains', 'si', 'tithe', 'solid'])
    bio_chem_list, word2ner, block_list = self.create_multilingual_examples(bio_chem_en , 'BIO_CHEM_ENTITY',  word2ner, block_list=block_list)
    anat_list, word2ner, block_list = self.create_multilingual_examples(anat_en , 'ANAT',  word2ner, block_list=block_list)
    medical_therapy_list, word2ner, block_list = self.create_multilingual_examples(medical_therapy_en , 'MEDICAL_THERAPY',  word2ner, block_list=block_list)
    plant_list, word2ner, block_list = self.create_multilingual_examples(plant_en , 'PLANT',  word2ner, block_list=block_list+['vanilla', 'face', 'surname_lin', 'iris', 'flag',  'arabica', 'lifeline', 'aspen', 'robust', 'blue_bugle', 'thrift', 'maria','broom','cos','althea','see', 'flamboyant', 'black_eye', 'purplish',  'purple', 'wickerworker', 'wicker',  'plane',  'bay',  'bittersweet',  'litmus', 'fire_opal',  'oliva', 'i_live_in_melbourne', 'absinth', 'cunt', 'prune', 'ricin', 'naked_lady',  'absinthe', 'mugwort', 'camomile_tea',  'mandarin_chinese', 'nonce', 'blond', 'over_embellished', 'mauve',  'bray', 'nincompoop', 'crucible', 'quack', 'oca', 'cherry_red', 'absent', ])
    print ('num of items', len(word2ner))
    animal_list, word2ner, block_list =  self.create_multilingual_examples(animal_en , 'ANIMAL',  word2ner, block_list=block_list + [ 'sidewinder', 'tuberculosis_germ', 'bug', 'ridley', 'neanderthal','ridley', 'neanderthal', 'pain', 'dipper', 'vampire', 'flicker', 'hsv', 'air_bladder_of_fish', 'sole', 'sultan', 'leatherback','moby_dick',  'mew', 'cramp', 'cough',  'idiot', 'fool',  'rot', 'alcedo', 'cast_net', 'ass', 'jackass', 'mustela',  'pica',  'grampus',  'ailuro',  'brown',  'defensive_opening_for_shogi', 'möwe', 'chump',  'rook', 'night_person', 'hart', 'buck',  'foolish', 'coward', 'simpleton', 'ignoramus', 'folly', 'stupidity', '900', 'hack', 'dunce', 'blockhead', 'nonsense', 'something_strange_or_suspicious', 'ester', 'who', 'ase', 'moke', 'æsir', 'tetrao', ])
    gpe_list, word2ner, block_list =  self.create_multilingual_examples(gpe_en , 'GPE',  word2ner, block_list=block_list +['christopher',  'sugarloaf', 'elysium', 'peter', 'echinopsis_pachanoi', 'civil_rights', 'valentine'])
    fac_list, word2ner, block_list =  self.create_multilingual_examples(fac_en , 'FAC',  word2ner, block_list=block_list + ['gradually', 'bit_by_bit', 'by_small_degrees', 'junior_school', 'lodging_together',  'center', 'aspirin'])
    location_list, word2ner, block_list =  self.create_multilingual_examples(location_en  , 'LOCATION',  word2ner, block_list=block_list +  ['helena', 'tram', "new_year's", 'year_end', 'hello', 'good_morning', 'how_do_you_do', 'have_nice_day', 'bat', 'here_you_are', 'no_thanks', 'sure_thing', 'no', 'please', 'excuse_me', 'pardon_me', 'you_are_welcome', 'heartily', 'gladly', 'union', 'eve', 'circumcision', 'daylight', 'daybreak', 'sleep_tight', 'nest', 'social_democrats',   'nothing', "it_doesn't_matter", 'big_dipper', 'spa'])
    event_list, word2ner, block_list =  self.create_multilingual_examples(event_en , 'EVENT',  word2ner, block_list=block_list)
    date_list, word2ner, block_list =  self.create_multilingual_examples(date_en , 'DATE', word2ner,  block_list=block_list+ ['latent_period', 'summer', 'mortality', 'deathrate', 'included', 'within_that', 'even_now', 'over_time','allhallowtide', 'western_calendar', 'deadline', 'in_end', 'around_clock', 'cold_snap', 'night_guard', 'holiday', 'workday', 'leisure', 'interim', 'ad', 'jubilee', 'moment',  '4_4', 'instantaneously', 'vacation', 'nonprofessional', 'leisure_time', 'shift', 'month', 'bright_moon', 'refers_to', 'year_of_our_lord', 'phases_of_moon', 'class', 'time_at_leisure', 'off_time', 'work_shift', 'respite', 'infant_mortality_rate'])
    law_list, word2ner, block_list =  self.create_multilingual_examples(law_en , 'LAW',  word2ner, block_list=block_list + ['freedom_of_opinion', 'jurisdiction', 'civil_marriage', 'capital_punishment', 'constitutional_state', 'trustee_beneficiary_relation', 'sexual_exploitation', ])
    food_list, word2ner, block_list =  self.create_multilingual_examples(food_en , 'FOOD',  word2ner, block_list=block_list+ ['tomato', 'chip',  'baseball_fan', 'patty',  'tisane', 'infusion',  'brackish', 'pop',  'tonic', 'rump',  'buttocks', 'formula',  'blanc',  'oil', 'table_talk', 'zwart', 'starter', 'testicle', 'ovum'])
    print ('num of items', len(word2ner))
    language_list, word2ner, block_list =  self.create_multilingual_examples(language_list_en , 'LANGUAGE',  word2ner, block_list=block_list+['ironworker', 'erse',])
    job_list, word2ner, block_list =  self.create_multilingual_examples(jobs_en , 'JOB',  word2ner, block_list= block_list + [ 'guarda_portão',  'door', 'root', 'landed', 'give', 'button', 'occupant', 'haughty', 'commender', 'head', 'command', 'lead', 'leveret',  'shop', 'superuser', 'legal_guardian', 'kanji_legs_radical', 'pedestrian_traffic', 'angel', 'clever', 'cast', 'bower_bird', 'calculator', 'catchword',  'generator', 'perpetrator', 'god_almighty', 'southeast', 'aesthetics', 'animated', 'email', 'wearer', 'girder', 'holder',  'luggage_carrier', 'carrying_bag', 'poster_child', 'bleaching_agent',  'utterer', 'falsifier', 'employee', 'curate', 'qualifier', 'gambler',  'playful', 'prayer',  'escort', 'reeve',  'superman', 'dignitary', 'leadership', 'first_place', 'seat_of_honor', 'jobholder', 'redact', 'pacer',  'marcher', 'drive_around', 'motorist', 'scout', 'conductive', 'dribbler', 'hauler', 'conveyor', 'tutorial', 'burn_down', "nobleman's_residence", 'digger', 'teller', 'accounts_office', 'wright', 'artisanal', 'attorney_at_law', 'jurist', 'judge', 'centerfold',  'compiler',  'curtain', 'portiere', 'savior',  'mentor', 'restrainer', 'wing', 'scepter',  'armed', 'seating', 'uncle',  'adjutant', 'egeria', 'consultative', 'consult', 'government',  'taker',  'harbinger', 'dispatch_rider', 'lion', 'steed', 'conveyance', 'sent', 'all_rounder', 'famous_and_virtuous_ancestors', 'pioneer', 'sawbones', 'supervisory_program', 'upper_part_of_flag', 'tour_guide','discussion',  'creative',  'maker', 'praeses', 'chairmanship', 'landowner', 'kitchen_manager', 'interior_decoration', 'regency',  'spin_doctor', 'event_companion', 'baton_twirler', 'puppeteer', 'augustin_eugene_scribe', 'nib', 'claimant', 'contact', 'helper', 'tosser', 'deputation', 'arousing', 'sir_isaac_pitman',  'pilot_light',  'dragoman','defamer', 'translation', 'regulation', 'ordinance', 'fixer', 'stakeholder',  'pictorial', 'woodsman', 'leshy', 'utility', 'recruit', 'person_with_whom_to_speak', 'someone_to_talk_to', 'comrade', 'friend', 'sputnik',  'forced_laborer', 'anesthesiology',  'hellhound', 'raider',  'nerd',  'national_park_service', 'road_map', 'leading', 'managerial', 'reigning',  'peasant', 'burglar',  "manage_one's_household", 'thrall', 'addict', 'nief',  'worshipper', 'mammy', 'manny', 'stonemasonry', 'moor', 'bobby', 'man_of_letters', 'literacy', 'connector', 'super', 'speller',  'oceanic', 'ocean', 'showman', 'general_business', 'general_affairs', 'manage', 'preside', 'apprentice',  'tray',  'special_duty', 'lens', 'rule', 'directing',  'do_it_yourselfer',  'rider', 'cleansing_agent', 'optional', 'artillery', 'artillerist', 'witch_doctor', 'person_who_gives_treatment', 'person_who_reads_cards', 'copper', 'bull', 'führer',  'sophisticate', 'dog', 'alfalfa', 'asia', 'nester', 'underling', 'copartner', 'rear_up', 'piper', 'supt',  'blunder', 'captive', 'optics', 'auriga', 'kolar', 'guidance', 'steering', 'drafter', 'frontbencher', 'ministerial', 'young_male_servant', 'stockbreeder', 'conciliatory', 'guide_book', 'kitchenhand', 'carder', 'dominee', 'inhabitant', 'concertante',  'mouthpiece', 'drugstore', 'agent', 'temporary_work', 'fipple_flute', 'chamberlain', 'thrifty', 'sensory', 'department_of_commerce', 'jackhammer', 'herd', 'chair',  'prime', 'fighter', 'army', 'officiate', 'dean', 'practitioner', 'modiste', 'hatter', 'sewer','plotting', 'planning', 'fuzz', 'first_proponent', 'initiator', 'bluejacket', 'fierce_man', 'vet', 'participation_in_government',  "grocer's", 'trading', 'chapman', 'commercialize', 'deal', 'trade', 'firm',  'merchandise', 'peddler', 'business_is_business', 'supplier', 'bender', 'autotroph', 'pastoral', 'grazier', 'bucolic', 'johann_gottfried_von_herder', 'ward', 'vigil', 'supervise', 'checker', 'standing_watch',  'prefect', 'scheme', 'plot', 'bring_about', 'commerce', 'commercial_enterprise','huckster', 'juggler', 'commodify', 'market', 'barter', 'negociate', 'industrial', 'enterprising_man', 'child_trafficking', 'trafficker', 'agency', 'retail_outlet', 'franchise', 'provider', \
                                                                   'pedestrian', 'ecclesiastical', 'decigram', 'rancher',  'crimp', 'goon',  'blood_sausage', 'gizzard', 'cattle_dog', 'boötes', \
                                                                   'cattle_droving', 'bull_run', 'purifier', 'vs', 'kosher',  'donor', 'issuer',  'windshield_wiper',  'scanner', 'literary_person', \
                                                                   'skilled', 'pictor', 'writing_desk', 'write_for_someone_else', 'victor', 'aye_aye_sir',  'main', 'shooter',  'whittler', \
                                                                   'pharmacy', 'downloader', 'lower_position',  'negroid', 'black_person',  'sorb', 'serve',\
                                                                   'album', 'vendémiaire','bookcase', 'bunting',  'clop', 'storage_battery', 'gardening', \
                                                                   'staff', 'otto_wagner', 'wagner', 'wilhelm_richard_wagner', 'conservative', 'military', 'edger', 'secant',  'sanitary', 'aesculapian',  'medical', 'factor', 'elizabeth_cochrane_seaman', 'getting_on_board', 'seafaring', 'nautical', 'oceangoing', 'marinate', 'marinade', 'crew', 'galoot', "ship's_company", 'jacob', 'passenger', 'ship', 'military_rank',   'misprint', 'chaser', 'exchange_traded_fund',  'orion', 'guided_tour', 'drawing_card',\
                                                                   'mop',  'national_flag', 'double_crosser', 'device_driver', 'residency',  'systematic', 'favourite', 'enclosed', 'aid', 'adjunct', 'adjoint', 'adjutant_bird', 'concubine',  'shower', 'obstetric', 'mechanized_cavalry', 'cavalry',  'minion', 'intendant', 'sycophant', 'boyfriend', 'pattern', 'friseur', 'samuel_barber', 'barbershop', 'villus', 'apus',  'henchman', \
                                                                   'man_of_war', 'rank_and_file',  'take_care', 'remedy',  'attorneyship', 'litigant',  'moan',  'lament', 'milling_machine', 'jellyfish', 'problem_solver', 'representative',  'betrayer', 'match', 'hunt', 'combatant', 'bold', 'troop',  'terminable', 'impermanent', 'pilot_program', 'squeegee', 'juror',  'gavel', 'pedicure',  'tool', 'mechanical', 'guerrilla', \
                                                                   'controller', 'control', 'pro',  'oar', 'gift', 'teaching', 'professorship', 'magister', 'aio', 'chaperone', 'maestro', 'surname_fu',  'helpmeet', 'bark_beetle', 'hardworking', 'proletarian',  'overall',  'maiden', 'wench', 'bondwoman',  'women',  'majordomo', 'employer', 'proficient',  'male', 'lord', 'cap', 'foster_parent',  'champion', 'skip', \
                                                                   'trust_busting', 'authority', 'craft',  'colleague', 'inn', 'commoner', 'web_server', 'household', 'home', 'pet',  'mechano', 'arm_wrestling', 'troops', 'voyager',  'earthwork', 'earthworks', 'seasonally',  'power_shovel', 'backhoe', 'rod', 'washing_machine', 'sir', 'sidekick', 'helpmate', 'flatterer',  'salary', 'wage', 'seiner', 'peterman', 'schemer', 'conspirator', 'hanging_strap',  'otter_civet', \
                                                                   'seeker',  'alumnus', 'son_translations', 'boy', 'columnist', 'correspond', 'write', 'clerkship',  'moonshee', 'gynecology',  'sales', 'apologist', 'good', 'contributor', 'benefactor', 'subscriber', 'suite',  'requisition', 'support', 'godparent', 'godfather', 'backer', 'dock', 'arranger', 'charger', 'magazine',  'piscatory', 'pescatore', 'kingfisher', \
                                                                   'writing', 'secretary_bird', 'homemade', 'attendee', 'post_carrier', 'postie', 'msw',  'readership', 'reviewer', 'attorney_in_fact', 'proxy', 'bushman', 'blackcoat', 'agriculturist', 'countryman', 'yokel', 'bathyergus', 'mud_dredger', \
                                                                   'flight_simulator',  'useful_person', 'manage_people', 'bantu', 'serf', 'stewardship', 'retinue', 'nigger', 'low_rank_person', 'employment', 'employ',  'service',  'partner', 'fellow', 'kill', 'employed', 'dogsbody', 'follower', 'satellite', 'used', 'abdi', \
                                                                   'ministry', 'vassal', 'stooge', 'abigail', 'assistance', 'subordinate_work', 'avocado', 'preach', 'proponent', 'defense', 'advocaat', 'disciple', 'defender', 'patron', 'promotor', 'seconder',  'eggnog',  'on_sale',  \
                                                                   'kitchen', 'coccus', 'coccal',  'editor_program', 'publishing_house', 'copywriting', 'correspondent', 'lookout','cadre', 'raising_cattle', 'unai', 'keeper', 'watchdog', 'protection', 'train', 'security', \
                                                                   'card',  'junior','strikebreaking', 'violator', 'callee', 'selectee', 'new_recruit', 'novice', 'rookie', 'barrater', 'destructor', 'subject', 'performer_of_action',  \
                                                                   'pollyannaish',  'milking_machine','guy', 'casanova', 'mobilize', 'dummy', 'exemplar', 'blueprint', 'host',   'toilet_bag', 'burner',])
    race_list,  word2ner, block_list =  self.create_multilingual_examples(race_list_en, 'RACE',  word2ner, block_list= block_list + ['oar', 'iris', 'france', 'français', 'frenchy', 'daniel_chester_french', 'frenchie', \
                                                                         'franco', 'potato_chip', 'francia', 'turki', 'francis', 'yankee_doodle', 'francium', 'dharma',\
                                                                         'dwarf','foreigner', 'double_dutch', 'araba','press_gang', 'romance', 'indigenous', 'countryfolk', 'indus', 'italy', 'specialization', 'painter', 'achromatic', 'arabian_horse', 'arabian_peninsula', 'roundoff', 'jewess', 'jews',  'hebrew', 'commonwealth_of_australia',  'white_translations', 'china', 'japan', 'native', 'chinese_language'])
    religious_member_list, word2ner, block_list =  self.create_multilingual_examples(religious_person_en , 'RELIGION_MEMBER',  word2ner, block_list= block_list + religion_list_en + ['duster', 'girlfriend', 'indian', 'roman', 'parse', 'cousin', 'jewry', 'christendom', 'christianity', ])
    religion_list, word2ner, block_list = self.create_multilingual_examples(religion_list_en , 'RELIGION',  word2ner, block_list= block_list + religious_person_en +  ['acid', 'duster', 'girlfriend', 'indian', 'roman', 'parse', 'cousin', 'jewry', 'jewess', 'ghetto', 'priesthood', 'church', 'sacred_teachings', 'christendom', 'portuguese_jesuits', 'person', 'mohammedan', 'moslem'])
    union_list, word2ner, block_list =  self.create_multilingual_examples(union_en , 'UNION',  word2ner, block_list= block_list + ['union', 'syndicate',])
    social_economic_class_list, word2ner, block_list =  self.create_multilingual_examples(social_economic_class_en , 'SOC_ECO_CLASS',   word2ner, block_list= block_list + ['brotherliness', 'freemasonry', 'fraternization', 'clan', 'fellowship', 'handicraft', 'commerce', 'commercialize', 'adele',\
                                                                                                                                                                                'edwin_herbert_land', 'work', 'press', 'time', 'service',  'wife', 'mother', 'female', 'times', 'anthology', 'better', 'breeding', 'choose', 'chivalry', 'noble', 'jue', 'set'])
    political_party_list, word2ner, block_list =self.create_multilingual_examples(politcal_party_en , 'POLITICAL_PARTY',  word2ner, block_list= block_list +['ungradable_opposition', 'work', 'time', 'service'])
    political_party_member_list,  word2ner, block_list = self.create_multilingual_examples(poltical_party_member_en , 'POLITICAL_PARTY_MEMBER',  word2ner, block_list=  ['red', 'rouge','communistically' ])
    public_figure_list, word2ner, block_list =  self.create_multilingual_examples(public_figure_en , 'PUBLIC_FIGURE',  word2ner, block_list= block_list+  ['gulf_of_saint_lawrence', 'allene', 'james', 'maldives',  'borgia', 'person_of_promise', 'zeeman', 'director', 'they', 'bi', 'kid', 'surname_franklin',  'will', 'edison', 'ego', 'me', "i'm", 'genus_of_humans', 'king', 'neumann', 'augustus', 'couple', 'norman', 'rus', 'foil', 'joule', 'an', 'majority', 'christopher', 'larry', 'wolf', 'swallow', 'ouch', 'elder', 'single', "o'clock", 'ptarmigan', \
                                                                                                              'forth',  'boast', 'fist',  'zen', 'bunch', 'ostrich', 'spouse', 'woman', 'goatfish', 'wife', 'chaise_longue', 'outer_wall', 'butene', 'paralysis_agitans',  'shrub', 'weaver',   'plough', 'rich', 'algeria', 'cockchafer', \
                                                                                                              'bumblebee',  'alaska',  'heart', 'flemish', 'schmidt_island', 'petro', 'herz_or_cycles_per_second', 'limb', \
                                                                                                              'everyone', 'everybody', 'yes', 'logos', 'messiah', 'turner', 'gymnast', 'return', 'gy',  'freemason', 'mason',  'lake_edward', 'alabama', 'los_angeles', 'loss', 'losings', 'passing', 'steed', 'ross_island', \
                                                                                                              'horse', 'rice', 'young', 'jr', 'japan_railways', 'jnr', 'gilbert_islands', 'slate',  'show',  'stone', 'stoned', 'nub', 'earthen',  'benedictine', 'fuse',  'oxford_shoe', 'dino_paul_crocetti', 'orphan_drug', \
                                                                                                              'leninist','ash', 'hunt', 'brown', 'brown_university', 'breezy', 'fresh', 'normans',  'grant', 'grunt', 'moses_basket', 'musa', 'burton', 'marx',  \
                                                                                                              'negroid',  'gray', 'grey', 'why', 'wood', 'man', 'beeper',  'hong_kong', 'never',  'vomit', 'puke', "st_george's", 'massachuset', 'massachusetts', 'track_clearing_vehicle', \
                                                                                                              'snowplow', 'word', 'ward', 'care',  'anger', 'grim', 'midnight', 'houston',  'farmer', 'post', 'morgan_horse', 'moody', \
                                                                                                              'life_annuity', 'hooker',  'leaky', 'jogging', 'cooper', 'white', 'white_person',  \
                                                                                                              'russia', 'echinopsis_pachanoi',  'touché', 'cytisus_scoparius', \
                                                                                                              'white_island',  'bark', 'hi_hat',  'troglodytidae', 'key', \
                                                                                                              'shop_assistant', 'clerk', 'kulak',  'crane', 'grus', \
                                                                                                              'new_jersey', 'mayor', 'baltic', 'marsh', 'gardener', 'toasting', 'fry', 'gold_translations', "poor_person's_house",  'isochrone', 'tacit',  \
                                                                                                              'too', 'missouri', 'father_christmas', 'duce', 'stalinist'])
    person_list, word2ner, block_list =  self.create_multilingual_examples(person_en , 'PERSON',  word2ner, block_list= block_list+ ['para_rubber', 'cosmic_dual_forces', 'book_of_changes', 'forth',  'boast', 'fist',  'zen', 'bunch', 'ostrich', 'spouse', 'woman', 'goatfish', 'wife', 'chaise_longue', 'outer_wall', 'butene', 'paralysis_agitans',  'shrub', 'weaver',   'plough', 'rich', 'algeria', 'cockchafer', \
                                                                                                              'bumblebee',  'alaska',  'heart', 'flemish', 'schmidt_island', 'petro', 'herz_or_cycles_per_second', 'limb', \
                                                                                                              'everyone', 'everybody', 'yes', 'logos', 'messiah', 'turner', 'gymnast', 'return', 'gy',  'freemason', 'mason',  'lake_edward', 'alabama', 'los_angeles', 'loss', 'losings', 'passing', 'steed', 'ross_island', \
                                                                                                              'horse', 'rice', 'young', 'jr', 'japan_railways', 'jnr', 'gilbert_islands', 'slate',  'show',  'stone', 'stoned', 'nub', 'earthen',  'benedictine', 'fuse',  'oxford_shoe', 'dino_paul_crocetti', 'orphan_drug', \
                                                                                                              'leninist','ash', 'hunt', 'brown', 'brown_university', 'breezy', 'fresh', 'normans',  'grant', 'grunt', 'moses_basket', 'musa', 'burton', 'marx',  \
                                                                                                              'negroid',  'gray', 'grey', 'why', 'wood', 'man', 'beeper',  'hong_kong', 'never',  'vomit', 'puke', "st_george's", 'massachuset', 'massachusetts', 'track_clearing_vehicle', \
                                                                                                              'snowplow', 'word', 'ward', 'care',  'anger', 'grim', 'midnight', 'houston',  'farmer', 'post', 'morgan_horse', 'moody', \
                                                                                                              'life_annuity', 'hooker',  'leaky', 'jogging', 'cooper', 'white', 'white_person',  \
                                                                                                              'russia', 'echinopsis_pachanoi',  'touché', 'cytisus_scoparius', \
                                                                                                              'white_island',  'bark', 'hi_hat',  'troglodytidae', 'key', \
                                                                                                              'shop_assistant', 'clerk', 'kulak',  'crane', 'grus', \
                                                                                                              'new_jersey', 'mayor', 'baltic', 'marsh', 'gardener', 'toasting', 'fry', 'gold_translations', "poor_person's_house",  'isochrone', 'tacit',  \
                                                                                                              'too', 'missouri', 'father_christmas', 'duce', 'stalinist'])

    org_list, word2ner, block_list =  self.create_multilingual_examples(organization_en, 'ORG',  word2ner, block_list=  block_list +  ['u.s.a', 'axle',  'heart', 'pit', 'sister', 'ivy', 'stone', 'europa', 'octane',  'tripoli', 'eleven', 'uno', 'epi', 'fn', 'protester', 'work', 'time', 'service', 'commonwealth',  'acid', 'church', 'protestantism', 'quaker', 'sunni_muslim', 'anglicanism', 'baptist', 'united_states_of_america', 'eve', 'us', 'americas',  'normalized_projection_coordinates', 'non_player_character', 'aviation', 'dark_blue', 'personal_handyphone_system', 'iaca', 'which', 'identity', 'mu', 'storm_troopers', 'kingdom', 'might', 'orthodoxy', 'western_christianity', 'hassidism', 'jesuits','eicosapentaenoic_acid', 'woman', 'mail', 'bebop', 'computer_integrated_manufacturing', 'intersymbol_interference', 'ice', 'weak', 'doctor', 'first_lord_of_treasury', 'sixth_form', 'peddlers_and_carriers', 'utility', 'gallery', 'fiction', 'st', 'fantasy_fiction', 'workforce', 'working_force', 'manpower', 'criminal_law', 'proletariat',  'ourselves', 'overtake',  'trojan', 'have_good_meal', 'talk_of_town', 'royalty', 'armada', 'minority_peoples', 'enlightenment', 'almighty', 'high_society', 'showbusiness', 'every_so_often', 'occasionally', 'sometimes', 'once_in_while', 'nicaea', 'istanbul', 'constantinople', 'bridal_shower', 'nazism', 'bank', 'deposit', 'passkey', 'loper', 'ticket', 'amis', 'date',  'market', 'exchange','cdu', 'national_insurance', 'jobcentre', 'constitutional_nationalist_party', 'guomindang_or_kuomintang',  'plaza', 'shopping_center', 'center', 'forefront', 'vanguard', 'bigger_than',  'unknown', 'latte', 'qin',  'supply_network', 'coup', 'putsch', 'overthrow', 'dance_studio', 'yellow_river_or_huang_he', 'huang_ho', 'china’s_sorrow','band', 'syndicate', 'union', 'stag_do', 'greenhorn', 'municipality', 'prefecture', 'royal', 'hiring_hall', 'downtown', 'high_street', 'rainforest', 'group_sounds', "worker's_party", 'indus', 'step_by_step', 'gradually', 'one_by_one', 'by_degrees', 'mincing', 'another_name_for',  'institute', 'bliss', 'on_cloud_nine', 'intelligence', 'antigua_and_barbuda', 'icu', 'fisc', 'estate', 'someone_else', 'firefighter', 'civic_center', 'gum', 'think_factory', 'soldier', 'fdp', 'ldp', 'tropical_forest', 'refrain', 'customs', 'videoconferencing', 'house', 'interest', 'lobby', 'st_john’s', 'civil_right',  'schoolmate', 'academic', 'flock', 'handiwork', 'handicraft', 'wake', 'omnipotent', 'on_and_off', 'occasional', 'season', 'at_times', 'ever_and_anon', 'intermittently', 'frequently', 'often', 'amongst', 'incidentally', 'finn', 'every_now_and_then', 'seldom', 'at_long_intervals', 'downpour', 'byzantium', 'trade', 'mall', 'centre','bustling_street', 'shopping_district', 'kernel', 'midst', 'middle', 'inner_city', 'frontline', 'underground', 'avantgarde', 'head', 'salient', 'forward', 'senate', 'insurgency', 'subversion', 'horizontal_union', 'orchestra', 'ensemble',  'crying', 'weeping', 'gradual', 'stepwise', 'by_small_degrees', 'drop', 'slowly', 'incrementally', 'tardily', 'gently', 'little_for_each', 'in_course_of_time', 'military_personnel',  'fighter', 'serviceman', 'collegiate', 'initiation', 'homemade', 'craft', 'omnipotence', 'constantly', 'more_often_than_not', 'oftentimes', 'unexpectedness', 'emergency', 'thick', 'great_deal', 'commonly', 'oft', 'repeatedly', 'inula', 'frequent', 'btw', 'rarely', 'commercialize', 'marketing', 'shop', 'store', 'midfielder', 'mid_way', 'nucleus', 'center_field', 'cell_nucleus', 'hub', 'midpoint', 'central', 'centrum', 'heartland', 'centrism', 'hit', 'right_in_midst_of', 'right_at_height_of', 'intermediate', 'midplane', 'about_middle', 'advanced_guard', 'rebellion', 'lamenting', 'quietly', 'soon', 'steadily', 'progressively', 'oozing_out', 'slowly_permeating', 'slowly_soaking_in', 'seeping_out', 'slow', 'deliberately', 'suddenly', 'abruptly', 'whisper', 'leisurely', 'surname_xu', 'behind', 'sailor', 'warrior', 'military_strength', 'soldiers', 'legion', 'host', 'private'])
    print ('num of items', len(word2ner))
    disease_list, word2ner, block_list = self.create_multilingual_examples(disease_list_en , 'DISEASE',  word2ner, block_list=block_list +  ['fume',  'shakes', 'see',  'pumpkinseed', 'calenture', 'insulation', 'heliotherapy', 'bleb', 'vesicle', 'hangnail', 'bubble', 'become_bruised', 'bluish_black', 'effort', 'judder', 'bowel',  'stooping', 'hungry', 'crossed_eyes',  'wound_suffered_in_fight', 'external_wound', 'purple_spot', 'hymen', 'kokborok', 'bow_leggedness', 'deuteranopic', 'concentration', 'vaccine', 'shin_splint', 'nasal_vowel', 'fistule', 'aphthous_fever', 'chimney', 'snake', 'ornitosis', 'steppe_murrain', 'vampirism', 'pustulate', 'blackhead', 'comedo', 'bread_mold', 'brand', 'mildew', 'oak_blight', "sow_one's_oats",  'skin',  'limp', 'hitch', 'polyopia',  'hindquarters', 'honeycomb', 'year', 'shawl', 'balance','shine', 'fan',  'rosaceae', 'rose_window',  'mound', 'comforter', 'strawberry', 'lollipop', 'croupy', 'useless', 'superfluous', 'clairvoyance', 'pesto', 'pain_in_ass', 'zostera',  'raft','germ', 'virus_that_infects_bacteria', 'amenorrheic', 'dizzy', 'dazzled', 'dazzle', 'giddy', 'feeling_of_swaying', 'fainting', 'dizzyness', 'dust_devil', 'blinding', 'probably', 'reeling', 'labored_breathing', 'panting', 'pant', 'abbot', 'volcanic_eruption', 'shrew', 'hiccups', 'abnormally_high_blood_sugar_level',  'weak_digestion', 'mule',  'bitterness', 'ketose', 'scruple',  'tingling', 'plight', 'miserable', 'asperity', 'harassment', 'dolor', 'tartar', 'dregs', 'sadness', 'whale', 'kipu', 'unpleasantness', 'quipu', 'pulse', 'extract_and_use', 'hamper', 'jerk', 'cramps', 'clamp',  'knock_on_effect', 'beech', 'wild_cherry', 'callous', 'rennet', 'curd', 'martyr', 'woe', 'flatulent', 'poignance', 'birth_pains', 'contractions', 'twinge', 'afflict', 'harass', 'cut_to_pieces', 'rack', 'persecute', 'crucifixion', 'agonize', 'painfully', 'convulsively', 'wince', 'convex', 'free_alongside_ship', 'beam', 'stage_fright',  'white_coloured_skin', 'hectic', \
                                                                                   'disinclination', 'morbidity', 'sob_convulsively', 'stranglehold', 'blockade', \
                                                                                   'suffocate', 'stifle', 'gag', 'house_mouse', 'sick',   'ailing', 'drowning', 'strangling',\
                                                                                   'deterioration', 'default_option', 'faint', 'mental_case', 'interruption', \
                                                                                   'nuisance', 'thirsty', 'famine', 'appetite', 'crave', 'drought', 'gose',  \
                                                                                   'redden', 'turn_red', 'feel_hot', 'cauterize', 'red', 'turn_pink', 'repair',\
                                                                                   'form', 'state', 'mental_determination', 'exuberance', 'viability', 'vigour',\
                                                                                   'liveliness', 'coolness', 'emanation', 'shining', 'ray', 'tanned', 'sun_tanning', \
                                                                                   'darken', 'burn_off', 'sear', 'char', 'erupt', 'kindle', 'wrath', 'irritability', \
                                                                                   'burn_hole', 'set_afire', 'pique', 'luminous', 'prickling', 'blanch', \
                                                                                   'hanker', 'hydrogen', 'gravy', 'break_out', 'bleaching', 'difficulty', 'adversity', \
                                                                                   'sorrow', 'worry', 'powerlessness', 'inability', 'languor', 'dysgenics', 'grogginess', 'despondency', \
                                                                                   'tiredness', 'diffraction_grating', 'misrepresent', 'freak', 'plant_process', 'enlargement', 'hunch', \
                                                                                   'binge', 'rabia', 'choler', 'ire', 'bile', 'ferocity', 'furor', 'captivation', 'furious',\
                                                                                   'insanity', 'bluster', 'hot_anger', 'violent_rage', 'angry', 'resentment', 'swell', 'talisman', 'amulet', 'ignition', \
                                                                                   'node', 'bulge', 'self_conceited', 'torridity', 'hot_weather', 'carsickness',  'tide',  'korea',  'anxiety', 'delusion', \
                                                                                   'raving', 'nightmare', 'lettuce', 'babble', 'stuttering', 'bumble', 'gluttony', 'destitution',  \
                                                                                   'what', 'ch_i', 'whether', 'flag', 'hangar', 'ordeal', 'sufferance', 'pain_and_difficulties', 'concern', 'distress_signal', 'grief', \
                                                                                   'attachment', 'pathological', 'disgust', 'nature_of_disease', 'demon_of_ill_health', 'fatigue', 'taint', 'catching', \
                                                                                   'spread', 'daze', 'misfortune', 'scathe', 'traumatism', 'score', 'infringement', 'violation', 'frisson', 'damage',  'impediment', 'traumatize', \
                                                                                   'tingle', 'prick', 'chagrin', 'be_sick', 'rankle', 'love_dearly', 'hurts', 'ail', 'simmer', 'if', 'fence', 'blemish', 'damnification', \
                                                                                   'nullity', 'cajolement',  'boiling_point', 'anginal', 'sharp_pain', 'griping_pain',  'sweetheart', 'melancholy', 'low', 'crisis', \
                                                                                   'craziness', 'morbid_fear', 'fissionable', 'cholesterol', 'suidae', 'cold_and_heat', 'malarial', 'tension', 'cave_in', 'crash', 'old',\
                                                                                   'fail', 'crumble', 'also', 'blunder', 'arrogance', 'amour_propre', 'greenery', 'tumo_u_r', 'plant', 'outgrowth', 'mass', 'tax', 'gavel', \
                                                                                   'able_man', 'fine_man', 'painful', 'raw', 'painfulness', 'bleed', 'bloodloss', 'resentful', 'suggillation', 'claw', 'architect', \
                                                                                   'hump', 'bozzo', 'hunk', 'dent', 'impact', 'protuberance', 'suspension',  'jolt', 'break_in', 'break_apart', 'fault', 'schism', 'breakage',  'tea', \
                                                                                   'break_off_relations_with', 'burst_open', 'obstruct', 'burstenness', 'severance', 'strut', 'stalemate', 'stab', 'bug_bite', 'compunction', 'resent', \
                                                                                   'encourage', 'stretch', 'dislocate', 'streak', 'linear_mark', 'mosquito', 'infestation', 'district', 'heatwave', 'tyrannize', 'screw', 'distort', 'twine', \
                                                                                   'tortuosity', 'quarrel', 'torsion', 'cistus',  'dartos', 'driveway', 'cigarette',  'contraction',  'incapacity', 'advantage', 'drawback', 'indolence', \
                                                                                   'fragility', 'palpitate', 'crown_shaped', 'hysterical', 'mishegoss', 'rampage', 'anomie',  'alarm',  'monkey', 'knob_on_tree', 'craw', 'neoplasia', 'spit', \
                                                                                   'skewer', 'swollen', 'gland', 'pick', 'pickaxe', 'fumigate', 'bay', 'tuberosity', 'nodule', 'large_breasts', 'reverse', 'brown',  'abrade',  'incision', 'nick',\
                                                                                   'attrition', 'rift', 'stria', 'paw', 'rub', 'stub', 'brush_against', 'scuff_mark', 'hack', 'chop', 'scarred_skin', 'insult', 'machine', 'cockroach', 'notch', 'slit', \
                                                                                   'brush', 'deletion', 'giulio_natta', 'be_paralyzed', 'standstill', 'paralytic', 'every_second_solar_term', \
                                                                                   'hindrance', 'flow', 'help', 'fuss', 'annoyance', 'scab', 'ringworm', 'fire_salamander', 'leper', 'rowan', 'nip', \
                                                                                   'shot', 'unevenness',  'lopsided',  'dissimilarity', 'unsoundness', 'deviation', 'falsehood', 'divergence', 'dependent', \
                                                                                   'outbuilding', 'reliance',  'relation', 'anesthetization', 'local_anesthetic', 'general_anesthetic', 'anesthesiologist', \
                                                                                   'bang', 'suppression', 'unrest', 'confusion', 'swage', 'sufferer', 'patient', 'irritate', 'eagerness', 'hankering', \
                                                                                   'willpower', 'warmheartedness', 'warmth', 'ardor', 'hot', 'bear', 'berry', 'hole', 'puberty', 'ardour', 'pulsation', 'vibe', 'love',\
                                                                                   'lecherousness', 'zeal', 'enthusiasm', 'violent_emotion', 'strong_emotion', 'temperature', 'displeasure', \
                                                                                   'burning_sensation', 'sultriness', 'blaze', 'passion_play', 'flame', 'shave', 'ferociousness', 'reproductive_procreative_power', \
                                                                                   'fruitfulness', 'fertility_rate', 'bloating', 'meteorism', 'fart', 'novelty', 'aplomb', 'carbon_oxide', 'high_fever', 'pyro', \
                                                                                   'filibuster', 'traffic_jam', 'brainwashing', 'drink',  'consequence', 'perplexity', 'parturiency', 'motherhood', \
                                                                                   'rest', 'deep_sleep', 'death', 'gound', 'quiescence', 'fruitlessness',  'unproductiveness', 'apparent_death', 'listlessness', 'stupor', 'apathy', \
                                                                                   'sleeplessness', 'eve', 'grinding', 'distortion', 'block', 'degeneration', 'regression_analysis', 'rancor', 'huff', 'sequence', \
                                                                                   'cross_eyed', 'sidelong', 'walleyes', 'stiff_neck', 'rearrangement', 'excited', 'longing', 'craving', 'wish', 'want', 'foreplay', \
                                                                                   'weakness', 'mischief', 'tribulation',  'health', 'discomfort', 'suffer', 'distressed', 'anguish', 'agonised', 'affect', 'have', \
                                                                                   'affection', 'misery', 'bout', 'sicken', 'menstruation', 'injured', 'wounded', 'injure', 'hardship',  'tin', 'skinny', 
                                                                                   'preponderance', 'fatness', 'silk', 'eunuch', 'side', 'seize', 'apprehend', 'take', 'more_commonly_known_as', 'goose_pimple', 'see_also', \
                                                                                   'gasp', 'clap', 'lobster', 'hit', 'trait', 'touch', 'ironing', 'fondle', 'fit', 'throw', 'chuck', 'terabyte', 'crayfish', 'consumption', \
                                                                                   'thickness', 'cold', 'fleshiness', 'caress', 'itch', 'blow', 'stoutness', 'crab'])
    product_list, word2ner, block_list = self.create_multilingual_examples(product_en ,  'PRODUCT', word2ner, block_list=block_list +  ['promnesia', 'nude_body', 'celestial_object', 'moon', 'undoubtedly',  'jump_up', "leap_to_one's_feet", 'day_of_wren', "friar's_lantern", 'orb', 'dead_end_street', 'no_through_road', 'closure', 'cul', 'deadlock', 'catch_22', 'england', 'britain', 'kingdom_of_great_britain', 'britannia'])
    #work_of_art_list, word2ner,  block_list = self.create_multilingual_examples(work_of_art_en ,  'WORK_OF_ART', word2ner, block_list=block_list)
    print ('num of items', len(word2ner))
    block_list = None
    langs =[]
    nerbylangs = {}
    self.ontology = {}
    self.ontology_compound_word_start = {}
    word2ner = list(set(word2ner))
    for word,label in word2ner:
      langs.extend(list(self.word2lang.get(word,[]))) 
      for lang in self.word2lang.get(word,[]):
        nerbylangs[lang] = nerbylangs.get(lang, [])+ [label]
    self.word2ner = word2ner = list(set([(word.translate(trannum), label) for word, label in word2ner]))
    print ('num of items', len(word2ner))
    os.system(f"mkdir -p {data_dir}")
    json.dump(word2ner, open(f"{data_dir}/{word2ner_file}", "w", encoding="utf8"), indent=1)
    os.system(f"mv {data_dir}/{word2ner_file} {shared_dir}/{word2ner_file}")
    self.add_to_ontology(word2ner)
    for lang in list(nerbylangs.keys()):
      nerbylangs[lang]= Counter(nerbylangs[lang]).most_common()
    for lang, cnt in Counter(langs).most_common():
      print ((lang, cnt))
      print ("  "+ str(nerbylangs[lang]))
    male_to_female_gender_swap = copy.copy(OntologyBuilder.male_to_female_gender_swap)
    binary_gender_swap = copy.copy(OntologyBuilder.female_to_male_gender_swap)
    other_gender_swap = copy.copy(OntologyBuilder.other_gender_swap)

    for a, b in male_to_female_gender_swap.items():
      binary_gender_swap[b] = a

    for word in language_list_en + race_list_en + jobs_en + religion_list_en:
      word = word.replace(" ", "_").replace("-", "_").lower().strip(".") 
      for gender in ['female', 'woman', 'lady', 'gay', 'lesbian']:
        for word2 in [gender+"_"+word, word+"_"+gender]: 
          if word2 in self.en:
            for gender2 in ['male', 'man', 'person']:
              for word3 in [gender2+"_"+word, word+"_"+gender2]: 
                if word3 in self.en:
                  if word2 not in binary_gender_swap:
                    binary_gender_swap[word2] = word3
                  if word3 not in binary_gender_swap:
                    binary_gender_swap[word3] = word2
              word3 = word
            if word3 in self.en:
              if word3 not in ('white', 'black', 'brown') and word2 not in ('white', 'black', 'brown') :
                if word2 not in binary_gender_swap:
                  binary_gender_swap[word2] = word3
                if word3 not in binary_gender_swap:
                  binary_gender_swap[word3] = word2
                  
    binary_gender_swap= self.create_multilingual_map(binary_gender_swap)
    other_gender_swap= self.create_multilingual_map(other_gender_swap)
    en_pronoun2gender= self.create_multilingual_map(self.pronoun2gender)
    en_pronoun2pronoun= self.create_multilingual_map(self.pronoun2pronoun)
    en_pronoun2title= self.create_multilingual_map(self.pronoun2title)
    person2religion= self.create_multilingual_map(self.person2religion)

    #export formats and words from faker
    lang2person = {}
    for lang in OntologyBuilder.faker_list:
      lang2 = lang.split("_")[0]
      aHash = lang2person.get(lang2, {})
      if not hasattr(faker.providers.person, lang):
        exec(f"import faker.providers.person.{lang}")
      provider = getattr(faker.providers.person,  lang)
      if hasattr(provider.Provider, 'formats'):
        ner_regexes = aHash.get('ner_regexes',[])
        if not type(provider.Provider.formats) is dict:
          ner_regexes += [("PERSON", "|".join([a.replace("{{","<").replace("}}",">\d+").upper()  for a in provider.Provider.formats]), False, ())]
        else:
          ner_regexes += [("PERSON", "|".join([a.replace("{{","<").replace("}}",">\d+").upper()  for a in  provider.Provider.formats.keys()]), False, ())]
        aHash['ner_regexes'] = ner_regexes
      if hasattr(provider.Provider, 'first_names_female'):
        if not type(provider.Provider.first_names_female) is dict:
          aHash['FIRST_NAME_FEMALE'] = list(set(aHash.get('FIRST_NAME_FEMALE', []) + list(provider.Provider.first_names_female)))
        else:
          aHash['FIRST_NAME_FEMALE'] = list(set(aHash.get('FIRST_NAME_FEMALE', []) + list(provider.Provider.first_names_female.keys())))
      if hasattr(provider.Provider, 'first_names_male'):
        if not type(provider.Provider.first_names_female) is dict:
          aHash['FIRST_NAME_MALE'] = list(set(aHash.get('FIRST_NAME_MALE', []) + list(provider.Provider.first_names_male)))
        else:
          aHash['FIRST_NAME_MALE'] = list(set(aHash.get('FIRST_NAME_MALE', []) + list(provider.Provider.first_names_male.keys())))
      if hasattr(provider.Provider, 'last_names_female'):
        if not type(provider.Provider.last_names_female) is dict:
          aHash['LAST_NAMES_FEMALE'] = list(set(aHash.get('LAST_NAMES_FEMALE', []) + list(provider.Provider.last_names_female)))
        else:
          aHash['LAST_NAMES_FEMALE'] = list(set(aHash.get('LAST_NAMES_FEMALE', []) + list(provider.Provider.last_names_female.keys())))
      if hasattr(provider.Provider, 'LAST_NAMES_MALE'):
        if not type(provider.Provider.last_names_male) is dict:
          aHash['LAST_NAMES_MALE'] = list(set(aHash.get('LAST_NAMES_MALE', []) + list(provider.Provider.last_names_male)))
        else:
          aHash['LAST_NAMES_MALE'] = list(set(aHash.get('LAST_NAMES_MALE', []) + list(provider.Provider.last_names_male.keys())))
      if hasattr(provider.Provider, 'prefixes_female'):
        if not type(provider.Provider.prefixes_male) is dict:
          aHash['PREFIX_FEMALE'] = list(set(aHash.get('PREFIX_FEMALE', []) + list(provider.Provider.prefixes_female)))
        else:
          aHash['PREFIX_FEMALE'] = list(set(aHash.get('PREFIX_FEMALE', []) + list(provider.Provider.prefixes_female.keys())))
      if hasattr(provider.Provider, 'prefixes_male'):
        if not type(provider.Provider.prefixes_male) is dict:
          aHash['PREFIX_MALE'] = list(set(aHash.get('PREFIX_MALE', []) + list(provider.Provider.prefixes_male)))
        else:
          aHash['PREFIX_MALE'] = list(set(aHash.get('PREFIX_MALE', []) + list(provider.Provider.prefixes_male.keys())))
      
      if hasattr(provider.Provider, 'sufixes_female'):
        if not type(provider.Provider.sufixes_male) is dict:
          aHash['SUFIX_FEMALE'] = list(set(aHash.get('SUFIX_FEMALE', []) + list(provider.Provider.sufixes_female)))
        else:
          aHash['SUFIX_FEMALE'] = list(set(aHash.get('SUFIX_FEMALE', []) + list(provider.Provider.sufixes_female.keys())))
      if hasattr(provider.Provider, 'sufixes_male'):
        if not type(provider.Provider.sufixes_male) is dict:
          aHash['SUFIX_MALE'] = list(set(aHash.get('SUFIX_MALE', []) + list(provider.Provider.sufixes_male)))
        else:
          aHash['SUFIX_MALE'] = list(set(aHash.get('SUFIX_MALE', []) + list(provider.Provider.sufixes_male.keys())))
      

      if hasattr(provider.Provider, 'first_names'):
        if not type(provider.Provider.first_names) is dict:
          aHash['FIRST_NAME'] = list(set(aHash.get('FIRST_NAME', []) + list(provider.Provider.first_names)))
        else:
          aHash['FIRST_NAME'] = list(set(aHash.get('FIRST_NAME', []) + list(provider.Provider.first_names.keys())))
      if hasattr(provider.Provider, 'last_names'):
        if not type(provider.Provider.last_names) is dict:
          aHash['LAST_NAME'] = list(set(aHash.get('LAST_NAME', []) + list(provider.Provider.last_names)))
        else:
          aHash['LAST_NAME'] = list(set(aHash.get('LAST_NAME', []) + list(provider.Provider.last_names.keys())))
      lang2person[lang2] = aHash

    lang2extra = {}
    for word, label in word2ner:
      if word in self.word2lang:
        if label == "OTHER_PRONOUN":
            for lang in self.word2lang.get(word, []):
              aHash = lang2person.get(lang, {})
              aHash[label] = aHash.get(label, []) + [word]
              lang2person[lang] = aHash  
        else:
          _, label2 = self.in_ontology(word)
          if label2 != label:
            for lang in self.word2lang.get(word,[]):
              aHash = lang2extra.get(lang, {})
              aHash[label] = aHash.get(label, []) + [word]
              lang2extra[lang] = aHash

    langs = list(set(list(lang2person.keys()) + list(binary_gender_swap.keys()) + list(other_gender_swap.keys()) + list(en_pronoun2gender.keys()) + list(en_pronoun2pronoun.keys()) + list(person2religion.keys())))
    for lang in langs:
      if lang in ('sw',):
        ret['LAST_NAME'] = list(set(ret.get('LAST_NAME', []) + [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in OntologyBuilder.bantu_surnames]))
      
      ret = lang2extra.get(lang, {})
      personHash = lang2person.get(lang, {})
      for key, val in personHash.items():
        #assume lang2exra are just lable => lists
        if key in ret:
          ret[key] = ret[key] + val
        else:
          ret[key] = val
      ret['FIRST_NAME_MALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('FIRST_NAME_MALE', [])]))
      ret['LAST_NAME_MALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('LAST_NAME_MALE', [])]))
      ret['FIRST_NAME_FEMALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('FIRST_NAME_FEMALE', [])]))
      ret['LAST_NAME_FEMALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('LAST_NAME_FEMALE', [])]))
      ret['FIRST_NAME'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('FIRST_NAME', [])]))
      ret['LAST_NAME'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('LAST_NAME', [])]))
      ret['SUFIX_MALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('SUFIX_MALE', [])]))
      ret['SUFIX_FEMALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in ret.get('SUFIX_FEMALE', [])]))
      
      ret['binary_gender_swap'] = binary_gender_swap.get(lang, {})
      ret['other_gender_swap'] = other_gender_swap.get(lang, {})
      ret['en_pronoun2gender'] =  en_pronoun2gender.get(lang, {})
      ret['en_pronoun2pronoun'] =  en_pronoun2pronoun.get(lang, {})
      ret['en_pronoun2title'] =  en_pronoun2title.get(lang, {})
      en_pronoun2title = ret['en_pronoun2title']
      ret['PREFIX_MALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in en_pronoun2title.get('he',[])+ret.get('PREFIX_MALE', [])]))
      ret['PREFIX_FEMALE'] =  list(set([word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in en_pronoun2title.get('she',[])+ret.get('PREFIX_FEMALE', [])]))
      ret['person2religion'] =  person2religion.get(lang, {})
      if lang == 'en':
        ret['ner_regexes'] =  ret.get('ner_regexes',[]) + OntologyBuilder.default_ner_regexes
      json.dump(ret, open(f"{data_dir}/{lang}.json", "w", encoding="utf8"), indent=1)
    self.save_x_lingual_lexicon_prefix_file()

  def load_ontology(self, word2ner_file=None):
    data_dir = self.data_dir
    shared_dir = self.shared_dir
    if not os.path.exists(word2ner_file):
      word2ner_file = f"{data_dir_dir}/{word2ner_file}"
    return json.load(open(word2ner_file, "rb").decode())

  def word2cat(self):
    shared_dir = self.shared_dir
    cat2word = json.load(open(f"{shared_dir}/conceptnet_ontology_cat2word.json", "rb").decode())
    from collections import Counter
    lst = list(cat2word.items())
    lst.sort(key=lambda a: a[1], reverse=True)
    print (lst[:10])
    word2cat= {}
    for cat, words in cat2word.items():
      for word in words:
        word2cat[word] = cat
    return word2cat

  def create_multilingual_map(self, en_examples, allow_list=None, cut_off_abs=4, ):
    en = self.en
    word2en = self.word2en
    word2lang  = self.word2lang
    keys = list(en_examples.keys())  
    allow_list = dict([(a, 1) for a in allow_list or []])  
    ret_hash = {}
    print (keys)
    if not keys: return ret_hash
    if type(en_examples[keys[0]]) is dict:
      for item in keys:
        mapHash = en_examples[item]
        for key in mapHash.keys():
          words = [word.replace(" ", "_").replace("-", "_").lower().strip(".") for word in mapHash[key]]
          for word in words:
            if word not in en:
              continue
            words2 = [word2 for word2 in en[word] if len(word2en[word2]) <= cut_off_abs]
            if not words2:
              continue
            words2.sort(key=lambda a: len(a))
            word2 = words2[0]
            for lang in word2lang.get(word2,[]):
              aHash = ret_hash.get(lang, {})
              if allow_list is not None and word2 not in allow_list: continue
              aHash[item] = list(set(list(aHash.get(item, [])) + [word2]))
              ret_hash[lang] = aHash
    elif type(en_examples[keys[0]]) is list:
      for item, words in en_examples.items():
        lang2words = {}
        for key2 in words:
          key2 = key2.replace(" ", "_").replace("-", "_").lower().strip(".")
          lang2words_list = itertools.chain(*[[(lang, word2) for lang in word2lang.get(word2, ['en'] if word2 == key2 else [])] \
                          for word2 in en.get(key2, []) + [key2] if word2 == key2 or (word2 in word2en and len(word2en[word2])) <= cut_off_abs])
          
          for lang, word2 in lang2words_list:
            lang2words[lang] = lang2words.get(lang, []) + [word2]
        lang2words['en'] = words
        for lang, words2 in lang2words.items():
          aHash = ret_hash.get(lang, {})
          for word in words2:
            if allow_list and word not in allow_list: continue
            aHash[item] = list(set(aHash.get(item, []) + [word]))
          ret_hash[lang] = aHash
    else:
      for key in en_examples:
        key2 = en_examples[key]
        key = key.replace(" ", "_").replace("-", "_").lower().strip(".")
        if key not in en:
          continue
        lang2words_list = itertools.chain(*[[(lang, word) for lang in word2lang.get(word, ['en'] if word == key else [])] \
                        for word in en.get(key, []) + [key] if word == key or (word in word2en and len(word2en[word])) <= cut_off_abs])
        lang2words = {}
        for lang, word in lang2words_list:
          lang2words[lang] = lang2words.get(lang, []) + [word]
        lang2words['en'] = [key]
        key2 = key2.replace(" ", "_").replace("-", "_").lower().strip(".")
        lang2words2_list = itertools.chain(*[[(lang, word2) for lang in word2lang.get(word2, ['en'] if word2 == key2 else [])] \
                        for word2 in en.get(key2, []) + [key2] if word2 == key2 or (word2 in word2en and len(word2en[word2])) <= cut_off_abs])
        lang2words2 = {}
        for lang, word2 in lang2words2_list:
          lang2words2[lang] = lang2words2.get(lang, []) + [word2]
        lang2words2['en'] = [key2]
        for lang in lang2words.keys():
          for word in lang2words[lang]:
            if allow_list and word not in allow_list: continue
            if lang not in lang2words2: continue
            word2 = random.choice(lang2words2[lang])
            if word2 == word:
              word2 = random.choice(lang2words2[lang])
            if word2 == word: continue
            if allow_list and word2 not in allow_list: continue
            aHash = ret_hash.get(lang, {})
            aHash[word] = word2
            ret_hash[lang] = aHash
    return ret_hash

  def create_multilingual_examples(self, en_examples, ner_label, word2ner, block_list=[], cut_off_abs=5, cut_off_per=0.5):
    def has_any(words, aSet):
      for w in words:
        if w in aSet: return True
      return False
    block_list = set(block_list)
    en = self.en
    word2en = self.word2en
    ret_list = []
    ret_hash = dict([(word.replace(" ", "_").replace("-", "_").lower().strip("."), 1) for word in en_examples])
    added = list(ret_hash.keys())
    for i in range(3):
      added2 = []
      for word in added:
        if word not in en: continue
        found = [(w, word2en[w]) for w in en[word]]
        found = [(w, [w2 for w2 in word2en[w] if w2 not in ret_hash], len([w2 for w2 in word2en[w] if w2 not in ret_hash])/len(word2en[w])) for w in en[word] if not  has_any(word2en[w], block_list)]
        found = [a for a in found if len(a[1]) < cut_off_abs  and a[2] < cut_off_per]
        unk = list(itertools.chain(*[a[1] for a in found]))
        if found:
          #print (word, found)
          added2.extend(unk)
          ret_list.extend([a[0] for a in found])
      if not added: 
        break
      else:
        #print (ner_label, Counter(added2))
        added = [a[0] for a in Counter(added2).items () if a[1] > 1]
        #print (ner_label, added)
        if not added:
          break
        for word in added:
          ret_hash[word] = 1
    block_list = list(set(list(block_list) + (ret_list + list(ret_hash.keys()))))
    ret = [(a, ner_label) for a in list(set(ret_list + list(ret_hash.keys())))]                              
    return ret, word2ner + ret, block_list


if __name__ == "__main__":  
  data_dir = shared_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  os.path.pardir, "data"))
  if "-s" in sys.argv:
    shared_dir = sys.argv[sys.argv.index("-s")+1]
  print (data_dir, shared_dir)
  if "-c" in sys.argv:
    builder = OntologyBuilder(shared_dir=shared_dir, data_dir=data_dir)
    rel2 = builder.save_cross_lingual_ontology()
    print(rel2)
    #builder.create_combined_cn_yago()
