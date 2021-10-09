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
from datasets import load_dataset
from collections import Counter
from itertools import chain
import os
import re
import glob
import math
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import random
import torch
from collections import Counter, OrderedDict
trannum = str.maketrans("0123456789", "1111111111")
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir, os.path.pardir)))

class RoundTripTranslate:  
  """
  A class for translating text from English to another language, and vice versa. Uses M2M100.  
  Can be used to create paraphrases in various languages, and manipulating the text in English and then back to the target langauge
  to provide multi-lingual capabilities to English only text processing.
  """

  def __init__(self, target_lang='fr'):
    """
    Create a translation engine from English to the target_lang, and
    vice versa.  Useful for round trip translation, paraphrase
    generation and manipulation and extraction in English and
    translation back to target_lang.
    """
    self.target_lang = target_lang
    self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    # we assume we are working only in CPU mode
    self.model =  torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
    self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    #todo, check if target_lang in the 100 lang list

  #TODO: We can vary the probabilties and sampling used to generate in order to get multiple paraphrases
  def translate_to_en(self, text):
    self.tokenizer.src_lang = self.target_lang
    encoded = self.tokenizer(text, return_tensors="pt")
    generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id("en"))
    return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

  def translate_from_en(self, text):
    self.tokenizer.src_lang = "en"
    encoded = self.tokenizer(text, return_tensors="pt")
    generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.get_lang_id(self.target_lang))
    return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
     
  def translate(self, sentence, fr='en', to=None):
    if fr == self.target_lang:
      to = 'en'
    if to == self.target_lang:
      fr = 'en'
    if fr == 'en':
      if to == None:
        to = self.target_lang
      if to != self.target_lang:
        raise RuntimeError("one of 'from' or 'to' must be in the target lang")
    if to == 'en':
      if fr == None:
        fr = self.target_lang
      if fr != self.target_lang:
        raise RuntimeError("one of 'from' or 'to' must be in the target lang")
    if fr != "en" and to != "en":
      raise RuntimeError("one of 'from' or 'to' must be in english")
    if fr == to:
      return sentence
    if fr == "en":
      return self.translate_from_en(sentence)[0]
    return self.translate_to_en(sentence)[0]

  def round_trip_translate(self, sentence, lang='en'): 
    if lang not in ('en', self.target_lang):
      raise RuntimeError("lang must be in english or the target_lang")
    if lang == self.target_lang:
      return self.translate(self.translate(sentence, fr=lang, to='en'), fr='en', to=lang)
    return self.translate(self.translate(sentence, fr=lang, to=self.target_lang), fr=self.target_lang, to=lang)

if __name__ == "__main__":  
  if "-t" in sys.argv:
    target_lang = sys.argv[sys.argv.index("-t")+1]    
    sentence = sys.argv[sys.argv.index("-t")+2]    
    rt = RoundTripTranslate(target_lang=target_lang)
    print (rt.translate(sentence, fr="en", to=target_lang))
  elif "-f" in sys.argv:
    target_lang = sys.argv[sys.argv.index("-f")+1]    
    sentence = sys.argv[sys.argv.index("-f")+2]    
    rt = RoundTripTranslate(target_lang=target_lang)
    print (rt.translate(sentence, fr=target_lang, to="en"))

