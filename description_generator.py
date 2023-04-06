import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import gensim
from gensim.test.utils import get_tmpfile
from gensim.parsing import preprocess_string, strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
import spacy, en_core_web_sm
from spacy.tokens import DocBin
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration,T5Tokenizer
import urllib.request
import pickle
import gzip
import io
import sklearn
import openai
from operator import itemgetter
import re
import Model_Constants as mc
#user input manager class
class input_manager:
  
  #initialize key dictionary from vector data frame and set community top N
  def __init__(self,key_df, slim_df, search_tokens, top_n=10):
        self.key_df = key_df
        self.slim_df = slim_df
        self.search_tokens = search_tokens
        self.key = dict(zip(list(key_df.columns),np.zeros(len(key_df.columns))))
        self.top_n = top_n

  #translate input text to vector
  def set_input(self,input_cats):
    from operator import itemgetter
    import re
    #need setup to apply correct group tag to values
    nlp=spacy.load("en_core_web_md")
    #separate known/unknown features
    k_flags = [cat for cat in input_cats if cat in list(self.key.keys())]
    unk_flags = [cat for cat in input_cats if cat not in list(self.key.keys())]
    
    #process within feature class similarity for each unknown input
    if len(unk_flags)>0:
      
      outs = []
      for word in unk_flags:
        if re.match(r"game_type_",word):
          tok = nlp(word.split("_")[-1])
          mtch = max([(key,key.similarity(tok)) for key in self.search_tokens[0]],key=itemgetter(1))
          #if no known match is found (model doesn't recognize input word), we're going to discard - other solutions performance prohibitive
          if mtch[1]>0:
            outs.append("game_type_"+mtch[0])
        elif re.match(r"mechanic_",word):
          tok = nlp(word.split("_")[-1])
          mtch = max([(key,key.similarity(tok)) for key in self.search_tokens[1]],key=itemgetter(1))
          if mtch[1]>0:
            outs.append("mechanic_"+mtch[0])
        elif re.match(r"category_",word):
          tok = nlp(word.split("_")[-1])
          mtch=max([(key,key.similarity(tok)) for key in self.search_tokens[2]],key=itemgetter(1))
          if mtch[1]>0:
            outs.append("category_"+mtch[0])
        elif re.match(r"family_",word):
          tok = nlp(word.split("_")[-1])
          mtch=max([(key,key.similarity(tok)) for key in self.search_tokens[3]],key=itemgetter(1))
          if mtch[1]>0:
            outs.append("family_"+str(mtch[0]))
      
      #if unks are processed, rejoin nearest match to known.
      k_flags = list(set(k_flags+outs))
    
    #preserve global key and ouput copy w/input keys activated to 1
    d = self.key.copy()
    for cat in k_flags:
      d[cat] = 1.0
    return d

  def input_parser(self,in_vec):
    #extracting keys from processed vector
    ks = [k for k,v in in_vec.items() if v == 1]

    return ks
  
class model_control:
  def __init__(self, apikey):
    self.api_key = apikey
    openai.api_key = self.api_key

    self.prompt = None
    
    self.model = openai.FineTune.retrieve(id=mc.SEND_MODEL()).fine_tuned_model

  def prompt_formatter(self,ks): 
    self.prompt = ". ".join(ks) + "\n\n###\n\n"

  def call_api(self,status=0):
    if status == 0:
      temp=0.5
      pres=0.7
    elif status == 1:
      temp=0.4
      pres=0.6
    elif status == 2:
      temp=0.5
      pres=0.8
    
    answer = openai.Completion.create(
      model=self.model,
      prompt=self.prompt,
      max_tokens=512,
      temperature=temp,
      stop=["END"],
      presence_penalty=pres,
      frequency_penalty=0.5
    )
    return answer['choices'][0]['text']

  def resp_cleanup(self,text):

    if ((text[-1] != "!") & (text[-1] != ".") & (text[-1] != "?")):
      text = " ".join([e+'.' for e in text.split('.')[0:-1] if e])

    sent = re.split(r'([.?!:])', text)
    phrases = ["[Dd]esigned by","[Dd]esigner of","[Aa]rt by","[Aa]rtist of","[Pp]ublished","[Pp]ublisher of"]

    pat = re.compile("(?:" + "|".join(phrases) + ")")
    fix = re.compile("(?<=[.!?])[.!?]")

    text = re.sub(fix,'',''.join([s for s in sent if pat.search(s) == None]))
    
    return text
