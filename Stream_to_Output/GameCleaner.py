import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.parsing import preprocess_string, strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
import spacy
from langdetect import detect
import pickle
import gzip
nltk.download('stopwords')

#function definitions

#strips values out of encoded stream lists
def text_col_cleaner(frame, cols, pattern):
    
    pattern = re.compile(pattern)
    
    for col in cols:
      frame[col] = frame[col].map(lambda x: [re.findall(pattern,val)[0].strip() for val in x], na_action='ignore')
    return frame

#converts specified columns to one-hot
def encode_columns(frame):
    targets = list(frame.columns)
    for t in targets:
        one_hot = pd.get_dummies(frame[t].apply(pd.Series).stack(),prefix=t).groupby(level=0).sum()
        frame = pd.concat([frame,one_hot],axis=1)
    return frame

#custom text processor for tokenizing descriptions by Kuan Chen & Nick Canu
def doc_text_preprocessing(ser):
  nlp=spacy.load("en_core_web_sm", exclude=['parser','ner','textcat'])

  """text processing steps"""
  stop_words=set(stopwords.words('english'))
  stop_words.update(['game','player','players','games', 'also', 
                     'description','publisher'])
  
  single_letter_replace=lambda c: re.sub("\s+\w{1}\s+|\n|-|—",'',c)
  to_lower_func=lambda c: c.lower()

  lemma_text=[preprocess_string(
      ' '.join([token.lemma_ for token in desc]
          ),[remove_stopwords,strip_numeric,strip_punctuation,strip_tags,
             strip_multiple_whitespaces,single_letter_replace,to_lower_func]
             ) for desc in ser.apply(lambda x: nlp(x))]

  tokenize_text=[[word for word in string if word not in stop_words] for string in lemma_text]

  return tokenize_text

#performs english language detection on the descriptions w/langdetect then additionally drops games using non-english characters in the name
def lang_cleanup(frame):
  nlp=spacy.load("en_core_web_sm")
  frame['description']=frame['description'].fillna('no words')
  frame = frame[frame['description']!='no words']
  frame['cleaned_descriptions']=doc_text_preprocessing(frame['description'])

  detected_lang = []
  for word in frame.cleaned_descriptions:
    word=', '.join(word)
    detected_lang.append(detect(word))
  frame['lang'] = detected_lang
  frame = frame[frame['lang']=='en']

  non_eng_title_filter = frame['name'].str.contains('[^\x00-\x7f]', flags=re.IGNORECASE)
  return frame[~non_eng_title_filter]


#column name stripper for creating key values
def column_fixer(frame,targ):
  return [col.replace(targ, "").strip('"') for col in frame.columns if col.startswith(targ)]

#creates key list for defining web app lists & nlp tokens of the same unknown input search
def key_collator(frame):
  nlp=spacy.load("en_core_web_sm")
  fam = column_fixer(frame,'family_')
  gt = column_fixer(frame,'game_type_')
  mec = column_fixer(frame,'mechanic_')
  cat = column_fixer(frame,'category_')

  current_keys = (['cooperative'],gt,mec,cat,fam)

  fam_keys = [nlp(w) for w in fam]
  gt_keys = [nlp(w) for w in gt]
  mec_keys = [nlp(w) for w in mec]
  cat_keys = [nlp(w) for w in cat]

  search_tokens = (gt_keys,mec_keys,cat_keys,fam_keys)

  return current_keys, search_tokens
   

#-----------

#reading in raw file & removing unranked and compilation game items
df = pd.read_json(r'./bgg_GameItem.jl', lines=True)
df['rank'] = df['rank'].fillna(0).astype(int)
df = df[(df['rank']>0) & (df['compilation']!=1)]

#separating and cleaning the one-hot target columns
in_df = text_col_cleaner(frame = df[['game_type','mechanic','category','family']],
                    cols = ['game_type','mechanic','category','family'], 
                    pattern = re.compile("([\S ]+)(?=:)"))

print('Text has been cleaned, now encoding one-hot columns')

#encoding one-hot columns and rejoining to features for output
proc_df = encode_columns(in_df)
step = df[['name','description','cooperative']]
join_df = pd.concat([step,proc_df.drop(['game_type','mechanic','category','family',
    'game_type_Amiga','game_type_Arcade','game_type_Atari ST',
    'game_type_Commodore 64'],axis=1)],axis=1)

print('Columns encoded, now performing english language detection and cleanup')

#english language detection steps & first data save
eng_df = lang_cleanup(join_df)
eng_df = eng_df.loc[:,~eng_df.columns.duplicated()].copy().reset_index(drop=True).fillna(0)

print('Creating vector-only dataframe & saving output')

#vector only data for operations
vector_df = eng_df.copy().drop(['name','description','cleaned_descriptions','lang'],axis=1)

eng_df.to_parquet('game_data.parquet.gzip',compression='gzip')
vector_df.to_parquet('game_vectors.parquet.gzip',compression='gzip')

print('Creating key lists')

#creating key lists - 1. string list of values by feature class for defining input selections & 2. nlp processed list for unknown input search
keys, search_toks = key_collator(vector_df)

with gzip.open("current_keys.gz", "wb") as f:
    pickle.dump(keys, f)
f.close()

with gzip.open("key_search_tokens.gz", "wb") as f:
    pickle.dump(search_toks, f)
f.close()

print('File creation is complete')