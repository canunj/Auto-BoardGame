import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from gensim.parsing import preprocess_string, strip_tags, strip_numeric, strip_multiple_whitespaces, stem_text, strip_punctuation, remove_stopwords
import spacy, en_core_web_sm
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration,T5Tokenizer

def doc_text_preprocessing(ser):
    nlp=spacy.load("en_core_web_sm", exclude=['parser','ner','textcat'])

    """text processing steps"""
    import re
    stop_words=set(stopwords.words('english'))
    #stop_words.update(['game','player','players','games', 'also', 
    #                   'description','publisher'])
        
    single_letter_replace=lambda c: re.sub("\s+\w{1}\s+|\n|-|—",'',c)
    to_lower_func=lambda c: c.lower()
    lemma_text=[preprocess_string(
        ' '.join([token.lemma_ for token in desc]
            ),[remove_stopwords,strip_numeric,strip_punctuation,strip_tags,
               strip_multiple_whitespaces,single_letter_replace,to_lower_func]
               ) for desc in ser.apply(lambda x: nlp(x))]

    tokenize_text=[[word for word in string if word not in stop_words] for string in lemma_text]

    return tokenize_text

class Title_Generator:

    def __init__(self, path, df):
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.game_df = df

    def candidate_generator(self, description):
        text =  "headline: " + description
        max_len = 256

        encoding = self.tokenizer.encode_plus(text, return_tensors = "pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        self.candidates = []

        beam_outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_masks,
            max_length = max_len,
            num_beams = 6,
            num_beam_groups=2,
            num_return_sequences=4,
            diversity_penalty=.2,
            repetition_penalty=.9,
            early_stopping = True)

        for result in beam_outputs:
            res = self.tokenizer.decode(result).replace('<pad> ','').replace('</s>','').replace('<pad>','')
            self.candidates.append(res)

        self.candidates = list(set([game[0] for game in list(zip(self.candidates,[len(self.game_df[self.game_df.name.isin([x])]) for x in self.candidates])) if game[1]==0]))

        return self.candidates, description
    
    def candidate_score(self):
        import random
        from operator import itemgetter
        nlp=spacy.load("en_core_web_md")

        clean_cand = list(set([game[0] for game in list(zip(self.candidates,[len(self.game_df[self.game_df.name.isin([x])]) for x in self.candidates])) if game[1]==0]))
        token_cand = doc_text_preprocessing(pd.Series(clean_cand))
        token_art = doc_text_preprocessing(pd.Series([self.candidates[1]]))
        sim = [nlp(title) for title in [" ".join(title) for title in token_cand]]
        doc = nlp(" ".join(token_art[0]))

        scores = [x if x !=0 else random.uniform(.3, .7) for x in [tok.similarity(doc) for tok in sim]]
        return max(list(zip(clean_cand,scores)),key=itemgetter(1))[0]