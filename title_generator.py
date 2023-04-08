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

#Custom text tokenizer from https://github.com/canunj/deconstructing_games by N Canu & K Chen
def doc_text_preprocessing(ser):
    nlp=spacy.load("en_core_web_sm", exclude=['parser','ner','textcat'])

    """text processing steps"""
    import re
    stop_words=set(stopwords.words('english'))
        
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
        
        self.title_iter = -1
        self.out_titles = None
        self.best_title = None
        self.description = None


    def candidate_generator(self, description):
        text =  "headline: " + description

        encoding = self.tokenizer.encode_plus(text, return_tensors = "pt")
        input_ids = encoding["input_ids"].to(self.device)
        attention_masks = encoding["attention_mask"].to(self.device)

        candidates = []

        beam_outputs = self.model.generate(
            input_ids = input_ids,
            attention_mask = attention_masks,
            max_length = 64,
            num_beams = 16,
            num_beam_groups=4,
            num_return_sequences=8,
            diversity_penalty=.1,
            repetition_penalty=.9,
            early_stopping = True)

        for result in beam_outputs:
            res = self.tokenizer.decode(result).replace('<pad> ','').replace('</s>','').replace('<pad>','')
            candidates.append(res)
        
        return candidates, description
    
    def candidate_score(self,candidates,ex_check=None):
        import random
        from operator import itemgetter
        
        if ex_check != None:
            pat = re.compile("((?:" + "|".join(map(re.escape, candidates[0]+[cand.upper() for cand in candidates[0]])) + "|" + "|".join(ex_check) +"))")
            desc = re.sub(pat, "__", candidates[1])
        else:
            pat = re.compile("((?:" + "|".join(map(re.escape, candidates[0]+[cand.upper() for cand in candidates[0]])) + "))")
            desc = re.sub(pat, "__", candidates[1])

        
        if re.search(re.compile(re.escape("__")), desc):
            reg = re.compile("("+"|".join(ex_check) + ")")
            hold = candidates[0]
            gen_desc = re.sub(re.compile(re.escape("__")),"",desc)
            candidates = self.candidate_generator(gen_desc)
            next = [cand for cand in candidates[0]+hold if not reg.match(cand)]
            candidates = (next, desc)
        
        #backup load function, will refactor
        nlp=spacy.load("en_core_web_md")

        #check for existing games and duplicates
        #transform function from https://stackoverflow.com/questions/42165779/python-how-to-remove-duplicate-valuescase-insensitive-from-a-list-with-same-o
        def transform(L):
            S = set(L)
            return [item.title() for item in L if item.lower() not in S and not S.add(item.lower())]


        clean_cand_step = list(set([game[0] for game in list(zip(candidates[0],[len(self.game_df[self.game_df.name.isin([x])]) for x in candidates[0]])) if game[1]==0]))
        clean_cand_step = transform(clean_cand_step)

        clean_cand_step = [re.sub(re.compile("(?<=[ ])And(?=[ ])"),'and',
                                  re.sub(re.compile("(?<=\S) (([(]|\b)[Ss]econd [Ee]dition([)]|\b)|[Ss]econd [Ee]dition|2[Nn][Dd] [Ee]dition|([(]|\b)[Tt]hird [Ee]dition([)]|\b)|3[Rr][Dd] [Ee]dition)"),"",
                                  re.sub(re.compile("(?<=[a-z])'S"),"'s",
                                  re.sub(re.compile("(?<=[ ])Of(?=[ ])"),"of",x)))) 
                                  for x in clean_cand_step]

        
        clean_cand = []
        for cand in clean_cand_step:
            try: 
                inter = cand.split(":")
                if inter[0].lower()==inter[1].lower():
                    clean_cand.append(inter[0])
                else:
                    clean_cand.append(cand)
            except:
                clean_cand.append(cand)

        #text processing
        token_cand = doc_text_preprocessing(pd.Series(clean_cand))
        token_art = doc_text_preprocessing(pd.Series([candidates[1]]))
        sim = [nlp(title) for title in [" ".join(title) for title in token_cand]]
        doc = nlp(" ".join(token_art[0]))

        #scores cosine similarity between generated titles and body text, if the word is unknown (i.e. generator knows it but spacy doesn't)
        #it assigns a random probability to populate

        scores = [x if x !=0 else random.uniform(.3, .7) for x in [tok.similarity(doc) for tok in sim]]
        
        out_titles = sorted(list(zip(clean_cand,scores)),key=itemgetter(1),reverse=True)
   
        pat = re.compile("(?<=[!.?])(?=[^\s])")
        pat2 = re.compile("([Ff]rom the [Pp]ublisher[: ]|[Ff]rom the [Dd]esigner[: ]|[Gg]ame [Dd]escription)")
        pat3 = re.compile(": [Tt]he [Gg]ame: [Tt]he [Gg]ame|: [Tt]he [Gg]ame")
        pat4 = re.compile("[Tt]he __")
        pat5 = re.compile("__ [Gg]ame")
        pat6 = re.compile("[Tt]he [Gg]ame [Oo]f __")
        
        desc = re.sub(pat," ",candidates[1])   
        desc = re.sub(pat2,"",desc)
        desc = re.sub(pat3,"",desc)
        desc = re.sub(pat4,"__",desc)
        desc = re.sub(pat5,"__",desc)
        desc = re.sub(pat6,"__",desc)

        return {'text':desc,'titles':out_titles}
