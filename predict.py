import treetaggerwrapper as ttpw
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import word_tokenize

import pandas as pd
import numpy as np
import string
from apriori import load_model,Apriori
# import glob
# import pickle
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.svm import SVC
# import ast
# from scipy import sparse
# from collections import Counter
# from itertools import chain, combinations
# from typing import Tuple,List
# from math import comb

pkl_dirs = "./"
negation_word = ['no','not','none','nobody','nothing','neither','nowhere', 
                'never', 'hardly', 'scarcely', 'barely']
word_df = pd.read_excel("./kansei words.xlsx"
                        , index_col=0)
excluded_pos_tags = []
tagger = ttpw.TreeTagger(TAGLANG='en', TAGDIR='treetagger/')
tokenizer = PunktSentenceTokenizer()

def wordtoken(sents):
  hasil = [word_tokenize(sent) for sent in sents]
  return [list(filter(lambda x: x not in string.punctuation and len(x)>1 and (x.isalpha() or "'" in x),katas)) for katas in hasil]

def poslemmatag(katas):
  global tagger
  data = []
  for kata in katas:
    tags = tagger.tag_text(kata)
    data.append(ttpw.make_tags(tags,exclude_nottags=True))
  return data

def extract_from_tags(tags):
  total_word, total_pos, total_lemma, total_lemma_pos = [],[],[],[]
  for text in tags:
    word,pos,lemma,lemma_pos = [],[],[],[]
    for sent in text:
      temp_word,temp_pos,temp_lemma,temp_lemma_pos = [],[],[],[]
      for w in sent:
        if w.word.isalpha() and len(w.word)>1 and (w.pos not in excluded_pos_tags or w.word in ['no']):
          temp_word.append(w.word)
          temp_pos.append(w.pos)
          temp_lemma.append(w.lemma)
          temp_lemma_pos.append({'lemma':w.lemma,'pos':w.pos})
      word.append(temp_word)
      pos.append(temp_pos)
      lemma.append(temp_lemma)
      lemma_pos.append(temp_lemma_pos)    
    total_word.append(word)
    total_pos.append(pos)
    total_lemma.append(lemma)
    total_lemma_pos.append(lemma_pos)
  return total_word,total_lemma,total_pos,total_lemma_pos

def findaff_neg(sents):
  hasil_aff = []
  hasil_attr = []
  for sent in sents:
    temp_aff = []
    temp_attr = []
    temp_idx_neg = -1
    for i,kata in enumerate(sent):
      if kata['lemma'] in negation_word:
        temp_idx_neg = i
      if kata['lemma'] in word_df['lemma'].to_list():
        attr,element = word_df.loc[word_df['lemma']==kata['lemma'],['attribute','element']].to_records(index=False)[0]
        if temp_idx_neg != -1 and temp_idx_neg < i:
          temp_aff.append(f"{sent[temp_idx_neg]['lemma']} + {kata['lemma']}")
          attr = word_df.loc[(word_df['element']==element) & (word_df['attribute']!=attr),'attribute'].unique()[0] #Reverse Attribute
          temp_idx_neg = -1
        else:
          temp_aff.append(f"{kata['lemma']}")
        temp_attr.append(attr)

    hasil_aff.append(temp_aff)
    hasil_attr.append(temp_attr)
  return {'aff_word':hasil_aff,'aff_attr':hasil_attr}

def heuristic_combination(posterior_rule,posterior_model):
  prediction_rule = list(posterior_rule.keys())[np.argmax(posterior_rule.values())]
  if prediction_rule!= 'Unknown' and max(posterior_rule.values()) == 1:
    return posterior_rule
  else:
    total_score = np.add( np.array(list(posterior_rule.values())), 0.5* np.array(list(posterior_model.values())))
    # print(total_score)
    return {label:total_score[i] for i,label in enumerate(posterior_model.keys())}

def combine_all(data):
  res = []
  for i in range(len(data['aff_word'])):
    if len(data['aff_word'][i]) == 0:
      continue
    res.append(list(set(data['aff_word'][i])) + list(set(data['aff_attr'][i])))
  return res

def get_review_prediction(review_text):
  pred_res = {
    'apriori':{},
    'svm':{},
    'heuristic':{}
  }
  for aspect_name in ['aesthetics','interfaces','mechanics','usability']:
    apriori_model = Apriori(f"{pkl_dirs}/Apriori Model/{aspect_name}_apriori_tuned.pkl")
    bow_model = load_model(f"{pkl_dirs}/BoW Model/bow-{aspect_name}-smote.pkl")
    svm_model = load_model(f"{pkl_dirs}/SVM Model/{aspect_name}_bow_smo-svm.pkl")
    
    review_sents = tokenizer.tokenize_sents(review_text)
    tags = [poslemmatag(x) for x in review_sents]
    _,_,_,lemma_pos = extract_from_tags(tags)
    rule_df = pd.DataFrame.from_dict([findaff_neg(x) for x in lemma_pos])
    # [list(findaff_neg(x).values()) for x in lemma_pos]
    apriori_res = rule_df['aff_word'].apply(lambda x: apriori_model.posterior_rules(np.unique(np.hstack(x)), 
                                                                            svm_model.classes_)).to_list()
    bow_text = bow_model.transform(review_text).toarray()
    svm_res = [{svm_model.classes_[i]:score for i,score in enumerate(row)} for row in svm_model.decision_function(bow_text)]

    heuristic_res = [heuristic_combination(posterior_rule, svm_res[i]) for i,posterior_rule in enumerate(apriori_res)]

    pred_res['apriori'][aspect_name] = apriori_res
    pred_res['svm'][aspect_name] = svm_res
    pred_res['heuristic'][aspect_name] = heuristic_res
  return pred_res