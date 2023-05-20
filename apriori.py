import pickle
from itertools import chain, combinations
import pandas as pd

def load_model(art_dir):
  model_dir = art_dir
  with open(model_dir,'rb') as f:
    model = pickle.load(f)
  return model

class Apriori():
  def __init__(self, transaction=None, min_support=None, min_confidence=0, min_lift=0, max_length=None, class_sup=None, 
               model_dir=None):
    if model_dir:
      self.__dict__ = load_model(model_dir)
    else:
      self.transaction = transaction
      self.min_support = min_support
      self.min_confidence = min_confidence
      self.min_lift = min_lift
      self.max_length = max_length
      self.class_sup = class_sup

  def fit(self, data):
    self.transaction = data
    return self

  def set_params(self, min_support=None, min_confidence=0, min_lift=0, max_length=None):
    self.min_support = min_support
    self.min_confidence = min_confidence
    self.min_lift = min_lift
    self.max_length = max_length
    return self

  def powerset(self,iterable):
    s = list(iterable)
    res = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return [list(el) for el in res]
  
  def sumset(self, items):
    res = []
    for trans in self.transaction:
      temp = [False]*len(items)
      for i,elem in enumerate(items):
        for word in trans:
          temp[i] = temp[i] or (elem in word)
          if temp[i]:
            break
      p = all(temp)
      res.append(p)
    return sum(res)
  
  def support(self, items):
    return self.sumset(items)/len(self.transaction)
  
  def confidence(self, antecedent, consequent):
    items = antecedent + consequent
    res = self.support(items)/self.support(antecedent) if self.support(antecedent) else 0
    return res
  
  def lift(self, antecedent, consequent):
    items = antecedent + consequent
    res = self.confidence(antecedent, consequent)/self.support(consequent) if self.support(consequent) else 0
    return res
  
  def count_rule(self, antecedent, consequent):
    items = antecedent + consequent
    return {'antecedent':antecedent, 'consequent':consequent, 
            'support':self.support(items), 'confidence': self.confidence(antecedent, consequent),
            'lift':self.lift(antecedent,consequent)}
  
  def print_rule(self, antecedent, consequent):
    stats = self.count_rule(antecedent, consequent)
    items = stats['antecedent'] + stats['consequent']
    print(f'''{items}, Support={stats['support']:.5f}
{stats['antecedent']} --> {stats['consequent']}, Confidence={stats['confidence']:.5f} & Lift={stats['lift']:.5f} 
          ''')
  
  def generate_rules(self, antecedents, consequents):
    rules = {}
    for consequent in consequents:
      consequent = [consequent] if type(consequent)==str else consequent
      for antecedent in self.powerset(antecedents):
        items = antecedent + consequent
        rules[",".join(items)] = []
        if self.min_support != None and self.support(antecedent) < self.min_support:
          continue
        rule_stat = self.count_rule(antecedent, consequent)
        rules[",".join(items)].append(rule_stat)
    return rules
  
  def posterior_rules(self, antecedents, consequents):
    posterior = {}
    for label in consequents:
      min_support = self.min_support if self.min_support != None else 0
      if self.class_sup:
        min_support = min_support * (self.class_sup[label]/sum(self.class_sup.values()) )
      posterior[label] = 0
      for antecedent in antecedents:
        rule = self.count_rule([antecedent],[label])
        # print(rule)
        if rule['support'] >= min_support and rule['confidence'] >= self.min_confidence and rule['lift'] >= self.min_lift:
          posterior[label] += self.confidence([antecedent],[label])
    #Normalize
    total_val = sum(posterior.values()) if sum(posterior.values())!= 0 else 1
    # print(antecedents, total_val)
    for key in posterior.keys():
      posterior[key] /= total_val
    return posterior
  
  def predict(self,antecedents, consequents):
    posterior = self.posterior_rules(antecedents, consequents)
    label = max(posterior, key=posterior.get)
    return 'Unknown' if posterior[label]==0 else label
  
  def generate_rules_df(self, consequents):
    antecedents = []
    for trans in self.transaction:
      for word in trans:
        if word in consequents:
          continue
        antecedents.append(word)
    antecedents = list(set(antecedents))
    rules = []
    for consequent in consequents:
      min_support = self.min_support if self.min_support != None else 0
      if self.class_sup:
        min_support = min_support * (self.class_sup[consequent]/sum(self.class_sup.values()) )
      for antecedent in antecedents:
        rule = self.count_rule([antecedent],[consequent])
        # print(rule)
        if rule['support'] >= min_support and rule['confidence'] >= self.min_confidence and rule['lift'] >= self.min_lift:
          rules.append(rule)
    return pd.DataFrame(rules)