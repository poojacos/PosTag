# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import nltk
import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

nltk.download('brown')
nltk.download('universal_tagset')

corpus = nltk.corpus.brown.tagged_sents(tagset='universal')

print(corpus[0])
print(type(len(corpus)))

# first two tagged sentences in the corpus
pp = pprint.PrettyPrinter()
pp.pprint(corpus[:2])

# create a corpus with start and end tags
corpus_new = []
for sent in corpus:
  corpus_new.append([('<s>', 'START')] + sent + [('</s>', 'END')])

corpus_new[:1]
corpus = corpus_new

tags = []
words = []
pos_dict = {}
words_dict = {}

for sent in corpus:
  for tup in sent:
    word, pos = tup
    if word in words_dict:
      words_dict[word] += 1
    else: 
      words_dict[word] = 1
    
    if pos in pos_dict:
      pos_dict[pos] += 1
    else:
      pos_dict[pos] = 1

tags = pos_dict.keys()
words = words_dict.keys()

# splitting into train and test sets
train_set, test_set = train_test_split(corpus,train_size = 0.95, test_size = 0.05, random_state = 123)

# putting those into their own lists
train_words = [tup for sent in train_set for tup in sent]
test_words = [tup[0] for sent in test_set for tup in sent]

# print out first five train set words
pp.pprint(train_words[:5])
pp.pprint(test_words[:5])

n, m = len(train_words), len(tags)
emissions = {} 
for tup in train_words:
  word_val, tag_val = tup
  if word_val in emissions:
    if tag_val in emissions[word_val]:
      emissions[word_val][tag_val] += 1
    else:
      emissions[word_val][tag_val] = 1
  else:
    emissions[word_val] = {}
    emissions[word_val][tag_val] = 1

emissions_df = pd.DataFrame(emissions).transpose()

transitions = {}
prev_tag = None
for tup in train_words:
  _, tag_val = tup
  if tag_val in transitions:
    if prev_tag in transitions[tag_val]:
      transitions[tag_val][prev_tag] += 1
    else:
      transitions[tag_val][prev_tag] = 1
  else:
    transitions[tag_val] = {}
    transitions[tag_val][prev_tag] = 1
  prev_tag = tag_val

transitions["START"]["END"] = 0
transitions_df = pd.DataFrame(transitions) #first index is the col i.e. next tag

def localMaximizer(words, train_on = train_words):
  tags = list(set([tup[1] for tup in train_on])) 
  states = []

  # go through the sequence of words
  for key, word in enumerate(words):
    # initialize matrices
    prob = []
    for tag in tags:
      if key == 0:
        transition = transitions_df.loc['END', tag]
      else:
        transition = transitions_df.loc[states[-1], tag]
        
      if emissions.get(word, {}).get(tag):
        emission = emissions[words[key]][tag]
      else:
        emission = 0.0

      # calculate emission probabilities * transition probabilities for all tags
      prob.append(transition * emission)

    # find the maximum probability state, backtrace
    max_state = tags[prob.index(max(prob))]
    states.append(max_state)
  
  return list(zip(words, states))

tests = test_set

# words with tags
test_with_tags = [tup for sent in tests for tup in sent]
# words without their tags
test_without_tags = [tup[0] for sent in tests for tup in sent]

find_sequence = localMaximizer(test_without_tags)


correct = []
for i, j in zip(find_sequence, test_with_tags):
  if i == j:
    correct.append(i)

accuracy = len(correct)/len(find_sequence)

# Implementing pseudo code here - https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode

class Viterbi():
  def __init__(self, Y):
    self.O = list(set([w_t[0] for w_t in train_words]))  # Observation space
    self.S = list(set([w_t[1] for w_t in train_words]))  # State space
    self.Y = Y  # Sequence of observations
    self.A = transitions_df  # Transition matrix
    self.B = emissions_df  # Emission matrix
    self.N = len(self.O)
    self.K = len(self.S)
    self.T = len(Y)
    self.T1 = [[0] * self.T for i in range(self.K)]
    self.T2 = [[0] * self.T for i in range(self.K)]
    self.X = [None] * self.T # Predicted tags

  def decode(self):
    """Run the algorithm"""
    delta = 0.0001
    print('T=',self.T, " N=", self.N, "K=",self.K)

    for i in range(self.K):
      if self.Y[1] in self.O:

        if pd.isnull(self.A.loc['START'][self.S[i]]) or pd.isnull(self.B.loc[self.Y[1]][self.S[i]]):
          self.T1[i][1] = float('-inf') #log of this would be a large negative number
        else:
          self.T1[i][1] = np.log(self.A.loc['START'][self.S[i]] + delta) + np.log(self.B.loc[self.Y[1]][self.S[i]] + delta)

    self.forward()
    self.backward()

    return self.X
  
  def forward(self):
    """Forward step"""
    delta = 0.0001
    for j in range(2,self.T):
      # ### DEBUG
      # if j%10000==0:
      #   print('\n')
      # elif j%1000==0:
      #   print(j, end=' ')
      # ### DEBUG

      for i in range(self.K):
        max_val, max_idx = float('-inf'), 0
        for k in range(self.K):

          Tkj_1 = self.T1[k][j-1]

          # Unseen word
          if not self.Y[j] in self.O:
            if pd.isnull(self.A.loc[self.S[k]][self.S[i]]):
              val = float('-inf') # we do not know anything about this word

            else:
              val = Tkj_1 + np.log(self.A.loc[self.S[k]][self.S[i]] + delta) # what is the most likely previous tag 

          # Dealing with nan
          elif pd.isnull(self.B.loc[self.Y[j]][self.S[i]]) or pd.isnull(self.A.loc[self.S[k]][self.S[i]]):
            val = Tkj_1
          
          # Viterbi Update
          else:
            val = Tkj_1 + np.log(self.A.loc[self.S[k]][self.S[i]] + delta) + np.log(self.B.loc[self.Y[j]][self.S[i]] + delta)
          
          if val>max_val:
            max_val, max_idx = val, k 
        self.T1[i][j] = max_val
        self.T2[i][j] = max_idx

        
  def backward(self):
    """Backward step"""
    max_end, zj = float('-inf'), 0
    for i in range(self.K):
      tmp = self.T1[i][self.T-1] 
      if tmp>max_end:
        max_end, zj = tmp, i
    self.X[self.T-1] = self.S[zj]

    for j in range(self.T-1, 0, -1):
      z = self.T2[zj][j]
      self.X[j-1] = self.S[z]
      zj = z

tests = test_set

# words with tags
test_with_tags = [tup for sent in tests for tup in sent]
# words without their tags
test_without_tags = [tup[0] for sent in tests for tup in sent]

# running your Viterbi function
decoder = Viterbi(test_without_tags)
pred = decoder.decode()

result = [(test_without_tags[i], pred[i]) for i in range(0, len(pred))] 

correct = []
for i, j in zip(result, test_with_tags):
  if i == j:
    correct.append(i)
  
accuracy = len(correct)/len(result)
accuracy

count = 0
print('Printing incorrectly tagged words')
for k, (i, j) in enumerate(zip(result, test_with_tags)):
  if i != j:
    count += 1
    print('word= ', test_without_tags[k], ' predicted tag= ',i, ' correct tag= ', j)
  if count>12:
    break
