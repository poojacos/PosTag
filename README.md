# PosTag
Implementing POS Tagging using Viterbi Algorithm and Local Maximizer

### Data
This project uses using the tagged sentences in the Brown corpus.
```
corpus = nltk.corpus.brown.tagged_sents(tagset='universal')
```

The corpus has tagged words  like so-

    [('The', 'DET'),
      ('Fulton', 'NOUN'),
      ('County', 'NOUN'),
      ('Grand', 'ADJ'),
      ('Jury', 'NOUN'),
      ('said', 'VERB'),
      ('Friday', 'NOUN'),
      ('an', 'DET'),
      ('investigation', 'NOUN'),
      ('of', 'ADP'),
      ("Atlanta's", 'NOUN'),
      ('recent', 'ADJ'),
      ('primary', 'NOUN'),
      ('election', 'NOUN'),
      ('produced', 'VERB'),
      ('``', '.'),
      ('no', 'DET'),
      ('evidence', 'NOUN'),
      ("''", '.'),
      ('that', 'ADP'),
      ('any', 'DET'),
      ('irregularities', 'NOUN'),
      ('took', 'VERB'),
      ('place', 'NOUN'),
      ('.', '.')]

### Hidden Makrov Model
Hidden Markov Model is a probabilistic sequence model that, given a sequence of words, computes a probability distribution over a sequence of POS tags. Given a trained HMM, the code computes the likelihood of a particular sequence using a forward algorithm.

Use the Hidden Markov Model to calculate the emission probabilities and transition probabilities.

### Viterbi Algorithm
The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and hidden Markov models (HMM).

The algorithm implements [this](https://en.wikipedia.org/wiki/Viterbi_algorithm#Pseudocode "this") pseudocode.

### Local Maximizer
LocalMaximizer only considers forward probabilities and thus leads to only a local optimum, not to a global optimum. This is NOT what the Viterbi algorithm does -- it instead stores a global optimum using (1) a partial, best forward probability, and (2) a backpointer to what the best predecessor is.


