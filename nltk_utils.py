#!/usr/bin/env python3

import nltk

from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')

def tokenize(sentence):
  """
  split sentence into array of tokens.
  a token can be a word, a punctuation character, a symbol or number
  """
  return nltk.word_tokenize(sentence)

def stem(word):
  """
  stemming is finding the root form of the word
  examples:
  words = ["organize", "organizes", "organizing"]
  [stem(w) for w in words] -> ["organ", "organ", "organ"]
  """
  return PorterStemmer().stem(word.lower())

def bag_of_words(tokenized_sentence, words):
  pass


# user_input = "what's on the menu?"
# print(tokenize(user_input))

# words = ["organize", "organizes", "organizing"]
# print([stem(w) for w in words])