#!/usr/bin/env python3

import json
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from chat_dataset import ChatDataset
from nltk_utils import bag_of_words, tokenize, stem

# create word lists
all_words, tags, xy = ([], [], [])

with open("intents.json", "r") as f:

  # loop through each sentence in our intents patterns
  for intent in json.load(f)['intents']:
      tag = intent['tag']

      # add to tags list
      tags.append(tag)

      for pattern in intent['patterns']:
          # tokenize (and lower) each word in the sentence
          words = tokenize(pattern)

          # add to our words list
          all_words.extend(words)

          # add to xy pair
          xy.append((words, tag))

# stem each word
ignore_symbols = ['.', ',', '-', '/', '!', '?', '$']
all_words = [stem(word) for word in all_words if word not in ignore_symbols]

# remove duplicates and sort
all_words, tags = sorted(set(all_words)), sorted(set(tags))

print(f"{len(xy)}: patterns")
print(f"{len(tags)} - tags: {tags}")
print(f"{len(all_words)} - stemmed words: {all_words}")

# create training data
X_train , y_train = ([], [])

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    X_train.append(bag_of_words(pattern_sentence, all_words))

    # y: PyTorch CrossEntropyLoss needs only class labels, NOT one-hot
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)


# Hyper-parameters
batch_size = 8


dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
