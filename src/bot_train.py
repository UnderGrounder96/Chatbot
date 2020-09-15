#!/usr/bin/env python3

import json
import torch
import numpy as np
import torch.nn as nn

from model import NeuralNet
from chat_dataset import ChatDataset

from torch.utils.data import DataLoader
from nltk_utils import bag_of_words, tokenize, stem

print("Processing training data...")

# create word lists
all_words, tags, xy = ([], [], [])

with open("intents.json", 'r') as json_data:

  # loop through each sentence in our intents patterns
  for intent in json.load(json_data)["intents"]:
      tag = intent["tag"]

      # add to tags list
      tags.append(tag)

      for pattern in intent["patterns"]:
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
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
# print(input_size, output_size)


dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)

        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


print(f"final loss: {loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"Training complete! File saved as {FILE}")
print("You can now run main.py")
