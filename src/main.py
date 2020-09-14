#!/usr/bin/env python3

import json
import torch
import secrets
import os.path as path

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

if not path.isfile("./data.pth"):
    raise("Please run bot_train.py to produce data.pth")

with open("intents.json", 'r') as json_data:
    intents = json.load(json_data)

data = torch.load("data.pth")

tags = data["tags"]
all_words = data["all_words"]
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
model_state = data["model_state"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")

while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit" or sentence == 'q':
        print("I hope to see you again. Take care!")
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])

    output = model(torch.from_numpy(X).to(device))
    _, predicted = torch.max(output, dim=1)

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    tag = tags[predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {secrets.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")