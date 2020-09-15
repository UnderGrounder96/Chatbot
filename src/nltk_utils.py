#!/usr/bin/env python3

import nltk
import secrets
import numpy as np

from nltk.tokenize import TweetTokenizer
from nltk.stem.snowball import SnowballStemmer

# nltk.download('all')


def tokenize(sentence):
    """
    split sentence into array of tokens.
    a token can be a word, a punctuation character, a symbol or number
    """
    # FIXME: improve this in-place replacement for contractions
    sentence = sentence.lower().split()
    sentence = [contractions_dict.get(word, word) for word in sentence]

    return TweetTokenizer().tokenize(' '.join(sentence))


def stem(word):
    """
    Stemmers remove morphological affixes from words, leaving only the word stem.
    examples:
      words = ['organize', 'organizes', 'organizing']
      [stem(w) for w in words] -> ['organ', 'organ', 'organ']
    """
    return SnowballStemmer("porter").stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    """
    returns array:
      1 for each known word that exists in the sentence, 0 otherwise
    example:
      tokenized_sentence = ['how', 'are', 'you']
      all_words = ['are', 'bye', 'how', 'I', 'you']
      bag_words = [   1,     0,      1,   0,     1]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]

    # initialize bag with 0 for each word
    # TODO: check if we can just use list_comprehension, instead of np.float32
    bag_words = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in sentence_words:
            bag_words[idx] = 1.0

    return bag_words


# TODO: Get the context from contraction and return accordingly
def get_contraction(contraction):
    """
    returns a random response from splitted contraction
    """
    return secrets.choice(contraction.split('/'))


contractions_dict = {
    "ain't": get_contraction("am not / are not / is not / has not / have not"),
    "aren't": "are not",
    "can't": "can not",
    "can't've": "can not have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": get_contraction("he had / he would"),
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": get_contraction("he shall have / he will have"),
    "he's": get_contraction("he has / he is"),
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": get_contraction("how has / how is / how does"),
    "i'd": get_contraction("I had / I would"),
    "i'd've": "I would have",
    "i'll": get_contraction("I shall / I will"),
    "i'll've": get_contraction("I shall have / I will have"),
    "i'm": "I am",
    "i've": "I have",
    "isn't": "is not",
    "it'd": get_contraction("it had / it would"),
    "it'd've": "it would have",
    "it'll": get_contraction("it shall / it will"),
    "it'll've": get_contraction("it shall have / it will have"),
    "it's": get_contraction("it has / it is"),
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": get_contraction("she had / she would"),
    "she'd've": get_contraction("she would have"),
    "she'll": get_contraction("she shall / she will"),
    "she'll've": get_contraction("she shall have / she will have"),
    "she's": get_contraction("she has / she is"),
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": get_contraction("so as / so is"),
    "that'd": get_contraction("that would / that had"),
    "that'd've": "that would have",
    "that's": get_contraction("that has / that is"),
    "there'd": get_contraction("there had / there would"),
    "there'd've": "there would have",
    "there's": get_contraction("there has / there is"),
    "they'd": get_contraction("they had / they would"),
    "they'd've": "they would have",
    "they'll": get_contraction("they shall / they will"),
    "they'll've": get_contraction("they shall have / they will have"),
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": get_contraction("we had / we would"),
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": get_contraction("what shall / what will"),
    "what'll've": get_contraction("what shall have / what will have"),
    "what're": "what are",
    "what's": get_contraction("what has / what is"),
    "what've": "what have",
    "when's": get_contraction("when has / when is"),
    "when've": "when have",
    "where'd": "where did",
    "where's": get_contraction("where has / where is"),
    "where've": "where have",
    "who'll": get_contraction("who shall / who will"),
    "who'll've": get_contraction("who shall have / who will have"),
    "who's": get_contraction("who has / who is"),
    "who've": "who have",
    "why's": get_contraction("why has / why is"),
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": get_contraction("you had / you would"),
    "you'd've": "you would have",
    "you'll": get_contraction("you shall / you will"),
    "you'll've": get_contraction("you shall have / you will have"),
    "you're": "you are",
    "you've": "you have"
}
