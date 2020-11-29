import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')


stemmer = PorterStemmer()


# split a sentence to words in list
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# get the root of each word in list
def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

