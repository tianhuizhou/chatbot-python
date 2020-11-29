import nltk
from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')


stemmer = PorterStemmer()


# split a sentence to words in list
def tokenize(sentence):
    return nltk.word_tokenize(sentence)


# get the root of each word in list
def stem(word):
    return stemmer.stem(word.lower())


words = ['organize', 'organization', 'organizing']
stemmed_word = []

for w in words:
    stemmed_word.append(stem(w))
print(stemmed_word)
