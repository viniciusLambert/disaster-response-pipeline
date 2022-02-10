import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def tokenize(text):
       

            norm_text = text.lower()
            tokens = nltk.tokenize.word_tokenize(norm_text)
            lemmatizer = nltk.stem.WordNetLemmatizer()

            clean_tokens = []
            for tok in tokens:
                clean_tok = lemmatizer.lemmatize(tok)
                clean_tokens.append(clean_tok)

            return clean_tokens
        return pd.Series(X).apply(tokenize).values