import sys
import nltk
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.metrics import classification_report

from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    norm_text = text.lower()
    tokens = nltk.tokenize.word_tokenize(norm_text)
    lemmatizer = nltk.stem.WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df =  pd.read_sql_query ( "SELECT * FROM DisasterResponse", engine)

    X = df["message"]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, y, y.columns


def build_model():
    pipeline = Pipeline([

        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),

        ('lr_multi', MultiOutputClassifier(
            RandomForestClassifier())
        )
    ])
    
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    y_pred = model.predict(X_test)

    print("Model Score: {}".format(model.score(X_test, Y_test)))

    for column in range(y_pred.shape[1]):
        print("Column: ",category_names[column])
        print(classification_report(Y_test.to_numpy()[:,column], y_pred[:,column]))
        print("----------------------------------------------------------*")


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()