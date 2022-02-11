import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import sys
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from herokutokenizer import Tokenizer


def load_data(database_filepath):
    """
    Load data from database and separe it in features, target and categories

    Parameters:
    database_filepath (str): the path of sqlite database file.
    
    Returns:
    X (panda dataframe): features
    y (panda dataframe): target
    y.columns (panda series): target categories names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df =  pd.read_sql_query ( "SELECT * FROM DisasterResponse", engine)
    
    X = df["message"]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, y, y.columns


def build_model():
    """
    Build and return a model.


    Returns:
    pipeline (sklearn pipeline): A pipeline to process text data
    """
    pipeline = Pipeline([   
        ('tokenizer',Tokenizer()), 
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),

        ('clf', MultiOutputClassifier(
            AdaBoostClassifier())) # the optmus values is 200, but it made models to
                                                      # big for heroku  
    ])
    
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2))
        #'vect__max_df': (0.5, 1.0) # 0.75
        #'vect__max_features': (None, 5000, 10000)
        #'tfidf__use_idf': (True, False),
       'clf__estimator__n_estimators' : [50,60,70,80]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

   
    return  cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evalueate each model target categorie.

    Parameters:
    model (sklearn Pipeline): the evaluated pipeline
    X_test (panda dataframe): test feature data
    Y_test (panda dataframe): test target data
    category_names (pandas Series): target categories name
    Returns:
    """
    y_pred = model.predict(X_test)

    print("Model Score: {}".format(model.score(X_test, Y_test)))
    #print("Model Best Params: {}".format(model.best_params_))
    for column in range(y_pred.shape[1]):
        print("Column: ",category_names[column])

        print(classification_report(Y_test[category_names[column]], y_pred[:,column], zero_division=0))
        print("----------------------------------------------------------*")


def save_model(model, model_filepath):
    """
    Dump model to a pickle file.

    Parameters:
    model (sklearn Pipeline): model to be stored
    model_filepath (str): the path of generated pickle file path
    
    
    Returns:
    None
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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