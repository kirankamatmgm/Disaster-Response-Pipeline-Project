import sys
import pandas as pd
import numpy as np
import pickle

from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """load data from the sqlite database
    input:
        database_filepath: File path where sql database was saved.
    output:
        X: Training features.
        Y: Training target.
        category_names: Categorical name for labeling.
    """
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('messages',engine)
    X = df['message']
    Y = df.iloc[:,3:]
    category_names = list(df.columns[3:])
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and lemmatize each word in a given text
    Input: Text data
    Output: List of clean tokens
    
    """
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # model pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier((AdaBoostClassifier())) )
    ])
    
#     # hyper-parameter grid
#     parameters = {
#     'tfidf__norm':['l2','l1'],
#     'clf__estimator__learning_rate' :[0.1, 0.5, 1],
#     'clf__estimator__n_estimators' : [50, 60, 70],
#     }
#     #create grid search object
#     cv = GridSearchCV(pipeline, param_grid=parameters,verbose=5,n_jobs=2)
#     return cv
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Shows model's performance on test data
    input:
    model: trained model
    X_test: Test features
    Y_test: Test targets
    category_names: Target labels
    """
    y_pred = model.predict(X_test)
    
    # print accuracy score
    print('Accuracy: {}'.format(np.mean(Y_test.values == y_pred)))
    
    for i in range(36):
        print("Precision, Recall, F1 Score for {}".format(Y_test.columns[i]))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i],target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves the model to a Python pickle file    
    input:
    model: Trained model
    model_filepath: Filepath to save the model
    """

    # save model to pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


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