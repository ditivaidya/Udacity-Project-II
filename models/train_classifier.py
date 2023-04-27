import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    '''
    load_data
    load database from SQL server
    Input:
     database_filepath - SQL database filepath 
    Output:
     df - pandas dataframe
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('Msg_Category', engine)
    all_cols = df.columns.to_list()
    x_cols = all_cols[0:3]
    y_cols =  all_cols
    del y_cols[0:3]

    df_X = df[x_cols]
    df_Y = df[y_cols]

    X = df_X.message.values
    y = df_Y.values
    cats = df_Y.columns.to_list()
    return X, y, cats


def tokenize(text):
    '''
    tokenize
    normalise words of a sentence by preprocessing it intro individual words.
    Input:
     text - A sentence string with disaster message 
    Output:
     words: A list of pre-processed words
    '''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in words]
    return words


def build_model():
    '''
    build_model
    builds a model based on results of the grid search (Please refer to the ML Pipeline file). 
    This shows how the best parameters were obtained.
    Input: 
    Output:
     suitable model
    '''
    # Please refer to the  ML Pipieline to see how grid search was used to obtain these parameters
    best_params = {'clf__estimator__C': 3, 'clf__estimator__max_iter': 400}
    pipeline = sklearn.pipeline.Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                          ('tfidf', TfidfTransformer()),
                                          ('clf', MultiOutputClassifier(LogisticRegression()))])
    return pipeline.set_params(**best_params)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    evaluates a model and saves a txt file with "Eval.score.txt" 
    Input: 
     model - model
     X_test - test data features
     Y_test - test data labels (categories)
     category_names - categories
    Output:
     saves text file
    '''
    Precision = []
    Recall = []
    FScore = []
    Acc = []
    y_pred = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        report = classification_report(Y_test[:, i], y_pred[:, i], output_dict=True)
        
        macro_precision = report['macro avg']['precision']
        Precision.append(macro_precision)
        
        macro_recall = report['macro avg']['recall']
        Recall.append(macro_recall)
        
        macro_f1_score = report['macro avg']['f1-score']
        FScore.append(macro_f1_score)

        accuracy = report['accuracy']
        Acc.append(accuracy)
    
    Evaluate_1 = pd.DataFrame(
        {'Category': category_names,
        'Precision' : Precision,
        'Recall' : Recall,
        'Accuracy': Acc,
        'F1': FScore
        })
    Evaluate_1.to_csv('models/Eval_score.txt', sep='\t', index=False)


def save_model(model, model_filepath):
    '''
    save_model
    saves the model to appropriate file path 
    Input: 
     model - model
     model_filepath - model pkl file
    Output:
     saves model pickle file
    '''
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=43)
        
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