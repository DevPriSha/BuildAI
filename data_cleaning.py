import json
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

#need features and labels

#clean data for sentiment analysis
def clean_data_sentiment(df, label='label',  features = ['text']):
    df = df.dropna(axis=0, how='any')
    df = df.drop_duplicates()
    try:
        df = df.drop(['id'], axis=1)
    except:
        pass
    df = df[df[label].isin(['Positive', 'Negative'])]
    df[label] = df[label].replace(['Positive', 'Negative'], [1, 0])
    lemmatizer = WordNetLemmatizer()

    for feature in features:
        if df[feature].dtype == 'object':
            df[feature] = df[feature].str.lower()
            df[feature] = df[feature].str.replace(r'[^\w\s]', '', regex=True)
            df[feature] = df[feature].str.replace(r'\d+', '', regex=True)
            df[feature] = df[feature].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    return df

#clean data for time series forecasting
def clean_data_timeseries(df):
    df = df.dropna()
    df = df.drop_duplicates()
    try:
        df = df.drop(['id'], axis=1)
    except:
        pass
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.sort_index()
    for column in df.columns:
        if df[column].dtype.name == 'category':
            df[column] = pd.Categorical(df[column]).codes 
    return df

#clean data for classification
def clean_data_classification(df, label='label'):
    df = df.dropna()
    df = df.drop_duplicates()
    try:
        df = df.drop(['id'], axis=1)
    except:
        pass
    df[label] = pd.Categorical(df[label]).codes 
    return df

#clean data for regression
def clean_data_regression(df):
    df = df.dropna()
    df = df.drop_duplicates()
    try:
        df = df.drop(['id'], axis=1)
    except:
        pass
    return df

def clean_data(df, label='label', task = 'sentiment', feature = ['text']):
    if task == 'sentiment':
        df = clean_data_sentiment(df, label, feature)
    elif task == 'timeseries':
        df = clean_data_timeseries(df)
    elif task == 'classification':
        df = clean_data_classification(df, label)
    elif task == 'regression':
        df = clean_data_regression(df)
    return df

#extract features
def extract_features_text(df, vectorizer='tfidf', features=['text']):
    for feature in features:
        if df[feature].dtype == 'object':
            if vectorizer == 'tfidf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(df[feature])
            elif vectorizer == 'count':
                from sklearn.feature_extraction.text import CountVectorizer
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(df[feature])
            elif vectorizer == 'hashing':
                from sklearn.feature_extraction.text import HashingVectorizer
                vectorizer = HashingVectorizer()
                X = vectorizer.fit_transform(df[feature])
            elif vectorizer == 'word2vec':
                from gensim.models import Word2Vec
                model = Word2Vec(df[features], min_count=1)
                X = model[df[feature]]
            elif vectorizer == 'doc2vec':
                from gensim.models.doc2vec import Doc2Vec, TaggedDocument
                documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(df[feature])]
                model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
                X = model[df[feature]]
    return X

#extract features numerical
def extract_features(df, features=['text'], label='label', vectorizer='tfidf'):
#X is a matrix of numeric features   
    df2 = pd.DataFrame()
    for feature in features:
        if df[feature].dtype == 'int64' or df[feature].dtype == 'float64':
            df2[feature] = df[feature].astype(df[feature].dtype)
            X = df2.values
        else:
            X = extract_features_text(df, vectorizer, features)

    X_train, X_test, y_train, y_test = train_test_split(X, df, label)
    return X_train, X_test, y_train, y_test


#train test split
def train_test_split(X, df, label='label', test_size=0.2):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, df[label], test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

#check task and apply appropriate data preprocessing
def check_task(df, task = 'sentiment'):
    if task == 'sentiment':
        df = clean_data_sentiment(df)
    elif task == 'timeseries':
        df = clean_data_timeseries(df)
    elif task == 'classification':
        df = clean_data_classification(df)
    elif task == 'regression':
        df = clean_data_regression(df)
    return df
