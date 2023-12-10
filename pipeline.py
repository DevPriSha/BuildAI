import data_cleaning
import evaluation
import pandas as pd
from ml_models import classifiers, regressors

def pipeline_default(df, task, feature_set, vectorizer, features, label, model):
    #clean data
    df = data_cleaning.clean_data(df, label, task, feature = features)
    #extract features
    X_train, X_test, y_train, y_test  = data_cleaning.extract_features(df, features, label, vectorizer='tfidf')
    #train model
    top_models, filenames = evaluation.comparisonvisualisations(X_train, y_train, X_test, y_test, model)
    return top_models, filenames

if __name__ == '__main__':
    df = pd.read_csv('sample.csv')
    top_models, filenames = pipeline_default(df, 'classification', 'tfidf', 'count', ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'], 'Species', list(classifiers.keys()))
    print(top_models)
    # print(filenames)