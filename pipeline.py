import data_cleaning
import evaluation

def pipeline_default(df, task, feature_set, vectorizer, features, label, model):
    #clean data
    df = data_cleaning.clean_data(df, task)
    #extract features
    X_train, X_test, y_train, y_test  = evaluation.extract_features(df, vectorizer)
    #train model
    top_models, filenames = evaluation.comparisonvisualisations(X_train, X_test, y_train, y_test, model)
    return top_models, filenames
