from ml_models import regressors, classifiers, get_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

#evaluation metrics of a model
def get_evaluation_metrics(modelname, X,y,X_test,y_test):
    if modelname in regressors:
        return get_regression_metrics(modelname, X,y,X_test,y_test)
    if modelname in classifiers:
        return get_classification_metrics(modelname, X,y,X_test,y_test)
    return None

#evaluation metrics of a regression model
def get_regression_metrics(modelname, X,y,X_test,y_test):
    model = get_model(modelname)
    y_pred = model.predict(X_test)
    return {
        'model': modelname,
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }

#evaluation metrics of a classification model
def get_classification_metrics(modelname, X,y,X_test,y_test):
    model = get_model(modelname)
    y_pred = model.predict(X_test)
    return {
        'model': modelname,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

#create a bar graph of the evaluation metrics of all models
def comparisonvisualisations(X,y,X_test,y_test, user_models = classifiers):
    eval_metrics = {}
    for model in user_models:
        if model in regressors:
            eval_metrics.append(get_regression_metrics(model, X,y,X_test,y_test))
        if model in classifiers:
            eval_metrics.append(get_classification_metrics(model, X,y,X_test,y_test))
    #create model vs metric dataframe
    df = pd.DataFrame(eval_metrics)
    #get top model names for each metric
    top_models = {}
    for metric in df.columns:
        if metric != 'model':
            top_models[metric] = df.nlargest(1, metric)['model'].values[0]
    filenames = []
    #create model vs metric bar graph for each metric
    for metric in df.columns:
        if metric == 'model':
            continue
        df.plot.bar(x='model', y=metric)
        plt.show()
        filename = '{}-{}.png'.format(metric, time.strftime("%d-%m-%Y-%H-%M-%S"))
        plt.savefig(filename)
        filenames.append(filename)
    return top_models, filenames

#heatmap of the correlation between the features
def correlationvisualisation(X):
    corr = X.corr()
    sns.heatmap(corr, 
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    plt.show()
    filename = 'heatmap-{}.png'.format(time.strftime("%d-%m-%Y-%H-%M-%S"))
    plt.savefig(filename)
    return filename

#scatter plot of the features
def scatterplotvisualisation(X,y):
    sns.pairplot(X)
    plt.show()
    filename = 'scatterplot-{}.png'.format(time.strftime("%d-%m-%Y-%H-%M-%S"))
    plt.savefig(filename)
    return filename


        



