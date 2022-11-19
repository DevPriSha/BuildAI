#create a basic flask app
import flask
from flask import request, jsonify, render_template
from ml_models import classifiers
from pipeline import pipeline_default
import pandas as pd

app = flask.Flask(__name__)
app.config["DEBUG"] = True

#defaults
task = 'sentiment-analysis'
feature_set = 'tfidf'
vectorizer = 'count'
features = ['text']
label = 'label'
model = classifiers
file = ""

# A route to return all of the available entries in our catalog.
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

#route that gets a csv file and returns a list of headers
@app.route('/headers', methods=['GET', 'POST'])
def api_headers():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'file' in request.args:
        global file
        file = request.args['file']
    else:
        return "Error: No file field provided. Please specify a file."

    #load data
    df = pd.read_csv(file)
    potential_features = list(df.columns)
    for column in list(df.columns):
        if column == label:
            potential_features.remove(column)
            potential_labels = [column]
        else:
            if task != 'regression':
                if df[column].dtype.name == 'category':
                    potential_labels.append(column)
            else:
                if df[column].dtype.name == 'float64':
                    potential_labels.append(column)
    return jsonify(potential_features, potential_labels)

#route that gets all the arguments and returns evaluation metrics with graph
@app.route('/models', methods=['GET', 'POST'])
def api_evaluate():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    global file
    if 'file' in request.args:
        file = request.args['file']
    else:
        return "Error: No file field provided. Please specify a file."
    if 'task' in request.args:
        global task
        task = request.args['task']
    else:
        task = 'sentiment-analysis'

    if 'feature_set' in request.args:
        global feature_set
        feature_set = request.args['feature_set']
    else:
        feature_set = 'tfidf'

    if 'vectorizer' in request.args:
        global vectorizer
        vectorizer = request.args['vectorizer']
    else:
        vectorizer = 'count'

    if 'features' in request.args:
        global features
        features = request.args['features']
    else:
        features = ['text']

    if 'label' in request.args:
        global label
        label = request.args['label']
    else:
        label = 'label'

    if 'model' in request.args:
        global model
        model = request.args['model']
    else:
        model = classifiers

    #load data
    df = pd.read_csv(file)
    #pipeline: clean data, extract features, train model, evaluate model
    #get evaluation metrics
    top_models, filenames = pipeline_default(df, task, feature_set, vectorizer, features, label, model)
    return jsonify(top_models, filenames)

if __name__ == '__main__':
    app.run()