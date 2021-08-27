import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from sklearn.base import BaseEstimator, TransformerMixin

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# write the tokenizer function for the CountVectorizer
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# writing a own VerExtraction class for the ml pipeline
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ["VB", "VBP"] or first_word == "RT":
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# loading the data from our created database
engine = create_engine("sqlite:///../data/disaster_response_db.db")
df = pd.read_sql_table("disaster_response_db_table", engine)

# loading our created classifier model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():

    # extract data needed for visuals
    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    cat_names = df.iloc[:, 4:].columns
    cat_sum = (df.iloc[:, 4:] != 0).sum().values
    cat_rel_sum = cat_sum / len(df)

    # create visuals
    graphs = [
        # GRAPH 1 - genre graph
        {
            "data": [Bar(x=genre_names, y=genre_counts)],
            "layout": {
                "title": "Distribution of Message Genres",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Genre"},
            },
        },
        # GRAPH 2 - which cat occures the most
        {
            "data": [Bar(x=cat_names, y=cat_sum)],
            "layout": {
                "title": "Distribution of Message Categories",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": 35},
            },
        },
        # Graph 3 - relative distribution of categories
        {
            "data": [Scatter(x=cat_names, y=cat_rel_sum)],
            "layout": {
                "title": f"Relative Distribution of Message Categories (One message can belong to multiple categories); n = {len(df)} messages",
                "yaxis": {"title": "Count"},
                "xaxis": {"title": "Category", "tickangle": 35},
            },
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    # app.run(host="0.0.0.0", port=3001, debug=True)
    app.run(debug=True)


if __name__ == "__main__":
    main()
