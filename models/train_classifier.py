# import libraries

import nltk
from nltk.sem.evaluate import _ELEMENT_SPLIT_RE

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 500)

import sys
import os
import re
from sqlalchemy import create_engine
import pickle


# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import nltk

nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# writing a fucntion for loading the data to train our model
def load_data(database_filepath):
    engine = create_engine("sqlite:///" + database_filepath)
    table = os.path.basename(database_filepath).replace(".db", "") + "_table"
    df = pd.read_sql_table(table_name=table, con=engine)

    # only value=0 occurences in child_alone -> drop
    df.drop("child_alone", axis=1, inplace=True)

    # drop rows with missing values
    df.dropna(inplace=True)

    # allow no 2 values - only boolean values 1 | 0
    df["related"] = df["related"].map(lambda x: 1 if x == 2 else x)

    # splitting into features and labels
    X = df["message"]
    y = df.iloc[:, 4:]

    plot_names = y.columns
    return X, y, plot_names


# writing a tokenizer function for the CountVectorizer including a regex for url detection (we want to exclude those )
def tokenize(text):
    # Replace all urls with a urlplaceholder string
    url_regex = (
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    # Extract all the urls from the provided text
    detected_urls = re.findall(url_regex, text)

    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, "url_placeholder")

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)

    # Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
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


# building the pipeline by using a AdaBoostClassifier as our estimator
def build_model():

    param_grid = {
        "classifier__estimator__learning_rate": [0.01, 0.02],
        "classifier__estimator__n_estimators": [10, 20],
    }

    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [
                                    (
                                        "count_vectorizer",
                                        CountVectorizer(tokenizer=tokenize),
                                    ),
                                    ("tfidf_transformer", TfidfTransformer()),
                                ]
                            ),
                        ),
                        ("starting_verb_transformer", StartingVerbExtractor()),
                    ]
                ),
            ),
            ("classifier", MultiOutputClassifier(AdaBoostClassifier())),
        ]
    )

    cv = GridSearchCV(pipeline, param_grid=param_grid, scoring="f1_micro", n_jobs=-1)
    return pipeline


# printing a overall accuracy and the classificiton report of sklearn
def evaluate_model(model, X_test, Y_test, category_names):
    pred = model.predict(X_test)
    acc = (pred == Y_test).mean().mean()
    confusion_mat = classification_report(Y_test, pred, target_names=category_names)
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_mat)


# save the model to use it within our webapp
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


# function for executing all together
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()
