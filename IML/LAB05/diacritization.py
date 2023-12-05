#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

def diacritize_word(input_word, vectorizer, clf):
    input_vector = vectorizer.transform([input_word])
    diacritized_word = clf.predict(input_vector)
    return diacritized_word[0]

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        train_data = train.data
        train_data = train_data.split('\n')
        train_target = train.target
        train_target = train_target.split('\n')

        tfidf_vectorizer = TfidfVectorizer()
        X = tfidf_vectorizer.fit_transform(train_data)
        from sklearn.preprocessing import LabelEncoder
        from sklearn.naive_bayes import MultinomialNB
        # Label encoding
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(train_target)

        # Train-test split
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model selection and training
        classifier = MultinomialNB()
        classifier.fit(X, y)

        # Model evaluation
        y_pred = classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        print(f'Accuracy: {accuracy}')

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(classifier, model_file)

        #store the vectorizer
        with lzma.open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(tfidf_vectorizer, vectorizer_file)

        with lzma.open('label_encoder.pkl', 'wb') as label_encoder_file:
            pickle.dump(label_encoder, label_encoder_file)
        
        y_label = label_encoder.inverse_transform(y_pred)
        return " ".join(y_label)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        with lzma.open('vectorizer.pkl', "rb") as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        with lzma.open('label_encoder.pkl', "rb") as label_encoder_file:
            label_encoder = pickle.load(label_encoder_file)

        test_data = test.data
        test_data = test_data.split('\n')
        test_data_matrix = vectorizer.transform(test_data)

        predicted_label = model.predict(test_data_matrix)
        predicted_diacritic = label_encoder.inverse_transform(predicted_label)

        return " ".join(predicted_diacritic)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
