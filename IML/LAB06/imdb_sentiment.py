#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="imdb_sentiment.model", type=str, help="Model path")
# TODO: Add other arguments (typically hyperparameters) as you need.


class Dataset:
    """IMDB dataset.

    This is a modified IMDB dataset for sentiment classification. The text is
    already tokenized and partially normalized.
    """
    def __init__(self,
                 name="imdb_train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []
        with open(name) as f_imdb:
            for line in f_imdb:
                label, text = line.split("\t", 1)
                self.data.append(text)
                self.target.append(int(label))


def load_word_embeddings(
        name="imdb_embeddings.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
    """Load word embeddings.

    These are selected word embeddings from FastText. For faster download, it
    only contains words that are in the IMDB dataset.
    """
    if not os.path.exists(name):
        print("Downloading embeddings {}...".format(name), file=sys.stderr)
        urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
        os.rename("{}.tmp".format(name), name)

    with open(name, "rb") as f_emb:
        data = np.load(f_emb)
        words = data["words"]
        vectors = data["vectors"]
    embeddings = {word: vector for word, vector in zip(words, vectors)}
    return embeddings


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    word_embeddings = load_word_embeddings()
    from sklearn.feature_extraction import _stop_words
    import re
    stop_words = list(_stop_words.ENGLISH_STOP_WORDS)
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        print("Preprocessing dataset.", file=sys.stderr)
        # TODO: Preprocess the text such that you have a single vector per movie
        # review. You can experiments with different ways of pooling the word
        # embeddings: averaging, max pooling, etc. You can also try to exlucde
        # words that do not contribute much to the meaning of the sentence (stop
        # words). See `sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS`.
        #remove punctuation
        train_data = [' '.join(re.sub(r'[^\w\s]', ' ', review).split()) for review in train.data]
        train_data = [' '.join([word for word in review.split() if word not in stop_words]) for review in train_data]
        print(train_data[:10])
        print(train.target[:10])

        train_as_embeddings = [[word_embeddings[w] for w in doc.split() if w in word_embeddings] for doc in train_data]

        train_as_embeddings = [np.mean(doc, axis=0) for doc in train_as_embeddings] 

        train_x, validation_x, train_y, validation_y = sklearn.model_selection.train_test_split(
            train_as_embeddings, train.target, test_size=0.25, random_state=args.seed)
        
        #train_x, validation_x = train_x.multiply(vectorizer.idf_), validation_x.multiply(vectorizer.idf_)
        print("Training.", file=sys.stderr)
        # TODO: Train a model of your choice on the given data.
        from sklearn import linear_model
        model = sklearn.linear_model.LogisticRegression()
        grid_search = sklearn.model_selection.GridSearchCV(model, {'C': np.arange(0, 30, 0.1), 'solver': ['lbfgs', 'liblinear']}, cv=5)
        grid_search.fit(train_x, train_y)
        model = grid_search.best_estimator_
        print(grid_search.best_params_)
        print("Evaluation.", file=sys.stderr)
        validation_predictions = model.predict(validation_x)
        validation_accuracy = sklearn.metrics.accuracy_score(validation_y, validation_predictions)
        print("Validation accuracy {:.2f}%".format(100 * validation_accuracy))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)



        # TODO: Start by preprocessing the test data, ideally using the same
        # code as during training.
        test_data = [' '.join(re.sub(r'[^\w\s]', ' ', review).split()) for review in test.data]
        test_data = [' '.join([word for word in review.split() if word not in stop_words]) for review in test_data]
        test_as_vectors = [[word_embeddings[w] for w in doc.split() if w in word_embeddings] for doc in test_data]
        test_as_vectors = [np.mean(doc, axis=0) for doc in test_as_vectors]

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test_as_vectors)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
