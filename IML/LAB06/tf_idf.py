#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=45, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=500, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names


def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.
    import re
    pattern = r'\w+'
    features = {}

    #remove punctuation in data
    train_data = [re.sub(r'[^\w\s]', ' ', document) for document in train_data]
    test_data = [re.sub(r'[^\w\s]', ' ', document) for document in test_data]

    for document in train_data:
        #extract every term from document
        for word in re.findall(pattern, document):
            #if term is already in features, increase its count
            if word in features:
                features[word] += 1
            #else add it to features
            else:
                features[word] = 1
    #print('Number of all words: ', len(features))
    #filter out words that are present less than twice
    features = {k:v for k,v in features.items() if v >= 2}
    print('Number of features: ', len(features))


    # TODO: For each document, compute its features as
    # - term frequency(TF), if `args.tf` is set;
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    TF_data = np.zeros((len(train_data), len(features)))
    TF_test = np.zeros((len(test_data), len(features)))
    features_index = {k:i for i,k in enumerate(features.keys())}
    if args.tf: #compute TF
        for i, document in enumerate(train_data):
            number_of_terms = 0
            for word in document.split():
                if word in features:
                    number_of_terms += 1
                    TF_data[i, features_index[word]] += 1
            TF_data[i] /= number_of_terms if number_of_terms > 0 else 0
        for i, document in enumerate(test_data):
            number_of_terms = 0
            for word in document.split():
                if word in features:
                    number_of_terms += 1
                    TF_test[i, features_index[word]] += 1
            TF_test[i] /= number_of_terms if number_of_terms > 0 else 0
    else: #compute binary indicators
        for i, document in enumerate(train_data):
            for word in document.split():
                if word in features:
                    TF_data[i, features_index[word]] = 1
        for i, document in enumerate(test_data):
            for word in document.split():
                if word in features:
                    TF_test[i, features_index[word]] = 1
    if args.idf:
        idf = np.zeros(len(features))
        number_of_documents_containing_term = np.zeros(len(features))
        number_of_documents = len(train_data)
        for i, document in enumerate(train_data):
            finded_words = []
            for word in document.split():
                if word in features:
                    if word not in finded_words:
                        finded_words.append(word)
                        number_of_documents_containing_term[features_index[word]] += 1
                    
                    
        idf = np.log((number_of_documents) / (number_of_documents_containing_term + 1))
        TF_data = [TF_data[i] * idf for i in range(len(TF_data))]
        TF_test = [TF_test[i] * idf for i in range(len(TF_test))]

    train_features = TF_data
    test_features = TF_test

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear")`
    # model on the train set, and classify the test set.
    model = sklearn.linear_model.LogisticRegression(solver="liblinear", C = 10_000)
    model.fit(train_features, train_target)

    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    predictions = model.predict(test_features)
    f1_score = sklearn.metrics.f1_score(test_target, predictions, average="macro")

    return 100 * f1_score


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))
