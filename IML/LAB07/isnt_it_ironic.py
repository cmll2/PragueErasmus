#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import re
import numpy as np
import numpy.typing as npt
#import stopword
from sklearn.feature_extraction import _stop_words
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    stopwords = _stop_words.ENGLISH_STOP_WORDS
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        print('Before preprocessing: ', train.data[0])
        # TODO: Train a model on the given dataset and store it in `model`.
        train_data = train.data
        train_target = train.target
        #split data into train and test
        from sklearn.model_selection import train_test_split
        train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.1, random_state=42)

        #use vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, max_df=0.8, max_features=5000)
        vectorizer.fit(train_data)
        train_data = vectorizer.transform(train_data)

        #Naive Bayes
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.model_selection import GridSearchCV
        model = MultinomialNB(fit_prior=True)
        params = {'alpha': [4, 5, 6, 7]}
        model = GridSearchCV(model, params, cv=5, n_jobs=-1)
        model.fit(train_data, train_target)
        print('Best alpha: ', model.best_params_['alpha'])
        model_nb = model.best_estimator_
        model_nb.fit(train_data, train_target)
        #fit two more naive bayes models
        from sklearn.naive_bayes import GaussianNB
        from sklearn.naive_bayes import BernoulliNB
        model_gauss = GaussianNB()
        params = {'var_smoothing': [1, 2, 3, 4, 5, 6]}
        model_gauss = GridSearchCV(model_gauss, params, cv=5, n_jobs=-1)
        model_gauss.fit(train_data.toarray(), train_target)
        print('Best var_smoothing: ', model_gauss.best_params_['var_smoothing'])
        model_gauss = model_gauss.best_estimator_
        model_gauss.fit(train_data.toarray(), train_target)
        model_bern = BernoulliNB()
        params = {'alpha': [4, 5, 6, 7]}
        model_bern = GridSearchCV(model_bern, params, cv=5, n_jobs=-1)
        model_bern.fit(train_data, train_target)
        print('Best alpha: ', model_bern.best_params_['alpha'])
        model_bern = model_bern.best_estimator_
        model_bern.fit(train_data, train_target)

        # #print their accuracies
        vectorized_test_data = vectorizer.transform(test_data)
        from sklearn.metrics import accuracy_score
        print('Multinomial Naive Bayes accuracy: ', accuracy_score(test_target, model_nb.predict(vectorized_test_data)))
        print('Gaussian Naive Bayes accuracy: ', accuracy_score(test_target, model_gauss.predict(vectorized_test_data.toarray())))
        print('Bernoulli Naive Bayes accuracy: ', accuracy_score(test_target, model_bern.predict(vectorized_test_data)))

        predictions= []
        #soft voting
        for i in range(len(test_data)):
            nb = model_nb.predict_proba(vectorizer.transform([test_data[i]]))[0]
            gauss = model_gauss.predict_proba(vectorizer.transform([test_data[i]]).toarray())[0]
            bern = model_bern.predict_proba(vectorizer.transform([test_data[i]]))[0]
            nb_index = np.argmax(nb)
            gauss_index = np.argmax(gauss)
            bern_index = np.argmax(bern)
            if nb[nb_index] > gauss[gauss_index] and nb[nb_index] > bern[bern_index]:
                predictions.append(nb_index)
            elif gauss[gauss_index] > nb[nb_index] and gauss[gauss_index] > bern[bern_index]:
                predictions.append(gauss_index)
            elif bern[bern_index] > nb[nb_index] and bern[bern_index] > gauss[gauss_index]:
                predictions.append(bern_index)
            else:
                predictions.append(nb_index)

        print('Soft voting accuracy: ', accuracy_score(test_target, predictions))
        #hard voting
        predictions = []
        for i in range(len(test_data)):
            nb = model_nb.predict(vectorizer.transform([test_data[i]]))[0]
            gauss = model_gauss.predict(vectorizer.transform([test_data[i]]).toarray())[0]
            bern = model_bern.predict(vectorizer.transform([test_data[i]]))[0]
            if nb == gauss and nb != bern:
                predictions.append(nb)
            elif nb == bern and nb != gauss:
                predictions.append(nb)
            elif gauss == bern and gauss != nb:
                predictions.append(gauss)
            else:
                predictions.append(nb)
        print('Hard voting accuracy: ', accuracy_score(test_target, predictions))
        with lzma.open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model_nb, model_file)

        with lzma.open('model_gauss.pkl', 'wb') as model_file:
            pickle.dump(model_gauss, model_file)

        with lzma.open('model_bern.pkl', 'wb') as model_file:
            pickle.dump(model_bern, model_file)

        return np.asarray(predictions).flatten()

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        with lzma.open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        with lzma.open('model_gauss.pkl', 'rb') as model_file:
            model_gauss = pickle.load(model_file)

        with lzma.open('model_bern.pkl', 'rb') as model_file:
            model_bern = pickle.load(model_file)

        test_data = test.data
        
        predictions = []
        #soft voting
        for i in range(len(test_data)):
            nb = model.predict_proba(vectorizer.transform([test_data[i]]))[0]
            gauss = model_gauss.predict_proba(vectorizer.transform([test_data[i]]).toarray())[0]
            bern = model_bern.predict_proba(vectorizer.transform([test_data[i]]))[0]
            nb_index = np.argmax(nb)
            gauss_index = np.argmax(gauss)
            bern_index = np.argmax(bern)
            if nb[nb_index] > gauss[gauss_index] and nb[nb_index] > bern[bern_index]:
                predictions.append(nb_index)
            elif gauss[gauss_index] > nb[nb_index] and gauss[gauss_index] > bern[bern_index]:
                predictions.append(gauss_index)
            elif bern[bern_index] > nb[nb_index] and bern[bern_index] > gauss[gauss_index]:
                predictions.append(bern_index)
            else:
                predictions.append(nb_index)
        
        return np.asarray(predictions).flatten()


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
