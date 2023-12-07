#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
from typing import Optional

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="data/nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        print('Before preprocessing: ', train.data[0])
        # TODO: Train a model on the given dataset and store it in `model`.
        train_data = train.data
        train_target = train.target
        #train_test_split
        from sklearn.model_selection import train_test_split
        # train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size=0.1, random_state=42)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_df=0.8, sublinear_tf=True, use_idf=True, ngram_range=(1, 3), max_features=10000, lowercase=True)
        train_data = vectorizer.fit_transform(train_data)
        #test_data = vectorizer.transform(test_data)
        from sklearn.linear_model import RidgeClassifier
        from sklearn.model_selection import GridSearchCV
        model = RidgeClassifier()
        parameters = {'alpha': [1.4], 'tol': [1e-4], 'solver': ['auto', 'svd', 'saga']}
        clf = GridSearchCV(model, parameters, cv=5, n_jobs=-1)
        clf.fit(train_data, train_target)
        model = clf.best_estimator_
        print('Best parameters: ', clf.best_params_)
        print('Ridge Classifier train accuracy: ', model.score(train_data, train_target))
        #print('Ridge Classifier test accuracy: ', model.score(test_data, test_target))
        from sklearn.linear_model import LogisticRegression
        model_lr = LogisticRegression()
        parameters = {'C': [1], 'tol': [1e-4], 'solver': ['lbfgs', 'saga']}
        clf = GridSearchCV(model_lr, parameters, cv=5, n_jobs=-1)
        clf.fit(train_data, train_target)
        model_lr = clf.best_estimator_
        print('Best parameters: ', clf.best_params_)
        print('Logistic Regression train accuracy: ', model_lr.score(train_data, train_target))
        from sklearn.svm import LinearSVC
        model_lsvm = LinearSVC()
        parameters = {'C': [1], 'tol': [1e-4], 'loss': ['hinge', 'squared_hinge']}
        clf = GridSearchCV(model_lsvm, parameters, cv=5, n_jobs=-1)
        clf.fit(train_data, train_target)
        model_lsvm = clf.best_estimator_
        print('Best parameters: ', clf.best_params_)
        print('Linear SVM train accuracy: ', model_lsvm.score(train_data, train_target))
        
        #majority vote
        from sklearn.ensemble import VotingClassifier
        model = VotingClassifier(estimators=[('rc', model), ('lr', model_lr), ('lsvm', model_lsvm)], voting='hard')
        model.fit(train_data, train_target)
        print('Voting Classifier train accuracy: ', model.score(train_data, train_target))

        with lzma.open('vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(vectorizer, vectorizer_file)
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        with lzma.open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)

        test_data = vectorizer.transform(test.data)
        predictions = model.predict(test_data)
        
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
