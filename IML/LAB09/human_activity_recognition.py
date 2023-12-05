#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="human_activity_recognition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["sitting", "sittingdown", "standing", "standingup", "walking"]

    def __init__(self,
                 name="human_activity_recognition.train.csv.xz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and if it contains column "class", split it to `targets`.
        self.data = pd.read_csv(name)
        if "class" in self.data:
            self.target = np.array([Dataset.CLASSES.index(target) for target in self.data["class"]], np.int32)
            self.data = self.data.drop("class", axis=1)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        from sklearn.model_selection import train_test_split
        train_data, train_target = train.data, train.target

        #preprocessing
        from sklearn.preprocessing import StandardScaler
        from sklearn.preprocessing import PolynomialFeatures
        # TODO: Train a model on the given dataset and store it in `model`.
        #use random forest
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import accuracy_score
        # model = GridSearchCV(Pipeline([('scaler', StandardScaler()), ('poly', PolynomialFeatures()), ('rf', RandomForestClassifier())]), param_grid={'rf__n_estimators': [10, 50, 100], 'rf__max_depth': [5, 10, 15]}, cv=5, n_jobs=-1)
        # model.fit(train_data, train_target)
        # print(model.best_params_)
        # print(model.best_score_)
        from sklearn.preprocessing import MinMaxScaler
        # scaler = MinMaxScaler()
        # scaler.fit(train_data)
        # train_data = scaler.transform(train_data)
        # test_data = scaler.transform(test_data)
        model = GridSearchCV(RandomForestClassifier(), param_grid={'n_estimators': [400,500,600,700], 'max_depth': [50,60,70,80]}, cv=5, n_jobs=-1)
        print('training...')
        model.fit(train_data, train_target)
        print('training done')
        print(model.best_params_)
        print(model.best_score_)
        model = model.best_estimator_
        print('train accuracy: ', accuracy_score(train_target, model.predict(train_data)))
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

        # with lzma.open('scaler.pkl', 'wb') as scaler_file:
        #     pickle.dump(scaler, scaler_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # with lzma.open('scaler.pkl', 'rb') as scaler_file:
        #     scaler = pickle.load(scaler_file)

        # test_data = scaler.transform(test.data)


        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
