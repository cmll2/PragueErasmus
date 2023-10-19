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
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        data = train.data
        target = train.target

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2, random_state=args.seed)

        # TODO: Train a model on the given dataset and store it in `model`.
        min_max_scaler = sklearn.preprocessing.StandardScaler()
        polynomial_features = sklearn.preprocessing.PolynomialFeatures()
        model = sklearn.linear_model.LogisticRegression(random_state=args.seed)

        pipeline = sklearn.pipeline.Pipeline(steps=[("min_max_scaler", min_max_scaler), ("polynomial_features", polynomial_features), ("logistic_regression", model)])

        import warnings
        warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
        grid_search = sklearn.model_selection.GridSearchCV(pipeline, param_grid={"polynomial_features__degree": [1, 2, 3], "logistic_regression__C": [0.01, 1, 100], "logistic_regression__solver": ["lbfgs", "sag"]}, cv=sklearn.model_selection.StratifiedKFold(5))
        grid_search.fit(X_train, y_train)

        predictions = grid_search.predict(X_test)

        test_accuracy = grid_search.score(X_test, y_test)
        print("Test accuracy: {:.2f}%".format(100 * test_accuracy))
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(grid_search, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)

    return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
