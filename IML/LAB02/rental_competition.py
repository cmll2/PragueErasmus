#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.compose
import sklearn.datasets
import sklearn.pipeline
import sklearn.preprocessing

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")
    

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
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
        labels = train.target
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=0.2, random_state=args.seed)
        # TODO: Train a model on the given dataset and store it in `model`.
        model = sklearn.linear_model.Ridge()

        categorical_colums_indices = [col for col in range(data.shape[1]) if np.all(data[:, col] == data[:, col].astype(int))]
        numerical_columns_indices = [col for col in range(data.shape[1]) if col not in categorical_colums_indices]

        preprocessor = sklearn.compose.ColumnTransformer(transformers=[("categorical", 
        sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_colums_indices),
        ("numerical", sklearn.preprocessing.StandardScaler(), numerical_columns_indices)])

        polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)

        pipeline = sklearn.pipeline.Pipeline(steps=[("preprocessor", preprocessor), ("polynomial_features", polynomial_features)])
        X_train_transformed = pipeline.fit_transform(X_train)
        X_test_transformed = pipeline.transform(X_test)

        lambdas = np.geomspace(0.01, 100, num=500)
        best_lambda = None
        best_rmse = float("inf")

        model = sklearn.linear_model.Ridge()
        for lambda_val in lambdas:
            model.set_params(alpha=lambda_val)
            model.fit(X_train_transformed, y_train)
            y_pred = model.predict(X_test_transformed)
            rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared=False)
            if rmse < best_rmse:
                best_rmse = rmse
                best_lambda = lambda_val
                predictions = y_pred
        print("best RMSE : ", best_rmse)
        model.set_params(alpha=best_lambda)
        model.fit(X_train_transformed, y_train)
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        with open("pipeline.pkl", "wb") as pipeline_file:
            pickle.dump(pipeline, pipeline_file)
    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open("pipeline.pkl", "rb") as pipeline_file:
            pipeline = pickle.load(pipeline_file)
        data = test.data
        data_transformed = pipeline.transform(data)
        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(data_transformed)
    return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    predictions = main(args)




