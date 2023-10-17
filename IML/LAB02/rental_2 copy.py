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
import copy

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
    numpy_file = 'best_features.npy'
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        
        # TODO: Train a model on the given dataset and store it in `model`.
        model = sklearn.linear_model.LinearRegression()
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.05, random_state=args.seed)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(np.mean((predictions - y_test)**2))
        print("RMSE : ", rmse)
        coefficients = model.coef_
        absolutes = np.abs(coefficients)
        best_features = np.argsort(absolutes)[::-1]

        generator = np.random.RandomState(92)
        weights = generator.uniform(size=X_train.shape[1], low=-0.1, high=0.1)
        batch_size = 10
        l2 = 0.0
        learning_rate = 0.005
        best_rmse = np.inf
        for i in range(len(best_features)):
            dataset = train.data[:, best_features[:i+1]]
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset, train.target, test_size=0.2, random_state=args.seed)
            categorical_colums_indices = [col for col in range(dataset.shape[1]) if np.all(dataset[:, col] == dataset[:, col].astype(int))]
            numerical_columns_indices = [col for col in range(dataset.shape[1]) if col not in categorical_colums_indices]
            preprocessor = sklearn.compose.ColumnTransformer(transformers=[("categorical", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_colums_indices),
            ("numerical", sklearn.preprocessing.StandardScaler(), numerical_columns_indices)
            ])
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            weights = generator.uniform(size=X_train.shape[1], low=-0.1, high=0.1)

            for _ in range(1000):
                permutation = generator.permutation(X_train.shape[0])
                for batch in range(0, len(X_train), batch_size):
                    batch_data = X_train[permutation[batch:batch+batch_size]]
                    batch_target = y_train[permutation[batch:batch+batch_size]]
                    batch_prediction = np.dot(batch_data, weights)
                    batch_residuals = batch_prediction - batch_target
                    gradient = np.dot(batch_residuals, batch_data) / len(batch_data)
                    if l2 != 0:
                        weights_with_bias_set_to_zero = weights.copy()
                        weights_with_bias_set_to_zero[-1] = 0
                        gradient+=l2 * weights_with_bias_set_to_zero
                    weights -= learning_rate*10 * gradient
            test_predictions = np.dot(X_test, weights)
            rmse = np.sqrt(np.mean((test_predictions - y_test)**2))
            if rmse < best_rmse:
                best_weights = copy.copy(weights)
                best_rmse = copy.copy(rmse)
                best_predictions = copy.copy(test_predictions)
                best_preprocessor = copy.copy(preprocessor)
                features = best_features[:i+1]
        print(features)
        np.save(numpy_file, features)
        np.save("best_weights.npy", best_weights)
        pickle.dump(best_preprocessor, open("preprocessor.pkl", "wb"))
        print("RMSE : ", best_rmse)
        return best_predictions

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        with open("preprocessor.pkl", "rb") as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
        features = np.load(numpy_file)
        weights = np.load("best_weights.npy")
        test.data = test.data[:, features]
        data = preprocessor.transform(test.data)
        # TODO: Generate `predictions` with the test set predictions.
        predictions = np.dot(data, weights)
    return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    predictions = main(args)