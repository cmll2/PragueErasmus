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
parser.add_argument("--seed", default=41, type=int, help="Random seed")
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
        train.data = np.delete(train.data, [0, 9], axis=1)
        # correlations_variables = np.abs(np.corrcoef(train.data.T)[:-1, :-1])
        # print("Correlations inter-variables : ", correlations_variables)
        # #print correlations matrice
        # import matplotlib.pyplot as plt
        # plt.matshow(correlations_variables)
        # plt.show()

        # TODO: Train a model on the given dataset and store it in `model`.

        #correlations between data and target
        best_features = np.abs(np.corrcoef(train.data.T, train.target.T)[-1, :-1])
        best_features = np.argsort(best_features)[::-1]
        print(best_features)

        # print("Importance des variables : ", correlations)
        #correlations between variables
        lambdas = np.geomspace(0.01, 10, num=500)
        best_rmse = np.inf
        model = sklearn.linear_model.Ridge()
        for i in range(len(best_features)):
            dataset = train.data[:, best_features[:i+1]]
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataset, train.target, test_size=0.1, random_state=args.seed)

            categorical_colums_indices = [col for col in range(dataset.shape[1]) if np.all(dataset[:, col] == dataset[:, col].astype(int))]
            numerical_columns_indices = [col for col in range(dataset.shape[1]) if col not in categorical_colums_indices]
            preprocessor = sklearn.compose.ColumnTransformer(transformers=[("categorical", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_colums_indices),
            ("numerical", sklearn.preprocessing.StandardScaler(), numerical_columns_indices)
            ])
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
            for lambda_val in lambdas:
                model.set_params(alpha=lambda_val)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                rmse = np.sqrt(np.mean((predictions - y_test)**2))
                if rmse < best_rmse:
                    best_rmse = copy.copy(rmse)
                    best_predictions = copy.copy(predictions)
                    best_model = copy.copy(model)
                    best_preprocessor = copy.copy(preprocessor)
                    features = best_features[:i+1]
        print(features)
        np.save(numpy_file, features)
        pickle.dump(best_preprocessor, open("preprocessor.pkl", "wb"))
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(best_model, model_file)
        print("Best RMSE : ", best_rmse)
        return best_predictions
    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open("preprocessor.pkl", "rb") as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)
        features = np.load(numpy_file)
        test.data = np.delete(test.data, [0, 9], axis=1)
        test.data = test.data[:, features]
        data = preprocessor.transform(test.data)
        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(data)
    return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    predictions = main(args)