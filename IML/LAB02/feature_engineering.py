#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.preprocessing._encoders")

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="diabetes", type=str, help="Standard sklearn dataset to load")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    dataset = getattr(sklearn.datasets, "load_{}".format(args.dataset))()

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    data = dataset.data
    labels = dataset.target

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, labels, test_size=args.test_size, random_state=args.seed)

    # TODO: Process the input columns in the following way:
    #
    # - if a column has only integer values, consider it a categorical column
    #   (days in a week, dog breed, ...; in general, integer values can also
    #   represent numerical non-categorical values, but we use this assumption
    #   for the sake of exercise). Encode the values with one-hot encoding
    #   using `sklearn.preprocessing.OneHotEncoder` (note that its output is by
    #   default sparse, you can use `sparse=False` to generate dense output;
    #   also use `handle_unknown="ignore"` to ignore missing values in test set).
    #
    # - for the rest of the columns, normalize their values so that they
    #   have mean 0 and variance 1; use `sklearn.preprocessing.StandardScaler`.
    #
    # In the output, first there should be all the one-hot categorical features,
    # and then the real-valued features. To process different dataset columns
    # differently, you can use `sklearn.compose.ColumnTransformer`.

    categorical_colums_indices = [col for col in range(data.shape[1]) if np.all(data[:, col] == data[:, col].astype(int))]
    numerical_columns_indices = [col for col in range(data.shape[1]) if col not in categorical_colums_indices]

    preprocessor = sklearn.compose.ColumnTransformer(transformers=[("categorical", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_colums_indices),
        ("numerical", sklearn.preprocessing.StandardScaler(), numerical_columns_indices)
    ])

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)


    # TODO: To the current features, append polynomial features of order 2.
    # If the input values are `[a, b, c, d]`, you should append
    # `[a^2, ab, ac, ad, b^2, bc, bd, c^2, cd, d^2]`. You can generate such polynomial
    # features either manually, or you can employ the provided transformer
    #   sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    # which appends such polynomial features of order 2 to the given features.

    polynomial_features = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False)
    X_train_polynomial = polynomial_features.fit_transform(X_train_transformed)
    X_test_polynomial = polynomial_features.transform(X_test_transformed)

    # TODO: You can wrap all the feature processing steps into one transformer
    # by using `sklearn.pipeline.Pipeline`. Although not strictly needed, it is
    # usually comfortable.

    pipeline = sklearn.pipeline.Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("polynomial_features", polynomial_features)
    ])

    # TODO: Fit the feature preprocessing steps (the composed pipeline with all of
    # them; or the individual steps, if you prefer) on the training data (using `fit`).
    # Then transform the training data into `train_data` (with a `transform` call;
    # however, you can combine the two methods into a single `fit_transform` call).
    # Finally, transform testing data to `test_data`.
    train_data = pipeline.fit_transform(X_train)
    test_data = pipeline.transform(X_test)

    assert np.allclose(train_data, X_train_polynomial), "train_data does not match X_train_polynomial"
    assert np.allclose(test_data, X_test_polynomial), "test_data does not match X_test_polynomial"

    return train_data[:5], test_data[:5]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_data, test_data = main(args)
    for dataset in [train_data, test_data]:
        for line in range(min(dataset.shape[0], 5)):
            print(" ".join("{:.4g}".format(dataset[line, column]) for column in range(min(dataset.shape[1], 140))),
                  *["..."] if dataset.shape[1] > 140 else [])