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
import scipy.ndimage
import sklearn.ensemble
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--alpha", default=0.0, type=float, help="Regularization strength")
parser.add_argument("--augment", default=False, action="store_true", help="Augment during training")
parser.add_argument("--epochs", default=15, type=int, help="Training epochs")
parser.add_argument("--hidden_layer", default=500, type=int, help="Hidden layer size")
parser.add_argument("--models", default=1, type=int, help="Model to train")
parser.add_argument("--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """
    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


# The following class modifies `MLPClassifier` to support full categorical distributions
# on input, i.e., each label should be a distribution over the predicted classes.
# During prediction, the most likely class is returned, but similarly to `MLPClassifier`,
# the `predict_proba` method returns the full distribution.
# Note that because we overwrite a private method, it is guaranteed to work only with
# scikit-learn 1.3.0, but it will most likely work with any 1.3.*.
class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(X, y, multi_output=True, dtype=(np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = y.shape[0]
        return X, y
    
def augment(x):
    x = x.reshape(28, 28)
    x = scipy.ndimage.zoom(x.reshape(28, 28), (np.random.uniform(0.86, 1.2), np.random.uniform(0.86, 1.2)))
    x = np.pad(x, [(2, 2), (2, 2)])
    os = [np.random.randint(size - 28 + 1) for size in x.shape]
    x = x[os[0]:os[0] + 28, os[1]:os[1] + 28]
    x = scipy.ndimage.rotate(x, np.random.uniform(-15, 15), reshape=False)
    x = np.clip(x, 0, 1)
    return x.reshape(-1)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ("MLPs", sklearn.ensemble.VotingClassifier([
                ("MLP{}".format(i), sklearn.neural_network.MLPClassifier(
                    tol=0, alpha=args.alpha, hidden_layer_sizes=args.hidden_layer, max_iter=1 if args.augment else args.epochs, verbose=False))
                for i in range(args.models)
            ], voting="soft")),
        ])
        model.fit(train.data, train.target)

        if args.augment:
            import multiprocessing
            pool = multiprocessing.Pool(16)
            for mlp in model["MLPs"].estimators_:
                for epoch in range(args.epochs - 1):
                    print("Augmenting data for epoch {}...".format(epoch), end="", flush=True)
                    augmented_data = pool.map(augment, model["scaler"].transform(train.data))
                    print("Done")
                    mlp.partial_fit(augmented_data, train.target)

        # If you trained one or more MLPs, you can use the following code
        # to compress it significantly (approximately 12 times). The snippet
        # assumes the trained `MLPClassifier` is in the `mlp` variable.
        #   mlp._optimizer = None
        #   for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        #   for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        #print teacher accuracy
        print("Teacher accuracy: {:.2f}%".format(100 * model.score(train.data, train.target)))
        #train a student model using the teacher model, knowledge distillation

        student = sklearn.pipeline.Pipeline([
            ("scaler", sklearn.preprocessing.MinMaxScaler()),
            ("MLP", MLPFullDistributionClassifier(
                tol=0, alpha=args.alpha, hidden_layer_sizes = args.hidden_layer, max_iter=1 if args.augment else 150, verbose=False))
        ])

        student.fit(train.data, model.predict_proba(train.data))

        if args.augment:
            import multiprocessing
            pool = multiprocessing.Pool(16)
            for epoch in range(150):
                print("Augmenting data for epoch {}...".format(epoch), end="", flush=True)
                augmented_data = pool.map(augment, student["scaler"].transform(train.data))
                print("Done")
                student["MLP"].partial_fit(augmented_data, model.predict_proba(train.data))

        student["MLP"]._optimizer = None
        for i in range(len(student["MLP"].coefs_)): student["MLP"].coefs_[i] = student["MLP"].coefs_[i].astype(np.float16)
        for i in range(len(student["MLP"].intercepts_)): student["MLP"].intercepts_[i] = student["MLP"].intercepts_[i].astype(np.float16)

        print("Student accuracy: {:.2f}%".format(100 * student.score(train.data, train.target)))
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(student, model_file)

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
