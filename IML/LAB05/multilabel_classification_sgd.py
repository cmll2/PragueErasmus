#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import MultiLabelBinarizer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int, help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)
    # Convert the list of lists to a dense binary matrix.
    mlb = MultiLabelBinarizer(classes = range(args.classes))
    target = mlb.fit_transform(target_list)
    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for i in range(0, train_data.shape[0], args.batch_size):
            batch = train_data[permutation[i:i + args.batch_size]]
            target_batch = train_target[permutation[i:i + args.batch_size]]
            batch_predict = 1 / (1 + np.exp(-batch @ weights))
            gradient = batch.T @ (batch_predict - target_batch) / args.batch_size
            weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        train_prediction = train_data @ weights
        test_prediction = test_data @ weights
        train_true_positives = []
        test_true_positives = []
        train_false_positives = []
        test_false_positives = []
        train_false_negatives = []
        test_false_negatives = []
        print(train_prediction)
        for i in range (args.classes):
            train_true_positives.append(np.sum(np.logical_and(train_prediction[:, i] > 0, train_target[:, i] == 1)))
            test_true_positives.append(np.sum(np.logical_and(test_prediction[:, i] > 0, test_target[:, i] == 1)))
            train_false_positives.append(np.sum(np.logical_and(train_prediction[:, i] > 0, train_target[:, i] == 0)))
            test_false_positives.append(np.sum(np.logical_and(test_prediction[:, i] > 0, test_target[:, i] == 0)))
            train_false_negatives.append(np.sum(np.logical_and(train_prediction[:, i] < 0, train_target[:, i] == 1)))
            test_false_negatives.append(np.sum(np.logical_and(test_prediction[:, i] < 0, test_target[:, i] == 1)))
        train_precision = [train_true_positives[i] / (train_true_positives[i] + train_false_positives[i])  if train_true_positives[i] + train_false_positives[i] != 0 else 0 for i in range(args.classes)]
        test_precision = [test_true_positives[i] / (test_true_positives[i] + test_false_positives[i]) if test_true_positives[i] + test_false_positives[i] != 0 else 0 for i in range(args.classes)]
        train_recall = [train_true_positives[i] / (train_true_positives[i] + train_false_negatives[i]) if train_true_positives[i] + train_false_negatives[i] != 0 else 0 for i in range(args.classes)]
        test_recall = [test_true_positives[i] / (test_true_positives[i] + test_false_negatives[i]) if test_true_positives[i] + test_false_negatives[i] != 0 else 0 for i in range(args.classes)]
        train_f1_scores = [2 * train_precision[i] * train_recall[i] / (train_precision[i] + train_recall[i]) if train_precision[i] + train_recall[i] != 0 else 0 for i in range(args.classes)]
        test_f1_scores = [2 * test_precision[i] * test_recall[i] / (test_precision[i] + test_recall[i]) if test_precision[i] + test_recall[i] != 0 else 0 for i in range(args.classes)]

        train_f1_macro = np.mean(train_f1_scores)
        test_f1_macro = np.mean(test_f1_scores)

        train_f1_micro = 2*(np.sum(train_true_positives)) / (2*np.sum(train_true_positives) + np.sum(train_false_positives) + np.sum(train_false_negatives))
        test_f1_micro = 2*(np.sum(test_true_positives)) / (2*np.sum(test_true_positives) + np.sum(test_false_positives) + np.sum(test_false_negatives))



        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
