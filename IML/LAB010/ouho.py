#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.tree import DecisionTreeRegressor
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)
    #one hot encode target
    classes = np.max(target) + 1
    target = np.eye(np.max(target) + 1)[target]
    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create gradient boosted trees on the classification training data.
    #
    # Notably, train for `args.trees` iteration. During the iteration `t`:
    # - the goal is to train `classes` regression trees, each predicting
    #   a part of logit for the corresponding class.
    # - compute the current predictions `y_{t-1}(x_i)` for every training example `i` as
    #     y_{t-1}(x_i)_c = \sum_{j=1}^{t-1} args.learning_rate * tree_{iter=j,class=c}.predict(x_i)
    #   (note that y_0 is zero)
    # - loss in iteration `t` is
    #     E = (\sum_i NLL(onehot_target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
    #         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
    # - for every class `c`:
    #   - start by computing `g_i` and `h_i` for every training example `i`;
    #     the `g_i` is the first and the `h_i` is the second derivative of
    #     NLL(onehot_target_i_c, softmax(y_{t-1}(x_i))_c) with respect to y_{t-1}(x_i)_c.
    #   - then, create a decision tree minimizing the loss E. According to the slides,
    #     the optimum prediction for a given node T with training examples I_T is
    #       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
    #     and the value of the loss with this prediction is
    #       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
    #     which you should use as a splitting criterion.
    #
    # During tree construction, we split a node if:
    # - its depth is less than `args.max_depth`
    # - there is more than 1 example corresponding to it (this was covered by
    #     a non-zero criterion value in the previous assignments)

    trees_to_train = args.trees
    learning_rate = args.learning_rate
    l2 = args.l2
    max_depth = args.max_depth

    y_pred = np.zeros((len(train_data), classes))
    decision_trees = []
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    # Create the {classes} first decision trees
    decision_trees_classes = []
    for c in range(classes):
        decision_trees_classes.append(DecisionTreeRegressor(max_depth=max_depth))
        decision_trees_classes[c].fit(train_data, train_target[:, c])

    decision_trees.append(decision_trees_classes)

    # Training loop
    for t in range(1, trees_to_train):
        decision_trees_classes = []
        for c in range(classes):
            # Compute the current predictions y_{t-1}(x_i) for every training example i
            y_pred[:, c] += learning_rate * decision_trees[t - 1][c].predict(train_data)

            # Compute the softmax and then g and h using the loss function above
            softmax = sigmoid(y_pred[:, c])

            g = softmax - train_target[:, c]
            h = softmax * (1 - softmax)

            # Create a decision tree minimizing the loss E
            decision_trees_classes.append(DecisionTreeRegressor(max_depth=max_depth))
            decision_trees_classes[c].fit(train_data, -g / (l2 + (h)))

        decision_trees.append(decision_trees_classes)
        

        # TODO: Finally, measure your training and testing accuracies when
        # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (pick the smallest class number if there is a tie).
    train_accuracies = []
    test_accuracies = []
    y_pred_train = np.zeros((len(train_data), classes), dtype=np.float64)
    y_pred_test = np.zeros((len(test_data), classes), dtype=np.float64)
    for t in range(1, trees_to_train + 1):
        # Predict using t trees
        for c in range(classes):
            y_pred_train[:, c] += decision_trees[t - 1][c].predict(train_data)
            y_pred_test[:, c] += decision_trees[t - 1][c].predict(test_data)
        print(y_pred_train[:10])
        # Compute the training and testing accuracies
        train_accuracy = np.mean(np.argmax(y_pred_train, axis=1) == np.argmax(train_target, axis=1))

        test_accuracy = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(test_target, axis=1))

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))