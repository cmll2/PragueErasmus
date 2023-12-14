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
def softmax(x):
    x_exp = np.exp(x)
    return x_exp / np.sum(x_exp, axis=1).reshape(-1,1)

class Leaf:
    def __init__(self, value):
        self.value = value
    def predict(self, x):
        return self.value
    
class Node:
    def __init__(self, feature, split, left, right, value=None):
        self.feature = feature
        self.split = split
        self.left = left
        self.right = right
        self.value = value
        
    def predict(self, x):
        if self.feature is None:
            return self.value
        if x[self.feature] <= self.split:
            return self.left.predict(x)
        else:
            return self.right.predict(x)
        
class Tree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = Node(None, None, None, None)

    def criterion(self, grad, hess, l2):
        return -0.5 * (grad ** 2 / (hess + l2))

    def fit(self, X, g, h, l2):
        self.root = self.__fit(X, g, h, l2, 0)

    def __fit(self, X, g, h, l2, depth):
        grad = g.sum()
        hess = h.sum()
        if depth == self.max_depth or len(X) == 1:
            return Leaf(-grad / (hess + l2))
        else:
            best_feature, best_split, best_gain = None, None, np.inf
            for feature in range(X.shape[1]):
                unique_values = np.unique(X[:, feature])
                for i in range(len(unique_values) - 1):
                    split = (unique_values[i] + unique_values[i + 1]) / 2
                    mask = X[:, feature] <= split
                    gain = self.criterion(g[mask].sum(), h[mask].sum(), l2) + self.criterion(g[~mask].sum(), h[~mask].sum(), l2)
                    if gain < best_gain:
                        best_feature, best_split, best_gain = feature, split, gain
            left_indices = X[:, best_feature] <= best_split
            right_indices = X[:, best_feature] > best_split
            if left_indices.sum() == 0 or right_indices.sum() == 0:
                return Leaf(-grad / (hess + l2))
            left = self.__fit(X[left_indices], g[left_indices], h[left_indices], l2, depth + 1)
            right = self.__fit(X[right_indices], g[right_indices], h[right_indices], l2, depth + 1)
            return Node(best_feature, best_split, left, right)
        
    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])
        
class GBTrees:
    def __init__(self, trees_depth, learning_rate, l2, max_depth):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_depth = max_depth
        self.trees = trees_depth

    def fit(self, X, y):
        self.classes = int(np.max(y) + 1)
        self.boosted_trees = [[Tree(self.max_depth) for _ in range(self.classes)] for _ in range(self.trees)]

        one_hot = np.eye(self.classes)[y]
        for depth in range(self.trees):
            prediction = self.predict(X, depth)
            g = prediction - one_hot
            h = prediction * (1 - prediction)
            for c in range(self.classes):
                self.boosted_trees[depth][c].fit(X, g[:, c], h[:, c], self.l2)

    def predict(self, X, depth):
        predictions = np.zeros((len(X), self.classes))
        for c in range(self.classes):
            for d in range(depth):
                predictions[:, c] += self.learning_rate * self.boosted_trees[d][c].predict(X)
        return softmax(predictions)
    
    def true_predict(self, X, depth):
        return np.argmax(self.predict(X, depth), axis=1)

def plot_tree(tree):
    import matplotlib.pyplot as plt
    def _plot_tree(tree, node, depth, pos, x_pos, y_pos, parent=None):
        if isinstance(node, Leaf):
            plt.text(x_pos, y_pos, node.value, bbox=dict(boxstyle="circle", fc="white", ec="black", lw=1, alpha=0.5))
        else:
            plt.text(x_pos, y_pos, node.split, bbox=dict(boxstyle="square", fc="white", ec="black", lw=1, alpha=0.5))
        if parent is not None:
            plt.plot([x_pos, pos[parent][0]], [y_pos, pos[parent][1]], "k-")
        if not isinstance(node, Leaf) and node.left is not None:
            pos[id(node.left)] = (x_pos - 2**(max_depth-depth-1), y_pos - 1)
            _plot_tree(tree, node.left, depth+1, pos, x_pos - 2**(max_depth-depth-1), y_pos - 1, id(node))
        if not isinstance(node, Leaf) and node.right is not None:
            pos[id(node.right)] = (x_pos + 2**(max_depth-depth-1), y_pos - 1)
            _plot_tree(tree, node.right, depth+1, pos, x_pos + 2**(max_depth-depth-1), y_pos - 1, id(node))
    max_depth = tree.max_depth
    plt.figure(figsize=(2**max_depth*5, max_depth*5))
    plt.axis("off")
    pos = {id(tree.root): (2**(max_depth-1), max_depth-1)}
    _plot_tree(tree, tree.root, 0, pos, 2**(max_depth-1), max_depth-1)
    plt.show()

def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

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

    gradient_boosted_trees = GBTrees(trees_to_train, learning_rate, l2, max_depth)
    gradient_boosted_trees.fit(train_data, train_target)
    
    # TODO: Finally, measure your training and testing accuracies when
    # using 1, 2, ..., `args.trees` of the created trees.
    #
    # To perform a prediction using t trees, compute the y_t(x_i) and return the
    # class with the highest value (pick the smallest class number if there is a tie).
    from sklearn.metrics import accuracy_score
    train_accuracies = [accuracy_score(train_target, gradient_boosted_trees.true_predict(train_data, depth + 1)) for depth in range(args.trees)]
    test_accuracies = [accuracy_score(test_target, gradient_boosted_trees.true_predict(test_data, depth + 1)) for depth in range(args.trees)]
    print(gradient_boosted_trees.true_predict(test_data, 1)[:10])

    # plot trees
    # for depth in range(args.trees):
    #   for c in range(gradient_boosted_trees.classes):
    #        plot_tree(gradient_boosted_trees.boosted_trees[depth][c])


    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))
