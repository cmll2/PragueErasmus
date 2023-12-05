#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1.0, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=44, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

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
        
class DecisionTree:
    def __init__(self, criterion, max_depth, min_to_split = 0, max_leaves = None, subsampler = None):
        self.criterion_method = criterion
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves
        self.root = None
        self.num_leaves = 0
        self.subsampler = subsampler
        
    def criterion(self, target):
        if self.criterion_method == "gini":
            probs = np.unique(target, return_counts=True)[1] / len(target)
            return len(target) * (probs * (1 - probs)).sum()
        elif self.criterion_method == "entropy":
            probs = np.unique(target, return_counts=True)[1] / len(target)
            return len(target) * (-probs * np.log2(probs)).sum()

    def fit(self, data, target):
        if self.subsampler is None:
            self.root = self._fit(data, target, 0)
        else:
            self.root = self._subsampler_fit(data, target, 0)
        
    def _fit(self, data, target, depth): #implementing the recursive approach if no args.max_leaves
        if self.max_leaves is None:
            if len(data) < self.min_to_split or self.criterion(target) == 0:
                self.num_leaves += 1
                return Leaf(np.bincount(target).argmax())
            if self.max_depth is not None and depth >= self.max_depth:
                self.num_leaves += 1
                return Leaf(np.bincount(target).argmax())
            best_feature = None
            best_split = None
            best_criterion = None
            for feature in range(data.shape[1]):
                unique_values = np.unique(data[:, feature])
                for i in range(len(unique_values)-1):
                    split = (unique_values[i] + unique_values[i+1]) / 2
                    left = data[:, feature] <= split
                    right = data[:, feature] > split
                    left_criterion = self.criterion(target[left])
                    right_criterion = self.criterion(target[right])
                    criterion = left_criterion + right_criterion - self.criterion(target)
                    if best_criterion is None or criterion < best_criterion:
                        best_feature = feature
                        best_split = split
                        best_criterion = criterion
            left = data[:, best_feature] <= best_split
            right = data[:, best_feature] > best_split
            left_node = self._fit(data[left], target[left], depth+1)
            right_node = self._fit(data[right], target[right], depth+1)
            return Node(best_feature, best_split, left_node, right_node)
        else: 
            root = Node(None, None, None, None, np.bincount(target).argmax())
            leaves = [(data, target, root, 0)]
            depth = 0
            while len(leaves)<self.max_leaves:
                best_criterion = np.inf
                best_feature = None
                best_split = None
                best_left = None
                best_right = None
                best_leaf = None
                split = False
                for index, leaf in enumerate(leaves):
                    data, target, node, leaf_depth = leaf
                    if len(data) >= self.min_to_split and self.criterion(target) != 0:
                        #check depth
                        if self.max_depth is None or leaf_depth < self.max_depth:
                            #compute criterion
                            for feature in range(data.shape[1]):
                                unique_values = np.unique(data[:, feature])
                                for i in range(len(unique_values)-1):
                                    split = (unique_values[i] + unique_values[i+1]) / 2
                                    left = data[:, feature] <= split
                                    right = data[:, feature] > split
                                    left_criterion = self.criterion(target[left])
                                    right_criterion = self.criterion(target[right])
                                    criterion = left_criterion + right_criterion - self.criterion(target)
                                    if best_criterion is None or criterion < best_criterion:
                                        best_feature = feature
                                        best_split = split
                                        best_criterion = criterion
                                        best_left = left
                                        best_right = right
                                        best_leaf = index
                                        depth = leaf_depth
                                        split = True
                        else:
                            continue
                    else:
                        continue
                if not split:
                    break
                else:
                    left = best_left
                    right = best_right
                    data_left, target_left = leaves[best_leaf][0][left], leaves[best_leaf][1][left]
                    data_right, target_right = leaves[best_leaf][0][right], leaves[best_leaf][1][right]
                    node = leaves[best_leaf][2]
                    leaves.pop(best_leaf)
                    left_node = Node(None, None, None, None, np.bincount(target_left).argmax())
                    right_node = Node(None, None, None, None, np.bincount(target_right).argmax())
                    node.feature = best_feature
                    node.split = best_split
                    node.left = left_node
                    node.right = right_node
                    print(best_leaf)
                    leaves.append((data_left, target_left, left_node, depth+1))
                    leaves.append((data_right, target_right, right_node, depth+1))
            return root
        
    def _subsampler_fit(self, data, target, depth): #implementing the recursive approach if args.max_leaves
        if self.max_leaves is None:
            node_criterion = self.criterion(target)
            if len(data) < self.min_to_split or node_criterion == 0:
                self.num_leaves += 1
                return Leaf(np.bincount(target).argmax())
            if self.max_depth is not None and depth >= self.max_depth:
                self.num_leaves += 1
                return Leaf(np.bincount(target).argmax())
            best_feature = None
            best_split = None
            best_criterion = None
            subsampled_features = self.subsampler(data.shape[1])
            for feature in subsampled_features:
                unique_values = np.unique(data[:, feature])
                for i in range(len(unique_values)-1):
                    split = (unique_values[i] + unique_values[i+1]) / 2
                    left = data[:, feature] <= split
                    right = data[:, feature] > split
                    left_criterion = self.criterion(target[left])
                    right_criterion = self.criterion(target[right])
                    criterion = left_criterion + right_criterion - node_criterion
                    if best_criterion is None or criterion < best_criterion:
                        best_feature = feature
                        best_split = split
                        best_criterion = criterion
                        best_left = left
                        best_right = right
            left_node = self._subsampler_fit(data[best_left], target[best_left], depth+1)
            right_node = self._subsampler_fit(data[best_right], target[best_right], depth+1)
            return Node(best_feature, best_split, left_node, right_node)
        else: 
            root = Node(None, None, None, None, np.bincount(target).argmax())
            leaves = [(data, target, root, 0)]
            depth = 0
            while len(leaves)<self.max_leaves:
                best_criterion = np.inf
                best_feature = None
                best_split = None
                best_left = None
                best_right = None
                best_leaf = None
                split = False
                for index, leaf in enumerate(leaves):
                    data, target, node, leaf_depth = leaf
                    if len(data) >= self.min_to_split and self.criterion(target) != 0:
                        #check depth
                        if self.max_depth is None or leaf_depth < self.max_depth:
                            #compute criterion
                            subsampled_features = self.subsampler(data.shape[1])
                            for feature in subsampled_features:
                                unique_values = np.unique(data[:, feature])
                                for i in range(len(unique_values)-1):
                                    split = (unique_values[i] + unique_values[i+1]) / 2
                                    left = data[:, feature] <= split
                                    right = data[:, feature] > split
                                    left_criterion = self.criterion(target[left])
                                    right_criterion = self.criterion(target[right])
                                    criterion = left_criterion + right_criterion - self.criterion(target)
                                    if best_criterion is None or criterion < best_criterion:
                                        best_feature = feature
                                        best_split = split
                                        best_criterion = criterion
                                        best_left = left
                                        best_right = right
                                        best_leaf = index
                                        depth = leaf_depth
                                        split = True
                        else:
                            continue
                    else:
                        continue
                if not split:
                    break
                else:
                    left = best_left
                    right = best_right
                    data_left, target_left = leaves[best_leaf][0][left], leaves[best_leaf][1][left]
                    data_right, target_right = leaves[best_leaf][0][right], leaves[best_leaf][1][right]
                    node = leaves[best_leaf][2]
                    leaves.pop(best_leaf)
                    left_node = Node(None, None, None, None, np.bincount(target_left).argmax())
                    right_node = Node(None, None, None, None, np.bincount(target_right).argmax())
                    node.feature = best_feature
                    node.split = best_split
                    node.left = left_node
                    node.right = right_node
                    leaves.append((data_left, target_left, left_node, depth+1))
                    leaves.append((data_right, target_right, right_node, depth+1))
            return root


    def predict(self, data):
        return np.array([self.root.predict(x) for x in data])
    
    def depth(self):
        return self._depth(self.root)
    
    def _depth(self, node):
        if isinstance(node, Leaf):
            return 0
        elif node.left is None and node.right is None:
            return 0
        return 1 + max(self._depth(node.left), self._depth(node.right))
    
    def leaves(self):
        return self._leaves(self.root)
    
    def _leaves(self, node):
        if isinstance(node, Leaf):
            return 1
        elif node.left is None and node.right is None:
            return 1
        return self._leaves(node.left) + self._leaves(node.right)
    
    def nodes(self):
        return self._nodes(self.root)
    
    def _nodes(self, node):
        if isinstance(node, Leaf):
            return 0
        return 1 + self._nodes(node.left) + self._nodes(node.right)
    
def plot_tree(tree):
    import matplotlib.pyplot as plt
    def _plot_tree(tree, node, depth, pos, x_pos, y_pos, parent=None):
        if isinstance(node, Leaf):
            plt.text(x_pos, y_pos, node.value, bbox=dict(boxstyle="circle", fc="white", ec="black", lw=1, alpha=0.5))
        else:
            plt.text(x_pos, y_pos, node.feature, bbox=dict(boxstyle="square", fc="white", ec="black", lw=1, alpha=0.5))
        if parent is not None:
            plt.plot([x_pos, pos[parent][0]], [y_pos, pos[parent][1]], "k-")
        if not isinstance(node, Leaf) and node.left is not None:
            pos[id(node.left)] = (x_pos - 2**(max_depth-depth-1), y_pos - 1)
            _plot_tree(tree, node.left, depth+1, pos, x_pos - 2**(max_depth-depth-1), y_pos - 1, id(node))
        if not isinstance(node, Leaf) and node.right is not None:
            pos[id(node.right)] = (x_pos + 2**(max_depth-depth-1), y_pos - 1)
            _plot_tree(tree, node.right, depth+1, pos, x_pos + 2**(max_depth-depth-1), y_pos - 1, id(node))
    max_depth = tree.depth()
    plt.figure(figsize=(2**max_depth, max_depth))
    plt.axis("off")
    pos = {id(tree.root): (2**(max_depth-1), max_depth-1)}
    _plot_tree(tree, tree.root, 0, pos, 2**(max_depth-1), max_depth-1)
    plt.show()

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators.
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features: int) -> np.ndarray:
        return np.sort(generator_feature_subsampling.choice(
            number_of_features, size=int(args.feature_subsampling * number_of_features), replace=False))

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data: np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    # TODO: Create a random forest on the training data.
    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, to split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targets are not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in the left subtree before the nodes in right subtree.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating the subsampled features using
    #     subsample_features(number_of_features)
    #   returning the features that should be used during the best split search.
    #   The features are returned in ascending order, so when `feature_subsampling == 1`,
    #   the `np.arange(number_of_features)` is returned.
    #
    # - train a random forest consisting of `args.trees` decision trees
    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with the smallest class number in case of a tie.

    decision_trees = []
    for _ in range(args.trees):
        if args.bagging:
            dataset_indices = bootstrap_dataset(train_data)
            decision_tree  = DecisionTree(
                criterion="entropy", max_depth=args.max_depth, subsampler=subsample_features)
            decision_tree.fit(train_data[dataset_indices], train_target[dataset_indices])
            decision_trees.append(decision_tree)
        else:
            decision_tree  = DecisionTree(
                criterion="entropy", max_depth=args.max_depth, subsampler=subsample_features)
            decision_tree.fit(train_data, train_target)
            decision_trees.append(decision_tree)
    
    def predict(data: np.ndarray) -> np.ndarray:
        predictions = np.zeros((len(data), args.trees))
        for i, decision_tree in enumerate(decision_trees):
            predictions[:, i] = decision_tree.predict(data)
        #convert predictions to int
        predictions = predictions.astype(int)
        return np.array([np.argmax(np.bincount(prediction)) for prediction in predictions])



    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy, test_accuracy = sklearn.metrics.accuracy_score(train_target, predict(train_data)), sklearn.metrics.accuracy_score(test_target, predict(test_data))
    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
