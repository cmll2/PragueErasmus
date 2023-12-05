#!/usr/bin/env python3
import argparse

import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.
import numpy as np
    
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
    def __init__(self, criterion, max_depth, min_to_split = 0, max_leaves = None):
        self.criterion_method = criterion
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves
        self.root = None
        self.num_leaves = 0
        
    def criterion(self, target):
        if self.criterion_method == "gini":
            probs = np.unique(target, return_counts=True)[1] / len(target)
            return len(target) * (probs * (1 - probs)).sum()
        elif self.criterion_method == "entropy":
            probs = np.unique(target, return_counts=True)[1] / len(target)
            return len(target) * (-probs * np.log2(probs)).sum()

    def fit(self, data, target):
        self.root = self._fit(data, target, 0)
        
    def _fit(self, data, target, depth): #implementing the recursive approach if no args.max_leaves
        if self.max_leaves is None:
            if len(data) < self.min_to_split or self.criterion(target) == 0:
                self.num_leaves += 1
                return Leaf(np.bincount(target).argmax())
            if self.max_depth is not None and depth >= self.max_depth:
                self.num_leaves += 1
                return Leaf(np.bincount(target).argmax())
            if self.max_leaves is not None and self.num_leaves >= self.max_leaves-1:
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

    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   the smallest number if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3, the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not `None`, its depth must be less than `args.max_depth`
    #     (depth of the root node is zero);
    #   - when `args.max_leaves` is not `None`, there are less than `args.max_leaves` leaves
    #     (a leaf is a tree node without children);
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is `None`, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not `None`), repeatably split a leaf where the
    #   constraints are valid and the overall criterion value ($c_left + c_right - c_node$)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).
    tree = DecisionTree(criterion=args.criterion, max_depth=args.max_depth, min_to_split=args.min_to_split, max_leaves=args.max_leaves)
    tree.fit(train_data, train_target) 
    print(tree.leaves())
    plot_tree(tree)
    train_predict = tree.predict(train_data)
    test_predict = tree.predict(test_data)
    # TODO: Finally, measure the training and testing accuracy.
    train_accuracy, test_accuracy = sklearn.metrics.accuracy_score(train_target, train_predict), sklearn.metrics.accuracy_score(test_target, test_predict)

    return 100 * train_accuracy, 100 * test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
