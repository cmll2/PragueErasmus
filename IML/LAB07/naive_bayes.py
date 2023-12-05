#!/usr/bin/env python3
import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian", choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    print(train_target[:20])
    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.

    prior_probabilities = []
    for i in range(args.classes):
        prior_probabilities.append(np.sum(train_target == i) / len(train_target))

    if args.naive_bayes_type == "gaussian": #training
        means = []
        variances = []
        for class_i in range(args.classes):
            class_data = train_data[train_target == class_i]
            class_mean = np.mean(class_data, axis=0)
            class_variance = (1/len(class_data)) * np.sum((class_data - class_mean)**2, axis=0) + args.alpha
            means.append(class_mean)
            variances.append(class_variance)
    elif args.naive_bayes_type == "multinomial":
        class_feature_counts = []
        for class_i in range(args.classes):
            class_data = train_data[train_target == class_i]
            class_feature_counts.append(np.sum(class_data, axis=0))
        prob_feature_given_class = [(class_feature_counts[i] + args.alpha) / (np.sum(class_feature_counts[i]) + args.alpha * train_data.shape[1]) for i in range(args.classes)]

    elif args.naive_bayes_type == "bernoulli":
        binarized_train_data = (train_data >= 8).astype(int)
        class_feature_counts = []
        for class_i in range(args.classes):
            class_data = binarized_train_data[train_target == class_i]
            class_feature_counts.append(np.sum(class_data, axis=0))
        prob_feature_given_class = [(class_feature_counts[i] + args.alpha) / (np.sum(train_target == i) + args.alpha * 2) for i in range(args.classes)]

    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    from scipy.stats import norm
    predicted_classes = []
    joint_log_probability = 0
    if args.naive_bayes_type == "gaussian":
        for sample, target in zip(test_data, test_target):
            probabilities = [np.sum(np.log(norm.pdf(sample, loc=means[c], scale=np.sqrt(variances[c])))) + np.log(prior_probabilities[c]) for c in range(args.classes)]
            #print(probabilities)
            predicted_classes.append(np.argmax(probabilities))
            joint_log_probability += (probabilities[target])
    elif args.naive_bayes_type == "multinomial":
        for sample, target in zip(test_data, test_target):
            probabilities = [np.sum(np.log(prob_feature_given_class[c] ** sample)) + np.log(prior_probabilities[c]) for c in range(args.classes)]
            predicted_classes.append(np.argmax(probabilities))
            joint_log_probability += (probabilities[target])
    elif args.naive_bayes_type == "bernoulli":
        for sample, target in zip(test_data, test_target):
            binarized_sample = (sample >= 8).astype(int)
            probabilities = [np.sum(np.log(np.where(binarized_sample, prob_feature_given_class[c], 1 - prob_feature_given_class[c])))+ np.log(prior_probabilities[c]) for c in range(args.classes)]
            predicted_classes.append(np.argmax(probabilities))
            joint_log_probability += (probabilities[target])
    test_accuracy, test_log_probability = np.mean(predicted_classes == test_target), joint_log_probability

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(test_accuracy, test_log_probability))
