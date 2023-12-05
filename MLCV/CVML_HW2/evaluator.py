
import argparse
from typing import Union, Sequence, Tuple
import numpy as np
import hw2

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--test", default="independent", type=str, help="Hypothesis/Classifier test to evaluate: 'independent', 'paired', 'corrected', 'mcnemar'.")
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")
parser.add_argument("--x_scatter", default=0.25, type=float, help="X scatter of the generated data.")
parser.add_argument("--train_size", default=200, type=int, help="Number of points in each class in the train set.")
parser.add_argument("--test_size", default=150, type=int, help="Number of points in each class in the test set.")
parser.add_argument("--k_neighbors", default=9, type=int, help="Number of points for KNN classification.")
parser.add_argument("--paired_k_splits", default=3, type=int, help="Number of splits used in the paired t-test.")
parser.add_argument("--corrected_kfold_splits", default=3, type=int, help="Number of folds considered in the corrected resampled t-test.")
parser.add_argument("--confidence_alpha", default=0.05, type=float, help="The alpha value considered when working with confidence of hypothesis rejection.")

def generateData(generator : np.random.RandomState, point_count : Union[Sequence, int], x_scatter : Union[Sequence, float], positions : Sequence = [1, 2, 3], scale : Sequence = [5, 5, 5]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates random data with custom point count and normal distribution in the X axis.
    Default parameters create three sets, each having a different label.
    Labels are marked from 0 to N - 1, where N is the number of generated sets.

    Arguments:
    - generator - Random number generator used for data generation.
    - pointCount - Number of points in each set. It can be a list or an int.
    - xScatter - Standard deviation in the X axis for the generated data. It can be a list or a float.
    - positions - Mean values of the generated sets. It has to be a sequence.
    - scale - Scale of the generated sets in the Y axis. It has to be a sequence.

    Returns:
    - [X, Y] coordinates of the generated data points (np.ndarray).
    - Labels of the generated data points (np.ndarray).
    """
    size = len(positions) if len(positions) == len(scale) else None
    if size is None:
        raise ValueError("Incorrect size of input arrays.")
    
    point_count = [point_count] * size if isinstance(point_count, int) else point_count
    x_scatter = [x_scatter] * size if isinstance(x_scatter, float) else x_scatter
    x_data = [positions[i] + x_scatter[i] * generator.randn(point_count[i]) for i in range(size)]
    x_data = np.hstack(x_data)

    y_data = [scale[i] * generator.rand(point_count[i]) for i in range(size)]
    y_data = np.hstack(y_data)

    labels = [np.ones(point_count[i]) * (i + 1) for i in range(size)]
    labels = np.hstack(labels)

    return np.vstack([x_data, y_data]).T, labels


def independentTest(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)
    train_data_nb, train_labels_nb = generateData(generator, args.train_size, args.x_scatter)
    train_data_knn, train_labels_knn = generateData(generator, args.train_size, args.x_scatter)
    test_data_nb, test_labels_nb = generateData(generator, args.test_size, args.x_scatter)
    test_data_knn, test_labels_knn = generateData(generator, args.test_size, args.x_scatter)
    train_set = {
        "data_nb" : train_data_nb,
        "labels_nb" : train_labels_nb,
        "data_knn" : train_data_knn,
        "labels_knn" : train_labels_knn,
    }
    test_set = {
        "data_nb" : test_data_nb,
        "labels_nb" : test_labels_nb,
        "data_knn" : test_data_knn,
        "labels_knn" : test_labels_knn,
    }
    tester = hw2.IndependentTest(args.confidence_alpha, args.k_neighbors, knn_metric="mahalanobis", knn_metric_params={'V' : [[0.1, 0], [0, 2]]})
    results = tester.compute(train_set, test_set)

    print("Error difference: {:.5f}, and error values NB: {:.5f}, KNN: {:.5f}.".format(results[hw2.Results.DIFF_ERR], results[hw2.Results.ERR_NB], results[hw2.Results.ERR_KNN]))
    print("Critical value of normal distribution (two-tailed) for alpha={:.2f} is {:.5f}.".format(args.confidence_alpha, results[hw2.Results.CRIT_VALUE]))
    print("Interval of the true error: {:.5f} +- {:.5f}".format(results[hw2.Results.DIFF_ERR], results[hw2.Results.HALF_INTERVAL]))
    print("Can we reject H0?, i.e. is 0 outside of the true error difference interval?: {}.".format(results[hw2.Results.REJECT_H0]))

def pairedTest(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)
    train_data_nb, train_labels_nb = generateData(generator, args.train_size, args.x_scatter)
    train_data_knn, train_labels_knn = generateData(generator, args.train_size, args.x_scatter)
    test_data, test_labels = generateData(generator, args.test_size, args.x_scatter)
    test_permut = generator.permutation(test_labels.size)
    train_set = {
        "data_nb" : train_data_nb,
        "labels_nb" : train_labels_nb,
        "data_knn" : train_data_knn,
        "labels_knn" : train_labels_knn,
    }
    test_set = {
        "data" : test_data[test_permut],
        "labels" : test_labels[test_permut],
    }
    tester = hw2.PairedTTest(args.paired_k_splits, args.confidence_alpha, args.k_neighbors, knn_metric="mahalanobis", knn_metric_params={'V' : [[0.1, 0], [0, 2]]})
    results = tester.compute(train_set, test_set)

    print("SE - variance of errors: {:.5f}".format(results[hw2.Results.SE]))
    print("Critical value of the Student's t-distribution (two-tailed) for alpha={:.2f} is {:.5f}".format(args.confidence_alpha, results[hw2.Results.CRIT_VALUE]))
    print("Interval of the true error difference: {:.5f} +- {:.5f}".format(results[hw2.Results.DIFF_MEAN], results[hw2.Results.HALF_INTERVAL]))
    print("Can we reject H0?, i.e. is 0 outside of the true error difference interval?: {}.".format(results[hw2.Results.REJECT_H0]))

def correctedResampledTest(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)
    data, labels = generateData(generator, args.train_size, args.x_scatter)
    permut = generator.permutation(labels.size)
    tester = hw2.CorrectedResampledTTest(args.corrected_kfold_splits, args.confidence_alpha, args.k_neighbors, knn_metric="mahalanobis", knn_metric_params={'V' : [[0.1, 0], [0, 2]]})
    results = tester.compute(data[permut], labels[permut])

    print("Corrected SE - variance of errors: {:.5f}".format(results[hw2.Results.SE]))
    print("Critical value of the Student's t-distribution (two-tailed) for alpha={:.2f} is {:.5f}.".format(args.confidence_alpha, results[hw2.Results.CRIT_VALUE]))
    print("Interval of the true error difference: {:.5f} +- {:.5f}".format(results[hw2.Results.DIFF_MEAN], results[hw2.Results.HALF_INTERVAL]))
    print("Can we reject H0?, i.e. is 0 outside of the true error difference interval?: {}.".format(results[hw2.Results.REJECT_H0]))

def mcnemarTest(args : argparse.Namespace):
    generator = np.random.RandomState(args.seed)
    train_data_nb, train_labels_nb = generateData(generator, args.train_size, args.x_scatter)
    train_data_knn, train_labels_knn = generateData(generator, args.train_size, args.x_scatter)
    test_data, test_labels = generateData(generator, args.test_size, args.x_scatter)
    train_set = {
        "data_nb" : train_data_nb,
        "labels_nb" : train_labels_nb,
        "data_knn" : train_data_knn,
        "labels_knn" : train_labels_knn,
    }
    test_set = {
        "data" : test_data,
        "labels" : test_labels,
    }
    tester = hw2.McNemarTest(args.confidence_alpha, args.k_neighbors, knn_metric="mahalanobis", knn_metric_params={'V' : [[0.1, 0], [0, 2]]})
    results = tester.compute(train_set, test_set)

    print("Marginal homogeneity M: {:.5f}".format(results[hw2.Results.M]))
    print("Critical value of chi2 distribution (single-tailed) with 1 DoF and alpha={:.2f} is {:.5f}".format(args.confidence_alpha, results[hw2.Results.CRIT_VALUE]))
    print("The number of different decisions of the classifier: {}, is the result meaningful?: {}".format(results[hw2.Results.DENOMINATOR], results[hw2.Results.DENOMINATOR] >= 25))
    print("Can we reject H0?, i.e. is M greater than the critical value?: {}.".format(results[hw2.Results.REJECT_H0]))

def main(args : argparse.Namespace) -> None:
    tests = {
        "independent" : independentTest,
        "paired" : pairedTest,
        "corrected" : correctedResampledTest,
        "mcnemar" : mcnemarTest,
    }
    if args.test not in tests.keys():
        raise ValueError("Requested hypothesis/classifier test is unknown: {}!".format(args.test))
    tests[args.test](args)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
