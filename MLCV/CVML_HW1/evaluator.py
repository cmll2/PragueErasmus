
import argparse
import time
from typing import Tuple
import numpy as np
import sklearn.mixture
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import hw1

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--data", default="debug", type=str,
                    help="Dataset used in the optimisation: 'debug' for simple example with hard-coded initial optimisation means, 'test' for data with three components and random initialisation.")
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG.")
parser.add_argument("--points", default=50, type=int, help="Total number of points for data generation.")

parser.add_argument("--n_components", default=2, type=int, help="Number of requested GMM components.")
parser.add_argument("--tol", default=0.001, type=float, help="Tolerance used as a stopping convergence criterion.")
parser.add_argument("--max_iter", default=100, type=int, help="Maximum number of allowed iterations.")

def generateTwoComponents(generator : np.random.RandomState, args : argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates simple 2 component data and defines initial means from which the algorithm should
    not diverge. This should be used for debugging.
    """
    mu1, sigma1, weight1 = [0, 0], [[1.0, 0.0], [0.0, 1.0]], 0.5
    data1 = generator.multivariate_normal(mu1, sigma1, int(weight1 * args.points))

    mu2, sigma2, weight2 = [4.0, 3.0], [[2.0, 0.0], [0.0, 2.0]], 0.5
    data2 = generator.multivariate_normal(mu2, sigma2, int(weight2 * args.points))

    data = np.vstack([data1, data2])
    labels = np.hstack([[0] * data1.shape[0], [1] * data2.shape[0]])
    init_means = np.asarray([[-4.0, -3.0], [6.0, 7.0]])

    return data, labels, init_means

def generateThreeComponents(generator : np.random.RandomState, args : argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates data with 3 components which should be used for more involved testing of the algorithm.
    Feel free to change the following parameters and observe how the behaviour of the algorithm changes.
    This dataset expects random mean initialisations.
    """
    mu1, sigma1, weight1 = [0.0, 0.0], [[1.0, 0.5], [0.5, 0.5]], 0.35
    data1 = generator.multivariate_normal(mu1, sigma1, int(weight1 * args.points))

    mu2, sigma2, weight2 = [3.0, 1.5], [[0.5, -0.2], [-0.2, 0.5]], 0.15
    data2 = generator.multivariate_normal(mu2, sigma2, int(weight2 * args.points))

    mu3, sigma3, weight3 = [7.0, 0.0], [[3.0, 0.0], [0.0, 1.0]], 0.5
    data3 =  generator.multivariate_normal(mu3, sigma3, int(weight3 * args.points))

    data = np.vstack([data1, data2, data3])
    labels = np.hstack([[0] * data1.shape[0], [1] * data2.shape[0], [2] * data3.shape[0]])

    return data, labels, None

def runExpectationMaximization(generator : np.random.RandomState, data : np.ndarray, labels : np.ndarray, init_means : np.ndarray) -> None:
    """
    Runs the optimisation of the implemented GMM algorithm and a builtin version of the algorithm.
    Then it prints out the optimised parameters which should be almost the same as the oens defined
    in the data generating functions, and displays the results visually. The behaviour of
    the implemented algorithm should be almost the same as the builtin version - small differences
    are expected.
    """
    print("Evaluation of Expectation-Maximisation GMM for {} points from data '{}' with {} components.".format(args.points, args.data, args.n_components))
    startGMM = time.time()
    emgmm = hw1.EMGMM(args.n_components, args.tol, args.max_iter, generator, init_means=init_means)
    emgmm.fit(data)
    predictions = emgmm.predict(data)
    endGMM = time.time()
    print(">>> The algorithm finished in {:>7.4f} seconds with {} iterations.".format(endGMM - startGMM, emgmm.iters_to_convergence))

    print("The computed parameters:")
    with np.printoptions(precision=4, suppress=True):
        print("Weights: {}".format(emgmm.weights))
        print("Means:\n{}".format(emgmm.means))
        print("Covariance matrices:\n{}".format(emgmm.covariances))

    builtin_gmm = sklearn.mixture.GaussianMixture(args.n_components, max_iter=args.max_iter, tol=args.tol, means_init=init_means)
    builtin_gmm.fit(data)
    builtin_predictions = builtin_gmm.predict(data)

    data_x_min, data_x_max, data_y_min, data_y_max = np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])
    show_delta = 4.0

    x, y = np.meshgrid(np.linspace(-5, 15), np.linspace(-10, 10))
    xx = np.vstack([x.ravel(), y.ravel()]).T

    neg_log_proba_emgmm = -emgmm.scoreSamples(xx)
    neg_log_proba_emgmm = np.reshape(neg_log_proba_emgmm, x.shape)
    neg_log_proba_builtin = -builtin_gmm.score_samples(xx)
    neg_log_proba_builtin = np.reshape(neg_log_proba_builtin, x.shape)

    fig, ax = plt.subplots(2, 3, figsize=(13, 6), subplot_kw={'aspect': 'equal'})
    for i, (nl_proba, name, pred) in enumerate(zip([neg_log_proba_emgmm, neg_log_proba_builtin], ["Implemented", "Builtin"], [predictions, builtin_predictions])):
        ax[i, 0].set_title("Original data")
        ax[i, 0].scatter(data[:, 0], data[:, 1], c=labels)
        ax[i, 1].set_title("{} GMM contours".format(name))
        ax[i, 1].scatter(data[:, 0], data[:, 1], color=(0.66, 0.66, 0.66), marker='o', s=0.5)
        ax[i, 1].set_xlim([data_x_min - show_delta, data_x_max + show_delta])
        ax[i, 1].set_ylim([data_y_min - show_delta, data_y_max + show_delta])
        ax[i, 1].contour(x, y, nl_proba, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 1, 10))
        ax[i, 2].set_title("{} GMM predictions".format(name))
        ax[i, 2].scatter(data[:, 0], data[:, 1], c=pred)
        if init_means is not None:
            ax[i, 0].scatter(init_means[:, 0], init_means[:, 1], marker="x", s=200, label="Initial mean")
            ax[i, 0].legend()
            ax[i, 2].scatter(init_means[:, 0], init_means[:, 1], marker="x", s=200, label="Initial mean")
            ax[i, 2].legend()
    fig.tight_layout()
    plt.show()

def main(args : argparse.Namespace) -> None:
    generator = np.random.RandomState(args.seed)

    datagen = {
        "debug" : generateTwoComponents,
        "test" : generateThreeComponents,
    }
    if args.data not in datagen.keys():
        raise ValueError("Requested data generation type is unknown: {}!".format(args.data))
    
    data, labels, init_means = datagen[args.data](generator, args)
    runExpectationMaximization(generator, data, labels, init_means)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
