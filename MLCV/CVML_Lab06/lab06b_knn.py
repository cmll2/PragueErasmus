
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sklearn.neighbors
import scipy.stats
import lab06_help
import lab06a_crossval

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="example", type=str, help="Executed task, one of: 'example', 'classify', 'visualize'")
parser.add_argument("--point_count", default=20, type=int, help="Number of points in each class.")
parser.add_argument("--x_scatter", default=0.1, type=float, help="X scatter of the generated data.")
parser.add_argument("--seed", default=42, type=int, help="Seed for RNG in this assignment.")
parser.add_argument("--example_k", default=9, type=int, help="K value used in the example.")
parser.add_argument("--example_point_x", default=1.7, type=float, help="X coordinate of the example point.")
parser.add_argument("--example_point_y", default=1.7, type=float, help="Y coordinate of the example point.")
parser.add_argument("--grid_size", default=100, type=int, help="Grid size for classification area visualisation.")
parser.add_argument("--clf_knn_k", default=11, type=int, help="Number of neighbors for the test set classification.")

def exampleKNN(args : argparse.Namespace, generator : np.random.RandomState, data : np.ndarray, labels : np.ndarray):
    # Create KNN models with different hyper-parameters.
    knn_eu = sklearn.neighbors.KNeighborsClassifier(args.example_k, metric="minkowski", p=2)
    knn_mn5 = sklearn.neighbors.KNeighborsClassifier(args.example_k, metric="minkowski", p=5)
    knn_mhl = sklearn.neighbors.KNeighborsClassifier(args.example_k, metric="mahalanobis", metric_params={'V' : [[0.1, 0], [0, 2]]})
    knn_cheb = sklearn.neighbors.KNeighborsClassifier(args.example_k, metric="chebyshev")
    
    # Fit the classifiers with our data.
    knn_eu.fit(data, labels)
    knn_mn5.fit(data, labels)
    knn_mhl.fit(data, labels)
    knn_cheb.fit(data, labels)

    # Create an unknown object.
    example_data = np.reshape([args.example_point_x, args.example_point_y], [1, 2])

    # Find the closest neighbours of the unknown object.
    eu_dist, eu_idx = knn_eu.kneighbors(example_data)
    mn5_dist, mn5_idx = knn_mn5.kneighbors(example_data)
    mhl_dist, mhl_idx = knn_mhl.kneighbors(example_data)
    cheb_dist, cheb_idx = knn_cheb.kneighbors(example_data)

    # TODO: Classify 'example_data':
    # - Find the most frequent element (mode) in the corresponding 'classes' values (for each distance metric).
    # - You can use 'scipy.stats.mode' which returns tuple with the most common value and its count.

    # Draw the generated data.
    fig, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'auto'})
    ax.set_title("KNN with different distance metrics")
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    ax.scatter(example_data[0, 0], example_data[0, 1], s=60, color='red', marker='x', linewidths=3)
    ax.scatter(data[eu_idx, 0], data[eu_idx, 1], edgecolors='g', marker='s', facecolors='none', s=300, linewidths=0.5, label="Euclidian")
    ax.scatter(data[mn5_idx, 0], data[mn5_idx, 1], edgecolors='g', marker='o', facecolors='none', s=300, linewidths=0.5, label="Minkowski p=5")
    ax.scatter(data[cheb_idx, 0], data[cheb_idx, 1], edgecolors='g', marker='*', facecolors='none', s=300, linewidths=0.5, label="Chebyshev")
    ax.scatter(data[mhl_idx, 0], data[mhl_idx, 1], edgecolors='g', marker='^', facecolors='none', s=300, linewidths=0.5, label="Mahalanobis")
    ax.legend()
    fig.tight_layout()
    plt.show()

def classifyKNN(args : argparse.Namespace, generator : np.random.RandomState, data : np.ndarray, labels : np.ndarray):
    # Generate 30 points from the population with 'lab06_help.generateData' - 10 for each class.
    x_data_test, y_data_test, labels_test = lab06_help.generateData(generator, 10, args.x_scatter)
    data_test = np.vstack([x_data_test, y_data_test]).T

    # TODO: Find 11-NN using euclidean and mahalanobis distance.
    # - Classify the data manually as in 'exampleKNN' (find the most common label for each data point).
    # - Compute misclassification rate using the function from 'lab06a_crossval'.
    knn_eu = sklearn.neighbors.KNeighborsClassifier(args.clf_knn_k, metric="minkowski", p=2)
    knn_mhl = sklearn.neighbors.KNeighborsClassifier(args.clf_knn_k, metric="mahalanobis", metric_params={'V' : [[0.1, 0], [0, 2]]})

    pred_eu_manual = None
    pred_mhl_manual = None

    print("Euclidean manual misclassification ratio: {:.4f}".format(lab06a_crossval.errorRate(pred_eu_manual, labels_test)))
    print("Mahalanobis manual misclassification ratio: {:.4f}".format(lab06a_crossval.errorRate(pred_mhl_manual, labels_test)))

    # TODO: Do the same using the method 'sklearn.neighbors.KNeighborsClassifier.predict'.
    # - You can use models created for the manual classification above.
    # - Predict for both euclidean and mahalanobis distance.
    # - Compute misclassification rate using the function from 'lab06a_crossval' (the results from manual classification should be the same).
    pred_eu_sklearn = None
    pred_mhl_sklearn = None

    print("Euclidean sklearn misclassification ratio: {:.4f}".format(lab06a_crossval.errorRate(pred_eu_sklearn, labels_test)))
    print("Mahalanobis sklearn misclassification ratio: {:.4f}".format(lab06a_crossval.errorRate(pred_mhl_sklearn, labels_test)))

def visualizeKNN(args : argparse.Namespace, generator : np.random.RandomState, data : np.ndarray, labels : np.ndarray):
    # Visualise decision boundaries.
    # - Classify all points on a grid.
    xx, yy = np.meshgrid(np.linspace(data[:, 0].min(), data[:, 0].max(), args.grid_size), np.linspace(data[:, 1].min(), data[:, 1].max(), args.grid_size))
    classifiable_grid = np.vstack([xx.ravel(), yy.ravel()]).T

    # NOTE: Example of grid classification using 3-NN with euclidean distance.
    knnEu = sklearn.neighbors.KNeighborsClassifier(3, metric="minkowski", p=2)
    knnEu.fit(data, labels)
    z = knnEu.predict(classifiable_grid)
    z = np.reshape(z, xx.shape)

    # TODO: Modify the following code such that it draws boundaries for K = 1, 3, 5, 7, 9, 11
    # and all distance metrics (euclidean, minkowski p=5, mahalanobis, chebyshev) in a single figure.
    # - Note that if you change '1, 1' in 'plt.subplots' then 'ax' will be a matrix with the
    #   corresponding shape.
    # - It is necessary to create and fit classifier for each combination of classifier/K and predict labels of 'classifiable_grid'
    fig, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'auto'})
    ax.pcolormesh(xx, yy, z, alpha=0.25, shading='auto')
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    fig.tight_layout()
    plt.show()

def main(args : argparse.Namespace):
    # Create a generator for reproducible RNG.
    generator = np.random.RandomState(args.seed)
    # Generate 2D points.
    xData, yData, labels = lab06_help.generateData(generator, args.point_count, args.x_scatter)
    data = np.vstack([xData, yData]).T

    tasks = {
        "example" : exampleKNN,
        "classify" : classifyKNN,
        "visualize" : visualizeKNN
    }
    if args.task not in tasks.keys():
        raise ValueError("Unrecognised task: '{}'".format(args.task))
    tasks[args.task](args, generator, data, labels)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
