
from typing import Tuple, Sequence
import numpy as np
import matplotlib.pyplot as plt

def plotDataWithPCs(data : np.ndarray, labels : np.ndarray, mu : Tuple[float, float] = None, pcs : np.ndarray = None) -> None:
    """
    Plots 2D data together with the axes formed by the principal components.
    Data (and mu) has to be in row form, the 'pcs' matrix has to have
    the principal components in columns.

    Arguments:
    - 'data' - Data points given in rows.
    - 'labels' - Numerical labels of the given data points.
    - 'mu' - Mean of the data points.
    - 'pcs' - Principle components given in columns.
    """
    axes = np.asarray([6, 3]) * pcs if pcs is not None else None
    fig, ax = plt.subplots(1, 1, subplot_kw={'aspect': 'equal'})
    ax.set_title("Data coloured according to labels{}".format("with PCs" if pcs is not None else ""))
    ax.scatter(data[:, 0], data[:, 1], c=labels)
    if pcs is not None:
        ax.plot([mu[0], mu[0] + axes[0, 0]], [mu[1], mu[1] + axes[1, 0]], color='blue', linewidth=5)
        ax.plot([mu[0], mu[0] + axes[0, 1]], [mu[1], mu[1] + axes[1, 1]], color='green', linewidth=5)
    fig.tight_layout()
    plt.show()

def plotHistograms(points : np.ndarray, labels : np.ndarray, colors : Sequence[str] = None) -> None:
    """
    Plots the dataset together with histograms showing the distribution of the individual
    classes.
    If there is more than 10 classes, attribute 'colors' has to be specified and it needs
    to define colour for every observation.

    Arguments:
    - 'points' - Data points given in rows.
    - 'labels' - Labels of the given data points.
    - 'colors' - Optional list of string values marking the colours of the classes.
    """
    labels = labels.flatten()
    classes, idx = np.unique(labels, return_index=True)
    classes = np.asarray(classes, np.int32)

    delta = np.mean(np.max(points, axis=0) - np.min(points, axis=0)) / 15
    binCount = (np.max(points, axis=0) - np.min(points, axis=0)) / delta
    binCount = np.asarray((binCount / np.max(binCount)) * 8 + 12, np.int32)
    binValues = [
        [
            np.histogram(points[labels == k, i], binCount[i], range=(np.min(points[:, i]), np.max(points[:, i])))
            for k in classes
        ] for i in range(points.shape[1])
    ]
    
    colourSelection = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'brown', 'coral', 'gold']
    if colors is None:
        colors = np.asarray(colourSelection)[np.asarray(labels, np.int8) - 1]
    else:
        colourSelection = colors[idx]
    
    fig, ax = plt.subplots(points.shape[1], points.shape[1], subplot_kw={'aspect': 'auto'})
    fig.suptitle("Discriminability of principle components")
    for i in range(points.shape[1]):
        for j in range(points.shape[1]):
            if i == j:
                for k in range(len(binValues[i])):
                    ax[i, j].step(binValues[i][k][1][:-1], binValues[i][k][0], where='post', c=colourSelection[classes[k] - 1])
            else:
                ax[i, j].scatter(points[:, j], points[:, i], c=colors)
    fig.tight_layout()
    plt.show()
