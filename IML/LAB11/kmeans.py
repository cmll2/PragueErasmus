#!/usr/bin/env python3
import argparse

import numpy as np

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", choices=["random", "kmeans++"], help="Initialization")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        plt.gcf().get_axes() or plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial dataset.
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.

    centers = np.zeros([args.clusters, data.shape[1]])

    if args.init == "random":
        centers = data[generator.choice(len(data), size=args.clusters, replace=False)]
    elif args.init == "kmeans++":
        centers[0] = data[generator.randint(len(data))] 
        for i in range(1, args.clusters):
            distances = np.zeros(len(data))  
            for j, point in enumerate(data):
                closest_center = np.argmin([np.linalg.norm(point - center) for center in centers[:i]])
                distances[j] = np.linalg.norm(point - centers[closest_center]) ** 2
            prob_dist = distances / np.sum(distances)
            next_centroid_index = generator.choice(np.arange(len(data)), p=prob_dist)
            centers[i] = data[next_centroid_index]

    else:
        raise ValueError("Unknown init method '{}'".format(args.init))

    if args.plot:
        plot(args, 0, data, centers, clusters=None)

    # Run `args.iterations` of the K-Means algorithm.
    for iteration in range(args.iterations):
        # TODO: Perform a single iteration of the K-Means algorithm, storing
        # zero-based cluster assignment to `clusters`.

        clusters = np.zeros(len(data))
        for i in range(len(data)):
            distances = [np.linalg.norm(data[i] - centers[j]) for j in range(args.clusters)]
            clusters[i] = np.argmin(distances)

        #update centers
        for i in range(args.clusters):
            centers[i] = np.mean(data[clusters == i], axis=0)

        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    clusters = main(args)
    print("Cluster assignments:", clusters, sep="\n")
