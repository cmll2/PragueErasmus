#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import msvcrt
parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate a binary classification data with labels {-1, 1}.
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0,
        n_clusters_per_class=1, flip_y=0, class_sep=2, random_state=args.seed)
    target = 2 * target - 1 
    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    data = np.concatenate((data, np.ones((data.shape[0], 1))), axis=1)

    # Generate initial perceptron weights.
    weights = np.zeros(data.shape[1])

    done = False
    iteration = 0

    plt.ion()
    fig, ax = plt.subplots(figsize = (7,8))
    plt.scatter(data[target == 1][:, 0], data[target == 1][:, 1], marker='o', label='Class 1', color='b')
    plt.scatter(data[target == -1][:, 0], data[target == -1][:, 1], marker='x', label='Class -1', color='r')
    plt.legend(loc='upper left')
    plt.title('Perceptron, iteration {}'.format(iteration))
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    x = np.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 100)
    if weights[1] != 0:
        y = -(weights[0] * x + weights[2]) / weights[1]
        line, = ax.plot(x, y, 'k-')
    else:
        line, = ax.plot(x, np.zeros(x.shape), 'k-')
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(10)
    msvcrt.getch()


    while not done:
        permutation = generator.permutation(data.shape[0])

        # TODO: Implement the perceptron algorithm, notably one iteration
        # over the training data in the order of `permutation`. During the
        # training data iteration, perform the required updates to the `weights`
        # for incorrectly classified examples. If all training instances are
        # correctly classified, set `done=True`, otherwise set `done=False`.

        for i in range(data.shape[0]):
            predict = np.dot(data[permutation[i]], weights)
            if predict * target[permutation[i]] <= 0:
                weights += target[permutation[i]] * data[permutation[i]]
                done = False
                break

        else:
            done = True
        iteration += 1
        plt.title('Perceptron, iteration {}'.format(iteration))
        y = -(weights[0] * x + weights[2]) / weights[1]
        line.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.4)

        if args.plot and not done:
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*3))
                plt.subplot(3, 3, 1 + len(plt.gcf().get_axes()))
            plt.scatter(data[:, 0], data[:, 1], c=target)
            xs = np.linspace(*plt.gca().get_xbound() + (50,))
            ys = np.linspace(*plt.gca().get_ybound() + (50,))
            plt.contour(xs, ys, [[[x, y, 1] @ weights for x in xs] for y in ys], levels=[0])
            plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")
        
    msvcrt.getch()

    return weights


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))
