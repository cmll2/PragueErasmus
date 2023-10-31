#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[tuple[np.ndarray, ...], list[float]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]

    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.
        #
        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where $ReLU(x) = max(x, 0)$, and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as `ReLU(inputs @ weights[0] + biases[0])`.
        # The value of the output layer is computed as `softmax(hidden_layer @ weights[1] + biases[1])`.
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        hidden_layer = np.maximum(inputs @ weights[0] + biases[0], 0)
        max = np.max(hidden_layer @ weights[1] + biases[1], axis=1, keepdims=True)
        output_layer = np.exp(hidden_layer @ weights[1] + biases[1] - max) / np.sum(np.exp(hidden_layer @ weights[1] + biases[1] - max), axis=1, keepdims=True)
        return hidden_layer, output_layer

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # The gradient used in SGD has now four parts, gradient of `weights[0]` and `weights[1]`
        # and gradient of `biases[0]` and `biases[1]`.
        #
        # You can either compute the gradient directly from the neural network formula,
        # i.e., as a gradient of $-log P(target | data)$, or you can compute
        # it step by step using the chain rule of derivatives, in the following order:
        # - compute the derivative of the loss with respect to *inputs* of the
        #   softmax on the last layer,
        # - compute the derivative with respect to `weights[1]` and `biases[1]`,
        # - compute the derivative with respect to the hidden layer output,
        # - compute the derivative with respect to the hidden layer input,
        # - compute the derivative with respect to `weights[0]` and `biases[0]`.

        for i in range(0, train_data.shape[0], args.batch_size):
            batch = train_data[permutation[i:i + args.batch_size]]
            batch_target = train_target[permutation[i:i + args.batch_size]]
            hidden_layer, output_layer = forward(batch)
            grad = [np.zeros_like(w) for w in weights]
            grad[1] = hidden_layer.T @ (output_layer - np.eye(args.classes)[batch_target]) / batch.shape[0]
            grad[0] = batch.T @ ((output_layer - np.eye(args.classes)[batch_target]) @ weights[1].T * (hidden_layer > 0)) / batch.shape[0]
            biases[1] -= args.learning_rate * np.mean(output_layer - np.eye(args.classes)[batch_target], axis=0)
            biases[0] -= args.learning_rate * np.mean(((output_layer - np.eye(args.classes)[batch_target]) @ weights[1].T * (hidden_layer > 0)), axis=0)
            weights = [w - args.learning_rate * g for w, g in zip(weights, grad)]

        # TODO: After the SGD epoch, measure the accuracy for both the
        # train test and the test set.
        train_accuracy, test_accuracy = np.mean(np.argmax(forward(train_data)[1], axis=1) == train_target), np.mean(np.argmax(forward(test_data)[1], axis=1) == test_target)

        print("After epoch {}: train acc {:.1f}%, test acc {:.1f}%".format(
            epoch + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases), [100 * train_accuracy, 100 * test_accuracy]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters, metrics = main(args)
    print("Learned parameters:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:12]] + ["..."]) for ws in parameters), sep="\n")
