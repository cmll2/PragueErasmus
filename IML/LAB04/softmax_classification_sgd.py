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
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)
    softmax = lambda x : np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=1, keepdims=True)
    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        #
        # Note that you need to be careful when computing softmax because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # $softmax(z) = softmax(z + any_constant)$ and compute $softmax(z) = softmax(z - maximum_of_z)$.
        # That way we only exponentiate non-positive values, and overflow does not occur.
        max = np.max(train_data @ weights, axis=1, keepdims=True)
        for i in range(0,train_data.shape[0],args.batch_size):
            batch = train_data[permutation[i:i+args.batch_size]]
            batch_target = train_target[permutation[i:i+args.batch_size]]
            batch_softmax = softmax(batch @ weights)
            grad = batch.T @ (batch_softmax - np.eye(args.classes)[batch_target]) / batch.shape[0]
            weights -= args.learning_rate * grad


        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        train_accuracy = np.mean(np.argmax(train_data @ weights, axis=1) == train_target)
        test_accuracy = np.mean(np.argmax(test_data @ weights, axis=1) == test_target)
        train_logits, test_logits = train_data @ weights, test_data @ weights
        max_train_logits, max_test_logits = np.max(train_logits, axis=1, keepdims=True), np.max(test_logits, axis=1, keepdims=True)
        train_log_sum_exp, test_log_sum_exp = np.log(np.sum(np.exp(train_logits - max_train_logits), axis=1, keepdims=True)) + max_train_logits, np.log(np.sum(np.exp(test_logits - max_test_logits), axis=1, keepdims=True)) + max_test_logits
        cross_entropy_train_loss, cross_entropy_test_loss = np.mean(train_log_sum_exp - train_logits[np.arange(train_logits.shape[0]), train_target.reshape(-1,)]), np.mean(test_log_sum_exp - test_logits[np.arange(test_logits.shape[0]), test_target.reshape(-1,)])
        train_loss, test_loss = cross_entropy_train_loss, cross_entropy_test_loss
        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
