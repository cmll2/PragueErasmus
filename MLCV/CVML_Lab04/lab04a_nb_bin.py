
import argparse
import numpy as np
import matplotlib.pyplot as plt
import lab04_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--threshold", default=0.4, type=float, help="Threshold for binarisation.")

class BernoulliNB:

    def __init__(self, threshold = None) -> None:
        """
        Creates a bernoulli naive bayes classifier with similar signature and methods as scikit-learn
        classes. In short, the method 'fit' is used to train data on a training set with its labels
        and method 'predict' returns probabilities of belonging to each class for a given data set.
        Method 'predict' can be called only after a call to 'fit' because training sets up variables
        necessary for class probability computation.

        Arguments:
        - 'threshold' - Specifies a threshold used before training and prediction. If not specified
          then data passed to 'fit' and 'predict' have to be binary.
        """
        self.threshold = threshold
        self._trainingCompleted = False

    def fit(self, data : np.ndarray, labels : np.ndarray) -> None:
        # Binarise data if the threshold was selected.
        data = data if self.threshold is None else np.asarray(data > self.threshold, float)

        # Store the number of training samples and the number of features for later.
        numFeatures = data.shape[1]
        numSamples = data.shape[0]

        # Preparation for Bayes - count the number of classes.
        self.cnames = np.unique(labels)
        self.numClasses = np.size(self.cnames)

        # ===== Bayes =====

        # Now, we have black-and-white images of digits and the pixel values are either
        # 0 or 1. Therefore, we are going to examine the probabilities of the individual
        # features (pixels in the image) having the value 1. Then we can compute probabilities
        # for the value 0 simply as the complement to one.

        # Probability that the feature k (the pixel k) equals 1 (is white) in the class c1 (P(x_k=1|omega_c1))
        self.p_1_class = np.zeros([numFeatures, self.numClasses])
        # Probability of class c1 (P(omega_c1)) (Prior)
        self.p_class = np.zeros([self.numClasses])

        for clIdx in range(np.size(self.cnames)):
            # Select all samples from the current class self.cnames[clIdx]
            Xi = data[labels == self.cnames[clIdx], :]
            # TODO: Compute the number of objects from the class where feature k has the value 1.
            # - It should be a vector with one value for each feature.
            N_ki = []
            for k in range (numFeatures):
                N_ki.append(np.sum(Xi[:, k] == 1))
            N_ki = np.asarray(N_ki)
            # TODO: Get the number of objects from the class.
            Ni = Xi.shape[0]
            # TODO: Compute the probability of pixels having the value 1 and use Laplace smoothing with value alpha=1.
            alpha = 1
            self.p_1_class[:, clIdx] = (N_ki+alpha) / (Ni + 2 * alpha)
            # TODO: Compute the probability of the class (Prior).
            self.p_class[clIdx] = Ni / numSamples

        # We need the probability that feature k equals 0 in the class as well.
        self.p_0_class = 1 - self.p_1_class

        # TODO: For numerical reasons we will be working with logarithms.
        # Compute the logarithms of our probability arrays and add 'eps' to those probabilities which were
        # not smoothed, and therefore, might be equal to 0 (which is, technically, undefined for logarithm).
        eps = np.spacing(1)
        # TODO: Compute the natural logarithm of probabilities that feature k equals 1.
        self.log_p_1_class = np.log(self.p_1_class + eps)
        # TODO: Do the same for p_0_class.
        self.log_p_0_class = np.log(self.p_0_class + eps)
        # TODO: Aaaand do the same for the prior probabilities.
        self.log_p_class = np.log(self.p_class + eps)

        self._trainingCompleted = True

    def predict(self, data : np.ndarray) -> np.ndarray:
        # Check if 'fit' was called before 'predict'.
        if not self._trainingCompleted:
            raise RuntimeError("BernoulliNB method 'predict' was called before 'fit'. Please ensure that the model finished training before prediction.")
        # Binarise data if the threshold was selected.
        data = data if self.threshold is None else np.asarray(data > self.threshold, float)

        # The "probabilities" of belonging to classes for every object.
        class_likelihood = np.zeros([data.shape[0], self.numClasses])
        for j in range(data.shape[0]):
            for c in range(self.numClasses):
                # TODO: Compute the likelihood (decision function) for every image and class.
                class_likelihood[j, c] = np.sum(self.log_p_1_class[:, c] * data[j, :] + self.log_p_0_class[:, c] * (1 - data[j, :])) + self.log_p_class[c]
        
        return class_likelihood

def main(args : argparse.Namespace):
    # ===== Naive Bayes for binary features =====

    # Load the MNIST data prepared in the first practical, you can use the MnistDataset class,
    # You might have to rename the variables in the following code depending on what you call
    # your data variables.
    # TODO: Ensure that the mnist arrays have the following shape:
    # - train.imgs as       60000x400
    # - train.labels as     1x60000 (1D vector of 60000 elements)
    # - test.imgs as        10000x400
    # - test.labels as      1x10000 (1D vector of 10000 elements)
    train = lab04_help.MnistDataset("../CVML_Lab01/mnist.npz")
    test = lab04_help.MnistDataset("../CVML_Lab01/t10k.npz")

    #assert train.images.shape == (60000, 400), "Train images have wrong shape."
    assert train.labels.shape == (60000,), "Train labels have wrong shape."
    #assert test.images.shape == (10000, 400), "Test images have wrong shape."
    assert test.labels.shape == (10000,), "Test labels have wrong shape."

    # Let us binarise the images (each pixel will be either 0 or 1)
    # - Make sure your images were normalised (pixel values in range from 0 to 1).
    #   Otherwise, change the threshold to a fraction of the pixel value interval.
    # TODO: Use 'args.threshold' to binarise image data from both training and testing sets.
    # - Bernoulli classifier wotks only with binary data.
    # - Complete the 'BernoulliNB.fit()' method.
    binTrainImgs = np.asarray(train.images > args.threshold, float)
    binTestImgs = np.asarray(test.images > args.threshold, float)
    bayesClassifier = BernoulliNB()
    bayesClassifier.fit(binTrainImgs, train.labels)

    # Let's evaluate
    # TODO: Now, complete the body of the function 'BernoulliNB.predict()'
    train_class_likelihood = bayesClassifier.predict(binTrainImgs)
    test_class_likelihood = bayesClassifier.predict(binTestImgs)

    # The right class is probably the most probable one.
    idx_train = np.argmax(train_class_likelihood, axis=1)
    idx_test = np.argmax(test_class_likelihood, axis=1)

    # TODO: Compute the accuracy of the classification.
    # - how many of the selected class indices match the original labels?
    train_accuracy = np.sum(idx_train == train.labels) / train.labels.shape[0]
    test_accuracy = np.sum(idx_test == test.labels) / test.labels.shape[0]

    # TODO: Try to change the binarisation threshold.
    # - Does the number of correctly classified images change?

    print("Train set accuracy: {:.2f}%".format(100 * train_accuracy))
    print("Test set accuracy: {:.2f}%".format(100 * test_accuracy))

    # Let's look at images created from the probabilities, they should resemble the digits.
    fig, ax = plt.subplots(1, len(bayesClassifier.cnames), figsize=(20, 3), subplot_kw={'aspect': 'equal'})
    side = int(np.round(np.sqrt(train.images.shape[1])))
    for clIdx in range(len(bayesClassifier.cnames)):
        ax[clIdx].imshow(np.reshape(bayesClassifier.p_1_class[:, bayesClassifier.cnames[clIdx]], [side, side]), cmap='gray')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
