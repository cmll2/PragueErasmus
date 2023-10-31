
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import lab04_help

class GaussianNB:

    def __init__(self) -> None:
        """
        Creates a Gaussian naive bayes classifier with similar signature and methods as scikit-learn
        classes. In short, the method 'fit' is used to train data on a training set with its labels
        and method 'predict' returns probabilities of belonging to each class for a given data set.
        Method 'predict' can be called only after a call to 'fit' because training sets up variables
        necessary for class probability computation.
        """
        self._trainingCompleted = False

    def fit(self, data : np.ndarray, labels : np.ndarray) -> None:
        # Store the number of training samples and the number of features for later.
        numFeatures = data.shape[1]
        numSamples = data.shape[0]

        # Preparation for Bayes - count the number of classes.
        self.cnames = np.unique(labels)
        self.numClasses = np.size(self.cnames)

        # ===== Bayes =====

        # Probability of classes.
        self.p_class = np.zeros([self.numClasses])
        # Mean of each feature for the classes.
        self.mean_class = np.zeros([numFeatures, self.numClasses])
        # Standard deviation of each feature for the classes.
        self.std_class = np.zeros([numFeatures, self.numClasses])

        for clIdx in range(np.size(self.cnames)):
            # Select all samples from the current class self.cnames[clIdx].
            Xi = data[labels == self.cnames[clIdx]]
            # TODO: Compute the mean and standard deviation for every feature(pixel).
            self.mean_class[:, clIdx] = None
            self.std_class[:, clIdx] = None
            # TODO: Compute the class probability (Prior).
            self.p_class[clIdx] = None
        
        self._trainingCompleted = True

    def predict(self, data : np.ndarray) -> np.ndarray:
        # Check if 'fit' was called before 'predict'.
        if not self._trainingCompleted:
            raise RuntimeError("BernoulliNB method 'predict' was called before 'fit'. Please ensure that the model finished training before prediction.")

        # The 'probabilities' of belonging to classes for every object.
        class_likelihood = np.zeros([data.shape[0], self.numClasses])

        eps = np.spacing(1)
        for clIdx in range(self.numClasses):
            mu = self.mean_class[:, clIdx]
            std = self.std_class[:, clIdx]
            # Evaluate the pdf of the data.
            # - If std==0 then pdf computation throws a divide by zero exception (look at gaussian pdf formula).
            p_cond = norm.pdf(data, mu, np.where(std > 0, std, eps))
            # We take the logarithm of probabilities - we have to take care of undefined values.
            p_cond[:, std == 0] = 0
            p_cond = np.log(np.where(p_cond > 0, p_cond, 1))
            # TODO: Compute the likelihood (decision function) for every image and class.
            class_likelihood[:, clIdx] = None
        
        return class_likelihood

def main():
    # ===== Naive Bayes based on gaussians =====

    # Load the MNIST data prepared in the first practical, you can use the MnistDataset class,
    # You might have to rename the variables in the following code depending on what you call
    # your data variables.
    # TODO: Ensure that the mnist arrays have the following shape:
    # - train.imgs as       60000x400
    # - train.labels as     1x60000 (1D vector of 60000 elements)
    # - test.imgs as        10000x400
    # - test.labels as      1x10000 (1D vector of 10000 elements)
    train = lab04_help.MnistDataset("mnist_train.npz")
    test = lab04_help.MnistDataset("mnist_test.npz")

    # TODO: Implement the 'fit' method of 'GaussianNB' at the top of the source file.
    # - We train the gaussian bayes classifer by estimating means and standard deviations of each class,
    #   which are then used for prediction.
    bayesClassifier = GaussianNB()
    bayesClassifier.fit(train.imgs, train.labels)

    # TODO: Now, complete the body of the function 'GaussianNB.predict()'.
    train_class_likelihood = bayesClassifier.predict(train.imgs)
    test_class_likelihood = bayesClassifier.predict(test.imgs)

    # Select the most probable classes.
    idx_train = np.argmax(train_class_likelihood, axis=1)
    idx_test = np.argmax(test_class_likelihood, axis=1)

    # TODO: Compute the accuracy of the classification.
    # - How many of the selected class indices match the original labels?
    train_accuracy = None
    test_accuracy = None

    print("Train set accuracy {:.2f}%".format(100 * train_accuracy))
    print("Test set accuracy {:.2f}%".format(100 * test_accuracy))

    # Let's look at images created from the probabilities, they should resemble
    # the digits.
    _, ax = plt.subplots(1, len(bayesClassifier.cnames), figsize=(20, 3), subplot_kw={'aspect': 'equal'})
    side = int(np.round(np.sqrt(train.imgs.shape[1])))
    for clIdx in range(len(bayesClassifier.cnames)):
        ax[clIdx].imshow(np.reshape(np.random.normal(bayesClassifier.mean_class[:, clIdx], bayesClassifier.std_class[:, clIdx]), [side, side]), cmap='Greys_r')
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
