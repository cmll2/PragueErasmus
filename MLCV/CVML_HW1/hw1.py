
import argparse
from typing import Union, Tuple
import numpy as np
import scipy.stats

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--example", default=42, type=int, help="Example argument for your custom tests.")

class EMGMM:
    """
    Custom implmentation of gaussian mixture model using the expectation-maximisation algorithm.
    """

    def __init__(self, n_components : int = 1, tol : float = 0.001, max_iter : int = 100, random_state : Union[int, np.random.RandomState] = None, init_means : np.ndarray = None) -> None:
        """
        Initialises the expectation-maximisation algorithm.

        Arguments:
        - 'n_components' - Number of gaussian components expected in the data.
        - 'tol' - The tolerance used as a stopping convergence criterion.
        - 'max_iter' - Number of maximum allowed iterations in case of slow convergence.
        - 'n_init' - Number of fitting retries.
        - 'random_state' - Random number generator that has to be used in all random computations for reproducibility.
        """
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.generator = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(random_state)
        self.fitted = False
        self.init_means = init_means

    def _computeProba(self, means : np.ndarray, covariances : np.ndarray, data : np.ndarray) -> np.ndarray:
        """
        Computes the probabilities of data points for each gaussian mixture component given by the
        multivariate normal distributions - the resulting values are the values of the pdfs at the data points.

        Arguments:
        - 'means' - Means of the gaussian components in rows.
        - 'covariances' - Covariance matrices of the gaussian components.
        - 'data' - The data points where we need to compute the pdfs.

        Returns:
        - Array of shape 'data.shape[0]'x'self.n_components' with pdf values of the gaussian components at the data points.
        """
        # TODO: Compute the probabilities (pdf values of the multivariate gaussian distribution) for all data points
        # and gaussian components. This is the 'phi' on slide 42 of the lecture 3 slides.
        # - To compute pdf of multivariate normal distribution for given data points, use function 'scipy.stats.multivariate_normal'
        #   and use its parameter 'allow_singular=True' to avoid random corner cases.
        # - Return numpy array.
        probs = np.zeros((data.shape[0], self.n_components))
        for i in range(self.n_components):
            covariances[i] += np.eye(covariances[i].shape[0]) * 1e-5
            probs[:, i] = scipy.stats.multivariate_normal.pdf(data, means[i], covariances[i], allow_singular=True)
        return probs


    def _expectation(self, weights : np.ndarray, means : np.ndarray, covariances : np.ndarray, data : np.ndarray) -> np.ndarray:
        """
        Computes the expectation step of the EM algorithm resulting in membership values (gamma).

        Arguments:
        - 'weights' - Vector of the weights of the gaussian components.
        - 'means' - Means of the gaussian components in rows.
        - 'covariances' - Covariance matrices of the gaussian components.
        - 'data' - The data points for which we need to compute membership.

        Returns:
        - Array of shape 'data.shape[0]'x'self.n_components' with membership values for every data point and gaussian component.
        """
        # TODO: Compute the membership of all data points for all gaussian components according to the formula on
        # slide 42 of the lecture 3 slides.
        # - Use 'self._computeProba' to compute the probabilities 'phi'.
        # - Return numpy array.
        probs = self._computeProba(means, covariances, data)
        membership = np.zeros((data.shape[0], self.n_components))
        for i in range(self.n_components):
            membership[:, i] = weights[i] * probs[:, i]
        membership /= np.sum(membership, axis=1, keepdims=True)
        return membership

    def _maximization(self, membership : np.ndarray, data : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the maximisation step of the EM algorithm resulting in weights, means and covariance matrices
        maximising the probability of gaussian parameters under the given membership.

        Arguments:
        - 'membership' - Membership (gamma) of all data points for all gaussian components.
        - 'data' - The data points where we compute GMM.

        Returns:
        - The new weights of the model according to the membership.
        - The new means of the model according to the membership.
        - The new covariance matrices of the model according to the membership.
        """
        # TODO: Compute new weights, means and covariance matrices of all gaussian components according to the
        # given data and membership of the current iteration given as an argument. The formulas are given
        # on slide 44 of the lecture 3 slides.
        # - 'membership' is represented by 'gamma' in the formulas and N is the number of all data points.
        # - We are computing covariance matrices and not just variance, i.e. the squared part of the 'sigma' formula
        #   is a matrix multiplication (remember our PCA implementation).
        #   - Also, the membership multiplication 'gamma' should happen only once, it should not be squared.
        # - Return numpy arrays.
        N = data.shape[0]
        weights = np.sum(membership, axis=0) / N
        means = np.zeros((self.n_components, data.shape[1]))
        covariances = np.zeros((self.n_components, data.shape[1], data.shape[1]))
        for i in range(self.n_components):
            means[i] = np.sum(membership[:, i][:, np.newaxis] * data, axis=0) / np.sum(membership[:, i])
            diff = data - means[i]
            covariances[i] = np.dot((membership[:, i][:, np.newaxis] * diff).T, diff) / np.sum(membership[:, i])
        return weights, means, covariances

    def fit(self, data : np.ndarray, labels : np.ndarray = None) -> None:
        """
        Runs the expectation-maximisation algorithm.

        Arguments:
        - 'data' - Data for the optimisation.
        - 'labels' - Class labels of the data points (unused).
        """
        # A hint about the expected shape of the computed parameters.
        self.membership = np.zeros((data.shape[0], self.n_components))
        self.weights = np.zeros(self.n_components)
        self.means = np.zeros((self.n_components, data.shape[1]))
        self.covariances = np.zeros((self.n_components, data.shape[1], data.shape[1]))
        self.iters_to_convergence = self.max_iter

        iter_membership, iter_weights, iter_means, iter_covariances = self._initParams(data)
        iter_convergence = self.max_iter
        for i in range(self.max_iter):
            new_iter_membership = self._expectation(iter_weights, iter_means, iter_covariances, data)
            iter_weights, iter_means, iter_covariances = self._maximization(new_iter_membership, data)
            diff = np.mean(np.abs(new_iter_membership - iter_membership))
            iter_membership = new_iter_membership
            if diff < self.tol:
                iter_convergence = i + 1
                break

        self.membership = iter_membership
        self.weights = iter_weights
        self.means = iter_means
        self.covariances = iter_covariances
        self.iters_to_convergence = iter_convergence

        self.fitted = True

    def predict(self, data : np.ndarray) -> np.ndarray:
        """
        Predicts the class of the given data points. Note that the model splits the data into logical
        groups but it doesn't say which group is which.

        Arguments:
        - 'data' - Data for prediction.

        Returns:
        - Labels of the data points according to the model - the values are different from the labels in the 'fit' method.
        """
        self._checkFitted()
        membership = self._expectation(self.weights, self.means, self.covariances, data)
        labels = np.argmax(membership, axis=1)
        return labels
    
    def scoreSamples(self, data : np.ndarray) -> np.ndarray:
        """
        Computes the log-probabilities of data points for the most likely gaussian component.

        Arguments:
        - 'data' - Data points for log-probability computation.

        Returns:
        - Log probabilities of the data points for the most likely component.
        """
        self._checkFitted()
        probs = self._computeProba(self.means, self.covariances, data)
        log_proba = np.log(np.max(probs, axis=1) + np.spacing(1))
        return log_proba
    
    def _checkFitted(self) -> None:
        """
        Raises an error if the method 'fit' hasn't been called so far.
        """
        if not self.fitted:
            raise RuntimeError("The fitting method 'EMGMM.fit()' has to be called before 'EMGMM.predict()'!")
    
    def _initParams(self, data : np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes initial parameters for the expectation-maximisation algorithm.

        Arguments:
        - 'data' - Data used for optimisation.

        Returns:
        - Uniform membership values.
        - Uniform weights.
        - Random means if 'self.init_means' is None, otherwise it uses 'self.init_means'.
        - Identity covariance matrices.
        """
        data_min, data_max = np.min(data, axis=0), np.max(data, axis=0)
        membership = np.ones((data.shape[0], self.n_components)) / self.n_components
        weights = np.ones(self.n_components) / self.n_components
        means = self.generator.random((self.n_components, data.shape[1])) * (data_max - data_min) + data_min if self.init_means is None else self.init_means
        covariances = np.stack([np.identity(data.shape[1]) for _ in range(self.n_components)])
        return membership, weights, means, covariances

def main(args : argparse.Namespace) -> None:
    # NOTE: Your solution will be evaluated exclusively through the 'evaluator'.py script, however,
    # if it is easier for you to run experiments within this script then you can edit this function
    # to run your tests.
    raise NotImplementedError()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
