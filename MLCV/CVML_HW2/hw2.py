
import argparse
from typing import Dict
import numpy as np
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.model_selection
from scipy.stats import norm, t, chi2

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
parser.add_argument("--example", default=42, type=int, help="Example argument for your custom tests.")

class Results:

    ERR_NB = "err_nb"
    ERR_KNN = "err_knn"
    DIFF_ERR = "diff_err"
    CRIT_VALUE = "crit_value"
    HALF_INTERVAL = "half_interval"
    REJECT_H0 = "reject_h0"
    SE = "se"
    DIFF_MEAN = "diff_mean"
    M = "m"
    DENOMINATOR = "denominator"

class ErrorTest:
    """Base class used to remove repetition of storing arguments."""

    def __init__(self, confidence_alpha : float, knn_k : int, knn_metric : str, knn_metric_params : dict) -> None:
        self.confidence_alpha = confidence_alpha
        self.knn_k = knn_k
        self.knn_metric = knn_metric
        self.knn_metric_params = knn_metric_params

class IndependentTest(ErrorTest):
    """Class implementing the basic independent difference in error hypothesis test."""

    def __init__(self, confidence_alpha : float, knn_k : int, knn_metric : str, knn_metric_params : dict) -> None:
        super().__init__(confidence_alpha, knn_k, knn_metric, knn_metric_params)

    def compute(self, train_set : Dict[str, np.ndarray], test_set : Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        This function evaluates the actual independent hypothesis test for error difference between two classifiers.

        Arguments:
        - 'train_set' - Dictionary with data intended for training of the classifiers. Values with keys 'data_nb' and 'labels_nb' are
                        intended for naive bayes, whereas 'data_knn' and 'labels_knn' are for K nearest neighbours.
        - 'test_set' - Dictionary with data intended for testing of the classifiers. Values with keys 'data_nb' and 'labels_nb' are
                       intended for naive bayes, whereas 'data_knn' and 'labels_knn' are for K nearest neighbours.

        Returns:
        - Dictionary of variables computed through the hypothesis test with the following values:
          - 'ERR_NB' - Classification error of the Naive Bayes classifier.
          - 'ERR_KNN' - Classification error of the K nearest neighbours classifier.
          - 'DIFF_ERR' - The difference of errors between NB and KNN.
          - 'CRIT_VALUE' - Critical value of the normal distribution for 'self.confidence_alpha'.
          - 'HALF_INTERVAL' - Half of the true error confidence interval (the +- part of the formulae)
          - 'REJECT_H0' - (bool) Whether we can reject the hypothesis H0 that the classifiers have the same error rate.
        """
        # TODO: Implement the computation of the independent hypothesis test (presentation MLCV_5.pdf, slides 25, 26).
        # - Train 'sklearn.naive_bayes.GaussianNB' and 'sklearn.neighbors.KNeighborsClassifier' using data given in 'train_set'.
        #   - Our zero hypothesis H0 is that these classifiers are equally good, i.e. the difference in their error is 0.
        # - The error of the classifiers should be computed from their performance on the 'test_set'.
        # - Use 'norm.ppf' to get the critical value of the normal distribution.
        # - Return a dictionary of the values specified above. The expected return structure is written for you.
        nb = sklearn.naive_bayes.GaussianNB()
        knn = sklearn.neighbors.KNeighborsClassifier(self.knn_k, metric=self.knn_metric, metric_params=self.knn_metric_params)
        
        nb.fit(train_set['data_nb'], train_set['labels_nb'])
        knn.fit(train_set['data_knn'], train_set['labels_knn'])

        err_nb = 1 - nb.score(test_set['data_nb'], test_set['labels_nb'])
        err_knn = 1 - knn.score(test_set['data_knn'], test_set['labels_knn'])

        diff_err = err_nb - err_knn

        crit_value = norm.ppf(1 - self.confidence_alpha / 2)

        std_error_diff = np.sqrt(err_nb * (1 - err_nb) / test_set['data_nb'].shape[0] + err_knn * (1 - err_knn) / test_set['data_knn'].shape[0])

        half_interval = crit_value * std_error_diff

        reject_h0 = abs(diff_err) > half_interval


        return {
            Results.ERR_NB : err_nb,
            Results.ERR_KNN : err_knn,
            Results.DIFF_ERR : diff_err,
            Results.CRIT_VALUE : crit_value,
            Results.HALF_INTERVAL : half_interval,
            Results.REJECT_H0 : reject_h0,
        }

class PairedTTest(ErrorTest):
    """Class implementing the paired t-test for error difference between classifiers."""

    def __init__(self, paired_k_splits : int, confidence_alpha : float, knn_k : int, knn_metric : str, knn_metric_params : dict) -> None:
        super().__init__(confidence_alpha, knn_k, knn_metric, knn_metric_params)
        self.paired_k_splits = paired_k_splits

    def compute(self, train_set : Dict[str, np.ndarray], test_set : Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        This function evaluates the actual paired t-test for error difference between two classifiers.

        Arguments:
        - 'train_set' - Dictionary with data intended for training of the classifiers. Values with keys 'data_nb' and 'labels_nb' are
                        intended for naive bayes, whereas 'data_knn' and 'labels_knn' are for K nearest neighbours.
        - 'test_set' - Dictionary with data intended for testing of the classifiers: 'data', 'labels'.

        Returns:
        - Dictionary of variables computed through the hypothesis test with the following values:
          - 'SE' - Variance of the error differences (Marked as SE in the formulae).
          - 'CRIT_VALUE' - Critical value of the student distribution for 'self.confidence_alpha'.
          - 'DIFF_MEAN' - The mean difference of errors between NB and KNN.
          - 'HALF_INTERVAL' - Half of the true error confidence interval (the +- part of the formulae)
          - 'REJECT_H0' - (bool) Whether we can reject the hypothesis H0 that the classifiers have the same error rate.
        """
        # TODO: Implement the computation of the paired t-test (presentation MLCV_5.pdf, slides 27, 28).
        # - Train 'sklearn.naive_bayes.GaussianNB' and 'sklearn.neighbors.KNeighborsClassifier' using data given in 'train_set'.
        #   - Our zero hypothesis H0 is that these classifiers are equally good, i.e. the difference in their error is 0.
        # - The error of the classifiers should be computed from their performance on the 'test_set'.
        # - Use 't.ppf' to get the critical value of the student's t-distribution.
        # - Return a dictionary of the values specified above. The expected return structure is written for you.
        #
        # - Equally split the test data for testing by 'np.split' into 'self.paired_k_splits' groups.
        nb = sklearn.naive_bayes.GaussianNB()
        knn = sklearn.neighbors.KNeighborsClassifier(self.knn_k, metric=self.knn_metric, metric_params=self.knn_metric_params)
        
        nb.fit(train_set['data_nb'], train_set['labels_nb'])
        knn.fit(train_set['data_knn'], train_set['labels_knn'])

        test_data = test_set['data']
        test_labels = test_set['labels']
        test_data_splits = np.split(test_data, self.paired_k_splits)
        test_labels_splits = np.split(test_labels, self.paired_k_splits)

        errors_nb = []
        errors_knn = []

        for i in range(self.paired_k_splits):
            error_nb = 1 - nb.score(test_data_splits[i], test_labels_splits[i])
            errors_nb.append(error_nb)

            error_knn = 1 - knn.score(test_data_splits[i], test_labels_splits[i])
            errors_knn.append(error_knn)

        differences = np.array(errors_nb) - np.array(errors_knn)

        diff_mean = np.mean(differences)
        se = np.std(differences, ddof=1) / np.sqrt(len(differences))

        crit_value = t.ppf(1 - (self.confidence_alpha / 2), len(differences) - 1)

        half_interval = crit_value * se

        reject_h0 = abs(diff_mean) > half_interval

        return {
            Results.SE : se,
            Results.CRIT_VALUE : crit_value,
            Results.DIFF_MEAN : diff_mean,
            Results.HALF_INTERVAL : half_interval,
            Results.REJECT_H0 : reject_h0,
        }

class CorrectedResampledTTest(ErrorTest):
    """Class implementing the corrected resampled t-test for error difference between classifiers."""

    def __init__(self, corrected_kfold_splits : int, confidence_alpha : float, knn_k : int, knn_metric : str, knn_metric_params : dict) -> None:
        super().__init__(confidence_alpha, knn_k, knn_metric, knn_metric_params)
        self.corrected_kfold_splits = corrected_kfold_splits

    def compute(self, data : np.ndarray, labels : np.ndarray) -> Dict[str, float]:
        """
        This function evaluates the actual corrected resampled t-test for error difference between two classifiers.

        Arguments:
        - 'data' - Data for training and testing through the cross-validation mechanism.
        - 'labels' - Labels for training and testing through the cross-validation mechanism.

        Returns:
        - Dictionary of variables computed through the hypothesis test with the following values:
          - 'SE' - Variance of the error differences (Marked as SE in the formulae).
          - 'CRIT_VALUE' - Critical value of the student's distribution for 'self.confidence_alpha'.
          - 'DIFF_MEAN' - The mean difference of errors between NB and KNN.
          - 'HALF_INTERVAL' - Half of the true error confidence interval (the +- part of the formulae)
          - 'REJECT_H0' - (bool) Whether we can reject the hypothesis H0 that the classifiers have the same error rate.
        """
        # TODO: Implement the computation of the corrected resampled t-test (presentation MLCV_5.pdf, slide 29).
        # - Train and test 'sklearn.naive_bayes.GaussianNB' and 'sklearn.neighbors.KNeighborsClassifier' using data given
        #   in the arguments. The test requires a cross-validation-like error computation as described on the slide and in the lecture.
        #   - Define the classifiers in the cross validation scheme as shown in the code and train them with appropriate data.
        #   - Our zero hypothesis H0 is that these classifiers are equally good, i.e. the difference in their error is 0.
        # - Use 't.ppf' to get the critical value of the student's t-distribution.
        # - Return a dictionary of the values specified above. The expected return structure is written for you.
        #
        # - Use 'sklearn.model_selection.KFold' to split the data into K sets. The K is given by 'self.corrected_kfold_splits'.
        #   - Then, enumeration of 'kfold.split(data)' to get the indices of the individual kfold splits.
        
        differences = []

        kfold = sklearn.model_selection.KFold(n_splits=self.corrected_kfold_splits)

        for train_index, test_index in kfold.split(data):

            train_data, test_data = data[train_index], data[test_index]
            train_labels, test_labels = labels[train_index], labels[test_index]

            nb = sklearn.naive_bayes.GaussianNB()
            knn = sklearn.neighbors.KNeighborsClassifier(self.knn_k, metric=self.knn_metric, metric_params=self.knn_metric_params)

            nb.fit(train_data, train_labels)
            knn.fit(train_data, train_labels)

            error_nb = 1 - nb.score(test_data, test_labels)
            error_knn = 1 - knn.score(test_data, test_labels)

            differences.append(error_nb - error_knn)

        diff_mean = np.mean(differences)
        se = np.sqrt((1/self.corrected_kfold_splits + len(test_data)/len(train_data)) * np.sum((diff_mean - differences)**2)/(self.corrected_kfold_splits-1))

        crit_value = t.ppf(1 - (self.confidence_alpha / 2), len(differences) - 1)

        half_interval = crit_value * se

        reject_h0 = abs(diff_mean) > half_interval

        return {
            Results.SE : se,
            Results.CRIT_VALUE : crit_value,
            Results.DIFF_MEAN : diff_mean,
            Results.HALF_INTERVAL : half_interval,
            Results.REJECT_H0 : reject_h0,
        }

class McNemarTest(ErrorTest):
    """Class implementing the McNemar test for error difference between classifiers."""

    def __init__(self, confidence_alpha : float, knn_k : int, knn_metric : str, knn_metric_params : dict) -> None:
        super().__init__(confidence_alpha, knn_k, knn_metric, knn_metric_params)

    def compute(self, train_set : Dict[str, np.ndarray], test_set : Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        This function evaluates the actual McNemar test for error difference between two classifiers.

        Arguments:
        - 'train_set' - Dictionary with data intended for training of the classifiers. Values with keys 'data_nb' and 'labels_nb' are
                        intended for naive bayes, whereas 'data_knn' and 'labels_knn' are for K nearest neighbours.
        - 'test_set' - Dictionary with data intended for testing of the classifiers: 'data', 'labels'.

        Returns:
        - Dictionary of variables computed through the hypothesis test with the following values:
          - 'M' - The M value computed from the contingency matrix.
          - 'CRIT_VALUE' - Critical value of the CHI^2 distribution for 'self.confidence_alpha'.
          - 'DENOMINATOR' - The denominator of the formula for 'M'.
          - 'REJECT_H0' - (bool) Whether we can reject the hypothesis H0 that the classifiers have the same error rate.
        """
        # TODO: Implement the computation of the paired t-test (presentation MLCV_5.pdf, slides 30, 31).
        # - Train 'sklearn.naive_bayes.GaussianNB' and 'sklearn.neighbors.KNeighborsClassifier' using data given in 'train_set'.
        #   - Our zero hypothesis H0 is that these classifiers are equally good, i.e. the difference in their error is 0.
        # - The error of the classifiers should be computed from their performance on the 'test_set'.
        # - Use 'chi2.ppf' to get the critical value of the Chi squared distribution.
        # - Return a dictionary of the values specified above. The expected return structure is written for you.
        nb = sklearn.naive_bayes.GaussianNB()
        knn = sklearn.neighbors.KNeighborsClassifier(self.knn_k, metric=self.knn_metric, metric_params=self.knn_metric_params)
        
        nb.fit(train_set['data_nb'], train_set['labels_nb'])
        knn.fit(train_set['data_knn'], train_set['labels_knn'])

        pred_nb = nb.predict(test_set['data'])
        pred_knn = knn.predict(test_set['data'])

        # n00 = np.sum((pred_nb == test_set['labels']) & (pred_knn == test_set['labels']))
        # n11 = np.sum((pred_nb != test_set['labels']) & (pred_knn != test_set['labels']))
        n01 = np.sum((pred_nb == test_set['labels']) & (pred_knn != test_set['labels']))
        n10 = np.sum((pred_nb != test_set['labels']) & (pred_knn == test_set['labels']))

        m = (np.abs(n01 - n10) - 1)**2 / (n01 + n10)

        denominator = n01 + n10

        crit_value = 3.84146

        reject_h0 = m > crit_value

        return {
            Results.M : m,
            Results.CRIT_VALUE : crit_value,
            Results.DENOMINATOR : denominator,
            Results.REJECT_H0 : reject_h0,
        }

def main(args : argparse.Namespace) -> None:
    # NOTE: Your solution will be evaluated exclusively through the 'evaluator.py' script, however,
    # if it is easier for you to run experiments within this script then you can edit this function
    # to do so.
    raise NotImplementedError()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
