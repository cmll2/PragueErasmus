
import argparse
from typing import Tuple, Sequence, List
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import sklearn.metrics
import lab05_help
import lab05b_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=1337, type=int, help="Seed for the random classifier.")
parser.add_argument("--num_thresholds", default=11, type=int, help="Number of thresholds for linearly-spaced curve plotting.")
parser.add_argument("--ap_clf", default=1, type=int, help="Index of the classifier shown in the 'ap' test.")
# TODO: Use 'test' argument to run the individual tests. First, complete 'pr', 'roc', 'f1' and then 'ap'.
parser.add_argument("--test", default="pr", type=str, help="Investigated curve, one of: 'pr', 'roc', 'f1', 'ap'.")

def computeThresholds(threshold_type : str, args: argparse.Namespace, classifier_data : np.ndarray = None) -> List[np.ndarray]:
    if threshold_type == "linspace":
        # TODO: Compute linearly spaced thresholds for each classifier.
        # - Use 'args.num_thresholds' as the number of linearly-spaced thresholds.
        # - Return a 'list' of 'np.ndarrray' thresholds.
        all_thresholds = []
        for i in range(5):
            thresholds = np.linspace(0, 1, args.num_thresholds)
            all_thresholds.append(thresholds)
        return all_thresholds

    elif threshold_type == "exact":
        # TODO: Compute exact threshold values for each classifier.
        # - Exact thresholds means that you should return all values where the classifier probabilities change in the ascending order.
        # - 'np.unique' is useful in this regard.
        # - Return a 'list' of 'np.ndarray' thresholds.
        all_thresholds = []
        for i in range(5):
            thresholds = np.unique(classifier_data[:, i])
            all_thresholds.append(thresholds)
    else:
        raise ValueError("Unrecognised thresholdType: '{}'".format(threshold_type))

def computeMetrics(thresholds : Sequence[np.ndarray], classifier_data : np.ndarray) -> Tuple[List[np.ndarray], ...]:
    # TODO: Use 'lab05_help.getConfusionMatrix' and 'lab05b_metrics.classifierMetrics' to compute false positive rate, true positive
    # rate, positive predictive value and f1 score for each classifier at the given thresholds.
    # The result should be a List of numpy arrays for each metric, e.g.,
    # - 'fprs' := [randomClassifierFprs, firstClassifierFprs, secondClassifierFprs, thirdClassifierFprs, fourthClassifierFprs, fifthClassifierFprs]
    # - the same for 'tprs', 'ppvs', 'f1s'.
    fprs = None
    tprs = None
    ppvs = None
    f1s = None
    return fprs, tprs, ppvs, f1s

def computeAAP(precisions : np.ndarray, recalls : np.ndarray) -> Tuple[np.ndarray, float]:
    # TODO: Compute approximated average precision.
    # Return new precision values 'ap_precisions' (for graph plotting) and the computed AAP value 'aap'.
    ap_precisions = None
    aap = None
    return ap_precisions, aap

def computeIAP(precisions : np.ndarray, recalls : np.ndarray) -> float:
    # TODO: Compute interpolated average precision.
    # Return new precision values 'ap_precisions' (for graph plotting) and the computed IAP value 'iap'.
    ap_precisions = None
    iap = None
    return ap_precisions, iap

def testAP(precisions_linspace : Sequence[np.ndarray], recalls_linspace : Sequence[np.ndarray], precisions_exact : Sequence[np.ndarray], recalls_exact : Sequence[np.ndarray], clf_idx : int):
    # TODO: Finish functions 'computeAAP' and 'computeIAP'.
    # This test will evaluate AAP and IAP for the classifier at index 'clf_idx' from linearly-spaced and exact thresholds.
    aap_precisions_lin, aap_lin = computeAAP(precisions_linspace[clf_idx], recalls_linspace[clf_idx])
    aap_precisions_ex, aap_ex = computeAAP(precisions_exact[clf_idx], recalls_exact[clf_idx])
    iap_precisions_lin, iap_lin = computeIAP(precisions_linspace[clf_idx], recalls_linspace[clf_idx])
    iap_precisions_ex, iap_ex = computeIAP(precisions_exact[clf_idx], recalls_exact[clf_idx])

    fig, ax = plt.subplots(1, 4, figsize=(16, 4), subplot_kw={'aspect': 'equal'})
    lab05_help.drawAPPlot(ax[0], precisions_linspace[clf_idx], aap_precisions_lin, recalls_linspace[clf_idx], aap_lin, "AAP Linspace")
    lab05_help.drawAPPlot(ax[1], precisions_exact[clf_idx], aap_precisions_ex, recalls_exact[clf_idx], aap_ex, "AAP Exact")
    lab05_help.drawAPPlot(ax[2], precisions_linspace[clf_idx], iap_precisions_lin, recalls_linspace[clf_idx], iap_lin, "IAP Linspace")
    lab05_help.drawAPPlot(ax[3], precisions_exact[clf_idx], iap_precisions_ex, recalls_exact[clf_idx], iap_ex, "IAP Exact")
    fig.tight_layout()
    plt.show()

def testPR(precisions_linspace : Sequence[np.ndarray], recalls_linspace : Sequence[np.ndarray], precisions_exact : Sequence[np.ndarray], recalls_exact : Sequence[np.ndarray], classifier_data : np.ndarray):
    # TODO: Complete this function by filling in plotting code for linearly-spaced, exact and sklearn PR curve.
    # - 'exact' and 'sklearn' curves should be exactly the same.
    # - Why is the curve based on linearly-spaced thresholds different?
    fig, ax = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'aspect': 'equal'})
    names = ["Random", "Clf 1", "Clf 2", "Clf 3", "Clf 4", "Clf 5"] # Use these names as plot legend labels.
    # TODO: Plot (ax[0].plot) linspace PR into ax[0] for all 6 classifiers.
    
    # TODO: Plot (ax[1].plot) exact PR into ax[1] for all 6 classifiers.
    
    # TODO: Plot (ax[2].plot) sklearn PR into ax[2] for all 6 classifiers.
    # - Use 'sklearn.metrics.precision_recall_curve' to compute precisions and recalls in scikit-learn.
    
    lab05_help.setAxes(ax[0], "PR curve (linspace)", "Recall", "Precision")
    lab05_help.setAxes(ax[1], "PR curve (exact)", "Recall", "Precision")
    lab05_help.setAxes(ax[2], "PR curve (sklearn)", "Recall", "Precision")
    fig.tight_layout()
    plt.show()

def testROC(fprs_linspace : Sequence[np.ndarray], tprs_linspace : Sequence[np.ndarray], fprs_exact : Sequence[np.ndarray], tprs_exact : Sequence[np.ndarray], classifier_data : np.ndarray):
    # TODO: Complete this function by filling in plotting code for linearly-spaced, exact and sklearn ROC curve.
    # - 'exact' and 'sklearn' curves should be exactly the same.
    # - Why is the curve based on linearly-spaced thresholds different?
    fig, ax = plt.subplots(1, 3, figsize=(14, 5), subplot_kw={'aspect': 'equal'})
    names = ["Random", "Clf 1", "Clf 2", "Clf 3", "Clf 4", "Clf 5"]
    # TODO: Plot (ax[0].plot) linspace ROC into ax[0] for all 6 classifiers.
    
    # TODO: Plot (ax[1].plot) exact ROC into ax[1] for all 6 classifiers.
    
    # TODO: Plot (ax[2].plot) sklearn ROC into ax[2] for all 6 classifiers.
    # - Use 'sklearn.metrics.roc_curve' to compute fprs and tprs in scikit-learn.
    
    lab05_help.setAxes(ax[0], "ROC curve (linspace)", "FPR", "TPR")
    lab05_help.setAxes(ax[1], "ROC curve (exact)", "FPR", "TPR")
    lab05_help.setAxes(ax[2], "ROC curve (sklearn)", "FPR", "TPR")
    fig.tight_layout()
    plt.show()

def testF1(thresholds_linspace : Sequence[np.ndarray], f1s_linspace : Sequence[np.ndarray], thresholds_exact : Sequence[np.ndarray], f1s_exact : Sequence[np.ndarray]):
    # TODO: Complete this function by filling in plotting code for linearly-spaced and exact F1 curve.
    # - Why do the curves differ?
    # - Which threshold maximises the F1 score?
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), subplot_kw={'aspect': 'equal'})
    names = ["Random", "Clf 1", "Clf 2", "Clf 3", "Clf 4", "Clf 5"]
    # TODO: Plot (ax[0].plot) linspace F1 curve into ax[0] for all 6 classifiers.
    
    # TODO: Plot (ax[1].plot) exact F1 curve into ax[1] for all 6 classifiers.
    
    lab05_help.setAxes(ax[0], "F1 curve (linspace)", "Threshold", "F1")
    lab05_help.setAxes(ax[1], "F1 curve (exact)", "Threshold", "F1")
    fig.tight_layout()
    plt.show()

def main(args : argparse.Namespace):
    # Example data
    # Columns 0 - 4 are the output of five different classifiers (probability of belonging to the class 1).
    # Column 5 is the true class.
    roc_data = scipy.io.loadmat("RocInput5.mat")
    roc_data = roc_data["RocInput5"]

    # Create a random classifier. 'args.seed' makes the results reproducible (if you set it to None then the results will be always different).
    generator = np.random.RandomState(args.seed)
    random_classifier = generator.random([roc_data.shape[0]])
    # Add the random classifier to the other ones so we can work with them in unified manner.
    # - NOTE: Now, column 0 is the random classifer, columns 1 - 5 are other classifiers and column 6 is the true class.
    classifier_data = np.hstack((np.c_[random_classifier], roc_data))

    # Compute linearly-spaced and exact thresholds.
    thresholds_linspace = computeThresholds("linspace", args, classifier_data)
    thresholds_exact = computeThresholds("exact", args, classifier_data)

    # Compute metrics at both threshold sets.
    fprs_lin, tprs_lin, ppvs_lin, f1s_lin = computeMetrics(thresholds_linspace, classifier_data)
    fprs_ex, tprs_ex, ppvs_ex, f1s_ex = computeMetrics(thresholds_exact, classifier_data)

    if args.test == "pr":
        testPR(ppvs_lin, tprs_lin, ppvs_ex, tprs_ex, classifier_data)
    elif args.test == "roc":
        testROC(fprs_lin, tprs_lin, fprs_ex, tprs_ex, classifier_data)
    elif args.test == "f1":
        testF1(thresholds_linspace, f1s_lin, thresholds_exact, f1s_ex)
    elif args.test == "ap":
        testAP(ppvs_lin, tprs_lin, ppvs_ex, tprs_ex, args.ap_clf)
    else:
        raise ValueError("Unrecognised test: '{}'".format(args.test))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
