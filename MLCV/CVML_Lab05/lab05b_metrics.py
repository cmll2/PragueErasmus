
import argparse
import typing
import numpy as np
import scipy.io
import lab05_help

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", default=0.75, type=float, help="Custom classification threshold.")

def classifierMetrics(confusion_matrix : np.ndarray) -> typing.Tuple[float, float, float, float, float]:
    # TODO: Compute classifier metrics listed below.
    # - ACC - accuracy / percentage of correctly classified examples
    # - ERR - error rate
    # - TPR - sensitivity / recall / true positive rate
    # - FPR - false positive rate
    # - PPV - precision / positive predicted value
    acc = (confusion_matrix[0][0] + confusion_matrix[1][1]) / (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])
    err = 1 - acc
    tpr = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
    fpr = confusion_matrix[1][0] / (confusion_matrix[1][0] + confusion_matrix[1][1])
    ppv = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
    
    return acc, err, tpr, fpr, ppv

def main(args : argparse.Namespace):
    # Example of data with binary classification.
    # Columns 0 - 4 are the output of five different classifiers (probability of belonging to the class 1).
    # Column 5 is the true class.
    roc_data = scipy.io.loadmat("RocInput5.mat")
    roc_data = roc_data["RocInput5"]
    
    # TODO: Implement the function 'classifierMetrics' so that it returns the correct values.
    # TODO: Compare the metric values for all 5 classifiers for the same threshold.
    # - use 'lab05help.getConfusionMatrix' to compute the confusion matrix
    t = args.threshold
    for i in range(5):
        cm = lab05_help.getConfusionMatrix(roc_data[:, i], roc_data[:, 5], t)
        acc, err, tpr, fpr, ppv = classifierMetrics(cm)
        print("Classifier {}: ACC: {:.5f}, ERR: {:.5f}, TPR: {:.5f}, FPR: {:.5f}, PPV: {:.5f}".format(i, acc, err, tpr, fpr, ppv))

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
