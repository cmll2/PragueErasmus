
import scipy.io
import argparse
import lab05_help

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", default=0.75, type=float, help="Custom classification threshold.")
parser.add_argument("--classifier", default=0, type=int, help="Classifier index.")

def main(args : argparse.Namespace):
    # Example of data with binary classification.
    # Columns 0 - 4 are the output of five different classifiers (probability of belonging to the class 1).
    # Column 5 is the true class.
    roc_data = scipy.io.loadmat("RocInput5.mat")
    roc_data = roc_data["RocInput5"]

    # TODO: Choose the classification threshold 't' from [0, 1] (modify the argument or set 't' to a number).
    t = args.threshold

    # TODO: Choose which classifier to evaluate.
    classif_idx = args.classifier
    confusion_matrix = lab05_help.getConfusionMatrix(roc_data[:, classif_idx], roc_data[:, 5], t)

    # TODO: Inspect the matrix. Are the data balanced?
    # - Print the confusion matrix.
    class_1 = confusion_matrix[0][0] + confusion_matrix[0][1]
    class_2 = confusion_matrix[1][0] + confusion_matrix[1][1]
    balance_factor = class_1 / class_2
    print(balance_factor)

    # TODO: Choose different thresholds and compare confusion matrices.
    # - Compute mutliple confusion matrices using different thresholds and visualise them using 'lab05_help.drawMatrices'.
    # - You can pass a list of matrices to 'lab05_help.drawMatrices'.
    t = [0.2, 0.4, 0.6, 0.75]
    confusion_matrices = []
    for threshold in t:
        confusion_matrices.append(lab05_help.getConfusionMatrix(roc_data[:, classif_idx], roc_data[:, 5], threshold))
    # Draw the confusion matrix.
    lab05_help.drawMatrices(confusion_matrices, log_color=True)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
