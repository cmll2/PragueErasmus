
import numpy as np
import matplotlib.pyplot as plt
import typing
import sklearn.metrics

def getConfusionMatrix(classifer_data : np.ndarray, true_class : np.ndarray, threshold : float) -> np.ndarray:
    """
    Returns a confusion matrix computed from classifier data and true classification according to
    the given threshold. The returned matrix is ordered [positive, negative] assuming that class 1
    marks positive examples and class 0 negative ones.

    Arguments:
    - 'classifierData' - 1D vector of probabilities of belonging to class 1.
    - 'trueClass' - 1D vector of gold labels (true classes of the data).
    - 'threshold' - The threshold for class prediction computation.

    Returns:
    - A square confusion matrix denoting the number of correctly/incorrectly classified samples.
    """
    
    # Compute classification for the selected threshold.
    classes = np.asarray(classifer_data >= threshold, np.int32)

    # Compute the confusion matrix for the chosen classifier.
    # We want to order the items in the matrix as [positive, negative] hence the argument label=[1, 0].
    confusion_matrix = sklearn.metrics.confusion_matrix(true_class, classes, labels=[1, 0])

    return confusion_matrix

def displayMatrix(axes : plt.Axes, matrix : np.ndarray, log_color : bool = False) -> None:
    """
    Displays the confusion matrix in the given pyplot Axes object coloured by matrix element
    values. Optionally, it computes colours according to the logarithm of matrix
    element values.
    Assumes the [positive, negative] ordering in the confusion matrix.

    Arguments:
    - 'axes' - Matplotlib pyplot Axes object where the matrix should be drawn.
    - 'matrix' - Confusion matrix to be drawn into the figure.
    - 'log_color' - Whether the colours shown in the figure should be computed according to the
                   logarithm of the values.
    """
    axes.matshow(np.log(matrix + np.spacing(1)) if log_color else matrix, cmap=plt.cm.Blues, alpha=0.5)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axes.text(x=j, y=i, s=matrix[i, j], va='center', ha='center')
    axes.set_xticks([0, 1])
    axes.set_yticks([0, 1])
    axes.set_xticklabels(["positive", "negative"])
    axes.set_yticklabels(["positive", "negative"], rotation=90)
    axes.set_xlabel("Predicted")
    axes.set_ylabel("True")
    axes.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    axes.xaxis.set_label_position("top")

def drawMatrices(matrices : typing.Union[np.ndarray, typing.Sequence[np.ndarray]], log_color : bool = False) -> None:
    """
    Draws one or more matrices in a single row of matplotlib subplots. The matrix cells can be
    coloured logarithmically to highlight smaller differences.

    Arguments:
    - 'matrices' - One np.ndarray confusion matrix or a list of them intended for visualisation.
    - 'log_color' - Whether to assign colours according to the logarithm of the matrix values.
    """
    is_list = isinstance(matrices, typing.Sequence)
    count = len(matrices) if is_list else 1 
    fig, ax = plt.subplots(1, count, figsize=(count * 4, 4), subplot_kw={'aspect': 'auto'})
    if count > 1:
        for i in range(len(matrices)):
            displayMatrix(ax[i], matrices[i], log_color)
    else:
        displayMatrix(ax, matrices[0] if is_list else matrices, log_color)
    fig.tight_layout()
    plt.show()

def setAxes(axes : plt.Axes, title : str, x_label : str, y_label : str):
    """
    Sets basic properties of pyplot Axes object. The functions sets axes labels, title and
    displays the legend of the figure.

    Arguments:
    - 'axes' - Pyplot Axes object to edit.
    - 'title' - The new title of the Axes object.
    - 'x_label' - The new label of the X axis.
    - 'y_label' - The new label of the Y axis.
    """
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend()

def drawAPPlot(axes : plt.Axes, precisions : np.ndarray, ap_precisions : np.ndarray, recalls : np.ndarray, ap_value : float, ap_name : str):
    """
    Draws an average precision curve together with the precision recall curve in the same
    pyplot Axes object. Name and a floating point value can be specified, which will be written
    in the title of the Axes object.

    Arguments:
    - 'axes' - Pyplot Axes object for drawing.
    - 'precisions' - Precision values for the plot.
    - 'ap_precisions' - Average precision values to draw (approximated/interpolated).
    - 'recalls' - Recall values for the plot.
    - 'ap_value' - Float value, which will be written in the title.
    - 'ap_name' - String name, which will be written in the title together with 'ap_value'.
    """
    axes.step(recalls, ap_precisions, where='post', label="AP")
    axes.plot(recalls, precisions, label="PR")
    axes.set_title("{}: {:.6f}".format(ap_name, ap_value))
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.legend()
