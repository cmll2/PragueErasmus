import argparse
import numpy as np
from scipy.spatial.distance import cdist
import lab02_help

parser = argparse.ArgumentParser()
# You may change the values of the arguments here (default) or in the commandline.
# This exercise has 3 tasks:
# - 'measure' - Implement and test 4 fitness measures from the lecture.
# - 'one-step' - Use the implemented fitness measures to compute one-step forward feature selection.
# - 'sequential' - Use the implemented fitness measures to compute sequential forward feature selection.
parser.add_argument("--task", default="measure", type=str, help="Performed task: 'measure', 'one-step', 'sequential'.")
parser.add_argument("--data", default="hriby2.txt", type=str, help="Path to the file with data.")

def taskMeasure(labels : np.ndarray, features : np.ndarray) -> None:
    # Create a list with indices of features you want to use.
    selection = [1, 4, 5, 6]
    print("Selected feature indices: {}".format(selection))

    # X_tilde - set of data for these features.
    X = features[:, selection]

    # TODO: ===== Task 1 ('measure') =====
    # Modify functions computing different fitness measure within this script such that
    # they return the correct value.
    # You can start with one or two functions, continue with the other tasks and then
    # return to the other functions.
    J = fitnessCons(labels, X)
    print("Consistency fitness:         {:.4f}".format(J))

    J = fitnessCbfs(labels, X)
    print("Correlation based fitness:   {:.4f}".format(J))

    J = fitnessIcd(labels, X)
    print("Interclass distance fitness: {:.4f}".format(J))

    J = fitnessMi(labels, X)
    print("Mutual information fitness:  {:.4f}".format(J))

def best_features_sequential(features, labels, fitness_function, k):
    selected_features = []
    num_features = features.shape[1]
    
    for _ in range(k):
        best_feature_idx = None
        best_fitness = -np.inf  # Initialize with negative infinity for maximization.
        for feature_idx in range(num_features):
            if feature_idx in selected_features:
                continue  # Skip features already selected.           
            # Create a subset of features including the current feature.
            selected_indices = selected_features + [feature_idx]
            selected_features_subset = features[:, selected_indices]           
            # Evaluate the fitness function for the selected feature set.
            fitness = fitness_function(labels, selected_features_subset)          
            # Check if this feature set provides better fitness.
            if fitness > best_fitness:
                best_fitness = fitness
                best_feature_idx = feature_idx
        # Add the best feature to the selected features.
        selected_features.append(best_feature_idx)
    return selected_features

def taskSequential(labels : np.ndarray, features : np.ndarray) -> None:
    # TODO: ===== Task 2 ('one-step') =====
    # Implement the One-step forward selection.
    #
    # Evaluate individual features, select top 3.
    # - 'np.argsort' can be used to find indices of a sorted array (to find the indices of features with the highest scores).
    # - 'np.sort' can be used to just sort an array of values, but 'np.argsort' is generally more useful.
    # Compare the sets of features chosen with different fitness functions.
    K = 3
    selected_features_cons = best_features_sequential(features, labels, fitnessCons, K)
    print("Consistency fitness:         {}".format(selected_features_cons))
    selected_features_cbfs = best_features_sequential(features, labels, fitnessCbfs, K)
    print("Correlation based fitness:   {}".format(selected_features_cbfs))
    selected_features_icd = best_features_sequential(features, labels, fitnessIcd, K)
    print("Interclass distance fitness: {}".format(selected_features_icd))
    selected_features_mi = best_features_sequential(features, labels, fitnessMi, K)
    print("Mutual information fitness:  {}".format(selected_features_mi))
    return None

def best_features_one_step(features, labels, fitness_function, k):
    selected_features = []
    features_scores = []
    num_features = features.shape[1]
    
    for feature_idx in range(num_features):
        X = features[:, feature_idx]
        fitness = fitness_function(labels, X)
        features_scores.append(fitness)
    minus_features = -np.asarray(features_scores)
    #sort features by fitness and return best K indexes
    return np.argsort(minus_features)[:k]

def taskOneStep(labels: np.ndarray, features : np.ndarray) -> None:
    # TODO: ===== Task 3 ('sequential') =====
    # Implement the Sequential forward selection.
    #
    # Evaluate individual features, select top 3.
    # - 'np.argmax' can be used to find the feature with the highest score.
    # - 'np.nanargmax' is 'np.argmax', which ignores NaN (Not a Number) values.
    # Compare the sets of features chosen with different fitness functions.
    K = 3
    selected_features_cons = best_features_one_step(features, labels, fitnessCons, K)
    print("Consistency fitness:         {}".format(selected_features_cons))
    selected_features_cbfs = best_features_one_step(features, labels, fitnessCbfs, K)
    print("Correlation based fitness:   {}".format(selected_features_cbfs))
    selected_features_icd = best_features_one_step(features, labels, fitnessIcd, K)
    print("Interclass distance fitness: {}".format(selected_features_icd))
    selected_features_mi = best_features_one_step(features, labels, fitnessMi, K)
    print("Mutual information fitness:  {}".format(selected_features_mi))
    return None


def fitnessCons(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Consistency measure
    
    Write the body of this function. It should calculate the fitness (consistency)
    of a subset of features.

    Arguments:
    - 'labels' - Vector of true classes.
    - 'X' - Observations of a subset of features.

    Returns:
    - Fitness of the provided feature set.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])

    # Find unique rows
    # - 'C' - unique rows
    # - 'ix' - index to the first occurence in X for each unique row
    # - 'ic' - index to C for each row in X
    # - 'np.max(ic) = len(ix) - 1' ... indexing starts at 0
    # - 'C = X[ix, :]', 'X = C[ic, :]'
    # - 'M' - number of occurences for each unique row
    C, ix, ic, M = np.unique(X, axis=0, return_index=True, return_inverse=True, return_counts=True)
    
    # TODO: For each unique row (a loop through ix) find the number of inconsistent classifications.
    IC = 0   
    for i in range(len(ix)):
        unique_row = C[i, :]
        row_indices = np.where(np.all(X == unique_row, axis=1))[0]
        row_classes = labels[row_indices]
        most_common_class = np.argmax(np.bincount(row_classes))
        IC += np.sum(row_classes != most_common_class)

    # TODO: Return the final value J according to the formula from the lecture.
    return 1 - (IC / len(labels))

def fitnessCbfs(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Correlation-based Feature selector

    Write the body of this function. It should calculate the fitness
    (Correlation-based Feature Selector) of a subset of features.
    
    Arguments:
    - 'labels' - Vector of true classes.
    - 'X' - Observations of a subset of features.

    Returns:
    - Fitness of the provided feature set.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Get the correlation coefficients.
    R = lab02_help.corrcoef(labels, X)
    # TODO: Compute the number of columns (features) in X.
    K = np.shape(X)[1]
    
    # The upper triangular matrix of ones.
    indT = np.tri(K + 1, k=-1, dtype=bool).T
    
    # NOTE: Matlab uses a lower triangular matrix because it reads values column by column
    # whereas numpy does it row by row. The matrix R is symmetric.
    coefs=R[indT]
    rcf = np.mean(coefs[0 : K])

    # TODO: Return the final value J according to the formula from the lecture.
    # - Think about how the formula changes when 'K == 1'.
    if K > 1:
        rff = np.mean(coefs[K :])
        return (K*rcf)/np.sqrt(K+K*(K-1)*rff)
    else:
        return rcf
    
def fitnessIcd(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Interclass distance

    Write the body of this function. It should calculate the fitness
    (Interclass distance) of a subset of features.

    Arguments:
    - 'labels' - Vector of true classes.
    - 'X' - Observations of a subset of features.

    Returns:
    - Fitness of the feature set as the mean class distance.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    X = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Objects belonging to class 1.
    X_1 = X[labels == 1, :]
    # Objects belonging to class 0.
    X_0 = X[labels == 0, :]
    # Matrix of pair-wise distances of points in X_1 and X_0.
    d_X = cdist(X_1, X_0, "euclidean")
    D_X = np.mean(d_X)

    prob= [len(X_1) / len(X), len(X_0) / len(X)]
    # TODO: Return the value of interclass distance according to the formula from the lecture.
    # - Compute the class probabilities as the fractions of objects with the particular classes.
    return prob[0] * prob[1] * D_X

def fitnessMi(labels : np.ndarray, X : np.ndarray) -> float:
    """
    Mutual information

    Write the body of this function. It should calculate the fitness
    (Mutual information) of a subset of features.

    Arguments:
    - 'labels' - Vector of the true class.
    - 'X' - Observations of a subset of features.

    Returns:
    - Mutual information of the feature subset and the classification.
    """
    # Make sure that the data array is a matrix(2D) and not a vector(1D).
    # - Renamed 'X' to 'Y' to match the formula from the slides.
    Y = X if len(X.shape) > 1 else np.reshape(X, [-1, 1])
    # Follow the formula I(Y;X) = H(Y) - H(Y|X)
    # The following line computes the entropy of the whole set (Y in our formula).
    # - NOTE: 'X' in code is not the same as 'X' in the formula.
    entr = getEntropy(Y)

    # TODO: Compute the remaining terms of the formula.
    # - Class probabilities are the fractions of objects with the particular classes.
    prob = []
    for i in range(labels.shape[0]):
        prob.append(np.sum(labels == i) / labels.shape[0])
    # TODO: Return the value of mutual information according to the formula from the lecture 'I(Y;X) = H(Y) - H(Y|X)'.
    conditional_entropy = 0
    for i in range(len(prob)):
        class_indices = np.where(labels == i)[0]
        class_entropy = getEntropy(Y[class_indices])
        conditional_entropy += prob[i] * class_entropy
    return entr - conditional_entropy

def getEntropy(X : np.ndarray) -> float:
    """
    Computes the entropy of a single data set.
    """
    # The following code computes the entropy of a single variable.
    # Find unique objects of the set.
    C, ix, ic, occur = np.unique(X, axis=0, return_index=True, return_inverse=True, return_counts=True)
    N = X.shape[0]
    # Compute the probability.
    prob = occur / N
    # Compute the entropy.
    entr = -np.sum(prob * np.log2(prob))
    return entr

def main(args : argparse.Namespace) -> None:
    # Load mushroom classification data.
    data = lab02_help.parseTextFile(args.data)

    # The first column in data matrix is the class.
    # The remaining columns 2:23 are features.
    # Each row is one observation.

    # Split data into two variables: 'labels' (class) and 'features' (features).
    labels = data[:, 0]
    features = data[:, 1:]

    tasks = {
        "measure" : taskMeasure,
        "one-step" : taskOneStep,
        "sequential" : taskSequential
    }
    if args.task not in tasks:
        raise ValueError("Task '{}' is not recognised!".format(args.task))
    tasks[args.task](labels, features)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
