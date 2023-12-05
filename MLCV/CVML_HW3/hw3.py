
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence
import lzma
import pickle
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

class DataStorage:

    FEATURES_KEY = "segmentation_features"
    LABELS_KEY = "segmentation_labels_num"
    FEATURE_NAMES_KEY = "feature_names"

    def __init__(self, features : np.ndarray, labels : np.ndarray, feature_names : np.ndarray = None, label_names : Sequence[str] = None) -> None:
        """
        Initialises the instance with data arrays.

        Arguments:
        - 'features' - Data feature values.
        - 'labels' - Data labels.
        - 'feature_names' - The names of the features.
        - 'label_names' - The names of the class labels.
        """
        self.features = features
        self.data_count : int = self.features.shape[0]
        self.feature_count : int = self.features.shape[1]
        self.labels = labels
        self.feature_names = feature_names
        self.label_names = label_names

    def getSubset(self, indices : np.ndarray) -> DataStorage:
        """
        Creates a new instance of this class with a subset of the data at the given indices.

        Arguments:
        - 'indices' - Indices to the subset of the data.

        Returns:
        - New instance of the class with data selected according to 'indices'.
        """
        return DataStorage(self.features[indices], self.labels[indices], self.feature_names, self.label_names)

    @classmethod
    def fromFile(cls, data_path : Path) -> DataStorage:
        """
        Loads the data from the given file.

        Arguments:
        - 'data_path' - Path to the file with the segmentation data.

        Returns:
        - An instance of this class with the data from the given file.
        """
        # Load the matlab file.
        data = scipy.io.loadmat(data_path)
        # Extract the feature values.
        features : np.ndarray = data[DataStorage.FEATURES_KEY]
        # Numerical labels in 1D array starting at index 0. 
        labels : np.ndarray = (data[DataStorage.LABELS_KEY] - 1).ravel()
        # Extract the feature names - this is a nested array.
        feature_names : np.ndarray = data[DataStorage.FEATURE_NAMES_KEY]       
        # Names of the classes from 0 to 6.
        label_names : List[str] = ["BRICKFACE", "CEMENT", "FOLIAGE", "GRASS", "PATH", "SKY", "WINDOW"]
        return cls(features, labels, feature_names, label_names)

class DataAnalysis:

    def __init__(self, train_data_path : Path, split_for_testing : bool = False, rng_seed = None) -> None:
        """
        Initialises the implementation class with trainign data and arguments which can help evaluate
        the implementation effectively.

        Arguments:
        - 'train_data_path' - Path to the training data file.
        - 'split_for_testing' - Whether the training data should be split for evaluation testing.
        - 'rng_seed' - Seed for a random number generator.
        """
        self._train_data : DataStorage = DataStorage.fromFile(train_data_path)
        self._split_for_testing = split_for_testing
        self._rng = np.random.RandomState(rng_seed)
        # TODO: If 'self._split_for_testing' is True then you can split 'self._train_data' into a training
        # and testing set using 'self._train_data.getSubset(indices)'. This can help you try the evaluation
        # function on "unseen" data to verify that you didn't make a mistake. Otherwise, the analysis of data
        # and the training of the best models should be done with all training data.
        # - Store the custom test set in 'self._test_data' so that the 'evaluation' function works as it is written.

    def featureAnalysis(self):
        """Implementation of the feature analysis task."""
        # TODO: You should implement your feature analysis testing code here.
        # - Do not forget to apply your final feature selection/transformation on the data before both training
        #   and testing. This method is only for finding an appropriate feature selection/transformation.
        raise NotImplementedError()

    def clusterAnalysis(self):
        """Implementation of the cluster analysis task."""
        # TODO: You should implement your cluster analysis code here.
        raise NotImplementedError()

    def trainSearch(self):
        """Implementation of the search for the best model parameters."""
        # TODO: You should implement your parameter search of the selected models. The results should be
        # the best parameters for your three selected models.
        # - 'trainSearch' and 'trainBest' do not have an expected structure. You can keep 'trainBest' empty
        #   if you do ALL training in this method (especially if you save the best models here).
        raise NotImplementedError()

    def trainBest(self):
        """Training of the selected models with the best parameters found in the search method."""
        # TODO: You should implement training of your selected models with the best parameters discovered
        # in the parameter search in this method.
        # - This method can be used to skip the lengthy parameter search when testing the evaluation method.
        # - You can save your models with the following example code to skip training entirely when you debug
        #   the evaluation task.
        self._my_model_name = "my_model"
        self._my_model = None
        with lzma.open(self._my_model_name, "wb") as model_file:
            pickle.dump(self._my_model, model_file)
        
        raise NotImplementedError()

    def evaluate(self, test_data_path : Path = None):
        """
        Implementation of the evaluation of best models on a testing dataset.

        Arguments:
        - 'test_data_path' - Path to the test data file or None if it unavailable.
        """
        # TODO: Implement evaluation of your models on test data stored in 'self._test_data'.
        # - You can load models saved in the training methods with the provided sample code.
        if test_data_path is not None:
            self._test_data : DataStorage = DataStorage.fromFile(test_data_path)

        with lzma.open(self._my_model_name, "rb") as model_file:
            self._my_model = pickle.load(model_file)

        raise NotImplementedError()
