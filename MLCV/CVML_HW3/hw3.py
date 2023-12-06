
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
        if self._split_for_testing:
            indices = self._rng.permutation(self._train_data.data_count)
            split = int(self._train_data.data_count * 0.8)
            self._test_data = self._train_data.getSubset(indices[split:])
            self._train_data = self._train_data.getSubset(indices[:split])

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import PolynomialFeatures
        scaler = MinMaxScaler()
        poly = PolynomialFeatures(2)
        self._pipeline = Pipeline([('scaler', scaler), ('poly', poly)])
        self._pipeline.fit(self._train_data.features)
        self._train_data.features = self._pipeline.transform(self._train_data.features)
        if self._split_for_testing:
            self._test_data.features = self._pipeline.transform(self._test_data.features)

    def featureAnalysis(self):
        """Implementation of the feature analysis task."""
        # TODO: You should implement your feature analysis testing code here.
        # - Do not forget to apply your final feature selection/transformation on the data before both training
        #   and testing. This method is only for finding an appropriate feature selection/transformation.
        correlation_matrix = np.corrcoef(self._train_data.features, rowvar=False)
        # Set a threshold for selecting features based on correlation
        threshold = 0.5 
        selected_features = []
        for i in range(correlation_matrix.shape[0]):
            for j in range(i+1, correlation_matrix.shape[1]):
                if abs(correlation_matrix[i, j]) > threshold:
                    selected_features.append(i)
                    break
        print("Selected features based on correlation with the target:")
        print(selected_features)
        # Update training and testing data with selected features
        self._train_data.features = self._train_data.features[:, selected_features]
        self._test_data.features = self._test_data.features[:, selected_features]
        self._best_features = selected_features 

    def clusterAnalysis(self):
        """Implementation of the cluster analysis task."""
        # TODO: You should implement your cluster analysis code here.
        from sklearn.cluster import KMeans
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(KMeans(n_init=10), {'n_clusters': np.arange(2, 10)}, cv=5)
        grid_search.fit(self._train_data.features, self._train_data.labels)
        kmeans = grid_search.best_estimator_
        print("Best number of clusters: {}".format(kmeans.n_clusters))
        kmeans.fit(self._train_data.features)
        self._cluster_centers = kmeans.cluster_centers_
        self._cluster_labels = kmeans.labels_
        self._cluster_counts = np.bincount(self._cluster_labels)
    
    def trainSearch(self):
        """Implementation of the search for the best model parameters."""
        # TODO: You should implement your parameter search of the selected models. The results should be
        # the best parameters for your three selected models.
        # - 'trainSearch' and 'trainBest' do not have an expected structure. You can keep 'trainBest' empty
        #   if you do ALL training in this method (especially if you save the best models here).
        from sklearn.model_selection import GridSearchCV
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression

        # Random Forest
        print("Searching for best parameters for random forest...")
        grid_search = GridSearchCV(RandomForestClassifier(), {'n_estimators': np.arange(10, 110, 10)}, cv=5)
        grid_search.fit(self._train_data.features, self._train_data.labels)
        self._rfc = grid_search.best_estimator_
        print("Best number of trees for random forest : {}".format(self._rfc.n_estimators))

        # SVM
        print("Searching for best parameters for SVM...")
        grid_search = GridSearchCV(SVC(), {'C': np.logspace(-3, 3, 7)}, cv=5)
        grid_search.fit(self._train_data.features, self._train_data.labels)
        self._svc = grid_search.best_estimator_
        print("Best C for SVM : {}".format(self._svc.C))

        # Logistic Regression
        print("Searching for best parameters for Logistic Regression...")
        grid_search = GridSearchCV(LogisticRegression(), {'C': np.logspace(-3, 3, 7)}, cv=5)
        grid_search.fit(self._train_data.features, self._train_data.labels)
        self._lr = grid_search.best_estimator_
        print("Best C for Logistic Regression : {}".format(self._lr.C))

        #fit the best models
        print("Fitting best Random Forest Classifier...")
        self._rfc.fit(self._train_data.features, self._train_data.labels)
        print("Fitting best SVM...")
        self._svc.fit(self._train_data.features, self._train_data.labels)
        print("Fitting best Logistic Regression...")
        self._lr.fit(self._train_data.features, self._train_data.labels)

        #save the best models
        print("Saving best models...")
        with lzma.open("models/rfc.model", "wb") as model_file:
            pickle.dump(self._rfc, model_file)

        with lzma.open("models/svc.model", "wb") as model_file:
            pickle.dump(self._svc, model_file)

        with lzma.open("models/lr.model", "wb") as model_file:
            pickle.dump(self._lr, model_file)

    def trainBest(self):
        """Training of the selected models with the best parameters found in the search method."""
        # TODO: You should implement training of your selected models with the best parameters discovered
        # in the parameter search in this method.
        # - This method can be used to skip the lengthy parameter search when testing the evaluation method.
        # - You can save your models with the following example code to skip training entirely when you debug
        #   the evaluation task.

        # self._my_model_name = "my_model"
        # self._my_model = None
        # with lzma.open(self._my_model_name, "wb") as model_file:
        #     pickle.dump(self._my_model, model_file)
        
        # raise NotImplementedError()
        return

    def evaluate(self, test_data_path : Path = None):
        """
        Implementation of the evaluation of best models on a testing dataset.

        Arguments:
        - 'test_data_path' - Path to the test data file or None if it unavailable.
        """
        # TODO: Implement evaluation of your models on test data stored in 'self._test_data'.
        # - You can load models saved in the training methods with the provided sample code.

        print("Loading best models...")
        with lzma.open("models/rfc.model", "rb") as model_file:
            self._rfc = pickle.load(model_file)

        with lzma.open("models/svc.model", "rb") as model_file:
            self._svc = pickle.load(model_file)

        with lzma.open("models/lr.model", "rb") as model_file:
            self._lr = pickle.load(model_file)
        if test_data_path is not None:
            self._test_data : DataStorage = DataStorage.fromFile(test_data_path)
            self._test_data.features = self._test_data.features[:, self._best_features]
            #evaluate on the custom test set
            print("Evaluating for random forest...")
            print("Accuracy: {}".format(self._rfc.score(self._test_data.features, self._test_data.labels)))
            print("Evaluating for SVM...")
            print("Accuracy: {}".format(self._svc.score(self._test_data.features, self._test_data.labels)))
            print("Evaluating for Logistic Regression...")
            print("Accuracy: {}".format(self._lr.score(self._test_data.features, self._test_data.labels)))

        elif self._split_for_testing:
            #evaluate on the custom test set
            print("Evaluating for random forest...")
            print("Accuracy: {}".format(self._rfc.score(self._test_data.features, self._test_data.labels)))
            print("Evaluating for SVM...")
            print("Accuracy: {}".format(self._svc.score(self._test_data.features, self._test_data.labels)))
            print("Evaluating for Logistic Regression...")
            print("Accuracy: {}".format(self._lr.score(self._test_data.features, self._test_data.labels)))
        else:
            #evaluate on the training set
            print("Evaluating for random forest...")
            print("Accuracy: {}".format(self._rfc.score(self._train_data.features, self._train_data.labels)))
            print("Evaluating for SVM...")
            print("Accuracy: {}".format(self._svc.score(self._train_data.features, self._train_data.labels)))
            print("Evaluating for Logistic Regression...")
            print("Accuracy: {}".format(self._lr.score(self._train_data.features, self._train_data.labels)))

        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_score
        from sklearn.metrics import recall_score
        
        rf_confusion_matrix = confusion_matrix(self._train_data.labels, self._rfc.predict(self._train_data.features))
        rf_precision = precision_score(self._train_data.labels, self._rfc.predict(self._train_data.features), average='macro')
        rf_recall = recall_score(self._train_data.labels, self._rfc.predict(self._train_data.features), average='macro')
        
        svm_confusion_matrix = confusion_matrix(self._train_data.labels, self._svc.predict(self._train_data.features))
        svm_precision = precision_score(self._train_data.labels, self._svc.predict(self._train_data.features), average='macro')
        svm_recall = recall_score(self._train_data.labels, self._svc.predict(self._train_data.features), average='macro')
        lr_confusion_matrix = confusion_matrix(self._train_data.labels, self._lr.predict(self._train_data.features))
        lr_precision = precision_score(self._train_data.labels, self._lr.predict(self._train_data.features), average='macro')
        lr_recall = recall_score(self._train_data.labels, self._lr.predict(self._train_data.features), average='macro')
        import os
        current_dir = os.getcwd()
        figures_dir = os.path.join(current_dir, 'figures')
        print("Saving figures to {}".format(figures_dir))
        #plot the results into a precision-recall space
        plt.figure()
        plt.plot(rf_precision, rf_recall, 'ro', label='Random Forest')
        plt.plot(svm_precision, svm_recall, 'bo', label='SVM')
        plt.plot(lr_precision, lr_recall, 'go', label='Logistic Regression')
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.legend()
        plt.savefig(figures_dir+'precision_recall.png')
        plt.close()
        #plot the confusion matrices
        plt.figure()
        plt.imshow(rf_confusion_matrix)
        plt.title('Random Forest Confusion Matrix')
        plt.colorbar()
        plt.savefig(figures_dir+'rf_confusion_matrix.png')
        plt.close()
        plt.figure()
        plt.imshow(svm_confusion_matrix)
        plt.title('SVM Confusion Matrix')
        plt.colorbar()
        plt.savefig(figures_dir+'svm_confusion_matrix.png')
        plt.close()
        plt.figure()
        plt.imshow(lr_confusion_matrix)
        plt.title('Logistic Regression Confusion Matrix')
        plt.colorbar()
        plt.savefig(figures_dir+'lr_confusion_matrix.png')
        plt.close()