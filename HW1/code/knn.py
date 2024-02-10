import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """

    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided

        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """

        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)

        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)

    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        num_test_samples, num_features = X.shape
        num_train_samples, _ = self.train_X.shape
        distances = np.zeros((num_test_samples, num_train_samples))
        for i in range(num_test_samples):
            for j in range(num_train_samples):
                distances[i, j] = np.sum(np.abs(X[i] - self.train_X[j]))

        return distances

    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        num_test_samples, num_features = X.shape
        num_train_samples, _ = self.train_X.shape
        distances = np.zeros((num_test_samples, num_train_samples))
        for i in range(num_test_samples):
            distances[i] = np.sum(np.abs(self.train_X - X[i]), axis=1)

        return distances

    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run

        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        """
        YOUR CODE IS HERE
        """
        num_test_samples, num_features = X.shape
        num_train_samples, _ = self.train_X.shape
        distances = np.abs(self.train_X - X[:, np.newaxis]).sum(axis=2)

        return distances

    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.array(["0"] * n_test)

        """
        YOUR CODE IS HERE
        """
        for i in range(n_test):
            k_nearest = np.argsort(distances[i])[: self.k]
            k_nearest_labels = self.train_y[k_nearest]
            class_0 = len(np.where(k_nearest_labels == "0")[0])
            class_1 = len(np.where(k_nearest_labels == "1")[0])
            if class_0 > class_1:
                prediction[i] = "0"
            elif class_1 > class_0:
                prediction[i] = "1"
            else:
                prediction[i] = np.random.choice(["0", "1"])

        return prediction

    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case

        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.array(["0"] * n_test)

        """
        YOUR CODE IS HERE
        """
        for i in range(n_test):
            k_nearest = np.argsort(distances[i])[: self.k]
            k_nearest_labels = self.train_y[k_nearest]
            unique_classes, counts_classes = np.unique(
                k_nearest_labels, return_counts=True
            )
            max_index = np.argmax(counts_classes)
            prediction[i] = unique_classes[max_index]

        return prediction
