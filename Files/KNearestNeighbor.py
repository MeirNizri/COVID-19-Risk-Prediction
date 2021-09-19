import scipy
import numpy as np

class KNearestNeighbor(object):
    """ a kNN classifier with several distance function"""

    def __init__(self):
        pass

    def train(self, X, y):
        """
        For k-nearest neighbors this is just memorizing the training data.
        
        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, dist_func="l2"):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - dist_func: Determines which function to use to compute distances between training points and testing points.
        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the test data, where y[i] is the predicted label for the test point X[i].
        """
        self.k = k
        self.dist_f = dist_func

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # calculate the distance matrix according to the distance function
        if dist_func == "l2":
            dists = scipy.spatial.distance_matrix(X, self.X_train)
        elif dist_func == "l1":
            dists = scipy.spatial.distance_matrix(X, self.X_train, p=1)
        elif dist_func == "linf":
            dists = scipy.spatial.distance_matrix(X, self.X_train, p=np.inf)
        else:
            raise ValueError('Invalid value %d for dist_func' % dist_func)
        
        # calculate prediction on X
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            k_nearest_idxs = np.argsort(dists[i, :])[:k]
            closest_y = self.y_train[k_nearest_idxs]
            y_pred[i] = np.argmax(np.bincount(closest_y))
            
        return y_pred

    def getK(self):
        return self.k

    def get_func(self):
        return self.dist_f