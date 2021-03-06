import numpy as np
import math
from scipy.spatial.distance import cdist
from sklearn import preprocessing


class Problem:

    def __init__(self, minimized):
        self.minimized = minimized

    def fitness(self, sol, gamma):
        return 10*sol.shape[0]+np.sum(sol**2-10*np.cos(2*math.pi*sol))
        # return np.sum(sol**2)

    def worst_fitness(self):
        w_f = float('inf') if self.minimized else float('-inf')
        return w_f

    def is_better(self, first, second):
        if self.minimized:
            return first < second
        else:
            return first > second


class CentroidClassification(Problem):

    def __init__(self, minimized, X, y, verbose=False):
        Problem.__init__(self, minimized=minimized)
        self.X = X
        self.y = y
        self.verbose = verbose
        self.n_samples, self.n_features = self.X.shape[0], self.X.shape[1]
        self.unique_label = np.unique(self.y)
        self.n_classes = len(self.unique_label)

    def fitness(self, sol, gamma):
        centroids = np.reshape(sol, (self.n_classes, self.n_features))

        D = cdist(self.X, centroids)
        pseu = np.argmin(D, axis=1)
        loss = 1-np.mean(self.y == pseu)
        #
        # label_matrix = np.zeros((self.n_samples, self.n_classes))
        # for i in range(self.n_classes):
        #     label_matrix[self.y == self.unique_label[i], i] = 1
        # D_class = np.multiply(label_matrix, D)
        # D_class_max = np.max(D_class)
        # ave_dis = (np.sum(D)/self.n_samples)/D_class_max
        ave_dis=0

        return loss, loss, ave_dis


class CentroidClassificationLimit(Problem):

    def __init__(self, minimized, X, y, verbose=False):
        Problem.__init__(self, minimized=minimized)
        self.X = X
        self.y = y
        self.verbose = verbose
        self.n_samples, self.n_features = self.X.shape[0], self.X.shape[1]
        self.unique_label = np.unique(self.y)
        self.n_classes = len(self.unique_label)

    def fitness(self, sol, gamma):
        centroids = np.reshape(sol[0:self.n_features*self.n_classes], (self.n_classes, self.n_features))

        n_selected_features = int(sol[self.n_features*self.n_classes]*self.n_features)
        if n_selected_features == 0:
            return self.worst_fitness(), -1, -1
        # normalize = preprocessing.MinMaxScaler().fit(centroids)
        # centroids_n = normalize.transform(centroids)
        vars = np.var(centroids, axis=0)
        idx = np.argsort(vars)[::-1]
        X_selected = self.X[:, idx[0:n_selected_features]]
        centroids_selected = centroids[:, idx[0:n_selected_features]]

        D = cdist(X_selected, centroids_selected, metric='cityblock')
        pseu = np.argmin(D, axis=1)
        loss = 1-np.mean(self.y == pseu)
        #
        # label_matrix = np.zeros((self.n_samples, self.n_classes))
        # for i in range(self.n_classes):
        #     label_matrix[self.y == self.unique_label[i], i] = 1
        # D_class = np.multiply(label_matrix, D)
        # D_class_max = np.max(D_class)
        # ave_dis = (np.sum(D)/self.n_samples)/D_class_max
        ave_dis = 0

        return (1-gamma)*loss+gamma*sol[self.n_features*self.n_classes], loss, ave_dis


