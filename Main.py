import scipy.io
from sklearn import svm, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking, generate_diagonal_matrix
import Problem
import numpy as np
import PSO
import time
from sklearn.metrics import balanced_accuracy_score
from numpy.linalg import norm
from scipy.spatial.distance import cdist

def main(dataset, run, alg):
    '''
    :param dataset:
    :param run: run index
    :param alg
    :return:
    '''

    # set seed for random, this seed is for PSO in PSO and PSOL
    # to evolve the solutions
    np.random.seed(1617*run)

    # load data
    mat = scipy.io.loadmat('/home/nguyenhoai2/Grid/data/FSMathlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]

    # ensure that y label start from 0, not 1
    num_class, count = np.unique(y, return_counts=True)
    n_classes = np.unique(y).shape[0]
    min_class = np.min(count)
    if np.max(y) >= len(num_class):
        y = y-1
    n_features = X.shape[1]

    # ensure that the division is the same for all algorithms, in all runs
    n_splits = min(min_class, 10)
    skf = StratifiedKFold(n_splits=n_splits, random_state=1617)

    to_print = 'Apply %d folds\n' %n_splits

    if alg=='PSO':
        if n_features < 100:
            num_selected_features = [i for i in range(1, n_features, n_features/10)]
        else:
            num_selected_features = [i for i in range(10, 110, 10)]
        selected_test_svm = np.array([0.0] * len(num_selected_features))
        selected_test_knn = np.array([0.0] * len(num_selected_features))
        full_test_svm = []
        full_test_knn = []

        for train_index, test_index in skf.split(X, y):
            to_print += '=========Fold ' + str(count) + '=========\n'
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # normalize data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)
            normalize = preprocessing.MinMaxScaler().fit(X_train)
            X_train = normalize.transform(X_train)
            X_test = normalize.transform(X_test)
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            # full results
            clf = svm.LinearSVC(random_state=1617)
            clf.fit(X_train, y_train)
            full_test_svm.append(np.mean(clf.predict(X_test) == y_test))

            clf = KNeighborsClassifier()
            clf.fit(X_train, y_train)
            full_test_knn.append(np.mean(clf.predict(X_test) == y_test))

            # prepare for PSO
            n_part = 50
            n_iter = 1000
            max_range = X_train.max(axis=0)
            min_range = X_train.min(axis=0)
            max_pos = np.tile(max_range, (n_classes,))
            min_pos = np.tile(min_range, (n_classes,))
            length = n_features * n_classes
            max_vel = np.array([0.05] * max_pos.shape[0])
            min_vel = -max_vel
            prob = Problem.CentroidClassification(minimized=True, X=X_train, y=y_train)
            swarm = PSO.Swarm(n_particle=n_part, length=length, problem=prob,
                              n_iterations=n_iter, max_pos=max_pos, min_pos=min_pos,
                              max_vel=max_vel, min_vel=min_vel)
            sol, fit, loss, dist = swarm.iterate()

            centroids = np.reshape(sol, (n_classes, n_features))
            normalize = preprocessing.MinMaxScaler().fit(centroids)
            centroids_n = normalize.transform(centroids)
            vars = np.var(centroids_n, axis=0)
            idx = np.argsort(vars)[::-1]

            for index, n_selected in enumerate(num_selected_features):
                X_train_selected = X_train[:, idx[0:n_selected]]
                X_test_selected = X_test[:, idx[0:n_selected]]

                # D_train = cdist(X_train, centroids)
                # pseu_train = np.argmin(D_train, axis=1)
                # print("Training accuracy: %f" %np.mean(y_train == pseu_train))
                #
                # D_test = cdist(X_test, centroids)
                # pseu_test = np.argmin(D_test, axis=1)
                # print("Testing accuracy: %f" %np.mean(y_test == pseu_test))

                clf = svm.LinearSVC(random_state=1617)
                clf.fit(X_train_selected, y_train)
                selected_test_svm[index] += np.mean(clf.predict(X_test_selected) == y_test)

                clf = KNeighborsClassifier()
                clf.fit(X_train_selected, y_train)
                selected_test_knn[index] += np.mean(clf.predict(X_test_selected) == y_test)

        selected_test_svm = np.array(selected_test_svm) / n_splits
        selected_test_knn = np.array(selected_test_knn) / n_splits
        test_svm = np.mean(full_test_svm)
        test_knn = np.mean(full_test_knn)

        print "-------------------KNN----------------------"
        print 'Full test: %f' % test_knn
        for n_features, selected_test in zip(num_selected_features, selected_test_knn):
            print '%d features: %f' % (n_features, selected_test)

        print "-------------------SVM----------------------"
        print 'Full test: %f' % test_svm
        for n_features, selected_test in zip(num_selected_features, selected_test_svm):
            print '%d features: %f' % (n_features, selected_test)

    elif alg=='PSOL':
        num_selected_features = []
        selected_test_knn = []
        selected_test_svm = []
        selected_test_embed = []
        full_test_svm = []
        full_test_knn = []

        for train_index, test_index in skf.split(X, y):
            to_print += '=========Fold ' + str(count) + '=========\n'
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # normalize data
            scaler = preprocessing.StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)
            normalize = preprocessing.MinMaxScaler().fit(X_train)
            X_train = normalize.transform(X_train)
            X_test = normalize.transform(X_test)
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            # full results
            clf = svm.LinearSVC(random_state=1617)
            clf.fit(X_train, y_train)
            full_test_svm.append(np.mean(clf.predict(X_test) == y_test))

            clf = KNeighborsClassifier()
            clf.fit(X_train, y_train)
            full_test_knn.append(np.mean(clf.predict(X_test) == y_test))

            # prepare for PSO
            n_part = 50
            n_iter = 1000
            max_range = X_train.max(axis=0)
            min_range = X_train.min(axis=0)
            max_pos = np.tile(max_range, (n_classes,))
            max_pos = np.append(max_pos, 1.0)
            min_pos = np.tile(min_range, (n_classes,))
            min_pos = np.append(min_pos, 0.0)
            length = n_features * n_classes + 1
            max_vel = np.array([0.05] * max_pos.shape[0])
            min_vel = -max_vel

            prob = Problem.CentroidClassificationLimit(minimized=True, X=X_train, y=y_train)
            swarm = PSO.Swarm(n_particle=n_part, length=length, problem=prob,
                              n_iterations=n_iter, max_pos=max_pos, min_pos=min_pos,
                              max_vel=max_vel, min_vel=min_vel)
            sol, fit, loss, dist = swarm.iterate()

            centroids = np.reshape(sol[0:n_features * n_classes], (n_classes, n_features))
            n_selected_features = int(sol[n_features * n_classes] * n_features)
            # normalize = preprocessing.MinMaxScaler().fit(centroids)
            # centroids_n = normalize.transform(centroids)
            vars = np.var(centroids, axis=0)
            idx = np.argsort(vars)[::-1]
            X_train_selected = X_train[:, idx[0:n_selected_features]]
            X_test_selected = X_test[:, idx[0:n_selected_features]]
            centroids_selected = centroids[:, idx[0:n_selected_features]]

            num_selected_features.append(n_selected_features)

            D = cdist(X_test_selected, centroids_selected, metric='cityblock')
            pseu = np.argmin(D, axis=1)
            selected_test_embed.append(np.mean(pseu == y_test))

            clf = svm.LinearSVC(random_state=1617)
            clf.fit(X_train_selected, y_train)
            selected_test_svm.append(np.mean(clf.predict(X_test_selected) == y_test))

            clf = KNeighborsClassifier()
            clf.fit(X_train_selected, y_train)
            selected_test_knn.append(np.mean(clf.predict(X_test_selected) == y_test))
            print selected_test_embed[-1]
            print full_test_knn[-1], selected_test_knn[-1]
            print full_test_svm[-1], selected_test_svm[-1]

        print "-------------------Centroid----------------------"
        print 'Centroid: %f' % np.mean(selected_test_embed)

        print "-------------------KNN----------------------"
        print 'Full test: %f' % np.mean(full_test_knn)
        print 'Select %f features with accuracy of %f' %(np.mean(num_selected_features), np.mean(selected_test_knn))

        print "-------------------SVM----------------------"
        print 'Full test: %f' % np.mean(full_test_svm)
        print 'Select %f features with accuracy of %f' %(np.mean(num_selected_features), np.mean(selected_test_svm))

    else:
        raise Exception('Algorithm %s has not been implemented!!!!' %alg)

if __name__ == '__main__':
    '''
    Main function to run feature selection algorithm
    argv[1]: the dataset to run
    argv[2]: run index
    argv[3]: algorithm type: PSO, PSOL
    '''
    import sys
    dataset = sys.argv[1]
    alg = sys.argv[2]
    run = int(sys.argv[3])
    main(dataset, run, alg)