import scipy.io
from sklearn import svm, preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from skfeature.utility.sparse_learning import construct_label_matrix, feature_ranking, generate_diagonal_matrix
import Problem
import numpy as np
import PSO
import time
from sklearn.metrics import balanced_accuracy_score
from numpy.linalg import norm

def main(dataset, run):
    '''
    :param dataset:
    :param run: run index
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
    full_tests = []

    to_print = 'Apply %d folds\n' %n_splits

    if n_features < 100:
        num_selected_features = [i for i in range(1, n_features, n_features/10)]
    else:
        num_selected_features = [i for i in range(10, 110, 10)]
    selected_test = np.array([0.0] * len(num_selected_features))

    count = 1
    exe_time = 0
    for train_index, test_index in skf.split(X, y):
        to_print += '=========Fold '+str(count)+'=========\n'
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # normalize data
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        # prepare for PSO
        max_range = X_train.max(axis=0)
        min_range = X_train.min(axis=0)
        max_pos = np.tile(max_range, (n_classes,))
        min_pos = np.tile(min_range, (n_classes,))
        max_vel = (max_pos-min_pos)/10
        min_vel = -max_vel
        length = n_features*n_classes
        n_part = 50
        n_iter = 100

        prob = Problem.CentroidClassification(minimized=True, X=X_train, y=y_train)
        swarm = PSO.Swarm(n_particle=n_part, length=length, problem=prob,
                          n_iterations=n_iter, max_pos=max_pos, min_pos=min_pos,
                          max_vel=max_vel, min_vel=min_vel)
        sol, fit, loss, dist = swarm.iterate()

        clf = svm.LinearSVC(random_state=1617)
        clf.fit(X_train, y_train)
        full_test = balanced_accuracy_score(y_test, clf.predict(X_test))
        full_tests.append(full_test)

        start = time.time()
        if alg == 'PSO':
            pos_max = 1 #np.min(1.0/(norm(X_train, axis=1)))/n_features_res**0.5
            pos_min = -pos_max
            prob = Problem.FeatureSelection(minimized=True, X=X_train, y=y_train, gamma=gamma,
                                            verbose=False, max_pos=pos_max)
            swarm = PSO.Swarm(n_particle=50, length=n_features_res*n_classes, problem=prob,
                              n_iterations=100, verbose=False, max_pos=pos_max, min_pos=pos_min)
            Weight, fitness = swarm.iterate()
            Weight = np.reshape(Weight[0:n_features*n_classes], (n_features, n_classes))

        elif alg == 'PSOL':
            pos_max = np.min(1.0/(norm(X_train, axis=1)))/n_features_res**0.5
            pos_min = -pos_max
            prob = Problem.FeatureSelectionLimit(minimized=True, X=X_train, y=y_train, gamma=gamma, verbose=False)
            n_particle = 30
            length = n_features_res*n_classes+1

            # create an initialization for PSO
            # accs = []
            # acc_max = -sys.float_info.max
            # acc_min = sys.float_info.max
            # f_train, f_test, f_y_train, f_y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=1617)
            # for idx_f in range(X_train.shape[1]):
            #     clf.fit(f_train[:, idx_f:idx_f+1], f_y_train)
            #     acc = balanced_accuracy_score(f_y_test, clf.predict(f_test[:, idx_f:idx_f+1]))
            #     if acc > acc_max:
            #         acc_max = acc
            #     if acc < acc_min:
            #         acc_min = acc
            #     accs.append(acc)
            # pop = []
            # for _ in range(n_particle):
            #     pos = []
            #     for idx_f in range(n_features):
            #         if np.random.rand() < (accs[idx_f]-acc_min)/(acc_max-acc_min):
            #             pos.extend(np.random.rand(n_classes))
            #         else:
            #             pos.extend(np.zeros(n_classes))
            #     pos.append(-1 + np.random.rand()*(1-(-1)))
            #     pop.append(np.array(pos))
            # swarm = PSO.Swarm(n_particle=n_particle, length=length, problem=prob, n_iterations=100, pop=pop)

            swarm = PSO.Swarm(n_particle=n_particle, length=length, problem=prob, n_iterations=100, verbose=True)
            Weight, fitness = swarm.iterate()
            f_ratio = (Weight[n_features_res * n_classes] + 1.0) / 2.0
            num_selected_features[count-1] = (int)(f_ratio * n_features)
            Weight = np.reshape(Weight[0:n_features*n_classes], (n_features, n_classes))

        elif alg == 'RFS':
            Weight = RFS.rfs(X_train, y_train, gamma=gamma, verbose=False)

        elif alg == 'GFS':
            Weight = GFS.gfs(X_train, y_train, lamb=gamma, verbose=False)

        exe_time += time.time()-start

        # to_print += 'Weight matrix: \n'
        # to_print += str(Weight)+'\n'
        idx = feature_ranking(Weight)
        to_print += 'Feature ranking: '
        for f_i in idx:
            to_print += str(f_i)+', '
        to_print += '\n'

        if alg == 'PSOL':
            num_feature = num_selected_features[count-1]
            X_train_selected = X_train[:, idx[0:num_feature]]
            X_test_selected = X_test[:, idx[0:num_feature]]
            # train a classification model with the selected features on the training dataset
            clf.fit(X_train_selected, y_train)
            # predict the class labels of test data
            y_predict = clf.predict(X_test_selected)
            # obtain the classification accuracy on the test data
            acc = balanced_accuracy_score(y_test, y_predict)
            selected_test[count-1] = acc
            to_print += 'Full test: '+str(full_test)+'\n'
            to_print += 'Select %d features \n' %num_feature
            to_print += 'Selected accuracy '+str(acc) +'\n'

        else:
            to_print += 'Full test: '+str(full_test)+'\n'
            for num_index, num_feature in enumerate(num_selected_features):
                X_train_selected = X_train[:, idx[0:num_feature]]
                X_test_selected = X_test[:, idx[0:num_feature]]
                # train a classification model with the selected features on the training dataset
                clf.fit(X_train_selected, y_train)
                # predict the class labels of test data
                y_predict = clf.predict(X_test_selected)
                # obtain the classification accuracy on the test data
                acc = balanced_accuracy_score(y_test, y_predict)
                selected_test[num_index] += acc
                to_print += 'Select %d features: %f\n' %(num_feature, acc)
        count = count+1

    to_print += '========= Final results =========\n'

    if alg == 'PSOL':
        to_print += 'Average full test: %f\n' %(np.mean(full_tests))
        to_print += 'Average selected features: %f\n' %(np.mean(num_selected_features))
        to_print += 'Average selected accuracy: %f\n' %(np.mean(selected_test))
    else:
        to_print += 'Average full test: %f\n' % (np.mean(full_tests))
        selected_test = np.array(selected_test)/n_splits
        for num_index, num_feature in enumerate(num_selected_features):
            to_print += "Average accuracy of %d features: %f\n" %(num_feature, selected_test[num_index])
    to_print += 'Running time: %f\n' % exe_time

    f_out = open(str(run)+'.txt', 'w')
    f_out.write(to_print)
    f_out.close()


if __name__ == '__main__':
    '''
    Main function to run feature selection algorithm
    argv[1]: the dataset to run
    argv[2]: run index
    '''
    import sys
    dataset = sys.argv[1]
    run = int(sys.argv[2])
    main(dataset, run)

