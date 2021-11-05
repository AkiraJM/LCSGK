# -----------------------------------------------------------------
# kernel_evaluation.py
#
# This file contains the code of cross-validation.
# November 5, Jianming Huang
# -----------------------------------------------------------------
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC, SVR


# 10-CV for linear svm with sparse feature vectors and hyperparameter selection.
def linear_svm_evaluation(all_feature_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False, split_random_state = 20):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_random_state)#KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(np.zeros(len(classes)), classes):#(list(range(len(classes)))):
            # Sample 10% split from training split for validation.
            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state = split_random_state)
            best_val_acc = 0.0
            best_gram_matrix = all_feature_matrices[0]
            best_c = C[0]

            for gram_matrix in all_feature_matrices:
                train = gram_matrix[train_index]
                val = gram_matrix[val_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = LinearSVC(C=c, tol=0.01, dual=False)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        # Get test acc.
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index]
            test = best_gram_matrix[test_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = LinearSVC(C=best_c, tol=0.01, dual=False)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
        #         np.array(test_accuracies_complete).std())
        return test_accuracies_all, test_accuracies_complete
    else:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
        return test_accuracies_all


# 10-CV for kernel svm and hyperparameter selection.
def kernel_svm_evaluation(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False, split_random_state = 20):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_random_state)#KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(np.zeros(len(classes)), classes):
            # Determine hyperparameters
            train_index, val_index = train_test_split(train_index, test_size=0.1, random_state = split_random_state)#, stratify=classes[train_index])
            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                train = gram_matrix[train_index, :]
                train = train[:, train_index]
                val = gram_matrix[val_index, :]
                val = val[:, train_index]

                c_train = classes[train_index]
                c_val = classes[val_index]

                for c in C:
                    clf = SVC(C=c, kernel="precomputed", tol=0.001)
                    clf.fit(train, c_train)
                    val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
        #         np.array(test_accuracies_complete).std())
        return test_accuracies_all, test_accuracies_complete
    else:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
        return test_accuracies_all


def kernel_svm_evaluation_nested(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False, split_random_state = 20):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_random_state)#KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(np.zeros(len(classes)), classes):
            # Determine hyperparameters
            #train_index, val_index = train_test_split(train_index, test_size=0.1, random_state = split_random_state)#, stratify=classes[train_index])

            best_val_acc = 0.0
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                for c in C:
                    accs = []
                    kf2 = StratifiedKFold(n_splits=10, shuffle=True, random_state=split_random_state)
                    for train_index2, val_index in kf2.split(np.zeros(len(train_index)), classes[train_index]):
                        inner_train_index = train_index[train_index2]
                        inner_test_index = train_index[val_index]

                        train = gram_matrix[inner_train_index, :]
                        train = train[:, inner_train_index]
                        val = gram_matrix[inner_test_index, :]
                        val = val[:, inner_train_index]

                        c_train = classes[inner_train_index]
                        c_val = classes[inner_test_index]


                        clf = SVC(C=c, kernel="precomputed", tol=0.001)
                        clf.fit(train, c_train)
                        val_acc = accuracy_score(c_val, clf.predict(val)) * 100.0
                        accs.append(val_acc)
                    val_acc = np.mean(np.array(accs))

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVC(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = accuracy_score(c_test, clf.predict(test)) * 100.0

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
        #         np.array(test_accuracies_complete).std())
        return test_accuracies_all, test_accuracies_complete
    else:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
        return test_accuracies_all

def kernel_svr_evaluation_nested(all_matrices, classes, num_repetitions=10,
                          C=[10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3], all_std=False, split_random_state = 20):
    # Acc. over all repetitions.
    test_accuracies_all = []
    # All acc. over all folds and repetitions.
    test_accuracies_complete = []

    for i in range(num_repetitions):
        # Test acc. over all folds.
        test_accuracies = []
        kf = KFold(n_splits=10, shuffle=True, random_state=split_random_state)#KFold(n_splits=10, shuffle=True)

        for train_index, test_index in kf.split(np.zeros(len(classes)), classes):
            # Determine hyperparameters
            #train_index, val_index = train_test_split(train_index, test_size=0.1, random_state = split_random_state)#, stratify=classes[train_index])

            best_val_mse = np.inf
            best_gram_matrix = all_matrices[0]
            best_c = C[0]

            for gram_matrix in all_matrices:
                for c in C:
                    mses = []
                    kf2 = KFold(n_splits=10, shuffle=True, random_state=split_random_state)
                    for train_index2, val_index in kf2.split(np.zeros(len(train_index)), classes[train_index]):
                        inner_train_index = train_index[train_index2]
                        inner_test_index = train_index[val_index]

                        train = gram_matrix[inner_train_index, :]
                        train = train[:, inner_train_index]
                        val = gram_matrix[inner_test_index, :]
                        val = val[:, inner_train_index]

                        c_train = classes[inner_train_index]
                        c_val = classes[inner_test_index]


                        clf = SVR(C=c, kernel="precomputed", tol=0.001)
                        clf.fit(train, c_train)
                        val_mse = mean_squared_error(c_val, clf.predict(val), squared=False)
                        mses.append(val_mse)
                    val_mse = np.mean(np.array(mses))

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_c = c
                        best_gram_matrix = gram_matrix

            # Determine test accuracy.
            train = best_gram_matrix[train_index, :]
            train = train[:, train_index]
            test = best_gram_matrix[test_index, :]
            test = test[:, train_index]

            c_train = classes[train_index]
            c_test = classes[test_index]
            clf = SVR(C=best_c, kernel="precomputed", tol=0.001)
            clf.fit(train, c_train)
            best_test = mean_squared_error(c_test, clf.predict(test), squared=False)

            test_accuracies.append(best_test)
            if all_std:
                test_accuracies_complete.append(best_test)
        test_accuracies_all.append(float(np.array(test_accuracies).mean()))

    if all_std:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std(),
        #         np.array(test_accuracies_complete).std())
        return test_accuracies_all, test_accuracies_complete
    else:
        # return (np.array(test_accuracies_all).mean(), np.array(test_accuracies_all).std())
        return test_accuracies_all