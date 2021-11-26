import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from hyperopt import hp

sol = pandas.read_csv("curated-solubility-dataset.csv")
Y = np.array(sol['Solubility'])

proplist = ['HeavyAtomCount',
       'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
       'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',
       'NumAliphaticRings', 'RingCount']

X = np.array([list(sol[prop]/sol['MolWt']) for prop in proplist]) # Many properties are extensive, so we divide by the molecular weight
X = np.insert(X, 0, list(np.log(sol['MolWt'])), axis=0) # add the log MolWt as well

X_train, X_test, y_train, y_test = train_test_split(np.transpose(X), Y, test_size=0.2, random_state=0)

sample_size = 2000  # len(X_train)
X_train = X_train[:sample_size][:sample_size]
y_train = y_train[:sample_size]

# Define vector of sigmas as the standard deviation of a feature
sigmas = []
f = X_train.shape[1]
for s in range(f):
    sigmas.append(1*np.std(X_train[:][s]))


def vector_kernal(X, x1, sig):
    # X is the matrix of training data with shape (N, f) where N is the number
    # of data points and f is the number of features for each data point
    # sig[f] is the standard deviation of feature f as an initial guess, this can be optimised

    vector_solution = np.exp(
        -(((abs(np.array(np.transpose(X)) - np.array(np.transpose(x1)))) ** 2) / (2 * np.array(sig) ** 2)).sum(axis=1))

    return vector_solution


def K_fold(X, y, K, sigmas, lam):
    # This function splits the training data into K folds and trains K-1 models
    # and tests the performance on 1 of the folds. This is repeated for each fold.
    # Returns an array of size K with the error or R^2 value from each run.#

    rs = []
    errors = []

    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    kf.get_n_splits(X)
    print(X.shape, y.shape)

    print(kf)
    for train_index, test_index in kf.split(X):
        X_train_k, X_test_k = X[train_index], X[test_index]
        y_train_k, y_test_k = y[train_index], y[test_index]

        X_train_k = np.transpose(X_train_k)
        X_test_k = np.transpose(X_test_k)
        print(X_train_k.shape, X_test_k.shape)
        print(y_train_k.shape, y_test_k.shape)

        N = X_train_k.shape[1]  # N is the number of data vectors in train set
        f = X_train_k.shape[0]  # f is the number of features
        K = np.zeros((N, N))  # Training kernel matrix

        for i in range(N):
            K[i, :] = vector_kernal(X_train_k, X_train_k[:, i], sigmas)

        I = np.identity(N)
        K_prime = K + lam * I

        # C is a vector that is the same size as Y
        C = np.linalg.lstsq(K_prime, y_train_k)[0]  # Solves the equation C = (K+lam*I)^-1.Y

        # Test model
        N_test = X_test_k.shape[1]
        K_test = np.zeros((N, N_test))

        for j in range(N_test):
            K_test[:, j] = vector_kernal(X_train_k, X_test_k[:, j], sigmas)

        Y_pred = np.transpose(K_test) @ C
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_test_k, Y_pred)
        error = np.sqrt(sum((y_test_k - Y_pred)**2)/len(y_test_k))

        rs.append(r_value)
        errors.append(error)

        Y_pred_train = K @ C
        slope, intercept, r_value, p_value, std_err = stats.linregress(y_train_k, Y_pred_train)
        print("Training set R^2:", r_value)

        error = np.average(errors)

        #plt.scatter(y_test_k, Y_pred)
        #plt.plot([-15, 5], [-15, 5], 'k--')
        #plt.show()

    return rs, error


rs, error = K_fold(X_train, y_train, 4, sigmas, 0.03)

