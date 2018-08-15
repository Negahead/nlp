import numpy as np
import pandas as pd


def load_data():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values

    p = Perceptron()
    p.fit(X, y)
    print(p.w_)


class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        """
        :param eta:  float, learning rate (between 0.0 and 1.0)
        :param n_iter: passes over the training dataset
        :param random_state: int, Random number generator seed for random weight initialization
        """
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_ = None  # weight after fitting
        self.errors_ = []  # number of mis-classifications (updates) in each epoch

    def fit(self, X, y):
        """

        :param X: {array-like}, shape=[n_sample, n_features]
        :param y: array-like, shape=[n_samples], target values.
        :return:
        """
        rgen = np.random.RandomState(self.random_state)
        # the probability density function of the normal distribution, loc is the mean, scale is the standard deviation,
        # size is the output shape, int or tuple of ints
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.normal.html#numpy.random.RandomState.normal
        # self.w_[0] represents the so-called bias unit
        # this vector contains small random numbers drawn from a normal distribution
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        predict the class label for weight update
        :param X:
        :return:
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

if __name__ == '__main__':
    load_data()