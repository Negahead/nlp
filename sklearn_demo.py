from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def plot_decision_region(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # arange : [start, stop) with step
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl,
                    edgecolors='black')
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolors='black', alpha=1.0,
                    linewidths=1, marker='o',
                    s=100, label='test set')


def sklearn_demo():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    # returns the three unique class labels stored in iris.target
    print('Class labels:', np.unique(y))  # [0 1 2]
    # Using the training_test_split function from scikit-learn's model_selection module,
    # we randomly split the X and y arrays into 30 percent test data(45 samples) and 70 percent
    # training data(105 examples), Note that the training_test_split function already shuffles the training
    # sets internally before splitting, otherwise, all class 0 and class 1 samples would have ended up in
    # the training set. Via the random_state parameter, we provided a fixed random seed for the internal pseudo-random
    # number generator that is used for shuffling the datasets prior to splitting
    # stratify=y, means that the train_test_split method returns training and test subsets that have the same
    # proportions of class labels as the input dataset.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    print("labels counts in y: ", np.bincount(y))  # [50, 50, 50] corresponding to 0 , 1 , 2
    print('Labels counts in y_train: ', np.bincount(y_train))  # 50 * 0.7 = 35
    print('Labels counts in y_test: ', np.bincount(y_test))  # 50 * 0.3 = 15
    # Compute the mean and standard deviation on a training set so as to be able to later reapply the same
    # transformation on the training set.
    scaler = StandardScaler().fit(X_train)
    # Standardization of datasets is a common requirement for many machine learning estimators implemented in
    # scikit-learn, they might behave badly if the individual features do not more or less look like standard
    # normally distributed data: Gaussian with zero mean and unit variance.
    X_train_std = scaler.transform(X_train)
    print(X_train_std.mean(axis=0))
    X_test_std = scaler.transform(X_test)
    print(X_test_std.mean(axis=0))
    print("Hello world")
    # use the random_state parameter to ensure to reproducibility of the initial shuffling of the training dataset
    # after each epoch
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
    # train the model via the fit method.
    ppn.fit(X_train_std, y_train)
    # having trained the model in scikit-learn, we can make predictions via the predict method.
    y_pred = ppn.predict(X_test_std)
    print("y_test with type {0} is ".format(type(y_test)))
    print(y_test)
    print("y_pred with type {} is ".format(type(y_pred)))
    print(y_pred)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    # calculate the classification accuracy of the perceptron on the test set.
    print("Accuracy: %.2f " % accuracy_score(y_test, y_pred))
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_region(X=X_combined_std, y=y_combined, classifier=ppn, test_idx=range(105, 150))
    plt.xlabel("petal length")
    plt.ylabel("petal width")
    plt.legend(loc='upper left')
    plt.show()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cost_1(z):
    return -np.log(sigmoid(z))


def cost_0(z):
    return -np.log(1 - sigmoid(z))



def plot_sigmoid():
    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 0.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_sigmoid()