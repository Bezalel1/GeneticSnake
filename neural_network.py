import numpy as np
import pandas as pd
import scipy as sc
from typing import Callable, NewType

Activation = NewType('Activation', Callable[[np.ndarray], np.ndarray])
sigmoid = Activation(lambda X: 1.0 / (1.0 + np.exp(-X)))
linear = Activation(lambda X: X)
relu = Activation(lambda X: np.maximum(0, X))
leaky_relu = Activation(lambda X: np.where(X > 0, X, X * 0.01))
tanh = Activation(lambda X: np.tanh(X))


class NN:
    """
    simple representation of neural network for the genetic algorithm
    """

    def __init__(self, W, activation: Activation = sigmoid, alpha=1e-7, landa=0., file_name='W') -> None:
        """
        # * 2 * eps - eps
        """
        super().__init__()
        self.W = W
        self.alpha, self.landa = alpha, landa
        self.activation = activation
        self.file_name = file_name

    def feedforward(self, X: np.ndarray) -> list:
        H = [X.T]
        for w in self.W:
            H[-1] = np.insert(H[-1], 0, np.ones((H[-1].shape[1]), dtype=H[-1].dtype), axis=0)
            H.append(self.activation(w @ H[-1]))
        return H

    def predict(self, X) -> np.ndarray:
        H = self.feedforward(X)
        p = np.argmax(H[-1], axis=0)
        return p

    def backpropagation(self, H, y):
        dW = self.grad(H, y)
        for w, dw in zip(self.W, dW):
            w[:] -= self.alpha * dw

    def grad(self, H, y):
        m, k = y.shape[0], H[-1].shape[0]
        K = np.arange(k)
        delta = [H[-1] - np.array(y == K[:, None])]
        dW = [delta[0] @ H[-2].T]
        dW[0][1:, :] += self.landa * self.W[-1][1:, :]
        dW[0] /= m

        for h0, h1, w0, w1 in zip(H[:-2][::-1], H[1:-1][::-1], self.W[:-1][::-1], self.W[1:][::-1]):
            delta.insert(0, (w1.T[1:, :] @ delta[0]) * (h1[1:, :] * (1 - h1[1:, :])))
            dW.insert(0, delta[0] @ h0.T)
            dW[0][:, 1:] += self.landa * w0[:, 1:]
            dW[0] /= m

        return dW

    def fit(self, X: np.ndarray, y, max_iter=1000):
        for i in range(max_iter):
            H = self.feedforward(X)
            self.backpropagation(H, y)

    def cost(self, X: np.ndarray, y):
        H = self.feedforward(X)
        a = H[-1]
        k, m = a.shape
        K = np.arange(k)
        pos = np.array(y == K[:, None])
        J = -(np.sum(np.log(a[pos])) + np.sum(np.log(1 - a[~pos])))
        J += (self.landa / 2) * np.sum([np.sum(w[:, 1:] ** 2) for w in self.W])
        J /= m
        return J

    def save(self):
        np.save(self.file_name, self.W)

    def load(self):
        self.W = np.load(self.file_name, allow_pickle=True)

    @staticmethod
    def init_W(layers):
        # np.random.seed(98) , * 0.12 * 2 - 0.12
        return np.array([np.random.rand(l1, l0 + 1) * 0.24 - 0.12 for l0, l1 in zip(layers[:-1], layers[1:])],
                        dtype=np.object)


if __name__ == '__main__':
    df = pd.read_csv('/home/bb/Downloads/data/iris.data')
    from scipy.stats import zscore

    # z = np.abs(zscore(df.iloc[:, :4]))
    z = np.abs(zscore(df.iloc[:, 1]))

    median = df.iloc[np.where(z <= 3)[0], 1].median()
    df.iloc[np.where(z > 3)[0], 1] = np.nan
    df.fillna(median, inplace=True)

    df0 = df.sample(frac=0.96, random_state=42)
    holdout = df.drop(df0.index)
    # Separate the feature columns (first 4) from the labels column (5th column)
    x = df0.iloc[:, :4]
    y = df0.iloc[:, 4]
    x_standard = x.apply(zscore)
    species_names = np.unique(np.array(y))

    # one hot encode the labels since they are categorical
    y_cat = pd.get_dummies(y, prefix='cat')
    y_cat.sample(10, random_state=42)
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x_standard, y_cat, test_size=0.5, random_state=42)
    x_train, y_train = x_train.to_numpy(), np.argmax(y_train.to_numpy(), axis=1)
    x_test, y_test = x_test.to_numpy(), np.argmax(y_test.to_numpy(), axis=1)

    file_name = 'W.npy'
    model = NN(np.load(file_name, allow_pickle=True), alpha=0.0005, landa=0.1)  # NN.init_W([4, 2, 2, 3])
    # model = NN(NN.init_W([4, 10, 6, 3]), alpha=2, landa=3)  # NN.init_W([4, 2, 2, 3])

    print(model.cost(x_train, y_train))
    model.fit(x_train, y_train, max_iter=100)
    print(model.cost(x_train, y_train))
    model.fit(x_train, y_train, max_iter=1000)
    print(model.cost(x_train, y_train))
    model.fit(x_train, y_train, max_iter=1000)
    print(model.cost(x_train, y_train))
    model.save()
    # print('------------------------------------')
    # # print(model.W[0])
    print(np.hstack((model.predict(x_test).reshape((-1, 1)), y_test.reshape((-1, 1)))))
    # print(np.mean(y))
    print(model.cost(x_test, y_test))
