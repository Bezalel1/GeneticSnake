import numpy as np
from typing import Callable, NewType

Activation = NewType('Activation', Callable[[np.ndarray], np.ndarray])
sigmoid = Activation(lambda X: 1. / (1. + np.exp(-X)))
linear = Activation(lambda X: X)
relu = Activation(lambda X: np.maximum(0, X))
leaky_relu = Activation(lambda X: np.where(X > 0, X, X * 0.01))
tanh = Activation(lambda X: np.tanh(X))


class NN:
    """
    simple representation of neural network for the genetic algorithm
    """

    def __init__(self, W, activation: Activation = sigmoid, alpha=1e-7, landa=0., file_name='W') -> None:
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
        return np.array([np.random.rand(l1, l0 + 1) * 2 - 1 for l0, l1 in zip(layers[:-1], layers[1:])],
                        dtype=np.object)
