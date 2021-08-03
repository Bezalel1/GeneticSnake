import numpy as np
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

    def __init__(self, layers: list) -> None:
        """
        :param layers: [input size,l0,l1,...,output size], list of layers for the neural network

        # * 2 * eps - eps
        """
        super().__init__()
        self.W: list[np.ndarray] = [np.random.rand(l1, l0 + 1) for l0, l1 in zip(layers[:-1], layers[1:])]

    def feedforward(self, X: np.ndarray, activation: Activation):
        H = [X.T]
        for w in self.W:
            H[-1] = np.insert(H[-1], 0, np.ones((H[-1].shape[1]), dtype=H[-1].dtype), axis=0)
            H.append(activation(w @ H[-1]))
        return H

    def predict(self, X, activation):
        h = self.feedforward(X, activation)[-1]
        p = np.argmax(h, axis=0)
        return p


class GA:

    def __init__(self, layers: list) -> None:
        super().__init__()

        self.nn = NN(layers)


def crossover(gen1, gen2):

    pass


def mutation():
    pass


def evaluation():
    pass


def temp():
    """
    # 2,[2,3,4]
    # X=mx2, w1=2x3, w2=3x3, w3=4x4
    #X_T=3xm,a1=(2+1)xm, a2=(3+1)xm, a3=4xm

    20,[16,10,4]
    X=20x1, w1=16x21, w2=10x17, w3=4x11
    X_=21x1,a1=16x1, a2=10x1, a3=4x1
    """
    X = np.empty((1, 20))
    network = NN([20, 16, 10, 4])
    print(network.predict(X, relu))


temp()
