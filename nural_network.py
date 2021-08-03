import numpy as np
from typing import Callable, NewType

# ActivationFunction=

class NN:
    def __init__(self, layers: np.ndarray, eps=0.12) -> None:
        super().__init__()
        self.W = [np.random.rand(l1, l0 + 1) * 2 * eps - eps for l0, l1 in zip(layers[:-1], layers[1:])]

    def feedforward(self, X: np.ndarray, activation: callable):
        H = [X.T]
        for w in self.W:
            H[-1] = np.insert(H[-1], 0, np.ones((H[-1].shape[1]), dtype=H[-1].dtype), axis=0)
            H.append(activation(w @ H[-1]))
        return H

    def predict(self, X, activation):
        h = self.feedforward(X, activation)[-1]
        p = np.argmax(h, axis=0)
        return p
