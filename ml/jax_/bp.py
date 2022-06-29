import jax
import jax.numpy as np


class BPNN:
    __k = jax.random.PRNGKey(0)

    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        
        self.weights = jax.random.normal(self.__k, (n_hidden, n_input))
        self.bias = jax.random.normal(self.__k, n_hidden)
        self.weights_output = jax.random.normal(self.__k, (n_output, n_hidden))
        self.bias_output = jax.random.normal(self.__k, n_output)

    def forward(self, x):
        ...

    def backward(self, x, y, learning_rate):
        ...

    def train(self, x, y, learning_rate):
        self.forward(x)
        self.backward(x, y, learning_rate)

    def predict(self, x):
        return self.forward(x)

    def save(self, filename):
        np.savez(filename, self.weights, self.bias, self.weights_output, self.bias_output)

    def load(self, filename):
        npz = np.load(filename)
        self.weights = npz['arr_0']
        self.bias = npz['arr_1']
        self.weights_output = npz['arr_2']
        self.bias_output = npz['arr_3']

    def __repr__(self):
        return f'BPNN(n_input={self.n_input}, n_hidden={self.n_hidden}, n_output={self.n_output})'
