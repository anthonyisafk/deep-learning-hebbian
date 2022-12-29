from typing import List
from layer import *


class Model:
    nlayers: int      # number of layers
    layers:np.ndarray # Layer class instances
    ni: int           # number of input nodes on first layer
    no: int           # number of output nodes
    eta: np.float32   # learning rate

    def __init__(self, n_nodes:List[int], eta):
        self.nlayers = len(n_nodes)
        check_nlayers(self.nlayers)
        check_input_output_nodes(n_nodes[0], n_nodes[1])
        self.ni = n_nodes[0]
        self.no = n_nodes[1]
        self.eta = eta
        self.layers = np.ndarray(dtype=Layer, shape=(2))
        self.layers[0] = Layer('i', n_nodes[0], 1, n_nodes[1])
        self.layers[1] = Layer('o', n_nodes[1], n_nodes[0], 1, mu=0, stddev=0.1/n_nodes[1])


    def train(self, xs):
        nsamples = len(xs)
        il = self.layers[0]
        ol = self.layers[1]
        epoch = 0
        while epoch < 10:
            print(f"\n ** Epoch : {epoch}\n")
            for p in range(nsamples):
                x = xs[p]
                self.set_ys(x)
                ol.set_xprime(x)
                for j in range(self.no):
                    # ol.xprime[j] = x_to_xprime(ol.w, x, ol.y, j)
                    ol.dw[j] = self.eta * (ol.y[j] * x - ol.y[j] * ol.xprime[j])
                    ol.w[j] += ol.dw[j]
            epoch += 1
            print(ol.w[5])

    def set_ys(self, x):
        self.layers[0].y = self.layers[0].get_y(x)
        self.layers[1].y = self.layers[1].get_y(self.layers[0].y)

