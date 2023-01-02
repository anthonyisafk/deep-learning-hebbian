from typing import List
from layer import *

MAX_EPOCHS = 100
TOLER = 5e-5

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
        self.layers[1] = Layer('o', n_nodes[1], n_nodes[0], 1, mu=0, stddev=1e-10)


    def train(self, xs):
        nsamples = len(xs)
        ol = self.layers[1]
        epoch = 0
        fit = False
        while epoch < MAX_EPOCHS and not fit:
            print(f"\n ** Epoch : {epoch}\n")
            dws = np.zeros(shape=(self.no, self.ni))
            xsshuf = np.random.permutation(xs)
            for p in range(nsamples):
                x = xsshuf[p]
                self.set_ys(x)
                ol.set_d()
                for j in range(self.no):
                    # ol.d[j, :] = np.dot(ol.w[:j, :].T, ol.y[:j])
                    ol.dw[j, :] = self.eta * ol.y[j] * (x - ol.d[j, :])
                    ol.w[j, :] += ol.dw[j, :]
                dws += ol.dw
            epoch += 1
            mean_dw = np.mean(np.abs(dws)) / self.eta / nsamples
            fit = (epoch > self.no) and (mean_dw <= TOLER)
            print(f" * mean : {mean_dw}")
            # print(ol.w[1])

    def set_ys(self, x):
        self.layers[0].y = self.layers[0].get_y(x)
        self.layers[1].y = self.layers[1].get_y(self.layers[0].y)

