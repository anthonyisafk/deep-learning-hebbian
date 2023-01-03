from typing import List
import time
from layer import *
from constants import MAX_EPOCHS, TOLERANCE_PER_OUT_NODE

class Model:
    nlayers: int         # number of layers
    layers:np.ndarray    # Layer class instances
    ni: int              # number of input nodes on first layer
    no: int              # number of output nodes
    eta: np.float32      # learning rate
    tolerance:np.float32 # minimum change of weights per epoch, that's considered `stable`
    _comps:np.ndarray    # result of fitting

    def __init__(self, n_nodes:List[int], eta):
        self.nlayers = len(n_nodes)
        check_nlayers(self.nlayers)
        check_input_output_nodes(n_nodes[0], n_nodes[1])
        self.ni = n_nodes[0]
        self.no = n_nodes[1]
        self.eta = eta
        self.tolerance = self.no * TOLERANCE_PER_OUT_NODE
        self.layers = np.ndarray(dtype=Layer, shape=(2))
        self.layers[0] = Layer('i', n_nodes[0], 1, n_nodes[1])
        self.layers[1] = Layer('o', n_nodes[1], n_nodes[0], 1, mu=0, stddev=1e-10)


    def fit(self, xs):
        print(f"\n  >> Fitting (Tolerance : {self.tolerance}) ...\n")
        start_time = time.time()
        nsamples = len(xs)
        ol = self.layers[1]
        epoch = 0
        fit = False
        self._comps = np.zeros(shape=(nsamples, self.no))
        while epoch < MAX_EPOCHS and not fit:
            print(f"** Epoch : {epoch} ...", end=" ")
            dws = np.zeros(shape=(self.no, self.ni))
            for p in range(nsamples):
                x = xs[p]
                self.set_ys(x)
                ol.set_d()
                for j in range(self.no):
                    # ol.d[j, :] = np.dot(ol.w[:j, :].T, ol.y[:j])
                    ol.dw[j, :] = self.eta * ol.y[j] * (x - ol.d[j, :])
                    ol.w[j, :] += ol.dw[j, :]
                dws += ol.dw
            epoch += 1
            mean_dw = np.mean(np.abs(dws)) / self.eta / nsamples
            fit = (epoch > self.no) and (mean_dw <= self.tolerance)
            print(f"mean : {mean_dw}")
        ttime = time.time() - start_time
        print(f"\n  >> Fitting was completed in {epoch} epochs ( {ttime:.3f} secs. )\n")
        self.pca(xs)


    def set_ys(self, x):
        self.layers[0].y = self.layers[0].get_y(x)
        self.layers[1].y = self.layers[1].get_y(self.layers[0].y)


    def pca(self, x):
        nsamples = len(x)
        for s in range(nsamples):
            self._comps[s, :] = self.layers[1].get_y(x[s])


    @property
    def comps(self):
        return self._comps
    @comps.setter
    def comps(self, comps):
        self._comps = comps
