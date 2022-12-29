from utils.hebb import *

class Layer:
    t:str              # type {input `i`, output `o`}
    n: int             # number of nodes on layer
    ni: int            # number of inputs (1 if input)
    no: int            # number of outputs (1 if output)
    _xprime:np.ndarray # altered x values used for training
    _w: np.ndarray     # weight vector per node
    _dw: np.ndarray    # weight update vector (of same dimensions)
    _y: np.ndarray     # output


    def __init__(
        self,
        t, n, ni, no,
        mu=0, stddev=1 # mean and std. dev. for weight distribution
    ):
        self.t = t
        self.n = n
        self.ni = ni
        self.no = no
        self._y = 0
        if t == 'o':
            self._w = np.zeros(dtype=np.float32, shape=(n, ni))
            self._dw = np.zeros(dtype=np.float32, shape=(n, ni))
            self._xprime = np.zeros(dtype=np.float32, shape=(n, ni))
            for i in range(n):
                rng = np.random.default_rng()
                self._w[i] = rng.normal(loc=mu, scale=stddev, size=ni)


    def get_y(self, x):
        if self.t == 'i':
            return x
        return np.dot(self._w, x.T)


    def set_xprime(self, x):
        for i in range(self.n):
            self._xprime[i] = x_to_xprime(self._w, x, self._y, i)


    @property
    def w(self):
        return self._w
    @w.setter
    def w(self, w):
        self._w = w

    @property
    def dw(self):
        return self._dw
    @dw.setter
    def dw(self, dw):
        self._dw = dw

    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, y):
        self._y = y

    @property
    def xprime(self):
        return self._xprime
    @xprime.setter
    def xprime(self, xprime):
        self._xprime = xprime