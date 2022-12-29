import numpy as np

max_layers = 2

def check_nlayers(n):
    if n == 1 or n > 2:
        msg = f"Number of layers given : {n}. Max number of layers supported : {max_layers}"
        raise Exception(msg)


def check_input_output_nodes(ni, no):
    if no >= ni:
        msg = f"Number of output nodes should be less than {ni}. `{no}` was given."
        raise Exception(msg)


def x_to_xprime(w, x, y, idx):
    # print("\n\n")
    d = np.zeros(shape=(np.shape(x)))
    for i in range(idx):
        # print(d)
        d += w[i, :] * y[i]
    return x - d