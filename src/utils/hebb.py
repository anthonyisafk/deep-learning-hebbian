import numpy as np
from constants import MAX_LAYERS

def check_nlayers(n):
    if n == 1 or n > MAX_LAYERS:
        msg = f"Number of layers given : {n}. Max number of layers supported : {MAX_LAYERS}"
        raise Exception(msg)


def check_input_output_nodes(ni, no):
    if no >= ni:
        msg = f"Number of output nodes should be less than {ni}. `{no}` was given."
        raise Exception(msg)


def get_explained_variance(comps, X):
    org_var = np.sum(np.diag(np.cov(X, rowvar=False)))
    C = np.cov(comps, rowvar=False)
    C = C / org_var
    if (comps.shape[1] == 1):
        return C
    Cdiag = np.diag(C)
    return np.sum(Cdiag)