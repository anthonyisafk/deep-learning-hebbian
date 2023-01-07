import argparse

def HebbianParser():
    p = argparse.ArgumentParser()
    p.add_argument('-eta', help='learning rate', type=float, default=1e-3)
    p.add_argument('-nc', help='number of components', type=int, default=1)
    return p


def PCAParser():
    p = argparse.ArgumentParser()
    p.add_argument('-nc', help='number of components', type=int, default=1)
    return p