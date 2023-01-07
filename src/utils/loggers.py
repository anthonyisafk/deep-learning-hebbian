import logging

def log_hebbian(filename, nc, eta, epochs, time, v):
    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.INFO,
        format='%(message)s',
    )
    msg = f"{nc},{eta},{epochs},{time:.3f},{v:.3f}"
    logging.info(msg)


def log_pca(filename, nc, time, v):
    logging.basicConfig(
        filename=filename,
        filemode='a',
        level=logging.INFO,
        format='%(message)s',
    )
    msg = f"{nc},{time:.3f},{v:.3f}"
    logging.info(msg)