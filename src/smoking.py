"""
@brief: PCA implemented with Hebbian Learning.
        Testing on the `body signals of smoking` dataset:
@instructions: You can change the number of components
               and other parameters in the `constants.py` file.
               Even better, try the command line options:
                 - `-eta` learning rate
                 - `-nc` number of components
****************************************
https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking
****************************************
@author: Antoniou, Antonios - 9482
@email: aantonii@ece.auth.gr
****************************************
School of Informatics
2023 Aristotle University Thessaloniki
"""

import pandas as pd
from model import *
from utils.preprocessing import *
from utils.parsers import HebbianParser
from utils.loggers import log_hebbian


if __name__ == '__main__':
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    nrows, ncols = np.shape(df)

    parser = HebbianParser()
    args = parser.parse_args()
    eta = args.eta
    ncomps = args.nc

    df, X = get_features(df, 'smoking', ['gender', 'oral', 'tartar'], ['ID'])
    hebb = Model([ncols-2, ncomps], eta)
    hebb.fit(X)

    exp_variance = get_explained_variance(hebb.comps, X)
    print(f" >>> Explained variance ratio : {exp_variance:.3f}")
    log_hebbian("results/smoking_hebbian.csv", ncomps, eta, hebb.epochs, hebb.ttime, exp_variance * 100)