"""
@brief: PCA implemented with Hebbian Learning.
        Testing on the `body signals of smoking` dataset:
@instructions: You can change the number of components
               and other parameters in the `constants.py` file.
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
from constants import eta, ncomps

if __name__ == '__main__':
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    nrows, ncols = np.shape(df)

    df, X = get_features(df, 'smoking', ['gender', 'oral', 'tartar'], ['ID'])
    org_var = np.sum(np.diag(np.cov(X, rowvar=False)))
    hebb = Model([ncols-2, ncomps], eta)
    hebb.fit(X)

    C = np.cov(hebb.comps, rowvar=False)
    print(C, end="\n\n")
    C = C / org_var
    print(np.diag(C))