import pandas as pd

from model import *
from utils.preprocessing import *

eta = 1e-6
ncomps = 10

if __name__ == '__main__':
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    df = df.head(1000)
    nrows, ncols = np.shape(df)
    # print(df)

    df, X = get_features(df, 'smoking', ['gender', 'oral', 'tartar'])
    X = np.asarray(df.drop(columns=['smoking']))

    hebb = Model([ncols-1, ncomps], eta)
    hebb.train(X)