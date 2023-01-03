from sklearn.decomposition import PCA
from utils.preprocessing import *
from constants import ncomps
import time
import pandas as pd

if __name__ == '__main__':
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    nrows, ncols = np.shape(df)

    df, X = get_features(df, 'smoking', ['gender', 'oral', 'tartar'], ['ID'])

    print("\n  >> Fitting ...")
    start_time = time.time()
    model = PCA(n_components=ncomps)
    model.fit(X)
    ftime = time.time() - start_time
    print(f"  >> Fitting completed ( {ftime:.3f} secs. ) \n")

    var_ratios = model.explained_variance_ratio_
    total_ratio = 100 * np.sum(var_ratios)
    print(f" ** Explained variance ratios :\n  {var_ratios}")
    print(f" ** Total percentage of variance explained: {total_ratio:.3f}\n")