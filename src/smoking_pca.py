import time
import pandas as pd
from sklearn.decomposition import PCA
from utils.preprocessing import *
from utils.parsers import PCAParser
from utils.loggers import log_pca

if __name__ == '__main__':
    filename = "smoking/smoking.csv"
    df = pd.read_csv(filename, delimiter=',', header=0)
    nrows, ncols = np.shape(df)

    df, X = get_features(df, 'smoking', ['gender', 'oral', 'tartar'], ['ID'])
    parser = PCAParser()
    ncomps = parser.parse_args().nc

    print(f"\n  >> Fitting ( components : {ncomps} ) ...")
    start_time = time.time()
    model = PCA(n_components=ncomps)
    model.fit(X)
    ftime = time.time() - start_time
    print(f"  >> Fitting completed ( {ftime:.3f} secs. ) \n")

    var_ratios = model.explained_variance_ratio_
    total_ratio = 100 * np.sum(var_ratios)
    print(f" ** Explained variance ratios :\n  {var_ratios}")
    print(f" ** Total percentage of variance explained: {total_ratio:.3f}\n")
    log_pca("results/smoking_pca.csv", ncomps, ftime, total_ratio)