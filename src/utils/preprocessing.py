import numpy as np

from sklearn.preprocessing import LabelEncoder, normalize
from typing import List

def get_features(
    df,
    class_col: str,
    encode: List[str] = None,
    drop: List[str] = None,
):
    if encode is not None:
        for e in encode:
            df[e] = LabelEncoder().fit_transform(df[e])
    if drop is not None:
        df = df.drop(columns=drop)
    x = np.asarray(df.drop(columns=[class_col]))
    x = normalize(x)
    return df, x