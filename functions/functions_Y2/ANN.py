import pandas as pd


def min_max_norm(col):
    minimum = col.min()
    f_range = col.max() - minimum
    return (col - minimum) / f_range


def decimal_scaling_norm(col):
    max_value = col.max()
    tenfold = 1
    while max_value > tenfold:
        tenfold *= 10
    return col / tenfold


def z_score_norm(col):
    mean = col.mean()
    std = col.std()
    return (col - mean) / std


def normalized_values(df, norm_funct):
    df_norm = pd.DataFrame()
    for column in df:
        df_norm[column] = norm_funct(df[column])
    return df_norm
