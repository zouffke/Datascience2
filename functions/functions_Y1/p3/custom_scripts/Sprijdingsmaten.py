import math

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


def bereik(series: pd.Series, meetniveau='interval'):
    if meetniveau == 'interval':
        return series.max() - series.min()
    elif meetniveau == 'ordinaal':
        return series.min(), series.max()


def kwartielen(series: pd.Series, output: int = 4):
    if output == 1:
        return series.quantile(0.25)
    elif output == 2:
        return series.median()
    elif output == 3:
        return series.quantile(0.75)
    elif output == 4:
        return series.quantile(q=[0.25, 0.5, 0.75])


def IQR(series: pd.Series):
    return stats.iqr(series.dropna())


def boxplot(series: pd.Series, columnName: [str], by: str = None):
    fig, ax = plt.subplots()
    return series.boxplot(ax=ax, by=by, column=columnName)


def outliers(series: pd.Series, mode: str = 'normal', output: str = 'normal'):
    Q1 = kwartielen(series, 1)
    Q3 = kwartielen(series, 3)
    iqr = IQR(series)
    if mode == 'normal':
        weight = 1.5
    elif mode == 'extreme':
        weight = 3

    if output == 'normal':
        return ~series.between(Q1 - weight * iqr, Q3 - weight * iqr)
    elif output == 'numbers':
        return series[outliers(series, mode=mode)]


def gemAfwijking(series: pd.Series):
    return (series - series.mean()).mean()


def gemAbsAfwijking(series: pd.Series):
    return (abs(series - series.mean())).mean()


def variantie(series: pd.Series, output: str = 'var'):
    if output == 'var':
        # σ² = (1/n) * n∑i=1(xi - μ)²
        return ((series - series.mean()) ** 2).mean()
    elif output == 'std':
        # σ = ⎷((1/n) * n∑i=1(xi - μ)²)
        return math.sqrt(((series - series.mean()) ** 2).mean())


def zScore(series: pd.Series, value: int):
    return (value - series.mean()) / series.std()
