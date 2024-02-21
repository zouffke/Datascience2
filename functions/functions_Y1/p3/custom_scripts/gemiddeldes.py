import pandas as pd
from scipy import stats


def gewogenGemiddelde(series: pd.Series, gewicht: pd.Series):
    return sum(gewicht * series) / sum(gewicht)


def meetkundigGemiddelde(series: pd.Series):
    return stats.gmean(series)


def harmonischGemiddelde(series: pd.Series):
    return stats.hmean(series)
