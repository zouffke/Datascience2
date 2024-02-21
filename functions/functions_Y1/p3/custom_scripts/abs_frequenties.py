import pandas as pd
import math


def aantalKlassen(series: pd.Series):
    return math.ceil(math.log2(len(series)) + 1)


def klassenRange(series: pd.Series):
    return range(math.floor(series.min()), math.ceil(series.max()), round(aantalKlassen(series)))
