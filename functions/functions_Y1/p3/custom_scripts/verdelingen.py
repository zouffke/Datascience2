import pandas as pd


def verdeling(series: pd.Series):
    me = series.median()
    mo = series.mode()
    m = series.mean()

    if m < me < mo:
        return 'Links scheef'
    elif m == me == mo:
        return 'Symetrisch'
    else:
        return 'Rechts scheef'
