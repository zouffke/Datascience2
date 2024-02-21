import math
import pandas as pd


def median(data: pd.Series):
    """
    Berekent de mediaan van gegevens op ordinale, interval of ratioschaal.
    :param data: inputgegevens als Pandas Series
    :return: de mediaan
    """
    d = pd.Series(data.dropna())
    n = len(d)
    middle = math.floor(n / 2)
    return d.sort_values().reset_index(drop=True)[middle]


def uitschieters(data: pd.Series, mode='normaal', output='index'):
    """
    In mode 'normaal' en output 'index' wordt van elk element van de input aangegeven of het uitschieters is (True)
    of niet (False). Deze gegevens kunnen gebruikt worden om te indexeren. In mode 'extreem' wordt er gewerkt met extreme uitschieters.
    Met de output gelijk aan 'grenzen' kunnen de grenzen van de uitschieters (normaal of extreem) bepaald worden.
    :param data: inputgegevens als Pandas Series
    :param mode: 'normaal' of 'extreem'
    :param output: 'index' of 'grenzen'
    :return: Pandas Series of grenzen
    """

    # bereken eerst Q1, Q3 and IQR
    Q1, Q3 = data.quantile([0.25, 0.75])
    IQR = Q3 - Q1

    # kan ook in één regel met zgn. walrus operator (:=), maar dit is niet per se nodig
    # IQR = (Q3 := data.quantile(0.75)) - (Q1 := data.quantile(0.25))

    # bereken de grenzen voor de uitschieters
    grenzen = Q1 - 3 * IQR, Q1 - 1.5 * IQR, Q3 + 1.5 * IQR, Q3 + 3 * IQR

    if output == 'grenzen' and mode == 'normaal':
        return grenzen[1], grenzen[2]

    if output == 'grenzen' and mode == 'extreem':
        return grenzen[0], grenzen[3]

    if mode == 'extreem':
        return ~data.between(grenzen[0], grenzen[3])

    return ~data.between(grenzen[1], grenzen[2])


def signif(x, digits=6):
    """
    Zet een getal om naar een aantal significante cijfers
    :param x: het getal
    :param digits: het aantal significante cijfers
    :return: het getal met digits significante cijfers
    """
    if x == 0 or not math.isfinite(x):
        return x
    digits -= math.ceil(math.log10(abs(x)))
    return round(x, digits)


def mean_median_mode(series: pd.Series):
    """
     Geeft gemiddelde, modus en mediaan van een pandas.series object input is een rij van float  of int
    :param series:
    """
    print(f"Het gemiddelde is:{series.mean()}")
    print(f"De mediaan is:{series.median()}")
    print(f"De modus is:{series.value_counts().idxmax()}")
