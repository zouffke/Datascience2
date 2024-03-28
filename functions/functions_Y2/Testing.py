import math
from scipy.stats import norm
from scipy import stats


def betrouwbaarheids_interval(data, b: float = 0.95, t: bool = True):
    """
    :param data: The data that is used to calculate the confidence interval
    :param b: the confidence level
    :param t: wetter to use the t-distribution or the normal distribution
    :return: the lower and upper bound of the confidence interval
    """
    if not 0 < b < 1:
        raise ValueError("b must be between 0 and 1")

    n = len(data)
    if t:
        f = stats.t.ppf((1 + b) / 2, n - 1)
    else:
        f = norm.ppf((1 + b) / 2)
    M = data.mean()
    s = data.std()
    l = M - f * s / math.sqrt(n)
    r = M + f * s / math.sqrt(n)
    return l, r
