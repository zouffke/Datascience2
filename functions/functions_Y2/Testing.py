import math
from scipy.stats import norm


def betrouwbaarheids_interval(data, b):
    """
    :param data: The data that is used to calculate the confidence interval
    :param b: the confidence level
    :return: the lower and upper bound of the confidence interval
    """
    n = len(data)
    f = norm.ppf((1 + b) / 2)
    M = data.mean()
    s = data.std()
    l = M - f * s / math.sqrt(n)
    r = M + f * s / math.sqrt(n)
    return l, r
