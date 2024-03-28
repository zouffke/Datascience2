import math
from scipy.stats import norm
from scipy import stats


def betrouwbaarheids_interval(data, b: float = 0.95, t: bool = True, do_print: bool = False):
    """
    :param data: The data that is used to calculate the confidence interval
    :param b: the confidence level
    :param t: wetter to use the t-distribution or the normal distribution
    :param do_print: print the confidence interval
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

    if do_print:
        print(f"{round(l, 4)} < x̄ < {round(r, 4)}")
        print(f"μ = {round(M, 4)}")
        print(f"σ = {round(s, 4)}")

    return l, r


def aanvaardings_interval(data, μ, b: float = 0.95, t: bool = True, do_print: bool = False):
    """
    :param data: The data that is used to calculate the confidence interval
    :param μ: the hypothesized mean
    :param b: the confidence level
    :param t: wetter to use the t-distribution or the normal distribution
    :param do_print: print the confidence interval
    :return: the lower and upper bound of the confidence interval
    """
    if not 0 < b < 1:
        raise ValueError("b must be between 0 and 1")
    n = len(data)
    s = data.std()
    if t:
        f = stats.t.ppf((1 + b) / 2, n - 1)
    else:
        f = norm.ppf((1 + b) / 2)
    l = μ - f * s / math.sqrt(n)
    r = μ + f * s / math.sqrt(n)

    if do_print:
        print(f"{round(l, 4)} < x̄ < {round(r, 4)}")
        print(f"μ = {round(data.mean(), 4)}")
        print(f"σ = {round(s, 4)}")

    return l, r


def p_value_test(data, μ, b: float = 0.95, do_print: bool = False):
    """
    :param data: The data that is used to calculate the p-value
    :param b: the confidence level
    :param μ: the hypothesized mean
    :param do_print: print the p-value
    :return: the p-value and the t-value
    """
    if not 0 < b <= 1:
        raise ValueError("b must be between 0 and 1")

    n = len(data)
    s = data.std()
    t = (data.mean() - μ) / (s / math.sqrt(n))
    p = stats.t.cdf(-abs(t), df=n - 1) * 2
    a = 1 - b

    if do_print:
        print(f"p = {round(p, 4)}")
        print(f"σ = {round(s, 4)}")
        print(f"t = {round(t, 4)}")
        print(f"α = {round(a, 4)}")
        if t < a:
            print("H0 is rejected")
        else:
            print("H0 is not rejected")

    return p, t


def z_test(data, μ, σ):
    """
    :param data: The data that is used to calculate the p-value
    :param μ: The hypothesized mean
    :param σ: The hypothesized standard deviation
    :return: the p-value and the z-value
    """
    n = len(data)
    z = (data.mean() - μ) / σ * math.sqrt(n)
    p = stats.norm.cdf(-abs(z)) * 2
    return p, z


def one_sided_a(data, limit, inf: int):
    if inf not in [1, -1]:
        raise ValueError("inf must be 1 or -1")

    n = len(data)
    s = data.std()

    if inf == 1:
        a = stats.t.cdf((data.mean() - limit) / s * math.sqrt(n), n - 1)
    else:
        a = stats.t.cdf((limit - data.mean()) / s * math.sqrt(n), n - 1)

    return a


def chi_squared_test(fo, fe):
    """
    :param fo: The observed frequency
    :param fe: The expected frequency
    :return: The chi-squared value
    """
    x = sum((fo - fe) ** 2 / fe)
    return x
