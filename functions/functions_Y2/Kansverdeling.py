import matplotlib.pyplot as plt
import math


def kansverdeling(data, xlabel: str, ylabel: str, title: str = "Kansverdeling"):
    fix, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.bar(data[0], data[1])
    ax.set_title(title)
    plt.show()


def average(value, probability, distribution: str = "default"):
    if distribution == "default":
        return (value * probability).sum()
    if distribution == "binomial":
        return value * probability
    else:
        raise Exception("Invalid distribution. Please use 'default' or 'binomial'.")


def standard_deviation(value, probability, distribution: str = "default"):
    if distribution == "default":
        return math.sqrt(((value - average(value, probability)) ** 2 * probability).sum())
    if distribution == "binomial":
        return math.sqrt(value * probability * (1 - probability))
    else:
        raise Exception("Invalid distribution. Please use 'default' or 'binomial'.")


def normaal_verdeling(x, p, μ: float, xlabel: str, ylabel: str, lower: float = None, upper: float = None):
    plt.plot(x, p)
    if lower is not None and upper is not None:
        plt.fill_between(x, p, where=(x >= lower) & (x <= upper), facecolor='red', alpha=.3)
    else:
        plt.fill_between(x, p, where=(x <= μ - 0.5), facecolor='red', alpha=.3)
        plt.fill_between(x, p, where=(x >= μ + 0.5), facecolor='green', alpha=.3)
    plt.fill_between(x, p, where=(x >= μ - 1) & (x <= μ + 1), facecolor='blue', alpha=.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title("Normaal verdeling")
    plt.show()
