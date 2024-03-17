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
