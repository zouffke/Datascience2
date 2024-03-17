import math


def average(value, probability):
    return (value * probability).sum()


def standard_deviation(value, probability):
    return math.sqrt(((value - average(value, probability)) ** 2 * probability).sum())
