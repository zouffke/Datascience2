import numpy as np


def laplace(G: float | int, U: float | int):
    return G / U


def sum(chances: np.array):
    return np.sum(chances)


def productIndependent(chances: np.array):
    return np.prod(chances)


def productDependent(A: float | int, commonChance: float | int):
    return A * commonChance
