import pandas as pd


def accuracy(confusion_matrix: pd.DataFrame):
    classes = len(confusion_matrix.index)
    sumT = 0
    for i in range(classes - 1):
        sumT += confusion_matrix.iloc[i, i]
    return sumT / confusion_matrix.iloc[classes - 1, classes - 1]
