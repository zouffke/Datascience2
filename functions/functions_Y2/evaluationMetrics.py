import pandas as pd


def accuracy(confusion_matrix: pd.DataFrame):
    classes = len(confusion_matrix.index)
    sumT = 0
    for i in range(classes - 1):
        sumT += confusion_matrix.iloc[i, i]
    return sumT / confusion_matrix.iloc[classes - 1, classes - 1]


def precision(confusion_matrix: pd.DataFrame, target: str):
    classes = confusion_matrix.index
    return confusion_matrix.loc[target, target] / confusion_matrix.loc[classes[len(classes) - 1], target]


def recall(confusion_matrix: pd.DataFrame, target: str):
    classes = confusion_matrix.index
    return confusion_matrix.loc[target, target] / confusion_matrix.loc[target, classes[len(classes) - 1]]


def evaluatue(confusion_matrix: pd.DataFrame):
    classes = confusion_matrix.index.tolist()
    classes.pop()
    precisions = pd.DataFrame([precision(confusion_matrix, cls) for cls in classes], columns=["Class Precision"],
                              index=classes).T
    recalls = pd.DataFrame([recall(confusion_matrix, cls) for cls in classes], columns=["Class Recall"], index=classes)
    return pd.concat([confusion_matrix, precisions]).join(recalls)
