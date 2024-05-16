import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


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


def f1_score(confusion_matrix: pd.DataFrame, target: str, β: float = 1):
    pr = precision(confusion_matrix, target)
    rc = recall(confusion_matrix, target)
    return ((pow(β, 2) + 1) * pr * rc) / ((pow(β, 2) * pr) + rc)


def evaluate(confusion_matrix: pd.DataFrame, extended_output: bool = False):
    classes = confusion_matrix.index.tolist()
    classes.pop()
    acc = accuracy(confusion_matrix)
    precisions = pd.DataFrame([precision(confusion_matrix, cls) for cls in classes], columns=["Class Precision"],
                              index=classes).T
    recalls = pd.DataFrame([recall(confusion_matrix, cls) for cls in classes], columns=["Class Recall"], index=classes)

    f1 = pd.DataFrame([f1_score(confusion_matrix, cls) for cls in classes], columns=["F1-Score"], index=classes)

    combined = pd.concat([confusion_matrix, precisions]).join(recalls).join(f1)
    if extended_output:
        return acc, precisions, recalls, f1, combined
    else:
        return acc, combined


def plot_roc(y_true, y_score, title='ROC Curve', **kwargs):
    if 'pos_label' in kwargs:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=kwargs.get('pos_label'))
        auc = roc_auc_score(y_true, y_score)
    else:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        auc = roc_auc_score(y_true, y_score)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    figsize = kwargs.get('figsize', (7, 7))
    fix, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(linestyle='--')

    ax.plot(fpr, tpr, color='darkorange', label='AUC: {}'.format(auc))
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (FPR)')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange', edgecolor='black')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               label='optimat cutoff {:.2f} op ({:.2f},{:.2f})'.format(optimal_threshold, fpr[optimal_idx],
                                                                       tpr[optimal_idx]), color='red')
    ax.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]], linestyle='--', color='red')
    ax.plot([0, fpr[optimal_idx]], [tpr[optimal_idx], tpr[optimal_idx]], linestyle='--', color='red')
    ax.legend(loc='lower right')
    plt.show()
