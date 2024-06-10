import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display_functions import display
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


def evaluate(confusion_matrix: pd.DataFrame, β: float = 1, extended_output: bool = False, do_print: bool = False):
    classes = confusion_matrix.index.tolist()
    classes.pop()
    acc = accuracy(confusion_matrix)
    precisions = pd.DataFrame([precision(confusion_matrix, cls) for cls in classes], columns=["Class Precision"],
                              index=classes).T
    recalls = pd.DataFrame([recall(confusion_matrix, cls) for cls in classes], columns=["Class Recall"], index=classes)

    f1 = pd.DataFrame([f1_score(confusion_matrix, cls, β=β) for cls in classes], columns=[f"F{β}-Score"], index=classes)

    combined = pd.concat([confusion_matrix, precisions]).join(recalls).join(f1)
    if extended_output:
        return acc, precisions, recalls, f1, combined
    if do_print:
        print(f'The accuracy is: {acc}')
        display(combined)
        return
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


def pos_neg(cm: pd.DataFrame, total_included: bool = True, target: str = None, do_print: bool = False,
            total_name: str = "Total"):
    if target is None:
        expected_shape = '(3, 3)' if total_included else '(2, 2)'
        if cm.shape != expected_shape:
            total_error_text = 'included' if total_included else 'not included'
            raise Exception(
                f'The confusion matrix should have a shape of {expected_shape} when target is not specified and the '
                f'Total is {total_error_text}.')
        tp = cm.iloc[0, 0]
        tn = cm.iloc[1, 1]
        fp = cm.iloc[1, 0]
        fn = cm.iloc[0, 1]
    else:
        tp = cm.loc[target, target]
        fp = cm.loc[cm.index.difference([target, total_name]), target].sum()
        fn = cm.loc[target, cm.index.difference([target, total_name])].sum()
        tn = None

    if do_print:
        print(f'The true positive rate is: {tp}')
        print(f'The false positive rate is: {fp}')
        print(f'The false negative rate is: {fn}')
        print(f'The true negative rate is: {tn}')
        return
    else:
        return tp, fp, fn, tn
