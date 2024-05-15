from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def lda_info(lda: LinearDiscriminantAnalysis, X):
    df1 = pd.DataFrame(lda.priors_, index=lda.classes_, columns=['prior probabilities'])
    df2 = pd.DataFrame(lda.means_, index=lda.classes_, columns=X.columns)
    df3 = pd.DataFrame(lda.scalings_, index=X.columns, columns=['LD1'])
    dfs1 = df1.style.set_caption('Prior probabilities of groups')
    dfs2 = df2.style.set_caption('Group means')
    dfs3 = df3.style.set_caption('Coefficients of linear discriminants')
    return dfs1, dfs2, dfs3


def ld1(lda: LinearDiscriminantAnalysis, X, target: pd.DataFrame, index: range):
    LD = lda.transform(X)
    LD = pd.DataFrame(zip(LD[:, 0], target), columns=['LD1', 'Target'], index=index)
    return LD


def vis_da(LD: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['green', 'red']
    x = np.linspace(-3, 6, 100)
    for i, target_value in enumerate(LD['Target'].unique()):
        mean = LD['LD1'][LD['Target'] == target_value].mean()
        std = LD['LD1'][LD['Target'] == target_value].std()
        verdeling = norm(loc=mean, scale=std)
        LD['LD1'][LD['Target'] == target_value].hist(ax=ax,
                                                     bins=25, density=True,
                                                     edgecolor='black', color=colors[i], alpha=0.5,
                                                     label=target_value)
        ax.plot(x, verdeling.pdf(x), color=colors[i], linewidth=3)
        ax.legend()
        ax.grid(axis='y')
        ax.set_xlim((-3, 6))
        ax.set_ylim((0, 1.5))
        ax.set_xlabel('LD1')
    return fig, ax
