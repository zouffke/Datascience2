from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from functions.functions_Y2.evaluationMetrics import *


def lda_info(lda: LinearDiscriminantAnalysis, X, do_print: bool = False):
    """
    Return or print the info of the given LDA.
    :param lda: The linear discriminant analysis object.
    :param X: The list of independent objects of the lda.
    :param do_print: Print the output of the function instead of returning the output

    :returns dfs1: The prior probabilities of groups.
    :returns dfs2: The group means
    :returns dfs3: The coefficients of linear discriminants
    :returns dimensions: The dimensions of the lda
    """
    df1 = pd.DataFrame(lda.priors_, index=lda.classes_, columns=['prior probabilities'])
    df2 = pd.DataFrame(lda.means_, index=lda.classes_, columns=X.columns)
    df3 = pd.DataFrame(lda.scalings_, index=X.columns,
                       columns=['LD' + str(i + 1) for i in range(lda.scalings_.shape[1])])
    dfs1 = df1.style.set_caption('Prior probabilities of groups')
    dfs2 = df2.style.set_caption('Group means')
    dfs3 = df3.style.set_caption('Coefficients of linear discriminants')
    dimensions = min(X.columns.size, lda.classes_.size - 1)
    if do_print:
        display(dfs1)
        display(dfs2)
        display(dfs3)
        print(f'The LD has {dimensions} dimension(s)')
    return dfs1, dfs2, dfs3, dimensions


def ld1(lda: LinearDiscriminantAnalysis, X, target: pd.DataFrame, index: range):
    LD = lda.transform(X)
    arr_t = np.transpose(LD)
    LD = pd.DataFrame(zip(*arr_t, target), columns=['LD' + str(i + 1) for i in range(LD.shape[1])] + ['Target'],
                      index=index)
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


def pred_da(lda: LinearDiscriminantAnalysis, observation: pd.DataFrame):
    return lda.predict_proba(observation), lda.predict(observation)


def evaluate_da(overview: pd.DataFrame, targetName: str):
    cm = pd.crosstab(overview.prediction, overview[targetName], margins='all', margins_name='total')
    classes = cm.index.tolist()
    acc = pd.DataFrame([accuracy(cm)],
                       index=["Total"], columns=["Accuracy"])
    pres = pd.DataFrame([cm[classes[0]][classes[0]] / cm[classes[2]][classes[0]],
                         cm[classes[1]][classes[1]] / cm[classes[2]][classes[1]]],
                        index=[classes[0], classes[1]],
                        columns=["Precision"])
    recall = pd.DataFrame([cm[classes[0]][classes[0]] / cm[classes[0]][classes[2]],
                           cm[classes[1]][classes[1]] / cm[classes[1]][classes[2]]],
                          index=[classes[0], classes[1]],
                          columns=["Recall"])
    f1 = pd.DataFrame([2 * (pres.iloc[0, 0] * recall.iloc[0, 0]) / (pres.iloc[0, 0] + recall.iloc[0, 0]),
                       2 * (pres.iloc[1, 0] * recall.iloc[1, 0]) / (pres.iloc[1, 0] + recall.iloc[1, 0])],
                      index=[classes[0], classes[1]],
                      columns=["F1"])
    return cm, acc, pres, recall, f1
