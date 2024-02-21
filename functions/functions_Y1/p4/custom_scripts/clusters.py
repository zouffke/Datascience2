import pandas as pd
from scipy.spatial.distance import cdist
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import cut_tree


def plotPoints(df: pd.DataFrame, x=None, y=None, names=False):
    """
    Plots the points of the dataframe df.
    :param df: dataframe with the points to plot
    :param x: name of the column with the x coordinates (opt)
    :param y: name of the column with the y coordinates (opt)
    :return: None
    """
    if x is None:
        x = df['x']
    else:
        x = df[x]
    if y is None:
        y = df['y']
    else:
        y = df[y]

    ax = df.plot(kind="scatter", x=x.name, y=y.name, figsize=(5, 5))

    _ = ax.set_xlim(x.min(),
                    x.max())
    _ = ax.set_ylim(y.min(),
                    y.max())
    _ = ax.grid(linestyle='--')
    if names:
        for i in range(0, len(df)):
            label = f'point {i + 1}'
            _ = ax.annotate(label,
                            fontsize=7,
                            xy=(x[i], y[i]),
                            xytext=(-5, 5),
                            textcoords='offset points', ha='center', va='bottom')
    return


def plotAfstanden(df: pd.DataFrame, afst: pd.DataFrame, x:str=None, y:str=None, names=False):
    try:
        if x is None:
            x = df['x']
        else:
            x = df[x]
        if y is None:
            y = df['y']
        else:
            y = df[y]
    except KeyError:
        raise Exception("If there is no column 'x' and 'y' present the x and y value needs to be passed as a "
                        "parameter as the name of the column in a String")

    afst = afst.where(np.tril(np.ones(afst.shape)).astype(bool))
    np.fill_diagonal(afst.values, np.nan)

    cm = sns.light_palette("green", as_cmap=True)

    # we plotten de afstandsmatrix als heatmap naast de punten
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
    _ = axes[0].scatter(x, y)
    _ = axes[0].set_xlim(x.min(),
                         x.max())
    _ = axes[0].set_ylim(y.min(),
                         y.max())
    _ = axes[0].set_xticks(range(int(x.min()),
                                 int(x.max())))
    _ = axes[0].set_yticks(range(int(y.max())))
    _ = axes[0].grid(linestyle='--')
    _ = axes[0].set_title('afstanden tussen de punten')
    if names:
        for i in range(0, len(df)):
            label = 'punt {}'.format(i + 1)
            _ = axes[0].annotate(label,
                                 xy=(x[i], y[i]),
                                 xytext=(-5, 5),
                                 textcoords='offset points', ha='center', va='bottom')
    _ = sns.heatmap(afst, annot=True, cmap="Blues", ax=axes[1])
    _ = axes[1].set_yticklabels(labels=df.index, rotation=0)
    _ = axes[1].set_title('Afstanden tussen de punten')
    return


def kMeans(df: pd.DataFrame, nClusters: int, x=None, y=None, maxIter=100):
    model = KMeans(n_clusters=nClusters, n_init='auto', max_iter=maxIter)
    model.fit(df)

    try:
        if x is None:
            x = df['x']
        else:
            x = df[x]
        if y is None:
            y = df['y']
        else:
            y = df[y]
    except KeyError:
        raise Exception("If there is no column 'x' and 'y' present the x and y value needs to be passed as a "
                        "parameter as the name of the column in a String")

    model_df = pd.DataFrame(zip(x, y, model.labels_), columns=["x", "y", "cluster"])

    cm = sns.color_palette("viridis", as_cmap=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    _ = ax.scatter(x, y, c=model.labels_, cmap=cm, s=10)
    _ = ax.scatter(model.cluster_centers_[:, 0],
                   model.cluster_centers_[:, 1], c='red', s=50)
    _ = ax.set_xlabel("x")
    _ = ax.set_ylabel("y")
    _ = ax.grid(linestyle='--')
    return model_df


def agglo(df: pd.DataFrame, nClusters: int, x=None, y=None):
    model = AgglomerativeClustering(n_clusters=nClusters, metric='cosine', linkage='complete')
    model.fit(df)

    try:
        if x is None:
            x = df['x']
        else:
            x = df[x]
        if y is None:
            y = df['y']
        else:
            y = df[y]
    except KeyError:
        raise Exception("If there is no column 'x' and 'y' present the x and y value needs to be passed as a "
                        "parameter as the name of the column in a String")

    cm = sns.color_palette("viridis", as_cmap=True)

    # model.labels_
    fig, ax = plt.subplots(figsize=(5, 5))
    _ = ax.scatter(x, y, c=model.labels_, cmap=cm, s=10)
    _ = ax.set_xlabel("x")
    _ = ax.set_ylabel("y")
    _ = ax.grid(linestyle='--')
    return


def dendogram(df: pd.DataFrame):
    colors = list(matplotlib.colors.cnames.keys())[0:100:2]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    distances_single = linkage(df, method='single')
    distances_complete = linkage(df, method='complete')
    _ = ax[0].set_title('Dendrogram met single linkage')
    _ = ax[0].set_xlabel('punt')
    _ = ax[0].set_ylabel('Euclidische afstand')
    _ = ax[0].grid(linestyle='--', axis='y')

    dgram = dendrogram(distances_single,
                       labels=list(range(1, 11)),
                       link_color_func=lambda x: colors[x],
                       leaf_font_size=15.,
                       ax=ax[0])

    _ = ax[1].set_title('Dendrogram met complete linkage')
    _ = ax[1].set_xlabel('punt')
    _ = ax[1].set_ylabel('Euclidische afstand')
    _ = ax[1].grid(linestyle='--', axis='y')

    dgram = dendrogram(distances_complete,
                       labels=list(range(1, 11)),
                       link_color_func=lambda x: colors[x],
                       leaf_font_size=15.,
                       ax=ax[1])
    return


def cuttree(df: pd.DataFrame):
    distances_single = linkage(df, method='single')
    cuttree = cut_tree(distances_single, 6)
    return pd.DataFrame(cuttree).sort_values(by=0)
