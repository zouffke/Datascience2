from id3.export import DotTree
import graphviz
from id3 import Id3Estimator, export_graphviz
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


def entropy(column: pd.Series, base=None):
    """
    Bereken de entropie van een kolom
    :param column: de kolom waarvan de entropie berekend moet worden
    :param base: de basis van de logaritme
    :return: de entropie van de kolom
    """
    # bepaal de fracties voor alle kolomwaarden v
    fracties = column.value_counts(normalize=True, sort=False)
    base = 2 if base is None else base
    return -(fracties * np.log(fracties) / np.log(base)).sum()


def information_gain(df: pd.DataFrame, s: str, target: str):
    """
    Bereken de information gain van een kolom
    :param df: de dataframe waarvan de information gain berekend moet worden
    :param s: de kolom waarvan de information gain berekend moet worden
    :param target: de kolom tot waar de information gain berekend moet worden
    :return: de information gain van de kolom
    """
    # bereken entropie van ouder tabel
    entropy_ouder = entropy(df[target])
    child_entropies = []
    child_weights = []
    # bereken entropieën van kindtabellen
    for (label, p) in df[s].value_counts().items():
        child_df = df[df[s] == label]
        child_entropies.append(entropy(child_df[target]))
        child_weights.append(int(p))
    # bereken het verschil tussen ouder-entropie en gewogen kind-entropieën
    return entropy_ouder - np.average(child_entropies, weights=child_weights)


def best_split(df: pd.DataFrame, target: str):
    """
    Bepaal de kolom met de hoogste information gain
    :param df: De parent tabel
    :param target: De target kolom
    :return: De kolom met de hoogste information gain en de information gain
    """
    # haal alle niet-target kolomlabels op (de features)
    features = df.drop(axis=1, labels=target).columns
    # bereken de information gains voor deze features
    gains = [information_gain(df, feature, target) for feature in features]
    # return kolom met hoogste information en de information gain
    return features[np.argmax(gains)], max(gains)


def trainID3(df: pd.DataFrame, target: str, extraRemove=None):
    """
    Train een ID3 model en geef de boom en het model terug.
    Maak gebruik van graphviz.Source(model_tree.dot_tree) om de boom te visualiseren.
    :param df: het dataframe waarvan het model getraind moet worden
    :param target: de target kolom, wat het model moet voorspellen
    :param extraRemove: extra kolommen die verwijderd moeten worden. Verwacht een array van Strings
    :return: het model en de boom
    """
    if extraRemove is None:
        extraRemove = [target]
    else:
        extraRemove = [target] + extraRemove

    model = Id3Estimator()
    # x = attributen, y = target
    x = df.drop(columns=extraRemove, axis=1).to_numpy().tolist()
    y = df[target].to_numpy().tolist()

    model.fit(x, y)

    model_tree = DotTree()
    export_graphviz(model.tree_, model_tree, feature_names=df.drop(extraRemove, axis=1).columns)
    return model, model_tree


def trainDecisionTree(df: pd.DataFrame, target: str, extraRemove=None):
    """
    Train een DecisionTree model en geef het model terug. De boom wordt gevisualiseerd.
    :param df: dataframe waarvan het model getraind moet worden
    :param target: de target kolom, wat het model moet voorspellen
    :param extraRemove: extra kolommen die verwijderd moeten worden. Verwacht een array van Strings
    :return: het model
    """
    if extraRemove is None:
        extraRemove = [target]
    else:
        extraRemove = [target] + extraRemove

    model = DecisionTreeClassifier(criterion='entropy')
    # x = attributen, y = target
    x = df.drop(columns=extraRemove)
    y = df[target]
    try:
        model.fit(x, y)
    except ValueError:
        x = pd.get_dummies(df.drop(columns=extraRemove))
        model.fit(x, y)

    _ = tree.plot_tree(model, feature_names=x.columns, class_names=np.unique(y), filled=True, fontsize=10,
                       rounded=True)
    return model