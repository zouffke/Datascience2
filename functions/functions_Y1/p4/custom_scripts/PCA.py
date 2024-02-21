import pandas as pd
from pca import pca
from matplotlib import pyplot as plt
import warnings


def createPCA(df: pd.DataFrame, n_components: int, value_to_drop: [], output: str = None, transform=False):
    x = df.dropna().drop(columns=value_to_drop)
    for i in range(len(value_to_drop)):
        y = df.dropna()[value_to_drop[i]]

    model = pca(normalize=True, n_components=n_components)
    out = model.fit_transform(x, verbose=False)

    if transform:
        if output is not None:
            warnings.warn('The output option will be ignored when the transform option is True')
        return model.transform(x)

    if output is None:
        return model
    else:
        output = str.lower(output)
        if output == 'loadings':
            return pd.DataFrame(out['loadings'])  # matrix met de coefficiënten van de PCA analyse
        elif output == 'var':
            return pd.DataFrame(zip(out['variance_ratio'], out['explained_var']),
                         index=[f'PC{i + 1}' for i in range(len(out['variance_ratio']))],
                         columns=['variance_ratio', 'explained_var']).T
        elif output == 'topfeat':
            return out['topfeat']
        elif output == 'plot':
            model.plot(figsize=(7, 5))
        elif output == 'biplot':
            model.biplot(cmap='viridis',
                         labels=y,
                         density=True,
                         n_feat=2,
                         s=20,
                         fontsize=10,
                         figsize=(10, 7))
        elif output == 'project':
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 12))
            _ = ax.view_init(azim=30, elev=10)
            _ = ax.set_xlim(-2, 10)
            _ = ax.set_ylim(-2, 5)
            _ = ax.set_zlim(-3, 2)
            _ = ax.set_box_aspect((1, 1, 1), zoom=1)
            _ = model.biplot3d(ax=ax, legend=False, density=True, fontsize=10, s=25, n_feat=3, labels=y, cmap='viridis')
        else:
            raise Exception('Wrong output method given. Following methods are accepted: "loadings", "var", "topfeat, '
                            '"plot", "biplot", "project", ')
