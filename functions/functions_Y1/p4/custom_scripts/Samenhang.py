import pandas as pd
import scipy.stats as stats
import numpy as np


def scatterPlot(dataSet: pd.Series, afhankelijk: pd.Series, onafhankelijk: pd.Series, title='default', grid='--',
                marker='o', xDivide=None, yDivide=None):
    """
    Used to make a ScatterPlot
    :param dataSet: de volledige dataset
    :param afhankelijk: de afhankelijke variabele
    :param onafhankelijk: de onafhankelijke variabele
    :param title: de titel van de grafiek (default: "onafhankelijke variabele" vs "afhankelijke variabele")
    :param grid: de stijl van het grid (default: '--')
    :param marker: de stijl van de markers (default: 'o')
    :param xDivide: een lijn die de x-as verdeelt (default: [True, 'r', '--'])
    :param yDivide: een lijn die de y-as verdeelt (default: [True, 'r', '--'])
    :return: de grafiek
    """

    if yDivide is None:
        yDivide = [True, 'r', '--']
    if xDivide is None:
        xDivide = [True, 'r', '--']

    if str.lower(title) == 'default':
        title = "{} vs {}".format(onafhankelijk.name, afhankelijk.name)

    ax = dataSet.plot(kind='scatter', x=onafhankelijk.name, y=afhankelijk.name,
                      figsize=(10, 5), title=title,
                      marker=marker, alpha=0.3)
    ax.grid(linestyle=grid)

    if not xDivide[0] and not yDivide[0]:
        return
    elif xDivide[0] and not yDivide[0]:
        ax.axvline(x=onafhankelijk.mean(), color=xDivide[1], linestyle=xDivide[2])
        return
    elif not xDivide[0] and yDivide[0]:
        ax.axhline(y=afhankelijk.mean(), color=yDivide[1], linestyle=yDivide[2])
        return
    else:
        ax.axvline(x=onafhankelijk.mean(), color=xDivide[1], linestyle=xDivide[2])
        ax.axhline(y=afhankelijk.mean(), color=yDivide[1], linestyle=yDivide[2])
        return


def correlatiecoefficient(dataSet: pd.Series, afhankelijk=None, onafhankelijk=None, mode='number',
                          colored=False, divide=True, grid='--', method='normal'):
    """
    Gebruikt om de correlatiecoefficient te berekenen of te plotten
    :param dataSet: de volledige dataset
    :param afhankelijk: de afhankelijke variabele
    :param onafhankelijk: de onafhankelijke variabele
    :param mode: de modus waarin de correlatiecoefficient wordt geplot (default: 'number', andere optie: 'graph')
    :param colored: of de grafiek gekleurd moet worden (default: False)
    :param divide: of de grafiek verdeeld moet worden (default: True)
    :param grid: de stijl van het grid (default: '--')
    :param method: de methode die gebruikt wordt om de correlatiecoefficient te berekenen (default: 'normal', andere optie: 'spearman' of 'kendall')
    :return: de grafiek of de correlatiecoefficient
    """

    if str.lower(mode) == 'graph':
        if afhankelijk is None or onafhankelijk is None:
            raise Exception("afhankelijk en onafhankelijk moeten worden ingevuld")

        zx = stats.zscore(onafhankelijk)
        zy = stats.zscore(afhankelijk)

        if not colored:
            standard = pd.DataFrame({'zx': zx, 'zy': zy})

            ax = standard.plot(kind='scatter', x='zx', y='zy',
                               figsize=(10, 5), title='Zx vs Zy',
                               marker='o', alpha=0.3)
            grid = ax.grid(linestyle=grid)
            if divide:
                ax.axvline(x=0, color='r', linestyle='-')
                return ax.axhline(y=0, color='r', linestyle='-')
            else:
                return
        else:
            print("This may take some time...")

            colors = ['g' if z >= 0 else 'r' for z in zx * zy]
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, (x, y) in enumerate(zip(zx, zy)):
                ax.scatter(x, y, c=colors[i], alpha=0.5)
            ax.grid(linestyle=grid)
            ax.set_xlabel('zx')
            ax.set_ylabel('zy')
            ax.set_title('Zx vs Zy')

            if divide:
                ax.axvline(x=0, color='b', linestyle='-')
                ax.axhline(y=0, color='b', linestyle='-')
                return
            else:
                return

    elif str.lower(mode) == 'number':
        if str.lower(method) == 'normal':
            return dataSet.corr()
        else:
            return dataSet.corr(method=method)


def linRegressie(dataSet: pd.Series, afhankelijk: pd.Series, onafhankelijk: pd.Series, mode='prediction',
                 valuesToPredict=None, interval=False):
    """
    Gebruikt om een lineaire regressie te berekenen of te plotten
    :param dataSet: de volledige dataset
    :param afhankelijk: de afhankelijke variabele
    :param onafhankelijk: de onafhankelijke variabele
    :param mode: de modus waarin de lineaire regressie wordt geplot (default: 'prediction', andere optie: 'function', 'graph' of 'se')
    :param valuesToPredict: de waarden die voorspeld moeten worden (default: None, te gebruiken bij mode='prediction' array met waarden)
    :param interval: of het betrouwbaarheidsinterval moet worden getoond op de grafiek (default: False)
    :return: de grafiek, de functie of de voorspelde waarden
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    model = LinearRegression()

    X = dataSet[[onafhankelijk.name]]  # 2D array
    y = afhankelijk  # 1D array

    model.fit(X, y)  # trainen van het model

    y_hat = model.predict(X)
    se = mean_squared_error(y, y_hat, squared=False)


    if str.lower(mode) == 'prediction':
        if valuesToPredict is None:
            raise Exception('valuesToPredict needs to be filled in to use this function')
        try:
            valuesToPredict[0]
        except:
            raise Exception("An array needs to be passed in this parameter")

        new_values = pd.DataFrame({onafhankelijk.name: valuesToPredict})
        return model.predict(new_values)

    elif str.lower(mode) == 'function':
        rxy = np.corrcoef(onafhankelijk, afhankelijk)[0, 1]
        sx = onafhankelijk.std()
        sy = afhankelijk.std()
        a = rxy * sy / sx
        b = afhankelijk.mean() - a * onafhankelijk.mean()
        print(f'{afhankelijk.name:s} = {a:.2f} • {onafhankelijk.name:s} + {b:.2f}')
        return
    elif str.lower(mode) == 'graph':
        ax = dataSet.plot(kind='scatter', x=onafhankelijk.name, y=afhankelijk.name,
                          figsize=(10, 5), title="{} vs {}".format(onafhankelijk.name, afhankelijk.name),
                          marker='o', alpha=0.3)
        ax.plot(X, y_hat, color='r', label='voorspelling ŷ=ax+b')

        if interval:
            ax.plot(X, y_hat + 2 * se, color='g', label='betrouwbaarheidsinterval')
            ax.plot(X, y_hat - 2 * se, color='g')

        ax.grid(linestyle='--')
        ax.legend()
        return
    elif str.lower(mode) == 'se':
        print(f'Standard Error: {se}')
    else:
        raise Exception("Invalid Mode given")
