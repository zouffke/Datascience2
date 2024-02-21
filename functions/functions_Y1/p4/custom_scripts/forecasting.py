import numpy as np
import pandas as pd

from functions.p4.forecast import *


def __drawGraph__(dataSet: pd.Series, n: int, title=None, yLabel=None, kwartaal=False, forecastAmount=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    if forecastAmount is None:
        forecastAmount = 0

    if title is not None:
        ax.set_title(title)

    if kwartaal:
        ax.set_xlabel('kwartielen')
        ax2 = ax.secondary_xaxis('top')
        ax2.set_xticks(range(n + forecastAmount))
        ax2.set_xticklabels(['Q{}'.format(j % 4 + 1) for j in range(n + forecastAmount)])
    else:
        ax.set_xlabel('tijd')

    if yLabel is not None:
        ax.set_ylabel(yLabel)

    ax.set_xticks(range(n + forecastAmount))
    ax.plot(dataSet, label='gegeven', color='C0', marker='o')
    for i in range(0, n + forecastAmount, 4):
        ax.axvline(i, color='gray', linewidth=0.5)

    return ax


def forecast(dataSet: pd.Series, title=None, yLabel=None, kwartaal=False, forecastMethod=None, forecastAmount=None,
             output='graph', mValue=4):
    n = dataSet.size

    if str.lower(output) == 'graph':

        ax = __drawGraph__(dataSet, n, title, yLabel, kwartaal, forecastAmount)

        if forecastMethod is not None:
            a = []
            method = str.lower(forecastMethod)

            if method in ('naive', 'average'):
                for i in range(forecastAmount):
                    if method == 'naive':
                        a += [naive(dataSet)]
                    else:
                        a += [average(dataSet)]

            elif method in ('moving_average', 'linear_combination'):
                for i in range(forecastAmount):
                    if method == 'moving_average':
                        a += [moving_average(dataSet, mValue)]
                    elif method == 'linear_combination':
                        a += [linear_combination(dataSet, mValue)]

                    dataSet = np.append(dataSet, [a[i]])
            else:
                raise Exception("Invalid forecast method, following methods are accepted: 'naive', 'average', "
                                "'moving_average', 'linear_combination'")

            forc = pd.Series(a, index=[i + n for i in range(forecastAmount)])
            ax.plot(forc, color='r', label=forecastMethod, marker='^')

        ax.legend()
        return

    if str.lower(output) == 'table':
        if forecastMethod is None:
            raise Exception("A forecast method is required for this mode")

        forc = pd.DataFrame(columns=[i + n for i in range(forecastAmount)], index=[forecastMethod])

        method = str.lower(forecastMethod)

        if method in ('naive', 'average'):
            for i in range(forecastAmount):
                if method == 'naive':
                    forc[i + n] = [naive(dataSet)]
                else:
                    forc[i + n] = [average(dataSet)]
        elif method in ('moving_average', 'linear_combination'):
            for i in range(forecastAmount):
                if method == 'moving_average':
                    forc[i + n] = [moving_average(dataSet, mValue)]
                elif method == 'linear_combination':
                    forc[i + n] = [linear_combination(dataSet, mValue)]

                dataSet = np.append(dataSet, [forc[i + n]])
        else:
            raise Exception("Invalid forecast method, following methods are accepted: 'naive', 'average', "
                            "'moving_average', 'linear_combination'")
        return forc

    else:
        raise Exception('Invalid output mode, "graph" and "table" are accepted')


def compare(dataSet: pd.Series, forecastMethod: str, title=None, yLabel=None, kwartaal=False, mValue=4, output='graph'):
    n = dataSet.size
    method = str.lower(forecastMethod)
    a = []

    for i in range(n):
        if method in ('naive', 'average'):
            if method == 'naive':
                a += [naive(dataSet[0:i])]
            else:
                a += [average(dataSet[0:i])]

        elif method in ('moving_average', 'linear_combination'):
            if method == 'moving_average':
                a += [moving_average(dataSet[0:i], mValue)]
            elif method == 'linear_combination':
                a += [linear_combination(dataSet[0:i], mValue)]
        else:
            raise Exception("Invalid forecast method, following methods are accepted: 'naive', 'average', "
                            "'moving_average', 'linear_combination'")

    if str.lower(output) == 'graph':
        ax = __drawGraph__(dataSet, n, title, yLabel, kwartaal)
        forc = pd.Series(a, index=[i for i in range(len(a))])
        ax.plot(forc, color='r', label=forecastMethod, marker='^')
        ax.legend()
        return

    elif str.lower(output) == 'table':
        temp = pd.DataFrame(dataSet, columns=['Given'])
        temp[forecastMethod] = a
        return temp
    else:
        raise Exception('Invalid output mode, "graph" and "table" are accepted')


def foutBepaling(dataSet: pd.Series, method: str, mValue=4):
    method = str.lower(method)
    methods = ['naive', 'average', 'moving_average', 'linear_combination']
    if method in ('naive', 'average', 'moving_average', 'linear_combination', 'all'):
        if method == 'all':
            r = pd.DataFrame()
            for i in range(len(methods)):
                t = compare(dataSet, forecastMethod=methods[i], output='table', mValue=mValue)
                r = pd.concat([r, forecast_errors(t.iloc[:, 0], t.iloc[:, 1], methods[i])])
            return r
        else:
            t = compare(dataSet, forecastMethod=method, output='table', mValue=mValue)
            return forecast_errors(t.iloc[:, 0], t.iloc[:, 1], method)
    else:
        raise Exception("Invalid method, following methods are accepted: 'naive', 'average', "
                        "'moving_average', 'linear_combination', 'all'")


def trendBepaling(dataSet: pd.Series, title=None, yLabel=None, kwartaal=False, forecastAmount=None):
    n = dataSet.size
    if forecastAmount is None:
        forecastAmount = 0
    ax = __drawGraph__(dataSet, n, title, yLabel, kwartaal, forecastAmount)

    x = create_trend_model(dataSet)
    a = []
    for i in range(n + forecastAmount):
        a += [x(i)]
    trend = pd.Series(a, index=[i for i in range(n + forecastAmount)])
    ax.plot(trend, color='r', label='trend', marker='^')
    ax.legend()
    return


def autocorrelatie(dataSet: pd.Series, mode='graph', maxlags=10, top_n=1, name=None):
    mode = str.lower(mode)
    if name is None:
        name = ''
    name = f'Auto correlation {name}'

    if mode == 'graph':
        fig, ax = plt.subplots(figsize=(10, 5))
        lags, acfs, _, _ = ax.acorr(dataSet)
        ax.set_xticks(range(-maxlags, maxlags + 1))
        ax.set_xlabel('offset')
        ax.set_ylabel('correlatie')
        ax.set_title(name)
        return
    elif mode == 'number':
        t = find_period(dataSet, maxlags, top_n)
        m = t[0]
        print(f'De m waarde is {m}')
        return m


def __componenten__(dataSet: pd.Series, comp: np.array, mode: str, title: str, output='graph', yLabel=None,
                    kwartaal=False, overlay= True):
    n = dataSet.size
    if str.lower(output) == 'graph':
        if overlay:
            ax = __drawGraph__(dataSet, n, title, yLabel, kwartaal, None)
            ax.plot(comp, color='r', label=mode, marker='^')
            ax.legend()
        else:
            ax = __drawGraph__(comp, n, title, yLabel, kwartaal, None)
            ax.legend()
            return
    if str.lower(output) == 'table':
        temp = pd.DataFrame(dataSet, columns=['Given'])
        temp[mode] = comp
        return temp


def trendComponent(dataSet: pd.Series, mValue=4, output='graph', title=None, yLabel=None, kwartaal=False):
    t = find_trend(dataSet, mValue)
    if title is None:
        title = ''
    return __componenten__(dataSet, t, 'T(i)', f'Trendcomponent {title}', output, yLabel, kwartaal)


def seizoensComponent(dataSet: pd.Series, mValue=4, output='graph', mode='additive', title=None, yLabel=None,
                      kwartaal=False, overlay=False):
    if mode not in ('additive', 'multiplicative'):
        raise Exception("Invalid mode, following modes are accepted: 'additive', 'multiplicative'")
    s = find_seasons(dataSet, mValue, mode)
    if title is None:
        title = ''
    return __componenten__(dataSet, s, f'S(i) {mode}', f'Seizoens component {title}', output, yLabel, kwartaal, overlay)
