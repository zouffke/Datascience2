# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

opbrengsten = np.array([20, 100, 175, 13, 37, 136, 245, 26, 75, 155, 326, 48, 92, 202, 384, 82, 176, 282, 445, 181],
                       dtype=float)


# %% voorspelling is vorige waarde
def naive(y: np.array):
    if y.size > 0:
        return y[-1]
    return np.nan


# %% voorspelling is gemiddelde van alle vorige waarden
def average(y: np.array):
    if y.size < 1:
        return np.nan

    return y.mean()


# %% voorspelling is voortschrijdend gemiddelde van m vorige waarden
def moving_average(y: np.array, m=4):
    if y.size < m:
        return np.nan

    return np.mean(y[-m:])


# %% voorspelling is een lineaire combinatie
def bereken_gewichten(y: np.array, m: int):
    n = y.size  # n is aantal elementen
    if n < 2 * m:  # we hebben > 2 * m elementen nodig
        return np.nan
    M = y[-(m + 1):-1]  # selecteer de laatste elementen
    for i in range(1, m):  # maak een matrix M van coëfficiënten
        M = np.vstack([M, y[-(m + i + 1):-(i + 1)]])

    v = np.flip(y[-m:])  # selecteer de bekenden
    return np.linalg.solve(M, v)  # los het stelsel van m vergelijkingen op


def linear_combination(y: np.array, m=4) -> np.ndarray:
    n = y.size
    # check op minstens 2*m gegevens
    if n < 2 * m:
        return np.nan
    # bereken de gewichten
    a = bereken_gewichten(y, m)
    # bereken de voorspelde waarde en geef de voorspelde waarde terug
    return np.sum(y[-m:] * a)


# %% prediction generator
def predictor(y: np.array, f, *argv):
    i = 0
    while True:
        if i <= y.size:
            yield f(y[:i], *argv)
        else:
            y = np.append(y, f(y, *argv))
            yield f(y, *argv)
        i += 1


# %% utility function
def predict(y: np.array, start, end, f, *argv):
    generator = predictor(y, f, *argv)
    predictions = np.array([next(generator) for _ in range(end)])
    predictions[:start] = np.nan
    return predictions


# %% general regression function
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score as r2score


class GeneralRegression:
    def __init__(self, degree=1, exp=False, log=False):
        self.degree = degree
        self.exp = exp
        self.log = log
        self.model = None
        self.x_orig = None
        self.y_orig = None
        self.X = None
        self.y = None

    def fit(self, x: np.array, y: np.array):
        self.x_orig = x
        self.y_orig = y
        self.X = x.reshape(-1, 1)

        if self.exp:
            self.y = np.log(y)

        else:
            self.y = y

        if self.log:
            self.X = np.log(self.X)

        self.model = make_pipeline(PolynomialFeatures(degree=self.degree), LinearRegression())
        self.model.fit(self.X, self.y)

    def predict(self, x: np.array):
        X = x.reshape(-1, 1)

        if self.exp:
            return np.exp(self.model.predict(X))

        if self.log:
            return self.model.predict(np.log(X))

        return self.model.predict(X)

    @property
    def r2_score(self):
        return r2score(self.y_orig, self.predict(self.x_orig))

    @property
    def se_(self):
        if self.exp:
            return mean_squared_error(self.predict(self.X), np.exp(self.y), squared=False)
        if self.log:
            return mean_squared_error(self.predict(self.X), np.log(self.y), squared=False)
        return mean_squared_error(self.predict(self.X), self.y, squared=False)

    @property
    def coef_(self):
        return self.model.steps[1][1].coef_

    @property
    def intercept_(self):
        return self.model.steps[1][1].intercept_

    def get_feature_names(self):
        return self.model.steps[0][1].get_feature_names()


# %% trend model
def create_trend_model(y: np.array):
    X = np.arange(0, y.size)  # we bouwen een lineaire regressie model
    model = GeneralRegression()
    model.fit(X, y)

    return lambda x: model.predict(np.array(x).reshape(-1, 1))  # we geven een voorspellersfunctie terug


# %%
def forecast_errors(x: np.array, f: np.array, method: str):
    e = x - f
    mae = np.nanmean(np.abs(e))
    rmse = np.sqrt(np.nanmean(e ** 2))
    mape = np.nanmean(np.abs(e / x))
    avg = (mae + rmse + mape) / 3
    return pd.DataFrame({'MAE': [mae], 'RMSE': [rmse], 'MAPE': [mape], 'AVG': [avg]}, index=[method])


# %% autocorrelate period
def find_period(y: np.array, maxlags=10, top_n=1) -> int:
    # autocorrelatie aan beide zijden
    acfs = np.correlate(y, y, mode='full') / np.sum(y ** 2)
    # midden bepalen
    middle = acfs.size // 2
    # omgekeerde argsort vanaf (midden + 1) tot maxlags + top selectie
    return (np.argsort(-1 * acfs[middle + 1: middle + maxlags]) + 1)[:top_n]


# %% smoother
def smooth(y: np.array, m: int):
    result = np.empty(0)
    for i in range(y.size - m + 1):
        result = np.append(result, [np.mean(y[i:i + m])])

    return result


# %% double filter function
def find_trend(y: np.array, m: int):
    result = smooth(y, m)
    nan = [np.nan] * int(m / 2)
    if m % 2 == 0:
        result = smooth(result, 2)
        result = np.hstack([nan, result, nan])

    return result


# %% seizoengemiddelden berekenen
def find_seasons(y: np.array, m: int, method='additive'):
    if method == 'multiplicative':
        seizoen_ruis = y / find_trend(y, m)
    else:
        seizoen_ruis = y - find_trend(y, m)

    n = seizoen_ruis.size

    seizoens_patroon = np.empty(0)
    for i in range(m):  # m groepjes middellen die telkens m stappen uit elkaar liggen
        seizoens_patroon = np.append(seizoens_patroon, np.nanmean(seizoen_ruis[np.arange(i, n, m)]))

    # patroon herhalen over volledige periode
    return np.tile(seizoens_patroon, n // m)  # n // m is het aantal seizoenen.


# %% find regression models
def find_regression_models(z: np.array, m: int, degree=1, exp=False):
    reg_models = []

    for i in range(z.size // m - 1):
        x = np.arange(i, opbrengsten.size, m).reshape(-1, 1)
        y = z[x]
        reg_models.append(GeneralRegression(degree, exp))
        reg_models[i].fit(x, y)

    return reg_models


# %% forecasting met seasonal decomposition components
def seasonal_decomposition_forecast(reg_model: GeneralRegression, sd_model, start, end, method='additive', m=None):
    if not m:
        m = find_period(sd_model.observed)[0]

    # seizoenen voldoende herhalen tot voorbij 'end'
    seasonal = np.tile(sd_model.seasonal[0:m], end // m + 1)
    if method.startswith('m'):
        return reg_model.predict(np.arange(start, end)) * seasonal[start:end]
    else:
        return reg_model.predict(np.arange(start, end)) + seasonal[start:end]


# %% seasonal_trend_forecast
def create_seasonal_trend_forecast(z: np.array, m: int, degree=1, exp=False):
    reg_models = find_regression_models(z, m, degree, exp)

    def forecast(x: np.array):
        predictions = np.empty(0)

        for i in range(x.size):
            y = reg_models[i % m].predict(x[i].reshape(1, -1))
            predictions = np.append(predictions, y)

        return predictions

    return forecast


# %% compare original and forecast
def plot_trends(y1: np.array, y2=None, sub_title=None, label1='gegeven', label2='voorspeld', color='C0', ax=None):
    if y2 is not None:
        n = max(y1.size, y2.size)
    else:
        n = y1.size

    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    if sub_title:
        fig.suptitle(sub_title, y=1.02)

    ax.set_title('Opbrengsten voorbije 5 jaar')
    ax.set_xlabel('kwartaal')
    ax.set_ylabel('opbrengst (€)')
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(['Q{}'.format(j % 4 + 1) for j in range(n)])

    ax.set_xticks(range(n))
    ax.plot(y1, label=label1, color=color, marker='o')
    if y2 is not None:
        ax.plot(y2, label=label2, color='C1', marker='^')
    for i in range(0, n, 4):
        ax.axvline(i, color='gray', linewidth=0.5)

    ax.legend()


# %% seasonal decomposition plotten
def plot_seasonal_decompositon(model, title: str, figsize=(8, 8)):
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    axes[0].plot(model.observed, 'o-', label='observed')
    axes[0].set_ylabel('observed')
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(model.trend, 'o-', color='orange', label='trend')
    axes[1].set_ylabel('trend')
    axes[1].legend()

    axes[2].plot(model.seasonal, 'o-', color='green', label='seasonal')
    axes[2].set_ylabel('season')
    axes[2].legend()

    axes[3].scatter(range(model.nobs[0]), model.resid, color='darkgrey', label='noise')
    axes[3].set_ylabel('residue')
    axes[3].set_xlabel('kwartaal')
    axes[3].legend()
