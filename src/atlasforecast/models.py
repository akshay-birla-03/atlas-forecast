"""Forecasting models behind a common `fit(history) -> predict(h)` interface.

- SeasonalNaive: strong, honest baseline (repeat last season).
- HoltWinters: additive triple exponential smoothing, implemented in pure numpy.
- GBMForecaster: gradient-boosted trees on lag/calendar features, forecast recursively.
- Ensemble: average of HoltWinters + GBM (usually beats either alone).

Keeping HoltWinters dependency-free means the core runs and tests offline with
only numpy; the GBM path uses scikit-learn.
"""
from __future__ import annotations

from typing import Protocol


class Forecaster(Protocol):
    name: str

    def fit(self, history: list[float]) -> "Forecaster": ...
    def predict(self, h: int) -> list[float]: ...


class SeasonalNaive:
    name = "seasonal_naive"

    def __init__(self, season: int = 7):
        self.season = season
        self._hist: list[float] = []

    def fit(self, history):
        self._hist = list(history)
        return self

    def predict(self, h):
        s, hist = self.season, self._hist
        if len(hist) < s:
            last = hist[-1] if hist else 0.0
            return [last] * h
        return [hist[-s + (i % s)] for i in range(h)]


class HoltWinters:
    """Additive Holt-Winters (level + trend + seasonal)."""

    name = "holt_winters"

    def __init__(self, season: int = 7, alpha=0.4, beta=0.05, gamma=0.3):
        self.season = season
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self._level = self._trend = 0.0
        self._seasonals: list[float] = []

    def fit(self, history):
        m, y = self.season, list(history)
        if len(y) < 2 * m:
            self._level = sum(y) / len(y)
            self._trend = 0.0
            self._seasonals = [0.0] * m
            return self
        n_seasons = len(y) // m
        season_avgs = [sum(y[i * m:(i + 1) * m]) / m for i in range(n_seasons)]
        seasonals = [0.0] * m
        for i in range(m):
            seasonals[i] = sum(y[j * m + i] - season_avgs[j] for j in range(n_seasons)) / n_seasons
        level = season_avgs[0]
        trend = (season_avgs[1] - season_avgs[0]) / m
        for t, val in enumerate(y):
            s_idx = t % m
            last_level = level
            level = self.alpha * (val - seasonals[s_idx]) + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - last_level) + (1 - self.beta) * trend
            seasonals[s_idx] = self.gamma * (val - level) + (1 - self.gamma) * seasonals[s_idx]
        self._level, self._trend, self._seasonals = level, trend, seasonals
        return self

    def predict(self, h):
        m = self.season
        return [max(0.0, self._level + (i + 1) * self._trend + self._seasonals[i % m])
                for i in range(h)]


class GBMForecaster:
    name = "gbm"

    def __init__(self, season: int = 7, lags=(1, 2, 3, 7, 14)):
        self.season = season
        self.lags = lags
        self._model = None
        self._hist: list[float] = []

    def _row(self, series, t):
        feats = [series[t - lag] for lag in self.lags]
        feats.append(sum(series[t - 7:t]) / 7 if t >= 7 else series[t - 1])
        feats.append((t % 7))
        return feats

    def fit(self, history):
        from sklearn.ensemble import HistGradientBoostingRegressor

        y = list(history)
        self._hist = y
        start = max(self.lags)
        X = [self._row(y, t) for t in range(start, len(y))]
        target = y[start:]
        self._model = HistGradientBoostingRegressor(max_depth=5, learning_rate=0.08,
                                                     max_iter=200)
        self._model.fit(X, target)
        return self

    def predict(self, h):
        series = list(self._hist)
        out = []
        for _ in range(h):
            t = len(series)
            pred = float(self._model.predict([self._row(series, t)])[0])
            pred = max(0.0, pred)
            out.append(pred)
            series.append(pred)
        return out


class Ensemble:
    name = "ensemble"

    def __init__(self, season: int = 7):
        self.models = [HoltWinters(season), GBMForecaster(season)]

    def fit(self, history):
        for m in self.models:
            m.fit(history)
        return self

    def predict(self, h):
        preds = [m.predict(h) for m in self.models]
        return [sum(p[i] for p in preds) / len(preds) for i in range(h)]


def build(name: str, season: int = 7) -> Forecaster:
    return {
        "seasonal_naive": SeasonalNaive,
        "holt_winters": HoltWinters,
        "gbm": GBMForecaster,
        "ensemble": Ensemble,
    }[name](season)


ALL_MODELS = ["seasonal_naive", "holt_winters", "gbm", "ensemble"]
