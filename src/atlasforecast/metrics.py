"""Forecast-accuracy metrics: MAPE, WAPE, RMSE, bias, and MASE.

MASE (Mean Absolute Scaled Error) is the honest one — it scales error by the
in-sample seasonal-naive error, so a value < 1 means "better than naive".
"""
from __future__ import annotations

import math


def _abs_err(y, yhat):
    return [abs(a - b) for a, b in zip(y, yhat)]


def mape(y, yhat) -> float:
    pairs = [(a, b) for a, b in zip(y, yhat) if a != 0]
    if not pairs:
        return float("nan")
    return sum(abs(a - b) / abs(a) for a, b in pairs) / len(pairs)


def wape(y, yhat) -> float:
    denom = sum(abs(a) for a in y)
    if denom == 0:
        return float("nan")
    return sum(_abs_err(y, yhat)) / denom


def rmse(y, yhat) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, yhat)) / max(1, len(y)))


def bias(y, yhat) -> float:
    return sum(b - a for a, b in zip(y, yhat)) / max(1, len(y))


def mase(y_true, y_pred, y_train, season: int = 7) -> float:
    """Scale absolute error by the in-sample seasonal-naive MAE."""
    if len(y_train) <= season:
        naive = [abs(y_train[i] - y_train[i - 1]) for i in range(1, len(y_train))]
    else:
        naive = [abs(y_train[i] - y_train[i - season]) for i in range(season, len(y_train))]
    scale = sum(naive) / max(1, len(naive))
    if scale == 0:
        return float("nan")
    mae = sum(_abs_err(y_true, y_pred)) / max(1, len(y_true))
    return mae / scale


def summary(y_true, y_pred, y_train, season: int = 7) -> dict:
    return {
        "mape": round(mape(y_true, y_pred), 4),
        "wape": round(wape(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "bias": round(bias(y_true, y_pred), 4),
        "mase": round(mase(y_true, y_pred, y_train, season), 4),
    }
