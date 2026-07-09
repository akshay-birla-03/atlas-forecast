"""Rolling-origin backtesting and automatic model selection.

For each SKU we walk the series forward: train on everything up to a cut point,
forecast the next `horizon` days, score against actuals, then slide the cut
forward. This mirrors how a forecast is actually used in production (no peeking
at the future) and is far more honest than a single train/test split.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import metrics
from .models import ALL_MODELS, build


@dataclass
class BacktestResult:
    sku: str
    horizon: int
    per_model: dict = field(default_factory=dict)   # model -> aggregated metrics
    best_model: str = ""


def backtest_series(demand: list[float], horizon: int = 14, n_folds: int = 4,
                    season: int = 7, models=None) -> dict:
    models = models or ALL_MODELS
    n = len(demand)
    min_train = max(2 * season + 1, n - n_folds * horizon)
    cuts = [min_train + i * horizon for i in range(n_folds)]
    cuts = [c for c in cuts if c + horizon <= n]

    scores = {m: {"mape": [], "wape": [], "mase": []} for m in models}
    for cut in cuts:
        train = demand[:cut]
        actual = demand[cut:cut + horizon]
        for m in models:
            preds = build(m, season).fit(train).predict(horizon)
            summ = metrics.summary(actual, preds, train, season)
            for k in ("mape", "wape", "mase"):
                if summ[k] == summ[k]:  # not NaN
                    scores[m][k].append(summ[k])

    per_model = {}
    for m in models:
        agg = {k: (sum(v) / len(v) if v else float("nan")) for k, v in scores[m].items()}
        per_model[m] = {k: round(x, 4) for k, x in agg.items()}
    return per_model


def run(series_list, horizon: int = 14, n_folds: int = 4, season: int = 7) -> list[BacktestResult]:
    results = []
    for s in series_list:
        per_model = backtest_series(s.demand, horizon, n_folds, season)
        rankable = {m: v["wape"] for m, v in per_model.items() if v["wape"] == v["wape"]}
        best = min(rankable, key=rankable.get) if rankable else "ensemble"
        results.append(BacktestResult(s.sku, horizon, per_model, best))
    return results
