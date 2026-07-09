"""End-to-end orchestration: generate → backtest → select best → plan orders."""
from __future__ import annotations

import math

from . import backtest
from .data import generate
from .inventory import SkuEconomics, recommend
from .models import build


def _resid_std(demand: list[float], model_name: str, season: int, horizon: int) -> float:
    """One-shot in-sample residual std to size forecast uncertainty."""
    cut = len(demand) - horizon
    if cut <= 2 * season:
        cut = len(demand)
    preds = build(model_name, season).fit(demand[:cut]).predict(len(demand) - cut) or []
    resid = [demand[cut + i] - preds[i] for i in range(len(preds))]
    if len(resid) < 2:
        m = sum(demand) / len(demand)
        return 0.15 * m
    mean = sum(resid) / len(resid)
    var = sum((r - mean) ** 2 for r in resid) / (len(resid) - 1)
    return math.sqrt(var)


def run(n_skus: int = 6, n_days: int = 730, horizon: int = 14, season: int = 7,
        seed: int = 42) -> dict:
    series = generate(n_skus=n_skus, n_days=n_days, seed=seed)
    bt = backtest.run(series, horizon=horizon, season=season)
    best_by_sku = {r.sku: r.best_model for r in bt}

    plans = []
    total_cost = 0.0
    for s in series:
        model = best_by_sku[s.sku]
        fc = build(model, season).fit(s.demand).predict(horizon)
        std = _resid_std(s.demand, model, season, horizon)
        avg_price = sum(s.price) / len(s.price)
        econ = SkuEconomics(
            unit_cost=round(avg_price * 0.6, 2),
            unit_price=round(avg_price, 2),
            holding_cost_per_unit=round(avg_price * 0.02, 3),
            stockout_penalty_per_unit=round(avg_price * 0.1, 2),
            lead_time_days=3,
        )
        rec = recommend(fc, std, econ)
        total_cost += rec["expected_cost"]
        plans.append({
            "sku": s.sku,
            "chosen_model": model,
            "forecast_next_h": [round(x, 1) for x in fc],
            "recommendation": rec,
        })

    return {
        "skus": n_skus,
        "horizon": horizon,
        "model_selection": best_by_sku,
        "backtest": {r.sku: r.per_model for r in bt},
        "plans": plans,
        "total_expected_cost": round(total_cost, 2),
    }
