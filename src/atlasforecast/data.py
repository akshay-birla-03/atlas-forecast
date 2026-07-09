"""Synthetic multi-SKU retail demand generator.

Each SKU gets a trend, weekly + yearly seasonality, promo spikes, price effects
and noise, so the forecasting models have real structure to learn and the
backtest numbers mean something. Deterministic given a seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class DemandSeries:
    sku: str
    dates: list[str]          # ISO daily dates
    demand: list[float]
    promo: list[int]
    price: list[float]


def _daterange(start_year: int, n_days: int) -> list[str]:
    # avoid datetime.now(); build dates arithmetically from a fixed epoch
    from datetime import date, timedelta

    d0 = date(start_year, 1, 1)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n_days)]


def generate(n_skus: int = 6, n_days: int = 730, seed: int = 42) -> list[DemandSeries]:
    rnd = random.Random(seed)
    dates = _daterange(2022, n_days)
    weekday = [(i % 7) for i in range(n_days)]
    series: list[DemandSeries] = []
    for s in range(n_skus):
        base = rnd.uniform(40, 200)
        trend = rnd.uniform(-0.03, 0.08)                 # units/day drift
        weekly_amp = rnd.uniform(0.1, 0.4) * base
        yearly_amp = rnd.uniform(0.05, 0.25) * base
        weekly_phase = rnd.uniform(0, 2 * math.pi)
        price0 = rnd.uniform(5, 60)
        elasticity = rnd.uniform(-1.5, -0.4)
        demand, promo, price = [], [], []
        for t in range(n_days):
            wk = weekly_amp * math.sin(2 * math.pi * weekday[t] / 7 + weekly_phase)
            yr = yearly_amp * math.sin(2 * math.pi * t / 365.25)
            is_promo = 1 if rnd.random() < 0.06 else 0
            p = round(price0 * (0.7 if is_promo else rnd.uniform(0.95, 1.05)), 2)
            price_effect = elasticity * (p - price0) / price0 * base
            promo_lift = 0.6 * base if is_promo else 0.0
            mu = base + trend * t + wk + yr + price_effect + promo_lift
            val = max(0.0, mu + rnd.gauss(0, 0.08 * base))
            demand.append(round(val, 2))
            promo.append(is_promo)
            price.append(p)
        series.append(DemandSeries(f"SKU_{s:02d}", dates, demand, promo, price))
    return series
