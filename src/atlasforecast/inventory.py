"""Inventory-optimization decision layer — turns forecasts into $ decisions.

Uses the classic newsvendor / order-up-to model: given a demand forecast and its
uncertainty, the economics of a stockout (lost margin) vs. overstock (holding +
obsolescence) imply an optimal service level, from which we derive safety stock
and a recommended order quantity — plus the projected cost of that decision.

This is the layer that makes the project speak to management: not "the forecast
is X" but "order Y units to hit a Z% service level at the lowest expected cost".
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _norm_pdf(z):
    return math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


@dataclass
class SkuEconomics:
    unit_cost: float
    unit_price: float
    holding_cost_per_unit: float          # per period, overage cost
    stockout_penalty_per_unit: float = 0.0  # extra cost beyond lost margin
    lead_time_days: int = 3


def recommend(forecast_mean_per_day: list[float], forecast_std_per_day: float,
              econ: SkuEconomics) -> dict:
    """Order-up-to recommendation over the lead time + review horizon."""
    horizon = len(forecast_mean_per_day)
    lead = max(1, econ.lead_time_days)
    protect = min(horizon, lead)                    # protection period
    mu = sum(forecast_mean_per_day[:protect])
    sigma = forecast_std_per_day * math.sqrt(protect)

    underage = (econ.unit_price - econ.unit_cost) + econ.stockout_penalty_per_unit
    overage = econ.holding_cost_per_unit
    underage = max(underage, 1e-6)
    critical_ratio = underage / (underage + overage)
    z = norm_ppf(critical_ratio)

    safety_stock = max(0.0, z * sigma)
    order_up_to = mu + safety_stock

    # expected units short/held per period (normal loss function)
    loss = _norm_pdf(z) - z * (1 - critical_ratio)
    expected_short = max(0.0, sigma * loss)
    expected_overage = max(0.0, safety_stock - expected_short)
    expected_cost = round(overage * expected_overage + underage * expected_short, 2)

    return {
        "protection_days": protect,
        "demand_mean": round(mu, 2),
        "demand_std": round(sigma, 2),
        "target_service_level": round(critical_ratio, 4),
        "safety_stock": round(safety_stock, 2),
        "order_up_to_level": round(order_up_to, 2),
        "expected_units_short": round(expected_short, 2),
        "expected_cost": expected_cost,
    }
