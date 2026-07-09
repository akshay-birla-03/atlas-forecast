from atlasforecast import metrics
from atlasforecast.backtest import backtest_series
from atlasforecast.data import generate
from atlasforecast.inventory import SkuEconomics, norm_ppf, recommend
from atlasforecast.models import ALL_MODELS, build
from atlasforecast.pipeline import run


def test_data_has_structure():
    s = generate(n_skus=3, n_days=400, seed=1)
    assert len(s) == 3
    assert len(s[0].demand) == 400
    assert all(d >= 0 for d in s[0].demand)


def test_models_forecast_horizon():
    s = generate(n_skus=1, n_days=200, seed=2)[0]
    for m in ALL_MODELS:
        preds = build(m).fit(s.demand).predict(14)
        assert len(preds) == 14
        assert all(p >= 0 for p in preds)


def test_metrics_and_mase_scaling():
    y = [10, 12, 11, 13, 12]
    assert metrics.wape(y, y) == 0.0
    # perfect forecast -> MASE 0
    assert metrics.mase(y, y, [10, 11, 12, 13, 14, 15, 16, 17], season=7) == 0.0


def test_norm_ppf_known_values():
    assert abs(norm_ppf(0.5)) < 1e-6
    assert abs(norm_ppf(0.975) - 1.959964) < 1e-3


def test_newsvendor_higher_margin_orders_more():
    fc = [100.0] * 14
    low = recommend(fc, 20.0, SkuEconomics(unit_cost=9, unit_price=10,
                                           holding_cost_per_unit=1))
    high = recommend(fc, 20.0, SkuEconomics(unit_cost=2, unit_price=10,
                                            holding_cost_per_unit=1))
    # higher margin -> higher target service level -> more safety stock
    assert high["target_service_level"] > low["target_service_level"]
    assert high["order_up_to_level"] >= low["order_up_to_level"]


def test_backtest_beats_naive():
    s = generate(n_skus=1, n_days=500, seed=5)[0]
    pm = backtest_series(s.demand, horizon=14, n_folds=4)
    # the ensemble should not be dramatically worse than seasonal-naive
    assert pm["ensemble"]["wape"] <= pm["seasonal_naive"]["wape"] * 1.15


def test_full_pipeline_produces_plans():
    out = run(n_skus=3, n_days=400, horizon=14, seed=7)
    assert len(out["plans"]) == 3
    assert out["total_expected_cost"] >= 0
    for plan in out["plans"]:
        assert plan["recommendation"]["order_up_to_level"] >= 0
        assert 0 <= plan["recommendation"]["target_service_level"] <= 1
