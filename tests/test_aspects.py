from src.aspects import (
    CoTConfig,
    SpikeConfig,
    UtilityWeights,
    apply_cot,
    calibrate_configs,
    compute_utility,
    detect_spikes,
    rolling_z,
)


def test_detect_spikes_with_refractory_window():
    cfg = SpikeConfig(tau_abs=3.0, tau_rel=1.0, window=2, refractory=2)
    spikes = detect_spikes([1.0, 4.0, 3.5, 1.0, 5.0], cfg)
    assert spikes == [1, 4]


def test_rolling_z_scores_zero_when_variance_zero():
    assert rolling_z([2.0, 2.0, 2.0])[-1] == 0.0


def test_apply_cot_prefixes_prompt():
    cfg = CoTConfig(prefix="Think aloud.", temperature=0.3, min_p=0.05)
    out = apply_cot("Question?", cfg)
    assert out["prompt"].startswith("Think aloud.")
    assert out["kwargs"]["temperature"] == 0.3
    assert out["kwargs"]["min_p"] == 0.05


def test_utility_calibration_picks_best_config():
    weights = UtilityWeights(value=1.0, cost_cot=0.2, cost_rb=0.05, cost_latency=0.01)
    configs = [
        {"name": "cfg1", "tau_abs": 4.0},
        {"name": "cfg2", "tau_abs": 5.0},
    ]
    stats = {
        "cfg1": {"delta_acc": 0.05, "cot_used": 1, "rollback_tokens": 5, "delta_time": 0.1},
        "cfg2": {"delta_acc": 0.02, "cot_used": 0, "rollback_tokens": 1, "delta_time": 0.05},
    }
    best, utility = calibrate_configs(configs, stats, weights)
    assert best["name"] == "cfg1"
    assert utility < 0.05  # utility includes penalties
    assert compute_utility(0.05, True, 5, 0.1, weights) == utility
