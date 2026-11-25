"""Non-regression tests for meca_latency_test.py.

Fast (default):   pytest
Slow (re-analyses every recorded v2 run and compares with its summary.json):   pytest -m slow
"""
import glob
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import meca_latency_test as m  # noqa: E402

SCRIPT = os.path.join(ROOT, "meca_latency_test.py")


# --------------------------------------------------------------------------- model fit
def test_fit_boxcar_recovers_delay_and_window():
    rng = np.random.default_rng(0)
    ct = np.arange(0, 1.4, 0.002)
    shape = m.piecewise([(0.15, 0.0), (0.3, 50.0), (0.3, 0.0), (0.3, -50.0), (0.35, 0.0)])
    cv = np.array([shape(t) for t in ct])
    tq = np.arange(-0.05, 1.55, 0.001)
    y = m.boxcar_response(tq, m._staircase_integral(ct, cv), 0.004, 0.0197) + rng.normal(0, 0.3, tq.size)
    f = m.fit_boxcar(ct, cv, tq, y)
    assert abs(f["tau_ms"] - 4.0) < 0.3
    assert abs(f["T_ms"] - 19.7) < 0.3


def test_model_summary_derived_values():
    s = m.model_summary([dict(tau_ms=4.0, T_ms=16.0, rms=0.1, span=50)])
    assert s["eq_delay_ms"] == pytest.approx(12.0)
    assert s["bw_3db_hz"] == pytest.approx(443.0 / 16.0)


# --------------------------------------------------------------------------- clock map
def test_clock_map_ignores_a_frame_stamped_too_early():
    rng = np.random.default_rng(1)
    r = np.arange(0, 60, 0.001)                       # robot clock, 60 s at 1 kHz
    offset, drift = 1234.5, 50e-6
    h = r * (1 + drift) + offset + 0.0003 + rng.exponential(0.002, r.size)  # true min latency 0.3 ms
    h[30_000] -= 0.009                                # one host stamp taken 9 ms too early
    cm = m.clock_map(h, r)
    err = cm(r) - (r * (1 + drift) + offset + 0.0003)
    assert np.max(np.abs(err)) < 3e-4                 # the bad frame does not shift the map


# --------------------------------------------------------------------------- loop model
def test_phase_margin_search_and_limits():
    mdl = dict(tau=0.004, T=0.0159, P=0.002, tau_b=0.0039, tau_s=0.0003, fb_actual=True)
    r60 = m.kp_for_pm(60, mdl)
    assert abs(r60["pm_deg"] - 60) < 2
    assert 4.0 < r60["fc_hz"] < 6.0                   # measured 2026-11-25: 4.9 Hz
    assert m.margins(1.0, mdl)["pm_deg"] > 85         # tiny gain: integrator only


def test_closed_loop_low_frequency_gain_is_one():
    mdl = dict(tau=0.004, T=0.0159, P=0.002, tau_b=0.0039, tau_s=0.0003, fb_actual=True)
    for ff in (False, True):
        assert abs(abs(m.closed_loop_frf(0.01, 20.0, mdl, ff, 0.014)) - 1) < 1e-3


# --------------------------------------------------------------------------- replay reference
def test_action_ref_uses_only_released_actions():
    t = np.array([0.0, 0.02, 0.04, 0.06])
    q = np.zeros((4, m.NJ))
    q[:, 0] = [0.0, 1.0, 2.0, 3.0]                    # 50 deg/s
    R = m.ActionRef(t, q, release=[0.0, 0.0, 0.04, 0.04], horizon=0.04)   # chunks of 2
    assert R.at(0.01, now=0.01)[0] == pytest.approx(0.5)      # interpolated inside the known chunk
    assert R.at(0.03, now=0.03)[0] == pytest.approx(1.5)      # next chunk not out yet: extrapolated
    assert R.slope(0.05, now=0.03)[0] == pytest.approx(50.0)
    assert R.at(0.20, now=0.03)[0] == pytest.approx(3.0)      # extrapolation capped by the horizon
    assert R.at(0.20, now=1.0)[0] == pytest.approx(3.0)       # trajectory over: held
    assert R.slope(0.20, now=1.0)[0] == 0.0


def test_policy_effects_chunks_and_release():
    a = m.parse_args(["--ref-chunk", "4", "--ref-chunk-offset-deg", "0.5", "--ref-jitter-ms", "3"])
    t = np.arange(0, 0.2, 0.02)
    q, rel = m.policy_effects(t, np.zeros((len(t), m.NJ)), a, [5])
    assert np.all(rel[:4] == rel[0]) and rel[4] >= t[4]       # a chunk is released at once, never early
    assert np.all(q[:4, 5] == 0) and np.ptp(q[4:8, 5]) == 0   # first chunk clean, one offset per chunk
    assert np.all(q[:, :5] == 0)                              # inactive joints untouched


# --------------------------------------------------------------------------- end to end (simulator)
def _sim(tmp_path, *args):
    out = tmp_path / "run"
    cmd = [sys.executable, SCRIPT, "--sim", "--out", str(out), "--vel-limit-pct", "20", "--repeats", "2",
           "--max-excursion", "20", *args]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "failed" not in p.stdout, p.stdout[-3000:]
    with open(out / "summary.json", encoding="utf-8") as f:
        return json.load(f)


def test_sim_step_and_vmaxcheck(tmp_path):
    S = _sim(tmp_path, "--tests", "step,vmaxcheck", "--vmax-joints", "1,6")
    assert abs(S["step"]["model"]["T_ms"] - 20.0) < 0.5       # simulator law at VelLimit 20 %
    vm = {r["joint"]: r for r in S["vmaxcheck"]}
    assert vm[1]["vmax_measured"] == pytest.approx(150, rel=0.02)
    assert vm[6]["vmax_measured"] == pytest.approx(500, rel=0.02)


def test_sim_servomulti_matches_model(tmp_path):
    S = _sim(tmp_path, "--tests", "step,servomulti", "--multi-joints", "1,6", "--servomulti-freqs", "1")
    for r in S["servomulti"]["rows"]:
        assert abs(r["err_rms_pct"] - r["model_err_rms_pct"]) < 3


def test_sim_replay_all_interfaces(tmp_path):
    S = _sim(tmp_path, "--tests", "step,replay", "--multi-joints", "1,6", "--replay-dur", "3",
             "--ref-chunk", "8", "--ref-noise-deg", "0.02", "--ref-jitter-ms", "5", "--ref-chunk-offset-deg", "0.1")
    rows = S["replay"]["rows"]
    assert {(r["mode"], r["joint"]) for r in rows} == {(mo, j) for mo in ("vel", "ff", "pos") for j in (1, 6)}
    for r in rows:
        assert -5 < r["delay_ms"] < 150, r
        assert r["rel_err_pct"] < 80, r


# --------------------------------------------------------------------------- recorded runs (slow)
RUNS = sorted(d for d in glob.glob(os.path.join(ROOT, "meca_latency_2026*"))
              if os.path.isfile(os.path.join(d, "raw.pkl.gz")) and os.path.isfile(os.path.join(d, "summary.json")))


def _compare(a, b, path=""):
    """Every value of the recorded summary `a` must be reproduced in `b`; keys added since are allowed."""
    if isinstance(a, dict):
        assert isinstance(b, dict), path
        for k in a:
            if k in ("config", "robot_info"):
                continue
            assert k in b, f"{path}/{k} missing"
            _compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None, path
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=2e-4), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a} != {b}"


@pytest.mark.slow
@pytest.mark.parametrize("run", RUNS, ids=[os.path.basename(r) for r in RUNS])
def test_reanalyze_matches_recorded_summary(run, tmp_path):
    out = tmp_path / "re"
    p = subprocess.run([sys.executable, SCRIPT, "--reanalyze", run, "--out", str(out)],
                       capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    with open(os.path.join(run, "summary.json"), encoding="utf-8") as f:
        ref = json.load(f)
    with open(out / "summary.json", encoding="utf-8") as f:
        new = json.load(f)
    _compare(ref, new)
