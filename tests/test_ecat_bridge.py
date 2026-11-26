"""Tests of ecat_bridge.py (Python side of meca_ecat) and of the EtherCAT clock map.

The round trip test writes a simulator run of meca_latency_test.py in the meca_ecat format (what
ecat/src/out.rs writes: ecat_run.json + npy/*.npy), imports it back and requires the same summary.
"""
import gzip
import json
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ecat_bridge as eb  # noqa: E402
import meca_latency_test as m  # noqa: E402

SCRIPT = os.path.join(ROOT, "meca_latency_test.py")
FIXTURES = os.path.join(ROOT, "ecat", "tests", "fixtures")


# --------------------------------------------------------------------------- EtherCAT clock map
def _beat_run(drift=40e-6, dur=60.0, cyc=1e-3, lat_min=60e-6, jitter=20e-6, seed=0):
    """Robot ticks every 1 ms of robot time, the master every 1 ms of host time: each master cycle
    brings the newest robot sample produced at least lat_min before the frame passes."""
    rng = np.random.default_rng(seed)
    offset = 4321.0
    host_of = lambda r: offset + r * (1 + drift)              # true robot -> host map
    tm = np.arange(0.0, dur, cyc) + host_of(0.003) + 0.37e-3 + rng.uniform(0, jitter, int(dur / cyc))
    n = np.floor(((tm - lat_min) - offset) / (1 + drift) / cyc).astype(np.int64)   # newest tick
    rts = n * cyc
    rx = tm + 30e-6                                            # frame back at the host
    first = np.r_[True, np.diff(rts) != 0]                     # Recorder keeps the first of each sample
    return rx[first], rts[first], host_of


def test_clock_map_ecat_finds_the_bottom_of_the_beat():
    h, r, host_of = _beat_run()
    cm = m.clock_map_ecat(h, r)
    err = cm(r) - host_of(r)                                   # true transport ~ lat_min + 30 us
    assert np.max(np.abs(err - 90e-6)) < 1e-4, (err.min(), err.max())
    assert cm.drift == pytest.approx(40e-6, abs=2e-6)
    beat = 1e-3 / cm.drift
    assert cm.span / beat > 2                                  # 60 s = 2.4 beats


def test_clock_map_ecat_survives_robot_clock_stalls():
    """Robot timestamp frozen for a while, then resuming where it stopped (seen on the Meca500,
    mostly during activation / homing): one offset per segment, the drift of the longest one."""
    h, r, host_of = _beat_run(dur=60.0)
    lost = np.zeros(len(r))
    for t_stall, ms in ((0.5, 479.0), (2.0, 10.0), (2.1, 70.0), (40.0, 4.0)):
        lost[r > t_stall] += ms * 1e-3       # host time spent while the robot clock was frozen
    for t_jump, ms in ((50.0, 16.0), (55.0, 5.0)):   # robot clock jumps forward (17 ms robot gap seen)
        lost[r > t_jump] -= ms * 1e-3
    h = h + lost
    cm = m.clock_map_ecat(h, r)
    err = cm(r) - (host_of(r) + lost)
    assert cm.stalls == 4 and cm.stall_lost == pytest.approx(0.563, abs=2e-3)
    assert cm.jumps == 2 and cm.jump_gained == pytest.approx(0.021, abs=2e-3)
    long_seg = (r > 3.0) & (r < 40.0)                  # 38 s: more than one beat (25 s at 40 ppm)
    assert np.max(np.abs(err[long_seg] - 90e-6)) < 1e-4, (err.min(), err.max())
    # segments shorter than a beat: never early, at most one cycle late (ecat_timing says so)
    assert -1e-4 < np.min(err - 90e-6) and np.max(err - 90e-6) < 1.1e-3
    assert cm.drift == pytest.approx(40e-6, abs=3e-6)


def test_clock_map_ecat_less_than_one_beat_is_flagged():
    h, r, _ = _beat_run(drift=5e-6, dur=20.0)                  # beat 200 s, 0.1 beat recorded
    cm = m.clock_map_ecat(h, r)
    E = m.analyze_ecat_timing(dict(cycle_s=1e-3), np.column_stack([h, r] + [np.full(len(h), np.nan)] * 52), cm)
    assert E["robot_clock_map_ok"] is False


def test_analyze_ecat_timing_columns():
    n = 1000
    ts = np.arange(n) * 1e-3
    B = np.column_stack([ts, ts + 20e-6, ts + 80e-6, ts, np.ones(n), np.full(n, 3.0)])
    B[10, 5], B[10, 3] = -1, np.nan                            # one failed exchange
    B[20, 4] = 0                                               # one cycle without a new sample
    fr = np.full((n, m.NCOL), np.nan)
    fr[:, m.HOST], fr[:, m.ROBOT] = ts + 80e-6, ts
    cm = m.clock_map_ecat(fr[:, m.HOST], fr[:, m.ROBOT])
    E = m.analyze_ecat_timing(dict(cycle_s=1e-3, bus_cycles=B, expected_wkc=3, byte_order="le"), fr, cm)
    assert E["failed_exchanges"] == 1 and E["bad_wkc"] == 0
    assert E["txrx_roundtrip_ms"]["median"] == pytest.approx(0.06, abs=1e-6)
    assert E["fresh_pct"] == pytest.approx(100 * 998 / 999)


# --------------------------------------------------------------------------- references
def test_replay_ref_is_the_python_chain():
    args = dict(multi_joints=[1, 6], replay_dur=2.0, ref_chunk=4, ref_noise_deg=0.02, ref_jitter_ms=3.0,
                ref_chunk_offset_deg=0.1, ref_smooth=5, ref_seed=3, vel_limit_pct=20)
    R = eb.replay_ref(args)
    a = eb._namespace(args)
    js = [0, 5]
    t, qc, src = m.make_reference(a, js, np.array(m.JOINT_VMAX) * 0.2)
    q, rel = m.policy_effects(t, qc, a, js)
    qs, rs, h = m.smooth_actions(q, rel, 5, js)
    assert R["source"] == src == "synthetic" and R["h_smooth"] == h == 2 and R["joints"] == js
    for k, v in dict(t=t, q_clean=qc, q=q, release=rel, q_cmd=qs, rel_cmd=rs).items():
        np.testing.assert_array_equal(np.array(R[k]), v)


def test_soak_ref_phases():
    R = eb.soak_ref(dict(ref_seed=7))
    np.testing.assert_array_equal(np.array(R["ph"]), np.random.default_rng(7).uniform(0, 2 * np.pi, (6, 3)))


# --------------------------------------------------------------------------- npy fixtures (shared with cargo test)
def test_npy_fixtures():
    a = np.load(os.path.join(FIXTURES, "f8_2x3.npy"))
    assert a.dtype == np.float64 and a.shape == (2, 3)
    np.testing.assert_array_equal(a, np.array([[0.0, 1.5, -2.0], [np.nan, np.inf, 1e-300]]))
    assert np.load(os.path.join(FIXTURES, "f8_0x9.npy")).shape == (0, 9)
    np.testing.assert_array_equal(np.load(os.path.join(FIXTURES, "f8_4.npy")), [1.0, 2.0, 3.0, 4.0])


# --------------------------------------------------------------------------- round trip of a run
def _encode(o, folder, frames, n_arr):
    """Python objects -> the JSON tree meca_ecat writes (mirror of ecat/src/out.rs)."""
    if isinstance(o, np.ndarray):
        o = np.asarray(o, float)
        if o.ndim == 2 and o.shape[1] == m.NCOL and len(o):
            i0 = int(np.searchsorted(frames[:, m.ROBOT], o[0, m.ROBOT]))
            if np.array_equal(frames[i0:i0 + len(o)], o, equal_nan=True):
                return {"$frames": [i0, i0 + len(o)]}
        if o.ndim >= 2 or o.size > 32:
            rel = f"npy/a{n_arr[0]:04d}.npy"
            n_arr[0] += 1
            np.save(os.path.join(folder, rel), o)
            return {"$npy": rel}
        return {"$arr": [None if not math.isfinite(x) else x for x in o.tolist()]}
    if isinstance(o, dict):
        return {str(k): _encode(v, folder, frames, n_arr) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_encode(v, folder, frames, n_arr) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if math.isfinite(x) else {"$f": "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")}
    return o


def _to_ecat(src, dst, ecat=None):
    with gzip.open(os.path.join(src, "raw.pkl.gz"), "rb") as f:
        D = pickle.load(f)
    os.makedirs(os.path.join(dst, "npy"), exist_ok=True)
    frames = D["frames"]
    np.save(os.path.join(dst, "npy", "frames.npy"), frames)
    n = [0]
    raw = dict(D["raw"])
    if ecat is not None:
        raw["_ecat"] = ecat
    J = dict(format=eb.FORMAT, args=_encode(D["args"], dst, frames, n), raw=_encode(raw, dst, frames, n),
             frames={"$npy": "npy/frames.npy"},
             send_logs=[_encode(np.asarray(c), dst, frames, n) for c in D["send_logs"]])
    with open(os.path.join(dst, "ecat_run.json"), "w", encoding="utf-8") as f:
        json.dump(J, f)
    return D


def _compare(a, b, path=""):
    if isinstance(a, dict):
        for k in a:
            if k in ("config", "robot_info", "ecat_timing"):
                continue
            assert k in b, f"{path}/{k} missing"
            _compare(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a} != {b}"


@pytest.fixture(scope="module")
def sim_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("sim") / "run"
    cmd = [sys.executable, SCRIPT, "--sim", "--out", str(out), "--vel-limit-pct", "20", "--repeats", "2",
           "--max-excursion", "20", "--tests", "step,sine,servo,servomulti,replay,soak", "--freqs", "1,3",
           "--servo-kps", "10", "--servo-freqs", "1", "--servomulti-freqs", "1", "--multi-joints", "1,6",
           "--replay-dur", "2", "--replay-modes", "vel,pos", "--ref-noise-deg", "0.02", "--soak-dur", "2",
           "--soak-loads", "idle"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert "failed" not in p.stdout, p.stdout[-3000:]
    return str(out)


def test_import_round_trip_reproduces_the_summary(sim_run, tmp_path):
    dst = str(tmp_path / "meca_latency_ecat_sim_20990101_000000_J6")
    _to_ecat(sim_run, dst)
    S = eb.import_run(dst)
    with open(os.path.join(sim_run, "summary.json"), encoding="utf-8") as f:
        ref = json.load(f)
    with open(os.path.join(dst, "summary.json"), encoding="utf-8") as f:
        new = json.load(f)
    _compare(ref, new)
    assert S is not None and "soak" in S and "replay" in S


def test_import_with_ecat_block(sim_run, tmp_path, capsys):
    dst = str(tmp_path / "ecat")
    with gzip.open(os.path.join(sim_run, "raw.pkl.gz"), "rb") as f:
        fr = pickle.load(f)["frames"]
    B = np.column_stack([fr[:, m.HOST] - 1e-4, fr[:, m.HOST] - 5e-5, fr[:, m.HOST], fr[:, m.ROBOT],
                         np.ones(len(fr)), np.full(len(fr), 3.0)])
    _to_ecat(sim_run, dst, ecat=dict(cycle_s=1e-3, bus_cycles=B, expected_wkc=3, byte_order="le",
                                     byte_order_confirmed=True, overruns=0, ring_drops=0, sim=True))
    S = eb.import_run(dst)
    out = capsys.readouterr().out
    assert "analysis of" not in out, out[-3000:]
    assert S["ecat_timing"]["cycles"] == len(fr) and "ETHERCAT BUS" in out
    assert S["config"]["transport"] == "ethercat"
    with open(os.path.join(dst, "summary.json"), encoding="utf-8") as f:
        assert json.load(f)["ecat_timing"]["failed_exchanges"] == 0


def test_quick_model_request(sim_run, tmp_path):
    with gzip.open(os.path.join(sim_run, "raw.pkl.gz"), "rb") as f:
        D = pickle.load(f)
    frames = D["frames"]
    d = tmp_path / "tmp"
    os.makedirs(d / "npy")
    np.save(d / "frames_quick.npy", frames)
    n = [0]
    req = dict(frames={"$npy": "frames_quick.npy"}, step=_encode(D["raw"]["step"], str(d), frames, n))
    with open(d / "quick_req.json", "w", encoding="utf-8") as f:
        json.dump(req, f)
    q = eb.quick(str(d / "quick_req.json"))
    assert abs(q["T_ms"] - 20.0) < 1.0                        # simulator law at SetJointVelLimit 20 %
    assert q["measured_acc"] > 0 and q["tau_ms"] is not None
    p = subprocess.run([sys.executable, os.path.join(ROOT, "ecat_bridge.py"), "quick-model", str(d / "quick_req.json")],
                       capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(p.stdout.strip().splitlines()[-1])["T_ms"] == pytest.approx(q["T_ms"], abs=1e-3)


def test_decode_markers(tmp_path):
    fr = np.arange(3 * m.NCOL, dtype=float).reshape(3, m.NCOL)
    o = eb.decode({"a": {"$arr": [1, None]}, "b": {"$frames": [1, 3]}, "c": {"$f": "-inf"}, "d": [1, {"$f": "nan"}]},
                  str(tmp_path), fr)
    assert np.isnan(o["a"][1]) and o["b"].shape == (2, m.NCOL) and o["c"] == -math.inf and math.isnan(o["d"][1])
