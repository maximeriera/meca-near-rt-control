#!/usr/bin/env python3
"""
ecat_bridge.py
==============
Python side of meca_ecat (the Rust / EtherCAT version of meca_latency_test.py, in ecat/).
meca_ecat drives the robot; everything that is analysis, or that must match the TCP runs
exactly, stays here and reuses meca_latency_test.py:

  import DIR        ecat_run.json + npy/*.npy (written by meca_ecat) -> raw.pkl.gz + raw.npz, the
                    v2 format of meca_latency_test.py, then the same analysis and plots
                    (summary.json, PNG). --no-analyze: conversion only.
  quick-model REQ   quick_accel + quick_model on the step test (EtherCAT clock map over every frame
                    recorded so far): the feedforward advance of the servo / replay / soak tests.
                    Prints one JSON line.
  replay-ref ARGS OUT   make_reference + policy_effects + smooth_actions: the same actions as a TCP
                    run with the same options and --ref-seed.
  soak-ref ARGS OUT     the random phases of the soak motion (same as a TCP run).

meca_ecat calls these between two tests, never inside its 1 ms cycle.

Format of ecat_run.json (paths relative to the file):
  {"format": "meca_ecat/1", "args": {...python option names...}, "raw": {...},
   "frames": {"$npy": "npy/frames.npy"}, "send_logs": [{"$npy": ...}, ...]}
Inside, {"$npy": path} is a float64 .npy file, {"$arr": [...]} a small array (null = NaN),
{"$frames": [i0, i1]} the rows i0:i1 of the frames array, {"$f": "nan" | "inf" | "-inf"} a
non-finite number. Plain JSON lists stay lists (joint indices, flags).

Usage
  python ecat_bridge.py import meca_latency_ecat_20261127_101500_J6
  python ecat_bridge.py import RUN --no-analyze
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meca_latency_test as m  # noqa: E402

FORMAT = "meca_ecat/1"
_NONFINITE = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def decode(o, base, frames=None):
    """JSON tree written by meca_ecat -> the Python objects meca_latency_test.py works with."""
    if isinstance(o, dict):
        if len(o) == 1:
            (k, v), = o.items()
            if k == "$npy":
                return np.load(os.path.join(base, v), allow_pickle=False).astype(float, copy=False)
            if k == "$arr":
                return np.array(_nan_lists(v), dtype=float)
            if k == "$frames":
                if frames is None:
                    raise ValueError("$frames reference without a frames array")
                i0, i1 = int(v[0]), int(v[1])
                return frames[i0:i1].copy()
            if k == "$f":
                return _NONFINITE[v]
        return {k: decode(v, base, frames) for k, v in o.items()}
    if isinstance(o, list):
        return [decode(v, base, frames) for v in o]
    return o


def _nan_lists(v):
    if isinstance(v, list):
        return [_nan_lists(x) for x in v]
    return math.nan if v is None else v


def load_run(folder):
    """Decoded content of ecat_run.json: dict(args, raw, frames, send_logs)."""
    path = os.path.join(folder, "ecat_run.json")
    with open(path, encoding="utf-8") as f:
        J = json.load(f)
    if J.get("format") != FORMAT:
        raise ValueError(f"{path}: format {J.get('format')!r}, expected {FORMAT!r}")
    frames = decode(J["frames"], folder)
    frames = np.asarray(frames, float).reshape(-1, m.NCOL)
    D = dict(args=dict(J["args"]), frames=frames,
             raw=decode(J["raw"], folder, frames),
             send_logs=[np.asarray(decode(x, folder, frames), float) for x in J.get("send_logs", [])])
    return D


def check(D):
    """The invariants the analysis relies on; returns a list of problems (empty = fine)."""
    bad = []
    fr = D["frames"]
    if fr.ndim != 2 or fr.shape[1] != m.NCOL:
        bad.append(f"frames shape {fr.shape}, expected (N, {m.NCOL})")
    elif len(fr) > 1:
        r = fr[:, m.ROBOT]
        if not np.all(np.diff(r[np.isfinite(r)]) > 0):
            bad.append("robot timestamps are not strictly increasing")
        j = int(D["args"].get("joint", 6)) - 1
        for k, kind in enumerate((m.K_TPOS, m.K_TVEL, m.K_APOS, m.K_AVEL)):
            a, b = fr[:, 2 + k], fr[:, m.col(kind, j)]
            if not np.array_equal(a, b, equal_nan=True):
                bad.append(f"column {2 + k} is not a copy of column {m.col(kind, j)} (tested joint)")
    for i, c in enumerate(D["send_logs"]):
        if c.ndim != 2 or c.shape[1] < 4:
            bad.append(f"send_logs[{i}] shape {c.shape}")
    return bad


def import_run(folder, analyze=True, show=False):
    """ecat_run.json -> raw.pkl.gz + raw.npz (meca_latency_test v2 format) [-> analysis]."""
    D = load_run(folder)
    problems = check(D)
    for p in problems:
        print(f"WARNING: {p}")
    args = D["args"]
    args.setdefault("transport", "ethercat")
    arrays = {"frames": D["frames"]}
    for i, c in enumerate(D["send_logs"]):
        arrays[f"cmd_{i:03d}"] = c
    np.savez_compressed(os.path.join(folder, "raw.npz"), **arrays)
    with gzip.open(os.path.join(folder, "raw.pkl.gz"), "wb", compresslevel=3) as f:
        pickle.dump(dict(version=2, raw=D["raw"], frames=D["frames"], send_logs=D["send_logs"], args=args), f)
    print(f"{folder}: {len(D['frames'])} frames, {len(D['send_logs'])} command logs, tests: "
          f"{', '.join(k for k in D['raw'] if not k.startswith('_')) or 'none'}")
    if analyze:
        if len(D["frames"]) < 10:
            print("No data recorded: nothing to analyse.")
            return None
        return m.reanalyze(folder, show=show)
    return None


def _namespace(args_json):
    """meca_latency_test options with the defaults of parse_args, overridden by args_json."""
    a = argparse.Namespace(**{**vars(m.parse_args([])), **(args_json or {})})
    m.apply_joint_vmax(a)
    return a


def quick(req_path):
    """quick_accel + quick_model of a step test, with the EtherCAT clock map of every frame so far."""
    base = os.path.dirname(os.path.abspath(req_path))
    with open(req_path, encoding="utf-8") as f:
        R = json.load(f)
    frames = np.asarray(decode(R["frames"], base), float).reshape(-1, m.NCOL)
    step = decode(R["step"], base, frames)
    cm = m.clock_map_ecat(frames[:, m.HOST], frames[:, m.ROBOT]) if len(frames) > 10 else None
    acc, lag = m.quick_accel(step, cm)
    qm = m.quick_model(step, cm)
    return dict(measured_acc=acc, servo_lag_ms=lag,
                tau_ms=(qm or {}).get("tau_ms"), T_ms=(qm or {}).get("T_ms"))


def replay_ref(args_json):
    a = _namespace(args_json)
    js = sorted({j - 1 for j in a.multi_joints})
    vlim_all = np.array(m.JOINT_VMAX) * a.vel_limit_pct / 100
    t, q_clean, src = m.make_reference(a, js, vlim_all)
    q, release = m.policy_effects(t, q_clean, a, js)
    q_cmd, rel_cmd, h = m.smooth_actions(q, release, a.ref_smooth, js)
    return dict(source=src, h_smooth=int(h), joints=js, t=t.tolist(), q_clean=q_clean.tolist(), q=q.tolist(),
                release=np.asarray(release, float).tolist(), q_cmd=np.asarray(q_cmd, float).tolist(),
                rel_cmd=np.asarray(rel_cmd, float).tolist())


def soak_ref(args_json):
    a = _namespace(args_json)
    freqs = [0.2, 0.45, 0.9]
    ph = np.random.default_rng(a.ref_seed).uniform(0, 2 * np.pi, (m.NJ, len(freqs)))
    return dict(freqs=freqs, ph=ph.tolist())


def _read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _write_json(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("import", help="ecat_run.json -> raw.pkl.gz + analysis")
    s.add_argument("folder")
    s.add_argument("--no-analyze", action="store_true")
    s.add_argument("--show", action="store_true")
    s = sub.add_parser("quick-model", help="quick model of a step test (one JSON line on stdout)")
    s.add_argument("request")
    for name in ("replay-ref", "soak-ref"):
        s = sub.add_parser(name)
        s.add_argument("args_json", help="JSON file of meca_latency_test option values")
        s.add_argument("out_json")
    a = p.parse_args(argv)
    if a.cmd == "import":
        import_run(a.folder, analyze=not a.no_analyze, show=a.show)
    elif a.cmd == "quick-model":
        print(json.dumps(m.jsonable(quick(a.request))))
    elif a.cmd == "replay-ref":
        _write_json(a.out_json, replay_ref(_read_json(a.args_json)))
    elif a.cmd == "soak-ref":
        _write_json(a.out_json, soak_ref(_read_json(a.args_json)))


if __name__ == "__main__":
    main()
