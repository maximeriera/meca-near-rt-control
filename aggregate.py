#!/usr/bin/env python3
"""
aggregate.py
============
Collect the summary.json of meca_latency_test.py runs into CSV tables, one row per
run x item, each row starting with the run's settings. These tables feed the white
paper's tables and charts.

Usage
  python aggregate.py                                        # every meca_latency_* here -> aggregate/
  python aggregate.py --runs "meca_latency_20261125_11*" --out aggregate_6axes

Tables written (only those with data):
  runs.csv           one row per run: settings, robot identity, model (tau, T), servo lag,
                     timing, predicted external loop
  accsweep.csv       SetJointVelLimit -> T          jaccsweep.csv   SetJointAcc -> T
  ampsweep.csv       step size -> T                 multijoint.csv  per joint, alone / together
  periodsweep.csv    command period -> delay        vmaxcheck.csv   measured joint speed limits
  sine.csv           velocity Bode                  impulse.csv     pulse responses
  posstep.csv        MoveJoints sizes               posstream.csv   MoveJoints streams
  posinterrupt.csv   ClearMotion re-targeting       servo_steps.csv / servo_sines.csv
  servomulti.csv     host loop on several joints at once
  replay.csv         policy-like action streams per interface and joint (with the stream settings)
  sensorsweep.csv    host loop through an emulated sensor   sensor_table.csv  predicted crossover
  cartvel.csv        Cartesian velocity interface per axis
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

# section name in summary.json -> how to get its list of rows
SECTIONS = {
    "accsweep": lambda S: S.get("accsweep"),
    "jaccsweep": lambda S: S.get("jaccsweep"),
    "ampsweep": lambda S: S.get("ampsweep"),
    "multijoint": lambda S: S.get("multijoint"),
    "periodsweep": lambda S: S.get("periodsweep"),
    "vmaxcheck": lambda S: S.get("vmaxcheck"),
    "sine": lambda S: S.get("sine"),
    "impulse": lambda S: S.get("impulse"),
    "posstep": lambda S: S.get("posstep"),
    "posstream": lambda S: S.get("posstream"),
    "posinterrupt": lambda S: [S["posinterrupt"]] if S.get("posinterrupt") else None,
    "servo_steps": lambda S: (S.get("servo") or {}).get("steps"),
    "servo_sines": lambda S: (S.get("servo") or {}).get("sines"),
    "servomulti": lambda S: (S.get("servomulti") or {}).get("rows"),
    "replay": lambda S: [{**{k: v for k, v in S["replay"].items() if k not in ("rows", "per_mode")}, **r}
                         for r in S["replay"]["rows"]] if S.get("replay") else None,
    "sensorsweep": lambda S: (S.get("sensorsweep") or {}).get("rows"),
    "sensor_table": lambda S: (S.get("external_loop_prediction") or {}).get("sensor_table"),
    "cartvel": lambda S: S.get("cartvel"),
}


def flatten(d, prefix=""):
    """Nested dicts -> col_sub keys; lists -> JSON text; stats dicts keep median / p95 / max only."""
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            if {"median", "p95", "max"} <= set(v):
                v = {s: v[s] for s in ("median", "p5", "p95", "max") if s in v}
            out.update(flatten(v, key + "_"))
        elif isinstance(v, list):
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def run_settings(folder, S):
    c = S.get("config", {})
    info = S.get("robot_info") or {}
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(folder))
    ident = info.get("identity") or {}
    acc = c.get("joint_acc_pct")
    return {
        "run": os.path.basename(folder.rstrip("/\\")),
        "date": f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]} {m.group(2)[:2]}:{m.group(2)[2:4]}" if m else "",
        "sim": c.get("sim"),
        "joint": c.get("joint"),
        "vel_limit_pct": c.get("vel_limit_pct"),
        "joint_acc_pct": acc if acc is not None else info.get("joint_acc_pct"),
        "period_ms": None if c.get("period") is None else 1e3 * c["period"],
        "tests": c.get("tests"),
        "rt_reader": c.get("rt_reader", "callback"),
        "high_priority": c.get("high_priority", False),
        "robot_model": ident.get("model"),
        "robot_revision": ident.get("revision"),
        "robot_firmware": ident.get("version"),
    }


def run_row(base, S):
    st = S.get("step") or {}
    row = dict(base)
    row.update(flatten({"model": st.get("model")}))
    lag = st.get("servo_lag_target_to_actual_ms") or {}
    row["servo_lag_ms"] = lag.get("median")
    acc = (st.get("target") or {}).get("accel_deg_s2") or {}
    row["target_accel_deg_s2"] = acc.get("median")
    row.update(flatten({"timing": S.get("timing")}))
    row.update(flatten({"pred": S.get("external_loop_prediction")}))
    sv = S.get("servo") or {}
    if sv:
        row["servo_fb_age_ms"] = (sv.get("fb_age_ms") or {}).get("median")
        row["servo_ff_advance_ms"] = sv.get("feedforward_advance_ms")
    return row


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="meca_latency_*", help="glob of run folders (default: meca_latency_*)")
    p.add_argument("--out", default="aggregate", help="output folder for the CSV tables")
    p.add_argument("--include-sim", action="store_true", help="keep --sim runs (dropped by default)")
    a = p.parse_args()

    folders = sorted(f for f in glob.glob(a.runs) if os.path.isfile(os.path.join(f, "summary.json")))
    tables = {"runs": []}
    skipped = []
    for folder in folders:
        try:
            with open(os.path.join(folder, "summary.json"), encoding="utf-8") as fh:
                S = json.load(fh)
        except (OSError, ValueError) as e:
            skipped.append(f"{folder} ({e})")
            continue
        base = run_settings(folder, S)
        if base["sim"] and not a.include_sim:
            continue
        tables["runs"].append(run_row(base, S))
        for name, get in SECTIONS.items():
            for item in get(S) or []:
                tables.setdefault(name, []).append({**base, **flatten(item)})
    os.makedirs(a.out, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            write_csv(os.path.join(a.out, f"{name}.csv"), rows)
            print(f"{name + '.csv':18s} {len(rows):4d} rows")
    print(f"{len(tables['runs'])} runs -> {os.path.abspath(a.out)}")
    for s in skipped:
        print(f"skipped {s}")


if __name__ == "__main__":
    main()
