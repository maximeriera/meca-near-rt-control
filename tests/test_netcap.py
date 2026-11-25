"""netcap.py on a synthetic capture: robot cycles skipped by the robot, delayed on the wire, or
dropped by Python must be told apart."""
import gzip
import os
import pickle
import struct
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import meca_latency_test as m  # noqa: E402
import netcap  # noqa: E402

ROBOT_IP, PC_IP = bytes([192, 168, 0, 100]), bytes([192, 168, 0, 1])
WALL0, PERF0 = 1.7e9, 100.0


def _pkt(src, dst, sport, dport, seq, payload, flags=0x18, win=8192):
    tcp = struct.pack(">HHIIBBHHH", sport, dport, seq, 0, 5 << 4, flags, win, 0, 0) + payload
    ip = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(tcp), 0, 0, 64, 6, 0, src, dst) + tcp
    return b"\x00" * 12 + b"\x08\x00" + ip


def _block(btype, body):
    body += b"\0" * ((4 - len(body) % 4) % 4)
    n = 12 + len(body)
    return struct.pack("<II", btype, n) + body + struct.pack("<I", n)


def write_pcapng(path, pkts):
    out = [_block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1)),
           _block(1, struct.pack("<HHI", 1, 0, 65535) + struct.pack("<HHB3x", 9, 1, 9) + struct.pack("<HH", 0, 0))]
    for t, p in pkts:
        ns = int(round(t * 1e9))
        out.append(_block(6, struct.pack("<IIIII", 0, ns >> 32, ns & 0xFFFFFFFF, len(p), len(p)) + p))
    with open(path, "wb") as f:
        f.write(b"".join(out))


def write_pcap(path, pkts):
    out = [struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)]
    for t, p in pkts:
        us = int(round(t * 1e6))
        out.append(struct.pack("<IIII", us // 1_000_000, us % 1_000_000, len(p), len(p)) + p)
    with open(path, "wb") as f:
        f.write(b"".join(out))


def scenario():
    """2000 robot cycles at 1 ms. Robot never sends cycle 500. Cycles 1000-1009 go 2 per segment.
    Cycles 1500-1519 are held on the wire and arrive with 1520 (a 20 ms wire gap). Cycle 700 is split
    over 2 segments, cycle 800 is retransmitted, the PC advertises one zero window. Python drops cycles
    300-305 (7 ms gap, below the 10 ms threshold) and 1201-1214 (15 ms gap with a regular wire)."""
    rng = np.random.default_rng(0)
    cyc = [i for i in range(2000) if i != 500]
    ts = {i: 5_000_000 + 1000 * i for i in cyc}
    t_wire = {i: WALL0 + 0.001 * i + 0.0001 * rng.random() for i in cyc}
    for i in range(1500, 1520):
        t_wire[i] = t_wire[1520] - 1e-6 * (1520 - i)
    for i in range(1000, 1010, 2):                    # sent together: both arrive with the second
        t_wire[i] = t_wire[i + 1]
    msg = {i: (f"[2200][{ts[i]},1.0,2.0,3.0,4.0,5.0,6.0]\0[2210][{ts[i]},1.0,2.0,3.0,4.0,5.0,6.0]\0"
               f"[2230][{ts[i]}]\0").encode() for i in cyc}
    pkts, seq = [], 1000
    i = 0
    while i < len(cyc):
        c = cyc[i]
        group = [c, cyc[i + 1]] if 1000 <= c < 1010 else [c]
        pay = b"".join(msg[g] for g in group)
        t = max(t_wire[g] for g in group)
        if c == 700:
            pkts.append((t - 2e-5, _pkt(ROBOT_IP, PC_IP, 10001, 50000, seq, pay[:30])))
            pkts.append((t, _pkt(ROBOT_IP, PC_IP, 10001, 50000, seq + 30, pay[30:])))
        else:
            pkts.append((t, _pkt(ROBOT_IP, PC_IP, 10001, 50000, seq, pay)))
        if c == 800:
            pkts.append((t + 2e-4, _pkt(ROBOT_IP, PC_IP, 10001, 50000, seq, pay)))
            pkts.append((t + 3e-4, _pkt(PC_IP, ROBOT_IP, 50000, 10001, 1, b"", flags=0x10, win=0)))
        seq += len(pay)
        i += len(group)
    pkts.sort(key=lambda x: x[0])
    rec = [c for c in cyc if not (300 <= c <= 305 or 1201 <= c <= 1214)]
    frames = np.full((len(rec), m.NCOL), np.nan)
    for k, c in enumerate(rec):
        frames[k, m.HOST] = t_wire[c] - WALL0 + PERF0 + 0.0003 + rng.exponential(0.0001)
        frames[k, m.ROBOT] = ts[c] * 1e-6
    clock =[(WALL0, PERF0), (WALL0 + 10.0, PERF0 + 10.0)]
    return pkts, frames, clock


@pytest.mark.parametrize("fmt", ["pcapng", "pcap"])
def test_netcap_attributes_losses_and_gaps(tmp_path, fmt):
    pkts, frames, clock = scenario()
    cap = tmp_path / f"cap.{fmt}"
    (write_pcapng if fmt == "pcapng" else write_pcap)(cap, pkts)
    run = tmp_path / "run"
    run.mkdir()
    np.savez_compressed(run / "raw.npz", frames=frames)
    with gzip.open(run / "raw.pkl.gz", "wb") as f:
        pickle.dump(dict(raw={"_clock": clock}), f)
    S = netcap.main([str(cap), str(run)])
    assert S["capture"]["retransmissions"] == 1 and S["capture"]["pc_zero_windows"] == 1
    w, p, g = S["wire"], S["python"], S["gaps"]
    assert w["cycles"] == 1999                                   # the split message was reassembled
    assert w["missing_cycles_pct"] == pytest.approx(100 / 2000, rel=0.01)
    assert w["n_interarrival_over_10ms"] == 1                     # the 20 ms hold on the wire
    assert w["cycles_per_segment"]["2"] > 0
    assert p["dropped_pct"] == pytest.approx(100 * 20 / 1999, rel=0.01)
    assert p["extra_delay_ms"]["median"] < 0.3
    assert abs(p["delay_vs_wall_clock_ms"]["median"] - 0.37) < 0.1   # 0.3 ms + the 0.07 ms wire jitter mean
    assert (g["n"], g["on_wire"], g["in_python"]) == (2, 1, 1)
    assert (run / "22_netcap.png").exists() and (run / "netcap.json").exists()


def test_message_parser_legacy_and_json():
    assert netcap._message_code_ts(b"[2200][123456,1.0,2.0]") == (2200, 123456)
    assert netcap._message_code_ts(b'{"code":2200,"data":{"timestampUs":42,"joints":[1,2]}}') == (2200, 42)
    assert netcap._message_code_ts(b"garbage")[0] is None
