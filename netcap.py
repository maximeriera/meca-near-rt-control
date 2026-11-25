#!/usr/bin/env python3
"""
netcap.py
=========
Where do the missing frames and the timing spikes come from: the robot, the network or Python?
Compares a packet capture of the robot's monitoring connection (TCP port 10001) with the frames
that a run of meca_latency_test.py recorded.

Capture with Wireshark's dumpcap while the test runs (`dumpcap -D` lists the interfaces):
  & "C:\\Program Files\\Wireshark\\dumpcap.exe" -i <interface> -f "host 192.168.0.100 and tcp port 10001" -w cap.pcapng
then
  python netcap.py cap.pcapng meca_latency_20261126_101500_J6

What it reports
  wire    one robot cycle = one MX_ST_RT_TARGET_JOINT_POS (2200) message and its timestamp.
          Cycles the robot never sent (gaps between consecutive timestamps), time between cycles
          on the wire, cycles per TCP segment (batching), retransmissions, and zero windows
          advertised by the PC (its receive buffer was full: Python did not read in time).
  python  cycles seen on the wire that the run did not record, and the extra delay from the wire
          to the recorder: host_rx minus the packet time, both clocks tied by the lower envelope
          (as clock_map does with the robot clock), so 0 = the fastest frame of the run.
  gaps    each gap of more than --gap-ms between two recorded frames is put on the wire (the
          packets themselves were that far apart) or on Python (they were not).
Writes netcap.json and 22_netcap.png into the run folder.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import pickle
import re
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meca_latency_test import HOST, ROBOT, clock_map, jsonable, tail  # noqa: E402

RT_TARGET_JOINT_POS = 2200
RT_CYCLE_END = 2230


# --------------------------------------------------------------------------- capture files
def read_capture(path):
    """Yield (time [s, capture clock], linktype, packet bytes) from a pcap or pcapng file.
    A truncated last packet (capture stopped abruptly) is ignored."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 24:
        return
    magic = data[:4]
    if magic == b"\x0a\x0d\x0d\x0a":
        yield from _read_pcapng(data)
        return
    for endian in "<>":
        m = struct.unpack(endian + "I", magic)[0]
        if m in (0xA1B2C3D4, 0xA1B23C4D):
            yield from _read_pcap(data, endian, 1e-9 if m == 0xA1B23C4D else 1e-6)
            return
    raise ValueError(f"{path}: not a pcap / pcapng file")


def _read_pcap(data, e, unit):
    link = struct.unpack(e + "I", data[20:24])[0] & 0xFFFF
    i = 24
    while i + 16 <= len(data):
        sec, frac, incl, _ = struct.unpack(e + "IIII", data[i:i + 16])
        if i + 16 + incl > len(data):
            return
        yield sec + frac * unit, link, data[i + 16:i + 16 + incl]
        i += 16 + incl


def _read_pcapng(data):
    i, e, ifaces = 0, "<", []
    while i + 12 <= len(data):
        btype = struct.unpack(e + "I", data[i:i + 4])[0]
        if btype == 0x0A0D0D0A:  # section header: byte order
            e = "<" if data[i + 8:i + 12] == b"\x4d\x3c\x2b\x1a" else ">"
            ifaces = []
        blen = struct.unpack(e + "I", data[i + 4:i + 8])[0]
        if blen < 12 or i + blen > len(data):
            return
        body = data[i + 8:i + blen - 4]
        if btype == 1:  # interface description: link type and time resolution
            link = struct.unpack(e + "H", body[:2])[0]
            res, j = 1e-6, 8
            while j + 4 <= len(body):
                code, ln = struct.unpack(e + "HH", body[j:j + 4])
                if code == 0:
                    break
                if code == 9 and ln >= 1:
                    v = body[j + 4]
                    res = 2.0 ** -(v & 0x7F) if v & 0x80 else 10.0 ** -v
                j += 4 + ((ln + 3) // 4) * 4
            ifaces.append((link, res))
        elif btype == 6 and ifaces:  # enhanced packet
            iid, hi, lo, cap = struct.unpack(e + "IIII", body[:16])
            link, res = ifaces[iid] if iid < len(ifaces) else ifaces[0]
            yield ((hi << 32) | lo) * res, link, body[20:20 + cap]
        i += blen


def tcp_segment(pkt, link):
    """(src ip, dst ip, sport, dport, seq, flags, window, payload) of an IPv4 TCP packet, else None."""
    if link == 1:  # Ethernet, optional VLAN tags
        off, et = 14, struct.unpack(">H", pkt[12:14])[0] if len(pkt) >= 14 else 0
        while et in (0x8100, 0x88A8) and len(pkt) >= off + 4:
            et, off = struct.unpack(">H", pkt[off + 2:off + 4])[0], off + 4
        if et != 0x0800:
            return None
    elif link in (101, 228):  # raw IP
        off = 0
    elif link == 0:  # BSD loopback
        off = 4
    else:
        return None
    ip = pkt[off:]
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 6:
        return None
    ihl = (ip[0] & 0x0F) * 4
    total = struct.unpack(">H", ip[2:4])[0] or len(ip)   # 0 with segment offload: take what was captured
    tcp = ip[ihl:min(total, len(ip))]
    if len(tcp) < 20:
        return None
    sport, dport, seq = struct.unpack(">HHI", tcp[:8])
    doff = (tcp[12] >> 4) * 4
    flags, win = tcp[13], struct.unpack(">H", tcp[14:16])[0]
    return ip[12:16], ip[16:20], sport, dport, seq, flags, win, tcp[doff:]


def _message_code_ts(msg):
    """(code, timestamp in us or None) of one monitoring message: '[2200][ts,...]' or JSON."""
    if not msg:
        return None, None
    if msg[:1] == b"{":
        try:
            d = json.loads(msg)
        except ValueError:
            return None, None
        code = d.get("code")
        stack = [d]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                for k, v in x.items():
                    if "timestamp" in k.lower() and isinstance(v, (int, float)):
                        return code, int(v)
                    stack.append(v)
            elif isinstance(x, list):
                stack.extend(x)
        return code, None
    m = re.match(rb"\[(\d+)\]\[([^,\]]*)", msg)
    if not m:
        return None, None
    try:
        return int(m.group(1)), int(float(m.group(2)))
    except ValueError:
        return int(m.group(1)), None


def rt_cycles(packets, port=10001):
    """Robot cycles on the wire. Reassembles the robot -> PC byte stream of the monitoring port
    (messages end with NUL) and keeps, per 2200 message, the time of the packet that completed it.
    Returns a dict of arrays: t (capture clock, s), ts_us (robot), seg (packet index), plus counters."""
    t_out, ts_out, seg_out = [], [], []
    buf, nxt, synced = b"", None, False
    n_pkt = n_retx = n_hole = n_zero = n_end = 0
    min_win = None
    k = -1
    for t, link, pkt in packets:
        s = tcp_segment(pkt, link)
        if s is None:
            continue
        _, _, sport, dport, seq, flags, win, pay = s
        if dport == port:  # PC -> robot: acknowledgements, advertised window
            if not flags & 0x07:  # not SYN / FIN / RST
                n_zero += win == 0
                min_win = win if min_win is None else min(min_win, win)
            continue
        if sport != port or not pay:
            continue
        k += 1
        n_pkt += 1
        if nxt is not None:
            d = (seq - nxt) & 0xFFFFFFFF
            if d >= 0x80000000:  # starts before what we have: retransmission
                n_retx += 1
                new = (seq + len(pay) - nxt) & 0xFFFFFFFF
                if new == 0 or new >= 0x80000000:
                    continue
                pay = pay[len(pay) - new:]
            elif d > 0:  # bytes missing from the capture: resync on the next message
                n_hole += 1
                buf, synced = b"", False
        nxt = (seq + len(pay)) & 0xFFFFFFFF
        data = buf + pay
        if not synced and data[:1] in (b"[", b"{"):  # the robot writes whole messages: a boundary
            synced = True
        if not synced:  # capture started (or resumed) inside a message: skip to the next one
            z = data.find(b"\0")
            if z < 0:
                buf = b""
                continue
            data, synced = data[z + 1:], True
        parts = data.split(b"\0")
        buf = parts.pop()
        for msg in parts:
            code, ts = _message_code_ts(msg)
            if code == RT_TARGET_JOINT_POS and ts is not None:
                t_out.append(t)
                ts_out.append(ts)
                seg_out.append(k)
            elif code == RT_CYCLE_END:
                n_end += 1
    ts = np.array(ts_out, np.int64)
    t = np.array(t_out, float)
    seg = np.array(seg_out, np.int64)
    if ts.size:  # one entry per robot cycle (a retransmitted message counts once)
        _, first = np.unique(ts, return_index=True)
        first.sort()
        t, ts, seg = t[first], ts[first], seg[first]
    return dict(t=t, ts_us=ts, seg=seg, packets=n_pkt, retransmissions=n_retx, capture_holes=n_hole,
                pc_zero_windows=int(n_zero), pc_min_window=min_win, cycle_end_messages=n_end)


# --------------------------------------------------------------------------- comparison
def load_run(run_dir):
    """Frames of a run (raw.npz) and its wall-clock <-> perf_counter pairs if recorded."""
    frames = np.load(os.path.join(run_dir, "raw.npz"))["frames"]
    clock = None
    p = os.path.join(run_dir, "raw.pkl.gz")
    if os.path.exists(p):
        try:
            with gzip.open(p, "rb") as f:
                clock = pickle.load(f)["raw"].get("_clock")
        except Exception:  # noqa: BLE001
            clock = None
    return frames, clock


def compare(wire, frames, clock=None, gap_ms=10.0):
    """Wire cycles vs recorded frames. Returns (summary dict, signals for the plot)."""
    h, r_us = frames[:, HOST], np.round(frames[:, ROBOT] * 1e6).astype(np.int64)
    out = dict(capture=dict(packets=wire["packets"], retransmissions=wire["retransmissions"],
                            capture_holes=wire["capture_holes"], pc_zero_windows=wire["pc_zero_windows"],
                            pc_min_window=wire["pc_min_window"], cycle_end_messages=wire["cycle_end_messages"]))
    sel = (wire["ts_us"] >= r_us.min()) & (wire["ts_us"] <= r_us.max())
    wt, wts, wseg = wire["t"][sel], wire["ts_us"][sel], wire["seg"][sel]
    if wts.size < 10:
        out["error"] = "fewer than 10 robot cycles of the capture fall within the run"
        return out, None
    per = float(np.median(np.diff(wts)))
    k = np.maximum(1, np.round(np.diff(wts) / per))
    _, segc = np.unique(wseg, return_counts=True)
    gaps_wire = np.diff(wt)
    out["wire"] = dict(cycles=int(wts.size), robot_period_ms=per * 1e-3,
                       missing_cycles_pct=float(100 * np.sum(k - 1) / np.sum(k)),
                       interarrival_ms=tail(gaps_wire, 1e3),
                       n_interarrival_over_5ms=int(np.sum(gaps_wire > 0.005)),
                       n_interarrival_over_10ms=int(np.sum(gaps_wire > 0.010)),
                       n_interarrival_over_20ms=int(np.sum(gaps_wire > 0.020)),
                       cycles_per_segment={str(n): float(np.mean(segc == n)) for n in (1, 2, 3)}
                       | {"4+": float(np.mean(segc >= 4))})
    common, iw, ir = np.intersect1d(wts, r_us, return_indices=True)
    rec = np.isin(wts, r_us)
    hm, tm = h[ir], wt[iw]
    t0 = float(tm[0])
    env = clock_map(hm, tm - t0)  # wire time -> host clock, lower envelope
    extra = hm - env(tm - t0)
    out["python"] = dict(recorded_cycles=int(common.size), dropped_pct=float(100 * np.mean(~rec)),
                         extra_delay_ms=tail(extra, 1e3),
                         recorded_frame_gap_ms=tail(np.diff(h[np.isin(r_us, wts)]), 1e3))
    if clock and len(clock) >= 2 and clock[-1][0] > clock[0][0]:
        (w0, p0), (w1, p1) = clock[0], clock[-1]
        absd = hm - (p0 + (tm - w0) * (p1 - p0) / (w1 - w0))
        out["python"]["delay_vs_wall_clock_ms"] = tail(absd, 1e3)  # indicative: two different clocks
    # big gaps between recorded frames: were the packets that far apart on the wire?
    hr = h[ir]
    order = np.argsort(hr)
    hr, ts_r, tw_r = hr[order], common[order], tm[order]
    dg = np.diff(hr)
    big = np.nonzero(dg > gap_ms * 1e-3)[0]
    on_wire = 0
    for b in big:
        a, z = np.searchsorted(wts, ts_r[b]), np.searchsorted(wts, ts_r[b + 1])
        span = np.diff(wt[a:z + 1]).max() if z > a else 0.0
        on_wire += span >= 0.5 * dg[b]
    out["gaps"] = dict(threshold_ms=gap_ms, n=int(big.size), on_wire=int(on_wire), in_python=int(big.size - on_wire))
    sig = dict(wire_gap=gaps_wire, rec_gap=np.diff(h), extra=extra, t=tm - t0)
    return out, sig


def plot(sig, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    fig, axs = plt.subplots(1, 3, figsize=(19, 5))
    for i, (x, lab) in enumerate(((sig["wire_gap"], "on the wire (packet capture)"),
                                  (sig["rec_gap"], "recorded by Python (host_rx)"))):
        x = np.sort(x[np.isfinite(x)]) * 1e3
        axs[0].step(x, 1.0 - np.arange(x.size) / x.size, where="post", color=f"C{i}", lw=1.5, label=lab)
    x = np.sort(sig["extra"]) * 1e3
    axs[1].step(np.maximum(x, 1e-3), 1.0 - np.arange(x.size) / x.size, where="post", color="C2", lw=1.5,
                label="wire -> recorder, above the fastest frame")
    sec = np.floor(sig["t"]).astype(int)
    mx = np.full(sec.max() + 1, np.nan)
    np.fmax.at(mx, sec, sig["extra"] * 1e3)
    axs[2].plot(np.arange(mx.size), mx, color="C2", lw=1, label="worst of each second")
    axs[0].set(title="Time between two robot cycles", xlim=(0.3, None))
    axs[1].set(title="Delay added between the wire and the recorder", xlim=(0.01, None))
    for ax in axs[:2]:
        ax.set(xscale="log", yscale="log", xlabel="ms (log)", ylabel="share of samples above (log)")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    axs[2].set(title="Same delay over time: worst of each second", xlabel="s", ylabel="ms", ylim=(0, None))
    for ax in axs:
        ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def print_report(S):
    if "error" in S:
        print(S["error"])
        return
    tl = lambda d: "n/a" if not d else f"{d['median']:6.2f} / {d['p99']:6.2f} / {d['p999']:6.2f} / {d['max']:7.2f}"
    c, w, p, g = S["capture"], S["wire"], S["python"], S["gaps"]
    print(f"CAPTURE  {c['packets']} robot packets, {c['retransmissions']} retransmissions, {c['capture_holes']} holes "
          f"in the capture, {c['pc_zero_windows']} zero windows from the PC (min window {c['pc_min_window']})")
    print(f"WIRE     {w['cycles']} cycles, period {w['robot_period_ms']:.3f} ms, cycles never sent by the robot "
          f"{w['missing_cycles_pct']:.2f} %")
    print(f"         time between cycles median / p99 / p99.9 / max {tl(w['interarrival_ms'])} ms; "
          f"> 5 / 10 / 20 ms: {w['n_interarrival_over_5ms']} / {w['n_interarrival_over_10ms']} / "
          f"{w['n_interarrival_over_20ms']}")
    print("         cycles per TCP segment: " + ", ".join(f"{k}: {100 * v:.1f} %" for k, v in w["cycles_per_segment"].items()))
    print(f"PYTHON   {p['recorded_cycles']} cycles recorded, {p['dropped_pct']:.2f} % of the wire cycles not recorded")
    print(f"         delay wire -> recorder above the fastest frame {tl(p['extra_delay_ms'])} ms")
    if p.get("delay_vs_wall_clock_ms"):
        print(f"         same against the wall clock (indicative)     {tl(p['delay_vs_wall_clock_ms'])} ms")
    print(f"GAPS     {g['n']} gaps > {g['threshold_ms']:g} ms between recorded frames: {g['on_wire']} already on the "
          f"wire, {g['in_python']} added by Python")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("capture", help="pcap / pcapng file of the monitoring connection")
    p.add_argument("run", help="run folder of meca_latency_test.py (raw.npz)")
    p.add_argument("--port", type=int, default=10001, help="robot monitoring port (default 10001)")
    p.add_argument("--gap-ms", type=float, default=10.0, help="gaps between recorded frames to attribute [ms]")
    a = p.parse_args(argv)
    print(f"Reading {a.capture} ...")
    wire = rt_cycles(read_capture(a.capture), a.port)
    if wire["ts_us"].size == 0:
        sys.exit(f"no robot cycle (message {RT_TARGET_JOINT_POS}) found on port {a.port}")
    frames, clock = load_run(a.run)
    S, sig = compare(wire, frames, clock, a.gap_ms)
    S["capture_file"] = os.path.basename(a.capture)
    with open(os.path.join(a.run, "netcap.json"), "w") as f:
        json.dump(jsonable(S), f, indent=2)
    print_report(S)
    if sig is not None:
        run_name = os.path.basename(os.path.normpath(a.run))
        plot(sig, os.path.join(a.run, "22_netcap.png"), f"{os.path.basename(a.capture)} vs {run_name}")
        print(f"-> {os.path.join(a.run, 'netcap.json')}, 22_netcap.png")
    return S


if __name__ == "__main__":
    main()
