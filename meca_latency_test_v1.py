#!/usr/bin/env python3
"""
meca_latency_test.py
====================
Command -> feedback latency and small-signal dynamics of a Mecademic Meca500
driven in joint-velocity mode (MoveJointsVel).

What is measured
----------------
    host send --(net + queue)--> TARGET  (trajectory generator, accel-limited)
                                   '--(servo)--> ACTUAL (encoders)
                                                   '--(RT monitoring, net)--> host receives

Two kinds of latency are reported everywhere:
  * host-seen   : host send time -> host receive time of the first frame that shows
                  the change. This is what a closed loop running on this PC "feels".
                  No clock sync needed.
  * robot clock : robot timestamps mapped onto the host clock (lower envelope of
                  host_rx - robot_ts, linear fit for drift). Lets you split the lag
                  into command transport / trajectory / servo. Biased late by the
                  minimum one-way feedback latency (usually well under 1 ms).

Tests (--tests, comma separated)
  step      velocity steps 0 -> +A -> 0 -> -A -> 0: dead time, rise 10-90 %,
            acceleration, overshoot, settling, target->actual servo lag
  impulse   short velocity pulses ("dirac") of several widths: response shape,
            peak, displacement vs commanded displacement
  sine      sine sweep on the velocity command: Bode gain / phase and the
            equivalent time delay at each frequency
  pingpong  closed-loop relay: reverse the command as soon as the feedback shows
            the motion. Per-event loop latency + limit-cycle frequency
  accsweep  same small step at several SetJointVelLimit values, to check
            "lower max speed = faster acceleration"
  timing    (always) RT frame period / jitter, feedback transport, send jitter

Usage
  python meca_latency_test.py --sim                                  # no robot
  python meca_latency_test.py --ip 192.168.0.100 --joint 6 --vel-limit-pct 20
  python meca_latency_test.py --ip 192.168.0.100 --tests step,sine --freqs 1,2,5,10,20

SAFETY: the robot WILL move. Default joint is 6 (wrist). Every test is bounded by
--max-excursion degrees from the start pose, checked on the feedback, and the
velocity timeout stops the robot if this script dies. Clear the workspace and keep
the E-stop at hand.

Requires: pip install mecademicpy numpy matplotlib   (Python >= 3.11 on Windows
for 1 ms sleep resolution)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np

# Nominal Meca500 R3 joint speed limits at 100 % SetJointVelLimit (deg/s).
# Check them against your robot's manual / firmware version.
JOINT_VMAX = [150.0, 150.0, 180.0, 300.0, 300.0, 500.0]

# Columns of one feedback frame
HOST, ROBOT, TPOS, TVEL, APOS, AVEL = range(6)


class SafetyStop(RuntimeError):
    """The joint left the allowed window or stopped responding."""


def sleep_until(t_target: float) -> None:
    """Precise wait on perf_counter. The last 2 ms spin with sleep(0), which
    releases the GIL so the socket / callback threads are not starved."""
    while True:
        dt = t_target - time.perf_counter()
        if dt <= 0:
            return
        if dt > 0.0025:
            time.sleep(dt - 0.002)
        else:
            time.sleep(0)


# --------------------------------------------------------------------------- #
# Feedback recorder                                                           #
# --------------------------------------------------------------------------- #
class Recorder:
    """Stores one row per RT cycle: host_rx, robot_ts [s], target pos/vel, actual pos/vel."""

    def __init__(self, capacity: int = 1_000_000):
        self.buf = np.full((capacity, 6), np.nan)
        self.n = 0
        self.cond = threading.Condition()
        self._last_robot_t = None

    def add(self, host_t, robot_t, tpos, tvel, apos, avel):
        if robot_t == self._last_robot_t:  # same cycle seen twice
            return
        self._last_robot_t = robot_t
        with self.cond:
            if self.n < len(self.buf):
                self.buf[self.n] = (host_t, robot_t, tpos, tvel, apos, avel)
                self.n += 1
            self.cond.notify_all()

    def wait_new(self, idx: int, timeout: float) -> int:
        with self.cond:
            self.cond.wait_for(lambda: self.n > idx, max(0.0, timeout))
            return self.n

    def latest(self):
        n = self.n
        return self.buf[n - 1].copy() if n else None

    def frames(self, i0: int = 0, i1: int | None = None) -> np.ndarray:
        return self.buf[i0:(self.n if i1 is None else i1)].copy()


# --------------------------------------------------------------------------- #
# Real robot                                                                  #
# --------------------------------------------------------------------------- #
class MecaIO:
    def __init__(self, args, rec: Recorder):
        try:
            import mecademicpy.robot as mdr
        except ImportError:
            sys.exit("mecademicpy is not installed:  pip install mecademicpy")
        self.mdr, self.args, self.rec = mdr, args, rec
        self.j = args.joint - 1
        self.robot = mdr.Robot()
        self.start_joints = None
        self._polling = False

    # -- setup ---------------------------------------------------------------
    def _rt_events(self):
        """Only the RT events we need, to keep 1 ms monitoring light. Falls back to 'all'."""
        try:
            from mecademicpy import mx_robot_def as mx
            names = ("MX_ST_RT_TARGET_JOINT_POS", "MX_ST_RT_TARGET_JOINT_VEL",
                     "MX_ST_RT_JOINT_POS", "MX_ST_RT_JOINT_VEL")
            return [int(getattr(mx.MxRobotStatusCode, n)) for n in names]
        except Exception:
            return ["all"]

    def connect(self):
        a, r, mdr = self.args, self.robot, self.mdr
        # mecademicpy only accepts callbacks before Connect()
        cb = mdr.RobotCallbacks()
        cb.on_end_of_cycle = self._on_cycle
        r.RegisterCallbacks(callbacks=cb, run_callbacks_in_separate_thread=True)
        print(f"Connecting to {a.ip} ...")
        r.Connect(address=a.ip, enable_synchronous_mode=False)
        if r.GetStatusRobot().error_status:
            print("Robot in error -> ResetError() / ResumeMotion()")
            r.ResetError()
            r.ResumeMotion()
        r.ActivateAndHome()
        r.WaitHomed()
        r.SetMonitoringInterval(a.monitor_interval)
        r.SetRealTimeMonitoring(*self._rt_events())
        r.SetVelTimeout(a.vel_timeout)
        r.SetJointVelLimit(a.vel_limit_pct)
        if a.joint_acc_pct is not None:
            r.SetJointAcc(a.joint_acc_pct)
        r.WaitIdle(30)
        time.sleep(0.5)
        if self.rec.n == 0:
            print("No on_end_of_cycle callback received -> polling GetRobotRtData() instead")
            self._polling = True
            threading.Thread(target=self._poll, daemon=True).start()
            time.sleep(0.5)
        if self.rec.n == 0:
            raise RuntimeError("No real-time data received (check SetRealTimeMonitoring / firmware)")
        j = r.GetRtTargetJointPos()
        self.start_joints = [float(x) for x in getattr(j, "data", j)]
        print(f"Start joints: {np.round(self.start_joints, 3)}")

    # -- feedback ------------------------------------------------------------
    def _push(self, d, host_t):
        j = self.j

        def get(field):
            x = getattr(d, field, None)
            if x is None or not getattr(x, "enabled", True):
                return math.nan
            data = getattr(x, "data", None)
            return float(data[j]) if data is not None and len(data) > j else math.nan

        ts = getattr(d.rt_target_joint_pos, "timestamp", None) or d.rt_joint_pos.timestamp
        self.rec.add(host_t, ts * 1e-6,  # robot timestamps are in microseconds
                     get("rt_target_joint_pos"), get("rt_target_joint_vel"),
                     get("rt_joint_pos"), get("rt_joint_vel"))

    def _on_cycle(self, *_, **__):
        if not self._polling:
            t = time.perf_counter()
            self._push(self.robot.GetRobotRtData(), t)

    def _poll(self):
        last = None
        while self._polling:
            d = self.robot.GetRobotRtData()
            ts = d.rt_target_joint_pos.timestamp
            if ts != last:
                last = ts
                self._push(d, time.perf_counter())
            else:
                time.sleep(0.0001)

    # -- commands ------------------------------------------------------------
    def send_vel(self, v: float):
        vel = [0.0] * 6
        vel[self.j] = float(v)
        self.robot.MoveJointsVel(*vel)

    def set_vel_limit(self, pct: float):
        self.robot.SetJointVelLimit(pct)
        self.robot.WaitIdle(10)

    def stop(self):
        self.send_vel(0.0)
        self.robot.WaitIdle(10)

    def return_to_start(self):
        self.stop()
        self.robot.MoveJoints(*self.start_joints)
        self.robot.WaitIdle(60)
        time.sleep(0.2)

    def close(self):
        self._polling = False
        try:
            self.send_vel(0.0)
            self.robot.WaitIdle(5)
        except Exception as e:  # noqa: BLE001
            print(f"while stopping: {e}")
        try:
            self.robot.Disconnect()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# Simulated robot (dry run only)                                              #
# --------------------------------------------------------------------------- #
class SimIO:
    """Crude stand-in so the script and the analysis can be checked without hardware.
    Invented numbers: ~0.5 ms uplink, 1 ms cycle, accel-limited target, 3-cycle +
    4 ms first-order servo, ~0.5 ms downlink, drifting robot clock."""
    CYCLE = 0.001

    def __init__(self, args, rec: Recorder):
        self.args, self.rec = args, rec
        self.j = args.joint - 1
        self.vlim = JOINT_VMAX[self.j] * args.vel_limit_pct / 100
        self.rng_in, self.rng_out = np.random.default_rng(1), np.random.default_rng(2)
        self.lock = threading.Lock()
        self.inbox, self.outbox = [], []
        self.cmd, self.cmd_t = 0.0, -1e9
        self.tp = self.tv = self.ap = self.av = 0.0
        self.dline = [0.0] * 3
        self.start_joints = [0.0] * 6
        self.mon_every = max(1, round(args.monitor_interval / self.CYCLE))
        self._last_in = self._last_out = 0.0
        self.running = False

    def connect(self):
        print("SIMULATION: invented robot model, the numbers say nothing about a real Meca500.")
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.3)

    def _run(self):
        dt, k = self.CYCLE, 0
        t_next = time.perf_counter()
        while self.running:
            t_next += dt
            sleep_until(t_next)
            now = time.perf_counter()
            with self.lock:
                while self.inbox and self.inbox[0][0] <= now:
                    self.cmd = self.inbox.pop(0)[1]
                    self.cmd_t = now
                cmd = self.cmd if now - self.cmd_t <= self.args.vel_timeout else 0.0
                cmd = min(max(cmd, -self.vlim), self.vlim)
                acc = self.args.sim_acc
                self.tv += min(max(cmd - self.tv, -acc * dt), acc * dt)
                self.tp += self.tv * dt
                self.dline.append(self.tv)
                ref = self.dline.pop(0)
                self.av += (ref - self.av) * dt / 0.004
                self.ap += self.av * dt
                k += 1
                if k % self.mon_every == 0:
                    arrive = max(self._last_out, now + 0.0003 + self.rng_out.exponential(0.0003))
                    self._last_out = arrive
                    rt = now * (1 + 40e-6) + 1234.5
                    self.outbox.append((arrive, rt, self.tp, self.tv,
                                        self.ap + self.rng_out.normal(0, 5e-4),
                                        self.av + self.rng_out.normal(0, 0.05)))
                ready = []
                while self.outbox and self.outbox[0][0] <= now:
                    ready.append(self.outbox.pop(0))
            for _, rt, *vals in ready:
                self.rec.add(time.perf_counter(), rt, *vals)

    def send_vel(self, v: float):
        now = time.perf_counter()
        with self.lock:
            arrive = max(self._last_in, now + 0.0004 + self.rng_in.exponential(0.0002))
            self._last_in = arrive
            self.inbox.append((arrive, float(v)))

    def set_vel_limit(self, pct: float):
        with self.lock:
            self.vlim = JOINT_VMAX[self.j] * pct / 100

    def stop(self):
        self.send_vel(0.0)
        time.sleep(0.1 + abs(self.tv) / self.args.sim_acc)

    def return_to_start(self):
        self.stop()
        with self.lock:
            self.tp = self.ap = self.tv = self.av = 0.0
            self.dline = [0.0] * 3
        time.sleep(0.1)

    def close(self):
        self.running = False


# --------------------------------------------------------------------------- #
# Test runner (acquisition only; analysis is done afterwards)                 #
# --------------------------------------------------------------------------- #
def piecewise(segs):
    """segs = [(duration, value), ...] -> f(t)."""
    bounds = np.cumsum([s[0] for s in segs])
    vals = [s[1] for s in segs]
    return lambda t: vals[min(int(np.searchsorted(bounds, t + 1e-9, side="right")), len(vals) - 1)]


class Tester:
    def __init__(self, io, rec: Recorder, args):
        self.io, self.rec, self.args = io, rec, args
        self.j = args.joint - 1
        self.vlim = JOINT_VMAX[self.j] * args.vel_limit_pct / 100
        amp = args.amp if args.amp else 0.5 * self.vlim
        if amp > 0.95 * self.vlim:
            print(f"--amp {amp} is above 95 % of the velocity limit ({self.vlim:.1f} deg/s): clamped")
            amp = 0.95 * self.vlim
        self.amp = amp
        self.send_logs = []          # every streamed command: [scheduled, before, after, v]
        self.start_pos = 0.0
        self.measured_acc = None

    # -- helpers -------------------------------------------------------------
    def home(self):
        self.io.return_to_start()
        fr = self.rec.latest()
        self.start_pos = float(fr[TPOS]) if fr is not None else 0.0

    def guard(self):
        fr = self.rec.latest()
        if fr is None:
            return
        pos = fr[APOS] if np.isfinite(fr[APOS]) else fr[TPOS]
        if abs(pos - self.start_pos) > self.args.max_excursion:
            raise SafetyStop(f"joint {self.j + 1} is {pos - self.start_pos:+.1f} deg from start "
                             f"(limit +/-{self.args.max_excursion})")

    def stream(self, profile, duration: float):
        """Send profile(t) every --period seconds. Returns (cmd log, frames)."""
        P = self.args.period
        n = int(round(duration / P))
        log = np.empty((n, 4))
        i0 = self.rec.n
        t0 = time.perf_counter() + 0.01
        done = 0
        try:
            for k in range(n):
                ts = t0 + k * P
                sleep_until(ts)
                v = profile(k * P)
                t1 = time.perf_counter()
                self.io.send_vel(v)
                t2 = time.perf_counter()
                log[k] = (ts, t1, t2, v)
                done = k + 1
                if k % 5 == 0:
                    self.guard()
        finally:
            self.io.send_vel(0.0)
        time.sleep(0.15)  # trailing feedback
        log = log[:done]
        self.send_logs.append(log)
        return log, self.rec.frames(i0)

    # -- tests ---------------------------------------------------------------
    def run_step(self, amp=None, repeats=None):
        A = amp or self.amp
        hold = min(self.args.step_hold, 0.7 * self.args.max_excursion / A)
        segs = [(0.15, 0.0), (hold, A), (hold, 0.0), (hold, -A), (0.35, 0.0)]
        runs = []
        for r in range(repeats or self.args.repeats):
            print(f"  step run {r + 1}: +/-{A:.1f} deg/s, hold {hold * 1e3:.0f} ms")
            cmd, fr = self.stream(piecewise(segs), sum(s[0] for s in segs))
            runs.append(dict(cmd=cmd, frames=fr))
            self.home()
        res = dict(amp=A, hold=hold, runs=runs)
        if amp is None:  # quick accel estimate used to size the sine test
            self.measured_acc = quick_accel(res)
            if self.measured_acc:
                print(f"  measured target acceleration ~ {self.measured_acc:.0f} deg/s^2")
        return res

    def run_impulse(self):
        P, A = self.args.period, self.amp
        widths = sorted({max(1, round(w * 1e-3 / P)) * P for w in self.args.pulse_widths})
        sched, t = [], 0.1
        for w in widths:
            for r in range(self.args.repeats):
                sched.append((t, w, 1 if r % 2 == 0 else -1))
                t += w + 0.35
        print(f"  {len(sched)} pulses, widths {[round(w * 1e3, 1) for w in widths]} ms, amp {A:.1f} deg/s")

        def prof(tt):
            for t0, w, s in sched:
                if t0 - 1e-9 <= tt < t0 + w - 1e-9:
                    return s * A
            return 0.0

        cmd, fr = self.stream(prof, t + 0.1)
        self.home()
        return dict(amp=A, widths=widths, cmd=cmd, frames=fr)

    def run_sine(self):
        acc_lim = self.args.sine_acc or (0.5 * self.measured_acc if self.measured_acc else 1000.0)
        runs = []
        for f in self.args.freqs:
            w = 2 * math.pi * f
            A = min(self.amp, acc_lim / w, 0.35 * self.args.max_excursion * w)
            settle = max(0.25, 1.0 / f)
            dur = settle + max(4.0 / f, 1.5)
            print(f"  sine {f:5.2f} Hz  A={A:6.2f} deg/s  ({dur:.1f} s)")
            cmd, fr = self.stream(lambda t, A=A, w=w: A * math.sin(w * t), dur)
            runs.append(dict(f=f, A=A, settle=settle, cmd=cmd, frames=fr))
            self.home()
        return dict(acc_lim=acc_lim, runs=runs)

    def run_pingpong(self):
        A, P = self.amp, self.args.period
        thr = self.args.pp_threshold * A
        out = {}
        for name, col, pcol in (("target", TVEL, TPOS), ("actual", AVEL, APOS)):
            fr = self.rec.latest()
            if not (np.isfinite(fr[col]) or np.isfinite(fr[pcol])):
                print(f"  pingpong on {name}: no data, skipped")
                continue
            print(f"  pingpong on {name} velocity: +/-{A:.1f} deg/s, switch at +/-{thr:.1f} deg/s, "
                  f"{self.args.pp_duration:.1f} s")
            events, cmds = [], []
            i0 = idx = self.rec.n
            prev = None

            def send(v):
                t = time.perf_counter()
                self.io.send_vel(v)
                cmds.append((t, v))
                return t

            sign = 1
            t_sent = t_last = send(A)
            t_end = t_sent + self.args.pp_duration
            loops = 0
            try:
                while time.perf_counter() < t_end:
                    n = self.rec.wait_new(idx, t_last + P - time.perf_counter())
                    if n > idx:
                        f_ = self.rec.buf[n - 1].copy()
                        idx = n
                        v = f_[col]
                        if not np.isfinite(v) and prev is not None and f_[ROBOT] > prev[ROBOT]:
                            v = (f_[pcol] - prev[pcol]) / (f_[ROBOT] - prev[ROBOT])
                        prev = f_
                        if sign * v >= thr:
                            events.append((t_sent, f_[HOST], time.perf_counter(), sign))
                            sign = -sign
                            t_sent = t_last = send(sign * A)
                            continue
                        if f_[HOST] - t_sent > 0.5:
                            raise SafetyStop("pingpong: no response within 0.5 s")
                    if time.perf_counter() - t_last >= P:
                        t_last = send(sign * A)
                    loops += 1
                    if loops % 10 == 0:
                        self.guard()
            finally:
                self.io.send_vel(0.0)
            time.sleep(0.15)
            out[name] = dict(amp=A, thr=thr, events=np.array(events), cmd=np.array(cmds),
                             frames=self.rec.frames(i0))
            print(f"    {len(events)} reversals")
            self.home()
        return out

    def run_accsweep(self):
        pcts = self.args.acc_sweep_pcts
        A = 0.8 * JOINT_VMAX[self.j] * min(pcts) / 100  # same amplitude for every setting
        out = []
        try:
            for p in pcts:
                print(f"  SetJointVelLimit({p})")
                self.io.set_vel_limit(p)
                out.append(dict(pct=p, **self.run_step(amp=A, repeats=max(2, self.args.repeats // 2))))
        finally:
            self.io.set_vel_limit(self.args.vel_limit_pct)
        return dict(amp=A, sweeps=out)


# --------------------------------------------------------------------------- #
# Analysis                                                                    #
# --------------------------------------------------------------------------- #
def clock_map(host, robot, win=1.0):
    """Robot timestamp -> host clock: lower envelope of (host_rx - robot_ts), with drift."""
    ok = np.isfinite(host) & np.isfinite(robot)
    h, r = host[ok], robot[ok]
    d = h - r
    r0 = r[0]
    if r[-1] - r0 < 4 * win:
        off = float(d.min())
        return lambda x: np.asarray(x) + off
    b = ((r - r0) // win).astype(int)
    starts = np.r_[0, np.nonzero(np.diff(b))[0] + 1, len(b)]
    xs, ys = [], []
    for a, z in zip(starts[:-1], starts[1:]):
        i = a + int(np.argmin(d[a:z]))
        xs.append(r[i] - r0)
        ys.append(d[i])
    p = np.polyfit(xs, ys, 1)
    p[1] += min(0.0, float(np.min(d - np.polyval(p, r - r0))))
    return lambda x: np.asarray(x) + np.polyval(p, np.asarray(x) - r0)


def prep(fr, cmap):
    d = dict(h=fr[:, HOST], r=fr[:, ROBOT], tp=fr[:, TPOS], tv=fr[:, TVEL],
             ap=fr[:, APOS], av=fr[:, AVEL])
    d["m"] = cmap(d["r"])
    for v, p in (("tv", "tp"), ("av", "ap")):
        if not np.isfinite(d[v]).any() and np.isfinite(d[p]).sum() > 3:
            d[v] = np.gradient(d[p], d["r"])  # derive velocity from position
    return d


def has(d, k):
    return np.isfinite(d[k]).any()


def cross_time(t, y, level, interp=True):
    """First time y >= level. Without interp, the time of the first sample (what the host sees)."""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    hit = y >= level
    if not hit.any():
        return math.nan
    i = int(np.argmax(hit))
    if i == 0 or not interp or y[i] == y[i - 1]:
        return float(t[i])
    return float(t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


def edge_metrics(t, y, te, tend, v0, v1, noise, interp=True):
    s = v1 - v0
    w = (t >= te) & (t <= tend)
    if s == 0 or w.sum() < 3:
        return None
    tr, yw = t[w] - te, y[w]
    yn = (yw - v0) / s
    thr = max(0.01, 4 * noise / abs(s))  # first visible departure, not part of the ramp
    c = lambda lvl: cross_time(tr, yn, lvl, interp)
    dead, t10, t20, t50, t80, t90 = c(thr), c(0.1), c(0.2), c(0.5), c(0.8), c(0.9)
    acc = math.nan
    if np.isfinite(t20) and np.isfinite(t80) and t80 > t20:
        m = (tr >= t20) & (tr <= t80) & np.isfinite(yw)
        acc = abs(np.polyfit(tr[m], yw[m], 1)[0]) if m.sum() >= 3 else abs(0.6 * s / (t80 - t20))
    fin = np.isfinite(yn)
    over = float(np.max(yn[fin]) - 1) if fin.any() else math.nan
    tol = max(0.02, 4 * noise / abs(s))
    bad = fin & (np.abs(yn - 1) > tol)
    settle = math.nan
    if np.isfinite(t90) and fin.any() and not bad[np.nonzero(fin)[0][-1]]:
        settle = float(tr[bad].max()) if bad.any() else 0.0
    return dict(dead=dead, t10=t10, t50=t50, t90=t90, rise=t90 - t10, acc=acc,
                overshoot=over, settle=settle)


def cmd_edges(c):
    v = c[:, 3]
    k = np.nonzero(np.diff(v) != 0)[0] + 1
    return [(float(c[i, 1]), float(v[i - 1]), float(v[i])) for i in k]


def stats(x, scale=1.0):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float) * scale
    if x.size == 0:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), mean=float(x.mean()),
                p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()))


def analyze_step(res, cmap):
    rows = []
    for run in res["runs"]:
        c, d = run["cmd"], prep(run["frames"], cmap)
        run["d"] = d
        edges = cmd_edges(c)
        ends = [e[0] for e in edges[1:]] + [c[-1, 1] + 0.12]
        for (te, v0, v1), tend in zip(edges, ends):
            row = dict(te=te, v0=v0, v1=v1, kind="accel" if abs(v1) > abs(v0) else "decel")
            for sig in ("tv", "av"):
                if not has(d, sig):
                    continue
                y = d[sig]
                pre = (d["m"] > te - 0.08) & (d["m"] < te) & np.isfinite(y)
                noise = float(np.std(y[pre] - v0)) if pre.sum() > 5 else 0.0
                row[sig] = edge_metrics(d["m"], y, te, tend, v0, v1, noise, True)
                hs = edge_metrics(d["h"], y, te, tend, v0, v1, noise, False)
                row[sig + "_host"] = hs["dead"] if hs else math.nan
            if row.get("tv") and row.get("av"):
                row["servo_lag"] = row["av"]["t50"] - row["tv"]["t50"]
            rows.append(row)

    def col(sig, key):
        return [r[sig][key] for r in rows if r.get(sig)]

    summ = dict(amp=res["amp"], n_edges=len(rows))
    for sig, name in (("tv", "target"), ("av", "actual")):
        if not any(r.get(sig) for r in rows):
            continue
        summ[name] = {
            "dead_ms": stats(col(sig, "dead"), 1e3),
            "dead_host_seen_ms": stats([r.get(sig + "_host") for r in rows], 1e3),
            "t50_ms": stats(col(sig, "t50"), 1e3),
            "rise_10_90_ms": stats(col(sig, "rise"), 1e3),
            "accel_deg_s2": stats(col(sig, "acc")),
            "overshoot_pct": stats(col(sig, "overshoot"), 100),
            "settle_ms": stats(col(sig, "settle"), 1e3),
        }
    summ["servo_lag_target_to_actual_ms"] = stats([r.get("servo_lag") for r in rows], 1e3)
    res["rows"] = rows
    return summ


def quick_accel(res):
    try:
        allf = np.vstack([r["frames"] for r in res["runs"]])
        cm = clock_map(allf[:, HOST], allf[:, ROBOT])
        s = analyze_step(dict(res), cm)
        return s["target"]["accel_deg_s2"]["median"]
    except Exception:  # noqa: BLE001
        return None


def analyze_impulse(res, cmap):
    c, d = res["cmd"], prep(res["frames"], cmap)
    A, P = res["amp"], float(np.median(np.diff(c[:, 0])))  # scheduled period
    v = c[:, 3]
    starts = np.nonzero((v[1:] != 0) & (v[:-1] == 0))[0] + 1
    grid = np.arange(-0.01, 0.2, 0.0005)
    per = {}
    for i in starts:
        j = i
        while j < len(v) and v[j] != 0:
            j += 1
        ts, s = c[i, 1], np.sign(v[i])
        te = c[j, 1] if j < len(v) else c[-1, 1]
        w_nom = round((j - i) * P, 4)
        win = (d["m"] >= ts - 0.03) & (d["m"] <= ts + 0.33)
        row = dict(width=w_nom, area_cmd=A * (te - ts))
        for sig, psig in (("tv", "tp"), ("av", "ap")):
            if not has(d, sig):
                continue
            t, y = d["m"][win] - ts, s * d[sig][win]
            pre = (t < 0) & np.isfinite(y)
            noise = float(np.std(y[pre])) if pre.sum() > 5 else 0.0
            thr = max(0.02 * A, 4 * noise)
            row[sig + "_dead"] = cross_time(t, y, thr)
            row[sig + "_dead_host"] = cross_time(d["h"][win] - ts, y, thr, interp=False)
            row[sig + "_peak"] = float(np.nanmax(y)) if np.isfinite(y).any() else math.nan
            row[sig + "_tpeak"] = float(t[np.nanargmax(y)]) if np.isfinite(y).any() else math.nan
            p = s * d[psig][win]
            if np.isfinite(p).sum() > 5:
                row[sig + "_disp"] = float(np.interp(0.3, t, p) - np.interp(-0.005, t, p))
            row[sig + "_curve"] = np.interp(grid, t, y, left=np.nan, right=np.nan) / A
        per.setdefault(w_nom, []).append(row)
    out = []
    for w in sorted(per):
        rs = per[w]
        o = dict(width_ms=w * 1e3, n=len(rs), area_cmd=float(np.mean([r["area_cmd"] for r in rs])))
        for sig in ("tv", "av"):
            if sig + "_peak" not in rs[0]:
                continue
            for k in ("dead", "dead_host", "tpeak"):
                o[f"{sig}_{k}_ms"] = float(np.nanmedian([r[f"{sig}_{k}"] for r in rs])) * 1e3
            o[f"{sig}_peak_ratio"] = float(np.nanmedian([r[sig + "_peak"] for r in rs])) / A
            if sig + "_disp" in rs[0]:
                o[f"{sig}_disp_ratio"] = float(np.nanmedian([r[sig + "_disp"] for r in rs])) / o["area_cmd"]
            o[sig + "_curve"] = np.nanmean([r[sig + "_curve"] for r in rs], axis=0)
        out.append(o)
    res["table"], res["grid"] = out, grid
    return [{k: v for k, v in o.items() if not k.endswith("_curve")} for o in out]


def sine_fit(t, y, w):
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 10:
        return math.nan, math.nan, math.nan
    X = np.column_stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    R, ph = math.hypot(coef[0], coef[1]), math.atan2(coef[1], coef[0])
    resid = y - X @ coef
    return R, ph, float(np.sqrt(np.mean(resid ** 2)) / (R / math.sqrt(2))) if R > 0 else math.nan


def analyze_sine(res, cmap):
    rows = []
    for run in res["runs"]:
        f, c = run["f"], run["cmd"]
        d = prep(run["frames"], cmap)
        run["d"] = d
        w = 2 * math.pi * f
        t0 = c[0, 1]
        ta, tb = t0 + run["settle"], c[-1, 1]
        m = (c[:, 1] >= ta)
        Au, phu, _ = sine_fit(c[m, 1] - t0, c[m, 3], w)
        row = dict(f=f, A=run["A"], A_cmd_fit=Au)
        for sig in ("tv", "av"):
            if not has(d, sig):
                continue
            for clk, key in (("m", ""), ("h", "_host")):
                mm = (d[clk] >= ta) & (d[clk] <= tb)
                Ay, phy, nl = sine_fit(d[clk][mm] - t0, d[sig][mm], w)
                lag = (phu - phy) % (2 * math.pi)
                if lag > 1.75 * math.pi:
                    lag -= 2 * math.pi
                row[f"{sig}{key}_gain"] = Ay / Au
                row[f"{sig}{key}_phase_deg"] = -math.degrees(lag)
                row[f"{sig}{key}_delay_ms"] = lag / w * 1e3
                if not key:
                    row[f"{sig}_nonlin"] = nl
        rows.append(row)
    res["rows"] = rows
    return rows


def analyze_pingpong(res, cmap):
    summ = {}
    for name, r in res.items():
        ev, d = r["events"], prep(r["frames"], cmap)
        r["d"] = d
        if len(ev) < 3:
            summ[name] = dict(n_events=int(len(ev)))
            continue
        ev = ev[1:]  # first one starts from rest
        detect = ev[:, 1] - ev[:, 0]
        react = ev[:, 2] - ev[:, 1]
        eff_h, eff_r = [], []
        tv, h, m = d["tv"], d["h"], d["m"]
        dtv = np.diff(tv)
        for ts, _, _, s in ev:
            k = np.nonzero((h[1:] > ts) & (s * dtv > 1e-6))[0]
            if len(k):
                eff_h.append(h[k[0] + 1] - ts)
                eff_r.append(m[k[0] + 1] - ts)
        sent = ev[:, 0]
        period = sent[2:] - sent[:-2] if len(sent) > 3 else np.array([])
        summ[name] = dict(
            n_events=int(len(ev)),
            detect_latency_ms=stats(detect, 1e3),
            effect_latency_host_seen_ms=stats(eff_h, 1e3),
            effect_latency_robot_clock_ms=stats(eff_r, 1e3),
            host_reaction_overhead_ms=stats(react, 1e3),
            limit_cycle_hz=float(1 / np.median(period)) if len(period) else None)
        r["detect"], r["eff_h"], r["eff_r"] = detect, np.array(eff_h), np.array(eff_r)
    return summ


def analyze_timing(frames, send_logs, cmap):
    h, r = frames[:, HOST], frames[:, ROBOT]
    m = cmap(r)
    out = dict(robot_frame_period_ms=stats(np.diff(r), 1e3),
               host_rx_interval_ms=stats(np.diff(h), 1e3),
               feedback_excess_latency_ms=stats(h - m, 1e3))
    if send_logs:
        c = np.vstack(send_logs)
        out["send_lateness_ms"] = stats(c[:, 1] - c[:, 0], 1e3)
        out["MoveJointsVel_call_ms"] = stats(c[:, 2] - c[:, 1], 1e3)
    return out, dict(rdt=np.diff(r), hdt=np.diff(h), lat=h - m,
                     late=(c[:, 1] - c[:, 0]) if send_logs else None,
                     call=(c[:, 2] - c[:, 1]) if send_logs else None)


def analyze_accsweep(res, cmap):
    rows = []
    for sw in res["sweeps"]:
        s = analyze_step(sw, cmap)
        row = dict(pct=sw["pct"])
        for name in ("target", "actual"):
            if name in s:
                for k in ("accel_deg_s2", "rise_10_90_ms", "dead_ms", "t50_ms"):
                    row[f"{name}_{k}"] = s[name][k]["median"] if s[name][k] else math.nan
        rows.append(row)
    res["rows"] = rows
    return rows


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _save(fig, out, name):
    p = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    print(f"  saved {p}")


def plot_step(res, out, plt):
    run = res["runs"][0]
    c, d = run["cmd"], run["d"]
    te0 = cmd_edges(c)[0][0]
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)
    ax = fig.add_subplot(gs[0, :])
    ax.step((c[:, 1] - te0) * 1e3, c[:, 3], where="post", color="k", lw=1, label="command (sent)")
    if has(d, "tv"):
        ax.plot((d["m"] - te0) * 1e3, d["tv"], color="C0", label="target vel")
    if has(d, "av"):
        ax.plot((d["m"] - te0) * 1e3, d["av"], color="C1", label="actual vel")
    ax.set(xlabel="time from first edge [ms] (robot clock mapped to host)",
           ylabel="deg/s", title=f"Velocity step response, run 1 (A = {res['amp']:.1f} deg/s)")
    ax.legend(); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 0])
    row = res["rows"][0]
    span = 60
    if row.get("tv") and np.isfinite(row["tv"]["t90"]):
        span = max(40, 2.5 * row["tv"]["t90"] * 1e3)
    if row.get("av") and np.isfinite(row["av"]["t90"]):
        span = max(span, 2 * row["av"]["t90"] * 1e3)
    sel = lambda t: ((t - te0) * 1e3 > -5) & ((t - te0) * 1e3 < span)
    m = sel(c[:, 1])
    ax.step((c[m, 1] - te0) * 1e3, c[m, 3], where="post", color="k", lw=1, label="command")
    for sig, col, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
        if has(d, sig):
            mm = sel(d["m"])
            ax.plot((d["m"][mm] - te0) * 1e3, d[sig][mm], ".-", color=col, ms=3, label=f"{lab} (robot clk)")
            mh = sel(d["h"])
            ax.plot((d["h"][mh] - te0) * 1e3, d[sig][mh], "x", color=col, ms=4, alpha=.6,
                    label=f"{lab} (when host got it)")
            if row.get(sig) and np.isfinite(row[sig]["dead"]):
                ax.axvline(row[sig]["dead"] * 1e3, color=col, ls="--", lw=.8)
    ax.set(xlabel="ms from command edge", ylabel="deg/s", title="Zoom on first edge (dashed = dead time)")
    ax.legend(fontsize=7); ax.grid(alpha=.3)

    ax = fig.add_subplot(gs[1, 1])
    groups, labels = [], []
    for key, lab in (("tv", "cmd->target\nrobot clk"), ("tv_host", "cmd->target\nhost sees"),
                     ("av", "cmd->actual\nrobot clk"), ("av_host", "cmd->actual\nhost sees"),
                     ("servo_lag", "target->actual\n(50 %)")):
        vals = []
        for r in res["rows"]:
            x = r.get(key)
            x = x["dead"] if isinstance(x, dict) else x
            if x is not None and np.isfinite(x):
                vals.append(x * 1e3)
        if vals:
            groups.append(vals); labels.append(lab)
    if groups:
        ax.boxplot(groups, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    ax.set(ylabel="ms", title=f"Dead times over {len(res['rows'])} edges"); ax.grid(alpha=.3)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    _save(fig, out, "1_step_response.png")


def plot_impulse(res, out, plt):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    g = res["grid"] * 1e3
    for i, o in enumerate(res["table"]):
        col = plt.cm.viridis(i / max(1, len(res["table"]) - 1))
        if "av_curve" in o:
            axs[0].plot(g, o["av_curve"], color=col, label=f"{o['width_ms']:.0f} ms actual")
        if "tv_curve" in o:
            axs[0].plot(g, o["tv_curve"], "--", color=col, lw=.8)
    axs[0].set(xlabel="ms from pulse start", ylabel="velocity / A",
               title="Pulse ('dirac') responses (solid actual, dashed target)", xlim=(-5, g[-1]))
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
    w = [o["width_ms"] for o in res["table"]]
    for sig, col, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
        if f"{sig}_peak_ratio" in res["table"][0]:
            axs[1].plot(w, [o[f"{sig}_peak_ratio"] for o in res["table"]], "o-", color=col, label=f"peak vel / A ({lab})")
        if f"{sig}_disp_ratio" in res["table"][0]:
            axs[1].plot(w, [o[f"{sig}_disp_ratio"] for o in res["table"]], "s--", color=col,
                        label=f"displacement / commanded ({lab})")
    axs[1].set_xscale("log")
    axs[1].set(xlabel="pulse width [ms]", ylabel="ratio", title="How much of a short pulse gets through")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3, which="both")
    _save(fig, out, "2_impulse_response.png")


def plot_bode(res, out, plt):
    rows = res["rows"]
    f = np.array([r["f"] for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    series = (("tv", "C0", "-", "target (robot clk)"), ("tv_host", "C0", ":", "target (host sees)"),
              ("av", "C1", "-", "actual (robot clk)"), ("av_host", "C1", ":", "actual (host sees)"))
    for key, col, ls, lab in series:
        if f"{key}_gain" not in rows[0]:
            continue
        g = np.array([r[f"{key}_gain"] for r in rows])
        ph = np.array([r[f"{key}_phase_deg"] for r in rows])
        dl = np.array([r[f"{key}_delay_ms"] for r in rows])
        if "host" not in key:
            axs[0].semilogx(f, 20 * np.log10(g), "o" + ls, color=col, label=lab)
        axs[1].semilogx(f, ph, "o" + ls, color=col, label=lab)
        axs[2].semilogx(f, dl, "o" + ls, color=col, label=lab)
    if "tv_delay_ms" in rows[0]:
        tau = np.nanmedian([r["tv_delay_ms"] for r in rows]) * 1e-3
        ff = np.logspace(np.log10(f.min()), np.log10(f.max()), 100)
        axs[1].semilogx(ff, -360 * ff * tau, "k--", lw=.8, label=f"pure delay {tau * 1e3:.1f} ms")
    axs[0].axhline(-3, color="gray", ls="--", lw=.8)
    axs[0].set(ylabel="gain [dB]", title="Velocity command -> response (sine sweep)")
    axs[1].set(ylabel="phase [deg]")
    axs[2].set(ylabel="equivalent delay [ms]", xlabel="frequency [Hz]")
    for a in axs:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    _save(fig, out, "3_bode.png")


def plot_sine_traces(res, out, plt):
    runs = res["runs"]
    pick = sorted({0, len(runs) // 2, len(runs) - 1})
    fig, axs = plt.subplots(len(pick), 1, figsize=(12, 3.2 * len(pick)))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, pick):
        run, d, c = runs[i], runs[i]["d"], runs[i]["cmd"]
        tb = c[-1, 1]
        ta = tb - 2.0 / run["f"]
        m = c[:, 1] >= ta
        ax.step((c[m, 1] - ta) * 1e3, c[m, 3], where="post", color="k", lw=1, label="command")
        for sig, col, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
            if has(d, sig):
                mm = (d["m"] >= ta) & (d["m"] <= tb)
                ax.plot((d["m"][mm] - ta) * 1e3, d[sig][mm], color=col, label=lab)
        ax.set(title=f"{run['f']} Hz, A = {run['A']:.2f} deg/s (last 2 periods)", xlabel="ms", ylabel="deg/s")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    _save(fig, out, "4_sine_traces.png")


def plot_pingpong(res, out, plt):
    names = [n for n in res if "detect" in res[n]]
    if not names:
        return
    fig, axs = plt.subplots(len(names), 2, figsize=(14, 4 * len(names)), squeeze=False)
    for (a0, a1), n in zip(axs, names):
        r = res[n]
        bins = 30
        a0.hist(r["detect"] * 1e3, bins=bins, alpha=.6, label="send -> host sees threshold")
        if len(r["eff_h"]):
            a0.hist(r["eff_h"] * 1e3, bins=bins, alpha=.6, label="send -> host sees target react")
        if len(r["eff_r"]):
            a0.hist(r["eff_r"] * 1e3, bins=bins, alpha=.6, label="send -> target reacts (robot clk)")
        a0.set(title=f"Ping-pong on {n} velocity", xlabel="ms", ylabel="count")
        a0.legend(fontsize=8); a0.grid(alpha=.3)
        d, c = r["d"], r["cmd"]
        t0 = c[0, 0] + 0.2
        sel = lambda t: (t >= t0) & (t <= t0 + 0.3)
        m = sel(c[:, 0])
        a1.step((c[m, 0] - t0) * 1e3, c[m, 1], where="post", color="k", lw=1, label="command")
        for sig, col, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
            if has(d, sig):
                mm = sel(d["m"])
                a1.plot((d["m"][mm] - t0) * 1e3, d[sig][mm], color=col, label=lab)
        for s in (1, -1):
            a1.axhline(s * r["thr"], color="gray", ls="--", lw=.8)
        a1.set(title="300 ms of the limit cycle", xlabel="ms", ylabel="deg/s")
        a1.legend(fontsize=8); a1.grid(alpha=.3)
    _save(fig, out, "5_pingpong.png")


def plot_timing(td, out, plt):
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    items = (("rdt", "RT frame period (robot clock)"), ("hdt", "Host receive interval"),
             ("lat", "Feedback latency above minimum (host_rx - robot ts)"),
             ("late", "Command send lateness vs schedule"))
    for ax, (k, t) in zip(axs.flat, items):
        x = td.get(k)
        if x is None or len(x) == 0:
            ax.set_visible(False); continue
        x = x[np.isfinite(x)] * 1e3
        hi = np.percentile(x, 99.9)
        ax.hist(np.clip(x, None, hi), bins=80)
        ax.set_yscale("log")
        ax.set(title=t, xlabel="ms (clipped at p99.9)", ylabel="count"); ax.grid(alpha=.3)
    _save(fig, out, "6_timing.png")


def plot_accsweep(res, out, plt):
    rows = res["rows"]
    p = [r["pct"] for r in rows]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, col in (("target", "C0"), ("actual", "C1")):
        if f"{name}_accel_deg_s2" not in rows[0]:
            continue
        axs[0].plot(p, [r[f"{name}_accel_deg_s2"] for r in rows], "o-", color=col, label=name)
        axs[1].plot(p, [r[f"{name}_rise_10_90_ms"] for r in rows], "o-", color=col, label=name)
        axs[2].plot(p, [r[f"{name}_dead_ms"] for r in rows], "o-", color=col, label=f"{name} dead time")
        axs[2].plot(p, [r[f"{name}_t50_ms"] for r in rows], "s--", color=col, label=f"{name} 50 %")
    axs[0].set(title=f"Acceleration (step of {res['amp']:.1f} deg/s)", xlabel="SetJointVelLimit [%]", ylabel="deg/s^2")
    axs[1].set(title="Rise time 10-90 %", xlabel="SetJointVelLimit [%]", ylabel="ms")
    axs[2].set(title="Delays", xlabel="SetJointVelLimit [%]", ylabel="ms")
    for a in axs:
        a.grid(alpha=.3); a.legend(fontsize=8)
    _save(fig, out, "7_accel_vs_vel_limit.png")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fl = lambda s: [float(x) for x in s.split(",") if x.strip()]
    p.add_argument("--ip", default="192.168.0.100")
    p.add_argument("--sim", action="store_true", help="dry run against a fake robot model")
    p.add_argument("--joint", type=int, default=6, choices=range(1, 7))
    p.add_argument("--tests", default="step,impulse,sine,pingpong,accsweep")
    p.add_argument("--period", type=float, default=0.002, help="command streaming period [s]")
    p.add_argument("--monitor-interval", type=float, default=0.001, help="SetMonitoringInterval [s]")
    p.add_argument("--vel-limit-pct", type=float, default=25, help="SetJointVelLimit [%%]")
    p.add_argument("--joint-acc-pct", type=float, default=None, help="SetJointAcc [%%] (default: untouched)")
    p.add_argument("--vel-timeout", type=float, default=0.05, help="SetVelTimeout [s]")
    p.add_argument("--amp", type=float, default=None, help="velocity amplitude [deg/s], default 50 %% of the limit")
    p.add_argument("--max-excursion", type=float, default=45, help="abort beyond this many deg from start")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--step-hold", type=float, default=0.3, help="duration of each step level [s]")
    p.add_argument("--pulse-widths", type=fl, default=fl("1,2,5,10,20,50"), help="ms, rounded to --period")
    p.add_argument("--freqs", type=fl, default=fl("0.5,1,2,3,5,7,10,14,20"))
    p.add_argument("--sine-acc", type=float, default=None,
                   help="cap on A*2*pi*f [deg/s^2]; default 50 %% of the accel measured in the step test")
    p.add_argument("--pp-threshold", type=float, default=0.15, help="ping-pong switch level, fraction of amp")
    p.add_argument("--pp-duration", type=float, default=3.0)
    p.add_argument("--acc-sweep-pcts", type=fl, default=fl("5,10,25,50,100"))
    p.add_argument("--sim-acc", type=float, default=2500.0, help="simulation only: accel [deg/s^2]")
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--show", action="store_true", help="open the plots at the end")
    return p.parse_args()


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def print_summary(S):
    ms = lambda s: "   n/a" if not s else f"{s['median']:6.2f}"
    if "step" in S:
        s = S["step"]
        print(f"\nSTEP  (A = {s['amp']:.1f} deg/s, {s['n_edges']} edges, medians)")
        for n in ("target", "actual"):
            if n in s:
                x = s[n]
                print(f"  {n:7s} dead {ms(x['dead_ms'])} ms | host-seen dead {ms(x['dead_host_seen_ms'])} ms | "
                      f"rise {ms(x['rise_10_90_ms'])} ms | accel {ms(x['accel_deg_s2'])} deg/s2 | "
                      f"overshoot {ms(x['overshoot_pct'])} % | settle {ms(x['settle_ms'])} ms")
        print(f"  servo lag target->actual (50 %): {ms(s['servo_lag_target_to_actual_ms'])} ms")
    if "impulse" in S:
        print("\nIMPULSE  width | dead tgt / act [ms] | peak act/A | disp act/cmd")
        for o in S["impulse"]:
            print(f"  {o['width_ms']:6.1f} ms | {o.get('tv_dead_ms', math.nan):6.2f} / {o.get('av_dead_ms', math.nan):6.2f} "
                  f"| {o.get('av_peak_ratio', math.nan):6.3f} | {o.get('av_disp_ratio', math.nan):6.3f}")
    if "sine" in S:
        print("\nSINE   f [Hz] |  A   | gain tgt  delay tgt | gain act  delay act | host-seen delay act [ms]")
        for r in S["sine"]:
            g = lambda k: r.get(k, math.nan)
            print(f"  {r['f']:6.2f} | {r['A']:5.1f} | {g('tv_gain'):6.3f} {g('tv_delay_ms'):8.2f} | "
                  f"{g('av_gain'):6.3f} {g('av_delay_ms'):8.2f} | {g('av_host_delay_ms'):8.2f}")
    if "pingpong" in S:
        print("\nPING-PONG (medians, ms)")
        for n, x in S["pingpong"].items():
            if "detect_latency_ms" in x:
                print(f"  on {n:6s}: send->host sees threshold {ms(x['detect_latency_ms'])} | "
                      f"send->target reacts host-seen {ms(x['effect_latency_host_seen_ms'])} / robot clk "
                      f"{ms(x['effect_latency_robot_clock_ms'])} | host reaction {ms(x['host_reaction_overhead_ms'])} "
                      f"| limit cycle {x['limit_cycle_hz'] or float('nan'):.1f} Hz")
    if "accsweep" in S:
        print("\nACCEL vs SetJointVelLimit")
        for r in S["accsweep"]:
            print(f"  {r['pct']:5.0f} % | target accel {r.get('target_accel_deg_s2', math.nan):8.0f} deg/s2 | "
                  f"rise {r.get('target_rise_10_90_ms', math.nan):6.2f} ms | "
                  f"actual 50 % at {r.get('actual_t50_ms', math.nan):6.2f} ms")
    t = S.get("timing", {})
    print("\nTIMING (median / p95, ms)")
    for k, v in t.items():
        if v:
            print(f"  {k:30s} {v['median']:7.3f} / {v['p95']:7.3f}   (max {v['max']:.3f})")


def main():
    args = parse_args()
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A 1 ms loop in Python needs the GIL to change hands quickly (default is 5 ms).
    sys.setswitchinterval(0.0002)

    tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    out = args.out or f"meca_latency_{'sim_' if args.sim else ''}{datetime.now():%Y%m%d_%H%M%S}"
    os.makedirs(out, exist_ok=True)

    rec = Recorder()
    io = SimIO(args, rec) if args.sim else MecaIO(args, rec)
    raw = {}
    tester = None
    try:
        io.connect()
        tester = Tester(io, rec, args)
        print(f"Joint {args.joint}, vel limit {args.vel_limit_pct} % = {tester.vlim:.1f} deg/s, "
              f"amp {tester.amp:.1f} deg/s, period {args.period * 1e3:.1f} ms")
        for t in tests:
            fn = getattr(tester, f"run_{t}", None)
            if fn is None:
                print(f"unknown test '{t}', skipped")
                continue
            print(f"\n== {t} ==")
            tester.home()
            raw[t] = fn()
    except SafetyStop as e:
        print(f"\nSAFETY STOP: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        io.close()

    frames = rec.frames(0)
    if len(frames) < 10:
        print("No data recorded.")
        return
    print(f"\nAnalysing {len(frames)} feedback frames -> {out}/")
    cmap = clock_map(frames[:, HOST], frames[:, ROBOT])
    S = {}
    if "step" in raw:
        S["step"] = analyze_step(raw["step"], cmap); plot_step(raw["step"], out, plt)
    if "impulse" in raw:
        S["impulse"] = analyze_impulse(raw["impulse"], cmap); plot_impulse(raw["impulse"], out, plt)
    if "sine" in raw and raw["sine"]["runs"]:
        S["sine"] = analyze_sine(raw["sine"], cmap)
        plot_bode(raw["sine"], out, plt); plot_sine_traces(raw["sine"], out, plt)
    if "pingpong" in raw and raw["pingpong"]:
        S["pingpong"] = analyze_pingpong(raw["pingpong"], cmap); plot_pingpong(raw["pingpong"], out, plt)
    if "accsweep" in raw and raw["accsweep"]["sweeps"]:
        S["accsweep"] = analyze_accsweep(raw["accsweep"], cmap); plot_accsweep(raw["accsweep"], out, plt)
    S["timing"], td = analyze_timing(frames, tester.send_logs if tester else [], cmap)
    plot_timing(td, out, plt)
    S["config"] = {k: v for k, v in vars(args).items()}

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(jsonable(S), f, indent=2)
    arrays = {"frames": frames}
    if tester:
        for i, c in enumerate(tester.send_logs):
            arrays[f"cmd_{i:03d}"] = c
    np.savez_compressed(os.path.join(out, "raw.npz"), **arrays)
    print_summary(S)
    print(f"\nResults in {os.path.abspath(out)}  (frames columns: host_rx, robot_ts, "
          "target_pos, target_vel, actual_pos, actual_vel)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
