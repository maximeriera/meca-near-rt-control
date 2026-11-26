#!/usr/bin/env python3
"""
meca_latency_test.py  (v2)
==========================
Command -> feedback latency and dynamics of a Mecademic Meca500 in joint-velocity mode
(MoveJointsVel), in joint-position mode (MoveJoints), and of a position loop closed on
the host through MoveJointsVel.

What is measured
----------------
    host send --(net + queue)--> TARGET  (trajectory generator)
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

Velocity-mode model (identified on J6, 2026-11-24, SetJointAcc at its default 80 %)
  target_vel(t) = moving average over T of cmd(t - tau): a pure delay tau ~ 3-4 ms, then a
  boxcar of length T that does NOT depend on the step size. T ~ max(15.6 ms, ~1 ms per % of
  SetJointVelLimit). Every velocity test fits (tau, T) so the law can be checked on other
  joints, settings and firmware versions.

Tests (--tests: comma separated names and/or groups vel, pos, servo, all)
 Robot check
  vmaxcheck   ask 1.25x the nominal speed limit at a low SetJointVelLimit: real limit of each joint
              (feed it back with --joint-vmax if it differs from the R3 values)
 Velocity mode (MoveJointsVel)
  step        velocity steps 0 -> +A -> 0 -> -A -> 0: dead time, rise, accel, overshoot,
              settling, servo lag + fit of (tau, T)
  impulse     short velocity pulses of several widths
  sine        sine sweep on the velocity command: Bode gain / phase, equivalent delay
  pingpong    closed-loop relay: reverse the command as soon as feedback shows the motion
  accsweep    same small step at several SetJointVelLimit values: T(%) law
  jaccsweep   same step at several SetJointAcc values (and 2 vel limits): does it move T?
  ampsweep    steps of growing size up to a -A -> +A swing: still linear (T constant)?
  multijoint  same step on each joint alone, then together: T per joint or synchronised?
  periodsweep same sine streamed every 1..20 ms: cost of a slower command rate
 Position mode (MoveJoints)
  posstep     single moves of several sizes: dead time, duration, settling, final error
  posstream   sine setpoints streamed as MoveJoints + SetBlending at policy-like rates
              (10-50 ms): tracking delay / gain, stop-and-go, queue backlog
  posinterrupt  long MoveJoints interrupted by ClearMotion + new MoveJoints: re-targeting
 External loop
  servo       position loop on the host, v = Kp (q_ref - q_meas) [+ feedforward], sent with
              MoveJointsVel: steps at several Kp, sine tracking with / without feedforward,
              all compared with the identified model
  servomulti  the same loop on every --multi-joints joint at once (sines in phase, with / without
              feedforward): does tracking hold when all joints move?
 Physical AI
  replay      a policy-like action stream on --multi-joints (CSV --ref-file, or synthetic motion with
              --ref-chunk / --ref-noise-deg / --ref-jitter-ms / --ref-chunk-offset-deg / --ref-lead-ms)
              through --replay-modes vel (loop + feedforward), ff (feedforward only), pos (one MoveJoints
              per action), posi (MoveJoints every --pos-interp-ms to the host-interpolated position):
              error, delay, error once delayed, smoothness, backlog. --ref-predict quad extrapolates
              with the last change of slope. --ref-smooth N tracks the actions through a centred
              Savitzky-Golay filter; with noise or chunk offsets the error is also given against the
              clean motion, and the acceleration against the clean motion's.
 Worst cases (teleoperation)
  soak        the host loop on --multi-joints for --soak-dur s per --soak-loads (idle, cpu = busy
              processes, gil = busy Python thread in this process): tails (p99, p99.9, max) of feedback
              age, send interval, frame gaps, missing frames. Long: not part of 'all'. netcap.py compares
              a packet capture of the monitoring port with the frames of a run.
 External sensor
  sensorsweep the host loop through an emulated sensor (--sensor-rates x --sensor-delays, Kp from
              the model for --sensor-pm): step + sines vs the model. --fb-delay-ms / --fb-rate-hz /
              --fb-noise-deg put the same sensor in servo, servomulti and replay.
 Cartesian
  cartvel     MoveLinVelWrf / Trf steps along --cart-axes from --cart-pose: tau, T, servo lag
  timing      (always) RT frame period / missing frames, feedback transport, send jitter

Usage
  python meca_latency_test.py --sim --tests all                          # dry run, no robot
  python meca_latency_test.py --ip 192.168.0.100 --joint 6 --vel-limit-pct 20 --tests vel
  python meca_latency_test.py --ip 192.168.0.100 --tests step,posstep,posstream,posinterrupt
  python meca_latency_test.py --ip 192.168.0.100 --tests step,servo --vel-limit-pct 15
  python meca_latency_test.py --reanalyze meca_latency_20261125_101500   # redo analysis + plots

EtherCAT: ecat/ (meca_ecat, Rust, Linux RT) runs step, sine, accsweep, multijoint, servo, servomulti,
replay and soak over EtherCAT at 1 ms; ecat_bridge.py imports its runs into this format and analyses
them here (raw["_ecat"]: EtherCAT clock map clock_map_ecat + ecat_timing). See ecat/README.md.

Settings left on the robot: position tests leave SetJointVel at --pos-joint-vel-pct and
SetBlending at --blending; jaccsweep restores SetJointAcc to --joint-acc-pct (or the 80 %
activation default).

SAFETY: the robot WILL move. Default joint is 6 (wrist). Every test is bounded by
--max-excursion degrees from the start pose on every joint, checked on the feedback, and
the velocity timeout stops the robot if this script dies. multijoint moves several joints
(--multi-joints, wrist only by default). Clear the workspace and keep the E-stop at hand.

Requires: pip install mecademicpy numpy matplotlib scipy   (Python >= 3.11 on Windows for
1 ms sleep resolution)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import numpy as np

# Meca500 joint speed limits at 100 % SetJointVelLimit (deg/s). Measured with the vmaxcheck test
# on a Meca500 R4, firmware 11.3.1 (2026-11-25): equal to these nominal values. A MoveJointsVel
# above the limit is clamped silently, not refused. Override with --joint-vmax if yours differ.
JOINT_VMAX = [150.0, 150.0, 180.0, 300.0, 300.0, 500.0]
NJ = 6
ENC_STEP = 0.000862          # J6 encoder step seen in rt_joint_pos [deg]
DEFAULT_JOINT_ACC = 80.0     # MX_MQ_DEFAULT_JOINT_ACC
DEFAULT_JOINT_VEL = 25.0     # MX_MQ_DEFAULT_JOINT_VEL
DEFAULT_BLENDING = 100.0     # MX_MQ_DEFAULT_BLENDING

# J6 law identified on 2026-11-24, used only when this run has no step test
J6_T_FLOOR_MS, J6_T_MS_PER_PCT, J6_TAU_MS = 15.6, 1.0, 3.5

# Columns of one feedback frame.
#   0-5  : the tested joint (--joint), same layout as v1 of this script
#   6-29 : every joint, see col()
#   30-53: Cartesian target / actual pose and velocity (x, y, z [mm], alpha, beta, gamma [deg]),
#          only filled when cartvel runs (runs recorded before 2026-11-25 14h have 30 columns)
HOST, ROBOT, TPOS, TVEL, APOS, AVEL = range(6)
K_TPOS, K_TVEL, K_APOS, K_AVEL, K_TCART, K_TCVEL, K_ACART, K_ACVEL = range(8)
NCOL = 6 + 8 * NJ
NAN6 = np.full(NJ, np.nan)

TEST_GROUPS = {
    "vel": ["step", "impulse", "sine", "pingpong", "accsweep", "jaccsweep", "ampsweep",
            "multijoint", "periodsweep"],
    "pos": ["posstep", "posstream", "posinterrupt"],
    "servo": ["servo"],
}
TEST_GROUPS["all"] = (["vmaxcheck"] + TEST_GROUPS["vel"] + TEST_GROUPS["pos"] + TEST_GROUPS["servo"]
                      + ["servomulti", "replay", "sensorsweep", "cartvel"])


def col(kind: int, j: int) -> int:
    """Column of joint j (0-based) for kind K_TPOS / K_TVEL / K_APOS / K_AVEL."""
    return 6 + kind * NJ + j


def alljoints(fr, kind):
    a = col(kind, 0)
    return fr[..., a:a + NJ]


class SafetyStop(RuntimeError):
    """A joint left the allowed window or stopped responding."""


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
    """One row per RT cycle (see the column constants). Grows by chunks so long
    sessions never run out of space and nothing is copied while recording."""
    CHUNK = 100_000

    def __init__(self, joint: int):
        self.j = joint
        self.chunks = [np.empty((self.CHUNK, NCOL))]
        self.n = 0
        self.cond = threading.Condition()
        self._last_robot_t = None

    def add(self, host_t, robot_t, tp, tv, ap, av, ctp=NAN6, ctv=NAN6, cap=NAN6, cav=NAN6):
        if robot_t == self._last_robot_t:  # same cycle seen twice
            return
        self._last_robot_t = robot_t
        j = self.j
        row = np.empty(NCOL)
        row[:6] = (host_t, robot_t, tp[j], tv[j], ap[j], av[j])
        row[6:] = np.concatenate([tp, tv, ap, av, ctp, ctv, cap, cav])
        with self.cond:
            c, i = divmod(self.n, self.CHUNK)
            if c == len(self.chunks):
                self.chunks.append(np.empty((self.CHUNK, NCOL)))
            self.chunks[c][i] = row
            self.n += 1
            self.cond.notify_all()

    def wait_new(self, idx: int, timeout: float) -> int:
        with self.cond:
            self.cond.wait_for(lambda: self.n > idx, max(0.0, timeout))
            return self.n

    def row(self, i: int):
        c, k = divmod(i, self.CHUNK)
        return self.chunks[c][k].copy()

    def latest(self):
        n = self.n
        return self.row(n - 1) if n else None

    def frames(self, i0: int = 0, i1: int | None = None) -> np.ndarray:
        i1 = self.n if i1 is None else min(i1, self.n)
        if i1 <= i0:
            return np.empty((0, NCOL))
        parts = []
        for c in range(i0 // self.CHUNK, (i1 - 1) // self.CHUNK + 1):
            a = max(i0, c * self.CHUNK) - c * self.CHUNK
            b = min(i1, (c + 1) * self.CHUNK) - c * self.CHUNK
            parts.append(self.chunks[c][a:b])
        return np.concatenate(parts).copy()


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
            names = ["MX_ST_RT_TARGET_JOINT_POS", "MX_ST_RT_TARGET_JOINT_VEL",
                     "MX_ST_RT_JOINT_POS", "MX_ST_RT_JOINT_VEL"]
            if "cartvel" in expand_tests(self.args.tests):  # Cartesian data only when needed
                names += ["MX_ST_RT_TARGET_CART_POS", "MX_ST_RT_TARGET_CART_VEL",
                          "MX_ST_RT_CART_POS", "MX_ST_RT_CART_VEL"]
            return [int(getattr(mx.MxRobotStatusCode, n)) for n in names]
        except Exception:
            return ["all"]

    def connect(self):
        a, r, mdr = self.args, self.robot, self.mdr
        if a.rt_reader == "callback":
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
        self.info_before = dict(joint_acc_pct=self.get_joint_acc(), joint_vel_limit_pct=self._get("GetJointVelLimit"))
        print("Robot settings before this script: " + ", ".join(f"{k} {v}" for k, v in self.info_before.items()))
        r.SetMonitoringInterval(a.monitor_interval)
        r.SetRealTimeMonitoring(*self._rt_events())
        r.SetVelTimeout(a.vel_timeout)
        r.SetJointVelLimit(a.vel_limit_pct)
        if a.joint_acc_pct is not None:
            r.SetJointAcc(a.joint_acc_pct)
        r.WaitIdle(30)
        time.sleep(0.5)
        if a.rt_reader == "poll" or self.rec.n == 0:
            print("Real-time data: polling GetRobotRtData() in a dedicated thread" if a.rt_reader == "poll"
                  else "No on_end_of_cycle callback received -> polling GetRobotRtData() instead")
            self._polling = True
            threading.Thread(target=self._poll, daemon=True).start()
            time.sleep(0.5)
        if self.rec.n == 0:
            raise RuntimeError("No real-time data received (check SetRealTimeMonitoring / firmware)")
        j = r.GetRtTargetJointPos()
        self.start_joints = [float(x) for x in getattr(j, "data", j)]
        print(f"Start joints: {np.round(self.start_joints, 3)}")
        self.info = dict(joint_acc_pct=self.get_joint_acc(), joint_vel_limit_pct=self._get("GetJointVelLimit"),
                         joint_vel_pct=self._get("GetJointVel"), blending_pct=self._get("GetBlending"),
                         before_script=self.info_before, identity=self._identity(), rt_reader=a.rt_reader)
        print("Robot settings used: " + ", ".join(f"{k} {v}" for k, v in self.info.items()
                                                  if k not in ("before_script", "identity")))
        ident = self.info["identity"]
        print("Robot: " + ", ".join(f"{k} {ident.get(k)}" for k in ("model", "revision", "version", "serial")
                                     if k in ident))

    def _identity(self):
        """Model, revision, firmware version, serial... as plain values (GetRobotInfo)."""
        try:
            info = self.robot.GetRobotInfo()
            return {k: (v if isinstance(v, (str, int, float, bool)) or v is None else str(v))
                    for k, v in vars(info).items() if not k.startswith("_")}
        except Exception as e:  # noqa: BLE001
            return dict(error=str(e))

    def recover_if_error(self):
        """True if the robot was in error (then reset and resumed)."""
        try:
            if self.robot.GetStatusRobot().error_status:
                self._recover()
                return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def _get(self, name):
        try:
            return float(getattr(self.robot, name)(timeout=5))
        except Exception as e:  # noqa: BLE001
            print(f"  {name}() failed: {e}")
            return None

    def get_joint_acc(self):
        return self._get("GetJointAcc")

    # -- feedback ------------------------------------------------------------
    def _push(self, d, host_t):
        def get(field):
            x = getattr(d, field, None)
            if x is None or not getattr(x, "enabled", True):
                return NAN6
            data = getattr(x, "data", None)
            if data is None or len(data) < NJ:
                return NAN6
            return np.asarray(data[:NJ], float)

        ts = getattr(d.rt_target_joint_pos, "timestamp", None) or d.rt_joint_pos.timestamp
        self.rec.add(host_t, ts * 1e-6,  # robot timestamps are in microseconds
                     get("rt_target_joint_pos"), get("rt_target_joint_vel"),
                     get("rt_joint_pos"), get("rt_joint_vel"),
                     get("rt_target_cart_pos"), get("rt_target_cart_vel"),
                     get("rt_cart_pos"), get("rt_cart_vel"))

    def _on_cycle(self, *_, **__):
        if not self._polling:
            # read the data first: a time taken before could be older than the data if this
            # thread is pre-empted in between, which breaks the lower-envelope clock map
            d = self.robot.GetRobotRtData()
            self._push(d, time.perf_counter())

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

    # -- settings (return False instead of raising when the robot refuses) ----
    def _setting(self, name, *a):
        try:
            getattr(self.robot, name)(*a)
            self.robot.WaitIdle(10)
            if self.robot.GetStatusRobot().error_status:
                raise RuntimeError("robot went into error")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  {name}{a} refused: {e}")
            self._recover()
            return False

    def _recover(self):
        try:
            if self.robot.GetStatusRobot().error_status:
                self.robot.ResetError()
            self.robot.ResumeMotion()
        except Exception as e:  # noqa: BLE001
            print(f"  recovery failed: {e}")

    def set_vel_limit(self, pct):
        return self._setting("SetJointVelLimit", pct)

    def set_joint_acc(self, pct):
        return self._setting("SetJointAcc", pct)

    def set_joint_vel(self, pct):
        return self._setting("SetJointVel", pct)

    def set_blending(self, pct):
        return self._setting("SetBlending", pct)

    def set_vel_timeout(self, t):
        return self._setting("SetVelTimeout", t)

    # -- commands ------------------------------------------------------------
    def send_vel(self, v):
        """Scalar = tested joint only, else a 6-vector."""
        if np.ndim(v) == 0:
            vel = [0.0] * NJ
            vel[self.j] = float(v)
        else:
            vel = [float(x) for x in v]
        self.robot.MoveJointsVel(*vel)

    def move_joints(self, q):
        self.robot.MoveJoints(*[float(x) for x in q])

    def send_cart_vel(self, v, frame="wrf"):
        """Cartesian velocity (vx, vy, vz [mm/s], wx, wy, wz [deg/s]) in the world or tool frame."""
        cmd = self.robot.MoveLinVelTrf if frame == "trf" else self.robot.MoveLinVelWrf
        cmd(*[float(x) for x in v])

    def set_cart_acc(self, pct):
        return self._setting("SetCartAcc", pct)

    def get_cart_acc(self):
        return self._get("GetCartAcc")

    def clear_motion(self):
        """ClearMotion includes an implicit PauseMotion, hence the ResumeMotion."""
        self.robot.ClearMotion()
        self.robot.ResumeMotion()

    def wait_idle(self, timeout):
        self.robot.WaitIdle(timeout)

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
    """Crude stand-in so every test and the analysis can be dry-run without hardware.
    Invented numbers loosely shaped on the J6 model of 2026-11-24: ~0.5 ms uplink, 1 ms
    cycle, velocity cmd -> moving average T = max(15.6 ms, 1 ms per % of SetJointVelLimit)
    * 80 / SetJointAcc, 2-cycle + 3 ms first-order servo, 0.00086 deg encoder steps,
    ~0.5 ms downlink, drifting robot clock. Position moves: per-joint accel-limited
    profile with crude blending. Nothing here says anything about a real Meca500."""
    CYCLE = 0.001
    POS_ACC = 2000.0  # position-move acceleration at SetJointAcc 100 % [deg/s^2], invented

    def __init__(self, args, rec: Recorder):
        self.args, self.rec = args, rec
        self.j = args.joint - 1
        self.rng_in, self.rng_out = np.random.default_rng(1), np.random.default_rng(2)
        self.lock = threading.Lock()
        self.inbox, self.outbox = [], []
        self.vl_pct = args.vel_limit_pct
        self.acc_pct = args.joint_acc_pct or DEFAULT_JOINT_ACC
        self.jv_pct, self.blend, self.vel_timeout = DEFAULT_JOINT_VEL, DEFAULT_BLENDING, args.vel_timeout
        self.start_joints = [0.0] * NJ
        self.mon_every = max(1, round(args.monitor_interval / self.CYCLE))
        self._last_in = self._last_out = 0.0
        self.running = False
        self._reset()

    def _reset(self):
        self.vcmd, self.vcmd_t = np.zeros(NJ), -1e9
        self.ring, self.k = np.zeros((4000, NJ)), 0
        self.tp, self.tv = np.zeros(NJ), np.zeros(NJ)
        self.ap, self.av = np.zeros(NJ), np.zeros(NJ)
        self.apq_prev = np.zeros(NJ)
        self.dline = [np.zeros(NJ), np.zeros(NJ)]
        self.queue, self.goal = [], None

    def _T(self):
        return max(0.0156, 0.001 * self.vl_pct) * DEFAULT_JOINT_ACC / max(self.acc_pct, 1.0)

    def _pos_acc(self):
        return self.POS_ACC * self.acc_pct / 100.0

    def _vel_active(self, now):
        return now - self.vcmd_t <= self.vel_timeout

    def connect(self):
        print("SIMULATION: invented robot model, the numbers say nothing about a real Meca500.")
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        time.sleep(0.3)

    def _pos_step(self, dt):
        if self.goal is None:
            if not self.queue:
                self.tv = np.zeros(NJ)
                return
            self.goal = np.asarray(self.queue.pop(0), float)
        a = self._pos_acc()
        vmax = np.array(JOINT_VMAX) * min(self.jv_pct, self.vl_pct) / 100
        e = self.goal - self.tp
        vdes = np.sign(e) * np.minimum(vmax, np.sqrt(2 * a * np.abs(e)))
        self.tv = self.tv + np.clip(vdes - self.tv, -a * dt, a * dt)
        if np.all(np.abs(e) < 5e-3) and np.all(np.abs(self.tv) < 5.0):
            self.tp, self.tv, self.goal = self.goal.copy(), np.zeros(NJ), None
        elif self.queue and self.blend > 0 and np.all(np.abs(e) <= self.tv ** 2 / (2 * a) * self.blend / 100 + 1e-6):
            self.goal = np.asarray(self.queue.pop(0), float)

    def _run(self):
        dt, k = self.CYCLE, 0
        t_next = time.perf_counter()
        while self.running:
            t_next += dt
            sleep_until(t_next)
            now = time.perf_counter()
            with self.lock:
                while self.inbox and self.inbox[0][0] <= now:
                    _, kind, val = self.inbox.pop(0)
                    if kind == "vel":
                        self.vcmd, self.vcmd_t = val, now
                    elif kind == "mj":
                        self.queue.append(val)
                    elif kind == "clear":
                        self.queue.clear()
                        if self.goal is not None:
                            self.goal = self.tp + self.tv * np.abs(self.tv) / (2 * self._pos_acc())
                vlim = np.array(JOINT_VMAX) * self.vl_pct / 100
                u = np.clip(self.vcmd, -vlim, vlim) if self._vel_active(now) else np.zeros(NJ)
                self.ring[self.k % len(self.ring)] = u
                self.k += 1
                N = max(1, min(len(self.ring), int(round(self._T() / dt))))
                vel_tv = self.ring[(self.k - np.arange(1, N + 1)) % len(self.ring)].mean(axis=0)
                if self._vel_active(now) or np.any(np.abs(vel_tv) > 1e-9):
                    self.tv = vel_tv
                else:
                    self._pos_step(dt)
                self.tp = self.tp + self.tv * dt
                self.dline.append(self.tv.copy())
                ref = self.dline.pop(0)
                self.av += (ref - self.av) * dt / 0.003
                self.ap += self.av * dt
                apq = np.round(self.ap / ENC_STEP) * ENC_STEP
                avq = (apq - self.apq_prev) / dt
                self.apq_prev = apq
                k += 1
                if k % self.mon_every == 0:
                    arrive = max(self._last_out, now + 0.0003 + self.rng_out.exponential(0.0003))
                    self._last_out = arrive
                    rt = now * (1 + 40e-6) + 1234.5
                    # no kinematics here: the "Cartesian" signals are the joint ones (fake, for dry runs)
                    self.outbox.append((arrive, rt, self.tp.copy(), self.tv.copy(), apq, avq,
                                        self.tp.copy(), self.tv.copy(), apq, avq))
                ready = []
                while self.outbox and self.outbox[0][0] <= now:
                    ready.append(self.outbox.pop(0))
            for _, rt, *vals in ready:
                self.rec.add(time.perf_counter(), rt, *vals)

    def _post(self, kind, val):
        now = time.perf_counter()
        with self.lock:
            arrive = max(self._last_in, now + 0.0004 + self.rng_in.exponential(0.0002))
            self._last_in = arrive
            self.inbox.append((arrive, kind, val))

    def send_vel(self, v):
        if np.ndim(v) == 0:
            vec = np.zeros(NJ)
            vec[self.j] = float(v)
        else:
            vec = np.asarray(v, float).copy()
        self._post("vel", vec)

    def move_joints(self, q):
        self._post("mj", np.asarray(q, float).copy())

    def send_cart_vel(self, v, frame="wrf"):
        self._post("vel", np.asarray(v, float).copy())  # fake: Cartesian axis k drives joint k

    def set_cart_acc(self, pct):
        return self._set("cart_acc_pct", pct)

    def get_cart_acc(self):
        return getattr(self, "cart_acc_pct", 100.0)

    def clear_motion(self):
        self._post("clear", None)

    def _set(self, attr, val):
        with self.lock:
            setattr(self, attr, val)
        return True

    def get_joint_acc(self):
        return self.acc_pct

    def recover_if_error(self):
        return False

    def set_vel_limit(self, pct):
        return self._set("vl_pct", pct)

    def set_joint_acc(self, pct):
        return self._set("acc_pct", pct)

    def set_joint_vel(self, pct):
        return self._set("jv_pct", pct)

    def set_blending(self, pct):
        return self._set("blend", pct)

    def set_vel_timeout(self, t):
        return self._set("vel_timeout", t)

    def wait_idle(self, timeout):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            with self.lock:
                idle = (not self.inbox and not self.queue and self.goal is None
                        and not self._vel_active(time.perf_counter())
                        and not np.any(np.abs(self.tv) > 1e-6))
            if idle:
                return
            time.sleep(0.002)
        raise TimeoutError("sim WaitIdle timeout")

    def stop(self):
        self.send_vel(0.0)
        self.wait_idle(10)

    def return_to_start(self):
        self.stop()
        with self.lock:
            self._reset()
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


class SensorEmu:
    """An external position sensor in the host loop (camera, tracker): it samples the robot
    position every 1/rate s (every loop tick when rate <= 0), adds white noise, and the controller
    gets the sample `delay` s later and holds it until the next one arrives. All zero (default):
    the robot's own feedback, untouched. The frame returned with the sample is the one it was
    taken from, so the logged feedback age includes the sensor delay and the hold."""

    def __init__(self, delay_s=0.0, rate_hz=0.0, noise_deg=0.0, seed=0):
        self.delay, self.rate, self.noise = float(delay_s), float(rate_hz), float(noise_deg)
        self.rng = np.random.default_rng(seed)
        self.queue, self.next_t, self.held = deque(), 0.0, None

    @classmethod
    def from_args(cls, a):
        return cls(getattr(a, "fb_delay_ms", 0.0) * 1e-3, getattr(a, "fb_rate_hz", 0.0),
                   getattr(a, "fb_noise_deg", 0.0), getattr(a, "ref_seed", 0))

    @property
    def passthrough(self):
        return self.delay == 0 and self.rate <= 0 and self.noise == 0

    def observe(self, t, y, frame):
        if self.passthrough:
            return y, frame
        if self.rate <= 0 or t >= self.next_t - 1e-12:
            noisy = y + (self.rng.normal(0.0, self.noise, np.shape(y)) if self.noise > 0 else 0.0)
            self.queue.append((t + self.delay, noisy, frame))
            if self.rate > 0:
                while self.next_t <= t + 1e-12:
                    self.next_t += 1.0 / self.rate
        while self.queue and self.queue[0][0] <= t + 1e-12:
            _, yh, fh = self.queue.popleft()
            self.held = (yh, fh)
        return self.held if self.held is not None else (y, frame)


class ActionRef:
    """A stream of actions q_k (offsets from the start pose, deg, one row of 6 per action) due at
    t_k and known to the controller from release_k on (a policy that is late, or that publishes a
    whole chunk at once). What the controller can compute at time `now` for time te: linear
    interpolation of the released actions; past the last released one, an extrapolation for at
    most `horizon` (a late action), then the position is held. order 1 continues the last slope;
    order 2 also uses the last change of slope (it removes the half-period lag of the slope,
    at the cost of amplifying noise)."""

    def __init__(self, t, q, release, horizon, order=1):
        self.t = np.asarray(t, float)
        self.q = np.asarray(q, float)
        self.rel = np.maximum.accumulate(np.asarray(release, float))
        self.s = np.zeros_like(self.q)
        if len(self.t) > 1:
            self.s[:-1] = np.diff(self.q, axis=0) / np.diff(self.t)[:, None]
        self.horizon = horizon
        self.order = order

    def released(self, now):
        return int(np.searchsorted(self.rel, now, side="right"))

    def _extrap(self, n, te):
        """(position, velocity) at te past the last released action (index n - 1)."""
        last = n - 1
        dt = min(te - self.t[last], self.horizon)
        if self.order == 2 and n >= 3:
            h1, h0 = self.t[last] - self.t[last - 1], self.t[last - 1] - self.t[last - 2]
            acc = (self.s[last - 1] - self.s[last - 2]) / (0.5 * (h1 + h0))
            v = self.s[last - 1] + acc * h1 / 2      # velocity at the last action
            return self.q[last] + v * dt + 0.5 * acc * dt * dt, v + acc * dt
        return self.q[last] + self.s[last - 1] * dt, self.s[last - 1].copy()

    def at(self, te, now):
        n = self.released(now)
        if n == 0:
            return np.zeros(NJ)
        i = int(np.searchsorted(self.t[:n], te, side="right")) - 1
        if i < 0:
            return self.q[0].copy()
        if i < n - 1:
            return self.q[i] + self.s[i] * (te - self.t[i])
        if n == 1 or n == len(self.t):
            return self.q[n - 1].copy()
        return self._extrap(n, te)[0]

    def slope(self, te, now):
        n = self.released(now)
        if n < 2:
            return np.zeros(NJ)
        i = int(np.searchsorted(self.t[:n], te, side="right")) - 1
        if i < 0:
            return np.zeros(NJ)
        if i < n - 1:
            return self.s[i].copy()
        if n == len(self.t) or te - self.t[n - 1] > self.horizon:
            return np.zeros(NJ)
        return self._extrap(n, te)[1]

    def true_at(self, te, j):
        """The commanded trajectory itself (every action, at its due time) for joint j."""
        return np.interp(np.asarray(te, float), self.t, self.q[:, j])

    def true_slope(self, te, j):
        i = np.clip(np.searchsorted(self.t, te, side="right") - 1, 0, len(self.t) - 1)
        return self.s[i, j]


def make_reference(a, js, vlim_all):
    """Actions (t, q offsets) from --ref-file (CSV: t, q1..q6 in deg, header allowed) or a synthetic
    smooth motion: 3 sines (0.3, 0.7, 1.3 Hz) per joint with random phases, started and ended at rest."""
    if a.ref_file:
        data = np.genfromtxt(a.ref_file, delimiter=",", comments="#")
        data = data[~np.isnan(data).all(axis=1)] if data.ndim == 2 else data.reshape(1, -1)
        if np.isnan(data[0]).any():
            data = data[1:]
        t = data[:, 0] - data[0, 0]
        q = np.zeros((len(t), NJ))
        m = min(NJ, data.shape[1] - 1)
        q[:, :m] = data[:, 1:1 + m]
        if not a.ref_relative:
            q[:, :m] -= q[0, :m]
        return t, q, os.path.basename(a.ref_file)
    rng = np.random.default_rng(a.ref_seed)
    P = a.ref_period_ms * 1e-3
    t = np.arange(0.0, a.replay_dur + 1e-9, P)
    q = np.zeros((len(t), NJ))
    freqs = np.array([0.3, 0.7, 1.3])
    ramp = np.clip(np.minimum(t, t[-1] - t) / 0.5, 0, 1)
    win = 0.5 - 0.5 * np.cos(np.pi * ramp)
    for j in js:
        A = min(a.ref_amp, 0.5 * vlim_all[j] / (2 * np.pi * freqs.mean()))  # peak speed <= half the limit
        ph = rng.uniform(0, 2 * np.pi, len(freqs))
        q[:, j] = win * sum(A / len(freqs) * np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, ph))
    return t, q, "synthetic"


def policy_effects(t, q, a, js):
    """What a policy adds to a clean motion: actions published by chunks of --ref-chunk (all known
    when the chunk's first action is due), a random offset per chunk (--ref-chunk-offset-deg: the
    jumps seen between chunks), white noise per action (--ref-noise-deg), a late publication per
    chunk (exponential, mean --ref-jitter-ms), and an early publication (--ref-lead-ms: the next
    chunk is out before its first action is due, as with asynchronous inference). Returns (q, release)."""
    rng = np.random.default_rng(a.ref_seed + 1)
    n, N = len(t), max(1, int(a.ref_chunk))
    chunk = np.arange(n) // N
    nc = int(chunk.max()) + 1
    q = q.copy()
    if a.ref_chunk_offset_deg > 0:
        off = rng.normal(0.0, a.ref_chunk_offset_deg, (nc, NJ))
        off[0] = 0.0
        q[:, js] += off[chunk][:, js]
    if a.ref_noise_deg > 0:
        q[:, js] += rng.normal(0.0, a.ref_noise_deg, (n, len(js)))
    jit = rng.exponential(a.ref_jitter_ms * 1e-3, nc) if a.ref_jitter_ms > 0 else np.zeros(nc)
    release = t[chunk * N] - a.ref_lead_ms * 1e-3 + jit[chunk]
    return q, release


def smooth_actions(q, release, n_win, js):
    """What a host can do with noisy actions before tracking them: a centred Savitzky-Golay filter
    over n_win actions (made odd; order 2, order 1 = moving average for 3). The smoothed action k
    needs the raw actions up to k + n_win // 2, so it is only released when they are: free when
    chunks are published ahead, a delay of n_win // 2 actions on a just-in-time stream.
    Returns (q, release, half window); unchanged below 3 actions."""
    h = int(n_win or 0) // 2
    n = len(q)
    if h < 1 or n < 2 * h + 1:
        return q, np.asarray(release, float), 0
    from scipy.signal import savgol_filter
    qs = np.array(q, float)
    qs[:, js] = savgol_filter(qs[:, js], 2 * h + 1, 1 if h == 1 else 2, axis=0, mode="interp")
    rel = np.asarray(release, float)[np.minimum(np.arange(n) + h, n - 1)]
    return qs, rel, h


class LoadGen:
    """Background load of the soak test, as a context manager. 'idle': none. 'cpu': busy processes
    (one per logical core by default) at normal priority, like other programs on the PC. 'gil': a busy
    pure-Python thread in this process, like a policy's Python code running next to the control loop
    (it competes for the interpreter lock, not only for the CPU)."""
    KINDS = ("idle", "cpu", "gil")

    def __init__(self, kind, procs=0, max_s=600.0):
        if kind not in self.KINDS:
            raise ValueError(f"unknown soak load '{kind}' (one of {', '.join(self.KINDS)})")
        self.kind, self.procs = kind, int(procs) or (os.cpu_count() or 2)
        self.max_s = float(max_s)  # the busy processes stop by themselves after this, even if we die
        self._procs, self._stop, self._thread = [], threading.Event(), None

    def _burn(self):
        x = 0
        while not self._stop.is_set():
            x += 1

    def __enter__(self):
        if self.kind == "cpu":
            flags = getattr(subprocess, "NORMAL_PRIORITY_CLASS", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            code = f"import time\nt = time.perf_counter() + {self.max_s}\nwhile time.perf_counter() < t: pass"
            self._procs = [subprocess.Popen([sys.executable, "-c", code], creationflags=flags)
                           for _ in range(self.procs)]
        elif self.kind == "gil":
            self._thread = threading.Thread(target=self._burn, daemon=True)
            self._thread.start()
        if self.kind != "idle":
            time.sleep(0.5)  # let the load settle
        return self

    def __exit__(self, *exc):
        self._stop.set()
        for p in self._procs:
            p.kill()
        for p in self._procs:
            try:
                p.wait(5)
            except Exception:  # noqa: BLE001
                pass
        if self._thread:
            self._thread.join(2)
        return False


class Tester:
    def __init__(self, io, rec: Recorder, args):
        self.io, self.rec, self.args = io, rec, args
        self.j = args.joint - 1
        self.vlim = JOINT_VMAX[self.j] * args.vel_limit_pct / 100
        self.vlim_all = np.array(JOINT_VMAX) * args.vel_limit_pct / 100
        amp = args.amp if args.amp else 0.5 * self.vlim
        if amp > 0.95 * self.vlim:
            print(f"--amp {amp} is above 95 % of the velocity limit ({self.vlim:.1f} deg/s): clamped")
            amp = 0.95 * self.vlim
        self.amp = amp
        self.send_logs = []          # streamed velocity commands: [scheduled, before, after, v_joint, v1..v6]
        self.start_pos = 0.0
        self.start_all = None
        self.measured_acc = None
        self.quick_servo_lag = None
        self.quick_model = None

    # -- helpers -------------------------------------------------------------
    def home(self):
        self.io.return_to_start()
        fr = self.rec.latest()
        if fr is not None:
            self.start_pos = float(fr[TPOS])
            self.start_all = alljoints(fr, K_TPOS).copy()

    def guard(self):
        """Every joint must stay within --max-excursion of the start pose."""
        fr = self.rec.latest()
        if fr is None or self.start_all is None:
            return
        pos = alljoints(fr, K_APOS)
        pos = np.where(np.isfinite(pos), pos, alljoints(fr, K_TPOS))
        dev = np.abs(pos - self.start_all)
        if np.isfinite(dev).any():
            k = int(np.nanargmax(dev))
            if dev[k] > self.args.max_excursion:
                raise SafetyStop(f"joint {k + 1} is {pos[k] - self.start_all[k]:+.1f} deg from start "
                                 f"(limit +/-{self.args.max_excursion})")

    def model_delay(self):
        """Command -> actual position delay tau + T/2 + P/2 + servo lag from this session's step
        test, else the J6 law."""
        if self.quick_model:
            tau, T = self.quick_model["tau_ms"], self.quick_model["T_ms"]
        else:
            tau, T = J6_TAU_MS, max(J6_T_FLOOR_MS, J6_T_MS_PER_PCT * self.args.vel_limit_pct)
        lag = max(0.0, self.quick_servo_lag or 0.0)
        return (tau + T / 2 + lag) * 1e-3 + self.args.period / 2

    def stream(self, profile, duration: float, period: float | None = None):
        """Send profile(t) (scalar: tested joint, or 6-vector) every period. Returns (cmd log, frames)."""
        P = period or self.args.period
        n = int(round(duration / P))
        log = np.full((n, 4 + NJ), np.nan)
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
                if np.ndim(v) == 0:
                    vec = np.zeros(NJ)
                    vec[self.j] = v
                else:
                    vec = np.asarray(v, float)
                log[k, :4] = (ts, t1, t2, vec[self.j])
                log[k, 4:] = vec
                done = k + 1
                if k % 5 == 0:
                    self.guard()
        finally:
            self.io.send_vel(0.0)
        time.sleep(0.15)  # trailing feedback
        log = log[:done]
        self.send_logs.append(log)
        return log, self.rec.frames(i0)

    # -- velocity-mode tests -------------------------------------------------
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
        if amp is None:  # quick estimates used to size the sine test and the servo feedforward
            self.measured_acc, self.quick_servo_lag = quick_accel(res)
            if self.measured_acc:
                print(f"  measured target acceleration ~ {self.measured_acc:.0f} deg/s^2")
            self.quick_model = quick_model(res)
            if self.quick_model:
                print(f"  model: tau ~ {self.quick_model['tau_ms']:.2f} ms, T ~ {self.quick_model['T_ms']:.2f} ms")
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

    def _sine_amp(self, f):
        acc_lim = self.args.sine_acc or (0.5 * self.measured_acc if self.measured_acc else 1000.0)
        w = 2 * math.pi * f
        return min(self.amp, acc_lim / w, 0.35 * self.args.max_excursion * w), acc_lim

    def run_sine(self):
        runs, acc_lim = [], None
        for f in self.args.freqs:
            w = 2 * math.pi * f
            A, acc_lim = self._sine_amp(f)
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
        for name, cl, pcol in (("target", TVEL, TPOS), ("actual", AVEL, APOS)):
            fr = self.rec.latest()
            if not (np.isfinite(fr[cl]) or np.isfinite(fr[pcol])):
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
                        f_ = self.rec.row(n - 1)
                        idx = n
                        v = f_[cl]
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
                if not self.io.set_vel_limit(p):
                    continue
                out.append(dict(pct=p, **self.run_step(amp=A, repeats=max(2, self.args.repeats // 2))))
        finally:
            self.io.set_vel_limit(self.args.vel_limit_pct)
        return dict(amp=A, sweeps=out)

    def run_jaccsweep(self):
        a = self.args
        base = self.io.get_joint_acc()  # restore exactly what the robot had
        if base is None:
            base = a.joint_acc_pct if a.joint_acc_pct is not None else DEFAULT_JOINT_ACC
            print(f"  GetJointAcc failed: SetJointAcc will be restored to {base:g} %")
        A = 0.8 * JOINT_VMAX[self.j] * min(a.jacc_vl_pcts) / 100
        out = []
        try:
            for vl in a.jacc_vl_pcts:
                if not self.io.set_vel_limit(vl):
                    continue
                for p in a.jacc_pcts:
                    print(f"  SetJointVelLimit({vl}) + SetJointAcc({p})")
                    if not self.io.set_joint_acc(p):
                        out.append(dict(vl=vl, acc=p, failed=True))
                        continue
                    out.append(dict(vl=vl, acc=p, **self.run_step(amp=A, repeats=max(2, a.repeats // 2))))
        finally:
            self.io.set_joint_acc(base)
            self.io.set_vel_limit(a.vel_limit_pct)
        return dict(amp=A, base_acc=base, sweeps=out)

    def run_ampsweep(self):
        runs = []
        for frac in self.args.amp_fracs:
            A = min(frac, 0.95) * self.vlim
            hold = min(self.args.step_hold, 0.7 * self.args.max_excursion / A)
            segs = [(0.15, 0.0), (hold, A), (hold, -A), (hold, 0.0), (0.2, 0.0)]
            rr = []
            for r in range(max(2, self.args.repeats // 2)):
                print(f"  0 -> +{A:.1f} -> -{A:.1f} -> 0 deg/s, hold {hold * 1e3:.0f} ms")
                cmd, fr = self.stream(piecewise(segs), sum(s[0] for s in segs))
                rr.append(dict(cmd=cmd, frames=fr))
                self.home()
            runs.append(dict(frac=frac, amp=A, hold=hold, runs=rr))
        return dict(runs=runs)

    def run_multijoint(self):
        a = self.args
        js = sorted({j - 1 for j in a.multi_joints})
        amps = a.multi_frac * self.vlim_all
        hold = min(a.step_hold, 0.7 * a.max_excursion / amps[js].max())
        shape = piecewise([(0.15, 0.0), (hold, 1.0), (hold, 0.0), (hold, -1.0), (0.35, 0.0)])
        dur = 0.5 + 3 * hold
        scen = [(f"J{j + 1} alone", [j]) for j in js] + ([("together", js)] if len(js) > 1 else [])
        out = []
        for name, act in scen:
            Avec = np.zeros(NJ)
            Avec[act] = amps[act]
            print(f"  {name}: " + ", ".join(f"J{j + 1} {Avec[j]:.1f} deg/s" for j in act))
            runs = []
            for _ in range(max(2, a.repeats // 2)):
                cmd, fr = self.stream(lambda t, Av=Avec: shape(t) * Av, dur)
                runs.append(dict(cmd=cmd, frames=fr))
                self.home()
            out.append(dict(name=name, active=act, A=Avec, runs=runs))
        return dict(scenarios=out, hold=hold)

    def run_periodsweep(self):
        a = self.args
        f = a.ps_freq
        w = 2 * math.pi * f
        A, _ = self._sine_amp(f)
        settle = max(0.25, 1.0 / f)
        dur = settle + max(4.0 / f, 1.5)
        runs = []
        try:
            for Pms in a.ps_periods:
                P = Pms * 1e-3
                self.io.set_vel_timeout(max(a.vel_timeout, 3 * P))
                print(f"  sine {f:g} Hz, A={A:.1f} deg/s, one MoveJointsVel every {Pms:g} ms")
                cmd, fr = self.stream(lambda t: A * math.sin(w * t), dur, period=P)
                runs.append(dict(P=P, f=f, A=A, settle=settle, cmd=cmd, frames=fr))
                self.home()
        finally:
            self.io.set_vel_timeout(a.vel_timeout)
        return dict(runs=runs)

    def run_vmaxcheck(self):
        """Ask 1.25x the nominal speed limit of each joint at a low SetJointVelLimit and read the
        plateau of the target velocity: the real limit, instead of the JOINT_VMAX (R3) assumption."""
        a = self.args
        pct = a.vmax_vl_pct
        runs = []
        try:
            if not self.io.set_vel_limit(pct):
                return dict(pct=pct, runs=runs)
            for j in sorted({j - 1 for j in a.vmax_joints}):
                v_nom = JOINT_VMAX[j] * pct / 100
                v_cmd = 1.25 * v_nom
                hold = min(0.4, 0.35 * a.max_excursion / v_cmd)
                unit = np.zeros(NJ)
                unit[j] = v_cmd
                shape = piecewise([(0.1, 0.0), (hold, 1.0), (0.1, 0.0), (hold, -1.0), (0.2, 0.0)])
                print(f"  J{j + 1}: ask {v_cmd:.1f} deg/s at SetJointVelLimit {pct:g} % "
                      f"(nominal limit there: {v_nom:.1f} deg/s)")
                cmd, fr = self.stream(lambda t, u=unit: shape(t) * u, 0.6 + 2 * hold)
                if self.io.recover_if_error():
                    print("    the robot refused the command (error reset)")
                    runs.append(dict(joint=j + 1, failed="robot error"))
                else:
                    runs.append(dict(joint=j + 1, pct=pct, v_cmd=v_cmd, v_nom=v_nom, cmd=cmd, frames=fr))
                self.home()
        finally:
            self.io.set_vel_limit(a.vel_limit_pct)
        return dict(pct=pct, runs=runs)

    # -- position-mode tests -------------------------------------------------
    def run_posstep(self):
        a, io, j = self.args, self.io, self.j
        io.set_joint_vel(a.pos_joint_vel_pct)
        sizes = [min(s, 0.9 * a.max_excursion) for s in a.pos_steps]
        i0 = self.rec.n
        moves = []
        q_home = self.start_all.copy()
        for s in sizes:
            print(f"  MoveJoints +/-{s:g} deg x {a.repeats} (SetJointVel {a.pos_joint_vel_pct:g} %)")
            for r in range(a.repeats):
                sign = 1 if r % 2 == 0 else -1
                for tgt in (sign * s, 0.0):
                    time.sleep(a.pos_pause)
                    q = q_home.copy()
                    q[j] += tgt
                    q_from = float(self.rec.latest()[TPOS])
                    t1 = time.perf_counter()
                    io.move_joints(q)
                    t2 = time.perf_counter()
                    moves.append((t1, t2, q_from, q[j], s))
                    io.wait_idle(60)
                    self.guard()
        time.sleep(a.pos_pause)
        return dict(moves=np.array(moves), frames=self.rec.frames(i0), tol=a.pos_tol,
                    vel_pct=a.pos_joint_vel_pct)

    def run_posstream(self):
        a, io, j = self.args, self.io, self.j
        io.set_joint_vel(a.pos_joint_vel_pct)
        io.set_blending(a.blending)
        A = min(a.pos_amp, 0.7 * a.max_excursion)
        runs = []
        for Pms in a.pos_periods:
            P = Pms * 1e-3
            for f in a.pos_freqs:
                w = 2 * math.pi * f
                settle = max(0.5, 1.0 / f)
                dur = settle + max(2.0 / f, 1.5)
                n = int(round(dur / P)) + 1
                q_home = self.start_all.copy()
                log = np.empty((n, 4))
                i0 = self.rec.n
                done = 0
                print(f"  MoveJoints every {Pms:g} ms, sine {f:g} Hz, A = {A:g} deg ({n} moves, "
                      f"blending {a.blending:g} %)")
                t0 = time.perf_counter() + 0.01
                try:
                    for k in range(n):
                        ts = t0 + k * P
                        sleep_until(ts)
                        qr = A * math.sin(w * k * P)
                        q = q_home.copy()
                        q[j] += qr
                        t1 = time.perf_counter()
                        io.move_joints(q)
                        t2 = time.perf_counter()
                        log[k] = (ts, t1, t2, qr)
                        done = k + 1
                        if k % 5 == 0:
                            self.guard()
                except BaseException:
                    io.clear_motion()
                    raise
                t_last = time.perf_counter()
                try:
                    io.wait_idle(a.pos_backlog_timeout)
                except Exception as e:  # noqa: BLE001
                    print(f"    queue not drained after {a.pos_backlog_timeout} s ({e}) -> ClearMotion")
                    io.clear_motion()
                t_idle = time.perf_counter()
                time.sleep(0.1)
                runs.append(dict(P=P, f=f, A=A, settle=settle, q0=float(q_home[j]), cmd=log[:done],
                                 frames=self.rec.frames(i0), t_last=t_last, t_idle=t_idle))
                self.home()
        return dict(runs=runs, blending=a.blending, vel_pct=a.pos_joint_vel_pct)

    def run_posinterrupt(self):
        a, io, j = self.args, self.io, self.j
        io.set_joint_vel(a.int_joint_vel_pct)
        d = min(a.int_dist, 0.8 * a.max_excursion)
        i0 = self.rec.n
        ev = []
        print(f"  MoveJoints {d:g} deg at SetJointVel {a.int_joint_vel_pct:g} %, ClearMotion after "
              f"{a.int_after * 1e3:.0f} ms, then MoveJoints back")
        try:
            for r in range(a.repeats):
                s = 1 if r % 2 == 0 else -1
                q_home = self.start_all.copy()
                q1 = q_home.copy()
                q1[j] += s * d
                t_go = time.perf_counter()
                io.move_joints(q1)
                time.sleep(a.int_after)
                tc1 = time.perf_counter()
                io.clear_motion()
                tc2 = time.perf_counter()
                io.move_joints(q_home)
                tn2 = time.perf_counter()
                io.wait_idle(60)
                ev.append((t_go, tc1, tc2, tc2, tn2, s))
                self.guard()
                time.sleep(0.2)
        finally:
            io.set_joint_vel(a.pos_joint_vel_pct)
        time.sleep(0.1)
        frames = self.rec.frames(i0)
        self.home()
        return dict(events=np.array(ev), frames=frames, dist=d, vel_pct=a.int_joint_vel_pct)

    # -- external position loop ----------------------------------------------
    def _servo_loop(self, ref, dref, kp, adv, dur, sensor=None):
        """Fixed-rate loop at --period: read the latest frame, v = kp (r - y) [+ dref(t + adv)],
        clamp, MoveJointsVel. The position goes through `sensor` (default: --fb-* options).
        Log: ts, t_send, t_after, v, r, y, robot_ts and host_rx of the frame behind the y used."""
        a, P = self.args, self.args.period
        fbcol = APOS if a.servo_fb == "actual" else TPOS
        vmax = 0.95 * self.vlim
        q0 = self.start_pos
        sensor = sensor or SensorEmu.from_args(a)
        i0 = self.rec.n
        log = []
        aborted = False
        t0 = time.perf_counter() + 0.01
        k = 0
        try:
            while k * P <= dur:
                ts = t0 + k * P
                sleep_until(ts)
                fr = self.rec.latest()
                y = (fr[fbcol] if np.isfinite(fr[fbcol]) else fr[TPOS]) - q0
                t = k * P
                y, fr = sensor.observe(t, y, fr)
                r = ref(t)
                v = kp * (r - y) + (dref(t + adv) if dref else 0.0)
                v = min(max(v, -vmax), vmax)
                t1 = time.perf_counter()
                self.io.send_vel(v)
                t2 = time.perf_counter()
                log.append((ts, t1, t2, v, r, y, fr[ROBOT], fr[HOST]))
                if abs(r - y) > a.servo_abort:
                    print(f"    |error| = {abs(r - y):.2f} deg > --servo-abort {a.servo_abort}: run stopped")
                    aborted = True
                    break
                if k % 10 == 0:
                    self.guard()
                k += 1
        finally:
            self.io.send_vel(0.0)
        time.sleep(0.2)
        log = np.array(log).reshape(-1, 8)
        if len(log):
            self.send_logs.append(np.column_stack([log[:, :4], np.full((len(log), NJ), np.nan)]))
        return dict(log=log, frames=self.rec.frames(i0), q0=q0, aborted=aborted)

    def run_servo(self):
        a = self.args
        adv = a.servo_adv_ms * 1e-3 if a.servo_adv_ms is not None else self.model_delay()
        vmax = 0.95 * self.vlim
        out = dict(steps=[], sines=[], P=a.period, vmax=vmax, adv=adv, fb=a.servo_fb)
        S, t_on, hold = a.servo_step, 0.2, a.servo_hold
        for kp in a.servo_kps:
            print(f"  servo step {S:g} deg, Kp = {kp:g} 1/s (feedback: {a.servo_fb} position)")
            res = self._servo_loop(lambda t: S if t_on <= t < t_on + hold else 0.0, None, kp, 0.0,
                                   t_on + 2 * hold)
            out["steps"].append(dict(kp=kp, step=S, t_on=t_on, hold=hold, **res))
            self.home()
            if res["aborted"]:
                print("    higher Kp skipped")
                break
        for ff in ((False, True) if a.servo_ff else (False,)):
            for f in a.servo_freqs:
                w = 2 * math.pi * f
                A = min(a.servo_amp, 0.5 * vmax / w)
                settle = max(0.5, 1.0 / f)
                dur = settle + max(3.0 / f, 1.5)
                print(f"  servo sine {f:g} Hz, A = {A:.2f} deg, Kp = {a.servo_kp:g}"
                      + (f", feedforward {adv * 1e3:.1f} ms ahead" if ff else ""))
                res = self._servo_loop(lambda t: A * math.sin(w * t),
                                       (lambda t: A * w * math.cos(w * t)) if ff else None,
                                       a.servo_kp, adv if ff else 0.0, dur)
                out["sines"].append(dict(kp=a.servo_kp, ff=ff, f=f, A=A, settle=settle, **res))
                self.home()
        return out

    def _servo_multi_loop(self, A, w, ff, adv, dur):
        """Sine references in phase on every joint with A > 0 (servomulti)."""
        return self._track_loop(lambda t: A * math.sin(w * t),
                                (lambda t: A * w * math.cos(w * (t + adv))) if ff else None,
                                self.args.servo_kp, dur, A > 0)

    def _track_loop(self, ref_fn, ff_fn, kp, dur, active, sensor=None, keep_frames=True):
        """Host position loop on the `active` joints at once, every --period:
        v = kp (ref_fn(t) - y) + ff_fn(t), one MoveJointsVel for all joints. t = loop time [s].
        The positions go through `sensor` (default: --fb-* options).
        Log: ts, t_send, t_after, v1..v6, r1..r6, y1..y6, robot_ts and host_rx of the frame used.
        keep_frames=False (long runs): no copy of the frames, only their range i0:i1 in the recorder."""
        a, P = self.args, self.args.period
        kind = K_APOS if a.servo_fb == "actual" else K_TPOS
        vmax = 0.95 * self.vlim_all
        q0 = self.start_all.copy()
        sensor = sensor or SensorEmu.from_args(a)
        i0 = self.rec.n
        log = []
        aborted = False
        t0 = time.perf_counter() + 0.01
        k = 0
        try:
            while k * P <= dur:
                ts = t0 + k * P
                sleep_until(ts)
                fr = self.rec.latest()
                y = alljoints(fr, kind)
                y = np.where(np.isfinite(y), y, alljoints(fr, K_TPOS)) - q0
                t = k * P
                y, fr = sensor.observe(t, y, fr)
                r = ref_fn(t)
                v = kp * (r - y) + (ff_fn(t) if ff_fn else 0.0)
                v = np.where(active, np.clip(v, -vmax, vmax), 0.0)
                t1 = time.perf_counter()
                self.io.send_vel(v)
                t2 = time.perf_counter()
                log.append(np.concatenate([[ts, t1, t2], v, r, y, [fr[ROBOT], fr[HOST]]]))
                err = np.abs(r - y)[active]
                if np.nanmax(err) > a.servo_abort:
                    print(f"    |error| = {np.nanmax(err):.2f} deg > --servo-abort {a.servo_abort}: run stopped")
                    aborted = True
                    break
                if k % 10 == 0:
                    self.guard()
                k += 1
        finally:
            self.io.send_vel(0.0)
        time.sleep(0.2)
        log = np.array(log).reshape(-1, 3 + 3 * NJ + 2)
        if len(log):
            self.send_logs.append(np.column_stack([log[:, :3], log[:, 3 + self.j], log[:, 3:3 + NJ]]))
        if not keep_frames:
            return dict(log=log, i0=i0, i1=self.rec.n, q0=q0, aborted=aborted, t0=t0)
        return dict(log=log, frames=self.rec.frames(i0), q0=q0, aborted=aborted, t0=t0)

    def _pos_replay(self, R, dur):
        """Each action as one MoveJoints (blending on), sent when it is both due and released."""
        a, io = self.args, self.io
        io.set_joint_vel(a.pos_joint_vel_pct)
        io.set_blending(a.blending)
        q0 = self.start_all.copy()
        i0 = self.rec.n
        send_at = np.maximum(R.t, R.rel)
        log = []
        t0 = time.perf_counter() + 0.01
        try:
            for k in range(len(R.t)):
                ts = t0 + send_at[k]
                sleep_until(ts)
                t1 = time.perf_counter()
                io.move_joints(q0 + R.q[k])
                t2 = time.perf_counter()
                log.append(np.concatenate([[ts, t1, t2], R.q[k]]))
                if k % 5 == 0:
                    self.guard()
        except BaseException:
            io.clear_motion()
            raise
        t_last = time.perf_counter()
        try:
            io.wait_idle(max(a.pos_backlog_timeout, dur))
        except Exception as e:  # noqa: BLE001
            print(f"    queue not drained ({e}) -> ClearMotion")
            io.clear_motion()
        time.sleep(0.1)
        return dict(log=np.array(log).reshape(-1, 3 + NJ), frames=self.rec.frames(i0), q0=q0, aborted=False,
                    t0=t0, t_last=t_last)

    def run_replay(self):
        """Replay a 6-joint action stream (CSV --ref-file, or synthetic policy-like actions) through
        --replay-modes: vel = host loop + feedforward, ff = feedforward only, pos = MoveJoints stream."""
        a = self.args
        js = sorted({j - 1 for j in a.multi_joints})
        t, q_clean, src = make_reference(a, js, self.vlim_all)
        q, release = policy_effects(t, q_clean, a, js)
        q_cmd, rel_cmd, h_smooth = smooth_actions(q, release, a.ref_smooth, js)
        peak = float(np.nanmax(np.abs(q)))
        if peak > 0.8 * a.max_excursion:
            raise SafetyStop(f"replay reference reaches {peak:.1f} deg from start (> 80 % of --max-excursion)")
        P_act = float(np.median(np.diff(t)))
        adv = a.servo_adv_ms * 1e-3 if a.servo_adv_ms is not None else self.model_delay()
        Pi = a.pos_interp_ms * 1e-3
        adv_pos = a.posi_adv_ms * 1e-3 if a.posi_adv_ms is not None else self.model_delay() - a.period / 2 + Pi / 2
        # the controller tracks the smoothed actions (the raw ones without --ref-smooth)
        R = ActionRef(t, q_cmd, rel_cmd, horizon=max(2 * P_act, adv + 2 * a.period, adv_pos + 2 * Pi) + h_smooth * P_act,
                      order=2 if a.ref_predict == "quad" else 1)
        active = np.zeros(NJ, bool)
        active[js] = True
        out = dict(joints=js, t=t, q=q, release=release, q_clean=q_clean, smooth=2 * h_smooth + 1 if h_smooth else 0,
                   source=src, P_action=P_act, chunk=a.ref_chunk,
                   noise_deg=a.ref_noise_deg, jitter_ms=a.ref_jitter_ms, chunk_offset_deg=a.ref_chunk_offset_deg,
                   lead_ms=a.ref_lead_ms, predict=a.ref_predict, posi_period=Pi, posi_adv=adv_pos,
                   kp=a.servo_kp, adv=adv, P=a.period, vmax=0.95 * self.vlim_all, fb=a.servo_fb,
                   settle=a.replay_settle, runs=[])
        dur = float(t[-1]) + 0.5
        print(f"  {src} reference: {len(t)} actions every {P_act * 1e3:.0f} ms on J"
              f"{', J'.join(str(j + 1) for j in js)}, chunks of {a.ref_chunk}, noise {a.ref_noise_deg:g} deg, "
              f"jitter {a.ref_jitter_ms:g} ms, chunk offset {a.ref_chunk_offset_deg:g} deg, peak {peak:.2f} deg"
              + (f", smoothed over {2 * h_smooth + 1} actions" if h_smooth else ""))
        for mode in a.replay_modes:
            if mode in ("vel", "ff"):
                kp = a.servo_kp if mode == "vel" else 0.0
                print(f"  replay through MoveJointsVel: " + (f"Kp {kp:g} + feedforward {adv * 1e3:.1f} ms ahead"
                                                            if kp else f"feedforward only, {adv * 1e3:.1f} ms ahead"))
                res = self._track_loop(lambda tt: R.at(tt, tt), lambda tt: R.slope(tt + adv, tt), kp, dur, active)
            elif mode == "pos":
                print(f"  replay through MoveJoints (blending {a.blending:g} %, SetJointVel {a.pos_joint_vel_pct:g} %)")
                res = self._pos_replay(R, dur)
            elif mode == "posi":
                print(f"  replay through MoveJoints every {Pi * 1e3:g} ms, interpolated on the host, "
                      f"{adv_pos * 1e3:.1f} ms ahead")
                res = self._posi_replay(R, dur, Pi, adv_pos)
            else:
                print(f"  unknown replay mode '{mode}', skipped")
                continue
            out["runs"].append(dict(mode=mode, **res))
            self.home()
            if res["aborted"]:
                print(f"    '{mode}' stopped early (error above --servo-abort); going on with the next interface")
        return out

    def _posi_replay(self, R, dur, Pi, adv):
        """MoveJoints every Pi to the position the host interpolates from the released actions,
        adv ahead (the robot's own delay on a fast MoveJoints stream)."""
        a, io = self.args, self.io
        io.set_joint_vel(a.pos_joint_vel_pct)
        io.set_blending(a.blending)
        q0 = self.start_all.copy()
        i0 = self.rec.n
        log = []
        t0 = time.perf_counter() + 0.01
        k = 0
        try:
            while k * Pi <= dur:
                ts = t0 + k * Pi
                sleep_until(ts)
                qk = R.at(k * Pi + adv, k * Pi)
                t1 = time.perf_counter()
                io.move_joints(q0 + qk)
                t2 = time.perf_counter()
                log.append(np.concatenate([[ts, t1, t2], qk]))
                if k % 10 == 0:
                    self.guard()
                k += 1
        except BaseException:
            io.clear_motion()
            raise
        t_last = time.perf_counter()
        try:
            io.wait_idle(max(a.pos_backlog_timeout, dur))
        except Exception as e:  # noqa: BLE001
            print(f"    queue not drained ({e}) -> ClearMotion")
            io.clear_motion()
        time.sleep(0.1)
        return dict(log=np.array(log).reshape(-1, 3 + NJ), frames=self.rec.frames(i0), q0=q0, aborted=False,
                    t0=t0, t_last=t_last)

    def run_sensorsweep(self):
        """Host loop on the tested joint through an emulated sensor, for each --sensor-rates x
        --sensor-delays: Kp set from the model for --sensor-pm of phase margin, then a step and
        sines at --sensor-freqs (feedback only: what the sensor limits)."""
        a = self.args
        qm = self.quick_model or {}
        tau = (qm.get("tau_ms") or J6_TAU_MS) * 1e-3
        T = (qm.get("T_ms") or max(J6_T_FLOOR_MS, J6_T_MS_PER_PCT * a.vel_limit_pct)) * 1e-3
        lag = max(0.0, self.quick_servo_lag or 0.0) * 1e-3
        vmax = 0.95 * self.vlim
        S, t_on, hold = a.servo_step, 0.2, a.servo_hold
        out = dict(configs=[], pm=a.sensor_pm, noise_deg=a.fb_noise_deg)
        for rate in a.sensor_rates:
            for dly in a.sensor_delays:
                m = dict(tau=tau, T=T, P=a.period, tau_s=lag, fb_actual=a.servo_fb == "actual",
                         tau_b=a.sensor_base_fb_ms * 1e-3 + dly * 1e-3 + (0.5 / rate if rate > 0 else 0.0))
                kp = float(min(kp_for_pm(a.sensor_pm, m)["kp"], 60.0))
                print(f"  sensor {'every frame' if rate <= 0 else f'{rate:g} Hz'} + {dly:g} ms: "
                      f"Kp {kp:.1f} 1/s for {a.sensor_pm:g} deg of phase margin")
                steps, sines = [], []
                res = self._servo_loop(lambda t: S if t_on <= t < t_on + hold else 0.0, None, kp, 0.0,
                                       t_on + 2 * hold, sensor=SensorEmu(dly * 1e-3, rate, a.fb_noise_deg, a.ref_seed))
                steps.append(dict(kp=kp, step=S, t_on=t_on, hold=hold, **res))
                self.home()
                for f in a.sensor_freqs:
                    w = 2 * math.pi * f
                    A = min(a.servo_amp, 0.5 * vmax / w)
                    settle = max(0.5, 1.0 / f)
                    res = self._servo_loop(lambda t: A * math.sin(w * t), None, kp, 0.0, settle + max(3.0 / f, 1.5),
                                           sensor=SensorEmu(dly * 1e-3, rate, a.fb_noise_deg, a.ref_seed))
                    sines.append(dict(kp=kp, ff=False, f=f, A=A, settle=settle, **res))
                    self.home()
                out["configs"].append(dict(rate=rate, delay_ms=dly, kp=kp,
                                           servo=dict(steps=steps, sines=sines, P=a.period, vmax=vmax, adv=0.0,
                                                      fb=a.servo_fb)))
        return out

    # -- Cartesian velocity -------------------------------------------------------
    def _goto(self, q):
        """Stop, MoveJoints to q (absolute joint angles) and make it the reference pose of the guard."""
        self.io.stop()
        self.io.move_joints(q)
        self.io.wait_idle(60)
        time.sleep(0.2)
        fr = self.rec.latest()
        self.start_all = alljoints(fr, K_TPOS).copy()
        self.start_pos = float(fr[TPOS])
        self.cart_start = alljoints(fr, K_TCART).copy()

    def cart_guard(self):
        a = self.args
        fr = self.rec.latest()
        if fr is None or getattr(self, "cart_start", None) is None or not np.isfinite(self.cart_start).all():
            return
        d = alljoints(fr, K_TCART) - self.cart_start
        lin = float(np.linalg.norm(d[:3]))
        ang = float(np.max(np.abs((d[3:] + 180) % 360 - 180)))
        if lin > 1.5 * a.cart_max_mm or ang > 1.5 * a.cart_max_deg:
            raise SafetyStop(f"tool moved {lin:.1f} mm / {ang:.1f} deg from its start "
                             f"(limits {a.cart_max_mm:g} mm, {a.cart_max_deg:g} deg, x1.5)")

    def stream_cart(self, profile, duration, frame):
        """Send the Cartesian velocity profile(t) (6-vector) every --period. Returns (cmd log, frames)."""
        P = self.args.period
        n = int(round(duration / P))
        log = np.full((n, 4 + NJ), np.nan)
        i0 = self.rec.n
        t0 = time.perf_counter() + 0.01
        done = 0
        try:
            for k in range(n):
                ts = t0 + k * P
                sleep_until(ts)
                v = np.asarray(profile(k * P), float)
                t1 = time.perf_counter()
                self.io.send_cart_vel(v, frame)
                t2 = time.perf_counter()
                log[k, :4] = (ts, t1, t2, v[int(np.argmax(np.abs(v)))] if np.any(v) else 0.0)
                log[k, 4:] = v
                done = k + 1
                if k % 5 == 0:
                    self.guard()
                    self.cart_guard()
        finally:
            self.io.send_cart_vel(np.zeros(NJ), frame)
        time.sleep(0.15)
        log = log[:done]
        self.send_logs.append(log)
        return log, self.rec.frames(i0)

    def run_cartvel(self):
        """Cartesian velocity steps (MoveLinVelWrf / Trf) along --cart-axes from --cart-pose (a pose
        away from the wrist singularity of the home position), at each --cart-acc-pcts: delay and
        moving average of the Cartesian interface, as for the joints."""
        a, io = self.args, self.io
        names = ["x", "y", "z", "rx", "ry", "rz"]
        pose = np.asarray(a.cart_pose, float)
        base_acc = io.get_cart_acc()
        accs = a.cart_acc_pcts or [None]
        runs = []
        print(f"  start pose {np.round(pose, 1)} deg, frame {a.cart_frame.upper()}, SetCartAcc before: {base_acc}")
        try:
            self._goto(pose)
            for acc in accs:
                if acc is not None:
                    print(f"  SetCartAcc({acc:g})")
                    if not io.set_cart_acc(acc):
                        continue
                for ax in a.cart_axes:
                    k = names.index(ax)
                    lin = k < 3
                    A = a.cart_amp_mm if lin else a.cart_amp_deg
                    hold = min(a.step_hold, 0.7 * (a.cart_max_mm if lin else a.cart_max_deg) / A)
                    shape = piecewise([(0.15, 0.0), (hold, 1.0), (hold, 0.0), (hold, -1.0), (0.35, 0.0)])
                    unit = np.zeros(NJ)
                    unit[k] = A
                    print(f"  {ax}: +/-{A:g} {'mm/s' if lin else 'deg/s'}, hold {hold * 1e3:.0f} ms")
                    for _ in range(max(2, a.repeats // 2)):
                        cmd, fr = self.stream_cart(lambda t, u=unit: shape(t) * u, 0.5 + 3 * hold, a.cart_frame)
                        if io.recover_if_error():
                            print("    the robot refused the motion (error reset): axis skipped")
                            self._goto(pose)
                            break
                        runs.append(dict(axis=ax, k=k, A=A, acc=acc, cmd=cmd, frames=fr))
                        self._goto(pose)
        finally:
            if a.cart_acc_pcts and base_acc is not None:
                io.set_cart_acc(base_acc)
            self.cart_start = None
            self.home()
        return dict(frame=a.cart_frame, pose=pose, cart_acc_before=base_acc, runs=runs)

    def run_servomulti(self):
        """Host position loop on all --multi-joints at once: sine references in phase, with and
        without feedforward, at --servomulti-freqs."""
        a = self.args
        js = sorted({j - 1 for j in a.multi_joints})
        vmax = 0.95 * self.vlim_all
        adv = a.servo_adv_ms * 1e-3 if a.servo_adv_ms is not None else self.model_delay()
        out = dict(joints=js, runs=[], P=a.period, vmax=vmax, adv=adv, kp=a.servo_kp, fb=a.servo_fb)
        for ff in ((False, True) if a.servo_ff else (False,)):
            for f in a.servomulti_freqs:
                w = 2 * math.pi * f
                A = np.zeros(NJ)
                A[js] = np.minimum(a.servo_amp, 0.5 * vmax[js] / w)
                settle = max(0.5, 1.0 / f)
                dur = settle + max(3.0 / f, 1.5)
                print(f"  servo on J{', J'.join(str(j + 1) for j in js)}: sine {f:g} Hz, Kp = {a.servo_kp:g}, "
                      f"A = {', '.join(f'{A[j]:.2f}' for j in js)} deg" + (f", feedforward {adv * 1e3:.1f} ms" if ff else ""))
                res = self._servo_multi_loop(A, w, ff, adv, dur)
                out["runs"].append(dict(f=f, ff=ff, A=A, settle=settle, **res))
                self.home()
                if res["aborted"]:
                    return out
        return out

    def run_soak(self):
        """Worst cases for teleoperation: the host loop (Kp --servo-kp + feedforward) on all
        --multi-joints for --soak-dur s under each --soak-loads, tracking a slow random-looking motion
        (3 sines per joint, --soak-amp). The analysis looks at the tails, not the medians."""
        a = self.args
        js = sorted({j - 1 for j in a.multi_joints})
        active = np.zeros(NJ, bool)
        active[js] = True
        vmax = 0.95 * self.vlim_all
        adv = a.servo_adv_ms * 1e-3 if a.servo_adv_ms is not None else self.model_delay()
        freqs = np.array([0.2, 0.45, 0.9])
        ph = np.random.default_rng(a.ref_seed).uniform(0, 2 * np.pi, (NJ, len(freqs)))
        A = np.zeros(NJ)
        A[js] = np.minimum(a.soak_amp, 0.5 * vmax[js] / (2 * np.pi * freqs.mean()))
        dur = float(a.soak_dur)

        def ref(t):
            w = min(max(min(t, dur - t) / 0.5, 0.0), 1.0)          # 0.5 s smooth start and stop
            w = 0.5 - 0.5 * math.cos(math.pi * w)
            return w * (A / len(freqs)) * np.sin(2 * np.pi * freqs * t + ph).sum(axis=1)

        def dref(t, h=1e-3):
            return (ref(t + adv + h) - ref(t + adv - h)) / (2 * h)

        out = dict(joints=js, A=A, freqs=freqs, kp=a.servo_kp, adv=adv, P=a.period, vmax=vmax, fb=a.servo_fb,
                   dur=dur, vel_timeout=a.vel_timeout, phases=[])
        for load in a.soak_loads:
            gen = LoadGen(load, a.stress_procs, max_s=dur + 60)
            print(f"  soak {dur:g} s, load '{load}'" + (f" ({gen.procs} busy processes)" if load == "cpu" else "")
                  + f": J{', J'.join(str(j + 1) for j in js)}, A {', '.join(f'{A[j]:.2f}' for j in js)} deg, "
                  f"Kp {a.servo_kp:g}, feedforward {adv * 1e3:.1f} ms")
            with gen:
                res = self._track_loop(ref, dref, a.servo_kp, dur, active, keep_frames=False)
            out["phases"].append(dict(load=load, procs=gen.procs if load == "cpu" else 0, **res))
            self.home()
            if res["aborted"]:
                print("    soak stopped (error above --servo-abort)")
                break
        return out


# --------------------------------------------------------------------------- #
# Analysis: generic helpers                                                   #
# --------------------------------------------------------------------------- #
def clock_map(host, robot, win=1.0):
    """Robot timestamp -> host clock: lower envelope of (host_rx - robot_ts), with linear drift.
    Robust to host timestamps taken too early (a thread pre-empted between reading the clock and
    the data): per-window minima that sit off the drift line are rejected before the fit, so one
    bad frame cannot shift the whole run (it did by ~8 ms on 2026-11-25 with v2.0)."""
    ok = np.isfinite(host) & np.isfinite(robot)
    h, r = host[ok], robot[ok]
    d = h - r
    r0 = r[0]
    if r[-1] - r0 < 4 * win:
        off = max(float(d.min()), float(np.percentile(d, 1)) - 1e-3)
        return lambda x: np.asarray(x) + off
    b = ((r - r0) // win).astype(int)
    starts = np.r_[0, np.nonzero(np.diff(b))[0] + 1, len(b)]
    xs, ys = [], []
    for a, z in zip(starts[:-1], starts[1:]):
        if z - a < 20:  # too few frames to trust the window minimum
            continue
        i = a + int(np.argmin(d[a:z]))
        xs.append(r[i] - r0)
        ys.append(d[i])
    xs, ys = np.array(xs), np.array(ys)
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        p = np.polyfit(xs[keep], ys[keep], 1)
        res = ys - np.polyval(p, xs)
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res - np.median(res[keep])) <= max(4 * mad, 3e-4)
    p[1] += min(0.0, float(res[keep].min()))
    return lambda x: np.asarray(x) + np.polyval(p, np.asarray(x) - r0)


def _hull_line(x, d, bin_s=0.01):
    """(slope, intercept) of the edge of the lower convex hull of (x, d) at the mean x: the highest
    line under every point. Points are thinned to the minimum of each bin_s bin first."""
    b = np.floor((x - x[0]) / bin_s).astype(np.int64)
    order = np.lexsort((d, b))
    _, first = np.unique(b[order], return_index=True)
    idx = order[first]                      # one point per bin: its minimum, in time order
    hull = []
    for px, py in zip(x[idx].tolist(), d[idx].tolist()):
        while len(hull) >= 2:
            (ax, ay), (bx, by) = hull[-2], hull[-1]
            if (bx - ax) * (py - ay) - (by - ay) * (px - ax) <= 0:   # b is not below a -> p
                hull.pop()
            else:
                break
        hull.append((px, py))
    xm = float(np.mean(x))
    slope, icpt = 0.0, hull[0][1]
    for (ax, ay), (bx, by) in zip(hull[:-1], hull[1:]):
        if ax <= xm <= bx and bx > ax:
            slope = (by - ay) / (bx - ax)
            icpt = ay - slope * ax
            break
    else:
        if len(hull) > 1 and xm > hull[-1][0]:
            icpt = hull[-1][1]
    return slope, icpt


def clock_map_ecat(host, robot, bin_s=0.01, stall_s=0.002):
    """Robot timestamp -> host clock for EtherCAT runs (raw['_ecat'], written by meca_ecat).

    The master and the robot both tick every ~1 ms, each on its own clock, so the age of the data
    a frame brings (host_rx - robot_ts) is not a random transport delay as over TCP: it is a
    sawtooth up to one cycle high that wraps every 1 / drift (a 'beat', ~15 s at 65 ppm). Minima
    over 1 s windows (clock_map) follow that sawtooth and can sit up to one cycle late.
    A sample cannot reach the host before the robot stamped it, so the map rests on the highest
    line under the points (host_rx - robot_ts against robot_ts): the edge of their lower convex
    hull. Over at least one beat it lies on the bottoms of the sawtooth (offset + drift + minimum
    transport, like clock_map over TCP); over less, it can be up to one cycle late.

    The Meca500 timestamp also STALLS: it stops while the exchanges go on, then resumes where it
    stopped (first run 2026-11-26, fw 11.3.3: 25 stalls, 1.28 s in 214 s, mostly during activation
    and homing, 3-60 ms during tests). It also jumps FORWARD at times (a robot gap of 17 ms between
    two consecutive samples 1 ms apart on the host). Each stall makes host_rx - robot_ts jump up,
    each forward jump makes it jump down. So the session is cut at every jump above stall_s, either
    way; the drift (the robot clock rate) comes from the longest segment, and each segment gets its
    own offset: its lowest point once the drift is removed. The returned function carries drift
    (s/s), span (s, longest segment), stalls / stall_lost (s) and jumps / jump_gained (s) for
    ecat_timing."""
    ok = np.isfinite(host) & np.isfinite(robot)
    h, r = np.asarray(host, float)[ok], np.asarray(robot, float)[ok]
    r0 = float(r[0])
    x, d = r - r0, h - r
    dd = np.diff(d)
    jump = np.abs(dd) > stall_s
    starts = np.r_[0, np.nonzero(jump)[0] + 1]
    ends = np.r_[starts[1:], len(d)]
    seg_len = x[ends - 1] - x[starts]
    L = int(np.argmax(seg_len))
    if ends[L] - starts[L] > 2:
        slope, _ = _hull_line(x[starts[L]:ends[L]], d[starts[L]:ends[L]], bin_s)
    else:
        slope = 0.0
    offs = np.array([float(np.min(d[a:b] - slope * x[a:b])) for a, b in zip(starts, ends)])
    seg_x0 = x[starts]

    def cm(xx):
        xx = np.asarray(xx, float)
        k = np.clip(np.searchsorted(seg_x0, xx - r0, side="right") - 1, 0, len(offs) - 1)
        return xx + offs[k] + slope * (xx - r0)

    up, down = dd > stall_s, dd < -stall_s
    cm.drift, cm.span = float(slope), float(seg_len[L])
    cm.stalls, cm.stall_lost = int(up.sum()), float(np.sum(dd[up]))
    cm.jumps, cm.jump_gained = int(down.sum()), float(-np.sum(dd[down]))
    return cm


def run_clock_map(raw, frames):
    """The clock map of a run: the EtherCAT one when meca_ecat recorded it, else the TCP one."""
    if isinstance(raw, dict) and raw.get("_ecat"):
        return clock_map_ecat(frames[:, HOST], frames[:, ROBOT])
    return clock_map(frames[:, HOST], frames[:, ROBOT])


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


def stats(x, scale=1.0):
    x = np.asarray([v for v in np.ravel(x) if v is not None and np.isfinite(v)], float) * scale
    if x.size == 0:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), mean=float(x.mean()),
                p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)),
                min=float(x.min()), max=float(x.max()))


def nmed(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else math.nan


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


def sine_vs(ct, cv, t, y, w, ta, tb, t0):
    """Gain, phase lag [rad] and relative residual of y(t) against the command (ct, cv)."""
    mc = (ct >= ta) & (ct <= tb)
    Au, phu, _ = sine_fit(ct[mc] - t0, cv[mc], w)
    mm = (t >= ta) & (t <= tb)
    Ay, phy, nl = sine_fit(t[mm] - t0, y[mm], w)
    lag = (phu - phy) % (2 * math.pi)
    if lag > 1.75 * math.pi:
        lag -= 2 * math.pi
    return Ay / Au, lag, nl


def cmd_edges(c):
    v = c[:, 3]
    k = np.nonzero(np.diff(v) != 0)[0] + 1
    return [(float(c[i, 1]), float(v[i - 1]), float(v[i])) for i in k]


# --------------------------------------------------------------------------- #
# Velocity-mode model: pure delay + moving average                           #
# --------------------------------------------------------------------------- #
_MODEL_DT = 1e-4
_WARNED = set()


def _staircase_integral(ct, cv):
    """Integral of the zero-order-hold command; zero after the last sample (stream() sends 0 then)."""
    tg = np.arange(ct[0] - 0.5, ct[-1] + 0.6, _MODEL_DT)
    idx = np.searchsorted(ct, tg, side="right") - 1
    u = np.where(idx >= 0, cv[np.clip(idx, 0, None)], 0.0)
    u[tg > ct[-1] + 1e-3] = 0.0
    return tg, np.cumsum(u) * _MODEL_DT


def boxcar_response(t, integ, tau, T):
    tg, U = integ
    x = np.asarray(t) - tau
    return (np.interp(x, tg, U) - np.interp(x - T, tg, U)) / T


def fit_boxcar(ct, cv, tq, y, lo=None, hi=None):
    """Least squares fit of y(t) = mean of u over [t - tau - T, t - tau], u = staircase (ct, cv).
    Returns tau_ms, T_ms, rms [deg/s], span of the command, or None."""
    try:
        from scipy.optimize import minimize
    except ImportError:
        if "scipy" not in _WARNED:
            print("scipy is missing (pip install scipy): model fits skipped")
            _WARNED.add("scipy")
        return None
    ct, cv, tq, y = (np.asarray(x, float) for x in (ct, cv, tq, y))
    if len(ct) < 3 or np.ptp(cv) == 0:
        return None
    lo = ct[0] - 0.02 if lo is None else lo
    hi = ct[-1] + 0.15 if hi is None else hi
    w = (tq >= lo) & (tq <= hi) & np.isfinite(y)
    if w.sum() < 20:
        return None
    integ = _staircase_integral(ct, cv)
    tq, y = tq[w], y[w]

    def cost(p):
        T = max(abs(p[1]), 0.2) * 1e-3
        return float(np.mean((boxcar_response(tq, integ, p[0] * 1e-3, T) - y) ** 2))

    best = None
    for T0 in (8.0, 20.0, 50.0, 120.0):
        s = minimize(cost, [3.0, T0], method="Nelder-Mead",
                     options=dict(xatol=1e-3, fatol=1e-10, maxiter=3000))
        if best is None or s.fun < best.fun:
            best = s
    return dict(tau_ms=float(best.x[0]), T_ms=float(max(abs(best.x[1]), 0.2)),
                rms=float(math.sqrt(best.fun)), span=float(np.ptp(cv)))


def model_summary(fits):
    fits = [f for f in fits if f]
    if not fits:
        return None
    tau = float(np.median([f["tau_ms"] for f in fits]))
    T = float(np.median([f["T_ms"] for f in fits]))
    return dict(n=len(fits), tau_ms=tau, T_ms=T,
                tau_p5_p95_ms=[float(np.percentile([f["tau_ms"] for f in fits], q)) for q in (5, 95)],
                T_p5_p95_ms=[float(np.percentile([f["T_ms"] for f in fits], q)) for q in (5, 95)],
                rms=float(np.median([f["rms"] for f in fits])),
                eq_delay_ms=tau + T / 2, bw_3db_hz=443.0 / T, first_zero_hz=1000.0 / T)


def vel_frf(f, tau, T, P=0.0):
    """Velocity command -> target velocity: moving average T, delay tau (+ P/2 of zero-order hold)."""
    f = np.asarray(f, float)
    return np.sinc(f * T) * np.exp(-1j * np.pi * f * T) * np.exp(-2j * np.pi * f * (tau + P / 2))


def _plant(f, m):
    """Velocity command -> (target position, fed-back position). Actual position lags the target
    by tau_s (servo); it is the one fed back when m['fb_actual']."""
    s = 2j * np.pi * np.asarray(f, float)
    G = vel_frf(f, m["tau"], m["T"], m["P"]) / s
    Ga = G * np.exp(-s * m.get("tau_s", 0.0))
    return s, Ga, (Ga if m.get("fb_actual", True) else G)


def loop_frf(f, kp, m):
    s, _, Gfb = _plant(f, m)
    return kp * Gfb * np.exp(-s * m["tau_b"])


def closed_loop_frf(f, kp, m, ff=False, adv=0.0):
    """Reference position -> actual position for v = kp (r - y delayed by tau_b) [+ dr/dt(t + adv)]."""
    s, Ga, Gfb = _plant(f, m)
    C = kp + (s * np.exp(s * adv) if ff else 0.0)
    return Ga * C / (1 + kp * Gfb * np.exp(-s * m["tau_b"]))


def margins(kp, m):
    f = np.logspace(-2, 2.5, 6000)
    L = loop_frf(f, kp, m)
    mag, ph = np.abs(L), np.degrees(np.unwrap(np.angle(L)))
    out = dict(kp=float(kp), fc_hz=math.nan, pm_deg=math.nan, f180_hz=math.nan, gm_db=math.nan)
    i = np.nonzero(mag < 1)[0]
    if len(i):
        out.update(fc_hz=float(f[i[0]]), pm_deg=float(180 + ph[i[0]]))
    k = np.nonzero(ph <= -180)[0]
    if len(k):
        out.update(f180_hz=float(f[k[0]]), gm_db=float(-20 * np.log10(mag[k[0]])))
    return out


def kp_for_pm(pm, m):
    kps = np.logspace(-0.5, 2.7, 200)
    res = [margins(k, m) for k in kps]
    i = int(np.nanargmin([abs(r["pm_deg"] - pm) if np.isfinite(r["pm_deg"]) else np.inf for r in res]))
    return res[i]


def sim_servo(ref, dref, kp, adv, m, dur, vmax, dt=2e-4):
    """Time-domain prediction of the host position loop on the identified model: the command
    computed every P on y(t - tau_b), applied after tau, moving average T, integrated; the
    actual position is the target delayed by tau_s. Returns t, actual position."""
    n = int(dur / dt) + 1
    t = np.arange(n) * dt
    y, U = np.zeros(n), np.zeros(n)
    NT = max(1, int(round(m["T"] / dt)))
    nd, nb = int(round(m["tau"] / dt)), int(round(m["tau_b"] / dt))
    ns = int(round(m.get("tau_s", 0.0) / dt))
    nfb = nb + (ns if m.get("fb_actual", True) else 0)
    next_send, pending, u = 0.0, [], 0.0
    for i in range(n - 1):
        if t[i] >= next_send - 1e-12:
            v = kp * (ref(t[i]) - y[max(0, i - nfb)]) + (dref(t[i] + adv) if dref else 0.0)
            pending.append((i + nd, min(max(v, -vmax), vmax)))
            next_send += m["P"]
        while pending and pending[0][0] <= i:
            u = pending.pop(0)[1]
        U[i + 1] = U[i] + u * dt
        y[i + 1] = y[i] + (U[i + 1] - U[max(0, i + 1 - NT)]) / (NT * dt) * dt
    ya = np.concatenate([np.zeros(ns), y[:n - ns]]) if ns else y
    return t, ya


def pick_model(S, args):
    """(tau, T, tau_s) for predictions: this run's step test, else --model-tau/T, else the J6 law.
    tau_s = target -> actual lag at 50 % of the velocity steps (0 without a step test)."""
    step = S.get("step") or {}
    lag = (step.get("servo_lag_target_to_actual_ms") or {}).get("median")
    base = dict(tau_s=max(0.0, lag or 0.0) * 1e-3, fb_actual=getattr(args, "servo_fb", "actual") == "actual")
    st = step.get("model")
    if st:
        return dict(base, tau=st["tau_ms"] * 1e-3, T=st["T_ms"] * 1e-3, src="step test of this run")
    if getattr(args, "model_tau", None) is not None and getattr(args, "model_T", None) is not None:
        return dict(base, tau=args.model_tau * 1e-3, T=args.model_T * 1e-3, src="--model-tau / --model-T")
    T = max(J6_T_FLOOR_MS, J6_T_MS_PER_PCT * args.vel_limit_pct)
    return dict(base, tau=J6_TAU_MS * 1e-3, T=T * 1e-3,
                src="J6 law of 2026-11-24 (no step test in this run: run 'step' for real numbers)")


# --------------------------------------------------------------------------- #
# Analysis: velocity mode                                                     #
# --------------------------------------------------------------------------- #
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


def analyze_step(res, cmap, fit=True):
    rows = []
    fits_t, fits_a = [], []
    for run in res["runs"]:
        c, d = run["cmd"], prep(run["frames"], cmap)
        run["d"] = d
        if fit:
            run["fit"] = fit_boxcar(c[:, 1], c[:, 3], d["m"], d["tv"])
            fits_t.append(run["fit"])
            if has(d, "av"):
                fits_a.append(fit_boxcar(c[:, 1], c[:, 3], d["m"], d["av"]))
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

    def colv(sig, key):
        return [r[sig][key] for r in rows if r.get(sig)]

    summ = dict(amp=res["amp"], n_edges=len(rows))
    for sig, name in (("tv", "target"), ("av", "actual")):
        if not any(r.get(sig) for r in rows):
            continue
        summ[name] = {
            "dead_ms": stats(colv(sig, "dead"), 1e3),
            "dead_host_seen_ms": stats([r.get(sig + "_host") for r in rows], 1e3),
            "t50_ms": stats(colv(sig, "t50"), 1e3),
            "rise_10_90_ms": stats(colv(sig, "rise"), 1e3),
            "accel_deg_s2": stats(colv(sig, "acc")),
            "overshoot_pct": stats(colv(sig, "overshoot"), 100),
            "settle_ms": stats(colv(sig, "settle"), 1e3),
        }
    summ["servo_lag_target_to_actual_ms"] = stats([r.get("servo_lag") for r in rows], 1e3)
    if fit:
        summ["model"] = model_summary(fits_t)
        summ["model_on_actual_vel"] = model_summary(fits_a)
    res["rows"] = rows
    return summ


def quick_accel(res, cmap=None):
    """(target acceleration [deg/s^2], target -> actual lag at 50 % [ms]) from a step test, None if unknown.
    cmap: clock map to use (meca_ecat passes the EtherCAT one); default: built from the step frames."""
    try:
        allf = np.vstack([r["frames"] for r in res["runs"]])
        cm = cmap or clock_map(allf[:, HOST], allf[:, ROBOT])
        s = analyze_step(dict(res), cm, fit=False)
        lag = s.get("servo_lag_target_to_actual_ms")
        return s["target"]["accel_deg_s2"]["median"], (lag["median"] if lag else None)
    except Exception:  # noqa: BLE001
        return None, None


def quick_model(res, cmap=None):
    try:
        allf = np.vstack([r["frames"] for r in res["runs"]])
        cm = cmap or clock_map(allf[:, HOST], allf[:, ROBOT])
        return model_summary([fit_boxcar(r["cmd"][:, 1], r["cmd"][:, 3], cm(r["frames"][:, ROBOT]),
                                         r["frames"][:, TVEL]) for r in res["runs"][:2]])
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
    gaps = np.diff(r)
    out = dict(robot_frame_period_ms=stats(gaps, 1e3),
               host_rx_interval_ms=stats(np.diff(h), 1e3),
               feedback_excess_latency_ms=stats(h - m, 1e3))
    if gaps.size:
        k = np.maximum(1, np.round(gaps / np.median(gaps)))
        out["missing_frames_pct"] = float(100 * np.sum(k - 1) / max(1.0, np.sum(k)))
    c = np.vstack([x[:, :4] for x in send_logs]) if send_logs else None
    if c is not None:
        out["send_lateness_ms"] = stats(c[:, 1] - c[:, 0], 1e3)
        out["MoveJointsVel_call_ms"] = stats(c[:, 2] - c[:, 1], 1e3)
    return out, dict(rdt=gaps, hdt=np.diff(h), lat=h - m,
                     late=(c[:, 1] - c[:, 0]) if c is not None else None,
                     call=(c[:, 2] - c[:, 1]) if c is not None else None)


ECAT_CYCLE_COLS = ("ts", "t1", "t2", "robot_ts", "fresh", "wkc")


def analyze_ecat_timing(ecat, frames, cmap):
    """EtherCAT bus of a meca_ecat run. bus_cycles holds one row per cycle: deadline ts, t1 just
    before the exchange, t2 when it returned, robot timestamp (NaN if the exchange failed), 1 if
    it brought a new robot sample, working counter (-1 if the exchange failed)."""
    B = np.asarray(ecat.get("bus_cycles", np.empty((0, 6))), float).reshape(-1, 6)
    cyc = float(ecat.get("cycle_s") or 1e-3)
    out = dict(cycle_ms=cyc * 1e3, cycles=int(len(B)), byte_order=ecat.get("byte_order"),
               byte_order_confirmed=ecat.get("byte_order_confirmed"), overruns=ecat.get("overruns"),
               ring_drops=ecat.get("ring_drops"), sim=bool(ecat.get("sim", False)))
    if len(B) > 1:
        ts, t1, t2, _, fresh, wkc = B.T
        ok = wkc >= 0
        out["wake_late_ms"] = tail(t1 - ts, 1e3)          # includes the controller's own compute time
        out["txrx_roundtrip_ms"] = tail((t2 - t1)[ok], 1e3)
        out["cycle_interval_ms"] = tail(np.diff(t1), 1e3)
        out["fresh_pct"] = 100 * float(np.mean(fresh[ok] > 0)) if ok.any() else math.nan
        out["failed_exchanges"] = int(np.sum(~ok))
        exp = ecat.get("expected_wkc")
        out["bad_wkc"] = int(np.sum(ok & (wkc != exp))) if exp else None
    drift = getattr(cmap, "drift", None)
    if drift is not None:
        beat = cyc / abs(drift) if drift else math.inf
        out["drift_ppm"] = drift * 1e6
        out["beat_s"] = beat if math.isfinite(beat) else None
        out["beats_covered"] = cmap.span / beat if math.isfinite(beat) else 0.0
        out["robot_clock_map_ok"] = bool(out["beats_covered"] >= 1.0)
        out["robot_clock_stalls"] = getattr(cmap, "stalls", 0)
        out["robot_clock_stall_lost_s"] = getattr(cmap, "stall_lost", 0.0)
        out["robot_clock_jumps"] = getattr(cmap, "jumps", 0)
        out["robot_clock_jump_gained_s"] = getattr(cmap, "jump_gained", 0.0)
    if len(frames):
        out["data_age_ms"] = tail(frames[:, HOST] - cmap(frames[:, ROBOT]), 1e3)
    return out


def analyze_accsweep(res, cmap):
    rows = []
    for sw in res["sweeps"]:
        s = analyze_step(sw, cmap)
        row = dict(pct=sw["pct"])
        for name in ("target", "actual"):
            if name in s:
                for k in ("accel_deg_s2", "rise_10_90_ms", "dead_ms", "t50_ms"):
                    row[f"{name}_{k}"] = s[name][k]["median"] if s[name][k] else math.nan
        md = s.get("model") or {}
        row.update(T_ms=md.get("T_ms", math.nan), tau_ms=md.get("tau_ms", math.nan),
                   fit_rms=md.get("rms", math.nan))
        rows.append(row)
    res["rows"] = rows
    return rows


def analyze_jaccsweep(res, cmap):
    rows = []
    for sw in res["sweeps"]:
        if sw.get("failed"):
            rows.append(dict(vl=sw["vl"], acc=sw["acc"], failed=True))
            continue
        s = analyze_step(sw, cmap)
        md = s.get("model") or {}
        acc = s.get("target", {}).get("accel_deg_s2")
        rows.append(dict(vl=sw["vl"], acc=sw["acc"], T_ms=md.get("T_ms", math.nan),
                         tau_ms=md.get("tau_ms", math.nan), fit_rms=md.get("rms", math.nan),
                         target_accel_deg_s2=acc["median"] if acc else math.nan))
    res["rows"] = rows
    return rows


def analyze_ampsweep(res, cmap):
    rows = []
    for rr in res["runs"]:
        for run in rr["runs"]:
            c, fr = run["cmd"], run["frames"]
            m = cmap(fr[:, ROBOT])
            edges = cmd_edges(c)
            ends = [e[0] for e in edges[1:]] + [c[-1, 1] + 0.15]
            for (te, v0, v1), tend in zip(edges, ends):
                f = fit_boxcar(c[:, 1], c[:, 3], m, fr[:, TVEL], lo=te - 0.01, hi=tend)
                if f:
                    dv = abs(v1 - v0)
                    rows.append(dict(frac=rr["frac"], v0=v0, v1=v1, dv=dv, tau_ms=f["tau_ms"],
                                     T_ms=f["T_ms"], rms=f["rms"], rel_rms_pct=100 * f["rms"] / dv,
                                     acc_eff=dv / (f["T_ms"] * 1e-3)))
    res["rows"] = rows
    levels = sorted({round(r["dv"], 1) for r in rows})
    return [dict(dv=lv, n=len(g), T_ms=nmed([r["T_ms"] for r in g]), tau_ms=nmed([r["tau_ms"] for r in g]),
                 acc_eff=nmed([r["acc_eff"] for r in g]), rel_rms_pct=nmed([r["rel_rms_pct"] for r in g]))
            for lv in levels for g in [[r for r in rows if round(r["dv"], 1) == lv]]]


def analyze_multijoint(res, cmap):
    """Per joint and scenario: (tau, T) fitted on the target velocity, the same fit on the actual
    velocity, and the servo lag = difference of their equivalent delays (tau + T/2)."""
    rows = []
    for sc in res["scenarios"]:
        for j in sc["active"]:
            ft, fa = [], []
            for run in sc["runs"]:
                c, fr = run["cmd"], run["frames"]
                m = cmap(fr[:, ROBOT])
                ft.append(fit_boxcar(c[:, 1], c[:, 4 + j], m, fr[:, col(K_TVEL, j)]))
                if np.isfinite(fr[:, col(K_AVEL, j)]).any():
                    fa.append(fit_boxcar(c[:, 1], c[:, 4 + j], m, fr[:, col(K_AVEL, j)]))
            mt, ma = model_summary(ft) or {}, model_summary(fa) or {}
            row = dict(scenario=sc["name"], joint=j + 1, A=float(sc["A"][j]), **mt)
            if mt and ma:
                row.update(servo_lag_ms=ma["eq_delay_ms"] - mt["eq_delay_ms"], actual_T_ms=ma["T_ms"],
                           actual_fit_rms=ma["rms"])
            rows.append(row)
    res["rows"] = rows
    return rows


def analyze_servomulti(res, cmap, mdl):
    """Host loop on several joints at once: per joint and frequency, gain / phase / tracking error,
    next to the single-joint model prediction."""
    js, vmax = res["joints"], res["vmax"]
    runs = [r for r in res["runs"] if len(r["log"]) > 10]
    if not runs:
        return {}
    ages = np.concatenate([r["log"][:, 1] - cmap(r["log"][:, 21]) for r in runs])
    m = dict(tau=mdl["tau"], T=mdl["T"], P=res["P"], tau_b=float(np.nanmedian(ages)), tau_s=mdl["tau_s"],
             fb_actual=res["fb"] == "actual")
    rows = []
    for r in runs:
        L, fr = r["log"], r["frames"]
        tm = cmap(fr[:, ROBOT])
        w = 2 * math.pi * r["f"]
        t0, ta, tb = L[0, 1], L[0, 1] + r["settle"], L[-1, 1]
        H = closed_loop_frf(r["f"], res["kp"], m, r["ff"], res["adv"] if r["ff"] else 0.0)
        for j in js:
            ap, tp = fr[:, col(K_APOS, j)], fr[:, col(K_TPOS, j)]
            y = (ap if np.isfinite(ap).any() else tp) - r["q0"][j]
            ref = L[:, 9 + j]
            g, lag, nl = sine_vs(L[:, 1], ref, tm, y, w, ta, tb, t0)
            mm = (tm >= ta) & (tm <= tb) & np.isfinite(y)
            err = y[mm] - np.interp(tm[mm], L[:, 1], ref)
            rows.append(dict(f=r["f"], ff=r["ff"], joint=j + 1, A=float(r["A"][j]), gain=g,
                             phase_deg=-math.degrees(lag), nonlin=nl,
                             err_rms_pct=100 * float(np.sqrt(np.mean(err ** 2))) / r["A"][j],
                             sat_pct=100 * float(np.mean(np.abs(L[:, 3 + j]) >= 0.999 * vmax[j])),
                             model_gain=float(abs(H)), model_phase_deg=float(np.degrees(np.angle(H))),
                             model_err_rms_pct=float(100 * abs(1 - H) / math.sqrt(2))))
    res["rows"] = rows
    return dict(kp=res["kp"], feedforward_advance_ms=res["adv"] * 1e3, joints=[j + 1 for j in js],
                fb_age_ms=stats(ages, 1e3), model=dict(tau_ms=m["tau"] * 1e3, T_ms=m["T"] * 1e3,
                                                       servo_lag_ms=m["tau_s"] * 1e3, source=mdl["src"]),
                rows=rows, aborted=any(r["aborted"] for r in res["runs"]))


def analyze_vmaxcheck(res, cmap):
    rows = []
    for r in res["runs"]:
        if r.get("failed"):
            rows.append(dict(joint=r["joint"], failed=r["failed"]))
            continue
        j = r["joint"] - 1
        c, fr = r["cmd"], r["frames"]
        tm, tv = cmap(fr[:, ROBOT]), fr[:, col(K_TVEL, j)]

        def plateau(ts, sign):  # median of the second half of a hold
            if len(ts) < 5:
                return math.nan
            w = (tm >= ts[0] + 0.5 * (ts[-1] - ts[0])) & (tm <= ts[-1]) & np.isfinite(tv)
            return float(np.median(sign * tv[w])) if w.sum() > 3 else math.nan

        vp = plateau(c[c[:, 4 + j] > 0, 1], 1)
        vn = plateau(c[c[:, 4 + j] < 0, 1], -1)
        v = nmed([vp, vn])
        clamped = bool(np.isfinite(v) and v < 0.98 * r["v_cmd"])
        rows.append(dict(joint=r["joint"], vel_limit_pct=r["pct"], asked=r["v_cmd"], plateau=v, plateau_pos=vp,
                         plateau_neg=vn, clamped=clamped, vmax_assumed=JOINT_VMAX[j],
                         vmax_measured=(v / (r["pct"] / 100)) if clamped else math.nan))
    res["rows"] = rows
    return rows


def _best_shift(tt, y, R, j):
    """Delay d minimising RMS(y(t) - ref(t - d)): coarse 1 ms grid then 0.1 ms."""
    def rms_at(d):
        return float(np.sqrt(np.mean((y - R.true_at(tt - d, j)) ** 2)))
    grid = np.arange(-0.02, 0.2001, 0.001)
    d0 = grid[int(np.argmin([rms_at(d) for d in grid]))]
    fine = np.arange(d0 - 0.001, d0 + 0.0011, 0.0001)
    errs = [rms_at(d) for d in fine]
    k = int(np.argmin(errs))
    return float(fine[k]), errs[k]


def analyze_replay(res, cmap):
    """Per interface (mode) and joint, against the commanded trajectory (actions at their due time):
    RMS / max error, delay (best time shift) and the error left once that delay is removed, target
    acceleration RMS and stall % (smoothness), saturation and queue backlog."""
    R = ActionRef(res["t"], res["q"], res["release"], horizon=0.0)
    js, settle, t_end = res["joints"], res["settle"], float(res["t"][-1])
    # runs since 2026-11-25 evening keep the motion before noise / chunk offsets: what the policy meant
    Rc = ActionRef(res["t"], res["q_clean"], res["release"], horizon=0.0) if "q_clean" in res else None
    ta_ = np.asarray(res["t"], float)
    win_a = (ta_ >= settle) & (ta_ <= t_end - 0.2)
    rows = []
    for run in res["runs"]:
        fr, L = run["frames"], run["log"]
        if len(fr) < 50 or len(L) < 5:
            continue
        tm = cmap(fr[:, ROBOT])
        tt = tm - run["t0"]
        run["tt"] = tt
        w = (tt >= settle) & (tt <= t_end - 0.2)
        for j in js:
            ap = fr[:, col(K_APOS, j)]
            y = (ap if np.isfinite(ap).any() else fr[:, col(K_TPOS, j)]) - run["q0"][j]
            ok = w & np.isfinite(y)
            if ok.sum() < 50:
                continue
            ref = R.true_at(tt[ok], j)
            e = y[ok] - ref
            ref_rms = float(np.std(ref)) or math.nan
            d, shape = _best_shift(tt[ok], y[ok], R, j)
            tv = fr[ok, col(K_TVEL, j)]
            acc = np.gradient(tv, tm[ok]) if np.isfinite(tv).sum() > 10 else np.full(tv.shape, math.nan)
            rs = np.abs(R.true_slope(tt[ok] - d, j))
            fast = rs > 0.5 * np.max(rs) if np.max(rs) > 0 else np.zeros(rs.shape, bool)
            row = dict(mode=run["mode"], joint=j + 1, aborted=bool(run.get("aborted")), ref_rms_deg=ref_rms,
                       rms_err_deg=float(np.sqrt(np.mean(e ** 2))), max_err_deg=float(np.max(np.abs(e))),
                       rel_err_pct=100 * float(np.sqrt(np.mean(e ** 2))) / ref_rms, delay_ms=d * 1e3,
                       shape_err_deg=shape, shape_rel_pct=100 * shape / ref_rms,
                       accel_rms=float(np.sqrt(np.nanmean(acc ** 2))),
                       stall_pct=100 * float(np.mean(np.abs(tv[fast]) < 0.1 * np.max(rs))) if fast.any() else math.nan)
            if run["mode"] in ("vel", "ff"):
                row["sat_pct"] = 100 * float(np.mean(np.abs(L[:, 3 + j]) >= 0.999 * res["vmax"][j]))
            if run["mode"] in ("pos", "posi"):  # target still moving after the last new position was sent
                tv_all = fr[:, col(K_TVEL, j)]
                mv = np.isfinite(tv_all) & (np.abs(tv_all) > 0.2)
                ch = np.nonzero(np.abs(np.diff(L[:, 3 + j])) > 1e-9)[0]
                t_cmd = L[ch[-1] + 1, 1] if len(ch) else run["t_last"]
                row["backlog_ms"] = (float(tm[np.nonzero(mv)[0][-1]]) - t_cmd) * 1e3 if mv.any() else math.nan
            if Rc is not None:  # against the clean motion: how much of the noise / offsets reached the robot
                refc = Rc.true_at(tt[ok], j)
                c_rms = float(np.std(refc)) or math.nan
                dc, shc = _best_shift(tt[ok], y[ok], Rc, j)
                qc = res["q_clean"][:, j]
                acc_c = np.gradient(np.gradient(qc, ta_), ta_)[win_a]
                ref_acc = float(np.sqrt(np.mean(acc_c ** 2))) if acc_c.size else math.nan
                row.update(clean_rel_err_pct=100 * float(np.sqrt(np.mean((y[ok] - refc) ** 2))) / c_rms,
                           clean_delay_ms=dc * 1e3, clean_shape_rel_pct=100 * shc / c_rms,
                           input_dev_pct=100 * float(np.std((res["q"][:, j] - qc)[win_a])) / c_rms,
                           ref_accel_rms=ref_acc, accel_ratio=row["accel_rms"] / ref_acc if ref_acc else math.nan)
            rows.append(row)
    res["rows"] = rows
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    keys = ["rel_err_pct", "rms_err_deg", "max_err_deg", "delay_ms", "shape_rel_pct", "accel_rms", "stall_pct",
            "sat_pct", "backlog_ms"]
    if Rc is not None:
        keys += ["clean_rel_err_pct", "clean_delay_ms", "clean_shape_rel_pct", "input_dev_pct", "accel_ratio"]
    per_mode = [dict(mode=mo, aborted=any(r["aborted"] for r in rows if r["mode"] == mo),
                     **{k: nmed([r.get(k) for r in rows if r["mode"] == mo]) for k in keys}) for mo in modes]
    return dict(source=res["source"], P_action_ms=res["P_action"] * 1e3, chunk=res["chunk"], smooth=res.get("smooth", 0),
                noise_deg=res["noise_deg"], jitter_ms=res["jitter_ms"], chunk_offset_deg=res["chunk_offset_deg"],
                lead_ms=res.get("lead_ms", 0.0), predict=res.get("predict", "linear"),
                posi_period_ms=res.get("posi_period", math.nan) * 1e3, posi_advance_ms=res.get("posi_adv", math.nan) * 1e3,
                kp=res["kp"], feedforward_advance_ms=res["adv"] * 1e3, joints=[j + 1 for j in js],
                per_mode=per_mode, rows=rows)


def tail(x, scale=1.0):
    """Median and tail of a timing signal: p99, p99.9, max."""
    x = np.asarray(x, float).ravel()
    x = x[np.isfinite(x)] * scale
    if x.size == 0:
        return None
    return dict(n=int(x.size), median=float(np.median(x)), p99=float(np.percentile(x, 99)),
                p999=float(np.percentile(x, 99.9)), max=float(x.max()))


def _soak_signals(ph, frames, cmap):
    """Per tick and per frame timing signals of one soak phase [s]."""
    L = ph["log"]
    t1, rts = L[:, 1], L[:, 3 + 3 * NJ]
    fresh = np.r_[True, np.diff(rts) != 0]
    fr = frames[ph["i0"]:ph["i1"]]
    return dict(t=t1 - t1[0], age=t1 - cmap(rts), late=t1 - L[:, 0], call=L[:, 2] - t1, send_gap=np.diff(t1),
                fresh=fresh, host_gap=np.diff(fr[:, HOST]), robot_gap=np.diff(fr[:, ROBOT]),
                transport=fr[:, HOST] - cmap(fr[:, ROBOT]))


def analyze_soak(res, frames, cmap, mdl):
    """Per load: tails of the feedback age at the moment a command is sent, of the send interval and
    lateness, of the gaps between frames received; missing frames; ticks without new feedback;
    tracking error; and the crossover the model allows (60 deg of phase margin) if the loop is
    designed for the median feedback age or for its p99.9."""
    js = res["joints"]
    m = dict(tau=mdl["tau"], T=mdl["T"], P=res["P"], tau_s=mdl["tau_s"], fb_actual=res["fb"] == "actual")
    rows = []
    for ph in res["phases"]:
        L = ph["log"]
        if len(L) < 100:
            continue
        s = _soak_signals(ph, frames, cmap)
        rg = s["robot_gap"]
        k = np.maximum(1, np.round(rg / np.median(rg))) if rg.size else np.ones(1)
        idx = np.nonzero(s["fresh"])[0]
        no_fb = np.diff(L[idx, 1]) if len(idx) > 1 else np.zeros(1)
        e = L[:, 3 + 2 * NJ + np.array(js)] - L[:, 3 + NJ + np.array(js)]
        age = tail(s["age"], 1e3)
        to = res.get("vel_timeout", 0.05)
        row = dict(load=ph["load"], procs=ph.get("procs", 0), dur_s=float(s["t"][-1]), ticks=len(L),
                   loop_hz=float((len(L) - 1) / s["t"][-1]) if s["t"][-1] > 0 else math.nan, aborted=bool(ph["aborted"]),
                   fb_age_ms=age, send_interval_ms=tail(s["send_gap"], 1e3), send_late_ms=tail(s["late"], 1e3),
                   call_ms=tail(s["call"], 1e3), host_rx_gap_ms=tail(s["host_gap"], 1e3),
                   transport_ms=tail(s["transport"], 1e3),
                   missing_frames_pct=float(100 * np.sum(k - 1) / max(1.0, np.sum(k))),
                   stale_ticks_pct=100 * float(np.mean(~s["fresh"])), longest_no_feedback_ms=float(no_fb.max() * 1e3),
                   n_fb_age_over_10ms=int(np.sum(s["age"] > 0.010)), n_fb_age_over_20ms=int(np.sum(s["age"] > 0.020)),
                   n_fb_age_over_50ms=int(np.sum(s["age"] > 0.050)),
                   n_send_gap_over_vel_timeout=int(np.sum(s["send_gap"] > to)),
                   err_rms_deg=float(np.max(np.sqrt(np.nanmean(e ** 2, axis=0)))), err_max_deg=float(np.nanmax(np.abs(e))))
        if age:
            row["fc60_hz_median_age"] = kp_for_pm(60, dict(m, tau_b=age["median"] * 1e-3))["fc_hz"]
            row["fc60_hz_p999_age"] = kp_for_pm(60, dict(m, tau_b=age["p999"] * 1e-3))["fc_hz"]
        rows.append(row)
    res["rows"] = rows
    return dict(joints=[j + 1 for j in js], kp=res["kp"], feedforward_advance_ms=res["adv"] * 1e3, P_ms=res["P"] * 1e3,
                dur_s=res["dur"], model=dict(tau_ms=m["tau"] * 1e3, T_ms=m["T"] * 1e3, servo_lag_ms=m["tau_s"] * 1e3,
                                             source=mdl["src"]), phases=rows)


def plot_soak(res, summ, frames, cmap, out, plt):
    """Feedback age over time (max per second) and the share of ticks / frames above each value."""
    phases = [p for p in res["phases"] if len(p["log"]) >= 100]
    if not phases:
        return
    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    t_off = 0.0
    for i, ph in enumerate(phases):
        s = _soak_signals(ph, frames, cmap)
        c, lab = f"C{i}", ph["load"]
        sec = np.floor(s["t"]).astype(int)
        full = sec < int(s["t"][-1])                 # whole seconds only
        mx = np.full(max(1, int(s["t"][-1])), np.nan)
        np.fmax.at(mx, sec[full], s["age"][full] * 1e3)
        axs[0, 0].plot(t_off + np.arange(len(mx)), mx, color=c, lw=1, label=f"{lab}: max per second")
        t_off += len(mx) + 2
        for ax, x in ((axs[0, 1], s["age"]), (axs[1, 0], s["send_gap"]), (axs[1, 1], s["host_gap"])):
            x = np.sort(x[np.isfinite(x)]) * 1e3
            if x.size:
                ax.step(x, 1.0 - np.arange(x.size) / x.size, where="post", color=c, lw=1.5, label=lab)
    axs[0, 0].set(xlabel="s (phases one after the other)", ylabel="ms", ylim=(0, None),
                  title="Feedback age when a command is sent: worst of each second")
    for ax, ttl in ((axs[0, 1], "Feedback age when a command is sent"), (axs[1, 0], "Interval between two commands"),
                    (axs[1, 1], "Interval between two frames received")):
        ax.set(xscale="log", yscale="log", xlabel="ms (log)", ylabel="share of samples above (log)", title=ttl)
        ax.set_xlim(left=0.5)
        from matplotlib.ticker import LogLocator
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    _plain_log_x(axs[0, 1], axs[1, 0], axs[1, 1])
    for ax in axs.ravel():
        ax.grid(alpha=.3, which="both")
        ax.legend(fontsize=8)
    fig.suptitle(f"Soak: {summ['dur_s']:g} s per load, J{', J'.join(str(j) for j in summ['joints'])}, "
                 f"Kp {summ['kp']:g}, period {summ['P_ms']:g} ms", fontsize=10)
    _save(fig, out, "21_soak.png", plt)


def analyze_cartvel(res, cmap):
    """Per Cartesian axis (and SetCartAcc): delay + moving average fitted on the target Cartesian
    velocity, the same on the actual one, and their difference (servo lag)."""
    names = ["x", "y", "z", "rx", "ry", "rz"]

    def cart_vel(fr, m, kv, kp, k):
        v = fr[:, col(kv, k)]
        if np.isfinite(v).any():
            return v
        p = fr[:, col(kp, k)]
        return np.gradient(p, m) if np.isfinite(p).sum() > 3 else v

    groups = {}
    for run in res["runs"]:
        groups.setdefault((run["axis"], run["acc"]), []).append(run)
    rows = []
    for (ax, acc), runs in groups.items():
        k = names.index(ax)
        ft, fa = [], []
        for run in runs:
            c, fr = run["cmd"], run["frames"]
            if fr.shape[1] < NCOL:
                continue
            m = cmap(fr[:, ROBOT])
            ft.append(fit_boxcar(c[:, 1], c[:, 4 + k], m, cart_vel(fr, m, K_TCVEL, K_TCART, k)))
            fa.append(fit_boxcar(c[:, 1], c[:, 4 + k], m, cart_vel(fr, m, K_ACVEL, K_ACART, k)))
        mt, ma = model_summary(ft) or {}, model_summary(fa) or {}
        row = dict(axis=ax, cart_acc_pct=acc, amplitude=runs[0]["A"], unit="mm/s" if k < 3 else "deg/s",
                   tau_ms=mt.get("tau_ms", math.nan), T_ms=mt.get("T_ms", math.nan), fit_rms=mt.get("rms", math.nan),
                   eq_delay_ms=mt.get("eq_delay_ms", math.nan))
        if mt and ma:
            row["servo_lag_ms"] = ma["eq_delay_ms"] - mt["eq_delay_ms"]
        rows.append(row)
    res["rows"] = rows
    return rows


def plot_cartvel(res, out, plt):
    rows = res.get("rows") or []
    if not rows:
        return
    lab = [f"{r['axis']}" + ("" if r["cart_acc_pct"] is None else f"\nacc {r['cart_acc_pct']:g} %") for r in rows]
    x = np.arange(len(rows))
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    axs[0].bar(x, [r["T_ms"] for r in rows], 0.5, color="C3")
    axs[0].set(ylabel="ms", title=f"Cartesian moving-average length T ({res['frame'].upper()})")
    axs[1].bar(x - 0.2, [r["tau_ms"] for r in rows], 0.4, label="pure delay tau")
    axs[1].bar(x + 0.2, [r.get("servo_lag_ms", math.nan) for r in rows], 0.4, label="servo lag (actual - target)")
    axs[1].set(ylabel="ms", title="Delays")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.grid(alpha=.3, axis="y")
    axs[1].legend(fontsize=8)
    _save(fig, out, "20_cartvel.png", plt)


def analyze_sensorsweep(res, cmap, mdl):
    """One analyze_servo per emulated sensor: measured step and closed-loop gain next to the
    model fed with the measured feedback age (sensor delay and hold included)."""
    rows = []
    for c in res["configs"]:
        s = analyze_servo(c["servo"], cmap, mdl)
        if not s:
            continue
        st = s["steps"][0] if s["steps"] else {}
        row = dict(rate_hz=c["rate"], delay_ms=c["delay_ms"], kp=c["kp"],
                   fb_age_ms=(s.get("fb_age_ms") or {}).get("median"),
                   model_pm_deg=st.get("model_pm_deg"), model_fc_hz=st.get("model_fc_hz"),
                   rise_ms=st.get("rise_ms"), model_rise_ms=st.get("model_rise_ms"),
                   overshoot_pct=st.get("overshoot_pct"), model_overshoot_pct=st.get("model_overshoot_pct"),
                   settle_ms=st.get("settle_ms"), model_settle_ms=st.get("model_settle_ms"))
        for sn in s["sines"]:
            row[f"gain_{sn['f']:g}hz"] = sn["gain"]
            row[f"model_gain_{sn['f']:g}hz"] = sn["model_gain"]
        rows.append(row)
    res["rows"] = rows
    return dict(pm=res["pm"], noise_deg=res["noise_deg"], rows=rows)


def plot_sensorsweep(res, summ, out, plt):
    rows = summ.get("rows") or []
    if not rows:
        return
    lab = [f"{'enc.' if r['rate_hz'] <= 0 else f'{r['rate_hz']:g} Hz'}\n+{r['delay_ms']:g} ms" for r in rows]
    x = np.arange(len(rows))
    fig, axs = plt.subplots(1, 3, figsize=(19, 4.8))
    axs[0].bar(x - 0.2, [r["overshoot_pct"] for r in rows], 0.4, label="measured")
    axs[0].bar(x + 0.2, [r["model_overshoot_pct"] for r in rows], 0.4, label="model")
    axs[0].set(ylabel="%", title=f"Step overshoot (Kp set for {summ['pm']:g} deg of phase margin)")
    axs[1].bar(x, [r["kp"] for r in rows], 0.5, color="C2", label="Kp")
    ax2 = axs[1].twinx()
    ax2.plot(x, [r["model_fc_hz"] for r in rows], "ko-", label="model crossover")
    ax2.set_ylabel("Hz")
    axs[1].set(ylabel="Kp [1/s]", title="Achievable gain and crossover")
    keys = sorted({k for r in rows for k in r if k.startswith("gain_")})
    for i, k in enumerate(keys):
        axs[2].plot(x, [r.get(k) for r in rows], "o-", color=f"C{i}", label=f"measured {k[5:]}")
        axs[2].plot(x, [r.get("model_" + k) for r in rows], "s--", color=f"C{i}", label=f"model {k[5:]}")
    axs[2].set(ylabel="closed-loop gain", title="Tracking gain (feedback only)")
    for ax in axs:
        ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    _save(fig, out, "19_sensor_sweep.png", plt)


def plot_replay(res, summ, out, plt):
    rows = summ.get("rows") or []
    if not rows:
        return
    R = ActionRef(res["t"], res["q"], res["release"], horizon=0.0)
    js = res["joints"]
    jb = max(js, key=lambda j: float(np.std(res["q"][:, j])))
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    ta = res["settle"]
    tb = min(ta + 3.0, float(res["t"][-1]))
    tg = np.linspace(ta, tb, 1500)
    axs[0].plot(tg, R.true_at(tg, jb), "k", lw=1.2, label="commanded (actions at due time)")
    if "q_clean" in res and np.any(res["q_clean"] != res["q"]):
        axs[0].plot(tg, np.interp(tg, res["t"], res["q_clean"][:, jb]), color="0.55", ls="--", lw=1.2,
                    label="clean motion (before noise / offsets)")
    for i, run in enumerate(r for r in res["runs"] if "tt" in r):
        fr, tt = run["frames"], run["tt"]
        sel = (tt >= ta) & (tt <= tb)
        ap = fr[sel, col(K_APOS, jb)]
        axs[0].plot(tt[sel], ap - run["q0"][jb], color=f"C{i}", lw=1, label=f"actual, {run['mode']}")
    axs[0].set(xlabel="s", ylabel="deg", title=f"J{jb + 1}: commanded vs actual (3 s)")
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    joints = sorted({r["joint"] for r in rows})
    x = np.arange(len(joints))
    wd = 0.8 / max(1, len(modes))
    for i, mo in enumerate(modes):
        rr = {r["joint"]: r for r in rows if r["mode"] == mo}
        xs = x - 0.4 + wd * (i + 0.5)
        axs[1].bar(xs, [rr[j]["rel_err_pct"] if j in rr else math.nan for j in joints], wd, color=f"C{i}", label=mo)
        axs[1].plot(xs, [rr[j]["shape_rel_pct"] if j in rr else math.nan for j in joints], "k_", ms=10, mew=2)
        axs[2].bar(xs, [rr[j]["delay_ms"] if j in rr else math.nan for j in joints], wd, color=f"C{i}", label=mo)
    for ax in axs[1:]:
        ax.set_xticks(x); ax.set_xticklabels([f"J{j}" for j in joints]); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    axs[1].set(ylabel="% of the reference RMS", title="Tracking error (bars) and error once delayed (black)")
    axs[2].set(ylabel="ms", title="Delay behind the commanded trajectory")
    axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)
    sub = (f"{summ['source']} actions every {summ['P_action_ms']:.0f} ms, chunks of {summ['chunk']}, "
           f"noise {summ['noise_deg']:g} deg, jitter {summ['jitter_ms']:g} ms, chunk offset {summ['chunk_offset_deg']:g} deg"
           + (f", smoothed over {summ['smooth']} actions" if summ.get("smooth") else ""))
    fig.suptitle(sub, fontsize=10)
    _save(fig, out, "18_replay.png", plt)


def analyze_periodsweep(res, cmap):
    rows = analyze_sine(res, cmap)
    for row, run in zip(rows, res["runs"]):
        c, d = run["cmd"], run["d"]
        f = fit_boxcar(c[:, 1], c[:, 3], d["m"], d["tv"]) or {}
        row.update(P_ms=run["P"] * 1e3, fit_tau_ms=f.get("tau_ms", math.nan), fit_T_ms=f.get("T_ms", math.nan))
        row["pred_delay_ms"] = row["fit_tau_ms"] + row["fit_T_ms"] / 2 + row["P_ms"] / 2
    return rows


# --------------------------------------------------------------------------- #
# Analysis: position mode and external loop                                   #
# --------------------------------------------------------------------------- #
def _last_true_then(t, bad):
    """Time after which `bad` stays False (0 if never bad, nan if bad at the end)."""
    if not bad.any():
        return 0.0
    k = np.nonzero(bad)[0][-1]
    return float(t[k + 1]) if k + 1 < len(t) else math.nan


def analyze_posstep(res, cmap):
    mv, d = res["moves"], prep(res["frames"], cmap)
    res["d"] = d
    m, h, tol = d["m"], d["h"], res["tol"]
    rows = []
    for i, (t1, t2, q_from, q_to, size) in enumerate(mv):
        t_next = mv[i + 1][0] if i + 1 < len(mv) else m[-1]
        dq = q_to - q_from
        w = (m >= t1 - 0.02) & (m <= t_next)
        if w.sum() < 10 or abs(dq) < 1e-6:
            continue
        t, hh = m[w] - t1, h[w] - t1
        tp, ap, tv = d["tp"][w], d["ap"][w], d["tv"][w]
        pt = (tp - q_from) / dq
        thr = max(0.002, 2e-4 / abs(dq))
        dead = cross_time(t, pt, thr)
        dead_h = cross_time(hh, pt, thr, interp=False)
        t_done = _last_true_then(t, np.abs(tp - q_to) >= 1e-4)
        row = dict(size=float(size), dq=float(dq), dead_ms=dead * 1e3, dead_host_ms=dead_h * 1e3,
                   target_done_ms=t_done * 1e3, move_ms=(t_done - dead) * 1e3,
                   peak_vel=float(np.nanmax(np.abs(tv))), call_ms=(t2 - t1) * 1e3)
        if np.isfinite(tv).sum() > 5:
            row["peak_acc"] = float(np.nanmax(np.abs(np.gradient(tv, t))))
        if np.isfinite(ap).any():
            pa = (ap - q_from) / dq
            fin = np.isfinite(ap)
            row["actual_settle_ms"] = _last_true_then(t[fin], np.abs(ap[fin] - q_to) > tol) * 1e3
            row["actual_t50_ms"] = cross_time(t, pa, 0.5) * 1e3
            row["actual_lag_t50_ms"] = row["actual_t50_ms"] - cross_time(t, pt, 0.5) * 1e3
            row["overshoot_deg"] = float(max(0.0, np.nanmax(pa) - 1) * abs(dq))
            tail = fin & (t > t[-1] - 0.05)
            row["final_err_deg"] = float(np.median(ap[tail]) - q_to) if tail.any() else math.nan
        rows.append(row)
    res["rows"] = rows
    keys = [k for k in (rows[0] if rows else {}) if k not in ("size", "dq")]
    return [dict(size=s, n=len(g), **{k: nmed([r.get(k) for r in g]) for k in keys})
            for s in sorted({r["size"] for r in rows}) for g in [[r for r in rows if r["size"] == s]]]


def analyze_posstream(res, cmap):
    rows = []
    for run in res["runs"]:
        c, d = run["cmd"], prep(run["frames"], cmap)
        run["d"] = d
        if len(c) < 10:
            continue
        f, A, q0 = run["f"], run["A"], run["q0"]
        w = 2 * math.pi * f
        t0, ta, tb = c[0, 1], c[0, 1] + run["settle"], c[-1, 1]
        row = dict(P_ms=run["P"] * 1e3, f=f, A=A, n_cmd=len(c),
                   call_ms=float(np.median(c[:, 2] - c[:, 1]) * 1e3))
        for sig, key in (("tp", "tgt"), ("ap", "act")):
            if not has(d, sig):
                continue
            for clk, suf in (("m", ""), ("h", "_host")):
                g, lag, nl = sine_vs(c[:, 1], c[:, 3], d[clk], d[sig] - q0, w, ta, tb, t0)
                row[f"{key}{suf}_gain"], row[f"{key}{suf}_delay_ms"] = g, lag / w * 1e3
                if not suf:
                    row[f"{key}_nonlin"] = nl
            mid = 0.5 * (ta + tb)
            _, l1, _ = sine_vs(c[:, 1], c[:, 3], d["m"], d[sig] - q0, w, ta, mid, t0)
            _, l2, _ = sine_vs(c[:, 1], c[:, 3], d["m"], d[sig] - q0, w, mid, tb, t0)
            row[f"{key}_delay_drift_ms_per_s"] = (l2 - l1) / w * 1e3 / max(1e-3, 0.5 * (tb - ta))
        # stop-and-go: target almost still while the (delayed) reference moves fast
        mm = (d["m"] >= ta) & (d["m"] <= tb) & np.isfinite(d["tv"])
        dl = row.get("tgt_delay_ms", 0.0) * 1e-3
        vref = A * w * np.cos(w * (d["m"][mm] - c[0, 0] - dl))
        fast = np.abs(vref) > 0.5 * A * w
        row["stall_pct"] = float(100 * np.mean(np.abs(d["tv"][mm][fast]) < 0.1 * A * w)) if fast.any() else math.nan
        moving = np.isfinite(d["tv"]) & (np.abs(d["tv"]) > max(0.2, 0.02 * A * w))
        row["backlog_ms"] = (float(d["m"][np.nonzero(moving)[0][-1]]) - c[-1, 1]) * 1e3 if moving.any() else math.nan
        rows.append(row)
    res["rows"] = rows
    return rows


def analyze_posinterrupt(res, cmap):
    d = prep(res["frames"], cmap)
    res["d"] = d
    m = d["m"]
    rows, traces = [], []
    for (t_go, tc1, tc2, tn1, tn2, s) in res["events"]:
        w = (m >= tc1 - 0.1) & (m <= tc1 + 2.0)
        t, v = m[w] - tc1, s * d["tv"][w]
        pre = (t < 0) & (t > -0.05) & np.isfinite(v)
        if pre.sum() < 3:
            continue
        v0 = float(np.median(v[pre]))
        if v0 <= 0.5:
            print(f"  posinterrupt: joint not moving when ClearMotion was sent (v = {v0:.2f} deg/s), "
                  "increase --int-dist or decrease --int-after")
            continue
        post = t >= 0

        def first(cond):
            k = np.nonzero(post & cond)[0]
            return float(t[k[0]]) if len(k) else math.nan

        decel = first(v < v0 - max(0.5, 0.02 * v0))
        stop = first(v <= max(0.2, 0.01 * v0))
        rev = first(v < -max(0.2, 0.01 * v0))
        rows.append(dict(v0=v0, clear_call_ms=(tc2 - tc1) * 1e3, clear_to_decel_ms=decel * 1e3,
                         clear_to_stop_ms=stop * 1e3, decel_ms=(stop - decel) * 1e3,
                         clear_to_reverse_ms=rev * 1e3, new_cmd_to_reverse_ms=(rev - (tn1 - tc1)) * 1e3,
                         stop_to_reverse_ms=(rev - stop) * 1e3))
        traces.append((t, v / v0))
    res["rows"], res["traces"] = rows, traces
    return dict(n=len(rows), **{k: stats([r[k] for r in rows]) for k in (rows[0] if rows else {})})


def _step_metrics(t, prog, t_end, tol):
    sel = (t >= 0) & (t <= t_end) & np.isfinite(prog)
    t, p = t[sel], prog[sel]
    if len(t) < 5:
        return {}
    t10, t90 = cross_time(t, p, 0.1), cross_time(t, p, 0.9)
    tail = t > t[-1] - 0.1
    return dict(t50_ms=cross_time(t, p, 0.5) * 1e3, rise_ms=(t90 - t10) * 1e3,
                overshoot_pct=100 * max(0.0, float(np.max(p) - 1)),
                settle_ms=_last_true_then(t, np.abs(p - 1) > tol) * 1e3,
                ss_err_pct=100 * float(np.median(p[tail]) - 1))


def analyze_servo(res, cmap, mdl):
    P, vmax = res["P"], res["vmax"]
    allruns = [r for r in res["steps"] + res["sines"] if len(r["log"])]
    if not allruns:
        return {}
    ages = np.concatenate([r["log"][:, 1] - cmap(r["log"][:, 6]) for r in allruns])
    per = np.concatenate([np.diff(r["log"][:, 1]) for r in allruns])
    m = dict(tau=mdl["tau"], T=mdl["T"], P=P, tau_b=float(np.nanmedian(ages)), tau_s=mdl["tau_s"],
             fb_actual=res["fb"] == "actual")
    res["model"] = m
    summ = dict(model=dict(tau_ms=m["tau"] * 1e3, T_ms=m["T"] * 1e3, P_ms=P * 1e3, servo_lag_ms=m["tau_s"] * 1e3,
                           fb_age_ms=m["tau_b"] * 1e3, source=mdl["src"]),
                feedforward_advance_ms=res["adv"] * 1e3, feedback=res["fb"],
                fb_age_ms=stats(ages, 1e3), loop_period_ms=stats(per, 1e3), steps=[], sines=[])
    for r in res["steps"]:
        L = r["log"]
        if not len(L):
            continue
        d = prep(r["frames"], cmap)
        r["d"] = d
        y = (d["ap"] if has(d, "ap") else d["tp"]) - r["q0"]
        on = np.nonzero(L[:, 4] != 0)[0]
        if not len(on):
            continue
        k_on, S = on[0], L[on[0], 4]
        t_on = L[k_on, 1]
        off = np.nonzero((L[:, 4] == 0) & (np.arange(len(L)) > k_on))[0]
        t_end = (L[off[0], 1] if len(off) else L[-1, 1]) - t_on
        tol = max(0.02, 3 * ENC_STEP / abs(S))
        met = _step_metrics(d["m"] - t_on, y / S, t_end, tol)
        sh = 10 * P
        ts, ys = sim_servo(lambda tt: S if tt >= sh - 1e-9 else 0.0, None, r["kp"], 0.0, m, sh + t_end, vmax)
        r["sim"] = (ts - sh, ys)
        pm = _step_metrics(ts - sh, ys / S, t_end, tol)
        mg = margins(r["kp"], m)
        r["summ"] = dict(kp=r["kp"], step=S, aborted=r["aborted"],
                         sat_pct=100 * float(np.mean(np.abs(L[:, 3]) >= 0.999 * vmax)), **met,
                         **{"model_" + k: v for k, v in pm.items()},
                         model_pm_deg=mg["pm_deg"], model_fc_hz=mg["fc_hz"], model_gm_db=mg["gm_db"])
        summ["steps"].append(r["summ"])
    for r in res["sines"]:
        L = r["log"]
        if len(L) < 10:
            continue
        d = prep(r["frames"], cmap)
        r["d"] = d
        y = (d["ap"] if has(d, "ap") else d["tp"]) - r["q0"]
        w = 2 * math.pi * r["f"]
        t0, ta, tb = L[0, 1], L[0, 1] + r["settle"], L[-1, 1]
        g, lag, nl = sine_vs(L[:, 1], L[:, 4], d["m"], y, w, ta, tb, t0)
        mm = (d["m"] >= ta) & (d["m"] <= tb) & np.isfinite(y)
        err = y[mm] - np.interp(d["m"][mm], L[:, 1], L[:, 4])
        H = closed_loop_frf(r["f"], r["kp"], m, r["ff"], res["adv"] if r["ff"] else 0.0)
        summ["sines"].append(dict(f=r["f"], ff=r["ff"], kp=r["kp"], A=r["A"], gain=g,
                                  phase_deg=-math.degrees(lag), delay_ms=lag / w * 1e3, nonlin=nl,
                                  err_rms_pct=100 * float(np.sqrt(np.mean(err ** 2))) / r["A"],
                                  sat_pct=100 * float(np.mean(np.abs(L[:, 3]) >= 0.999 * vmax)),
                                  model_gain=float(abs(H)), model_phase_deg=float(np.degrees(np.angle(H)))))
    return summ


def predict_external_loop(S, args, mdl):
    fb = (S.get("timing", {}).get("feedback_excess_latency_ms") or {}).get("median")
    src_fb = "median feedback excess latency + 0.5 ms"
    tau_b = (fb if fb is not None else 3.0) * 1e-3 + 0.5e-3
    if (S.get("servo") or {}).get("model"):
        tau_b, src_fb = S["servo"]["model"]["fb_age_ms"] * 1e-3, "measured in the servo test"
    m = dict(tau=mdl["tau"], T=mdl["T"], P=args.period, tau_b=tau_b, tau_s=mdl["tau_s"], fb_actual=mdl["fb_actual"])
    ts = m["tau_s"] if m["fb_actual"] else 0.0
    out = dict(model_source=mdl["src"], feedback_source=src_fb, tau_ms=m["tau"] * 1e3, T_ms=m["T"] * 1e3,
               P_ms=m["P"] * 1e3, servo_lag_ms=ts * 1e3, fb_ms=tau_b * 1e3,
               loop_delay_ms=(m["tau"] + m["T"] / 2 + m["P"] / 2 + ts + tau_b) * 1e3,
               vel_bw_3db_hz=0.443 / m["T"])
    for pm in (60, 45):
        r = kp_for_pm(pm, m)
        out[f"pm{pm}"] = dict(kp=r["kp"], fc_hz=r["fc_hz"], gm_db=r["gm_db"])
    if "sensorsweep" in S:  # a few seconds of computation: only when the sensor is the question
        out["sensor_table"] = sensor_table(m)
    return out


def sensor_table(m, rates=(1000, 240, 120, 60, 30), latencies=(5, 10, 20, 33, 50)):
    """Loop crossover at 60 deg of phase margin when the position comes from an external sensor
    sampling at `rate` with a total `latency` (exposure to controller): tau_b = latency + 1/(2 rate)."""
    rows = []
    for rate in rates:
        for lat in latencies:
            r = kp_for_pm(60, dict(m, tau_b=lat * 1e-3 + 0.5 / rate))
            rows.append(dict(rate_hz=rate, latency_ms=lat, kp60=r["kp"], fc60_hz=r["fc_hz"]))
    return rows


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _plain_log_x(*axs):
    from matplotlib.ticker import FuncFormatter, NullFormatter
    for a in axs:
        a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        a.xaxis.set_minor_formatter(NullFormatter())


def _save(fig, out, name, plt):
    p = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(p, dpi=130)
    plt.close(fig)
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
    if run.get("fit"):
        f = run["fit"]
        integ = _staircase_integral(c[:, 1], c[:, 3])
        ax.plot((d["m"] - te0) * 1e3, boxcar_response(d["m"], integ, f["tau_ms"] * 1e-3, f["T_ms"] * 1e-3),
                "k--", lw=.8, label=f"model: delay {f['tau_ms']:.1f} ms + moving avg {f['T_ms']:.1f} ms")
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
    for sig, cl, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
        if has(d, sig):
            mm = sel(d["m"])
            ax.plot((d["m"][mm] - te0) * 1e3, d[sig][mm], ".-", color=cl, ms=3, label=f"{lab} (robot clk)")
            mh = sel(d["h"])
            ax.plot((d["h"][mh] - te0) * 1e3, d[sig][mh], "x", color=cl, ms=4, alpha=.6,
                    label=f"{lab} (when host got it)")
            if row.get(sig) and np.isfinite(row[sig]["dead"]):
                ax.axvline(row[sig]["dead"] * 1e3, color=cl, ls="--", lw=.8)
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
    _save(fig, out, "1_step_response.png", plt)


def plot_impulse(res, out, plt):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    g = res["grid"] * 1e3
    for i, o in enumerate(res["table"]):
        cl = plt.cm.viridis(i / max(1, len(res["table"]) - 1))
        if "av_curve" in o:
            axs[0].plot(g, o["av_curve"], color=cl, label=f"{o['width_ms']:.0f} ms actual")
        if "tv_curve" in o:
            axs[0].plot(g, o["tv_curve"], "--", color=cl, lw=.8)
    axs[0].set(xlabel="ms from pulse start", ylabel="velocity / A",
               title="Pulse ('dirac') responses (solid actual, dashed target)", xlim=(-5, g[-1]))
    axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
    w = [o["width_ms"] for o in res["table"]]
    for sig, cl, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
        if f"{sig}_peak_ratio" in res["table"][0]:
            axs[1].plot(w, [o[f"{sig}_peak_ratio"] for o in res["table"]], "o-", color=cl, label=f"peak vel / A ({lab})")
        if f"{sig}_disp_ratio" in res["table"][0]:
            axs[1].plot(w, [o[f"{sig}_disp_ratio"] for o in res["table"]], "s--", color=cl,
                        label=f"displacement / commanded ({lab})")
    axs[1].set_xscale("log")
    axs[1].set(xlabel="pulse width [ms]", ylabel="ratio", title="How much of a short pulse gets through")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=.3, which="both")
    _save(fig, out, "2_impulse_response.png", plt)


def plot_bode(res, out, plt):
    rows = res["rows"]
    f = np.array([r["f"] for r in rows])
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    series = (("tv", "C0", "-", "target (robot clk)"), ("tv_host", "C0", ":", "target (host sees)"),
              ("av", "C1", "-", "actual (robot clk)"), ("av_host", "C1", ":", "actual (host sees)"))
    for key, cl, ls, lab in series:
        if f"{key}_gain" not in rows[0]:
            continue
        g = np.array([r[f"{key}_gain"] for r in rows])
        ph = np.array([r[f"{key}_phase_deg"] for r in rows])
        dl = np.array([r[f"{key}_delay_ms"] for r in rows])
        if "host" not in key:
            axs[0].semilogx(f, 20 * np.log10(g), "o" + ls, color=cl, label=lab)
        axs[1].semilogx(f, ph, "o" + ls, color=cl, label=lab)
        axs[2].semilogx(f, dl, "o" + ls, color=cl, label=lab)
    ff = np.logspace(np.log10(f.min()), np.log10(f.max()), 200)
    mdl = res.get("model")
    if mdl:
        H = vel_frf(ff, mdl["tau"], mdl["T"], mdl["P"])
        lab = f"model: delay {mdl['tau'] * 1e3:.1f} ms + moving avg {mdl['T'] * 1e3:.1f} ms (+P/2)"
        axs[0].semilogx(ff, 20 * np.log10(np.abs(H)), "k--", lw=.9, label=lab)
        dly = mdl["tau"] + mdl["T"] / 2 + mdl["P"] / 2
        axs[1].semilogx(ff, -360 * ff * dly, "k--", lw=.9, label=lab)
        axs[2].axhline(dly * 1e3, color="k", ls="--", lw=.9, label=lab)
    elif "tv_delay_ms" in rows[0]:
        tau = np.nanmedian([r["tv_delay_ms"] for r in rows]) * 1e-3
        axs[1].semilogx(ff, -360 * ff * tau, "k--", lw=.8, label=f"pure delay {tau * 1e3:.1f} ms")
    axs[0].axhline(-3, color="gray", ls="--", lw=.8)
    axs[0].set(ylabel="gain [dB]", title="Velocity command -> response (sine sweep)")
    axs[1].set(ylabel="phase [deg]")
    axs[2].set(ylabel="equivalent delay [ms]", xlabel="frequency [Hz]")
    for a in axs:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    _save(fig, out, "3_bode.png", plt)


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
        for sig, cl, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
            if has(d, sig):
                mm = (d["m"] >= ta) & (d["m"] <= tb)
                ax.plot((d["m"][mm] - ta) * 1e3, d[sig][mm], color=cl, label=lab)
        ax.set(title=f"{run['f']} Hz, A = {run['A']:.2f} deg/s (last 2 periods)", xlabel="ms", ylabel="deg/s")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    _save(fig, out, "4_sine_traces.png", plt)


def plot_pingpong(res, out, plt):
    names = [n for n in res if "detect" in res[n]]
    if not names:
        return
    fig, axs = plt.subplots(len(names), 2, figsize=(14, 4 * len(names)), squeeze=False)
    for (a0, a1), n in zip(axs, names):
        r = res[n]
        a0.hist(r["detect"] * 1e3, bins=30, alpha=.6, label="send -> host sees threshold")
        if len(r["eff_h"]):
            a0.hist(r["eff_h"] * 1e3, bins=30, alpha=.6, label="send -> host sees target react")
        if len(r["eff_r"]):
            a0.hist(r["eff_r"] * 1e3, bins=30, alpha=.6, label="send -> target reacts (robot clk)")
        a0.set(title=f"Ping-pong on {n} velocity", xlabel="ms", ylabel="count")
        a0.legend(fontsize=8); a0.grid(alpha=.3)
        d, c = r["d"], r["cmd"]
        t0 = c[0, 0] + 0.2
        sel = lambda t: (t >= t0) & (t <= t0 + 0.3)
        m = sel(c[:, 0])
        a1.step((c[m, 0] - t0) * 1e3, c[m, 1], where="post", color="k", lw=1, label="command")
        for sig, cl, lab in (("tv", "C0", "target"), ("av", "C1", "actual")):
            if has(d, sig):
                mm = sel(d["m"])
                a1.plot((d["m"][mm] - t0) * 1e3, d[sig][mm], color=cl, label=lab)
        for s in (1, -1):
            a1.axhline(s * r["thr"], color="gray", ls="--", lw=.8)
        a1.set(title="300 ms of the limit cycle", xlabel="ms", ylabel="deg/s")
        a1.legend(fontsize=8); a1.grid(alpha=.3)
    _save(fig, out, "5_pingpong.png", plt)


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
    _save(fig, out, "6_timing.png", plt)


def plot_accsweep(res, out, plt):
    rows = res["rows"]
    p = [r["pct"] for r in rows]
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
    for name, cl in (("target", "C0"), ("actual", "C1")):
        if f"{name}_accel_deg_s2" not in rows[0]:
            continue
        axs[0].plot(p, [r[f"{name}_accel_deg_s2"] for r in rows], "o-", color=cl, label=name)
        axs[1].plot(p, [r[f"{name}_rise_10_90_ms"] for r in rows], "o-", color=cl, label=name)
        axs[2].plot(p, [r[f"{name}_dead_ms"] for r in rows], "o-", color=cl, label=f"{name} dead time")
        axs[2].plot(p, [r[f"{name}_t50_ms"] for r in rows], "s--", color=cl, label=f"{name} 50 %")
    axs[3].plot(p, [r["T_ms"] for r in rows], "o-", color="C3", label="fitted T")
    axs[3].plot(p, [r["tau_ms"] for r in rows], "s-", color="C2", label="fitted delay tau")
    pp = np.linspace(min(p), max(p), 100)
    axs[3].plot(pp, np.maximum(J6_T_FLOOR_MS, J6_T_MS_PER_PCT * pp), "k:", lw=.9,
                label=f"J6 2026-11-24: max({J6_T_FLOOR_MS} ms, {J6_T_MS_PER_PCT:g} ms/%)")
    axs[0].set(title=f"Acceleration (step of {res['amp']:.1f} deg/s)", xlabel="SetJointVelLimit [%]", ylabel="deg/s^2")
    axs[1].set(title="Rise time 10-90 %", xlabel="SetJointVelLimit [%]", ylabel="ms")
    axs[2].set(title="Delays", xlabel="SetJointVelLimit [%]", ylabel="ms")
    axs[3].set(title="Model: moving-average length T", xlabel="SetJointVelLimit [%]", ylabel="ms")
    for a in axs:
        a.grid(alpha=.3); a.legend(fontsize=8)
    _save(fig, out, "7_accel_vs_vel_limit.png", plt)


def plot_jaccsweep(res, out, plt):
    rows = [r for r in res["rows"] if not r.get("failed")]
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, vl in enumerate(sorted({r["vl"] for r in rows})):
        rr = sorted([r for r in rows if r["vl"] == vl], key=lambda r: r["acc"])
        acc, T = np.array([r["acc"] for r in rr]), np.array([r["T_ms"] for r in rr])
        axs[0].plot(acc, T, "o-", color=f"C{i}", label=f"SetJointVelLimit {vl:g} %")
        axs[1].plot(acc, T * acc / 100, "o-", color=f"C{i}", label=f"SetJointVelLimit {vl:g} %")
    axs[0].set(xscale="log", yscale="log", xlabel="SetJointAcc [%]", ylabel="T [ms]",
               title="Moving-average length vs SetJointAcc")
    axs[1].set(xscale="log", xlabel="SetJointAcc [%]", ylabel="T x SetJointAcc / 100 [ms]",
               title="Flat line = T inversely proportional to SetJointAcc")
    for a in axs:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    _save(fig, out, "8_jacc_sweep.png", plt)


def plot_ampsweep(res, out, plt):
    rows = res["rows"]
    if not rows:
        return
    dv = np.array([r["dv"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, lab in ((axs[0], "T_ms", "fitted T [ms]"), (axs[1], "acc_eff", "|dv| / T [deg/s^2]"),
                         (axs[2], "rel_rms_pct", "fit residual / |dv| [%]")):
        y = np.array([r[key] for r in rows])
        ax.plot(dv, y, "o", alpha=.5)
        lv = sorted(set(np.round(dv, 1)))
        ax.plot(lv, [np.median(y[np.round(dv, 1) == v]) for v in lv], "k-", lw=1, label="median")
        ax.set(xlabel="velocity step |dv| [deg/s]", ylabel=lab); ax.grid(alpha=.3); ax.legend(fontsize=8)
    Tm = np.median([r["T_ms"] for r in rows]) * 1e-3
    axs[1].plot([0, dv.max()], [0, dv.max() / Tm], "C3:", label="linear (constant T)")
    axs[1].legend(fontsize=8)
    axs[0].set_title("Linear system <=> T independent of step size")
    axs[1].set_title("Acceleration saturation would bend this line")
    axs[2].set_title("Model fit quality per edge")
    _save(fig, out, "9_amp_sweep.png", plt)


def plot_multijoint(res, out, plt):
    rows = res["rows"]
    if not rows:
        return
    joints = sorted({r["joint"] for r in rows})
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.5))
    x = np.arange(len(joints))
    for i, (sc_name, sel) in enumerate((("alone", lambda r: r["scenario"].endswith("alone")),
                                        ("together", lambda r: r["scenario"] == "together"))):
        for ax, key in zip(axs, ("T_ms", "tau_ms", "servo_lag_ms")):
            vals = [next((r.get(key, math.nan) for r in rows if sel(r) and r["joint"] == j), math.nan) for j in joints]
            ax.bar(x - 0.2 + 0.4 * i, vals, 0.4, label=sc_name)
    for ax, t in zip(axs, ("Moving-average length T", "Pure delay tau", "Servo lag (actual - target)")):
        ax.set_xticks(x); ax.set_xticklabels([f"J{j}" for j in joints])
        ax.set(ylabel="ms", title=t); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    _save(fig, out, "10_multijoint.png", plt)


def plot_servomulti(res, summ, out, plt):
    rows = summ.get("rows") or []
    if not rows:
        return
    joints = sorted({r["joint"] for r in rows})
    freqs = sorted({r["f"] for r in rows})
    ffs = sorted({r["ff"] for r in rows})
    fig, axs = plt.subplots(1, len(ffs), figsize=(8 * len(ffs), 4.8), squeeze=False)
    x = np.arange(len(joints))
    wd = 0.8 / len(freqs)
    for ax, ff in zip(axs[0], ffs):
        for i, f in enumerate(freqs):
            rr = {r["joint"]: r for r in rows if r["ff"] == ff and r["f"] == f}
            xs = x - 0.4 + wd * (i + 0.5)
            ax.bar(xs, [rr[j]["err_rms_pct"] if j in rr else math.nan for j in joints], wd,
                   color=plt.cm.Blues(0.35 + 0.6 * i / max(1, len(freqs) - 1)), label=f"{f:g} Hz")
            ax.plot(xs, [rr[j]["model_err_rms_pct"] if j in rr else math.nan for j in joints], "k_", ms=12, mew=2)
        ax.set_xticks(x); ax.set_xticklabels([f"J{j}" for j in joints])
        ax.set(ylabel="tracking error RMS [% of amplitude]",
               title=f"All joints at once, Kp {summ['kp']:g}, " + ("with feedforward" if ff else "feedback only")
               + " (black: single-joint model)")
        ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
    _save(fig, out, "16_servo_multi.png", plt)


def plot_periodsweep(res, out, plt):
    rows = res["rows"]
    if not rows:
        return
    P = [r["P_ms"] for r in rows]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].plot(P, [r.get("tv_delay_ms") for r in rows], "o-", label="target (robot clk)")
    axs[0].plot(P, [r.get("tv_host_delay_ms") for r in rows], "o:", label="target (host sees)")
    if "av_delay_ms" in rows[0]:
        axs[0].plot(P, [r.get("av_delay_ms") for r in rows], "s-", label="actual (robot clk)")
    axs[0].plot(P, [r["pred_delay_ms"] for r in rows], "k--", lw=.9, label="fitted tau + T/2 + P/2")
    axs[1].plot(P, [r.get("tv_gain") for r in rows], "o-", label="target")
    if "av_gain" in rows[0]:
        axs[1].plot(P, [r.get("av_gain") for r in rows], "s-", label="actual")
    axs[0].set(xlabel="command period P [ms]", ylabel="equivalent delay [ms]",
               title=f"Sine {rows[0]['f']:g} Hz: delay vs streaming period")
    axs[1].set(xlabel="command period P [ms]", ylabel="gain", title="Gain vs streaming period")
    for a in axs:
        a.grid(alpha=.3); a.legend(fontsize=8)
    _save(fig, out, "11_period_sweep.png", plt)


def plot_posstep(res, summ, out, plt):
    rows, d, mv = res["rows"], res["d"], res["moves"]
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    sizes = sorted({r["size"] for r in rows})
    for i, s in enumerate(sizes):
        k = next(k for k, m_ in enumerate(mv) if m_[4] == s and m_[3] != m_[2])
        t1, _, q_from, q_to, _ = mv[k]
        t_next = mv[k + 1][0] if k + 1 < len(mv) else d["m"][-1]
        w = (d["m"] >= t1 - 0.02) & (d["m"] <= t_next)
        t = (d["m"][w] - t1) * 1e3
        axs[0].plot(t, (d["tp"][w] - q_from) / (q_to - q_from), "--", color=f"C{i}", lw=.9)
        if has(d, "ap"):
            axs[0].plot(t, (d["ap"][w] - q_from) / (q_to - q_from), color=f"C{i}", label=f"{s:g} deg")
    axs[0].set(xlabel="ms from MoveJoints sent", ylabel="progress", title="MoveJoints (dashed target, solid actual)")
    sz = [r["size"] for r in summ]
    for key, lab, mk in (("target_done_ms", "send -> target at goal", "o-"),
                         ("move_ms", "target motion duration", "s-"),
                         ("actual_settle_ms", f"send -> actual within {res['tol']:g} deg", "^-")):
        if key in summ[0]:
            axs[1].plot(sz, [r[key] for r in summ], mk, label=lab)
    for key, lab, mk in (("dead_ms", "dead time (robot clk)", "o-"), ("dead_host_ms", "dead time (host sees)", "o:"),
                         ("actual_lag_t50_ms", "actual lag behind target at 50 %", "s-")):
        if key in summ[0]:
            axs[2].plot(sz, [r[key] for r in summ], mk, label=lab)
    axs[1].set(xscale="log", xlabel="move size [deg]", ylabel="ms", title=f"Durations (SetJointVel {res['vel_pct']:g} %)")
    axs[2].set(xscale="log", xlabel="move size [deg]", ylabel="ms", title="Delays")
    for a in axs:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    _plain_log_x(axs[1], axs[2])
    _save(fig, out, "12_pos_step.png", plt)


def plot_posstream(res, out, plt):
    rows = res["rows"]
    if not rows:
        return
    Ps = sorted({r["P_ms"] for r in rows})
    ncol = max(3, len(Ps))
    fig, axs = plt.subplots(2, ncol, figsize=(5.5 * ncol, 8), squeeze=False)
    fmax = max(r["f"] for r in rows)
    for i, P in enumerate(Ps):
        ax = axs[0, i]
        run = next((r for r in res["runs"] if abs(r["P"] * 1e3 - P) < 1e-6 and r["f"] == fmax and "d" in r), None)
        if run is None:
            continue
        c, d = run["cmd"], run["d"]
        tb = c[-1, 1]
        ta = tb - 2.0 / run["f"]
        m = c[:, 1] >= ta
        ax.step((c[m, 1] - ta) * 1e3, c[m, 3], where="post", color="k", lw=1, label="MoveJoints targets")
        mm = (d["m"] >= ta) & (d["m"] <= tb + 0.3)
        ax.plot((d["m"][mm] - ta) * 1e3, d["tp"][mm] - run["q0"], color="C0", label="target")
        if has(d, "ap"):
            ax.plot((d["m"][mm] - ta) * 1e3, d["ap"][mm] - run["q0"], color="C1", label="actual")
        ax.set(title=f"one MoveJoints every {P:g} ms, {run['f']:g} Hz", xlabel="ms", ylabel="deg")
        ax.grid(alpha=.3); ax.legend(fontsize=8)
    for j in range(len(Ps), ncol):
        axs[0, j].set_visible(False)
    for i, P in enumerate(Ps):
        rr = sorted([r for r in rows if r["P_ms"] == P], key=lambda r: r["f"])
        f = [r["f"] for r in rr]
        axs[1, 0].plot(f, [r.get("tgt_delay_ms") for r in rr], "o-", color=f"C{i}", label=f"target, P={P:g} ms")
        axs[1, 0].plot(f, [r.get("act_delay_ms") for r in rr], "s--", color=f"C{i}", label=f"actual, P={P:g} ms")
        axs[1, 1].plot(f, [r.get("tgt_gain") for r in rr], "o-", color=f"C{i}", label=f"target, P={P:g} ms")
        axs[1, 2].plot(f, [r.get("backlog_ms") for r in rr], "o-", color=f"C{i}", label=f"backlog, P={P:g} ms")
        axs[1, 2].plot(f, [r.get("stall_pct") for r in rr], "x:", color=f"C{i}", label=f"stall %, P={P:g} ms")
    axs[1, 0].set(xlabel="Hz", ylabel="ms", title="Tracking delay (robot clock)")
    axs[1, 1].set(xlabel="Hz", ylabel="gain", title="Tracking gain")
    axs[1, 2].set(xlabel="Hz", ylabel="ms  /  %", title="Queue backlog after last send / stop-and-go")
    for a in axs[1, :3]:
        a.grid(alpha=.3); a.legend(fontsize=7)
    for j in range(3, ncol):
        axs[1, j].set_visible(False)
    _save(fig, out, "13_pos_stream.png", plt)


def plot_posinterrupt(res, out, plt):
    rows, traces = res["rows"], res["traces"]
    if not rows:
        return
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    for t, v in traces:
        axs[0].plot(t * 1e3, v, lw=.9)
    for key, cl in (("clear_to_decel_ms", "C1"), ("clear_to_stop_ms", "C2"), ("clear_to_reverse_ms", "C3")):
        axs[0].axvline(nmed([r[key] for r in rows]), color=cl, ls="--", lw=.8, label=key.replace("_ms", ""))
    axs[0].set(xlabel="ms from ClearMotion sent", ylabel="target vel / cruise vel",
               title="ClearMotion + new MoveJoints", xlim=(-50, 300))
    axs[0].grid(alpha=.3); axs[0].legend(fontsize=8)
    keys = ["clear_to_decel_ms", "decel_ms", "stop_to_reverse_ms", "clear_to_reverse_ms"]
    axs[1].boxplot([[r[k] for r in rows if np.isfinite(r[k])] or [math.nan] for k in keys], showmeans=True)
    axs[1].set_xticks(range(1, len(keys) + 1))
    axs[1].set_xticklabels([k.replace("_ms", "").replace("_", "\n") for k in keys], fontsize=8)
    axs[1].set(ylabel="ms", title=f"Re-targeting latency over {len(rows)} interruptions"); axs[1].grid(alpha=.3)
    _save(fig, out, "14_pos_interrupt.png", plt)


def plot_servo(res, summ, out, plt):
    if not summ:
        return
    fig, axs = plt.subplots(2, 3, figsize=(19, 9))
    for i, r in enumerate([r for r in res["steps"] if "summ" in r]):
        d, L, s = r["d"], r["log"], r["summ"]
        on = np.nonzero(L[:, 4] != 0)[0]
        t_on = L[on[0], 1]
        y = (d["ap"] if has(d, "ap") else d["tp"]) - r["q0"]
        t = (d["m"] - t_on) * 1e3
        sel = (t > -50) & (t < r.get("hold", 0.6) * 1e3 + 50)
        axs[0, 0].plot(t[sel], y[sel] / s["step"], color=f"C{i}", label=f"Kp {r['kp']:g}")
        ts, ys = r["sim"]
        axs[0, 0].plot(ts * 1e3, ys / s["step"], "--", color=f"C{i}", lw=.8)
    axs[0, 0].set(xlabel="ms from step", ylabel="y / step", title="Host position loop: step (dashed = model)")
    kp = [s["kp"] for s in summ["steps"]]
    for ax, key, lab in ((axs[0, 1], "overshoot_pct", "overshoot [%]"), (axs[0, 2], "settle_ms", "settling [ms]")):
        ax.plot(kp, [s.get(key) for s in summ["steps"]], "o-", label="measured")
        ax.plot(kp, [s.get("model_" + key) for s in summ["steps"]], "k--", lw=.9, label="model")
        ax.set(xlabel="Kp [1/s]", ylabel=lab)
    axs[0, 2].plot(kp, [s.get("rise_ms") for s in summ["steps"]], "s-", label="rise 10-90 % measured")
    ax2 = axs[0, 1].twinx()
    ax2.plot(kp, [s["model_pm_deg"] for s in summ["steps"]], "C2:", label="model phase margin")
    ax2.set_ylabel("phase margin [deg]", color="C2")
    m = res.get("model")
    fgrid = np.logspace(np.log10(min([s["f"] for s in summ["sines"]] or [0.5])) - 0.2, 1.5, 300)
    for k, ff in enumerate((False, True)):
        ss = [s for s in summ["sines"] if s["ff"] == ff]
        if not ss:
            continue
        lab = "with feedforward" if ff else "feedback only"
        f = [s["f"] for s in ss]
        axs[1, 0].semilogx(f, [20 * np.log10(s["gain"]) for s in ss], "os"[k], color=f"C{k}", label=lab)
        axs[1, 1].semilogx(f, [s["phase_deg"] for s in ss], "os"[k], color=f"C{k}", label=lab)
        axs[1, 2].semilogx(f, [s["err_rms_pct"] for s in ss], "os"[k] + "-", color=f"C{k}", label=lab)
        if m:
            H = closed_loop_frf(fgrid, ss[0]["kp"], m, ff, res["adv"] if ff else 0.0)
            axs[1, 0].semilogx(fgrid, 20 * np.log10(np.abs(H)), "--", color=f"C{k}", lw=.9, label=lab + " (model)")
            axs[1, 1].semilogx(fgrid, np.degrees(np.unwrap(np.angle(H))), "--", color=f"C{k}", lw=.9)
    axs[1, 0].axhline(-3, color="gray", ls=":", lw=.8)
    kps = summ["sines"][0]["kp"] if summ["sines"] else math.nan
    axs[1, 0].set(xlabel="Hz", ylabel="dB", title=f"Closed-loop q / q_ref (Kp {kps:g})")
    axs[1, 1].set(xlabel="Hz", ylabel="deg", title="Closed-loop phase")
    md = summ["model"]
    axs[1, 2].set(xlabel="Hz", ylabel="% of amplitude",
                  title=f"Tracking error RMS (fb age {md['fb_age_ms']:.1f} ms, P {md['P_ms']:g} ms)")
    for a in axs.flat:
        a.grid(alpha=.3, which="both"); a.legend(fontsize=8)
    _plain_log_x(axs[1, 2])
    _save(fig, out, "15_servo.png", plt)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    fl = lambda s: [float(x) for x in s.split(",") if x.strip()]
    il = lambda s: [int(x) for x in s.split(",") if x.strip()]
    p.add_argument("--ip", default="192.168.0.100")
    p.add_argument("--sim", action="store_true", help="dry run against a fake robot model")
    p.add_argument("--reanalyze", default=None, metavar="DIR", help="redo analysis + plots of a v2 run folder")
    p.add_argument("--joint", type=int, default=6, choices=range(1, 7))
    p.add_argument("--tests", default="step,impulse,sine,pingpong,accsweep",
                   help="names and/or groups: " + ", ".join(f"{k}={'+'.join(v)}" for k, v in TEST_GROUPS.items()
                                                           if k != "all") + ", all")
    p.add_argument("--period", type=float, default=0.002, help="command streaming / servo period [s]")
    p.add_argument("--monitor-interval", type=float, default=0.001, help="SetMonitoringInterval [s]")
    p.add_argument("--vel-limit-pct", type=float, default=25, help="SetJointVelLimit [%%]")
    p.add_argument("--joint-acc-pct", type=float, default=None,
                   help="SetJointAcc [%%] (default: untouched, 80 %% after activation)")
    p.add_argument("--vel-timeout", type=float, default=0.05, help="SetVelTimeout [s]")
    p.add_argument("--amp", type=float, default=None, help="velocity amplitude [deg/s], default 50 %% of the limit")
    p.add_argument("--max-excursion", type=float, default=45, help="abort beyond this many deg from start")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--rt-reader", choices=("callback", "poll"), default="callback",
                   help="how real-time frames are read: mecademicpy end-of-cycle callback (recommended), or a "
                        "polling thread (2026-11-25: loses 38 %% of frames instead of 16 %%, and with "
                        "--high-priority it stalled the control loop for 48 ms)")
    p.add_argument("--high-priority", action="store_true",
                   help="HIGH process priority + 1 ms timer resolution (Windows), nice -10 elsewhere; "
                        "recommended with the callback reader (lower feedback age and send spikes)")
    p.add_argument("--joint-vmax", type=fl, default=None,
                   help="joint speed limits at 100 %% [deg/s], 6 values; overrides the R3 values (see vmaxcheck)")
    g = p.add_argument_group("velocity tests")
    g.add_argument("--vmax-joints", type=il, default=il("1,2,3,4,5,6"), help="joints checked by vmaxcheck")
    g.add_argument("--vmax-vl-pct", type=float, default=10.0, help="SetJointVelLimit used by vmaxcheck [%%]")
    g.add_argument("--step-hold", type=float, default=0.3, help="duration of each step level [s]")
    g.add_argument("--pulse-widths", type=fl, default=fl("1,2,5,10,20,50"), help="ms, rounded to --period")
    g.add_argument("--freqs", type=fl, default=fl("0.5,1,2,3,5,7,10,14,20"))
    g.add_argument("--sine-acc", type=float, default=None,
                   help="cap on A*2*pi*f [deg/s^2]; default 50 %% of the accel measured in the step test")
    g.add_argument("--pp-threshold", type=float, default=0.15, help="ping-pong switch level, fraction of amp")
    g.add_argument("--pp-duration", type=float, default=3.0)
    g.add_argument("--acc-sweep-pcts", type=fl, default=fl("5,10,15,20,25,30,50,100"),
                   help="SetJointVelLimit values of accsweep [%%]")
    g.add_argument("--jacc-pcts", type=fl, default=fl("25,50,80,100,150"),
                   help="SetJointAcc values of jaccsweep [%%] (refused values are skipped)")
    g.add_argument("--jacc-vl-pcts", type=fl, default=fl("10,25"),
                   help="SetJointVelLimit values of jaccsweep: one on the T floor, one above")
    g.add_argument("--amp-fracs", type=fl, default=fl("0.1,0.3,0.6,0.9"),
                   help="ampsweep amplitudes, fraction of the velocity limit")
    g.add_argument("--multi-joints", type=il, default=il("5,6"), help="joints moved by multijoint")
    g.add_argument("--multi-frac", type=float, default=0.5, help="multijoint amplitude, fraction of each joint limit")
    g.add_argument("--ps-periods", type=fl, default=fl("1,2,5,10,20"), help="periodsweep command periods [ms]")
    g.add_argument("--ps-freq", type=float, default=3.0, help="periodsweep sine frequency [Hz]")
    g = p.add_argument_group("position tests")
    g.add_argument("--pos-steps", type=fl, default=fl("0.5,2,5,15"), help="posstep move sizes [deg]")
    g.add_argument("--pos-joint-vel-pct", type=float, default=50, help="SetJointVel for posstep / posstream [%%]")
    g.add_argument("--pos-pause", type=float, default=0.3, help="rest between posstep moves [s]")
    g.add_argument("--pos-tol", type=float, default=0.005, help="settling tolerance on actual position [deg]")
    g.add_argument("--pos-periods", type=fl, default=fl("10,20,50"), help="posstream MoveJoints periods [ms]")
    g.add_argument("--pos-freqs", type=fl, default=fl("0.5,1,2"), help="posstream sine frequencies [Hz]")
    g.add_argument("--pos-amp", type=float, default=5.0, help="posstream sine amplitude [deg]")
    g.add_argument("--blending", type=float, default=100.0, help="SetBlending for posstream [%%]")
    g.add_argument("--pos-backlog-timeout", type=float, default=60.0, help="max wait for the queue to drain [s]")
    g.add_argument("--int-dist", type=float, default=20.0, help="posinterrupt move size [deg]")
    g.add_argument("--int-after", type=float, default=0.2, help="posinterrupt: ClearMotion after [s]")
    g.add_argument("--int-joint-vel-pct", type=float, default=10.0, help="posinterrupt SetJointVel [%%]")
    g = p.add_argument_group("external position loop (servo)")
    g.add_argument("--servo-kps", type=fl, default=fl("5,10,20,30,45"), help="Kp of the step tests [1/s]")
    g.add_argument("--servo-step", type=float, default=1.0, help="step size [deg]")
    g.add_argument("--servo-hold", type=float, default=0.6, help="step hold [s]")
    g.add_argument("--servo-kp", type=float, default=20.0, help="Kp of the sine tests [1/s]")
    g.add_argument("--servo-freqs", type=fl, default=fl("0.5,1,2,3,5,8,12"), help="sine tracking frequencies [Hz]")
    g.add_argument("--servo-amp", type=float, default=2.0, help="sine amplitude [deg] (reduced at high f)")
    g.add_argument("--no-servo-ff", dest="servo_ff", action="store_false", help="skip the feedforward runs")
    g.add_argument("--servo-adv-ms", type=float, default=None,
                   help="feedforward advance [ms]; default tau + T/2 + P/2 from this run's step test")
    g.add_argument("--servo-fb", choices=("actual", "target"), default="actual", help="position fed back")
    g.add_argument("--servo-abort", type=float, default=10.0, help="stop a run if |error| exceeds this [deg]")
    g.add_argument("--servomulti-freqs", type=fl, default=fl("1,3,5"),
                   help="servomulti sine frequencies [Hz] (joints = --multi-joints)")
    g = p.add_argument_group("replay (policy-like action streams, joints = --multi-joints)")
    g.add_argument("--replay-modes", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=["vel", "ff", "pos", "posi"],
                   help="interfaces: vel = host loop (Kp --servo-kp) + feedforward, ff = feedforward only, "
                        "pos = one MoveJoints per action with blending, posi = MoveJoints every --pos-interp-ms "
                        "to the host-interpolated position, --posi-adv-ms ahead")
    g.add_argument("--ref-file", default=None,
                   help="CSV of actions: t [s], q1..q6 [deg] (header allowed); default: synthetic motion")
    g.add_argument("--ref-absolute", dest="ref_relative", action="store_false",
                   help="--ref-file holds absolute joint angles (its first row becomes the start pose offset)")
    g.add_argument("--ref-period-ms", type=float, default=20.0, help="synthetic action period [ms] (50 Hz)")
    g.add_argument("--ref-amp", type=float, default=5.0, help="synthetic amplitude per joint [deg] (capped by speed)")
    g.add_argument("--replay-dur", type=float, default=8.0, help="synthetic duration [s]")
    g.add_argument("--ref-chunk", type=int, default=1, help="actions published by chunks of N (1 = streaming)")
    g.add_argument("--ref-chunk-offset-deg", type=float, default=0.0, help="random jump between chunks [deg]")
    g.add_argument("--ref-noise-deg", type=float, default=0.0, help="white noise per action [deg]")
    g.add_argument("--ref-jitter-ms", type=float, default=0.0, help="mean late publication per chunk [ms]")
    g.add_argument("--ref-lead-ms", type=float, default=0.0,
                   help="publish each chunk this early, before its first action is due [ms]")
    g.add_argument("--ref-seed", type=int, default=0, help="random seed of the synthetic actions")
    g.add_argument("--replay-settle", type=float, default=0.5, help="start of the analysis window [s]")
    g.add_argument("--ref-predict", choices=("linear", "quad"), default="linear",
                   help="extrapolation past the last released action: last slope, or slope + its last change")
    g.add_argument("--pos-interp-ms", type=float, default=5.0, help="posi mode: one MoveJoints every ... [ms]")
    g.add_argument("--posi-adv-ms", type=float, default=None,
                   help="posi mode: send the position this far ahead [ms]; default tau + T/2 + P/2 + servo lag")
    g.add_argument("--ref-smooth", type=int, default=0,
                   help="track the actions through a centred Savitzky-Golay filter over N actions (odd, >= 3; "
                        "0 = raw actions); each smoothed action waits for the N // 2 next ones")
    g = p.add_argument_group("soak (worst cases, joints = --multi-joints, loop = --servo-kp + feedforward)")
    g.add_argument("--soak-dur", type=float, default=120.0, help="duration per load [s]")
    g.add_argument("--soak-loads", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=["idle", "cpu", "gil"],
                   help="loads, one phase each: idle, cpu (busy processes), gil (busy Python thread in this process)")
    g.add_argument("--stress-procs", type=int, default=0, help="busy processes of the 'cpu' load (0 = one per core)")
    g.add_argument("--soak-amp", type=float, default=2.0, help="amplitude per joint [deg] (capped by speed)")
    g = p.add_argument_group("external sensor (servo, servomulti, replay vel/ff, sensorsweep)")
    g.add_argument("--fb-delay-ms", type=float, default=0.0, help="extra delay of the position fed back [ms]")
    g.add_argument("--fb-rate-hz", type=float, default=0.0,
                   help="sensor rate: the position is sampled and held at this rate (0 = every loop tick)")
    g.add_argument("--fb-noise-deg", type=float, default=0.0, help="white noise on the position fed back [deg]")
    g.add_argument("--sensor-rates", type=fl, default=fl("0,120,60,30"), help="sensorsweep rates [Hz], 0 = encoder")
    g.add_argument("--sensor-delays", type=fl, default=fl("0,20"), help="sensorsweep extra delays [ms]")
    g.add_argument("--sensor-freqs", type=fl, default=fl("1,2"), help="sensorsweep sine frequencies [Hz]")
    g.add_argument("--sensor-pm", type=float, default=60.0, help="sensorsweep: Kp for this phase margin [deg]")
    g.add_argument("--sensor-base-fb-ms", type=float, default=3.7,
                   help="sensorsweep: feedback age before the sensor, to choose Kp [ms] (measured 2026-11-25)")
    g = p.add_argument_group("Cartesian velocity (cartvel)")
    g.add_argument("--cart-axes", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=["x", "z", "rz"], help="axes among x, y, z, rx, ry, rz")
    g.add_argument("--cart-frame", choices=("wrf", "trf"), default="wrf", help="MoveLinVelWrf or MoveLinVelTrf")
    g.add_argument("--cart-pose", type=fl, default=fl("0,0,0,0,45,0"),
                   help="joint pose [deg] the Cartesian steps start from (home has J5 = 0: wrist singularity)")
    g.add_argument("--cart-amp-mm", type=float, default=20.0, help="linear step [mm/s]")
    g.add_argument("--cart-amp-deg", type=float, default=10.0, help="angular step [deg/s]")
    g.add_argument("--cart-max-mm", type=float, default=15.0, help="linear excursion bound [mm]")
    g.add_argument("--cart-max-deg", type=float, default=10.0, help="angular excursion bound [deg]")
    g.add_argument("--cart-acc-pcts", type=fl, default=[], help="SetCartAcc values to sweep [%%] (default: as is)")
    g.add_argument("--model-tau", type=float, default=None, help="model delay for predictions [ms] (no step test)")
    g.add_argument("--model-T", type=float, default=None, help="model moving average [ms] (no step test)")
    p.add_argument("--out", default=None, help="output folder")
    p.add_argument("--show", action="store_true", help="open the plots at the end")
    return p.parse_args(argv)


def apply_joint_vmax(a):
    """--joint-vmax replaces the nominal R3 limits everywhere (the list is shared, so mutate it)."""
    v = getattr(a, "joint_vmax", None)
    if v:
        if len(v) != NJ:
            sys.exit(f"--joint-vmax needs {NJ} values, got {len(v)}")
        JOINT_VMAX[:] = [float(x) for x in v]


def set_high_priority():
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00000080)  # HIGH_PRIORITY_CLASS
            ctypes.windll.winmm.timeBeginPeriod(1)
        else:
            os.nice(-10)
        print("Process priority raised")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Could not raise the process priority: {e}")
        return False


def clock_pair(n=20):
    """(time.time(), perf_counter()) read as close together as possible: the tightest of n tries."""
    best = None
    for _ in range(n):
        a = time.perf_counter()
        w = time.time()
        b = time.perf_counter()
        if best is None or b - a < best[2]:
            best = (w, 0.5 * (a + b), b - a)
    return best[:2]


def expand_tests(spec):
    tests = []
    for t in (x.strip() for x in spec.split(",")):
        for x in TEST_GROUPS.get(t, [t] if t else []):
            if x not in tests:
                tests.append(x)
    return tests


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _f(x, fmt="6.2f"):
    return "   n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, fmt)


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
        md = s.get("model")
        if md:
            print(f"  MODEL target vel = delay tau + moving average T: tau {md['tau_ms']:.2f} ms, T {md['T_ms']:.2f} ms "
                  f"(fit rms {md['rms']:.3f} deg/s) -> eq. delay {md['eq_delay_ms']:.2f} ms, -3 dB at "
                  f"{md['bw_3db_hz']:.1f} Hz, first zero at {md['first_zero_hz']:.1f} Hz")
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
        print("\nACCSWEEP: SetJointVelLimit -> moving average T")
        for r in S["accsweep"]:
            print(f"  {r['pct']:5.0f} % | T {_f(r.get('T_ms'))} ms | tau {_f(r.get('tau_ms'))} ms | target accel "
                  f"{_f(r.get('target_accel_deg_s2'), '8.0f')} deg/s2 | rise {_f(r.get('target_rise_10_90_ms'))} ms")
    if "jaccsweep" in S:
        print("\nJACCSWEEP: SetJointAcc -> T")
        for r in S["jaccsweep"]:
            if r.get("failed"):
                print(f"  VelLimit {r['vl']:5.0f} % | SetJointAcc {r['acc']:5.0f} % | refused by the robot")
            else:
                print(f"  VelLimit {r['vl']:5.0f} % | SetJointAcc {r['acc']:5.0f} % | T {_f(r['T_ms'])} ms | "
                      f"tau {_f(r['tau_ms'])} ms | T*acc/100 {_f(r['T_ms'] * r['acc'] / 100)} ms")
    if "ampsweep" in S:
        print("\nAMPSWEEP: velocity step size -> T (constant T = linear system)")
        for r in S["ampsweep"]:
            print(f"  |dv| {r['dv']:6.1f} deg/s | T {_f(r['T_ms'])} ms | tau {_f(r['tau_ms'])} ms | "
                  f"|dv|/T {_f(r['acc_eff'], '8.0f')} deg/s2 | fit residual {_f(r['rel_rms_pct'])} %")
    if "multijoint" in S:
        print("\nMULTIJOINT: T per joint, alone vs together")
        for r in S["multijoint"]:
            print(f"  {r['scenario']:10s} J{r['joint']} ({r['A']:5.1f} deg/s) | T {_f(r.get('T_ms'))} ms | "
                  f"tau {_f(r.get('tau_ms'))} ms | servo lag {_f(r.get('servo_lag_ms'))} ms")
    if S.get("servomulti"):
        x = S["servomulti"]
        print(f"\nSERVOMULTI (host loop on J{', J'.join(str(j) for j in x['joints'])} at once, Kp {x['kp']:g}, "
              f"feedback age {_f((x['fb_age_ms'] or {}).get('median'))} ms)" + ("  ABORTED" if x["aborted"] else ""))
        for r in x["rows"]:
            print(f"  {r['f']:5.2f} Hz {'FF' if r['ff'] else '  '} J{r['joint']} | gain {r['gain']:6.3f} (model "
                  f"{r['model_gain']:6.3f}) | phase {r['phase_deg']:7.1f} (model {r['model_phase_deg']:7.1f}) deg | "
                  f"error rms {r['err_rms_pct']:6.1f} % (model {r['model_err_rms_pct']:5.1f}) | sat {r['sat_pct']:.0f} %")
    if "vmaxcheck" in S:
        print("\nVMAXCHECK: real joint speed limit (target velocity plateau / SetJointVelLimit)")
        for r in S["vmaxcheck"]:
            if r.get("failed"):
                print(f"  J{r['joint']}: refused ({r['failed']})")
            elif not r["clamped"]:
                print(f"  J{r['joint']}: asked {r['asked']:.1f} deg/s and got it ({_f(r['plateau'], '.1f')}): "
                      "not clamped, the limit is higher than assumed")
            else:
                print(f"  J{r['joint']}: plateau {r['plateau']:6.2f} deg/s at {r['vel_limit_pct']:g} % -> "
                      f"vmax {r['vmax_measured']:6.1f} deg/s (assumed {r['vmax_assumed']:g})")
    if "periodsweep" in S:
        print("\nPERIODSWEEP: command period -> delay")
        for r in S["periodsweep"]:
            print(f"  P {r['P_ms']:5.1f} ms | delay target {_f(r.get('tv_delay_ms'))} ms (host sees "
                  f"{_f(r.get('tv_host_delay_ms'))}) | predicted tau+T/2+P/2 {_f(r['pred_delay_ms'])} ms | "
                  f"gain {_f(r.get('tv_gain'), '6.3f')}")
    if "posstep" in S:
        print("\nPOSSTEP (MoveJoints, medians, ms)")
        for r in S["posstep"]:
            print(f"  {r['size']:6.2f} deg | dead {_f(r.get('dead_ms'))} (host {_f(r.get('dead_host_ms'))}) | "
                  f"target at goal {_f(r.get('target_done_ms'), '7.1f')} | actual settled "
                  f"{_f(r.get('actual_settle_ms'), '7.1f')} | peak {_f(r.get('peak_vel'), '6.1f')} deg/s | "
                  f"final err {_f(r.get('final_err_deg'), '8.5f')} deg")
    if "posstream" in S:
        print("\nPOSSTREAM (MoveJoints + blending)  P | f | delay tgt / act [ms] | gain tgt | drift [ms/s] | "
              "stall % | backlog [ms]")
        for r in S["posstream"]:
            print(f"  {r['P_ms']:5.1f} ms | {r['f']:4.2f} Hz | {_f(r.get('tgt_delay_ms'), '7.1f')} / "
                  f"{_f(r.get('act_delay_ms'), '7.1f')} | {_f(r.get('tgt_gain'), '5.3f')} | "
                  f"{_f(r.get('tgt_delay_drift_ms_per_s'), '6.1f')} | {_f(r.get('stall_pct'), '5.1f')} | "
                  f"{_f(r.get('backlog_ms'), '7.1f')}")
    if "posinterrupt" in S:
        x = S["posinterrupt"]
        print(f"\nPOSINTERRUPT ({x['n']} interruptions, medians, ms)")
        for k in ("clear_to_decel_ms", "decel_ms", "clear_to_stop_ms", "stop_to_reverse_ms", "clear_to_reverse_ms"):
            if x.get(k):
                print(f"  {k:24s} {x[k]['median']:7.2f}   (p95 {x[k]['p95']:.2f})")
    if S.get("servo"):
        x = S["servo"]
        md = x["model"]
        print(f"\nSERVO (host loop, feedback {x['feedback']}, P {md['P_ms']:g} ms, feedback age "
              f"{ms(x['fb_age_ms'])} ms; model tau {md['tau_ms']:.1f} T {md['T_ms']:.1f} ms from {md['source']})")
        for s in x["steps"]:
            print(f"  step Kp {s['kp']:5.1f} | rise {_f(s.get('rise_ms'))} (model {_f(s.get('model_rise_ms'))}) ms | "
                  f"overshoot {_f(s.get('overshoot_pct'))} (model {_f(s.get('model_overshoot_pct'))}) % | settle "
                  f"{_f(s.get('settle_ms'), '6.0f')} (model {_f(s.get('model_settle_ms'), '6.0f')}) ms | model PM "
                  f"{_f(s['model_pm_deg'], '5.1f')} deg fc {_f(s['model_fc_hz'], '4.1f')} Hz | sat {s['sat_pct']:.0f} %"
                  + ("  ABORTED" if s["aborted"] else ""))
        for s in x["sines"]:
            print(f"  sine {s['f']:5.2f} Hz {'FF' if s['ff'] else '  '} | gain {s['gain']:6.3f} (model "
                  f"{s['model_gain']:6.3f}) | phase {s['phase_deg']:7.1f} (model {s['model_phase_deg']:7.1f}) deg | "
                  f"error rms {s['err_rms_pct']:6.1f} % | sat {s['sat_pct']:.0f} %")
    if S.get("replay"):
        x = S["replay"]
        print(f"\nREPLAY ({x['source']} actions every {x['P_action_ms']:.0f} ms on J{', J'.join(str(j) for j in x['joints'])}, "
              f"chunks of {x['chunk']}, noise {x['noise_deg']:g} deg, jitter {x['jitter_ms']:g} ms, chunk offset "
              f"{x['chunk_offset_deg']:g} deg" + (f", smoothed over {x['smooth']} actions" if x.get("smooth") else "")
              + f"; vel: Kp {x['kp']:g}, feedforward {x['feedforward_advance_ms']:.1f} ms ahead)")
        clean = any("clean_rel_err_pct" in r for r in x["per_mode"])
        print("  mode | error % of ref (median over joints) | delay ms | error once delayed % | max err deg | "
              "accel rms deg/s2 | stall % | sat % | backlog ms"
              + (" || vs clean motion: error % | once delayed % | accel / clean accel  (actions off by ...%)" if clean else ""))
        for r in x["per_mode"]:
            print(f"  {r['mode']:4s} | {_f(r['rel_err_pct'], '6.1f')} | {_f(r['delay_ms'], '6.1f')} | "
                  f"{_f(r['shape_rel_pct'], '6.1f')} | {_f(r['max_err_deg'], '6.3f')} | {_f(r['accel_rms'], '7.0f')} | "
                  f"{_f(r['stall_pct'], '5.1f')} | {_f(r['sat_pct'], '5.1f')} | {_f(r['backlog_ms'], '6.1f')}"
                  + (f" || {_f(r.get('clean_rel_err_pct'), '6.1f')} | {_f(r.get('clean_shape_rel_pct'), '6.1f')} | "
                     f"{_f(r.get('accel_ratio'), '5.2f')}  ({_f(r.get('input_dev_pct'), '4.1f')} %)" if clean else "")
                  + ("  STOPPED EARLY" if r.get("aborted") else ""))
    if S.get("soak"):
        x = S["soak"]
        print(f"\nSOAK ({x['dur_s']:g} s per load on J{', J'.join(str(j) for j in x['joints'])}, Kp {x['kp']:g}, "
              f"period {x['P_ms']:g} ms; median / p99 / p99.9 / max in ms)")
        tl = lambda d: "n/a" if not d else f"{d['median']:5.1f} / {d['p99']:5.1f} / {d['p999']:5.1f} / {d['max']:6.1f}"
        for r in x["phases"]:
            print(f"  {r['load']:4s}{f' x{r['procs']}' if r['procs'] else '':4s} | feedback age {tl(r['fb_age_ms'])} | "
                  f"send interval {tl(r['send_interval_ms'])} | frame gap {tl(r['host_rx_gap_ms'])}"
                  + ("  STOPPED EARLY" if r.get("aborted") else ""))
            print(f"            missing frames {r['missing_frames_pct']:4.1f} % | ticks without new feedback "
                  f"{r['stale_ticks_pct']:4.1f} % (longest {r['longest_no_feedback_ms']:.1f} ms) | age > 10/20/50 ms: "
                  f"{r['n_fb_age_over_10ms']}/{r['n_fb_age_over_20ms']}/{r['n_fb_age_over_50ms']} | send gap > vel timeout: "
                  f"{r['n_send_gap_over_vel_timeout']} | error rms {r['err_rms_deg']:.3f} max {r['err_max_deg']:.3f} deg | "
                  f"crossover at 60 deg: {_f(r.get('fc60_hz_median_age'), '4.1f')} Hz on the median age, "
                  f"{_f(r.get('fc60_hz_p999_age'), '4.1f')} Hz on its p99.9")
    if S.get("cartvel"):
        print("\nCARTVEL (MoveLinVel steps: target velocity = delay tau + moving average T)")
        for r in S["cartvel"]:
            acc = "" if r["cart_acc_pct"] is None else f" SetCartAcc {r['cart_acc_pct']:g} %"
            print(f"  {r['axis']:2s} +/-{r['amplitude']:g} {r['unit']}{acc} | tau {_f(r['tau_ms'])} ms | "
                  f"T {_f(r['T_ms'])} ms | eq. delay {_f(r['eq_delay_ms'])} ms | servo lag "
                  f"{_f(r.get('servo_lag_ms'))} ms | fit rms {_f(r['fit_rms'], '.3f')}")
    if S.get("sensorsweep"):
        x = S["sensorsweep"]
        print(f"\nSENSORSWEEP (host loop through an emulated sensor, Kp for {x['pm']:g} deg of phase margin, "
              f"noise {x['noise_deg']:g} deg)")
        for r in x["rows"]:
            gains = " | ".join(f"{k[5:]} {r[k]:.3f} (model {r['model_' + k]:.3f})" for k in r if k.startswith("gain_"))
            print(f"  {'encoder' if r['rate_hz'] <= 0 else f'{r['rate_hz']:g} Hz':8s} +{r['delay_ms']:3g} ms | fb age "
                  f"{_f(r['fb_age_ms'], '5.1f')} ms | Kp {r['kp']:5.1f} | model fc {_f(r['model_fc_hz'], '4.1f')} Hz PM "
                  f"{_f(r['model_pm_deg'], '4.0f')} | overshoot {_f(r['overshoot_pct'], '5.1f')} (model "
                  f"{_f(r['model_overshoot_pct'], '5.1f')}) % | gain {gains}")
    if "external_loop_prediction" in S:
        x = S["external_loop_prediction"]
        print(f"\nPREDICTED HOST POSITION LOOP (P controller via MoveJointsVel; model from {x['model_source']})")
        print(f"  loop delay tau + T/2 + P/2 + servo + feedback = {x['tau_ms']:.1f} + {x['T_ms'] / 2:.1f} + "
              f"{x['P_ms'] / 2:.1f} + {x['servo_lag_ms']:.1f} + {x['fb_ms']:.1f} = {x['loop_delay_ms']:.1f} ms | "
              f"velocity interface -3 dB at {x['vel_bw_3db_hz']:.1f} Hz")
        for pm in (60, 45):
            r = x[f"pm{pm}"]
            print(f"  phase margin {pm} deg: Kp {r['kp']:5.1f} 1/s -> crossover {r['fc_hz']:4.1f} Hz, "
                  f"gain margin {_f(r['gm_db'], '4.1f')} dB")
        if x.get("sensor_table"):
            lats = sorted({r["latency_ms"] for r in x["sensor_table"]})
            print("  external sensor -> crossover at 60 deg [Hz]; rows: sensor rate, columns: total latency [ms]")
            print("            " + "".join(f"{lat:>7g}" for lat in lats))
            for rate in sorted({r["rate_hz"] for r in x["sensor_table"]}, reverse=True):
                vals = {r["latency_ms"]: r["fc60_hz"] for r in x["sensor_table"] if r["rate_hz"] == rate}
                print(f"  {rate:6g} Hz " + "".join(f"{vals[lat]:7.1f}" for lat in lats))
    if S.get("ecat_timing"):
        x = S["ecat_timing"]
        tl = lambda d: "n/a" if not d else f"{d['median']:6.3f} / {d['p99']:6.3f} / {d['p999']:6.3f} / {d['max']:7.3f}"
        print(f"\nETHERCAT BUS ({x['cycles']} cycles of {x['cycle_ms']:g} ms, byte order {x.get('byte_order')}"
              + (" (sim)" if x.get("sim") else "") + "; median / p99 / p99.9 / max in ms)")
        print(f"  wake-up late {tl(x.get('wake_late_ms'))} | tx_rx round trip {tl(x.get('txrx_roundtrip_ms'))}")
        print(f"  cycle interval {tl(x.get('cycle_interval_ms'))} | data age at the host {tl(x.get('data_age_ms'))}")
        print(f"  new robot sample in {_f(x.get('fresh_pct'), '5.1f')} % of cycles | overruns {x.get('overruns')} | "
              f"failed exchanges {x.get('failed_exchanges')} | bad WKC {x.get('bad_wkc')} | ring drops {x.get('ring_drops')}")
        if x.get("drift_ppm") is not None:
            beat = x.get("beat_s")
            print(f"  robot clock drift {x['drift_ppm']:+.1f} ppm -> beat {_f(beat, '.0f')} s, "
                  f"{x['beats_covered']:.1f} beats recorded"
                  + ("" if x.get("robot_clock_map_ok") else
                     ": less than one beat, robot-clock delays may read up to one cycle late"))
            if x.get("robot_clock_stalls") or x.get("robot_clock_jumps"):
                print(f"  robot clock stalled {x['robot_clock_stalls']} times ({x['robot_clock_stall_lost_s']:.3f} s "
                      f"in all) and jumped forward {x.get('robot_clock_jumps', 0)} times "
                      f"({x.get('robot_clock_jump_gained_s', 0.0):.3f} s): one clock offset per segment in between")
        print("  (TIMING below: MoveJointsVel_call = tx_rx round trip of the cycle that carries the command; "
              "host-seen delays have a one-cycle resolution)")
    t = S.get("timing", {})
    print("\nTIMING (median / p95, ms)")
    for k, v in t.items():
        if isinstance(v, dict):
            print(f"  {k:30s} {v['median']:7.3f} / {v['p95']:7.3f}   (max {v['max']:.3f})")
        elif v is not None:
            print(f"  {k:30s} {v:7.2f}")


def _try(name, fn):
    try:
        return fn()
    except Exception:  # noqa: BLE001
        print(f"  analysis of '{name}' failed:")
        traceback.print_exc()
        return None


def analyze_all(raw, frames, send_logs, args, out, plt):
    print(f"\nAnalysing {len(frames)} feedback frames -> {out}/")
    cmap = run_clock_map(raw, frames)
    S = {}

    def run(name, an, pl=None):
        """Analyse then plot one test; a failure is reported and skipped, never fatal."""
        r = _try(name, an)
        if r is not None:
            S[name] = r
            if pl:
                _try(name + " plot", lambda: pl(r))

    S["timing"], td = analyze_timing(frames, send_logs, cmap)
    _try("timing plot", lambda: plot_timing(td, out, plt))
    if raw.get("_ecat"):
        run("ecat_timing", lambda: analyze_ecat_timing(raw["_ecat"], frames, cmap))
    if "step" in raw:
        run("step", lambda: analyze_step(raw["step"], cmap), lambda r: plot_step(raw["step"], out, plt))
    mdl = pick_model(S, args)
    if "impulse" in raw:
        run("impulse", lambda: analyze_impulse(raw["impulse"], cmap), lambda r: plot_impulse(raw["impulse"], out, plt))
    if "sine" in raw and raw["sine"]["runs"]:
        if not mdl["src"].startswith("J6 law"):
            raw["sine"]["model"] = dict(tau=mdl["tau"], T=mdl["T"], P=args.period)
        run("sine", lambda: analyze_sine(raw["sine"], cmap),
            lambda r: (plot_bode(raw["sine"], out, plt), plot_sine_traces(raw["sine"], out, plt)))
    if "pingpong" in raw and raw["pingpong"]:
        run("pingpong", lambda: analyze_pingpong(raw["pingpong"], cmap),
            lambda r: plot_pingpong(raw["pingpong"], out, plt))
    if "vmaxcheck" in raw and raw["vmaxcheck"]["runs"]:
        run("vmaxcheck", lambda: analyze_vmaxcheck(raw["vmaxcheck"], cmap))
    simple = (("accsweep", "sweeps", analyze_accsweep, plot_accsweep),
              ("jaccsweep", "sweeps", analyze_jaccsweep, plot_jaccsweep),
              ("ampsweep", "runs", analyze_ampsweep, plot_ampsweep),
              ("multijoint", "scenarios", analyze_multijoint, plot_multijoint),
              ("periodsweep", "runs", analyze_periodsweep, plot_periodsweep),
              ("posstream", "runs", analyze_posstream, plot_posstream),
              ("posinterrupt", "events", analyze_posinterrupt, plot_posinterrupt))
    for name, key, an, pl in simple:
        if name in raw and len(raw[name][key]):
            run(name, lambda: an(raw[name], cmap), lambda r: pl(raw[name], out, plt))
    if "posstep" in raw and len(raw["posstep"]["moves"]):
        run("posstep", lambda: analyze_posstep(raw["posstep"], cmap),
            lambda r: plot_posstep(raw["posstep"], r, out, plt))
    if "servo" in raw:
        run("servo", lambda: analyze_servo(raw["servo"], cmap, mdl),
            lambda r: plot_servo(raw["servo"], r, out, plt))
    if "servomulti" in raw and raw["servomulti"]["runs"]:
        run("servomulti", lambda: analyze_servomulti(raw["servomulti"], cmap, mdl),
            lambda r: plot_servomulti(raw["servomulti"], r, out, plt))
    if "replay" in raw and raw["replay"]["runs"]:
        run("replay", lambda: analyze_replay(raw["replay"], cmap), lambda r: plot_replay(raw["replay"], r, out, plt))
    if "soak" in raw and raw["soak"]["phases"]:
        run("soak", lambda: analyze_soak(raw["soak"], frames, cmap, mdl),
            lambda r: plot_soak(raw["soak"], r, frames, cmap, out, plt))
    if "sensorsweep" in raw and raw["sensorsweep"]["configs"]:
        run("sensorsweep", lambda: analyze_sensorsweep(raw["sensorsweep"], cmap, mdl),
            lambda r: plot_sensorsweep(raw["sensorsweep"], r, out, plt))
    if "cartvel" in raw and raw["cartvel"]["runs"]:
        run("cartvel", lambda: analyze_cartvel(raw["cartvel"], cmap), lambda r: plot_cartvel(raw["cartvel"], out, plt))
    run("external_loop_prediction", lambda: predict_external_loop(S, args, mdl))
    S["robot_info"] = raw.get("_robot_info")
    S["config"] = {k: v for k, v in vars(args).items()}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(jsonable(S), f, indent=2)
    print_summary(S)
    return S


def reanalyze(folder, out=None, show=False):
    """Redo the analysis and plots of a v2 run folder (this script's, or meca_ecat's once imported
    by ecat_bridge.py). --out keeps the original summary / plots untouched. Returns the summary."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = os.path.join(folder, "raw.pkl.gz")
    if not os.path.exists(p):
        sys.exit(f"{p} not found (only runs recorded with v2 of this script can be re-analysed)")
    out = out or folder
    os.makedirs(out, exist_ok=True)
    with gzip.open(p, "rb") as f:
        D = pickle.load(f)
    saved = argparse.Namespace(**{**vars(parse_args([])), **D["args"]})
    saved.reanalyze, saved.show, saved.out = folder, show, out
    apply_joint_vmax(saved)
    S = analyze_all(D["raw"], D["frames"], D["send_logs"], saved, out, plt)
    if show:
        plt.show()
    return S


def main():
    args = parse_args()
    if args.reanalyze:
        reanalyze(args.reanalyze, args.out, args.show)
        return
    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # A 1 ms loop in Python needs the GIL to change hands quickly (default is 5 ms).
    sys.setswitchinterval(0.0002)
    apply_joint_vmax(args)
    if args.high_priority:
        set_high_priority()

    tests = expand_tests(args.tests)
    unknown = [t for t in tests if not hasattr(Tester, f"run_{t}")]
    if unknown:
        sys.exit(f"unknown test(s): {', '.join(unknown)}")
    bad = [x for x in args.soak_loads if x not in LoadGen.KINDS]
    if "soak" in tests and bad:
        sys.exit(f"unknown soak load(s): {', '.join(bad)} (use {', '.join(LoadGen.KINDS)})")
    out = args.out or f"meca_latency_{'sim_' if args.sim else ''}{datetime.now():%Y%m%d_%H%M%S}_J{args.joint}"
    os.makedirs(out, exist_ok=True)

    rec = Recorder(args.joint - 1)
    io = SimIO(args, rec) if args.sim else MecaIO(args, rec)
    raw = {"_clock": [clock_pair()]}  # wall clock <-> perf_counter, for packet captures (netcap.py)
    tester = None
    try:
        io.connect()
        raw["_robot_info"] = getattr(io, "info", None)
        tester = Tester(io, rec, args)
        print(f"Joint {args.joint}, vel limit {args.vel_limit_pct} % = {tester.vlim:.1f} deg/s, "
              f"amp {tester.amp:.1f} deg/s, period {args.period * 1e3:.1f} ms, tests: {', '.join(tests)}")
        for t in tests:
            print(f"\n== {t} ==")
            tester.home()
            raw[t] = getattr(tester, f"run_{t}")()
    except SafetyStop as e:
        print(f"\nSAFETY STOP: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        io.close()
    raw["_clock"].append(clock_pair())

    frames = rec.frames(0)
    if len(frames) < 10:
        print("No data recorded.")
        return
    send_logs = tester.send_logs if tester else []
    arrays = {"frames": frames}
    for i, c in enumerate(send_logs):
        arrays[f"cmd_{i:03d}"] = c
    np.savez_compressed(os.path.join(out, "raw.npz"), **arrays)
    with gzip.open(os.path.join(out, "raw.pkl.gz"), "wb", compresslevel=3) as f:
        pickle.dump(dict(version=2, raw=raw, frames=frames, send_logs=send_logs, args=vars(args)), f)
    analyze_all(raw, frames, send_logs, args, out, plt)
    print(f"\nResults in {os.path.abspath(out)}  (frames columns: host_rx, robot_ts, then target_pos, "
          f"target_vel, actual_pos, actual_vel of joint {args.joint}, then the same 4 signals for J1..J6)")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
