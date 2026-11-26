//! The tests, a port of class Tester of meca_latency_test.py (lines 889-1766) on top of the
//! cycle-synchronous controllers: same amplitudes, durations, safety bounds, and the same result
//! keys, so the Python analysis reads them unchanged. Scope: step, sine, accsweep, multijoint,
//! servo, servomulti, replay (vel, ff, pos, posi), soak (+ timing, always).

use std::f64::consts::PI;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Map, Value};

use crate::actionref::{piecewise, ActionRef, SensorEmu};
use crate::bridge::{quick_request, Bridge};
use crate::bus::GuardCfg;
use crate::cli::RunArgs;
use crate::out::{arr, frames_ref, num, write_npy, RunOut};
use crate::pdo::NCOL;
use crate::robot::{Robot, Stop};
use crate::stream::{LoopKind, PosStream, Profile, Sched, TrackCfg, TrackLoop, VelStream};
use crate::{rt, say};

const NAN: f64 = f64::NAN;
// J6 law identified on 2026-11-24, used only when this run has no step test (as in Python)
const J6_T_FLOOR_MS: f64 = 15.6;
const J6_T_MS_PER_PCT: f64 = 1.0;
const J6_TAU_MS: f64 = 3.5;

pub const SUPPORTED: [&str; 8] = ["step", "sine", "accsweep", "multijoint", "servo", "servomulti", "replay", "soak"];

/// Test names and groups (vel, servo, all; soak is long and never part of a group).
pub fn expand_tests(spec: &str) -> Result<Vec<String>> {
    let mut out: Vec<String> = Vec::new();
    for t in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let group: Vec<&str> = match t {
            "vel" => vec!["step", "sine", "accsweep", "multijoint"],
            "servo" => vec!["servo"],
            "all" => vec!["step", "sine", "accsweep", "multijoint", "servo", "servomulti", "replay"],
            x if SUPPORTED.contains(&x) => vec![x],
            x => bail!("unknown or not ported test '{x}' (meca_ecat runs: {}; groups vel, servo, all)", SUPPORTED.join(", ")),
        };
        for g in group {
            if !out.iter().any(|o| o == g) {
                out.push(g.to_string());
            }
        }
    }
    Ok(out)
}

fn on_joint(j: usize, f: impl Fn(f64) -> f64 + Send + 'static) -> Profile {
    Box::new(move |t| {
        let mut v = [0.0; 6];
        v[j] = f(t);
        v
    })
}

fn merge(a: Value, b: Map<String, Value>) -> Value {
    let mut m = match a {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    m.extend(b);
    Value::Object(m)
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Background load of the soak test: idle, cpu (busy child processes, normal priority, they die
/// with this process), gil (a busy thread in this process: the TCP runs' name is kept).
pub struct LoadGen {
    pub procs: usize,
    children: Vec<Child>,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl LoadGen {
    pub const KINDS: [&'static str; 3] = ["idle", "cpu", "gil"];

    pub fn start(kind: &str, procs: usize, max_s: f64) -> Result<LoadGen> {
        let mut g = LoadGen {
            procs: if procs > 0 { procs } else { rt::num_cpus() },
            children: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
        };
        match kind {
            "idle" => return Ok(g),
            "cpu" => {
                let exe = std::env::current_exe()?;
                for _ in 0..g.procs {
                    let c = Command::new(&exe)
                        .args(["burn", "--secs", &format!("{max_s}")])
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn()?;
                    g.children.push(c);
                }
            }
            "gil" | "thread" => {
                let stop = g.stop.clone();
                g.thread = Some(std::thread::spawn(move || {
                    let mut x = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        x = std::hint::black_box(x.wrapping_add(1));
                    }
                }));
            }
            other => bail!("unknown soak load '{other}' (idle, cpu, gil)"),
        }
        rt::sleep_s(0.5); // let the load settle
        Ok(g)
    }

    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for c in self.children.iter_mut() {
            let _ = c.kill();
        }
        for c in self.children.iter_mut() {
            let _ = c.wait();
        }
        self.children.clear();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for LoadGen {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The `burn` subcommand: one busy process of the cpu load.
pub fn burn(secs: f64) {
    rt::die_with_parent();
    let t_end = rt::now() + secs;
    let mut x = 0u64;
    while rt::now() < t_end {
        for _ in 0..10_000 {
            x = std::hint::black_box(x.wrapping_add(1));
        }
    }
}

pub struct Tester<'t, 'r> {
    pub io: &'t mut Robot<'r>,
    pub a: &'t RunArgs,
    pub out: &'t mut RunOut,
    pub bridge: &'t Bridge,
    pub py_args: &'t Map<String, Value>,
    pub start_joints: [f64; 6],
    j: usize,
    vmax_tab: [f64; 6],
    vlim: f64,
    vlim_all: [f64; 6],
    amp: f64,
    cycle: f64,
    start_pos: f64,
    start_all: [f64; 6],
    measured_acc: Option<f64>,
    quick_servo_lag: Option<f64>,
    quick: Option<(f64, f64)>,
}

struct LoopRes {
    map: Map<String, Value>,
    aborted: bool,
}

impl<'t, 'r> Tester<'t, 'r> {
    pub fn new(
        io: &'t mut Robot<'r>,
        a: &'t RunArgs,
        out: &'t mut RunOut,
        bridge: &'t Bridge,
        py_args: &'t Map<String, Value>,
        start_joints: [f64; 6],
    ) -> Self {
        let j = a.joint as usize - 1;
        let vmax_tab = a.vmax();
        let vlim = vmax_tab[j] * a.vel_limit_pct / 100.0;
        let vlim_all = vmax_tab.map(|v| v * a.vel_limit_pct / 100.0);
        let mut amp = a.amp.filter(|x| *x > 0.0).unwrap_or(0.5 * vlim);
        if amp > 0.95 * vlim {
            say!("--amp {amp} is above 95 % of the velocity limit ({vlim:.1} deg/s): clamped");
            amp = 0.95 * vlim;
        }
        let cycle = io.cycle_s;
        Tester {
            io,
            a,
            out,
            bridge,
            py_args,
            start_joints,
            j,
            vmax_tab,
            vlim,
            vlim_all,
            amp,
            cycle,
            start_pos: start_joints[j],
            start_all: start_joints,
            measured_acc: None,
            quick_servo_lag: None,
            quick: None,
        }
    }

    pub fn describe(&self, tests: &[String]) -> String {
        format!(
            "Joint {}, vel limit {} % = {:.1} deg/s, amp {:.1} deg/s, period {:.1} ms, tests: {}",
            self.a.joint,
            self.a.vel_limit_pct,
            self.vlim,
            self.amp,
            self.a.period * 1e3,
            tests.join(", ")
        )
    }

    // -- helpers ------------------------------------------------------------------------------

    /// Back to the start joints; the guard measures the excursion from there.
    pub fn home(&mut self) -> Result<()> {
        self.io.set_guard(None)?;
        self.io.move_joints_wait(self.start_joints, 60.0)?;
        let q = self.io.target_joints().ok_or_else(|| anyhow!("no feedback"))?;
        self.start_pos = q[self.j];
        self.start_all = q;
        self.io.set_guard(Some(GuardCfg { start: q, max_exc: self.a.max_excursion }))
    }

    /// Command -> actual position delay tau + T/2 + servo lag + P/2 (this run's step test, else
    /// the J6 law).
    fn model_delay(&self) -> f64 {
        let (tau, t) = self.quick.unwrap_or((J6_TAU_MS, J6_T_FLOOR_MS.max(J6_T_MS_PER_PCT * self.a.vel_limit_pct)));
        let lag = self.quick_servo_lag.unwrap_or(0.0).max(0.0);
        (tau + t / 2.0 + lag) * 1e-3 + self.a.period / 2.0
    }

    fn sensor(&self) -> SensorEmu {
        SensorEmu::new(self.a.fb_delay_ms * 1e-3, self.a.fb_rate_hz, self.a.fb_noise_deg, self.a.ref_seed)
    }

    /// MoveJointsVel(profile(k P)) for `duration` (+ zero and 0.15 s of trailing feedback).
    fn stream(&mut self, profile: Profile, duration: f64) -> Result<(Value, Value)> {
        let s = VelStream::new(profile, self.j, duration, self.a.period, self.cycle);
        let s = self.io.exec(s, duration + 5.0)?;
        let cmd = self.out.npy("cmd", &s.log, VelStream::COLS)?;
        self.out.send_logs.push(cmd.clone());
        Ok((cmd, frames_ref(s.i0, s.i1)))
    }

    fn quick_estimates(&mut self, step: &Value) {
        match self.try_quick(step) {
            Ok(q) => {
                self.measured_acc = q.measured_acc;
                self.quick_servo_lag = q.servo_lag_ms;
                if let Some(a) = q.measured_acc {
                    say!("  measured target acceleration ~ {a:.0} deg/s^2");
                }
                if let (Some(tau), Some(t)) = (q.tau_ms, q.t_ms) {
                    self.quick = Some((tau, t));
                    say!("  model: tau ~ {tau:.2} ms, T ~ {t:.2} ms");
                }
            }
            Err(e) => say!("  quick model not available ({e:#}): J6 law"),
        }
    }

    fn try_quick(&mut self, step: &Value) -> Result<crate::bridge::QuickModel> {
        let n = self.io.frames_recorded();
        let frames = self.io.frames(0, n)?;
        let flat: Vec<f64> = frames.iter().flat_map(|r| r.iter().copied()).collect();
        let rel = "npy/_frames_quick.npy";
        let path = self.out.dir.join(rel);
        write_npy(&path, &flat, &[frames.len(), NCOL])?;
        let req = quick_request(&self.out.dir, rel, step)?;
        let q = self.bridge.quick_model(&req);
        let _ = std::fs::remove_file(&req);
        let _ = std::fs::remove_file(&path);
        q
    }

    // -- velocity mode ------------------------------------------------------------------------

    pub fn run_step(&mut self, amp: Option<f64>, repeats: Option<usize>) -> Result<Value> {
        let a = amp.unwrap_or(self.amp);
        let hold = self.a.step_hold.min(0.7 * self.a.max_excursion / a);
        let segs = [(0.15, 0.0), (hold, a), (hold, 0.0), (hold, -a), (0.35, 0.0)];
        let dur: f64 = segs.iter().map(|s| s.0).sum();
        let mut runs = Vec::new();
        for r in 0..repeats.unwrap_or(self.a.repeats) {
            say!("  step run {}: +/-{:.1} deg/s, hold {:.0} ms", r + 1, a, hold * 1e3);
            let (cmd, fr) = self.stream(on_joint(self.j, piecewise(&segs)), dur)?;
            runs.push(json!({"cmd": cmd, "frames": fr}));
            self.home()?;
        }
        let res = json!({"amp": a, "hold": hold, "runs": runs});
        if amp.is_none() {
            self.quick_estimates(&res);
        }
        Ok(res)
    }

    fn sine_amp(&self, f: f64) -> (f64, f64) {
        let acc_lim = self
            .a
            .sine_acc
            .filter(|x| *x > 0.0)
            .unwrap_or_else(|| self.measured_acc.map(|m| 0.5 * m).unwrap_or(1000.0));
        let w = 2.0 * PI * f;
        (self.amp.min(acc_lim / w).min(0.35 * self.a.max_excursion * w), acc_lim)
    }

    pub fn run_sine(&mut self) -> Result<Value> {
        let mut runs = Vec::new();
        let mut acc_lim = Value::Null;
        for f in self.a.freqs.clone() {
            let w = 2.0 * PI * f;
            let (amp, lim) = self.sine_amp(f);
            acc_lim = json!(lim);
            let settle = 0.25f64.max(1.0 / f);
            let dur = settle + (4.0 / f).max(1.5);
            say!("  sine {f:5.2} Hz  A={amp:6.2} deg/s  ({dur:.1} s)");
            let (cmd, fr) = self.stream(on_joint(self.j, move |t| amp * (w * t).sin()), dur)?;
            runs.push(json!({"f": f, "A": amp, "settle": settle, "cmd": cmd, "frames": fr}));
            self.home()?;
        }
        Ok(json!({"acc_lim": acc_lim, "runs": runs}))
    }

    pub fn run_accsweep(&mut self) -> Result<Value> {
        let pcts = self.a.acc_sweep_pcts.clone();
        let pmin = pcts.iter().cloned().fold(f64::INFINITY, f64::min);
        let amp = 0.8 * self.vmax_tab[self.j] * pmin / 100.0; // same amplitude for every setting
        let reps = (self.a.repeats / 2).max(2);
        let mut sweeps = Vec::new();
        let res = (|| -> Result<()> {
            for p in pcts {
                say!("  SetJointVelLimit({p})");
                if !self.io.set_vel_limit(p)? {
                    continue;
                }
                let r = self.run_step(Some(amp), Some(reps))?;
                sweeps.push(merge(json!({"pct": p}), r.as_object().cloned().unwrap_or_default()));
            }
            Ok(())
        })();
        let restored = self.io.set_vel_limit(self.a.vel_limit_pct);
        res?;
        restored?;
        Ok(json!({"amp": amp, "sweeps": sweeps}))
    }

    pub fn run_multijoint(&mut self) -> Result<Value> {
        let js = self.a.multi_js();
        let amps = self.vlim_all.map(|v| v * self.a.multi_frac);
        let amax = js.iter().map(|&j| amps[j]).fold(0.0, f64::max);
        let hold = self.a.step_hold.min(0.7 * self.a.max_excursion / amax);
        let shape = piecewise(&[(0.15, 0.0), (hold, 1.0), (hold, 0.0), (hold, -1.0), (0.35, 0.0)]);
        let dur = 0.5 + 3.0 * hold;
        let mut scen: Vec<(String, Vec<usize>)> = js.iter().map(|&j| (format!("J{} alone", j + 1), vec![j])).collect();
        if js.len() > 1 {
            scen.push(("together".into(), js.clone()));
        }
        let mut out = Vec::new();
        for (name, act) in scen {
            let mut av = [0.0; 6];
            for &j in &act {
                av[j] = amps[j];
            }
            say!(
                "  {name}: {}",
                act.iter().map(|&j| format!("J{} {:.1} deg/s", j + 1, av[j])).collect::<Vec<_>>().join(", ")
            );
            let mut runs = Vec::new();
            for _ in 0..(self.a.repeats / 2).max(2) {
                let sh = shape.clone();
                let (cmd, fr) = self.stream(Box::new(move |t| av.map(|x| x * sh(t))), dur)?;
                runs.push(json!({"cmd": cmd, "frames": fr}));
                self.home()?;
            }
            out.push(json!({"name": name, "active": act, "A": arr(&av), "runs": runs}));
        }
        Ok(json!({"scenarios": out, "hold": hold}))
    }

    // -- external position loop ---------------------------------------------------------------

    /// _servo_loop: the tested joint, log of 8 columns (send_logs gets [ts, t1, t2, v, NaN x 6]).
    fn servo_loop(&mut self, r: Profile, ff: Option<Profile>, kp: f64, dur: f64) -> Result<LoopRes> {
        let mut vmax = [0.0; 6];
        vmax[self.j] = 0.95 * self.vlim;
        let mut q0 = self.start_all;
        q0[self.j] = self.start_pos;
        let cfg = TrackCfg {
            kind: LoopKind::Single(self.j),
            kp,
            vmax,
            q0,
            fb_actual: self.a.servo_fb == "actual",
            period: self.a.period,
            cycle: self.cycle,
            dur,
            abort_err: self.a.servo_abort,
        };
        let l = self.io.exec(TrackLoop::new(cfg, r, ff, self.sensor()), dur + 5.0)?;
        if l.aborted {
            say!("    |error| = {:.2} deg > --servo-abort {}: run stopped", l.max_err, self.a.servo_abort);
        }
        let log = self.out.npy("log", &l.log, 8)?;
        let n = l.log.len() / 8;
        if n > 0 {
            let mut sl = Vec::with_capacity(n * 10);
            for row in l.log.chunks(8) {
                sl.extend_from_slice(&row[..4]);
                sl.extend_from_slice(&[NAN; 6]);
            }
            let c = self.out.npy("cmd", &sl, 10)?;
            self.out.send_logs.push(c);
        }
        let mut m = Map::new();
        m.insert("log".into(), log);
        m.insert("frames".into(), frames_ref(l.i0, l.i1));
        m.insert("q0".into(), json!(self.start_pos));
        m.insert("aborted".into(), json!(l.aborted));
        Ok(LoopRes { map: m, aborted: l.aborted })
    }

    /// _track_loop: the `active` joints at once, log of 23 columns.
    fn track_loop(&mut self, r: Profile, ff: Option<Profile>, kp: f64, dur: f64, active: [bool; 6], keep_frames: bool) -> Result<LoopRes> {
        let cfg = TrackCfg {
            kind: LoopKind::Multi(active),
            kp,
            vmax: self.vlim_all.map(|v| 0.95 * v),
            q0: self.start_all,
            fb_actual: self.a.servo_fb == "actual",
            period: self.a.period,
            cycle: self.cycle,
            dur,
            abort_err: self.a.servo_abort,
        };
        let l = self.io.exec(TrackLoop::new(cfg, r, ff, self.sensor()), dur + 5.0)?;
        if l.aborted {
            say!("    |error| = {:.2} deg > --servo-abort {}: run stopped", l.max_err, self.a.servo_abort);
        }
        let log = self.out.npy("log", &l.log, 23)?;
        let n = l.log.len() / 23;
        if n > 0 {
            let mut sl = Vec::with_capacity(n * 10);
            for row in l.log.chunks(23) {
                sl.extend_from_slice(&row[..3]);
                sl.push(row[3 + self.j]);
                sl.extend_from_slice(&row[3..9]);
            }
            let c = self.out.npy("cmd", &sl, 10)?;
            self.out.send_logs.push(c);
        }
        let mut m = Map::new();
        m.insert("log".into(), log);
        if keep_frames {
            m.insert("frames".into(), frames_ref(l.i0, l.i1));
        } else {
            m.insert("i0".into(), json!(l.i0));
            m.insert("i1".into(), json!(l.i1));
        }
        m.insert("q0".into(), arr(&self.start_all));
        m.insert("aborted".into(), json!(l.aborted));
        m.insert("t0".into(), num(l.t0));
        Ok(LoopRes { map: m, aborted: l.aborted })
    }

    fn adv(&self) -> f64 {
        self.a.servo_adv_ms.map(|x| x * 1e-3).unwrap_or_else(|| self.model_delay())
    }

    fn ff_variants(&self) -> Vec<bool> {
        if self.a.no_servo_ff {
            vec![false]
        } else {
            vec![false, true]
        }
    }

    pub fn run_servo(&mut self) -> Result<Value> {
        let adv = self.adv();
        let vmax = 0.95 * self.vlim;
        let (s, t_on, hold) = (self.a.servo_step, 0.2, self.a.servo_hold);
        let mut steps = Vec::new();
        let mut sines = Vec::new();
        for kp in self.a.servo_kps.clone() {
            say!("  servo step {s} deg, Kp = {kp} 1/s (feedback: {} position)", self.a.servo_fb);
            let r = self.servo_loop(on_joint(self.j, move |t| if t_on <= t && t < t_on + hold { s } else { 0.0 }), None, kp, t_on + 2.0 * hold)?;
            steps.push(merge(json!({"kp": kp, "step": s, "t_on": t_on, "hold": hold}), r.map));
            self.home()?;
            if r.aborted {
                say!("    higher Kp skipped");
                break;
            }
        }
        for ff in self.ff_variants() {
            for f in self.a.servo_freqs.clone() {
                let w = 2.0 * PI * f;
                let amp = self.a.servo_amp.min(0.5 * vmax / w);
                let settle = 0.5f64.max(1.0 / f);
                let dur = settle + (3.0 / f).max(1.5);
                say!(
                    "  servo sine {f} Hz, A = {amp:.2} deg, Kp = {}{}",
                    self.a.servo_kp,
                    if ff { format!(", feedforward {:.1} ms ahead", adv * 1e3) } else { String::new() }
                );
                let rf = on_joint(self.j, move |t| amp * (w * t).sin());
                let fff = ff.then(|| on_joint(self.j, move |t| amp * w * (w * (t + adv)).cos()));
                let r = self.servo_loop(rf, fff, self.a.servo_kp, dur)?;
                sines.push(merge(json!({"kp": self.a.servo_kp, "ff": ff, "f": f, "A": amp, "settle": settle}), r.map));
                self.home()?;
            }
        }
        Ok(json!({"steps": steps, "sines": sines, "P": self.a.period, "vmax": vmax, "adv": adv, "fb": self.a.servo_fb}))
    }

    pub fn run_servomulti(&mut self) -> Result<Value> {
        let js = self.a.multi_js();
        let vmax = self.vlim_all.map(|v| 0.95 * v);
        let adv = self.adv();
        let mut runs = Vec::new();
        'outer: for ff in self.ff_variants() {
            for f in self.a.servomulti_freqs.clone() {
                let w = 2.0 * PI * f;
                let mut amp = [0.0; 6];
                for &j in &js {
                    amp[j] = self.a.servo_amp.min(0.5 * vmax[j] / w);
                }
                let settle = 0.5f64.max(1.0 / f);
                let dur = settle + (3.0 / f).max(1.5);
                say!(
                    "  servo on {}: sine {f} Hz, Kp = {}, A = {}{}",
                    js.iter().map(|j| format!("J{}", j + 1)).collect::<Vec<_>>().join(", "),
                    self.a.servo_kp,
                    js.iter().map(|&j| format!("{:.2}", amp[j])).collect::<Vec<_>>().join(", "),
                    if ff { format!(", feedforward {:.1} ms", adv * 1e3) } else { String::new() }
                );
                let rf: Profile = Box::new(move |t| amp.map(|a| a * (w * t).sin()));
                let fff: Option<Profile> = if ff { Some(Box::new(move |t| amp.map(|a| a * w * (w * (t + adv)).cos()))) } else { None };
                let r = self.track_loop(rf, fff, self.a.servo_kp, dur, amp.map(|a| a > 0.0), true)?;
                runs.push(merge(json!({"f": f, "ff": ff, "A": arr(&amp), "settle": settle}), r.map));
                self.home()?;
                if r.aborted {
                    break 'outer;
                }
            }
        }
        Ok(json!({"joints": js, "runs": runs, "P": self.a.period, "vmax": arr(&vmax), "adv": adv, "kp": self.a.servo_kp, "fb": self.a.servo_fb}))
    }

    // -- physical AI: replay --------------------------------------------------------------------

    pub fn run_replay(&mut self) -> Result<Value> {
        let js = self.a.multi_js();
        let rr = self.bridge.replay_ref(self.py_args, &self.out.dir)?;
        let peak = rr.q.iter().flat_map(|r| r.iter()).filter(|x| x.is_finite()).fold(0.0f64, |m, x| m.max(x.abs()));
        if peak > 0.8 * self.a.max_excursion {
            bail!(Stop::Safety(format!("replay reference reaches {peak:.1} deg from start (> 80 % of --max-excursion)")));
        }
        let t = rr.t.clone();
        let p_act = median(t.windows(2).map(|w| w[1] - w[0]).collect());
        let adv = self.adv();
        let pi = self.a.pos_interp_ms * 1e-3;
        let adv_pos = self.a.posi_adv_ms.map(|x| x * 1e-3).unwrap_or_else(|| self.model_delay() - self.a.period / 2.0 + pi / 2.0);
        let h = rr.h_smooth;
        let horizon = (2.0 * p_act).max(adv + 2.0 * self.a.period).max(adv_pos + 2.0 * pi) + h as f64 * p_act;
        let order = if self.a.ref_predict == "quad" { 2 } else { 1 };
        let r = Arc::new(ActionRef::new(t.clone(), rr.q_cmd.clone(), &rr.rel_cmd, horizon, order));
        let mut active = [false; 6];
        for &j in &js {
            active[j] = true;
        }
        let mut out = Map::new();
        out.insert("joints".into(), json!(js));
        out.insert("t".into(), self.out.npy("ref_t", &t, 0)?);
        out.insert("q".into(), self.out.npy_rows("ref_q", &rr.q)?);
        out.insert("release".into(), self.out.npy("ref_release", &rr.release, 0)?);
        out.insert("q_clean".into(), self.out.npy_rows("ref_q_clean", &rr.q_clean)?);
        out.insert("smooth".into(), json!(if h > 0 { 2 * h + 1 } else { 0 }));
        out.insert("source".into(), json!(rr.source));
        out.insert("P_action".into(), num(p_act));
        out.insert("chunk".into(), json!(self.a.ref_chunk));
        out.insert("noise_deg".into(), json!(self.a.ref_noise_deg));
        out.insert("jitter_ms".into(), json!(self.a.ref_jitter_ms));
        out.insert("chunk_offset_deg".into(), json!(self.a.ref_chunk_offset_deg));
        out.insert("lead_ms".into(), json!(self.a.ref_lead_ms));
        out.insert("predict".into(), json!(self.a.ref_predict));
        out.insert("posi_period".into(), json!(pi));
        out.insert("posi_adv".into(), num(adv_pos));
        out.insert("kp".into(), json!(self.a.servo_kp));
        out.insert("adv".into(), num(adv));
        out.insert("P".into(), json!(self.a.period));
        out.insert("vmax".into(), arr(&self.vlim_all.map(|v| 0.95 * v)));
        out.insert("fb".into(), json!(self.a.servo_fb));
        out.insert("settle".into(), json!(self.a.replay_settle));
        let dur = t.last().copied().unwrap_or(0.0) + 0.5;
        say!(
            "  {} reference: {} actions every {:.0} ms on {}, chunks of {}, noise {} deg, jitter {} ms, chunk offset {} deg, peak {:.2} deg{}",
            rr.source,
            t.len(),
            p_act * 1e3,
            js.iter().map(|j| format!("J{}", j + 1)).collect::<Vec<_>>().join(", "),
            self.a.ref_chunk,
            self.a.ref_noise_deg,
            self.a.ref_jitter_ms,
            self.a.ref_chunk_offset_deg,
            peak,
            if h > 0 { format!(", smoothed over {} actions", 2 * h + 1) } else { String::new() }
        );
        let mut runs = Vec::new();
        for mode in self.a.replay_modes.clone() {
            let res = match mode.as_str() {
                "vel" | "ff" => {
                    let kp = if mode == "vel" { self.a.servo_kp } else { 0.0 };
                    say!(
                        "  replay through MoveJointsVel: {}",
                        if kp > 0.0 { format!("Kp {kp} + feedforward {:.1} ms ahead", adv * 1e3) } else { format!("feedforward only, {:.1} ms ahead", adv * 1e3) }
                    );
                    let (r1, r2) = (r.clone(), r.clone());
                    let rf: Profile = Box::new(move |tt| r1.at(tt, tt));
                    let fff: Profile = Box::new(move |tt| r2.slope(tt + adv, tt));
                    self.track_loop(rf, Some(fff), kp, dur, active, true)?
                }
                "pos" => {
                    say!("  replay through MoveJoints (blending {} %, SetJointVel {} %)", self.a.blending, self.a.pos_joint_vel_pct);
                    let due: Vec<f64> = r.t.iter().zip(r.rel.iter()).map(|(a, b)| a.max(*b)).collect();
                    self.pos_replay(Sched::List { due, q: r.q.clone() }, dur)?
                }
                "posi" => {
                    say!("  replay through MoveJoints every {:.0} ms, interpolated on the host, {:.1} ms ahead", pi * 1e3, adv_pos * 1e3);
                    self.pos_replay(Sched::Interp { r: r.clone(), pi, adv: adv_pos, dur }, dur)?
                }
                other => {
                    say!("  unknown replay mode '{other}', skipped");
                    continue;
                }
            };
            let aborted = res.aborted;
            runs.push(merge(json!({"mode": mode}), res.map));
            self.home()?;
            if aborted {
                say!("    '{mode}' stopped early (error above --servo-abort); going on with the next interface");
            }
        }
        out.insert("runs".into(), json!(runs));
        Ok(Value::Object(out))
    }

    /// _pos_replay / _posi_replay: SetJointVel + SetBlending, then the MoveJoints stream.
    fn pos_replay(&mut self, sched: Sched, dur: f64) -> Result<LoopRes> {
        self.io.set_joint_vel(self.a.pos_joint_vel_pct)?;
        self.io.set_blending(self.a.blending)?;
        let drain = self.a.pos_backlog_timeout.max(dur);
        let p = PosStream::new(sched, self.start_all, drain, self.cycle);
        let p = self.io.exec(p, dur + drain + 30.0)?;
        if !p.drained {
            say!("    queue not drained -> ClearMotion");
        }
        let mut m = Map::new();
        m.insert("log".into(), self.out.npy("log", &p.log, PosStream::COLS)?);
        m.insert("frames".into(), frames_ref(p.i0, p.i1));
        m.insert("q0".into(), arr(&self.start_all));
        m.insert("aborted".into(), json!(false));
        m.insert("t0".into(), num(p.t0));
        m.insert("t_last".into(), num(p.t_last));
        Ok(LoopRes { map: m, aborted: false })
    }

    // -- worst cases: soak ------------------------------------------------------------------------

    pub fn run_soak(&mut self) -> Result<Value> {
        let js = self.a.multi_js();
        let mut active = [false; 6];
        for &j in &js {
            active[j] = true;
        }
        let vmax = self.vlim_all.map(|v| 0.95 * v);
        let adv = self.adv();
        let freqs = [0.2, 0.45, 0.9];
        let fmean = freqs.iter().sum::<f64>() / 3.0;
        let ph = self.bridge.soak_phases(self.py_args, &self.out.dir)?;
        if ph.len() < 6 {
            bail!("soak-ref returned {} rows of phases", ph.len());
        }
        let mut ph6 = [[0.0; 3]; 6];
        ph6.copy_from_slice(&ph[..6]);
        let mut amp = [0.0; 6];
        for &j in &js {
            amp[j] = self.a.soak_amp.min(0.5 * vmax[j] / (2.0 * PI * fmean));
        }
        let dur = self.a.soak_dur;
        let reference = move |t: f64| -> [f64; 6] {
            let w = ((t.min(dur - t)) / 0.5).clamp(0.0, 1.0); // 0.5 s smooth start and stop
            let w = 0.5 - 0.5 * (PI * w).cos();
            let mut q = [0.0; 6];
            for j in 0..6 {
                let s: f64 = (0..3).map(|k| (2.0 * PI * freqs[k] * t + ph6[j][k]).sin()).sum();
                q[j] = w * (amp[j] / 3.0) * s;
            }
            q
        };
        let mut phases = Vec::new();
        for load in self.a.soak_loads.clone() {
            let mut gen = LoadGen::start(&load, self.a.stress_procs, dur + 60.0)?;
            say!(
                "  soak {dur} s, load '{load}'{}: {}, A {} deg, Kp {}, feedforward {:.1} ms",
                if load == "cpu" { format!(" ({} busy processes)", gen.procs) } else { String::new() },
                js.iter().map(|j| format!("J{}", j + 1)).collect::<Vec<_>>().join(", "),
                js.iter().map(|&j| format!("{:.2}", amp[j])).collect::<Vec<_>>().join(", "),
                self.a.servo_kp,
                adv * 1e3
            );
            let rf: Profile = Box::new(reference);
            let fff: Profile = Box::new(move |t| {
                let h = 1e-3;
                let (a, b) = (reference(t + adv + h), reference(t + adv - h));
                let mut v = [0.0; 6];
                for j in 0..6 {
                    v[j] = (a[j] - b[j]) / (2.0 * h);
                }
                v
            });
            let r = self.track_loop(rf, Some(fff), self.a.servo_kp, dur, active, false);
            gen.stop();
            let r = r?;
            let procs = if load == "cpu" { gen.procs } else { 0 };
            phases.push(merge(json!({"load": load, "procs": procs}), r.map));
            self.home()?;
            if r.aborted {
                say!("    soak stopped (error above --servo-abort)");
                break;
            }
        }
        Ok(json!({
            "joints": js, "A": arr(&amp), "freqs": arr(&freqs), "kp": self.a.servo_kp, "adv": adv,
            "P": self.a.period, "vmax": arr(&vmax), "fb": self.a.servo_fb, "dur": dur,
            "vel_timeout": self.a.vel_timeout, "phases": phases
        }))
    }
}

/// Every test in order, each from the start pose (home), each result into out.raw[test] once
/// it finished (an interrupted test leaves no result, as in Python).
pub fn run_all(
    io: &mut Robot,
    a: &RunArgs,
    out: &mut RunOut,
    bridge: &Bridge,
    py_args: &Map<String, Value>,
    start_joints: [f64; 6],
) -> Result<()> {
    let tests = expand_tests(&a.tests)?;
    let mut t = Tester::new(io, a, out, bridge, py_args, start_joints);
    say!("{}", t.describe(&tests));
    for name in tests {
        say!("\n== {name} ==");
        t.home()?;
        let v = match name.as_str() {
            "step" => t.run_step(None, None)?,
            "sine" => t.run_sine()?,
            "accsweep" => t.run_accsweep()?,
            "multijoint" => t.run_multijoint()?,
            "servo" => t.run_servo()?,
            "servomulti" => t.run_servomulti()?,
            "replay" => t.run_replay()?,
            "soak" => t.run_soak()?,
            other => bail!("unknown test {other}"),
        };
        t.out.raw.insert(name, v);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_groups() {
        assert_eq!(expand_tests("vel,servo").unwrap(), vec!["step", "sine", "accsweep", "multijoint", "servo"]);
        assert_eq!(expand_tests("step, step,soak").unwrap(), vec!["step", "soak"]);
        assert!(expand_tests("impulse").is_err());
        assert!(!expand_tests("all").unwrap().contains(&"soak".to_string()));
    }

    #[test]
    fn median_of_diffs() {
        assert_eq!(median(vec![0.02, 0.02, 0.03]), 0.02);
        assert_eq!(median(vec![1.0, 2.0]), 1.5);
    }
}
