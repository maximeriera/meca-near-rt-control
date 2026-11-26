//! Test controllers (feature "suite"), one per streaming primitive of meca_latency_test.py:
//!   VelStream  <- Tester.stream()                       send_logs rows: ts, t1, t2, v_joint, v1..v6
//!   TrackLoop  <- Tester._servo_loop / _track_loop      log rows: 8 columns (one joint) or 23 (all)
//!   PosStream  <- Tester._pos_replay / _posi_replay     log rows: ts, t1, t2, q1..q6 (offsets)
//! ts is the deadline of the cycle that carries the command (the scheduled time for PosStream),
//! t1 / t2 are taken around the exchange of that cycle. t1 / t2 are NaN until sent() fills them.

use std::sync::Arc;

use crate::actionref::{ActionRef, SensorEmu};
use crate::bus::{Controller, Cycle, Feedback, Flow};
use crate::controllers::{next_move_id, ClearMotion, Still};
use crate::into_any;
use crate::pdo::{cmd, col, Outputs, HOST, K_APOS, K_TPOS, ROBOT};

const NAN: f64 = f64::NAN;

pub type Profile = Box<dyn FnMut(f64) -> [f64; 6] + Send>;

fn cycles_per(period: f64, cycle: f64) -> u64 {
    ((period / cycle).round() as u64).max(1)
}

// ------------------------------------------------------------------------------------------- //
// VelStream                                                                                    //
// ------------------------------------------------------------------------------------------- //

/// MoveJointsVel(profile(k P)) every period P for round(duration / P) commands, then zero for
/// 0.15 s of trailing feedback.
pub struct VelStream {
    profile: Profile,
    j: usize,
    n: usize,
    m: u64,
    p: f64,
    trail: u64,
    cyc: u64,
    k: usize,
    trailing: Option<u64>,
    pending: Option<usize>,
    started: bool,
    pub log: Vec<f64>,
    pub i0: u64,
    pub i1: u64,
}

impl VelStream {
    pub const COLS: usize = 10;

    pub fn new(profile: Profile, j: usize, duration: f64, period: f64, cycle: f64) -> VelStream {
        let n = (duration / period).round() as usize;
        VelStream {
            profile,
            j,
            n,
            m: cycles_per(period, cycle),
            p: period,
            trail: (0.15 / cycle).round() as u64,
            cyc: 0,
            k: 0,
            trailing: None,
            pending: None,
            started: false,
            log: Vec::with_capacity(n * Self::COLS),
            i0: 0,
            i1: 0,
        }
    }
}

impl Controller for VelStream {
    fn name(&self) -> &'static str {
        "vel_stream"
    }

    fn guarded(&self) -> bool {
        true
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        if !self.started {
            self.started = true;
            self.i0 = fb.next_frame;
        }
        if cx.abort {
            out.joint_vel(&[0.0; 6]);
            self.i1 = fb.next_frame;
            return Flow::Done;
        }
        if let Some(left) = self.trailing {
            out.joint_vel(&[0.0; 6]);
            if left == 0 {
                self.i1 = fb.next_frame;
                return Flow::Done;
            }
            self.trailing = Some(left - 1);
            return Flow::Continue;
        }
        if self.cyc % self.m == 0 && self.k < self.n {
            let v = (self.profile)(self.k as f64 * self.p);
            out.joint_vel(&v);
            self.pending = Some(self.log.len() / Self::COLS);
            self.log.extend_from_slice(&[cx.ts, NAN, NAN, v[self.j]]);
            self.log.extend_from_slice(&v);
            self.k += 1;
        }
        self.cyc += 1;
        if self.k >= self.n && self.cyc % self.m == 0 {
            // the last command was held for one period: zero from the next cycle on
            self.trailing = Some(self.trail);
        }
        Flow::Continue
    }

    fn sent(&mut self, _cx: &Cycle, t1: f64, t2: f64) {
        if let Some(r) = self.pending.take() {
            self.log[r * Self::COLS + 1] = t1;
            self.log[r * Self::COLS + 2] = t2;
        }
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// TrackLoop                                                                                    //
// ------------------------------------------------------------------------------------------- //

#[derive(Clone, Copy, Debug)]
pub enum LoopKind {
    /// _servo_loop: the tested joint only, log of 8 columns.
    Single(usize),
    /// _track_loop: several joints at once, log of 23 columns.
    Multi([bool; 6]),
}

/// Host position loop, one update per period P while k P <= dur:
/// v = kp (ref(t) - y) + ff(t), clamped to vmax, MoveJointsVel; y = feedback position - q0 through
/// the sensor. Stops early when |error| > abort_err. Then zero velocity for 0.2 s.
pub struct TrackLoop {
    kind: LoopKind,
    reference: Profile,
    ff: Option<Profile>,
    kp: f64,
    vmax: [f64; 6],
    q0: [f64; 6],
    fb_actual: bool,
    sensor: SensorEmu,
    m: u64,
    p: f64,
    dur: f64,
    abort_err: f64,
    trail: u64,
    cyc: u64,
    k: u64,
    trailing: Option<u64>,
    pending: Option<usize>,
    started: bool,
    pub log: Vec<f64>,
    pub cols: usize,
    pub aborted: bool,
    pub max_err: f64,
    pub t0: f64,
    pub i0: u64,
    pub i1: u64,
}

pub struct TrackCfg {
    pub kind: LoopKind,
    pub kp: f64,
    pub vmax: [f64; 6],
    pub q0: [f64; 6],
    pub fb_actual: bool,
    pub period: f64,
    pub cycle: f64,
    pub dur: f64,
    pub abort_err: f64,
}

impl TrackLoop {
    pub fn new(c: TrackCfg, reference: Profile, ff: Option<Profile>, sensor: SensorEmu) -> TrackLoop {
        let cols = match c.kind {
            LoopKind::Single(_) => 8,
            LoopKind::Multi(_) => 23,
        };
        let n = (c.dur / c.period).floor() as usize + 2;
        TrackLoop {
            kind: c.kind,
            reference,
            ff,
            kp: c.kp,
            vmax: c.vmax,
            q0: c.q0,
            fb_actual: c.fb_actual,
            sensor,
            m: cycles_per(c.period, c.cycle),
            p: c.period,
            dur: c.dur,
            abort_err: c.abort_err,
            trail: (0.2 / c.cycle).round() as u64,
            cyc: 0,
            k: 0,
            trailing: None,
            pending: None,
            started: false,
            log: Vec::with_capacity(n * cols),
            cols,
            aborted: false,
            max_err: 0.0,
            t0: 0.0,
            i0: 0,
            i1: 0,
        }
    }
}

impl Controller for TrackLoop {
    fn name(&self) -> &'static str {
        "track_loop"
    }

    fn guarded(&self) -> bool {
        true
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        if !self.started {
            self.started = true;
            self.t0 = cx.ts;
            self.i0 = fb.next_frame;
        }
        if cx.abort {
            out.joint_vel(&[0.0; 6]);
            self.i1 = fb.next_frame;
            return Flow::Done;
        }
        if let Some(left) = self.trailing {
            out.joint_vel(&[0.0; 6]);
            if left == 0 {
                self.i1 = fb.next_frame;
                return Flow::Done;
            }
            self.trailing = Some(left - 1);
            return Flow::Continue;
        }
        if self.cyc % self.m == 0 {
            let t = self.k as f64 * self.p;
            if t > self.dur {
                out.joint_vel(&[0.0; 6]);
                self.trailing = Some(self.trail.saturating_sub(1));
                self.cyc += 1;
                return Flow::Continue;
            }
            let row = &fb.row;
            let kind = if self.fb_actual { K_APOS } else { K_TPOS };
            let mut y = [NAN; 6];
            for j in 0..6 {
                let a = row[col(kind, j)];
                y[j] = (if a.is_finite() { a } else { row[col(K_TPOS, j)] }) - self.q0[j];
            }
            let (y, (frt, fht)) = self.sensor.observe(t, y, (row[ROBOT], row[HOST]));
            let r = (self.reference)(t);
            let f = match self.ff.as_mut() {
                Some(g) => g(t),
                None => [0.0; 6],
            };
            let mut v = [0.0; 6];
            let mut err = 0f64;
            match self.kind {
                LoopKind::Single(j) => {
                    v[j] = (self.kp * (r[j] - y[j]) + f[j]).clamp(-self.vmax[j], self.vmax[j]);
                    let e = (r[j] - y[j]).abs();
                    if e.is_finite() {
                        err = e;
                    }
                }
                LoopKind::Multi(active) => {
                    for j in 0..6 {
                        if active[j] {
                            v[j] = (self.kp * (r[j] - y[j]) + f[j]).clamp(-self.vmax[j], self.vmax[j]);
                            let e = (r[j] - y[j]).abs();
                            if e.is_finite() {
                                err = err.max(e); // nanmax
                            }
                        }
                    }
                }
            }
            for x in v.iter_mut() {
                if !x.is_finite() {
                    *x = 0.0; // no feedback yet: do not move
                }
            }
            out.joint_vel(&v);
            self.pending = Some(self.log.len() / self.cols);
            match self.kind {
                LoopKind::Single(j) => {
                    self.log.extend_from_slice(&[cx.ts, NAN, NAN, v[j], r[j], y[j], frt, fht]);
                }
                LoopKind::Multi(_) => {
                    self.log.extend_from_slice(&[cx.ts, NAN, NAN]);
                    self.log.extend_from_slice(&v);
                    self.log.extend_from_slice(&r);
                    self.log.extend_from_slice(&y);
                    self.log.extend_from_slice(&[frt, fht]);
                }
            }
            self.max_err = self.max_err.max(err);
            self.k += 1;
            if err > self.abort_err {
                self.aborted = true; // this command goes out, then zero (break + finally)
                self.trailing = Some(self.trail);
            }
        }
        self.cyc += 1;
        Flow::Continue
    }

    fn sent(&mut self, _cx: &Cycle, t1: f64, t2: f64) {
        if let Some(r) = self.pending.take() {
            self.log[r * self.cols + 1] = t1;
            self.log[r * self.cols + 2] = t2;
        }
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// PosStream                                                                                    //
// ------------------------------------------------------------------------------------------- //

pub enum Sched {
    /// pos: one MoveJoints per action, at max(t_k, release_k) (offsets from t0).
    List { due: Vec<f64>, q: Vec<[f64; 6]> },
    /// posi: a MoveJoints every pi to ActionRef.at(k pi + adv, k pi), while k pi <= dur.
    Interp { r: Arc<ActionRef>, pi: f64, adv: f64, dur: f64 },
}

enum PosPhase {
    Send,
    Drain(f64),
    Clear(ClearMotion),
    Trail(u64),
}

/// A stream of MoveJoints with one command in flight (the next one goes when the robot echoes
/// the MoveID of the previous one; if it is late, its real send time t1 is logged), then wait
/// until the target reaches the last position (drain_timeout, then ClearMotion), 0.1 s trailing.
pub struct PosStream {
    sched: Sched,
    q0: [f64; 6],
    drain_timeout: f64,
    phase: PosPhase,
    k: usize,
    waiting: Option<u16>,
    last_goal: [f64; 6],
    still: Still,
    pending: Option<usize>,
    started: bool,
    trail: u64,
    pub log: Vec<f64>,
    pub t0: f64,
    pub t_last: f64,
    pub i0: u64,
    pub i1: u64,
    pub drained: bool,
    pub aborted: bool,
}

impl PosStream {
    pub const COLS: usize = 9;

    pub fn new(sched: Sched, q0: [f64; 6], drain_timeout: f64, cycle: f64) -> PosStream {
        let n = match &sched {
            Sched::List { due, .. } => due.len(),
            Sched::Interp { pi, dur, .. } => (dur / pi).floor() as usize + 2,
        };
        PosStream {
            sched,
            q0,
            drain_timeout,
            phase: PosPhase::Send,
            k: 0,
            waiting: None,
            last_goal: q0,
            still: Still::default(),
            pending: None,
            started: false,
            trail: (0.1 / cycle).round() as u64,
            log: Vec::with_capacity(n * Self::COLS),
            t0: 0.0,
            t_last: 0.0,
            i0: 0,
            i1: 0,
            drained: false,
            aborted: false,
        }
    }

    /// Scheduled time (offset from t0) and position offsets of command k, or None after the last.
    fn item(&self, k: usize, due_only: bool) -> Option<(f64, [f64; 6])> {
        match &self.sched {
            Sched::List { due, q } => due.get(k).map(|&d| (d, q[k])),
            Sched::Interp { r, pi, adv, dur } => {
                let tk = k as f64 * pi;
                if tk > *dur {
                    return None;
                }
                Some((tk, if due_only { [0.0; 6] } else { r.at(tk + adv, tk) }))
            }
        }
    }
}

impl Controller for PosStream {
    fn name(&self) -> &'static str {
        "pos_stream"
    }

    fn guarded(&self) -> bool {
        true
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        if !self.started {
            self.started = true;
            self.t0 = cx.ts;
            self.i0 = fb.next_frame;
        }
        self.still.update(fb);
        if cx.abort && matches!(self.phase, PosPhase::Send | PosPhase::Drain(_)) {
            self.aborted = true;
            self.phase = PosPhase::Clear(ClearMotion::new());
        }
        if let PosPhase::Send = self.phase {
            if let Some(mid) = self.waiting {
                if fb.ok && fb.inp.move_id == mid {
                    self.waiting = None;
                }
            }
            if self.waiting.is_none() {
                match self.item(self.k, true) {
                    Some((d, _)) if cx.ts >= self.t0 + d - 1e-9 => {
                        let (_, q) = self.item(self.k, false).unwrap_or((d, [0.0; 6]));
                        let mut goal = [0.0; 6];
                        for j in 0..6 {
                            goal[j] = self.q0[j] + q[j];
                        }
                        let mid = next_move_id();
                        out.queued(cmd::MOVE_JOINTS, &goal, mid);
                        self.waiting = Some(mid);
                        self.last_goal = goal;
                        self.pending = Some(self.log.len() / Self::COLS);
                        self.log.extend_from_slice(&[self.t0 + d, NAN, NAN]);
                        self.log.extend_from_slice(&q);
                        self.k += 1;
                    }
                    Some(_) => {}
                    None => {
                        self.still.n = 0;
                        self.phase = PosPhase::Drain(cx.ts);
                    }
                }
            }
            return Flow::Continue;
        }
        match &mut self.phase {
            PosPhase::Send => {}
            PosPhase::Drain(t) => {
                let there = (0..6).all(|j| (fb.inp.joint_set[j] - self.last_goal[j]).abs() < 0.01);
                if there && self.still.n >= 5 && fb.inp.eob {
                    self.drained = true;
                    self.phase = PosPhase::Trail(self.trail);
                } else if cx.ts - *t > self.drain_timeout {
                    self.phase = PosPhase::Clear(ClearMotion::new()); // queue not drained
                }
            }
            PosPhase::Clear(c) => {
                if let Flow::Done = c.step(cx, fb, out) {
                    self.phase = PosPhase::Trail(self.trail);
                }
            }
            PosPhase::Trail(left) => {
                if *left == 0 {
                    self.i1 = fb.next_frame;
                    return Flow::Done;
                }
                *left -= 1;
            }
        }
        Flow::Continue
    }

    fn sent(&mut self, _cx: &Cycle, t1: f64, t2: f64) {
        if let Some(r) = self.pending.take() {
            self.log[r * Self::COLS + 1] = t1;
            self.log[r * Self::COLS + 2] = t2;
            self.t_last = t2;
        }
    }

    into_any!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bus::{run_sync, Core};
    use crate::controllers::{Bits, MoveAndWait};
    use crate::pdo::{ByteOrder, Codec};
    use crate::sim::{SimConfig, SimRobot};

    fn ready() -> (Core, SimRobot) {
        let mut c = Core::new(Codec::new(ByteOrder::Le), false, 5, 1e-3, [32, 40, 42, 41]);
        let mut sim = SimRobot::new(SimConfig::default());
        assert!(run_sync(&mut c, &mut sim, &mut Bits::activate(false), 2.0));
        assert!(run_sync(&mut c, &mut sim, &mut Bits::home(), 2.0));
        (c, sim)
    }

    #[test]
    fn velocity_stream_moves_the_joint_and_logs_every_command() {
        let (mut c, mut sim) = ready();
        let f = crate::actionref::piecewise(&[(0.05, 0.0), (0.2, 10.0), (0.05, 0.0)]);
        let mut s = VelStream::new(Box::new(move |t| [0.0, 0.0, 0.0, 0.0, 0.0, f(t)]), 5, 0.3, 0.001, 0.001);
        assert!(run_sync(&mut c, &mut sim, &mut s, 2.0));
        assert_eq!(s.log.len(), 300 * VelStream::COLS);
        assert!(s.log.chunks(10).all(|r| r[1].is_finite() && r[2] >= r[1]));
        let moved = c.fb.inp.joint_set[5];
        assert!((moved - 2.0).abs() < 0.1, "10 deg/s for 0.2 s: {moved}");
        assert!(s.i1 > s.i0 + 250);
    }

    #[test]
    fn track_loop_follows_a_step() {
        let (mut c, mut sim) = ready();
        let cfg = TrackCfg {
            kind: LoopKind::Single(5),
            kp: 10.0,
            vmax: [0.0, 0.0, 0.0, 0.0, 0.0, 100.0],
            q0: c.fb.inp.joint_set,
            fb_actual: true,
            period: 0.001,
            cycle: 0.001,
            dur: 1.5,
            abort_err: 10.0,
        };
        let mut l = TrackLoop::new(cfg, Box::new(|t| [0.0, 0.0, 0.0, 0.0, 0.0, if t >= 0.1 { 1.0 } else { 0.0 }]), None,
                                   SensorEmu::new(0.0, 0.0, 0.0, 0));
        assert!(run_sync(&mut c, &mut sim, &mut l, 3.0));
        assert!(!l.aborted);
        let last = &l.log[l.log.len() - 8..];
        assert!((last[5] - 1.0).abs() < 0.05, "y at the end: {}", last[5]);
    }

    #[test]
    fn position_stream_one_in_flight_then_drain() {
        let (mut c, mut sim) = ready();
        let q0 = c.fb.inp.joint_set;
        assert!(run_sync(&mut c, &mut sim, &mut MoveAndWait::new(q0, 5.0), 6.0));
        let due: Vec<f64> = (0..20).map(|k| k as f64 * 0.02).collect();
        let q: Vec<[f64; 6]> = (0..20).map(|k| [0.0, 0.0, 0.0, 0.0, 0.0, k as f64 * 0.1]).collect();
        let mut p = PosStream::new(Sched::List { due, q }, q0, 10.0, 0.001);
        assert!(run_sync(&mut c, &mut sim, &mut p, 15.0));
        assert!(p.drained && !p.aborted);
        assert_eq!(p.log.len(), 20 * PosStream::COLS);
        assert!((c.fb.inp.joint_set[5] - (q0[5] + 1.9)).abs() < 0.01);
    }
}
