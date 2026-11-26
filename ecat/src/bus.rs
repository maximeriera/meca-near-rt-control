//! The bus thread: one process data exchange per cycle, the active controller computing the
//! outputs of each cycle from the newest feedback, the safety guard, and the recording.
//!
//! Cycle k, in this order: wake at the deadline ts_k; the controller writes the outputs from the
//! frame received last (cycle k-1 or earlier), with its reference at t = k * P; encode; t1; tx_rx;
//! t2; decode; record the frame if it is a new robot sample (its host time is t2); guard;
//! heartbeat. A log row therefore holds the ts / t1 / t2 of the cycle that carries the command,
//! and the robot / host time of the frame it was computed from.
//!
//! Nothing allocates or locks in the cycle (rings to the writer and the orchestrator, try_lock
//! for the snapshot), except at a safety trip or an emergency exit.

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use rtrb::{Consumer, Producer};

use crate::controllers::ExitSeq;
use crate::fieldbus::{BusError, Fieldbus};
use crate::pdo::{col, frame_row, Codec, Inputs, Outputs, Row, IN_LEN, K_APOS, K_TPOS, NCOL, OUT_LEN};
use crate::rt;

pub struct Cycle {
    /// Cycle number since the bus thread started.
    pub k: u64,
    /// Deadline of this cycle [host s].
    pub ts: f64,
    /// Cycle period [s].
    pub dt: f64,
    /// The orchestrator asked the controller to stop (signal, time limit, shutdown).
    pub abort: bool,
}

#[derive(Clone, Copy)]
pub struct Feedback {
    /// Inputs of the last successful exchange.
    pub inp: Inputs,
    /// Last recorded frame (newest robot sample) and whether there is one yet.
    pub row: Row,
    pub have_row: bool,
    /// The last exchange brought a new robot sample.
    pub fresh: bool,
    /// The last exchange succeeded (and the robot processed the frame).
    pub ok: bool,
    /// Seconds since the last successful exchange.
    pub ok_age: f64,
    /// Index the next recorded frame will get (global, as in the frames array of the run).
    pub next_frame: u64,
    pub wkc: i32,
}

impl Default for Feedback {
    fn default() -> Self {
        Feedback {
            inp: Inputs::default(),
            row: [f64::NAN; NCOL],
            have_row: false,
            fresh: false,
            ok: false,
            ok_age: 0.0,
            next_frame: 0,
            wkc: -1,
        }
    }
}

pub enum Flow {
    Continue,
    Done,
}

/// Something that drives the outputs for a while: a velocity stream, a host loop, an activation,
/// the exit sequence... It runs in the bus thread, so it must not block, allocate or print.
pub trait Controller: Send + 'static {
    fn name(&self) -> &'static str;
    /// Write the outputs of this cycle. `fb` is the newest feedback (previous exchange).
    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow;
    /// The outputs of this cycle went out at t1 and the exchange returned at t2 (host s).
    fn sent(&mut self, _cx: &Cycle, _t1: f64, _t2: f64) {}
    /// Test motions are watched by the guard (max excursion, robot error); housekeeping is not.
    fn guarded(&self) -> bool {
        false
    }
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;
}

/// Boilerplate of Controller::into_any.
#[macro_export]
macro_rules! into_any {
    () => {
        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    };
}

// ------------------------------------------------------------------------------------------- //
// Guard                                                                                        //
// ------------------------------------------------------------------------------------------- //

#[derive(Clone, Copy, Debug)]
pub struct GuardCfg {
    /// Joint positions the excursion is measured from (the start pose of the test) [deg].
    pub start: [f64; 6],
    pub max_exc: f64,
}

#[derive(Default)]
pub struct Guard {
    pub cfg: Option<GuardCfg>,
    /// A guarded controller is running.
    pub active: bool,
    /// Latched message of the first trip (never cleared: the run goes to the exit sequence).
    pub tripped: Option<String>,
    fired: bool,
}

/// What the outputs become once the guard has tripped: velocity zero, queued motion cleared.
fn safe_outputs(out: &mut Outputs) {
    out.zero_velocity();
    if !crate::pdo::cmd::is_velocity(out.command_id) && out.move_id != 0 {
        out.clear_move = true;
    }
}

// ------------------------------------------------------------------------------------------- //
// Core: one cycle                                                                              //
// ------------------------------------------------------------------------------------------- //

pub struct Sinks {
    pub frames: Producer<Row>,
    pub cycles: Producer<[f64; 6]>,
}

#[derive(Clone, Debug, Default)]
pub struct BusStats {
    pub cycles: u64,
    pub overruns: u64,
    pub failed: u64,
    /// Exchanges with a working counter below the full one (outputs not processed).
    pub partial: u64,
    pub ring_drops: u64,
    pub max_wkc: u16,
    pub backwards: u64,
    pub rt_problems: Vec<String>,
}

pub struct Core {
    pub codec: Codec,
    /// The robot echoed the dynamic slot types we wrote: our byte order is right both ways.
    pub order_confirmed: bool,
    pub out: Outputs,
    pub fb: Feedback,
    pub joint: usize,
    pub cycle_s: f64,
    pub guard: Guard,
    pub stats: BusStats,
    last_robot_t: f64,
    frame_idx: u64,
    sinks: Option<Sinks>,
    obuf: [u8; OUT_LEN],
    ibuf: [u8; IN_LEN],
    last_ok: f64,
}

impl Core {
    /// `joint`: the tested joint (0-based), copied into columns 2-5 of every frame.
    pub fn new(codec: Codec, order_confirmed: bool, joint: usize, cycle_s: f64, dyn_slots: [u32; 4]) -> Core {
        let out = Outputs { dyn_types: dyn_slots, ..Default::default() };
        Core {
            codec,
            order_confirmed,
            out,
            fb: Feedback::default(),
            joint,
            cycle_s,
            guard: Guard::default(),
            stats: BusStats::default(),
            last_robot_t: f64::NEG_INFINITY,
            frame_idx: 0,
            sinks: None,
            obuf: [0; OUT_LEN],
            ibuf: [0; IN_LEN],
            last_ok: rt::now(),
        }
    }

    pub fn set_sinks(&mut self, s: Sinks) {
        self.sinks = Some(s);
    }

    pub fn frames_recorded(&self) -> u64 {
        self.frame_idx
    }

    /// One cycle: controller -> outputs -> exchange -> inputs -> record -> guard.
    pub fn cycle(&mut self, bus: &mut dyn Fieldbus, mut ctrl: Option<&mut dyn Controller>, cx: &Cycle) -> Flow {
        let flow = match ctrl.as_mut() {
            Some(c) => c.step(cx, &self.fb, &mut self.out),
            None => {
                self.out.zero_velocity();
                Flow::Continue
            }
        };
        if self.guard.tripped.is_some() {
            safe_outputs(&mut self.out);
        }
        self.out.encode(&self.codec, &mut self.obuf, !self.order_confirmed);
        let t1 = rt::now();
        let res = bus.exchange(&self.obuf, &mut self.ibuf);
        let t2 = rt::now();
        if let Some(c) = ctrl.as_mut() {
            c.sent(cx, t1, t2);
        }
        self.absorb(res, cx.ts, t1, t2);
        flow
    }

    fn absorb(&mut self, res: Result<u16, BusError>, ts: f64, t1: f64, t2: f64) {
        self.stats.cycles += 1;
        let mut robot_t = f64::NAN;
        let mut fresh = false;
        let wkc: i32 = match res {
            Ok(w) if w > 0 => {
                // A partial working counter (e.g. the robot fell back to SAFE-OP after a watchdog:
                // inputs still read, outputs ignored) is not a working exchange: our commands,
                // DisableEtherCAT included, do not reach the robot. It counts as failed, so a
                // lasting one ends as "bus lost" and the way back goes through restore.
                self.stats.max_wkc = self.stats.max_wkc.max(w);
                let full = w >= self.stats.max_wkc;
                let inp = Inputs::decode(&self.codec, &self.ibuf);
                self.fb.inp = inp;
                self.fb.ok = full;
                if full {
                    self.last_ok = t2;
                } else {
                    self.stats.partial += 1;
                }
                robot_t = inp.robot_time();
                if !self.order_confirmed
                    && self.out.dyn_types.iter().any(|&t| t != 0)
                    && inp.dyn_type == self.out.dyn_types
                {
                    self.order_confirmed = true;
                }
                if robot_t > self.last_robot_t {
                    fresh = true;
                    self.last_robot_t = robot_t;
                    let row = frame_row(&inp, t2, self.joint);
                    self.fb.row = row;
                    self.fb.have_row = true;
                    match self.sinks.as_mut() {
                        Some(s) => {
                            if s.frames.push(row).is_ok() {
                                self.frame_idx += 1;
                            } else {
                                self.stats.ring_drops += 1;
                            }
                        }
                        None => self.frame_idx += 1,
                    }
                    self.check_guard(&row, &inp);
                } else if robot_t < self.last_robot_t {
                    self.stats.backwards += 1;
                }
                w as i32
            }
            Ok(w) => {
                self.fb.ok = false;
                self.stats.failed += 1;
                w as i32
            }
            Err(_) => {
                self.fb.ok = false;
                self.stats.failed += 1;
                -1
            }
        };
        self.fb.fresh = fresh;
        self.fb.wkc = wkc;
        self.fb.ok_age = t2 - self.last_ok;
        self.fb.next_frame = self.frame_idx;
        if let Some(s) = self.sinks.as_mut() {
            let _ = s.cycles.push([ts, t1, t2, robot_t, if fresh { 1.0 } else { 0.0 }, wkc as f64]);
        }
    }

    fn check_guard(&mut self, row: &Row, inp: &Inputs) {
        if !self.guard.active || self.guard.tripped.is_some() {
            return;
        }
        let Some(g) = self.guard.cfg else { return };
        for j in 0..6 {
            let a = row[col(K_APOS, j)];
            let p = if a.is_finite() { a } else { row[col(K_TPOS, j)] };
            let d = p - g.start[j];
            if d.abs() > g.max_exc {
                self.guard.tripped =
                    Some(format!("joint {} is {:+.1} deg from start (limit +/-{})", j + 1, d, g.max_exc));
                self.guard.fired = true;
                return;
            }
        }
        if inp.error_code != 0 || inp.estop {
            self.guard.tripped = Some(format!("robot error {} (E-stop {})", inp.error_code, inp.estop as u8));
            self.guard.fired = true;
        }
    }
}

/// Run one controller on the calling thread (no real-time thread): bring-up, probe, the exit
/// sequence when the bus thread is gone, restore. Returns true if it finished within max_s.
pub fn run_sync(core: &mut Core, bus: &mut dyn Fieldbus, ctrl: &mut dyn Controller, max_s: f64) -> bool {
    let dt = core.cycle_s;
    let mut ts = rt::now() + dt;
    let t_end = ts + max_s;
    let mut k = 0u64;
    loop {
        rt::sleep_until(ts);
        let cx = Cycle { k, ts, dt, abort: false };
        if let Flow::Done = core.cycle(bus, Some(&mut *ctrl), &cx) {
            return true;
        }
        if rt::now() > t_end {
            return false;
        }
        ts += dt;
        if rt::now() - ts > 5.0 * dt {
            ts = rt::now();
        }
        k += 1;
    }
}

// ------------------------------------------------------------------------------------------- //
// Bus thread and its links                                                                     //
// ------------------------------------------------------------------------------------------- //

pub enum Cmd {
    Install(Box<dyn Controller>),
    Abort,
    Guard(Option<GuardCfg>),
    Shutdown,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Finished,
    Aborted,
    Safety(String),
    BusLost,
}

pub enum Event {
    Done(Box<dyn Controller>, Outcome),
}

#[derive(Clone, Copy)]
pub struct Snapshot {
    pub fb: Feedback,
    pub out: Outputs,
    pub order_confirmed: bool,
    pub tripped: bool,
}

pub struct Shared {
    pub heartbeat: AtomicU64,
    pub snap: Mutex<Snapshot>,
    pub frames: AtomicU64,
    /// The bus thread started the exit sequence itself (second stop request).
    pub emergency_exit: AtomicBool,
    /// No successful exchange for BusCfg::lost_after_s (outside the exit sequence).
    pub bus_lost: AtomicBool,
}

impl Shared {
    pub fn new(core: &Core) -> Shared {
        Shared {
            heartbeat: AtomicU64::new(0),
            snap: Mutex::new(Snapshot {
                fb: core.fb,
                out: core.out,
                order_confirmed: core.order_confirmed,
                tripped: false,
            }),
            frames: AtomicU64::new(0),
            emergency_exit: AtomicBool::new(false),
            bus_lost: AtomicBool::new(false),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BusCfg {
    pub cycle_s: f64,
    pub prio: i32,
    pub cpu: Option<usize>,
    pub lost_after_s: f64,
}

#[derive(Clone, Debug)]
pub struct BusReport {
    pub stats: BusStats,
    pub order_confirmed: bool,
    pub tripped: Option<String>,
    pub frames: u64,
}

pub fn bus_thread<F: Fieldbus>(
    mut core: Core,
    mut fbus: F,
    mut cmd_rx: Consumer<Cmd>,
    mut ev_tx: Producer<Event>,
    shared: &Shared,
    cfg: BusCfg,
) -> BusReport {
    core.stats.rt_problems = rt::make_realtime(cfg.prio, cfg.cpu);
    let dt = cfg.cycle_s;
    let mut ts = rt::now() + dt;
    let mut k = 0u64;
    let mut ctrl: Option<Box<dyn Controller>> = None;
    let mut abort = false;
    let mut shutdown = false;
    let mut emergency = false;
    loop {
        while let Ok(c) = cmd_rx.pop() {
            match c {
                Cmd::Install(c) => {
                    if ctrl.is_some() || emergency {
                        let _ = ev_tx.push(Event::Done(c, Outcome::Aborted));
                    } else {
                        core.guard.active = c.guarded();
                        ctrl = Some(c);
                        abort = false;
                    }
                }
                Cmd::Abort => abort = true,
                Cmd::Guard(g) => core.guard.cfg = g,
                Cmd::Shutdown => shutdown = true,
            }
        }
        if shutdown {
            if ctrl.is_none() {
                break;
            }
            abort = true; // the exit sequence ignores it and finishes
        }
        // Second stop request: start the exit sequence here, whatever the orchestrator is doing.
        if !emergency && rt::stop_requests() >= 2 {
            emergency = true;
            shared.emergency_exit.store(true, Ordering::SeqCst);
            if ctrl.as_ref().map_or(true, |c| c.name() != ExitSeq::NAME) {
                if let Some(c) = ctrl.take() {
                    let _ = ev_tx.push(Event::Done(c, Outcome::Aborted));
                }
                core.guard.active = false;
                ctrl = Some(Box::new(ExitSeq::new()));
                abort = false;
            }
        }
        rt::sleep_until(ts);
        let late = rt::now() - ts;
        if late > dt {
            core.stats.overruns += 1;
            if late > 5.0 * dt {
                ts = rt::now(); // resynchronise after a long stall instead of bursting
            }
        }
        let cx = Cycle { k, ts, dt, abort };
        let flow = core.cycle(&mut fbus, ctrl.as_deref_mut(), &cx);

        let mut outcome = None;
        if core.guard.fired {
            core.guard.fired = false;
            if ctrl.as_ref().is_some_and(|c| c.guarded()) {
                outcome = Some(Outcome::Safety(core.guard.tripped.clone().unwrap_or_default()));
            }
        }
        if outcome.is_none() && matches!(flow, Flow::Done) {
            outcome = Some(if abort { Outcome::Aborted } else { Outcome::Finished });
        }
        let in_exit = ctrl.as_ref().is_some_and(|c| c.name() == ExitSeq::NAME);
        if core.fb.ok_age > cfg.lost_after_s && !in_exit {
            shared.bus_lost.store(true, Ordering::SeqCst);
            if outcome.is_none() && ctrl.is_some() {
                outcome = Some(Outcome::BusLost);
            }
        }
        if let Some(o) = outcome {
            if let Some(c) = ctrl.take() {
                core.guard.active = false;
                let _ = ev_tx.push(Event::Done(c, o));
            }
        }
        shared.heartbeat.store(k + 1, Ordering::SeqCst);
        shared.frames.store(core.frames_recorded(), Ordering::SeqCst);
        if let Ok(mut s) = shared.snap.try_lock() {
            *s = Snapshot {
                fb: core.fb,
                out: core.out,
                order_confirmed: core.order_confirmed,
                tripped: core.guard.tripped.is_some(),
            };
        }
        ts += dt;
        k += 1;
    }
    BusReport {
        stats: core.stats.clone(),
        order_confirmed: core.order_confirmed,
        tripped: core.guard.tripped.clone(),
        frames: core.frames_recorded(),
    }
}

/// Everything the bus recorded: frames (fresh robot samples) and one row per cycle.
#[derive(Default)]
pub struct Store {
    pub frames: Vec<Row>,
    pub cycles: Vec<[f64; 6]>,
}

/// Drains the rings into the store (normal priority), so nothing is lost while the orchestrator
/// waits on Python or on the disk.
pub fn writer_thread(mut fr: Consumer<Row>, mut cy: Consumer<[f64; 6]>, store: &Mutex<Store>, stop: &AtomicBool) {
    let mut bf: Vec<Row> = Vec::with_capacity(8192);
    let mut bc: Vec<[f64; 6]> = Vec::with_capacity(8192);
    loop {
        let done = stop.load(Ordering::SeqCst);
        while let Ok(r) = fr.pop() {
            bf.push(r);
        }
        while let Ok(c) = cy.pop() {
            bc.push(c);
        }
        if !bf.is_empty() || !bc.is_empty() {
            if let Ok(mut s) = store.lock() {
                s.frames.append(&mut bf);
                s.cycles.append(&mut bc);
            }
        }
        if done {
            break;
        }
        rt::sleep_s(0.002);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controllers::{Bits, Config};
    use crate::pdo::{ByteOrder, Codec};
    use crate::sim::{SimConfig, SimRobot};

    fn core() -> Core {
        Core::new(Codec::new(ByteOrder::Le), false, 5, 1e-3, [32, 40, 42, 41])
    }

    #[test]
    fn activation_handshake_and_config_on_the_simulator() {
        let mut c = core();
        let mut sim = SimRobot::new(SimConfig::default());
        let mut act = Bits::activate(false);
        assert!(run_sync(&mut c, &mut sim, &mut act, 2.0));
        assert!(act.ok && c.fb.inp.activated);
        let mut home = Bits::home();
        assert!(run_sync(&mut c, &mut sim, &mut home, 2.0));
        assert!(home.ok && c.fb.inp.homed);
        assert!(c.order_confirmed, "slot types echoed");
        let mut cfg = Config::new(crate::pdo::cmd::SET_JOINT_VEL_LIMIT, 20.0, 2.0);
        assert!(run_sync(&mut c, &mut sim, &mut cfg, 3.0));
        assert!(cfg.ok, "SetJointVelLimit acknowledged");
        assert!(c.frames_recorded() > 100);
    }

    #[test]
    fn exit_sequence_deactivates_then_disables() {
        let mut c = core();
        let mut sim = SimRobot::new(SimConfig::default());
        let mut act = Bits::activate(false);
        assert!(run_sync(&mut c, &mut sim, &mut act, 2.0));
        let mut ex = ExitSeq::new();
        assert!(run_sync(&mut c, &mut sim, &mut ex, 20.0));
        assert!(ex.was_activated && ex.deactivated);
        assert!(ex.link_lost_after.is_some(), "the robot left EtherCAT");
        assert!(sim.has_left_ethercat());
    }

    #[test]
    fn exit_sequence_on_a_dead_bus_does_not_hang() {
        let mut c = core();
        let mut sim = SimRobot::new(SimConfig::default());
        sim.mute = true;
        let mut ex = ExitSeq::new();
        assert!(run_sync(&mut c, &mut sim, &mut ex, 30.0));
        assert!(!sim.has_left_ethercat() && ex.link_lost_after.is_some());
    }
}
