//! The orchestrator side: runs controllers on the bus thread one at a time and waits for them
//! (with deadlines, stop requests and a heartbeat watch), the robot preparation (activation,
//! homing, settings) and the exit sequence. Also `drive`, which spawns the bus and writer threads
//! around a body (the tests) and always ends with the exit sequence.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use anyhow::{anyhow, bail, Result};
use rtrb::{Consumer, Producer, RingBuffer};

use crate::bus::{bus_thread, writer_thread, BusCfg, BusReport, Cmd, Controller, Core, Event, GuardCfg, Outcome, Shared, Sinks, Snapshot, Store};
use crate::controllers::{Bits, ClearMotion, Config, ExitSeq, MoveAndWait, WaitStill};
use crate::fieldbus::Fieldbus;
use crate::pdo::{alljoints, cmd, Row, K_TPOS};
use crate::rt;
use crate::say;

/// Why a test sequence stopped early (anything else is an ordinary error).
#[derive(Debug, Clone)]
pub enum Stop {
    Interrupted,
    Safety(String),
    BusLost,
    BusDead(String),
    Timeout(String),
}

impl std::fmt::Display for Stop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Stop::Interrupted => write!(f, "interrupted"),
            Stop::Safety(m) => write!(f, "SAFETY STOP: {m}"),
            Stop::BusLost => write!(f, "EtherCAT bus lost (no answer from the robot)"),
            Stop::BusDead(m) => write!(f, "bus thread not running: {m}"),
            Stop::Timeout(m) => write!(f, "time limit: {m}"),
        }
    }
}

impl std::error::Error for Stop {}

pub struct Robot<'a> {
    cmd_tx: Producer<Cmd>,
    ev_rx: Consumer<Event>,
    pub shared: &'a Shared,
    pub store: &'a Mutex<Store>,
    pub cycle_s: f64,
    outstanding: bool,
    exit_done: Option<ExitSeq>,
}

impl<'a> Robot<'a> {
    pub fn new(cmd_tx: Producer<Cmd>, ev_rx: Consumer<Event>, shared: &'a Shared, store: &'a Mutex<Store>, cycle_s: f64) -> Self {
        Robot { cmd_tx, ev_rx, shared, store, cycle_s, outstanding: false, exit_done: None }
    }

    pub fn snapshot(&self) -> Snapshot {
        match self.shared.snap.lock() {
            Ok(s) => *s,
            Err(p) => *p.into_inner(),
        }
    }

    /// Latest recorded frame (newest robot sample).
    pub fn latest(&self) -> Option<Row> {
        let s = self.snapshot();
        s.fb.have_row.then_some(s.fb.row)
    }

    pub fn target_joints(&self) -> Option<[f64; 6]> {
        self.latest().map(|r| alljoints(&r, K_TPOS))
    }

    fn push(&mut self, c: Cmd) -> Result<()> {
        self.cmd_tx.push(c).map_err(|_| anyhow!(Stop::BusDead("command ring full".into())))
    }

    pub fn set_guard(&mut self, g: Option<GuardCfg>) -> Result<()> {
        self.push(Cmd::Guard(g))
    }

    /// Run one controller on the bus thread and wait for it. A stop request or `max_s` aborts it.
    pub fn exec<C: Controller>(&mut self, c: C, max_s: f64) -> Result<C> {
        if self.shared.emergency_exit.load(Ordering::SeqCst) {
            bail!(Stop::Interrupted);
        }
        self.push(Cmd::Install(Box::new(c)))?;
        self.outstanding = true;
        let t_end = rt::now() + max_s;
        let mut aborted = false;
        let mut timed_out = false;
        let mut hb = (self.shared.heartbeat.load(Ordering::SeqCst), rt::now());
        loop {
            while let Ok(Event::Done(b, o)) = self.ev_rx.pop() {
                match b.into_any().downcast::<C>() {
                    Ok(c) => {
                        self.outstanding = false;
                        return match o {
                            Outcome::Finished if !timed_out => Ok(*c),
                            Outcome::Finished | Outcome::Aborted if timed_out => {
                                Err(anyhow!(Stop::Timeout(format!("{} s", max_s))))
                            }
                            Outcome::Finished | Outcome::Aborted => Err(anyhow!(Stop::Interrupted)),
                            Outcome::Safety(m) => Err(anyhow!(Stop::Safety(m))),
                            Outcome::BusLost => Err(anyhow!(Stop::BusLost)),
                        };
                    }
                    Err(other) => {
                        if let Ok(e) = other.downcast::<ExitSeq>() {
                            self.exit_done = Some(*e); // the bus thread's own exit sequence
                        }
                    }
                }
            }
            let now = rt::now();
            if !aborted && (rt::stop_requests() >= 1 || now > t_end) {
                timed_out = rt::stop_requests() == 0;
                self.push(Cmd::Abort)?;
                aborted = true;
            }
            let h = self.shared.heartbeat.load(Ordering::SeqCst);
            if h != hb.0 {
                hb = (h, now);
            } else if now - hb.1 > 0.5 {
                bail!(Stop::BusDead("no bus cycle for 0.5 s".into()));
            }
            rt::sleep_s(0.001);
        }
    }

    /// Hold a robot control bit until the status confirms it; false if it timed out.
    pub fn bits(&mut self, b: Bits) -> Result<bool> {
        let r = self.exec(b, 60.0)?;
        Ok(r.ok)
    }

    /// A queued setting; on refusal: ResetError + ResumeMotion, false (like MecaIO._setting).
    pub fn setting(&mut self, id: u32, value: f64) -> Result<bool> {
        let c = self.exec(Config::new(id, value, 10.0), 15.0)?;
        if c.ok {
            return Ok(true);
        }
        say!("  setting {id}({value}) refused (error {})", c.error);
        self.recover()?;
        Ok(false)
    }

    pub fn set_vel_limit(&mut self, pct: f64) -> Result<bool> {
        self.setting(cmd::SET_JOINT_VEL_LIMIT, pct)
    }

    pub fn set_joint_acc(&mut self, pct: f64) -> Result<bool> {
        self.setting(cmd::SET_JOINT_ACC, pct)
    }

    pub fn set_joint_vel(&mut self, pct: f64) -> Result<bool> {
        self.setting(cmd::SET_JOINT_VEL, pct)
    }

    pub fn set_blending(&mut self, pct: f64) -> Result<bool> {
        self.setting(cmd::SET_BLENDING, pct)
    }

    pub fn set_vel_timeout(&mut self, s: f64) -> Result<bool> {
        self.setting(cmd::SET_VEL_TIMEOUT, s)
    }

    /// ResetError if in error, then ResumeMotion.
    pub fn recover(&mut self) -> Result<()> {
        if self.snapshot().fb.inp.error_code != 0 {
            self.bits(Bits::reset_error())?;
        }
        self.exec(ClearMotion::resume_only(), 5.0)?;
        Ok(())
    }

    pub fn clear_motion(&mut self) -> Result<()> {
        self.exec(ClearMotion::new(), 5.0)?;
        Ok(())
    }

    pub fn wait_still(&mut self, timeout: f64) -> Result<bool> {
        Ok(self.exec(WaitStill::new(timeout), timeout + 2.0)?.ok)
    }

    /// Stop, MoveJoints(q), wait until there (return_to_start of the Python script).
    pub fn move_joints_wait(&mut self, q: [f64; 6], timeout: f64) -> Result<()> {
        let m = self.exec(MoveAndWait::new(q, timeout), timeout + 5.0)?;
        if !m.ok {
            bail!(Stop::Timeout(format!("MoveJoints to {q:?} not reached in {timeout} s")));
        }
        Ok(())
    }

    /// Number of frames the bus recorded so far.
    pub fn frames_recorded(&self) -> u64 {
        self.shared.frames.load(Ordering::SeqCst)
    }

    /// Copy of the frames [i0, i1) once the writer has stored them.
    pub fn frames(&self, i0: u64, i1: u64) -> Result<Vec<Row>> {
        let t_end = rt::now() + 2.0;
        loop {
            {
                let s = self.store.lock().map_err(|_| anyhow!("frame store poisoned"))?;
                if s.frames.len() as u64 >= i1 {
                    return Ok(s.frames[i0 as usize..i1 as usize].to_vec());
                }
            }
            if rt::now() > t_end {
                bail!("frames {i0}..{i1} not stored after 2 s");
            }
            rt::sleep_s(0.003);
        }
    }

    /// Wait until the bus thread has cycled and the robot answered (the snapshot is only
    /// meaningful after that: before, it is the default feedback, "not activated, no error").
    pub fn wait_ready(&self, timeout: f64) -> Result<()> {
        let t_end = rt::now() + timeout;
        loop {
            if self.shared.heartbeat.load(Ordering::SeqCst) >= 20 && self.snapshot().fb.ok {
                return Ok(());
            }
            if rt::now() > t_end {
                bail!(Stop::BusLost);
            }
            rt::sleep_s(0.002);
        }
    }

    /// Activation, homing and the settings every run starts from. Returns the start joints.
    pub fn prepare(&mut self, p: &Prepare) -> Result<[f64; 6]> {
        self.wait_ready(2.0)?;
        let s = self.snapshot();
        say!("Robot status: {}", s.fb.inp.status_line());
        if s.fb.inp.error_code != 0 {
            say!("Robot in error {} -> ResetError", s.fb.inp.error_code);
            if !self.bits(Bits::reset_error())? {
                bail!("the robot error {} could not be reset", s.fb.inp.error_code);
            }
        }
        let s = self.snapshot();
        if s.fb.inp.activated && s.fb.inp.sim_activated != p.robot_sim {
            say!("Activated with simulation mode {}: Deactivate first", s.fb.inp.sim_activated as u8);
            if !self.bits(Bits::deactivate())? {
                bail!("Deactivate refused");
            }
        }
        if !self.snapshot().fb.inp.activated {
            say!("Activate{}", if p.robot_sim { " (robot simulation mode: the arm will NOT move)" } else { "" });
            if !self.bits(Bits::activate(p.robot_sim))? {
                bail!("activation failed (status: {})", self.snapshot().fb.inp.status_line());
            }
        }
        if !self.snapshot().fb.inp.homed {
            say!("Home");
            if !self.bits(Bits::home())? {
                bail!("homing failed (status: {})", self.snapshot().fb.inp.status_line());
            }
        }
        if self.snapshot().fb.inp.paused {
            self.exec(ClearMotion::resume_only(), 5.0)?;
        }
        for (name, id, v) in [
            ("SetVelTimeout", cmd::SET_VEL_TIMEOUT, p.vel_timeout),
            ("SetJointVelLimit", cmd::SET_JOINT_VEL_LIMIT, p.vel_limit_pct),
            ("SetJointAcc", cmd::SET_JOINT_ACC, p.joint_acc_pct),
        ] {
            if !self.setting(id, v)? {
                bail!("{name}({v}) refused by the robot");
            }
        }
        self.wait_still(10.0)?;
        rt::sleep_s(0.3);
        let q = self.target_joints().ok_or_else(|| anyhow!("no feedback frame recorded"))?;
        say!("Start joints: {}", fmt6(&q));
        Ok(q)
    }

    /// The exit sequence on the bus thread (stop, Deactivate, DisableEtherCAT), then stop the bus.
    /// Never aborted by stop requests. None if the bus thread could not run it.
    pub fn exit_sequence(&mut self) -> Option<ExitSeq> {
        // 1. get back whatever controller is still out there
        if self.outstanding {
            let _ = self.push(Cmd::Abort);
            let t_end = rt::now() + 3.0;
            while self.outstanding && rt::now() < t_end {
                while let Ok(Event::Done(b, _)) = self.ev_rx.pop() {
                    match b.into_any().downcast::<ExitSeq>() {
                        Ok(e) => self.exit_done = Some(*e),
                        Err(_) => self.outstanding = false,
                    }
                }
                rt::sleep_s(0.001);
            }
            self.outstanding = false;
        }
        // 2. the exit sequence, unless the bus thread already runs its own (second Ctrl-C)
        let emergency = self.shared.emergency_exit.load(Ordering::SeqCst);
        if self.exit_done.is_none() && !emergency && self.push(Cmd::Install(Box::new(ExitSeq::new()))).is_err() {
            return None;
        }
        let t_end = rt::now() + 40.0;
        let mut hb = (self.shared.heartbeat.load(Ordering::SeqCst), rt::now());
        while self.exit_done.is_none() {
            while let Ok(Event::Done(b, _)) = self.ev_rx.pop() {
                if let Ok(e) = b.into_any().downcast::<ExitSeq>() {
                    if e.started() {
                        self.exit_done = Some(*e);
                    } else if !self.shared.emergency_exit.load(Ordering::SeqCst) {
                        // handed back unrun (a controller was still running): install it again
                        rt::sleep_s(0.005);
                        let _ = self.push(Cmd::Install(Box::new(ExitSeq::new())));
                    }
                    // emergency: the bus thread runs its own, keep waiting for it
                }
            }
            let now = rt::now();
            let h = self.shared.heartbeat.load(Ordering::SeqCst);
            if h != hb.0 {
                hb = (h, now);
            } else if now - hb.1 > 1.0 {
                say!("exit sequence: the bus thread stopped cycling");
                break;
            }
            if now > t_end {
                say!("exit sequence: no end after 40 s");
                break;
            }
            rt::sleep_s(0.002);
        }
        let t_end = rt::now() + 2.0;
        while self.push(Cmd::Shutdown).is_err() && rt::now() < t_end {
            rt::sleep_s(0.001);
        }
        self.exit_done.take()
    }
}

pub struct Prepare {
    pub robot_sim: bool,
    pub vel_timeout: f64,
    pub vel_limit_pct: f64,
    pub joint_acc_pct: f64,
}

pub fn fmt6(q: &[f64; 6]) -> String {
    format!("[{}]", q.iter().map(|x| format!("{x:.3}")).collect::<Vec<_>>().join(", "))
}

pub struct Drive<T> {
    pub body: Result<T>,
    pub exit: Option<ExitSeq>,
    pub bus: Option<BusReport>,
    pub store: Store,
}

fn panic_text(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".into()
    }
}

/// Bus thread + writer thread around `body`; always ends with the exit sequence (if the bus
/// thread still runs) and returns what was recorded. A panic in `body` is caught.
pub fn drive<F, T>(fbus: F, mut core: Core, cfg: BusCfg, body: impl FnOnce(&mut Robot) -> Result<T>) -> Drive<T>
where
    F: Fieldbus,
{
    // Small up-front allocations only: when this runs the robot is already in OP and nothing
    // exchanges process data until the bus thread starts (the SM watchdog is ~100 ms; with
    // mlockall every allocated page is touched now). The writer thread grows the store.
    let shared = Shared::new(&core);
    let store = Mutex::new(Store {
        frames: Vec::with_capacity(1 << 16),
        cycles: Vec::with_capacity(1 << 16),
    });
    let (cmd_tx, cmd_rx) = RingBuffer::<Cmd>::new(64);
    let (ev_tx, ev_rx) = RingBuffer::<Event>::new(64);
    let (fr_tx, fr_rx) = RingBuffer::<Row>::new(1 << 14);
    let (cy_tx, cy_rx) = RingBuffer::<[f64; 6]>::new(1 << 15);
    core.set_sinks(Sinks { frames: fr_tx, cycles: cy_tx });
    let wstop = AtomicBool::new(false);
    let cycle_s = cfg.cycle_s;
    let (body_res, exit, bus_rep) = std::thread::scope(|s| {
        let (sh, st, ws) = (&shared, &store, &wstop);
        let writer = match std::thread::Builder::new()
            .name("ecat-writer".into())
            .spawn_scoped(s, move || writer_thread(fr_rx, cy_rx, st, ws))
        {
            Ok(w) => w,
            Err(e) => return (Err(anyhow!("cannot spawn the writer thread: {e}")), None, None),
        };
        let bus = match std::thread::Builder::new()
            .name("ecat-bus".into())
            .spawn_scoped(s, move || bus_thread(core, fbus, cmd_rx, ev_tx, sh, cfg))
        {
            Ok(b) => b,
            Err(e) => {
                ws.store(true, Ordering::SeqCst);
                let _ = writer.join();
                return (Err(anyhow!("cannot spawn the bus thread: {e}")), None, None);
            }
        };
        let mut robot = Robot::new(cmd_tx, ev_rx, sh, st, cycle_s);
        let body_res = match catch_unwind(AssertUnwindSafe(|| body(&mut robot))) {
            Ok(r) => r,
            Err(p) => Err(anyhow!("panic in the test code: {}", panic_text(&p))),
        };
        let exit = robot.exit_sequence();
        let bus_rep = match bus.join() {
            Ok(r) => Some(r),
            Err(p) => {
                say!("the bus thread panicked: {}", panic_text(&p));
                None
            }
        };
        ws.store(true, Ordering::SeqCst);
        let _ = writer.join();
        (body_res, exit, bus_rep)
    });
    let store = store.into_inner().unwrap_or_else(|p| p.into_inner());
    Drive { body: body_res, exit, bus: bus_rep, store }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdo::{ByteOrder, Codec};
    use crate::sim::{SimConfig, SimRobot};

    #[test]
    fn drive_prepares_moves_and_always_exits() {
        let sim = SimRobot::new(SimConfig::default());
        let core = Core::new(Codec::new(ByteOrder::Le), false, 5, 1e-3, [32, 40, 42, 41]);
        let cfg = BusCfg { cycle_s: 1e-3, prio: 0, cpu: None, lost_after_s: 1.0 };
        let d: Drive<()> = drive(sim, core, cfg, |r| {
            let q = r.prepare(&Prepare { robot_sim: false, vel_timeout: 0.05, vel_limit_pct: 20.0, joint_acc_pct: 80.0 })?;
            let mut goal = q;
            goal[5] += 2.0;
            r.move_joints_wait(goal, 10.0)?;
            bail!("a test failing in the middle")
        });
        assert!(d.body.is_err());
        let e = d.exit.expect("exit sequence ran");
        assert!(e.was_activated && e.deactivated && e.link_lost_after.is_some(), "{}", e.summary());
        assert!(d.store.frames.len() > 500 && d.store.cycles.len() >= d.store.frames.len());
        let rep = d.bus.expect("bus report");
        assert_eq!(rep.frames as usize, d.store.frames.len());
    }

    #[test]
    fn drive_survives_a_panic_in_the_body() {
        let sim = SimRobot::new(SimConfig::default());
        let core = Core::new(Codec::new(ByteOrder::Le), false, 5, 1e-3, [32, 40, 42, 41]);
        let cfg = BusCfg { cycle_s: 1e-3, prio: 0, cpu: None, lost_after_s: 1.0 };
        let d: Drive<()> = drive(sim, core, cfg, |_| panic!("boom"));
        assert!(format!("{:#}", d.body.unwrap_err()).contains("boom"));
        assert!(d.exit.is_some_and(|e| e.link_lost_after.is_some()));
    }
}
