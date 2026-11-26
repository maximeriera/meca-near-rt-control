//! Housekeeping controllers: robot control bits (activate, home, reset error...), queued settings
//! (SetJointVelLimit...), MoveJoints and wait, ClearMotion / ResumeMotion, waiting for the robot
//! to stand still, and the exit sequence (stop, Deactivate, DisableEtherCAT).

use std::sync::atomic::{AtomicU32, Ordering};

use crate::bus::{Controller, Cycle, Feedback, Flow};
use crate::into_any;
use crate::pdo::{cmd, Inputs, Outputs};

static MOVE_ID: AtomicU32 = AtomicU32::new(0);

/// A new non-zero MoveID for a queued command (the robot runs it when the MoveID changes).
pub fn next_move_id() -> u16 {
    (MOVE_ID.fetch_add(1, Ordering::Relaxed) % 65535 + 1) as u16
}

/// Counts consecutive new samples where the target joint positions did not move.
#[derive(Clone, Copy, Default)]
pub struct Still {
    prev: Option<[f64; 6]>,
    pub n: u32,
}

impl Still {
    pub fn update(&mut self, fb: &Feedback) {
        if !fb.fresh {
            return;
        }
        let q = fb.inp.joint_set;
        if let Some(p) = self.prev {
            let moved = (0..6).any(|j| (q[j] - p[j]).abs() > 1e-4);
            self.n = if moved { 0 } else { self.n + 1 };
        }
        self.prev = Some(q);
    }
}

// ------------------------------------------------------------------------------------------- //
// Bits                                                                                         //
// ------------------------------------------------------------------------------------------- //

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitSel {
    Activate,
    Deactivate,
    Home,
    ResetError,
}

/// Hold one robot control bit until the status confirms it (3 cycles in a row), then release it.
pub struct Bits {
    pub sel: BitSel,
    until: fn(&Inputs) -> bool,
    timeout: f64,
    sim_mode: Option<bool>,
    t0: Option<f64>,
    hits: u32,
    pub ok: bool,
}

impl Bits {
    pub fn new(sel: BitSel, until: fn(&Inputs) -> bool, timeout: f64) -> Bits {
        Bits { sel, until, timeout, sim_mode: None, t0: None, hits: 0, ok: false }
    }

    /// Activate; `sim`: in the robot's own simulation mode (SimMode bit, the arm does not move).
    pub fn activate(sim: bool) -> Bits {
        Bits { sim_mode: Some(sim), ..Bits::new(BitSel::Activate, |i| i.activated, 30.0) }
    }

    pub fn home() -> Bits {
        Bits::new(BitSel::Home, |i| i.homed, 30.0)
    }

    pub fn deactivate() -> Bits {
        Bits::new(BitSel::Deactivate, |i| !i.activated, 10.0)
    }

    pub fn reset_error() -> Bits {
        Bits::new(BitSel::ResetError, |i| i.error_code == 0, 5.0)
    }

    fn set(&self, out: &mut Outputs, v: bool) {
        match self.sel {
            BitSel::Activate => out.activate = v,
            BitSel::Deactivate => out.deactivate = v,
            BitSel::Home => out.home = v,
            BitSel::ResetError => out.reset_error = v,
        }
    }
}

impl Controller for Bits {
    fn name(&self) -> &'static str {
        "bits"
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        let t0 = *self.t0.get_or_insert(cx.ts);
        if let Some(s) = self.sim_mode {
            out.sim_mode = s; // stays set while the robot is activated in simulation
        }
        out.zero_velocity();
        if fb.ok && (self.until)(&fb.inp) {
            self.hits += 1;
        } else {
            self.hits = 0;
        }
        if self.hits >= 3 || cx.abort {
            self.set(out, false);
            self.ok = self.hits >= 3;
            return Flow::Done;
        }
        if cx.ts - t0 > self.timeout {
            self.set(out, false);
            return Flow::Done;
        }
        self.set(out, true);
        Flow::Continue
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// Queued setting (SetJointVelLimit, SetJointAcc, SetVelTimeout, SetJointVel, SetBlending)     //
// ------------------------------------------------------------------------------------------- //

pub struct Config {
    pub id: u32,
    pub value: f64,
    timeout: f64,
    phase: u8,
    t0: f64,
    mid: u16,
    waited: u32,
    pub ok: bool,
    pub error: u16,
}

impl Config {
    pub fn new(id: u32, value: f64, timeout: f64) -> Config {
        Config { id, value, timeout, phase: 0, t0: 0.0, mid: 0, waited: 0, ok: false, error: 0 }
    }
}

impl Controller for Config {
    fn name(&self) -> &'static str {
        "config"
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        match self.phase {
            0 => {
                self.t0 = cx.ts;
                self.mid = next_move_id();
                out.queued(self.id, &[self.value, 0.0, 0.0, 0.0, 0.0, 0.0], self.mid);
                self.phase = 1;
            }
            1 => {
                if fb.ok && fb.inp.move_id == self.mid {
                    self.phase = 2;
                }
            }
            _ => {
                self.waited += 1;
                if fb.inp.error_code != 0 {
                    self.error = fb.inp.error_code;
                    return Flow::Done;
                }
                if self.waited >= 20 && fb.inp.eob {
                    self.ok = true;
                    return Flow::Done;
                }
            }
        }
        if cx.abort || cx.ts - self.t0 > self.timeout {
            self.error = fb.inp.error_code;
            return Flow::Done;
        }
        Flow::Continue
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// ClearMotion (ClearMove, then a ResumeMotion edge) and ResumeMotion alone                    //
// ------------------------------------------------------------------------------------------- //

pub struct ClearMotion {
    clear: bool,
    phase: u8,
    t0: Option<f64>,
    n: u32,
    pub ok: bool,
}

impl ClearMotion {
    pub fn new() -> ClearMotion {
        ClearMotion { clear: true, phase: 0, t0: None, n: 0, ok: false }
    }

    pub fn resume_only() -> ClearMotion {
        ClearMotion { clear: false, phase: 1, t0: None, n: 0, ok: false }
    }
}

impl Default for ClearMotion {
    fn default() -> Self {
        ClearMotion::new()
    }
}

impl Controller for ClearMotion {
    fn name(&self) -> &'static str {
        "clear_motion"
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        let t0 = *self.t0.get_or_insert(cx.ts);
        out.zero_velocity();
        match self.phase {
            0 => {
                out.clear_move = true;
                if (fb.ok && fb.inp.cleared && cx.ts - t0 > 0.003) || cx.ts - t0 > 1.0 {
                    out.clear_move = false;
                    self.phase = 1;
                }
            }
            _ => {
                // ResumeMotion acts on a rising edge, with Pause and ClearMove released.
                self.n += 1;
                out.pause = false;
                out.clear_move = false;
                out.resume_motion = self.n >= 2;
                if self.n > 3 && fb.ok && !fb.inp.paused {
                    out.resume_motion = false;
                    self.ok = true;
                    return Flow::Done;
                }
                if cx.ts - t0 > 2.5 {
                    out.resume_motion = false;
                    self.ok = !fb.inp.paused;
                    return Flow::Done;
                }
            }
        }
        Flow::Continue
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// Wait until the robot stands still                                                            //
// ------------------------------------------------------------------------------------------- //

pub struct WaitStill {
    still: Still,
    timeout: f64,
    t0: Option<f64>,
    pub ok: bool,
}

impl WaitStill {
    pub fn new(timeout: f64) -> WaitStill {
        WaitStill { still: Still::default(), timeout, t0: None, ok: false }
    }
}

impl Controller for WaitStill {
    fn name(&self) -> &'static str {
        "wait_still"
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        let t0 = *self.t0.get_or_insert(cx.ts);
        out.zero_velocity();
        self.still.update(fb);
        if self.still.n >= 10 && fb.inp.eob && cx.ts - t0 > 0.02 {
            self.ok = true;
            return Flow::Done;
        }
        if cx.abort || cx.ts - t0 > self.timeout {
            return Flow::Done;
        }
        Flow::Continue
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// MoveJoints and wait (return_to_start / home of the tests)                                    //
// ------------------------------------------------------------------------------------------- //

/// Stop a running velocity command, MoveJoints to `q`, wait until the target is there and still,
/// then 0.2 s more (like MecaIO.return_to_start).
pub struct MoveAndWait {
    pub q: [f64; 6],
    timeout: f64,
    phase: u8,
    t0: Option<f64>,
    t_phase: f64,
    mid: u16,
    still: Still,
    pub ok: bool,
}

impl MoveAndWait {
    pub fn new(q: [f64; 6], timeout: f64) -> MoveAndWait {
        MoveAndWait { q, timeout, phase: 0, t0: None, t_phase: 0.0, mid: 0, still: Still::default(), ok: false }
    }
}

impl Controller for MoveAndWait {
    fn name(&self) -> &'static str {
        "move_and_wait"
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        let t0 = *self.t0.get_or_insert(cx.ts);
        self.still.update(fb);
        if cx.abort || cx.ts - t0 > self.timeout {
            out.zero_velocity();
            return Flow::Done;
        }
        match self.phase {
            0 => {
                // stop a velocity stream first (stop() of the Python script)
                let vel = cmd::is_velocity(out.command_id) && out.move_id == 0;
                out.zero_velocity();
                if !vel || (self.still.n >= 10 && cx.ts - t0 > 0.02) || cx.ts - t0 > 2.0 {
                    self.mid = next_move_id();
                    out.queued(cmd::MOVE_JOINTS, &self.q, self.mid);
                    self.phase = 1;
                }
            }
            1 => {
                if fb.ok && fb.inp.move_id == self.mid {
                    self.phase = 2;
                    self.still.n = 0;
                }
            }
            2 => {
                let there = (0..6).all(|j| (fb.inp.joint_set[j] - self.q[j]).abs() < 0.01);
                if there && self.still.n >= 5 && fb.inp.eob {
                    self.phase = 3;
                    self.t_phase = cx.ts;
                }
            }
            _ => {
                if cx.ts - self.t_phase >= 0.2 {
                    self.ok = true;
                    return Flow::Done;
                }
            }
        }
        Flow::Continue
    }

    into_any!();
}

// ------------------------------------------------------------------------------------------- //
// Exit sequence                                                                                //
// ------------------------------------------------------------------------------------------- //

/// The end of every EtherCAT session, on every path: stop (velocity zero, queued motion cleared),
/// Deactivate, then DisableEtherCAT held until the robot stops answering. It ignores abort
/// requests: nothing may cut it short (it has its own time limits, ~25 s at most).
/// Leaving the bus only shows the robot dropped EtherCAT; the caller then checks TCP port 10000.
pub struct ExitSeq {
    phase: u8,
    t0: Option<f64>,
    t_phase: f64,
    still: Still,
    pub skipped: bool,
    pub was_activated: bool,
    pub deactivated: bool,
    pub disable_started: Option<f64>,
    pub link_lost_after: Option<f64>,
    pub timed_out: bool,
}

impl ExitSeq {
    pub const NAME: &'static str = "exit";

    pub fn new() -> ExitSeq {
        ExitSeq {
            phase: 0,
            t0: None,
            t_phase: 0.0,
            still: Still::default(),
            skipped: false,
            was_activated: false,
            deactivated: false,
            disable_started: None,
            link_lost_after: None,
            timed_out: false,
        }
    }

    pub fn summary(&self) -> String {
        if self.skipped {
            return "bus already lost: nothing sent".into();
        }
        format!(
            "{}; DisableEtherCAT {}",
            if self.was_activated {
                if self.deactivated { "deactivated" } else { "NOT deactivated (timeout)" }
            } else {
                "was not activated"
            },
            match self.link_lost_after {
                Some(t) => format!("-> robot left the bus after {:.2} s", t),
                None => "held 15 s, the robot still answers".into(),
            }
        )
    }
}

impl Default for ExitSeq {
    fn default() -> Self {
        ExitSeq::new()
    }
}

impl Controller for ExitSeq {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        let t0 = *self.t0.get_or_insert(cx.ts);
        self.still.update(fb);
        let dt = cx.ts - self.t_phase;
        out.activate = false;
        out.home = false;
        out.reset_error = false;
        out.resume_motion = false;
        match self.phase {
            0 => {
                if !fb.ok && fb.ok_age > 0.5 && cx.ts - t0 > 0.05 {
                    // nobody answers any more: the robot is gone already (or the cable)
                    self.skipped = true;
                    self.link_lost_after = Some(0.0);
                    return Flow::Done;
                }
                self.was_activated = fb.inp.activated;
                self.phase = 1;
                self.t_phase = cx.ts;
                out.zero_velocity();
            }
            1 => {
                // stop: velocity zero; a queued motion is cleared (ClearMove pauses it)
                out.zero_velocity();
                let queued = !cmd::is_velocity(out.command_id) && out.move_id != 0;
                if queued && fb.inp.activated && !fb.inp.eob {
                    out.clear_move = true;
                }
                let stopped = self.still.n >= 10 || !fb.inp.activated;
                if (stopped && dt > 0.05) || dt > 1.5 {
                    if cmd::is_velocity(out.command_id) {
                        out.set_point = false;
                    }
                    self.phase = 2;
                    self.t_phase = cx.ts;
                }
            }
            2 => {
                if fb.ok && !fb.inp.activated {
                    out.deactivate = false;
                    out.clear_move = false;
                    out.sim_mode = false;
                    self.deactivated = self.was_activated;
                    self.phase = 3;
                    self.t_phase = cx.ts;
                    self.disable_started = Some(cx.ts);
                } else if dt > 5.0 || (!fb.ok && fb.ok_age > 0.5) {
                    out.deactivate = false;
                    self.phase = 3;
                    self.t_phase = cx.ts;
                    self.disable_started = Some(cx.ts);
                } else {
                    out.deactivate = true;
                }
            }
            _ => {
                out.disable_ethercat = true;
                if !fb.ok && fb.ok_age >= 0.5 && dt > 0.05 {
                    self.link_lost_after = Some(dt);
                    return Flow::Done;
                }
                if dt > 15.0 {
                    self.timed_out = true;
                    return Flow::Done;
                }
            }
        }
        Flow::Continue
    }

    into_any!();
}
