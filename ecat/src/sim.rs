//! A crude Meca500 behind the PDO image, for `--sim` dry runs and the unit tests. It follows the
//! cyclic protocol (MoveID handshake, bits, velocity commands re-applied every cycle with the
//! velocity timeout) and the J6 velocity law of meca_latency_test.py's SimIO: delay 3.5 ms then a
//! moving average T = max(15.6 ms, 1 ms per % of SetJointVelLimit) * 80 / SetJointAcc. Its clock
//! ticks every 1 ms of its own time and drifts against the host (40 ppm), like a real slave.
//! Nothing here says anything about a real robot.

use std::collections::VecDeque;

use crate::fieldbus::{BusError, Fieldbus};
use crate::pdo::{cmd, dyn_type, ByteOrder, Codec, Inputs, Outputs, IN_LEN, OUT_LEN};
use crate::rt;

const DT: f64 = 1e-3;
const ENC_STEP: f64 = 0.000862;

#[derive(Clone, Debug)]
pub struct SimConfig {
    pub order: ByteOrder,
    pub joint_vmax: [f64; 6],
    pub start: [f64; 6],
    /// Robot clock rate error against the host [s/s].
    pub drift: f64,
    /// A robot tick is seen by a frame only this long after it happened [s].
    pub transport: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        SimConfig {
            order: ByteOrder::Le,
            joint_vmax: [150.0, 150.0, 180.0, 300.0, 300.0, 500.0],
            start: [0.0; 6],
            drift: 40e-6,
            transport: 60e-6,
        }
    }
}

pub struct SimRobot {
    c: Codec,
    cfg: SimConfig,
    h0: Option<f64>,
    n: u64,
    out: Outputs,
    prev: Outputs,
    activated: bool,
    homed: bool,
    sim: bool,
    paused: bool,
    cleared: bool,
    error: u16,
    act_at: Option<u64>,
    home_at: Option<u64>,
    vl_pct: f64,
    acc_pct: f64,
    jv_pct: f64,
    blend_pct: f64,
    vel_timeout: f64,
    last_mid: u16,
    echo: u16,
    queue: VecDeque<[f64; 6]>,
    vcmd: [f64; 6],
    vcmd_tick: u64,
    hist: VecDeque<[f64; 6]>,
    tp: [f64; 6],
    tv: [f64; 6],
    servo: [f64; 6],
    ap: [f64; 6],
    av: [f64; 6],
    tp_hist: VecDeque<[f64; 6]>,
    gone: bool,
    /// Test hook: no answer at all (cable pulled) while true.
    pub mute: bool,
}

impl SimRobot {
    pub fn new(cfg: SimConfig) -> Self {
        let start = cfg.start;
        SimRobot {
            c: Codec::new(cfg.order),
            cfg,
            h0: None,
            n: 0,
            out: Outputs::default(),
            prev: Outputs::default(),
            activated: false,
            homed: false,
            sim: false,
            paused: false,
            cleared: false,
            error: 0,
            act_at: None,
            home_at: None,
            vl_pct: 100.0,
            acc_pct: 80.0,
            jv_pct: 25.0,
            blend_pct: 100.0,
            vel_timeout: 0.05,
            last_mid: 0,
            echo: 0,
            queue: VecDeque::with_capacity(16384),
            vcmd: [0.0; 6],
            vcmd_tick: 0,
            hist: VecDeque::with_capacity(512),
            tp: start,
            tv: [0.0; 6],
            servo: start,
            ap: start,
            av: [0.0; 6],
            tp_hist: VecDeque::with_capacity(8),
            gone: false,
            mute: false,
        }
    }

    pub fn has_left_ethercat(&self) -> bool {
        self.gone
    }

    fn ready(&self) -> bool {
        self.activated && self.homed && self.error == 0 && !self.paused
    }

    fn execute(&mut self, id: u32, a: [f64; 6]) {
        match id {
            cmd::MOVE_JOINTS => {
                if self.activated && self.homed && self.error == 0 {
                    self.queue.push_back(a);
                } else {
                    self.error = 1005; // motion while not ready
                }
            }
            cmd::SET_BLENDING => self.blend_pct = a[0].clamp(0.0, 100.0),
            cmd::SET_JOINT_VEL => self.jv_pct = a[0].clamp(0.001, 100.0),
            cmd::SET_JOINT_ACC => {
                if (0.001..=150.0).contains(&a[0]) {
                    self.acc_pct = a[0];
                } else {
                    self.error = 1006;
                }
            }
            cmd::SET_VEL_TIMEOUT => self.vel_timeout = a[0].clamp(0.001, 1.0),
            cmd::SET_JOINT_VEL_LIMIT => {
                if (0.001..=100.0).contains(&a[0]) {
                    self.vl_pct = a[0];
                } else {
                    self.error = 1006;
                }
            }
            _ => {}
        }
    }

    fn latch(&mut self, o: Outputs) {
        self.prev = self.out;
        self.out = o;
        if o.disable_ethercat {
            self.gone = true;
        }
        let args = o.args.map(|x| x as f64);
        if o.set_point && o.move_id != 0 && o.move_id != self.last_mid && !o.clear_move {
            self.last_mid = o.move_id;
            self.execute(o.command_id, args);
            self.echo = o.move_id;
        }
        if o.set_point && o.move_id == 0 && o.command_id == cmd::MOVE_JOINTS_VEL && self.ready() {
            for j in 0..6 {
                let lim = self.cfg.joint_vmax[j] * self.vl_pct / 100.0;
                self.vcmd[j] = args[j].clamp(-lim, lim); // above the limit: clamped silently
            }
            self.vcmd_tick = self.n;
        }
        if o.resume_motion && !self.prev.resume_motion && !o.pause && !o.clear_move {
            self.paused = false;
            self.cleared = false;
        }
    }

    fn tick(&mut self) {
        let o = self.out;
        if o.deactivate {
            self.activated = false;
            self.homed = false;
            self.sim = false;
            self.act_at = None;
            self.home_at = None;
            self.queue.clear();
            self.vcmd = [0.0; 6];
            self.tv = [0.0; 6];
        } else if o.activate && !self.activated && self.act_at.is_none() {
            self.act_at = Some(self.n + 50);
        }
        if let Some(t) = self.act_at {
            if self.n >= t {
                self.activated = true;
                self.sim = o.sim_mode;
                self.paused = false;
                self.act_at = None;
            }
        }
        if o.home && self.activated && !self.homed && self.home_at.is_none() {
            self.home_at = Some(self.n + 100);
        }
        if let Some(t) = self.home_at {
            if self.n >= t {
                self.homed = true;
                self.home_at = None;
            }
        }
        if o.reset_error {
            self.error = 0;
        }
        if o.pause {
            self.paused = true;
        }
        if o.clear_move {
            self.queue.clear();
            self.vcmd = [0.0; 6];
            self.cleared = true;
            self.paused = true;
        }
        let vel_mode = cmd::is_velocity(o.command_id) && o.move_id == 0 && o.set_point;
        let fresh = (self.n.saturating_sub(self.vcmd_tick)) as f64 * DT <= self.vel_timeout;
        let u = if vel_mode && fresh && self.ready() { self.vcmd } else { [0.0; 6] };
        self.hist.push_back(u);
        while self.hist.len() > 400 {
            self.hist.pop_front();
        }

        if let Some(goal) = self.queue.front().copied() {
            if !self.paused {
                let mut done = true;
                let mut far = 0f64;
                for j in 0..6 {
                    let vmax = self.cfg.joint_vmax[j] * self.jv_pct / 100.0;
                    let acc = self.cfg.joint_vmax[j] * self.acc_pct / 100.0 * 10.0;
                    let d = goal[j] - self.tp[j];
                    let vd = d.signum() * vmax.min((2.0 * acc * d.abs()).sqrt());
                    self.tv[j] += (vd - self.tv[j]).clamp(-acc * DT, acc * DT);
                    let step = self.tv[j] * DT;
                    if step.abs() >= d.abs() {
                        self.tp[j] = goal[j];
                        self.tv[j] = 0.0;
                    } else {
                        self.tp[j] += step;
                    }
                    far = far.max((goal[j] - self.tp[j]).abs());
                    if (goal[j] - self.tp[j]).abs() > 1e-6 {
                        done = false;
                    }
                }
                if done || (self.queue.len() > 1 && far < 0.5 * self.blend_pct / 100.0) {
                    self.queue.pop_front();
                }
            } else {
                for v in self.tv.iter_mut() {
                    *v *= 0.9;
                }
            }
        } else {
            let t_ms = (15.6f64.max(self.vl_pct) * 80.0 / self.acc_pct).max(1.0);
            let nt = t_ms.round() as usize;
            let tau = 4usize;
            let len = self.hist.len();
            let mut s = [0.0; 6];
            for k in 0..nt {
                if len > tau + k {
                    let h = self.hist[len - 1 - tau - k];
                    for j in 0..6 {
                        s[j] += h[j];
                    }
                }
            }
            for j in 0..6 {
                self.tv[j] = s[j] / nt as f64;
                self.tp[j] += self.tv[j] * DT;
            }
        }
        self.tp_hist.push_back(self.tp);
        while self.tp_hist.len() > 3 {
            self.tp_hist.pop_front();
        }
        let delayed = self.tp_hist[0];
        for j in 0..6 {
            self.servo[j] += (delayed[j] - self.servo[j]) * DT / 0.003;
            let q = (self.servo[j] / ENC_STEP).round() * ENC_STEP;
            self.av[j] = (q - self.ap[j]) / DT;
            self.ap[j] = q;
        }
    }

    fn inputs(&self) -> Inputs {
        let total_us = 1_000_000_000u64 + self.n * 1000;
        let still = self.tv.iter().all(|v| v.abs() < 1e-6);
        let mut i = Inputs {
            busy: self.act_at.is_some() || self.home_at.is_some(),
            activated: self.activated,
            homed: self.homed,
            sim_activated: self.sim,
            error_code: self.error,
            move_id: self.echo,
            fifo_space: (13000 - self.queue.len().min(13000)) as u16,
            paused: self.paused,
            eob: self.queue.is_empty() && still,
            eom: still,
            cleared: self.cleared,
            joint_set: self.tp,
            ts_s: (total_us / 1_000_000) as u32,
            ts_us: (total_us % 1_000_000) as u32,
            dyn_cycles: self.n as u32,
            ..Default::default()
        };
        for k in 0..4 {
            let t = self.out.dyn_types[k];
            i.dyn_type[k] = t;
            i.dyn_val[k] = match t {
                dyn_type::TARGET_JOINT_VEL => self.tv,
                dyn_type::JOINT_POS => self.ap,
                dyn_type::JOINT_VEL => self.av,
                _ => [0.0; 6],
            };
        }
        i
    }
}

impl Fieldbus for SimRobot {
    fn exchange(&mut self, out: &[u8; OUT_LEN], inp: &mut [u8; IN_LEN]) -> Result<u16, BusError> {
        if self.gone || self.mute {
            rt::sleep_s(0.002); // what an unanswered frame costs
            return Err(if self.gone { BusError::Gone } else { BusError::Timeout });
        }
        let now = rt::now();
        let h0 = *self.h0.get_or_insert(now - 0.0005);
        let target = ((now - self.cfg.transport - h0) / (DT * (1.0 + self.cfg.drift))).floor();
        let target = if target > 0.0 { target as u64 } else { 0 };
        let mut budget = 20_000; // after a long pause, do not spin forever
        while self.n < target && budget > 0 {
            self.tick();
            self.n += 1;
            budget -= 1;
        }
        self.n = self.n.max(target);
        self.inputs().encode(&self.c, inp);
        let o = Outputs::decode(&self.c, out);
        self.latch(o);
        Ok(3)
    }

    fn is_sim(&self) -> bool {
        true
    }
}
