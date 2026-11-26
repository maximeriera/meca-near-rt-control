//! Command line. Test options keep the names and defaults of meca_latency_test.py (parse_args),
//! so a run can be compared option for option with a TCP run; python_args() writes them under the
//! Python names into ecat_run.json.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::{json, Map, Value};

#[derive(Parser, Debug)]
#[command(name = "meca_ecat", version, about = "Meca500 latency tests over EtherCAT (Rust side of meca_latency_test.py)")]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run the tests. Supervised: if the test process dies, the supervisor gets the robot back
    /// to TCP/IP (restore).
    #[cfg(feature = "suite")]
    Run(RunArgs),
    /// The supervised test process (started by `run`).
    #[cfg(feature = "suite")]
    #[command(hide = true)]
    Worker(RunArgs),
    /// Switch to EtherCAT, OP, show the decoded feedback for a few seconds (no motion), then back
    /// to TCP/IP. The first thing to run on a new station.
    Probe(ProbeArgs),
    /// Get the robot back to TCP/IP (nothing if it already answers).
    Restore(RestoreArgs),
    /// TCP/IP or EtherCAT? (identity / EtherCAT state)
    Status(EcatOpts),
    /// CPU load of the soak test (child process).
    #[command(hide = true)]
    Burn {
        #[arg(long, default_value_t = 600.0)]
        secs: f64,
    },
}

#[derive(Args, Debug, Clone)]
pub struct EcatOpts {
    /// Robot address on TCP/IP (before and after the EtherCAT session)
    #[arg(long, default_value = "192.168.0.100")]
    pub ip: String,
    /// Network interface the robot's ETHERNET1 (EtherCAT IN) is plugged into, e.g. enp3s0
    #[arg(long)]
    pub iface: Option<String>,
    /// EtherCAT cycle [us] (the Meca500 supports 1 ms)
    #[arg(long, default_value_t = 1000)]
    pub cycle_us: u64,
    /// Byte order of the cyclic data: auto (detected), le, be
    #[arg(long, default_value = "auto")]
    pub byte_order: String,
    /// Dynamic data types of the 4 input slots (32 target joint vel, 40 joint pos, 42 joint vel,
    /// 41 pose, 34 target cart vel, 44 cart vel)
    #[arg(long, value_delimiter = ',', default_value = "32,40,42,41")]
    pub dyn_slots: Vec<u32>,
    /// SCHED_FIFO priority of the bus thread
    #[arg(long, default_value_t = 80)]
    pub rt_prio: i32,
    /// SCHED_FIFO priority of the ethercrab tx/rx thread
    #[arg(long, default_value_t = 90)]
    pub txrx_prio: i32,
    /// CPU of the bus thread (an isolated core)
    #[arg(long)]
    pub bus_cpu: Option<usize>,
    /// CPU of the tx/rx thread (another isolated core; keep the NIC interrupts elsewhere)
    #[arg(long)]
    pub txrx_cpu: Option<usize>,
    /// How long to look for the robot on EtherCAT after SwitchToEtherCAT [s]
    #[arg(long, default_value_t = 60.0)]
    pub scan_timeout: f64,
    /// How long to wait for TCP/IP after DisableEtherCAT [s]
    #[arg(long, default_value_t = 120.0)]
    pub tcp_wait: f64,
    /// ethercrab PDU timeout [ms]
    #[arg(long, default_value_t = 20)]
    pub pdu_timeout_ms: u64,
    /// Do not check that --ip is routed through --iface
    #[arg(long)]
    pub no_route_check: bool,
}

impl EcatOpts {
    pub fn iface(&self) -> Result<String> {
        self.iface.clone().ok_or_else(|| anyhow!("--iface is required (the port the robot's ETHERNET1 is plugged into)"))
    }

    pub fn cycle_s(&self) -> f64 {
        self.cycle_us as f64 * 1e-6
    }

    pub fn slots(&self) -> [u32; 4] {
        let mut s = [0u32; 4];
        for (i, v) in self.dyn_slots.iter().take(4).enumerate() {
            s[i] = *v;
        }
        s
    }
}

#[derive(Args, Debug, Clone)]
pub struct ProbeArgs {
    #[command(flatten)]
    pub ecat: EcatOpts,
    /// How long to show the feedback [s]
    #[arg(long, default_value_t = 5.0)]
    pub seconds: f64,
}

#[derive(Args, Debug, Clone)]
pub struct RestoreArgs {
    #[command(flatten)]
    pub ecat: EcatOpts,
    /// Write DisableEtherCAT over CoE (SDO 0x7200:07) in PRE-OP instead of the OP sequence
    #[arg(long)]
    pub sdo: bool,
}

#[cfg(feature = "suite")]
#[derive(Args, Debug, Clone)]
pub struct RunArgs {
    #[command(flatten)]
    pub ecat: EcatOpts,
    /// Dry run against a simulated robot (no EtherCAT, no TCP)
    #[arg(long)]
    pub sim: bool,
    /// Byte order of the simulated robot (to exercise the detection)
    #[arg(long, default_value = "le")]
    pub sim_byte_order: String,
    /// Activate the robot in its own simulation mode (SimMode bit): real EtherCAT, the arm does not move
    #[arg(long)]
    pub robot_sim: bool,
    #[arg(long, default_value_t = 6, value_parser = clap::value_parser!(u8).range(1..=6))]
    pub joint: u8,
    /// Tests: step, sine, accsweep, multijoint, servo, servomulti, replay, soak; groups vel, servo, all
    #[arg(long, default_value = "step,sine,accsweep")]
    pub tests: String,
    /// Command / servo period [s], a multiple of the cycle (Python default 2 ms; here one cycle)
    #[arg(long, default_value_t = 0.001)]
    pub period: f64,
    #[arg(long, default_value_t = 25.0)]
    pub vel_limit_pct: f64,
    /// SetJointAcc [%], always applied (not readable over EtherCAT); 80 = activation default
    #[arg(long, default_value_t = 80.0)]
    pub joint_acc_pct: f64,
    #[arg(long, default_value_t = 0.05)]
    pub vel_timeout: f64,
    #[arg(long)]
    pub amp: Option<f64>,
    #[arg(long, default_value_t = 45.0)]
    pub max_excursion: f64,
    #[arg(long, default_value_t = 5)]
    pub repeats: usize,
    #[arg(long, value_delimiter = ',')]
    pub joint_vmax: Option<Vec<f64>>,
    // velocity tests
    #[arg(long, default_value_t = 0.3)]
    pub step_hold: f64,
    #[arg(long, value_delimiter = ',', default_value = "0.5,1,2,3,5,7,10,14,20")]
    pub freqs: Vec<f64>,
    #[arg(long)]
    pub sine_acc: Option<f64>,
    #[arg(long, value_delimiter = ',', default_value = "5,10,15,20,25,30,50,100")]
    pub acc_sweep_pcts: Vec<f64>,
    #[arg(long, value_delimiter = ',', default_value = "5,6")]
    pub multi_joints: Vec<u8>,
    #[arg(long, default_value_t = 0.5)]
    pub multi_frac: f64,
    // position (replay pos / posi)
    #[arg(long, default_value_t = 50.0)]
    pub pos_joint_vel_pct: f64,
    #[arg(long, default_value_t = 100.0)]
    pub blending: f64,
    #[arg(long, default_value_t = 60.0)]
    pub pos_backlog_timeout: f64,
    // servo
    #[arg(long, value_delimiter = ',', default_value = "5,10,20,30,45")]
    pub servo_kps: Vec<f64>,
    #[arg(long, default_value_t = 1.0)]
    pub servo_step: f64,
    #[arg(long, default_value_t = 0.6)]
    pub servo_hold: f64,
    #[arg(long, default_value_t = 20.0)]
    pub servo_kp: f64,
    #[arg(long, value_delimiter = ',', default_value = "0.5,1,2,3,5,8,12")]
    pub servo_freqs: Vec<f64>,
    #[arg(long, default_value_t = 2.0)]
    pub servo_amp: f64,
    #[arg(long)]
    pub no_servo_ff: bool,
    #[arg(long)]
    pub servo_adv_ms: Option<f64>,
    /// actual or target
    #[arg(long, default_value = "actual")]
    pub servo_fb: String,
    #[arg(long, default_value_t = 10.0)]
    pub servo_abort: f64,
    #[arg(long, value_delimiter = ',', default_value = "1,3,5")]
    pub servomulti_freqs: Vec<f64>,
    // replay
    #[arg(long, value_delimiter = ',', default_value = "vel,ff,pos,posi")]
    pub replay_modes: Vec<String>,
    #[arg(long)]
    pub ref_file: Option<PathBuf>,
    #[arg(long)]
    pub ref_absolute: bool,
    #[arg(long, default_value_t = 20.0)]
    pub ref_period_ms: f64,
    #[arg(long, default_value_t = 5.0)]
    pub ref_amp: f64,
    #[arg(long, default_value_t = 8.0)]
    pub replay_dur: f64,
    #[arg(long, default_value_t = 1)]
    pub ref_chunk: usize,
    #[arg(long, default_value_t = 0.0)]
    pub ref_chunk_offset_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    pub ref_noise_deg: f64,
    #[arg(long, default_value_t = 0.0)]
    pub ref_jitter_ms: f64,
    #[arg(long, default_value_t = 0.0)]
    pub ref_lead_ms: f64,
    #[arg(long, default_value_t = 0)]
    pub ref_seed: u64,
    #[arg(long, default_value_t = 0.5)]
    pub replay_settle: f64,
    /// linear or quad
    #[arg(long, default_value = "linear")]
    pub ref_predict: String,
    #[arg(long, default_value_t = 5.0)]
    pub pos_interp_ms: f64,
    #[arg(long)]
    pub posi_adv_ms: Option<f64>,
    #[arg(long, default_value_t = 0)]
    pub ref_smooth: usize,
    // soak
    #[arg(long, default_value_t = 120.0)]
    pub soak_dur: f64,
    /// idle, cpu (busy processes), gil (busy thread in this process; name kept from the TCP runs)
    #[arg(long, value_delimiter = ',', default_value = "idle,cpu,gil")]
    pub soak_loads: Vec<String>,
    #[arg(long, default_value_t = 0)]
    pub stress_procs: usize,
    #[arg(long, default_value_t = 2.0)]
    pub soak_amp: f64,
    // external sensor
    #[arg(long, default_value_t = 0.0)]
    pub fb_delay_ms: f64,
    #[arg(long, default_value_t = 0.0)]
    pub fb_rate_hz: f64,
    #[arg(long, default_value_t = 0.0)]
    pub fb_noise_deg: f64,
    // model (analysis)
    #[arg(long)]
    pub model_tau: Option<f64>,
    #[arg(long = "model-T")]
    pub model_t: Option<f64>,
    // output
    /// Output folder (default meca_latency_ecat_[sim_]YYYYmmdd_HHMMSS_J<joint>)
    #[arg(long)]
    pub out: Option<PathBuf>,
    /// Do not run the Python import + analysis at the end
    #[arg(long)]
    pub no_analyze: bool,
    #[arg(long, default_value = "python3")]
    pub python: String,
    /// ecat_bridge.py (default: the one of this repository)
    #[arg(long)]
    pub bridge: Option<PathBuf>,
}

pub const NJ: usize = 6;
pub const R3_VMAX: [f64; 6] = [150.0, 150.0, 180.0, 300.0, 300.0, 500.0];

#[cfg(feature = "suite")]
impl RunArgs {
    pub fn vmax(&self) -> [f64; 6] {
        match &self.joint_vmax {
            Some(v) if v.len() == NJ => [v[0], v[1], v[2], v[3], v[4], v[5]],
            _ => R3_VMAX,
        }
    }

    /// 0-based, sorted, unique (sorted({j - 1 for j in multi_joints})).
    pub fn multi_js(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.multi_joints.iter().filter(|&&j| (1..=6).contains(&j)).map(|&j| j as usize - 1).collect();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The options under the names of meca_latency_test.py (summary.json "config", aggregate.py).
    pub fn python_args(&self) -> Map<String, Value> {
        let mut m = Map::new();
        let mut put = |k: &str, v: Value| {
            m.insert(k.to_string(), v);
        };
        let tests = crate::suite::expand_tests(&self.tests).unwrap_or_default();
        put("ip", json!(self.ecat.ip));
        put("sim", json!(self.sim));
        put("joint", json!(self.joint));
        put("tests", json!(tests.join(",")));
        put("period", json!(self.period));
        put("monitor_interval", json!(self.ecat.cycle_s()));
        put("vel_limit_pct", json!(self.vel_limit_pct));
        put("joint_acc_pct", json!(self.joint_acc_pct));
        put("vel_timeout", json!(self.vel_timeout));
        put("amp", json!(self.amp));
        put("max_excursion", json!(self.max_excursion));
        put("repeats", json!(self.repeats));
        put("rt_reader", json!("ethercat"));
        put("high_priority", json!(!self.sim));
        put("joint_vmax", json!(self.joint_vmax));
        put("step_hold", json!(self.step_hold));
        put("freqs", json!(self.freqs));
        put("sine_acc", json!(self.sine_acc));
        put("acc_sweep_pcts", json!(self.acc_sweep_pcts));
        put("multi_joints", json!(self.multi_joints));
        put("multi_frac", json!(self.multi_frac));
        put("pos_joint_vel_pct", json!(self.pos_joint_vel_pct));
        put("blending", json!(self.blending));
        put("pos_backlog_timeout", json!(self.pos_backlog_timeout));
        put("servo_kps", json!(self.servo_kps));
        put("servo_step", json!(self.servo_step));
        put("servo_hold", json!(self.servo_hold));
        put("servo_kp", json!(self.servo_kp));
        put("servo_freqs", json!(self.servo_freqs));
        put("servo_amp", json!(self.servo_amp));
        put("servo_ff", json!(!self.no_servo_ff));
        put("servo_adv_ms", json!(self.servo_adv_ms));
        put("servo_fb", json!(self.servo_fb));
        put("servo_abort", json!(self.servo_abort));
        put("servomulti_freqs", json!(self.servomulti_freqs));
        put("replay_modes", json!(self.replay_modes));
        put("ref_file", json!(self.ref_file.as_ref().map(|p| std::fs::canonicalize(p).unwrap_or(p.clone()).to_string_lossy().to_string())));
        put("ref_relative", json!(!self.ref_absolute));
        put("ref_period_ms", json!(self.ref_period_ms));
        put("ref_amp", json!(self.ref_amp));
        put("replay_dur", json!(self.replay_dur));
        put("ref_chunk", json!(self.ref_chunk));
        put("ref_chunk_offset_deg", json!(self.ref_chunk_offset_deg));
        put("ref_noise_deg", json!(self.ref_noise_deg));
        put("ref_jitter_ms", json!(self.ref_jitter_ms));
        put("ref_lead_ms", json!(self.ref_lead_ms));
        put("ref_seed", json!(self.ref_seed));
        put("replay_settle", json!(self.replay_settle));
        put("ref_predict", json!(self.ref_predict));
        put("pos_interp_ms", json!(self.pos_interp_ms));
        put("posi_adv_ms", json!(self.posi_adv_ms));
        put("ref_smooth", json!(self.ref_smooth));
        put("soak_dur", json!(self.soak_dur));
        put("soak_loads", json!(self.soak_loads));
        put("stress_procs", json!(self.stress_procs));
        put("soak_amp", json!(self.soak_amp));
        put("fb_delay_ms", json!(self.fb_delay_ms));
        put("fb_rate_hz", json!(self.fb_rate_hz));
        put("fb_noise_deg", json!(self.fb_noise_deg));
        put("model_tau", json!(self.model_tau));
        put("model_T", json!(self.model_t));
        put("transport", json!("ethercat"));
        put("iface", json!(self.ecat.iface));
        put("cycle_us", json!(self.ecat.cycle_us));
        put("dyn_slots", json!(self.ecat.dyn_slots));
        put("robot_sim", json!(self.robot_sim));
        put("rt_prio", json!(self.ecat.rt_prio));
        put("txrx_prio", json!(self.ecat.txrx_prio));
        put("bus_cpu", json!(self.ecat.bus_cpu));
        put("txrx_cpu", json!(self.ecat.txrx_cpu));
        m
    }
}
