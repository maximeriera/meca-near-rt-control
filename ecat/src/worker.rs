//! The test process: switch to EtherCAT, run the tests on the bus thread, always the exit
//! sequence, then TCP/IP must answer again (else restore), then write the run folder and hand it
//! to the Python analysis. Exit codes (read by the supervisor):
//!   0 tests done, robot back on TCP/IP      2 tests stopped / failed, robot back on TCP/IP
//!   3 never switched to EtherCAT             10 robot NOT confirmed back on TCP/IP

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{bail, Result};
use serde_json::{json, Map, Value};

use crate::bridge::{self, Bridge};
use crate::bus::{BusCfg, Core};
use crate::cli::RunArgs;
use crate::fieldbus::Fieldbus;
use crate::lifecycle::{self, SessionOpts};
use crate::out::RunOut;
use crate::pdo::{ByteOrder, Codec, IN_LEN, OUT_LEN};
use crate::robot::{drive, Drive, Prepare, Robot, Stop};
use crate::sim::{SimConfig, SimRobot};
use crate::{log, rt, say, suite};

pub const BACK_OK: u8 = 0;
pub const BACK_AFTER_ERROR: u8 = 2;
pub const NOT_SWITCHED: u8 = 3;
pub const NOT_BACK: u8 = 10;

pub fn run(a: RunArgs) -> ExitCode {
    match run_inner(&a) {
        Ok(c) => ExitCode::from(c),
        Err(e) => {
            say!("ERROR: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn check_args(a: &RunArgs) -> Result<()> {
    suite::expand_tests(&a.tests)?;
    for l in &a.soak_loads {
        if !suite::LoadGen::KINDS.contains(&l.as_str()) && l != "thread" {
            bail!("unknown soak load '{l}' (idle, cpu, gil)");
        }
    }
    let m = a.period / a.ecat.cycle_s();
    if m < 0.999 || (m - m.round()).abs() > 1e-6 {
        bail!("--period {} must be a multiple of the cycle ({} us)", a.period, a.ecat.cycle_us);
    }
    if !["actual", "target"].contains(&a.servo_fb.as_str()) {
        bail!("--servo-fb actual|target");
    }
    if !["linear", "quad"].contains(&a.ref_predict.as_str()) {
        bail!("--ref-predict linear|quad");
    }
    if !a.sim {
        a.ecat.iface()?;
    }
    Ok(())
}

fn run_inner(a: &RunArgs) -> Result<u8> {
    if let Err(e) = rt::install_stop_signals() {
        say!("WARNING: {e}");
    }
    check_args(a)?;
    let stamp = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let dir = a.out.clone().unwrap_or_else(|| {
        PathBuf::from(format!("meca_latency_ecat_{}{}_J{}", if a.sim { "sim_" } else { "" }, stamp, a.joint))
    });
    let mut out = RunOut::new(dir.clone())?;
    log::set_file(&dir.join("session.log"));
    say!("meca_ecat {} -> {}", env!("CARGO_PKG_VERSION"), dir.display());
    if let Err(e) = rt::lock_memory() {
        if !a.sim {
            say!("WARNING: {e}");
        }
    }
    let br = Bridge { python: a.python.clone(), script: a.bridge.clone().unwrap_or_else(bridge::default_script) };
    let py_args = a.python_args();
    let cycle = a.ecat.cycle_s();
    let j = a.joint as usize - 1;
    let slots = a.ecat.slots();
    let mut clock = vec![rt::clock_pair()];
    let cfg = BusCfg { cycle_s: cycle, prio: if a.sim { 0 } else { a.ecat.rt_prio }, cpu: a.ecat.bus_cpu, lost_after_s: 1.0 };
    let prep = Prepare { robot_sim: a.robot_sim, vel_timeout: a.vel_timeout, vel_limit_pct: a.vel_limit_pct, joint_acc_pct: a.joint_acc_pct };

    // ---- before EtherCAT: identity and SwitchToEtherCAT (an error here leaves the robot on TCP)
    let identity: Map<String, Value> = if a.sim {
        let mut m = Map::new();
        m.insert("model".into(), json!("Meca500 (meca_ecat simulator)"));
        m
    } else {
        match lifecycle::enter(&a.ecat) {
            Ok(id) => id.unwrap_or_default(),
            Err(e) => {
                say!("Not switched to EtherCAT: {e:#}");
                return Ok(NOT_SWITCHED);
            }
        }
    };
    let info = json!({
        "joint_acc_pct": a.joint_acc_pct, "joint_vel_limit_pct": a.vel_limit_pct,
        "joint_vel_pct": Value::Null, "blending_pct": Value::Null, "before_script": {},
        "identity": identity, "rt_reader": "ethercat", "transport": "ethercat", "robot_sim_mode": a.robot_sim,
    });

    // ---- the tests, on the bus thread; drive() always ends with the exit sequence
    let mut slave = Map::new();
    let mut order_used = ByteOrder::Le;
    let (drive_res, back) = if a.sim {
        let order = if a.sim_byte_order == "be" { ByteOrder::Be } else { ByteOrder::Le };
        let mut sim = SimRobot::new(SimConfig { order, joint_vmax: a.vmax(), ..Default::default() });
        let (o, _) = lifecycle::decide_order(&a.ecat, &sample_inputs(&mut sim, cycle));
        order_used = o;
        let core = Core::new(Codec::new(o), false, j, cycle, slots);
        let d = drive(sim, core, cfg, |r| tests_body(r, &prep, &mut out, &info, a, &br, &py_args));
        (Some(d), true)
    } else {
        // The closure returns (recorded run, exit sequence confirmed). Confirmed = DisableEtherCAT
        // was sent and the robot then left the bus; not confirmed (skipped on a dead bus, or the
        // robot still answering) -> come_back restores after 5 s instead of waiting --tcp-wait.
        let r = lifecycle::ecat_session(&a.ecat, SessionOpts::default(), |ctx| {
            slave = ctx.info.clone();
            order_used = ctx.order;
            if !ctx.order_detected && a.ecat.byte_order == "auto" {
                say!("Byte order unknown: no test, exit sequence only");
                let ex = lifecycle::exit_on(ctx, cycle);
                say!("  exit sequence: {}", ex.summary());
                return Ok((None, ex.confirmed()));
            }
            let mut core = Core::new(Codec::new(ctx.order), false, j, cycle, slots);
            core.set_expected_wkc(ctx.expected_wkc);
            let d = drive((ctx.make)(), core, cfg.clone(), |r| tests_body(r, &prep, &mut out, &info, a, &br, &py_args));
            let mut confirmed = d.exit.as_ref().is_some_and(|e| e.confirmed());
            if !confirmed {
                // the bus thread could not finish it: once more from this thread, fresh exchange
                say!("Exit sequence not confirmed on the bus thread: again from the main thread");
                let ex = lifecycle::exit_on(ctx, cycle);
                say!("  exit sequence: {}", ex.summary());
                confirmed = ex.confirmed();
            }
            Ok((Some(d), confirmed))
        });
        let (d, confirmed) = match r {
            Ok(Some((d, c))) => (d, c),
            Ok(None) => (None, false),
            Err(e) => {
                say!("EtherCAT session: {e:#}");
                (None, false)
            }
        };
        // ---- the way back: TCP port 10000 must answer (else restore, second master)
        let back = lifecycle::come_back(&a.ecat, true, confirmed);
        (d, back)
    };
    clock.push(rt::clock_pair());

    // ---- report and run folder
    let mut code = if back { BACK_OK } else { NOT_BACK };
    let mut ecat = Map::new();
    ecat.insert("cycle_s".into(), json!(cycle));
    ecat.insert("byte_order".into(), json!(order_used.name()));
    ecat.insert("sim".into(), json!(a.sim));
    ecat.insert("robot_sim_mode".into(), json!(a.robot_sim));
    ecat.insert("slave".into(), Value::Object(slave));
    ecat.insert("back_on_tcp".into(), json!(back));
    ecat.insert("soak_gil_load".into(), json!("busy thread in the meca_ecat process (no GIL in Rust)"));
    let mut frames = Vec::new();
    let mut cycles = Vec::new();
    if let Some(Drive { body, exit, bus, store }) = drive_res {
        if let Err(e) = &body {
            match e.downcast_ref::<Stop>() {
                Some(Stop::Interrupted) => say!("\nInterrupted"),
                Some(s) => say!("\n{s}"),
                None => say!("\nTests stopped: {e:#}"),
            }
            if code == BACK_OK {
                code = BACK_AFTER_ERROR;
            }
        }
        if let Some(e) = &exit {
            say!("Exit sequence: {}", e.summary());
            ecat.insert(
                "exit".into(),
                json!({"was_activated": e.was_activated, "deactivated": e.deactivated,
                       "link_lost_after_s": e.link_lost_after, "timed_out": e.timed_out, "skipped": e.skipped}),
            );
        }
        if let Some(b) = &bus {
            let s = &b.stats;
            say!(
                "Bus: {} cycles, {} overruns, {} failed exchanges, {} ring drops, byte order confirmed {}{}",
                s.cycles,
                s.overruns,
                s.failed,
                s.ring_drops,
                b.order_confirmed,
                if s.rt_problems.is_empty() { String::new() } else { format!(", RT: {}", s.rt_problems.join("; ")) }
            );
            ecat.insert("byte_order_confirmed".into(), json!(b.order_confirmed));
            ecat.insert("expected_wkc".into(), json!(s.max_wkc));
            ecat.insert("overruns".into(), json!(s.overruns));
            ecat.insert("ring_drops".into(), json!(s.ring_drops));
            ecat.insert("failed_exchanges".into(), json!(s.failed));
            ecat.insert("partial_wkc_exchanges".into(), json!(s.partial));
            ecat.insert("robot_clock_backwards".into(), json!(s.backwards));
            ecat.insert("rt_problems".into(), json!(s.rt_problems));
            ecat.insert("guard_tripped".into(), json!(b.tripped));
        }
        frames = store.frames;
        cycles = store.cycles;
    }
    out.raw.insert("_clock".into(), json!(clock.iter().map(|(w, h)| [*w, *h]).collect::<Vec<_>>()));
    if frames.len() < 10 {
        say!("No data recorded.");
        return Ok(code);
    }
    let path = out.finish(py_args, &frames, &cycles, ecat)?;
    say!("Run folder: {} ({} frames)", path.display(), frames.len());
    if !a.no_analyze {
        if let Err(e) = br.import(&dir, true) {
            say!("Python import / analysis failed: {e:#}\n  (later: {} {} import {})", br.python, br.script.display(), dir.display());
        }
    }
    if !back {
        say!("{}", lifecycle::MANUAL_RESET);
    }
    Ok(code)
}

/// What runs on the orchestrator while the bus thread cycles: preparation, then the tests.
fn tests_body(
    r: &mut Robot,
    prep: &Prepare,
    out: &mut RunOut,
    info: &Value,
    a: &RunArgs,
    br: &Bridge,
    py_args: &Map<String, Value>,
) -> Result<()> {
    let q = r.prepare(prep)?;
    out.raw.insert("_robot_info".into(), info.clone());
    suite::run_all(r, a, out, br, py_args, q)
}

/// A few dozen input images while nothing moves (byte order detection, same path as the robot).
fn sample_inputs(bus: &mut dyn Fieldbus, cycle: f64) -> Vec<[u8; IN_LEN]> {
    let zero = [0u8; OUT_LEN];
    let mut ib = [0u8; IN_LEN];
    let mut v = Vec::with_capacity(160);
    for _ in 0..150 {
        if let Ok(w) = bus.exchange(&zero, &mut ib) {
            if w > 0 {
                v.push(ib);
            }
        }
        rt::sleep_s(cycle);
    }
    v
}
