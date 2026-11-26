//! `run` (supervisor of the test process), `probe`, `restore`, `status`, `burn`.
//!
//! The supervisor survives what the worker cannot handle itself (SIGKILL, abort, OOM, stack
//! overflow, Ctrl-\): if the worker did not report the robot back on TCP/IP, it checks port 10000
//! and runs `restore` itself.

use std::process::{Command as Proc, ExitCode};
use std::time::Duration;

use crate::bus::{run_sync, Controller, Core, Cycle, Feedback, Flow};
use crate::cli::{EcatOpts, ProbeArgs, RestoreArgs};
use crate::controllers::ExitSeq;
use crate::into_any;
use crate::lifecycle::{self, SessionOpts};
use crate::pdo::{Codec, Outputs};
use crate::{rt, say, tcp};

#[cfg(feature = "suite")]
pub fn run(a: crate::cli::RunArgs) -> ExitCode {
    use crate::worker;
    if a.sim {
        return worker::run(a); // nothing to restore in simulation
    }
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            say!("cannot find my own executable: {e}");
            return ExitCode::from(1);
        }
    };
    // same arguments, `run` -> `worker`
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if let Some(p) = args.iter().position(|x| x == "run") {
        args[p] = "worker".into();
    }
    let mut child = match Proc::new(&exe).args(&args).spawn() {
        Ok(c) => c,
        Err(e) => {
            say!("cannot start the test process: {e}");
            return ExitCode::from(1);
        }
    };
    // after the spawn: the worker keeps the default dispositions (Ctrl-\ still kills it)
    if let Err(e) = rt::install_supervisor_signals() {
        say!("WARNING: {e}");
    }
    let st = match child.wait() {
        Ok(s) => s,
        Err(e) => {
            say!("waiting for the test process: {e}");
            return check_and_restore(&a.ecat);
        }
    };
    match st.code() {
        Some(c) if c == worker::BACK_OK as i32 || c == worker::BACK_AFTER_ERROR as i32 || c == worker::NOT_SWITCHED as i32 => {
            ExitCode::from(c as u8)
        }
        _ => {
            say!("\nThe test process ended without the robot confirmed on TCP/IP ({st}): supervisor check");
            check_and_restore(&a.ecat)
        }
    }
}

/// TCP/IP answers? Else restore (fresh process, so the main PDU storage is free).
fn check_and_restore(o: &EcatOpts) -> ExitCode {
    if let Some(w) = tcp::alive(&o.ip, Duration::from_secs(3)) {
        say!("TCP/IP answers on {}: {w}", o.ip);
        return ExitCode::from(0);
    }
    match lifecycle::restore(o, SessionOpts { sdo_if_no_op: true, ..Default::default() }) {
        Ok(true) => ExitCode::from(0),
        Ok(false) | Err(_) => {
            say!("{}", lifecycle::MANUAL_RESET);
            ExitCode::from(10)
        }
    }
}

pub fn restore_cmd(r: RestoreArgs) -> ExitCode {
    let _ = rt::install_stop_signals();
    match lifecycle::restore(&r.ecat, SessionOpts { sdo_disable: r.sdo, sdo_if_no_op: true, ..Default::default() }) {
        Ok(true) => ExitCode::from(0),
        Ok(false) => {
            say!("{}", lifecycle::MANUAL_RESET);
            ExitCode::from(10)
        }
        Err(e) => {
            say!("restore: {e:#}\n{}", lifecycle::MANUAL_RESET);
            ExitCode::from(10)
        }
    }
}

pub fn status_cmd(o: EcatOpts) -> ExitCode {
    match lifecycle::status(&o) {
        Ok(()) => ExitCode::from(0),
        Err(e) => {
            say!("status: {e:#}");
            ExitCode::from(1)
        }
    }
}

pub fn burn(secs: f64) -> ExitCode {
    rt::die_with_parent();
    let t_end = rt::now() + secs;
    let mut x = 0u64;
    while rt::now() < t_end {
        for _ in 0..10_000 {
            x = std::hint::black_box(x.wrapping_add(1));
        }
    }
    ExitCode::from(0)
}

/// Shows the decoded feedback twice a second, outputs left at zero (no command, no motion).
struct Watch {
    secs: f64,
    t0: Option<f64>,
    last_print: f64,
    cycles: u64,
    fresh: u64,
    failed: u64,
    first: Option<(f64, f64)>,
    last: Option<(f64, f64)>,
}

impl Controller for Watch {
    fn name(&self) -> &'static str {
        "watch"
    }

    fn step(&mut self, cx: &Cycle, fb: &Feedback, out: &mut Outputs) -> Flow {
        let t0 = *self.t0.get_or_insert(cx.ts);
        out.zero_velocity();
        self.cycles += 1;
        if !fb.ok {
            self.failed += 1;
        }
        if fb.fresh {
            self.fresh += 1;
            let p = (cx.ts, fb.inp.robot_time());
            self.first.get_or_insert(p);
            self.last = Some(p);
        }
        if cx.ts - self.last_print >= 0.5 {
            self.last_print = cx.ts;
            let i = &fb.inp;
            say!("  {}", i.status_line());
            say!(
                "    joints {}  slots {:?} -> {}",
                crate::robot::fmt6(&i.joint_set),
                i.dyn_type,
                i.dyn_val.iter().map(crate::robot::fmt6).collect::<Vec<_>>().join(" ")
            );
        }
        if cx.ts - t0 >= self.secs {
            return Flow::Done;
        }
        Flow::Continue
    }

    into_any!();
}

pub fn probe(p: ProbeArgs) -> ExitCode {
    let _ = rt::install_stop_signals();
    let o = &p.ecat;
    if let Err(e) = lifecycle::enter(o) {
        say!("Not switched to EtherCAT: {e:#}");
        return ExitCode::from(3);
    }
    let cycle = o.cycle_s();
    let r = lifecycle::ecat_session(o, SessionOpts::default(), |ctx| {
        say!("Slave: {}", serde_json::Value::Object(ctx.info.clone()));
        let mut core = Core::new(Codec::new(ctx.order), false, 5, cycle, o.slots());
        let mut fb = (ctx.make)();
        let mut w = Watch { secs: p.seconds, t0: None, last_print: 0.0, cycles: 0, fresh: 0, failed: 0, first: None, last: None };
        run_sync(&mut core, &mut *fb, &mut w, p.seconds + 2.0);
        let rate = match (w.first, w.last) {
            (Some(a), Some(b)) if b.0 > a.0 => format!("{:.6}", (b.1 - a.1) / (b.0 - a.0)),
            _ => "?".into(),
        };
        say!(
            "Probe: {} cycles, new robot sample in {:.1} %, {} failed exchanges, robot clock / host clock = {rate}, \
             byte order {} {}",
            w.cycles,
            100.0 * w.fresh as f64 / w.cycles.max(1) as f64,
            w.failed,
            ctx.order.name(),
            if core.order_confirmed { "(confirmed: the dynamic slot types are echoed)" } else { "(NOT confirmed by the slot echo)" }
        );
        let mut ex = ExitSeq::new();
        run_sync(&mut core, &mut *fb, &mut ex, 40.0);
        say!("Exit sequence: {}", ex.summary());
        Ok(())
    });
    if let Err(e) = r {
        say!("probe: {e:#}");
    }
    if lifecycle::come_back(o, true, true) {
        ExitCode::from(0)
    } else {
        ExitCode::from(10)
    }
}
