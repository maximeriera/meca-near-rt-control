//! Platform layer: the host clock, real-time scheduling, memory locking and stop signals.
//! Linux (PREEMPT_RT) is the target. Elsewhere the functions fall back to plain std so that the
//! simulator (`--sim`) and the unit tests still run.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Seconds on CLOCK_MONOTONIC. The single host clock of a run: frames, command logs, bus cycles,
/// clock pairs (the Python analysis needs one clock for all of them).
#[cfg(target_os = "linux")]
pub fn now() -> f64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: valid pointer to a timespec.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as f64 + ts.tv_nsec as f64 * 1e-9
}

#[cfg(not(target_os = "linux"))]
pub fn now() -> f64 {
    use std::sync::OnceLock;
    static T0: OnceLock<std::time::Instant> = OnceLock::new();
    T0.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() + 1000.0
}

/// Sleep until the absolute host time `t` [s] (clock_nanosleep TIMER_ABSTIME on Linux).
#[cfg(target_os = "linux")]
pub fn sleep_until(t: f64) {
    if !(t.is_finite()) {
        return;
    }
    let ts = libc::timespec { tv_sec: t.floor() as libc::time_t, tv_nsec: ((t - t.floor()) * 1e9) as libc::c_long };
    loop {
        // SAFETY: valid timespec, null remainder (absolute sleep).
        let r = unsafe {
            libc::clock_nanosleep(libc::CLOCK_MONOTONIC, libc::TIMER_ABSTIME, &ts, std::ptr::null_mut())
        };
        if r != libc::EINTR {
            break;
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn sleep_until(t: f64) {
    let dt = t - now();
    if dt > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(dt));
    }
}

pub fn sleep_s(s: f64) {
    if s > 0.0 {
        std::thread::sleep(Duration::from_secs_f64(s));
    }
}

/// (unix wall time, host time) read as close together as possible: the tightest of n tries.
/// Same as clock_pair() of meca_latency_test.py (netcap.py uses it).
pub fn clock_pair() -> (f64, f64) {
    let mut best = (0.0, 0.0, f64::INFINITY);
    for _ in 0..20 {
        let a = now();
        let w = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let b = now();
        if b - a < best.2 {
            best = (w, 0.5 * (a + b), b - a);
        }
    }
    (best.0, best.1)
}

/// SCHED_FIFO at `prio` for the calling thread.
#[cfg(target_os = "linux")]
pub fn set_realtime(prio: i32) -> Result<(), String> {
    let p = libc::sched_param { sched_priority: prio };
    // SAFETY: valid sched_param, current thread.
    let r = unsafe { libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &p) };
    if r == 0 {
        Ok(())
    } else {
        Err(format!("SCHED_FIFO {prio} refused (errno {r}: cap_sys_nice / rtprio limit?)"))
    }
}

#[cfg(not(target_os = "linux"))]
pub fn set_realtime(_prio: i32) -> Result<(), String> {
    Err("real-time scheduling is only implemented on Linux".into())
}

/// Pin the calling thread to one CPU.
#[cfg(target_os = "linux")]
pub fn pin_cpu(cpu: usize) -> Result<(), String> {
    // SAFETY: cpu_set_t is plain data, zero-initialised then filled with CPU_SET.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) == 0 {
            Ok(())
        } else {
            Err(format!("cannot pin to CPU {cpu}: {}", std::io::Error::last_os_error()))
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_cpu(_cpu: usize) -> Result<(), String> {
    Err("CPU pinning is only implemented on Linux".into())
}

/// Lock current and future pages (no page fault in the cycle).
#[cfg(target_os = "linux")]
pub fn lock_memory() -> Result<(), String> {
    // SAFETY: plain syscall.
    if unsafe { libc::mlockall(libc::MCL_CURRENT | libc::MCL_FUTURE) } == 0 {
        Ok(())
    } else {
        Err(format!("mlockall: {} (cap_ipc_lock / memlock limit?)", std::io::Error::last_os_error()))
    }
}

#[cfg(not(target_os = "linux"))]
pub fn lock_memory() -> Result<(), String> {
    Err("mlockall is only implemented on Linux".into())
}

/// Apply priority and CPU to the calling thread; returns what failed (empty = all fine).
pub fn make_realtime(prio: i32, cpu: Option<usize>) -> Vec<String> {
    let mut bad = Vec::new();
    if prio > 0 {
        if let Err(e) = set_realtime(prio) {
            bad.push(e);
        }
    }
    if let Some(c) = cpu {
        if let Err(e) = pin_cpu(c) {
            bad.push(e);
        }
    }
    bad
}

/// Does SCHED_FIFO work for this process? (checked in a throw-away thread)
pub fn can_realtime(prio: i32) -> Result<(), String> {
    std::thread::spawn(move || set_realtime(prio)).join().unwrap_or_else(|_| Err("thread panicked".into()))
}

// --------------------------------------------------------------------------------------------- //
// Stop signals                                                                                  //
// --------------------------------------------------------------------------------------------- //

/// Number of stop requests (Ctrl-C, SIGTERM, SIGHUP) received. 1: the running test motion is
/// aborted at once (velocity zero, queued motion cleared), the completed tests are kept, then the
/// exit sequence. 2: the bus thread starts the exit sequence itself, right away.
/// The exit sequence itself (Deactivate + DisableEtherCAT) is never skipped: only Ctrl-\ (SIGQUIT,
/// default action) kills the worker, and the supervisor then runs `restore`.
pub static STOP: AtomicUsize = AtomicUsize::new(0);

pub fn stop_requests() -> usize {
    STOP.load(Ordering::SeqCst)
}

/// Install the stop handlers of the worker (counting).
pub fn install_stop_signals() -> Result<(), String> {
    #[cfg(unix)]
    let sigs = [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM, signal_hook::consts::SIGHUP];
    #[cfg(not(unix))]
    let sigs = [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM];
    for s in sigs {
        // SAFETY: the handler only touches an atomic (async-signal-safe).
        unsafe {
            signal_hook::low_level::register(s, || {
                STOP.fetch_add(1, Ordering::SeqCst);
            })
        }
        .map_err(|e| format!("signal {s}: {e}"))?;
    }
    Ok(())
}

/// The supervisor survives the signals that stop the worker, so it can run `restore` after it.
/// Handlers (not SIG_IGN) so that the worker it spawns gets default dispositions back at exec.
pub fn install_supervisor_signals() -> Result<(), String> {
    #[cfg(unix)]
    let sigs = [
        signal_hook::consts::SIGINT,
        signal_hook::consts::SIGTERM,
        signal_hook::consts::SIGHUP,
        signal_hook::consts::SIGQUIT,
    ];
    #[cfg(not(unix))]
    let sigs = [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM];
    for s in sigs {
        // SAFETY: empty handler.
        unsafe { signal_hook::low_level::register(s, || {}) }.map_err(|e| format!("signal {s}: {e}"))?;
    }
    Ok(())
}

/// In a `burn` child: die with the parent (no stray busy processes after a crash).
pub fn die_with_parent() {
    #[cfg(target_os = "linux")]
    // SAFETY: plain prctl.
    unsafe {
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
        if libc::getppid() == 1 {
            libc::_exit(0);
        }
    }
}

pub fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2)
}
