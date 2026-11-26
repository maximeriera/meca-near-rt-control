//! EtherCAT life cycle of the robot: checks before switching, SwitchToEtherCAT over TCP, scan and
//! OP with ethercrab, byte order detection, and the way back (exit sequence, then TCP port 10000
//! must answer; otherwise `restore`: a fresh master, INIT -> OP, Deactivate, DisableEtherCAT;
//! SDO write of 0x7200:07 as a fallback; then the manual network reset).
//!
//! The ethercrab group type is never named: the exchange closure is built by a macro where the
//! group is in scope, and handed out as `Box<dyn Fieldbus>` by a factory (EcatCtx::make), so the
//! exit sequence can still run on the main thread if the bus thread died.

use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::bus::{run_sync, Core};
use crate::cli::EcatOpts;
use crate::controllers::ExitSeq;
use crate::fieldbus::{BusError, Fieldbus};
use crate::pdo::{detect_order, ByteOrder, Codec, Inputs, IN_LEN, OUT_LEN, PRODUCT_CODE, VENDOR_ID};
use crate::{rt, say, tcp};

/// What `work` gets inside an EtherCAT session (robot in OP).
pub struct EcatCtx<'a> {
    /// A new process data exchange on the robot (one per bus thread; another one for the
    /// fallback exit sequence if that thread died).
    pub make: &'a dyn Fn() -> Box<dyn Fieldbus + 'a>,
    pub order: ByteOrder,
    pub order_detected: bool,
    pub info: Map<String, Value>,
}

#[derive(Clone, Copy, Default)]
pub struct SessionOpts {
    /// Use the second PDU storage (a fresh master in a process that already had one).
    pub rescue: bool,
    /// Write DisableEtherCAT over CoE (SDO 0x7200:07) in PRE-OP instead of cycling in OP.
    pub sdo_disable: bool,
    /// If OP cannot be reached, still try the SDO write.
    pub sdo_if_no_op: bool,
    /// Only scan and print the robot's EtherCAT state (status).
    pub scan_only: bool,
}

pub fn boxed<'a, F: Fieldbus + 'a>(f: F) -> Box<dyn Fieldbus + 'a> {
    Box::new(f)
}

// ------------------------------------------------------------------------------------------- //
// Before switching                                                                             //
// ------------------------------------------------------------------------------------------- //

/// Everything that can be checked while the robot is still reachable over TCP.
#[cfg(target_os = "linux")]
pub fn preflight(o: &EcatOpts) -> Result<()> {
    let iface = o.iface()?;
    let sys = format!("/sys/class/net/{iface}");
    if !std::path::Path::new(&sys).exists() {
        bail!("no network interface '{iface}' (see `ip link`)");
    }
    let carrier = std::fs::read_to_string(format!("{sys}/carrier")).unwrap_or_default();
    if carrier.trim() != "1" {
        say!("WARNING: no link on {iface}: the cable must be on the robot's ETHERNET1 port (EtherCAT IN)");
    }
    // raw socket (cap_net_raw), what ethercrab needs
    // SAFETY: plain socket / close syscalls.
    let fd = unsafe { libc::socket(libc::AF_PACKET, libc::SOCK_RAW, (libc::ETH_P_ALL as u16).to_be() as i32) };
    if fd < 0 {
        bail!(
            "cannot open a raw socket ({}): run ecat/setcap.sh on the binary (cap_net_raw) or run it as root",
            std::io::Error::last_os_error()
        );
    }
    // SAFETY: fd is ours.
    unsafe { libc::close(fd) };
    rt::can_realtime(o.rt_prio).map_err(|e| anyhow!("{e} (ecat/setcap.sh gives cap_sys_nice)"))?;
    if !o.no_route_check {
        let out = std::process::Command::new("ip").args(["-o", "route", "get", &o.ip]).output();
        match out {
            Ok(x) if x.status.success() => {
                let s = String::from_utf8_lossy(&x.stdout);
                let dev = s.split_whitespace().skip_while(|w| *w != "dev").nth(1).unwrap_or("?");
                if dev != iface {
                    bail!(
                        "{} is reached through '{dev}', not '{iface}': give {iface} an address in the robot's subnet \
                         (after the test the robot comes back on TCP/IP through this port), or --no-route-check",
                        o.ip
                    );
                }
            }
            _ => say!("WARNING: could not check the route to {} (`ip route get`)", o.ip),
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn preflight(_o: &EcatOpts) -> Result<()> {
    bail!("the EtherCAT master runs on Linux only (use --sim here)")
}

/// Checks, then the identity and SwitchToEtherCAT over TCP. None: TCP did not answer (the robot
/// is probably still in EtherCAT mode from a previous run: the scan will tell).
pub fn enter(o: &EcatOpts) -> Result<Option<Map<String, Value>>> {
    preflight(o)?;
    if tcp::alive(&o.ip, Duration::from_secs(3)).is_none() {
        say!("No TCP/IP answer from {}: assuming the robot is already in EtherCAT mode", o.ip);
        return Ok(None);
    }
    say!("Robot answers on {}:{}: identity, then SwitchToEtherCAT", o.ip, tcp::PORT);
    let id = tcp::switch_to_ethercat(&o.ip)?;
    let v = id.get("version").and_then(Value::as_str).unwrap_or("?").to_string();
    say!(
        "  {} R{} serial {}, firmware {v}",
        id.get("model").and_then(Value::as_str).unwrap_or("?"),
        id.get("revision").map(|x| x.to_string()).unwrap_or_default(),
        id.get("serial").and_then(Value::as_str).unwrap_or("?")
    );
    if !v.starts_with("11.3.3") {
        say!("  WARNING: firmware {v}: 11.3.3 fixes a high latency of velocity commands at 1 kHz over EtherCAT");
    }
    Ok(Some(id))
}

// ------------------------------------------------------------------------------------------- //
// The ethercrab session                                                                        //
// ------------------------------------------------------------------------------------------- //

#[cfg(target_os = "linux")]
mod imp {
    use super::*;
    use ethercrab::std::{ethercat_now, tx_rx_task};
    use ethercrab::{MainDevice, MainDeviceConfig, PduStorage, Timeouts};

    const MAX_SD: usize = 2;
    const PDI_LEN: usize = 512;
    const MAX_FRAMES: usize = 16;
    const MAX_PDU_DATA: usize = PduStorage::element_size(1100);
    type Storage = PduStorage<MAX_FRAMES, MAX_PDU_DATA>;
    // Two storages: a PduStorage can be split only once per process; the second one is for a
    // fresh master in the same process (in-process restore).
    static PDU_MAIN: Storage = PduStorage::new();
    static PDU_RESCUE: Storage = PduStorage::new();

    fn ec(e: ethercrab::error::Error) -> anyhow::Error {
        anyhow!("{e:?}")
    }

    fn be(e: ethercrab::error::Error) -> BusError {
        match e {
            ethercrab::error::Error::Timeout(_) => BusError::Timeout,
            other => BusError::Other(format!("{other:?}")),
        }
    }

    /// The process data exchange as a closure over the group and the main device.
    macro_rules! ecat_exchange {
        ($g:expr, $md:expr) => {{
            let (g, md) = ($g, $md);
            move |out: &[u8; OUT_LEN], inp: &mut [u8; IN_LEN]| -> Result<u16, BusError> {
                {
                    let sd = g.subdevice(md, 0).map_err(be)?;
                    let mut o = sd.outputs_raw_mut();
                    o[..OUT_LEN].copy_from_slice(out);
                }
                let r = smol::block_on(g.tx_rx(md)).map_err(be)?;
                {
                    let sd = g.subdevice(md, 0).map_err(be)?;
                    let i = sd.inputs_raw();
                    inp.copy_from_slice(&i[..IN_LEN]);
                }
                Ok(r.working_counter)
            }
        }};
    }

    pub fn ecat_session<R>(
        o: &EcatOpts,
        so: SessionOpts,
        work: impl for<'a> FnOnce(&EcatCtx<'a>) -> Result<R>,
    ) -> Result<Option<R>> {
        let storage: &'static Storage = if so.rescue { &PDU_RESCUE } else { &PDU_MAIN };
        let (tx, rx, pdu_loop) =
            storage.try_split().map_err(|_| anyhow!("EtherCAT storage already used in this process"))?;
        let md = MainDevice::new(
            pdu_loop,
            Timeouts {
                state_transition: Duration::from_secs(10),
                pdu: Duration::from_millis(o.pdu_timeout_ms),
                wait_loop_delay: Duration::ZERO,
                mailbox_response: Duration::from_millis(1000),
                ..Timeouts::default()
            },
            MainDeviceConfig::default(),
        );
        let iface = o.iface()?;
        let task = tx_rx_task(&iface, tx, rx)
            .map_err(|e| anyhow!("cannot open {iface} for EtherCAT: {e:?} (ecat/setcap.sh: cap_net_raw)"))?;
        let (prio, cpu) = (o.txrx_prio, o.txrx_cpu);
        std::thread::Builder::new()
            .name("ecat-txrx".into())
            .spawn(move || {
                for b in rt::make_realtime(prio, cpu) {
                    say!("  tx/rx thread: {b}");
                }
                if let Err(e) = smol::block_on(task) {
                    say!("EtherCAT tx/rx task ended: {e:?}");
                }
            })
            .context("spawn the tx/rx thread")?;
        let res = inner(o, so, &md, &iface, work);
        // Stop the tx/rx task. Never joined: after DisableEtherCAT a lost frame can keep it busy.
        // SAFETY: the groups built on this main device are gone (inner returned).
        unsafe {
            let _ = md.release_all();
        }
        res
    }

    fn inner<R>(
        o: &EcatOpts,
        so: SessionOpts,
        md: &MainDevice<'static>,
        iface: &str,
        work: impl for<'a> FnOnce(&EcatCtx<'a>) -> Result<R>,
    ) -> Result<Option<R>> {
        say!("Scanning {iface} for the robot (up to {:.0} s)...", o.scan_timeout);
        let t_end = rt::now() + o.scan_timeout;
        let mut last = String::from("nothing found");
        let group = loop {
            match smol::block_on(md.init_single_group::<MAX_SD, PDI_LEN>(ethercat_now)) {
                Ok(g) if g.len() == 1 => break g,
                Ok(g) => {
                    if g.len() > 1 {
                        bail!("{} EtherCAT devices on {iface}: the robot must be alone on this port", g.len());
                    }
                }
                Err(e) => last = format!("{e:?}"),
            }
            if rt::now() > t_end || rt::stop_requests() >= 3 {
                bail!("no EtherCAT robot on {iface} after {:.0} s ({last})", o.scan_timeout);
            }
            rt::sleep_s(1.0);
        };
        let mut info = Map::new();
        {
            let sd = group.subdevice(md, 0).map_err(ec)?;
            let id = sd.identity();
            say!(
                "  {} vendor {:#x} product {:#x} rev {} serial {}",
                sd.name(),
                id.vendor_id,
                id.product_id,
                id.revision,
                id.serial
            );
            info.insert("name".into(), json!(sd.name()));
            info.insert("vendor_id".into(), json!(id.vendor_id));
            info.insert("product_id".into(), json!(id.product_id));
            info.insert("revision".into(), json!(id.revision));
            info.insert("ecat_serial".into(), json!(id.serial));
            if let Ok((state, code)) = smol::block_on(sd.status()) {
                say!("  EtherCAT state {state:?}, AL status {code:?}");
            }
            if id.vendor_id != VENDOR_ID || id.product_id != PRODUCT_CODE {
                bail!("{} is not a Meca500 (vendor {:#x}, product {:#x})", sd.name(), id.vendor_id, id.product_id);
            }
            if so.scan_only {
                return Ok(None);
            }
            if so.sdo_disable {
                say!("  SDO write DisableEtherCAT (0x7200:07 = 1) in PRE-OP");
                smol::block_on(sd.sdo_write(0x7200, 7u8, 1u8)).map_err(ec).context("SDO 0x7200:07")?;
                return Ok(None);
            }
        }
        let group = smol::block_on(group.into_safe_op(md)).map_err(ec).context("PRE-OP -> SAFE-OP")?;
        let (ilen, olen) = {
            let sd = group.subdevice(md, 0).map_err(ec)?;
            let io = sd.io_raw();
            (io.inputs().len(), io.outputs().len())
        };
        info.insert("in_bytes".into(), json!(ilen));
        info.insert("out_bytes".into(), json!(olen));
        if ilen < IN_LEN || olen < OUT_LEN {
            bail!("process image of {olen} out / {ilen} in bytes, expected {OUT_LEN} / {IN_LEN} (ESI 11.3)");
        }
        let group = smol::block_on(group.request_into_op(md)).map_err(ec).context("SAFE-OP -> OP")?;
        // cyclic zero outputs until the robot is in OP (the SM watchdog wants process data)
        let t_end = rt::now() + 10.0;
        let mut last = String::new();
        loop {
            {
                let sd = group.subdevice(md, 0).map_err(ec)?;
                let mut out = sd.outputs_raw_mut();
                for b in out.iter_mut() {
                    *b = 0;
                }
            }
            match smol::block_on(group.tx_rx(md)) {
                Ok(r) if r.all_op() => break,
                Ok(r) => last = format!("states {:?}", r.subdevice_states),
                Err(e) => last = format!("{e:?}"),
            }
            if rt::now() > t_end {
                if so.sdo_if_no_op {
                    let sd = group.subdevice(md, 0).map_err(ec)?;
                    say!("  no OP: SDO write DisableEtherCAT (0x7200:07 = 1)");
                    let _ = smol::block_on(sd.sdo_write(0x7200, 7u8, 1u8));
                }
                bail!("the robot did not reach OP in 10 s ({last})");
            }
            rt::sleep_s(o.cycle_s());
        }
        say!("  robot in OP");
        let gr = &group;
        let make = move || boxed(crate::fieldbus::FnBus(ecat_exchange!(gr, md)));
        // byte order, from ~150 cycles of inputs while nothing moves
        let (order, detected) = {
            let mut fb = make();
            let zero = [0u8; OUT_LEN];
            let mut ib = [0u8; IN_LEN];
            let mut samples = Vec::with_capacity(160);
            for _ in 0..150 {
                if let Ok(w) = fb.exchange(&zero, &mut ib) {
                    if w > 0 {
                        samples.push(ib);
                    }
                }
                rt::sleep_s(o.cycle_s());
            }
            decide_order(o, &samples)
        };
        info.insert("byte_order".into(), json!(order.name()));
        info.insert("byte_order_detected".into(), json!(detected));
        let r = work(&EcatCtx { make: &make, order, order_detected: detected, info })?;
        Ok(Some(r))
    }
}

/// --byte-order given, or detected from the samples (None of them plausible: little-endian with
/// DisableEtherCAT mirrored, the caller then only runs the exit sequence).
pub fn decide_order(o: &EcatOpts, samples: &[[u8; IN_LEN]]) -> (ByteOrder, bool) {
    match o.byte_order.as_str() {
        "le" => (ByteOrder::Le, false),
        "be" => (ByteOrder::Be, false),
        _ => match detect_order(samples) {
            Ok(b) => {
                let i = Inputs::decode(&Codec::new(b), samples.last().unwrap_or(&[0; IN_LEN]));
                say!("  byte order {} (robot clock {:.3} s, J1..J6 {:?})", b.name(), i.robot_time(), i.joint_set);
                (b, true)
            }
            Err(e) => {
                say!("  byte order NOT detected: {e}");
                (ByteOrder::Le, false)
            }
        },
    }
}

#[cfg(target_os = "linux")]
pub use imp::ecat_session;

#[cfg(not(target_os = "linux"))]
pub fn ecat_session<R>(
    _o: &EcatOpts,
    _so: SessionOpts,
    _work: impl for<'a> FnOnce(&EcatCtx<'a>) -> Result<R>,
) -> Result<Option<R>> {
    bail!("the EtherCAT master runs on Linux only (use --sim here)")
}

// ------------------------------------------------------------------------------------------- //
// The way back                                                                                 //
// ------------------------------------------------------------------------------------------- //

/// The exit sequence on the calling thread, on a fresh exchange (no bus thread).
pub fn exit_on(ctx: &EcatCtx<'_>, cycle_s: f64) -> ExitSeq {
    let mut core = Core::new(Codec::new(ctx.order), false, 5, cycle_s, [0; 4]);
    let mut fb = (ctx.make)();
    let mut ex = ExitSeq::new();
    run_sync(&mut core, &mut *fb, &mut ex, 40.0);
    ex
}

pub const MANUAL_RESET: &str = "\
The robot did not come back on TCP/IP. Try, in this order:
  1. meca_ecat restore --iface <IF>            (fresh master: OP, Deactivate, DisableEtherCAT)
  2. meca_ecat restore --iface <IF> --sdo      (DisableEtherCAT written over CoE in PRE-OP)
  3. power-cycle the robot: EtherCAT mode is persistent, so this alone does not bring TCP/IP back
  4. network configuration reset (Meca500 user manual, troubleshooting): power off; on the R4 engage
     the E-stop; hold the base Power button while powering on, ~40 s, until the red and yellow LEDs
     blink in sequence. IP 192.168.0.100, TCP/IP mode. Programs are kept.";

/// Wait for TCP port 10000; if it does not answer, restore with a fresh master (in this process:
/// second storage). Returns true once TCP/IP answers.
pub fn come_back(o: &EcatOpts, rescue: bool) -> bool {
    say!("Waiting for TCP/IP on {}:{} (up to {:.0} s)...", o.ip, tcp::PORT, o.tcp_wait);
    if let Some((w, t)) = tcp::wait_alive(&o.ip, o.tcp_wait) {
        say!("TCP/IP back after {t:.1} s: {w}  -> MecaPortal: http://{}", o.ip);
        return true;
    }
    say!("No TCP/IP answer: restore with a fresh EtherCAT master");
    match restore(o, SessionOpts { rescue, sdo_if_no_op: true, ..Default::default() }) {
        Ok(true) => true,
        Ok(false) | Err(_) => {
            say!("{MANUAL_RESET}");
            false
        }
    }
}

/// `restore`: nothing if TCP answers; else a fresh master, OP, exit sequence (or the SDO write),
/// then wait for TCP. Ok(true) once TCP/IP answers.
pub fn restore(o: &EcatOpts, so: SessionOpts) -> Result<bool> {
    if let Some(w) = tcp::alive(&o.ip, Duration::from_secs(3)) {
        say!("TCP/IP already answers on {}: {w}", o.ip);
        return Ok(true);
    }
    let cycle = o.cycle_s();
    let r = ecat_session(o, so, |ctx| {
        if !ctx.order_detected {
            say!("  (byte order not confirmed: DisableEtherCAT is written in both byte orders)");
        }
        let ex = exit_on(ctx, cycle);
        say!("  exit sequence: {}", ex.summary());
        Ok(())
    });
    if let Err(e) = &r {
        say!("restore: {e:#}");
    }
    match tcp::wait_alive(&o.ip, o.tcp_wait) {
        Some((w, t)) => {
            say!("TCP/IP back after {t:.1} s: {w}  -> MecaPortal: http://{}", o.ip);
            Ok(true)
        }
        None => {
            say!("TCP/IP still silent after {:.0} s", o.tcp_wait);
            Ok(false)
        }
    }
}

/// `status`: TCP/IP or EtherCAT? (the scan puts the robot's EtherCAT state machine in PRE-OP)
pub fn status(o: &EcatOpts) -> Result<()> {
    if let Some(w) = tcp::alive(&o.ip, Duration::from_secs(3)) {
        say!("TCP/IP mode: {}:{} answers {w}", o.ip, tcp::PORT);
        return Ok(());
    }
    say!("No TCP/IP answer from {}: looking for the robot on EtherCAT", o.ip);
    o.iface()?;
    ecat_session(o, SessionOpts { scan_only: true, ..Default::default() }, |_| Ok(()))?;
    Ok(())
}
