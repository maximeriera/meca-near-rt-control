//! Minimal client of the Meca500 TCP/IP control port (10000): the welcome message, the identity
//! (serial, firmware, product) before switching, SwitchToEtherCAT, and the check that the port
//! answers again after DisableEtherCAT (the MecaPortal uses the same TCP/IP stack).
//! Messages are ASCII, "[code][text]", terminated by a NUL byte.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::rt;

pub const PORT: u16 = 10000;
pub const CONNECTED: u32 = 3000;
pub const USER_ALREADY: u32 = 3001;

pub struct Meca {
    s: TcpStream,
    buf: Vec<u8>,
}

fn addr(ip: &str) -> Result<SocketAddr> {
    (ip, PORT).to_socket_addrs()?.next().ok_or_else(|| anyhow!("bad address {ip}"))
}

fn parse(msg: &str) -> Option<(u32, String)> {
    let a = msg.find('[')?;
    let b = msg[a..].find(']')? + a;
    let code: u32 = msg[a + 1..b].trim().parse().ok()?;
    let rest = &msg[b + 1..];
    let text = match (rest.find('['), rest.rfind(']')) {
        (Some(x), Some(y)) if y > x => rest[x + 1..y].to_string(),
        _ => rest.to_string(),
    };
    Some((code, text))
}

impl Meca {
    /// Connect and read the welcome message: (connection, code, text).
    pub fn connect(ip: &str, timeout: Duration) -> Result<(Meca, u32, String)> {
        let s = TcpStream::connect_timeout(&addr(ip)?, timeout).with_context(|| format!("TCP {ip}:{PORT}"))?;
        s.set_nodelay(true)?;
        let mut m = Meca { s, buf: Vec::new() };
        let (code, text) = m.read_msg(timeout)?;
        Ok((m, code, text))
    }

    fn read_msg(&mut self, timeout: Duration) -> Result<(u32, String)> {
        let t_end = rt::now() + timeout.as_secs_f64();
        loop {
            if let Some(p) = self.buf.iter().position(|&b| b == 0) {
                let raw: Vec<u8> = self.buf.drain(..=p).collect();
                let msg = String::from_utf8_lossy(&raw[..raw.len() - 1]).to_string();
                return parse(&msg).ok_or_else(|| anyhow!("unexpected message {msg:?}"));
            }
            let left = t_end - rt::now();
            if left <= 0.0 {
                bail!("no message within {:?}", timeout);
            }
            self.s.set_read_timeout(Some(Duration::from_secs_f64(left.max(0.001))))?;
            let mut chunk = [0u8; 1024];
            let n = self.s.read(&mut chunk)?;
            if n == 0 {
                bail!("connection closed by the robot");
            }
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }

    pub fn send(&mut self, cmd: &str) -> Result<()> {
        self.s.write_all(cmd.as_bytes())?;
        self.s.write_all(&[0])?;
        Ok(())
    }

    /// Send a command and wait for one of the expected codes (other messages are skipped).
    pub fn command(&mut self, cmd: &str, expect: &[u32], timeout: Duration) -> Result<(u32, String)> {
        self.send(cmd)?;
        let t_end = rt::now() + timeout.as_secs_f64();
        loop {
            let left = (t_end - rt::now()).max(0.0);
            let (c, t) = self.read_msg(Duration::from_secs_f64(left))?;
            if expect.contains(&c) {
                return Ok((c, t));
            }
            if (1000..2000).contains(&c) {
                bail!("{cmd}: robot error [{c}][{t}]");
            }
        }
    }
}

/// Does the TCP/IP control port answer (welcome [3000], or [3001] when another client, e.g. the
/// MecaPortal, holds the control connection)? Returns the welcome text.
pub fn alive(ip: &str, timeout: Duration) -> Option<String> {
    match Meca::connect(ip, timeout) {
        Ok((_, c, t)) if c == CONNECTED || c == USER_ALREADY => Some(format!("[{c}][{t}]")),
        _ => None,
    }
}

/// Poll `alive` every second until it answers or `wait_s` is over.
pub fn wait_alive(ip: &str, wait_s: f64) -> Option<(String, f64)> {
    let t0 = rt::now();
    loop {
        if let Some(w) = alive(ip, Duration::from_secs(2)) {
            return Some((w, rt::now() - t0));
        }
        if rt::now() - t0 > wait_s || rt::stop_requests() >= 3 {
            return None;
        }
        rt::sleep_s(1.0);
    }
}

/// Identity as in meca_latency_test.py's _robot_info.identity (keys read by aggregate.py).
pub fn identity(m: &mut Meca, welcome: &str, ip: &str) -> Map<String, Value> {
    let mut id = Map::new();
    let t = Duration::from_secs(3);
    let product = m.command("GetProductType", &[2084], t).map(|x| x.1).ok();
    let fw = m.command("GetFwVersionFull", &[2082], t).map(|x| x.1).ok();
    let serial = m.command("GetRobotSerial", &[2083], t).map(|x| x.1).ok();
    let rev = welcome
        .split(|c: char| c == ' ' || c == '_')
        .find_map(|w| w.strip_prefix('R').and_then(|r| r.parse::<u32>().ok()));
    let model = product.clone().or_else(|| welcome.split_whitespace().find(|w| w.starts_with("Meca")).map(String::from));
    id.insert("model".into(), json!(model));
    id.insert("revision".into(), json!(rev));
    id.insert("version".into(), json!(fw.map(|v| v.trim_start_matches('v').to_string())));
    id.insert("serial".into(), json!(serial));
    id.insert("ip_address".into(), json!(ip));
    id.insert("welcome".into(), json!(welcome));
    id
}

/// Read the identity, then SwitchToEtherCAT. The robot closes the connection when it switches.
pub fn switch_to_ethercat(ip: &str) -> Result<Map<String, Value>> {
    let (mut m, code, welcome) = Meca::connect(ip, Duration::from_secs(3))?;
    if code == USER_ALREADY {
        bail!("another client holds the robot's control port (close the MecaPortal / other scripts): [{code}][{welcome}]");
    }
    if code != CONNECTED {
        bail!("unexpected welcome [{code}][{welcome}]");
    }
    let id = identity(&mut m, &welcome, ip);
    m.send("SwitchToEtherCAT")?;
    match m.read_msg(Duration::from_secs(5)) {
        Ok((c, t)) => crate::say!("  SwitchToEtherCAT -> [{c}][{t}]"),
        Err(e) => crate::say!("  SwitchToEtherCAT sent ({e})"),
    }
    Ok(id)
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn parse_messages() {
        assert_eq!(parse("[3000][Connected to Meca500 R4 v11.3.3.]"), Some((3000, "Connected to Meca500 R4 v11.3.3.".into())));
        assert_eq!(parse("[2083][M500-2227]"), Some((2083, "M500-2227".into())));
        assert_eq!(parse("garbage"), None);
    }
}
