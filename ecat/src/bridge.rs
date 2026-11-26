//! Calls to ecat_bridge.py, between tests only (never from the bus thread): the quick model of
//! the step test, the replay / soak references (same numpy RNG as the TCP runs), and the import +
//! analysis of the finished run.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::rt;

#[derive(Clone, Debug)]
pub struct Bridge {
    pub python: String,
    pub script: PathBuf,
}

/// Default location of ecat_bridge.py: the repository root, next to this crate.
pub fn default_script() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("ecat_bridge.py")
}

impl Bridge {
    fn run(&self, args: &[&str], timeout: f64, capture: bool) -> Result<String> {
        let mut c = Command::new(&self.python);
        c.arg(&self.script).args(args);
        if capture {
            c.stdout(Stdio::piped());
        }
        let mut child = c.spawn().with_context(|| format!("run {} {}", self.python, self.script.display()))?;
        // read stdout while the child runs (a full pipe would block it)
        let reader = child.stdout.take().map(|mut so| {
            std::thread::spawn(move || {
                use std::io::Read;
                let mut s = String::new();
                let _ = so.read_to_string(&mut s);
                s
            })
        });
        let t_end = rt::now() + timeout;
        loop {
            if let Some(st) = child.try_wait()? {
                let out = reader.map(|h| h.join().unwrap_or_default()).unwrap_or_default();
                if !st.success() {
                    bail!("ecat_bridge.py {} failed ({st})", args.first().unwrap_or(&""));
                }
                return Ok(out);
            }
            if rt::now() > t_end {
                let _ = child.kill();
                bail!("ecat_bridge.py {} took more than {timeout} s", args.first().unwrap_or(&""));
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// quick_accel + quick_model of the step test. `req` is the request file (see ecat_bridge.py).
    pub fn quick_model(&self, req: &Path) -> Result<QuickModel> {
        let out = self.run(&["quick-model", &req.to_string_lossy()], 120.0, true)?;
        let line = out.lines().rev().find(|l| l.trim_start().starts_with('{')).ok_or_else(|| anyhow!("no JSON from quick-model"))?;
        let v: Value = serde_json::from_str(line)?;
        let f = |k: &str| v.get(k).and_then(Value::as_f64);
        Ok(QuickModel { measured_acc: f("measured_acc"), servo_lag_ms: f("servo_lag_ms"), tau_ms: f("tau_ms"), t_ms: f("T_ms") })
    }

    fn ref_call(&self, which: &str, args: &Map<String, Value>, dir: &Path) -> Result<Value> {
        let a = dir.join(format!("_{which}_args.json"));
        let o = dir.join(format!("_{which}.json"));
        std::fs::write(&a, serde_json::to_vec(&Value::Object(args.clone()))?)?;
        self.run(&[which, &a.to_string_lossy(), &o.to_string_lossy()], 120.0, false)?;
        let v: Value = serde_json::from_slice(&std::fs::read(&o)?)?;
        let _ = std::fs::remove_file(&a);
        let _ = std::fs::remove_file(&o);
        Ok(v)
    }

    pub fn replay_ref(&self, args: &Map<String, Value>, dir: &Path) -> Result<ReplayRef> {
        let v = self.ref_call("replay-ref", args, dir)?;
        let vec1 = |k: &str| -> Result<Vec<f64>> {
            v.get(k).and_then(Value::as_array).ok_or_else(|| anyhow!("replay-ref: no {k}"))?
                .iter().map(|x| x.as_f64().ok_or_else(|| anyhow!("replay-ref: {k} not numeric"))).collect()
        };
        let vec6 = |k: &str| -> Result<Vec<[f64; 6]>> {
            v.get(k).and_then(Value::as_array).ok_or_else(|| anyhow!("replay-ref: no {k}"))?
                .iter()
                .map(|r| {
                    let r = r.as_array().ok_or_else(|| anyhow!("replay-ref: {k} row"))?;
                    let mut a = [0.0; 6];
                    for (i, x) in r.iter().take(6).enumerate() {
                        a[i] = x.as_f64().unwrap_or(f64::NAN);
                    }
                    Ok(a)
                })
                .collect()
        };
        Ok(ReplayRef {
            source: v.get("source").and_then(Value::as_str).unwrap_or("?").to_string(),
            h_smooth: v.get("h_smooth").and_then(Value::as_u64).unwrap_or(0) as usize,
            t: vec1("t")?,
            q_clean: vec6("q_clean")?,
            q: vec6("q")?,
            release: vec1("release")?,
            q_cmd: vec6("q_cmd")?,
            rel_cmd: vec1("rel_cmd")?,
        })
    }

    pub fn soak_phases(&self, args: &Map<String, Value>, dir: &Path) -> Result<Vec<[f64; 3]>> {
        let v = self.ref_call("soak-ref", args, dir)?;
        let rows = v.get("ph").and_then(Value::as_array).ok_or_else(|| anyhow!("soak-ref: no ph"))?;
        Ok(rows
            .iter()
            .map(|r| {
                let mut a = [0.0; 3];
                if let Some(r) = r.as_array() {
                    for (i, x) in r.iter().take(3).enumerate() {
                        a[i] = x.as_f64().unwrap_or(0.0);
                    }
                }
                a
            })
            .collect())
    }

    /// raw.pkl.gz + raw.npz, then the analysis (output on the console).
    pub fn import(&self, dir: &Path, analyze: bool) -> Result<()> {
        let d = dir.to_string_lossy().to_string();
        let mut args = vec!["import", d.as_str()];
        if !analyze {
            args.push("--no-analyze");
        }
        self.run(&args, 3600.0, false)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct QuickModel {
    pub measured_acc: Option<f64>,
    pub servo_lag_ms: Option<f64>,
    pub tau_ms: Option<f64>,
    pub t_ms: Option<f64>,
}

pub struct ReplayRef {
    pub source: String,
    pub h_smooth: usize,
    pub t: Vec<f64>,
    pub q_clean: Vec<[f64; 6]>,
    pub q: Vec<[f64; 6]>,
    pub release: Vec<f64>,
    pub q_cmd: Vec<[f64; 6]>,
    pub rel_cmd: Vec<f64>,
}

/// The request file of quick-model: every frame so far + the step test (paths relative to dir).
pub fn quick_request(dir: &Path, frames_rel: &str, step: &Value) -> Result<PathBuf> {
    let req = dir.join("_quick_req.json");
    let v = json!({"frames": {"$npy": frames_rel}, "step": step});
    std::fs::write(&req, serde_json::to_vec(&v)?)?;
    Ok(req)
}
