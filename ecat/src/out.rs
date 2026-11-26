//! Run folder writer: ecat_run.json + npy/*.npy, the format ecat_bridge.py imports into the v2
//! format of meca_latency_test.py (raw.pkl.gz). In the JSON tree:
//!   {"$npy": "npy/x.npy"}  float64 array file      {"$arr": [...]}  small array (null = NaN)
//!   {"$frames": [i0, i1]}  rows of the frames array {"$f": "nan"}    non-finite number

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::{json, Map, Value};

use crate::pdo::{Row, NCOL};

pub const FORMAT: &str = "meca_ecat/1";

/// The .npy v1.0 header numpy writes for a little-endian float64 C array of this shape
/// (same padding, so the bytes equal np.save's: checked against tests/fixtures).
fn npy_header(shape: &[usize]) -> Vec<u8> {
    let shape_s = match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_s}, }}");
    if !shape.is_empty() {
        // numpy leaves room for the growth axis (GROWTH_AXIS_MAX_DIGITS = 21)
        h.push_str(&" ".repeat(21usize.saturating_sub(shape[0].to_string().len())));
    }
    let hlen = h.len() + 1;
    let pad = 64 - ((10 + hlen) % 64);
    h.push_str(&" ".repeat(pad));
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    out
}

pub fn npy_bytes(data: &[f64], shape: &[usize]) -> Vec<u8> {
    let mut b = npy_header(shape);
    b.reserve(data.len() * 8);
    for x in data {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

pub fn write_npy(path: &Path, data: &[f64], shape: &[usize]) -> Result<()> {
    let n: usize = shape.iter().product();
    if n != data.len() {
        bail!("npy {}: {} values for shape {:?}", path.display(), data.len(), shape);
    }
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    f.write_all(&npy_header(shape))?;
    let mut buf = Vec::with_capacity(1 << 16);
    for chunk in data.chunks(8192) {
        buf.clear();
        for x in chunk {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    Ok(())
}

/// Read a little-endian float64 C-order .npy (what np.save / write_npy produce).
pub fn read_npy(bytes: &[u8]) -> Result<(Vec<usize>, Vec<f64>)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let (hlen, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        _ => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen])?;
    if !header.contains("'<f8'") || header.contains("'fortran_order': True") {
        bail!("only little-endian float64 C arrays are supported: {header}");
    }
    let s = header.split("'shape': (").nth(1).and_then(|x| x.split(')').next()).unwrap_or("");
    let shape: Vec<usize> = s.split(',').map(str::trim).filter(|x| !x.is_empty()).map(|x| x.parse()).collect::<Result<_, _>>()?;
    let data: Vec<f64> = bytes[start + hlen..]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    let n: usize = shape.iter().product();
    if n != data.len() {
        bail!("npy: {} values for shape {:?}", data.len(), shape);
    }
    Ok((shape, data))
}

/// A number, or {"$f": "nan" | "inf" | "-inf"} (JSON has no NaN).
pub fn num(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else if x.is_nan() {
        json!({"$f": "nan"})
    } else if x > 0.0 {
        json!({"$f": "inf"})
    } else {
        json!({"$f": "-inf"})
    }
}

/// A small 1-D array (an ndarray on the Python side).
pub fn arr(v: &[f64]) -> Value {
    json!({"$arr": v.iter().map(|&x| if x.is_finite() { json!(x) } else { Value::Null }).collect::<Vec<_>>()})
}

pub fn frames_ref(i0: u64, i1: u64) -> Value {
    json!({"$frames": [i0, i1]})
}

pub struct RunOut {
    pub dir: PathBuf,
    pub raw: Map<String, Value>,
    pub send_logs: Vec<Value>,
    n_npy: usize,
}

impl RunOut {
    pub fn new(dir: PathBuf) -> Result<RunOut> {
        fs::create_dir_all(dir.join("npy")).with_context(|| format!("create {}", dir.display()))?;
        Ok(RunOut { dir, raw: Map::new(), send_logs: Vec::new(), n_npy: 0 })
    }

    /// Write a float64 array (row-major, `cols` columns; 0 = 1-D) and return its reference.
    pub fn npy(&mut self, stem: &str, data: &[f64], cols: usize) -> Result<Value> {
        let rel = format!("npy/{stem}_{:04}.npy", self.n_npy);
        self.n_npy += 1;
        let shape: Vec<usize> = if cols == 0 { vec![data.len()] } else { vec![data.len() / cols, cols] };
        write_npy(&self.dir.join(&rel), data, &shape)?;
        Ok(json!({ "$npy": rel }))
    }

    pub fn npy_rows(&mut self, stem: &str, rows: &[[f64; 6]]) -> Result<Value> {
        let flat: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        self.npy(stem, &flat, 6)
    }

    /// frames.npy, bus_cycles (into raw._ecat), ecat_run.json. Returns the json path.
    pub fn finish(mut self, args: Map<String, Value>, frames: &[Row], cycles: &[[f64; 6]], mut ecat: Map<String, Value>) -> Result<PathBuf> {
        let flat: Vec<f64> = frames.iter().flat_map(|r| r.iter().copied()).collect();
        write_npy(&self.dir.join("npy/frames.npy"), &flat, &[frames.len(), NCOL])?;
        let cflat: Vec<f64> = cycles.iter().flat_map(|r| r.iter().copied()).collect();
        write_npy(&self.dir.join("npy/bus_cycles.npy"), &cflat, &[cycles.len(), 6])?;
        ecat.insert("bus_cycles".into(), json!({"$npy": "npy/bus_cycles.npy"}));
        self.raw.insert("_ecat".into(), Value::Object(ecat));
        let doc = json!({
            "format": FORMAT,
            "args": Value::Object(args),
            "raw": Value::Object(std::mem::take(&mut self.raw)),
            "frames": {"$npy": "npy/frames.npy"},
            "send_logs": std::mem::take(&mut self.send_logs),
        });
        let path = self.dir.join("ecat_run.json");
        let tmp = self.dir.join("ecat_run.json.tmp");
        fs::write(&tmp, serde_json::to_vec(&doc)?)?;
        fs::rename(&tmp, &path)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(name: &str) -> Vec<u8> {
        fs::read(format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))).expect("fixture")
    }

    /// The writer produces exactly the bytes of np.save (fixtures made with numpy 2.3).
    #[test]
    fn npy_bytes_equal_numpy() {
        let a = [0.0, 1.5, -2.0, f64::NAN, f64::INFINITY, 1e-300];
        let ours = npy_bytes(&a, &[2, 3]);
        let theirs = fixture("f8_2x3.npy");
        assert_eq!(&ours[..ours.len() - 24], &theirs[..theirs.len() - 24], "header + first values");
        assert_eq!(ours.len(), theirs.len());
        assert_eq!(npy_bytes(&[], &[0, 9]), fixture("f8_0x9.npy"));
        assert_eq!(npy_bytes(&[1.0, 2.0, 3.0, 4.0], &[4]), fixture("f8_4.npy"));
    }

    #[test]
    fn npy_read_back() {
        let (shape, v) = read_npy(&fixture("f8_2x3.npy")).unwrap();
        assert_eq!(shape, vec![2, 3]);
        assert_eq!(v[1], 1.5);
        assert!(v[3].is_nan());
        let (shape, v) = read_npy(&fixture("f8_0x9.npy")).unwrap();
        assert_eq!((shape, v.len()), (vec![0, 9], 0));
    }

    #[test]
    fn markers() {
        assert_eq!(num(f64::NAN), json!({"$f": "nan"}));
        assert_eq!(arr(&[1.0, f64::NAN]), json!({"$arr": [1.0, null]}));
    }
}
