//! Console + file logging that never panics: after a SIGHUP (SSH dropped) stdout may be gone and
//! `println!` would panic with EPIPE / EIO in the middle of the exit sequence.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

static FILE: Mutex<Option<File>> = Mutex::new(None);

/// Also copy every line to this file (the run folder's session.log).
pub fn set_file(path: &Path) {
    if let Ok(f) = File::options().create(true).append(true).open(path) {
        if let Ok(mut g) = FILE.lock() {
            *g = Some(f);
        }
    }
}

pub fn line(s: &str) {
    let _ = writeln!(std::io::stdout().lock(), "{s}");
    if let Ok(mut g) = FILE.lock() {
        if let Some(f) = g.as_mut() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    }
}

/// `println!` that cannot panic, and that also goes to the session log.
#[macro_export]
macro_rules! say {
    ($($t:tt)*) => { $crate::log::line(&format!($($t)*)) };
}
