//! Ports of small pieces of meca_latency_test.py that run inside the cycle: ActionRef (released
//! actions, interpolation, extrapolation), SensorEmu (an external sensor in the host loop) and
//! piecewise profiles. The reference arrays themselves come from ecat_bridge.py (same numpy RNG
//! as the TCP runs).

use std::collections::VecDeque;

/// segs = [(duration, value), ...] -> value at t (like piecewise() in Python).
pub fn piecewise(segs: &[(f64, f64)]) -> impl Fn(f64) -> f64 + Send + Clone {
    let mut bounds = Vec::with_capacity(segs.len());
    let mut acc = 0.0;
    for s in segs {
        acc += s.0;
        bounds.push(acc);
    }
    let vals: Vec<f64> = segs.iter().map(|s| s.1).collect();
    move |t: f64| {
        // searchsorted(bounds, t + 1e-9, side="right")
        let i = bounds.partition_point(|&b| b <= t + 1e-9);
        vals[i.min(vals.len() - 1)]
    }
}

/// Released actions q_k (offsets from the start pose, deg) due at t_k, known from release_k on.
/// See ActionRef in meca_latency_test.py: same interpolation, extrapolation (order 1 or 2) and
/// horizon.
pub struct ActionRef {
    pub t: Vec<f64>,
    pub q: Vec<[f64; 6]>,
    pub rel: Vec<f64>,
    s: Vec<[f64; 6]>,
    pub horizon: f64,
    pub order: u8,
}

impl ActionRef {
    pub fn new(t: Vec<f64>, q: Vec<[f64; 6]>, release: &[f64], horizon: f64, order: u8) -> ActionRef {
        let n = t.len();
        let mut rel = Vec::with_capacity(n);
        let mut m = f64::NEG_INFINITY;
        for &r in release {
            m = m.max(r);
            rel.push(m);
        }
        let mut s = vec![[0.0; 6]; n];
        for k in 0..n.saturating_sub(1) {
            let dt = t[k + 1] - t[k];
            for j in 0..6 {
                s[k][j] = (q[k + 1][j] - q[k][j]) / dt;
            }
        }
        ActionRef { t, q, rel, s, horizon, order }
    }

    pub fn len(&self) -> usize {
        self.t.len()
    }

    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }

    fn released(&self, now: f64) -> usize {
        self.rel.partition_point(|&r| r <= now)
    }

    /// Index of the last action with t <= te among the first n, or None.
    fn seg(&self, n: usize, te: f64) -> Option<usize> {
        let i = self.t[..n].partition_point(|&x| x <= te);
        if i == 0 {
            None
        } else {
            Some(i - 1)
        }
    }

    fn extrap(&self, n: usize, te: f64) -> ([f64; 6], [f64; 6]) {
        let last = n - 1;
        let dt = (te - self.t[last]).min(self.horizon);
        let mut p = [0.0; 6];
        let mut v = [0.0; 6];
        if self.order == 2 && n >= 3 {
            let h1 = self.t[last] - self.t[last - 1];
            let h0 = self.t[last - 1] - self.t[last - 2];
            for j in 0..6 {
                let acc = (self.s[last - 1][j] - self.s[last - 2][j]) / (0.5 * (h1 + h0));
                let v0 = self.s[last - 1][j] + acc * h1 / 2.0;
                p[j] = self.q[last][j] + v0 * dt + 0.5 * acc * dt * dt;
                v[j] = v0 + acc * dt;
            }
        } else {
            for j in 0..6 {
                p[j] = self.q[last][j] + self.s[last - 1][j] * dt;
                v[j] = self.s[last - 1][j];
            }
        }
        (p, v)
    }

    /// Position the controller can compute at time `now` for time te.
    pub fn at(&self, te: f64, now: f64) -> [f64; 6] {
        let n = self.released(now);
        if n == 0 {
            return [0.0; 6];
        }
        let Some(i) = self.seg(n, te) else { return self.q[0] };
        if i < n - 1 {
            let mut p = [0.0; 6];
            for j in 0..6 {
                p[j] = self.q[i][j] + self.s[i][j] * (te - self.t[i]);
            }
            return p;
        }
        if n == 1 || n == self.len() {
            return self.q[n - 1];
        }
        self.extrap(n, te).0
    }

    /// Velocity (feedforward) the controller can compute at time `now` for time te.
    pub fn slope(&self, te: f64, now: f64) -> [f64; 6] {
        let n = self.released(now);
        if n < 2 {
            return [0.0; 6];
        }
        let Some(i) = self.seg(n, te) else { return [0.0; 6] };
        if i < n - 1 {
            return self.s[i];
        }
        if n == self.len() || te - self.t[n - 1] > self.horizon {
            return [0.0; 6];
        }
        self.extrap(n, te).1
    }
}

/// splitmix64 + Box-Muller: noise for SensorEmu (the numpy stream is not reproduced).
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self, sd: f64) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        sd * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// An external position sensor in the host loop: samples every 1 / rate s (every tick if rate <= 0),
/// adds white noise, the controller gets the sample `delay` s later and holds it. All zero: the
/// robot's own feedback, untouched. The (robot, host) time returned is the one of the frame the
/// sample came from, so the logged feedback age includes the sensor. Port of SensorEmu.
pub struct SensorEmu {
    delay: f64,
    rate: f64,
    noise: f64,
    rng: Rng,
    queue: VecDeque<(f64, [f64; 6], (f64, f64))>,
    next_t: f64,
    held: Option<([f64; 6], (f64, f64))>,
}

impl SensorEmu {
    pub fn new(delay_s: f64, rate_hz: f64, noise_deg: f64, seed: u64) -> SensorEmu {
        SensorEmu {
            delay: delay_s,
            rate: rate_hz,
            noise: noise_deg,
            rng: Rng::new(seed),
            queue: VecDeque::with_capacity(4096),
            next_t: 0.0,
            held: None,
        }
    }

    pub fn passthrough(&self) -> bool {
        self.delay == 0.0 && self.rate <= 0.0 && self.noise == 0.0
    }

    pub fn observe(&mut self, t: f64, y: [f64; 6], frame: (f64, f64)) -> ([f64; 6], (f64, f64)) {
        if self.passthrough() {
            return (y, frame);
        }
        if self.rate <= 0.0 || t >= self.next_t - 1e-12 {
            let mut noisy = y;
            if self.noise > 0.0 {
                for v in noisy.iter_mut() {
                    *v += self.rng.normal(self.noise);
                }
            }
            if self.queue.len() < self.queue.capacity() {
                self.queue.push_back((t + self.delay, noisy, frame));
            }
            if self.rate > 0.0 {
                while self.next_t <= t + 1e-12 {
                    self.next_t += 1.0 / self.rate;
                }
            }
        }
        while let Some(&(due, yh, fh)) = self.queue.front() {
            if due <= t + 1e-12 {
                self.held = Some((yh, fh));
                self.queue.pop_front();
            } else {
                break;
            }
        }
        self.held.unwrap_or((y, frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same cases as test_action_ref_uses_only_released_actions in tests/test_analysis.py.
    #[test]
    fn action_ref_uses_only_released_actions() {
        let t = vec![0.0, 0.02, 0.04, 0.06];
        let q: Vec<[f64; 6]> = [0.0, 1.0, 2.0, 3.0].iter().map(|&x| [x, 0.0, 0.0, 0.0, 0.0, 0.0]).collect();
        let r = ActionRef::new(t, q, &[0.0, 0.0, 0.04, 0.04], 0.04, 1);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(r.at(0.01, 0.01)[0], 0.5));
        assert!(close(r.at(0.03, 0.03)[0], 1.5));
        assert!(close(r.slope(0.05, 0.03)[0], 50.0));
        assert!(close(r.at(0.20, 0.03)[0], 3.0));
        assert!(close(r.at(0.20, 1.0)[0], 3.0));
        assert_eq!(r.slope(0.20, 1.0)[0], 0.0);
    }

    #[test]
    fn quadratic_extrapolation_follows_a_parabola() {
        let t: Vec<f64> = (0..6).map(|k| k as f64 * 0.02).collect();
        let q: Vec<[f64; 6]> = t.iter().map(|&x| [x * x * 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]).collect();
        let rel = [0.0, 0.0, 0.0, 0.0, 10.0, 10.0]; // 4 actions known
        let r = ActionRef::new(t, q, &rel, 1.0, 2);
        let p = r.at(0.08, 0.07)[0]; // past the last released one (t = 0.06)
        assert!((p - 0.64).abs() < 1e-9, "{p}");
    }

    #[test]
    fn piecewise_profile() {
        let f = piecewise(&[(0.15, 0.0), (0.3, 5.0), (0.3, 0.0)]);
        assert_eq!(f(0.0), 0.0);
        assert_eq!(f(0.15), 5.0);
        assert_eq!(f(0.449), 5.0);
        assert_eq!(f(0.46), 0.0);
        assert_eq!(f(9.0), 0.0);
    }

    #[test]
    fn sensor_delay_and_hold() {
        let mut s = SensorEmu::new(0.010, 100.0, 0.0, 0);
        let mut seen = Vec::new();
        for k in 0..40 {
            let t = k as f64 * 0.001;
            let (y, _) = s.observe(t, [t, 0.0, 0.0, 0.0, 0.0, 0.0], (t, t));
            seen.push(y[0]);
        }
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(seen[5], 0.005)); // nothing received yet: passes y through
        assert!(close(seen[12], 0.0)); // sample of t = 0 received at 10 ms, held
        assert!(close(seen[25], 0.010)); // sample of t = 10 ms received at 20 ms
    }
}
