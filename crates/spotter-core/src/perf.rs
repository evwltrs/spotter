//! Where the time goes. With SPOTTER_TIMES set, each stage of a frame notes how long it took, and an
//! export ends with a table of them; otherwise none of this does anything.

use std::sync::{Mutex, OnceLock};
use std::time::Instant;

fn on() -> bool {
	static ON: OnceLock<bool> = OnceLock::new();
	*ON.get_or_init(|| std::env::var_os("SPOTTER_TIMES").is_some())
}

/// (stage, milliseconds in all, times)
static TABLE: Mutex<Vec<(&'static str, f64, u64)>> = Mutex::new(Vec::new());

/// Count `ms` towards `stage`.
pub fn add(stage: &'static str, ms: f64) {
	if !on() {
		return;
	}
	let mut t = TABLE.lock().unwrap();
	match t.iter_mut().find(|r| r.0 == stage) {
		Some(r) => {
			r.1 += ms;
			r.2 += 1;
		}
		None => t.push((stage, ms, 1)),
	}
}

/// A stopwatch for a run of stages: each `mark` notes the time since the last.
pub struct Lap(Option<Instant>);

impl Lap {
	pub fn start() -> Self {
		Self(on().then(Instant::now))
	}

	pub fn mark(&mut self, stage: &'static str) {
		if let Some(t) = self.0 {
			add(stage, t.elapsed().as_secs_f64() * 1000.0);
			self.0 = Some(Instant::now());
		}
	}
}

/// The table so far, per frame of `frames`, slowest first; empty when not timing.
pub fn report(frames: u64) -> String {
	if !on() {
		return String::new();
	}
	let mut t = TABLE.lock().unwrap().clone();
	t.sort_by(|a, b| b.1.total_cmp(&a.1));
	let all: f64 = t.iter().map(|r| r.1).sum();
	let mut out =
		format!("where the time went ({frames} frames, {:.1} ms a frame noted):\n", all / frames.max(1) as f64);
	for (stage, ms, n) in t {
		out += &format!(
			"  {stage:<16} {:>7.1} ms a frame  ({:.1} ms each, {n} times)\n",
			ms / frames.max(1) as f64,
			ms / n.max(1) as f64
		);
	}
	out
}

/// Start the table over.
pub fn clear() {
	TABLE.lock().unwrap().clear();
}
