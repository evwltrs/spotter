//! Prints where the footage cuts, as the look ahead finds them, and the frames that came close:
//!   cargo run -p spotter-core --release --example cuts -- clip.mp4 [more clips]
//! CUTS_AT=4.2,7.9 also prints how much the frames changed around those times.
use spotter_core::Frame;
use spotter_core::tape::{Looker, cuts};
use spotter_core::video::{Decoder, Input, probe};
use std::path::Path;

fn main() -> anyhow::Result<()> {
	for clip in std::env::args().skip(1) {
		let p = probe(Path::new(&clip))?;
		let scale = (640.0 / p.w.min(p.h) as f64).min(1.0);
		let (w, h) = (((p.w as f64 * scale) as u32) / 2 * 2, ((p.h as f64 * scale) as u32) / 2 * 2);
		let mut dec = Decoder::open(
			Path::new(&clip),
			&Input { start: 0.0, length: None, crop: None, w, h, fps: p.fps, colour: p.colour.clone() },
		)?;
		let mut f = Frame::default();
		let mut l = Looker::default();
		let mut diff = Vec::new();
		while dec.next(&mut f)? {
			diff.push(l.change(&f));
		}
		let jumps = vec![false; diff.len()];
		let found = cuts(&diff, &jumps);
		let at = |i: usize| i as f64 / p.fps;
		let m = |i: usize| diff.get(i).map_or(0.0, |d| d.moved);
		let show = |i: usize| {
			format!("{:.2}({:.1}/{:.1}/{:.1} c{:.2})", at(i), m(i.saturating_sub(1)), m(i), m(i + 1), diff[i].colours)
		};
		let said: Vec<String> = (1..diff.len()).filter(|&i| found[i]).map(show).collect();
		// close: a spike well over its neighbours, or colours that changed a lot, that wasn't taken
		let near: Vec<String> = (1..diff.len().saturating_sub(1))
			.filter(|&i| {
				!found[i]
					&& ((m(i) > 2.0 && m(i) > 1.6 * m(i - 1).max(m(i + 1)))
						|| diff[i].colours > 0.12
						|| (m(i) > 6.0 && m(i) > 5.0 * m(i - 1)))
			})
			.map(show)
			.collect();
		if let Ok(w) = std::env::var("CUTS_AT") {
			for t in w.split(',') {
				let t: f64 = t.parse()?;
				let i = (t * p.fps).round() as usize;
				let v: Vec<String> = (i.saturating_sub(3)..(i + 4).min(diff.len()))
					.map(|k| format!("{:.1}/c{:.2}", diff[k].moved, diff[k].colours))
					.collect();
				println!("  around {t}: {}", v.join(" "));
			}
		}
		println!("{clip}: cuts {}\n  near {}", said.join(" "), near.join(" "));
	}
	Ok(())
}
