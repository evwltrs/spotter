//! Prints what the detector finds on each frame of a stretch of a video:
//!   cargo run -p spotter-core --release --example scan -- clip.mp4 START LENGTH [model]
use spotter_core::Frame;
use spotter_core::detector::Device;
use spotter_core::export::{fit, load_detector};
use spotter_core::video::{Decoder, Input, probe};
use std::path::Path;

fn main() -> anyhow::Result<()> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let p = probe(Path::new(&a[0]))?;
	let start: f64 = a[1].parse()?;
	let length: f64 = a[2].parse()?;
	let model = a.get(3).cloned().unwrap_or("x365".into());
	let (mut d, _, _) = load_detector(&model, Device::DirectMl)?;
	let (w, h) = fit(p.w, p.h, 640.0);
	let mut dec = Decoder::open(
		Path::new(&a[0]),
		&Input { start, length: Some(length), crop: None, w, h, fps: p.fps, colour: p.colour.clone() },
	)?;
	let mut f = Frame::default();
	let mut n = 0;
	while dec.next(&mut f)? {
		let found = d.detect(&f, 0.5, "things that move")?;
		let s: Vec<String> = found
			.iter()
			.map(|d| format!("{} {:.2} @{:.2},{:.2} {:.2}x{:.2}", d.cls, d.score, d.x, d.y, d.w, d.h))
			.collect();
		println!("{:.3} {}", start + n as f64 / p.fps, s.join(" | "));
		n += 1;
	}
	Ok(())
}
