//! Runs the detector over one picture and says what it found and how long it took:
//!   cargo run -p spotter-core --release --example detect -- picture.jpg [cpu|directml|webgpu] [s|m|l|x|model.onnx]

use spotter_core::detector::{Detector, Device, SIZES, dfine};
use spotter_core::video::still;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let mut args = std::env::args().skip(1);
	let path = args.next().expect("a picture to look at");
	let device: Device = args.next().as_deref().unwrap_or("cpu").parse()?;
	let frame = still(path.as_ref())?;
	let model = match args.next() {
		Some(m) if SIZES.contains(&m.as_str()) => dfine(&m)?,
		Some(m) => m.into(),
		None => dfine("s")?,
	};
	let t = Instant::now();
	let mut d = Detector::load(&model, device)?;
	println!("{device:?}: loaded in {:.2}s", t.elapsed().as_secs_f32());
	let mut found = d.detect(&frame, 0.5, "everything")?;
	let n = 20;
	let t = Instant::now();
	for _ in 0..n {
		found = d.detect(&frame, 0.5, "everything")?;
	}
	println!("{:.1} ms a look ({:.1} ms in the model)", t.elapsed().as_secs_f32() * 1000.0 / n as f32, d.ms);
	found.sort_by(|a, b| b.score.total_cmp(&a.score));
	for f in found.iter().take(8) {
		println!(
			"  {:<12} {:.2}  {:>4},{:<4} {}x{}",
			f.cls,
			f.score,
			(f.x * frame.w as f32) as i32,
			(f.y * frame.h as f32) as i32,
			(f.w * frame.w as f32) as i32,
			(f.h * frame.h as f32) as i32
		);
	}
	Ok(())
}
