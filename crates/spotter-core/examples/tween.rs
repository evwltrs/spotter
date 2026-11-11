//! Makes the frame halfway between two frames of a video with RIFE, and writes the three side by side:
//!   cargo run -p spotter-core --release --example tween -- clip.mp4 SECONDS out.png
use spotter_core::Frame;
use spotter_core::detector::Device;
use spotter_core::interp::Interpolator;
use spotter_core::video::{Decoder, Input, probe};
use std::path::Path;

fn main() -> anyhow::Result<()> {
	let a: Vec<String> = std::env::args().skip(1).collect();
	let p = probe(Path::new(&a[0]))?;
	let t: f64 = a[1].parse()?;
	let mut dec = Decoder::open(
		Path::new(&a[0]),
		&Input { start: t, length: Some(0.5), crop: None, w: p.w, h: p.h, fps: p.fps, colour: p.colour.clone() },
	)?;
	let (mut f0, mut f1) = (Frame::default(), Frame::default());
	dec.next(&mut f0)?;
	// two frames apart, so there's movement to fill
	dec.next(&mut f1)?;
	dec.next(&mut f1)?;
	let mut i = Interpolator::load(Device::DirectMl)?;
	let mut mid = Frame::default();
	i.between(&f0, &f1, 0.5, &mut mid)?;
	for _ in 0..5 {
		i.between(&f0, &f1, 0.5, &mut mid)?;
	}
	println!("{:.1} ms a frame between", i.ms);
	let (w, h) = (f0.w as usize, f0.h as usize);
	let mut sheet = vec![0u8; w * 3 * h * 4];
	for (k, f) in [&f0, &mid, &f1].iter().enumerate() {
		for y in 0..h {
			sheet[(y * w * 3 + k * w) * 4..][..w * 4].copy_from_slice(&f.rgba[y * w * 4..][..w * 4]);
		}
	}
	image_write(&a[2], (w * 3) as u32, h as u32, &sheet)
}

fn image_write(path: &str, w: u32, h: u32, rgba: &[u8]) -> anyhow::Result<()> {
	let mut c = spotter_core::video::ffmpeg();
	let mut child = c
		.args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{w}x{h}"), "-i", "-", path])
		.stdin(std::process::Stdio::piped())
		.spawn()?;
	use std::io::Write;
	child.stdin.take().unwrap().write_all(rgba)?;
	child.wait()?;
	Ok(())
}
