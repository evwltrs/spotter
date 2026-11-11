//! Runs a model over and over on the GPU and says how long a run takes, for finding out
//! what's fast on this machine:
//!   cargo run -p spotter-core --release --example bench -- NAME|FILE [input=1x3x1024x1024 …] [runs]
//! SPOTTER_LEVEL=0..3 sets ONNX Runtime's graph rewriting. vs=OTHER.onnx also runs that model on the same
//! inputs and says how far its outputs are from this one's (for a half-precision copy, say).
use spotter_core::detector::Device;
use spotter_core::model::{Data, Model, Tensor, get};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
	let mut args = std::env::args().skip(1);
	let which = args.next().expect("a model");
	let path: PathBuf = if which.ends_with(".onnx") { which.into() } else { get(&which)? };
	let mut shapes: Vec<(String, Vec<i64>)> = Vec::new();
	let mut runs = 20;
	let mut vs: Option<PathBuf> = None;
	for a in args {
		if let Some(p) = a.strip_prefix("vs=") {
			vs = Some(p.into());
			continue;
		}
		match a.split_once('=') {
			Some((n, s)) => shapes.push((n.to_string(), s.split('x').map(|v| v.parse().unwrap()).collect())),
			None => runs = a.parse()?,
		}
	}
	let level: u8 = std::env::var("SPOTTER_LEVEL").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
	let t = Instant::now();
	let mut m = Model::load_with(&path, Device::DirectMl, level)?;
	println!("loaded in {:.1}s", t.elapsed().as_secs_f32());
	for p in m.inputs() {
		println!("in  {} {:?} {:?}", p.name, p.kind, p.shape);
	}
	let inputs: Vec<(String, Tensor)> = m
		.inputs()
		.iter()
		.map(|p| {
			let shape: Vec<i64> = shapes
				.iter()
				.find(|s| s.0 == p.name)
				.map(|s| s.1.clone())
				.unwrap_or_else(|| p.shape.iter().map(|&d| if d < 0 { 1 } else { d }).collect());
			let n = shape.iter().product::<i64>().max(0) as usize;
			let t = match p.kind.as_str() {
				"Int64" => Tensor { shape, data: Data::I64(vec![0; n]) },
				_ => Tensor::f32(&shape, (0..n).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect()),
			};
			(p.name.clone(), t)
		})
		.collect();
	let refs: Vec<(&str, &Tensor)> = inputs.iter().map(|(n, t)| (n.as_str(), t)).collect();
	for _ in 0..3 {
		m.run(&refs)?;
	}
	let mut times = Vec::new();
	let t = Instant::now();
	for _ in 0..runs {
		let r = Instant::now();
		m.run(&refs)?;
		times.push(r.elapsed().as_secs_f32() * 1000.0);
	}
	times.sort_by(f32::total_cmp);
	println!(
		"{runs} runs in {:.1}s: {:.1} ms a run (fastest {:.1}, in the model {:.1})",
		t.elapsed().as_secs_f32(),
		times.iter().sum::<f32>() / runs as f32,
		times[0],
		m.ms
	);
	if let Some(other) = vs {
		let a = m.run(&refs)?;
		let mut o = Model::load_with(&other, Device::DirectMl, level)?;
		let b = o.run(&refs)?;
		for ((name, x), (_, y)) in a.iter().zip(&b) {
			let (Some(x), Some(y)) = (x.as_f32(), y.as_f32()) else { continue };
			let span = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-9);
			let (mut most, mut sum) = (0f32, 0f64);
			for (p, q) in x.iter().zip(y) {
				let d = (p - q).abs();
				most = most.max(d);
				sum += d as f64;
			}
			println!(
				"vs {}: {name} largest difference {:.2e} of its size, mean {:.2e}",
				other.display(),
				most / span,
				sum / x.len() as f64 / span as f64
			);
		}
	}
	Ok(())
}
