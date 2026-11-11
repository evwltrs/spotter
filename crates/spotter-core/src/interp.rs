//! Slow motion that's smooth: the frames between two frames, made by RIFE 4.9 (MIT) on the graphics
//! card, rather than each frame shown again and again. Given two pictures and a moment between them
//! (0 the first, 1 the second), it works out how everything moved and draws the picture at that
//! moment. About 20 ms for 576x1024 on DirectML.

use crate::Frame;
use crate::detector::Device;
use crate::model::{Data, Model, Tensor, get_for};
use anyhow::{Result, bail};
use rayon::prelude::*;

/// RIFE halves the picture several times over, so it's padded to a multiple of this
const MULTIPLE: usize = 64;

pub struct Interpolator {
	model: Model,
	a: Vec<f32>,
	b: Vec<f32>,
	pub ms: f32,
}

/// A frame as RIFE takes it: RGB planes, 0-1, padded out to `pw` by `ph` by repeating the edges.
fn planes(f: &Frame, pw: usize, ph: usize, into: &mut Vec<f32>) {
	let (w, h) = (f.w as usize, f.h as usize);
	into.resize(3 * pw * ph, 0.0);
	let plane = pw * ph;
	let (r, rest) = into.split_at_mut(plane);
	let (g, b) = rest.split_at_mut(plane);
	r.par_chunks_mut(pw).zip(g.par_chunks_mut(pw)).zip(b.par_chunks_mut(pw)).enumerate().for_each(
		|(y, ((r, g), b))| {
			let sy = y.min(h - 1);
			for x in 0..pw {
				let p = &f.rgba[(sy * w + x.min(w - 1)) * 4..][..3];
				r[x] = p[0] as f32 / 255.0;
				g[x] = p[1] as f32 / 255.0;
				b[x] = p[2] as f32 / 255.0;
			}
		},
	);
}

impl Interpolator {
	pub fn load(device: Device) -> Result<Self> {
		Ok(Self {
			model: Model::load(&get_for("rife-4.9", device, false)?, device)?,
			a: Vec::new(),
			b: Vec::new(),
			ms: 0.0,
		})
	}

	/// The picture `t` of the way from `a` to `b` (the same size), into `out`.
	pub fn between(&mut self, a: &Frame, b: &Frame, t: f32, out: &mut Frame) -> Result<()> {
		if a.w != b.w || a.h != b.h || a.w == 0 {
			bail!("frames to go between must be the same size");
		}
		let (w, h) = (a.w as usize, a.h as usize);
		if t <= 0.001 || t >= 0.999 {
			let from = if t <= 0.001 { a } else { b };
			out.w = from.w;
			out.h = from.h;
			out.rgba.clone_from(&from.rgba);
			return Ok(());
		}
		let t0 = std::time::Instant::now();
		let (pw, ph) = (w.div_ceil(MULTIPLE) * MULTIPLE, h.div_ceil(MULTIPLE) * MULTIPLE);
		planes(a, pw, ph, &mut self.a);
		planes(b, pw, ph, &mut self.b);
		let shape = [1, 3, ph as i64, pw as i64];
		let ta = Tensor::f32(&shape, std::mem::take(&mut self.a));
		let tb = Tensor::f32(&shape, std::mem::take(&mut self.b));
		let ts = Tensor::f32(&[1], vec![t]);
		let got = self.model.run(&[("img0", &ta), ("img1", &tb), ("timestep", &ts)]);
		if let (Data::F32(x), Data::F32(y)) = (ta.data, tb.data) {
			self.a = x;
			self.b = y;
		}
		let got = got?;
		let Some(o) = got.first().and_then(|(_, t)| t.as_f32()) else { bail!("RIFE gave nothing") };
		if o.len() < 3 * pw * ph {
			bail!("RIFE gave {} values, not {}", o.len(), 3 * pw * ph);
		}
		out.w = a.w;
		out.h = a.h;
		out.rgba.resize(w * h * 4, 255);
		let plane = pw * ph;
		out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
			for x in 0..w {
				for c in 0..3 {
					line[x * 4 + c] = (o[c * plane + y * pw + x] * 255.0).round().clamp(0.0, 255.0) as u8;
				}
				line[x * 4 + 3] = 255;
			}
		});
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		Ok(())
	}
}
