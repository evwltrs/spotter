//! Making a small picture big with detail rather than blur: Real-ESRGAN's small general model
//! (realesr-general-x4v3, BSD-3-Clause) makes it four times bigger on the GPU, and that is
//! then averaged down to the size asked for. It runs before anything is drawn, so the lines, writing
//! and grain all go on at the full size and stay sharp.

use crate::Frame;
use crate::detector::Device;
use crate::model::{Model, get_for};
use anyhow::{Result, anyhow, bail};
use ort::value::TensorRef;
use rayon::prelude::*;
use std::time::Instant;

pub const MODEL: &str = "realesr-general-x4v3";

pub struct Upscaler {
	model: Model,
	input: Vec<f32>,
	/// how long the last frame took, in milliseconds
	pub ms: f32,
}

impl Upscaler {
	pub fn load(device: Device) -> Result<Self> {
		let path = get_for(MODEL, device, true)?;
		Ok(Self { model: Model::load(&path, device)?, input: Vec::new(), ms: 0.0 })
	}

	pub fn device(&self) -> Device {
		self.model.device
	}

	/// `src` four times bigger, then averaged down (or, if four times isn't enough, stretched) to
	/// `w` by `h`, into `out`.
	pub fn run(&mut self, src: &Frame, w: u32, h: u32, out: &mut Frame) -> Result<()> {
		// on the GPU's own thread (see gpu.rs)
		if self.model.device == Device::Cpu {
			self.run_here(src, w, h, out)
		} else {
			crate::gpu::run_on_gpu("the upscaler", || self.run_here(src, w, h, out))
		}
	}

	fn run_here(&mut self, src: &Frame, w: u32, h: u32, out: &mut Frame) -> Result<()> {
		let (sw, sh) = (src.w as usize, src.h as usize);
		if sw == 0 || sh == 0 {
			bail!("nothing to make bigger");
		}
		let t0 = Instant::now();
		let plane = sw * sh;
		self.input.resize(3 * plane, 0.0);
		let (r, rest) = self.input.split_at_mut(plane);
		let (g, b) = rest.split_at_mut(plane);
		r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).zip(src.rgba.par_chunks_exact(4)).for_each(
			|(((r, g), b), p)| {
				*r = p[0] as f32 / 255.0;
				*g = p[1] as f32 / 255.0;
				*b = p[2] as f32 / 255.0;
			},
		);
		let tensor = TensorRef::from_array_view(([1usize, 3, sh, sw], &self.input[..])).map_err(|e| anyhow!("{e}"))?;
		let result = self.model.session.run(ort::inputs![tensor]).map_err(|e| anyhow!("the upscaler failed: {e}"))?;
		let (shape, big) = result[0].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
		let (bh, bw) = (shape[2] as usize, shape[3] as usize);
		let bplane = bw * bh;
		out.w = w;
		out.h = h;
		out.rgba.resize((w * h * 4) as usize, 255);
		let (sx, sy) = (bw as f32 / w as f32, bh as f32 / h as f32);
		let byte = |v: f32| (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
		out.rgba.par_chunks_exact_mut(w as usize * 4).enumerate().for_each(|(y, line)| {
			// each output pixel is the mean of the big picture's pixels it covers (at least one)
			let y0 = (y as f32 * sy) as usize;
			let y1 = (((y + 1) as f32 * sy) as usize).clamp(y0 + 1, bh);
			for x in 0..w as usize {
				let x0 = (x as f32 * sx) as usize;
				let x1 = (((x + 1) as f32 * sx) as usize).clamp(x0 + 1, bw);
				let mut sum = [0.0f32; 3];
				for yy in y0.min(bh - 1)..y1 {
					let row = yy * bw;
					for xx in x0.min(bw - 1)..x1 {
						let i = row + xx;
						sum[0] += big[i];
						sum[1] += big[bplane + i];
						sum[2] += big[2 * bplane + i];
					}
				}
				let n = ((y1 - y0.min(bh - 1)) * (x1 - x0.min(bw - 1))) as f32;
				let p = &mut line[x * 4..x * 4 + 4];
				p[0] = byte(sum[0] / n);
				p[1] = byte(sum[1] / n);
				p[2] = byte(sum[2] / n);
				p[3] = 255;
			}
		});
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		Ok(())
	}
}
