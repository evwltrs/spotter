//! Depth: how far away each part of the picture is, from Depth Anything V2 small (Apache-2.0) on the
//! graphics card, and what that can do to the picture: far things fade into a colour (fog), lines
//! drawn where the scene crosses each of a set of depths, like a map's contours (slices), a band
//! of depth sweeping from far to near and back, lighting what it passes (sweep), or a lens pulling
//! focus to the main thing followed, everything nearer or farther softening the further it is from it
//! (focus).
//!
//! The model gives relative nearness, not metres, and its range wanders from frame to frame; it's
//! spread between the 2nd and 98th percentiles, eased over time, so the picture doesn't pulse.

use crate::Frame;
use crate::detector::Device;
use crate::look::Look;
use crate::model::{Data, Model, Tensor, get_for};
use crate::overlay::colour;
use anyhow::{Result, bail};
use rayon::prelude::*;
use tiny_skia::{LineCap, Paint, PathBuilder, PixmapMut, Stroke, Transform};

/// The models, best first, each with its input size (fixed, a multiple of its 14 pixel patches, so
/// DirectML plans once): exports take the first, the window the second.
pub const MODELS: [(&str, usize); 2] = [("depth-anything-v2-large", 518), ("depth-anything-v2-small", 392)];

pub fn active(l: &Look) -> bool {
	l.depth != "off" && !l.depth.is_empty()
}

/// Whether the look needs depth worked out: for what depth does itself, or to place splats.
pub fn wanted(l: &Look) -> bool {
	active(l) || crate::splat::wants_depth(l)
}

pub struct Depther {
	/// none in a shell: the window's copy that draws with what the models' thread found
	model: Option<Model>,
	small: Frame,
	input: Vec<f32>,
	/// the nearness range it's spread over, eased
	lo: Option<(f32, f32)>,
	/// 0 far to 1 near, `size` by `size` over the frame
	pub near: Vec<f32>,
	/// focus: how near the thing in focus is, eased so the lens pulls rather than jumps
	focus: Option<f32>,
	size: usize,
	pub ms: f32,
}

impl Depther {
	/// `which` is one of `MODELS`.
	pub fn load(which: (&str, usize), device: Device, half: bool) -> Result<Self> {
		let (name, size) = which;
		Ok(Self {
			model: Some(Model::load(&get_for(name, device, half)?, device)?),
			small: Frame::new(size as u32, size as u32),
			input: vec![0.0; 3 * size * size],
			lo: None,
			near: Vec::new(),
			focus: None,
			size,
			ms: 0.0,
		})
	}

	/// A copy without the model, to draw with what another (on the models' thread) finds.
	pub fn shell(&self) -> Depther {
		Depther {
			model: None,
			small: Frame::default(),
			input: Vec::new(),
			lo: self.lo,
			near: self.near.clone(),
			focus: self.focus,
			size: self.size,
			ms: 0.0,
		}
	}

	/// Whether it can work depth out itself (it's not a shell).
	pub fn has_model(&self) -> bool {
		self.model.is_some()
	}

	/// What another found for a frame.
	pub fn set(&mut self, near: &[f32], range: (f32, f32)) {
		self.near.clear();
		self.near.extend_from_slice(near);
		self.lo = Some(range);
	}

	/// The range nearness was last spread over, for recall.rs.
	pub fn range(&self) -> Option<(f32, f32)> {
		self.lo
	}

	/// What `run` found for a frame before, from recall.rs.
	pub fn recall(&mut self, near: &[u16], range: (f32, f32)) {
		self.near.clear();
		self.near.extend(near.iter().map(|&v| v as f32 / 65535.0));
		self.lo = Some(range);
	}

	pub fn reset(&mut self) {
		self.lo = None;
		self.focus = None;
	}

	/// Pull focus toward what's in `r` (output pixels of a w by h frame): how near the middle of it is,
	/// the median over a grid across its middle, so the background showing past its edges doesn't
	/// count. With nothing to aim at, the focus holds.
	pub fn aim(&mut self, r: Option<crate::Rect>, w: usize, h: usize) {
		let Some(r) = r else { return };
		if self.near.len() != self.size * self.size || r.w < 2.0 || r.h < 2.0 {
			return;
		}
		let mut v: Vec<f32> = Vec::with_capacity(64);
		for j in 0..8 {
			for i in 0..8 {
				let x = r.x + r.w * (0.25 + 0.5 * (i as f32 + 0.5) / 8.0);
				let y = r.y + r.h * (0.25 + 0.5 * (j as f32 + 0.5) / 8.0);
				if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
					v.push(self.at(x as usize, y as usize, w, h));
				}
			}
		}
		if v.is_empty() {
			return;
		}
		v.sort_by(f32::total_cmp);
		let m = v[v.len() / 2];
		self.focus = Some(match self.focus {
			Some(f) => f + (m - f) * 0.2,
			None => m,
		});
	}

	/// Work out how near each part of `frame` is, into `self.near`.
	pub fn run(&mut self, frame: &Frame) -> Result<()> {
		let t0 = std::time::Instant::now();
		let size = self.size;
		frame.resize_fast_into(&mut self.small);
		let plane = size * size;
		let (mean, sd) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
		let (r, rest) = self.input.split_at_mut(plane);
		let (g, b) = rest.split_at_mut(plane);
		r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).zip(self.small.rgba.par_chunks_exact(4)).for_each(
			|(((r, g), b), p)| {
				*r = (p[0] as f32 / 255.0 - mean[0]) / sd[0];
				*g = (p[1] as f32 / 255.0 - mean[1]) / sd[1];
				*b = (p[2] as f32 / 255.0 - mean[2]) / sd[2];
			},
		);
		let picture = Tensor::f32(&[1, 3, size as i64, size as i64], std::mem::take(&mut self.input));
		let Some(model) = self.model.as_mut() else { bail!("a shell has no depth model") };
		let out = model.run(&[("pixel_values", &picture)]);
		self.input = match picture.data {
			Data::F32(v) => v,
			_ => unreachable!(),
		};
		let out = out?;
		let Some(d) = out.first().and_then(|(_, t)| t.as_f32()) else { bail!("the depth model gave nothing") };
		if d.len() != plane {
			bail!("the depth model gave {} values, not {plane}", d.len());
		}
		// spread between the 2nd and 98th percentiles, eased so it doesn't pulse
		let mut sample: Vec<f32> = d.iter().step_by(37).copied().collect();
		sample.sort_by(f32::total_cmp);
		let (p2, p98) = (sample[sample.len() * 2 / 100], sample[sample.len() * 98 / 100]);
		let (lo, hi) = match self.lo {
			Some((lo, hi)) => (lo + (p2 - lo) * 0.15, hi + (p98 - hi) * 0.15),
			None => (p2, p98),
		};
		self.lo = Some((lo, hi));
		let span = (hi - lo).max(1e-6);
		self.near.clear();
		self.near.extend(d.iter().map(|v| ((v - lo) / span).clamp(0.0, 1.0)));
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		Ok(())
	}

	/// Nearness at an output pixel of a w by h frame, 0 far to 1 near, once it's been worked out.
	pub fn near_at(&self, x: usize, y: usize, w: usize, h: usize) -> Option<f32> {
		(self.near.len() == self.size * self.size).then(|| self.at(x, y, w, h))
	}

	/// Nearness at an output pixel of a w by h frame.
	#[inline]
	fn at(&self, x: usize, y: usize, w: usize, h: usize) -> f32 {
		let size = self.size;
		let fx = ((x as f32 + 0.5) / w as f32 * size as f32 - 0.5).clamp(0.0, (size - 1) as f32);
		let fy = ((y as f32 + 0.5) / h as f32 * size as f32 - 0.5).clamp(0.0, (size - 1) as f32);
		let (x0, y0) = (fx as usize, fy as usize);
		let (x1, y1) = ((x0 + 1).min(size - 1), (y0 + 1).min(size - 1));
		let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
		let n = &self.near;
		let top = n[y0 * size + x0] * (1.0 - tx) + n[y0 * size + x1] * tx;
		let bot = n[y1 * size + x0] * (1.0 - tx) + n[y1 * size + x1] * tx;
		top * (1.0 - ty) + bot * ty
	}

	/// Change the picture by depth, as the look says. `t` is the source's time, for the sweep.
	pub fn apply(&self, out: &mut Frame, l: &Look, t: f64) {
		let size = self.size;
		if self.near.len() != size * size {
			return;
		}
		let (w, h) = (out.w as usize, out.h as usize);
		let c = colour(&l.depth_color, 1.0);
		let tint = [c.red() * 255.0, c.green() * 255.0, c.blue() * 255.0];
		let amount = l.depth_amount.clamp(0.0, 1.0);
		match l.depth.as_str() {
			"focus" => {
				// a soft copy: a quarter the size, box-blurred twice, read back smoothly
				let (sw, sh) = ((w / 4).max(1), (h / 4).max(1));
				let small = out.resized(sw as u32, sh as u32);
				let r = (2.0 + 6.0 * amount).round() as usize;
				let soft = box_blur(&box_blur(&small.rgba, sw, sh, r), sw, sh, r);
				let focus = self.focus.unwrap_or(1.0);
				// how far from the focus is fully soft: closer for more
				let depth_of_field = 0.35 - 0.25 * amount;
				out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
					let fy = ((y as f32 + 0.5) / 4.0 - 0.5).clamp(0.0, (sh - 1) as f32);
					let (y0, ty) = (fy as usize, fy.fract());
					let y1 = (y0 + 1).min(sh - 1);
					for x in 0..w {
						let off = (self.at(x, y, w, h) - focus).abs();
						let k = ((off - 0.04) / depth_of_field).clamp(0.0, 1.0);
						let k = k * k * (3.0 - 2.0 * k);
						if k <= 0.01 {
							continue;
						}
						let fx = ((x as f32 + 0.5) / 4.0 - 0.5).clamp(0.0, (sw - 1) as f32);
						let (x0, tx) = (fx as usize, fx.fract());
						let x1 = (x0 + 1).min(sw - 1);
						for c in 0..3 {
							let at = |xx: usize, yy: usize| soft[(yy * sw + xx) * 4 + c];
							let top = at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx;
							let bot = at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx;
							let s = top * (1.0 - ty) + bot * ty;
							let p = &mut line[x * 4 + c];
							*p = (*p as f32 + (s - *p as f32) * k).round().clamp(0.0, 255.0) as u8;
						}
					}
				});
			}
			"fog" => {
				out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
					for x in 0..w {
						let far = (1.0 - self.at(x, y, w, h)).powf(1.6) * amount;
						for k in 0..3 {
							let p = &mut line[x * 4 + k];
							*p = (*p as f32 + (tint[k] - *p as f32) * far).round() as u8;
						}
					}
				});
			}
			"sweep" => {
				// far to near and back over `depthSpeed` seconds
				let period = l.depth_speed.max(0.2) as f64;
				let phase = (t / period).fract() as f32;
				let at = 1.0 - (phase * 2.0 - 1.0).abs();
				let band = 0.03 + 0.05 * amount;
				out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
					for x in 0..w {
						let d = (self.at(x, y, w, h) - at).abs();
						if d >= band {
							continue;
						}
						let k = (1.0 - d / band).powi(2) * (0.4 + 0.6 * amount);
						for c in 0..3 {
							let p = &mut line[x * 4 + c];
							*p = (*p as f32 + (tint[c] - *p as f32) * k).round() as u8;
						}
					}
				});
			}
			"slices" => {
				let levels = (l.depth_lines.round() as usize).clamp(2, 60);
				let (sx, sy) = (w as f32 / size as f32, h as f32 / size as f32);
				let mut pb = PathBuilder::new();
				let n = &self.near;
				for i in 1..levels {
					let level = i as f32 / levels as f32;
					for y in 0..size - 1 {
						for x in 0..size - 1 {
							let (a, b, c2, d) = (
								n[y * size + x],
								n[y * size + x + 1],
								n[(y + 1) * size + x + 1],
								n[(y + 1) * size + x],
							);
							let case = (a > level) as u8
								| ((b > level) as u8) << 1
								| ((c2 > level) as u8) << 2
								| ((d > level) as u8) << 3;
							if case == 0 || case == 15 {
								continue;
							}
							let lerp = |p: f32, q: f32| {
								if (q - p).abs() < 1e-6 { 0.5 } else { ((level - p) / (q - p)).clamp(0.0, 1.0) }
							};
							let pt = |px: f32, py: f32| ((px + 0.5) * sx, (py + 0.5) * sy);
							let (fx, fy) = (x as f32, y as f32);
							let top = pt(fx + lerp(a, b), fy);
							let right = pt(fx + 1.0, fy + lerp(b, c2));
							let bottom = pt(fx + lerp(d, c2), fy + 1.0);
							let left = pt(fx, fy + lerp(a, d));
							let mut seg = |p: (f32, f32), q: (f32, f32)| {
								pb.move_to(p.0, p.1);
								pb.line_to(q.0, q.1);
							};
							match case {
								1 | 14 => seg(left, top),
								2 | 13 => seg(top, right),
								3 | 12 => seg(left, right),
								4 | 11 => seg(right, bottom),
								5 => {
									seg(left, top);
									seg(right, bottom);
								}
								6 | 9 => seg(top, bottom),
								7 | 8 => seg(left, bottom),
								10 => {
									seg(top, right);
									seg(left, bottom);
								}
								_ => {}
							}
						}
					}
				}
				let Some(path) = pb.finish() else { return };
				let u = w.min(h) as f32 / 1080.0;
				let Some(mut pm) = PixmapMut::from_bytes(&mut out.rgba, out.w, out.h) else { return };
				let mut paint = Paint::default();
				paint.set_color(colour(&l.depth_color, (0.25 + 0.75 * amount).min(1.0)));
				paint.anti_alias = true;
				let stroke = Stroke { width: (1.2 * u).max(0.75), line_cap: LineCap::Round, ..Default::default() };
				pm.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
			}
			_ => {}
		}
	}
}

/// A box blur of radius `r` across and then down an RGBA picture, as floats.
fn box_blur<T: Copy + Into<f32> + Sync>(src: &[T], w: usize, h: usize, r: usize) -> Vec<f32> {
	let n = (2 * r + 1) as f32;
	let mut across = vec![0.0f32; w * h * 4];
	across.par_chunks_mut(w * 4).enumerate().for_each(|(y, out)| {
		for c in 0..4 {
			let at = |x: isize| -> f32 { src[(y * w + x.clamp(0, w as isize - 1) as usize) * 4 + c].into() };
			let mut sum: f32 = (-(r as isize)..=r as isize).map(at).sum();
			for x in 0..w {
				out[x * 4 + c] = sum / n;
				sum += at(x as isize + r as isize + 1) - at(x as isize - r as isize);
			}
		}
	});
	let mut down = vec![0.0f32; w * h * 4];
	for c in 0..4 {
		for x in 0..w {
			let at = |y: isize| across[(y.clamp(0, h as isize - 1) as usize * w + x) * 4 + c];
			let mut sum: f32 = (-(r as isize)..=r as isize).map(at).sum();
			for y in 0..h {
				down[(y * w + x) * 4 + c] = sum / n;
				sum += at(y as isize + r as isize + 1) - at(y as isize - r as isize);
			}
		}
	}
	down
}
