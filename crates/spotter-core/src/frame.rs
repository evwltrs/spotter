//! Pictures and boxes.

use rayon::prelude::*;

/// A picture as straight RGBA, 8 bits a channel, rows top to bottom with no padding.
#[derive(Clone, Default, Debug)]
pub struct Frame {
	pub w: u32,
	pub h: u32,
	pub rgba: Vec<u8>,
}

impl Frame {
	/// Opaque black.
	pub fn new(w: u32, h: u32) -> Self {
		let mut rgba = vec![0; (w * h * 4) as usize];
		for p in rgba.chunks_exact_mut(4) {
			p[3] = 255;
		}
		Self { w, h, rgba }
	}

	pub fn from_rgba(w: u32, h: u32, rgba: Vec<u8>) -> Self {
		assert_eq!(rgba.len(), (w * h * 4) as usize, "a {w}x{h} frame needs {} bytes", w * h * 4);
		Self { w, h, rgba }
	}

	/// Shrunk (or grown) to w by h. Shrinking averages every source pixel that falls in a target pixel,
	/// so small copies don't shimmer the way skipping pixels does.
	pub fn resized(&self, w: u32, h: u32) -> Frame {
		let mut out = Frame { w, h, rgba: vec![0; (w * h * 4) as usize] };
		self.resize_into(&mut out);
		out
	}

	pub fn resize_into(&self, out: &mut Frame) {
		let (sw, sh, w, h) = (self.w as usize, self.h as usize, out.w as usize, out.h as usize);
		if sw == 0 || sh == 0 || w == 0 || h == 0 {
			return;
		}
		if sw == w && sh == h {
			out.rgba.copy_from_slice(&self.rgba);
			return;
		}
		let src = &self.rgba;
		let rows = out.rgba.par_chunks_mut(w * 4).enumerate();
		if w >= sw || h >= sh {
			// growing, or close to it: bilinear
			let (sx, sy) = (sw as f32 / w as f32, sh as f32 / h as f32);
			rows.for_each(|(y, row)| {
				let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
				let (y0, ty) = (fy as usize, fy.fract());
				let y1 = (y0 + 1).min(sh - 1);
				let (a, b) = (&src[y0 * sw * 4..][..sw * 4], &src[y1 * sw * 4..][..sw * 4]);
				for x in 0..w {
					let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
					let (x0, tx) = (fx as usize, fx.fract());
					let x1 = (x0 + 1).min(sw - 1);
					for c in 0..4 {
						let top = a[x0 * 4 + c] as f32 + (a[x1 * 4 + c] as f32 - a[x0 * 4 + c] as f32) * tx;
						let bot = b[x0 * 4 + c] as f32 + (b[x1 * 4 + c] as f32 - b[x0 * 4 + c] as f32) * tx;
						row[x * 4 + c] = (top + (bot - top) * ty).round() as u8;
					}
				}
			});
			return;
		}
		rows.for_each(|(y, row)| {
			let y0 = y * sh / h;
			let y1 = ((y + 1) * sh / h).max(y0 + 1);
			for x in 0..w {
				let x0 = x * sw / w;
				let x1 = ((x + 1) * sw / w).max(x0 + 1);
				let mut sum = [0u32; 4];
				for yy in y0..y1 {
					let line = &src[(yy * sw + x0) * 4..(yy * sw + x1) * 4];
					for p in line.chunks_exact(4) {
						sum[0] += p[0] as u32;
						sum[1] += p[1] as u32;
						sum[2] += p[2] as u32;
						sum[3] += p[3] as u32;
					}
				}
				let n = ((y1 - y0) * (x1 - x0)) as u32;
				for c in 0..4 {
					row[x * 4 + c] = ((sum[c] + n / 2) / n) as u8;
				}
			}
		});
	}

	/// As `resize_into`, but growing in whole numbers with the columns worked out once: a few times
	/// quicker, and up to a level different here and there, which a model taking it doesn't mind (the
	/// looks keep to `resize_into`, whose every pixel the golden test pins).
	pub fn resize_fast_into(&self, out: &mut Frame) {
		let (sw, sh, w, h) = (self.w as usize, self.h as usize, out.w as usize, out.h as usize);
		if sw == 0 || sh == 0 || w == 0 || h == 0 || !(w >= sw || h >= sh) || (sw == w && sh == h) {
			return self.resize_into(out);
		}
		let src = &self.rgba;
		let rows = out.rgba.par_chunks_mut(w * 4).enumerate();
		// growing, or close to it: bilinear, in whole numbers (weights out of 256), with where each
		// column samples from worked out once rather than on every row
		let (sx, sy) = (sw as f32 / w as f32, sh as f32 / h as f32);
		let cols: Vec<(usize, usize, u32)> = (0..w)
			.map(|x| {
				let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
				let x0 = fx as usize;
				(x0 * 4, (x0 + 1).min(sw - 1) * 4, (fx.fract() * 256.0).round() as u32)
			})
			.collect();
		rows.for_each(|(y, row)| {
			let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
			let (y0, ty) = (fy as usize, (fy.fract() * 256.0).round() as u32);
			let y1 = (y0 + 1).min(sh - 1);
			let (a, b) = (&src[y0 * sw * 4..][..sw * 4], &src[y1 * sw * 4..][..sw * 4]);
			for (o, &(i0, i1, tx)) in row.chunks_exact_mut(4).zip(&cols) {
				for c in 0..4 {
					let top = a[i0 + c] as u32 * (256 - tx) + a[i1 + c] as u32 * tx;
					let bot = b[i0 + c] as u32 * (256 - tx) + b[i1 + c] as u32 * tx;
					o[c] = ((top * (256 - ty) + bot * ty + 32768) >> 16) as u8;
				}
			}
		});
	}

	/// A window of the picture (x, y, w, h in its pixels, any fractions) scaled into `out` at out's
	/// size: smoothly when it's grown or kept about the same, averaged when it's shrunk a lot. What
	/// a virtual camera sees.
	pub fn crop_into(&self, win: [f32; 4], out: &mut Frame) {
		let (sw, sh, w, h) = (self.w as usize, self.h as usize, out.w as usize, out.h as usize);
		if sw == 0 || sh == 0 || w == 0 || h == 0 {
			return;
		}
		let [wx, wy, ww, wh] = win;
		let (sx, sy) = (ww / w as f32, wh / h as f32);
		let src = &self.rgba;
		out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
			if sx <= 1.5 && sy <= 1.5 {
				let fy = (wy + (y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
				let (y0, ty) = (fy as usize, fy.fract());
				let y1 = (y0 + 1).min(sh - 1);
				let (a, b) = (&src[y0 * sw * 4..][..sw * 4], &src[y1 * sw * 4..][..sw * 4]);
				for x in 0..w {
					let fx = (wx + (x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
					let (x0, tx) = (fx as usize, fx.fract());
					let x1 = (x0 + 1).min(sw - 1);
					for c in 0..3 {
						let top = a[x0 * 4 + c] as f32 + (a[x1 * 4 + c] as f32 - a[x0 * 4 + c] as f32) * tx;
						let bot = b[x0 * 4 + c] as f32 + (b[x1 * 4 + c] as f32 - b[x0 * 4 + c] as f32) * tx;
						row[x * 4 + c] = (top + (bot - top) * ty).round() as u8;
					}
					row[x * 4 + 3] = 255;
				}
			} else {
				let y0 = ((wy + y as f32 * sy) as usize).min(sh - 1);
				let y1 = ((wy + (y + 1) as f32 * sy) as usize).clamp(y0 + 1, sh);
				for x in 0..w {
					let x0 = ((wx + x as f32 * sx) as usize).min(sw - 1);
					let x1 = ((wx + (x + 1) as f32 * sx) as usize).clamp(x0 + 1, sw);
					let mut sum = [0u32; 3];
					for yy in y0..y1 {
						for p in src[(yy * sw + x0) * 4..(yy * sw + x1) * 4].chunks_exact(4) {
							sum[0] += p[0] as u32;
							sum[1] += p[1] as u32;
							sum[2] += p[2] as u32;
						}
					}
					let n = ((y1 - y0) * (x1 - x0)) as u32;
					for c in 0..3 {
						row[x * 4 + c] = ((sum[c] + n / 2) / n) as u8;
					}
					row[x * 4 + 3] = 255;
				}
			}
		});
	}
}

/// A box, in whatever units the caller is using.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
}
