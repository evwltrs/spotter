//! Layers: what the tracker knows changing the picture itself, not only what's drawn over it. The
//! things followed get a matte, a soft-edged shape saying how much of each pixel is them, and the
//! matte splits the picture in two: the things (`within`) and the rest of the world (`world`), each
//! made into something else (left as they are, grey, dimmed, black, heat, night vision, two tones,
//! big pixels, blurred, flat colour, 1-bit dither, halftone print, pixel-sorted…). A contour can be
//! drawn along the matte's edge.
//!
//! The shape comes from the boxes: cut out by colour (what in the box doesn't look like what's just
//! around it, which makes clean silhouettes against sky, snow or water), or a rounded shape inside
//! each; or from the movement finder's own idea of what's moving (organic, but coarse); or both. It's worked out at a quarter of the frame's
//! size and blurred, which gives the soft edge and lets `grow` swell or shrink it.

use crate::Frame;
use crate::Rect;
use crate::look::Look;
use crate::outline::{MASK, Shape};
use crate::overlay::colour;
use rayon::prelude::*;
use tiny_skia::{LineCap, LineJoin, Paint, PathBuilder, PixmapMut, Stroke, Transform};

/// Output pixels per matte pixel.
pub const SCALE: usize = 2;

/// Whether a look uses layers at all; if not, nothing here runs and the picture is untouched.
pub fn active(l: &Look) -> bool {
	l.world != "same"
		|| l.within != "same"
		|| l.contour > 0.0
		|| l.rings > 0.0
		|| l.reveal == "things"
		|| l.reveal == "world"
		|| (!l.mix.trim().is_empty() && (l.mix_how == "things" || l.mix_how == "world"))
		|| (l.mosh != "off" && l.mosh_where != "all")
}

/// Whether a look wants EdgeTAM's outlines.
pub fn wants_outlines(l: &Look) -> bool {
	(active(l) || crate::echo::active(l) || crate::glyphs::needs_matte(l)) && l.matte == "outline"
}

/// The movement finder's mask, for a matte made from it.
pub struct Moving<'a> {
	/// 0-1 a cell, `cols` by `rows`
	pub mask: &'a [f32],
	pub cols: usize,
	pub rows: usize,
	/// how many output pixels a cell is across
	pub cell: f32,
	/// keep to inside the boxes (when only some things are followed)
	pub only_in_boxes: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Matte {
	pub w: usize,
	pub h: usize,
	/// 0-1 a pixel: how much of it is the things followed
	pub a: Vec<f32>,
}

impl Matte {
	/// The matte for a frame, from boxes (in output pixels) and the movement finder's mask, as the look
	/// says. `shapes`, an outline for each box in order, are used when the look asks for outlines;
	/// without them it cuts out by colour.
	pub fn build(src: &Frame, boxes: &[Rect], moving: Option<Moving>, shapes: Option<&[Shape]>, l: &Look) -> Matte {
		let (fw, fh) = (src.w, src.h);
		let (w, h) = ((fw as usize).div_ceil(SCALE).max(1), (fh as usize).div_ceil(SCALE).max(1));
		let mut a = vec![0.0f32; w * h];
		let s = SCALE as f32;
		let use_boxes = l.matte == "boxes";
		let use_moving = l.matte == "movement" || l.matte == "both";
		let outlined = l.matte == "outline" && shapes.is_some_and(|sh| sh.len() == boxes.len());
		if outlined {
			let shapes = shapes.unwrap();
			let (mx, my) = (MASK as f32 / fw as f32, MASK as f32 / fh as f32);
			for (b, shape) in boxes.iter().zip(shapes) {
				// no shape yet (the window's models haven't answered for it): cut out by colour
				if shape.logits.is_empty() {
					cutout(src, *b, &mut a, w, h);
					continue;
				}
				// kept to its box, a little out, since a shape can run into what touches it
				let pad = b.w.max(b.h) * 0.15;
				let x0 = ((b.x - pad) / s).floor().max(0.0) as usize;
				let x1 = (((b.x + b.w + pad) / s).ceil() as usize).min(w);
				let y0 = ((b.y - pad) / s).floor().max(0.0) as usize;
				let y1 = (((b.y + b.h + pad) / s).ceil() as usize).min(h);
				for y in y0..y1 {
					let ly = (y as f32 + 0.5) * s * my - 0.5;
					for x in x0..x1 {
						let lx = (x as f32 + 0.5) * s * mx - 0.5;
						let logit = bilinear(&shape.logits, MASK, MASK, lx, ly);
						let v = smooth(-3.0, 3.0, logit);
						let i = y * w + x;
						if v > a[i] {
							a[i] = v;
						}
					}
				}
			}
		}
		if (l.matte == "cutout" || l.matte == "both" || (l.matte == "outline" && !outlined)) && !outlined {
			for b in boxes {
				cutout(src, *b, &mut a, w, h);
			}
		}
		if use_boxes {
			for b in boxes {
				// a rounded shape filling most of the box, as most things are rounder than their box
				let (cx, cy) = ((b.x + b.w / 2.0) / s, (b.y + b.h / 2.0) / s);
				let (rx, ry) = ((b.w / 2.0 / s).max(0.5), (b.h / 2.0 / s).max(0.5));
				let x0 = (cx - rx).floor().max(0.0) as usize;
				let x1 = ((cx + rx).ceil() as usize).min(w);
				let y0 = (cy - ry).floor().max(0.0) as usize;
				let y1 = ((cy + ry).ceil() as usize).min(h);
				for y in y0..y1 {
					let dy = ((y as f32 + 0.5 - cy) / ry).abs();
					for x in x0..x1 {
						let dx = ((x as f32 + 0.5 - cx) / rx).abs();
						if dx.powf(2.6) + dy.powf(2.6) <= 1.0 {
							a[y * w + x] = 1.0;
						}
					}
				}
			}
		}
		if use_moving
			&& let Some(m) = moving
			&& m.cols > 0
			&& m.rows > 0
		{
			let per = s / m.cell;
			let near_box = |x: f32, y: f32| {
				boxes.iter().any(|b| {
					let pad = b.w.max(b.h) * 0.1;
					x >= b.x - pad && x <= b.x + b.w + pad && y >= b.y - pad && y <= b.y + b.h + pad
				})
			};
			a.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
				let cy = (y as f32 + 0.5) * per - 0.5;
				for (x, v) in line.iter_mut().enumerate() {
					if m.only_in_boxes && !near_box((x as f32 + 0.5) * s, (y as f32 + 0.5) * s) {
						continue;
					}
					let mv = bilinear(m.mask, m.cols, m.rows, (x as f32 + 0.5) * per - 0.5, cy);
					*v = v.max(smooth(0.25, 0.6, mv));
				}
			});
		}
		let mut m = Matte { w, h, a };
		// soften and swell: blur over the feather, then move where the edge falls by `grow`
		let r = ((l.feather.max(0.02) * 0.04 * fw.min(fh) as f32) / s).round().max(1.0) as usize;
		m.blur(r);
		m.blur(r);
		let edge = 0.5 - l.grow.clamp(-0.45, 0.45);
		let soft = 0.08 + l.feather * 0.4;
		for v in &mut m.a {
			*v = smooth(edge - soft / 2.0, edge + soft / 2.0, *v);
		}
		m
	}

	fn blur(&mut self, r: usize) {
		let (w, h) = (self.w, self.h);
		let mut tmp = vec![0.0f32; w * h];
		let n = (2 * r + 1) as f32;
		// along rows
		tmp.par_chunks_mut(w).zip(self.a.par_chunks(w)).for_each(|(out, line)| {
			let mut sum: f32 = (0..=r).map(|i| line[i.min(w - 1)]).sum::<f32>() + line[0] * r as f32;
			for x in 0..w {
				out[x] = sum / n;
				let add = line[(x + r + 1).min(w - 1)];
				let sub = line[x.saturating_sub(r)];
				sum += add - sub;
			}
		});
		// down columns
		let a = &mut self.a;
		for x in 0..w {
			let at = |y: usize| tmp[y.min(h - 1) * w + x];
			let mut sum: f32 = (0..=r).map(at).sum::<f32>() + at(0) * r as f32;
			for y in 0..h {
				a[y * w + x] = sum / n;
				sum += at(y + r + 1) - at(y.saturating_sub(r));
			}
		}
	}

	/// The matte at an output pixel.
	#[inline]
	pub fn at(&self, x: usize, y: usize) -> f32 {
		let s = SCALE as f32;
		bilinear(&self.a, self.w, self.h, (x as f32 + 0.5) / s - 0.5, (y as f32 + 0.5) / s - 0.5)
	}

	/// Where the matte crosses `level`, as line segments in output pixels (marching squares).
	pub fn contour(&self, level: f32) -> Vec<[f32; 4]> {
		let (w, h) = (self.w, self.h);
		let s = SCALE as f32;
		let mut out = Vec::new();
		let v = |x: usize, y: usize| self.a[y * w + x];
		let px = |x: f32, y: f32| [(x + 0.5) * s, (y + 0.5) * s];
		for y in 0..h.saturating_sub(1) {
			for x in 0..w.saturating_sub(1) {
				let (a, b, c, d) = (v(x, y), v(x + 1, y), v(x + 1, y + 1), v(x, y + 1));
				let case =
					(a > level) as u8 | ((b > level) as u8) << 1 | ((c > level) as u8) << 2 | ((d > level) as u8) << 3;
				if case == 0 || case == 15 {
					continue;
				}
				let lerp =
					|p: f32, q: f32| if (q - p).abs() < 1e-6 { 0.5 } else { ((level - p) / (q - p)).clamp(0.0, 1.0) };
				let (fx, fy) = (x as f32, y as f32);
				let top = px(fx + lerp(a, b), fy);
				let right = px(fx + 1.0, fy + lerp(b, c));
				let bottom = px(fx + lerp(d, c), fy + 1.0);
				let left = px(fx, fy + lerp(a, d));
				let mut seg = |p: [f32; 2], q: [f32; 2]| out.push([p[0], p[1], q[0], q[1]]);
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
		out
	}
}

/// The picture's colour around an output pixel (a matte pixel's worth, averaged).
fn colour_at(src: &Frame, x: f32, y: f32) -> [f32; 3] {
	let (w, h) = (src.w as usize, src.h as usize);
	let (cx, cy) = ((x as isize).clamp(0, w as isize - 1) as usize, (y as isize).clamp(0, h as isize - 1) as usize);
	let mut sum = [0.0f32; 3];
	let mut n = 0.0;
	for yy in cy.saturating_sub(1)..(cy + 2).min(h) {
		for xx in cx.saturating_sub(1)..(cx + 2).min(w) {
			let p = &src.rgba[(yy * w + xx) * 4..][..3];
			sum[0] += p[0] as f32;
			sum[1] += p[1] as f32;
			sum[2] += p[2] as f32;
			n += 1.0;
		}
	}
	[sum[0] / n, sum[1] / n, sum[2] / n]
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
	((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Cut a thing out of its box by colour: the colours of a ring just outside the box stand for the
/// background, and whatever inside doesn't look like any of them is the thing. Kept to a rounded
/// shape a little bigger than the box, so a box that's drifted can't cut out half the sky.
fn cutout(src: &Frame, b: Rect, a: &mut [f32], w: usize, h: usize) {
	let s = SCALE as f32;
	let (fw, fh) = (src.w as f32, src.h as f32);
	if b.w < 2.0 || b.h < 2.0 {
		return;
	}
	// the ring: around the box, a little out from it
	let pad = (b.w.max(b.h) * 0.08).max(3.0);
	let (rx0, ry0, rx1, ry1) = (b.x - pad, b.y - pad, b.x + b.w + pad, b.y + b.h + pad);
	let mut ring: Vec<[f32; 3]> = Vec::with_capacity(96);
	let per = 2.0 * ((rx1 - rx0) + (ry1 - ry0));
	for i in 0..96 {
		let d = i as f32 / 96.0 * per;
		let (x, y) = if d < rx1 - rx0 {
			(rx0 + d, ry0)
		} else if d < (rx1 - rx0) + (ry1 - ry0) {
			(rx1, ry0 + d - (rx1 - rx0))
		} else if d < 2.0 * (rx1 - rx0) + (ry1 - ry0) {
			(rx1 - (d - (rx1 - rx0) - (ry1 - ry0)), ry1)
		} else {
			(rx0, ry1 - (d - 2.0 * (rx1 - rx0) - (ry1 - ry0)))
		};
		if x >= 0.0 && y >= 0.0 && x < fw && y < fh {
			ring.push(colour_at(src, x, y));
		}
	}
	if ring.len() < 8 {
		return;
	}
	// how much the background varies by itself: each ring colour's distance to its nearest other
	let mut spread: Vec<f32> = ring
		.iter()
		.enumerate()
		.map(|(i, c)| {
			ring.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, d)| distance(*c, *d)).fold(f32::MAX, f32::min)
		})
		.collect();
	spread.sort_by(f32::total_cmp);
	let noise = spread[spread.len() * 3 / 4];
	let (lo, hi) = (noise * 1.5 + 10.0, noise * 3.0 + 28.0);
	let (cx, cy) = ((b.x + b.w / 2.0) / s, (b.y + b.h / 2.0) / s);
	let (rx, ry) = ((b.w / 2.0 / s * 1.12).max(1.0), (b.h / 2.0 / s * 1.12).max(1.0));
	let x0 = (cx - rx).floor().max(0.0) as usize;
	let x1 = ((cx + rx).ceil() as usize).min(w);
	let y0 = (cy - ry).floor().max(0.0) as usize;
	let y1 = ((cy + ry).ceil() as usize).min(h);
	for y in y0..y1 {
		let dy = ((y as f32 + 0.5 - cy) / ry).abs();
		for x in x0..x1 {
			let dx = ((x as f32 + 0.5 - cx) / rx).abs();
			let shape = 1.0 - smooth(0.85, 1.0, dx.powf(3.0) + dy.powf(3.0));
			if shape <= 0.0 {
				continue;
			}
			let c = colour_at(src, (x as f32 + 0.5) * s, (y as f32 + 0.5) * s);
			let d = ring.iter().map(|r| distance(c, *r)).fold(f32::MAX, f32::min);
			let v = smooth(lo, hi, d) * shape;
			let i = y * w + x;
			if v > a[i] {
				a[i] = v;
			}
		}
	}
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
	let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
	t * t * (3.0 - 2.0 * t)
}

pub(crate) fn bilinear(a: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
	let x = x.clamp(0.0, (w - 1) as f32);
	let y = y.clamp(0.0, (h - 1) as f32);
	let (x0, y0) = (x as usize, y as usize);
	let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
	let (fx, fy) = (x - x0 as f32, y - y0 as f32);
	let top = a[y0 * w + x0] * (1.0 - fx) + a[y0 * w + x1] * fx;
	let bot = a[y1 * w + x0] * (1.0 - fx) + a[y1 * w + x1] * fx;
	top * (1.0 - fy) + bot * fy
}

fn rgb(hex: &str) -> [f32; 3] {
	let c = colour(hex, 1.0);
	[c.red() * 255.0, c.green() * 255.0, c.blue() * 255.0]
}

/// Heat: black through purple, red and orange to yellow and white.
fn thermal(v: f32) -> [f32; 3] {
	const STOPS: [(f32, [f32; 3]); 6] = [
		(0.0, [8.0, 4.0, 20.0]),
		(0.25, [80.0, 16.0, 120.0]),
		(0.5, [200.0, 40.0, 60.0]),
		(0.7, [245.0, 120.0, 20.0]),
		(0.88, [250.0, 210.0, 60.0]),
		(1.0, [255.0, 255.0, 235.0]),
	];
	let v = v.clamp(0.0, 1.0);
	for i in 1..STOPS.len() {
		let (t1, c1) = STOPS[i];
		if v <= t1 {
			let (t0, c0) = STOPS[i - 1];
			let k = (v - t0) / (t1 - t0);
			return [c0[0] + (c1[0] - c0[0]) * k, c0[1] + (c1[1] - c0[1]) * k, c0[2] + (c1[2] - c0[2]) * k];
		}
	}
	STOPS[5].1
}

/// Everything a pixel can be made into, given the picture around it.
struct Modes<'a> {
	graded: &'a [u8],
	src: &'a [u8],
	w: usize,
	/// a small copy of the picture, for a blurred world
	low: &'a Frame,
	fw: usize,
	fh: usize,
	shadow: [f32; 3],
	highlight: [f32; 3],
	flat: [f32; 3],
	/// big pixels' size, in output pixels
	block: usize,
	/// a halftone cell's size, and a dither dot's, in output pixels
	cell: f32,
	dot: usize,
	/// how far bent (stride, bits), and the picture as a crushed JPEG, if a mode wants it
	bend: f32,
	crushed: Option<Vec<u8>>,
	/// which few frames it is, for modes that jump about now and then
	tick: i64,
}

/// Bayer's 8 by 8 ordered dither: the order each cell's dots turn on in.
const BAYER: [u8; 64] = [
	0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30,
	54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23,
	61, 29, 53, 21,
];

/// The halftone screen's angle, as printers turn it so the dots don't read as a grid: 15 degrees.
const SCREEN: (f32, f32) = (0.258_819, 0.965_926);

impl Modes<'_> {
	fn pixel(&self, mode: &str, x: usize, y: usize) -> [f32; 3] {
		let i = (y * self.w + x) * 4;
		let p = &self.graded[i..i + 3];
		let g = [p[0] as f32, p[1] as f32, p[2] as f32];
		let luma = (0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]) / 255.0;
		match mode {
			"untouched" => {
				let q = &self.src[i..i + 3];
				[q[0] as f32, q[1] as f32, q[2] as f32]
			}
			"mono" => [luma * 255.0; 3],
			"dim" => [g[0] * 0.3, g[1] * 0.3, g[2] * 0.3],
			"black" => [0.0; 3],
			"flat" => self.flat,
			"invert" => [255.0 - g[0], 255.0 - g[1], 255.0 - g[2]],
			"thermal" => thermal(luma),
			"night" => {
				let v = luma.powf(0.8);
				[v * 40.0, v * 255.0, v * 90.0]
			}
			"duotone" => {
				let (s, h) = (self.shadow, self.highlight);
				[s[0] + (h[0] - s[0]) * luma, s[1] + (h[1] - s[1]) * luma, s[2] + (h[2] - s[2]) * luma]
			}
			"pixels" => {
				let b = self.block;
				let (bx, by) = ((x / b * b + b / 2).min(self.fw - 1), (y / b * b + b / 2).min(self.fh - 1));
				let j = (by * self.w + bx) * 4;
				[self.graded[j] as f32, self.graded[j + 1] as f32, self.graded[j + 2] as f32]
			}
			"blur" => self.soft(x as f32 + 0.5, y as f32 + 0.5).unwrap_or(g),
			// 1-bit: each dot shadow or highlight, ordered so the tones come through as patterns
			"dither" => {
				let (dx, dy) = (x / self.dot % 8, y / self.dot % 8);
				let level = (BAYER[dy * 8 + dx] as f32 + 0.5) / 64.0;
				if luma.powf(0.9) > level { self.highlight } else { self.shadow }
			}
			// newsprint: shadow-coloured dots on highlight-coloured paper, bigger where it's darker, on a
			// screen turned 15 degrees
			"halftone" => {
				let (s, c) = SCREEN;
				let cell = self.cell;
				let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
				let (u, v) = (px * c + py * s, -px * s + py * c);
				let (cu, cv) = (((u / cell).floor() + 0.5) * cell, ((v / cell).floor() + 0.5) * cell);
				// the tone where the dot's middle is, from the small soft copy
				let (mx, my) = (cu * c - cv * s, cu * s + cv * c);
				let tone = self.soft(mx, my).map_or(luma, |t| (0.2126 * t[0] + 0.7152 * t[1] + 0.0722 * t[2]) / 255.0);
				let r = cell * 0.5 * std::f32::consts::SQRT_2 * (1.0 - tone).max(0.0).sqrt();
				let d = (u - cu).hypot(v - cv);
				let ink = smooth(r + 0.6, r - 0.6, d);
				let (paper, dot) = (self.highlight, self.shadow);
				[
					paper[0] + (dot[0] - paper[0]) * ink,
					paper[1] + (dot[1] - paper[1]) * ink,
					paper[2] + (dot[2] - paper[2]) * ink,
				]
			}
			// saved as a JPEG crushed to nothing and opened again
			"crushed" => match &self.crushed {
				Some(c) => [c[i] as f32, c[i + 1] as f32, c[i + 2] as f32],
				None => g,
			},
			// raw data read with the wrong row width: each row starts further on than it should, so the
			// picture shears into a slant that wraps round; bent far, a byte late too, so its colours turn
			"stride" => {
				let total = self.w * self.fh;
				let k = 1 + (self.bend * 8.0) as usize;
				let p = (y * (self.w + k) + x) % total;
				let late = usize::from(self.bend >= 0.5);
				let at = |c: usize| self.graded[(p * 4 + c + late) % (total * 4)] as f32;
				[at(0), at(1), at(2)]
			}
			// bits lost: a few levels a colour, and in bands (more, bent further) colours swapped and
			// bits flipped, as memory gone wrong
			"bits" => {
				let drop = 8 - (2 + ((1.0 - self.bend) * 3.0) as u32);
				let q = |v: u8| ((v >> drop) << drop) as f32;
				if crate::tracking::hash((y / 6) as i64, 3301) < self.bend * 0.6 {
					[q(p[1] ^ 0x55), q(p[2]), q(p[0] ^ 0x55)]
				} else {
					[q(p[0]), q(p[1]), q(p[2])]
				}
			}
			// neon bands: the tones put into hard steps of a palette from night to sun
			"vapor" => {
				const BANDS: [[f32; 3]; 7] = [
					[16.0, 6.0, 40.0],
					[151.0, 0.0, 204.0],
					[246.0, 1.0, 157.0],
					[255.0, 56.0, 100.0],
					[45.0, 226.0, 230.0],
					[255.0, 211.0, 25.0],
					[255.0, 248.0, 240.0],
				];
				BANDS[((luma.powf(0.85) * BANDS.len() as f32) as usize).min(BANDS.len() - 1)]
			}
			// macroblocks slid about, jumping every few frames: each block shows one a few blocks away
			"blocks" => {
				let b = (self.block * 2).max(8);
				let (bx, by) = ((x / b) as i64, (y / b) as i64);
				let k = bx * 7919 + by * 104_729;
				let reach = 1.0 + self.bend * 6.0;
				let moved = crate::tracking::hash(k, self.tick) < 0.25 + self.bend * 0.5;
				let (ox, oy) = if moved {
					(
						((crate::tracking::hash(k, self.tick + 91) - 0.5) * 2.0 * reach).round() as i64,
						((crate::tracking::hash(k, self.tick + 37) - 0.5) * reach).round() as i64,
					)
				} else {
					(0, 0)
				};
				let sx = ((bx + ox) * b as i64 + (x % b) as i64).clamp(0, self.fw as i64 - 1) as usize;
				let sy = ((by + oy) * b as i64 + (y % b) as i64).clamp(0, self.fh as i64 - 1) as usize;
				let j = (sy * self.w + sx) * 4;
				[self.graded[j] as f32, self.graded[j + 1] as f32, self.graded[j + 2] as f32]
			}
			// sorted after the rest (`sort_runs`); until then, as it is
			_ => g,
		}
	}

	/// The small soft copy of the picture at an output pixel position, if there is one.
	fn soft(&self, x: f32, y: f32) -> Option<[f32; 3]> {
		let l = self.low;
		if l.w == 0 {
			return None;
		}
		let (lx, ly) = (x / self.fw as f32 * l.w as f32 - 0.5, y / self.fh as f32 * l.h as f32 - 0.5);
		let lx = lx.clamp(0.0, (l.w - 1) as f32);
		let ly = ly.clamp(0.0, (l.h - 1) as f32);
		let (x0, y0) = (lx as usize, ly as usize);
		let (x1, y1) = ((x0 + 1).min(l.w as usize - 1), (y0 + 1).min(l.h as usize - 1));
		let (fx, fy) = (lx - x0 as f32, ly - y0 as f32);
		let at = |xx: usize, yy: usize, c: usize| l.rgba[(yy * l.w as usize + xx) * 4 + c] as f32;
		let mut out = [0.0; 3];
		for (c, o) in out.iter_mut().enumerate() {
			let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
			let bot = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
			*o = top * (1.0 - fy) + bot * fy;
		}
		Some(out)
	}
}

/// Pixel sorting, as Kim Asendorf did it: along each column (or row), runs of pixels brighter than
/// `sortLevel` in the part sorted are put in order of brightness, so light smears into streaks and the
/// dark breaks them. The runs shimmer from frame to frame with the picture: that's the look.
fn sort_runs(out: &mut Frame, m: &Matte, l: &Look) {
	let (world, within) = (l.world == "sorted", l.within == "sorted");
	if !world && !within {
		return;
	}
	let (w, h) = (out.w as usize, out.h as usize);
	let level = l.sort_level.clamp(0.0, 1.0) * 255.0;
	let down = l.sort_dir != "across";
	let snap = &out.rgba;
	let luma = |p: &[u8]| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
	// one line at a time: the pixels along it, which may be sorted, and the sorted line
	let (lines, len) = if down { (w, h) } else { (h, w) };
	let at = |line: usize, k: usize| if down { k * w + line } else { line * w + k };
	let sorted: Vec<Vec<[u8; 4]>> = (0..lines)
		.into_par_iter()
		.map(|line| {
			let mut px: Vec<[u8; 4]> = (0..len).map(|k| snap[at(line, k) * 4..][..4].try_into().unwrap()).collect();
			let open = |k: usize| {
				let (x, y) = if down { (line, k) } else { (k, line) };
				let inside = m.at(x, y) >= 0.5;
				((world && !inside) || (within && inside)) && luma(&px[k]) > level
			};
			let marks: Vec<bool> = (0..len).map(open).collect();
			let mut k = 0;
			while k < len {
				if !marks[k] {
					k += 1;
					continue;
				}
				let start = k;
				while k < len && marks[k] {
					k += 1;
				}
				px[start..k].sort_by(|a, b| luma(a).total_cmp(&luma(b)));
			}
			px
		})
		.collect();
	for (line, px) in sorted.iter().enumerate() {
		for (k, p) in px.iter().enumerate() {
			out.rgba[at(line, k) * 4..][..4].copy_from_slice(p);
		}
	}
}

/// Make the picture's things and world into what the look says, through the matte. `out` holds the
/// graded picture and is changed in place; `src` is the picture before grading.
pub fn composite(out: &mut Frame, src: &Frame, low: &Frame, m: &Matte, l: &Look, t: f64) {
	if l.world == "same" && l.within == "same" {
		return;
	}
	let (w, h) = (out.w as usize, out.h as usize);
	let graded = out.rgba.clone();
	let modes = Modes {
		graded: &graded,
		src: if src.w == out.w && src.h == out.h { &src.rgba } else { &graded },
		w,
		low,
		fw: w,
		fh: h,
		shadow: rgb(&l.shadow),
		highlight: rgb(&l.highlight),
		flat: rgb(&l.matte_color),
		block: ((w.min(h) as f32 / 1080.0 * 18.0).round() as usize).max(2),
		cell: (w.min(h) as f32 / 1080.0 * l.dot_size.max(2.0)).max(3.0),
		dot: ((w.min(h) as f32 / 1080.0 * l.dot_size.max(2.0) / 4.0).round() as usize).max(1),
		bend: l.bend.clamp(0.0, 1.0),
		tick: (t * 30.0 / 4.0).floor() as i64,
		crushed: (l.world == "crushed" || l.within == "crushed").then(|| crate::glitch::crush(&graded, w, h, l.crush)),
	};
	let amount = l.world_amount.clamp(0.0, 1.0);
	let (world, within) = (l.world.as_str(), l.within.as_str());
	out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
		for x in 0..w {
			let a = m.at(x, y);
			let i = x * 4;
			let base = [line[i] as f32, line[i + 1] as f32, line[i + 2] as f32];
			let outside = if world == "same" || a >= 0.999 {
				base
			} else {
				let v = modes.pixel(world, x, y);
				[
					base[0] + (v[0] - base[0]) * amount,
					base[1] + (v[1] - base[1]) * amount,
					base[2] + (v[2] - base[2]) * amount,
				]
			};
			let inside = if within == "same" || a <= 0.001 { base } else { modes.pixel(within, x, y) };
			for c in 0..3 {
				line[i + c] = (outside[c] + (inside[c] - outside[c]) * a).round().clamp(0.0, 255.0) as u8;
			}
		}
	});
	sort_runs(out, m, l);
}

/// Sonar: rings spreading from the things across the world, each fainter than the one inside it, by
/// each pixel's distance from the nearest thing (a chamfer distance over the matte), so they follow
/// the things' shapes. With `pulses` (seconds since each recent beat), one ring goes out a beat.
pub fn draw_rings(out: &mut Frame, m: &Matte, l: &Look, clock: f64, pulses: Option<&[f32]>) {
	if l.rings <= 0.0 || m.w == 0 {
		return;
	}
	let (mw, mh) = (m.w, m.h);
	// how far each matte pixel is from the things, in matte pixels
	let far = 1e9f32;
	let mut d: Vec<f32> = m.a.iter().map(|&a| if a >= 0.5 { 0.0 } else { far }).collect();
	if !d.contains(&0.0) {
		return;
	}
	let diag = std::f32::consts::SQRT_2;
	for y in 0..mh {
		for x in 0..mw {
			let i = y * mw + x;
			let mut v = d[i];
			if x > 0 {
				v = v.min(d[i - 1] + 1.0);
			}
			if y > 0 {
				v = v.min(d[i - mw] + 1.0);
				if x > 0 {
					v = v.min(d[i - mw - 1] + diag);
				}
				if x + 1 < mw {
					v = v.min(d[i - mw + 1] + diag);
				}
			}
			d[i] = v;
		}
	}
	for y in (0..mh).rev() {
		for x in (0..mw).rev() {
			let i = y * mw + x;
			let mut v = d[i];
			if x + 1 < mw {
				v = v.min(d[i + 1] + 1.0);
			}
			if y + 1 < mh {
				v = v.min(d[i + mw] + 1.0);
				if x + 1 < mw {
					v = v.min(d[i + mw + 1] + diag);
				}
				if x > 0 {
					v = v.min(d[i + mw - 1] + diag);
				}
			}
			d[i] = v;
		}
	}
	let (w, h) = (out.w as usize, out.h as usize);
	let u = w.min(h) as f32 / 1080.0;
	let spacing = (l.ring_spacing * u).max(4.0);
	let half = (l.contour_width.max(1.0) * u * 0.5).max(0.6);
	let moved = (l.ring_speed * u) as f64 * clock;
	let reach = spacing * 7.0;
	let ink = rgb(&l.ring_color);
	let strength = l.rings.clamp(0.0, 1.0);
	let s = SCALE as f32;
	out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
		for x in 0..w {
			let dist = bilinear(&d, mw, mh, (x as f32 + 0.5) / s - 0.5, (y as f32 + 0.5) / s - 0.5) * s;
			if dist <= 0.5 || dist > reach {
				continue;
			}
			// how far from the nearest ring: rings moving outward with time, or one from each beat
			let off = match pulses {
				Some(ages) => ages
					.iter()
					.map(|&a| (dist - (l.ring_speed.max(20.0) * 4.0 * u * a + spacing * 0.5)).abs())
					.fold(f32::MAX, f32::min),
				None => {
					let phase = ((dist as f64 - moved) / spacing as f64).rem_euclid(1.0) as f32;
					phase.min(1.0 - phase) * spacing
				}
			};
			let line_a = 1.0 - smooth(half - 0.7, half + 0.7, off);
			if line_a <= 0.0 {
				continue;
			}
			// fading as they spread, and coming in softly just outside the thing
			let k = line_a * strength * (1.0 - dist / reach).powf(1.5) * smooth(0.0, spacing * 0.5, dist);
			for c in 0..3 {
				let p = &mut line[x * 4 + c];
				*p = (*p as f32 + (ink[c] - *p as f32) * k).round().clamp(0.0, 255.0) as u8;
			}
		}
	});
}

/// The contour along the matte's edge, in the look's contour colour and width.
pub fn draw_contour(out: &mut Frame, m: &Matte, l: &Look) {
	if l.contour <= 0.0 {
		return;
	}
	let segs = m.contour(0.5);
	if segs.is_empty() {
		return;
	}
	let u = out.w.min(out.h) as f32 / 1080.0;
	let Some(mut pm) = PixmapMut::from_bytes(&mut out.rgba, out.w, out.h) else { return };
	let mut pb = PathBuilder::new();
	for s in &segs {
		pb.move_to(s[0], s[1]);
		pb.line_to(s[2], s[3]);
	}
	let Some(path) = pb.finish() else { return };
	let mut paint = Paint::default();
	paint.set_color(colour(&l.contour_color, l.contour.clamp(0.0, 1.0)));
	paint.anti_alias = true;
	let stroke = Stroke {
		width: (l.contour_width * u).max(0.5),
		line_cap: LineCap::Round,
		line_join: LineJoin::Round,
		..Default::default()
	};
	pm.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
}
