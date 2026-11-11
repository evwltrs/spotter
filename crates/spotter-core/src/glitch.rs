//! Broken video, on purpose.
//!
//! - Datamosh: the picture dragged along by its own motion, block by block, rather than drawn again,
//!   as a video plays when its key frames are lost. The motion is found the way a video encoder finds
//!   it, a block at a time on a small grey copy; each frame the held picture is pushed by it and only
//!   some blocks are drawn fresh. Always (blocks come back now and then) or on cuts (the last shot is
//!   pushed about by the new one's motion before it breaks up).
//! - Tear: a worn tape, rows wobbling, a tracking tear rolling down, colour bleeding sideways,
//!   scanlines, and noise where the heads switch at the bottom.
//! - Trichrome: red from a moment ago, green from half that, blue from now, so what's still stays in
//!   its colours and what moves splits into three.
//! - Stutter: a skipping buffer, a few frames looped, on the music's beats when there are any.
//! - And the ones the matte uses, which need no memory: a JPEG crushed to nothing (`crush`), and the
//!   picture read with the wrong row width as raw data (in matte.rs).

use crate::Frame;
use crate::look::Look;
use crate::matte::Matte;
use crate::tracking::hash;
use rayon::prelude::*;
use std::collections::VecDeque;

/// the small grey copy motion's found on: this wide
const SMALL: usize = 192;
/// blocks of this many small pixels, searched this far each way
const BLOCK: usize = 8;
const REACH: i32 = 6;

#[derive(Default)]
pub struct Glitch {
	/// datamosh: the held picture, the grey copies now and before, their size, each block's motion
	/// (where its contents were before, in small pixels), seconds left holding after a cut, seconds
	/// since the hold ended
	held: Frame,
	small: Vec<u8>,
	was: Vec<u8>,
	sw: usize,
	sh: usize,
	moved: Vec<(i8, i8)>,
	hold: f64,
	since: f64,
	frames: u64,
	/// trichrome: pictures lately, with their source times
	past: VecDeque<(f64, Frame)>,
	/// stutter: pictures lately, and the loop being played (the first picture of it, how many, how
	/// far through, how many times round to go)
	recent: VecDeque<Frame>,
	looping: Option<(usize, usize, usize, usize)>,
	/// kaleidoscope: where its middle is now (0-1 across and down), easing toward what it's on
	middle: Option<(f32, f32)>,
}

impl Glitch {
	/// A jump (a seek, round to the start): nothing before it belongs with what's after.
	pub fn jumped(&mut self) {
		self.held = Frame::default();
		self.was.clear();
		self.hold = 0.0;
		self.since = 10.0;
		self.past.clear();
		self.recent.clear();
		self.looping = None;
	}

	/// Datamosh `out` (the look so far) as `l.mosh` says, the motion found from `src` (the frame as
	/// it came). `cut` says the footage cut here; `matte` is the things', for moshing only the world or
	/// only them; `dt` the output's seconds a frame; `fresh` false draws the same frame again.
	pub fn mosh(
		&mut self,
		out: &mut Frame,
		src: &Frame,
		l: &Look,
		matte: Option<&Matte>,
		cut: bool,
		dt: f64,
		fresh: bool,
	) {
		// off, or the real thing (realmosh.rs), done to the clip before it gets here
		if l.mosh == "off" || l.mosh.is_empty() || l.mosh.starts_with("real") {
			self.held = Frame::default();
			return;
		}
		if fresh {
			let cut = self.motion(src) || cut;
			if self.held.w != out.w || self.held.h != out.h {
				self.held = out.clone();
				return;
			}
			// how likely each block is to be drawn fresh this frame
			let hold = (l.mosh_hold as f64).max(0.05);
			let fresh_blocks = if l.mosh == "always" {
				(dt / hold).clamp(0.01, 1.0) as f32
			} else {
				if cut {
					self.hold = hold;
				}
				if self.hold > 0.0 {
					self.hold -= dt;
					self.since = 0.0;
					0.0
				} else {
					// coming back a block at a time over half a second
					self.since += dt;
					(self.since / 0.5).min(1.0) as f32
				}
			};
			self.frames += 1;
			if fresh_blocks >= 1.0 {
				self.held.rgba.copy_from_slice(&out.rgba);
			} else {
				self.push(out, fresh_blocks);
			}
		}
		if self.held.w != out.w || self.held.h != out.h {
			return;
		}
		// the held picture in place of the look, where it goes
		let (w, held) = (out.w as usize, &self.held.rgba);
		match (l.mosh_where.as_str(), matte) {
			("world" | "things", Some(m)) => {
				let things = l.mosh_where == "things";
				out.rgba.par_chunks_mut(w * 4).zip(held.par_chunks(w * 4)).enumerate().for_each(|(y, (o, s))| {
					for x in 0..w {
						let a = m.at(x, y);
						let k = if things { a } else { 1.0 - a };
						if k <= 0.0 {
							continue;
						}
						for c in 0..3 {
							let i = x * 4 + c;
							o[i] = (o[i] as f32 + (s[i] as f32 - o[i] as f32) * k).round() as u8;
						}
					}
				});
			}
			_ => out.rgba.copy_from_slice(held),
		}
	}

	/// Find each block's motion from the last frame to this one; says whether it looks like a cut.
	fn motion(&mut self, src: &Frame) -> bool {
		let (w, h) = (src.w as usize, src.h as usize);
		if w == 0 || h == 0 {
			return false;
		}
		let sw = SMALL;
		let sh = ((SMALL * h / w) / BLOCK * BLOCK).max(BLOCK);
		std::mem::swap(&mut self.small, &mut self.was);
		self.small.resize(sw * sh, 0);
		self.small.par_chunks_mut(sw).enumerate().for_each(|(y, line)| {
			let sy = (y * h / sh).min(h - 1);
			for (x, v) in line.iter_mut().enumerate() {
				let p = &src.rgba[(sy * w + (x * w / sw).min(w - 1)) * 4..];
				*v = ((p[0] as u32 * 54 + p[1] as u32 * 183 + p[2] as u32 * 19) >> 8) as u8;
			}
		});
		let (cols, rows) = (sw / BLOCK, sh / BLOCK);
		if self.was.len() != self.small.len() || self.sw != sw || self.sh != sh {
			self.sw = sw;
			self.sh = sh;
			self.moved = vec![(0, 0); cols * rows];
			return false;
		}
		let (now, was) = (&self.small, &self.was);
		let differ: u64 =
			now.par_iter().zip(was.par_iter()).map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64).sum();
		let cut = differ as f64 / now.len() as f64 > 40.0;
		// each block: the nearby place in the last frame it's most like, standing still unless
		// moving is clearly better (flat sky would otherwise wander)
		self.moved = (0..cols * rows)
			.into_par_iter()
			.map(|b| {
				let (bx, by) = ((b % cols * BLOCK) as i32, (b / cols * BLOCK) as i32);
				let cost = |dx: i32, dy: i32| -> u32 {
					let mut s = 0u32;
					for y in 0..BLOCK as i32 {
						let (yy, py) = (by + y, (by + y + dy).clamp(0, sh as i32 - 1));
						for x in 0..BLOCK as i32 {
							let px = (bx + x + dx).clamp(0, sw as i32 - 1);
							s += (now[(yy * sw as i32 + bx + x) as usize] as i32
								- was[(py * sw as i32 + px) as usize] as i32)
								.unsigned_abs();
						}
					}
					s
				};
				let mut best = (cost(0, 0) as f32 * 0.9, 0i8, 0i8);
				for dy in -REACH..=REACH {
					for dx in -REACH..=REACH {
						if dx == 0 && dy == 0 {
							continue;
						}
						let c = cost(dx, dy) as f32 + (dx.abs() + dy.abs()) as f32 * 4.0;
						if c < best.0 {
							best = (c, dx as i8, dy as i8);
						}
					}
				}
				(best.1, best.2)
			})
			.collect();
		cut
	}

	/// Push the held picture along by each block's motion, and draw `fresh` of the blocks anew from
	/// `out`.
	fn push(&mut self, out: &Frame, fresh: f32) {
		let (w, h) = (out.w as usize, out.h as usize);
		let (cols, rows) = (self.sw / BLOCK, self.sh / BLOCK);
		if self.moved.len() != cols * rows || cols == 0 {
			return;
		}
		let (kx, ky) = (w as f32 / self.sw as f32, h as f32 / self.sh as f32);
		let old = std::mem::take(&mut self.held.rgba);
		let (moved, frames) = (&self.moved, self.frames);
		let mut new = vec![0u8; old.len()];
		new.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
			let r = ((y as f32 / ky) as usize / BLOCK).min(rows - 1);
			for x in 0..w {
				let c = ((x as f32 / kx) as usize / BLOCK).min(cols - 1);
				let b = r * cols + c;
				let i = x * 4;
				// some blocks come through as they are now
				if hash(b as i64 * 7919 + frames as i64, 2203) < fresh {
					line[i..i + 4].copy_from_slice(&out.rgba[(y * w + x) * 4..][..4]);
					continue;
				}
				let (dx, dy) = moved[b];
				let sx = (x as f32 + dx as f32 * kx).round().clamp(0.0, (w - 1) as f32) as usize;
				let sy = (y as f32 + dy as f32 * ky).round().clamp(0.0, (h - 1) as f32) as usize;
				line[i..i + 4].copy_from_slice(&old[(sy * w + sx) * 4..][..4]);
			}
		});
		self.held.rgba = new;
	}

	/// After everything else: trichrome, stutter and tear, as the look says. `t` is the source's
	/// seconds, `hit` whether a beat has come (None: there are no beats).
	pub fn after(
		&mut self,
		out: &mut Frame,
		l: &Look,
		t: f64,
		dt: f64,
		hit: Option<bool>,
		fresh: bool,
		on: Option<(f32, f32)>,
	) {
		self.kaleidoscope(out, l, t, fresh, on);
		self.trichrome(out, l, t, fresh);
		self.stutter(out, l, dt, hit, fresh);
		tear(out, l, t);
	}

	/// A kaleidoscope of `l.kaleido` wedges, each the first one again (every other one mirrored), turning
	/// `l.kaleido_spin` times a minute round `on` (0-1 across and down: the biggest thing on show),
	/// which it eases after rather than jumping to.
	fn kaleidoscope(&mut self, out: &mut Frame, l: &Look, t: f64, fresh: bool, on: Option<(f32, f32)>) {
		let n = l.kaleido.round();
		if n < 2.0 || out.w == 0 {
			self.middle = None;
			return;
		}
		let target = on.unwrap_or((0.5, 0.5));
		let m = match self.middle {
			Some((x, y)) if fresh => (x + (target.0 - x) * 0.12, y + (target.1 - y) * 0.12),
			Some(m) => m,
			None => target,
		};
		self.middle = Some(m);
		let (w, h) = (out.w as usize, out.h as usize);
		let (cx, cy) = (m.0 * w as f32, m.1 * h as f32);
		let wedge = std::f32::consts::TAU / n;
		let turn = (t as f32 / 60.0 * l.kaleido_spin * std::f32::consts::TAU) % std::f32::consts::TAU;
		let copy = out.rgba.clone();
		// a point off the picture comes back in, as if the picture were mirrored at its edges
		let fold = |v: f32, size: f32| {
			let p = v.rem_euclid(2.0 * size);
			if p < size { p } else { 2.0 * size - p - 1.0 }
		};
		out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
			let dy = y as f32 + 0.5 - cy;
			for x in 0..w {
				let dx = x as f32 + 0.5 - cx;
				let r = dx.hypot(dy);
				let a = (dy.atan2(dx) - turn).rem_euclid(wedge);
				let a = if a > wedge / 2.0 { wedge - a } else { a } + turn;
				let sx = fold(cx + r * a.cos(), w as f32) as usize;
				let sy = fold(cy + r * a.sin(), h as f32) as usize;
				let j = (sy.min(h - 1) * w + sx.min(w - 1)) * 4;
				line[x * 4..x * 4 + 3].copy_from_slice(&copy[j..j + 3]);
			}
		});
	}

	fn trichrome(&mut self, out: &mut Frame, l: &Look, t: f64, fresh: bool) {
		let d = (l.trichrome as f64).min(1.0);
		if d <= 0.0 {
			self.past.clear();
			return;
		}
		if fresh {
			self.past.push_back((t, out.clone()));
			while self.past.len() > 2
				&& (self.past.front().is_some_and(|p| p.0 < t - 2.0 * d - 0.2) || self.past.len() > 90)
			{
				self.past.pop_front();
			}
		}
		let at =
			|when: f64| self.past.iter().min_by(|a, b| (a.0 - when).abs().total_cmp(&(b.0 - when).abs())).map(|p| &p.1);
		let (Some(red), Some(green)) = (at(t - 2.0 * d), at(t - d)) else { return };
		if red.w != out.w || red.h != out.h || green.w != out.w || green.h != out.h {
			return;
		}
		out.rgba.par_chunks_mut(4).zip(red.rgba.par_chunks(4)).zip(green.rgba.par_chunks(4)).for_each(|((o, r), g)| {
			o[0] = r[0];
			o[1] = g[1];
		});
	}

	fn stutter(&mut self, out: &mut Frame, l: &Look, dt: f64, hit: Option<bool>, fresh: bool) {
		if l.stutter <= 0.0 {
			self.recent.clear();
			self.looping = None;
			return;
		}
		let len = (l.stutter_len.round() as usize).clamp(2, 16);
		if fresh {
			self.frames += 1;
			if self.looping.is_none() {
				// on a beat (some of them), or now and then without any
				let chance = hash(self.frames as i64, 6113);
				let start = match hit {
					Some(beat) => beat && chance < l.stutter,
					None => chance < (l.stutter as f64 * dt * 2.0) as f32,
				};
				if start && self.recent.len() >= len {
					let times = 1 + (hash(self.frames as i64, 991) * 3.0) as usize;
					self.looping = Some((self.recent.len() - len, len, 0, times));
				} else {
					self.recent.push_back(out.clone());
					while self.recent.len() > 16 {
						self.recent.pop_front();
					}
					return;
				}
			} else if let Some(lp) = self.looping.as_mut() {
				lp.2 += 1;
				if lp.2 >= lp.1 * lp.3 {
					self.looping = None;
					self.recent.push_back(out.clone());
					while self.recent.len() > 16 {
						self.recent.pop_front();
					}
					return;
				}
			}
		}
		if let Some((first, n, k, _)) = self.looping
			&& let Some(f) = self.recent.get(first + k % n)
			&& f.w == out.w
			&& f.h == out.h
		{
			out.rgba.copy_from_slice(&f.rgba);
		}
	}
}

/// A worn tape, `l.tear` strong: each row slides sideways (a slow wobble, a tracking tear that rolls
/// down the picture, the head switching at the bottom), red and blue bleed apart, every other row's a
/// little darker, and snow in the tear.
pub fn tear(out: &mut Frame, l: &Look, t: f64) {
	let s = l.tear.clamp(0.0, 1.0);
	if s <= 0.0 || out.w == 0 {
		return;
	}
	let (w, h) = (out.w as usize, out.h as usize);
	let unit = w.min(h) as f32 / 576.0;
	let copy = out.rgba.clone();
	let tt = t as f32;
	let tick = (t * 30.0).floor() as i64;
	// the tear: where it is (rolling down, now and then jumping), how tall
	let roll = (t * 0.09).floor() as i64;
	let band = ((t * 0.09).fract() as f32 + hash(roll, 4021)).fract();
	let tall = 0.03 + hash(roll, 77) * 0.04;
	let bleed = (7.0 * s * unit).round() as isize;
	out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
		let yf = y as f32 / h as f32;
		let mut off = ((yf * 13.0 + tt * 1.7).sin() * 0.6 + (yf * 41.0 - tt * 3.1).sin() * 0.3) * 6.0 * s * unit;
		off += (hash(y as i64 + tick * 4099, 13) - 0.5) * 1.5 * s * unit;
		let into = (yf - band).abs() < tall / 2.0;
		if into {
			off += (8.0 + (hash(y as i64 / 2 + tick * 31, 7) - 0.5) * 50.0) * s * unit;
		}
		let bottom = 1.0 - (h - y) as f32 / (h as f32 * 0.025);
		if bottom > 0.0 {
			off += bottom * 40.0 * s * unit * (0.6 + hash(tick, y as i64) * 0.4);
		}
		let off = off.round() as isize;
		let dark = if y % 2 == 1 { 1.0 - 0.15 * s } else { 1.0 };
		let row = &copy[y * w * 4..(y + 1) * w * 4];
		// soft, as tape is: each colour a little blurred along the row
		let at = |x: isize, c: usize| {
			let p = |x: isize| row[(x.clamp(0, w as isize - 1) as usize) * 4 + c] as f32;
			(p(x - 1) + 2.0 * p(x) + p(x + 1)) / 4.0
		};
		// a dropout: a white streak where the tape lost its coating
		let drop = (hash(y as i64 + tick * 7451, 17) > 1.0 - 0.006 * s).then(|| {
			let a = (hash(y as i64, tick) * w as f32) as usize;
			(a, a + (hash(tick, y as i64 + 3) * w as f32 * 0.4) as usize)
		});
		for x in 0..w {
			let sx = x as isize - off;
			let i = x * 4;
			let mut v = [at(sx - bleed, 0), at(sx, 1), at(sx + bleed, 2)];
			// colours faded
			let luma = 0.3 * v[0] + 0.59 * v[1] + 0.11 * v[2];
			for c in &mut v {
				*c += (luma - *c) * 0.3 * s;
			}
			if (into && hash(x as i64 + y as i64 * 7907, tick) > 0.97) || drop.is_some_and(|(a, b)| x >= a && x < b) {
				v = [235.0; 3];
			}
			for c in 0..3 {
				line[i + c] = (v[c] * dark).round().clamp(0.0, 255.0) as u8;
			}
		}
	});
}

/// JPEG's tables for brightness and colour, at quality 50.
const LUMA_Q: [u16; 64] = [
	16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51,
	87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101,
	72, 92, 95, 98, 112, 100, 103, 99,
];
const CHROMA_Q: [u16; 64] = [
	17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99,
	99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
	99, 99, 99, 99, 99, 99,
];

/// The 8-point DCT's basis: COS[u][x].
fn basis() -> &'static [[f32; 8]; 8] {
	static B: std::sync::OnceLock<[[f32; 8]; 8]> = std::sync::OnceLock::new();
	B.get_or_init(|| {
		let mut b = [[0.0; 8]; 8];
		for (u, row) in b.iter_mut().enumerate() {
			let c = if u == 0 { (0.125f32).sqrt() } else { 0.5 };
			for (x, v) in row.iter_mut().enumerate() {
				*v = c * (((2 * x + 1) as f32 * u as f32 * std::f32::consts::PI) / 16.0).cos();
			}
		}
		b
	})
}

/// One 8 by 8 block through JPEG: into frequencies, rounded to `table`'s steps, and back.
fn through(block: &mut [f32; 64], table: &[f32; 64]) {
	let b = basis();
	let mut tmp = [0.0f32; 64];
	// rows, then columns
	for y in 0..8 {
		for u in 0..8 {
			tmp[y * 8 + u] = (0..8).map(|x| b[u][x] * block[y * 8 + x]).sum();
		}
	}
	let mut f = [0.0f32; 64];
	for v in 0..8 {
		for u in 0..8 {
			f[v * 8 + u] = (0..8).map(|y| b[v][y] * tmp[y * 8 + u]).sum();
		}
	}
	for (k, q) in f.iter_mut().zip(table) {
		*k = (*k / q).round() * q;
	}
	for y in 0..8 {
		for u in 0..8 {
			tmp[y * 8 + u] = (0..8).map(|v| b[v][y] * f[v * 8 + u]).sum();
		}
	}
	for y in 0..8 {
		for x in 0..8 {
			block[y * 8 + x] = (0..8).map(|u| b[u][x] * tmp[y * 8 + u]).sum();
		}
	}
}

/// A picture (RGBA, w by h) saved as a JPEG `amount` crushed (1 is quality 1) and opened again:
/// colour at a quarter of the detail, in 8 by 8 blocks rounded hard, so it goes blocky, rings at its
/// edges and its colours smear.
pub fn crush(rgba: &[u8], w: usize, h: usize, amount: f32) -> Vec<u8> {
	let q = (1.0 + (1.0 - amount.clamp(0.0, 1.0)) * 40.0).round();
	let scale = if q < 50.0 { 5000.0 / q } else { 200.0 - 2.0 * q };
	let table = |t: &[u16; 64]| -> [f32; 64] {
		std::array::from_fn(|i| ((t[i] as f32 * scale + 50.0) / 100.0).floor().clamp(1.0, 255.0))
	};
	let (lq, cq) = (table(&LUMA_Q), table(&CHROMA_Q));
	// brightness at full size, colour at half, each padded to whole blocks
	let (bw, bh) = (w.div_ceil(8), h.div_ceil(8));
	let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
	let (cbw, cbh) = (cw.div_ceil(8), ch.div_ceil(8));
	let px = |x: usize, y: usize| {
		let p = &rgba[(y.min(h - 1) * w + x.min(w - 1)) * 4..];
		(p[0] as f32, p[1] as f32, p[2] as f32)
	};
	let ycc = |(r, g, b): (f32, f32, f32)| {
		(0.299 * r + 0.587 * g + 0.114 * b, -0.1687 * r - 0.3313 * g + 0.5 * b, 0.5 * r - 0.4187 * g - 0.0813 * b)
	};
	let luma: Vec<[f32; 64]> = (0..bw * bh)
		.into_par_iter()
		.map(|k| {
			let (bx, by) = (k % bw * 8, k / bw * 8);
			let mut b: [f32; 64] = std::array::from_fn(|i| ycc(px(bx + i % 8, by + i / 8)).0 - 128.0);
			through(&mut b, &lq);
			b
		})
		.collect();
	let chroma: Vec<([f32; 64], [f32; 64])> = (0..cbw * cbh)
		.into_par_iter()
		.map(|k| {
			let (bx, by) = (k % cbw * 8, k / cbw * 8);
			let mut cb = [0.0f32; 64];
			let mut cr = [0.0f32; 64];
			for i in 0..64 {
				let (x, y) = ((bx + i % 8) * 2, (by + i / 8) * 2);
				let mut s = (0.0, 0.0);
				for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
					let c = ycc(px(x + dx, y + dy));
					s.0 += c.1 / 4.0;
					s.1 += c.2 / 4.0;
				}
				cb[i] = s.0;
				cr[i] = s.1;
			}
			through(&mut cb, &cq);
			through(&mut cr, &cq);
			(cb, cr)
		})
		.collect();
	let mut out = vec![255u8; w * h * 4];
	out.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
		for x in 0..w {
			let yy = luma[(y / 8) * bw + x / 8][(y % 8) * 8 + x % 8] + 128.0;
			let (cx, cy) = (x / 2, y / 2);
			let (cb, cr) = &chroma[(cy / 8) * cbw + cx / 8];
			let (u, v) = (cb[(cy % 8) * 8 + cx % 8], cr[(cy % 8) * 8 + cx % 8]);
			let rgb = [yy + 1.402 * v, yy - 0.344_136 * u - 0.714_136 * v, yy + 1.772 * u];
			for c in 0..3 {
				line[x * 4 + c] = rgb[c].round().clamp(0.0, 255.0) as u8;
			}
		}
	});
	out
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_crushed_picture_keeps_its_colours_roughly_and_goes_blocky() {
		// a smooth ramp, red to blue
		let (w, h) = (32, 16);
		let rgba: Vec<u8> = (0..w * h).flat_map(|i| [(i % w * 8) as u8, 60, (255 - i % w * 8) as u8, 255]).collect();
		let gentle = crush(&rgba, w, h, 0.0);
		let hard = crush(&rgba, w, h, 1.0);
		let err = |o: &[u8]| {
			o.iter().zip(&rgba).map(|(a, b)| (*a as i32 - *b as i32).abs()).sum::<i32>() as f32 / rgba.len() as f32
		};
		assert!(err(&gentle) < 6.0, "{}", err(&gentle));
		assert!(err(&hard) > 2.0 * err(&gentle), "{} {}", err(&hard), err(&gentle));
		// colour is kept at half the detail: neighbouring pairs agree
		assert_eq!(hard[0], hard[4]);
		assert_eq!(hard.len(), rgba.len());
	}

	#[test]
	fn mosh_follows_the_motion() {
		// a bright square moving right 2 small pixels a frame; held on, it's pushed along, not redrawn
		let (w, h) = (192usize, 96usize);
		let frame = |at: usize| {
			let mut f = Frame::new(w as u32, h as u32);
			for y in 40..56 {
				for x in at..at + 16 {
					f.rgba[(y * w + x) * 4..][..3].copy_from_slice(&[255, 255, 255]);
				}
			}
			f
		};
		let l = Look { mosh: "always".into(), mosh_where: "all".into(), mosh_hold: 1000.0, ..Look::default() };
		let mut g = Glitch::default();
		let mut out = frame(40);
		g.mosh(&mut out, &frame(40), &l, None, false, 1.0 / 30.0, true);
		g.mosh(&mut out, &frame(40), &l, None, false, 1.0 / 30.0, true);
		// the next frame's square has moved; what's shown is the old picture pushed by the motion
		let mut out = Frame::new(w as u32, h as u32);
		g.mosh(&mut out, &frame(42), &l, None, false, 1.0 / 30.0, true);
		let lit = |f: &Frame, x: usize| f.rgba[(48 * w + x) * 4] > 128;
		assert!(lit(&out, 42) && lit(&out, 57) && !lit(&out, 60), "the square went with the motion");
	}
}
