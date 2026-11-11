//! Echo: time in the picture. The things followed leave copies of themselves behind as they go, a
//! crisp cut-out every so often that fades as it ages (Marey's chronophotographs); or a long
//! exposure of them alone, smearing light while the world stays sharp; or "motion extraction", the
//! picture laid over an inverted copy of itself from a moment ago, so anything still turns to flat
//! grey and only what moved shows; or a slit-scan, each row of the world from a moment further back
//! than the one above it, so whatever moves through it shears and stretches, while the things
//! followed stay as they are now.
//!
//! Copies ride along with the camera's own movement, as the tracks do, so on a handheld shot they
//! stay where they were in the scene rather than where they were in the frame. The long exposure and
//! the past pictures for extract and slit-scan are lined up with the picture now the same way (by
//! the camera's whole movement), so on a handheld shot what stood still in the scene still cancels
//! out, and only what really moved shows.

use crate::Frame;
use crate::Rect;
use crate::look::Look;
use crate::matte::Matte;
use crate::subject::Subject;
use rayon::prelude::*;
use std::collections::VecDeque;

struct Sprite {
	/// its middle, in track units, so the camera's movement can carry it
	cx: f32,
	cy: f32,
	w: usize,
	h: usize,
	/// premultiplied by the matte
	rgba: Vec<u8>,
	/// output seconds
	born: f64,
}

#[derive(Default)]
pub struct Echo {
	sprites: Vec<Sprite>,
	/// output seconds when copies were last taken
	last: Option<f64>,
	/// the long exposure so far, RGB a pixel
	acc: Vec<f32>,
	acc_at: f64,
	/// where the camera had got to when the exposure was last lined up (track units)
	acc_cam: (f32, f32),
	/// past pictures for motion extraction and slit-scan, with their output seconds and where the
	/// camera had got to then
	past: VecDeque<(f64, (f32, f32), Vec<u8>)>,
	/// how far the camera has moved, all told, in track units
	cam: (f32, f32),
}

/// Whether a look echoes.
pub fn active(l: &Look) -> bool {
	l.echo >= 1.0
}

impl Echo {
	/// After a jump in the picture: what went before is gone.
	pub fn clear(&mut self) {
		self.sprites.clear();
		self.last = None;
		self.acc.clear();
		self.past.clear();
	}

	/// The camera moved: move the copies with it. `unit` is the movement finder's pixels a track unit.
	pub fn carry(&mut self, sub: &Subject, unit: f32) {
		for s in &mut self.sprites {
			sub.carry(&mut s.cx, &mut s.cy, unit);
		}
		let (dx, dy) = sub.shift();
		self.cam.0 += dx / unit.max(1e-6);
		self.cam.1 += dy / unit.max(1e-6);
	}

	/// Take copies when they're due (on a new frame) and draw the echo into `out`. `boxes` are in output
	/// pixels, `m` their matte, `clock` output seconds, `per_unit` output pixels a track unit; `beat`,
	/// with copies on the music, says whether a beat has just come.
	pub fn apply(
		&mut self,
		out: &mut Frame,
		m: &Matte,
		boxes: &[Rect],
		l: &Look,
		clock: f64,
		fresh: bool,
		per_unit: f32,
		beat: Option<bool>,
	) {
		let (w, h) = (out.w as usize, out.h as usize);
		let every = (l.echo_every as f64).max(0.02);
		let life = l.echo.round() as f64 * every;
		match l.echo_mode.as_str() {
			"exposure" => {
				if self.acc.len() != w * h * 3 {
					self.acc = vec![0.0; w * h * 3];
					self.acc_at = clock;
					self.acc_cam = self.cam;
				}
				// the exposure moved with the camera, whole pixels at a time
				let (ox, oy) = (
					((self.cam.0 - self.acc_cam.0) * per_unit).round() as isize,
					((self.cam.1 - self.acc_cam.1) * per_unit).round() as isize,
				);
				if (ox != 0 || oy != 0) && fresh {
					self.acc = shifted(&self.acc, w, h, 3, ox, oy);
					self.acc_cam.0 += ox as f32 / per_unit;
					self.acc_cam.1 += oy as f32 / per_unit;
				}
				if fresh {
					let dt = (clock - self.acc_at).max(0.0);
					self.acc_at = clock;
					let keep = (-dt / life.max(0.05)).exp() as f32;
					let fade = 1.0 - l.echo_fade.clamp(0.0, 1.0) * (1.0 - keep);
					self.acc.par_chunks_mut(w * 3).zip(out.rgba.par_chunks(w * 4)).enumerate().for_each(
						|(y, (acc, line))| {
							for x in 0..w {
								let a = m.at(x, y);
								for c in 0..3 {
									let v = line[x * 4 + c] as f32 * a;
									let i = x * 3 + c;
									acc[i] = (acc[i] * fade).max(v);
								}
							}
						},
					);
				}
				// lighten: the trails show where they're brighter than what's there now
				out.rgba.par_chunks_mut(w * 4).zip(self.acc.par_chunks(w * 3)).for_each(|(line, acc)| {
					for x in 0..w {
						for c in 0..3 {
							let v = acc[x * 3 + c].round().clamp(0.0, 255.0) as u8;
							if v > line[x * 4 + c] {
								line[x * 4 + c] = v;
							}
						}
					}
				});
			}
			"slit" => {
				// the world down the frame from now to `life` seconds ago, a row at a time
				let span = life.max(0.05);
				if fresh {
					self.past.push_back((clock, self.cam, out.rgba.clone()));
					// as many pictures as that takes, and no more than about 400 MB of them
					let most = (400_000_000 / out.rgba.len().max(1)).max(2);
					while self.past.len() > most || (self.past.len() > 2 && self.past[1].0 <= clock - span) {
						self.past.pop_front();
					}
				}
				if self.past.front().is_none_or(|p| p.2.len() != out.rgba.len()) {
					self.past.clear();
					return;
				}
				let past = &self.past;
				let mix = l.echo_fade.clamp(0.0, 1.0);
				let cam = self.cam;
				let now = out.rgba.clone();
				out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
					let want = clock - span * y as f64 / h.max(1) as f64;
					// the picture nearest that moment, lined up with now
					let i = past.partition_point(|p| p.0 < want).min(past.len() - 1);
					let i = if i > 0 && (past[i - 1].0 - want).abs() < (past[i].0 - want).abs() { i - 1 } else { i };
					let (ox, oy) = (
						((cam.0 - past[i].1.0) * per_unit).round() as isize,
						((cam.1 - past[i].1.1) * per_unit).round() as isize,
					);
					let sy = y as isize - oy;
					for x in 0..w {
						// the things stay as they are now; the world is from then
						let k = (1.0 - m.at(x, y)) * mix;
						if k <= 0.0 {
							continue;
						}
						let sx = x as isize - ox;
						let then = if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
							&past[i].2[(sy as usize * w + sx as usize) * 4..][..3]
						} else {
							&now[(y * w + x) * 4..][..3]
						};
						for c in 0..3 {
							let p = &mut line[x * 4 + c];
							*p = (*p as f32 + (then[c] as f32 - *p as f32) * k).round() as u8;
						}
					}
				});
			}
			"extract" => {
				if fresh {
					self.past.push_back((clock, self.cam, out.rgba.clone()));
					while self.past.len() > 2 && self.past[1].0 <= clock - every {
						self.past.pop_front();
					}
				}
				let Some((_, then_cam, then)) = self.past.front() else { return };
				if then.len() != out.rgba.len() {
					self.past.clear();
					return;
				}
				// the picture from a moment ago, lined up with now by how far the camera's moved since
				let (ox, oy) = (
					((self.cam.0 - then_cam.0) * per_unit).round() as isize,
					((self.cam.1 - then_cam.1) * per_unit).round() as isize,
				);
				// now, over that picture turned inside out: still is grey
				let mix = l.echo_fade.clamp(0.0, 1.0);
				out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
					let sy = y as isize - oy;
					for x in 0..w {
						let sx = x as isize - ox;
						// where the camera's brought in what wasn't in the picture then, nothing to take away: grey
						let q = if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
							Some(&then[(sy as usize * w + sx as usize) * 4..][..3])
						} else {
							None
						};
						let p = &mut line[x * 4..x * 4 + 3];
						for c in 0..3 {
							// the difference at full strength, about grey (the classic half-and-half is faint)
							let e = match q {
								Some(q) => (128.0 + p[c] as f32 - q[c] as f32).clamp(0.0, 255.0),
								None => 128.0,
							};
							p[c] = (p[c] as f32 + (e - p[c] as f32) * mix).round() as u8;
						}
					}
				});
			}
			_ => {
				// copies, or fainter grey ghosts of them
				if fresh {
					self.sprites.retain(|s| clock - s.born < life + 1e-6);
					let due = match beat {
						Some(hit) => hit,
						None => self.last.is_none_or(|t| clock - t >= every - 1e-6),
					};
					if due {
						self.last = Some(clock);
						for b in boxes {
							if let Some(s) = take(out, m, *b, clock, per_unit) {
								self.sprites.push(s);
							}
						}
					}
				}
				if self.sprites.is_empty() {
					return;
				}
				let live = out.rgba.clone();
				let ghosts = l.echo_mode == "ghosts";
				// each copy where it goes and how strongly, oldest first so the newer lie over them
				let copies: Vec<(&Sprite, f32, isize, isize)> = self
					.sprites
					.iter()
					.filter_map(|s| {
						let age = ((clock - s.born) / life.max(1e-3)).clamp(0.0, 1.0) as f32;
						let opacity = (1.0 - age * l.echo_fade.clamp(0.0, 1.0)) * if ghosts { 0.45 } else { 1.0 };
						let x0 = (s.cx * per_unit - s.w as f32 / 2.0).round() as isize;
						let y0 = (s.cy * per_unit - s.h as f32 / 2.0).round() as isize;
						(opacity > 0.01).then_some((s, opacity, x0, y0))
					})
					.collect();
				// a row of the picture at a time, every copy that crosses it, then the things as they are
				// now back on top of their own echoes
				out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
					for &(s, opacity, x0, y0) in &copies {
						let sy = y as isize - y0;
						if sy < 0 || sy >= s.h as isize {
							continue;
						}
						let src = &s.rgba[sy as usize * s.w * 4..][..s.w * 4];
						let (from, to) = ((-x0).max(0) as usize, (w as isize - x0).clamp(0, s.w as isize) as usize);
						for sx in from..to {
							let q = &src[sx * 4..sx * 4 + 4];
							let a = q[3] as f32 / 255.0 * opacity;
							if a <= 0.004 {
								continue;
							}
							let p = &mut line[(x0 + sx as isize) as usize * 4..][..3];
							let grey = (q[0] as f32 * 0.2126 + q[1] as f32 * 0.7152 + q[2] as f32 * 0.0722) as u8;
							for c in 0..3 {
								let v = if ghosts { grey } else { q[c] } as f32 * opacity;
								p[c] = (p[c] as f32 * (1.0 - a) + v).round().clamp(0.0, 255.0) as u8;
							}
						}
					}
					let now = &live[y * w * 4..][..w * 4];
					for b in boxes {
						if (y as f32) < b.y.floor() || (y as f32) >= (b.y + b.h).ceil() {
							continue;
						}
						let x0 = (b.x.floor().max(0.0) as usize).min(w);
						let x1 = ((b.x + b.w).ceil().max(0.0) as usize).min(w);
						for x in x0..x1 {
							let a = m.at(x, y);
							if a <= 0.004 {
								continue;
							}
							for c in 0..3 {
								let i = x * 4 + c;
								line[i] = (line[i] as f32 * (1.0 - a) + now[i] as f32 * a).round() as u8;
							}
						}
					}
				});
			}
		}
	}
}

/// A cut-out of the picture through the matte, over a box.
fn take(out: &Frame, m: &Matte, b: Rect, clock: f64, per_unit: f32) -> Option<Sprite> {
	let (w, h) = (out.w as usize, out.h as usize);
	let pad = b.w.max(b.h) * 0.15;
	let x0 = ((b.x - pad).floor().max(0.0) as usize).min(w);
	let x1 = ((b.x + b.w + pad).ceil().max(0.0) as usize).min(w);
	let y0 = ((b.y - pad).floor().max(0.0) as usize).min(h);
	let y1 = ((b.y + b.h + pad).ceil().max(0.0) as usize).min(h);
	let (sw, sh) = (x1.checked_sub(x0)?, y1.checked_sub(y0)?);
	if sw < 2 || sh < 2 {
		return None;
	}
	let mut rgba = vec![0u8; sw * sh * 4];
	let mut any = false;
	for y in 0..sh {
		for x in 0..sw {
			let a = m.at(x0 + x, y0 + y);
			if a <= 0.02 {
				continue;
			}
			any = true;
			let p = &out.rgba[((y0 + y) * w + x0 + x) * 4..][..3];
			let q = &mut rgba[(y * sw + x) * 4..][..4];
			for c in 0..3 {
				q[c] = (p[c] as f32 * a).round() as u8;
			}
			q[3] = (a * 255.0).round() as u8;
		}
	}
	any.then(|| Sprite {
		cx: (x0 as f32 + sw as f32 / 2.0) / per_unit,
		cy: (y0 as f32 + sh as f32 / 2.0) / per_unit,
		w: sw,
		h: sh,
		rgba,
		born: clock,
	})
}

/// A w by h picture of `n` values a pixel moved by (ox, oy) pixels; what comes in at the edges is 0.
fn shifted(a: &[f32], w: usize, h: usize, n: usize, ox: isize, oy: isize) -> Vec<f32> {
	let mut out = vec![0.0; a.len()];
	out.par_chunks_mut(w * n).enumerate().for_each(|(y, line)| {
		let sy = y as isize - oy;
		if sy < 0 || sy >= h as isize {
			return;
		}
		let src = &a[sy as usize * w * n..][..w * n];
		for x in 0..w {
			let sx = x as isize - ox;
			if sx >= 0 && (sx as usize) < w {
				line[x * n..x * n + n].copy_from_slice(&src[sx as usize * n..sx as usize * n + n]);
			}
		}
	});
	out
}
