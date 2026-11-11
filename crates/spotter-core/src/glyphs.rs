//! Glyphs: the picture as characters. A grid of cells over the frame, each a character from a ramp
//! picked by how bright that part of the picture is, in the look's font, on a plain ground: over the
//! things followed, over the world around them, or over all of it.
//!
//! Or words: the picture becomes a sheet of paper, and each thing is typed out as its own name over and
//! over, BOAT BOAT BOAT, filling its shape, the rows set off from each other as a typewriter's lines
//! would be and the ink a little uneven. The world is left as paper, or typed as what it is where the
//! scene's been labelled.

use crate::Frame;
use crate::Rect;
use crate::look::Look;
use crate::matte::{Matte, bilinear};
use crate::outline::{MASK, Shape};
use crate::overlay::colour;
use crate::text::Face;
use rayon::prelude::*;
use tiny_skia::{FillRule, Paint, Pixmap, Transform};

/// Whether a look draws glyphs.
pub fn active(l: &Look) -> bool {
	l.glyphs != "off" && !l.glyphs.is_empty()
}

/// Whether glyphs need to know where the things are.
pub fn needs_matte(l: &Look) -> bool {
	l.glyphs == "things" || l.glyphs == "world" || l.glyphs == "words"
}

/// A thing to type out: where it is (output pixels), what it's called, and its shape if there is one.
pub struct Word<'a> {
	pub rect: Rect,
	pub word: String,
	pub shape: Option<&'a Shape>,
}

/// A number from a cell, the same every time, 0-1.
fn hash(a: usize, b: usize, salt: u32) -> f32 {
	let mut h =
		(a as u32).wrapping_mul(0x9E37_79B1) ^ (b as u32).wrapping_mul(0x85EB_CA77) ^ salt.wrapping_mul(0xC2B2_AE3D);
	h ^= h >> 15;
	h = h.wrapping_mul(0x2C1B_3C6D);
	h ^= h >> 12;
	(h & 0xFFFF) as f32 / 65535.0
}

#[derive(Default)]
pub struct Glyphs {
	/// what the cached characters were made for: font, size, ramp
	made: String,
	cw: usize,
	ch: usize,
	/// each character of the ramp as coverage, cw by ch
	marks: Vec<Vec<u8>>,
	/// for words: each letter as coverage, made as it's first needed, and what they were made for
	letters: std::collections::HashMap<char, Vec<u8>>,
	letters_for: String,
}

impl Glyphs {
	fn make(&mut self, l: &Look, cw: usize, ch: usize) {
		let key = format!("{}|{cw}x{ch}|{}", l.font, l.glyph_ramp);
		if key == self.made {
			return;
		}
		self.made = key;
		self.cw = cw;
		self.ch = ch;
		let face = Face::new(&l.font, ch as f32 * 0.92);
		let mut paint = Paint::default();
		paint.set_color(tiny_skia::Color::WHITE);
		paint.anti_alias = true;
		self.marks = l
			.glyph_ramp
			.chars()
			.map(|c| {
				let mut pm = Pixmap::new(cw as u32, ch as u32).expect("a glyph's size");
				let s = c.to_string();
				let x = (cw as f32 - face.measure(&s)) / 2.0;
				if let Some(p) = face.path(&s, x, face.baseline(ch as f32 / 2.0)) {
					pm.fill_path(&p, &paint, FillRule::Winding, Transform::identity(), None);
				}
				pm.data().chunks_exact(4).map(|p| p[3]).collect()
			})
			.collect();
	}

	/// A letter as coverage, `cw` by `ch`.
	fn letter(&mut self, l: &Look, c: char, cw: usize, ch: usize) -> &[u8] {
		let key = format!("{}|{cw}x{ch}", l.font);
		if key != self.letters_for {
			self.letters.clear();
			self.letters_for = key;
		}
		self.letters.entry(c).or_insert_with(|| {
			let face = Face::new(&l.font, ch as f32 * 0.92);
			let mut paint = Paint::default();
			paint.set_color(tiny_skia::Color::WHITE);
			paint.anti_alias = true;
			let mut pm = Pixmap::new(cw as u32, ch as u32).expect("a letter's size");
			let s = c.to_string();
			let x = (cw as f32 - face.measure(&s)) / 2.0;
			if let Some(p) = face.path(&s, x, face.baseline(ch as f32 / 2.0)) {
				pm.fill_path(&p, &paint, FillRule::Winding, Transform::identity(), None);
				// a typewriter strikes harder than a screen font draws: thickened a touch
				let stroke = tiny_skia::Stroke { width: (ch as f32 * 0.05).max(0.6), ..Default::default() };
				pm.stroke_path(&p, &paint, &stroke, Transform::identity(), None);
			}
			pm.data().chunks_exact(4).map(|p| p[3]).collect()
		})
	}

	/// Type the things out as their names on paper, over the whole of `out`. `m` is the things' matte;
	/// `world` names the world at an output pixel, where the scene's been labelled.
	pub fn draw_words(
		&mut self,
		out: &mut Frame,
		m: &Matte,
		things: &[Word],
		world: &(dyn Fn(f32, f32) -> Option<String> + Sync),
		l: &Look,
	) {
		let (w, h) = (out.w as usize, out.h as usize);
		let u = w.min(h) as f32 / 1080.0;
		let ch = ((l.glyph_size * u).round() as usize).max(4);
		let cw = ((ch as f32 * 0.6).round() as usize).max(3);
		let lead = ((ch as f32 * 1.15).round() as usize).max(ch);
		let (cols, rows) = (w / cw, h / lead);
		let (ox, oy) = ((w - cols * cw) / 2, (h - rows * lead) / 2);
		let rgb = |hex: &str| {
			let c = colour(hex, 1.0);
			[c.red() * 255.0, c.green() * 255.0, c.blue() * 255.0]
		};
		let (paper, ink) = (rgb(&l.glyph_back), rgb(&l.glyph_color));
		// how dark the picture is in each cell, before it's paper: the world's ink follows it, so plain
		// bright sky is typed faintly and the dark sea hard
		let dark: Vec<f32> = (0..rows * cols)
			.into_par_iter()
			.map(|i| {
				let (r, c) = (i / cols, i % cols);
				let (x0, y0) = (ox + c * cw, oy + r * lead);
				let mut sum = 0.0;
				for (dx, dy) in [(1, 1), (cw - 2, 1), (cw / 2, lead / 2), (1, lead - 2), (cw - 2, lead - 2)] {
					let p = &out.rgba[((y0 + dy.min(lead - 1)) * w + x0 + dx.min(cw - 1)) * 4..][..3];
					sum += (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) / 255.0;
				}
				1.0 - sum / 5.0
			})
			.collect();
		// the paper, with a little grain
		out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
			for x in 0..w {
				let g = (hash(x, y, 7) - 0.5) * 7.0 + (hash(x / 3, y / 3, 11) - 0.5) * 5.0;
				for c in 0..3 {
					line[x * 4 + c] = (paper[c] + g).round().clamp(0.0, 255.0) as u8;
				}
			}
		});
		let (mx, my) = (MASK as f32 / w as f32, MASK as f32 / h as f32);
		// each cell: whose it is (the thing whose shape is surest there, else the smallest box it's in),
		// how much (the matte there), and so which letter
		// the things' names as typed, once
		let typed: Vec<Vec<char>> =
			things.iter().map(|t| t.word.to_uppercase().chars().chain(std::iter::once(' ')).collect()).collect();
		// each row of cells, worked out side by side: for each cell, whose it is (the thing whose shape
		// is surest there, else the smallest box it's in, else the world), how much, and so which letter
		let cells: Vec<Vec<(usize, char, f32)>> = (0..rows)
			.into_par_iter()
			.map(|r| {
				let mut world_typed: Vec<(String, Vec<char>)> = Vec::new();
				let mut row = Vec::new();
				for c in 0..cols {
					let (px, py) = ((ox + c * cw) as f32 + cw as f32 / 2.0, (oy + r * lead) as f32 + lead as f32 / 2.0);
					let a = m.at(px as usize, py as usize);
					let mut best: Option<(f32, usize)> = None;
					if a > 0.15 {
						for (i, t) in things.iter().enumerate() {
							let b = t.rect;
							let pad = b.w.max(b.h) * 0.15;
							if px < b.x - pad || px > b.x + b.w + pad || py < b.y - pad || py > b.y + b.h + pad {
								continue;
							}
							let v = match t.shape {
								Some(s) => bilinear(&s.logits, MASK, MASK, px * mx - 0.5, py * my - 0.5),
								None if px >= b.x && px <= b.x + b.w && py >= b.y && py <= b.y + b.h => {
									-(b.w * b.h) / (w * h) as f32
								}
								None => continue,
							};
							if best.is_none_or(|q| v > q.0) {
								best = Some((v, i));
							}
						}
					}
					let (text, cover): (&[char], f32) = match best {
						Some((_, i)) => (&typed[i], a),
						None => match world(px, py) {
							Some(word) => {
								let k = match world_typed.iter().position(|w| w.0 == word) {
									Some(k) => k,
									None => {
										let t = word.to_uppercase().chars().chain(std::iter::once(' ')).collect();
										world_typed.push((word, t));
										world_typed.len() - 1
									}
								};
								(
									&world_typed[k].1,
									(1.0 - a)
										* l.world_ink.clamp(0.0, 1.0) * (0.25 + 0.75 * dark[r * cols + c].powf(0.8)),
								)
							}
							None => continue,
						},
					};
					// each line starts somewhere different along the word, the same for each word every frame
					let shift = (hash(r, text.len(), 3) * text.len() as f32) as usize;
					let ch_at = text[(c + shift) % text.len()];
					if ch_at == ' ' {
						continue;
					}
					// a typewriter's ink: a touch uneven from letter to letter
					let strength = cover.clamp(0.0, 1.0).powf(0.8) * (0.78 + 0.22 * hash(c, r, 5));
					if strength >= 0.04 {
						row.push((c, ch_at, strength));
					}
				}
				row
			})
			.collect();
		// every letter needed, made before the typing starts
		for row in &cells {
			for &(_, letter, _) in row {
				self.letter(l, letter, cw, ch);
			}
		}
		let letters = &self.letters;
		// a line of type at a time, side by side: each covers its own rows of pixels
		let top = oy + (lead - ch) / 2;
		out.rgba[top * w * 4..].par_chunks_mut(lead * w * 4).zip(cells.par_iter()).for_each(|(band, row)| {
			for &(c, letter, strength) in row {
				let Some(mark) = letters.get(&letter) else { continue };
				let left = ox + c * cw;
				for y in 0..ch.min(band.len() / (w * 4)) {
					let line = &mut band[(y * w + left) * 4..][..cw * 4];
					for x in 0..cw {
						let k = mark[y * cw + x] as f32 / 255.0 * strength;
						if k <= 0.0 {
							continue;
						}
						for i in 0..3 {
							let p = &mut line[x * 4 + i];
							*p = (*p as f32 + (ink[i] - *p as f32) * k).round() as u8;
						}
					}
				}
			}
		});
	}

	/// Draw them into `out` (the picture so far), where the look says.
	pub fn draw(&mut self, out: &mut Frame, m: Option<&Matte>, l: &Look) {
		let (w, h) = (out.w as usize, out.h as usize);
		let u = w.min(h) as f32 / 1080.0;
		let ch = ((l.glyph_size * u).round() as usize).max(4);
		let cw = ((ch as f32 * 0.6).round() as usize).max(3);
		self.make(l, cw, ch);
		if self.marks.is_empty() {
			return;
		}
		let ground = colour(&l.glyph_back, 1.0);
		let ground = [ground.red() * 255.0, ground.green() * 255.0, ground.blue() * 255.0];
		let ink = colour(&l.glyph_color, 1.0);
		let ink = [ink.red() * 255.0, ink.green() * 255.0, ink.blue() * 255.0];
		let tinted = l.glyph_tint;
		let (cols, rows) = (w / cw, h / ch);
		let ox = (w - cols * cw) / 2;
		let marks = &self.marks;
		let levels = marks.len();
		let mode = l.glyphs.as_str();
		// a row of cells at a time
		out.rgba.par_chunks_mut(w * 4 * ch).enumerate().for_each(|(r, band)| {
			let top = r * ch;
			if r >= rows || top + ch > h || band.len() < w * 4 * ch {
				return;
			}
			for c in 0..cols {
				let left = ox + c * cw;
				// what's here: inside the things or not, by the cell's middle
				let a = m.map_or(1.0, |m| m.at(left + cw / 2, top + ch / 2));
				let here = match mode {
					"things" => a > 0.5,
					"world" => a <= 0.5,
					_ => true,
				};
				if !here {
					continue;
				}
				let (mut sum, mut col) = (0.0f32, [0.0f32; 3]);
				for y in 0..ch {
					for x in 0..cw {
						let p = &band[(y * w + left + x) * 4..][..3];
						sum += 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
						col[0] += p[0] as f32;
						col[1] += p[1] as f32;
						col[2] += p[2] as f32;
					}
				}
				let n = (cw * ch) as f32;
				let v = (sum / n / 255.0).clamp(0.0, 1.0);
				let mark = &marks[((v * levels as f32) as usize).min(levels - 1)];
				// the ink: the look's colour, or the picture's own there, lifted so it reads
				let ink = if tinted {
					let k = 255.0 / (col.iter().copied().fold(1.0, f32::max) / n).max(1.0);
					[(col[0] / n * k).min(255.0), (col[1] / n * k).min(255.0), (col[2] / n * k).min(255.0)]
				} else {
					ink
				};
				for y in 0..ch {
					for x in 0..cw {
						let cov = mark[y * cw + x] as f32 / 255.0;
						let p = &mut band[(y * w + left + x) * 4..][..3];
						for k in 0..3 {
							p[k] = (ground[k] + (ink[k] - ground[k]) * cov).round() as u8;
						}
					}
				}
			}
		});
	}
}
