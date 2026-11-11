//! The picture under the boxes: brightness, contrast, palettes, grain and the rest of the look's
//! picture settings, applied to every frame. Ported from 1nf.me's `src/lib/tracker/grade.ts`, where
//! it's one WebGL2 fragment shader; here the same sums run on the CPU, a row at a time across
//! every core.
//!
//! The shader's ways are kept: rows and the grain's pixel numbers count from the bottom, as
//! `gl_FragCoord` does, every sample off a pixel's centre is bilinear with the edges held, as the
//! texture's LINEAR and CLAMP_TO_EDGE were, and sizes go by `unit` (the frame's shorter side over
//! 1080) so a look comes out the same at any resolution.

use crate::Frame;
use crate::look::Look;
use rayon::prelude::*;

#[derive(Default)]
pub struct Grade {}

impl Grade {
	pub fn new() -> Self {
		Self::default()
	}

	/// Whether anything in the look changes the picture at all; when it doesn't, `render` can be skipped.
	/// Each test is the one the shader itself makes before doing anything.
	pub fn changes(l: &Look) -> bool {
		l.brightness != 0.0
			|| l.contrast != 1.0
			|| l.saturation != 1.0
			|| l.hue != 0.0
			|| l.sharpen > 0.0
			|| l.solarize > 0.0
			|| l.posterize >= 2.0
			|| l.invert != 0.0
			|| (palette(&l.palette) > 0 && l.palette_mix != 0.0)
			|| l.split > 0.0
			|| l.scanlines > 0.0
			|| l.grain > 0.0
			|| l.vignette > 0.0
			|| l.pixelate > 1.0
			|| l.glitch > 0.0
	}

	/// Grade `src` into `out` (same size). `time` is the video's time in seconds; `fresh` is true for a
	/// new frame rather than the same one again with other settings.
	pub fn render(&mut self, src: &Frame, l: &Look, time: f32, fresh: bool, out: &mut Frame) {
		let _ = fresh;
		let (w, h) = (src.w as usize, src.h as usize);
		out.w = src.w;
		out.h = src.h;
		out.rgba.resize(w * h * 4, 0);
		if w == 0 || h == 0 {
			return;
		}
		let u = Uniforms::new(l, w, h, time);
		let rgba = &src.rgba;
		out.rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| u.row(rgba, y, row));
	}
}

/// u_pal: which palette, 0 for none
fn palette(name: &str) -> u8 {
	match name {
		"duotone" => 1,
		"thermal" => 2,
		"night" => 3,
		_ => 0,
	}
}

/// "#rrggbb" as 0-1 channels, read the way the site's `parseInt(hex.slice(1), 16)` does: whatever
/// follows the first character, up to the first thing that isn't a hex digit, and black if nothing is.
fn rgb(hex: &str) -> [f32; 3] {
	let mut n = 0u32;
	for c in hex.chars().skip(1) {
		let Some(d) = c.to_digit(16) else { break };
		n = n.wrapping_mul(16).wrapping_add(d);
	}
	[((n >> 16) & 255) as f32 / 255.0, ((n >> 8) & 255) as f32 / 255.0, (n & 255) as f32 / 255.0]
}

/// GLSL's mix, in the same form, so a weight of 0 or 1 gives back exactly one side
#[inline]
fn mix(x: f32, y: f32, a: f32) -> f32 {
	x * (1.0 - a) + y * a
}

#[inline]
fn mix3(x: [f32; 3], y: [f32; 3], a: f32) -> [f32; 3] {
	[mix(x[0], y[0], a), mix(x[1], y[1], a), mix(x[2], y[2], a)]
}

/// GLSL's fract, which is from the floor (Rust's `fract` is from zero, and differs below it)
#[inline]
fn fract(x: f32) -> f32 {
	x - x.floor()
}

#[inline]
fn lum(c: [f32; 3]) -> f32 {
	c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

/// the shader's rnd: the old sine hash, only used for which glitch bands jump
fn rnd(x: f32, y: f32) -> f32 {
	// the shader's constant as written there; an f32 holds it as 43758.547
	#[allow(clippy::excessive_precision)]
	const SCALE: f32 = 43758.5453;
	fract((x * 12.9898 + y * 78.233).sin() * SCALE)
}

/// The site's scanlines use 3.14159 rather than π, and the looks are held to the site's pictures.
#[allow(clippy::approx_constant)]
const SITE_PI: f32 = 3.14159;

/// fresh noise per pixel per frame; whatever goes in as the frame has to mix properly, or the same
/// pattern just slides across the picture
#[inline]
fn noise(x: u32, y: u32, frame: u32) -> f32 {
	let mut h = x.wrapping_mul(1597334677) ^ y.wrapping_mul(3812015801) ^ frame.wrapping_mul(2798796415);
	h ^= h >> 15;
	h = h.wrapping_mul(2246822519);
	h ^= h >> 13;
	h = h.wrapping_mul(3266489917);
	h ^= h >> 16;
	h as f32 / 4294967295.0
}

const THERMAL: [[f32; 3]; 5] =
	[[0.02, 0.0, 0.14], [0.38, 0.0, 0.58], [0.92, 0.14, 0.22], [1.0, 0.72, 0.05], [1.0, 1.0, 0.9]];

fn thermal(l: f32) -> [f32; 3] {
	let x = l.clamp(0.0, 1.0) * 4.0;
	let i = x.floor().min(3.0) as usize;
	mix3(THERMAL[i], THERMAL[i + 1], x - i as f32)
}

/// Two rows of the source around a height, for bilinear samples along it.
struct Rows<'a> {
	a: &'a [u8],
	b: &'a [u8],
	t: f32,
	w: usize,
	wf: f32,
}

impl<'a> Rows<'a> {
	/// `v` is the texture's, from the bottom; the frame's rows are from the top
	fn new(src: &'a [u8], w: usize, h: usize, v: f32) -> Self {
		let k = (1.0 - v) * h as f32 - 0.5;
		let k0 = k.floor();
		let top = h as isize - 1;
		let i0 = (k0 as isize).clamp(0, top) as usize;
		let i1 = (k0 as isize + 1).clamp(0, top) as usize;
		Self {
			a: &src[i0 * w * 4..(i0 + 1) * w * 4],
			b: &src[i1 * w * 4..(i1 + 1) * w * 4],
			t: k - k0,
			w,
			wf: w as f32,
		}
	}

	#[inline]
	fn at(&self, u: f32) -> [f32; 3] {
		let k = u * self.wf - 0.5;
		let k0 = k.floor();
		let s = k - k0;
		let last = self.w as isize - 1;
		let j0 = (k0 as isize).clamp(0, last) as usize * 4;
		let j1 = (k0 as isize + 1).clamp(0, last) as usize * 4;
		let (a, b, t) = (self.a, self.b, self.t);
		let mut c = [0.0; 3];
		for (ch, o) in c.iter_mut().enumerate() {
			let top = a[j0 + ch] as f32 * (1.0 - s) + a[j1 + ch] as f32 * s;
			let bot = b[j0 + ch] as f32 * (1.0 - s) + b[j1 + ch] as f32 * s;
			*o = (top * (1.0 - t) + bot * t) * (1.0 / 255.0);
		}
		c
	}
}

/// What the shader's uniforms were, worked out once a frame.
struct Uniforms {
	w: usize,
	h: usize,
	wf: f32,
	hf: f32,
	unit: f32,
	bright: f32,
	contrast: f32,
	sat: f32,
	hue: bool,
	hue_cos: f32,
	hue_sin: f32,
	sharp: f32,
	solar: f32,
	solar_at: f32,
	post: f32,
	inv: f32,
	pal: u8,
	pal_mix: f32,
	shadow: [f32; 3],
	high: [f32; 3],
	split: f32,
	scan: f32,
	grain: f32,
	vig: f32,
	pix: f32,
	glitch: f32,
	beat: f32,
	frame: u32,
	/// anything that samples off the pixel's own centre
	moves: bool,
}

impl Uniforms {
	fn new(l: &Look, w: usize, h: usize, time: f32) -> Self {
		let unit = w.min(h) as f32 / 1080.0;
		let a = l.hue.to_radians();
		// The site sets u_pix = pixelate·unit and the shader acts once that's over 1, so above 1080p
		// even the slider's "off" (1) came out as bigger pixels. Here 1 is off at any size, which is
		// what `changes` has to assume without knowing the size.
		let pix = if l.pixelate > 1.0 { l.pixelate * unit } else { 0.0 };
		let split = l.split * unit;
		Self {
			w,
			h,
			wf: w as f32,
			hf: h as f32,
			unit,
			bright: l.brightness,
			contrast: l.contrast,
			sat: l.saturation,
			hue: l.hue != 0.0,
			hue_cos: a.cos(),
			hue_sin: a.sin(),
			sharp: l.sharpen,
			solar: l.solarize,
			solar_at: l.solar_at,
			post: l.posterize,
			inv: l.invert,
			pal: palette(&l.palette),
			pal_mix: l.palette_mix,
			shadow: rgb(&l.shadow),
			high: rgb(&l.highlight),
			split,
			scan: l.scanlines,
			grain: l.grain,
			vig: l.vignette,
			pix,
			glitch: l.glitch,
			beat: (time * 11.0).floor(),
			frame: (time * 24.0) as u32,
			moves: l.glitch > 0.0 || pix > 1.0 || split > 0.0 || l.sharpen > 0.0,
		}
	}

	/// One row of the output, `y` from the top.
	fn row(&self, src: &[u8], y: usize, out: &mut [u8]) {
		let (w, h, wf, hf) = (self.w, self.h, self.wf, self.hf);
		// gl_FragCoord and v_uv count from the bottom
		let yg = (h - 1 - y) as u32;
		let frag_y = yg as f32 + 0.5;
		let vy = frag_y / hf;

		let mut shift = 0.0;
		if self.glitch > 0.0 {
			let band = (vy * 26.0).floor();
			if rnd(band, self.beat) > 1.0 - self.glitch * 0.35 {
				shift = (rnd(band, self.beat + 1.0) - 0.5) * self.glitch * 0.3;
			}
		}
		let pixelate = self.pix > 1.0;
		let (gx, gy) = (wf / self.pix, hf / self.pix);
		let v = if pixelate { ((vy * gy).floor() + 0.5) / gy } else { vy };
		let s = self.split / wf;
		let (dx, dy) = (1.6 * self.unit / wf, 1.6 * self.unit / hf);
		let mid = Rows::new(src, w, h, v);
		let (up, down) = if self.sharp > 0.0 {
			(Some(Rows::new(src, w, h, v + dy)), Some(Rows::new(src, w, h, v - dy)))
		} else {
			(None, None)
		};
		let own = &src[y * w * 4..(y + 1) * w * 4];

		let scan = if self.scan > 0.0 {
			1.0 - self.scan * (0.5 + 0.5 * (frag_y * SITE_PI / (1.5 * self.unit)).sin())
		} else {
			1.0
		};
		let qy = vy - 0.5;
		let qy2 = qy * qy;

		// a pixelated block all samples the same spot, so the last sample is kept for the next pixel
		let mut last_u = f32::NAN;
		let mut last = [0.0; 3];

		for (x, o) in out.chunks_exact_mut(4).enumerate() {
			let mut c = if !self.moves {
				let p = &own[x * 4..x * 4 + 3];
				[p[0] as f32 * (1.0 / 255.0), p[1] as f32 * (1.0 / 255.0), p[2] as f32 * (1.0 / 255.0)]
			} else {
				let mut u = (x as f32 + 0.5) / wf + shift;
				if pixelate {
					u = ((u * gx).floor() + 0.5) / gx;
				}
				if u == last_u {
					last
				} else {
					let mut c =
						if self.split > 0.0 { [mid.at(u + s)[0], mid.at(u)[1], mid.at(u - s)[2]] } else { mid.at(u) };
					if let (Some(up), Some(down)) = (&up, &down) {
						let (r, l, a, b) = (mid.at(u + dx), mid.at(u - dx), up.at(u), down.at(u));
						for ch in 0..3 {
							let soft = (r[ch] + l[ch] + a[ch] + b[ch]) * 0.25;
							c[ch] += (c[ch] - soft) * self.sharp * 2.0;
						}
					}
					last_u = u;
					last = c;
					c
				}
			};

			c = self.colour(c);

			if self.scan > 0.0 {
				for v in &mut c {
					*v *= scan;
				}
			}
			if self.vig > 0.0 {
				let qx = (x as f32 + 0.5) / wf - 0.5;
				let t = (((qx * qx + qy2).sqrt() - 0.2) / 0.6).clamp(0.0, 1.0);
				let k = 1.0 - self.vig * (t * t * (3.0 - 2.0 * t));
				for v in &mut c {
					*v *= k;
				}
			}
			if self.grain > 0.0 {
				let g = (noise(x as u32, yg, self.frame) - 0.5) * self.grain * 0.6;
				for v in &mut c {
					*v += g;
				}
			}
			o[0] = (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			o[1] = (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			o[2] = (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
			o[3] = 255;
		}
	}

	/// Everything that only depends on the colour: contrast through invert.
	#[inline]
	fn colour(&self, mut c: [f32; 3]) -> [f32; 3] {
		for v in &mut c {
			*v = (*v - 0.5) * self.contrast + 0.5 + self.bright;
		}
		if self.hue {
			// turned about the grey axis
			let (co, si, k) = (self.hue_cos, self.hue_sin, 0.57735);
			let cross = [k * (c[2] - c[1]), k * (c[0] - c[2]), k * (c[1] - c[0])];
			let along = k * (c[0] + c[1] + c[2]) * k * (1.0 - co);
			c = [
				c[0] * co + cross[0] * si + along,
				c[1] * co + cross[1] * si + along,
				c[2] * co + cross[2] * si + along,
			];
		}
		let l = lum(c);
		for v in &mut c {
			*v = mix(l, *v, self.sat).clamp(0.0, 1.0);
		}

		if self.solar > 0.0 {
			// folds everything brighter than 'at' back down, each channel on its own
			let at = self.solar_at;
			for v in &mut c {
				let folded = if *v >= at { at * (1.0 - (*v - at).max(0.0) / (1.0 - at)) } else { *v };
				*v = mix(*v, folded, self.solar);
			}
		}
		if self.post >= 2.0 {
			for v in &mut c {
				*v = (*v * self.post).floor() / (self.post - 1.0);
			}
		}

		if self.pal > 0 {
			let l = lum(c).clamp(0.0, 1.0);
			let p = match self.pal {
				1 => mix3(self.shadow, self.high, l),
				2 => thermal(l),
				_ => {
					let g = l.powf(0.8);
					let glow = 0.6 * l.powi(6);
					[mix(0.0, 0.34, g) + glow, mix(0.04, 1.0, g) + glow, mix(0.012, 0.42, g) + glow]
				}
			};
			c = mix3(c, p, self.pal_mix);
		}

		for v in &mut c {
			*v = mix(*v, 1.0 - *v, self.inv);
		}
		c
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// a picture with every kind of colour in it
	fn picture(w: u32, h: u32) -> Frame {
		let mut rgba = Vec::with_capacity((w * h * 4) as usize);
		for y in 0..h {
			for x in 0..w {
				let n = noise(x, y, 7);
				rgba.extend_from_slice(&[
					(x * 255 / w.max(2).saturating_sub(1).max(1)) as u8,
					(y * 255 / h.max(2).saturating_sub(1).max(1)) as u8,
					(n * 255.0) as u8,
					255,
				]);
			}
		}
		Frame::from_rgba(w, h, rgba)
	}

	fn graded(src: &Frame, l: &Look, time: f32) -> Frame {
		let mut out = Frame::default();
		Grade::new().render(src, l, time, true, &mut out);
		out
	}

	#[test]
	fn defaults_leave_the_picture_alone() {
		let l = Look::default();
		assert!(!Grade::changes(&l));
		let src = picture(97, 61);
		assert_eq!(graded(&src, &l, 3.2).rgba, src.rgba);
	}

	#[test]
	fn every_setting_counts_as_a_change() {
		let edits: [fn(&mut Look); 16] = [
			|l| l.brightness = 0.1,
			|l| l.contrast = 1.2,
			|l| l.saturation = 0.0,
			|l| l.hue = 30.0,
			|l| l.sharpen = 0.5,
			|l| l.solarize = 0.5,
			|l| l.posterize = 4.0,
			|l| l.invert = 1.0,
			|l| l.palette = "thermal".into(),
			|l| l.split = 3.0,
			|l| l.scanlines = 0.3,
			|l| l.grain = 0.3,
			|l| l.vignette = 0.5,
			|l| l.pixelate = 6.0,
			|l| l.glitch = 0.4,
			|l| l.palette = "night".into(),
		];
		for (i, edit) in edits.iter().enumerate() {
			let mut l = Look::default();
			edit(&mut l);
			assert!(Grade::changes(&l), "setting {i}");
		}
		let mut l = Look { palette: "duotone".into(), ..Look::default() };
		l.palette_mix = 0.0;
		assert!(!Grade::changes(&l), "a palette mixed in not at all");
		l.palette = "no such".into();
		l.palette_mix = 1.0;
		assert!(!Grade::changes(&l), "a palette it doesn't know");
		l.posterize = 1.5;
		assert!(!Grade::changes(&l), "posterize under 2");
	}

	#[test]
	fn invert_flips() {
		let l = Look { invert: 1.0, ..Look::default() };
		let src = picture(64, 40);
		let out = graded(&src, &l, 0.0);
		for (a, b) in src.rgba.chunks_exact(4).zip(out.rgba.chunks_exact(4)) {
			for ch in 0..3 {
				assert!((255 - a[ch] as i32 - b[ch] as i32).abs() <= 1, "{a:?} {b:?}");
			}
			assert_eq!(b[3], 255);
		}
	}

	#[test]
	fn no_saturation_is_grey() {
		let l = Look { saturation: 0.0, ..Look::default() };
		let src = picture(64, 40);
		let out = graded(&src, &l, 0.0);
		for (a, b) in src.rgba.chunks_exact(4).zip(out.rgba.chunks_exact(4)) {
			assert!(b[0].abs_diff(b[1]) <= 1 && b[1].abs_diff(b[2]) <= 1, "{b:?}");
			let want = (a[0] as f32 * 0.2126 + a[1] as f32 * 0.7152 + a[2] as f32 * 0.0722).round();
			assert!((b[1] as f32 - want).abs() <= 1.0, "{a:?} {b:?}");
		}
	}

	#[test]
	fn posterize_quantises() {
		let l = Look { posterize: 4.0, ..Look::default() };
		let out = graded(&picture(64, 40), &l, 0.0);
		for p in out.rgba.chunks_exact(4) {
			for &v in &p[..3] {
				assert!([0, 85, 170, 255].contains(&v), "{v}");
			}
		}
	}

	#[test]
	fn pixelate_makes_blocks() {
		// at 64 high, unit is 64/1080, so this is 8 pixels a block
		let l = Look { pixelate: 8.0 * 1080.0 / 64.0, ..Look::default() };
		let src = picture(64, 64);
		let out = graded(&src, &l, 0.0);
		let px = |x: usize, y: usize| &out.rgba[(y * 64 + x) * 4..(y * 64 + x) * 4 + 3];
		for y in 0..64 {
			for x in 0..64 {
				assert_eq!(px(x, y), px(x / 8 * 8, y / 8 * 8), "{x},{y}");
			}
		}
		assert_ne!(px(0, 0), px(8, 0));
		// while "1" is off at any size
		let l = Look { pixelate: 1.0, brightness: 0.0001, ..Look::default() };
		let big = picture(400, 2160);
		assert_eq!(graded(&big, &l, 0.0).rgba, big.rgba);
	}

	#[test]
	fn grain_is_new_every_frame() {
		let l = Look { grain: 0.5, ..Look::default() };
		let src = picture(64, 40);
		let a = graded(&src, &l, 1.0);
		assert_eq!(a.rgba, graded(&src, &l, 1.0).rgba, "the same frame twice is the same");
		assert_ne!(a.rgba, graded(&src, &l, 1.0 + 1.0 / 24.0).rgba);
		assert_ne!(a.rgba, src.rgba);
	}

	#[test]
	fn palettes_and_the_rest_stay_in_range() {
		// everything at once, mostly to see nothing panics or goes out of bounds at the edges
		let src = picture(37, 23);
		for pal in ["duotone", "thermal", "night"] {
			let l = Look {
				palette: pal.into(),
				palette_mix: 0.7,
				brightness: 0.1,
				contrast: 1.3,
				saturation: 1.4,
				hue: -40.0,
				sharpen: 1.0,
				solarize: 0.6,
				posterize: 5.0,
				split: 30.0 * 1080.0 / 23.0,
				scanlines: 0.4,
				grain: 0.3,
				vignette: 0.6,
				pixelate: 2.0 * 1080.0 / 23.0,
				glitch: 1.0,
				..Look::default()
			};
			let out = graded(&src, &l, 12.3);
			assert_eq!(out.rgba.len(), src.rgba.len());
		}
	}

	#[test]
	fn duotone_runs_shadow_to_highlight() {
		let l = Look {
			palette: "duotone".into(),
			shadow: "#000000".into(),
			highlight: "#ff8000".into(),
			..Look::default()
		};
		let src = Frame::from_rgba(2, 1, vec![0, 0, 0, 255, 255, 255, 255, 255]);
		let out = graded(&src, &l, 0.0);
		assert_eq!(&out.rgba[..4], &[0, 0, 0, 255]);
		assert_eq!(&out.rgba[4..], &[255, 128, 0, 255]);
	}

	#[test]
	fn hex_reads_like_the_site() {
		assert_eq!(rgb("#ff8000"), [1.0, 128.0 / 255.0, 0.0]);
		assert_eq!(rgb("#04060f"), [4.0 / 255.0, 6.0 / 255.0, 15.0 / 255.0]);
		assert_eq!(rgb("nonsense"), [0.0, 0.0, 0.0]);
		// parseInt("fff", 16) is 4095: no doubling up of short forms
		assert_eq!(rgb("#fff"), [0.0, 15.0 / 255.0, 1.0]);
	}

	#[test]
	fn noise_matches_the_shader() {
		// worked through by hand from the GLSL: (0, 0, 0) hashes to 0
		assert_eq!(noise(0, 0, 0), 0.0);
		let n = noise(1, 0, 0);
		assert!((0.0..=1.0).contains(&n) && n != noise(0, 1, 0));
	}

	/// `cargo test -p spotter-core --release grade::tests::timing -- --ignored --nocapture`
	#[test]
	#[ignore]
	fn timing() {
		let src = picture(1920, 1080);
		let mut out = Frame::default();
		let mut g = Grade::new();
		let looks = [
			(
				"contrast, saturation, grain, vignette",
				Look { contrast: 1.2, saturation: 1.3, grain: 0.2, vignette: 0.5, ..Look::default() },
			),
			(
				"+ sharpen, split, scanlines",
				Look {
					contrast: 1.2,
					saturation: 1.3,
					grain: 0.2,
					vignette: 0.5,
					sharpen: 0.6,
					split: 4.0,
					scanlines: 0.3,
					..Look::default()
				},
			),
			(
				"night palette, hue, posterize",
				Look { palette: "night".into(), hue: 20.0, posterize: 6.0, grain: 0.2, ..Look::default() },
			),
		];
		for (name, l) in &looks {
			g.render(&src, l, 0.0, true, &mut out);
			let n = 30;
			let t = std::time::Instant::now();
			for i in 0..n {
				g.render(&src, l, i as f32 / 30.0, true, &mut out);
			}
			println!("{name}: {:.2} ms a 1920x1080 frame", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
		}
	}
}
