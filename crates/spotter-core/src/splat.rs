//! Splats: the picture lifted into a cloud of 3D shapes, one to each little square of it (its colour,
//! as far away as depth or its brightness says), and drawn again through a camera that swings round
//! them, on the GPU with wgpu (splat.wgsl). The shapes:
//!
//! - soft: Gaussians, drawn the way 3D Gaussian splatting draws them (projected, sorted far to near
//!   and laid over each other), stretched along the picture's edges;
//! - glow: discs, as a lens draws what's out of focus, added up like light rather than laid over, the
//!   bright far more than the dark, so highlights burn white and shadows sink;
//! - pixel: hard squares with gaps between, an LED wall coming apart in depth;
//! - voxel: blocks on terraces of depth, each running back as far as its farthest neighbour (a step in
//!   depth is a wall), lit by side and outlined, so the picture becomes a landscape of blocks.
//!
//! Where each goes in depth: the depth model's guess, how bright it is (bright comes forward, or dark
//! does), or nowhere (flat, and the camera tilts it); `splat_depth` deepens it. Then the cloud is bent
//! as a file is databent:
//!
//! - wave: a ripple running out through its depth from the middle, swelling on each beat;
//! - bits: bits flipped in the shapes' numbers as they're kept (32-bit floats), in runs, as a damaged
//!   file has them, so some swell, some vanish, some jump and some change colour;
//! - slip: bands of rows read from the wrong place, colour, size and depth taken from others along the
//!   row, and past half a few bands read some numbers out of step;
//! - burst: blown out toward the camera on the beat (or every few seconds without one);
//! - smear: shapes kept from the frames before rather than made again, only some fresh each frame and
//!   the old ones swelling, so what moves leaves itself behind (a datamosh of shapes).

use crate::Frame;
use crate::depth::Depther;
use crate::look::Look;
use crate::overlay::colour;
use crate::tracking::hash;
use anyhow::{Context, Result, anyhow, bail};
use rayon::prelude::*;
use std::f32::consts::{FRAC_PI_2, PI, TAU};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

/// the camera's focal length, in halves of the picture's height
const FOCAL: f32 = 1.4;
/// how far away a flat cloud is; how far apart the nearest and farthest are (as one over the
/// distance) at depth 1; the farthest any may be put
const FLAT: f32 = 1.0;
const SPREAD: f32 = 1.3;
const FARTHEST: f32 = 6.0;
/// what the camera turns round, the most it swings either way (radians), and its near and far planes
const PIVOT: f32 = 1.1;
const SWING: f32 = 0.35;
const CLIP: f32 = 0.05;
const FAR: f32 = 100.0;
/// how far a soft splat spreads, as a part of the gap between them, so neighbours overlap and cover,
/// and how much further a glowing one does
const SOFT: f32 = 0.72;
const GLOW: f32 = 1.5;
/// how much of its square a pixel fills (the rest is gap) and a voxel; how many terraces of depth
/// voxels stand on; the furthest back a voxel runs, as a part of its distance
const PIXEL_FILL: f32 = 0.84;
const VOXEL_FILL: f32 = 1.0;
const TERRACES: f32 = 12.0;
const WALL: f32 = 0.3;
/// how solid a splat is at its middle, and how thick, as a part of its smaller size
const SOLID: f32 = 0.99;
const THIN: f32 = 0.25;
/// the biggest a splat may be, and how far from the middle (in the cloud's units: 1 is the distance
/// of a flat cloud)
const BIGGEST: f32 = 0.12;
const REACH: f32 = 40.0;
/// the most a soft splat may spread on the picture, as a part of its longer side
const WIDEST: f32 = 0.15;
/// how long the GPU may take over a frame before it's given up on
const PATIENCE: Duration = Duration::from_secs(10);

pub fn active(l: &Look) -> bool {
	l.splat != "off" && !l.splat.is_empty()
}

/// Whether the look's splats are placed by the depth model.
pub fn wants_depth(l: &Look) -> bool {
	active(l) && l.splat_relief == "depth"
}

/// What each splat is drawn as.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
	Soft,
	Glow,
	Pixel,
	Voxel,
}

impl Shape {
	pub fn of(l: &Look) -> Shape {
		match l.splat_shape.as_str() {
			"glow" => Shape::Glow,
			"pixel" => Shape::Pixel,
			"voxel" => Shape::Voxel,
			_ => Shape::Soft,
		}
	}

	/// Drawn solid, through a depth buffer, rather than see-through.
	fn hard(self) -> bool {
		matches!(self, Shape::Pixel | Shape::Voxel)
	}
}

/// One splat, as the GPU reads it (three vec4s).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat {
	/// where it is: x right, y down, z away from the camera
	pub pos: [f32; 3],
	/// how solid it is at its middle, 0-1 (a pixel or a voxel under a half isn't drawn)
	pub alpha: f32,
	/// its standard deviations along its two axes, the first's angle in the picture's plane, and its
	/// thickness, as a part of the smaller
	pub scale: [f32; 2],
	pub angle: f32,
	pub thin: f32,
	/// red, green, blue 0-1, and a fourth that's never read
	pub colour: [f32; 4],
}

/// The 32-bit numbers in a splat as it's kept, and the one that's never read.
const WORDS: usize = 12;
const UNREAD: usize = 11;

/// A picture's splats, one to each little square of it, row by row.
#[derive(Clone, Default)]
pub struct Grid {
	pub cols: usize,
	pub rows: usize,
	pub splats: Vec<Splat>,
}

/// How many squares across and down a w by h picture is cut into, for splats `size` pixels of a 1080
/// picture apart: the same for the window's smaller picture as for an export's.
pub fn grid(w: u32, h: u32, size: f32) -> (usize, usize) {
	let gap = (size.max(1.0) * w.min(h) as f32 / 1080.0).max(1.0);
	(((w as f32 / gap).round() as usize).max(1), ((h as f32 / gap).round() as usize).max(1))
}

/// A square's number, for what's random about it.
fn cell(c: usize, r: usize) -> i64 {
	(r * 8192 + c) as i64
}

/// How far away something `near` (0 far to 1 near) is, in a cloud `deep` deep (0 flat).
fn distance(near: f32, deep: f32) -> f32 {
	1.0 / (1.0 + (near.clamp(0.0, 1.0) - 0.5) * SPREAD * deep.max(0.0)).max(1.0 / FARTHEST)
}

/// Splats for `frame`, as `l` makes them, placed by `depth` if it wants depth and there's any.
pub fn build(frame: &Frame, depth: Option<&Depther>, l: &Look) -> Grid {
	let (w, h) = (frame.w as usize, frame.h as usize);
	if w == 0 || h == 0 {
		return Grid::default();
	}
	let shape = Shape::of(l);
	// soft ones stretch along edges (as far as the look says, left alone); hard ones are square
	let stretch = match shape {
		Shape::Pixel | Shape::Voxel => 0.0,
		_ if l.splat == "plain" => l.splat_amount.clamp(0.0, 1.0),
		_ => 0.6,
	};
	let spread = SOFT * if shape == Shape::Glow { GLOW } else { 1.0 };
	let deep = l.splat_depth.clamp(0.0, 1.5);
	let (cols, rows) = grid(frame.w, frame.h, l.splat_size);
	let (gx, gy) = (w as f32 / cols as f32, h as f32 / rows as f32);
	let half = h as f32 / 2.0;
	let px = &frame.rgba;
	let lum = |x: usize, y: usize| {
		let p = &px[(y * w + x) * 4..];
		(p[0] as f32 * 0.2126 + p[1] as f32 * 0.7152 + p[2] as f32 * 0.0722) / 255.0
	};
	let mut splats = vec![Splat::default(); cols * rows];
	splats.par_chunks_mut(cols).enumerate().for_each(|(r, line)| {
		let y0 = ((r as f32 * gy) as usize).min(h - 1);
		let y1 = (((r + 1) as f32 * gy) as usize).clamp(y0 + 1, h);
		for (c, s) in line.iter_mut().enumerate() {
			let x0 = ((c as f32 * gx) as usize).min(w - 1);
			let x1 = (((c + 1) as f32 * gx) as usize).clamp(x0 + 1, w);
			// its colour, the square's mean; its edge, from the square's structure tensor
			let step = ((x1 - x0).max(y1 - y0) / 6).max(1);
			let (mut sum, mut n) = ([0f32; 3], 0f32);
			let (mut jxx, mut jxy, mut jyy) = (0f32, 0f32, 0f32);
			for y in (y0..y1).step_by(step) {
				for x in (x0..x1).step_by(step) {
					let p = &px[(y * w + x) * 4..];
					sum[0] += p[0] as f32;
					sum[1] += p[1] as f32;
					sum[2] += p[2] as f32;
					n += 1.0;
					if stretch > 0.0 {
						let dx = lum((x + 1).min(w - 1), y) - lum(x.saturating_sub(1), y);
						let dy = lum(x, (y + 1).min(h - 1)) - lum(x, y.saturating_sub(1));
						jxx += dx * dx;
						jxy += dx * dy;
						jyy += dy * dy;
					}
				}
			}
			let colour = [sum[0] / n / 255.0, sum[1] / n / 255.0, sum[2] / n / 255.0, 1.0];
			let id = cell(c, r);
			// the edge runs across the gradient, and how sure that is says how far it's stretched;
			// where there's no edge a soft one lies any way at all, and a hard one square
			let diff = jxx - jyy;
			let sure = if jxx + jyy > 1e-6 { (diff * diff + 4.0 * jxy * jxy).sqrt() / (jxx + jyy) } else { 0.0 };
			let along = if shape.hard() {
				0.0
			} else if sure > 0.05 {
				0.5 * (2.0 * jxy).atan2(diff) + FRAC_PI_2
			} else {
				hash(id, 13) * PI
			};
			let k = stretch * sure;
			// soft ones a little off the grid, so it doesn't show; hard ones on it
			let jitter = if shape.hard() { 0.0 } else { 0.5 };
			let cx = (c as f32 + 0.5 + (hash(id, 11) - 0.5) * jitter) * gx;
			let cy = (r as f32 + 0.5 + (hash(id, 12) - 0.5) * jitter) * gy;
			let light = colour[0] * 0.2126 + colour[1] * 0.7152 + colour[2] * 0.0722;
			let near = match l.splat_relief.as_str() {
				"light" => Some(light),
				"dark" => Some(1.0 - light),
				"flat" => None,
				_ => depth.and_then(|d| d.near_at((cx as usize).min(w - 1), (cy as usize).min(h - 1), w, h)),
			};
			let near = if shape == Shape::Voxel { near.map(|n| (n * TERRACES).round() / TERRACES) } else { near };
			let z = near.map_or(FLAT, |n| distance(n, deep));
			// the cloud's units to a pixel, this far away
			let unit = z / FOCAL / half;
			let sigma = spread * 0.5 * (gx + gy) * unit;
			*s = Splat {
				pos: [(cx - w as f32 / 2.0) * unit, (cy - half) * unit, z],
				alpha: SOLID,
				scale: [sigma * (1.0 + 1.2 * k), sigma * (1.0 - 0.4 * k)],
				angle: along,
				thin: THIN,
				colour,
			};
		}
	});
	Grid { cols, rows, splats }
}

/// A splat that can be drawn, whatever was done to its numbers: nothing that isn't a number, nothing
/// so big it would cover the picture many times over, colours 0-1.
fn sane(s: &mut Splat) {
	let fix = |v: f32, or: f32| if v.is_finite() { v } else { or };
	for p in &mut s.pos {
		*p = fix(*p, 0.0).clamp(-REACH, REACH);
	}
	s.alpha = fix(s.alpha, 0.0).abs().min(1.0);
	for v in &mut s.scale {
		*v = fix(*v, 0.0).abs().clamp(1e-5, BIGGEST);
	}
	s.angle = fix(s.angle, 0.0).clamp(-1e4, 1e4);
	s.thin = fix(s.thin, 0.0).abs().min(1.0);
	for v in &mut s.colour {
		*v = fix(*v, 0.0).abs().min(1.0);
	}
}

/// What the look does to its splats. `t` is the source's time; `since` seconds since the last beat
/// (or burst time).
pub fn bend(g: &mut Grid, l: &Look, t: f64, since: f32) {
	let amount = l.splat_amount.clamp(0.0, 1.0);
	match l.splat.as_str() {
		"wave" => wave(g, amount, t, since),
		"bits" => bits(g, amount, t),
		"slip" => slip(g, amount, t),
		"burst" => burst(g, amount, since),
		_ => {}
	}
	g.splats.par_iter_mut().for_each(sane);
}

/// A ripple running out from the middle through the cloud's depth (two rings across the picture),
/// stronger for a moment after each beat. Each keeps its place on the picture, so nearer is bigger.
fn wave(g: &mut Grid, amount: f32, t: f64, since: f32) {
	let (cols, rows) = (g.cols.max(1), g.rows.max(1));
	let aspect = cols as f32 / rows as f32;
	let swell = 0.4 + 0.6 * (-since.max(0.0) / 0.4).exp();
	let amp = amount * 0.5 * swell;
	let phase = (t * 0.8).fract() as f32;
	g.splats.par_iter_mut().enumerate().for_each(|(i, s)| {
		let u = ((i % cols) as f32 + 0.5) / cols as f32 - 0.5;
		let v = ((i / cols) as f32 + 0.5) / rows as f32 - 0.5;
		let d = ((u * aspect).powi(2) + v * v).sqrt();
		let z = s.pos[2].max(CLIP);
		let nz = (z * (1.0 - amp * ((d * 2.0 - phase) * TAU).sin())).max(2.0 * CLIP);
		let f = nz / z;
		s.pos = [s.pos[0] * f, s.pos[1] * f, nz];
	});
}

/// how many splats are in a run of bits
const RUN: usize = 48;

/// Bits flipped in the splats' numbers: in runs (a damaged stretch of the file, every splat in it
/// garbled) and a few anywhere. Mostly the mantissa's higher bits (a number a quarter or half off:
/// colours shift, splats jump along their row), now and then the exponent's (doubled, halved or far
/// worse). Each run changes several times a second.
fn bits(g: &mut Grid, amount: f32, t: f64) {
	let words: &mut [u32] = bytemuck::cast_slice_mut(&mut g.splats);
	words.par_chunks_mut(RUN * WORDS).enumerate().for_each(|(b, run)| {
		let b = b as i64;
		let tick = (t * 8.0 + hash(b, 21) as f64).floor() as i64;
		let hot = hash(b, tick * 2 + 1) < 0.18 * amount;
		let rate = if hot { 0.12 + 0.3 * amount } else { 0.0008 * amount };
		for (i, v) in run.iter_mut().enumerate() {
			if i % WORDS == UNREAD {
				continue;
			}
			let n = b * (RUN * WORDS) as i64 + i as i64;
			if hash(n, tick) >= rate {
				continue;
			}
			let pick = hash(n, tick + 7);
			let bit =
				if pick < 0.88 { 19 + (pick / 0.88 * 4.0) as u32 } else { 23 + ((pick - 0.88) / 0.12 * 8.0) as u32 };
			*v ^= 1 << bit.min(30);
		}
	});
}

/// Bands of rows slipped along themselves: each splat's colour, shape and depth read from others along
/// its row, as a picture read from the wrong place slides apart. Past half, a few bands are read some
/// numbers out of step, so a splat's place is another's colour (its size stays its own).
fn slip(g: &mut Grid, amount: f32, t: f64) {
	let (cols, rows) = (g.cols, g.rows);
	if cols == 0 {
		return;
	}
	let was = g.splats.clone();
	let flat: &[f32] = bytemuck::cast_slice(&was);
	let band = (rows / 30).max(1);
	g.splats.par_chunks_mut(cols).enumerate().for_each(|(r, line)| {
		let b = (r / band) as i64;
		let tick = (t * 5.0 + hash(b, 31) as f64).floor() as i64;
		if hash(b, tick * 8) >= 0.1 + 0.45 * amount {
			return;
		}
		let reach = (cols as f32 * 0.35 * amount * hash(b, tick * 8 + 1)) as isize + 1;
		let m = if hash(b, tick * 8 + 2) < 0.5 { reach } else { -reach };
		let out_of_step = hash(b, tick * 8 + 3) < (amount - 0.5).max(0.0) * 0.25;
		let k = 1 + (hash(b, tick * 8 + 4) * 10.0) as usize;
		for (c, s) in line.iter_mut().enumerate() {
			let at = |by: isize| &was[r * cols + (c as isize + by).rem_euclid(cols as isize) as usize];
			s.colour = at(m).colour;
			s.scale = at(m / 2).scale;
			s.angle = at(m / 2).angle;
			// a new depth, but seen where it was
			let z = at(m * 3 / 4).pos[2];
			let f = z / s.pos[2].max(CLIP);
			s.pos = [s.pos[0] * f, s.pos[1] * f, z];
			if out_of_step {
				let i = (r * cols + c) * WORDS + k;
				if let Some(read) = flat.get(i..i + WORDS) {
					let o: Splat = bytemuck::cast_slice::<f32, Splat>(read)[0];
					(s.pos, s.alpha, s.colour) = (o.pos, o.alpha, o.colour);
				}
			}
		}
	});
}

/// Blown out toward the camera, and outward from the middle, fading over a third of a second or so.
fn burst(g: &mut Grid, amount: f32, since: f32) {
	let k = amount * (-since.max(0.0) / 0.35).exp();
	if k < 0.002 {
		return;
	}
	g.splats.par_iter_mut().enumerate().for_each(|(i, s)| {
		let id = i as i64;
		let r = 0.25 + 0.75 * hash(id, 41);
		let z = s.pos[2].max(CLIP);
		let nz = z * (1.0 - 0.75 * k * r).max(0.1);
		let out = nz / z * (1.0 + 0.8 * k * r);
		s.pos[0] = s.pos[0] * out + (hash(id, 42) - 0.5) * 0.4 * k * r * nz;
		s.pos[1] = s.pos[1] * out + (hash(id, 43) - 0.5) * 0.4 * k * r * nz;
		s.pos[2] = nz;
		s.angle += (hash(id, 44) - 0.5) * 6.0 * k;
	});
}

/// How far back each voxel runs (kept in its thickness): to the farthest of its neighbours, so a step
/// in depth is a wall rather than a gap, but no more than `WALL` of its distance, so the biggest steps
/// are slabs floating in front rather than walls to the back of the cloud.
fn walls(g: &mut Grid) {
	let cols = g.cols.max(1);
	let z: Vec<f32> = g.splats.iter().map(|s| s.pos[2]).collect();
	g.splats.par_iter_mut().enumerate().for_each(|(i, s)| {
		let (c, r) = (i % cols, i / cols);
		let mut far = z[i];
		for (ok, j) in
			[(c > 0, i.wrapping_sub(1)), (c + 1 < cols, i + 1), (r > 0, i.wrapping_sub(cols)), (true, i + cols)]
		{
			if ok && let Some(&v) = z.get(j) {
				far = far.max(v);
			}
		}
		s.thin = (far - z[i]).min(WALL * z[i].max(CLIP));
	});
}

/// Where the camera is: turned round a point in the cloud, swinging as the look says, and drawn in a
/// little as it swings so less of the cloud's edge shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
	/// the turn, by rows
	pub turn: [[f32; 3]; 3],
	pub w: u32,
	pub h: u32,
	pub zoom: f32,
}

impl Camera {
	/// Swinging side to side and a little up and down (a figure of eight), `splat_orbit` of the most,
	/// round and back every `splat_speed` seconds.
	pub fn orbit(l: &Look, t: f64, w: u32, h: u32) -> Camera {
		let most = l.splat_orbit.clamp(0.0, 1.0) * SWING;
		let phase = ((t / l.splat_speed.max(0.5) as f64).fract() * std::f64::consts::TAU) as f32;
		let (ay, ax) = (most * phase.sin(), most * 0.45 * (2.0 * phase + 0.6).sin());
		let (sy, cy, sx, cx) = (ay.sin(), ay.cos(), ax.sin(), ax.cos());
		// turned about y, then about x
		let turn = [[cy, 0.0, sy], [sx * sy, cx, -sx * cy], [-cx * sy, sx, cx * cy]];
		Camera { turn, w, h, zoom: 1.0 + 0.5 * most }
	}

	/// How far a splat is from the camera, along its view.
	fn depth(&self, p: [f32; 3]) -> f32 {
		let r = self.turn[2];
		r[0] * p[0] + r[1] * p[1] + r[2] * (p[2] - PIVOT) + PIVOT
	}

	fn uniform(&self, shape: Shape) -> Uniform {
		let t = self.turn;
		let col = |c: usize| [t[0][c], t[1][c], t[2][c], 0.0];
		let (w, h) = (self.w as f32, self.h as f32);
		let widest = WIDEST * w.max(h) / 3.0;
		let fill = if shape == Shape::Voxel { VOXEL_FILL } else { PIXEL_FILL };
		// glow's gain: white where it's even comes out twice as bright, a mid grey as it was (each disc
		// covers about 1.65 standard deviations round)
		let glow = 2.0 / (PI * (1.65 * SOFT * GLOW).powi(2));
		Uniform {
			r0: col(0),
			r1: col(1),
			r2: col(2),
			pivot: [0.0, 0.0, PIVOT, FOCAL * h / 2.0 * self.zoom],
			view: [w, h, widest * widest, CLIP],
			extra: [0.0, fill / (2.0 * SOFT), glow, FAR],
		}
	}
}

/// The camera as splat.wgsl reads it.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniform {
	r0: [f32; 4],
	r1: [f32; 4],
	r2: [f32; 4],
	pivot: [f32; 4],
	view: [f32; 4],
	extra: [f32; 4],
}

/// The splats far to near as the camera sees them (at the same distance, in the order they were made,
/// so a flat cloud doesn't shimmer).
pub fn sort(splats: &[Splat], cam: &Camera) -> Vec<Splat> {
	let mut keyed: Vec<(f32, u32)> = splats.par_iter().enumerate().map(|(i, s)| (cam.depth(s.pos), i as u32)).collect();
	keyed.par_sort_unstable_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
	keyed.par_iter().map(|&(_, i)| splats[i as usize]).collect()
}

/// How a frame's splats are drawn: their shape, and the colour behind them (0-1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
	pub shape: Shape,
	pub back: [f64; 3],
}

impl Style {
	pub fn of(l: &Look) -> Style {
		let c = colour(&l.splat_back, 1.0);
		Style { shape: Shape::of(l), back: [c.red() as f64, c.green() as f64, c.blue() as f64] }
	}
}

/// What's kept of a look's splats from frame to frame (smear's), so it's the engine's, not the GPU's.
#[derive(Default)]
pub struct Cloud {
	held: Grid,
}

impl Cloud {
	pub fn clear(&mut self) {
		self.held = Grid::default();
	}

	/// Draw `out` (the look so far) again as splats, bent as `l` says, with `depth` if there's any.
	/// `t` is the source's time, `dt` the output's seconds a frame, `since` seconds since the last
	/// beat (or burst time); `fresh` false draws the same frame again.
	pub fn draw(
		&mut self,
		gpu: &mut Splatter,
		out: &mut Frame,
		depth: Option<&Depther>,
		l: &Look,
		t: f64,
		dt: f64,
		since: f32,
		fresh: bool,
	) -> Result<()> {
		let mut g = if l.splat == "smear" {
			let (cols, rows) = grid(out.w, out.h, l.splat_size);
			let same = self.held.cols == cols && self.held.rows == rows && !self.held.splats.is_empty();
			if !same {
				self.held = build(out, depth, l);
			} else if fresh {
				let new = build(out, depth, l);
				self.smear(&new, l.splat_amount.clamp(0.0, 1.0), t, dt);
			}
			self.held.clone()
		} else {
			self.held = Grid::default();
			build(out, depth, l)
		};
		bend(&mut g, l, t, since);
		let style = Style::of(l);
		if style.shape == Shape::Voxel {
			walls(&mut g);
		}
		let cam = Camera::orbit(l, t, out.w, out.h);
		// only soft ones are laid over each other in order: glow adds up, hard ones have a depth buffer
		let splats = if style.shape == Shape::Soft { sort(&g.splats, &cam) } else { g.splats };
		gpu.draw(&splats, &cam, &style, out)
	}

	/// Some splats made again from `new`, the rest kept (and swelling) for about `0.15 + 2.85 *
	/// amount` seconds each.
	fn smear(&mut self, new: &Grid, amount: f32, t: f64, dt: f64) {
		let keep = 0.15 + 2.85 * amount as f64;
		let fresh = (dt / keep).clamp(0.0, 1.0) as f32;
		let tick = (t * 1000.0).round() as i64;
		let swell = 1.0 + 0.9 * dt as f32;
		self.held.splats.par_iter_mut().zip(new.splats.par_iter()).enumerate().for_each(|(i, (h, n))| {
			if hash(i as i64, tick) < fresh {
				*h = *n;
			} else {
				let most = 4.0 * n.scale[0].max(n.scale[1]);
				h.scale = h.scale.map(|v| (v * swell).min(most));
			}
		});
	}
}

/// wgpu's device, one for the whole program, made the first time splats are wanted.
struct Gpu {
	device: wgpu::Device,
	queue: wgpu::Queue,
	name: String,
}

fn gpu() -> Result<&'static Gpu> {
	static G: OnceLock<Result<Gpu, String>> = OnceLock::new();
	G.get_or_init(|| open().map_err(|e| format!("{e:#}"))).as_ref().map_err(|e| anyhow!("{e}"))
}

fn open() -> Result<Gpu> {
	let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
	// DX12 alone on Windows, as the window draws with: a Vulkan instance loads every GPU's driver
	desc.backends = if cfg!(windows) { wgpu::Backends::DX12 } else { wgpu::Backends::VULKAN | wgpu::Backends::METAL };
	let instance = wgpu::Instance::new(desc);
	let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
		power_preference: wgpu::PowerPreference::HighPerformance,
		..Default::default()
	}))
	.context("no GPU to draw splats with")?;
	let (device, queue) = pollster::block_on(
		adapter.request_device(&wgpu::DeviceDescriptor { label: Some("splats"), ..Default::default() }),
	)
	.context("the GPU wouldn't draw splats")?;
	// what goes wrong is caught where it happens; anything else is said, not a crash
	device.on_uncaptured_error(Arc::new(|e| eprintln!("splats: {e}")));
	Ok(Gpu { device, queue, name: adapter.get_info().name })
}

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const ATTRIBUTES: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4, 2 => Float32x4];

/// Draws splats on the GPU. It keeps no splats between frames, so it can be lent to a look mixed in.
pub struct Splatter {
	soft: wgpu::RenderPipeline,
	glow: wgpu::RenderPipeline,
	pixel: wgpu::RenderPipeline,
	voxel: wgpu::RenderPipeline,
	group: wgpu::BindGroup,
	camera: wgpu::Buffer,
	/// the splats, and how many it has room for
	splats: Option<(wgpu::Buffer, usize)>,
	target: Option<Target>,
	/// the GPU's name, and how long the last frame took
	pub name: String,
	pub ms: f32,
}

/// What a w by h picture is drawn into (with its depth, for hard shapes), and read back through (rows
/// padded as copies want them).
struct Target {
	w: u32,
	h: u32,
	texture: wgpu::Texture,
	view: wgpu::TextureView,
	depth: wgpu::TextureView,
	back: wgpu::Buffer,
	row: u32,
}

impl Target {
	fn new(d: &wgpu::Device, w: u32, h: u32) -> Target {
		let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
		let texture = d.create_texture(&wgpu::TextureDescriptor {
			label: Some("splats"),
			size,
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: FORMAT,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
			view_formats: &[],
		});
		let depth = d.create_texture(&wgpu::TextureDescriptor {
			label: Some("splats' depth"),
			size,
			mip_level_count: 1,
			sample_count: 1,
			dimension: wgpu::TextureDimension::D2,
			format: DEPTH,
			usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
			view_formats: &[],
		});
		let view = texture.create_view(&Default::default());
		let depth = depth.create_view(&Default::default());
		let row = (w * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
		let back = d.create_buffer(&wgpu::BufferDescriptor {
			label: Some("splats read back"),
			size: row as u64 * h as u64,
			usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
			mapped_at_creation: false,
		});
		Target { w, h, texture, view, depth, back, row }
	}
}

impl Splatter {
	pub fn new() -> Result<Splatter> {
		let g = gpu()?;
		let d = &g.device;
		let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
		let shader = d.create_shader_module(wgpu::ShaderModuleDescriptor {
			label: Some("splats"),
			source: wgpu::ShaderSource::Wgsl(include_str!("splat.wgsl").into()),
		});
		let bind = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
			label: Some("splat camera"),
			entries: &[wgpu::BindGroupLayoutEntry {
				binding: 0,
				visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
				ty: wgpu::BindingType::Buffer {
					ty: wgpu::BufferBindingType::Uniform,
					has_dynamic_offset: false,
					min_binding_size: None,
				},
				count: None,
			}],
		});
		let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
			label: Some("splats"),
			bind_group_layouts: &[Some(&bind)],
			immediate_size: 0,
		});
		let pipeline = |vs: &str, fs: &str, strip: bool, blend: Option<wgpu::BlendState>, solid: bool| {
			d.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
				label: Some("splats"),
				layout: Some(&layout),
				vertex: wgpu::VertexState {
					module: &shader,
					entry_point: Some(vs),
					compilation_options: Default::default(),
					buffers: &[Some(wgpu::VertexBufferLayout {
						array_stride: std::mem::size_of::<Splat>() as u64,
						step_mode: wgpu::VertexStepMode::Instance,
						attributes: &ATTRIBUTES,
					})],
				},
				primitive: wgpu::PrimitiveState {
					topology: if strip {
						wgpu::PrimitiveTopology::TriangleStrip
					} else {
						wgpu::PrimitiveTopology::TriangleList
					},
					..Default::default()
				},
				depth_stencil: solid.then_some(wgpu::DepthStencilState {
					format: DEPTH,
					depth_write_enabled: Some(true),
					depth_compare: Some(wgpu::CompareFunction::Less),
					stencil: Default::default(),
					bias: Default::default(),
				}),
				multisample: Default::default(),
				fragment: Some(wgpu::FragmentState {
					module: &shader,
					entry_point: Some(fs),
					compilation_options: Default::default(),
					targets: &[Some(wgpu::ColorTargetState {
						format: FORMAT,
						blend,
						write_mask: wgpu::ColorWrites::ALL,
					})],
				}),
				multiview_mask: None,
				cache: None,
			})
		};
		let add = wgpu::BlendComponent {
			src_factor: wgpu::BlendFactor::One,
			dst_factor: wgpu::BlendFactor::One,
			operation: wgpu::BlendOperation::Add,
		};
		let soft = pipeline("vs", "fs", true, Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING), false);
		let glow = pipeline("vs", "fs_glow", true, Some(wgpu::BlendState { color: add, alpha: add }), false);
		let pixel = pipeline("vs_pixel", "fs_hard", true, None, true);
		let voxel = pipeline("vs_voxel", "fs_hard", false, None, true);
		let camera = d.create_buffer(&wgpu::BufferDescriptor {
			label: Some("splat camera"),
			size: std::mem::size_of::<Uniform>() as u64,
			usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
			mapped_at_creation: false,
		});
		let group = d.create_bind_group(&wgpu::BindGroupDescriptor {
			label: Some("splat camera"),
			layout: &bind,
			entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
		});
		if let Some(e) = pollster::block_on(scope.pop()) {
			bail!("splats couldn't be set up: {e}");
		}
		Ok(Splatter {
			soft,
			glow,
			pixel,
			voxel,
			group,
			camera,
			splats: None,
			target: None,
			name: g.name.clone(),
			ms: 0.0,
		})
	}

	/// Draw `splats` (soft ones sorted far to near) as `cam` sees them, as `style` says, into `out`.
	pub fn draw(&mut self, splats: &[Splat], cam: &Camera, style: &Style, out: &mut Frame) -> Result<()> {
		let t0 = Instant::now();
		let g = gpu()?;
		let d = &g.device;
		let (w, h) = (cam.w, cam.h);
		if w == 0 || h == 0 || out.w != w || out.h != h {
			return Ok(());
		}
		let valid = d.push_error_scope(wgpu::ErrorFilter::Validation);
		let memory = d.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
		let drawn = self.draw_in(g, splats, cam, style, out);
		// the scopes go in the order they came
		let memory = pollster::block_on(memory.pop());
		let valid = pollster::block_on(valid.pop());
		if let Some(e) = memory.or(valid) {
			bail!("the GPU couldn't draw the splats: {e}");
		}
		drawn?;
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		Ok(())
	}

	fn draw_in(&mut self, g: &Gpu, splats: &[Splat], cam: &Camera, style: &Style, out: &mut Frame) -> Result<()> {
		let (d, q) = (&g.device, &g.queue);
		let (w, h) = (cam.w, cam.h);
		if self.target.as_ref().is_none_or(|t| t.w != w || t.h != h) {
			self.target = Some(Target::new(d, w, h));
		}
		let need = splats.len().max(1);
		if self.splats.as_ref().is_none_or(|(_, room)| *room < need) {
			let room = need + need / 2;
			let b = d.create_buffer(&wgpu::BufferDescriptor {
				label: Some("splats"),
				size: (room * std::mem::size_of::<Splat>()) as u64,
				usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
				mapped_at_creation: false,
			});
			self.splats = Some((b, room));
		}
		let (Some((buf, _)), Some(t)) = (self.splats.as_ref(), self.target.as_ref()) else { unreachable!() };
		if !splats.is_empty() {
			q.write_buffer(buf, 0, bytemuck::cast_slice(splats));
		}
		q.write_buffer(&self.camera, 0, bytemuck::bytes_of(&cam.uniform(style.shape)));
		let (pipeline, corners) = match style.shape {
			Shape::Soft => (&self.soft, 4),
			Shape::Glow => (&self.glow, 4),
			Shape::Pixel => (&self.pixel, 4),
			Shape::Voxel => (&self.voxel, 36),
		};
		let [r, gr, b] = style.back;
		let mut enc = d.create_command_encoder(&Default::default());
		{
			let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
				label: Some("splats"),
				color_attachments: &[Some(wgpu::RenderPassColorAttachment {
					view: &t.view,
					depth_slice: None,
					resolve_target: None,
					ops: wgpu::Operations {
						load: wgpu::LoadOp::Clear(wgpu::Color { r, g: gr, b, a: 1.0 }),
						store: wgpu::StoreOp::Store,
					},
				})],
				depth_stencil_attachment: style.shape.hard().then_some(wgpu::RenderPassDepthStencilAttachment {
					view: &t.depth,
					depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
					stencil_ops: None,
				}),
				..Default::default()
			});
			pass.set_pipeline(pipeline);
			pass.set_bind_group(0, &self.group, &[]);
			pass.set_vertex_buffer(0, buf.slice(..));
			pass.draw(0..corners, 0..splats.len() as u32);
		}
		enc.copy_texture_to_buffer(
			wgpu::TexelCopyTextureInfo {
				texture: &t.texture,
				mip_level: 0,
				origin: wgpu::Origin3d::ZERO,
				aspect: wgpu::TextureAspect::All,
			},
			wgpu::TexelCopyBufferInfo {
				buffer: &t.back,
				layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(t.row), rows_per_image: Some(h) },
			},
			wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
		);
		q.submit([enc.finish()]);
		let slice = t.back.slice(..);
		let (tx, rx) = std::sync::mpsc::channel();
		slice.map_async(wgpu::MapMode::Read, move |r| {
			let _ = tx.send(r);
		});
		d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(PATIENCE) })
			.context("the GPU didn't finish the splats")?;
		rx.recv_timeout(PATIENCE).context("the splats never came back")?.context("the splats couldn't be read back")?;
		{
			let data = slice.get_mapped_range().context("the splats couldn't be read back")?;
			let bytes: &[u8] = &data;
			let (row, line) = (t.row as usize, w as usize * 4);
			out.rgba.par_chunks_mut(line).enumerate().for_each(|(y, o)| {
				o.copy_from_slice(&bytes[y * row..y * row + line]);
				for p in o.chunks_exact_mut(4) {
					p[3] = 255;
				}
			});
		}
		t.back.unmap();
		Ok(())
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A smooth picture with a hard edge down the middle.
	fn picture(w: u32, h: u32) -> Frame {
		let mut f = Frame::new(w, h);
		for y in 0..h as usize {
			for x in 0..w as usize {
				let p = (y * w as usize + x) * 4;
				let left = x < w as usize / 2;
				f.rgba[p] = if left { 200 } else { 30 };
				f.rgba[p + 1] = (40 + y * 150 / h as usize) as u8;
				f.rgba[p + 2] = (60 + x * 120 / w as usize) as u8;
			}
		}
		f
	}

	fn look(mode: &str, shape: &str, size: f32) -> Look {
		Look {
			splat: mode.into(),
			splat_shape: shape.into(),
			splat_size: size,
			splat_amount: 0.8,
			splat_orbit: 0.0,
			splat_relief: "depth".into(),
			splat_depth: 1.0,
			..Look::default()
		}
	}

	#[test]
	fn the_window_and_an_export_cut_the_picture_alike() {
		assert_eq!(grid(1920, 1080, 6.0), (320, 180));
		assert_eq!(grid(960, 540, 6.0), (320, 180));
		assert_eq!(grid(1080, 1920, 6.0), (180, 320));
		// never more squares than pixels
		assert_eq!(grid(320, 180, 2.0), (320, 180));
	}

	#[test]
	fn a_flat_cloud_lands_on_its_squares_in_their_colours() {
		let f = picture(192, 108);
		let g = build(&f, None, &Look { splat_amount: 0.5, ..look("plain", "soft", 12.0) });
		assert_eq!((g.cols, g.rows), grid(192, 108, 12.0));
		assert_eq!(g.splats.len(), g.cols * g.rows);
		let cam = Camera::orbit(&look("plain", "soft", 12.0), 0.0, 192, 108);
		let (k, gx) = (FOCAL * 54.0, 192.0 / g.cols as f32);
		for (i, s) in g.splats.iter().enumerate() {
			assert_eq!(s.pos[2], FLAT);
			assert!((cam.depth(s.pos) - FLAT).abs() < 1e-5);
			// where it lands on the picture is within its square
			let x = 96.0 + k * s.pos[0] / s.pos[2];
			let c = (i % g.cols) as f32;
			assert!(x >= c * gx && x <= (c + 1.0) * gx, "splat {i} at {x}");
		}
		// the left of the picture is red, the right isn't
		assert!(g.splats[0].colour[0] > 0.7 && g.splats[g.cols - 1].colour[0] < 0.2);
		// splats by the edge lie along it (up and down), stretched
		let edge = &g.splats[g.cols / 2];
		assert!((edge.angle.sin().abs() - 1.0).abs() < 0.05 && edge.scale[0] > edge.scale[1], "{edge:?}");
	}

	#[test]
	fn hard_shapes_sit_square_on_the_grid_and_brightness_can_place_them() {
		let f = picture(192, 108);
		let l = Look { splat_relief: "light".into(), ..look("plain", "voxel", 12.0) };
		let g = build(&f, None, &l);
		let (k, gx) = (FOCAL * 54.0, 192.0 / g.cols as f32);
		for (i, s) in g.splats.iter().enumerate() {
			assert_eq!((s.angle, s.scale[0]), (0.0, s.scale[1]));
			let x = 96.0 + k * s.pos[0] / s.pos[2];
			assert!((x - ((i % g.cols) as f32 + 0.5) * gx).abs() < 1e-3);
		}
		// the left is brighter, so nearer; dark would turn that round
		let (left, right) = (g.splats[0].pos[2], g.splats[g.cols - 1].pos[2]);
		assert!(left < right, "{left} {right}");
		let dark = build(&f, None, &Look { splat_relief: "dark".into(), ..l.clone() });
		assert!(dark.splats[0].pos[2] > dark.splats[dark.cols - 1].pos[2]);
		// and none of it without depth
		let flat = build(&f, None, &Look { splat_depth: 0.0, ..l });
		assert!(flat.splats.iter().all(|s| (s.pos[2] - FLAT).abs() < 1e-6));
	}

	#[test]
	fn bends_are_steady_and_drawable() {
		let f = picture(192, 108);
		for mode in ["plain", "wave", "bits", "slip", "burst"] {
			let l = look(mode, "soft", 4.0);
			let once = |t: f64| {
				let mut g = build(&f, None, &l);
				bend(&mut g, &l, t, 0.1);
				g.splats
			};
			let a = once(1.3);
			assert_eq!(a, once(1.3), "{mode} isn't the same twice");
			for s in &a {
				let numbers = s.pos.iter().chain(&s.scale).chain(&s.colour).chain([&s.alpha, &s.angle, &s.thin]);
				assert!(numbers.into_iter().all(|v| v.is_finite()), "{mode}: {s:?}");
				assert!(s.scale.iter().all(|&v| v > 0.0 && v <= BIGGEST) && s.colour.iter().all(|&v| v <= 1.0));
			}
			let plain = build(&f, None, &l).splats;
			let changed = a.iter().zip(&plain).filter(|(x, y)| x != y).count();
			match mode {
				"plain" => assert_eq!(changed, 0),
				// bits and slip bend some of it, the rest (nearly) all
				"bits" | "slip" => assert!(changed > 0 && changed < a.len(), "{mode} changed {changed}"),
				_ => assert!(changed > a.len() * 9 / 10, "{mode} changed {changed}"),
			}
		}
	}

	#[test]
	fn far_comes_first_and_a_flat_cloud_keeps_its_order() {
		let mut s = vec![Splat { pos: [0.0, 0.0, 1.0], ..Default::default() }; 5];
		s[3].pos[2] = 2.0;
		s[1].pos[2] = 0.5;
		for (i, v) in s.iter_mut().enumerate() {
			v.colour[3] = i as f32;
		}
		let cam = Camera::orbit(&look("plain", "soft", 6.0), 0.0, 10, 10);
		let order: Vec<f32> = sort(&s, &cam).iter().map(|v| v.colour[3]).collect();
		assert_eq!(order, vec![3.0, 0.0, 2.0, 4.0, 1.0]);
	}

	#[test]
	fn smear_keeps_some_and_lets_go_on_a_jump() {
		let f = picture(96, 54);
		let mut c = Cloud::default();
		let l = Look { splat_amount: 0.5, ..look("smear", "soft", 12.0) };
		let (cols, rows) = grid(96, 54, 12.0);
		c.held = build(&f, None, &l);
		let other = build(&Frame::new(96, 54), None, &l);
		c.smear(&other, l.splat_amount, 0.5, 1.0 / 30.0);
		let made_again = c.held.splats.iter().filter(|s| s.colour[0] == 0.0).count();
		assert!(made_again > 0 && made_again < cols * rows / 4, "{made_again}");
		c.clear();
		assert!(c.held.splats.is_empty());
	}

	/// Drawn by the GPU: each shape left alone makes the picture again, near enough (pixels with their
	/// gaps showing what's behind, glow brighter). Skipped where there's no GPU to draw with.
	#[test]
	fn every_shape_draws_the_picture_again() {
		let mut gpu = match Splatter::new() {
			Ok(s) => s,
			Err(e) => {
				eprintln!("no GPU for splats, so not drawn: {e}");
				return;
			}
		};
		let f = picture(320, 180);
		let n = f.rgba.len() as f64;
		let bright = |v: &[u8]| v.iter().map(|&x| x as f64).sum::<f64>() / n;
		let draw = |gpu: &mut Splatter, l: &Look| {
			let mut out = f.clone();
			Cloud::default().draw(gpu, &mut out, None, l, 0.0, 1.0 / 30.0, 9.0, true).unwrap();
			assert!(out.rgba.chunks_exact(4).all(|p| p[3] == 255));
			let off = f.rgba.iter().zip(&out.rgba).map(|(a, b)| a.abs_diff(*b) as f64).sum::<f64>() / n;
			(out, off)
		};
		let (soft, off) = draw(&mut gpu, &look("plain", "soft", 4.0));
		assert!(off < 8.0, "soft splats are {off:.1} levels off the picture on average");
		assert!((bright(&soft.rgba) / bright(&f.rgba) - 1.0).abs() < 0.03, "soft splats change how bright it is");
		// voxels: blocks of the picture, a little darker for their outlines
		let (voxels, off) = draw(&mut gpu, &look("plain", "voxel", 48.0));
		assert!(off < 20.0, "voxels are {off:.1} levels off the picture on average");
		assert!(bright(&voxels.rgba) > bright(&f.rgba) * 0.75, "voxels are too dark");
		// glow: the bright red on the left brighter, the dark red on the right darker
		let (glow, _) = draw(&mut gpu, &look("plain", "glow", 8.0));
		let red = |v: &[u8], left: bool| {
			let row = |y: usize| (if left { 20..140 } else { 180..300 }).map(move |x| v[(y * 320 + x) * 4] as f64);
			(40..140).flat_map(row).sum::<f64>() / (100.0 * 120.0)
		};
		assert!(red(&glow.rgba, true) > red(&f.rgba, true) + 20.0, "glow doesn't burn the highlights");
		assert!(red(&glow.rgba, false) < red(&f.rgba, false) - 10.0, "glow doesn't sink the shadows");
		let back = Look { splat_back: "#ff00ff".into(), ..look("plain", "pixel", 60.0) };
		let (pixels, _) = draw(&mut gpu, &back);
		let gaps = pixels.rgba.chunks_exact(4).filter(|p| p[..3] == [255, 0, 255]).count();
		assert!(gaps > 320 * 180 / 10, "only {gaps} pixels show the gaps between pixels");
		// a bent cloud is something else
		let (bent, _) = draw(&mut gpu, &look("wave", "voxel", 48.0));
		assert_ne!(bent.rgba, voxels.rgba);
	}
}
