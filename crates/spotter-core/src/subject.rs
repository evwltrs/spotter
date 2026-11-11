//! Telling the subject from the background, with no model: what's there that doesn't belong, once the
//! camera's movement is taken out and the parts of the scene that never sit still are discounted.
//!
//!   1. Camera. Blocks of this frame are matched against the last one and a single affine fit to the
//!      answers (pan, tilt, zoom, roll): whatever most of the picture agrees on is the camera. When
//!      the camera is moving, near things and far things shift by different amounts, so the leftovers
//!      are smoothed into a gentle local correction on top.
//!   2. Background. Each spot keeps a slowly learned idea of what it normally looks like, carried
//!      along with the camera. A subject is what differs from that, so it comes out solid rather than
//!      as an outline, and it stops counting the moment the spot goes back to normal, so no trails.
//!   3. What always flickers. Trees, water and noise never settle, so each spot also learns how much
//!      it normally changes, and only difference beyond that counts.
//!   4. Shapes. Things have to have moved to be picked up at all (or the background's own mistakes
//!      would count) and then stay for as long as they differ from the background. The connected hot
//!      areas, with their holes filled, are the subjects.
//!
//! The method, numbers and thresholds are 1nf.me's (`src/lib/tracker/subject.ts`). What a browser
//! can't afford, this does: the fine block match looks at every pixel of a block rather than every
//! other one each way, and the block search and the per-pixel model are spread over threads.

use rayon::prelude::*;

use crate::Frame;
use crate::Rect;
use crate::look::Look;

/// the heat map and mask are this many analysis pixels to a cell
pub const CELL: usize = 4;

/// how many frames a spot has to have been watched before it can call anything out of place
const SETTLED: u8 = 8;
/// block size on the small copy
const B: usize = 8;
/// how far to look, on the small copy
const R: usize = 6;
/// about how many blocks across and down
const ACROSS: usize = 9;
/// how far the full-size match looks either side of the small copy's answer
const FINE: isize = 3;

#[derive(Clone, Copy, Debug, Default)]
pub struct Blob {
	pub x0: f32,
	pub y0: f32,
	pub x1: f32,
	pub y1: f32,
	pub score: f32,
}

/// One block's answer: where it is (centred on the frame), how far it moved, and its place in the grid.
#[derive(Clone, Copy, Debug)]
struct Vector {
	x: f64,
	y: f64,
	u: f64,
	v: f64,
	gx: usize,
	gy: usize,
}

/// u = a + b·x + c·y, v = d + e·x + f·y
#[derive(Clone, Copy, Debug)]
struct Camera {
	a: f64,
	b: f64,
	c: f64,
	d: f64,
	e: f64,
	f: f64,
}

const STILL: Camera = Camera { a: 0.0, b: 0.0, c: 0.0, d: 0.0, e: 0.0, f: 0.0 };

/// What's known about one spot of the scene.
#[derive(Clone, Copy, Debug, Default)]
struct Spot {
	/// what it normally looks like
	back: f32,
	/// how much it normally changes from one frame to the next
	flick: f32,
	/// how many frames it's been watched, up to 255
	age: u8,
}

/// A connected hot area, its cells being `cells[first..end]`.
#[derive(Clone, Copy, Debug)]
struct Found {
	first: usize,
	end: usize,
	score: f32,
	x0: usize,
	y0: usize,
	x1: usize,
	y1: usize,
}

#[derive(Default)]
pub struct Subject {
	pub w: usize,
	pub h: usize,
	pub cols: usize,
	pub rows: usize,
	/// per pixel: how different from where the camera says it was
	pub residual: Vec<f32>,
	/// per cell: 1 inside a subject
	pub mask: Vec<f32>,
	/// the subjects, in analysis pixels
	pub blobs: Vec<Blob>,
	/// how much of the frame the subjects cover, 0-1
	pub coverage: f32,
	/// there's a mask worth using
	pub active: bool,
	/// a new camera estimate that the tracks haven't moved by yet
	pub fresh: bool,

	/// per cell: where this cell's contents were, last frame
	flow_u: Vec<f32>,
	flow_v: Vec<f32>,
	/// per cell: the same to the nearest pixel, as the background is looked up
	step_u: Vec<i32>,
	step_v: Vec<i32>,
	/// per pixel, and the next frame's being written
	spots: Vec<Spot>,
	spots_next: Vec<Spot>,
	/// per pixel row and cell column: how much moved, and how much stood out, beyond the usual
	moved: Vec<f32>,
	differs: Vec<f32>,
	/// per pixel row: its total difference, added up in order afterwards so a cut is called the same
	/// however the rows were shared out between threads
	lines: Vec<f32>,
	/// per cell
	heat: Vec<f32>,
	heat_next: Vec<f32>,
	/// the heat blurred by the last `segment`, which the next `update` reads as what's near a subject
	blur: Vec<f32>,
	swap: Vec<f32>,
	named: Vec<f32>,
	label: Vec<bool>,
	small_now: Vec<f32>,
	small_was: Vec<f32>,

	// kept between frames only so they needn't be made again
	blocks: Vec<Option<Vector>>,
	vs: Vec<Vector>,
	inliers: Vec<Vector>,
	us: Vec<f64>,
	ws: Vec<f64>,
	off_u: Vec<f32>,
	off_v: Vec<f32>,
	grid_u: Vec<f32>,
	grid_v: Vec<f32>,
	stack: Vec<usize>,
	cells: Vec<usize>,
	found: Vec<Found>,
	left: Vec<usize>,
	right: Vec<usize>,
	top: Vec<usize>,
	bottom: Vec<usize>,
}

/// The middle one: the element at len / 2 once sorted, not an average of two.
fn median(a: &mut [f64]) -> f64 {
	let mid = a.len() / 2;
	*a.select_nth_unstable_by(mid, f64::total_cmp).1
}

/// Least-squares p + q·x + r·y through the (centred) vectors. None when they're too bunched up to say.
/// In doubles: the coordinates reach a couple of hundred, so the determinant's terms are around 1e15,
/// further apart than a float can tell.
fn plane(vs: &[Vector], value: impl Fn(&Vector) -> f64) -> Option<[f64; 3]> {
	let (mut n, mut sx, mut sy, mut xx, mut xy, mut yy, mut sv, mut xv, mut yv) =
		(0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	for o in vs {
		let t = value(o);
		n += 1.0;
		sx += o.x;
		sy += o.y;
		xx += o.x * o.x;
		xy += o.x * o.y;
		yy += o.y * o.y;
		sv += t;
		xv += o.x * t;
		yv += o.y * t;
	}
	let det = n * (xx * yy - xy * xy) - sx * (sx * yy - xy * sy) + sy * (sx * xy - xx * sy);
	if det.abs() < 1e-6 {
		return None;
	}
	Some([
		(sv * (xx * yy - xy * xy) - sx * (xv * yy - xy * yv) + sy * (xv * xy - xx * yv)) / det,
		(n * (xv * yy - xy * yv) - sv * (sx * yy - xy * sy) + sy * (sx * yv - xv * sy)) / det,
		(n * (xx * yv - xv * xy) - sx * (sx * yv - xv * sy) + sv * (sx * xy - xx * sy)) / det,
	])
}

fn fit_camera(vs: &[Vector], inliers: &mut Vec<Vector>, scratch: &mut Vec<f64>) -> Camera {
	if vs.len() < 6 {
		return STILL;
	}
	// the majority is the camera: start from the median, then fit and tighten
	scratch.clear();
	scratch.extend(vs.iter().map(|o| o.u));
	let a = median(scratch);
	scratch.clear();
	scratch.extend(vs.iter().map(|o| o.v));
	let d = median(scratch);
	let shift = Camera { a, d, ..STILL };
	let mut cam = shift;
	for limit in [6.0, 3.5, 2.0] {
		let c = cam;
		inliers.clear();
		inliers.extend(
			vs.iter()
				.filter(|o| (o.u - (c.a + c.b * o.x + c.c * o.y)).hypot(o.v - (c.d + c.e * o.x + c.f * o.y)) < limit),
		);
		if (inliers.len() as f64) < 6f64.max(vs.len() as f64 * 0.3) {
			return shift;
		}
		let (Some(pu), Some(pv)) = (plane(inliers, |o| o.u), plane(inliers, |o| o.v)) else {
			return shift;
		};
		cam = Camera { a: pu[0], b: pu[1], c: pu[2], d: pv[0], e: pv[1], f: pv[2] };
	}
	// more zoom or roll than a frame can really have means the fit went wrong
	if cam.b.abs().max(cam.c.abs()).max(cam.e.abs()).max(cam.f.abs()) > 0.2 { shift } else { cam }
}

/// The sum of absolute differences between two rows, kept in eight running sums so it vectorises.
fn sad(a: &[f32], b: &[f32]) -> f32 {
	let mut s = [0.0f32; 8];
	let (pa, pb) = (a.chunks_exact(8), b.chunks_exact(8));
	let mut rest = 0.0;
	for (p, q) in pa.remainder().iter().zip(pb.remainder()) {
		rest += (p - q).abs();
	}
	for (p, q) in pa.zip(pb) {
		for k in 0..8 {
			s[k] += (p[k] - q[k]).abs();
		}
	}
	((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7])) + rest
}

/// What the block at (bx, by) on the small copy moved by: matched coarsely there, then finely at full
/// size from the middle of the block. None for a blank patch, which matches anywhere, so gets no say.
fn match_block(
	small: &[f32],
	old: &[f32],
	now: &[f32],
	was: &[f32],
	w: usize,
	h: usize,
	cols: usize,
	bx: usize,
	by: usize,
) -> Option<(isize, isize)> {
	let (mut lo, mut hi) = (1.0f32, 0.0f32);
	for y in 0..B {
		for &v in &small[(by + y) * cols + bx..][..B] {
			lo = lo.min(v);
			hi = hi.max(v);
		}
	}
	if hi - lo < 0.06 {
		return None;
	}

	let reach = R as isize;
	let (mut best, mut du, mut dv) = (f32::INFINITY, 0, 0);
	for sy in -reach..=reach {
		for sx in -reach..=reach {
			// a whisker of preference for "didn't move", so ties don't wander
			let mut total = (sx.abs() + sy.abs()) as f32 * 0.002;
			for y in 0..B {
				if total >= best {
					break;
				}
				let a = &small[(by + y) * cols + bx..][..B];
				let o = &old[(by + y).wrapping_add_signed(sy) * cols + bx.wrapping_add_signed(sx)..][..B];
				total += sad(a, o);
			}
			if total < best {
				best = total;
				du = sx;
				dv = sy;
			}
		}
	}

	// refine at full size, from the middle of the block
	let (px, py, size) = (bx * CELL + 4, by * CELL + 4, B * CELL - 8);
	let (cu, cv) = (du * CELL as isize, dv * CELL as isize);
	let (mut fine, mut fu, mut fv) = (f32::INFINITY, cu, cv);
	for sy in cv - FINE..=cv + FINE {
		for sx in cu - FINE..=cu + FINE {
			if px as isize + sx < 0
				|| py as isize + sy < 0
				|| (px + size) as isize + sx >= w as isize
				|| (py + size) as isize + sy >= h as isize
			{
				continue;
			}
			let mut total = 0.0;
			for y in 0..size {
				if total >= fine {
					break;
				}
				let a = &now[(py + y) * w + px..][..size];
				let o = &was[(py + y).wrapping_add_signed(sy) * w + px.wrapping_add_signed(sx)..][..size];
				total += sad(a, o);
			}
			if total < fine {
				fine = total;
				fu = sx;
				fv = sy;
			}
		}
	}
	Some((fu, fv))
}

/// Each cell's average, into `to`.
fn shrink(from: &[f32], to: &mut [f32], w: usize, cols: usize, rows: usize) {
	for cy in 0..rows {
		for cx in 0..cols {
			let mut s = 0.0;
			for y in 0..CELL {
				for &v in &from[(cy * CELL + y) * w + cx * CELL..][..CELL] {
					s += v;
				}
			}
			to[cy * cols + cx] = s / (CELL * CELL) as f32;
		}
	}
}

/// One pass of a 3x3 average, where the edges average what they have.
fn box3(from: &[f32], to: &mut [f32], cols: usize, rows: usize) {
	for r in 0..rows {
		for c in 0..cols {
			let (mut s, mut n) = (0.0f32, 0.0f32);
			for y in r.saturating_sub(1)..=(r + 1).min(rows - 1) {
				for &v in &from[y * cols + c.saturating_sub(1)..=y * cols + (c + 1).min(cols - 1)] {
					s += v;
					n += 1.0;
				}
			}
			to[r * cols + c] = s / n;
		}
	}
}

/// What each row of the background model reads.
struct Model<'a> {
	now: &'a [f32],
	was: &'a [f32],
	spots: &'a [Spot],
	step_u: &'a [i32],
	step_v: &'a [i32],
	w: usize,
	h: usize,
	cols: usize,
	rows: usize,
	learn: f32,
}

impl Model<'_> {
	/// Steps 2 and 3 for one row of pixels: how each differs from where the camera says it was, what's
	/// learned about it, and how much moved and stood out in each cell's part of the row. Returns the
	/// row's total difference.
	fn row(&self, y: usize, residual: &mut [f32], next: &mut [Spot], moved: &mut [f32], differs: &mut [f32]) -> f32 {
		let (w, h, learn) = (self.w, self.h, self.learn);
		let now = &self.now[y * w..][..w];
		let cells = (y / CELL).min(self.rows - 1) * self.cols;
		let mut total = 0.0;
		for cx in 0..self.cols {
			// the last cell takes in any pixels left over at the edge
			let x0 = cx * CELL;
			let x1 = if cx + 1 == self.cols { w } else { x0 + CELL };
			let c = cells + cx;
			let du = self.step_u[c] as isize;
			let py = y as isize + self.step_v[c] as isize;
			let (mut m, mut d) = (0.0f32, 0.0f32);
			if py < 1 || py >= h as isize - 1 {
				for x in x0..x1 {
					residual[x] = 0.0;
					next[x] = Spot { back: now[x], flick: 0.0, age: 0 };
				}
				moved[cx] = 0.0;
				differs[cx] = 0.0;
				continue;
			}
			let py = py as usize;
			let (wu, wm, wd) =
				(&self.was[(py - 1) * w..][..w], &self.was[py * w..][..w], &self.was[(py + 1) * w..][..w]);
			let (su, sm, sd) =
				(&self.spots[(py - 1) * w..][..w], &self.spots[py * w..][..w], &self.spots[(py + 1) * w..][..w]);
			for x in x0..x1 {
				let v = now[x];
				let px = x as isize + du;
				if px < 1 || px >= w as isize - 1 {
					// just came into frame: nothing known about it yet
					residual[x] = 0.0;
					next[x] = Spot { back: v, flick: 0.0, age: 0 };
					continue;
				}
				let px = px as usize;
				// the best of five spots: a camera estimate is only good to about a pixel
				let r = (v - wm[px])
					.abs()
					.min((v - wm[px - 1]).abs())
					.min((v - wm[px + 1]).abs())
					.min((v - wu[px]).abs())
					.min((v - wd[px]).abs());
				residual[x] = r;
				total += r;

				let s = sm[px];
				let flick = s.flick + (r - s.flick) * if r > s.flick { 0.05 } else { 0.02 };
				let noise = 0.035 + 2.5 * s.flick;
				if r > noise {
					m += r - noise;
				}

				let mut out = 0.0;
				if s.age >= SETTLED {
					let b = (v - s.back)
						.abs()
						.min((v - sm[px - 1].back).abs())
						.min((v - sm[px + 1].back).abs())
						.min((v - su[px].back).abs())
						.min((v - sd[px].back).abs());
					out = b - noise;
					if out > 0.0 {
						d += out;
					}
				}
				// quick at first, to get a picture of the place; then slow, and slower still for
				// whatever looks like a subject, so it isn't learned as scenery while it's in view
				let rate = (1.0 / (s.age as f32 + 2.0)).max(if out > 0.0 { learn * 0.15 } else { learn });
				let back = if s.age > 0 { s.back + (v - s.back) * rate } else { v };
				next[x] = Spot { back, flick, age: s.age.saturating_add(1) };
			}
			moved[cx] = m;
			differs[cx] = d;
		}
		total
	}
}

impl Subject {
	pub fn resize(&mut self, w: usize, h: usize) {
		self.w = w;
		self.h = h;
		self.cols = w / CELL;
		self.rows = h / CELL;
		let px = w * h;
		let cells = self.cols * self.rows;
		self.residual = vec![0.0; px];
		self.spots = vec![Spot::default(); px];
		self.spots_next = vec![Spot::default(); px];
		self.moved = vec![0.0; h * self.cols];
		self.differs = vec![0.0; h * self.cols];
		self.lines = vec![0.0; h];
		self.mask = vec![0.0; cells];
		self.flow_u = vec![0.0; cells];
		self.flow_v = vec![0.0; cells];
		self.step_u = vec![0; cells];
		self.step_v = vec![0; cells];
		self.heat = vec![0.0; cells];
		self.heat_next = vec![0.0; cells];
		self.blur = vec![0.0; cells];
		self.swap = vec![0.0; cells];
		self.named = vec![0.0; cells];
		self.label = vec![false; cells];
		self.small_now = vec![0.0; cells];
		self.small_was = vec![0.0; cells];
		self.reset();
	}

	/// Forget everything learned about the scene: a seek or a new video.
	pub fn reset(&mut self) {
		self.spots.fill(Spot::default());
		self.heat.fill(0.0);
		self.blur.fill(0.0);
		self.flow_u.fill(0.0);
		self.flow_v.fill(0.0);
		self.mask.fill(0.0);
		self.residual.fill(0.0);
		self.blobs.clear();
		self.coverage = 0.0;
		self.active = false;
		self.fresh = false;
	}

	/// Move a point (track units) along with the camera, from where it was last frame to where it is now.
	/// How far the picture as a whole moved since the last frame, in this copy's pixels: the median of
	/// every cell's movement, so the camera's pan or shake and not one thing crossing it.
	pub fn shift(&self) -> (f32, f32) {
		if self.cols == 0 || self.flow_u.is_empty() {
			return (0.0, 0.0);
		}
		let median = |v: &[f32]| {
			let mut s = v.to_vec();
			s.sort_by(f32::total_cmp);
			s[s.len() / 2]
		};
		// carry() takes a point to where it is now by taking the flow off, so the picture moved by minus it
		(-median(&self.flow_u), -median(&self.flow_v))
	}

	pub fn carry(&self, x: &mut f32, y: &mut f32, unit: f32) {
		if self.cols == 0 {
			return;
		}
		let cy = ((*y * unit / CELL as f32).floor().max(0.0) as usize).min(self.rows - 1);
		let cx = ((*x * unit / CELL as f32).floor().max(0.0) as usize).min(self.cols - 1);
		let c = cy * self.cols + cx;
		*x -= self.flow_u[c] / unit;
		*y -= self.flow_v[c] / unit;
	}

	/// Step 1: blocks matched coarsely on the small copies, then finely on the full ones, into a flow per cell.
	fn camera(&mut self, now: &[f32], was: &[f32]) {
		let (w, h, cols, rows) = (self.w, self.h, self.cols, self.rows);
		shrink(now, &mut self.small_now, w, cols, rows);
		shrink(was, &mut self.small_was, w, cols, rows);
		// blocks are spaced to make about ACROSS each way, each keeping R clear of the edges to look in
		let span = B + 2 * R;
		let step_x = (cols.saturating_sub(span) / ACROSS).max(1);
		let step_y = (rows.saturating_sub(span) / ACROSS).max(1);
		let ny = if rows >= span { (rows - span) / step_y + 1 } else { 0 };
		let nx = if ny > 0 && cols >= span { (cols - span) / step_x + 1 } else { 0 };

		let (small, old) = (&self.small_now, &self.small_was);
		self.blocks.clear();
		self.blocks.resize(nx * ny, None);
		self.blocks.par_iter_mut().enumerate().for_each(|(i, slot)| {
			let (gx, gy) = (i % nx, i / nx);
			let (bx, by) = (R + gx * step_x, R + gy * step_y);
			*slot = match_block(small, old, now, was, w, h, cols, bx, by).map(|(u, v)| {
				let size = B * CELL - 8;
				let (px, py) = (bx * CELL + 4, by * CELL + 4);
				Vector {
					x: (px + size / 2) as f64 - w as f64 / 2.0,
					y: (py + size / 2) as f64 - h as f64 / 2.0,
					u: u as f64,
					v: v as f64,
					gx,
					gy,
				}
			});
		});
		self.vs.clear();
		self.vs.extend(self.blocks.iter().flatten());

		let cam = fit_camera(&self.vs, &mut self.inliers, &mut self.us);
		let reach = w.max(h) as f64 / 2.0;
		let moving =
			cam.a.hypot(cam.d) > 0.4 || cam.b.abs().max(cam.c.abs()).max(cam.e.abs()).max(cam.f.abs()) * reach > 1.0;

		// the local correction: what each block disagreed with the fit by, with the outliers (which are
		// probably subjects) voted away by their neighbours. A camera on a pole gets none of this, or a
		// big enough subject would drag the background along with it.
		let left = (R * CELL + B * CELL / 2) as f64 - w as f64 / 2.0;
		let top = (R * CELL + B * CELL / 2) as f64 - h as f64 / 2.0;
		let (grid_u, grid_v) = (&mut self.grid_u, &mut self.grid_v);
		grid_u.clear();
		grid_u.resize(nx * ny, 0.0);
		grid_v.clear();
		grid_v.resize(nx * ny, 0.0);
		if moving && nx > 2 && ny > 2 {
			let (off_u, off_v) = (&mut self.off_u, &mut self.off_v);
			off_u.clear();
			off_u.resize(nx * ny, f32::NAN);
			off_v.clear();
			off_v.resize(nx * ny, f32::NAN);
			for o in &self.vs {
				off_u[o.gy * nx + o.gx] = (o.u - (cam.a + cam.b * o.x + cam.c * o.y)) as f32;
				off_v[o.gy * nx + o.gx] = (o.v - (cam.d + cam.e * o.x + cam.f * o.y)) as f32;
			}
			let (us, ws) = (&mut self.us, &mut self.ws);
			for gy in 0..ny {
				for gx in 0..nx {
					us.clear();
					ws.clear();
					for y in gy.saturating_sub(2)..=(gy + 2).min(ny - 1) {
						for x in gx.saturating_sub(2)..=(gx + 2).min(nx - 1) {
							if off_u[y * nx + x].is_nan() {
								continue;
							}
							us.push(off_u[y * nx + x] as f64);
							ws.push(off_v[y * nx + x] as f64);
						}
					}
					if us.len() < 5 {
						continue;
					}
					grid_u[gy * nx + gx] = median(us).clamp(-10.0, 10.0) as f32;
					grid_v[gy * nx + gx] = median(ws).clamp(-10.0, 10.0) as f32;
				}
			}
		}

		let span_x = (step_x * CELL) as f64;
		let span_y = (step_y * CELL) as f64;
		let mixing = nx > 1 && ny > 1;
		for r in 0..rows {
			let yy = (r * CELL + 2) as f64 - h as f64 / 2.0;
			let gyf = ((yy - top) / span_y).clamp(0.0, (ny as f64 - 1.001).max(0.0));
			let gy0 = gyf.floor();
			let ty = gyf - gy0;
			for c in 0..cols {
				let xx = (c * CELL + 2) as f64 - w as f64 / 2.0;
				let (mut lu, mut lv) = (0.0, 0.0);
				if mixing {
					let gxf = ((xx - left) / span_x).clamp(0.0, nx as f64 - 1.001);
					let gx0 = gxf.floor();
					let tx = gxf - gx0;
					let i00 = gy0 as usize * nx + gx0 as usize;
					let (i10, i01, i11) = (i00 + 1, i00 + nx, i00 + nx + 1);
					let mix = |g: &[f32]| {
						(g[i00] as f64 * (1.0 - tx) + g[i10] as f64 * tx) * (1.0 - ty)
							+ (g[i01] as f64 * (1.0 - tx) + g[i11] as f64 * tx) * ty
					};
					lu = mix(grid_u);
					lv = mix(grid_v);
				}
				let i = r * cols + c;
				self.flow_u[i] = (cam.a + cam.b * xx + cam.c * yy + lu) as f32;
				self.flow_v[i] = (cam.d + cam.e * xx + cam.f * yy + lv) as f32;
			}
		}
	}

	/// Steps 1 to 3, and the first half of 4: called with each new frame's brightness (0-1, w by h),
	/// and the last one's.
	pub fn update(&mut self, now: &[f32], was: &[f32], l: &Look) {
		let (w, h, cols, rows) = (self.w, self.h, self.cols, self.rows);
		if cols == 0 || rows == 0 {
			return;
		}
		debug_assert!(now.len() == w * h && was.len() == w * h, "a {w}x{h} subject needs {} pixels a frame", w * h);
		if l.stabilize {
			self.camera(now, was);
		} else {
			self.flow_u.fill(0.0);
			self.flow_v.fill(0.0);
		}
		for (s, f) in self.step_u.iter_mut().zip(&self.flow_u) {
			*s = f.round() as i32;
		}
		for (s, f) in self.step_v.iter_mut().zip(&self.flow_v) {
			*s = f.round() as i32;
		}

		// memory is how long something that stops moving takes to melt into the background
		let learn = 0.003 + 0.07 * (1.0 - l.memory).powi(2);
		let model =
			Model { now, was, spots: &self.spots, step_u: &self.step_u, step_v: &self.step_v, w, h, cols, rows, learn };
		self.residual
			.par_chunks_mut(w)
			.zip(self.spots_next.par_chunks_mut(w))
			.zip(self.moved.par_chunks_mut(cols))
			.zip(self.differs.par_chunks_mut(cols))
			.zip(self.lines.par_iter_mut())
			.enumerate()
			.with_min_len(8)
			.for_each(|(y, ((((residual, next), moved), differs), line))| {
				*line = model.row(y, residual, next, moved, differs)
			});
		let total: f64 = self.lines.iter().map(|&t| t as f64).sum();

		// a cut: nothing here relates to what came before
		if total / (w * h) as f64 > 0.16 {
			self.reset();
			return;
		}
		std::mem::swap(&mut self.spots, &mut self.spots_next);

		let (heat, hotter, shape) = (&self.heat, &mut self.heat_next, &self.blur);
		let decay = 0.88 + 0.11 * l.memory;
		let norm = 10.0 / (CELL * CELL) as f32;
		for r in 0..rows {
			// the last row of cells takes in any rows of pixels left over at the bottom
			let lines = r * CELL..if r + 1 == rows { h } else { (r + 1) * CELL };
			for c in 0..cols {
				let i = r * cols + c;
				let qx = c as isize + (self.flow_u[i] / CELL as f32).round() as isize;
				let qy = r as isize + (self.flow_v[i] / CELL as f32).round() as isize;
				let (was, near) = if qx >= 0 && qy >= 0 && qx < cols as isize && qy < rows as isize {
					let q = qy as usize * cols + qx as usize;
					(heat[q], shape[q])
				} else {
					(0.0, 0.0)
				};
				let (mut moved, mut differs) = (0.0f32, 0.0f32);
				for y in lines.clone() {
					moved += self.moved[y * cols + c];
					differs += self.differs[y * cols + c];
				}
				let m = (moved * norm).min(1.0);
				let d = (differs * norm * 0.6).min(1.0);
				// standing out only counts next to something that moved, or the background's own
				// mistakes would become subjects; movement can start one anywhere
				let target = m.max(d * (near * 5.0 + m * 4.0).min(1.0));
				let mut next = if target > was { was + (target - was) * 0.55 } else { was * decay };
				// back to looking like the background: it's gone, and this is what stops trails
				if d < 0.06 && m < 0.15 {
					next *= 0.55;
				}
				hotter[i] = next;
			}
		}
		std::mem::swap(&mut self.heat, &mut self.heat_next);
		self.fresh = true;
		self.active = true;
	}

	/// Keep the parts of the picture an edit closes off (`allowed`, a pixel at a time, false where
	/// closed) from ever warming up, so nothing there becomes a subject.
	pub fn exclude(&mut self, allowed: &[bool]) {
		let (w, cols, rows) = (self.w, self.cols, self.rows);
		if allowed.len() != w * self.h || cols == 0 {
			return;
		}
		for r in 0..rows {
			let y = (r * CELL + CELL / 2).min(self.h - 1);
			for c in 0..cols {
				let x = (c * CELL + CELL / 2).min(w - 1);
				if !allowed[y * w + x] {
					self.heat[r * cols + c] = 0.0;
				}
			}
		}
	}

	/// Step 4: heat into shapes, the mask and the blobs. Separate from update so its settings work on a
	/// paused frame.
	pub fn segment(&mut self, l: &Look) {
		if !self.active {
			return;
		}
		let (cols, rows) = (self.cols, self.rows);
		// twice over 3x3, which bridges small gaps between the moving bits of one thing
		box3(&self.heat, &mut self.swap, cols, rows);
		box3(&self.swap, &mut self.blur, cols, rows);
		let a = &self.blur;

		let limit = 0.27 - 0.22 * l.sensitivity;
		let smallest = (l.subject_min / 100.0 * (cols * rows) as f32).max(4.0);
		let (label, found, cells, stack) = (&mut self.label, &mut self.found, &mut self.cells, &mut self.stack);
		label.fill(false);
		found.clear();
		cells.clear();

		for start in 0..a.len() {
			if label[start] || a[start] < limit {
				continue;
			}
			label[start] = true;
			stack.clear();
			stack.push(start);
			let first = cells.len();
			let (mut sum, mut x0, mut y0, mut x1, mut y1) = (0.0f32, cols, rows, 0, 0);
			while let Some(i) = stack.pop() {
				cells.push(i);
				sum += a[i];
				let (x, y) = (i % cols, i / cols);
				x0 = x0.min(x);
				x1 = x1.max(x);
				y0 = y0.min(y);
				y1 = y1.max(y);
				let mut reach = |j: usize| {
					if !label[j] && a[j] >= limit {
						label[j] = true;
						stack.push(j);
					}
				};
				if x > 0 {
					reach(i - 1);
				}
				if x < cols - 1 {
					reach(i + 1);
				}
				if y > 0 {
					reach(i - cols);
				}
				if y < rows - 1 {
					reach(i + cols);
				}
			}
			// edge to edge means the camera estimate failed, not a very large subject
			let wall = x0 == 0 && x1 == cols - 1 && (x1 - x0) as f32 > cols as f32 * 0.85;
			let pillar = y0 == 0 && y1 == rows - 1 && (y1 - y0) as f32 > rows as f32 * 0.85;
			if ((cells.len() - first) as f32) < smallest || wall || pillar {
				cells.truncate(first);
				continue;
			}
			let mx = (x0 + x1) as f32 / 2.0 / cols as f32 - 0.5;
			let my = (y0 + y1) as f32 / 2.0 / rows as f32 - 0.5;
			// bigger and hotter wins, with a nudge toward the middle, where people point cameras
			found.push(Found {
				first,
				end: cells.len(),
				score: sum * (1.4 - (mx.hypot(my) * 1.4).min(1.0) * 0.6),
				x0,
				y0,
				x1,
				y1,
			});
		}

		found.sort_by(|p, q| q.score.total_cmp(&p.score));
		let mask = &mut self.mask;
		mask.fill(0.0);
		self.blobs.clear();
		let keep = (l.subjects.round().max(0.0) as usize).min(found.len());
		for s in &found[..keep] {
			// fill the holes: a cell is in if the shape is on both sides of it, along its row and column
			let (left, right, top, bottom) = (&mut self.left, &mut self.right, &mut self.top, &mut self.bottom);
			let (sw, sh) = (s.x1 - s.x0 + 1, s.y1 - s.y0 + 1);
			left.clear();
			left.resize(sh, usize::MAX);
			right.clear();
			right.resize(sh, 0);
			top.clear();
			top.resize(sw, usize::MAX);
			bottom.clear();
			bottom.resize(sw, 0);
			for &i in &cells[s.first..s.end] {
				let (x, y) = (i % cols, i / cols);
				let (row, col) = (y - s.y0, x - s.x0);
				left[row] = left[row].min(x);
				right[row] = right[row].max(x);
				top[col] = top[col].min(y);
				bottom[col] = bottom[col].max(y);
			}
			for y in s.y0..=s.y1 {
				for x in s.x0..=s.x1 {
					let (row, col) = (y - s.y0, x - s.x0);
					if x >= left[row] && x <= right[row] && y >= top[col] && y <= bottom[col] {
						mask[y * cols + x] = 1.0;
					}
				}
			}
			self.blobs.push(Blob {
				x0: (s.x0 * CELL) as f32,
				y0: (s.y0 * CELL) as f32,
				x1: ((s.x1 + 1) * CELL) as f32,
				y1: ((s.y1 + 1) * CELL) as f32,
				score: s.score,
			});
		}
		self.coverage = mask.iter().sum::<f32>() / mask.len() as f32;
	}

	/// Blend the detector's objects (in track units) into the mask, `blend` 0 all movement and 1 all
	/// detector. Each box counts as a rounded blob inside it, so its empty corners aren't subject; the
	/// movement's own shapes are left alone, for the boxes drawn round them.
	pub fn mix_objects(&mut self, list: &[Rect], unit: f32, blend: f32) {
		let (cols, rows) = (self.cols, self.rows);
		if cols == 0 {
			return;
		}
		let named = &mut self.named;
		named.fill(0.0);
		let to_cell = unit / CELL as f32;
		for b in list {
			let cx = (b.x + b.w / 2.0) * to_cell;
			let cy = (b.y + b.h / 2.0) * to_cell;
			let rx = (b.w / 2.0 * to_cell).max(0.75);
			let ry = (b.h / 2.0 * to_cell).max(0.75);
			let x0 = (cx - rx).floor().max(0.0) as usize;
			let y0 = (cy - ry).floor().max(0.0) as usize;
			let x1 = ((cx + rx).ceil() as isize).min(cols as isize - 1);
			let y1 = ((cy + ry).ceil() as isize).min(rows as isize - 1);
			for y in y0 as isize..=y1 {
				for x in x0 as isize..=x1 {
					if ((x as f32 + 0.5 - cx) / rx).powi(2) + ((y as f32 + 0.5 - cy) / ry).powi(2) <= 1.2 {
						named[y as usize * cols + x as usize] = 1.0;
					}
				}
			}
		}
		for (m, n) in self.mask.iter_mut().zip(named.iter()) {
			*m = *m * (1.0 - blend) + n * blend;
		}
	}

	/// What it sees, for `showSubject`: red is subject, blue is movement it decided against (the
	/// blurred heat). One pixel a cell, to be stretched over the frame without smoothing.
	pub fn paint(&self) -> Option<Frame> {
		let (cols, rows) = (self.cols, self.rows);
		if cols == 0 || rows == 0 {
			return None;
		}
		let mut rgba = vec![0u8; cols * rows * 4];
		for ((p, &m), &heat) in rgba.chunks_exact_mut(4).zip(&self.mask).zip(&self.blur) {
			if m > 0.02 {
				p.copy_from_slice(&[255, 40, 70, (130.0 * m).round() as u8]);
			} else {
				p.copy_from_slice(&[40, 150, 255, ((heat * 3.0).min(1.0) * 150.0).round() as u8]);
			}
		}
		Some(Frame::from_rgba(cols as u32, rows as u32, rgba))
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// The analysis size at the default detail, for a 16:9 video.
	const W: usize = 426;
	const H: usize = 240;

	/// A patchwork of 3-pixel squares of random greys, so every block has something to match on and
	/// nothing repeats within the search.
	fn texture(x: i32, y: i32) -> f32 {
		let (px, py) = (x.div_euclid(3) as u32, y.div_euclid(3) as u32);
		let mut h = px.wrapping_mul(0x9e37_79b1) ^ py.wrapping_mul(0x85eb_ca77) ^ 0x1234_5678;
		h ^= h >> 15;
		h = h.wrapping_mul(0x2c1b_3c6d);
		h ^= h >> 12;
		h = h.wrapping_mul(0x297a_2d39);
		h ^= h >> 15;
		0.2 + 0.5 * (h & 0xffff) as f32 / 65535.0
	}

	/// The texture panned right by `pan` pixels and zoomed by `zoom` about the middle, with a white
	/// square (x, y, size) on top.
	fn scene(pan: i32, zoom: f32, square: Option<(i32, i32, i32)>) -> Vec<f32> {
		let mut out = vec![0.0; W * H];
		let (mx, my) = (W as f32 / 2.0, H as f32 / 2.0);
		for y in 0..H as i32 {
			for x in 0..W as i32 {
				// sampled at the middle of the pixel, or a zoom would also shift everything half a pixel
				let (tx, ty) = (
					((x as f32 + 0.5 - mx) / zoom + mx).floor() as i32,
					((y as f32 + 0.5 - my) / zoom + my).floor() as i32,
				);
				let mut v = texture(tx - pan, ty);
				if let Some((sx, sy, size)) = square
					&& x >= sx && x < sx + size
					&& y >= sy && y < sy + size
				{
					v = 0.95;
				}
				out[y as usize * W + x as usize] = v;
			}
		}
		out
	}

	/// `frames` frames of a scene panning `pan` and zooming `zoom` a frame, with a square moving `dx` a
	/// frame, fed to a new Subject as Finder does; `each` sees it after every frame.
	fn run(
		pan: i32,
		zoom: f32,
		square: Option<(i32, i32, i32, i32)>,
		frames: i32,
		mut each: impl FnMut(i32, &mut Subject),
	) -> Subject {
		let l = Look::default();
		let mut s = Subject::default();
		s.resize(W, H);
		let at = |k: i32| scene(pan * k, zoom.powi(k), square.map(|(x, y, size, dx)| (x + dx * k, y, size)));
		let mut was = at(0);
		for k in 1..=frames {
			let now = at(k);
			s.update(&now, &was, &l);
			s.segment(&l);
			each(k, &mut s);
			was = now;
		}
		s
	}

	fn flow_range(s: &Subject) -> (f32, f32, f32, f32) {
		let lo = |v: &[f32]| v.iter().copied().fold(f32::INFINITY, f32::min);
		let hi = |v: &[f32]| v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
		(lo(&s.flow_u), hi(&s.flow_u), lo(&s.flow_v), hi(&s.flow_v))
	}

	#[test]
	fn a_square_moving_over_a_still_background_is_the_subject() {
		let frames = 16;
		let s = run(0, 1.0, Some((16, 44, 24, 3)), frames, |k, s| {
			assert!(s.active && s.fresh, "frame {k} counted as a cut");
			// the square's blocks disagree, and are outvoted
			let (u0, u1, v0, v1) = flow_range(s);
			assert!(
				u0 > -0.25 && u1 < 0.25 && v0 > -0.25 && v1 < 0.25,
				"frame {k}: the camera should be still, flow is u {u0}..{u1}, v {v0}..{v1}"
			);
		});
		let (cx, cy) = ((16 + 3 * frames + 12) as f32, (44 + 12) as f32);
		assert!(!s.blobs.is_empty(), "no subject found");
		let b = s.blobs[0];
		assert!(
			b.x0 <= cx && cx <= b.x1 && b.y0 <= cy && cy <= b.y1,
			"the best blob {b:?} isn't around the square's middle ({cx}, {cy})"
		);
		assert!(s.coverage > 0.0 && s.coverage < 0.05, "coverage {}", s.coverage);
		// the square's middle is in the mask, and painted red
		let cell = (cy as usize / CELL) * s.cols + cx as usize / CELL;
		assert_eq!(s.mask[cell], 1.0);
		let p = s.paint().expect("a picture");
		assert_eq!((p.w as usize, p.h as usize), (s.cols, s.rows));
		assert_eq!(&p.rgba[cell * 4..cell * 4 + 4], &[255, 40, 70, 130]);
		// a corner well away from it isn't
		assert_eq!(s.mask[0], 0.0);
		assert_eq!(&p.rgba[0..3], &[40, 150, 255]);
	}

	#[test]
	fn a_panning_camera_is_taken_out() {
		let unit = W.min(H) as f32;
		let s = run(2, 1.0, None, 16, |k, s| {
			assert!(s.active && s.fresh, "frame {k} counted as a cut");
			// content moving right means it was 2 pixels to the left last frame
			let (u0, u1, v0, v1) = flow_range(s);
			assert!(
				u0 > -2.25 && u1 < -1.75 && v0 > -0.25 && v1 < 0.25,
				"frame {k}: flow should be (-2, 0), is u {u0}..{u1}, v {v0}..{v1}"
			);
			// and what's carried along goes with the picture
			let (mut x, mut y) = (0.6, 0.4);
			s.carry(&mut x, &mut y, unit);
			assert!(
				(x - (0.6 + 2.0 / unit)).abs() < 1e-3 && (y - 0.4).abs() < 1e-3,
				"frame {k}: carried to ({x}, {y})"
			);
		});
		// the picture lines up, so almost nothing differs and nothing is a subject
		let mean = s.residual.iter().sum::<f32>() / s.residual.len() as f32;
		assert!(mean < 0.01, "mean residual {mean}");
		assert!(s.blobs.is_empty(), "found {:?}", s.blobs);
		assert_eq!(s.coverage, 0.0);
	}

	#[test]
	fn a_subject_is_found_while_the_camera_pans() {
		// the square crosses faster than the pan, so it moves against the background
		let frames = 24;
		let s = run(2, 1.0, Some((60, 100, 32, 5)), frames, |k, s| {
			let (u0, u1, ..) = flow_range(s);
			assert!(u0 > -2.25 && u1 < -1.75, "frame {k}: flow u {u0}..{u1}, the square dragging the camera along");
		});
		let (cx, cy) = ((60 + 5 * frames + 16) as f32, (100 + 16) as f32);
		assert!(
			s.blobs.iter().any(|b| b.x0 <= cx && cx <= b.x1 && b.y0 <= cy && cy <= b.y1),
			"no blob around ({cx}, {cy}): {:?}",
			s.blobs
		);
		assert!(s.coverage > 0.0 && s.coverage < 0.1, "coverage {}", s.coverage);
	}

	#[test]
	fn a_zoom_is_fitted() {
		// in by 1% a frame: everything spreads from the middle, about 2.1 px at the sides and 1.2 at the
		// top and bottom. Blocks only answer in whole pixels, so a frame's fit can be some way off; on
		// average it isn't.
		let frames = 12;
		let mut sum = [0.0f32; 4];
		run(0, 1.01, None, frames, |k, s| {
			assert!(s.active, "frame {k} counted as a cut");
			let (cols, rows) = (s.cols, s.rows);
			let at = |r: usize, c: usize| (s.flow_u[r * cols + c], s.flow_v[r * cols + c]);
			let sides = [at(rows / 2, 0).0, at(rows / 2, cols - 1).0, at(0, cols / 2).1, at(rows - 1, cols / 2).1];
			let (middle_u, middle_v) = at(rows / 2, cols / 2);
			assert!(
				sides[0] > 0.0 && sides[1] < 0.0 && sides[2] > 0.0 && sides[3] < 0.0,
				"frame {k}: left, right, top, bottom {sides:?}"
			);
			assert!(middle_u.abs() < 0.3 && middle_v.abs() < 0.3, "frame {k}: middle ({middle_u}, {middle_v})");
			for (t, v) in sum.iter_mut().zip(sides) {
				*t += v / frames as f32;
			}
		});
		let [left, right, top, bottom] = sum;
		assert!((left - 2.1).abs() < 0.4 && (right + 2.1).abs() < 0.4, "left {left}, right {right}");
		assert!((top - 1.2).abs() < 0.4 && (bottom + 1.2).abs() < 0.4, "top {top}, bottom {bottom}");
	}

	#[test]
	fn a_cut_forgets_the_scene() {
		let l = Look::default();
		let mut s = run(0, 1.0, Some((16, 44, 24, 3)), 12, |_, _| {});
		assert!(!s.blobs.is_empty());
		let was = scene(0, 1.0, Some((16 + 3 * 12, 44, 24)));
		let white = vec![1.0; W * H];
		s.update(&white, &was, &l);
		assert!(!s.active && !s.fresh && s.blobs.is_empty() && s.coverage == 0.0);
		assert!(s.spots.iter().all(|p| p.age == 0) && s.heat.iter().all(|&v| v == 0.0));
		// and segment leaves it be until a frame that follows on
		s.segment(&l);
		assert!(s.blobs.is_empty() && s.mask.iter().all(|&m| m == 0.0));
	}

	#[test]
	fn the_detectors_boxes_blend_into_the_mask() {
		let mut s = Subject::default();
		s.resize(160, 120);
		let unit = 120.0;
		// a box 40 by 40 analysis pixels at (40, 40): cells 10 to 20
		let r = Rect { x: 40.0 / unit, y: 40.0 / unit, w: 40.0 / unit, h: 40.0 / unit };
		s.mix_objects(&[r], unit, 0.5);
		assert_eq!(s.mask[15 * s.cols + 15], 0.5, "the middle");
		assert_eq!(s.mask[10 * s.cols + 10], 0.0, "a corner, outside the rounded blob");
		assert_eq!(s.mask[5 * s.cols + 30], 0.0, "elsewhere");
	}

	/// cargo test -p spotter-core --release subject::tests::per_frame_time -- --ignored --nocapture
	#[test]
	#[ignore]
	fn per_frame_time() {
		let l = Look::default();
		let mut s = Subject::default();
		s.resize(W, H);
		// a pan of a pixel a frame, with a square crossing it a little faster
		let at = |k: i32| scene(k, 1.0, Some((20 + 2 * k, 90, 40)));
		let mut was = at(0);
		let (mut update, mut segment) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
		let frames = 160;
		for k in 1..=frames {
			let now = at(k);
			let start = std::time::Instant::now();
			s.update(&now, &was, &l);
			let mid = std::time::Instant::now();
			s.segment(&l);
			if k > 10 {
				update += mid - start;
				segment += mid.elapsed();
			}
			was = now;
		}
		let now = at(frames + 1);
		let start = std::time::Instant::now();
		for _ in 0..200 {
			s.camera(&now, &was);
		}
		let camera = start.elapsed().as_secs_f64() * 1000.0 / 200.0;
		let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0 / (frames - 10) as f64;
		println!(
			"{W}x{H}: update {:.3} ms (camera {camera:.3}), segment {:.3} ms a frame; {} blobs, coverage {:.3}",
			ms(update),
			ms(segment),
			s.blobs.len(),
			s.coverage
		);
	}
}
