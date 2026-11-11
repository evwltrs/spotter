//! The tracking itself, with no model: look at a small copy of each frame, score every pixel by how
//! much it stands out (an edge, something moving, something very bright or dark), keep the best one in
//! each little cell of a grid, and then match those points to last frame's so each keeps its number for
//! as long as it can be followed. Clumps of points become the big boxes.
//!
//! Positions are in "units" where 1 is the frame's shorter side, so nothing here cares about
//! resolution. A port of 1nf.me's `src/lib/tracker/tracking.ts`.

use crate::Frame;
use crate::Rect;
use crate::look::Look;
use crate::objects::{ObjectTrack, ObjectTracks};
use crate::subject::{CELL, Subject};

#[derive(Clone, Copy, Debug)]
pub struct Point {
	pub x: f32,
	pub y: f32,
	pub score: f32,
}

#[derive(Clone, Debug)]
pub struct Track {
	pub id: i64,
	pub x: f32,
	pub y: f32,
	/// frames it's been followed for
	pub age: u32,
	/// frames since it was last seen
	pub miss: u32,
	pub score: f32,
	/// how far it moves per frame, eased
	pub speed: f32,
	/// where it's been, oldest first, as x, y, x, y…
	pub path: Option<Vec<f32>>,
}

/// A big box around something moving.
#[derive(Clone, Debug)]
pub struct Region {
	pub id: i64,
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
	pub miss: u32,
	/// where it's been, oldest first, as x, y, x, y…
	pub path: Option<Vec<f32>>,
}

impl Region {
	pub fn rect(&self) -> Rect {
		Rect { x: self.x, y: self.y, w: self.w, h: self.h }
	}
}

/// A repeatable 0-1 from a number, so what a track looks like is decided once by its id and never
/// flickers. The same mixing as the site's, to 24 bits.
pub fn hash(n: i64, salt: i64) -> f32 {
	let mut h = ((n as i32) ^ ((salt as i32).wrapping_add(1)).wrapping_mul(0x9e3779b1u32 as i32))
		.wrapping_mul(0x85ebca6bu32 as i32);
	h ^= ((h as u32) >> 13) as i32;
	h = h.wrapping_mul(0xc2b2ae35u32 as i32);
	h ^= ((h as u32) >> 16) as i32;
	((h as u32) >> 8) as f32 / 16_777_216.0
}

fn extend(path: Option<Vec<f32>>, x: f32, y: f32, most: usize, ease: f32) -> Option<Vec<f32>> {
	let mut p = path.unwrap_or_default();
	let n = p.len();
	if n >= 2 {
		let (px, py) = (p[n - 2], p[n - 1]);
		p.push(px + (x - px) * ease);
		p.push(py + (y - py) * ease);
	} else {
		p.push(x);
		p.push(y);
	}
	if p.len() > most * 2 {
		let cut = p.len() - most * 2;
		p.drain(..cut);
	}
	Some(p)
}

#[derive(Default)]
pub struct Finder {
	/// the small copy of the frame everything is worked out from
	pub low: Frame,
	now: Vec<f32>,
	was: Vec<f32>,
	/// frames looked at since the last jump in the picture; motion needs two
	frames: u32,
	pub w: usize,
	pub h: usize,
	pub subject: Subject,
	/// where in the small copy things may be followed, a pixel at a time; empty is everywhere
	pub allowed: Vec<bool>,
}

impl Finder {
	pub fn new() -> Self {
		Self::default()
	}

	/// Take a new frame.
	pub fn grab(&mut self, src: &Frame, l: &Look) {
		let scale = l.detail.round() / src.w.min(src.h) as f32;
		let w = ((src.w as f32 * scale).round() as usize).max(4);
		let h = ((src.h as f32 * scale).round() as usize).max(4);
		if w != self.w || h != self.h {
			self.w = w;
			self.h = h;
			self.low = Frame::new(w as u32, h as u32);
			self.now = vec![0.0; w * h];
			self.was = vec![0.0; w * h];
			self.frames = 0;
			self.subject.resize(w, h);
		}
		src.resize_into(&mut self.low);
		std::mem::swap(&mut self.now, &mut self.was);
		for (v, p) in self.now.iter_mut().zip(self.low.rgba.chunks_exact(4)) {
			*v = (p[0] as f32 * 0.2126 + p[1] as f32 * 0.7152 + p[2] as f32 * 0.0722) / 255.0;
		}
		self.frames += 1;
		if self.frames > 1 {
			self.subject.update(&self.now, &self.was, l);
		}
	}

	/// Forget the last frame, after a seek or a new video: the difference would be nonsense.
	pub fn reset(&mut self) {
		self.frames = 0;
		self.subject.reset();
	}

	pub fn find(&mut self, l: &Look, objects: &[ObjectTrack], blend: f32) -> Vec<Point> {
		let (w, h) = (self.w, self.h);
		if w == 0 {
			return Vec::new();
		}
		let unit = w.min(h) as f32;
		let open = if self.allowed.len() == w * h { Some(&self.allowed[..]) } else { None };
		if let Some(a) = open {
			self.subject.exclude(a);
		}
		self.subject.segment(l);
		if !objects.is_empty() {
			let rects: Vec<Rect> = objects.iter().map(ObjectTrack::rect).collect();
			self.subject.mix_objects(&rects, unit, blend);
		}
		let sub = &self.subject;
		let now = &self.now;
		let moving = self.frames > 1 && l.motion > 0.0;
		// no subject means plain scores; movement's focus comes in with the area covered, so that one
		// small speck, quite possibly a wrong one, can't pull every point in the frame onto itself.
		// The detector named what it found, so as it takes over that caution goes.
		let cautious = if sub.active { l.focus * (sub.coverage / 0.006).min(1.0) } else { 0.0 };
		let focus = if objects.is_empty() { cautious } else { cautious + (l.focus - cautious) * blend };
		let residual = &sub.residual;
		let mask = &sub.mask;
		let cell = (l.spacing.round() as usize).max(2);
		let cols = w.div_ceil(cell);
		let mut best = vec![0.0f32; cols * h.div_ceil(cell)];
		let mut place = vec![usize::MAX; best.len()];

		for y in 1..h - 1 {
			let row = (y / cell) * cols;
			for x in 1..w - 1 {
				let i = y * w + x;
				if open.is_some_and(|a| !a[i]) {
					continue;
				}
				let v = now[i];
				let mut s = 0.0;
				if l.edges > 0.0 {
					s += l.edges * (((now[i + 1] - now[i - 1]).abs() + (now[i + w] - now[i - w]).abs()) * 2.5).min(1.0);
				}
				if moving {
					s += l.motion * (residual[i] * 6.0).min(1.0);
				}
				if l.bright > 0.0 {
					s += l.bright * v * v * v;
				}
				if l.dark > 0.0 {
					s += l.dark * (1.0 - v) * (1.0 - v) * (1.0 - v);
				}
				if focus > 0.0 && sub.cols > 0 {
					// the background fades out by `focus`; inside a subject even a flat patch can hold a
					// point, so it fills in rather than only outlining, and the sharpest bit still wins
					let inside = mask[(y / CELL).min(sub.rows - 1) * sub.cols + (x / CELL).min(sub.cols - 1)];
					s = s * (1.0 - focus * (1.0 - inside)) + focus * inside * 0.6;
				}
				let c = row + x / cell;
				if s > best[c] {
					best[c] = s;
					place[c] = i;
				}
			}
		}

		let mut out: Vec<Point> = Vec::new();
		for c in 0..best.len() {
			if best[c] < l.threshold || place[c] == usize::MAX {
				continue;
			}
			let i = place[c];
			out.push(Point { x: ((i % w) as f32 + 0.5) / unit, y: ((i / w) as f32 + 0.5) / unit, score: best[c] });
		}
		out.sort_by(|a, b| b.score.total_cmp(&a.score));
		out
	}
}

#[derive(Default)]
pub struct Follower {
	pub tracks: Vec<Track>,
	/// the big boxes around what's moving
	pub regions: Vec<Region>,
	/// the detector's objects on show
	pub named: Vec<ObjectTrack>,
	next: i64,
	base: Option<f32>,
}

/// Give out the next number, starting again from the first id whenever that setting changes.
fn take_id(next: &mut i64, base: &mut Option<f32>, l: &Look) -> i64 {
	if *base != Some(l.id_base) {
		*base = Some(l.id_base);
		*next = l.id_base as i64;
	}
	*next += 1;
	*next - 1
}

impl Follower {
	pub fn new() -> Self {
		Self::default()
	}

	/// The next number, for things followed elsewhere that share the count.
	pub fn next_id(&mut self, l: &Look) -> i64 {
		take_id(&mut self.next, &mut self.base, l)
	}

	/// Back to no tracks (a seek); the numbers carry on climbing.
	pub fn clear(&mut self) {
		self.tracks.clear();
		self.regions.clear();
		self.named.clear();
	}

	/// Back to no tracks and the first number.
	pub fn restart(&mut self) {
		self.clear();
		self.base = None;
	}

	pub fn step(
		&mut self,
		found: Vec<Point>,
		l: &Look,
		sub: &mut Subject,
		unit: f32,
		mut objects: Option<&mut ObjectTracks>,
	) {
		// everything rides along with the camera first, so what's matched next is what really moved
		let moved = sub.fresh;
		if sub.fresh {
			sub.fresh = false;
			for t in &mut self.tracks {
				carry(&mut t.x, &mut t.y, &mut t.path, sub, unit);
			}
			for r in &mut self.regions {
				carry(&mut r.x, &mut r.y, &mut r.path, sub, unit);
			}
			if let Some(o) = objects.as_deref_mut() {
				for t in &mut o.list {
					let (x, y) = (t.x, t.y);
					carry(&mut t.x, &mut t.y, &mut t.path, sub, unit);
					// a tape's things are already where they are on this frame; only their trails ride
					if o.pinned {
						(t.x, t.y) = (x, y);
					}
				}
				o.advance();
			}
		}
		let most = l.points.round().max(0.0) as usize;
		// only this frame's strongest count, so a stale edge gives up its place the moment something
		// better turns up, instead of holding it forever because it never moves
		let points = &found[..found.len().min(most)];
		let reach = (l.reach / 100.0).powi(2);
		let mut claimed = vec![false; points.len()];
		let ease = 1.0 - l.smooth;
		let mut kept: Vec<Track> = Vec::with_capacity(self.tracks.len() + points.len());

		// oldest first (they were added in order), so the longest-followed get first pick
		for mut t in std::mem::take(&mut self.tracks) {
			let mut hit = None;
			let mut closest = reach;
			for (i, p) in points.iter().enumerate() {
				if claimed[i] {
					continue;
				}
				let d = (p.x - t.x).powi(2) + (p.y - t.y).powi(2);
				if d < closest {
					closest = d;
					hit = Some(i);
				}
			}
			if let Some(i) = hit {
				let p = points[i];
				claimed[i] = true;
				let nx = t.x + (p.x - t.x) * ease;
				let ny = t.y + (p.y - t.y) * ease;
				t.speed += ((nx - t.x).hypot(ny - t.y) - t.speed) * 0.25;
				t.x = nx;
				t.y = ny;
				t.score = p.score;
				t.miss = 0;
				t.age += 1;
				kept.push(t);
			} else {
				t.miss += 1;
				if t.miss as f32 <= l.linger {
					kept.push(t);
				}
			}
		}

		for (i, p) in points.iter().enumerate() {
			if claimed[i] {
				continue;
			}
			let id = take_id(&mut self.next, &mut self.base, l);
			kept.push(Track { id, x: p.x, y: p.y, age: 0, miss: 0, score: p.score, speed: 0.0, path: None });
		}
		// anything over the limit is a lost track still lingering; those go first, longest lost soonest
		if kept.len() > most {
			let mut lost: Vec<usize> = (0..kept.len()).filter(|&i| kept[i].miss > 0).collect();
			lost.sort_by(|&a, &b| kept[b].miss.cmp(&kept[a].miss));
			lost.truncate(kept.len() - most);
			let mut drop = vec![false; kept.len()];
			for i in lost {
				drop[i] = true;
			}
			let mut i = 0;
			kept.retain(|_| {
				i += 1;
				!drop[i - 1]
			});
		}
		self.tracks = kept;

		self.named = match objects.as_deref_mut() {
			Some(o) => {
				let (next, base) = (&mut self.next, &mut self.base);
				o.shown(l, || take_id(next, base, l))
			}
			None => Vec::new(),
		};
		self.boxes(l, sub, unit);
		if moved {
			self.trace(l, objects);
		}
	}

	/// Note where everything is now. The big boxes go by the middle of their bottom edge, which is
	/// where a thing meets the ground.
	fn trace(&mut self, l: &Look, objects: Option<&mut ObjectTracks>) {
		let most = l.trail.round().max(0.0) as usize;
		let points = most > 0 && l.trail_of != "objects";
		let big = most > 0 && l.trail_of != "points";
		for t in &mut self.tracks {
			t.path = if points { extend(t.path.take(), t.x, t.y, most, 1.0) } else { None };
		}
		// eased, since the detector's corrections would otherwise draw as kinks
		for r in &mut self.regions {
			r.path = if big { extend(r.path.take(), r.x + r.w / 2.0, r.y + r.h, most, 0.4) } else { None };
		}
		if let Some(o) = objects {
			for t in &mut o.list {
				t.path = if big { extend(t.path.take(), t.x + t.w / 2.0, t.y + t.h, most, 0.4) } else { None };
			}
			// what's on show is a copy taken before this, so give it the new paths too
			for n in &mut self.named {
				if let Some(t) = o.list.iter().find(|t| t.id == n.id) {
					n.path = t.path.clone();
				}
			}
		}
	}

	/// The big boxes: around the subjects where there are any, otherwise around clumps of points.
	fn clumps(&self, l: &Look, sub: &Subject, unit: f32, cell: f32) -> Vec<[f32; 4]> {
		if l.focus > 0.0 && sub.active && !sub.blobs.is_empty() {
			// a touch of margin, and the same size floor a subject has to meet anyway
			let pad = CELL as f32 / 2.0 / unit;
			return sub
				.blobs
				.iter()
				.map(|b| [b.x0 / unit - pad, b.y0 / unit - pad, b.x1 / unit + pad, b.y1 / unit + pad])
				.collect();
		}
		// in the order each bin was first filled, as the site's Map keeps them, so ties sort the same
		let mut keys: Vec<i64> = Vec::new();
		let mut bins: std::collections::HashMap<i64, ([f32; 4], u32)> = std::collections::HashMap::new();
		for t in &self.tracks {
			if t.miss > 0 {
				continue;
			}
			let key = (t.y / cell).floor() as i64 * 1000 + (t.x / cell).floor() as i64;
			match bins.get_mut(&key) {
				None => {
					keys.push(key);
					bins.insert(key, ([t.x, t.y, t.x, t.y], 1));
				}
				Some((b, n)) => {
					b[0] = b[0].min(t.x);
					b[1] = b[1].min(t.y);
					b[2] = b[2].max(t.x);
					b[3] = b[3].max(t.y);
					*n += 1;
				}
			}
		}
		let mut out: Vec<([f32; 4], u32)> = keys
			.iter()
			.map(|k| bins[k])
			.filter(|(b, n)| *n as f32 >= l.region_min && b[2] > b[0] && b[3] > b[1])
			.collect();
		out.sort_by_key(|b| std::cmp::Reverse(b.1));
		out.into_iter().map(|(b, _)| b).collect()
	}

	fn boxes(&mut self, l: &Look, sub: &Subject, unit: f32) {
		if l.regions <= 0.0 {
			self.regions.clear();
			return;
		}
		let cell = l.region_size / 100.0;
		let mut clumps = self.clumps(l, sub, unit, cell);
		clumps.truncate(l.regions.round().max(0.0) as usize);

		let reach = cell.max(0.15).powi(2);
		let mut taken = vec![false; clumps.len()];
		let mut kept: Vec<Region> = Vec::new();
		let keep = (l.linger * 2.0).max(4.0);
		for mut r in std::mem::take(&mut self.regions) {
			let mut hit = None;
			let mut closest = reach;
			for (i, b) in clumps.iter().enumerate() {
				if taken[i] {
					continue;
				}
				let d =
					((b[0] + b[2]) / 2.0 - r.x - r.w / 2.0).powi(2) + ((b[1] + b[3]) / 2.0 - r.y - r.h / 2.0).powi(2);
				if d < closest {
					closest = d;
					hit = Some(i);
				}
			}
			if let Some(i) = hit {
				let b = clumps[i];
				taken[i] = true;
				let ease = 0.35;
				r.x += (b[0] - r.x) * ease;
				r.y += (b[1] - r.y) * ease;
				r.w += (b[2] - b[0] - r.w) * ease;
				r.h += (b[3] - b[1] - r.h) * ease;
				r.miss = 0;
				kept.push(r);
			} else {
				r.miss += 1;
				if r.miss as f32 <= keep {
					kept.push(r);
				}
			}
		}
		for (i, b) in clumps.iter().enumerate() {
			if !taken[i] && (kept.len() as f32) < l.regions {
				let id = take_id(&mut self.next, &mut self.base, l);
				kept.push(Region { id, x: b[0], y: b[1], w: b[2] - b[0], h: b[3] - b[1], miss: 0, path: None });
			}
		}
		self.regions = kept;
	}
}

/// Move a thing, and everywhere it's been, along with the camera.
fn carry(x: &mut f32, y: &mut f32, path: &mut Option<Vec<f32>>, sub: &Subject, unit: f32) {
	sub.carry(x, y, unit);
	if let Some(p) = path {
		for xy in p.chunks_exact_mut(2) {
			let (mut px, mut py) = (xy[0], xy[1]);
			sub.carry(&mut px, &mut py, unit);
			xy[0] = px;
			xy[1] = py;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::hash;

	#[test]
	fn hash_matches_the_site() {
		// from the site's hash(): (h >>> 0) / 4294967296, here to 24 bits
		let site = |n: i64, salt: i64| {
			let mut h = ((n as i32) ^ ((salt as i32) + 1).wrapping_mul(0x9e3779b1u32 as i32))
				.wrapping_mul(0x85ebca6bu32 as i32);
			h ^= ((h as u32) >> 13) as i32;
			h = h.wrapping_mul(0xc2b2ae35u32 as i32);
			h ^= ((h as u32) >> 16) as i32;
			(h as u32) as f64 / 4294967296.0
		};
		for (n, s) in [(37000, 1), (37001, 3), (-1, 9), (99000, 37050)] {
			assert!((hash(n, s) as f64 - site(n, s)).abs() < 1e-6);
		}
	}
}
