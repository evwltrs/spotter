//! What the detector has found, followed over time. The detector only looks every so often, so between
//! looks each object coasts on its own velocity and rides along with the camera; each look corrects
//! them. Positions are in the same units as the point tracks (1 is the frame's shorter side).
//! A port of 1nf.me's `src/lib/tracker/objects.ts`.

use crate::Rect;
use crate::look::Look;

/// One thing the detector saw in one frame, 0-1 across the frame.
#[derive(Clone, Debug)]
pub struct Detection {
	pub cls: String,
	pub score: f32,
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
}

#[derive(Clone, Debug)]
pub struct ObjectTrack {
	pub id: i64,
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
	pub miss: u32,
	pub cls: String,
	pub conf: f32,
	pub hits: u32,
	pub vx: f32,
	pub vy: f32,
	/// video frames since the detector last confirmed it
	pub frames: u32,
	/// the score each kind has collected, in the order first seen
	pub votes: Vec<(String, f32)>,
	pub shown: bool,
	/// 0-1: how far it has appeared
	pub fade: f32,
	/// where it's been, oldest first, as x, y, x, y…
	pub path: Option<Vec<f32>>,
}

impl ObjectTrack {
	pub fn rect(&self) -> Rect {
		Rect { x: self.x, y: self.y, w: self.w, h: self.h }
	}
}

fn iou(a: Rect, b: Rect) -> f32 {
	let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
	let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
	if w <= 0.0 || h <= 0.0 {
		return 0.0;
	}
	(w * h) / (a.w * a.h + b.w * b.h - w * h)
}

/// Two boxes on one object: nearly the same box whatever it's called, or one of a kind almost wholly
/// inside another of that kind.
fn same_thing(a: &ObjectTrack, b: &ObjectTrack) -> bool {
	if iou(a.rect(), b.rect()) > 0.6 {
		return true;
	}
	if a.cls != b.cls {
		return false;
	}
	let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
	let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
	w > 0.0 && h > 0.0 && (w * h) / (a.w * a.h).min(b.w * b.h) > 0.85
}

pub const UNNUMBERED: i64 = -1;

/// rounds of the detector an object can go unseen for before it's dropped
pub const HOLD: u32 = 6;

/// How visible an object is right now: coming in when new, going out as it stops being seen.
pub fn presence(t: &ObjectTrack) -> f32 {
	t.fade * (1.0 - t.miss as f32 / (HOLD + 1) as f32)
}

#[derive(Default)]
pub struct ObjectTracks {
	pub list: Vec<ObjectTrack>,
	/// the last number given out when objects count by themselves (`ownNumbers`)
	pub counted: i64,
	/// laid out from a tape each frame (tape.rs), so neither the camera nor their speed moves them
	pub pinned: bool,
}

impl ObjectTracks {
	pub fn clear(&mut self) {
		self.list.clear();
	}

	/// One video frame has gone by.
	pub fn advance(&mut self) {
		if self.pinned {
			return;
		}
		for t in &mut self.list {
			t.x += t.vx;
			t.y += t.vy;
			t.vx *= 0.92;
			t.vy *= 0.92;
			t.frames += 1;
			if t.shown {
				t.fade = (t.fade + 0.1).min(1.0);
			}
		}
	}

	/// Paused: there are no frames for anything to fade in over.
	pub fn settle(&mut self) {
		for t in &mut self.list {
			if t.shown {
				t.fade = 1.0;
			}
		}
	}

	/// Take in a round of detections. `sx`, `sy` turn 0-1 into track units; `lag` is how many video
	/// frames ago the detector looked.
	pub fn ingest(&mut self, found: &[Detection], l: &Look, sx: f32, sy: f32, lag: f32) {
		let dets: Vec<Detection> = found
			.iter()
			.map(|d| Detection { x: d.x * sx, y: d.y * sy, w: d.w * sx, h: d.h * sy, ..d.clone() })
			.collect();
		let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
		for (ti, t) in self.list.iter().enumerate() {
			for (di, d) in dets.iter().enumerate() {
				// compared where it was when the detector looked, a frame or two ago
				let moved = Rect { x: d.x + t.vx * lag, y: d.y + t.vy * lag, w: d.w, h: d.h };
				let mut v = iou(t.rect(), moved);
				if v == 0.0 {
					// small and quick things can clear their own width between looks
					let dist = (t.x + t.w / 2.0 - moved.x - d.w / 2.0).hypot(t.y + t.h / 2.0 - moved.y - d.h / 2.0);
					if dist < t.w.max(t.h).max(d.w).max(d.h) * 0.75 {
						v = 0.12;
					}
				}
				// a car called a truck for a moment is still that car, but it takes more overlap to say so
				if d.cls != t.cls && v < 0.45 {
					continue;
				}
				if v >= 0.1 {
					pairs.push((ti, di, v + if d.cls == t.cls { 0.2 } else { 0.0 }));
				}
			}
		}
		pairs.sort_by(|a, b| b.2.total_cmp(&a.2));
		let mut t_used = vec![false; self.list.len()];
		let mut d_used = vec![false; dets.len()];

		for (ti, di, _) in pairs {
			if t_used[ti] || d_used[di] {
				continue;
			}
			t_used[ti] = true;
			d_used[di] = true;
			let t = &mut self.list[ti];
			let d = &dets[di];
			let dx = d.x + t.vx * lag + d.w / 2.0 - (t.x + t.w / 2.0);
			let dy = d.y + t.vy * lag + d.h / 2.0 - (t.y + t.h / 2.0);
			let elapsed = t.frames.max(1) as f32;
			t.vx += (dx / elapsed) * 0.25;
			t.vy += (dy / elapsed) * 0.25;
			let cx = t.x + t.w / 2.0 + dx * 0.7;
			let cy = t.y + t.h / 2.0 + dy * 0.7;
			t.w += (d.w - t.w) * 0.5;
			t.h += (d.h - t.h) * 0.5;
			t.x = cx - t.w / 2.0;
			t.y = cy - t.h / 2.0;
			t.conf += (d.score - t.conf) * 0.5;
			match t.votes.iter_mut().find(|(k, _)| *k == d.cls) {
				Some(v) => v.1 += d.score,
				None => t.votes.push((d.cls.clone(), d.score)),
			}
			let mut best = t.votes.iter().find(|(k, _)| *k == t.cls).map_or(0.0, |v| v.1);
			for (k, v) in &t.votes {
				if *v > best {
					best = *v;
					t.cls = k.clone();
				}
			}
			t.hits += 1;
			t.miss = 0;
			t.frames = 0;
		}

		let mut kept: Vec<ObjectTrack> = Vec::with_capacity(self.list.len() + dets.len());
		for (i, mut t) in std::mem::take(&mut self.list).into_iter().enumerate() {
			if t_used[i] {
				kept.push(t);
			} else {
				t.miss += 1;
				if t.miss <= HOLD {
					kept.push(t);
				}
			}
		}
		for (i, d) in dets.into_iter().enumerate() {
			if d_used[i] || d.score < l.confidence {
				continue;
			}
			kept.push(ObjectTrack {
				id: UNNUMBERED,
				votes: vec![(d.cls.clone(), d.score)],
				cls: d.cls,
				conf: d.score,
				x: d.x,
				y: d.y,
				w: d.w,
				h: d.h,
				miss: 0,
				hits: 1,
				vx: 0.0,
				vy: 0.0,
				frames: 0,
				shown: false,
				fade: 0.0,
				path: None,
			});
		}
		self.list = kept;
	}

	/// The ones to draw: seen twice, or sure enough the first time; the ones already up get first claim
	/// on the places. Anything newly shown gets a number: its own count with `ownNumbers`, otherwise
	/// from `next`, the count the points share.
	pub fn shown(&mut self, l: &Look, mut next: impl FnMut() -> i64) -> Vec<ObjectTrack> {
		let rank = |t: &ObjectTrack| t.conf + if t.shown { 0.15 } else { 0.0 };
		let mut ready: Vec<usize> =
			(0..self.list.len()).filter(|&i| self.list[i].hits >= 2 || self.list[i].conf >= 0.65).collect();
		ready.sort_by(|&a, &b| rank(&self.list[b]).total_cmp(&rank(&self.list[a])));
		// the same thing found twice: the surer one speaks for it
		let mut kept: Vec<usize> = Vec::new();
		for i in ready {
			// (a tape has already told doubles from things that overlap)
			if !self.pinned && kept.iter().any(|&k| same_thing(&self.list[k], &self.list[i])) {
				self.list[i].shown = false;
			} else {
				kept.push(i);
			}
		}
		let most = l.objects.round().max(0.0) as usize;
		for (n, &i) in kept.iter().enumerate() {
			let t = &mut self.list[i];
			t.shown = n < most;
			if t.shown && t.id == UNNUMBERED {
				t.id = if l.own_numbers {
					self.counted += 1;
					self.counted
				} else {
					next()
				};
			}
		}
		kept.iter().take(most).map(|&i| self.list[i].clone()).collect()
	}
}
