//! What the slow models said about each frame of the clip in the window, kept so that playing it
//! again (round the loop, or paused with a slider moving) needn't ask them again: the outline each
//! thing was given, what each part of the scene is, and how near. The look ahead already makes the
//! boxes the same on every play, so the same frame with the same boxes is the same answer, whatever
//! colours or styles change meanwhile. Only the window keeps these; exports go through once.

use crate::Rect;
use std::collections::{HashMap, HashSet};

/// the most it keeps, in bytes; past that, the frames furthest from where it's playing go
const CAP: usize = 1536 << 20;

/// A frame of the source (its number), and the part of it in view (follow cam), to a ten-thousandth.
pub type At = (i64, [i32; 4]);

/// A box, to a hundredth of a pixel.
fn key(r: Rect) -> [i32; 4] {
	[
		(r.x * 100.0).round() as i32,
		(r.y * 100.0).round() as i32,
		(r.w * 100.0).round() as i32,
		(r.h * 100.0).round() as i32,
	]
}

/// A thing's outline as the steadying keeps it (see outline.rs): its shape over its box (0-1), how
/// sure the model was, and whether it was kept for the frames after.
pub type Kept = (Vec<f32>, f32, bool);

/// An outline as kept: its shape to a 255th, how sure, and whether kept for the frames after.
type KeptShape = (Vec<u8>, f32, bool);

#[derive(Default)]
pub struct Recall {
	/// what it's keeping for (the clip, its size, its crop); something else starts it over
	context: String,
	shapes: HashMap<(At, [i32; 4]), KeptShape>,
	scenes: HashMap<At, (usize, Vec<u8>)>,
	depths: HashMap<At, (Vec<u16>, (f32, f32))>,
	bytes: usize,
	/// the most it keeps, if not `CAP`
	cap: Option<usize>,
	/// frames whose models it had the answers to, and didn't
	pub hits: u64,
	pub misses: u64,
}

impl Recall {
	/// One that keeps little: an export mixing two looks only needs a frame's answers for the other
	/// look on the same frame.
	pub fn small() -> Self {
		Self { cap: Some(256 << 20), ..Self::default() }
	}

	/// Keep for `context` from now on, forgetting everything if that's new.
	pub fn keep_for(&mut self, context: &str) {
		if self.context != context {
			*self = Self {
				context: context.to_string(),
				hits: self.hits,
				misses: self.misses,
				cap: self.cap,
				..Self::default()
			};
		}
	}

	/// Which frame source time `t` is, at `fps`, seen through `view`; none between frames (slow
	/// motion's made ones).
	pub fn at(t: f64, fps: f64, view: [f32; 4]) -> Option<At> {
		let f = t * fps;
		let i = f.round();
		((f - i).abs() < 0.02 && i >= 0.0).then(|| (i as i64, view.map(|v| (v * 10000.0).round() as i32)))
	}

	fn count(&mut self, hit: bool) {
		if hit {
			self.hits += 1;
		} else {
			self.misses += 1;
		}
	}

	/// The outlines kept for every one of `boxes` (the frame's pixels) at `at`, or none if any is missing.
	pub fn shapes(&mut self, at: At, boxes: &[Rect]) -> Option<Vec<Kept>> {
		let got: Option<Vec<Kept>> = boxes
			.iter()
			.map(|b| {
				self.shapes.get(&(at, key(*b))).map(|(p, s, k)| (p.iter().map(|&v| v as f32 / 255.0).collect(), *s, *k))
			})
			.collect();
		self.count(got.is_some());
		got
	}

	/// Keep these outlines, where there's none kept for that box yet (two looks mixed share them, and
	/// the first to ask is the one kept).
	pub fn keep_shapes(&mut self, at: At, boxes: &[Rect], kept: &[Kept]) {
		for (b, (p, s, k)) in boxes.iter().zip(kept) {
			if let std::collections::hash_map::Entry::Vacant(v) = self.shapes.entry((at, key(*b))) {
				let q: Vec<u8> = p.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
				self.bytes += q.len();
				v.insert((q, *s, *k));
			}
		}
		self.room(at.0);
	}

	/// What each part of the scene is at `at`: the grid's size and the kind in each cell.
	pub fn scene(&mut self, at: At) -> Option<(usize, &[u8])> {
		let hit = self.scenes.contains_key(&at);
		self.count(hit);
		self.scenes.get(&at).map(|(g, k)| (*g, k.as_slice()))
	}

	pub fn keep_scene(&mut self, at: At, grid: usize, kind: &[u8]) {
		self.bytes += kind.len();
		if let Some(old) = self.scenes.insert(at, (grid, kind.to_vec())) {
			self.bytes -= old.1.len();
		}
		self.room(at.0);
	}

	/// How near each part is at `at` (0-1, to a 65535th), and the range it was spread over.
	pub fn depth(&mut self, at: At) -> Option<(&[u16], (f32, f32))> {
		let hit = self.depths.contains_key(&at);
		self.count(hit);
		self.depths.get(&at).map(|(n, lo)| (n.as_slice(), *lo))
	}

	pub fn keep_depth(&mut self, at: At, near: &[f32], range: (f32, f32)) {
		let q: Vec<u16> = near.iter().map(|&v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
		self.bytes += q.len() * 2;
		if let Some(old) = self.depths.insert(at, (q, range)) {
			self.bytes -= old.0.len() * 2;
		}
		self.room(at.0);
	}

	pub fn bytes(&self) -> usize {
		self.bytes
	}

	/// Over the most it keeps: forget the frames furthest from `now` until it's a fifth under.
	fn room(&mut self, now: i64) {
		let cap = self.cap.unwrap_or(CAP);
		if self.bytes <= cap {
			return;
		}
		let mut per: HashMap<i64, usize> = HashMap::new();
		for (k, v) in &self.shapes {
			*per.entry(k.0.0).or_default() += v.0.len();
		}
		for (k, v) in &self.scenes {
			*per.entry(k.0).or_default() += v.1.len();
		}
		for (k, v) in &self.depths {
			*per.entry(k.0).or_default() += v.0.len() * 2;
		}
		let mut frames: Vec<(i64, usize)> = per.into_iter().collect();
		frames.sort_by_key(|&(f, _)| std::cmp::Reverse((f - now).abs()));
		let mut gone = HashSet::new();
		let mut left = self.bytes;
		for (f, b) in frames {
			if left <= cap / 5 * 4 {
				break;
			}
			gone.insert(f);
			left -= b;
		}
		self.shapes.retain(|k, _| !gone.contains(&k.0.0));
		self.scenes.retain(|k, _| !gone.contains(&k.0));
		self.depths.retain(|k, _| !gone.contains(&k.0));
		self.bytes = left;
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_same_frame_and_boxes_are_remembered_and_nothing_else() {
		let mut r = Recall::default();
		r.keep_for("clip 576x1024");
		let at = Recall::at(2.0 / 30.0, 30.0, crate::edit::WHOLE).unwrap();
		assert_eq!(at.0, 2);
		// between frames isn't a frame
		assert!(Recall::at(2.5 / 30.0, 30.0, crate::edit::WHOLE).is_none());
		let b = [Rect { x: 10.0, y: 20.0, w: 30.0, h: 40.0 }, Rect { x: 50.0, y: 60.0, w: 10.0, h: 10.0 }];
		r.keep_shapes(at, &b, &[(vec![0.0, 0.5, 1.0], 0.9, true), (vec![1.0], 0.7, false)]);
		let got = r.shapes(at, &b).unwrap();
		assert!((got[0].0[1] - 0.5).abs() < 0.01 && got[1].1 == 0.7 && !got[1].2);
		// a box that's moved, or another frame: ask the model
		assert!(r.shapes(at, &[Rect { x: 11.0, ..b[0] }]).is_none());
		assert!(r.shapes((3, at.1), &b[..1]).is_none());
		r.keep_scene(at, 2, &[1, 2, 3, 4]);
		assert_eq!(r.scene(at).unwrap().1, &[1, 2, 3, 4]);
		r.keep_depth(at, &[0.25, 1.0], (0.1, 0.9));
		let (n, lo) = r.depth(at).unwrap();
		assert!((n[0] as f32 / 65535.0 - 0.25).abs() < 1e-4 && lo == (0.1, 0.9));
		// another clip starts over
		r.keep_for("other 576x1024");
		assert!(r.scene(at).is_none() && r.bytes() == 0);
	}

	#[test]
	fn the_furthest_frames_go_first() {
		let mut r = Recall { cap: Some(10_000), ..Recall::default() };
		let some = vec![0.5f32; 500];
		for f in 0..30 {
			r.keep_depth((f, [0; 4]), &some, (0.0, 1.0));
		}
		assert!(r.bytes() <= 10_000);
		// it was at frame 29 when it ran out of room, so the first frames went
		assert!(r.depth((29, [0; 4])).is_some() && r.depth((0, [0; 4])).is_none());
	}
}
