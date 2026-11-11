//! Looking ahead. Drawing frames as they come, the way the site does, things have to be followed
//! cautiously: a box only moves part of the way to where the detector says, a thing must be seen
//! twice before it shows, and nothing knows the footage has cut until boxes stop being found. So
//! everything lags, and boxes from one shot ride over the next. An export (or a preview that has
//! looked ahead) can see the whole clip before it draws any of it, so it doesn't have to guess:
//!
//! - `Looker` goes over the clip first, the detector on every frame, and notes where the footage cuts.
//!   That's the `Record`, kept with every detection above a low floor so any look's settings can be
//!   laid over it later without looking again.
//! - `Tape::build` follows the things through the record with hindsight, cheaply enough to redo
//!   whenever a setting changes: a cut ends everything, a thing is there from the first frame it was
//!   seen to the last, short gaps are filled in, boxes are smoothed both ways in time (so they never
//!   trail behind), things only seen for a moment are dropped, and each is called whatever it was
//!   called most surely over its whole life.

use crate::Frame;
use crate::Rect;
use crate::detector::{Detector, wanted};
use crate::edit::{Edit, Pick, View, WHOLE};
use crate::look::Look;
use crate::objects::Detection;
use anyhow::Result;

/// The lowest score kept when looking, under any look's line, so the line can be drawn later.
pub const FLOOR: f32 = 0.2;

/// What the detector saw on one frame, 0-1 across the source, and how different the frame was from
/// the one before.
#[derive(Clone, Debug)]
pub struct Seen {
	/// seconds into the source
	pub t: f64,
	/// the frame doesn't follow on from the last one looked at: the start, or a stretch after a cut
	/// taken out
	pub jump: bool,
	/// how different it is from the last frame (`Looker::change`)
	pub change: Change,
	pub found: Vec<Detection>,
}

/// How different a frame is from the one before.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Change {
	/// the mean difference of small grey copies of the two, lined up as well as they'll go, in percent
	pub moved: f32,
	/// how far apart their colours are, 0-1: how much of one's histogram would have to move to make
	/// the other's
	pub colours: f32,
}

/// Which frames start a new shot, from how much each differed from the one before. A cut is a spike:
/// a frame far more different from the last, even lined up, than its neighbours are from theirs, the
/// colours changing too; or, into a moving shot, a leap from stillness the next frames only come down
/// from. A camera jerking, a light flaring, quick steady movement and a slow fade aren't cuts. Checked
/// against every cut in a set of test clips, and the false ones of a shaky walk along a platform.
pub fn cuts(change: &[Change], jumps: &[bool]) -> Vec<bool> {
	let n = change.len();
	let mut out = vec![false; n];
	for i in 0..n {
		if i == 0 || jumps[i] {
			out[i] = true;
			continue;
		}
		// a flash right after a cut isn't another one
		if out[i - 1] || (i > 1 && out[i - 2]) {
			continue;
		}
		let around = |k: isize| {
			let j = i as isize + k;
			if j < 1 || j >= n as isize || jumps[j as usize] { 0.0 } else { change[j as usize].moved }
		};
		let Change { moved: d, colours: c } = change[i];
		let (before, after) = (around(-1), around(1));
		let spike = d / before.max(after).max(0.3);
		out[i] = (d > 2.5 && ((spike >= 2.8 && c >= 0.2) || (spike >= 2.0 && c >= 0.4)))
			|| (d > 10.0 && d > 8.0 * before && d >= after);
	}
	out
}

/// Everything seen over a clip.
#[derive(Clone, Debug, Default)]
pub struct Record {
	pub frames: Vec<Seen>,
	/// frames a second of the source it was looked at
	pub fps: f64,
	/// the source's width over its height
	pub aspect: f32,
	/// the part of the source the detector looked at, 0-1 across it
	pub view: View,
	/// which detector looked
	pub model: String,
}

/// The small grey copy frames are compared in, and how far (in its pixels) one is slid over the last
/// to find where they line up.
const THUMB: (usize, usize) = (128, 72);
const SLIDE: isize = 8;
/// a histogram's bins for each of red, green and blue
const BINS: usize = 16;

/// Goes over a clip a frame at a time: what's in each frame, and how much it changed.
pub struct Looker {
	thumb: Frame,
	grey: Vec<u8>,
	last: Vec<u8>,
	hist: Vec<u32>,
	was: Vec<u32>,
}

impl Default for Looker {
	fn default() -> Self {
		Self {
			thumb: Frame::new(THUMB.0 as u32, THUMB.1 as u32),
			grey: Vec::new(),
			last: Vec::new(),
			hist: vec![0; BINS * 3],
			was: Vec::new(),
		}
	}
}

impl Looker {
	/// Look at the next frame, `view` being where it sits in the source. `jump` is a frame that
	/// doesn't follow on from the last one (the start, or a new stretch after a cut is taken out).
	pub fn look(&mut self, d: &mut Detector, frame: &Frame, t: f64, view: View, jump: bool) -> Result<Seen> {
		let change = self.change(frame);
		let [vx, vy, vw, vh] = view;
		let found = d
			.detect(frame, FLOOR + 0.12, "everything")?
			.into_iter()
			.map(|f| Detection { x: vx + f.x * vw, y: vy + f.y * vh, w: f.w * vw, h: f.h * vh, ..f })
			.collect();
		Ok(Seen { t, jump, change, found })
	}

	/// How much this frame differs from the last: small grey copies of the two slid over each other to
	/// where they line up best, so a camera that jerks or pans changes little and a new shot a lot; and
	/// how far apart their colours are.
	pub fn change(&mut self, frame: &Frame) -> Change {
		frame.resize_into(&mut self.thumb);
		self.grey.clear();
		self.grey.extend(
			self.thumb
				.rgba
				.chunks_exact(4)
				.map(|p| ((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8),
		);
		self.hist.iter_mut().for_each(|v| *v = 0);
		for p in self.thumb.rgba.chunks_exact(4) {
			for (c, &v) in p.iter().take(3).enumerate() {
				self.hist[c * BINS + v as usize * BINS / 256] += 1;
			}
		}
		let colours = if self.was.len() == self.hist.len() {
			let moved: u32 = self.hist.iter().zip(&self.was).map(|(a, b)| a.abs_diff(*b)).sum();
			moved as f32 / (2 * 3 * THUMB.0 * THUMB.1) as f32
		} else {
			0.0
		};
		self.was.clone_from(&self.hist);
		if self.last.len() != self.grey.len() {
			self.last = self.grey.clone();
			return Change::default();
		}
		let (w, h) = (THUMB.0 as isize, THUMB.1 as isize);
		let (a, b) = (&self.grey, &self.last);
		let mut best = f32::MAX;
		for dy in -SLIDE..=SLIDE {
			for dx in -SLIDE..=SLIDE {
				let (x0, x1) = (0.max(-dx), w.min(w - dx));
				let (y0, y1) = (0.max(-dy), h.min(h - dy));
				let mut sad = 0u32;
				for y in y0..y1 {
					let (ra, rb) = ((y * w) as usize, ((y + dy) * w + dx) as usize);
					for x in x0 as usize..x1 as usize {
						sad += (a[ra + x] as i32 - b[rb + x] as i32).unsigned_abs();
					}
				}
				let mean = sad as f32 / ((x1 - x0) * (y1 - y0)) as f32;
				best = best.min(mean);
			}
		}
		std::mem::swap(&mut self.grey, &mut self.last);
		Change { moved: best * 100.0 / 255.0, colours }
	}
}

/// A thing followed through the clip, from its first frame to its last.
#[derive(Clone, Debug)]
pub struct Thing {
	pub cls: String,
	/// how sure the detector was of it, on average
	pub conf: f32,
	pub first: usize,
	/// where it is on each frame from `first` on, 0-1 across the source
	pub boxes: Vec<Rect>,
	/// how far it's shown on each frame: it comes in and goes out over a few frames, except at a cut
	pub fade: Vec<f32>,
}

impl Thing {
	pub fn last(&self) -> usize {
		self.first + self.boxes.len() - 1
	}
}

/// The things in a clip, worked out with hindsight.
#[derive(Clone, Debug, Default)]
pub struct Tape {
	pub fps: f64,
	/// the source's width over its height
	pub aspect: f32,
	/// each frame's time in the source
	pub times: Vec<f64>,
	/// the frames a new shot starts on
	pub cuts: Vec<bool>,
	pub things: Vec<Thing>,
	/// which things are there on each frame
	alive: Vec<Vec<usize>>,
	/// on each frame, how much the detector's things should take over from the boxes around movement:
	/// 1 while there are any, and leaning in for half a second either side within the shot
	pub blend: Vec<f32>,
	/// the source time it's sure of up to: a record still being made is only sure of what's well
	/// behind the newest frame, since things there may yet carry on
	pub until: f64,
	/// things the edit took out (its drops), left out of everything else and kept here for the
	/// timeline to show, to bring back
	pub dropped: Vec<Dropped>,
}

/// A thing taken out by hand: it, where a pick (or drop) on it lands, and the edit's drops that
/// meant it.
#[derive(Clone, Debug)]
pub struct Dropped {
	pub thing: Thing,
	pub pick: Pick,
	pub drops: Vec<Pick>,
}

/// Which things are on each frame, and how far the boxes lean to the detector's on each (1 while
/// there are any, leaning in for half a second either side within the shot).
fn lay(things: &[Thing], n: usize, cuts: &[bool], fps: f64) -> (Vec<Vec<usize>>, Vec<f32>) {
	let mut alive = vec![Vec::new(); n];
	for (k, t) in things.iter().enumerate() {
		for a in &mut alive[t.first..=t.last()] {
			a.push(k);
		}
	}
	let reach = ((0.5 * fps).round() as usize).max(1);
	let mut near = vec![usize::MAX; n];
	let mut since = usize::MAX;
	for i in 0..n {
		since = if !alive[i].is_empty() {
			0
		} else if cuts[i] {
			usize::MAX
		} else {
			since.saturating_add(1)
		};
		near[i] = since;
	}
	let mut until = usize::MAX;
	for i in (0..n).rev() {
		until = if !alive[i].is_empty() { 0 } else { until.saturating_add(1) };
		near[i] = near[i].min(until);
		if cuts[i] {
			until = usize::MAX;
		}
	}
	let blend =
		near.iter().map(|&d| if d == usize::MAX { 0.0 } else { (1.0 - d as f32 / reach as f32).max(0.0) }).collect();
	(alive, blend)
}

/// Overlap over union.
pub fn iou(a: Rect, b: Rect) -> f32 {
	let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
	let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
	if w <= 0.0 || h <= 0.0 {
		return 0.0;
	}
	(w * h) / (a.w * a.h + b.w * b.h - w * h)
}

fn rect(d: &Detection) -> Rect {
	Rect { x: d.x, y: d.y, w: d.w, h: d.h }
}

/// A thing being followed while the tape's being made.
struct Open {
	/// (frame, box, score, kind) for each frame it was found on
	seen: Vec<(usize, Rect, f32, String)>,
	/// how far its middle moves a frame
	v: (f32, f32),
}

impl Open {
	fn last(&self) -> (usize, Rect) {
		let s = self.seen.last().unwrap();
		(s.0, s.1)
	}

	/// Where it should be on frame `i`, carried on at the speed it was going.
	fn guess(&self, i: usize) -> Rect {
		let (at, r) = self.last();
		let n = (i - at) as f32;
		Rect { x: r.x + self.v.0 * n, y: r.y + self.v.1 * n, ..r }
	}
}

/// Smooth `v` with a Gaussian of `sigma` samples, both ways, the ends weighed only by what's there.
fn smooth(v: &[f32], sigma: f32) -> Vec<f32> {
	if sigma < 0.3 || v.len() < 3 {
		return v.to_vec();
	}
	let reach = (sigma * 3.0).ceil() as isize;
	let weights: Vec<f32> = (-reach..=reach).map(|k| (-(k * k) as f32 / (2.0 * sigma * sigma)).exp()).collect();
	let n = v.len() as isize;
	(0..n)
		.map(|i| {
			let (mut sum, mut total) = (0.0, 0.0);
			for k in -reach..=reach {
				let j = i + k;
				if j >= 0 && j < n {
					let w = weights[(k + reach) as usize];
					sum += v[j as usize] * w;
					total += w;
				}
			}
			sum / total
		})
		.collect()
}

impl Tape {
	/// Follow the things in `r` as the look and edit ask: their confidence and kinds, the edit's kinds
	/// and zones, how many to show is left to the drawing.
	pub fn build(r: &Record, l: &Look, edit: &Edit) -> Tape {
		let n = r.frames.len();
		let fps = r.fps.max(1.0);
		let aspect = r.aspect.max(0.01);
		let high = l.confidence;
		let low = (high - 0.12).max(FLOOR);
		// a thing can go unseen this long and still be the same thing
		let gap = ((0.4 * fps).round() as usize).max(3);
		let cuts = cuts(
			&r.frames.iter().map(|f| f.change).collect::<Vec<_>>(),
			&r.frames.iter().map(|f| f.jump).collect::<Vec<_>>(),
		);
		// each frame's shot: where it starts and ends
		let mut shot = vec![(0usize, 0usize); n];
		let mut a = 0;
		for i in 0..n {
			if cuts[i] {
				a = i;
			}
			shot[i].0 = a;
		}
		let mut b = n.saturating_sub(1);
		for i in (0..n).rev() {
			shot[i].1 = b;
			if cuts[i] && i > 0 {
				b = i - 1;
			}
		}
		// distances and sizes on the same footing across and down
		let far =
			|p: Rect, q: Rect| ((p.x + p.w / 2.0 - q.x - q.w / 2.0) * aspect).hypot(p.y + p.h / 2.0 - q.y - q.h / 2.0);
		let size = |p: Rect| (p.w * aspect).max(p.h);

		let mut open: Vec<Open> = Vec::new();
		let mut done: Vec<Open> = Vec::new();
		for (i, f) in r.frames.iter().enumerate() {
			if cuts[i] {
				done.append(&mut open);
			}
			// the ones lost too long are finished
			let (gone, kept): (Vec<Open>, Vec<Open>) = open.into_iter().partition(|o| i - o.last().0 > gap);
			done.extend(gone);
			open = kept;

			let mut found: Vec<&Detection> = f
				.found
				.iter()
				.filter(|d| d.score >= low && wanted(&l.object_kinds, &d.cls) && edit.admits(d, WHOLE))
				.collect();
			found.sort_by(|a, b| b.score.total_cmp(&a.score));
			// the same thing found twice (a car that's also a truck, or a person and an unsure part of
			// them): the surer one speaks for it. Two people one behind the other, both found surely,
			// are two people.
			let mut dets: Vec<&Detection> = Vec::new();
			for d in found {
				let inside = |k: &&Detection| {
					let (p, q) = (rect(k), rect(d));
					let w = (p.x + p.w).min(q.x + q.w) - p.x.max(q.x);
					let h = (p.y + p.h).min(q.y + q.h) - p.y.max(q.y);
					k.cls == d.cls
						&& d.score < k.score * 0.75
						&& w > 0.0 && h > 0.0
						&& w * h / (p.w * p.h).min(q.w * q.h) > 0.85
				};
				if !dets.iter().any(|k| iou(rect(k), rect(d)) > 0.7 || inside(k)) {
					dets.push(d);
				}
			}

			let mut pairs: Vec<(usize, usize, f32)> = Vec::new();
			for (oi, o) in open.iter().enumerate() {
				let guess = o.guess(i);
				let kind = &o.seen.last().unwrap().3;
				for (di, d) in dets.iter().enumerate() {
					let r = rect(d);
					let mut v = iou(guess, r);
					// small quick things can clear their own width between frames
					if v < 0.1 && far(guess, r) < size(guess).max(size(r)) * 0.6 {
						v = 0.1;
					}
					if d.cls != *kind && v < 0.45 {
						continue;
					}
					if v >= 0.1 {
						pairs.push((
							oi,
							di,
							v + if d.cls == *kind { 0.2 } else { 0.0 } + if d.score >= high { 0.1 } else { 0.0 },
						));
					}
				}
			}
			pairs.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
			let mut o_used = vec![false; open.len()];
			let mut d_used = vec![false; dets.len()];
			for (oi, di, _) in pairs {
				if o_used[oi] || d_used[di] {
					continue;
				}
				o_used[oi] = true;
				d_used[di] = true;
				let o = &mut open[oi];
				let d = dets[di];
				let (at, was) = o.last();
				let steps = (i - at) as f32;
				let r = rect(d);
				let moved =
					((r.x + r.w / 2.0 - was.x - was.w / 2.0) / steps, (r.y + r.h / 2.0 - was.y - was.h / 2.0) / steps);
				o.v =
					if o.seen.len() == 1 { moved } else { (o.v.0 * 0.5 + moved.0 * 0.5, o.v.1 * 0.5 + moved.1 * 0.5) };
				o.seen.push((i, r, d.score, d.cls.clone()));
			}
			// only sure finds start something new; unsure ones only carry on what's already followed
			for (di, d) in dets.iter().enumerate() {
				if !d_used[di] && d.score >= high {
					open.push(Open { seen: vec![(i, rect(d), d.score, d.cls.clone())], v: (0.0, 0.0) });
				}
			}
		}
		done.append(&mut open);
		// in order of first appearance, then left to right, so the same clip always numbers the same
		done.sort_by(|a, b| a.seen[0].0.cmp(&b.seen[0].0).then(a.seen[0].1.x.total_cmp(&b.seen[0].1.x)));

		let sigma = (l.ahead_smooth.max(0.0) as f64 * fps) as f32;
		let ramp = ((0.1 * fps).round() as usize).max(1);
		let mut things = Vec::new();
		for o in done {
			let first = o.seen[0].0;
			let last = o.seen.last().unwrap().0;
			let sure = o.seen.iter().filter(|s| s.2 >= high).count();
			let (sa, sb) = (shot[first].0, shot[first].1);
			// seen surely on a few frames, and there for a third of a second, or most of a shot too short
			// for that: anything briefer pops in and out
			let shot_len = sb - sa + 1;
			if sure < 3.min(shot_len) || ((last - first + 1) as f64) < (0.3 * fps).min(0.6 * shot_len as f64) {
				continue;
			}
			// gaps filled in a straight line from one sighting to the next
			let mut boxes = Vec::with_capacity(last - first + 1);
			for w in o.seen.windows(2) {
				let ((i0, a, ..), (i1, b, ..)) = (&w[0], &w[1]);
				for i in *i0..*i1 {
					let k = (i - i0) as f32 / (i1 - i0) as f32;
					boxes.push(Rect {
						x: a.x + (b.x - a.x) * k,
						y: a.y + (b.y - a.y) * k,
						w: a.w + (b.w - a.w) * k,
						h: a.h + (b.h - a.h) * k,
					});
				}
			}
			boxes.push(o.seen.last().unwrap().1);
			// smoothed both ways in time: the middle a little, the size (which wobbles more) twice as much
			let cx = smooth(&boxes.iter().map(|b| b.x + b.w / 2.0).collect::<Vec<_>>(), sigma);
			let cy = smooth(&boxes.iter().map(|b| b.y + b.h / 2.0).collect::<Vec<_>>(), sigma);
			let w = smooth(&boxes.iter().map(|b| b.w).collect::<Vec<_>>(), sigma * 2.0);
			let h = smooth(&boxes.iter().map(|b| b.h).collect::<Vec<_>>(), sigma * 2.0);
			let boxes: Vec<Rect> = (0..boxes.len())
				.map(|k| Rect { x: cx[k] - w[k] / 2.0, y: cy[k] - h[k] / 2.0, w: w[k], h: h[k] })
				.collect();
			// what it's called: the kind with the most score over its whole life
			let mut votes: Vec<(&str, f32)> = Vec::new();
			for s in &o.seen {
				match votes.iter_mut().find(|v| v.0 == s.3) {
					Some(v) => v.1 += s.2,
					None => votes.push((&s.3, s.2)),
				}
			}
			let cls = votes.iter().fold(votes[0], |best, v| if v.1 > best.1 { *v } else { best }).0.to_string();
			let conf = o.seen.iter().map(|s| s.2).sum::<f32>() / o.seen.len() as f32;
			// in and out over a few frames, unless the shot starts or ends with it there
			let len = boxes.len();
			let fade = (0..len)
				.map(|k| {
					let i = first + k;
					let inn =
						if first == shot[first].0 { 1.0 } else { ((i - first + 1) as f32 / ramp as f32).min(1.0) };
					let out = if last == shot[last].1 { 1.0 } else { ((last - i + 1) as f32 / ramp as f32).min(1.0) };
					inn.min(out)
				})
				.collect();
			things.push(Thing { cls, conf, first, boxes, fade });
		}

		let (alive, blend) = lay(&things, n, &cuts, fps);
		let mut tape = Tape {
			fps,
			aspect,
			times: r.frames.iter().map(|f| f.t).collect(),
			cuts,
			things,
			alive,
			blend,
			until: f64::MAX,
			dropped: Vec::new(),
		};
		// things taken out by hand: found as picks are, then left out of everything, kept aside
		let mut gone: Vec<(usize, Vec<Pick>)> = Vec::new();
		for d in &edit.drops {
			if let Some(k) = tape.pick_of(d) {
				match gone.iter_mut().find(|g| g.0 == k) {
					Some(g) => g.1.push(*d),
					None => gone.push((k, vec![*d])),
				}
			}
		}
		if !gone.is_empty() {
			let picks: Vec<Pick> = gone.iter().map(|g| tape.pick_for(g.0)).collect();
			let mut kept = Vec::new();
			for (k, th) in std::mem::take(&mut tape.things).into_iter().enumerate() {
				match gone.iter().position(|g| g.0 == k) {
					Some(j) => tape.dropped.push(Dropped { thing: th, pick: picks[j], drops: gone[j].1.clone() }),
					None => kept.push(th),
				}
			}
			let (alive, blend) = lay(&kept, n, &tape.cuts, fps);
			tape.things = kept;
			tape.alive = alive;
			tape.blend = blend;
		}
		tape
	}

	/// The frame nearest `t` seconds into the source, if the tape covers it.
	pub fn at(&self, t: f64) -> Option<usize> {
		if t > self.until {
			return None;
		}
		let i = self.times.partition_point(|&x| x < t);
		let best = [i.checked_sub(1), (i < self.times.len()).then_some(i)]
			.into_iter()
			.flatten()
			.min_by(|&a, &b| (self.times[a] - t).abs().total_cmp(&(self.times[b] - t).abs()))?;
		((self.times[best] - t).abs() <= 1.5 / self.fps).then_some(best)
	}

	/// Whether a shot starts on any frame after `from` up to and including `to`.
	pub fn cut_in(&self, from: usize, to: usize) -> bool {
		to > from && self.cuts[from + 1..=to.min(self.cuts.len() - 1)].iter().any(|&c| c)
	}

	/// The things on frame `i`: which, where (0-1 across the source) and how far shown.
	pub fn here(&self, i: usize) -> impl Iterator<Item = (usize, Rect, f32)> + '_ {
		self.alive.get(i).into_iter().flatten().map(move |&k| {
			let t = &self.things[k];
			(k, t.boxes[i - t.first], t.fade[i - t.first])
		})
	}

	/// What a camera should aim at on frame `i`, 0-1 across the source: the things among `which` that
	/// are there, together, or failing those the biggest thing.
	pub fn aim(&self, i: usize, which: &[usize]) -> Option<Rect> {
		let chosen: Vec<Rect> = self.here(i).filter(|t| which.contains(&t.0)).map(|t| t.1).collect();
		let rects = if chosen.is_empty() {
			self.here(i).map(|t| t.1).max_by(|a, b| (a.w * a.h).total_cmp(&(b.w * b.h))).into_iter().collect()
		} else {
			chosen
		};
		let first = rects.first()?;
		let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x + first.w, first.y + first.h);
		for r in &rects[1..] {
			(x0, y0, x1, y1) = (x0.min(r.x), y0.min(r.y), x1.max(r.x + r.w), y1.max(r.y + r.h));
		}
		Some(Rect { x: x0, y: y0, w: x1 - x0, h: y1 - y0 })
	}

	/// The thing a pick means, for its whole life: the smallest thing there when and where the pick
	/// was made, or the nearest close by.
	pub fn pick_of(&self, p: &Pick) -> Option<usize> {
		let aspect = self.aspect;
		let i = self.at(p.t)?;
		let inside = |r: Rect| p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
		self.here(i)
			.filter(|t| inside(t.1))
			.min_by(|a, b| (a.1.w * a.1.h).total_cmp(&(b.1.w * b.1.h)))
			.map(|t| t.0)
			.or_else(|| {
				self.here(i)
					.map(|t| (t.0, ((t.1.x + t.1.w / 2.0 - p.x) * aspect).hypot(t.1.y + t.1.h / 2.0 - p.y)))
					.filter(|t| t.1 < 0.12)
					.min_by(|a, b| a.1.total_cmp(&b.1))
					.map(|t| t.0)
			})
	}

	/// The things the edit's picks mean, once each.
	pub fn picked(&self, edit: &Edit) -> Vec<usize> {
		let mut out = Vec::new();
		for k in edit.picks.iter().filter_map(|p| self.pick_of(p)) {
			if !out.contains(&k) {
				out.push(k);
			}
		}
		out
	}

	/// A pick that means thing `k` and nothing else: the middle of its box, on the frame nearest the
	/// middle of its life where that isn't on something smaller in front of it.
	pub fn pick_for(&self, k: usize) -> Pick {
		let th = &self.things[k];
		let n = th.boxes.len();
		let mid = n / 2;
		let at = |j: usize| {
			let b = th.boxes[j];
			Pick { t: self.times[(th.first + j).min(self.times.len() - 1)], x: b.x + b.w / 2.0, y: b.y + b.h / 2.0 }
		};
		for d in 0..n {
			for j in [mid.checked_sub(d), (d > 0 && mid + d < n).then_some(mid + d)].into_iter().flatten() {
				let p = at(j);
				if self.pick_of(&p) == Some(k) {
					return p;
				}
			}
		}
		at(mid)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn det(cls: &str, score: f32, x: f32, y: f32) -> Detection {
		Detection { cls: cls.into(), score, x, y, w: 0.1, h: 0.1 }
	}

	fn record(frames: Vec<(bool, Vec<Detection>)>) -> Record {
		Record {
			frames: frames
				.into_iter()
				.enumerate()
				.map(|(i, (cut, found))| Seen {
					t: i as f64 / 30.0,
					jump: false,
					change: Change { moved: if cut { 40.0 } else { 1.0 }, colours: if cut { 0.5 } else { 0.01 } },
					found,
				})
				.collect(),
			fps: 30.0,
			aspect: 1.0,
			view: WHOLE,
			model: String::new(),
		}
	}

	#[test]
	fn a_cut_ends_everything_and_flickers_go() {
		let mut frames = Vec::new();
		for i in 0..20 {
			let mut f = vec![det("person", 0.9, 0.2 + i as f32 * 0.01, 0.4)];
			// seen once, never again
			if i == 5 {
				f.push(det("car", 0.9, 0.7, 0.7));
			}
			frames.push((i == 10, f));
		}
		let tape = Tape::build(&record(frames), &Look::default(), &Edit::default());
		// the person before the cut, and the "same" person after it, are two things; the car isn't one
		assert_eq!(tape.things.len(), 2);
		assert_eq!((tape.things[0].first, tape.things[0].last()), (0, 9));
		assert_eq!((tape.things[1].first, tape.things[1].last()), (10, 19));
		// both start with a shot, so they're there at once
		assert_eq!(tape.things[1].fade[0], 1.0);
		assert!(tape.cut_in(9, 10) && !tape.cut_in(10, 15));
	}

	#[test]
	fn a_pick_from_a_thing_means_only_it() {
		// a person, with a dog in front of their middle for the first half of the clip
		let mut frames = Vec::new();
		for i in 0..20 {
			let mut f = vec![Detection { cls: "person".into(), score: 0.9, x: 0.3, y: 0.2, w: 0.3, h: 0.6 }];
			if i < 12 {
				f.push(Detection { cls: "dog".into(), score: 0.9, x: 0.4, y: 0.45, w: 0.1, h: 0.1 });
			}
			frames.push((false, f));
		}
		let tape = Tape::build(&record(frames), &Look::default(), &Edit::default());
		assert_eq!(tape.things.len(), 2);
		let person = tape.things.iter().position(|t| t.cls == "person").unwrap();
		let dog = 1 - person;
		// the middle of the person's life has the dog in front, so the pick moves on to where it isn't
		let p = tape.pick_for(person);
		assert_eq!(tape.pick_of(&p), Some(person));
		assert!(p.t >= 12.0 / 30.0, "{p:?}");
		assert_eq!(tape.pick_of(&tape.pick_for(dog)), Some(dog));
		// two picks of the same thing are one
		let e = Edit { picks: vec![p, Pick { t: 0.6, ..p }], ..Edit::default() };
		assert_eq!(tape.picked(&e), vec![person]);
	}

	#[test]
	fn a_dropped_thing_is_left_out_and_kept_aside() {
		// a person all the way through, and a "car" (a false positive) in the second half
		let mut frames = Vec::new();
		for i in 0..20 {
			let mut f = vec![det("person", 0.9, 0.2, 0.4)];
			if i >= 10 {
				f.push(det("car", 0.9, 0.7, 0.7));
			}
			frames.push((false, f));
		}
		let r = record(frames);
		let all = Tape::build(&r, &Look::default(), &Edit::default());
		assert_eq!(all.things.len(), 2);
		let car = all.things.iter().position(|t| t.cls == "car").unwrap();
		let e = Edit { drops: vec![all.pick_for(car)], ..Edit::default() };
		let some = Tape::build(&r, &Look::default(), &e);
		assert_eq!(some.things.len(), 1);
		assert_eq!(some.things[0].cls, "person");
		assert_eq!(some.dropped.len(), 1);
		assert_eq!(some.dropped[0].thing.cls, "car");
		// not there on any frame, and a pick on it now finds nothing
		assert!((0..20).all(|i| some.here(i).all(|t| some.things[t.0].cls == "person")));
		assert_eq!(some.pick_of(&some.dropped[0].pick), None);
	}

	#[test]
	fn gaps_fill_and_unsure_finds_carry_on() {
		let mut frames = Vec::new();
		for i in 0..12 {
			let x = 0.1 + i as f32 * 0.02;
			let f = match i {
				4 | 5 => vec![],
				6 => vec![det("dog", 0.4, x, 0.5)],
				_ => vec![det("dog", 0.8, x, 0.5)],
			};
			frames.push((false, f));
		}
		let tape = Tape::build(&record(frames), &Look::default(), &Edit::default());
		assert_eq!(tape.things.len(), 1);
		let t = &tape.things[0];
		assert_eq!((t.first, t.last()), (0, 11));
		// the gap is filled where it should be
		assert!((t.boxes[4].x - 0.18).abs() < 0.01, "{}", t.boxes[4].x);
		assert_eq!(tape.at(4.0 / 30.0), Some(4));
		assert_eq!(tape.at(10.0), None);
	}

	#[test]
	fn cuts_are_spikes() {
		let jumps = vec![false; 8];
		let ch = |v: &[f32], c: f32| {
			v.iter().map(|&m| Change { moved: m, colours: if m > 20.0 { c } else { 0.02 } }).collect::<Vec<_>>()
		};
		// steady quick movement, then a cut
		assert_eq!(
			cuts(&ch(&[0.0, 6.0, 7.0, 6.5, 30.0, 6.0, 6.0, 7.0], 0.4), &jumps),
			vec![true, false, false, false, true, false, false, false]
		);
		// the same spike with the colours staying put is a flare, not a cut
		assert_eq!(
			cuts(&ch(&[0.0, 6.0, 7.0, 6.5, 30.0, 6.0, 6.0, 7.0], 0.1), &jumps).iter().filter(|&&c| c).count(),
			1
		);
		// from stillness into a quick pan
		assert!(cuts(&ch(&[0.0, 0.3, 0.2, 0.2, 25.0, 18.0, 12.0, 8.0], 0.1), &jumps)[4]);
		// a slow fade
		assert_eq!(cuts(&ch(&[0.0, 2.0, 4.0, 6.0, 8.0, 6.0, 4.0, 2.0], 0.4), &jumps).iter().filter(|&&c| c).count(), 1);
		let mut l = Looker::default();
		let flat = |v: u8| Frame::from_rgba(96, 54, [v, v, v, 255].repeat(96 * 54));
		assert_eq!(l.change(&flat(30)), Change::default());
		let c = l.change(&flat(81));
		assert!((c.moved - 20.0).abs() < 0.01 && (c.colours - 1.0).abs() < 0.01, "{c:?}");
	}
}
