//! What's done to one clip, as against how it looks: which kinds of thing are recognised, which of
//! them are followed, which parts of the picture are left alone, and which stretches of time are cut
//! out or left bare. A look is a style and can go on any clip; an edit belongs to its clip.
//!
//! Places are 0-1 across the whole source picture (before any cut to shape), and times are seconds of
//! the source, so an edit made in the preview means the same in an export at any shape or size.

use crate::Rect;
use crate::objects::{Detection, ObjectTrack};
use crate::tracking::{Region, Track};
use serde::{Deserialize, Serialize};

/// Which of the things found are followed and drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Keep {
	/// everything, as the look says
	#[default]
	All,
	/// only what's been picked, by clicking on it
	Picked,
	/// only the biggest thing, held onto until something much bigger comes along
	Main,
	/// only the thing nearest the middle of the picture, held onto while it lasts
	Centre,
}

/// A thing picked by clicking on it: when, and where.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pick {
	pub t: f64,
	pub x: f32,
	pub y: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ZoneMode {
	/// nothing in here is followed or recognised
	#[default]
	Ignore,
	/// only things in here (and in any other "only" zone) are
	Only,
}

/// A part of the picture, 0-1 across the source.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Zone {
	pub x: f32,
	pub y: f32,
	pub w: f32,
	pub h: f32,
	#[serde(default)]
	pub mode: ZoneMode,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RangeMode {
	/// taken out of the export altogether
	#[default]
	Cut,
	/// kept, with nothing drawn over it
	Bare,
	/// kept and drawn, but without the look mixed in (the look alone)
	Unmix,
}

/// A stretch of the source, in seconds.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Range {
	pub from: f64,
	pub to: f64,
	#[serde(default)]
	pub mode: RangeMode,
}

/// How long the drawing takes to fade out at the start of a bare stretch, and back in at its end.
pub const BARE_FADE: f64 = 0.3;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Edit {
	/// kinds of thing to recognise, by the detector's names ("person", "car"…); empty is whatever the
	/// look's "look for" says
	pub kinds: Vec<String>,
	pub keep: Keep,
	pub picks: Vec<Pick>,
	/// things taken out by hand (false positives, say), found as picks are: where and when
	pub drops: Vec<Pick>,
	pub zones: Vec<Zone>,
	pub ranges: Vec<Range>,
	/// a song under the video
	pub music: Option<Music>,
}

/// A song to play under the video, in place of the clip's own sound or with it. It runs on the
/// output's clock: from the in point, straight on over stretches cut out, and at its own speed when
/// the picture's slowed down.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Music {
	/// the song: a file or a web address
	pub path: String,
	/// seconds into the song where the video starts
	pub from: f64,
	/// how loud the song is: 1 as it is
	pub gain: f32,
	/// how much of the clip's own sound stays: 0 none, 1 all of it
	pub keep: f32,
}

impl Default for Music {
	fn default() -> Self {
		Self { path: String::new(), from: 0.0, gain: 1.0, keep: 0.0 }
	}
}

impl Music {
	/// The song's name, from its file: without the folder, the ending, or the id a fetched song's
	/// name ends with ("Flickermood [293].m4a" is Flickermood).
	pub fn name(&self) -> String {
		let p = self.path.trim_end_matches(['/', '\\']);
		let file = p.rsplit(['/', '\\']).next().unwrap_or(p);
		let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
		let stem = match stem.rsplit_once(" [") {
			Some((s, id)) if id.ends_with(']') && !s.is_empty() => s,
			_ => stem,
		};
		stem.to_string()
	}

	/// Where the song's beats (seconds into it) fall in the source, for looks on the beat: the video
	/// starting at `origin`, running to `end`, at `speed`.
	pub fn beats_in_source(&self, beats: &[f64], edit: &Edit, origin: f64, end: f64, speed: f64) -> Vec<f64> {
		beats.iter().filter_map(|&b| edit.source_time(origin, end, b - self.from, speed)).collect()
	}
}

/// Where the frame the engine sees sits in the source: x, y, w, h, 0-1 across it.
pub type View = [f32; 4];

pub const WHOLE: View = [0.0, 0.0, 1.0, 1.0];

impl Edit {
	pub fn is_empty(&self) -> bool {
		*self == Edit::default()
	}

	/// Whether a point (0-1 across the source) is somewhere things may be followed.
	pub fn open(&self, x: f32, y: f32) -> bool {
		let inside = |z: &Zone| x >= z.x && x <= z.x + z.w && y >= z.y && y <= z.y + z.h;
		let mut only = self.zones.iter().filter(|z| z.mode == ZoneMode::Only).peekable();
		if only.peek().is_some() && !only.any(inside) {
			return false;
		}
		!self.zones.iter().any(|z| z.mode == ZoneMode::Ignore && inside(z))
	}

	/// Which of a w by h picture of the view may be followed, a pixel at a time; empty when all of it.
	pub fn allowed(&self, view: View, w: usize, h: usize) -> Vec<bool> {
		if self.zones.is_empty() || w == 0 || h == 0 {
			return Vec::new();
		}
		let mut out = vec![false; w * h];
		for y in 0..h {
			let sy = view[1] + (y as f32 + 0.5) / h as f32 * view[3];
			for x in 0..w {
				let sx = view[0] + (x as f32 + 0.5) / w as f32 * view[2];
				out[y * w + x] = self.open(sx, sy);
			}
		}
		out
	}

	/// Whether a detection (0-1 across the view) is wanted: one of the kinds asked for, with its middle
	/// somewhere open.
	pub fn admits(&self, d: &Detection, view: View) -> bool {
		if !self.kinds.is_empty() && !self.kinds.iter().any(|k| k.eq_ignore_ascii_case(&d.cls)) {
			return false;
		}
		let (x, y) = to_source(view, d.x + d.w / 2.0, d.y + d.h / 2.0);
		self.open(x, y)
	}

	/// How bare the picture is at `t`: 0 drawn over as usual, 1 not at all, fading over `BARE_FADE`.
	pub fn bare(&self, t: f64) -> f32 {
		self.inside(t, RangeMode::Bare)
	}

	/// How far the look mixed in is left out at `t`: 0 not at all, 1 altogether, fading over
	/// `BARE_FADE`.
	pub fn unmixed(&self, t: f64) -> f32 {
		self.inside(t, RangeMode::Unmix)
	}

	/// How far into a stretch of `mode` `t` is: 1 inside, fading to 0 over `BARE_FADE` at its ends.
	fn inside(&self, t: f64, mode: RangeMode) -> f32 {
		let mut most = 0.0f64;
		for r in self.ranges.iter().filter(|r| r.mode == mode) {
			let v = if t >= r.from && t <= r.to {
				((t - r.from) / BARE_FADE).min((r.to - t) / BARE_FADE).min(1.0)
			} else {
				0.0
			};
			most = most.max(v);
		}
		most as f32
	}

	/// Seconds of output from `origin` (the in point) to source time `t`: what's not cut out, slowed or
	/// sped by `speed`; before `origin`, negative.
	pub fn output_time(&self, origin: f64, t: f64, speed: f64) -> f64 {
		let speed = speed.max(0.01);
		if t <= origin {
			return (t - origin) / speed;
		}
		self.stretches(origin, t).iter().map(|(a, b)| b - a).sum::<f64>() / speed
	}

	/// The source time `o` seconds of output after `origin`, if that's before `end`.
	pub fn source_time(&self, origin: f64, end: f64, o: f64, speed: f64) -> Option<f64> {
		let mut left = o * speed.max(0.01);
		if left < 0.0 {
			return None;
		}
		for (a, b) in self.stretches(origin, end) {
			if left <= b - a {
				return Some(a + left);
			}
			left -= b - a;
		}
		None
	}

	/// The stretches of `from`..`to` left once the cuts are taken out, in order.
	pub fn stretches(&self, from: f64, to: f64) -> Vec<(f64, f64)> {
		let mut cuts: Vec<(f64, f64)> =
			self.ranges.iter().filter(|r| r.mode == RangeMode::Cut && r.to > r.from).map(|r| (r.from, r.to)).collect();
		cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
		let mut out = Vec::new();
		let mut at = from;
		for (a, b) in cuts {
			if b <= at {
				continue;
			}
			if a >= to {
				break;
			}
			if a > at {
				out.push((at, a));
			}
			at = at.max(b);
		}
		if at < to {
			out.push((at, to));
		}
		// a sliver too short to show a frame of only makes a jump
		out.retain(|(a, b)| b - a >= 0.05);
		out
	}

	/// If `t` is inside a cut, where the cut ends.
	pub fn cut_at(&self, t: f64) -> Option<f64> {
		self.ranges
			.iter()
			.filter(|r| r.mode == RangeMode::Cut && t >= r.from && t < r.to)
			.map(|r| r.to)
			.fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))))
	}

	/// Add a stretch, merging it with any of the same kind it touches.
	pub fn add_range(&mut self, from: f64, to: f64, mode: RangeMode) {
		let (mut from, mut to) = (from.min(to), from.max(to));
		self.ranges.retain(|r| {
			let touches = r.mode == mode && r.from <= to && r.to >= from;
			if touches {
				from = from.min(r.from);
				to = to.max(r.to);
			}
			!touches
		});
		self.ranges.push(Range { from, to, mode });
		self.ranges.sort_by(|a, b| a.from.total_cmp(&b.from));
	}
}

/// 0-1 across the view to 0-1 across the source.
pub fn to_source(view: View, x: f32, y: f32) -> (f32, f32) {
	(view[0] + x * view[2], view[1] + y * view[3])
}

/// 0-1 across the source to 0-1 across the view.
pub fn to_view(view: View, x: f32, y: f32) -> (f32, f32) {
	((x - view[0]) / view[2], (y - view[1]) / view[3])
}

/// Something the chooser can hold on to.
#[derive(Clone, Debug)]
struct Held {
	id: i64,
	/// the detector's kind, or "" for a box around movement
	cls: String,
	rect: Rect,
	/// output seconds when it was last seen, while it's lost
	lost: Option<f64>,
	/// the pick it came from, if it came from one
	from: Option<Pick>,
}

/// How a pick has gone so far.
#[derive(Clone, Debug)]
struct Tried {
	pick: Pick,
	/// output seconds when it came due
	since: f64,
	/// what it turned out to be, once it has
	found: Option<i64>,
	/// found, or given up on
	settled: bool,
}

/// Works out which things an edit's picks (or "main") mean, frame by frame, and holds on to them: a
/// picked thing that's lost for a moment and turns up again nearby under a new number is still the
/// picked thing.
#[derive(Clone, Debug, Default)]
pub struct Chooser {
	held: Vec<Held>,
	tried: Vec<Tried>,
	/// numbers known ahead, from a pass over the clip before this one: these are picked from the
	/// moment they turn up
	pub ahead: Vec<i64>,
}

/// How long a lost thing is looked for, in output seconds.
const LOST_FOR: f64 = 1.5;

fn middle(r: Rect) -> (f32, f32) {
	(r.x + r.w / 2.0, r.y + r.h / 2.0)
}

fn overlap(a: Rect, b: Rect) -> f32 {
	let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
	let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
	if w <= 0.0 || h <= 0.0 { 0.0 } else { w * h / (a.w * a.h).min(b.w * b.h).max(1e-6) }
}

impl Chooser {
	/// After a jump in the picture: what was held is gone from view, but the picks still stand.
	pub fn jumped(&mut self) {
		self.held.clear();
		self.tried.clear();
	}

	/// What the picks have turned out to be so far, for a pass ahead to hand over.
	pub fn found(&self) -> Vec<i64> {
		self.tried.iter().filter_map(|t| t.found).collect()
	}

	pub fn active(edit: &Edit) -> bool {
		edit.keep != Keep::All
	}

	/// The numbers held now.
	pub fn ids(&self) -> Vec<i64> {
		self.held.iter().filter(|h| h.lost.is_none()).map(|h| h.id).collect()
	}

	/// Where the things held are, in track units (lost ones where they were last seen, for a moment).
	pub fn rects(&self) -> Vec<Rect> {
		self.held.iter().map(|h| h.rect).collect()
	}

	/// Look at this frame's things and work out which are chosen. `t` is source seconds, `clock`
	/// output seconds; `units` turns 0-1 across the view into track units (x, y).
	pub fn update(
		&mut self,
		edit: &Edit,
		t: f64,
		clock: f64,
		view: View,
		units: (f32, f32),
		named: &[ObjectTrack],
		regions: &[Region],
	) {
		if edit.keep == Keep::All {
			self.held.clear();
			return;
		}
		let things: Vec<(i64, &str, Rect)> = named
			.iter()
			.map(|o| (o.id, o.cls.as_str(), o.rect()))
			.chain(regions.iter().filter(|r| r.miss == 0).map(|r| (r.id, "", r.rect())))
			.collect();
		let alive = |id: i64| things.iter().any(|t| t.0 == id);
		// picks taken away since last time take what they found with them
		self.held.retain(|h| h.from.is_none_or(|p| edit.picks.contains(&p)));
		self.tried.retain(|t| edit.picks.contains(&t.pick));

		// keep each held thing up to date, or note it's lost
		for h in &mut self.held {
			match things.iter().find(|t| t.0 == h.id) {
				Some(t) => {
					h.rect = t.2;
					h.lost = None;
				}
				None => {
					if h.lost.is_none() {
						h.lost = Some(clock);
					}
				}
			}
		}
		// a lost thing turning up again nearby, under a new number, of the same kind
		let held_ids: Vec<i64> = self.held.iter().map(|h| h.id).collect();
		for h in &mut self.held {
			let Some(since) = h.lost else { continue };
			let (hx, hy) = middle(h.rect);
			let reach = h.rect.w.max(h.rect.h).max(0.05) * 1.2;
			let again = things
				.iter()
				.filter(|t| !held_ids.contains(&t.0) && (t.1 == h.cls || h.cls.is_empty() || t.1.is_empty()))
				.map(|t| {
					let (x, y) = middle(t.2);
					(t, (x - hx).hypot(y - hy))
				})
				.filter(|(_, d)| *d < reach)
				.min_by(|a, b| a.1.total_cmp(&b.1));
			if let Some((t, _)) = again {
				h.id = t.0;
				h.cls = t.1.to_string();
				h.rect = t.2;
				h.lost = None;
			} else if clock - since > LOST_FOR {
				h.id = i64::MIN;
			}
		}
		self.held.retain(|h| h.id != i64::MIN);
		// something held that the detector has since named: hold the name too
		let mut also = Vec::new();
		for h in self.held.iter().filter(|h| h.cls.is_empty() && h.lost.is_none()) {
			for o in named {
				if !self.held.iter().any(|x| x.id == o.id) && overlap(h.rect, o.rect()) > 0.5 {
					also.push(Held { id: o.id, cls: o.cls.clone(), rect: o.rect(), lost: None, from: h.from });
				}
			}
		}
		self.held.extend(also);

		match edit.keep {
			Keep::All => {}
			Keep::Picked => {
				for &id in &self.ahead {
					if alive(id) && !self.held.iter().any(|h| h.id == id) {
						let t = things.iter().find(|t| t.0 == id).unwrap();
						self.held.push(Held { id, cls: t.1.to_string(), rect: t.2, lost: None, from: None });
					}
				}
				for p in &edit.picks {
					if t + 1e-6 < p.t {
						continue;
					}
					let i = match self.tried.iter().position(|x| x.pick == *p) {
						Some(i) => i,
						None => {
							self.tried.push(Tried { pick: *p, since: clock, found: None, settled: false });
							self.tried.len() - 1
						}
					};
					if self.tried[i].settled {
						continue;
					}
					let (vx, vy) = to_view(view, p.x, p.y);
					let (px, py) = (vx * units.0, vy * units.1);
					// the smallest thing the click is inside, else the nearest close by; named things first
					let inside = |r: Rect| px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h;
					let best = things
						.iter()
						.filter(|t| inside(t.2))
						.min_by(|a, b| {
							a.1.is_empty().cmp(&b.1.is_empty()).then((a.2.w * a.2.h).total_cmp(&(b.2.w * b.2.h)))
						})
						.or_else(|| {
							things
								.iter()
								.map(|t| {
									let (x, y) = middle(t.2);
									(t, (x - px).hypot(y - py))
								})
								.filter(|(_, d)| *d < 0.12)
								.min_by(|a, b| a.1.total_cmp(&b.1))
								.map(|(t, _)| t)
						});
					if let Some(t) = best {
						self.tried[i].settled = true;
						self.tried[i].found = Some(t.0);
						if !self.held.iter().any(|h| h.id == t.0) {
							self.held.push(Held {
								id: t.0,
								cls: t.1.to_string(),
								rect: t.2,
								lost: None,
								from: Some(*p),
							});
						}
					} else if clock - self.tried[i].since > 1.0 {
						// nothing there for a second: give up on it
						self.tried[i].settled = true;
					}
				}
			}
			Keep::Centre => {
				// the thing nearest the middle, named things first; held while it lasts
				if self.held.iter().all(|h| h.lost.is_some()) {
					let (mx, my) = (units.0 / 2.0, units.1 / 2.0);
					let far = |r: Rect| {
						let (x, y) = middle(r);
						(x - mx).hypot(y - my)
					};
					let named_first = things.iter().any(|t| !t.1.is_empty());
					let best = things
						.iter()
						.filter(|t| !named_first || !t.1.is_empty())
						.min_by(|a, b| far(a.2).total_cmp(&far(b.2)));
					if let Some(b) = best {
						self.held = vec![Held { id: b.0, cls: b.1.to_string(), rect: b.2, lost: None, from: None }];
					}
				}
			}
			Keep::Main => {
				let size = |r: Rect| r.w * r.h;
				// the detector's things if it has any, else the boxes around movement
				let pool: Vec<&(i64, &str, Rect)> = if things.iter().any(|t| !t.1.is_empty()) {
					things.iter().filter(|t| !t.1.is_empty()).collect()
				} else {
					things.iter().collect()
				};
				let biggest = pool.iter().max_by(|a, b| size(a.2).total_cmp(&size(b.2)));
				let current = self.held.iter().find(|h| h.lost.is_none()).map(|h| (h.id, size(h.rect)));
				match (current, biggest) {
					(Some((id, s)), Some(b)) if b.0 != id && size(b.2) > s * 1.6 => {
						self.held = vec![Held { id: b.0, cls: b.1.to_string(), rect: b.2, lost: None, from: None }];
					}
					(None, Some(b)) if self.held.iter().all(|h| h.lost.is_some()) => {
						self.held = vec![Held { id: b.0, cls: b.1.to_string(), rect: b.2, lost: None, from: None }];
					}
					_ => {}
				}
				self.held.truncate(2);
			}
		}
	}

	/// Narrow what's drawn to what's chosen: its boxes, and the points on and around it.
	pub fn narrow(
		&self,
		edit: &Edit,
		tracks: &mut Vec<Track>,
		regions: &mut Vec<Region>,
		named: &mut Vec<ObjectTrack>,
	) {
		if edit.keep == Keep::All {
			return;
		}
		let ids = self.ids();
		named.retain(|o| ids.contains(&o.id));
		let boxes: Vec<Rect> = named
			.iter()
			.map(ObjectTrack::rect)
			.chain(self.held.iter().filter(|h| h.lost.is_none()).map(|h| h.rect))
			.collect();
		regions.retain(|r| ids.contains(&r.id) || boxes.iter().any(|&b| overlap(r.rect(), b) > 0.5));
		tracks.retain(|p| {
			boxes.iter().any(|b| {
				let pad = b.w.max(b.h) * 0.15 + 0.02;
				p.x >= b.x - pad && p.x <= b.x + b.w + pad && p.y >= b.y - pad && p.y <= b.y + b.h + pad
			})
		});
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn music_runs_on_the_outputs_clock() {
		let e = Edit { ranges: vec![Range { from: 4.0, to: 6.0, mode: RangeMode::Cut }], ..Edit::default() };
		// from an in point at 1 s: 3 s before the cut, none in it, then on
		assert_eq!(e.output_time(1.0, 3.0, 1.0), 2.0);
		assert_eq!(e.output_time(1.0, 7.0, 1.0), 4.0);
		assert_eq!(e.output_time(1.0, 0.5, 1.0), -0.5);
		// at half speed the output is twice as long
		assert_eq!(e.output_time(1.0, 3.0, 0.5), 4.0);
		assert_eq!(e.source_time(1.0, 10.0, 4.0, 1.0), Some(7.0));
		assert_eq!(e.source_time(1.0, 10.0, 2.5, 1.0), Some(3.5));
		assert_eq!(e.source_time(1.0, 10.0, 3.5, 1.0), Some(6.5));
		assert_eq!(e.source_time(1.0, 10.0, 20.0, 1.0), None);
		// a song started 10 s in: its beat at 12.5 s is 2.5 s of output, at 13.5 s past the cut; one
		// before it starts or after the video ends isn't anywhere
		let m = Music { path: "song.mp3".into(), from: 10.0, ..Music::default() };
		assert_eq!(m.beats_in_source(&[9.0, 12.5, 13.5, 30.0], &e, 1.0, 10.0, 1.0), vec![3.5, 6.5]);
		assert_eq!(m.name(), "song");
		assert_eq!(Music { path: "C:\\songs\\Flickermood [293].m4a".into(), ..Music::default() }.name(), "Flickermood");
	}

	#[test]
	fn cuts_leave_the_rest_in_order() {
		let mut e = Edit::default();
		e.add_range(2.0, 3.0, RangeMode::Cut);
		e.add_range(5.0, 6.0, RangeMode::Cut);
		e.add_range(5.5, 7.0, RangeMode::Cut);
		e.add_range(4.0, 4.5, RangeMode::Bare);
		assert_eq!(e.ranges.len(), 3);
		assert_eq!(e.stretches(0.0, 10.0), vec![(0.0, 2.0), (3.0, 5.0), (7.0, 10.0)]);
		assert_eq!(e.stretches(2.5, 6.5), vec![(3.0, 5.0)]);
		assert_eq!(e.cut_at(5.7), Some(7.0));
		assert_eq!(e.cut_at(4.2), None);
		assert!(e.bare(4.25) > 0.5 && e.bare(3.9) == 0.0);
	}

	#[test]
	fn zones_close_off_the_picture() {
		let e =
			Edit { zones: vec![Zone { x: 0.0, y: 0.0, w: 0.5, h: 1.0, mode: ZoneMode::Ignore }], ..Default::default() };
		assert!(!e.open(0.2, 0.5) && e.open(0.7, 0.5));
		// seen through a cut of the right half only, all of it is open
		let a = e.allowed([0.5, 0.0, 0.5, 1.0], 10, 4);
		assert!(a.iter().all(|&v| v));
		let only =
			Edit { zones: vec![Zone { x: 0.4, y: 0.4, w: 0.2, h: 0.2, mode: ZoneMode::Only }], ..Default::default() };
		assert!(only.open(0.5, 0.5) && !only.open(0.1, 0.1));
	}

	#[test]
	fn an_edit_reads_back() {
		let e: Edit = serde_json::from_str(r#"{"kinds":["car"],"keep":"picked","picks":[{"t":1.5,"x":0.3,"y":0.6}],"zones":[{"x":0,"y":0,"w":0.2,"h":0.2}],"ranges":[{"from":1,"to":2,"mode":"bare"}]}"#).unwrap();
		assert_eq!(e.keep, Keep::Picked);
		assert_eq!(e.zones[0].mode, ZoneMode::Ignore);
		assert_eq!(e.ranges[0].mode, RangeMode::Bare);
	}
}
