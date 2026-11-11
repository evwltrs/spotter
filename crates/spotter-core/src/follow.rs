//! Follow cam: a virtual camera that holds what's followed in frame, the way an operator would, so a
//! landscape clip can become a 9:16 one that keeps the subject in it. The camera is a window moving
//! over the source picture.
//!
//! An export knows the whole clip, so it follows first and frames second: a quick pass finds where
//! the subject is on every frame, gaps are filled, and the path is smoothed backwards and forwards in
//! time, so the camera moves before the subject gets away rather than chasing it, and never jitters.
//! A preview can't see ahead, so it uses a spring that follows a little behind.

use crate::Rect;
use crate::look::Look;

/// Where the camera's window goes on each frame, in source pixels: x, y, w, h.
pub type Window = [f32; 4];

/// The biggest window of shape `aspect` (width over height) that fits a w by h picture.
pub fn widest(w: f32, h: f32, aspect: f32) -> (f32, f32) {
	if w / h > aspect { (h * aspect, h) } else { (w, w / aspect) }
}

/// How much of the window's height the subject takes up at a zoom (0 doesn't zoom at all).
fn fill(zoom: f32) -> f32 {
	0.85 - 0.6 * zoom.clamp(0.0, 1.0)
}

/// The window around a subject: centred on it, zoomed in so it fills a share of the height (if
/// zooming), kept inside the picture.
pub fn window_for(cx: f32, cy: f32, size: f32, sw: f32, sh: f32, aspect: f32, zoom: f32) -> Window {
	let (_, mh) = widest(sw, sh, aspect);
	let wh = if zoom <= 0.0 { mh } else { (size / fill(zoom)).clamp(mh / 4.0, mh) };
	let ww = wh * aspect;
	let x = (cx - ww / 2.0).clamp(0.0, (sw - ww).max(0.0));
	let y = (cy - wh / 2.0).clamp(0.0, (sh - wh).max(0.0));
	[x, y, ww.min(sw), wh.min(sh)]
}

/// The size a subject needs room for: its height, or its width if that's what won't fit the shape.
fn need(r: Rect, aspect: f32) -> f32 {
	r.h.max(r.w / aspect)
}

/// Blur a sequence with a Gaussian of `sigma` samples, looking both ways (so it doesn't lag).
fn smooth(v: &[f32], sigma: f32) -> Vec<f32> {
	if sigma < 0.5 || v.len() < 3 {
		return v.to_vec();
	}
	let r = (sigma * 3.0).ceil() as isize;
	let weights: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
	let n = v.len() as isize;
	(0..n)
		.map(|i| {
			let (mut sum, mut total) = (0.0, 0.0);
			for (k, w) in weights.iter().enumerate() {
				// the ends are held, as if the clip went on as it starts and finishes
				let j = (i + k as isize - r).clamp(0, n - 1) as usize;
				sum += v[j] * w;
				total += w;
			}
			sum / total
		})
		.collect()
}

/// Fill the frames where nothing was found from the ones either side: in between, a straight line
/// from one to the next; before the first and after the last, held.
fn fill_gaps(seen: &[Option<Rect>]) -> Option<Vec<Rect>> {
	let known: Vec<usize> = (0..seen.len()).filter(|&i| seen[i].is_some()).collect();
	let (&first, &last) = (known.first()?, known.last()?);
	let mut out = Vec::with_capacity(seen.len());
	let mut k = 0;
	for i in 0..seen.len() {
		out.push(match seen[i] {
			Some(r) => r,
			None if i < first => seen[first].unwrap(),
			None if i > last => seen[last].unwrap(),
			None => {
				while known[k + 1] < i {
					k += 1;
				}
				let (a, b) = (known[k], known[k + 1]);
				let (ra, rb) = (seen[a].unwrap(), seen[b].unwrap());
				let t = (i - a) as f32 / (b - a) as f32;
				Rect {
					x: ra.x + (rb.x - ra.x) * t,
					y: ra.y + (rb.y - ra.y) * t,
					w: ra.w + (rb.w - ra.w) * t,
					h: ra.h + (rb.h - ra.h) * t,
				}
			}
		});
	}
	Some(out)
}

/// The camera's path for a whole clip: where the subject is on each frame (in source pixels, None
/// where it wasn't found) to a window for each frame. `cuts` are the frames a new shot starts on: the
/// camera cuts with the footage rather than gliding across.
pub fn path(seen: &[Option<Rect>], cuts: &[bool], sw: f32, sh: f32, aspect: f32, l: &Look, fps: f64) -> Vec<Window> {
	let mut out = Vec::with_capacity(seen.len());
	let mut a = 0;
	for i in 1..=seen.len() {
		if i == seen.len() || cuts.get(i).copied().unwrap_or(false) {
			out.extend(shot(&seen[a..i], sw, sh, aspect, l, fps));
			a = i;
		}
	}
	out
}

/// The camera's path through one shot.
fn shot(seen: &[Option<Rect>], sw: f32, sh: f32, aspect: f32, l: &Look, fps: f64) -> Vec<Window> {
	let Some(boxes) = fill_gaps(seen) else {
		// nothing found at all: the middle, still
		let (w, h) = widest(sw, sh, aspect);
		return vec![[(sw - w) / 2.0, (sh - h) / 2.0, w, h]; seen.len()];
	};
	let sigma = (l.follow_smooth.max(0.05) as f64 * fps) as f32;
	let cx = smooth(&boxes.iter().map(|r| r.x + r.w / 2.0).collect::<Vec<_>>(), sigma);
	// a subject's middle is a little high for a person or a plane's tail; aim a touch below it
	let cy = smooth(&boxes.iter().map(|r| r.y + r.h * 0.52).collect::<Vec<_>>(), sigma);
	// zoom changes slower than the pan, as an operator's does
	let size = smooth(&boxes.iter().map(|r| need(*r, aspect)).collect::<Vec<_>>(), sigma * 2.0);
	(0..boxes.len()).map(|i| window_for(cx[i], cy[i], size[i], sw, sh, aspect, l.follow_zoom)).collect()
}

/// A camera for a preview, which can't see ahead: it eases toward the subject like a spring, so it
/// follows a little behind but never jumps.
#[derive(Clone, Debug, Default)]
pub struct Spring {
	at: Option<[f32; 3]>,
	vel: [f32; 3],
}

impl Spring {
	pub fn reset(&mut self) {
		self.at = None;
		self.vel = [0.0; 3];
	}

	/// Move toward the subject (None holds still) over `dt` seconds and say where the window is.
	pub fn step(&mut self, target: Option<Rect>, sw: f32, sh: f32, aspect: f32, l: &Look, dt: f32) -> Window {
		let goal = target.map(|r| [r.x + r.w / 2.0, r.y + r.h * 0.52, need(r, aspect)]);
		let at = match (self.at, goal) {
			(None, Some(g)) => g,
			(None, None) => [sw / 2.0, sh / 2.0, sh],
			(Some(a), None) => a,
			(Some(a), Some(g)) => {
				// critically damped, settling in about the smoothing time
				let omega = 4.0 / l.follow_smooth.max(0.05);
				let mut next = a;
				for c in 0..3 {
					let k = if c == 2 { omega / 2.0 } else { omega };
					let accel = k * k * (g[c] - a[c]) - 2.0 * k * self.vel[c];
					self.vel[c] += accel * dt;
					next[c] = a[c] + self.vel[c] * dt;
				}
				next
			}
		};
		self.at = Some(at);
		window_for(at[0], at[1], at[2], sw, sh, aspect, l.follow_zoom)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn the_camera_stays_on_the_subject_and_in_the_picture() {
		let l = Look { follow: true, follow_smooth: 0.5, ..Default::default() };
		// a box crossing a 1920x1080 picture left to right, lost for a while in the middle
		let seen: Vec<Option<Rect>> = (0..120)
			.map(|i| {
				if (50..70).contains(&i) {
					None
				} else {
					Some(Rect { x: 100.0 + i as f32 * 14.0, y: 500.0, w: 120.0, h: 80.0 })
				}
			})
			.collect();
		let p = path(&seen, &[], 1920.0, 1080.0, 9.0 / 16.0, &l, 30.0);
		assert_eq!(p.len(), 120);
		for (i, w) in p.iter().enumerate() {
			assert!(w[0] >= 0.0 && w[0] + w[2] <= 1920.01 && w[1] >= 0.0 && w[1] + w[3] <= 1080.01, "frame {i}: {w:?}");
		}
		// in the middle of the run, the subject's middle is in the window's middle third
		let mid = 100.0 + 90.0 * 14.0 + 60.0;
		let w = p[90];
		assert!(mid > w[0] + w[2] / 3.0 && mid < w[0] + w[2] * 2.0 / 3.0, "{mid} in {w:?}");
	}

	#[test]
	fn the_camera_cuts_with_the_footage() {
		let l = Look { follow_smooth: 1.0, ..Default::default() };
		// something on the left for a second, then a cut to something on the right
		let seen: Vec<Option<Rect>> = (0..60)
			.map(|i| Some(Rect { x: if i < 30 { 100.0 } else { 1700.0 }, y: 500.0, w: 100.0, h: 100.0 }))
			.collect();
		let cuts: Vec<bool> = (0..60).map(|i| i == 30).collect();
		let p = path(&seen, &cuts, 1920.0, 1080.0, 9.0 / 16.0, &l, 30.0);
		// the last frame before the cut is still on the left, the first after it already on the right
		assert!(p[29][0] < 200.0 && p[30][0] + p[30][2] > 1800.0, "{:?} {:?}", p[29], p[30]);
	}
}
