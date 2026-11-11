//! Outlines: the true shape of each thing followed, not its box. EdgeTAM (Apache-2.0, a small SAM 2)
//! looks at the frame once, on the GPU, and then, given each thing's box, says which
//! pixels are that thing, on the CPU. spotter's own tracks say what's what and where; EdgeTAM
//! only gives the shape, so it works for anything with a box, named by the detector or not.
//!
//! The picture is squeezed to 1024 by 1024, as the model was trained, and each mask comes back at
//! 256 by 256 over the whole frame, as logits (above 0 is the thing).
//!
//! EdgeTAM looks at each frame afresh, and for each box offers a few shapes (a part of the thing,
//! more of it, all of it), which it's about equally sure of from one frame to the next: taken as they
//! come, a skier's skis are there on one frame and gone on the next. `Steady` keeps each thing's shape
//! from frame to frame, in the thing's own box so it moves with it: of the shapes offered it takes the
//! one most like the thing's shape a moment ago, eases toward it, and holds the last one through a
//! frame EdgeTAM doubts.

use crate::Frame;
use crate::Rect;
use crate::detector::Device;
use crate::matte::bilinear;
use crate::model::{Data, Model, Tensor, get_for};
use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;
use std::time::Instant;

const SIZE: usize = 1024;
pub const MASK: usize = 256;

/// The models (encoder, decoder), best first: exports take the first, the window the second. SAM 2.1
/// large sees truer outlines; EdgeTAM is a small SAM 2, quick enough to keep up with a preview. Both
/// take and give the same things.
pub const MODELS: [(&str, &str); 2] =
	[("sam2.1-large-encoder", "sam2.1-large-decoder"), ("edgetam-encoder", "edgetam-decoder")];

pub struct Outliner {
	encoder: Model,
	decoder: Model,
	small: Frame,
	input: Vec<f32>,
	/// how long the last frame took, in milliseconds
	pub ms: f32,
}

/// The shape of one thing: logits, `MASK` by `MASK`, over the whole frame.
#[derive(Clone, Debug)]
pub struct Shape {
	pub logits: Vec<f32>,
	/// how sure the model is of it
	pub score: f32,
}

/// What EdgeTAM offers for one box: a few shapes (logits, `MASK` by `MASK`, over the whole frame),
/// how sure it is of each, and whether it thinks there's a thing there at all (above 0 it does).
#[derive(Clone, Debug)]
pub struct Offer {
	pub masks: Vec<Vec<f32>>,
	pub scores: Vec<f32>,
	pub present: f32,
}

impl Offer {
	/// Whether it believes in any of its shapes.
	fn believed(&self) -> bool {
		self.present >= 0.0 && self.scores.iter().any(|&s| s >= 0.5)
	}
}

/// A thing's shape is kept over its box and a margin round it, this many cells each way.
const CELLS: usize = 160;
/// how far toward the new shape each frame goes (the rest is the shape a moment ago)
const EASE: f32 = 0.6;
/// frames a thing's shape is kept for when it isn't seen (the window's models answer every few frames)
const KEEP: u64 = 15;

/// Where a thing's shape is kept: its box with the margin the matte allows (in the frame's pixels).
fn around(b: Rect) -> Rect {
	let pad = b.w.max(b.h) * 0.15;
	Rect { x: b.x - pad, y: b.y - pad, w: b.w + 2.0 * pad, h: b.h + 2.0 * pad }
}

/// A shape over the whole frame (logits) as the chance each cell of `r` is the thing.
fn kept_from(logits: &[f32], r: Rect, fw: u32, fh: u32) -> Vec<f32> {
	let (mx, my) = (MASK as f32 / fw as f32, MASK as f32 / fh as f32);
	let mut p = vec![0.0; CELLS * CELLS];
	for v in 0..CELLS {
		let y = r.y + (v as f32 + 0.5) / CELLS as f32 * r.h;
		for u in 0..CELLS {
			let x = r.x + (u as f32 + 0.5) / CELLS as f32 * r.w;
			let l = bilinear(logits, MASK, MASK, x * mx - 0.5, y * my - 0.5);
			p[v * CELLS + u] = 1.0 / (1.0 + (-l).exp());
		}
	}
	p
}

/// A kept shape back over the whole frame, as logits, nothing outside `r`.
fn logits_from(p: &[f32], r: Rect, fw: u32, fh: u32) -> Vec<f32> {
	let mut out = vec![-10.0; MASK * MASK];
	let (sx, sy) = (fw as f32 / MASK as f32, fh as f32 / MASK as f32);
	let x0 = ((r.x / sx).floor().max(0.0) as usize).min(MASK);
	let x1 = ((((r.x + r.w) / sx).ceil()).max(0.0) as usize).min(MASK);
	let y0 = ((r.y / sy).floor().max(0.0) as usize).min(MASK);
	let y1 = ((((r.y + r.h) / sy).ceil()).max(0.0) as usize).min(MASK);
	for my in y0..y1 {
		let v = ((my as f32 + 0.5) * sy - r.y) / r.h * CELLS as f32 - 0.5;
		for mx in x0..x1 {
			let u = ((mx as f32 + 0.5) * sx - r.x) / r.w * CELLS as f32 - 0.5;
			if u < -0.5 || v < -0.5 || u > CELLS as f32 - 0.5 || v > CELLS as f32 - 0.5 {
				continue;
			}
			let q = bilinear(p, CELLS, CELLS, u, v).clamp(1e-4, 1.0 - 1e-4);
			out[my * MASK + mx] = (q / (1.0 - q)).ln();
		}
	}
	out
}

/// How alike two kept shapes are: their overlap over their union, 0-1.
fn alike(a: &[f32], b: &[f32]) -> f32 {
	let (mut both, mut either) = (0.0, 0.0);
	for (x, y) in a.iter().zip(b) {
		both += x.min(*y);
		either += x.max(*y);
	}
	if either > 0.0 { both / either } else { 1.0 }
}

/// Each thing's shape, kept from frame to frame.
#[derive(Default)]
pub struct Steady {
	/// (the thing's number, its shape over its box, the frame it was last seen on)
	kept: Vec<(i64, Vec<f32>, u64)>,
	/// each box's shape the last time round, how sure, and whether it was kept: what recall.rs keeps
	pub last: Vec<crate::recall::Kept>,
}

impl Steady {
	pub fn clear(&mut self) {
		self.kept.clear();
	}

	/// The shape of each thing in `boxes` (the frame's pixels) this frame, numbered `ids`, from what
	/// EdgeTAM `offers` for each. `frame` counts frames; with `commit` false (the same frame drawn
	/// again) what's kept stays as it was.
	pub fn shapes(
		&mut self,
		offers: &[Offer],
		boxes: &[Rect],
		ids: &[i64],
		fw: u32,
		fh: u32,
		frame: u64,
		commit: bool,
	) -> Vec<Shape> {
		let mut out = Vec::with_capacity(boxes.len());
		let mut now: Vec<(i64, Vec<f32>, u64)> = Vec::new();
		self.last.clear();
		for (i, (b, offer)) in boxes.iter().zip(offers).enumerate() {
			let id = ids.get(i).copied().unwrap_or(crate::objects::UNNUMBERED);
			let r = around(*b);
			let before = (id != crate::objects::UNNUMBERED)
				.then(|| self.kept.iter().find(|k| k.0 == id && frame.saturating_sub(k.2) <= KEEP))
				.flatten()
				.map(|k| &k.1);
			let best_score = offer.scores.iter().copied().fold(0.0, f32::max);
			let p = if offer.believed() {
				let offered: Vec<Vec<f32>> = offer.masks.par_iter().map(|m| kept_from(m, r, fw, fh)).collect();
				// the shape most like it a moment ago, among those it's fairly sure of; else its surest
				let pick = (0..offered.len())
					.filter(|&c| offer.scores[c] >= best_score - 0.15)
					.max_by(|&a, &c| {
						let worth = |k: usize| offer.scores[k] + before.map_or(0.0, |p| 2.0 * alike(&offered[k], p));
						worth(a).total_cmp(&worth(c))
					})
					.unwrap_or(0);
				match before {
					Some(p) => offered[pick].iter().zip(p).map(|(n, o)| n * EASE + o * (1.0 - EASE)).collect(),
					None => offered[pick].clone(),
				}
			} else {
				// doubted: the shape it had, if it had one; otherwise nothing
				match before {
					Some(p) => p.clone(),
					None => vec![0.0; CELLS * CELLS],
				}
			};
			out.push(Shape { logits: logits_from(&p, r, fw, fh), score: best_score });
			let keep = id != crate::objects::UNNUMBERED && (offer.believed() || before.is_some());
			self.last.push((p.clone(), best_score, keep));
			if keep {
				now.push((
					id,
					p,
					if offer.believed() { frame } else { self.kept.iter().find(|k| k.0 == id).map_or(frame, |k| k.2) },
				));
			}
		}
		self.commit(now, frame, commit);
		out
	}

	/// The shapes again from what recall.rs kept of them for this frame and these boxes, as `shapes`
	/// gave them then, without the model.
	pub fn recalled(
		&mut self,
		known: Vec<crate::recall::Kept>,
		boxes: &[Rect],
		ids: &[i64],
		fw: u32,
		fh: u32,
		frame: u64,
		commit: bool,
	) -> Vec<Shape> {
		let mut out = Vec::with_capacity(boxes.len());
		let mut now: Vec<(i64, Vec<f32>, u64)> = Vec::new();
		for (i, (b, (p, score, keep))) in boxes.iter().zip(known).enumerate() {
			let id = ids.get(i).copied().unwrap_or(crate::objects::UNNUMBERED);
			out.push(Shape { logits: logits_from(&p, around(*b), fw, fh), score });
			if keep && id != crate::objects::UNNUMBERED {
				now.push((id, p, frame));
			}
		}
		self.commit(now, frame, commit);
		out
	}

	/// Each of `boxes` (numbered `ids`) with the shape kept for it, where it is now; one not kept yet
	/// gets an empty shape (the matte cuts it out by colour until it has one). For the window, whose
	/// models answer on their own time: a shape kept relative to its box follows the thing.
	pub fn current(&self, boxes: &[Rect], ids: &[i64], fw: u32, fh: u32) -> Vec<Shape> {
		boxes
			.iter()
			.enumerate()
			.map(|(i, b)| {
				let id = ids.get(i).copied().unwrap_or(crate::objects::UNNUMBERED);
				match self.kept.iter().find(|k| k.0 == id && id != crate::objects::UNNUMBERED) {
					Some(k) => Shape { logits: logits_from(&k.1, around(*b), fw, fh), score: 1.0 },
					None => Shape { logits: Vec::new(), score: 0.0 },
				}
			})
			.collect()
	}

	/// What's kept from now on: this frame's shapes, and those not seen in it for a while yet.
	fn commit(&mut self, mut now: Vec<(i64, Vec<f32>, u64)>, frame: u64, commit: bool) {
		if commit {
			// what wasn't seen this frame is kept a while, in case it's back
			for k in self.kept.drain(..) {
				if !now.iter().any(|n| n.0 == k.0) && frame.saturating_sub(k.2) <= KEEP {
					now.push(k);
				}
			}
			self.kept = now;
		}
	}
}

impl Outliner {
	/// Both halves of `which` (one of `MODELS`) on `device`, in half precision with `half`.
	pub fn load(which: (&str, &str), device: Device, half: bool) -> Result<Self> {
		let encoder = Model::load(&get_for(which.0, device, half)?, device)?;
		let decoder = Model::load(&get_for(which.1, device, half)?, device)?;
		Ok(Self {
			encoder,
			decoder,
			small: Frame::new(SIZE as u32, SIZE as u32),
			input: vec![0.0; 3 * SIZE * SIZE],
			ms: 0.0,
		})
	}

	pub fn device(&self) -> Device {
		self.encoder.device
	}

	/// The shapes offered for what's in each box (in the frame's pixels), in the same order.
	pub fn outline(&mut self, frame: &Frame, boxes: &[Rect]) -> Result<Vec<Offer>> {
		if boxes.is_empty() || frame.w == 0 {
			return Ok(Vec::new());
		}
		let t0 = Instant::now();
		let mut lap = crate::perf::Lap::start();
		frame.resize_fast_into(&mut self.small);
		lap.mark("outline: squeeze");
		let plane = SIZE * SIZE;
		let (mean, sd) = ([0.485f32, 0.456, 0.406], [0.229f32, 0.224, 0.225]);
		let (r, rest) = self.input.split_at_mut(plane);
		let (g, b) = rest.split_at_mut(plane);
		r.par_iter_mut().zip(g.par_iter_mut()).zip(b.par_iter_mut()).zip(self.small.rgba.par_chunks_exact(4)).for_each(
			|(((r, g), b), p)| {
				*r = (p[0] as f32 / 255.0 - mean[0]) / sd[0];
				*g = (p[1] as f32 / 255.0 - mean[1]) / sd[1];
				*b = (p[2] as f32 / 255.0 - mean[2]) / sd[2];
			},
		);
		lap.mark("outline: normalise");
		let picture = Tensor::f32(&[1, 3, SIZE as i64, SIZE as i64], std::mem::take(&mut self.input));
		let seen = self.encoder.run(&[("pixel_values", &picture)]);
		self.input = match picture.data {
			Data::F32(v) => v,
			_ => unreachable!(),
		};
		let seen = seen?;
		lap.mark("outline: encoder");
		let t_enc = t0.elapsed().as_secs_f32() * 1000.0;
		let embedding = |name: &str| {
			seen.iter().find(|(n, _)| n == name).map(|(_, t)| t).ok_or_else(|| anyhow!("the encoder gave no {name}"))
		};
		let (sx, sy) = (SIZE as f32 / frame.w as f32, SIZE as f32 / frame.h as f32);
		// the GPU works out a plan for each new number of boxes, which takes a while, so
		// there are always 4, 8, 16 or 32 of them, the spare ones copies of the first
		let room = [4usize, 8, 16, 32].into_iter().find(|&n| n >= boxes.len()).unwrap_or(boxes.len());
		let corners: Vec<f32> = (0..room)
			.flat_map(|i| {
				let b = boxes.get(i).unwrap_or(&boxes[0]);
				[b.x * sx, b.y * sy, (b.x + b.w) * sx, (b.y + b.h) * sy]
			})
			.collect();
		let boxes_in = Tensor::f32(&[1, room as i64, 4], corners);
		// no points, only boxes
		let points = Tensor::f32(&[1, 1, 0, 2], Vec::new());
		let labels = Tensor::i64(&[1, 1, 0], Vec::new());
		let out = self.decoder.run(&[
			("input_points", &points),
			("input_labels", &labels),
			("input_boxes", &boxes_in),
			("image_embeddings.0", embedding("image_embeddings.0")?),
			("image_embeddings.1", embedding("image_embeddings.1")?),
			("image_embeddings.2", embedding("image_embeddings.2")?),
		])?;
		lap.mark("outline: decoder");
		let get = |name: &str| {
			out.iter().find(|(n, _)| n == name).map(|(_, t)| t).ok_or_else(|| anyhow!("the decoder gave no {name}"))
		};
		let (scores, masks) = (get("iou_scores")?, get("pred_masks")?);
		let (Some(scores), Some(m)) = (scores.as_f32(), masks.as_f32()) else {
			bail!("the decoder gave the wrong kind of numbers")
		};
		let s = &masks.shape;
		if s.len() != 5 || s[3] as usize != MASK || s[4] as usize != MASK {
			bail!("the decoder's masks came out {s:?}");
		}
		let (n, k) = ((s[1] as usize).min(boxes.len()), s[2] as usize);
		// whether it thinks there's a thing in each box at all
		let present = get("object_score_logits").ok().and_then(|t| t.as_f32().map(<[f32]>::to_vec));
		let debug = std::env::var_os("SPOTTER_TIMES").is_some();
		let mut shapes = Vec::with_capacity(n);
		for i in 0..n {
			let there = present.as_ref().and_then(|p| p.get(i)).copied().unwrap_or(1.0);
			let offer = Offer {
				masks: (0..k).map(|c| m[(i * k + c) * MASK * MASK..][..MASK * MASK].to_vec()).collect(),
				scores: scores[i * k..(i + 1) * k].to_vec(),
				present: there,
			};
			if debug && std::env::var_os("SPOTTER_TIMES_EACH").is_some() {
				eprintln!("  box {i}: scores {:?}, present {there:.1}", offer.scores);
			}
			shapes.push(offer);
		}
		lap.mark("outline: offers");
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		if std::env::var_os("SPOTTER_TIMES").is_some() && std::env::var_os("SPOTTER_TIMES_EACH").is_some() {
			eprintln!(
				"outline: {t_enc:.1} ms to encode (encoder {:.1}), {:.1} ms in all, decoder {:.1}",
				self.encoder.ms, self.ms, self.decoder.ms
			);
		}
		Ok(shapes)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Logits over the whole frame: the thing inside `r` (the frame is 256 by 256).
	fn mask(r: Rect) -> Vec<f32> {
		(0..MASK * MASK)
			.map(|i| (i % MASK, i / MASK))
			.map(|(x, y)| {
				if (x as f32) >= r.x && (x as f32) < r.x + r.w && (y as f32) >= r.y && (y as f32) < r.y + r.h {
					8.0
				} else {
					-8.0
				}
			})
			.collect()
	}

	#[test]
	fn a_shape_stays_the_same_choice_and_holds_through_doubt() {
		let b = Rect { x: 64.0, y: 64.0, w: 128.0, h: 128.0 };
		let (all, half) = (mask(b), mask(Rect { w: 64.0, ..b }));
		let mut s = Steady::default();
		let area = |sh: &Shape| sh.logits.iter().filter(|&&l| l > 0.0).count();
		// all of it first; then half of it scored a little higher, which a frame alone would take
		let first = s.shapes(
			&[Offer { masks: vec![all.clone(), half.clone()], scores: vec![0.9, 0.8], present: 5.0 }],
			&[b],
			&[7],
			256,
			256,
			1,
			true,
		);
		let second = s.shapes(
			&[Offer { masks: vec![all.clone(), half.clone()], scores: vec![0.85, 0.9], present: 5.0 }],
			&[b],
			&[7],
			256,
			256,
			2,
			true,
		);
		assert!(
			area(&second[0]) as f32 > area(&first[0]) as f32 * 0.9,
			"{} then {}",
			area(&first[0]),
			area(&second[0])
		);
		// a frame it doubts keeps the shape rather than losing the thing
		let third = s.shapes(
			&[Offer { masks: vec![half.clone()], scores: vec![0.2], present: -3.0 }],
			&[b],
			&[7],
			256,
			256,
			3,
			true,
		);
		assert!(area(&third[0]) as f32 > area(&first[0]) as f32 * 0.9);
	}
}
