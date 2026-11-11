//! The scene: what each part of the picture is (sky, sea, tree, grass, road, building, mountain… 150
//! kinds), from SegFormer trained on ADE20K, on the GPU. The detector names things; this
//! names the stuff around them, so the words look can type the world out as what it is.
//!
//! Exports use the biggest SegFormer (B5, 640 pixels, about 90 ms a frame); the window a smaller one
//! (B2, 512, about 35 ms). What it says wavers from frame to frame where two kinds look alike (sea or
//! water, tree or plant), so its answers are eased over time before the surest is taken.

use crate::Frame;
use crate::detector::Device;
use crate::look::Look;
use crate::model::{Data, Model, Tensor, get_for};
use anyhow::{Result, bail};
use rayon::prelude::*;

/// ADE20K's 150 kinds, in the model's order, named as they'd be typed.
const KINDS: [&str; 150] = [
	"wall",
	"building",
	"sky",
	"floor",
	"tree",
	"ceiling",
	"road",
	"bed",
	"window",
	"grass",
	"cabinet",
	"sidewalk",
	"person",
	"ground",
	"door",
	"table",
	"mountain",
	"plant",
	"curtain",
	"chair",
	"car",
	"water",
	"painting",
	"sofa",
	"shelf",
	"house",
	"sea",
	"mirror",
	"rug",
	"field",
	"armchair",
	"seat",
	"fence",
	"desk",
	"rock",
	"wardrobe",
	"lamp",
	"bathtub",
	"railing",
	"cushion",
	"base",
	"box",
	"column",
	"sign",
	"drawers",
	"counter",
	"sand",
	"sink",
	"skyscraper",
	"fireplace",
	"fridge",
	"grandstand",
	"path",
	"stairs",
	"runway",
	"case",
	"pool table",
	"pillow",
	"screen door",
	"stairway",
	"river",
	"bridge",
	"bookcase",
	"blind",
	"coffee table",
	"toilet",
	"flower",
	"book",
	"hill",
	"bench",
	"countertop",
	"stove",
	"palm",
	"kitchen island",
	"computer",
	"swivel chair",
	"boat",
	"bar",
	"arcade machine",
	"hut",
	"bus",
	"towel",
	"light",
	"truck",
	"tower",
	"chandelier",
	"awning",
	"streetlight",
	"booth",
	"television",
	"airplane",
	"dirt track",
	"clothes",
	"pole",
	"land",
	"bannister",
	"escalator",
	"ottoman",
	"bottle",
	"buffet",
	"poster",
	"stage",
	"van",
	"ship",
	"fountain",
	"conveyor belt",
	"canopy",
	"washer",
	"toy",
	"swimming pool",
	"stool",
	"barrel",
	"basket",
	"waterfall",
	"tent",
	"bag",
	"minibike",
	"cradle",
	"oven",
	"ball",
	"food",
	"step",
	"tank",
	"sign",
	"microwave",
	"pot",
	"animal",
	"bicycle",
	"lake",
	"dishwasher",
	"screen",
	"blanket",
	"sculpture",
	"hood",
	"sconce",
	"vase",
	"traffic light",
	"tray",
	"bin",
	"fan",
	"pier",
	"screen",
	"plate",
	"monitor",
	"noticeboard",
	"shower",
	"radiator",
	"glass",
	"clock",
	"flag",
];

/// Whether a look labels the scene.
pub fn active(l: &Look) -> bool {
	l.glyphs == "words" && l.scene
}

/// The models, best first: exports take the first, the window the second.
pub const MODELS: [(&str, usize); 2] = [("segformer-b5-ade", 640), ("segformer-b2-ade", 512)];

pub struct Scener {
	/// none in a shell: the window's copy that answers with what the models' thread found
	model: Option<Model>,
	size: usize,
	small: Frame,
	input: Vec<f32>,
	/// the model's answers, eased over time: KINDS by `grid` by `grid`
	eased: Vec<f32>,
	grid: usize,
	/// the surest kind in each cell of the grid, and how sure (the eased answers' margin)
	pub kind: Vec<u8>,
	pub ms: f32,
}

impl Scener {
	/// `which` is one of `MODELS`.
	pub fn load(which: (&str, usize), device: Device, half: bool) -> Result<Self> {
		let (name, size) = which;
		Ok(Self {
			model: Some(Model::load(&get_for(name, device, half)?, device)?),
			size,
			small: Frame::new(size as u32, size as u32),
			input: vec![0.0; 3 * size * size],
			eased: Vec::new(),
			grid: 0,
			kind: Vec::new(),
			ms: 0.0,
		})
	}

	/// A copy without the model, to answer with what another (on the models' thread) finds.
	pub fn shell(&self) -> Scener {
		Scener {
			model: None,
			size: self.size,
			small: Frame::default(),
			input: Vec::new(),
			eased: Vec::new(),
			grid: self.grid,
			kind: self.kind.clone(),
			ms: 0.0,
		}
	}

	/// Whether it can work the scene out itself (it's not a shell).
	pub fn has_model(&self) -> bool {
		self.model.is_some()
	}

	pub fn grid(&self) -> usize {
		self.grid
	}

	/// What `run` found for a frame before, from recall.rs.
	pub fn recall(&mut self, grid: usize, kind: &[u8]) {
		self.grid = grid;
		self.kind.clear();
		self.kind.extend_from_slice(kind);
	}

	/// After a jump or a cut: what the last picture was says nothing about this one.
	pub fn reset(&mut self) {
		self.eased.clear();
	}

	/// Work out what each part of `frame` is.
	pub fn run(&mut self, frame: &Frame) -> Result<()> {
		let t0 = std::time::Instant::now();
		let size = self.size;
		frame.resize_fast_into(&mut self.small);
		let plane = size * size;
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
		let picture = Tensor::f32(&[1, 3, size as i64, size as i64], std::mem::take(&mut self.input));
		let Some(model) = self.model.as_mut() else { bail!("a shell has no scene model") };
		let out = model.run(&[("pixel_values", &picture)]);
		self.input = match picture.data {
			Data::F32(v) => v,
			_ => unreachable!(),
		};
		let out = out?;
		let Some((_, t)) = out.first() else { bail!("the scene model gave nothing") };
		let Some(logits) = t.as_f32() else { bail!("the scene model gave the wrong kind of numbers") };
		let s = &t.shape;
		if s.len() != 4 || s[1] as usize != KINDS.len() || s[2] != s[3] {
			bail!("the scene model's answer came out {s:?}");
		}
		let grid = s[2] as usize;
		let cells = grid * grid;
		// eased (half the new answer, half what it's said lately) and the surest kind taken, in one pass
		// over a run of cells at a time, reading each kind's answers in the order they lie in memory
		let fresh = self.eased.len() != logits.len() || self.grid != grid;
		if fresh {
			self.eased = logits.to_vec();
			self.grid = grid;
		}
		self.kind.resize(cells, 0);
		const RUN: usize = 1024;
		let ptr = self.eased.as_mut_ptr() as usize;
		self.kind.par_chunks_mut(RUN).enumerate().for_each(|(n, kinds)| {
			let a = n * RUN;
			let mut best = vec![f32::MIN; kinds.len()];
			for k in 0..KINDS.len() {
				// SAFETY: each run of cells is only ever touched by this one task, in every kind's plane
				let plane =
					unsafe { std::slice::from_raw_parts_mut((ptr as *mut f32).add(k * cells + a), kinds.len()) };
				let new = &logits[k * cells + a..][..kinds.len()];
				for i in 0..kinds.len() {
					let v = if fresh { new[i] } else { plane[i] * 0.5 + new[i] * 0.5 };
					plane[i] = v;
					if v > best[i] {
						best[i] = v;
						kinds[i] = k as u8;
					}
				}
			}
		});
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		Ok(())
	}

	/// What's at a point, 0-1 across and down the frame.
	pub fn at(&self, x: f32, y: f32) -> Option<&'static str> {
		if self.kind.is_empty() {
			return None;
		}
		let g = self.grid;
		let (cx, cy) = (((x * g as f32) as usize).min(g - 1), ((y * g as f32) as usize).min(g - 1));
		KINDS.get(self.kind[cy * g + cx] as usize).copied()
	}
}
