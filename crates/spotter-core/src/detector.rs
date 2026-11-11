//! Recognising what things are: D-FINE (Apache-2.0), run through ONNX Runtime on the CPU or the
//! graphics card. It comes trained on COCO's 80 kinds of thing (the site's), or on Objects365's 365,
//! which tell a helicopter from an aeroplane and a ship from a boat, and know cranes, fire engines and
//! surveillance cameras. The model is downloaded once, into the user's cache.

use crate::Frame;
use crate::objects::Detection;
use anyhow::{Result, anyhow, bail};
use ort::session::Session;
use ort::value::TensorRef;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const LABELS: [&str; 80] = [
	"person",
	"bicycle",
	"car",
	"motorbike",
	"aeroplane",
	"bus",
	"train",
	"truck",
	"boat",
	"traffic light",
	"fire hydrant",
	"stop sign",
	"parking meter",
	"bench",
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe",
	"backpack",
	"umbrella",
	"handbag",
	"tie",
	"suitcase",
	"frisbee",
	"skis",
	"snowboard",
	"sports ball",
	"kite",
	"baseball bat",
	"baseball glove",
	"skateboard",
	"surfboard",
	"tennis racket",
	"bottle",
	"wine glass",
	"cup",
	"fork",
	"knife",
	"spoon",
	"bowl",
	"banana",
	"apple",
	"sandwich",
	"orange",
	"broccoli",
	"carrot",
	"hot dog",
	"pizza",
	"donut",
	"cake",
	"chair",
	"sofa",
	"pottedplant",
	"bed",
	"diningtable",
	"toilet",
	"tvmonitor",
	"laptop",
	"mouse",
	"remote",
	"keyboard",
	"cell phone",
	"microwave",
	"oven",
	"toaster",
	"sink",
	"refrigerator",
	"book",
	"clock",
	"vase",
	"scissors",
	"teddy bear",
	"hair drier",
	"toothbrush",
];

/// Objects365's kinds, lowercased, in the model's order (the first is "none").
const OBJ365: &str = include_str!("../../../assets/labels-obj365.txt");

const PEOPLE: &[&str] = &["person"];
const VEHICLES: &[&str] = &[
	"bicycle",
	"car",
	"motorbike",
	"aeroplane",
	"bus",
	"train",
	"truck",
	"boat", // COCO's, as the site names them
	"suv",
	"van",
	"motorcycle",
	"sailboat",
	"pickup truck",
	"machinery vehicle",
	"airplane",
	"sports car",
	"scooter",
	"tricycle",
	"fire truck",
	"heavy truck",
	"ship",
	"carriage",
	"ambulance",
	"helicopter",
	"rickshaw",
	"formula 1",
	"hot-air balloon",
];
const ANIMALS: &[&str] = &[
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe", //
	"wild bird",
	"other fish",
	"duck",
	"pigeon",
	"chicken",
	"deer",
	"goose",
	"penguin",
	"swan",
	"goldfish",
	"pig",
	"camel",
	"crab",
	"antelope",
	"parrot",
	"seal",
	"butterfly",
	"donkey",
	"lion",
	"dolphin",
	"jellyfish",
	"monkey",
	"rabbit",
	"yak",
];

/// Whether a kind of thing is wanted under the look's "look for" choice.
pub fn wanted(kinds: &str, label: &str) -> bool {
	let has = |set: &[&str]| set.contains(&label);
	match kinds {
		"things that move" => has(PEOPLE) || has(VEHICLES) || has(ANIMALS),
		"people" => has(PEOPLE),
		"vehicles" => has(VEHICLES),
		"animals" => has(ANIMALS),
		_ => true,
	}
}

/// Where the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
	Cpu,
	/// any DirectX 12 graphics card, on Windows
	DirectMl,
	/// the GPU through Dawn (D3D12 on Windows, Vulkan or Metal elsewhere)
	WebGpu,
}

impl std::str::FromStr for Device {
	type Err = anyhow::Error;
	fn from_str(s: &str) -> Result<Self> {
		Ok(match s.to_ascii_lowercase().as_str() {
			"cpu" => Device::Cpu,
			"directml" | "dml" => Device::DirectMl,
			"webgpu" | "gpu" => Device::WebGpu,
			_ => bail!("no device called {s}: cpu, directml or webgpu"),
		})
	}
}

/// D-FINE comes in four sizes: s, m, l and x, each slower and surer than the last. On a recent
/// graphics card even x keeps up with video.
pub const SIZES: [&str; 4] = ["s", "m", "l", "x"];

/// What it's been taught to recognise: COCO's 80 kinds, or Objects365's 365.
pub const SETS: [&str; 2] = ["coco", "obj365"];

/// A model named the way spotter's settings name them: a size, with "365" after it for the
/// Objects365 one ("x", "x365"). Gives the size and the set.
pub fn parse_model(name: &str) -> Result<(&str, &str)> {
	let (size, set) = match name.strip_suffix("365") {
		Some(s) => (s, "obj365"),
		None => (name, "coco"),
	};
	if !SIZES.contains(&size) {
		bail!("D-FINE comes in s, m, l and x (with 365 after it for 365 kinds), not {name}");
	}
	Ok((size, set))
}

/// Where downloaded models are kept (paths.rs).
pub fn model_dir() -> PathBuf {
	crate::paths::models()
}

/// D-FINE at a size, downloading it from Hugging Face and fixing it for DirectML the first time.
pub fn dfine(size: &str) -> Result<PathBuf> {
	dfine_in(size, "coco")
}

/// Whether D-FINE at a size, taught `set`, is downloaded already (named as the window names it:
/// "m", or "m365" for Objects365's kinds).
pub fn dfine_ready(name: &str) -> bool {
	let size = name.trim_end_matches("365");
	let set = if name.ends_with("365") { "obj365" } else { "coco" };
	model_dir().join(format!("dfine_{size}_{set}.onnx")).exists()
}

/// D-FINE at a size, taught one of the `SETS`.
pub fn dfine_in(size: &str, set: &str) -> Result<PathBuf> {
	if !SIZES.contains(&size) {
		bail!("D-FINE comes in s, m, l and x, not {size}");
	}
	if !SETS.contains(&set) {
		bail!("D-FINE is taught coco or obj365, not {set}");
	}
	let dir = model_dir();
	let path = dir.join(format!("dfine_{size}_{set}.onnx"));
	if path.exists() {
		return Ok(path);
	}
	// one download at a time (the window's detector and the look ahead's both ask for it at once,
	// and wrote the same file together); whoever waits finds it there
	let _one = crate::model::FETCHING.lock().unwrap_or_else(|e| e.into_inner());
	if path.exists() {
		return Ok(path);
	}
	let url = format!("https://huggingface.co/onnx-community/dfine_{size}_{set}-ONNX/resolve/main/onnx/model.onnx");
	let raw = dir.join(format!("dfine_{size}_{set}.download"));
	let what = format!("d-fine {size}{}", if set == "obj365" { " 365" } else { "" });
	crate::model::download(&what, &url, &raw)?;
	// its matrix product with a plain list turns its guesses into box edges
	let _saying = crate::model::Saying::fixing(&what);
	crate::onnx::patch(&raw, &path)?;
	let _ = std::fs::remove_file(&raw);
	Ok(path)
}

/// ONNX Runtime's one environment for the whole program.
pub(crate) fn environment() -> Result<&'static ort::environment::Environment> {
	static ENV: std::sync::OnceLock<ort::environment::Environment> = std::sync::OnceLock::new();
	if let Some(e) = ENV.get() {
		return Ok(e);
	}
	let e = ort::environment::init()
		.with_name("spotter")
		.build()
		.map_err(|e| anyhow!("couldn't start ONNX Runtime: {e}"))?;
	Ok(ENV.get_or_init(|| e))
}

pub struct Detector {
	session: Session,
	pub device: Device,
	/// the model looks at the whole frame squeezed to this square, which is how it was trained
	size: u32,
	pixels: Vec<f32>,
	small: Frame,
	/// what each of the model's answers is called
	labels: Vec<String>,
	/// how long the last look took, in milliseconds
	pub ms: f32,
}

impl Detector {
	pub fn load(model: &Path, device: Device) -> Result<Self> {
		// on the GPU, it's set up and run on the GPU's own thread (see gpu.rs)
		if device == Device::Cpu {
			Self::load_here(model, device)
		} else {
			crate::gpu::on_gpu(|| Self::load_here(model, device))
		}
	}

	fn load_here(model: &Path, device: Device) -> Result<Self> {
		let session = crate::model::session(model, device, 3)?;
		let size = 640;
		// the Objects365 models say so in their name
		let labels: Vec<String> = if model.to_string_lossy().contains("obj365") {
			OBJ365.lines().map(str::to_string).collect()
		} else {
			LABELS.iter().map(|l| l.to_string()).collect()
		};
		let mut d = Self {
			session,
			device,
			size,
			pixels: vec![0.0; (3 * size * size) as usize],
			small: Frame::new(size, size),
			labels,
			ms: 0.0,
		};
		// the first look sets everything up on the GPU and takes seconds; get it over with
		d.run()?;
		Ok(d)
	}

	fn run(&mut self) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
		if self.device == Device::Cpu {
			self.run_here()
		} else {
			crate::gpu::run_on_gpu("the detector", || self.run_here())
		}
	}

	fn run_here(&mut self) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
		let s = self.size as usize;
		let input = TensorRef::from_array_view(([1usize, 3, s, s], &self.pixels[..])).map_err(|e| anyhow!("{e}"))?;
		let out =
			self.session.run(ort::inputs!["pixel_values" => input]).map_err(|e| anyhow!("the model failed: {e}"))?;
		let (shape, logits) = out["logits"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
		let (queries, classes) = (shape[1] as usize, shape[2] as usize);
		let (_, boxes) = out["pred_boxes"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
		Ok((logits.to_vec(), boxes.to_vec(), queries, classes))
	}

	/// What's in the frame and where (0-1 across it). `confidence` is the look's setting: things
	/// already being followed may wobble a little under it without blinking off, so this reports down
	/// to a little below it and leaves the line to the tracks. `kinds` is the look's "look for".
	pub fn detect(&mut self, frame: &Frame, confidence: f32, kinds: &str) -> Result<Vec<Detection>> {
		frame.resize_fast_into(&mut self.small);
		let n = (self.size * self.size) as usize;
		for (i, p) in self.small.rgba.chunks_exact(4).enumerate() {
			self.pixels[i] = p[0] as f32 / 255.0;
			self.pixels[n + i] = p[1] as f32 / 255.0;
			self.pixels[2 * n + i] = p[2] as f32 / 255.0;
		}
		let t0 = Instant::now();
		let (scores, boxes, queries, classes) = self.run()?;
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;

		let floor = (confidence - 0.12).max(0.1);
		let line = (floor / (1.0 - floor)).ln();
		// a model with one answer fewer than it has names leaves out the first, "none"
		let skip = if classes + 1 == self.labels.len() { 1 } else { 0 };
		let mut found = Vec::new();
		for q in 0..queries {
			let mut best = line;
			let mut cls = None;
			for c in 0..classes.min(self.labels.len() - skip) {
				if scores[q * classes + c] > best {
					best = scores[q * classes + c];
					cls = Some(c + skip);
				}
			}
			let Some(c) = cls else { continue };
			let label = &self.labels[c];
			if label == "none" || !wanted(kinds, label) {
				continue;
			}
			let (cx, cy, w, h) = (boxes[q * 4], boxes[q * 4 + 1], boxes[q * 4 + 2], boxes[q * 4 + 3]);
			found.push(Detection {
				cls: label.clone(),
				score: 1.0 / (1.0 + (-best).exp()),
				x: cx - w / 2.0,
				y: cy - h / 2.0,
				w,
				h,
			});
		}
		Ok(found)
	}
}
