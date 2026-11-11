//! Any ONNX model, on the CPU or the GPU, and the list of models spotter knows how to
//! fetch. DirectML gets some models wrong without saying so (D-FINE's box edges all came out as -4
//! until its 1-D matrix products were rewritten), so every model is checked against the CPU before
//! it's trusted: `spotter model check <file>` runs one on both with the same input and prints how far
//! apart the answers are.

use crate::detector::{Device, environment};
use anyhow::{Context, Result, anyhow, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{DynValue, TensorElementType, TensorRef, ValueType};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A tensor's numbers, of the kinds models here use.
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
	F32(Vec<f32>),
	I64(Vec<i64>),
	I32(Vec<i32>),
	U8(Vec<u8>),
	Bool(Vec<bool>),
}

impl Data {
	pub fn len(&self) -> usize {
		match self {
			Data::F32(v) => v.len(),
			Data::I64(v) => v.len(),
			Data::I32(v) => v.len(),
			Data::U8(v) => v.len(),
			Data::Bool(v) => v.len(),
		}
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Every value as an f64, for comparing.
	pub fn values(&self) -> Vec<f64> {
		match self {
			Data::F32(v) => v.iter().map(|&x| x as f64).collect(),
			Data::I64(v) => v.iter().map(|&x| x as f64).collect(),
			Data::I32(v) => v.iter().map(|&x| x as f64).collect(),
			Data::U8(v) => v.iter().map(|&x| x as f64).collect(),
			Data::Bool(v) => v.iter().map(|&x| x as u8 as f64).collect(),
		}
	}
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
	pub shape: Vec<i64>,
	pub data: Data,
}

impl Tensor {
	pub fn f32(shape: &[i64], v: Vec<f32>) -> Self {
		Self { shape: shape.to_vec(), data: Data::F32(v) }
	}

	pub fn i64(shape: &[i64], v: Vec<i64>) -> Self {
		Self { shape: shape.to_vec(), data: Data::I64(v) }
	}

	pub fn as_f32(&self) -> Option<&[f32]> {
		match &self.data {
			Data::F32(v) => Some(v),
			_ => None,
		}
	}
}

/// One of a model's inputs or outputs: its name, kind of number and shape (-1 where it can vary).
#[derive(Clone, Debug)]
pub struct Port {
	pub name: String,
	pub kind: String,
	pub shape: Vec<i64>,
}

fn port(name: &str, t: &ValueType) -> Port {
	match t {
		ValueType::Tensor { ty, shape, .. } => {
			Port { name: name.to_string(), kind: format!("{ty:?}"), shape: shape.to_vec() }
		}
		other => Port { name: name.to_string(), kind: format!("{other:?}"), shape: Vec::new() },
	}
}

fn extract(v: &DynValue) -> Result<Tensor> {
	let fail = |e: ort::Error| anyhow!("{e}");
	let ValueType::Tensor { ty, .. } = v.dtype() else { bail!("an output that isn't a tensor") };
	Ok(match ty {
		TensorElementType::Float32 => {
			let (s, d) = v.try_extract_tensor::<f32>().map_err(fail)?;
			Tensor { shape: s.to_vec(), data: Data::F32(d.to_vec()) }
		}
		TensorElementType::Int64 => {
			let (s, d) = v.try_extract_tensor::<i64>().map_err(fail)?;
			Tensor { shape: s.to_vec(), data: Data::I64(d.to_vec()) }
		}
		TensorElementType::Int32 => {
			let (s, d) = v.try_extract_tensor::<i32>().map_err(fail)?;
			Tensor { shape: s.to_vec(), data: Data::I32(d.to_vec()) }
		}
		TensorElementType::Uint8 => {
			let (s, d) = v.try_extract_tensor::<u8>().map_err(fail)?;
			Tensor { shape: s.to_vec(), data: Data::U8(d.to_vec()) }
		}
		TensorElementType::Bool => {
			let (s, d) = v.try_extract_tensor::<bool>().map_err(fail)?;
			Tensor { shape: s.to_vec(), data: Data::Bool(d.to_vec()) }
		}
		other => bail!("an output of {other:?}, which spotter doesn't read"),
	})
}

/// A model loaded on a device.
pub struct Model {
	pub session: Session,
	pub device: Device,
	pub path: PathBuf,
	/// how long the last run took, in milliseconds
	pub ms: f32,
}

impl Model {
	pub fn load(path: &Path, device: Device) -> Result<Self> {
		Self::load_with(path, device, 3)
	}

	/// `level` is how hard ONNX Runtime rewrites the graph: 0 not at all, up to 3 (the default). A
	/// model DirectML gets wrong is sometimes right at a lower level, the fault being in a fused step.
	pub fn load_with(path: &Path, device: Device, level: u8) -> Result<Self> {
		// on the GPU, it's set up on the GPU's own thread (see gpu.rs)
		if device == Device::Cpu {
			Self::load_here(path, device, level)
		} else {
			crate::gpu::on_gpu(|| Self::load_here(path, device, level))
		}
	}

	fn load_here(path: &Path, device: Device, level: u8) -> Result<Self> {
		Ok(Self { session: session(path, device, level)?, device, path: path.to_path_buf(), ms: 0.0 })
	}

	pub fn inputs(&self) -> Vec<Port> {
		self.session.inputs().iter().map(|o| port(o.name(), o.dtype())).collect()
	}

	pub fn outputs(&self) -> Vec<Port> {
		self.session.outputs().iter().map(|o| port(o.name(), o.dtype())).collect()
	}

	/// Run it on these inputs and copy every output out.
	pub fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<(String, Tensor)>> {
		if self.device == Device::Cpu {
			self.run_here(inputs)
		} else {
			let name = self.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
			crate::gpu::run_on_gpu(&name, || self.run_here(inputs))
		}
	}

	fn run_here(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<(String, Tensor)>> {
		let fail = |e: ort::Error| anyhow!("{e}");
		let mut values: Vec<(String, ort::session::SessionInputValue<'_>)> = Vec::with_capacity(inputs.len());
		for (name, t) in inputs {
			let shape = t.shape.clone();
			let v: ort::session::SessionInputValue<'_> = match &t.data {
				Data::F32(d) => TensorRef::from_array_view((shape, &d[..])).map_err(fail)?.into(),
				Data::I64(d) => TensorRef::from_array_view((shape, &d[..])).map_err(fail)?.into(),
				Data::I32(d) => TensorRef::from_array_view((shape, &d[..])).map_err(fail)?.into(),
				Data::U8(d) => TensorRef::from_array_view((shape, &d[..])).map_err(fail)?.into(),
				Data::Bool(d) => TensorRef::from_array_view((shape, &d[..])).map_err(fail)?.into(),
			};
			values.push((name.to_string(), v));
		}
		let t0 = Instant::now();
		let out = self.session.run(values).map_err(|e| anyhow!("{} failed: {e}", self.path.display()))?;
		self.ms = t0.elapsed().as_secs_f32() * 1000.0;
		let mut got = Vec::with_capacity(out.len());
		for (name, v) in out.iter() {
			got.push((name.to_string(), extract(&v)?));
		}
		Ok(got)
	}
}

/// An ONNX Runtime session for the model at `path` on `device`, its graph rewritten at `level` (0 not
/// at all, up to 3).
pub(crate) fn session(path: &Path, device: Device, level: u8) -> Result<Session> {
	let fail = |e: ort::Error<_>| anyhow!("{e}");
	let level = match level {
		0 => GraphOptimizationLevel::Disable,
		1 => GraphOptimizationLevel::Level1,
		2 => GraphOptimizationLevel::Level2,
		_ => GraphOptimizationLevel::Level3,
	};
	let mut b =
		Session::builder(environment()?).map_err(|e| anyhow!("{e}"))?.with_optimization_level(level).map_err(fail)?;
	b = match device {
		Device::Cpu => b,
		// DirectML can't take ONNX Runtime's memory patterns or parallel running
		#[cfg(feature = "directml")]
		Device::DirectMl => b
			.with_memory_pattern(false)
			.map_err(fail)?
			.with_parallel_execution(false)
			.map_err(fail)?
			.with_execution_providers([ort::ep::DirectML::default().build().error_on_failure()])
			.map_err(fail)?,
		#[cfg(feature = "webgpu")]
		Device::WebGpu => {
			let mut ep = ort::ep::WebGPU::default();
			// some nodes trip shape faults in this WebGPU build, and are small enough for the CPU
			if let Ok(names) = std::env::var("SPOTTER_CPU_NODES") {
				ep = ep.with_force_cpu_node_names(names);
			}
			b.with_execution_providers([ep.build().error_on_failure()]).map_err(fail)?
		}
		#[allow(unreachable_patterns)]
		other => bail!("this build of spotter can't use {other:?}"),
	};
	b.commit_from_file(path).map_err(|e| anyhow!("couldn't load {}: {e}", path.display()))
}

/// splitmix64, for inputs that come out the same every time.
fn noise(seed: &mut u64) -> f32 {
	*seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
	let mut z = *seed;
	z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
	z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
	((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32
}

/// What `check` feeds a model.
#[derive(Clone, Debug, Default)]
pub struct CheckInput {
	/// sizes for dimensions that can vary, by input name; any left over are 1
	pub shapes: Vec<(String, Vec<i64>)>,
	/// a picture for a [1, 3, h, w] input, as 0-1 RGB; otherwise inputs are seeded noise
	pub picture: Option<crate::Frame>,
	/// take ImageNet's mean and spread off the picture
	pub imagenet: bool,
	/// ONNX Runtime's graph rewriting level for the GPU (0-3)
	pub level: u8,
	pub runs: u32,
}

/// How one output compares between the CPU and the GPU.
#[derive(Clone, Debug)]
pub struct Agreement {
	pub name: String,
	pub shape: Vec<i64>,
	pub largest: f64,
	pub mean: f64,
	/// the largest difference over the largest value the CPU gave
	pub relative: f64,
	/// the difference 99 values in 100 are within, over the largest value the CPU gave. A detector's
	/// unsure guesses can come in a different order on different hardware, which shows in `relative`
	/// but not here
	pub p99: f64,
}

#[derive(Clone, Debug)]
pub struct Check {
	pub inputs: Vec<Port>,
	pub outputs: Vec<Agreement>,
	pub cpu_ms: f32,
	pub gpu_ms: f32,
	pub gpu: Device,
}

impl Check {
	/// Whether 99 in 100 of every output's values agree to within `tolerance` of its size.
	pub fn agrees(&self, tolerance: f64) -> bool {
		self.outputs.iter().all(|o| o.p99 <= tolerance)
	}
}

/// The inputs `check` makes for a model, from its ports.
pub fn check_inputs(ports: &[Port], c: &CheckInput) -> Result<Vec<(String, Tensor)>> {
	let mut seed = 7u64;
	let mut out = Vec::new();
	for p in ports {
		let given = c.shapes.iter().find(|(n, _)| *n == p.name).map(|(_, s)| s.clone());
		let shape: Vec<i64> = match given {
			Some(s) => s,
			None => p.shape.iter().map(|&d| if d < 0 { 1 } else { d }).collect(),
		};
		let n: usize = shape.iter().map(|&d| d.max(0) as usize).product();
		let t = match p.kind.as_str() {
			"Float32" => {
				let v = match &c.picture {
					Some(f) if shape.len() == 4 && shape[1] == 3 => {
						let (h, w) = (shape[2] as u32, shape[3] as u32);
						let mut small = crate::Frame::new(w, h);
						f.resize_into(&mut small);
						let plane = (w * h) as usize;
						let mut v = vec![0.0; 3 * plane];
						let (mean, sd) = if c.imagenet {
							([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
						} else {
							([0.0; 3], [1.0; 3])
						};
						for (i, px) in small.rgba.chunks_exact(4).enumerate() {
							for ch in 0..3 {
								v[ch * plane + i] = (px[ch] as f32 / 255.0 - mean[ch]) / sd[ch];
							}
						}
						v
					}
					_ => (0..n).map(|_| noise(&mut seed)).collect(),
				};
				Tensor::f32(&shape, v)
			}
			"Int64" => Tensor { shape, data: Data::I64(vec![1; n]) },
			"Int32" => Tensor { shape, data: Data::I32(vec![1; n]) },
			"Uint8" => Tensor { shape, data: Data::U8((0..n).map(|_| (noise(&mut seed) * 255.0) as u8).collect()) },
			"Bool" => Tensor { shape, data: Data::Bool(vec![true; n]) },
			other => bail!("{} wants {other}, which check can't make", p.name),
		};
		out.push((p.name.clone(), t));
	}
	Ok(out)
}

/// Run a model on the CPU and on the GPU with the same input, and compare.
pub fn check(path: &Path, gpu: Device, c: &CheckInput) -> Result<Check> {
	let mut cpu = Model::load(path, Device::Cpu).context("on the processor")?;
	let ports = cpu.inputs();
	let inputs = check_inputs(&ports, c)?;
	let refs: Vec<(&str, &Tensor)> = inputs.iter().map(|(n, t)| (n.as_str(), t)).collect();
	let runs = c.runs.max(1);
	let time = |m: &mut Model| -> Result<(Vec<(String, Tensor)>, f32)> {
		let first = m.run(&refs)?;
		let t0 = Instant::now();
		for _ in 0..runs {
			m.run(&refs)?;
		}
		Ok((first, t0.elapsed().as_secs_f32() * 1000.0 / runs as f32))
	};
	let (want, cpu_ms) = time(&mut cpu)?;
	drop(cpu);
	let mut g = Model::load_with(path, gpu, c.level).with_context(|| format!("on {gpu:?}"))?;
	let (got, gpu_ms) = time(&mut g)?;
	let mut outputs = Vec::new();
	for (name, a) in &want {
		let b = got
			.iter()
			.find(|(n, _)| n == name)
			.map(|(_, t)| t)
			.ok_or_else(|| anyhow!("the graphics card gave no {name}"))?;
		if a.shape != b.shape {
			bail!("{name} came out {:?} on the processor and {:?} on the graphics card", a.shape, b.shape);
		}
		let (x, y) = (a.data.values(), b.data.values());
		let (mut sum, mut top) = (0f64, 0f64);
		let mut diffs: Vec<f64> = x
			.iter()
			.zip(&y)
			.map(|(p, q)| {
				top = top.max(p.abs());
				let d = if p.is_nan() && q.is_nan() {
					0.0
				} else if p.is_nan() || q.is_nan() {
					f64::INFINITY
				} else {
					(p - q).abs()
				};
				sum += d;
				d
			})
			.collect();
		diffs.sort_by(f64::total_cmp);
		let largest = diffs.last().copied().unwrap_or(0.0);
		let p99 = diffs.get((diffs.len() as f64 * 0.99) as usize).copied().unwrap_or(largest);
		let mean = if x.is_empty() { 0.0 } else { sum / x.len() as f64 };
		let of = |d: f64| if top > 0.0 { d / top } else { d };
		outputs.push(Agreement {
			name: name.clone(),
			shape: a.shape.clone(),
			largest,
			mean,
			relative: of(largest),
			p99: of(p99),
		});
	}
	Ok(Check { inputs: ports, outputs, cpu_ms, gpu_ms, gpu })
}

// ---- fetching -------------------------------------------------------------------------------

/// A model spotter can download.
#[derive(Clone, Copy, Debug)]
pub struct Known {
	/// what it's called here, and the folder it's kept in
	pub name: &'static str,
	/// a Hugging Face repo
	pub repo: &'static str,
	/// the files to fetch, as paths in the repo; the first is the model to load. Weights kept beside
	/// a model (`.onnx_data`) must be listed and land next to it under their own names
	pub files: &'static [&'static str],
	pub licence: &'static str,
	/// whether to rewrite it for DirectML (onnx.rs)
	pub patch: bool,
	/// input sizes to fix it at before that (as name:1,3,1024,1024), so the shape arithmetic DirectML
	/// can't do is worked out ahead (`fold`); empty to leave it
	pub slim: &'static str,
	/// run a half-precision copy on the GPU (scripts/fp16.py), which is faster for this one
	/// and as good: measured against the full one on this machine before saying so
	pub half: bool,
	pub what: &'static str,
}

pub const KNOWN: &[Known] = &[
	Known {
		name: "realesr-general-x4v3",
		repo: "jonathanst29/tinier-upscale-models",
		files: &["realesr-general-x4v3.onnx"],
		licence: "BSD-3-Clause",
		patch: false,
		slim: "",
		half: true,
		what: "Real-ESRGAN's small general model: makes a picture four times bigger, with detail",
	},
	Known {
		name: "edgetam-encoder",
		repo: "onnx-community/EdgeTAM-ONNX",
		files: &["onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx_data"],
		licence: "Apache-2.0",
		patch: true,
		slim: "pixel_values:1,3,1024,1024",
		half: true,
		what: "EdgeTAM's picture encoder, for cutting things out along their outlines",
	},
	Known {
		name: "edgetam-decoder",
		repo: "onnx-community/EdgeTAM-ONNX",
		files: &["onnx/prompt_encoder_mask_decoder.onnx", "onnx/prompt_encoder_mask_decoder.onnx_data"],
		licence: "Apache-2.0",
		patch: false,
		slim: "",
		half: true,
		what: "EdgeTAM's outline maker: given a box, the shape of the thing in it",
	},
	Known {
		name: "depth-anything-v2-small",
		repo: "onnx-community/depth-anything-v2-small-ONNX",
		files: &["onnx/model.onnx", "onnx/model.onnx_data"],
		licence: "Apache-2.0",
		patch: true,
		slim: "pixel_values:1,3,392,392",
		half: false,
		what: "Depth Anything V2 small: how far away each part of the picture is",
	},
	Known {
		name: "depth-anything-v2-large",
		repo: "onnx-community/depth-anything-v2-large-ONNX",
		files: &["onnx/model.onnx", "onnx/model.onnx_data"],
		licence: "CC-BY-NC-4.0",
		patch: true,
		slim: "pixel_values:1,3,518,518",
		half: true,
		what: "Depth Anything V2 large: the best of its depths, for exports",
	},
	Known {
		name: "sam2.1-large-encoder",
		repo: "onnx-community/sam2.1-hiera-large-ONNX",
		files: &["onnx/vision_encoder.onnx", "onnx/vision_encoder.onnx_data"],
		licence: "Apache-2.0",
		patch: true,
		slim: "pixel_values:1,3,1024,1024",
		half: true,
		what: "SAM 2.1 large's picture encoder: truer outlines than EdgeTAM's, for exports",
	},
	Known {
		name: "sam2.1-large-decoder",
		repo: "onnx-community/sam2.1-hiera-large-ONNX",
		files: &["onnx/prompt_encoder_mask_decoder.onnx", "onnx/prompt_encoder_mask_decoder.onnx_data"],
		licence: "Apache-2.0",
		patch: false,
		slim: "",
		half: true,
		what: "SAM 2.1 large's outline maker",
	},
	Known {
		name: "rife-4.9",
		repo: "yuvraj108c/rife-onnx",
		files: &["rife49_ensemble_True_scale_1_sim.onnx"],
		licence: "MIT",
		patch: false,
		slim: "",
		half: false,
		what: "RIFE 4.9: the frames between two frames, for smooth slow motion",
	},
	Known {
		name: "segformer-b2-ade",
		repo: "Xenova/segformer-b2-finetuned-ade-512-512",
		files: &["onnx/model.onnx"],
		// NVIDIA's SegFormer licence: research and other non-commercial use
		licence: "NVIDIA Source Code License (non-commercial)",
		// no 1-D matrix products in it (and it's an older opset than the script takes)
		patch: false,
		slim: "pixel_values:1,3,512,512",
		half: true,
		what: "SegFormer B2 on ADE20K: what each part of the scene is (sky, sea, tree, grass, road… 150 kinds)",
	},
	Known {
		name: "segformer-b5-ade",
		repo: "Xenova/segformer-b5-finetuned-ade-640-640",
		files: &["onnx/model.onnx"],
		licence: "NVIDIA Source Code License (non-commercial)",
		patch: false,
		slim: "pixel_values:1,3,640,640",
		half: true,
		what: "SegFormer B5 on ADE20K at 640: the biggest, surest SegFormer, for labelling the scene",
	},
];

pub fn known(name: &str) -> Option<&'static Known> {
	KNOWN.iter().find(|k| k.name == name)
}

/// The Python that makes half-precision copies: `SPOTTER_PYTHON`, else the one spotter keeps in its
/// folder (`%LOCALAPPDATA%\spotter\py`, with `onnx` and `onnxruntime` in it), else `python`.
pub fn python() -> PathBuf {
	if let Ok(p) = std::env::var("SPOTTER_PYTHON") {
		return p.into();
	}
	let py = crate::paths::home().join("py");
	let own = if cfg!(windows) { py.join("Scripts").join("python.exe") } else { py.join("bin").join("python") };
	if own.exists() { own } else { "python".into() }
}

/// Held while a model's downloaded and fixed, so only one is at a time.
pub(crate) static FETCHING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// What's being downloaded or fixed the first time it's needed, for a status line.
#[derive(Clone, Debug, PartialEq)]
pub struct Getting {
	pub what: String,
	/// bytes in so far and in all (0 if the server didn't say), while downloading
	pub bytes: Option<(u64, u64)>,
}

impl Getting {
	pub fn line(&self) -> String {
		let mb = |b: u64| b as f64 / 1e6;
		match self.bytes {
			Some((got, 0)) => format!("getting {} · {:.0} MB · first time only", self.what, mb(got)),
			Some((got, of)) => format!("getting {} · {:.0} of {:.0} MB · first time only", self.what, mb(got), mb(of)),
			None => format!("fixing {} for the graphics card · first time only", self.what),
		}
	}
}

/// What's being got: its name, and the file it's downloading into with the size it will be.
type Got = Option<(String, Option<(PathBuf, u64)>)>;
static GETTING: std::sync::Mutex<Got> = std::sync::Mutex::new(None);

/// What's being downloaded or fixed, if anything.
pub fn getting() -> Option<Getting> {
	let g = GETTING.lock().unwrap_or_else(|e| e.into_inner()).clone()?;
	let bytes = g.1.map(|(part, of)| (std::fs::metadata(part).map_or(0, |m| m.len()), of));
	Some(Getting { what: g.0, bytes })
}

/// Says what's being got until it's dropped.
pub(crate) struct Saying;

impl Saying {
	pub(crate) fn fixing(what: &str) -> Self {
		*GETTING.lock().unwrap_or_else(|e| e.into_inner()) = Some((what.to_string(), None));
		Saying
	}
}

impl Drop for Saying {
	fn drop(&mut self) {
		*GETTING.lock().unwrap_or_else(|e| e.into_inner()) = None;
	}
}

/// How big the file at `url` is, as the server says after any redirects; 0 if it doesn't.
fn size_of(url: &str) -> u64 {
	let Ok(out) =
		crate::quiet(&mut std::process::Command::new("curl")).args(["-sIL", "--max-time", "15", url]).output()
	else {
		return 0;
	};
	let head = String::from_utf8_lossy(&out.stdout);
	head.lines()
		.filter_map(|l| l.split_once(':'))
		.filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
		.filter_map(|(_, v)| v.trim().parse().ok())
		.next_back()
		.unwrap_or(0)
}

/// Download one file with curl into `to`, unless it's there already, saying it's getting `what`
/// meanwhile. It goes into `<to>.part` first, which a later try carries on from.
pub fn download(what: &str, url: &str, to: &Path) -> Result<()> {
	if to.exists() {
		return Ok(());
	}
	if let Some(d) = to.parent() {
		std::fs::create_dir_all(d)?;
	}
	let part = crate::paths::part(to);
	*GETTING.lock().unwrap_or_else(|e| e.into_inner()) = Some((what.to_string(), Some((part.clone(), 0))));
	let _saying = Saying;
	let of = size_of(url);
	if let Some(g) = GETTING.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
		g.1 = Some((part.clone(), of));
	}
	eprintln!("downloading {url}");
	// curl's own progress bar only where someone's watching it (`getting` tells the window)
	let quiet = if std::io::IsTerminal::is_terminal(&std::io::stderr()) { "-#" } else { "-sS" };
	let ok = crate::quiet(&mut std::process::Command::new("curl"))
		.args(["-fL", "--retry", "3", "-C", "-", quiet, "-o"])
		.arg(&part)
		.arg(url)
		.status()
		.context("couldn't run curl to download")?;
	if !ok.success() {
		bail!("couldn't download {what} from {url}");
	}
	std::fs::rename(&part, to)?;
	Ok(())
}

/// Save what ONNX Runtime makes of a model on the CPU with its input sizes fixed (`name:1,3,64,64`):
/// the sizes inside worked out ahead, since DirectML fails on some of that arithmetic. Only the basic
/// rewrites run, which suit any device; the weights go in a file of their own.
pub fn fold(from: &Path, sizes: &str, to: &Path) -> Result<()> {
	let named = crate::onnx::named_sizes(&crate::onnx::load(from)?, sizes)?;
	let part = crate::paths::part(to);
	let data = format!("{}_data", to.file_name().context("a model with no file name")?.to_string_lossy());
	let fail = |e: ort::Error<_>| anyhow!("{e}");
	let mut b = Session::builder(environment()?)
		.map_err(|e| anyhow!("{e}"))?
		.with_optimization_level(GraphOptimizationLevel::Level1)
		.map_err(fail)?
		.with_optimized_model_path(&part)
		.map_err(fail)?
		.with_config_entry("session.optimized_model_external_initializers_file_name", &data)
		.map_err(fail)?
		.with_config_entry("session.optimized_model_external_initializers_min_size_in_bytes", "1024")
		.map_err(fail)?;
	for (name, size) in &named {
		b = b.with_dimension_override(name, *size).map_err(fail)?;
	}
	let s = b.commit_from_file(from).map_err(|e| anyhow!("couldn't simplify {}: {e}", from.display()))?;
	drop(s);
	std::fs::rename(&part, to)?;
	Ok(())
}

/// The file a known model is loaded from, once it's fetched.
pub fn ready(k: &Known) -> PathBuf {
	let main = Path::new(k.files[0]).file_name().unwrap();
	crate::detector::model_dir().join(k.name).join(if k.patch { Path::new("patched.onnx").as_os_str() } else { main })
}

/// A known model, downloaded (and fixed for DirectML) the first time. Returns the file to load.
pub fn fetch(k: &Known) -> Result<PathBuf> {
	let ready = ready(k);
	let dir = ready.parent().unwrap().to_path_buf();
	let main = Path::new(k.files[0]).file_name().unwrap();
	if ready.exists() {
		return Ok(ready);
	}
	// one download at a time: two at once (the window's model and the look ahead's, say) wrote the
	// same file together; whoever waits finds it there
	let _one = FETCHING.lock().unwrap_or_else(|e| e.into_inner());
	if ready.exists() {
		return Ok(ready);
	}
	let raw = dir.join("raw");
	for f in k.files {
		let name = Path::new(f).file_name().unwrap();
		download(k.name, &format!("https://huggingface.co/{}/resolve/main/{f}", k.repo), &raw.join(name))?;
	}
	if !k.patch {
		for f in k.files {
			let name = Path::new(f).file_name().unwrap();
			std::fs::rename(raw.join(name), dir.join(name))?;
		}
		let _ = std::fs::remove_dir(&raw);
		return Ok(ready);
	}
	let _saying = Saying::fixing(k.name);
	let mut from = raw.join(main);
	if !k.slim.is_empty() {
		let folded = raw.join("folded.onnx");
		fold(&from, k.slim, &folded)?;
		from = folded;
	}
	crate::onnx::patch(&from, &ready)?;
	// the fixed one has everything in it
	let _ = std::fs::remove_dir_all(&raw);
	Ok(ready)
}

/// A known model by name, fetched.
pub fn get(name: &str) -> Result<PathBuf> {
	fetch(known(name).with_context(|| format!("spotter doesn't know a model called {name}"))?)
}

/// A known model by name, fetched, as it should run on `device`: with `half`, the half-precision copy
/// where that's faster and as good (`SPOTTER_FULL=1` keeps to the full one). The window keeps to full
/// precision: half models beside its detectors make DirectML fail a run now and then.
pub fn get_for(name: &str, device: Device, half: bool) -> Result<PathBuf> {
	let k = known(name).with_context(|| format!("spotter doesn't know a model called {name}"))?;
	let full = fetch(k)?;
	if !half || !k.half || device == Device::Cpu || std::env::var_os("SPOTTER_FULL").is_some() {
		return Ok(full);
	}
	let half = full.with_extension("half.onnx");
	if half.exists() {
		return Ok(half);
	}
	// Python is asked once a run: without it every model would start a program that fails
	static NO_PYTHON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
	if NO_PYTHON.load(std::sync::atomic::Ordering::Relaxed) {
		return Ok(full);
	}
	let script = crate::detector::model_dir().join("fp16.py");
	std::fs::write(&script, HALF)?;
	let py = python();
	eprintln!("making a half-precision copy of {name} for the graphics card");
	let part = crate::paths::part(&half);
	let out = crate::quiet(&mut std::process::Command::new(&py)).arg("-I").arg(&script).arg(&full).arg(&part).output();
	if let Ok(o) = &out
		&& o.status.success()
		&& std::fs::rename(&part, &half).is_ok()
	{
		return Ok(half);
	}
	NO_PYTHON.store(true, std::sync::atomic::Ordering::Relaxed);
	let _ = std::fs::remove_file(&part);
	let why = match &out {
		Ok(o) => String::from_utf8_lossy(&o.stderr).lines().last().unwrap_or_default().to_string(),
		Err(e) => format!("couldn't run {}: {e}", py.display()),
	};
	eprintln!("no half-precision copies this run (they need Python with `onnx` and `onnxruntime`): {why}");
	Ok(full)
}

/// scripts/fp16.py, written out beside the models when it's needed.
const HALF: &str = include_str!("../../../scripts/fp16.py");
