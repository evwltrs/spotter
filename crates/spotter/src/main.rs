//! spotter from the command line: render a video with a look drawn over it, every frame worked out in
//! turn, at any shape, with the sound kept.
//!
//!   spotter render clip.mp4 --look cctv --aspect 9:16 -o clip-cctv.mp4
//!   spotter render random --look wake          (a video from files.1nf.me, as the site picks them)
//!   spotter render files:random/clip.mp4       (a particular one)
//!   spotter files [words]                      (what's on files.1nf.me)
//!   spotter looks

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use spotter_core::Look;
use spotter_core::detector::Device;
use spotter_core::edit::{Edit, Keep, Pick, RangeMode, Zone, ZoneMode};
use spotter_core::export::{Export, gpu, plan, run};
use spotter_core::project::Project;
use spotter_core::sources::{Dice, LEFT_OUT, catalog, decode, resolve};
use spotter_core::video::probe;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "spotter", version, about = "Finds and follows what moves in a video, and draws over it")]
struct Cli {
	#[command(subcommand)]
	cmd: Cmd,
}

// one of these is made a run
#[allow(clippy::large_enum_variant)]
#[derive(Subcommand)]
enum Cmd {
	/// Render a video with a look drawn over it.
	Render(Render),
	/// List the looks, or print one's settings as JSON (to save, change, and use with --settings or in a project).
	Looks { name: Option<String> },
	/// List the videos on files.1nf.me, or those whose path has all these words in it.
	Files { words: Vec<String> },
	/// The models: check one on the graphics card against the processor, fetch or list them.
	Model {
		#[command(subcommand)]
		cmd: ModelCmd,
	},
	/// Get what spotter needs that isn't here yet: ffmpeg (on Windows), and the detector.
	Setup,
	/// Say what spotter has to work with here, and what's missing.
	Doctor,
}

#[derive(Subcommand)]
enum ModelCmd {
	/// Run a model on the processor and the graphics card with the same input and say how far apart
	/// the answers are, and how long each took.
	Check {
		/// an .onnx file, or the name of one spotter knows (`spotter model list`)
		model: String,
		/// a size for an input whose size can vary, as name=1x3x640x640; can be repeated
		#[arg(long = "shape", value_name = "NAME=AxBxC")]
		shapes: Vec<String>,
		/// feed this picture to a [1, 3, h, w] input rather than noise
		#[arg(long)]
		picture: Option<PathBuf>,
		/// take ImageNet's mean and spread off the picture first
		#[arg(long)]
		imagenet: bool,
		/// ONNX Runtime's graph rewriting on the graphics card, 0 (none) to 3
		#[arg(long, default_value_t = 3)]
		level: u8,
		#[arg(long, default_value_t = 5)]
		runs: u32,
		/// the graphics card's way in: directml or webgpu
		#[arg(long, default_value = "directml")]
		device: String,
	},
	/// Download a model spotter knows, or a detector by its size (s, m, l, x; s365 for 365 kinds).
	Fetch { name: String },
	/// Rewrite a model's steps DirectML gets wrong (1-D matrix products, Reshape's allowzero).
	Patch { from: PathBuf, to: PathBuf },
	/// The models spotter knows.
	List,
}

#[derive(Parser)]
struct Render {
	/// the video to read: a file, a web address, files:<path> for one on files.1nf.me, or random; or a
	/// project saved from the window (.spotter), whose look, edit and export the other flags change
	input: String,
	/// where to write it; by default next to the input, named after the look
	#[arg(short, long)]
	output: Option<PathBuf>,
	/// a look by name (see `spotter looks`)
	#[arg(short, long, default_value = "smoke")]
	look: String,
	/// a saved look as JSON, laid over --look
	#[arg(long)]
	settings: Option<PathBuf>,
	/// change one setting, as name=value (camelCase names, as on the site); can be repeated
	#[arg(long = "set", value_name = "NAME=VALUE")]
	sets: Vec<String>,
	/// cut the picture to this shape, from the middle: 9:16, 4:5, 1:1, 16:9, …
	#[arg(long)]
	aspect: Option<String>,
	/// a shape and size for posting: reel (1080x1920), post (1080x1350), square (1080x1080) or
	/// wide (1920x1080)
	#[arg(long)]
	shape: Option<String>,
	/// make a picture smaller than the output bigger with Real-ESRGAN on the graphics card, rather
	/// than stretching it; with no --size or --shape the output grows to the look's longest side
	#[arg(long)]
	upscale: bool,
	/// use the window's quicker models (EdgeTAM, Depth Anything small, SegFormer B2) rather than the
	/// best ones (SAM 2.1 large, Depth Anything large, SegFormer B5)
	#[arg(long)]
	fast: bool,
	/// the exact size to write, like 1080x1920; the picture is cut to its shape and scaled to fit
	#[arg(long)]
	size: Option<String>,
	/// seconds into the video to start from
	#[arg(long, default_value_t = 0.0)]
	start: f64,
	/// a still instead: the frame this many seconds in, as a PNG (worked out from two seconds before)
	#[arg(long, value_name = "SECONDS")]
	still: Option<f64>,
	/// how many seconds to render
	#[arg(long)]
	length: Option<f64>,
	/// frames a second to write; by default the source's, up to 60
	#[arg(long)]
	fps: Option<f64>,
	/// where recognition runs: gpu, cpu, or off
	#[arg(long, default_value = "gpu")]
	device: String,
	/// the detector: its size, s, m, l or x, taught Objects365's 365 kinds of thing with 365 after it
	/// (x365) or COCO's 80 without (x)
	#[arg(long, default_value = "x365")]
	model: String,
	/// x264 quality: 18 looks lossless, lower is bigger
	#[arg(long, default_value_t = 18)]
	crf: u32,
	/// a saved edit as JSON (what's followed, where, and what's cut); the flags below add to it
	#[arg(long)]
	edit: Option<PathBuf>,
	/// recognise only these kinds, by the detector's names: person,car,aeroplane…
	#[arg(long, value_delimiter = ',')]
	kinds: Vec<String>,
	/// which things to follow: all, picked (see --pick) or main (the biggest)
	#[arg(long)]
	keep: Option<String>,
	/// follow the thing at this place at this time, as seconds,x,y with x and y 0-1 across the
	/// source; can be repeated. Sets --keep picked
	#[arg(long = "pick", value_name = "T,X,Y")]
	picks: Vec<String>,
	/// take out the thing at this place at this time (a false positive), as seconds,x,y with x and y
	/// 0-1 across the source; can be repeated
	#[arg(long = "drop", value_name = "T,X,Y")]
	drops: Vec<String>,
	/// leave this part of the picture alone, as x,y,w,h 0-1 across the source; add :only to follow
	/// nothing outside it instead; can be repeated
	#[arg(long = "zone", value_name = "X,Y,W,H[:only]")]
	zones: Vec<String>,
	/// cut this stretch out, as from-to in seconds of the source; can be repeated
	#[arg(long = "cut", value_name = "FROM-TO")]
	cuts: Vec<String>,
	/// draw nothing over this stretch, as from-to in seconds of the source; can be repeated
	#[arg(long = "bare", value_name = "FROM-TO")]
	bares: Vec<String>,
	/// leave the look mixed in out of this stretch (the look alone), as from-to in seconds of the
	/// source; can be repeated
	#[arg(long = "unmix", value_name = "FROM-TO")]
	unmixes: Vec<String>,
	/// a song under the video (a file, or a SoundCloud or YouTube link, fetched with yt-dlp), from the
	/// start (see --music-from); it runs on through cuts, and looks on the beat follow it
	#[arg(long)]
	music: Option<String>,
	/// seconds into the song where the video starts
	#[arg(long, default_value_t = 0.0)]
	music_from: f64,
	/// how loud the song is: 1 as it is
	#[arg(long, default_value_t = 1.0)]
	music_gain: f32,
	/// how much of the clip's own sound stays under the song: 0 none, 1 all
	#[arg(long, default_value_t = 0.0)]
	clip_sound: f32,
}

fn main() -> Result<()> {
	match Cli::parse().cmd {
		Cmd::Looks { name: None } => {
			for name in Look::presets() {
				println!("{name}");
			}
			for (name, _) in spotter_core::look::mine() {
				println!("{name}  (yours)");
			}
			Ok(())
		}
		Cmd::Looks { name: Some(name) } => {
			let look = Look::named(&name).with_context(|| format!("there's no look called {name}"))?;
			println!("{}", serde_json::to_string_pretty(&look)?);
			Ok(())
		}
		Cmd::Render(r) => render(r),
		Cmd::Files { words } => {
			let words: Vec<String> = words.iter().map(|w| w.to_lowercase()).collect();
			for c in catalog()? {
				let path = decode(&c.path).to_lowercase();
				if words.iter().all(|w| path.contains(w)) {
					let out =
						if LEFT_OUT.iter().any(|d| c.path.starts_with(d)) { "  (not picked at random)" } else { "" };
					println!("files:{}  {:.0} MB{out}", decode(&c.path), c.bytes as f64 / 1e6);
				}
			}
			Ok(())
		}
		Cmd::Model { cmd } => model(cmd),
		Cmd::Doctor => doctor(),
		Cmd::Setup => {
			use spotter_core::video::{FFMPEG_BYTES, fetch_ffmpeg, have_ffmpeg, tool};
			if !have_ffmpeg() {
				println!("getting ffmpeg ({:.0} MB)", FFMPEG_BYTES as f64 / 1e6);
				fetch_ffmpeg()?;
			}
			println!("ffmpeg: {}", tool("ffmpeg").display());
			println!("detector: {}", spotter_core::detector::dfine_in("x", "obj365")?.display());
			Ok(())
		}
	}
}

/// What spotter has to work with here. Fails if it couldn't read video or download anything.
fn doctor() -> Result<()> {
	use spotter_core::{detector, model, paths, video};
	let first_line = |program: &Path, arg: &str| -> Option<String> {
		let out = spotter_core::quiet(&mut std::process::Command::new(program)).arg(arg).output().ok()?;
		out.status
			.success()
			.then(|| String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or_default().to_string())
	};
	let line = |what: &str, said: &str| println!("{what:<10} {said}");
	line("spotter", env!("CARGO_PKG_VERSION"));
	line("folder", &paths::home().display().to_string());
	let mut missing = Vec::new();
	for t in ["ffmpeg", "ffprobe"] {
		let p = video::tool(t);
		match first_line(&p, "-version") {
			Some(v) => {
				let at = if p.is_absolute() { p.display().to_string() } else { "on the PATH".into() };
				line(t, &format!("{} ({at})", v.split_whitespace().nth(2).unwrap_or("?")))
			}
			None => {
				line(t, "missing: spotter setup gets it on Windows; elsewhere, a package manager");
				missing.push(t);
			}
		}
	}
	match first_line(Path::new("curl"), "--version") {
		Some(v) => line("curl", v.split_whitespace().take(2).collect::<Vec<_>>().join(" ").as_str()),
		None => {
			line("curl", "missing: nothing can be downloaded");
			missing.push("curl");
		}
	}
	let name = "x365";
	let device = gpu();
	let way = match device {
		Device::DirectMl => "DirectML",
		Device::WebGpu => "WebGPU",
		Device::Cpu => "the processor",
	};
	if detector::dfine_ready(name) {
		let (size, set) = detector::parse_model(name)?;
		let t = Instant::now();
		let found = detector::Detector::load(&detector::dfine_in(size, set)?, device)
			.and_then(|mut d| d.detect(&spotter_core::Frame::new(640, 360), 0.5, "everything").map(|_| d.ms));
		match found {
			Ok(ms) => line(
				"graphics",
				&format!("{way} works: d-fine x 365 set up in {:.1} s, {ms:.0} ms a look", t.elapsed().as_secs_f32()),
			),
			Err(e) => line("graphics", &format!("{way} failed ({e}); spotter will use the processor")),
		}
	} else {
		line("graphics", "not tried yet: there's no detector (spotter setup gets d-fine x 365, 252 MB)");
	}
	let got: Vec<&str> = model::KNOWN.iter().filter(|k| model::ready(k).exists()).map(|k| k.name).collect();
	let detectors = detector::SIZES
		.iter()
		.flat_map(|s| [s.to_string(), format!("{s}365")])
		.filter(|n| detector::dfine_ready(n))
		.collect::<Vec<_>>();
	line("detectors", &if detectors.is_empty() { "none yet".into() } else { detectors.join(" ") });
	let rest = if got.len() < model::KNOWN.len() { " (the rest come when a look needs them)" } else { "" };
	line("models", &format!("{} of {} downloaded{rest}", got.len(), model::KNOWN.len()));
	let py = model::python();
	line(
		"python",
		&match first_line(&py, "--version") {
			Some(v) => format!("{v} ({}), for half-precision copies", py.display()),
			None => "none, so the graphics card runs full-precision models (a little slower)".into(),
		},
	);
	if !missing.is_empty() {
		bail!("missing {}", missing.join(", "));
	}
	Ok(())
}

fn model(cmd: ModelCmd) -> Result<()> {
	use spotter_core::model::{CheckInput, KNOWN, check, fetch, known};
	match cmd {
		ModelCmd::List => {
			use spotter_core::detector::{SIZES, dfine_ready};
			let wide = KNOWN.iter().map(|k| k.licence.len()).max().unwrap_or(0);
			let line = |name: &str, licence: &str, what: &str, got: bool| {
				println!("{name:<24} {licence:<wide$}  {what}{}", if got { "" } else { "  (not downloaded)" });
			};
			for size in SIZES {
				for (name, kinds) in [(size.to_string(), "COCO's 80"), (format!("{size}365"), "Objects365's 365")] {
					line(&name, "Apache-2.0", &format!("D-FINE {size}: {kinds} kinds of thing"), dfine_ready(&name));
				}
			}
			for k in KNOWN {
				line(k.name, k.licence, k.what, spotter_core::model::ready(k).exists());
			}
		}
		ModelCmd::Fetch { name } => {
			let path = match known(&name) {
				Some(k) => fetch(k)?,
				None => {
					let (size, set) = spotter_core::detector::parse_model(&name)
						.with_context(|| format!("spotter doesn't know a model called {name}"))?;
					spotter_core::detector::dfine_in(size, set)?
				}
			};
			println!("{}", path.display());
		}
		ModelCmd::Patch { from, to } => {
			let p = spotter_core::onnx::patch(&from, &to)?;
			println!("{} matrix products made 2-D, allowzero turned off on {} Reshapes", p.matmuls, p.reshapes);
			if !p.unknown.is_empty() {
				println!(
					"{} matrix products left alone, the model not saying how many dimensions they take",
					p.unknown.len()
				);
			}
		}
		ModelCmd::Check { model, shapes, picture, imagenet, level, runs, device } => {
			let path = match known(&model) {
				Some(k) => fetch(k)?,
				None => PathBuf::from(&model),
			};
			let mut c = CheckInput { level, runs, imagenet, ..Default::default() };
			for s in &shapes {
				let (n, dims) = s.split_once('=').with_context(|| format!("--shape wants name=AxBxC, not {s}"))?;
				let dims = dims
					.split('x')
					.map(|d| d.parse::<i64>())
					.collect::<Result<Vec<_>, _>>()
					.with_context(|| format!("{dims} isn't a shape like 1x3x640x640"))?;
				c.shapes.push((n.to_string(), dims));
			}
			if let Some(p) = &picture {
				c.picture = Some(spotter_core::video::still(p)?);
			}
			let r = check(&path, device.parse()?, &c)?;
			for p in &r.inputs {
				println!("in   {:<34} {:<8} {:?}", p.name, p.kind, p.shape);
			}
			for o in &r.outputs {
				let verdict = if o.relative <= 1e-3 {
					"same"
				} else if o.p99 <= 1e-2 {
					"close"
				} else {
					"DIFFERENT"
				};
				println!(
					"out  {:<34} {:?}  largest difference {:.2e} of its size, 99% within {:.2e}, mean {:.2e}  {verdict}",
					o.name, o.shape, o.relative, o.p99, o.mean
				);
			}
			println!("processor {:.1} ms, {:?} {:.1} ms a run", r.cpu_ms, r.gpu, r.gpu_ms);
		}
	}
	Ok(())
}

fn look_for(r: &Render, project: Option<&Project>) -> Result<Look> {
	let mut look = match project {
		Some(p) => p.look(),
		None => Look::named(&r.look)
			.with_context(|| format!("there's no look called {}; `spotter looks` lists them", r.look))?,
	};
	if let Some(p) = &r.settings {
		let saved: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p)?)
			.with_context(|| format!("{} isn't JSON", p.display()))?;
		let mut merged = serde_json::to_value(&look)?;
		if let (Some(m), Some(s)) = (merged.as_object_mut(), saved.as_object()) {
			for (k, v) in s {
				m.insert(k.clone(), v.clone());
			}
		}
		look = Look::revive(&merged);
	}
	for s in &r.sets {
		let (k, v) = s.split_once('=').with_context(|| format!("--set wants name=value, not {s}"))?;
		let old = look.get(k).with_context(|| format!("there's no setting called {k}"))?;
		let value = match old {
			serde_json::Value::Number(_) => {
				serde_json::json!(v.parse::<f64>().with_context(|| format!("{k} wants a number"))?)
			}
			serde_json::Value::Bool(_) => serde_json::json!(matches!(v, "1" | "true" | "on" | "yes")),
			// a setting made of settings (mixSet): as JSON
			serde_json::Value::Object(_) => {
				serde_json::from_str(v).with_context(|| format!("{k} wants JSON, like {{\"insetZoom\":2}}"))?
			}
			_ => serde_json::json!(v.replace("\\n", "\n")),
		};
		if !look.set(k, value) {
			bail!("couldn't set {k} to {v}");
		}
	}
	Ok(look)
}

fn numbers(s: &str, n: usize, what: &str) -> Result<Vec<f64>> {
	let v = s
		.split([',', '-'])
		.map(|x| x.trim().parse::<f64>())
		.collect::<Result<Vec<_>, _>>()
		.with_context(|| format!("{what} wants numbers, not {s}"))?;
	if v.len() != n {
		bail!("{what} wants {n} numbers, not {s}");
	}
	Ok(v)
}

fn edit_for(r: &Render, project: Option<&Project>) -> Result<Edit> {
	let mut e: Edit = match (&r.edit, project) {
		(Some(p), _) => serde_json::from_str(&std::fs::read_to_string(p)?)
			.with_context(|| format!("{} isn't an edit", p.display()))?,
		(None, Some(p)) => p.edit.clone(),
		(None, None) => Edit::default(),
	};
	e.kinds.extend(r.kinds.iter().map(|k| k.trim().to_string()).filter(|k| !k.is_empty()));
	for p in &r.picks {
		let v = numbers(p, 3, "--pick")?;
		e.picks.push(Pick { t: v[0], x: v[1] as f32, y: v[2] as f32 });
		e.keep = Keep::Picked;
	}
	for p in &r.drops {
		let v = numbers(p, 3, "--drop")?;
		e.drops.push(Pick { t: v[0], x: v[1] as f32, y: v[2] as f32 });
	}
	if let Some(k) = &r.keep {
		e.keep = serde_json::from_value(serde_json::json!(k))
			.with_context(|| format!("--keep is all, picked or main, not {k}"))?;
	}
	for z in &r.zones {
		let (rect, only) = match z.strip_suffix(":only") {
			Some(rest) => (rest, true),
			None => (z.as_str(), false),
		};
		let v: Vec<f64> = rect
			.split(',')
			.map(|x| x.trim().parse::<f64>())
			.collect::<Result<Vec<_>, _>>()
			.with_context(|| format!("--zone wants x,y,w,h, not {z}"))?;
		if v.len() != 4 {
			bail!("--zone wants x,y,w,h, not {z}");
		}
		e.zones.push(Zone {
			x: v[0] as f32,
			y: v[1] as f32,
			w: v[2] as f32,
			h: v[3] as f32,
			mode: if only { ZoneMode::Only } else { ZoneMode::Ignore },
		});
	}
	for (list, mode) in [(&r.cuts, RangeMode::Cut), (&r.bares, RangeMode::Bare), (&r.unmixes, RangeMode::Unmix)] {
		for c in list {
			let (a, b) = c.split_once('-').with_context(|| format!("a stretch is from-to in seconds, not {c}"))?;
			e.add_range(a.trim().parse()?, b.trim().parse()?, mode);
		}
	}
	if let Some(m) = &r.music {
		// a SoundCloud or YouTube link: its sound, fetched once
		let path = if spotter_core::songs::is_link(m) {
			eprintln!("getting the song at {m}");
			let p = spotter_core::songs::fetch(m, Default::default())?;
			p.to_string_lossy().into_owned()
		} else {
			m.clone()
		};
		e.music = Some(spotter_core::edit::Music { path, from: r.music_from, gain: r.music_gain, keep: r.clip_sound });
	}
	Ok(e)
}

fn ratio(s: &str) -> Result<f64> {
	let (a, b) = s.split_once([':', '/', 'x']).with_context(|| format!("{s} isn't a shape like 9:16"))?;
	let (a, b): (f64, f64) = (a.trim().parse()?, b.trim().parse()?);
	if a <= 0.0 || b <= 0.0 {
		bail!("{s} isn't a shape like 9:16");
	}
	Ok(a / b)
}

fn render(r: Render) -> Result<()> {
	// a project saved from the window: its source, look, edit and export, which the flags change
	let project = if r.input.ends_with(&format!(".{}", spotter_core::project::EXTENSION)) {
		Some(Project::read(Path::new(&r.input))?)
	} else {
		None
	};
	let look = look_for(&r, project.as_ref())?;
	let (input, mut name, clip) = resolve(project.as_ref().map_or(&r.input, |p| &p.source))?;
	if let Some(p) = project.as_ref().filter(|p| !p.name.is_empty()) {
		name = p.name.clone();
	}
	let remote = clip.is_some() || input.starts_with("http");
	if remote {
		eprintln!("reading {input}");
	}
	let size = match (&r.shape, &r.size) {
		(Some(name), _) => Some(
			spotter_core::export::shape(name)
				.with_context(|| format!("there's no shape called {name}: reel, post, square or wide"))?,
		),
		(None, s) => match s {
			Some(s) => {
				let (a, b) = s.split_once('x').with_context(|| format!("--size wants WxH, not {s}"))?;
				Some((a.parse::<u32>()?, b.parse::<u32>()?))
			}
			None => project.as_ref().and_then(|p| p.size),
		},
	};
	let (mut start, mut length) = (r.start, r.length);
	if let Some(p) = &project
		&& start == 0.0
		&& length.is_none()
	{
		start = p.start.unwrap_or(0.0);
		length = p.end.map(|e| (e - start).max(0.1));
	}
	// a long video picked at random starts somewhere in the middle, as on the site, and runs 20 seconds
	if r.input == "random" && clip.as_ref().is_some_and(|c| c.wanders()) {
		let duration = probe(Path::new(&input))?.duration;
		let l = length.unwrap_or(20.0).min(duration);
		if start == 0.0 {
			start = Dice::default().unit() * (duration - l).max(0.0);
		}
		length = Some(l);
	}
	let device = match r.device.as_str() {
		"gpu" | "off" => gpu(),
		other => other.parse::<Device>()?,
	};
	let look_name = project.as_ref().map_or(r.look.clone(), |p| p.preset.clone());
	let out = r.output.clone().unwrap_or_else(|| default_out(Path::new(&input), &name, &look_name, remote));
	let model = match &project {
		Some(p) if r.model == "x365" && !p.model.is_empty() => p.model.clone(),
		_ => r.model.clone(),
	};
	let e = Export {
		input: input.clone(),
		name: name.clone(),
		look,
		start,
		length,
		aspect: r.aspect.as_deref().map(ratio).transpose()?,
		size,
		fps: r.fps,
		model: (r.device != "off").then_some(model),
		device,
		crf: r.crf,
		out: out.clone(),
		edit: edit_for(&r, project.as_ref())?,
		upscale: r.upscale || project.as_ref().is_some_and(|p| p.upscale),
		record: None,
		fast: r.fast,
		still: r.still.is_some(),
	};
	// a still: from two seconds before it to it, and a PNG unless told otherwise
	let mut e = e;
	if let Some(t) = r.still {
		let rate = probe(Path::new(&input))?.fps.max(1.0);
		e.start = (t - 2.0).max(0.0);
		e.length = Some(t - e.start + 1.0 / rate);
		if r.output.is_none() {
			e.out = e.out.with_extension("png");
		}
	}
	let p = plan(&e, &probe(Path::new(&input))?)?;
	eprintln!(
		"{name} from {start:.1}s → {} at {}x{}, {:.2} fps, {:.1}s, {} frames, look {}",
		out.display(),
		p.w,
		p.h,
		p.fps,
		p.length / p.speed,
		p.frames,
		look_name
	);
	let began = Instant::now();
	let mut said = Instant::now();
	let p = run(&e, &mut |n, total| {
		if said.elapsed().as_secs_f32() >= 1.0 {
			said = Instant::now();
			eprint!("\r  {n}/{total} frames, {:.1} a second   ", n as f32 / began.elapsed().as_secs_f32());
		}
		true
	})?;
	let secs = began.elapsed().as_secs_f32();
	eprintln!("\r  {} frames in {secs:.1}s, {:.1} a second. Wrote {}", p.frames, p.frames as f32 / secs, out.display());
	Ok(())
}

/// Next to the input when it's on this computer, otherwise in the folder spotter was run from.
fn default_out(input: &Path, name: &str, look: &str, remote: bool) -> PathBuf {
	let file = format!("{}-{}.mp4", name.replace(['/', '\\', ':'], "-"), look.replace(' ', "-"));
	if remote { PathBuf::from(file) } else { input.with_file_name(file) }
}
