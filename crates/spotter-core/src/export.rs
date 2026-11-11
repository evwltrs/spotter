//! Rendering a whole video to a file: every frame read, tracked, recognised where the look asks for it,
//! drawn and written, in turn, with the sound carried across. The command line and the app both use it.

use crate::Engine;
use crate::Frame;
use crate::detector::{Detector, Device, dfine_in, parse_model};
use crate::edit::{Edit, Keep, View, WHOLE};
use crate::look::Look;
use crate::tape::{Looker, Record, Tape};
use crate::upscale::Upscaler;
use crate::video::{Decoder, Encoder, Input, Output, Probe, probe};
use anyhow::{Result, bail};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What to render, and how.
#[derive(Clone, Debug)]
pub struct Export {
	/// what ffmpeg reads: a file or a web address
	pub input: String,
	/// shown in the {file} token
	pub name: String,
	pub look: Look,
	/// seconds into the source
	pub start: f64,
	/// seconds to render; None is to the end
	pub length: Option<f64>,
	/// cut the picture to this shape (width over height) from the middle
	pub aspect: Option<f64>,
	/// the exact size to write; otherwise the source's (cut to shape), no longer than the look's longest side
	pub size: Option<(u32, u32)>,
	/// frames a second to write; otherwise the source's, up to 60
	pub fps: Option<f64>,
	/// D-FINE's size (s, m, l or x, with 365 after it for Objects365's kinds), or None for movement alone
	pub model: Option<String>,
	pub device: Device,
	/// x264 quality: 18 looks lossless
	pub crf: u32,
	pub out: PathBuf,
	/// what's followed, where, and which stretches are cut or left bare
	pub edit: Edit,
	/// make a picture smaller than the output bigger with Real-ESRGAN rather than by stretching it;
	/// with no `size`, the output then grows to the look's longest side
	pub upscale: bool,
	/// what the detector saw over the whole clip, if it's been looked at already (the window does, as
	/// it plays): used instead of looking again when it was looked at frame for frame as this export reads
	pub record: Option<Arc<Record>>,
	/// use the window's quicker models (EdgeTAM, Depth Anything small, SegFormer B2) rather than the
	/// best ones
	pub fast: bool,
	/// a still: only the last frame is kept, as a PNG at `out` (the frames before it give trails,
	/// copies and outlines their history)
	pub still: bool,
}

/// Shapes and sizes made for where videos are posted.
pub const SHAPES: [(&str, u32, u32, &str); 4] = [
	("reel", 1080, 1920, "9:16, for reels, stories and shorts"),
	("post", 1080, 1350, "4:5, for a post in a feed"),
	("square", 1080, 1080, "1:1"),
	("wide", 1920, 1080, "16:9, for X and anywhere landscape"),
];

/// A shape by name: its width and height.
pub fn shape(name: &str) -> Option<(u32, u32)> {
	SHAPES.iter().find(|s| s.0 == name).map(|s| (s.1, s.2))
}

/// The sizes and timings an export works out to.
#[derive(Clone, Debug)]
pub struct Plan {
	/// the size written
	pub w: u32,
	pub h: u32,
	/// the size read from the source: smaller than the size written when it's made bigger
	pub dw: u32,
	pub dh: u32,
	/// made bigger by the upscaler
	pub upscale: bool,
	pub crop: Option<[u32; 4]>,
	pub fps: f64,
	pub speed: f64,
	/// seconds of the source, cuts taken out
	pub length: f64,
	/// the stretches of the source that are kept, in source seconds
	pub stretches: Vec<(f64, f64)>,
	pub frames: u64,
	pub audio: bool,
	/// where the picture sits in the source, 0-1 across it
	pub view: View,
}

/// The middle part of a w by h picture that has the shape `want` (width over height), in even numbers.
pub fn cut(w: u32, h: u32, want: f64) -> [u32; 4] {
	let have = w as f64 / h as f64;
	let even = |v: f64| ((v / 2.0).floor() as u32 * 2).max(2);
	if have > want {
		let cw = even(h as f64 * want).min(w);
		[(w - cw) / 2, 0, cw, h]
	} else {
		let ch = even(w as f64 / want).min(h);
		[0, (h - ch) / 2, w, ch]
	}
}

/// The largest even size no bigger than (w, h) whose longest side is at most `most`.
pub fn fit(w: u32, h: u32, most: f64) -> (u32, u32) {
	let scale = (most / w.max(h) as f64).min(1.0);
	(((w as f64 * scale / 2.0).round() as u32 * 2).max(2), ((h as f64 * scale / 2.0).round() as u32 * 2).max(2))
}

pub fn plan(e: &Export, src: &Probe) -> Result<Plan> {
	let shape = e.aspect.or(e.size.map(|(w, h)| w as f64 / h as f64));
	let crop = shape.map(|want| cut(src.w, src.h, want));
	let (cw, ch) = crop.map_or((src.w, src.h), |c| (c[2], c[3]));
	let (w, h) = e.size.map(|(w, h)| (w / 2 * 2, h / 2 * 2)).unwrap_or_else(|| {
		if e.upscale { fit(cw * 4, ch * 4, e.look.out_max as f64) } else { fit(cw, ch, e.look.out_max as f64) }
	});
	// read at the source's own size and let the upscaler do the rest, when there's a rest to do
	let upscale = e.upscale && (cw < w || ch < h);
	let (dw, dh) = if upscale { (cw / 2 * 2, ch / 2 * 2) } else { (w, h) };
	let fps = e.fps.unwrap_or(src.fps.min(60.0));
	let speed = (e.look.speed as f64).clamp(0.05, 4.0);
	let left = (src.duration - e.start).max(0.0);
	let length = e.length.map_or(left, |l| l.min(left));
	if length <= 0.0 {
		bail!("{} is only {:.1}s long", e.name, src.duration);
	}
	let stretches = e.edit.stretches(e.start, e.start + length);
	if stretches.is_empty() {
		bail!("the cuts take out all of it");
	}
	let length = stretches.iter().map(|(a, b)| b - a).sum::<f64>();
	let frames = stretches.iter().map(|(a, b)| stretch_frames(b - a, speed, fps)).sum::<u64>().max(1);
	let view = crop.map_or(WHOLE, |c| {
		[c[0] as f32 / src.w as f32, c[1] as f32 / src.h as f32, c[2] as f32 / src.w as f32, c[3] as f32 / src.h as f32]
	});
	Ok(Plan { w, h, dw, dh, upscale, crop, fps, speed, length, stretches, frames, audio: src.audio, view })
}

fn stretch_frames(seconds: f64, speed: f64, fps: f64) -> u64 {
	(seconds / speed * fps).round().max(1.0) as u64
}

/// D-FINE at a size on a device, falling back to the CPU if the GPU can't run it.
/// Says which it ended up on.
pub fn load_detector(name: &str, device: Device) -> Result<(Detector, Device, Option<String>)> {
	let (size, set) = parse_model(name)?;
	let model = dfine_in(size, set)?;
	match Detector::load(&model, device) {
		Ok(d) => Ok((d, device, None)),
		Err(e) if device != Device::Cpu => {
			let why =
				format!("the graphics card couldn't run the detector ({e}); using the processor, which is much slower");
			Ok((Detector::load(&model, Device::Cpu)?, Device::Cpu, Some(why)))
		}
		Err(e) => Err(e),
	}
}

/// The graphics card's way in on this computer.
pub fn gpu() -> Device {
	if cfg!(windows) { Device::DirectMl } else { Device::WebGpu }
}

/// Render it. `progress` hears (frames done, frames in all) as it goes and can return false to stop,
/// in which case the half-written file is removed.
pub fn run(e: &Export, progress: &mut dyn FnMut(u64, u64) -> bool) -> Result<Plan> {
	let src = probe(Path::new(&e.input))?;
	let p = plan(e, &src)?;
	let input = PathBuf::from(&e.input);
	let follow = e.look.follow;
	// looked at already, by the same detector, frame for frame as this reads it
	let looked = e.record.clone().filter(|r| {
		e.look.detect
			&& e.look.ahead
			&& e.model.as_deref() == Some(r.model.as_str())
			&& (r.fps - p.fps / p.speed).abs() < 1e-3
			&& !r.frames.is_empty()
	});
	let mut detector = None;
	if e.look.detect
		&& looked.is_none()
		&& let Some(size) = &e.model
	{
		let (d, _, warning) = load_detector(size, e.device)?;
		if let Some(w) = warning {
			eprintln!("{w}");
		}
		detector = Some(d);
	}
	// looking ahead: the detector goes over the whole clip first, then takes no part in the drawing
	let ahead = e.look.ahead && detector.is_some();
	let mut tape: Option<Arc<Tape>> = looked.map(|r| Arc::new(Tape::build(&r, &e.look, &e.edit)));
	if let Some(t) = &tape
		&& std::env::var_os("SPOTTER_TIMES").is_some()
	{
		eprintln!(
			"looked ahead already: {}, {}",
			crate::counted(t.times.len(), "frame"),
			crate::counted(t.things.len(), "thing")
		);
	}
	let fresh = |detector: Option<Detector>, tape: &Option<Arc<Tape>>| {
		let mut engine = Engine::new(e.look.clone());
		engine.name = e.name.clone();
		engine.fps = p.fps;
		engine.edit = e.edit.clone();
		engine.view = p.view;
		engine.detector = detector;
		engine.set_tape(tape.clone());
		engine
	};
	// with the follow cam, the whole picture is read and the camera's window cut from it
	let (rw, rh) = if follow { fit(src.w, src.h, 2560.0) } else { (p.dw, p.dh) };
	// the models a look wants: the best of each (the window's quicker ones if those won't load), or
	// the quicker ones throughout when `fast`
	fn best<T>(what: &str, fast: bool, mut load: impl FnMut(usize) -> Result<T>) -> Option<T> {
		if !fast {
			match load(0) {
				Ok(m) => return Some(m),
				Err(err) => eprintln!("the best {what} wouldn't load ({err}); using the quicker one"),
			}
		}
		match load(1) {
			Ok(m) => Some(m),
			Err(err) => {
				eprintln!("no {what} ({err})");
				None
			}
		}
	}
	let gpu_on = e.device != Device::Cpu;
	// outlines, for layers that want things' true shapes (otherwise they're cut out by colour)
	let mut outliner = (e.look.either(crate::matte::wants_outlines) && gpu_on)
		.then(|| {
			best("outlines", e.fast, |i| crate::outline::Outliner::load(crate::outline::MODELS[i], e.device, true))
		})
		.flatten();
	let mut depther = (e.look.either(crate::depth::active) && gpu_on)
		.then(|| best("depth", e.fast, |i| crate::depth::Depther::load(crate::depth::MODELS[i], e.device, true)))
		.flatten();
	// SegFormer, for the words look's world
	let mut scener = (e.look.either(crate::scene::active) && gpu_on)
		.then(|| best("scene", e.fast, |i| crate::scene::Scener::load(crate::scene::MODELS[i], e.device, true)))
		.flatten();
	let open_at = |a: f64, b: f64, w: u32, h: u32, crop: Option<[u32; 4]>| {
		Decoder::open(
			&input,
			&Input { start: a, length: Some(b - a), crop, w, h, fps: p.fps / p.speed, colour: src.colour.clone() },
		)
	};
	let open = |a: f64, b: f64| open_at(a, b, rw, rh, if follow { None } else { p.crop });
	// slowed down: the source's own frames, and the ones between made by RIFE
	let mut tween = if e.look.smooth_slow && p.speed < 0.99 && e.device != Device::Cpu {
		match crate::interp::Interpolator::load(e.device) {
			Ok(i) => Some(i),
			Err(err) => {
				eprintln!("no smooth slow motion ({err}); each frame is shown again instead");
				None
			}
		}
	} else {
		None
	};
	let native = |a: f64, b: f64| {
		Decoder::open(
			&input,
			&Input {
				start: a,
				length: Some(b - a),
				crop: if follow { None } else { p.crop },
				w: rw,
				h: rh,
				fps: src.fps,
				colour: src.colour.clone(),
			},
		)
	};
	let mut frame = Frame::default();
	let mut upscaler = if p.upscale { Some(Upscaler::load(e.device)?) } else { None };
	let mut big = Frame::default();
	// the frame as the engine gets it: read, and made bigger if it's being
	let mut prepare = |frame: &Frame, big: &mut Frame| -> Result<bool> {
		match upscaler.as_mut() {
			Some(u) => {
				u.run(frame, p.w, p.h, big)?;
				Ok(true)
			}
			None => Ok(false),
		}
	};

	let total = p.frames * (1 + follow as u64 + ahead as u64);
	// frames done in the passes before this one
	let mut base = 0u64;
	if ahead && let Some(d) = detector.as_mut() {
		// what the detector looks at: the export's own view (smaller things show up better in a cut of
		// the picture than in all of it), or all of it for a follow cam, whose view moves; big enough
		// for the detector, no bigger
		let (view, crop, cw, ch) = match p.crop {
			Some(c) if !follow => (p.view, Some(c), c[2], c[3]),
			_ => (WHOLE, None, src.w, src.h),
		};
		let k = (640.0 / cw.min(ch) as f64).min(1.0);
		let even = |v: f64| ((v / 2.0).round() as u32 * 2).max(2);
		let (aw, ah) = (even(cw as f64 * k), even(ch as f64 * k));
		let mut rec = Record {
			frames: Vec::with_capacity(p.frames as usize),
			fps: p.fps / p.speed,
			aspect: src.w as f32 / src.h as f32,
			view,
			model: e.model.clone().unwrap_or_default(),
		};
		let mut looker = Looker::default();
		for &(a, b) in &p.stretches {
			let mut dec = open_at(a, b, aw, ah, crop)?;
			let want = stretch_frames(b - a, p.speed, p.fps);
			let mut n = 0u64;
			let mut lap = crate::perf::Lap::start();
			while n < want && dec.next(&mut frame)? {
				lap.mark("ahead: decode");
				rec.frames.push(looker.look(d, &frame, a + n as f64 * p.speed / p.fps, view, n == 0)?);
				lap.mark("ahead: detector");
				n += 1;
				if !progress(base + rec.frames.len() as u64, total) {
					bail!("stopped");
				}
			}
		}
		base += p.frames;
		let t = Tape::build(&rec, &e.look, &e.edit);
		if std::env::var_os("SPOTTER_TIMES").is_some() {
			let short: Vec<String> = t
				.things
				.iter()
				.filter(|th| (th.boxes.len() as f64) < 0.4 * rec.fps)
				.map(|th| format!("{} {:.2}s {}f", th.cls, t.times[th.first], th.boxes.len()))
				.collect();
			eprintln!(
				"looked ahead: {}, {}, {}; short ones: {}",
				crate::counted(rec.frames.len(), "frame"),
				crate::counted(t.cuts.iter().filter(|&&c| c).count(), "shot"),
				crate::counted(t.things.len(), "thing"),
				short.join(", ")
			);
		}
		tape = Some(Arc::new(t));
		// the tape says what's where from here on
		detector = None;
	}

	// follow cam: first, where the subject is on every frame, from a small copy of the whole picture;
	// then the camera's path through the clip, smoothed both ways in time
	let mut camera: Vec<crate::follow::Window> = Vec::new();
	if follow {
		let (sw, sh) = fit(src.w, src.h, 640.0);
		let mut engine = fresh(detector.take(), &tape);
		engine.view = WHOLE;
		if engine.edit.keep == Keep::All {
			engine.edit.keep = Keep::Main;
		}
		let mut seen: Vec<Option<crate::Rect>> = Vec::with_capacity(p.frames as usize);
		let mut cuts: Vec<bool> = Vec::with_capacity(p.frames as usize);
		for (i, &(a, b)) in p.stretches.iter().enumerate() {
			if i > 0 {
				engine.jumped();
			}
			let mut dec = open_at(a, b, sw, sh, None)?;
			let want = stretch_frames(b - a, p.speed, p.fps);
			let mut n = 0u64;
			while n < want && dec.next(&mut frame)? {
				let t = a + n as f64 * p.speed / p.fps;
				engine.follow_only(&frame, t, 1.0 / p.fps)?;
				seen.push(engine.target().map(|r| crate::Rect {
					x: r.x * src.w as f32,
					y: r.y * src.h as f32,
					w: r.w * src.w as f32,
					h: r.h * src.h as f32,
				}));
				cuts.push(n == 0 || tape.as_ref().is_some_and(|tp| tp.at(t).is_some_and(|k| tp.cuts[k])));
				n += 1;
				if !progress(base + seen.len() as u64, total) {
					bail!("stopped");
				}
			}
			while n < want {
				seen.push(seen.last().copied().flatten());
				cuts.push(n == 0);
				n += 1;
			}
		}
		camera = crate::follow::path(&seen, &cuts, src.w as f32, src.h as f32, p.w as f32 / p.h as f32, &e.look, p.fps);
		detector = engine.detector.take();
		base += p.frames;
	}

	// a pick is a time and a place, and which thing it means is only known by following the clip up to
	// it: following comes out the same each time, so a pass up to the last pick finds them (the follow
	// cam's pass, or a tape, knows them already)
	let mut picked = Vec::new();
	if !follow && tape.is_none() && e.edit.keep == Keep::Picked && !e.edit.picks.is_empty() {
		let last = e.edit.picks.iter().map(|k| k.t).fold(f64::MIN, f64::max) + 1.2;
		let mut engine = fresh(detector.take(), &tape);
		let mut first = true;
		'pass: for &(a, b) in p.stretches.iter().filter(|s| s.0 < last) {
			if !first {
				engine.jumped();
			}
			first = false;
			let mut dec = open(a, b)?;
			let mut n = 0u64;
			while dec.next(&mut frame)? {
				let t = a + n as f64 * p.speed / p.fps;
				if t > last {
					break 'pass;
				}
				let f = if prepare(&frame, &mut big)? { &big } else { &frame };
				engine.follow_only(f, t, 1.0 / p.fps)?;
				n += 1;
			}
		}
		picked = engine.chooser.found();
		detector = engine.detector.take();
	}

	let mut engine = fresh(detector, &tape);
	// on the beat: the song's, where there's one, else the clip's own
	let end = p.stretches.last().map_or(e.start, |s| s.1);
	if e.look.beat {
		match &e.edit.music {
			Some(m) => match crate::beats::find(&m.path) {
				Ok(b) => engine.set_beats(Some(Arc::new(m.beats_in_source(&b, &e.edit, e.start, end, p.speed)))),
				Err(err) => eprintln!("no beats in {} ({err})", m.name()),
			},
			None if p.audio => match crate::beats::find(&e.input) {
				Ok(b) => engine.set_beats(Some(Arc::new(b))),
				Err(err) => eprintln!("no beats ({err})"),
			},
			None => {}
		}
	}
	// the real datamosh: the clip from the start to the end re-encoded, its frames rearranged (on the
	// beats when it's asked to be and there are any, else on its cuts), and read in its place
	let moshed = match e.look.real_mosh() {
		Some(beats) => {
			eprintln!("moshing the clip (re-encoding it and rearranging its frames)");
			let when = match engine.beats() {
				Some(b) if beats && !b.is_empty() => crate::realmosh::When::Beats(b.to_vec()),
				_ => crate::realmosh::When::Cuts,
			};
			let inp = Input {
				start: e.start,
				length: Some(end - e.start),
				crop: if follow { None } else { p.crop },
				w: rw,
				h: rh,
				fps: src.fps,
				colour: src.colour.clone(),
			};
			Some(crate::realmosh::prepare(&input, &inp, &when, e.look.mosh_hold as f64)?)
		}
		None => None,
	};
	engine.chooser.ahead = picked;
	engine.outliner = outliner.take();
	engine.depther = depther.take();
	engine.scener = scener.take();
	// two looks mixed: the models' answers for one are kept for the other on the same frame
	if !e.look.mix.trim().is_empty() {
		engine.recall = Some(crate::recall::Recall::small());
	}
	// following picked things, the camera keeps them in the middle, so that's what to draw
	if follow && engine.edit.keep == Keep::Picked {
		engine.edit.keep = Keep::Centre;
	}
	let mut window = Frame::new(p.dw, p.dh);
	let mut done = base;
	let mut shown = 0usize;
	let keep: Vec<(f64, f64)> = p.stretches.iter().map(|(a, b)| (a - e.start, b - e.start)).collect();
	let sound = p.audio.then(|| (input.clone(), e.start, Some(keep.last().map_or(0.0, |k| k.1))));
	let max_mbps = (e.look.bitrate > 0.0).then_some(e.look.bitrate as f64);
	let music = e.edit.music.as_ref().map(|m| crate::video::Song {
		path: m.path.clone().into(),
		from: m.from,
		length: p.frames as f64 / p.fps,
		gain: m.gain,
		keep: m.keep,
	});
	let mut enc = if e.still {
		None
	} else {
		Some(Encoder::create(
			&e.out,
			&Output { w: p.w, h: p.h, fps: p.fps, crf: e.crf, max_mbps, sound, keep, speed: p.speed, music },
		)?)
	};
	let mut out = Frame::default();
	for (i, &(a, b)) in p.stretches.iter().enumerate() {
		if i > 0 {
			engine.jumped();
		}
		let mut dec = match &moshed {
			// the real datamosh: the moshed copy, already cut to shape
			Some(m) => Decoder::open(
				m,
				&Input {
					start: a - e.start,
					length: Some(b - a),
					crop: None,
					w: rw,
					h: rh,
					fps: if tween.is_some() { src.fps } else { p.fps / p.speed },
					colour: Default::default(),
				},
			)?,
			None if tween.is_some() => native(a, b)?,
			None => open(a, b)?,
		};
		// exactly as many frames as the sound has room for, so cuts don't drift apart
		let want = stretch_frames(b - a, p.speed, p.fps);
		let mut n = 0u64;
		let mut lap = crate::perf::Lap::start();
		// with RIFE: the two source frames either side of this moment, and which source frame the first is
		let (mut f0, mut f1, mut at0) = (Frame::default(), Frame::default(), 0i64);
		if tween.is_some() {
			if !dec.next(&mut f0)? {
				continue;
			}
			if !dec.next(&mut f1)? {
				f1.clone_from(&f0);
			}
		}
		loop {
			if n >= want {
				break;
			}
			let t = a + n as f64 * p.speed / p.fps;
			// the source frame (and how far on to the next) this moment falls on
			let between = if tween.is_some() {
				let pos = (t - a) * src.fps;
				while pos >= (at0 + 1) as f64 {
					std::mem::swap(&mut f0, &mut f1);
					at0 += 1;
					if !dec.next(&mut f1)? {
						f1.clone_from(&f0);
						break;
					}
				}
				Some(((pos - at0 as f64) as f32).clamp(0.0, 1.0))
			} else {
				if !dec.next(&mut frame)? {
					break;
				}
				None
			};
			lap.mark("decode");
			let win = follow.then(|| {
				let win = camera[shown.min(camera.len() - 1)];
				let k = rw as f32 / src.w as f32;
				engine.view =
					[win[0] / src.w as f32, win[1] / src.h as f32, win[2] / src.w as f32, win[3] / src.h as f32];
				[win[0] * k, win[1] * k, win[2] * k, win[3] * k]
			});
			let read: &Frame = match (between, win) {
				// the camera's window from both frames, then the moment between
				(Some(k), Some(w)) => {
					f0.crop_into(w, &mut window);
					let mut w1 = Frame::new(window.w, window.h);
					f1.crop_into(w, &mut w1);
					let w0 = window.clone();
					tween.as_mut().unwrap().between(&w0, &w1, k, &mut window)?;
					&window
				}
				(Some(k), None) => {
					tween.as_mut().unwrap().between(&f0, &f1, k, &mut frame)?;
					&frame
				}
				(None, Some(w)) => {
					frame.crop_into(w, &mut window);
					&window
				}
				(None, None) => &frame,
			};
			lap.mark("in between");
			let f = if prepare(read, &mut big)? { &big } else { read };
			lap.mark("upscale");
			engine.step(f, t, 1.0 / p.fps, &mut out)?;
			lap.mark("(engine)");
			if let Some(enc) = enc.as_mut() {
				enc.write(&out)?;
			}
			lap.mark("encode");
			n += 1;
			done += 1;
			shown += 1;
			if !progress(done, total) {
				drop(enc);
				let _ = std::fs::remove_file(&e.out);
				bail!("stopped");
			}
		}
		// a stretch the decoder ran short on is padded with its last frame
		while n < want && out.w > 0 {
			if let Some(enc) = enc.as_mut() {
				enc.write(&out)?;
			}
			n += 1;
			done += 1;
			shown += 1;
		}
	}
	match enc {
		Some(enc) => enc.finish()?,
		None => write_png(&e.out, &out)?,
	}
	eprint!("{}", crate::perf::report(p.frames));
	Ok(p)
}

/// A frame written as a PNG (by ffmpeg, which every export needs anyway).
pub fn write_png(path: &Path, f: &Frame) -> Result<()> {
	if f.w == 0 {
		bail!("there's no frame to write");
	}
	let mut c = crate::video::ffmpeg();
	c.args([
		"-v",
		"error",
		"-nostdin",
		"-y",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"rgba",
		"-s",
		&format!("{}x{}", f.w, f.h),
		"-i",
		"-",
		"-frames:v",
		"1",
	])
	.arg(path);
	let mut child = c
		.stdin(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped())
		.spawn()
		.map_err(|e| anyhow::anyhow!("couldn't run ffmpeg to write the still: {e}"))?;
	{
		use std::io::Write;
		let mut pipe = child.stdin.take().unwrap();
		pipe.write_all(&f.rgba)?;
	}
	let done = child.wait_with_output()?;
	if !done.status.success() {
		bail!("writing {} failed: {}", path.display(), String::from_utf8_lossy(&done.stderr).trim());
	}
	Ok(())
}
