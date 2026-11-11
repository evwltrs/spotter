//! The thread that plays the video: it reads frames with ffmpeg, runs each through the engine and
//! hands the result to the window's surface, keeping to the video's own pace. The window talks to it
//! with commands and reads back a status. Exports run on threads of their own.
//!
//! Opening a video and seeking in one can take seconds when it's on files.1nf.me, so those happen on
//! a thread of their own too: what's playing carries on until the new picture is ready, and if
//! another is asked for meanwhile, only the newest counts.
//!
//! Once a video's open, a detector of its own goes over the whole of it on another thread (looking
//! ahead, tape.rs). Wherever it has been, the preview draws what it found, worked out with hindsight,
//! as an export does; beyond that, the live detector carries on as the site does.

use rune::NativeSurface;
use rune::native_surface::{ByteFrame, ColorInfo, Format};
use spotter_core::edit::{Edit, Pick, View, WHOLE, Zone, ZoneMode, to_view};
use spotter_core::engine::LiveDetector;
use spotter_core::export::{self, Export, cut, fit, gpu, load_detector};
use spotter_core::follow::{Spring, Window, widest};
use spotter_core::sources::{Dice, resolve};
use spotter_core::tape::{Looker, Record, Seen, Tape};
use spotter_core::video::{Decoder, Input, Probe, probe};
use spotter_core::{Engine, Frame, Look, counted};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// The longest side the preview is worked out at. Exports use the full size.
const PREVIEW: f64 = 1280.0;

// sent at a person's pace, so a big look in one of them costs nothing worth boxing
#[allow(clippy::large_enum_variant)]
pub enum Cmd {
	/// a file, a web address, files:<path>, or random
	Open(String),
	Toggle,
	Seek(f64),
	Look(Look),
	/// cut the preview to this shape, as the export will be
	Aspect(Option<f64>),
	/// export these, one after another (a reel and a wide one of the same, say)
	Export(Vec<Export>),
	StopExport,
	/// what's followed, where and when
	Edit(Edit),
	/// a zone being dragged out, shown until it's let go
	Draft(Option<Zone>),
	/// draw the edit's zones and picks over the preview
	Guides(bool),
	/// change the detector (a size, with 365 after it for Objects365's kinds)
	Detector(String),
	/// the preview's sound off or on
	Mute(bool),
	/// the in point, where a song under the video starts
	In(Option<f64>),
	/// how loud the preview plays (1 as it is)
	Volume(f32),
	/// show the picture as it is, with nothing drawn over it (comparing), or go back to the look
	Bare(bool),
}

#[derive(Clone, Default, PartialEq)]
pub struct Status {
	pub name: String,
	/// what was opened, as ffmpeg reads it
	pub input: String,
	pub duration: f64,
	/// the clip's frames a second, for stepping a frame at a time
	pub rate: f64,
	pub t: f64,
	pub playing: bool,
	pub opening: bool,
	/// the preview's size
	pub w: u32,
	pub h: u32,
	/// frames a second the preview is managing
	pub fps: u32,
	/// what the detector is up to
	pub detector: String,
	/// how looking ahead is going
	pub ahead: String,
	/// the real datamosh being made, if it is
	pub mosh: String,
	/// points, big boxes and named objects on show
	pub counts: (usize, usize, usize),
	pub error: Option<String>,
	/// frames done and in all, while exporting
	pub export: Option<(u64, u64)>,
	/// which of the files being exported is under way, and how many there are
	pub export_part: (usize, usize),
	/// how the last export went, and the first file it wrote
	pub exported: Option<String>,
	pub wrote: Option<PathBuf>,
	/// where the preview sits in the source, 0-1 across it
	pub view: View,
	/// the kinds of thing the detector has named in this video so far
	pub seen: Vec<String>,
	/// counts up each time the strip (what's in the clip, for the timeline) changes
	pub strip_version: u64,
}

/// What the look ahead has found in the clip, for the timeline: where the footage cuts, and each
/// thing's life, packed into lanes. Kept out of `Status`, which is compared on every poll.
#[derive(Clone, Debug, Default)]
pub struct Strip {
	/// seconds the look ahead is sure of up to
	pub until: f64,
	/// seconds each new shot starts at (not the first)
	pub cuts: Vec<f64>,
	pub things: Vec<Span>,
	/// where the clip's sound hits (seconds)
	pub beats: Vec<f64>,
}

/// A thing's life on the timeline.
#[derive(Clone, Debug)]
pub struct Span {
	pub cls: String,
	pub conf: f32,
	pub from: f64,
	pub to: f64,
	pub lane: usize,
	/// the pick pressing its bar makes, which means it and nothing else
	pub pick: Pick,
	/// the edit's picks that mean it (when the strip was made)
	pub picks: Vec<Pick>,
	/// taken out by hand, and the edit's drops that took it out
	pub dropped: bool,
	pub drops: Vec<Pick>,
}

impl Strip {
	/// From a tape: its cuts, and its things packed greedily into lanes, longest first, each into the
	/// first lane free over its whole life.
	pub fn of(t: &Tape, edit: &Edit) -> Strip {
		let n = t.times.len();
		if n == 0 {
			return Strip::default();
		}
		let at = |i: usize| t.times[i.min(n - 1)];
		let cuts = (1..n).filter(|&i| t.cuts[i]).map(at).collect();
		// the things, and those taken out (shown in their places, to bring back), longest first
		let all: Vec<(&spotter_core::tape::Thing, Option<usize>)> = t
			.things
			.iter()
			.map(|th| (th, None))
			.chain(t.dropped.iter().enumerate().map(|(j, d)| (&d.thing, Some(j))))
			.collect();
		let mut order: Vec<usize> = (0..all.len()).collect();
		order.sort_by(|&a, &b| all[b].0.boxes.len().cmp(&all[a].0.boxes.len()).then(a.cmp(&b)));
		let mut lanes: Vec<Vec<(f64, f64)>> = Vec::new();
		let mut things = Vec::with_capacity(order.len());
		for k in order {
			let (th, gone) = all[k];
			let (from, to) = (at(th.first), at(th.last()));
			let lane = match lanes.iter().position(|l| l.iter().all(|&(a, b)| to < a || from > b)) {
				Some(l) => l,
				None => {
					lanes.push(Vec::new());
					lanes.len() - 1
				}
			};
			lanes[lane].push((from, to));
			match gone {
				Some(j) => {
					let d = &t.dropped[j];
					things.push(Span {
						cls: th.cls.clone(),
						conf: th.conf,
						from,
						to,
						lane,
						pick: d.pick,
						picks: Vec::new(),
						dropped: true,
						drops: d.drops.clone(),
					});
				}
				None => {
					let picks = edit.picks.iter().copied().filter(|p| t.pick_of(p) == Some(k)).collect();
					things.push(Span {
						cls: th.cls.clone(),
						conf: th.conf,
						from,
						to,
						lane,
						pick: t.pick_for(k),
						picks,
						dropped: false,
						drops: Vec::new(),
					});
				}
			}
		}
		Strip { until: t.until.min(at(n - 1)), cuts, things, beats: Vec::new() }
	}
}

pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// A video being read, with the frame it's on.
struct Clip {
	input: String,
	name: String,
	probe: Probe,
	dec: Decoder,
	w: u32,
	h: u32,
	/// the part of the source shown, in its pixels
	crop: Option<[u32; 4]>,
	/// source time of the next frame to be read
	t: f64,
	frame: Frame,
	/// the moshed copy being read in place of the clip (realmosh.rs), already cut to shape
	video: Option<std::path::PathBuf>,
}

/// What the look ahead's thread hands back.
enum Said {
	Seen(Vec<Seen>),
	Done,
	Failed(String),
}

/// Looking ahead over the open video: a detector of its own going over all of it, frame by frame,
/// handing back what it sees as it goes.
struct Ahead {
	input: String,
	rx: Receiver<Said>,
	stop: Arc<AtomicBool>,
	record: Record,
	/// frames in the video, about
	total: usize,
	done: bool,
	failed: Option<String>,
	/// frames the tape was last worked out from
	built: usize,
}

impl Drop for Ahead {
	fn drop(&mut self) {
		self.stop.store(true, Ordering::Relaxed);
	}
}

/// Start looking ahead over `input` with the detector `model`.
fn look_ahead(input: String, p: &Probe, model: String) -> Ahead {
	let (tx, rx) = channel();
	let stop = Arc::new(AtomicBool::new(false));
	let record =
		Record { frames: Vec::new(), fps: p.fps, aspect: p.w as f32 / p.h as f32, view: WHOLE, model: model.clone() };
	{
		let (input, stop, colour, fps, (w, h)) = (input.clone(), stop.clone(), p.colour.clone(), p.fps, (p.w, p.h));
		std::thread::Builder::new()
			.name("look ahead".into())
			.spawn(move || {
				// it can wait for the preview's own work on the GPU
				spotter_core::gpu::background();
				let run = || -> anyhow::Result<()> {
					let (mut d, _, _) = load_detector(&model, gpu())?;
					// the whole picture, big enough for the detector and no bigger
					let k = (640.0 / w.min(h) as f64).min(1.0);
					let even = |v: f64| ((v / 2.0).round() as u32 * 2).max(2);
					let mut dec = Decoder::open(
						Path::new(&input),
						&Input {
							start: 0.0,
							length: None,
							crop: None,
							w: even(w as f64 * k),
							h: even(h as f64 * k),
							fps,
							colour,
						},
					)?;
					let (mut f, mut looker, mut batch, mut n) = (Frame::default(), Looker::default(), Vec::new(), 0u64);
					while !stop.load(Ordering::Relaxed) && dec.next(&mut f)? {
						batch.push(looker.look(&mut d, &f, n as f64 / fps, WHOLE, n == 0)?);
						n += 1;
						if batch.len() >= 10 && tx.send(Said::Seen(std::mem::take(&mut batch))).is_err() {
							return Ok(());
						}
					}
					if !stop.load(Ordering::Relaxed) {
						let _ = tx.send(Said::Seen(batch));
						let _ = tx.send(Said::Done);
					}
					Ok(())
				};
				if let Err(e) = run() {
					let _ = tx.send(Said::Failed(e.to_string()));
				}
			})
			.expect("a thread to look ahead");
	}
	Ahead {
		input,
		rx,
		stop,
		record,
		total: (p.duration * p.fps).round().max(1.0) as usize,
		done: false,
		failed: None,
		built: 0,
	}
}

/// What a thread set up, once it's there (or why not), after which it isn't waited for.
fn arrived<T, E>(rx: &mut Option<Receiver<Result<T, E>>>) -> Option<Result<T, E>> {
	let got = rx.as_ref()?.try_recv().ok()?;
	*rx = None;
	Some(got)
}

/// A clip got ready on another thread, with the number of the request it answers.
struct Loaded {
	seq: u64,
	/// a new video rather than a new place in the same one
	fresh: bool,
	clip: anyhow::Result<Clip>,
}

struct Worker {
	surface: NativeSurface,
	wake: Waker,
	shared: Arc<Mutex<Status>>,
	/// what's in the clip, for the window's timeline
	strip: Arc<Mutex<Option<Arc<Strip>>>>,
	status: Status,
	engine: Engine,
	clip: Option<Clip>,
	aspect: Option<f64>,
	playing: bool,
	next_due: Instant,
	out: Frame,
	pushed: u64,
	/// something changed that a paused picture should show
	dirty: bool,
	/// drawing failed outright last time: nothing's drawn again until something's asked
	stalled: bool,
	published: Instant,
	shown: u32,
	shown_since: Instant,
	stop_export: Arc<AtomicBool>,
	detector: Option<Receiver<Result<(LiveDetector, String), String>>>,
	/// the newest request for a clip; anything older that turns up is thrown away
	asked: u64,
	/// the seek being read (its number), and where to go once it's in: only one reading at a time,
	/// however fast seeks are asked for (dragging along the timeline asks for one each movement, and
	/// each reading is an ffmpeg; hundreds at once took the whole computer down)
	seeking: Option<u64>,
	then: Option<f64>,
	/// the same for opening: the video being opened (its request's number), and the newest asked
	/// for since, which is all that's kept (ctrl+r held down asks for one a key repeat)
	opening_seq: Option<u64>,
	open_next: Option<String>,
	/// whether the newest request is for a new video
	opening: bool,
	/// reached the end, or a cut, and waiting for the picture after it
	looping: bool,
	loads: (Sender<Loaded>, Receiver<Loaded>),
	draft: Option<Zone>,
	guides: bool,
	/// follow cam: the camera, where its window is now (source pixels), what it's aiming at, and the
	/// picture it sees
	spring: Spring,
	/// EdgeTAM, being set up for a look that wants outlines
	outlines: Option<Receiver<Result<spotter_core::outline::Outliner, String>>>,
	/// Depth Anything, being set up for a look that uses depth
	depths: Option<Receiver<Result<spotter_core::depth::Depther, String>>>,
	/// SegFormer, being set up for the words look
	scenes: Option<Receiver<Result<spotter_core::scene::Scener, String>>>,
	window: Option<Window>,
	aim: Option<spotter_core::Rect>,
	seen: Frame,
	/// the preview's sound, while it's playing
	sound: Option<crate::sound::Sound>,
	muted: bool,
	/// how loud the preview plays, as an f32's bits (the sound reads it as it plays)
	volume: Arc<std::sync::atomic::AtomicU32>,
	/// showing the picture as it is (comparing)
	bare: bool,
	/// when to start the sound again after a jump: not straight away, so dragging along the
	/// timeline doesn't start (and, over the web, connect) a new one for every place it passes
	sound_due: Option<Instant>,
	/// the detector, as `load_detector` names it
	model: String,
	/// looking ahead over the open video
	ahead: Option<Ahead>,
	/// the settings the tape was last worked out with
	tape_key: String,
	/// smooth slow motion: RIFE, being set up and set up; the source frame before the one read, its
	/// time, how far on from it the picture is (0-1), and the picture made between them, with its time
	tweens: Option<Receiver<Result<spotter_core::interp::Interpolator, String>>>,
	tween: Option<spotter_core::interp::Interpolator>,
	prev: Frame,
	prev_t: f64,
	phase: f64,
	tweened: Frame,
	tween_t: f64,
	/// when the last picture between was shown, so the clip moves at its speed however quickly they come
	tween_at: Option<Instant>,
	/// the beats of the open clip's sound, being found, and found
	beats_rx: Option<(String, Receiver<Vec<f64>>)>,
	beats: Option<(String, Arc<Vec<f64>>)>,
	/// the in point (seconds of the source), where the output's clock, and a song, start
	origin: f64,
	/// the real datamosh: the copy being made (what for, and the answer), and the one made (the clip
	/// as shown it's for, what for, and where it is); what for when it couldn't be made
	moshing: Option<(String, Receiver<Result<std::path::PathBuf, String>>)>,
	moshed: Option<(String, String, std::path::PathBuf)>,
	mosh_failed: Option<String>,
	/// when the clip was last read again for the mosh (or back from it)
	mosh_sought: Option<Instant>,
}

/// Set a detector up on a thread of its own: it takes a second or two on the GPU (and the
/// first time, a download), and the picture doesn't wait for it.
fn set_up(name: String) -> Receiver<Result<(LiveDetector, String), String>> {
	let (dtx, drx) = channel();
	std::thread::Builder::new()
		.name("detector setup".into())
		.spawn(move || {
			let kinds = if name.ends_with("365") { "365 kinds" } else { "80 kinds" };
			let got = load_detector(&name, gpu()).map(|(d, device, warning)| {
				let size = name.trim_end_matches("365");
				(
					LiveDetector::spawn(d),
					warning.unwrap_or_else(|| {
						format!(
							"d-fine {size}, {kinds}, on {}",
							if device == gpu() { "the graphics card" } else { "the processor" }
						)
					}),
				)
			});
			let _ = dtx.send(got.map_err(|e| e.to_string()));
		})
		.expect("a thread to set up the detector");
	drx
}

pub fn spawn(
	surface: NativeSurface,
	wake: Waker,
	shared: Arc<Mutex<Status>>,
	strip: Arc<Mutex<Option<Arc<Strip>>>>,
	look: Look,
	detector: String,
) -> Sender<Cmd> {
	let (tx, rx) = channel();
	let drx = set_up(detector.clone());
	std::thread::Builder::new()
		.name("playback".into())
		.spawn(move || {
			// below normal, so the desktop and other programs come first
			crate::safety::lower_priority();
			let now = Instant::now();
			let mut w = Worker {
				surface,
				wake,
				shared,
				strip,
				status: Status { detector: "setting up the detector".into(), ..Status::default() },
				engine: {
					// the slow models on a thread of their own, so the picture never waits for them
					let mut e = Engine::new(look);
					e.background = Some(spotter_core::background::Background::new());
					e
				},
				clip: None,
				aspect: None,
				playing: false,
				next_due: now,
				out: Frame::default(),
				pushed: 0,
				dirty: false,
				stalled: false,
				published: now,
				shown: 0,
				shown_since: now,
				stop_export: Arc::new(AtomicBool::new(false)),
				detector: Some(drx),
				asked: 0,
				seeking: None,
				then: None,
				opening_seq: None,
				open_next: None,
				opening: false,
				looping: false,
				loads: channel(),
				draft: None,
				guides: false,
				spring: Spring::default(),
				outlines: None,
				depths: None,
				scenes: None,
				window: None,
				aim: None,
				seen: Frame::default(),
				sound: None,
				muted: false,
				volume: Arc::new(std::sync::atomic::AtomicU32::new(1.0f32.to_bits())),
				bare: false,
				sound_due: None,
				model: detector,
				ahead: None,
				tape_key: String::new(),
				beats_rx: None,
				beats: None,
				origin: 0.0,
				moshing: None,
				moshed: None,
				mosh_failed: None,
				mosh_sought: None,
				tweens: None,
				tween: None,
				prev: Frame::default(),
				prev_t: 0.0,
				phase: 0.0,
				tweened: Frame::default(),
				tween_t: 0.0,
				tween_at: None,
			};
			w.run(rx);
		})
		.expect("a thread to play the video");
	tx
}

/// Open `input` at `t` at the preview's size and shape, and read the frame there. Runs off the
/// playback thread, since over the web it can take a while.
fn load(
	input: String,
	name: String,
	p: Probe,
	t: f64,
	aspect: Option<f64>,
	most: f64,
	follow: bool,
	video: Option<std::path::PathBuf>,
) -> anyhow::Result<Clip> {
	// with the follow cam, the whole picture is read and the camera's window cut from it as it plays
	let crop = if follow { None } else { aspect.map(|a| cut(p.w, p.h, a)) };
	let (cw, ch) = crop.map_or((p.w, p.h), |c| (c[2], c[3]));
	let (w, h) = fit(cw, ch, PREVIEW.min(most));
	// on a frame, as playing from the start is, so what the models said about it is found again
	let last = ((p.duration - 0.05).max(0.0) * p.fps).floor();
	let t = (t * p.fps).round().clamp(0.0, last) / p.fps;
	let mut dec = match &video {
		Some(v) => Decoder::open(
			v,
			&Input { start: t, length: None, crop: None, w, h, fps: p.fps, colour: Default::default() },
		)?,
		None => Decoder::open(
			Path::new(&input),
			&Input { start: t, length: None, crop, w, h, fps: p.fps, colour: p.colour.clone() },
		)?,
	};
	let mut frame = Frame::default();
	if !dec.next(&mut frame)? {
		anyhow::bail!("{name} has no picture at {t:.1}s");
	}
	Ok(Clip { input, name, probe: p, dec, w, h, crop, t, frame, video })
}

/// The clip as it's read: which, cut and sized how. A moshed copy is made for one of these.
fn shape_key(c: &Clip) -> String {
	format!("{} {:?} {}x{}", c.input, c.crop, c.w, c.h)
}

impl Worker {
	fn run(&mut self, rx: Receiver<Cmd>) {
		let mut checked = Instant::now();
		let mut failing = 0u32;
		loop {
			match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.turn(&rx, &mut checked))) {
				Ok(true) => failing = 0,
				Ok(false) => return,
				// a look or a model failed outright: stop and say so, rather than lose the picture
				// for good or go round failing; whatever's asked next tries again
				Err(p) => {
					self.playing = false;
					self.stalled = true;
					self.status.error = Some(format!("{} (spotter.log has more)", crate::log::text(&*p)));
					self.publish(true);
					// failing outside drawing too: wait longer each time, so the log isn't flooded
					failing += 1;
					std::thread::sleep(Duration::from_millis(200 * failing.min(25) as u64));
				}
			}
		}
	}

	/// Take what's been asked and what's come back, and draw the next frame if it's due. False once
	/// the window's gone.
	fn turn(&mut self, rx: &Receiver<Cmd>, checked: &mut Instant) -> bool {
		// using a lot of memory: let go of what's kept to save time, well before the ceiling
		if checked.elapsed() >= Duration::from_secs(2) {
			*checked = Instant::now();
			let used = crate::safety::memory();
			if used > crate::safety::SHED_AT {
				self.engine.shed();
				self.status.error = Some(format!(
					"spotter was using {:.1} GB of memory, so it let go of what it had kept to save time",
					used as f64 / (1u64 << 30) as f64
				));
			}
		}
		let wait = if self.playing && self.clip.is_some() {
			self.next_due.saturating_duration_since(Instant::now())
		} else {
			Duration::from_millis(40)
		};
		match rx.recv_timeout(wait.min(Duration::from_millis(40))) {
			Ok(cmd) => {
				self.handle(cmd);
				// take everything else that's waiting before drawing, so a drag through the
				// settings or along the timeline draws only where it ended up
				while let Ok(cmd) = rx.try_recv() {
					self.handle(cmd);
				}
			}
			Err(RecvTimeoutError::Timeout) => {}
			Err(RecvTimeoutError::Disconnected) => return false,
		}
		let mut lap = spotter_core::perf::Lap::start();
		self.take_detector();
		self.take_outliner();
		self.take_depther();
		self.take_scener();
		lap.mark("loop: models");
		self.take_ahead();
		lap.mark("loop: look ahead");
		self.take_beats();
		self.want_mosh();
		lap.mark("loop: beats, mosh");
		if let Some(rx) = &self.tweens
			&& let Ok(got) = rx.try_recv()
		{
			match got {
				Ok(t) => self.tween = Some(t),
				Err(e) => self.status.error = Some(format!("no smooth slow motion ({e})")),
			}
			self.tweens = None;
		}
		if self.sound_due.is_some_and(|due| Instant::now() >= due) {
			self.sound_due = None;
			self.restart_sound();
		}
		self.take_loads();
		if !self.stalled {
			if self.playing && !self.looping && self.clip.is_some() && Instant::now() >= self.next_due {
				self.advance();
			} else if !self.playing && self.clip.is_some() && (self.dirty || self.engine.settling()) {
				self.redraw();
			}
		}
		lap.mark("loop: frame");
		self.publish(false);
		lap.mark("loop: publish");
		true
	}

	/// Take in what looking ahead has seen since last time, and work the tape out again once there's a
	/// couple of seconds more of it, or it's done.
	fn take_ahead(&mut self) {
		let Some(a) = self.ahead.as_mut() else { return };
		let mut grew = false;
		while let Ok(said) = a.rx.try_recv() {
			match said {
				Said::Seen(v) => {
					grew |= !v.is_empty();
					a.record.frames.extend(v);
				}
				Said::Done => {
					a.done = true;
					grew = true;
				}
				Said::Failed(e) => {
					a.done = true;
					a.failed = Some(e);
				}
			}
		}
		let every = (a.record.fps * 2.0).max(10.0) as usize;
		if grew && (a.done || a.record.frames.len() >= a.built + every) {
			self.build_tape();
		}
		let l = &self.engine.look;
		self.status.ahead = match &self.ahead {
			_ if !l.detect || !l.ahead => String::new(),
			None => String::new(),
			Some(a) if a.failed.is_some() => {
				format!("couldn't look ahead: {}", a.failed.as_deref().unwrap_or_default())
			}
			Some(a) if a.done => match self.engine.tape() {
				Some(t) => format!(
					"looked ahead over all of it: {}, {}",
					counted(t.cuts.iter().filter(|&&c| c).count(), "shot"),
					counted(t.things.len(), "thing")
				),
				None => "looked ahead over all of it".into(),
			},
			Some(a) => format!("looking ahead: {}%", (a.record.frames.len() * 100 / a.total).min(99)),
		};
	}

	/// Work the tape out again from what's been looked at, with the look's and the edit's settings.
	fn build_tape(&mut self) {
		let l = &self.engine.look;
		self.tape_key = format!("{} {} {} {} {}", l.detect, l.ahead, l.confidence, l.object_kinds, l.ahead_smooth);
		let tape = match self.ahead.as_mut() {
			Some(a) if l.detect && l.ahead && a.failed.is_none() && !a.record.frames.is_empty() => {
				a.built = a.record.frames.len();
				let mut t = Tape::build(&a.record, l, &self.engine.edit);
				// things near the newest frame looked at may yet carry on past it
				if !a.done {
					t.until = a.record.frames.last().map_or(0.0, |f| f.t) - 0.5;
				}
				Some(Arc::new(t))
			}
			_ => None,
		};
		if tape.is_some() || self.engine.tape().is_some() {
			let beats = self.engine.beats().map(|b| b.to_vec()).unwrap_or_default();
			*self.strip.lock().unwrap() =
				tape.as_deref().map(|t| Arc::new(Strip { beats: beats.clone(), ..Strip::of(t, &self.engine.edit) }));
			self.status.strip_version += 1;
			self.engine.set_tape(tape);
			self.dirty = true;
		}
	}

	/// The real datamosh (realmosh.rs): when the look wants it, have the moshed copy of the clip made
	/// (as it's shown, on the beats if it's to be and they're found, else on its cuts), and read it
	/// in the clip's place once it's there; when the look doesn't, go back to the clip.
	fn want_mosh(&mut self) {
		let Some(c) = self.clip.as_ref() else { return };
		if self.opening {
			return;
		}
		// what's needed of the clip, so it can be read again from here
		let (back, current, shape) = (c.t - 1.0 / c.probe.fps, c.video.clone(), shape_key(c));
		let input = std::path::PathBuf::from(&c.input);
		let inp = Input {
			start: 0.0,
			length: None,
			crop: c.crop,
			w: c.w,
			h: c.h,
			fps: c.probe.fps,
			colour: c.probe.colour.clone(),
		};
		// a reading of the clip asked for a moment ago is still on its way
		let waiting = self.mosh_sought.is_some_and(|at| at.elapsed() < Duration::from_millis(1500));
		let Some(on_beats) = self.engine.look.real_mosh() else {
			self.status.mosh.clear();
			if current.is_some() && !waiting {
				self.mosh_sought = Some(Instant::now());
				self.seek(back);
			}
			return;
		};
		// on the beats: wait for them while they're being found
		if on_beats && self.beats_rx.is_some() {
			return;
		}
		let when = match self.engine.beats() {
			Some(b) if on_beats && !b.is_empty() => spotter_core::realmosh::When::Beats(b.to_vec()),
			_ => spotter_core::realmosh::When::Cuts,
		};
		let hold = self.engine.look.mosh_hold as f64;
		let key = format!("{shape} {when:?} {hold:.2}");
		if self.mosh_failed.as_ref() == Some(&key) {
			return;
		}
		if let Some(m) = &self.moshed
			&& m.1 == key
		{
			if current.as_ref() != Some(&m.2) && !waiting {
				self.mosh_sought = Some(Instant::now());
				self.seek(back);
			}
			return;
		}
		if let Some((k, rx)) = &self.moshing
			&& *k == key
		{
			match rx.try_recv() {
				Ok(Ok(path)) => {
					self.moshed = Some((shape, key, path));
					self.moshing = None;
					self.status.mosh.clear();
					self.mosh_sought = Some(Instant::now());
					self.seek(back);
				}
				Ok(Err(e)) => {
					self.status.error = Some(format!("no real datamosh: {e}"));
					self.status.mosh.clear();
					self.mosh_failed = Some(key);
					self.moshing = None;
				}
				Err(_) => {}
			}
			return;
		}
		let (tx, rx) = channel();
		std::thread::Builder::new()
			.name("mosh".into())
			.spawn(move || {
				let _ = tx.send(spotter_core::realmosh::prepare(&input, &inp, &when, hold).map_err(|e| e.to_string()));
			})
			.expect("a thread to mosh the clip");
		self.moshing = Some((key, rx));
		self.status.mosh = "moshing the clip (re-encoding it, rearranging its frames)…".into();
	}

	/// Find the beats in the song under the video, or else the open clip's sound, on a thread of its
	/// own, if they aren't found yet.
	fn start_beats(&mut self) {
		let Some(c) = self.clip.as_ref() else { return };
		let input = match &self.engine.edit.music {
			Some(m) => m.path.clone(),
			None if c.probe.audio => c.input.clone(),
			None => return,
		};
		if self.beats.as_ref().is_some_and(|b| b.0 == input) || self.beats_rx.as_ref().is_some_and(|b| b.0 == input) {
			return;
		}
		let (tx, rx) = channel();
		let key = input.clone();
		std::thread::Builder::new()
			.name("beats".into())
			.spawn(move || {
				let _ = tx.send(spotter_core::beats::find(&input).unwrap_or_default());
			})
			.expect("a thread to find the beats");
		self.beats_rx = Some((key, rx));
	}

	fn take_beats(&mut self) {
		let Some((input, rx)) = &self.beats_rx else { return };
		let Ok(b) = rx.try_recv() else { return };
		self.beats = Some((input.clone(), Arc::new(b)));
		self.beats_rx = None;
		self.place_beats();
	}

	/// Give the look the beats it should follow, in the source's seconds: the song's, laid along the
	/// output's clock from the in point (over cuts, slowed with the picture), or the clip's own.
	fn place_beats(&mut self) {
		let Some(c) = self.clip.as_ref() else { return };
		let placed = match (&self.beats, &self.engine.edit.music) {
			(Some((k, b)), Some(m)) if *k == m.path => {
				let speed = (self.engine.look.speed as f64).clamp(0.05, 4.0);
				Some(Arc::new(m.beats_in_source(b, &self.engine.edit, self.origin, c.probe.duration, speed)))
			}
			(Some((k, b)), None) if *k == c.input => Some(b.clone()),
			// the song's beats aren't found yet (or are another's): none for now
			_ => None,
		};
		let same = match (self.engine.beats(), &placed) {
			(Some(a), Some(b)) => a == b,
			(None, None) => true,
			_ => false,
		};
		if same {
			return;
		}
		self.engine.set_beats(placed.clone());
		// onto the timeline
		let mut strip = self.strip.lock().unwrap();
		let mut s = strip.as_deref().cloned().unwrap_or_default();
		s.beats = placed.map(|b| b.to_vec()).unwrap_or_default();
		*strip = Some(Arc::new(s));
		self.status.strip_version += 1;
		self.dirty = true;
	}

	/// Start looking ahead over the open video, if the look wants it and it isn't already.
	fn start_ahead(&mut self, again: bool) {
		let l = &self.engine.look;
		let Some(c) = self.clip.as_ref() else { return };
		if !l.detect || !l.ahead || (!again && self.ahead.as_ref().is_some_and(|a| a.input == c.input)) {
			return;
		}
		self.ahead = Some(look_ahead(c.input.clone(), &c.probe, self.model.clone()));
		self.engine.set_tape(None);
	}

	fn take_scener(&mut self) {
		match arrived(&mut self.scenes) {
			Some(Ok(s)) => {
				self.engine.adopt_scener(s);
				self.dirty = true;
			}
			Some(Err(e)) => self.status.error = Some(format!("no scene ({e})")),
			None => {}
		}
	}

	fn take_depther(&mut self) {
		match arrived(&mut self.depths) {
			Some(Ok(d)) => {
				self.engine.adopt_depther(d);
				self.dirty = true;
			}
			Some(Err(e)) => self.status.error = Some(format!("no depth ({e})")),
			None => {}
		}
	}

	fn take_outliner(&mut self) {
		match arrived(&mut self.outlines) {
			Some(Ok(o)) => {
				self.engine.adopt_outliner(o);
				self.dirty = true;
			}
			Some(Err(e)) => {
				self.status.error = Some(format!("no outlines ({e}); cutting things out by colour instead"))
			}
			None => {}
		}
	}

	fn take_detector(&mut self) {
		match arrived(&mut self.detector) {
			Some(Ok((live, said))) => {
				self.engine.live = Some(live);
				self.status.detector = said;
				self.dirty = true;
			}
			Some(Err(e)) => self.status.detector = format!("no detector: {e}"),
			None => {}
		}
	}

	fn take_loads(&mut self) {
		while let Ok(l) = self.loads.1.try_recv() {
			if self.seeking == Some(l.seq) {
				self.seeking = None;
			}
			if self.opening_seq == Some(l.seq) {
				self.opening_seq = None;
				// another's been asked for since: open that instead of showing this one
				if self.open_next.is_some() {
					continue;
				}
			}
			if l.seq != self.asked {
				continue;
			}
			self.opening = false;
			self.status.opening = false;
			match l.clip {
				Ok(c) => self.show(c, l.fresh),
				Err(e) => self.status.error = Some(e.to_string()),
			}
			self.publish(true);
		}
		if self.opening_seq.is_none()
			&& let Some(a) = self.open_next.take()
		{
			self.open(a);
		}
		// the newest seek asked for while that one was being read
		if self.seeking.is_none()
			&& let Some(t) = self.then.take()
		{
			self.seek(t);
		}
	}

	fn handle(&mut self, cmd: Cmd) {
		self.stalled = false;
		match cmd {
			Cmd::Open(arg) => self.open(arg),
			Cmd::Toggle => {
				self.playing = !self.playing && self.clip.is_some();
				self.next_due = Instant::now();
				self.sound_soon();
			}
			Cmd::Bare(b) => {
				self.bare = b;
				self.push();
			}
			Cmd::Mute(m) => {
				self.muted = m;
				self.sound_soon();
			}
			Cmd::Volume(v) => self.volume.store(v.max(0.0).to_bits(), Ordering::Relaxed),
			Cmd::In(t) => {
				let t = t.unwrap_or(0.0);
				if t != self.origin {
					self.origin = t;
					self.place_beats();
					if self.engine.edit.music.is_some() {
						self.sound_due = Some(Instant::now() + Duration::from_millis(150));
					}
				}
			}
			Cmd::Seek(t) => self.seek(t),
			Cmd::Look(l) => {
				// the first look to want outlines sets EdgeTAM up, on a thread of its own
				if l.either(spotter_core::matte::wants_outlines)
					&& !self.engine.has_outliner()
					&& self.outlines.is_none()
				{
					let (otx, orx) = channel();
					std::thread::Builder::new()
						.name("outliner setup".into())
						.spawn(move || {
							let _ = otx.send(
								spotter_core::outline::Outliner::load(spotter_core::outline::MODELS[1], gpu(), false)
									.map_err(|e| e.to_string()),
							);
						})
						.expect("a thread to set up the outliner");
					self.outlines = Some(orx);
				}
				if l.either(spotter_core::depth::active) && self.engine.depther.is_none() && self.depths.is_none() {
					let (dtx, drx) = channel();
					std::thread::Builder::new()
						.name("depth setup".into())
						.spawn(move || {
							let _ = dtx.send(
								spotter_core::depth::Depther::load(spotter_core::depth::MODELS[1], gpu(), false)
									.map_err(|e| e.to_string()),
							);
						})
						.expect("a thread to set up depth");
					self.depths = Some(drx);
				}
				// slowed down and smooth: RIFE, to make the frames between
				if l.smooth_slow && l.speed < 0.99 && self.tween.is_none() && self.tweens.is_none() {
					let (ttx, trx) = channel();
					std::thread::Builder::new()
						.name("rife setup".into())
						.spawn(move || {
							let _ =
								ttx.send(spotter_core::interp::Interpolator::load(gpu()).map_err(|e| e.to_string()));
						})
						.expect("a thread to set up RIFE");
					self.tweens = Some(trx);
				}
				// the window's SegFormer is the quicker one (exports use the best)
				if l.either(spotter_core::scene::active) && self.engine.scener.is_none() && self.scenes.is_none() {
					let (stx, srx) = channel();
					std::thread::Builder::new()
						.name("scene setup".into())
						.spawn(move || {
							let _ = stx.send(
								spotter_core::scene::Scener::load(spotter_core::scene::MODELS[1], gpu(), false)
									.map_err(|e| e.to_string()),
							);
						})
						.expect("a thread to set up the scene");
					self.scenes = Some(srx);
				}
				let resize = self.clip.is_some()
					&& (l.out_max != self.engine.look.out_max || l.follow != self.engine.look.follow);
				let speed = l.speed != self.engine.look.speed;
				self.engine.look = l;
				if speed {
					// the song's beats fall elsewhere in the source, and the sound plays at the new speed
					self.place_beats();
					self.sound_due = Some(Instant::now() + Duration::from_millis(150));
				}
				self.dirty = true;
				let l = &self.engine.look;
				if format!("{} {} {} {} {}", l.detect, l.ahead, l.confidence, l.object_kinds, l.ahead_smooth)
					!= self.tape_key
				{
					self.start_ahead(false);
					self.build_tape();
				}
				if resize {
					self.seek(self.clip.as_ref().map_or(0.0, |c| c.t));
				}
			}
			Cmd::Aspect(a) => {
				self.aspect = a;
				if let Some(t) = self.clip.as_ref().map(|c| c.t) {
					self.seek(t);
				}
			}
			Cmd::Export(es) => self.export(es),
			Cmd::StopExport => self.stop_export.store(true, Ordering::Relaxed),
			Cmd::Edit(e) => {
				let music = e.music != self.engine.edit.music;
				self.engine.edit = e;
				self.dirty = true;
				// a song added, moved or turned up: its beats, and the sound from where it is now
				if music {
					self.start_beats();
					self.sound_due = Some(Instant::now() + Duration::from_millis(150));
				}
				self.place_beats();
				// its kinds, places and picks change what the tape holds
				self.build_tape();
			}
			Cmd::Draft(z) => {
				self.draft = z;
				self.push();
			}
			Cmd::Guides(on) => {
				self.guides = on;
				self.push();
			}
			Cmd::Detector(name) => {
				self.engine.live = None;
				// the first time a size is used it's downloaded (40 to 250 MB), which the status line follows
				self.status.detector = "setting up the detector".into();
				self.detector = Some(set_up(name.clone()));
				self.model = name;
				self.dirty = true;
				self.start_ahead(true);
			}
		}
	}

	fn open(&mut self, arg: String) {
		if self.opening_seq.is_some() {
			self.open_next = Some(arg);
			return;
		}
		self.asked += 1;
		self.opening_seq = Some(self.asked);
		self.opening = true;
		self.status.opening = true;
		self.status.error = None;
		self.publish(true);
		let (seq, aspect, most, back, follow) =
			(self.asked, self.aspect, self.engine.look.out_max as f64, self.loads.0.clone(), self.engine.look.follow);
		std::thread::Builder::new()
			.name("open".into())
			.spawn(move || {
				let clip = (|| {
					let (input, name, clip) = resolve(&arg)?;
					let p = probe(Path::new(&input))?;
					// a long video picked at random starts somewhere in the middle, as on the site
					let start = if arg == "random" && clip.is_some_and(|c| c.wanders()) {
						Dice::default().unit() * (p.duration - 20.0).max(0.0)
					} else {
						0.0
					};
					load(input, name, p, start, aspect, most, follow, None)
				})();
				let _ = back.send(Loaded { seq, fresh: true, clip });
			})
			.expect("a thread to open a video");
	}

	/// Go to `t` in what's playing. Until the new place is ready, the old one carries on.
	fn seek(&mut self, t: f64) {
		let Some(c) = self.clip.as_ref() else { return };
		// a seek doesn't cancel the opening of a new video
		if self.opening {
			return;
		}
		// one reading at a time: while one's on its way, only the newest place asked for is kept
		if self.seeking.is_some() {
			self.then = Some(t);
			return;
		}
		self.asked += 1;
		self.seeking = Some(self.asked);
		let (seq, aspect, most, back, follow) =
			(self.asked, self.aspect, self.engine.look.out_max as f64, self.loads.0.clone(), self.engine.look.follow);
		let (input, name, p) = (c.input.clone(), c.name.clone(), c.probe.clone());
		// the moshed copy, if the look's on it and it's made for this clip as it's shown
		let video = self
			.moshed
			.as_ref()
			.filter(|m| self.engine.look.real_mosh().is_some() && m.0 == shape_key(c))
			.map(|m| m.2.clone());
		std::thread::Builder::new()
			.name("seek".into())
			.spawn(move || {
				let clip = load(input, name, p, t, aspect, most, follow, video);
				let _ = back.send(Loaded { seq, fresh: false, clip });
			})
			.expect("a thread to seek");
	}

	/// Put a newly read clip on screen, starting from its first frame.
	fn show(&mut self, c: Clip, fresh: bool) {
		if fresh {
			self.engine.name = c.name.clone();
			self.engine.fps = c.probe.fps;
			self.status.name = c.name.clone();
			self.status.input = c.input.clone();
			self.status.duration = c.probe.duration;
			self.status.rate = c.probe.fps;
			self.status.seen.clear();
			self.playing = true;
		}
		let view = c.crop.map_or(WHOLE, |k| {
			[
				k[0] as f32 / c.probe.w as f32,
				k[1] as f32 / c.probe.h as f32,
				k[2] as f32 / c.probe.w as f32,
				k[3] as f32 / c.probe.h as f32,
			]
		});
		self.engine.view = view;
		self.status.view = view;
		if self.looping {
			self.looping = false;
			self.playing = true;
		}
		self.status.w = c.w;
		self.status.h = c.h;
		self.engine.jumped();
		self.spring.reset();
		self.window = None;
		self.aim = None;
		self.prev = Frame::default();
		// what the models said is kept for this clip at this size and crop (a loop or a seek keeps it)
		let context = format!("{} {}x{} {:?} {:?}", c.input, c.w, c.h, c.crop, c.video);
		self.engine.recall.get_or_insert_with(Default::default).keep_for(&context);
		self.clip = Some(c);
		if fresh {
			self.engine.set_beats(None);
			self.start_ahead(false);
			self.start_beats();
			// found already (the same clip again): back on
			self.place_beats();
		}
		if let Err(e) = self.step() {
			self.status.error = Some(e.to_string());
		}
		self.next_due = Instant::now();
		// a new video starts its sound now; a jump in this one once the picture has stayed put a moment
		self.sound = None;
		self.sound_due = Some(Instant::now() + Duration::from_millis(if fresh { 0 } else { 350 }));
	}

	/// Sound off at once, and on again (if it should be) once play and mute have been left alone a
	/// moment: space pressed over and over starts one ffmpeg, not one a press.
	fn sound_soon(&mut self) {
		self.sound = None;
		self.sound_due = Some(Instant::now() + Duration::from_millis(100));
	}

	/// Start the sound again from where the picture is, or stop it if it shouldn't be playing.
	fn restart_sound(&mut self) {
		self.sound = None;
		self.sound_due = None;
		let Some(c) = self.clip.as_ref() else { return };
		let music = self.engine.edit.music.as_ref().filter(|m| !m.path.is_empty());
		if !self.playing || self.muted || (!c.probe.audio && music.is_none()) {
			return;
		}
		// the frame on screen is a frame behind where the clip has read to
		let t = (c.t - 1.0 / c.probe.fps).max(0.0);
		let speed = (self.engine.look.speed as f64).clamp(0.05, 4.0);
		// the song where the output's clock is: from the in point, over cuts, at its own speed
		let song = music.map(|m| crate::sound::Song {
			path: m.path.clone(),
			at: m.from + self.engine.edit.output_time(self.origin, t, speed),
			gain: m.gain,
			keep: m.keep,
		});
		let clip = (c.probe.audio && song.as_ref().is_none_or(|s| s.keep > 0.0)).then_some(c.input.as_str());
		match crate::sound::Sound::start(clip, t, speed, song.as_ref(), self.volume.clone()) {
			Ok(s) => self.sound = Some(s),
			Err(e) => self.status.error = Some(format!("no sound: {e}")),
		}
	}

	/// Work out the frame the clip is on and show it, then move the clip's time on by a frame.
	fn step(&mut self) -> anyhow::Result<()> {
		self.show_frame(false)
	}

	/// Work out the picture and show it: the frame the clip is on (then the clip's time moves on by a
	/// frame), or with `tweened`, the one made between two frames for smooth slow motion.
	fn show_frame(&mut self, tweened: bool) -> anyhow::Result<()> {
		let Some(c) = self.clip.as_mut() else { return Ok(()) };
		let fps = c.probe.fps;
		let speed = (self.engine.look.speed as f64).clamp(0.05, 4.0);
		let (t, dt) = if tweened { (self.tween_t, 1.0 / fps) } else { (c.t, 1.0 / (fps * speed)) };
		let src: &Frame = if tweened { &self.tweened } else { &c.frame };
		if self.engine.look.follow && c.crop.is_none() {
			// the camera eases toward what it was aiming at last frame, and sees through its window
			let (sw, sh) = (c.w as f32, c.h as f32);
			let aspect = self.aspect.map_or(sw / sh, |a| a as f32);
			// the footage cut to a new shot: the camera cuts with it, straight to what's there
			if let Some(r) = self.engine.aim_at_cut(t) {
				self.spring.reset();
				self.aim = Some(spotter_core::Rect { x: r.x * sw, y: r.y * sh, w: r.w * sw, h: r.h * sh });
			}
			let win = self.spring.step(self.aim, sw, sh, aspect, &self.engine.look, (1.0 / fps) as f32);
			let (mw, mh) = widest(sw, sh, aspect);
			let (pw, ph) = ((mw / 2.0).round() as u32 * 2, (mh / 2.0).round() as u32 * 2);
			if self.seen.w != pw || self.seen.h != ph {
				self.seen = Frame::new(pw.max(2), ph.max(2));
			}
			src.crop_into(win, &mut self.seen);
			let view = [win[0] / sw, win[1] / sh, win[2] / sw, win[3] / sh];
			self.engine.view = view;
			self.status.view = view;
			self.status.w = self.seen.w;
			self.status.h = self.seen.h;
			self.window = Some(win);
			self.engine.step(&self.seen, t, dt, &mut self.out)?;
			self.aim = self.engine.target().map(|r| spotter_core::Rect {
				x: win[0] + r.x * win[2],
				y: win[1] + r.y * win[3],
				w: r.w * win[2],
				h: r.h * win[3],
			});
		} else {
			self.engine.step(src, t, dt, &mut self.out)?;
		}
		if !tweened {
			c.t += 1.0 / fps;
		}
		self.status.t = t;
		self.status.counts = self.engine.counts();
		self.note_kinds();
		let mut lap = spotter_core::perf::Lap::start();
		self.push();
		lap.mark("show");
		self.shown += 1;
		// timing (SPOTTER_TIMES): where the time of the last 150 frames went
		static TIMED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
		let n = TIMED.fetch_add(1, Ordering::Relaxed) + 1;
		if n.is_multiple_of(150) {
			let r = spotter_core::perf::report(150);
			if !r.is_empty() {
				let (hits, misses, mb) =
					self.engine.recall.as_ref().map_or((0, 0, 0), |r| (r.hits, r.misses, r.bytes() >> 20));
				eprint!(
					"at {t:.1}s, {} fps, models remembered {hits} times, asked {misses} ({mb} MB kept): {r}",
					self.status.fps
				);
				spotter_core::perf::clear();
			}
		}
		Ok(())
	}

	/// Slowed down and smooth: move on by `speed` of a source frame, reading frames as that passes
	/// them, and show the picture made between the two either side. Says false at the end of the clip.
	fn advance_tween(&mut self, fps: f64, speed: f64) -> anyhow::Result<bool> {
		let Some(c) = self.clip.as_mut() else { return Ok(false) };
		if self.prev.w == 0 || self.prev.w != c.frame.w || self.prev.h != c.frame.h {
			// starting: from the frame on screen to the next one
			self.prev.clone_from(&c.frame);
			self.prev_t = c.t - 1.0 / fps;
			if !c.dec.next(&mut c.frame)? {
				return Ok(false);
			}
			c.t += 1.0 / fps;
			self.phase = 0.0;
			self.tween_at = Some(Instant::now());
		} else {
			// on by the time that's passed (at the clip's rate, slowed), however long the last took
			let now = Instant::now();
			let passed = self.tween_at.map_or(1.0 / fps, |t| now.duration_since(t).as_secs_f64()).min(0.25);
			self.tween_at = Some(now);
			self.phase += speed * fps * passed;
			while self.phase >= 1.0 {
				self.prev.clone_from(&c.frame);
				self.prev_t = c.t - 1.0 / fps;
				if !c.dec.next(&mut c.frame)? {
					return Ok(false);
				}
				c.t += 1.0 / fps;
				self.phase -= 1.0;
			}
		}
		let tween = self.tween.as_mut().expect("RIFE is set up");
		if let Err(e) = tween.between(&self.prev, &c.frame, self.phase as f32, &mut self.tweened) {
			self.status.error = Some(format!("slow motion: {e}"));
			self.tweened.clone_from(&c.frame);
		}
		self.tween_t = self.prev_t + self.phase / fps;
		self.show_frame(true)?;
		Ok(true)
	}

	fn advance(&mut self) {
		let (fps, speed) =
			(self.clip.as_ref().map_or(30.0, |c| c.probe.fps), (self.engine.look.speed as f64).clamp(0.05, 4.0));
		let look = &self.engine.look;
		if self.tween.is_some() && look.smooth_slow && speed < 0.99 {
			match self.advance_tween(fps, speed) {
				Ok(true) => {
					let (next, duration) = self.clip.as_ref().map_or((0.0, 0.0), |c| (c.t, c.probe.duration));
					if let Some(end) = self.engine.edit.cut_at(next) {
						self.looping = true;
						self.seek(if end >= duration - 0.05 { 0.0 } else { end });
					}
				}
				Ok(false) => {
					self.looping = true;
					self.seek(0.0);
				}
				Err(e) => {
					self.status.error = Some(e.to_string());
					self.playing = false;
				}
			}
			// shown at the clip's own rate, each a fraction of a source frame on
			self.next_due += Duration::from_secs_f64(1.0 / fps);
			let now = Instant::now();
			if now > self.next_due + Duration::from_millis(250) {
				self.next_due = now;
			}
			return;
		}
		self.prev = Frame::default();
		let mut lap = spotter_core::perf::Lap::start();
		let read = match self.clip.as_mut() {
			Some(c) => c.dec.next(&mut c.frame),
			None => return,
		};
		lap.mark("decode");
		match read {
			Ok(true) => {
				if let Err(e) = self.step() {
					self.status.error = Some(e.to_string());
					self.playing = false;
				}
				// into a cut: on to where it ends, or round to the start
				let (next, duration) = self.clip.as_ref().map_or((0.0, 0.0), |c| (c.t, c.probe.duration));
				if let Some(end) = self.engine.edit.cut_at(next) {
					self.looping = true;
					self.seek(if end >= duration - 0.05 { 0.0 } else { end });
				}
			}
			// the end: round again from the start, as the site does
			Ok(false) => {
				self.looping = true;
				self.seek(0.0);
			}
			Err(e) => {
				self.status.error = Some(e.to_string());
				self.playing = false;
			}
		}
		self.next_due += Duration::from_secs_f64(1.0 / (fps * speed));
		let now = Instant::now();
		// fallen well behind (a slow look, a seek): carry on from now rather than rushing to catch up
		if now > self.next_due + Duration::from_millis(250) {
			self.next_due = now;
		}
	}

	fn redraw(&mut self) {
		self.dirty = false;
		let Some(c) = self.clip.as_ref() else { return };
		if c.frame.w == 0 {
			return;
		}
		let t = (c.t - 1.0 / c.probe.fps).max(0.0);
		// paused with the follow cam: the same window again
		let frame = if self.engine.look.follow && c.crop.is_none() && self.seen.w > 0 {
			self.seen.clone()
		} else {
			c.frame.clone()
		};
		if let Err(e) = self.engine.redraw(&frame, t, 0.04, &mut self.out) {
			self.status.error = Some(e.to_string());
		}
		self.status.counts = self.engine.counts();
		self.note_kinds();
		self.push();
	}

	fn note_kinds(&mut self) {
		for o in self.engine.named() {
			if !self.status.seen.contains(&o.cls) {
				self.status.seen.push(o.cls.clone());
				self.status.seen.sort();
			}
		}
	}

	fn push(&mut self) {
		// comparing: the picture as the look gets it, nothing drawn
		let raw = if self.engine.look.follow && self.seen.w > 0 {
			&self.seen
		} else {
			self.clip.as_ref().map_or(&self.out, |c| &c.frame)
		};
		let f = if self.bare && raw.w > 0 { raw } else { &self.out };
		if f.w == 0 {
			return;
		}
		let mut bgra = f.rgba.clone();
		if self.guides {
			let t = self.clip.as_ref().map_or(0.0, |c| (c.t - 1.0 / c.probe.fps).max(0.0));
			guides(&mut bgra, f.w as usize, f.h as usize, &self.engine.edit, self.engine.view, self.draft, t);
		}
		for p in bgra.chunks_exact_mut(4) {
			p.swap(0, 2);
		}
		self.pushed += 1;
		let _ = self.surface.push_bytes(
			ByteFrame {
				width: f.w,
				height: f.h,
				format: Format::Bgra,
				color: ColorInfo::default(),
				planes: [bgra, Vec::new()],
			},
			self.pushed,
		);
	}

	fn export(&mut self, mut es: Vec<Export>) {
		if es.is_empty() {
			return;
		}
		// what looking ahead found, if it's been over all of this video, saves looking again
		if let Some(a) = &self.ahead
			&& a.done && a.failed.is_none()
		{
			let record = Arc::new(a.record.clone());
			for e in es.iter_mut().filter(|e| e.input == a.input) {
				e.record = Some(record.clone());
			}
		}
		{
			let mut s = self.shared.lock().unwrap();
			if s.export.is_some() {
				return;
			}
			s.export = Some((0, 1));
			s.export_part = (1, es.len());
			s.exported = None;
			s.wrote = None;
		}
		self.stop_export.store(false, Ordering::Relaxed);
		(self.wake)();
		let (shared, wake, stop) = (self.shared.clone(), self.wake.clone(), self.stop_export.clone());
		std::thread::Builder::new()
			.name("export".into())
			.spawn(move || {
				let began = Instant::now();
				let mut said = Instant::now();
				let mut wrote: Vec<String> = Vec::new();
				let mut failed = None;
				for (i, e) in es.iter().enumerate() {
					shared.lock().unwrap().export_part = (i + 1, es.len());
					let result = export::run(e, &mut |n, total| {
						if said.elapsed() >= Duration::from_millis(200) {
							said = Instant::now();
							shared.lock().unwrap().export = Some((n, total));
							wake();
						}
						!stop.load(Ordering::Relaxed)
					});
					match result {
						Ok(p) => wrote.push(format!("{} ({}x{})", e.out.display(), p.w, p.h)),
						Err(err) => {
							failed = Some(format!("{err}"));
							break;
						}
					}
				}
				let mut s = shared.lock().unwrap();
				s.export = None;
				s.wrote = (!wrote.is_empty()).then(|| es[0].out.clone());
				s.exported = Some(match failed {
					None => format!("wrote {} in {:.0}s", wrote.join(", "), began.elapsed().as_secs_f32()),
					Some(err) if wrote.is_empty() => format!("export stopped: {err}"),
					Some(err) => format!("wrote {}, then stopped: {err}", wrote.join(", ")),
				});
				drop(s);
				wake();
			})
			.expect("a thread to export");
	}

	/// Tell the window how things are: straight away when `now`, otherwise ten times a second at most.
	fn publish(&mut self, now: bool) {
		let since = self.shown_since.elapsed();
		if since >= Duration::from_secs(1) {
			self.status.fps = (self.shown as f64 / since.as_secs_f64()).round() as u32;
			self.shown = 0;
			self.shown_since = Instant::now();
		}
		if let Some(e) = self.engine.trouble.take() {
			self.status.error = Some(e);
		}
		self.status.playing = self.playing;
		if !now && self.published.elapsed() < Duration::from_millis(100) {
			return;
		}
		self.published = Instant::now();
		let mut s = self.shared.lock().unwrap();
		// the export thread keeps its own two fields up to date
		let next =
			Status { export: s.export, exported: s.exported.clone(), wrote: s.wrote.clone(), ..self.status.clone() };
		if *s != next {
			*s = next;
			drop(s);
			(self.wake)();
		}
	}
}

/// The edit's zones and picks over the preview, so what's closed off can be seen: closed parts
/// darkened with a red edge, a zone being dragged out edged in yellow, and the picks near this moment
/// as yellow crosses.
fn guides(px: &mut [u8], w: usize, h: usize, edit: &Edit, view: View, draft: Option<Zone>, t: f64) {
	if w == 0 || h == 0 {
		return;
	}
	let open = edit.allowed(view, w, h);
	if !open.is_empty() {
		for (p, ok) in px.chunks_exact_mut(4).zip(&open) {
			if !ok {
				for c in &mut p[..3] {
					*c = (*c as u32 * 2 / 5) as u8;
				}
			}
		}
	}
	let mut edge = |z: &Zone, colour: [u8; 3]| {
		let (x0, y0) = to_view(view, z.x, z.y);
		let (x1, y1) = to_view(view, z.x + z.w, z.y + z.h);
		let to = |v: f32, n: usize| ((v * n as f32).round() as isize).clamp(0, n as isize - 1) as usize;
		let (x0, x1, y0, y1) = (to(x0.min(x1), w), to(x0.max(x1), w), to(y0.min(y1), h), to(y0.max(y1), h));
		for y in y0..=y1 {
			for x in x0..=x1 {
				if x < x0 + 2 || x + 2 > x1 || y < y0 + 2 || y + 2 > y1 {
					px[(y * w + x) * 4..][..3].copy_from_slice(&colour);
				}
			}
		}
	};
	for z in &edit.zones {
		edge(z, if z.mode == ZoneMode::Only { [0xfa, 0xbd, 0x2f] } else { [0xfb, 0x49, 0x34] });
	}
	if let Some(z) = draft {
		edge(&z, [0xfa, 0xbd, 0x2f]);
	}
	for k in edit.picks.iter().filter(|k| (k.t - t).abs() < 0.6) {
		let (x, y) = to_view(view, k.x, k.y);
		let (cx, cy) = ((x * w as f32) as isize, (y * h as f32) as isize);
		for d in -10isize..=10 {
			for (x, y) in [(cx + d, cy), (cx, cy + d), (cx + d, cy + 1), (cx + 1, cy + d)] {
				if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && d.abs() > 2 {
					px[(y as usize * w + x as usize) * 4..][..3].copy_from_slice(&[0xfa, 0xbd, 0x2f]);
				}
			}
		}
	}
}
