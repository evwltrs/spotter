//! Ties it together: for each new frame of the video, find and follow the points, ask the detector
//! what things are when it's due, grade the picture and draw the overlay.
//!
//! There are two ways to recognise things. An export uses `detector` and waits for it on the frames
//! it looks at, so every frame comes out the same however long it takes. A preview uses `live`, a
//! detector on its own thread, the way the site works: the picture never waits, and what the detector
//! says arrives a frame or two later and is lined up with where things have moved to since.
//!
//! Time comes in two kinds. `t` is where the frame is in the source (the {clock} on the frame).
//! `dt` is how long the frame is on screen in the output, which is what the site's timings were in:
//! how often the detector looks, and how quickly the boxes fade between movement and the detector.

use crate::Frame;
use crate::detector::Detector;
use crate::edit::{Chooser, Edit, View, WHOLE};
use crate::grade::Grade;
use crate::look::Look;
use crate::matte::{Matte, Moving};
use crate::objects::{Detection, ObjectTrack, ObjectTracks, UNNUMBERED};
use crate::overlay::{Scene, draw};
use crate::tape::Tape;
use crate::tracking::{Finder, Follower};
use anyhow::Result;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex};

struct Job {
	frame: Frame,
	at: u64,
	confidence: f32,
	kinds: String,
}

/// A detector on its own thread. It takes the newest frame it's given and drops any it didn't get to.
pub struct LiveDetector {
	slot: Arc<(Mutex<Option<Job>>, Condvar)>,
	results: Receiver<Result<(Vec<Detection>, u64), String>>,
	busy: Arc<AtomicBool>,
	stop: Arc<AtomicBool>,
	/// how long the last look took, in milliseconds
	pub ms: Arc<Mutex<f32>>,
}

impl LiveDetector {
	pub fn spawn(mut d: Detector) -> Self {
		let slot: Arc<(Mutex<Option<Job>>, Condvar)> = Arc::default();
		let busy = Arc::new(AtomicBool::new(false));
		let stop = Arc::new(AtomicBool::new(false));
		let ms = Arc::new(Mutex::new(0.0));
		let (tx, results): (Sender<_>, _) = channel();
		{
			let (slot, busy, stop, ms) = (slot.clone(), busy.clone(), stop.clone(), ms.clone());
			std::thread::Builder::new()
				.name("detector".into())
				.spawn(move || {
					loop {
						let job = {
							let (lock, ready) = &*slot;
							let mut g = lock.lock().unwrap();
							while g.is_none() && !stop.load(Ordering::Relaxed) {
								g = ready.wait(g).unwrap();
							}
							if stop.load(Ordering::Relaxed) {
								return;
							}
							g.take().unwrap()
						};
						let out = d
							.detect(&job.frame, job.confidence, &job.kinds)
							.map(|f| (f, job.at))
							.map_err(|e| e.to_string());
						*ms.lock().unwrap() = d.ms;
						busy.store(false, Ordering::Release);
						if tx.send(out).is_err() {
							return;
						}
					}
				})
				.expect("a thread for the detector");
		}
		Self { slot, results, busy, stop, ms }
	}

	/// Whether a look is under way.
	pub fn busy(&self) -> bool {
		self.busy.load(Ordering::Acquire)
	}

	fn submit(&self, frame: &Frame, at: u64, l: &Look) {
		self.busy.store(true, Ordering::Release);
		let (lock, ready) = &*self.slot;
		*lock.lock().unwrap() =
			Some(Job { frame: frame.clone(), at, confidence: l.confidence, kinds: l.object_kinds.clone() });
		ready.notify_one();
	}

	fn poll(&self) -> Option<Result<(Vec<Detection>, u64), String>> {
		self.results.try_recv().ok()
	}
}

impl Drop for LiveDetector {
	fn drop(&mut self) {
		self.stop.store(true, Ordering::Relaxed);
		self.slot.1.notify_all();
	}
}

pub struct Engine {
	pub look: Look,
	/// shown in the {file} token
	pub name: String,
	/// recognises things, waited for (exports)
	pub detector: Option<Detector>,
	/// recognises things on its own thread, never waited for (previews)
	pub live: Option<LiveDetector>,
	/// gives the true shapes of things, for layers that ask for outlines
	pub outliner: Option<crate::outline::Outliner>,
	/// how far away each part of the picture is, for looks that use depth
	pub depther: Option<crate::depth::Depther>,
	/// what each part of the scene is, for the words look
	pub scener: Option<crate::scene::Scener>,
	/// what those three said about each frame, kept to play it again without them (the window's)
	pub recall: Option<crate::recall::Recall>,
	/// the window's: those three on a thread of their own (background.rs), the last of its answers
	/// taken in, and the frame (by count) last handed to it
	pub background: Option<crate::background::Background>,
	bg_seen: u64,
	bg_frame: u64,
	/// the frame the scene model last looked at, to know when what it eased toward is old
	scene_ran: Option<i64>,
	/// the last thing that went wrong while drawing (the detector, a look mixed in, depth)
	pub trouble: Option<String>,
	/// frames a second the output plays at, for the {fps} token
	pub fps: f64,
	/// the {date} token's time at the start, in seconds since 1970
	pub started: f64,
	/// what's followed, where and when (see edit.rs)
	pub edit: Edit,
	/// where the frames it's given sit in the source, 0-1 across it, for the edit's places
	pub view: View,
	/// which things the edit's picks mean
	pub chooser: Chooser,
	/// with everything drawn, the main thing, for a follow cam to aim at
	aim: Chooser,
	/// the things' copies and trails left behind
	echo: crate::echo::Echo,
	/// the frame being drawn is new, not the same one again
	fresh_frame: bool,
	insets: crate::insets::Insets,
	glyphs: crate::glyphs::Glyphs,
	/// each thing's outline, kept steady from frame to frame
	steady: crate::outline::Steady,
	finder: Finder,
	follow: Follower,
	objects: ObjectTracks,
	grade: Grade,
	/// 0 is all movement, 1 all detector; moves between them over a second or two
	blend: f32,
	/// output time when the detector last looked
	looked: Option<f64>,
	/// output time so far
	clock: f64,
	/// frames worked out since the start
	frame: u64,
	/// the source time of the frame being drawn
	t: f64,
	/// the settings that change what's found, as they were last time it was worked out
	finding: String,
	/// what's in the clip, worked out ahead (tape.rs); takes over from the detector where it covers
	tape: Option<Arc<Tape>>,
	/// the tape's frame for the frame being drawn
	tape_at: Option<usize>,
	/// the tape's frame whose things are laid out now (None: none are, or they need laying again)
	tape_laid: Option<usize>,
	/// the number each of the tape's things got when it was first shown
	tape_ids: Vec<i64>,
	/// the tape's things that are in `objects` now, in the same order
	tape_alive: Vec<usize>,
	/// the tape's things the edit's picks mean
	tape_picked: Vec<usize>,
	/// where the clip's sound hits, in source seconds (beats.rs), for looks on the beat
	beats: Option<Arc<Vec<f64>>>,
	/// the source time of the frame before, to tell whether a beat has come since
	last_t: Option<f64>,
	/// mixing: the other look's engine, what its look was made from (to know when to make it again),
	/// and what it drew this frame
	with: Option<Box<Engine>>,
	with_key: String,
	with_frame: Frame,
	/// the things' matte this frame, kept for mixing through after the other look's drawn
	mix_matte: Option<Matte>,
	/// datamosh, tape tears, trichrome and stutter, and whether a beat came this frame (None: no beats)
	glitch: crate::glitch::Glitch,
	beat_hit: Option<bool>,
	/// drawing only the marks (boxes, labels, text, insets) onto a picture that's already made: the
	/// other look's, for mixing over or under
	marks_only: bool,
}

impl Engine {
	pub fn new(look: Look) -> Self {
		let started =
			std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
		Self {
			look,
			name: String::new(),
			detector: None,
			live: None,
			outliner: None,
			depther: None,
			scener: None,
			recall: None,
			background: None,
			bg_seen: 0,
			bg_frame: u64::MAX,
			scene_ran: None,
			trouble: None,
			fps: 30.0,
			started,
			edit: Edit::default(),
			view: WHOLE,
			chooser: Chooser::default(),
			aim: Chooser::default(),
			echo: Default::default(),
			fresh_frame: false,
			insets: Default::default(),
			glyphs: Default::default(),
			steady: Default::default(),
			finder: Finder::new(),
			follow: Follower::new(),
			objects: ObjectTracks::default(),
			grade: Grade::new(),
			blend: 0.0,
			looked: None,
			clock: 0.0,
			frame: 0,
			t: 0.0,
			finding: String::new(),
			tape: None,
			tape_at: None,
			tape_laid: None,
			tape_ids: Vec::new(),
			tape_alive: Vec::new(),
			tape_picked: Vec::new(),
			beats: None,
			last_t: None,
			with: None,
			with_key: String::new(),
			with_frame: Frame::default(),
			mix_matte: None,
			glitch: Default::default(),
			beat_hit: None,
			marks_only: false,
		}
	}

	/// The picture jumped (a seek, a new video): the last frame means nothing now.
	pub fn jumped(&mut self) {
		self.glitch.jumped();
		self.finder.reset();
		self.follow.clear();
		self.objects.clear();
		self.chooser.jumped();
		self.aim.jumped();
		self.echo.clear();
		self.insets.clear();
		self.steady.clear();
		if let Some(d) = self.depther.as_mut() {
			d.reset();
		}
		if let Some(s) = self.scener.as_mut() {
			s.reset();
		}
		self.looked = None;
		self.last_t = None;
		self.tape_at = None;
		self.tape_laid = None;
		// whatever the models' thread is working on belongs to the old picture too
		if let Some(b) = &self.background {
			self.bg_seen = b.sent();
		}
		// whatever the live detector is looking at now belongs to the old picture
		self.frame += 1000;
	}

	/// What's in the clip worked out ahead, or None to go back to the detector. A thing the last tape
	/// had too (one starting on the same frame in the same place: a tape worked out again with other
	/// settings, or from more of the clip) keeps its number; the rest get new ones as they're shown.
	pub fn set_tape(&mut self, tape: Option<Arc<Tape>>) {
		let mut ids = vec![UNNUMBERED; tape.as_ref().map_or(0, |t| t.things.len())];
		if let (Some(old), Some(new)) = (&self.tape, &tape) {
			for (k, t) in new.things.iter().enumerate() {
				let same = |o: &crate::tape::Thing| {
					let (a, b) = (o.boxes[0], t.boxes[0]);
					o.first == t.first
						&& (a.x - b.x).abs() + (a.y - b.y).abs() + (a.w - b.w).abs() + (a.h - b.h).abs() < 0.04
				};
				if let Some(j) = old.things.iter().position(same) {
					ids[k] = self.tape_ids[j];
				}
			}
		}
		self.tape_ids = ids;
		self.tape_picked = tape.as_ref().map(|t| t.picked(&self.edit)).unwrap_or_default();
		self.tape = tape;
		self.tape_laid = None;
		self.tape_alive.clear();
		if self.tape.is_none() {
			self.objects.clear();
			self.objects.pinned = false;
		}
		self.find_on_tape(false);
	}

	/// Where a camera should be aiming when the footage has just cut to a new shot at `t`, 0-1 across
	/// the source (None if it hasn't, or the tape doesn't know): what's picked there, or the biggest thing.
	pub fn aim_at_cut(&self, t: f64) -> Option<crate::Rect> {
		let tape = self.tape.as_ref()?;
		let i = tape.at(t)?;
		if !tape.cuts[i] || i == 0 {
			return None;
		}
		tape.aim(i, &self.tape_picked)
	}

	pub fn tape(&self) -> Option<&Arc<Tape>> {
		self.tape.as_ref()
	}

	/// Where the clip's sound hits (source seconds), for looks on the beat.
	pub fn beats(&self) -> Option<&Arc<Vec<f64>>> {
		self.beats.as_ref()
	}

	pub fn set_beats(&mut self, beats: Option<Arc<Vec<f64>>>) {
		self.beats = beats;
	}

	/// The beats the look uses: every `beat_every`th, when it's on the beat and there are any.
	fn used_beats(&self, l: &Look) -> Option<Vec<f64>> {
		let b = self.beats.as_ref().filter(|b| l.beat && !b.is_empty())?;
		let every = (l.beat_every.round() as usize).max(1);
		Some(b.iter().step_by(every).copied().collect())
	}

	/// Whether the tape covers the frame being drawn.
	pub fn on_tape(&self) -> bool {
		self.tape_at.is_some()
	}

	/// Find the frame being drawn on the tape. With `cuts`, a new shot since the last frame starts the
	/// movement over, as a jump does, so nothing from the last shot is carried into this one.
	fn find_on_tape(&mut self, cuts: bool) {
		let Some(tape) = self.tape.clone() else {
			self.tape_at = None;
			return;
		};
		let at = tape.at(self.t);
		if cuts
			&& let (Some(i), Some(last)) = (at, self.tape_at)
			&& tape.cut_in(last, i)
		{
			self.finder.reset();
			self.follow.clear();
			self.objects.clear();
			self.echo.clear();
			self.insets.clear();
			self.steady.clear();
			if let Some(d) = self.depther.as_mut() {
				d.reset();
			}
			if let Some(s) = self.scener.as_mut() {
				s.reset();
			}
			self.tape_laid = None;
		}
		self.tape_at = at;
	}

	/// Put the tape's things on this frame into `objects`, where the drawing takes them from, mapped
	/// into the view. Says whether anything changed.
	fn lay_tape(&mut self, sx: f32, sy: f32) -> bool {
		let (Some(tape), Some(i)) = (self.tape.clone(), self.tape_at) else { return false };
		if self.tape_laid == Some(i) {
			return false;
		}
		self.tape_laid = Some(i);
		let old = std::mem::take(&mut self.objects.list);
		self.objects.pinned = true;
		self.tape_alive.clear();
		let [vx, vy, vw, vh] = self.view;
		for (k, r, fade) in tape.here(i) {
			// into the view, and cut to it; a thing mostly outside it isn't there
			let (x0, y0, x1, y1) = ((r.x - vx) / vw, (r.y - vy) / vh, (r.x + r.w - vx) / vw, (r.y + r.h - vy) / vh);
			let (cx0, cy0, cx1, cy1) = (x0.max(0.0), y0.max(0.0), x1.min(1.0), y1.min(1.0));
			if cx1 <= cx0 || cy1 <= cy0 || (cx1 - cx0) * (cy1 - cy0) < 0.25 * (x1 - x0) * (y1 - y0) {
				continue;
			}
			let id = self.tape_ids[k];
			let was = (id != UNNUMBERED).then(|| old.iter().find(|o| o.id == id)).flatten();
			let thing = &tape.things[k];
			self.objects.list.push(ObjectTrack {
				id,
				x: cx0 * sx,
				y: cy0 * sy,
				w: (cx1 - cx0) * sx,
				h: (cy1 - cy0) * sy,
				miss: 0,
				cls: thing.cls.clone(),
				conf: thing.conf,
				hits: u32::MAX / 2,
				vx: 0.0,
				vy: 0.0,
				frames: 0,
				votes: vec![(thing.cls.clone(), thing.conf)],
				shown: was.is_some_and(|o| o.shown),
				fade,
				path: was.and_then(|o| o.path.clone()),
			});
			self.tape_alive.push(k);
		}
		true
	}

	/// Note the numbers the tape's things were given, and hold the picked ones from their first frame.
	fn tape_numbers(&mut self) {
		for (j, &k) in self.tape_alive.iter().enumerate() {
			if let Some(o) = self.objects.list.get(j) {
				self.tape_ids[k] = o.id;
			}
		}
		for &k in &self.tape_picked {
			let id = self.tape_ids[k];
			if id != UNNUMBERED && !self.chooser.ahead.contains(&id) {
				self.chooser.ahead.push(id);
			}
		}
	}

	/// Let go of everything kept to save time (what the models said, glitches' past pictures), when
	/// memory's running short.
	pub fn shed(&mut self) {
		if let Some(r) = self.recall.as_mut() {
			r.keep_for("");
		}
		self.glitch.jumped();
		self.echo.clear();
		if let Some(w) = self.with.as_mut() {
			w.glitch.jumped();
			w.echo.clear();
		}
	}

	/// Forget every track and start the numbers over.
	pub fn restart_ids(&mut self) {
		self.objects.counted = 0;
		self.follow.restart();
		self.objects.clear();
	}

	/// The detector's things on show now.
	pub fn named(&self) -> &[ObjectTrack] {
		&self.follow.named
	}

	/// What a follow cam should keep in frame: everything chosen, together, 0-1 across the frame.
	/// Needs the edit to choose (anything but keeping everything).
	pub fn target(&self) -> Option<crate::Rect> {
		let (fw, fh) = (self.finder.w as f32, self.finder.h as f32);
		if fw == 0.0 {
			return None;
		}
		let unit = fw.min(fh);
		let rects = if Chooser::active(&self.edit) { self.chooser.rects() } else { self.aim.rects() };
		let first = rects.first()?;
		let (mut x0, mut y0, mut x1, mut y1) = (first.x, first.y, first.x + first.w, first.y + first.h);
		for r in &rects[1..] {
			x0 = x0.min(r.x);
			y0 = y0.min(r.y);
			x1 = x1.max(r.x + r.w);
			y1 = y1.max(r.y + r.h);
		}
		let (ux, uy) = (fw / unit, fh / unit);
		Some(crate::Rect { x: x0 / ux, y: y0 / uy, w: (x1 - x0) / ux, h: (y1 - y0) / uy })
	}

	/// How many things are on show: points, big boxes and named objects.
	pub fn counts(&self) -> (usize, usize, usize) {
		(self.follow.tracks.len(), self.follow.regions.len(), self.follow.named.len())
	}

	/// Work out one new frame of the source, `src` at the output's size, and draw the result into `out`.
	pub fn step(&mut self, src: &Frame, t: f64, dt: f64, out: &mut Frame) -> Result<()> {
		self.t = t;
		self.clock += dt;
		self.frame += 1;
		self.find_on_tape(true);
		let l = self.look.clone();
		let mut lap = crate::perf::Lap::start();
		self.finder.grab(src, &l);
		lap.mark("movement copy");
		self.track(src, &l, dt, true)?;
		lap.mark("tracking");
		self.draw_all(src, &l, t, dt, out, true);
		lap.mark("drawing");
		Ok(())
	}

	/// Everything after tracking: this look, the one mixed in with it, and the glitches after.
	fn draw_all(&mut self, src: &Frame, l: &Look, t: f64, dt: f64, out: &mut Frame, fresh: bool) {
		// an unmix stretch: this look alone, the one mixed in left out (faded at its ends)
		let alone = self.edit.unmixed(t);
		let mixing = !l.mix.trim().is_empty() && alone < 1.0;
		// over: this look all done, then the other look's marks on top of it
		if mixing && l.mix_how == "over" {
			self.fresh_frame = fresh;
			self.render(src, l, out);
			self.after(out, l, t, dt, fresh);
			let base = (alone > 0.0).then(|| out.clone());
			self.step_with(src, t, dt, fresh, Some(out));
			if let Some(b) = base {
				fade(out, &b, alone);
			}
			return;
		}
		// under: the other look all done, then this look's marks on top of it
		if mixing && l.mix_how == "under" && alone < 0.5 {
			self.step_with(src, t, dt, fresh, None);
			self.fresh_frame = fresh;
			if self.with_frame.w == src.w && self.with_frame.h == src.h {
				out.clone_from(&self.with_frame);
				let was = std::mem::replace(&mut self.marks_only, true);
				self.render(src, l, out);
				self.marks_only = was;
			} else {
				self.render(src, l, out);
			}
			self.after(out, l, t, dt, fresh);
			return;
		}
		// the rest: the look with more things draws first, so the other finds every outline it
		// needs already made (in an unmix stretch, the one mixed in waits)
		let left_out = !l.mix.trim().is_empty() && !mixing;
		let first = self.draws_first();
		if !first && !left_out {
			self.step_with(src, t, dt, fresh, None);
		}
		self.fresh_frame = fresh;
		self.render(src, l, out);
		if first && !left_out {
			self.step_with(src, t, dt, fresh, None);
		}
		if !left_out {
			let base = (alone > 0.0).then(|| out.clone());
			self.mix_in(out, l);
			if let Some(b) = base {
				fade(out, &b, alone);
			}
		}
		self.after(out, l, t, dt, fresh);
	}

	/// The glitches that come after everything (glitch.rs), unless only marks are being drawn.
	fn after(&mut self, out: &mut Frame, l: &Look, t: f64, dt: f64, fresh: bool) {
		if self.marks_only {
			return;
		}
		let on = self.biggest();
		let hit = if fresh { self.beat_hit } else { None };
		self.glitch.after(out, l, t, dt, hit, fresh, on);
	}

	/// The middle of the biggest thing on show, 0-1 across and down the picture.
	fn biggest(&self) -> Option<(f32, f32)> {
		let (fw, fh) = (self.finder.w as f32, self.finder.h as f32);
		if fw <= 0.0 || fh <= 0.0 {
			return None;
		}
		let unit = fw.min(fh);
		self.follow
			.named
			.iter()
			.max_by(|a, b| (a.w * a.h).total_cmp(&(b.w * b.h)))
			.map(|o| ((o.x + o.w / 2.0) * unit / fw, (o.y + o.h / 2.0) * unit / fh))
	}

	/// Whether this look draws before the one mixed in with it: it shows as many things or more.
	fn draws_first(&self) -> bool {
		self.with.as_ref().is_none_or(|w| self.look.objects >= w.look.objects)
	}

	/// Mixing: the other look's picture put together with this one's, as the look says.
	fn mix_in(&mut self, out: &mut Frame, l: &Look) {
		let matte = self.mix_matte.take();
		if self.with.is_none() || l.mix.trim().is_empty() || self.with_frame.w != out.w || self.with_frame.h != out.h {
			return;
		}
		let beats = self.used_beats(l);
		put_together(
			out,
			&self.with_frame,
			&l.mix_how,
			l.mix_at,
			l.mix_time,
			l.mix_jitter,
			self.clock,
			self.t,
			matte.as_ref(),
			beats.as_deref(),
		);
	}

	/// Follow one new frame without drawing it, for a pass over a clip that only needs to know where
	/// things go (what an edit's picks mean). Tracks exactly as `step` does.
	pub fn follow_only(&mut self, src: &Frame, t: f64, dt: f64) -> Result<()> {
		self.t = t;
		self.clock += dt;
		self.frame += 1;
		self.find_on_tape(true);
		let l = self.look.clone();
		self.finder.grab(src, &l);
		self.track(src, &l, dt, true)
	}

	/// Draw the same frame again: paused, with a setting changed or something new from the detector.
	/// It's only tracked again if a setting that changes what's found has moved.
	pub fn redraw(&mut self, src: &Frame, t: f64, dt: f64, out: &mut Frame) -> Result<()> {
		self.t = t;
		self.clock += dt;
		self.find_on_tape(false);
		let l = self.look.clone();
		if self.finder.w == 0 {
			self.frame += 1;
			self.finder.grab(src, &l);
		}
		self.track(src, &l, dt, false)?;
		// a paused picture gets no frames of its own for things to fade in over
		self.objects.settle();
		for o in &mut self.follow.named {
			o.fade = 1.0;
		}
		self.draw_all(src, &l, t, dt, out, false);
		Ok(())
	}

	/// Mixing: make the other look's engine (or give it a new look) when the mix changes, keep it in
	/// step with this one, and have it draw this frame too, lent this one's models and what they've
	/// said.
	fn step_with(&mut self, src: &Frame, t: f64, dt: f64, fresh: bool, onto: Option<&mut Frame>) {
		let l = &self.look;
		if l.mix.trim().is_empty() {
			self.with = None;
			self.with_key.clear();
			return;
		}
		let key = format!(
			"{}|{}|{}|{}|{}",
			l.mix.trim(),
			l.finding_key(),
			l.ahead,
			l.ahead_smooth,
			serde_json::to_string(&l.mix_set).unwrap_or_default()
		);
		if key != self.with_key {
			self.with_key = key;
			match l.mixed() {
				Some(m) => match self.with.as_mut() {
					Some(w) => w.look = m,
					None => {
						let mut w = Engine::new(m);
						w.started = self.started;
						self.with = Some(Box::new(w));
					}
				},
				None => {
					self.with = None;
					self.trouble = Some(format!("no look called {} to mix with", l.mix.trim()));
				}
			}
		}
		let name = l.mix.trim().to_string();
		let Some(w) = self.with.as_mut() else { return };
		w.fps = self.fps;
		w.view = self.view;
		if w.name != self.name {
			w.name.clone_from(&self.name);
		}
		if w.edit != self.edit {
			w.edit = self.edit.clone();
		}
		let same = |a: Option<&Arc<Tape>>, b: Option<&Arc<Tape>>| match (a, b) {
			(Some(a), Some(b)) => Arc::ptr_eq(a, b),
			(None, None) => true,
			_ => false,
		};
		if !same(w.tape.as_ref(), self.tape.as_ref()) {
			w.set_tape(self.tape.clone());
		}
		let beats_same = match (&w.beats, &self.beats) {
			(Some(a), Some(b)) => Arc::ptr_eq(a, b),
			(None, None) => true,
			_ => false,
		};
		if !beats_same {
			w.beats = self.beats.clone();
		}
		// one set of models, lent
		std::mem::swap(&mut self.outliner, &mut w.outliner);
		std::mem::swap(&mut self.depther, &mut w.depther);
		std::mem::swap(&mut self.scener, &mut w.scener);
		std::mem::swap(&mut self.recall, &mut w.recall);
		std::mem::swap(&mut self.detector, &mut w.detector);
		std::mem::swap(&mut self.background, &mut w.background);
		let r = match onto {
			// over: only its marks, onto this look's finished picture
			Some(o) => {
				w.marks_only = true;
				let r = if fresh { w.step(src, t, dt, o) } else { w.redraw(src, t, dt, o) };
				w.marks_only = false;
				r
			}
			None => {
				if fresh {
					w.step(src, t, dt, &mut self.with_frame)
				} else {
					w.redraw(src, t, dt, &mut self.with_frame)
				}
			}
		};
		std::mem::swap(&mut self.outliner, &mut w.outliner);
		std::mem::swap(&mut self.depther, &mut w.depther);
		std::mem::swap(&mut self.scener, &mut w.scener);
		std::mem::swap(&mut self.recall, &mut w.recall);
		std::mem::swap(&mut self.detector, &mut w.detector);
		std::mem::swap(&mut self.background, &mut w.background);
		if let Some(e) = w.trouble.take() {
			self.trouble = Some(format!("{name} (mixed in): {e}"));
		}
		if let Err(e) = r {
			self.trouble = Some(format!("{name} (mixed in): {e}"));
		}
	}

	/// Whether a paused picture would change if drawn again: the boxes are still fading between
	/// movement and the detector, or the live detector is still to answer.
	pub fn settling(&self) -> bool {
		let recognising = self.look.detect && self.live.is_some();
		(self.blend > 0.0 && self.blend < 1.0)
			|| (recognising && self.live.as_ref().is_some_and(LiveDetector::busy))
			|| self
				.background
				.as_ref()
				.is_some_and(|b| (b.busy() && self.bg_frame == self.frame) || b.newer(self.bg_seen).is_some())
	}

	/// A model as it comes (the window loads them on threads of their own): to the models' thread
	/// when there is one (keeping a shell to draw with), else kept here.
	pub fn adopt_outliner(&mut self, o: crate::outline::Outliner) {
		match self.background.as_mut() {
			Some(b) => b.give_outliner(o),
			None => self.outliner = Some(o),
		}
	}

	pub fn adopt_depther(&mut self, d: crate::depth::Depther) {
		match self.background.as_mut() {
			Some(b) => {
				self.depther = Some(d.shell());
				b.give_depther(d);
			}
			None => self.depther = Some(d),
		}
	}

	pub fn adopt_scener(&mut self, s: crate::scene::Scener) {
		match self.background.as_mut() {
			Some(b) => {
				self.scener = Some(s.shell());
				b.give_scener(s);
			}
			None => self.scener = Some(s),
		}
	}

	/// Whether outlines can be had, here or on the models' thread.
	pub fn has_outliner(&self) -> bool {
		self.outliner.is_some() || self.background.as_ref().is_some_and(|b| b.has_outliner)
	}

	/// The models' thread's newest answers, taken in: outlines into the steadying (for the boxes they
	/// were worked out for), depth and the scene into the shells, and all of it kept for the frame
	/// it's for.
	fn take_done(&mut self) {
		let Some(done) = self.background.as_ref().and_then(|b| b.newer(self.bg_seen)) else { return };
		self.bg_seen = done.seq;
		if let Some(offers) = &done.offers {
			self.steady.shapes(offers, &done.boxes, &done.ids, done.w, done.h, self.frame, true);
			if let (Some(a), Some(r)) = (done.at, self.recall.as_mut()) {
				r.keep_shapes(a, &done.boxes, &self.steady.last);
			}
		}
		if let Some((near, range)) = &done.depth {
			if let Some(d) = self.depther.as_mut() {
				d.set(near, *range);
			}
			if let (Some(a), Some(r)) = (done.at, self.recall.as_mut()) {
				r.keep_depth(a, near, *range);
			}
		}
		if let Some((grid, kind)) = &done.scene {
			if let Some(s) = self.scener.as_mut() {
				s.recall(*grid, kind);
			}
			if let (Some(a), Some(r)) = (done.at, self.recall.as_mut()) {
				r.keep_scene(a, *grid, kind);
			}
		}
		if let Some(e) = &done.error {
			self.trouble = Some(e.clone());
		}
	}

	fn track(&mut self, src: &Frame, l: &Look, dt: f64, fresh: bool) -> Result<()> {
		let unit = self.finder.w.min(self.finder.h) as f32;
		let (fw, fh) = (self.finder.w as f32, self.finder.h as f32);
		// where the tape covers this frame it says what's here, and the detectors rest
		let taped = l.detect && self.tape.is_some() && self.tape_at.is_some();
		let recognising = l.detect && (taped || self.detector.is_some() || self.live.is_some());
		let finding = l.finding_key();
		let changed = finding != self.finding;
		let mut news = false;

		// what the live detector has said since last time, lined up with where things are now
		if let Some(live) = &self.live {
			while let Some(said) = live.poll() {
				match said {
					Ok((mut found, at)) if recognising && !taped && at <= self.frame && self.frame - at < 1000 => {
						found.retain(|d| self.edit.admits(d, self.view));
						let lag = (self.frame - at) as f32;
						self.objects.ingest(&found, l, fw / unit, fh / unit, lag);
						news = true;
					}
					Ok(_) => {}
					Err(e) => self.trouble = Some(format!("the detector: {e}")),
				}
			}
		}

		// an export's detector looks at this frame now if it's due, but what it says is taken in once
		// the tracks have moved on to this frame, so it lines up with where things are
		let mut found = None;
		if taped {
			news |= self.lay_tape(fw / unit, fh / unit);
		} else if recognising {
			let due = self.looked.is_none_or(|at| (self.clock - at) * 1000.0 >= l.detect_every as f64 - 1e-6);
			if let Some(d) = self.detector.as_mut() {
				if fresh && due {
					let mut f = d.detect(src, l.confidence, &l.object_kinds)?;
					f.retain(|d| self.edit.admits(d, self.view));
					found = Some(f);
					self.looked = Some(self.clock);
				}
			} else if let Some(live) = &self.live
				&& !live.busy()
				&& ((fresh && due) || (!fresh && changed))
			{
				live.submit(src, self.frame, l);
				self.looked = Some(self.clock);
			}
		} else if !self.objects.list.is_empty() {
			self.objects.clear();
			news = true;
		}
		if !taped && self.objects.pinned {
			// off the tape (past what it covers, or it's gone): back to following the detector's way
			self.objects.clear();
			self.objects.pinned = false;
			self.tape_laid = None;
		}

		let shown = if recognising {
			let follow = &mut self.follow;
			self.objects.shown(l, || follow.next_id(l))
		} else {
			Vec::new()
		};
		let blend = match (taped, self.tape.as_ref(), self.tape_at) {
			// the tape knows when things are coming and going, so the boxes lean toward them in time
			(true, Some(tape), Some(i)) => {
				self.blend = tape.blend[i];
				let b = self.blend;
				b * b * (3.0 - 2.0 * b)
			}
			_ => self.ease(!shown.is_empty(), dt),
		};
		if !self.edit.zones.is_empty() {
			self.finder.allowed = self.edit.allowed(self.view, self.finder.w, self.finder.h);
		} else if !self.finder.allowed.is_empty() {
			self.finder.allowed.clear();
		}
		if fresh || changed || news || (blend > 0.0 && blend < 1.0) {
			let points = self.finder.find(l, &shown, blend);
			let objects = if recognising { Some(&mut self.objects) } else { None };
			let camera_moved = self.finder.subject.fresh;
			self.follow.step(points, l, &mut self.finder.subject, unit, objects);
			// echoes ride along with the camera, as the tracks just did
			if camera_moved {
				self.echo.carry(&self.finder.subject, unit);
			}
			self.finding = finding;
		}
		if let Some(found) = found {
			self.objects.ingest(&found, l, fw / unit, fh / unit, 0.0);
			let follow = &mut self.follow;
			let named = self.objects.shown(l, || follow.next_id(l));
			self.follow.named = named;
		}
		if taped {
			self.tape_numbers();
		}
		if Chooser::active(&self.edit) {
			self.chooser.update(
				&self.edit,
				self.t,
				self.clock,
				self.view,
				(fw / unit, fh / unit),
				&self.follow.named,
				&self.follow.regions,
			);
		} else if l.follow {
			let main = Edit { keep: crate::edit::Keep::Main, ..Edit::default() };
			self.aim.update(
				&main,
				self.t,
				self.clock,
				self.view,
				(fw / unit, fh / unit),
				&self.follow.named,
				&self.follow.regions,
			);
		}
		Ok(())
	}

	fn render(&mut self, src: &Frame, l: &Look, out: &mut Frame) {
		let mut lap = crate::perf::Lap::start();
		let b = self.blend;
		let blend = b * b * (3.0 - 2.0 * b);
		if self.marks_only {
			// the picture's made already: only the marks go on it
		} else if Grade::changes(l) {
			self.grade.render(src, l, self.t as f32, true, out);
		} else {
			out.w = src.w;
			out.h = src.h;
			out.rgba.clear();
			out.rgba.extend_from_slice(&src.rgba);
		}
		lap.mark("grade");
		// the models' answers, taken in whatever's drawn (a bare stretch draws nothing, but waits for
		// nothing either)
		self.take_done();
		let bare = self.edit.bare(self.t);
		if bare >= 1.0 {
			self.beat_hit = None;
			return;
		}
		let clean = (bare > 0.0).then(|| out.rgba.clone());
		// on the beat: whether one has come since the last frame, and how long ago the recent ones were
		let beats = self.used_beats(l);
		let hit = beats
			.as_ref()
			.map(|b| self.fresh_frame && b.iter().any(|&x| x <= self.t && self.last_t.is_some_and(|p| x > p)));
		self.beat_hit = hit;
		let pulses: Option<Vec<f32>> = beats
			.as_ref()
			.map(|b| b.iter().filter(|&&x| x <= self.t && self.t - x < 3.0).map(|&x| (self.t - x) as f32).collect());
		if self.fresh_frame {
			self.last_t = Some(self.t);
		}
		// which frame this is, for what the models said about it before (playing it again)
		let at = self.recall.as_ref().and_then(|_| crate::recall::Recall::at(self.t, self.fps, self.view));
		// what's wanted of the window's models this frame (depth, the scene)
		let mut want = (false, false);
		// depth: worked out on each new frame (used once the layers are done)
		if crate::depth::active(l)
			&& !self.marks_only
			&& self.fresh_frame
			&& let Some(d) = self.depther.as_mut()
		{
			match at.and_then(|a| self.recall.as_mut().and_then(|r| r.depth(a))) {
				Some((near, range)) => d.recall(near, range),
				None if !d.has_model() => want.0 = true,
				None => match d.run(src) {
					Ok(()) => {
						if let (Some(a), Some(r), Some(range)) = (at, self.recall.as_mut(), d.range()) {
							r.keep_depth(a, &d.near, range);
						}
					}
					Err(e) => self.trouble = Some(format!("depth: {e}")),
				},
			}
		}
		lap.mark("depth model");
		if crate::scene::active(l)
			&& !self.marks_only
			&& self.fresh_frame
			&& let Some(s) = self.scener.as_mut()
		{
			match at.and_then(|a| self.recall.as_mut().and_then(|r| r.scene(a))) {
				Some((grid, kind)) => s.recall(grid, kind),
				None if !s.has_model() => want.1 = true,
				None => {
					// after frames it remembered, what it eased toward is from a while ago
					if let Some(a) = at
						&& self.scene_ran.is_some_and(|r| r + 1 != a.0)
					{
						s.reset();
					}
					match s.run(src) {
						Ok(()) => {
							if let (Some(a), Some(r)) = (at, self.recall.as_mut()) {
								r.keep_scene(a, s.grid(), &s.kind);
							}
							self.scene_ran = at.map(|a| a.0);
						}
						Err(e) => self.trouble = Some(format!("scene: {e}")),
					}
				}
			}
		}
		lap.mark("scene model");
		let mut tracks = std::mem::take(&mut self.follow.tracks);
		let mut regions = std::mem::take(&mut self.follow.regions);
		let mut named = std::mem::take(&mut self.follow.named);
		let low = std::mem::take(&mut self.finder.low);
		let narrowed = Chooser::active(&self.edit).then(|| (tracks.clone(), regions.clone(), named.clone()));
		if narrowed.is_some() {
			self.chooser.narrow(&self.edit, &mut tracks, &mut regions, &mut named);
		}
		// for revealing the things or the world as they were, after everything's drawn
		let mut revealing: Option<Matte> = None;
		// the window: the things to outline on the models' thread
		let mut outline_job: Option<(Vec<crate::Rect>, Vec<i64>)> = None;
		// layers: the things and the world made into something else, through a matte
		if !self.marks_only && (crate::matte::active(l) || crate::echo::active(l) || crate::glyphs::needs_matte(l)) {
			let m = out.w.min(out.h) as f32;
			let px = |r: crate::Rect| crate::Rect { x: r.x * m, y: r.y * m, w: r.w * m, h: r.h * m };
			// only what the detector saw at its last look (a box coasting on after a cut would have the
			// ground outlined); looking ahead, only its things (boxes round movement outline spray,
			// cloud and shadow)
			let taped = l.detect && self.tape_at.is_some();
			let (boxes, ids, words): (Vec<crate::Rect>, Vec<i64>, Vec<String>) =
				if taped || (blend > 0.5 && !named.is_empty()) {
					named
						.iter()
						.filter(|o| o.miss == 0)
						.map(|o| (px(o.rect()), o.id, o.cls.clone()))
						.collect::<Vec<_>>()
						.into_iter()
						.fold((Vec::new(), Vec::new(), Vec::new()), |mut v, (b, i, w)| {
							v.0.push(b);
							v.1.push(i);
							v.2.push(w);
							v
						})
				} else {
					regions
						.iter()
						.filter(|r| r.miss == 0)
						.map(|r| (px(r.rect()), r.id, "moving".to_string()))
						.collect::<Vec<_>>()
						.into_iter()
						.fold((Vec::new(), Vec::new(), Vec::new()), |mut v, (b, i, w)| {
							v.0.push(b);
							v.1.push(i);
							v.2.push(w);
							v
						})
				};
			let sub = &self.finder.subject;
			let moving = (sub.cols > 0 && self.finder.w > 0).then(|| Moving {
				mask: &sub.mask,
				cols: sub.cols,
				rows: sub.rows,
				cell: crate::subject::CELL as f32 * out.w as f32 / self.finder.w as f32,
				only_in_boxes: Chooser::active(&self.edit),
			});
			lap.mark("before matte");
			let known = (l.matte == "outline" && self.has_outliner())
				.then(|| {
					at.filter(|_| !boxes.is_empty())
						.and_then(|a| self.recall.as_mut().and_then(|r| r.shapes(a, &boxes)))
				})
				.flatten();
			let shapes = if l.matte != "outline" {
				None
			} else if let Some(known) = known {
				// this frame with these boxes before: the same outlines
				Some(self.steady.recalled(known, &boxes, &ids, src.w, src.h, self.frame, self.fresh_frame))
			} else if let Some(o) = self.outliner.as_mut() {
				match o.outline(src, &boxes) {
					Ok(offers) => {
						let shapes =
							self.steady.shapes(&offers, &boxes, &ids, src.w, src.h, self.frame, self.fresh_frame);
						if let (Some(a), Some(r)) = (at, self.recall.as_mut())
							&& self.fresh_frame
						{
							r.keep_shapes(a, &boxes, &self.steady.last);
						}
						Some(shapes)
					}
					Err(e) => {
						self.trouble = Some(format!("outlines: {e}"));
						None
					}
				}
			} else if self.background.as_ref().is_some_and(|b| b.has_outliner) {
				// the window: the outlines the models' thread gave last, on where the things are now
				if !boxes.is_empty() {
					outline_job = Some((boxes.clone(), ids.clone()));
				}
				Some(self.steady.current(&boxes, &ids, src.w, src.h))
			} else {
				None
			};
			lap.mark("outlines");
			let matte = Matte::build(src, &boxes, moving, shapes.as_deref(), l);
			lap.mark("matte");
			crate::matte::composite(out, src, &low, &matte, l, self.t);
			lap.mark("composite");
			crate::matte::draw_rings(out, &matte, l, self.clock, pulses.as_deref());
			lap.mark("rings");
			if crate::echo::active(l) {
				self.echo.apply(out, &matte, &boxes, l, self.clock, self.fresh_frame, m, hit);
			}
			lap.mark("echo");
			if l.glyphs == "words" {
				let things: Vec<crate::glyphs::Word> = boxes
					.iter()
					.enumerate()
					.map(|(i, b)| crate::glyphs::Word {
						rect: *b,
						word: words[i].clone(),
						shape: shapes.as_ref().and_then(|s| s.get(i)).filter(|s| !s.logits.is_empty()),
					})
					.collect();
				let named_world = (!l.world_word.trim().is_empty()).then(|| l.world_word.trim().to_string());
				let scene = self.scener.as_ref().filter(|_| crate::scene::active(l));
				let (fw, fh) = (out.w as f32, out.h as f32);
				self.glyphs.draw_words(
					out,
					&matte,
					&things,
					&|x, y| {
						scene.and_then(|s| s.at(x / fw, y / fh)).map(str::to_string).or_else(|| named_world.clone())
					},
					l,
				);
			} else if crate::glyphs::needs_matte(l) {
				self.glyphs.draw(out, Some(&matte), l);
			}
			lap.mark("glyphs");
			crate::matte::draw_contour(out, &matte, l);
			lap.mark("contour");
			if l.reveal == "things"
				|| l.reveal == "world"
				|| l.mix_how == "things"
				|| l.mix_how == "world"
				|| (l.mosh != "off" && l.mosh_where != "all")
			{
				revealing = Some(matte);
			}
		}
		// the window's models: this frame to their thread, if it's free, anything's wanted of it and it
		// hasn't had this frame already
		if let Some(bg) = self.background.as_mut()
			&& !bg.busy()
			&& self.bg_frame != self.frame
			&& (want.0 || want.1 || outline_job.is_some())
		{
			let (boxes, ids) = outline_job.take().unwrap_or_default();
			bg.submit(crate::background::Job {
				at,
				frame: src.clone(),
				outline: !boxes.is_empty(),
				boxes,
				ids,
				depth: want.0,
				scene: want.1,
			});
			self.bg_frame = self.frame;
		}
		// focus: the lens pulls to the biggest thing on show (the picked one, when there are picks)
		if l.depth == "focus"
			&& !self.marks_only
			&& self.fresh_frame
			&& let Some(d) = self.depther.as_mut()
		{
			let m = out.w.min(out.h) as f32;
			let main = named.iter().max_by(|a, b| (a.w * a.h).total_cmp(&(b.w * b.h))).map(|o| crate::Rect {
				x: o.x * m,
				y: o.y * m,
				w: o.w * m,
				h: o.h * m,
			});
			d.aim(main, out.w as usize, out.h as usize);
		}
		if crate::depth::active(l)
			&& !self.marks_only
			&& let Some(d) = self.depther.as_ref()
		{
			d.apply(out, l, self.t);
		}
		lap.mark("depth drawing");
		if crate::glyphs::active(l) && !crate::glyphs::needs_matte(l) && !self.marks_only {
			self.glyphs.draw(out, None, l);
		}
		if l.show_subject
			&& !self.marks_only
			&& let Some(seen) = self.finder.subject.paint()
		{
			over(out, &seen);
		}
		// datamosh: the picture pushed along by its own motion (under what's drawn over it)
		if l.mosh != "off" && !self.marks_only {
			let cut =
				self.tape.as_ref().zip(self.tape_at).is_some_and(|(t, i)| t.cuts.get(i).copied().unwrap_or(false));
			self.glitch.mosh(out, src, l, revealing.as_ref(), cut, 1.0 / self.fps.max(1.0), self.fresh_frame);
		}
		lap.mark("mosh");
		// insets crop the picture before anything's drawn over it
		// (drawing only marks, the insets show the picture as it came: untouched by the other look)
		let unmarked = crate::insets::active(l)
			.then(|| if self.marks_only && src.w == out.w && src.h == out.h { src.clone() } else { out.clone() });
		{
			let hud = |k: &str| self.token(k, &tracks, &regions, &named);
			let scene = Scene { tracks: &tracks, regions: &regions, objects: &named, blend, low: &low, hud: &hud };
			draw(out, &scene, l);
		}
		lap.mark("overlay");
		self.draw_foresight(out, l, &named);
		if let Some(unmarked) = unmarked {
			let m = out.w.min(out.h) as f32;
			let px = |r: crate::Rect| crate::Rect { x: r.x * m, y: r.y * m, w: r.w * m, h: r.h * m };
			let things: Vec<crate::insets::Thing> = if (l.detect && self.tape_at.is_some())
				|| (blend > 0.5 && !named.is_empty())
			{
				named
					.iter()
					.map(|o| crate::insets::Thing { id: o.id, cls: o.cls.clone(), conf: o.conf, rect: px(o.rect()) })
					.collect()
			} else {
				regions
					.iter()
					.filter(|r| r.miss == 0)
					.map(|r| crate::insets::Thing {
						id: r.id,
						cls: "moving".into(),
						conf: f32::NAN,
						rect: px(r.rect()),
					})
					.collect()
			};
			self.insets.draw(out, &unmarked, &things, l);
		}
		if let Some((t, r, n)) = narrowed {
			(tracks, regions, named) = (t, r, n);
		}
		self.follow.tracks = tracks;
		self.follow.regions = regions;
		self.follow.named = named;
		self.finder.low = low;
		lap.mark("insets");
		if src.w == out.w && src.h == out.h && !self.marks_only {
			put_together(
				out,
				src,
				&l.reveal,
				l.reveal_at,
				l.reveal_time,
				l.reveal_jitter,
				self.clock,
				self.t,
				revealing.as_ref(),
				beats.as_deref(),
			);
		}
		lap.mark("reveal");
		// mixing: the matte, for putting the other look in after it's drawn (both fade to the picture
		// as it was in bare stretches, so the mix does too)
		self.mix_matte =
			if self.with.is_some() && (l.mix_how == "things" || l.mix_how == "world") { revealing } else { None };
		// fading into or out of a bare stretch
		if let Some(clean) = clean {
			let k = (bare * 256.0) as u32;
			for (o, c) in out.rgba.iter_mut().zip(&clean) {
				*o = ((*o as u32 * (256 - k) + *c as u32 * k) >> 8) as u8;
			}
		}
	}

	/// Foresight: where each thing on show is going over the next `foresight` seconds, from the tape (so
	/// only where the clip's been looked ahead over): its path as a line fading toward the future, a dot
	/// each half second, and a faint box where it ends up.
	fn draw_foresight(&self, out: &mut Frame, l: &Look, named: &[ObjectTrack]) {
		if l.foresight <= 0.0 {
			return;
		}
		let (Some(tape), Some(i)) = (self.tape.as_ref(), self.tape_at) else { return };
		let [vx, vy, vw, vh] = self.view;
		let (fw, fh) = (out.w as f32, out.h as f32);
		let to_px = |r: crate::Rect| ((r.x - vx) / vw * fw, (r.y - vy) / vh * fh, r.w / vw * fw, r.h / vh * fh);
		let ahead = (l.foresight as f64 * tape.fps).round() as usize;
		let u = fw.min(fh) / 1080.0;
		let Some(mut pm) = tiny_skia::PixmapMut::from_bytes(&mut out.rgba, out.w, out.h) else { return };
		let ink = crate::overlay::colour(&l.foresight_color, 1.0);
		let stroke = |w: f32| tiny_skia::Stroke {
			width: (w * u).max(0.75),
			line_cap: tiny_skia::LineCap::Round,
			line_join: tiny_skia::LineJoin::Round,
			..Default::default()
		};
		let paint_at = |a: f32| {
			let mut p = tiny_skia::Paint::default();
			let mut c = ink;
			c.set_alpha(a.clamp(0.0, 1.0));
			p.set_color(c);
			p.anti_alias = true;
			p
		};
		let half = ((tape.fps * 0.5).round() as usize).max(1);
		for o in named {
			let Some(k) = self.tape_ids.iter().position(|&id| id == o.id && id != crate::objects::UNNUMBERED) else {
				continue;
			};
			let thing = &tape.things[k];
			if i < thing.first || i > thing.last() {
				continue;
			}
			let end = (i + ahead).min(thing.last());
			if end <= i {
				continue;
			}
			// the path of its middle, a piece at a time so it fades
			let points: Vec<(f32, f32)> = (i..=end)
				.map(|j| {
					let (x, y, w, h) = to_px(thing.boxes[j - thing.first]);
					(x + w / 2.0, y + h / 2.0)
				})
				.collect();
			let n = points.len();
			for (s, pair) in points.windows(2).enumerate() {
				let mut pb = tiny_skia::PathBuilder::new();
				pb.move_to(pair[0].0, pair[0].1);
				pb.line_to(pair[1].0, pair[1].1);
				if let Some(path) = pb.finish() {
					let a = 0.95 * (1.0 - s as f32 / n as f32);
					pm.stroke_path(&path, &paint_at(a), &stroke(2.0), tiny_skia::Transform::identity(), None);
				}
			}
			// a dot each half second
			for (s, p) in points.iter().enumerate().skip(half).step_by(half) {
				let r = 3.0 * u.max(0.6);
				if let Some(c) = tiny_skia::PathBuilder::from_circle(p.0, p.1, r) {
					pm.fill_path(
						&c,
						&paint_at(0.9 * (1.0 - s as f32 / n as f32) + 0.1),
						tiny_skia::FillRule::Winding,
						tiny_skia::Transform::identity(),
						None,
					);
				}
			}
			// where it ends up
			let (x, y, w, h) = to_px(thing.boxes[end - thing.first]);
			if let Some(r) = tiny_skia::Rect::from_xywh(x, y, w.max(1.0), h.max(1.0)) {
				let mut pb = tiny_skia::PathBuilder::new();
				pb.push_rect(r);
				if let Some(path) = pb.finish() {
					let dash = tiny_skia::StrokeDash::new(vec![6.0 * u.max(0.5), 5.0 * u.max(0.5)], 0.0);
					let mut st = stroke(1.2);
					st.dash = dash;
					pm.stroke_path(&path, &paint_at(0.55), &st, tiny_skia::Transform::identity(), None);
				}
			}
		}
	}

	/// Move the blend toward the detector while it has something to show, back when it doesn't, and
	/// return it eased.
	fn ease(&mut self, toward: bool, dt: f64) -> f32 {
		let step = (dt * 1000.0 / if toward { 1800.0 } else { 1200.0 }) as f32;
		self.blend = (self.blend + if toward { step } else { -step }).clamp(0.0, 1.0);
		let b = self.blend;
		b * b * (3.0 - 2.0 * b)
	}

	fn token(
		&self,
		k: &str,
		tracks: &[crate::tracking::Track],
		regions: &[crate::tracking::Region],
		named: &[ObjectTrack],
	) -> Option<String> {
		let t = self.t;
		let two = |n: f64| format!("{:02}", n.floor() as i64);
		Some(match k {
			"n" => tracks.len().to_string(),
			"regions" => (if self.blend > 0.5 { named.len() } else { regions.len() }).to_string(),
			"counts" => {
				if self.blend <= 0.5 || named.is_empty() {
					format!("moving {}", regions.len())
				} else {
					let mut by: Vec<(&str, usize)> = Vec::new();
					for o in named {
						match by.iter_mut().find(|(c, _)| *c == o.cls) {
							Some(e) => e.1 += 1,
							None => by.push((&o.cls, 1)),
						}
					}
					by.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
					by.iter().map(|(c, n)| format!("{c} {n}")).collect::<Vec<_>>().join("\n")
				}
			}
			"list" => {
				if self.blend <= 0.5 || named.is_empty() {
					regions.iter().map(|r| format!("{} moving", r.id)).collect::<Vec<_>>().join("\n")
				} else {
					let mut v: Vec<&ObjectTrack> = named.iter().collect();
					v.sort_by_key(|o| o.id);
					v.iter()
						.map(|o| format!("{:>3} {} {}%", o.id, o.cls, (o.conf * 100.0).round() as i64))
						.collect::<Vec<_>>()
						.join("\n")
				}
			}
			"frame" => self.frame.to_string(),
			"fps" => format!("{}", self.fps.round() as i64),
			"file" => self.name.clone(),
			"rec" => (if (self.clock * 1000.0 / 600.0).floor() as i64 % 2 == 1 { "●" } else { " " }).to_string(),
			"clock" => format!("{}:{}:{}", two(t / 60.0), two(t % 60.0), two((t % 1.0) * 100.0)),
			"date" => date(self.started + self.clock),
			_ => return None,
		})
	}
}

/// Lay a small see-through picture over the whole frame, each of its pixels a hard-edged block.
fn over(out: &mut Frame, top: &Frame) {
	if top.w == 0 || top.h == 0 {
		return;
	}
	let (w, h) = (out.w as usize, out.h as usize);
	for y in 0..h {
		let ty = (y * top.h as usize / h).min(top.h as usize - 1);
		for x in 0..w {
			let tx = (x * top.w as usize / w).min(top.w as usize - 1);
			let s = &top.rgba[(ty * top.w as usize + tx) * 4..][..4];
			let a = s[3] as u32;
			if a == 0 {
				continue;
			}
			let d = &mut out.rgba[(y * w + x) * 4..][..3];
			for c in 0..3 {
				d[c] = ((s[c] as u32 * a + d[c] as u32 * (255 - a) + 127) / 255) as u8;
			}
		}
	}
}

/// A time as "yyyy-mm-dd hh:mm:ss", in UTC, as the site's toISOString gives it.
pub fn date(secs: f64) -> String {
	let s = secs.floor() as i64;
	let (days, rem) = (s.div_euclid(86_400), s.rem_euclid(86_400));
	// days since 1970 to a calendar date (Howard Hinnant's method)
	let z = days + 719_468;
	let era = z.div_euclid(146_097);
	let doe = z.rem_euclid(146_097);
	let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let d = doy - (153 * mp + 2) / 5 + 1;
	let m = if mp < 10 { mp + 3 } else { mp - 9 };
	let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
	format!("{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}", rem / 3600, rem % 3600 / 60, rem % 60)
}

/// Another picture put together with the look in `out` (the same size): the picture as it was, for
/// reveal, or another look, for mixing, in one of the ways `Look::reveal` and `Look::mix` list.
/// `clock` is the output's seconds so far, `t` the source's.
pub(crate) fn put_together(
	out: &mut Frame,
	other: &Frame,
	how: &str,
	at: f32,
	time: f32,
	jitter: f32,
	clock: f64,
	t: f64,
	matte: Option<&Matte>,
	beats: Option<&[f64]>,
) {
	use rayon::prelude::*;
	if other.w != out.w || other.h != out.h {
		return;
	}
	let src = other;
	let (w, at) = (out.w as usize, at.clamp(0.0, 1.0));
	let time = (time as f64).max(0.02);
	let columns = |out: &mut Frame, upto: f32| {
		let e = ((upto * w as f32).round() as usize).min(w);
		if e > 0 {
			out.rgba
				.par_chunks_mut(w * 4)
				.zip(src.rgba.par_chunks(w * 4))
				.for_each(|(o, s)| o[..e * 4].copy_from_slice(&s[..e * 4]));
		}
	};
	match how {
		"split" => columns(out, at),
		"sweep" => {
			let phase = (clock / time).fract() as f32;
			// across and back, easing at each end
			let p = if phase < 0.5 { phase * 2.0 } else { 2.0 - phase * 2.0 };
			columns(out, p * p * (3.0 - 2.0 * p) * at);
		}
		"flick" => {
			// on the music: the other for the first `at` of the time to the next beat, jitter skipping
			// some beats
			let shown = if let Some(b) = beats {
				match crate::beats::around(b, t) {
					Some((k, from, next)) => {
						crate::tracking::hash(k as i64, 9173) >= jitter.clamp(0.0, 1.0) * 0.6
							&& t < from + (next - from) * at as f64
					}
					None => false,
				}
			} else {
				// each beat of `time`, the other for the first `at` of it; with jitter, some beats skip
				// it and some hold it, the same every time it's rendered
				let beat = clock / time;
				let k = beat.floor() as i64;
				let regular = (beat.fract() as f32) < at;
				let chance = crate::tracking::hash(k, 9173);
				let wild = crate::tracking::hash(k, 4211) < at;
				if chance < jitter.clamp(0.0, 1.0) { wild } else { regular }
			};
			if shown {
				out.rgba.copy_from_slice(&src.rgba);
			}
		}
		"things" | "world" => {
			let Some(m) = matte else { return };
			let things = how == "things";
			out.rgba.par_chunks_mut(w * 4).zip(src.rgba.par_chunks(w * 4)).enumerate().for_each(|(y, (o, s))| {
				for x in 0..w {
					let a = m.at(x, y);
					let k = if things { a } else { 1.0 - a } * at;
					if k <= 0.0 {
						continue;
					}
					for c in 0..3 {
						let i = x * 4 + c;
						o[i] = (o[i] as f32 + (s[i] as f32 - o[i] as f32) * k).round() as u8;
					}
				}
			});
		}
		"blend" => {
			let k = (at * 256.0) as u32;
			out.rgba.par_chunks_mut(w * 4).zip(src.rgba.par_chunks(w * 4)).for_each(|(o, s)| {
				for (o, s) in o.iter_mut().zip(s) {
					*o = ((*o as u32 * (256 - k) + *s as u32 * k) >> 8) as u8;
				}
			});
		}
		"lighter" => {
			out.rgba.par_chunks_mut(w * 4).zip(src.rgba.par_chunks(w * 4)).for_each(|(o, s)| {
				for (o, s) in o.iter_mut().zip(s) {
					*o = (*o).max(*s);
				}
			});
		}
		_ => {}
	}
}

/// `out` faded `k` of the way to `to` (the same size).
fn fade(out: &mut Frame, to: &Frame, k: f32) {
	use rayon::prelude::*;
	if to.w != out.w || to.h != out.h {
		return;
	}
	let k = (k.clamp(0.0, 1.0) * 256.0) as u32;
	out.rgba.par_chunks_mut(4096).zip(to.rgba.par_chunks(4096)).for_each(|(o, b)| {
		for (o, b) in o.iter_mut().zip(b) {
			*o = ((*o as u32 * (256 - k) + *b as u32 * k) >> 8) as u8;
		}
	});
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn two_pictures_put_together() {
		let flat = |v: u8| Frame::from_rgba(4, 2, [v, v, v, 255].repeat(8));
		let row = |f: &Frame| f.rgba.chunks(4).take(4).map(|p| p[0]).collect::<Vec<u8>>();
		let (a, b) = (flat(0), flat(200));
		// split: the other on the left, up to halfway
		let mut out = a.clone();
		put_together(&mut out, &b, "split", 0.5, 1.0, 0.0, 0.0, 0.0, None, None);
		assert_eq!(row(&out), vec![200, 200, 0, 0]);
		// blend: a quarter of the other
		let mut out = a.clone();
		put_together(&mut out, &b, "blend", 0.25, 1.0, 0.0, 0.0, 0.0, None, None);
		assert_eq!(row(&out), vec![50; 4]);
		// lighter, either way round
		let mut out = b.clone();
		put_together(&mut out, &a, "lighter", 1.0, 1.0, 0.0, 0.0, 0.0, None, None);
		assert_eq!(row(&out), vec![200; 4]);
		// flick: the other for the first half of each half second
		let mut out = a.clone();
		put_together(&mut out, &b, "flick", 0.5, 0.5, 0.0, 0.1, 0.1, None, None);
		assert_eq!(row(&out), vec![200; 4]);
		let mut out = a.clone();
		put_together(&mut out, &b, "flick", 0.5, 0.5, 0.0, 0.4, 0.4, None, None);
		assert_eq!(row(&out), vec![0; 4]);
		// pictures of different sizes are left alone
		let mut out = a.clone();
		put_together(&mut out, &Frame::from_rgba(2, 2, vec![255; 16]), "blend", 1.0, 1.0, 0.0, 0.0, 0.0, None, None);
		assert_eq!(row(&out), vec![0; 4]);
	}

	#[test]
	fn dates_read_like_iso() {
		assert_eq!(date(0.0), "1970-01-01 00:00:00");
		assert_eq!(date(1_794_279_845.0), "2026-11-10 03:04:05");
		assert_eq!(date(951_782_400.0), "2000-02-29 00:00:00");
	}

	fn square(i: usize) -> Frame {
		let mut f = Frame::new(320, 180);
		for y in 60..100 {
			for x in (20 + i * 8)..(60 + i * 8) {
				let p = (y * 320 + x) * 4;
				f.rgba[p..p + 3].copy_from_slice(&[240, 240, 240]);
			}
		}
		f
	}

	#[test]
	fn a_moving_square_gets_points_and_a_drawing() {
		let mut e = Engine::new(Look::default());
		let mut out = Frame::default();
		for i in 0..20 {
			e.step(&square(i), i as f64 / 30.0, 1.0 / 30.0, &mut out).unwrap();
		}
		assert!(!e.follow.tracks.is_empty());
		assert_eq!((out.w, out.h), (320, 180));
	}

	#[test]
	fn a_paused_frame_redraws_with_a_new_look() {
		let mut e = Engine::new(Look::default());
		let mut out = Frame::default();
		for i in 0..10 {
			e.step(&square(i), i as f64 / 30.0, 1.0 / 30.0, &mut out).unwrap();
		}
		let before = out.rgba.clone();
		e.look = Look::preset("ember").unwrap();
		e.redraw(&square(9), 9.0 / 30.0, 0.0, &mut out).unwrap();
		assert_ne!(before, out.rgba, "the new look shows");
	}
}
