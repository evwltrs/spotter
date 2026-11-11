//! The window's slow models (outlines, depth, what the scene is) on a thread of their own, so the
//! picture never waits for them. The engine hands over the newest frame whenever the thread's free
//! (never a queue: a frame that's been waiting is a frame that's gone) and draws with whatever came
//! back last: an outline is kept relative to its thing's box (outline.rs), so one a few frames old
//! still sits on the thing where it is now, and depth and the scene's words change slowly anyway.
//! The answers are shared, numbered, so two looks mixed in each take every one. Exports don't use
//! this: they wait for every frame's answers, as they should.

use crate::Frame;
use crate::Rect;
use crate::depth::Depther;
use crate::outline::{Offer, Outliner};
use crate::recall::At;
use crate::scene::Scener;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};

/// One frame to work out.
pub struct Job {
	pub at: Option<At>,
	pub frame: Frame,
	/// for outlines: the things' boxes in the frame's pixels, and their numbers
	pub boxes: Vec<Rect>,
	pub ids: Vec<i64>,
	pub outline: bool,
	pub depth: bool,
	pub scene: bool,
}

/// What the models said about a job's frame.
pub struct Done {
	pub seq: u64,
	pub at: Option<At>,
	pub w: u32,
	pub h: u32,
	pub boxes: Vec<Rect>,
	pub ids: Vec<i64>,
	pub offers: Option<Vec<Offer>>,
	/// nearness (0-1, the depther's size squared) and the range it was spread over
	pub depth: Option<(Vec<f32>, (f32, f32))>,
	/// the scene's grid and the kind in each cell
	pub scene: Option<(usize, Vec<u8>)>,
	pub error: Option<String>,
}

enum Msg {
	Job(u64, Job),
	Outliner(Outliner),
	Depther(Depther),
	Scener(Scener),
}

pub struct Background {
	tx: Sender<Msg>,
	latest: Arc<Mutex<Option<Arc<Done>>>>,
	busy: Arc<AtomicBool>,
	/// the last job handed over
	sent: u64,
	done: Arc<AtomicU64>,
	pub has_outliner: bool,
	pub has_depther: bool,
	pub has_scener: bool,
}

impl Default for Background {
	fn default() -> Self {
		Self::new()
	}
}

impl Background {
	pub fn new() -> Self {
		let (tx, rx) = channel::<Msg>();
		let latest: Arc<Mutex<Option<Arc<Done>>>> = Arc::default();
		let busy = Arc::new(AtomicBool::new(false));
		let done = Arc::new(AtomicU64::new(0));
		let (l, b, d) = (latest.clone(), busy.clone(), done.clone());
		std::thread::Builder::new()
			.name("models".into())
			.spawn(move || {
				let (mut outliner, mut depther, mut scener): (Option<Outliner>, Option<Depther>, Option<Scener>) =
					(None, None, None);
				while let Ok(m) = rx.recv() {
					let (seq, job) = match m {
						Msg::Outliner(o) => {
							outliner = Some(o);
							continue;
						}
						Msg::Depther(x) => {
							depther = Some(x);
							continue;
						}
						Msg::Scener(x) => {
							scener = Some(x);
							continue;
						}
						Msg::Job(seq, job) => (seq, job),
					};
					let mut error = None;
					let offers = match outliner.as_mut().filter(|_| job.outline && !job.boxes.is_empty()) {
						Some(o) => match o.outline(&job.frame, &job.boxes) {
							Ok(x) => Some(x),
							Err(e) => {
								error = Some(format!("outlines: {e}"));
								None
							}
						},
						None => None,
					};
					let depth = match depther.as_mut().filter(|_| job.depth) {
						Some(x) => match x.run(&job.frame) {
							Ok(()) => x.range().map(|r| (x.near.clone(), r)),
							Err(e) => {
								error = Some(format!("depth: {e}"));
								None
							}
						},
						None => None,
					};
					let scene = match scener.as_mut().filter(|_| job.scene) {
						Some(x) => match x.run(&job.frame) {
							Ok(()) => Some((x.grid(), x.kind.clone())),
							Err(e) => {
								error = Some(format!("scene: {e}"));
								None
							}
						},
						None => None,
					};
					let (w, h) = (job.frame.w, job.frame.h);
					*l.lock().unwrap() = Some(Arc::new(Done {
						seq,
						at: job.at,
						w,
						h,
						boxes: job.boxes,
						ids: job.ids,
						offers,
						depth,
						scene,
						error,
					}));
					d.store(seq, Ordering::Release);
					b.store(false, Ordering::Release);
				}
			})
			.expect("a thread for the models");
		Self { tx, latest, busy, sent: 0, done, has_outliner: false, has_depther: false, has_scener: false }
	}

	pub fn give_outliner(&mut self, o: Outliner) {
		self.has_outliner = true;
		let _ = self.tx.send(Msg::Outliner(o));
	}

	pub fn give_depther(&mut self, d: Depther) {
		self.has_depther = true;
		let _ = self.tx.send(Msg::Depther(d));
	}

	pub fn give_scener(&mut self, s: Scener) {
		self.has_scener = true;
		let _ = self.tx.send(Msg::Scener(s));
	}

	/// Whether a job's being worked on.
	pub fn busy(&self) -> bool {
		self.busy.load(Ordering::Acquire)
	}

	/// Hand over a frame (only when it's not busy). Says its number.
	pub fn submit(&mut self, job: Job) -> u64 {
		self.sent += 1;
		self.busy.store(true, Ordering::Release);
		if self.tx.send(Msg::Job(self.sent, job)).is_err() {
			self.busy.store(false, Ordering::Release);
		}
		self.sent
	}

	/// The number of the last job handed over.
	pub fn sent(&self) -> u64 {
		self.sent
	}

	/// The answers for the last job done, if they're newer than `seen`.
	pub fn newer(&self, seen: u64) -> Option<Arc<Done>> {
		if self.done.load(Ordering::Acquire) <= seen {
			return None;
		}
		self.latest.lock().unwrap().clone().filter(|d| d.seq > seen)
	}
}
