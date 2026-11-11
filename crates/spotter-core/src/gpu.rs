//! One thread for the GPU. DirectML fails now and then, deep inside itself, when models on
//! it are set up and run from different threads, as the window's detector, outliner and exports
//! would, even taking turns. An export on one thread never did. So everything that touches the
//! graphics card through ONNX Runtime is handed to this one thread and waited for.
//!
//! Work that can wait (the window looking ahead over a clip, `background`) waits: it goes when nothing
//! else is queued, or once it's waited long enough that it still gets somewhere, so the preview
//! doesn't stutter behind it.

use std::cell::Cell;
use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::mpsc::sync_channel;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

type Job = Box<dyn FnOnce() + Send + 'static>;

thread_local! {
	static ON_GPU: Cell<bool> = const { Cell::new(false) };
	static BACKGROUND: Cell<bool> = const { Cell::new(false) };
}

/// How long work that can wait waits at most, when the card is busy with work that can't.
const PATIENCE: Duration = Duration::from_millis(60);
/// How long the card must have been free of work that can't wait before work that can goes: the
/// preview's runs come a few milliseconds apart (an encoder, the CPU's share, a decoder), and a
/// 40 ms run slipped into each gap would hold every frame up.
const QUIET: Duration = Duration::from_millis(25);

/// Mark this thread's graphics card work as able to wait.
pub fn background() {
	BACKGROUND.with(|b| b.set(true));
}

#[derive(Default)]
struct Queues {
	now: VecDeque<Job>,
	later: VecDeque<(Instant, Job)>,
	/// when work that can't wait last finished
	busy_until: Option<Instant>,
}

/// How much of the time the GPU may be kept busy, in percent: after each job it rests
/// for long enough to leave the rest (the desktop and everything else need it too). 100 is no rest.
static SHARE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(100);

/// Keep the GPU busy at most `percent` of the time (the window sets this; exports on the
/// command line use all of it).
pub fn set_share(percent: u32) {
	SHARE.store(percent.clamp(10, 100), std::sync::atomic::Ordering::Relaxed);
}

fn queues() -> &'static (Mutex<Queues>, Condvar) {
	static Q: OnceLock<(Mutex<Queues>, Condvar)> = OnceLock::new();
	Q.get_or_init(|| {
		std::thread::Builder::new()
			.name("graphics card".into())
			.spawn(|| {
				ON_GPU.with(|g| g.set(true));
				let (lock, ready) = queues();
				loop {
					let (job, urgent) = {
						let mut q = lock.lock().unwrap_or_else(|e| e.into_inner());
						loop {
							let waited = q.later.front().is_some_and(|(since, _)| since.elapsed() >= PATIENCE);
							let quiet = q.busy_until.is_none_or(|t| t.elapsed() >= QUIET);
							if (waited || (q.now.is_empty() && quiet))
								&& let Some((_, j)) = q.later.pop_front()
							{
								break (j, false);
							}
							if let Some(j) = q.now.pop_front() {
								break (j, true);
							}
							// nothing to run yet: wait for work, or for the card to have been quiet long enough
							q = if q.later.is_empty() {
								ready.wait(q).unwrap_or_else(|e| e.into_inner())
							} else {
								ready.wait_timeout(q, QUIET).unwrap_or_else(|e| e.into_inner()).0
							};
						}
					};
					let started = Instant::now();
					job();
					let took = started.elapsed();
					if urgent {
						lock.lock().unwrap_or_else(|e| e.into_inner()).busy_until = Some(Instant::now());
					}
					// a rest, so the card's never kept busy all the time
					let share = SHARE.load(std::sync::atomic::Ordering::Relaxed);
					if share < 100 {
						std::thread::sleep(took * (100 - share) / share);
					}
				}
			})
			.expect("a thread for the graphics card");
		(Mutex::new(Queues::default()), Condvar::new())
	})
}

fn send(job: Job) {
	let (lock, ready) = queues();
	let mut q = lock.lock().unwrap_or_else(|e| e.into_inner());
	if BACKGROUND.with(Cell::get) {
		q.later.push_back((Instant::now(), job));
	} else {
		q.now.push_back(job);
	}
	ready.notify_one();
}

/// Run `f` on the GPU's thread and wait for it. On that thread already, it just runs.
pub fn on_gpu<R: Send>(f: impl FnOnce() -> R + Send) -> R {
	if ON_GPU.with(Cell::get) {
		return f();
	}
	let (done, wait) = sync_channel(1);
	let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
		let _ = done.send(catch_unwind(AssertUnwindSafe(f)));
	});
	// SAFETY: the job borrows from this call's frame, and this call doesn't return until the job
	// has run to the end (or the thread holding it has gone, in which case it never ran)
	let job: Job = unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + '_>, Job>(job) };
	send(job);
	match wait.recv() {
		Ok(Ok(r)) => r,
		Ok(Err(panic)) => resume_unwind(panic),
		Err(_) => panic!("the graphics card's thread stopped"),
	}
}

/// Run `f` on the GPU's thread, trying again (twice) if it fails. With several models on
/// the card at once (the window's preview and an export), DirectML now and then fails a run with
/// "an unhandled exception" deep inside it; the same run straight after works.
pub fn run_on_gpu<R: Send>(what: &str, mut f: impl FnMut() -> anyhow::Result<R> + Send) -> anyhow::Result<R> {
	on_gpu(|| {
		let mut tries = 0;
		loop {
			match f() {
				Ok(r) => return Ok(r),
				Err(e) if tries < 2 => {
					tries += 1;
					eprintln!(
						"{what} failed on the graphics card, trying again: {}",
						e.to_string().chars().take(200).collect::<String>()
					);
				}
				Err(e) => return Err(e),
			}
		}
	})
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn work_runs_on_one_thread_and_comes_back() {
		let mut v = vec![1, 2, 3];
		let here = std::thread::current().id();
		let (sum, there) = on_gpu(|| {
			v.push(4);
			(v.iter().sum::<i32>(), std::thread::current().id())
		});
		assert_eq!(sum, 10);
		assert_ne!(here, there);
		let again = std::thread::spawn(|| on_gpu(|| std::thread::current().id())).join().unwrap();
		assert_eq!(there, again);
		assert_eq!(on_gpu(|| on_gpu(|| 7)), 7);
	}
}
