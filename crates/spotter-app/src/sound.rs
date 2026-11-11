//! Sound for the preview: ffmpeg decodes the clip's sound (and a song under it) from where the picture is, at the
//! speaker's own rate and number of channels, into a queue the speaker plays from. It's started
//! again whenever the picture jumps (a seek, a cut, a new video) and dropped when it pauses; over a
//! clip's length the two drift apart by no more than the speaker's buffer.

use anyhow::{Context, Result, anyhow, bail};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::collections::VecDeque;
use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A song under the clip's sound: from `at` seconds into it (before its start, a pause first), as
/// loud as `gain`, the clip's own sound kept at `keep`.
pub struct Song {
	pub path: String,
	pub at: f64,
	pub gain: f32,
	pub keep: f32,
}

pub struct Sound {
	_stream: cpal::Stream,
	child: Child,
	stop: Arc<AtomicBool>,
	/// samples the speaker has taken, for checking it plays
	played: Arc<std::sync::atomic::AtomicUsize>,
}

impl Drop for Sound {
	fn drop(&mut self) {
		if std::env::var_os("SPOTTER_TIMES").is_some() {
			eprintln!("sound: {} samples played", self.played.load(Ordering::Relaxed));
		}
		self.stop.store(true, Ordering::Relaxed);
		let _ = self.child.kill();
		let _ = self.child.wait();
	}
}

impl Sound {
	/// Play the clip's sound (`clip`, if any) from `t` seconds, at `speed` (0.5 is half speed), with a
	/// song under it (if any).
	/// `volume` (an f32's bits, 1 as it is) is read as it plays, so moving it needs no restart.
	pub fn start(
		clip: Option<&str>,
		t: f64,
		speed: f64,
		song: Option<&Song>,
		volume: Arc<std::sync::atomic::AtomicU32>,
	) -> Result<Sound> {
		if clip.is_none() && song.is_none() {
			bail!("there's no sound to play");
		}
		let host = cpal::default_host();
		let device = host.default_output_device().ok_or_else(|| anyhow!("there's nothing to play sound on"))?;
		let config = device.default_output_config().context("the speaker won't say how it plays")?;
		if config.sample_format() != cpal::SampleFormat::F32 {
			bail!("the speaker wants {:?} samples, which the preview doesn't make", config.sample_format());
		}
		let (channels, rate) = (config.channels() as usize, config.sample_rate());
		let mut c = spotter_core::video::ffmpeg();
		c.args(["-v", "error", "-nostdin"]);
		let input = |c: &mut Command, path: &str, at: f64| {
			if at > 0.0 {
				c.args(["-ss", &format!("{at:.3}")]);
			}
			if path.starts_with("http://") || path.starts_with("https://") {
				c.args(["-multiple_requests", "1", "-rw_timeout", "15000000"]);
			}
			c.args(["-i", path]);
		};
		if let Some(clip) = clip {
			input(&mut c, clip, t);
		}
		if let Some(s) = song {
			input(&mut c, &s.path, s.at);
		}
		// the clip's sound slowed or sped with the picture; the song as it is, after a pause if the
		// picture's before where it starts
		let mut steps = Vec::new();
		if (speed - 1.0).abs() > 1e-3 {
			let mut left = speed.clamp(0.05, 4.0);
			while left < 0.5 {
				steps.push("atempo=0.5".to_string());
				left /= 0.5;
			}
			while left > 2.0 {
				steps.push("atempo=2".to_string());
				left /= 2.0;
			}
			steps.push(format!("atempo={left}"));
		}
		let song_steps = |s: &Song| {
			let mut v = vec![format!("volume={:.3}", s.gain)];
			if s.at < 0.0 {
				v.push(format!("adelay=delays={}:all=1", (-s.at * 1000.0).round() as i64));
			}
			v.join(",")
		};
		c.args(["-vn", "-sn"]);
		match (clip, song) {
			(Some(_), Some(s)) => {
				steps.push(format!("volume={:.3}", s.keep));
				let graph = format!(
					"[0:a]{}[clip];[1:a]{}[song];[clip][song]amix=inputs=2:duration=longest:normalize=0[out]",
					steps.join(","),
					song_steps(s)
				);
				c.args(["-filter_complex", &graph, "-map", "[out]"]);
			}
			(None, Some(s)) => {
				c.args(["-af", &song_steps(s)]);
			}
			_ => {
				if !steps.is_empty() {
					c.args(["-af", &steps.join(",")]);
				}
			}
		}
		c.args(["-f", "f32le", "-ac", &channels.to_string(), "-ar", &rate.to_string(), "-"]);
		let mut child = c
			.stdin(Stdio::null())
			.stdout(Stdio::piped())
			.stderr(Stdio::null())
			.spawn()
			.context("couldn't run ffmpeg for the sound")?;
		let mut out = child.stdout.take().unwrap();
		let queue: Arc<Mutex<VecDeque<f32>>> = Arc::default();
		let stop = Arc::new(AtomicBool::new(false));
		// keep about a second ready; ffmpeg waits on the pipe when it's ahead
		let most = rate as usize * channels;
		{
			let (queue, stop) = (queue.clone(), stop.clone());
			std::thread::Builder::new()
				.name("sound".into())
				.spawn(move || {
					let mut buf = vec![0u8; 16 * 1024];
					let mut carry: Vec<u8> = Vec::new();
					while !stop.load(Ordering::Relaxed) {
						if queue.lock().unwrap().len() > most {
							std::thread::sleep(std::time::Duration::from_millis(10));
							continue;
						}
						let n = match out.read(&mut buf) {
							Ok(0) | Err(_) => return,
							Ok(n) => n,
						};
						carry.extend_from_slice(&buf[..n]);
						let whole = carry.len() / 4 * 4;
						let mut q = queue.lock().unwrap();
						q.extend(carry[..whole].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])));
						drop(q);
						carry.drain(..whole);
					}
				})
				.expect("a thread for the sound");
		}
		// SPOTTER_SOUND_GAIN turns it down (0 is silent, for testing)
		let gain: f32 = std::env::var("SPOTTER_SOUND_GAIN").ok().and_then(|g| g.parse().ok()).unwrap_or(1.0);
		let played = Arc::new(std::sync::atomic::AtomicUsize::new(0));
		let counted = played.clone();
		let stream = device
			.build_output_stream(
				&config.config(),
				move |data: &mut [f32], _| {
					let mut q = queue.lock().unwrap();
					let mut n = 0;
					let gain = gain * f32::from_bits(volume.load(Ordering::Relaxed));
					for s in data.iter_mut() {
						*s = match q.pop_front() {
							Some(v) => {
								n += 1;
								v * gain
							}
							None => 0.0,
						};
					}
					counted.fetch_add(n, Ordering::Relaxed);
				},
				|e| eprintln!("sound: {e}"),
				None,
			)
			.context("couldn't open the speaker")?;
		stream.play().context("couldn't start the speaker")?;
		Ok(Sound { _stream: stream, child, stop, played })
	}
}
