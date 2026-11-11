//! Beats: where the clip's sound hits, so a look can cut, copy or pulse on the music. ffmpeg decodes
//! the sound to one channel at 11 kHz; every 10 ms the loudness of the low end (kicks) and of
//! everything is compared with the 10 ms before, and where it jumps far more than it has been jumping
//! around then (a second either side), with at least 150 ms since the last, is a beat. The same sound
//! gives the same beats every time.

use anyhow::{Context, Result};
use std::io::Read;
use std::process::Stdio;

const RATE: u32 = 11_025;
/// 10 ms
const HOP: usize = 110;

/// The beats in `input`'s sound, in seconds of the source; none if it has no sound.
pub fn find(input: &str) -> Result<Vec<f64>> {
	let mut c = crate::video::ffmpeg();
	c.args(["-v", "error", "-nostdin"]);
	if input.starts_with("http://") || input.starts_with("https://") {
		c.args(["-multiple_requests", "1", "-rw_timeout", "15000000"]);
	}
	let mut child = c
		.args(["-i", input, "-vn", "-sn", "-ac", "1", "-ar", &RATE.to_string(), "-f", "f32le", "-"])
		.stdin(Stdio::null())
		.stdout(Stdio::piped())
		.stderr(Stdio::null())
		.spawn()
		.context("couldn't run ffmpeg for the sound")?;
	let mut bytes = Vec::new();
	child.stdout.take().unwrap().read_to_end(&mut bytes)?;
	let _ = child.wait();
	let samples: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
	Ok(onsets(&samples, RATE))
}

/// Where the sound in `samples` (one channel at `rate`) hits, in seconds.
pub fn onsets(samples: &[f32], rate: u32) -> Vec<f64> {
	let hop = (HOP as u64 * rate as u64 / RATE as u64).max(1) as usize;
	let n = samples.len() / hop;
	if n < 3 {
		return Vec::new();
	}
	// the low end: a one-pole low-pass at about 150 Hz
	let a = (-2.0 * std::f32::consts::PI * 150.0 / rate as f32).exp();
	let mut low = 0.0f32;
	let mut energy = vec![(0.0f32, 0.0f32); n];
	for (i, e) in energy.iter_mut().enumerate() {
		let (mut lo, mut all) = (0.0f32, 0.0f32);
		for &x in &samples[i * hop..(i + 1) * hop] {
			low = a * low + (1.0 - a) * x;
			lo += low * low;
			all += x * x;
		}
		*e = ((lo / hop as f32 + 1e-9).ln(), (all / hop as f32 + 1e-9).ln());
	}
	// how much louder each moment is than the one before, the low end counting double
	let mut flux = vec![0.0f32; n];
	for i in 1..n {
		flux[i] = 2.0 * (energy[i].0 - energy[i - 1].0).max(0.0) + (energy[i].1 - energy[i - 1].1).max(0.0);
	}
	let smooth: Vec<f32> =
		(0..n).map(|i| (flux[i.saturating_sub(1)] + 2.0 * flux[i] + flux[(i + 1).min(n - 1)]) / 4.0).collect();
	// against how much it's been jumping a second either side
	let reach = (rate as usize / hop).max(1);
	let mut prefix = vec![0.0f64; n + 1];
	let mut prefix2 = vec![0.0f64; n + 1];
	for i in 0..n {
		prefix[i + 1] = prefix[i] + smooth[i] as f64;
		prefix2[i + 1] = prefix2[i] + (smooth[i] as f64).powi(2);
	}
	let gap = (0.15 * rate as f64 / hop as f64).round() as usize;
	let near = (0.05 * rate as f64 / hop as f64).round().max(1.0) as usize;
	let mut out = Vec::new();
	let mut last: Option<usize> = None;
	for i in 1..n.saturating_sub(1) {
		let (a, b) = (i.saturating_sub(reach), (i + reach + 1).min(n));
		let m = (b - a) as f64;
		let mean = (prefix[b] - prefix[a]) / m;
		let sd = ((prefix2[b] - prefix2[a]) / m - mean * mean).max(0.0).sqrt();
		let v = smooth[i] as f64;
		if v < mean + 1.5 * sd || v < 0.3 {
			continue;
		}
		// the top of its own peak
		let (pa, pb) = (i.saturating_sub(near), (i + near + 1).min(n));
		if smooth[pa..pb].iter().any(|&w| w as f64 > v) {
			continue;
		}
		if last.is_some_and(|l| i - l < gap) {
			continue;
		}
		last = Some(i);
		out.push(i as f64 * hop as f64 / rate as f64);
	}
	out
}

/// The beat at or before `t` among `beats` (in order), and the one after it.
pub fn around(beats: &[f64], t: f64) -> Option<(usize, f64, f64)> {
	let i = beats.partition_point(|&b| b <= t);
	if i == 0 {
		return None;
	}
	let next = beats.get(i).copied().unwrap_or(beats[i - 1] + 0.5);
	Some((i - 1, beats[i - 1], next))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_click_track_beats_where_it_clicks() {
		// two seconds of quiet hiss with a thump every half second from 0.25 s
		let rate = RATE;
		let mut s = vec![0.0f32; rate as usize * 3];
		let mut seed = 1u32;
		for x in s.iter_mut() {
			seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
			*x = ((seed >> 16) as f32 / 65_536.0 - 0.5) * 0.01;
		}
		let clicks = [0.25, 0.75, 1.25, 1.75, 2.25];
		for &c in &clicks {
			let at = (c * rate as f64) as usize;
			for k in 0..800 {
				s[at + k] += (k as f32 * 0.02).sin() * 0.8 * (1.0 - k as f32 / 800.0);
			}
		}
		let found = onsets(&s, rate);
		assert_eq!(found.len(), clicks.len(), "{found:?}");
		for (f, c) in found.iter().zip(clicks) {
			assert!((f - c).abs() < 0.03, "{found:?}");
		}
		assert_eq!(around(&found, 1.0).map(|b| b.0), Some(1));
	}
}
