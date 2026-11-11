//! Datamosh the real way, as bitrot (github.com/sixem/bitrot) does it: the clip encoded as MPEG-4
//! Part 2, a codec simple enough to cut up by hand, then its frames rearranged in the encoded stream
//! before it's played again. A video stores most frames as changes to the one before (P-frames:
//! which way each block moved, and what's different), with a whole picture (a key frame) now and
//! then. Take a key frame out and the changes after it are drawn on whatever picture came before:
//!
//! - on cuts: each cut's key frame is taken out (and the headers before it, which would start the
//!   decoder afresh), so the new shot's motion and changes are drawn over the old shot's picture,
//!   until a key frame `hold` seconds on puts it right;
//! - on beats: on each beat, one frame's motion is played over and over (P-frame duplication, for
//!   half the hold), so the picture blooms out along it, put right `hold` seconds on.
//!
//! Every frame stays (a frame taken out is replaced by a copy of one next to it), so the moshed copy
//! lines up with the clip, its sound and what's tracked. The copy is cropped and sized as the picture
//! is read, played back to a file in the temp folder, and kept: the same clip moshed the same way
//! again is there at once.

use crate::Frame;
use crate::tape::{Looker, cuts};
use crate::video::{Decoder, Input};
use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Stdio;

/// Where the mosh happens.
#[derive(Clone, Debug, PartialEq)]
pub enum When {
	/// the footage's own cuts, found here
	Cuts,
	/// these moments (source seconds): the song's beats
	Beats(Vec<f64>),
}

/// What to do to the stream, by frame (from the copy's first).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
	/// frames whose key frames are taken out
	pub drop: Vec<usize>,
	/// frames whose motion is played again over the `burst` frames after them
	pub bloom: Vec<usize>,
	pub burst: usize,
	/// frames given a key frame that's kept (putting the picture right)
	pub mend: Vec<usize>,
}

impl Plan {
	/// The plan for `marks` (cut or beat frames) in `frames` frames at `fps`, put right `hold` seconds
	/// on (where that comes before the next mark).
	pub fn new(beats: bool, marks: &[usize], frames: usize, fps: f64, hold: f64) -> Plan {
		let mut marks: Vec<usize> = marks.iter().copied().filter(|&m| m > 0 && m + 1 < frames).collect();
		marks.sort_unstable();
		marks.dedup();
		let hold = ((hold * fps).round() as usize).max(2);
		// a beat's motion played over half the time until it's put right
		let burst = (hold / 2).clamp(2, 60);
		let mend = marks.iter().enumerate().filter_map(|(i, &m)| {
			let at = m + hold.max(burst + 1);
			(at < frames && marks.get(i + 1).is_none_or(|&next| at < next)).then_some(at)
		});
		let mend = mend.collect();
		if beats {
			Plan { drop: Vec::new(), bloom: marks, burst, mend }
		} else {
			Plan { drop: marks, bloom: Vec::new(), burst, mend }
		}
	}

	/// The frames that need a key frame when the clip's encoded.
	fn keys(&self) -> Vec<usize> {
		let mut k: Vec<usize> = self.drop.iter().chain(&self.mend).copied().collect();
		k.sort_unstable();
		k.dedup();
		k
	}
}

/// An MPEG-4 Part 2 stream split into its units, each from its start code (00 00 01) to the next.
fn units(d: &[u8]) -> Vec<&[u8]> {
	let mut starts = Vec::new();
	let mut i = 0;
	while i + 3 <= d.len() {
		if d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1 {
			starts.push(i);
			i += 3;
		} else {
			i += 1;
		}
	}
	starts.iter().enumerate().map(|(k, &s)| &d[s..starts.get(k + 1).copied().unwrap_or(d.len())]).collect()
}

/// A unit's kind of frame, if it's a frame (a VOP): 0 a key frame, 1 a P-frame.
fn frame_kind(u: &[u8]) -> Option<u8> {
	(u.len() > 4 && u[3] == 0xB6).then(|| u[4] >> 6)
}

/// The stream as `plan` says: key frames taken out (with the headers before them), each replaced by
/// the P-frame after it; a beat's frame copied over the frames after it. As many frames as before.
pub fn mosh_stream(d: &[u8], plan: &Plan) -> Vec<u8> {
	let us = units(d);
	let frames: Vec<usize> = us.iter().enumerate().filter(|(_, u)| frame_kind(u).is_some()).map(|(i, _)| i).collect();
	let kind = |f: usize| frames.get(f).and_then(|&u| frame_kind(us[u]));
	// frame number → the frame drawn in its place
	let mut instead: HashMap<usize, usize> = HashMap::new();
	for &f in &plan.drop {
		if f > 0 && kind(f) == Some(0) && kind(f + 1) == Some(1) {
			instead.insert(f, f + 1);
		}
	}
	for &b in &plan.bloom {
		if kind(b) != Some(1) {
			continue;
		}
		for k in 1..=plan.burst {
			if kind(b + k) == Some(1) {
				instead.entry(b + k).or_insert(b);
			}
		}
	}
	let mut out = Vec::with_capacity(d.len());
	let mut held: Vec<&[u8]> = Vec::new();
	let mut f = 0;
	for u in &us {
		if frame_kind(u).is_none() {
			// headers wait to see whether the frame after them stays
			held.push(u);
			continue;
		}
		match instead.get(&f) {
			Some(&g) => {
				if kind(f) != Some(0) {
					held.drain(..).for_each(|h| out.extend_from_slice(h));
				}
				held.clear();
				out.extend_from_slice(us[frames[g]]);
			}
			None => {
				held.drain(..).for_each(|h| out.extend_from_slice(h));
				out.extend_from_slice(u);
			}
		}
		f += 1;
	}
	held.drain(..).for_each(|h| out.extend_from_slice(h));
	out
}

fn ffmpeg() -> std::process::Command {
	let mut c = crate::video::ffmpeg();
	c.args(["-v", "error", "-nostdin", "-y"]);
	c
}

/// The cuts in what `inp` reads of `input`, as frame numbers, found as the look ahead finds them.
fn find_cuts(input: &Path, inp: &Input) -> Result<Vec<usize>> {
	let k = (160.0 / inp.w.max(inp.h) as f64).min(1.0);
	let even = |v: f64| ((v / 2.0).round() as u32 * 2).max(2);
	let mut dec = Decoder::open(input, &Input { w: even(inp.w as f64 * k), h: even(inp.h as f64 * k), ..inp.clone() })?;
	let mut f = Frame::default();
	let mut looker = Looker::default();
	let mut change = Vec::new();
	while dec.next(&mut f)? {
		change.push(looker.change(&f));
	}
	let jumps = vec![false; change.len()];
	Ok(cuts(&change, &jumps).iter().enumerate().filter(|(i, c)| **c && *i > 0).map(|(i, _)| i).collect())
}

/// The moshed copy of what `inp` reads of `input` (cropped and sized as it says), moshed `when`,
/// put right `hold` seconds on: made in the temp folder, or found there from before.
pub fn prepare(input: &Path, inp: &Input, when: &When, hold: f64) -> Result<PathBuf> {
	let key = format!("{} {:?} {when:?} {hold:.2}", input.display(), inp);
	let mut h: u64 = 0xcbf2_9ce4_8422_2325;
	for b in key.bytes() {
		h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
	}
	let dir = std::env::temp_dir().join("spotter-mosh");
	std::fs::create_dir_all(&dir)?;
	let out = dir.join(format!("{h:016x}.mp4"));
	if out.exists() {
		return Ok(out);
	}
	let fps = inp.fps;
	// how many frames there'll be, and where the marks fall
	let (marks, frames) = match when {
		When::Cuts => {
			let c = find_cuts(input, inp)?;
			(c, usize::MAX)
		}
		When::Beats(b) => (
			b.iter().map(|t| ((t - inp.start) * fps).round()).filter(|&f| f >= 1.0).map(|f| f as usize).collect(),
			usize::MAX,
		),
	};
	let plan = Plan::new(matches!(when, When::Beats(_)), &marks, frames, fps, hold);
	// 1. the clip encoded, key frames where they're wanted and nowhere else
	let raw = dir.join(format!("{h:016x}.m4v"));
	let keys: Vec<String> = plan.keys().iter().map(|&k| format!("{:.4}", (k as f64 - 0.3) / fps)).collect();
	let mut c = ffmpeg();
	c.args([
		"-f",
		"rawvideo",
		"-pix_fmt",
		"rgba",
		"-s",
		&format!("{}x{}", inp.w, inp.h),
		"-r",
		&format!("{fps}"),
		"-i",
		"-",
		"-an",
		"-c:v",
		"mpeg4",
		"-q:v",
		"2",
		"-g",
		"600",
		"-bf",
		"0",
		"-sc_threshold",
		"1000000000",
	]);
	if !keys.is_empty() {
		c.args(["-force_key_frames", &keys.join(",")]);
	}
	c.args(["-f", "m4v"]).arg(&raw);
	let mut enc = c
		.stdin(Stdio::piped())
		.stderr(Stdio::piped())
		.spawn()
		.context("couldn't run ffmpeg to encode the clip for moshing")?;
	let mut dec = Decoder::open(input, inp)?;
	let mut f = Frame::default();
	let mut n = 0usize;
	{
		let mut pipe = enc.stdin.take().unwrap();
		while dec.next(&mut f)? {
			if pipe.write_all(&f.rgba).is_err() {
				break;
			}
			n += 1;
		}
	}
	let done = enc.wait_with_output()?;
	if !done.status.success() || n == 0 {
		bail!("encoding the clip for moshing failed: {}", String::from_utf8_lossy(&done.stderr).trim());
	}
	// 2. its frames rearranged
	let stream = std::fs::read(&raw)?;
	let moshed = mosh_stream(&stream, &plan);
	let bent = dir.join(format!("{h:016x}.moshed.m4v"));
	std::fs::write(&bent, &moshed)?;
	let _ = std::fs::remove_file(&raw);
	// 3. played back (the decoder told to carry on through what it finds wrong) into a file that
	//    seeks quickly
	let part = out.with_extension("part.mp4");
	let ok = ffmpeg()
		.args(["-fflags", "+genpts", "-err_detect", "ignore_err", "-r", &format!("{fps}"), "-f", "m4v", "-i"])
		.arg(&bent)
		.args([
			"-c:v",
			"libx264",
			"-crf",
			"12",
			"-preset",
			"veryfast",
			"-g",
			"30",
			"-pix_fmt",
			"yuv420p",
			"-r",
			&format!("{fps}"),
		])
		.arg(&part)
		.stderr(Stdio::piped())
		.output()
		.context("couldn't run ffmpeg to play the moshed clip back")?;
	let _ = std::fs::remove_file(&bent);
	if !ok.status.success() {
		let _ = std::fs::remove_file(&part);
		bail!("playing the moshed clip back failed: {}", String::from_utf8_lossy(&ok.stderr).trim());
	}
	std::fs::rename(&part, &out)?;
	Ok(out)
}

#[cfg(test)]
mod tests {
	use super::*;

	/// A unit: start code, kind, payload.
	fn unit(code: u8, first: u8, tag: u8) -> Vec<u8> {
		vec![0, 0, 1, code, first, tag]
	}

	#[test]
	fn key_frames_go_and_frames_stay() {
		// headers, I, P, P, headers, I (a cut), P, P
		let (i, p) = (0b0000_0000, 0b0100_0000);
		let parts = [
			unit(0xB0, 1, 0),
			unit(0xB6, i, 10),
			unit(0xB6, p, 11),
			unit(0xB6, p, 12),
			unit(0xB0, 1, 1),
			unit(0xB6, i, 13),
			unit(0xB6, p, 14),
			unit(0xB6, p, 15),
		];
		let s: Vec<u8> = parts.concat();
		let plan = Plan { drop: vec![3], ..Plan::default() };
		let out = mosh_stream(&s, &plan);
		let us = units(&out);
		let tags: Vec<u8> = us.iter().map(|u| u[5]).collect();
		// the first headers and key frame stay; the cut's headers go and its key frame is the P after it
		assert_eq!(tags, vec![0, 10, 11, 12, 14, 14, 15]);
		assert_eq!(us.iter().filter(|u| frame_kind(u).is_some()).count(), 6);
		// a beat at frame 1: its motion over the two after it, but a key frame stays a key frame
		let plan = Plan { bloom: vec![1], burst: 2, ..Plan::default() };
		let tags: Vec<u8> = units(&mosh_stream(&s, &plan)).iter().map(|u| u[5]).collect();
		assert_eq!(tags, vec![0, 10, 11, 11, 1, 13, 14, 15]);
	}

	#[test]
	fn plans_mend_before_the_next_mark() {
		let p = Plan::new(false, &[0, 30, 40, 200], 300, 30.0, 2.0);
		assert_eq!(p.drop, vec![30, 40, 200]);
		// 30 + 60 would be past 40; 40 + 60 = 100 is before 200; 200 + 60 = 260 is in the clip
		assert_eq!(p.mend, vec![100, 260]);
		let b = Plan::new(true, &[15, 90], 300, 30.0, 1.0);
		assert_eq!((b.bloom.clone(), b.burst, b.mend.clone()), (vec![15, 90], 15, vec![45, 120]));
	}
}
