//! Songs from links, for music under a video: a SoundCloud or YouTube page (anything yt-dlp reads),
//! fetched once, only the sound, into spotter's songs folder (`%LOCALAPPDATA%\spotter\songs`), named
//! after the song with its id, and kept, so the same link again is there at once. yt-dlp is the one
//! on the PATH, or spotter's own copy, downloaded the first time it's needed and brought up to date
//! when a fetch fails (sites change, and yt-dlp follows them). YouTube needs a JavaScript runtime:
//! Node, when it's installed. Also the song's shape, for picking the part of it to use.

use anyhow::{Context, Result, bail};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Where fetched songs are kept.
pub fn songs_dir() -> PathBuf {
	crate::paths::home().join("songs")
}

/// Whether what was typed is a link to fetch a song from, rather than a file.
pub fn is_link(s: &str) -> bool {
	let s = s.trim();
	s.starts_with("http://") || s.starts_with("https://")
}

fn command(p: &Path) -> Command {
	let mut c = Command::new(p);
	crate::quiet(&mut c);
	c
}

/// spotter's own copy of yt-dlp
fn own() -> PathBuf {
	crate::paths::bin().join(crate::paths::exe("yt-dlp"))
}

/// yt-dlp: SPOTTER_YTDLP, spotter's own copy, the one on the PATH, or a copy downloaded now.
fn ytdlp() -> Result<PathBuf> {
	if let Ok(p) = std::env::var("SPOTTER_YTDLP") {
		return Ok(p.into());
	}
	let own = own();
	if own.exists() {
		return Ok(own);
	}
	if command(Path::new("yt-dlp")).arg("--version").output().is_ok_and(|o| o.status.success()) {
		return Ok("yt-dlp".into());
	}
	let name = if cfg!(windows) {
		"yt-dlp.exe"
	} else if cfg!(target_os = "macos") {
		"yt-dlp_macos"
	} else {
		"yt-dlp"
	};
	crate::model::download(
		"yt-dlp",
		&format!("https://github.com/yt-dlp/yt-dlp/releases/latest/download/{name}"),
		&own,
	)
	.context("couldn't download yt-dlp")?;
	#[cfg(unix)]
	{
		use std::os::unix::fs::PermissionsExt;
		std::fs::set_permissions(&own, std::fs::Permissions::from_mode(0o755))?;
	}
	Ok(own)
}

/// Whether Node runs here, for YouTube's player code.
fn node() -> bool {
	command(Path::new("node")).arg("--version").output().is_ok_and(|o| o.status.success())
}

/// The sound of the song at `url`, fetched into the songs folder (or found there): its file.
/// `progress` goes from 0 to 1000 as it downloads.
pub fn fetch(url: &str, progress: Arc<AtomicU32>) -> Result<PathBuf> {
	let dir = songs_dir();
	std::fs::create_dir_all(&dir)?;
	let y = ytdlp()?;
	let node = node();
	let run = || -> Result<PathBuf> {
		let mut c = command(&y);
		c.args(["--no-playlist", "-f", "bestaudio/best", "--progress", "--newline", "--print", "after_move:filepath"]);
		if node {
			c.args(["--js-runtimes", "node"]);
		}
		c.arg("-o").arg(dir.join("%(title).80B [%(id)s].%(ext)s")).arg(url.trim());
		let mut child = c
			.stdin(Stdio::null())
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
			.with_context(|| format!("couldn't run {}", y.display()))?;
		// progress comes on either stream; what went wrong on the second
		let said = {
			let (mut err, progress) = (child.stderr.take().unwrap(), progress.clone());
			std::thread::spawn(move || {
				let mut all = String::new();
				let _ = err.read_to_string(&mut all);
				for line in all.lines() {
					note_progress(line, &progress);
				}
				all
			})
		};
		let mut got = None;
		for line in BufReader::new(child.stdout.take().unwrap()).lines().map_while(Result::ok) {
			note_progress(&line, &progress);
			let p = PathBuf::from(line.trim());
			if p.is_absolute() && p.exists() {
				got = Some(p);
			}
		}
		let ok = child.wait()?.success();
		let said = said.join().unwrap_or_default();
		match got {
			Some(p) if ok => Ok(p),
			_ => {
				let why: Vec<&str> = said.lines().filter(|l| l.starts_with("ERROR")).collect();
				bail!(
					"{}",
					if why.is_empty() {
						said.lines().last().unwrap_or("yt-dlp gave nothing").to_string()
					} else {
						why.join("; ")
					}
				)
			}
		}
	};
	match run() {
		Ok(p) => Ok(p),
		// sites change and yt-dlp follows them: bring spotter's own copy up to date, and try again
		Err(e) if y == own() => {
			let updated = command(&y).arg("-U").output().is_ok_and(|o| o.status.success());
			if updated { run().map_err(|_| e) } else { Err(e) }
		}
		Err(e) => Err(e),
	}
}

/// "[download]  45.3% of 4.12MiB at …": 453.
fn note_progress(line: &str, progress: &AtomicU32) {
	if let Some(rest) = line.strip_prefix("[download]")
		&& let Some((n, _)) = rest.trim_start().split_once('%')
		&& let Ok(p) = n.trim().parse::<f32>()
	{
		progress.store((p * 10.0).round().clamp(0.0, 1000.0) as u32, Ordering::Relaxed);
	}
}

/// The song's shape: how loud each of `n` stretches of it is at its loudest, 0-1 (the loudest 1).
pub fn shape(path: &Path, n: usize) -> Result<Vec<f32>> {
	let out = crate::video::ffmpeg()
		.args(["-v", "error", "-nostdin", "-i"])
		.arg(path)
		.args(["-vn", "-ac", "1", "-ar", "2000", "-f", "f32le", "-"])
		.output()
		.context("couldn't run ffmpeg")?;
	if !out.status.success() {
		bail!("ffmpeg couldn't read {}", path.display());
	}
	let s: Vec<f32> = out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]).abs()).collect();
	if s.is_empty() {
		bail!("{} has no sound", path.display());
	}
	let n = n.max(1);
	let mut peaks: Vec<f32> = (0..n)
		.map(|i| {
			s[i * s.len() / n..((i + 1) * s.len() / n).max(i * s.len() / n + 1).min(s.len())]
				.iter()
				.copied()
				.fold(0.0, f32::max)
		})
		.collect();
	let top = peaks.iter().copied().fold(1e-6, f32::max);
	for p in &mut peaks {
		*p /= top;
	}
	Ok(peaks)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn progress_and_links() {
		let p = AtomicU32::new(0);
		note_progress("[download]  45.3% of    4.12MiB at  1.20MiB/s ETA 00:02", &p);
		assert_eq!(p.load(Ordering::Relaxed), 453);
		note_progress("[youtube] jNQXAC9IVRw: Downloading webpage", &p);
		assert_eq!(p.load(Ordering::Relaxed), 453);
		assert!(is_link(" https://soundcloud.com/forss/flickermood") && !is_link("C:\\music\\a.mp3"));
	}
}
