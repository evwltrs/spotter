//! Where spotter keeps what it downloads and what you make: one folder, `%LOCALAPPDATA%\spotter` on
//! Windows and `~/.cache/spotter` elsewhere, or `SPOTTER_HOME`. `SPOTTER_MODELS` moves the models.

use std::path::PathBuf;

pub fn home() -> PathBuf {
	if let Some(d) = std::env::var_os("SPOTTER_HOME") {
		return d.into();
	}
	let base = std::env::var_os("LOCALAPPDATA")
		.map(PathBuf::from)
		.or_else(|| std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from))
		.or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
		.unwrap_or_else(std::env::temp_dir);
	base.join("spotter")
}

pub fn models() -> PathBuf {
	std::env::var_os("SPOTTER_MODELS").map(PathBuf::from).unwrap_or_else(|| home().join("models"))
}

/// Programs spotter downloads for itself: yt-dlp, and ffmpeg when asked.
pub fn bin() -> PathBuf {
	home().join("bin")
}

/// `name` as a program's file name here (`ffmpeg.exe` on Windows).
pub fn exe(name: &str) -> String {
	if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

/// `<file>.part`: where a file is written before it's renamed into place, once whole.
pub fn part(path: &std::path::Path) -> PathBuf {
	let mut s = path.as_os_str().to_owned();
	s.push(".part");
	s.into()
}
