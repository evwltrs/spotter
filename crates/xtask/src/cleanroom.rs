//! A first run on a computer with nothing on it: the zip unpacked into a fresh folder and run there with
//! no ffmpeg, Python or models of its own (only Windows' folders on the PATH). doctor has to say what's
//! missing, setup has to get it, and a render has to come out.

use anyhow::{Context, Result, ensure};
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn run(zip: Option<PathBuf>) -> Result<()> {
	let zip = match zip {
		Some(z) => z,
		None => newest_zip()?,
	};
	let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos() as u32;
	let room = std::env::temp_dir().join(format!("spotter-cleanroom-{:08x}", stamp ^ std::process::id()));
	let app = room.join("app");
	std::fs::create_dir_all(&app)?;
	crate::run(crate::tar().arg("-x").arg("-f").arg(&zip).arg("-C").arg(&app))?;
	let home = room.join("home");
	let windows = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
	let path = format!(r"{windows}\System32;{windows};{windows}\System32\WindowsPowerShell\v1.0");
	let bare = |program: &Path| {
		let mut c = Command::new(program);
		c.env("SPOTTER_HOME", &home)
			.env("SPOTTER_PYTHON", room.join("no-python.exe"))
			.env("SPOTTER_SOUND_GAIN", "0")
			.env("PATH", &path);
		for v in ["SPOTTER_MODELS", "SPOTTER_FFMPEG", "SPOTTER_FFPROBE", "SPOTTER_YTDLP"] {
			c.env_remove(v);
		}
		c
	};
	let cli = app.join("spotter.exe");
	let doctor = bare(&cli).arg("doctor").status()?;
	ensure!(!doctor.success(), "doctor should say ffmpeg is missing");
	crate::run(bare(&cli).arg("setup")).context("setup failed")?;
	crate::run(bare(&cli).arg("doctor")).context("doctor still finds something missing after setup")?;
	// a clip to render: three seconds of the test pattern from the ffmpeg setup just got
	let clip = room.join("pattern.mp4");
	crate::run(
		bare(&home.join("bin").join("ffmpeg.exe"))
			.args(["-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30", "-t", "3"])
			.args(["-pix_fmt", "yuv420p"])
			.arg(&clip),
	)?;
	let out = room.join("smoke.mp4");
	crate::run(
		bare(&cli).arg("render").arg(&clip).args(["--look", "smoke", "--length", "2", "--fast", "-o"]).arg(&out),
	)?;
	ensure!(out.exists(), "the render made nothing");
	println!("a first run works: {}", room.display());
	Ok(())
}

fn newest_zip() -> Result<PathBuf> {
	std::fs::read_dir("dist")
		.context("no dist folder: run cargo xtask package first")?
		.filter_map(|e| e.ok())
		.filter(|e| e.path().extension().is_some_and(|x| x == "zip"))
		.max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
		.map(|e| e.path())
		.context("no zip in dist")
}
