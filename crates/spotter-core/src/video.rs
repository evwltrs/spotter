//! Reading and writing video through ffmpeg: the one `SPOTTER_FFMPEG` and `SPOTTER_FFPROBE` name,
//! else the one on the PATH, else spotter's own (`fetch_ffmpeg`). Frames go in and out as raw RGBA over
//! pipes, so any format ffmpeg can read works, and the encoder is ffmpeg's own x264.

use crate::Frame;
use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

/// Which ffmpeg or ffprobe to run.
pub fn tool(name: &str) -> PathBuf {
	if let Some(p) = std::env::var_os(format!("SPOTTER_{}", name.to_uppercase())) {
		return p.into();
	}
	let exe = crate::paths::exe(name);
	let on_path = std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(&exe).is_file()));
	let own = crate::paths::bin().join(&exe);
	if !on_path && own.is_file() { own } else { name.into() }
}

fn command(name: &str) -> Command {
	let mut c = Command::new(tool(name));
	crate::quiet(&mut c);
	c
}

/// ffmpeg, with no window of its own.
pub fn ffmpeg() -> Command {
	command("ffmpeg")
}

/// Whether there's an ffmpeg to run, and an ffprobe.
pub fn have_ffmpeg() -> bool {
	["ffmpeg", "ffprobe"].iter().all(|t| command(t).arg("-version").output().is_ok_and(|o| o.status.success()))
}

/// Gyan Doshi's Windows build of ffmpeg (GPL), its essentials: the zip, its size, its SHA-256.
const FFMPEG: (&str, u64, &str) = (
	"https://github.com/GyanD/codexffmpeg/releases/download/9.0.2/ffmpeg-9.0.2-essentials_build.zip",
	114_768_076,
	"60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba",
);

/// How big the ffmpeg `fetch_ffmpeg` gets is, in bytes.
pub const FFMPEG_BYTES: u64 = FFMPEG.1;

/// Download ffmpeg and ffprobe into spotter's own folder (`paths::bin`), for a computer with none.
/// Only when asked, since it's 115 MB; Windows only (elsewhere a package manager has it).
pub fn fetch_ffmpeg() -> Result<()> {
	if !cfg!(windows) {
		bail!("spotter fetches ffmpeg on Windows only; install it with your package manager");
	}
	let (url, bytes, sha) = FFMPEG;
	let bin = crate::paths::bin();
	let zip = bin.join(url.rsplit('/').next().unwrap());
	crate::model::download("ffmpeg", url, &zip)?;
	let data = std::fs::read(&zip)?;
	let got: String = hmac_sha256::Hash::hash(&data).iter().map(|b| format!("{b:02x}")).collect();
	if data.len() as u64 != bytes || got != sha {
		let _ = std::fs::remove_file(&zip);
		bail!("the ffmpeg download isn't the one expected ({} bytes, SHA-256 {got})", data.len());
	}
	drop(data);
	// Windows' own tar reads zips
	let tar =
		std::env::var_os("SystemRoot").map_or("tar".into(), |r| PathBuf::from(r).join("System32").join("tar.exe"));
	let inside = zip.file_stem().unwrap().to_string_lossy().into_owned();
	let into = bin.join(format!("{inside}.part"));
	std::fs::create_dir_all(&into)?;
	let out = crate::quiet(&mut Command::new(&tar))
		.arg("-xf")
		.arg(&zip)
		.arg("-C")
		.arg(&into)
		.args(["ffmpeg.exe", "ffprobe.exe"].map(|e| format!("{inside}/bin/{e}")))
		.output()
		.with_context(|| format!("couldn't run {}", tar.display()))?;
	if !out.status.success() {
		bail!("couldn't unpack ffmpeg: {}", String::from_utf8_lossy(&out.stderr).trim());
	}
	for e in ["ffmpeg.exe", "ffprobe.exe"] {
		std::fs::rename(into.join(&inside).join("bin").join(e), bin.join(e))?;
	}
	let _ = std::fs::remove_dir_all(&into);
	let _ = std::fs::remove_file(&zip);
	Ok(())
}

/// Before `-i` for anything read over the web: keep to one connection, since ffmpeg's Windows TLS
/// fails on the second one a file with its index at the end needs, and give up after 15 seconds of
/// nothing rather than waiting forever.
fn net(c: &mut Command, input: &Path) {
	let s = input.to_string_lossy();
	if s.starts_with("http://") || s.starts_with("https://") {
		c.args(["-multiple_requests", "1", "-rw_timeout", "15000000"]);
	}
}

/// What a video file holds, as it would be shown (rotation applied).
#[derive(Clone, Debug)]
pub struct Probe {
	pub path: PathBuf,
	pub w: u32,
	pub h: u32,
	/// frames a second; variable-rate video is read at its average rate
	pub fps: f64,
	/// seconds
	pub duration: f64,
	pub audio: bool,
	/// how the stored colours are to be read, from the file's own notes
	pub colour: Colour,
}

impl Probe {
	pub fn frames(&self) -> u64 {
		(self.duration * self.fps).round().max(1.0) as u64
	}
}

/// How a video's colours are to be turned into RGB.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Colour {
	/// the YUV matrix to read with, as ffmpeg names it (bt709, bt470bg, bt2020…); None leaves it to
	/// ffmpeg, which goes by the file's notes
	pub matrix: Option<String>,
	/// high dynamic range (HLG or PQ, as phones record), brought down to ordinary range
	pub hdr: bool,
}

impl Colour {
	/// From ffprobe's notes on a stream. A file with no notes is read the way players read one: as
	/// HD (BT.709) when it's bigger than standard definition, which ffmpeg alone would read as BT.601
	/// and so shift every colour a little on its way through.
	fn from_stream(v: &Value, w: u32, h: u32) -> Self {
		let tag = |k: &str| v[k].as_str().filter(|s| !s.is_empty() && *s != "unknown").map(str::to_string);
		let transfer = tag("color_transfer");
		let hdr = matches!(transfer.as_deref(), Some("arib-std-b67" | "smpte2084"));
		// ffprobe's names for the YUV matrices, as ffmpeg's scaler takes them; anything else (an RGB
		// picture's "gbr", say) is left to ffmpeg
		let known = |m: &str| match m {
			"bt709" => Some("bt709"),
			"smpte170m" | "bt601" => Some("smpte170m"),
			"bt470bg" => Some("bt470"),
			"bt2020nc" | "bt2020c" | "bt2020" => Some("bt2020"),
			"fcc" => Some("fcc"),
			"smpte240m" => Some("smpte240m"),
			_ => None,
		};
		// only YUV has a matrix to read with; an RGB picture is left alone
		let pix = v["pix_fmt"].as_str().unwrap_or("yuv");
		let yuv = pix.starts_with("yuv") || pix.starts_with("yuvj") || pix.starts_with("nv") || pix.starts_with("p01");
		let matrix = match tag("color_space") {
			_ if !yuv => None,
			Some(m) => known(&m).map(str::to_string),
			// mpv's rule
			None if w >= 1280 || h > 576 => Some("bt709".into()),
			None => None,
		};
		Self { matrix, hdr }
	}
}

fn ratio(s: &str) -> Option<f64> {
	let (a, b) = s.split_once('/').unwrap_or((s, "1"));
	let (a, b): (f64, f64) = (a.parse().ok()?, b.parse().ok()?);
	(a > 0.0 && b > 0.0).then_some(a / b)
}

pub fn probe(path: &Path) -> Result<Probe> {
	let s = path.to_string_lossy();
	if !s.contains("://") && !path.exists() {
		bail!("there's no file at {}", path.display());
	}
	let mut c = command("ffprobe");
	c.args(["-v", "error", "-show_streams", "-show_format", "-of", "json"]);
	net(&mut c, path);
	let out = c.arg(path).output().with_context(|| {
		format!("couldn't run {}: ffmpeg isn't on the PATH (spotter setup gets it)", tool("ffprobe").display())
	})?;
	if !out.status.success() {
		// ffprobe starts its message with the file's name again
		let said = String::from_utf8_lossy(&out.stderr);
		let why = said.trim().rsplit(": ").next().unwrap_or_default();
		bail!("ffmpeg couldn't read {}: {why}", path.file_name().unwrap_or(path.as_os_str()).to_string_lossy());
	}
	let j: Value = serde_json::from_slice(&out.stdout)?;
	let streams = j["streams"].as_array().cloned().unwrap_or_default();
	let v = streams
		.iter()
		.find(|s| s["codec_type"] == "video" && s["disposition"]["attached_pic"] != 1)
		.ok_or_else(|| anyhow!("{} has no video in it", path.display()))?;
	let (mut w, mut h) = (v["width"].as_u64().unwrap_or(0) as u32, v["height"].as_u64().unwrap_or(0) as u32);
	if w == 0 || h == 0 {
		bail!("{} has a video stream with no size", path.display());
	}
	// phones store portrait video sideways with a note to turn it; ffmpeg turns it when decoding
	let turned = v["side_data_list"]
		.as_array()
		.into_iter()
		.flatten()
		.filter_map(|d| d["rotation"].as_f64())
		.chain(v["tags"]["rotate"].as_str().and_then(|r| r.parse().ok()))
		.any(|r: f64| (r.abs() as i64 / 90) % 2 == 1);
	if turned {
		std::mem::swap(&mut w, &mut h);
	}
	let fps = v["avg_frame_rate"]
		.as_str()
		.and_then(ratio)
		.or_else(|| v["r_frame_rate"].as_str().and_then(ratio))
		.filter(|f| *f > 0.5 && *f < 1000.0)
		.unwrap_or(30.0);
	let num = |x: &Value| x.as_str().and_then(|s| s.parse::<f64>().ok());
	let duration = num(&v["duration"]).or_else(|| num(&j["format"]["duration"])).unwrap_or(0.0);
	let audio = streams.iter().any(|s| s["codec_type"] == "audio");
	let colour = Colour::from_stream(v, w, h);
	Ok(Probe { path: path.to_path_buf(), w, h, fps, duration, audio, colour })
}

/// Collects what an ffmpeg process says on stderr, so a failure can say why.
fn drain(child: &mut Child) -> Arc<Mutex<String>> {
	let said = Arc::new(Mutex::new(String::new()));
	if let Some(mut err) = child.stderr.take() {
		let said = said.clone();
		std::thread::spawn(move || {
			let mut buf = String::new();
			let _ = err.read_to_string(&mut buf);
			*said.lock().unwrap() = buf;
		});
	}
	said
}

/// Which part of the source to read, and at what size.
#[derive(Clone, Debug)]
pub struct Input {
	/// seconds into the source to start from
	pub start: f64,
	/// how many seconds to read; None is to the end
	pub length: Option<f64>,
	/// part of the picture to keep, in source pixels (x, y, w, h); None is all of it
	pub crop: Option<[u32; 4]>,
	/// the size frames come out at
	pub w: u32,
	pub h: u32,
	/// frames a second to read at; frames are repeated or dropped to keep to it
	pub fps: f64,
	/// how to read the colours (from `Probe::colour`)
	pub colour: Colour,
}

pub struct Decoder {
	child: Child,
	out: ChildStdout,
	pub w: u32,
	pub h: u32,
	said: Arc<Mutex<String>>,
}

impl Decoder {
	pub fn open(path: &Path, r: &Input) -> Result<Self> {
		let mut filters = Vec::new();
		if let Some([x, y, w, h]) = r.crop {
			filters.push(format!("crop={w}:{h}:{x}:{y}"));
		}
		filters.push(format!("fps={}", r.fps));
		if r.colour.hdr {
			// to linear light, into BT.709's colours, Hable's curve down to ordinary range, and back
			filters.push("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p".into());
			filters.push(format!("scale={}:{}:flags=lanczos:in_color_matrix=bt709", r.w, r.h));
		} else if let Some(m) = &r.colour.matrix {
			filters.push(format!("scale={}:{}:flags=lanczos:in_color_matrix={m}", r.w, r.h));
		} else {
			filters.push(format!("scale={}:{}:flags=lanczos", r.w, r.h));
		}
		filters.push("format=rgba".into());
		let mut c = command("ffmpeg");
		c.args(["-v", "error", "-nostdin"]);
		if r.start > 0.0 {
			c.args(["-ss", &format!("{:.3}", r.start)]);
		}
		net(&mut c, path);
		c.arg("-i").arg(path);
		if let Some(t) = r.length {
			c.args(["-t", &format!("{t:.3}")]);
		}
		c.args(["-an", "-sn", "-vf", &filters.join(","), "-f", "rawvideo", "-pix_fmt", "rgba", "-"]);
		let mut child = c
			.stdin(Stdio::null())
			.stdout(Stdio::piped())
			.stderr(Stdio::piped())
			.spawn()
			.with_context(|| format!("couldn't run {}", tool("ffmpeg").display()))?;
		let said = drain(&mut child);
		let out = child.stdout.take().unwrap();
		Ok(Self { child, out, w: r.w, h: r.h, said })
	}

	/// The next frame into `f`, or false at the end.
	pub fn next(&mut self, f: &mut Frame) -> Result<bool> {
		let n = (self.w * self.h * 4) as usize;
		f.w = self.w;
		f.h = self.h;
		f.rgba.resize(n, 0);
		let mut at = 0;
		while at < n {
			match self.out.read(&mut f.rgba[at..]) {
				Ok(0) => break,
				Ok(k) => at += k,
				Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
				Err(e) => return Err(e.into()),
			}
		}
		if at == n {
			return Ok(true);
		}
		let status = self.child.wait()?;
		if !status.success() {
			// give the stderr reader a moment to finish
			std::thread::sleep(std::time::Duration::from_millis(50));
			bail!("ffmpeg stopped reading the video: {}", self.said.lock().unwrap().trim());
		}
		Ok(false)
	}
}

impl Drop for Decoder {
	fn drop(&mut self) {
		let _ = self.child.kill();
		let _ = self.child.wait();
	}
}

/// How the file comes out.
#[derive(Clone, Debug)]
pub struct Output {
	pub w: u32,
	pub h: u32,
	pub fps: f64,
	/// x264's constant quality: 0 is lossless, 18 looks it, 23 is its default
	pub crf: u32,
	/// a ceiling in megabits a second, for places that turn down big files; None for none
	pub max_mbps: Option<f64>,
	/// take the sound from this file, from `start` for `length` seconds, slowed or sped by `speed`
	pub sound: Option<(PathBuf, f64, Option<f64>)>,
	/// the stretches of that sound to keep, in seconds from `start`, when some are cut out; empty or
	/// one stretch keeps it all
	pub keep: Vec<(f64, f64)>,
	/// playback speed: 0.5 is half speed. The frames are written at `fps` as they come (read the
	/// source at `fps / speed` to keep motion smooth); this stretches the sound to match
	pub speed: f64,
	/// a song under it, from `from` seconds in, for the `length` seconds the video runs, mixed with
	/// the clip's sound (kept at `keep`, 0 for none)
	pub music: Option<Song>,
}

/// A song for `Output`.
#[derive(Clone, Debug)]
pub struct Song {
	pub path: PathBuf,
	pub from: f64,
	pub length: f64,
	pub gain: f32,
	pub keep: f32,
}

/// How long a file plays, from ffprobe: a song's length, say.
pub fn length(path: &Path) -> Result<f64> {
	let mut c = command("ffprobe");
	c.args(["-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]);
	net(&mut c, path);
	let out = c.arg(path).output().with_context(|| format!("couldn't run {}", tool("ffprobe").display()))?;
	let s = String::from_utf8_lossy(&out.stdout);
	s.trim().parse::<f64>().with_context(|| format!("{} has no length ffprobe can read", path.display()))
}

/// ffmpeg's atempo steps for a speed (each goes from half to double).
fn tempo(speed: f64) -> Vec<String> {
	let mut steps = Vec::new();
	if (speed - 1.0).abs() <= 1e-3 {
		return steps;
	}
	let mut left = speed;
	while left < 0.5 {
		steps.push("atempo=0.5".to_string());
		left /= 0.5;
	}
	while left > 2.0 {
		steps.push("atempo=2".to_string());
		left /= 2.0;
	}
	steps.push(format!("atempo={left}"));
	steps
}

pub struct Encoder {
	child: Child,
	inp: Option<ChildStdin>,
	path: PathBuf,
	said: Arc<Mutex<String>>,
	done: bool,
}

impl Drop for Encoder {
	/// Given up on before it was finished: stop ffmpeg rather than leave it waiting for frames.
	fn drop(&mut self) {
		if !self.done {
			drop(self.inp.take());
			let _ = self.child.kill();
			let _ = self.child.wait();
		}
	}
}

impl Encoder {
	pub fn create(path: &Path, o: &Output) -> Result<Self> {
		let mut c = command("ffmpeg");
		c.args([
			"-v",
			"error",
			"-y",
			"-f",
			"rawvideo",
			"-pix_fmt",
			"rgba",
			"-s",
			&format!("{}x{}", o.w, o.h),
			"-framerate",
			&format!("{}", o.fps),
			"-i",
			"-",
		]);
		// with a song and none of the clip's sound kept, the clip's isn't read at all
		let sound = o.sound.as_ref().filter(|_| o.speed > 0.0 && o.music.as_ref().is_none_or(|m| m.keep > 0.0));
		if let Some((src, start, length)) = sound {
			if *start > 0.0 {
				c.args(["-ss", &format!("{start:.3}")]);
			}
			if let Some(t) = length {
				c.args(["-t", &format!("{t:.3}")]);
			}
			net(&mut c, src);
			c.arg("-i").arg(src);
		}
		if let Some(m) = &o.music {
			if m.from > 0.0 {
				c.args(["-ss", &format!("{:.3}", m.from)]);
			}
			c.args(["-t", &format!("{:.3}", m.length + 0.5)]);
			net(&mut c, &m.path);
			c.arg("-i").arg(&m.path);
		}
		c.args(["-map", "0:v:0"]);
		if let Some(m) = &o.music {
			// the song (faded out over its last second and a half if it's cut short), and the clip's
			// sound under it as it would be without, turned down to `keep`
			let song = if sound.is_some() { 2 } else { 1 };
			let fade = (m.length - 1.5).max(0.0);
			let mut graph = format!("[{song}:a]volume={:.3},afade=t=out:st={fade:.3}:d=1.5[song]", m.gain);
			if sound.is_some() {
				let mut steps = Vec::new();
				if o.keep.len() > 1 {
					let between: Vec<String> =
						o.keep.iter().map(|(a, b)| format!("between(t,{a:.4},{b:.4})")).collect();
					steps.push(format!("aselect='{}',asetpts=N/SR/TB", between.join("+")));
				}
				steps.extend(tempo(o.speed));
				steps.push(format!("volume={:.3}", m.keep));
				graph += &format!(
					";[1:a]{}[clip];[clip][song]amix=inputs=2:duration=longest:normalize=0[sound]",
					steps.join(",")
				);
			} else {
				graph = graph.replace("[song]", "[sound]");
			}
			c.args(["-filter_complex", &graph, "-map", "[sound]", "-c:a", "aac", "-b:a", "192k", "-shortest"]);
		} else if sound.is_some() {
			c.args(["-map", "1:a:0?"]);
			let mut steps = Vec::new();
			if o.keep.len() > 1 {
				let between: Vec<String> = o.keep.iter().map(|(a, b)| format!("between(t,{a:.4},{b:.4})")).collect();
				steps.push(format!("aselect='{}',asetpts=N/SR/TB", between.join("+")));
			}
			if (o.speed - 1.0).abs() > 1e-3 {
				// atempo only goes from half to double in one step
				let mut left = o.speed;
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
			if !steps.is_empty() {
				c.args(["-af", &steps.join(",")]);
			}
			c.args(["-c:a", "aac", "-b:a", "192k", "-shortest"]);
		}
		c.args(["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"]);
		c.args(["-c:v", "libx264", "-preset", "slow", "-profile:v", "high", "-crf", &o.crf.to_string()]);
		if let Some(m) = o.max_mbps {
			c.args(["-maxrate", &format!("{m}M"), "-bufsize", &format!("{}M", m * 2.0)]);
		}
		c.args(["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart"]);
		c.arg(path);
		let mut child = c
			.stdin(Stdio::piped())
			.stdout(Stdio::null())
			.stderr(Stdio::piped())
			.spawn()
			.with_context(|| format!("couldn't run {}", tool("ffmpeg").display()))?;
		let said = drain(&mut child);
		let inp = child.stdin.take();
		Ok(Self { child, inp, path: path.to_path_buf(), said, done: false })
	}

	pub fn write(&mut self, f: &Frame) -> Result<()> {
		let inp = self.inp.as_mut().ok_or_else(|| anyhow!("the encoder is closed"))?;
		if inp.write_all(&f.rgba).is_err() {
			let _ = self.child.wait();
			std::thread::sleep(std::time::Duration::from_millis(50));
			bail!("ffmpeg stopped writing {}: {}", self.path.display(), self.said.lock().unwrap().trim());
		}
		Ok(())
	}

	/// Close the file and wait for ffmpeg to finish it.
	pub fn finish(mut self) -> Result<()> {
		drop(self.inp.take());
		self.done = true;
		let status = self.child.wait()?;
		if !status.success() {
			std::thread::sleep(std::time::Duration::from_millis(50));
			bail!("ffmpeg couldn't finish {}: {}", self.path.display(), self.said.lock().unwrap().trim());
		}
		Ok(())
	}
}

/// One picture (a photo, or a video's first frame) at its own size.
pub fn still(path: &Path) -> Result<Frame> {
	let p = probe(path)?;
	let mut d = Decoder::open(
		path,
		&Input { start: 0.0, length: None, crop: None, w: p.w, h: p.h, fps: p.fps, colour: p.colour.clone() },
	)?;
	let mut f = Frame::default();
	if !d.next(&mut f)? {
		bail!("{} has no frames", path.display());
	}
	Ok(f)
}

/// Save a frame as a PNG (for checking things).
pub fn save_png(f: &Frame, path: &Path) -> Result<()> {
	let mut c = command("ffmpeg");
	c.args([
		"-v",
		"error",
		"-y",
		"-f",
		"rawvideo",
		"-pix_fmt",
		"rgba",
		"-s",
		&format!("{}x{}", f.w, f.h),
		"-i",
		"-",
		"-frames:v",
		"1",
	])
	.arg(path);
	let mut child = c.stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::piped()).spawn()?;
	child.stdin.take().unwrap().write_all(&f.rgba)?;
	let out = child.wait_with_output()?;
	if !out.status.success() {
		bail!("ffmpeg couldn't save {}: {}", path.display(), String::from_utf8_lossy(&out.stderr).trim());
	}
	Ok(())
}
