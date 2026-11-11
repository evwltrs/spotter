//! Videos from files.1nf.me, the way the site's tracker picks them: every video on the server, found
//! by walking copyparty's `?ls` listings, less the folders the random pick stays out of, shuffled, and
//! nothing played twice until everything has been. ffmpeg reads them straight from the server.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

pub const FILES: &str = "https://files.1nf.me";
/// folders on files.1nf.me that the random pick stays out of, as on the site
pub const LEFT_OUT: [&str; 2] = ["aviation/flightsim/", "ski/"];
/// research has its own pages and isn't browsable on files.1nf.me
const RESEARCH: &str = "research/";
/// the extensions the site's tracker picks from
const VIDEO: [&str; 3] = ["mp4", "webm", "m4v"];
/// how long a walk of the server is good for, as on the site
const MAX_AGE: u64 = 30 * 60;
const PARALLEL: usize = 6;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Clip {
	/// percent-encoded path from the server root
	pub path: String,
	/// the file's name without its extension, decoded
	pub name: String,
	pub bytes: u64,
}

impl Clip {
	pub fn url(&self) -> String {
		format!("{FILES}/{}", self.path)
	}

	/// Long enough that it's best started somewhere in the middle rather than at a title card.
	pub fn wanders(&self) -> bool {
		self.bytes > 200_000_000
	}
}

#[derive(Deserialize)]
struct Entry {
	href: String,
	#[serde(default)]
	sz: u64,
}

#[derive(Deserialize)]
struct Listing {
	#[serde(default)]
	dirs: Vec<Entry>,
	#[serde(default)]
	files: Vec<Entry>,
}

#[derive(Serialize, Deserialize, Default)]
struct Saved {
	at: u64,
	clips: Vec<Clip>,
}

fn home() -> PathBuf {
	crate::paths::home()
}

fn now() -> u64 {
	SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A percent-encoded path, readable.
pub fn decode(s: &str) -> String {
	let b = s.as_bytes();
	let mut out = Vec::with_capacity(b.len());
	let mut i = 0;
	while i < b.len() {
		let hex = |c: u8| (c as char).to_digit(16);
		if b[i] == b'%'
			&& i + 2 < b.len()
			&& let (Some(h), Some(l)) = (hex(b[i + 1]), hex(b[i + 2]))
		{
			out.push((h * 16 + l) as u8);
			i += 3;
		} else {
			out.push(b[i]);
			i += 1;
		}
	}
	String::from_utf8_lossy(&out).into_owned()
}

fn listing(dir: &str) -> Result<Listing> {
	let url = format!("{FILES}/{dir}?ls");
	let out = crate::quiet(&mut Command::new("curl"))
		.args(["-sf", "--compressed", "--max-time", "15", "-H", "accept: application/json"])
		.arg(&url)
		.output()
		.context("couldn't run curl")?;
	if !out.status.success() {
		bail!("{url} didn't answer");
	}
	Ok(serde_json::from_slice(&out.stdout)?)
}

/// Every video on files.1nf.me the site's tracker would pick from, walking the server at most every
/// half hour.
pub fn catalog() -> Result<Vec<Clip>> {
	let saved_at = home().join("files.json");
	if let Ok(text) = std::fs::read_to_string(&saved_at)
		&& let Ok(s) = serde_json::from_str::<Saved>(&text)
		&& now().saturating_sub(s.at) < MAX_AGE
	{
		return Ok(s.clips);
	}
	let mut clips = Vec::new();
	let mut queue = vec![String::new()];
	let mut answered = false;
	while !queue.is_empty() {
		// a few folders at a time; each one's subfolders join the queue
		let batch: Vec<String> = queue.drain(..queue.len().min(PARALLEL)).collect();
		let results: Vec<(String, Result<Listing>)> = std::thread::scope(|s| {
			let handles: Vec<_> = batch.iter().map(|dir| s.spawn(move || (dir.clone(), listing(dir)))).collect();
			handles.into_iter().map(|h| h.join().expect("a listing thread")).collect()
		});
		for (dir, l) in results {
			let Ok(l) = l else { continue };
			answered = true;
			for d in l.dirs {
				if dir.is_empty() && d.href == RESEARCH {
					continue;
				}
				queue.push(format!("{dir}{}", d.href));
			}
			for f in l.files {
				let ext = f.href.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
				if !VIDEO.contains(&ext.as_str()) {
					continue;
				}
				let file = decode(&f.href);
				let name = file.rsplit_once('.').map_or(file.clone(), |(n, _)| n.to_string());
				clips.push(Clip { path: format!("{dir}{}", f.href), name, bytes: f.sz });
			}
		}
	}
	if !answered {
		bail!("{FILES} didn't answer");
	}
	clips.sort_by(|a, b| a.path.cmp(&b.path));
	let _ = std::fs::create_dir_all(home());
	let _ = std::fs::write(&saved_at, serde_json::to_string(&Saved { at: now(), clips: clips.clone() })?);
	Ok(clips)
}

/// A small random number generator, seeded from the clock, for shuffling.
pub struct Dice(u64);

impl Default for Dice {
	fn default() -> Self {
		let t = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1);
		Self(t ^ 0x9e37_79b9_7f4a_7c15)
	}
}

impl Dice {
	/// 0 to below n
	pub fn below(&mut self, n: usize) -> usize {
		// splitmix64
		self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
		let mut z = self.0;
		z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
		z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
		((z ^ (z >> 31)) % n.max(1) as u64) as usize
	}

	/// 0 to 1
	pub fn unit(&mut self) -> f64 {
		self.below(1 << 30) as f64 / (1u64 << 30) as f64
	}
}

/// A video the random pick hasn't played yet, as the site's "another" chooses: none of the left-out
/// folders, and once everything has played, round again. It's remembered as played straight away.
pub fn another() -> Result<Clip> {
	let all: Vec<Clip> = catalog()?.into_iter().filter(|c| !LEFT_OUT.iter().any(|d| c.path.starts_with(d))).collect();
	if all.is_empty() {
		bail!("there are no videos on {FILES} to pick from");
	}
	let memory = home().join("played.json");
	let mut played: Vec<String> =
		std::fs::read_to_string(&memory).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
	let mut fresh: Vec<&Clip> = all.iter().filter(|c| !played.contains(&c.path)).collect();
	if fresh.is_empty() {
		// that's all of them: go round again, just not starting with the last one
		let last = played.pop();
		played.clear();
		fresh = all.iter().filter(|c| Some(&c.path) != last.as_ref()).collect();
		if fresh.is_empty() {
			fresh = all.iter().collect();
		}
	}
	let pick = fresh[Dice::default().below(fresh.len())].clone();
	played.push(pick.path.clone());
	let _ = std::fs::create_dir_all(home());
	let _ = std::fs::write(&memory, serde_json::to_string(&played)?);
	Ok(pick)
}

/// What a video argument names: a file on this computer, a web address, `files:<path>` for a file on
/// files.1nf.me (as the site's share links write `f:`), or `random` for one the site would pick.
/// Gives back what ffmpeg should read and a name for it.
pub fn resolve(arg: &str) -> Result<(String, String, Option<Clip>)> {
	if arg == "random" {
		let c = another()?;
		return Ok((c.url(), c.name.clone(), Some(c)));
	}
	if let Some(path) = arg.strip_prefix("files:").or_else(|| arg.strip_prefix("f:")) {
		let path = path.trim_start_matches('/');
		let clip = catalog().ok().and_then(|all| all.into_iter().find(|c| c.path == path || decode(&c.path) == path));
		let encoded = clip.as_ref().map_or_else(|| path.replace(' ', "%20"), |c| c.path.clone());
		let file = decode(&encoded);
		let name =
			file.rsplit('/').next().unwrap_or(&file).rsplit_once('.').map_or(file.clone(), |(n, _)| n.to_string());
		return Ok((format!("{FILES}/{encoded}"), name, clip));
	}
	let name = arg.rsplit(['/', '\\']).next().unwrap_or(arg);
	let name = decode(name.split('?').next().unwrap_or(name));
	let name = name.rsplit_once('.').map_or(name.clone(), |(n, _)| n.to_string());
	Ok((arg.to_string(), name, None))
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn names_come_out_decoded() {
		assert_eq!(decode("Articulated%20truck%20night.mp4"), "Articulated truck night.mp4");
		assert_eq!(decode("100%"), "100%");
		let (url, name, _) = resolve("C:\\clips\\a b.mp4").unwrap();
		assert_eq!((url.as_str(), name.as_str()), ("C:\\clips\\a b.mp4", "a b"));
	}

	#[test]
	fn dice_stay_in_range() {
		let mut d = Dice::default();
		for _ in 0..1000 {
			assert!(d.below(7) < 7);
			assert!((0.0..1.0).contains(&d.unit()));
		}
	}
}
