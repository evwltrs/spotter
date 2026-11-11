//! The release zip: spotter's two programs, DirectML.dll, the README, the licence and the licences
//! of what it's built with and downloads. ffmpeg and the models come the first time they're needed.

use anyhow::{Context, Result, ensure};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub fn run(tag: Option<String>) -> Result<()> {
	let version = crate::version()?;
	if let Some(tag) = tag {
		ensure!(tag == format!("v{version}"), "the tag is {tag}, Cargo.toml says {version}");
	}
	crate::run(crate::cargo().args(["build", "--release", "-p", "spotter", "-p", "spotter-app"]))?;
	third_party()?;
	let name = format!("spotter-{version}-windows-x64");
	let dir = Path::new("dist").join(&name);
	if dir.exists() {
		fs::remove_dir_all(&dir)?;
	}
	fs::create_dir_all(&dir)?;
	let built = ["spotter-app.exe", "spotter.exe", "DirectML.dll"];
	let ours = ["README.md", "LICENSE", "THIRD-PARTY.md"];
	let release = crate::target().join("release");
	for (from, f) in built.iter().map(|f| (release.join(f), f)).chain(ours.iter().map(|f| (f.into(), f))) {
		fs::copy(&from, dir.join(f)).with_context(|| format!("couldn't copy {}", from.display()))?;
	}
	let files = built.iter().chain(&ours);
	let zip = std::env::current_dir()?.join("dist").join(format!("{name}.zip"));
	if zip.exists() {
		fs::remove_file(&zip)?;
	}
	crate::run(crate::tar().arg("-a").arg("-c").arg("-f").arg(&zip).arg("-C").arg(&dir).args(files))?;
	println!("{}  {:.1} MB", zip.display(), fs::metadata(&zip)?.len() as f64 / 1048576.0);
	Ok(())
}

/// THIRD-PARTY.md: what the Windows build is made with, by licence, from `cargo tree`, and what it
/// downloads when it's used.
pub fn third_party() -> Result<()> {
	let tree = crate::cargo()
		.args([
			"tree",
			"-e",
			"normal",
			"--prefix",
			"none",
			"--format",
			"{p}\t{l}",
			"-p",
			"spotter",
			"-p",
			"spotter-app",
		])
		.args(["--target", "x86_64-pc-windows-msvc", "--color", "never"])
		.output()?;
	ensure!(tree.status.success(), "cargo tree failed: {}", String::from_utf8_lossy(&tree.stderr));
	let mut by: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
	for line in String::from_utf8_lossy(&tree.stdout).lines() {
		let Some((p, licence)) = line.split_once('\t') else { continue };
		let mut p = p.split_whitespace();
		let (Some(name), Some(version)) = (p.next(), p.next()) else { continue };
		if name.starts_with("spotter") {
			continue;
		}
		let mut licence = licence.trim().trim_end_matches("(*)").trim().replace('/', " OR ");
		// the same choice however it's spelt: MIT/Apache-2.0, Apache-2.0 OR MIT
		if licence.contains(" OR ") && !licence.contains(" AND ") && !licence.contains('(') {
			let mut each: Vec<&str> = licence.split(" OR ").collect();
			each.sort();
			licence = each.join(" OR ");
		}
		let licence = if licence.is_empty() { "not stated".to_string() } else { licence };
		by.entry(licence).or_default().insert(format!("{name} {version}"));
	}
	let mut groups: Vec<(&String, &BTreeSet<String>)> = by.iter().collect();
	groups.sort_by_key(|(licence, names)| (std::cmp::Reverse(names.len()), *licence));
	let mut out = vec![
		"# third-party".to_string(),
		String::new(),
		"## built in".into(),
		String::new(),
		"- ONNX Runtime (MIT), through the `ort` crate".into(),
		"- DirectML.dll (Microsoft DirectML licence)".into(),
		"- DM Mono, Barlow Semi Condensed, Share Tech Mono, VT323, IBM Plex Sans (SIL Open Font License 1.1)".into(),
		String::new(),
		format!("## rust crates ({})", by.values().map(BTreeSet::len).sum::<usize>()),
		String::new(),
	];
	for (licence, names) in groups {
		let names: Vec<&str> = names.iter().map(String::as_str).collect();
		out.push(format!("**{licence}** ({}): {}.", names.len(), names.join(", ")));
		out.push(String::new());
	}
	out.extend(
		[
			"## downloaded when used",
			"",
			"| what | licence | from |",
			"|---|---|---|",
			"| D-FINE s, m, l, x (COCO and Objects365) | Apache-2.0 | huggingface.co/onnx-community |",
			"| EdgeTAM | Apache-2.0 | huggingface.co/onnx-community/EdgeTAM-ONNX |",
			"| SAM 2.1 large | Apache-2.0 | huggingface.co/onnx-community/sam2.1-hiera-large-ONNX |",
			"| Depth Anything V2 small | Apache-2.0 | huggingface.co/onnx-community/depth-anything-v2-small-ONNX |",
			"| Depth Anything V2 large | CC-BY-NC-4.0 (not for commercial use) | huggingface.co/onnx-community/depth-anything-v2-large-ONNX |",
			"| SegFormer B2 and B5, ADE20K | NVIDIA Source Code License (not for commercial use) | huggingface.co/Xenova |",
			"| Real-ESRGAN general x4v3 | BSD-3-Clause | huggingface.co/jonathanst29/tinier-upscale-models |",
			"| RIFE 4.9 | MIT | huggingface.co/yuvraj108c/rife-onnx |",
			"| ffmpeg 9.0.2, essentials build (only when asked) | GPL-3.0 | github.com/GyanD/codexffmpeg |",
			"| yt-dlp (only for a song from a link) | Unlicense | github.com/yt-dlp/yt-dlp |",
			"",
		]
		.map(str::to_string),
	);
	fs::write("THIRD-PARTY.md", out.join("\n"))?;
	Ok(())
}
