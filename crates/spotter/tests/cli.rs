//! The command line as someone types it: what it prints, what it refuses, and a short render when
//! there's an ffmpeg to make one with.

use spotter_core::Look;
use std::process::{Command, Output};

fn spotter(args: &[&str]) -> Output {
	Command::new(env!("CARGO_BIN_EXE_spotter")).args(args).output().unwrap()
}

fn said(o: &Output) -> String {
	String::from_utf8_lossy(&o.stdout).into_owned() + &String::from_utf8_lossy(&o.stderr)
}

#[test]
fn it_says_its_version_and_lists_every_look() {
	let v = spotter(&["--version"]);
	assert!(v.status.success() && said(&v).contains(env!("CARGO_PKG_VERSION")), "{}", said(&v));
	let looks = spotter(&["looks"]);
	// yours are marked after two spaces
	let listed: Vec<String> =
		String::from_utf8_lossy(&looks.stdout).lines().map(|l| l.split("  ").next().unwrap_or(l).to_string()).collect();
	for p in Look::presets() {
		assert!(listed.iter().any(|l| l == p), "{p} isn't listed");
	}
}

#[test]
fn a_look_printed_reads_back_the_same() {
	let o = spotter(&["looks", "heat"]);
	assert!(o.status.success(), "{}", said(&o));
	let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
	assert_eq!(Look::revive(&v), Look::preset("heat").unwrap());
}

#[test]
fn what_it_cant_do_it_says_and_fails() {
	for (args, want) in [
		(&["looks", "no such look"][..], "no look called"),
		(&["render", "clip.mp4", "--look", "no such look"][..], "no look called"),
		(&["render", "clip.mp4", "--set", "nosuch=1"][..], "no setting called"),
		(&["model", "fetch", "no-such-model"][..], "doesn't know a model"),
	] {
		let o = spotter(args);
		assert!(!o.status.success() && said(&o).contains(want), "{args:?}: {}", said(&o));
	}
}

#[test]
fn a_short_render_comes_out() {
	if !spotter_core::video::have_ffmpeg() {
		eprintln!("no ffmpeg here, so no render");
		return;
	}
	// a second of ffmpeg's test pattern, so the test needs no clip of its own
	let clip = std::env::temp_dir().join(format!("spotter-cli-{}-in.mp4", std::process::id()));
	let pattern = spotter_core::video::ffmpeg()
		.args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30", "-t", "1"])
		.args(["-pix_fmt", "yuv420p"])
		.arg(&clip)
		.status();
	assert!(pattern.is_ok_and(|s| s.success()), "ffmpeg didn't make a test pattern");
	let out = std::env::temp_dir().join(format!("spotter-cli-{}.mp4", std::process::id()));
	let o = spotter(&[
		"render",
		clip.to_str().unwrap(),
		"--look",
		"smoke",
		"--device",
		"off",
		"--length",
		"0.5",
		"-o",
		out.to_str().unwrap(),
	]);
	let made = std::fs::metadata(&out).map_or(0, |m| m.len());
	let _ = std::fs::remove_file(&out);
	let _ = std::fs::remove_file(&clip);
	assert!(o.status.success() && made > 1000, "{}", said(&o));
}
