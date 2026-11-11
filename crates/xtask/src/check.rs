//! What a change has to pass before it's committed. Besides formatting, clippy and the tests, a
//! comment runs to five lines at most (a module's opening may be longer), and comments in Rust say
//! GPU and CPU. Words this clone keeps out go in .git/info/words, one to a line: what it is, a tab,
//! and a regular expression (only what's after // counts in Rust).

use anyhow::{Result, bail};
use regex::Regex;
use std::process::Command;

pub fn run(quick: bool) -> Result<()> {
	let mut failed = Vec::new();
	let mut step = |what: &str, args: &[&str]| {
		println!("== {what}");
		if !crate::cargo().args(args).status().is_ok_and(|s| s.success()) {
			failed.push(what.to_string());
		}
	};
	step("formatting", &["fmt", "--all", "--check"]);
	step("clippy", &["clippy", "--release", "--workspace", "--all-targets", "--", "-D", "warnings"]);
	if !quick {
		step("tests", &["test", "--release", "--workspace"]);
	}
	println!("== writing");
	if writing()? > 0 {
		failed.push("writing".into());
	}
	if !failed.is_empty() {
		bail!("failed: {}", failed.join(", "));
	}
	println!("all passed");
	Ok(())
}

/// How many lines break the writing rules, each one printed.
fn writing() -> Result<usize> {
	let mut rules =
		vec![("user words in code (GPU, CPU)".to_string(), Regex::new("the (graphics card|processor)")?, true)];
	if let Ok(words) = std::fs::read_to_string(".git/info/words") {
		for line in words.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')) {
			if let Some((what, pattern)) = line.split_once('\t') {
				rules.push((what.to_string(), Regex::new(pattern)?, false));
			}
		}
	}
	let listed = Command::new("git").args(["ls-files", "*.rs", "*.md", "*.py", "*.yml", "*.toml"]).output()?;
	let mut found = 0;
	for f in String::from_utf8_lossy(&listed.stdout).lines() {
		if f.ends_with("onnx/proto.rs") {
			continue;
		}
		let rust = f.ends_with(".rs");
		let text = std::fs::read_to_string(f)?;
		let mut run = 0;
		// (an empty line after the last, so a comment at the very end is measured too)
		for (i, line) in text.lines().chain([""]).enumerate() {
			// in Rust only comments count; the command line's help is for users, so it keeps their words
			let said = if rust { line.find("//").map_or("", |at| &line[at..]) } else { line };
			for (what, pattern, rust_only) in &rules {
				if *rust_only && (!rust || f == "crates/spotter/src/main.rs") {
					continue;
				}
				if pattern.is_match(said) {
					println!("{f}:{}: {what}: {}", i + 1, line.trim());
					found += 1;
				}
			}
			let t = line.trim_start();
			if rust && t.starts_with("//") && !t.starts_with("//!") {
				run += 1;
			} else {
				if run > 5 {
					println!("{f}:{}: a comment of {run} lines", i + 1 - run);
					found += 1;
				}
				run = 0;
			}
		}
	}
	Ok(found)
}
