//! The jobs around spotter: checking a change, making the release zip, trying that zip on a computer
//! with nothing on it, and watching spotter run.
//!
//!   cargo xtask check [--quick]
//!   cargo xtask package [--tag v0.1.0]
//!   cargo xtask third-party
//!   cargo xtask cleanroom [zip]
//!   cargo xtask watch [-- what to open]

mod check;
mod cleanroom;
mod package;
mod watch;

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser)]
#[command(about = "spotter's own jobs")]
struct Cli {
	#[command(subcommand)]
	job: Job,
}

#[derive(Subcommand)]
enum Job {
	/// Formatting, clippy with warnings as errors, every test, and the writing.
	Check {
		/// leave out the tests
		#[arg(long)]
		quick: bool,
	},
	/// Build spotter and put what Windows needs to run it in dist/spotter-<version>-windows-x64.zip.
	Package {
		/// stop unless this is v and the version in Cargo.toml
		#[arg(long)]
		tag: Option<String>,
	},
	/// Write THIRD-PARTY.md: what spotter is built with and downloads, by licence.
	ThirdParty,
	/// Run the zip as a first run on a computer with nothing on it.
	Cleanroom {
		/// the zip (the newest in dist/ if not given)
		zip: Option<PathBuf>,
	},
	/// Run spotter-app with a line a second of what it and the computer are doing in watch.log.
	Watch {
		/// passed on to spotter-app: a project to open, say
		#[arg(trailing_var_arg = true, allow_hyphen_values = true)]
		args: Vec<String>,
	},
}

fn main() -> Result<()> {
	std::env::set_current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))?;
	match Cli::parse().job {
		Job::Check { quick } => check::run(quick),
		Job::Package { tag } => package::run(tag),
		Job::ThirdParty => package::third_party(),
		Job::Cleanroom { zip } => cleanroom::run(zip),
		Job::Watch { args } => watch::run(&args),
	}
}

/// The cargo that started this one.
fn cargo() -> Command {
	Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
}

/// Run it, and stop if it fails.
fn run(c: &mut Command) -> Result<()> {
	let status = c.status().with_context(|| format!("couldn't start {:?}", c.get_program()))?;
	if !status.success() {
		bail!("{:?} failed ({status})", c.get_program());
	}
	Ok(())
}

/// spotter's version, from Cargo.toml.
fn version() -> Result<String> {
	let toml = std::fs::read_to_string("Cargo.toml")?;
	toml.lines()
		.find_map(|l| l.strip_prefix("version = \"")?.strip_suffix('"').map(str::to_string))
		.context("no version in Cargo.toml")
}

/// Where cargo builds to.
fn target() -> PathBuf {
	std::env::var_os("CARGO_TARGET_DIR").map_or_else(|| "target".into(), PathBuf::from)
}

/// Windows' own tar, which reads and writes zips (Git's doesn't).
fn tar() -> Command {
	let root = std::env::var_os("SystemRoot").map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from);
	Command::new(root.join("System32").join("tar.exe"))
}
