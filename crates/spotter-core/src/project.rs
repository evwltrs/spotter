//! A project: one clip and everything done to it, saved as a `.spotter` file (JSON) to come back to
//! or render from the command line. The look is kept whole, so a project made before a setting
//! existed still opens (it gets that setting's default).

use crate::edit::Edit;
use crate::look::Look;
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

pub const EXTENSION: &str = "spotter";

/// The layout of the file this spotter writes. One from a newer spotter is refused, not half read.
pub const VERSION: u32 = 1;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Project {
	/// the file's layout (`VERSION`); 0 in files saved before it was written down, which are the same
	pub version: u32,
	/// what ffmpeg reads: a file or a web address
	pub source: String,
	/// shown in the {file} token
	pub name: String,
	/// the look it started from, for the window's looks row
	pub preset: String,
	/// the look, whole
	pub look: Value,
	pub edit: Edit,
	/// the size to export at, when it's one of the posting shapes
	pub size: Option<(u32, u32)>,
	pub upscale: bool,
	/// the detector (x, x365…)
	pub model: String,
	/// the in and out points, seconds of the source
	#[serde(rename = "in")]
	pub start: Option<f64>,
	#[serde(rename = "out")]
	pub end: Option<f64>,
}

impl Project {
	pub fn look(&self) -> Look {
		Look::revive(&self.look)
	}

	pub fn read(path: &Path) -> Result<Project> {
		let text = std::fs::read_to_string(path).with_context(|| format!("couldn't read {}", path.display()))?;
		let p: Project =
			serde_json::from_str(&text).with_context(|| format!("{} isn't a spotter project", path.display()))?;
		if p.version > VERSION {
			bail!(
				"{} was saved by a newer spotter (its version {}; this one reads up to {VERSION})",
				path.display(),
				p.version
			);
		}
		Ok(p)
	}

	/// Write it by way of a `.part` beside it, so a project is never left half written.
	pub fn write(&self, path: &Path) -> Result<()> {
		let text = serde_json::to_string_pretty(&Project { version: VERSION, ..self.clone() })?;
		let part = crate::paths::part(path);
		std::fs::write(&part, text).with_context(|| format!("couldn't write {}", path.display()))?;
		Ok(std::fs::rename(&part, path)?)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn a_project_reads_back() {
		let mut look = Look::preset("heat").unwrap();
		look.follow = true;
		let p = Project {
			source: "clip.mp4".into(),
			preset: "heat".into(),
			look: serde_json::to_value(&look).unwrap(),
			size: Some((1080, 1920)),
			model: "x365".into(),
			start: Some(2.0),
			..Default::default()
		};
		let back: Project = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
		assert_eq!(back.look(), look);
		assert_eq!(back.size, Some((1080, 1920)));
		assert_eq!(back.start, Some(2.0));
		// an old project with a setting missing still opens
		let old: Project = serde_json::from_str(r#"{"source":"a.mp4","look":{"points":9}}"#).unwrap();
		assert_eq!(old.look().points, 9.0);
	}
}
