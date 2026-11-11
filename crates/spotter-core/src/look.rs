//! Everything about how a video is drawn over, as one flat set of plain values, so a look can be
//! saved, copied and swapped whole. The defaults, the presets and the settings panel's groups are in
//! `assets/looks.json`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::OnceLock;

const LOOKS: &str = include_str!("../../../assets/looks.json");

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Look {
	pub edges: f32,
	pub motion: f32,
	pub bright: f32,
	pub dark: f32,
	pub threshold: f32,
	pub spacing: f32,
	pub points: f32,
	pub detail: f32,
	pub detect: bool,
	pub object_kinds: String,
	pub confidence: f32,
	pub objects: f32,
	pub per_object: f32,
	pub detect_every: f32,
	pub region_template: String,
	pub region_spot: String,
	pub own_numbers: bool,
	pub focus: f32,
	pub stabilize: bool,
	pub sensitivity: f32,
	pub memory: f32,
	pub subjects: f32,
	pub subject_min: f32,
	pub show_subject: bool,
	pub reach: f32,
	pub linger: f32,
	pub smooth: f32,
	pub id_base: f32,
	pub boxes: f32,
	pub size_min: f32,
	pub size_max: f32,
	pub box_style: String,
	pub line_width: f32,
	pub color: String,
	pub opacity: f32,
	pub regions: f32,
	pub region_size: f32,
	pub region_min: f32,
	pub region_style: String,
	pub color_by: String,
	pub labels: f32,
	pub template: String,
	pub words: String,
	pub font: String,
	pub label_size: f32,
	pub label_color: String,
	pub label_backing: bool,
	pub label_back: String,
	pub label_spot: String,
	pub link_mode: String,
	pub hubs: f32,
	pub link_count: f32,
	pub link_max: f32,
	pub link_width: f32,
	pub link_color: String,
	pub link_opacity: f32,
	pub trail: f32,
	pub trail_of: String,
	pub trail_width: f32,
	pub patches: f32,
	pub patch_mode: String,
	pub patch_color: String,
	pub brightness: f32,
	pub contrast: f32,
	pub saturation: f32,
	pub hue: f32,
	pub sharpen: f32,
	pub solarize: f32,
	pub solar_at: f32,
	pub posterize: f32,
	pub invert: f32,
	pub palette: String,
	pub palette_mix: f32,
	pub shadow: String,
	pub highlight: String,
	pub split: f32,
	pub scanlines: f32,
	pub grain: f32,
	pub vignette: f32,
	pub pixelate: f32,
	pub glitch: f32,
	pub hud: String,
	pub hud_spot: String,
	pub hud_size: f32,
	pub hud_color: String,
	pub speed: f32,
	pub out_max: f32,
	pub bitrate: f32,

	// the rest, off by default so the overlay looks don't change
	/// layers: what the matte is made from (boxes, movement, both)
	pub matte: String,
	/// what the world outside the things becomes
	pub world: String,
	pub world_amount: f32,
	/// what the things themselves become
	pub within: String,
	pub matte_color: String,
	pub feather: f32,
	pub grow: f32,
	pub contour: f32,
	pub contour_width: f32,
	pub contour_color: String,
	/// sonar: rings spreading out from the things (how strong, how far apart and how fast, in pixels of
	/// a 1080 picture, and their colour)
	pub rings: f32,
	pub ring_spacing: f32,
	pub ring_speed: f32,
	pub ring_color: String,
	/// pixel sorting: how bright a run must be to be sorted, and along columns (down) or rows (across)
	pub sort_level: f32,
	pub sort_dir: String,
	/// a halftone dot's size (and a tenth of a dither's), in pixels of a 1080 picture
	pub dot_size: f32,
	/// follow cam: move the frame with what's followed
	pub follow: bool,
	pub follow_zoom: f32,
	/// seconds the camera takes to settle
	pub follow_smooth: f32,
	/// echo: how many copies (or how long the exposure lasts), how often, how they fade, and what
	/// kind (copies, ghosts, exposure, extract)
	pub echo: f32,
	pub echo_every: f32,
	pub echo_fade: f32,
	pub echo_mode: String,
	/// insets: how many, along which side, how big, how close in, the caption and its colour
	pub insets: f32,
	pub inset_side: String,
	pub inset_size: f32,
	pub inset_zoom: f32,
	pub inset_template: String,
	pub inset_color: String,
	/// glyphs: where (off, things, world, all), how big, which characters, ink, ground, and whether
	/// the ink takes the picture's colours
	pub glyphs: String,
	pub glyph_size: f32,
	pub glyph_ramp: String,
	pub glyph_color: String,
	pub glyph_back: String,
	pub glyph_tint: bool,
	/// words: what the world is typed as, where nothing's labelled it (empty leaves it paper)
	pub world_word: String,
	/// words: label the scene, so the world is typed as what it is (sky, sea, tree…), and how strongly
	/// the world is typed next to the things
	pub scene: bool,
	pub world_ink: f32,
	/// reveal: the picture as it was against the look: split (as it was up to `reveal_at` across),
	/// sweep (that line going across and back over `reveal_time`), flick (cuts to it for `reveal_at` of
	/// each `reveal_time`, as regular as `reveal_jitter` allows), things or world, or off
	pub reveal: String,
	pub reveal_at: f32,
	pub reveal_time: f32,
	pub reveal_jitter: f32,
	/// mix: another look (by name, "" for none) drawn over the same clip and put together with this
	/// one, `mix_how`: split, sweep, flick, things or world as reveal does, blend (`mix_at` of the
	/// other), lighter (the lighter of the two), or over and under (one's marks on the other)
	pub mix: String,
	pub mix_how: String,
	pub mix_at: f32,
	pub mix_time: f32,
	pub mix_jitter: f32,
	/// the look mixed in, changed: its settings that differ from it as it comes (name → value)
	#[serde(default)]
	pub mix_set: Map<String, Value>,
	/// glitches (glitch.rs): how far the picture's bent (read with the wrong row width, bits lost)
	/// and crushed (a JPEG's quality, 1 the worst), for the world and things modes of those names
	pub bend: f32,
	pub crush: f32,
	/// datamosh: off, on cuts or always (drawn), or real cuts or real beats (the clip re-encoded and
	/// its frames rearranged, realmosh.rs); on all the picture, the world or the things (drawn only);
	/// how many seconds a shot's held after a cut, or (always) a block lasts before it's drawn again
	pub mosh: String,
	pub mosh_where: String,
	pub mosh_hold: f32,
	/// a worn tape, 0-1; trichrome's seconds between its colours (0 is off); stutter's chance (each
	/// beat, or about how often a second without the music's) and how many frames it loops
	pub tear: f32,
	pub trichrome: f32,
	pub stutter: f32,
	pub stutter_len: f32,
	/// kaleidoscope: how many mirrored wedges (0 is off), turning this many times a minute, round
	/// the biggest thing on show (or the middle)
	pub kaleido: f32,
	pub kaleido_spin: f32,
	/// on the beat: flicks cut, copies are taken and sonar rings pulse on the music's beats (where the
	/// clip has sound), every `beat_every`th one
	pub beat: bool,
	pub beat_every: f32,
	/// foresight: how many seconds ahead to draw where each thing is going (0 is off), and in what colour
	pub foresight: f32,
	pub foresight_color: String,
	/// slowed down (speed under 1), make the frames between with RIFE rather than showing each again
	pub smooth_slow: bool,
	/// depth: what it does (off, fog, slices, sweep), how much, in what colour, how many slices, and
	/// how long a sweep takes
	pub depth: String,
	pub depth_amount: f32,
	pub depth_color: String,
	pub depth_lines: f32,
	pub depth_speed: f32,
	/// look ahead: work out what's in the whole clip before drawing any of it (tape.rs), and how long
	/// its boxes are smoothed over, both ways in time
	pub ahead: bool,
	pub ahead_smooth: f32,
}

/// One control in the settings panel.
#[derive(Clone, Debug, Deserialize)]
pub struct Knob {
	/// the setting it changes, as written in the look (camelCase)
	pub k: String,
	pub label: String,
	pub hint: Option<String>,
	pub min: Option<f32>,
	pub max: Option<f32>,
	pub step: Option<f32>,
	pub options: Option<Vec<String>>,
	/// "color", "text", "lines" or "toggle"
	pub kind: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct KnobGroup {
	pub name: String,
	/// a line under the heading, when the names don't say enough
	pub note: Option<String>,
	pub knobs: Vec<Knob>,
}

#[derive(Deserialize)]
struct Looks {
	default: Map<String, Value>,
	presets: Map<String, Value>,
	knobs: Vec<KnobGroup>,
	finding: Vec<String>,
}

fn looks() -> &'static Looks {
	static L: OnceLock<Looks> = OnceLock::new();
	L.get_or_init(|| serde_json::from_str(LOOKS).expect("assets/looks.json is valid"))
}

fn from_map(m: Map<String, Value>) -> Look {
	serde_json::from_value(Value::Object(m)).expect("every setting has a value of the right kind")
}

impl Default for Look {
	fn default() -> Self {
		from_map(looks().default.clone())
	}
}

impl Look {
	/// The names of the presets, in their order in looks.json.
	pub fn presets() -> impl Iterator<Item = &'static str> {
		looks().presets.keys().map(String::as_str)
	}

	/// A look by name: one of the presets, or one of your own (`mine`).
	pub fn named(name: &str) -> Option<Look> {
		Look::preset(name).or_else(|| mine().into_iter().find(|(n, _)| n == name).map(|(_, l)| l))
	}

	/// A preset laid over the defaults, or None if there's no such preset.
	pub fn preset(name: &str) -> Option<Look> {
		let p = looks().presets.get(name)?.as_object()?;
		let mut m = looks().default.clone();
		for (k, v) in p {
			m.insert(k.clone(), v.clone());
		}
		Some(from_map(m))
	}

	/// A saved look brought up to date: settings added since keep their defaults, ones removed are
	/// dropped, and anything of the wrong kind is ignored.
	pub fn revive(saved: &Value) -> Look {
		let mut m = looks().default.clone();
		if let Some(s) = saved.as_object() {
			for (k, v) in s {
				let same = m.get(k).is_some_and(|d| {
					std::mem::discriminant(d) == std::mem::discriminant(v) || (d.is_number() && v.is_number())
				});
				if same {
					m.insert(k.clone(), v.clone());
				}
			}
		}
		from_map(m)
	}

	/// The groups of controls for a settings panel.
	pub fn knobs() -> &'static [KnobGroup] {
		&looks().knobs
	}

	/// One setting by its camelCase name, as JSON.
	pub fn get(&self, key: &str) -> Option<Value> {
		serde_json::to_value(self).ok()?.get(key).cloned()
	}

	/// Change one setting by its camelCase name. Returns false if there's no such setting or the
	/// value is the wrong kind.
	pub fn set(&mut self, key: &str, value: Value) -> bool {
		let Ok(Value::Object(mut m)) = serde_json::to_value(&*self) else { return false };
		match m.get(key) {
			Some(old) if std::mem::discriminant(old) == std::mem::discriminant(&value) => {}
			_ => return false,
		}
		m.insert(key.to_string(), value);
		match serde_json::from_value(Value::Object(m)) {
			Ok(l) => {
				*self = l;
				true
			}
			Err(_) => false,
		}
	}

	/// The look this one mixes with, as it's drawn: its own settings, but finding things as this one
	/// does (so both have the same things), and mixing with nothing itself. Reads your own looks from
	/// disk, so it's for when the look changes, not every frame.
	pub fn mixed(&self) -> Option<Look> {
		let name = self.mix.trim();
		if name.is_empty() {
			return None;
		}
		Some(self.mixed_with(Look::named(name)?))
	}

	/// `base` (the look mixed in, as it comes) as it's drawn mixed into this one: changed as
	/// `mix_set` says, finding things as this one does, and mixing with nothing itself.
	pub fn mixed_with(&self, base: Look) -> Look {
		let mut m = base;
		for (k, v) in &self.mix_set {
			m.set(k, v.clone());
		}
		m.mix.clear();
		m.mix_set.clear();
		let Ok(Value::Object(mine)) = serde_json::to_value(self) else { return m };
		for k in &looks().finding {
			if let Some(v) = mine.get(k) {
				m.set(k, v.clone());
			}
		}
		m.ahead = self.ahead;
		m.ahead_smooth = self.ahead_smooth;
		m
	}

	/// The settings a look mixed in takes from the one it's mixed into (what's found, looking ahead,
	/// the camera, speed, the export's), which changing on it would do nothing.
	pub fn shared_keys() -> Vec<String> {
		let mut k = looks().finding.clone();
		k.extend(
			[
				"ahead",
				"aheadSmooth",
				"mix",
				"mixHow",
				"mixAt",
				"mixTime",
				"mixJitter",
				"mixSet",
				"follow",
				"followZoom",
				"followSmooth",
				"speed",
				"smoothSlow",
				"outMax",
				"bitrate",
			]
			.map(String::from),
		);
		k
	}

	/// Whether this look, or the one it mixes with, wants something (a model, say).
	pub fn either(&self, wants: impl Fn(&Look) -> bool) -> bool {
		wants(self) || self.mixed().is_some_and(|m| wants(&m))
	}

	/// The real datamosh the look asks for (realmosh.rs), if any: on beats, or on cuts.
	pub fn real_mosh(&self) -> Option<bool> {
		match self.mosh.as_str() {
			"real cuts" => Some(false),
			"real beats" => Some(true),
			_ => None,
		}
	}

	/// The settings that change what's found, so a paused picture only re-tracks when one moves.
	pub fn finding_key(&self) -> String {
		let Ok(Value::Object(m)) = serde_json::to_value(self) else { return String::new() };
		looks().finding.iter().map(|k| m.get(k).map(Value::to_string).unwrap_or_default()).collect::<Vec<_>>().join(",")
	}
}

/// Where your own looks are kept: a folder of JSON files (`%LOCALAPPDATA%\spotter\looks` on
/// Windows), one look to a file, named by its file's name.
pub fn mine_dir() -> std::path::PathBuf {
	crate::paths::home().join("looks")
}

/// Your own looks, by name, in name order. A file that won't read is left out.
pub fn mine() -> Vec<(String, Look)> {
	let Ok(dir) = std::fs::read_dir(mine_dir()) else { return Vec::new() };
	let mut out: Vec<(String, Look)> = dir
		.flatten()
		.filter(|e| e.path().extension().is_some_and(|x| x == "json"))
		.filter_map(|e| {
			let name = e.path().file_stem()?.to_string_lossy().into_owned();
			let v: Value = serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok()?;
			Some((name, Look::revive(&v)))
		})
		.collect();
	out.sort_by(|a, b| a.0.cmp(&b.0));
	out
}

/// A name as it can be a file's: lowercase letters, digits, spaces and dashes, at most 32 of them.
pub fn clean_name(name: &str) -> String {
	let s: String = name
		.trim()
		.to_lowercase()
		.chars()
		.filter(|c| c.is_ascii_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
		.take(32)
		.collect();
	s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Keep `look` as one of your own under `name` (made a file's name, and not a preset's: "heat" would
/// become "heat 2"). Says the name it's kept under.
pub fn save_mine(name: &str, look: &Look) -> anyhow::Result<String> {
	let mut name = clean_name(name);
	if name.is_empty() {
		anyhow::bail!("a look needs a name");
	}
	if Look::preset(&name).is_some() {
		let base = name.clone();
		let mut n = 2;
		while Look::preset(&name).is_some() {
			name = format!("{base} {n}");
			n += 1;
		}
	}
	let dir = mine_dir();
	std::fs::create_dir_all(&dir)?;
	let file = dir.join(format!("{name}.json"));
	let part = crate::paths::part(&file);
	std::fs::write(&part, serde_json::to_string_pretty(look)?)?;
	std::fs::rename(&part, &file)?;
	Ok(name)
}

/// Forget one of your own looks.
pub fn delete_mine(name: &str) -> anyhow::Result<()> {
	Ok(std::fs::remove_file(mine_dir().join(format!("{}.json", clean_name(name))))?)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn names_become_file_names() {
		assert_eq!(clean_name("  My   Look!/.. "), "my look");
		assert_eq!(clean_name("night-ops_2"), "night-ops_2");
	}

	#[test]
	fn every_preset_reads() {
		for name in Look::presets() {
			let l = Look::preset(name).unwrap();
			assert!(l.points > 0.0, "{name}");
		}
		assert_eq!(Look::preset("cctv").unwrap().font, "mono");
		assert_eq!(Look::default().id_base, 37000.0);
	}

	#[test]
	fn revive_keeps_what_fits() {
		let l = Look::revive(&serde_json::json!({ "points": 9, "font": 3, "gone": true }));
		assert_eq!(l.points, 9.0);
		assert_eq!(l.font, Look::default().font);
	}
}
