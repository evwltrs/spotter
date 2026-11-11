// a window of its own, without a terminal beside it (messages still go wherever the one that
// started it sends them: the watchdog's log, say)
#![cfg_attr(windows, windows_subsystem = "windows")]

//! spotter's window: the looks down the left, the picture in the middle, one inspector on
//! the right for the three jobs (the look, what's followed, exporting), time along the bottom and a
//! status line under it, in the site's colours. Without a video it waits for one to be opened,
//! pasted or dropped.
//!
//!   spotter-app [video]        a file, a web address, files:<path>, or random,
//!                              or a project saved from the window (.spotter)

mod export;
mod follow;
mod keys;
mod log;
mod rail;
mod safety;
mod settings;
mod sound;
mod theme;
mod timeline;
mod widgets;
mod worker;

use export::*;
use follow::*;
use keys::*;
use rail::*;
use rune::tok::{Family, TextStyle};
use rune::{
	AppSpec, Color, Cursor, Cx, Drag, El, FileDialogOptions, Key, KeyEvent, NativeSurface, Phase, ScrollSpec,
	SurfaceFit, TextEdit, Tx, col, el, row, spacer, text,
};
use rune_kit::{Kit, Radius, TextArea, TextInput};
use serde_json::Value;
use settings::*;
use spotter_core::Look;
use spotter_core::edit::{Edit, Keep, Pick, RangeMode, Zone, ZoneMode, to_source};
use spotter_core::export::{Export, gpu};
use spotter_core::look::Knob;
use spotter_core::project::{EXTENSION, Project};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use theme::*;
use timeline::*;
use widgets::*;
use worker::{Cmd, Status, Strip};

const FONTS: &[&[u8]] = &[
	include_bytes!("../../../assets/fonts/DMMono-Regular.ttf"),
	include_bytes!("../../../assets/fonts/DMMono-Medium.ttf"),
];
const VIDEO: &[&str] = &["mp4", "mov", "mkv", "webm", "m4v", "avi", "ts"];
/// songs, to put under the video
const AUDIO: &[&str] = &["mp3", "wav", "m4a", "aac", "flac", "ogg", "opus"];
/// the shapes to export at: as shot, or cut to a shape and written at the size it's posted at
const ASPECTS: [(&str, Option<(u32, u32)>); 5] = [
	("as shot", None),
	("reel 9:16", Some((1080, 1920))),
	("post 4:5", Some((1080, 1350))),
	("square", Some((1080, 1080))),
	("wide 16:9", Some((1920, 1080))),
];

fn shape_of(i: usize) -> Option<f64> {
	ASPECTS[i].1.map(|(w, h)| w as f64 / h as f64)
}
const MODELS: [&str; 4] = ["s", "m", "l", "x"];
/// how far a download has got, out of 1000, and where it went (or why not)
type Progress = Arc<std::sync::atomic::AtomicU32>;
type Fetched = std::sync::mpsc::Receiver<Result<PathBuf, String>>;
/// how long something that went wrong stays on the status line (esc clears it sooner)
const ERROR_FOR: Duration = Duration::from_secs(8);
/// the kinds offered before the detector has named anything, as each vocabulary names them
const KINDS: [&str; 10] =
	["person", "car", "truck", "bus", "motorbike", "bicycle", "aeroplane", "boat", "train", "bird"];
const KINDS_365: [&str; 14] = [
	"person",
	"car",
	"suv",
	"van",
	"truck",
	"bus",
	"motorcycle",
	"bicycle",
	"airplane",
	"helicopter",
	"boat",
	"ship",
	"train",
	"wild bird",
];

/// The inspector's three jobs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
	Look,
	Follow,
	Export,
}

/// What a press on the picture does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tool {
	None,
	/// follow the thing pressed on
	Pick,
	/// drag out a part of the picture to leave alone
	Ignore,
	/// drag out the only part of the picture to follow things in
	Only,
}

// the site's Gruvbox

struct App {
	surface: NativeSurface,
	cmd: Option<Sender<Cmd>>,
	shared: Arc<Mutex<Status>>,
	st: Status,
	look: Look,
	preset: String,
	/// the setting groups that are open
	open: Vec<String>,
	/// the text being typed into each text setting, and the source box ("" key)
	edits: HashMap<String, TextEdit>,
	aspect: usize,
	model: usize,
	/// Objects365's 365 kinds of thing rather than COCO's 80
	vocab: bool,
	trim_in: Option<f64>,
	trim_out: Option<f64>,
	/// make small pictures bigger with Real-ESRGAN when exporting
	upscale: bool,
	/// export with the quick models the window uses, rather than the best ones
	quick: bool,
	/// the preview's sound is off
	muted: bool,
	/// how loud the preview plays (1 as it is, up to 1.5), kept for next time, and as last sent
	volume: f32,
	sent_volume: Option<f32>,
	/// where the playhead is being dragged to, before the video gets there
	scrub: Option<f64>,
	/// what's followed, where and when
	clip_edit: Edit,
	tool: Tool,
	/// where a zone being dragged out started, 0-1 across the source
	drag_from: Option<(f32, f32)>,
	/// a stretch being marked: where it starts, and whether it's a cut or bare
	marking: Option<(f64, RangeMode)>,
	/// for testing the window's export without its save dialog: SPOTTER_TEST_EXPORT names a file
	/// to export to as soon as a video is open
	test_export: Option<PathBuf>,
	/// something the window itself has to say (a project that wouldn't open or save)
	note: Option<String>,
	/// which of the inspector's jobs is showing
	tab: Tab,
	/// the look tab's search
	search: TextEdit,
	/// groups (and the follow tab's kinds) opened past "+ n more"
	more: Vec<String>,
	/// the keys, listed in the inspector
	show_keys: bool,
	/// what's downloading or being fixed the first time, for the status line
	getting: Option<spotter_core::model::Getting>,
	/// whether ffmpeg is missing, once that's known; and spotter's own copy on its way, if asked for
	no_ffmpeg: Option<std::sync::mpsc::Receiver<bool>>,
	needs_ffmpeg: bool,
	ffmpeg_coming: Option<std::sync::mpsc::Receiver<Result<(), String>>>,
	/// the video last asked for, to open again once there's an ffmpeg to read it
	asked: String,
	/// when what's open was last kept in last.spotter, and as what
	autosaved: (Instant, String),
	/// when a folder was last shown, so pressing again at once doesn't open another
	shown_folder: Option<Instant>,
	/// the last things that went wrong, the newest first, for ? to list; and when the newest came,
	/// shown until a while after that or until esc
	errors: Vec<String>,
	error_since: Option<Instant>,
	/// the looks and the inspector showing (f hides them)
	panels: bool,
	/// showing the picture as it is, nothing drawn
	compare: bool,
	/// the address box open in the top bar
	addressing: bool,
	/// what's in the clip, as the worker last said (for the timeline), and where it says it
	strip: Option<Arc<Strip>>,
	strip_shared: Arc<Mutex<Option<Arc<Strip>>>>,
	/// what's under the pointer on the timeline, for the status line
	hovering: Option<String>,
	/// the thing's bar under the pointer on the timeline, for delete to take it out
	hover_bar: Option<worker::Span>,
	/// the stretch selected on the timeline (delete removes it), one being dragged out, and the kind
	/// a drag makes
	selected_range: Option<usize>,
	drafting: Option<(f64, f64)>,
	range_mode: RangeMode,
	/// a stretch's edge being dragged: which stretch, and whether it's the start
	grab: Option<(usize, bool)>,
	/// other shapes to export at beside the chosen one
	also: Vec<usize>,
	/// the in point as the worker last heard it (None: not told yet)
	sent_in: Option<Option<f64>>,
	/// how long the song under the video is, once asked (its path, seconds)
	music_len: Option<(String, f64)>,
	/// a project's song's length, being found (an ffprobe, so not on the window's thread)
	len_rx: Option<(String, std::sync::mpsc::Receiver<f64>)>,
	/// the box for a video's path or address, while nothing is open
	link: TextEdit,
	/// the link box for a song, a song being fetched from a link (the link, the answer, how far
	/// it's got out of 1000), and the song's shape (its path, how loud along it), being worked out
	song_link: TextEdit,
	fetching: Option<(String, Fetched, Progress)>,
	song_shape: Option<(String, Arc<Vec<f32>>)>,
	shape_rx: Option<(String, std::sync::mpsc::Receiver<Vec<f32>>)>,
	/// dragging the part of the song used: how far into it the pointer took hold
	song_grab: Option<f64>,
	/// where vary's chances go next
	vary_seed: u64,
	/// the look tab is showing the look mixed in, not this one
	mix_tab: bool,
	/// your own looks, kept in %LOCALAPPDATA%\\spotter\\looks
	mine: Vec<(String, Look)>,
	/// the name being typed for the look, to keep it as one of yours
	naming: Option<TextEdit>,
	/// the looks and edits to undo back to, and to redo, and when the last was kept
	history: Vec<(String, Look, Edit)>,
	future: Vec<(String, Look, Edit)>,
	remembered: Option<Instant>,
}

impl App {
	fn send(&self, c: Cmd) {
		if let Some(tx) = &self.cmd {
			let _ = tx.send(c);
		}
	}

	fn set_look(&mut self, l: Look) {
		if l != self.look {
			self.remember();
		}
		self.look = l;
		self.send(Cmd::Look(self.look.clone()));
	}

	/// Keep the look and the edit as they are now, to undo back to. Changes in quick succession (a
	/// slider dragged, a colour typed) count as one.
	fn remember(&mut self) {
		let now = Instant::now();
		if self.remembered.is_some_and(|t| now.duration_since(t).as_millis() < 400) {
			self.remembered = Some(now);
			return;
		}
		self.remembered = Some(now);
		self.history.push((self.preset.clone(), self.look.clone(), self.clip_edit.clone()));
		if self.history.len() > 100 {
			self.history.remove(0);
		}
		self.future.clear();
	}

	/// Back to how the look and the edit were before the last change (or on again, `redo`).
	fn undo(&mut self, redo: bool) {
		let (from, to) =
			if redo { (&mut self.future, &mut self.history) } else { (&mut self.history, &mut self.future) };
		let Some((preset, look, edit)) = from.pop() else {
			self.note = Some(if redo { "nothing to redo".into() } else { "nothing to undo".into() });
			return;
		};
		to.push((self.preset.clone(), self.look.clone(), self.clip_edit.clone()));
		self.remembered = None;
		self.note = Some(if redo { "redone".into() } else { "undone".into() });
		self.preset = preset;
		self.edits.retain(|k, _| k.is_empty());
		self.look = look;
		self.send(Cmd::Look(self.look.clone()));
		self.clip_edit = edit;
		self.send(Cmd::Edit(self.clip_edit.clone()));
	}

	/// A look by name: one of yours, or a preset.
	fn base_look(&self, name: &str) -> Option<Look> {
		Look::preset(name).or_else(|| self.mine.iter().find(|(n, _)| n == name).map(|(_, l)| l.clone()))
	}

	/// Keep the look as it is now as one of yours, and switch to it.
	fn save_mine(&mut self, name: &str) {
		let mixed = self.editing_mix();
		match spotter_core::look::save_mine(name, &self.shown_look()) {
			Ok(kept) if mixed => {
				// the look mixed in, as changed, is yours now: that's what's mixed in
				self.mine = spotter_core::look::mine();
				let mut l = self.look.clone();
				l.mix = kept.clone();
				l.mix_set.clear();
				self.set_look(l);
				self.naming = None;
				self.note = Some(format!("kept as \"{kept}\" and mixed in as that"));
			}
			Ok(kept) => {
				self.mine = spotter_core::look::mine();
				self.preset = kept.clone();
				self.naming = None;
				self.note = Some(format!("kept as \"{kept}\", at the top of the looks"));
			}
			Err(e) => self.note = Some(format!("couldn't keep it: {e:#}")),
		}
	}

	/// A look someone saved (a JSON file from their looks folder), kept as one of yours under its
	/// file's name, and used.
	fn add_look_file(&mut self, p: &std::path::Path) {
		let read = std::fs::read_to_string(p)
			.map_err(anyhow::Error::from)
			.and_then(|t| Ok(serde_json::from_str::<serde_json::Value>(&t)?))
			.and_then(|v| {
				let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
				spotter_core::look::save_mine(&name, &Look::revive(&v))
			});
		match read {
			Ok(kept) => {
				self.mine = spotter_core::look::mine();
				self.pick_preset(&kept);
				self.note = Some(format!("\"{kept}\" is one of your looks now"));
			}
			Err(e) => self.note = Some(format!("{} isn't a look spotter can read: {e:#}", p.display())),
		}
	}

	/// Forget one of your looks; the picture keeps it until another is chosen.
	fn delete_mine(&mut self, name: &str) {
		match spotter_core::look::delete_mine(name) {
			Ok(()) => {
				self.mine = spotter_core::look::mine();
				self.note =
					Some(format!("\"{name}\" is gone from your looks (ctrl+z doesn't bring it back; save it again)"));
			}
			Err(e) => self.note = Some(format!("couldn't delete it: {e:#}")),
		}
	}

	/// The look as the rail calls it: picked, or with shift held, mixed in.
	fn press_look(&mut self, name: &str, tx: &Tx) {
		if tx.mods().shift { self.mix_with(name) } else { self.pick_preset(name) }
	}

	/// Mix another look in with this one, or take it out if it's the one mixed in already.
	fn mix_with(&mut self, name: &str) {
		if name == self.preset {
			self.note = Some(format!("{name} is the look already: right-click another to mix it in"));
			return;
		}
		let mut l = self.look.clone();
		let gone = l.mix.trim() == name;
		l.set("mix", Value::String(if gone { String::new() } else { name.to_string() }));
		// changes made to the last one mixed in were its own
		l.mix_set.clear();
		self.set_look(l);
		if gone {
			self.show_mix(false);
		}
		self.note = Some(if gone {
			format!("{name} taken out")
		} else {
			format!("{name} mixed in, on the things to start with: how it's put together is under mix in the look tab")
		});
	}

	fn pick_preset(&mut self, name: &str) {
		if let Some(mut l) = self.base_look(name) {
			// a look mixed in stays when another is picked (not when the look's put back, or the look
			// picked is the one mixed in)
			let mixing = self.look.mix.trim().to_string();
			if name != self.preset && !mixing.is_empty() && mixing != name {
				for k in ["mix", "mixHow", "mixAt", "mixTime", "mixJitter", "mixSet"] {
					if let Some(v) = self.look.get(k) {
						l.set(k, v);
					}
				}
			}
			// a new look is always its own step to undo
			self.remembered = None;
			self.remember();
			self.remembered = None;
			self.preset = name.to_string();
			self.edits.retain(|k, _| k.is_empty());
			self.look = l;
			self.send(Cmd::Look(self.look.clone()));
		}
	}

	/// Whether the look tab is on the look mixed in (there being one).
	fn editing_mix(&self) -> bool {
		self.mix_tab && !self.look.mix.trim().is_empty()
	}

	/// The look the look tab shows: this one, or the one mixed in as it's drawn.
	fn shown_look(&self) -> Look {
		if self.editing_mix()
			&& let Some(base) = self.base_look(self.look.mix.trim())
		{
			return self.look.mixed_with(base);
		}
		self.look.clone()
	}

	/// The look mixed in, as it comes (what its changes are kept against).
	fn mix_base(&self) -> Option<Look> {
		self.base_look(self.look.mix.trim())
	}

	/// Put the look the look tab shows: this one, or (on the look mixed in) kept as what differs
	/// from it as it comes.
	fn set_shown(&mut self, shown: Look) {
		if !self.editing_mix() {
			self.set_look(shown);
			return;
		}
		let Some(base) = self.mix_base() else { return };
		let (Ok(Value::Object(now)), Ok(Value::Object(was))) =
			(serde_json::to_value(&shown), serde_json::to_value(&base))
		else {
			return;
		};
		let shared = Look::shared_keys();
		let changes: serde_json::Map<String, Value> =
			now.into_iter().filter(|(k, v)| !shared.contains(k) && was.get(k) != Some(v)).collect();
		let mut l = self.look.clone();
		l.mix_set = changes;
		self.set_look(l);
	}

	/// Show the look mixed in in the look tab, or this one.
	fn show_mix(&mut self, on: bool) {
		if self.mix_tab != on {
			self.mix_tab = on;
			// the text boxes held this look's words
			self.edits.clear();
			self.naming = None;
		}
	}

	fn set(&mut self, key: &str, v: Value) {
		if self.editing_mix() {
			let mut shown = self.shown_look();
			if shown.set(key, v) {
				self.set_shown(shown);
			}
			return;
		}
		let mut l = self.look.clone();
		if l.set(key, v) {
			// each way of mixing starts from where it's at its best: half and half for split, blend
			// and flick, all the way for the rest
			if key == "mixHow" {
				l.mix_at = if matches!(l.mix_how.as_str(), "split" | "blend" | "flick") { 0.5 } else { 1.0 };
			}
			self.set_look(l);
		}
	}

	fn edit(&mut self, key: &str) -> &mut TextEdit {
		let current = match self.shown_look().get(key) {
			Some(Value::String(s)) => s,
			_ => String::new(),
		};
		self.edits.entry(key.to_string()).or_insert_with(|| TextEdit::new(current))
	}

	/// Keep what's open in last.spotter, every few seconds while it changes.
	fn autosave(&mut self) {
		if self.st.input.is_empty() || self.autosaved.0.elapsed() < Duration::from_secs(5) {
			return;
		}
		let p = self.project();
		let text = serde_json::to_string(&p).unwrap_or_default();
		if text != self.autosaved.1 && p.write(&last_project()).is_ok() {
			self.autosaved.1 = text;
		}
		self.autosaved.0 = Instant::now();
	}

	/// The folder a file's in, with the file selected.
	fn show_in_folder(&mut self, file: &std::path::Path) {
		if self.shown_folder.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
			return;
		}
		self.shown_folder = Some(Instant::now());
		let shown = if cfg!(windows) {
			let mut select = std::ffi::OsString::from("/select,");
			select.push(file);
			std::process::Command::new("explorer").arg(select).spawn()
		} else if cfg!(target_os = "macos") {
			std::process::Command::new("open").arg("-R").arg(file).spawn()
		} else {
			std::process::Command::new("xdg-open").arg(file.parent().unwrap_or(file)).spawn()
		};
		if let Err(e) = shown {
			self.note = Some(format!("couldn't show the folder: {e}"));
		}
	}

	/// spotter's own ffmpeg, downloaded on a thread of its own (the status line follows it).
	fn get_ffmpeg(&mut self) {
		if self.ffmpeg_coming.is_some() {
			return;
		}
		let (tx, rx) = std::sync::mpsc::channel();
		std::thread::spawn(move || tx.send(spotter_core::video::fetch_ffmpeg().map_err(|e| e.to_string())));
		self.ffmpeg_coming = Some(rx);
	}

	fn open_source(&mut self, arg: String) {
		self.asked = arg.clone();
		self.trim_in = None;
		self.trim_out = None;
		self.marking = None;
		// an edit belongs to its clip (the kinds looked for, and the song, carry on to the next)
		self.set_edit(Edit {
			kinds: self.clip_edit.kinds.clone(),
			music: self.clip_edit.music.clone(),
			..Edit::default()
		});
		// what's undone belongs to the last clip
		self.history.clear();
		self.future.clear();
		self.send(Cmd::Open(arg));
	}

	fn set_edit(&mut self, e: Edit) {
		if e != self.clip_edit {
			self.remember();
		}
		self.clip_edit = e;
		self.send(Cmd::Edit(self.clip_edit.clone()));
	}

	fn change_edit(&mut self, f: impl FnOnce(&mut Edit)) {
		let mut e = self.clip_edit.clone();
		f(&mut e);
		self.set_edit(e);
	}

	/// The detector as spotter names it: "x", or "x365" for 365 kinds.
	fn detector(&self) -> String {
		format!("{}{}", MODELS[self.model], if self.vocab { "365" } else { "" })
	}

	fn set_detector(&mut self, model: usize, vocab: bool) {
		if (model, vocab) == (self.model, self.vocab) {
			return;
		}
		// the names of kinds differ between the two
		if vocab != self.vocab {
			self.change_edit(|e| e.kinds.clear());
		}
		self.model = model;
		self.vocab = vocab;
		self.send(Cmd::Detector(self.detector()));
	}

	fn set_tool(&mut self, t: Tool) {
		self.tool = if self.tool == t { Tool::None } else { t };
		self.drag_from = None;
		self.send(Cmd::Draft(None));
		self.send(Cmd::Guides(self.tool != Tool::None));
	}

	/// A place on the video element (its own points, and its size) as 0-1 across the source, if it's
	/// on the picture.
	fn on_picture(&self, x: f32, y: f32, w: f32, h: f32) -> Option<(f32, f32)> {
		let (fw, fh) = (self.st.w as f32, self.st.h as f32);
		if fw <= 0.0 || fh <= 0.0 || w <= 0.0 || h <= 0.0 {
			return None;
		}
		let sc = (w / fw).min(h / fh);
		let (ox, oy) = ((w - fw * sc) / 2.0, (h - fh * sc) / 2.0);
		let (u, v) = ((x - ox) / (fw * sc), (y - oy) / (fh * sc));
		if !(-0.02..=1.02).contains(&u) || !(-0.02..=1.02).contains(&v) {
			return None;
		}
		Some(to_source(self.st.view, u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
	}

	fn press_picture(&mut self, d: &Drag) {
		let Some((x, y)) = self.on_picture(d.x, d.y, d.w, d.h) else { return };
		match self.tool {
			Tool::None => {}
			Tool::Pick => {
				if d.phase == Phase::Up && !d.cancelled() {
					let t = self.position();
					self.change_edit(|e| {
						e.picks.push(Pick { t, x, y });
						e.keep = Keep::Picked;
					});
				}
			}
			Tool::Ignore | Tool::Only => {
				let mode = if self.tool == Tool::Only { ZoneMode::Only } else { ZoneMode::Ignore };
				match d.phase {
					Phase::Down => self.drag_from = Some((x, y)),
					Phase::Move | Phase::Up => {
						let Some((x0, y0)) = self.drag_from else { return };
						let z = Zone { x: x0.min(x), y: y0.min(y), w: (x - x0).abs(), h: (y - y0).abs(), mode };
						if d.phase == Phase::Move {
							self.send(Cmd::Draft(Some(z)));
						} else {
							self.drag_from = None;
							self.send(Cmd::Draft(None));
							if z.w > 0.01 && z.h > 0.01 && !d.cancelled() {
								self.change_edit(|e| e.zones.push(z));
							}
						}
					}
				}
			}
		}
	}

	fn position(&self) -> f64 {
		self.scrub.unwrap_or(self.st.t)
	}

	/// Everything about what's open, to save.
	fn project(&self) -> Project {
		Project {
			version: spotter_core::project::VERSION,
			source: self.st.input.clone(),
			name: self.st.name.clone(),
			preset: self.preset.clone(),
			look: serde_json::to_value(&self.look).unwrap_or_default(),
			edit: self.clip_edit.clone(),
			size: ASPECTS[self.aspect].1,
			upscale: self.upscale,
			model: self.detector(),
			start: self.trim_in,
			end: self.trim_out,
		}
	}

	/// Open a saved project: its video, look, edit and export.
	fn open_project(&mut self, path: &std::path::Path) {
		let p = match Project::read(path) {
			Ok(p) => p,
			Err(e) => {
				self.note = Some(format!("{e:#}"));
				return;
			}
		};
		self.note = None;
		self.open_source(p.source.clone());
		self.preset = p.preset.clone();
		self.edits.retain(|k, _| k.is_empty());
		self.set_look(p.look());
		self.set_edit(p.edit.clone());
		self.aspect = ASPECTS.iter().position(|a| a.1 == p.size).unwrap_or(0);
		self.send(Cmd::Aspect(shape_of(self.aspect)));
		self.upscale = p.upscale;
		if let Some(i) = MODELS.iter().position(|m| *m == p.model.trim_end_matches("365")) {
			self.set_detector(i, p.model.ends_with("365"));
		}
		self.trim_in = p.start;
		self.trim_out = p.end;
	}

	fn seek(&mut self, t: f64) {
		let t = t.clamp(0.0, (self.st.duration - 0.02).max(0.0));
		self.scrub = Some(t);
		self.send(Cmd::Seek(t));
	}

	/// Start marking a stretch of time at the playhead, or end the one being marked there.
	fn mark(&mut self, mode: RangeMode) {
		match self.marking {
			Some((from, m)) if m == mode => {
				let to = self.position();
				self.marking = None;
				if (to - from).abs() > 0.05 {
					self.change_edit(|e| e.add_range(from, to, mode));
				}
			}
			_ => self.marking = Some((self.position(), mode)),
		}
	}

	/// Where the footage cuts, as the look ahead has found so far (seconds).
	fn strip_cuts(&self) -> Vec<f64> {
		self.strip.as_ref().map(|s| s.cuts.clone()).unwrap_or_default()
	}

	fn ask_open(&mut self, tx: &mut Tx) {
		let o = FileDialogOptions { title: "Open a video".into(), ..Default::default() }.filter("Video", VIDEO);
		tx.open_file(o, |s: &mut App, r, _| {
			if let Ok(Some(p)) = r {
				s.open_source(p.to_string_lossy().into_owned());
			}
		});
	}

	/// Choose a song to put under the video.
	fn ask_music(&mut self, tx: &mut Tx) {
		let o = FileDialogOptions { title: "Music for the video".into(), ..Default::default() }.filter("Music", AUDIO);
		tx.open_file(o, |s: &mut App, r, _| {
			if let Ok(Some(p)) = r {
				s.add_music(&p);
			}
		});
	}

	/// Get the song at the link in the box (SoundCloud, YouTube…) and put it under the video; a file's
	/// path goes straight under.
	fn fetch_song(&mut self) {
		let url = self.song_link.text.trim().to_string();
		if url.is_empty() {
			return;
		}
		if !spotter_core::songs::is_link(&url) {
			let p = PathBuf::from(url.trim_matches('"'));
			if p.exists() {
				self.song_link = TextEdit::new("");
				self.add_music(&p);
			} else {
				self.note = Some(format!("{url} isn't a link or a file"));
			}
			return;
		}
		let (tx, rx) = std::sync::mpsc::channel();
		let progress = Arc::new(std::sync::atomic::AtomicU32::new(0));
		let (u, p) = (url.clone(), progress.clone());
		std::thread::Builder::new()
			.name("song".into())
			.spawn(move || {
				let _ = tx.send(spotter_core::songs::fetch(&u, p).map_err(|e| e.to_string()));
			})
			.expect("a thread to get the song");
		self.fetching = Some((url, rx, progress));
		self.song_link = TextEdit::new("");
		self.note = Some("getting the song (yt-dlp, only its sound)…".into());
	}

	/// Pick the part of the song that plays: pressed inside it, it's dragged along; pressed elsewhere,
	/// it centres there. `len` is the song's length, `out` the video's.
	fn drag_song(&mut self, d: &Drag, len: f64, out: f64) {
		let Some(m) = self.clip_edit.music.as_ref() else { return };
		let t = (d.x / d.w.max(1.0)).clamp(0.0, 1.0) as f64 * len;
		let room = (len - out).max(0.0);
		let grab = match d.phase {
			Phase::Down => {
				let g = if t >= m.from && t <= m.from + out { t - m.from } else { out.min(len) / 2.0 };
				self.song_grab = Some(g);
				g
			}
			Phase::Move => match self.song_grab {
				Some(g) => g,
				None => return,
			},
			Phase::Up => {
				self.song_grab = None;
				return;
			}
		};
		let from = ((t - grab).clamp(0.0, room) * 10.0).round() / 10.0;
		if (from - m.from).abs() > 1e-6 {
			self.change_edit(|e| {
				if let Some(m) = e.music.as_mut() {
					m.from = from;
				}
			});
		}
	}

	/// Put a song under the video, from its start (a song already there keeps how loud it was).
	fn add_music(&mut self, p: &std::path::Path) {
		let path = p.to_string_lossy().into_owned();
		match spotter_core::video::length(p) {
			Ok(len) => self.music_len = Some((path.clone(), len)),
			Err(e) => {
				self.note = Some(format!("can't play {}: {e}", p.display()));
				return;
			}
		}
		let old = self.clip_edit.music.clone().unwrap_or_default();
		let m = spotter_core::edit::Music { path, from: 0.0, ..old };
		self.note = Some(format!(
			"{} under the video: where it starts and how loud are in the export tab; beat looks follow it",
			m.name()
		));
		self.change_edit(|e| e.music = Some(m));
	}

	fn ask_open_project(&mut self, tx: &mut Tx) {
		let o = FileDialogOptions { title: "Open a project".into(), ..Default::default() }
			.filter("spotter project", &[EXTENSION]);
		tx.open_file(o, |s: &mut App, r, _| {
			if let Ok(Some(p)) = r {
				s.open_project(&p);
			}
		});
	}

	fn ask_save(&mut self, tx: &mut Tx) {
		if self.st.input.is_empty() {
			return;
		}
		let name = format!("{}.{EXTENSION}", self.st.name.replace(['/', '\\', ':'], "-"));
		let o = FileDialogOptions { title: "Save the project".into(), file_name: Some(name), ..Default::default() }
			.filter("spotter project", &[EXTENSION]);
		tx.save_file(o, |s: &mut App, r, _| {
			if let Ok(Some(p)) = r {
				s.note = match s.project().write(&p) {
					Ok(()) => Some(format!("saved {}", p.display())),
					Err(e) => Some(format!("{e:#}")),
				};
			}
		});
	}

	fn ask_export(&mut self, tx: &mut Tx) {
		if self.st.input.is_empty() || self.st.export.is_some() {
			return;
		}
		let name = format!("{}-{}.mp4", self.st.name.replace(['/', '\\', ':'], "-"), self.preset.replace(' ', "-"));
		let o = FileDialogOptions { title: "Export to".into(), file_name: Some(name), ..Default::default() }
			.filter("MP4 video", &["mp4"]);
		tx.save_file(o, |s: &mut App, r, _| {
			if let Ok(Some(p)) = r {
				s.export_to(p);
			}
		});
	}

	/// What the status line says on its left: a tool's hint, a stretch being marked, a note, or nothing.
	fn hint(&self) -> Option<(String, Color)> {
		if let Some(e) = self.st.error.as_ref().filter(|_| self.error_since.is_some()) {
			return Some((e.clone(), red()));
		}
		let tool = match self.tool {
			Tool::Pick => Some("pick: press a thing on the picture to follow only it · esc to stop"),
			Tool::Ignore => Some("ignore area: drag over a part of the picture to leave alone · esc to stop"),
			Tool::Only => Some("only area: drag over the only part to follow things in · esc to stop"),
			Tool::None => None,
		};
		if let Some(t) = tool {
			return Some((t.to_string(), ink()));
		}
		if let Some((from, m)) = self.marking {
			let (what, key) = match m {
				RangeMode::Cut => ("cut".to_string(), "x"),
				RangeMode::Bare => ("bare stretch".to_string(), "b"),
				RangeMode::Unmix => (format!("stretch without {}", self.look.mix.trim()), "u"),
			};
			return Some((format!("marking a {what} from {} · {key} again to end it here", clock(from)), accent()));
		}
		if self.compare {
			return Some(("the picture as it is · ` to go back to the look".into(), ink()));
		}
		if let Some(h) = &self.hovering {
			return Some((h.clone(), dim()));
		}
		self.note.as_ref().map(|n| (n.clone(), dim()))
	}

	/// Export to `out` in the chosen shape, and the other shapes asked for beside it, each named after
	/// its shape (clip-heat-wide.mp4).
	fn export_to(&mut self, out: PathBuf) {
		let mut shapes = vec![self.aspect];
		shapes.extend(self.also.iter().copied().filter(|&a| a != self.aspect));
		let es: Vec<Export> = shapes
			.iter()
			.enumerate()
			.map(|(i, &a)| {
				let path = if i == 0 {
					out.clone()
				} else {
					let stem = out.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
					let word = match ASPECTS[a].1 {
						None => "as-shot",
						Some(_) => ASPECTS[a].0.split(' ').next().unwrap_or("shape"),
					};
					out.with_file_name(format!("{stem}-{word}.mp4"))
				};
				self.export_one(path, a)
			})
			.collect();
		self.send(Cmd::Export(es));
	}

	fn export_one(&self, out: PathBuf, aspect: usize) -> Export {
		let start = self.trim_in.unwrap_or(0.0);
		let length = self.trim_out.map(|o| (o - start).max(0.1));
		Export {
			edit: self.clip_edit.clone(),
			upscale: self.upscale,
			input: self.st.input.clone(),
			name: self.st.name.clone(),
			look: self.look.clone(),
			start,
			length,
			aspect: shape_of(aspect),
			size: ASPECTS[aspect].1,
			fps: None,
			model: Some(self.detector()),
			device: gpu(),
			crf: 18,
			out,
			record: None,
			fast: self.quick,
			still: false,
		}
	}

	/// Ask where to keep a still of the frame on screen, as the export would make it.
	fn ask_still(&mut self, tx: &mut Tx) {
		if self.st.input.is_empty() || self.st.export.is_some() {
			return;
		}
		let t = self.position();
		let name =
			format!("{}-{}-{:.1}s.png", self.st.name.replace(['/', '\\', ':'], "-"), self.preset.replace(' ', "-"), t);
		let o = FileDialogOptions { title: "Keep a still".into(), file_name: Some(name), ..Default::default() }
			.filter("PNG picture", &["png"]);
		tx.save_file(o, move |s: &mut App, r, _| {
			if let Ok(Some(p)) = r {
				s.still_to(p, t);
			}
		});
	}

	/// A still of the frame at `t`, as the export makes it (its shape, size, upscaling and models),
	/// worked out from two seconds before so trails, copies and outlines have their history.
	fn still_to(&mut self, out: PathBuf, t: f64) {
		let mut e = self.export_one(out, self.aspect);
		let rate = if self.st.rate > 0.0 { self.st.rate } else { 30.0 };
		e.start = (t - 2.0).max(0.0);
		e.length = Some(t - e.start + 1.0 / rate);
		e.still = true;
		self.send(Cmd::Export(vec![e]));
	}
}

/// Where what's open is kept as it changes, to come back to with `spotter-app last`.
fn last_project() -> PathBuf {
	spotter_core::paths::home().join(format!("last.{EXTENSION}"))
}

/// Where the preview's volume is kept between runs.
fn volume_file() -> PathBuf {
	spotter_core::paths::home().join("volume.txt")
}

fn volume_kept() -> f32 {
	std::fs::read_to_string(volume_file())
		.ok()
		.and_then(|s| s.trim().parse::<f32>().ok())
		.unwrap_or(1.0)
		.clamp(0.0, 1.5)
}

impl App {
	/// How loud the preview plays: sent to the sound, kept for next time, and moving it turns the
	/// sound on if it was off.
	fn set_volume(&mut self, v: f32) {
		self.volume = v.clamp(0.0, 1.5);
		if self.muted && self.volume > 0.0 {
			self.muted = false;
			self.send(Cmd::Mute(false));
		}
		let _ = std::fs::write(volume_file(), format!("{:.2}", self.volume));
	}
}

fn init() -> App {
	App {
		surface: NativeSurface::new(),
		cmd: None,
		shared: Arc::default(),
		st: Status::default(),
		look: Look::default(),
		preset: "smoke".into(),
		open: vec!["objects".into()],
		edits: HashMap::new(),
		aspect: 0,
		model: 3,
		vocab: true,
		trim_in: None,
		trim_out: None,
		upscale: true,
		quick: false,
		muted: false,
		volume: volume_kept(),
		sent_volume: None,
		scrub: None,
		clip_edit: Edit::default(),
		tool: Tool::None,
		drag_from: None,
		marking: None,
		test_export: std::env::var_os("SPOTTER_TEST_EXPORT").map(PathBuf::from),
		note: None,
		tab: Tab::Look,
		search: TextEdit::new(""),
		more: Vec::new(),
		show_keys: false,
		getting: None,
		no_ffmpeg: None,
		needs_ffmpeg: false,
		ffmpeg_coming: None,
		asked: String::new(),
		autosaved: (Instant::now(), String::new()),
		shown_folder: None,
		errors: Vec::new(),
		error_since: None,
		panels: true,
		compare: false,
		addressing: false,
		strip: None,
		strip_shared: Arc::default(),
		hovering: None,
		hover_bar: None,
		selected_range: None,
		drafting: None,
		range_mode: RangeMode::Cut,
		grab: None,
		also: Vec::new(),
		sent_in: None,
		music_len: None,
		len_rx: None,
		link: TextEdit::new(""),
		song_link: TextEdit::new(""),
		fetching: None,
		song_shape: None,
		shape_rx: None,
		song_grab: None,
		vary_seed: 0x9E37_79B9_7F4A_7C15,
		mix_tab: false,
		mine: spotter_core::look::mine(),
		naming: None,
		history: Vec::new(),
		future: Vec::new(),
		remembered: None,
	}
}

fn setup(s: &mut App, wake: rune::Waker) {
	s.surface.set_waker(wake.clone());
	// a headless run (a snapshot, a script) has no player unless SPOTTER_LIVE asks for one; it gets a
	// made-up clip instead, so snapshots of the window show something
	if rune::tooling::is_headless() && std::env::var_os("SPOTTER_LIVE").is_none() {
		fixture(s);
		return;
	}
	// two programs started to find out, so not on the window's thread
	let (tx, rx) = std::sync::mpsc::channel();
	std::thread::spawn(move || tx.send(!spotter_core::video::have_ffmpeg()));
	s.no_ffmpeg = Some(rx);
	let tx =
		worker::spawn(s.surface.clone(), wake, s.shared.clone(), s.strip_shared.clone(), s.look.clone(), s.detector());
	s.cmd = Some(tx);
	// a project, or a video; "last" is what was open last time; nothing, and the window waits
	match std::env::args().skip(1).find(|a| !a.starts_with('-')) {
		None => {}
		Some(f) if f == "last" => s.open_project(&last_project()),
		Some(f) if f.ends_with(&format!(".{EXTENSION}")) => s.open_project(std::path::Path::new(&f)),
		Some(f) => {
			s.asked = f.clone();
			s.send(Cmd::Open(f));
		}
	}
}

/// A made-up clip for headless snapshots: ten seconds long, four shots, six things, a pick, an area and
/// a cut, over a still of `bench/sweden.mp4` if there is one (the test clips aren't in git).
fn fixture(s: &mut App) {
	use worker::Span;
	let mut st = Status {
		name: "sweden.mp4".into(),
		input: "bench/sweden.mp4".into(),
		duration: 10.1,
		rate: 30.0,
		t: 3.4,
		fps: 24,
		w: 576,
		h: 1024,
		detector: "d-fine x, 365 kinds, on the graphics card".into(),
		ahead: "looked ahead over all of it: 4 shots, 6 things".into(),
		counts: (30, 4, 2),
		seen: vec!["person".into(), "airplane".into(), "suv".into(), "truck".into()],
		view: spotter_core::edit::WHOLE,
		strip_version: 1,
		..Status::default()
	};
	if let Ok(f) = spotter_core::video::still(std::path::Path::new("bench/sweden.mp4")) {
		st.w = f.w;
		st.h = f.h;
		let mut bgra = f.rgba.clone();
		for p in bgra.chunks_exact_mut(4) {
			p.swap(0, 2);
		}
		let _ = s.surface.push_bytes(
			rune::native_surface::ByteFrame {
				width: f.w,
				height: f.h,
				format: rune::native_surface::Format::Bgra,
				color: rune::native_surface::ColorInfo::default(),
				planes: [bgra, Vec::new()],
			},
			1,
		);
	}
	let span = |cls: &str, from: f64, to: f64, lane: usize| Span {
		cls: cls.into(),
		conf: 0.9,
		from,
		to,
		lane,
		pick: Pick { t: (from + to) / 2.0, x: 0.5, y: 0.5 },
		picks: Vec::new(),
		dropped: false,
		drops: Vec::new(),
	};
	let picked = Pick { t: 1.4, x: 0.3, y: 0.6 };
	let strip = Strip {
		until: 10.1,
		cuts: vec![0.87, 2.67, 5.43],
		things: vec![
			span("person", 0.0, 0.85, 0),
			span("person", 0.0, 0.85, 1),
			span("person", 0.9, 2.6, 0),
			Span { picks: vec![picked], ..span("person", 1.2, 2.6, 1) },
			span("airplane", 2.7, 5.4, 0),
			span("suv", 5.5, 10.1, 0),
		],
		beats: (0..23).map(|i| 0.2 + i as f64 * 0.43).collect(),
	};
	*s.strip_shared.lock().unwrap() = Some(Arc::new(strip));
	*s.shared.lock().unwrap() = st;
	s.clip_edit = Edit {
		picks: vec![picked],
		zones: vec![Zone { x: 0.0, y: 0.8, w: 0.3, h: 0.2, mode: ZoneMode::Ignore }],
		ranges: vec![spotter_core::edit::Range { from: 6.2, to: 7.0, mode: RangeMode::Cut }],
		..Edit::default()
	};
}

fn poll(s: &mut App, _: f64, _: &mut Tx) -> bool {
	let mut changed = false;
	// the preview's volume, as the slider has it
	if s.sent_volume != Some(s.volume) {
		s.sent_volume = Some(s.volume);
		s.send(Cmd::Volume(s.volume));
	}
	// a song from a link: in, or why not; while it comes, the bar moves
	if let Some((url, rx, _)) = &s.fetching {
		changed = true;
		match rx.try_recv() {
			Ok(Ok(p)) => {
				s.fetching = None;
				s.add_music(&p);
			}
			Ok(Err(e)) => {
				s.note = Some(format!("couldn't get {url}: {e}"));
				s.fetching = None;
			}
			Err(_) => {}
		}
	}
	// the song's shape, for picking the part of it to use
	if let Some(m) = &s.clip_edit.music {
		let have = s.song_shape.as_ref().is_some_and(|x| x.0 == m.path);
		let asked = s.shape_rx.as_ref().is_some_and(|x| x.0 == m.path);
		if !have && !asked {
			let (tx, rx) = std::sync::mpsc::channel();
			let path = m.path.clone();
			std::thread::Builder::new()
				.name("song shape".into())
				.spawn(move || {
					let _ = tx.send(spotter_core::songs::shape(std::path::Path::new(&path), 400).unwrap_or_default());
				})
				.expect("a thread for the song's shape");
			s.shape_rx = Some((m.path.clone(), rx));
		}
	}
	if let Some((path, rx)) = &s.shape_rx
		&& let Ok(shape) = rx.try_recv()
	{
		s.song_shape = Some((path.clone(), Arc::new(shape)));
		s.shape_rx = None;
		changed = true;
	}
	// the in point is where a song starts
	if s.sent_in != Some(s.trim_in) {
		s.sent_in = Some(s.trim_in);
		s.send(Cmd::In(s.trim_in));
	}
	// a song from a project: how long it is, for the slider
	if let Some(m) = &s.clip_edit.music
		&& s.music_len.as_ref().is_none_or(|l| l.0 != m.path)
		&& s.len_rx.as_ref().is_none_or(|l| l.0 != m.path)
	{
		let (tx, rx) = std::sync::mpsc::channel();
		let path = m.path.clone();
		std::thread::spawn(move || tx.send(spotter_core::video::length(std::path::Path::new(&path)).unwrap_or(600.0)));
		s.len_rx = Some((m.path.clone(), rx));
	}
	if let Some((path, rx)) = &s.len_rx
		&& let Ok(len) = rx.try_recv()
	{
		s.music_len = Some((path.clone(), len));
		s.len_rx = None;
		changed = true;
	}
	if let Some(rx) = &s.no_ffmpeg
		&& let Ok(no) = rx.try_recv()
	{
		s.needs_ffmpeg = no;
		s.no_ffmpeg = None;
		changed = true;
	}
	if let Some(rx) = &s.ffmpeg_coming
		&& let Ok(got) = rx.try_recv()
	{
		s.ffmpeg_coming = None;
		match got {
			Ok(()) => {
				s.needs_ffmpeg = false;
				if !s.asked.is_empty() {
					s.open_source(s.asked.clone());
				}
			}
			Err(e) => s.note = Some(format!("couldn't get ffmpeg: {e}")),
		}
		changed = true;
	}
	// (not the made-up clip of a snapshot, which has no player)
	if s.cmd.is_some() {
		s.autosave();
	}
	let getting = spotter_core::model::getting();
	if getting != s.getting {
		s.getting = getting;
		changed = true;
	}
	if s.error_since.is_some_and(|t| t.elapsed() > ERROR_FOR) {
		s.error_since = None;
		changed = true;
	}
	let next = s.shared.lock().unwrap().clone();
	if next.error != s.st.error
		&& let Some(e) = &next.error
	{
		s.errors.insert(0, e.clone());
		s.errors.truncate(20);
		s.error_since = Some(Instant::now());
	}
	if next != s.st {
		if next.t != s.st.t && s.scrub.is_some_and(|t| (t - next.t).abs() < 0.2) {
			s.scrub = None;
		}
		if next.strip_version != s.st.strip_version {
			s.strip = s.strip_shared.lock().unwrap().clone();
		}
		s.st = next;
		// (once looking ahead is done, so the export can use it)
		if !s.st.input.is_empty()
			&& !s.st.opening
			&& s.st.t > 1.0
			&& !s.st.ahead.starts_with("looking")
			&& let Some(p) = s.test_export.take()
		{
			s.export_to(p);
		}
		true
	} else {
		changed
	}
}

fn view(s: &App, cx: &mut Cx<App>) -> El {
	let k = kit();
	let track_w = cx.layout_size("tl").map(|v| v[0]);
	let empty = s.st.name.is_empty() && !s.st.opening;
	let stage = col().key("stage").grow(1.0).min_w(0.0).h_full().fill(Color::BLACK).child(if empty {
		col().key("empty").grow(1.0).w_full().center().gap(12.0).children(if s.needs_ffmpeg {
			let asked = s.ffmpeg_coming.is_some();
			vec![
				text("spotter reads video with ffmpeg, and there isn't one here").font(mono(12.0)).color(dim()),
				if cfg!(windows) {
					let mb = spotter_core::video::FFMPEG_BYTES as f64 / 1e6;
					button(&format!("get ffmpeg · {mb:.0} MB"), asked, App::get_ffmpeg).key("get-ffmpeg")
				} else {
					text("install it (brew install ffmpeg), then open spotter again").font(mono(12.0)).color(dim())
				},
			]
		} else {
			start(s, k)
		})
	} else {
		let mut v = el().key("video").grow(1.0).w_full().min_h(0.0).native_surface(&s.surface, SurfaceFit::Fit);
		if s.tool != Tool::None {
			v = v.cursor(Cursor::Crosshair).on_drag(|s: &mut App, d: &mut Drag, tx: &mut Tx| {
				s.press_picture(d);
				tx.instant();
			});
		}
		v
	});
	let mut middle = row().grow(1.0).min_h(0.0).w_full();
	if s.panels {
		middle = middle.child(rail(s));
	}
	middle = middle.child(stage);
	if s.panels {
		middle = middle.child(inspector(s, k));
	}
	let body = col()
		.w_full()
		.h_full()
		.fill(bg())
		.on_drop_files(|s: &mut App, paths: &[PathBuf], _: &mut Tx| {
			if let Some(p) = paths.first() {
				let ext = p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
				if ext == EXTENSION {
					s.open_project(p);
				} else if ext == "json" {
					s.add_look_file(p);
				} else if AUDIO.contains(&ext.as_str()) {
					s.add_music(p);
				} else {
					s.open_source(p.to_string_lossy().into_owned());
				}
			}
		})
		.accepts_extensions(&[VIDEO, AUDIO, &[EXTENSION, "json"]].concat())
		.child(top(s, k))
		.child(middle)
		.child(timeline(s, track_w))
		.child(status_line(s));
	rune_kit::root(body, Vec::new())
}

/// What the window shows with nothing open: open a video, paste where one is, or a random one.
fn start(s: &App, k: &Kit) -> Vec<El> {
	let link = TextInput::new(&s.link)
		.placeholder("or paste a path or an address, then enter")
		.width(Some(360.0))
		.on_edit(|s: &mut App| &mut s.link)
		.on_submit(|s: &mut App, _: &mut Tx| {
			let arg = typed(&s.link.text);
			if !arg.is_empty() {
				s.link = TextEdit::new("");
				s.open_source(arg);
			}
		})
		.build(k)
		.key("input-link");
	vec![
		text("drop a video here").font(mono(12.0)).color(dim()),
		row()
			.gap(8.0)
			.center_y()
			.child(primary("open a video…", |_: &mut App| {}).key("start-open").on_press_tx(App::ask_open))
			.child(button("random video", false, |s: &mut App| s.open_source("random".into())).key("start-random")),
		link,
	]
}

/// A path or an address as typed or pasted, without the quotes Windows' "copy as path" puts round it.
fn typed(s: &str) -> String {
	s.trim().trim_matches('"').trim().to_string()
}

/// The top bar: the name, the clip (press it for an address box), and opening and saving.
fn top(s: &App, k: &Kit) -> El {
	let clip: El = if s.addressing {
		TextInput::new(s.edits.get("").unwrap_or(&TextEdit::new("")))
			.placeholder("a file, a web address, files:path or random")
			.width(Some(420.0))
			.on_edit(|s: &mut App| s.edits.entry(String::new()).or_insert_with(|| TextEdit::new("")))
			.on_submit(|s: &mut App, _: &mut Tx| {
				let arg = s.edits.get("").map(|e| typed(&e.text)).unwrap_or_default();
				if !arg.is_empty() {
					s.addressing = false;
					s.open_source(arg);
				}
			})
			.build(k)
			.key("input-address")
	} else {
		el().key("clip")
			.h(ROW)
			.px(6.0)
			.center_y()
			.min_w(0.0)
			.hover(raised())
			.focusable(true)
			.role("button")
			.label("open an address")
			.on_press_tx(|s: &mut App, tx: &mut Tx| {
				s.addressing = true;
				tx.focus("input-address", false);
			})
			.child(
				text(if s.st.opening {
					"opening…".to_string()
				} else if s.st.name.is_empty() {
					"no video".to_string()
				} else {
					s.st.name.clone()
				})
				.font(mono(12.0))
				.color(ink())
				.truncate()
				.min_w(0.0)
				.help_when_truncated(s.st.input.clone()),
			)
	};
	let word = |label: &str, key: &str| button(label, false, |_: &mut App| {}).key(key.to_string());
	row()
		.h(TOP)
		.w_full()
		.px(10.0)
		.gap(10.0)
		.center_y()
		.fill(surface())
		.border_sides([0.0, 0.0, 1.0, 0.0], line())
		.child(text("spotter").font(strong(13.0)).color(ink()))
		.child(clip)
		.child(spacer())
		.child(word("random", "random").on_press(|s: &mut App| s.open_source("random".into())))
		.child(word("open…", "open").on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_open(tx)))
		.child(word("open project…", "open-project").on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_open_project(tx)))
		.child(word("save project…", "save-project").on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_save(tx)))
		.child(button("?", s.show_keys, |s: &mut App| s.show_keys = !s.show_keys).key("keys"))
}

/// The inspector: three tabs, and the one showing (or the keys, after ?).
fn inspector(s: &App, k: &Kit) -> El {
	let tabs_el = tabs(
		&[
			("look", s.tab == Tab::Look, "tab-look"),
			("follow", s.tab == Tab::Follow, "tab-follow"),
			("export", s.tab == Tab::Export, "tab-export"),
		],
		|s: &mut App, i| {
			s.tab = [Tab::Look, Tab::Follow, Tab::Export][i];
			s.show_keys = false;
		},
	);
	let content = if s.show_keys {
		let keys = KEYS.iter().map(|(key, what)| {
			row()
				.w_full()
				.gap(10.0)
				.child(text(key.to_string()).font(mono(12.0)).color(ink()).w(72.0))
				.child(text(what.to_string()).font(mono(12.0)).color(dim()).wrap().grow(1.0).min_w(0.0))
		});
		let errors = s.errors.iter().map(|e| text(e.clone()).font(mono(12.0)).color(dim()).wrap().w_full());
		col()
			.w_full()
			.gap(2.0)
			.child(header("keys"))
			.children(keys)
			.children((!s.errors.is_empty()).then(|| header("what went wrong")))
			.children(errors)
	} else {
		match s.tab {
			Tab::Look => look_tab(s, k),
			Tab::Follow => follow(s),
			Tab::Export => export(s, k),
		}
	};
	col()
		.key("inspector")
		.w(INSPECTOR)
		.h_full()
		.min_h(0.0)
		.fill(surface())
		.border_sides([0.0, 0.0, 0.0, 1.0], line())
		.child(row().w_full().px(12.0).pt(4.0).border_sides([0.0, 0.0, 1.0, 0.0], line()).child(tabs_el))
		.child(
			col()
				.key("controls")
				.grow(1.0)
				.min_h(0.0)
				.w_full()
				.scroll(ScrollSpec { stick_bottom: false })
				.child(col().w_full().padding(12.0).gap(8.0).child(content)),
		)
}

/// The status line: what a tool, a mark or a note says on the left; the detector, the look ahead, the
/// speed, the size and the counts on the right.
fn status_line(s: &App) -> El {
	let (said, colour) = s.hint().unwrap_or_default();
	let mut right: Vec<String> = Vec::new();
	if let Some((n, total)) = s.st.export {
		let (i, of) = s.st.export_part;
		right.push(if of > 1 {
			format!("exporting {i} of {of}: {}%", n * 100 / total.max(1))
		} else {
			format!("exporting {}%", n * 100 / total.max(1))
		});
	}
	match &s.getting {
		Some(g) => right.push(g.line()),
		None if !s.st.detector.is_empty() => right.push(s.st.detector.clone()),
		None => {}
	}
	if s.st.w > 0 {
		right.push(format!("{} fps", s.st.fps));
		right.push(format!("{}x{}", s.st.w, s.st.h));
		right.push(format!("{} pts {} moving {} named", s.st.counts.0, s.st.counts.1, s.st.counts.2));
	}
	row()
		.h(STATUS)
		.w_full()
		.px(10.0)
		.gap(12.0)
		.center_y()
		.fill(surface())
		.border_sides([1.0, 0.0, 0.0, 0.0], line())
		.child(text(said).font(mono(11.0)).color(colour).truncate().grow(1.0).min_w(0.0))
		.children((!s.st.mosh.is_empty()).then(|| {
			text(s.st.mosh.clone()).key("mosh").font(mono(11.0)).color(orange()).truncate().min_w(0.0).shrink()
		}))
		.children((!s.st.ahead.is_empty()).then(|| {
			text(s.st.ahead.clone()).key("ahead").font(mono(11.0)).color(dim()).truncate().min_w(0.0).shrink()
		}))
		.child(text(right.join(" · ")).font(digits(11.0)).color(dim()).truncate().min_w(0.0).shrink())
}

fn main() {
	log::start();
	let _ = rune::set_app_icon(include_bytes!("../../../assets/icon.png").as_slice());
	// whatever spotter does, the rest of the computer comes first (safety.rs)
	safety::guard();
	rune::run(AppSpec::<App> {
		fonts: FONTS,
		families: ("DM Mono", "DM Mono"),
		setup: Some(setup),
		poll: Some(poll),
		on_key: Some(on_key),
		min_size: Some((900.0, 560.0)),
		..AppSpec::new("spotter", (1440.0, 860.0), init, view)
	});
}
