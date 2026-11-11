//! The keyboard (? shows the table in the window). Single letters do nothing
//! while a text box has the focus, so typing a search doesn't change the look.

use crate::*;

/// What each key does, for ? to show.
pub(crate) const KEYS: &[(&str, &str)] = &[
	("space", "play or pause"),
	("m", "sound on or off"),
	("- =", "the preview's volume down and up"),
	("← →", "a second back or on (shift: five)"),
	(", .", "a frame back or on"),
	("[ ]", "the cut in the footage before or after"),
	("home end", "the start or the end (in or out if set)"),
	("j k", "the next or the last look"),
	("1 2 3", "look, follow, export"),
	("/", "search the look's settings"),
	("d", "vary the look: nudge its settings by chance (ctrl+z to go back)"),
	("`", "the picture as it is, nothing drawn (again to go back)"),
	("p a A", "pick a thing, ignore an area, only an area"),
	("i o", "in or out at the playhead (shift: clear it)"),
	("x b u", "start or end a cut, a bare stretch, or (with a look mixed in) one without it"),
	("delete", "remove the stretch selected on the timeline"),
	("f", "the picture and the timeline only"),
	("ctrl+o", "open a video"),
	("ctrl+s", "save the project"),
	("ctrl+e", "export"),
	("ctrl+shift+e", "a still of the frame on screen, as the export would make it (PNG)"),
	("ctrl+r", "a random video"),
	("ctrl+z", "undo a change to the look or the edit (shift, or ctrl+y: redo)"),
	("?", "these keys"),
	("esc", "drop the tool, close the box, clear the search or the error"),
];

/// Whether a text box has the focus: every one has a key starting `input-`.
fn typing(tx: &Tx) -> bool {
	tx.focused().is_some_and(|k| k.starts_with("input-"))
}

pub(crate) fn on_key(s: &mut App, ev: &KeyEvent, tx: &mut Tx) -> bool {
	let shift = ev.mods.shift;
	let t = s.position();
	let rate = if s.st.rate > 0.0 { s.st.rate } else { 30.0 };
	match &ev.key {
		Key::Escape => {
			if s.tool != Tool::None {
				s.set_tool(Tool::None);
			} else if s.addressing {
				s.addressing = false;
			} else if s.naming.is_some() {
				s.naming = None;
			} else if !s.search.text.is_empty() {
				s.search = TextEdit::new("");
			} else if s.show_keys {
				s.show_keys = false;
			} else if s.error_since.is_some() {
				s.error_since = None;
			} else {
				return false;
			}
		}
		Key::Delete | Key::Backspace if !typing(tx) && s.selected_range.is_some() => {
			let i = s.selected_range.take().unwrap_or(usize::MAX);
			s.change_edit(|e| {
				if i < e.ranges.len() {
					e.ranges.remove(i);
				}
			});
		}
		// on a thing's bar: take it out (or bring it back)
		Key::Delete | Key::Backspace if !typing(tx) && s.hover_bar.is_some() => {
			if let Some(sp) = s.hover_bar.clone() {
				s.toggle_drop(&sp);
				s.hover_bar = None;
			}
		}
		Key::Left => s.seek(t - if shift { 5.0 } else { 1.0 }),
		Key::Right => s.seek(t + if shift { 5.0 } else { 1.0 }),
		Key::Home => s.seek(s.trim_in.unwrap_or(0.0)),
		Key::End => s.seek(s.trim_out.unwrap_or(s.st.duration - 0.05)),
		Key::Command(c) => match c {
			'o' => s.ask_open(tx),
			's' => s.ask_save(tx),
			'e' if ev.mods.shift => s.ask_still(tx),
			'e' => s.ask_export(tx),
			'r' => s.open_source("random".into()),
			'z' => s.undo(ev.mods.shift),
			'y' => s.undo(true),
			_ => return false,
		},
		Key::Text(c) if typing(tx) => {
			let _ = c;
			return false;
		}
		Key::Text(c) => match c.as_str() {
			" " => s.send(Cmd::Toggle),
			"m" => {
				s.muted = !s.muted;
				s.send(Cmd::Mute(s.muted));
			}
			"," => s.seek(t - 1.0 / rate),
			"." => s.seek(t + 1.0 / rate),
			"[" => {
				if let Some(c) = s.strip_cuts().into_iter().rev().find(|&c| c < t - 0.05) {
					s.seek(c);
				}
			}
			"]" => {
				if let Some(c) = s.strip_cuts().into_iter().find(|&c| c > t + 0.05) {
					s.seek(c);
				}
			}
			"j" => s.pick_preset(&rail::step(&s.mine, &s.preset, 1)),
			"k" => s.pick_preset(&rail::step(&s.mine, &s.preset, -1)),
			"1" => s.tab = Tab::Look,
			"2" => s.tab = Tab::Follow,
			"3" => s.tab = Tab::Export,
			"/" => {
				s.tab = Tab::Look;
				tx.focus("input-search", true);
			}
			"`" => {
				s.compare = !s.compare;
				s.send(Cmd::Bare(s.compare));
			}
			"d" => vary(s),
			"-" => s.set_volume(s.volume - 0.1),
			"=" | "+" => s.set_volume(s.volume + 0.1),
			"p" => s.set_tool(Tool::Pick),
			"a" => s.set_tool(Tool::Ignore),
			"A" => s.set_tool(Tool::Only),
			"i" => s.trim_in = Some(t),
			"o" => s.trim_out = Some(t),
			"I" => s.trim_in = None,
			"O" => s.trim_out = None,
			"x" => s.mark(RangeMode::Cut),
			"b" => s.mark(RangeMode::Bare),
			"u" => s.mark(RangeMode::Unmix),
			"f" => s.panels = !s.panels,
			"?" => {
				s.show_keys = !s.show_keys;
				if s.show_keys {
					s.tab = Tab::Look;
				}
			}
			_ => return false,
		},
		_ => return false,
	}
	tx.instant();
	true
}
