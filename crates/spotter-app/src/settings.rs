//! The look tab: the look's settings, the ones it uses first. A group is live when its switch is on
//! (echo has copies, glyphs aren't off…); live groups come first with their switch and the knobs the
//! look sets or you've moved, then "+ n more"; groups that are off, and the base ones every look has,
//! follow as names to press. A knob moved from the look gets an orange dot that puts it back; a slider
//! marks the look's own value. Typing in the search lists every knob that matches.

use crate::*;
use serde_json::Map;

/// The knobs that switch each group on; a group not listed has none (it's always there).
pub(crate) fn masters(group: &str) -> &'static [&'static str] {
	match group {
		"objects" => &["detect"],
		"boxes" => &["boxes", "regions"],
		"labels" => &["labels"],
		"lines" => &["linkMode"],
		"trails" => &["trail"],
		"patches" => &["patches"],
		"overlay text" => &["hud"],
		"layers" => &["world", "within", "contour", "rings"],
		"echo" => &["echo"],
		"insets" => &["insets"],
		"glyphs" => &["glyphs"],
		"depth" => &["depth"],
		"camera" => &["follow"],
		"reveal" => &["reveal"],
		"mix" => &["mix"],
		"glitch" => &["mosh", "tear", "trichrome", "stutter", "kaleido"],
		"beat" => &["beat"],
		"foresight" => &["foresight"],
		_ => &[],
	}
}

/// Whether a switch's value means on.
pub(crate) fn is_on(v: Option<&Value>) -> bool {
	match v {
		Some(Value::Bool(b)) => *b,
		Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
		Some(Value::String(s)) => !matches!(s.as_str(), "" | "off" | "same" | "none"),
		_ => false,
	}
}

/// The look as it is, the preset it came from and spotter's defaults, each as name → value, so a
/// knob's three values can be read without turning the whole look into JSON every time.
pub(crate) struct Values {
	pub now: Map<String, Value>,
	pub look: Map<String, Value>,
	pub default: Map<String, Value>,
}

impl Values {
	pub(crate) fn of(s: &App) -> Self {
		let map = |l: &Look| match serde_json::to_value(l) {
			Ok(Value::Object(m)) => m,
			_ => Map::new(),
		};
		// on the look mixed in: it as it's drawn, against it as it comes
		if s.editing_mix() {
			let base = s.mix_base().unwrap_or_default();
			return Values { now: map(&s.shown_look()), look: map(&base), default: map(&Look::default()) };
		}
		Values {
			now: map(&s.look),
			look: map(&s.base_look(&s.preset).unwrap_or_default()),
			default: map(&Look::default()),
		}
	}

	/// Moved from the look's own value.
	pub(crate) fn moved(&self, k: &str) -> bool {
		self.now.get(k) != self.look.get(k)
	}

	/// Set by the look, or moved: what to show of a group without opening it all.
	pub(crate) fn touched(&self, k: &str) -> bool {
		self.look.get(k) != self.default.get(k) || self.moved(k)
	}
}

/// How a group stands for this look.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stand {
	Live,
	Off,
	Base,
}

pub(crate) fn stand(g: &spotter_core::look::KnobGroup, v: &Values) -> Stand {
	let m = masters(&g.name);
	if m.iter().any(|k| is_on(v.now.get(*k))) {
		return Stand::Live;
	}
	if !m.is_empty() {
		return Stand::Off;
	}
	if g.knobs.iter().any(|kb| v.touched(&kb.k)) { Stand::Live } else { Stand::Base }
}

pub(crate) fn look_tab(s: &App, k: &Kit) -> El {
	let v = Values::of(s);
	let on_mix = s.editing_mix();
	// on the look mixed in, what it takes from this one (what's found, mixing) isn't its own to change
	let shared = if on_mix { Look::shared_keys() } else { Vec::new() };
	let shown = |kb: &&Knob| !shared.contains(&kb.k);
	let changed = Look::knobs().iter().flat_map(|g| &g.knobs).filter(shown).any(|kb| v.moved(&kb.k));
	let preset = s.preset.clone();
	let yours = s.mine.iter().any(|(n, _)| *n == s.preset);
	let gone = s.preset.clone();
	let mixing = s.look.mix.trim().to_string();
	let title = if on_mix { mixing.clone() } else { s.preset.clone() };
	let mut c = col().w_full().gap(8.0).child(
		row()
			.w_full()
			.h(ROW + 4.0)
			.center_y()
			.gap(6.0)
			.child(text(title).font(strong(13.0)).color(if on_mix { aqua() } else { ink() }).truncate().min_w(0.0))
			.children(
				(!mixing.is_empty() && !on_mix)
					.then(|| text(format!("+ {mixing}")).font(strong(13.0)).color(aqua()).truncate().min_w(0.0)),
			)
			.children(
				on_mix.then(|| text(format!("in {}", s.preset)).font(strong(13.0)).color(dim()).truncate().min_w(0.0)),
			)
			.children(changed.then(|| text("*").font(strong(13.0)).color(orange())))
			.child(spacer())
			.children(changed.then(|| {
				if on_mix {
					button("reset", false, |s: &mut App| {
						let mut l = s.look.clone();
						l.mix_set.clear();
						s.set_look(l);
					})
					.key("reset-look")
				} else {
					button("reset", false, move |s: &mut App| s.pick_preset(&preset)).key("reset-look")
				}
			}))
			.children(
				(yours && !changed && !on_mix)
					.then(|| button("delete", false, move |s: &mut App| s.delete_mine(&gone)).key("delete-look")),
			)
			.child(button("vary", false, vary).key("vary"))
			.child(button("save as…", false, |_: &mut App| {}).key("save-look").on_press_tx(
				|s: &mut App, tx: &mut Tx| {
					s.naming = Some(TextEdit::new(""));
					tx.focus("input-name", false);
				},
			)),
	);
	// two looks mixed: whose settings these are
	if !mixing.is_empty() {
		let opts = vec![(s.preset.clone(), !on_mix), (mixing.clone(), on_mix)];
		c = c.child(
			row()
				.w_full()
				.gap(8.0)
				.center_y()
				.child(text("settings of").font(mono(12.0)).color(dim()))
				.child(keyed_chips("editing", &opts, |s: &mut App, i| s.show_mix(i == 1))),
		);
	}
	if let Some(name) = &s.naming {
		c = c.child(
			TextInput::new(name)
				.placeholder("a name for this look, then enter")
				.on_edit(|s: &mut App| s.naming.get_or_insert_with(|| TextEdit::new("")))
				.on_submit(|s: &mut App, _: &mut Tx| {
					let name = s.naming.as_ref().map(|e| e.text.clone()).unwrap_or_default();
					s.save_mine(&name);
				})
				.build(k)
				.key("input-name"),
		);
	}
	c = c.child(
		TextInput::new(&s.search)
			.placeholder("/ search settings")
			.on_edit(|s: &mut App| &mut s.search)
			.build(k)
			.key("input-search"),
	);
	let query = s.search.text.trim().to_lowercase();
	if !query.is_empty() {
		// every knob that matches, with its group named on the right
		for g in Look::knobs() {
			for kb in &g.knobs {
				if shown(&kb)
					&& (kb.label.to_lowercase().contains(&query)
						|| kb.k.to_lowercase().contains(&query)
						|| g.name.contains(&query))
				{
					c = c.child(knob(s, k, kb, &v, Some(&g.name)));
				}
			}
		}
		return c;
	}
	let groups = Look::knobs();
	let mut off: Vec<&String> = Vec::new();
	let mut base: Vec<&String> = Vec::new();
	for g in groups {
		// on the look mixed in, its mixing and finding are this look's
		if on_mix && g.knobs.iter().all(|kb| shared.contains(&kb.k)) {
			continue;
		}
		let st = stand(g, &v);
		let opened = s.open.contains(&g.name);
		if st == Stand::Live || opened {
			c = c.child(group(s, k, g, &v, st));
		} else if st == Stand::Off {
			off.push(&g.name);
		} else {
			base.push(&g.name);
		}
	}
	for (title, names) in [("off", off), ("base", base)] {
		if names.is_empty() {
			continue;
		}
		c = c.child(el().h(8.0)).child(header(title)).child(row().w_full().gap(2.0).flex_wrap().children(
			names.into_iter().map(|n| {
				let name = n.clone();
				button(n, false, move |s: &mut App| s.open.push(name.clone())).key(format!("group-{n}"))
			}),
		));
	}
	c
}

/// A live (or opened) group: its switch, the knobs the look sets or you've moved, and the rest behind
/// "+ n more", or all of them once that's pressed.
fn group(s: &App, k: &Kit, g: &spotter_core::look::KnobGroup, v: &Values, st: Stand) -> El {
	let name = g.name.clone();
	let moved = g.knobs.iter().filter(|kb| v.moved(&kb.k)).count();
	let all = s.more.contains(&g.name) || (st != Stand::Live && s.open.contains(&g.name));
	let head = row()
		.key(format!("group-{}", g.name))
		.w_full()
		.h(ROW)
		.center_y()
		.gap(8.0)
		.focusable(true)
		.role("button")
		.label(g.name.clone())
		.on_press(move |s: &mut App| {
			// a group opened from the off or base names closes again from its heading
			s.open.retain(|n| *n != name);
			s.more.retain(|n| *n != name);
		})
		.child(text(g.name.clone()).font(mono(11.0)).color(dim()))
		.child(el().grow(1.0).h(1.0).fill(line()))
		.children((moved > 0).then(|| text(format!("{moved} changed")).font(mono(11.0)).color(orange())));
	let m = masters(&g.name);
	// on the look mixed in, what it takes from this one isn't shown
	let shared = if s.editing_mix() { Look::shared_keys() } else { Vec::new() };
	let own: Vec<&Knob> = g.knobs.iter().filter(|kb| !shared.contains(&kb.k)).collect();
	let shown: Vec<&Knob> =
		own.iter().copied().filter(|kb| all || m.contains(&kb.k.as_str()) || v.touched(&kb.k)).collect();
	let hidden = own.len() - shown.len();
	let mut c = col().w_full().gap(6.0).child(el().h(6.0)).child(head);
	if all && let Some(note) = &g.note {
		c = c.child(text(note.clone()).font(mono(11.0)).color(mute()).wrap());
	}
	for kb in shown {
		c = c.child(knob(s, k, kb, v, None));
	}
	if hidden > 0 {
		let name = g.name.clone();
		c = c.child(
			el().key(format!("more-{}", g.name))
				.h(ROW)
				.center_y()
				.focusable(true)
				.role("button")
				.on_press(move |s: &mut App| s.more.push(name.clone()))
				.child(text(format!("+ {hidden} more")).font(mono(11.0)).color(mute())),
		);
	}
	c
}

/// One knob: an orange dot if it's moved from the look (pressing it puts it back), its name, and its
/// control; `group` names its group on the right, in a search's list.
pub(crate) fn knob(s: &App, k: &Kit, kb: &Knob, v: &Values, group: Option<&str>) -> El {
	let key = kb.k.clone();
	let value = v.now.get(&key).cloned().unwrap_or(Value::Null);
	let moved = v.moved(&key);
	let back = v.look.get(&key).cloned().unwrap_or(Value::Null);
	let dot = {
		let (key, back) = (key.clone(), back.clone());
		el().w(12.0)
			.h(ROW)
			.center_y()
			.key(format!("reset-{key}"))
			.children(moved.then(|| text("•").font(mono(12.0)).color(orange())))
			.on_press(move |s: &mut App| {
				if moved {
					s.set(&key, back.clone());
				}
			})
	};
	let label = text(kb.label.clone())
		.font(mono(12.0))
		.color(dim())
		.w(122.0)
		.min_w(122.0)
		.truncate()
		.help(kb.hint.clone().unwrap_or_default());
	let named = |r: El| match group {
		Some(g) => r.child(text(g.to_string()).font(mono(11.0)).color(mute())),
		None => r,
	};
	if let (Some(min), Some(max), Some(step)) = (kb.min, kb.max, kb.step) {
		let x = value.as_f64().unwrap_or(min as f64) as f32;
		let decimals = if step >= 1.0 {
			0
		} else if step >= 0.1 {
			1
		} else {
			2
		};
		let at = back.as_f64().map(|b| b as f32);
		return named(
			row()
				.w_full()
				.gap(6.0)
				.center_y()
				.child(dot)
				.child(label)
				.child(slider(key, x, min, max, step, at))
				.child(text(format!("{x:.decimals$}")).font(digits(11.0)).color(ink()).w(40.0)),
		);
	}
	if let Some(options) = &kb.options {
		let current = value.as_str().unwrap_or("").to_string();
		let opts: Vec<(String, bool)> = options.iter().map(|o| (o.clone(), *o == current)).collect();
		let names = options.clone();
		let prefix = format!("opt-{key}");
		return col().w_full().gap(4.0).child(named(row().w_full().gap(6.0).center_y().child(dot).child(label))).child(
			row()
				.w_full()
				.pl(18.0)
				.child(keyed_chips(&prefix, &opts, move |s: &mut App, i| s.set(&key, Value::String(names[i].clone())))),
		);
	}
	match kb.kind.as_deref() {
		// another look, chosen in the rail
		Some("look") => {
			let name = value.as_str().unwrap_or("").trim().to_string();
			let shown = if name.is_empty() {
				text("right-click a look on the left").font(mono(12.0)).color(mute())
			} else {
				text(name.clone()).font(mono(12.0)).color(aqua())
			};
			named(
				row()
					.w_full()
					.gap(6.0)
					.center_y()
					.child(dot)
					.child(label)
					.child(shown.truncate().min_w(0.0))
					.child(spacer())
					.children(
						(!name.is_empty())
							.then(|| button("take out", false, move |s: &mut App| s.mix_with(&name)).key("mix-out")),
					),
			)
		}
		Some("toggle") => {
			let on = value.as_bool().unwrap_or(false);
			named(row().w_full().gap(6.0).center_y().child(dot).child(label).child(toggle(
				&format!("toggle-{key}"),
				on,
				move |s: &mut App, b| s.set(&key, Value::Bool(b)),
			)))
		}
		Some("color") => {
			let swatch = rune::color::hex(value.as_str().unwrap_or("#ffffff")).unwrap_or(Color::WHITE);
			let (k1, k2) = (key.clone(), key.clone());
			named(
				row()
					.w_full()
					.gap(6.0)
					.center_y()
					.child(dot)
					.child(label)
					.child(el().size(16.0, 16.0).fill(swatch).border(1.0, faint()))
					.child(
						TextInput::new(s.edits.get(&key).unwrap_or(&TextEdit::new(value.as_str().unwrap_or(""))))
							.width(Some(96.0))
							.on_edit(move |s: &mut App| s.edit(&k1))
							.on_change(move |s: &mut App, _: &mut Tx| {
								let t = s.edits.get(&k2).map(|e| e.text.trim().to_string()).unwrap_or_default();
								if t.len() == 7 && t.starts_with('#') && rune::color::hex(&t).is_some() {
									s.set(&k2, Value::String(t));
								}
							})
							.build(k)
							.key(format!("input-colour-{key}")),
					),
			)
		}
		kind => {
			let (k1, k2) = (key.clone(), key.clone());
			let current = TextEdit::new(value.as_str().unwrap_or(""));
			let edit = s.edits.get(&key).unwrap_or(&current);
			let change = move |s: &mut App, _: &mut Tx| {
				let t = s.edits.get(&k2).map(|e| e.text.clone()).unwrap_or_default();
				s.set(&k2, Value::String(t));
			};
			let field = if kind == Some("lines") {
				TextArea::new(edit).auto_grow(2, 6).on_edit(move |s: &mut App| s.edit(&k1)).on_change(change).build(k)
			} else {
				TextInput::new(edit).on_edit(move |s: &mut App| s.edit(&k1)).on_change(change).build(k)
			}
			.key(format!("input-text-{key}"));
			col()
				.w_full()
				.gap(4.0)
				.child(named(row().w_full().gap(6.0).center_y().child(dot).child(label)))
				.child(row().w_full().pl(18.0).child(field))
		}
	}
}

/// Settings that vary leaves alone: what's recognised and how, the export's, and words to type.
const STEADY: &[&str] = &[
	"detect",
	"objectKinds",
	"confidence",
	"detectEvery",
	"ahead",
	"aheadSmooth",
	"speed",
	"outMax",
	"bitrate",
	"smoothSlow",
	"follow",
	"followZoom",
	"followSmooth",
	"idBase",
	"scene",
	"beat",
	"matte",
	"points",
	"objects",
	"font",
];

/// The site's colours, for vary to choose from.
const GRUVBOX: &[&str] = &[
	"#fb4934", "#b8bb26", "#fabd2f", "#83a598", "#d3869b", "#8ec07c", "#fe8019", "#ebdbb2", "#a89984", "#1d2021",
	"#cc241d", "#98971a", "#d79921", "#458588", "#b16286", "#689d6a", "#d65d0e",
];

/// A new look near this one, for finding something by chance: the sliders of the groups the look
/// uses each nudged up to a third of their range, now and then a different option or colour. One
/// step to undo.
pub(crate) fn vary(s: &mut App) {
	let v = Values::of(s);
	let mut seed = s.vary_seed.wrapping_add(
		std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64),
	) | 1;
	s.vary_seed = seed;
	let mut next = || {
		seed ^= seed << 13;
		seed ^= seed >> 7;
		seed ^= seed << 17;
		(seed >> 11) as f32 / (1u64 << 53) as f32
	};
	let mut look = s.shown_look();
	let shared = if s.editing_mix() { Look::shared_keys() } else { Vec::new() };
	for g in Look::knobs() {
		if stand(g, &v) != Stand::Live {
			continue;
		}
		for kb in &g.knobs {
			if STEADY.contains(&kb.k.as_str()) || shared.contains(&kb.k) {
				continue;
			}
			let now = v.now.get(&kb.k).cloned().unwrap_or(Value::Null);
			let new = if let (Some(min), Some(max), Some(step)) = (kb.min, kb.max, kb.step) {
				let x = now.as_f64().unwrap_or(min as f64) as f32;
				let y = (x + (next() - 0.5) * 0.66 * (max - min)).clamp(min, max);
				Some(serde_json::json!(((y / step).round() * step).clamp(min, max)))
			} else if let Some(options) = &kb.options {
				// switches stay as they are, so the group stays on
				(next() < 0.3 && !masters(&g.name).contains(&kb.k.as_str()) && !options.is_empty())
					.then(|| Value::String(options[(next() * options.len() as f32) as usize % options.len()].clone()))
			} else if kb.kind.as_deref() == Some("color") {
				(next() < 0.35).then(|| {
					Value::String(GRUVBOX[(next() * GRUVBOX.len() as f32) as usize % GRUVBOX.len()].to_string())
				})
			} else {
				None
			};
			if let Some(n) = new {
				look.set(&kb.k, n);
			}
		}
	}
	s.remembered = None;
	s.set_shown(look);
	s.note = Some("varied: ctrl+z goes back, d varies again".into());
}

#[cfg(test)]
mod tests {
	use super::*;

	/// Every look shows something in the look tab: a live group with something in it.
	#[test]
	fn every_look_has_a_live_group() {
		for name in Look::presets() {
			let look = Look::preset(name).unwrap();
			let map = |l: &Look| match serde_json::to_value(l) {
				Ok(Value::Object(m)) => m,
				_ => Map::new(),
			};
			let v = Values { now: map(&look), look: map(&look), default: map(&Look::default()) };
			let live = Look::knobs().iter().filter(|g| stand(g, &v) == Stand::Live).count();
			assert!(live > 0, "{name} shows no live group");
		}
	}
}
