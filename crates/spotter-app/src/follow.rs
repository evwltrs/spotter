//! The follow tab: what's followed (everything, picked things, the main thing), the detector, which
//! kinds, the tools for the picture (pick, ignore an area, only an area), and every pick, area and
//! stretch of time as a row of its own, each removable alone and a press away from its moment.

use crate::*;

/// Words for a key, dim, after a tool's name.
fn hint(e: El, key: &str) -> El {
	e.child(text(key.to_string()).font(mono(11.0)).color(mute()))
}

pub(crate) fn follow(s: &App) -> El {
	let e = &s.clip_edit;
	let keeps = [("everything", Keep::All), ("picked", Keep::Picked), ("main thing", Keep::Main)];
	let keep_chips: Vec<(String, bool)> = keeps.iter().map(|(l, k)| (l.to_string(), e.keep == *k)).collect();
	let models: Vec<(String, bool)> = MODELS.iter().enumerate().map(|(i, m)| (m.to_string(), i == s.model)).collect();
	let vocabs: Vec<(String, bool)> = vec![("80 kinds".into(), !s.vocab), ("365 kinds".into(), s.vocab)];
	// the kinds seen in this clip first, the rest folded away
	let mut kinds: Vec<String> = Vec::new();
	for k in s.st.seen.iter().chain(&e.kinds) {
		if !kinds.contains(k) {
			kinds.push(k.clone());
		}
	}
	let seen = kinds.len();
	let all_kinds = s.more.iter().any(|m| m == "kinds");
	if all_kinds || kinds.is_empty() {
		let usual: Vec<&str> = if s.vocab { KINDS_365.to_vec() } else { KINDS.to_vec() };
		for k in usual {
			if !kinds.iter().any(|x| x == k) {
				kinds.push(k.to_string());
			}
		}
	}
	let folded = if all_kinds || seen == 0 {
		0
	} else {
		(if s.vocab { KINDS_365.len() } else { KINDS.len() }).saturating_sub(seen)
	};
	let mut kind_chips: Vec<(String, bool)> = vec![("any".into(), e.kinds.is_empty())];
	kind_chips.extend(kinds.iter().map(|k| (k.clone(), e.kinds.contains(k))));
	let tools = [
		("pick", "p", Tool::Pick, "tool-pick"),
		("ignore area", "a", Tool::Ignore, "tool-ignore"),
		("only area", "A", Tool::Only, "tool-only"),
	];
	let tool_row = row().gap(4.0).flex_wrap().w_full().children(tools.iter().map(|(l, key, t, k)| {
		let t = *t;
		hint(button(l, s.tool == t, move |s: &mut App| s.set_tool(t)).key(*k).gap(6.0), key)
	}));
	let line_of = |label: &str, control: El| {
		row()
			.w_full()
			.gap(8.0)
			.center_y()
			.child(text(label.to_string()).font(mono(12.0)).color(dim()).w(64.0).min_w(64.0))
			.child(control)
	};
	let mut c = col()
		.w_full()
		.gap(10.0)
		.child(line_of(
			"keep",
			keyed_chips("keep", &keep_chips, move |s: &mut App, i| {
				let k = keeps[i].1;
				s.change_edit(|e| e.keep = k);
			}),
		))
		.child(line_of("detector", keyed_chips("model", &models, |s: &mut App, i| s.set_detector(i, s.vocab))))
		.child(line_of("knows", keyed_chips("vocab", &vocabs, |s: &mut App, i| s.set_detector(s.model, i == 1))))
		.child(el().h(4.0))
		.child(header("kinds"))
		.child(chips(&kind_chips, move |s: &mut App, i| {
			if i == 0 {
				s.change_edit(|e| e.kinds.clear());
			} else {
				let k = kinds[i - 1].clone();
				s.change_edit(|e| match e.kinds.iter().position(|x| *x == k) {
					Some(at) => {
						e.kinds.remove(at);
					}
					None => e.kinds.push(k),
				});
			}
		}));
	if folded > 0 {
		c = c.child(
			el().key("more-kinds")
				.h(ROW)
				.center_y()
				.focusable(true)
				.role("button")
				.on_press(|s: &mut App| s.more.push("kinds".into()))
				.child(text(format!("+ {folded} others")).font(mono(11.0)).color(mute())),
		);
	}
	c = c.child(el().h(4.0)).child(header("tools")).child(tool_row);
	// every pick, area and stretch, one to a row
	let remove = |key: String, f: Box<dyn Fn(&mut Edit)>| {
		button("remove", false, move |s: &mut App| s.change_edit(|e| f(e))).key(key)
	};
	let seek_row = |t: f64, words: String, colour: Color| {
		el().grow(1.0)
			.h(ROW)
			.center_y()
			.focusable(true)
			.role("button")
			.hover(raised())
			.on_press(move |s: &mut App| {
				s.scrub = Some(t);
				s.send(Cmd::Seek(t));
			})
			.child(text(words).font(digits(12.0)).color(colour))
	};
	if !e.picks.is_empty() {
		c = c.child(el().h(4.0)).child(header("picked"));
		for (i, p) in e.picks.iter().enumerate() {
			c = c.child(
				row()
					.w_full()
					.gap(6.0)
					.center_y()
					.child(seek_row(
						p.t,
						format!("{}  at {:.0}%, {:.0}%", clock(p.t), p.x * 100.0, p.y * 100.0),
						accent(),
					))
					.child(remove(
						format!("remove-pick-{i}"),
						Box::new(move |e: &mut Edit| {
							if i < e.picks.len() {
								e.picks.remove(i);
							}
						}),
					)),
			);
		}
	}
	// things taken out, named from the timeline where it has them
	if !e.drops.is_empty() {
		c = c.child(el().h(4.0)).child(header("taken out"));
		for (i, p) in e.drops.iter().enumerate() {
			let named = s
				.strip
				.as_ref()
				.and_then(|st| st.things.iter().find(|sp| sp.dropped && (sp.pick == *p || sp.drops.contains(p))));
			let words = match named {
				Some(sp) => format!("{} {} – {}", sp.cls, clock(sp.from), clock(sp.to)),
				None => format!("{}  at {:.0}%, {:.0}%", clock(p.t), p.x * 100.0, p.y * 100.0),
			};
			c = c.child(
				row().w_full().gap(6.0).center_y().child(seek_row(p.t, words, red())).child(
					button("bring back", false, move |s: &mut App| {
						s.change_edit(|e| {
							if i < e.drops.len() {
								e.drops.remove(i);
							}
						})
					})
					.key(format!("undrop-{i}")),
				),
			);
		}
	}
	if !e.zones.is_empty() {
		c = c.child(el().h(4.0)).child(header("areas"));
		for (i, z) in e.zones.iter().enumerate() {
			let words = format!(
				"{} {:.0},{:.0} {:.0}x{:.0}",
				if z.mode == ZoneMode::Only { "only  " } else { "ignore" },
				z.x * 100.0,
				z.y * 100.0,
				z.w * 100.0,
				z.h * 100.0
			);
			c = c.child(
				row().w_full().gap(6.0).center_y().child(text(words).font(digits(12.0)).color(ink()).grow(1.0)).child(
					remove(
						format!("remove-zone-{i}"),
						Box::new(move |e: &mut Edit| {
							if i < e.zones.len() {
								e.zones.remove(i);
							}
						}),
					),
				),
			);
		}
	}
	// stretches of time: start marking at the playhead, end where it's got to (x, b and u do the same)
	let mark_button = |label: &str, key: &str, mode: RangeMode| {
		let marking = s.marking.is_some_and(|(_, m)| m == mode);
		let what = if marking { format!("end {label} here") } else { format!("{label} from here") };
		hint(button(&what, marking, move |s: &mut App| s.mark(mode)).gap(6.0), key)
	};
	let mixing = !s.look.mix.trim().is_empty();
	c = c.child(el().h(4.0)).child(header("time")).child(
		row()
			.gap(4.0)
			.flex_wrap()
			.w_full()
			.child(mark_button("cut", "x", RangeMode::Cut))
			.child(mark_button("bare", "b", RangeMode::Bare))
			// the look alone, without the one mixed in
			.children(mixing.then(|| mark_button("unmix", "u", RangeMode::Unmix))),
	);
	for (i, r) in e.ranges.iter().enumerate() {
		let (word, colour) = match r.mode {
			RangeMode::Cut => ("cut  ", red()),
			RangeMode::Bare => ("bare ", ink()),
			RangeMode::Unmix => ("unmix", aqua()),
		};
		c = c.child(
			row()
				.w_full()
				.gap(6.0)
				.center_y()
				.child(seek_row(r.from, format!("{word} {} – {}", clock(r.from), clock(r.to)), colour))
				.child(remove(
					format!("remove-range-{i}"),
					Box::new(move |e: &mut Edit| {
						if i < e.ranges.len() {
							e.ranges.remove(i);
						}
					}),
				)),
		);
	}
	c
}
