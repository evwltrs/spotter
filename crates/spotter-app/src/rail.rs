//! The looks, down the left: in families, two to a row, always in the same places, the current one
//! yellow (with a * once it's been changed), one mixed in with it aqua. Right-click (or shift-click)
//! a look to mix it in, again to take it out. j and k step through them in this order.

use crate::*;

/// the looks by family; one not listed here falls into "other"
pub(crate) const FAMILIES: &[(&str, &[&str])] = &[
	(
		"overlay",
		&[
			"smoke",
			"cctv",
			"panopticon",
			"ember",
			"night watch",
			"blacksite",
			"sprawl",
			"census",
			"manifest",
			"lock-on",
			"wake",
			"flow",
			"broadcast",
			"network",
			"radar",
			"survey",
			"telemetry",
			"stars",
			"captions",
		],
	),
	("cutout", &["isolate", "redact", "heat", "spotlight", "neon", "infrared", "x-ray"]),
	("time", &["marey", "beat copies", "stroboscope", "exposure", "light trails", "phantom", "extract", "slitscan"]),
	("type", &["teletype", "terminal", "words", "words flick", "words beat", "transcript"]),
	("print", &["print", "pop art", "riso", "1-bit", "blueprint", "sketch"]),
	(
		"glitch",
		&[
			"real mosh",
			"mosh beats",
			"datamosh",
			"bloom",
			"crush",
			"corrupt",
			"blocks",
			"vhs",
			"vapor",
			"trichrome",
			"kaleido",
			"skip",
			"databend",
			"glitch",
			"melt",
		],
	),
	("depth", &["rack focus", "sounding", "contours", "haze"]),
	("splats", &["splats", "needles", "splat rot", "slip", "burst", "splatmosh"]),
	("signal", &["sonar", "pulse", "night ops", "plates"]),
	("ahead", &["foresight", "trajectory"]),
];

/// Every look, family by family, in the rail's order: yours first, if you've kept any.
pub(crate) fn order(mine: &[(String, Look)]) -> Vec<(&'static str, Vec<String>)> {
	let mut out: Vec<(&'static str, Vec<String>)> = Vec::new();
	if !mine.is_empty() {
		out.push(("mine", mine.iter().map(|(n, _)| n.clone()).collect()));
	}
	let mut listed: Vec<&str> = Vec::new();
	for (family, names) in FAMILIES {
		let have: Vec<&'static str> = Look::presets().filter(|n| names.contains(n)).collect();
		listed.extend(&have);
		// in the family's own order
		let mut have = have;
		have.sort_by_key(|n| names.iter().position(|m| m == n));
		out.push((family, have.into_iter().map(str::to_string).collect()));
	}
	let other: Vec<String> = Look::presets().filter(|n| !listed.contains(n)).map(str::to_string).collect();
	if !other.is_empty() {
		out.push(("other", other));
	}
	out
}

/// The look after (or before, `by` -1) the current one, round the end.
pub(crate) fn step(mine: &[(String, Look)], current: &str, by: i32) -> String {
	let all: Vec<String> = order(mine).into_iter().flat_map(|(_, n)| n).collect();
	let i = all.iter().position(|n| n == current).unwrap_or(0) as i32;
	all[((i + by).rem_euclid(all.len() as i32)) as usize].clone()
}

pub(crate) fn rail(s: &App) -> El {
	let changed = s.base_look(&s.preset).is_some_and(|p| p != s.look);
	let mut c = col().w_full().padding(8.0).gap(1.0);
	for (family, names) in order(&s.mine) {
		c = c.child(el().h(4.0)).child(text(family).font(mono(11.0)).color(dim()).px(8.0).h(18.0).center_y());
		let mut cells: Vec<El> = names
			.iter()
			.map(|name| {
				let (n, m) = (name.clone(), name.clone());
				look_button(
					name,
					s.preset == *name,
					changed && s.preset == *name,
					s.look.mix.trim() == name,
					move |s: &mut App, tx: &mut Tx| s.press_look(&n, tx),
					move |s: &mut App| s.mix_with(&m),
				)
				.w_frac(0.5)
				.h(20.0)
			})
			.collect();
		while !cells.is_empty() {
			let rest = cells.split_off(cells.len().min(2));
			c = c.child(row().w_full().children(std::mem::replace(&mut cells, rest)));
		}
	}
	col().key("rail").w(RAIL).h_full().min_h(0.0).fill(surface()).border_sides([0.0, 1.0, 0.0, 0.0], line()).child(
		col().key("rail-scroll").grow(1.0).min_h(0.0).w_full().scroll(ScrollSpec { stick_bottom: false }).child(c),
	)
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn every_look_has_a_family() {
		let families = order(&[]);
		assert!(families.iter().all(|(f, _)| *f != "other"), "{:?}", families.last());
		let shown: usize = families.iter().map(|(_, n)| n.len()).sum();
		assert_eq!(shown, Look::presets().count());
	}
}
