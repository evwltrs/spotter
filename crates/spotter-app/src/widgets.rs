//! The window's small parts: words to press, sets of options, toggles, sliders, headers, tabs.

use crate::*;

pub(crate) fn clock(t: f64) -> String {
	let t = t.max(0.0);
	format!("{}:{:02}.{}", (t / 60.0).floor() as i64, (t % 60.0).floor() as i64, ((t % 1.0) * 10.0).floor() as i64)
}

/// A word you press: plain words on a raised strip, lit when hovered; `on` (a tool in use, a stretch
/// being marked) holds it lit.
pub(crate) fn button<S: 'static>(label: &str, on: bool, f: impl Fn(&mut S) + 'static) -> El {
	el().px(8.0)
		.h(ROW)
		.center()
		.fill(if on { faint() } else { raised() })
		.hover(faint())
		.focusable(true)
		.role("button")
		.label(label)
		.on_press(f)
		.child(text(label).font(mono(12.0)).color(ink()))
}

/// The one button in a panel that matters most (export): solid yellow.
pub(crate) fn primary<S: 'static>(label: &str, f: impl Fn(&mut S) + 'static) -> El {
	el().px(12.0)
		.h(ROW + 4.0)
		.center()
		.fill(accent())
		.hover(hex("#fcd15c"))
		.focusable(true)
		.role("button")
		.label(label)
		.on_press(f)
		.child(text(label).font(strong(12.0)).color(bg()))
}

/// A group's or section's name, dim, with a hairline running on to the right edge.
pub(crate) fn header(label: &str) -> El {
	row()
		.w_full()
		.gap(8.0)
		.center_y()
		.h(ROW)
		.child(text(label).font(mono(11.0)).color(dim()))
		.child(el().grow(1.0).h(1.0).fill(line()))
}

/// A set of options of which one (or, for `chips`, several) is chosen, on one raised strip that wraps:
/// the chosen ones in ink on a lighter block, the rest dim; hovered words turn ink.
fn option_cells(prefix: &str, options: &[(String, bool)], f: impl Fn(&mut App, usize) + Clone + 'static) -> El {
	let cells: Vec<El> = options
		.iter()
		.enumerate()
		.map(|(i, (label, on))| {
			let f = f.clone();
			let cell = el()
				.px(7.0)
				.h(ROW)
				.center()
				.fill(if *on { faint() } else { raised() })
				.hover(if *on { faint() } else { line() })
				.focusable(true)
				.role("button")
				.label(label.clone())
				.on_press(move |s: &mut App| f(s, i))
				.child(text(label.clone()).font(mono(12.0)).color(if *on { ink() } else { dim() }));
			if prefix.is_empty() { cell } else { cell.key(format!("{prefix}-{i}")) }
		})
		.collect();
	row().w_full().gap(1.0).flex_wrap().children(cells)
}

pub(crate) fn chips(options: &[(String, bool)], f: impl Fn(&mut App, usize) + Clone + 'static) -> El {
	option_cells("", options, f)
}

/// Options with keys (prefix-0, prefix-1…), for scripts to press.
pub(crate) fn keyed_chips(
	prefix: &str,
	options: &[(String, bool)],
	f: impl Fn(&mut App, usize) + Clone + 'static,
) -> El {
	option_cells(prefix, options, f)
}

/// Off or on, as a pair of options, as wide as its words.
pub(crate) fn toggle(key: &str, on: bool, f: impl Fn(&mut App, bool) + Clone + 'static) -> El {
	let cell = |label: &str, chosen: bool, to: bool| {
		let f = f.clone();
		el().px(7.0)
			.h(ROW)
			.center()
			.fill(if chosen { faint() } else { raised() })
			.hover(if chosen { faint() } else { line() })
			.focusable(true)
			.role("button")
			.label(label.to_string())
			.on_press(move |s: &mut App| f(s, to))
			.child(text(label.to_string()).font(mono(12.0)).color(if chosen { ink() } else { dim() }))
	};
	row().key(key.to_string()).gap(1.0).child(cell("off", !on, false)).child(cell("on", on, true))
}

/// A look in the rail: yellow when it's the look, aqua when it's mixed in; pressed to pick it (shift
/// to mix it in), right-clicked to mix it in.
pub(crate) fn look_button(
	name: &str,
	current: bool,
	changed: bool,
	mixed: bool,
	press: impl Fn(&mut App, &mut Tx) + 'static,
	mix: impl Fn(&mut App) + 'static,
) -> El {
	let fill = if current {
		accent()
	} else if mixed {
		aqua()
	} else {
		surface()
	};
	let ink_on = if current || mixed { bg() } else { ink() };
	el().key(format!("preset-{name}"))
		.px(8.0)
		.h(ROW)
		.w_full()
		.center_y()
		.gap(4.0)
		.fill(fill)
		.hover(if current || mixed { fill } else { raised() })
		.focusable(true)
		.role("button")
		.label(name)
		.help("press to use; right-click (or shift-click) to mix in with the look")
		.on_press_tx(press)
		.on_secondary_press(move |s: &mut App, _, _| mix(s))
		.children(mixed.then(|| text("+").font(mono(12.0)).color(bg())))
		.child(text(name).font(mono(12.0)).color(ink_on).truncate().min_w(0.0))
		.children(changed.then(|| text("*").font(mono(12.0)).color(if current { bg() } else { orange() })))
}

/// Words for tabs: the current one in ink with a yellow underline.
pub(crate) fn tabs(options: &[(&str, bool, &str)], f: impl Fn(&mut App, usize) + Clone + 'static) -> El {
	row().gap(16.0).h(ROW + 6.0).children(options.iter().enumerate().map(|(i, (label, on, key))| {
		let f = f.clone();
		col()
			.key(key.to_string())
			.focusable(true)
			.role("tab")
			.label(label.to_string())
			.on_press(move |s: &mut App| f(s, i))
			.child(el().h(ROW).center_y().child(text(label.to_string()).font(mono(12.0)).color(if *on {
				ink()
			} else {
				mute()
			})))
			.child(el().w_full().h(2.0).fill(if *on { accent() } else { Color::TRANSPARENT }))
	}))
}

/// A slider as wide as it's given: a thick track filled to the value (orange when it's moved from the
/// look's own value, which a tick marks), dragged, pressed along to jump, or stepped with the arrows.
pub(crate) fn slider(key: String, v: f32, min: f32, max: f32, step: f32, look_at: Option<f32>) -> El {
	let (k1, k2) = (key.clone(), key.clone());
	slider_by(
		key,
		v,
		min,
		max,
		step,
		look_at,
		move |s: &App| s.shown_look().get(&k1).and_then(|v| v.as_f64()).unwrap_or(min as f64) as f32,
		move |s: &mut App, x| s.set(&k2, serde_json::json!(x)),
	)
}

/// A slider for anything: `get` reads its value, `set` sets it.
pub(crate) fn slider_by(
	key: String,
	v: f32,
	min: f32,
	max: f32,
	step: f32,
	look_at: Option<f32>,
	get: impl Fn(&App) -> f32 + Clone + 'static,
	set: impl Fn(&mut App, f32) + Clone + 'static,
) -> El {
	let h = ROW;
	let span = (max - min).max(1e-6);
	let at = ((v - min) / span).clamp(0.0, 1.0);
	let snap = move |x: f32, w: f32| {
		let raw = min + (x / w.max(1.0)).clamp(0.0, 1.0) * span;
		((raw / step).round() * step).clamp(min, max)
	};
	let changed = look_at.is_some_and(|l| (l - v).abs() > step * 0.5);
	let (get2, set2) = (get.clone(), set.clone());
	let mut e = el()
		.key(format!("slider-{key}"))
		.role("slider")
		.label(key.clone())
		.focusable(true)
		.range_value(v as f64, min as f64, max as f64)
		.range_step(step as f64)
		.on_step(move |s: &mut App, steps, _: &mut Tx| {
			let next = ((get(s) + steps as f32 * step) / step).round() * step;
			set(s, next.clamp(min, max));
		})
		.grow(1.0)
		.min_w(60.0)
		.h(h)
		.on_drag(move |s: &mut App, d: &mut Drag, tx: &mut Tx| match d.phase {
			Phase::Down => {
				d.data = get2(s);
				set2(s, snap(d.x, d.w));
			}
			Phase::Move => {
				set2(s, snap(d.x, d.w));
				tx.instant();
			}
			Phase::Up => {
				if d.cancelled() {
					set2(s, d.data);
				}
			}
		})
		.child(el().w_full().h(6.0).absolute(0.0, (h - 6.0) / 2.0).fill(line()))
		.child(el().w_frac(at.max(0.004)).h(6.0).absolute(0.0, (h - 6.0) / 2.0).fill(if changed {
			orange()
		} else {
			dim()
		}));
	// the look's own value, as a tick
	if let Some(l) = look_at {
		let f = ((l - min) / span).clamp(0.0, 1.0);
		e = e.child(
			row()
				.w_full()
				.h(10.0)
				.absolute(0.0, (h - 10.0) / 2.0)
				.child(el().w_frac(f).h_full())
				.child(el().w(1.0).h_full().fill(ink())),
		);
	}
	e
}
