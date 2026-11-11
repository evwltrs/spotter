//! The timeline, the window's whole width: a ruler with in and out, the shots the look ahead found
//! (dim past where it's got to), each thing's life as a bar in its lane (yellow when picked), and the
//! stretches cut or left bare, with play, sound and the time under them. Press or drag anywhere to go
//! there; press a thing's bar to pick it (again to let it go); drag in the bottom band to mark a stretch,
//! of the kind the switch beside it says; press a stretch to select it, delete to remove it.

use crate::*;
use rune::{Shape, canvas};
use spotter_core::counted;
use worker::Span;

/// the bands, top to bottom, in points
const RULER: f32 = 16.0;
const SHOTS: f32 = 12.0;
const LANE: f32 = 5.0;
const LANES: usize = 6;
const STRETCHES: f32 = 14.0;
const GAP: f32 = 4.0;
/// where each band starts
const SHOTS_Y: f32 = RULER + GAP;
const THINGS_Y: f32 = SHOTS_Y + SHOTS + GAP;
const THINGS_H: f32 = LANES as f32 * (LANE + 1.0);
const STRETCH_Y: f32 = THINGS_Y + THINGS_H + GAP;
const TRACK: f32 = STRETCH_Y + STRETCHES;
/// the gutter of names on the left
const GUTTER: f32 = 64.0;
pub(crate) const TIMELINE: f32 = TRACK + 8.0 + 28.0;

/// Which band a height on the track is in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Band {
	Ruler,
	Things,
	Stretches,
}

fn band(y: f32) -> Band {
	if y >= STRETCH_Y - GAP / 2.0 {
		Band::Stretches
	} else if y >= THINGS_Y - GAP / 2.0 {
		Band::Things
	} else {
		Band::Ruler
	}
}

/// Whether this thing is picked: from its bar, or on the picture where the look ahead says it was.
fn picked(e: &Edit, sp: &Span) -> bool {
	e.picks.iter().any(|p| *p == sp.pick || sp.picks.contains(p))
}

/// The thing under a place on the track, if any.
fn bar_at(s: &App, t: f64, y: f32) -> Option<Span> {
	let strip = s.strip.as_ref()?;
	let lane = ((y - THINGS_Y) / (LANE + 1.0)).floor();
	if lane < 0.0 || lane as usize >= LANES {
		return None;
	}
	strip.things.iter().find(|sp| sp.lane == lane as usize && t >= sp.from - 0.05 && t <= sp.to + 0.05).cloned()
}

/// The stretch under a time, if any.
fn range_at(e: &Edit, t: f64) -> Option<usize> {
	e.ranges.iter().position(|r| t >= r.from && t <= r.to)
}

pub(crate) fn timeline(s: &App, width: Option<f32>) -> El {
	let d = s.st.duration.max(0.001);
	let x = |t: f64| ((t / d).clamp(0.0, 1.0) * 1000.0) as f32;
	let e = &s.clip_edit;
	let mut shape = Shape::new();
	let rect = |sh: Shape, a: f32, b: f32, y: f32, h: f32, c: Color| {
		sh.fill(vec![[a, y], [b.max(a + 0.6), y], [b.max(a + 0.6), y + h], [a, y + h]], c)
	};
	// the ruler: a tick each second, a longer one every five
	let step = if d > 120.0 {
		10.0
	} else if d > 40.0 {
		5.0
	} else {
		1.0
	};
	let mut t = 0.0;
	while t <= d {
		let big = (t / step).round() as i64 % 5 == 0;
		shape = shape.line(
			[x(t), RULER - if big { 7.0 } else { 3.0 }],
			[x(t), RULER],
			1.0,
			rune::LineCap::Butt,
			if big { dim() } else { faint() },
		);
		t += step;
	}
	// the music's beats, as small ticks under the ruler's
	if let Some(strip) = &s.strip {
		for &b in strip.beats.iter().filter(|&&b| b < d) {
			shape = shape.line([x(b), RULER + 1.0], [x(b), RULER + GAP - 0.5], 1.0, rune::LineCap::Butt, mute());
		}
	}
	// the shots, alternating, and the part not looked at yet
	match &s.strip {
		Some(strip) => {
			let mut edges: Vec<f64> = vec![0.0];
			edges.extend(strip.cuts.iter().copied().filter(|&c| c > 0.0 && c < d));
			edges.push(d);
			for (i, w) in edges.windows(2).enumerate() {
				shape =
					rect(shape, x(w[0]), x(w[1]) - 1.0, SHOTS_Y, SHOTS, if i % 2 == 0 { surface() } else { raised() });
			}
			if strip.until < d - 0.05 {
				shape = rect(shape, x(strip.until.max(0.0)), 1000.0, SHOTS_Y, SHOTS + GAP + THINGS_H, bg().alpha(0.7));
			}
			for sp in strip.things.iter().filter(|sp| sp.lane < LANES) {
				let y = THINGS_Y + sp.lane as f32 * (LANE + 1.0);
				if sp.dropped {
					// taken out: red, struck through
					shape = rect(shape, x(sp.from), x(sp.to), y, LANE, red().alpha(0.35));
					shape = shape.line(
						[x(sp.from), y + LANE / 2.0],
						[x(sp.to).max(x(sp.from) + 0.6), y + LANE / 2.0],
						1.0,
						rune::LineCap::Butt,
						red(),
					);
				} else {
					shape = rect(
						shape,
						x(sp.from),
						x(sp.to),
						y,
						LANE,
						if picked(e, sp) { accent() } else { aqua().alpha(0.8) },
					);
				}
			}
		}
		None => shape = rect(shape, 0.0, 1000.0, SHOTS_Y, SHOTS, surface()),
	}
	// stretches: cut red, bare dim, the selected one outlined, the one being marked yellow
	shape = rect(shape, 0.0, 1000.0, STRETCH_Y, STRETCHES, surface());
	for (i, r) in e.ranges.iter().enumerate() {
		let c = match r.mode {
			RangeMode::Cut => red().alpha(0.75),
			RangeMode::Bare => dim().alpha(0.45),
			RangeMode::Unmix => aqua().alpha(0.55),
		};
		shape = rect(shape, x(r.from), x(r.to), STRETCH_Y, STRETCHES, c);
		if s.selected_range == Some(i) {
			shape = shape.stroke_with(
				vec![
					[x(r.from), STRETCH_Y],
					[x(r.to), STRETCH_Y],
					[x(r.to), STRETCH_Y + STRETCHES],
					[x(r.from), STRETCH_Y + STRETCHES],
				],
				true,
				1.0,
				rune::LineCap::Butt,
				ink(),
			);
		}
	}
	if let Some((from, to)) = s.marking.map(|(f, _)| (f, s.position())).or(s.drafting) {
		shape = rect(shape, x(from.min(to)), x(from.max(to)), STRETCH_Y, STRETCHES, accent().alpha(0.5));
	}
	// in and out: outside them dimmed through every band, brackets on the ruler
	if let Some(a) = s.trim_in {
		shape = rect(shape, 0.0, x(a), 0.0, TRACK, bg().alpha(0.6)).line(
			[x(a), 2.0],
			[x(a), TRACK],
			1.0,
			rune::LineCap::Butt,
			ink(),
		);
	}
	if let Some(b) = s.trim_out {
		shape = rect(shape, x(b), 1000.0, 0.0, TRACK, bg().alpha(0.6)).line(
			[x(b), 2.0],
			[x(b), TRACK],
			1.0,
			rune::LineCap::Butt,
			ink(),
		);
	}
	// an export's progress, a thin line along the ruler
	if let Some((n, total)) = s.st.export {
		let (a, b) = (s.trim_in.unwrap_or(0.0), s.trim_out.unwrap_or(d));
		shape = rect(shape, x(a), x(a + (b - a) * n as f64 / total.max(1) as f64), RULER - 2.0, 2.0, accent());
	}
	let at = (s.position() / d).clamp(0.0, 1.0) as f32;
	let mut track = el()
		.key("tl")
		.grow(1.0)
		.min_w(0.0)
		.h(TRACK)
		.role("slider")
		.label("timeline")
		.child(canvas(shape.view_box(1000.0, TRACK)).w_full().h(TRACK).absolute(0.0, 0.0))
		// the playhead, its own element so playing doesn't redraw the rest
		.child(
			row()
				.w_full()
				.h(TRACK)
				.absolute(0.0, 0.0)
				.child(el().w_frac(at).h_full())
				.child(el().w(1.0).h_full().fill(accent())),
		)
		.on_drag(|s: &mut App, d: &mut Drag, tx: &mut Tx| {
			s.drag_timeline(d);
			tx.instant();
		})
		.on_pointer_move_local(move |s: &mut App, p: rune::PointerEvent, _: &mut Tx| s.hover_timeline(p.x, p.y, width))
		// right-click a thing's bar: take it out (or bring it back)
		.on_secondary_press_local(move |s: &mut App, p: rune::PointerEvent, _: &mut Tx| {
			if let Some(w) = width
				&& band(p.y) == Band::Things
				&& let Some(sp) = bar_at(s, s.time_at(p.x, w), p.y)
			{
				s.toggle_drop(&sp);
				s.hover_timeline(p.x, p.y, width);
			}
		})
		.on_hover_change(|s: &mut App, inside: bool, _: &mut Tx| {
			if !inside {
				s.hovering = None;
				s.hover_bar = None;
			}
		});
	// time labels along the ruler, where the track's width is known
	if let Some(w) = width {
		let every = if d > 120.0 {
			30.0
		} else if d > 40.0 {
			10.0
		} else if d > 12.0 {
			2.0
		} else {
			1.0
		};
		let mut t = 0.0;
		while t < d - every * 0.3 {
			let px = (t / d) as f32 * w;
			if px + 34.0 < w {
				track = track.child(
					text(clock(t).trim_end_matches(".0").to_string())
						.font(digits(10.0))
						.color(mute())
						.absolute(px + 3.0, 0.0),
				);
			}
			t += every;
		}
	}
	let strip = s.strip.as_ref();
	let more = strip.map_or(0, |st| st.things.iter().filter(|sp| sp.lane >= LANES).count());
	let name = |label: &str, y: f32, h: f32| {
		el().h(h).w_full().center_y().absolute(0.0, y).child(text(label.to_string()).font(mono(10.0)).color(mute()))
	};
	// what a drag in the bottom band marks: pressed, the next kind (unmix only with a look mixed in);
	// with a stretch selected, that stretch becomes it too
	let label = match s.range_mode {
		RangeMode::Cut => "cut",
		RangeMode::Bare => "bare",
		RangeMode::Unmix => "unmix",
	};
	let switch = row().absolute(0.0, STRETCH_Y - 1.0).child(
		el().key("tl-mode")
			.px(4.0)
			.h(STRETCHES + 2.0)
			.center()
			.fill(faint())
			.focusable(true)
			.role("button")
			.help("what dragging in this band marks: cut (taken out), bare (nothing drawn), unmix (without the look mixed in); press to change it, and the selected stretch with it")
			.on_press(|s: &mut App| {
				let mixing = !s.look.mix.trim().is_empty();
				let next = match s.range_mode {
					RangeMode::Cut => RangeMode::Bare,
					RangeMode::Bare if mixing => RangeMode::Unmix,
					_ => RangeMode::Cut,
				};
				s.range_mode = next;
				if let Some(i) = s.selected_range {
					s.change_edit(|e| {
						if let Some(r) = e.ranges.get_mut(i) {
							r.mode = next;
						}
					});
				}
			})
			.child(text(format!("{label} ›")).font(mono(10.0)).color(match s.range_mode {
				RangeMode::Cut => red(),
				RangeMode::Bare => ink(),
				RangeMode::Unmix => aqua(),
			})),
	);
	let gutter = el()
		.w(GUTTER)
		.min_w(GUTTER)
		.h(TRACK)
		.child(name("shots", SHOTS_Y, SHOTS))
		.child(name(if more > 0 { "things +" } else { "things" }, THINGS_Y, LANE * 2.0))
		.children((more > 0).then(|| name(&format!("{more} more"), THINGS_Y + LANE * 2.0 + 2.0, LANE * 2.0)))
		.child(switch);
	let summary = match strip {
		Some(st) => format!("{} {}", counted(st.cuts.len() + 1, "shot"), counted(st.things.len(), "thing")),
		None => String::new(),
	};
	col().key("timeline")
		.w_full()
		.h(TIMELINE)
		.fill(bg())
		.border_sides([1.0, 0.0, 0.0, 0.0], line())
		.px(10.0)
		.pt(6.0)
		.child(row().w_full().gap(8.0).child(gutter).child(track))
		.child(
			row().w_full()
				.h(28.0)
				.gap(10.0)
				.center_y()
				.child(button(if s.st.playing { "pause" } else { "play" }, false, |s: &mut App| s.send(Cmd::Toggle)).key("play"))
				.child(button(if s.muted { "sound off" } else { "sound on" }, false, |s: &mut App| {
					s.muted = !s.muted;
					s.send(Cmd::Mute(s.muted));
				}))
				// how loud the preview plays (not the export: that's the music's levels)
				.child(
					row().w(150.0)
						.gap(6.0)
						.center_y()
						.help("how loud the preview plays (- and = to turn it down and up); the export's own levels are under music in the export tab")
						.child(slider_by("volume".into(), if s.muted { 0.0 } else { s.volume }, 0.0, 1.5, 0.05, Some(1.0), |s: &App| s.volume, |s: &mut App, v| s.set_volume(v)))
						.child(text(format!("{:>3}%", (if s.muted { 0.0 } else { s.volume } * 100.0).round() as i32)).font(digits(11.0)).color(dim()).w(36.0)),
				)
				.child(text(format!("{} / {}", clock(s.position()), clock(s.st.duration))).font(digits(12.0)).color(ink()))
				.child(spacer())
				.child(text(summary).font(mono(11.0)).color(mute())),
		)
}

impl App {
	fn time_at(&self, px: f32, w: f32) -> f64 {
		(px / w.max(1.0)).clamp(0.0, 1.0) as f64 * self.st.duration
	}

	/// A time near a cut in the footage or the playhead (within 6 points) goes to it.
	fn snap(&self, t: f64, w: f32) -> f64 {
		let reach = 6.0 / w.max(1.0) as f64 * self.st.duration;
		let mut best = (reach, t);
		for c in self.strip_cuts().into_iter().chain(std::iter::once(self.st.t)) {
			if (c - t).abs() < best.0 {
				best = ((c - t).abs(), c);
			}
		}
		best.1
	}

	/// A press or drag on the track: where it starts says what it does (see the top of this file).
	fn drag_timeline(&mut self, d: &Drag) {
		let t = self.time_at(d.x, d.w);
		let start = self.time_at(d.start_x, d.w);
		let moved = (d.x - d.start_x).abs() > 3.0;
		match band(d.start_y) {
			Band::Stretches => match d.phase {
				Phase::Down => {
					// on a stretch's edge: move that edge; otherwise select what's there, or start a new one
					let near = 5.0 / d.w.max(1.0) as f64 * self.st.duration;
					self.grab = self.clip_edit.ranges.iter().enumerate().find_map(|(i, r)| {
						if (r.from - t).abs() < near {
							Some((i, true))
						} else if (r.to - t).abs() < near {
							Some((i, false))
						} else {
							None
						}
					});
					self.selected_range = self.grab.map(|g| g.0).or_else(|| range_at(&self.clip_edit, t));
					self.drafting = None;
				}
				Phase::Move if self.grab.is_some() => {
					let (i, from) = self.grab.unwrap_or_default();
					let t = self.snap(t, d.w);
					self.change_edit(|e| {
						if let Some(r) = e.ranges.get_mut(i) {
							if from {
								r.from = t.min(r.to - 0.05).max(0.0);
							} else {
								r.to = t.max(r.from + 0.05);
							}
						}
					});
				}
				Phase::Move if moved => self.drafting = Some((self.snap(start, d.w), self.snap(t, d.w))),
				Phase::Move => {}
				Phase::Up => {
					self.grab = None;
					if let Some((a, b)) = self.drafting.take()
						&& (b - a).abs() > 0.05
						&& !d.cancelled()
					{
						let mode = self.range_mode;
						self.change_edit(|e| e.add_range(a.min(b), a.max(b), mode));
						self.selected_range = None;
					}
				}
			},
			Band::Things if !moved && d.phase == Phase::Up => {
				match bar_at(self, start, d.start_y) {
					// a thing taken out: pressed, it's back
					Some(sp) if sp.dropped => {
						self.toggle_drop(&sp);
						self.hover_timeline(d.start_x, d.start_y, Some(d.w));
					}
					Some(sp) => {
						self.toggle_pick(&sp);
						// the line under the pointer says what pressing does now
						self.hover_timeline(d.start_x, d.start_y, Some(d.w));
					}
					None => self.seek(t),
				}
			}
			_ => {
				if d.phase != Phase::Up {
					self.seek(self.snap(t, d.w));
				}
			}
		}
	}

	/// Take a thing out (a false positive, say), or bring it back if it's out. Taking it out lets go
	/// of any pick on it; with none left picked, everything gets the look again.
	pub(crate) fn toggle_drop(&mut self, sp: &Span) {
		let sp = sp.clone();
		if sp.dropped {
			self.change_edit(|e| e.drops.retain(|p| *p != sp.pick && !sp.drops.contains(p)));
			self.note = Some(format!("{} is back", sp.cls));
		} else {
			self.change_edit(|e| {
				e.drops.push(sp.pick);
				e.picks.retain(|p| *p != sp.pick && !sp.picks.contains(p));
				if e.picks.is_empty() && e.keep == Keep::Picked {
					e.keep = Keep::All;
				}
			});
			self.note = Some(format!(
				"{} {} – {} taken out: right-click its bar (or press it) to bring it back",
				sp.cls,
				clock(sp.from),
				clock(sp.to)
			));
		}
	}

	/// Pick a thing from its bar, or let it go if it's picked; with none left picked, everything
	/// gets the look again.
	fn toggle_pick(&mut self, sp: &Span) {
		let sp = sp.clone();
		if picked(&self.clip_edit, &sp) {
			self.change_edit(|e| {
				e.picks.retain(|p| *p != sp.pick && !sp.picks.contains(p));
				if e.picks.is_empty() && e.keep == Keep::Picked {
					e.keep = Keep::All;
				}
			});
		} else {
			self.change_edit(|e| {
				e.picks.push(sp.pick);
				e.keep = Keep::Picked;
			});
		}
	}

	/// What's under the pointer, for the status line.
	fn hover_timeline(&mut self, px: f32, py: f32, width: Option<f32>) {
		let Some(w) = width else { return };
		let t = self.time_at(px, w);
		self.hover_bar = if band(py) == Band::Things { bar_at(self, t, py) } else { None };
		self.hovering = Some(match band(py) {
			Band::Things => match bar_at(self, t, py) {
				Some(sp) if sp.dropped => format!(
					"{} · {} – {} · taken out · press, right-click or delete to bring it back",
					sp.cls,
					clock(sp.from),
					clock(sp.to)
				),
				Some(sp) => {
					let how = if picked(&self.clip_edit, &sp) {
						"picked · press to let it go"
					} else if self.clip_edit.keep == Keep::Picked && !self.clip_edit.picks.is_empty() {
						"press to pick it too"
					} else {
						"press to pick it: then only picked things get the look"
					};
					format!("{} · {} – {} · {:.0}% sure · {how}", sp.cls, clock(sp.from), clock(sp.to), sp.conf * 100.0)
				}
				None => clock(t),
			},
			Band::Stretches => match range_at(&self.clip_edit, t) {
				Some(i) => {
					let r = &self.clip_edit.ranges[i];
					let word = match r.mode {
						RangeMode::Cut => "cut".to_string(),
						RangeMode::Bare => "bare".to_string(),
						RangeMode::Unmix => format!("without {}", self.look.mix.trim()),
					};
					format!(
						"{word} {} – {} · press, then delete to remove (or the switch on the left to change it)",
						clock(r.from),
						clock(r.to)
					)
				}
				None => format!(
					"{} · drag to mark a {} stretch (the switch on the left says which); drag a stretch's edge to move it",
					clock(t),
					match self.range_mode {
						RangeMode::Cut => "cut",
						RangeMode::Bare => "bare",
						RangeMode::Unmix => "unmix",
					}
				),
			},
			Band::Ruler => clock(t),
		});
	}
}
