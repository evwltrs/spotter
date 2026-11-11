//! The export tab: the shape, drawn to proportion; upscaling, the models, the follow cam; in and out;
//! what will be written; and the one yellow button.

use crate::*;
use rune::{Shape, canvas};

/// A shape to export at, drawn as its own proportions in a small cell, its name under it.
fn tile(i: usize, label: &str, ratio: f32, on: bool) -> El {
	let cell = 40.0;
	let (w, h) = if ratio >= 1.0 { (cell, cell / ratio) } else { (cell * ratio, cell) };
	col()
		.key(format!("aspect-{i}"))
		.gap(4.0)
		.items_center()
		.focusable(true)
		.role("button")
		.label(label.to_string())
		.on_press(move |s: &mut App| {
			s.aspect = i;
			s.send(Cmd::Aspect(shape_of(i)));
		})
		.child(
			el().size(cell + 8.0, cell + 8.0)
				.center()
				.fill(if on { faint() } else { raised() })
				.hover(faint())
				.child(el().size(w, h).border(1.0, if on { ink() } else { mute() })),
		)
		.child(
			text(if label == "as shot" { label } else { label.split(' ').next().unwrap_or(label) }.to_string())
				.font(mono(11.0))
				.color(if on { ink() } else { dim() }),
		)
}

/// The song's shape with the part that plays (`out` seconds from `from`, of `len`) picked out in
/// yellow; dragged to move it.
fn song_picker(shape: Option<&[f32]>, from: f64, len: f64, out: f64) -> El {
	let h = 40.0;
	let len = len.max(0.1);
	let x = |t: f64| ((t / len).clamp(0.0, 1.0) * 1000.0) as f32;
	let (x0, x1) = (x(from), x((from + out).min(len)));
	let rect = |sh: Shape, a: f32, b: f32, y: f32, hh: f32, c: Color| {
		sh.fill(vec![[a, y], [b.max(a + 0.6), y], [b.max(a + 0.6), y + hh], [a, y + hh]], c)
	};
	let mut sh = rect(Shape::new(), 0.0, 1000.0, 0.0, h, raised());
	sh = rect(sh, x0, x1, 0.0, h, accent().alpha(0.16));
	match shape {
		Some(peaks) if !peaks.is_empty() => {
			let w = 1000.0 / peaks.len() as f32;
			for (i, p) in peaks.iter().enumerate() {
				let bx = i as f32 * w;
				let bh = (p * (h - 6.0)).max(1.0);
				let inside = bx + w > x0 && bx < x1;
				sh = rect(sh, bx, bx + (w - 0.8).max(0.6), (h - bh) / 2.0, bh, if inside { accent() } else { mute() });
			}
		}
		_ => sh = rect(sh, 0.0, 1000.0, h / 2.0 - 0.5, 1.0, mute()),
	}
	sh = sh.line([x0, 0.0], [x0, h], 1.5, rune::LineCap::Butt, accent()).line(
		[x1, 0.0],
		[x1, h],
		1.5,
		rune::LineCap::Butt,
		accent(),
	);
	el().key("song-picker")
		.w_full()
		.h(h)
		.role("slider")
		.label("the part of the song that plays")
		.child(canvas(sh.view_box(1000.0, h)).w_full().h(h))
}

pub(crate) fn export(s: &App, k: &Kit) -> El {
	let shot = if s.st.w > 0 && s.st.h > 0 && s.aspect == 0 { s.st.w as f32 / s.st.h as f32 } else { 16.0 / 9.0 };
	let tiles = ASPECTS
		.iter()
		.enumerate()
		.map(|(i, (l, size))| tile(i, l, size.map_or(shot, |(w, h)| w as f32 / h as f32), i == s.aspect));
	let line_of = |label: &str, help: &str, control: El| {
		row()
			.w_full()
			.gap(8.0)
			.center_y()
			.child(text(label.to_string()).font(mono(12.0)).color(dim()).w(80.0).min_w(80.0).help(help.to_string()))
			.child(control)
	};
	let models = vec![("quick".to_string(), s.quick), ("best".to_string(), !s.quick)];
	let mut c = col()
		.w_full()
		.gap(10.0)
		.child(header("shape"))
		.child(row().w_full().gap(6.0).flex_wrap().children(tiles))
		.child({
			// the other shapes, to write beside it: a reel for one place and a wide one for another
			let others: Vec<(usize, String, bool)> = ASPECTS.iter().enumerate().filter(|(i, _)| *i != s.aspect).map(|(i, (l, _))| (i, l.split(' ').next().unwrap_or(l).to_string(), s.also.contains(&i))).collect();
			let opts: Vec<(String, bool)> = others.iter().map(|(_, l, on)| (if l == "as" { "as shot".to_string() } else { l.clone() }, *on)).collect();
			let which: Vec<usize> = others.iter().map(|o| o.0).collect();
			row().w_full().gap(8.0).center_y().child(text("also").font(mono(12.0)).color(dim()).w(80.0).min_w(80.0).help("write these shapes too, beside the chosen one, each named after its shape".to_string())).child(keyed_chips("also", &opts, move |s: &mut App, i| {
				let a = which[i];
				match s.also.iter().position(|x| *x == a) {
					Some(at) => {
						s.also.remove(at);
					}
					None => s.also.push(a),
				}
			}))
		})
		.child(el().h(4.0))
		.child(line_of("upscale", "make a picture smaller than the export bigger with Real-ESRGAN on the graphics card, rather than stretching it; slower", toggle("toggle-upscale", s.upscale, |s: &mut App, b| s.upscale = b)))
		.child(line_of("models", "best: SAM 2.1 large for outlines, Depth Anything large, SegFormer B5, a few times slower; quick: the ones the preview uses", keyed_chips("models", &models, |s: &mut App, i| s.quick = i == 0).key("toggle-quick")));
	if s.aspect != 0 {
		c = c.child(line_of(
			"camera",
			"follow cam: the frame moves with what's followed, so a landscape clip becomes this shape around it",
			toggle("toggle-follow", s.look.follow, |s: &mut App, b| s.set("follow", Value::Bool(b))),
		));
	}
	// in and out, set at the playhead (or with i and o)
	let mark = |label: &'static str, t: Option<f64>, set: fn(&mut App, Option<f64>)| {
		row()
			.gap(6.0)
			.center_y()
			.child(text(label).font(mono(12.0)).color(dim()).w(28.0))
			.child(
				text(t.map_or("—".to_string(), clock))
					.font(digits(12.0))
					.color(if t.is_some() { ink() } else { mute() })
					.w(56.0),
			)
			.child(button("set", false, move |s: &mut App| set(s, Some(s.position()))))
			.children(t.is_some().then(|| button("clear", false, move |s: &mut App| set(s, None))))
	};
	c = c.child(el().h(4.0)).child(header("time")).child(mark("in", s.trim_in, |s, t| s.trim_in = t)).child(mark(
		"out",
		s.trim_out,
		|s, t| s.trim_out = t,
	));
	// a song under it: from the in point, over cuts, at its own speed
	c = c.child(el().h(4.0)).child(header("music"));
	// the link box: a SoundCloud or YouTube page, fetched with yt-dlp
	let link = TextInput::new(&s.song_link)
		.placeholder(if s.clip_edit.music.is_some() {
			"another: paste a SoundCloud or YouTube link"
		} else {
			"paste a SoundCloud or YouTube link, then enter"
		})
		.width(None)
		.on_edit(|s: &mut App| &mut s.song_link)
		.on_submit(|s: &mut App, _: &mut Tx| s.fetch_song())
		.build(k)
		.key("input-song-link");
	let fetching = s.fetching.as_ref().map(|(url, _, p)| {
		let done = p.load(std::sync::atomic::Ordering::Relaxed) as f32 / 1000.0;
		col()
			.w_full()
			.gap(4.0)
			.child(text(format!("getting {url}")).font(mono(11.0)).color(dim()).truncate().min_w(0.0))
			.child(el().w_full().h(4.0).fill(line()).child(el().w_frac(done.max(0.01)).h_full().fill(aqua())))
	});
	c = match &s.clip_edit.music {
		None => c
			.child(
				row()
					.w_full()
					.gap(8.0)
					.center_y()
					.child(
						button("add music…", false, |_: &mut App| {})
							.key("add-music")
							.on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_music(tx)),
					)
					.child(text("or drop a song on the window").font(mono(11.0)).color(mute())),
			)
			.child(link)
			.children(fetching),
		Some(m) => {
			let len = s.music_len.as_ref().filter(|l| l.0 == m.path).map_or(600.0, |l| l.1);
			// the part of the song that plays: as long as the video
			let (a, b) = (s.trim_in.unwrap_or(0.0), s.trim_out.unwrap_or(s.st.duration));
			let cut: f64 = s
				.clip_edit
				.ranges
				.iter()
				.filter(|r| r.mode == RangeMode::Cut)
				.map(|r| (r.to.min(b) - r.from.max(a)).max(0.0))
				.sum();
			let out = ((b - a - cut) / (s.look.speed as f64).clamp(0.05, 4.0)).max(0.1);
			let picker =
				song_picker(s.song_shape.as_ref().filter(|x| x.0 == m.path).map(|x| x.1.as_slice()), m.from, len, out);
			let music = |s: &App| s.clip_edit.music.clone().unwrap_or_default();
			let change = |s: &mut App, f: &dyn Fn(&mut spotter_core::edit::Music)| {
				s.change_edit(|e| {
					if let Some(m) = e.music.as_mut() {
						f(m);
					}
				})
			};
			let beats = s.strip.as_ref().map_or(0, |st| st.beats.len());
			c.child(
				row()
					.w_full()
					.gap(8.0)
					.center_y()
					.child(text(m.name()).font(mono(12.0)).color(aqua()).truncate().min_w(0.0).help(m.path.clone()))
					.child(spacer())
					.child(
						button("change…", false, |_: &mut App| {})
							.key("change-music")
							.on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_music(tx)),
					)
					.child(button("take out", false, |s: &mut App| s.change_edit(|e| e.music = None)).key("music-out")),
			)
			.child(picker.on_drag(move |s: &mut App, d: &mut Drag, tx: &mut Tx| {
				s.drag_song(d, len, out);
				tx.instant();
			}))
			.child(
				row()
					.w_full()
					.gap(6.0)
					.child(
						text(format!("{} to {}", clock(m.from), clock((m.from + out).min(len))))
							.font(digits(11.0))
							.color(ink()),
					)
					.child(
						text(format!("of {}; drag along the song to pick the part", clock(len)))
							.font(mono(11.0))
							.color(mute())
							.truncate()
							.min_w(0.0),
					),
			)
			.child(line_of(
				"song",
				"how loud the song is (1 as it is)",
				row()
					.grow(1.0)
					.gap(6.0)
					.center_y()
					.child(slider_by(
						"music-gain".into(),
						m.gain,
						0.0,
						1.5,
						0.05,
						Some(1.0),
						move |s: &App| music(s).gain,
						move |s: &mut App, v| change(s, &|m| m.gain = v),
					))
					.child(text(format!("{:.2}", m.gain)).font(digits(11.0)).color(ink()).w(48.0)),
			))
			.child(line_of(
				"clip",
				"how much of the clip's own sound stays under the song: 0 none of it",
				row()
					.grow(1.0)
					.gap(6.0)
					.center_y()
					.child(slider_by(
						"music-keep".into(),
						m.keep,
						0.0,
						1.0,
						0.05,
						None,
						move |s: &App| music(s).keep,
						move |s: &mut App, v| change(s, &|m| m.keep = v),
					))
					.child(text(format!("{:.2}", m.keep)).font(digits(11.0)).color(ink()).w(48.0)),
			))
			.child(
				text(if beats > 0 {
					format!("{beats} beats on the timeline; looks on the beat (beat in the look tab) cut to them")
				} else {
					"finding its beats…".to_string()
				})
				.font(mono(11.0))
				.color(dim())
				.wrap(),
			)
			.child(link)
			.children(fetching)
		}
	};
	// what will be written
	let (a, b) = (s.trim_in.unwrap_or(0.0), s.trim_out.unwrap_or(s.st.duration));
	let cut: f64 = s
		.clip_edit
		.ranges
		.iter()
		.filter(|r| r.mode == RangeMode::Cut)
		.map(|r| (r.to.min(b) - r.from.max(a)).max(0.0))
		.sum();
	let size = std::iter::once(s.aspect)
		.chain(s.also.iter().copied().filter(|&a| a != s.aspect))
		.map(|a| ASPECTS[a].1.map_or("as shot".to_string(), |(w, h)| format!("{w}x{h}")))
		.collect::<Vec<_>>()
		.join(" + ");
	let length = if cut > 0.0 {
		format!("{:.1} s − {:.1} s cut = {:.1} s", b - a, cut, b - a - cut)
	} else {
		format!("{:.1} s", (b - a).max(0.0))
	};
	c =
		c.child(el().h(4.0)).child(header("writes")).child(text(length).font(digits(12.0)).color(ink())).child(
			text(format!("{size} · {} · d-fine {}", s.preset, s.detector())).font(mono(11.0)).color(dim()).wrap(),
		);
	c = match s.st.export {
		Some((n, total)) => c.child(
			row()
				.w_full()
				.gap(8.0)
				.center_y()
				.child(
					el().grow(1.0)
						.h(6.0)
						.fill(line())
						.child(el().w_frac((n as f32 / total.max(1) as f32).clamp(0.0, 1.0)).h_full().fill(accent())),
				)
				.child(
					text(if s.st.export_part.1 > 1 {
						format!("{} of {} · {n}/{total}", s.st.export_part.0, s.st.export_part.1)
					} else {
						format!("{n}/{total}")
					})
					.font(digits(11.0))
					.color(dim()),
				)
				.child(button("stop", false, |s: &mut App| s.send(Cmd::StopExport))),
		),
		None => c.child(el().h(4.0)).child(
			row()
				.gap(8.0)
				.center_y()
				.child(
					primary("export…", |_: &mut App| {})
						.key("export")
						.on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_export(tx)),
				)
				.child(
					button("still…", false, |_: &mut App| {})
						.key("export-still")
						.on_press_tx(|s: &mut App, tx: &mut Tx| s.ask_still(tx)),
				)
				.child(text("the frame on screen, as a PNG").font(mono(11.0)).color(mute())),
		),
	};
	if let Some(said) = &s.st.exported {
		c = c.child(text(said.clone()).font(mono(11.0)).color(dim()).wrap());
	}
	if let Some(p) = &s.st.wrote {
		let p = p.clone();
		c = c.child(button("show in folder", false, move |s: &mut App| s.show_in_folder(&p)).key("show-in-folder"));
	}
	c
}
