//! Every look, drawn over the same made-up clip, must come out as it did when `looks.txt` was last
//! written: once alone, once with things followed through the clip (labels, insets, foresight and
//! picks draw then), mixed in pairs, and under an edit with every kind of mark.
//!
//! After a change that's meant to alter a look, rewrite the file and read the diff:
//! `LOOKS_BLESS=1 cargo test --release -p spotter-core --test looks`.

use spotter_core::edit::{Edit, Pick, Range, RangeMode, WHOLE, Zone, ZoneMode};
use spotter_core::objects::Detection;
use spotter_core::tape::{Change, Record, Seen, Tape};
use spotter_core::{Engine, Frame, Look};
use std::collections::BTreeMap;
use std::sync::Arc;

const W: usize = 320;
const H: usize = 180;
const FRAMES: usize = 16;

/// The three moving things: where each is on frame `i` (x, y, side, in pixels), and what it's called.
fn things(i: usize) -> [(usize, usize, usize, &'static str); 3] {
	[(20 + i * 9, 60, 36, "person"), (260 - i * 6, 110, 28, "car"), (140, 10 + i * 5, 22, "dog")]
}

/// A textured background with the three things on it.
fn scene(i: usize) -> Frame {
	let mut f = Frame::new(W as u32, H as u32);
	for y in 0..H {
		for x in 0..W {
			let p = (y * W + x) * 4;
			let n = ((x * 7919 + y * 104_729) % 251) as u8 / 8;
			f.rgba[p] = (40 + x / 8) as u8 + n;
			f.rgba[p + 1] = (50 + y / 6) as u8 + n;
			f.rgba[p + 2] = 70 + n;
		}
	}
	let colours = [[235, 230, 220], [20, 20, 30], [250, 190, 40]];
	for (k, (x0, y0, s, _)) in things(i).into_iter().enumerate() {
		for y in y0..(y0 + s).min(H) {
			for x in x0..(x0 + s).min(W) {
				let (dx, dy) = (x as f32 - (x0 + s / 2) as f32, y as f32 - (y0 + s / 2) as f32);
				if k == 1 && dx * dx + dy * dy > (s * s / 4) as f32 {
					continue;
				}
				let p = (y * W + x) * 4;
				f.rgba[p..p + 3].copy_from_slice(&colours[k]);
			}
		}
	}
	f
}

/// The detector's view of the clip, as the look ahead would have recorded it.
fn record() -> Record {
	let frames = (0..FRAMES)
		.map(|i| Seen {
			t: i as f64 / 30.0,
			jump: i == 0,
			change: Change { moved: 1.0, colours: 0.01 },
			found: things(i)
				.iter()
				.map(|&(x, y, s, cls)| Detection {
					cls: cls.into(),
					score: 0.9,
					x: x as f32 / W as f32,
					y: y as f32 / H as f32,
					w: s as f32 / W as f32,
					h: s as f32 / H as f32,
				})
				.collect(),
		})
		.collect();
	Record { frames, fps: 30.0, aspect: W as f32 / H as f32, view: WHOLE, model: String::new() }
}

/// An edit with one of everything: a pick, a thing taken out, an area left alone, and a cut, a bare
/// and an unmix stretch.
fn marked() -> Edit {
	Edit {
		picks: vec![Pick { t: 0.2, x: 0.25, y: 0.43 }],
		drops: vec![Pick { t: 0.2, x: 0.13, y: 0.28 }],
		zones: vec![Zone { x: 0.0, y: 0.8, w: 0.3, h: 0.2, mode: ZoneMode::Ignore }],
		ranges: vec![
			Range { from: 0.10, to: 0.15, mode: RangeMode::Cut },
			Range { from: 0.20, to: 0.30, mode: RangeMode::Bare },
			Range { from: 0.35, to: 0.45, mode: RangeMode::Unmix },
		],
		..Edit::default()
	}
}

fn fnv(b: &[u8]) -> u64 {
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	for &x in b {
		h ^= x as u64;
		h = h.wrapping_mul(0x0100_0000_01b3);
	}
	h
}

/// The last frame's hash: `look` over the clip, followed through with the record when `taped`,
/// under `edit`.
fn render(look: Look, taped: bool, edit: &Edit) -> u64 {
	let mut e = Engine::new(look.clone());
	e.started = 1_794_279_845.0;
	e.name = "looks".into();
	e.edit = edit.clone();
	if taped {
		e.set_tape(Some(Arc::new(Tape::build(&record(), &look, edit))));
	}
	let mut out = Frame::default();
	for i in 0..FRAMES {
		e.step(&scene(i), i as f64 / 30.0, 1.0 / 30.0, &mut out).unwrap();
	}
	fnv(&out.rgba)
}

fn mixed(base: &str, other: &str, how: &str) -> Look {
	let mut l = Look::preset(base).unwrap();
	l.mix = other.into();
	l.mix_how = how.into();
	l.mix_at = if matches!(how, "split" | "blend" | "flick") { 0.5 } else { 1.0 };
	l
}

/// Every case's name and hash.
fn all() -> BTreeMap<String, u64> {
	let mut got = BTreeMap::new();
	let plain = Edit::default();
	for name in Look::presets() {
		let l = Look::preset(name).unwrap();
		got.insert(format!("alone/{name}"), render(l.clone(), false, &plain));
		got.insert(format!("taped/{name}"), render(l, true, &plain));
	}
	for (base, other, how) in [
		("databend", "plates", "over"),
		("plates", "corrupt", "under"),
		("heat", "sketch", "split"),
		("heat", "sketch", "things"),
		("marey", "pop art", "flick"),
		("neon", "riso", "blend"),
	] {
		got.insert(format!("mixed/{base} + {other} {how}"), render(mixed(base, other, how), true, &plain));
	}
	for name in ["smoke", "plates", "heat", "isolate"] {
		got.insert(format!("marked/{name}"), render(mixed(name, "sketch", "things"), true, &marked()));
	}
	got
}

fn file() -> std::path::PathBuf {
	std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("looks.txt")
}

#[test]
fn every_look_is_unchanged() {
	let got = all();
	if std::env::var_os("LOOKS_BLESS").is_some() {
		let text: String = got.iter().map(|(k, v)| format!("{v:016x}  {k}\n")).collect();
		std::fs::write(file(), text).unwrap();
		return;
	}
	let want: BTreeMap<String, u64> = std::fs::read_to_string(file())
		.expect("tests/looks.txt (write it with LOOKS_BLESS=1)")
		.lines()
		.filter_map(|l| l.split_once("  ").map(|(h, k)| (k.to_string(), u64::from_str_radix(h, 16).unwrap())))
		.collect();
	let changed: Vec<&String> = got.iter().filter(|(k, v)| want.get(*k) != Some(v)).map(|(k, _)| k).collect();
	let gone: Vec<&String> = want.keys().filter(|k| !got.contains_key(*k)).collect();
	assert!(changed.is_empty() && gone.is_empty(), "changed: {changed:?}\nmissing: {gone:?}");
}
