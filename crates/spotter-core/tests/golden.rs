//! Every overlay look, drawn over the same made-up video, must come out exactly as it did when this
//! was written. The other looks' settings are all off by default, so nothing added may change these.
//! If one changes on purpose, run with `GOLDEN_PRINT=1 cargo test --release -p
//! spotter-core --test golden -- --nocapture` and paste the new numbers in.

use spotter_core::{Engine, Frame, Look};

/// A textured background with three things moving across it, the same every time.
fn scene(i: usize) -> Frame {
	let (w, h) = (320usize, 180usize);
	let mut f = Frame::new(w as u32, h as u32);
	for y in 0..h {
		for x in 0..w {
			let p = (y * w + x) * 4;
			let n = ((x * 7919 + y * 104_729) % 251) as u8 / 8;
			f.rgba[p] = (40 + x / 8) as u8 + n;
			f.rgba[p + 1] = (50 + y / 6) as u8 + n;
			f.rgba[p + 2] = 70 + n;
		}
	}
	let mut paint = |x0: usize, y0: usize, s: usize, c: [u8; 3], round: bool| {
		for y in y0..(y0 + s).min(h) {
			for x in x0..(x0 + s).min(w) {
				let (dx, dy) = (x as f32 - (x0 + s / 2) as f32, y as f32 - (y0 + s / 2) as f32);
				if round && dx * dx + dy * dy > (s * s / 4) as f32 {
					continue;
				}
				let p = (y * w + x) * 4;
				f.rgba[p..p + 3].copy_from_slice(&c);
			}
		}
	};
	paint(20 + i * 9, 60, 36, [235, 230, 220], false);
	paint(260 - i * 6, 110, 28, [20, 20, 30], true);
	paint(140, 10 + i * 5, 22, [250, 190, 40], false);
	f
}

fn fnv(b: &[u8]) -> u64 {
	let mut h = 0xcbf2_9ce4_8422_2325u64;
	for &x in b {
		h ^= x as u64;
		h = h.wrapping_mul(0x0100_0000_01b3);
	}
	h
}

fn render(name: &str) -> u64 {
	let mut e = Engine::new(Look::preset(name).unwrap());
	e.started = 1_794_279_845.0;
	e.name = "golden".into();
	let mut out = Frame::default();
	for i in 0..16 {
		e.step(&scene(i), i as f64 / 30.0, 1.0 / 30.0, &mut out).unwrap();
	}
	fnv(&out.rgba)
}

const GOLDEN: &[(&str, u64)] = &[
	("smoke", 0x57a0b97de7f656a5),
	("cctv", 0x20a16e69f81bd62d),
	("panopticon", 0xe892c66f33a0b045),
	("ember", 0x438a47645ff7f5c5),
	("night watch", 0x7074e60b76b22e26),
	("blacksite", 0xfea13e01320dd35c),
	("sprawl", 0x2f46c5f0d8f0d796),
	("census", 0x996a64a2dc230a68),
	("manifest", 0x295a8090fad22322),
	("lock-on", 0xcd5d71e0fb9022ce),
	("wake", 0xabeb83213cc17598),
	("flow", 0x5ce75b23c0107d7f),
	("broadcast", 0xec28891aabf2a91d),
	("network", 0xdbb3fe8c9b2ea899),
	("radar", 0xcfd276096b85d280),
	("survey", 0x71ea733cba1aa3ef),
	("telemetry", 0xa7e69267f1f93b2f),
	("stars", 0x728583a90b3d4839),
	("captions", 0x4577e385d698253d),
];

#[test]
fn the_overlay_looks_are_unchanged() {
	let print = std::env::var_os("GOLDEN_PRINT").is_some();
	let mut wrong = Vec::new();
	for &(name, want) in GOLDEN {
		let got = render(name);
		if print {
			println!("\t(\"{name}\", 0x{got:016x}),");
		}
		if got != want {
			wrong.push(name);
		}
	}
	assert!(wrong.is_empty(), "these looks came out differently: {wrong:?}");
}

#[test]
fn a_render_repeats_exactly() {
	assert_eq!(render("cctv"), render("cctv"));
	assert_eq!(render("wake"), render("wake"));
}
