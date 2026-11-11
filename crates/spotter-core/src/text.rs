//! Lettering, drawn as outlines so it sits on the frame exactly as the boxes do. The fonts are the
//! site's (all under the SIL Open Font License, see assets/fonts), built into the program.

use ab_glyph::{Font, FontArc, FontVec, OutlineCurve, PxScale, ScaleFont, VariableFont};
use std::sync::OnceLock;
use tiny_skia::{Path, PathBuilder};

const CONDENSED: &[u8] = include_bytes!("../../../assets/fonts/BarlowSemiCondensed-Medium.ttf");
const MONO: &[u8] = include_bytes!("../../../assets/fonts/DMMono-Medium.ttf");
const TECH: &[u8] = include_bytes!("../../../assets/fonts/ShareTechMono-Regular.ttf");
const TERMINAL: &[u8] = include_bytes!("../../../assets/fonts/VT323-Regular.ttf");
const SYSTEM: &[u8] = include_bytes!("../../../assets/fonts/IBMPlexSans.ttf");

struct Fonts {
	condensed: FontArc,
	mono: FontArc,
	tech: FontArc,
	terminal: FontArc,
	system: FontArc,
}

fn fonts() -> &'static Fonts {
	static F: OnceLock<Fonts> = OnceLock::new();
	F.get_or_init(|| {
		let load = |b: &'static [u8]| FontArc::try_from_slice(b).expect("a built-in font reads");
		// the site asks for weight 500; Plex is a variable font, so set it
		let mut plex = FontVec::try_from_vec(SYSTEM.to_vec()).expect("a built-in font reads");
		plex.set_variation(b"wght", 500.0);
		Fonts {
			condensed: load(CONDENSED),
			mono: load(MONO),
			tech: load(TECH),
			terminal: load(TERMINAL),
			system: FontArc::new(plex),
		}
	})
}

/// A font at a size, by the look's name for it (condensed, mono, tech, terminal or system).
#[derive(Clone)]
pub struct Face {
	font: FontArc,
	/// pixels per em, as CSS sizes fonts
	size: f32,
}

impl Face {
	pub fn new(name: &str, size: f32) -> Self {
		let f = fonts();
		let font = match name {
			"condensed" => &f.condensed,
			"mono" => &f.mono,
			"tech" => &f.tech,
			"terminal" => &f.terminal,
			_ => &f.system,
		}
		.clone();
		Self { font, size }
	}

	fn scale(&self) -> PxScale {
		// ab_glyph's scale is the height from descender to ascender; CSS's is the em
		let upem = self.font.units_per_em().unwrap_or(1000.0);
		PxScale::from(self.size * self.font.height_unscaled() / upem)
	}

	/// How wide a line of text comes out, in pixels.
	pub fn measure(&self, text: &str) -> f32 {
		let s = self.font.as_scaled(self.scale());
		let mut w = 0.0;
		let mut last = None;
		for c in text.chars() {
			let id = s.glyph_id(c);
			if let Some(prev) = last {
				w += s.kern(prev, id);
			}
			w += s.h_advance(id);
			last = Some(id);
		}
		w
	}

	/// Where the alphabetic baseline goes for text whose middle (canvas's `textBaseline = 'middle'`,
	/// halfway down the em box) is at `y`.
	pub fn baseline(&self, y: f32) -> f32 {
		let s = self.font.as_scaled(self.scale());
		y + (s.ascent() + s.descent()) / 2.0
	}

	/// The outline of a line of text starting at `x` with its baseline at `y`, or None for nothing.
	pub fn path(&self, text: &str, x: f32, y: f32) -> Option<Path> {
		let s = self.font.as_scaled(self.scale());
		let (sx, sy) = (s.h_scale_factor(), s.v_scale_factor());
		let mut pb = PathBuilder::new();
		let mut pen = x;
		let mut last = None;
		for c in text.chars() {
			let id = s.glyph_id(c);
			if let Some(prev) = last {
				pen += s.kern(prev, id);
			}
			if let Some(o) = self.font.outline(id) {
				let at = |p: ab_glyph::Point| (pen + p.x * sx, y - p.y * sy);
				let mut cur: Option<(f32, f32)> = None;
				for curve in &o.curves {
					let start = match curve {
						OutlineCurve::Line(a, _) | OutlineCurve::Quad(a, _, _) | OutlineCurve::Cubic(a, _, _, _) => {
							at(*a)
						}
					};
					if cur.is_none_or(|c| (c.0 - start.0).abs() > 1e-3 || (c.1 - start.1).abs() > 1e-3) {
						if cur.is_some() {
							pb.close();
						}
						pb.move_to(start.0, start.1);
					}
					let end = match curve {
						OutlineCurve::Line(_, b) => {
							let b = at(*b);
							pb.line_to(b.0, b.1);
							b
						}
						OutlineCurve::Quad(_, b, c) => {
							let (b, c) = (at(*b), at(*c));
							pb.quad_to(b.0, b.1, c.0, c.1);
							c
						}
						OutlineCurve::Cubic(_, b, c, d) => {
							let (b, c, d) = (at(*b), at(*c), at(*d));
							pb.cubic_to(b.0, b.1, c.0, c.1, d.0, d.1);
							d
						}
					};
					cur = Some(end);
				}
				if cur.is_some() {
					pb.close();
				}
			}
			pen += s.h_advance(id);
			last = Some(id);
		}
		pb.finish()
	}
}

#[cfg(test)]
mod tests {
	use super::Face;

	#[test]
	fn every_font_draws() {
		for name in ["condensed", "mono", "tech", "terminal", "system"] {
			let f = Face::new(name, 20.0);
			let w = f.measure("SUBJECT 37001");
			assert!(w > 60.0 && w < 260.0, "{name}: {w}");
			assert!(f.path("SUBJECT 37001", 0.0, 20.0).is_some(), "{name}");
		}
		// a monospace font's characters are all one width
		let m = Face::new("mono", 20.0);
		assert!((m.measure("iiii") - m.measure("WWWW")).abs() < 0.01);
	}
}
