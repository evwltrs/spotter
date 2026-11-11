//! Drawing the tracks over the picture: lines first, then filled patches, then boxes and numbers.
//! What a track looks like (its size, whether it has a box, what fills it) is decided by hashing its
//! id, so it stays the same for as long as the track lives. All sizes in the settings are as if the
//! frame's shorter side were 1080 px, so a look comes out the same at any resolution.
//! A port of 1nf.me's `src/lib/tracker/overlay.ts`, drawing with tiny-skia where the site uses a 2D canvas.

use crate::Frame;
use crate::look::Look;
use crate::objects::{ObjectTrack, presence};
use crate::text::Face;
use crate::tracking::{Region, Track, hash};
use tiny_skia::{
	BlendMode, Color, FillRule, LineCap, LineJoin, Paint, Path, PathBuilder, Pixmap, PixmapMut, PixmapPaint, Rect,
	Stroke, StrokeDash, Transform,
};

/// Fill {tokens} into a template; a token nothing answers to is left as it was.
pub fn fill(template: &str, get: &dyn Fn(&str) -> Option<String>) -> String {
	let mut out = String::with_capacity(template.len() + 16);
	let mut rest = template;
	while let Some(open) = rest.find('{') {
		out.push_str(&rest[..open]);
		let after = &rest[open + 1..];
		let name_len = after.find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).unwrap_or(after.len());
		if name_len > 0 && after[name_len..].starts_with('}') {
			let name = &after[..name_len];
			match get(name) {
				Some(v) => out.push_str(&v),
				None => {
					out.push('{');
					out.push_str(name);
					out.push('}');
				}
			}
			rest = &after[name_len + 1..];
		} else {
			out.push('{');
			rest = after;
		}
	}
	out.push_str(rest);
	out
}

/// What's on the frame this time.
pub struct Scene<'a> {
	pub tracks: &'a [Track],
	pub regions: &'a [Region],
	/// the detector's objects on show
	pub objects: &'a [ObjectTrack],
	/// 0 draws the movement's big boxes, 1 the detector's, and in between it crossfades
	pub blend: f32,
	/// the small copy of the frame, for pixelated patches
	pub low: &'a Frame,
	/// the tokens for the writing on the frame
	pub hud: &'a dyn Fn(&str) -> Option<String>,
}

/// A colour as the look writes it ("#rgb" or "#rrggbb"), at an opacity.
pub fn colour(hex: &str, alpha: f32) -> Color {
	let h = hex.trim_start_matches('#');
	let v = |s: &str| u8::from_str_radix(s, 16).ok();
	let rgb = match h.len() {
		3 => {
			let d: Vec<u8> = h.chars().filter_map(|c| c.to_digit(16).map(|d| d as u8 * 17)).collect();
			(d.len() == 3).then(|| (d[0], d[1], d[2]))
		}
		6 => match (v(&h[0..2]), v(&h[2..4]), v(&h[4..6])) {
			(Some(r), Some(g), Some(b)) => Some((r, g, b)),
			_ => None,
		},
		_ => None,
	};
	let (r, g, b) = rgb.unwrap_or((255, 255, 255));
	Color::from_rgba8(r, g, b, (alpha.clamp(0.0, 1.0) * 255.0).round() as u8)
}

fn box_of(t: &Track, l: &Look, u: f32) -> (f32, f32) {
	let r = hash(t.id, 1);
	// cubed, so most are small and a few are large
	let size = (l.size_min + (l.size_max.max(l.size_min) - l.size_min) * r * r * r) * u;
	let aspect = 0.45 + hash(t.id, 2) * 1.9;
	(size * aspect.sqrt(), size / aspect.sqrt())
}

// the first few are the kinds seen most, kept apart so that a person and a car are never the same colour
const COLOURS: [&str; 8] = ["#3dff8b", "#ffd23f", "#39c6ff", "#ff6b6b", "#c79bff", "#ff9f43", "#5ef2d6", "#ff7ad9"];
const COMMON: [&str; 8] = ["person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat"];

fn pick(h: f32) -> &'static str {
	COLOURS[((h * COLOURS.len() as f32) as usize).min(COLOURS.len() - 1)]
}

fn kind_colour(kind: &str) -> &'static str {
	if let Some(i) = COMMON.iter().position(|k| *k == kind) {
		return COLOURS[i];
	}
	// the site's string hash: n * 31 + each UTF-16 unit, kept to 32 bits
	let mut n: i32 = 0;
	for c in kind.encode_utf16() {
		n = n.wrapping_mul(31).wrapping_add(c as i32);
	}
	pick(hash(n as i64, 9))
}

/// The colour something's drawn in, or None when it's the look's one colour.
fn tint_of(l: &Look, id: i64, kind: Option<&str>) -> Option<&'static str> {
	match (l.color_by.as_str(), kind) {
		("each its own", _) => Some(pick(hash(id, 9))),
		("by kind", Some(k)) => Some(kind_colour(k)),
		_ => None,
	}
}

/// One path per colour, in the order the colours first came up, so a frame costs a few strokes
/// however many things are on it.
struct Pens(Vec<(String, PathBuilder)>);

impl Pens {
	fn new() -> Self {
		Self(Vec::new())
	}
	fn pen(&mut self, colour: &str) -> &mut PathBuilder {
		let i = match self.0.iter().position(|(c, _)| c == colour) {
			Some(i) => i,
			None => {
				self.0.push((colour.to_string(), PathBuilder::new()));
				self.0.len() - 1
			}
		};
		&mut self.0[i].1
	}
	fn finish(self) -> Vec<(String, Path)> {
		self.0.into_iter().filter_map(|(c, pb)| pb.finish().map(|p| (c, p))).collect()
	}
}

fn rect(x: f32, y: f32, w: f32, h: f32) -> Option<Rect> {
	Rect::from_xywh(x, y, w.max(0.01), h.max(0.01))
}

fn stroke_box(c: &mut PathBuilder, x: f32, y: f32, w: f32, h: f32, style: &str, fw: f32, fh: f32) {
	let x = x.round() + 0.5;
	let y = y.round() + 0.5;
	let w = w.round();
	let h = h.round();
	let mx = x + w / 2.0;
	let my = y + h / 2.0;
	let seg = |c: &mut PathBuilder, pts: &[(f32, f32)]| {
		c.move_to(pts[0].0, pts[0].1);
		for p in &pts[1..] {
			c.line_to(p.0, p.1);
		}
	};
	match style {
		"corners" | "sights" => {
			let a = w.min(h) * 0.28;
			seg(c, &[(x, y + a), (x, y), (x + a, y)]);
			seg(c, &[(x + w - a, y), (x + w, y), (x + w, y + a)]);
			seg(c, &[(x + w, y + h - a), (x + w, y + h), (x + w - a, y + h)]);
			seg(c, &[(x + a, y + h), (x, y + h), (x, y + h - a)]);
			if style == "sights" {
				seg(c, &[(mx, 0.0), (mx, y)]);
				seg(c, &[(mx, y + h), (mx, fh)]);
				seg(c, &[(0.0, my), (x, my)]);
				seg(c, &[(x + w, my), (fw, my)]);
			}
		}
		"cross" => {
			let a = w.min(h) * 0.5;
			seg(c, &[(mx - a, my), (mx + a, my)]);
			seg(c, &[(mx, my - a), (mx, my + a)]);
		}
		"brackets" => {
			let a = (w * 0.22).min(h * 0.4);
			seg(c, &[(x + a, y), (x, y), (x, y + h), (x + a, y + h)]);
			seg(c, &[(x + w - a, y), (x + w, y), (x + w, y + h), (x + w - a, y + h)]);
		}
		"diamond" => seg(c, &[(mx, y), (x + w, my), (mx, y + h), (x, my), (mx, y)]),
		"ring" | "dot" => {
			// held to a size, or something filling the frame would get a ring as big as the frame
			let most = fw.min(fh) * if style == "dot" { 0.01 } else { 0.2 };
			let r = most.min(if style == "dot" { 1.5f32.max(w.min(h) / 4.0) } else { (w * h).sqrt() / 2.0 });
			if r > 0.0 {
				c.push_circle(mx, my, r);
			}
		}
		"foot" => {
			let rx = w * 0.62;
			let ry = (w * 0.17).max(3.0);
			if let Some(r) = Rect::from_ltrb(mx - rx, y + h - ry, mx + rx, y + h + ry) {
				c.push_oval(r);
			}
		}
		"none" => {}
		_ => {
			if let Some(r) = rect(x, y, w, h) {
				c.push_rect(r);
			}
		}
	}
}

/// What the canvas's state would be: how wide lines are, how see-through, and so on.
struct Canvas<'a> {
	pm: PixmapMut<'a>,
	w: f32,
	h: f32,
}

impl Canvas<'_> {
	fn stroke(&mut self, path: &Path, colour: &str, alpha: f32, width: f32, dash: Option<[f32; 2]>, cap: LineCap) {
		if alpha <= 0.0 {
			return;
		}
		let mut paint = Paint::default();
		paint.set_color(colour_with(colour, alpha));
		paint.anti_alias = true;
		let stroke = Stroke {
			width,
			miter_limit: 10.0,
			line_cap: cap,
			line_join: LineJoin::Miter,
			dash: dash.and_then(|d| StrokeDash::new(d.to_vec(), 0.0)),
		};
		self.pm.stroke_path(path, &paint, &stroke, Transform::identity(), None);
	}

	fn fill(&mut self, path: &Path, colour: Color, blend: BlendMode) {
		if colour.alpha() <= 0.0 {
			return;
		}
		let mut paint = Paint::default();
		paint.set_color(colour);
		paint.anti_alias = true;
		paint.blend_mode = blend;
		self.pm.fill_path(path, &paint, FillRule::Winding, Transform::identity(), None);
	}

	fn fill_rect(&mut self, x: f32, y: f32, w: f32, h: f32, colour: Color, blend: BlendMode) {
		if let Some(r) = rect(x, y, w, h) {
			let mut paint = Paint::default();
			paint.set_color(colour);
			paint.anti_alias = true;
			paint.blend_mode = blend;
			self.pm.fill_rect(r, &paint, Transform::identity(), None);
		}
	}

	fn text(&mut self, face: &Face, text: &str, x: f32, middle: f32, colour: Color) {
		if let Some(p) = face.path(text, x, face.baseline(middle)) {
			self.fill(&p, colour, BlendMode::SourceOver);
		}
	}
}

fn colour_with(hex: &str, alpha: f32) -> Color {
	colour(hex, alpha)
}

/// The box's own style, with dashes for "dashed" and a fill for "dot".
fn paint(c: &mut Canvas, path: &Path, style: &str, colour: &str, alpha: f32, width: f32, u: f32) {
	let dash = (style == "dashed").then_some([6.0 * u, 5.0 * u]);
	c.stroke(path, colour, alpha, width, dash, LineCap::Butt);
	if style == "dot" {
		c.fill(path, colour_with(colour, alpha), BlendMode::SourceOver);
	}
}

fn centre(r: (f32, f32, f32, f32)) -> (f32, f32) {
	(r.0 + r.2 / 2.0, r.1 + r.3 / 2.0)
}

fn object_links(c: &mut Canvas, s: &Scene, l: &Look, m: f32, u: f32) {
	let items: Vec<(f32, f32, f32, f32)> = if s.blend > 0.5 {
		s.objects.iter().map(|o| (o.x, o.y, o.w, o.h)).collect()
	} else {
		s.regions.iter().map(|r| (r.x, r.y, r.w, r.h)).collect()
	};
	if items.len() < 2 || l.link_opacity <= 0.0 {
		return;
	}
	let most = (l.link_max / 100.0).powi(2);
	let mut pb = PathBuilder::new();
	for (i, a) in items.iter().enumerate() {
		let (ax, ay) = centre(*a);
		let mut near: Vec<(f32, usize)> = items
			.iter()
			.enumerate()
			.filter(|(j, _)| *j != i)
			.map(|(j, t)| {
				let (tx, ty) = centre(*t);
				((ax - tx).powi(2) + (ay - ty).powi(2), j)
			})
			.filter(|(d, _)| *d <= most)
			.collect();
		near.sort_by(|p, q| p.0.total_cmp(&q.0));
		for (_, j) in near.into_iter().take(l.link_count.round().max(0.0) as usize) {
			let (tx, ty) = centre(items[j]);
			pb.move_to(ax * m, ay * m);
			pb.line_to(tx * m, ty * m);
		}
	}
	if let Some(p) = pb.finish() {
		c.stroke(&p, &l.link_color, l.link_opacity, l.link_width * u, None, LineCap::Butt);
	}
}

/// how many steps of fading a trail is drawn in
const FADES: usize = 5;

fn trails(c: &mut Canvas, s: &Scene, l: &Look, m: f32, u: f32) {
	if l.trail <= 0.0 {
		return;
	}
	let mut pens: Vec<Pens> = (0..FADES).map(|_| Pens::new()).collect();
	let mut add = |p: Option<&Vec<f32>>, colour: &str, strength: f32| {
		let Some(p) = p else { return };
		if p.len() < 4 || strength <= 0.0 {
			return;
		}
		let n = p.len() / 2;
		for i in 1..n {
			let fade = (FADES - 1).min(((i as f32 / (n - 1) as f32) * strength * FADES as f32).floor() as usize);
			let line = pens[fade].pen(colour);
			line.move_to(p[i * 2 - 2] * m, p[i * 2 - 1] * m);
			line.line_to(p[i * 2] * m, p[i * 2 + 1] * m);
		}
	};
	for t in s.tracks {
		add(t.path.as_ref(), tint_of(l, t.id, None).unwrap_or(&l.color), 1.0 - t.miss as f32 / (l.linger + 1.0));
	}
	for r in s.regions {
		add(r.path.as_ref(), tint_of(l, r.id, None).unwrap_or(&l.color), 1.0 - s.blend);
	}
	for o in s.objects {
		add(o.path.as_ref(), tint_of(l, o.id, Some(&o.cls)).unwrap_or(&l.color), presence(o) * s.blend);
	}
	for (fade, set) in pens.into_iter().enumerate() {
		let alpha = l.opacity * (fade + 1) as f32 / FADES as f32;
		for (colour, path) in set.finish() {
			c.stroke(&path, &colour, alpha, l.trail_width * u, None, LineCap::Round);
		}
	}
}

fn links(c: &mut Canvas, s: &Scene, l: &Look, m: f32, u: f32) {
	let live = s.tracks;
	if live.len() < 2 || l.link_opacity <= 0.0 {
		return;
	}
	let most = (l.link_max / 100.0).powi(2);
	let k = l.link_count.round().max(0.0) as usize;
	let mut pb = PathBuilder::new();
	if l.link_mode == "chain" {
		for i in 1..live.len() {
			let (a, b) = (&live[i - 1], &live[i]);
			if (a.x - b.x).powi(2) + (a.y - b.y).powi(2) > most {
				continue;
			}
			pb.move_to(a.x * m, a.y * m);
			pb.line_to(b.x * m, b.y * m);
		}
	} else if k > 0 {
		// the hubs are the oldest tracks, and each picks its partners by hash, so a fan of lines holds
		// still instead of redrawing itself every frame
		let nearest = l.link_mode == "nearest";
		let from = if nearest { live } else { &live[..live.len().min(l.hubs.round().max(0.0) as usize)] };
		for a in from {
			let mut near: Vec<(f32, usize)> = Vec::with_capacity(k);
			for (j, t) in live.iter().enumerate() {
				if std::ptr::eq(t, a) {
					continue;
				}
				let dist = (a.x - t.x).powi(2) + (a.y - t.y).powi(2);
				if dist > most {
					continue;
				}
				let d = if nearest { dist } else { hash(a.id, t.id) };
				if near.len() == k && d >= near[k - 1].0 {
					continue;
				}
				if near.len() == k {
					near.pop();
				}
				let at = near.iter().position(|n| n.0 > d).unwrap_or(near.len());
				near.insert(at, (d, j));
			}
			for (_, j) in near {
				pb.move_to(a.x * m, a.y * m);
				pb.line_to(live[j].x * m, live[j].y * m);
			}
		}
	}
	if let Some(p) = pb.finish() {
		c.stroke(&p, &l.link_color, l.link_opacity, l.link_width * u, None, LineCap::Butt);
	}
}

const PATCH_KINDS: [&str; 7] = ["fill", "fill", "fill", "invert", "invert", "pixels", "zoom"];

fn patches(c: &mut Canvas, s: &Scene, l: &Look, m: f32, u: f32) {
	if l.patches <= 0.0 {
		return;
	}
	let low_scale = s.low.w as f32 / c.w;
	for t in s.tracks {
		if hash(t.id, 5) >= l.patches {
			continue;
		}
		let (bw, bh) = box_of(t, l, u);
		// patches run wider than the boxes do: thin slices read better than tiny squares
		let w = (bw * 2.4).max(16.0 * u);
		let h = (bh * 0.8).max(10.0 * u);
		let x = t.x * m - w / 2.0;
		let y = t.y * m - h / 2.0;
		let kind = if l.patch_mode == "mixed" {
			PATCH_KINDS[((hash(t.id, 6) * PATCH_KINDS.len() as f32) as usize).min(PATCH_KINDS.len() - 1)]
		} else {
			l.patch_mode.as_str()
		};
		match kind {
			"fill" => {
				let hex =
					if hash(t.id, 7) < 0.35 && l.patch_mode == "mixed" { "#000000" } else { l.patch_color.as_str() };
				c.fill_rect(x, y, w, h, colour(hex, 0.92), BlendMode::SourceOver);
			}
			"invert" => c.fill_rect(x, y, w, h, Color::WHITE, BlendMode::Difference),
			"pixels" => {
				// a much smaller slice of the small copy, blown up, so the blocks show
				let a = (w * low_scale * 0.25).max(2.0);
				let b = (h * low_scale * 0.25).max(2.0);
				let sx = (x + w / 2.0) * low_scale - a / 2.0;
				let sy = (y + h / 2.0) * low_scale - b / 2.0;
				blow_up(c, s.low, (sx, sy, a, b), (x, y, w, h), false);
			}
			_ => {
				let snapshot = grab(c, (x + w * 0.3, y + h * 0.3, w * 0.4, h * 0.4));
				if let Some((f, ox, oy)) = snapshot {
					blow_up(c, &f, (x + w * 0.3 - ox, y + h * 0.3 - oy, w * 0.4, h * 0.4), (x, y, w, h), true);
				}
			}
		}
	}
}

/// A copy of part of the canvas as it is now, and where the copy starts.
fn grab(c: &Canvas, (x, y, w, h): (f32, f32, f32, f32)) -> Option<(Frame, f32, f32)> {
	let (cw, ch) = (c.w as i32, c.h as i32);
	let x0 = (x.floor() as i32 - 1).clamp(0, cw);
	let y0 = (y.floor() as i32 - 1).clamp(0, ch);
	let x1 = ((x + w).ceil() as i32 + 1).clamp(0, cw);
	let y1 = ((y + h).ceil() as i32 + 1).clamp(0, ch);
	if x1 <= x0 || y1 <= y0 {
		return None;
	}
	let data = c.pm.data_mut_ref();
	let mut out = Frame::new((x1 - x0) as u32, (y1 - y0) as u32);
	for row in y0..y1 {
		let from = ((row * cw + x0) * 4) as usize;
		let to = ((row - y0) * (x1 - x0) * 4) as usize;
		let n = ((x1 - x0) * 4) as usize;
		out.rgba[to..to + n].copy_from_slice(&data[from..from + n]);
	}
	Some((out, x0 as f32, y0 as f32))
}

/// Draw `src`'s part (sx, sy, sw, sh) into the canvas at (x, y, w, h), smoothly or as hard blocks,
/// the way canvas drawImage does with smoothing on or off.
fn blow_up(
	c: &mut Canvas,
	src: &Frame,
	(sx, sy, sw, sh): (f32, f32, f32, f32),
	(x, y, w, h): (f32, f32, f32, f32),
	smooth: bool,
) {
	if w <= 0.0 || h <= 0.0 || src.w == 0 {
		return;
	}
	let (cw, ch) = (c.w as i32, c.h as i32);
	let (fw, fh) = (src.w as i32, src.h as i32);
	let x0 = x.round().max(0.0) as i32;
	let y0 = y.round().max(0.0) as i32;
	let x1 = ((x + w).round() as i32).min(cw);
	let y1 = ((y + h).round() as i32).min(ch);
	let data = c.pm.data_mut();
	let px = |xx: i32, yy: i32| -> [f32; 4] {
		let i = ((yy.clamp(0, fh - 1) * fw + xx.clamp(0, fw - 1)) * 4) as usize;
		[src.rgba[i] as f32, src.rgba[i + 1] as f32, src.rgba[i + 2] as f32, src.rgba[i + 3] as f32]
	};
	for dy in y0..y1 {
		let v = sy + (dy as f32 + 0.5 - y) * sh / h;
		for dx in x0..x1 {
			let uu = sx + (dx as f32 + 0.5 - x) * sw / w;
			let p = if smooth {
				let (fx, fy) = (uu - 0.5, v - 0.5);
				let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
				let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
				let (a, b, cc, d) = (px(ix, iy), px(ix + 1, iy), px(ix, iy + 1), px(ix + 1, iy + 1));
				let mut o = [0.0; 4];
				for k in 0..4 {
					let top = a[k] + (b[k] - a[k]) * tx;
					let bot = cc[k] + (d[k] - cc[k]) * tx;
					o[k] = top + (bot - top) * ty;
				}
				o
			} else {
				px(uu.floor() as i32, v.floor() as i32)
			};
			let i = ((dy * cw + dx) * 4) as usize;
			for k in 0..3 {
				data[i + k] = p[k].round().clamp(0.0, 255.0) as u8;
			}
			data[i + 3] = 255;
		}
	}
}

trait DataRef {
	fn data_mut_ref(&self) -> &[u8];
}

impl DataRef for PixmapMut<'_> {
	fn data_mut_ref(&self) -> &[u8] {
		self.as_ref().data()
	}
}

/// Something a label can be written about.
struct Item<'a> {
	id: i64,
	/// in track units; a box's is its middle
	x: f32,
	y: f32,
	cls: Option<&'a str>,
	conf: Option<f32>,
}

fn label(
	c: &mut Canvas,
	face: &Face,
	text: &str,
	x: f32,
	y: f32,
	w: f32,
	h: f32,
	l: &Look,
	u: f32,
	size: f32,
	spot: &str,
	tint: Option<&str>,
	alpha: f32,
) {
	if alpha <= 0.0 || text.is_empty() {
		return;
	}
	let spot = if spot == "auto" { if w > 90.0 * u && h > 50.0 * u { "center" } else { "right" } } else { spot };
	// with a backing, a label above is a tab sitting right on the box
	let tab = l.label_backing && spot == "above";
	let gap = if tab { 0.0 } else { 5.0 * u };
	let lines: Vec<&str> = text.split('\n').collect();
	let lh = size * 1.1;
	let block = lh * lines.len() as f32;
	let mut tx = x + w + gap;
	let mut ty = y + h / 2.0 - block / 2.0 + lh / 2.0;
	if spot == "center" {
		tx = x + w / 2.0 - face.measure(lines[0]) / 2.0;
	} else if spot == "above" || spot == "below" {
		tx = if tab { x + 3.0 * u } else { x };
		ty = if spot == "above" { y - gap - block + lh / 2.0 } else { y + h + gap + lh / 2.0 };
		// at the edge of the frame, it comes inside the box rather than off the picture
		if ty < lh / 2.0 {
			ty = y + gap + lh / 2.0;
		} else if ty + block - lh / 2.0 > c.h {
			ty = y + h - gap - block + lh / 2.0;
		}
		let widest = lines.iter().map(|s| face.measure(s)).fold(0.0, f32::max);
		tx = (3.0 * u).max(tx.min(c.w - widest - 3.0 * u));
	}
	for (i, line) in lines.iter().enumerate() {
		let ly = ty + i as f32 * lh;
		if l.label_backing {
			c.fill_rect(
				tx - 3.0 * u,
				ly - lh / 2.0,
				face.measure(line) + 6.0 * u,
				lh,
				colour(tint.unwrap_or(&l.label_back), alpha),
				BlendMode::SourceOver,
			);
		}
		let ink = if l.label_backing { l.label_color.as_str() } else { tint.unwrap_or(&l.label_color) };
		c.text(face, line, tx, ly, colour(ink, alpha));
	}
}

/// The writing in one corner of the frame, with a soft shadow so it reads over anything.
fn hud(c: &mut Canvas, s: &Scene, l: &Look, u: f32) {
	if l.hud.trim().is_empty() {
		return;
	}
	let hs = l.hud_size * u;
	let pad = 44.0 * u;
	let face = Face::new(&l.font, hs);
	let text = fill(&l.hud, s.hud);
	let lines: Vec<&str> = text.split('\n').collect();
	let right = l.hud_spot.ends_with("right");
	let bottom = l.hud_spot.starts_with("bottom");
	let n = lines.len() as f32;
	let mut pb = PathBuilder::new();
	let mut any = false;
	for (i, line) in lines.iter().enumerate() {
		let y = if bottom {
			c.h - pad - (n - 1.0 - i as f32) * hs * 1.25 - hs / 2.0
		} else {
			pad + hs / 2.0 + i as f32 * hs * 1.25
		};
		let x = if right { c.w - pad - face.measure(line) } else { pad };
		if let Some(p) = face.path(line, x, face.baseline(y)) {
			pb.push_path(&p);
			any = true;
		}
	}
	let Some(path) = (if any { pb.finish() } else { None }) else { return };
	// canvas's shadowBlur is twice the blur's standard deviation
	shadow(c, &path, 5.0 * u / 2.0, colour("#000000", 0.75));
	c.fill(&path, colour(&l.hud_color, 1.0), BlendMode::SourceOver);
}

/// A blurred copy of a shape, under where it'll be drawn.
fn shadow(c: &mut Canvas, path: &Path, sigma: f32, ink: Color) {
	let b = path.bounds();
	let margin = (sigma * 3.0).ceil() + 2.0;
	let (x0, y0) = ((b.left() - margin).floor().max(0.0), (b.top() - margin).floor().max(0.0));
	let (x1, y1) = ((b.right() + margin).ceil().min(c.w), (b.bottom() + margin).ceil().min(c.h));
	let (w, h) = ((x1 - x0) as u32, (y1 - y0) as u32);
	let Some(mut mask) = Pixmap::new(w.max(1), h.max(1)) else { return };
	let mut paint = Paint::default();
	paint.set_color(ink);
	paint.anti_alias = true;
	mask.fill_path(path, &paint, FillRule::Winding, Transform::from_translate(-x0, -y0), None);
	blur(mask.data_mut(), w as usize, h as usize, sigma);
	c.pm.draw_pixmap(x0 as i32, y0 as i32, mask.as_ref(), &PixmapPaint::default(), Transform::identity(), None);
}

/// Three box blurs each way, which comes out close to a gaussian. Works on premultiplied RGBA.
fn blur(data: &mut [u8], w: usize, h: usize, sigma: f32) {
	if sigma < 0.5 || w == 0 || h == 0 {
		return;
	}
	let r = (((12.0 * sigma * sigma / 3.0 + 1.0).sqrt() - 1.0) / 2.0).round().max(1.0) as usize;
	let mut buf: Vec<f32> = data.iter().map(|&v| v as f32).collect();
	let mut tmp = vec![0.0f32; buf.len()];
	for _ in 0..3 {
		pass(&buf, &mut tmp, w, h, r, true);
		pass(&tmp, &mut buf, w, h, r, false);
	}
	for (d, v) in data.iter_mut().zip(buf) {
		*d = v.round().clamp(0.0, 255.0) as u8;
	}
}

fn pass(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize, across: bool) {
	let (len, lines) = if across { (w, h) } else { (h, w) };
	let at = |line: usize, i: usize| if across { (line * w + i) * 4 } else { (i * w + line) * 4 };
	let n = (2 * r + 1) as f32;
	for line in 0..lines {
		for ch in 0..4 {
			let mut sum = 0.0;
			for i in 0..=r.min(len - 1) {
				sum += src[at(line, i) + ch];
			}
			for i in 0..len {
				dst[at(line, i) + ch] = sum / n;
				if i + r + 1 < len {
					sum += src[at(line, i + r + 1) + ch];
				}
				if i >= r {
					sum -= src[at(line, i - r) + ch];
				}
			}
		}
	}
}

/// Draw the look's overlay onto `frame` (opaque RGBA, which is also tiny-skia's premultiplied form).
pub fn draw(frame: &mut Frame, s: &Scene, l: &Look) {
	let (fw, fh) = (frame.w as f32, frame.h as f32);
	let Some(pm) = PixmapMut::from_bytes(&mut frame.rgba, frame.w, frame.h) else { return };
	let c = &mut Canvas { pm, w: fw, h: fh };
	let m = fw.min(fh);
	let u = m / 1080.0;
	let words: Vec<&str> = if l.words.contains("{tag}") || l.template.contains("{tag}") {
		l.words.split(',').map(str::trim).filter(|w| !w.is_empty()).collect()
	} else {
		Vec::new()
	};

	if l.link_mode == "objects" {
		object_links(c, s, l, m, u);
	} else if l.link_mode != "off" {
		links(c, s, l, m, u);
	}
	trails(c, s, l, m, u);
	patches(c, s, l, m, u);

	let size = l.label_size * u;
	let face = Face::new(&l.font, size);

	let sureness = |t: &Item, score: f32| t.conf.unwrap_or((score / 2.0 + 0.3).min(0.99));
	let tokens = |t: &Item, w: f32, h: f32, speed: f32, age: u32, score: f32, k: &str| -> Option<String> {
		Some(match k {
			"id" => t.id.to_string(),
			"x" => ((t.x * m).round() as i64).to_string(),
			"y" => ((t.y * m).round() as i64).to_string(),
			"w" => (w.round() as i64).to_string(),
			"h" => (h.round() as i64).to_string(),
			"age" => age.to_string(),
			"v" => format!("{:.1}", speed * m),
			"conf" => format!("{:.2}", sureness(t, score)),
			"pct" => format!("{}%", (sureness(t, score) * 100.0).round() as i64),
			"class" => t.cls.unwrap_or("subject").to_string(),
			"CLASS" => t.cls.unwrap_or("subject").to_uppercase(),
			"tag" => match t.cls {
				Some(c) => c.to_uppercase(),
				None if !words.is_empty() => {
					words[((hash(t.id, 8) * words.len() as f32) as usize).min(words.len() - 1)].to_string()
				}
				None => String::new(),
			},
			_ => return None,
		})
	};

	// points on something the detector named: a few per object, or they bury the box's own label,
	// and the limit closes in as the detector takes over, rather than all at once
	let b = s.blend;
	let room = (1.0 - b).powi(2) * 40.0;
	let named: &[ObjectTrack] = if b > 0.0 { s.objects } else { &[] };
	let host_of =
		|t: &Track| named.iter().position(|o| t.x >= o.x && t.x <= o.x + o.w && t.y >= o.y && t.y <= o.y + o.h);
	let limited = |seen: &mut Vec<u32>, host: Option<usize>, most: f32| -> bool {
		let Some(hh) = host else { return false };
		if seen.len() < named.len() {
			seen.resize(named.len(), 0);
		}
		if seen[hh] as f32 >= most {
			return true;
		}
		seen[hh] += 1;
		false
	};

	// strokes first, then the numbers on top, so a box never covers a number
	let width = l.line_width * u;
	let big = if l.region_style == "same" { l.box_style.as_str() } else { l.region_style.as_str() };
	let mut small = Pens::new();
	let mut boxed: Vec<u32> = Vec::new();
	for t in s.tracks {
		if hash(t.id, 3) >= l.boxes {
			continue;
		}
		let host = host_of(t);
		if limited(&mut boxed, host, l.per_object + 1.0 + room) {
			continue;
		}
		let (w, h) = box_of(t, l, u);
		let hobj = host.map(|i| &named[i]);
		let tint = tint_of(l, hobj.map_or(t.id, |o| o.id), hobj.map(|o| o.cls.as_str())).unwrap_or(&l.color);
		stroke_box(small.pen(tint), t.x * m - w / 2.0, t.y * m - h / 2.0, w, h, &l.box_style, fw, fh);
	}
	for (colour, path) in small.finish() {
		paint(c, &path, &l.box_style, &colour, l.opacity, width, u);
	}
	if b < 1.0 {
		let mut moving = Pens::new();
		for r in s.regions {
			stroke_box(
				moving.pen(tint_of(l, r.id, None).unwrap_or(&l.color)),
				r.x * m,
				r.y * m,
				r.w * m,
				r.h * m,
				big,
				fw,
				fh,
			);
		}
		for (colour, path) in moving.finish() {
			paint(c, &path, big, &colour, l.opacity * (1.0 - b), width, u);
		}
	}
	for o in named {
		let mut pb = PathBuilder::new();
		stroke_box(&mut pb, o.x * m, o.y * m, o.w * m, o.h * m, big, fw, fh);
		if let Some(p) = pb.finish() {
			paint(
				c,
				&p,
				big,
				tint_of(l, o.id, Some(&o.cls)).unwrap_or(&l.color),
				l.opacity * b * presence(o),
				width,
				u,
			);
		}
	}

	// the big boxes from movement are labelled as the points are; the detector's say what they are
	if b < 1.0 && l.labels > 0.0 && !l.template.is_empty() {
		for r in s.regions {
			let (w, h) = (r.w * m, r.h * m);
			let it = Item { id: r.id, x: r.x + r.w / 2.0, y: r.y + r.h / 2.0, cls: None, conf: None };
			let text = fill(&l.template, &|k| tokens(&it, w, h, 0.0, 0, 1.0, k));
			label(c, &face, &text, r.x * m, r.y * m, w, h, l, u, size, &l.label_spot, tint_of(l, r.id, None), 1.0 - b);
		}
	}
	if !l.region_template.is_empty() {
		for o in named {
			let (w, h) = (o.w * m, o.h * m);
			let it = Item { id: o.id, x: o.x + o.w / 2.0, y: o.y + o.h / 2.0, cls: Some(&o.cls), conf: Some(o.conf) };
			let text = fill(&l.region_template, &|k| tokens(&it, w, h, o.vx.hypot(o.vy), o.hits, 1.0, k));
			label(
				c,
				&face,
				&text,
				o.x * m,
				o.y * m,
				w,
				h,
				l,
				u,
				size,
				&l.region_spot,
				tint_of(l, o.id, Some(&o.cls)),
				b * presence(o),
			);
		}
	}

	if l.labels > 0.0 && !l.template.is_empty() {
		let mut labelled: Vec<u32> = Vec::new();
		for t in s.tracks {
			if hash(t.id, 4) >= l.labels {
				continue;
			}
			let host = host_of(t);
			if limited(&mut labelled, host, l.per_object + room) {
				continue;
			}
			let (w, h) = box_of(t, l, u);
			let has_box = hash(t.id, 3) < l.boxes;
			// a point with no box of its own: the number sits right against it
			let (bw, bh) = if has_box { (w, h) } else { (2.0 * u, 2.0 * u) };
			let hobj = host.map(|i| &named[i]);
			// on a named object it says what that object is, not a word picked at random
			let said = match hobj {
				Some(o) if b > 0.5 => Item { id: t.id, x: t.x, y: t.y, cls: Some(&o.cls), conf: Some(o.conf) },
				_ => Item { id: t.id, x: t.x, y: t.y, cls: None, conf: None },
			};
			let text = fill(&l.template, &|k| tokens(&said, w, h, t.speed, t.age, t.score, k));
			let tint = tint_of(l, hobj.map_or(t.id, |o| o.id), hobj.map(|o| o.cls.as_str()));
			label(
				c,
				&face,
				&text,
				t.x * m - bw / 2.0,
				t.y * m - bh / 2.0,
				bw,
				bh,
				l,
				u,
				size,
				&l.label_spot,
				tint,
				1.0,
			);
		}
	}

	hud(c, s, l, u);
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn templates_fill_like_the_site() {
		let get = |k: &str| match k {
			"id" => Some("37001".to_string()),
			"class" => Some("car".to_string()),
			_ => None,
		};
		assert_eq!(fill("{class} {id}", &get), "car 37001");
		assert_eq!(fill("{nope} {id}!", &get), "{nope} 37001!");
		assert_eq!(fill("a { b } {", &get), "a { b } {");
		assert_eq!(fill("", &get), "");
	}

	#[test]
	fn every_preset_draws() {
		let low = Frame::new(64, 36);
		let tracks: Vec<Track> = (0..40)
			.map(|i| Track {
				id: 37000 + i,
				x: 0.1 + (i as f32 * 0.037) % 1.6,
				y: 0.1 + (i as f32 * 0.051) % 0.8,
				age: 5,
				miss: 0,
				score: 0.8,
				speed: 0.002,
				path: Some(vec![0.1, 0.1, 0.12, 0.11, 0.14, 0.13]),
			})
			.collect();
		let regions = vec![Region { id: 37100, x: 0.3, y: 0.3, w: 0.2, h: 0.3, miss: 0, path: None }];
		let objects = vec![ObjectTrack {
			id: 37200,
			x: 0.6,
			y: 0.2,
			w: 0.3,
			h: 0.5,
			miss: 0,
			cls: "person".into(),
			conf: 0.9,
			hits: 3,
			vx: 0.0,
			vy: 0.0,
			frames: 0,
			votes: vec![],
			shown: true,
			fade: 1.0,
			path: Some(vec![0.7, 0.7, 0.72, 0.71]),
		}];
		let hud = |k: &str| Some(format!("<{k}>"));
		for name in Look::presets() {
			let l = Look::preset(name).unwrap();
			// looks that are all picture (no words on the things, no HUD, no trails), or typed out in
			// words, may add nothing over it
			if l.glyphs == "words" || (l.region_template.trim().is_empty() && l.hud.trim().is_empty() && l.trail == 0.0)
			{
				continue;
			}
			let mut f = Frame::new(640, 360);
			let s = Scene { tracks: &tracks, regions: &regions, objects: &objects, blend: 0.6, low: &low, hud: &hud };
			draw(&mut f, &s, &l);
			assert!(f.rgba.chunks_exact(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0), "{name} drew nothing");
			assert!(f.rgba.chunks_exact(4).all(|p| p[3] == 255), "{name} left the frame see-through");
		}
	}
}
