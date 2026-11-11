//! Insets: magnified live crops of the things followed, docked in a row or column along one edge of
//! the frame, each with a caption and a hairline out to the thing itself. Each thing keeps its cell
//! for as long as it's followed, so cells don't swap places.

use crate::Frame;
use crate::Rect;
use crate::look::Look;
use crate::overlay::{colour, fill};
use crate::text::Face;
use tiny_skia::{FillRule, LineCap, Paint, PathBuilder, PixmapMut, Stroke, Transform};

/// A thing to show: its number, what it is, how sure, and where (output pixels).
pub struct Thing {
	pub id: i64,
	pub cls: String,
	pub conf: f32,
	pub rect: Rect,
}

/// Whether a look shows insets.
pub fn active(l: &Look) -> bool {
	l.insets >= 1.0
}

#[derive(Default)]
pub struct Insets {
	/// which thing has each cell
	slots: Vec<Option<i64>>,
	cell: Frame,
}

impl Insets {
	pub fn clear(&mut self) {
		self.slots.clear();
	}

	/// Draw the insets into `out`, cropping from `clean`, the picture before anything was drawn on it.
	pub fn draw(&mut self, out: &mut Frame, clean: &Frame, things: &[Thing], l: &Look) {
		let (fw, fh) = (out.w as f32, out.h as f32);
		let short = fw.min(fh);
		let u = short / 1080.0;
		let size = (l.inset_size.clamp(0.05, 0.5) * short).round().max(16.0);
		let margin = (0.035 * short).round();
		let gap = (0.02 * short).round();
		let caption = (l.label_size.max(10.0) * u * 0.9).max(9.0);
		let vertical = l.inset_side == "left" || l.inset_side == "right";
		// as many cells as fit along the side
		let room = if vertical { fh - 2.0 * margin } else { fw - 2.0 * margin };
		let per = size + gap + caption * 1.6;
		let most = ((room + gap) / per).floor().max(0.0) as usize;
		let n = (l.insets.round() as usize).min(most);
		self.slots.resize(n, None);
		self.slots.truncate(n);
		// keep cells for things still here, give empty ones to newcomers (oldest numbers first)
		for s in &mut self.slots {
			if s.is_some_and(|id| !things.iter().any(|t| t.id == id)) {
				*s = None;
			}
		}
		let mut waiting: Vec<&Thing> = things.iter().filter(|t| !self.slots.contains(&Some(t.id))).collect();
		waiting.sort_by_key(|t| t.id);
		for s in self.slots.iter_mut().filter(|s| s.is_none()) {
			if waiting.is_empty() {
				break;
			}
			*s = Some(waiting.remove(0).id);
		}
		let face = Face::new(&l.font, caption);
		let ink = colour(&l.inset_color, 1.0);
		let stroke = Stroke { width: (1.2 * u).max(1.0), line_cap: LineCap::Square, ..Default::default() };
		for (i, slot) in self.slots.iter().enumerate() {
			let Some(t) = slot.and_then(|id| things.iter().find(|t| t.id == id)) else { continue };
			let along = margin + i as f32 * per;
			let (x, y) = match l.inset_side.as_str() {
				"left" => (margin, along),
				"top" => (along, margin + caption * 1.6),
				"bottom" => (along, fh - margin - size),
				_ => (fw - margin - size, along),
			};
			// the crop: the thing with a little room, closer in with zoom, square
			let r = t.rect;
			let side = (r.w.max(r.h) * 1.25 / l.inset_zoom.max(0.25)).max(8.0);
			let (cx, cy) = (r.x + r.w / 2.0, r.y + r.h / 2.0);
			let win = [
				(cx - side / 2.0).clamp(0.0, (fw - side).max(0.0)),
				(cy - side / 2.0).clamp(0.0, (fh - side).max(0.0)),
				side.min(fw),
				side.min(fh),
			];
			let s = size as u32;
			if self.cell.w != s || self.cell.h != s {
				self.cell = Frame::new(s, s);
			}
			clean.crop_into(win, &mut self.cell);
			let (ox, oy) = (x as usize, y as usize);
			for row in 0..s as usize {
				let yy = oy + row;
				if yy >= out.h as usize {
					break;
				}
				let width = (s as usize).min(out.w as usize - ox.min(out.w as usize));
				let dst = (yy * out.w as usize + ox) * 4;
				out.rgba[dst..dst + width * 4].copy_from_slice(&self.cell.rgba[row * s as usize * 4..][..width * 4]);
			}
			let Some(mut pm) = PixmapMut::from_bytes(&mut out.rgba, out.w, out.h) else { return };
			let mut paint = Paint::default();
			paint.set_color(ink);
			paint.anti_alias = true;
			// the frame, and a hairline from its nearest edge to the thing
			let mut pb = PathBuilder::new();
			if let Some(rect) = tiny_skia::Rect::from_xywh(x, y, size, size) {
				pb.push_rect(rect);
			}
			let (ax, ay) = match l.inset_side.as_str() {
				"left" => (x + size, y + size / 2.0),
				"top" => (x + size / 2.0, y + size),
				"bottom" => (x + size / 2.0, y),
				_ => (x, y + size / 2.0),
			};
			let (bx, by) = (ax.clamp(r.x, r.x + r.w), ay.clamp(r.y, r.y + r.h));
			pb.move_to(ax, ay);
			pb.line_to(bx, by);
			if let Some(path) = pb.finish() {
				pm.stroke_path(&path, &paint, &stroke, Transform::identity(), None);
			}
			// the caption, above the cell
			let words = fill(&l.inset_template, &|k: &str| {
				Some(match k {
					"id" => t.id.to_string(),
					"class" => t.cls.clone(),
					"CLASS" => t.cls.to_uppercase(),
					// movement has no sureness to give
					"pct" if t.conf.is_nan() => String::new(),
					"pct" => format!("{}%", (t.conf * 100.0).round() as i64),
					"w" => (r.w.round() as i64).to_string(),
					"h" => (r.h.round() as i64).to_string(),
					"x" => ((r.x + r.w / 2.0).round() as i64).to_string(),
					"y" => ((r.y + r.h / 2.0).round() as i64).to_string(),
					"zoom" => format!("{:.1}x", size / side),
					_ => return None,
				})
			});
			if let Some(p) = face.path(&words, x, face.baseline(y - caption * 0.8)) {
				pm.fill_path(&p, &paint, FillRule::Winding, Transform::identity(), None);
			}
		}
	}
}
