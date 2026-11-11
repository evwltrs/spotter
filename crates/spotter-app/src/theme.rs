//! The site's Gruvbox, DM Mono, and the kit set up square. Each colour means one thing:
//! yellow is the current thing (the look, the tab, the playhead, what's picked), orange is changed from
//! the look, aqua is what the detector found, red is cut, error or stop.

use crate::*;

pub(crate) fn hex(h: &str) -> Color {
	rune::color::hex(h).expect("a colour")
}

/// the window, the timeline
pub(crate) fn bg() -> Color {
	hex("#1d2021")
}

/// the rail, the inspector
pub(crate) fn surface() -> Color {
	hex("#282828")
}

/// controls, rows hovered
pub(crate) fn raised() -> Color {
	hex("#32302f")
}

/// dividers, a slider's track
pub(crate) fn line() -> Color {
	hex("#3c3836")
}

/// a control hovered, the chosen option, borders
pub(crate) fn faint() -> Color {
	hex("#504945")
}

/// off, disabled, not looked at yet
pub(crate) fn mute() -> Color {
	hex("#7c6f64")
}

/// labels, headers
pub(crate) fn dim() -> Color {
	hex("#bdae93")
}

/// values, the chosen option's words
pub(crate) fn ink() -> Color {
	hex("#ebdbb2")
}

/// the current thing
pub(crate) fn accent() -> Color {
	hex("#fabd2f")
}

/// changed from the look
pub(crate) fn orange() -> Color {
	hex("#fe8019")
}

/// what the detector found
pub(crate) fn aqua() -> Color {
	hex("#8ec07c")
}

/// cut, error, stop
pub(crate) fn red() -> Color {
	hex("#fb4934")
}

pub(crate) fn mono(size: f32) -> TextStyle {
	TextStyle::new(size, 400.0).family(Family::Named("DM Mono"))
}

/// numbers that change, so they don't jiggle
pub(crate) fn digits(size: f32) -> TextStyle {
	mono(size).features("'tnum' 1")
}

pub(crate) fn strong(size: f32) -> TextStyle {
	TextStyle::new(size, 500.0).family(Family::Named("DM Mono"))
}

pub(crate) fn kit() -> &'static Kit {
	static K: OnceLock<Kit> = OnceLock::new();
	K.get_or_init(|| {
		let mut k = Kit::from_palette(rune_kit::PaletteSeed { bg: bg(), fg: ink(), accent: accent() });
		k.radius = Radius { xs: 0.0, sm: 0.0, md: 0.0, lg: 0.0, xl: 0.0 };
		k
	})
}

/// The fixed sizes of the window's parts, in points.
pub(crate) const TOP: f32 = 28.0;
pub(crate) const RAIL: f32 = 236.0;
pub(crate) const INSPECTOR: f32 = 336.0;
pub(crate) const STATUS: f32 = 22.0;
/// a row or a control
pub(crate) const ROW: f32 = 22.0;
