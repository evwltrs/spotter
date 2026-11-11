//! spotter's engine: finds what moves in a video and follows it, recognises what things are with a
//! detector, and draws a look over the picture. The same code renders the app's preview and the
//! exported file, so what you see is what you get.

pub mod background;
pub mod beats;
pub mod depth;
pub mod detector;
pub mod echo;
pub mod edit;
pub mod engine;
pub mod export;
pub mod follow;
pub mod frame;
pub mod glitch;
pub mod glyphs;
pub mod gpu;
pub mod grade;
pub mod insets;
pub mod interp;
pub mod look;
pub mod matte;
pub mod model;
pub mod objects;
pub mod onnx;
pub mod outline;
pub mod overlay;
pub mod paths;
pub mod perf;
pub mod project;
pub mod realmosh;
pub mod recall;
pub mod scene;
pub mod songs;
pub mod sources;
pub mod subject;
pub mod tape;
pub mod text;
pub mod tracking;
pub mod upscale;
pub mod video;

pub use engine::Engine;
pub use frame::{Frame, Rect};
pub use look::Look;

/// "1 shot", "2 shots": a count with its word.
pub fn counted(n: usize, what: &str) -> String {
	if n == 1 { format!("1 {what}") } else { format!("{n} {what}s") }
}

/// A program spotter starts, with no window of its own (a terminal flashing up on Windows).
pub fn quiet(c: &mut std::process::Command) -> &mut std::process::Command {
	#[cfg(windows)]
	{
		use std::os::windows::process::CommandExt;
		const CREATE_NO_WINDOW: u32 = 0x0800_0000;
		c.creation_flags(CREATE_NO_WINDOW);
	}
	c
}
