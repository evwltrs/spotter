//! Prints the beats spotter finds in a clip's sound:
//!   cargo run -p spotter-core --release --example beats -- clip.mp4 [more]
fn main() -> anyhow::Result<()> {
	for clip in std::env::args().skip(1) {
		let b = spotter_core::beats::find(&clip)?;
		let gaps: Vec<f64> = b.windows(2).map(|w| w[1] - w[0]).collect();
		let mut sorted = gaps.clone();
		sorted.sort_by(f64::total_cmp);
		let median = sorted.get(sorted.len() / 2).copied().unwrap_or(0.0);
		println!(
			"{clip}: {} beats, median gap {:.2}s ({:.0} a minute): {}",
			b.len(),
			median,
			if median > 0.0 { 60.0 / median } else { 0.0 },
			b.iter().take(16).map(|t| format!("{t:.2}")).collect::<Vec<_>>().join(" ")
		);
	}
	Ok(())
}
