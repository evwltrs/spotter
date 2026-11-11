//! Saved work has to keep opening as it was saved: a project and one of your own looks as this
//! version writes them (in `tests/saved/`, written once with `SAVED_BLESS=1` and then left alone), and
//! a project from before projects said their version. A new layout adds files; it doesn't rewrite these.

use spotter_core::edit::{Edit, Keep, Music, Pick, Range, RangeMode, Zone, ZoneMode};
use spotter_core::look::Look;
use spotter_core::project::{Project, VERSION};
use std::path::PathBuf;

fn dir() -> PathBuf {
	PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("saved")
}

/// A preset, changed, with another look mixed in over it and that one changed too.
fn look() -> Look {
	let mut l = Look::preset("databend").unwrap();
	l.points = 12.0;
	l.follow = true;
	l.follow_zoom = 0.4;
	l.mix = "plates".into();
	l.mix_how = "over".into();
	l.mix_at = 1.0;
	l.mix_set.insert("insetZoom".into(), serde_json::json!(3.0));
	l
}

/// One of every mark, and a song.
fn edit() -> Edit {
	Edit {
		kinds: vec!["person".into(), "car".into()],
		keep: Keep::Picked,
		picks: vec![Pick { t: 1.25, x: 0.4, y: 0.6 }],
		drops: vec![Pick { t: 6.5, x: 0.8, y: 0.2 }],
		zones: vec![Zone { x: 0.0, y: 0.75, w: 0.3, h: 0.25, mode: ZoneMode::Ignore }],
		ranges: vec![
			Range { from: 2.0, to: 2.5, mode: RangeMode::Cut },
			Range { from: 4.0, to: 5.0, mode: RangeMode::Bare },
			Range { from: 5.0, to: 7.0, mode: RangeMode::Unmix },
		],
		music: Some(Music { path: "song.mp3".into(), from: 30.0, gain: 0.8, keep: 0.25 }),
	}
}

fn project() -> Project {
	Project {
		version: VERSION,
		source: "clip.mp4".into(),
		name: "clip".into(),
		preset: "databend".into(),
		look: serde_json::to_value(look()).unwrap(),
		edit: edit(),
		size: Some((1080, 1920)),
		upscale: true,
		model: "x365".into(),
		start: Some(1.5),
		end: Some(8.0),
	}
}

#[test]
fn saved_work_opens_as_it_was() {
	let (project_file, look_file) = (dir().join("project-1.spotter"), dir().join("mine-1.json"));
	if std::env::var_os("SAVED_BLESS").is_some() {
		project().write(&project_file).unwrap();
		std::fs::write(&look_file, serde_json::to_string_pretty(&look()).unwrap()).unwrap();
		return;
	}
	let p = Project::read(&project_file).unwrap();
	assert_eq!(p.version, 1);
	assert_eq!(p.look(), look());
	assert_eq!(p.edit, edit());
	assert_eq!(
		(p.size, p.upscale, p.model.as_str(), p.start, p.end),
		(Some((1080, 1920)), true, "x365", Some(1.5), Some(8.0))
	);
	let mine: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(look_file).unwrap()).unwrap();
	assert_eq!(Look::revive(&mine), look());
}

#[test]
fn a_project_from_before_versions_opens() {
	let p = Project::read(&dir().join("project-0.spotter")).unwrap();
	assert_eq!(p.version, 0);
	// what it gave, and the default for what it didn't
	assert_eq!(p.look().points, 9.0);
	assert_eq!(p.look().follow_zoom, Look::default().follow_zoom);
	assert_eq!((p.edit.picks.len(), p.start), (1, Some(1.5)));
}

#[test]
fn a_project_from_a_newer_spotter_is_refused() {
	let file = std::env::temp_dir().join(format!("spotter-newer-{}.spotter", std::process::id()));
	std::fs::write(&file, format!(r#"{{"version": {}, "source": "clip.mp4"}}"#, VERSION + 1)).unwrap();
	let e = Project::read(&file).unwrap_err().to_string();
	let _ = std::fs::remove_file(&file);
	assert!(e.contains("newer spotter"), "{e}");
}
