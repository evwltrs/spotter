//! What the window says as it goes, and what went wrong: it has no terminal, so its messages go to
//! spotter.log in spotter's folder (the last one kept as spotter.old.log once it's past 1 MB).

/// Send this run's messages to spotter.log, unless they already go somewhere (a terminal, a file).
pub fn start() {
	#[cfg(windows)]
	win::to_file();
	let default = std::panic::take_hook();
	std::panic::set_hook(Box::new(move |info| {
		default(info);
		// the window's own thread can't carry on: say so before it goes
		#[cfg(windows)]
		if std::thread::current().name() == Some("main") && !rune::tooling::is_headless() {
			win::tell(&format!("spotter stopped: {}\n\nspotter.log, in {}, has more.", said(info), folder()));
		}
	}));
}

fn folder() -> String {
	spotter_core::paths::home().display().to_string()
}

/// What a panic said.
pub fn said(info: &std::panic::PanicHookInfo) -> String {
	text(info.payload())
}

/// A caught panic's message.
pub fn text(p: &(dyn std::any::Any + Send)) -> String {
	p.downcast_ref::<&str>()
		.map(|s| s.to_string())
		.or_else(|| p.downcast_ref::<String>().cloned())
		.unwrap_or_else(|| "something went wrong".into())
}

#[cfg(windows)]
mod win {
	use std::os::windows::io::IntoRawHandle;
	use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
	use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, SetStdHandle};

	pub fn to_file() {
		let h = unsafe { GetStdHandle(STD_ERROR_HANDLE) };
		if !h.is_null() && h != INVALID_HANDLE_VALUE {
			return;
		}
		let dir = spotter_core::paths::home();
		let _ = std::fs::create_dir_all(&dir);
		let log = dir.join("spotter.log");
		if std::fs::metadata(&log).is_ok_and(|m| m.len() > 1 << 20) {
			let _ = std::fs::rename(&log, dir.join("spotter.old.log"));
		}
		let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(&log) else { return };
		unsafe { SetStdHandle(STD_ERROR_HANDLE, f.into_raw_handle() as _) };
		let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64());
		eprintln!("\nspotter {} started {} UTC", env!("CARGO_PKG_VERSION"), spotter_core::engine::date(now));
	}

	/// A message box, for when the window itself has gone.
	pub fn tell(what: &str) {
		use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MessageBoxW};
		let wide = |s: &str| s.encode_utf16().chain([0]).collect::<Vec<u16>>();
		unsafe {
			MessageBoxW(std::ptr::null_mut(), wide(what).as_ptr(), wide("spotter").as_ptr(), MB_OK | MB_ICONERROR)
		};
	}
}
