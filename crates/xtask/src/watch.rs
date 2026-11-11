//! spotter-app with a watch on it: a line a second of what it and the computer are doing, written to
//! spotter's folder as watch.log as it goes (so it's there even if the computer has to be switched
//! off), and its own messages to spotter-errors.log beside it. Stops it if it takes more than 10 GB or
//! the computer gets down to its last 3 GB free.

use anyhow::Result;

#[cfg(windows)]
const GB: f64 = 1073741824.0;

#[cfg(not(windows))]
pub fn run(_: &[String]) -> Result<()> {
	anyhow::bail!("watch reads Windows' counters, so it's Windows only")
}

#[cfg(windows)]
pub fn run(args: &[String]) -> Result<()> {
	use std::io::Write;
	use std::os::windows::io::AsRawHandle;
	let exe = crate::target().join("release").join("spotter-app.exe");
	let dir = std::path::PathBuf::from(std::env::var("LOCALAPPDATA")?).join("spotter");
	std::fs::create_dir_all(&dir)?;
	let log_path = dir.join("watch.log");
	let mut log = std::fs::OpenOptions::new().create(true).append(true).open(&log_path)?;
	let errors = std::fs::File::create(dir.join("spotter-errors.log"))?;
	let mut child = std::process::Command::new(&exe).args(args).stderr(errors).spawn()?;
	let pid = child.id();
	writeln!(log, "--- spotter started {}, pid {pid}", win::now(true))?;
	println!("watching spotter (pid {pid}); the log is {}", log_path.display());
	let counters = win::Counters::new(pid);
	let cores = std::thread::available_parallelism().map_or(1, |n| n.get()) as f64;
	let handle = child.as_raw_handle();
	let (mut last_cpu, mut last_at) = (win::cpu_time(handle), std::time::Instant::now());
	let code = loop {
		std::thread::sleep(std::time::Duration::from_millis(700));
		if let Some(status) = child.try_wait()? {
			break status.code();
		}
		let (cpu_time, at) = (win::cpu_time(handle), std::time::Instant::now());
		let cpu = (cpu_time - last_cpu) / at.duration_since(last_at).as_secs_f64() * 100.0 / cores;
		(last_cpu, last_at) = (cpu_time, at);
		let ram = win::private_bytes(handle) as f64 / GB;
		let free = win::free_bytes() as f64 / GB;
		let (threads, helpers) = win::threads_and_helpers(pid);
		let (gpu, vram, total) = counters.read();
		let or_unknown = |v: Option<f64>, width: usize, places: usize| {
			v.map_or_else(|| format!("{:>width$}", "?"), |v| format!("{v:>width$.places$}"))
		};
		writeln!(
			log,
			"{}  spotter ram {ram:5.2} GB cpu {cpu:3.0}%  gpu {}%  vram {} GB  threads {threads:3}  ffmpeg {helpers}  |  pc cpu {}%  free {free:5.1} GB",
			win::now(false),
			or_unknown(gpu, 4, 0),
			or_unknown(vram.map(|v| v / GB), 5, 2),
			or_unknown(total, 3, 0),
		)?;
		if ram > 10.0 || free < 3.0 {
			writeln!(log, "!!! stopped spotter: ram {ram:.2} GB, free {free:.1} GB")?;
			println!("stopped spotter: ram {ram:.2} GB, free {free:.1} GB");
			child.kill()?;
			break child.wait()?.code();
		}
	};
	writeln!(log, "--- spotter ended {}, code {}", win::now(true), code.map_or("?".into(), |c| c.to_string()))?;
	println!("spotter ended; the log is {}", log_path.display());
	Ok(())
}

#[cfg(windows)]
mod win {
	use std::ffi::c_void;
	use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, INVALID_HANDLE_VALUE, SYSTEMTIME};
	use windows_sys::Win32::System::Diagnostics::ToolHelp::{
		CreateToolhelp32Snapshot, PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPPROCESS,
	};
	use windows_sys::Win32::System::Performance::{
		PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_HCOUNTER, PDH_HQUERY, PDH_MORE_DATA, PdhAddEnglishCounterW,
		PdhCloseQuery, PdhCollectQueryData, PdhGetFormattedCounterArrayW, PdhOpenQueryW,
	};
	use windows_sys::Win32::System::ProcessStatus::{
		K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
	};
	use windows_sys::Win32::System::SystemInformation::{GetLocalTime, GlobalMemoryStatusEx, MEMORYSTATUSEX};
	use windows_sys::Win32::System::Threading::GetProcessTimes;

	/// The local time, with the date or without.
	pub fn now(date: bool) -> String {
		let mut t: SYSTEMTIME = unsafe { std::mem::zeroed() };
		unsafe { GetLocalTime(&mut t) };
		let time = format!("{:02}:{:02}:{:02}", t.wHour, t.wMinute, t.wSecond);
		if date { format!("{:04}-{:02}-{:02} {time}", t.wYear, t.wMonth, t.wDay) } else { time }
	}

	/// Seconds of CPU the process has had, all its threads together.
	pub fn cpu_time(process: *mut c_void) -> f64 {
		let mut times: [FILETIME; 4] = unsafe { std::mem::zeroed() };
		let [created, exited, kernel, user] = &mut times;
		if unsafe { GetProcessTimes(process, created, exited, kernel, user) } == 0 {
			return 0.0;
		}
		let ticks = |f: &FILETIME| ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) as f64 / 1e7;
		ticks(kernel) + ticks(user)
	}

	/// Memory the process has taken for itself.
	pub fn private_bytes(process: *mut c_void) -> u64 {
		let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
		c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
		let p = &mut c as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS;
		if unsafe { K32GetProcessMemoryInfo(process, p, c.cb) } == 0 { 0 } else { c.PrivateUsage as u64 }
	}

	/// Memory the computer has free.
	pub fn free_bytes() -> u64 {
		let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
		m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
		if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 { 0 } else { m.ullAvailPhys }
	}

	/// The process's threads, and how many ffmpeg and yt-dlp programs are running.
	pub fn threads_and_helpers(pid: u32) -> (u32, usize) {
		let (mut threads, mut helpers) = (0, 0);
		let snap = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
		if snap == INVALID_HANDLE_VALUE {
			return (0, 0);
		}
		let mut e: PROCESSENTRY32W = unsafe { std::mem::zeroed() };
		e.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
		let mut more = unsafe { Process32FirstW(snap, &mut e) } != 0;
		while more {
			if e.th32ProcessID == pid {
				threads = e.cntThreads;
			}
			let len = e.szExeFile.iter().position(|&c| c == 0).unwrap_or(e.szExeFile.len());
			let name = String::from_utf16_lossy(&e.szExeFile[..len]).to_ascii_lowercase();
			if name == "ffmpeg.exe" || name == "yt-dlp.exe" {
				helpers += 1;
			}
			more = unsafe { Process32NextW(snap, &mut e) } != 0;
		}
		unsafe { CloseHandle(snap) };
		(threads, helpers)
	}

	/// Windows' performance counters for the GPU's engines and memory as the process uses them, and
	/// the whole computer's CPU.
	pub struct Counters {
		query: PDH_HQUERY,
		gpu: PDH_HCOUNTER,
		vram: PDH_HCOUNTER,
		total: PDH_HCOUNTER,
	}

	impl Counters {
		pub fn new(pid: u32) -> Counters {
			let mut c: Counters = unsafe { std::mem::zeroed() };
			unsafe { PdhOpenQueryW(std::ptr::null(), 0, &mut c.query) };
			let add = |path: String, counter: &mut PDH_HCOUNTER| {
				let wide: Vec<u16> = path.encode_utf16().chain([0]).collect();
				unsafe { PdhAddEnglishCounterW(c.query, wide.as_ptr(), 0, counter) };
			};
			add(format!(r"\GPU Engine(pid_{pid}*)\Utilization Percentage"), &mut c.gpu);
			add(format!(r"\GPU Process Memory(pid_{pid}*)\Dedicated Usage"), &mut c.vram);
			add(r"\Processor(_Total)\% Processor Time".into(), &mut c.total);
			unsafe { PdhCollectQueryData(c.query) };
			c
		}

		/// The GPU's use (%), its memory (bytes) and the computer's CPU (%) since the last read.
		pub fn read(&self) -> (Option<f64>, Option<f64>, Option<f64>) {
			if unsafe { PdhCollectQueryData(self.query) } != 0 {
				return (None, None, None);
			}
			(sum(self.gpu), sum(self.vram), sum(self.total))
		}
	}

	impl Drop for Counters {
		fn drop(&mut self) {
			unsafe { PdhCloseQuery(self.query) };
		}
	}

	/// A counter's values added up over all its instances.
	fn sum(counter: PDH_HCOUNTER) -> Option<f64> {
		if counter.is_null() {
			return None;
		}
		let (mut bytes, mut count) = (0u32, 0u32);
		let asked = unsafe {
			PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut bytes, &mut count, std::ptr::null_mut())
		};
		if asked != PDH_MORE_DATA {
			return if asked == 0 { Some(0.0) } else { None };
		}
		// the items, then the instance names they point into
		let item = std::mem::size_of::<PDH_FMT_COUNTERVALUE_ITEM_W>();
		let mut buf = vec![0u64; (bytes as usize).div_ceil(8).max(item.div_ceil(8))];
		let items = buf.as_mut_ptr() as *mut PDH_FMT_COUNTERVALUE_ITEM_W;
		if unsafe { PdhGetFormattedCounterArrayW(counter, PDH_FMT_DOUBLE, &mut bytes, &mut count, items) } != 0 {
			return None;
		}
		let items = unsafe { std::slice::from_raw_parts(items, count as usize) };
		Some(items.iter().map(|i| unsafe { i.FmtValue.Anonymous.doubleValue }).sum())
	}
}
