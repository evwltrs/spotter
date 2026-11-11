//! Keeping the rest of the computer safe from spotter, whatever it gets up to. Set up as it starts:
//!
//! - its heavy work (the pool every picture's drawn on, and playback) runs at a little under normal
//!   priority and leaves two of the CPU's threads free, so the desktop and other programs
//!   always come first;
//! - the GPU's never kept busy all the time (gpu.rs rests it for a share of each job);
//! - on Windows it's put in a job with ceilings: on its own memory, on its memory and everything it
//!   starts together (half of what there is), and on how many programs it can have running at once
//!   (64), so if it ever runs away, it's stopped (or can't start more) rather than the whole machine
//!   grinding on the page file. Many small programs at once can do that as surely as one big one.
//!   Whatever it started goes when it does, rather than carrying on in the background;
//! - it lets go of what it's kept (`Engine::shed`) well before that, when it's using a lot.

/// How much of the time the window may keep the GPU busy, in percent.
const GPU_SHARE: u32 = 85;

/// Past this much memory (bytes), spotter lets go of what it's kept to save time.
pub const SHED_AT: u64 = 6 << 30;

/// All of it, at the start.
pub fn guard() {
	let n = std::thread::available_parallelism().map_or(4, |n| n.get());
	let _ = rayon::ThreadPoolBuilder::new()
		.num_threads(n.saturating_sub(2).max(2))
		.thread_name(|i| format!("work {i}"))
		.start_handler(|_| lower_priority())
		.build_global();
	spotter_core::gpu::set_share(GPU_SHARE);
	#[cfg(windows)]
	win::ceiling();
}

/// This thread, at a little under normal priority.
pub fn lower_priority() {
	#[cfg(windows)]
	unsafe {
		use windows_sys::Win32::System::Threading::{
			GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
		};
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
	}
}

/// How much memory spotter has taken for itself (bytes), where that can be told.
pub fn memory() -> u64 {
	#[cfg(windows)]
	unsafe {
		use windows_sys::Win32::System::ProcessStatus::{
			K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
		};
		use windows_sys::Win32::System::Threading::GetCurrentProcess;
		let mut c: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
		c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
		if K32GetProcessMemoryInfo(
			GetCurrentProcess(),
			&mut c as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS,
			c.cb,
		) != 0
		{
			return c.PrivateUsage as u64;
		}
	}
	0
}

#[cfg(windows)]
mod win {
	use windows_sys::Win32::Foundation::CloseHandle;
	use windows_sys::Win32::System::JobObjects::{
		AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_ACTIVE_PROCESS, JOB_OBJECT_LIMIT_JOB_MEMORY,
		JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
		JobObjectExtendedLimitInformation, SetInformationJobObject,
	};
	use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
	use windows_sys::Win32::System::Threading::GetCurrentProcess;

	/// spotter in a job of its own: itself at most two fifths of the memory there is (12 GB at most),
	/// it and all it starts at most half, no more than 64 programs at once, and what it started ends
	/// with it. The job's handle stays open for as long as spotter runs.
	pub fn ceiling() {
		unsafe {
			let mut m: MEMORYSTATUSEX = std::mem::zeroed();
			m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
			let total = if GlobalMemoryStatusEx(&mut m) != 0 { m.ullTotalPhys } else { 16 << 30 };
			let limit = (total / 5 * 2).min(12 << 30);
			let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
			if job.is_null() {
				return;
			}
			let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
			info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY
				| JOB_OBJECT_LIMIT_JOB_MEMORY
				| JOB_OBJECT_LIMIT_ACTIVE_PROCESS
				| JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
			info.BasicLimitInformation.ActiveProcessLimit = 64;
			info.ProcessMemoryLimit = limit as usize;
			info.JobMemoryLimit = (total / 2) as usize;
			let set = SetInformationJobObject(
				job,
				JobObjectExtendedLimitInformation,
				&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION as *const core::ffi::c_void,
				std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
			);
			if set == 0 || AssignProcessToJobObject(job, GetCurrentProcess()) == 0 {
				CloseHandle(job);
			}
		}
	}
}
