//! The icon and version Windows shows for the program. Without a resource compiler (the Windows
//! SDK's rc.exe, or windres) it's built without them.

fn main() {
	if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
		return;
	}
	println!("cargo:rerun-if-changed=../../assets/icon.ico");
	let mut r = winresource::WindowsResource::new();
	r.set_icon("../../assets/icon.ico")
		.set("FileDescription", "spotter, from the command line")
		.set("ProductName", "spotter");
	if let Err(e) = r.compile() {
		println!("cargo:warning=built without its icon and version: {e}");
	}
}
