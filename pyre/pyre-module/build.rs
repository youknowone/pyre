//! Compile the `_ctypes` SEH fence. `__try`/`__except` is the only thing
//! that reaches a structured exception and no Rust compiler emits one, so
//! the fence a foreign call is made inside (`src/module/_ctypes/seh.rs`)
//! is a C translation unit of its own.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.ends_with("-pc-windows-msvc") {
        println!("cargo:rerun-if-changed=src/module/_ctypes/seh.c");
        cc::Build::new()
            .file("src/module/_ctypes/seh.c")
            .compile("pyre_ctypes_seh");
    }
}
