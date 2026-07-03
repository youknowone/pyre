//! Build step for the `wasm_vfs` feature.
//!
//! The browser/web wasm target has no filesystem, so the pure-Python stdlib
//! closure that `import re` transitively needs cannot be read from disk at
//! runtime.  When `wasm_vfs` is enabled this packs that closure into a single
//! lz4-compressed blob under `OUT_DIR`; `importing.rs` embeds the blob with
//! `include_bytes!` and decompresses it into an in-memory VFS at startup.
//!
//! When the feature is off (every native build) this returns immediately and
//! produces nothing.
//!
//! This build script runs on the host at compile time — it is not part of the
//! sandbox binary — so the sandbox compile-out fence (`ci/clippy-sandbox`) does
//! not apply to its host filesystem/env access.
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use std::path::Path;

/// Pure-Python files reachable from `import re`.  The C-level dependencies
/// (`_sre`, `_abc`, `_weakref`, `itertools`, `_collections`, `_thread`,
/// `operator`) are builtin modules and are not embedded.  `abc.py` resolves
/// `_abc` (a builtin), so `_py_abc.py` is not in the closure.
const RE_CLOSURE: &[&str] = &[
    "_collections_abc.py",
    "abc.py",
    "collections/__init__.py",
    "copyreg.py",
    "enum.py",
    "functools.py",
    "keyword.py",
    "re/__init__.py",
    "re/_casefix.py",
    "re/_compiler.py",
    "re/_constants.py",
    "re/_parser.py",
    "reprlib.py",
    "types.py",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Only do work for the wasm_vfs feature; native builds need nothing here.
    if std::env::var_os("CARGO_FEATURE_WASM_VFS").is_none() {
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let stdlib_root = Path::new(&manifest_dir).join("../../lib-python/3");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");

    // Serialize the closure into length-prefixed records:
    //   [count: u32 LE]
    //   repeated: [name_len: u32 LE][name utf8][src_len: u32 LE][src utf8]
    let mut raw: Vec<u8> = Vec::new();
    raw.extend_from_slice(&(RE_CLOSURE.len() as u32).to_le_bytes());
    for rel in RE_CLOSURE {
        let path = stdlib_root.join(rel);
        println!("cargo:rerun-if-changed={}", path.display());
        let src = std::fs::read(&path)
            .unwrap_or_else(|e| panic!("wasm_vfs: cannot read {}: {e}", path.display()));
        // VFS keys use forward slashes regardless of host separator.
        let name = rel.as_bytes();
        raw.extend_from_slice(&(name.len() as u32).to_le_bytes());
        raw.extend_from_slice(name);
        raw.extend_from_slice(&(src.len() as u32).to_le_bytes());
        raw.extend_from_slice(&src);
    }

    let compressed = lz4_flex::block::compress_prepend_size(&raw);
    let blob_path = Path::new(&out_dir).join("stdlib_vfs.lz4");
    std::fs::write(&blob_path, &compressed)
        .unwrap_or_else(|e| panic!("wasm_vfs: cannot write {}: {e}", blob_path.display()));
}
