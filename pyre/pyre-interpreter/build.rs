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

    // PyPy's `_multibytecodec` uses the cjkcodecs engine and mapping tables
    // verbatim (`pypy/module/_multibytecodec/c_codecs.py` ECI). Compile the
    // Japanese codec unit together with the shared stateful buffer engine;
    // Rust only provides the interpreter wrappers around this ABI.
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.starts_with("wasm32-") {
        let cjk_root = Path::new("src/module/_multibytecodec");
        let cjk_sources = [
            "src/module/_multibytecodec/src/cjkcodecs/_codecs_jp.c",
            "src/module/_multibytecodec/src/cjkcodecs/multibytecodec.c",
        ];
        for source in cjk_sources {
            println!("cargo:rerun-if-changed={source}");
        }
        for header in [
            "src/precommondefs.h",
            "src/cjkcodecs/alg_jisx0201.h",
            "src/cjkcodecs/cjkcodecs.h",
            "src/cjkcodecs/emu_jisx0213_2000.h",
            "src/cjkcodecs/fixnames.h",
            "src/cjkcodecs/mappings_jisx0213_pair.h",
            "src/cjkcodecs/mappings_jp.h",
            "src/cjkcodecs/multibytecodec.h",
        ] {
            println!("cargo:rerun-if-changed={}", cjk_root.join(header).display());
        }
        cc::Build::new()
            .files(cjk_sources)
            .include(cjk_root)
            .warnings(false)
            .compile("pyre_cjkcodecs_jp");
    }

    // `__try`/`__except` is the only thing that reaches a structured exception
    // and no Rust compiler emits one, so the fence a foreign call is made
    // inside (`src/module/_ctypes/seh.rs`) is a C translation unit of its own.
    if target.ends_with("-pc-windows-msvc") {
        println!("cargo:rerun-if-changed=src/module/_ctypes/seh.c");
        cc::Build::new()
            .file("src/module/_ctypes/seh.c")
            .compile("pyre_ctypes_seh");
    }

    // `sys.version` names the C compiler the build used, and on an MSVC target
    // that name is the `MSC v.<_MSC_VER>` token
    // (`rpython/rlib/compilerinfo.py:22`).  `ctypes.util._get_build_version`
    // reads the number back out to decide which C runtime `find_library("c")`
    // may hand out; with no token at all it assumes MSVC 6 and answers
    // `msvcrt.dll`, a runtime this build does not share an `errno` with.  The
    // number comes from the preprocessor rather than a parsed banner: `/EP`
    // writes the expansion to stdout and nothing else.
    if target.ends_with("-pc-windows-msvc")
        && let Some(msc_ver) = msc_ver()
    {
        println!("cargo:rustc-env=PYRE_MSC_VER={msc_ver}");
    }

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

/// `_MSC_VER`, as the compiler itself expands it.  `None` when the probe
/// cannot be compiled, which leaves `sys.version` naming Rust alone.
fn msc_ver() -> Option<String> {
    let out_dir = std::env::var("OUT_DIR").ok()?;
    let probe = Path::new(&out_dir).join("pyre_msc_ver.c");
    std::fs::write(&probe, "_MSC_VER\n").ok()?;
    let output = cc::Build::new()
        .get_compiler()
        .to_command()
        .arg("/nologo")
        .arg("/EP")
        .arg(&probe)
        .output()
        .ok()?;
    let expanded = String::from_utf8_lossy(&output.stdout);
    let digits: String = expanded.chars().filter(char::is_ascii_digit).collect();
    (!digits.is_empty()).then_some(digits)
}
