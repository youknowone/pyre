//! Phase F build script: force cargo to rebuild this crate when
//! pyre-interpreter source changes.
//!
//! ## Positioning (PRE-EXISTING-ADAPTATION)
//!
//! RPython has no build-script layer; `rpython/jit/codewriter/` consumes
//! `translator.graphs` built in-process by the rtyper. pyre ships the
//! pyre-interpreter source as embedded `&'static str`s via `include_str!`
//! inside `src/generated.rs`, which means cargo's default source-tracking
//! only invalidates the crate on `src/**` changes — NOT on changes to
//! the embedded pyre-interpreter files that live in a sibling crate.
//!
//! This script closes that gap: it emits `cargo:rerun-if-changed=...`
//! for every source file `src/generated.rs` embeds, so edits in
//! pyre-interpreter propagate into a majit-translate rebuild (and
//! therefore into the lazy `OnceLock` rebuild on the next test / pyre
//! run).
//!
//! ## Out of scope
//!
//! - This script does NOT execute `make_jitcodes` at build time. The
//!   plan in `docs/plans/lucky-growing-puzzle.md` calls for bincode
//!   serialization here, but doing so requires separating the builder
//!   into its own crate (circular-dep: build.rs cannot import its
//!   own crate's lib). The runtime `OnceLock` path in
//!   `src/generated.rs` provides the same "static data embedded in
//!   the binary" effect via `include_str!`, at the cost of a one-time
//!   pipeline run on first access. If build time becomes the bottleneck
//!   for a consumer (Phase G onward), promote this to a sibling builder
//!   crate and generate bincode here.

fn main() {
    // Keep these paths in lock-step with the `PYRE_JIT_GRAPH_SOURCES`
    // manifest in `src/generated.rs`. Adding a file to one without the
    // other desynchronises cargo's rebuild tracking from the source set
    // compiled into the `OnceLock` — edits to the missing file would no
    // longer trigger a rebuild even though `all_jitcodes()` reads them.
    for path in [
        // Freestanding `opcode_*` handlers.
        "../../pyre/pyre-interpreter/src/pyopcode.rs",
        // `PyFrame` trait impls (LocalOpcodeHandler / SharedOpcodeHandler
        // / ControlOpcodeHandler / etc.).
        "../../pyre/pyre-interpreter/src/eval.rs",
        // Inherent `impl PyFrame` helpers (push / pop / peek /
        // check_exc_match).
        "../../pyre/pyre-interpreter/src/pyframe.rs",
        // Freestanding shared helpers (opcode_make_function / opcode_call
        // / opcode_build_{list,tuple,map} / opcode_store_subscr /
        // opcode_list_append / opcode_unpack_sequence / opcode_load_attr
        // / opcode_store_attr). Imported from `pyopcode.rs:6` and called
        // directly by default trait methods (pyopcode.rs:821).
        "../../pyre/pyre-interpreter/src/shared_opcode.rs",
        // Phase D0 — portal runner `eval_loop_jit` + resume helpers
        // (`allocate_struct` / `allocate_with_vtable` / exception glue).
        // RPython `warmspot.py::portal_runner` analogue: this is the
        // graph `find_all_graphs(portal)` seeds from at
        // `call.py:57`. Handlers reached from here (opcode_*) are
        // discovered as BFS callees, not as entry points.
        "../../pyre/pyre-jit/src/eval.rs",
    ] {
        println!("cargo:rerun-if-changed={}", path);
    }
}
