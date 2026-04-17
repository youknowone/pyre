//! Drift guard for the assembled `OpcodeHandler` trait impls.
//!
//! `pyre/pyre-jit-trace/build.rs` writes `OUT_DIR/jit_trace_trait_impls.rs`
//! by concatenating three pieces:
//!
//!   1. `src/opcode_handler_impls_pre.template.rs` — variant
//!      `SharedOpcodeHandler` impl + the file header. Hand-maintained.
//!   2. `majit_codewriter::handler_spec::emit_simple_trait_impls()` — the 5
//!      simple traits (Constant/Stack/Truth/Iter/Local) emitted from a spec
//!      table.
//!   3. `src/opcode_handler_impls_post.template.rs` — variant
//!      `ControlFlow/Branch/Namespace/Arithmetic` impls. Hand-maintained.
//!
//! This test recomputes the assembly from the same three sources and snapshots
//! it with `insta`, so `cargo insta test` / `cargo insta accept` drive the
//! review flow instead of a hand-rolled byte-compare panic.
//!
//! To intentionally update the assembled output:
//!   1. Edit the relevant template file or `handler_spec.rs`.
//!   2. Run `cargo insta test -p pyre-jit-trace --test trait_impls_snapshot --features cranelift`
//!   3. Review the `.snap.new` diff and `cargo insta accept`
//!   4. Commit the source change together with the accepted snapshot.

use std::path::PathBuf;

fn manifest_path(rel: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join(rel)
}

#[test]
fn assembled_trait_impls_match_snapshot() {
    let pre_path = manifest_path("src/opcode_handler_impls_pre.template.rs");
    let post_path = manifest_path("src/opcode_handler_impls_post.template.rs");
    let snapshot_dir = manifest_path("tests/snapshots");

    let pre = std::fs::read_to_string(&pre_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", pre_path.display()));
    let post = std::fs::read_to_string(&post_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", post_path.display()));
    let simple = majit_codewriter::handler_spec::emit_simple_trait_impls();
    let assembled = format!("{pre}{simple}{post}");

    let mut settings = insta::Settings::clone_current();
    settings.set_snapshot_path(snapshot_dir);
    settings.set_prepend_module_to_snapshot(false);
    settings.bind(|| {
        insta::assert_snapshot!("opcode_handler_impls", assembled);
    });
}
