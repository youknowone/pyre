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
//! This test recomputes the assembly from the same three sources and
//! byte-compares against `tests/snapshots/opcode_handler_impls.snap`. Any
//! drift — whether to a template file or the spec emit — requires a
//! deliberate matching snapshot update, surfacing the change in code review.
//!
//! To intentionally update the assembled output:
//!   1. Edit the relevant template file or `handler_spec.rs`.
//!   2. Run `cargo build -p pyre-jit-trace --features cranelift` once so
//!      build.rs writes the new OUT_DIR/jit_trace_trait_impls.rs.
//!   3. `cp <OUT_DIR>/jit_trace_trait_impls.rs tests/snapshots/opcode_handler_impls.snap`
//!   4. Re-run the test and commit all changes together.

use std::path::PathBuf;

const PRE_REL: &str = "src/opcode_handler_impls_pre.template.rs";
const POST_REL: &str = "src/opcode_handler_impls_post.template.rs";
const SNAPSHOT_REL: &str = "tests/snapshots/opcode_handler_impls.snap";

fn manifest_path(rel: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join(rel)
}

#[test]
fn assembled_trait_impls_match_snapshot() {
    let pre_path = manifest_path(PRE_REL);
    let post_path = manifest_path(POST_REL);
    let snapshot_path = manifest_path(SNAPSHOT_REL);

    let pre = std::fs::read_to_string(&pre_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", pre_path.display()));
    let post = std::fs::read_to_string(&post_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", post_path.display()));
    let simple = majit_codewriter::handler_spec::emit_simple_trait_impls();
    let assembled = format!("{pre}{simple}{post}");

    let snapshot = std::fs::read_to_string(&snapshot_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", snapshot_path.display()));

    if assembled == snapshot {
        return;
    }

    let assembled_lines: Vec<&str> = assembled.lines().collect();
    let snapshot_lines: Vec<&str> = snapshot.lines().collect();
    let mut diff_line = 0;
    for (i, (a, s)) in assembled_lines
        .iter()
        .zip(snapshot_lines.iter())
        .enumerate()
    {
        if a != s {
            diff_line = i + 1;
            break;
        }
    }
    if diff_line == 0 {
        diff_line = assembled_lines.len().min(snapshot_lines.len()) + 1;
    }

    panic!(
        "\n\
        Assembled OpcodeHandler trait impls have diverged from the snapshot.\n\
        \n\
        First difference at line {diff_line}:\n  \
            assembled: {assembled_line}\n  \
            snapshot:  {snapshot_line}\n\
        \n\
        Total lines:  assembled={assembled_total}  snapshot={snapshot_total}\n\
        \n\
        Sources of the assembled output:\n  \
            pre:    {pre}\n  \
            simple: majit_codewriter::handler_spec::emit_simple_trait_impls()\n  \
            post:   {post}\n\
        \n\
        If this change is intentional:\n  \
            1. cargo build -p pyre-jit-trace --features cranelift\n  \
            2. cp <OUT_DIR>/jit_trace_trait_impls.rs {snapshot}\n  \
            3. commit the snapshot together with the source change.\n",
        diff_line = diff_line,
        assembled_line = assembled_lines
            .get(diff_line - 1)
            .copied()
            .unwrap_or("<eof>"),
        snapshot_line = snapshot_lines
            .get(diff_line - 1)
            .copied()
            .unwrap_or("<eof>"),
        assembled_total = assembled_lines.len(),
        snapshot_total = snapshot_lines.len(),
        pre = pre_path.display(),
        post = post_path.display(),
        snapshot = snapshot_path.display(),
    );
}
