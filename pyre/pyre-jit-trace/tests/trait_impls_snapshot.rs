//! Drift guard for `src/opcode_handler_impls.template.rs`.
//!
//! The template is the source-of-truth for the `OpcodeHandler` trait impls
//! that `build.rs` copies into `OUT_DIR/jit_trace_trait_impls.rs`. The
//! impls are currently a *transcription* of what `trace_opcode.rs` used to
//! contain — they call `pub(crate)` helpers on `MIFrame` whose semantics
//! could change without the template noticing.
//!
//! This test compares the template against a checked-in snapshot. Any
//! edit to the template requires a deliberate matching edit to the
//! snapshot, surfacing drift in code review.
//!
//! To intentionally update the template:
//!   1. Edit `src/opcode_handler_impls.template.rs`
//!   2. `cp src/opcode_handler_impls.template.rs tests/snapshots/opcode_handler_impls.snap`
//!   3. Re-run the test and commit both files together.

use std::path::PathBuf;

const TEMPLATE_REL: &str = "src/opcode_handler_impls.template.rs";
const SNAPSHOT_REL: &str = "tests/snapshots/opcode_handler_impls.snap";

fn manifest_path(rel: &str) -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir).join(rel)
}

#[test]
fn opcode_handler_impls_matches_snapshot() {
    let template_path = manifest_path(TEMPLATE_REL);
    let snapshot_path = manifest_path(SNAPSHOT_REL);

    let template = std::fs::read_to_string(&template_path)
        .unwrap_or_else(|e| panic!("cannot read template at {}: {e}", template_path.display()));
    let snapshot = std::fs::read_to_string(&snapshot_path)
        .unwrap_or_else(|e| panic!("cannot read snapshot at {}: {e}", snapshot_path.display()));

    if template == snapshot {
        return;
    }

    // Show first divergence line so the test output is useful.
    let template_lines: Vec<&str> = template.lines().collect();
    let snapshot_lines: Vec<&str> = snapshot.lines().collect();
    let mut diff_line = 0;
    for (i, (t, s)) in template_lines.iter().zip(snapshot_lines.iter()).enumerate() {
        if t != s {
            diff_line = i + 1;
            break;
        }
    }
    if diff_line == 0 {
        diff_line = template_lines.len().min(snapshot_lines.len()) + 1;
    }

    panic!(
        "\n\
        opcode_handler_impls.template.rs has diverged from the checked-in snapshot.\n\
        \n\
        First difference at line {diff_line}:\n  \
            template: {template_line}\n  \
            snapshot: {snapshot_line}\n\
        \n\
        Total lines:  template={template_total}  snapshot={snapshot_total}\n\
        \n\
        If this change is intentional, refresh the snapshot:\n  \
            cp {template} {snapshot}\n\
        and commit both files together. The snapshot is a deliberate-action\n\
        guard against accidental drift between the trait impl template and\n\
        the helper functions it calls.\n",
        diff_line = diff_line,
        template_line = template_lines
            .get(diff_line - 1)
            .copied()
            .unwrap_or("<eof>"),
        snapshot_line = snapshot_lines
            .get(diff_line - 1)
            .copied()
            .unwrap_or("<eof>"),
        template_total = template_lines.len(),
        snapshot_total = snapshot_lines.len(),
        template = template_path.display(),
        snapshot = snapshot_path.display(),
    );
}
