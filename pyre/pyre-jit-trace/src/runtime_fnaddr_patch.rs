//! Runtime patcher for stale build-time fnaddrs in deserialized JitCodes.
//!
//! RPython's translator AOT-compiles every helper into the same C binary as
//! the runtime metainterp, so `JitCode.fnaddr` and the funcptr entries the
//! codewriter materializes into `JitCode.constants_i` (`jtransform.py:455-471`
//! `handle_residual_call` + `:614-623 direct_funcptr_value`) are linker-
//! resolved C addresses that the runtime executes via `cpu.bh_call_*`
//! without further bookkeeping.
//!
//! Pyre's `majit-translate` runs in `pyre-jit-trace/build.rs` — a separate
//! cargo build-script process from `pyre-dynasm` (and any other pyre runtime
//! binary).  The fnaddrs the codewriter captured therefore reflect the
//! build-script process's `pyre_interpreter::jit_trace_fnaddrs()` snapshot,
//! whose addresses are invalidated by ASLR (per-process random slide) and by
//! the divergent executable layouts (the build-script binary embeds a
//! subset of the runtime's symbols).  A walker that follows
//! `execute_residual_call`'s elidable-EI branch (`jitcode_dispatch.rs:3192-
//! 3239`) into one of those stale addresses dereferences arbitrary memory →
//! SEGV.
//!
//! This module bridges that gap.  At build time
//! `pyre-jit-trace/build.rs` serialises the `(path, build_fnaddr)` table
//! that `pyre_interpreter::jit_trace_fnaddrs()` returned for the
//! codewriter.  At runtime [`patch_constants_i_fnaddrs`] re-queries
//! `jit_trace_fnaddrs()` (now reading the runtime process's addresses),
//! builds a `build_fnaddr → runtime_fnaddr` correspondence keyed by the
//! shared `path`, and rewrites every stale value in
//! `JitCode.constants_i` and `JitCode.fnaddr`.  After the patch the
//! walker's `call_int_function(funcptr, args)` invokes the correct
//! runtime entry point, matching the upstream linker-resolved
//! invariant.

use std::collections::HashMap;
use std::sync::Arc;

use majit_translate::jitcode::JitCode;

/// Build-time `(path, build_fnaddr)` snapshot — bincoded by
/// `pyre-jit-trace/build.rs` from
/// `pyre_interpreter::jit_trace_fnaddrs()` immediately before the
/// codewriter consumes it.  Each entry shares its `path` with the
/// runtime call to `jit_trace_fnaddrs()` below; only the `i64` address
/// differs across processes.
fn build_time_fnaddr_bindings() -> Vec<(String, i64)> {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_fnaddr_bindings.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_fnaddr_bindings.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
}

/// Apply the build → runtime fnaddr correspondence to every JitCode the
/// caller just deserialised.  Mutates each Arc in place — refcount must
/// be 1 on entry (the canonical `ALL_JITCODES` LazyLock satisfies this
/// because `bincode::deserialize` produces fresh `Arc::new(...)` shells
/// before any consumer can clone them).
///
/// `JitCode.fnaddr` carries the shell-level fnaddr the codewriter
/// recorded in `CallControl::get_jitcode` (`call.rs:2647`); the per-
/// instruction funcptr operands the residual_call dispatcher reads
/// land in `JitCodeBody.constants_i` via the assembler's
/// `emit_const_i_from_const` path (`assembler.rs:2453-2473`).  Both
/// surfaces are patched so the walker sees the same address regardless
/// of which lookup it routes through.
pub fn patch_constants_i_fnaddrs(jitcodes: &mut Vec<Arc<JitCode>>) {
    let build_bindings = build_time_fnaddr_bindings();
    let runtime_bindings = pyre_interpreter::jit_trace_fnaddrs();

    let runtime_map: HashMap<&'static str, i64> = runtime_bindings.into_iter().collect();

    // `correspondence[build_fnaddr] = runtime_fnaddr` — only entries
    // whose runtime lookup actually disagrees with the build value get
    // patched; identical entries are dropped so the constants_i scan
    // can early-exit on a `HashMap::get` miss without comparing.
    let mut correspondence: HashMap<i64, i64> = HashMap::new();
    for (path, build_fnaddr) in &build_bindings {
        if let Some(&runtime_fnaddr) = runtime_map.get(path.as_str()) {
            if *build_fnaddr != runtime_fnaddr {
                correspondence.insert(*build_fnaddr, runtime_fnaddr);
            }
        }
    }

    if correspondence.is_empty() {
        return;
    }

    for arc in jitcodes.iter_mut() {
        let jc = Arc::get_mut(arc).expect(
            "patch_constants_i_fnaddrs: Arc<JitCode> already shared before patch — \
             every caller must run this before publishing the table to consumers",
        );
        if let Some(&runtime) = correspondence.get(&jc.fnaddr) {
            jc.fnaddr = runtime;
        }
        // Some shells reach the persisted table without a committed body
        // (e.g. `Default::default()` placeholders kept for `Arc<JitCode>::
        // default()` consumers in `BlackholeInterpreter::new`); they carry
        // empty `constants_i` so skipping the body-mut access is safe.
        if jc.try_body().is_some() {
            for c in jc.body_mut().constants_i.iter_mut() {
                if let Some(&runtime) = correspondence.get(c) {
                    *c = runtime;
                }
            }
        }
    }
}
