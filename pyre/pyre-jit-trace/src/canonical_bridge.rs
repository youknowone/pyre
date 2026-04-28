//! Canonical → runtime JitCode bridge.
//!
//! The pyre type split from `pyjitcode.rs` documentation: build-time
//! `majit_translate::jitcode::JitCode` (CanonicalJitCode) carries the
//! codewriter pipeline output as serialized in
//! `$OUT_DIR/opcode_jitcodes.bin`; runtime
//! `majit_metainterp::jitcode::JitCode` (RuntimeJitCode) carries the same
//! canonical fields *plus* per-jitcode runtime adapter pools
//! (`exec.descrs: Vec<RuntimeBhDescr>`) that `JitCodeBuilder` populates
//! at lazy compile time.
//!
//! This module bridges the two representations at the **field level only**.
//! The bridged `RuntimeJitCode` carries:
//!
//! * Every canonical field (name, code, constants_*, c_num_regs_*, fnaddr,
//!   calldescr, jitdriver_sd, index).
//! * An empty `exec.descrs` pool by default (`bridge_canonical_jitcode_basic`),
//!   or a populated pool from full bridge (`bridge_canonical_jitcode_full`).
//!
//! ## Source-only vs legacy descr-resolution surface (G.3-pre Step 3 audit, 2026-04-25)
//!
//! There are two competing descr-resolution surfaces in pyre's runtime:
//!
//! | Encoding family | Argcode | Resolution surface | Bridged jitcode safe? |
//! |---|---|---|---|
//! | Source-only `residual_call_*/iRd>k`, `inline_call_*/dR>k`, `getfield_gc_*`, `getarrayitem_gc_*`, … | `'d'` / `'j'` | Global `bh.descrs` (`Vec<BhDescr>` = `ALL_DESCRS`) — `read_descr` (`blackhole.rs:5558`) | ✅ — global pool already populated by metainterp init |
//! | Legacy `JitCodeBuilder` (per-CodeObject lazy compile) `BC_CALL_INT`/`BC_INLINE_CALL` | `next_u16` (raw 2-byte index) | Per-jitcode `self.jitcode.exec.descrs` (`Vec<RuntimeBhDescr>`) — `JitCode::call_target` (`jitcode/mod.rs:699`) | ❌ — bridge must populate the slot |
//!
//! The build-time portal jitcode (`execute_opcode_step` /
//! `eval_loop_jit`) is produced by the source-only translator
//! (`analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings`), so 100%
//! of its `'d'`/`'j'` operand sites resolve through the **global pool**.
//! The per-jitcode `exec.descrs` is dead weight for portal execution.
//!
//! Diagnostic probe `portal_descr_argcode_emitter_breakdown` (this
//! module) confirms the breakdown:
//!
//! | Opname | `d`/`j` operand sites |
//! |---|---|
//! | `getarrayitem_gc_r` | 16 |
//! | `getfield_gc_r` | 26 |
//! | `inline_call_ir_r` | 4 |
//! | `inline_call_r_r` | 98 |
//! | `residual_call_r_r` | 247 |
//!
//! Zero legacy `BC_CALL_*` / `BC_RESIDUAL_CALL_*` opcodes — the source-only
//! translator never emits them.
//!
//! ## Bridge variants
//!
//! * `bridge_canonical_jitcode_basic`: copies canonical fields + empty
//!   `exec.descrs`. Sufficient for portal entry because the source-only
//!   path bypasses `exec.descrs` entirely.
//! * `bridge_canonical_jitcode_full`: walks the bytecode and populates
//!   `exec.descrs` slots referenced by `'j'`/`'d'` argcodes. Kept around
//!   for future legacy-encoded jitcodes (pyre's per-CodeObject lazy
//!   compile path) that genuinely need the per-jitcode pool. **Not** a
//!   prerequisite for the portal G.3 redirect.
//!
//! ## Higher-level `PyJitCode` wrapper (G.3a)
//!
//! `install_portal_for(code_ptr, w_code) -> Arc<PyJitCode>` composes
//! `portal_jitcode()` + `bridge_canonical_jitcode_basic` and wraps the
//! result in a `PyJitCode` whose pyre-specific `metadata` is empty.
//! Empty metadata is the marker that distinguishes a portal-bridged
//! `PyJitCode` from a per-CodeObject one produced by
//! `compile_jitcode_for_callee`. See the function docstring for the
//! full reader-site audit (G.3a).
//!
//! No production caller invokes `bridge_canonical_jitcode_*` or
//! `install_portal_for` at this commit — the functions are the
//! building blocks G.3b (reader portal-mode awareness) and G.3c
//! (caller flip) will consume.
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/jitcode.py:9-43` — single `JitCode` class
//!   that carries every field RPython needs (canonical+runtime fused).
//! - `rpython/jit/metainterp/blackhole.py:103,288` — global
//!   `BlackholeInterpBuilder.descrs` / `BlackholeInterpreter.descrs`,
//!   the analog of pyre's `ALL_DESCRS`.
//! - `rpython/jit/metainterp/blackhole.py:154-157` — `j` argcode
//!   resolution `descrs[idx]` followed by `isinstance(value, JitCode)`,
//!   the upstream version of pyre's `RuntimeBhDescr::as_jitcode`.

use std::collections::HashMap;
use std::sync::Arc;

use majit_metainterp::jitcode::{
    JitCallTarget, JitCode as RuntimeJitCode, JitCodeExecState, RuntimeBhDescr,
};
use majit_translate::jitcode::{BhDescr, JitCode as CanonicalJitCode};

use crate::pyjitcode::{PyJitCode, PyJitCodeMetadata};

/// Field-level bridge from canonical → runtime jitcode.
///
/// Constructs a fresh `Arc<RuntimeJitCode>` whose canonical fields
/// (name, code, constants, register counts, fnaddr, calldescr,
/// jitdriver_sd, index) mirror `canonical`. The runtime adapter pool
/// `exec.descrs` is left empty — see the module docstring for the
/// resulting safe-execution envelope.
///
/// Constant pool encoding mirrors RPython exactly:
/// * `constants_i` is a straight `Vec<i64>` clone (RPython
///   `jitcode.py:32` keeps the raw int).
/// * `constants_r` widens `Vec<u64>` (build-time GCREF placeholder) to
///   `Vec<i64>` because the runtime side stores raw addresses as
///   `i64`. RPython uses `lltype.cast_opaque_ptr(GCREF, ...)` which
///   ends up the same width; see `majit_translate::jitcode` field
///   docstring for the GCREF/u64/i64 alignment note.
/// * `constants_f` reinterprets the `f64` bits as `i64` via
///   `f64::to_bits` → `i64`, matching RPython's
///   `longlong.float2longlong` (jitcode.py:34) bitwise round-trip.
///
/// Panics if `canonical.body()` has not been committed (the canonical
/// jitcode is still a shell from `CallControl::get_jitcode`).
pub fn bridge_canonical_jitcode_basic(canonical: &CanonicalJitCode) -> Arc<RuntimeJitCode> {
    build_runtime_jitcode(
        canonical,
        JitCodeExecState::default(),
        canonical.jitdriver_sd(),
    )
}

/// Shared builder used by both `bridge_canonical_jitcode_basic` (which
/// passes a default-empty `exec`) and `bridge_canonical_jitcode_full`
/// (which passes a populated descr pool). The two public entry points
/// differ only in how `exec` is sourced — the canonical-field copy is
/// identical.
///
/// `jitdriver_sd` is taken as a parameter because callers disagree on
/// the value: the descr-population bridges (basic + full) keep the
/// canonical's `jitdriver_sd` so trace recording sees a portal-flagged
/// jitcode; `install_portal_for` (G.4.3a-fix) clears it to `None` to
/// match upstream's `call.py:147` `jd.mainjitcode = self.get_jitcode(
/// jd.portal_graph)` invariant — only ONE jitcode in
/// `metainterp_sd.jitcodes` carries `jitdriver_sd`, never the
/// per-CodeObject installs.
fn build_runtime_jitcode(
    canonical: &CanonicalJitCode,
    exec: JitCodeExecState,
    jitdriver_sd: Option<usize>,
) -> Arc<RuntimeJitCode> {
    let body = canonical.body();

    let constants_r: Vec<i64> = body.constants_r.iter().map(|&u| u as i64).collect();
    let constants_f: Vec<i64> = body
        .constants_f
        .iter()
        .map(|&f| f.to_bits() as i64)
        .collect();

    // RPython codewriter.py:68 `jitcode.index = index` — preserved across
    // the bridge. Canonical `index` is set once via `OnceLock`; runtime
    // `index` is `AtomicI64` so `state::jitcode_for` can later
    // back-stamp the SD-local position. Use -1 as "unset" sentinel
    // matching the existing `pyre-jit-trace` snapshot convention.
    let initial_index = canonical.try_index().map(|i| i as i64).unwrap_or(-1);

    Arc::new(RuntimeJitCode {
        name: canonical.name.clone(),
        code: body.code.clone(),
        c_num_regs_i: body.c_num_regs_i as u16,
        c_num_regs_r: body.c_num_regs_r as u16,
        c_num_regs_f: body.c_num_regs_f as u16,
        constants_i: body.constants_i.clone(),
        constants_r,
        constants_f,
        jitdriver_sd,
        fnaddr: canonical.fnaddr,
        calldescr: body.calldescr.clone(),
        index: std::sync::atomic::AtomicI64::new(initial_index),
        exec,
    })
}

/// Placeholder `RuntimeBhDescr` for slots whose canonical variant has
/// no `RuntimeBhDescr` analog (Field/Array/Size/Switch/Vable*/VtableMethod).
///
/// Those bytecode opcodes resolve their descrs through the global
/// `BlackholeInterpreter.descrs` pool — same index, but reading from
/// `bh.descrs[idx]` (`Vec<BhDescr>`) instead of
/// `self.jitcode.exec.descrs[idx]` (`Vec<RuntimeBhDescr>`). The slot
/// in `exec.descrs` therefore must exist (so the index space stays
/// dense and bytecode operands resolve correctly for `j`/`d` argcodes
/// that *do* read from `exec.descrs`), but the slot itself is never
/// consumed.
///
/// The placeholder is `RuntimeBhDescr::Call(JitCallTarget { null,
/// null })`. Picking `Call` over `JitCode` matters because
/// `RuntimeBhDescr::JitCode` carries an `Arc<RuntimeJitCode>` — the
/// least-cost neutral variant is `Call` whose two raw pointers are
/// trivially constructible. If any opcode wrongly reads the slot the
/// null pointers fail-fast (segv via `cpu.bh_call_*`), making the
/// invariant violation immediately visible.
fn placeholder_runtime_bhdescr() -> RuntimeBhDescr {
    RuntimeBhDescr::Call(JitCallTarget::new(std::ptr::null(), std::ptr::null()))
}

/// Full bridge — constructs a `RuntimeJitCode` whose `exec.descrs` is
/// populated by walking the global `ALL_DESCRS` pool, converting each
/// canonical `BhDescr` to the matching `RuntimeBhDescr` variant, and
/// recursively bridging callee jitcodes referenced via
/// `BhDescr::JitCode { jitcode_index }`.
///
/// Index preservation: the bridged `exec.descrs.len() ==
/// ALL_DESCRS.len()` and `exec.descrs[i]` corresponds to
/// `ALL_DESCRS[i]`. The canonical bytecode's `j`/`d` argcode operands
/// therefore continue to resolve correctly without any bytecode
/// rewriting — the runtime BC_INLINE_CALL handler reads
/// `self.jitcode.exec.descrs[sub_idx]` and lands on the right slot.
///
/// ## Cycle handling
///
/// `BridgeContext::cache` memoizes finished bridges by canonical
/// `index`. Cycles in the call graph (where canonical jitcode A's
/// `BhDescr::JitCode` references B and B references A) cannot be
/// resolved without interior mutability of `Arc<RuntimeJitCode>` —
/// the cached `Arc` is already shared by the time recursion would
/// need to install descrs into it. Step 2 detects cycles via the
/// `in_progress` set and panics with a clear message; cycle support is
/// a follow-up commit (G.3-pre Step 2.5) that switches `exec` to
/// `OnceLock<JitCodeExecState>` or `UnsafeCell`-based interior
/// mutability.
///
/// The build-time portal jitcode `execute_opcode_step` is acyclic —
/// it dispatches to leaf opcode handlers that do not recurse back
/// into the dispatch loop — so cycle handling is not yet needed for
/// the production portal use case.
pub fn bridge_canonical_jitcode_full(canonical: &CanonicalJitCode) -> Arc<RuntimeJitCode> {
    let mut ctx = BridgeContext::default();
    ctx.bridge(canonical)
}

#[derive(Default)]
struct BridgeContext {
    /// canonical `JitCode.index` → finished `Arc<RuntimeJitCode>`.
    /// Only completed entries (with populated `exec.descrs`) live here.
    cache: HashMap<usize, Arc<RuntimeJitCode>>,
    /// canonical `JitCode.index` set, populated on entry to
    /// `bridge(canonical)` and removed on exit. Used for cycle
    /// detection — a recursive call into the same index hits this set
    /// and panics rather than building a partially-populated cycle.
    in_progress: std::collections::HashSet<usize>,
}

impl BridgeContext {
    fn bridge(&mut self, canonical: &CanonicalJitCode) -> Arc<RuntimeJitCode> {
        let idx = canonical.index();

        if let Some(existing) = self.cache.get(&idx) {
            return existing.clone();
        }

        if !self.in_progress.insert(idx) {
            // Cycle: jitcode A's bytecode references jitcode B's descr
            // and B's bytecode (transitively) references A's descr (or
            // A self-INLINE_CALLs). Resolving cleanly requires interior
            // mutability of `Arc<RuntimeJitCode>` — the outer A bridge
            // call cannot install descrs into B's Arc once B's Arc has
            // already been cloned for the outer chain.
            //
            // Step 2's tactical workaround: return a placeholder Arc
            // (built via `bridge_canonical_jitcode_basic`, i.e. empty
            // `exec.descrs`). At runtime any BC_INLINE_CALL through
            // this slot would land on an empty exec pool and panic via
            // `BlackholeInterpreter::dispatch`'s slot-out-of-range
            // check — fail-fast rather than silent corruption.
            //
            // **G.3-pre Step 2.5** (follow-up commit) replaces this
            // with interior-mutability cycle handling once a real
            // production cycle proves it necessary. The portal jitcode
            // call surface contains a `load_const_value` self-cycle in
            // current build-time output (see test
            // `full_bridge_*`); investigating whether that is genuine
            // call recursion or a codewriter mis-emit is the gating
            // research item.
            // NOTE: do NOT mutate `in_progress` here — the outer
            // caller is still holding `idx` and will remove it on its
            // own return path.
            let cj = crate::jitcode_runtime::all_jitcodes()
                .get(idx)
                .expect("in_progress index must exist in ALL_JITCODES");
            return bridge_canonical_jitcode_basic(cj);
        }

        // Walk this jitcode's bytecode to find the descr indices it
        // actually consumes via `j`/`d` argcodes. Build `exec.descrs`
        // dense at `ALL_DESCRS.len()` so the operand-index space stays
        // intact, but only **referenced slots** get a real conversion;
        // every other slot keeps the null-pointer placeholder. This
        // confines recursive `convert_descr` (which can fire for
        // `BhDescr::JitCode`) to slots actually reachable from this
        // jitcode's bytecode — exhaustively walking `ALL_DESCRS`
        // instead would visit every callee descr including the current
        // jitcode's own slot, producing a spurious self-cycle even for
        // acyclic callgraphs.
        let body = canonical.body();
        let referenced = collect_referenced_descr_indices(&body.code);

        let pool_len = crate::jitcode_runtime::all_descrs().len();
        let mut descrs: Vec<RuntimeBhDescr> = (0..pool_len)
            .map(|_| placeholder_runtime_bhdescr())
            .collect();

        let canonical_descrs = crate::jitcode_runtime::all_descrs();
        for slot in referenced {
            if slot >= pool_len {
                panic!(
                    "canonical_bridge: bytecode of jitcode {idx} ({}) references descr \
                     slot {slot} but ALL_DESCRS has only {pool_len} entries",
                    canonical.name,
                );
            }
            descrs[slot] = self.convert_descr(&canonical_descrs[slot]);
        }

        let arc = build_runtime_jitcode(
            canonical,
            JitCodeExecState { descrs },
            canonical.jitdriver_sd(),
        );

        self.in_progress.remove(&idx);
        self.cache.insert(idx, arc.clone());
        arc
    }

    fn convert_descr(&mut self, canonical_descr: &BhDescr) -> RuntimeBhDescr {
        match canonical_descr {
            BhDescr::JitCode { jitcode_index, .. } => {
                // Recursive bridge of the callee. Cache hit short-circuits
                // every transitive caller after the first.
                let all = crate::jitcode_runtime::all_jitcodes();
                let callee = all.get(*jitcode_index).unwrap_or_else(|| {
                    panic!(
                        "canonical_bridge: BhDescr::JitCode references jitcode_index \
                         {jitcode_index} but ALL_JITCODES has only {} entries",
                        all.len(),
                    )
                });
                let runtime_arc = self.bridge(callee);
                RuntimeBhDescr::JitCode(runtime_arc)
            }
            // Build-time `BhDescr::Call { calldescr }` carries only the
            // calling-convention envelope; trace/concrete fn pointers
            // are resolved by the codewriter from `getfunctionptr(graph)`
            // (RPython `call.py:181-187`). For the bridged jitcode they
            // come from `pyre-jit-trace::jit_trace_fnaddrs()` — wired
            // through the production build-time
            // `analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings`.
            // The pyre runtime `RuntimeBhDescr::Call` carries the
            // resolved `(trace_ptr, concrete_ptr)` pair which the
            // canonical `BhDescr::Call` does not have at this layer; we
            // therefore install a null-pointer placeholder here. A
            // follow-up commit (G.3-pre Step 3) wires real fnaddrs by
            // looking up the canonical jitcode's `fnaddr` field and the
            // corresponding trace-helper symbol.
            BhDescr::Call { .. } => placeholder_runtime_bhdescr(),
            // Field/Array/Size/Switch/Vable*/VtableMethod: read through
            // the global `BlackholeInterpreter.descrs` pool (see module
            // docstring). The slot in `exec.descrs` is never consumed.
            BhDescr::Field { .. }
            | BhDescr::Array { .. }
            | BhDescr::Size { .. }
            | BhDescr::Switch { .. }
            | BhDescr::VableField { .. }
            | BhDescr::VableArray { .. }
            | BhDescr::VtableMethod { .. } => placeholder_runtime_bhdescr(),
        }
    }
}

/// Walk a jitcode's bytecode and return every descr index its `j`/`d`
/// argcode operands reference. The returned indices land in
/// `ALL_DESCRS`'s index space (assembler.py:181-196 — descrs are
/// emitted as a 2-byte little-endian operand).
///
/// Used by `BridgeContext::bridge` to confine recursive descr
/// conversion to slots the bytecode actually consumes; see the
/// "spurious self-cycle" note inside `bridge`.
fn collect_referenced_descr_indices(code: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    for op in crate::jitcode_runtime::decoded_ops(code) {
        let mut cursor = op.pc + 1;
        if op.opname == "live" {
            // `live/` carries an inline 2-byte liveness offset, no descr
            // operand. `decoded_ops` already advanced `next_pc` past
            // it, so the loop below would walk no chars regardless;
            // skip explicitly to avoid index drift.
            continue;
        }
        let mut chars = op.argcodes.chars();
        while let Some(c) = chars.next() {
            match c {
                'i' | 'c' | 'r' | 'f' => cursor += 1,
                'L' => cursor += 2,
                'd' | 'j' => {
                    // RPython assembler.py:184 — 2-byte LE descr index.
                    let lo = code[cursor] as usize;
                    let hi = code[cursor + 1] as usize;
                    out.push(lo | (hi << 8));
                    cursor += 2;
                }
                'I' | 'R' | 'F' => {
                    // varlist: 1-byte length + N register bytes.
                    let list_len = code[cursor] as usize;
                    cursor += 1 + list_len;
                }
                '>' => {
                    // result-type marker; following char names the
                    // result kind, then a 1-byte register destination.
                    let _ = chars.next();
                    cursor += 1;
                }
                _ => break,
            }
        }
    }
    out
}

/// G.3a — Wrap the build-time portal canonical jitcode in a
/// `PyJitCode` whose pyre-specific `metadata` is the portal-bridge
/// derivation product.
///
/// Body: `portal_jitcode()` (canonical) → `bridge_canonical_jitcode_basic`
/// (runtime `Arc<RuntimeJitCode>`) → wrap in `PyJitCode` with a
/// `PyJitCodeMetadata` whose `pc_map` is **empty** (the discriminator
/// `PyJitCode::is_portal_bridge()` checks) but whose other fields are
/// populated from the user `CodeObject`:
///   * `depth_at_py_pc` from the `LiveVars` forward stack analysis
///     (G.4.2 — `pyre-jit-trace/src/liveness.rs::depth_at_py_pc`).
///   * `portal_frame_reg` / `portal_ec_reg` from canonical inputarg
///     slot assignment (G.3h — `eval_loop_jit`'s sole Ref inputarg
///     lives at register 0; ec is read from frame on demand so its
///     fill is skipped via `u16::MAX` sentinel).
///   * `stack_base` from `code.varnames.len() + ncells(code)` to match
///     the per-CodeObject `compile_jitcode_for_callee` derivation.
///
/// The empty metadata is **not** an oversight — it is the marker that
/// distinguishes a portal-bridged `PyJitCode` from a per-CodeObject one
/// produced by `compile_jitcode_for_callee`. Per-CodeObject jitcodes
/// own a `pc_map` mapping Python PCs to JitCode byte offsets because
/// their bytecode IS the linear translation of one Python function.
/// The portal jitcode (`execute_opcode_step` /eventually `eval_loop_jit`)
/// dispatches on `pycode.instructions[pc]` *at runtime* via its own
/// dispatch arms — user PCs do not map to portal byte offsets.
///
/// ## Caller invariant (G.3a → G.3b → G.3c gate)
///
/// A portal-bridged `PyJitCode` MUST NOT reach readers that index
/// `metadata.pc_map[pc]`. The audit below classifies every reader site
/// in `pyre-jit-trace/src` and `pyre-jit/src` (19 sites total).
///
/// ### Reader audit (2026-04-25, blackhole-activate HEAD)
///
/// **Category A — panic / `BlackholeResult::Failed` on empty `pc_map`** (must
/// be portal-mode-aware before G.3c flips a caller):
///
/// | Site | Behavior |
/// |---|---|
/// | `pyre-jit-trace/src/state.rs:712-720` `frame_value_count_at` | G.3e/G.3f ✅ portal-mode-aware: `is_portal_bridge()` → return 0. Decoder consumes zero items for that frame; encoder side (`get_list_of_active_boxes` below) also returns an empty list, so the encoder/decoder counts match (RPython invariant). Guards trip → empty fail_args → graceful Failed. |
/// | `pyre-jit-trace/src/trace_opcode.rs:634-639` `get_list_of_active_boxes` skeleton check | G.3d/G.3f ✅ portal-mode-aware: `is_portal_bridge()` → empty `Vec`. G.3d initially emitted a conservative non-NONE register set here, which broke encoder/decoder symmetry against G.3e's 0-return; G.3f restored that invariant by emitting an empty list. Non-portal skeleton still panics. |
/// | `pyre-jit-trace/src/trace_opcode.rs:649-661` `pc_map.get(live_pc).copied().unwrap_or_else(panic!)` | `panic!("no pc_map entry …")` |
/// | `pyre-jit/src/call_jit.rs:807-820` resume blackhole `pc_map[py_pc]` | bounds-checked: returns `BlackholeResult::Failed` on miss. Diagnostic-only — no jit run. |
///
/// **Category B — graceful empty `pc_map`** (return `None`, empty `Vec`,
/// or boolean `false`):
///
/// | Site | Behavior |
/// |---|---|
/// | `pyre-jit-trace/src/state.rs:822-824` `frame_liveness_reg_indices_at` | `pc_map.get(pc)?` → returns empty `Vec`. |
/// | `pyre-jit-trace/src/state.rs:4176-4178` (`is_stack_live` lookup) | `pc_map.get(live_pc)?` → returns `false`. |
/// | `pyre-jit-trace/src/trace_opcode.rs:506-541` mid-trace lazy-load preamble | `if pc_map.is_empty() { Vec::new() }` — graceful skip. |
/// | `pyre-jit-trace/src/pyjitcode.rs:93` `is_populated()` predicate | boolean — caller decides. |
/// | `pyre-jit/src/call_jit.rs:1674` `pc_map.get(pc as usize).copied()?` | returns `None` → resume falls back. |
///
/// **Category C — portal_frame_reg / portal_ec_reg / stack_base reads**
/// (zero placeholder is silently wrong; readers consume them as register
/// numbers to fill at blackhole entry — wrong reg → vable shadow seeded
/// from reg 0 instead of the portal's frame slot):
///
/// | Site | Field |
/// |---|---|
/// | `pyre-jit/src/call_jit.rs:935-939` (`fill_portal_registers` after section setup) | `portal_frame_reg`, `portal_ec_reg` |
/// | `pyre-jit/src/call_jit.rs:1684` resume `with_virtualizable_stack_base` | `stack_base` |
/// | `pyre-jit/src/call_jit.rs:1787-1791` (`fill_portal_registers` chain walk) | `portal_frame_reg`, `portal_ec_reg` |
///
/// **Category D — writers / test fixtures** (out of scope for portal mode
/// — these only fire under `compile_jitcode_for_callee` or unit tests):
///
/// | Site | Use |
/// |---|---|
/// | `pyre-jit-trace/src/state.rs:5367` test fixture `pc_map.resize(…)` | unit-test scaffolding |
/// | `pyre-jit-trace/src/state.rs:6482` test fixture `pc_map.push(0)` | unit-test scaffolding |
/// | `pyre-jit/src/jit/codewriter.rs:6602` test assertion `!pc_map.is_empty()` | drain-result assertion |
///
/// ### G.3b scope (next commit)
///
/// Categories A and C must be portal-mode-aware. Two viable paths per
/// site (commit-time decision):
///
/// 1. **Branch on `metadata.pc_map.is_empty()`** — readers that are
///    portal-only translate user `py_pc` to portal entry-point pc via a
///    different path (e.g., constant entry pc, since portal handles
///    every user opcode through the same entry).
/// 2. **Pre-populate `metadata` from canonical inputargs** — extract
///    `portal_frame_reg`/`portal_ec_reg` from canonical `body().c_num_regs_*`
///    + `inputargs` slot assignment; keep `pc_map` empty since user PCs
///    are not portal PCs.
///
/// G.3b chooses per-category based on what semantics the reader actually
/// requires. G.3c then flips one caller (likely
/// `state::jitcode_for(code)` portal short-circuit, or
/// `inline_function_call`'s residual-call replacement) to use
/// `install_portal_for` instead of the existing per-CodeObject path,
/// gated by env var so dynasm/cranelift baselines can A/B compare.
///
/// ### Caller (G.3c)
///
/// `state::jitcode_for(code)` invokes `install_portal_for` when the
/// `PYRE_PORTAL_REDIRECT` environment variable is set, replacing the
/// per-CodeObject `COMPILE_JITCODE_FN` callback path. Flag OFF
/// preserves byte-for-byte the pre-G.3c behavior; flag ON is the
/// controlled probe surface for G.3d empirical reader-failure data.
pub fn install_portal_for(
    code_ptr: *const pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Arc<PyJitCode> {
    let canonical = crate::jitcode_runtime::portal_jitcode()
        .expect("install_portal_for: build-time portal canonical jitcode must exist");
    // G.4.3a-fix: clear `jitdriver_sd` on the per-CodeObject install.
    // Upstream `call.py:147` `jd.mainjitcode = self.get_jitcode(jd.
    // portal_graph)` followed by `jd.mainjitcode.jitdriver_sd = jd`
    // assigns the flag to **exactly one** jitcode in
    // `metainterp_sd.jitcodes`.  Pyre's portal-bridge wrapper produces
    // a fresh Arc per CodeObject; if every clone retained
    // `jitdriver_sd`, every per-CodeObject install would impersonate
    // the canonical mainjitcode and violate the single-portal invariant
    // (e.g. `state::jitcode_for` portal probes scanning `jitdriver_sd`
    // would find duplicates).
    let runtime = build_runtime_jitcode(&canonical, JitCodeExecState::default(), None);

    // Portal-bridge installs do not run the per-CodeObject regalloc, so
    // the post-regalloc colors of the portal red inputargs (`pypy/module/
    // pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`) are unknown
    // here.  The snapshot serializer at
    // `trace_opcode::get_list_of_active_boxes` sentinel-skips
    // `u16::MAX` so portal-bridge frames that hit guard capture fall
    // through to the locals/stack-only resolution path.
    //
    // RPython parity: `rpython/jit/metainterp/warmspot.py setup_jit`
    // derives the red arg inputarg indices from the jitdriver's
    // `reds=[]` declaration, not from the canonical jitcode's name.
    // Pyre lacks the jitdriver greens/reds → register-slot
    // infrastructure to do the analogous derivation here; the sentinel
    // is the safe default until the portal-bridge readers consult the
    // canonical's own portal_red_colors directly.
    let portal_frame_reg = u16::MAX;
    let portal_ec_reg = u16::MAX;

    // `stack_base` is the absolute index of the operand stack within
    // `PyFrame.locals_cells_stack_w`, matching the per-CodeObject
    // `compile_jitcode_for_callee` derivation
    // (`pyre-jit/src/jit/codewriter.rs:5233`).
    //
    // G.4.2 also derives `depth_at_py_pc` from the same user
    // `CodeObject` via the cached `LiveVars` forward stack analysis
    // (`pyre-jit-trace/src/liveness.rs:225-260`).  This populates the
    // portal-bridge metadata table that G.4.3 routes both the encoder
    // (`trace_opcode.rs:get_list_of_active_boxes`) and the decoder
    // (`state.rs:frame_value_count_at` / `restore_guard_failure_values`)
    // through, restoring the upstream packed-liveness invariant for
    // portal-bridge frames.  Both sides consume `stack_base +
    // depth_at_py_pc[pc]` Ref-typed boxes (G.4.3a-fix — pre-fix used
    // `nlocals_from_code() = varnames.len()` which dropped `ncells`).
    //
    // `pc_map` stays empty so `PyJitCode::is_portal_bridge()`
    // (`pyjitcode.rs:117-119`) keeps reporting `true` for portal-bridge
    // installs — the discriminator the G.3 reader audit branches on.
    // Portal-bridge install does not run regalloc, so the post-color
    // maps are populated as identity over the PyFrame slot ranges. Both
    // map lengths must match the runtime PyFrame allocation
    // (`pyframe.rs:1576` `nlocals + ncells + max_stackdepth`) so the
    // bridge fallback at `state.rs::setup_bridge_sym`
    // (`stack_base + stack_slot_color_map.len()`) reconstructs the full
    // `len(locals_cells_stack_w)` virtualizable shape rather than
    // collapsing to `stack_base + 0`. The contract is documented at
    // `PyreJitCode::stack_slot_color_map`. With no regalloc the color of each slot is
    // its own pre-color, hence the identity fill.
    let (stack_base, depth_at_py_pc, stack_slot_color_map, pyre_color_for_semantic_local) =
        if code_ptr.is_null() {
            (0, Vec::new(), Vec::new(), Vec::new())
        } else {
            let code = unsafe { &*code_ptr };
            let nlocals = code.varnames.len();
            let stack_base = nlocals + pyre_interpreter::pyframe::ncells(code);
            let depth_at_py_pc = crate::liveness::liveness_for(code_ptr).depth_at_py_pc();
            let max_stackdepth = code.max_stackdepth as usize;
            let stack_slot_color_map: Vec<u16> = (0..max_stackdepth as u16)
                .map(|d| stack_base as u16 + d)
                .collect();
            let pyre_color_for_semantic_local: Vec<u16> = (0..nlocals as u16).collect();
            (
                stack_base,
                depth_at_py_pc,
                stack_slot_color_map,
                pyre_color_for_semantic_local,
            )
        };

    Arc::new(PyJitCode {
        jitcode: runtime,
        metadata: PyJitCodeMetadata {
            pc_map: Vec::new(),
            depth_at_py_pc,
            portal_frame_reg,
            portal_ec_reg,
            stack_base,
            stack_slot_color_map,
            pyre_color_for_semantic_local,
        },
        code_ptr,
        w_code,
        has_abort: false,
        merge_point_pc: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::jitcode_runtime::portal_jitcode;

    /// Phase G G.3-pre Step 1 acceptance: the bridge preserves every
    /// canonical field on the build-time portal jitcode. `exec.descrs`
    /// stays empty — Step 2 fills it from the global `ALL_DESCRS` pool.
    #[test]
    fn bridge_preserves_portal_canonical_fields() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let runtime = bridge_canonical_jitcode_basic(&canonical);

        assert_eq!(runtime.name, canonical.name, "name must be preserved");
        let body = canonical.body();
        assert_eq!(
            runtime.code, body.code,
            "bytecode must be cloned verbatim — the bridge must not rewrite the canonical stream"
        );
        assert_eq!(
            runtime.c_num_regs_i, body.c_num_regs_i as u16,
            "c_num_regs_i widened from u8 to u16"
        );
        assert_eq!(runtime.c_num_regs_r, body.c_num_regs_r as u16);
        assert_eq!(runtime.c_num_regs_f, body.c_num_regs_f as u16);
        assert_eq!(
            runtime.constants_i, body.constants_i,
            "integer constant pool is a straight clone"
        );
        assert_eq!(
            runtime.constants_r.len(),
            body.constants_r.len(),
            "ref constant pool length matches"
        );
        assert_eq!(
            runtime.constants_f.len(),
            body.constants_f.len(),
            "float constant pool length matches"
        );
        assert_eq!(
            runtime.jitdriver_sd,
            canonical.jitdriver_sd(),
            "portal flag must propagate so `JitCode::jitdriver_sd` is `Some`"
        );
        assert_eq!(runtime.fnaddr, canonical.fnaddr);
        assert!(
            runtime.exec.descrs.is_empty(),
            "Step 1 leaves exec.descrs empty by design — Step 2 will populate j/d argcode slots"
        );
    }

    /// The bridge must produce a fresh `Arc` allocation; sharing
    /// canonical state by Arc would tie the runtime jitcode's mutable
    /// fields (e.g. `index`) to the canonical's `OnceLock` semantics
    /// and defeat the back-stamp path in `state::jitcode_for`.
    #[test]
    fn bridge_returns_independent_arc() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let r1 = bridge_canonical_jitcode_basic(&canonical);
        let r2 = bridge_canonical_jitcode_basic(&canonical);
        assert!(
            !Arc::ptr_eq(&r1, &r2),
            "each bridge call must produce a fresh Arc — runtime back-stamping needs unique ownership"
        );
    }

    /// Constants-f bit-pattern fidelity — bridging then reading back
    /// through `f64::from_bits` must round-trip every payload.
    #[test]
    fn bridge_constants_f_round_trip_via_bits() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let body = canonical.body();
        let runtime = bridge_canonical_jitcode_basic(&canonical);
        for (orig, bridged) in body.constants_f.iter().zip(runtime.constants_f.iter()) {
            let recovered = f64::from_bits(*bridged as u64);
            // NaN → NaN bit pattern equality (totalEq), regular floats → bitwise eq.
            assert_eq!(
                orig.to_bits(),
                recovered.to_bits(),
                "f64 → i64 bit cast must round-trip identically (longlong.float2longlong parity)"
            );
        }
    }

    /// Phase G G.3-pre Step 2 acceptance: the full bridge produces a
    /// `RuntimeJitCode` whose `exec.descrs` length matches the global
    /// `ALL_DESCRS` pool. Index preservation is the load-bearing
    /// invariant — the canonical bytecode's `j`/`d` argcode operands
    /// must continue to resolve correctly without rewriting.
    #[test]
    fn full_bridge_exec_descrs_length_matches_all_descrs() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let runtime = bridge_canonical_jitcode_full(&canonical);
        let pool_len = crate::jitcode_runtime::all_descrs().len();
        assert_eq!(
            runtime.exec.descrs.len(),
            pool_len,
            "full bridge must mirror ALL_DESCRS length so j/d argcode operands index correctly"
        );
    }

    /// Every canonical `BhDescr::JitCode` slot **referenced by the
    /// portal's bytecode** must land as a populated
    /// `RuntimeBhDescr::JitCode(arc)`. Unreferenced slots stay as the
    /// null-pointer placeholder by design — the bridge confines descr
    /// conversion to slots actually consumed by the bytecode (see
    /// `BridgeContext::bridge` for the rationale).
    ///
    /// This is the BC_INLINE_CALL safety invariant: every `j` argcode
    /// operand in the portal's bytecode must resolve to a real callee
    /// jitcode Arc when the runtime indexes `self.jitcode.exec.descrs`.
    #[test]
    fn full_bridge_jitcode_variant_slots_carry_valid_arcs() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let runtime = bridge_canonical_jitcode_full(&canonical);

        let body = canonical.body();
        let referenced = collect_referenced_descr_indices(&body.code);
        let canonical_descrs = crate::jitcode_runtime::all_descrs();

        let mut checked = 0;
        for slot in referenced {
            if let majit_translate::jitcode::BhDescr::JitCode { .. } = &canonical_descrs[slot] {
                checked += 1;
                let runtime_descr = &runtime.exec.descrs[slot];
                let _arc = runtime_descr.as_jitcode().unwrap_or_else(|| {
                    panic!(
                        "exec.descrs[{slot}] referenced by portal bytecode should be \
                         RuntimeBhDescr::JitCode for canonical BhDescr::JitCode, got \
                         {runtime_descr:?}"
                    )
                });
            }
        }
        // The portal's bytecode must reference at least one
        // BC_INLINE_CALL target — `execute_opcode_step` dispatches to
        // opcode handlers via `j` argcodes. Vacuous-coverage guard.
        assert!(
            checked > 0,
            "expected at least one BC_INLINE_CALL target referenced by portal bytecode"
        );
    }

    /// Every canonical descr variant that has no `RuntimeBhDescr`
    /// analog (Field/Array/Size/Switch/Vable*/VtableMethod) must land
    /// as the placeholder `RuntimeBhDescr::Call(null,null)`. This pins
    /// the placeholder choice — a future commit that wires real
    /// fnaddrs for Call slots must still leave the non-Call variants
    /// as null placeholders, since their slots are never consumed.
    #[test]
    fn full_bridge_non_call_variants_use_null_placeholder() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let runtime = bridge_canonical_jitcode_full(&canonical);

        let canonical_descrs = crate::jitcode_runtime::all_descrs();
        for (i, canonical_descr) in canonical_descrs.iter().enumerate() {
            let is_non_callable = matches!(
                canonical_descr,
                majit_translate::jitcode::BhDescr::Field { .. }
                    | majit_translate::jitcode::BhDescr::Array { .. }
                    | majit_translate::jitcode::BhDescr::Size { .. }
                    | majit_translate::jitcode::BhDescr::Switch { .. }
                    | majit_translate::jitcode::BhDescr::VableField { .. }
                    | majit_translate::jitcode::BhDescr::VableArray { .. }
                    | majit_translate::jitcode::BhDescr::VtableMethod { .. }
            );
            if !is_non_callable {
                continue;
            }
            let runtime_descr = &runtime.exec.descrs[i];
            let target = runtime_descr.as_call().unwrap_or_else(|| {
                panic!(
                    "exec.descrs[{i}] expected placeholder Call for non-callable canonical \
                     descr {canonical_descr:?}, got {runtime_descr:?}"
                )
            });
            assert!(
                target.trace_ptr.is_null() && target.concrete_ptr.is_null(),
                "non-Call canonical variant at slot {i} must produce null-pointer placeholder, \
                 got trace={:?} concrete={:?}",
                target.trace_ptr,
                target.concrete_ptr,
            );
        }
    }

    /// Diagnostic + invariant for G.3-pre Step 3 gating.
    ///
    /// Counts every descr slot the portal bytecode references and asserts
    /// the call-emitting families (`Call`) are non-zero — without that,
    /// G.3 redirect would never need real fnaddrs and Step 3 could stay
    /// deferred. Production snapshot (2026-04-25, portal =
    /// `execute_opcode_step`) lands at:
    ///
    /// | Variant   | Count |
    /// |---|---|
    /// | Call      | 247   |
    /// | JitCode   | 102   |
    /// | Field     |  26   |
    /// | Array     |  16   |
    ///
    /// 247 Call slots establishes Step 3 as a hard prerequisite — every
    /// `BC_RESIDUAL_CALL_*` / `BC_CALL_*` would segv on a null
    /// `JitCallTarget` if the portal were entered without Step 3.
    ///
    /// After Step 3 lands, a follow-up test
    /// `full_bridge_call_slots_carry_real_fnaddrs` will pin the
    /// non-null invariant; this test stays as the upstream gating
    /// signal.
    #[test]
    fn portal_bytecode_descr_slot_class_breakdown() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let body = canonical.body();
        let descrs = crate::jitcode_runtime::all_descrs();

        let mut counts = std::collections::BTreeMap::<&'static str, usize>::new();
        for slot in collect_referenced_descr_indices(&body.code) {
            let kind: &'static str = match &descrs[slot] {
                BhDescr::JitCode { .. } => "JitCode",
                BhDescr::Call { .. } => "Call",
                BhDescr::Field { .. } => "Field",
                BhDescr::Array { .. } => "Array",
                BhDescr::Size { .. } => "Size",
                BhDescr::Switch { .. } => "Switch",
                BhDescr::VableField { .. } => "VableField",
                BhDescr::VableArray { .. } => "VableArray",
                BhDescr::VtableMethod { .. } => "VtableMethod",
            };
            *counts.entry(kind).or_default() += 1;
        }

        let total: usize = counts.values().sum();
        eprintln!(
            "portal `{}` bytecode references {total} descr slots:",
            canonical.name
        );
        for (kind, count) in &counts {
            eprintln!("  {kind:<15}{count}");
        }

        let call_count = counts.get("Call").copied().unwrap_or(0);
        assert!(
            call_count > 0,
            "portal bytecode has zero Call descrs — G.3-pre Step 3 \
             prerequisite must be re-evaluated"
        );
    }

    /// Diagnostic probe (G.3 prerequisite triage): for every `d`/`j`
    /// argcode operand the portal bytecode emits, group by emitting
    /// opname so we can decide which descr-resolution surface is in
    /// play. Two competing surfaces:
    ///
    /// * **Orthodox source-only path**: `residual_call_*/iRd>k`,
    ///   `call_may_force/iIRFd>k`, getfield/setfield, … — `read_descr`
    ///   (`blackhole.rs:5558`) resolves through the **global**
    ///   `bh.descrs` pool (`majit-translate` `Vec<BhDescr>` =
    ///   `ALL_DESCRS`). The per-jitcode `exec.descrs` slot is unused.
    /// * **Legacy `JitCodeBuilder` path**: `BC_CALL_INT` /
    ///   `BC_RESIDUAL_CALL_VOID` / `BC_INLINE_CALL` — descr index
    ///   resolved against `self.jitcode.exec.descrs` (per-jitcode
    ///   `Vec<RuntimeBhDescr>`). The bridge's Call slot must carry the
    ///   real `JitCallTarget`.
    ///
    /// If 100% of portal `d`/`j` references come from orthodox
    /// opnames, Step 3 (Call slot fnaddr resolution) is unnecessary
    /// for portal entry — the bridge's null-pointer placeholder in
    /// `exec.descrs` is dead weight, never consumed.
    #[test]
    fn portal_descr_argcode_emitter_breakdown() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let body = canonical.body();

        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        for op in crate::jitcode_runtime::decoded_ops(&body.code) {
            // Count d/j operand sites per opname.
            let dj_in_op = op
                .argcodes
                .chars()
                .filter(|c| matches!(c, 'd' | 'j'))
                .count();
            if dj_in_op == 0 {
                continue;
            }
            *counts.entry(op.opname.to_string()).or_default() += dj_in_op;
        }

        let total: usize = counts.values().sum();
        eprintln!(
            "portal `{}` bytecode emits {total} d/j operand sites by opname:",
            canonical.name
        );
        for (opname, count) in &counts {
            eprintln!("  {opname:<35} {count}");
        }
    }

    /// G.3a — `install_portal_for` returns a fresh `Arc<PyJitCode>`
    /// each call (no shared interior state between bridges).
    #[test]
    fn install_portal_for_returns_independent_arcs() {
        let r1 = install_portal_for(std::ptr::null(), std::ptr::null());
        let r2 = install_portal_for(std::ptr::null(), std::ptr::null());
        assert!(
            !Arc::ptr_eq(&r1, &r2),
            "each install_portal_for call must produce a fresh Arc"
        );
    }

    /// G.3a — A portal-bridged `PyJitCode` keeps `pc_map` empty so
    /// `is_portal_bridge()` (`pyjitcode.rs:117-119`) reports `true`.
    /// This is the **portal-mode marker** the G.3 reader audit branches
    /// on.  G.4.2 added the `depth_at_py_pc` derivation but only
    /// populates it when `code_ptr` is non-null — the null-fixture path
    /// preserves the original empty observation so the marker invariant
    /// still surfaces here.
    #[test]
    fn install_portal_for_metadata_is_empty_marker() {
        let pyjit = install_portal_for(std::ptr::null(), std::ptr::null());
        assert!(
            pyjit.metadata.pc_map.is_empty(),
            "portal-bridged pc_map must be empty (G.3a marker invariant)"
        );
        // Null code_ptr ⇒ no user CodeObject to analyse ⇒
        // `depth_at_py_pc` falls back to `Vec::new()`.  Production
        // callers always pass a non-null wrapper; that path is exercised
        // by `install_portal_for_populates_depth_at_py_pc_for_user_code`.
        assert!(pyjit.metadata.depth_at_py_pc.is_empty());
        // RPython-orthodox revert: `portal_frame_reg` /
        // `portal_ec_reg` are sentinel-skipped pending the proper
        // greens/reds → register-slot derivation port.  See
        // `install_portal_for` for the rationale.
        let _canonical = portal_jitcode().expect("build-time portal canonical jitcode must exist");
        assert_eq!(pyjit.metadata.portal_frame_reg, u16::MAX);
        assert_eq!(pyjit.metadata.portal_ec_reg, u16::MAX);
        // Test fixture passes null code_ptr; install_portal_for falls
        // back to stack_base = 0 (production callers always pass a
        // non-null wrapper, so stack_base = nlocals + ncells).
        assert_eq!(pyjit.metadata.stack_base, 0);
        assert!(!pyjit.has_abort);
        assert!(pyjit.merge_point_pc.is_none());
        assert!(
            !pyjit.is_populated(),
            "is_populated() must report false for portal-bridged jitcode \
             so existing skeleton-aware readers fall back gracefully"
        );
    }

    /// G.4.2 — Non-null `code_ptr` path: `install_portal_for` derives
    /// `depth_at_py_pc` from the user `CodeObject` via the cached
    /// `LiveVars` forward stack analysis.  The vector must be sized to
    /// `code.instructions.len()` so the portal-bridge encoder/decoder
    /// (G.4.3+) can index it by user PC.  `pc_map` stays empty and
    /// `is_portal_bridge()` keeps reporting `true` — the discriminator
    /// is independent of the depth table.
    ///
    /// Sanity bounds:
    ///   * Length matches `code.instructions.len()`.
    ///   * Maximum derived depth never exceeds `code.max_stackdepth`
    ///     (CPython compiler invariant — the max recorded by the
    ///     compiler is an upper bound on the runtime stack height at
    ///     any reachable PC).
    ///   * `is_portal_bridge()` stays true.
    #[test]
    fn install_portal_for_populates_depth_at_py_pc_for_user_code() {
        let source = "\
def f(x, y):
    a = x + y
    b = a * x
    return b - y
";
        let module = pyre_interpreter::compile_exec(source).expect("compile failed");
        let user_code = module
            .constants
            .iter()
            .find_map(|c| match c {
                pyre_interpreter::bytecode::ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object");
        let user_code_ptr = Box::into_raw(Box::new(user_code));

        let pyjit = install_portal_for(user_code_ptr, std::ptr::null());

        let code_ref = unsafe { &*user_code_ptr };
        assert_eq!(
            pyjit.metadata.depth_at_py_pc.len(),
            code_ref.instructions.len(),
            "depth_at_py_pc must have one entry per Python PC so the \
             portal-bridge encoder/decoder can index by user PC"
        );
        let max_observed = pyjit
            .metadata
            .depth_at_py_pc
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        assert!(
            (max_observed as u32) <= code_ref.max_stackdepth,
            "derived depth {} exceeds compiler's max_stackdepth {}",
            max_observed,
            code_ref.max_stackdepth
        );
        assert!(
            pyjit.is_portal_bridge(),
            "populating depth_at_py_pc must NOT flip the portal-bridge \
             discriminator — pc_map stays empty"
        );
        assert!(
            pyjit.metadata.pc_map.is_empty(),
            "portal-bridge pc_map must remain empty (G.3a marker invariant)"
        );
        // stack_base = nlocals + ncells for a non-null code_ptr; for
        // this fixture (no closures / cell vars), ncells = 0.
        assert_eq!(
            pyjit.metadata.stack_base,
            code_ref.varnames.len() + pyre_interpreter::pyframe::ncells(code_ref)
        );

        // Drop the leaked Box now the wrapper has been examined.
        drop(unsafe { Box::from_raw(user_code_ptr) });
    }

    /// G.3b — `PyJitCode::is_portal_bridge()` distinguishes a
    /// portal-bridged install from `PyJitCode::skeleton(...)` and from
    /// a fully-drained per-CodeObject `PyJitCode`. This is the hook
    /// G.3c readers will branch on.
    ///
    /// * Portal-bridge: `jitcode.code` non-empty + `pc_map` empty.
    /// * Skeleton: `jitcode.code` empty (RuntimeJitCode::default()) +
    ///   `pc_map` empty.
    /// * Drained per-CodeObject: `jitcode.code` non-empty +
    ///   `pc_map` populated by `compile_jitcode_for_callee`'s
    ///   drain.
    ///
    /// The discriminator must be true for portal-bridge and false for
    /// the other two states.
    #[test]
    fn is_portal_bridge_distinguishes_install_from_skeleton() {
        let portal = install_portal_for(std::ptr::null(), std::ptr::null());
        assert!(
            portal.is_portal_bridge(),
            "install_portal_for product must report is_portal_bridge() == true"
        );

        let skeleton = PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        assert!(
            !skeleton.is_portal_bridge(),
            "skeleton has empty jitcode.code — must NOT report portal-bridge"
        );

        // Drained per-CodeObject simulation: clone the portal jitcode
        // (so `code` is non-empty) and stamp a synthesized `pc_map`
        // entry. This mirrors what `compile_jitcode_for_callee`'s
        // drain produces — both fields populated.
        let mut drained_runtime = (*portal.jitcode).clone();
        // Defensively assert the precondition the discriminator relies
        // on — if a future refactor empties `code` here, the
        // assertion below would still pass but would no longer
        // exercise the intended case.
        assert!(
            !drained_runtime.code.is_empty(),
            "test setup: cloned portal jitcode must have non-empty code"
        );
        let drained = PyJitCode {
            jitcode: std::sync::Arc::new(drained_runtime),
            metadata: PyJitCodeMetadata {
                pc_map: vec![0],
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
                stack_slot_color_map: Vec::new(),
                pyre_color_for_semantic_local: Vec::new(),
            },
            code_ptr: std::ptr::null(),
            w_code: std::ptr::null(),
            has_abort: false,
            merge_point_pc: None,
        };
        assert!(
            !drained.is_portal_bridge(),
            "drained per-CodeObject (pc_map populated) must NOT report portal-bridge"
        );
    }

    /// G.3a — The wrapped runtime jitcode preserves every canonical
    /// field that `bridge_canonical_jitcode_basic` preserves on its
    /// own, so installing through `install_portal_for` does not
    /// quietly drop bridge fidelity. Cross-checks with the existing
    /// Step 1 acceptance test
    /// (`bridge_preserves_portal_canonical_fields`) by re-asserting
    /// the same invariants on the wrapped Arc — except `jitdriver_sd`,
    /// which `install_portal_for` deliberately clears to `None` so the
    /// per-CodeObject install does not impersonate the canonical
    /// mainjitcode (G.4.3a-fix; upstream `call.py:147` invariant —
    /// `jitdriver_sd` is set on exactly one jitcode per
    /// `JitDriverStaticData`).
    #[test]
    fn install_portal_for_jitcode_field_matches_basic_bridge() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let direct = bridge_canonical_jitcode_basic(&canonical);
        let pyjit = install_portal_for(std::ptr::null(), std::ptr::null());
        assert_eq!(pyjit.jitcode.name, direct.name);
        assert_eq!(pyjit.jitcode.code, direct.code);
        assert_eq!(pyjit.jitcode.c_num_regs_i, direct.c_num_regs_i);
        assert_eq!(pyjit.jitcode.c_num_regs_r, direct.c_num_regs_r);
        assert_eq!(pyjit.jitcode.c_num_regs_f, direct.c_num_regs_f);
        assert_eq!(pyjit.jitcode.constants_i, direct.constants_i);
        assert_eq!(pyjit.jitcode.constants_r, direct.constants_r);
        assert_eq!(pyjit.jitcode.constants_f, direct.constants_f);
        assert_eq!(pyjit.jitcode.fnaddr, direct.fnaddr);
        assert!(
            direct.jitdriver_sd.is_some(),
            "build-time canonical carries jitdriver_sd (it IS the portal); \
             this test pins the precondition the install_portal_for clear \
             invariant rests on"
        );
        assert!(
            pyjit.jitcode.jitdriver_sd.is_none(),
            "install_portal_for must clear jitdriver_sd so the per-CodeObject \
             clone does not impersonate the canonical mainjitcode \
             (call.py:147 single-jitdriver_sd invariant)"
        );
        assert!(
            pyjit.jitcode.exec.descrs.is_empty(),
            "install_portal_for delegates to bridge_canonical_jitcode_basic — \
             exec.descrs stays empty (Step 1 contract); the source-only \
             portal bypasses exec.descrs entirely (Step 3 audit)"
        );
    }
}
