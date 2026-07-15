//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use majit_backend::Backend;
use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_metainterp::virtualizable::VirtualizableInfo;
use majit_metainterp::{
    BridgeInlineCarrier, JitDriverStaticData, JitState, ReconstructRecipe,
    ResidualVirtualizableSync, TraceAction, TraceCtx,
};

use pyre_interpreter::bytecode::{CodeObject, ComparisonOperator, Instruction};
use pyre_interpreter::pyframe::PyFrame;
use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::pyobject::{
    FLOAT_TYPE, INT_TYPE, get_instantiate, is_bool, is_float, is_int, py_type_check,
};
use pyre_object::{PY_NULL, w_float_get_value, w_int_get_value, w_int_new};

/// jitcode.py:9-21 / codewriter.py:68: JitCode — compiled bytecode unit.
///
/// MetaInterpStaticData-side wrapper around the shared
/// [`crate::PyJitCode`] payload. The payload `Arc` is the same heap
/// allocation that `pyre_jit::jit::call::CallControl.jitcodes` holds
/// for the same CodeObject — RPython's `MetaInterpStaticData.jitcodes`
/// list and `CallControl.jitcodes` dict reference identical
/// `JitCode` Python objects through Python's refcount semantics, and
/// pyre mirrors that with a shared `Arc`. The wrapper keeps only
/// `index` (the SD-local `jitcode.index = len(all_jitcodes)` from
/// codewriter.py:68) on the SD side; `code_for_jitcode_index` recovers
/// the wrapping `PyObjectRef` from the payload's `code_ptr` through the
/// live-wrapper registry, so the wrapper no longer stores a `code`
/// field.
// SAFETY: JitCode is only written once (during creation) and then
// read-only. The code pointer is stable for the program lifetime.
unsafe impl Sync for JitCode {}

pub(crate) struct JitCode {
    /// codewriter.py:68: jitcode.index = len(all_jitcodes).
    pub index: i32,
    /// Shared `PyJitCode` payload. Same `Arc` instance also lives in
    /// `CallControl.jitcodes`. A rebuild (a later primary compile
    /// re-queuing an inlined callee graph) replaces this `Arc` so cached
    /// `*const JitCode` pointers see the refreshed payload on the next
    /// field access.
    pub payload: std::sync::Arc<crate::PyJitCode>,
}

impl JitCode {
    /// Extract raw CodeObject from this JitCode's payload.
    #[inline]
    pub unsafe fn raw_code(&self) -> *const CodeObject {
        self.payload.code_ptr
    }
}

/// warmspot.py:148-282: MetaInterpStaticData — per-driver compile-time data.
///
/// RPython: created by WarmRunnerDesc, holds jitcodes list populated
/// by codewriter.make_jitcodes(). Accessed as MetaInterp.staticdata.
///
/// pyre: per-thread equivalent (no-GIL runtime). Populated from the
/// authoritative `CodeWriter.make_jitcodes()` result before tracing.
///
/// TODO: the RPython-orthodox
/// `MetaInterpStaticData` lives in
/// `majit_metainterp::pyjitpl::MetaInterpStaticData`.  pyre embeds it
/// as the `canonical` field below and delegates every RPython method
/// (`setup_indirectcalltargets`, `bytecode_for_address`, …) through
/// it, so there is exactly one port of each of those methods.  The
/// pyre-local fields (`jitcodes`, `finish_setup_done`, `op_*`,
/// `liveness_info`) stay here because the surrounding runtime still
/// stores Python-code metadata outside the canonical staticdata.
struct MetaInterpStaticData {
    /// warmspot.py:282: self.metainterp_sd.jitcodes = jitcodes.
    /// Box<JitCode> for address stability across vec growth.
    jitcodes: Vec<Box<JitCode>>,
    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// frozen snapshot of the assembler's `all_liveness` buffer. In RPython
    /// this is set once at `finish_setup` time; in pyre the assembler is
    /// long-lived and liveness accumulates across lazy JitCode compiles,
    /// so this field is resynced after every `intern_liveness` write.
    ///
    /// Stored as `Arc<[u8]>` so `liveness_info_snapshot()` can hand out
    /// shared read-only slices (`metainterp_sd.liveness_info` parity in
    /// resume.py:1022) without cloning the byte buffer per BH entry.
    liveness_info: std::sync::Arc<[u8]>,
    /// pyjitpl.py:2255 `finish_setup` is per MetaInterpStaticData instance.
    /// `METAINTERP_SD` is thread-local in pyre, so this guard must live on
    /// the thread-local object rather than in a process-global `Once`.
    ///
    /// Unlike RPython, pyre assembles lazily and the writer-side
    /// `Assembler.insns` table grows over time. The guard therefore marks
    /// "initial finish_setup done" only; later snapshots still refresh the
    /// cached opcode ids and liveness bytes in place.
    finish_setup_done: bool,

    // pyjitpl.py:2236-2243 opcode number cache filled by `setup_insns`.
    // RPython stores every field even when the runtime currently does
    // not read them, so the structural parity is preserved. Sentinel
    // `u8::MAX` matches `insns.get('…', -1)` for lookups that happen
    // before `setup_insns` runs (e.g. early fallback paths).
    /// pyjitpl.py:2236 `self.op_live = insns.get('live/', -1)`.
    op_live: u8,
    /// pyjitpl.py:2237 `self.op_goto = insns.get('goto/L', -1)`.
    op_goto: u8,
    /// pyjitpl.py:2238 `self.op_catch_exception = insns.get('catch_exception/L', -1)`.
    op_catch_exception: u8,
    /// pyjitpl.py:2239 `self.op_rvmprof_code = insns.get('rvmprof_code/ii', -1)`.
    op_rvmprof_code: u8,
    /// pyjitpl.py:2240 `self.op_int_return = insns.get('int_return/i', -1)`.
    op_int_return: u8,
    /// pyjitpl.py:2241 `self.op_ref_return = insns.get('ref_return/r', -1)`.
    op_ref_return: u8,
    /// pyjitpl.py:2242 `self.op_float_return = insns.get('float_return/f', -1)`.
    op_float_return: u8,
    /// pyjitpl.py:2243 `self.op_void_return = insns.get('void_return/', -1)`.
    op_void_return: u8,

    /// pyjitpl.py:2190 `class MetaInterpStaticData` — the canonical,
    /// RPython-orthodox port.  Owns `indirectcalltargets`,
    /// `globaldata.indirectcall_dict`, and every RPython staticdata
    /// method (`setup_indirectcalltargets`, `bytecode_for_address`,
    /// `setup_descrs`, `setup_list_of_addr2name`, …).  Reached from
    /// pyre's module-level wrappers (`state::setup_indirectcalltargets`,
    /// `state::bytecode_for_address`) and from future callers that
    /// hold a `&mut MetaInterpStaticData` directly.
    canonical: majit_metainterp::MetaInterpStaticData,
}

#[allow(dead_code)]
impl MetaInterpStaticData {
    fn new() -> Self {
        Self {
            jitcodes: Vec::new(),
            liveness_info: std::sync::Arc::<[u8]>::from(Vec::<u8>::new().into_boxed_slice()),
            finish_setup_done: false,
            op_live: u8::MAX,
            op_goto: u8::MAX,
            op_catch_exception: u8::MAX,
            op_rvmprof_code: u8::MAX,
            op_int_return: u8::MAX,
            op_ref_return: u8::MAX,
            op_float_return: u8::MAX,
            op_void_return: u8::MAX,
            canonical: majit_metainterp::MetaInterpStaticData::new(),
        }
    }

    /// pyjitpl.py:2248-2249 `setup_indirectcalltargets(indirectcalltargets)`.
    /// Thin delegate to the canonical port; exists solely so callers
    /// that hold `&mut state::MetaInterpStaticData` can reach the
    /// method without constructing a reference to the canonical
    /// field themselves.
    fn setup_indirectcalltargets(
        &mut self,
        targets: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
    ) {
        self.canonical.setup_indirectcalltargets(targets);
    }

    /// pyjitpl.py:2326-2343 `bytecode_for_address(fnaddress)`.
    /// Thin delegate to the canonical port.
    fn bytecode_for_address(
        &mut self,
        fnaddress: usize,
    ) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
        self.canonical.bytecode_for_address(fnaddress)
    }
}

impl MetaInterpStaticData {
    /// pyjitpl.py:2227-2243 `MetaInterpStaticData.setup_insns(self, insns)`:
    /// copy opcode numbers for the well-known bytecodes out of the
    /// assembler's `insns` dict in the same order as upstream.
    fn setup_insns(&mut self, insns: &indexmap::IndexMap<String, u8>) {
        self.op_live = insns.get("live/").copied().unwrap_or(u8::MAX);
        self.op_goto = insns.get("goto/L").copied().unwrap_or(u8::MAX);
        self.op_catch_exception = insns.get("catch_exception/L").copied().unwrap_or(u8::MAX);
        self.op_rvmprof_code = insns.get("rvmprof_code/ii").copied().unwrap_or(u8::MAX);
        self.op_int_return = insns.get("int_return/i").copied().unwrap_or(u8::MAX);
        self.op_ref_return = insns.get("ref_return/r").copied().unwrap_or(u8::MAX);
        self.op_float_return = insns.get("float_return/f").copied().unwrap_or(u8::MAX);
        self.op_void_return = insns.get("void_return/").copied().unwrap_or(u8::MAX);
    }

    /// pyjitpl.py:2255-2264 `finish_setup`: wire the assembler's opcode table
    /// into this staticdata object and snapshot the current `all_liveness`.
    fn finish_setup_if_needed(
        &mut self,
        insns: &indexmap::IndexMap<String, u8>,
        all_liveness: Vec<u8>,
    ) {
        let was_done = self.finish_setup_done;
        self.setup_insns(insns);
        self.liveness_info = std::sync::Arc::<[u8]>::from(all_liveness.into_boxed_slice());
        self.finish_setup_done = true;
        // pyjitpl.py:2287-2290 `finish_setup_descrs`: PyPy invokes this
        // immediately after `finish_setup(codewriter)` from
        // `warmspot.py:289`. Pyre's `ensure_finish_setup` is the lazy
        // first-trace gate, so the descr bitstring compaction runs
        // exactly once on the very first JIT entry — the
        // `finish_setup_done` flag above this block keeps subsequent
        // calls idempotent. Bitstring writes land directly on each
        // descr's `ei_index` slot via `effectinfo::compute_bitstrings`
        // (`effectinfo.py:526 descr.ei_index = …`); no process-global
        // side table.
        if !was_done {
            self.canonical.finish_setup_descrs();
        }
    }

    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// refreshes the staticdata mirror after each writer-side append.
    pub(crate) fn set_liveness_info(&mut self, bytes: Vec<u8>) {
        self.liveness_info = std::sync::Arc::<[u8]>::from(bytes.into_boxed_slice());
    }
}

/// Crate-local helper so sibling modules can refresh
/// `MetaInterpStaticData.liveness_info` without leaking the private
/// type through public API signatures.
pub(crate) fn publish_liveness_info(bytes: Vec<u8>) {
    METAINTERP_SD.with(|r| r.borrow_mut().set_liveness_info(bytes));
}

impl MetaInterpStaticData {
    #[inline]
    fn canonical_code_key_opt(code: *const ()) -> Option<usize> {
        if code.is_null() {
            return None;
        }
        // A raw `CodeObject*` IS the canonical key (jitcodes compare
        // `raw_code()` identity).  Post-#16 callers pass the raw code
        // identity directly (`ReconstructRecipe.code_ptr`); recognize it via
        // the live-wrapper registry — keyed by raw code identity — before
        // dereferencing `code` as a `PyCode` wrapper, which would read a
        // garbage field out of a raw `CodeObject`.
        if !pyre_interpreter::live_code_wrapper(code).is_null() {
            return Some(code as usize);
        }
        let raw = unsafe { pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef) };
        if raw.is_null() {
            None
        } else {
            Some(raw as usize)
        }
    }

    /// codewriter.py:67-68 parity — stamp the SD-local `idx` onto the
    /// shared `Arc<majit_metainterp::jitcode::JitCode>` carried by the
    /// payload.  Upstream RPython unconditionally writes
    /// `jitcode.index = index`; a mismatch between an existing
    /// canonical stamp and the SD-local slot is a parity violation, so
    /// rely on `JitCode::set_index`'s own assert (same value: OK,
    /// different value: panic) instead of silently skipping.
    fn stamp_payload_index(idx: i32, payload: &std::sync::Arc<crate::PyJitCode>) {
        payload.jitcode.set_index(idx as usize);
    }

    /// warmspot.py:281-282:
    ///
    /// ```python
    /// jitcodes = self.codewriter.make_jitcodes(verbose=verbose)
    /// self.metainterp_sd.jitcodes = jitcodes
    /// ```
    ///
    /// RPython runs `make_jitcodes()` once before tracing, so existing
    /// frame pointers and jitcode indices never move. Pyre still reaches
    /// this boundary from lazy portal-entry setup, so merge new payloads
    /// into the existing trace-side list instead of replacing the list.
    /// This preserves RPython's stable `metainterp_sd.jitcodes` invariant
    /// for already-captured resume data.
    fn set_jitcodes_from_make_result(&mut self, payloads: Vec<std::sync::Arc<crate::PyJitCode>>) {
        for payload in payloads {
            assert!(
                !payload.code_ptr.is_null(),
                "make_jitcodes returned a JitCode without PyCode identity"
            );
            assert!(
                !payload.is_skeleton(),
                "make_jitcodes returned an unpopulated JitCode skeleton"
            );
            let raw_key = payload.code_ptr as usize;
            let existing_pos = self.installed_jitcode_pos_for_raw_key(raw_key);
            match existing_pos {
                Some(pos) if Self::slot_accepts_payload(&self.jitcodes[pos], &payload) => {
                    let index = self.jitcodes[pos].index;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes[pos].payload = payload;
                }
                _ => {
                    let index = self.jitcodes.len() as i32;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes.push(Box::new(JitCode { index, payload }));
                }
            }
        }
    }

    /// May `slot`'s payload be replaced by `payload` in place (keeping
    /// the slot's jitcode index)?
    ///
    /// RPython's `metainterp_sd.jitcodes[i]` entries are immutable after
    /// `warmspot.py:282`: every `rd_numb` frame records `(jitcode_index,
    /// pc)` and the blackhole decoder re-derives the per-frame value
    /// count from that index's pc_map + liveness tables, so a populated
    /// entry must never change shape under an index that live resume
    /// data already references. Pyre's runtime codewriter can re-splice
    /// a graph (when a later primary compile re-queues an inlined
    /// callee's graph at the drain
    /// boundary); such a rebuild must take a FRESH index, leaving the
    /// old entry intact for old resume data. Only not-yet-populated skeletons
    /// (with no liveness any rd_numb could reference) are filled in place, matching RPython's
    /// "same JitCode object is filled later" setup-time flow.
    fn slot_accepts_payload(slot: &JitCode, payload: &std::sync::Arc<crate::PyJitCode>) -> bool {
        !slot.payload.is_populated() || std::sync::Arc::ptr_eq(&slot.payload, payload)
    }

    fn installed_jitcode_pos_for_raw_key(&self, raw_key: usize) -> Option<usize> {
        // A re-spliced graph appends a fresh entry (see
        // `slot_accepts_payload`); new frames must bind the newest build,
        // while old indices keep resolving to their original entries.
        self.jitcodes
            .iter()
            .rposition(|jitcode| unsafe { jitcode.raw_code() as usize } == raw_key)
    }

    /// codewriter.py:67-68 / call.py:155-172 adapter: install or return the
    /// trace-side wrapper for `code`. When `supplied` is present it is the
    /// populated PyJitCode Arc from CodeWriter's pending-graph drain; otherwise
    /// this creates the same empty skeleton shape as CallControl.get_jitcode()
    /// before the drain fills it.
    fn jitcode_for(
        &mut self,
        code: *const (),
        supplied: Option<std::sync::Arc<crate::PyJitCode>>,
    ) -> *const JitCode {
        let raw_key = Self::canonical_code_key_opt(code).unwrap_or(0);
        if let Some(pos) = self.installed_jitcode_pos_for_raw_key(raw_key) {
            match supplied {
                Some(payload) if Self::slot_accepts_payload(&self.jitcodes[pos], &payload) => {
                    let index = self.jitcodes[pos].index;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes[pos].payload = payload;
                }
                Some(payload) => {
                    // Re-spliced build of an already-populated entry:
                    // append under a fresh index (see `slot_accepts_payload`).
                    let index = self.jitcodes.len() as i32;
                    Self::stamp_payload_index(index, &payload);
                    self.jitcodes.push(Box::new(JitCode { index, payload }));
                    let pos = self.jitcodes.len() - 1;
                    return &*self.jitcodes[pos] as *const JitCode;
                }
                None => {}
            }
            return &*self.jitcodes[pos] as *const JitCode;
        }

        let payload = supplied.unwrap_or_else(|| {
            let raw_code = if raw_key == 0 {
                std::ptr::null()
            } else {
                raw_key as *const CodeObject
            };
            std::sync::Arc::new(crate::PyJitCode::skeleton(raw_code))
        });
        let index = self.jitcodes.len() as i32;
        Self::stamp_payload_index(index, &payload);
        let jitcode = Box::new(JitCode { index, payload });
        let ptr = &*jitcode as *const JitCode;
        self.jitcodes.push(jitcode);
        ptr
    }

    /// Return the installed SD entry for a `PyCode`.
    /// RPython's runtime lookup never compiles and never creates a
    /// skeleton here: every entry must already have arrived through
    /// `make_jitcodes()` and `warmspot.py:282`.
    fn compiled_jitcode_lookup(&self, code: *const ()) -> Option<*const JitCode> {
        let key = Self::canonical_code_key_opt(code)?;
        self.jitcodes
            .iter()
            .rev()
            .find(|jitcode| unsafe { jitcode.raw_code() as usize } == key)
            .filter(|jitcode| !jitcode.payload.is_skeleton())
            .map(|jitcode| &**jitcode as *const JitCode)
    }
}

/// RPython assembler.py:234-248 `Assembler._encode_liveness` parity:
/// intern one `[live_i, live_r, live_f]` triple in the assembler's
/// `all_liveness` buffer and return its 2-byte offset.
///
/// Writes land on `AssemblerState` (the writer side); `MetaInterpStaticData`
/// receives a fresh snapshot via the final `METAINTERP_SD.liveness_info`
/// assignment — matching `pyjitpl.py:2264`'s
/// `self.liveness_info = "".join(asm.all_liveness)` after each append.
pub fn intern_liveness(live_i: &[u8], live_r: &[u8], live_f: &[u8]) -> Option<u16> {
    use crate::assembler::ASSEMBLER_STATE;
    use majit_translate::liveness::encode_liveness;

    ensure_finish_setup();

    let snapshot = ASSEMBLER_STATE.with(|r| -> Option<(u16, Vec<u8>)> {
        let mut asm = r.borrow_mut();
        // assembler.py:149 `self.num_liveness_ops += 1` — counted once per
        // `-live-` instruction in `write_insn`, before the dedup lookup
        // inside `_encode_liveness`. The counter measures write-insn call
        // frequency, not unique-entry count.
        asm.num_liveness_ops += 1;
        let key = (live_i.to_vec(), live_r.to_vec(), live_f.to_vec());
        if let Some(&pos) = asm.all_liveness_positions.get(&key) {
            return Some((pos, asm.all_liveness.clone()));
        }
        let pos = asm.all_liveness_length;
        let encoded_i = encode_liveness(live_i);
        let encoded_r = encode_liveness(live_r);
        let encoded_f = encode_liveness(live_f);
        if live_i.len() > u8::MAX as usize
            || live_r.len() > u8::MAX as usize
            || live_f.len() > u8::MAX as usize
            || pos > u16::MAX as usize
        {
            return None;
        }
        let pos_u16 = pos as u16;
        asm.all_liveness_positions.insert(key, pos_u16);
        asm.all_liveness.push(live_i.len() as u8);
        asm.all_liveness.push(live_r.len() as u8);
        asm.all_liveness.push(live_f.len() as u8);
        asm.all_liveness.extend(encoded_i);
        asm.all_liveness.extend(encoded_r);
        asm.all_liveness.extend(encoded_f);
        asm.all_liveness_length = asm.all_liveness.len();
        Some((pos_u16, asm.all_liveness.clone()))
    })?;

    let (pos, all_liveness) = snapshot;
    METAINTERP_SD.with(|r| {
        r.borrow_mut().liveness_info = std::sync::Arc::<[u8]>::from(all_liveness.into_boxed_slice())
    });
    Some(pos)
}

/// RPython resume.py:1022 parity: read-only snapshot of
/// `metainterp_sd.liveness_info` for the ResumeDataDirectReader.
///
/// Returns a shared `Arc<[u8]>` so the caller observes the same packed
/// buffer the assembler published via `intern_liveness` /
/// `publish_liveness_info` without copying its bytes (the previous
/// implementation cloned the underlying `Vec<u8>` per BH entry — RPython
/// upstream simply reads `metainterp_sd.liveness_info` straight off the
/// staticdata object).
pub fn liveness_info_snapshot() -> std::sync::Arc<[u8]> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| std::sync::Arc::clone(&r.borrow().liveness_info))
}

/// pyjitpl.py:2236 parity: expose the staticdata `live/` opcode for callers
/// that need to decode inline liveness offsets from a JitCode.
pub fn op_live() -> u8 {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().op_live)
}

/// blackhole.py:72-74 cached control opcodes consumed by pyre's
/// blackhole-resume adapters. Returns the same `insns.get(..., -1)`
/// values RPython would have stored on `BlackholeInterpBuilder`.
pub fn blackhole_control_opcodes() -> (i32, i32, i32) {
    use crate::assembler::ASSEMBLER_STATE;

    let needs_refresh = METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let missing_live = sd.op_live == u8::MAX;
        let missing_catch_exception = sd.op_catch_exception == u8::MAX;
        let missing_rvmprof_code = sd.op_rvmprof_code == u8::MAX;
        if !sd.finish_setup_done {
            return true;
        }
        if !(missing_live || missing_catch_exception || missing_rvmprof_code) {
            return false;
        }
        ASSEMBLER_STATE.with(|a| {
            let asm = a.borrow();
            (missing_live && asm.insns.contains_key("live/"))
                || (missing_catch_exception && asm.insns.contains_key("catch_exception/L"))
                || (missing_rvmprof_code && asm.insns.contains_key("rvmprof_code/ii"))
        })
    });
    if needs_refresh {
        ensure_finish_setup();
    }
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let decode = |opcode: u8| -> i32 { if opcode == u8::MAX { -1 } else { opcode as i32 } };
        (
            decode(sd.op_live),
            decode(sd.op_catch_exception),
            decode(sd.op_rvmprof_code),
        )
    })
}

/// pyjitpl.py:2248-2249 module-level entry point for
/// `MetaInterpStaticData::setup_indirectcalltargets`.
///
/// RPython sets this at `pyjitpl.py:2262` during
/// `finish_setup(codewriter, optimizer)` by piping
/// `codewriter.assembler.indirectcalltargets` straight through.  pyre's
/// codewriter driver calls this after every `Assembler::assemble`
/// session so the staticdata reflects the assembler's latest target
/// set.  Matching shape: `Vec<Arc<JitCode>>` comes from
/// `Assembler::indirectcalltargets_vec` in `pyre-jit`.
pub fn setup_indirectcalltargets(targets: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().setup_indirectcalltargets(targets));
}

/// pyjitpl.py:2326-2343 module-level entry point for
/// `MetaInterpStaticData::bytecode_for_address`.
///
/// RPython calls this from `MIFrame.do_residual_or_indirect_call`
/// (`pyjitpl.py:2174-2186`) to check whether a `funcbox.getaddr()`
/// Const corresponds to a known indirect-call target.  pyre's tracer
/// consumer routes through here.
pub fn bytecode_for_address(
    fnaddress: usize,
) -> Option<std::sync::Arc<majit_metainterp::jitcode::JitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().bytecode_for_address(fnaddress))
}

use std::cell::RefCell;

thread_local! {
    /// warmspot.py:282: MetaInterp.staticdata (per-thread for no-GIL).
    pub(crate) static METAINTERP_SD: RefCell<MetaInterpStaticData> =
        RefCell::new(MetaInterpStaticData::new());

    /// Set when a guard's frame reports a resume coordinate the jitcode
    /// `pc_map` cannot resolve — the cross-frame snapshot gap (an inlined
    /// callee + exception-resume shape whose parent resume pc was never
    /// recorded; #124/#130).  `get_list_of_active_boxes` raises it instead of
    /// panicking; `metainterp::interpret` polls it each step and aborts the
    /// trace.  The trace is discarded before any code is installed, so the
    /// abort is side-effect free and the location interprets rather than the
    /// process crashing.
    static TRACE_ABORT_REQUESTED: std::cell::Cell<bool> = std::cell::Cell::new(false);
}

/// Request that the in-progress trait-tracer trace abort at the next step
/// boundary.  See [`TRACE_ABORT_REQUESTED`].
pub fn request_trace_abort() {
    TRACE_ABORT_REQUESTED.with(|c| c.set(true));
}

/// Read-and-clear the trace-abort request.  `metainterp::interpret` calls
/// this after every step; `trace_bytecode` calls it once at entry to drop a
/// flag a prior aborted trace may have left set.
pub fn take_trace_abort_requested() -> bool {
    TRACE_ABORT_REQUESTED.with(|c| c.replace(false))
}

/// Request a trace abort and return a bit-14-encodable stand-in for a resume
/// pc that does not fit the marker scheme (`>= AFTER_RESIDUAL_CALL_PC_FLAG`).
///
/// The bit-14 resume asserts (`trace_opcode.rs marker_aware_*_resume_pc`)
/// document that an unencodable pc is meant to fall back to the interpreter
/// via the recording loop's `catch_unwind` — but the pyre tracer runs its own
/// `metainterp::interpret`, which has no such catch, so the bare assert
/// crashed the process instead.  The cross-frame snapshot coordinate gap
/// (#124/#130) can hand such a pc (e.g. one already carrying the marker bit
/// from a corrupted cross-frame coordinate).  Requesting an abort discards the
/// guard with the (pre-install) trace, so the clamped value is never decoded.
pub fn abort_unencodable_resume_pc(pc: usize) -> usize {
    let flag = majit_ir::resumedata::AFTER_RESIDUAL_CALL_PC_FLAG as usize;
    request_trace_abort();
    pc & (flag - 1)
}

/// pyjitpl.py:2255 `MetaInterpStaticData.finish_setup` parity entry point.
///
/// RPython runs `finish_setup` once per `MetaInterpStaticData` object. Pyre's
/// `METAINTERP_SD` is thread-local, so the setup guard also lives on
/// `MetaInterpStaticData`; only the unrelated callback registration below is
/// process-global.
fn ensure_finish_setup() {
    use crate::assembler::ASSEMBLER_STATE;
    use std::sync::Once;
    static FRAME_VALUE_COUNT_INIT: Once = Once::new();
    FRAME_VALUE_COUNT_INIT.call_once(|| {
        majit_ir::resumedata::set_frame_value_count_fn(frame_value_count_at);
    });
    let (insns, all_liveness) = ASSEMBLER_STATE.with(|a| {
        let asm = a.borrow();
        (asm.insns.clone(), asm.all_liveness.clone())
    });
    METAINTERP_SD.with(|r| {
        r.borrow_mut().finish_setup_if_needed(&insns, all_liveness);
    });
}

/// pyjitpl.py:74: frame.jitcode — resolve the JitCode for the frame's code
/// object through the writer-side CallControl.get_jitcode path.
pub(crate) fn jitcode_for(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
        return existing;
    }
    if code.is_null() {
        return METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, None));
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if !raw_code.is_null() {
        let callbacks = crate::callbacks::try_get().unwrap_or_else(|| {
            panic!(
                "CallJitCallbacks not initialized while resolving JitCode for PyCode {:p}",
                code
            )
        });
        (callbacks.ensure_majit_jitcode)(raw_code, code);
        if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
            return existing;
        }
        panic!(
            "ensure_majit_jitcode did not install a populated JitCode for PyCode {:p}",
            code
        );
    }
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, None))
}

/// Install one CodeWriter-owned PyJitCode payload into trace-side
/// MetaInterpStaticData. Used by the lazy CallControl.get_jitcode drain path
/// to publish the same Arc that CallControl.jitcodes stores.
pub fn install_jitcode_for(
    code: *const (),
    payload: std::sync::Arc<crate::PyJitCode>,
) -> *const () {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, Some(payload)) as *const ())
}

/// `framework.py root_walker.walk_roots` parity for the persistent
/// `MetaInterpStaticData.jitcodes` list (warmspot.py:282
/// `self.metainterp_sd.jitcodes = jitcodes`).  Each entry's PyCode
/// wrapper is recovered from its `payload.code_ptr` via the live-wrapper
/// registry.
///
/// Intentionally not yet registered as a root walker: PyCode is
/// host-allocated via `Box::into_raw` (pycode.rs), not in the GC heap, so
/// the moving collector never sweeps or relocates it and there is nothing
/// to root.  When code objects become GC-managed this gets wired into
/// `majit_gc::register_extra_root_walker`, at which point the recovered
/// wrapper becomes a `GcRef` slot and `visit` lets a moving collector
/// rewrite it in place.
#[allow(dead_code)]
pub fn walk_jitcode_code_roots(mut visit: impl FnMut(&mut *const ())) {
    METAINTERP_SD.with(|r| {
        for jc in r.borrow_mut().jitcodes.iter() {
            let mut wrapper =
                pyre_interpreter::live_code_wrapper(jc.payload.code_ptr as *const ()) as *const ();
            visit(&mut wrapper);
        }
    });
}

/// Install the complete `CodeWriter.make_jitcodes()` result into
/// `MetaInterpStaticData.jitcodes`. Setup-time bulk publish only.
///
/// RPython warmspot.py:281-282 stores the list returned by
/// `codewriter.make_jitcodes()` directly on `metainterp_sd.jitcodes`.
/// This function is the pyre-side analog and is invoked exclusively
/// from `register_portal_jitdriver` after the JitDriver-rooted
/// `make_jitcodes()` drain. Pyre may call that boundary more than once
/// while it still has lazy portal-entry setup, so the implementation
/// merges payloads without moving existing SD entries.
pub fn install_jitcodes(jitcodes: Vec<std::sync::Arc<crate::PyJitCode>>) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        r.borrow_mut().set_jitcodes_from_make_result(jitcodes);
    });
}

/// Return the SD-local `jitcode.index` for this `PyCode`, ensuring
/// the entry through the same `jitcode_for()` / CallControl path used by
/// trace frame setup.
pub fn ensure_jitcode_index(code: *const ()) -> Option<i32> {
    if code.is_null() {
        return None;
    }
    let jitcode = jitcode_for(code);
    Some(unsafe { (*jitcode).index })
}

/// Return the `JitCode*` for this `PyCode` as an opaque pointer,
/// ensuring the entry through the same path as `ensure_jitcode_index`.
#[doc(hidden)]
pub fn ensure_jitcode_ptr(code: *const ()) -> Option<*const ()> {
    if code.is_null() {
        return None;
    }
    Some(jitcode_for(code) as *const ())
}

#[doc(hidden)]
pub fn frame_locals_cells_stack_array_ref(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    frame_locals_cells_stack_array(ctx, frame)
}

/// warmspot.py:282 metainterp_sd.jitcodes[jitcode_index]:
/// Resolve jitcode_index (sequential int from snapshot numbering)
/// to the corresponding CodeObject pointer.
pub fn code_for_jitcode_index(jitcode_index: i32) -> Option<*const ()> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes.get(idx).map(|jc| {
            pyre_interpreter::live_code_wrapper(jc.payload.code_ptr as *const ()) as *const ()
        })
    })
}

/// warmspot.py:282 `metainterp_sd.jitcodes[jitcode_index]` helper:
/// resolve the indexed runtime entry to its canonical raw `CodeObject*`.
///
/// Unlike [`code_for_jitcode_index`], this strips the wrapper round-trip
/// up front for callers that only need the graph identity to re-enter
/// `CallControl.jitcodes`.
pub fn raw_code_for_jitcode_index(jitcode_index: i32) -> Option<*const CodeObject> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes.get(idx).map(|jc| unsafe { jc.raw_code() })
    })
}

/// Resolve `MetaInterpStaticData.jitcodes[jitcode_index]` to the same
/// PyJitCode payload the trace-side frame used. This keeps blackhole /
/// resume consumers on the RPython single-store path instead of
/// re-looking-up through pyre-jit's CodeWriter side cache.
pub fn pyjitcode_for_jitcode_index(jitcode_index: i32) -> Option<std::sync::Arc<crate::PyJitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes
            .get(idx)
            .map(|jc| std::sync::Arc::clone(&jc.payload))
    })
}

/// Resolve a stored JitCode offset back to its Python instruction coordinate.
pub fn python_pc_for_jitcode_pc_public(jitcode_index: i32, offset: i32) -> Option<i32> {
    let payload = pyjitcode_for_jitcode_index(jitcode_index)?;
    Some(
        crate::jitcode_dispatch::python_pc_for_jitcode_pc(&payload.metadata, offset as usize)
            as i32,
    )
}

/// Advance a Python instruction coordinate past resume trivia when code is available.
pub fn skip_python_trivia_forward_public(
    jitcode_index: i32,
    raw_py_pc: i32,
) -> Option<(i32, bool)> {
    let payload = pyjitcode_for_jitcode_index(jitcode_index)?;
    if payload.code_ptr.is_null() {
        return Some((raw_py_pc, false));
    }
    let code = unsafe { &*payload.code_ptr };
    Some((
        crate::jitcode_dispatch::skip_python_trivia_forward(code, raw_py_pc as usize) as i32,
        true,
    ))
}

/// Translate a resume-frame pc word to a Python instruction coordinate.
///
/// A negative word (sentinel / branch-orgpc tag) has no Python coordinate; per
/// the `decode_resume_pc` contract it passes through so the caller's `pc < 0`
/// screen rejects it (the internal metadata lookups below are bounds-checked and
/// never index with the word, so no wrap results).
pub fn backxlat_py_pc(jitcode_index: i32, pc_word: i32) -> i32 {
    let fallback = majit_ir::resumedata::decode_resume_pc(pc_word).0;
    python_pc_for_jitcode_pc_public(jitcode_index, pc_word)
        .and_then(|raw_py_pc| skip_python_trivia_forward_public(jitcode_index, raw_py_pc))
        .map(|(py_pc, _)| py_pc)
        .unwrap_or(fallback)
}

/// `framework.py` `root_walker.walk_roots` hook for the boxed `Ref`
/// constants embedded in every live jitcode's `constants_r` pool.
///
/// RPython keeps these alive implicitly: a `JitCode` is a GC object and
/// its `constants_ptr` array is traced through the object graph, so a
/// boxed constant reachable only from a jitcode survives collection.
/// pyre's jitcodes live in Rust `Arc` memory (the append-only
/// `MetaInterpStaticData.jitcodes` store, warmspot.py:282), so the
/// boxed-ref constant slots need an explicit walker — the same shape
/// `MetaInterp::walk_rd_consts_refs` uses for resume-data consts.
///
/// `blackhole_from_resumedata` seeds the blackhole register file directly
/// from `jitcode.constants_r` (`init_register_files_from_runtime_jitcode`),
/// so a constant boxed object reachable only from a jitcode and swept
/// between trace executions leaves the next guard-failure resume reading a
/// freed pointer.  The constant pool is immutable after build and its
/// objects are non-moving — interpreter-routed int/float consts live in
/// the non-moving old-gen, build-time consts are `malloc_typed`-immortal —
/// so the slots are marked in place without forwarding.
pub fn walk_jitcode_constants_refs(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let data = capture_jitcode_constants_root_area();
    unsafe { walk_jitcode_constants_refs_area(data, visitor) };
}

pub fn capture_jitcode_constants_root_area() -> *const () {
    METAINTERP_SD.with(|state| state as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_jitcode_constants_root_area`], and the
/// owning thread must be quiesced.
pub unsafe fn walk_jitcode_constants_refs_area(
    data: *const (),
    visitor: &mut dyn FnMut(&mut majit_ir::GcRef),
) {
    let state = unsafe { &*(data as *const RefCell<MetaInterpStaticData>) };
    // A collection can fire while the owner holds a shared borrow. Preserve
    // the existing skip for that re-entrant single-thread case.
    let Ok(sd) = state.try_borrow() else {
        return;
    };
    for jc in sd.jitcodes.iter() {
        for &slot in jc.payload.jitcode.constants_r.iter() {
            let mut gcref = majit_ir::GcRef(slot as usize);
            visitor(&mut gcref);
        }
    }
}

/// Resolve by PyCode wrapper through the trace-side
/// MetaInterpStaticData store. Used by blackhole paths that must see
/// CodeWriter-drained entries.
pub fn pyjitcode_for_code(code: *const ()) -> Option<std::sync::Arc<crate::PyJitCode>> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let ptr = sd.compiled_jitcode_lookup(code)?;
        Some(std::sync::Arc::clone(unsafe { &(*ptr).payload }))
    })
}

/// Build a `SubJitCodeBody` view over the callee per-fn JitCode for a
/// PyCode, building+installing it on demand (`jitcode_for`) when the
/// lazy per-fn build has not run yet.  Used by full-body-walk call inlining
/// to obtain a sub-walk body for a runtime callable's code.
///
/// The `Arc<PyJitCode>` is held for the program lifetime in the append-only
/// `MetaInterpStaticData.jitcodes` store (warmspot.py:282) and its bytecode /
/// constant pools are immutable after build, so extending those slices to
/// `'static` is sound — the same justification as the per-fn arm-entry borrow
/// extension at `trace.rs:363` / `trace_opcode.rs:6735`.  Returns `None` when
/// the code is null or the on-demand build did not install a payload.
pub(crate) fn sub_jitcode_body_for_code(
    code: *const (),
) -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
    if code.is_null() {
        return None;
    }
    if jitcode_for(code).is_null() {
        return None;
    }
    let pjc = pyjitcode_for_code(code)?;
    let jc = &pjc.jitcode;
    // SAFETY: the payload lives for the program in the append-only jitcodes
    // store (a second Arc keeps the allocation alive after this local clone
    // drops); the pools are immutable post-build.
    Some(unsafe {
        crate::jitcode_dispatch::SubJitCodeBody {
            code: &*(jc.code.as_slice() as *const [u8]),
            num_regs_r: jc.num_regs_r() as usize,
            num_regs_i: jc.num_regs_i() as usize,
            num_regs_f: jc.num_regs_f() as usize,
            constants_i: &*(jc.constants_i.as_slice() as *const [i64]),
            constants_r: &*(jc.constants_r.as_slice() as *const [i64]),
            constants_f: &*(jc.constants_f.as_slice() as *const [i64]),
        }
    })
}

/// Per-PC `(color, slot)` entries the resume map records at the callee entry
/// PC (`metadata.pcdep_color_slots[0]`).  Used by full-body-walk call inlining
/// to seed positional args at the registers the callee body actually reads its
/// params from — the canonical splice regalloc does not pin local-i inputargs
/// to identity colors, so `r0..nparams` seeding reads the wrong bank slots.
/// At entry the operand stack is empty, so every entry names a local slot;
/// a param dead at entry has no entry (the body never reads it).  `None` when
/// no payload is installed or the jitcode was never colored (empty
/// `pcdep_color_slots`, a portal/skeleton install whose colors are
/// slot-identity).
pub(crate) fn sub_jitcode_entry_param_colors(code: *const ()) -> Option<Vec<(u8, u16, u16)>> {
    if code.is_null() {
        return None;
    }
    let pjc = pyjitcode_for_code(code)?;
    pjc.metadata.pcdep_color_slots.first().cloned()
}

pub(crate) type SubDescrPool = (
    &'static [DescrRef],
    &'static [majit_metainterp::jitcode::RuntimeBhDescr],
    &'static crate::jitcode_dispatch::SubJitCodeLookup,
);

/// Build (memoized on the payload itself) the per-fn descr pool a callee body
/// needs when inlined by full-body-walk call inlining: its OWN adapted
/// `descr_refs` + raw `RuntimeBhDescr` slice (for `RawDescrPool::PerFn`) +
/// `sub_jitcode_lookup`.  Mirror of the top-level diagnostic walk's per-fn
/// descr-pool construction (`trace.rs:363-400`): a callee body's
/// `d`/`j` descr operands index its OWN `exec.descrs`, not the caller's pool.
///
/// The pool lives on `PyJitCodePayload.sub_descr_pool`, not in a side table
/// keyed by object identity: RPython has no such table — the active MIFrame's
/// JitCode carries its descriptors directly — and an identity-keyed cache
/// would survive `PyJitCode::replace_with`, which refills the body in place
/// under the same outer allocation (an in-place refill overwrites the inner
/// `exec.descrs` the pool borrows). Carried on the payload, the pool is
/// dropped by the same `replace_with` that invalidates its borrows.
///
/// The returned slices are `'static` because the payload lives for the
/// program in the append-only `MetaInterpStaticData.jitcodes` store
/// (warmspot.py:282) — the same justification as `sub_jitcode_body_for_code`
/// below; the adapted `descr_refs` Vec and the lookup closure are leaked once
/// per distinct callee body (bounded by the program's callable count ×
/// refinements, not per trace attempt). `'static` coerces to the walker's
/// `'static_a`.
pub(crate) fn sub_jitcode_descr_pool_for_code(code: *const ()) -> Option<SubDescrPool> {
    use majit_metainterp::jitcode::RuntimeBhDescr;
    if code.is_null() || jitcode_for(code).is_null() {
        return None;
    }
    let pjc = pyjitcode_for_code(code)?;
    Some(*pjc.sub_descr_pool.get_or_init(|| {
        // SAFETY: the borrow is valid for `'static` because the payload is
        // retained for the program lifetime by the append-only
        // `MetaInterpStaticData.jitcodes` store, and `exec.descrs` is
        // immutable post-build. The one mutation path, `replace_with`,
        // replaces this `OnceCell` together with the body, so the pool can
        // never be returned for a body other than the one it was built from.
        let perfn_descrs: &'static [RuntimeBhDescr] =
            unsafe { &*(pjc.jitcode.exec.descrs.as_slice() as *const [RuntimeBhDescr]) };
        let descr_refs: Vec<DescrRef> = perfn_descrs
            .iter()
            .enumerate()
            .map(|(i, d)| match d {
                RuntimeBhDescr::Descr(bh) => crate::descr::make_descr_from_bh(bh),
                RuntimeBhDescr::JitCode(_)
                | RuntimeBhDescr::Call(_)
                | RuntimeBhDescr::AssemblerToken(_) => crate::descr::make_jitcode_descr(i),
            })
            .collect();
        let descr_refs: &'static [DescrRef] = Box::leak(descr_refs.into_boxed_slice());
        let lookup: Box<crate::jitcode_dispatch::SubJitCodeLookup> = Box::new(move |idx: usize| {
            perfn_descrs
                .get(idx)
                .and_then(|d| d.as_jitcode())
                .map(|jc| crate::jitcode_dispatch::SubJitCodeBody {
                    code: jc.code.as_slice(),
                    num_regs_r: jc.num_regs_r() as usize,
                    num_regs_i: jc.num_regs_i() as usize,
                    num_regs_f: jc.num_regs_f() as usize,
                    constants_i: jc.constants_i.as_slice(),
                    constants_r: jc.constants_r.as_slice(),
                    constants_f: jc.constants_f.as_slice(),
                })
        });
        let lookup: &'static crate::jitcode_dispatch::SubJitCodeLookup = Box::leak(lookup);
        (descr_refs, perfn_descrs, lookup)
    }))
}

/// `resume.py:1049` `consume_one_section` → `enumerate_vars` parity:
/// return the number of tagged values encoded for a frame at
/// (jitcode_index, pc).
///
/// Upstream `pyjitpl.py:199` / `jitcode.py:82-93`: decode the `-live-`
/// offset from the jitcode byte stream at `jitcode.get_live_vars_info(
/// pc, op_live)`, then read the three-byte `[len_i][len_r][len_f]`
/// header in `all_liveness`. Total live value count = `len_i +
/// len_r + len_f`.
///
/// Fallback: when the jitcode is still a skeleton payload (pc_map
/// empty) or has no backing CodeObject, decode via the pyre-jit-trace
/// `LiveVars` analysis over the Python bytecode. This path is used
/// for inlined callee frames whose majit_jitcode has not been built
/// at trace time.
pub fn frame_value_count_at(jitcode_index: i32, pc: i32) -> usize {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        let jc = match sd.jitcodes.get(idx) {
            Some(jc) => jc,
            None => return 0,
        };
        let payload = &jc.payload;
        // Snapshot publication stores only a decodable JitCode `-live-`
        // coordinate. An unrepresentable coordinate must have declined during
        // capture, before it could reach this frame-boundary decoder.
        let resolved_jit_pc: Option<usize> = if pc >= 0
            && payload
                .jitcode
                .can_decode_live_vars(pc as usize, sd.op_live)
        {
            Some(pc as usize)
        } else {
            None
        };
        if let Some(jit_pc) = resolved_jit_pc {
            let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
            let all_liveness: &[u8] = &sd.liveness_info;
            if off + 2 < all_liveness.len() {
                let length_i = all_liveness[off] as usize;
                let length_r = all_liveness[off + 1] as usize;
                let length_f = all_liveness[off + 2] as usize;
                return length_i + length_r + length_f;
            }
        }
        // A published non-decodable coordinate violates the capture contract.
        // This remains a fail-loud internal invariant, not a fallback path.
        panic!(
            "frame_value_count_at: fallback hit for jitcode_index={} pc={} \
             (pc_map.len={}, all_liveness.len={}). Phase X-0/X-1 removed \
             all known triggers — further hits are bugs.",
            jitcode_index,
            pc,
            payload.metadata.first_jit_pc_by_py_pc.len(),
            sd.liveness_info.len(),
        );
    })
}

/// Resolve the JitCode byte offset the full-body walk should RESUME at for a
/// bridge whose guard carried `carried_jitcode_pc`. Mirrors the blackhole's
/// `resolve_resume_pc_with_jitcode_pc` (`call_jit.rs` `resolve_jitcode`): a
/// kept-stack branch guard resumes at its OWN mid-opcode jitcode offset (the
/// `goto_if_not`), not the opcode-entry marker `pc_map[py_pc]` — re-executing
/// the whole opcode from entry would read abstract-register colors dead at the
/// guard. Returns `None` when no coordinate resolves (the caller keeps the
/// caller declines).
pub fn resolve_bridge_walk_entry_at(
    jitcode_index: i32,
    pc: i32,
    carried_jitcode_pc: i32,
) -> Option<usize> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let jc = sd.jitcodes.get(jitcode_index as usize)?;
        jc.payload
            .resolve_resume_pc_with_jitcode_pc(pc, carried_jitcode_pc, sd.op_live)
    })
}

/// virtualizable.py:86-98 `read_boxes` parity: assemble the
/// `virtualizable_boxes` layout the tracing-time vable mirror expects
/// and hand it to `TraceCtx::init_virtualizable_boxes`. Used by both
/// the root portal seed (`initialize_sym`) and the bridge entry rebuild
/// (`setup_bridge_sym`). Matches RPython `virtualizable.py:139
/// load_list_of_boxes`, producing `[static_fields..., array_items...,
/// vable_box]` (the trailing `vable_box` is appended by
/// `init_virtualizable_boxes`).
///
/// * `scalar_oprefs` — the NUM_VABLE_SCALARS static field OpRefs in
///   declaration order (last_instr, pycode, valuestackdepth, debugdata,
///   lastblock, w_globals). Excludes both the frame-identity slot and
///   any non-vable extra reds (e.g. `ec`); virtualizable_boxes only
///   carries the vable static fields plus array items.
/// * `array_items` — pre-resolved OpRefs for the heap-side
///   `locals_cells_stack_w` array. Entries past `array_len` are ignored;
///   short lists are padded with a shared const-NULL Ref so the vable
///   mirror covers every interpreter-visible slot (virtualizable.py:109
///   `assert len(boxes) == i + 1`).
/// * `array_len` — the runtime PyFrame's
///   `locals_cells_stack_w.len()`; `init_virtualizable_boxes` stores this
///   as the sole entry in `virtualizable_array_lengths`.
pub(crate) fn seed_virtualizable_boxes(
    ctx: &mut TraceCtx,
    vable_ref: OpRef,
    vable_ref_value: majit_ir::Value,
    scalar_oprefs: &[OpRef],
    array_items: &[OpRef],
    array_len: usize,
    input_values: &[majit_ir::Value],
    heap_ptr: *const u8,
) {
    let info = crate::frame_layout::build_pyframe_virtualizable_info();
    let expected_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;
    assert_eq!(
        scalar_oprefs.len(),
        expected_scalars,
        "seed_virtualizable_boxes: scalar_oprefs.len() must equal NUM_VABLE_SCALARS",
    );
    let mut input_oprefs: Vec<OpRef> = Vec::with_capacity(expected_scalars + array_len);
    input_oprefs.extend_from_slice(scalar_oprefs);
    let taken = array_items.len().min(array_len);
    input_oprefs.extend_from_slice(&array_items[..taken]);
    if taken < array_len {
        let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
        for _ in taken..array_len {
            input_oprefs.push(null_ref);
        }
    }
    let array_lengths = vec![array_len];
    // virtualizable.py:139 load_list_of_boxes parity: the concrete half of
    // virtualizable_boxes is sourced from the caller (heap read for portal
    // entry / resume-data stream for bridge entry), never synthesized here.
    // Callers pass an empty slice to disable the concrete shadow
    // (unit-test / init-before-run path).
    ctx.init_virtualizable_boxes(
        &info,
        vable_ref,
        vable_ref_value,
        &input_oprefs,
        input_values,
        &array_lengths,
    );
    // pyjitpl.py:3446 synchronize_virtualizable parity: cache the live heap
    // pointer on TraceCtx so subsequent vable setfield / setarrayitem calls
    // can mirror their shadow updates into the live virtualizable.  Pass
    // a null pointer to disable (unit-test / init-before-run path).
    ctx.set_virtualizable_heap_ptr(heap_ptr);
}

/// Decode a raw `vinfo.read_all_boxes` entry into the typed Value a
/// virtualizable shadow slot expects.  Local helper that mirrors
/// `majit-metainterp::pyjitpl::heap_value_for` so both seed sites (root
/// portal in `initialize_virtualizable` and bridge entry here) use the
/// same raw-bit → typed-Value rule.
fn value_for_slot(ty: Type, bits: i64) -> majit_ir::Value {
    match ty {
        Type::Int => majit_ir::Value::Int(bits),
        Type::Float => majit_ir::Value::Float(f64::from_bits(bits as u64)),
        Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(bits as usize)),
        Type::Void => majit_ir::Value::Void,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameLivenessRegIndices {
    pub int: Vec<u32>,
    pub ref_: Vec<u32>,
    pub float: Vec<u32>,
}

impl FrameLivenessRegIndices {
    pub fn total_len(&self) -> usize {
        self.int.len() + self.ref_.len() + self.float.len()
    }

    pub fn flattened(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.total_len());
        out.extend_from_slice(&self.int);
        out.extend_from_slice(&self.ref_);
        out.extend_from_slice(&self.float);
        out
    }
}

/// Test/diagnostic-only direct JitCode `-live-` query: return live register
/// indices split by the three liveness banks.
/// RPython writes decoded values through `_callback_i/_callback_r/_callback_f`
/// into `registers_i/r/f[index]`; keeping the banks separate prevents Ref-only
/// semantic-slot remapping from swallowing Int/Float slots.
pub fn frame_liveness_reg_indices_by_bank_at(
    jitcode_index: i32,
    pc: i32,
) -> FrameLivenessRegIndices {
    frame_liveness_reg_indices_by_bank_from_pc(jitcode_index, pc)
}

/// `#124` Approach B encoder/decoder liveness query: identical to
/// [`frame_liveness_reg_indices_by_bank_at`] but resolves the JitCode
/// coordinate through [`PyJitCode::resolve_resume_pc_with_jitcode_pc`]. The
/// snapshot encoder (`collect_outer_active_boxes`) and the bridge/inline
/// decoders (`setup_bridge_sym`, `rebuild_inline_callee`) call this with
/// the SAME carried word so their register color sets agree.
pub fn frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(
    jitcode_index: i32,
    pc: i32,
    carried_jitcode_pc: i32,
) -> FrameLivenessRegIndices {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        let Some(jc) = sd.jitcodes.get(idx) else {
            return FrameLivenessRegIndices::default();
        };
        let payload = &jc.payload;
        // No Python-pc reconstruction is available here: an absent carried
        // coordinate returns the empty result to its decline-aware caller.
        let resolved_jit_pc: Option<usize> =
            payload.resolve_resume_pc_with_jitcode_pc(pc, carried_jitcode_pc, sd.op_live);
        let Some(jit_pc) = resolved_jit_pc else {
            return FrameLivenessRegIndices::default();
        };
        let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
        let all_liveness = &sd.liveness_info;
        if off + 2 >= all_liveness.len() {
            return FrameLivenessRegIndices::default();
        }
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        // Diagnostic removed — use post-decode logging instead.
        let mut cursor = off + 3;
        use majit_translate::liveness::LivenessIterator;

        fn read_bank(cursor: &mut usize, length: u32, all_liveness: &[u8]) -> Vec<u32> {
            if length == 0 {
                return Vec::new();
            }
            let mut it = LivenessIterator::new(*cursor, length, all_liveness);
            let mut out = Vec::with_capacity(length as usize);
            while let Some(reg_idx) = it.next() {
                out.push(reg_idx);
            }
            *cursor = it.offset;
            out
        }

        let int = read_bank(&mut cursor, length_i, all_liveness);
        let ref_ = read_bank(&mut cursor, length_r, all_liveness);
        let float = read_bank(&mut cursor, length_f, all_liveness);
        FrameLivenessRegIndices { int, ref_, float }
    })
}

/// Post-flip decode entry: `pc` is already the JitCode byte offset (the flip
/// stores the resolved offset in the frame pc word), so resolve the liveness
/// coordinate from `pc` itself — no carried twin needed. Byte-identical to the
/// `_with_jitcode_pc` form for every frame whose stored `pc` equals the twin
/// offset (all corpus frames).
pub(crate) fn frame_liveness_reg_indices_by_bank_from_pc(
    jitcode_index: i32,
    pc: i32,
) -> FrameLivenessRegIndices {
    frame_liveness_reg_indices_by_bank_at_with_jitcode_pc(jitcode_index, pc, pc)
}

/// Whether the resume frame's pc word names a resolved JitCode `-live-` offset
/// (the flip stores a decodable offset when the guard's resume marker resolves,
/// else the raw Python pc). Post-flip this reproduces the retired
/// `frame.jitcode_pc != NO_JITCODE_PC` twin test: the twin is non-sentinel iff a
/// marker resolved iff `frame.pc` is a decodable offset.
pub(crate) fn frame_pc_is_resolved_offset_at(jitcode_index: i32, pc: i32) -> bool {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let Some(jc) = sd.jitcodes.get(jitcode_index as usize) else {
            return false;
        };
        pc >= 0
            && jc
                .payload
                .jitcode
                .can_decode_live_vars(pc as usize, sd.op_live)
    })
}

pub fn frame_liveness_reg_indices_at(jitcode_index: i32, pc: i32) -> Vec<u32> {
    frame_liveness_reg_indices_by_bank_at(jitcode_index, pc).flattened()
}

/// Test-only fixture seeder for `_with_compiled_trace_jitcode` unit tests.
/// Gated behind `#[cfg(any(test, feature = "test-support"))]` so production
/// builds cannot reach the synthetic register-bank fill — RPython has no
/// counterpart and live holes get filled with `0xfeed` / `0` dummy consts
/// that would corrupt a real trace if invoked outside fixtures.
///
/// `pyjitpl.py:218-225 get_list_of_active_boxes` reads each live register
/// directly from its kind-specific bank (`registers_i[reg]` /
/// `registers_r[reg]` / `registers_f[reg]`); commit
/// `3fdb617f5d1` removed pyre's prior `registers_r_semantic` fallback to
/// match that contract. Production tracers fill the kind banks via
/// `bcd_op` dispatch as the trace is recorded, but unit-test fixtures
/// build a `PyreSym` directly through `from_test_state` and never run the
/// dispatch loop — without seeding, `current_fail_args` returns a vector
/// full of `OpRef::NONE` holes.
///
/// This helper mirrors what production code does:
///   1. `setup_kind_register_banks` to size the three banks +
///      copy-constants per `pyjitpl.py:97-119`.
///   2. Place each `(stack_depth, opref)` at the semantic frame-mirror
///      slot `registers_r[nlocals + depth]`. Production guard capture
///      then materializes the color-indexed Ref bank from the same
///      semantic/vable source before reading liveness.
///   3. Fill any still-`OpRef::NONE` Int/Float bank slot listed in
///      `live_pc`'s bank-split liveness with a typed dummy constant.
///      Ref slots are provided by the caller-provided stack slot map above.
#[cfg(any(test, feature = "test-support"))]
pub fn seed_compiled_trace_jitcode_test_state(
    sym: &mut PyreSym,
    ctx: &mut TraceCtx,
    jitcode_index: i32,
    live_jit_pc: i32,
    stack_slots: &[(usize, OpRef)],
) {
    if !sym.jitcode.is_null() {
        sym.setup_kind_register_banks(ctx);
    }

    for &(depth, opref) in stack_slots {
        // Match production stack writes: keep `registers_r` as a
        // semantic frame mirror. The encoder builds the color-indexed
        // bank from this mirror/vable view at snapshot time.
        let reg_idx = crate::trace_opcode::stack_slot_reg_idx(sym, depth);
        if reg_idx >= sym.registers_r.len() {
            sym.registers_r.resize(reg_idx + 1, OpRef::NONE);
        }
        sym.registers_r[reg_idx] = opref;
    }

    let banks = frame_liveness_reg_indices_by_bank_from_pc(jitcode_index, live_jit_pc);
    for &reg in &banks.int {
        let r = reg as usize;
        if r >= sym.registers_i.len() {
            sym.registers_i.resize(r + 1, OpRef::NONE);
        }
        if sym.registers_i[r].is_none() {
            sym.registers_i[r] = ctx.const_int(0xfeed);
        }
    }
    for &reg in &banks.float {
        let r = reg as usize;
        if r >= sym.registers_f.len() {
            sym.registers_f.resize(r + 1, OpRef::NONE);
        }
        if sym.registers_f[r].is_none() {
            sym.registers_f[r] = ctx.const_float(0);
        }
    }
}

/// Per-PC `(color, semantic_slot)` resume entries for the registered
/// jitcode at `jitcode_index` (`metadata.pcdep_color_slots[py_pc]`).
/// Empty when the index or PC is out of range, or the jitcode was never
/// colored (portal/skeleton installs).
///
/// Used by tests + tooling that need to translate a physical frame slot
/// (local `i` for `i < code.varnames.len()`, `metadata.stack_base + d` for
/// operand-stack depth `d`) into the post-rename register color the dispatcher would touch
/// at a given PC — colors are per-program-point, so a flat
/// slot-arithmetic lookup does not exist.
pub fn pcdep_color_slots_at(jitcode_index: i32, py_pc: i32) -> Vec<(u8, u16, u16)> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .and_then(|jc| {
                jc.payload
                    .metadata
                    .pcdep_color_slots
                    .get(usize::try_from(py_pc).ok()?)
                    .cloned()
            })
            .unwrap_or_default()
    })
}

/// The post-regalloc Ref-bank color of the call-result operand-stack slot
/// (top of stack) at `pc` — the not-yet-produced slot the inline multiframe
/// capture (`compute_inline_caller_frame`) nulls before serializing the
/// paused caller frame.  Returns `None` when the stack is empty there
/// (`u16::MAX` sentinel) or `pc` / the jitcode is out of range.  Sources the
/// codewriter-precomputed `metadata.result_color_at_pc`, so the capture no
/// longer reads the flat `stack_slot_color_map` at runtime.
pub fn result_color_at_pc_at(jitcode_index: i32, pc: usize) -> Option<usize> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let jc = sd.jitcodes.get(jitcode_index as usize)?;
        let c = jc.payload.metadata.result_color_at_pc.get(pc).copied()?;
        (c != u16::MAX).then_some(c as usize)
    })
}

/// Whether `pcdep_color_slots[pc]` maps register color `color` to a semantic
/// frame slot — i.e. the register allocator assigned this color to a real
/// local/stack value at `pc`, so the color does NOT carry its force-alived
/// portal-red meaning there (`collect_outer_active_boxes` scratch gate /
/// `setup_bridge_sym` ec-seed gate).
pub fn pcdep_color_names_frame_slot_at(jitcode_index: i32, pc: usize, color: u16) -> bool {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .and_then(|jc| jc.payload.metadata.pcdep_color_slots.get(pc))
            .is_some_and(|entries| entries.iter().any(|&(b, c, _)| b == 1 && c == color))
    })
}

/// Depth-based `valuestackdepth` for `w_code` at `py_pc`:
/// `nlocals + ncells + depth_at_py_pc[py_pc]`.  Mirrors the encoder's
/// published vsd (the `jitcode_dispatch` valuestackdepth publish).  The
/// stack base is `varnames + ncells` (`pyframe.py:111 valuestackdepth =
/// co_nlocals + ncellvars + nfreevars`), not `varnames` alone, so a code
/// object with cells/freevars (a closure/nested function) is not
/// under-counted — see `concrete_nlocals`.
///
/// A multi-frame (inlined-callee) guard restores the whole virtualizable
/// positionally via `write_from_resume_data_partial`, which writes the
/// CHAIN frame's `valuestackdepth` — the OUTER section's depth.  When the
/// resume pc is then overridden to the innermost section's `py_pc`, the
/// physical frame's vsd must be corrected to that section's depth, else
/// the interpreter resumes at the inner pc carrying the outer depth (an
/// over-count that materializes a stray operand slot).  Returns `None`
/// when the code or the liveness entry for `py_pc` is missing.
pub fn depth_based_vsd_for_wcode(w_code: usize, py_pc: usize) -> Option<usize> {
    if w_code == 0 {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let code = unsafe { &*raw_code };
    let stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    let depth = crate::liveness::liveness_for(raw_code)
        .depth_at_py_pc()
        .get(py_pc)
        .copied()?;
    Some(stack_base + depth as usize)
}

/// Inputs `setup_bridge_sym` needs to rebuild the slot-indexed semantic
/// register file from the color-indexed resume decode of the parent
/// jitcode at the guard-failure PC. Mirrors the input block
/// `collect_outer_active_boxes` (jitcode_dispatch.rs) builds for the
/// encode side, but keyed by `jitcode_index` rather than a live
/// `sym.jitcode`, so the bridge decoder can invert each live Ref color to
/// its `locals_cells_stack_w` slot via `semantic_ref_slot_for_reg_color`.
///
/// `has_color_map` is `false`
/// for a jitcode the codewriter never colored (empty `pcdep_color_slots`) —
/// byte-identical over the corpus to the retired flat-map identity test
/// (`local_color_map.is_empty() && stack_color_map.is_empty()`), which a
/// zero-local frame owning a freely-colored operand stack correctly failed
/// (its `stack_color_map` was non-empty → its `pcdep_color_slots` is too).
pub(crate) struct BridgeSemanticMaps {
    /// `true` when the codewriter colored this jitcode (non-empty
    /// `pcdep_color_slots`); drives the per-slot pcdep inversion. `false` for
    /// skeletons whose color bank is the slot mirror.
    pub has_color_map: bool,
    pub stack_depth_at_pc: usize,
    /// #348 Part (2): per-PC `(color, slot)` entries at the resume PC.
    /// Non-empty only for gated jitcodes; when present it is the
    /// authoritative color→slot inversion (the same per-program-point color
    /// space the `-live-` markers and encode side use), superseding the flat
    /// `local_color_map` / `stack_color_map` for this bridge.
    /// Tuples: `(bank, color, slot)` where bank = Kind::index()
    /// (0=Int, 1=Ref, 2=Float).
    pub pcdep_entries: Vec<(u8, u16, u16)>,
}

pub(crate) fn bridge_semantic_maps_at(jitcode_index: i32, pc: i32) -> BridgeSemanticMaps {
    bridge_semantic_maps_at_with_jitcode_pc(jitcode_index, pc, majit_ir::resumedata::NO_JITCODE_PC)
}

/// When a kept-stack branch guard carries the guard's own jitcode
/// coordinate (`jitcode_pc != NO_JITCODE_PC`), the pcdep/depth tables
/// must be keyed at the GUARD's Python PC — the encode side
/// (`collect_outer_active_boxes`) already keys its pcdep by
/// `liveness_py_pc = guard_py_pc`, and the liveness decode
/// (`frame_liveness_reg_indices_by_bank_at_with_jitcode_pc`) resolves
/// through the carried `jitcode_pc`. Without this the decode-side
/// color→slot inversion reads the merge-target PC's pcdep (where the
/// computed kept temp is dead), leaving the temp as `OpRef::NONE`.
pub(crate) fn bridge_semantic_maps_at_with_jitcode_pc(
    jitcode_index: i32,
    pc: i32,
    jitcode_pc: i32,
) -> BridgeSemanticMaps {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let Some(jc) = sd.jitcodes.get(jitcode_index as usize) else {
            return BridgeSemanticMaps {
                has_color_map: false,
                stack_depth_at_pc: 0,
                pcdep_entries: Vec::new(),
            };
        };
        let payload = &jc.payload;
        // #73 S3.5: expand a tagged branch `orgpc` word to the block-head marker
        // BEFORE the `>= 0` / offset uses below — a tagged NEGATIVE word cast to
        // usize would otherwise be a huge OOB index. No-op for offsets /
        // NO_JITCODE_PC (flip-off), so byte-identical when off.
        let jitcode_pc = crate::jitcode_dispatch::expand_branch_carried(payload, jitcode_pc);
        // The rd_numb pc word may carry an after-residual-call marker;
        // recover the plain Python PC for the py_pc-keyed liveness/depth
        // tables (same decode as
        // `frame_liveness_reg_indices_by_bank_at_with_jitcode_pc`).
        //
        // Coordinate note (#423): the Ref bank is decoded marker-aware (at the
        // post-call jitcode pc via the carried `jitcode_pc`), but `pcdep_entries`
        // / `stack_depth_at_pc` / `live_locals` here key by the marker-STRIPPED
        // `real_pc` (= the CALL `orgpc` for an after-residual-call guard). The
        // encode side (`get_list_of_active_boxes`) keys its pcdep by
        // `live_pc = fallthrough_pc` (the post-call pc). For a kept operand-stack
        // Ref below the call window the slot index and the (flat-base) color are
        // identical at `orgpc` and `fallthrough_pc`, and the entries that DO
        // differ (the call-window arg slots present only at the pre-call depth)
        // sit above the post-call `valuestackdepth` and are clamped out by the
        // `s >= semantic_prefix_len` bound in `setup_bridge_sym`, so the
        // inversion agrees in practice (residual-call-in-try kept-Ref corpus is
        // byte-exact gate-on vs gate-off on both backends). A confirmed trigger
        // would be fixed by keying these three tables off the same post-call
        // coordinate the Ref bank uses when the marker is set; that change is
        // deferred to the symbolic-stack work (#423) since an unvalidated
        // resume-coordinate flip can itself miscompile.
        //
        // When the carried `jitcode_pc` is set (kept-stack branch
        // guard), resolve the guard's Python PC from the jitcode coordinate
        // and use it for the pcdep/depth lookup, matching the encode side's
        // `liveness_py_pc = guard_py_pc` keying. The guard PC is where the
        // computed kept operand-stack temps are live; at the merge-target PC
        // they've been consumed and carry no pcdep entry.
        // Index the py_pc-keyed depth/pcdep tables at a resolved Python PC.
        let via_py_pc = |rp: usize| {
            let depth = payload.metadata.depth_at_py_pc.get(rp).copied().unwrap_or(0) as usize;
            let pcdep = payload
                .metadata
                .pcdep_color_slots
                .get(rp)
                .cloned()
                .unwrap_or_default();
            (depth, pcdep)
        };
        let (stack_depth_at_pc, pcdep_entries) = if jitcode_pc
            != majit_ir::resumedata::NO_JITCODE_PC
            && jitcode_pc >= 0
        {
            let jp = jitcode_pc as usize;
            // Validate with `can_decode_live_vars` — symmetric with the
            // liveness decode in `resolve_resume_pc_with_jitcode_pc`.
            // A non-decodable carried coordinate falls back to the
            // merge-target PC so liveness and pcdep key the same point.
            if payload.jitcode.can_decode_live_vars(jp, sd.op_live) {
                // Decode-identity path: source depth/pcdep directly from the
                // carried genuine `jitcode_pc` via the predecessor-keyed twins,
                // bypassing the `python_pc_for_jitcode_pc` re-inversion. Valid
                // exactly where the twins are populated (a colored jitcode); the
                // equality with the py_pc tables is what `PYRE_PCMAP_BRIDGE_AUDIT`
                // certifies. Empty twins fall through to the re-inversion.
                let via_twin = match (
                    payload.depth_for_jitcode_pc_pred(jp),
                    payload.pcdep_for_jitcode_pc(jp),
                ) {
                    (Some(depth), Some(pcdep)) => Some((depth as usize, pcdep)),
                    _ => None,
                };
                match via_twin {
                    Some(pair) => pair,
                    None => {
                        let rp =
                            crate::jitcode_dispatch::python_pc_for_jitcode_pc(&payload.metadata, jp)
                                as usize;
                        // task#50 phase-1: certify the predecessor-keyed jitcode-pc
                        // twins reproduce the py_pc-indexed pcdep/depth this seam
                        // reads via the re-inversion. When the carried `jitcode_pc`
                        // resolves to `rp`, the twins keyed on `jp` must equal the
                        // tables keyed on `rp` (both compile-time derivations of the
                        // same coordinates). This is the precondition certificate for
                        // the decode-identity path above. Off in production.
                        if crate::jitcode_dispatch::bridge_audit_enabled() {
                            if let Some(twin) = payload.pcdep_for_jitcode_pc(jp) {
                                let via_py = payload
                                    .metadata
                                    .pcdep_color_slots
                                    .get(rp)
                                    .cloned()
                                    .unwrap_or_default();
                                assert_eq!(
                                    twin, via_py,
                                    "pcdep_by_jit_pc diverges from pcdep_color_slots at jp={jp} rp={rp}"
                                );
                            }
                            if let Some(twin_d) = payload.depth_for_jitcode_pc_pred(jp) {
                                let via_py_d = payload
                                    .metadata
                                    .depth_at_py_pc
                                    .get(rp)
                                    .copied()
                                    .unwrap_or(0);
                                assert_eq!(
                                    twin_d, via_py_d,
                                    "depth_pred_by_jit_pc diverges from depth_at_py_pc at jp={jp} rp={rp}"
                                );
                            }
                        }
                        via_py_pc(rp)
                    }
                }
            } else {
                via_py_pc(majit_ir::resumedata::decode_resume_pc(pc).0 as usize)
            }
        } else {
            via_py_pc(majit_ir::resumedata::decode_resume_pc(pc).0 as usize)
        };
        BridgeSemanticMaps {
            // #73: the codewriter colored this jitcode iff `pcdep_color_slots`
            // is non-empty — the field-free replacement for the retired flat
            // `local_color_map.is_empty() && stack_color_map.is_empty()` test.
            has_color_map: !payload.metadata.pcdep_color_slots.is_empty(),
            stack_depth_at_pc,
            pcdep_entries,
        }
    })
}

pub(crate) fn bridge_semantic_maps_from_pc(jitcode_index: i32, pc: i32) -> BridgeSemanticMaps {
    bridge_semantic_maps_at_with_jitcode_pc(jitcode_index, pc, pc)
}

/// Per-PC operand-stack Ref CONSTANTS (`(semantic_slot, raw_ref)`) at the
/// resume PC of a jitcode. The pcdep color map records live Variables only;
/// `reconstruct_inline_recipe` uses this to refill the registerless constant
/// slots an inlined-callee guard resume leaves empty after the color→slot
/// inversion. Keyed by the guard's Python PC — when the carried
/// `jitcode_pc` is set, resolves through `python_pc_for_jitcode_pc` so
/// the constant lookup uses the same coordinate as `pcdep_entries` and
/// the liveness decode.
pub(crate) fn const_ref_slots_at_pc_at(
    jitcode_index: i32,
    pc: i32,
    jitcode_pc: i32,
) -> Vec<(u16, i64)> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let Some(jc) = sd.jitcodes.get(jitcode_index as usize) else {
            return Vec::new();
        };
        // #73 S3.5: expand a tagged branch `orgpc` word to the block-head marker
        // before the `>= 0` / offset uses below. No-op for offsets /
        // NO_JITCODE_PC (flip-off), so byte-identical when off.
        let jitcode_pc = crate::jitcode_dispatch::expand_branch_carried(&jc.payload, jitcode_pc);
        let real_pc = if jitcode_pc != majit_ir::resumedata::NO_JITCODE_PC && jitcode_pc >= 0 {
            let jp = jitcode_pc as usize;
            if jc.payload.jitcode.can_decode_live_vars(jp, sd.op_live) {
                // gh#73 S3.2: source the const slots directly from the carried
                // genuine `jitcode_pc` via the predecessor-keyed twin, bypassing
                // the `python_pc_for_jitcode_pc` re-inversion — the same
                // decode-identity shape the pcdep/depth twins use at
                // `bridge_semantic_maps_at_with_jitcode_pc`. A colored jitcode
                // returns `Some` (an empty slot list is a legitimate hit); an
                // empty twin (skeleton / fixture) returns `None` and falls
                // through to the re-inversion, which still keys the populated
                // `const_ref_slots_at_pc`. Equal by construction (built in the
                // same `by_off` loop, same predecessor keying, as the twins
                // check.py certifies on the hot bridge path).
                if let Some(slots) = jc.payload.const_ref_slots_for_jitcode_pc(jp) {
                    return slots;
                }
                crate::jitcode_dispatch::python_pc_for_jitcode_pc(&jc.payload.metadata, jp) as usize
            } else {
                majit_ir::resumedata::decode_resume_pc(pc).0 as usize
            }
        } else {
            majit_ir::resumedata::decode_resume_pc(pc).0 as usize
        };
        jc.payload
            .metadata
            .const_ref_slots_at_pc
            .get(real_pc)
            .cloned()
            .unwrap_or_default()
    })
}

pub(crate) fn const_ref_slots_from_pc(jitcode_index: i32, pc: i32) -> Vec<(u16, i64)> {
    const_ref_slots_at_pc_at(jitcode_index, pc, pc)
}

/// Return the post-regalloc Ref-bank colors of the portal red args
/// (`pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`) for
/// the registered jitcode at `jitcode_index`. Both are `u16::MAX` for skeletons.
///
/// Used by bridge resume to thread the ec OpRef from
/// `sym.registers_r[portal_ec_reg]` (populated by the liveness-driven
/// `consume_boxes` fill) back into `sym.execution_context`, mirroring
/// `resume.py:1077-1081 _callback_r` which writes the same slot in
/// RPython's BH register bank.
pub fn portal_red_regs_at(jitcode_index: i32) -> (u16, u16) {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| {
                (
                    jc.payload.metadata.portal_frame_reg,
                    jc.payload.metadata.portal_ec_reg,
                )
            })
            .unwrap_or((u16::MAX, u16::MAX))
    })
}

pub fn built_as_portal_at(jitcode_index: i32) -> bool {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        sd.jitcodes
            .get(jitcode_index as usize)
            .map(|jc| jc.payload.metadata.built_as_portal)
            .unwrap_or(false)
    })
}

/// Map a post-regalloc Ref-bank color back to the semantic
/// `locals_cells_stack_w` slot it denotes at the current PC.
///
/// **Decoder-only**: the encoder
/// (`get_list_of_active_boxes`) reads `registers_r[color]`
/// directly and derives `semantic_idx` inline.  This function is
/// still used by `restore_guard_failure_values` (the decoder) which
/// needs the semantic slot index to call `set_local_at` /
/// `set_stack_at` on the concrete PyFrame.
///
/// After stack-slot pinning removal, stack slots are no longer forced to
/// occupy colors `nlocals + d`; the reverse lookup consults the per-PC
/// `pcdep_color_slots` entries, bounded to the LIVE stack prefix at the
/// current PC. Only if no live stack slot owns the color can the color
/// resolve to a local slot.
pub(crate) fn semantic_ref_slot_for_reg_color(
    nlocals: usize,
    stack_only: usize,
    pcdep_entries: &[(u8, u16, u16)],
    reg: usize,
) -> Option<usize> {
    semantic_slot_for_reg_color(nlocals, stack_only, pcdep_entries, 1, reg)
}

/// Per-PC `(bank, color, slot)` map inversion: given a register bank and
/// color, return the semantic `locals_cells_stack_w` slot it maps to at the
/// current PC. Prefer stack slots over locals (stack_match.or(local_match)).
pub(crate) fn semantic_slot_for_reg_color(
    nlocals: usize,
    stack_only: usize,
    pcdep_entries: &[(u8, u16, u16)],
    bank: u8,
    reg: usize,
) -> Option<usize> {
    // The per-PC `(bank, color, slot)` map is the authoritative color→slot
    // inversion at this resume PC — each slot's TRUE per-program-point
    // color rather than a flat one-color-per-slot label. Prefer the
    // smallest operand-stack slot carrying this color, else the smallest
    // local slot (entries are sorted by `(bank, color, slot)`, so locals
    // precede stack within a color). A stack slot counts only while it is
    // below the live stack depth (`stack_only`) at THIS resume — the
    // per-PC entries are gated by the compile-time depth, which can
    // exceed the runtime `stack_only` at a residual-call fallthrough.
    //
    // An empty map yields None (no live color owns this slot at this pc).
    let mut local_match: Option<usize> = None;
    let mut stack_match: Option<usize> = None;
    for &(b, color, slot) in pcdep_entries {
        if b != bank || color as usize != reg {
            continue;
        }
        let s = slot as usize;
        if s >= nlocals {
            if s - nlocals < stack_only && stack_match.is_none() {
                stack_match = Some(s);
            }
        } else if local_match.is_none() {
            local_match = Some(s);
        }
    }
    stack_match.or(local_match)
}

/// Bank-generic inverse of [`semantic_slot_for_reg_color`]: given a semantic
/// `locals_cells_stack_w` slot and a bank tag (`pyjitcode.rs:198-204`:
/// 0=Int, 1=Ref, 2=Float), return the color of that bank owning the slot at
/// this PC per the `(bank, color, slot)` map. Returns the smallest matching
/// color; `None` when no live color of that bank owns the slot at this PC.
fn semantic_slot_color_for_slot(
    pcdep_entries: &[(u8, u16, u16)],
    slot: usize,
    bank: u8,
) -> Option<usize> {
    let mut best: Option<usize> = None;
    for &(b, color, s) in pcdep_entries {
        if b != bank || s as usize != slot {
            continue;
        }
        let c = color as usize;
        best = Some(best.map_or(c, |cur: usize| cur.min(c)));
    }
    best
}

/// Inverse of [`semantic_slot_for_reg_color`] for the Ref bank: given a
/// semantic `locals_cells_stack_w` slot, return the Ref-bank color that owns
/// it at this PC per the `(bank, color, slot)` map. Used by the deep-kept
/// operand-stack recovery in `walker_capture_snapshot_for_last_guard_impl` to
/// name the guard-PC register (`registers_r[color]`) holding a kept operand-
/// stack slot the walk mirror lost. Returns the smallest matching color;
/// `None` when no live Ref color owns the slot at this PC.
pub(crate) fn semantic_slot_color_for_ref_slot(
    pcdep_entries: &[(u8, u16, u16)],
    slot: usize,
) -> Option<usize> {
    semantic_slot_color_for_slot(pcdep_entries, slot, 1)
}

/// Int-bank inverse of [`semantic_slot_for_reg_color`]: given a semantic
/// `locals_cells_stack_w` slot, return the Int-bank color (`registers_i`) that
/// owns it at this PC per the `(bank, color, slot)` map. The Int-bank sibling
/// of [`semantic_slot_color_for_ref_slot`], feeding the deep-kept UNBOXED-INT
/// operand-stack recovery in `walker_capture_snapshot_for_last_guard_impl`: a
/// bank-0 (Int) stack slot names the guard-PC `registers_i[color]` holding the
/// raw int, which the capture then boxes into a `W_IntObject`.
/// `get_list_of_active_boxes` (`pyjitpl.py:206-210`) captures the i-bank via a
/// separate `if length_i:` section
/// (`add_box_to_storage(self.registers_i[index])`). Returns the smallest
/// matching color; `None` when no live Int color owns the slot — which is
/// every current-frontend stack slot, since pyre banks `locals_cells_stack_w`
/// uniformly as Ref (bank-1). See the caller for why the Ref-bank Int-typed
/// hole is NOT recovered by boxing at capture time.
pub(crate) fn semantic_slot_color_for_int_slot(
    pcdep_entries: &[(u8, u16, u16)],
    slot: usize,
) -> Option<usize> {
    semantic_slot_color_for_slot(pcdep_entries, slot, 0)
}

// Sentinel null JitCode for uninitialized PyreSym.
//
// Cannot be `static` because `Arc::new` is not const; use a thread_local
// LazyCell so the initialiser runs once per thread and the resulting
// reference stays valid for the thread's lifetime.
thread_local! {
    static NULL_JITCODE_CELL: std::cell::OnceCell<JitCode> = const { std::cell::OnceCell::new() };
}

fn null_jitcode() -> &'static JitCode {
    NULL_JITCODE_CELL.with(|cell| {
        let r = cell.get_or_init(|| JitCode {
            index: -1,
            payload: std::sync::Arc::new(crate::PyJitCode::skeleton(std::ptr::null())),
        });
        // SAFETY: per-thread `OnceCell` initialises once; the
        // resulting reference lives for the thread's lifetime.
        unsafe { &*(r as *const JitCode) }
    })
}

/// Traced value — RPython `FrontendOp(position, _resint/_resref/_resfloat)` parity.
///
/// Carries both the symbolic IR reference (OpRef) and the concrete
/// execution value (ConcreteValue). Created by opcode handlers that
/// compute concrete results alongside IR recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontendOp {
    pub opref: OpRef,
    pub concrete: ConcreteValue,
}

impl FrontendOp {
    pub fn new(opref: OpRef, concrete: ConcreteValue) -> Self {
        Self { opref, concrete }
    }

    /// `history.py:649-700` `FrontendOp(pos)` parity — the `type='v'`
    /// (void) variant carries only a recorder position with no value
    /// attribute. Pyre folds RPython's class hierarchy
    /// (`IntFrontendOp`/`RefFrontendOp`/`FloatFrontendOp`/bare
    /// `FrontendOp`) into `ConcreteValue` variants, so the void case is
    /// `concrete: Null`.
    pub fn void(opref: OpRef) -> Self {
        Self {
            opref,
            concrete: ConcreteValue::Null,
        }
    }
}

/// Typed concrete value — RPython `FrontendOp._resint/_resref/_resfloat` parity.
///
/// Python bytecode uses untyped locals, so we use a tagged enum instead of
/// RPython's separate `registers_i/r/f` arrays. Each variant corresponds to
/// one of RPython's Box types: `BoxInt`, `BoxPtr`, `BoxFloat`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConcreteValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Ref(PyObjectRef),
    Null,
}

/// Convert a frame slot value to ConcreteValue, preserving null pointers
/// as Ref(PY_NULL) instead of ConcreteValue::Null. Frame slots always
/// contain known values — null means "uninitialized local", not "untracked".
pub(crate) fn concrete_value_from_slot(obj: PyObjectRef) -> ConcreteValue {
    if obj.is_null() {
        return ConcreteValue::Ref(pyre_object::PY_NULL);
    }
    ConcreteValue::from_pyobj(obj)
}

fn concrete_value_from_ir_value(value: majit_ir::Value) -> ConcreteValue {
    match value {
        Value::Ref(gc_ref) => concrete_value_from_slot(gc_ref.0 as PyObjectRef),
        Value::Int(n) => ConcreteValue::Int(n),
        Value::Float(n) => ConcreteValue::Float(n),
        Value::Void => ConcreteValue::Null,
    }
}

impl ConcreteValue {
    /// Convert from PyObjectRef (unbox if possible).
    /// Null pointers become ConcreteValue::Null ("untracked").
    pub fn from_pyobj(obj: PyObjectRef) -> Self {
        if obj.is_null() {
            return ConcreteValue::Null;
        }
        unsafe {
            if is_bool(obj) {
                ConcreteValue::Bool(w_bool_get_value(obj))
            } else if is_trace_plain_int(obj) {
                ConcreteValue::Int(w_int_get_value(obj))
            } else if is_trace_plain_float(obj) {
                ConcreteValue::Float(w_float_get_value(obj))
            } else {
                ConcreteValue::Ref(obj)
            }
        }
    }

    /// Convert to PyObjectRef (box if needed).
    pub fn to_pyobj(self) -> PyObjectRef {
        match self {
            ConcreteValue::Int(v) => w_int_new(v),
            ConcreteValue::Float(v) => pyre_object::w_float_new(v),
            ConcreteValue::Bool(v) => pyre_object::w_bool_from(v),
            ConcreteValue::Ref(obj) => obj,
            ConcreteValue::Null => PY_NULL,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, ConcreteValue::Null)
    }

    /// RPython box.getint() parity.
    pub fn getint(&self) -> Option<i64> {
        match self {
            ConcreteValue::Int(v) => Some(*v),
            ConcreteValue::Bool(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// RPython box.getfloatstorage() parity.
    pub fn getfloat(&self) -> Option<f64> {
        match self {
            ConcreteValue::Float(v) => Some(*v),
            ConcreteValue::Int(v) => Some(*v as f64),
            ConcreteValue::Bool(v) => Some(*v as i64 as f64),
            _ => None,
        }
    }

    /// RPython box.getref_base() parity.
    pub fn getref(&self) -> PyObjectRef {
        self.to_pyobj()
    }

    /// Convert to majit IR Type.
    pub fn ir_type(&self) -> Type {
        match self {
            ConcreteValue::Int(_) | ConcreteValue::Bool(_) => Type::Int,
            ConcreteValue::Float(_) => Type::Float,
            ConcreteValue::Ref(_) => Type::Ref,
            ConcreteValue::Null => Type::Ref,
        }
    }

    /// Convert to majit IR `Value` for a Ref-typed virtualizable slot, or
    /// `None` when there is no valid heap pointer to record.
    ///
    /// `locals_cells_stack_w` is declared as a W_Root array
    /// (virtualizable.py:86-98), so slots mirror RPython `Box(W_Root)` —
    /// `read_boxes` / `write_boxes` always see real boxed W_Root values.
    /// Pyre's lazy boxing means `wrapint` / `wrapfloat` emit a
    /// `NewWithVtable` OpRef without eagerly allocating a `W_IntObject` /
    /// `W_FloatObject`, so there is no heap pointer to flush back through
    /// `synchronize_virtualizable`. Returning `None` lets callers skip the
    /// concrete half of the shadow and update only the OpRef via
    /// `set_virtualizable_box_at`, preserving whatever valid W_Root the
    /// slot previously held instead of corrupting the PyFrame with an
    /// invalid pointer.
    pub fn to_ir_ref_value(&self) -> Option<majit_ir::Value> {
        match self {
            ConcreteValue::Ref(obj) => Some(majit_ir::Value::Ref(majit_ir::GcRef(*obj as usize))),
            ConcreteValue::Null => Some(majit_ir::Value::Ref(majit_ir::GcRef(0))),
            ConcreteValue::Int(_) | ConcreteValue::Float(_) | ConcreteValue::Bool(_) => None,
        }
    }
}

#[inline]
unsafe fn is_trace_plain_int(obj: PyObjectRef) -> bool {
    // A tagged immediate is always an exact `int` (never a subclass), so it
    // is a plain int without the `w_class` deref below. Gated on
    // `CAN_BE_TAGGED` (default false).
    if pyre_object::tagged_int::CAN_BE_TAGGED && pyre_object::tagged_int::is_tagged_int(obj) {
        return true;
    }
    if !unsafe { py_type_check(obj, &INT_TYPE) } {
        return false;
    }
    let int_typeobj = get_instantiate(&INT_TYPE);
    if int_typeobj.is_null() {
        return unsafe { (*obj).w_class.is_null() };
    }
    let w_class = unsafe { (*obj).w_class };
    w_class.is_null() || std::ptr::eq(w_class, int_typeobj)
}

#[inline]
unsafe fn is_trace_plain_float(obj: PyObjectRef) -> bool {
    if !unsafe { py_type_check(obj, &FLOAT_TYPE) } {
        return false;
    }
    let float_typeobj = get_instantiate(&FLOAT_TYPE);
    if float_typeobj.is_null() {
        return unsafe { (*obj).w_class.is_null() };
    }
    let w_class = unsafe { (*obj).w_class };
    w_class.is_null() || std::ptr::eq(w_class, float_typeobj)
}

use crate::descr::{
    PY_OBJECT_ARRAY_GC_TYPE_ID, float_floatval_descr, int_intval_descr, make_array_descr_with_type,
    w_float_size_descr, w_int_size_descr,
};
use crate::frame_layout::{
    PYFRAME_DEBUGDATA_OFFSET, PYFRAME_LASTBLOCK_OFFSET, PYFRAME_LOCALS_CELLS_STACK_OFFSET,
    PYFRAME_PYCODE_OFFSET, PYFRAME_VALUESTACKDEPTH_OFFSET, PYFRAME_W_GLOBALS_OFFSET,
};
use crate::helpers::emit_box_float_inline;

// Re-export liveness items so downstream `pyre_jit_trace::state::*` keeps working.
pub use crate::liveness::{LiveVars, liveness_for};

/// Interpreter state exposed to the JIT framework.
///
/// Built from `PyFrame` before calling `back_edge`, and synced back
/// after compiled code runs.
/// Heap is the single source of truth (RPython parity).
/// next_instr / valuestackdepth live on the PyFrame heap object
/// and are accessed via read_frame_usize / write_frame_usize.
#[derive(majit_macros::VirtualizableState)]
pub struct PyreJitState {
    #[vable(frame)]
    pub frame: usize,
    /// blackhole.py:337 parity: liveness PC from rd_numb (setposition PC).
    /// When set, `restore_guard_failure_values` uses this instead of
    /// next_instr for liveness lookup — matching RPython's pattern where
    /// `blackholeinterp.setposition(jitcode, pc)` is called before
    /// `consume_one_section`.
    pub resume_pc: Option<usize>,
}

/// Meta information for a trace — describes the shape of the code being traced.
#[derive(Clone, majit_macros::VirtualizableMeta)]
pub struct PyreMeta {
    #[vable(num_locals)]
    pub num_locals: usize,
    pub ns_len: usize,
    pub namespace_dependent: bool,
    #[vable(valuestackdepth)]
    pub valuestackdepth: usize,
    /// Full `locals_cells_stack_w` length on the heap object
    /// (`co_nlocals + ncellvars + nfreevars + co_stacksize`).
    /// This stays separate from `valuestackdepth`, which is the live depth
    /// (`pyframe.py:111`) in the RPython model.
    pub array_capacity: usize,
    /// Temporary staging count for extra portal reds that sit between the
    /// frame red and the expanded virtualizable payload. Root portal traces
    /// now carry `ec` here; guard-resume bridge traces still use 0 until the
    /// resumedata path is migrated to the same contract.
    pub trace_extra_reds: usize,
    pub has_virtualizable: bool,
    #[vable(slot_types)]
    pub slot_types: Vec<Type>,
}

/// Symbolic state during tracing.
///
/// `frame` maps to a live IR `OpRef`. Symbolic frame field tracking
/// (locals, stack, valuestackdepth, next_instr) persists across instructions.
/// Locals and stack are virtualized (carried through JUMP args);
/// only next_instr and valuestackdepth are synced before guards / loop close.
#[derive(Clone, majit_macros::VirtualizableSym)]
pub struct PyreSym {
    /// OpRef for the owning PyFrame pointer.
    #[vable(frame)]
    pub frame: OpRef,
    /// PyPy portal second red: `ec` from `interp_jit.py:67`.
    ///
    /// This is not part of the virtualizable payload; it lives between the
    /// frame red and the virtualizable expansion when running under the
    /// canonical PyPy driver descriptor.
    pub(crate) execution_context: OpRef,
    // ── Persistent symbolic frame field tracking ──
    // The Python stack (`locals_cells_stack_w[nlocals..]`)
    // lives in the tail of `registers_r`. The macro's `collect_stack`
    // emits `registers_r[nlocals..nlocals + stack_only_depth]` so JUMP /
    // GUARD args carry locals followed by stack in one contiguous window.
    // This matches RPython's MIFrame register file (`pyjitpl.py:70-78`),
    // which treats locals and the stack as a single abstract register
    // vector.
    #[vable(local_types)]
    pub(crate) symbolic_local_types: Vec<Type>,
    #[vable(stack_types)]
    pub symbolic_stack_types: Vec<Type>,
    pub pending_next_instr: Option<usize>,
    pub(crate) locals_cells_stack_array_ref: OpRef,
    #[vable(valuestackdepth)]
    pub(crate) valuestackdepth: usize,
    #[vable(nlocals)]
    pub(crate) nlocals: usize,
    /// Bridge-specific override for the locals slice of registers_r.
    /// resume.py:1042 parity: when set, init_symbolic uses these OpRefs
    /// (mapped from RebuiltValue::Box(n) in rebuild_from_resumedata) instead
    /// of the vable_array_base-based layout. This ensures bridge traces see
    /// frame locals as symbolic InputArgs, not concrete values.
    pub(crate) bridge_local_oprefs: Option<Vec<OpRef>>,
    /// Bridge-specific override for the kept operand-stack slice of
    /// registers_r ([nlocals..nlocals+stack_only], semantic-slot == color
    /// in that prefix). resume.py:1042 parity: setup_bridge_sym resolves
    /// the live operand-stack temps from the guard's resume data; this
    /// preserves them so init_symbolic (which runs AFTER setup_bridge_sym
    /// in pyre's bridge launcher) does not clobber the rebuilt stack tail
    /// back to NONE, and so the full-body-walk argbox seed can recover the
    /// kept conditional-expression / short-circuit value (#124).
    pub(crate) bridge_stack_oprefs: Option<Vec<OpRef>>,
    /// Kept-stack branch-guard resume coordinate for the full-body walk: the
    /// guard's OWN jitcode byte offset (`frame0.jitcode_pc`, the mid-opcode
    /// `goto_if_not`), resolved the same way the blackhole resolves its
    /// `setposition` (`resolve_resume_pc_with_jitcode_pc`).  A chained-compare /
    /// short-circuit guard resumes mid-opcode: the opcode-entry marker
    /// (`pc_map[py_pc]`) re-executes the whole comparison from the top, reading
    /// abstract-register colors that were live at entry but dead (recolored /
    /// consumed) at the guard — colors the guard snapshot never preserved.  The
    /// walk must resume where the blackhole does (the guard offset), so the
    /// re-executed suffix only reads colors the resume data actually carries.
    /// `None` for a non-branch-guard resume, where the walk
    /// keeps the `pc_map[py_pc]` opcode-entry offset.
    pub(crate) bridge_walk_entry_pc: Option<usize>,
    /// The color-indexed Ref register bank as `consume_boxes`
    /// (resume.py:1055) fills `f.registers_r` — one box per abstract
    /// register color the guard's resume numbering named. This is the
    /// authoritative `_get_list_of_active_boxes` (pyjitpl.py:216-233)
    /// source: it reads `registers_r[color]` by color at snapshot time.
    ///
    /// `sym.registers_r` diverges from this on the Ref bank: for the
    /// tracer's own LOAD_FAST/STORE_FAST reads pyre needs a SEMANTIC
    /// slot-indexed mirror (`[locals.., stack_tail..]`) because the
    /// per-CodeObject regalloc colors a slot at a color other than its
    /// slot index. `setup_bridge_sym` therefore overwrites
    /// `sym.registers_r` with that mirror and `init_symbolic` rebuilds it
    /// again from `bridge_local_oprefs`/`bridge_stack_oprefs`, so the
    /// color-indexed decode is lost from `sym.registers_r`. The int/float
    /// banks keep their color decode (init_symbolic rebuilds only the Ref
    /// bank), so only Ref needs this side field. A cross-frame bridge
    /// resume snapshot (`compute_bridge_root_parent_frame`) reads this to
    /// recover an operand live across a resumed call (e.g. `t1` in
    /// `return fib(n-1)+fib(n-2)`) that the semantic mirror does not carry
    /// by color.
    pub(crate) bridge_registers_r: Option<Vec<OpRef>>,
    /// Bridge-specific override for symbolic_local_types.
    /// virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
    /// is a W_Root array → all items are Type::Ref. setup_bridge_sym
    /// populates this with all-Ref; downstream unboxing happens in
    /// opcode handlers via guard_class + getfield_gc_pure_i/_f, not at
    /// the virtualizable slot level.
    pub(crate) bridge_local_types: Option<Vec<Type>>,
    // virtualizable.py:86-93: ALL static fields in declared order.
    // RPython's unroll_static_fields includes every field from
    // _virtualizable_; ALL must be inputarg (not info_only). The `type`
    // tag below mirrors pyframe.py's static-field declared types so the
    // macro mints `OpRef::input_arg_int/ref` matching the RPython
    // `InputArgInt`/`InputArgRef` class for that slot
    // (resoperation.py:719/739).
    #[vable(inputarg, type = int)]
    pub(crate) vable_last_instr: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_pycode: OpRef,
    #[vable(inputarg, type = int)]
    pub(crate) vable_valuestackdepth: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_debugdata: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_lastblock: OpRef,
    #[vable(inputarg, type = ref)]
    pub(crate) vable_w_globals: OpRef,
    #[vable(array_base)]
    pub(crate) vable_array_base: Option<u32>,
    /// True when this frame's `locals_cells_stack_w` array IS the active
    /// virtualizable shadow for the current trace — i.e. when reads /
    /// writes against this frame's locals must consult / mutate
    /// `TraceCtx::virtualizable_boxes` (RPython `metainterp
    /// .virtualizable_boxes`, pyjitpl.py:1230) instead of going through
    /// regular MIFrame `registers_X`.
    ///
    /// Invariant: `is_active_vable_owner == vable_array_base.is_some()`.
    /// The boolean predicate (this field) and the u32 OpRef-offset (the
    /// `Option<u32>` value of `vable_array_base`) are split because their
    /// semantic roles differ: the predicate decides "consult vable
    /// shadow vs. registers", while the offset is only used as a
    /// fallback OpRef synthesizer at `trace_opcode.rs:1248` when the
    /// metainterp-scope shadow is not yet populated. RPython has neither
    /// per-frame state — the codewriter emits `getarrayitem_vable`
    /// opcodes only on the toplevel frame's bytecode. Pyre dispatches
    /// the same `load_local_value` for all frames, so we encode the
    /// "active vable owner" identity per-frame.
    ///
    /// Helpers `become_active_vable_owner` / `clear_active_vable`
    /// maintain the invariant; do not write `vable_array_base` directly
    /// outside of those (or the macro-generated `init_vable_indices`
    /// invoked through `become_active_vable_owner`).
    pub(crate) is_active_vable_owner: bool,
    // ── MIFrame concrete Box tracking (RPython registers_i/r/f parity) ──
    // Concrete Python object values for locals and stack, tracked in
    // parallel with `registers_r`. Each opcode handler updates these
    // alongside the symbolic OpRefs so that guard decisions, branch
    // directions, and call results use internally tracked values
    // instead of reading from an external PyFrame snapshot.
    pub(crate) concrete_locals: Vec<ConcreteValue>,
    pub concrete_stack: Vec<ConcreteValue>,
    /// pyjitpl.py:74: frame.jitcode — JitCode reference.
    /// Provides both .code (CodeObject*) and .index (snapshot encoding).
    pub(crate) jitcode: *const JitCode,
    /// Namespace for global lookups (W_DictObject / W_ModuleDictObject).
    pub(crate) concrete_namespace: PyObjectRef,
    /// Execution context pointer (for creating callee frames).
    pub(crate) concrete_execution_context: *const pyre_interpreter::PyExecutionContext,
    /// Virtualizable object pointer (PyFrame).
    /// RPython MetaInterp stores the virtualizable separately from MIFrame.
    pub(crate) concrete_vable_ptr: *mut u8,
    /// Live (interpreter-owned) virtualizable `PyFrame` behind the tracing
    /// snapshot, or 0 when tracing runs without one (tests).
    /// `concrete_vable_ptr` points at the `snapshot_for_tracing` copy whose
    /// `debugdata` / `lastblock` are owned clones freed when the snapshot
    /// drops; vable-statics capture (`flush_to_frame`) reads those
    /// pointer-valued fields from this frame so the trace's resume data
    /// never carries snapshot-owned pointers.  RPython has no snapshot —
    /// `read_boxes` (virtualizable.py:86-93) always reads the live
    /// virtualizable, which is what this field restores.
    pub(crate) live_vable_frame_addr: usize,
    /// Function-entry traces use typed locals (RPython MIFrame parity).
    pub(crate) is_function_entry_trace: bool,
    /// RPython MetaInterp.last_exc_value (pyjitpl.py:2745): concrete
    /// exception object pending during tracing. Set by execute_ll_raised
    /// (raise_varargs), consumed by handle_possible_exception.
    pub(crate) last_exc_value: pyre_object::PyObjectRef,
    /// RPython MetaInterp.class_of_last_exc_is_const (pyjitpl.py:2754):
    /// True after GUARD_EXCEPTION or GUARD_CLASS on the exception.
    pub(crate) class_of_last_exc_is_const: bool,
    /// RPython MetaInterp.last_exc_box (pyjitpl.py:1696, 3386): symbolic
    /// OpRef for the exception value. Set directly by `opimpl_raise`, or
    /// by handle_possible_exception after GUARD_EXCEPTION, then consumed
    /// by finishframe_exception for stack push.
    pub(crate) last_exc_box: OpRef,
    /// E1: maps the OpRef of a trace-built (fresh `NewWithVtable`)
    /// exception to its trace-time concrete instance.  `RAISE_VARARGS`
    /// reuses the instance to take the instance fast path — skip the
    /// residual `normalize_raise_varargs_jit` publish + `GUARD_EXCEPTION`
    /// round-trip so the exception stays virtualizable — and to apply the
    /// unconditional `__context__ = ec.sys_exc_value` chaining that is
    /// valid only for a freshly constructed exception (w_context still
    /// null, self-cycle impossible).
    ///
    /// The entry is dropped as soon as freshness can no longer be proven:
    /// `RAISE_VARARGS` consumes it (a re-raise of the same object must
    /// take the residual path — its `w_context` is set by then), and
    /// `box_value_for_python_helper` removes any value escaping into a
    /// python-helper residual call (which may mutate the exception).
    pub(crate) trace_built_exc: indexmap::IndexMap<OpRef, pyre_object::PyObjectRef>,
    /// Symbolic mirror of executioncontext.current_exception/sys_exc_info.
    /// Used by PUSH_EXC_INFO / POP_EXCEPT to preserve nested handler state.
    pub(crate) current_exc_value: pyre_object::PyObjectRef,
    pub(crate) current_exc_box: OpRef,
    /// pyjitpl.py:2597 virtualref_boxes: pairs of (jit_virtual, real_vref).
    /// Each pair: (symbolic OpRef, concrete pointer).
    /// resume.py:1093 restores virtual references on guard failure.
    /// Pairs stored flat: [virt_sym, virt_ptr, real_sym, real_ptr, ...].
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,
    // ── RPython MIFrame.registers_{i,r,f} port (pyjitpl.py:74-90) ──
    //
    // RPython reference (target shape):
    //   self.registers_i = [history.CONST_NULL] * jitcode.num_regs_and_consts_i()
    //   self.registers_r = [history.CONST_NULL] * jitcode.num_regs_and_consts_r()
    //   self.registers_f = [history.CONST_NULL] * jitcode.num_regs_and_consts_f()
    //
    // Each bank is sized to `num_regs_X + len(constants_X)` and indexed
    // by post-regalloc-color: `[0, num_regs_X)` are register slots
    // initialised to `CONST_NULL`, `[num_regs_X, ...)` are the constant
    // pool entries copied from `jitcode.constants_X`.
    //
    // SSA-authoritative live_r layout:
    //   - `setup_kind_register_banks` sizes all three banks to
    //     `num_regs_and_consts_X` when the owning JitCode is bound.
    //   - `registers_i` / `registers_f` are indexed by post-regalloc
    //     color. The encoder (`get_list_of_active_boxes`) reads them
    //     directly via the bank clone.
    //   - `registers_r` remains pyre's semantic frame mirror for
    //     `locals_cells_stack_w` because stack colors may coalesce with
    //     local colors. The encoder materializes a temporary
    //     color-indexed Ref-bank snapshot before reading liveness.
    pub(crate) registers_i: Vec<OpRef>,
    #[vable(locals)]
    pub(crate) registers_r: Vec<OpRef>,
    pub(crate) registers_f: Vec<OpRef>,
}

#[doc(hidden)]
pub struct TestSymState {
    pub frame: OpRef,
    pub jitcode: *const (),
    pub nlocals: usize,
    pub valuestackdepth: usize,
    pub locals_cells_stack_array_ref: OpRef,
    pub symbolic_local_types: Vec<Type>,
    pub symbolic_stack_types: Vec<Type>,
    pub registers_r: Vec<OpRef>,
    pub concrete_stack: Vec<ConcreteValue>,
    pub concrete_namespace: PyObjectRef,
    pub vable_last_instr: OpRef,
    pub vable_pycode: OpRef,
    pub vable_valuestackdepth: OpRef,
    pub vable_debugdata: OpRef,
    pub vable_lastblock: OpRef,
    pub vable_w_globals: OpRef,
}

/// Trace-time view over the virtualizable `PyFrame`.
///
/// Per-instruction wrapper that borrows persistent symbolic state from
/// `PyreSym` via raw pointer. The symbolic tracking (locals, stack,
/// valuestackdepth, next_instr) lives in PyreSym and survives across
/// instructions; this struct provides the per-instruction context
/// (ctx, fallthrough_pc).
pub struct MIFrame {
    pub(crate) ctx: *mut TraceCtx,
    pub(crate) sym: *mut PyreSym,
    pub(crate) fallthrough_pc: usize,
    /// Concrete PyFrame address for exception table lookup.
    pub(crate) concrete_frame_addr: usize,
    /// RPython pyjitpl.py orgpc parity: the PC at the START of the current
    /// opcode. All guards within one opcode capture this as their resume PC
    /// so that guard failure re-executes the opcode from the beginning.
    pub(crate) orgpc: usize,
    /// RPython `capture_resumedata(resumepc=orgpc)`
    /// Opcode-start snapshot of the unified `registers_r` file used by
    /// guard/resumedata capture for this one opcode. When `None`, guard
    /// capture reads the live register file directly. Shadow-owner snapshots
    /// are semantic frame prefixes sourced from `virtualizable_boxes`;
    /// non-owner snapshots clone the semantic `registers_r` mirror.
    pub(crate) pre_opcode_registers_r: Option<Vec<OpRef>>,
    /// Semantic frame prefix length at opcode start. This is distinct from
    /// `pre_opcode_registers_r.len()` for non-owner traces because the Ref
    /// bank may include post-regalloc color slots above the semantic
    /// locals+stack prefix.
    pub(crate) pre_opcode_semantic_depth: Option<usize>,
    /// PyPy capture_resumedata: parent frame chain for multi-frame guards.
    /// Each entry points at one parent frame plus the resumepc that
    /// should be used when that parent is snapshotted. This stays much
    /// closer to RPython's `self.framestack` than the old flattened
    /// `(fail_args, fail_arg_types, resumepc, jitcode_index)` tuples.
    pub parent_frames: Vec<ResumeFrameState>,
    /// `pyjitpl.py:181-193` `_result_argcode` analogue for non-top-frame
    /// snapshotting. When present, `get_list_of_active_boxes(in_a_call=True)`
    /// overwrites this caller stack slot with a zero/null placeholder before
    /// liveness encoding.
    pub pending_result_stack_idx: Option<usize>,
    pub pending_result_type: Option<Type>,
    pub pending_inline_frame: Option<PendingInlineFrame>,
    /// For an `in_a_call` parent frame snapshotted by
    /// `get_list_of_active_boxes`, the Python pc of the CALL whose
    /// post-residual-call `-live-`/`catch_exception` this frame must read
    /// liveness at (so encode and the marker-routed blackhole resume share
    /// the one `-live-` the way `pyjitpl.py:194-195 pc=self.pc` does).
    /// `None` keeps the legacy `fallthrough_pc` liveness.
    pub residual_call_pc: Option<usize>,
    /// Resume-marker twin for the loop-close guards' snapshot word.
    /// `close_loop_args_at` sets and clears it around the `GuardEvalBreaker`
    /// and `GuardFutureCondition` emits; it is `None` outside that window.
    pub(crate) loop_close_marker_jit_pc: Option<usize>,
}

pub(crate) fn instruction_consumes_comparison_truth(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::PopJumpIfFalse { .. } | Instruction::PopJumpIfTrue { .. }
    )
}

pub(crate) fn instruction_is_trivia_between_compare_and_branch(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken
            | Instruction::ToBool
    )
}

pub(crate) fn instruction_needs_pre_opcode_snapshot(instruction: Instruction) -> bool {
    // Only keep the opcode-start snapshot for bytecodes that can emit a
    // guard after mutating the logical stack/register state. A larger
    // "may raise" set still needs GUARD_{NO_,}EXCEPTION handling, but
    // opcodes like GET_ITER / FOR_ITER / GET_LEN / IMPORT_FROM inspect
    // the current stack via peek-at-TOS and do not need a pre-pop
    // resumestate. Unsupported tracer opcodes also stay out of this set
    // until they gain a real guard-producing lowering.
    matches!(
        instruction,
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::StoreSubscr
            | Instruction::BinaryOp { .. }
            | Instruction::CompareOp { .. }
            | Instruction::UnaryNegative
            | Instruction::UnaryNot
            | Instruction::UnaryInvert
            | Instruction::RaiseVarargs { .. }
            // Both pop the operand stack (BUILD_TUPLE pop_n, UNPACK_SEQUENCE
            // pop_value) before emitting a guard: trace_build_tuple_value emits
            // the specialised-tuple w_class guards on the popped items, and
            // unpack_sequence_value emits the sequence class / length guards on
            // the popped sequence. Without the opcode-start snapshot the guard's
            // resume state reflects the post-pop stack, so resuming at the
            // opcode start restores the consumed operands as null / mismatched.
            | Instruction::BuildTuple { .. }
            | Instruction::UnpackSequence { .. }
            // LOAD_ATTR reaches the trait leg (execute_opcode_step) in two
            // cases the dispatch gate carves out of the arm walk: the foldable
            // builtin list-method form (append/pop/reverse on a list receiver)
            // and any method-load inside an inline frame. Both run load_method,
            // which pop_value's the receiver first, then emits a non-residual
            // receiver class guard (guard_class) plus a version_tag guard_value
            // — resume_pc=orgpc, so the consumed receiver must be in the
            // opcode-start snapshot. The arm-walk leg (the common non-foldable
            // form) ignores pre_opcode_registers_r, so capturing it there is
            // inert; only the trait leg consults it.
            | Instruction::LoadAttr { .. }
            // LIST_APPEND reaches the trait leg in an inline frame (it is
            // walker-routed at the root). The `list_append` hook pop_value's
            // the appended value, then records the generic `jit_list_append`
            // residual over the peeked list and popped value at
            // resume_pc=orgpc, so both consumed operands must be in the
            // opcode-start snapshot. The sibling comprehension helpers
            // (SET_ADD / MAP_ADD / LIST_EXTEND / …) have no tracer override
            // and abort before popping, so only LIST_APPEND qualifies.
            | Instruction::ListAppend { .. }
    )
}

/// RPython exc=True parity: instructions that correspond to JitCode ops
/// with exc=True. Only external calls and operations that invoke arbitrary
/// Python code need GUARD_NO_EXCEPTION. Arithmetic, comparisons, and
/// local variable access are lowered to primitive IR ops (exc=False) in
/// RPython and protected by type-specific guards instead.
pub(crate) fn instruction_may_raise(instruction: Instruction) -> bool {
    matches!(
        instruction,
        // RPython exc=True: external calls and attribute access that
        // may invoke arbitrary Python code (__getattr__, descriptors).
        // Unsupported tracer opcodes stay out of this set until they gain a
        // real lowering; otherwise the default "not implemented" error is
        // mis-recorded as a traced GUARD_EXCEPTION.
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::StoreAttr { .. }
            | Instruction::StoreSubscr
            | Instruction::ImportFrom { .. } // RPython raise/reraise are dedicated opimpls, not generic
                                             // exc=True execute_varargs sites. They unwind directly instead
                                             // of going through handle_possible_exception().
    )
}

/// Environment context — currently unused.
pub struct PyreEnv;

/// Descriptor for raw `PyObjectRef` item pointers that already address
/// `items[0]` (no length prefix to skip). Used after an explicit
/// `IntAdd(block, ITEMS_BLOCK_ITEMS_OFFSET)` converts a
/// `*mut ItemsBlock` block-base pointer into the items-base pointer —
/// see `load_namespace_value` in `trace_opcode.rs` for the canonical
/// pattern. `GETARRAYITEM_GC_R(ptr, i)` lands on `ptr + i * item_size`.
///
/// For callers that hold the block-base pointer directly (i.e.
/// `*mut ItemsBlock` for `W_ListObject.items` / tuple backing storage,
/// or `*mut FixedObjectArray` for `PyFrame.locals_cells_stack_w`), use
/// [`pyobject_gcarray_descr`] instead — its
/// `base_size = FIXED_ARRAY_ITEMS_OFFSET` makes the descriptor itself
/// skip the length prefix.
pub(crate) fn pyobject_array_descr() -> DescrRef {
    // `nolength=True` shape (descr.py:359-360): items start at offset 0,
    // no length header — `GETARRAYITEM_GC_R(ptr, i)` lands on
    // `ptr + i * item_size`.  Stable identity carrier
    // `"pyre::pyobject_array_nolength"` so every `pyobject_array_descr()`
    // call returns the same Arc per PyPy `cpu.arraydescrof(ARRAY)`
    // singleton — `gc_cache._cache_array[LLType::Array(path_hash(...))]`
    // canonicalizes across analyzer / runtime consumers.
    crate::descr::make_array_descr_with_full_id(
        0,
        std::mem::size_of::<usize>(),
        0,
        None,
        Type::Ref,
        false,
        Some("pyre::pyobject_array_nolength".to_string()),
    )
}

/// Descriptor for RPython `Ptr(GcArray(PyObjectRef))` containers —
/// `[len][items...]` layout where the pointer addresses the length
/// header. Used by virtual array materialization (`NewArray` +
/// `SetarrayitemGc` in `decode_virtual_info`, resume.py:653-670) and
/// by the virtualizable-frame array field (`locals_cells_stack_w` via
/// the autogenerated `frame_locals_cells_stack_descr()` in
/// `virtualizable_gen.rs`). `base_size = FIXED_ARRAY_ITEMS_OFFSET`
/// skips the length prefix so `GETARRAYITEM_GC_R(array_ptr, i)` lands
/// on items[i] directly.
pub fn pyobject_gcarray_descr() -> DescrRef {
    // type_id = PY_OBJECT_ARRAY_GC_TYPE_ID so the GC tracer walks each
    // item slot as a Ref. Without this, gen_initialize_tid stamps
    // OBJECT_GC_TYPE_ID into the GC header, the tracer treats the
    // allocation as a 16-byte PyObject, and the variable-part items
    // (the inline-emitted PyFrame.locals_cells_stack_w) survive the
    // first young collection without forwarding — manifesting as
    // wrong fib values on dynasm and SIGSEGV on cranelift once the
    // recursion depth fills the nursery (rgc parity:
    // gctypelayout.py:266-291 T_IS_VARSIZE / T_IS_GCARRAY_OF_GCPTR).
    // Length-prefixed shape (descr.py:362): the array_ptr addresses the
    // length header at offset 0, items start at `FIXED_ARRAY_ITEMS_OFFSET`.
    make_array_descr_with_type(
        pyre_object::FIXED_ARRAY_ITEMS_OFFSET,
        std::mem::size_of::<usize>(),
        PY_OBJECT_ARRAY_GC_TYPE_ID,
        Some(0),
        Type::Ref,
        false,
    )
}

/// `Ptr(GcArray(Signed))` — the `IntegerListStrategy` backing block
/// (`erase([int])`). Length-prefixed `[capacity][i64...]`: `base_size` skips
/// the capacity header so `GetarrayitemGcI(block, i)` lands on items[i], and
/// the op routes through the `(array, descr, index)` heap cache (CSE), unlike
/// the raw `int_array_descr`.
pub(crate) fn int_gcarray_descr() -> DescrRef {
    crate::descr::make_array_descr_with_type(
        pyre_object::TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
        8,
        pyre_object::GC_INT_ARRAY_GC_TYPE_ID,
        Some(0),
        Type::Int,
        true,
    )
}

/// `Ptr(GcArray(Float))` — the `FloatListStrategy` backing block
/// (`erase([float])`). See [`int_gcarray_descr`].
pub(crate) fn float_gcarray_descr() -> DescrRef {
    crate::descr::make_array_descr_with_type(
        pyre_object::TYPED_ITEMS_BLOCK_ITEMS_OFFSET,
        8,
        pyre_object::GC_FLOAT_ARRAY_GC_TYPE_ID,
        Some(0),
        Type::Float,
        false,
    )
}

/// `descr.py SizeDescr` for the host `PyFrame` virtualizable struct.
///
/// All `PyFrame` field descriptors point at this SizeDescr via
/// `FieldDescr.parent_descr` so the optimizer's `ensure_ptr_info_arg0`
/// (`optimizer.py:478-484`) can dispatch the GETFIELD/SETFIELD branch
/// to `InstancePtrInfo` / `StructPtrInfo`. Also handed to
/// `VirtualizableInfo::set_parent_descr` so virtualizable field
/// descriptors share the same parent.
pub fn pyframe_size_descr() -> DescrRef {
    crate::descr::pyframe_size_descr()
}

pub(crate) fn frame_locals_cells_stack_descr() -> DescrRef {
    crate::descr::pyframe_locals_cells_stack_descr()
}

// R3.3: frame_dict_storage_descr retired — frame_get_namespace now
// reads through w_globals → dict_storage_proxy.

pub(crate) fn wrapint(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    let boxed =
        crate::helpers::emit_box_int_inline(ctx, value, w_int_size_descr(), int_intval_descr());
    // A JIT-made int box is provably a heap `W_IntObject`; record its class so a
    // later unbox of a loop-carried box skips the tag block AND the redundant
    // `GuardClass`, taking the `GetfieldGc` path that folds through this box's
    // `SetfieldGc` at loop-close. Without this the unbox falls into the
    // `CastPtrToInt` tag-arith leg (a JIT box is not tag-known), which does NOT
    // fold through `NewWithVtable`+`SetfieldGc` and leaves a per-iteration
    // rebox+`GuardTrue(lowbit)` in the steady loop that fails every back-edge.
    // Gated on `CAN_BE_TAGGED` so flag-false unbox emission is byte-identical.
    if pyre_object::tagged_int::CAN_BE_TAGGED {
        let int_type = &pyre_object::pyobject::INT_TYPE as *const _ as i64;
        ctx.heap_cache_mut().class_now_known(boxed, int_type);
    }
    boxed
}

/// pyjitpl.py:3514 find_biggest_function
pub(crate) fn biggest_inline_trace_key(state: &mut MIFrame) -> Option<u64> {
    state.with_ctx(|_, ctx| ctx.find_biggest_function())
}

pub(crate) fn note_root_trace_too_long(green_key: u64) {
    let (driver, _) = crate::driver::driver_pair();
    let warm_state = driver.meta_interp_mut().warm_state_mut();
    warm_state.trace_next_iteration(green_key);
    warm_state.mark_force_finish_tracing(green_key);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][trace-too-long] trace_next_iteration + mark_force_finish_tracing key={}",
            green_key
        );
    }
}

pub(crate) fn wrapfloat(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_box_float_inline(ctx, value, w_float_size_descr(), float_floatval_descr())
}

pub(crate) fn ensure_boxed_for_ca(ctx: &mut TraceCtx, state: &MIFrame, value: OpRef) -> OpRef {
    match state.value_type(value) {
        Type::Int => wrapint(ctx, value),
        Type::Float => wrapfloat(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

pub(crate) fn box_value_for_python_helper(
    state: &mut MIFrame,
    ctx: &mut TraceCtx,
    value: OpRef,
) -> OpRef {
    match state.value_type(value) {
        Type::Int => wrapint(ctx, value),
        Type::Float => wrapfloat(ctx, value),
        Type::Ref | Type::Void => {
            // A trace-built exception escaping into a python-helper
            // residual is no longer provably fresh (the callee may set
            // `__context__` or other state); drop it so a later
            // RAISE_VARARGS takes the residual path, whose runtime
            // `attach_raise_cause` chaining is conditional.
            state.sym_mut().trace_built_exc.swap_remove(&value);
            value
        }
    }
}

pub(crate) fn box_args_for_python_helper(
    state: &mut MIFrame,
    ctx: &mut TraceCtx,
    args: &[OpRef],
) -> Vec<OpRef> {
    args.iter()
        .map(|&arg| box_value_for_python_helper(state, ctx, arg))
        .collect()
}

// RPython parity note: pyjitpl.py (tracer) records GETFIELD_GC ops WITHOUT
// any constant folding. Folding happens exclusively in the optimizer's
// `optimize_GETFIELD_GC_I` (heap.py:639-646), which delegates to
// `optimizer.constant_fold(op)` → `_execute_arglist` → `do_getfield_gc_*`.
// pyre's `OptContext::constant_fold` in optimizeopt/mod.rs is the exact
// port of that path — it handles Int/Float/Ref via `execute_nonspec_const`
// dispatched on `field_type()` and `field_size()`.
//
// The previous tracer-level `try_trace_const_pure_int_field` helper was a
// pyre-specific pre-optimization that duplicated (and mistyped) the
// optimizer logic. It has been removed for structural parity with RPython.

pub(crate) fn try_trace_const_boxed_int(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        if is_bool(concrete_value) {
            return Some(ctx.const_int(if w_bool_get_value(concrete_value) {
                1
            } else {
                0
            }));
        }
        if is_trace_plain_int(concrete_value) {
            return Some(ctx.const_int(w_int_get_value(concrete_value)));
        }
    }
    None
}

/// pyjitpl.py:750-758: read container length.
///
/// RPython's `arraylen_gc` reads the GC array header — there is exactly one
/// length per array, so RPython keeps a per-box `heapc_deps[0]` slot. pyre
/// stores list/bytes/tuple lengths as plain struct fields, so the cached
/// value lives in `HeapCache.heap_cache[descr] -> CacheEntry`
/// (heapcache.py:172).  `opimpl_getfield_gc_i` already does that lookup,
/// so this helper is now just a thin alias kept for source-stability
/// with the call sites.
pub(crate) fn trace_arraylen_gc(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    opimpl_getfield_gc_i(ctx, obj, descr)
}

/// pyjitpl.py:744-748 `opimpl_arraylen_gc`. Emits the actual
/// `ArraylenGc` op against the GcArray header (rlist.py:251
/// `len(l.items)` reads the array's length-prefix). Caller is the
/// items_block Ref (output of `opimpl_getfield_gc_r` on
/// `tuple_wrappeditems_descr` / `list_items_descr`); `descr` is the
/// matching array descr (e.g. `pyobject_gcarray_descr`).
///
/// Distinct from `trace_arraylen_gc` above — that helper reads
/// pyre-specific length FIELDS off the host wrapper struct via
/// getfield (TODO: bring to parity) and is reserved for callers
/// that still go through that path (`str_len`, `dict_len`,
/// `list_length`, typed list `int_items.len`).
pub(crate) fn opimpl_arraylen_gc(ctx: &mut TraceCtx, array: OpRef, descr: DescrRef) -> OpRef {
    if let Some(cached) = ctx.heap_cache().arraylen(array) {
        // pyjitpl.py:889-893 `_opimpl_arraylen_gc` cache hit:
        //     if length is not None:
        //         self.metainterp.staticdata.profiler.count_ops(rop.ARRAYLEN_GC, Counters.HEAPCACHED_OPS)
        //         return lengthbox
        ctx.profiler().count_ops(
            OpCode::ArraylenGc,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::ArraylenGc, &[array], descr.clone());
    // Box(value) parity: executor.py:188 do_arraylen_gc returns
    // BoxInt(cpu.bh_arraylen_gc(array, arraydescr)). Stamp the result
    // OpRef with the live length so downstream box_value(result) sees
    // the same value (matches RPython BoxInt(length) carrier).
    if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(array) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            if let Some(live) = ctx.arraylen_sanity_load(struct_ptr, &descr) {
                ctx.set_opref_concrete(result, live);
            }
        }
    }
    ctx.heap_cache_mut().arraylen_now_known(array, result);
    result
}

pub(crate) fn opimpl_getfield_gc_i(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    // pyjitpl.py:opimpl_getfield_gc_i parity: the tracer does NOT fold
    // pure field reads on constant objects. Folding happens in the
    // optimizer (heap.py:optimize_GETFIELD_GC_I → optimizer.constant_fold),
    // which pyre ports in OptContext::execute_nonspec_const with correct
    // type dispatch (Int/Float/Ref). The tracer only records the GC op.
    //
    // heapcache.py: check if this field was already read/written in this trace
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (int arm). The
        // line-by-line port runs `executor.execute(cpu, mi, opnum,
        // fielddescr, box)` and asserts `resvalue ==
        // upd.currfieldbox.getint()`.  The cached Box's intrinsic
        // value is fetched via `box_value(cached)` — covering const
        // pool, standard-virtualizable shadow, and the frontend
        // object's `value` field (RPython `currfieldbox.getint()`
        // dispatch parity).
        let expected_int = match ctx.box_value(cached) {
            Some(majit_ir::Value::Int(n)) => Some(n),
            _ => None,
        };
        if let Some(cached_int) = expected_int {
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(majit_ir::Value::Int(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
                    {
                        if loaded != cached_int && crate::probe_subscr_enabled() {
                            eprintln!(
                                "[PYRE_PROBE_SUBSCR] sanity-int-mismatch obj={:?} field_index={} struct_ptr={:#x} descr_pure={} cached={} loaded={}",
                                obj,
                                field_index,
                                struct_ptr,
                                descr.is_always_pure(),
                                cached_int,
                                loaded
                            );
                        }
                        assert_eq!(
                            loaded, cached_int,
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (int): loaded {loaded} != cached \
                             {cached_int} (field_index={field_index}, \
                             struct_ptr={struct_ptr:#x})"
                        );
                    }
                }
            }
        }
        // pyjitpl.py:946 cache-hit accounting:
        //   self.metainterp.staticdata.profiler.count_ops(rop.GETFIELD_GC_I, Counters.HEAPCACHED_OPS)
        //   return upd.currfieldbox
        ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    // pyjitpl.py:1074-1089: quasi-immutable field handling.
    // Record the field as quasi-immut known so subsequent reads skip
    // the QUASIIMMUT_FIELD op. Emit GUARD_NOT_INVALIDATED if needed.
    // NOTE: GuardNotInvalidated is NOT emitted here — it requires
    // PyreSym.generate_guard for proper snapshot/fail_args (pyjitpl.py:1087
    // generate_guard parity). Instead, set a flag on ctx so the caller
    // (PyreSym with_ctx block) can emit it with full resume data.
    if descr.is_quasi_immutable() {
        if ctx.heap_cache().is_quasi_immut_known(obj, field_index) {
            // pyjitpl.py:1077-1080 cache hit:
            //   if heapcache.is_quasi_immut_known(fielddescr, box):
            //       profiler.count_ops(rop.QUASIIMMUT_FIELD, HEAPCACHED_OPS)
            //       return
            ctx.profiler().count_ops(
                OpCode::QuasiimmutField,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
        } else {
            ctx.heap_cache_mut().quasi_immut_now_known(obj, field_index);
            ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
            if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
                ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
            }
        }
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureI
    } else {
        OpCode::GetfieldGcI
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr.clone());
    // pyjitpl.py:948-949 `resbox = execute_with_descr(...); upd.getfield_now_known(resbox)`.
    // `resbox` carries the loaded value; pair the recorded opref with
    // the live int from `field_sanity_load` so subsequent
    // `box_value(result)` mirrors RPython's executor-returned Box.
    let live_value = if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Int)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(live_value) = live_value {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

/// pyjitpl.py:874-882 `opimpl_getfield_gc_r`. Same shape as `_i`
/// modulo the rop variant — folding lives in the optimizer
/// (`optimize_GETFIELD_GC_R = optimize_GETFIELD_GC_I` per RPython's
/// alias), so the tracer only records the GC op.
pub(crate) fn opimpl_getfield_gc_r(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    let field_index = descr.index();
    if let Some(cached) = ctx.heapcache_getfield_cached(obj, field_index) {
        // pyjitpl.py:934-945 cache-hit sanity check (ref arm).
        // `box_value(cached)` resolves the upstream
        // `currfieldbox.getref_base()` payload through the full chain
        // (const pool, standard-virtualizable shadow, the frontend
        // object's `value` field).
        let expected_ref = match ctx.box_value(cached) {
            Some(majit_ir::Value::Ref(r)) => Some(r),
            _ => None,
        };
        if let Some(cached_ref) = expected_ref {
            if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
                let struct_ptr = struct_ref.0 as i64;
                if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
                    if let Some(majit_ir::Value::Ref(loaded)) =
                        ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref)
                    {
                        assert_eq!(
                            loaded, cached_ref,
                            "_opimpl_getfield_gc_any_pureornot sanity \
                             check (ref): loaded {:#x} != cached {:#x} \
                             (field_index={field_index}, struct_ptr=\
                             {struct_ptr:#x})",
                            loaded.0, cached_ref.0,
                        );
                    }
                }
            }
        }
        // pyjitpl.py:946 — RPython hardcodes `GETFIELD_GC_I` regardless
        // of the rop variant (`_i` / `_r` / `_f`); pyre matches.
        ctx.profiler().count_ops(
            OpCode::GetfieldGcI,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return cached;
    }
    if descr.is_quasi_immutable() {
        if ctx.heap_cache().is_quasi_immut_known(obj, field_index) {
            // pyjitpl.py:1077-1080 cache hit (see opimpl_getfield_gc_i above).
            ctx.profiler().count_ops(
                OpCode::QuasiimmutField,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
        } else {
            ctx.heap_cache_mut().quasi_immut_now_known(obj, field_index);
            ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
            if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
                ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
            }
        }
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureR
    } else {
        OpCode::GetfieldGcR
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr.clone());
    // pyjitpl.py:948-949 `resbox = execute_with_descr(...); upd.getfield_now_known(resbox)`.
    // Pair the recorded opref with the live ref so subsequent
    // `box_value(result)` mirrors RPython's executor-returned Box.
    let live_value = if let Some(majit_ir::Value::Ref(struct_ref)) = ctx.box_value(obj) {
        let struct_ptr = struct_ref.0 as i64;
        if struct_ptr != usize::MAX as i64 && struct_ptr != 0 {
            ctx.field_sanity_load(struct_ptr, &descr, majit_ir::Type::Ref)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(live_value) = live_value {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getfield_now_known(obj, field_index, result);
    result
}

// Note: pyre does not currently route GetfieldGcF/GetfieldGcPureF through
// state.rs. Float field unboxing goes via the codewriter-generated
// `getfield_gc_f_pureornot` (majit-translate/src/codegen.rs),
// which — matching RPython's pyjitpl.py opimpl_getfield_gc_f — records
// the GC op without folding. The optimizer's `optimize_GETFIELD_GC_F`
// (= `optimize_GETFIELD_GC_I` via RPython's alias) handles folding.

/// Unbox int with proper GuardClass resume data via the frame impl's
/// `generate_guard`.  Generic over `WalkerFrameOps` so both `MIFrame`
/// (trait dispatch) and `WalkContext` (walker dispatch) can invoke the
/// same lowering.
pub(crate) fn trace_unbox_int_with_resume<F: crate::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: OpRef,
    int_type_addr: i64,
) -> OpRef {
    trace_unbox_int_with_resume_descr(frame, obj, int_type_addr, crate::descr::int_intval_descr())
}

/// `(guard_class type addr, intval descr)` for unboxing a concrete int- or
/// bool-valued operand. `W_BoolObject` is a `W_IntObject` subclass sharing
/// the `intval` field, so a bool unboxes through the same `getfield` but
/// guards its own `&BOOL_TYPE` vtable and keys the heapcache by the bool
/// field descr.
pub(crate) fn int_or_bool_unbox_type_descr(
    concrete: pyre_object::PyObjectRef,
) -> (i64, majit_ir::DescrRef) {
    if pyre_object::tagged_int::CAN_BE_TAGGED
        && !concrete.is_null()
        && pyre_object::tagged_int::is_tagged_int(concrete)
    {
        return (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            crate::descr::int_intval_descr(),
        );
    }
    if !concrete.is_null() && unsafe { pyre_object::is_bool(concrete) } {
        (
            &pyre_object::pyobject::BOOL_TYPE as *const _ as i64,
            crate::descr::bool_intval_descr(),
        )
    } else {
        (
            &pyre_object::pyobject::INT_TYPE as *const _ as i64,
            crate::descr::int_intval_descr(),
        )
    }
}

pub(crate) fn trace_unbox_int_with_resume_descr<F: crate::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    if pyre_object::tagged_int::CAN_BE_TAGGED {
        if let Some(majit_ir::Value::Ref(r)) = frame.ctx().concrete_of_opref(obj) {
            if r != majit_ir::GcRef::NO_CONCRETE {
                let o = r.as_usize() as pyre_object::PyObjectRef;
                if !o.is_null() {
                    if pyre_object::tagged_int::is_tagged_int(o) {
                        let lowbit =
                            crate::helpers::emit_tag_lowbit_test(frame.ctx_mut(), obj, true);
                        frame.generate_guard(OpCode::GuardTrue, &[lowbit]);
                        return crate::helpers::emit_untag_int(
                            frame.ctx_mut(),
                            obj,
                            pyre_object::tagged_int::untag_int(o),
                        );
                    } else {
                        let lowbit =
                            crate::helpers::emit_tag_lowbit_test(frame.ctx_mut(), obj, false);
                        frame.generate_guard(OpCode::GuardFalse, &[lowbit]);
                    }
                }
            }
        }
    }
    // pyjitpl.py GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0.
    if !frame.ctx().heap_cache().is_class_known(obj) {
        let type_const = frame.ctx_mut().const_int(type_addr);
        frame.generate_guard(OpCode::GuardClass, &[obj, type_const]);
        frame
            .ctx_mut()
            .heap_cache_mut()
            .class_now_known(obj, type_addr);
    }
    crate::trace_unbox_int(
        frame.ctx_mut(),
        obj,
        type_addr,
        crate::descr::ob_type_descr(),
        intval_descr,
    )
}

/// Unbox a `W_LongObject` (whose BigInt fits in i64) into a raw i64 OpRef.
///
/// Lowering — mirrors `_int_w_unsafe()` (longobject.py:127) under the
/// fits_int precondition (`is_plain_int1` accepts both W_IntObject and
/// fits_int W_LongObject; listobject.py:1957-1958):
///
/// 1. `GUARD_CLASS(obj, LONG_TYPE)` — concrete type observed at trace.
/// 2. `residual_call(jit_w_long_fits_int, obj) -> i64` — runtime
///    fits_int probe. Subsequent trace executions may see a W_LongObject
///    whose BigInt has grown out of i64 range; the residual call captures
///    that observation per execution.
/// 3. `GUARD_TRUE(fits_int_result)` — bridge if the runtime BigInt does
///    not fit (alternatively: deopt back to the interpreter).
/// 4. `residual_call(jit_w_long_toint, obj) -> i64` —
///    `W_LongObject.toint()` (`longobject.py:138`) → `rbigint.toint()`
///    (`rbigint.py:465`, elidable). OverflowError is statically
///    unreachable post-fits-int GUARD_TRUE.
///
/// Unlike the int arms (`trace_unbox_int_with_resume_descr`,
/// `trace_guarded_int_payload`), step 1's `GUARD_CLASS` carries no
/// `is_tagged_int` pre-check before its `ob_type` deref, and needs none: a
/// tagged immediate can never select this arm. Both call sites gate it on
/// `is_long(concrete)` — `trace_plain_int_payload` (`trace_opcode.rs` `if
/// is_long(concrete_item)`) and `unbox_int_or_long_for_int_strategy` (fed by
/// `unbox_long = is_long(concrete_value)` in `detect_list_setitem_strategy`).
/// `is_long` routes through `py_type_check`, which short-circuits a tagged
/// immediate to `ptr::eq(tp, &INT_TYPE)` — false for `LONG_TYPE` — before any
/// `ob_type` deref. A `W_LongObject` is a distinct 8-aligned heap struct and is
/// never tagged, so `is_long` is definitionally false for an odd pointer. The
/// `GUARD_CLASS` deref therefore only ever sees an aligned heap pointer. Both
/// call sites also early-return `value_type == Int` operands before this arm.
pub(crate) fn trace_unbox_long_with_resume<F: crate::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: OpRef,
    long_type_addr: i64,
) -> OpRef {
    if !frame.ctx().heap_cache().is_class_known(obj) {
        let type_const = frame.ctx_mut().const_int(long_type_addr);
        frame.generate_guard(OpCode::GuardClass, &[obj, type_const]);
        frame
            .ctx_mut()
            .heap_cache_mut()
            .class_now_known(obj, long_type_addr);
    }
    let fits_fn = frame
        .ctx_mut()
        .const_int(pyre_object::longobject::jit_w_long_fits_int as *const () as usize as i64);
    let fits_descr = crate::descr::make_jit_w_long_fits_int_calldescr();
    let fits_result =
        frame
            .ctx_mut()
            .record_op_with_descr(OpCode::CallI, &[fits_fn, obj], fits_descr);
    frame.generate_guard(OpCode::GuardTrue, &[fits_result]);
    let unbox_fn = frame
        .ctx_mut()
        .const_int(pyre_object::longobject::jit_w_long_toint as *const () as usize as i64);
    let unbox_descr = crate::descr::make_jit_w_long_toint_calldescr();
    frame
        .ctx_mut()
        .record_op_with_descr(OpCode::CallI, &[unbox_fn, obj], unbox_descr)
}

/// Unbox float with proper GuardClass resume data via the frame impl's
/// `generate_guard`.  Generic over `WalkerFrameOps`.
pub(crate) fn trace_unbox_float_with_resume<F: crate::walker_frame_ops::WalkerFrameOps>(
    frame: &mut F,
    obj: OpRef,
    float_type_addr: i64,
) -> OpRef {
    if !frame.ctx().heap_cache().is_class_known(obj) {
        let type_const = frame.ctx_mut().const_int(float_type_addr);
        frame.generate_guard(OpCode::GuardClass, &[obj, type_const]);
        frame
            .ctx_mut()
            .heap_cache_mut()
            .class_now_known(obj, float_type_addr);
    }
    crate::trace_unbox_float(
        frame.ctx_mut(),
        obj,
        float_type_addr,
        crate::descr::ob_type_descr(),
        crate::descr::float_floatval_descr(),
    )
}

pub(crate) unsafe fn objspace_compare_ints(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    unsafe {
        let lhs = w_int_get_value(lhs_obj);
        let rhs = w_int_get_value(rhs_obj);
        match op {
            ComparisonOperator::Less => lhs < rhs,
            ComparisonOperator::LessOrEqual => lhs <= rhs,
            ComparisonOperator::Greater => lhs > rhs,
            ComparisonOperator::GreaterOrEqual => lhs >= rhs,
            ComparisonOperator::Equal => lhs == rhs,
            ComparisonOperator::NotEqual => lhs != rhs,
        }
    }
}

/// baseobjspace as_float: coerce int|float → f64.
/// Called only for int/float operands in the tracing fast path.
/// Long operands are handled by residual fallback, not this function.
unsafe fn as_float_for_trace(obj: PyObjectRef) -> f64 {
    unsafe {
        if is_float(obj) {
            w_float_get_value(obj)
        } else if is_int(obj) {
            w_int_get_value(obj) as f64
        } else {
            0.0 // unreachable in trace fast path — long triggers residual
        }
    }
}

/// Compare two numeric values as floats. Handles float_pair (int+float)
/// via as_float coercion matching baseobjspace::float_lt/le/gt/ge/eq/ne.
/// Long operands don't reach here — they trigger residual fallback.
pub(crate) unsafe fn objspace_compare_floats(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    unsafe {
        let lhs = as_float_for_trace(lhs_obj);
        let rhs = as_float_for_trace(rhs_obj);
        match op {
            ComparisonOperator::Less => lhs < rhs,
            ComparisonOperator::LessOrEqual => lhs <= rhs,
            ComparisonOperator::Greater => lhs > rhs,
            ComparisonOperator::GreaterOrEqual => lhs >= rhs,
            ComparisonOperator::Equal => lhs == rhs,
            ComparisonOperator::NotEqual => lhs != rhs,
        }
    }
}

/// virtualizable.py:94 `getattr(virtualizable, fieldname)` parity for
/// the `locals_cells_stack_w` array field. Materialises the array
/// pointer that step 2 (`lst[i]` → `GETARRAYITEM_GC_R`) indexes.
///
/// TODO: the upstream-orthodox emission is
/// `OpCode::GetfieldGcR` because `pyframe_locals_cells_stack_descr`
/// is field 0 of `PYFRAME_DESCR_GROUP` with `field_type = Type::Ref`
/// on a `PYFRAME_GC_TYPE_ID`-typed PyFrame, so the read goes through
/// the GC barrier in RPython's `rclass.py` getfield emission.
///
/// Cranelift backend status (MAJIT_PROBE_GETFIELD_GC_R=1):
///   - `OpCode::GetfieldGcR` lowering exists at
///     `majit-backend-cranelift/src/compiler.rs:10691`.
///   - Direct swap on fib_recursive panics inside
///     `gc_alloc_nursery_shim` with non-unwinding abort.  The
///     post-getfield write-barrier path triggers a nursery
///     allocation (remembered-set slot) which overflows or hits
///     a missing slow-path stub.
///
/// The convergence path is to either (a) implement the missing
/// nursery allocation slow-path for the post-GetfieldGcR remembered
/// set write, or (b) audit why the GC barrier emits a nursery
/// allocation here when dynasm doesn't.  Both are separate
/// cranelift backend work that is not yet done.  Until then the
/// emission stays
/// `GetfieldRawI` and the runtime descr's `field_type = Type::Ref`
/// preserves the optimizer's boxed-pointer view.
pub(crate) fn frame_locals_cells_stack_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldRawI,
        &[frame],
        frame_locals_cells_stack_descr(),
    )
}

/// Read from frame's `locals_cells_stack_w` array. Caller passes the
/// `*mut FixedObjectArray` block-base pointer (output of
/// `frame_locals_cells_stack_array`); the
/// [`pyobject_gcarray_descr`] sets
/// `base_size = FIXED_ARRAY_ITEMS_OFFSET` so the descriptor skips the
/// length prefix and `GETARRAYITEM_GC_R(array, i)` lands on
/// `items[i]`.
///
/// Uses GcR (Ref-typed) to match RPython's GETARRAYITEM_GC_R, ensuring
/// the optimizer knows these are boxed pointers.
pub(crate) fn trace_array_getitem_value(ctx: &mut TraceCtx, array: OpRef, index: OpRef) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, array, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// Helper: project (array, index) operands and dispatch
/// `array_sanity_load` to recover the executor-returned payload for
/// the fresh GetarrayitemGc<I|R|F> result.  Mirrors
/// `rpython/jit/metainterp/executor.py:117` `do_getarrayitem_gc_*`
/// which reads `arraybox.getref_base()` / `indexbox.getint()` from
/// the Box directly — pyre's `box_value` exposes the same chain (Const
/// pool / standard-virtualizable shadow / `opref_concrete` stamp) so
/// any operand whose Box.value is known unblocks the sanity load
/// (not gated on the Const-only fast path).
fn array_load_for_cache(
    ctx: &majit_metainterp::TraceCtx,
    array: OpRef,
    index: OpRef,
    descr: &DescrRef,
    kind: majit_ir::Type,
) -> Option<majit_ir::Value> {
    let Some(majit_ir::Value::Ref(array_ref)) = ctx.box_value(array) else {
        return None;
    };
    let array_ptr = array_ref.0 as i64;
    if array_ptr == usize::MAX as i64 || array_ptr == 0 {
        return None;
    }
    let Some(majit_ir::Value::Int(index_value)) = ctx.box_value(index) else {
        return None;
    };
    ctx.array_sanity_load(array_ptr, index_value, descr, kind)
}

/// Read from frame's locals_cells_stack_w — namespace access path.
pub(crate) fn trace_raw_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, array, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// `pyjitpl.py:832` `arraybox = opimpl_getfield_gc_r(listbox, itemsdescr)`
/// followed by `getarrayitem_gc(arraybox, idx, arraydescr)`.
///
/// Caller passes the `items_block` Ref (output of
/// `opimpl_getfield_gc_r` against the `items` /  `wrappeditems` field)
/// directly. The `pyobject_gcarray_descr` here is
/// `Ptr(GcArray(OBJECTPTR))` with `base_size = ITEMS_BLOCK_ITEMS_OFFSET`
/// (= length-prefix size), `item_size = sizeof(Ptr)` (word-sized: 4 on
/// wasm32, 8 on 64-bit), `item_type = Ref`, matching
/// `rpython/rtyper/lltypesystem/rlist.py:84` `GcArray(OBJECTPTR)`.
///
/// Replaces the prior two-step `IntAdd(items_block, OFFSET) +
/// raw-array op` deviation with the upstream single-op shape.
pub(crate) fn trace_items_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Pure variant of [`trace_items_block_getitem_value`] — emits
/// `getarrayitem_gc_pure_r(block, index)` against the SAME
/// `pyobject_gcarray_descr` (`Ptr(GcArray(OBJECTPTR))`) for an
/// IMMUTABLE backing array (`W_TupleObject.wrappeditems`,
/// `tupleobject.py:381` `_immutable_fields_ = ['wrappeditems[*]']`).
///
/// Purity is carried ONLY by selecting `GetarrayitemGcPureR`; the descr
/// is the unchanged shared singleton (the items/gcarray descr must NOT
/// be marked pure — that would make every container of this strategy
/// non-invalidatable). `OptPure` CSEs / const-folds the pure op and
/// never invalidates it on an intervening write, which is sound here
/// because the tuple body is immutable.
///
/// Recording the pure op directly is the walker-native analogue of the
/// codewriter, which reaches `getarrayitem_gc_*_pure` through the
/// oopspec lowering of an immutable/foldable read (`jtransform.py:1891`);
/// the opcode-level effect is identical.
pub(crate) fn trace_items_block_getitem_value_pure(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_gcarray_descr();
    let result =
        ctx.record_op_with_descr(OpCode::GetarrayitemGcPureR, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Ref) {
        ctx.set_opref_concrete(result, live_value);
    }
    result
}

/// Companion of [`trace_items_block_getitem_value`] — emits
/// `setarrayitem_gc(block, index, value)` against `pyobject_gcarray_descr`.
pub(crate) fn trace_items_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_gcarray_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    // pyjitpl.py:980 `upd.setarrayitem(valuebox)` — cache stores the
    // Box identity (`value` OpRef); cache-hit readers resolve the
    // intrinsic value via `box_value(cached)` at hit time.
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// Write to frame's locals_cells_stack_w array.
/// Uses Gc (GC-typed) to match RPython's SETARRAYITEM_GC.
pub(crate) fn trace_raw_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[array, index, value], descr);
    // pyjitpl.py:980 `upd.setarrayitem(valuebox)` parity — cache
    // stores the Box identity (`value` OpRef); cache-hit readers
    // resolve the intrinsic value via `box_value(cached)` at hit
    // time.
    ctx.heapcache_setarrayitem(array, index, descr_idx, value);
}

/// `GetarrayitemGcI(block, index)` against `int_gcarray_descr` — the
/// `IntegerListStrategy` GC-array read. Mirrors
/// [`trace_items_block_getitem_value`] (the Object-strategy GcArray(OBJECTPTR)
/// read) but with an `Int` item type, so the read routes through the heap cache.
pub(crate) fn trace_int_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = int_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcI, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Int) {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Companion of [`trace_int_block_getitem_value`] — `SetarrayitemGc(block,
/// index, value)` against `int_gcarray_descr` (int items carry no pointer, so
/// no write barrier is emitted).
pub(crate) fn trace_int_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = int_gcarray_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// `GetarrayitemGcF(block, index)` against `float_gcarray_descr` — the
/// `FloatListStrategy` GC-array read (heap-cached). See
/// [`trace_int_block_getitem_value`].
pub(crate) fn trace_float_block_getitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = float_gcarray_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heapcache_getarrayitem(block, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcF, &[block, index], descr.clone());
    if let Some(live_value) = array_load_for_cache(ctx, block, index, &descr, majit_ir::Type::Float)
    {
        ctx.set_opref_concrete(result, live_value);
    }
    ctx.heapcache_getarrayitem_now_known(block, index, descr_idx, result);
    result
}

/// Companion of [`trace_float_block_getitem_value`] — `SetarrayitemGc(block,
/// index, value)` against `float_gcarray_descr`.
pub(crate) fn trace_float_block_setitem_value(
    ctx: &mut TraceCtx,
    block: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = float_gcarray_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[block, index, value], descr);
    ctx.heapcache_setarrayitem(block, index, descr_idx, value);
}

/// pyframe.py:49 `self.w_globals` — read the canonical dict object
/// from the frame.  Returns a PyObjectRef (W_DictObject or
/// W_ModuleDictObject).  FFI helpers that receive namespace_ptr
/// chase dict_storage_proxy internally.
pub(crate) fn frame_get_globals_obj(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldGcR,
        &[frame],
        crate::descr::pyframe_w_globals_obj_descr(),
    )
}

/// Read through w_globals → dict_storage_proxy to reach the raw
/// DictStorage* for slot-based namespace reads (celldict quasiimmut
/// path).  Only valid when globals is a W_ModuleDictObject.
/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
pub fn concrete_stack_value(frame: usize, abs_idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    let arr = unsafe { &*arr_ptr };
    arr.as_slice().get(abs_idx).copied()
}

/// Read up to `max_len` slots of the GC-rooted live virtualizable frame's
/// `locals_cells_stack_w` as concrete `Ref` values.  The live frame
/// (`sym.concrete_vable_ptr`) sits on the CURRENT_FRAME chain, so
/// `walk_pyframe_roots` forwards its slots on every collection — reading
/// them here is GC-safe.  This is the source the bridge concrete shadow
/// must use instead of the off-heap resume-decoded array, whose `Ref`
/// entries dangle once a minor collection (residual virtual materialization
/// during bridge setup) moves the referent but cannot forward the decode
/// `Vec`.  Falls back to `fallback` (the decoded values) when no live
/// frame/array is bound (unit-test / init-before-run path, no GC hazard).
fn live_frame_array_values(
    vable_ptr: usize,
    max_len: usize,
    fallback: &[majit_ir::Value],
) -> Vec<majit_ir::Value> {
    if vable_ptr == 0 {
        return fallback.to_vec();
    }
    let f = unsafe { &*(vable_ptr as *const pyre_interpreter::pyframe::PyFrame) };
    let lp = f.locals_cells_stack_w;
    if lp.is_null() {
        return fallback.to_vec();
    }
    let arr = unsafe { &*lp };
    let base = arr.items_ptr() as *const pyre_object::PyObjectRef;
    let n = max_len.min(arr.len());
    (0..n)
        .map(|i| majit_ir::Value::Ref(majit_ir::GcRef(unsafe { *base.add(i) } as usize)))
        .collect()
}

/// pyframe.py:107-110: `locals_cells_stack_w` length =
/// `co_nlocals + ncellvars + nfreevars + co_stacksize`. Returns the
/// full heap-side array length (matching `virtualizable.py:86-99
/// read_boxes` which iterates `len(lst)` over the full array).
pub(crate) fn concrete_frame_array_len(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *const pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() {
        return None;
    }
    Some(unsafe { &*arr_ptr }.as_slice().len())
}

/// pyframe.py:111: valuestackdepth = co_nlocals + ncellvars + nfreevars.
/// Returns the stack base index (nlocals + ncells) for the given frame.
/// This is the number of non-stack slots in the unified locals_cells_stack_w
/// array: local variables + cell/free variable slots.
pub(crate) fn concrete_nlocals(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    let code = unsafe { &*raw_code };
    let nlocals = code.varnames.len();
    let ncells = pyre_interpreter::pyframe::ncells(code);
    Some(nlocals + ncells)
}

/// Return the absolute valuestackdepth.
pub(crate) fn concrete_stack_depth(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    Some(unsafe { *(frame_ptr.add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize) })
}

/// Write the absolute valuestackdepth into the concrete `PyFrame` at
/// `frame`. The trait `interpret()` leg traces on a heap snapshot that
/// is never concretely stepped (`trace.rs` KNOWN DIVERGENCE), so a
/// multi-frame bridge's snapshot keeps the `valuestackdepth` it was
/// reconstructed with (the guard-failure resume depth, mid-iteration).
/// The symbolic walk DOES track the live depth (`push_typed_value` /
/// `pop_value`). At the loop-header close the back-edge depth is the
/// loop-header invariant — the live symbolic value — so syncing the
/// snapshot here lets `concrete_valuestackdepth()` (the reader
/// `close_loop_args_at` trusts, matching RPython reading the single real
/// frame) report the merge-point depth instead of the stale seed.
pub(crate) fn set_concrete_stack_depth(frame: usize, depth: usize) {
    if frame != 0 {
        unsafe {
            *((frame as *mut u8).add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *mut usize) = depth;
        }
    }
}

/// Derive `(num_locals, num_locals + max_stackdepth)` from a `CodeObject`.
///
/// Mirrors the `(callee_nlocals, callee_vsd)` pair the trace-side reads
/// from `driver.get_compiled_meta(callee_key)` for CALL_ASSEMBLER
/// emission (`trace_opcode.rs:4448-4450`). The second value is sized to
/// match `pyframe.rs:1576` (`alloc_fixed_array_with_header(num_locals +
/// num_cells + max_stack, ...)`) — i.e. heap capacity rather than live
/// depth. Used as the fallback shape when no `compiled_meta` exists yet
/// (e.g. tmp_callback target where `compile_tmp_callback` produced a
/// JCT but no compiled-loop metadata).
#[allow(dead_code)]
pub(crate) fn callee_layout_for_call_assembler(
    code: &pyre_interpreter::CodeObject,
) -> (usize, usize) {
    let nlocals = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);
    let stack_only = code.max_stackdepth as usize;
    (nlocals, nlocals + stack_only)
}

/// `celldict.py:42-50 getdictvalue_no_unwrapping` slot lookup against the
/// module dict's `ModuleDictStorage` (the cell store), bypassing the
/// `DictStorage` shadow.  `None` for non-module dicts and after
/// `switch_to_object_strategy`.  Used by the cell fast path to derive the
/// elidable lookup key.
pub(crate) fn module_dict_cell_slot_direct(obj: PyObjectRef, name: &str) -> Option<usize> {
    unsafe { pyre_object::dictmultiobject::module_dict_cell_slot_of(obj, name) }
}

/// `celldict.py:53-54 _getdictvalue_no_unwrapping_pure` — the raw stored
/// value-or-cell at `slot` (not unwrapped), read at trace time to
/// classify the cell (`ObjectMutableCell` → live `GetfieldGcR`, raw value
/// → const fold, `IntMutableCell` → name-based fallback).
pub(crate) fn module_dict_cell_value_direct(obj: PyObjectRef, slot: usize) -> Option<PyObjectRef> {
    unsafe { pyre_object::dictmultiobject::module_dict_cell_at(obj, slot) }
}

/// pyjitpl.py:1074-1089 `opimpl_record_quasiimmut_field` for namespace
/// slot folds: record the dependency marker and arm the pending
/// GUARD_NOT_INVALIDATED once per heapcache epoch.
pub(crate) fn record_namespace_quasiimmut_field(
    ctx: &mut TraceCtx,
    obj: OpRef,
    slot: OpRef,
    slot_index: u32,
) {
    if ctx.heap_cache().is_quasi_immut_known(obj, slot_index) {
        ctx.profiler().count_ops(
            OpCode::QuasiimmutField,
            majit_metainterp::counters::HEAPCACHED_OPS,
        );
        return;
    }
    ctx.heap_cache_mut().quasi_immut_now_known(obj, slot_index);
    ctx.record_op(OpCode::QuasiimmutField, &[obj, slot]);
    if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
        ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
    }
}

/// virtualizable.py:44 + interp_jit.py:25-31 —
/// `locals_cells_stack_w[*]` is declared as a W_Root array, so every
/// item's JIT type is GCREF (Type::Ref). W_IntObject/W_FloatObject are
/// stored as Ref pointers; unboxing happens inside trace opcode handlers
/// via `guard_class + getfield_gc_pure_i` / `_f`, never at the
/// virtualizable slot level.
pub(crate) fn concrete_virtualizable_slot_type(_value: PyObjectRef) -> Type {
    Type::Ref
}

/// pyjitpl.py:3048-3091 `raise_continue_running_normally` parity for the
/// authoritative full-body walk.  The walk concretely executed the traced
/// region's residual calls, so a walk end that returns control to the
/// interpreter must hand back the END-of-walk frame state — the same
/// contract the guard-failure blackhole writeback satisfies
/// (`handle_jitexception` re-enters `eval_loop_jit` from the frame's own
/// `last_instr`).  Without this the interpreter (or a freshly entered
/// compiled loop, whose preamble reloads vable fields from the heap)
/// re-runs the walked region from its START state, re-applying every
/// concretely executed side effect.
///
/// Writes every live `locals_cells_stack_w` slot from the virtualizable
/// shadow's concrete half (`virtualizable_entry_at` — the value half of
/// the Box pair the authoritative walk maintains), `valuestackdepth` =
/// the `LiveVars` forward-stack-analysis depth at the merge point (the
/// cached `depth_at_py_pc` derivation, and `last_instr = resume_py_pc - 1` so
/// `next_instr()` re-enters at the merge point.  A bridge walk enters at
/// a guard pc whose stack depth differs from the merge point's, so the
/// depth must come from the analysis, not from the frame's entry value.
///
/// All-or-nothing: returns false (frame untouched) when any live slot
/// lacks a shadow entry, when the depth analysis has no entry for the
/// merge pc, or when the walked region net-changed the frame's block
/// chain (`lastblock` — the flush writes only locals/stack/vsd/
/// last_instr, so a block push/pop inside the walked region would leave
/// the adopted frame's chain inconsistent with its pc).  The caller then
/// keeps the legacy replay-from-start behavior.
pub(crate) fn flush_walk_end_state_to_frame(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
) -> bool {
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, None, &[])
}

/// `flush_walk_end_state_to_frame` plus an optional in-flight FOR_ITER item
/// delivery (#32 S2).  When `push` is `Some((item, body_pc))` and the flush at
/// `resume_py_pc` (a FOR_ITER header) commits, the consumed `item` is pushed
/// one slot above the flushed operand stack and `last_instr` is repositioned to
/// `body_pc - 1` so `next_instr()` re-enters the FOR_ITER body — delivering the
/// already-advanced iteration exactly once (the FOR_ITER itself is NOT re-run).
/// `push = None` is byte-identical to the plain flush.
pub(crate) fn flush_walk_end_state_to_frame_with_item(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
    push: Option<(PyObjectRef, usize)>,
) -> bool {
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, push, &[])
}

pub(crate) fn flush_walk_end_state_to_frame_with_stack_overrides(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
    stack_overrides: &[(usize, PyObjectRef)],
) -> bool {
    flush_walk_end_state_to_frame_inner(ctx, frame, resume_py_pc, None, stack_overrides)
}

fn flush_walk_end_state_to_frame_inner(
    ctx: &TraceCtx,
    frame: usize,
    resume_py_pc: usize,
    push: Option<(PyObjectRef, usize)>,
    stack_overrides: &[(usize, PyObjectRef)],
) -> bool {
    if frame == 0 {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return false;
    };
    // Stack depth at the merge point from the cached forward analysis:
    // absolute vsd = stack_base + depth, stack_base = nlocals + ncells
    // = `concrete_nlocals`.
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return false;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    let Some(depth) = crate::liveness::liveness_for(raw_code)
        .depth_at_py_pc()
        .get(resume_py_pc)
        .copied()
    else {
        return false;
    };
    let end_vsd = nlocals + depth as usize;
    let live = end_vsd.max(nlocals);
    let base = info.num_static_extra_boxes;
    // Block-chain net-change check: the shadow's `lastblock` static box
    // still holds the entry chain head iff the walk pushed/popped no
    // blocks (a balanced push+pop allocates a fresh head and also
    // declines — conservative).
    let lastblock_static = info
        .static_fields
        .iter()
        .position(|f| f.name == "lastblock");
    let Some(lastblock_idx) = lastblock_static else {
        return false;
    };
    let Some((_opref, shadow_lastblock)) = ctx.virtualizable_entry_at(lastblock_idx) else {
        return false;
    };
    let frame_lastblock = unsafe { *(frame_ptr.add(PYFRAME_LASTBLOCK_OFFSET) as *const usize) };
    match shadow_lastblock {
        Value::Ref(r) => {
            if r.0 != frame_lastblock {
                return false;
            }
        }
        _ => return false,
    }
    // Validation pass first: it allocates nothing, so entry presence
    // cannot change under it.  Commit only when every live slot resolves.
    let stack_override_at = |abs: usize| -> Option<PyObjectRef> {
        stack_overrides
            .iter()
            .find_map(|&(slot, value)| (slot == abs).then_some(value))
    };
    for abs in 0..live {
        let Some((_opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
            return false;
        };
        if abs >= nlocals && !stack_overrides.is_empty() {
            if stack_override_at(abs).is_none() {
                return false;
            }
            continue;
        }
        // An operand-STACK slot (`abs >= nlocals`) that resolves to a NULL Ref
        // is an UNPOPULATED shadow slot, not a live value: the virtualizable
        // shadow tracks locals/cells faithfully but its stack region is only
        // valid at a merge point (loop header) where the walk's stack effects
        // have settled.  At an arbitrary mid-opcode resume pc (an
        // `abort_permanent` marker reached partway through an opcode's stack
        // build — e.g. a MAKE_FUNCTION whose LOAD_CONST'd code object the walk
        // tracked symbolically, never writing it to the shadow) those slots
        // read back NULL.  Writing NULL into a live operand-stack slot and
        // resuming there faults the interpreter (it pops NULL where a real
        // object is expected).  Decline so the legacy replay reconstructs the
        // frame from its start state instead.  A local slot may legitimately
        // be NULL (an unbound local), so this only guards the stack region.
        if abs >= nlocals && matches!(value, Value::Ref(r) if r.0 == 0) {
            return false;
        }
    }
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    // A pushed in-flight item (#32 S2) needs one slot above the `live` flushed
    // slots; require the capacity up front so a decline happens BEFORE any
    // frame mutation (all-or-nothing).
    let need = if push.is_some() { live + 1 } else { live };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < need {
        return false;
    }
    // Commit one slot at a time, re-reading the shadow entry per slot:
    // boxing an Int/Float slot allocates and may trigger a minor
    // collection, which moves nursery objects — the trace-ctx forwarding
    // hook keeps the shadow entries current, and each already-written
    // slot is reachable from the (rooted) frame, so neither side goes
    // stale across the loop.
    for abs in 0..live {
        let Some((_opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
            return false;
        };
        let boxed = if abs >= nlocals {
            stack_override_at(abs).unwrap_or_else(|| boxed_slot_value_for_type(Type::Ref, &value))
        } else {
            boxed_slot_value_for_type(Type::Ref, &value)
        };
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = boxed;
        }
    }
    unsafe {
        let pf = &mut *(frame as *mut PyFrame);
        pf.valuestackdepth = end_vsd;
        pf.last_instr = resume_py_pc as isize - 1;
    }
    // #32 S2: deliver the in-flight FOR_ITER item.  The flush wrote the
    // FOR_ITER-header operand stack (the iterator on TOS) into slots
    // `0..end_vsd`; the continue arm keeps the iterator and pushes the
    // consumed item above it (codewriter FOR_ITER continue arm,
    // `opcode_for_iter` never pops the iterator), so a single write at
    // `end_vsd` lands the item where the body's first opcode expects TOS.
    // `body_pc` is the FOR_ITER `orgpc + 1`; resume there so the FOR_ITER is
    // not re-executed (which would re-advance the iterator).  The slot at
    // `end_vsd` is within capacity (the early `need = live + 1` validation).
    if let Some((item, body_pc)) = push {
        unsafe {
            (*arr_ptr).as_mut_slice()[end_vsd] = item;
            let pf = &mut *(frame as *mut PyFrame);
            pf.valuestackdepth = end_vsd + 1;
            pf.last_instr = body_pc as isize - 1;
        }
    }
    true
}

/// gh#467 forward-flush AT an inlined-callee CALL boundary.  When an
/// supported abort fires inside an inline sub-walk whose callee executed no
/// concrete effect, the outer frame is flushed as of the CALL that
/// entered the callee: the locals/cells region from the vable shadow (exactly
/// like [`flush_walk_end_state_to_frame`]), the operand-stack region rebuilt
/// from the concrete `call_stack` (`[callable, null_or_self, args...]` the
/// walker holds), `valuestackdepth` set to cover both, and
/// `last_instr = call_py_pc - 1` so `next_instr()` re-executes the CALL in the
/// interpreter — running the callee from scratch.  Everything the walk applied
/// BEFORE the CALL stands (the caller commits the store journals), so the
/// non-journaled pre-CALL store applies exactly once.  This is the frame-level
/// analogue of `run_blackhole_interp_to_cancel_tracing` (`pyjitpl.py:2949`)
/// continuing forward from the abort, without the inner-frame reconstruction
/// (#126/#215): the outer frame re-runs the whole call.
///
/// Unlike the merge-point flush, the operand stack does NOT come from the vable
/// shadow (whose stack region is only valid at a merge point — at a mid-
/// statement CALL those slots read NULL); it is the caller-provided
/// `call_stack`, whose height MUST match the forward analysis's `depth_at_py_pc`
/// at `call_py_pc` (a mismatch means the reconstruction disagrees with the
/// encoded stack shape → decline).  All-or-nothing: returns false (frame
/// untouched) on any depth mismatch, an unresolved live local, a net block-chain
/// change, or insufficient array capacity; the caller then keeps the legacy
/// replay.
pub(crate) fn flush_walk_end_state_at_outer_call(
    ctx: &TraceCtx,
    frame: usize,
    call_py_pc: usize,
    call_stack: &[PyObjectRef],
) -> bool {
    if frame == 0 {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return false;
    };
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return false;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    // The analysis depth at the CALL pc is the live operand-stack height there
    // (`callable + null_or_self + args`).  The reconstructed `call_stack` must
    // match it exactly — a disagreement means the residual_call operand list
    // does not model the interpreter's CALL stack for this call shape, so the
    // resumed frame would be mis-sized.  Decline.
    let Some(depth) = crate::liveness::liveness_for(raw_code)
        .depth_at_py_pc()
        .get(call_py_pc)
        .copied()
    else {
        return false;
    };
    if depth as usize != call_stack.len() {
        return false;
    }
    let end_vsd = nlocals + call_stack.len();
    let base = info.num_static_extra_boxes;
    // Block-chain net-change check (identical to `flush_walk_end_state_to_frame`):
    // the flush writes only locals/stack/vsd/last_instr, so a push/pop inside the
    // walked region would leave the adopted frame's chain inconsistent with its
    // resumed pc.
    let Some(lastblock_idx) = info
        .static_fields
        .iter()
        .position(|f| f.name == "lastblock")
    else {
        return false;
    };
    let Some((_opref, shadow_lastblock)) = ctx.virtualizable_entry_at(lastblock_idx) else {
        return false;
    };
    let frame_lastblock = unsafe { *(frame_ptr.add(PYFRAME_LASTBLOCK_OFFSET) as *const usize) };
    match shadow_lastblock {
        Value::Ref(r) => {
            if r.0 != frame_lastblock {
                return false;
            }
        }
        _ => return false,
    }
    // Validation pass first (allocates nothing): every LOCAL slot must resolve
    // in the shadow.  The stack region is supplied by `call_stack`, not the
    // shadow, so it is not validated here.
    for abs in 0..nlocals {
        if ctx.virtualizable_entry_at(base + abs).is_none() {
            return false;
        }
    }
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < end_vsd {
        return false;
    }
    // Commit the operand stack FIRST: `call_stack` holds live nursery-resident
    // refs, and boxing an Int/Float local below can trigger a minor collection.
    // Once written into the (rooted) frame array they are forwarded with it, so
    // landing them before any allocation keeps them current.
    for (i, &value) in call_stack.iter().enumerate() {
        unsafe {
            (*arr_ptr).as_mut_slice()[nlocals + i] = value;
        }
    }
    // Commit the locals from the shadow (re-reading per slot: boxing may move
    // nursery objects, but each written slot is frame-reachable and the
    // trace-ctx forwarding hook keeps the shadow current).
    for abs in 0..nlocals {
        let Some((_opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
            return false;
        };
        let boxed = boxed_slot_value_for_type(Type::Ref, &value);
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = boxed;
        }
    }
    unsafe {
        let pf = &mut *(frame as *mut PyFrame);
        pf.valuestackdepth = end_vsd;
        pf.last_instr = call_py_pc as isize - 1;
    }
    true
}

/// Allocation-free preflight for the depth-1 post-CALL delivery. Every gate
/// is checked before the rebuilt callee runs, so a later decline cannot replay
/// effects the plain interpreter already committed.
pub(crate) fn can_flush_walk_end_state_after_outer_call(
    ctx: &TraceCtx,
    frame: usize,
    call_py_pc: usize,
    post_call_py_pc: usize,
    call_stack_len: usize,
) -> bool {
    if frame == 0 {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return false;
    };
    let frame_ptr = frame as *const u8;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_PYCODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return false;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return false;
    }
    let depths = crate::liveness::liveness_for(raw_code).depth_at_py_pc();
    if depths.get(call_py_pc).copied().map(usize::from) != Some(call_stack_len)
        || depths.get(post_call_py_pc).copied() != Some(1)
    {
        return false;
    }
    let Some(lastblock_idx) = info
        .static_fields
        .iter()
        .position(|f| f.name == "lastblock")
    else {
        return false;
    };
    let Some((_opref, Value::Ref(shadow_lastblock))) = ctx.virtualizable_entry_at(lastblock_idx)
    else {
        return false;
    };
    let frame_lastblock = unsafe { *(frame_ptr.add(PYFRAME_LASTBLOCK_OFFSET) as *const usize) };
    if shadow_lastblock.0 != frame_lastblock {
        return false;
    }
    let base = info.num_static_extra_boxes;
    if (0..nlocals).any(|abs| ctx.virtualizable_entry_at(base + abs).is_none()) {
        return false;
    }
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    !arr_ptr.is_null() && unsafe { &*arr_ptr }.as_slice().len() >= nlocals + 1
}

/// Materialize the outer frame's locals from the virtualizable shadow.
/// Callers must run their complete preflight before executing a rebuilt
/// callee; after that point a failure would make replay unsafe.
pub(crate) fn write_back_outer_locals(ctx: &TraceCtx, frame: usize) -> bool {
    if frame == 0 {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let Some(info) = ctx.virtualizable_info() else {
        return false;
    };
    let frame_ptr = frame as *mut u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    if arr_ptr.is_null() || unsafe { &*arr_ptr }.as_slice().len() < nlocals {
        return false;
    }
    let base = info.num_static_extra_boxes;
    for abs in 0..nlocals {
        let Some((_opref, value)) = ctx.virtualizable_entry_at(base + abs) else {
            return false;
        };
        let boxed = boxed_slot_value_for_type(Type::Ref, &value);
        unsafe {
            (*arr_ptr).as_mut_slice()[abs] = boxed;
        }
    }
    true
}

/// `_setup_return_value_r` parity for the outer half of a two-frame abort:
/// install the rebuilt callee's return value and resume after the CALL.
pub(crate) fn flush_walk_end_state_after_outer_call(
    ctx: &TraceCtx,
    frame: usize,
    call_py_pc: usize,
    post_call_py_pc: usize,
    call_stack_len: usize,
    retval: PyObjectRef,
) -> bool {
    if !can_flush_walk_end_state_after_outer_call(
        ctx,
        frame,
        call_py_pc,
        post_call_py_pc,
        call_stack_len,
    ) {
        return false;
    }
    let Some(nlocals) = concrete_nlocals(frame) else {
        return false;
    };
    let frame_ptr = frame as *mut u8;
    let arr_ptr = unsafe {
        *(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
            as *const *mut pyre_object::FixedObjectArray)
    };
    // Land the nursery-resident result before boxing locals can collect.
    unsafe {
        (*arr_ptr).as_mut_slice()[nlocals] = retval;
    }
    if !write_back_outer_locals(ctx, frame) {
        return false;
    }
    unsafe {
        let pf = &mut *(frame as *mut PyFrame);
        pf.valuestackdepth = nlocals + 1;
        pf.last_instr = post_call_py_pc as isize - 1;
    }
    true
}

pub(crate) fn looks_like_heap_ref(value: PyObjectRef) -> bool {
    let addr = value as usize;
    let word_align = std::mem::align_of::<usize>() - 1;
    addr >= 0x1_0000 && addr < ((1u64 << 56) as usize) && (addr & word_align) == 0
}

pub(crate) fn extract_concrete_typed_value(slot_type: Type, value: PyObjectRef) -> Value {
    match slot_type {
        Type::Int => {
            if value.is_null() {
                Value::Int(0)
            } else if looks_like_heap_ref(value) && unsafe { is_int(value) } {
                Value::Int(unsafe { w_int_get_value(value) })
            } else {
                Value::Int(value as i64)
            }
        }
        Type::Float => {
            if value.is_null() {
                Value::Float(0.0)
            } else if looks_like_heap_ref(value) && unsafe { is_float(value) } {
                Value::Float(unsafe { pyre_object::floatobject::w_float_get_value(value) })
            } else {
                Value::Float(f64::from_bits(value as u64))
            }
        }
        Type::Ref | Type::Void => Value::Ref(majit_ir::GcRef(value as usize)),
    }
}

pub(crate) fn concrete_slot_types(
    frame: usize,
    num_locals: usize,
    valuestackdepth: usize,
) -> Vec<Type> {
    let stack_only = valuestackdepth.saturating_sub(num_locals);
    let mut types = Vec::with_capacity(num_locals + stack_only);
    for idx in 0..num_locals {
        types.push(
            concrete_stack_value(frame, idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    for stack_idx in 0..stack_only {
        types.push(
            concrete_stack_value(frame, num_locals + stack_idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    types
}

impl PyreMeta {
    /// Heap capacity of `locals_cells_stack_w`, distinct from the live
    /// `valuestackdepth`.
    pub fn array_stack_only_depth(&self) -> usize {
        self.array_capacity.saturating_sub(self.num_locals)
    }
}

pub(crate) fn boxed_slot_i64_for_type(slot_type: Type, raw: i64) -> PyObjectRef {
    match slot_type {
        Type::Int => w_int_new(raw),
        Type::Float => pyre_object::floatobject::w_float_new(f64::from_bits(raw as u64)),
        Type::Ref | Type::Void => raw as PyObjectRef,
    }
}

/// virtualizable.py:136 `lst[j] = reader.load_next_value_of_type(ARRAYITEMTYPE)`:
/// pyre's `locals_cells_stack_w` array item type is GCREF, so every write to
/// a frame slot must produce a boxed PyObjectRef regardless of the label's
/// argument type (a Body-Label Int OpRef still stores a W_IntObject in the
/// array). `slot_type` is retained for call-site symmetry but no longer
/// gates the boxing decision.
pub(crate) fn boxed_slot_value_for_type(_slot_type: Type, value: &Value) -> PyObjectRef {
    match value {
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Void => PY_NULL,
    }
}

pub(crate) fn fail_arg_opref_for_typed_value(ctx: &mut TraceCtx, value: Value) -> OpRef {
    match value {
        Value::Int(v) => ctx.const_int(v),
        Value::Float(v) => ctx.const_int(v.to_bits() as i64),
        Value::Ref(r) => ctx.const_ref(r.as_usize() as i64),
        Value::Void => ctx.const_ref(PY_NULL as i64),
    }
}

pub(crate) fn frame_callable_arg_types(nargs: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(2 + nargs);
    types.push(Type::Ref);
    types.push(Type::Ref);
    for _ in 0..nargs {
        types.push(Type::Ref);
    }
    types
}

pub(crate) fn one_arg_callee_frame_helper(
    arg_type: Type,
    is_self_recursive: bool,
) -> (*const (), Vec<Type>) {
    match (is_self_recursive, arg_type) {
        (true, Type::Int) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Int],
        ),
        (true, _) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1,
            vec![Type::Ref, Type::Ref],
        ),
        (false, Type::Int) => (
            crate::callbacks::get().jit_create_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Ref, Type::Int],
        ),
        (false, _) => (
            crate::callbacks::get().jit_create_callee_frame_1,
            vec![Type::Ref, Type::Ref, Type::Ref],
        ),
    }
}

#[allow(dead_code)]
pub(crate) fn fail_arg_types_for_virtualizable_state(len: usize) -> Vec<Type> {
    let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    crate::virtualizable_gen::virt_live_value_types(len.saturating_sub(n))
}

/// `pyjitpl.py:98-119 MIFrame.copy_constants` line-by-line port.
///
/// ```python
/// def copy_constants(self, registers, constants, targetindex, ConstClass):
///     num_regs_and_consts = targetindex + len(constants)
///     if registers is None or len(registers) < num_regs_and_consts:
///         registers = [missing] * num_regs_and_consts
///     elif not we_are_translated():
///         for i in range(len(registers)):
///             registers[i] = missing
///     for i in range(len(constants)):
///         registers[targetindex] = ConstClass(constants[i])
///         targetindex += 1
///     return registers
/// ```
///
/// `ConstClass(constants[i])` is supplied as the closure `mint`. The
/// `elif not we_are_translated()` debug wipe is upstream-only
/// (translation-time fill of stale slots with `missing`); pyre always
/// runs the translated path. `MIFrame.setup` can be re-invoked on
/// `free_frames_list` reused frames (`pyjitpl.py:2471`), and each call
/// re-mints `ConstClass(constants[i])` unconditionally — the fill loop
/// here matches that.
///
/// TODOs (pre-main divergences kept for now):
///
/// 1. Grow-in-place via `resize(num_regs_and_consts, NONE)` instead of
///    full replacement `[missing] * num_regs_and_consts` (the
///    `if registers is None or len(registers) < ...` arm at
///    `pyjitpl.py:109`). Convergence is blocked by Adaptation 2 below:
///    callers like `trace_opcode.rs:5466-5476` push callee args into
///    `sym.registers_r[0..args.len()]` BEFORE invoking this helper, and
///    a full replacement would zero those slots. Migrating to
///    full-replace requires reordering the trace_opcode callee setup
///    so the regalloc-color slot writes happen after
///    `setup_kind_register_banks`, plus matching opcode-dispatch reads
///    against the post-color slots rather than sequential
///    `[0..nlocals)`. That is a multi-file restructuring outside the
///    scope of this helper.
///
/// 2. `registers_r` carries both pyre's semantic locals/stack mirror
///    AND the post-regalloc-color Ref bank (see the wider doc on
///    `setup_kind_register_banks` below). RPython's `registers_r` is a
///    pure kind-specific Ref bank — semantic locals go through
///    opcode-dispatch reads of `registers_r[color]`. Pyre's dual-role
///    is what makes Adaptation 1 load-bearing; retiring one entails
///    retiring both.
///
/// 3. Ref constants land via `ctx.const_ref(val)` — a plain Ref
///    constant — instead of upstream's `ConstPtrJitCode` subclass
///    (`history.py:361-365`) that carries an `opencoder_index = -1`
///    cache field. Upstream uses that field to fast-path the trace
///    encoder's pointer dedup at `opencoder.py:583-601
///    _cached_const_ptr` (over its `_refs_dict`). Pyre lacks the
///    per-instance cache: every `ctx.const_ref(val)` mints a fresh
///    constant OpRef (`constant_pool.rs:57-103
///    get_or_insert{,_typed}` — no HashMap-keyed dedup at this
///    layer). The two upstream dedup layers exist independently in
///    pyre:
///      - **Trace encoding** (port of `opencoder.py:583
///        _cached_const_ptr`): `opencoder.rs:1873 _encode_ptr` dedups
///        by address via `_refs_dict: HashMap<u64, u32>`, mirroring
///        upstream's `_refs_dict` lookup. The
///        `ConstPtrJitCode.opencoder_index` cache that bypasses the
///        dict has no pyre counterpart, but the underlying dedup is
///        preserved.
///      - **Resume-data numbering** (port of `resume.py:148-181
///        ResumeDataLoopMemo.large_ints / .refs`):
///        `resume.rs:3460-3490 ResumeDataLoopMemo.large_ints /
///        .refs` memo separately dedups constants when assigning
///        resume numbering tags.
fn copy_constants<F>(registers: &mut Vec<OpRef>, constants: &[i64], targetindex: usize, mut mint: F)
where
    F: FnMut(i64) -> OpRef,
{
    let num_regs_and_consts = targetindex + constants.len();
    if registers.len() < num_regs_and_consts {
        registers.resize(num_regs_and_consts, OpRef::NONE);
    }
    for (i, &val) in constants.iter().enumerate() {
        registers[targetindex + i] = mint(val);
    }
}

impl PyreSym {
    pub(crate) fn new_uninit(frame: OpRef) -> Self {
        Self {
            frame,
            execution_context: OpRef::NONE,
            symbolic_local_types: Vec::new(),
            symbolic_stack_types: Vec::new(),
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            bridge_local_oprefs: None,
            bridge_stack_oprefs: None,
            bridge_walk_entry_pc: None,
            bridge_registers_r: None,
            bridge_local_types: None,
            vable_last_instr: OpRef::NONE,
            vable_pycode: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            vable_debugdata: OpRef::NONE,
            vable_lastblock: OpRef::NONE,
            vable_w_globals: OpRef::NONE,
            vable_array_base: None,
            is_active_vable_owner: false,
            concrete_locals: Vec::new(),
            concrete_stack: Vec::new(),
            // jitcode and concrete_namespace initialized below
            jitcode: null_jitcode() as *const JitCode,
            concrete_namespace: std::ptr::null_mut(),
            is_function_entry_trace: false,
            concrete_execution_context: std::ptr::null(),
            concrete_vable_ptr: std::ptr::null_mut(),
            live_vable_frame_addr: 0,
            last_exc_value: std::ptr::null_mut(),
            class_of_last_exc_is_const: false,
            last_exc_box: OpRef::NONE,
            trace_built_exc: indexmap::IndexMap::new(),
            current_exc_value: pyre_interpreter::eval::get_current_exception(),
            current_exc_box: OpRef::NONE,
            virtualref_boxes: Vec::new(),
            // RPython pyjitpl.py:74-78 init: registers_X[i] = CONST_NULL for
            // i in num_regs. Sized lazily here — `setup_kind_register_banks`
            // resizes `registers_i` / `registers_f` once the owning JitCode is
            // bound. `registers_r` continues to be driven by the existing
            // semantic-slot logic; the encoder is not yet rewired to
            // per-bank reads for `registers_r`.
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
        }
    }

    /// `pyjitpl.py:74-90 MIFrame.setup` parity for the per-kind register
    /// files. Sizes `registers_i` and `registers_f` to
    /// `num_regs_and_consts_X`, matching the RPython MIFrame layout where
    /// each bank holds the post-regalloc-color register slots followed by
    /// the constant pool entries. The leading `num_regs_X` slots stay
    /// `OpRef::NONE` (RPython's `CONST_NULL` placeholder); the trailing
    /// `[num_regs_X..num_regs_and_consts_X)` slots are filled with the
    /// constant-pool OpRefs in `pyjitpl.py:97-119 copy_constants` order:
    ///   - `registers_i[num_regs_i + i]` ← `ctx.const_int(constants_i[i])`
    ///   - `registers_r[num_regs_r + i]` ← `ctx.const_ref(constants_r[i])`
    ///   - `registers_f[num_regs_f + i]` ← `ctx.const_float(constants_f[i])`
    /// `TraceCtx::const_int` / `const_ref` / `const_float` all mint a
    /// fresh constant OpRef per call (`constant_pool.rs:57-103
    /// get_or_insert{,_typed}`), matching RPython
    /// `ConstClass(constants[i])`'s fresh-Box allocation
    /// (`history.py:220/261/307`). Per-value dedup, where it exists
    /// upstream, lives in the resume memo
    /// (`resume.rs:3460-3490 ResumeDataLoopMemo.large_ints / .refs`
    /// per `resume.py:148-181`), not in the constant pool. Re-entering
    /// this helper therefore overwrites the trailing slots with freshly
    /// minted OpRefs across all three banks, matching `copy_constants`
    /// overwrite semantics line-by-line.
    ///
    /// `registers_r` carries the unified locals + stack-tail abstract
    /// register file. The SSA-authoritative live_r work extends the resize to
    /// `registers_r` so its size matches the post-regalloc-color shape
    /// `num_regs_and_consts_r` already in use for `registers_i` /
    /// `registers_f`.
    ///
    /// This helper ports the full upstream `pyjitpl.py:74-90
    /// MIFrame.setup` body (resize + `copy_constants`).  Pyre still keeps
    /// `registers_r` as the semantic frame mirror for stack/local writes;
    /// guard capture materializes the post-regalloc-color Ref bank from
    /// that mirror or the virtualizable shadow.  The trailing constant
    /// slots `[num_regs_X, num_regs_and_consts_X)` are filled by this
    /// helper; no production reader consumes them yet (pending the
    /// encoder's constant-bank read path).
    ///
    /// Safe to call when `self.jitcode` points at the thread-local
    /// `null_jitcode()` placeholder — the skeleton's
    /// `num_regs_and_consts_X` values are zero and the constant pools
    /// are empty, which makes both the resize and the constant fill a
    /// no-op.
    pub(crate) fn setup_kind_register_banks(&mut self, ctx: &mut TraceCtx) {
        debug_assert!(!self.jitcode.is_null());
        let (num_regs_i, num_regs_r, num_regs_f, constants_i, constants_r, constants_f) = {
            let jc = unsafe { &*self.jitcode };
            let runtime_jc = &jc.payload.jitcode;
            (
                runtime_jc.num_regs_i() as usize,
                runtime_jc.num_regs_r() as usize,
                runtime_jc.num_regs_f() as usize,
                runtime_jc.constants_i.clone(),
                runtime_jc.constants_r.clone(),
                runtime_jc.constants_f.clone(),
            )
        };
        // pyjitpl.py:98-119 `MIFrame.copy_constants` line-by-line:
        //   num_regs_and_consts = targetindex + len(constants)
        //   if registers is None or len(registers) < num_regs_and_consts:
        //       registers = [missing] * num_regs_and_consts
        //   for i in range(len(constants)):
        //       registers[targetindex] = ConstClass(constants[i])
        //       targetindex += 1
        // The "translation-time-only debug wipe" arm
        // (`elif not we_are_translated()`) has no pyre analogue.
        copy_constants(&mut self.registers_i, &constants_i, num_regs_i, |v| {
            ctx.const_int(v)
        });
        copy_constants(&mut self.registers_r, &constants_r, num_regs_r, |v| {
            ctx.const_ref(v)
        });
        copy_constants(&mut self.registers_f, &constants_f, num_regs_f, |v| {
            ctx.const_float(v)
        });
    }

    /// True when this frame is allowed to mirror writes into the
    /// metainterp-scope `TraceCtx::virtualizable_boxes` cache (RPython
    /// `metainterp.virtualizable_boxes`, pyjitpl.py:1230). Two disjoint
    /// states satisfy the predicate:
    ///
    ///   1. **portal entry** — `is_active_vable_owner == true` after
    ///      `become_active_vable_owner` (which wraps the macro-generated
    ///      `init_vable_indices`). The frame's `locals_cells_stack_w`
    ///      array IS the active virtualizable.
    ///   2. **bridge entry** — `bridge_local_oprefs == Some(...)` after
    ///      `setup_bridge_sym` calls `seed_virtualizable_boxes` to
    ///      repopulate the shadow from resume data. `is_active_vable_owner`
    ///      is cleared (`clear_active_vable`) because the bridge's
    ///      inputarg layout lacks the `[frame, last_instr, pycode,
    ///      valuestackdepth, debugdata, lastblock, w_globals]` scalar
    ///      header that `init_vable_indices` assumes (see
    ///      `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS` for the
    ///      canonical 6-scalar layout — line-by-line PyPy parity with
    ///      `interp_jit.py:25-31`); the frame still owns the shadow
    ///      semantically though.
    ///
    /// Callee inline frames (`inline_function_call` allocates a fresh
    /// `PyreSym::new_uninit`) keep both fields at their defaults and
    /// must NOT mirror into the caller's shadow — their
    /// `nlocals + stack_idx` space is the callee's own, not the
    /// caller's, so writing to `NUM_VABLE_SCALARS + reg_idx` in the
    /// shared `TraceCtx` shadow would corrupt the caller's portal
    /// layout. opencoder.py:718 `_list_of_boxes_virtualizable` treats
    /// `self.virtualizable_boxes` as the single source of truth; this
    /// predicate names the set of syms that are allowed to update it.
    pub(crate) fn owns_virtualizable_shadow(&self) -> bool {
        self.is_active_vable_owner || self.bridge_local_oprefs.is_some()
    }

    /// Promote this frame to active virtualizable owner. Wraps the
    /// macro-generated `init_vable_indices` so the per-frame
    /// `is_active_vable_owner` boolean and the u32 OpRef-offset stay in
    /// lock-step. Call this everywhere `init_vable_indices()` was
    /// previously invoked directly.
    pub(crate) fn become_active_vable_owner(&mut self) {
        self.init_vable_indices(crate::virtualizable_gen::FIRST_VABLE_SCALAR_IDX);
        self.is_active_vable_owner = true;
        debug_assert!(
            self.vable_array_base.is_some(),
            "init_vable_indices must seed vable_array_base = Some(...)"
        );
    }

    /// Demote this frame from active virtualizable owner. Used at bridge
    /// setup (`setup_bridge_sym`) where the bridge's inputarg layout
    /// does not have the `[frame, last_instr, pycode, valuestackdepth,
    /// debugdata, lastblock, w_globals]` scalar header that the
    /// loop-portal `init_vable_indices` assumes (canonical 6-scalar
    /// layout in `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`);
    /// subsequent reads consult `bridge_local_oprefs` or fall through
    /// to the heap array via `locals_cells_stack_array_ref`.
    pub(crate) fn clear_active_vable(&mut self) {
        self.vable_array_base = None;
        self.is_active_vable_owner = false;
    }

    #[doc(hidden)]
    pub fn from_test_state(state: TestSymState) -> Self {
        let mut sym = Self::new_uninit(state.frame);
        sym.jitcode = state.jitcode as *const JitCode;
        sym.nlocals = state.nlocals;
        sym.valuestackdepth = state.valuestackdepth;
        sym.locals_cells_stack_array_ref = state.locals_cells_stack_array_ref;
        sym.symbolic_local_types = state.symbolic_local_types;
        sym.symbolic_stack_types = state.symbolic_stack_types;
        sym.registers_r = state.registers_r;
        sym.concrete_stack = state.concrete_stack;
        sym.concrete_namespace = state.concrete_namespace;
        sym.vable_last_instr = state.vable_last_instr;
        sym.vable_pycode = state.vable_pycode;
        sym.vable_valuestackdepth = state.vable_valuestackdepth;
        sym.vable_debugdata = state.vable_debugdata;
        sym.vable_lastblock = state.vable_lastblock;
        sym.vable_w_globals = state.vable_w_globals;
        sym
    }

    #[doc(hidden)]
    pub fn set_test_execution_context(&mut self, execution_context: OpRef) {
        self.execution_context = execution_context;
    }

    /// Initialize symbolic tracking state. Called once when the owning
    /// MetaInterpFrame is pushed (trace.rs for root frame). Callee (inline)
    /// frames set symbolic state manually in perform_call
    /// (trace_opcode.rs:3323-3424) and do NOT call this.
    pub(crate) fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        self.is_function_entry_trace = ctx.header_pc == 0;
        let nlocals = concrete_nlocals(concrete_frame).unwrap_or(0);
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][init-sym] concrete_frame={:#x} nlocals={} vable_base={:?} header_pc={} func_entry={}",
                concrete_frame,
                nlocals,
                self.vable_array_base,
                ctx.header_pc,
                self.is_function_entry_trace
            );
        }
        let valuestackdepth = concrete_stack_depth(concrete_frame).unwrap_or(nlocals);
        let stack_only_depth = valuestackdepth.saturating_sub(nlocals);
        self.nlocals = nlocals;
        self.locals_cells_stack_array_ref = if self.is_active_vable_owner {
            OpRef::NONE
        } else {
            frame_locals_cells_stack_array(ctx, self.frame)
        };
        // RPython pyjitpl.py:74-78 init analogue for the local-slot view
        // of registers_r. The bridge override / vable inputarg / NONE
        // shape is the per-trace seed; subsequent load_local_value /
        // store_local_value updates the per-color slot directly.
        self.registers_r = if let Some(ref overrides) = self.bridge_local_oprefs {
            // resume.py:1042 parity: bridge trace uses OpRefs derived from
            // rebuild_from_resumedata (Box(n) → bridge InputArg OpRef::from_raw(n)).
            let mut locals = overrides.clone();
            locals.resize(nlocals, OpRef::NONE);
            locals
        } else if let Some(base) = self.vable_array_base {
            (0..nlocals)
                .map(|i| OpRef::input_arg_ref(base + i as u32))
                .collect()
        } else {
            vec![OpRef::NONE; nlocals]
        };
        // RPython resume.py:1042 parity: bridge traces enter with the
        // failing guard's saved boxes, NOT with the loop's full
        // virtualizable inputarg layout. Each `bridge_local_oprefs[i]`
        // points at the bridge inputarg slot the rebuilt frame placed in
        // local i, so the type for local i must come from
        // `inputarg_types[bridge_local_oprefs[i].0]` instead of the
        // loop's `vable_array_base + i` indexing (which only applies to
        // loops where every local sits in a fixed virtualizable slot).
        //
        // Loop / function-entry traces still use vable_array_base because
        // their inputarg list is the full vable layout.
        let inputarg_slot_types = if let Some(ref overrides) = self.bridge_local_oprefs {
            let inputarg_types = ctx.inputarg_types();
            let locals: Vec<Type> = (0..nlocals)
                .map(|i| {
                    overrides
                        .get(i)
                        .and_then(|opref| {
                            if opref.is_none() || opref.is_constant() {
                                None
                            } else {
                                inputarg_types.get(opref.raw() as usize).copied()
                            }
                        })
                        .unwrap_or(Type::Ref)
                })
                .collect();
            // resume.py:1042 consume_boxes + pyjitpl.py:2899 parity:
            // guard-failure resume reconstructs the CURRENT frame state,
            // including the live Python stack. `bridge_local_oprefs`
            // overrides only the locals slice; stack slots still follow
            // the virtualizable Ref contract until setup_bridge_sym
            // overwrites the rebuilt tail with the precise resume-data
            // OpRefs.
            Some((locals, vec![Type::Ref; stack_only_depth]))
        } else {
            self.vable_array_base.map(|base| {
                let inputarg_types = ctx.inputarg_types();
                let locals: Vec<Type> = (0..nlocals)
                    .map(|i| {
                        inputarg_types
                            .get(base as usize + i)
                            .copied()
                            .unwrap_or(Type::Ref)
                    })
                    .collect();
                let stack: Vec<Type> = (0..stack_only_depth)
                    .map(|i| {
                        inputarg_types
                            .get(base as usize + nlocals + i)
                            .copied()
                            .unwrap_or(Type::Ref)
                    })
                    .collect();
                (locals, stack)
            })
        };
        if self.is_function_entry_trace {
            // virtualizable.py:86 read_boxes() parity: all array items
            // are GC pointers → Ref. No pre-unboxing at function entry.
            self.symbolic_local_types = vec![Type::Ref; nlocals];
        } else if let Some(ref overrides) = self.bridge_local_types {
            // virtualizable.py:44: bridge locals keep the Ref contract of
            // the virtualizable array. `bridge_local_types` is always
            // all-Ref (populated by setup_bridge_sym); the override is
            // retained only for length parity with bridge_local_oprefs.
            let mut types = overrides.clone();
            types.resize(nlocals, Type::Ref);
            self.symbolic_local_types = types;
        } else if let Some((ref local_types, _)) = inputarg_slot_types {
            // warmstate.py:73 wrap(): inputargs resolved from the JIT
            // contract retain whatever declared type the input signature
            // records. For virtualizable array inputargs this is Ref.
            self.symbolic_local_types = local_types.clone();
        } else if self.symbolic_local_types.len() != nlocals {
            self.symbolic_local_types = concrete_slot_types(concrete_frame, nlocals, nlocals);
        }
        // Seed the stack portion of `registers_r`
        // directly. `registers_r` is the unified abstract register
        // file — locals occupy `[..nlocals]` and stack slots occupy
        // `[nlocals..nlocals + stack_only_depth]` (RPython
        // `pyjitpl.py:70-78` MIFrame parity).
        let stack_seed: Vec<OpRef> = if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef::input_arg_ref(stack_base + i as u32))
                .collect()
        } else if let Some(ref bridge_stack) = self.bridge_stack_oprefs {
            // #124: a bridge resumes from a guard whose live operand stack
            // setup_bridge_sym already resolved into these OpRefs. The local
            // override (above) plus the comment at the bridge_local_oprefs
            // branch assume setup_bridge_sym overwrites the rebuilt tail,
            // but pyre's launcher runs setup_bridge_sym BEFORE this; so
            // preserve the kept temps here rather than reset them to NONE.
            let mut seed = bridge_stack.clone();
            seed.resize(stack_only_depth, OpRef::NONE);
            seed
        } else {
            vec![OpRef::NONE; stack_only_depth]
        };
        self.registers_r.extend(stack_seed.iter().copied());
        if let Some((_, ref stack_types)) = inputarg_slot_types {
            self.symbolic_stack_types = stack_types.clone();
        } else if self.symbolic_stack_types.len() != stack_only_depth {
            self.symbolic_stack_types =
                concrete_slot_types(concrete_frame, nlocals, valuestackdepth)
                    .into_iter()
                    .skip(nlocals)
                    .collect();
        }
        self.pending_next_instr = None;
        self.valuestackdepth = valuestackdepth;
        // MIFrame concrete Box tracking: populate concrete value arrays
        // from the concrete frame snapshot (RPython MIFrame.setup_call parity).
        // Use concrete_value_from_slot to distinguish "real null pointer"
        // (Ref(PY_NULL)) from "untracked" (ConcreteValue::Null).
        self.concrete_locals = (0..nlocals)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        self.concrete_stack = (0..stack_only_depth)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, nlocals + i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        // Extract frame metadata pointers for use without concrete_frame
        if concrete_frame != 0 {
            let frame = unsafe { &*(concrete_frame as *const pyre_interpreter::pyframe::PyFrame) };
            self.jitcode = jitcode_for(frame.pycode);
            self.concrete_namespace = frame.w_globals;
            self.concrete_execution_context = frame.execution_context;
            self.concrete_vable_ptr = concrete_frame as *mut u8;
            // pyjitpl.py:74-90 MIFrame.setup parity for the per-kind banks
            // (including pyjitpl.py:97-119 copy_constants).
            self.setup_kind_register_banks(ctx);
        }
        // pyjitpl.py:3458-3462 / virtualizable.py:86-99 read_boxes parity:
        // seed the tracing-time `virtualizable_boxes` cache with the
        // InputArg OpRefs that correspond to the portal's virtualizable
        // layout (scalar fields followed by array items, with the
        // virtualizable identity appended at `boxes[-1]`). Without this
        // seeding, `vable_setarrayitem_indexed` / `vable_getarrayitem_*`
        // fall through to raw SetarrayitemGc / GetarrayitemGc ops, and
        // `close_loop_args_at`'s `set_virtualizable_box_at` mirror
        // becomes a no-op — the very reason `MIFrame::store_local_value`
        // cannot route through the standard vable path today.
        if let Some(base) = self.vable_array_base {
            // virtualizable.py:86-99 read_boxes iterates `len(lst)` — the
            // heap-side `locals_cells_stack_w` length, which is
            // `num_locals + num_cells + max_stack` (pyframe.py:107-110).
            // Match that shape so every interpreter-visible slot has a
            // tracing-time mirror. The live prefix [0, nlocals+stack_only_depth)
            // references the recorder's InputArg stream; reserved stack slots
            // beyond the current stack pointer are NULL on the PyFrame heap
            // (alloc_fixed_array_with_header(..., PY_NULL)) and the helper pads
            // them with a shared const-NULL OpRef.
            let num_vable_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;
            let live_prefix = nlocals + stack_only_depth;
            let array_len = concrete_frame_array_len(concrete_frame).unwrap_or(live_prefix);
            // pyjitpl.py:3302 initialize_virtualizable parity: the concrete
            // half of virtualizable_boxes at portal entry comes from a live
            // heap read (vinfo.read_boxes(cpu, virtualizable, 0)). There is
            // no resume-data stream at root-trace start.
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            // Static fields inputargs at FIRST_VABLE_SCALAR_IDX..+NUM_VABLE_SCALARS.
            // virtualizable_boxes carries vable static fields only — non-vable
            // extra reds (e.g. `ec`) sit between frame and vable scalars in the
            // inputarg space (`pyjitpl.py:2957 redboxes` then `:2964
            // + virtualizable_boxes`) and never enter the shadow. Per-slot
            // type follows `info.static_fields[i].field_type` so the OpRef
            // variant matches RPython's BoxInt/BoxRef inputarg classes.
            let first = crate::virtualizable_gen::FIRST_VABLE_SCALAR_IDX;
            let scalar_oprefs: Vec<OpRef> = (0..num_vable_scalars)
                .map(|i| {
                    let pos = first + i as u32;
                    let tp = info.static_fields[i].field_type;
                    OpRef::input_arg_typed(pos, tp)
                })
                .collect();
            // Array items inputargs at base..base + live_prefix. Items are
            // W_Root (Ref) per `array_item_type = Ref` in virtualizable_gen.
            let array_items: Vec<OpRef> = (0..live_prefix)
                .map(|i| OpRef::input_arg_ref(base + i as u32))
                .collect();
            // SYM_FRAME_IDX is the virtualizable identity (Ref).
            let vable_ref = OpRef::input_arg_ref(crate::virtualizable_gen::SYM_FRAME_IDX);
            let array_lengths = [array_len];
            let (input_values, vable_ref_value) = if concrete_frame != 0 {
                let (static_boxes, array_boxes) =
                    unsafe { info.read_all_boxes(concrete_frame as *const u8, &array_lengths) };
                let mut values = Vec::with_capacity(num_vable_scalars + array_len);
                for (i, bits) in static_boxes.iter().enumerate() {
                    values.push(value_for_slot(info.static_fields[i].field_type, *bits));
                }
                for (a, items) in array_boxes.iter().enumerate() {
                    let item_ty = info.array_fields[a].item_type;
                    for (item_idx, bits) in items.iter().enumerate() {
                        // Seed a tagged-immediate local as the heap `W_IntObject`
                        // it stands for, so the recorded body reads it through the
                        // flag-false `GuardClass`+`GetfieldGcPure` arm rather than
                        // the tagged `CastPtrToInt`+`IntAnd` arm. The runtime
                        // (`execute_assembler`) converts the live frame's local
                        // slots to heap boxes before the compiled loop reads them;
                        // the tracer works on a `snapshot_for_tracing` copy whose
                        // slots are still tagged, so the same conversion must run
                        // here at record time. Only the locals region is converted
                        // (cells/free vars and stack temps stay untouched).
                        if pyre_object::tagged_int::CAN_BE_TAGGED
                            && item_ty == Type::Ref
                            && item_idx < nlocals
                        {
                            let obj = *bits as pyre_object::PyObjectRef;
                            if !obj.is_null() && pyre_object::tagged_int::is_tagged_int(obj) {
                                let value = pyre_object::tagged_int::untag_int(obj);
                                let heap = pyre_object::intobject::w_int_new_unique(value);
                                values.push(majit_ir::Value::Ref(majit_ir::GcRef(heap as usize)));
                                continue;
                            }
                        }
                        values.push(value_for_slot(item_ty, *bits));
                    }
                }
                while values.len() < num_vable_scalars + array_len {
                    values.push(majit_ir::Value::Ref(majit_ir::GcRef::NULL));
                }
                // gap 10 slice 2b: bake the virtualizable identity against the
                // LIVE interpreter frame (the frame the compiled loop runs on),
                // not the discarded `snapshot_for_tracing` copy.  At root entry
                // the snapshot is a fresh copy so its field VALUES equal the
                // live frame's (read_all_boxes above); only the identity ADDRESS
                // must be the live one, matching the SYM_FRAME_IDX input arg's
                // runtime value (the live frame supplied by extract_live_values).
                // Falls back to the snapshot address when no live frame was
                // threaded (unit-test / init-before-run path).
                let vable_identity_addr = if self.live_vable_frame_addr != 0 {
                    self.live_vable_frame_addr
                } else {
                    concrete_frame
                };
                (
                    values,
                    majit_ir::Value::Ref(majit_ir::GcRef(vable_identity_addr)),
                )
            } else {
                (Vec::new(), majit_ir::Value::Void)
            };
            crate::state::seed_virtualizable_boxes(
                ctx,
                vable_ref,
                vable_ref_value,
                &scalar_oprefs,
                &array_items,
                array_len,
                &input_values,
                concrete_frame as *const u8,
            );
        }
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at_opt(&self, abs_idx: usize) -> Option<ConcreteValue> {
        if abs_idx < self.nlocals {
            self.concrete_locals.get(abs_idx).copied()
        } else {
            let stack_idx = abs_idx - self.nlocals;
            self.concrete_stack.get(stack_idx).copied()
        }
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at(&self, abs_idx: usize) -> ConcreteValue {
        self.concrete_value_at_opt(abs_idx)
            .unwrap_or(ConcreteValue::Null)
    }

    /// #143 frame-advance: number of local slots whose post-trace concrete
    /// state is tracked (`min(concrete_locals.len(), nlocals)`).
    pub fn tracked_local_count(&self) -> usize {
        self.concrete_locals.len().min(self.nlocals)
    }

    /// #143 frame-advance: materialize local slot `i`'s post-trace concrete
    /// value as a boxed `PyObjectRef` (lazy `Int`/`Float` allocated via
    /// `w_int_new`/`w_float_new`; `Ref` returned as-is). `None` for a `Null`
    /// (dead/uninitialized) slot. The caller must root the returned box
    /// before the next allocation.
    pub fn materialize_local(&self, i: usize) -> Option<PyObjectRef> {
        let cv = *self.concrete_locals.get(i)?;
        if cv.is_null() {
            None
        } else {
            Some(cv.to_pyobj())
        }
    }
}

/// pyjitpl.py:1789-1814 opimpl_virtual_ref parity.
/// Creates a concrete JitVirtualRef via virtual_ref_during_tracing(),
/// records VIRTUAL_REF(box, cindex), and pushes
/// [virtualbox, vrefbox] onto virtualref_boxes.
///
/// Called from metainterp push_inline_frame (executioncontext.enter parity).
pub(crate) fn opimpl_virtual_ref(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    virtual_obj: OpRef,
    virtual_obj_ptr: usize,
) -> OpRef {
    // pyjitpl.py:1804: virtual_ref_during_tracing(virtual_obj)
    let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
    let vref_ptr = vref_info.virtual_ref_during_tracing(virtual_obj_ptr as *mut u8);
    // pyjitpl.py:1805: cindex = ConstInt(len(virtualref_boxes) // 2)
    let cindex = ctx.const_int((sym.virtualref_boxes.len() / 2) as i64);
    // pyjitpl.py:1806: record VIRTUAL_REF(box, cindex)
    let vref = ctx.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex]);
    // pyjitpl.py:1807: heapcache.new(resbox)
    ctx.heap_cache_mut().new_box(vref);
    // pyjitpl.py:1814: virtualref_boxes += [virtualbox, vrefbox]
    sym.virtualref_boxes.push((virtual_obj, virtual_obj_ptr));
    sym.virtualref_boxes.push((vref, vref_ptr as usize));
    vref
}

/// pyjitpl.py:1819-1831 opimpl_virtual_ref_finish parity.
/// Pops vrefbox and lastbox from virtualref_boxes (LIFO),
/// asserts `box == lastbox`, records VIRTUAL_REF_FINISH if still virtual.
///
/// Called from metainterp finishframe_inline/exception (executioncontext.leave parity).
pub(crate) fn opimpl_virtual_ref_finish(ctx: &mut TraceCtx, sym: &mut PyreSym, virtual_obj: OpRef) {
    if sym.virtualref_boxes.len() < 2 {
        return;
    }
    // pyjitpl.py:1821: vrefbox = virtualref_boxes.pop()
    let (vref_opref, vref_ptr) = sym.virtualref_boxes.pop().unwrap();
    // pyjitpl.py:1822: lastbox = virtualref_boxes.pop()
    let (lastbox_opref, _lastbox_ptr) = sym.virtualref_boxes.pop().unwrap();
    // pyjitpl.py:1823: assert box.getref_base() == lastbox.getref_base()
    debug_assert_eq!(
        virtual_obj, lastbox_opref,
        "opimpl_virtual_ref_finish: leaving frame box != top virtualref box"
    );
    // pyjitpl.py:1831: if is_virtual_ref(vref) → record VIRTUAL_REF_FINISH
    let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
    let is_vref = vref_ptr != 0 && unsafe { vref_info.is_virtual_ref(vref_ptr as *const u8) };
    if is_vref {
        // pyjitpl.py:1832: VIRTUAL_REF_FINISH(vrefbox, nullbox)
        let null = ctx.const_ref(0);
        let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref_opref, null]);
    }
}

impl PyreJitState {
    /// Canonical PyPy portal driver layout from `interp_jit.py:67-74`.
    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_driver_descriptor() -> JitDriverStaticData {
        let mut descriptor = JitDriverStaticData::with_virtualizable(
            vec![
                ("next_instr", Type::Int),
                ("is_being_profiled", Type::Int),
                ("pycode", Type::Ref),
            ],
            vec![("frame", Type::Ref), ("ec", Type::Ref)],
            Some("frame"),
        );
        descriptor.is_recursive = true;
        descriptor
    }

    fn execution_context_as_usize(&self) -> usize {
        let Some(frame_ptr) = self.frame_ptr() else {
            return 0;
        };
        unsafe { (*(frame_ptr as *const PyFrame)).execution_context as usize }
    }

    fn expanded_virtualizable_live_values_with_extra_reds(
        &self,
        meta: &PyreMeta,
        extra_reds: &[Value],
    ) -> Vec<Value> {
        let base = crate::virtualizable_gen::virt_extract_live_values(
            self.frame,
            self.execution_context_as_usize(),
            self.last_instr_as_usize(),
            self.pycode_as_usize(),
            self.valuestackdepth(),
            self.debugdata_as_usize(),
            self.lastblock_as_usize(),
            self.w_globals_as_usize(),
            meta.num_locals,
            meta.valuestackdepth,
            |i| self.local_at(i).unwrap_or(PY_NULL) as usize,
            |i| self.stack_at(i).unwrap_or(PY_NULL) as usize,
        );
        let Some((&frame, tail)) = base.split_first() else {
            return extra_reds.to_vec();
        };
        let mut values = Vec::with_capacity(base.len() + extra_reds.len());
        values.push(frame);
        values.extend_from_slice(extra_reds);
        values.extend_from_slice(tail);
        values
    }

    fn expanded_virtualizable_live_value_types_with_extra_reds(
        meta: &PyreMeta,
        extra_red_types: &[Type],
    ) -> Vec<Type> {
        let base = crate::virtualizable_gen::virt_live_value_types(meta.slot_types.len());
        let Some((&frame_tp, tail)) = base.split_first() else {
            return extra_red_types.to_vec();
        };
        let mut types = Vec::with_capacity(base.len() + extra_red_types.len());
        types.push(frame_tp);
        types.extend_from_slice(extra_red_types);
        types.extend_from_slice(tail);
        types
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_live_values_with_ec(&self, meta: &PyreMeta) -> Vec<Value> {
        let _ = meta;
        vec![
            Value::Ref(majit_ir::GcRef(self.frame)),
            Value::Ref(majit_ir::GcRef(self.execution_context_as_usize())),
        ]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_live_value_types_with_ec(meta: &PyreMeta) -> Vec<Type> {
        let _ = meta;
        vec![Type::Ref, Type::Ref]
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_collect_jump_args(sym: &PyreSym) -> Vec<OpRef> {
        let base = sym.vable_collect_jump_args();
        let Some((&frame, tail)) = base.split_first() else {
            unreachable!(
                "vable_collect_jump_args returned empty — virtualizable macro \
                 did not emit frame_field push (see majit-macros/.../derive.rs:381)"
            );
        };
        let mut args = Vec::with_capacity(base.len() + 1);
        args.push(frame);
        args.push(sym.execution_context);
        args.extend_from_slice(tail);
        args
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_collect_typed_jump_args(sym: &PyreSym) -> Vec<(OpRef, Type)> {
        let base = sym.vable_collect_typed_jump_args();
        let Some(&(frame, frame_tp)) = base.first() else {
            unreachable!(
                "vable_collect_typed_jump_args returned empty — virtualizable \
                 macro did not emit frame_field push \
                 (see majit-macros/.../derive.rs:389)"
            );
        };
        let mut args = Vec::with_capacity(base.len() + 1);
        args.push((frame, frame_tp));
        args.push((sym.execution_context, Type::Ref));
        args.extend_from_slice(&base[1..]);
        args
    }

    #[cfg_attr(not(test), allow(dead_code))]
    fn pypyjit_create_sym(meta: &PyreMeta, _header_pc: usize) -> PyreSym {
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_typed(0, Type::Ref));
        sym.execution_context = OpRef::input_arg_typed(1, Type::Ref);
        // `become_active_vable_owner` calls `init_vable_indices(FIRST_VABLE_SCALAR_IDX)`
        // where `FIRST_VABLE_SCALAR_IDX = 1 + NUM_EXTRA_REDS` already accounts for
        // the `ec` extra red. Do not re-shift the slot indices here — doing so
        // would double-count, leaving e.g. `vable_pycode` at the valuestackdepth
        // slot (Type::Int) instead of the pycode slot (Type::Ref) and tripping
        // the resume-time type assertion in `value_to_static_vable_bits`.
        sym.become_active_vable_owner();
        sym.nlocals = meta.num_locals;
        sym.valuestackdepth = meta.valuestackdepth;
        sym.symbolic_local_types = vec![Type::Ref; meta.num_locals];
        sym.symbolic_stack_types = vec![Type::Ref; meta.vable_stack_only_depth()];
        let stack_only = meta.vable_stack_only_depth();
        sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
        sym
    }

    fn restore_expanded_virtualizable_values_with_extra_reds(
        &mut self,
        meta: &PyreMeta,
        values: &[Value],
        extra_reds: usize,
    ) {
        let Some(frame) = values.first() else {
            return;
        };
        self.frame = value_to_usize(frame);
        if values.len() <= 1 + extra_reds {
            return;
        }

        if meta.has_virtualizable {
            // next_instr is already synced to the PyFrame heap by the
            // compiled code's virtualizable sync before JUMP.
            self.set_valuestackdepth(meta.valuestackdepth);
            let nlocals = self.local_count();
            let stack_only = meta.valuestackdepth.saturating_sub(nlocals);
            let _ = extra_reds; // already folded into NUM_SCALAR_INPUTARGS
            let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            for local_idx in 0..nlocals {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ =
                        self.set_local_at(local_idx, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
        } else {
            let nlocals = self.local_count();
            let stack_only_depth = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1 + extra_reds;
            for local_idx in 0..nlocals {
                let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                let _ = self.set_local_at(
                    local_idx,
                    boxed_slot_value_for_type(slot_type, &values[idx]),
                );
                idx += 1;
            }
            for i in 0..stack_only_depth {
                let slot_type = meta
                    .slot_types
                    .get(nlocals + i)
                    .copied()
                    .unwrap_or(Type::Ref);
                let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, &values[idx]));
                idx += 1;
            }
            self.set_valuestackdepth(meta.valuestackdepth);
        }
    }

    /// virtualizable.py:126-137 write_from_resume_data_partial parity.
    ///
    /// Restore virtualizable frame state from raw output buffer values.
    /// interp_jit.py:25: locals_cells_stack_w ARRAYITEMTYPE = GCREF.
    /// All raw values are PyObjectRef pointers — write directly.
    pub fn restore_virtualizable_from_raw(&mut self, raw_values: &[i64]) -> bool {
        if raw_values.is_empty() {
            return false;
        }
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, raw_values);

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for local_idx in 0..nlocals {
            if idx < raw_values.len() {
                let _ = self.set_local_at(local_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        for stack_idx in 0..stack_only {
            if idx < raw_values.len() {
                let _ = self.set_stack_at(stack_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        true
    }

    /// Returns true if the optimizer virtualizable mechanism is active.
    fn has_virtualizable_info(&self) -> bool {
        // pyre always uses virtualizable (JitDriverStaticData::with_virtualizable)
        true
    }

    fn frame_ptr(&self) -> Option<*mut u8> {
        (self.frame != 0).then_some(self.frame as *mut u8)
    }

    fn frame_array(&self, offset: usize) -> Option<&pyre_object::FixedObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        let arr_ptr =
            unsafe { *(frame_ptr.add(offset) as *const *const pyre_object::FixedObjectArray) };
        Some(unsafe { &*arr_ptr })
    }

    fn frame_array_mut(&mut self, offset: usize) -> Option<&mut pyre_object::FixedObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        let arr_ptr =
            unsafe { *(frame_ptr.add(offset) as *const *mut pyre_object::FixedObjectArray) };
        Some(unsafe { &mut *arr_ptr })
    }

    fn read_frame_usize(&self, offset: usize) -> Option<usize> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { *(frame_ptr.add(offset) as *const usize) })
    }

    fn write_frame_usize(&mut self, offset: usize, value: usize) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        unsafe {
            *(frame_ptr.add(offset) as *mut usize) = value;
        }
        true
    }

    fn locals_cells_stack_array(&self) -> Option<&pyre_object::FixedObjectArray> {
        self.frame_array(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn locals_cells_stack_array_mut(&mut self) -> Option<&mut pyre_object::FixedObjectArray> {
        self.frame_array_mut(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn namespace_len(&self) -> usize {
        let Some(frame_ptr) = self.frame_ptr() else {
            return 0;
        };
        let w_globals = unsafe {
            *(frame_ptr.add(PYFRAME_W_GLOBALS_OFFSET) as *const pyre_object::PyObjectRef)
        };
        if w_globals.is_null() {
            return 0;
        }
        // `dictmultiobject.py:107-109 W_DictMultiObject.length`. The common
        // module-dict case reads `ModuleDictStorage` directly (O(1), no
        // `dict_storage_proxy` reconciliation) — this guard is on the
        // per-portal-entry path. Plain dict globals (exec/eval) fall back to
        // the polymorphic strategy length.
        unsafe {
            pyre_object::dictmultiobject::module_dict_storage_len(w_globals)
                .unwrap_or_else(|| pyre_object::dictmultiobject::w_dict_len(w_globals))
        }
    }

    fn restore_single_frame(&mut self, meta: &PyreMeta, values: &[i64]) {
        let Some(&frame) = values.first() else {
            return;
        };
        self.frame = frame as usize;
        if values.len() == 1 {
            return;
        }
        if meta.has_virtualizable {
            self.restore_virtualizable_i64(values);
        } else {
            let nlocals = self.local_count();
            let stack_only = self.valuestackdepth().saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                if idx < values.len() {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ = self
                        .set_local_at(local_idx, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if idx < values.len() {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
        }
    }

    pub fn local_at(&self, idx: usize) -> Option<PyObjectRef> {
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(idx).copied())
    }

    /// Number of local variable slots.
    pub fn local_count(&self) -> usize {
        concrete_nlocals(self.frame).unwrap_or(0)
    }

    pub fn set_local_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(idx) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Read a stack value at stack-relative index `idx` (0-based from stack bottom).
    pub fn stack_at(&self, idx: usize) -> Option<PyObjectRef> {
        let nlocals = self.local_count();
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(nlocals + idx).copied())
    }

    /// Total capacity of the unified array.
    pub fn array_capacity(&self) -> usize {
        self.locals_cells_stack_array()
            .map(pyre_object::FixedObjectArray::len)
            .unwrap_or(0)
    }

    /// Set a stack value at stack-relative index `idx`.
    pub fn set_stack_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let nlocals = self.local_count();
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(nlocals + idx) else {
            return false;
        };
        *slot = value;
        true
    }

    // ── Heap accessors: single source of truth (RPython parity) ──
    // RPython's virtualizable IS the heap object — getattr/setattr go
    // directly to the heap.  These accessors do the same via frame_ptr.

    pub fn last_instr_as_usize(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe { (*(frame_ptr as *const PyFrame)).last_instr as usize }
    }

    pub fn set_last_instr(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (*(frame_ptr as *mut PyFrame)).last_instr = value as isize;
        }
    }

    pub fn next_instr(&self) -> usize {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe { (&*(frame_ptr as *const PyFrame)).next_instr() }
    }

    pub fn set_next_instr(&mut self, value: usize) {
        let frame_ptr = self
            .frame_ptr()
            .expect("PyreJitState.frame must point to a valid PyFrame");
        unsafe {
            (&mut *(frame_ptr as *mut PyFrame)).set_last_instr_from_next_instr(value);
        }
    }

    pub fn valuestackdepth(&self) -> usize {
        self.read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    pub fn set_valuestackdepth(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Null the locals_cells_stack slots at and above `depth`, the
    /// fresh-frame parity clear (`write_from_resume_data_partial` does not
    /// trim).  Used after a vsd correction so a GC scan before the next
    /// push does not observe a stale operand pointer above the live depth.
    pub fn clear_stack_above(&mut self, depth: usize) {
        if let Some(arr) = self.locals_cells_stack_array_mut() {
            let slice = arr.as_mut_slice();
            for slot in slice.iter_mut().skip(depth) {
                *slot = pyre_object::PY_NULL;
            }
        }
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn pycode_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_PYCODE_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Read the w_globals pointer from the heap frame.
    pub fn w_globals_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_W_GLOBALS_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Read the execution context pointer from the heap frame.
    ///
    /// `interp_jit.py:67 reds = ['frame', 'ec']`: ec is a non-vable red
    /// inputarg in RPython. pyre's PyFrame carries it inline at
    /// `execution_context`, so this accessor derefs the heap; from the
    /// macro-generated layout's perspective ec sits at SYM_EC_IDX between
    /// the frame pointer and the vable scalar block (`pyjitpl.py:2957
    /// redboxes` then `:2964 + virtualizable_boxes`).
    pub fn ec_as_usize(&self) -> usize {
        self.read_frame_usize(crate::frame_layout::PYFRAME_EXECUTION_CONTEXT_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Write the execution context pointer into the heap frame.
    ///
    /// Called by `virt_restore_scalars` when reconstructing red inputargs
    /// from a guard-failure resume vector.
    pub fn set_ec(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(crate::frame_layout::PYFRAME_EXECUTION_CONTEXT_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn code_as_usize(&self) -> usize {
        self.pycode_as_usize()
    }

    /// Read the namespace pointer from the heap frame.
    pub fn namespace_as_usize(&self) -> usize {
        self.w_globals_as_usize()
    }

    /// Write the pycode pointer to the heap frame.
    /// virtualizable.py:101-107 write_boxes: ALL static fields written.
    pub fn set_pycode(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_PYCODE_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Write the w_globals pointer to the heap frame.
    pub fn set_w_globals(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_W_GLOBALS_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Compatibility wrapper for older callers that still speak in
    /// terms of `code` / `namespace`.
    pub fn set_code(&mut self, value: usize) {
        self.set_pycode(value);
    }

    pub fn set_namespace(&mut self, value: usize) {
        self.set_w_globals(value);
    }

    /// pyframe.py:82 debugdata — read from heap frame.
    pub fn debugdata_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_DEBUGDATA_OFFSET).unwrap_or(0)
    }

    /// pyframe.py:82 debugdata — write to heap frame.
    pub fn set_debugdata(&mut self, value: usize) {
        let _ = self.write_frame_usize(PYFRAME_DEBUGDATA_OFFSET, value);
    }

    /// pyframe.py:86 lastblock — read from heap frame.
    pub fn lastblock_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_LASTBLOCK_OFFSET).unwrap_or(0)
    }

    /// pyframe.py:86 lastblock — write to heap frame.
    pub fn set_lastblock(&mut self, value: usize) {
        let _ = self.write_frame_usize(PYFRAME_LASTBLOCK_OFFSET, value);
    }

    /// Validate that the frame pointer is usable (fields readable, array present).
    fn validate_frame(&self) -> bool {
        self.frame_ptr().is_some()
            && self
                .read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
                .is_some()
            && self.locals_cells_stack_array().is_some()
    }

    /// virtualizable.py:126-137 write_from_resume_data_partial parity.
    ///
    /// Restores virtualizable array slots from the fail_args layout:
    ///   [frame, scalars..., active_locals..., active_stack...]
    ///
    /// interp_jit.py:25 declares locals_cells_stack_w as a single
    /// virtualizable array with uniform ARRAYITEMTYPE = GCREF.
    /// resume.py:1408 calls write_from_resume_data_partial which loops
    /// over the array calling reader.load_next_value_of_type(GCREF),
    /// i.e. reader.next_ref() — every slot is a GCREF pointer.
    ///
    /// The raw i64 values here are already PyObjectRef pointers from
    /// the backend's Ref register bank. Write them directly without
    /// per-slot type dispatch.
    fn restore_virtualizable_i64(&mut self, values: &[i64]) {
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, values);

        // virtualizable.py:134-137:
        //   for ARRAYITEMTYPE, fieldname in unroll_array_fields:
        //       lst = getattr(virtualizable, fieldname)
        //       for j in range(len(lst)):
        //           lst[j] = reader.load_next_value_of_type(ARRAYITEMTYPE)
        // ARRAYITEMTYPE is always GCREF for locals_cells_stack_w.
        let nlocals = self.local_count();
        for i in 0..nlocals {
            if idx < values.len() {
                let _ = self.set_local_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }

        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for i in 0..stack_only {
            if idx < values.len() {
                let _ = self.set_stack_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }
    }

    fn import_virtualizable_state(
        &mut self,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        // virtualizable.py:126-137 write_from_resume_data_partial parity:
        // write ALL static fields to heap via VirtualizableInfo.
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        if !self.virt_import_static_boxes(&info, static_boxes) {
            return false;
        }

        // virtualizable.py:134-137: write array items to heap.
        // Validate array structure matches VirtualizableInfo.
        if array_boxes.len() != info.array_fields.len() {
            return false;
        }
        let Some(unified) = array_boxes.first() else {
            return info.array_fields.is_empty();
        };
        let Some(frame_arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        if frame_arr.len() != unified.len() {
            return false;
        }
        for (dst, &src) in frame_arr.as_mut_slice().iter_mut().zip(unified) {
            *dst = src as PyObjectRef;
        }
        true
    }

    fn export_virtualizable_state(&self) -> (Vec<i64>, Vec<Vec<i64>>) {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        self.virt_export_all(&info)
    }

    pub fn sync_from_virtualizable(&mut self, info: &VirtualizableInfo) -> bool {
        let _ = info;
        // Heap IS the source of truth. Just validate the frame is usable.
        self.validate_frame()
    }

    pub fn sync_to_virtualizable(&self, info: &VirtualizableInfo) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        // Heap is the single source of truth — no state-backed fields to
        // flush.  Only the vable_token needs resetting (virtualizable.py:218
        // force_now: set vable_token to TOKEN_NONE).
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
        true
    }
}

/// resume.py:945-956 getvirtual_ptr parity, trace-time variant.
///
/// `materialize_virtual_from_rd` (eval.rs) does the same job at *runtime*
/// for the blackhole resume path: walks the `RdVirtualInfo` for `vidx`
/// and allocates a real heap object. This function does the same walk
/// at *trace* time, emitting `NEW_WITH_VTABLE` + `SETFIELD_GC` ops into
/// the bridge's trace via `ctx`. Returns the OpRef of the materialized
/// virtual.
///
/// Mirrors RPython's `ResumeDataBoxReader.consume_boxes` →
/// `rd_virtuals[i].allocate(decoder, i)` where `decoder.allocate_with_vtable`
/// is `metainterp.execute_new_with_vtable` (resume.py:1111-1112). The
/// recorded ops appear at the start of the bridge trace, before any
/// python interpreter opcodes are recorded — so when the bridge tracer
/// encounters the first `LOAD_FAST` of a previously-virtual local, it
/// sees the materialized OpRef in `bridge_local_oprefs` instead of
/// falling through to a stale vable-array read.
///
/// resume.py:1143-1188 shared oopspec-call emitter for the four
/// concat/slice materializers. Looks up the call info via
/// `ctx.callinfocollection` and emits a `CALL_R(func, args...)` with the
/// matching calldescr. Panics if `callinfocollection` is not attached to
/// the TraceCtx (should never happen once pyjitpl wiring is complete).
fn emit_stroruni_oopspec_call(
    ctx: &mut majit_metainterp::TraceCtx,
    oopspec: majit_ir::effectinfo::OopSpecIndex,
    args: &[OpRef],
) -> OpRef {
    let cic = ctx
        .callinfocollection
        .as_ref()
        .expect(
            "TraceCtx.callinfocollection missing — bridge-virtual VStr/VUni \
             Concat/Slice materialization requires pyjitpl to populate it \
             (resume.py:1143-1188)",
        )
        .clone();
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr.expect("callinfo_for_oopspec missing entry for VStr/VUni oopspec");
    let func_const = ctx.const_int(func as i64);
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(func_const);
    call_args.extend_from_slice(args);
    ctx.record_op_with_descr(majit_ir::OpCode::CallR, &call_args, calldescr.clone())
}

struct BridgeVirtualCache {
    virtuals_ptr_cache: Vec<Option<OpRef>>,
    virtuals_int_cache: Vec<Option<OpRef>>,
    /// resume.py:882 virtual_ptr_cache concrete parity: raw GcRef
    /// allocation for each virtual, unified with the symbolic OpRef cache
    /// above (RPython's VirtualCache stores both in one object).
    concrete_ptr_cache: Vec<Option<majit_ir::GcRef>>,
    /// resume.py:883 virtual_int_cache concrete parity.  Raw buffers and
    /// slices are INT virtuals, not GC refs.
    concrete_int_cache: Vec<Option<i64>>,
}

impl BridgeVirtualCache {
    fn new(size: usize) -> Self {
        Self {
            virtuals_ptr_cache: vec![None; size],
            virtuals_int_cache: vec![None; size],
            concrete_ptr_cache: vec![None; size],
            concrete_int_cache: vec![None; size],
        }
    }

    fn get_any(&self, i: usize) -> Option<OpRef> {
        self.virtuals_ptr_cache
            .get(i)
            .copied()
            .flatten()
            .or_else(|| self.virtuals_int_cache.get(i).copied().flatten())
    }

    fn set_ptr(&mut self, i: usize, v: OpRef) {
        self.virtuals_ptr_cache[i] = Some(v);
    }

    fn set_int(&mut self, i: usize, v: OpRef) {
        self.virtuals_int_cache[i] = Some(v);
    }

    fn get_concrete_ptr(&self, i: usize) -> Option<majit_ir::GcRef> {
        self.concrete_ptr_cache.get(i).copied().flatten()
    }

    fn set_concrete_ptr(&mut self, i: usize, v: majit_ir::GcRef) {
        if i < self.concrete_ptr_cache.len() {
            self.concrete_ptr_cache[i] = Some(v);
        }
    }

    fn get_concrete_int(&self, i: usize) -> Option<i64> {
        self.concrete_int_cache.get(i).copied().flatten()
    }

    fn set_concrete_int(&mut self, i: usize, v: i64) {
        if i < self.concrete_int_cache.len() {
            self.concrete_int_cache[i] = Some(v);
        }
    }
}

/// resume.py:1245-1264 unified decode_box parity.
/// Returns `(OpRef, Value)`: symbolic OpRef for trace recording, concrete
/// Value for shadow slots / continue_tracing. Replaces the separate
/// `resolve()` + `decode_concrete()` closures so both paths always execute
/// together — no drift between symbolic and concrete materialization.
/// Decode one inlined-callee resume frame
/// (`resume_data.frames[i]`, `i >= 1`) into a [`ReconstructRecipe`] for the
/// multi-frame bridge carrier. Mirrors the per-bank `consume_boxes` decode
/// `setup_bridge_sym` runs for the portal frame (resume.py:1054) but writes a
/// fresh recipe instead of the root `sym`, and emits NO trace IR (the
/// forward-call guard_value / callee-frame helper / no-exception guard belong
/// to a call with a LIVE caller; a reconstructed suspended frame has none).
///
/// The decoded bank vectors are indexed by pyre's semantic register index
/// (Python-bytecode `locals_cells_stack_w` position, not an RPython color),
/// so they align with `write_stack_slot`'s `nlocals + stack_idx`. The
/// assembly into a `PyFrame` + `PyreSym` is deferred to `trace_bytecode`
/// (the carrier drain), where the root concrete frame's globals/EC are
/// available and the rebuilt locals are GC-rooted immediately.
///
/// Returns `None` (→ abort the multi-frame path, fall back to the
/// single-frame bridge) when the callee cannot be faithfully rebuilt:
///   - `pc` is the no-snapshot sentinel (`< 0`),
///   - `jitcode_index` does not resolve to a code object,
///   - the callee has cell or free variables — `locals_cells_stack_w` then
///     carries cell objects whose contents are not in the resume liveness
///     stream, so a dead-frame rebuild would seed null cells and LOAD_DEREF
///     would raise `NameError`,
///   - the liveness enumeration count disagrees with the encoded section.
/// `rd_virtuals[vidx]` with negative-index resolution already applied by the
/// caller. Returns the `RdVirtualInfo` behind the `Rc`, or `None` when the
/// index is out of range or no virtuals were decoded.
fn rd_virtual_at(
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    vidx: usize,
) -> Option<&majit_ir::RdVirtualInfo> {
    rd_virtuals.and_then(|v| v.get(vidx)).map(|rc| &**rc)
}

/// P2 multi-frame bridge drain (`PYRE_P2_DRAIN`, default OFF): gates the
/// `trace.rs` carrier drain sub-walk+inject deviation. The carrier itself is
/// installed for every `frames.len() > 1` resume (safety floor: a carrier routes
/// a hot multi-frame guard to the `CarrierAbort` blackhole instead of the
/// degenerate root-at-innermost-pc bridge); this gate only selects the drain
/// driver over the default framestack path. Until the drain is net-positive the
/// gate stays off so a carrier resume falls through to the framestack walk /
/// clean abort.
pub(crate) fn p2_drain_enabled() -> bool {
    std::env::var_os("PYRE_P2_DRAIN").is_some()
}

/// Decode one suspended inline-callee frame's resume section into a
/// [`ReconstructRecipe`], or `None` to decline the multi-frame inline rebuild.
///
/// `resume.py rebuild_from_resumedata` (`newframe(jitcode)` +
/// `setup_resume_at_op` + `consume_boxes` per suspended frame) ALWAYS
/// allocates and refills every inlined frame — its snapshot writer guarantees
/// the liveness invariants the reader consumes. Pyre's per-function portal
/// model + super-instruction bytecode cannot guarantee those invariants for
/// every reconstructed callee, so this returns `None` exactly where a rebuild
/// would be unsound: a no-snapshot pc, an unresolved jitcode, cell/free vars
/// (cell contents live outside the resume stream → LOAD_DEREF NameError),
/// unrecoverable callee globals, an `enumerate_vars` count that disagrees with
/// the encoded section (`consume_boxes`), or an int/float-bank register with no
/// boxed-Ref source.
///
/// The `None` fallback is semantically equivalent in program result: the caller
/// declines the multi-frame inline reconstruction and routes to the
/// conservative single-frame bridge / blackhole resume, which re-enters the
/// interpreter and resumes correctly — only the inline-frame reconstruction
/// optimization is forfeited, never correctness. The forward inline capture
/// declines on the same conditions.
fn reconstruct_inline_recipe(
    ctx: &mut majit_metainterp::TraceCtx,
    frame: &majit_ir::resumedata::RebuiltFrame,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache,
    in_a_call: bool,
) -> Option<ReconstructRecipe> {
    if frame.pc < 0 {
        return None;
    }
    let py_pc = backxlat_py_pc(frame.jitcode_index, frame.pc) as usize;
    let w_code = code_for_jitcode_index(frame.jitcode_index)?;
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let code_ref = unsafe { &*raw_code };
    if !code_ref.freevars.is_empty() || pyre_interpreter::pyframe::ncells(code_ref) != 0 {
        return None;
    }
    // pyframe.py:128-132 get_w_globals_storage(): the reconstructed callee frame's
    // globals come from its own pycode (`assemble_bridge_inline_pending`
    // resolves them the same way via `recover_inline_callee_globals`). If the
    // callee code never ran under known globals (no live wrapper recovers a
    // namespace), there is nothing to restore, so abort to the single-frame
    // bridge — the forward inline path declines the same way.
    if recover_inline_callee_globals(raw_code as *const ()).is_null() {
        return None;
    }
    let nlocals = code_ref.varnames.len();

    let stack_only = match crate::liveness::liveness_for(raw_code).stack_depth_at(py_pc) {
        Some(d) => d,
        None => return None,
    };
    let pending_result_abs_slot = if in_a_call && stack_only > 0 {
        Some(nlocals + stack_only - 1)
    } else {
        None
    };
    let pending_result_color = if in_a_call {
        Some(result_color_at_pc_at(frame.jitcode_index, py_pc)?)
    } else {
        None
    };
    let mut reg_indices = frame_liveness_reg_indices_by_bank_from_pc(frame.jitcode_index, frame.pc);
    // The encoder enumerates live boxes in bank order [int | ref | float] and
    // `frame.values` mirrors that order. A pending call-result color (the dst of
    // an in-flight residual call) is removed from the ref bank below because the
    // framestack walk delivers that value via `make_result_of_lastop`, not from
    // resumedata. Drop its encoded value from the SAME position so the liveness
    // enumeration and the consumed value section stay aligned
    // (resume.py:1054 consume_boxes).
    let mut pending_value_index: Option<usize> = None;
    if let Some(color) = pending_result_color {
        if reg_indices.int.iter().any(|&c| c as usize == color)
            || reg_indices.float.iter().any(|&c| c as usize == color)
        {
            return None;
        }
        if let Some(ref_pos) = reg_indices.ref_.iter().position(|&c| c as usize == color) {
            pending_value_index = Some(reg_indices.int.len() + ref_pos);
            reg_indices.ref_.remove(ref_pos);
        }
    }
    let values: Vec<&majit_ir::resumedata::RebuiltValue> = match pending_value_index {
        Some(idx) => frame
            .values
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if i == idx { None } else { Some(v) })
            .collect(),
        None => frame.values.iter().collect(),
    };
    // resume.py:1054 consume_boxes: the liveness enumeration count must match
    // the (pending-excluded) encoded frame section exactly.
    if reg_indices.total_len() != values.len() {
        return None;
    }
    // virtualizable.py:86-98: at a bytecode boundary (every resume pc is one)
    // the frame's locals_cells_stack_w is a W_Root array — all live slots are
    // boxed Ref. The encoder confirms this empirically (both reconstructed
    // function_calls frames decode 0 int / 0 float registers). A reconstructed
    // inline callee has no virtualizable array, so an int/float-bank register
    // here would have no boxed-Ref source to seed `registers_r` (the reader
    // always reads `registers_r[k]`, trace_opcode.rs:2128/684). Fall back to
    // the single-frame bridge rather than synthesize an unboxed local.
    if !reg_indices.int.is_empty() || !reg_indices.float.is_empty() {
        return None;
    }

    // Virtualizable-callee shape (pyre's "every function is its own portal"
    // model): the callee's only live ref registers are the portal reds
    // [frame, ec], and its locals live in the `frame` red's own
    // `locals_cells_stack_w` virtual array — the same place the ROOT frame keeps
    // its locals — NOT in the register section. Recover them into REGISTER-
    // SECTION slots so the bridge rebuilds the callee as a plain MIFrame
    // (resume.py:1042-1057 newframe + reload, reading `registers_r`), convergent
    // with the RPython frame shape and creating NO new vable. The color==
    // semantic identity check below only accepts a register-section callee
    // (locals at colors 0..nlocals), which this shape never satisfies.
    let (pframe_reg, pec_reg) = portal_red_regs_at(frame.jitcode_index);
    let (pframe_reg, pec_reg) = (pframe_reg as u32, pec_reg as u32);
    if pframe_reg != u32::from(u16::MAX)
        && pec_reg != u32::from(u16::MAX)
        && reg_indices.ref_.len() == 2
        && reg_indices.ref_.contains(&pframe_reg)
        && reg_indices.ref_.contains(&pec_reg)
    {
        use majit_ir::resumedata::{RebuiltValue, TAGVIRTUAL, UNINITIALIZED_TAG, untag};
        let frame_pos = reg_indices.ref_.iter().position(|&c| c == pframe_reg)?;
        let RebuiltValue::Virtual(frame_vidx) = values[frame_pos] else {
            return None;
        };
        // The `frame` red virtual is a PyFrame VirtualInfo; its
        // `locals_cells_stack_w` array field is at PYFRAME_LOCALS_CELLS_STACK_OFFSET.
        let array_vidx = {
            let majit_ir::RdVirtualInfo::VirtualInfo {
                fieldnums,
                fielddescrs,
                ..
            } = rd_virtual_at(rd_virtuals, *frame_vidx)?
            else {
                return None;
            };
            let arr_field_idx = fielddescrs.iter().position(|fd| {
                fd.offset == crate::frame_layout::PYFRAME_LOCALS_CELLS_STACK_OFFSET
            })?;
            let (av, tb) = untag(*fieldnums.get(arr_field_idx)?);
            if tb != TAGVIRTUAL {
                return None;
            }
            if av < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + av) as usize
            } else {
                av as usize
            }
        };
        let arr: Vec<i16> = match rd_virtual_at(rd_virtuals, array_vidx)? {
            majit_ir::RdVirtualInfo::VArrayInfoClear { fieldnums, .. }
            | majit_ir::RdVirtualInfo::VArrayInfoNotClear { fieldnums, .. } => fieldnums.clone(),
            _ => return None,
        };
        let valuestackdepth = nlocals + stack_only;
        if valuestackdepth > arr.len() {
            return None;
        }
        let callinfocollection = ctx.callinfocollection.clone();
        let mut registers_r = vec![OpRef::NONE; valuestackdepth];
        let mut concrete_r = vec![majit_ir::Value::Void; valuestackdepth];
        // locals_cells_stack_w is SEMANTIC-slot-ordered ([locals | stack]) and a
        // W_Root array — every live slot is Ref. An UNINITIALIZED local stays
        // NULL in the rebuilt frame; a missing operand-stack slot is
        // unreconstructable and declines below.
        for k in 0..valuestackdepth {
            if Some(k) == pending_result_abs_slot {
                continue;
            }
            let tag = arr[k];
            if tag == UNINITIALIZED_TAG {
                continue;
            }
            registers_r[k] = decode_fieldnum(ctx, tag, rd_virtuals, resume_data, cache);
            let bits = decode_tagged_concrete(
                tag,
                Type::Ref,
                rd_virtuals,
                fail_values,
                resume_data.num_failargs,
                resume_data.storage.as_ref(),
                backend,
                callinfocollection.as_ref(),
                cache,
            );
            concrete_r[k] = value_for_slot(Type::Ref, bits);
        }
        for s in nlocals..valuestackdepth {
            if Some(s) == pending_result_abs_slot {
                continue;
            }
            if registers_r[s] == OpRef::NONE {
                return None;
            }
        }
        return Some(ReconstructRecipe {
            code_ptr: raw_code as *const (),
            jitcode_index: frame.jitcode_index,
            jitcode_pc: frame.pc,
            nlocals,
            valuestackdepth,
            registers_i: Vec::new(),
            registers_r,
            registers_f: Vec::new(),
            concrete_r,
            nargs: nlocals,
        });
    }

    // The recipe banks are written and read by the register COLOR reported in
    // the liveness stream (`registers_r[reg_idx]` below), but every consumer
    // indexes by the SEMANTIC `locals_cells_stack_w` slot: the trait re-trace's
    // `load_local_value` reads `registers_r[local_idx]` (trace_opcode.rs:2360)
    // and `stack_slot_reg_idx` reads `registers_r[nlocals + stack_idx]`
    // (trace_opcode.rs:628), and `assemble_bridge_inline_pending` reads
    // `concrete_r[k]` by semantic slot k. This recipe is therefore only valid
    // when each live color equals the semantic slot it denotes (color ==
    // semantic). Regalloc pins that identity at call-boundary resume pcs (a
    // straight-line callee resumes after its CALL with locals at colors
    // `0..nlocals`), but a MID-BODY resume — e.g. a guard fired inside a
    // callee branch (`goto_if_not` target) — coalesces colors so the live
    // stack value sits at a renamed color. #73: the per-PC `pcdep_entries`
    // color→slot map is the authoritative inversion; when it is empty at this
    // resume pc there is no per-pc map to faithfully rebuild the frame, so
    // decline to the single-frame bridge (whose vable payload IS semantic-
    // ordered) rather than rebuild the frame with mis-slotted boxes.
    let maps = bridge_semantic_maps_at(frame.jitcode_index, frame.pc);
    if maps.pcdep_entries.is_empty() {
        return None;
    }
    for &color in &reg_indices.ref_ {
        match semantic_ref_slot_for_reg_color(
            nlocals,
            stack_only,
            &maps.pcdep_entries,
            color as usize,
        ) {
            // color == semantic slot: the recipe's color-indexed fill is
            // also the correct semantic fill.
            Some(semantic_idx) if semantic_idx == color as usize => {}
            // A live color whose semantic slot differs (or is unmappable):
            // the color-indexed recipe would mis-slot it. Decline.
            _ => return None,
        }
    }

    let mut registers_i: Vec<OpRef> = Vec::new();
    let mut registers_f: Vec<OpRef> = Vec::new();
    // The Ref bank decodes COLOR-indexed (the liveness register `reg_idx` is the
    // post-regalloc color). The reconstructed inline frame's `registers_r`/
    // `concrete_r` are read SLOT-indexed (`assemble_bridge_inline_pending` seeds
    // `locals_cells_stack_w[k]` and the resumed tracer reads LOAD_FAST's
    // `nlocals + stack_idx`), so invert color→slot below — for a borrowed local
    // pushed on the stack the color ≠ slot, and a color-indexed seed lands the
    // value at the wrong slot.
    let mut by_color_r: Vec<OpRef> = Vec::new();
    let mut by_color_c: Vec<majit_ir::Value> = Vec::new();
    let mut value_cursor = 0usize;
    for &reg_idx in &reg_indices.int {
        let (op, _val) = bridge_decode_box(
            ctx,
            values[value_cursor],
            Type::Int,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_i.len() {
            registers_i.resize(reg_idx + 1, OpRef::NONE);
        }
        registers_i[reg_idx] = op;
        value_cursor += 1;
    }
    for &reg_idx in &reg_indices.ref_ {
        let (op, val) = bridge_decode_box(
            ctx,
            values[value_cursor],
            Type::Ref,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= by_color_r.len() {
            by_color_r.resize(reg_idx + 1, OpRef::NONE);
            by_color_c.resize(reg_idx + 1, majit_ir::Value::Void);
        }
        by_color_r[reg_idx] = op;
        by_color_c[reg_idx] = val;
        value_cursor += 1;
    }
    for &reg_idx in &reg_indices.float {
        let (op, _val) = bridge_decode_box(
            ctx,
            values[value_cursor],
            Type::Float,
            rd_virtuals,
            resume_data,
            fail_values,
            fail_types,
            backend,
            cache,
        );
        let reg_idx = reg_idx as usize;
        if reg_idx >= registers_f.len() {
            registers_f.resize(reg_idx + 1, OpRef::NONE);
        }
        registers_f[reg_idx] = op;
        value_cursor += 1;
    }

    // pyframe.py:107-110: locals + cells + stack. Cells are gated out above.
    // The semantic `valuestackdepth` is `stack_base() + operand_depth`, where
    // `operand_depth` is the logical stack height the codewriter's forward
    // dataflow computes for this pc (`LiveVars::stack_depth_at`). The portal
    // frame reads the equivalent figure from the encoded vable
    // `valuestackdepth` scalar (state.rs:6382); an inline frame has no such
    // scalar, so derive it from the bytecode here. An unreachable resume pc
    // aborts the multi-frame path.
    let valuestackdepth = nlocals + stack_only;

    // Invert the COLOR-indexed decode into SLOT-indexed `registers_r`/
    // `concrete_r`, mirroring the root frame's `setup_bridge_sym` color→slot
    // mirror. `maps.pcdep_entries` is the per-PC `(color, slot)` map (non-empty
    // — the identity gate above declined otherwise): each decoded color is
    // placed at the semantic slot it denotes at this resume pc, so a borrowed
    // local on the operand stack lands at its stack slot, not its color.
    let mut registers_r = vec![OpRef::NONE; valuestackdepth];
    let mut concrete_r = vec![majit_ir::Value::Void; valuestackdepth];
    for &(bank, color, slot) in &maps.pcdep_entries {
        if bank != 1 {
            continue;
        } // Ref bank only
        let s = slot as usize;
        let c = color as usize;
        if s < valuestackdepth && c < by_color_r.len() && by_color_r[c] != OpRef::NONE {
            registers_r[s] = by_color_r[c];
            concrete_r[s] = by_color_c[c];
        }
    }

    // Refill registerless operand-stack constants the color map omits (an
    // inlined callee has no value-stack resumedata to rematerialize them from).
    for (slot, raw) in const_ref_slots_from_pc(frame.jitcode_index, frame.pc) {
        let s = slot as usize;
        if Some(s) == pending_result_abs_slot {
            continue;
        }
        if s < valuestackdepth {
            registers_r[s] = ctx.const_ref(raw);
            concrete_r[s] = value_for_slot(Type::Ref, raw);
        }
    }

    // Every operand-stack slot is live (the logical stack is dense). If one is
    // still unfilled after the color→slot inversion and the constant refill,
    // the slot holds a value neither path can reconstruct (an int/float operand
    // constant, or an unsupported shape) — decline to the single-frame bridge
    // rather than seed a NULL operand the re-executed bridge would deref.
    for s in nlocals..valuestackdepth {
        if Some(s) == pending_result_abs_slot {
            continue;
        }
        if registers_r[s] == OpRef::NONE {
            return None;
        }
    }

    Some(ReconstructRecipe {
        code_ptr: raw_code as *const (),
        jitcode_index: frame.jitcode_index,
        jitcode_pc: frame.pc,
        nlocals,
        valuestackdepth,
        registers_i,
        registers_r,
        registers_f,
        concrete_r,
        nargs: nlocals,
    })
}

fn bridge_decode_box(
    ctx: &mut majit_metainterp::TraceCtx,
    v: &majit_ir::resumedata::RebuiltValue,
    expected_kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    fail_values: &[i64],
    fail_types: &[Type],
    backend: &dyn majit_backend::Backend,
    cache: &mut BridgeVirtualCache,
) -> (OpRef, majit_ir::Value) {
    use majit_ir::resumedata::RebuiltValue;
    let callinfocollection = ctx.callinfocollection.clone();
    match v {
        RebuiltValue::Box(n, tp) => {
            let opref = OpRef::input_arg_typed(*n as u32, *tp);
            let bits = fail_values[*n];
            let effective_tp = fail_types.get(*n).copied().unwrap_or(*tp);
            // resume.py:1264 assert box.type == kind
            assert!(
                effective_tp == expected_kind,
                "bridge_decode_box: Box({n}) type {effective_tp:?} != expected {expected_kind:?}"
            );
            (opref, value_for_slot(effective_tp, bits))
        }
        RebuiltValue::Const(c) => {
            let opref = match c.get_type() {
                Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                Type::Float => ctx.const_float(c.getfloatstorage()),
                _ => ctx.const_int(c.getint()),
            };
            // resume.py:1264 assert box.type == kind
            assert!(
                c.get_type() == expected_kind,
                "bridge_decode_box: Const type {:?} != expected {:?}",
                c.get_type(),
                expected_kind,
            );
            (opref, value_for_slot(c.get_type(), c.as_raw_i64()))
        }
        RebuiltValue::Virtual(vidx) => {
            let opref = materialize_bridge_virtual(ctx, *vidx, rd_virtuals, resume_data, cache);
            if expected_kind == Type::Int {
                let value = materialize_concrete_virtual_int(
                    *vidx,
                    rd_virtuals,
                    fail_values,
                    resume_data.num_failargs,
                    resume_data.storage.as_ref(),
                    backend,
                    callinfocollection.as_ref(),
                    cache,
                );
                (opref, majit_ir::Value::Int(value))
            } else {
                let gcref = materialize_concrete_virtual_ptr(
                    *vidx,
                    rd_virtuals,
                    fail_values,
                    resume_data.num_failargs,
                    resume_data.storage.as_ref(),
                    backend,
                    callinfocollection.as_ref(),
                    cache,
                );
                (opref, majit_ir::Value::Ref(gcref))
            }
        }
        RebuiltValue::Unassigned => (OpRef::NONE, majit_ir::Value::Void),
    }
}

/// resume.py:1245-1264 decode_box concrete parity for tagged fieldnums.
/// Converts a tagged i16 (from `rd_virtuals[*].fieldnums`) into raw i64
/// bits suitable for backend concrete setters/calls.
fn decode_tagged_concrete(
    tagged: i16,
    expected_kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> i64 {
    use majit_ir::resumedata::{
        NULLREF, TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, UNINITIALIZED_TAG, untag,
    };
    if tagged == UNINITIALIZED_TAG {
        return 0;
    }
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGBOX => {
            let idx = if val < 0 {
                (val + num_failargs) as usize
            } else {
                val as usize
            };
            // resume.py:1261 fail-loud: direct indexing, not silent fallback
            fail_values[idx]
        }
        TAGINT => val as i64,
        TAGCONST => {
            if tagged == NULLREF {
                return 0;
            }
            let ci = (val - TAG_CONST_OFFSET) as usize;
            // resume.py:1251 fail-loud: direct indexing
            let storage = storage.expect("decode_tagged_concrete: TAGCONST requires storage");
            storage.rd_consts()[ci].as_raw_i64()
        }
        TAGVIRTUAL => {
            // resume.py:278-284 nested virtuals are numbered negatively;
            // getvirtual resolves them via Python negative list indexing
            // into rd_virtuals (resume.py:951-954).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            if expected_kind == Type::Int {
                materialize_concrete_virtual_int(
                    vidx,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                )
            } else {
                let gcref = materialize_concrete_virtual_ptr(
                    vidx,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                gcref.0 as i64
            }
        }
        _ => 0,
    }
}

fn bh_field_descr_from_info(fd: &majit_ir::FieldDescrInfo) -> majit_translate::jitcode::BhDescr {
    majit_translate::jitcode::BhDescr::Field {
        offset: fd.offset,
        field_size: fd.field_size,
        field_type: fd.field_type,
        field_flag: majit_ir::ArrayFlag::from_field_type(fd.field_type),
        is_field_signed: matches!(fd.field_type, Type::Int),
        is_immutable: false,
        is_quasi_immutable: false,
        index_in_parent: fd.index as usize,
        parent: None,
        name: String::new(),
        owner: String::new(),
    }
}

fn bh_size_descr_from_size_descr(
    size_descr: &dyn majit_ir::descr::SizeDescr,
    vtable: usize,
) -> majit_translate::jitcode::BhDescr {
    majit_translate::jitcode::BhDescr::Size {
        size: size_descr.size(),
        // descr.py type_id is the dense GC tid for alloc_nursery_typed,
        // not the cache_key structural identity
        type_id: size_descr.type_id() as u64,
        vtable: vtable as u64,
        owner: String::new(),
        all_fielddescrs: majit_translate::jitcode::bh_field_specs_from_size_descr(size_descr),
        // Round-trip the GC-header flag off the descr.
        is_gc_managed: size_descr.is_gc_managed(),
    }
}

fn bh_array_descr_from_descr(descr: &majit_ir::DescrRef) -> majit_translate::jitcode::BhDescr {
    let ad = descr
        .as_array_descr()
        .expect("resume.py: allocate_array requires ArrayDescr");
    majit_translate::jitcode::BhDescr::from_array_descr(ad)
}

fn decode_tagged_for_kind(
    tagged: i16,
    kind: Type,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> i64 {
    decode_tagged_concrete(
        tagged,
        kind,
        rd_virtuals,
        fail_values,
        num_failargs,
        storage,
        backend,
        callinfocollection,
        cache,
    )
}

fn setfield_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    struct_ptr: i64,
    fd: &majit_ir::FieldDescrInfo,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) {
    let descr = bh_field_descr_from_info(fd);
    match fd.field_type {
        Type::Ref => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setfield_gc_r(struct_ptr, majit_ir::GcRef(value as usize), &descr);
        }
        Type::Float => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Float,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setfield_gc_f(struct_ptr, f64::from_bits(value as u64), &descr);
        }
        _ => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setfield_gc_i(struct_ptr, value, &descr);
        }
    }
}

fn setarrayitem_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    array_ptr: i64,
    index: usize,
    arraydescr: &dyn majit_ir::descr::ArrayDescr,
    bh_descr: &majit_translate::jitcode::BhDescr,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) {
    if arraydescr.is_array_of_pointers() {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Ref,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        backend.bh_setarrayitem_gc_r(
            array_ptr,
            index as i64,
            majit_ir::GcRef(value as usize),
            bh_descr,
        );
    } else if arraydescr.is_array_of_floats() {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Float,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        backend.bh_setarrayitem_gc_f(
            array_ptr,
            index as i64,
            f64::from_bits(value as u64),
            bh_descr,
        );
    } else {
        let value = decode_tagged_for_kind(
            fieldnum,
            Type::Int,
            rd_virtuals,
            fail_values,
            num_failargs,
            storage,
            backend,
            callinfocollection,
            cache,
        );
        backend.bh_setarrayitem_gc_i(array_ptr, index as i64, value, bh_descr);
    }
}

/// resume.py:1543-1552 BlackholeResumeDataReader.setinteriorfield_{int,
/// ref,float}: dispatch on the `InteriorFieldDescr.fielddescr` type to
/// `cpu.bh_setinteriorfield_gc_{i,r,f}`.
fn setinteriorfield_concrete_from_tagged(
    backend: &dyn majit_backend::Backend,
    array_ptr: i64,
    index: usize,
    interior_descr: &dyn majit_ir::descr::Descr,
    fieldnum: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) {
    let ifd = interior_descr
        .as_interior_field_descr()
        .expect("VArrayStructInfo: fielddescr is not an InteriorFieldDescr");
    let bh = majit_translate::jitcode::BhDescr::from_interior_field_descr(ifd);
    match ifd.field_descr().field_type() {
        Type::Ref => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setinteriorfield_gc_r(
                array_ptr,
                index as i64,
                majit_ir::GcRef(value as usize),
                &bh,
            );
        }
        Type::Float => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Float,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setinteriorfield_gc_f(
                array_ptr,
                index as i64,
                f64::from_bits(value as u64),
                &bh,
            );
        }
        _ => {
            let value = decode_tagged_for_kind(
                fieldnum,
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            backend.bh_setinteriorfield_gc_i(array_ptr, index as i64, value, &bh);
        }
    }
}

fn bh_call_r_for_oopspec(
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    oopspec: majit_ir::OopSpecIndex,
    args_i: Option<&[i64]>,
    args_r: Option<&[i64]>,
) -> majit_ir::GcRef {
    let cic = callinfocollection.expect(
        "TraceCtx.callinfocollection missing — concrete VStr/VUni materialization \
         requires resume.py DirectReader funcptr_for_oopspec parity",
    );
    let (calldescr, func) = cic.callinfo_for_oopspec(oopspec);
    let calldescr = calldescr
        .expect("callinfo_for_oopspec missing entry for concrete VStr/VUni materialization");
    let cd = calldescr
        .as_call_descr()
        .expect("VStr/VUni oopspec calldescr must be CallDescr");
    let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_call_descr(cd);
    backend.bh_call_r(func as i64, args_i, args_r, None, &bh_calldescr)
}

/// resume.py:945-956 getvirtual_ptr concrete parity.
/// Lazily allocate a concrete object for virtual index `vidx`, caching in
/// `BridgeVirtualCache.concrete_ptr_cache` so shared/recursive virtuals
/// allocate exactly once.
///
/// resume.py:618-760 polymorphic dispatch by RdVirtualInfo variant.
/// BlackholeResumeDataReader (resume.py:1430-1460) routes each kind
/// through the CPU backend: bh_new_with_vtable, bh_new, bh_new_array,
/// bh_newstr, bh_newunicode, bh_call_i/r.
fn materialize_concrete_virtual_ptr(
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> majit_ir::GcRef {
    if let Some(cached) = cache.get_concrete_ptr(vidx) {
        return cached;
    }
    // resume.py:953 assert self.rd_virtuals is not None
    let virtuals = rd_virtuals.expect("materialize_concrete_virtual_ptr: rd_virtuals is None");
    // resume.py:954 self.rd_virtuals[index].allocate(self, index) — direct indexing
    let entry = &virtuals[vidx];
    match entry.as_ref() {
        // resume.py:618-621 VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let descr = descr.as_ref().expect("VirtualInfo: descr is None");
            let size_descr = descr.as_size_descr().expect("VirtualInfo: not a SizeDescr");
            let vtable = size_descr.vtable();
            // resume.py:1111 allocate_with_vtable(descr) → cpu.bh_new_with_vtable(descr)
            let bh_descr = bh_size_descr_from_size_descr(size_descr, vtable);
            let ptr = backend.bh_new_with_vtable(&bh_descr);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            // Pyre adaptation: bh_new_with_vtable writes vtable at
            // vtable_offset but PyObject.w_class needs separate init
            // (pyobject.rs:51). Matches materialize_virtual_object at
            // state.rs:7220.
            if vtable != 0 {
                unsafe {
                    let pyobj = ptr as *mut pyre_object::PyObject;
                    (*pyobj).w_class = pyre_object::pyobject::get_instantiate(
                        &*(vtable as *const pyre_object::pyobject::PyType),
                    );
                }
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            // resume.py:620 cache BEFORE filling fields (circular ref safe)
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:597-603 setfields — range(len(fielddescrs)), index
            // fieldnums[i]. The len-equality assert (resume.py:606) is in
            // debug_prints, not this allocate path: a short fieldnums raises
            // IndexError here, a longer one is ignored.
            for i in 0..fielddescrs.len() {
                let fd = &fielddescrs[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setfield_concrete_from_tagged(
                    backend,
                    ptr,
                    fd,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            gcref
        }
        // resume.py:633-637 VStructInfo.allocate — no vtable
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let descr = typedescr.as_ref().expect("VStructInfo: typedescr is None");
            let size_descr = descr.as_size_descr().expect("VStructInfo: not a SizeDescr");
            // resume.py:1114 allocate_struct(typedescr) → cpu.bh_new(typedescr)
            let bh_descr = bh_size_descr_from_size_descr(size_descr, 0);
            let ptr = backend.bh_new(&bh_descr);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:637 setfields — range(len(fielddescrs)), index
            // fieldnums[i]. The len-equality assert (resume.py:606) is in
            // debug_prints, not this allocate path: a short fieldnums raises
            // IndexError here, a longer one is ignored.
            for i in 0..fielddescrs.len() {
                let fd = &fielddescrs[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setfield_concrete_from_tagged(
                    backend,
                    ptr,
                    fd,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            gcref
        }
        // resume.py:650-670 VArrayInfo.allocate
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let length = fieldnums.len();
            let is_clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            // resume.py:1117 allocate_array(length, arraydescr, clear)
            // → cpu.bh_new_array_clear / bh_new_array
            let descr = arraydescr.as_ref().expect("VArrayInfo: arraydescr is None");
            let ad = descr
                .as_array_descr()
                .expect("VArrayInfo: not an ArrayDescr");
            let bh_descr = bh_array_descr_from_descr(descr);
            let ptr = if is_clear {
                backend.bh_new_array_clear(length as i64, &bh_descr)
            } else {
                backend.bh_new_array(length as i64, &bh_descr)
            };
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:660-670 setarrayitem per element
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                setarrayitem_concrete_from_tagged(
                    backend,
                    ptr,
                    i,
                    ad,
                    &bh_descr,
                    fnum,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    callinfocollection,
                    cache,
                );
            }
            gcref
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            // resume.py:749 array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            let bh_array = bh_array_descr_from_descr(array_descr);
            // resume.py:749 clear=True → bh_new_array_clear
            let ptr = backend.bh_new_array_clear(*size as i64, &bh_array);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num,
            //                   self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    setinteriorfield_concrete_from_tagged(
                        backend,
                        ptr,
                        i,
                        fielddescrs[j].as_ref(),
                        fnum,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        callinfocollection,
                        cache,
                    );
                }
            }
            gcref
        }
        // resume.py:763-783 VStrPlainInfo.allocate
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
            let length = fieldnums.len();
            // resume.py:1134 allocate_string(length) → cpu.bh_newstr(length)
            let ptr = backend.bh_newstr(length as i64);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:1138 string_setitem → cpu.bh_strsetitem
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_tagged_concrete(
                    fnum,
                    Type::Int,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                backend.bh_strsetitem(ptr, i as i64, value);
            }
            gcref
        }
        // resume.py:820-840 VUniPlainInfo.allocate
        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let length = fieldnums.len();
            // resume.py:1162 allocate_unicode(length) → cpu.bh_newunicode(length)
            let ptr = backend.bh_newunicode(length as i64);
            if ptr == 0 {
                return majit_ir::GcRef::NULL;
            }
            let gcref = majit_ir::GcRef(ptr as usize);
            cache.set_concrete_ptr(vidx, gcref);
            // resume.py:1166 unicode_setitem → cpu.bh_unicodesetitem
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_tagged_concrete(
                    fnum,
                    Type::Int,
                    rd_virtuals,
                    fail_values,
                    num_failargs,
                    storage,
                    backend,
                    callinfocollection,
                    cache,
                );
                backend.bh_unicodesetitem(ptr, i as i64, value);
            }
            gcref
        }
        // resume.py:785-805 VStrConcatInfo / VUniConcatInfo
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            assert!(fieldnums.len() == 2, "concat: expected 2 fieldnums");
            let str1 = decode_tagged_concrete(
                fieldnums[0],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let str2 = decode_tagged_concrete(
                fieldnums[1],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let oopspec = if matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VStrConcatInfo { .. }
            ) {
                majit_ir::descr::OopSpecIndex::StrConcat
            } else {
                majit_ir::descr::OopSpecIndex::UniConcat
            };
            // resume.py:1462-1497 DirectReader concat_* resolves funcptr
            // via callinfocollection.funcptr_for_oopspec, not a placeholder.
            let gcref = bh_call_r_for_oopspec(
                backend,
                callinfocollection,
                oopspec,
                None,
                Some(&[str1, str2]),
            );
            cache.set_concrete_ptr(vidx, gcref);
            gcref
        }
        // resume.py:805-818 VStrSliceInfo / VUniSliceInfo
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            assert!(fieldnums.len() == 3, "slice: expected 3 fieldnums");
            let strbox = decode_tagged_concrete(
                fieldnums[0],
                Type::Ref,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let start = decode_tagged_concrete(
                fieldnums[1],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let length = decode_tagged_concrete(
                fieldnums[2],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let stop = start + length;
            let oopspec = if matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VStrSliceInfo { .. }
            ) {
                majit_ir::descr::OopSpecIndex::StrSlice
            } else {
                majit_ir::descr::OopSpecIndex::UniSlice
            };
            // resume.py:1472-1507 DirectReader slice_* passes stop =
            // start + length to the funcptr resolved from callinfocollection.
            let gcref = bh_call_r_for_oopspec(
                backend,
                callinfocollection,
                oopspec,
                Some(&[start, stop]),
                Some(&[strbox]),
            );
            cache.set_concrete_ptr(vidx, gcref);
            gcref
        }
        // Raw virtuals are INT-kind — should not appear in getvirtual_ptr
        majit_ir::RdVirtualInfo::VRawBufferInfo { .. }
        | majit_ir::RdVirtualInfo::VRawSliceInfo { .. } => {
            panic!(
                "materialize_concrete_virtual_ptr: raw virtual at vidx={vidx} is INT-kind, not PTR"
            )
        }
        // resume.py:954 getvirtual_ptr calls rd_virtuals[index].allocate()
        // directly; a None/Empty slot is never referenced by a TAGVIRTUAL tag
        // in a well-formed resume stream, so reaching it is a corrupt-stream
        // bug, not a NULL fallback (mirrors materialize_concrete_virtual_int
        // and eval.rs materialize_virtual, which already panic on Empty).
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_concrete_virtual_ptr: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

/// resume.py:958-967 getvirtual_int concrete parity.  Only raw-buffer
/// virtuals are INT virtuals (`VAbstractRawInfo.kind = INT`).
fn materialize_concrete_virtual_int(
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    fail_values: &[i64],
    num_failargs: i32,
    storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    backend: &dyn majit_backend::Backend,
    callinfocollection: Option<&std::sync::Arc<majit_ir::CallInfoCollection>>,
    cache: &mut BridgeVirtualCache,
) -> i64 {
    if let Some(cached) = cache.get_concrete_int(vidx) {
        return cached;
    }
    // resume.py:959 assert self.virtuals_cache is not None (cache is not None by construction)
    // resume.py:962 self.rd_virtuals[index] — direct indexing, not silent fallback
    let virtuals = rd_virtuals.expect("materialize_concrete_virtual_int: rd_virtuals is None");
    let entry = &virtuals[vidx];
    // resume.py:964 assert v.is_about_raw and isinstance(v, VAbstractRawInfo)
    match entry.as_ref() {
        // resume.py:700-709 VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py:1452-1456 BlackholeResumeDataReader.allocate_raw_buffer:
            //   cic = self.callinfocollection
            //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
            //   return self.cpu.bh_call_i(func, [size], None, None, calldescr)
            // No fallback upstream — a missing CIC entry is a metainterp-setup
            // bug, surfaced fail-loud rather than papered over with a synthetic
            // calldescr.
            let cic = callinfocollection.expect("allocate_raw_buffer: callinfocollection is None");
            // resume.py:1455: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py:444-447) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1456: self.cpu.bh_call_i(func, [size], None, None,
            //   calldescr). A missing entry surfaces here, as the calldescr is
            // consumed by the backend call, not as a separate lookup assertion.
            let descr_ref =
                calldescr.expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)");
            let cd = descr_ref
                .as_call_descr()
                .expect("OS_RAW_MALLOC_VARSIZE_CHAR: not a CallDescr");
            let bh = majit_translate::jitcode::BhCallDescr::from_arg_classes(
                cd.arg_classes(),
                cd.result_class(),
                cd.get_extra_info().clone(),
            );
            let buffer = backend.bh_call_i(*func, Some(&[*size as i64]), None, None, &bh);
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_concrete_int(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets) unconditionally
            // (no buffer == 0 guard), indexing self.descrs[i] and
            // self.fieldnums[i] by the same i — a short descrs/fieldnums raises
            // IndexError here (encoder bug), a longer one is ignored. No
            // len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py:1543 assert not descr.is_array_of_pointers()
                assert!(
                    descrs[i].item_type != 0,
                    "setrawbuffer_item: pointer arraydescr in raw buffer"
                );
                // resume.py:1545-1552 setrawbuffer_item: the arraydescr drives
                // the store width via the backend's write_{int,float}_at_mem,
                // so the value is stored through bh_raw_store_{f,i} with the
                // real descr — no caller-side field-size truncation.
                let store_descr =
                    majit_translate::jitcode::BhDescr::from_array_descr_info(&descrs[i]);
                if descrs[i].item_type == 2 {
                    let value = decode_tagged_concrete(
                        fnum,
                        Type::Float,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        backend,
                        callinfocollection,
                        cache,
                    );
                    // resume.py:1548-1549 bh_raw_store_f(ptr, offset, floatval, descr)
                    backend.bh_raw_store_f(
                        buffer,
                        off as i64,
                        f64::from_bits(value as u64),
                        &store_descr,
                    );
                } else {
                    let value = decode_tagged_concrete(
                        fnum,
                        Type::Int,
                        rd_virtuals,
                        fail_values,
                        num_failargs,
                        storage,
                        backend,
                        callinfocollection,
                        cache,
                    );
                    // resume.py:1550-1552 bh_raw_store_i(ptr, offset, intval, descr)
                    backend.bh_raw_store_i(buffer, off as i64, value, &store_descr);
                }
            }
            buffer
        }
        // resume.py:722-728 VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            let base = decode_tagged_concrete(
                fieldnums[0],
                Type::Int,
                rd_virtuals,
                fail_values,
                num_failargs,
                storage,
                backend,
                callinfocollection,
                cache,
            );
            let buffer = base + *offset;
            cache.set_concrete_int(vidx, buffer);
            buffer
        }
        // resume.py:963-964 getvirtual_int asserts is_about_raw
        other => panic!(
            "materialize_concrete_virtual_int: non-raw virtual kind at vidx={vidx}: {other:?}"
        ),
    }
}

/// resume.py:1556-1564 decode_box parity for fieldnums (i16 tagged): decode one
/// tagged array/field value into its bridge `OpRef` (typed InputArg for TAGBOX,
/// const for TAGINT/TAGCONST, recursively materialized virtual for TAGVIRTUAL).
fn decode_fieldnum(
    ctx: &mut majit_metainterp::TraceCtx,
    tagged: i16,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    cache: &mut BridgeVirtualCache,
) -> OpRef {
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};
    // resume.py:1245 `decode_box` dispatches purely on the tag bits;
    // it has no UNINITIALIZED case. The UNINITIALIZED skip lives in
    // the callers (e.g. VArrayStructInfo.allocate, resume.py:629),
    // so this decoder mirrors `decode_box` exactly — an UNINITIALIZED
    // tag reaching here falls into the TAGCONST arm and fails loud on
    // the out-of-range const index, matching upstream's IndexError.
    let (val, tagbits) = untag(tagged);
    match tagbits {
        TAGBOX => {
            // resume.py:1247-1264 decode_box parity:
            //   if num < 0: num += len(liveboxes)
            //   return self.liveboxes[num]
            // The returned Box object carries `box.type` intrinsically
            // (history.py:220). For the bridge tracer, those liveboxes
            // are the bridge's `InputArg{Int,Ref,Float}` slots, so we
            // mint the typed `OpRef::input_arg_typed` variant matching
            // `fail_arg_types[idx]` rather than a bare untyped raw
            // OpRef — variant-aware Eq (resoperation.rs:290) requires
            // the optimizer/heap-cache key to be the same typed variant
            // the bridge inputarg list produces.
            let idx = if val < 0 {
                val + resume_data.num_failargs
            } else {
                val
            };
            // resume.py:1261 `box = self.liveboxes[num]` — direct
            // indexing, IndexError on out-of-range. Encoder /
            // decoder asymmetry is a bug, not a silent fallback;
            // mirror the upstream fail-loud contract.
            let tp = *resume_data
                .fail_arg_types
                .get(idx as usize)
                .unwrap_or_else(|| {
                    panic!(
                        "decode_fieldnum TAGBOX out-of-range: idx={} num_failargs={} \
                         fail_arg_types.len()={} (encoder/decoder mismatch — see \
                         resume.py:1245-1264 decode_box)",
                        idx,
                        resume_data.num_failargs,
                        resume_data.fail_arg_types.len()
                    )
                });
            OpRef::input_arg_typed(idx as u32, tp)
        }
        TAGINT => ctx.const_int(val as i64),
        TAGCONST => {
            // resume.py:1247-1251 decode_box parity:
            //   if tag == TAGCONST:
            //       if tagged_eq(tagged, NULLREF):
            //           box = CONST_NULL
            //       else:
            //           box = self.consts[num - TAG_CONST_OFFSET]
            if tagged == majit_ir::resumedata::NULLREF {
                return ctx.const_null();
            }
            let ci = (val - TAG_CONST_OFFSET) as usize;
            // resume.py:1251 `box = self.consts[num - TAG_CONST_OFFSET]`
            // — direct indexing, fail-fast on out-of-range (mirrors
            // Python IndexError; never silently substitutes).
            // compile.py:853 `ResumeGuardDescr` storage — read off
            // the shared Arc so the bridge tracer observes the
            // same pool the GC walker updates.
            let storage = resume_data
                .storage
                .as_ref()
                .expect("resume_data.storage missing");
            let c = storage.rd_consts()[ci];
            match c.get_type() {
                majit_ir::Type::Ref => ctx.const_ref(c.getref_base().as_usize() as i64),
                majit_ir::Type::Float => ctx.const_float(c.getfloatstorage()),
                _ => ctx.const_int(c.getint()),
            }
        }
        TAGVIRTUAL => {
            // resume.py:278-284 nested virtuals are numbered negatively;
            // getvirtual resolves them via Python negative list indexing
            // into rd_virtuals (resume.py:951-954).
            let vidx = if val < 0 {
                (rd_virtuals.map_or(0, |v| v.len()) as i32 + val) as usize
            } else {
                val as usize
            };
            materialize_bridge_virtual(ctx, vidx, rd_virtuals, resume_data, cache)
        }
        _ => OpRef::NONE,
    }
}

fn materialize_bridge_virtual(
    ctx: &mut majit_metainterp::TraceCtx,
    vidx: usize,
    rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    cache: &mut BridgeVirtualCache,
) -> OpRef {
    use majit_ir::OpCode;
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};

    // resume.py:874-899 VirtualCache: list caches indexed by virtual number
    // (ptr and int banks). This bridge helper is still OpRef-typed, so it
    // probes both banks before allocating.
    if let Some(cached) = cache.get_any(vidx) {
        return cached;
    }

    // resume.py:947 assert self.virtuals_cache is not None — a TAGVIRTUAL in
    // the stream guarantees rd_virtuals is present; None is an encoder bug.
    let virtuals = rd_virtuals.expect("materialize_bridge_virtual: rd_virtuals is None");
    // resume.py:951 self.rd_virtuals[index] — direct indexing, IndexError on
    // an out-of-range virtual number is a bug, not a silent NONE fallback.
    let entry = &virtuals[vidx];

    // resume.py:612-760 dispatch by virtual kind.
    // RPython: rd_virtuals[index].allocate(self, index) — polymorphic on
    // the AbstractVirtualInfo subclass. Rust equivalent: match on
    // RdVirtualInfo enum variant.

    /// resume.py:591-603 AbstractVirtualStructInfo.setfields helper.
    /// Walks fielddescrs in lock-step with fieldnums, decoding each
    /// fieldnum and emitting SETFIELD_GC.
    fn setfields(
        ctx: &mut majit_metainterp::TraceCtx,
        struct_op: OpRef,
        fielddescrs: &[majit_ir::FieldDescrInfo],
        fieldnums: &[i16],
        parent_descr: majit_ir::DescrRef,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut BridgeVirtualCache,
    ) {
        // resume.py:597-603 setfields — range(len(fielddescrs)), index
        // fieldnums[i]. The len-equality assert (resume.py:606) is in
        // debug_prints, not this allocate path: a short fieldnums raises
        // IndexError here, a longer one is ignored.
        for i in 0..fielddescrs.len() {
            let fd_info = &fielddescrs[i];
            let fnum = fieldnums[i];
            if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                continue;
            }
            let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
            if value.is_none() {
                continue;
            }
            // resume.py:597-603 self.setfields → decoder.setfield(struct,
            // fieldnum, fielddescr): reuse the parent SizeDescr's live
            // FieldDescr (canonical immutable / quasi-immutable / ei_index)
            // rather than reconstructing a partial copy. The descr is keyed
            // by index_in_parent (small sequential), not the 268M-hash
            // stable_field_index.
            let field_descr =
                crate::descr::make_field_descr_with_parent(parent_descr.clone(), fd_info.offset);
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_op, value], field_descr.clone());
            // Bridge virtual rematerialisation — `upd.setfield(valuebox)`
            // parity: cache stores the Box identity (`value` OpRef).
            // Cache-hit readers resolve the intrinsic value via
            // `box_value(cached)` at hit time (covering const pool,
            // standard-virtualizable shadow, and the frontend object's
            // `value` field) — non-Const operands whose runtime concrete
            // was stamped at the original record site (or threaded from
            // the parent guard's fail_args via `set_opref_concrete`)
            // surface through that `value` field; unstamped operands
            // return `None` so the downstream sanity check skips.
            ctx.heapcache_setfield_cached(struct_op, fd_info.index, value);
        }
    }

    match entry.as_ref() {
        // resume.py:612-621 VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(size_descr) = descr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:619 decoder.allocate_with_vtable(descr=self.descr)
            let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:620 decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            // resume.py:621 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                size_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VirtualInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:628-637 VStructInfo.allocate
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(struct_descr) = typedescr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:635 decoder.allocate_struct(self.typedescr)
            let new_op = ctx.record_op_with_descr(OpCode::New, &[], struct_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:636 decoder.virtuals_cache.set_ptr(index, struct)
            cache.set_ptr(vidx, new_op);
            // resume.py:637 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                struct_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:649-671 AbstractVArrayInfo.allocate (clear=True or False)
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            fieldnums,
            kind,
            arraydescr,
            ..
        } => {
            let clear = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VArrayInfoClear { .. }
            );
            let kind = *kind;
            let length = fieldnums.len();
            let len_ref = ctx.const_int(length as i64);
            // resume.py:653 decoder.allocate_array(length, arraydescr, self.clear)
            let alloc_opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            // resume.py:645 AbstractVArrayInfo.__init__ asserts arraydescr is
            // not None; resume.py:652 allocate reads self.arraydescr directly.
            let array_descr = arraydescr.clone().expect("VArrayInfo: arraydescr is None");
            let new_op = ctx.record_op_with_descr(alloc_opcode, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:654 decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            // resume.py:656-670 element loop: dispatch by arraydescr kind
            // NB. the check for the kind of array elements is moved out of the loop
            let set_opcode = match kind {
                0 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_pointers()
                2 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_floats() — TODO: SetarrayitemRaw/Float
                _ => OpCode::SetarrayitemGc, // int
            };
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if value.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                // resume.py:660/665/670 setarrayitem_{ref,float,int}
                ctx.record_op_with_descr(
                    set_opcode,
                    &[new_op, idx_ref, value],
                    array_descr.clone(),
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayInfo(clear={}) → OpRef::from_raw({})",
                    vidx,
                    clear,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            fielddescrs,
            size,
            fieldnums,
            ..
        } => {
            let len_ref = ctx.const_int(*size as i64);
            // resume.py:749: array = decoder.allocate_array(self.size,
            // self.arraydescr, clear=True) — uses the live `self.arraydescr`
            // directly.
            let array_descr = arraydescr
                .as_ref()
                .expect("VArrayStructInfo: arraydescr is None");
            let new_op =
                ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:751: decoder.virtuals_cache.set_ptr(index, array)
            cache.set_ptr(vidx, new_op);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            // resume.py:752-759 reads exactly size × len(fielddescrs) entries
            // via self.fieldnums[p] with no length-equality check: a short
            // fieldnums is an out-of-bounds error here (IndexError parity), a
            // longer one leaves its tail unread.
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                    if value.is_none() {
                        continue;
                    }
                    let idx_ref = ctx.const_int(i as i64);
                    // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                    ctx.record_op_with_descr(
                        OpCode::SetinteriorfieldGc,
                        &[new_op, idx_ref, value],
                        fielddescrs[j].clone(),
                    );
                }
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayStructInfo → OpRef::from_raw({})",
                    vidx,
                    new_op.raw(),
                );
            }
            new_op
        }
        // resume.py:700-709 VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            // resume.py:703: buffer = decoder.allocate_raw_buffer(self.func, self.size)
            // resume.py:1124-1132: ResumeDataBoxReader.allocate_raw_buffer →
            //   execute_and_record_varargs(rop.CALL_I, [ConstInt(func), ConstInt(size)], calldescr)
            let func_ref = ctx.const_int(*func);
            let size_ref = ctx.const_int(*size as i64);
            // resume.py:1124-1126: calldescr comes from the shared
            // callinfocollection, not a freshly minted synthetic descr. The
            // func is NOT taken from callinfo_for_oopspec (resume.py:1127-1130:
            // several malloc variants share the oopspec), so only the calldescr
            // is read from the CIC and *func from the VRawBufferInfo is kept.
            let cic = ctx
                .callinfocollection
                .as_ref()
                .expect(
                    "TraceCtx.callinfocollection missing — bridge-virtual \
                     VRawBufferInfo materialization requires pyjitpl to populate \
                     it (resume.py:1124-1126)",
                )
                .clone();
            // resume.py:1126: calldescr, _ = cic.callinfo_for_oopspec(
            //   OS_RAW_MALLOC_VARSIZE_CHAR). callinfo_for_oopspec returns
            // (None, 0) on a missing entry (effectinfo.py:444-447) — no
            // lookup-time check; the calldescr is used directly.
            let (calldescr, _) =
                cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
            // resume.py:1131-1132: execute_and_record_varargs(CALL_I,
            //   [func, size], calldescr). A missing entry surfaces here, as the
            // calldescr is consumed by the CALL_I op, not as a separate lookup
            // assertion.
            let buffer = ctx.record_op_with_descr(
                OpCode::CallI,
                &[func_ref, size_ref],
                calldescr
                    .cloned()
                    .expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)"),
            );
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            // resume.py:705-708 iterate by len(self.offsets), indexing
            // self.descrs[i] and self.fieldnums[i] by the same i — a short
            // descrs/fieldnums raises IndexError here (encoder bug), a longer
            // one is ignored. No len-equality assert (VRawBufferInfo has none).
            for i in 0..offsets.len() {
                let off = offsets[i];
                let fnum = fieldnums[i];
                // resume.py:701-708 VRawBufferStateInfo.allocate_int passes
                // fieldnums[i] straight to setrawbuffer_item with no
                // UNINITIALIZED skip (unlike VArrayStructInfo) — a raw buffer
                // is fully written by the encoder.
                // resume.py:1232: itembox = self.decode_box(fieldnum, kind).
                // `decode_box` always returns a box (no UNINITIALIZED case),
                // so the store is unconditional, matching setrawbuffer_item.
                let item = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                // resume.py:1225-1234: setrawbuffer_item (direct reader).
                // Dispatches pointer/float/int via arraydescr — all types allowed.
                let di = &descrs[i];
                let tp = match di.item_type {
                    0 => majit_ir::Type::Ref,
                    2 => majit_ir::Type::Float,
                    _ => majit_ir::Type::Int,
                };
                let store_descr = crate::descr::make_array_descr(
                    di.base_size,
                    di.item_size,
                    di.len_offset,
                    tp,
                    di.is_signed,
                );
                let offset_ref = ctx.const_int(off as i64);
                ctx.record_op_with_descr(
                    OpCode::RawStore,
                    &[buffer, offset_ref, item],
                    store_descr,
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawBufferInfo(func={:#x}, size={}) → OpRef::from_raw({})",
                    vidx,
                    func,
                    size,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py:722-728 VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            // resume.py:725: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base_buffer = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            // resume.py:726: buffer = decoder.int_add_const(base_buffer, self.offset)
            let offset_ref = ctx.const_int(*offset as i64);
            let buffer = ctx.record_op(OpCode::IntAdd, &[base_buffer, offset_ref]);
            // resume.py:727: decoder.virtuals_cache.set_int(index, buffer)
            cache.set_int(vidx, buffer);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawSliceInfo(offset={}) → OpRef::from_raw({})",
                    vidx,
                    offset,
                    buffer.raw(),
                );
            }
            buffer
        }
        // resume.py:766-775 VStrPlainInfo.allocate / resume.py:820-829
        // VUniPlainInfo.allocate — `ResumeDataBoxReader.allocate_string /
        // allocate_unicode` followed by `string_setitem` / `unicode_setitem`
        // per character.
        //
        //     length = len(self.fieldnums)
        //     string = decoder.allocate_string(length)        # NEWSTR
        //     decoder.virtuals_cache.set_ptr(index, string)
        //     for i in range(length):
        //         charnum = self.fieldnums[i]
        //         if not tagged_eq(charnum, UNINITIALIZED):
        //             decoder.string_setitem(string, i, charnum)  # STRSETITEM
        //     return string
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
            );
            let length = fieldnums.len();
            let length_ref = ctx.const_int(length as i64);
            let (alloc_opcode, set_opcode) = if is_unicode {
                (OpCode::Newunicode, OpCode::Unicodesetitem)
            } else {
                (OpCode::Newstr, OpCode::Strsetitem)
            };
            // resume.py:769: string = decoder.allocate_string(length)
            let string = ctx.record_op(alloc_opcode, &[length_ref]);
            // resume.py:770: decoder.virtuals_cache.set_ptr(index, string)
            cache.set_ptr(vidx, string);
            // resume.py:771-774: string_setitem for each filled char.
            for (i, &charnum) in fieldnums.iter().enumerate() {
                if charnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py:1138-1141 ResumeDataBoxReader.string_setitem:
                //   charbox = self.decode_box(charnum, INT)
                //   execute_and_record(rop.STRSETITEM, string, ConstInt(index), charbox)
                let charbox = decode_fieldnum(ctx, charnum, rd_virtuals, resume_data, cache);
                if charbox.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                ctx.record_op(set_opcode, &[string, idx_ref, charbox]);
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}PlainInfo(length={}) → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    length,
                    string.raw(),
                );
            }
            string
        }
        // resume.py:785-793 VStrConcatInfo.allocate / resume.py:840-848
        // VUniConcatInfo.allocate:
        //
        //     left, right = self.fieldnums
        //     string = decoder.concat_strings(left, right)   # CALL_R(OS_STR_CONCAT)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.concat_strings` at resume.py:1143-1149:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_CONCAT)
        //     str1box = self.decode_box(str1num, REF)
        //     str2box = self.decode_box(str2num, REF)
        //     execute_and_record_varargs(CALL_R, [ConstInt(func), str1box, str2box], calldescr)
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
            );
            debug_assert_eq!(
                fieldnums.len(),
                2,
                "VStr/VUniConcatInfo must have exactly 2 fieldnums (left, right)"
            );
            let left = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let right = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniConcat
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrConcat
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[left, right]);
            cache.set_ptr(vidx, string);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}ConcatInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py:805-809 VStrSliceInfo.allocate / resume.py:860-864
        // VUniSliceInfo.allocate:
        //
        //     largerstr, start, length = self.fieldnums
        //     string = decoder.slice_string(largerstr, start, length)
        //     decoder.virtuals_cache.set_ptr(index, string)
        //
        // `ResumeDataBoxReader.slice_string` at resume.py:1151-1160 /
        // `slice_unicode` at resume.py:1179-1188:
        //
        //     cic = self.metainterp.staticdata.callinfocollection
        //     calldescr, func = cic.callinfo_for_oopspec(OS_STR_SLICE)
        //     strbox = self.decode_box(strnum, REF)
        //     startbox = self.decode_box(startnum, INT)
        //     lengthbox = self.decode_box(lengthnum, INT)
        //     stopbox = execute_and_record(INT_ADD, startbox, lengthbox)
        //     execute_and_record_varargs(CALL_R,
        //         [ConstInt(func), strbox, startbox, stopbox], calldescr)
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums, .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums, .. } => {
            let is_unicode = matches!(
                entry.as_ref(),
                majit_ir::RdVirtualInfo::VUniSliceInfo { .. }
            );
            debug_assert_eq!(
                fieldnums.len(),
                3,
                "VStr/VUniSliceInfo must have exactly 3 fieldnums (largerstr, start, length)"
            );
            let largerstr = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            let start = decode_fieldnum(ctx, fieldnums[1], rd_virtuals, resume_data, cache);
            let length = decode_fieldnum(ctx, fieldnums[2], rd_virtuals, resume_data, cache);
            // resume.py:1157-1158 / :1185-1186: stopbox = INT_ADD(startbox, lengthbox)
            let stop = ctx.record_op(OpCode::IntAdd, &[start, length]);
            let oopspec = if is_unicode {
                majit_ir::effectinfo::OopSpecIndex::UniSlice
            } else {
                majit_ir::effectinfo::OopSpecIndex::StrSlice
            };
            let string = emit_stroruni_oopspec_call(ctx, oopspec, &[largerstr, start, stop]);
            cache.set_ptr(vidx, string);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} V{}SliceInfo → OpRef::from_raw({})",
                    vidx,
                    if is_unicode { "Uni" } else { "Str" },
                    string.raw(),
                );
            }
            string
        }
        // resume.py:951/954 getvirtual_ptr direct-indexes rd_virtuals[index];
        // the function preamble already documents this as fail-loud ("not a
        // silent NONE fallback"). An Empty hole here means the resume stream
        // tagged a virtual index that was never assigned a real virtual
        // (encoder/decoder asymmetry) — surface it rather than poison the
        // operand chain with OpRef::NONE.
        majit_ir::RdVirtualInfo::Empty => panic!(
            "materialize_bridge_virtual: null rd_virtuals[{vidx}] (resume.py: null rd_virtuals[index])"
        ),
    }
}

impl JitState for PyreJitState {
    type Meta = PyreMeta;
    type Sym = PyreSym;
    type Env = PyreEnv;

    fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {
        let num_locals = self.local_count();
        let vsd = self.valuestackdepth();
        let slot_types = concrete_slot_types(self.frame, num_locals, vsd);
        // The valuestackdepth → heap array capacity flip is not used here
        // because it activates the broken VableExpansion path. The
        // `capacity` reference uses the pre-flip semantics:
        // array_capacity == self.array_capacity().
        let capacity = self.array_capacity();
        PyreMeta {
            num_locals,
            ns_len: self.namespace_len(),
            // Provisional seed only.  build_meta runs at trace START, before the
            // walk records any LOAD_GLOBAL/LOAD_NAME, so the true value does not
            // exist yet.  finish_trace_namespace_dependency overwrites this from
            // TraceCtx.reads_module_global on every trace_bytecode return path,
            // and the entry-bridge fold ORs the live flag mid-walk (`false` is
            // the OR identity).  Both invariants must hold for this seed to stay
            // safe to drop.
            namespace_dependent: false,
            valuestackdepth: vsd,
            array_capacity: capacity,
            // virtualizable_gen.rs:24-31 wires `extra_reds = { ec: Ref }` per
            // interp_jit.py:67 `reds = ['frame', 'ec']`, so the per-Sym
            // helpers (vable_collect_jump_args, NUM_EXTRA_REDS) already
            // thread the ec slot. The runtime flag here remains 0 because
            // pypy/module/pypyjit/interp_jit.py:67 PyPyJitDriver reds=
            // ['frame', 'ec']. With vable heap-writeback landed
            // (vable heap-writeback active behind the reduced-LABEL
            // gate), the descriptor activation chain is unblocked.
            // Cluster 2 flip — pairs with `driver_descriptor() = Some(...)`
            // below.
            trace_extra_reds: 1,
            has_virtualizable: self.has_virtualizable_info(),
            slot_types,
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.extract_live_values(_meta)
            .into_iter()
            .map(|value| match value {
                Value::Int(v) => v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect()
    }

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        if meta.trace_extra_reds == 1 {
            self.pypyjit_live_values_with_ec(meta)
        } else {
            self.expanded_virtualizable_live_values_with_extra_reds(meta, &[])
        }
    }

    fn close_loop_live_values(
        ctx: &majit_metainterp::TraceCtx,
        sym: &Self::Sym,
        _meta: &Self::Meta,
        live_arg_boxes: &[OpRef],
    ) -> Option<Vec<Value>> {
        let mut values = Vec::with_capacity(live_arg_boxes.len());
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let num_vable_scalars = crate::virtualizable_gen::NUM_VABLE_SCALARS;

        let frame_addr = if sym.live_vable_frame_addr != 0 {
            sym.live_vable_frame_addr
        } else {
            sym.concrete_vable_ptr as usize
        };
        let frame_value = if frame_addr != 0 {
            Value::Ref(majit_ir::GcRef(frame_addr))
        } else {
            ctx.box_value(*live_arg_boxes.first()?)?
        };
        values.push(frame_value);

        if crate::virtualizable_gen::NUM_EXTRA_REDS == 1 {
            let ec_addr = sym.concrete_execution_context as usize;
            let ec_value = if ec_addr != 0 {
                Value::Ref(majit_ir::GcRef(ec_addr))
            } else {
                ctx.box_value(*live_arg_boxes.get(1)?)?
            };
            values.push(ec_value);
        }

        let vable_start = 1 + crate::virtualizable_gen::NUM_EXTRA_REDS;
        for i in 0..num_vable_scalars {
            let slot = vable_start + i;
            let value = ctx
                .virtualizable_entry_at(i)
                .map(|(_, value)| value)
                .or_else(|| {
                    live_arg_boxes
                        .get(slot)
                        .and_then(|opref| ctx.box_value(*opref))
                })?;
            values.push(value);
        }

        let array_slots = live_arg_boxes.len().saturating_sub(num_scalars);
        for slot in 0..array_slots {
            let concrete = if slot < sym.nlocals {
                sym.concrete_locals
                    .get(slot)
                    .copied()
                    .unwrap_or(ConcreteValue::Ref(PY_NULL))
            } else {
                let stack_idx = slot - sym.nlocals;
                let live_stack = sym.valuestackdepth.saturating_sub(sym.nlocals);
                if stack_idx < live_stack {
                    sym.concrete_stack
                        .get(stack_idx)
                        .copied()
                        .unwrap_or(ConcreteValue::Ref(PY_NULL))
                } else {
                    ConcreteValue::Ref(PY_NULL)
                }
            };
            values.push(Value::Ref(majit_ir::GcRef(concrete.to_pyobj() as usize)));
        }

        Some(values)
    }

    // virtualizable.py:86 read_boxes() + warmstate.py:73 wrap() parity:
    // Array items (locals_cells_stack_w) are GC pointers → RefFrontendOp.
    // No pre-unboxing at function entry. Unboxing happens during tracing
    // via guard_class + getfield_gc_i when arithmetic/compare handlers
    // encounter Ref-typed operands.

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        if meta.trace_extra_reds == 1 {
            Self::pypyjit_live_value_types_with_ec(meta)
        } else {
            Self::expanded_virtualizable_live_value_types_with_extra_reds(meta, &[])
        }
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        if _meta.trace_extra_reds == 1 {
            Self::pypyjit_create_sym(_meta, _header_pc)
        } else {
            let mut sym = PyreSym::new_uninit(OpRef::input_arg_typed(0, Type::Ref));
            // `extra_reds = { ec: Ref }` in virtualizable_gen.rs places ec
            // at OpRef::from_raw(1). `init_vable_indices` shifts the vable static
            // field + array_base OpRefs but does not own the extra_red
            // sym storage; initialize it explicitly. NUM_EXTRA_REDS == 1 is
            // a crate-level const-assertion (see `lib.rs`).
            sym.execution_context = OpRef::input_arg_typed(1, Type::Ref);
            sym.become_active_vable_owner();
            sym.nlocals = _meta.num_locals;
            sym.valuestackdepth = _meta.valuestackdepth;
            // virtualizable.py:44 + interp_jit.py:25-31: all locals_cells_stack_w
            // items are W_Root → Type::Ref. Unboxing happens inside trace opcode
            // handlers (guard_class + getfield_gc_pure_i/_f), not at slot setup.
            sym.symbolic_local_types =
                vec![Type::Ref; _meta.num_locals.min(_meta.slot_types.len())];
            sym.symbolic_stack_types =
                vec![Type::Ref; _meta.slot_types.len().saturating_sub(_meta.num_locals)];
            let stack_only = _meta.vable_stack_only_depth();
            sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
            sym
        }
    }

    fn initialize_sym(&self, sym: &mut Self::Sym, _meta: &Self::Meta) {
        // jitdriver.rs:2947-2992 sequencing parity: `start_bridge_tracing`
        // calls `create_sym` → `initialize_sym` → `setup_bridge_sym` BEFORE
        // `trace_bytecode` (which is where `init_symbolic` would otherwise
        // populate `concrete_vable_ptr`). `setup_bridge_sym` reads
        // `sym.concrete_vable_ptr` to size the vable shadow via
        // `concrete_frame_array_len`; without this seed the pointer is
        // null, the helper falls back to `nlocals`, and stack pushes
        // beyond the first array slot panic at `set_virtualizable_entry_at`
        // on recursive benchmarks (fib_recursive). Mirror PyreJitState's
        // live frame pointer so the bridge seed sees the real
        // `locals_cells_stack_w` length.
        sym.concrete_vable_ptr = self.frame as *mut u8;
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        // pypy/module/pypyjit/interp_jit.py:67-70 PyPyJitDriver:
        //   reds = ['frame', 'ec']
        //   greens = ['next_instr', 'is_being_profiled', 'pycode']
        //   virtualizables = ['frame']
        //
        // History (now landed): the atomic flip needed a vable
        // heap-writeback pass that pyre's tracer originally did not
        // emit. Concretely,
        // `patch_new_loop_to_load_virtualizable_fields` (compile.py:425-461)
        // collapses the patched LABEL to `[reds]` and prepends a
        // GETFIELD_GC + GETARRAYITEM_GC preamble. The body then reads
        // every vable static field and locals_cells_stack_w slot from
        // the frame heap object on each iteration. pyre's tracer
        // updates symbolic state (`sym.vable_*`, `sym.registers_r[i]`)
        // but never emits the matching `SetfieldGc(frame, value, descr)`
        // or `SetarrayitemGc(array_ref, idx, value)` — only the
        // freshly-allocated W_IntObject.intval setfields appear in
        // trace dumps. With descriptor=None, this is fine because the
        // closing JUMP carries the post-loop state via expanded inputargs;
        // the patched-LABEL contract makes the heap the source of truth
        // and breaks that invariant.
        //
        // Prerequisites for the flip (all atomic with descriptor=Some):
        //   (a) `trace_extra_reds=1` (build_meta:3970) — emit
        //       `live_values=[frame, ec]` matching descriptor reds. The
        //       legacy `extract_live_values` shape relies on
        //       `descriptor=None`; flipping (a) alone breaks dynasm
        //       nested_loop / fannkuch / nbody (live_values consumers
        //       expect the expanded shape) and panics on
        //       `live_values[index>1]` access without (b).
        //   (b) `initialize_virtualizable` short-live_values gate
        //       (pyjitpl.rs:1744): allow the heap-read branch when
        //       `vable_ptr` is non-null. Required to consume (a)'s
        //       reds-only live_values.
        //   (c1) `pending_frontend_boxes.clone()` (pyjitpl.rs:7338):
        //        the second `compile_bridge` call needs the same stash
        //        as the first; current `take()` empties on the first
        //        call and the second hits `frontend_boxes.len()=0 vs
        //        liveboxes.len()=N`.
        //   (c2) Bridge JUMP arity vs patched LABEL: source emits the
        //        live-window shape (`num_scalars + valuestackdepth`) but
        //        the patched LABEL grows to full vable capacity. Const
        //        padding SSA-forwards `PY_NULL` into outer-loop locals
        //        on cranelift; heap-read padding via
        //        `trace_array_getitem_value` against
        //        `locals_cells_stack_array_ref` is the correct shape.
        //   (d) **Vable heap-writeback infrastructure** — *blocking*. With
        //       (a)+(b)+(c1) applied, cranelift nested_loop dispatches
        //       the bridge 5622+ times with identical inputs `[fr, ec,
        //       500, 2507475000]` because the patched parent reloads
        //       stale state from heap each iteration; the bridge body
        //       has no `SetfieldGc`/`SetarrayitemGc` to advance the
        //       heap. RPython's OptVirtualize emits these via its
        //       `force_at_end_of_preamble` pass; pyre's
        //       `OptVirtualize::force_virtualizable` (virtualize.rs:348)
        //       has the SETFIELD_RAW emission machinery but is not
        //       wired to fire at JUMP. An eager
        //       `gen_writeback_vable_to_heap` helper at
        //       `close_loop_args_at` / the intermediate merge point was
        //       tried, then removed once the hot-loop LABEL carried the
        //       live vable values as inputargs — the patched parent no
        //       longer reloads stale heap state each iteration, and a
        //       guard failure rebuilds the heap frame from resume-data
        //       (`consume_vref_and_vable_boxes` → `write_boxes`),
        //       matching RPython's no-eager-writeback model.
        //   (e) dynasm recursive CA frame contract — *blocking*
        //       for dynasm SIGSEGV at fib(24).
        //
        // Status: (a)+(b)+(c1)+(c2) have landed and descriptor is
        // active; (d)'s eager writeback was removed in favor of the
        // no-eager-writeback model (live values carried as inputargs,
        // heap frame rebuilt from resume-data). (e) dynasm recursive CA
        // frame contract remains a separate open item driving
        // fib_recursive on dynasm.
        Some(Self::pypyjit_driver_descriptor())
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // warmstate.py:503-511: RPython enters assembler unconditionally
        // when procedure_token exists. No next_instr check —
        // the compiled code's preamble handles entry from any PC.
        // Shape checks ensure the frame layout matches.  Namespace length is
        // load-bearing only for traces that read module globals: pure-compute
        // and builtin-only loops do not depend on later top-level binds, while
        // module-global reads still need this conservative gate because
        // same-key value rebinds are not value-guarded yet.
        self.local_count() == meta.num_locals
            && (!meta.namespace_dependent || self.namespace_len() == meta.ns_len)
    }

    fn setup_bridge_sym(
        sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        fail_values: &[i64],
        fail_types: &[Type],
    ) {
        let bridge_stamp_enabled = std::env::var("PYRE_FBW_BRIDGE_STAMP").as_deref() != Ok("0");
        if resume_data.frames.is_empty() {
            return;
        }

        // virtualizable.py:139 load_list_of_boxes parity: decode each
        // RebuiltValue in the resume stream into a typed Value. The type
        // is the fixed Box kind the encoder recorded at numbering time
        // (fail_arg_types[idx] for Box variants), matching RPython's
        // immutable Box.type invariant. No heap read.

        // resume.py:874-899 VirtualCache parity: single cache indexed by
        // virtual number, holding both symbolic OpRef (for trace ops) and
        // concrete GcRef (for shadow values / continue_tracing). RPython's
        // VirtualCache stores both in one object; pyre unifies them here.
        let mut virtuals_cache = BridgeVirtualCache::new(rd_virtuals.map_or(0, |v| v.len()));
        let (driver, _) = crate::driver::driver_pair();
        let backend = driver.meta_interp().backend();

        // resume.py:1245 decode_box parity — unified via bridge_decode_box.
        // Each call returns (OpRef, Value), eliminating the separate
        // resolve()/decode_concrete() paths so symbolic and concrete
        // materialization are always invoked together.

        let nlocals = sym.nlocals;
        // virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
        // items are declared Ref (W_Root array). Bridge resume slots stay
        // Ref at the virtualizable contract; any Int/Float unboxing must
        // happen inside trace opcode handlers, not at the inputarg level.
        let bridge_local_types = vec![Type::Ref; nlocals];

        // rebuild_state_after_failure (pyjitpl.py:3400-3437) keeps three
        // streams apart:
        //   virtualizable_boxes ← consume_virtualizable_boxes (vable mirror)
        //   registers_r/_i/_f    ← consume_boxes(f.get_current_position_info(),
        //                                        f.registers_i/_r/_f)
        //   virtualref_boxes     ← consume_virtualref_boxes
        // The majit decoder already splits rd_numb into the same three
        // streams (`resume_data.virtualizable_values`,
        // `resume_data.frames[*].values`, `resume_data.virtualref_values`).
        // This function consumes them in the same order and purpose.
        //
        // Part 1 — virtualizable payload (consume_virtualizable_boxes):
        // decode scalar header into sym.vable_* and capture the array-item
        // prefix for init_virtualizable_boxes below. Matches
        // virtualizable.py:86 `read_boxes` layout
        //   [vable_ptr, static_fields..., array_items...].
        let vvals = &resume_data.virtualizable_values;
        // Resume virtualizable payload mirrors RPython
        // opencoder.py:718-725 + virtualizable.py:139-154:
        //   [vable, vable_static_fields..., array_items...]
        // Non-vable extra reds (e.g. `ec`) are root inputargs, not part of
        // this payload, so the boundary is NUM_VABLE_SCALARS, not
        // NUM_SCALAR_INPUTARGS.
        let first_vable_scalar_idx = 1usize;
        let vable_array_start =
            first_vable_scalar_idx + crate::virtualizable_gen::NUM_VABLE_SCALARS;
        let mut oprefs: Vec<OpRef> = Vec::with_capacity(vvals.len());
        let mut concrete_values: Vec<majit_ir::Value> = Vec::with_capacity(vvals.len());
        // resume.py:1264 `assert box.type == kind`: the vable payload is
        // NOT uniformly Ref — the static fields carry their declared
        // kinds (interp_jit.py:25-31: last_instr/valuestackdepth are Int).
        // `virt_live_value_types` yields the full live layout WITH the
        // extra reds ([frame, <NUM_EXTRA_REDS>, <NUM_VABLE_SCALARS>,
        // array...]); the vvals stream omits the extra reds, so strip them
        // to recover the per-slot kind for each payload position.
        let array_item_count = vvals
            .len()
            .saturating_sub(1 + crate::virtualizable_gen::NUM_VABLE_SCALARS);
        let full_types = crate::virtualizable_gen::virt_live_value_types(array_item_count);
        let nreds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        let mut vvals_types: Vec<Type> = Vec::with_capacity(vvals.len());
        vvals_types.push(full_types.first().copied().unwrap_or(Type::Ref));
        vvals_types.extend_from_slice(&full_types[(1 + nreds).min(full_types.len())..]);
        for (idx, v) in vvals.iter().enumerate() {
            let expected_kind = vvals_types.get(idx).copied().unwrap_or(Type::Ref);
            let (op, val) = bridge_decode_box(
                ctx,
                v,
                expected_kind,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            oprefs.push(op);
            concrete_values.push(val);
        }
        sym.restore_inputarg_oprefs(&oprefs, first_vable_scalar_idx);
        let vable_ref_value = concrete_values
            .first()
            .copied()
            .unwrap_or(majit_ir::Value::Void);
        let vable_scalar_values: Vec<majit_ir::Value> = concrete_values
            .iter()
            .skip(first_vable_scalar_idx)
            .take(crate::virtualizable_gen::NUM_VABLE_SCALARS)
            .copied()
            .collect();
        let vable_array_items: Vec<OpRef> =
            oprefs.iter().skip(vable_array_start).copied().collect();
        let vable_array_values: Vec<majit_ir::Value> = concrete_values
            .iter()
            .skip(vable_array_start)
            .copied()
            .collect();
        let bridge_valuestackdepth = concrete_values
            // virtualizable_values has no ec red: [vable, last_instr,
            // pycode, valuestackdepth, debugdata, lastblock, w_globals, ...].
            .get(first_vable_scalar_idx + 2)
            .map(value_to_usize)
            .unwrap_or(sym.valuestackdepth)
            .max(nlocals);

        // Part 2 — frame registers (consume_boxes): walk the frame section
        // in liveness enumeration order ([int..., ref..., float...]), keep
        // each bank's register indices separate via
        // `frame_liveness_reg_indices_by_bank_at`, and write each decoded
        // value into the corresponding MIFrame register bank. This mirrors
        // resume.py:1054 `consume_boxes`
        // (`_callback_i/_r/_f(register_index)` writing to
        // `f.registers_i/_r/_f[index]` at the exact slot liveness
        // declared, not at an enumerate-order position). RPython indexes
        // a single `registers_X` vector by abstract register color —
        // there is no `idx < nlocals` decode.
        let frame0 = &resume_data.frames[0];
        let reg_indices = crate::state::frame_liveness_reg_indices_by_bank_from_pc(
            frame0.jitcode_index,
            frame0.pc,
        );
        // For a kept-stack branch guard, the vable's
        // `valuestackdepth` may reflect the merge-target depth (consumed
        // stack) rather than the guard's deeper live depth. The guard PC's
        // pcdep `stack_depth_at_pc` (from `depth_at_py_pc`, resolved through
        // the carried `jitcode_pc`) IS the guard-time depth. Use the larger
        // of the two so the color→slot inversion covers the kept temps.
        // This is deferred until after `maps` is read (below) via a
        // re-adjustment of the semantic mirror length.
        let stack_only = bridge_valuestackdepth.saturating_sub(nlocals);
        let bridge_reg_len = nlocals + stack_only;
        let mut bridge_registers_r = vec![OpRef::NONE; bridge_reg_len];
        // RPython parity: after A.1 the guard-recovery path calls
        // `synchronize_virtualizable()` / `write_boxes()`
        // (pyjitpl.py:3430) before `start_bridge_tracing`, so the
        // physical vable image the tracer is about to read is already
        // resume-data-complete. The
        // bridge register file is therefore expected to be fully
        // populated by the liveness-driven zip below; any remaining
        // OpRef::NONE signals a liveness-coverage gap (the tracer keeps a
        // local live past the `-live-` marker) and must be surfaced by
        // the assert rather than papered over with a vable-mirror read.
        assert!(
            reg_indices.total_len() == frame0.values.len(),
            "setup_bridge_sym: reg_indices len={} != frame.values len={} at pc={}",
            reg_indices.total_len(),
            frame0.values.len(),
            frame0.pc,
        );
        // gap-10 GotoIfNotValueNotConcrete (bridge sub-class): the resume
        // data carries the concrete runtime value for every live frame
        // register, but the consume_boxes loops below keep only the OpRef
        // and DROP the concrete — so a bridge resume leaves loop-carried
        // locals symbolic and a data-dependent branch derived from one
        // (`(i%7) and ...` → TO_BOOL → goto_if_not) can't fold its
        // direction, aborting the walk.  Stamp each decoded
        // concrete onto its OpRef so the symbolic walk folds the branch
        // per the actual failing-iteration path (the iteration the
        // compiled trace re-runs), emitting a real
        // GuardTrue/GuardFalse — orthodox meta-tracing ("trace the
        // concrete path, guard it"; the IR keeps the symbolic InputArg,
        // the concrete is a trace-time shadow only, so the optimizer does
        // NOT const-fold the loop-variant value).  Default-ON (`=0` opts
        // out) once validated, mirroring the LoadGlobal-fold / multiframe
        // gap-10 flips; the opt-out keeps the prior symbolic-bridge
        // behavior available for A/B.
        let seed_bridge_locals = std::env::var("PYRE_FBW_BRIDGE_LOCAL_SEED").as_deref() != Ok("0");
        // For kept-stack branch guards the body-internal marker's
        // liveness colors do not 1:1-correspond to semantic slots — a
        // color that lives a temp at the body marker may name a different
        // slot at the guard PC.  Seeding concrete values at the
        // consume_boxes stage (color-indexed) stamps the wrong value onto
        // the OpRef, causing downstream branch folds to take the wrong
        // direction.  Defer seeding to the post-overlay stage where the
        // mirror is slot-indexed and values are authoritative.
        let seed_deferred_to_overlay =
            crate::state::frame_pc_is_resolved_offset_at(frame0.jitcode_index, frame0.pc);
        let mut bridge_stamp_orphans =
            (bridge_stamp_enabled && seed_deferred_to_overlay).then(Vec::new);
        let mut value_cursor = 0usize;
        for &reg_idx in &reg_indices.int {
            let value = &frame0.values[value_cursor];
            let (resolved, concrete_val) = bridge_decode_box(
                ctx,
                value,
                Type::Int,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            // The Int bank is color-indexed (no slot overlay to defer to), and
            // for a kept-stack branch guard the walk resumes at the guard's own
            // coordinate where `reg_indices.int` colors are authoritative — so
            // stamp the concrete directly here (the `seed_deferred_to_overlay`
            // deferral only applies to the Ref slot-mirror). The guard's
            // GUARD_TRUE/GUARD_FALSE needs this concrete to fold its direction.
            if seed_bridge_locals && !matches!(concrete_val, majit_ir::Value::Void) {
                ctx.try_set_opref_concrete(resolved, concrete_val);
            }
            let reg_idx = reg_idx as usize;
            if reg_idx >= sym.registers_i.len() {
                sym.registers_i.resize(reg_idx + 1, OpRef::NONE);
            }
            sym.registers_i[reg_idx] = resolved;
            value_cursor += 1;
        }
        for &reg_idx in &reg_indices.ref_ {
            let value = &frame0.values[value_cursor];
            let (resolved, concrete_val) = bridge_decode_box(
                ctx,
                value,
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            if seed_bridge_locals
                && !seed_deferred_to_overlay
                && !matches!(concrete_val, majit_ir::Value::Void)
            {
                ctx.try_set_opref_concrete(resolved, concrete_val);
            }
            if let Some(orphan_stamps) = bridge_stamp_orphans.as_mut() {
                orphan_stamps.push((resolved, concrete_val));
            }
            let reg_idx = reg_idx as usize;
            if reg_idx >= bridge_registers_r.len() {
                bridge_registers_r.resize(reg_idx + 1, OpRef::NONE);
            }
            bridge_registers_r[reg_idx] = resolved;
            value_cursor += 1;
        }
        for &reg_idx in &reg_indices.float {
            let value = &frame0.values[value_cursor];
            let (resolved, concrete_val) = bridge_decode_box(
                ctx,
                value,
                Type::Float,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            if seed_bridge_locals
                && !seed_deferred_to_overlay
                && !matches!(concrete_val, majit_ir::Value::Void)
            {
                ctx.try_set_opref_concrete(resolved, concrete_val);
            }
            let reg_idx = reg_idx as usize;
            if reg_idx >= sym.registers_f.len() {
                sym.registers_f.resize(reg_idx + 1, OpRef::NONE);
            }
            sym.registers_f[reg_idx] = resolved;
            value_cursor += 1;
        }
        // Reconstruct the slot-indexed semantic register file
        // (`[locals.., stack_tail..]`) from the color-indexed resume decode.
        // The decode just filled `bridge_registers_r` by abstract-register
        // color (`reg_indices.ref_`); the bridge trace, however, reads
        // `sym.registers_r` and the kept-stack/local oprefs by SEMANTIC slot
        // (LOAD_FAST `registers_r[var_num]`, stack `nlocals + depth`). When
        // the per-CodeObject regalloc colors a local/stack slot at a color
        // other than its slot index, a slot-indexed read of the color bank
        // returns a foreign value (a dead temp, a portal red, or a constant)
        // — a corruption the codewriter used to mask by pinning the
        // `[0,nlocals)` prefix to identity colors (now retired). Invert each
        // live color to its slot via `semantic_ref_slot_for_reg_color` so the
        // mirror is correct under freely-colored locals.
        let maps = crate::state::bridge_semantic_maps_from_pc(frame0.jitcode_index, frame0.pc);
        // For a kept-stack branch guard, the vable's runtime
        // `valuestackdepth` reflects the merge-target depth (post
        // consumption) rather than the guard's deeper live depth. The
        // guard-PC pcdep `stack_depth_at_pc` IS the guard-time depth
        // (with kept temps live). Widen `semantic_prefix_len` to the
        // pcdep depth so the color→slot inversion covers kept temps.
        let stack_only = stack_only.max(maps.stack_depth_at_pc);
        let semantic_prefix_len = nlocals + stack_only;
        // Extend bridge_registers_r to cover the wider depth.
        if bridge_registers_r.len() < semantic_prefix_len {
            bridge_registers_r.resize(semantic_prefix_len, OpRef::NONE);
        }
        if majit_metainterp::majit_log_enabled()
            && crate::state::frame_pc_is_resolved_offset_at(frame0.jitcode_index, frame0.pc)
        {
            let old_so = bridge_valuestackdepth.saturating_sub(nlocals);
            eprintln!(
                "[jit][kept-stack-bridge] jitcode_pc={} pc={} pcdep={:?} \
                 depth_at_guard={} vsd={} stack_only={}→{}",
                frame0.pc,
                frame0.pc,
                maps.pcdep_entries,
                maps.stack_depth_at_pc,
                bridge_valuestackdepth,
                old_so,
                stack_only,
            );
        }
        // virtualizable.py:86-98 + pyjitpl.py:3430 synchronize_virtualizable:
        // the frame's `locals_cells_stack_w` array (the vable image) is the
        // authoritative post-guard source for the frame's locals. At an
        // arbitrary interior resume pc a local slot's jitcode color may hold
        // a dead temp that decodes to a NULL constant (`ConstPtr(GcRef(0))`);
        // such a dead-temp NULL must not shadow the vable's live local, so for
        // local slots prefer the vable array item over a NONE/null-const value.
        // Stack slots keep the NONE-only fallback (a real stack value must not
        // be overwritten).
        //
        // GC-rooted concrete source for the per-local stamp below: read the
        // live virtualizable frame's slots directly (post the ref/int/float
        // decode loops above, which allocate and may trigger a minor GC) so
        // the stamp never points at a stale off-heap `vable_array_values`
        // copy whose `Ref`s a collection has since moved.  See
        // `live_frame_array_values`.
        let live_local_values = live_frame_array_values(
            sym.concrete_vable_ptr as usize,
            usize::MAX,
            &vable_array_values,
        );
        let mut overlay_local = |slot: &mut OpRef, s: usize| {
            let slot_is_null_const = matches!(*slot, OpRef::ConstPtr(v) if v.0 == 0);
            if slot.is_none() || slot_is_null_const {
                if let Some(v) = vable_array_items.get(s).copied() {
                    if !v.is_none() {
                        *slot = v;
                        // A local resolved from the vable image: stamp its
                        // concrete from the GC-rooted live frame slot
                        // (`live_local_values`, not the off-heap decoded
                        // array) so the seeded bridge walk can fold a branch
                        // derived from it without risking a moved-pointer
                        // stamp (gap-10 bridge sub-class; see seed note above).
                        if seed_bridge_locals {
                            if let Some(&cv) = live_local_values.get(s) {
                                if !matches!(cv, majit_ir::Value::Void) {
                                    ctx.try_set_opref_concrete(v, cv);
                                }
                            }
                        }
                    } else if slot.is_none() {
                        *slot = OpRef::NONE;
                    }
                }
            }
        };
        let semantic_mirror: Vec<OpRef> = if !maps.has_color_map {
            // No per-CodeObject regalloc: colors are slot-identity, so the
            // color bank IS the slot mirror over the semantic prefix. Keep the
            // in-place identity overlay (stack slots NONE-only, locals vable).
            // `!has_color_map` (empty `pcdep_color_slots`) is the field-free
            // successor to the flat `local/stack_color_map.is_empty()` guard, so
            // a zero-local frame that still owns a freely-colored operand stack
            // (non-empty `pcdep_color_slots`) falls to the else branch (per-slot
            // inversion) instead of reading the color-indexed bank as if it were
            // slot-indexed.
            for (idx, slot) in bridge_registers_r
                .iter_mut()
                .enumerate()
                .take(semantic_prefix_len)
            {
                let is_local = idx < nlocals;
                let slot_is_null_const = matches!(*slot, OpRef::ConstPtr(v) if v.0 == 0);
                let want_vable = slot.is_none() || (is_local && slot_is_null_const);
                if want_vable {
                    if let Some(v) = vable_array_items.get(idx).copied() {
                        if !v.is_none() {
                            *slot = v;
                        } else if slot.is_none() {
                            *slot = OpRef::NONE;
                        }
                    }
                }
            }
            bridge_registers_r
                .iter()
                .take(semantic_prefix_len)
                .copied()
                .collect()
        } else {
            // Per-CodeObject: fill each live local/stack slot from its color.
            let mut mirror = vec![OpRef::NONE; semantic_prefix_len];
            // #348 Part (2): per-PC slot fill. `pcdep_entries` maps each live
            // slot to its TRUE per-program-point color, so drive the fill by
            // SLOT (not color): write `mirror[slot] = bridge_registers_r[color]`
            // for every entry. A color shared by multiple slots — an aliased
            // `DUP_TOP` / `ROT_THREE` operand-stack pair, or a local aliased
            // onto the stack — writes its single value into EVERY slot it
            // covers. A prior color→slot inversion kept only one slot per
            // color (stack-first tie-break), leaving the sibling aliased slot
            // `OpRef::NONE`; the vable image is also NULL for pure trace temps
            // (`kept_stack_branch_depths` `0 < a < b < 9` keeps two copies of
            // the same compare operand across the guard), so that slot folded
            // to concrete `GcRef(0)` and the residual declined. Out-of-prefix
            // slots are dropped by the `s >= semantic_prefix_len` guard.  #73:
            // pcdep is the SOLE color→slot source here; the flat
            // `local_color_map` / `stack_color_map` fallback is drained.
            for &(bank, color, slot) in &maps.pcdep_entries {
                // Only Ref-bank colors map to bridge_registers_r slots.
                // Int/Float bank entries are structurally recorded but
                // currently unreachable (operand stack is always Ref).
                if bank != 1 {
                    continue;
                }
                let s = slot as usize;
                if s >= semantic_prefix_len {
                    continue;
                }
                let col = color as usize;
                if col < bridge_registers_r.len() {
                    mirror[s] = bridge_registers_r[col];
                }
            }
            // pcdep-totality guard (#73): the drained flat-else previously
            // inverted live operand-stack slots from `stack_color_map` when
            // `pcdep_entries` was empty.  The corpus proof
            // (`validate_pcdep_color_map`, injective + total) shows an empty
            // `pcdep_entries` here carries no live operand stack — locals
            // still refill from the vable image via `overlay_local` below.
            // Fail loud under `PYRE_PCDEP_VALIDATE` if a live stack slot ever
            // reaches here uncovered (a totality regression).
            if maps.pcdep_entries.is_empty() && std::env::var_os("PYRE_PCDEP_VALIDATE").is_some() {
                // `stack_depth_at_pc` never exceeds `max_stackdepth` (the retired
                // `stack_color_map.len()` clamp), so the runtime `stack_only`
                // bound is the only one that matters.
                let live_stack = maps.stack_depth_at_pc.min(stack_only);
                if live_stack > 0 {
                    eprintln!(
                        "PCDEP-TOTALITY-VIOLATION: empty pcdep_entries with \
                         {live_stack} live stack slot(s) at bridge resume \
                         (jitcode_index={}, pc={})",
                        frame0.jitcode_index, frame0.pc
                    );
                }
            }
            for s in 0..nlocals {
                overlay_local(&mut mirror[s], s);
            }
            // Pcdep-live kept-stack colors absent from the body marker's
            // liveness leave their stack slots NONE after the color→slot
            // inversion above.  The vable image (`locals_cells_stack_w`)
            // is authoritative post-guard, so fill NONE stack slots from
            // it — symmetric to the local overlay.
            for s in nlocals..semantic_prefix_len.min(mirror.len()) {
                if mirror[s].is_none() {
                    if let Some(v) = vable_array_items.get(s).copied() {
                        if !v.is_none() {
                            mirror[s] = v;
                            if seed_bridge_locals {
                                if let Some(&cv) = live_local_values.get(s) {
                                    if !matches!(cv, majit_ir::Value::Void) {
                                        ctx.try_set_opref_concrete(v, cv);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            mirror
        };
        let bridge_locals: Vec<OpRef> = semantic_mirror.iter().take(nlocals).copied().collect();
        // #124 kept operand-stack temps: the stack tail of the same
        // slot-indexed mirror (`[nlocals, nlocals + stack_only)`), so a
        // resumed conditional-expression / short-circuit value survives
        // init_symbolic's later NONE reset regardless of its abstract color.
        let bridge_stack: Vec<OpRef> = semantic_mirror
            .iter()
            .skip(nlocals)
            .take(stack_only)
            .copied()
            .collect();

        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-sym] frames[0].values={} reg_indices={:?} \
                 bridge_locals={:?} vable_array_items={:?}",
                frame0.values.len(),
                reg_indices,
                bridge_locals,
                vable_array_items,
            );
        }
        // Override sym.registers_r so subsequent LOAD_FAST sees the
        // bridge inputarg OpRefs, not the parent's vable_array_base+i
        // OpRefs that init_symbolic seeded before setup_bridge_sym ran.
        //
        // pyre's start_bridge_tracing calls initialize_sym() (which runs
        // init_symbolic) BEFORE setup_bridge_sym, so init_symbolic sees
        // bridge_local_oprefs == None and falls into the vable_array_base
        // branch (init_vable_indices hard-codes vable_array_base = 7 for
        // pyre's 7-slot virtualizable header). That branch produces
        // OpRef::from_raw(base+i) values from the PARENT trace's namespace, leaving
        // stale parent OpRefs in registers_r after we set
        // bridge_local_oprefs here.
        //
        // virtualizable.py:44 + interp_jit.py:25-31: array item types are
        // all Ref; RETURN_VALUE / arithmetic paths unbox via
        // `trace_guarded_int_payload` (guard_class + getfield_gc_pure_i),
        // matching the RPython unbox-at-consumer model. The slot-level
        // type override is NOT how RPython avoids the guarded path.
        //
        // `semantic_mirror` is slot-indexed (`[locals.., stack_tail..]`),
        // matching what the trace-time mirror reads expect — NOT the
        // color-indexed `bridge_registers_r`. Portal reds (frame/ec) live in
        // their dedicated `sym` fields, so they are absent here by design.
        //
        // Deferred concrete seeding for kept-stack branch guards.
        // At the consume_boxes stage the register bank is color-indexed,
        // and the body marker's colors may not correspond 1:1 to semantic
        // slots — seeding there stamps the wrong value.  After the
        // overlay the mirror is slot-indexed and authoritative, so seed
        // each non-NONE slot from the GC-rooted live frame values.
        if seed_bridge_locals && seed_deferred_to_overlay {
            for (s, opref) in semantic_mirror.iter().enumerate() {
                if !opref.is_none() {
                    if let Some(&cv) = live_local_values.get(s) {
                        // Skip a NULL (`GcRef(0)`) source: an operand-stack slot
                        // above the frame's materialized `valuestackdepth` reads
                        // NULL from `locals_cells_stack_w` because the kept temp
                        // lives in the guard's register file / resume data, not
                        // the frame array. When a color aliases two slots
                        // (`DUP_TOP` / `ROT_THREE`), one slot may carry the real
                        // value and its sibling the NULL hole; both stamp the
                        // SAME opref, so a NULL stamp would clobber the real one
                        // (last write wins) and fold the residual's Ref arg to
                        // concrete NULL → `MayForceNullRefArgUnsupported`. A real
                        // frame Ref is never NULL here, so skipping NULL only
                        // drops the unmaterialized-hole case.
                        if !matches!(
                            cv,
                            majit_ir::Value::Void | majit_ir::Value::Ref(majit_ir::GcRef(0))
                        ) {
                            ctx.try_set_opref_concrete(*opref, cv);
                        }
                    }
                }
            }
        }
        // Orphaned colors have decoded values but no semantic-mirror slot to
        // re-stamp; seed them so bridge-walk residuals can resolve arguments.
        if let Some(orphan_stamps) = bridge_stamp_orphans.as_ref() {
            for (opref, cv) in orphan_stamps {
                if opref.is_none() || semantic_mirror.contains(opref) {
                    continue;
                }
                if !matches!(
                    cv,
                    majit_ir::Value::Void | majit_ir::Value::Ref(majit_ir::GcRef(0))
                ) {
                    ctx.try_set_opref_concrete(*opref, *cv);
                }
            }
        }
        sym.registers_r = semantic_mirror;
        sym.symbolic_local_types = {
            let mut types = bridge_local_types.clone();
            types.resize(sym.nlocals, Type::Ref);
            types
        };
        // The bridge inputs do NOT have the 7-slot scalar header that
        // init_vable_indices assumes. Demote this frame from active
        // virtualizable owner so any later LOAD_FAST falling through to
        // the vable_array_base branch uses the heap-array path instead
        // of synthesizing parent OpRefs.
        sym.clear_active_vable();
        // `pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`:
        // ec is a portal red arg, hence a JitCode inputarg present in
        // every `-live-` op's R-bank. Because the codewriter
        // (jit/codewriter.rs:2364 `filter_liveness_in_place`) seeds
        // `portal_ec_reg` into `lv_live`, bridge resume's liveness-
        // driven `consume_boxes` fill at lines 4880-4893 has already
        // written the resolved ec OpRef into
        // `bridge_registers_r[portal_ec_reg]` — the same slot
        // `_callback_r(register_index)` (resume.py:1077-1081) writes
        // in RPython's BH register bank.
        //
        // Thread that slot back into the dedicated `sym.execution_context`
        // field so the symbolic state retains pyre's split shape (ec
        // accessed through a named field rather than by register index).
        // The OpRef value is identical to what RPython's ec inputarg
        // OpRef would be after resume.
        // `portal_ec_reg` is an abstract-register COLOR with no semantic
        // `locals_cells_stack_w` slot (it is a portal red, not a local/stack
        // value), so read it from the color-indexed decode `bridge_registers_r`
        // — `sym.registers_r` is now the slot-indexed mirror and does not carry
        // portal-red colors.
        //
        // At PCs where the register allocator reuses the ec color for a real
        // frame slot (a call result live across a later call), the snapshot
        // encoder recorded the SLOT value in this register, not the ec
        // (`collect_outer_active_boxes` portal-red scratch gate); seeding
        // `sym.execution_context` from it would hand a frame value to every
        // downstream ec consumer.  Detect the collision through the same
        // per-PC color→slot table the encoder consulted and leave ec to the
        // `ensure_execution_context` frame-field recovery instead.
        let (_pfr, portal_ec_reg) = crate::state::portal_red_regs_at(frame0.jitcode_index);
        if portal_ec_reg != u16::MAX {
            let ec_color_names_frame_slot = crate::state::pcdep_color_names_frame_slot_at(
                frame0.jitcode_index,
                frame0.pc as usize,
                portal_ec_reg,
            );
            if !ec_color_names_frame_slot {
                let slot = portal_ec_reg as usize;
                assert!(
                    slot < bridge_registers_r.len(),
                    "setup_bridge_sym: portal_ec_reg={} out of bridge_registers_r range (len={})",
                    slot,
                    bridge_registers_r.len(),
                );
                sym.execution_context = bridge_registers_r[slot];
            }
        }
        // pyjitpl.py:3400-3430 rebuild_state_after_failure parity: after
        // a guard failure the tracing-time `virtualizable_boxes` mirror
        // must be rebuilt from the resume data so subsequent vable
        // ops see OpRefs drawn from the bridge's inputarg stream, not
        // the parent loop's vable_array_base+i indices that
        // init_symbolic seeded before setup_bridge_sym ran.
        //
        // Layout mirrors virtualizable.py:86-98 read_boxes():
        //   boxes[0..NUM_SCALARS-1] = scalar fields 1..NUM_SCALARS
        //     (vable_last_instr, vable_pycode, vable_valuestackdepth,
        //      vable_debugdata, vable_lastblock, vable_w_globals)
        //   boxes[NUM_SCALARS-1..NUM_SCALARS-1+array_len] = array items
        //     (bridge_locals followed by reserved stack slots)
        //   boxes[-1] = vable identity (sym.frame)
        // pyframe.py:107-110 `locals_cells_stack_w` length =
        // `nlocals + ncells + max_stack`. Pad beyond the bridge's live
        // local prefix with a shared const-NULL OpRef so every
        // interpreter-visible slot has a tracing-time mirror (matches
        // the portal path above).
        // pyframe.py:107-110 + pyjitpl.py:3437: the virtualizable shape
        // committed at portal-entry time (`nlocals + ncells +
        // max_stackdepth` array slots) does not change at guard-failure
        // resume; `rebuild_state_after_failure` writes the resume blob
        // into the same `virtualizable_boxes` layout the portal seeded.
        // pyre's root portal seeds via `initialize_virtualizable` with
        // exactly that full layout, so the bridge entry must match it
        // here. Earlier `unwrap_or(nlocals)` undersized the shadow when
        // `concrete_frame_array_len` returned None (e.g. when
        // `sym.concrete_vable_ptr` had not yet been bound at
        // setup_bridge_sym time), causing pushes past `nlocals`/the
        // local prefix to panic at `set_virtualizable_entry_at: index N
        // out of range for N slots`. A probe captured the
        // mismatch directly: root portal sized `vable_boxes_len=25`
        // (= 6 + 18 + 1) but a fannkuch bridge fell back to
        // `bridge_array_len=14` → `vable_boxes_len=21`, then pushed
        // `flat_idx=21` and panicked. Fall back to the metadata-derived
        // size — `metadata.stack_base + metadata.max_stackdepth` is the
        // same `nlocals + ncells + max_stackdepth` the codewriter
        // committed to and the runtime PyFrame allocates (pyframe.rs:1576).
        let bridge_array_len = concrete_frame_array_len(sym.concrete_vable_ptr as usize)
            .or_else(|| {
                METAINTERP_SD.with(|r| {
                    let sd = r.borrow();
                    sd.jitcodes.get(frame0.jitcode_index as usize).map(|jc| {
                        jc.payload.metadata.stack_base + jc.payload.metadata.max_stackdepth
                    })
                })
            })
            .unwrap_or(nlocals);
        let scalar_oprefs = [
            sym.vable_last_instr,
            sym.vable_pycode,
            sym.vable_valuestackdepth,
            sym.vable_debugdata,
            sym.vable_lastblock,
            sym.vable_w_globals,
        ];
        // virtualizable.py:139 load_list_of_boxes parity: the OpRef half of
        // virtualizable_boxes comes from the resume-data stream
        // (`bridge_decode_box`). The CONCRETE array shadow, however, is read
        // from the restored live virtualizable — NOT from the resume-decoded
        // `vable_array_values`. Those decoded values are off-heap copies of
        // young GC pointers captured at guard-fail time; a minor collection
        // during bridge setup (residual virtual materialization / op recording
        // allocates) moves the referenced objects but cannot forward the
        // off-heap decode Vec, leaving dangling pointers that crash the walk's
        // getarrayitem_vable. The live frame (`sym.concrete_vable_ptr`,
        // restored by `decode_and_restore_guard_failure`) sits on the
        // CURRENT_FRAME chain, so `walk_pyframe_roots` forwards its
        // `locals_cells_stack_w` items on every collection — its slots are
        // always live. Read them directly, mirroring the root-trace seed
        // (`read_all_boxes` from the rooted portal frame). Falls back to the
        // decoded values only when no live pointer is bound (unit-test /
        // init-before-run). The seed helper pads short arrays with const-NULL
        // OpRef; match that here by padding concrete values with
        // Value::Ref(GcRef::NULL) to the same length.
        let live_array_values = live_frame_array_values(
            sym.concrete_vable_ptr as usize,
            bridge_array_len,
            &vable_array_values,
        );
        sym.concrete_locals = (0..nlocals)
            .map(|i| {
                live_array_values
                    .get(i)
                    .copied()
                    .map(concrete_value_from_ir_value)
                    .unwrap_or(ConcreteValue::Ref(pyre_object::PY_NULL))
            })
            .collect();
        sym.concrete_stack = (0..stack_only)
            .map(|i| {
                live_array_values
                    .get(nlocals + i)
                    .copied()
                    .map(concrete_value_from_ir_value)
                    .unwrap_or(ConcreteValue::Ref(pyre_object::PY_NULL))
            })
            .collect();
        let mut concrete_values = Vec::with_capacity(vable_scalar_values.len() + bridge_array_len);
        concrete_values.extend_from_slice(&vable_scalar_values);
        let taken_concrete = live_array_values.len().min(bridge_array_len);
        concrete_values.extend_from_slice(&live_array_values[..taken_concrete]);
        while concrete_values.len() < vable_scalar_values.len() + bridge_array_len {
            concrete_values.push(majit_ir::Value::Ref(majit_ir::GcRef::NULL));
        }
        crate::state::seed_virtualizable_boxes(
            ctx,
            sym.frame,
            vable_ref_value,
            &scalar_oprefs,
            &vable_array_items,
            bridge_array_len,
            &concrete_values,
            sym.concrete_vable_ptr as *const u8,
        );
        // resume.py:1042-1057 `rebuild_from_resumedata` parity: bridge
        // tracing resumes from the full restored frame state via the
        // vable scalar reads + consume_boxes — `valuestackdepth` is
        // recovered from the decoded virtualizable resume payload
        // (`bridge_valuestackdepth` derived from vable.valuestackdepth), not
        // hardcoded to `nlocals`. Keep the stack tail in the unified
        // register file and expose Ref-typed virtualizable slots to
        // subsequent LOAD_FAST / close_loop_args_at calls.
        sym.symbolic_stack_types = vec![Type::Ref; stack_only];
        sym.valuestackdepth = bridge_valuestackdepth;
        sym.bridge_local_oprefs = Some(bridge_locals);
        // #124: preserve the resolved kept operand-stack temps. `bridge_stack`
        // is the stack tail of the slot-indexed `semantic_mirror` (computed
        // above by inverting each live color to its slot), so it is correct
        // even when the stack slot's abstract color is not `nlocals + depth`.
        // init_symbolic runs AFTER setup_bridge_sym in pyre's bridge launcher
        // and would otherwise reset this tail to NONE; keeping it lets both the
        // rebuilt registers_r and the full-body-walk argbox seed recover the
        // kept conditional-expression / short-circuit value.
        sym.bridge_stack_oprefs = Some(bridge_stack);
        // Kept-stack branch guards resume the full-body walk at the guard's OWN
        // mid-opcode jitcode offset — the same resolved coordinate stored in
        // the frame pc — instead of the opcode-entry marker `pc_map[py_pc]`.
        // `None` leaves the walk on the pc_map entry.
        sym.bridge_walk_entry_pc =
            crate::state::frame_pc_is_resolved_offset_at(frame0.jitcode_index, frame0.pc)
                .then_some(frame0.pc as usize);
        sym.bridge_local_types = Some(bridge_local_types);
        // consume_boxes (resume.py:1055) fills `f.registers_r` by abstract
        // register color; keep that color-indexed decode so a cross-frame
        // bridge resume snapshot can read `registers_r[color]`
        // (`_get_list_of_active_boxes`, pyjitpl.py:216-233). `sym.registers_r`
        // is about to be overwritten with the slot-indexed semantic mirror and
        // then rebuilt by init_symbolic, losing this color decode.
        sym.bridge_registers_r = Some(bridge_registers_r.clone());

        // pyjitpl.py:3424 `rebuild_state_after_failure` tail —
        // `consume_virtualref_boxes` (resume.py:1093):
        //   for i in range(0, len(virtualref_boxes), 2):
        //       virtual_box = virtualref_boxes[i]
        //       vref_box    = virtualref_boxes[i + 1]
        //   self.virtualref_boxes += [virtual_box, vref_box]
        // The parent guard already encoded the resumed pair sequence into
        // `rd_numb`'s vref section (resume.py:738-754 `consume_vref_and_vable`
        // → `consume_virtualref_info`), and `rebuild_from_numbering`
        // (majit/majit-ir/src/resumedata.rs:402-416) decoded it into
        // `resume_data.virtualref_values`. Materialize the OpRef + concrete
        // pointer for each pair through the same `resolve` / `decode_concrete`
        // callbacks used for `frames[0].values`, then push the pairs into
        // `sym.virtualref_boxes` so `opimpl_virtual_ref` /
        // `opimpl_virtual_ref_finish` handlers (state.rs:3475-3500) observe
        // the parent's still-open virtualref scope at bridge entry.
        //
        // For traces with no active `virtual_ref`, `virtualref_values` is
        // empty and this loop is a no-op.
        let vref_values = &resume_data.virtualref_values;
        // resume.py:1397 `vrefinfo.continue_tracing(vref, virtual_val)` —
        // restore `JitVirtualRef.forced = virtual` and clear
        // `virtual_token` for every still-active pair. The runtime
        // mutation lives on the live JitVirtualRef heap object, so the
        // call is gated on a non-null concrete `vref_ptr`. Pyre's
        // `VirtualRefInfo::new()` (majit-metainterp/src/virtualref.rs:257)
        // is a cheap struct-only constructor — instantiate once outside
        // the loop instead of per-pair.
        let vrefinfo = majit_metainterp::virtualref::VirtualRefInfo::new();
        // resume.py:1093 `consume_virtualref_boxes` decodes exactly
        // `size * 2` entries; a malformed odd-length stream is a bug.
        assert!(
            vref_values.len() % 2 == 0,
            "virtualref_values must contain an even number of entries (got {})",
            vref_values.len(),
        );
        for pair in vref_values.chunks_exact(2) {
            let (virt_opref, virt_val) = bridge_decode_box(
                ctx,
                &pair[0],
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let (vref_opref, vref_val) = bridge_decode_box(
                ctx,
                &pair[1],
                Type::Ref,
                rd_virtuals,
                resume_data,
                fail_values,
                fail_types,
                backend,
                &mut virtuals_cache,
            );
            let virt_ptr = value_to_usize(&virt_val);
            let vref_ptr = value_to_usize(&vref_val);
            sym.virtualref_boxes.push((virt_opref, virt_ptr));
            sym.virtualref_boxes.push((vref_opref, vref_ptr));
            // pyjitpl.py:3438 / resume.py:1397: continue_tracing is called
            // unconditionally for every (vref, real_object) pair. The
            // is_virtual_ref(vref) guard (virtualref.py:123) and the
            // `assert real_object` invariant (virtualref.py:125, ported as
            // debug_assert!) both live inside continue_tracing itself
            // (virtualref.rs:419/424) — the outer virt_ptr guard masked
            // exactly the case RPython asserts on, so do not pre-gate here.
            unsafe {
                vrefinfo.continue_tracing(vref_ptr as *mut u8, virt_ptr as *mut u8);
            }
        }

        // pyjitpl.py:3443 `synchronize_virtualizable()` inside
        // `rebuild_state_after_failure` (pyjitpl.py:3454) writes
        // `virtualizable_boxes` to the heap via `write_boxes()`
        // (virtualizable.py:101-113). This is the ONLY call in
        // RPython's bridge-resume path — there is no second call.
        //
        // In pyre, the equivalent write is `sync_virtualizable_after_
        // guard_failure` (eval.rs:5709, `ResumeVableMode::
        // GuardFailureSync`) which runs in the compiled bridge's
        // guard-failure recovery chain BEFORE `setup_bridge_sym`.
        // No additional synchronize call is needed here.

        // Multi-frame bridge. The body above reconstructed
        // the portal (`frames[0]`) into the caller-visible root `sym`. When
        // the guard fired inside inlined callees, `frames[1..]` (OUTERMOST-
        // FIRST) must also be reconstructed and pushed so the framestack
        // matches the inline depth (`rebuild_from_resumedata` resume.py:1049-
        // 1056: every frame via `newframe`+`consume_boxes`, then tracing
        // continues at `framestack[-1]`). Decode each into a lightweight
        // recipe here (while `resume_data` / `virtuals_cache` are in scope);
        // `trace_bytecode` assembles+pushes them right before `interpret()`.
        // Any callee that cannot be faithfully rebuilt aborts the whole
        // multi-frame path back to the single-frame bridge.
        // The outermost (root) frame resumes at its OWN pc once the
        // reconstructed callees return — NOT at the innermost pc that
        // `decode_and_restore_guard_failure` returns as the trace start.
        // Thread `frames[0].pc` so `trace_bytecode` can root the outermost
        // frame at its own pc while the carrier callees resume at their
        // frames[i].pc. A negative (no-snapshot) root pc aborts the
        // multi-frame path.
        if std::env::var_os("PYRE_MFRAME_DIAG").is_some() {
            let pcs: Vec<i32> = resume_data.frames.iter().map(|f| f.pc).collect();
            eprintln!(
                "[mframe] resume frames.len()={} pcs={pcs:?}",
                resume_data.frames.len()
            );
        }
        if resume_data.frames.len() > 1 {
            let root_jitcode_index = resume_data.frames[0].jitcode_index;
            let root_pc_valid = resume_data.frames[0].pc >= 0;
            let root_pc = if root_pc_valid {
                resume_data.frames[0].pc as usize
            } else {
                0
            };
            let mut recipes: Vec<ReconstructRecipe> =
                Vec::with_capacity(resume_data.frames.len() - 1);
            let mut ok = root_pc_valid;
            if root_pc_valid {
                let inline_frames = &resume_data.frames[1..];
                for (idx, frame) in inline_frames.iter().enumerate() {
                    let in_a_call = idx + 1 < inline_frames.len();
                    match reconstruct_inline_recipe(
                        ctx,
                        frame,
                        rd_virtuals,
                        resume_data,
                        fail_values,
                        fail_types,
                        backend,
                        &mut virtuals_cache,
                        in_a_call,
                    ) {
                        Some(recipe) => recipes.push(recipe),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
            }
            if !ok {
                // A multi-frame resume whose callee chain cannot be
                // reconstructed must still be MARKED as multi-frame: without a
                // carrier the trace-start routing treats the resume as a
                // single-frame bridge and walks the ROOT frame's code at the
                // INNERMOST frame's pc, compiling a degenerate bridge that
                // finishes with a root resume slot as the call result
                // (dropping the whole in-flight callee continuation — wrong
                // values, or a SEGV when the slot is not an int box).  An
                // empty-recipe carrier routes the walk to its NoRecipes abort
                // instead, degrading to the blackhole re-interpret.
                recipes.clear();
            }
            ctx.set_bridge_inline_carrier(BridgeInlineCarrier {
                root_pc,
                root_jitcode_index,
                recipes,
            });
        }
    }

    /// resume.py:1042-1057 rebuild_from_resumedata parity.
    ///
    /// Decodes rd_numb via `majit_ir::resumedata::rebuild_from_numbering`.
    /// Frame box counts come from jitcode liveness (jitcode.position_info)
    /// at the frame's resume pc — the same data the encoder uses via
    /// `get_list_of_active_boxes`.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        fail_arg_types: &[Type],
        storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
    ) -> Option<majit_metainterp::ResumeDataResult> {
        use majit_ir::resumedata::rebuild_from_numbering;

        // interp_jit.py:67-74 PyPyJitDriver portal reds are `[frame, ec]`;
        // virtualizable boxes are appended later by
        // MetaInterp::initialize_virtualizable(), not carried as reds.
        _meta.trace_extra_reds = 1;
        let storage = storage?;
        let rd_numb = storage.rd_numb.as_slice();
        // resume.py:1071 `self.consts = storage.rd_consts` — borrow
        // the shared pool; `ResumeDataResult` carries the Arc handle
        // so downstream virtual materialization reads the same pool
        // the GC walker updates.
        let rd_consts = storage.rd_consts();

        // resume.py:1049-1055 parity: consume_boxes(f.get_current_position_info())
        // RPython uses jitcode liveness via get_current_position_info; majit
        // routes the same lookup through `frame_value_count_at`.
        let cb = crate::state::frame_value_count_at;
        let (num_failargs, vable_values, vref_values, frames) = rebuild_from_numbering(
            rd_numb,
            rd_consts,
            fail_arg_types,
            Some(&cb),
            storage.rd_virtuals.len(),
        );

        if frames.is_empty() {
            return None;
        }

        Some(majit_metainterp::ResumeDataResult {
            frames,
            virtualizable_values: vable_values,
            virtualref_values: vref_values,
            storage: Some(storage.clone()),
            // resume.py:1042 num_failargs from rd_numb header. Used by
            // bridge virtual materialization (resume.py:1556-1564 decode_box
            // negative-index normalization: `num + len(liveboxes)`).
            num_failargs,
            // compile.py:797 ResumeGuardDescr.fail_arg_types — kept so
            // `materialize_bridge_virtual::decode_fieldnum` can mint typed
            // `OpRef::input_arg_typed(idx, fail_arg_types[idx])` per
            // resume.py:1247-1264 (`return self.liveboxes[num]` whose Box
            // carries history.py:220 `box.type`).
            fail_arg_types: fail_arg_types.to_vec(),
        })
    }

    /// pyjitpl.py:2982 get_procedure_token: compute green key for a PC.
    fn green_key_for_pc(&self, pc: usize) -> Option<u64> {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return None;
        }
        let code = unsafe { (*frame_ptr).pycode };
        Some(crate::driver::make_green_key(code, pc))
    }

    fn code_ptr(&self) -> usize {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return 0;
        }
        unsafe { (*frame_ptr).pycode as usize }
    }

    fn update_meta_for_cut(meta: &mut Self::Meta, _header_pc: usize, original_box_types: &[Type]) {
        // Current pyre cut traces still recover only the expanded array-item
        // count from the merge-point box vector. Preserve the provisional live
        // depth and update just the heap-array capacity until the later
        // red-only/virtualizable-box migration lands.
        // `NUM_SCALAR_INPUTARGS` already counts `NUM_EXTRA_REDS` (frame +
        // extra_reds + vable scalars), so do NOT add `trace_extra_reds`.
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let _ = meta.trace_extra_reds; // staging copy retained for activation gates only
        if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            let new_capacity = original_box_types
                .len()
                .saturating_sub(NUM_SCALAR_INPUTARGS);
            if meta.slot_types.len() > new_capacity {
                meta.slot_types.truncate(new_capacity);
            } else if meta.slot_types.len() < new_capacity {
                meta.slot_types.resize(new_capacity, Type::Ref);
            }
            meta.array_capacity = new_capacity;
        }
    }

    fn build_meta_from_merge_point(
        provisional: &PyreMeta,
        _header_pc: usize,
        original_boxes: &[majit_metainterp::GreenBox],
    ) -> PyreMeta {
        let original_box_count = original_boxes.len();
        // `NUM_SCALAR_INPUTARGS` already counts `NUM_EXTRA_REDS` (frame +
        // extra_reds + vable scalars), so do NOT add `trace_extra_reds`.
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let _ = provisional.trace_extra_reds; // staging copy only
        // RPython parity: Python locals/stack are always Ref.
        let slot_types = if original_box_count >= NUM_SCALAR_INPUTARGS {
            vec![Type::Ref; original_box_count.saturating_sub(NUM_SCALAR_INPUTARGS)]
        } else {
            Vec::new()
        };
        let array_capacity = if original_box_count >= NUM_SCALAR_INPUTARGS {
            original_box_count.saturating_sub(NUM_SCALAR_INPUTARGS)
        } else {
            provisional.array_capacity
        };
        PyreMeta {
            num_locals: provisional.num_locals,
            ns_len: provisional.ns_len,
            namespace_dependent: provisional.namespace_dependent,
            valuestackdepth: provisional.valuestackdepth,
            array_capacity,
            trace_extra_reds: provisional.trace_extra_reds,
            has_virtualizable: provisional.has_virtualizable,
            slot_types,
        }
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        if values.is_empty() {
            return;
        }

        // Multi-frame format: [num_frames, size_0, data_0..., size_1, data_1...]
        // Detect: values[0] is a small number (1-10) = frame count
        // Legacy: values[0] is a large pointer
        let first = values[0];
        if first >= 1 && first <= 10 && values.len() > 2 {
            let _num_frames = first as usize;
            let outer_size = values[1] as usize;
            if outer_size > 0 && 2 + outer_size <= values.len() {
                // Restore outermost frame only.
                // Inner frame guard failure → interpreter re-executes callee call.
                self.restore_single_frame(meta, &values[2..2 + outer_size]);
                return;
            }
        }

        // Legacy single-frame format
        self.restore_single_frame(meta, values);
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let Some(_frame) = values.first() else {
            return;
        };
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] before arg0={:?} meta.vsd={} has_vable={} values={:?}",
                arg0, meta.valuestackdepth, meta.has_virtualizable, values
            );
        }
        self.restore_expanded_virtualizable_values_with_extra_reds(
            meta,
            values,
            meta.trace_extra_reds,
        );
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] after arg0={:?} ni={} vsd={}",
                arg0,
                self.next_instr(),
                self.valuestackdepth()
            );
        }
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        if !meta.has_virtualizable {
            self.restore_values(meta, values);
            return true;
        }

        let Some(frame) = values.first() else {
            return false;
        };
        self.frame = value_to_usize(frame);
        if values.len() == 1 {
            return self.validate_frame();
        }

        // The rd_numb deopt path restores the whole virtualizable
        // positionally before this runs (`resume.rs consume_vable_info` →
        // `write_from_resume_data_partial`, resume.py:1399-1408).  The
        // per-frame liveness section fills the blackhole REGISTERS and never
        // touches the vable (blackhole.py:1376+), so the positional restore
        // is authoritative for the frame.  Validate it and clear stale slots
        // beyond valuestackdepth (blackhole fresh-frame parity), which the
        // vable section restore does not perform.
        if !self.validate_frame() {
            return false;
        }
        let vsd = self.valuestackdepth();
        if let Some(arr) = self.locals_cells_stack_array_mut() {
            for i in vsd..arr.len() {
                arr[i] = pyre_object::PY_NULL;
            }
        }
        true
    }

    /// resume.py:1077 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
    /// Return the type of each slot in the resumed frame section.
    /// In pyre, all frame slots are PyObjectRef (GCREF), so every slot
    /// is Ref. RPython uses typed registers (boxes_i/r/f) but pyre's
    /// virtualizable array is uniformly Ref.
    fn reconstructed_frame_value_types(
        &self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        // resume.py:1077: consume_boxes fills boxes_i/boxes_r/boxes_f.
        // pyre frame slots (locals_cells_stack_w) are all GCREF (Ref).
        let nlocals = meta.num_locals;
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // Header [frame_ptr=Ref, ni=Int, code=Ref, vsd=Int, ns=Ref] + all locals/stack as Ref.
        Some(crate::virtualizable_gen::virt_live_value_types(
            nlocals + stack_only,
        ))
    }

    /// resume.py:1049 parity: restore frame register state from decoded values.
    /// resume.py:1077 consume_boxes → _prepare_next_section → enumerate_vars:
    /// each callback_r writes a ref value to the register at the given index.
    /// In pyre, this writes values to the PyFrame's locals/stack via the
    /// virtualizable mechanism (restore_virtualizable_state handles the
    /// full [frame, ni, code, vsd, ns, locals..., stack...] layout).
    fn restore_reconstructed_frame_values(
        &mut self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        values: &[Value],
        exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        // resume.py:1077 consume_boxes parity: write values to the frame.
        // blackhole.py:337: setposition(jitcode, pc) before consume_one_section —
        // frame_pc from rd_numb is the liveness PC (orgpc).
        self.resume_pc = Some(_frame_pc as usize);
        self.restore_guard_failure_values(meta, values, exception)
    }

    /// blackhole.py:1800 parity: multi-frame support.
    fn supports_multi_frame_restore(&self) -> bool {
        true
    }

    /// blackhole.py:1333 parity: push outer frame for chain.
    /// Multi-frame recovery handled by blackhole chain in call_jit.rs
    /// which receives all frame sections in the typed vector.
    fn push_caller_frame(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _values: &[Value],
        _pc: u64,
        _jitcode_index: i32,
    ) -> bool {
        true
    }

    /// blackhole.py:1760 parity: frame transition via chain.
    fn pop_to_caller_frame(&mut self, _meta: &Self::Meta) -> bool {
        false // Blackhole chain handles this directly.
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        crate::virtualizable_gen::virt_heap_ptr(self, _virtualizable)
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        if info.array_fields.is_empty() {
            return Some(Vec::new());
        }
        // virtualizable.py:86 parity: full array length read from the
        // live heap object (`lst = getattr(virtualizable, fieldname);
        // append(len(lst))`). Upstream has no fallback — if the heap
        // isn't readable here, return `None` so the caller skips.
        if info.can_read_all_array_lengths_from_heap() {
            if let Some(frame_ptr) = self.frame_ptr() {
                let lens = unsafe { info.read_array_lengths_from_heap(frame_ptr) };
                return Some(lens);
            }
        }
        None
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        // Heap is source of truth — just validate the frame is usable.
        if !self.validate_frame() {
            return false;
        }
        // virtualizable.py:170 force_token_before_residual_call parity:
        // clear vable_token so the JIT knows the virtualizable is synced.
        if let Some(frame_ptr) = self.frame_ptr() {
            unsafe { info.reset_vable_token(frame_ptr) };
        }
        true
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(frame_ptr) = self.frame_ptr() else {
            return;
        };
        // Heap is source of truth — nothing to sync. Just reset token.
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
    }

    fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
        crate::virtualizable_gen::virt_sync_before_residual(self, ctx)
    }

    fn sync_virtualizable_after_residual_call(
        &self,
        _ctx: &mut TraceCtx,
    ) -> ResidualVirtualizableSync {
        crate::virtualizable_gen::virt_sync_after_residual(self, _ctx)
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        self.import_virtualizable_state(static_boxes, array_boxes)
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        Some(self.export_virtualizable_state())
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        Self::pypyjit_collect_jump_args(sym)
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::pypyjit_collect_typed_jump_args(sym)
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }

    fn validate_close_with_jump_args(
        _sym: &Self::Sym,
        meta: &Self::Meta,
        jump_args: &[OpRef],
    ) -> bool {
        let _ = meta;
        // pyre does not close loops by re-reading symbolic stack state from
        // the trace header.  Instead it materializes explicit jump args from
        // the concrete frame-backed virtualizable state at the merge point:
        //   [frame, next_instr, valuestackdepth, locals..., stack...]
        //
        // Retraces/bridges may start from a transient stackful state
        // (e.g. direct-call trace-through in progress) and still legally
        // jump back to a target loop whose merge-point stack is smaller.
        // RPython closes those traces against the target token contract, not
        // the retrace entry state's stack depth.  So for pyre's explicit
        // jump-arg model, the trace-start `meta.valuestackdepth` is not a
        // sound validator here.
        jump_args.len() >= crate::virtualizable_gen::NUM_SCALAR_INPUTARGS
    }

    /// RPython resume.py: materialize a virtual object from resume data.
    ///
    /// Called during guard failure recovery when the optimizer kept an
    /// object virtual (New + SetfieldGc eliminated). The resume mechanism
    /// reconstructs the object so the interpreter can use it.
    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, &[])
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, materialized_refs)
    }
}

impl PyreJitState {
    fn materialize_virtual_ref_from_layout(
        &mut self,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        use majit_metainterp::resume::MaterializedVirtual;

        match materialized {
            // resume.py:618-620 VirtualInfo.allocate — `allocate_with_vtable(descr=self.descr)`
            // followed by `setfields`. descr carries both vtable and obj_size.
            MaterializedVirtual::Obj {
                descr: Some(descr),
                fields,
                ..
            } => materialize_virtual_object(descr, fields, materialized_refs),
            // resume.py:633-636 VStructInfo.allocate — `allocate_struct(self.typedescr)`
            // + `setfields`. No vtable write.
            MaterializedVirtual::Struct {
                descr: Some(descr),
                fields,
                ..
            } => materialize_virtual_struct(descr, fields, materialized_refs),
            MaterializedVirtual::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => materialize_virtual_raw_buffer(
                *func,
                *size,
                offsets,
                descrs,
                values,
                materialized_refs,
            ),
            _ => None,
        }
    }
}

/// resume.py:618-621 VirtualInfo.allocate parity — `allocate_with_vtable(descr)`
/// then `setfields`.
///
/// Allocates `descr.size()` bytes aligned to 8, seeds the PyObject header
/// (`ob_type = descr.vtable()`, `w_class = get_instantiate(vtable)`), then
/// replays each traced field using the FieldDescr's byte offset/size/type
/// from the SizeDescr's `all_fielddescrs()` table (resume.py:597-603
/// setfields loop — `decoder.setfield(struct, num, descr)`).
///
/// The vtable source is the descr itself — there is no `type_id` special-casing,
/// so bool (type_id=0, vtable=&BOOL_TYPE), range-iter (type_id=0,
/// vtable=&RANGE_ITER_TYPE) and custom classes all dispatch uniformly.
fn materialize_virtual_object(
    descr: &DescrRef,
    fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    use pyre_object::pyobject::{
        OB_TYPE_OFFSET, PyObject, PyType, W_CLASS_OFFSET, get_instantiate,
    };

    let size_descr = descr.as_size_descr()?;
    let vtable = size_descr.vtable();
    let obj_size = size_descr.size();
    if vtable == 0 || obj_size < std::mem::size_of::<PyObject>() {
        return None;
    }

    // resume.py:619 allocate_with_vtable — raw heap allocation that
    // matches `Box::leak(Box::new(...))` for pyre's existing W_*Object
    // builders. 8-byte alignment matches the natural alignment of
    // `#[repr(C)]` structs whose first field is an 8-byte pointer.
    let layout = std::alloc::Layout::from_size_align(obj_size, 8).ok()?;
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return None;
    }

    unsafe {
        let ptr = raw as *mut PyObject;
        (*ptr).ob_type = vtable as *const PyType;
        // rclass.py:739-743 set `w_class` from the cached instantiate
        // pointer on the PyType. Tracing may later overwrite this via
        // an explicit `SetfieldGc(w_class)`; the field replay below
        // takes precedence for that case (heaptracker.py:66-style
        // "typeptr" filter does NOT apply to w_class in pyre).
        (*ptr).w_class = get_instantiate(&*(vtable as *const PyType));
    }

    // resume.py:597-603 setfields parity: for each traced field,
    // look up the FieldDescr from the SizeDescr table and write the
    // value at descr.offset() with descr.field_size() bytes.
    for (field_idx, value) in fields {
        let Some(field_descr) = size_descr
            .all_fielddescrs()
            .iter()
            .find(|fd| fd.index() == *field_idx)
        else {
            // Field not in the SizeDescr table — skip (ob_type handled
            // above; anything else is a tracer bug we just log-silently
            // drop rather than corrupt memory).
            continue;
        };
        let offset = field_descr.offset();
        let field_size = field_descr.field_size();
        if offset == OB_TYPE_OFFSET && field_size == std::mem::size_of::<*const PyType>() {
            // ob_type already written from vtable; honour explicit override.
            let concrete = value.resolve_with_refs(materialized_refs)?;
            unsafe {
                (raw.add(offset) as *mut usize).write(concrete as usize);
            }
            continue;
        }
        if offset == W_CLASS_OFFSET && field_size == std::mem::size_of::<*mut PyObject>() {
            let concrete = value.resolve_with_refs(materialized_refs)?;
            unsafe {
                (raw.add(offset) as *mut usize).write(concrete as usize);
            }
            continue;
        }
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            write_field_bytes(raw, offset, field_size, concrete);
        }
    }

    Some(majit_ir::GcRef(raw as usize))
}

/// resume.py:633-636 VStructInfo.allocate parity — `allocate_struct(typedescr)`
/// (no vtable) + `setfields`.
fn materialize_virtual_struct(
    descr: &DescrRef,
    fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    let size_descr = descr.as_size_descr()?;
    let obj_size = size_descr.size();
    if obj_size == 0 {
        return None;
    }
    let layout = std::alloc::Layout::from_size_align(obj_size, 8).ok()?;
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        return None;
    }
    for (field_idx, value) in fields {
        let Some(field_descr) = size_descr
            .all_fielddescrs()
            .iter()
            .find(|fd| fd.index() == *field_idx)
        else {
            continue;
        };
        let offset = field_descr.offset();
        let field_size = field_descr.field_size();
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            write_field_bytes(raw, offset, field_size, concrete);
        }
    }
    Some(majit_ir::GcRef(raw as usize))
}

/// Write an i64 value into a byte-size field at `offset` inside `raw`.
/// Supports 1/2/4/8 byte widths. 8-byte write also handles Ref/Float
/// (Float arrives as `value.to_bits() as i64`; raw bytes reinterpret).
///
/// # Safety
/// Caller guarantees `raw + offset + field_size` is within the object allocation.
unsafe fn write_field_bytes(raw: *mut u8, offset: usize, field_size: usize, value: i64) {
    unsafe {
        let dst = raw.add(offset);
        match field_size {
            1 => (dst as *mut u8).write(value as u8),
            2 => (dst as *mut u16).write(value as u16),
            4 => (dst as *mut u32).write(value as u32),
            8 => (dst as *mut u64).write(value as u64),
            _ => {
                // Fallback: write as raw bytes LE, truncated.
                let bytes = value.to_le_bytes();
                for i in 0..field_size.min(bytes.len()) {
                    dst.add(i).write(bytes[i]);
                }
            }
        }
    }
}

fn materialize_virtual_raw_buffer(
    func: i64,
    size: usize,
    offsets: &[i64],
    descrs: &[majit_ir::ArrayDescrInfo],
    values: &[majit_metainterp::resume::MaterializedValue],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    // resume.py:700-709 VRawBufferInfo.allocate_int iterates len(self.offsets)
    // unconditionally, indexing self.descrs[i]/self.fieldnums[i] by the same i.
    // No len-equality assert (VRawBufferInfo has none); a short descrs/values
    // raises IndexError here (encoder bug), a longer one leaves its tail unread.
    let (driver, _) = crate::driver::driver_pair();
    // resume.py:1452-1456 allocate_raw_buffer:
    //   cic = self.callinfocollection
    //   calldescr, _ = cic.callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
    //   return self.cpu.bh_call_i(func, [size], None, None, calldescr)
    // The calldescr comes from the shared callinfocollection, not a freshly
    // minted MOST_GENERAL descr.  func stays the VRawBufferInfo.func (resume.py
    // discards the callinfo's func as `_`).
    let cic = driver
        .meta_interp()
        .callinfocollection()
        .expect(
            "materialize_virtual_raw_buffer: MetaInterp.callinfocollection \
             required for VRawBufferInfo recovery (resume.py:1453)",
        )
        .clone();
    // resume.py:1455: calldescr, _ = cic.callinfo_for_oopspec(...).
    // callinfo_for_oopspec returns (None, 0) on a miss (effectinfo.py:444-447)
    // — no lookup-time guard; the calldescr is consumed directly below.
    let (calldescr, _) =
        cic.callinfo_for_oopspec(majit_ir::descr::OopSpecIndex::RawMallocVarsizeChar);
    let descr_ref = calldescr.expect("OS_RAW_MALLOC_VARSIZE_CHAR calldescr (callinfocollection)");
    let cd = descr_ref
        .as_call_descr()
        .expect("OS_RAW_MALLOC_VARSIZE_CHAR: not a CallDescr");
    let bh_calldescr = majit_translate::jitcode::BhCallDescr::from_arg_classes(
        cd.arg_classes(),
        cd.result_class(),
        cd.get_extra_info().clone(),
    );
    // resume.py:703: buffer = self.cpu.bh_call_i(func, [size], None, None, calldescr)
    let buffer = driver.meta_interp().backend().bh_call_i(
        func,
        Some(&[size as i64]),
        None,
        None,
        &bh_calldescr,
    );

    let backend = driver.meta_interp().backend();
    // resume.py:705-708: per-item bh_raw_store_i/f, run unconditionally (no
    // buffer == 0 guard).
    for i in 0..offsets.len() {
        let concrete = values[i].resolve_with_refs(materialized_refs)?;
        let di = &descrs[i];
        let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(di);
        // resume.py:1544: assert not descr.is_array_of_pointers()
        assert!(
            !bh_descr.is_array_of_pointers(),
            "raw buffer entry must not be pointer type"
        );
        let offset = offsets[i];
        if di.item_type == 2 {
            // resume.py:1545-1547: descr.is_array_of_floats() → bh_raw_store_f
            backend.bh_raw_store_f(buffer, offset, f64::from_bits(concrete as u64), &bh_descr);
        } else {
            // resume.py:1548-1550: else → bh_raw_store_i
            backend.bh_raw_store_i(buffer, offset, concrete, &bh_descr);
        }
    }

    Some(majit_ir::GcRef(buffer as usize))
}

fn value_to_usize(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    //! Most tests in this module are original Rust regressions.
    //!
    //! PyPy/Pyre upstream tests usually assert behavior at the meta-interpreter
    //! level, not at the typed-raw tracer seam exercised here. These tests keep
    //! local descriptor choices, virtualizable restoration, and raw-int/raw-f64
    //! fast paths from drifting silently. Where a test is tied directly to an
    //! upstream parity point, that reference is called out inline.

    use super::*;
    use crate::helpers::TraceHelperAccess;
    use majit_metainterp::JitState;
    use majit_metainterp::resume::{MaterializedValue, MaterializedVirtual};
    use pyre_interpreter::bytecode::{BinaryOperator, CodeObject, ConstantData, Instruction};
    use pyre_interpreter::pyopcode::decode_instruction_at;
    use pyre_interpreter::{
        LocalOpcodeHandler, Mode, OpcodeStepExecutor, PyErrorKind, SharedOpcodeHandler,
        compile_exec, compile_source,
    };
    use pyre_object::OB_TYPE_OFFSET;
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::w_list_getitem;
    use pyre_object::pyobject::{INT_TYPE, LIST_TYPE, PyType, is_list};
    use std::cell::{Cell, UnsafeCell};

    thread_local! {
        static TEST_CALLBACKS_INIT: Cell<bool> = const { Cell::new(false) };
        static TEST_JIT_DRIVER: UnsafeCell<crate::driver::JitDriverPair> = UnsafeCell::new({
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            let mut driver = majit_metainterp::JitDriver::new(1);
            driver.set_virtualizable_info(info.clone());
            driver.meta_interp_mut().num_scalar_inputargs =
                crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            (driver, info)
        });
    }

    #[test]
    fn concrete_value_preserves_int_subclass_identity() {
        pyre_interpreter::typedef::init_typeobjects();
        let obj = pyre_object::intobject::w_int_new_unique(7);
        unsafe {
            (*obj).w_class = pyre_object::w_none();
        }

        assert_eq!(ConcreteValue::from_pyobj(obj), ConcreteValue::Ref(obj));
    }

    #[test]
    fn concrete_value_still_unboxes_exact_int_and_bool() {
        pyre_interpreter::typedef::init_typeobjects();
        let int_obj = pyre_object::w_int_new(11);
        let true_obj = pyre_object::w_bool_from(true);
        let float_obj = pyre_object::w_float_new(3.14);

        assert_eq!(ConcreteValue::from_pyobj(int_obj), ConcreteValue::Int(11));
        assert_eq!(
            ConcreteValue::from_pyobj(true_obj),
            ConcreteValue::Bool(true)
        );
        assert_eq!(
            ConcreteValue::from_pyobj(float_obj),
            ConcreteValue::Float(3.14)
        );
    }

    #[test]
    fn concrete_value_preserves_float_subclass_identity() {
        pyre_interpreter::typedef::init_typeobjects();
        let obj = pyre_object::w_float_new(2.5);
        unsafe {
            (*obj).w_class = pyre_object::w_none();
        }

        assert_eq!(ConcreteValue::from_pyobj(obj), ConcreteValue::Ref(obj));
    }

    fn ensure_test_callbacks() {
        TEST_CALLBACKS_INIT.with(|init| {
            if init.get() {
                return;
            }
            init.set(true);
            let cb = Box::leak(Box::new(crate::callbacks::CallJitCallbacks {
                callee_frame_helper: |_| None,
                recursive_force_cache_safe: |_| false,
                jit_drop_callee_frame: std::ptr::null(),
                jit_frame_set_slot_ref: std::ptr::null(),
                jit_frame_set_slot_int: std::ptr::null(),
                jit_frame_set_slot_float: std::ptr::null(),
                jit_force_callee_frame: std::ptr::null(),
                jit_force_recursive_call_1: std::ptr::null(),
                jit_force_recursive_call_argraw_boxed_1: std::ptr::null(),
                jit_force_self_recursive_call_argraw_boxed_1: std::ptr::null(),
                jit_create_callee_frame_1: std::ptr::null(),
                jit_create_callee_frame_1_raw_int: std::ptr::null(),
                jit_create_self_recursive_callee_frame_1: std::ptr::null(),
                jit_create_self_recursive_callee_frame_1_raw_int: std::ptr::null(),
                driver_pair: || TEST_JIT_DRIVER.with(|cell| cell.get() as *mut u8),
                ensure_majit_jitcode: |_, _| {},
                drain_backend_jit_exc: || {},
            }));
            crate::callbacks::init(cb);
        });
    }

    /// Install the real `hash_w` on this test thread so object/str-keyed
    /// dicts built while running interpreted Python bucket through the
    /// single hash path (`baseobjspace.py:840-845`).  These tests don't go
    /// through `init_jit_hooks`, which installs the hook at boot for the
    /// pyrex binary; the thread-local cell must be set on the test thread.
    fn install_test_hash_hook() {
        unsafe fn test_hash_w(obj: pyre_object::PyObjectRef) -> i64 {
            match pyre_interpreter::builtins::try_hash_value(obj) {
                Ok(h) => h,
                Err(e) => {
                    pyre_interpreter::baseobjspace::set_pending_hash_error(e);
                    pyre_object::dict_eq_hook::signal_hash_error(obj);
                    0
                }
            }
        }
        pyre_object::dict_eq_hook::register_hash_w_hook(test_hash_w);
    }

    /// Install a populated `PyJitCode` for `code_ref` so a subsequent
    /// `jitcode_for(code_ref)` returns a valid SD entry. The trace-side
    /// `MetaInterpStaticData.jitcodes` list is now populated only by
    /// `CodeWriter.make_jitcodes()` (warmspot.py:281-282); tests that
    /// exercise tracer logic directly must still arrive at that
    /// post-`make_jitcodes` state, so we install a minimal assembled
    /// jitcode with a `live/` anchor at offset 0 — every can-raise
    /// CALL_* in the tracer now emits an inline `GuardNoException`
    /// (pyjitpl.py:2082) whose resumedata snapshot routes through
    /// `get_list_of_active_boxes` → `JitCode::get_live_vars_info`
    /// (jitcode.py:80-93), so the test fixture must satisfy that
    /// liveness lookup.
    fn install_test_jitcode(code: &CodeObject, code_ref: *const ()) {
        install_test_jitcode_with_liveness(code, code_ref, &[0, 0, 0], 1);
    }

    /// Register a skeleton jitcode for `code_ref` in `METAINTERP_SD` (keyed
    /// on the PyCode wrapper), publishing `all_liveness` so the
    /// resume/blackhole decoder (`restore_guard_failure_values`) reads the
    /// caller's `[length_i, length_r, length_f, <colors…>]` buffer at offset
    /// 0 instead of the empty `[0, 0, 0]` default. The single `live/` op the
    /// builder emits patches to liveness offset 0.
    fn install_test_jitcode_with_liveness(
        code: &CodeObject,
        code_ref: *const (),
        all_liveness: &[u8],
        num_liveness_ops: usize,
    ) {
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code_ref as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let mut builder = majit_metainterp::JitCodeBuilder::default();
        let live_patch = builder.live_placeholder();
        builder.patch_live_offset(live_patch, 0);
        let mut insns = indexmap::IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, all_liveness, all_liveness.len(), num_liveness_ops);
        let mut pyjit = crate::PyJitCode::skeleton(raw_code);
        pyjit.jitcode = std::sync::Arc::new(builder.finish());
        pyjit.metadata.is_drained = true;
        METAINTERP_SD.with(|r| {
            r.borrow_mut()
                .set_jitcodes_from_make_result(vec![std::sync::Arc::new(pyjit)]);
        });
    }

    /// Bind `sym.jitcode` to a populated `JitCode` whose `raw_code()`
    /// resolves to a real `PyCode` (derived from `w_code`), so the
    /// methods that hard-deref `(*sym.jitcode).raw_code()` — notably the
    /// `close_loop_args` snapshot encoder at trace_opcode.rs:4874 — do not
    /// null-deref. Mirrors the
    /// `get_list_of_active_boxes_reads_kind_specific_register_banks`
    /// harness (trace_opcode.rs:11151) but with a non-null `JitCode.code`.
    ///
    /// `w_code` must be a real `PyCode` wrapper (e.g. `frame.pycode`,
    /// or `w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())`
    /// for the no-frame case). `live_{i,r,f}` are the live color indices
    /// per kind bank; they seed the published `BC_LIVE` buffer so the
    /// snapshot/active-box readers find the expected registers.
    ///
    /// Returns the leaked `JitCode` pointer; the caller must reclaim it via
    /// `unsafe { Box::from_raw(ptr) }` after the test to avoid the leak.
    fn bind_real_jitcode(
        sym: &mut PyreSym,
        w_code: *const (),
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) -> *mut JitCode {
        use majit_translate::liveness::encode_liveness;
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let mut all_liveness = vec![live_i.len() as u8, live_r.len() as u8, live_f.len() as u8];
        all_liveness.extend(encode_liveness(live_i));
        all_liveness.extend(encode_liveness(live_r));
        all_liveness.extend(encode_liveness(live_f));
        let mut insns: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        insns.insert(
            "live/".to_string(),
            majit_metainterp::jitcode::insns::BC_LIVE,
        );
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let num_regs = |live: &[u8]| {
            live.iter()
                .copied()
                .max()
                .map_or(4u16, |m| (m as u16 + 1).max(4))
        };
        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("close_loop_args_test");
            inner.set_body(majit_translate::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
                c_num_regs_i: num_regs(live_i),
                c_num_regs_r: num_regs(live_r),
                c_num_regs_f: num_regs(live_f),
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(raw_code);
        pyjit.jitcode = std::sync::Arc::new(runtime_jc);
        pyjit.metadata.is_drained = true;
        let inner_jc = JitCode {
            index: 0,
            payload: std::sync::Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));
        sym.jitcode = inner_jc_ptr;
        inner_jc_ptr
    }

    #[test]
    fn semantic_ref_slot_prefers_live_stack_color_reuse() {
        // Color 0 is owned by both local slot 0 and live operand-stack slot 0
        // (abs slot nlocals+0 = 2). With the stack slot in the live window
        // (stack_only=1) the inverse prefers the stack slot.
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 1, &[(1, 0, 0), (1, 0, 2), (1, 1, 1)], 0),
            Some(2),
        );
    }

    #[test]
    fn semantic_ref_slot_falls_back_to_local_color_map() {
        // Color 1 is owned only by local slot 1 (the sole live operand-stack
        // slot carries color 3), so the inverse falls through to the local.
        assert_eq!(
            semantic_ref_slot_for_reg_color(2, 1, &[(1, 1, 1), (1, 3, 2), (1, 4, 0)], 1),
            Some(1),
        );
    }

    #[test]
    fn semantic_ref_slot_ignores_dead_local_color_reuse() {
        // A dead local is simply absent from the per-PC entries (they record
        // only live slots), so color 0 has no live owner here -> None.
        assert_eq!(semantic_ref_slot_for_reg_color(2, 0, &[(1, 1, 1)], 0), None,);
    }

    #[test]
    fn semantic_ref_slot_none_for_beyond_window_stack_color() {
        // At BUILD_LIST entry (pc=40) the runtime stack is 3 deep
        // (stack_only=3), but the compile-time entries also carry color 5 at
        // operand-stack slot 3 (abs slot nlocals+3 = 5), live only at the
        // depth-4 sibling PC.  The inverse must classify color 5 as dead-here
        // (None) — its stack slot sits past the live window (3 >= stack_only)
        // — so `collect_outer_active_boxes` substitutes a CONST_NULL
        // placeholder rather than reading an unpopulated register.
        let pcdep = [
            (1u8, 2u16, 0u16),
            (1, 2, 1),
            (1, 2, 2),
            (1, 3, 3),
            (1, 4, 4),
            (1, 5, 5),
        ];
        assert_eq!(semantic_ref_slot_for_reg_color(2, 3, &pcdep, 5), None,);
        assert!(pcdep.iter().any(|&(_, c, _)| c == 5));
    }

    /// Encode<->decode resume-symmetry round trip over a NON-IDENTITY,
    /// coalesced color map shaped exactly as the codewriter publishes it as
    /// per-PC `(color, slot)` entries. The forward map assigns one
    /// post-regalloc Ref color per semantic slot; chordal coalescing
    /// legitimately reuses a color across slots that are never simultaneously
    /// live, so the shared inverse `semantic_ref_slot_for_reg_color` — called
    /// by the decode side (restore_guard_failure_values) — must disambiguate
    /// by the live window: the live stack prefix first, then the live locals.
    ///
    /// Layout: nlocals=2, max_stackdepth=3, live stack depth 2 (stack_only=2).
    ///   local 0 -> color 5, local 1 -> color 6
    ///   stack 0 -> color 6 (shared with live local 1)
    ///   stack 1 -> color 7
    ///   stack 2 -> color 5 (shared with live local 0; DEAD, index >= stack_only)
    ///
    /// Expected inverses are DERIVED from the published entries, so this
    /// asserts a true publish<->inverse identity rather than ad-hoc literals.
    /// The per-PC entries carry every live local plus the full compile-time
    /// stack depth; the inverse clamps out-of-window stack slots by stack_only.
    #[test]
    fn resume_symmetry_roundtrip_coalesced_color_map() {
        let nlocals = 2usize;
        // live stack depth = valuestackdepth - nlocals; the published stack map
        // is the FULL max_stackdepth (3), but only the live prefix is in window.
        let stack_only = 2usize;
        let local_map = [5u16, 6u16];
        let stack_map = [6u16, 7u16, 5u16];

        // Build the per-PC (color, slot) entries the codewriter publishes:
        // one entry per live local (slot = local index) plus one per
        // compile-time operand-stack slot (slot = nlocals + depth). Sorted by
        // (color, slot) so locals precede stack within a shared color.
        let mut pcdep: Vec<(u8, u16, u16)> = Vec::new();
        for (i, &c) in local_map.iter().enumerate() {
            pcdep.push((1, c, i as u16));
        }
        for (d, &c) in stack_map.iter().enumerate() {
            pcdep.push((1, c, (nlocals + d) as u16));
        }
        pcdep.sort();

        let invert =
            |reg: u16| semantic_ref_slot_for_reg_color(nlocals, stack_only, &pcdep, reg as usize);

        // Round-trip closure: every LIVE stack slot's published color inverts
        // back to its own semantic slot (nlocals + d) — the stack map is its
        // own inverse over the live prefix.
        for d in 0..stack_only {
            assert_eq!(invert(stack_map[d]), Some(nlocals + d));
        }

        // Color 6 is shared by live local 1 and live stack slot 0. The decoder
        // scans the live stack prefix first, so it MUST resolve to the stack
        // slot (Some(2)), never the local (Some(1)). This is the kept-stack /
        // local coalescing case the guard-failure deopt depends on.
        assert_eq!(invert(stack_map[0]), Some(2));
        assert_eq!(invert(local_map[1]), Some(2));

        // Color 5 is shared by live local 0 and the DEAD stack slot 2 (index
        // >= stack_only, outside the live window). It must route to the live
        // local, not the dead stack slot.
        assert_eq!(invert(local_map[0]), Some(0));

        // The dead stack slot's color must never recover as that slot.
        assert_ne!(invert(stack_map[2]), Some(nlocals + 2));
    }

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            num_locals: 0,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 0,
            array_capacity: 0,
            trace_extra_reds: 0,
            slot_types: Vec::new(),
            has_virtualizable: false,
        }
    }

    fn empty_state() -> PyreJitState {
        PyreJitState {
            frame: 0,
            resume_pc: None,
        }
    }

    fn compile_function_body(src: &str) -> CodeObject {
        let module = compile_source(src, Mode::Exec).expect("compile should succeed");
        module
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("source should contain a function body")
    }

    fn contains_instruction(code: &CodeObject, predicate: impl Fn(Instruction) -> bool) -> bool {
        (0..code.instructions.len()).any(|pc| {
            decode_instruction_at(code, pc)
                .map(|(instruction, _)| predicate(instruction))
                .unwrap_or(false)
        })
    }

    #[test]
    fn is_compatible_ignores_namespace_length_when_not_namespace_dependent() {
        // A pure-compute / builtin-only trace does not fold a module global, so
        // its compiled code is independent of the globals dict: a later
        // top-level bind (namespace growth) must not refuse re-entry.
        let state = empty_state();
        let mut meta = empty_meta();
        meta.num_locals = state.local_count();
        meta.namespace_dependent = false;
        meta.ns_len = state.namespace_len() + 5;
        assert!(<PyreJitState as JitState>::is_compatible(&state, &meta));
    }

    #[test]
    fn is_compatible_enforces_namespace_length_when_namespace_dependent() {
        // A trace that read a module global keeps the conservative length gate,
        // since same-key value rebinds are not value-guarded: a namespace-length
        // mismatch must refuse re-entry.
        let state = empty_state();
        let mut meta = empty_meta();
        meta.num_locals = state.local_count();
        meta.namespace_dependent = true;
        meta.ns_len = state.namespace_len() + 5;
        assert!(!<PyreJitState as JitState>::is_compatible(&state, &meta));
    }

    #[test]
    fn test_pre_opcode_snapshot_gate_skips_peek_only_and_no_guard_opcodes() {
        let iter_code =
            compile_function_body("def f(xs):\n    for x in xs:\n        return len(xs)\n");
        assert!(contains_instruction(&iter_code, |instruction| {
            matches!(instruction, Instruction::GetIter)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));
        assert!(contains_instruction(&iter_code, |instruction| {
            matches!(instruction, Instruction::ForIter { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let match_code = compile_function_body(
            "def f(xs):\n    match xs:\n        case [a, b]:\n            return 1\n    return 0\n",
        );
        assert!(contains_instruction(&match_code, |instruction| {
            matches!(instruction, Instruction::GetLen)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let import_code =
            compile_function_body("def f(pkg):\n    from pkg import name\n    return name\n");
        assert!(contains_instruction(&import_code, |instruction| {
            matches!(instruction, Instruction::ImportFrom { .. })
                && instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let except_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError('x')\n    except ValueError:\n        return 0\n",
        );
        assert!(contains_instruction(&except_code, |instruction| {
            matches!(instruction, Instruction::CheckExcMatch)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let none_branch_code = compile_function_body(
            "def f(x):\n    if x is not None:\n        return 1\n    return 0\n",
        );
        assert!(contains_instruction(&none_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfNone { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let not_none_branch_code =
            compile_function_body("def f(x):\n    if x is None:\n        return 1\n    return 0\n");
        assert!(contains_instruction(&not_none_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfNotNone { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let truth_branch_code =
            compile_function_body("def f(x):\n    if x:\n        return 1\n    return 0\n");
        assert!(contains_instruction(&truth_branch_code, |instruction| {
            matches!(instruction, Instruction::PopJumpIfFalse { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let not_truth_branch_code =
            compile_function_body("def f(x):\n    if not x:\n        return 1\n    return 0\n");
        assert!(contains_instruction(
            &not_truth_branch_code,
            |instruction| {
                matches!(instruction, Instruction::PopJumpIfTrue { .. })
                    && !instruction_needs_pre_opcode_snapshot(instruction)
            }
        ));

        let contains_code = compile_function_body("def f(a, b):\n    return a in b\n");
        assert!(contains_instruction(&contains_code, |instruction| {
            matches!(instruction, Instruction::ContainsOp { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let with_except_code = compile_function_body(
            "def f(cm):\n    try:\n        with cm:\n            return 1\n    except Exception:\n        return 0\n",
        );
        assert!(contains_instruction(&with_except_code, |instruction| {
            matches!(instruction, Instruction::WithExceptStart)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let import_name_code =
            compile_function_body("def f():\n    import math\n    return math\n");
        assert!(contains_instruction(&import_name_code, |instruction| {
            matches!(instruction, Instruction::ImportName { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let call_code = compile_function_body("def f(g, x):\n    return g(x)\n");
        let call_kw_instruction = (0..call_code.instructions.len())
            .find_map(|pc| match decode_instruction_at(&call_code, pc) {
                Some((Instruction::Call { argc }, _)) => Some(Instruction::CallKw { argc }),
                _ => None,
            })
            .expect("source should contain a Call instruction to reuse argc shape");
        assert!(instruction_may_raise(call_kw_instruction));
        assert!(instruction_needs_pre_opcode_snapshot(call_kw_instruction));

        assert!(!instruction_may_raise(Instruction::CallFunctionEx));
        assert!(!instruction_needs_pre_opcode_snapshot(
            Instruction::CallFunctionEx
        ));

        let delete_attr_code = compile_function_body("def f(obj):\n    del obj.x\n");
        assert!(contains_instruction(&delete_attr_code, |instruction| {
            matches!(instruction, Instruction::DeleteAttr { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let delete_subscr_code = compile_function_body("def f(a, i):\n    del a[i]\n");
        assert!(contains_instruction(&delete_subscr_code, |instruction| {
            matches!(instruction, Instruction::DeleteSubscr)
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_attr_code = compile_function_body("def f(obj, v):\n    obj.x = v\n");
        assert!(contains_instruction(&store_attr_code, |instruction| {
            matches!(instruction, Instruction::StoreAttr { .. })
                && instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let dict_update_code = compile_function_body("def f(a, b):\n    return {**a, **b}\n");
        assert!(contains_instruction(&dict_update_code, |instruction| {
            matches!(instruction, Instruction::DictUpdate { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let binary_slice_code = compile_function_body("def f(a, b, c):\n    return a[b:c]\n");
        assert!(contains_instruction(&binary_slice_code, |instruction| {
            matches!(instruction, Instruction::BinarySlice)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_slice_code =
            compile_function_body("def f(a, b, c, v):\n    a[b:c] = v\n    return a\n");
        assert!(contains_instruction(&store_slice_code, |instruction| {
            matches!(instruction, Instruction::StoreSlice)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let list_extend_code = compile_function_body("def f(a, b):\n    return [*a, *b]\n");
        assert!(contains_instruction(&list_extend_code, |instruction| {
            matches!(instruction, Instruction::ListExtend { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let set_update_code = compile_function_body("def f(a, b):\n    return {*a, *b}\n");
        assert!(contains_instruction(&set_update_code, |instruction| {
            matches!(instruction, Instruction::SetUpdate { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let map_add_code = compile_function_body("def f(xs):\n    return {x: x for x in xs}\n");
        assert!(contains_instruction(&map_add_code, |instruction| {
            matches!(instruction, Instruction::MapAdd { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let set_add_code = compile_function_body("def f(xs):\n    return {x for x in xs}\n");
        assert!(contains_instruction(&set_add_code, |instruction| {
            matches!(instruction, Instruction::SetAdd { .. })
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let reraise_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError(1)\n    except Exception:\n        raise\n",
        );
        assert!(contains_instruction(&reraise_code, |instruction| {
            matches!(instruction, Instruction::Reraise { .. })
                && !instruction_may_raise(instruction)
                && !instruction_needs_pre_opcode_snapshot(instruction)
        }));
    }

    #[test]
    fn test_pre_opcode_snapshot_gate_keeps_pop_before_guard_opcodes() {
        let call_code = compile_function_body("def f(g, x):\n    return g(x)\n");
        assert!(contains_instruction(&call_code, |instruction| {
            matches!(instruction, Instruction::Call { .. })
                && instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let unary_code = compile_function_body("def f(x):\n    return -x\n");
        assert!(contains_instruction(&unary_code, |instruction| {
            matches!(instruction, Instruction::UnaryNegative)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let store_subscr_code =
            compile_function_body("def f(a, i, v):\n    a[i] = v\n    return a\n");
        assert!(contains_instruction(&store_subscr_code, |instruction| {
            matches!(instruction, Instruction::StoreSubscr)
                && instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        let raise_code = compile_function_body(
            "def f():\n    try:\n        raise ValueError(1)\n    except Exception:\n        raise\n",
        );
        assert!(contains_instruction(&raise_code, |instruction| {
            matches!(instruction, Instruction::RaiseVarargs { .. })
                && !instruction_may_raise(instruction)
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // BUILD_TUPLE pops its operands before trace_build_tuple_value emits
        // the specialised-tuple w_class guards.
        let build_tuple_code = compile_function_body("def f(a, b):\n    return (a, b)\n");
        assert!(contains_instruction(&build_tuple_code, |instruction| {
            matches!(instruction, Instruction::BuildTuple { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // UNPACK_SEQUENCE pops the sequence before unpack_sequence_value emits
        // the sequence class / length guards.
        let unpack_code = compile_function_body("def f(s):\n    a, b = s\n    return a + b\n");
        assert!(contains_instruction(&unpack_code, |instruction| {
            matches!(instruction, Instruction::UnpackSequence { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // LOAD_ATTR's foldable list-method form (lst.append) reaches the trait
        // leg load_method, which pop_value's the receiver before emitting the
        // receiver class / version_tag guards.
        let load_attr_code = compile_function_body("def f(lst, x):\n    return lst.append(x)\n");
        assert!(contains_instruction(&load_attr_code, |instruction| {
            matches!(instruction, Instruction::LoadAttr { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));

        // LIST_APPEND (list comprehension) pops the appended value before the
        // strategy fast path emits its list class / strategy / unbox guards.
        let list_append_code = compile_function_body("def f(xs):\n    return [x for x in xs]\n");
        assert!(contains_instruction(&list_append_code, |instruction| {
            matches!(instruction, Instruction::ListAppend { .. })
                && instruction_needs_pre_opcode_snapshot(instruction)
        }));
    }

    /// Seed `sym` with `slots` on the symbolic value stack (TOS last).
    ///
    /// Mirrors the invariant a real trace would carry into RERAISE:
    /// `pyopcode.py:1361` reads `peekvalue(oparg)` and `:1364` does
    /// `popvalue()`, both of which require the stack to be populated with
    /// the saved lasti (when `oparg != 0`) and the exception object on
    /// top.  Sets `concrete_stack`, `symbolic_stack_types`, `registers_r`,
    /// and `valuestackdepth` in lockstep.
    fn seed_stack(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        slots: &[(OpRef, ConcreteValue, majit_ir::Type)],
    ) {
        sym.nlocals = 0;
        sym.valuestackdepth = slots.len();
        sym.concrete_stack = slots.iter().map(|(_, cv, _)| *cv).collect();
        sym.symbolic_stack_types = slots.iter().map(|(_, _, t)| *t).collect();
        sym.registers_r = slots.iter().map(|(op, _, _)| *op).collect();
        // Tests don't exercise the vable shadow path; pop_value's
        // `is_active_vable_owner` branch stays inactive.
        sym.locals_cells_stack_array_ref = ctx.const_ref(0);
    }

    #[test]
    fn test_reraise_reuses_last_exception_object() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_opref = ctx.const_ref(exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = exc_opref;
        // pyopcode.py:1353 — for `RERAISE 0`, stack is `[..., exc]`.
        seed_stack(
            &mut ctx,
            &mut sym,
            &[(exc_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref)],
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err = OpcodeStepExecutor::reraise(&mut state, 0).expect_err("reraise should raise");
        assert_eq!(err.to_exc_object(), exc);
        // pyopcode.py:1363 — oparg==0 → reraise_lasti = -1.
        assert_eq!(err.reraise_lasti, -1);
        // pyopcode.py:1376 — RaiseWithExplicitTraceback → attach_tb=False.
        assert!(!err.attach_tb);
    }

    #[test]
    fn test_reraise_nonzero_oparg_threads_saved_lasti() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_opref = ctx.const_ref(exc as i64);
        // pyopcode.py:165-170 lasti push synthesizes
        // `space.newint(lasti_value)` — a fresh W_IntObject.
        let saved_lasti_value: i64 = 42;
        let saved_lasti_obj = pyre_object::w_int_new(saved_lasti_value);
        let lasti_opref = ctx.const_ref(saved_lasti_obj as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = exc_opref;
        // pyopcode.py:1353 — for `RERAISE 1`, stack is `[..., lasti, exc]`.
        seed_stack(
            &mut ctx,
            &mut sym,
            &[
                (
                    lasti_opref,
                    ConcreteValue::Ref(saved_lasti_obj),
                    majit_ir::Type::Ref,
                ),
                (exc_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref),
            ],
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err = OpcodeStepExecutor::reraise(&mut state, 1).expect_err("reraise should raise");
        assert_eq!(err.to_exc_object(), exc);
        // pyopcode.py:1361 — `reraise_lasti = self.space.int_w(self.peekvalue(oparg))`.
        assert_eq!(err.reraise_lasti, saved_lasti_value as i32);
        assert!(!err.attach_tb);
    }

    #[test]
    fn test_reraise_nonconst_lasti_signals_abort_to_dispatcher() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_opref = ctx.const_ref(exc as i64);
        // Non-Int slot at peek(oparg): an exception object stands in for
        // any non-Int concrete value (a const-int box is the only shape
        // the trace can fold at compile time).
        let non_int_opref = ctx.const_ref(exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = exc_opref;
        seed_stack(
            &mut ctx,
            &mut sym,
            &[
                (non_int_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref),
                (exc_opref, ConcreteValue::Ref(exc), majit_ir::Type::Ref),
            ],
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err = OpcodeStepExecutor::reraise(&mut state, 1).expect_err("reraise should raise");
        // Non-Int lasti slot → reraise_lasti < 0; dispatcher detects the
        // combination `oparg != 0 && reraise_lasti < 0` and routes to
        // TraceAction::Abort, letting the interpreter handle the rare
        // non-const case via the concrete-frame fallback.
        assert!(err.reraise_lasti < 0);
    }

    #[test]
    fn test_raise_varargs_zero_reuses_last_exception_object() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.last_exc_value = exc;
        sym.last_exc_box = ctx.const_ref(exc as i64);
        pyre_interpreter::eval::set_current_exception(PY_NULL);
        pyre_interpreter::eval::set_current_exception(exc);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 0).expect_err("bare raise should raise");
        assert_eq!(err.to_exc_object(), exc);
        pyre_interpreter::eval::set_current_exception(PY_NULL);
    }

    #[test]
    fn test_raise_varargs_seeds_last_exception_box_for_finishframe_exception() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let exc_ref = ctx.const_ref(exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(exc_ref, ConcreteValue::Ref(exc)),
        )
        .expect("push exception");

        let err = OpcodeStepExecutor::raise_varargs(&mut state, 1)
            .expect_err("explicit raise should raise");
        assert_eq!(state.sym().last_exc_value, exc);
        assert_eq!(state.sym().last_exc_box, exc_ref);
        assert!(state.sym().class_of_last_exc_is_const);
        assert_eq!(err.to_exc_object(), exc);
    }

    #[test]
    fn test_raise_varargs_rejects_non_exception_values_like_interpreter() {
        let mut ctx = TraceCtx::for_test(1);
        let bad = pyre_object::w_int_new(7);
        let bad_ref = ctx.const_ref(bad as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(bad_ref, ConcreteValue::Ref(bad)),
        )
        .expect("push invalid raise value");

        let err = OpcodeStepExecutor::raise_varargs(&mut state, 1)
            .expect_err("invalid raise should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_raise_varargs_rejects_non_exception_types_like_interpreter() {
        let code = compile_exec("x = int\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        frame
            .execute_frame(None, None)
            .expect("module body should execute");
        let ty = unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), "x") }
            .expect("namespace should contain x");

        let mut ctx = TraceCtx::for_test(1);
        let ty_ref = ctx.const_ref(ty as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(ty_ref, ConcreteValue::Ref(ty)),
        )
        .expect("push non-exception type");

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 1).expect_err("raising int should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_raise_varargs_rejects_builtin_callables_that_are_not_exception_classes() {
        let code = compile_exec("x = len\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        frame
            .execute_frame(None, None)
            .expect("module body should execute");
        let callable = unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), "x") }
            .expect("namespace should contain x");

        let mut ctx = TraceCtx::for_test(1);
        let callable_ref = ctx.const_ref(callable as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(callable_ref, ConcreteValue::Ref(callable)),
        )
        .expect("push builtin callable");

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 1).expect_err("raising len should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_raise_varargs_sets_cause_like_interpreter() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let cause = pyre_interpreter::PyError::value_error("root").to_exc_object();
        let exc_ref = ctx.const_ref(exc as i64);
        let cause_ref = ctx.const_ref(cause as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(exc_ref, ConcreteValue::Ref(exc)),
        )
        .expect("push exception");
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(cause_ref, ConcreteValue::Ref(cause)),
        )
        .expect("push cause");

        let err =
            OpcodeStepExecutor::raise_varargs(&mut state, 2).expect_err("raise from should raise");
        assert_eq!(err.to_exc_object(), exc);
        assert_eq!(
            pyre_interpreter::getattr_str(exc, "__cause__").expect("read cause"),
            cause
        );
    }

    #[test]
    fn test_raise_varargs_rejects_invalid_cause_like_interpreter() {
        let mut ctx = TraceCtx::for_test(1);
        let exc = pyre_interpreter::PyError::runtime_error("boom").to_exc_object();
        let cause = pyre_object::w_int_new(5);
        let exc_ref = ctx.const_ref(exc as i64);
        let cause_ref = ctx.const_ref(cause as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(exc_ref, ConcreteValue::Ref(exc)),
        )
        .expect("push exception");
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(cause_ref, ConcreteValue::Ref(cause)),
        )
        .expect("push invalid cause");

        let err = OpcodeStepExecutor::raise_varargs(&mut state, 2)
            .expect_err("invalid cause should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(
            err.message,
            "exception cause must be None or derive from BaseException"
        );
        assert_eq!(state.sym().last_exc_value, PY_NULL);
        assert_eq!(state.sym().last_exc_box, OpRef::NONE);
    }

    #[test]
    fn test_push_exc_info_and_pop_except_preserve_symbolic_previous_exception() {
        let code = compile_exec("try:\n    raise ValueError\nexcept Exception:\n    pass\n")
            .expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code);
        let prev_exc = pyre_interpreter::PyError::value_error("prev").to_exc_object();
        let caught_exc = pyre_interpreter::PyError::runtime_error("caught").to_exc_object();

        // get/set_current_exception read/write `(*ec).sys_exc_value` on the
        // thread's current EC; production establishes it via
        // `set_last_exec_ctx(frame.execution_context)` (eval.rs:841). Mirror
        // that here so the save/restore round-trips like a real frame.  The
        // prior ctx is restored on Drop so an assert failure mid-test still
        // unwinds without leaking the frame's EC into sibling tests.
        struct ExecCtxRestore(*const pyre_interpreter::PyExecutionContext);
        impl Drop for ExecCtxRestore {
            fn drop(&mut self) {
                pyre_interpreter::call::set_last_exec_ctx(self.0);
            }
        }
        let _saved_ctx = ExecCtxRestore(pyre_interpreter::call::take_last_exec_ctx());
        pyre_interpreter::call::set_last_exec_ctx(frame.execution_context);

        let mut ctx = TraceCtx::for_test(1);
        let prev_exc_ref = ctx.const_ref(prev_exc as i64);
        let caught_exc_ref = ctx.const_ref(caught_exc as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.current_exc_value = prev_exc;
        sym.current_exc_box = prev_exc_ref;
        pyre_interpreter::eval::set_current_exception(prev_exc);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        frame.push(caught_exc);
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(caught_exc_ref, ConcreteValue::Ref(caught_exc)),
        )
        .expect("push caught exception");

        OpcodeStepExecutor::push_exc_info(&mut state).expect("push_exc_info should succeed");
        assert_eq!(state.sym().current_exc_value, caught_exc);
        assert_eq!(state.sym().current_exc_box, caught_exc_ref);

        let pushed_exc = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("caught exception should remain on stack");
        assert_eq!(pushed_exc.opref, caught_exc_ref);
        let restored_prev = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("previous exception should be underneath the caught exception");
        // push_exc_info emits `GetfieldGcR(ec)` on `ec.sys_exc_value`
        // (pyopcode.py:786 runtime save-restore parity), so the previous-
        // exception slot carries that op's OpRef and its concrete value is
        // read back from the EC field — `prev_exc`, which the test seeded.
        assert_ne!(restored_prev.opref, OpRef::NONE);
        assert_eq!(restored_prev.concrete.to_pyobj(), prev_exc);

        <MIFrame as SharedOpcodeHandler>::push_value(&mut state, restored_prev.clone())
            .expect("restore previous exception for POP_EXCEPT");
        OpcodeStepExecutor::pop_except(&mut state).expect("pop_except should succeed");
        assert_eq!(state.sym().current_exc_value, prev_exc);
        assert_eq!(state.sym().current_exc_box, restored_prev.opref);
        assert_eq!(pyre_interpreter::eval::get_current_exception(), prev_exc);
        pyre_interpreter::eval::set_current_exception(PY_NULL);
    }

    #[test]
    fn test_trace_ob_type_descr_uses_immutable_header_field_descr() {
        let descr = crate::descr::ob_type_descr();
        let field = descr
            .as_field_descr()
            .expect("ob_type descr must be a field descr");
        assert_eq!(field.offset(), OB_TYPE_OFFSET);
        assert_eq!(field.field_type(), Type::Int);
        assert!(descr.is_always_pure());
        assert!(field.is_immutable());
    }

    #[test]
    fn test_pypyjit_driver_descriptor_matches_interp_jit_layout() {
        let descriptor = PyreJitState::pypyjit_driver_descriptor();

        assert!(descriptor.is_recursive);
        assert_eq!(descriptor.num_greens(), 3);
        assert_eq!(descriptor.num_reds(), 2);
        assert_eq!(descriptor.virtualizable.as_deref(), Some("frame"));
        assert_eq!(descriptor.index_of_virtualizable, 0);

        let greens = descriptor.greens();
        assert_eq!(greens[0].name, "next_instr");
        assert_eq!(greens[0].tp, Type::Int);
        assert_eq!(greens[1].name, "is_being_profiled");
        assert_eq!(greens[1].tp, Type::Int);
        assert_eq!(greens[2].name, "pycode");
        assert_eq!(greens[2].tp, Type::Ref);

        let reds = descriptor.reds();
        assert_eq!(reds[0].name, "frame");
        assert_eq!(reds[0].tp, Type::Ref);
        assert_eq!(reds[1].name, "ec");
        assert_eq!(reds[1].tp, Type::Ref);
    }

    #[test]
    fn test_pypyjit_live_values_with_ec_insert_execution_context_after_frame() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        let meta = state.build_meta(0, &PyreEnv);
        let with_ec = state.extract_live_values(&meta);

        assert_eq!(with_ec.len(), 2);
        assert_eq!(with_ec[0], Value::Ref(majit_ir::GcRef(frame_ptr)));
        assert_eq!(
            with_ec[1],
            Value::Ref(majit_ir::GcRef(frame.execution_context as usize))
        );
    }

    #[test]
    fn test_pypyjit_live_value_types_with_ec_match_descriptor_red_prefix() {
        let meta = PyreMeta {
            num_locals: 2,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 3,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref; 4],
        };

        let descriptor = PyreJitState::pypyjit_driver_descriptor();
        let with_ec = <PyreJitState as JitState>::live_value_types(&empty_state(), &meta);

        assert_eq!(
            with_ec,
            vec![descriptor.reds()[0].tp, descriptor.reds()[1].tp]
        );
    }

    #[test]
    fn test_pypyjit_collect_jump_args_inserts_ec_after_frame() {
        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(0x1000);
        let ec_ref = ctx.const_ref(0x7000);
        let code_ref = ctx.const_ref(0x2000);
        let namespace_ref = ctx.const_ref(0x3000);
        let local0 = ctx.const_ref(0x4000);
        let stack0 = ctx.const_ref(0x5000);
        let stack1 = ctx.const_ref(0x6000);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = ctx.const_int(33);
        sym.vable_pycode = code_ref;
        sym.vable_valuestackdepth = ctx.const_int(3);
        sym.vable_debugdata = ctx.const_ref(0);
        sym.vable_lastblock = ctx.const_ref(0);
        sym.vable_w_globals = namespace_ref;
        sym.execution_context = ec_ref;
        sym.registers_r = vec![local0, stack0, stack1];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let base = sym.vable_collect_jump_args();
        let with_ec = PyreJitState::pypyjit_collect_jump_args(&sym);
        let typed_with_ec = PyreJitState::pypyjit_collect_typed_jump_args(&sym);

        assert_eq!(with_ec.len(), base.len() + 1);
        assert_eq!(with_ec[0], frame_ref);
        assert_eq!(with_ec[1], ec_ref);
        assert_eq!(&with_ec[2..], &base[1..]);
        assert_eq!(typed_with_ec[0], (frame_ref, Type::Ref));
        assert_eq!(typed_with_ec[1], (ec_ref, Type::Ref));
    }

    #[test]
    fn test_pypyjit_create_sym_shifts_virtualizable_indices_after_ec() {
        let meta = PyreMeta {
            num_locals: 2,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 4,
            array_capacity: 5,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref; 5],
        };

        let sym = PyreJitState::pypyjit_create_sym(&meta, 0);

        // virtualizable derive `init_vable_indices` mints typed
        // `OpRef::input_arg_*` variants per `#[vable(inputarg, type = ...)]`
        // (state.rs:1428-1438). The expected slot variants must match those
        // exact types so variant-aware OpRef Eq (resoperation.rs:290)
        // compares correctly against later optimizer/heap-cache keys.
        assert_eq!(sym.frame, OpRef::input_arg_ref(0));
        assert_eq!(sym.execution_context, OpRef::input_arg_ref(1));
        assert_eq!(sym.vable_last_instr, OpRef::input_arg_int(2));
        assert_eq!(sym.vable_pycode, OpRef::input_arg_ref(3));
        assert_eq!(sym.vable_valuestackdepth, OpRef::input_arg_int(4));
        assert_eq!(sym.vable_debugdata, OpRef::input_arg_ref(5));
        assert_eq!(sym.vable_lastblock, OpRef::input_arg_ref(6));
        assert_eq!(sym.vable_w_globals, OpRef::input_arg_ref(7));
        assert_eq!(sym.vable_array_base, Some(8));
        assert_eq!(sym.symbolic_local_types.len(), 2);
        assert_eq!(sym.symbolic_stack_types.len(), 2);
    }

    #[test]
    fn test_restore_expanded_virtualizable_values_with_extra_reds_skips_ec_slot() {
        use majit_ir::GcRef;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};

        let module = compile_exec("def f(a, b, c):\n    i = 0\n    return i\nf(1, 2, 3)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = PyreJitState {
            frame: frame_ptr,
            resume_pc: None,
        };
        state.set_next_instr(0);
        state.set_valuestackdepth(4);
        let meta = PyreMeta {
            num_locals: 4,
            ns_len: 0,
            namespace_dependent: false,
            valuestackdepth: 4,
            array_capacity: 4,
            trace_extra_reds: 1,
            has_virtualizable: true,
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),                        // frame
            Value::Ref(GcRef(frame.execution_context as usize)), // ec
            Value::Int(8),                                       // last_instr
            Value::Ref(GcRef(frame.pycode as usize)),            // pycode
            Value::Int(4),                                       // valuestackdepth
            Value::Ref(GcRef(0)),                                // debugdata
            Value::Ref(GcRef(0)),                                // lastblock
            Value::Ref(GcRef(0)),                                // w_globals
            Value::Ref(GcRef(w_int_new(1) as usize)),            // local a
            Value::Ref(GcRef(w_int_new(2) as usize)),            // local b
            Value::Ref(GcRef(w_int_new(3) as usize)),            // local c
            Value::Int(7),                                       // local i
        ];

        state.restore_expanded_virtualizable_values_with_extra_reds(&meta, &values, 1);

        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_guard_class_uses_guard_nonnull_class() {
        let mut ctx = TraceCtx::for_test(1);
        // registers_r[i] tracks locals_cells_stack_w[*] — W_Root array, Type::Ref.
        let obj = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        state.with_ctx(|this, ctx| {
            this.guard_class(ctx, obj, &INT_TYPE as *const PyType);
        });

        let tree_loop = ctx.into_tree_loop();
        let op = tree_loop.ops.last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardClass);
        assert_eq!(op.arg(0).to_opref(), obj);
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload() {
        // value_type is read from the recorder's inputarg type (Phase α/β: Box.type
        // intrinsic parity, history.py:220) so the inputarg must be Ref for
        // trace_guarded_int_payload to take the fast path rather than short-circuit.
        let mut ctx = TraceCtx::for_test_types(&[Type::Ref]);
        // value_type is Ref per the comment above; registers_r[0] = inputarg slot 0.
        let int_obj = OpRef::input_arg_ref(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![int_obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj));

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::GuardClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI
                && op
                    .getarglist()
                    .iter()
                    .map(|a| a.to_opref())
                    .collect::<Vec<_>>()
                    == vec![int_obj]
            {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardClass"
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI"
        );
    }

    #[test]
    fn test_trace_unbox_int_with_resume_skips_guard_for_constant_object() {
        let mut ctx = TraceCtx::for_test(0);
        let int_obj = ctx.const_ref(w_int_new(7) as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let payload = {
            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                parent_frames: Vec::new(),
                pending_result_stack_idx: None,
                pending_result_type: None,
                pending_inline_frame: None,
                residual_call_pc: None,
                loop_close_marker_jit_pc: None,
                orgpc: 0,
                concrete_frame_addr: 0,
                pre_opcode_registers_r: None,
                pre_opcode_semantic_depth: None,
            };
            trace_unbox_int_with_resume(&mut state, int_obj, &INT_TYPE as *const PyType as i64)
        };

        let recorder = ctx.into_recorder();
        let payload_op = recorder
            .get_op_by_pos(payload)
            .expect("payload op should be present");
        assert_eq!(payload_op.opcode, OpCode::GetfieldGcPureI);
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GuardClass,
                "constant unbox must not emit GUARD_CLASS",
            );
        }
    }

    #[test]
    fn test_load_method_accepts_plain_python_instance_method() {
        install_test_hash_hook();
        let code = compile_exec("class C:\n    def f(self):\n        return self\nc = C()\n")
            .expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new(code.clone());
        frame
            .execute_frame(None, None)
            .expect("class body should execute");
        let instance = unsafe { pyre_object::w_dict_getitem_str(frame.get_w_globals(), "c") }
            .expect("namespace should contain c");

        install_test_jitcode(&code, frame.pycode);
        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(frame.pycode);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: (&mut *frame) as *mut pyre_interpreter::PyFrame as usize,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let instance_ref = ctx.const_ref(instance as i64);
        <MIFrame as SharedOpcodeHandler>::push_value(
            &mut state,
            FrontendOp::new(instance_ref, ConcreteValue::Ref(instance)),
        )
        .expect("push instance");

        state.load_method("f").expect("load_method should succeed");
        let receiver = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("receiver should be present");
        let callable = <MIFrame as SharedOpcodeHandler>::pop_value(&mut state)
            .expect("callable should be present");

        // callmethod.py:66-68 fast path — a plain function found through the
        // type, not shadowed by an instance attribute, is pushed unbound as
        // `[w_descr, w_obj]`; the receiver slot carries the instance.
        assert_eq!(receiver.concrete.to_pyobj(), instance);
        unsafe {
            assert!(pyre_interpreter::function::is_function(
                callable.concrete.to_pyobj()
            ));
        }
    }

    #[test]
    fn test_init_symbolic_skips_heap_array_read_for_standard_virtualizable() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = PyFrame::new(code.clone());
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        install_test_jitcode(&code, frame.pycode);
        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.become_active_vable_owner();

        sym.init_symbolic(&mut ctx, frame_ptr);

        assert_eq!(sym.locals_cells_stack_array_ref, OpRef::NONE);
        let recorder = ctx.into_recorder();
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GetfieldRawI,
                "standard virtualizable init should not read locals array from heap"
            );
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_float_object() {
        let mut state = empty_state();
        let meta = empty_meta();
        let value = 3.25f64;
        let descr = crate::descr::w_float_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: crate::descr::W_FLOAT_GC_TYPE_ID,
            fields: vec![(
                crate::descr::float_floatval_descr().index(),
                MaterializedValue::Value(value.to_bits() as i64),
            )],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("float virtual should materialize");

        unsafe {
            assert!(is_float(ptr.0 as PyObjectRef));
            assert_eq!(w_float_get_value(ptr.0 as PyObjectRef), value);
        }
    }

    /// W_BoolObject has a dedicated GC type id (`W_BOOL_GC_TYPE_ID = 5`)
    /// but shares `W_IntObject`'s layout, distinguished by the
    /// `&BOOL_TYPE` vtable. This test passes `type_id: 0` to confirm the
    /// orthodox materializer dispatches on `descr.vtable()` rather than
    /// `type_id`.
    #[test]
    fn test_materialize_virtual_ref_reconstructs_bool_object() {
        use pyre_object::boolobject::w_bool_get_value;
        let mut state = empty_state();
        let meta = empty_meta();
        let descr = crate::descr::w_bool_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: 0,
            fields: vec![(
                crate::descr::bool_intval_descr().index(),
                MaterializedValue::Value(1),
            )],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("bool virtual should materialize");

        unsafe {
            assert!(is_bool(ptr.0 as PyObjectRef));
            assert!(w_bool_get_value(ptr.0 as PyObjectRef));
        }
    }

    /// Second type_id=0 case: W_IntRangeIterator has three i64 fields.
    /// Verifies generic field replay at different offsets (no
    /// hard-coded PAYLOAD_0/PAYLOAD_1 dispatch).
    #[test]
    fn test_materialize_virtual_ref_reconstructs_range_iterator() {
        use pyre_object::functional::W_IntRangeIterator;
        let mut state = empty_state();
        let meta = empty_meta();
        let descr = crate::descr::w_range_iter_size_descr();
        let materialized = MaterializedVirtual::Obj {
            descr: Some(descr.clone()),
            type_id: 0,
            fields: vec![
                (
                    crate::descr::range_iter_current_descr().index(),
                    MaterializedValue::Value(7),
                ),
                (
                    crate::descr::range_iter_remaining_descr().index(),
                    MaterializedValue::Value(42),
                ),
                (
                    crate::descr::range_iter_step_descr().index(),
                    MaterializedValue::Value(3),
                ),
            ],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("range-iter virtual should materialize");

        unsafe {
            let iter = &*(ptr.0 as *const W_IntRangeIterator);
            assert_eq!(iter.current, 7);
            assert_eq!(iter.remaining, 42);
            assert_eq!(iter.step, 3);
        }
    }

    // Needs a real raw-buffer allocator (`func` passed as a valid function
    // pointer into `bh_call_i`) and a callinfocollection carrying
    // OS_RAW_MALLOC_VARSIZE_CHAR. The current test supplies `func: 0` with no
    // registered calldescr, so `materialize_virtual_raw_buffer` cannot run the
    // resume.py:1456 bh_call_i; wiring up the backend allocator is a follow-up.
    #[ignore = "VRawBufferInfo recovery requires MetaInterp.callinfocollection with OS_RAW_MALLOC_VARSIZE_CHAR; current test has func=0/no calldescr"]
    #[test]
    fn test_materialize_virtual_ref_reconstructs_list_from_raw_buffer_ref() {
        ensure_test_callbacks();
        let mut state = empty_state();
        let meta = empty_meta();
        let first = w_int_new(2);
        let second = w_int_new(4);

        let raw_items = MaterializedVirtual::RawBuffer {
            func: 0,
            size: 16,
            offsets: vec![0, 8],
            descrs: vec![
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
                    len_offset: None,
                },
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
                    len_offset: None,
                },
            ],
            values: vec![
                MaterializedValue::Value(first as i64),
                MaterializedValue::Value(second as i64),
            ],
        };
        let raw_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &raw_items,
            &[],
        )
        .expect("raw items buffer should materialize");

        let list_virtual = MaterializedVirtual::Obj {
            descr: None,
            type_id: 0,
            fields: vec![
                (
                    crate::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&LIST_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::descr::list_length_descr().index(),
                    MaterializedValue::Value(2),
                ),
                (
                    crate::descr::list_items_descr().index(),
                    MaterializedValue::VirtualRef(0),
                ),
            ],
        };

        let list_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            1,
            &list_virtual,
            &[Some(raw_ptr)],
        )
        .expect("list virtual should materialize");

        unsafe {
            assert!(is_list(list_ptr.0 as PyObjectRef));
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 0).unwrap()),
                2
            );
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_virtualizable_array_lengths_uses_full_array() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let full_len = frame.locals_w().len();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        state.set_valuestackdepth(2);
        let info = crate::frame_layout::build_pyframe_virtualizable_info();

        // virtualizable.py:86 parity: full array length, not valuestackdepth.
        assert_eq!(
            <PyreJitState as JitState>::virtualizable_array_lengths(
                &state,
                &empty_meta(),
                "frame",
                &info,
            ),
            Some(vec![full_len])
        );
    }

    #[test]
    #[ignore = "PyreSym::new_uninit hits the Phase X-1 skeleton-panic since the \
                debug-only fallback was removed; needs a populated-jitcode harness."]
    fn test_load_local_checked_value_respects_symbolic_local_type() {
        let run_case = |symbolic_type: Type, name: &str, expected_guard: Option<OpCode>| {
            let mut ctx = TraceCtx::for_test(1);
            // The slot type matches `symbolic_type` (resoperation.py:719/727/739
            // InputArg{Int,Float,Ref}); Void has no inputarg variant in RPython.
            let local = OpRef::input_arg_typed(0, symbolic_type);
            let mut sym = PyreSym::new_uninit(OpRef::NONE);
            sym.registers_r = vec![local];
            sym.symbolic_local_types = vec![symbolic_type];
            sym.nlocals = 1;

            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                parent_frames: Vec::new(),
                pending_result_stack_idx: None,
                pending_result_type: None,
                pending_inline_frame: None,
                residual_call_pc: None,
                loop_close_marker_jit_pc: None,
                orgpc: 0,
                concrete_frame_addr: 0,
                pre_opcode_registers_r: None,
                pre_opcode_semantic_depth: None,
            };

            let loaded =
                <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, name)
                    .expect("local should load");
            assert_eq!(loaded.opref, local);

            let tree_loop = ctx.into_tree_loop();
            assert_eq!(tree_loop.ops.last().map(|op| op.opcode), expected_guard);
        };

        run_case(Type::Int, "j", None);
        run_case(Type::Ref, "b", Some(OpCode::GuardNonnull));
    }

    #[test]
    fn test_store_local_value_preserves_ref_slot_without_reboxing() {
        // RPython Box.type parity: `_opimpl_setarrayitem_vable`
        // (pyjitpl.py:1242-1247) writes the value's Ref box directly —
        // it never reboxes at the consumer. `locals_cells_stack_w` is a
        // W_Root array (virtualizable.py:86-98), so the producer side
        // (`push_typed_value` on the operand stack) is responsible for
        // wrapping Int/Float with `wrapint` / `wrapfloat` before the
        // value flows into the stack or local slot. Pin the contract
        // here: storing a pre-wrapped Ref leaves `registers_r[idx]`
        // pointing at the same OpRef with no additional op emitted.
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.locals_w_mut()[0] = w_int_new(41);
        frame.fix_array_ptrs();
        let mut ctx = TraceCtx::for_test(1);
        // Pre-wrapped Ref — this is the shape producers hand us.
        let ref_value = ctx.const_ref(pyre_object::PY_NULL as i64);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        state
            .with_ctx(|this, ctx| this.store_local_value(ctx, 0, ref_value, ConcreteValue::Null))
            .expect("store of pre-wrapped Ref should succeed");
        assert_eq!(
            state.sym().registers_r[0],
            ref_value,
            "Ref value must be stored as-is, not reboxed",
        );
        assert_eq!(state.sym().symbolic_local_types[0], Type::Ref);
    }

    #[test]
    fn test_trace_binary_value_boxes_typed_raw_operands_for_python_helper() {
        let code = compile_exec("x = 1.0 ** 2").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef::input_arg_float(0);
        let rhs = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(code_ref);
        sym.registers_r = vec![lhs, rhs];
        sym.symbolic_local_types = vec![Type::Float, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = <MIFrame as TraceHelperAccess>::trace_binary_value(
            &mut state,
            lhs,
            rhs,
            BinaryOperator::Power,
        )
        .expect("generic helper call should box raw operands first");

        let recorder = ctx.into_recorder();
        // GuardNotForced (pyjitpl.py:2079) + GuardNoException (pyjitpl.py:2082)
        // follow the residual may-force Call*, so look up the Call* op
        // explicitly rather than via `ops().last()`.
        let call = recorder
            .ops()
            .iter()
            .rev()
            .find(|op| {
                matches!(
                    op.opcode,
                    OpCode::CallI
                        | OpCode::CallR
                        | OpCode::CallF
                        | OpCode::CallN
                        | OpCode::CallMayForceI
                        | OpCode::CallMayForceR
                        | OpCode::CallMayForceF
                        | OpCode::CallMayForceN
                )
            })
            .expect("call op should be present");
        assert_ne!(call.arg(0).to_opref(), lhs);
        assert_ne!(call.arg(1).to_opref(), rhs);
    }

    #[test]
    fn test_trace_known_builtin_call_boxes_typed_raw_args_for_python_helper_boundary() {
        let code = compile_exec("x = abs(1)").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut ctx = TraceCtx::for_test(2);
        let callable = OpRef::input_arg_ref(0);
        let arg = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = jitcode_for(code_ref);
        sym.registers_r = vec![callable, arg];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let _ = state
            .trace_known_builtin_call(callable, &[arg])
            .expect("known builtin helper boundary should box raw int args");

        let recorder = ctx.into_recorder();
        // GuardNoException (pyjitpl.py:2082) follows the residual Call*, so
        // look up the Call* op explicitly rather than via `ops().last()`.
        let call = recorder
            .ops()
            .iter()
            .rev()
            .find(|op| {
                matches!(
                    op.opcode,
                    OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
                )
            })
            .expect("call op should be present");
        assert_ne!(call.getarglist().last().map(|a| a.to_opref()), Some(arg));
    }

    #[test]
    fn test_compare_value_direct_emits_raw_truth_for_immediate_branch_consumer() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("if 1 < 2:\n    x = 3\n").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");
        let branch_pc = ((compare_pc + 1)..code.instructions.len())
            .find(|&pc| {
                decode_instruction_at(&code, pc)
                    .map(|(instruction, _)| instruction_consumes_comparison_truth(instruction))
                    .unwrap_or(false)
            })
            .expect("test bytecode should contain POP_JUMP_IF after COMPARE_OP");

        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        // Typed `InputArgInt` inputarg slots — `compare_value_direct`
        // routes through `is_int_typed` lookups (history.py:220
        // box.type) and Untyped slots silently fall through to the
        // boxed-bool path under variant-aware Eq.
        let lhs = OpRef::input_arg_int(0);
        let rhs = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: branch_pc,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let concrete_lhs = w_int_new(10);
        let concrete_rhs = w_int_new(20);
        let result = state
            .compare_value_direct(
                lhs,
                rhs,
                ComparisonOperator::Less,
                concrete_lhs,
                concrete_rhs,
            )
            .expect("int comparison should trace");

        let recorder = ctx.into_recorder();
        let mut saw_cmp = false;
        let mut saw_bool_call = false;
        let mut saw_bool_unbox = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::IntLt {
                saw_cmp = true;
            }
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
            if op.opcode == OpCode::GetfieldGcI {
                saw_bool_unbox = true;
            }
        }
        assert!(
            saw_cmp,
            "branch compare should still emit raw int comparison"
        );
        assert_eq!(
            result,
            // IntLt at op pos 2 — `IntOp` mixin (resoperation.py:564),
            // `box.type='i'` (history.py:220).
            OpRef::int_op(2),
            "with two input args, the immediate branch consumer should receive the raw comparison truth"
        );
        assert!(
            !saw_bool_call,
            "immediate branch consumer should not allocate a bool object"
        );
        assert!(
            !saw_bool_unbox,
            "immediate branch consumer should not unbox a transient bool object"
        );
    }

    #[test]
    fn test_compare_value_direct_boxes_bool_when_not_immediately_consumed_by_branch() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut frame = PyFrame::new(code);
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        // Typed `InputArgInt` inputarg slots — `compare_value_direct`
        // routes through `is_int_typed` lookups (history.py:220
        // box.type) and Untyped slots silently fall through to the
        // boxed-bool path under variant-aware Eq.
        let lhs = OpRef::input_arg_int(0);
        let rhs = OpRef::input_arg_int(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let concrete_lhs = w_int_new(10);
        let concrete_rhs = w_int_new(20);
        let _ = state
            .compare_value_direct(
                lhs,
                rhs,
                ComparisonOperator::Less,
                concrete_lhs,
                concrete_rhs,
            )
            .expect("non-branch compare should trace");

        let recorder = ctx.into_recorder();
        let mut saw_bool_call = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_raw_pos(pos) else {
                continue;
            };
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
        }
        assert!(
            saw_bool_call,
            "non-branch compare should continue to materialize a Python bool result"
        );
    }

    #[test]
    fn test_next_instruction_consumes_comparison_truth_skips_extended_arg_trivia() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");

        let first_after_compare = decode_instruction_at(&code, compare_pc + 1)
            .map(|(instruction, _)| instruction)
            .expect("bytecode should continue after COMPARE_OP");
        assert!(
            instruction_is_trivia_between_compare_and_branch(first_after_compare),
            "test source should force trivia between COMPARE_OP and POP_JUMP_IF"
        );

        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        install_test_jitcode(&code, code_ref);
        let mut frame = PyFrame::new(code.clone());
        frame.fix_array_ptrs();

        let mut ctx = TraceCtx::for_test(2);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: compare_pc + 1,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        assert!(
            state.next_instruction_consumes_comparison_truth(),
            "branch fusion should survive EXTENDED_ARG/other trivia before the branch"
        );
    }

    // the preserves-comparison-truth-across-extended-arg-trivia test was
    // removed: the last_comparison_truth cache no longer exists. Trivia
    // skipping between COMPARE_OP and POP_JUMP_IF* is now verified through
    // the fused-dispatch path (try_fused_compare_goto_if_not), which uses
    // semantic_fallthrough_pc to locate the branch across trivia.

    // Tests for concrete_popped_value, concrete_binary_operands,
    // concrete_store_subscr_operands removed: these stack-based concrete
    // read methods were replaced by direct FrontendOp.concrete parameter passing.

    // test_concrete_branch_truth_reads_last_popped_slot removed:
    // concrete_branch_truth now requires explicit concrete parameter.

    // test_concrete_branch_truth_uses_cached_comparison_truth_without_stack_value
    // and test_truth_value_direct_caches_concrete_truth_for_raw_int_branch_consumer
    // (consolidated on main as test_branch_truth_concrete_cache_paths) removed:
    // all three exercised the last_comparison_truth / last_comparison_concrete_truth
    // cache fields. Those fields were eliminated when the
    // try_fused_compare_goto_if_not dispatcher consumed the fused
    // COMPARE_OP + POP_JUMP_IF* pair directly (pyjitpl.py:541-556 parity).

    // test_close_loop_args_at_target_pc_preserves_virtualizable_stack moved
    // to `pyre-jit` so close_loop_args_at runs with a real compiled jitcode.

    /// `pyjitpl.py:74-90 MIFrame.setup` parity: after
    /// `setup_kind_register_banks` runs, `registers_i` / `registers_r` /
    /// `registers_f` are sized to `num_regs_X + len(constants_X)` and
    /// the trailing `[num_regs_X..)` slots hold the constant-pool
    /// OpRefs from `pyjitpl.py:97-119 copy_constants`
    /// (`ctx.const_int(constants_i[i])` for the int bank, `ctx.const_ref`
    /// for the ref bank, `ctx.const_float` for the float bank). The
    /// leading `[..num_regs_X)` register slots stay `OpRef::NONE` (the
    /// `CONST_NULL`-shaped placeholder).
    #[test]
    fn test_setup_kind_register_banks_sizes_per_num_regs_and_consts() {
        let mut runtime_jc = majit_metainterp::jitcode::JitCode::default();
        runtime_jc.body_mut().c_num_regs_i = 3;
        runtime_jc.body_mut().c_num_regs_r = 4;
        runtime_jc.body_mut().c_num_regs_f = 2;
        runtime_jc.body_mut().constants_i = vec![100, 200];
        runtime_jc.body_mut().constants_r = vec![0xAABB_CCDD_u64 as i64];
        runtime_jc.body_mut().constants_f = vec![3.14_f64.to_bits() as i64];

        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null());
        pyjit.jitcode = std::sync::Arc::new(runtime_jc);
        let inner_jc = super::JitCode {
            index: -1,
            payload: std::sync::Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;

        assert_eq!(sym.registers_i.len(), 0);
        assert_eq!(sym.registers_r.len(), 0);
        assert_eq!(sym.registers_f.len(), 0);

        let mut ctx = TraceCtx::for_test(1);
        sym.setup_kind_register_banks(&mut ctx);

        // bank size = num_regs_X + len(constants_X)
        assert_eq!(sym.registers_i.len(), 3 + 2, "registers_i sized to 5");
        assert_eq!(sym.registers_r.len(), 4 + 1, "registers_r sized to 5");
        assert_eq!(sym.registers_f.len(), 2 + 1, "registers_f sized to 3");

        // Leading register slots — CONST_NULL placeholder per
        // pyjitpl.py:86-90.
        for i in 0..3 {
            assert_eq!(sym.registers_i[i], OpRef::NONE, "registers_i[{i}] reg-slot");
        }
        for i in 0..4 {
            assert_eq!(sym.registers_r[i], OpRef::NONE, "registers_r[{i}] reg-slot");
        }
        for i in 0..2 {
            assert_eq!(sym.registers_f[i], OpRef::NONE, "registers_f[{i}] reg-slot");
        }
        // Trailing constant slots — copy_constants populated. The pool
        // dedups by (type, raw value) so each kind's trailing slot
        // resolves to a distinct constant OpRef.
        for i in 0..2 {
            let op = sym.registers_i[3 + i];
            assert_ne!(op, OpRef::NONE, "registers_i[{}] constant slot", 3 + i);
            let val = ctx
                .constants_get_value(op)
                .expect("constants pool resolves trailing int slot");
            assert_eq!(val, majit_ir::Value::Int(100 + 100 * i as i64));
        }
        let op_r = sym.registers_r[4];
        assert_ne!(op_r, OpRef::NONE);
        assert!(matches!(
            ctx.constants_get_value(op_r),
            Some(majit_ir::Value::Ref(_))
        ));
        let op_f = sym.registers_f[2];
        assert_ne!(op_f, OpRef::NONE);
        assert!(matches!(
            ctx.constants_get_value(op_f),
            Some(majit_ir::Value::Float(_))
        ));

        // Calling twice does not shrink the register banks. `history.py:220
        // ConstInt`, `:261 ConstFloat`, `:307 ConstPtr` are all fresh-alloc
        // per construction; `Const.same_constant` (history.py:204) is the
        // upstream value-equality predicate. Assert value-equality across
        // both calls via `constants_get_value` rather than OpRef identity,
        // independent of whether the pool internally dedups.
        let trailing_i_value_before = ctx
            .constants_get_value(sym.registers_i[3])
            .expect("first-call trailing int slot resolves to a constant");
        let trailing_r_value_before = ctx
            .constants_get_value(sym.registers_r[4])
            .expect("first-call trailing ref slot resolves to a constant");
        let trailing_f_value_before = ctx
            .constants_get_value(sym.registers_f[2])
            .expect("first-call trailing float slot resolves to a constant");
        sym.setup_kind_register_banks(&mut ctx);
        assert_eq!(sym.registers_i.len(), 5);
        assert_eq!(sym.registers_r.len(), 5);
        assert_eq!(sym.registers_f.len(), 3);
        assert_eq!(
            ctx.constants_get_value(sym.registers_i[3]),
            Some(trailing_i_value_before),
        );
        assert_eq!(
            ctx.constants_get_value(sym.registers_r[4]),
            Some(trailing_r_value_before),
        );
        assert_eq!(
            ctx.constants_get_value(sym.registers_f[2]),
            Some(trailing_f_value_before),
        );

        // SAFETY: drop the boxed JitCode; sym.jitcode now dangles but goes
        // out of scope at the end of this test.
        unsafe {
            let _ = Box::from_raw(inner_jc_ptr);
        }
    }

    /// `setup_kind_register_banks` is safe to call when `self.jitcode`
    /// points at the thread-local `null_jitcode()` placeholder — the
    /// skeleton's `num_regs_and_consts_X` values are zero and the
    /// constant pools are empty so the resize and the constant fill are
    /// both no-ops.
    #[test]
    fn test_setup_kind_register_banks_is_no_op_for_null_jitcode_placeholder() {
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        // sym.jitcode was initialized by new_uninit to null_jitcode().
        let mut ctx = TraceCtx::for_test(1);
        sym.setup_kind_register_banks(&mut ctx);
        assert_eq!(sym.registers_i.len(), 0);
        assert_eq!(sym.registers_f.len(), 0);
        assert_eq!(sym.registers_r.len(), 0);
    }

    #[test]
    fn test_setup_bridge_sym_preserves_resumed_stack_tail() {
        use majit_ir::resumedata::{RebuiltFrame, RebuiltValue};
        use majit_metainterp::jitcode::JitCodeBuilder;
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();

        let raw_code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = PyFrame::new(raw_code);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let code_ref = frame.pycode as *const ();

        // Seed the live/ -> BC_LIVE opcode mapping so the jitcode decoder
        // recognises the pc=0 liveness marker. `op_live` defaults to
        // `u8::MAX` until published; without this the decode underflows
        // (`pc -= OFFSET_SIZE + 1` at jitcode.rs). `publish_state` clears
        // `all_liveness`, so it must run before `intern_liveness` below.
        {
            let mut insns = indexmap::IndexMap::new();
            insns.insert(
                "live/".to_string(),
                majit_metainterp::jitcode::insns::BC_LIVE,
            );
            crate::assembler::publish_state(&insns, &[], 0, 0);
        }
        let live_off = crate::state::intern_liveness(&[], &[0, 1, 2], &[])
            .expect("bridge liveness must fit in the shared buffer");
        let mut builder = JitCodeBuilder::new();
        let patch = builder.live_placeholder();
        builder.patch_live_offset(patch, live_off);
        let runtime_jitcode = std::sync::Arc::new(builder.finish());
        let pyjit = std::sync::Arc::new(crate::PyJitCode::from_parts(
            runtime_jitcode,
            crate::PyJitCodeMetadata {
                after_residual_call_resume_marker_by_jit_pc: Vec::new(),
                after_residual_call_resume_pred_by_jit_pc: Vec::new(),
                first_jit_pc_by_py_pc: vec![0],
                block_head_py_by_jit_pc: vec![(0, 0)],
                carryfwd_resume_pc: Vec::new(),
                merge_entry_by_green: Vec::new(),
                pcdep_by_jit_pc: vec![(0, Vec::new())],
                depth_pred_by_jit_pc: vec![(0, 2)],
                depth_trivia_marker_by_jit_pc: vec![(0, Some(2))],
                depth_trivia_pred_by_jit_pc: vec![(0, Some(2))],
                resume_marker_marker_by_jit_pc: Vec::new(),
                resume_marker_pred_by_jit_pc: vec![(0, Some(0))],
                after_residual_marker_marker_by_jit_pc: Vec::new(),
                after_residual_marker_pred_by_jit_pc: Vec::new(),
                depth_at_py_pc: vec![2],
                result_color_at_pc: Vec::new(),
                result_color_by_jit_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                built_as_portal: true,
                stack_base: 1,
                max_stackdepth: 0,
                pcdep_color_slots: Vec::new(),
                const_ref_slots_at_pc: Vec::new(),
                const_ref_slots_by_jit_pc: Vec::new(),
                is_drained: true,
            },
            std::ptr::null(),
            false,
        ));
        let jitcode_index = METAINTERP_SD.with(|r| unsafe {
            let ptr = r.borrow_mut().jitcode_for(code_ref, Some(pyjit));
            (*ptr).index
        });

        let input_types = [
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // lastblock
            Type::Ref, // w_globals
            Type::Ref, // local0
            Type::Ref, // stack0
            Type::Ref, // stack1
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);
        // Slots 0 (frame) and 1 (ec) are both Ref-typed per `input_types`
        // — production `init_vable_indices` mints typed `InputArgRef`
        // variants here (resoperation.py:739, state.rs:1428-1438), so
        // variant-aware Eq (resoperation.rs:290) requires the matching
        // `OpRef::input_arg_ref` shape.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.frame = OpRef::input_arg_ref(0);
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 1;
        sym.valuestackdepth = 1;
        sym.concrete_vable_ptr = frame_ptr as *mut u8;

        let local0 = w_int_new(41) as i64;
        let stack0 = w_int_new(42) as i64;
        let stack1 = w_int_new(43) as i64;
        let globals = w_int_new(44) as i64;
        let fail_values = [
            frame_ptr as i64,
            0,
            77,
            code_ref as i64,
            3,
            0,
            0,
            globals,
            local0,
            stack0,
            stack1,
        ];
        let fail_types = [
            Type::Ref,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
        ];
        let resume_data = majit_metainterp::ResumeDataResult {
            frames: vec![RebuiltFrame {
                jitcode_index,
                pc: 0,
                values: vec![
                    RebuiltValue::Box(8, Type::Ref),
                    RebuiltValue::Box(9, Type::Ref),
                    RebuiltValue::Box(10, Type::Ref),
                ],
            }],
            virtualizable_values: vec![
                RebuiltValue::Box(0, Type::Ref),
                RebuiltValue::Box(2, Type::Int),
                RebuiltValue::Box(3, Type::Ref),
                RebuiltValue::Box(4, Type::Int),
                RebuiltValue::Box(5, Type::Ref),
                RebuiltValue::Box(6, Type::Ref),
                RebuiltValue::Box(7, Type::Ref),
                RebuiltValue::Box(8, Type::Ref),
                RebuiltValue::Box(9, Type::Ref),
                RebuiltValue::Box(10, Type::Ref),
            ],
            virtualref_values: Vec::new(),
            storage: None,
            num_failargs: fail_values.len() as i32,
            fail_arg_types: fail_types.to_vec(),
        };

        <PyreJitState as majit_metainterp::JitState>::setup_bridge_sym(
            &mut sym,
            &mut ctx,
            &resume_data,
            None,
            &fail_values,
            &fail_types,
        );

        assert_eq!(sym.valuestackdepth, 3);
        // setup_bridge_sym now restores typed `InputArg*` OpRefs from
        // `RebuiltValue::Box(idx, tp)` per state.rs:4647. The resolve
        // closure produces `OpRef::input_arg_typed(idx, tp)`; expectations
        // must match so variant-aware Eq lines up with the resolved
        // bridge inputarg list.
        assert_eq!(
            sym.registers_r,
            vec![
                OpRef::input_arg_ref(8),
                OpRef::input_arg_ref(9),
                OpRef::input_arg_ref(10)
            ]
        );
        assert_eq!(sym.symbolic_local_types, vec![Type::Ref]);
        assert_eq!(sym.symbolic_stack_types, vec![Type::Ref, Type::Ref]);
        assert_eq!(sym.bridge_local_oprefs, Some(vec![OpRef::input_arg_ref(8)]));
    }

    #[test]
    fn test_close_loop_args_preserves_ec_between_frame_and_virtualizable_header() {
        ensure_test_callbacks();
        let input_types = [
            Type::Ref, // frame
            Type::Ref, // ec
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // lastblock
            Type::Ref, // w_globals
            Type::Ref, // local0
            Type::Ref, // stack0
            Type::Ref, // stack1
        ];
        let mut ctx = TraceCtx::for_test_types(&input_types);

        // The vable static-field types come from `state.rs:1428-1438`
        // `#[vable(inputarg, type = ...)]` annotations: int/ref/int/ref/
        // ref/ref. Mint typed `OpRef::input_arg_*` variants matching
        // those tags so variant-aware Eq (resoperation.rs:290) lines up
        // with what the production `init_vable_indices` produces.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.execution_context = OpRef::input_arg_ref(1);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = OpRef::input_arg_int(2);
        sym.vable_pycode = OpRef::input_arg_ref(3);
        sym.vable_valuestackdepth = OpRef::input_arg_int(4);
        sym.vable_debugdata = OpRef::input_arg_ref(5);
        sym.vable_lastblock = OpRef::input_arg_ref(6);
        sym.vable_w_globals = OpRef::input_arg_ref(7);
        // local0 / stack0 / stack1 are Ref-typed per `symbolic_local_types`
        // / `symbolic_stack_types` below — the macro mints the matching
        // `InputArgRef` variant.
        sym.registers_r = vec![
            OpRef::input_arg_ref(8),
            OpRef::input_arg_ref(9),
            OpRef::input_arg_ref(10),
        ];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Null, ConcreteValue::Null];
        // Bind a populated jitcode with a real PyCode so the
        // close_loop_args snapshot encoder (trace_opcode.rs:4874) reads
        // code.varnames/ncells/max_stackdepth instead of null-derefing. No
        // frame here, so build a standalone wrapper (pattern B). The three
        // live Ref colors map to local0/stack0/stack1 in registers_r.
        let code = compile_exec("len(x)").expect("test code should compile");
        let w_code = pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
            as *const ();
        let jc_ptr = bind_real_jitcode(&mut sym, w_code, &[], &[0, 1, 2], &[]);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));

        assert_eq!(jump_args.len(), 11);
        assert_eq!(jump_args[0], OpRef::input_arg_ref(0));
        assert_eq!(jump_args[1], OpRef::input_arg_ref(1));
        assert_eq!(
            &jump_args[8..],
            &[
                OpRef::input_arg_ref(8),
                OpRef::input_arg_ref(9),
                OpRef::input_arg_ref(10)
            ]
        );
        assert_eq!(state.sym().execution_context, OpRef::input_arg_ref(1));

        unsafe {
            let _ = Box::from_raw(jc_ptr);
        }
    }

    #[test]
    fn test_close_loop_args_uses_full_root_virtualizable_array_capacity() {
        use pyre_interpreter::pyframe::PyFrame;

        ensure_test_callbacks();
        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = PyFrame::new(code);
        frame.locals_w_mut()[0] = w_int_new(41);
        frame.locals_w_mut()[1] = w_int_new(7);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let array_len =
            concrete_frame_array_len(frame_ptr).expect("frame should expose locals_cells_stack_w");
        assert!(
            array_len >= 2,
            "test frame must have stack capacity beyond nlocals"
        );

        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let mut input_types = vec![
            Type::Ref, // frame
            Type::Int, // next_instr
            Type::Ref, // pycode
            Type::Int, // valuestackdepth
            Type::Ref, // debugdata
            Type::Ref, // lastblock
            Type::Ref, // w_globals
        ];
        input_types.extend(std::iter::repeat(Type::Ref).take(array_len));
        let mut ctx = TraceCtx::for_test_types(&input_types);

        // Mint typed `OpRef::input_arg_*` matching each `input_types`
        // slot — production `init_vable_indices` (state.rs:1428-1438
        // `#[vable(inputarg, type = ...)]`) always selects `InputArgInt`
        // / `InputArgRef` (resoperation.py:719/739) per the static-field
        // tag, so variant-aware Eq (resoperation.rs:290) requires the
        // matching variant here too.
        let mut sym = PyreSym::new_uninit(OpRef::input_arg_ref(0));
        sym.nlocals = 1;
        sym.valuestackdepth = 1;
        sym.vable_last_instr = OpRef::input_arg_int(1);
        sym.vable_pycode = OpRef::input_arg_ref(2);
        sym.vable_valuestackdepth = OpRef::input_arg_int(3);
        sym.vable_debugdata = OpRef::input_arg_ref(4);
        sym.vable_lastblock = OpRef::input_arg_ref(5);
        sym.vable_w_globals = OpRef::input_arg_ref(6);
        sym.registers_r = vec![OpRef::input_arg_ref(7)];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = Vec::new();
        sym.concrete_vable_ptr = frame_ptr as *mut u8;
        // Bind a populated jitcode (real PyCode = frame.pycode) so
        // any sym.jitcode/payload read is well-formed; the single live Ref
        // color 0 maps to local0 in registers_r.
        let jc_ptr = bind_real_jitcode(&mut sym, frame.pycode, &[], &[0], &[]);
        // Seed ctx.virtualizable_array_lengths with the full root array
        // capacity so close_loop_args' target_array_capacity equals the
        // concrete array length (the nlocals+stack_only fallback would only
        // yield 1). Empty input_values disables the concrete shadow; the
        // non-owner path then reads local0 from registers_r and PY_NULLs the
        // dead stack tail (trace_opcode.rs:3470).
        seed_virtualizable_boxes(
            &mut ctx,
            OpRef::input_arg_ref(0),
            majit_ir::Value::Ref(majit_ir::GcRef(frame_ptr)),
            &[
                OpRef::input_arg_int(1),
                OpRef::input_arg_ref(2),
                OpRef::input_arg_int(3),
                OpRef::input_arg_ref(4),
                OpRef::input_arg_ref(5),
                OpRef::input_arg_ref(6),
            ],
            &[OpRef::input_arg_ref(7)],
            array_len,
            &[],
            std::ptr::null(),
        );

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            residual_call_pc: None,
            loop_close_marker_jit_pc: None,
            orgpc: 0,
            concrete_frame_addr: frame_ptr,
            pre_opcode_registers_r: None,
            pre_opcode_semantic_depth: None,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));

        assert_eq!(
            jump_args.len(),
            n + array_len,
            "root close_loop_args must carry full virtualizable array capacity"
        );
        assert!(
            jump_args[n..].iter().all(|arg| !arg.is_none()),
            "full-capacity root jump args must materialize missing stack slots"
        );

        unsafe {
            let _ = Box::from_raw(jc_ptr);
        }
    }
}

// ── Virtualizable configuration ──────────────────────────────────────
//
// PyPy's `pypy/interpreter/pyframe.py` declares:
//
//     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
//                         'last_instr', ...]
//
// Our Rust equivalent uses explicit byte offsets instead of name-based
// introspection. The JIT optimizer's Virtualize pass uses this info
// to keep frame fields in CPU registers, eliminating heap accesses
// for LoadFast/StoreFast and stack push/pop during compiled code.
//
// The shared frame layout contract now also lives in `pyre-jit/src/frame_layout.rs`
// so the tracer can compute the same offsets without depending on
// `pyre-interpreter`. Driver registration still happens in `pyre-jit/src/eval.rs`.

/// TODO: deferred `MetaInterp.perform_call`
/// (`rpython/jit/metainterp/pyjitpl.py`) for pyre.  RPython constructs
/// and pushes the callee `MIFrame` directly inside `perform_call`; pyre
/// returns this struct from the trace step so the framestack mutation
/// happens in `MetaInterpreter::push_inline_frame` after the trace
/// handler releases its borrow on `MetaInterpFrame`.  No upstream
/// counterpart.
pub struct PendingInlineFrame {
    pub sym: PyreSym,
    pub concrete_frame: pyre_interpreter::pyframe::PyFrame,
    pub drop_frame_opref: Option<OpRef>,
    pub green_key: u64,
    /// Raw `(code_ptr, target_pc)` greenkey components for element-
    /// wise recursion-depth comparison. `green_key` above is the u64
    /// hash derived from this pair and stays the identity key for
    /// HashMap lookups; `green_key_raw` element-wise equality matches
    /// rpython/jit/metainterp/pyjitpl.py:1396-1401 `for i in
    /// range(len(gk)): if not gk[i].same_constant(greenboxes[i])`
    /// without the hash-collision risk a u64-only comparison carries.
    pub green_key_raw: (usize, usize),
    /// opencoder.py:819-834: accumulated parent frame chain.
    pub parent_frames: Vec<ResumeFrameState>,
    pub nargs: usize,
    pub caller_result_stack_idx: Option<usize>,
    pub caller_result_type: Option<Type>,
    /// Symbolic callable + arg OpRefs of the CALL that pushed this frame.
    /// Consumed by the inline back-edge CALL_ASSEMBLER path
    /// (`do_recursive_call`) to shape the GuardNotForced /
    /// GuardNoException resume via `push_call_replay_stack` — on guard
    /// failure the parent re-executes the CALL, mirroring the residual
    /// call capture. `OpRef::NONE` when the path is unavailable
    /// (bridge-reconstructed frames).
    pub replay_callable: OpRef,
    pub replay_args: Vec<OpRef>,
}

/// Reify one Ref-bank recipe slot as the boxed `W_Root` pointer that
/// `locals_cells_stack_w` (a W_Root array, virtualizable.py:86-98) must
/// hold. `reconstruct_inline_recipe` gates int/float banks out of multi-
/// frame recipes, so a live slot is always `Value::Ref` (the decoded box,
/// possibly null) and a dead slot is `Value::Void`; the int/float arms only
/// exist defensively (box back to a heap object) and never fire in practice.
fn recipe_slot_to_pyobj(v: majit_ir::Value) -> PyObjectRef {
    match v {
        majit_ir::Value::Ref(gc) => gc.0 as PyObjectRef,
        majit_ir::Value::Void => pyre_object::PY_NULL,
        majit_ir::Value::Int(i) => w_int_new(i),
        majit_ir::Value::Float(f) => pyre_object::w_float_new(f),
    }
}

/// Recover the callee module's globals OBJECT for a reconstructed inline frame.
///
/// The globals-stamped `PyCode` wrapper is recovered from the raw code pointer
/// through the `code_ptr → live wrapper` registry. Returns `PY_NULL` when no
/// live wrapper is registered or it carries no globals yet — the callers
/// (`reconstruct_inline_recipe` and `assemble_bridge_inline_pending`) treat a
/// null result as "decline the multi-frame path".
pub(crate) fn recover_inline_callee_globals(code_ptr: *const ()) -> pyre_object::PyObjectRef {
    let live = pyre_interpreter::live_code_wrapper(code_ptr);
    if !live.is_null() {
        let globals = unsafe { pyre_interpreter::w_code_get_w_globals(live) };
        if !globals.is_null() {
            return globals;
        }
    }
    pyre_object::PY_NULL
}

/// Assemble one decoded inline-callee [`ReconstructRecipe`]
/// into a [`PendingInlineFrame`] (concrete `PyFrame` + symbolic `PyreSym`)
/// for `trace_bytecode` to push onto the bridge framestack.
///
/// Mirrors `build_pending_inline_frame`'s IR-FREE FAST branch
/// (`trace_opcode.rs:6035-6084`): set `jitcode` FIRST (the
/// `setup_kind_register_banks` debug_assert requires a non-null jitcode),
/// then fill the per-bank register files + the lazy-boxing concrete shadow.
/// It emits NO trace IR — the forward FAST branch's guard_value /
/// callee-frame helper / GUARD_NO_EXCEPTION belong to a LIVE caller making a
/// call; a reconstructed suspended frame is simply pushed.
///
/// The callee frame's globals come from its OWN pycode (`recipe.code_ptr`),
/// matching `pyframe.py:128-132 get_w_globals_storage()` where a frame's globals
/// derive from its promoted pycode rather than the caller; a cross-module
/// inlined callee's LOAD_GLOBAL then resolves against the callee module's
/// namespace. `execution_context` is the per-thread singleton and comes from
/// the bridge's root concrete frame.
///
/// `parent_frames` is the OUTER chain (immediate parent first) the drain
/// builds from the framestack; `push_inline_frame` stamps
/// `parent_frames.first().pending_result_*` with the caller result slot.
pub(crate) fn assemble_bridge_inline_pending(
    ctx: &mut TraceCtx,
    recipe: &ReconstructRecipe,
    execution_context: *const pyre_interpreter::PyExecutionContext,
    parent_frames: Vec<ResumeFrameState>,
) -> PendingInlineFrame {
    use pyre_interpreter::pyframe::PyFrame;

    let nlocals = recipe.nlocals;
    let valuestackdepth = recipe.valuestackdepth;

    // Recover the callee's globals-stamped PyCode wrapper from the
    // `code_ptr -> live-wrapper` registry. It is the SAME wrapper the forward
    // inline push obtains via `getcode(callable)` (raw code identity is 1:1
    // with its wrapper), so keying the reconstructed frame's greenkey by it
    // preserves the recursion-depth / inline-position identity.
    let w_code = pyre_interpreter::live_code_wrapper(recipe.code_ptr) as *const ();

    // pyframe.py:128-132 get_w_globals_storage(): a frame's globals come from its OWN
    // pycode (`jit.promote(self.pycode).w_globals`), not the caller. Resolve
    // the callee's globals OBJECT for `recipe.code_ptr` — the same
    // `pycode.w_globals` the callee module exposes through its function's
    // `w_func_globals_obj` — so a cross-module inlined callee's LOAD_GLOBAL
    // sees the callee module's namespace. `reconstruct_inline_recipe` aborts
    // the multi-frame path when the callee code has no resolved globals
    // object, so this is non-null here.
    let w_globals = recover_inline_callee_globals(recipe.code_ptr);

    // resume.py:1042-1057 newframe + reload: build a fresh concrete frame for
    // the callee's own pycode wrapper and seed
    // `locals_cells_stack_w[0..valuestackdepth]` from the decoded boxes. The
    // callee has no cells/freevars (gated in `reconstruct_inline_recipe`), so
    // `closure = PY_NULL` and the array layout is `[locals | stack]` with
    // `stack_base() == nlocals`. The builder re-derives the storage proxy from
    // the (non-null) globals object, so the raw `globals` arg is unused — pass
    // null rather than reading the off-GC `code.w_globals` proxy.
    let mut concrete_frame = PyFrame::new_for_call_with_closure_and_globals_obj(
        w_code,
        &[],
        std::ptr::null_mut(),
        w_globals,
        execution_context,
        pyre_object::PY_NULL,
        pyre_interpreter::pyframe::FrameLocalsArrayAllocation::StdAlloc,
    );
    {
        let arr = concrete_frame.locals_w_mut();
        for k in 0..nlocals {
            arr[k] = recipe_slot_to_pyobj(recipe.concrete_r[k]);
        }
    }
    for k in nlocals..valuestackdepth {
        concrete_frame.push(recipe_slot_to_pyobj(recipe.concrete_r[k]));
    }
    // last_instr is one before the recipe's Python pc so next_instr() resumes there.
    concrete_frame
        .set_last_instr_from_next_instr(
            backxlat_py_pc(recipe.jitcode_index, recipe.jitcode_pc) as usize
        );

    // Symbolic side: mirror the FAST branch field-for-field.
    let mut sym = PyreSym::new_uninit(OpRef::NONE);
    sym.nlocals = nlocals;
    sym.valuestackdepth = valuestackdepth;
    // jitcode FIRST — setup_kind_register_banks debug_asserts non-null.
    sym.jitcode = jitcode_for(w_code);
    // The Ref bank IS the unified locals+stack register file decoded by
    // `reconstruct_inline_recipe`; int/float banks are empty (gated out).
    sym.registers_i = recipe.registers_i.clone();
    sym.registers_r = recipe.registers_r.clone();
    sym.registers_f = recipe.registers_f.clone();
    // locals_cells_stack_w is a W_Root array — every live slot is Ref.
    sym.symbolic_local_types = vec![Type::Ref; nlocals];
    sym.symbolic_stack_types = vec![Type::Ref; valuestackdepth - nlocals];
    // Box-identity shadow: unbox per the FAST branch, preserving PY_NULL
    // frame slots as Ref(PY_NULL) so uninitialized locals stay distinct from
    // untracked values.
    sym.concrete_locals = (0..nlocals)
        .map(|k| concrete_value_from_slot(recipe_slot_to_pyobj(recipe.concrete_r[k])))
        .collect();
    sym.concrete_stack = (nlocals..valuestackdepth)
        .map(|k| concrete_value_from_slot(recipe_slot_to_pyobj(recipe.concrete_r[k])))
        .collect();
    sym.concrete_namespace = w_globals;
    sym.concrete_execution_context = execution_context;
    // perform_call threads the caller's `ec` down to every inlined callee
    // (reds=['frame','ec'], interp_jit.py:67), so the callee shares the
    // caller's ExecutionContext OpRef. Seed `sym.execution_context` from the
    // immediate parent's symbolic ec (the root caller's `setup_bridge_sym`
    // resolved it from `portal_ec_reg`, state.rs:7399) so an in-callee
    // `ensure_execution_context` returns it directly. `sym.frame` is NONE for
    // a reconstructed inline frame, so without this seed the recovery path
    // would emit `GetfieldGcR(NONE)` (an unbacked VoidOp arg) and fail
    // bridge regalloc.
    if let Some(parent) = parent_frames.first() {
        let parent_ec = unsafe { (*parent.sym).execution_context };
        if !parent_ec.is_none() {
            sym.execution_context = parent_ec;
        }
    }
    // pyjitpl.py:74-90 MIFrame.setup: size the per-kind banks + copy_constants.
    // The constant tail lands at `[num_regs_X..]`, beyond the live
    // valuestackdepth prefix (`num_regs_r` is the full Ref register file),
    // so the live slots set above are preserved. `locals_cells_stack_array_
    // ref` stays NONE: dead (NONE) slots are never read — the liveness stream
    // only carries live values, so no heap fall-through is needed (and
    // `s.frame` is NONE, so emitting the array getfield would be invalid).
    sym.setup_kind_register_banks(ctx);

    PendingInlineFrame {
        sym,
        concrete_frame,
        // No virtual_ref for a reconstructed frame: the forward trace's
        // virtual_ref was already finished/encoded; None skips the
        // opimpl_virtual_ref emission in push_inline_frame.
        drop_frame_opref: None,
        // The reconstructed frame represents the same inlined call the
        // forward trace pushed at function entry; match its (code, 0)
        // greenkey identity for recursion-depth + inline-position tracking.
        green_key: crate::driver::make_green_key(w_code, 0),
        green_key_raw: (w_code as usize, 0),
        parent_frames,
        nargs: recipe.nargs,
        caller_result_stack_idx: None,
        caller_result_type: Some(Type::Ref),
        // Reconstructed frames carry no CALL-site OpRefs; the inline
        // back-edge CALL_ASSEMBLER path requires drop_frame_opref and
        // is gated out for them anyway.
        replay_callable: OpRef::NONE,
        replay_args: Vec::new(),
    }
}

/// Build the per-frame setup for ONE reconstructed bridge-carrier callee so
/// the full-body walker can drive it (issue #215 item 2, P2 drain):
///
///   - emit the frame vable seeded with the recipe's LOCALS (slots
///     `0..nlocals`); the live operand-stack temps stay in the abstract
///     register file, seeded into `argboxes_r` below.  `valuestackdepth`
///     passed to the vable builder is `nlocals`, matching the forward-inline
///     callee (the stack is rebuilt in registers as the body runs);
///   - assemble the symbolic + concrete frame
///     (`assemble_bridge_inline_pending`) and bind `sym.frame` to the emitted
///     vable so the portal body reads its locals through `getarrayitem_vable`
///     (a NONE `sym.frame` would take the nonstandard-vable Void leg);
///   - seed the walk's initial register file: frame/ec at their portal-red
///     colors + the live stack temps at `[nlocals..valuestackdepth)` (the same
///     semantic-prefix convention the root bridge seeding uses, trace.rs).
///
/// Returns `(pending, argboxes_r)`; `None` when the callee body/layout is
/// unavailable.  Records the vable ops into `ctx` in trace order, so the
/// caller must invoke this at the point the callee frame is reconstructed.
pub(crate) fn setup_reconstructed_callee_frame(
    ctx: &mut TraceCtx,
    recipe: &ReconstructRecipe,
    execution_context: *const pyre_interpreter::PyExecutionContext,
    parent_frames: Vec<ResumeFrameState>,
) -> Option<(PendingInlineFrame, Vec<OpRef>)> {
    let raw_code = recipe.code_ptr as *const pyre_interpreter::CodeObject;
    if raw_code.is_null() {
        return None;
    }
    let code_ref = unsafe { &*raw_code };
    let (_nlocals_plus_cells, frame_array_size) = callee_layout_for_call_assembler(code_ref);
    let nlocals = recipe.nlocals;
    let valuestackdepth = recipe.valuestackdepth;
    if recipe.registers_r.len() < valuestackdepth || nlocals > valuestackdepth {
        return None;
    }

    let (frame_reg, ec_reg) = portal_red_regs_at(recipe.jitcode_index);
    if frame_reg == u16::MAX || ec_reg == u16::MAX {
        return None;
    }

    let w_code = pyre_interpreter::live_code_wrapper(recipe.code_ptr) as *const ();
    let w_globals = recover_inline_callee_globals(recipe.code_ptr);
    let pycode_const = ctx.const_ref(w_code as i64);
    let w_globals_const = ctx.const_ref(w_globals as i64);
    let ec_const = ctx.const_ref(execution_context as i64);

    let locals_boxes: Vec<OpRef> = recipe.registers_r[..nlocals].to_vec();
    let frame_vable = crate::helpers::emit_new_pyframe_inline_with_params(
        ctx,
        &locals_boxes,
        frame_array_size,
        nlocals,
        pycode_const,
        w_globals_const,
        ec_const,
    );

    let mut pending = assemble_bridge_inline_pending(ctx, recipe, execution_context, parent_frames);
    pending.sym.frame = frame_vable;
    // The portal reds are [frame, ec], force-alive at every pc; a guard snapshot
    // (`collect_outer_active_boxes`) reads ec at portal_ec_reg via
    // `sym.execution_context`.  `assemble_bridge_inline_pending` only seeds it
    // from `parent_frames.first()`, so seed it directly here for the
    // walker-driven callee (an unset ec surfaces as a liveness-active NONE
    // panic at the first in-callee guard).
    if pending.sym.execution_context.is_none() {
        pending.sym.execution_context = ec_const;
    }

    let max_reg = valuestackdepth
        .max(frame_reg as usize + 1)
        .max(ec_reg as usize + 1);
    let mut argboxes_r: Vec<OpRef> = vec![OpRef::NONE; max_reg];
    argboxes_r[frame_reg as usize] = frame_vable;
    argboxes_r[ec_reg as usize] = ec_const;
    // The recipe's `registers_r` is SEMANTIC-slot-indexed (filled from the
    // frame vable's `locals_cells_stack_w` array), but the re-executed callee
    // reads its registers by post-rename COLOR. After stack-slot-pinning
    // removal a live operand-stack slot's color is per-program-point
    // (`pcdep_color_slots`) and need not equal `nlocals + d` — the encoder
    // (`get_list_of_active_boxes`) maps color→slot via
    // `semantic_ref_slot_for_reg_color`, so invert it here to land each stack
    // value at the color the dispatcher will touch. Placing it at the raw slot
    // index (the old identity assumption) lands it under the wrong register and
    // leaves the true color still holding the portal-red seed (frame/ec) whose
    // color the register allocator reused for this stack slot. Runs AFTER the
    // frame/ec seeding so a reused color resolves to the live stack value.
    // Falls back to identity when no live color owns the slot (empty map /
    // non-diverging coloring).
    let py_pc = backxlat_py_pc(recipe.jitcode_index, recipe.jitcode_pc);
    let pcdep = pcdep_color_slots_at(recipe.jitcode_index, py_pc);
    for k in nlocals..valuestackdepth {
        let opref = recipe.registers_r[k];
        if opref.is_none() {
            continue;
        }
        let color = semantic_slot_color_for_ref_slot(&pcdep, k).unwrap_or(k);
        if color >= argboxes_r.len() {
            argboxes_r.resize(color + 1, OpRef::NONE);
        }
        argboxes_r[color] = opref;
        // `box.value` parity at the frame boundary (mirror `ref_return/r`):
        // thread the reconstructed slot's concrete onto the OpRef-keyed shadow
        // so the re-executed callee's speculation gates
        // (`walker_int_specialization_operands`) see the live value's class. The
        // emitted specialization stays runtime-correct (guard_class + int_op on
        // the unboxed register); the concrete is only the tracing shadow, as for
        // any resumed operand. Skips constants (`constants.get_value` is
        // authoritative) and non-value (`Void`) slots.
        if !opref.is_constant() {
            if let Some(v @ (majit_ir::Value::Ref(_) | majit_ir::Value::Int(_))) =
                recipe.concrete_r.get(k).copied()
            {
                ctx.set_opref_concrete(opref, v);
            }
        }
    }

    Some((pending, argboxes_r))
}

pub enum InlineTraceStepAction {
    Trace(TraceAction),
    PushFrame(PendingInlineFrame),
}

pub fn execute_inline_residual_call(
    frame: &mut pyre_interpreter::pyframe::PyFrame,
    nargs: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let required = nargs + 2; // callable + null/self + args
    if frame.valuestackdepth < frame.stack_base() + required {
        return Err(pyre_interpreter::PyError::type_error(
            "inline residual call stack underflow",
        ));
    }

    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();
    let _null_or_self = frame.pop();
    let callable = frame.pop();
    let result = pyre_interpreter::call::call_callable_inline_residual(frame, callable, &args)?;
    frame.push(result);
    Ok(())
}

// inline_trace_and_execute / trace_through_callee removed — the trait
// meta-interpreter that replaced them (PyreMetaInterp.interpret() +
// push_inline_frame) is itself retired (#203 gap 10); the FBW walker
// handles both root and inline frames.

/// `pypy/objspace/std/listobject.py:2390 is_plain_int1` parity.
///
/// IntegerListStrategy stores raw i64 and so cannot preserve the
/// W_*Object pointer identity of its elements. PyPy therefore demotes
/// to object strategy on insertion of any value whose exact type is
/// not `W_IntObject` — bools, int subclasses (PyPy `W_IntObjectUser`
/// from `interpreter/typedef.py:205 subcls`), and `W_LongObject` whose
/// value doesn't fit in a machine int.
///
/// Delegates to the single `pyre_object::is_plain_int1` helper that
/// already implements the full upstream predicate including the
/// `w_class != get_instantiate(&INT_TYPE)` int-subclass rejection
/// (`listobject.rs:235`); a previous in-place `py_type_check` shortcut
/// at this site was a deviation that mishandled int subclasses
/// because pyre stores them with `ob_type == &INT_TYPE` and only
/// distinguishes them via `w_class` (`typedef.rs:686 w_int_new_unique`).
pub unsafe fn int_strategy_preserves_identity(value: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_object::is_plain_int1(value) }
}

/// Test-only RAII guard: swap a `MetaInterpStaticData` into the
/// thread-local `METAINTERP_SD` and restore the previous value on drop,
/// so tests sharing a thread (`--test-threads=1`) do not observe each
/// other's registrations.
#[cfg(test)]
pub(crate) struct MetainterpSdGuard {
    prev: Option<MetaInterpStaticData>,
}

#[cfg(test)]
impl MetainterpSdGuard {
    pub(crate) fn swap(sd: MetaInterpStaticData) -> Self {
        let prev = METAINTERP_SD.with(|slot| std::mem::replace(&mut *slot.borrow_mut(), sd));
        Self { prev: Some(prev) }
    }
}

#[cfg(test)]
impl Drop for MetainterpSdGuard {
    fn drop(&mut self) {
        if let Some(prev) = self.prev.take() {
            METAINTERP_SD.with(|slot| *slot.borrow_mut() = prev);
        }
    }
}

#[cfg(test)]
mod indirectcalltargets_tests {
    //! Line-by-line parity tests for `pyjitpl.py:2248-2249` and
    //! `pyjitpl.py:2326-2343`.  Tests exercise the local
    //! `MetaInterpStaticData` methods directly — independent of the
    //! thread-local `METAINTERP_SD` singleton so concurrent callers
    //! (and unrelated tests that use the thread-local) do not alias.
    use super::{
        MetaInterpStaticData, MetainterpSdGuard, pyjitcode_for_jitcode_index,
        raw_code_for_jitcode_index,
    };
    use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};
    use pyre_interpreter::bytecode::CodeObject;
    use std::sync::Arc;

    fn make_jitcode_with_fnaddr(fnaddr: usize) -> Arc<JitCode> {
        let mut jc = JitCodeBuilder::default().finish();
        jc.fnaddr = fnaddr as i64;
        Arc::new(jc)
    }

    /// Freshly-constructed staticdata has no indirect-call targets.
    /// Delegates through `canonical`; matches the behaviour of
    /// `majit_metainterp::MetaInterpStaticData::new()` where the
    /// `MetaInterpGlobalData` inside starts with every lazy cache
    /// (`indirectcall_dict`, `addr2name`) set to `None`.
    #[test]
    fn bytecode_for_address_none_when_empty() {
        let mut sd = MetaInterpStaticData::new();
        assert!(sd.bytecode_for_address(0xdeadbeef).is_none());
    }

    /// `pyjitpl.py:2326-2343` hit path: registered fnaddrs resolve to
    /// their JitCode.
    #[test]
    fn bytecode_for_address_returns_jitcode_when_registered() {
        let mut sd = MetaInterpStaticData::new();
        let j100 = make_jitcode_with_fnaddr(0x100);
        let j200 = make_jitcode_with_fnaddr(0x200);
        let j300 = make_jitcode_with_fnaddr(0x300);
        sd.setup_indirectcalltargets(vec![j100.clone(), j200.clone(), j300.clone()]);

        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x100).unwrap(), &j100));
        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x200).unwrap(), &j200));
        assert!(Arc::ptr_eq(&sd.bytecode_for_address(0x300).unwrap(), &j300));
        assert!(sd.bytecode_for_address(0x400).is_none());
    }

    /// `pyjitpl.py:2248-2249` `setup_indirectcalltargets` parity:
    /// every call replaces the targets list and invalidates the lazy
    /// dict so the next lookup rebuilds from the new list.
    #[test]
    fn setup_indirectcalltargets_invalidates_cache() {
        let mut sd = MetaInterpStaticData::new();
        sd.setup_indirectcalltargets(vec![make_jitcode_with_fnaddr(0x100)]);
        assert!(sd.bytecode_for_address(0x100).is_some());
        sd.setup_indirectcalltargets(vec![
            make_jitcode_with_fnaddr(0x200),
            make_jitcode_with_fnaddr(0x300),
        ]);
        assert!(sd.bytecode_for_address(0x100).is_none());
        assert!(sd.bytecode_for_address(0x200).is_some());
        assert!(sd.bytecode_for_address(0x300).is_some());
    }

    fn make_code(source: &str) -> (*const (), *const CodeObject) {
        let raw_code = pyre_interpreter::compile_exec(source).expect("source must compile");
        let code = pyre_interpreter::w_code_new(Box::into_raw(Box::new(raw_code)) as *const ())
            as *const ();
        let raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject
        };
        (code, raw_code)
    }

    fn populated_pyjit(raw_code: *const CodeObject) -> Arc<crate::PyJitCode> {
        let mut pyjit = crate::PyJitCode::skeleton(raw_code);
        // A drained PerCodeObject install has non-empty `code` and
        // `is_drained` set; give the fixture both so it is not classified
        // as a skeleton (`is_skeleton()` now tests `code.is_empty()`).
        let runtime_jc = majit_metainterp::jitcode::JitCode::new("populated_pyjit_test");
        runtime_jc.set_body(majit_translate::jitcode::JitCodeBody {
            code: vec![majit_metainterp::jitcode::insns::BC_LIVE, 0, 0],
            startpoints: Some([0_usize].into_iter().collect()),
            ..Default::default()
        });
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.is_drained = true;
        Arc::new(pyjit)
    }

    #[should_panic(expected = "make_jitcodes returned an unpopulated JitCode skeleton")]
    #[test]
    fn install_jitcodes_rejects_skeleton_payload() {
        let mut sd = MetaInterpStaticData::new();
        let (_code, raw_code) = make_code("x = 1\n");
        let skeleton = Arc::new(crate::PyJitCode::skeleton(raw_code));
        sd.set_jitcodes_from_make_result(vec![skeleton]);
    }

    #[test]
    fn compiled_jitcode_lookup_returns_populated_entry() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("populated payload should be installed by make_jitcodes");
        assert!(std::ptr::eq(sd.jitcodes[0].as_ref(), hit));
    }

    #[test]
    fn compiled_jitcode_lookup_scans_by_raw_code_identity() {
        let mut sd = MetaInterpStaticData::new();
        let (code, raw_code) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(raw_code)]);

        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("wrapper pointer should resolve through raw CodeObject identity");
        assert_eq!(unsafe { (*hit).raw_code() }, raw_code);
    }

    #[test]
    fn raw_code_for_jitcode_index_returns_canonical_graph_pointer() {
        let mut sd = MetaInterpStaticData::new();
        let (_code, expected_raw) = make_code("x = 1\n");
        sd.set_jitcodes_from_make_result(vec![populated_pyjit(expected_raw)]);
        let _sd_guard = MetainterpSdGuard::swap(sd);

        let hit = raw_code_for_jitcode_index(0).expect("jitcode index 0 must resolve");
        assert_eq!(hit, expected_raw);
    }
}

#[derive(Clone, Copy)]
pub struct ResumeFrameState {
    pub sym: *mut PyreSym,
    pub concrete_frame_addr: usize,
    pub resume_pc: usize,
    /// Jitcode-space twin of the pc word this frame will encode (gh#369).
    pub resume_marker_jit_pc: Option<usize>,
    /// Python pc of the CALL whose residual call this frame is the caller
    /// of (the call still on this frame's stack when the callee was
    /// inlined).  When that call sits in a try-block the jitcode emits a
    /// post-call `-live-`/`catch_exception` keyed by this pc
    /// (the post-call catch-marker twin); on a guard that deopts mid-callee
    /// the blackhole must resume this frame AT that catch
    /// (`blackhole.py:396-410 handle_exception_in_frame`,
    /// `pyjitpl.py:2601-2602`).  `None` for frames whose call has no catch
    /// marker.
    pub call_pc: Option<usize>,
    /// pyjitpl.py:181-193 `get_list_of_active_boxes(in_a_call=True)`.
    /// Non-top frames clear the caller's pending result slot before
    /// snapshotting liveness so the undefined call result does not leak
    /// stale boxes into guard fail_args.
    pub pending_result_stack_idx: Option<usize>,
    pub pending_result_type: Option<Type>,
}

#[cfg(test)]
mod finish_setup_tests {
    use super::{MetaInterpStaticData, MetainterpSdGuard, blackhole_control_opcodes};
    use crate::assembler::publish_state;

    #[test]
    fn finish_setup_refreshes_opcode_cache_after_initial_empty_snapshot() {
        let mut sd = MetaInterpStaticData::new();
        let empty: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        sd.finish_setup_if_needed(&empty, Vec::new());
        assert_eq!(sd.op_live, u8::MAX);
        assert_eq!(sd.op_goto, u8::MAX);

        let mut insns = indexmap::IndexMap::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("goto/L".to_string(), 16u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        insns.insert("int_return/i".to_string(), 148u8);
        insns.insert("ref_return/r".to_string(), 76u8);
        insns.insert("float_return/f".to_string(), 149u8);
        insns.insert("void_return/".to_string(), 150u8);

        sd.finish_setup_if_needed(&insns, vec![1, 2, 3]);
        assert_eq!(sd.op_live, 88);
        assert_eq!(sd.op_goto, 16);
        assert_eq!(sd.op_catch_exception, 89);
        assert_eq!(sd.op_rvmprof_code, 91);
        assert_eq!(sd.op_int_return, 148);
        assert_eq!(sd.op_ref_return, 76);
        assert_eq!(sd.op_float_return, 149);
        assert_eq!(sd.op_void_return, 150);
        assert_eq!(&*sd.liveness_info, &[1u8, 2, 3][..]);
        assert!(sd.finish_setup_done);
    }

    #[test]
    fn blackhole_control_opcodes_reflect_finish_setup_cache() {
        let mut sd = MetaInterpStaticData::new();
        let mut insns = indexmap::IndexMap::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        sd.finish_setup_if_needed(&insns, Vec::new());
        let _sd_guard = MetainterpSdGuard::swap(sd);

        assert_eq!(blackhole_control_opcodes(), (88, 89, 91));
    }

    #[test]
    fn blackhole_control_opcodes_refresh_after_initial_empty_snapshot() {
        let mut sd = MetaInterpStaticData::new();
        let empty: indexmap::IndexMap<String, u8> = indexmap::IndexMap::new();
        sd.finish_setup_if_needed(&empty, Vec::new());
        let _sd_guard = MetainterpSdGuard::swap(sd);

        let mut insns = indexmap::IndexMap::new();
        insns.insert("live/".to_string(), 88u8);
        insns.insert("catch_exception/L".to_string(), 89u8);
        insns.insert("rvmprof_code/ii".to_string(), 91u8);
        publish_state(&insns, &[], 0, 0);

        assert_eq!(blackhole_control_opcodes(), (88, 89, 91));
    }
}
