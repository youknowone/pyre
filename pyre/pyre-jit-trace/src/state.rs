//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use majit_backend::Backend;
use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_metainterp::virtualizable::VirtualizableInfo;
use majit_metainterp::{
    JitDriverStaticData, JitState, ResidualVirtualizableSync, TraceAction, TraceCtx,
};
use pyre_interpreter::PyError;

use pyre_interpreter::bytecode::{CodeObject, ComparisonOperator, Instruction};
use pyre_interpreter::pyframe::{PendingInlineResult, PyFrame};
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::pyobject::{FLOAT_TYPE, INT_TYPE, LIST_TYPE, is_bool, is_float, is_int};
use pyre_object::{PY_NULL, w_float_get_value, w_int_get_value, w_int_new, w_list_new};
use std::collections::HashMap;

/// jitcode.py:9-21 / codewriter.py:68: JitCode — compiled bytecode unit.
///
/// MetaInterpStaticData-side wrapper around the shared
/// [`crate::PyJitCode`] payload. The payload `Arc` is the same heap
/// allocation that `pyre_jit::jit::call::CallControl.jitcodes` holds
/// for the same CodeObject — RPython's `MetaInterpStaticData.jitcodes`
/// list and `CallControl.jitcodes` dict reference identical
/// `JitCode` Python objects through Python's refcount semantics, and
/// pyre mirrors that with a shared `Arc`. The wrapper keeps `code`
/// (so `code_for_jitcode_index` can recover the wrapping
/// `PyObjectRef` from a numeric index) and `index` (the SD-local
/// `jitcode.index = len(all_jitcodes)` from codewriter.py:68) on the
/// SD side, since neither is intrinsic to the `PyJitCode` payload.
// SAFETY: JitCode is only written once (during creation) and then
// read-only. The code pointer is stable for the program lifetime.
unsafe impl Sync for JitCode {}

pub(crate) struct JitCode {
    /// Pointer to the Code object (W_CodeObject).
    /// Matches frame.code and getcode(func).
    pub code: *const (),
    /// codewriter.py:68: jitcode.index = len(all_jitcodes).
    pub index: i32,
    /// Shared `PyJitCode` payload. Same `Arc` instance also lives in
    /// `CallControl.jitcodes`. Refinements (e.g. a `merge_point_pc`
    /// rebuild) replace this `Arc` in place so cached `*const JitCode`
    /// pointers see the refreshed payload on the next field access.
    pub payload: std::sync::Arc<crate::PyJitCode>,
}

impl JitCode {
    /// Extract raw CodeObject from the W_CodeObject stored in this JitCode.
    #[inline]
    pub unsafe fn raw_code(&self) -> *const CodeObject {
        unsafe {
            if self.code.is_null() {
                return std::ptr::null();
            }
            pyre_interpreter::w_code_get_ptr(self.code as pyre_object::PyObjectRef)
                as *const CodeObject
        }
    }
}

/// warmspot.py:148-282: MetaInterpStaticData — per-driver compile-time data.
///
/// RPython: created by WarmRunnerDesc, holds jitcodes list populated
/// by codewriter.make_jitcodes(). Accessed as MetaInterp.staticdata.
///
/// pyre: per-thread equivalent (no-GIL runtime). Indices assigned
/// lazily at trace-time (no codewriter phase). Portal gets 0,
/// inline callees get 1, 2, ….
///
/// PRE-EXISTING-ADAPTATION: the RPython-orthodox
/// `MetaInterpStaticData` lives in
/// `majit_metainterp::pyjitpl::MetaInterpStaticData`.  pyre embeds it
/// as the `canonical` field below and delegates every RPython method
/// (`setup_indirectcalltargets`, `bytecode_for_address`, …) through
/// it, so there is exactly one port of each of those methods.  The
/// pyre-local fields (`by_code`, `jitcodes`, `finish_setup_done`,
/// `op_*`, `liveness_info`) stay here because pyre's lazy-compilation
/// model has no RPython equivalent for them; they are a minimal pyre
/// adaptation layer on top of the canonical staticdata.
struct MetaInterpStaticData {
    /// codewriter.py:80: CodeObject* → index into jitcodes vec.
    by_code: std::collections::HashMap<usize, usize>,
    /// warmspot.py:282: self.metainterp_sd.jitcodes = jitcodes.
    /// Box<JitCode> for address stability across vec growth.
    jitcodes: Vec<Box<JitCode>>,
    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// frozen snapshot of the assembler's `all_liveness` buffer. In RPython
    /// this is set once at `finish_setup` time; in pyre the assembler is
    /// long-lived and liveness accumulates across lazy JitCode compiles,
    /// so this field is resynced after every `intern_liveness` write.
    liveness_info: Vec<u8>,
    /// pyjitpl.py:2255 `finish_setup` is per MetaInterpStaticData instance.
    /// `METAINTERP_SD` is thread-local in pyre, so this guard must live on
    /// the thread-local object rather than in a process-global `Once`.
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

impl MetaInterpStaticData {
    fn new() -> Self {
        Self {
            by_code: std::collections::HashMap::new(),
            jitcodes: Vec::new(),
            liveness_info: Vec::new(),
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
    fn setup_insns(&mut self, insns: &HashMap<&'static str, u8>) {
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
    fn finish_setup_if_needed(&mut self, insns: &HashMap<&'static str, u8>, all_liveness: Vec<u8>) {
        if self.finish_setup_done {
            return;
        }
        self.setup_insns(insns);
        self.liveness_info = all_liveness;
        self.finish_setup_done = true;
    }

    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// refreshes the staticdata mirror after each writer-side append.
    pub(crate) fn set_liveness_info(&mut self, bytes: Vec<u8>) {
        self.liveness_info = bytes;
    }
}

/// Crate-local helper so sibling modules can refresh
/// `MetaInterpStaticData.liveness_info` without leaking the private
/// type through public API signatures.
pub(crate) fn publish_liveness_info(bytes: Vec<u8>) {
    METAINTERP_SD.with(|r| r.borrow_mut().set_liveness_info(bytes));
}

#[allow(dead_code)]
impl MetaInterpStaticData {
    /// codewriter.py:68: get or create JitCode for a CodeObject.
    /// Returns a stable pointer (Box ensures no reallocation moves).
    ///
    /// `supplied` carries the shared `Arc<PyJitCode>` from the
    /// upstream codewriter (see [`COMPILE_JITCODE_FN`]); when present
    /// we install it as the entry's payload so both stores reference
    /// the same allocation. On a `by_code` hit a fresh `supplied`
    /// replaces the existing payload — this is how a
    /// `merge_point_pc`-driven recompile in `CallControl` propagates
    /// to readers that hold a cached `*const JitCode`. When
    /// `supplied` is `None` (no callback registered, or callback
    /// declined), we fall back to a skeleton so the entry slot
    /// exists with a well-formed `Arc`.
    fn jitcode_for(
        &mut self,
        code: *const (),
        supplied: Option<std::sync::Arc<crate::PyJitCode>>,
    ) -> *const JitCode {
        let key = code as usize;
        if let Some(&idx) = self.by_code.get(&key) {
            if let Some(arc) = supplied {
                // codewriter.py:67-68 parity — the SD-local position is
                // invariant across `merge_point_pc`-driven recompiles of
                // the same CodeObject; the new payload must carry the
                // same `jitcode.index` as the old one so any
                // outstanding `frame.jitcode.index` read still maps to
                // the same `metainterp_sd.jitcodes[idx]` slot.  Stamp
                // BEFORE installing so the replacement is always
                // consistent even if a reader races on `self.jitcodes[
                // idx].payload`.
                Self::stamp_payload_index(idx as i32, &arc);
                self.jitcodes[idx].payload = arc;
            }
            return &*self.jitcodes[idx] as *const JitCode;
        }
        let payload =
            supplied.unwrap_or_else(|| std::sync::Arc::new(crate::PyJitCode::skeleton(None)));
        let index = self.jitcodes.len() as i32;
        if std::env::var_os("MAJIT_LOG").is_some() && index > 30 {
            eprintln!("[jitcode-for] new idx={} code={:#x}", index, key);
        }
        // codewriter.py:68 `jitcode.index = index` — back-stamp the
        // canonical `metainterp_sd.jitcodes` position onto the inner
        // `Arc<majit_metainterp::jitcode::JitCode>` so a snapshot whose
        // `frame.jitcode` points to this allocation encodes the
        // correct `metainterp_sd.jitcodes[index]` lookup target on the
        // resume side.  Same helper serves the hit path above — both
        // paths must keep the shared `JitCode` object's `index` field
        // aligned with the SD slot.
        Self::stamp_payload_index(index, &payload);
        let jitcode = Box::new(JitCode {
            code,
            index,
            payload,
        });
        let ptr = &*jitcode as *const JitCode;
        self.by_code.insert(key, self.jitcodes.len());
        self.jitcodes.push(jitcode);
        ptr
    }

    /// codewriter.py:67-68 parity — stamp the SD-local `idx` onto the
    /// shared `Arc<majit_metainterp::jitcode::JitCode>` carried by the
    /// payload.  Shared helper used by both the by_code miss path
    /// (fresh SD slot) and the by_code hit path (payload swap driven
    /// by `merge_point_pc` refinement in CallControl).  The atomic
    /// store works regardless of the outer Arc<PyJitCode> refcount;
    /// without interior mutability the back-stamp would require
    /// Arc::get_mut which fails as soon as CallControl.jitcodes also
    /// holds the same allocation.
    fn stamp_payload_index(idx: i32, payload: &std::sync::Arc<crate::PyJitCode>) {
        payload
            .jitcode
            .index
            .store(idx as i64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Return the existing SD entry only when its payload is already
    /// populated. Skeleton entries must still go through the compile
    /// callback so `CallControl` can drain and fill the bytecode body.
    ///
    /// PRE-EXISTING-ADAPTATION: RPython's `MetaInterpStaticData` never
    /// observes a skeleton entry — every `self.jitcodes` slot is
    /// populated by `make_jitcodes` (codewriter.py:74-89) before any
    /// trace runs. Pyre compiles lazily via `set_compile_jitcode_fn`,
    /// so a cached entry here can still be the shell produced by
    /// `JitCode(name, fnaddr, calldescr, ...)` (call.py:168) before
    /// `assembler.assemble(ssarepr, jitcode, num_regs)`
    /// (codewriter.py:67) has run.
    fn compiled_jitcode_lookup(&self, code: *const ()) -> Option<*const JitCode> {
        let idx = *self.by_code.get(&(code as usize))?;
        let jitcode = &self.jitcodes[idx];
        if !jitcode.payload.is_populated() {
            return None;
        }
        Some(&**jitcode as *const JitCode)
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
    METAINTERP_SD.with(|r| r.borrow_mut().liveness_info = all_liveness);
    Some(pos)
}

/// RPython resume.py:1022 parity: read-only snapshot of
/// `metainterp_sd.liveness_info` for the ResumeDataDirectReader. Returns a
/// Vec copy because thread-local borrows cannot escape the closure.
pub fn liveness_info_snapshot() -> Vec<u8> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().liveness_info.clone())
}

/// pyjitpl.py:2236 parity: expose the staticdata `live/` opcode for callers
/// that need to decode inline liveness offsets from a JitCode.
pub fn op_live() -> u8 {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| r.borrow().op_live)
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
/// consumer will route through here once Step 3 lands.
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

/// Process-global hook the upper `pyre-jit` crate registers from
/// `CodeWriter::new()` so the trace-side `jitcode_for` can drive the
/// codewriter pipeline without `pyre_jit_trace` taking a build
/// dependency on `pyre-jit`. RPython parity: jtransform's lookup of a
/// callee graph in `cc.callcontrol.get_jitcode(callee)` (call.py:155)
/// directly populates `CallControl.jitcodes` and queues the entry on
/// `unfinished_graphs`; `make_jitcodes`'s drain (codewriter.py:79-89)
/// then calls `transform_graph_to_jitcode` and `assembler.finished()`
/// in one transitive pass. Pyre cannot perform that pass eagerly at
/// warmspot time (no static callee discovery), so the lazy analog
/// fires on every trace-side `jitcode_for` instead.
///
/// The callback returns the shared `Arc<PyJitCode>` it just installed
/// in `CallControl.jitcodes`; `jitcode_for` then stores the same
/// `Arc` on the SD entry so both stores reference one allocation —
/// the Rust analog of RPython's two stores holding the same Python
/// object. The fn pointer stays `Sync` because the upstream callback
/// (`compile_jitcode_via_w_code`) reads `CodeWriter::instance()`
/// which is itself per-thread.
pub type CompileJitcodeFn = fn(*const ()) -> Option<std::sync::Arc<crate::PyJitCode>>;

static COMPILE_JITCODE_FN: std::sync::atomic::AtomicPtr<()> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Registered once per process from `CodeWriter::new()` (under a
/// `Once` guard). Invoked by `jitcode_for` before the canonical
/// `MetaInterpStaticData.jitcodes` insert so that
/// `CallControl.jitcodes` is populated under the same RPython
/// invariant: any code object whose `JitCode` index appears in
/// `MetaInterpStaticData` also has a populated `PyJitCode` reachable
/// through `CallControl.find_jitcode` for the blackhole/resume
/// lookup (resume.py:1338 `metainterp_sd.jitcodes[jitcode_pos]`).
pub fn set_compile_jitcode_fn(f: CompileJitcodeFn) {
    COMPILE_JITCODE_FN.store(f as *mut (), std::sync::atomic::Ordering::Relaxed);
}

/// pyjitpl.py:74: frame.jitcode — get or create JitCode for CodeObject.
/// RPython: MetaInterp.staticdata.jitcodes[idx]; pyre: METAINTERP_SD.
///
/// Fires the registered compile callback before installing the SD
/// entry; the callback returns the shared `Arc<PyJitCode>` from the
/// upstream `CallControl.jitcodes` so both stores reference one
/// allocation. RPython's equivalent is `cc.callcontrol.get_jitcode(graph)`
/// inside jtransform, which inserts the JitCode object into the
/// `CallControl.jitcodes` dict and lets the surrounding `make_jitcodes`
/// drain populate `MetaInterpStaticData.jitcodes` — both holding the
/// same Python object. Pyre's lazy split runs that pipeline for one
/// code object per trace-side reference; see `set_compile_jitcode_fn`.
pub(crate) fn jitcode_for(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    if let Some(existing) = METAINTERP_SD.with(|r| r.borrow().compiled_jitcode_lookup(code)) {
        return existing;
    }
    let cb = COMPILE_JITCODE_FN.load(std::sync::atomic::Ordering::Relaxed);
    let supplied = if cb.is_null() {
        None
    } else {
        // SAFETY: only `set_compile_jitcode_fn` writes to this slot,
        // and it stores a `CompileJitcodeFn` cast through `as *mut ()`.
        let f: CompileJitcodeFn = unsafe { std::mem::transmute(cb) };
        f(code)
    };
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code, supplied))
}

/// Read-only JitCode lookup by CodeObject-wrapper pointer.
///
/// Blackhole/resume paths must not invoke the compile callback: any
/// nested `jitcode_for` call re-enters the JIT pipeline (compile,
/// warmstate bookkeeping) from inside a guard-failure decoder, which
/// caused SIGSEGVs on nbody. Returns null if the code has never been
/// registered — callers decide whether to fall back to the LiveVars
/// path.
///
/// Uses the same key as `MetaInterpStaticData::jitcode_for`
/// (`code as usize` — the tracer-side `jitcode_for(w_code)` call
/// sites at trace_opcode.rs:3501 / :3580 / state.rs:1908 all pass
/// the w_code wrapper pointer).
pub(crate) fn jitcode_lookup(code: *const ()) -> *const JitCode {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let key = code as usize;
        match sd.by_code.get(&key) {
            Some(&idx) => &*sd.jitcodes[idx] as *const JitCode,
            None => std::ptr::null(),
        }
    })
}

/// warmspot.py:282 metainterp_sd.jitcodes[jitcode_index]:
/// Resolve jitcode_index (sequential int from snapshot numbering)
/// to the corresponding CodeObject pointer.
pub fn code_for_jitcode_index(jitcode_index: i32) -> Option<*const ()> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes.get(idx).map(|jc| jc.code)
    })
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
        if let Some(&jit_pc) = payload.metadata.pc_map.get(pc as usize) {
            let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
            let all_liveness = &sd.liveness_info;
            if off + 2 < all_liveness.len() {
                let length_i = all_liveness[off] as usize;
                let length_r = all_liveness[off + 1] as usize;
                let length_f = all_liveness[off + 2] as usize;
                return length_i + length_r + length_f;
            }
        }
        // Skeleton / uninit-jitcode fallback. Production runs go
        // through the `jitcode_for` drain which fills pc_map +
        // liveness before any guard capture (audit 2026-04-20: 0 hits
        // across pyre/check.sh), but unit-test entry points
        // (`PyreSym::new_uninit`) legitimately land here. Mirror
        // trace_opcode.rs:get_list_of_active_boxes: iterate locals +
        // stack filtering by is_local_live / is_stack_live so the
        // decoded value count agrees with the encoder's BC_LIVE box
        // count.
        if !jc.code.is_null() {
            let raw = unsafe { jc.raw_code() };
            let live = crate::liveness::liveness_for(raw);
            let code_ref = unsafe { &*raw };
            let nlocals = code_ref.varnames.len();
            let live_locals = (0..nlocals)
                .filter(|&i| live.is_local_live(pc as usize, i))
                .count();
            let stack_depth = live.stack_depth_at(pc as usize);
            let live_stack = (0..stack_depth)
                .filter(|&i| live.is_stack_live(pc as usize, i))
                .count();
            return live_locals + live_stack;
        }
        0
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
/// * `scalar_oprefs` — the NUM_SCALAR_INPUTARGS − 1 static field OpRefs
///   in declaration order (last_instr, pycode, valuestackdepth,
///   debugdata, lastblock, w_globals).
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
    let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    let expected_scalars = num_scalars - 1;
    assert_eq!(
        scalar_oprefs.len(),
        expected_scalars,
        "seed_virtualizable_boxes: scalar_oprefs.len() must equal NUM_SCALAR_INPUTARGS - 1",
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

/// resume.py:1054 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
/// Returns the register indices for live values at (jitcode_index, pc) in
/// the same order the encoder wrote them — [int bank..., ref bank..., float
/// bank...]. Each entry is the frame register/slot number that the decoded
/// frames[0].values[j] should be written to, matching RPython's per-bank
/// callback walk.
pub fn frame_liveness_reg_indices_at(jitcode_index: i32, pc: i32) -> Vec<u32> {
    ensure_finish_setup();
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        let Some(jc) = sd.jitcodes.get(idx) else {
            return Vec::new();
        };
        let payload = &jc.payload;
        let Some(&jit_pc) = payload.metadata.pc_map.get(pc as usize) else {
            return Vec::new();
        };
        let off = payload.jitcode.get_live_vars_info(jit_pc, sd.op_live);
        let all_liveness = &sd.liveness_info;
        if off + 2 >= all_liveness.len() {
            return Vec::new();
        }
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        let mut cursor = off + 3;
        let mut out: Vec<u32> = Vec::with_capacity((length_i + length_r + length_f) as usize);
        use majit_translate::liveness::LivenessIterator;
        for length in [length_i, length_r, length_f] {
            if length == 0 {
                continue;
            }
            let mut it = LivenessIterator::new(cursor, length, all_liveness);
            while let Some(reg_idx) = it.next() {
                out.push(reg_idx);
            }
            cursor = it.offset;
        }
        out
    })
}

/// Sentinel null JitCode for uninitialized PyreSym.
///
/// Cannot be `static` because `Arc::new` is not const; use a thread_local
/// LazyCell so the initialiser runs once per thread and the resulting
/// reference stays valid for the thread's lifetime.
thread_local! {
    static NULL_JITCODE_CELL: std::cell::OnceCell<JitCode> = const { std::cell::OnceCell::new() };
}

fn null_jitcode() -> &'static JitCode {
    NULL_JITCODE_CELL.with(|cell| {
        let r = cell.get_or_init(|| JitCode {
            code: std::ptr::null(),
            index: -1,
            payload: std::sync::Arc::new(crate::PyJitCode::skeleton(None)),
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

    pub fn opref_only(opref: OpRef) -> Self {
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

/// Convert pyre's `ConcreteValue` (tagged Int/Float/Ref/Null) into the
/// `majit_ir::Value` encoding used by `TraceCtx.virtualizable_values`.
/// `Null` maps to `Value::Ref(GcRef::NULL)` — the vable shadow must hold a
/// typed value for every slot, so we round-trip untracked refs as NULL
/// pointers rather than losing the type.
pub(crate) fn concrete_to_value(concrete: ConcreteValue) -> majit_ir::Value {
    match concrete {
        ConcreteValue::Int(v) => majit_ir::Value::Int(v),
        ConcreteValue::Float(v) => majit_ir::Value::Float(v),
        ConcreteValue::Ref(r) => majit_ir::Value::Ref(majit_ir::GcRef(r as usize)),
        ConcreteValue::Null => majit_ir::Value::Ref(majit_ir::GcRef::NULL),
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
            if is_int(obj) {
                ConcreteValue::Int(w_int_get_value(obj))
            } else if is_float(obj) {
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
            _ => None,
        }
    }

    /// RPython box.getfloatstorage() parity.
    pub fn getfloat(&self) -> Option<f64> {
        match self {
            ConcreteValue::Float(v) => Some(*v),
            ConcreteValue::Int(v) => Some(*v as f64),
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
            ConcreteValue::Int(_) => Type::Int,
            ConcreteValue::Float(_) => Type::Float,
            ConcreteValue::Ref(_) => Type::Ref,
            ConcreteValue::Null => Type::Ref,
        }
    }

    /// Truth value (RPython box.getint() != 0 for goto_if_not).
    pub fn is_truthy(&self) -> bool {
        match self {
            ConcreteValue::Int(v) => *v != 0,
            ConcreteValue::Float(v) => *v != 0.0,
            ConcreteValue::Ref(obj) => objspace_truth_value(*obj),
            ConcreteValue::Null => false,
        }
    }
}

/// Convert a bytecode constant to ConcreteValue.
pub fn load_const_concrete(constant: &pyre_interpreter::bytecode::ConstantData) -> ConcreteValue {
    use pyre_interpreter::bytecode::ConstantData;
    match constant {
        ConstantData::Integer { value } => match i64::try_from(value).ok() {
            Some(v) => ConcreteValue::Int(v),
            None => ConcreteValue::Ref(pyre_object::w_long_new(value.clone())),
        },
        ConstantData::Float { value } => ConcreteValue::Float(*value),
        ConstantData::Boolean { value } => ConcreteValue::Int(*value as i64),
        ConstantData::Str { value } => ConcreteValue::Ref(pyre_object::w_str_new(
            value.as_str().expect("non-UTF-8 string constant"),
        )),
        ConstantData::None => ConcreteValue::Ref(pyre_object::w_none()),
        _ => ConcreteValue::Null,
    }
}

use pyre_interpreter::{DictStorage, decode_instruction_at};

use crate::descr::{
    float_floatval_descr, int_intval_descr, make_array_descr, make_size_descr, w_float_size_descr,
    w_int_size_descr,
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
    #[vable(valuestackdepth)]
    pub valuestackdepth: usize,
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
    // ── Persistent symbolic frame field tracking ──
    // Stage 3.4 Phase C: the Python stack (`locals_cells_stack_w[nlocals..]`)
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
    /// Bridge-specific override for symbolic_local_types.
    /// virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
    /// is a W_Root array → all items are Type::Ref. setup_bridge_sym
    /// populates this with all-Ref; downstream unboxing happens in
    /// opcode handlers via guard_class + getfield_gc_pure_i/_f, not at
    /// the virtualizable slot level.
    pub(crate) bridge_local_types: Option<Vec<Type>>,
    // virtualizable.py:86-93: ALL static fields in declared order.
    // RPython's unroll_static_fields includes every field from
    // _virtualizable_; ALL must be inputarg (not info_only).
    #[vable(inputarg)]
    pub(crate) vable_last_instr: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_pycode: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_valuestackdepth: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_debugdata: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_lastblock: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_w_globals: OpRef,
    #[vable(array_base)]
    pub(crate) vable_array_base: Option<u32>,
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
    /// Namespace for global lookups.
    pub(crate) concrete_namespace: *mut pyre_interpreter::DictStorage,
    /// Execution context pointer (for creating callee frames).
    pub(crate) concrete_execution_context: *const pyre_interpreter::PyExecutionContext,
    /// Virtualizable object pointer (PyFrame).
    /// RPython MetaInterp stores the virtualizable separately from MIFrame.
    pub(crate) concrete_vable_ptr: *mut u8,
    /// Function-entry traces use typed locals (RPython MIFrame parity).
    pub(crate) is_function_entry_trace: bool,
    /// RPython `capture_resumedata(resumepc=orgpc)`
    /// (`pyjitpl.py:2586-2602`) swaps `frame.pc` to `orgpc` while
    /// snapshotting resume data, because a guard firing mid-opcode
    /// must resume at the opcode's start on failure. In RPython that
    /// one pc swap is enough: each jitcode instruction writes its
    /// result AFTER the guard it might fire, so `registers_{i,r,f}`
    /// still carry the pre-opcode values at guard time.
    ///
    /// Pyre dispatches a single Python bytecode into multiple SSA ops,
    /// some of which may write `registers_r` BEFORE a mid-opcode
    /// guard fires. That makes register state at guard time diverge
    /// from orgpc state, so pyre additionally snapshots
    /// `valuestackdepth` (`pre_opcode_vsd`) and the full `registers_r`
    /// (`pre_opcode_registers_r` below) at opcode start. Encoder
    /// consumers read through these snapshots. PRE-EXISTING-
    /// ADAPTATION for the stack-machine → register-machine dispatch
    /// difference.
    pub(crate) pre_opcode_vsd: Option<usize>,
    /// Stage 3.4 Phase B: full snapshot of `registers_r` at opcode
    /// start. Encoder consumers (`get_list_of_active_boxes`,
    /// `build_virtualizable_boxes`) read the unified abstract register
    /// file through this snapshot so that locals (idx < nlocals) and
    /// stack tail (idx >= nlocals) share a single indexing rule. When
    /// `None` the consumer falls back to the live `registers_r`.
    pub(crate) pre_opcode_registers_r: Option<Vec<OpRef>>,
    /// RPython MetaInterp.last_exc_value (pyjitpl.py:2745): concrete
    /// exception object pending during tracing. Set by execute_ll_raised
    /// (raise_varargs), consumed by handle_possible_exception.
    pub(crate) last_exc_value: pyre_object::PyObjectRef,
    /// RPython MetaInterp.class_of_last_exc_is_const (pyjitpl.py:2754):
    /// True after GUARD_EXCEPTION or GUARD_CLASS on the exception.
    pub(crate) class_of_last_exc_is_const: bool,
    /// RPython MetaInterp.last_exc_box (pyjitpl.py:3386): symbolic OpRef
    /// for the exception value. Set by handle_possible_exception after
    /// GUARD_EXCEPTION, consumed by finishframe_exception for stack push.
    pub(crate) last_exc_box: OpRef,
    /// pyjitpl.py:2597 virtualref_boxes: pairs of (jit_virtual, real_vref).
    /// Each pair: (symbolic OpRef, concrete pointer).
    /// resume.py:1093 restores virtual references on guard failure.
    /// Pairs stored flat: [virt_sym, virt_ptr, real_sym, real_ptr, ...].
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,
    // ── RPython MIFrame.registers_{i,r,f} port (pyjitpl.py:74-78) ──
    //
    // RPython reference (target shape):
    //   self.registers_i = [history.CONST_NULL] * jitcode.num_regs_i()
    //   self.registers_r = [history.CONST_NULL] * jitcode.num_regs_r()
    //   self.registers_f = [history.CONST_NULL] * jitcode.num_regs_f()
    //
    // `registers_r` is the abstract register file: index `[0, nlocals)`
    // holds local slots, index `[nlocals, nlocals+stack_only)` holds the
    // operand-stack tail. The tracer and the blackhole bridge-resume
    // decoder index this single Vec — Stage 3.4 Phase A/B/C collapsed
    // the legacy `symbolic_stack` side-Vec into the tail.
    //
    // `registers_i` / `registers_f` are placeholders for the same
    // abstract-register-file shape per kind (RPython's int/float
    // register banks). They are currently unused because
    // `filter_liveness_in_place` (`pyre-jit/src/jit/codewriter.rs`)
    // drops non-Ref registers from the live set — extending these
    // banks is part of the tracer rework required to drop the
    // LiveVars intersection and make SSA authoritative (see
    // `phase4_ssa_liveness_blocker_2026_04_18.md`).
    pub(crate) registers_i: Vec<OpRef>,
    #[vable(locals)]
    pub(crate) registers_r: Vec<OpRef>,
    pub(crate) registers_f: Vec<OpRef>,
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
    /// PyPy capture_resumedata: parent frame chain for multi-frame guards.
    /// Each element is (fail_args, fail_arg_types, resumepc, jitcode_index).
    /// opencoder.py:819-834: walks framestack to build parent snapshot chain.
    pub parent_frames: Vec<(Vec<OpRef>, Vec<Type>, usize, i32)>,
    pub pending_inline_frame: Option<PendingInlineFrame>,
}

pub(crate) fn code_has_backward_jump(code: &CodeObject) -> bool {
    for pc in 0..code.instructions.len() {
        let Some((instruction, _)) = decode_instruction_at(code, pc) else {
            continue;
        };
        if matches!(
            instruction,
            Instruction::JumpBackward { .. } | Instruction::JumpBackwardNoInterrupt { .. }
        ) {
            return true;
        }
    }
    false
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
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::CallFunctionEx { .. }
            | Instruction::StoreAttr { .. }
            | Instruction::DeleteAttr { .. }
            | Instruction::StoreSubscr
            | Instruction::DeleteSubscr
            | Instruction::ImportName { .. }
            | Instruction::ImportFrom { .. }
            // RPython opimpl_raise/opimpl_reraise: always Err, needs
            // GUARD_EXCEPTION + finishframe_exception for exception-path tracing.
            | Instruction::RaiseVarargs { .. }
            | Instruction::Reraise { .. }
    )
}

/// Environment context — currently unused.
pub struct PyreEnv;

pub(crate) fn pyobject_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Ref, false)
}

pub(crate) fn int_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Int, true)
}

pub(crate) fn float_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Float, false)
}

/// resume.py:656 arraydescr kind dispatch for virtual array materialization.
/// kind: 0=ref (is_array_of_pointers), 1=int, 2=float (is_array_of_floats).
pub(crate) fn array_descr_for_kind(kind: u8, _descr_index: u32) -> DescrRef {
    match kind {
        0 => pyobject_array_descr(),
        2 => float_array_descr(),
        _ => int_array_descr(),
    }
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
    make_size_descr(std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>())
}

pub(crate) fn frame_locals_cells_stack_descr() -> DescrRef {
    crate::descr::pyframe_locals_cells_stack_descr()
}

pub(crate) fn frame_dict_storage_descr() -> DescrRef {
    crate::descr::pyframe_dict_storage_descr()
}

pub(crate) fn wrapint(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    crate::helpers::emit_box_int_inline(ctx, value, w_int_size_descr(), int_intval_descr())
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
        Type::Ref | Type::Void => value,
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
        if is_int(concrete_value) {
            return Some(ctx.const_int(w_int_get_value(concrete_value)));
        }
        if is_bool(concrete_value) {
            return Some(ctx.const_int(if w_bool_get_value(concrete_value) {
                1
            } else {
                0
            }));
        }
    }
    None
}

/// pyjitpl.py:750-758: read container length.
///
/// RPython's `arraylen_gc` reads the GC array header — there is exactly one
/// length per array, so RPython keeps a per-box `heapc_deps[0]` slot. pyre
/// stores list/bytes/tuple lengths as plain struct fields, so the cached
/// value lives in the regular field cache (`heap_cache.field_cache`).
/// `opimpl_getfield_gc_i` already does that lookup, so this helper is now
/// just a thin alias kept for source-stability with the call sites.
pub(crate) fn trace_arraylen_gc(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    opimpl_getfield_gc_i(ctx, obj, descr)
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
    if let Some(cached) = ctx.heap_cache().getfield_cached(obj, field_index) {
        return cached;
    }
    // pyjitpl.py:1074-1089: quasi-immutable field handling.
    // Record the field as quasi-immut known so subsequent reads skip
    // the QUASIIMMUT_FIELD op. Emit GUARD_NOT_INVALIDATED if needed.
    // NOTE: GuardNotInvalidated is NOT emitted here — it requires
    // PyreSym.generate_guard for proper snapshot/fail_args (pyjitpl.py:1087
    // generate_guard parity). Instead, set a flag on ctx so the caller
    // (PyreSym with_ctx block) can emit it with full resume data.
    if descr.is_quasi_immutable() && !ctx.heap_cache().is_quasi_immut_known(obj, field_index) {
        ctx.heap_cache_mut().quasi_immut_now_known(obj, field_index);
        ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
        if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
            ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
        }
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureI
    } else {
        OpCode::GetfieldGcI
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr);
    ctx.heap_cache_mut()
        .getfield_now_known(obj, field_index, result);
    result
}

// Note: pyre does not currently route GetfieldGcF/GetfieldGcPureF through
// state.rs. Float field unboxing goes via the codewriter-generated
// `getfield_gc_f_pureornot` (majit-translate/src/codegen.rs),
// which — matching RPython's pyjitpl.py opimpl_getfield_gc_f — records
// the GC op without folding. The optimizer's `optimize_GETFIELD_GC_F`
// (= `optimize_GETFIELD_GC_I` via RPython's alias) handles folding.

/// Unbox int with proper GuardClass resume data via MIFrame::generate_guard.
pub(crate) fn trace_unbox_int_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    int_type_addr: i64,
) -> OpRef {
    trace_unbox_int_with_resume_descr(
        frame,
        ctx,
        obj,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )
}

pub(crate) fn trace_unbox_int_with_resume_descr(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    // pyjitpl.py GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0.
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut()
            .class_now_known(obj, majit_ir::GcRef(type_addr as usize));
    }
    crate::generated::trace_unbox_int(
        ctx,
        obj,
        type_addr,
        crate::descr::ob_type_descr(),
        intval_descr,
        &[],
    )
}

/// Unbox float with proper GuardClass resume data via MIFrame::generate_guard.
pub(crate) fn trace_unbox_float_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    float_type_addr: i64,
) -> OpRef {
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(float_type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut()
            .class_now_known(obj, majit_ir::GcRef(float_type_addr as usize));
    }
    crate::generated::trace_unbox_float(
        ctx,
        obj,
        float_type_addr,
        crate::descr::ob_type_descr(),
        crate::descr::float_floatval_descr(),
        &[],
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

pub(crate) fn frame_locals_cells_stack_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldRawI,
        &[frame],
        frame_locals_cells_stack_descr(),
    )
}

/// Read from frame's locals_cells_stack_w array.
/// Uses GcR (Ref-typed) to match RPython's GETARRAYITEM_GC_R,
/// ensuring the optimizer knows these are boxed pointers.
pub(crate) fn trace_array_getitem_value(ctx: &mut TraceCtx, array: OpRef, index: OpRef) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heap_cache().getarrayitem_cache(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr);
    ctx.heap_cache_mut()
        .getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// Read from frame's locals_cells_stack_w — namespace access path.
pub(crate) fn trace_raw_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heap_cache().getarrayitem_cache(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr);
    ctx.heap_cache_mut()
        .getarrayitem_now_known(array, index, descr_idx, result);
    result
}

pub(crate) fn trace_raw_int_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetarrayitemRawI, &[array, index], int_array_descr())
}

pub(crate) fn trace_raw_float_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetarrayitemRawF,
        &[array, index],
        float_array_descr(),
    )
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
    ctx.heap_cache_mut()
        .setarrayitem_cache(array, index, descr_idx, value);
}

pub(crate) fn trace_raw_int_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        int_array_descr(),
    );
}

pub(crate) fn trace_raw_float_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        float_array_descr(),
    );
}

pub(crate) fn frame_get_namespace(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldGcR, &[frame], frame_dict_storage_descr())
}

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

pub(crate) fn dict_storage_slot_direct(ns: *mut DictStorage, name: &str) -> Option<usize> {
    if ns.is_null() {
        return None;
    }
    unsafe { &*ns }.slot_of(name)
}

pub(crate) fn dict_storage_value_direct(ns: *mut DictStorage, idx: usize) -> Option<PyObjectRef> {
    if ns.is_null() {
        return None;
    }
    unsafe { &*ns }.get_slot(idx)
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

/// virtualizable.py:126/139 parity: box value for frame array slot.
/// Frame array items (locals_cells_stack_w[*]) are declared as GCREF
/// (interp_jit.py:25). The optimizer may unbox ints/floats in fail_args;
/// this function re-boxes them for the frame. Ref values pass through.
pub(crate) fn virtualizable_box_value(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
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

pub fn pending_inline_result_from_concrete(
    result_type: Type,
    concrete_result: PyObjectRef,
) -> PendingInlineResult {
    match result_type {
        Type::Int => PendingInlineResult::Int(unsafe { w_int_get_value(concrete_result) }),
        Type::Float => PendingInlineResult::Float(unsafe {
            pyre_object::floatobject::w_float_get_value(concrete_result)
        }),
        Type::Ref | Type::Void => PendingInlineResult::Ref(concrete_result),
    }
}

pub fn materialize_pending_inline_result(result: PendingInlineResult) -> PyObjectRef {
    match result {
        PendingInlineResult::Ref(result) => result,
        PendingInlineResult::Int(value) => w_int_new(value),
        PendingInlineResult::Float(value) => pyre_object::floatobject::w_float_new(value),
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

pub(crate) fn fail_arg_types_for_virtualizable_state(len: usize) -> Vec<Type> {
    let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    crate::virtualizable_gen::virt_live_value_types(len.saturating_sub(n))
}

impl PyreSym {
    pub(crate) fn new_uninit(frame: OpRef) -> Self {
        Self {
            frame,
            symbolic_local_types: Vec::new(),
            symbolic_stack_types: Vec::new(),
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            bridge_local_oprefs: None,
            bridge_local_types: None,
            vable_last_instr: OpRef::NONE,
            vable_pycode: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            vable_debugdata: OpRef::NONE,
            vable_lastblock: OpRef::NONE,
            vable_w_globals: OpRef::NONE,
            vable_array_base: None,
            concrete_locals: Vec::new(),
            concrete_stack: Vec::new(),
            // jitcode and concrete_namespace initialized below
            jitcode: null_jitcode() as *const JitCode,
            concrete_namespace: std::ptr::null_mut(),
            is_function_entry_trace: false,
            concrete_execution_context: std::ptr::null(),
            concrete_vable_ptr: std::ptr::null_mut(),
            pre_opcode_vsd: None,
            pre_opcode_registers_r: None,
            last_exc_value: std::ptr::null_mut(),
            class_of_last_exc_is_const: false,
            last_exc_box: OpRef::NONE,
            virtualref_boxes: Vec::new(),
            // RPython pyjitpl.py:74-78 init: registers_X[i] = CONST_NULL for
            // i in num_regs. Sized lazily here — Stage 3.2 will resize each
            // register file when the owning JitCode's num_regs is known.
            registers_i: Vec::new(),
            registers_r: Vec::new(),
            registers_f: Vec::new(),
        }
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
        self.locals_cells_stack_array_ref = if self.vable_array_base.is_some() {
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
            // rebuild_from_resumedata (Box(n) → bridge InputArg OpRef(n)).
            let mut locals = overrides.clone();
            locals.resize(nlocals, OpRef::NONE);
            locals
        } else if let Some(base) = self.vable_array_base {
            (0..nlocals).map(|i| OpRef(base + i as u32)).collect()
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
                                inputarg_types.get(opref.0 as usize).copied()
                            }
                        })
                        .unwrap_or(Type::Ref)
                })
                .collect();
            // Bridges have no symbolic stack at entry — the stack is
            // empty when control resumes from the failing guard. Leave
            // the stack-types vector empty so it gets reconstructed from
            // the concrete frame below.
            Some((locals, Vec::<Type>::new()))
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
        // Stage 3.4 Phase C: seed the stack portion of `registers_r`
        // directly. `registers_r` is the unified abstract register
        // file — locals occupy `[..nlocals]` and stack slots occupy
        // `[nlocals..nlocals + stack_only_depth]` (RPython
        // `pyjitpl.py:70-78` MIFrame parity).
        let stack_seed: Vec<OpRef> = if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef(stack_base + i as u32))
                .collect()
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
            // (FixedObjectArray::filled(..., PY_NULL)) and the helper pads
            // them with a shared const-NULL OpRef.
            let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            let live_prefix = nlocals + stack_only_depth;
            let array_len = concrete_frame_array_len(concrete_frame).unwrap_or(live_prefix);
            // Static fields inputargs OpRef(1)..OpRef(NUM_SCALARS).
            let scalar_oprefs: Vec<OpRef> = (1..num_scalars).map(|i| OpRef(i as u32)).collect();
            // Array items inputargs OpRef(base..base + live_prefix).
            let array_items: Vec<OpRef> =
                (0..live_prefix).map(|i| OpRef(base + i as u32)).collect();
            let vable_ref = OpRef(crate::virtualizable_gen::SYM_FRAME_IDX);
            // pyjitpl.py:3302 initialize_virtualizable parity: the concrete
            // half of virtualizable_boxes at portal entry comes from a live
            // heap read (vinfo.read_boxes(cpu, virtualizable, 0)). There is
            // no resume-data stream at root-trace start.
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            let array_lengths = [array_len];
            let (input_values, vable_ref_value) = if concrete_frame != 0 {
                let (static_boxes, array_boxes) =
                    unsafe { info.read_all_boxes(concrete_frame as *const u8, &array_lengths) };
                let mut values = Vec::with_capacity(num_scalars - 1 + array_len);
                for (i, bits) in static_boxes.iter().enumerate() {
                    values.push(value_for_slot(info.static_fields[i].field_type, *bits));
                }
                for (a, items) in array_boxes.iter().enumerate() {
                    let item_ty = info.array_fields[a].item_type;
                    for bits in items {
                        values.push(value_for_slot(item_ty, *bits));
                    }
                }
                while values.len() < (num_scalars - 1) + array_len {
                    values.push(majit_ir::Value::Ref(majit_ir::GcRef::NULL));
                }
                (
                    values,
                    majit_ir::Value::Ref(majit_ir::GcRef(concrete_frame)),
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

    /// Stack-only depth (number of values on the operand stack).
    #[inline]
    pub fn stack_only_depth(&self) -> usize {
        self.valuestackdepth.saturating_sub(self.nlocals)
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at(&self, abs_idx: usize) -> ConcreteValue {
        if abs_idx < self.nlocals {
            self.concrete_locals
                .get(abs_idx)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        } else {
            let stack_idx = abs_idx - self.nlocals;
            self.concrete_stack
                .get(stack_idx)
                .copied()
                .unwrap_or(ConcreteValue::Null)
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

    fn namespace_ptr(&self) -> Option<*mut DictStorage> {
        let frame_ptr = self.frame_ptr()?;
        let namespace_ptr =
            unsafe { *(frame_ptr.add(PYFRAME_W_GLOBALS_OFFSET) as *const *mut DictStorage) };
        (!namespace_ptr.is_null()).then_some(namespace_ptr)
    }

    fn namespace_len(&self) -> usize {
        let Some(namespace_ptr) = self.namespace_ptr() else {
            return 0;
        };
        unsafe { (*namespace_ptr).len() }
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
/// Currently unused: setup_bridge_sym still routes Virtual entries via
/// the existing vable scalar override path. The helper is added now so
/// future setup_bridge_sym work can drop the OpRef::NONE fallback for
/// RebuiltValue::Virtual without re-deriving the materialization logic.
#[allow(dead_code)]
fn materialize_bridge_virtual(
    ctx: &mut majit_metainterp::TraceCtx,
    vidx: usize,
    rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    cache: &mut std::collections::HashMap<usize, OpRef>,
) -> OpRef {
    use majit_ir::OpCode;
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};

    // resume.py:951 virtuals_cache.get_ptr(index): hit → return cached.
    if let Some(&cached) = cache.get(&vidx) {
        return cached;
    }

    let Some(virtuals) = rd_virtuals else {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-virtual] missing rd_virtuals (vidx={}), abort materialization",
                vidx
            );
        }
        return OpRef::NONE;
    };
    let Some(entry) = virtuals.get(vidx) else {
        return OpRef::NONE;
    };

    // resume.py:1556-1564 decode_box parity for fieldnums (i16 tagged).
    fn decode_fieldnum(
        ctx: &mut majit_metainterp::TraceCtx,
        tagged: i16,
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut std::collections::HashMap<usize, OpRef>,
    ) -> OpRef {
        if tagged == majit_ir::resumedata::UNINITIALIZED_TAG {
            return OpRef::NONE;
        }
        let (val, tagbits) = untag(tagged);
        match tagbits {
            TAGBOX => {
                // resume.py:1556-1564 decode_box parity:
                //   if num < 0: num += len(liveboxes)
                //   return liveboxes[num]
                // Negative `val` is Python-style indexing into the parent
                // guard's liveboxes array (`num_failargs` long). For
                // bridges, the boxes are inputargs at OpRef(0..n_inputargs).
                let idx = if val < 0 {
                    val + resume_data.num_failargs
                } else {
                    val
                };
                if idx < 0 {
                    OpRef::NONE
                } else {
                    OpRef(idx as u32)
                }
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
                let (raw, tp) = resume_data.rd_consts[ci];
                match tp {
                    majit_ir::Type::Ref => ctx.const_ref(raw),
                    majit_ir::Type::Float => ctx.const_float(raw),
                    _ => ctx.const_int(raw),
                }
            }
            TAGVIRTUAL => {
                materialize_bridge_virtual(ctx, val as usize, rd_virtuals, resume_data, cache)
            }
            _ => OpRef::NONE,
        }
    }

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
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut std::collections::HashMap<usize, OpRef>,
    ) {
        for (fd_info, &fnum) in fielddescrs.iter().zip(fieldnums.iter()) {
            if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                continue;
            }
            let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
            if value.is_none() {
                continue;
            }
            let signed = matches!(
                fd_info.field_type,
                majit_ir::Type::Int | majit_ir::Type::Float
            );
            // RPython: decoder.setfield(struct, fieldnum, fielddescr).
            // fielddescr carries parent_descr so descr_index() resolves
            // to index_in_parent (small sequential) not stable_field_index (268M hash).
            let field_descr = crate::descr::make_field_descr_with_parent(
                parent_descr.clone(),
                fd_info.offset,
                fd_info.field_size,
                fd_info.field_type,
                signed,
            );
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_op, value], field_descr.clone());
            ctx.heap_cache_mut()
                .setfield_cached(struct_op, fd_info.index, value);
        }
    }

    match entry {
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
            cache.insert(vidx, new_op);
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
                    "[jit][bridge-virtual] vidx={} VirtualInfo → OpRef({})",
                    vidx, new_op.0,
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
            cache.insert(vidx, new_op);
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
                    "[jit][bridge-virtual] vidx={} VStructInfo → OpRef({})",
                    vidx, new_op.0,
                );
            }
            new_op
        }
        // resume.py:649-671 AbstractVArrayInfo.allocate (clear=True or False)
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            fieldnums,
            kind,
            descr_index,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            fieldnums,
            kind,
            descr_index,
            ..
        } => {
            let clear = matches!(entry, majit_ir::RdVirtualInfo::VArrayInfoClear { .. });
            let kind = *kind;
            let descr_index = *descr_index;
            let length = fieldnums.len();
            let len_ref = ctx.const_int(length as i64);
            // resume.py:653 decoder.allocate_array(length, arraydescr, self.clear)
            let alloc_opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            let array_descr = array_descr_for_kind(kind, descr_index);
            let new_op = ctx.record_op_with_descr(alloc_opcode, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:654 decoder.virtuals_cache.set_ptr(index, array)
            cache.insert(vidx, new_op);
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
                    "[jit][bridge-virtual] vidx={} VArrayInfo(clear={}) → OpRef({})",
                    vidx, clear, new_op.0,
                );
            }
            new_op
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            descr_index,
            fielddescrs,
            size,
            base_size,
            item_size,
            fieldnums,
            ..
        } => {
            let len_ref = ctx.const_int(*size as i64);
            // resume.py:749: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
            let array_descr = arraydescr.clone().unwrap_or_else(|| {
                crate::descr::make_struct_array_descr(*descr_index, *base_size, *item_size)
            });
            let new_op =
                ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:751: decoder.virtuals_cache.set_ptr(index, array)
            cache.insert(vidx, new_op);
            // resume.py:752-759:
            //   p = 0
            //   for i in range(self.size):
            //       for j in range(len(self.fielddescrs)):
            //           num = self.fieldnums[p]
            //           if not tagged_eq(num, UNINITIALIZED):
            //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
            //           p += 1
            let num_fields = fielddescrs.len();
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    if p >= fieldnums.len() {
                        break;
                    }
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
                    "[jit][bridge-virtual] vidx={} VArrayStructInfo → OpRef({})",
                    vidx, new_op.0,
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
            let calldescr = crate::descr::make_raw_malloc_calldescr();
            let buffer = ctx.record_op_with_descr(OpCode::CallI, &[func_ref, size_ref], calldescr);
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.insert(vidx, buffer);
            // resume.py:705-708: setrawbuffer_item for each offset/descr
            for (i, (&off, &fnum)) in offsets.iter().zip(fieldnums.iter()).enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                // resume.py:1232: itembox = self.decode_box(fieldnum, kind)
                let item = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if item.is_none() {
                    continue;
                }
                // resume.py:1225-1234: setrawbuffer_item (direct reader).
                // Dispatches pointer/float/int via arraydescr — all types allowed.
                let di = &descrs[i];
                let tp = match di.item_type {
                    0 => majit_ir::Type::Ref,
                    2 => majit_ir::Type::Float,
                    _ => majit_ir::Type::Int,
                };
                let store_descr =
                    crate::descr::make_array_descr(di.base_size, di.item_size, tp, di.is_signed);
                let offset_ref = ctx.const_int(off as i64);
                ctx.record_op_with_descr(
                    OpCode::RawStore,
                    &[buffer, offset_ref, item],
                    store_descr,
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawBufferInfo(func={:#x}, size={}) → OpRef({})",
                    vidx, func, size, buffer.0,
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
            cache.insert(vidx, buffer);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawSliceInfo(offset={}) → OpRef({})",
                    vidx, offset, buffer.0,
                );
            }
            buffer
        }
        // resume.py:763-870 VStr/VUni*Info — bridge-virtual materialization
        // for virtual strings requires vstring.py force_box port (info.py:142
        // VStringPlainInfo etc.) + host runtime hooks for allocate_string /
        // string_setitem / concat_strings / slice_string. Until those land,
        // the bridge tracer must fail loudly on these variants rather than
        // return NONE and corrupt downstream OpRef references.
        majit_ir::RdVirtualInfo::VStrPlainInfo { .. }
        | majit_ir::RdVirtualInfo::VStrConcatInfo { .. }
        | majit_ir::RdVirtualInfo::VStrSliceInfo { .. }
        | majit_ir::RdVirtualInfo::VUniPlainInfo { .. }
        | majit_ir::RdVirtualInfo::VUniConcatInfo { .. }
        | majit_ir::RdVirtualInfo::VUniSliceInfo { .. } => {
            panic!(
                "pyre-jit-trace bridge-virtual: VStr/VUni RdVirtualInfo \
                 reached without vstring.py materialization support \
                 (see resume.py:763-870)"
            )
        }
        majit_ir::RdVirtualInfo::Empty => OpRef::NONE,
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
        PyreMeta {
            num_locals,
            ns_len: self.namespace_len(),
            valuestackdepth: vsd,
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
        crate::virtualizable_gen::virt_extract_live_values(
            self.frame,
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
        )
    }

    // virtualizable.py:86 read_boxes() + warmstate.py:73 wrap() parity:
    // Array items (locals_cells_stack_w) are GC pointers → RefFrontendOp.
    // No pre-unboxing at function entry. Unboxing happens during tracing
    // via guard_class + getfield_gc_i when arithmetic/compare handlers
    // encounter Ref-typed operands.

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        crate::virtualizable_gen::virt_live_value_types(meta.slot_types.len())
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.init_vable_indices();
        sym.nlocals = _meta.num_locals;
        sym.valuestackdepth = _meta.valuestackdepth;
        // virtualizable.py:44 + interp_jit.py:25-31: all locals_cells_stack_w
        // items are W_Root → Type::Ref. Unboxing happens inside trace opcode
        // handlers (guard_class + getfield_gc_pure_i/_f), not at slot setup.
        sym.symbolic_local_types = vec![Type::Ref; _meta.num_locals.min(_meta.slot_types.len())];
        sym.symbolic_stack_types =
            vec![Type::Ref; _meta.slot_types.len().saturating_sub(_meta.num_locals)];
        let stack_only = _meta.vable_stack_only_depth();
        sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
        sym
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        // pypy/module/pypyjit/interp_jit.py:67-70 PyPyJitDriver:
        //   reds = ['frame', 'ec']
        //   greens = ['next_instr', 'is_being_profiled', 'pycode']
        //   virtualizables = ['frame']
        //
        // Returns None until the CA emission path is migrated to the
        // reds-only shape. Activating this triggers
        // `patch_new_loop_to_load_virtualizable_fields` (compile.py:504-511),
        // which reduces the compiled callee loop's inputargs to
        // `[frame]`. That breaks the caller-side `CallAssemblerR` whose
        // `VableExpansion` still passes the 8-arg expanded shape to the
        // callee (rewrite.py:672-683 contract: CA arg count must match
        // target num_red_args). Both ends must flip together.
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // warmstate.py:503-511: RPython enters assembler unconditionally
        // when procedure_token exists. No next_instr check —
        // the compiled code's preamble handles entry from any PC.
        // Shape checks (nlocals, namespace) ensure frame layout matches.
        self.local_count() == meta.num_locals && self.namespace_len() == meta.ns_len
    }

    fn update_meta_for_bridge(meta: &mut Self::Meta, fail_arg_types: &[Type]) {
        meta.vable_update_vsd_from_len(
            fail_arg_types.len(),
            crate::virtualizable_gen::NUM_SCALAR_INPUTARGS,
        );
    }

    fn setup_bridge_sym(
        sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
        fail_values: &[i64],
        fail_types: &[Type],
    ) {
        use majit_ir::resumedata::RebuiltValue;

        if resume_data.frames.is_empty() {
            return;
        }

        // virtualizable.py:139 load_list_of_boxes parity: decode each
        // RebuiltValue in the resume stream into a typed Value. The type
        // is the fixed Box kind the encoder recorded at numbering time
        // (fail_arg_types[idx] for Box variants), matching RPython's
        // immutable Box.type invariant. No heap read.
        let decode_concrete = |v: &RebuiltValue| -> majit_ir::Value {
            match v {
                RebuiltValue::Box(n, tp) => {
                    // resume.py:1260 decode_box Box branch: liveboxes[num]
                    // is authoritative. The encoder guarantees num in range.
                    // Silent unwrap_or(0) would mask encoder/decoder drift
                    // and write null into Ref slots / zero into Int slots.
                    let bits = *fail_values.get(*n).unwrap_or_else(|| {
                        panic!(
                            "decode_concrete: fail_values[{n}] out of range \
                             (len={}) — encoder/decoder liveboxes mismatch",
                            fail_values.len()
                        )
                    });
                    let effective_tp = fail_types.get(*n).copied().unwrap_or(*tp);
                    value_for_slot(effective_tp, bits)
                }
                RebuiltValue::Int(v) => majit_ir::Value::Int(*v as i64),
                RebuiltValue::Const(raw, tp) => value_for_slot(*tp, *raw),
                // resume.py:1245 TAGVIRTUAL → getvirtual_ptr: RPython returns
                // a Ref-typed Box pointing at the virtual's materialized
                // allocation. pyre delays materialization to bridge runtime
                // (NEW_WITH_VTABLE inside the trace), so the concrete
                // pointer is not observable at trace time. Seed the shadow
                // with a Ref-typed null placeholder to preserve Box.type
                // immutability at the label (virtualstate.py:417) — the
                // real pointer reaches the live heap when the bridge
                // executes its NEW_WITH_VTABLE + SETFIELD sequence.
                RebuiltValue::Virtual(_) => majit_ir::Value::Ref(majit_ir::GcRef::NULL),
                RebuiltValue::Unassigned => majit_ir::Value::Void,
            }
        };

        // resume.py:874-899 VirtualCache parity: per-bridge cache so shared
        // / recursive virtuals materialize exactly once.
        let mut virtuals_cache: std::collections::HashMap<usize, OpRef> =
            std::collections::HashMap::new();

        // resume.py:1245 decode_box parity. Each tagged variant resolves
        // to a live box whose `.type` is the kind the encoder dispatched:
        //   TAGBOX   → liveboxes[n]                (Box)
        //   TAGINT   → ConstInt(num)               (Int)
        //   TAGCONST → self.consts[num-OFFSET]     (Const)
        //   TAGVIRTUAL → self.getvirtual_*(num)    (Virtual)
        // In majit a "live box" is an OpRef plus a registered entry in
        // the trace's constant pool (for constants) or an inputarg slot
        // (for boxes). Constants therefore MUST be registered through
        // `ctx.const_int/const_ref/const_float` so the returned OpRef
        // maps to a real value+type in the pool; synthesizing
        // `OpRef::from_const(cursor)` without registration leaves a
        // dangling index that the optimizer later re-types arbitrarily.
        let resolve = |ctx: &mut majit_metainterp::TraceCtx,
                       cache: &mut std::collections::HashMap<usize, OpRef>,
                       v: &RebuiltValue|
         -> OpRef {
            match v {
                RebuiltValue::Box(n, _tp) => OpRef(*n as u32),
                RebuiltValue::Int(v) => ctx.const_int(*v as i64),
                RebuiltValue::Const(raw, tp) => match tp {
                    Type::Ref => ctx.const_ref(*raw),
                    Type::Float => ctx.const_float(*raw),
                    _ => ctx.const_int(*raw),
                },
                RebuiltValue::Virtual(vidx) => {
                    materialize_bridge_virtual(ctx, *vidx as usize, rd_virtuals, resume_data, cache)
                }
                _ => OpRef::NONE,
            }
        };

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
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let mut vable_array_items: Vec<OpRef> =
            Vec::with_capacity(vvals.len().saturating_sub(num_scalars));
        let mut vable_scalar_values: Vec<majit_ir::Value> =
            Vec::with_capacity(num_scalars.saturating_sub(1));
        let mut vable_array_values: Vec<majit_ir::Value> =
            Vec::with_capacity(vvals.len().saturating_sub(num_scalars));
        let mut vable_ref_value = majit_ir::Value::Void;
        for (i, v) in vvals.iter().enumerate() {
            let resolved = resolve(ctx, &mut virtuals_cache, v);
            let concrete = decode_concrete(v);
            match i {
                0 => vable_ref_value = concrete, // frame_ptr — implicit OpRef, explicit concrete
                1 => {
                    sym.vable_last_instr = resolved;
                    vable_scalar_values.push(concrete);
                }
                2 => {
                    sym.vable_pycode = resolved;
                    vable_scalar_values.push(concrete);
                }
                3 => {
                    sym.vable_valuestackdepth = resolved;
                    vable_scalar_values.push(concrete);
                }
                4 => {
                    sym.vable_debugdata = resolved;
                    vable_scalar_values.push(concrete);
                }
                5 => {
                    sym.vable_lastblock = resolved;
                    vable_scalar_values.push(concrete);
                }
                6 => {
                    sym.vable_w_globals = resolved;
                    vable_scalar_values.push(concrete);
                }
                _ => {
                    vable_array_items.push(resolved);
                    vable_array_values.push(concrete);
                }
            }
        }

        // Part 2 — frame registers (consume_boxes): walk the frame section
        // in liveness enumeration order ([int..., ref..., float...]), map
        // each decoded value back to its register slot via
        // `frame_liveness_reg_indices_at`, and write it into the unified
        // abstract register file (`registers_r`). This mirrors
        // resume.py:1054 `consume_boxes`
        // (`_callback_i/_r/_f(register_index)` writing to
        // `f.registers_i/_r/_f[index]` at the exact slot liveness
        // declared, not at an enumerate-order position). RPython indexes
        // a single `registers_X` vector by abstract register color —
        // there is no `idx < nlocals` decode.
        let frame0 = &resume_data.frames[0];
        let reg_indices =
            crate::state::frame_liveness_reg_indices_at(frame0.jitcode_index, frame0.pc);
        let stack_only = sym.stack_only_depth();
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
            reg_indices.len() == frame0.values.len(),
            "setup_bridge_sym: reg_indices len={} != frame.values len={} at pc={}",
            reg_indices.len(),
            frame0.values.len(),
            frame0.pc,
        );
        for (value, &reg_idx) in frame0.values.iter().zip(reg_indices.iter()) {
            let resolved = resolve(ctx, &mut virtuals_cache, value);
            let idx = reg_idx as usize;
            if idx < bridge_registers_r.len() {
                bridge_registers_r[idx] = resolved;
            }
        }
        for (idx, slot) in bridge_registers_r.iter_mut().enumerate() {
            if slot.is_none() {
                // Fallback retained only for slots genuinely outside the
                // liveness coverage (stack-only depth ahead of the local
                // prefix). vable_array_items is now A.1-synchronized so
                // the read is authoritative.
                *slot = vable_array_items.get(idx).copied().unwrap_or(OpRef::NONE);
            }
        }
        let bridge_locals: Vec<OpRef> = bridge_registers_r.iter().take(nlocals).copied().collect();

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
        // OpRef(base+i) values from the PARENT trace's namespace, leaving
        // stale parent OpRefs in registers_r after we set
        // bridge_local_oprefs here.
        //
        // virtualizable.py:44 + interp_jit.py:25-31: array item types are
        // all Ref; RETURN_VALUE / arithmetic paths unbox via
        // `trace_guarded_int_payload` (guard_class + getfield_gc_pure_i),
        // matching the RPython unbox-at-consumer model. The slot-level
        // type override is NOT how RPython avoids the guarded path.
        sym.registers_r = bridge_registers_r.clone();
        sym.symbolic_local_types = {
            let mut types = bridge_local_types.clone();
            types.resize(sym.nlocals, Type::Ref);
            types
        };
        // The bridge inputs do NOT have the 7-slot scalar header that
        // init_vable_indices assumes. Clear vable_array_base so any later
        // LOAD_FAST falling through to the vable_array_base branch uses
        // the heap-array path instead of synthesizing parent OpRefs.
        sym.vable_array_base = None;
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
        let bridge_array_len =
            concrete_frame_array_len(sym.concrete_vable_ptr as usize).unwrap_or(nlocals);
        let scalar_oprefs = [
            sym.vable_last_instr,
            sym.vable_pycode,
            sym.vable_valuestackdepth,
            sym.vable_debugdata,
            sym.vable_lastblock,
            sym.vable_w_globals,
        ];
        // virtualizable.py:139 load_list_of_boxes parity: both halves of
        // virtualizable_boxes come from the resume-data stream — OpRefs via
        // the resolve() closure above, concrete Values via decode_concrete().
        // No heap read. The seed helper pads short arrays with const-NULL
        // OpRef; match that here by padding concrete values with
        // Value::Ref(GcRef::NULL) to the same length.
        let mut concrete_values = Vec::with_capacity(vable_scalar_values.len() + bridge_array_len);
        concrete_values.extend_from_slice(&vable_scalar_values);
        let taken_concrete = vable_array_values.len().min(bridge_array_len);
        concrete_values.extend_from_slice(&vable_array_values[..taken_concrete]);
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
        // Bridge stack: the target loop header has stack_only=0, so no
        // stack tail needs to be appended to registers_r.
        sym.symbolic_stack_types = Vec::new();
        sym.valuestackdepth = sym.nlocals;
        sym.bridge_local_oprefs = Some(bridge_locals);
        sym.bridge_local_types = Some(bridge_local_types);
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
        rd_numb: Option<&[u8]>,
        rd_consts: Option<&[(i64, Type)]>,
    ) -> Option<majit_metainterp::ResumeDataResult> {
        use majit_ir::resumedata::rebuild_from_numbering;

        let rd_numb = rd_numb?;
        let rd_consts = rd_consts.unwrap_or(&[]);

        // resume.py:1049-1055 parity: consume_boxes(f.get_current_position_info())
        // RPython uses jitcode liveness via get_current_position_info; majit
        // routes the same lookup through `frame_value_count_at`.
        let cb = crate::state::frame_value_count_at;
        let (num_failargs, vable_values, vref_values, frames) =
            rebuild_from_numbering(rd_numb, rd_consts, fail_arg_types, Some(&cb));

        if frames.is_empty() {
            return None;
        }

        Some(majit_metainterp::ResumeDataResult {
            frames,
            virtualizable_values: vable_values,
            virtualref_values: vref_values,
            // resume.py:1071 self.consts = storage.rd_consts. Retained as
            // a slice-owned copy so `decode_box` in virtual materialization
            // (resume.py:1251) can index `consts[num - TAG_CONST_OFFSET]`.
            rd_consts: rd_consts.to_vec(),
            // resume.py:1042 num_failargs from rd_numb header. Used by
            // bridge virtual materialization (resume.py:1556-1564 decode_box
            // negative-index normalization: `num + len(liveboxes)`).
            num_failargs,
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
        // Update valuestackdepth from the merge point's box layout.
        // Layout: [Ref(frame), Int(ni), Ref(code), Int(vsd), Ref(ns), locals..., stack...]
        // PyreMeta.valuestackdepth is ABSOLUTE (nlocals + stack_items).
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            let new_vsd = original_box_types.len() - NUM_SCALAR_INPUTARGS;
            if new_vsd < meta.valuestackdepth && meta.slot_types.len() > new_vsd {
                meta.slot_types.truncate(new_vsd);
            } else if new_vsd > meta.valuestackdepth && meta.slot_types.len() < new_vsd {
                meta.slot_types.resize(new_vsd, Type::Ref);
            }
            meta.valuestackdepth = new_vsd;
        }
    }

    fn build_meta_from_merge_point(
        provisional: &PyreMeta,
        _header_pc: usize,
        original_box_types: &[Type],
    ) -> PyreMeta {
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // RPython parity: Python locals/stack are always Ref.
        let slot_types = if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            vec![Type::Ref; original_box_types.len() - NUM_SCALAR_INPUTARGS]
        } else {
            Vec::new()
        };
        let vsd = if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            original_box_types.len() - NUM_SCALAR_INPUTARGS
        } else {
            provisional.valuestackdepth
        };
        PyreMeta {
            num_locals: provisional.num_locals,
            ns_len: provisional.ns_len,
            valuestackdepth: vsd,
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
        let Some(frame) = values.first() else {
            return;
        };
        self.frame = value_to_usize(frame);
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
        if values.len() == 1 {
            return;
        }

        if meta.has_virtualizable {
            // next_instr is already synced to the PyFrame heap by the
            // compiled code's virtualizable sync before JUMP.
            self.set_valuestackdepth(meta.valuestackdepth);
            let nlocals = self.local_count();
            let stack_only = meta.valuestackdepth.saturating_sub(nlocals);
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
            let mut idx = 1;
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

        // virtualizable.py:126-137 write_from_resume_data_partial:
        // ALL static fields in unroll_static_fields order.
        if let Some(last_instr) = values
            .get(crate::virtualizable_gen::SYM_LAST_INSTR_IDX as usize)
            .map(value_to_usize)
        {
            self.set_last_instr(last_instr);
        }
        if let Some(code) = values
            .get(crate::virtualizable_gen::SYM_PYCODE_IDX as usize)
            .map(value_to_usize)
        {
            self.set_pycode(code);
        }
        if let Some(vsd) = values
            .get(crate::virtualizable_gen::SYM_VALUESTACKDEPTH_IDX as usize)
            .map(value_to_usize)
        {
            // Sanity check: vsd must not exceed the frame's total capacity
            // (nlocals + stacksize). A bad vsd from stale guard recovery
            // values can corrupt the frame and crash in as_mut_slice.
            let max_vsd = self
                .locals_cells_stack_array()
                .map(|arr| arr.len())
                .unwrap_or(0);
            let safe_vsd = vsd.min(max_vsd);
            self.set_valuestackdepth(safe_vsd);
        }
        if let Some(ns) = values
            .get(crate::virtualizable_gen::SYM_W_GLOBALS_IDX as usize)
            .map(value_to_usize)
        {
            self.set_w_globals(ns);
        }

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // resume.py:1077 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
        // RPython's consume_boxes uses position_info (liveness at resume PC)
        // to map compact active_boxes back to register indices. Dead registers
        // are skipped in the compact array — only live registers advance the
        // value index.
        //
        // values[3..] = compact active_boxes from get_list_of_active_boxes,
        // which filters by liveness. Use the same liveness table to restore.
        // Two distinct pointers for the same PyFrame.pycode:
        //   - `w_code_ptr`  : Python CodeObject WRAPPER (`PyObjectRef`).
        //                     Key used by `jitcode_for`/`jitcode_lookup`
        //                     (see trace_opcode.rs:3501/:3580 and
        //                     state.rs:1908 — all pass the wrapper).
        //   - `raw_code_ptr`: unwrapped Rust `CodeObject`. Consumed by
        //                     `liveness_for` (the LiveVars cache key).
        let (w_code_ptr, raw_code_ptr) = if self.frame != 0 {
            let w_code = unsafe {
                *((self.frame as *const u8).add(crate::frame_layout::PYFRAME_PYCODE_OFFSET)
                    as *const *const ())
            };
            if !w_code.is_null() {
                let raw = unsafe {
                    pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject
                };
                (w_code, raw)
            } else {
                (std::ptr::null(), std::ptr::null())
            }
        } else {
            (std::ptr::null(), std::ptr::null())
        };
        // resume.py:1383: info = blackholeinterp.get_current_position_info()
        // blackhole.py:337: position was set by setposition(jitcode, pc) where
        // pc comes from rd_numb — the same orgpc used by get_list_of_active_boxes.
        // next_instr = orgpc + 1 + caches, which may have different liveness.
        let live_pc = self.resume_pc.take().unwrap_or_else(|| self.next_instr());
        let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // resume.py:1077 consume_boxes parity — iterate the SAME
        // `all_liveness` BC_LIVE data the encoder used in
        // `trace_opcode.rs::get_list_of_active_boxes`. For every live
        // reg idx, consume one tagged `values[idx]` and route to the
        // PyFrame slot:
        //   - `reg < nlocals` → `set_local_at(reg, …)`
        //   - `nlocals <= reg < nlocals + stack_only` → `set_stack_at`
        // This keeps the encoder/decoder contract aligned on jitcode
        // SSA liveness rather than the LiveVars approximation.
        //
        // Falls back to the LiveVars path below only when the frame's
        // code is not (yet) registered or has no pc_map entry — i.e.
        // the same fallback branch trace_opcode.rs:313-334 takes.
        let decoded_via_jitcode: bool = (|| {
            if w_code_ptr.is_null() {
                return false;
            }
            let jc_ptr = jitcode_lookup(w_code_ptr);
            if jc_ptr.is_null() {
                return false;
            }
            let jc = unsafe { &*jc_ptr };
            let payload = &jc.payload;
            let Some(&jit_pc) = payload.metadata.pc_map.get(live_pc) else {
                return false;
            };
            let all_liveness = liveness_info_snapshot();
            let off = payload.jitcode.get_live_vars_info(jit_pc, op_live());
            if off + 2 >= all_liveness.len() {
                return false;
            }
            let length_i = all_liveness[off] as u32;
            let length_r = all_liveness[off + 1] as u32;
            let length_f = all_liveness[off + 2] as u32;
            let mut cursor = off + 3;
            use majit_translate::liveness::LivenessIterator;
            for length in [length_i, length_r, length_f] {
                if length == 0 {
                    continue;
                }
                let mut it = LivenessIterator::new(cursor, length, &all_liveness);
                while let Some(reg_idx) = it.next() {
                    let reg = reg_idx as usize;
                    if let Some(value) = values.get(idx) {
                        let boxed = virtualizable_box_value(value);
                        if reg < nlocals {
                            let _ = self.set_local_at(reg, boxed);
                        } else {
                            let stack_idx = reg - nlocals;
                            if stack_idx < stack_only {
                                let _ = self.set_stack_at(stack_idx, boxed);
                            }
                        }
                    }
                    idx += 1;
                }
                cursor = it.offset;
            }
            true
        })();
        if !decoded_via_jitcode {
            // Fallback: LiveVars-based decode (skeleton jitcode / no
            // pc_map entry / unregistered code). Retained for RPython
            // parity with `get_list_of_active_boxes` skeleton path.
            let live = if !raw_code_ptr.is_null() {
                Some(liveness_for(raw_code_ptr))
            } else {
                None
            };
            for local_idx in 0..nlocals {
                let is_live = live.map_or(true, |lv| lv.is_local_live(live_pc, local_idx));
                if is_live {
                    if let Some(value) = values.get(idx) {
                        let boxed = virtualizable_box_value(value);
                        let _ = self.set_local_at(local_idx, boxed);
                    }
                    idx += 1;
                }
            }
            for stack_idx in 0..stack_only {
                let is_live = live.map_or(true, |lv| lv.is_stack_live(live_pc, stack_idx));
                if is_live {
                    if let Some(value) = values.get(idx) {
                        let boxed = virtualizable_box_value(value);
                        let _ = self.set_stack_at(stack_idx, boxed);
                    }
                    idx += 1;
                }
            }
        }

        // Clear stale slots beyond valuestackdepth (blackhole fresh frame parity).
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
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        crate::virtualizable_gen::virt_array_lengths(self, _virtualizable, _info)
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
        sym.vable_collect_jump_args()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        sym.vable_collect_typed_jump_args()
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
            MaterializedVirtual::Struct { fields, .. }
            | MaterializedVirtual::Obj { fields, .. } => {
                self.materialize_virtual_object(fields, materialized_refs)
            }
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

    /// resume.py:597-602 AbstractVirtualStructInfo.setfields parity:
    /// ALL traced fields are restored, including w_class.
    fn materialize_virtual_object(
        &mut self,
        fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        let mut ob_type: usize = 0;
        let mut w_class: usize = 0;
        let mut int_payload: i64 = 0;
        let mut list_items_ptr: usize = 0;
        let mut list_items_len: usize = 0;

        use pyre_object::pyobject::{OB_TYPE_OFFSET, W_CLASS_OFFSET};
        // PyObject layout: [ob_type @ 0, w_class @ 8]
        // Payload fields start at sizeof(PyObject) = 16.
        const PAYLOAD_0: usize = std::mem::size_of::<pyre_object::pyobject::PyObject>();
        const PAYLOAD_1: usize = PAYLOAD_0 + 8;

        for (field_idx, value) in fields {
            let offset = extract_pyre_field_offset(*field_idx);
            let concrete = value.resolve_with_refs(materialized_refs)?;
            match offset {
                Some(o) if o == OB_TYPE_OFFSET => ob_type = concrete as usize,
                // resume.py:598-602: restore ALL fields, including w_class.
                Some(o) if o == W_CLASS_OFFSET => w_class = concrete as usize,
                Some(o) if o == PAYLOAD_0 => {
                    if ob_type == &LIST_TYPE as *const _ as usize {
                        list_items_ptr = concrete as usize;
                    } else {
                        int_payload = concrete;
                    }
                }
                Some(o) if o == PAYLOAD_1 && ob_type == &LIST_TYPE as *const _ as usize => {
                    list_items_len = concrete as usize;
                }
                _ => {}
            }
        }

        let ptr = if ob_type == &INT_TYPE as *const _ as usize {
            pyre_object::intobject::w_int_new(int_payload)
        } else if ob_type == &FLOAT_TYPE as *const _ as usize {
            pyre_object::floatobject::w_float_new(f64::from_bits(int_payload as u64))
        } else if ob_type == &LIST_TYPE as *const _ as usize {
            let items = if list_items_len == 0 {
                Vec::new()
            } else {
                if list_items_ptr == 0 {
                    return None;
                }
                unsafe {
                    std::slice::from_raw_parts(list_items_ptr as *const PyObjectRef, list_items_len)
                        .to_vec()
                }
            };
            w_list_new(items)
        } else {
            return None;
        };

        // resume.py:602 setfields parity: restore traced w_class if recorded.
        // Without this, builtin subclasses or __class__-mutated objects would
        // revert to the default builtin type after guard failure.
        if w_class != 0 {
            unsafe {
                (*(ptr as *mut pyre_object::pyobject::PyObject)).w_class =
                    w_class as *mut pyre_object::pyobject::PyObject;
            }
        }

        Some(majit_ir::GcRef(ptr as usize))
    }
}

fn materialize_virtual_raw_buffer(
    func: i64,
    size: usize,
    offsets: &[usize],
    descrs: &[majit_ir::ArrayDescrInfo],
    values: &[majit_metainterp::resume::MaterializedValue],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    assert_eq!(offsets.len(), descrs.len());
    assert_eq!(offsets.len(), values.len());

    // resume.py:703: buffer = decoder.allocate_raw_buffer(func, size)
    let (driver, _) = crate::driver::driver_pair();
    let calldescr = majit_translate::jitcode::BhCallDescr {
        arg_classes: "i".into(),
        result_type: 'i',
    };
    let buffer = driver.meta_interp().backend().bh_call_i(
        func,
        Some(&[size as i64]),
        None,
        None,
        &calldescr,
    );
    if buffer == 0 {
        return None;
    }

    let backend = driver.meta_interp().backend();
    // resume.py:705-708: per-item bh_raw_store_i/f
    for i in 0..offsets.len() {
        let concrete = values[i].resolve_with_refs(materialized_refs)?;
        let di = &descrs[i];
        let bh_descr = majit_translate::jitcode::BhDescr::from_array_descr_info(di);
        // resume.py:1544: assert not descr.is_array_of_pointers()
        assert!(
            !bh_descr.is_array_of_pointers(),
            "raw buffer entry must not be pointer type"
        );
        let offset = offsets[i] as i64;
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

/// Extract byte offset from a pyre field descriptor index.
/// Mirrors `extract_field_offset` in majit-opt/virtualize.rs.
fn extract_pyre_field_offset(descr_idx: u32) -> Option<usize> {
    const FIELD_DESCR_TAG: u32 = 0x1000_0000;
    if descr_idx & 0xF000_0000 != FIELD_DESCR_TAG {
        return None;
    }
    Some(((descr_idx >> 4) & 0x000f_ffff) as usize)
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
    use crate::descr::{list_float_items_len_descr, list_int_items_len_descr};
    use crate::helpers::TraceHelperAccess;
    use majit_metainterp::JitState;
    use majit_metainterp::resume::{MaterializedValue, MaterializedVirtual};
    use pyre_interpreter::bytecode::BinaryOperator;
    use pyre_interpreter::eval::eval_frame_plain;
    use pyre_interpreter::{
        BranchOpcodeHandler, IterOpcodeHandler, LocalOpcodeHandler, OpcodeStepExecutor,
        SharedOpcodeHandler, compile_exec,
    };
    use pyre_object::OB_TYPE_OFFSET;
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::{
        w_list_can_append_without_realloc, w_list_getitem, w_list_uses_float_storage,
        w_list_uses_int_storage,
    };
    use pyre_object::pyobject::{PyType, is_list};
    use std::cell::{Cell, UnsafeCell};
    use std::rc::Rc;

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
            }));
            crate::callbacks::init(cb);
        });
    }

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            num_locals: 0,
            ns_len: 0,
            valuestackdepth: 0,
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
    fn test_guard_class_uses_guard_nonnull_class() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.guard_class(ctx, obj, &INT_TYPE as *const PyType);
        });

        let recorder = ctx.into_recorder();
        let op = recorder.ops().last().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardNonnullClass);
        assert_eq!(op.args[0], obj);
    }

    #[test]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload() {
        let mut ctx = TraceCtx::for_test(1);
        let int_obj = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![int_obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let _ = state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj));

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::GuardNonnullClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI && op.args.as_slice() == &[int_obj] {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardNonnullClass"
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI"
        );
    }

    #[test]
    fn test_load_method_accepts_plain_python_instance_method() {
        let code = compile_exec("class C:\n    def f(self):\n        return self\nc = C()\n")
            .expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new_with_context(
            code,
            Rc::new(pyre_interpreter::PyExecutionContext::default()),
        );
        eval_frame_plain(&mut frame).expect("class body should execute");
        let instance = unsafe {
            (*frame.fget_w_globals())
                .get("c")
                .copied()
                .expect("namespace should contain c")
        };

        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: (&mut frame as *mut pyre_interpreter::PyFrame) as usize,
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

        assert!(receiver.concrete.to_pyobj().is_null());
        unsafe {
            assert!(pyre_object::is_method(callable.concrete.to_pyobj()));
        }
    }

    #[test]
    fn test_init_symbolic_skips_heap_array_read_for_standard_virtualizable() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.init_vable_indices();

        sym.init_symbolic(&mut ctx, frame_ptr);

        assert_eq!(sym.locals_cells_stack_array_ref, OpRef::NONE);
        let recorder = ctx.into_recorder();
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
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
        let materialized = MaterializedVirtual::Obj {
            type_id: 0,
            descr_index: crate::descr::w_float_size_descr().index(),
            fields: vec![
                (
                    crate::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&FLOAT_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::descr::float_floatval_descr().index(),
                    MaterializedValue::Value(value.to_bits() as i64),
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
        .expect("float virtual should materialize");

        unsafe {
            assert!(is_float(ptr.0 as PyObjectRef));
            assert_eq!(w_float_get_value(ptr.0 as PyObjectRef), value);
        }
    }

    // Needs a real raw-buffer allocator (`func` passed as a valid function
    // pointer into `bh_call_i`). The current test supplies `func: 0`, so
    // `materialize_virtual_raw_buffer` bails out at the `buffer == 0`
    // guard. Previously the test tripped on an uninitialised
    // `CallJitCallbacks` before ever reaching that guard; wiring up the
    // backend allocator is a follow-up.
    #[ignore]
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
                },
                majit_ir::ArrayDescrInfo {
                    index: 0,
                    base_size: 0,
                    item_size: 8,
                    item_type: 1,
                    is_signed: false,
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
            type_id: 0,
            descr_index: 0,
            fields: vec![
                (
                    crate::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&LIST_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::descr::list_items_ptr_descr().index(),
                    MaterializedValue::VirtualRef(0),
                ),
                (
                    crate::descr::list_items_len_descr().index(),
                    MaterializedValue::Value(2),
                ),
                (
                    crate::descr::list_items_heap_cap_descr().index(),
                    MaterializedValue::Value(2),
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
        let mut frame = Box::new(PyFrame::new(code));
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
    fn test_restore_guard_failure_uses_runtime_value_kinds_for_virtualizable_locals() {
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

        let mut frame = Box::new(PyFrame::new(code));
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
            valuestackdepth: 4,
            has_virtualizable: true,
            // Stale trace-entry types: all locals looked boxed refs when the
            // trace started, but guard failure can still carry a raw int in
            // slot `i`.
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),             // frame
            Value::Int(8),                            // last_instr
            Value::Ref(GcRef(frame.pycode as usize)), // pycode
            Value::Int(4),                            // valuestackdepth
            Value::Ref(GcRef(0)),                     // debugdata
            Value::Ref(GcRef(0)),                     // lastblock
            Value::Ref(GcRef(0)),                     // w_globals
            Value::Ref(GcRef(w_int_new(1) as usize)), // local a
            Value::Ref(GcRef(w_int_new(2) as usize)), // local b
            Value::Ref(GcRef(w_int_new(3) as usize)), // local c
            Value::Int(7),                            // local i
        ];

        assert!(<PyreJitState as JitState>::restore_guard_failure_values(
            &mut state,
            &meta,
            &values,
            &majit_metainterp::blackhole::ExceptionState::default(),
        ));

        assert_eq!(state.next_instr(), 9);
        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    fn test_load_local_checked_value_respects_symbolic_local_type() {
        let run_case = |symbolic_type: Type, name: &str, expected_guard: Option<OpCode>| {
            let mut ctx = TraceCtx::for_test(1);
            let local = OpRef(0);
            let mut sym = PyreSym::new_uninit(OpRef::NONE);
            sym.registers_r = vec![local];
            sym.symbolic_local_types = vec![symbolic_type];
            sym.nlocals = 1;

            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                parent_frames: Vec::new(),
                pending_inline_frame: None,
                orgpc: 0,
                concrete_frame_addr: 0,
            };

            let loaded =
                <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, name)
                    .expect("local should load");
            assert_eq!(loaded.opref, local);

            let recorder = ctx.into_recorder();
            assert_eq!(recorder.ops().last().map(|op| op.opcode), expected_guard);
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
        let mut frame = Box::new(PyFrame::new(code));
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
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state
            .with_ctx(|this, ctx| this.store_local_value(ctx, 0, ref_value))
            .expect("store of pre-wrapped Ref should succeed");
        assert_eq!(
            state.sym().registers_r[0],
            ref_value,
            "Ref value must be stored as-is, not reboxed",
        );
        assert_eq!(state.sym().symbolic_local_types[0], Type::Ref);
    }

    #[test]
    fn test_trace_dynamic_list_index_typed_int_skips_object_unbox() {
        let mut ctx = TraceCtx::for_test(2);
        let key = OpRef(0);
        let len = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![key];
        sym.symbolic_local_types = vec![Type::Int];
        sym.nlocals = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let raw_index = state.with_ctx(|this, ctx| this.trace_dynamic_list_index(ctx, key, len, 2));
        assert_eq!(raw_index, key);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(recorder.num_guards(), 2);
    }

    #[test]
    fn test_trace_binary_value_boxes_typed_raw_operands_for_python_helper() {
        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![lhs, rhs];
        sym.symbolic_local_types = vec![Type::Float, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let _ = <MIFrame as TraceHelperAccess>::trace_binary_value(
            &mut state,
            lhs,
            rhs,
            BinaryOperator::Power,
        )
        .expect("generic helper call should box raw operands first");

        let recorder = ctx.into_recorder();
        let call = recorder.ops().last().expect("call op should be present");
        assert!(matches!(
            call.opcode,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        ));
        assert_ne!(call.args[0], lhs);
        assert_ne!(call.args[1], rhs);
    }

    #[test]
    fn test_trace_known_builtin_call_boxes_typed_raw_args_for_python_helper_boundary() {
        let mut ctx = TraceCtx::for_test(2);
        let callable = OpRef(0);
        let arg = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![callable, arg];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let _ = state
            .trace_known_builtin_call(callable, &[arg])
            .expect("known builtin helper boundary should box raw int args");

        let recorder = ctx.into_recorder();
        let call = recorder.ops().last().expect("call op should be present");
        assert!(matches!(
            call.opcode,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        ));
        assert_ne!(call.args.last().copied(), Some(arg));
    }

    #[test]
    fn test_compare_value_direct_emits_raw_truth_for_immediate_branch_consumer() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("if 1 < 2:\n    x = 3\n").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
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

        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: branch_pc,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
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
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
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
            OpRef(2),
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
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
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
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
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
        let mut frame = Box::new(PyFrame::new(code.clone()));
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
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        assert!(
            state.next_instruction_consumes_comparison_truth(),
            "branch fusion should survive EXTENDED_ARG/other trivia before the branch"
        );
    }

    // test_trace_code_step_preserves_comparison_truth_across_extended_arg_trivia
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

    #[test]
    fn test_branch_guard_preserves_pre_pop_stack_shape() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.set_last_instr_from_next_instr(456);
        frame.fix_array_ptrs();
        let mut ctx = TraceCtx::for_test(3);
        let frame_ref = OpRef(0);
        let lower_stack = OpRef(1);
        let truth = OpRef(2);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.nlocals() + 1;
        sym.registers_r = vec![OpRef::NONE; sym.nlocals];
        sym.registers_r.push(lower_stack);
        sym.symbolic_local_types = vec![Type::Ref; sym.nlocals];
        sym.symbolic_stack_types = vec![Type::Ref];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 459,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.record_branch_guard(ctx, OpRef::NONE, truth, true, this.fallthrough_pc);
        });

        let recorder = ctx.into_recorder();
        let guard = recorder
            .ops()
            .last()
            .expect("branch guard should be recorded");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("branch guard should carry explicit fail args");
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        // fail_args layout: [frame, {scalar header fields...}, stack slots...]
        // Scalar header has NUM_SCALAR_INPUTARGS - 1 entries; total scalar
        // count including frame is NUM_SCALAR_INPUTARGS. First box slot is
        // at index NUM_SCALAR_INPUTARGS.
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        assert!(fail_args.len() >= n + 1);
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[n + sym.nlocals], lower_stack);

        let mut frame = Box::new(PyFrame::new(code));
        frame.set_last_instr_from_next_instr(456);
        frame.fix_array_ptrs();
        let mut ctx = TraceCtx::for_test(3);
        let frame_ref = OpRef(0);
        let lower_stack = OpRef(1);
        let truth = OpRef(2);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.nlocals() + 1;
        sym.registers_r = vec![OpRef::NONE; sym.nlocals];
        sym.registers_r.push(lower_stack);
        sym.symbolic_local_types = vec![Type::Ref; sym.nlocals];
        sym.symbolic_stack_types = vec![Type::Ref];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 459,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.generate_guard(ctx, OpCode::GuardTrue, &[truth]);
        });

        let recorder = ctx.into_recorder();
        let guard = recorder
            .ops()
            .last()
            .expect("branch guard should be recorded");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("branch guard should carry explicit fail args");
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        // fail_args: [frame, ni, code, vsd, debugdata, lastblock, ns, lower_stack, ...]
        assert!(fail_args.len() >= 8);
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[7], lower_stack);
    }

    #[test]
    fn test_branch_truth_uses_concrete_parameter() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(1));
        frame.fix_array_ptrs();

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = OpRef(0);
        let truth = OpRef(1);
        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth;
        sym.registers_r = vec![OpRef::NONE; sym.nlocals];
        sym.registers_r.push(truth);
        sym.symbolic_local_types = vec![Type::Ref; sym.nlocals];
        sym.symbolic_stack_types = vec![Type::Int];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.generate_guard(ctx, OpCode::GuardTrue, &[truth]);
        });
        // concrete_branch_truth_for_value now takes concrete value as parameter
        assert_eq!(
            state
                .concrete_branch_truth_for_value(truth, w_int_new(1))
                .unwrap(),
            true
        );
        <MIFrame as BranchOpcodeHandler>::leave_branch_truth(&mut state).unwrap();
    }

    #[test]
    fn test_current_fail_args_flushes_virtualizable_header_before_capture() {
        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = OpRef(0);
        let local0 = OpRef(1);
        let stack0 = OpRef(2);
        let stack1 = OpRef(3);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_last_instr = ctx.const_int(12);
        sym.vable_pycode = ctx.const_ref(0);
        sym.vable_valuestackdepth = ctx.const_int(1);
        sym.vable_debugdata = ctx.const_ref(0);
        sym.vable_lastblock = ctx.const_ref(0);
        sym.vable_w_globals = ctx.const_ref(0);
        sym.registers_r = vec![local0, stack0, stack1];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let fail_args = state.with_ctx(|this, ctx| this.current_fail_args(ctx));

        // fail_args layout: [frame, {scalar header fields...}, locals..., stack...]
        // virtualizable.py:86 read_boxes: all static fields in order.
        // Scalar header = NUM_SCALAR_INPUTARGS - 1 (excludes frame) fields:
        //   last_instr, pycode, valuestackdepth, debugdata, lastblock, w_globals.
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        assert_eq!(fail_args.len(), n + 3);
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[n], local0);
        assert_eq!(fail_args[n + 1], stack0);
        assert_eq!(fail_args[n + 2], stack1);
    }

    #[test]
    fn test_current_fail_args_materializes_symbolic_holes_from_concrete_frame() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        let local_ref = w_int_new(41);
        let stack_int = w_int_new(7);
        frame.locals_w_mut()[0] = local_ref;
        frame.locals_w_mut()[1] = stack_int;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let frame_ref = OpRef(0);
        let code_ref = frame.pycode;
        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.nlocals = 1;
        sym.valuestackdepth = 2;
        sym.vable_last_instr = ctx.const_int(33);
        sym.vable_pycode = ctx.const_ref(0);
        sym.vable_valuestackdepth = ctx.const_int(0);
        sym.vable_debugdata = ctx.const_ref(0);
        sym.vable_lastblock = ctx.const_ref(0);
        sym.vable_w_globals = ctx.const_ref(0);
        sym.registers_r = vec![OpRef::NONE, OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Int];
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: frame_ptr,
        };

        let fail_args = state.with_ctx(|this, ctx| this.current_fail_args(ctx));

        // fail_args: [frame, pc_const, vsd_const, live_slots...]
        // Liveness-based: only slots live at orgpc are included.
        assert!(
            fail_args.len() >= crate::virtualizable_gen::NUM_SCALAR_INPUTARGS,
            "must have frame + ni + code + vsd + ns header"
        );
        assert_eq!(fail_args[0], frame_ref);
        assert!(
            fail_args.iter().all(|arg| !arg.is_none()),
            "materialized fail args should not contain OpRef::NONE holes"
        );
    }

    #[test]
    fn test_close_loop_args_at_target_pc_preserves_virtualizable_stack() {
        // PyPy reached_loop_header carries self.virtualizable_boxes[:-1]
        // into the JUMP unchanged. virtualizable.py:86-98 read_boxes
        // reads the full locals_cells_stack_w array, so close_loop_args_at
        // must preserve stack slots instead of truncating to nlocals.
        ensure_test_callbacks();
        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(0x1000);
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
        sym.registers_r = vec![local0, stack0, stack1];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Null, ConcreteValue::Null];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args_at(ctx, Some(77)));

        assert_eq!(
            jump_args.len(),
            crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + 3,
            "JUMP carries local and stack slots from the virtualizable array"
        );
        assert_eq!(state.sym().valuestackdepth, 3);
        let nlocals = state.sym().nlocals;
        let stack_only = state.sym().valuestackdepth - nlocals;
        assert_eq!(state.sym().registers_r.len(), nlocals + stack_only);
        assert!(
            state.sym().registers_r[nlocals..]
                .iter()
                .all(|opref| !opref.is_none())
        );
    }

    #[test]
    fn test_direct_len_value_returns_typed_raw_len_for_integer_list() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
        }
        let arg_idx = frame.stack_base() + 2;
        frame.locals_w_mut()[arg_idx] = list;
        frame.fix_array_ptrs();
        let mut ctx = TraceCtx::for_test(2);
        let value = OpRef(0);
        let callable = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![value, callable];
        sym.symbolic_local_types = vec![Type::Ref, Type::Ref];
        sym.nlocals = 2;
        sym.valuestackdepth = frame.stack_base();

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let len = state
            .direct_len_value(callable, value, list)
            .expect("integer-list len fast path should trace");
        let len_type = state.value_type(len);
        assert_eq!(
            len_type,
            Type::Int,
            "len fast path should produce an Int-typed result"
        );

        let recorder = ctx.into_recorder();
        assert_ne!(
            recorder.ops().last().map(|op| op.opcode),
            Some(OpCode::CallI),
            "len(list) should not fall back to the builtin helper call path"
        );
        let mut saw_len_field = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op.descr.as_ref().map(|d| d.index()) == Some(list_int_items_len_descr().index())
            {
                saw_len_field = true;
            }
        }
        assert_eq!(len_type, Type::Int);
        assert!(
            saw_len_field,
            "list len fast path should read the integer-list length field"
        );
        assert!(
            !saw_new,
            "len(list) should not allocate a W_Int box in the trace"
        );
    }

    #[test]
    fn test_trace_direct_float_list_getitem_uses_gc_field_loads_for_list_object() {
        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let key = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.registers_r = vec![list, key];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let result = state.with_ctx(|this, ctx| {
            crate::generated_list_getitem_by_strategy(this, ctx, list, key, 2, 2)
        });
        assert_eq!(state.value_type(result), Type::Float);

        let recorder = ctx.into_recorder();
        let mut saw_gc_field = false;
        let mut saw_raw_field = false;
        let mut saw_raw_array = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI => saw_gc_field = true,
                OpCode::GetfieldRawI => saw_raw_field = true,
                OpCode::GetarrayitemRawF => saw_raw_array = true,
                _ => {}
            }
        }
        assert!(saw_gc_field, "list object fields should use GetfieldGcI");
        assert!(
            !saw_raw_field,
            "float-list fast path should not use raw field loads on GC list objects"
        );
        assert!(saw_raw_array, "payload array access stays raw");
    }

    #[test]
    fn test_list_append_value_uses_raw_storage_fast_paths() {
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let run_case = |concrete_list: pyre_object::PyObjectRef,
                        symbolic_value_type: Type,
                        concrete_value: pyre_object::PyObjectRef,
                        expected_array_descr_idx: u32,
                        expected_len_descr_idx: u32,
                        case_name: &str,
                        expect_box_alloc: bool| {
            let mut ctx = TraceCtx::for_test_types(&[Type::Ref, symbolic_value_type]);
            let list = OpRef(0);
            let value = OpRef(1);
            let mut frame = Box::new(PyFrame::new(code.clone()));
            frame.fix_array_ptrs();

            let mut sym = PyreSym::new_uninit(OpRef::NONE);
            sym.registers_r = vec![list, value];
            sym.symbolic_local_types = vec![Type::Ref, symbolic_value_type];
            sym.nlocals = 2;

            let mut state = MIFrame {
                ctx: &mut ctx,
                sym: &mut sym,
                fallthrough_pc: 0,
                parent_frames: Vec::new(),
                pending_inline_frame: None,
                orgpc: 0,
                concrete_frame_addr: 0,
            };

            state
                .list_append_value(list, value, concrete_list, concrete_value)
                .expect("raw-storage append fast path should trace");

            let recorder = ctx.into_recorder();
            let mut saw_raw_setitem = false;
            let mut saw_len_update = false;
            let mut saw_call = false;
            let mut saw_new = false;
            for pos in 2..(2 + recorder.num_ops() as u32) {
                let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                    continue;
                };
                if matches!(
                    op.opcode,
                    OpCode::CallI | OpCode::CallN | OpCode::CallR | OpCode::CallF
                ) {
                    saw_call = true;
                }
                if op.opcode == OpCode::New {
                    saw_new = true;
                }
                if op.opcode == OpCode::SetarrayitemRaw
                    && op.descr.as_ref().map(|d| d.index()) == Some(expected_array_descr_idx)
                {
                    saw_raw_setitem = true;
                }
                if op.opcode == OpCode::SetfieldGc
                    && op.descr.as_ref().map(|d| d.index()) == Some(expected_len_descr_idx)
                {
                    saw_len_update = true;
                }
            }

            assert!(
                saw_raw_setitem,
                "{} append should write through the raw payload array",
                case_name
            );
            assert!(
                saw_len_update,
                "{} append should update the strategy length field",
                case_name
            );
            assert!(
                !saw_call,
                "{} append should not fall back to jit_list_append",
                case_name
            );
            assert_eq!(
                saw_new, expect_box_alloc,
                "{} append unexpected boxing allocation behavior",
                case_name
            );
        };

        let int_list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        unsafe {
            assert!(w_list_uses_int_storage(int_list));
            assert!(w_list_can_append_without_realloc(int_list));
        }
        run_case(
            int_list,
            Type::Int,
            w_int_new(42),
            int_array_descr().index(),
            list_int_items_len_descr().index(),
            "integer-list",
            false,
        );

        let float_list = w_list_new(vec![
            pyre_object::floatobject::w_float_new(1.5),
            pyre_object::floatobject::w_float_new(2.5),
        ]);
        unsafe {
            assert!(w_list_uses_float_storage(float_list));
            assert!(w_list_can_append_without_realloc(float_list));
        }
        run_case(
            float_list,
            Type::Float,
            pyre_object::w_float_new(3.14),
            float_array_descr().index(),
            list_float_items_len_descr().index(),
            "float-list",
            false,
        );
    }

    #[test]
    fn test_iter_next_value_for_range_iterator_uses_gc_fields_and_returns_raw_int() {
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_object::w_range_iter_new;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let range_iter = w_range_iter_new(0, 2, 1);
        frame.push(range_iter);
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let iter = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        // nlocals == 0 for this test; stack-only tail starts at registers_r[0].
        sym.registers_r = vec![iter];
        sym.symbolic_stack_types = vec![Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Ref(range_iter)];
        sym.valuestackdepth = 1;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let next = MIFrame::iter_next_value(&mut state, iter, range_iter)
            .expect("range iterator fast path should trace");
        assert_eq!(state.value_type(next.opref), Type::Int);
        <MIFrame as IterOpcodeHandler>::guard_optional_value(&mut state, next, true)
            .expect("typed range next should not need optional guard");

        let recorder = ctx.into_recorder();
        let mut saw_getfield_gc = false;
        let mut saw_setfield_gc = false;
        let mut saw_setfield_raw = false;
        let mut saw_getfield_raw = false;
        let mut saw_new = false;
        let mut saw_optional_guard = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI => saw_getfield_gc = true,
                OpCode::SetfieldGc => saw_setfield_gc = true,
                OpCode::SetfieldRaw => saw_setfield_raw = true,
                OpCode::GetfieldRawI => saw_getfield_raw = true,
                OpCode::New => saw_new = true,
                OpCode::GuardNonnull | OpCode::GuardIsnull => saw_optional_guard = true,
                _ => {}
            }
        }
        assert!(
            saw_getfield_gc,
            "range iterator fields should use GetfieldGcI"
        );
        assert!(
            saw_setfield_gc,
            "range iterator current update should use SetfieldGc"
        );
        assert!(
            !saw_setfield_raw,
            "range iterator fast path should not use raw stores on GC iterator objects"
        );
        assert!(
            !saw_getfield_raw,
            "range iterator fast path should not use raw field loads on GC iterator objects"
        );
        assert!(
            !saw_new,
            "range iterator fast path should not box the yielded int"
        );
        assert!(
            !saw_optional_guard,
            "raw-int range iteration should not emit optional-value guards"
        );
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

/// PRE-EXISTING-ADAPTATION: deferred `MetaInterp.perform_call`
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
    /// Each: (fail_args, types, resumepc, jitcode_index).
    pub parent_frames: Vec<(Vec<OpRef>, Vec<Type>, usize, i32)>,
    pub nargs: usize,
    pub caller_result_stack_idx: Option<usize>,
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

// inline_trace_and_execute removed — replaced by PyreMetaInterp.interpret()
// which uses a single framestack for both root and inline frames.
// trace_through_callee removed — replaced by build_pending_inline_frame +
// MetaInterp.push_inline_frame (RPython perform_call parity).

/// listobject.rs:241-249 parity: int strategy only preserves identity for
/// canonical cached ints. Unique small ints (from w_int_new_unique) trigger
/// de-specialization to object strategy.
///
/// For large ints (outside small cache range), the strategy always keeps them
/// as raw i64 values regardless of pointer identity.
pub unsafe fn int_strategy_preserves_identity(value: pyre_object::PyObjectRef) -> bool {
    unsafe {
        let v = pyre_object::w_int_get_value(value);
        if pyre_object::w_int_small_cached(v) {
            // Small cached range: only canonical pointer preserves int strategy.
            std::ptr::eq(value, pyre_object::w_int_new(v))
        } else {
            // Large ints are always stored as raw i64 in int strategy.
            true
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
    use super::MetaInterpStaticData;
    use majit_metainterp::jitcode::{JitCode, JitCodeBuilder};
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

    #[test]
    fn compiled_jitcode_lookup_ignores_skeleton_payload() {
        let mut sd = MetaInterpStaticData::new();
        let code = 0x1234usize as *const ();
        let skeleton = Arc::new(crate::PyJitCode::skeleton(None));
        let inserted = sd.jitcode_for(code, Some(skeleton));
        assert!(std::ptr::eq(inserted, sd.jitcodes[0].as_ref()));
        assert!(sd.compiled_jitcode_lookup(code).is_none());
    }

    #[test]
    fn compiled_jitcode_lookup_returns_populated_entry() {
        let mut sd = MetaInterpStaticData::new();
        let code = 0x5678usize as *const ();
        let mut pyjit = crate::PyJitCode::skeleton(None);
        pyjit.metadata.pc_map.push(0);
        let inserted = sd.jitcode_for(code, Some(Arc::new(pyjit)));
        let hit = sd
            .compiled_jitcode_lookup(code)
            .expect("populated payload should short-circuit callback");
        assert!(std::ptr::eq(inserted, hit));
    }

    /// codewriter.py:67-68 parity for the `by_code` hit path.  When a
    /// subsequent `jitcode_for(code, Some(new_arc))` call replaces a
    /// previously-registered payload (e.g. a `merge_point_pc`-driven
    /// recompile in CallControl), the new payload's shared
    /// `Arc<majit JitCode>` must carry the SAME `index` as the
    /// original — otherwise a snapshot whose `frame.jitcode` holds the
    /// new allocation would encode a stale index that no longer maps
    /// to the correct `metainterp_sd.jitcodes` slot.
    #[test]
    fn jitcode_for_hit_path_restamps_payload_index() {
        use std::sync::atomic::Ordering;
        let mut sd = MetaInterpStaticData::new();
        // Prime the SD with TWO distinct codes so the second allocation
        // lands at wrapper.index == 1.  This makes the regression test
        // sensitive to whether the hit-path re-stamp uses the actual
        // SD slot (1) rather than the inner JitCode's default (0).
        let code_a = 0xA000usize as *const ();
        let code_b = 0xB000usize as *const ();
        let _ = sd.jitcode_for(code_a, Some(Arc::new(crate::PyJitCode::skeleton(None))));
        let first = Arc::new(crate::PyJitCode::skeleton(None));
        let _ = sd.jitcode_for(code_b, Some(first.clone()));
        assert_eq!(first.jitcode.index.load(Ordering::Relaxed), 1);

        // Now swap in a second PyJitCode for the SAME code (hit path).
        // The replacement's inner JitCode starts with index=0 (the
        // default from JitCodeBuilder::finish); after jitcode_for, it
        // must carry index=1 — matching the SD slot it fills.
        let second = Arc::new(crate::PyJitCode::skeleton(None));
        assert_eq!(
            second.jitcode.index.load(Ordering::Relaxed),
            0,
            "fresh PyJitCode starts with default index=0"
        );
        let _ = sd.jitcode_for(code_b, Some(second.clone()));
        assert_eq!(
            second.jitcode.index.load(Ordering::Relaxed),
            1,
            "hit-path replacement payload must be re-stamped with the SD slot"
        );
        // The first PyJitCode is no longer the payload but still carries
        // its own index=1 — the stamp is idempotent, not a swap.
        assert_eq!(first.jitcode.index.load(Ordering::Relaxed), 1);
    }
}
