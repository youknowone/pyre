mod assembler;
mod embedded;

pub(crate) use assembler::scalar_size;
pub use assembler::{JitCodeBuilder, live_slots_for_state_field_jit};
pub use embedded::EmbeddedJitCodeTable;
pub use majit_translate::jitcode::{
    BhCallDescr as CanonicalBhCallDescr, BhDescr as CanonicalBhDescr, BhInteriorFieldSpec,
    JitCode as CanonicalJitCode,
};

// `BC_*` constants and `MAX_HOST_CALL_ARITY` live in
// `majit_translate::insns` (, slice #86c). The module is
// re-exported here as the canonical access path; in-crate and external
// consumers reach `BC_*` / `MAX_HOST_CALL_ARITY` via
// `jitcode::insns::BC_*`.
pub use majit_translate::insns;

/// Alias for `BC_JUMP`; used in dispatch JitCode loop-close tests
/// (7) and `jitcode_lower::lower_dispatch_body` jump emission.
pub const BC_GOTO: u8 = insns::BC_JUMP;

// `insn_byte` and `wellknown_bh_insns` were moved to
// `majit_translate::insns` in slice #86d. Re-exports keep
// internal callers (`jitcode::assembler::JitCodeBuilder`) and external
// callers (`pyre/pyre-jit/src/jit/assembler.rs`) resolving unchanged
// — the import-path sweep is slice #86e.
pub(crate) use majit_translate::insns::insn_byte;

pub use majit_translate::insns::{extension_insns, wellknown_bh_insns};

/// Re-export of the canonical `enumerate_vars` function so existing
/// metainterp callers can keep using `crate::jitcode::enumerate_vars`.
///
/// RPython places this function in `rpython/jit/codewriter/jitcode.py`,
/// not in metainterp. majit follows the same module placement: the
/// definition lives in `majit_translate::jitcode::enumerate_vars`.
pub use majit_translate::jitcode::enumerate_vars;

// Runtime descr pool types — RPython
// `BlackholeInterpBuilder.descrs` / `BlackholeInterpreter.descrs`
// (`blackhole.py:103`, `blackhole.py:288`).
//
// RPython keeps the descr pool on the blackhole interpreter, NOT on
// the JitCode object.  In majit the canonical
// `majit_translate::jitcode::JitCode` mirrors that — it is a
// source-only RPython parity type with no descrs field.  The runtime
// adapter state (descrs pool + call/assembler targets) lives here
// alongside the wrapper `JitCode` defined below, which carries
// `pub exec: JitCodeExecState` as a sibling of the canonical core.
//
// These types are runtime-only — they reference raw `*const ()`
// trampoline addresses and live `Arc<JitCode>` callee handles, neither
// of which has a representation in the codewriter source layer.

/// Trace-side function target descriptor for `BC_CALL_*` /
/// `BC_RESIDUAL_CALL_*`.  RPython `blackhole.py:1225-1256` reads the
/// callee function address from an int register (`i` argcode) and the
/// calling convention from a descr (`d` argcode); pyre bundles the
/// trace-side and concrete (non-JIT) function pointers into a single
/// descriptor slot because the runtime emitter wires both pointers
/// through one indirection.
///
/// `effect_info_slot` is the per-target analyzer-result classification
/// (`call.py getcalldescr`'s `extraeffect` selection without
/// the graph-based analyzer chain — see
/// [`crate::call_descr::EffectInfoSlot`]).  Callers that have a
/// resolved `JitCallTarget` thread the slot through
/// `make_call_descr_from_target_slot` so the recorded descr carries
/// the right `EffectInfo` instead of the `default_effect_info()`
/// fallback.  The default ([`crate::call_descr::EffectInfoSlot::CanRaise`]) preserves the
/// pre-G-2 behaviour for every existing construction site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallTarget {
    pub trace_ptr: *const (),
    pub concrete_ptr: *const (),
    pub effect_info_slot: crate::call_descr::EffectInfoSlot,
    /// Per-callee `save_err` decoration mirroring upstream
    /// `rffi.py call_external_function._call_aroundstate_target_ =
    /// (funcptr, save_err)`.  Read at descr-build time by
    /// `codewriter/call.py getcalldescr` to populate
    /// `EffectInfo.call_release_gil_target = (realfuncaddr, tgt_saveerr)`
    /// (`effectinfo.py:114, 197`).  `RFFI_ERR_NONE = 0` matches the
    /// `llexternal` default (`rffi.py`); release-gil callees that
    /// preserve `errno`, `winerror`, etc. carry one of the
    /// `RFFI_ERR_*` flags (`rffi.py:121-167`).
    pub save_err: i32,
}

impl JitCallTarget {
    pub fn new(trace_ptr: *const (), concrete_ptr: *const ()) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
            effect_info_slot: crate::call_descr::EffectInfoSlot::CanRaise,
            save_err: 0,
        }
    }

    /// Construct a target with an explicit
    /// [`crate::call_descr::EffectInfoSlot`] classification.  Used by
    /// the macro-time helper registration paths that statically know
    /// the callee's `_canraise` / `_elidable_function_` /
    /// `_jit_loop_invariant_` flags.
    pub fn with_effect_info_slot(
        trace_ptr: *const (),
        concrete_ptr: *const (),
        effect_info_slot: crate::call_descr::EffectInfoSlot,
    ) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
            effect_info_slot,
            save_err: 0,
        }
    }

    /// Construct a release-gil target carrying the wrapper callable's
    /// `_call_aroundstate_target_ = (funcptr, save_err)` decoration
    /// (`rffi.py:228`).  `effect_info_slot` is unused by release-gil
    /// dispatchers but kept for the dedup key triple.
    pub fn with_save_err(
        trace_ptr: *const (),
        concrete_ptr: *const (),
        effect_info_slot: crate::call_descr::EffectInfoSlot,
        save_err: i32,
    ) -> Self {
        Self {
            trace_ptr,
            concrete_ptr,
            effect_info_slot,
            save_err,
        }
    }
}

/// Compiled-loop target for `BC_CALL_ASSEMBLER_*`.  The `token_number`
/// names a `CompiledLoopToken` (RPython `compile.py
/// CompiledLoopToken.number`) that the tracer hands to
/// `ctx.call_assembler_*_typed`; `concrete_ptr` is the pointer the
/// blackhole interpreter calls when the trace bails out before the
/// loop is compiled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallAssemblerTarget {
    pub token_number: u64,
    pub concrete_ptr: *const (),
}

impl JitCallAssemblerTarget {
    pub fn new(token_number: u64, concrete_ptr: *const ()) -> Self {
        Self {
            token_number,
            concrete_ptr,
        }
    }
}

/// Per-arg kind tag for typed call argument streams.  Mirrors the
/// `i`/`r`/`f` register-bank chars RPython carries in
/// `BlackholeInterpBuilder.descrs` argcode bytes (`blackhole.py`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitArgKind {
    Int = 0,
    Ref = 1,
    Float = 2,
}

impl JitArgKind {
    pub fn encode(self) -> u8 {
        self as u8
    }

    pub fn decode(byte: u8) -> Self {
        match byte {
            0 => Self::Int,
            1 => Self::Ref,
            2 => Self::Float,
            other => panic!("unknown jitcode arg kind {other}"),
        }
    }

    /// Map a [`majit_ir::Type`] to its `JitArgKind`.  RPython encodes
    /// the same mapping inline in `_build_allboxes` per
    /// `pyjitpl.py:1969-1989` (`history.INT`/`history.REF`/`history.FLOAT`
    /// chars + `'S'` single-float / `'L'` long-long aliases).  Pyre's
    /// `Type::Void` has no JitArgKind because void calls carry no
    /// argbox.
    pub fn from_type(ty: majit_ir::Type) -> Option<Self> {
        match ty {
            majit_ir::Type::Int => Some(Self::Int),
            majit_ir::Type::Ref => Some(Self::Ref),
            majit_ir::Type::Float => Some(Self::Float),
            majit_ir::Type::Void => None,
        }
    }
}

/// Typed call argument: a register index plus its kind tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JitCallArg {
    pub kind: JitArgKind,
    pub reg: u16,
}

impl JitCallArg {
    pub fn int(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Int,
            reg,
        }
    }

    pub fn reference(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Ref,
            reg,
        }
    }

    pub fn float(reg: u16) -> Self {
        Self {
            kind: JitArgKind::Float,
            reg,
        }
    }
}

/// Runtime descriptor entry — heterogeneous pool element indexed by
/// `j` / `d` argcodes at dispatch time.  Equivalent of RPython
/// `self.descrs[idx]` where each entry is an instance of one of the
/// `AbstractDescr` subclasses (`FieldDescr`, `ArrayDescr`, `JitCode`,
/// ...).  RPython uses `isinstance(value, JitCode)` to discriminate at
/// runtime; pyre encodes the same discrimination in the enum tag.
#[derive(Clone, Debug)]
pub enum RuntimeBhDescr {
    /// Ordinary blackhole descriptor for a `d` argcode (`FieldDescr`,
    /// `ArrayDescr`, virtualizable descriptors, ...).  RPython keeps all
    /// of these in `BlackholeInterpBuilder.descrs`; pyre's runtime
    /// `JitCodeBuilder` uses the per-JitCode pool described below.
    Descr(Box<CanonicalBhDescr>),
    /// Target JitCode for a `j` argcode (`BC_INLINE_CALL`).  RPython:
    /// `blackhole.py:150-157` — `argtype == 'j' → descrs[idx]` asserted
    /// `isinstance(value, JitCode)`.
    JitCode(std::sync::Arc<JitCode>),
    /// The same `j` edge, non-owning: the back edge of a self- or mutually
    /// recursive `#[jit_inline]` helper, pointing at a jitcode that is still
    /// being assembled when the edge is recorded.
    ///
    /// `CallControl.get_jitcode` hands a recursive graph the shell it is
    /// already registered under, and CPython's collector reclaims the resulting
    /// cycle.  `Arc` has no collector, so the ownership classification stands in
    /// for one: the owning edge keeps the callee alive, the back edge does not,
    /// and a helper that names itself does not pin itself forever.
    ///
    /// Both variants name the same allocation, so everything downstream reads
    /// identically through [`Self::as_jitcode_owned`].
    JitCodeBackEdge(std::sync::Weak<JitCode>),
    /// Target function for `BC_CALL_*` / `BC_RESIDUAL_CALL_*`.
    /// RPython `blackhole.py:1225-1256` reads the function address
    /// from an int register (`i` argcode) and the calling convention
    /// from a descr (`d` argcode); pyre keeps the two together in
    /// `JitCallTarget` because the runtime emitter wires trace-side
    /// and blackhole-side function pointers in a single indirection
    /// slot.  Once pyre emits the function address via an int register
    /// this variant can split into the RPython-shaped pair.
    Call(Box<JitCallTarget>),
    /// Compiled-assembler target for `BC_CALL_ASSEMBLER_*`.  The
    /// `token_number` identifies a `CompiledLoopToken` (RPython
    /// `compile.py CompiledLoopToken.number`) that the tracer hands
    /// to `ctx.call_assembler_*_typed` so the metainterp can chain
    /// this trace into an already-compiled one.
    AssemblerToken(JitCallAssemblerTarget),
}

impl RuntimeBhDescr {
    /// Extract an ordinary blackhole descriptor for `d` argcodes.
    pub fn as_bh_descr(&self) -> Option<&CanonicalBhDescr> {
        match self {
            Self::Descr(descr) => Some(descr.as_ref()),
            _ => None,
        }
    }

    /// RPython parity: `isinstance(value, JitCode)` assertion at
    /// `blackhole.py:156`.  Returns the callee JitCode for `BC_INLINE_CALL`.
    ///
    /// **Owning edges only.**  A back edge answers `None` here, and callers
    /// that walk the pool to number or publish sub-jitcodes rely on that:
    /// `build_jitcode_registry` asserts each sub-jitcode it reaches is unnumbered,
    /// and a recursive helper's back edge names a jitcode that walk has already
    /// numbered.  Use [`Self::as_jitcode_owned`] where the question is "what
    /// does this `j` operand execute".
    pub fn as_jitcode(&self) -> Option<&std::sync::Arc<JitCode>> {
        match self {
            Self::JitCode(arc) => Some(arc),
            _ => None,
        }
    }

    /// The callee a `j` operand names, whichever kind of edge recorded it.
    ///
    /// `None` for a back edge whose callee is still under construction — the
    /// `Weak` cannot be upgraded from inside `Arc::new_cyclic`.  Every caller
    /// must treat that as "not answerable yet" and decline, never unwrap: it is
    /// a build-time window, not a missing target, and by the time anything
    /// executes the `j` operand the upgrade succeeds.
    pub fn as_jitcode_owned(&self) -> Option<std::sync::Arc<JitCode>> {
        match self {
            Self::JitCode(arc) => Some(std::sync::Arc::clone(arc)),
            Self::JitCodeBackEdge(weak) => weak.upgrade(),
            _ => None,
        }
    }

    /// Extract the `Call` target for `BC_CALL_*` / `BC_RESIDUAL_CALL_*`.
    pub fn as_call(&self) -> Option<&JitCallTarget> {
        match self {
            Self::Call(target) => Some(target),
            _ => None,
        }
    }

    /// Extract the assembler-call target for `BC_CALL_ASSEMBLER_*`.
    pub fn as_assembler_token(&self) -> Option<&JitCallAssemblerTarget> {
        match self {
            Self::AssemblerToken(target) => Some(target),
            _ => None,
        }
    }
}

/// What the `Assembler`'s helper cache holds for one `#[jit_inline]` helper.
///
/// Two states because a recursive helper is reachable while it is still being
/// assembled, and the answer differs: mid-assembly there is no `Arc` to hand
/// out yet, only the `Weak` that `Arc::new_cyclic` supplies.
pub enum InlineJitCodeSlot {
    /// The body is being assembled right now.  Anything that reaches the helper
    /// from inside its own assembly gets this.
    UnderConstruction(std::sync::Weak<JitCode>),
    /// Assembly finished; later callers link to the finished jitcode directly.
    Finished(std::sync::Arc<JitCode>),
}

/// The edge an inline call site should record for a helper.
pub enum InlineJitCodeRef {
    /// A normal call: the caller owns a reference to its callee.
    Strong(std::sync::Arc<JitCode>),
    /// A recursive call: the callee is (transitively) the caller, so an owning
    /// edge would be a cycle of `Arc`s that never drops.
    BackEdge(std::sync::Weak<JitCode>),
}

/// `call.py CallControl.get_jitcode`'s cache probe.
///
/// `None` means this helper has not been entered, and the caller must assemble
/// it — registering the shell with [`begin_inline_jitcode`] BEFORE the body, so
/// that a self-call arriving during assembly finds this probe answering.
pub fn lookup_inline_jitcode(asm: &crate::Assembler, key: usize) -> Option<InlineJitCodeRef> {
    let slot = asm.inline_jitcode_slot(key)?;
    match (**slot).downcast_ref::<InlineJitCodeSlot>()? {
        InlineJitCodeSlot::UnderConstruction(weak) => {
            Some(InlineJitCodeRef::BackEdge(weak.clone()))
        }
        InlineJitCodeSlot::Finished(arc) => {
            Some(InlineJitCodeRef::Strong(std::sync::Arc::clone(arc)))
        }
    }
}

/// Register the helper's identity before its body exists.
///
/// This is the whole mechanism: `make_jitcodes` mints one JitCode per graph up
/// front and fills bodies afterwards, so the identity a recursive call links to
/// is available before there is anything to link.
pub fn begin_inline_jitcode(
    asm: &mut crate::Assembler,
    key: usize,
    weak: std::sync::Weak<JitCode>,
) {
    asm.inline_jitcode_insert(
        key,
        std::sync::Arc::new(InlineJitCodeSlot::UnderConstruction(weak)),
    );
}

/// Publish the finished jitcode, so later call sites take an owning edge.
pub fn finish_inline_jitcode(asm: &mut crate::Assembler, key: usize, arc: std::sync::Arc<JitCode>) {
    asm.inline_jitcode_insert(key, std::sync::Arc::new(InlineJitCodeSlot::Finished(arc)));
}

/// Runtime view of the process-global build-time descriptor pool.
pub trait RuntimeDescrTable: Sync {
    fn get(&self, index: usize) -> Option<&'static RuntimeBhDescr>;
    fn len(&self) -> usize;

    /// The build-time `all_jitcodes` list this pool's `j` operands index
    /// (`codewriter.py make_jitcodes`), positioned by `jitcode.index`.
    ///
    /// Empty by default: a host that decodes its jitcodes on demand has no
    /// such list to hand over, and nothing here requires one. A host that
    /// does return it is stating that those indices are already assigned, so
    /// a second numbering must continue above them rather than restart —
    /// `resume.py:1338-1340` indexes one list by a frame's `jitcode_pos`, and
    /// two numberings over the same slots make that lookup ambiguous.
    ///
    /// The default is therefore only sound while no `JitCode` this table hands
    /// out can reach a dispatch registry's numbering walk. Two properties keep
    /// a lazily-decoded table on that side of the line, and **both** are load
    /// bearing: a walk reads a jitcode's own `exec.descrs`, never this pool
    /// (which is consulted only through [`JitCode::descr_at`]'s fallback), and
    /// a build-time shell's `exec.descrs` is empty. A host that grows an
    /// inline-call policy puts `JitCode` entries into a *runtime-emitted*
    /// `exec.descrs`, where the walk does see them — such a host must
    /// implement this method, and one that does not is caught by the assert in
    /// `build_jitcode_registry` rather than silently misnumbered. Doing so also
    /// requires one `Arc` per jitcode index shared by every descr naming it:
    /// minting a fresh shell per descr breaks the identity
    /// `codewriter.py:80 all_jitcodes[jitcode.index] is jitcode` asserts.
    fn jitcodes(&self) -> &'static [std::sync::Arc<JitCode>] {
        &[]
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Process-global build-time descr pool — RPython's single shared
/// `Assembler.descrs` (`assembler.py`).  Runtime-emitted jitcodes keep a
/// per-`JitCode` `exec.descrs` pool (the lazy-emit adaptation described on
/// [`JitCodeExecState`]); build-time (LLBC-extracted) jitcodes instead carry
/// an empty per-jitcode pool and resolve their `d`/`j` argcodes through this
/// shared pool via [`JitCode::descr_at`].  Installed once by the embedding
/// crate (`pyre-jit-trace`) from its build-time `ALL_DESCRS` / `ALL_JITCODES`
/// tables; `majit-metainterp` cannot build it because those tables live above
/// it.
static GLOBAL_BUILD_DESCR_POOL: std::sync::OnceLock<&'static dyn RuntimeDescrTable> =
    std::sync::OnceLock::new();

/// Install the process-global build-time descr pool.  Idempotent: the first
/// call wins and later calls are ignored (the pool is a frozen build artifact,
/// identical across callers).  See `GLOBAL_BUILD_DESCR_POOL`.
///
pub fn init_global_build_descr_pool(table: &'static dyn RuntimeDescrTable) {
    let _ = GLOBAL_BUILD_DESCR_POOL.set(table);
}

/// The installed global build-time descr pool, or `None` if the embedding
/// crate has not installed one (e.g. a standalone metainterp unit test that
/// only exercises runtime-built jitcodes).
pub(crate) fn global_build_descr_pool() -> Option<&'static dyn RuntimeDescrTable> {
    GLOBAL_BUILD_DESCR_POOL.get().copied()
}

/// The build-time `all_jitcodes` the installed pool numbers against, or empty.
///
/// See [`RuntimeDescrTable::jitcodes`]: this is the prefix of the flat registry
/// that is already assigned, so any numbering done at run time starts above it.
pub(crate) fn global_build_jitcodes() -> &'static [std::sync::Arc<JitCode>] {
    global_build_descr_pool().map_or(&[], |pool| pool.jitcodes())
}

/// Per-`JitCode` descrs.  Pyre's analog of
/// `BlackholeInterpBuilder.descrs` (`blackhole.py`) /
/// `BlackholeInterpreter.descrs` (`blackhole.py`).  RPython has a
/// single shared global pool because translation-time JitCodes are
/// produced eagerly; pyre's runtime jitcodes are emitted on demand
/// per-Python-frame and lack a global allocation index, so the pool
/// is per-`JitCode` here as a sibling of the canonical `core`.
#[derive(Clone, Debug, Default)]
pub struct JitCodeExecState {
    /// Descriptor pool — indexed by the 2-byte `j`/`d` argcode operand.
    pub descrs: Vec<RuntimeBhDescr>,
    /// Sidetable mapping the canonical-call `d` argcode descriptor slot
    /// back to pyre's full `JitCallTarget` (`{trace_ptr, concrete_ptr}`).
    /// RPython stores the callable address in the `i` operand and the
    /// signature/effect policy in the `d` operand. Pyre's runtime emitter
    /// still has a trace/concrete pointer split, so this is the minimal
    /// adaptation needed for trace recording while preserving the
    /// RPython-shaped `residual_call_*_v` payload. Keying by descriptor
    /// slot keeps the bridge per callsite; keying by int-const pool slot
    /// would collapse distinct trace targets that share a concrete
    /// pointer.
    pub call_descr_to_call_target: indexmap::IndexMap<u16, JitCallTarget>,
    /// Bytecode offset of the `BC_JIT_MERGE_POINT(_C)` opcode byte for
    /// the dispatch JitCode emitted by `lower_dispatch_body`.  `None`
    /// for non-dispatch JitCodes (helpers, sub-arms). The LLBC route
    /// permits one marker per dispatch body; `JitCodeBuilder` tolerates
    /// additional markers and retains the first marker's offset.
    ///
    /// Captured by `JitCodeBuilder::jit_merge_point` at the
    /// `self.code.len()` immediately before the opcode byte is pushed,
    /// so consumers reading `jit_merge_point_offset` land on the
    /// opcode byte itself (decoded the same way as `frame.next_u8()`
    /// would deliver it).  `register_dispatch_jitcode` reads this
    /// field to validate the green/red list counts against the
    /// declared `JitDriverDescriptor` schema without re-scanning the
    /// bytecode — RPython `blackhole.py:107-156` argcode-based decode
    /// parity, no payload-byte collision risk.
    pub jit_merge_point_offset: Option<usize>,
}

// Wrapper `JitCode` — runtime jitcode = canonical core + descr pool.
//
// RPython parity:
//   * `core` is the source-only `rpython/jit/codewriter/jitcode.py`
//     `JitCode` analog (`majit_translate::jitcode::JitCode`).  It
//     holds `name`, `fnaddr`, `jitdriver_sd`, `index`, body
//     (`code`, `constants_*`, `c_num_regs_*`, ...) — exactly the
//     fields RPython's `JitCode` carries.
//   * `exec` mirrors the descr pool RPython keeps on the
//     `BlackholeInterpBuilder` (`blackhole.py`).  In RPython the
//     pool is shared globally; pyre keeps it per-jitcode for the lazy
//     emit reasons described above on `JitCodeExecState`.
//
// Existing `jitcode.code`, `jitcode.set_body(...)`, `jitcode.body()`,
// `jitcode.fnaddr` etc. continue to work via `Deref<Target=core>` —
// the wrapper is transparent to read-side callers.  Only writers
// that require `&mut core` need `DerefMut`.
//
// Serde: the wrapper itself is intentionally NOT
// `Serialize`/`Deserialize`.  The build-time bincode embed in
// `pyre-jit-trace::jitcode_runtime` serializes
// `Vec<Arc<majit_translate::jitcode::JitCode>>` (canonical core)
// because build-time jitcodes never carry descrs.  Wrappers are
// constructed at the runtime ingress (where the canonical Arc enters
// dispatch) via `JitCode::from_canonical`.  Per-CodeObject runtime
// jitcodes are produced directly as wrappers by
// `JitCodeBuilder::finish()`.

/// Runtime JitCode = canonical RPython parity core + descr pool.
#[derive(Debug)]
pub struct JitCode {
    /// Canonical source-only `JitCode` (RPython
    /// `rpython/jit/codewriter/jitcode.py class JitCode`).
    core: majit_translate::jitcode::JitCode,
    /// Per-jitcode descr pool — pyre's analog of
    /// `BlackholeInterpBuilder.descrs` (RPython
    /// `blackhole.py:103`).  Empty for build-time canonical jitcodes
    /// (descrs resolved through the global `ALL_DESCRS` table); the
    /// `JitCodeBuilder` populates this during runtime per-CodeObject
    /// emission.
    pub exec: JitCodeExecState,
    /// Reachable symbolic residual targets, computed after the runtime wrapper
    /// has received its final function-address bindings and descriptor pool.
    ///
    /// This belongs to the JitCode it describes rather than to a side table.
    /// [`EmbeddedJitCodeTable::materialize_with_symbolic_fnaddrs`] constructs
    /// new wrappers after applying replacements, so a newly materialized table
    /// necessarily starts with a fresh answer for its bindings.
    reachable_symbolic_residuals: std::sync::OnceLock<ReachableSymbolicResiduals>,
}

/// The static residual-call refusal facts reachable from one JitCode.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReachableSymbolicResiduals {
    /// Distinct unbound symbolic function addresses, in numeric order.
    pub targets: Vec<i64>,
    /// Number of distinct JitCode objects visited through `inline_call*`.
    pub visited_jitcodes: usize,
}

// SAFETY: `JitCallTarget` / `JitCallAssemblerTarget` carry `*const ()`
// JIT-emitted code addresses; `RuntimeBhDescr::JitCode` carries
// `Arc<JitCode>` which is itself Send+Sync.  The pool is mutated only
// during `JitCodeBuilder::finish()` (single-threaded) and read
// thereafter; matches RPython's translation-time blackhole-builder
// publication flow.
unsafe impl Send for JitCode {}
unsafe impl Sync for JitCode {}

impl JitCode {
    /// Construct a fresh runtime jitcode wrapping a canonical
    /// `majit_translate::jitcode::JitCode::new(name)` core with an
    /// empty descr pool.  RPython `jitcode.py:14-20`
    /// `JitCode.__init__(name, fnaddr=None, calldescr=None, called_from=None)`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            core: majit_translate::jitcode::JitCode::new(name),
            exec: JitCodeExecState::default(),
            reachable_symbolic_residuals: std::sync::OnceLock::new(),
        }
    }

    /// `jitcode.py JitCode.name` accessor — proxies to the canonical
    /// source-only core for diagnostic / parity-validator messages.
    pub fn name(&self) -> &str {
        &self.core.name
    }

    /// Wrap a pre-built canonical `JitCode` (e.g. one produced by
    /// `CodeWriter::make_jitcodes()` at build time) with an empty
    /// descr pool.  Build-time jitcodes resolve their `'d'`/`'j'`
    /// argcodes through the global `ALL_DESCRS` table and never
    /// populate `exec.descrs`.
    pub fn from_canonical(core: majit_translate::jitcode::JitCode) -> Self {
        // The offset comes across because the two routes record it at the
        // same point and a consumer cannot recover it afterwards: an operand
        // byte may equal the opcode byte, so only the encoder knows which
        // position is an instruction start. Without this, a body assembled by
        // `majit-translate` reaches `register_dispatch_jitcode` looking like
        // one that has no marker at all.
        let jit_merge_point_offset = core.body().jit_merge_point_offset;
        Self {
            core,
            exec: JitCodeExecState {
                jit_merge_point_offset,
                ..JitCodeExecState::default()
            },
            reachable_symbolic_residuals: std::sync::OnceLock::new(),
        }
    }

    /// Borrow the canonical core (e.g. for serialization that
    /// re-serializes only the canonical fields).
    pub fn core(&self) -> &majit_translate::jitcode::JitCode {
        &self.core
    }

    /// Mutable canonical core access for in-place mutation (used by
    /// post-`set_body` `body_mut()` etc.).  RPython mutates `JitCode`
    /// fields directly post-`setup()`; pyre routes the mutation
    /// through this accessor so the wrapper stays transparent.
    pub fn core_mut(&mut self) -> &mut majit_translate::jitcode::JitCode {
        &mut self.core
    }
}

impl Default for JitCode {
    fn default() -> Self {
        Self::from_canonical(majit_translate::jitcode::JitCode::default())
    }
}

impl Clone for JitCode {
    fn clone(&self) -> Self {
        Self {
            core: self.core.clone(),
            exec: self.exec.clone(),
            reachable_symbolic_residuals: self.reachable_symbolic_residuals.clone(),
        }
    }
}

impl JitCode {
    /// Decode this body and every JitCode named by an `inline_call*`, and
    /// return the residual-call targets that are still symbolic addresses.
    ///
    /// A residual call whose funcptr operand names a runtime Int register is
    /// deliberately absent: its target is not a static property of the body,
    /// so the per-instruction refusal in `report_symbolic_residual_call_target`
    /// remains its backstop. Assembled bodies carry `startpoints`, which lets
    /// this scan distinguish opcode bytes from identical bytes in operands
    /// without maintaining a second opcode table.
    pub fn reachable_symbolic_residuals(&self) -> &ReachableSymbolicResiduals {
        self.reachable_symbolic_residuals
            .get_or_init(|| compute_reachable_symbolic_residuals(self))
    }
}

fn compute_reachable_symbolic_residuals(root: &JitCode) -> ReachableSymbolicResiduals {
    fn is_residual_call(opcode: u8) -> bool {
        matches!(
            opcode,
            insns::BC_RESIDUAL_CALL_R_V
                | insns::BC_RESIDUAL_CALL_IR_V
                | insns::BC_RESIDUAL_CALL_IRF_V
                | insns::BC_RESIDUAL_CALL_R_I
                | insns::BC_RESIDUAL_CALL_IR_I
                | insns::BC_RESIDUAL_CALL_IRF_I
                | insns::BC_RESIDUAL_CALL_R_R
                | insns::BC_RESIDUAL_CALL_IR_R
                | insns::BC_RESIDUAL_CALL_IRF_R
                | insns::BC_RESIDUAL_CALL_IRF_F
        )
    }

    fn is_inline_call(opcode: u8) -> bool {
        matches!(
            opcode,
            insns::BC_INLINE_CALL
                | insns::BC_INLINE_CALL_R_I
                | insns::BC_INLINE_CALL_R_R
                | insns::BC_INLINE_CALL_R_V
                | insns::BC_INLINE_CALL_IR_I
                | insns::BC_INLINE_CALL_IR_R
                | insns::BC_INLINE_CALL_IR_V
                | insns::BC_INLINE_CALL_IRF_I
                | insns::BC_INLINE_CALL_IRF_R
                | insns::BC_INLINE_CALL_IRF_F
                | insns::BC_INLINE_CALL_IRF_V
        )
    }

    fn visit(
        jitcode: &JitCode,
        visited: &mut std::collections::BTreeSet<usize>,
        targets: &mut std::collections::BTreeSet<i64>,
    ) {
        if !visited.insert(jitcode as *const JitCode as usize) {
            return;
        }
        let Some(starts) = jitcode.startpoints.as_ref() else {
            // Hand-built bodies that bypass the assembler do not identify
            // instruction boundaries. Their runtime residuals remain guarded
            // by the site-level refusal rather than guessing that an operand
            // byte is an opcode.
            return;
        };
        for &pc in starts {
            let Some(&opcode) = jitcode.code.get(pc) else {
                continue;
            };
            if is_residual_call(opcode) {
                let Some(&funcptr_reg) = jitcode.code.get(pc + 1) else {
                    continue;
                };
                let funcptr_reg = funcptr_reg as usize;
                if funcptr_reg < jitcode.num_regs_i() {
                    continue;
                }
                let Some(&target) = jitcode.constants_i.get(funcptr_reg - jitcode.num_regs_i())
                else {
                    continue;
                };
                if crate::pyjitpl::resolve_symbolic_fnaddr_path(target).is_some()
                    || majit_translate::codewriter::call::is_symbolic_fnaddr(target)
                {
                    targets.insert(target);
                }
                continue;
            }
            if !is_inline_call(opcode) {
                continue;
            }
            let Some((&lo, &hi)) = jitcode.code.get(pc + 1).zip(jitcode.code.get(pc + 2)) else {
                continue;
            };
            let descr_index = u16::from_le_bytes([lo, hi]) as usize;
            if let Some(callee) = jitcode
                .descr_at(descr_index)
                .and_then(RuntimeBhDescr::as_jitcode_owned)
            {
                visit(callee.as_ref(), visited, targets);
            }
        }
    }

    let mut visited = std::collections::BTreeSet::new();
    let mut targets = std::collections::BTreeSet::new();
    visit(root, &mut visited, &mut targets);
    ReachableSymbolicResiduals {
        targets: targets.into_iter().collect(),
        visited_jitcodes: visited.len(),
    }
}

impl std::ops::Deref for JitCode {
    type Target = majit_translate::jitcode::JitCode;
    fn deref(&self) -> &majit_translate::jitcode::JitCode {
        &self.core
    }
}

impl std::ops::DerefMut for JitCode {
    fn deref_mut(&mut self) -> &mut majit_translate::jitcode::JitCode {
        &mut self.core
    }
}

impl std::fmt::Display for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.core, f)
    }
}

/// Helper preserved from the runtime jitcode era so callers that
/// expected the runtime `JitCode` body fields at the top level keep
/// working through `Deref<Target=JitCodeBody>`.
///
/// `trailing_return_info` is wired here because it depends on the
/// runtime BC_* opcode bytes (`BC_VOID_RETURN`, `BC_INT_RETURN`,
/// `BC_REF_RETURN`, `BC_FLOAT_RETURN`) which are runtime-defined; the
/// canonical jitcode crate does not import them. Provided as a free
/// function so the call sites can keep `jitcode.trailing_return_info()`
/// syntax via the existing trait impl below.
pub trait JitCodeRuntimeExt {
    /// Inspect the trailing typed return opcode of a helper jitcode.
    fn trailing_return_info(&self) -> Option<(JitArgKind, u16)>;
}

impl JitCode {
    /// Resolve `BC_CALL_*` / `BC_RESIDUAL_CALL_*` function-target
    /// descr.  Mirrors RPython `blackhole.py:1225-1256` where the
    /// calling-convention descr travels through `descrs[idx]`; pyre
    /// additionally bundles the trace and concrete fn pointers in
    /// the `Call` variant because the call encoding pre-dates the
    /// RPython-orthodox register-fed function address.
    pub fn call_target(&self, index: usize) -> &JitCallTarget {
        match self.descr_at(index) {
            Some(RuntimeBhDescr::Call(target)) => target,
            other => {
                panic!("BC_CALL_*/RESIDUAL_CALL_*: descrs[{index}] is not a Call entry: {other:?}",)
            }
        }
    }

    /// Transitional CALL_ASSEMBLER target lookup for the hardcoded
    /// JitCodeBuilder bytecode.  RPython stores the callee loop token
    /// in descriptor data threaded through the shared `descrs` pool;
    /// pyre mirrors the shape via the `AssemblerToken` variant.
    pub fn call_assembler_target(&self, index: usize) -> (u64, *const ()) {
        let target = self
            .descr_at(index)
            .and_then(RuntimeBhDescr::as_assembler_token)
            .unwrap_or_else(|| {
                panic!("BC_CALL_ASSEMBLER_*: descrs[{index}] is not an AssemblerToken entry",)
            });
        (target.token_number, target.concrete_ptr)
    }

    /// Resolve a `d`/`j` argcode descr for this jitcode.  Runtime-emitted
    /// jitcodes answer from their per-jitcode `exec.descrs` pool; build-time
    /// (LLBC-extracted) jitcodes carry an empty pool and fall through to the
    /// process-global `GLOBAL_BUILD_DESCR_POOL` (RPython's single shared
    /// `Assembler.descrs`).  A populated per-jitcode slot always wins, so the
    /// runtime path stays byte-identical to a direct `exec.descrs` read.
    pub fn descr_at(&self, index: usize) -> Option<&RuntimeBhDescr> {
        if let Some(entry) = self.exec.descrs.get(index) {
            return Some(entry);
        }
        global_build_descr_pool().and_then(|pool| pool.get(index))
    }

    /// The `BC_INLINE_CALL` whose encoding ends exactly at `end_pc`.
    ///
    /// A caller frame suspended in an inline call carries two things its own
    /// resume section does not: which callee it is waiting on, and which of
    /// its registers the result lands in.  Neither is in the stream by design
    /// — `opencoder.py _ensure_parent_resumedata` reads a parent frame's
    /// liveness with `in_a_call=True`, which blanks the result register first,
    /// because nothing has written it yet.  `pyjitpl.py
    /// MIFrame.make_result_of_lastop` recovers it from the bytecode instead,
    /// taking the register from `ord(self.bytecode[self.pc - 1])` and its kind
    /// from `self.jitcode._resulttypes[self.pc]`.  This is that read, for the
    /// grouped encoding `inline_call_typed` (`jitcode/assembler.rs`) emits.
    ///
    /// The instruction cannot be decoded backwards, so it is decoded forwards
    /// from the only start position that can produce it.  Its width is
    /// `1 + 2 + 2 + 3 * num_args + 3` — opcode, sub-JitCode index, argument
    /// count, one `(kind, caller_src, callee_dst)` triple per argument, then
    /// the three optional return slots — so each candidate `num_args` names
    /// exactly one start, and that start is the real one only when the opcode
    /// byte is there AND the count it encodes is the count that was assumed.
    /// A position satisfying both while still ending at `end_pc` has decoded
    /// itself; requiring the match to be unique is what turns a coincidence
    /// into a decline rather than into a wrong answer.
    ///
    /// `None` means `end_pc` is not the far side of a `BC_INLINE_CALL` in this
    /// jitcode — including when it is one of the typed `BC_INLINE_CALL_*`
    /// variants, which no `JitCodeBuilder` emits.  Callers read it as "cannot
    /// resume through this frame" rather than guessing.
    pub fn inline_call_ending_at(&self, end_pc: usize) -> Option<InlineCallSite> {
        let body = self.try_body()?;
        let code = &body.code;
        if end_pc > code.len() {
            return None;
        }
        let mut found: Option<InlineCallSite> = None;
        for num_args in 0.. {
            let Some(start) = end_pc.checked_sub(INLINE_CALL_FIXED_WIDTH + 3 * num_args) else {
                break;
            };
            if code.get(start).copied() != Some(insns::BC_INLINE_CALL) {
                continue;
            }
            let mut cursor = start + 1;
            let sub_idx = read_u16(code, &mut cursor) as usize;
            if read_u16(code, &mut cursor) as usize != num_args {
                continue;
            }
            cursor += 3 * num_args;
            let return_slot = |cursor: &mut usize| match read_reg(code, cursor) {
                NO_RETURN_REG => None,
                reg => Some(reg as usize),
            };
            let site = InlineCallSite {
                sub_idx,
                return_i: return_slot(&mut cursor),
                return_r: return_slot(&mut cursor),
                return_f: return_slot(&mut cursor),
            };
            debug_assert_eq!(
                cursor, end_pc,
                "the inline-call width formula disagrees with the decode",
            );
            // `inline_call_typed` refuses to emit more than one filled slot,
            // so a decode holding two has not landed on a real instruction.
            if site.filled_return_slots() > 1 {
                continue;
            }
            if found.replace(site).is_some() {
                // Two start positions both decode into an instruction ending
                // here, so the bytes do not name one call.
                return None;
            }
        }
        let site = found?;
        // `pyjitpl.py make_result_of_lastop`'s own check, in its own place:
        //
        //     assert typeof[self.jitcode._resulttypes[self.pc]] == got_type
        //
        // The writer records the kind at end-of-instruction position
        // (`record_resulttype`, `assembler.py`), which makes it an independent
        // witness of which return slot this call filled.
        if body.resulttypes.as_ref()?.get(&end_pc).copied() != site.recorded_resulttype() {
            return None;
        }
        Some(site)
    }
}

/// The encoded width of a `BC_INLINE_CALL` that passes no arguments: the
/// opcode byte, the `u16` sub-JitCode index, the `u16` argument count, and the
/// three return-slot register bytes.  Each argument adds a
/// `(kind, caller_src, callee_dst)` triple of one byte each.
const INLINE_CALL_FIXED_WIDTH: usize = 1 + 2 + 2 + 3;

/// The operands of one `BC_INLINE_CALL`, as [`JitCode::inline_call_ending_at`]
/// recovers them from the instruction's far side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InlineCallSite {
    /// The caller `descrs` slot holding the callee JitCode — the `j` argcode
    /// `blackhole.py` resolves through `self.descrs[idx]`.
    pub sub_idx: usize,
    /// The caller register the callee's int result lands in, if it returns one.
    pub return_i: Option<usize>,
    /// The caller register the callee's ref result lands in, if it returns one.
    pub return_r: Option<usize>,
    /// The caller register the callee's float result lands in, if it returns one.
    pub return_f: Option<usize>,
}

impl InlineCallSite {
    /// The caller-side result register and the bank it lands in.
    ///
    /// `None` for a call whose result is discarded, which is the `_v` shape:
    /// all three slots hold [`NO_RETURN_REG`].
    pub fn result_slot(&self) -> Option<(JitArgKind, usize)> {
        match (self.return_i, self.return_r, self.return_f) {
            (Some(dst), None, None) => Some((JitArgKind::Int, dst)),
            (None, Some(dst), None) => Some((JitArgKind::Ref, dst)),
            (None, None, Some(dst)) => Some((JitArgKind::Float, dst)),
            _ => None,
        }
    }

    /// How many of the three return slots name a register.  A real
    /// `BC_INLINE_CALL` has at most one.
    fn filled_return_slots(&self) -> usize {
        [self.return_i, self.return_r, self.return_f]
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    /// The `_resulttypes` entry these return slots imply, in the writer's own
    /// spelling (`record_resulttype`, `jitcode/assembler.rs`).
    fn recorded_resulttype(&self) -> Option<char> {
        match self.result_slot()? {
            (JitArgKind::Int, _) => Some('i'),
            (JitArgKind::Ref, _) => Some('r'),
            (JitArgKind::Float, _) => Some('f'),
        }
    }
}

impl JitCodeRuntimeExt for JitCode {
    fn trailing_return_info(&self) -> Option<(JitArgKind, u16)> {
        let body = self.try_body()?;
        let code = &body.code;
        if code.last().copied() == Some(insns::BC_VOID_RETURN) || code.len() < 2 {
            return None;
        }
        // Typed-return opcodes carry a 1-byte
        // register source operand (`int_return/i`, `ref_return/r`,
        // `float_return/f` per RPython argcode contract).
        let opcode_pos = code.len() - 2;
        let opcode = code[opcode_pos];
        let src = code[opcode_pos + 1] as u16;
        match opcode {
            insns::BC_INT_RETURN => Some((JitArgKind::Int, src)),
            insns::BC_REF_RETURN => Some((JitArgKind::Ref, src)),
            insns::BC_FLOAT_RETURN => Some((JitArgKind::Float, src)),
            _ => None,
        }
    }
}

/// The encoded width of a jitcode **register operand**: exactly one byte.
///
/// A register operand occupies a single byte in the jitcode stream — the
/// canonical encoding (`chr(reg.index)` when assembling, `ord(code[position])`
/// when decoding). The 256-per-kind register limit, the `num_regs < 256`
/// assemble-time decline, and every register decode site all rest on this
/// width. Encode register operands with `JitCodeBuilder::push_reg_u8` and
/// decode them with `read_reg` / `next_reg`, so the width lives in one place.
///
/// This MUST remain `u8`. Do NOT widen it to `u16` (or any wider type):
/// doing so silently desyncs the encoder from the 1-byte register decoders and
/// reintroduces exactly the register-width divergence this alias exists to
/// prevent. Non-register operands (descr / field / array indexes) are `u16`
/// and are deliberately NOT covered by this alias.
pub type JitcodeReg = u8;

/// The register-operand value that encodes "no register" / "no return slot"
/// (e.g. a `recursive_call_void` result slot, or an `inline_call` with no
/// caller destination). It is [`JitcodeReg::MAX`], safe as a sentinel because
/// `try_finish` declines any JitCode with `num_regs >= 256`, so every real
/// register index is `<= 254`. Encode and decode both reference this constant
/// so the sentinel never drifts.
pub const NO_RETURN_REG: JitcodeReg = JitcodeReg::MAX;

pub(crate) fn read_u8(code: &[u8], cursor: &mut usize) -> u8 {
    let value = *code.get(*cursor).expect("truncated jitcode");
    *cursor += 1;
    value
}

/// Read one register operand ([`JitcodeReg`]) from `code` at `*cursor`,
/// advancing past it. Use this — never a bare `read_u8` — wherever the byte
/// is a register index, so the 1-byte register width stays enforced in one
/// place. See [`JitcodeReg`].
pub(crate) fn read_reg(code: &[u8], cursor: &mut usize) -> JitcodeReg {
    read_u8(code, cursor)
}

pub(crate) fn read_u16(code: &[u8], cursor: &mut usize) -> u16 {
    let lo = *code.get(*cursor).expect("truncated jitcode");
    let hi = *code.get(*cursor + 1).expect("truncated jitcode");
    *cursor += 2;
    u16::from_le_bytes([lo, hi])
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_translate::jitcode::{JitCode as BuildJitCode, JitCodeBody as BuildJitCodeBody};

    /// `register_dispatch_jitcode` refuses a portal whose `exec` does not say
    /// where its marker is, and only the encoder can say: an operand byte may
    /// equal the opcode byte, so the offset cannot be recovered by scanning.
    /// Both routes into a body now record it, and this is the crossing where a
    /// body assembled by `majit-translate` would otherwise arrive looking like
    /// one that has no marker at all.
    #[test]
    fn from_canonical_carries_the_bodys_merge_point_offset() {
        let core = BuildJitCode::new("portal");
        core.set_body(BuildJitCodeBody {
            jit_merge_point_offset: Some(3),
            ..BuildJitCodeBody::default()
        });
        assert_eq!(
            JitCode::from_canonical(core).exec.jit_merge_point_offset,
            Some(3)
        );
    }

    /// A body with no marker keeps `None`, so "has no merge point" stays
    /// distinguishable from "has one at offset 0".
    #[test]
    fn from_canonical_leaves_a_markerless_body_without_an_offset() {
        let core = BuildJitCode::new("callee");
        core.set_body(BuildJitCodeBody::default());
        assert_eq!(
            JitCode::from_canonical(core).exec.jit_merge_point_offset,
            None
        );
    }

    #[test]
    fn wellknown_bh_insns_stays_canonical_and_avoids_false_call_family_keys() {
        use majit_translate::insns as ti;
        let insns = wellknown_bh_insns();
        assert!(
            !insns.contains_key("jump/L"),
            "wellknown_bh_insns must keep the canonical goto/L spelling",
        );
        // Canonical RPython `conditional_call_*` / `record_known_result_*`
        // keys (`blackhole.py:1258-1296` + `:621-630`) are pinned at the
        // distinct bytes [`BC_CONDITIONAL_CALL_*`] / [`BC_RECORD_KNOWN_RESULT_*`].
        // The pyre-only helper-side proc-macro adapter keys
        // `cond_call_*_ext/P` / `record_known_result_*_ext/P` reuse the
        // legacy [`BC_COND_CALL_*`] / [`BC_RECORD_KNOWN_RESULT_*`] bytes
        // (`extension_insns()`).  The two byte ranges must stay
        // disjoint so the canonical and adapter forms cannot collide on
        // dispatch.
        assert_eq!(
            insns.get("conditional_call_ir_v/iiIRd").copied(),
            Some(ti::BC_CONDITIONAL_CALL_IR_V),
        );
        assert_eq!(
            insns.get("conditional_call_value_ir_i/iiIRd>i").copied(),
            Some(ti::BC_CONDITIONAL_CALL_VALUE_IR_I),
        );
        assert_eq!(
            insns.get("conditional_call_value_ir_r/riIRd>r").copied(),
            Some(ti::BC_CONDITIONAL_CALL_VALUE_IR_R),
        );
        assert_eq!(
            insns.get("record_known_result_i_ir_v/iiIRd").copied(),
            Some(ti::BC_RECORD_KNOWN_RESULT_I_IR_V),
        );
        assert_eq!(
            insns.get("record_known_result_r_ir_v/riIRd").copied(),
            Some(ti::BC_RECORD_KNOWN_RESULT_R_IR_V),
        );
        assert_ne!(
            insns.get("conditional_call_ir_v/iiIRd").copied(),
            Some(ti::BC_COND_CALL_VOID),
            "canonical conditional_call_ir_v byte must NOT collide with \
             helper-side BC_COND_CALL_VOID adapter byte",
        );
        assert_ne!(
            insns.get("record_known_result_r_ir_v/riIRd").copied(),
            Some(ti::BC_RECORD_KNOWN_RESULT_REF),
            "canonical record_known_result_r_ir_v byte must NOT collide \
             with helper-side BC_RECORD_KNOWN_RESULT_REF adapter byte",
        );
        // Canonical `inline_call_*/d{R,IR,IRF}>{i,r,v,f}` keys live in
        // `wellknown_bh_insns()` with their own distinct `BC_*` bytes
        // (187-194); the pyre-only nested-bytecode adapter
        // `inline_call_nested_ext/P` reuses `BC_INLINE_CALL = 17` and is
        // quarantined in `extension_insns()`.  The two byte ranges
        // are disjoint, so they cannot collide on dispatch.
        assert!(insns.contains_key("inline_call_ir_r/dIR>r"));
        assert!(insns.contains_key("inline_call_irf_f/dIRF>f"));
        assert_ne!(
            insns.get("inline_call_ir_r/dIR>r").copied(),
            Some(majit_translate::insns::BC_INLINE_CALL),
            "canonical inline_call_ir_r byte must NOT collide with \
             helper-side BC_INLINE_CALL adapter byte",
        );
        assert_eq!(
            insns.get("getfield_vable_i/rd>i"),
            Some(&super::insns::BC_GETFIELD_VABLE_I)
        );
        assert_eq!(
            insns.get("getfield_vable_r/rd>r"),
            Some(&super::insns::BC_GETFIELD_VABLE_R)
        );
        assert_eq!(
            insns.get("getfield_vable_f/rd>f"),
            Some(&super::insns::BC_GETFIELD_VABLE_F)
        );
        assert_eq!(
            insns.get("setfield_vable_i/rid"),
            Some(&super::insns::BC_SETFIELD_VABLE_I)
        );
        assert_eq!(
            insns.get("setfield_vable_r/rrd"),
            Some(&super::insns::BC_SETFIELD_VABLE_R)
        );
        assert_eq!(
            insns.get("setfield_vable_f/rfd"),
            Some(&super::insns::BC_SETFIELD_VABLE_F)
        );
        assert_eq!(
            insns.get("getarrayitem_vable_i/ridd>i"),
            Some(&super::insns::BC_GETARRAYITEM_VABLE_I)
        );
        assert_eq!(
            insns.get("getarrayitem_vable_r/ridd>r"),
            Some(&super::insns::BC_GETARRAYITEM_VABLE_R)
        );
        assert_eq!(
            insns.get("getarrayitem_vable_f/ridd>f"),
            Some(&super::insns::BC_GETARRAYITEM_VABLE_F)
        );
        assert_eq!(
            insns.get("setarrayitem_vable_i/riidd"),
            Some(&super::insns::BC_SETARRAYITEM_VABLE_I)
        );
        assert_eq!(
            insns.get("setarrayitem_vable_r/rirdd"),
            Some(&super::insns::BC_SETARRAYITEM_VABLE_R)
        );
        assert_eq!(
            insns.get("setarrayitem_vable_f/rifdd"),
            Some(&super::insns::BC_SETARRAYITEM_VABLE_F)
        );
        assert_eq!(
            insns.get("arraylen_vable/rdd>i"),
            Some(&super::insns::BC_ARRAYLEN_VABLE)
        );
        assert_eq!(
            insns.get("hint_force_virtualizable/r"),
            Some(&super::insns::BC_HINT_FORCE_VIRTUALIZABLE)
        );
        // of `pyre-call-family-canonical-migration.md` — canonical
        // residual_call_*_v opcodes reserved for emit migration.
        assert_eq!(
            insns.get("residual_call_r_v/iRd"),
            Some(&super::insns::BC_RESIDUAL_CALL_R_V),
        );
        assert_eq!(
            insns.get("residual_call_ir_v/iIRd"),
            Some(&super::insns::BC_RESIDUAL_CALL_IR_V),
        );
        assert_eq!(
            insns.get("residual_call_irf_v/iIRFd"),
            Some(&super::insns::BC_RESIDUAL_CALL_IRF_V),
        );
    }

    /// The `extension_insns()` quarantine holds 8 keys arising from
    /// the borrow-checker abort signals (2) and the proc-macro JIT-machine
    /// state addressing (6), plus 3 more pyre-only
    /// keys — `inline_call_nested_ext/P` (nested-bytecode `inline_call`
    /// adapter, `BC_INLINE_CALL = 17`), `abort/>r` (Ref-result variant of
    /// `abort/`), `vtable_method_ptr/rd>i` (dyn-trait method-pointer
    /// reification) — so the `extension_insns()` table now holds 11
    /// entries total.  `wellknown_bh_insns()` is a strict
    /// subset of RPython's canonical opname universe; `insn_byte` merges both
    /// tables so build-time `write_insn(...)` callers continue to resolve
    /// unchanged.
    #[test]
    fn extension_insns_quarantines_runtime_keys_out_of_wellknown() {
        let wellknown = wellknown_bh_insns();
        let extension = extension_insns();

        let pairs = [
            // Borrow-checker abort signals.
            ("abort/", insns::BC_ABORT),
            ("abort_permanent/", insns::BC_ABORT_PERMANENT),
            // Proc-macro JIT-machine state addressing.
            ("load_state_field_ref/dr", insns::BC_LOAD_STATE_FIELD_REF),
            ("store_state_field_ref/dr", insns::BC_STORE_STATE_FIELD_REF),
            ("load_state_field/di", insns::BC_LOAD_STATE_FIELD),
            ("store_state_field/di", insns::BC_STORE_STATE_FIELD),
            ("load_state_array/dii", insns::BC_LOAD_STATE_ARRAY),
            ("store_state_array/dii", insns::BC_STORE_STATE_ARRAY),
            // pyre nested-bytecode inline_call (pyre-only `P` argcode).
            ("inline_call_nested_ext/P", insns::BC_INLINE_CALL),
            // Ref-result variant of the borrow-checker abort signal.
            ("abort/>r", majit_translate::insns::BC_ABORT_RESULT_R),
            // dyn-trait method pointer reification (backend epic).
            (
                "vtable_method_ptr/rd>i",
                majit_translate::insns::BC_VTABLE_METHOD_PTR,
            ),
        ];

        for (key, expected_byte) in pairs {
            assert!(
                !wellknown.contains_key(key),
                "{key} must be quarantined in extension_insns(), not \
                 wellknown_bh_insns()",
            );
            assert_eq!(
                extension.get(key),
                Some(&expected_byte),
                "{key} must be present in extension_insns() with the \
                 fixed BC_* byte",
            );
            assert_eq!(
                majit_translate::insns::insn_byte(key),
                expected_byte,
                "insn_byte must resolve {key} via the merged extension+\
                 wellknown table",
            );
        }
    }

    #[test]
    fn canonical_build_jitcode_sizes_blackhole_register_files_without_conversion() {
        // Extract the upstream-common part of blackhole.py setposition
        // (register sizing + constant copy) and apply it directly to the
        // canonical codewriter JitCode. Dispatch still needs the runtime
        // adapter JitCode for exec.* pools, but the register-file setup no
        // longer needs a build→runtime conversion just to match RPython's
        // `num_regs_* + len(constants_*)` logic.
        //
        // RPython: `blackhole.py setposition` allocates `num_regs_i +
        // len(constants_i)` slots per register file and copies each constant
        // into the tail portion of the file. We verify both — the array
        // sizes and the copied-in constants.
        use crate::blackhole::BlackholeInterpBuilder;

        let body = BuildJitCodeBody {
            code: vec![insns::BC_LIVE, 0x00, 0x00], // live/ with 2-byte offset
            c_num_regs_i: 4,
            c_num_regs_r: 2,
            c_num_regs_f: 1,
            constants_i: vec![100, 200, 300],
            constants_r: vec![
                (0xAABB_CCDD_EEFF_0011_u64 as i64).into(),
                (0x2233_4455_6677_8899_u64 as i64).into(),
            ],
            constants_f: vec![f64::to_bits(1.25_f64) as i64],
            ..Default::default()
        };
        let bt = BuildJitCode::new("slice2/test");
        bt.set_body(body);

        let mut builder = BlackholeInterpBuilder::new();
        let mut bh = builder.acquire_interp();
        bh.prepare_registers_for_canonical_jitcode(&bt, 0);

        // num_regs_and_consts_i = 4 + 3 = 7; constants occupy [4..7].
        assert_eq!(bh.registers_i.len(), 7);
        assert_eq!(&bh.registers_i[4..7], &[100, 200, 300]);
        // Working regs remain zero-initialised.
        assert_eq!(&bh.registers_i[0..4], &[0, 0, 0, 0]);

        // Refs: u64 bit pattern reinterpreted as i64 by the conversion.
        assert_eq!(bh.registers_r.len(), 4); // 2 regs + 2 constants
        assert_eq!(bh.registers_r[2], 0xAABB_CCDD_EEFF_0011_u64 as i64);
        assert_eq!(bh.registers_r[3], 0x2233_4455_6677_8899_u64 as i64);

        // Floats: f64 bits reinterpreted; round-trip through f64::to_bits
        // must match what BlackholeInterpreter sees.
        assert_eq!(bh.registers_f.len(), 2);
        assert_eq!(bh.registers_f[1], f64::to_bits(1.25_f64) as i64);

        assert_eq!(bh.position, 0);
        assert!(bh.jitcode.code.is_empty());
    }
}
