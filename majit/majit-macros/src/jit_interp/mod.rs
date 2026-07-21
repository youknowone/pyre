//! `#[jit_interp]` proc macro implementation.
//!
//! Transforms an interpreter mainloop function into a JIT-enabled version by:
//! 1. Generating `trace_instruction` from the match dispatch
//! 2. Generating `JitState` types and impl
//! 3. Replacing `jit_merge_point!()` / `can_enter_jit!()` markers

pub(crate) mod call_policy_byte;
mod classify;
mod codegen_state;
mod codegen_trace;
mod green_type_tag;
pub(crate) mod jitcode_lower;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Expr, Ident, ItemFn, LitBool, Path, Token, braced, bracketed,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Parsed configuration from `#[jit_interp(...)]` attributes.
///
/// ## Helper discovery
///
/// Helpers (functions called from traced match arms) can be declared in
/// three ways, from most explicit to most concise:
///
/// 1. **`calls = { helper_a, helper_b => residual_int, ... }`**
///    Brace-delimited list with optional per-helper policy overrides.
///
/// 2. **`helpers = [helper_a, helper_b, helper_c]`**
///    Bracket-delimited shorthand — all helpers use auto-inferred policy.
///    Equivalent to listing each in `calls = { ... }` without a `=>` override.
///    Can be combined with `calls` for helpers that need explicit policies.
///
/// 3. **`auto_calls = true`**
///    Infer helper policies from sidecar `#[elidable]` / `#[dont_look_inside]`
///    / `#[jit_inline]` attributes on every call site in the traced arms.
///    Value-call inference is currently limited to helpers whose result bank
///    is statically int-shaped; ref/float-return helpers still need explicit
///    `calls = { helper => ... }` overrides such as `inline_ref`,
///    `inline_float`, `residual_ref_wrapped`, or `residual_float_wrapped`.
///
/// ### Module-level discovery
///
/// For automatic helper discovery, use `#[jit_module]` on the enclosing
/// `mod` block. It scans all items for JIT-annotated functions and
/// generates hidden registry constants (`__MAJIT_DISCOVERED_HELPERS`,
/// `__MAJIT_HELPER_POLICIES`). Alternatively, use `helpers` or `calls`
/// to explicitly list the functions that need JIT integration.
pub struct JitInterpConfig {
    /// The interpreter state type (e.g., `AheuiState`).
    pub state_type: Ident,
    /// The environment type (e.g., `Program`).
    pub env_type: Ident,
    /// Interpreter I/O function → JIT shim function mapping.
    pub io_shims: Vec<(Path, Ident)>,
    /// Interpreter function call policies for helper calls.
    /// Populated from both `calls = { ... }` and `helpers = [...]`.
    pub calls: Vec<CallEntry>,
    /// Whether direct helper calls should be auto-inferred from sidecar metadata.
    pub auto_calls: bool,
    /// Optional structured green-key expressions for marker rewrite.
    pub greens: Vec<Expr>,
    /// Slice (audit Issue #6) — explicit red declarations for the
    /// dispatch JitCode `BC_JIT_MERGE_POINT` payload.  RPython
    /// `jtransform.py:1700 make_three_lists(op.args[2+num_green_args:])`
    /// derives reds from the marker call's tail args; pyre's marker is
    /// stateless (no tail args), so consumers declare the reds via
    /// this config slot instead.  Empty = use the default candidate
    /// list `[program, pc(+ optional vable)]` with declared greens
    /// filtered out (the pre-Issue-#6 pyre behavior).
    pub reds: Vec<Expr>,
    /// Per-green type tag from `greens = [name: tag]` syntax.
    /// Lockstep with [`Self::greens`] (same length, same order).  `None`
    /// at a given index means the green flows through the
    /// `<_ as GreenAsI64>::__green_repr` trait dispatch unchanged;
    /// `Some(tag)` overrides the dispatch with an explicit
    /// `(value_bits, GreenType::<tag>)` pair so str / unicode greens route
    /// through the hardcoded `default_str_eq` / `default_str_hash` /
    /// `default_unicode_hash` in `majit-ir/src/value.rs`
    /// (`warmstate.py:108-128 lltype.Ptr STR/UNICODE` parity, no
    /// frontend override).
    pub green_type_tags: Vec<Option<green_type_tag::GreenTypeTag>>,
    /// Virtualizable frame field declaration.
    ///
    /// RPython equivalent: jtransform.py's `is_virtualizable_getset()`.
    /// When set, the proc macro rewrites field accesses on the virtualizable
    /// variable to use TraceCtx vable_* methods instead of heap operations.
    pub virtualizable_decl: Option<VirtualizableDecl>,
    /// State field declarations for register/tape machines.
    ///
    /// When set, the macro tracks state struct fields as JIT-managed values
    /// instead of requiring a storage pool. Enables `state.field` and
    /// `state.array[index]` patterns in match arms.
    pub state_fields: Option<StateFieldsConfig>,
    /// Optional recovery method path called after compiled loop guard failure.
    /// The method is called on `&mut self` (the state type) to re-derive
    /// tracked state fields from shared mutable state after the compiled
    /// loop's residual calls have mutated it.
    pub recover: Option<Path>,
    /// Optional concrete fallback for `recursive_portal_call!` (a recursive
    /// portal re-entry).  The function is the runtime analog of the portal
    /// runner (`jd.portal_runner_ptr`, call.py:363): the transformed
    /// (concrete) function calls it with the macro greens forwarded
    /// positionally (jitdriver declaration order), while the dispatch JitCode
    /// emits `BC_RECURSIVE_CALL_*`.  Required iff the body uses
    /// `recursive_portal_call!`.
    pub recursive_entry: Option<Path>,
    /// Residual helpers that mutate a `state.<ref_scalar>.<field>` through
    /// opaque host code, declared as `residual_writes = { <ref_scalar>.<field>
    /// [@ StructType] => [helpers] }`.  `@ StructType` adds a nominal-layout
    /// alias for a type-punned ref scalar.  Drives the write-set `EffectInfo` the lowering attaches
    /// so the optimizer invalidates the matching cached `getfield_gc_i` after
    /// the call.  Empty for interpreters with no residual field mutators.
    pub residual_writes: Vec<ResidualWriteEntry>,
    /// `ref(T)` state scalars that are bases of a contiguous raw-pointer array
    /// (`[*mut U; N]` at offset 0 of `T`), declared as
    /// `pool_arrays = { <ref> => <getter> [-> ElementType], ... }`.  The
    /// indexing marker call `<getter>(state.<ref>, <int>)` lowers to
    /// `getarrayitem_gc_r` (a
    /// re-producible heap read) instead of an opaque residual CALL_R, so the
    /// loaded element re-derives from the index each loop entry and the short
    /// preamble can re-emit it.  Selection is keyed on the `getter` function
    /// identity (not the arg shape alone).  Empty for interpreters with no
    /// pool-array indexing.
    pub pool_arrays: Vec<PoolArrayEntry>,
    /// Struct field type declarations for ref-kind field access through
    /// `ref(T)` state scalars and non-state ref bindings.  Declared as
    /// `ref_fields = { Struct::field => PointeeType, ... }`.  When the
    /// lowerer encounters `state.<ref_scalar>.<field>` or `<binding>.<field>`
    /// and `(Struct, field)` is listed here, it emits `getfield_gc_r` /
    /// `setfield_gc_r` (ref-kind) instead of the default `_gc_i` (int-kind).
    /// The `PointeeType` is recorded so subsequent field access on the
    /// returned ref can resolve its struct layout.
    pub ref_fields: Vec<RefFieldEntry>,
    /// Struct type annotations for ref-returning call results.
    /// `call_returns = { func_path => StructType, ... }`.  When a
    /// `residual_ref` (or similar ref-returning) call result is bound,
    /// the lowerer sets `Binding.struct_type` to the declared type so
    /// subsequent `result.field` access resolves through `ref_fields`.
    pub call_returns: Vec<(Path, Path)>,
    /// Struct literal → concrete allocator mapping for the concrete path.
    /// `struct_allocs = { StructType => allocator_func, ... }`.  When the
    /// concrete-path `RefFieldRewriter` encounters a struct literal
    /// `StructType { f0: v0, f1: v1 }`, it rewrites the literal to
    /// `allocator_func(v0, v1)`.  The JIT path already handles struct
    /// literals natively via `lower_struct_value` (New + SetfieldGc).
    pub struct_allocs: Vec<(Path, Path)>,
    /// Structs whose `New` allocation should use the headerless nursery opcode.
    /// `headerless_structs = { StructType, ... }`.
    pub headerless_structs: Vec<Path>,
    /// Pure function → native IR integer binop aliases.
    /// `native_int_binops = { val_add => IntAdd, ... }`.  When the JIT-path
    /// lowerer encounters a call whose path matches a key, it emits the named
    /// IR opcode (e.g. `IntAdd`) directly instead of routing through the
    /// call-policy machinery.  The concrete path calls the function normally.
    /// In RPython, `int + int` is lowered to `int_add` at rtype time
    /// (rint.py:314 `_rtype_template`), so the JIT codewriter never sees a
    /// call.  In pyre's LLBC tracer, the equivalent Rust function call
    /// (e.g. `val_add`) is still visible, so this config rewrites it to a
    /// native IR binop at codewriter time.
    pub native_int_binops: Vec<(Path, Ident)>,
    /// native_tag_small = { jit_retag_small } — unary fns lowered to (x<<1)|1 native IR.
    pub native_tag_small: Vec<Path>,
    /// Opt-in: route pure forward-advancing dispatch arms (those whose body
    /// only does work then `pc += N`, with no back-edge / `can_enter_jit!` /
    /// early return) through the per-arm sub-JitCode path with a pc-returning
    /// `inline_call_<types>_i`, instead of force-inlining them into the
    /// dispatch JitCode.  This keeps the dispatch JitCode's register/const
    /// footprint small (the heavy arm bodies move into their own sub-JitCodes,
    /// each with its own ≤256 budget) — mirrors `next_instr = self.OPCODE(...)`.
    /// Off by default, so kernels that do not set it are byte-identical.
    pub split_dispatch: bool,
    /// Opt-in: lower the top-level opcode dispatch discrimination as one
    /// `switch/id` op instead of a `goto_if_not_int_eq` guard chain. Off by
    /// default so existing interpreters keep byte-identical dispatch JitCode.
    pub switch_dispatch: bool,
}

/// Virtualizable frame field declaration for `#[jit_interp]`.
///
/// RPython equivalent: VirtualizableInfo from virtualizable.py, combined
/// with jtransform.py's field-to-descriptor mapping.
///
/// Syntax in attribute:
/// ```ignore
/// virtualizable_fields = {
///     var: frame,
///     token_offset: PYFRAME_VABLE_TOKEN_OFFSET,
///     fields: { next_instr: int @ NEXT_INSTR_OFFSET, ... },
///     arrays: {
///         locals_w: ref @ LOCALS_OFFSET,
///         stack: int @ (DATA_PTR_OFFSET + SLOT_OFFSET) {
///             ptr_offset: 0,
///             length_offset: LENGTH_OFFSET_MINUS_DATA_PTR,
///             items_offset: 0,
///         },
///     },
/// }
/// ```
pub struct VirtualizableDecl {
    /// Expression for the virtualizable variable in the mainloop body.
    pub var_name: Ident,
    /// Constant path for the vable_token field offset.
    pub token_offset: Path,
    /// Static fields: name, type (int/ref/float), byte offset constant.
    pub fields: Vec<VableFieldDecl>,
    /// Array fields: name, item type (int/ref/float), byte offset constant.
    pub arrays: Vec<VableArrayDecl>,
}

/// A single virtualizable static field declaration.
pub struct VableFieldDecl {
    /// Field name as it appears in the struct.
    pub name: Ident,
    /// Field type: `int`, `ref`, or `float`.
    pub field_type: Ident,
    /// Byte offset expression (e.g., `PYFRAME_NEXT_INSTR_OFFSET`,
    /// `STORAGEPOOL_LENGTHS_OFFSET + 8`).
    pub offset: Expr,
}

/// A single virtualizable array field declaration.
pub struct VableArrayDecl {
    /// Array field name as it appears in the struct.
    pub name: Ident,
    /// Item type: `int`, `ref`, or `float`.
    pub item_type: Ident,
    /// Physical layout of the array field inside the virtualizable object.
    pub layout: VableArrayLayoutDecl,
}

/// Layout description for a virtualizable array field.
pub enum VableArrayLayoutDecl {
    /// Direct pointer field: the virtualizable stores a pointer to the
    /// array object. `length_offset` and `items_offset` describe where,
    /// relative to that pointer, the length prefix and items begin.
    /// Matches RPython `Ptr(GcArray(T))`: `length_offset = 0`,
    /// `items_offset = size_of::<usize>()` (8 on 64-bit).
    ///
    /// `None` defaults to `0` for back-compat with pointer-to-items shapes
    /// where items sit directly at the field pointer.
    Direct {
        field_offset: Expr,
        length_offset: Option<Expr>,
        items_offset: Option<Expr>,
    },
    /// Embedded container layout: pointer/length live in sibling fields or
    /// an inline container relative to the declared field offset.
    Embedded {
        field_offset: Expr,
        ptr_offset: Expr,
        length_offset: Expr,
        items_offset: Expr,
    },
}

/// State field declaration for register/tape machines.
///
/// Syntax: `state_fields = { a: int, regs: [int], ... }`
///
/// Current implementation supports only `int` and `[int]`.
pub struct StateFieldsConfig {
    pub fields: Vec<StateFieldDecl>,
}

/// A single state field declaration.
pub struct StateFieldDecl {
    pub name: Ident,
    pub kind: StateFieldKind,
}

/// Whether a state field is a scalar, array, virtualizable array, or
/// an opaque pass-through (RPython parity: a field that lives on the state
/// struct but is invisible to the JIT — accessed only through explicit
/// interpreter code or via raw-pointer handles. Used for `Storage`-like
/// pools with polymorphic dispatch that can not be flattened as ints).
pub enum StateFieldKind {
    /// Scalar value.
    ///
    /// Syntax: `a: int` (default i64 storage) or `a: int(usize)` /
    /// `a: int(i32)` to declare a different Rust storage type. RPython
    /// parity: `lltype.Signed` is i64 word-sized, `lltype.Unsigned` is
    /// usize word-sized — both render to a single Int register in IR
    /// but the user-visible struct field carries the declared type. The
    /// macro emits `as i64` / `as <type>` casts at the JIT boundary so
    /// IR sees a uniform i64 while the interpreter keeps natural Rust
    /// types (e.g. aheui's `selected: usize`, `stacksize: i32`).
    Scalar {
        ir_type: Ident,
        /// `None` ⇒ Rust storage type matches `ir_type` (default i64
        /// for `int`). `Some(path)` ⇒ explicit override from
        /// `int(<path>)` syntax.
        rust_type: Option<syn::Path>,
    },
    /// Array value (e.g., `regs: [int]`) — flattened into inputargs.
    Array(Ident),
    /// Virtualizable array (e.g., `tape: [int; virt]`) — NOT flattened.
    /// Only the data pointer and length are tracked as inputargs.
    /// Element access emits GETARRAYITEM_RAW_I / SETARRAYITEM_RAW IR ops.
    VirtArray(Ident),
    /// Opaque pass-through (e.g., `storage: opaque(Storage)`) — the field
    /// keeps its declared type on the state struct but is NOT enumerated
    /// as a JIT inputarg, fail_arg, or sym slot. The codegen skips it
    /// entirely; the interpreter is responsible for accessing it directly.
    /// Used to carry pools/handles whose internal layout is not flat ints
    /// (e.g. polymorphic `Storage`). Pair with `opaque(Type)` raw-pointer
    /// handles passed via additional `int` scalars when the JIT needs to
    /// touch the underlying memory through `jit_promote!` + raw-load IR.
    #[allow(dead_code)]
    Opaque(syn::Path),
    /// Ref-typed scalar (e.g., `sel: ref(Stack)`) — a genuine `InputArgRef`
    /// threaded through the ref register bank (`registers_r`) rather than the
    /// int bank.  Storage on the state struct is a `usize` carrier (raw
    /// GcRef / pointer bits); the `ref(T)` type path `T` names the heap class
    /// the value points at.  Routing it through the ref bank lets the
    /// optimizer heap-cache `getfield_gc` off it (a stable ref box), unlike
    /// the cosmetic `input_arg_ref` virt-array pointer which flows through the
    /// int bank.
    #[allow(dead_code)]
    Ref(syn::Path),
}

/// One entry in the `calls = { ... }` / `helpers = [ ... ]` map.
#[derive(Clone)]
pub struct CallEntry {
    pub path: Path,
    pub policy: Option<CallPolicyKind>,
}

/// One entry in the `residual_writes = { <ref_scalar>.<field> [@ StructType] => [helpers] }`
/// map.  Each listed residual helper mutates `<ref_scalar>.<field>` through
/// opaque host code; the lowering attaches a write-set `EffectInfo` naming that
/// field so `OptHeap::force_from_effectinfo` invalidates the cached
/// `getfield_gc_i` after the call (the residual analogue of a traced mutator's
/// in-trace `setfield_gc` write barrier).  `ref_scalar` is resolved against
/// `state_ref_scalars` to recover the struct `Path` for `offset_of!` +
/// `struct_type_id`.
#[derive(Clone)]
pub struct ResidualWriteEntry {
    pub ref_scalar: Ident,
    pub field: Ident,
    /// Optional nominal struct layout alias.  This is needed when the raw
    /// pointer carried by `ref_scalar` type-puns several repr(C) layouts whose
    /// shared field offsets nevertheless have distinct field-descr identities.
    pub struct_type: Option<Path>,
    pub helpers: Vec<Path>,
}

/// A `ref(T)` state scalar that is the base of a contiguous raw-pointer array,
/// paired with the marker `getter` function whose call indexes it.  Declared as
/// `pool_arrays = { <ref> => <getter> [-> ElementType], ... }`.  The lowering
/// recognizes a pool read only when BOTH the call's function path matches
/// `getter` AND arg0 is `state.<base>` — operation identity, not arg shape
/// alone, so an unrelated helper that happens to take the same
/// `(state.<base>, int)` shape is not miscompiled into a `getarrayitem_gc_r`.
#[derive(Clone)]
pub struct PoolArrayEntry {
    pub base: Ident,
    pub getter: Path,
    /// The concrete element type each array slot points at (`pools: [*mut T; N]`),
    /// declared as `<base> => <getter> -> T`.  Supplies `struct_type` for the
    /// getter's ref binding so field access on a LOCAL `let x = <getter>(...)`
    /// resolves the layout (state-field refs get this from their decl instead).
    pub element_type: Option<Path>,
}

/// One entry in `ref_fields = { Struct::field => PointeeType, ... }`.
/// Declares that `Struct.field` is a ref-kind (pointer) field whose pointee
/// is `PointeeType`.  The lowerer uses this to emit `getfield_gc_r` /
/// `setfield_gc_r` for the field access, and tags the resulting ref binding
/// with `pointee_type` so subsequent field access on it resolves the layout.
#[derive(Clone)]
pub struct RefFieldEntry {
    /// The struct that owns the field (e.g. `Stack`, `Node`).
    pub struct_type: Path,
    /// The field name within the struct (e.g. `head`, `next`).
    pub field: Ident,
    /// The struct the pointer points to (e.g. `Node`).
    pub pointee_type: Path,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallPolicyKind {
    ResidualVoid,
    ResidualVoidWrapped,
    /// `EF_CANNOT_RAISE` (`call.py:303 getcalldescr`'s non-elidable
    /// `else` branch).  Producers pick this when they statically know
    /// the callee cannot raise but is otherwise neither elidable nor
    /// loop-invariant — e.g. flat TLS / buffer shims.  Maps to
    /// [`CondCallEffectSlot::CannotRaise`].
    ResidualVoidCannotRaise,
    ResidualVoidCannotRaiseWrapped,
    MayForceVoid,
    MayForceVoidWrapped,
    ReleaseGilVoid,
    ReleaseGilVoidWrapped,
    LoopInvariantVoid,
    LoopInvariantVoidWrapped,
    ResidualInt,
    ResidualIntWrapped,
    /// `EF_CANNOT_RAISE` (`call.py:303 getcalldescr`'s non-elidable
    /// `else` branch) for int-returning residual helpers.  Mirrors
    /// the void-side `ResidualVoidCannotRaise` pair.  Producers pick
    /// this when the callee is statically known to be non-elidable
    /// AND cannot raise.
    ResidualIntCannotRaise,
    ResidualIntCannotRaiseWrapped,
    MayForceInt,
    MayForceIntWrapped,
    ReleaseGilInt,
    ReleaseGilIntWrapped,
    LoopInvariantInt,
    LoopInvariantIntWrapped,
    // `EF_ELIDABLE_CAN_RAISE` (call.py:297). Default elidable variant —
    // emits trailing `GUARD_NO_EXCEPTION`.
    ElidableInt,
    ElidableIntWrapped,
    // `EF_ELIDABLE_CANNOT_RAISE` (call.py:299). Skips the
    // `GUARD_NO_EXCEPTION` because `effectinfo.check_can_raise(False)`
    // is false for `extraeffect == 0`.
    ElidableIntCannotRaise,
    ElidableIntCannotRaiseWrapped,
    // `EF_ELIDABLE_OR_MEMORYERROR` (call.py:295). Same dispatch as
    // `_can_raise` but distinguishes memory-only failure modes.
    ElidableIntOrMemerror,
    ElidableIntOrMemerrorWrapped,
    ResidualRef,
    /// Like `ResidualRef` but stamps the descr EffectInfo with
    /// `PyreHelperKind::NurseryAlloc` (aheui inline nursery bump). Result is a
    /// real escaping Ref — no virtualization.
    NurseryAllocRef,
    ResidualRefWrapped,
    /// `EF_CANNOT_RAISE` for ref-returning residual helpers.
    /// Mirrors `ResidualIntCannotRaiseWrapped`; the unwrapped variant
    /// is absent because the inferred lower path cannot recover a
    /// static ref-return BindingKind from the policy byte alone
    /// (mirrors the existing `ResidualRefWrapped`-only shape).
    ResidualRefCannotRaiseWrapped,
    MayForceRefWrapped,
    // ReleaseGilRefWrapped intentionally absent: resoperation.py:1243-1244
    // (`# no such thing`) excludes CALL_RELEASE_GIL_R from the upstream
    // opcode table, so a `'release_gil_ref_wrapped'` policy could only
    // emit an IR op the optimizer/backend cannot consume.
    LoopInvariantRefWrapped,
    ElidableRefWrapped,
    ElidableRefCannotRaiseWrapped,
    ElidableRefOrMemerrorWrapped,
    ResidualFloatWrapped,
    /// `EF_CANNOT_RAISE` for float-returning residual helpers.
    /// Mirrors `ResidualIntCannotRaiseWrapped` / `ResidualRefCannotRaiseWrapped`.
    ResidualFloatCannotRaiseWrapped,
    MayForceFloatWrapped,
    ReleaseGilFloatWrapped,
    LoopInvariantFloatWrapped,
    ElidableFloatWrapped,
    ElidableFloatCannotRaiseWrapped,
    ElidableFloatOrMemerrorWrapped,
    InlineInt,
    InlineRef,
    InlineFloat,
    InlineVoid,
    /// `BC_INLINE_CALL` ('j' argcode) into a pipeline-built sub-jitcode
    /// resolved by name through the host's `__majit_pipeline_jitcode`, rather
    /// than a macro-generated `__majit_inline_jitcode_<name>` helper. Int /
    /// Ref / Float select the trailing-return register kind read back into the
    /// caller binding. `make_jitcodes()` (codewriter.py:89) builds the callee;
    /// the dispatch traces into it the same way `Inline*` does, only the
    /// jitcode source differs.
    InlinePipelineInt,
    InlinePipelineRef,
    InlinePipelineFloat,
    /// Calls the function on the concrete path only; the JIT-path lowerer
    /// emits no IR ops at all. Use for operations that have no JIT-visible
    /// side effects (e.g. returning a node to a free-list -- RPython's `del`
    /// generates zero IR ops).
    ConcreteOnlyVoid,
}

pub(crate) fn parse_call_policy_kind(kind: &Ident) -> Option<CallPolicyKind> {
    Some(match kind.to_string().as_str() {
        "residual_void" => CallPolicyKind::ResidualVoid,
        "residual_void_wrapped" => CallPolicyKind::ResidualVoidWrapped,
        "residual_void_cannot_raise" => CallPolicyKind::ResidualVoidCannotRaise,
        "residual_void_cannot_raise_wrapped" => CallPolicyKind::ResidualVoidCannotRaiseWrapped,
        "may_force_void" => CallPolicyKind::MayForceVoid,
        "may_force_void_wrapped" => CallPolicyKind::MayForceVoidWrapped,
        "release_gil_void" => CallPolicyKind::ReleaseGilVoid,
        "release_gil_void_wrapped" => CallPolicyKind::ReleaseGilVoidWrapped,
        "loopinvariant_void" => CallPolicyKind::LoopInvariantVoid,
        "loopinvariant_void_wrapped" => CallPolicyKind::LoopInvariantVoidWrapped,
        "residual_int" => CallPolicyKind::ResidualInt,
        "residual_int_wrapped" => CallPolicyKind::ResidualIntWrapped,
        "residual_int_cannot_raise" => CallPolicyKind::ResidualIntCannotRaise,
        "residual_int_cannot_raise_wrapped" => CallPolicyKind::ResidualIntCannotRaiseWrapped,
        "may_force_int" => CallPolicyKind::MayForceInt,
        "may_force_int_wrapped" => CallPolicyKind::MayForceIntWrapped,
        "release_gil_int" => CallPolicyKind::ReleaseGilInt,
        "release_gil_int_wrapped" => CallPolicyKind::ReleaseGilIntWrapped,
        "loopinvariant_int" => CallPolicyKind::LoopInvariantInt,
        "loopinvariant_int_wrapped" => CallPolicyKind::LoopInvariantIntWrapped,
        // `call.py:292-299 _canraise(op)` 3-way pick on the elidable
        // branch. `elidable_*` (no suffix) is the EF_ELIDABLE_CAN_RAISE
        // default; `_cannot_raise` / `_or_memerror` map to
        // EF_ELIDABLE_CANNOT_RAISE / EF_ELIDABLE_OR_MEMORYERROR.
        "elidable_int" => CallPolicyKind::ElidableInt,
        "elidable_int_wrapped" => CallPolicyKind::ElidableIntWrapped,
        "elidable_int_cannot_raise" => CallPolicyKind::ElidableIntCannotRaise,
        "elidable_int_cannot_raise_wrapped" => CallPolicyKind::ElidableIntCannotRaiseWrapped,
        "elidable_int_or_memerror" => CallPolicyKind::ElidableIntOrMemerror,
        "elidable_int_or_memerror_wrapped" => CallPolicyKind::ElidableIntOrMemerrorWrapped,
        "residual_ref" => CallPolicyKind::ResidualRef,
        "nursery_alloc_ref" => CallPolicyKind::NurseryAllocRef,
        "residual_ref_wrapped" => CallPolicyKind::ResidualRefWrapped,
        "residual_ref_cannot_raise_wrapped" => CallPolicyKind::ResidualRefCannotRaiseWrapped,
        "may_force_ref_wrapped" => CallPolicyKind::MayForceRefWrapped,
        // "release_gil_ref_wrapped" intentionally rejected per
        // resoperation.py:1243-1244 — see CallPolicyKind comment.
        "loopinvariant_ref_wrapped" => CallPolicyKind::LoopInvariantRefWrapped,
        "elidable_ref_wrapped" => CallPolicyKind::ElidableRefWrapped,
        "elidable_ref_cannot_raise_wrapped" => CallPolicyKind::ElidableRefCannotRaiseWrapped,
        "elidable_ref_or_memerror_wrapped" => CallPolicyKind::ElidableRefOrMemerrorWrapped,
        "residual_float_wrapped" => CallPolicyKind::ResidualFloatWrapped,
        "residual_float_cannot_raise_wrapped" => CallPolicyKind::ResidualFloatCannotRaiseWrapped,
        "may_force_float_wrapped" => CallPolicyKind::MayForceFloatWrapped,
        "release_gil_float_wrapped" => CallPolicyKind::ReleaseGilFloatWrapped,
        "loopinvariant_float_wrapped" => CallPolicyKind::LoopInvariantFloatWrapped,
        "elidable_float_wrapped" => CallPolicyKind::ElidableFloatWrapped,
        "elidable_float_cannot_raise_wrapped" => CallPolicyKind::ElidableFloatCannotRaiseWrapped,
        "elidable_float_or_memerror_wrapped" => CallPolicyKind::ElidableFloatOrMemerrorWrapped,
        "inline_int" => CallPolicyKind::InlineInt,
        "inline_ref" => CallPolicyKind::InlineRef,
        "inline_float" => CallPolicyKind::InlineFloat,
        "inline_void" => CallPolicyKind::InlineVoid,
        "inline_pipeline_int" => CallPolicyKind::InlinePipelineInt,
        "inline_pipeline_ref" => CallPolicyKind::InlinePipelineRef,
        "inline_pipeline_float" => CallPolicyKind::InlinePipelineFloat,
        "concrete_only_void" => CallPolicyKind::ConcreteOnlyVoid,
        _ => return None,
    })
}

impl Parse for JitInterpConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut env_type = None;
        let mut io_shims = None;
        let mut calls: Vec<CallEntry> = Vec::new();
        let mut auto_calls = None;
        let mut greens = None;
        let mut reds = None;
        let mut virtualizable_decl = None;
        let mut state_fields = None;
        let mut recover: Option<Path> = None;
        let mut recursive_entry: Option<Path> = None;
        let mut residual_writes: Vec<ResidualWriteEntry> = Vec::new();
        let mut pool_arrays: Vec<PoolArrayEntry> = Vec::new();
        let mut ref_fields: Vec<RefFieldEntry> = Vec::new();
        let mut call_returns: Vec<(Path, Path)> = Vec::new();
        let mut struct_allocs: Vec<(Path, Path)> = Vec::new();
        let mut headerless_structs: Vec<Path> = Vec::new();
        let mut native_int_binops: Vec<(Path, Ident)> = Vec::new();
        let mut native_tag_small: Vec<Path> = Vec::new();
        let mut split_dispatch = false;
        let mut switch_dispatch = false;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "state" => {
                    state_type = Some(input.parse::<Ident>()?);
                }
                "env" => {
                    env_type = Some(input.parse::<Ident>()?);
                }
                "io_shims" => {
                    io_shims = Some(parse_io_shim_map(input)?);
                }
                "calls" => {
                    calls.extend(parse_call_map(input)?);
                }
                "helpers" => {
                    calls.extend(parse_helpers_list(input)?);
                }
                "auto_calls" => {
                    auto_calls = Some(input.parse::<LitBool>()?.value);
                }
                "greens" => {
                    let specs = green_type_tag::parse_green_spec_list(input)?;
                    greens = Some(specs);
                }
                "reds" => {
                    if reds.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `reds`"));
                    }
                    reds = Some(parse_expr_list(input)?);
                }
                "virtualizable_fields" => {
                    virtualizable_decl = Some(parse_virtualizable_decl(input)?);
                }
                "state_fields" => {
                    state_fields = Some(parse_state_fields(input)?);
                }
                "recover" => {
                    recover = Some(input.parse::<Path>()?);
                }
                "recursive_entry" => {
                    recursive_entry = Some(input.parse::<Path>()?);
                }
                "residual_writes" => {
                    residual_writes = parse_residual_writes_map(input)?;
                }
                "pool_arrays" => {
                    pool_arrays = parse_pool_arrays_map(input)?;
                }
                "ref_fields" => {
                    ref_fields = parse_ref_fields_map(input)?;
                }
                "call_returns" => {
                    call_returns = parse_call_returns_map(input)?;
                }
                "struct_allocs" => {
                    struct_allocs = parse_call_returns_map(input)?;
                }
                "headerless_structs" => {
                    headerless_structs = parse_path_set(input)?;
                }
                "native_int_binops" => {
                    native_int_binops = parse_native_int_binops_map(input)?;
                }
                "native_tag_small" => {
                    native_tag_small = parse_native_tag_small_list(input)?;
                }
                "split_dispatch" => {
                    split_dispatch = input.parse::<LitBool>()?.value;
                }
                "switch_dispatch" => {
                    switch_dispatch = input.parse::<LitBool>()?.value;
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_interp parameter: `{other}`"),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        let state_type =
            state_type.ok_or_else(|| syn::Error::new(input.span(), "missing `state` parameter"))?;
        let env_type =
            env_type.ok_or_else(|| syn::Error::new(input.span(), "missing `env` parameter"))?;

        if state_fields.is_none() {
            return Err(syn::Error::new(
                input.span(),
                "missing `state_fields` parameter",
            ));
        }

        let greens_specs = greens.unwrap_or_default();
        let (green_exprs, green_type_tags): (Vec<Expr>, Vec<Option<green_type_tag::GreenTypeTag>>) =
            greens_specs
                .into_iter()
                .map(|spec| (spec.expr, spec.type_tag))
                .unzip();

        Ok(JitInterpConfig {
            state_type,
            env_type,
            io_shims: io_shims.unwrap_or_default(),
            calls,
            auto_calls: auto_calls.unwrap_or(false),
            greens: green_exprs,
            reds: reds.unwrap_or_default(),
            green_type_tags,
            virtualizable_decl,
            state_fields,
            recover,
            recursive_entry,
            residual_writes,
            pool_arrays,
            ref_fields,
            call_returns,
            struct_allocs,
            headerless_structs,
            native_int_binops,
            native_tag_small,
            split_dispatch,
            switch_dispatch,
        })
    }
}

/// Parse `residual_writes = { <ref_scalar>.<field> [@ StructType] => [helper, ...], ... }`.
/// Each map key is a `state` ref-scalar field path (`selected_ref.size`) and
/// the value is a bracketed list of residual helper function paths that mutate
/// it.  See [`ResidualWriteEntry`].
fn parse_residual_writes_map(input: ParseStream) -> syn::Result<Vec<ResidualWriteEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let ref_scalar: Ident = content.parse()?;
        content.parse::<Token![.]>()?;
        let field: Ident = content.parse()?;
        let struct_type = if content.peek(Token![@]) {
            content.parse::<Token![@]>()?;
            Some(content.parse::<Path>()?)
        } else {
            None
        };
        content.parse::<Token![=>]>()?;
        let helpers_content;
        bracketed!(helpers_content in content);
        let helpers: Punctuated<Path, Token![,]> =
            helpers_content.parse_terminated(Path::parse, Token![,])?;
        entries.push(ResidualWriteEntry {
            ref_scalar,
            field,
            struct_type,
            helpers: helpers.into_iter().collect(),
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

/// Parse `pool_arrays = { <ref> => <getter> [-> ElementType], ... }`.  Each
/// entry maps a `ref(T)` state scalar (the array base) to the marker function
/// whose call indexes it, optionally tagging the loaded element's struct type.
fn parse_pool_arrays_map(input: ParseStream) -> syn::Result<Vec<PoolArrayEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let base: Ident = content.parse()?;
        content.parse::<Token![=>]>()?;
        let getter: Path = content.parse()?;
        let element_type = if content.peek(Token![->]) {
            content.parse::<Token![->]>()?;
            Some(content.parse::<Path>()?)
        } else {
            None
        };
        entries.push(PoolArrayEntry {
            base,
            getter,
            element_type,
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

/// Parse `ref_fields = { Struct::field => PointeeType, ... }`.
/// Each entry declares that `Struct.field` is a ref-kind (pointer) field
/// whose pointee is `PointeeType`.  The path `Struct::field` is parsed as
/// a single `Path` and then split: the last segment is the field name, the
/// remaining prefix is the struct type.
/// Parse `call_returns = { func_path => StructType, ... }`.
pub(crate) fn parse_call_returns_map(input: ParseStream) -> syn::Result<Vec<(Path, Path)>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let func_path: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let return_type: Path = content.parse()?;
        entries.push((func_path, return_type));
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_native_int_binops_map(input: ParseStream) -> syn::Result<Vec<(Path, Ident)>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let func_path: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let opcode: Ident = content.parse()?;
        entries.push((func_path, opcode));
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_native_tag_small_list(input: ParseStream) -> syn::Result<Vec<Path>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let func_path: Path = content.parse()?;
        entries.push(func_path);
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

pub(crate) fn parse_path_set(input: ParseStream) -> syn::Result<Vec<Path>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        entries.push(content.parse::<Path>()?);
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

fn split_struct_field_path(full_path: Path) -> syn::Result<(Path, Ident)> {
    let mut segments: Vec<_> = full_path.segments.into_iter().collect();
    if segments.len() < 2 {
        return Err(syn::Error::new_spanned(
            &full_path.leading_colon,
            "entry must be `Struct::field`",
        ));
    }
    let field_seg = segments.pop().unwrap();
    let field = field_seg.ident;
    let struct_type = syn::Path {
        leading_colon: full_path.leading_colon,
        segments: segments.into_iter().collect(),
    };
    Ok((struct_type, field))
}

pub(crate) fn parse_ref_fields_map(input: ParseStream) -> syn::Result<Vec<RefFieldEntry>> {
    let content;
    braced!(content in input);
    let mut entries = Vec::new();
    while !content.is_empty() {
        let (struct_type, field) = split_struct_field_path(content.parse::<Path>()?)?;
        content.parse::<Token![=>]>()?;
        let pointee_type: Path = content.parse()?;
        entries.push(RefFieldEntry {
            struct_type,
            field,
            pointee_type,
        });
        let _ = content.parse::<Token![,]>();
    }
    Ok(entries)
}

fn parse_expr_list(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let content;
    bracketed!(content in input);
    let exprs: Punctuated<Expr, Token![,]> = content.parse_terminated(Expr::parse, Token![,])?;
    Ok(exprs.into_iter().collect())
}

/// Parse virtualizable_fields = { var: IDENT, token_offset: PATH, fields: { ... }, arrays: { ... } }
///
/// Parse `state_fields = { name: type, ... }` where type is `int`, `[int]`,
/// `[int; virt]`, or `opaque(TypePath)`.
fn parse_state_fields(input: ParseStream) -> syn::Result<StateFieldsConfig> {
    let content;
    braced!(content in input);
    let mut fields = Vec::new();

    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        let kind = if content.peek(syn::token::Bracket) {
            // Array: [int] or virtualizable: [int; virt]
            let inner;
            bracketed!(inner in content);
            let item_type: Ident = inner.parse()?;
            if item_type != "int" {
                return Err(syn::Error::new(
                    item_type.span(),
                    format!(
                        "state_fields array `{name}` uses unsupported item type `{item_type}`; \
                         only `int` is currently supported"
                    ),
                ));
            }
            if inner.peek(Token![;]) {
                inner.parse::<Token![;]>()?;
                let flag: Ident = inner.parse()?;
                if flag == "virt" {
                    StateFieldKind::VirtArray(item_type)
                } else {
                    return Err(syn::Error::new(
                        flag.span(),
                        format!("unknown array modifier `{flag}`, expected `virt`"),
                    ));
                }
            } else {
                StateFieldKind::Array(item_type)
            }
        } else if content.peek(Token![ref]) {
            // `ref(<TypePath>)` — a ref-typed scalar carried in the ref
            // register bank.  `ref` is a reserved keyword, so it is matched
            // with `Token![ref]` before the identifier-based scalar forms.
            // Storage on the state struct is a usize carrier; `T` names the
            // pointed-at heap class.
            content.parse::<Token![ref]>()?;
            let inner;
            syn::parenthesized!(inner in content);
            let type_path: syn::Path = inner.parse()?;
            StateFieldKind::Ref(type_path)
        } else {
            // Scalar forms: `int`, `int(<TypePath>)`, or `opaque(TypePath)`.
            let head: Ident = content.parse()?;
            if head == "opaque" {
                let inner;
                syn::parenthesized!(inner in content);
                let type_path: syn::Path = inner.parse()?;
                StateFieldKind::Opaque(type_path)
            } else if head == "int" {
                // RPython parity: optional `int(<TypePath>)` declares the
                // Rust storage type when it differs from i64.
                let rust_type = if content.peek(syn::token::Paren) {
                    let inner;
                    syn::parenthesized!(inner in content);
                    Some(inner.parse::<syn::Path>()?)
                } else {
                    None
                };
                StateFieldKind::Scalar {
                    ir_type: head,
                    rust_type,
                }
            } else {
                return Err(syn::Error::new(
                    head.span(),
                    format!(
                        "state_fields scalar `{name}` uses unsupported type `{head}`; \
                         supported: `int`, `int(<TypePath>)`, `opaque(TypePath)`"
                    ),
                ));
            }
        };

        fields.push(StateFieldDecl { name, kind });
        let _ = content.parse::<Token![,]>();
    }

    Ok(StateFieldsConfig { fields })
}

/// RPython equivalent: VirtualizableInfo construction from virtualizable.py
/// + jtransform.py's field-to-descriptor mapping.
fn parse_virtualizable_decl(input: ParseStream) -> syn::Result<VirtualizableDecl> {
    let content;
    braced!(content in input);

    let mut var_name = None;
    let mut token_offset = None;
    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    while !content.is_empty() {
        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "var" => {
                var_name = Some(content.parse::<Ident>()?);
            }
            "token_offset" => {
                token_offset = Some(content.parse::<Path>()?);
            }
            "fields" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let field_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    fields.push(VableFieldDecl {
                        name,
                        field_type,
                        offset,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            "arrays" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let item_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let field_offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    let layout = if inner.peek(syn::token::Brace) {
                        let layout_content;
                        braced!(layout_content in inner);
                        let mut ptr_offset = None;
                        let mut length_offset = None;
                        let mut items_offset = None;
                        while !layout_content.is_empty() {
                            let layout_key: Ident = layout_content.parse()?;
                            layout_content.parse::<Token![:]>()?;
                            match layout_key.to_string().as_str() {
                                "ptr_offset" => {
                                    ptr_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "length_offset" => {
                                    length_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "items_offset" => {
                                    items_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                other => {
                                    return Err(syn::Error::new(
                                        layout_key.span(),
                                        format!(
                                            "unknown virtualizable array layout parameter: `{other}`"
                                        ),
                                    ));
                                }
                            }
                            let _ = layout_content.parse::<Token![,]>();
                        }
                        if let Some(ptr_offset) = ptr_offset {
                            VableArrayLayoutDecl::Embedded {
                                field_offset,
                                ptr_offset,
                                length_offset: length_offset.ok_or_else(|| {
                                    syn::Error::new(
                                        inner.span(),
                                        "missing `length_offset` in embedded virtualizable array layout",
                                    )
                                })?,
                                items_offset: items_offset.ok_or_else(|| {
                                    syn::Error::new(
                                        inner.span(),
                                        "missing `items_offset` in embedded virtualizable array layout",
                                    )
                                })?,
                            }
                        } else {
                            VableArrayLayoutDecl::Direct {
                                field_offset,
                                length_offset,
                                items_offset,
                            }
                        }
                    } else {
                        VableArrayLayoutDecl::Direct {
                            field_offset,
                            length_offset: None,
                            items_offset: None,
                        }
                    };
                    arrays.push(VableArrayDecl {
                        name,
                        item_type,
                        layout,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown virtualizable_fields parameter: `{other}`"),
                ));
            }
        }
        let _ = content.parse::<Token![,]>();
    }

    let var_name = var_name
        .ok_or_else(|| syn::Error::new(content.span(), "missing `var` in virtualizable_fields"))?;
    let token_offset = token_offset.ok_or_else(|| {
        syn::Error::new(
            content.span(),
            "missing `token_offset` in virtualizable_fields",
        )
    })?;

    Ok(VirtualizableDecl {
        var_name,
        token_offset,
        fields,
        arrays,
    })
}

/// Parse `{ path::func => jit_func, ... }`.
fn parse_io_shim_map(input: ParseStream) -> syn::Result<Vec<(Path, Ident)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let shim: Ident = content.parse()?;
        map.push((func, shim));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `{ path::func, path::func => residual_int, ... }`.
///
/// Per-entry forms:
///   - `path::func`                (default policy)
///   - `path::func => policy_kind` (explicit policy)
fn parse_call_map(input: ParseStream) -> syn::Result<Vec<CallEntry>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        let policy = if content.peek(Token![=>]) {
            content.parse::<Token![=>]>()?;
            let kind: Ident = content.parse()?;
            Some(parse_call_policy_kind(&kind).ok_or_else(|| {
                syn::Error::new(
                    kind.span(),
                    "call policy must be a supported residual/may_force/release_gil/loopinvariant policy or inline_int/inline_ref/inline_float",
                )
            })?)
        } else {
            None
        };
        map.push(CallEntry { path: func, policy });
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `[func_a, func_b, func_c]` — shorthand helper list with auto-inferred policies.
fn parse_helpers_list(input: ParseStream) -> syn::Result<Vec<CallEntry>> {
    let content;
    bracketed!(content in input);
    let paths: Punctuated<Path, Token![,]> = content.parse_terminated(Path::parse, Token![,])?;
    Ok(paths
        .into_iter()
        .map(|p| CallEntry {
            path: p,
            policy: None,
        })
        .collect())
}

/// Reject a plain `[int]` state-array that is stored-to inside the traced
/// loop. Such an array is *loop-carried*: its elements live only in trace
/// registers and are NOT restored to the array on a guard deopt — they read
/// back as the pre-loop value, silently producing a wrong result. The
/// supported mechanism for a mutated, loop-carried array is `[int; virt]`,
/// whose stores write through to the heap-backing `Vec` that the deopt path
/// reads directly. A plain `[int]` array that is only *read* in the loop (or
/// only written before it) is loop-invariant and stays valid, so the check
/// fires solely on stores reached by the dispatch loop the lowerer traces.
fn validate_state_fields(config: &JitInterpConfig, func: &ItemFn) -> syn::Result<()> {
    let Some(sf) = &config.state_fields else {
        return Ok(());
    };
    let plain_arrays: std::collections::HashSet<String> = sf
        .fields
        .iter()
        .filter(|f| matches!(f.kind, StateFieldKind::Array(_)))
        .map(|f| f.name.to_string())
        .collect();
    if plain_arrays.is_empty() {
        return Ok(());
    }
    // Only stores reached by the traced dispatch loop are a hazard; init
    // stores before the loop are not traced.
    let Some(loop_body) = find_traced_loop_body(&func.block) else {
        return Ok(());
    };
    let mut finder = PlainArrayStoreFinder {
        plain_arrays: &plain_arrays,
        offender: None,
    };
    syn::visit::Visit::visit_block(&mut finder, loop_body);
    if let Some((name, span)) = finder.offender {
        return Err(syn::Error::new(
            span,
            format!(
                "state field `{name}` is a plain `[int]` array stored inside the traced loop, \
                 so it is loop-carried. A plain `[int]` element is held in a trace register and \
                 is not restored to the array when a guard deopts (it reads back as the pre-loop \
                 value, silently miscompiling). Declare it as `[int; virt]`: a virtualizable \
                 array writes through to the heap-backing Vec that the deopt path reads directly."
            ),
        ));
    }
    Ok(())
}

/// The `loop {}` / `while {}` body that contains the `jit_merge_point!()`
/// marker — the region the lowerer traces.
fn find_traced_loop_body(block: &syn::Block) -> Option<&syn::Block> {
    for stmt in &block.stmts {
        let syn::Stmt::Expr(expr, _) = stmt else {
            continue;
        };
        let body = match expr {
            Expr::Loop(l) => &l.body,
            Expr::While(w) => &w.body,
            _ => continue,
        };
        let has_merge_point = body
            .stmts
            .iter()
            .any(|s| matches!(s, syn::Stmt::Macro(m) if m.mac.path.is_ident("jit_merge_point")));
        if has_merge_point {
            return Some(body);
        }
    }
    None
}

struct PlainArrayStoreFinder<'a> {
    plain_arrays: &'a std::collections::HashSet<String>,
    offender: Option<(String, proc_macro2::Span)>,
}

impl<'a> PlainArrayStoreFinder<'a> {
    /// `state.<field>[_]` with `<field>` a plain array → the field name.
    fn plain_array_index_target(&self, left: &Expr) -> Option<String> {
        let Expr::Index(idx) = left else {
            return None;
        };
        let Expr::Field(f) = &*idx.expr else {
            return None;
        };
        if !matches!(&*f.base, Expr::Path(p) if p.qself.is_none() && p.path.is_ident("state")) {
            return None;
        }
        let syn::Member::Named(member) = &f.member else {
            return None;
        };
        let name = member.to_string();
        self.plain_arrays.contains(&name).then_some(name)
    }

    fn check_assign_target(&mut self, left: &Expr) {
        if self.offender.is_some() {
            return;
        }
        if let Some(name) = self.plain_array_index_target(left) {
            use syn::spanned::Spanned as _;
            self.offender = Some((name, left.span()));
        }
    }
}

impl<'ast, 'a> syn::visit::Visit<'ast> for PlainArrayStoreFinder<'a> {
    fn visit_expr_assign(&mut self, node: &'ast syn::ExprAssign) {
        self.check_assign_target(&node.left);
        syn::visit::visit_expr_assign(self, node);
    }

    fn visit_expr_binary(&mut self, node: &'ast syn::ExprBinary) {
        // Compound assignment (`state.regs[i] += 1`) is a store too.
        use syn::BinOp::*;
        if matches!(
            node.op,
            AddAssign(_)
                | SubAssign(_)
                | MulAssign(_)
                | DivAssign(_)
                | RemAssign(_)
                | BitXorAssign(_)
                | BitAndAssign(_)
                | BitOrAssign(_)
                | ShlAssign(_)
                | ShrAssign(_)
        ) {
            self.check_assign_target(&node.left);
        }
        syn::visit::visit_expr_binary(self, node);
    }
}

/// Main entry point: transform the function with JIT support.
pub fn transform_jit_interp(config: JitInterpConfig, func: ItemFn) -> TokenStream {
    if let Err(err) = validate_state_fields(&config, &func) {
        return err.to_compile_error();
    }
    let trace_fn = codegen_trace::generate_trace_fn(&config, &func);
    let state_impl = codegen_state::generate_jit_state(&config, &func);
    let merge_wrapper = generate_merge_wrapper(&config, &func);
    let transformed_fn = transform_function(&config, &func);

    quote! {
        #state_impl
        #trace_fn
        #merge_wrapper
        #transformed_fn
    }
}

/// Generate a `#[cold]` out-of-line wrapper for the merge_point call.
///
/// This keeps the mainloop hot path thin — only an `is_tracing()` flag check
/// appears inline, while the closure capture and tracing logic live here.
fn generate_merge_wrapper(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);
    let trace_fn_name = quote::format_ident!("__trace_{}", fn_name);
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    quote! {
        #[cold]
        #[inline(never)]
        #[allow(non_snake_case)]
        fn #merge_fn_name(
            __driver: &mut majit_metainterp::JitDriver<#state_type>,
            __env: &#env_type,
            __pc: usize,
        ) -> ::core::option::Option<(usize, ::std::vec::Vec<majit_metainterp::Value>)> {
            // Clone the dispatch JitCode Arc before the mutable
            // `merge_point` borrow so the closure can forward it to
            // `#trace_fn_name` without holding a `JitDriver` reference.
            let __dispatch_jitcode: Option<::std::sync::Arc<majit_metainterp::JitCode>> =
                __driver.dispatch_jitcode().cloned();
            __driver.merge_point(|__meta, __sym| {
                use majit_metainterp::JitCodeSym;
                if __sym.trace_started && __pc == __sym.loop_header_pc() {
                    if let Some(__ctx) = __meta.trace_ctx() {
                        __ctx.walk_final_pc = Some(__pc);
                    }
                    return majit_metainterp::TraceAction::CloseLoop;
                }
                // Slice X-D production wire-up: split-borrow the active
                // TraceCtx and a `jitcell_token_by_number` resolver so
                // the dispatcher's BC_CALL_ASSEMBLER_* path can route
                // through the production `Arc<JitCellToken>` rather
                // than the synth-Arc `_by_number_typed` fallback.  The
                // helper also hands over the #184 recursive-call seams
                // (green-key target resolver, inline decision, and the
                // `execute_token_raw` concrete executor) wired to the
                // production warmstate / backend.
                let __result = __meta
                    .with_trace_ctx_and_token_resolver(
                        |__ctx,
                         __resolve,
                         __rec_target,
                         __rec_decision,
                         __rec_exec,
                         __rec_exec_ref,
                         __rec_exec_float,
                         __rec_exec_void| {
                            let __runtime =
                                majit_metainterp::ClosureRuntimeWithResolver::new(
                                    |pc: usize| pc,
                                    __resolve,
                                    __rec_target,
                                    __rec_decision,
                                    __rec_exec,
                                    __rec_exec_ref,
                                    __rec_exec_float,
                                    __rec_exec_void,
                                );
                            #trace_fn_name(
                                __ctx,
                                __sym,
                                __env,
                                __pc,
                                &__runtime,
                                __dispatch_jitcode.as_ref(),
                            )
                        },
                    )
                    .expect("merge_point invariant: tracing must be Some");
                __sym.trace_started = true;
                // pyjitpl.py:2843 blackhole_if_trace_too_long — check
                // AFTER executing the step (RPython _interpret loop order).
                let __too_long = __meta
                    .trace_ctx()
                    .map(|__ctx| __ctx.is_too_long())
                    .unwrap_or(false);
                if __too_long {
                    majit_metainterp::debug::log_one("jit-abort", "trace too long, aborting");
                    return majit_metainterp::TraceAction::Abort;
                }
                __result
            });
            // Single-pass tracing: surface the walk-final (pc, reds) snapshot
            // the CloseLoop arm stashed onto the MetaInterp before draining the
            // ctx. `None` outside single-pass (the hot path) and whenever the
            // walk did not populate reds — in which case the caller's hook is
            // inert and the observer/replay path runs unchanged.
            __driver.take_single_pass_outcome()
        }
    }
}

/// Transform the original function: replace jit_merge_point!() and can_enter_jit!() markers.
fn transform_function(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    use syn::visit_mut::VisitMut;

    let vis = &func.vis;
    let sig = &func.sig;
    let attrs = &func.attrs;
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);

    // A field access through a `ref(T)` state scalar — `state.<ref>.<member>` —
    // lowers to a `getfield_gc_*` / `setfield_gc_*` on the JIT side, but the
    // concrete (non-JIT) function carries the ref as raw pointer bits (`usize`),
    // so the same source has no `.<member>` to resolve.  Rewrite each such
    // access in the concrete body to an unsafe deref of the raw pointer through
    // the declared `ref(T)` struct type, matching the heap object the JIT
    // getfield/setfield touches.
    struct RefFieldRewriter {
        // ref-scalar field name -> the `ref(T)` struct path `T`.
        ref_fields: std::collections::HashMap<String, syn::Path>,
        // `ref_fields` config entries: "StructLast::field" → pointee Path.
        // Used to determine which fields are ref-kind and propagate struct
        // type to local bindings.
        field_pointees: std::collections::HashMap<String, syn::Path>,
        // Local variable name -> struct type it points to (for ref bindings
        // returned from getfield_gc_r on state ref scalars or other locals).
        local_ref_types: std::collections::HashMap<String, syn::Path>,
        // `call_returns` entries: canonical func path segments → return struct type.
        // When `let x = func(...)` and func is in this map, `x` is recorded
        // as a local ref binding pointing to the declared struct type.
        call_returns: std::collections::HashMap<Vec<String>, syn::Path>,
        // `struct_allocs` entries: struct type segments → allocator function path.
        // When `let x = StructType { f0, f1 }` and StructType is in this map,
        // the concrete path rewrites the struct literal to `allocator(f0, f1)`.
        struct_allocs: std::collections::HashMap<Vec<String>, syn::Path>,
    }
    impl RefFieldRewriter {
        // For `e == state.<ref_scalar>`, return the `ref(T)` struct path.
        fn ref_struct_of_base(&self, e: &Expr) -> Option<syn::Path> {
            let Expr::Field(f) = e else { return None };
            if !matches!(&*f.base, Expr::Path(p) if p.path.is_ident("state")) {
                return None;
            }
            let syn::Member::Named(ref_name) = &f.member else {
                return None;
            };
            self.ref_fields.get(&ref_name.to_string()).cloned()
        }

        // For `e == <local_ident>`, return the struct type if it's a known
        // local ref binding.
        fn local_ref_struct_of_base(&self, e: &Expr) -> Option<syn::Path> {
            let Expr::Path(p) = e else { return None };
            let ident = p.path.get_ident()?;
            self.local_ref_types.get(&ident.to_string()).cloned()
        }

        // Given a struct path and a field name, check if the field is
        // declared as ref-kind in `field_pointees`.  Returns the pointee
        // type if so.
        fn field_pointee(&self, struct_path: &syn::Path, field_name: &str) -> Option<syn::Path> {
            let struct_last = struct_path.segments.last()?.ident.to_string();
            let key = format!("{}::{}", struct_last, field_name);
            self.field_pointees.get(&key).cloned()
        }

        // Record a local binding as pointing to a struct type.
        fn record_local_ref(&mut self, name: &str, struct_type: syn::Path) {
            self.local_ref_types.insert(name.to_string(), struct_type);
        }

        // Check if a call expression's function is in `call_returns`.
        fn call_return_type(&self, func: &Expr) -> Option<syn::Path> {
            let Expr::Path(p) = func else { return None };
            let segs: Vec<String> = p
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            self.call_returns.get(&segs).cloned()
        }
    }
    impl VisitMut for RefFieldRewriter {
        fn visit_expr_mut(&mut self, expr: &mut Expr) {
            // Write-through: `state.<ref>.<member> = <rhs>` ->
            // `unsafe { (*(state.<ref> as *mut T)).<member> = <rhs> }`.
            if let Expr::Assign(assign) = expr {
                if let Expr::Field(lhs) = &*assign.left {
                    // Unify state-ref and local-ref write-through.
                    let base_struct = self
                        .ref_struct_of_base(&lhs.base)
                        .or_else(|| self.local_ref_struct_of_base(&lhs.base));
                    if let Some(struct_path) = base_struct {
                        let base = (*lhs.base).clone();
                        let member = lhs.member.clone();
                        let member_name = match &lhs.member {
                            syn::Member::Named(id) => id.to_string(),
                            _ => String::new(),
                        };
                        let mut rhs = (*assign.right).clone();
                        self.visit_expr_mut(&mut rhs);
                        if let Some(pointee) = self.field_pointee(&struct_path, &member_name) {
                            // Ref-kind field → cast RHS from usize to *mut Pointee.
                            *expr = syn::parse_quote! {
                                unsafe { (*(#base as *mut #struct_path)).#member = #rhs as *mut #pointee }
                            };
                        } else {
                            *expr = syn::parse_quote! {
                                unsafe { (*(#base as *mut #struct_path)).#member = #rhs }
                            };
                        }
                        return;
                    }
                }
            }
            // Read: `state.<ref>.<member>` reads a mutable heap field live from
            // the object pointer. The walk is the sole executor, so the field
            // holds the current value; the JIT side records BC_GETFIELD_GC_I.
            if let Expr::Field(field) = expr {
                if let Some(struct_path) = self.ref_struct_of_base(&field.base) {
                    let base = (*field.base).clone();
                    let member = field.member.clone();
                    let member_name = match &field.member {
                        syn::Member::Named(id) => id.to_string(),
                        _ => String::new(),
                    };
                    let is_ref_field = self.field_pointee(&struct_path, &member_name).is_some();
                    if is_ref_field {
                        // Ref-kind field → cast to usize (JIT ref bank).
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member as usize
                                }
                            }
                        };
                    } else {
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member
                                }
                            }
                        };
                    }
                    return;
                }
                // Read on local ref binding: `<local>.<member>`.
                if let Some(struct_path) = self.local_ref_struct_of_base(&field.base) {
                    let base = (*field.base).clone();
                    let member = field.member.clone();
                    let member_name = match &field.member {
                        syn::Member::Named(id) => id.to_string(),
                        _ => String::new(),
                    };
                    let is_ref_field = self.field_pointee(&struct_path, &member_name).is_some();
                    if is_ref_field {
                        // Ref-kind field → cast result to usize (JIT ref bank).
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member as usize
                                }
                            }
                        };
                    } else {
                        // Int-kind field → value used as-is.
                        *expr = syn::parse_quote! {
                            {
                                let __majit_getfield_obj = #base;
                                unsafe {
                                    (*(__majit_getfield_obj as *const #struct_path)).#member
                                }
                            }
                        };
                    }
                    return;
                }
            }
            syn::visit_mut::visit_expr_mut(self, expr);
        }

        // Track local `let` bindings that produce ref-typed values from
        // ref field reads, so subsequent field access rewrites work.
        fn visit_local_mut(&mut self, local: &mut syn::Local) {
            // First, recurse into the init expression to rewrite any ref
            // field reads contained within.
            syn::visit_mut::visit_local_mut(self, local);

            // After rewriting, check if the init expression was a
            // state.<ref>.<member> or local.<member> read whose field is
            // ref-kind, and if so, record the local as a ref binding.
            //
            // The rewriter has already transformed the init expr, but we
            // can recover the original shape from the pattern: if the init
            // was `state.selected_ref.head` and `Stack::head => Node` is
            // in field_pointees, the local should be typed as `*mut Node`.
            //
            // Since VisitMut has already rewritten the init expr, we check
            // the LOCAL PATTERN instead and rely on a pre-pass that recorded
            // the binding name before rewriting.  We handle this via
            // visit_stmt_mut below.
        }

        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // Before the default visit rewrites the init expression, check
            // if this is a `let <ident> = <base>.<field>` where <base> is
            // a state ref scalar or a local ref binding, and the field is
            // declared ref-kind.
            if let syn::Stmt::Local(local) = stmt {
                if let Some(init) = &local.init {
                    if let Expr::Field(field) = &*init.expr {
                        let member_name = match &field.member {
                            syn::Member::Named(id) => Some(id.to_string()),
                            _ => None,
                        };
                        if let Some(member_name) = member_name {
                            // Check if base is state.<ref_scalar>
                            let struct_path = self
                                .ref_struct_of_base(&field.base)
                                .or_else(|| self.local_ref_struct_of_base(&field.base));
                            if let Some(struct_path) = struct_path {
                                if let Some(pointee) =
                                    self.field_pointee(&struct_path, &member_name)
                                {
                                    // This field is ref-kind → the local is a
                                    // ref binding pointing to pointee type.
                                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                                        self.record_local_ref(
                                            &pat_ident.ident.to_string(),
                                            pointee,
                                        );
                                    }
                                }
                            }
                        }
                    }
                    // `let x = func(...)` where func is in `call_returns` →
                    // record x as a local ref binding with the declared type.
                    if let Expr::Call(call) = &*init.expr {
                        if let Some(struct_type) = self.call_return_type(&call.func) {
                            if let syn::Pat::Ident(pat_ident) = &local.pat {
                                self.record_local_ref(&pat_ident.ident.to_string(), struct_type);
                            }
                        }
                    }
                }
            }
            // Rewrite struct literal inits on the concrete path:
            // `let x = StructType { f0: v0, f1: v1 }` where StructType is
            // in `struct_allocs` → `let x = allocator_func(v0, v1)`.
            // The JIT path handles struct literals via `lower_struct_value`
            // (New + SetfieldGc); this rewrite makes the concrete path
            // produce the same allocation through the runtime allocator.
            if let syn::Stmt::Local(local) = stmt {
                if let Some(init) = &mut local.init {
                    if let Expr::Struct(s) = &*init.expr {
                        let segs: Vec<String> = s
                            .path
                            .segments
                            .iter()
                            .map(|seg| seg.ident.to_string())
                            .collect();
                        if let Some(alloc_func) = self.struct_allocs.get(&segs).cloned() {
                            let field_args: Vec<Expr> =
                                s.fields.iter().map(|f| f.expr.clone()).collect();
                            init.expr = Box::new(syn::parse_quote! {
                                #alloc_func(#(#field_args),*)
                            });
                        }
                    }
                }
            }
            // Now run the default visitor which will rewrite the init expr.
            syn::visit_mut::visit_stmt_mut(self, stmt);
        }
    }

    let mut block = func.block.clone();
    if let Some(sf) = &config.state_fields {
        let ref_fields: std::collections::HashMap<String, syn::Path> = sf
            .fields
            .iter()
            .filter_map(|f| match &f.kind {
                StateFieldKind::Ref(p) => Some((f.name.to_string(), p.clone())),
                _ => None,
            })
            .collect();
        if !ref_fields.is_empty() {
            // Build field_pointees map from the macro's `ref_fields` config.
            let field_pointees: std::collections::HashMap<String, syn::Path> = config
                .ref_fields
                .iter()
                .map(|entry| {
                    let struct_last = entry
                        .struct_type
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default();
                    let key = format!("{}::{}", struct_last, entry.field);
                    (key, entry.pointee_type.clone())
                })
                .collect();
            let call_returns_map: std::collections::HashMap<Vec<String>, syn::Path> = config
                .call_returns
                .iter()
                .map(|(func, ret_type)| {
                    let segs: Vec<String> =
                        func.segments.iter().map(|s| s.ident.to_string()).collect();
                    (segs, ret_type.clone())
                })
                .chain(config.pool_arrays.iter().filter_map(|entry| {
                    let ret_type = entry.element_type.clone()?;
                    let segs: Vec<String> = entry
                        .getter
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect();
                    Some((segs, ret_type))
                }))
                .collect();
            let struct_allocs_map: std::collections::HashMap<Vec<String>, syn::Path> = config
                .struct_allocs
                .iter()
                .map(|(struct_path, alloc_func)| {
                    let segs: Vec<String> = struct_path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect();
                    (segs, alloc_func.clone())
                })
                .collect();
            RefFieldRewriter {
                ref_fields,
                field_pointees,
                local_ref_types: std::collections::HashMap::new(),
                call_returns: call_returns_map,
                struct_allocs: struct_allocs_map,
            }
            .visit_block_mut(&mut block);
        }
    }

    // Rewrite the function body, replacing marker macros
    let body = rewrite_body(
        &block,
        &merge_fn_name,
        &config.greens,
        &config.green_type_tags,
        config.recursive_entry.as_ref(),
    );

    quote! {
        #(#attrs)*
        #vis #sig {
            #body
        }
    }
}

/// Rewrite function body: replace jit_merge_point!() and can_enter_jit!() calls.
fn rewrite_body(
    block: &syn::Block,
    merge_fn_name: &Ident,
    default_greens: &[Expr],
    default_green_type_tags: &[Option<green_type_tag::GreenTypeTag>],
    recursive_entry: Option<&Path>,
) -> TokenStream {
    use syn::visit_mut::VisitMut;

    #[derive(Default, Clone)]
    struct MergePointArgs {
        driver: Option<Expr>,
        env: Option<Expr>,
        pc: Option<Expr>,
        /// Single-pass tracing opt-in handle: the mutable native `JitState`
        /// binding. Supplied as the first expr after `;`
        /// (`jit_merge_point!(driver, env, pc; state)`). When present, the
        /// expansion emits the gated post-walk transfer hook; when absent
        /// (the default `jit_merge_point!()` form), the expansion is the
        /// byte-identical observer/replay statement.
        state: Option<Expr>,
    }

    impl Parse for MergePointArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.is_empty() {
                return Ok(Self::default());
            }
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pc: Expr = input.parse()?;
            let mut state = None;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let tail: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
                // The first tail expr is the single-pass `state` handle; any
                // further exprs remain accepted-and-ignored (legacy form).
                state = tail.into_iter().next();
            }
            Ok(Self {
                driver: Some(driver),
                env: Some(env),
                pc: Some(pc),
                state,
            })
        }
    }

    struct CanEnterJitArgs {
        driver: Expr,
        target: Expr,
        state: Expr,
        env: Expr,
        pre_run: Expr,
        pc: Option<Expr>,
        stacksize: Option<Expr>,
        greens: Vec<Expr>,
    }

    impl Parse for CanEnterJitArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let target: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let state: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pre_run: Expr = input.parse()?;

            let mut pc = None;
            let mut stacksize = None;
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                pc = Some(input.parse::<Expr>()?);
                input.parse::<Token![,]>()?;
                stacksize = Some(input.parse::<Expr>()?);
            }

            let mut greens = Vec::new();
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let exprs: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
                greens = exprs.into_iter().collect();
            }

            Ok(Self {
                driver,
                target,
                state,
                env,
                pre_run,
                pc,
                stacksize,
                greens,
            })
        }
    }

    /// Emit a single `(i64, majit_ir::GreenType)` pair for one green
    /// expression, dispatching on the optional per-green type tag.
    /// Untagged (`None`) greens flow through the
    /// `<_ as GreenAsI64>::__green_repr` trait. Tagged greens emit
    /// explicit casts so
    /// `&str`-bearing greens carry `GreenType::Str` instead of being
    /// silently routed through the blanket `impl<T: ?Sized>` Ref impl.
    fn emit_green_repr(spec_expr: &Expr, tag: Option<green_type_tag::GreenTypeTag>) -> TokenStream {
        use green_type_tag::GreenTypeTag;
        match tag {
            // Untagged greens go through the `GreenAsI64` trait —
            // primitive types route to `GreenType::Int`, floats to
            // `GreenType::Float`, references to `GreenType::Ref`.  An
            // untagged `&str` lands on the blanket `impl<T: ?Sized>
            // GreenAsI64 for &T` and carries `GreenType::Ref` (raw
            // pointer identity).  Consumers wanting RPython STR /
            // UNICODE content semantics MUST tag explicitly via
            // `greens = [name: str]` / `greens = [name: unicode]`;
            // `&str` is ambiguous between `rstr.STR` (UTF-8 byte
            // string) and `rstr.UNICODE` (codepoint sequence) and the
            // macro cannot pick one without an explicit declaration.
            None => quote! {
                <_ as majit_ir::GreenAsI64>::__green_repr(#spec_expr)
            },
            Some(GreenTypeTag::Int) => quote! {
                ((#spec_expr) as i64, majit_ir::GreenType::Int)
            },
            Some(GreenTypeTag::Float) => quote! {
                {
                    let __green_f: f64 = (#spec_expr) as f64;
                    (__green_f.to_bits() as i64, majit_ir::GreenType::Float)
                }
            },
            Some(GreenTypeTag::Ref) => quote! {
                ((#spec_expr) as *const _ as *const () as usize as i64,
                 majit_ir::GreenType::Ref)
            },
            // ABI: the i64 is the address of a `'static` slot holding
            // a `&'static str`.  `majit_ir::value::default_str_eq` /
            // `default_str_hash` / `default_unicode_hash` dereference
            // the i64 as `*const &'static str` and read the fat
            // pointer (data + len) — RPython's `rstr.STR*` /
            // `rstr.UNICODE*` carry their length internally; pyre
            // mirrors that contract by storing the fat pointer at a
            // stable address rather than the bare data pointer.
            //
            // TODO: allocation lifetime divergence
            // from RPython, intentionally deferred):
            //
            //   * RPython: `rstr.STR*` is GC-allocated *once per
            //     JitCell* and naturally stable for the JitCell's
            //     lifetime.  `JitCell.greenargs[i]` holds the rstr
            //     pointer; subsequent merge-point hits for the same
            //     `(jitdriver, greenkey)` pass the existing pointer
            //     into `equal_whatever` / `hash_whatever`.  No
            //     re-allocation, GC frees the rstr when the cell
            //     dies.
            //
            //   * pyre: `&str` has no stable backing-storage address
            //     by default, so this macro emits a fresh slot via
            //     `Box::leak` *every merge-point hit* — not once per
            //     JitCell.  The GreenKey HashMap content-de-dupes via
            //     `default_str_eq` / `default_str_hash`, so multiple
            //     slots with the same content collapse to a single
            //     cache entry, but the slot leaks themselves grow
            //     unboundedly with merge-point hit count for
            //     long-running programs.
            //
            // A structural fix (per-JitCell owned-string field
            // instead of leaked slot, e.g. reshaping
            // `GreenKey::values` from `Vec<i64>` to a typed enum
            // carrying `Box<str>` for str/unicode greens, with the
            // macro emitting a temporary the cache promotes on
            // insertion) is a larger refactor and is
            // intentionally deferred — a global intern side-table
            // was rejected as non-orthodox (RPython does not
            // maintain one).  Functional behavior matches RPython
            // (content-keyed compare/hash); only the lifetime /
            // allocation profile differs.
            Some(GreenTypeTag::Str) => quote! {
                (
                    majit_ir::make_str_slot(#spec_expr),
                    majit_ir::GreenType::Str,
                )
            },
            Some(GreenTypeTag::Unicode) => quote! {
                (
                    majit_ir::make_str_slot(#spec_expr),
                    majit_ir::GreenType::Unicode,
                )
            },
        }
    }

    fn green_key_expr(
        target: &Expr,
        greens: &[Expr],
        green_type_tags: &[Option<green_type_tag::GreenTypeTag>],
    ) -> Option<TokenStream> {
        if greens.is_empty() {
            None
        } else {
            // Per-green dispatch through `majit_ir::GreenAsI64::__green_repr`
            // so the `(i64-bits, GreenType)` pair travels together for each
            // green expression. `with_types` builds the typed schema;
            // `warmstate.py:575 _green_args_spec` keys per-type
            // `equal_whatever`/`hash_whatever` off the green's lltype, so a
            // Ref-typed green must compare by pointer identity, a Float by
            // bit pattern, and an Int by raw value — collapsing all to
            // `GreenType::Int` (the previous `GreenKey::new` shape) made
            // Float / Ref greens equal under bit-equal Int values they
            // should not be equal to.
            //
            // `green_type_tags` is the lockstep
            // `Vec<Option<GreenTypeTag>>` carried alongside `greens`
            // (`JitInterpConfig.green_type_tags`).  Tagged greens
            // bypass the trait dispatch with explicit casts so str /
            // unicode greens carry `GreenType::Str` (warmstate.py:108-128
            // ll_streq / ll_strhash routing).  Untagged greens
            // (`None`) keep the trait path unchanged.
            let green_reprs: Vec<TokenStream> = greens
                .iter()
                .enumerate()
                .map(|(i, expr)| {
                    let tag = green_type_tags.get(i).copied().flatten();
                    emit_green_repr(expr, tag)
                })
                .collect();
            Some(quote! {
                {
                    let (__values, __types): (Vec<i64>, Vec<majit_ir::GreenType>) = vec![
                        <_ as majit_ir::GreenAsI64>::__green_repr(#target),
                        #(#green_reprs),*
                    ].into_iter().unzip();
                    majit_ir::GreenKey::with_types(__values, __types)
                }
            })
        }
    }

    struct MarkerRewriter {
        merge_fn_name: Ident,
        default_greens: Vec<Expr>,
        default_green_type_tags: Vec<Option<green_type_tag::GreenTypeTag>>,
        recursive_entry: Option<Path>,
    }

    impl VisitMut for MarkerRewriter {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // First recurse into children
            syn::visit_mut::visit_stmt_mut(self, stmt);

            // Check if this statement is a macro invocation
            if let syn::Stmt::Macro(stmt_macro) = stmt {
                let mac = &stmt_macro.mac;
                let path_str = mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");

                if path_str == "jit_merge_point" || path_str.ends_with("::jit_merge_point") {
                    let args =
                        syn::parse2::<MergePointArgs>(mac.tokens.clone()).unwrap_or_default();
                    let merge_fn = &self.merge_fn_name;
                    let driver = args.driver.unwrap_or_else(|| syn::parse_quote!(driver));
                    let env = args.env.unwrap_or_else(|| syn::parse_quote!(program));
                    let pc = args.pc.unwrap_or_else(|| syn::parse_quote!(pc));
                    // jit_merge_point!() in #[jit_interp] dispatch portals
                    // expands to a single merge_wrapper invocation.  The wrapper
                    // (generate_merge_wrapper) clones the dispatch JitCode Arc and calls
                    // `driver.merge_point(...)` exactly once — there is no additional
                    // outer merge-point hook.  The BC_JIT_MERGE_POINT(_C) IR op lives
                    // inside the dispatch JitCode body (lower_dispatch_body),
                    // not in the outer Rust source.
                    //
                    // RPython parity: source-level jit_merge_point() is a codewriter
                    // marker; the runtime hook is the JitCode IR op
                    // (interp_jit.py:88-90).
                    //
                    // The `is_tracing()` guard here is a hot-path short-circuit
                    // (avoids the cold `__merge_*` call when not tracing).  It does NOT
                    // add a second merge-point dispatch — `driver.merge_point` guards
                    // again internally, but the closure runs only once.
                    let new_tokens: TokenStream = if let Some(state) = &args.state {
                        // Single-pass close: after the walk, if the CloseLoop
                        // populated a walk-final (pc, reds) snapshot, transfer
                        // it into native state and resume the native loop at the
                        // close pc, skipping the body re-run. `#merge_fn` returns
                        // `None` whenever the walk did not populate reds.
                        quote! {
                            if #driver.is_tracing() {
                                let __mp_out = #merge_fn(&mut #driver, #env, #pc);
                                if let Some((__sp_pc, __sp_reds)) = __mp_out
                                {
                                    // Push the walk-mutated scalar state fields
                                    // (e.g. a walk-advanced `selected`) into
                                    // native `state`. The walk advances these on
                                    // the sym via BC_STORE_STATE_FIELD only;
                                    // native `state` is frozen at trace-start
                                    // (S_k), so without this the scalars stay
                                    // stale — and a `recover` that re-derives
                                    // caches FROM a scalar (stacksize from
                                    // selected) would then read the wrong index.
                                    // The values were captured off the still-live
                                    // sym inside the CloseLoop arm (the arm clears
                                    // the sym before returning, so they cannot be
                                    // re-read here) and stashed on the MetaInterp;
                                    // this consumes the stash. Must precede
                                    // `recover`, which refines storage-backed
                                    // caches from the current scalars. No-op with
                                    // no scalar state fields or when no CloseLoop
                                    // stashed values.
                                    #driver.writeback_scalar_state_fields(
                                        &mut #state,
                                    );
                                    // Push the walk-final loop-carried virt-array
                                    // element values into native `state` too. The
                                    // walk mutates the array on the trace-ctx
                                    // shadow only; native `state`'s array is
                                    // frozen at trace-start (synchronize_
                                    // virtualizable skips the RustVec write-back
                                    // during tracing). Without this the compiled-
                                    // loop seed (extract_live reads native state)
                                    // reflects the trace-start array and the loop
                                    // re-executes the peeled iteration, double-
                                    // firing any side-effecting residual. No-op
                                    // when the state has no virtualizable array.
                                    #driver.writeback_virt_array_state_fields(
                                        &mut #state,
                                    );
                                    // Transfer any loop-carried reds the walk
                                    // captured, then re-derive the
                                    // storage-backed cache fields (stacksize,
                                    // selected/storage refs) from the
                                    // walk-advanced shared storage via the
                                    // `recover` hook. `__sp_reds` is empty today
                                    // (merge-point red-operand slots don't map to
                                    // native state fields; the scalar write-back
                                    // above is the authoritative scalar source),
                                    // so this block is scaffolding for a future
                                    // JIT whose merge point carries real reds.
                                    if !__sp_reds.is_empty() {
                                        let __sp_meta = majit_metainterp::JitState::build_meta(
                                            &#state, __sp_pc, #env,
                                        );
                                        majit_metainterp::JitState::restore_values(
                                            &mut #state, &__sp_meta, &__sp_reds,
                                        );
                                    }
                                    majit_metainterp::JitState::recover_after_compiled_run(
                                        &mut #state,
                                    );
                                    #driver.log_single_pass_parity_resume(
                                        __sp_pc,
                                        &#state,
                                        #env,
                                    );
                                    #driver.arm_single_pass_label_entry_on_next_back_edge(
                                        &#state,
                                    );
                                    // RPython pyjitpl.py:3119-3123 ->
                                    // 3072-3091 parity: a successful translated
                                    // CloseLoop raises ContinueRunningNormally,
                                    // returning to the interpreter. The compiled
                                    // loop is entered later via can_enter_jit /
                                    // warmstate, not by immediate direct entry.
                                    #driver.discard_single_pass_resume();
                                    // A terminal dispatch return (Finish) means
                                    // the interpreted function has returned:
                                    // exit the native dispatch loop and run its
                                    // own post-loop return exactly once. Resuming
                                    // at the captured pc instead would re-enter
                                    // the loop body when the terminal opcode is
                                    // mid-program (its post-advance pc lands
                                    // before the program end). A CloseLoop
                                    // back-edge keeps interpreting from the
                                    // merge-point pc.
                                    if #driver.take_single_pass_finish() {
                                        break;
                                    }
                                    #pc = __sp_pc;
                                    continue;
                                }
                            }
                        }
                    } else {
                        quote! {
                            if #driver.is_tracing() {
                                #merge_fn(&mut #driver, #env, #pc);
                            }
                        }
                    };
                    *stmt =
                        syn::parse2(new_tokens).expect("failed to parse merge_point replacement");
                }
            }
        }

        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            // First recurse
            syn::visit_mut::visit_expr_mut(self, expr);

            // `recursive_portal_call!(driver, green0, green1, ...)` — the
            // concrete (non-JIT) path re-enters the portal via the declared
            // `recursive_entry` function, forwarding the greens positionally
            // (jitdriver declaration order).  The dispatch JitCode lowers the
            // same intrinsic to `BC_RECURSIVE_CALL_*`.
            if let Expr::Macro(em) = expr {
                let path_str = em
                    .mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if path_str == "recursive_portal_call"
                    || path_str.ends_with("::recursive_portal_call")
                {
                    let args = em
                        .mac
                        .parse_body_with(Punctuated::<Expr, Token![,]>::parse_terminated)
                        .expect("recursive_portal_call! takes `driver, green0, green1, ...`");
                    let mut iter = args.into_iter();
                    let _driver = iter
                        .next()
                        .expect("recursive_portal_call! requires a driver argument");
                    let greens: Vec<Expr> = iter.collect();
                    let entry = self.recursive_entry.as_ref().unwrap_or_else(|| {
                        panic!(
                            "recursive_portal_call! used but `#[jit_interp(..)]` declares no \
                             `recursive_entry = <fn path>` for the concrete fallback"
                        )
                    });
                    let new_tokens = quote! { #entry(#(#greens),*) };
                    *expr = syn::parse2(new_tokens)
                        .expect("failed to parse recursive_portal_call concrete fallback");
                    return;
                }
            }
        }

        fn visit_block_mut(&mut self, block: &mut syn::Block) {
            let mut new_stmts = Vec::new();
            let mut i = 0;
            while i < block.stmts.len() {
                let stmt = &block.stmts[i];

                // Check if this is can_enter_jit!(driver, target, state, env, pre_run, ...)
                if let syn::Stmt::Macro(stmt_macro) = stmt {
                    let mac = &stmt_macro.mac;
                    let path_str = mac
                        .path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");

                    if path_str == "can_enter_jit" || path_str.ends_with("::can_enter_jit") {
                        let tokens = &mac.tokens;
                        if let Ok(args) = syn::parse2::<CanEnterJitArgs>(tokens.clone()) {
                            let driver_expr = &args.driver;
                            let target_expr = &args.target;
                            let state_expr = &args.state;
                            let env_expr = &args.env;
                            let pre_run_expr = &args.pre_run;
                            let pc_expr = args
                                .pc
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(pc));
                            let _stacksize_expr = args
                                .stacksize
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(stacksize));
                            let (greens, green_type_tags) = if args.greens.is_empty() {
                                (
                                    self.default_greens.clone(),
                                    self.default_green_type_tags.clone(),
                                )
                            } else {
                                // RPython `can_enter_jit` and
                                // `jit_merge_point` reference the same
                                // `JitDriver` greens spec — the marker op
                                // args sit in fixed positional slots whose
                                // lltype is fixed by declaration
                                // (`warmstate.py:564 _green_args_spec`,
                                // `support.py:126 decode_hp_hint_args`
                                // asserts on count mismatch at translation
                                // time).  Positional inheritance of
                                // declaration tags therefore matches
                                // RPython parity: a `str`-tagged green
                                // stays Str-keyed through the override
                                // path, routing through the canonical
                                // slot ABI rather than silently falling
                                // to the blanket `GreenAsI64 for &T`
                                // (Ref / pointer identity) implementation.
                                //
                                // If a downstream override expression has
                                // a different Rust type than the
                                // declaration's tag indicates (e.g. tag
                                // `Int` with `&str` override), the
                                // emitted explicit cast (`(<expr>) as
                                // i64`) fails at compile time — fail-loud
                                // at the macro / type-check boundary
                                // rather than at runtime with a
                                // misshaped key.
                                //
                                // Marker arity is structurally fixed in
                                // RPython, so a count mismatch is a
                                // hard user error. pyre fails loud at
                                // proc-macro expansion (compile-time)
                                // with a clear message rather than
                                // silently falling back to an untagged
                                // trait path that would emit a misshaped
                                // key schema.
                                if args.greens.len() != self.default_green_type_tags.len() {
                                    panic!(
                                        "can_enter_jit! override green count {} does not match \
                                         the JitDriver declaration's green count {} — RPython \
                                         marker arity is fixed (rpython/jit/codewriter/support.py \
                                         decode_hp_hint_args asserts on mismatch). Fix the \
                                         override expression list to match the declaration's \
                                         `greens=[...]` arity.",
                                        args.greens.len(),
                                        self.default_green_type_tags.len(),
                                    );
                                }
                                (args.greens.clone(), self.default_green_type_tags.clone())
                            };
                            // compile.py:711 parity: back_edge returns
                            // Some(resume_pc) on guard failure (blackhole
                            // resume) or FINISH (loop header re-entry).
                            // state.restore_values already restores all
                            // state fields (including stacksize) from the
                            // compiled loop's exit state, so no explicit
                            // stacksize reset is needed here.
                            let back_edge: TokenStream = if let Some(green_key) =
                                green_key_expr(target_expr, &greens, &green_type_tags)
                            {
                                quote! {
                                    if let Some(__resume_pc) = #driver_expr.back_edge_structured(#green_key, #target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                        #pc_expr = __resume_pc;
                                        continue;
                                    }
                                }
                            } else {
                                quote! {
                                    if let Some(__resume_pc) = #driver_expr.back_edge(#target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                        #pc_expr = __resume_pc;
                                        continue;
                                    }
                                }
                            };
                            let parsed: syn::Stmt =
                                syn::parse2(back_edge).expect("failed to parse back_edge");
                            new_stmts.push(parsed);
                            i += 1;
                            continue;
                        }
                    }
                }

                let mut cloned = block.stmts[i].clone();
                self.visit_stmt_mut(&mut cloned);
                new_stmts.push(cloned);
                i += 1;
            }
            block.stmts = new_stmts;
        }
    }

    let mut cloned_block = block.clone();
    let mut rewriter = MarkerRewriter {
        merge_fn_name: merge_fn_name.clone(),
        default_greens: default_greens.to_vec(),
        default_green_type_tags: default_green_type_tags.to_vec(),
        recursive_entry: recursive_entry.cloned(),
    };
    rewriter.visit_block_mut(&mut cloned_block);

    let stmts = &cloned_block.stmts;
    quote! { #(#stmts)* }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn parse_helpers_list_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [helper_add, helper_sub, helper_mul]
        };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].path.segments.last().unwrap().ident.to_string(),
            "helper_add"
        );
        assert!(result[0].policy.is_none());
        assert_eq!(
            result[1].path.segments.last().unwrap().ident.to_string(),
            "helper_sub"
        );
        assert_eq!(
            result[2].path.segments.last().unwrap().ident.to_string(),
            "helper_mul"
        );
    }

    #[test]
    fn parse_helpers_list_empty() {
        let tokens: proc_macro2::TokenStream = parse_quote! { [] };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert!(result.is_empty());
    }

    #[test]
    fn parse_helpers_list_with_path() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [module::helper_a, helper_b]
        };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        // First has two path segments
        assert_eq!(result[0].path.segments.len(), 2);
        assert_eq!(result[0].path.segments[0].ident.to_string(), "module");
        assert_eq!(result[0].path.segments[1].ident.to_string(), "helper_a");
    }

    /// Wrapper to make `parse_helpers_list` testable via `syn::parse2`.
    struct HelpersListWrapper(Vec<CallEntry>);
    impl Parse for HelpersListWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_helpers_list(input)?))
        }
    }

    /// Wrapper to make `parse_call_map` testable via `syn::parse2`.
    struct CallMapWrapper(Vec<CallEntry>);
    impl std::fmt::Debug for CallMapWrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_list()
                .entries(self.0.iter().map(|e| {
                    e.path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default()
                }))
                .finish()
        }
    }
    impl Parse for CallMapWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_call_map(input)?))
        }
    }

    #[test]
    fn parse_call_map_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { foo, bar => may_force_int }
        };
        let result: Vec<CallEntry> = syn::parse2::<CallMapWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].path.segments.last().unwrap().ident.to_string(),
            "foo"
        );
        assert!(result[0].policy.is_none());
        assert_eq!(
            result[1].path.segments.last().unwrap().ident.to_string(),
            "bar"
        );
        assert_eq!(result[1].policy, Some(CallPolicyKind::MayForceInt));
    }

    #[test]
    fn parse_call_map_rejects_slot_effect_blocks() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { pop_only { reads: [stackpos] } }
        };
        let err = syn::parse2::<CallMapWrapper>(tokens).unwrap_err();
        assert!(
            err.to_string().contains("expected identifier")
                || err.to_string().contains("expected `,`"),
            "expected call-map parser to reject slot-effect block, got: {err}",
        );
    }

    struct VirtualizableWrapper(VirtualizableDecl);
    impl Parse for VirtualizableWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_virtualizable_decl(input)?))
        }
    }

    struct StateFieldsWrapper(StateFieldsConfig);
    impl Parse for StateFieldsWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_state_fields(input)?))
        }
    }

    #[test]
    fn parse_state_fields_accepts_opaque_pass_through() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                storage: opaque(test_fixtures::storage::Storage),
                selected: int,
                tape: [int; virt],
            }
        };
        let parsed = syn::parse2::<StateFieldsWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.fields.len(), 3);
        assert_eq!(parsed.fields[0].name.to_string(), "storage");
        match &parsed.fields[0].kind {
            StateFieldKind::Opaque(p) => {
                assert_eq!(p.segments.last().unwrap().ident.to_string(), "Storage");
            }
            _ => panic!("expected Opaque variant"),
        }
        assert!(matches!(
            parsed.fields[1].kind,
            StateFieldKind::Scalar { .. }
        ));
        assert!(matches!(
            parsed.fields[2].kind,
            StateFieldKind::VirtArray(_)
        ));
    }

    #[test]
    fn parse_state_fields_accepts_ref_scalar() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                a: int,
                sel: ref(crate::Stack),
            }
        };
        let parsed = syn::parse2::<StateFieldsWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.fields.len(), 2);
        assert!(matches!(
            parsed.fields[0].kind,
            StateFieldKind::Scalar { .. }
        ));
        match &parsed.fields[1].kind {
            StateFieldKind::Ref(p) => {
                assert_eq!(p.segments.last().unwrap().ident.to_string(), "Stack");
            }
            _ => panic!("expected Ref variant"),
        }
    }

    #[test]
    fn parse_state_fields_rejects_unknown_scalar_type() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { val: float }
        };
        let err = match syn::parse2::<StateFieldsWrapper>(tokens) {
            Ok(_) => panic!("expected parse error for unknown scalar type"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("supported: `int`, `int(<TypePath>)`, `opaque(TypePath)`"),
            "msg: {msg}"
        );
    }

    #[test]
    fn parse_virtualizable_decl_keeps_direct_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: { next_instr: int @ NEXT_INSTR_OFFSET },
                arrays: { locals_w: ref @ LOCALS_OFFSET },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Direct {
                field_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(quote::quote!(#field_offset).to_string(), "LOCALS_OFFSET");
                assert!(length_offset.is_none());
                assert!(items_offset.is_none());
            }
            VableArrayLayoutDecl::Embedded { .. } => {
                panic!("expected direct array layout");
            }
        }
    }

    #[test]
    fn parse_virtualizable_decl_supports_embedded_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: {},
                arrays: {
                    stack: int @ (STORAGEPOOL_DATA_PTRS_OFFSET + SLOT) {
                        ptr_offset: 0,
                        length_offset: STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET,
                        items_offset: 0,
                    },
                },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Embedded {
                field_offset,
                ptr_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(
                    quote::quote!(#field_offset).to_string(),
                    "STORAGEPOOL_DATA_PTRS_OFFSET + SLOT"
                );
                assert_eq!(quote::quote!(#ptr_offset).to_string(), "0");
                assert_eq!(
                    quote::quote!(#length_offset).to_string(),
                    "STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET"
                );
                assert_eq!(quote::quote!(#items_offset).to_string(), "0");
            }
            VableArrayLayoutDecl::Direct { .. } => {
                panic!("expected embedded array layout");
            }
        }
    }

    fn validate(config_tokens: proc_macro2::TokenStream, func: ItemFn) -> syn::Result<()> {
        let config: JitInterpConfig = syn::parse2(config_tokens).unwrap();
        validate_state_fields(&config, &func)
    }

    /// A plain `[int]` array stored inside the traced loop must be rejected:
    /// the loop-carried element is lost on a guard deopt.
    #[test]
    fn validate_rejects_loop_carried_plain_array() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut state = S { regs: vec![0; 3] };
                let mut pc: usize = 0;
                loop {
                    jit_merge_point!();
                    let op = program[pc];
                    match op {
                        0 => { state.regs[0] = state.regs[0] + 1; pc += 1; }
                        _ => return state.regs[0],
                    }
                }
            }
        };
        let err = validate(
            quote! { state = S, env = Bytecode, greens = [pc, program],
            state_fields = { regs: [int] } },
            func,
        )
        .expect_err("loop-carried plain [int] store must be rejected");
        assert!(err.to_string().contains("[int; virt]"));
        assert!(err.to_string().contains("regs"));
    }

    /// `[int; virt]` is the supported loop-carried form — accepted.
    #[test]
    fn validate_accepts_virt_array() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut state = S { regs: vec![0; 3] };
                let mut pc: usize = 0;
                loop {
                    jit_merge_point!();
                    let op = program[pc];
                    match op {
                        0 => { state.regs[0] = state.regs[0] + 1; pc += 1; }
                        _ => return state.regs[0],
                    }
                }
            }
        };
        validate(
            quote! { state = S, env = Bytecode, greens = [pc, program],
            state_fields = { regs: [int; virt] } },
            func,
        )
        .expect("virt array must be accepted");
    }

    /// A plain `[int]` array only *read* in the loop is loop-invariant and
    /// stays valid — no false positive.
    #[test]
    fn validate_accepts_read_only_plain_array() {
        let func: ItemFn = parse_quote! {
            fn mainloop(program: &Bytecode, threshold: u32) -> i64 {
                let mut state = S { regs: vec![0; 3] };
                let mut pc: usize = 0;
                let mut acc: i64 = 0;
                loop {
                    jit_merge_point!();
                    let op = program[pc];
                    match op {
                        0 => { acc += state.regs[0]; pc += 1; }
                        _ => return acc,
                    }
                }
            }
        };
        validate(
            quote! { state = S, env = Bytecode, greens = [pc, program],
            state_fields = { regs: [int] } },
            func,
        )
        .expect("read-only plain [int] array must be accepted");
    }
}
