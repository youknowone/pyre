/// Descriptor traits for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py (AbstractDescr)
/// and rpython/jit/backend/llsupport/descr.py.
///
/// Descriptors carry type metadata needed by the optimizer and backend
/// for field access, array access, function calls, and guard failures.
///
/// # Descr identity status (Unified-Descr Port follow-up)
///
/// **Done.** `Op::descr` is `Option<DescrRef>` (resoperation.rs:992) and
/// every resume-metadata variant that carries a descr now stores it as
/// `Option<DescrRef>` rather than a `descr_index: u32` handle:
///
/// * `RdVirtualInfo` variants (resoperation.rs:519-643).  `PartialEq`
///   (resoperation.rs:664-814) uses `opt_descr_ptr_eq` (Arc::ptr_eq),
///   matching `history.py:125 id(descr)`.
/// * `ExitVirtualLayout` (`majit-backend/src/lib.rs:370`) and
///   `ExitPendingFieldLayout` (`majit-backend/src/lib.rs:549`) compare
///   descrs via `opt_descr_ptr_eq` for the same reason.
/// * `ResolvedPendingFieldWrite` / `EncodedPendingFieldWrite`
///   (`majit-metainterp/src/resume.rs:1496/1523`) and
///   `PendingFieldLayoutSummary` (`majit-ir/src/resumedata.rs:137`) use
///   the canonical `resumedata::opt_descr_arc_ptr_eq`.
/// * `GuardPendingFieldEntry` (resoperation.rs:892) carries
///   `Option<DescrRef>` directly and intentionally has no `PartialEq`
///   impl: `PENDINGFIELDSTRUCT` (`resume.py:87-92`) is write-only
///   resume data that RPython never compares by value.
/// * The `descr_index: u32` retained on individual descriptors
///   (`get_descr_index`/`set_descr_index`) is now the pure serialization
///   handle assigned by `descr.py:28 v.descr_index = len(all_descrs)`
///   via `optimizer::ensure_descr_index`.
///
/// **Arc cycle audit (Phase D prereq B, 2026-05-19).**
///
/// | Edge | Direction | Strength | Status |
/// |---|---|---|---|
/// | `SimpleSizeDescr.all_fielddescrs` → `FieldDescr` | child | strong | OK |
/// | `SimpleFieldDescr.parent_descr` → `SizeDescr` | parent | **Weak** (descr.rs:3147, 3272) | OK — cycle broken |
/// | `SimpleFieldDescr.vinfo` → `VirtualizableInfo` | back | **Weak** (descr.rs:3154, 3281) | OK — cycle broken |
/// | `VirtualizableInfo._static_field_descrs` → `FieldDescr` | child | strong | OK (paired with Weak above) |
/// | `CallDescr → EffectInfo._readonly/write_descrs_*` → `FieldDescr/ArrayDescr` | uni | strong | OK — no back-ref |
/// | `SimpleArrayDescr.lendescr` → length `FieldDescr` (parent_descr=None) | uni | strong | OK — no back-ref |
/// | `SimpleArrayDescr.all_interiorfielddescrs` ↔ `SimpleInteriorFieldDescr.array_descr` | both | **strong both ways** | **CYCLE, accepted** |
///
/// The struct-array interior cycle (last row) mirrors PyPy's
/// `arraydescr.all_interiorfielddescrs = descrs` +
/// `InteriorFieldDescr.arraydescr = arraydescr`
/// (`descr.py:372-375` + `descr.py:388-391`).  Python tolerates it via
/// cycle-collecting GC; Rust's `Arc` does not.  In practice the descr
/// graph is process-lifetime pinned by `GcCache._cache_size` /
/// `_cache_array` / `_cache_interiorfield` (descr.rs:498/660/710) — the
/// global singleton `gc_cache()` keeps strong roots for every minted
/// descr, so dropping IR-side `DescrRef` clones never decrements the
/// last strong count, and the cycle never gets a chance to leak.  If a
/// future port ever needs per-test descr teardown, weaken
/// `SimpleInteriorFieldDescr.array_descr` to `Weak<dyn ArrayDescr>`
/// (parity: PyPy reaches the array descr via `gccache._cache_array`
/// rather than relying on the InteriorFieldDescr backreference for
/// identity).
///
/// **C (landed 2026-05-19).**  Trait-dispatch + structural keying
/// migration for the virtualizable / virtualref field paths:
///
/// * `optimize_setfield_gc`, `optimize_getfield_gc`, and
///   `resolve_array_source` read `op.descr.as_field_descr()?.offset()`
///   and `.is_typeptr()` directly (RPython `op.getdescr().offset` /
///   `heaptracker.py:66 name == 'typeptr'` parity).
/// * `optimize_(get|set)interiorfield_gc` extract the inner
///   `FieldDescr` via `as_interior_field_descr().field_descr()`
///   (`descr.py:388 InteriorFieldDescr.__init__` parity) — they no
///   longer panic for InteriorFieldDescr ops.
/// * `VirtualizableFieldState.fields` is now keyed by
///   `FieldDescr::index_in_parent()` (RPython
///   `info.AbstractStructPtrInfo._fields[fielddescr.get_index()]`,
///   `info.py:203-206`) instead of the packed
///   `FIELD_DESCR_TAG | offset << 4 | size << 1 | type_bits` u32.
///   The synthetic fallback at `init` assigns `1 + field_idx_in_vinfo`
///   for static slots and `1 + num_static + array_idx` for array
///   slots, matching `virtualizable.py:71-72 build_field_descr`.
/// * The pyre-only `FieldIndexDescr`, `make_field_index_descr`,
///   `virtualizable_field_index`, `extract_field_offset`,
///   `descr_index` helper, and `debug_assert_no_typeptr_in_virtual_fields`
///   are deleted.  The typeptr-exclusion invariant is enforced at
///   `optimize_setfield_gc`'s typeptr fold (RPython
///   `heaptracker.py:66-67`), not by a runtime assert.
/// * `VirtualRefInfo`'s pyre-only `descr_virtual_token: u32` /
///   `descr_forced: u32` / `descr_size: u32` placeholders and the
///   matching `virtualref.rs` `VREF_FIELD_*` packed constants are
///   retired.  RPython caches real `Arc<dyn FieldDescr>` /
///   `Arc<dyn SizeDescr>` on the equivalent struct
///   (`virtualref.py:40-42 cpu.fielddescrof`); pyre now does the same
///   with `VirtualRefInfo::{descr, descr_virtual_token, descr_forced}`.
///   The module-level vref descriptor constructors return the same
///   cached Arcs, so `OptVirtualize` emits SETFIELD_GC ops with the
///   same identities carried by `MetaInterp.virtualref_info`.
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock;
use std::sync::Weak;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

use crate::OpRef;
use crate::effectinfo::{DescrMintEntry, DescrMintSpec, DescrSetMember};
use crate::resoperation::{GuardPendingFieldEntry, RdVirtualInfo};
use crate::value::{Const, Type};
use serde::{Deserialize, Serialize};

/// Opaque reference to a descriptor, shared across the JIT pipeline.
pub type DescrRef = Arc<dyn Descr>;

/// Thin-pointer wrapper for descrs whose address is baked into JIT-emitted
/// code.  `history.py:109-114 AbstractDescr.{hide,show}` parity: PyPy bakes
/// the GCREF of an `AbstractFailDescr` directly and recovers via
/// `cast_gcref_to_instance(AbstractDescr, descr_gcref)` — a pure cast
/// because every RPython GCREF is a typed pointer.
///
/// Rust's `Arc<dyn Descr>` is a fat pointer (data + vtable).  Codegen can
/// only bake the data half, so reconstructing the fat Arc at recovery
/// time needs a side-channel for the vtable.  `FailDescrCell` is a
/// concrete-typed wrapper: `Arc<FailDescrCell>` is thin, so
/// `Arc::as_ptr(&cell) as *const () as usize` bakes a complete identity
/// and `Arc::from_raw(addr as *const FailDescrCell)` recovers it
/// without a registry.
///
/// The cell is the unit kept alive by
/// `CompiledLoopToken.asmmemmgr_gcreftracers` (`model.py:294`);
/// the inner `DescrRef` carries the dynamic trait dispatch.
pub struct FailDescrCell {
    pub descr: DescrRef,
}

impl FailDescrCell {
    pub fn wrap(descr: DescrRef) -> Arc<Self> {
        Arc::new(Self { descr })
    }
}

impl std::ops::Deref for FailDescrCell {
    type Target = dyn Descr + 'static;
    fn deref(&self) -> &(dyn Descr + 'static) {
        &*self.descr
    }
}

impl std::fmt::Debug for FailDescrCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FailDescrCell(cell={:p}, descr={:p})",
            self as *const Self,
            Arc::as_ptr(&self.descr) as *const ()
        )
    }
}

/// `history.py:113 AbstractDescr.show(cpu, descr_gcref)` parity.
///
/// # Safety
/// `addr` MUST be the address of a live `Arc<FailDescrCell>` whose
/// strong refcount is held by `CompiledLoopToken.asmmemmgr_gcreftracers`
/// (or an equivalent keep-alive collection) while the baked JIT code
/// references this address.  Calling with any other address — including
/// the address of a different concrete type — is undefined behavior.
pub unsafe fn recover_fail_descr_cell(addr: usize) -> Arc<FailDescrCell> {
    let ptr = addr as *const FailDescrCell;
    unsafe {
        Arc::increment_strong_count(ptr);
        Arc::from_raw(ptr)
    }
}

/// descr.py: GcCache dict keys.
///
/// RPython uses the actual lltype object (STRUCT, ARRAY_OR_STRUCT,
/// FuncType) as dict key — identity-based for type objects. In Rust
/// we use opaque u64 handles assigned by the host/translator, so two
/// distinct type definitions always get distinct keys even if they
/// share a name or layout. For call descriptors, PyPy uses a
/// structural tuple key (descr.py:665).
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum LLType {
    /// descr.py:109: cache[STRUCT].
    /// Opaque handle for a STRUCT/GcStruct type definition.
    /// The host assigns a unique u64 per distinct type.
    Struct(u64),
    /// descr.py:350: cache[ARRAY_OR_STRUCT].
    /// Opaque handle for an ARRAY/GcArray type definition.
    Array(u64),
    /// descr.py:665: (arg_classes, result_type, result_signed,
    ///   RESULT_ERASED, extrainfo).
    /// Two calls with the same signature and canonical EffectInfo identity
    /// share one CallDescr. `effectinfo.py:147-148` interns the EI before
    /// `descr.py:665` places that object in this tuple.
    Func {
        arg_classes: String,
        result_type: Type,
        /// `descr.py:665` keys on the RAW result char, so `'S'` (singlefloat)
        /// and `'L'` do not share a descr with the `'i'`/`'f'` they normalise
        /// to.  Callers holding only a `Type` pass the char that `Type`
        /// derives, which leaves their key partition unchanged.
        result_class: char,
        /// descr.py:664: result_signed = get_type_flag(RESULT) == FLAG_SIGNED
        result_signed: bool,
        /// descr.py:662: result_size = symbolic.get_size(RESULT_ERASED, tsc)
        result_size: usize,
        /// `descr.py:665` stores the canonical `EffectInfo` object in the key.
        /// `effectinfo.py:147-148` interns ordinary EIs process-globally, so
        /// object identity carries all six raw sets without copying them into
        /// every call-cache key. Release-gil EIs receive a fresh canonical
        /// cell and therefore a fresh identity as well.
        effect_info_identity: usize,
    },
}

impl LLType {
    /// descr.py:109: cache[STRUCT] — STRUCT type identity.
    pub fn struct_key(type_id: u64) -> Self {
        LLType::Struct(type_id)
    }
    /// descr.py:350: cache[ARRAY_OR_STRUCT] — array type identity.
    pub fn array_key(type_id: u64) -> Self {
        LLType::Array(type_id)
    }
}

/// Path-stable hash for `LLType::Struct` / `LLType::Array` identity.
///
/// RPython compares `lltype.Struct` objects by Python-object identity:
/// `cpu.fielddescrof(STRUCT, name)` returns the same `FieldDescr` Arc
/// regardless of which translator pass triggered the mint, because both
/// passes resolve `STRUCT` to the same `lltype.Struct` instance through
/// the shared `RPythonTyper` type registry. Pyre lacks RPython's type
/// registry; the analyzer-time codewriter and the runtime
/// `__majit_register_descrs` macro both need a path-stable key that
/// maps `module_path::StructName` to a single `u64` — the macro
/// emits `path_hash(concat!(module_path!(), "::", stringify!(Struct)))`
/// at expansion time, the analyzer computes the same hash from
/// `field.owner_root`, and both routes converge on the same
/// `LLType::Struct(hash)` cache key.
///
/// Determinism is within-process only — `DefaultHasher::new()` returns a
/// hasher with fixed SipHash keys at construction
/// (`hashmap.rs::DefaultHasher::default`); hashing the same string
/// twice in one process gives the same `u64`. Cross-process / cross-build
/// stability is *not* required: every `LLType::Struct(_)` lookup happens
/// against `_cache_field` etc. which are also process-local.
pub fn path_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

/// Whether a struct SizeDescr still carries the truncated low word of its
/// serialized structural identity instead of a collector-issued type id.
///
/// The `BhDescr::Size` wire slot carries the full `cache_key`. A value above
/// the u32 range cannot itself be a dense collector id; if resolving that key
/// only recovers its truncated low word, the layoutbuilder registration step
/// has not stamped the shared descriptor yet.
pub fn struct_tid_is_unresolved(serialized_cache_key: u64, resolved_tid: u32) -> bool {
    serialized_cache_key > u32::MAX as u64 && resolved_tid == serialized_cache_key as u32
}

/// `path_hash` sibling that drops the leading `<crate>::` segment from
/// `module_path` before hashing.  PyPy/RPython has no notion of a crate
/// boundary — `lltype.Struct` identity is keyed on the Python module
/// path alone (`descr.py:105 cache[STRUCT]`).  Pyre's `module_path!()`
/// macro produces the full `crate::module::sub::...` form, whereas the
/// analyzer-side `module_path_from_source_file` (and the hard-coded
/// `build_object_descr_group_with_def_path` def-paths) strip the crate
/// segment.  Stripping the crate here aligns the macro-emitted
/// `__majit_type_id()` with both, giving a single `path_hash` namespace
/// across analyzer / hard-coded runtime publish / generic `#[jit_struct]`
/// runtime publish.
///
/// `module_path` empty or single-segment (no `::`) → hash the struct
/// name alone (the segment IS the crate root; nothing to strip).
pub fn path_hash_stripped_crate(module_path: &str, struct_name: &str) -> u64 {
    let stripped_module = match module_path.split_once("::") {
        Some((_crate, rest)) => rest,
        None => "",
    };
    // `module_path!()` and `stringify!` spell a raw identifier with its
    // sigil (`module::r#struct`, `r#type`), while
    // `module_path_from_source_file` and the Charon-side idents the analyzer
    // hashes never carry one.  Strip it from every segment of both halves, so
    // a type declared with — or inside — a keyword-named ident lands in one
    // `path_hash` namespace instead of two.
    let struct_name = strip_raw_ident_sigils(struct_name);
    if stripped_module.is_empty() {
        path_hash(&struct_name)
    } else {
        path_hash(&format!(
            "{}::{struct_name}",
            strip_raw_ident_sigils(stripped_module)
        ))
    }
}

/// Every `::`-separated segment of `path` with its raw-identifier sigil
/// removed, borrowing when there is none to remove.  `#` cannot occur inside
/// an identifier, so an `r#` in the path is always the sigil.
fn strip_raw_ident_sigils(path: &str) -> std::borrow::Cow<'_, str> {
    if !path.contains("r#") {
        return std::borrow::Cow::Borrowed(path);
    }
    std::borrow::Cow::Owned(
        path.split("::")
            .map(|seg| seg.strip_prefix("r#").unwrap_or(seg))
            .collect::<Vec<_>>()
            .join("::"),
    )
}

/// Object-identity token for a STRUCT / enum-variant type definition —
/// the layout-layer analog of RPython's `lltype.Struct` Python-object
/// identity (`descr.py:109 cache[STRUCT]`).
///
/// RPython keys every field offset / size on the `lltype.Struct` OBJECT
/// (`symbolic.get_field_token(STRUCT, name)` is `@specialize.memo()` over
/// that object), so two distinct type definitions that happen to share a
/// leaf name or a field shape are simply two distinct objects producing
/// two independent tokens — a "shared leaf → single last-writer offset"
/// collision is structurally inexpressible.  Pyre's layout layer lacks a
/// live type object, so this token stands in for it: a `u64` minted from
/// the FULLY-QUALIFIED canonical path (`module::Type`), never a bare
/// leaf.  It wraps the SAME `path_hash` the descr layer already keys
/// `LLType::Struct` on (the macro-emitted `__majit_type_id` and the
/// analyzer converge on it), so the layout layer and the descr layer
/// share ONE identity by construction.
///
/// Mint ONLY from a qualified / already-canonical name (`from_canonical`);
/// minting from a bare leaf reintroduces the cross-module collision this
/// token exists to remove.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StructId(u64);

impl StructId {
    /// Mint from a canonical (qualified) struct / enum path such as
    /// `module::Type` or `module::Enum::Variant`.  The caller is
    /// responsible for passing a name that has already been resolved
    /// through `canonical_struct_name` (or is qualified at the source) —
    /// a bare leaf would collide across modules.
    pub fn from_canonical(canonical_path: &str) -> Self {
        StructId(path_hash(canonical_path))
    }

    /// Derive the identity of one concrete generic instantiation from the
    /// defining type's identity and its balanced `<...>` argument spelling.
    ///
    /// Rust monomorphizations do not share a physical layout: `Option<bool>`
    /// and `Option<i64>` may differ in both size and field representation.
    /// RPython's analogue is one distinct low-level `Struct` object per
    /// specialized representation.  Folding the argument list away therefore
    /// collapses distinct layout/descr identities.  The template id keeps path
    /// aliases converged; the argument suffix keeps monomorphizations apart.
    pub fn instantiate(self, generic_args: &str) -> Self {
        StructId(path_hash(&format!(
            "__majit_generic_struct__::{:016x}{generic_args}",
            self.0
        )))
    }

    /// The underlying path-stable `u64`, identical to the descr layer's
    /// `LLType::Struct(_)` key for the same canonical path.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Resolve a struct / enum-variant name in ANY spelling a consumer might
/// hold — full-crate `crate::mod::Type`, crate-stripped `mod::Type`, bare
/// `Type`, or the enum-variant analogs — back to the one canonical
/// [`StructId`].
///
/// RPython holds a live `lltype.Struct` object at every use site, so a
/// reference always denotes the one identity regardless of how its name
/// is spelled.  Pyre's layout consumers that only have a *string* (a
/// rendered field type, an `lltype.Struct._name`) resolve it through this
/// table — the layout-layer analog of dereferencing back to that object.
/// Origins that carry the full Charon `name_path()` at their source
/// (`FieldDescriptor.owner_id`, the per-field row token) attach the
/// `StructId` directly and never need this; this table backs the origins
/// that only have a string (e.g. `llmemory::FieldOffset`'s `st._name`).
///
/// Value is `Option<StructId>`: a bare-leaf spelling shared by two
/// distinct modules maps to `None` (ambiguous) so an unqualified lookup
/// resolves to no identity rather than silently binding the wrong type —
/// the string-keyed "last-writer-wins" collision made structurally
/// unrepresentable.
static STRUCT_ID_BY_NAME: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, Option<StructId>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Populate / replace the global name → [`StructId`] resolution table.
/// The front-end builds it once at populate, inserting every
/// spelling-variant of each type that maps unambiguously, and `None` for
/// any bare leaf two distinct modules share.
pub fn register_struct_ids(table: std::collections::HashMap<String, Option<StructId>>) {
    let mut guard = STRUCT_ID_BY_NAME.lock().unwrap();
    *guard = table;
}

/// Look up the canonical [`StructId`] for a struct / enum-variant name in
/// any spelling, stripping a leading reference / raw-pointer marker first
/// (`&T` / `&mut T` / `*const T` / `*mut T` → `T`).  Returns `None` when
/// the name is unknown or its bare leaf is cross-module-ambiguous.
pub fn struct_id_for_name(raw: &str) -> Option<StructId> {
    let s = raw
        .trim_start_matches("*const ")
        .trim_start_matches("*mut ")
        .trim_start_matches("&mut ")
        .trim_start_matches('&')
        .trim();
    let guard = STRUCT_ID_BY_NAME.lock().unwrap();
    if let Some(id) = guard.get(s).copied().flatten() {
        return Some(id);
    }
    if is_shaped_tuple_name(s) || is_shaped_array_name(s) {
        return None;
    }
    let generic_args = generic_args_span(s)?;
    let template = strip_generic_args(s);
    guard
        .get(template.as_ref())
        .copied()
        .flatten()
        .map(|id| id.instantiate(generic_args))
}

/// Resolve the defining/template identity of a name, intentionally erasing a
/// concrete generic argument list while retaining an enum-variant tail.
/// Annotation metadata such as `_immutable_fields_` belongs to the declared
/// class/template and uses this lookup; physical layouts use
/// [`struct_id_for_name`] instead.
pub fn struct_template_id_for_name(raw: &str) -> Option<StructId> {
    let s = raw
        .trim_start_matches("*const ")
        .trim_start_matches("*mut ")
        .trim_start_matches("&mut ")
        .trim_start_matches('&')
        .trim();
    let template = if is_shaped_tuple_name(s) || is_shaped_array_name(s) {
        std::borrow::Cow::Borrowed(s)
    } else {
        strip_generic_args(s)
    };
    STRUCT_ID_BY_NAME
        .lock()
        .unwrap()
        .get(template.as_ref())
        .copied()
        .flatten()
}

/// The first balanced generic argument group in `name`, including brackets.
/// A trailing enum variant (`Result<T>::Ok`) is deliberately excluded: the
/// template [`StructId`] already distinguishes the base and each variant.
fn generic_args_span(name: &str) -> Option<&str> {
    let start = name.find('<')?;
    let mut depth = 0usize;
    for (offset, byte) in name.as_bytes()[start..].iter().copied().enumerate() {
        match byte {
            b'<' => depth += 1,
            b'>' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&name[start..=start + offset]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Use-import resolver / module-aware canonicalisation table.
///
/// Maps `bare_struct_name → defining_module_path` as discovered by
/// the MIR front-end (`majit_translate::front::mir`) `struct_origins`
/// collection.  Once the
/// analyzer pipeline populates this registry, `canonical_struct_name`
/// can transform a bare token like `"W_IntObject"` into the canonical
/// `"intobject::W_IntObject"` that the runtime's
/// `build_object_descr_group_with_def_path` dual-publishes — bringing
/// analyzer-side `path_hash` into structural parity with PyPy's
/// `cache[STRUCT]` lltype-object identity (descr.py:108-118).
///
/// Global state mirrors the existing `gc_cache()` / descr-registry
/// pattern: per-process, written once at JIT-start, read freely.
/// Empty when the analyzer was invoked via the bare `parse_source`
/// entry (no module_path supplied); consumers then fall back to the
/// bare name, which still resolves through the runtime dual-publish's
/// simple-name slot.
static STRUCT_ORIGIN_REGISTRY: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, String>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Populate / replace the global `STRUCT_ORIGIN_REGISTRY` with a fresh
/// `(bare_name, defining_module_path)` table.  Analyzer calls this
/// once after `collect_program_metadata_pub` so subsequent
/// `canonical_struct_name` lookups see the resolver output.
pub fn register_struct_origins(origins: std::collections::HashMap<String, String>) {
    let mut guard = STRUCT_ORIGIN_REGISTRY.lock().unwrap();
    *guard = origins;
}

/// Canonicalise a struct name into its `defining_module_path::Bare`
/// form when the resolver has recorded an origin, otherwise return
/// the input unchanged.  Used at every `path_hash` site that hashes
/// a STRUCT identity so the cache key lands on the same slot the
/// runtime's qualified def-path dual-publish wrote to.
///
/// Already-qualified inputs (containing `::`) pass through verbatim
/// — they are presumed canonical (typically produced by upstream
/// `use foo::bar::Baz` path expansion that yields `foo::bar::Baz`
/// in the AST).
pub fn canonical_struct_name(name: &str) -> String {
    // A per-instantiation name carries a `<…>` argument suffix
    // (`Result<Tuple>`).  Canonicalize the bare base and reattach the
    // suffix verbatim so a leaf (`Result<Tuple>`) and a qualified
    // (`core::result::Result<Tuple>`) spelling normalise to one key,
    // exactly as their bare bases reconcile — the base never contains a
    // `<`, so the recursion bottoms out in the ungeneric path below.
    if let Some(lt) = name.find('<') {
        let (base, suffix) = name.split_at(lt);
        return format!("{}{}", canonical_struct_name(base), suffix);
    }
    if name.contains("::") {
        return name.to_string();
    }
    let guard = STRUCT_ORIGIN_REGISTRY.lock().unwrap();
    match guard.get(name) {
        Some(module_path) if !module_path.is_empty() => {
            format!("{}::{}", module_path, name)
        }
        _ => name.to_string(),
    }
}

/// Drop a trailing generic-argument suffix (`Foo<bar>` → `Foo`,
/// `m::Foo<a, b>` → `m::Foo`), returning the ungeneric root.
/// Truncates at the first `<`, so a nested argument's own `::` / `<`
/// never leaks into the root; a name with no `<` is returned unchanged.
///
/// A per-instantiation enum receiver is spelled `Result<bool>`, but the
/// discriminant→variant table is keyed by the bare template root
/// `Result` (one template per generic ADT).  The narrowing resolver
/// strips the suffix for that table lookup while the variant `ClassDef`
/// key keeps it.
/// Is `name` a tuple identity carrying its item shape (`Tuple<T, ...>`), as
/// opposed to the bare `Tuple` root?
///
/// `TupleRepr` creates a distinct `GcStruct` per item shape
/// (`rtyper/rtuple.py:115-125`), so `(A,)` and `(A, B)` are different
/// low-level struct objects.  Charon spells that synthetic identity as
/// `Tuple<T, ...>` because Rust's built-in tuple has no nominal `TypeDecl`,
/// which makes it the one generic spelling that must NOT collapse onto its
/// template the way `Result<T>::Ok` does.
pub fn is_shaped_tuple_name(name: &str) -> bool {
    name != "Tuple" && strip_instantiation_suffix(name) == "Tuple"
}

/// Is `name` an array identity carrying both its item type and fixed length
/// (`Array<T;N>`), as opposed to the bare `Array` root?
///
/// RPython's fixed-size array representation is a concrete low-level
/// allocation shape, so two item/length pairs must not share the template
/// identity used by ordinary nominal generic ADTs.
pub fn is_shaped_array_name(name: &str) -> bool {
    name != "Array" && strip_instantiation_suffix(name) == "Array"
}

pub fn strip_instantiation_suffix(name: &str) -> &str {
    match name.find('<') {
        Some(i) => &name[..i],
        None => name,
    }
}

/// Whether `spelling` is a list-shaped container type (`Vec<T>`,
/// `VecDeque<T>`, `&[T]`, `[T; N]`) that the annotator's
/// `project_pyre_field_type` maps to a `SomeList` element model rather
/// than a class instance.  The named-ADT root resolver
/// (`adt_node_class_root`) answers `None` for the core/std/alloc
/// container family, so a list-typed parameter would otherwise seed the
/// classdef-less `SomeInstance(None)` shell; this recognizer lets the
/// parameter seam route it through the list model instead.  Mirrors the
/// `Vec<` / `[` arms of `project_pyre_field_type`.
pub fn is_list_container_spelling(spelling: &str) -> bool {
    let stripped = spelling
        .trim()
        .trim_start_matches('&')
        .trim_start_matches("mut ")
        .trim_start_matches("*const ")
        .trim_start_matches("*mut ")
        .trim();
    stripped.starts_with("Vec<") || stripped.starts_with("VecDeque<") || stripped.starts_with('[')
}

/// Remove the first balanced generic-argument group from a (possibly
/// variant-qualified) name, preserving any trailing path segment:
/// `Result<Tuple>::Ok` → `Result::Ok`, `Result<Tuple>` → `Result`,
/// `m::Result<core::option::Option<i32>>::Ok` → `m::Result::Ok`.  A name
/// with no `<` is borrowed unchanged (no allocation).
///
/// Unlike [`strip_instantiation_suffix`] (which truncates at the first
/// `<`), this keeps the `::variant` tail a per-instantiation enum variant
/// key carries, so a `Result<Tuple>::Ok` field lookup resolves through the
/// bare `Result::Ok` rows the template registered.
pub fn strip_generic_args(name: &str) -> std::borrow::Cow<'_, str> {
    let Some(start) = name.find('<') else {
        return std::borrow::Cow::Borrowed(name);
    };
    let mut depth = 0usize;
    let mut end = None;
    for (i, b) in name.bytes().enumerate().skip(start) {
        match b {
            b'<' => depth += 1,
            b'>' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => {}
        }
    }
    match end {
        Some(e) => {
            let mut out = String::with_capacity(name.len() - (e + 1 - start));
            out.push_str(&name[..start]);
            out.push_str(&name[e + 1..]);
            std::borrow::Cow::Owned(out)
        }
        None => std::borrow::Cow::Borrowed(name),
    }
}

/// The arg-class char a `Type` derives, for callers that never saw the raw
/// codewriter char.  `'S'` and `'L'` normalise INTO `Type::Int`/`Type::Float`,
/// so this direction can only produce the four plain classes.
pub fn result_class_of(result_type: Type) -> char {
    match result_type {
        Type::Int => 'i',
        Type::Ref => 'r',
        Type::Float => 'f',
        Type::Void => 'v',
    }
}

impl LLType {
    /// descr.py:665: get_call_descr key tuple.
    pub fn func_key(
        arg_types: &[Type],
        result_type: Type,
        result_class: char,
        result_signed: bool,
        result_size: usize,
        effect: &Arc<crate::effectinfo::EffectInfoCell>,
    ) -> Self {
        let mut arg_classes = String::new();
        for t in arg_types {
            arg_classes.push(match t {
                Type::Int => 'i',
                Type::Ref => 'r',
                Type::Float => 'f',
                Type::Void => 'v',
            });
        }
        LLType::Func {
            arg_classes,
            result_type,
            result_class,
            result_signed,
            result_size,
            effect_info_identity: Arc::as_ptr(effect) as usize,
        }
    }
}

static NEXT_CALL_DESCR_HEAPCACHE_INDEX: AtomicU32 = AtomicU32::new(1_000_000_000);

/// Stable per-call-descr identity used by heapcache call-loop-invariant
/// slots.  The allocation is shared by every `GcCache._cache_call`
/// producer so a descriptor returned from `get_call_descr` and one
/// returned from metainterp's production factory have the same identity
/// guarantees.
pub fn next_call_descr_heapcache_index() -> u32 {
    NEXT_CALL_DESCR_HEAPCACHE_INDEX.fetch_add(1, Ordering::Relaxed)
}

/// Counters behind [`GcCache::field_position_census`]. Process-global like the
/// gccache itself; only ever incremented under its lock.
static FIELD_PARENT_ABSENT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FIELD_PARENT_EMPTY: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FIELD_INDEX_REDERIVED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FIELD_INDEX_UNRESOLVED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FIELD_MINT_TRACE_ENABLED: OnceLock<bool> = OnceLock::new();
static FIELD_MINT_BACKTRACE_ENABLED: OnceLock<bool> = OnceLock::new();

/// Which exit supplied one `compute_struct_size` result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StructSizePath {
    Layout,
    Heuristic,
    FieldsMissing,
}

/// Which source supplied one field offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldOffsetSource {
    ConcreteHit,
    TemplateHit,
    AccumulatorFallback,
}

macro_rules! define_field_mint_census {
    ($($name:ident),+ $(,)?) => {
        struct FieldMintCounters {
            $($name: std::sync::atomic::AtomicUsize,)+
        }

        static FIELD_MINT: FieldMintCounters = FieldMintCounters {
            $($name: std::sync::atomic::AtomicUsize::new(0),)+
        };

        /// Zero every field-mint counter, so a subsequent snapshot describes
        /// only the work done after this call. `pyre-jit-trace`'s build script
        /// runs code generation twice in one process under
        /// `PYRE_CODEGEN_DETERMINISM_CHECK=in-process`; without this the second
        /// generation's census would carry the first generation's increments.
        pub fn reset_field_mint_census() {
            $(FIELD_MINT.$name.store(0, std::sync::atomic::Ordering::Relaxed);)+
        }

        /// Release-safe field-mint census snapshotted across the build/runtime process
        /// boundary by `pyre-jit-trace`'s build script.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
        pub struct FieldMintCensus {
            $(pub $name: usize,)+
        }

        impl FieldMintCensus {
            pub fn fields(&self) -> [(&'static str, usize); [$(stringify!($name)),+].len()] {
                [$( (stringify!($name), self.$name), )+]
            }
        }

        impl std::ops::Add for FieldMintCensus {
            type Output = Self;

            fn add(self, rhs: Self) -> Self {
                Self {
                    $($name: self.$name + rhs.$name,)+
                }
            }
        }

        pub fn field_mint_census_snapshot() -> FieldMintCensus {
            FieldMintCensus {
                $($name: FIELD_MINT.$name.load(std::sync::atomic::Ordering::Relaxed),)+
            }
        }
    };
}

define_field_mint_census! {
    cache_hit_disagree,
    cache_hit_offset,
    cache_hit_size,
    cache_hit_type,
    cache_hit_immutability,
    cache_hit_virtualizable,
    cache_hit_index_in_parent,
    offset_layout_hit,
    offset_accumulator_fallback,
    struct_size_layout,
    struct_size_heuristic,
    struct_size_fields_missing,
    owner_id_registry_miss,
    fieldless_size_shell_mints,
    fieldless_size_shell_upgrades,
    ei_differing,
    ei_identical,
    ei_struct_size,
    ei_offset,
    ei_field_size,
    ei_field_type,
    ei_flag,
    ei_index_in_parent,
    ei_immutable,
    ei_quasi_immutable,
    offset_concrete_hit,
    offset_template_hit,
}

pub fn field_mint_trace_enabled() -> bool {
    *FIELD_MINT_TRACE_ENABLED.get_or_init(|| {
        std::env::var_os("MAJIT_FIELD_MINT_TRACE").as_deref() == Some(std::ffi::OsStr::new("1"))
    })
}

fn field_mint_backtrace_enabled() -> bool {
    *FIELD_MINT_BACKTRACE_ENABLED.get_or_init(|| std::env::var_os("RUST_BACKTRACE").is_some())
}

pub fn record_field_offset_source(source: FieldOffsetSource) {
    use std::sync::atomic::Ordering::Relaxed;
    match source {
        FieldOffsetSource::ConcreteHit => {
            FIELD_MINT.offset_layout_hit.fetch_add(1, Relaxed);
            FIELD_MINT.offset_concrete_hit.fetch_add(1, Relaxed);
        }
        FieldOffsetSource::TemplateHit => {
            FIELD_MINT.offset_layout_hit.fetch_add(1, Relaxed);
            FIELD_MINT.offset_template_hit.fetch_add(1, Relaxed);
        }
        FieldOffsetSource::AccumulatorFallback => {
            FIELD_MINT.offset_accumulator_fallback.fetch_add(1, Relaxed);
        }
    }
}

pub fn record_compute_struct_size_path(path: StructSizePath) {
    use std::sync::atomic::Ordering::Relaxed;
    match path {
        StructSizePath::Layout => FIELD_MINT.struct_size_layout.fetch_add(1, Relaxed),
        StructSizePath::Heuristic => FIELD_MINT.struct_size_heuristic.fetch_add(1, Relaxed),
        StructSizePath::FieldsMissing => {
            FIELD_MINT.struct_size_fields_missing.fetch_add(1, Relaxed)
        }
    };
}

pub fn record_field_owner_id_registry_miss() {
    FIELD_MINT
        .owner_id_registry_miss
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// One `FIELD_INDEX_UNRESOLVED` event, named, keyed so identical mints fold
/// into one row with a count.
///
/// `BTreeMap`, not `HashMap`: the whole point of the table is to be printed,
/// and a hash-ordered print makes two runs of the same program produce
/// different bytes for the same finding.
static FIELD_UNRESOLVED_NAMES: std::sync::Mutex<
    std::collections::BTreeMap<UnresolvedFieldMint, usize>,
> = std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Whether to build [`FIELD_UNRESOLVED_NAMES`] at all, read once.
///
/// The COUNT (`field_pos_unresolved`) is ungated and always accurate; only the
/// names cost anything, so they sit behind a knob like
/// `PYRE_SIZE_SHELL_OWNERS`. Read through this one function so the variable has
/// exactly one spelling in-tree — the collector lives here and the printer
/// lives in `pyre-jit-trace`, and two independently-checked names would let the
/// printer report "nothing unresolved" from a run that never recorded.
///
/// Unreachable on wasm32 for the ordinary reason: a
/// `wasm32-unknown-unknown` guest's `std::env` is permanently empty, so the
/// knob cannot be set there and this returns false however the host is
/// configured.
///
/// The names are out of reach there, but the COUNT is not. The guest's export
/// list (`pyre-wasm/src/lib.rs`) carries all eight `pyre_jit_field_pos_*`
/// counters, `field_pos_unresolved` among them, and `pyre-wasm-runner` asks for
/// all eight — so on wasm this hazard is measurable as a number, and only the
/// per-mint naming is unavailable. Read the count; do not infer it from the
/// names being absent.
///
/// Do not read a missing counter as a zero, and do not assume the runner
/// would object to one: it refuses a counter it ASKS for and cannot resolve,
/// which structurally cannot catch a counter nobody added to its list. A guard
/// against wrong entries is not a guard against missing ones — which is why the
/// four census dispositions had to be exported deliberately rather than waited
/// for.
pub fn field_position_unresolved_naming_enabled() -> bool {
    field_position_unresolved_limit().is_some()
}

/// How many rows the knob asks for: `MAJIT_FIELD_POS_UNRESOLVED=<n>`, or the
/// whole table for `1` / any non-numeric value. `None` when unset.
///
/// A cap is a parameter, not a constant. `size_shell_owner_sample`'s sibling
/// diagnostic hardcodes 24 and prints it against a count of 155 — so the rows
/// it shows are the alphabetically-first sixth of a `BTreeSet`, which is the
/// one thing a reader must not compute a proportion from. Whoever asks a census
/// for names is asking a question about the whole population; let them say how
/// much of it they want.
fn field_position_unresolved_limit() -> Option<usize> {
    static LIMIT: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let raw = std::env::var("MAJIT_FIELD_POS_UNRESOLVED").ok()?;
        Some(match raw.trim().parse::<usize>() {
            Ok(0) | Ok(1) | Err(_) => usize::MAX,
            Ok(n) => n,
        })
    })
}

/// A field mint whose name the parent's positional list does not contain.
///
/// Everything needed to decide WHERE the fix belongs is in the key, because
/// the layer is not derivable from the count:
/// * `label` — the descr's display name, `Owner.field` or the `T<type_id>.`
///   stand-in. Says which producer minted it.
/// * `lookup_key` — the string actually searched for (`field_key`), which is
///   not always the tail of `label`; a producer that spells the key one way
///   and the display name another is a different defect from one that ranks
///   the field wrongly.
/// * `caller_index` / `parent_len` — the un-arbitrated `index_in_parent` the
///   caller supplied, against the length of the list it would index. The pair
///   says whether the number is merely wrong or out of range. `None` is a
///   producer that never claimed a slot at all, which is a different finding
///   from one that claimed slot 0 — and while this field was a plain `usize`
///   the two were the same row, so the table could be read but not split.
/// * `parent_keys` — what the parent DOES list. This is the discriminator
///   between "the field is missing from its own parent" and "the parent is a
///   different struct that shares an identity key", and neither the count nor
///   the label can distinguish them.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct UnresolvedFieldMint {
    label: String,
    lookup_key: String,
    caller_index: Option<usize>,
    parent_len: usize,
    parent_keys: String,
}

/// `FIELD_INDEX_UNRESOLVED` split by whether the producer had claimed a slot.
///
/// Ungated, unlike the row table above: the rows are only recorded when
/// `MAJIT_FIELD_POS_UNRESOLVED` was set before the first mint, so a late-armed
/// gate yields a short table, and a split derived from a short table is a split
/// of the wrong denominator. These two are bumped on every unresolved mint and
/// sum to `FIELD_INDEX_UNRESOLVED` exactly.
static FIELD_INDEX_UNRESOLVED_CLAIMED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FIELD_INDEX_UNRESOLVED_PLACEHOLDER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counters behind [`GcCache::spec_position_census`] — the same question the
/// four above ask, put to the producer instead of to the reader.
static FIELD_SPEC_CHECKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FIELD_SPEC_MISPLACED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counters behind [`GcCache::attached_position_census`].
static FIELD_ATTACHED_CHECKED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FIELD_ATTACHED_MISPLACED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Counters behind [`GcCache::mint_index_census`].
static FIELD_MINT_INDEX_CLAIMED: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static FIELD_MINT_INDEX_PLACEHOLDER: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Whether a field mint's `index_in_parent` was WRITTEN by a lookup arm or is
/// still the initialiser the mint opened with.
///
/// No other counter in this module records where that value came from. The ones
/// that look at it at all compare its VALUE — `census_attached_index` against
/// the attached parent's slot, `census_spec_positions` against the submitted
/// list's own order, `derive_index_in_parent` against a name lookup — and the
/// rest report a disposition reached before any comparison happens. An
/// unwritten initialiser is the literal `0`, so a mint that never claimed a slot
/// and a mint that claims slot 0 are the same bytes in the same table to every
/// one of them, and no reader can separate them at any level of care.
///
/// This is the missing bit, taken at the only place that still holds it.
///
/// UNGUARDED BY DESIGN, and that is the point. The censuses that ask about a
/// mint's `index_in_parent` are conditional on a usable parent —
/// `census_attached_index`'s caller needs `Some(parent_spec)` and a resolvable
/// slot, and `derive_index_in_parent` diverts to `FIELD_PARENT_ABSENT` /
/// `FIELD_PARENT_EMPTY` before it reaches the comparison. In exactly the case
/// where the placeholder survives unrepaired FOR WANT OF A PARENT, both of them
/// are silent. Counting here, before any parent is consulted, is what reaches
/// that hole. Do not move this call inside a parent check.
///
/// What does NOT bump it, so the number is read against the right
/// population: only `codewriter::assembler::fielddescrof` calls this, so
/// `claimed + placeholder` is the count of `fielddescrof` mints and NOT of
/// field descrs in the program.
///
/// The vable-field and vable-array synthesizers in `blackhole.rs` also emit
/// `index_in_parent: 0` — but as a literal, beside `parent: None`, with no
/// lookup in front of it that could have written something else. There is no
/// provenance question to ask of a constant: those mints never had a parent
/// list to claim a slot in, so counting them would add rows to `placeholder`
/// that are not the hazard this pair exists to size.
pub fn census_mint_index_provenance(claimed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let counter = if claimed {
        &FIELD_MINT_INDEX_CLAIMED
    } else {
        &FIELD_MINT_INDEX_PLACEHOLDER
    };
    counter.fetch_add(1, Relaxed);
}

/// One field descr's two halves, compared against each other.
///
/// `expected` is the slot of the attached parent's `all_fielddescrs` that this
/// field's own offset occupies; `actual` is the `index_in_parent`
/// (`descr.py:228`) the producer stamped on it. Upstream cannot disagree —
/// `heaptracker.py:60-72` and `:96-112` are one walker — so every disagreement
/// is a descr declaring itself to be at a slot its parent fills with a
/// different field.
///
/// Caller supplies `expected`, because only it holds the parent the producer
/// actually attached. Once the descr reaches `get_field_descr` the parent is
/// whatever `_cache_size` holds for the key, which is a different question.
///
/// `actual` is an `Option` because the producer's slot claim now carries its own
/// provenance, but this pair deliberately folds `None` to `0` before comparing:
/// `0` is the exact value the descr used to carry in that state, so folding
/// keeps `[checked, misplaced]` numerically identical to every reading taken
/// before the widening. The provenance split belongs to the counters that were
/// added to answer it — [`census_mint_index_provenance`] on the mint side and
/// `FIELD_INDEX_UNRESOLVED_{CLAIMED,PLACEHOLDER}` on the resolve side — not
/// here, where redefining `misplaced` would silently move a published number.
pub fn census_attached_index(expected: usize, actual: Option<usize>) {
    use std::sync::atomic::Ordering::Relaxed;
    FIELD_ATTACHED_CHECKED.fetch_add(1, Relaxed);
    if expected != actual.unwrap_or(0) {
        FIELD_ATTACHED_MISPLACED.fetch_add(1, Relaxed);
    }
}

/// `field_specs[i].index_in_parent == i`, counted on the list as handed in.
///
/// This is [`GcCache::positional_invariant_census`]'s predicate moved from the
/// published list to the submitted one, and the move is the whole point.
/// `get_field_descr` re-derives every index against the parent before the list
/// is frozen, so a census taken afterwards reads the reader's answer; the
/// producer's own number survives nowhere else. Taken here it needs no parent,
/// no name and no cache — the caller supplies both halves of the claim.
///
/// Both factories feed this, so it also covers the `cache_key == 0` fresh-mint
/// groups that never enter `_cache_size` and are therefore outside
/// `positional_invariant_census`'s domain entirely.
fn census_spec_positions(field_specs: &[SimpleFieldDescrSpec]) {
    use std::sync::atomic::Ordering::Relaxed;
    FIELD_SPEC_CHECKED.fetch_add(field_specs.len(), Relaxed);
    let misplaced = field_specs
        .iter()
        .enumerate()
        .filter(|(i, spec)| spec.index_in_parent != *i)
        .count();
    if misplaced > 0 {
        FIELD_SPEC_MISPLACED.fetch_add(misplaced, Relaxed);
    }
}

/// descr.py:14-23 GcCache.
///
/// Per-type descriptor caches keyed by LLType (structural equality).
/// Factory functions (get_size_descr, get_field_descr, etc.) check
/// the cache first and return the existing object on hit.
///
/// setup_descrs() iterates caches in RPython's fixed order:
///   _cache_size, _cache_field, _cache_array, _cache_arraylen,
///   _cache_call, _cache_interiorfield
pub struct GcCache {
    /// descr.py:18: _cache_size[STRUCT]
    pub _cache_size: HashMap<LLType, DescrRef>,
    /// descr.py:19: _cache_field[STRUCT][fieldname]. Typed `Arc<SimpleFieldDescr>`
    /// per PyPy's concrete `FieldDescr` (descr.py:146 `class FieldDescr(ArrayOrFieldDescr)`).
    /// Concrete-Arc return enables analyzer-side `cc.fielddescrof_concrete`
    /// to cache-hit a previously-minted runtime `__majit_register_descrs`
    /// Arc without an `Arc<dyn Descr>` → `Arc<SimpleFieldDescr>` downcast.
    pub _cache_field: HashMap<LLType, HashMap<String, Arc<SimpleFieldDescr>>>,
    /// descr.py:20: _cache_array[ARRAY_OR_STRUCT]
    pub _cache_array: HashMap<LLType, DescrRef>,
    /// descr.py:21: _cache_arraylen[ARRAY_OR_STRUCT]
    pub _cache_arraylen: HashMap<LLType, DescrRef>,
    /// descr.py:22: _cache_call[(arg_classes, ...)]
    pub _cache_call: HashMap<LLType, DescrRef>,
    /// descr.py:23: _cache_interiorfield[(ARRAY, name, arrayfieldname)]
    pub _cache_interiorfield: HashMap<(LLType, String, String), DescrRef>,

    // ── Creation-order tracking ──
    // Rust HashMap iteration is non-deterministic. setup_descrs()
    // must iterate in creation order to match PyPy's dict iteration.
    // Each Vec records descriptors in insertion order.
    _cache_size_order: Vec<DescrRef>,
    _cache_field_order: Vec<DescrRef>,
    _cache_array_order: Vec<DescrRef>,
    _cache_arraylen_order: Vec<DescrRef>,
    _cache_call_order: Vec<DescrRef>,
    _cache_interiorfield_order: Vec<DescrRef>,

    /// Superseded SizeDescr Arcs retired by `register_keyed_size`'s
    /// fuller-layout upgrade. Kept out of `_cache_size_order`.
    _size_keepalive: Vec<DescrRef>,

    /// `gctypelayout.py:301-357 TypeLayoutBuilder.get_type_id` analog —
    /// the shared dense sequential GC type-id allocator covering both
    /// `GcStruct` and `GcArray`.  PyPy's `type_info_group.add_member`
    /// returns a monotonically-increasing index across `id_of_type`;
    /// `init_size_descr` (`gc.py:536-542`) and `init_array_descr`
    /// (`gc.py:544-549`) call `layoutbuilder.get_type_id(TYPE)` to
    /// stamp `descr.tid`.  Pyre lifts the allocator onto `GcCache`
    /// itself (no separate layoutbuilder object) — analyzer-side
    /// SizeDescr + ArrayDescr cache-miss-mint each pull one tid from
    /// this counter via the matching `init_*_descr` hook, mirroring
    /// PyPy's structure.  Tid 0 is reserved (`gctypelayout.py:328-331`
    /// "don't use typeid 0, may help debugging").
    next_type_id: u32,
}

impl Default for GcCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GcCache {
    pub fn new() -> Self {
        GcCache {
            _cache_size: HashMap::new(),
            _cache_field: HashMap::new(),
            _cache_array: HashMap::new(),
            _cache_arraylen: HashMap::new(),
            _cache_call: HashMap::new(),
            _cache_interiorfield: HashMap::new(),
            _cache_size_order: Vec::new(),
            _cache_field_order: Vec::new(),
            _cache_array_order: Vec::new(),
            _cache_arraylen_order: Vec::new(),
            _cache_call_order: Vec::new(),
            _cache_interiorfield_order: Vec::new(),
            _size_keepalive: Vec::new(),
            // tid 0 is reserved as "no class" / sentinel —
            // `gctypelayout.py:328-331 make_type_info_group` adds a
            // DUMMY member at index 0.
            next_type_id: 1,
        }
    }

    /// descriptor cardinality — the cardinality [`Self::setup_descrs`] would produce, **without
    /// its `set_descr_index` side effect**.
    ///
    /// The selection-vs-pool fork needs `all_descrs` and the assembler's
    /// `opcode_descrs` measured in the same generation: `all_descrs` stable
    /// while the pool moves is a *selection* defect; both moving is a *pool*
    /// defect. `setup_descrs` cannot serve — it stamps `descr_index` on every
    /// entry, so calling it merely to count would perturb the thing measured.
    pub fn all_descrs_len(&self) -> usize {
        self._cache_size_order.len()
            + self._cache_field_order.len()
            + self._cache_array_order.len()
            + self._cache_arraylen_order.len()
            + self._cache_call_order.len()
            + self._cache_interiorfield_order.len()
    }

    /// descr.py:25-47 setup_descrs().
    ///
    /// Iterates per-type caches in fixed group order (size, field, array,
    /// arraylen, call, interiorfield), and within each group in creation
    /// order (insertion order). Assigns sequential descr_index.
    pub fn setup_descrs(&self) -> Vec<DescrRef> {
        let mut all_descrs: Vec<DescrRef> = Vec::new();
        // descr.py:27-29: _cache_size
        for v in &self._cache_size_order {
            v.set_descr_index(all_descrs.len() as i32);
            all_descrs.push(v.clone());
        }
        // descr.py:30-33: _cache_field (nested)
        for v in &self._cache_field_order {
            v.set_descr_index(all_descrs.len() as i32);
            all_descrs.push(v.clone());
        }
        // descr.py:34-36: _cache_array
        for v in &self._cache_array_order {
            v.set_descr_index(all_descrs.len() as i32);
            all_descrs.push(v.clone());
        }
        // descr.py:37-39: _cache_arraylen
        for v in &self._cache_arraylen_order {
            v.set_descr_index(all_descrs.len() as i32);
            all_descrs.push(v.clone());
        }
        // descr.py:40-42: _cache_call
        for v in &self._cache_call_order {
            v.set_descr_index(all_descrs.len() as i32);
            all_descrs.push(v.clone());
        }
        // descr.py:43-45: _cache_interiorfield
        for v in &self._cache_interiorfield_order {
            v.set_descr_index(all_descrs.len() as i32);
            all_descrs.push(v.clone());
        }
        assert!(
            all_descrs.len() < (1 << 15),
            "descr.py:46: assert len(all_descrs) < 2**15"
        );
        all_descrs
    }

    /// Register the GC layouts of synthetic structs whose serialized
    /// structural identity is still standing in for a collector type id.
    ///
    /// `gc.py:536-542 GcLLDescr_framework.init_size_descr` asks
    /// `gctypelayout.py:333-357 TypeLayoutBuilder.get_type_id` for the
    /// collector id and stores it on the shared SizeDescr.  Pyre performs the
    /// collector-specific half through `register`, keeping majit-ir
    /// independent of majit-gc.
    pub fn register_unresolved_struct_tids(
        &mut self,
        mut register: impl FnMut(usize, Vec<usize>) -> u32,
    ) -> usize {
        let mut registered = 0;
        for descr in &self._cache_size_order {
            let Some(sd) = descr
                .as_any()
                .and_then(|descr| descr.downcast_ref::<SimpleSizeDescr>())
            else {
                continue;
            };
            if !sd.is_gc_managed()
                || sd.headerless()
                || sd.cache_key() == 0
                || !struct_tid_is_unresolved(sd.cache_key(), sd.type_id())
            {
                continue;
            }
            let ref_offsets = sd
                .gc_fielddescrs()
                .iter()
                .map(|field| field.offset())
                .collect();
            let tid = register(sd.size(), ref_offsets);
            sd.set_type_id(tid);
            registered += 1;
        }
        registered
    }

    /// `gc.py:536-542 GcLLDescr_framework.init_size_descr` analog.
    /// Allocates a dense GC tid via `next_type_id` (the
    /// `TypeLayoutBuilder.get_type_id` analog) and stamps
    /// `SimpleSizeDescr.type_id`.  Called BEFORE Arc wrap so the
    /// mutation lands on the unwrapped descriptor, matching PyPy's
    /// mutable-object semantics.
    pub fn init_size_descr(&mut self, _key: &LLType, sizedescr: &mut SimpleSizeDescr) {
        let type_id = self.alloc_type_id();
        sizedescr.set_type_id(type_id);
    }

    /// `gc.py:544-549 GcLLDescr_framework.init_array_descr` analog.
    /// Same shape as `init_size_descr` — share the `next_type_id`
    /// counter per PyPy's single `type_info_group` covering both
    /// GcStruct and GcArray.
    pub fn init_array_descr(&mut self, _key: &LLType, arraydescr: &mut SimpleArrayDescr) {
        let type_id = self.alloc_type_id();
        arraydescr.set_type_id(type_id);
    }

    /// `gctypelayout.py:349 type_info_group.add_member` analog — hand
    /// out the next sequential tid and bump the counter.  u32::MAX
    /// overflow panics (analyzer trace pool cannot exceed 2^32 distinct
    /// types in any conceivable program; assertion guards programmer
    /// error sooner than overflow corruption).
    fn alloc_type_id(&mut self) -> u32 {
        let tid = self.next_type_id;
        self.next_type_id = self
            .next_type_id
            .checked_add(1)
            .expect("GcCache type_id overflow (u32)");
        tid
    }
}

// descr.py:105-127, 218-239, 256-267, 348-378, 647-675:
// get_size_descr, get_field_descr, get_field_arraylen_descr,
// get_array_descr, get_call_descr are methods on GcCache (see below).
// PyPy passes `gccache` as the first argument to these free functions;
// in Rust they are &mut self methods on GcCache.

impl GcCache {
    /// descr.py:105-127 get_size_descr(gccache, STRUCT, vtable).
    ///
    /// `key`: LLType::Struct — STRUCT identity (no vtable in key).
    /// `vtable` is a payload/assertion parameter, not part of the key.
    /// `immutable_flag`: descr.py:112 heaptracker.is_immutable_struct(STRUCT).
    ///
    /// The numeric `tid` stamped on the returned SizeDescr is allocated
    /// by `init_size_descr` from the shared `next_type_id` counter
    /// (analog of `TypeLayoutBuilder.get_type_id` in
    /// `gctypelayout.py:333-357`).  Caller does not supply it.  This
    /// guarantees dense, collision-free tids per distinct key regardless
    /// of how the caller derived the `LLType::Struct(u64)` identity.
    pub fn get_size_descr(
        &mut self,
        key: LLType,
        size: usize,
        vtable: usize,
        immutable_flag: bool,
    ) -> DescrRef {
        // descr.py:108-109: cache hit
        if let Some(descr) = self._cache_size.get(&key) {
            return descr.clone();
        }
        // descr.py:117-118: SizeDescr(size, vtable=vtable, immutable_flag=immutable_flag)
        // `type_id` placeholder 0 — overwritten by `init_size_descr`
        // below per `gc.py:536-542` structure.
        let mut sd = if vtable != 0 {
            SimpleSizeDescr::with_vtable(u32::MAX, size, 0, vtable)
        } else {
            SimpleSizeDescr::new(u32::MAX, size, 0)
        };
        sd.is_immutable = immutable_flag;
        // Stamp the original `LLType::Struct(u64)` cache key onto the
        // descr so the inverse `bh_size_spec_from_descr` reader
        // (`assembler.rs`) recovers the same key the analyzer-side
        // `bh_size_spec_from_callcontrol` stamps.  Without this slot,
        // `bh_size_spec_from_descr` returned `type_id` widened to u64,
        // landing on a different `_cache_size` slot when round-tripped
        // through `simple_descr_group_from_bh_size`.
        if let LLType::Struct(k) = &key {
            sd.set_cache_key(*k);
        }
        sd.mark_fieldless_shell_mint();
        FIELD_MINT
            .fieldless_size_shell_mints
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // descr.py:119: gccache.init_size_descr(STRUCT, sizedescr)
        // gc.py:536-542: sets descr.tid — must happen BEFORE Arc wrap.
        self.init_size_descr(&key, &mut sd);
        let descr: DescrRef = Arc::new(sd);
        // descr.py:120: cache[STRUCT] = sizedescr
        self._cache_size.insert(key, descr.clone());
        self._cache_size_order.push(descr.clone());
        // descr.py:123-126: gc_fielddescrs / all_fielddescrs populated
        // externally via SimpleSizeDescr::with_all_fielddescrs.  The walker
        // exists (`majit-translate codewriter::heaptracker`) but lives in the
        // crate that depends on this one, so it cannot be called from here;
        // its callers thread the resulting list in.
        descr
    }

    /// Census of what `get_field_descr` found when it bound a field to a parent.
    ///
    /// `descr_set_absent` / `descr_set_ambiguous` cannot see any of this: they
    /// ask whether a raw-set member resolved to SOME descr, so they read 0 while
    /// a field sits under a parent that numbers its list differently. They count
    /// slots filled, not slots filled correctly. These four count the latter.
    ///
    /// Upstream needs no such census because `heaptracker.py:60-72` and `:96-112`
    /// share one walker, making `all_fielddescrs(S)[i].get_index() == i` true by
    /// construction; every nonzero below is a place pyre's split producers reach
    /// a state `descr.py` cannot represent.
    pub fn field_position_census() -> [usize; 4] {
        use std::sync::atomic::Ordering::Relaxed;
        [
            FIELD_PARENT_ABSENT.load(Relaxed),
            FIELD_PARENT_EMPTY.load(Relaxed),
            FIELD_INDEX_REDERIVED.load(Relaxed),
            FIELD_INDEX_UNRESOLVED.load(Relaxed),
        ]
    }

    /// `[checked, misplaced]` over every field spec handed to a group factory.
    ///
    /// Read this next to [`field_position_census`]'s `rederived`, which cannot
    /// answer the same question: `derive_index_in_parent` is both the judge and
    /// the repairman. It counts only the disagreements it *reached* — a parent
    /// already in `_cache_size`, a non-empty list, a name that resolves — and it
    /// overwrites the caller's number in the same expression, so a nonzero
    /// `rederived` is a repair log, not a defect report, and a zero one is
    /// silence about a population it mostly never examined
    /// (`parent_absent` is the count of mints where the question was skipped
    /// outright, and it dwarfs `rederived` by three orders of magnitude, because
    /// `make_simple_descr_group_keyed_with_headerless` mints every field BEFORE
    /// `register_keyed_size` publishes the parent they will be indexed against).
    ///
    /// `misplaced` here is the same defect measured where nothing can have
    /// repaired it yet.
    ///
    /// [`field_position_census`]: Self::field_position_census
    pub fn spec_position_census() -> [usize; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [
            FIELD_SPEC_CHECKED.load(Relaxed),
            FIELD_SPEC_MISPLACED.load(Relaxed),
        ]
    }

    /// The `unresolved` population, named — one line per distinct mint, most
    /// frequent first.
    ///
    /// `field_position_census`'s fourth number says how many field mints named
    /// a field their parent's positional list does not contain. It cannot say
    /// WHERE the fix belongs, and that is the whole decision: a population
    /// concentrated in one producer is fixed at that producer, one spread
    /// across every owner is fixed at the resolution layer. Counting cannot
    /// distinguish those two and no amount of re-running changes that, so the
    /// records have to be printed.
    ///
    /// Deliberately NOT reconstructed from the final cache the way
    /// [`size_shell_owner_sample`]'s `orphan_fields` section is. That one
    /// re-asks the question at exit, against parents that have grown since, and
    /// it dedupes into a set — so it answers a neighbouring question and its
    /// total does not match this counter's. These rows are the mint events
    /// themselves, which is the population the counter counts.
    ///
    /// The first line is a header carrying the ungated counter next to the
    /// number of records actually collected. They agree only if the knob was
    /// set before the first mint; printing both makes a late-armed gate read as
    /// a short table rather than as a small finding. `all_descrs` is the
    /// caller's, for the same reason `descr_set_resolved` rides its own line —
    /// a run where the descr pool never loaded reports zero here, and without a
    /// denominator that zero is indistinguishable from a clean tree.
    ///
    /// `shown`, `distinct` and `limit` are all printed, and the knob's own value
    /// sets the cap. `limit` is not redundant with `shown`: they coincide only
    /// while the table is truncated, so without it a complete table cannot be
    /// told from one that happens to fit. `order` is printed for the same
    /// reason — rows are ranked by FREQUENCY, so a prefix is the most frequent
    /// rows and never a sample, and a reader who takes a proportion from
    /// `shown < distinct` is reading a ranking's head.
    ///
    /// [`size_shell_owner_sample`]: Self::size_shell_owner_sample
    pub fn field_position_unresolved_sample(all_descrs: usize) -> Vec<String> {
        let limit = field_position_unresolved_limit().unwrap_or(0);
        let [parent_absent, parent_empty, rederived, unresolved] = Self::field_position_census();
        let table = FIELD_UNRESOLVED_NAMES
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let recorded: usize = table.values().sum();
        // Frequency first, then the key's own order, so the ranking is total
        // and two runs of one program print the same bytes.
        let mut rows: Vec<(&UnresolvedFieldMint, usize)> =
            table.iter().map(|(k, v)| (k, *v)).collect();
        rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        // `shown` equals `limit` exactly when the table is truncated, so the cap
        // is recoverable from a SHORT table but not from a complete one. Print it
        // either way: a reader who sees `distinct=158 shown=158` still has to know
        // whether they asked for 158 or for everything, because only the second
        // says no further rows exist.
        let limit_shown = if limit == usize::MAX {
            "all".to_string()
        } else {
            limit.to_string()
        };
        // `claimed + placeholder == unresolved` by construction — both are bumped
        // on the same arm, outside the naming gate — so the pair splits the
        // COUNT and not the (possibly short) table. Printing it beside
        // `recorded` is what lets a reader tell a producer that ranked the field
        // wrongly from one that never ranked it at all; while `caller_index` was
        // a plain `usize` those were the same rows.
        let [unresolved_claimed, unresolved_placeholder] = Self::unresolved_index_provenance();
        let mut lines = vec![format!(
            "field_pos_unresolved_names unresolved={unresolved} \
             unresolved_claimed={unresolved_claimed} \
             unresolved_placeholder={unresolved_placeholder} recorded={recorded} \
             distinct={} shown={} limit={limit_shown} order=count-desc,key-asc \
             mints={} all_descrs={all_descrs} names_scope=native-only",
            table.len(),
            rows.len().min(limit),
            parent_absent + parent_empty + rederived + unresolved,
        )];
        // The ordering is a total one and it is NOT the population's shape: rows
        // are ranked by frequency, so a prefix is the most frequent rows, not a
        // sample. Named in the output rather than only in this doc comment,
        // because the reader taking a proportion has the line and not the source.
        // Two clauses with two different subjects, and only one of them ever went
        // stale: the COUNT's export exists now, the LISTING's absence is still
        // real. Do not delete this line as stale — it is the only warning a wasm
        // reader gets that the rows below are missing rather than empty.
        lines.push(
            "field_pos_unresolved_note names-listing only: the COUNT reaches wasm as \
             pyre_jit_field_pos_unresolved, but the guest has no stderr, so these per-mint \
             rows never appear there. Read the count; do not read their absence as a zero"
                .to_string(),
        );
        lines.extend(rows.iter().take(limit).map(|(mint, count)| {
            // `none` rather than a number: the whole reason this field is an
            // `Option` is that a producer which resolved nothing must not print
            // the same token as one that resolved slot 0.
            let caller_index = match mint.caller_index {
                Some(i) => i.to_string(),
                None => "none".to_string(),
            };
            format!(
                "field_pos_unresolved {count}x {} key={} caller_index={caller_index} \
                 parent_fields={} parent=[{}]",
                mint.label, mint.lookup_key, mint.parent_len, mint.parent_keys,
            )
        }));
        lines
    }

    /// `[claimed, placeholder]` over the mints `derive_index_in_parent` could not
    /// resolve — the split of `field_pos_unresolved` by whether the producer had
    /// written an `index_in_parent` at all.
    ///
    /// Sums to `field_position_census()[3]` exactly. This is the resolve-side
    /// half of the question [`census_mint_index_provenance`] asks on the mint
    /// side, and the two run in DIFFERENT PROCESSES — pyre mints in
    /// `pyre-jit-trace/build.rs` and resolves in the program — so neither pair
    /// can be read off the other's numbers.
    pub fn unresolved_index_provenance() -> [usize; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [
            FIELD_INDEX_UNRESOLVED_CLAIMED.load(Relaxed),
            FIELD_INDEX_UNRESOLVED_PLACEHOLDER.load(Relaxed),
        ]
    }

    /// `[checked, misplaced]` over field descrs compared against the parent
    /// their own producer attached — see [`census_attached_index`].
    ///
    /// This is the surface [`spec_position_census`] does NOT cover. That one
    /// reads a parent's positional list, which every producer builds by
    /// enumerating the list it just sorted, so it is self-consistent by
    /// construction and reads zero even while a standalone field descr pointing
    /// INTO that list carries a stale rank. The assembler mints exactly such a
    /// descr: `add_struct_field_descr` resolves the rank against the layout as
    /// it stands at the emit site, and `register_struct_layout` re-indexes the
    /// layout on every later merge.
    ///
    /// [`spec_position_census`]: Self::spec_position_census
    pub fn attached_position_census() -> [usize; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [
            FIELD_ATTACHED_CHECKED.load(Relaxed),
            FIELD_ATTACHED_MISPLACED.load(Relaxed),
        ]
    }

    /// `[claimed, placeholder]` over every `fielddescrof` mint — see
    /// [`census_mint_index_provenance`] for what each arm means and for why the
    /// call site is deliberately outside every parent check.
    ///
    /// Read it as the denominator the other two censuses cannot supply. A
    /// `placeholder` of 0 settles the question outright: no mint carried an
    /// unwritten index, so no row anywhere downstream can be a placeholder, and
    /// every `index_in_parent == 0` in the program is a real slot claim. A
    /// nonzero bounds the population instead of splitting it — it says how many
    /// unwritten indices were minted, not how many of them reached any
    /// particular table.
    pub fn mint_index_census() -> [usize; 2] {
        use std::sync::atomic::Ordering::Relaxed;
        [
            FIELD_MINT_INDEX_CLAIMED.load(Relaxed),
            FIELD_MINT_INDEX_PLACEHOLDER.load(Relaxed),
        ]
    }

    /// How many published parents still list no fields, and how many of those
    /// are shadowing a layout this same cache already knows.
    ///
    /// `get_size_descr` returns on cache hit (`descr.py:108-109`), so whoever
    /// publishes a key FIRST decides what every later consumer sees. Upstream
    /// that is harmless because `descr.py:111-127` is the only producer and it
    /// always builds the full `SizeDescr` before inserting. A pyre mint that
    /// publishes a caller-sized, vtable-less shell — a shape `descr.py:111-116`
    /// cannot produce — therefore silently outranks the real layout a later
    /// producer would have installed.
    ///
    /// Returns `[published, fieldless, shadowing, aliased]`:
    /// - `published` — `_cache_size` entries, the denominator.
    /// - `fieldless` — entries whose `all_fielddescrs()` is empty. Benign on its
    ///   own: a struct may legitimately have nothing the JIT indexes.
    /// - `shadowing` — fieldless entries for which `_cache_field` holds fields
    ///   under the SAME key. Not benign: the cache simultaneously knows the
    ///   struct has fields and answers "it has none" to
    ///   `all_fielddescrs_from_descr`, which is what `force_box` indexes.
    /// - `aliased` — shadowing entries whose struct ALSO reaches this cache under
    ///   some other key that did get a populated list. These are the ones the
    ///   layout exists for; they are shells only because one struct is reachable
    ///   under several identity keys. The rest are structs no producer ever
    ///   published a layout for at all.
    ///
    /// The `shadowing` / `aliased` split is what decides where the fix belongs:
    /// canonicalizing the identity keys fixes the first group and cannot fix the
    /// second. `field_position_census`'s `parent_empty` cannot answer this — it
    /// counts mint attempts, so one shell hit by many fields reads as many.
    ///
    /// `aliased` matches on SHAPE, not on name. The mint sites that publish
    /// shells carry only the numeric struct identity, so their fields get
    /// `get_field_descr`'s `T<type_id>.` stand-in display name; comparing those
    /// against real `Owner.field` spellings can only ever report zero, which
    /// would read as "no aliasing" when it means "nothing was compared". Same
    /// declared size, and every cached field landing on a matching
    /// `(offset, field_size)` slot, is a statement about the struct itself.
    /// `aliased_multi` is the same match restricted to shells carrying at least
    /// two cached fields. A one-field shell matches any populated struct of the
    /// same size that happens to own that one slot, so `aliased` alone overstates
    /// its case; the gap between the two is how much of the finding rests on
    /// single-slot evidence.
    pub fn size_shell_census(&self) -> [usize; 5] {
        let mut populated: Vec<(usize, std::collections::HashSet<(usize, usize)>)> = Vec::new();
        for descr in self._cache_size.values() {
            let Some(sd) = descr.as_size_descr() else {
                continue;
            };
            let slots = sd.all_fielddescrs();
            if slots.is_empty() {
                continue;
            }
            populated.push((
                sd.size(),
                slots.iter().map(|f| (f.offset(), f.field_size())).collect(),
            ));
        }

        let mut fieldless = 0;
        let mut shadowing = 0;
        let mut aliased = 0;
        let mut aliased_multi = 0;
        for (key, descr) in &self._cache_size {
            let Some(sd) = descr.as_size_descr() else {
                continue;
            };
            if !sd.all_fielddescrs().is_empty() {
                continue;
            }
            fieldless += 1;
            let Some(cached) = self._cache_field.get(key).filter(|m| !m.is_empty()) else {
                continue;
            };
            shadowing += 1;
            let want: Vec<(usize, usize)> =
                cached.values().map(|f| (f.offset, f.field_size)).collect();
            if populated
                .iter()
                .any(|(size, slots)| *size == sd.size() && want.iter().all(|w| slots.contains(w)))
            {
                aliased += 1;
                if want.len() > 1 {
                    aliased_multi += 1;
                }
            }
        }
        [
            self._cache_size.len(),
            fieldless,
            shadowing,
            aliased,
            aliased_multi,
        ]
    }

    /// The upstream invariant, counted: `all_fielddescrs(S)[i].get_index() == i`.
    ///
    /// `heaptracker.py:60-72 get_fielddescr_index_in` and `:96-112
    /// all_fielddescrs` are one walker sharing one skip set, so upstream this
    /// holds by construction and there is nothing to check. pyre builds the
    /// positional list from cache-or-mint results
    /// (`make_simple_descr_group_keyed_with_headerless`), so a descr minted
    /// EARLIER by a header-counting producer keeps its own number while this
    /// producer places it at its own position — and the two need not agree.
    ///
    /// Returns `[checked, disagreeing]` over every populated `_cache_size` entry.
    /// `disagreeing` is the count of slots where the descr sitting at position
    /// `i` reports some other `index_in_parent`, which is precisely the state
    /// `optimizeopt/info.rs force_box` cannot survive: it indexes the list by the
    /// descr's own number.
    pub fn positional_invariant_census(&self) -> [usize; 2] {
        let mut checked = 0;
        let mut disagreeing = 0;
        for descr in self._cache_size.values() {
            let Some(sd) = descr.as_size_descr() else {
                continue;
            };
            for (i, fd) in sd.all_fielddescrs().iter().enumerate() {
                checked += 1;
                if fd.index_in_parent() != i {
                    disagreeing += 1;
                }
            }
        }
        [checked, disagreeing]
    }

    /// Whether one identity key is carrying more than one struct.
    ///
    /// `descr.py` keys `_cache_size` / `_cache_field` on the lltype STRUCT
    /// object, so a key means exactly one struct and `id(STRUCT)` never aliases.
    /// pyre keys on `path_hash(<some spelling>)`, and the spellings are minted by
    /// several producers, so two different structs can land on one key while one
    /// struct lands on several.
    ///
    /// Returns `[compared, conflicting]` over fields that are cached under a key
    /// whose parent also lists them: `conflicting` is the count naming the same
    /// field at a DIFFERENT offset. A field cannot be at two offsets in one
    /// struct, so every one of those is two structs sharing a key — the failure
    /// `last_instr` was caught in (offset 48 from one producer, 32 from another).
    ///
    /// This is the question `positional_invariant_census` cannot ask: that one
    /// checks a list against itself and passes as long as each producer is
    /// internally consistent, which two colliding structs both are.
    pub fn identity_collision_census(&self) -> [usize; 2] {
        let mut compared = 0;
        let mut conflicting = 0;
        for (key, cached) in &self._cache_field {
            let Some(sd) = self._cache_size.get(key).and_then(|d| d.as_size_descr()) else {
                continue;
            };
            for fd in cached.values() {
                let Some(listed) = sd
                    .all_fielddescrs()
                    .iter()
                    .find(|f| f.field_key() == fd.field_key())
                else {
                    continue;
                };
                compared += 1;
                if listed.offset() != fd.offset {
                    conflicting += 1;
                }
            }
        }
        [compared, conflicting]
    }

    /// The owner names behind [`size_shell_census`]'s `shadowing` and
    /// `populated` groups, so `aliased=0` can be read as a finding rather than
    /// taken on faith.
    ///
    /// `get_field_descr` falls back to a `T<type_id>.` stand-in when a mint site
    /// carries only the numeric struct identity, and a stand-in can never match a
    /// real struct name. Without seeing the strings, an `aliased` of zero is
    /// indistinguishable from an owner comparison that had nothing to compare.
    ///
    /// [`size_shell_census`]: Self::size_shell_census
    pub fn size_shell_owner_sample(&self, limit: usize) -> Vec<String> {
        fn owner_of(display_name: &str) -> &str {
            display_name
                .rsplit_once('.')
                .map_or(display_name, |(o, _)| o)
        }
        let mut shell: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        let mut populated: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for (key, descr) in &self._cache_size {
            match descr.as_size_descr() {
                Some(sd) if !sd.all_fielddescrs().is_empty() => {
                    populated.extend(
                        sd.all_fielddescrs()
                            .iter()
                            .map(|f| owner_of(f.field_name())),
                    );
                }
                _ => {
                    if let Some(m) = self._cache_field.get(key) {
                        shell.extend(m.values().map(|f| owner_of(f.field_name())));
                    }
                }
            }
        }
        let take = |s: &std::collections::BTreeSet<&str>| {
            s.iter().take(limit).copied().collect::<Vec<_>>().join(",")
        };
        // The `unresolved` population, named. A field cached under a key whose
        // parent lists OTHER fields but not this one means the two sides
        // disagree about what the struct contains — either the field belongs to
        // a different struct that shares the key, or the parent's list is short.
        // The counter alone cannot tell those apart; the names can.
        let mut orphans: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for (key, cached) in &self._cache_field {
            let Some(sd) = self._cache_size.get(key).and_then(|d| d.as_size_descr()) else {
                continue;
            };
            let listed = sd.all_fielddescrs();
            if listed.is_empty() {
                continue;
            }
            for fd in cached.values() {
                if !listed.iter().any(|f| f.field_key() == fd.field_key()) {
                    orphans.insert(format!(
                        "{}@{}~[{}]",
                        fd.field_key(),
                        fd.offset,
                        listed
                            .iter()
                            .map(|f| f.field_key())
                            .collect::<Vec<_>>()
                            .join("|")
                    ));
                }
            }
        }
        vec![
            format!("shell_owners({}) {}", shell.len(), take(&shell)),
            format!("populated_owners({}) {}", populated.len(), take(&populated)),
            format!(
                "orphan_fields({}) {}",
                orphans.len(),
                orphans
                    .iter()
                    .take(limit)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("  ")
            ),
        ]
    }

    /// Where `field_name`/`offset` sits in `parent`'s own `all_fielddescrs`.
    ///
    /// `None` when there is nothing to derive from — no parent published yet, or
    /// a parent whose list is empty — in which case the caller's index is all
    /// there is. `None` also when the field is absent from a NON-empty list:
    /// that means two producers disagree about which fields the struct has, and
    /// inventing a position would be worse than leaving the caller's, since the
    /// existing `debug_assert!` canaries are what should catch it.
    /// The lookup itself, with no census side effect.
    ///
    /// Separate from [`derive_index_in_parent`] because the cache-hit path has to
    /// ask the same question and must not be counted: the census measures what
    /// producers hand in, and a cached field is one producer's answer read back,
    /// not a second producer disagreeing.
    ///
    /// Name only. An offset fallback looks tempting but is unsound while one
    /// struct is reachable under several identity keys: the parent found may
    /// belong to a different struct, and a same-offset field of another type
    /// then reads as this one.
    ///
    /// [`derive_index_in_parent`]: Self::derive_index_in_parent
    fn find_index_in_parent(parent: Option<&DescrRef>, field_name: &str) -> Option<usize> {
        let fields = parent.and_then(|p| p.as_size_descr())?.all_fielddescrs();
        fields.iter().position(|f| f.field_key() == field_name)
    }

    /// `label` is the descr's display name, used only to name the mint in the
    /// `unresolved` table. It is not consulted for the lookup — `field_name` is
    /// the key the parent's list is searched by, and the two disagreeing is one
    /// of the states the table exists to report.
    ///
    /// `caller_index` is `None` when the producer never resolved a slot. The
    /// existing counters compare against `caller_index.unwrap_or(0)`, which is
    /// the literal the descr used to carry in that state, so their published
    /// values are unchanged by the widening; the new information is taken by the
    /// `UNRESOLVED_{CLAIMED,PLACEHOLDER}` pair and by the table row, which keep
    /// the `Option` intact.
    fn derive_index_in_parent(
        parent: Option<&DescrRef>,
        field_name: &str,
        caller_index: Option<usize>,
        label: &str,
    ) -> Option<usize> {
        use std::sync::atomic::Ordering::Relaxed;
        let Some(size_descr) = parent.and_then(|p| p.as_size_descr()) else {
            FIELD_PARENT_ABSENT.fetch_add(1, Relaxed);
            return None;
        };
        if size_descr.all_fielddescrs().is_empty() {
            FIELD_PARENT_EMPTY.fetch_add(1, Relaxed);
            return None;
        }
        match Self::find_index_in_parent(parent, field_name) {
            Some(i) => {
                if i != caller_index.unwrap_or(0) {
                    FIELD_INDEX_REDERIVED.fetch_add(1, Relaxed);
                }
                Some(i)
            }
            None => {
                FIELD_INDEX_UNRESOLVED.fetch_add(1, Relaxed);
                // Outside the naming gate, so the split is taken over the whole
                // unresolved population and not over the rows a late-armed knob
                // happened to catch.
                if caller_index.is_some() {
                    FIELD_INDEX_UNRESOLVED_CLAIMED.fetch_add(1, Relaxed);
                } else {
                    FIELD_INDEX_UNRESOLVED_PLACEHOLDER.fetch_add(1, Relaxed);
                }
                if field_position_unresolved_naming_enabled() {
                    let listed = size_descr.all_fielddescrs();
                    let mint = UnresolvedFieldMint {
                        label: label.to_string(),
                        lookup_key: field_name.to_string(),
                        caller_index,
                        parent_len: listed.len(),
                        parent_keys: listed
                            .iter()
                            .map(|f| f.field_key())
                            .collect::<Vec<_>>()
                            .join("|"),
                    };
                    *FIELD_UNRESOLVED_NAMES
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .entry(mint)
                        .or_insert(0) += 1;
                }
                None
            }
        }
    }

    /// descr.py:218-239 get_field_descr(gccache, STRUCT, fieldname).
    ///
    /// `struct_key`: LLType::Struct — the owning type identity.
    /// `index_in_parent`: descr.py:228 heaptracker.get_fielddescr_index_in(STRUCT, fieldname).
    ///   The structural slot number within the parent struct's field list.
    ///   Caller must provide it: `heaptracker::get_fielddescr_index_in` runs
    ///   in `majit-translate`, one crate above this one. `None` is a caller that
    ///   never resolved one — the state a deserialized `BhDescr::Field` can be
    ///   in, and the only state a plain `usize` could not express, since an
    ///   unwritten claim and a real slot-0 claim are the same integer.
    /// `flag`: descr.py:226 get_type_flag(FIELDTYPE).
    ///
    /// descr.py:234-238: parent_descr = get_size_descr(gccache, STRUCT, vtable).
    /// Looked up from _cache_size[STRUCT]. Caller must ensure get_size_descr
    /// was called first (matches RPython's call at descr.py:238).
    ///
    /// `display_name`: descr.py:227 `'%s.%s' % (STRUCT._name, fieldname)`.
    /// Callers that know the owning struct's source name pass the full
    /// `Owner.field` spelling; `None` falls back to the `T<type_id>.` stand-in
    /// for the mint sites that only carry the numeric struct identity.
    #[expect(
        clippy::too_many_arguments,
        reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
    )]
    pub fn get_field_descr(
        &mut self,
        struct_key: LLType,
        field_name: &str,
        display_name: Option<&str>,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
        is_quasi_immutable: bool,
        flag: ArrayFlag,
        index: u32,
        virtualizable: bool,
        index_in_parent: Option<usize>,
    ) -> Arc<SimpleFieldDescr> {
        // descr.py:234-238: parent_descr = get_size_descr(gccache, STRUCT, vtable)
        let parent = self._cache_size.get(&struct_key).cloned();
        // descr.py:220-221: cache[STRUCT][fieldname]
        let cached = self
            ._cache_field
            .get(&struct_key)
            .and_then(|inner| inner.get(field_name))
            .cloned();
        if let Some(descr) = cached {
            // field-cache identity: does this slot answer for the field the caller asked
            // about?  Passing the *cached* values for the three components
            // whose expected value is derived differently per build profile
            // (immutability pair below, `index_in_parent` further down) leaves
            // offset / size / type / virtualizable as the discriminator — the
            // physical payload, compared identically in every profile.
            FIELD_DESCR_CACHE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if !descr.describes_same_field(
                offset,
                field_size,
                field_type,
                descr.is_immutable,
                descr.is_quasi_immutable(),
                virtualizable,
                descr.index_in_parent,
            ) {
                FIELD_DESCR_CACHE_COLLISIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            // `front/mir.rs` leaves `SemanticProgram::immutable_fields`
            // empty for the whole LLBC pipeline — Charon serializes doc
            // comments but not the `#[jit_immutable_fields]` hint — so
            // every spec built from that side reports the pair
            // `(false, false)` for a field no matter what
            // `_immutable_fields_` says about it. That pair is the absence
            // of the declaration channel, not a claim the field is mutable,
            // and the hit path resolves it by keeping the cached descr's
            // flags. Compare what is kept. The opposite direction — a
            // caller claiming purity a cached descr denies — is a real
            // disagreement and still trips.
            let (expected_immutable, expected_quasi_immutable) =
                if !is_immutable && !is_quasi_immutable {
                    (descr.is_immutable, descr.is_quasi_immutable())
                } else {
                    (is_immutable, is_quasi_immutable)
                };
            // descr.py:220-221 returns the cached descr with no caller re-check.
            // The authoritative slot number is the cached descr's own — it was
            // normalised (rederived against the parent's positional list) when
            // minted, so a field that was rederived once stays rederived. When
            // the parent carries a positional list, `find_index_in_parent`
            // arbitrates and the canary stays strict; when it does not (the mint
            // path binds a fieldless size-descr shell), defer to the cached
            // value rather than the caller's un-arbitrable raw number, which
            // still carries whichever producer's header convention minted it.
            // The arbitration walks the parent's whole positional list, and the
            // canary below is its only reader — `debug_assert!` keeps the code
            // but guards it on the same `cfg!(debug_assertions)`, so a release
            // build would scan and discard. Repeat lookups are the common case
            // (`make_simple_descr_group_keyed_with_headerless` calls this once
            // per field spec under the `gc_cache` lock), so spell the gate out
            // and let the release build keep the cached number.
            let expected_index_in_parent = if cfg!(debug_assertions) {
                Self::find_index_in_parent(parent.as_ref(), field_name)
                    .unwrap_or(descr.index_in_parent)
            } else {
                descr.index_in_parent
            };
            let offset_disagrees = descr.offset != offset;
            let size_disagrees = descr.field_size != field_size;
            let type_disagrees = descr.field_type != field_type;
            let immutability_disagrees = descr.is_immutable != expected_immutable
                || descr.is_quasi_immutable() != expected_quasi_immutable;
            let virtualizable_disagrees = descr.virtualizable != virtualizable;
            let index_disagrees = descr.index_in_parent != expected_index_in_parent;
            let disagrees = offset_disagrees
                || size_disagrees
                || type_disagrees
                || immutability_disagrees
                || virtualizable_disagrees
                || index_disagrees;
            if disagrees {
                use std::sync::atomic::Ordering::Relaxed;
                FIELD_MINT.cache_hit_disagree.fetch_add(1, Relaxed);
                if offset_disagrees {
                    FIELD_MINT.cache_hit_offset.fetch_add(1, Relaxed);
                }
                if size_disagrees {
                    FIELD_MINT.cache_hit_size.fetch_add(1, Relaxed);
                }
                if type_disagrees {
                    FIELD_MINT.cache_hit_type.fetch_add(1, Relaxed);
                }
                if immutability_disagrees {
                    FIELD_MINT.cache_hit_immutability.fetch_add(1, Relaxed);
                }
                if virtualizable_disagrees {
                    FIELD_MINT.cache_hit_virtualizable.fetch_add(1, Relaxed);
                }
                if index_disagrees {
                    FIELD_MINT.cache_hit_index_in_parent.fetch_add(1, Relaxed);
                }
                if field_mint_trace_enabled() {
                    let cached_parent_has_positional_list = descr
                        .get_parent_descr()
                        .and_then(|p| p.as_size_descr().map(|sd| !sd.all_fielddescrs().is_empty()))
                        .unwrap_or(false);
                    eprintln!(
                        "MAJIT_FIELD_MINT_TRACE cache_hit_disagree field_key={field_name:?} \
                         display_name={display_name:?} cached_display_name={:?} \
                         struct_key={struct_key:?} cached_offset={} requested_offset={offset} \
                         cached_size={} requested_size={field_size} cached_type={:?} \
                         requested_type={field_type:?} cached_flag={:?} requested_flag={flag:?} \
                         cached_immutable={} requested_immutable={is_immutable} \
                         effective_requested_immutable={expected_immutable} \
                         cached_quasi_immutable={} \
                         requested_quasi_immutable={is_quasi_immutable} \
                         effective_requested_quasi_immutable={expected_quasi_immutable} \
                         cached_virtualizable={} requested_virtualizable={virtualizable} \
                         cached_index_in_parent={} caller_index_in_parent={index_in_parent:?} \
                         expected_index_in_parent={expected_index_in_parent} \
                         cached_parent_has_positional_list={cached_parent_has_positional_list}",
                        descr.name,
                        descr.offset,
                        descr.field_size,
                        descr.field_type,
                        descr.flag,
                        descr.is_immutable,
                        descr.is_quasi_immutable(),
                        descr.virtualizable,
                        descr.index_in_parent,
                    );
                    if field_mint_backtrace_enabled() {
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                    }
                }
            }
            debug_assert!(
                descr.describes_same_field(
                    offset,
                    field_size,
                    field_type,
                    expected_immutable,
                    expected_quasi_immutable,
                    virtualizable,
                    expected_index_in_parent,
                ),
                "get_field_descr cache hit for {field_name} disagrees with the caller: \
                     cached {:?} (offset {}, size {}, type {:?}, immutable {}, quasi {}, vable {}, \
                     index_in_parent {}) vs requested {display_name:?} (offset {offset}, \
                     size {field_size}, type {field_type:?}, immutable \
                     {expected_immutable}, quasi {expected_quasi_immutable}, \
                     vable {virtualizable}, index_in_parent \
                     {expected_index_in_parent} [caller said {index_in_parent:?}])",
                descr.name,
                descr.offset,
                descr.field_size,
                descr.field_type,
                descr.is_immutable,
                descr.is_quasi_immutable(),
                descr.virtualizable,
                descr.index_in_parent,
            );
            return descr;
        }
        // descr.py:227: name = '%s.%s' % (STRUCT._name, fieldname)
        let name = match display_name {
            Some(n) => n.to_string(),
            None => {
                let type_id = match &struct_key {
                    LLType::Struct(id) => *id,
                    _ => 0,
                };
                format!("T{type_id}.{field_name}")
            }
        };
        // descr.py:230-231: FieldDescr(name, offset, size, flag, index_in_parent, is_pure)
        let mut fd = SimpleFieldDescr::new_with_name(
            index,
            offset,
            field_size,
            field_type,
            is_immutable,
            flag,
            name,
            field_name.to_string(),
        );
        fd.virtualizable = virtualizable;
        // descr.py:228: index_in_parent (from heaptracker).
        //
        // Upstream has no caller-supplied index: `heaptracker.py:60-72
        // get_fielddescr_index_in` is the sole numberer and shares its skip set
        // with `:96-112 all_fielddescrs`, so `all_fielddescrs(S)[i].get_index()
        // == i` holds by construction. pyre has several independent numberers
        // that disagree — the analyzer's `field_pos` and the assembler's
        // `specs.len()` both COUNT the two header words at offsets 0 and 8 that
        // the runtime publish skips, because the skip sets only drop `typeptr`
        // and pyre spells that header `ob_type` / `w_class`. A field carrying
        // one convention under a parent numbered by another is not a cosmetic
        // mismatch: `all_fielddescrs()[index_in_parent]` is a load-bearing
        // lookup (`optimizeopt/info.rs force_box`), so a too-high index either
        // runs off the end or silently names a DIFFERENT field and emits the
        // store against it.
        //
        // So take the index from the parent that will actually be indexed,
        // rather than from whoever happened to call.
        //
        // The two fallbacks are not the same question. The parent's answer wins
        // when it has one; failing that the caller's claim stands; failing THAT
        // there is no claim at all, and `0` is what the descr has always carried
        // in that state — written here as an explicit floor rather than as an
        // initialiser, so the next reader sees that it is a fallback and not a
        // position anyone computed.
        fd.index_in_parent =
            Self::derive_index_in_parent(parent.as_ref(), field_name, index_in_parent, &fd.name)
                .or(index_in_parent)
                .unwrap_or(0);
        // descr.py:229 `is_quasi_immutable = '%s?' in STRUCT._hints.get(
        // '_immutable_fields_', ())` parity.  The analyzer side reads
        // `#[jit_immutable_fields(..., "field?", ...)]` via
        // `ImmutableRank::is_quasi_immutable` and threads the boolean
        // through here so `jtransform.rewrite_op_getfield` emits the
        // `record_quasiimmut_field` guard before the pure read
        // (`jtransform.py:895-903`).
        fd = fd.with_quasi_immutable(is_quasi_immutable);
        // descr.py:238: fielddescr.parent_descr = get_size_descr(gccache, STRUCT, vtable)
        if let Some(ref p) = parent {
            fd.parent_descr = RwLock::new(Some(Arc::downgrade(p)));
        }
        let descr = Arc::new(fd);
        // descr.py:232-233: cachedict = cache.setdefault(STRUCT, {})
        let inner = self._cache_field.entry(struct_key).or_default();
        inner.insert(field_name.to_string(), descr.clone());
        self._cache_field_order.push(descr.clone() as DescrRef);
        descr
    }

    /// descr.py:256-267 get_field_arraylen_descr(gccache, ARRAY_OR_STRUCT).
    ///
    /// Creates a FieldDescr("len", ofs, WORD_SIZE, FLAG_SIGNED) for the
    /// length field of an array. parent_descr = None.
    pub fn get_field_arraylen_descr(&mut self, key: LLType, length_offset: usize) -> DescrRef {
        // descr.py:258-259: cache hit
        if let Some(descr) = self._cache_arraylen.get(&key) {
            return descr.clone();
        }
        // descr.py:263: size = symbolic.get_size(lltype.Signed, tsc)
        let word_size = std::mem::size_of::<usize>();
        // descr.py:264: FieldDescr("len", ofs, size, get_type_flag(lltype.Signed))
        let descr: DescrRef = Arc::new(SimpleFieldDescr::new_with_name(
            u32::MAX,
            length_offset,
            word_size,
            Type::Int,
            false,
            ArrayFlag::Signed, // descr.py:264: get_type_flag(lltype.Signed)
            "len".to_string(),
            "len".to_string(),
        ));
        // descr.py:265: result.parent_descr = None (no parent)
        self._cache_arraylen.insert(key, descr.clone());
        self._cache_arraylen_order.push(descr.clone());
        descr
    }

    /// descr.py:348-378 get_array_descr(gccache, ARRAY_OR_STRUCT).
    ///
    /// `key`: LLType::Array — opaque array type identity.
    /// `flag`: descr.py:363 get_type_flag(ARRAY_INSIDE.OF) — element type
    ///   classification. Caller must compute this from the actual element type
    ///   (signed vs unsigned integer, pointer, float, struct).
    /// `item_type`: IR-level element type (for ArrayDescr::item_type()).
    /// `nolength`: descr.py:359 ARRAY_INSIDE._hints.get('nolength', False).
    /// `length_offset`: offset of the length field (only used when !nolength).
    /// `is_pure`: descr.py:364 bool(ARRAY_INSIDE._immutable_field(None)).
    /// `concrete_type`: descr.py:366-370 '\x00' or 'f' for Float/SingleFloat.
    #[expect(
        clippy::too_many_arguments,
        reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
    )]
    pub fn get_array_descr(
        &mut self,
        key: LLType,
        base_size: usize,
        item_size: usize,
        flag: ArrayFlag,
        item_type: Type,
        nolength: bool,
        length_offset: usize,
        is_pure: bool,
        concrete_type: char,
    ) -> DescrRef {
        // descr.py:350-351: cache hit
        if let Some(descr) = self._cache_array.get(&key) {
            return descr.clone();
        }
        // descr.py:359-362: lendescr
        let lendescr = if nolength {
            None
        } else {
            Some(self.get_field_arraylen_descr(key.clone(), length_offset))
        };
        // descr.py:365: ArrayDescr(basesize, itemsize, lendescr, flag, is_pure, concrete_type)
        let mut ad =
            SimpleArrayDescr::with_flag(u32::MAX, base_size, item_size, 0, item_type, flag);
        ad.lendescr = lendescr;
        ad.is_pure = is_pure;
        ad.concrete_type = concrete_type;
        // Stamp the original `LLType::Array(u64)` cache key onto the
        // descr so the inverse `BhDescr::Array.type_id` reader
        // (`assembler.rs`, `jitcode.rs`) recovers the same key the
        // analyzer-side `arraydescrof_concrete` stamped.  Without this
        // slot, those readers returned `type_id` widened to u64, landing
        // on a different `_cache_array` slot when round-tripped through
        // `make_struct_array_descr_full_keyed`.
        if let LLType::Array(k) = &key {
            ad.set_cache_key(*k);
        }
        // descr.py:377: gccache.init_array_descr(ARRAY_OR_STRUCT, arraydescr)
        // gc.py:544-549: sets descr.tid — must happen BEFORE Arc wrap.
        self.init_array_descr(&key, &mut ad);
        let descr: DescrRef = Arc::new(ad);
        // descr.py:371: cache[ARRAY_OR_STRUCT] = arraydescr
        self._cache_array.insert(key, descr.clone());
        self._cache_array_order.push(descr.clone());
        // descr.py:372-375: all_interiorfielddescrs for struct arrays
        // — set externally via SimpleArrayDescr::set_all_interiorfielddescrs
        descr
    }

    /// Read-only resolution of `LLType::Struct(cache_key)` to the dense GC
    /// `tid` stamped on the cached `SizeDescr` (`gc.py:536-542` descr.tid),
    /// or `None` when the key was never minted.  The blackhole/resume
    /// allocation path holds only the `cache_key` (path_hash) that the
    /// `BhDescr` serialized (`descr.py:108-118` structural identity) and must
    /// recover the real GC tid the header write needs, without minting a
    /// fresh descr the way `get_size_descr` would.  A real dense tid never
    /// keys a `_cache_size` slot, so a miss reports the value was already a
    /// tid and the caller keeps it verbatim.
    pub fn resolve_struct_tid(&self, cache_key: u64) -> Option<u32> {
        self._cache_size
            .get(&LLType::Struct(cache_key))
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.type_id())
    }

    /// Array counterpart of [`resolve_struct_tid`] keyed on
    /// `LLType::Array(cache_key)` (`descr.py:350`).
    pub fn resolve_array_tid(&self, cache_key: u64) -> Option<u32> {
        self._cache_array
            .get(&LLType::Array(cache_key))
            .and_then(|d| d.as_array_descr())
            .map(|ad| ad.type_id())
    }

    /// `descr.py:423-438 get_interiorfield_descr(gc_ll_descr, ARRAY, name, arrayfieldname=None)`
    /// cache-or-mint.  Keys `_cache_interiorfield[(ARRAY, name, arrayfieldname)]`
    /// so the analyzer's `cc.interiorfielddescrof` and the struct-array
    /// `all_interiorfielddescrs` population path share a single
    /// `Arc<SimpleInteriorFieldDescr>` per `(ARRAY, name)` tuple,
    /// matching PyPy's `cpu.interiorfielddescrof(ARRAY, name)` per-tuple
    /// object identity.
    ///
    /// Pyre callers pre-resolve the constituent `array_descr` /
    /// `field_descr` Arcs via [`Self::get_array_descr`] and
    /// [`Self::get_field_descr`] and pass them in; on a cache miss those
    /// arcs feed [`SimpleInteriorFieldDescr::new`].  This mirrors PyPy
    /// `descr.py:430-435` which calls `get_array_descr(ARRAY)` and
    /// `get_field_descr(REALARRAY.OF, name)` inside the cache-miss arm.
    ///
    /// `arrayfieldname` is `""` (empty string) for the GcArray-of-Structs
    /// case (`descr.py:431-432 if arrayfieldname is None: REALARRAY = ARRAY`).
    /// The non-empty case (`descr.py:433-434` GcStruct containing an
    /// inlined GcArray) is the only other PyPy variant; pyre carries the
    /// same `(LLType, String, String)` key shape so both variants share
    /// the same cache layout.
    pub fn get_interiorfield_descr(
        &mut self,
        array_key: LLType,
        name: String,
        arrayfieldname: String,
        array_descr_arc: Arc<dyn ArrayDescr>,
        field_descr_arc: Arc<dyn FieldDescr>,
    ) -> DescrRef {
        let key = (array_key, name, arrayfieldname);
        // `descr.py:427-428 try: return cache[(ARRAY, name,
        // arrayfieldname)]` — cache hit returns the cached object
        // VERBATIM, regardless of its concrete type.  Any-typed Arc in
        // the slot survives; callers that need a concrete-type view
        // downcast at consumption.  The previous "downcast or mint
        // fresh" path overwrote the cache slot and broke PyPy
        // per-tuple object identity.
        if let Some(descr) = self._cache_interiorfield.get(&key) {
            return descr.clone();
        }
        // descr.py:436: descr = InteriorFieldDescr(arraydescr, fielddescr)
        let descr: DescrRef = Arc::new(SimpleInteriorFieldDescr::new(
            u32::MAX,
            array_descr_arc,
            field_descr_arc,
        ));
        // descr.py:437: cache[(ARRAY, name, arrayfieldname)] = descr
        self._cache_interiorfield.insert(key, descr.clone());
        self._cache_interiorfield_order.push(descr.clone());
        descr
    }

    /// descr.py:647-675 get_call_descr(gccache, ARGS, RESULT, extrainfo).
    ///
    /// descr.py:665: key = (arg_classes, result_type, result_signed,
    ///   RESULT_ERASED, extrainfo)
    pub fn get_call_descr(
        &mut self,
        arg_types: Vec<Type>,
        result_type: Type,
        result_signed: bool,
        result_size: usize,
        effect: EffectInfo,
    ) -> DescrRef {
        let effect = crate::effectinfo::intern_effect_info(effect);
        let key = LLType::func_key(
            &arg_types,
            result_type,
            result_class_of(result_type),
            result_signed,
            result_size,
            &effect,
        );
        // descr.py:667-668: cache hit
        if let Some(descr) = self._cache_call.get(&key) {
            return descr.clone();
        }
        // descr.py:670-671: CallDescr(arg_classes, result_type, result_signed,
        //   result_size, extrainfo)
        let descr: DescrRef = Arc::new(SimpleCallDescr::new_with_effect_cell(
            next_call_descr_heapcache_index(),
            arg_types,
            result_type,
            result_signed,
            result_size,
            effect,
        ));
        self._cache_call.insert(key, descr.clone());
        self._cache_call_order.push(descr.clone());
        descr
    }

    //
    // PyPy `gc_cache._cache_*` is populated *exclusively* via the
    // cache-or-mint `get_*_descr` API.  Pyre's lift currently has many
    // mint sites that build descrs ad-hoc (`make_simple_descr_group`,
    // `field_descr_from_bh_field`, the `jit_struct!` macro pre-wiring,
    // bare `Arc::new(SimpleFieldDescr::new(...))`).  The
    // `register_external_*` methods below let those sites publish the
    // freshly-minted descr into the cache's per-category insertion
    // order so `setup_descrs()` enumerates the full population and
    // `compute_bitstrings()` sees every descr.
    //
    // TODO: temporary surface while mint sites
    // migrate to `get_*_descr(LLType, ...)`.  Each migrated site
    // drops `register_external_*` in favour of the keyed cache-or-mint
    // call.  The end state retires every `register_external_*` call
    // and the per-category Vec is populated solely by the keyed path.
    //
    // De-dup is by `Arc::ptr_eq` — same Arc clone re-registered is a
    // no-op; structurally-distinct Arcs (even with matching field
    // offsets / names) stay separate, mirroring PyPy `_cache_*` dict
    // identity post-mint.

    /// External registration for size descrs minted outside
    /// `get_size_descr`.  PyPy `descr.py:25-29` cache iteration parity.
    pub fn register_external_size(&mut self, descr: DescrRef) {
        if !arc_in_vec(&self._cache_size_order, &descr) {
            self._cache_size_order.push(descr);
        }
    }

    /// Keyed sibling of [`Self::register_external_size`] — also
    /// populates `_cache_size[key]` so subsequent `get_size_descr(key, ...)`
    /// calls hit the cache instead of minting a duplicate descr.
    /// `descr.py:108-118 get_size_descr` cache-miss branch writes both
    /// the keyed map and the order Vec; this method mirrors that for
    /// mint sites that bypass `get_size_descr` (`make_simple_descr_group`,
    /// runtime macro `__majit_register_descrs`).
    pub fn register_keyed_size(&mut self, key: LLType, descr: DescrRef) {
        // descr.py:108-118 caches the SizeDescr. Multiple pyre producers may
        // report partial layouts, so the cached owner is upgraded when the
        // incoming frozen list has more fields.
        //
        // Field count alone is not authority, and "more fields" is not
        // "closer to upstream". Where the two producers collide today, the
        // analyzer-side one (`majit-translate`
        // `bh_size_spec_from_callcontrol`) is longer by exactly the header
        // words at offsets 0 and 8 — and `heaptracker.py:60-71
        // all_fielddescrs`, the list `descr.py:125-126` assigns onto the
        // cached descr, skips `typeptr` and `c__pad*` outright while the GC
        // header is not part of the lltype STRUCT at all. Upstream's
        // positional list is therefore the header-free one, which is what the
        // runtime groups build.
        //
        // The vtable is the reliable discriminator between the two: only the
        // runtime publish has one, because a build-time spec has no runtime
        // type object to point at. It decides in both directions, so the
        // surviving descr is the same whichever producer registers first —
        // the closest this can get to upstream's single owner per STRUCT.
        // Losing the vtable is separately fatal —
        // `new_with_vtable` skips its typeptr store (`majit-gc/src/rewrite.rs`
        // gates `gen_initialize_vtable` on `vtable != 0`) over zeroed nursery
        // memory, so the fresh object reads back with a null `ob_type`.
        //
        // Upstream has no upgrade rule to arbitrate at all: `descr.py:105-127
        // get_size_descr` takes the vtable as a parameter, mints once per
        // STRUCT, and assigns `all_fielddescrs` onto that one owner;
        // `gc.py:536-542 init_size_descr` stamps the tid on it. Two producers
        // for one key is pyre's own condition. Where their specs describe
        // genuinely different layouts — disagreeing on field offsets and
        // sizes rather than one being a subset of the other — neither side is
        // right and the key itself is the defect; that is the struct-identity
        // spelling collision, tracked separately.
        let should_insert = match self._cache_size.get(&key) {
            None => true,
            Some(existing) => {
                let existing_count = existing
                    .as_size_descr()
                    .map(|sd| sd.all_fielddescrs().len())
                    .unwrap_or(0);
                let new_count = descr
                    .as_size_descr()
                    .map(|sd| sd.all_fielddescrs().len())
                    .unwrap_or(0);
                let existing_vtable = existing.as_size_descr().map_or(0, |sd| sd.vtable());
                let new_vtable = descr.as_size_descr().map_or(0, |sd| sd.vtable());
                // Both directions, so the surviving descr does not depend on
                // which producer happened to register first. The 18
                // module-scope groups are published before the build-time
                // specs, but the per-exception-class groups behind
                // `W_BASE_EXCEPTION_DESCR_CACHE` are not — they arrive after,
                // and a one-sided rule would reject their vtable exactly the
                // way the other order loses it.
                match (existing_vtable != 0, new_vtable != 0) {
                    (true, false) => false,
                    (false, true) => true,
                    _ => new_count > existing_count,
                }
            }
        };
        if should_insert {
            let upgrades_fieldless_shell = self
                ._cache_size
                .get(&key)
                .and_then(|old| old.as_any())
                .and_then(|old| old.downcast_ref::<SimpleSizeDescr>())
                .is_some_and(|old| old.fieldless_shell_mint)
                && descr
                    .as_size_descr()
                    .is_some_and(|sd| !sd.all_fielddescrs().is_empty());
            if upgrades_fieldless_shell {
                FIELD_MINT
                    .fieldless_size_shell_upgrades
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(old) = self._cache_size.get(&key)
                && !arc_in_vec(&self._size_keepalive, old)
            {
                self._size_keepalive.push(old.clone());
            }
            self._cache_size.insert(key.clone(), descr.clone());
            self._cache_size_order.retain(|d| {
                d.as_size_descr()
                    .map(|sd| {
                        sd.cache_key() != descr.as_size_descr().map(|s| s.cache_key()).unwrap_or(0)
                    })
                    .unwrap_or(true)
            });
            if !arc_in_vec(&self._cache_size_order, &descr) {
                self._cache_size_order.push(descr.clone());
            }
            if let Some(fields) = self._cache_field.get(&key) {
                for field in fields.values() {
                    field.set_parent_descr(&descr);
                }
            }
        }
    }

    /// External registration for field descrs minted outside
    /// `get_field_descr`.  PyPy `descr.py:30-33`.
    pub fn register_external_field(&mut self, descr: DescrRef) {
        if !arc_in_vec(&self._cache_field_order, &descr) {
            self._cache_field_order.push(descr);
        }
    }

    /// Keyed sibling of [`Self::register_external_field`] — also
    /// populates `_cache_field[struct_key][field_name]` so subsequent
    /// `get_field_descr(struct_key, field_name, ...)` calls hit the
    /// cache.  Mirrors `descr.py:218-239 get_field_descr` cache-miss
    /// `cachedict[fieldname] = fielddescr`.  First-write wins.
    pub fn register_keyed_field(
        &mut self,
        struct_key: LLType,
        field_name: String,
        descr: Arc<SimpleFieldDescr>,
    ) {
        let inner = self._cache_field.entry(struct_key).or_default();
        if let std::collections::hash_map::Entry::Vacant(e) = inner.entry(field_name) {
            e.insert(descr.clone());
            let as_ref: DescrRef = descr as DescrRef;
            if !arc_in_vec(&self._cache_field_order, &as_ref) {
                self._cache_field_order.push(as_ref);
            }
        }
    }

    /// External registration for array descrs minted outside
    /// `get_array_descr`.  PyPy `descr.py:34-36`.
    pub fn register_external_array(&mut self, descr: DescrRef) {
        if !arc_in_vec(&self._cache_array_order, &descr) {
            self._cache_array_order.push(descr);
        }
    }

    /// Keyed sibling of [`Self::register_external_array`] — also
    /// populates `_cache_array[key]`.  Mirrors `descr.py:348-378
    /// get_array_descr` cache-miss `cache[ARRAY_OR_STRUCT] = arraydescr`.
    pub fn register_keyed_array(&mut self, key: LLType, descr: DescrRef) {
        // `descr.py:25-47 setup_descrs` cache-iteration invariant —
        // cache hit must NOT push the losing Arc onto `_cache_array_order`.
        let entry = self
            ._cache_array
            .entry(key)
            .or_insert_with(|| descr.clone());
        if Arc::ptr_eq(entry, &descr) && !arc_in_vec(&self._cache_array_order, &descr) {
            self._cache_array_order.push(descr);
        }
    }

    /// External registration for arraylen descrs.  PyPy `descr.py:37-39`.
    pub fn register_external_arraylen(&mut self, descr: DescrRef) {
        if !arc_in_vec(&self._cache_arraylen_order, &descr) {
            self._cache_arraylen_order.push(descr);
        }
    }

    /// Keyed sibling of [`Self::register_external_arraylen`] — also
    /// populates `_cache_arraylen[key]`.  Mirrors
    /// `descr.py:256-267 get_field_arraylen_descr` cache-miss
    /// `cache[ARRAY_OR_STRUCT] = result`.
    pub fn register_keyed_arraylen(&mut self, key: LLType, descr: DescrRef) {
        // `descr.py:25-47 setup_descrs` cache-iteration invariant —
        // cache hit must NOT push the losing Arc onto `_cache_arraylen_order`.
        let entry = self
            ._cache_arraylen
            .entry(key)
            .or_insert_with(|| descr.clone());
        if Arc::ptr_eq(entry, &descr) && !arc_in_vec(&self._cache_arraylen_order, &descr) {
            self._cache_arraylen_order.push(descr);
        }
    }

    /// External registration for interiorfield descrs minted outside
    /// `get_interiorfield_descr`.  PyPy `descr.py:43-45`.
    pub fn register_external_interiorfield(&mut self, descr: DescrRef) {
        if !arc_in_vec(&self._cache_interiorfield_order, &descr) {
            self._cache_interiorfield_order.push(descr);
        }
    }

    /// Keyed sibling of [`Self::register_external_interiorfield`] —
    /// also populates `_cache_interiorfield[(array_key, name,
    /// arrayfieldname)]`.  Mirrors `descr.py:404-433
    /// get_interiorfield_descr` cache-miss
    /// `cache[(ARRAY, name, arrayfieldname)] = interiorfielddescr`.
    /// `arrayfieldname == ""` denotes PyPy `arrayfieldname=None`
    /// (the GcArray-of-Structs case, `descr.py:431-432`); a
    /// non-empty string denotes the GcStruct-containing-inlined-GcArray
    /// case (`descr.py:433-434`).
    pub fn register_keyed_interiorfield(
        &mut self,
        array_key: LLType,
        name: String,
        arrayfieldname: String,
        descr: DescrRef,
    ) {
        // `descr.py:25-47 setup_descrs` cache-iteration invariant —
        // cache hit must NOT push the losing Arc onto
        // `_cache_interiorfield_order`.
        let entry = self
            ._cache_interiorfield
            .entry((array_key, name, arrayfieldname))
            .or_insert_with(|| descr.clone());
        if Arc::ptr_eq(entry, &descr) && !arc_in_vec(&self._cache_interiorfield_order, &descr) {
            self._cache_interiorfield_order.push(descr);
        }
    }

    /// Keyed call-descr interning for production factories outside
    /// `get_call_descr`.
    ///
    /// This is the direct Rust lift of `descr.py:666-673`:
    /// `_cache_call[key]` is owned by `GcCache`, and a miss both stores
    /// the descriptor in the keyed map and appends it to the call group
    /// iteration order used by `setup_descrs()`.  The concrete descriptor
    /// type is supplied by the caller so metainterp can mint its
    /// `MetaCallDescr` while preserving PyPy's cache ownership.
    pub fn intern_call_descr_with<F>(&mut self, key: LLType, make_descr: F) -> DescrRef
    where
        F: FnOnce() -> DescrRef,
    {
        if let Some(descr) = self._cache_call.get(&key) {
            return descr.clone();
        }
        let descr = make_descr();
        self._cache_call.insert(key, descr.clone());
        self._cache_call_order.push(descr.clone());
        descr
    }

    //
    // `setup_descrs()` returns the full enumeration in PyPy group order;
    // these accessors expose individual groups for callers that need to
    // build that sequence incrementally.

    /// `descr.py:27-29 _cache_size` snapshot in insertion order.
    pub fn snapshot_sizes(&self) -> Vec<DescrRef> {
        self._cache_size_order.clone()
    }

    /// `descr.py:30-33 _cache_field` snapshot in insertion order.
    pub fn snapshot_fields(&self) -> Vec<DescrRef> {
        self._cache_field_order.clone()
    }

    /// `descr.py:34-36 _cache_array` snapshot in insertion order.
    pub fn snapshot_arrays(&self) -> Vec<DescrRef> {
        self._cache_array_order.clone()
    }

    /// `descr.py:37-39 _cache_arraylen` snapshot in insertion order.
    pub fn snapshot_arraylens(&self) -> Vec<DescrRef> {
        self._cache_arraylen_order.clone()
    }

    /// `descr.py:40-42 _cache_call` snapshot in insertion order.
    pub fn snapshot_calls(&self) -> Vec<DescrRef> {
        self._cache_call_order.clone()
    }

    /// `descr.py:43-45 _cache_interiorfield` snapshot in insertion order.
    pub fn snapshot_interiorfields(&self) -> Vec<DescrRef> {
        self._cache_interiorfield_order.clone()
    }

    /// Per-category counts for diagnostics / tests.  Tuple order:
    /// `(sizes, fields, arrays, arraylens, calls, interiorfields)`,
    /// matching PyPy `descr.py:25-47` group iteration order.
    pub fn category_counts(&self) -> (usize, usize, usize, usize, usize, usize) {
        (
            self._cache_size_order.len(),
            self._cache_field_order.len(),
            self._cache_array_order.len(),
            self._cache_arraylen_order.len(),
            self._cache_call_order.len(),
            self._cache_interiorfield_order.len(),
        )
    }
}

#[inline]
fn arc_in_vec(haystack: &[DescrRef], needle: &DescrRef) -> bool {
    haystack.iter().any(|d| Arc::ptr_eq(d, needle))
}

/// `Arc<dyn Descr>` → `Arc<T>` safe downcast.
///
/// Equivalent of `Arc::downcast` for the `Any+Send+Sync` family, but
/// reified through the `Descr::as_any` trait method (so the concrete
/// type asserts its identity rather than relying on a global `Any`
/// supertrait on `Descr`).  The unsafe `Arc::from_raw` is gated on
/// `as_any().is::<T>()` — the `Any` invariant guarantees the
/// underlying allocation IS a `T`, making the pointer reinterpret
/// sound.
///
/// PyPy parity: `gc_cache._cache_array[ARRAY_OR_STRUCT]` is typed
/// `ArrayDescr` in Python.  Pyre's lift stores `Arc<dyn Descr>`; this
/// helper restores the concrete-type Arc identity at the consumer
/// boundary so `SimpleInteriorFieldDescr::new` can wrap the cached Arc
/// directly without a fresh allocation.
pub fn try_downcast_arc<T: 'static>(arc: DescrRef) -> Result<Arc<T>, DescrRef> {
    if arc.as_any().is_some_and(|a| a.is::<T>()) {
        // SAFETY: as_any returned a ref typed as T, so the underlying
        // Arc<dyn Descr> allocation contains a T. Arc::from_raw on the
        // typed ptr preserves the reference count.
        let raw = Arc::into_raw(arc) as *const T;
        Ok(unsafe { Arc::from_raw(raw) })
    } else {
        Err(arc)
    }
}

/// Convert `Arc<dyn Descr>` → `Arc<dyn ArrayDescr>` if the underlying
/// concrete type is `SimpleArrayDescr`.
///
/// Rust does not provide direct subtrait downcast on trait objects
/// (`dyn Descr` is the supertrait of `dyn ArrayDescr`).  This helper
/// downcasts to the sole concrete `ArrayDescr` impl and upcasts the
/// resulting `Arc` to the trait-object view — PyPy
/// `cpu.arraydescrof(ARRAY)` per-tuple Arc identity for callers that
/// hold a type-erased `DescrRef` and need the `ArrayDescr` trait
/// surface (e.g. `SimpleInteriorFieldDescr.array_descr`).
pub fn descr_arc_as_array_descr(arc: DescrRef) -> Option<Arc<dyn ArrayDescr>> {
    match try_downcast_arc::<SimpleArrayDescr>(arc) {
        Ok(simple) => Some(simple),
        Err(_) => None,
    }
}

/// Process-global `GcCache` slot — pyre's lift of PyPy's per-CPU
/// `gc_ll_descr.gc_cache`.  PyPy supports multiple CPUs in principle but
/// production targets one CPU per process, so the singleton lift stays
/// faithful: every mint site that goes through `get_*_descr` /
/// `register_external_*` lands in the same cache, and
/// `MetaInterpStaticData::finish_setup_descrs` enumerates from this
/// slot.
///
/// `OnceLock<Mutex<GcCache>>` initialises lazily on first access.  The
/// `Mutex` provides interior mutability for the cache-or-mint dict
/// updates without requiring `&mut MetaInterpStaticData` (which would
/// thread through every codewriter site).
static GC_CACHE: OnceLock<Mutex<GcCache>> = OnceLock::new();

/// Acquire a handle to the process-global `GcCache`.  Mirrors
/// `cpu.gc_ll_descr.gc_cache` access in PyPy — every mint / lookup
/// site for size / field / array / arraylen / call / interiorfield
/// descrs flows through this single instance.
pub fn gc_cache() -> &'static Mutex<GcCache> {
    GC_CACHE.get_or_init(|| Mutex::new(GcCache::new()))
}

/// field-cache identity — `get_field_descr` cache hits, and the subset whose cached payload
/// **disagrees** with what the caller asked for.
///
/// A disagreeing hit means one `(LLType, fieldname)` slot is answering for two
/// different logical fields: the key is not injective over the field universe.
/// That is what can change the number of distinct `Arc`s the descr pool sees —
/// and it is invisible to any artefact diff, which can only show that the bytes
/// moved, never whether the pool over-split or the mint minted twice.
///
/// **Why this and not "distinct `Arc`s per `(LLType, fieldname)`".** That
/// quantity is identically 1: `_cache_field` *is* keyed by that pair, so the
/// map holds one `Arc` per key by construction and a counter reading it back
/// would report `max = 1` on every possible input. Counting the collisions is
/// the same question asked where it can still be answered.
///
/// Deliberately **not** `cfg!(debug_assertions)`-gated. The `debug_assert!`
/// in `get_field_descr` already computes a disagreement and discards it in
/// release; a counter that compiled out would read zero and look clean.
static FIELD_DESCR_CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FIELD_DESCR_CACHE_COLLISIONS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `(hits, collisions)` for the `get_field_descr` cache since process start.
pub fn field_descr_cache_collisions() -> (u64, u64) {
    (
        FIELD_DESCR_CACHE_HITS.load(std::sync::atomic::Ordering::Relaxed),
        FIELD_DESCR_CACHE_COLLISIONS.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// The mint arguments behind every `GcCache` slot an `EffectInfo` raw set
/// names — the projection of this cache that has to survive the
/// analyzer→runtime process split.
///
/// Upstream needs no such table: `setup_descrs` (`descr.py:25-47`) snapshots
/// the one gccache the raw-set descrs were minted into, so
/// `compute_bitstrings` (`effectinfo.py:465-499`) unions descrs that are
/// already present. Pyre carries only the assembler's opcode table across
/// (`pyjitpl.py:2261 setup_descrs(asm.descrs)`), so a descr minted *solely* for
/// a raw set — referenced by no opcode — would otherwise have no slot to
/// resolve against on the far side.
///
/// Insertion-ordered and keyed by the member, so a repeated mint of the same
/// `(STRUCT, fieldname)` records once, exactly as the cache itself dedups.
static EI_DESCR_MINTS: OnceLock<Mutex<indexmap::IndexMap<DescrSetMember, DescrMintSpec>>> =
    OnceLock::new();

fn ei_descr_mints() -> &'static Mutex<indexmap::IndexMap<DescrSetMember, DescrMintSpec>> {
    EI_DESCR_MINTS.get_or_init(|| Mutex::new(indexmap::IndexMap::new()))
}

/// Record how a raw-set member's slot was filled, at the mint site where the
/// layout is in scope. First writer wins, matching the cache-or-mint order in
/// `descr.py:220-233`.
pub fn record_ei_descr_mint(member: DescrSetMember, spec: DescrMintSpec) {
    use std::sync::atomic::Ordering::Relaxed;

    fn count_field_axes(stored: &DescrMintSpec, rejected: &DescrMintSpec) {
        let (stored, rejected) = match (stored, rejected) {
            (
                DescrMintSpec::InteriorField { field: stored, .. },
                DescrMintSpec::InteriorField {
                    field: rejected, ..
                },
            ) => (stored.as_ref(), rejected.as_ref()),
            (stored @ DescrMintSpec::Field { .. }, rejected @ DescrMintSpec::Field { .. }) => {
                (stored, rejected)
            }
            _ => return,
        };
        let (
            DescrMintSpec::Field {
                struct_size: stored_struct_size,
                offset: stored_offset,
                field_size: stored_field_size,
                field_type: stored_field_type,
                flag: stored_flag,
                is_immutable: stored_immutable,
                is_quasi_immutable: stored_quasi_immutable,
                index_in_parent: stored_index_in_parent,
            },
            DescrMintSpec::Field {
                struct_size: rejected_struct_size,
                offset: rejected_offset,
                field_size: rejected_field_size,
                field_type: rejected_field_type,
                flag: rejected_flag,
                is_immutable: rejected_immutable,
                is_quasi_immutable: rejected_quasi_immutable,
                index_in_parent: rejected_index_in_parent,
            },
        ) = (stored, rejected)
        else {
            return;
        };
        if stored_struct_size != rejected_struct_size {
            FIELD_MINT.ei_struct_size.fetch_add(1, Relaxed);
        }
        if stored_offset != rejected_offset {
            FIELD_MINT.ei_offset.fetch_add(1, Relaxed);
        }
        if stored_field_size != rejected_field_size {
            FIELD_MINT.ei_field_size.fetch_add(1, Relaxed);
        }
        if stored_field_type != rejected_field_type {
            FIELD_MINT.ei_field_type.fetch_add(1, Relaxed);
        }
        if stored_flag != rejected_flag {
            FIELD_MINT.ei_flag.fetch_add(1, Relaxed);
        }
        if stored_index_in_parent != rejected_index_in_parent {
            FIELD_MINT.ei_index_in_parent.fetch_add(1, Relaxed);
        }
        if stored_immutable != rejected_immutable {
            FIELD_MINT.ei_immutable.fetch_add(1, Relaxed);
        }
        if stored_quasi_immutable != rejected_quasi_immutable {
            FIELD_MINT.ei_quasi_immutable.fetch_add(1, Relaxed);
        }
    }

    let mut guard = ei_descr_mints().lock().unwrap_or_else(|e| e.into_inner());
    match guard.entry(member) {
        indexmap::map::Entry::Vacant(entry) => {
            entry.insert(spec);
        }
        indexmap::map::Entry::Occupied(entry) => {
            let stored = entry.get();
            if stored == &spec {
                FIELD_MINT.ei_identical.fetch_add(1, Relaxed);
            } else {
                FIELD_MINT.ei_differing.fetch_add(1, Relaxed);
                count_field_axes(stored, &spec);
                if field_mint_trace_enabled() {
                    eprintln!(
                        "MAJIT_FIELD_MINT_TRACE ei_descr_mint_disagree member={:?} \
                         stored_spec={stored:?} rejected_spec={spec:?}",
                        entry.key(),
                    );
                    if field_mint_backtrace_enabled() {
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                    }
                }
            }
        }
    }
}

/// Every recorded `(member, mint spec)` pair, in mint order.
pub fn ei_descr_mints_snapshot() -> Vec<DescrMintEntry> {
    ei_descr_mints()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(member, spec)| DescrMintEntry {
            member: member.clone(),
            spec: spec.clone(),
            ei_index: u32::MAX,
        })
        .collect()
}
/// history.py: TargetToken / JitCellToken identity. PyPy keys
/// `target_tokens_currently_compiling` and `consider_jump`'s
/// `jump_target_descr` by descriptor object identity (Python's default
/// `dict[obj]`). Mirror that here by hashing on the underlying allocation
/// address of the `Arc<dyn Descr>`.
pub fn descr_identity(descr: &DescrRef) -> usize {
    Arc::as_ptr(descr) as *const () as usize
}

/// backend/*/regalloc.py: LABEL/JUMP arg location payload attached to
/// TargetToken descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetArgLoc {
    Reg {
        regnum: u8,
        is_xmm: bool,
    },
    Frame {
        position: usize,
        ebp_offset: i32,
        is_float: bool,
    },
    Ebp {
        ebp_offset: i32,
        is_float: bool,
    },
    Immed {
        value: i64,
        is_float: bool,
    },
    Addr {
        base: u8,
        index: u8,
        scale: u8,
        offset: i32,
    },
}

/// history.py: TargetToken backend-visible state.
pub trait LoopTargetDescr: Descr {
    fn token_id(&self) -> u64;
    fn is_preamble_target(&self) -> bool;
    fn ll_loop_code(&self) -> usize;
    fn set_ll_loop_code(&self, loop_code: usize);
    /// `assembler.py:2456-2462 closing_jump` reads `_ll_loop_code`
    /// at JMP-emit time.  Cranelift needs the *address* of the slot
    /// (not its value) so the in-code dispatch can load the latest
    /// entry on every call.  Default impl panics: only
    /// `LoopTargetDescr` impls backed by an `AtomicUsize` slot can
    /// support a stable address.
    fn ll_loop_code_ptr(&self) -> *const std::sync::atomic::AtomicUsize {
        panic!(
            "ll_loop_code_ptr requires an AtomicUsize-backed slot \
             (LoopTargetDescr impl: {})",
            std::any::type_name::<Self>(),
        );
    }

    /// Cranelift parity for `assembler.py:990-993 TargetToken._ll_loop_code`
    /// per-LABEL entry: PyPy stores each TargetToken's instruction address
    /// directly, so a JMP lands at the LABEL's first instruction (skipping
    /// any preamble that precedes a different LABEL).  Cranelift can't expose
    /// internal block addresses, so the body function takes a `label_index`
    /// parameter and `br_table`s to the right per-LABEL entry block at
    /// runtime.  This slot records WHICH LABEL within the compiled body
    /// function this TargetToken refers to (0 for the first LABEL, 1 for
    /// the second, ...).  Set by `compile_loop`'s LABEL-registration loop
    /// at codegen completion.  Read by `emit_attached_loop_dispatch` so the
    /// source body's `return_call_indirect` passes the right index to the
    /// target body's `br_table` entry.
    ///
    /// Dynasm backend ignores this slot; PyPy x86 instead emits the LABEL
    /// address directly into `_ll_loop_code`.
    fn label_block_id(&self) -> u32 {
        0
    }
    /// Default panics so an incomplete `LoopTargetDescr` migration that
    /// reaches `set_dispatch_target` is surfaced loudly instead of
    /// silently dropping `block_id` and routing multi-LABEL entries to
    /// block 0.  Matches the panic-by-default discipline on the
    /// `FailDescr` write-side defaults.
    fn set_label_block_id(&self, _id: u32) {
        panic!(
            "set_label_block_id requires an AtomicU32-backed slot \
             (LoopTargetDescr impl: {})",
            std::any::type_name::<Self>(),
        );
    }
    fn label_block_id_ptr(&self) -> *const std::sync::atomic::AtomicU32 {
        panic!(
            "label_block_id_ptr requires an AtomicU32-backed slot \
             (LoopTargetDescr impl: {})",
            std::any::type_name::<Self>(),
        );
    }

    /// Target loop's frame depth (`max_output_slots + num_ref_roots`).
    /// Read by the closing-jump dispatcher to verify the source loop's
    /// already-allocated JITFRAME has enough capacity for the target
    /// before tail-calling into it.  `0` until the target has been
    /// registered.
    fn target_frame_depth(&self) -> usize {
        0
    }
    fn set_target_frame_depth(&self, _depth: usize) {
        panic!(
            "set_target_frame_depth requires an AtomicUsize-backed slot \
             (LoopTargetDescr impl: {})",
            std::any::type_name::<Self>(),
        );
    }
    fn target_frame_depth_ptr(&self) -> *const std::sync::atomic::AtomicUsize {
        panic!(
            "target_frame_depth_ptr requires an AtomicUsize-backed slot \
             (LoopTargetDescr impl: {})",
            std::any::type_name::<Self>(),
        );
    }

    /// Publish `(ll_loop_code, label_block_id, target_frame_depth)` as
    /// one coherent dispatch target.  Readers gate on
    /// `ll_loop_code != 0` (Acquire) and then read `label_block_id` and
    /// `target_frame_depth` via baked addresses, so both companion
    /// cells MUST become visible before `ll_loop_code` is non-zero —
    /// otherwise a reader can observe the new code pointer alongside a
    /// stale companion (0 block-id routing into the wrong LABEL, or 0
    /// depth bypassing the frame-capacity check).  Default impl stores
    /// the companions first, then `ll_loop_code`, matching the
    /// readiness ordering.
    fn set_dispatch_target(&self, loop_code: usize, block_id: u32, frame_depth: usize) {
        self.set_target_frame_depth(frame_depth);
        self.set_label_block_id(block_id);
        self.set_ll_loop_code(loop_code);
    }

    fn target_arglocs(&self) -> Vec<TargetArgLoc>;
    fn set_target_arglocs(&self, arglocs: Vec<TargetArgLoc>);

    /// `history.py:493 self.original_jitcell_token = original_jitcell_token`.
    /// The owning JitCellToken's `number` for this TargetToken — set by
    /// `compile.py:237` / `compile.py:289` once the freshly-made JitCellToken
    /// is bound to the loop. Returns `None` for a TargetToken constructed
    /// before the owner exists (the preamble sentinel at
    /// `unroll.rs:196 TargetToken::new_preamble(0)`); `record_loop_or_bridge`
    /// (`compile.py:197-199`) must then leave this JUMP branch unhandled.
    fn original_jitcell_token_number(&self) -> Option<u64> {
        None
    }
    fn set_original_jitcell_token_number(&self, _num: u64) {}
}

#[derive(Debug, Default)]
struct BasicLoopTargetDescrState {
    target_arglocs: Vec<TargetArgLoc>,
    /// `history.py:493 self.original_jitcell_token`. Backfilled at
    /// compile-time once the owning JitCellToken is created.
    original_jitcell_token_number: Option<u64>,
}

#[derive(Debug)]
struct BasicLoopTargetDescr {
    token_id: u64,
    is_preamble_target: bool,
    /// `history.py:478` `TargetToken._ll_loop_code` parity: a single
    /// integer recording the address of the loop's compiled entry
    /// point.  RPython sets this with a plain `setattr` (atomic w.r.t.
    /// the GIL); pyre uses `AtomicUsize` so cranelift-emitted in-code
    /// `closing_jump` dispatch can read it without holding a Mutex.
    /// Offset of this field is baked into the JIT'd code via
    /// `loop_target_ll_loop_code_ptr` so a `JMP imm(target)` parity
    /// instruction can load the latest entry address.
    ll_loop_code: std::sync::atomic::AtomicUsize,
    /// `assembler.py:990-993 TargetToken._ll_loop_code` per-LABEL parity:
    /// records which LABEL within the body function (0 for first LABEL,
    /// 1 for second, ...) so cranelift's per-LABEL `br_table` dispatch
    /// can route the `return_call_indirect` to the right entry block.
    /// Set by cranelift `compile_loop`'s LABEL-registration loop after
    /// codegen; default 0 covers single-LABEL traces and dynasm (which
    /// ignores the slot and uses raw LABEL addresses in `_ll_loop_code`).
    label_block_id: std::sync::atomic::AtomicU32,
    /// `max_output_slots + num_ref_roots` for the target loop.
    /// Published with the dispatch target so the cranelift in-code
    /// closing-jump dispatch can compare against the source frame's
    /// `JF_FRAME_LENGTH` and fall back to the host loop when the
    /// already-allocated frame is too small for the target.
    target_frame_depth: std::sync::atomic::AtomicUsize,
    state: Mutex<BasicLoopTargetDescrState>,
}

impl BasicLoopTargetDescr {
    fn new(token_id: u64, is_preamble_target: bool) -> Self {
        Self {
            token_id,
            is_preamble_target,
            ll_loop_code: std::sync::atomic::AtomicUsize::new(0),
            label_block_id: std::sync::atomic::AtomicU32::new(0),
            target_frame_depth: std::sync::atomic::AtomicUsize::new(0),
            state: Mutex::new(BasicLoopTargetDescrState::default()),
        }
    }
}

impl Descr for BasicLoopTargetDescr {
    fn index(&self) -> u32 {
        self.token_id as u32
    }

    fn repr(&self) -> String {
        if self.is_preamble_target {
            format!("LoopTargetDescr(start:{})", self.token_id)
        } else {
            format!("LoopTargetDescr({})", self.token_id)
        }
    }

    fn as_loop_target_descr(&self) -> Option<&dyn LoopTargetDescr> {
        Some(self)
    }
}

impl LoopTargetDescr for BasicLoopTargetDescr {
    fn token_id(&self) -> u64 {
        self.token_id
    }

    fn is_preamble_target(&self) -> bool {
        self.is_preamble_target
    }

    fn ll_loop_code(&self) -> usize {
        self.ll_loop_code.load(std::sync::atomic::Ordering::Acquire)
    }

    fn set_ll_loop_code(&self, loop_code: usize) {
        self.ll_loop_code
            .store(loop_code, std::sync::atomic::Ordering::Release);
    }

    fn ll_loop_code_ptr(&self) -> *const std::sync::atomic::AtomicUsize {
        &self.ll_loop_code as *const _
    }

    fn label_block_id(&self) -> u32 {
        self.label_block_id
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn set_label_block_id(&self, id: u32) {
        self.label_block_id
            .store(id, std::sync::atomic::Ordering::Release);
    }

    fn label_block_id_ptr(&self) -> *const std::sync::atomic::AtomicU32 {
        &self.label_block_id as *const _
    }

    fn target_frame_depth(&self) -> usize {
        self.target_frame_depth
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn set_target_frame_depth(&self, depth: usize) {
        self.target_frame_depth
            .store(depth, std::sync::atomic::Ordering::Release);
    }

    fn target_frame_depth_ptr(&self) -> *const std::sync::atomic::AtomicUsize {
        &self.target_frame_depth as *const _
    }

    fn target_arglocs(&self) -> Vec<TargetArgLoc> {
        self.state.lock().unwrap().target_arglocs.clone()
    }

    fn set_target_arglocs(&self, arglocs: Vec<TargetArgLoc>) {
        self.state.lock().unwrap().target_arglocs = arglocs;
    }

    fn original_jitcell_token_number(&self) -> Option<u64> {
        self.state.lock().unwrap().original_jitcell_token_number
    }

    fn set_original_jitcell_token_number(&self, num: u64) {
        self.state.lock().unwrap().original_jitcell_token_number = Some(num);
    }
}

/// Base trait for all descriptors.
///
/// Mirrors rpython/jit/metainterp/history.py AbstractDescr.
pub trait Descr: Send + Sync + std::fmt::Debug {
    /// Unique index of this descriptor (for serialization).
    /// Returns u32::MAX if not assigned.
    fn index(&self) -> u32 {
        u32::MAX
    }

    /// history.py:95-101: AbstractDescr.get_descr_index()
    /// Returns -1 if not yet assigned by setup_descrs().
    fn get_descr_index(&self) -> i32 {
        -1
    }

    /// descr.py:28: v.descr_index = len(all_descrs)
    /// Called by setup_descrs() to assign a sequential index.
    fn set_descr_index(&self, _index: i32) {}

    /// Codewriter-side `index()` setter — pyre adaptation for the
    /// per-trace tracking key used by `pyre-jit-trace::state` field
    /// descr lookup (`fd.index() == field_idx`, state.rs:5879/5933).
    /// `gc_cache.get_field_descr` cache-hit path lets the analyzer's
    /// `fielddescrof_concrete` stamp its `descr_indices.field_index`
    /// value onto a previously-cached `Arc<SimpleFieldDescr>` so
    /// trace serialization round-trips on the analyzer's per-trace
    /// codewriter id even when the cache was first populated by
    /// `__majit_register_descrs`.  Default no-op — concrete descrs
    /// that participate in trace serialization override.
    fn set_index(&self, _index: u32) {}

    /// `effectinfo.py:496` `descr.ei_index = sys.maxint` — initial sentinel
    /// before `compute_bitstrings` partitions descrs into (eisetr, eisetw)
    /// equivalence classes (`effectinfo.py:524-526`). Concrete field /
    /// array / interiorfield descrs override this to expose their
    /// `AtomicU32` storage; other descrs (calldescr, sizedescr, faildescr…)
    /// keep the trait default and never participate in bitstring
    /// classification — `compute_bitstrings` collects only descrs that
    /// appear in some `EffectInfo._{readonly,write}_descrs_*` set.
    fn get_ei_index(&self) -> u32 {
        u32::MAX
    }

    /// `effectinfo.py:526` `descr.ei_index = mapping.setdefault(...)`.
    /// Default no-op; field / array / interiorfield descrs override
    /// with `AtomicU32` storage that `compute_bitstrings` publishes.
    fn set_ei_index(&self, _ei_index: u32) {}

    /// `effectinfo.py:537-538` `setattr(ei, 'bitstring_*_descrs_*', ...)` —
    /// per-EI bitstring publication after `compute_bitstrings` has
    /// resolved each EI's (eisetr, eisetw) class membership. Default
    /// no-op for descrs without `EffectInfo`; descrs that own a
    /// mutable EI (i.e. `CallDescr` impls whose `get_extra_info()`
    /// returns a stable address) override to atomically swap their
    /// six bitstring slots.
    ///
    /// `compute_bitstrings` is invoked exactly once at JIT setup
    /// (`pyjitpl.py:2287-2290 finish_setup_descrs`). Implementations
    /// rely on that single-writer happens-before ordering — readers
    /// (heap.rs / virtualize.rs / rewrite.rs) only run after the JIT
    /// has been initialised, after the bitstring write completes.
    #[allow(clippy::too_many_arguments)]
    fn set_effect_bitstrings(
        &self,
        _readonly_descrs_fields: Option<Vec<u8>>,
        _write_descrs_fields: Option<Vec<u8>>,
        _readonly_descrs_arrays: Option<Vec<u8>>,
        _write_descrs_arrays: Option<Vec<u8>>,
        _readonly_descrs_interiorfields: Option<Vec<u8>>,
        _write_descrs_interiorfields: Option<Vec<u8>>,
    ) {
    }

    /// Human-readable representation for debugging.
    fn repr(&self) -> String {
        format!("{:?}", self)
    }

    /// compile.py: clone() — create a subtype-preserving copy with a fresh
    /// fail_index. Returns None if this descriptor type doesn't support cloning.
    /// RPython: `olddescr.clone()` preserves the concrete type
    /// (ResumeGuardDescr, CompileLoopVersionDescr, etc.).
    fn clone_descr(&self) -> Option<DescrRef> {
        None
    }

    // ── Downcasting helpers ──

    /// Generic `Any` downcast escape hatch.  Default `None`; descriptor
    /// types that need consumer-side concrete-type identification (e.g.
    /// pyre's `CallDescrStub` lowering through `flatten_descr_by_ptr`)
    /// override to return `Some(self)`.  Avoids forcing every `Descr`
    /// impl to add `Any` as a supertrait when only a small subset
    /// participates in downstream downcast paths.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        None
    }
    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        None
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        None
    }
    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        None
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        None
    }
    /// Downcast to `JitCodeDescr` when this descriptor identifies a
    /// sub-jitcode (emitted by codewriter for `inline_call_*` opnames).
    /// Default `None` — only descriptors that wrap a `JitCode` body
    /// override.
    fn as_jitcode_descr(&self) -> Option<&dyn JitCodeDescr> {
        None
    }
    /// Downcast to `SwitchDescr` for codewriter-emitted `switch/id`
    /// descriptors. RPython stores a concrete `SwitchDictDescr` object
    /// in `Assembler.descrs`; Rust keeps the same descriptor shape behind
    /// this trait object.
    fn as_switch_descr(&self) -> Option<&dyn SwitchDescr> {
        None
    }
    fn as_loop_token_descr(&self) -> Option<&dyn LoopTokenDescr> {
        None
    }
    fn as_interior_field_descr(&self) -> Option<&dyn InteriorFieldDescr> {
        None
    }
    fn as_loop_target_descr(&self) -> Option<&dyn LoopTargetDescr> {
        None
    }

    /// Whether the field/array described is always pure (immutable).
    fn is_always_pure(&self) -> bool {
        false
    }

    /// Whether the field is quasi-immutable (rarely changes but can).
    /// quasiimmut.py: fields marked _immutable_fields_ = ['x?']
    fn is_quasi_immutable(&self) -> bool {
        false
    }

    /// Whether this descriptor marks a loop version guard.
    ///
    /// Loop version guards have their alternative path compiled immediately
    /// after the main loop, rather than lazily on failure.
    fn is_loop_version(&self) -> bool {
        false
    }

    /// Whether this descriptor refers to a virtualizable field.
    ///
    /// Virtualizable fields (e.g. linked-list head/size) are not force-emitted
    /// at guards; they go into pendingfields instead, matching RPython's
    /// treatment of virtualizable fields in force_lazy_sets_for_guard.
    fn is_virtualizable(&self) -> bool {
        false
    }

    /// compile.py: isinstance(resumekey, ResumeAtPositionDescr).
    /// Guards created during loop unrolling / short preamble inlining
    /// return true. When bridge compilation starts from such a guard,
    /// inline_short_preamble is set to false.
    fn is_resume_at_position(&self) -> bool {
        false
    }

    /// compile.py:924-927: `invent_fail_descr_for_op` mints
    /// `ResumeGuardForcedDescr` for `GUARD_NOT_FORCED` /
    /// `GUARD_NOT_FORCED_2`.  Allows descr-level dispatch in places
    /// that today switch on opcode (e.g. `handle_fail` for forced
    /// guards routes through `resume_in_blackhole` with the
    /// virtualref/virtualizable cache attached at async-forcing time
    /// — compile.py:953).
    fn is_guard_forced(&self) -> bool {
        false
    }

    /// compile.py:923-927: `invent_fail_descr_for_op` mints
    /// `ResumeGuardExcDescr` (or `ResumeGuardCopiedExcDescr` on the
    /// sharing path) for `GUARD_EXCEPTION` / `GUARD_NO_EXCEPTION`.
    /// The exception-flow special-casing in `handle_fail`
    /// (compile.py:932-937) keys off this subtype.
    fn is_guard_exc(&self) -> bool {
        false
    }

    /// compile.py:832-851: `ResumeGuardCopiedDescr(prev)` parity.  A
    /// shared-resume guard whose `get_resumestorage()` (compile.py:849)
    /// returns the donor `ResumeGuardDescr` rather than self.  Used by
    /// `_copy_resume_data_from` (`optimizer.py:688-700`) to share
    /// `rd_numb` / `rd_consts` / `rd_virtuals` / `rd_pendingfields`
    /// with a previous guard.
    fn is_resume_guard_copied(&self) -> bool {
        false
    }

    /// optimizer.py:723 / compile.py:838 `assert isinstance(descr,
    /// compile.ResumeGuardDescr)` parity.  Returns true on
    /// `ResumeGuardDescr` and the subclasses that inherit it
    /// (`ResumeAtPositionDescr`, `ResumeGuardForcedDescr`,
    /// `ResumeGuardExcDescr`, `CompileLoopVersionDescr`).  Returns false
    /// on `ResumeGuardCopiedDescr` (a sibling, not a subclass — its
    /// resume reads chase `prev`) and on plain `MetaFailDescr` /
    /// other non-resume `FailDescr` subtypes.  Callers that need to
    /// store fresh resume data (`store_final_boxes_in_guard`,
    /// `make_resume_guard_copied_descr`) must assert this before
    /// touching `set_rd_numb` / `set_rd_consts` etc., otherwise the
    /// default panicking setters fire late.
    fn is_resume_guard(&self) -> bool {
        false
    }

    /// compile.py:849: `ResumeGuardCopiedDescr.get_resumestorage(): return prev`.
    /// Returns the donor descr for shared-resume guards; `None` for
    /// non-copied descrs (RPython's default `get_resumestorage()`
    /// returns `self` — pyre callers detect that case via
    /// `is_resume_guard_copied()`).
    fn prev_descr(&self) -> Option<DescrRef> {
        None
    }

    /// `compile.py:840-842 ResumeGuardCopiedDescr.copy_all_attributes_from`:
    /// `self.prev = other.prev`.  Overwrites the donor pointer in
    /// place, preserving the receiver's `fail_index` / status.  Default
    /// no-op for descrs that don't carry a `prev` slot — implementations
    /// on `ResumeGuardCopiedDescr` / `ResumeGuardCopiedExcDescr`
    /// override.
    fn set_prev_descr(&self, _prev: DescrRef) {}

    /// Pyre-only: the FOR_ITER green key whose walker-native range class
    /// guard this descr backs, or `None`.  `handle_fail` reads it off the
    /// failing descr to demote the range specialization on the first class
    /// mismatch, independent of the guard's per-trace fail index (which
    /// optimizer guard-folding / unroll can shift).  A shared-resume copied
    /// guard carries no marker of its own, so the default chases `prev` to
    /// the donor `ResumeGuardDescr` that holds the tag; `ResumeGuardDescr`
    /// overrides to read its own slot.
    fn range_foriter_green_key(&self) -> Option<u64> {
        self.prev_descr()
            .and_then(|prev| prev.range_foriter_green_key())
    }

    /// intbounds.py: descr.is_integer_bounded() / get_integer_min/max.
    /// Returns (field_size_bytes, is_signed) if this is a field descriptor.
    /// Used by intbounds to narrow GETFIELD result bounds.
    fn field_size_and_sign(&self) -> (usize, bool) {
        if let Some(fd) = self.as_field_descr() {
            (fd.field_size(), fd.is_field_signed())
        } else {
            (0, false)
        }
    }
}

/// Descriptor for guard failures — carries resume information.
///
/// Mirrors rpython/jit/metainterp/history.py AbstractFailDescr.
pub trait FailDescr: Descr {
    /// Index in the fail descr table.
    fn fail_index(&self) -> u32;

    /// The types of the fail arguments.
    fn fail_arg_types(&self) -> &[Type];

    /// In-place update of the descr's per-slot type vector after
    /// `store_final_boxes_in_guard` has computed `livebox_types` from
    /// numbering.
    ///
    /// PYRE-ADAPTATION: RPython `Box`es carry their own `.type`, so
    /// `ResumeGuardDescr.store_final_boxes` only writes
    /// `guard_op.setfailargs(boxes)` and `store_hash` (compile.py:869);
    /// no per-slot type vector lives on the descr. Pyre's `OpRef` is
    /// untyped, so we cache `Vec<Type>` on the descr and refresh it
    /// here. Concrete `MetaFailDescr` / `ResumeGuardDescr` /
    /// `ResumeAtPositionDescr` / `CompileLoopVersionDescr` use
    /// `UnsafeCell<Vec<Type>>` so the existing `Arc<dyn FailDescr>`
    /// identity (fail_index, vector_info, subtype) is preserved across
    /// the optimizer pass — matching the load-bearing contract that
    /// subtype markers (`is_resume_at_position()`, `loop_version()`)
    /// survive `store_final_boxes_in_guard`.
    ///
    /// Default impl panics: matches RPython's
    /// `assert isinstance(descr, ResumeGuardDescr)` at
    /// optimizer.py:724 — non-guard descrs must never reach
    /// `store_final_boxes_in_guard`. Callers in non-guard paths must
    /// not invoke this method.
    fn set_fail_arg_types(&self, _types: Vec<Type>) {
        panic!(
            "set_fail_arg_types invoked on a FailDescr that does not \
             carry a mutable type vector (RPython optimizer.py:724 \
             `assert isinstance(descr, ResumeGuardDescr)`)"
        );
    }

    // compile.py:855 ResumeGuardDescr._attrs_ = ('rd_numb', 'rd_consts',
    //   'rd_virtuals', 'rd_pendingfields', 'status')
    //
    // Resume payload accessors. `ResumeGuardDescr` (and its tag-only
    // newtype subtypes — Forced / Exc / AtPosition / CompileLoopVersion)
    // store these in `UnsafeCell<Option<Vec<…>>>` so the optimizer can
    // mutate them in place without breaking the `Arc<dyn FailDescr>`
    // identity stamped on the op. `ResumeGuardCopiedDescr(prev)` chases
    // through `prev.rd_*()` (compile.py:849
    // `get_resumestorage(): return prev`).
    //
    // Default impls return `None` / panic for non-resume FailDescrs
    // (e.g. `_DoneWithThisFrameDescr` family,
    // `ExitFrameWithExceptionDescrRef`) — these never carry resume
    // data, matching RPython where the `_attrs_` only live on
    // `AbstractResumeGuardDescr` subclasses.

    /// resume.py:450 — compact resume numbering bytes.
    fn rd_numb(&self) -> Option<&[u8]> {
        None
    }

    /// `compile.py:864 self.rd_numb = other.rd_numb` parity: the
    /// reference-share variant of `rd_numb()`.  `Arc<[u8]>` lets
    /// `copy_all_attributes_from` clone the donor's payload with a
    /// single refcount bump rather than allocating a fresh buffer.
    /// Returns `None` for non-resume descrs.
    fn rd_numb_arc(&self) -> Option<std::sync::Arc<[u8]>> {
        None
    }

    /// resume.py:450 — write-through. Default panics: a setter call on
    /// a non-resume FailDescr is a soundness bug (the optimizer should
    /// never ask a `_DoneWithThisFrameDescr` to carry resume data).
    fn set_rd_numb(&self, _value: Option<Vec<u8>>) {
        panic!(
            "set_rd_numb invoked on a FailDescr that does not carry \
             rd_numb (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    /// `compile.py:864 self.rd_numb = other.rd_numb` reference-share
    /// setter.  Default panics for non-resume descrs (same contract as
    /// `set_rd_numb`).
    fn set_rd_numb_arc(&self, _value: Option<std::sync::Arc<[u8]>>) {
        panic!(
            "set_rd_numb_arc invoked on a FailDescr that does not \
             carry rd_numb (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    /// resume.py:451 — shared constant pool referenced by `rd_numb`.
    fn rd_consts(&self) -> Option<&[Const]> {
        None
    }

    fn rd_consts_arc(&self) -> Option<std::sync::Arc<crate::SharedConstPool>> {
        None
    }

    fn set_rd_consts(&self, _value: Option<Vec<Const>>) {
        panic!(
            "set_rd_consts invoked on a FailDescr that does not carry \
             rd_consts (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    fn set_rd_consts_arc(&self, _value: Option<std::sync::Arc<crate::SharedConstPool>>) {
        panic!(
            "set_rd_consts_arc invoked on a FailDescr that does not \
             carry rd_consts (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    /// resume.py:488 — virtual object field info referenced by `rd_numb`.
    fn rd_virtuals(&self) -> Option<&[std::rc::Rc<RdVirtualInfo>]> {
        None
    }

    fn rd_virtuals_arc(&self) -> Option<std::sync::Arc<[std::rc::Rc<RdVirtualInfo>]>> {
        None
    }

    fn set_rd_virtuals(&self, _value: Option<Vec<std::rc::Rc<RdVirtualInfo>>>) {
        panic!(
            "set_rd_virtuals invoked on a FailDescr that does not carry \
             rd_virtuals (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    fn set_rd_virtuals_arc(&self, _value: Option<std::sync::Arc<[std::rc::Rc<RdVirtualInfo>]>>) {
        panic!(
            "set_rd_virtuals_arc invoked on a FailDescr that does not \
             carry rd_virtuals (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    /// resume.py: rd_pendingfields — deferred heap writes.
    fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        None
    }

    fn rd_pendingfields_arc(&self) -> Option<std::sync::Arc<[GuardPendingFieldEntry]>> {
        None
    }

    fn set_rd_pendingfields(&self, _value: Option<Vec<GuardPendingFieldEntry>>) {
        panic!(
            "set_rd_pendingfields invoked on a FailDescr that does not \
             carry rd_pendingfields (compile.py:855 `_attrs_` only on \
             AbstractResumeGuardDescr subclasses)"
        );
    }

    fn set_rd_pendingfields_arc(&self, _value: Option<std::sync::Arc<[GuardPendingFieldEntry]>>) {
        panic!(
            "set_rd_pendingfields_arc invoked on a FailDescr that does \
             not carry rd_pendingfields (compile.py:855 `_attrs_` only \
             on AbstractResumeGuardDescr subclasses)"
        );
    }

    /// Whether this fail descriptor represents a FINISH exit.
    fn is_finish(&self) -> bool {
        false
    }

    /// `compile.py:658-662` `ExitFrameWithExceptionDescrRef` parity:
    /// whether this FINISH descr was emitted for
    /// `pyjitpl.py:3238-3245 compile_exit_frame_with_exception` rather
    /// than `pyjitpl.py:3198-3220 compile_done_with_this_frame`.  The
    /// runtime classifier uses this flag to route the exit to
    /// `jitexc.ExitFrameWithExceptionRef` (`jitexc.py:45`) instead of
    /// `jitexc.DoneWithThisFrame*` — equivalent to dispatching on the
    /// `handle_fail` method of the corresponding descr subclass.
    fn is_exit_frame_with_exception(&self) -> bool {
        false
    }

    /// history.py:470-499 TargetToken parity: whether this exit corresponds
    /// to an external JUMP whose target lives in a different compiled
    /// function. Backends that can't emit raw inter-function JMPs (Cranelift)
    /// flag the exit so the dispatcher re-enters the target via
    /// `target_descr()`. assembler.py:2456-2462 closing_jump.
    fn is_external_jump(&self) -> bool {
        false
    }

    /// history.py:470 TargetToken descriptor identifying the JUMP target.
    /// Present only when `is_external_jump()` is true.
    ///
    /// Returns an owned `DescrRef` (cloned `Arc`) — the
    /// the cranelift implementation to a backend-static side-table that
    /// cannot hand out a borrow under a lock.
    fn target_descr(&self) -> Option<DescrRef> {
        None
    }

    /// Pyre-only cranelift cross-loop JUMP target publish.
    /// Writes the target `DescrRef` into the per-emission slot read
    /// back via `target_descr` / `is_external_jump`.  Default panics —
    /// only Resume-family descrs (`ResumeGuardDescr` /
    /// `ResumeGuardCopiedDescr` + their subclass wrappers) own the
    /// slot.  Callers reach this through trait dispatch on whatever
    /// descr the JUMP synthesised carries.
    fn set_external_jump_target(&self, _target: DescrRef) {
        panic!(
            "set_external_jump_target invoked on a FailDescr that does \
             not carry the per-emission external_jump_target slot \
             (only Resume-family descrs synthesised for cross-loop \
             JUMP exits in cranelift collect_guards own it)"
        );
    }

    /// history.py:137-139: exits_early()
    /// Is this guard a guard_early_exit or moved before one?
    fn exits_early(&self) -> bool {
        false
    }

    /// history.py:141-143: loop_version()
    /// Should a loop version be compiled out of this guard?
    fn loop_version(&self) -> bool {
        false
    }

    /// Identifier of the compiled trace that owns this exit.
    ///
    /// Backends that lower loops and bridges as separate compiled traces use
    /// this to let the frontend distinguish root-loop exits from bridge exits.
    fn trace_id(&self) -> u64 {
        0
    }

    /// Stamp the owning trace's identifier onto a resume-guard descr.
    ///
    /// Pyre-only: RPython resolves descr identity by Python `id(descr)`
    /// (`history.py:125`), so the same `ResumeGuardDescr` object the
    /// metainterp stamps is what `cpu.get_latest_descr()` later returns.
    /// Pyre's runtime exit path uses a `(trace_id, fail_index)` lookup
    /// key (`runner.rs::find_descr` / `compiler.rs::find_descr_by_ptr`)
    /// that needs the owning trace_id captured on the descr; the
    /// `record_loop_or_bridge` walker (`compile.py:185-186`) is the
    /// natural stamp site since it already has `trace_id` in scope and
    /// already dispatches on `is_resume_guard()`.
    ///
    /// Default panic — only `ResumeGuardDescr`-family descrs override.
    fn set_trace_id(&self, _trace_id: u64) {
        panic!(
            "set_trace_id invoked on a FailDescr that does not carry a \
             trace_id slot (compile.py:185 isinstance(descr, ResumeDescr) \
             gates the stamp; only ResumeGuardDescr-family descrs accept it)"
        );
    }

    /// Per-trace fail-index assigned by `compile.rs::build_guard_metadata`.
    ///
    /// Pyre-only: pyre's runtime exit path uses a `(trace_id, fail_index)`
    /// key where `fail_index` is the per-trace numbering the optimizer
    /// hands the backend (matching `assembler.py:227 self.faildescr.index
    /// = i` semantics).  The descr's structural `fail_index` is allocated
    /// from a global counter (`alloc_fail_index`), so a separate slot
    /// captures the per-trace key for lookup parity.  Default 0 — non-
    /// resume FailDescrs are not threaded through `build_guard_metadata`.
    fn fail_index_per_trace(&self) -> u32 {
        0
    }

    /// `build_guard_metadata` per-trace fail-index stamp setter.  Default
    /// panic — only `ResumeGuardDescr`-family descrs override.
    fn set_fail_index_per_trace(&self, _fail_index: u32) {
        panic!(
            "set_fail_index_per_trace invoked on a FailDescr that does not \
             carry a per-trace fail_index slot (only ResumeGuardDescr-family \
             descrs reach the build_guard_metadata pipeline)"
        );
    }

    /// `compile.py:186` `descr.rd_loop_token = clt` line-by-line port.
    ///
    /// Returns the owning `Arc<CompiledLoopToken>` typed as `&dyn Any`
    /// (the `token_handle_any` pattern — `majit-ir` cannot reference
    /// `majit-backend::CompiledLoopToken` without a dependency cycle).
    /// Consumers in `majit-metainterp` downcast to
    /// `Arc<CompiledLoopToken>` and chain `clt.upgrade_loop_token()` to
    /// reach the owning `JitCellToken` via the weakref clt holds per
    /// `compile.py:180-181`.
    ///
    /// Default `None` — non-resume FailDescrs (`_DoneWithThisFrameDescr`
    /// family, `ExitFrameWithExceptionDescrRef`) are skipped by
    /// `compile.py:185 isinstance(descr, ResumeDescr)` and never receive
    /// a clt stamp.  `ResumeGuardDescr` (and subclasses) override this to
    /// return the captured `Arc<CompiledLoopToken>`; the bridge-source
    /// path consumes the metainterp `AbstractFailDescr` Arc directly so
    /// no proxy override is needed.
    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// `compile.py:186` setter for the new clt-typed slot. Default
    /// panics — only resume guard descrs accept this.  Implementations
    /// store the Arc cast back to `Arc<CompiledLoopToken>` via
    /// `downcast`; the trait keeps the parameter as
    /// `Arc<dyn Any + Send + Sync>` so `majit-ir` does not depend on
    /// `majit-backend`.
    fn set_rd_loop_token_clt(&self, _clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        panic!(
            "set_rd_loop_token_clt invoked on a non-resume FailDescr \
             (compile.py:186 only writes to ResumeGuardDescr objects)"
        );
    }

    /// `history.py:132` `AbstractFailDescr._attrs_ = ('adr_jump_offset',
    /// 'rd_locs', 'rd_loop_token', 'rd_vector_info')`.
    ///
    /// `assembler.py:966` `tok.faildescr.adr_jump_offset = addr` — the
    /// address in compiled code where the guard's conditional jump
    /// offset is stored.  `assembler.py:987` `faildescr.adr_jump_offset
    /// = 0` after `patch_jump_for_descr` redirects the guard to a
    /// bridge ("0 means patched").
    ///
    /// Default `0` — every `AbstractFailDescr` instance has the slot
    /// (`_attrs_`), but only `ResumeGuardDescr`-family guards reach the
    /// backend codegen path that stamps it via
    /// `patch_pending_failure_recoveries` (`assembler.py:849`).
    fn adr_jump_offset(&self) -> usize {
        0
    }

    /// `assembler.py:966,987` write side.  Default panics — non-`ResumeDescr`
    /// FailDescrs (`DoneWithThisFrame*`, `ExitFrameWithExceptionDescrRef`,
    /// `PropagateExceptionDescr`) never go through
    /// `patch_pending_failure_recoveries` and so never receive a stamp.
    fn set_adr_jump_offset(&self, _offset: usize) {
        panic!(
            "set_adr_jump_offset invoked on a FailDescr that does not \
             carry the AbstractFailDescr.adr_jump_offset slot \
             (history.py:132 — only ResumeGuardDescr-family guards \
             reach assembler.py:849 patch_pending_failure_recoveries)"
        );
    }

    /// `history.py:132` `AbstractFailDescr._attrs_` `rd_locs` —
    /// `llsupport/assembler.py:279 guardtok.faildescr.rd_locs =
    /// positions` writes the per-fail-arg jitframe slot positions as a
    /// `Vec<u16>`.  `llsupport/llmodel.py:424 descr.rd_locs[index] *
    /// WORD` reads to compute the absolute jitframe offset during
    /// `get_value_direct`.
    ///
    /// Default `&[]` — every `AbstractFailDescr` instance has the slot,
    /// empty by default; populated by
    /// `llsupport/assembler.py:225 write_failure_recovery_description`.
    fn rd_locs(&self) -> &[u16] {
        &[]
    }

    /// `llsupport/assembler.py:279` write side.  Default panics — only
    /// `ResumeGuardDescr`-family guards reach this writer.
    fn set_rd_locs(&self, _locs: Vec<u16>) {
        panic!(
            "set_rd_locs invoked on a FailDescr that does not carry \
             the AbstractFailDescr.rd_locs slot (history.py:132)"
        );
    }

    /// Whether the given exit slot should be treated as a real GC root.
    ///
    /// Backends may override this to distinguish rooted refs from opaque
    /// handles that reuse `Type::Ref`, such as FORCE_TOKEN values.
    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        if !matches!(self.fail_arg_types().get(slot), Some(Type::Ref)) {
            return false;
        }
        // Exclude force-token positions: their Type::Ref typing is
        // synthetic — they carry opaque virtualizable handles (FORCE_TOKEN
        // op output), not real GC pointers.  The retired
        // `CraneliftFailDescr` wrapper performed this exclusion explicitly
        // before computing `fail_descr_gc_map`; the metainterp
        // `ResumeGuardDescr` now owns the slot list directly, so the
        // exclusion moves to this trait default and is inherited by every
        // Resume-family impl that overrides `force_token_slots()`.
        !self.force_token_slots().contains(&slot)
    }

    /// Exit slot indices that carry opaque force-token handles.
    ///
    /// Returns owned `Vec<usize>` (cloned per call) — the
    /// the cranelift implementation to `FORCE_TOKEN_SLOTS_TABLE`, a
    /// backend-static side-table that cannot hand out a borrow under a
    /// lock.
    fn force_token_slots(&self) -> Vec<usize> {
        Vec::new()
    }

    /// Pyre-only per-emission write of the force-token slot list.
    /// `assembler.py:write_failure_recovery_description` bakes the
    /// equivalent GC map into machine code at codegen time per
    /// emission; the slot list is the cranelift analog and follows
    /// the same per-emission classification as `rd_locs`
    /// (`assembler.py:279`).
    ///
    /// Default panics — only `ResumeGuardDescr`-family carries the
    /// slot.  Callers must gate by `is_resume_guard() ||
    /// is_resume_guard_copied()` before invoking, mirroring
    /// `set_trace_id` / `set_rd_*` / `set_adr_jump_offset`.
    /// Implementations must sort+dedup so consumers can `binary_search`.
    fn set_force_token_slots(&self, _slots: Vec<usize>) {
        panic!(
            "set_force_token_slots invoked on a FailDescr that does not \
             carry the per-emission force_token_slots slot (only \
             ResumeGuardDescr / ResumeGuardCopiedDescr own it)"
        );
    }

    /// Pyre-only per-emission failure counter.  PyPy carries the
    /// equivalent jitcounter hash in the per-descr `status` slot
    /// (`compile.py:683` `AbstractResumeGuardDescr._attrs_ =
    /// ('status',)`) — both `ResumeGuardDescr` and
    /// `ResumeGuardCopiedDescr` get their own status by inheritance
    /// from `AbstractResumeGuardDescr`.  Pyre's counter follows the
    /// same per-emission classification.  Default returns 0 for
    /// descrs that never compile bridges.
    fn fail_count(&self) -> u32 {
        0
    }

    /// Pyre-only per-emission failure-count increment.  Returns the
    /// post-increment value.  Default panics — only Resume-family
    /// guards carry the counter (`compile.py:683`
    /// `AbstractResumeGuardDescr._attrs_ = ('status',)`).  Callers
    /// must gate by `is_resume_guard() || is_resume_guard_copied()`.
    fn increment_fail_count(&self) -> u32 {
        panic!(
            "increment_fail_count invoked on a FailDescr that does not \
             carry a fail_count slot (compile.py:683 `_attrs_` only on \
             AbstractResumeGuardDescr / ResumeGuardCopiedDescr)"
        );
    }

    /// Pyre-only per-emission `CompiledTraceInfo` slot.  Returned as
    /// `Arc<dyn Any + Send + Sync>` so the trait can live in
    /// `majit-ir` without depending on `majit-backend`'s
    /// `CompiledTraceInfo`; concrete callers downcast.  Per-emission
    /// classification matches `record_loop_or_bridge`
    /// (`compile.py:185-186`) which stamps the loop-token-equivalent
    /// data on each emitted descr; copied descrs in the same trace
    /// share the same `CompiledTraceInfo` value but write into their
    /// own slot.  Default `None`.
    fn trace_info_any(&self) -> Option<std::sync::Arc<dyn std::any::Any + Send + Sync>> {
        None
    }

    /// Pyre-only per-emission `CompiledTraceInfo` slot write.  Caller
    /// passes an `Arc<CompiledTraceInfo>` upcast to `Arc<dyn Any>`;
    /// the impl downcasts.  Default panics — only Resume-family
    /// guards own the slot.  Callers must gate by `is_resume_guard()
    /// || is_resume_guard_copied()`.
    fn set_trace_info_any(&self, _info: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        panic!(
            "set_trace_info_any invoked on a FailDescr that does not \
             carry the per-emission trace_info slot (only Resume-family \
             guards reached by record_loop_or_bridge at compile.py:185)"
        );
    }

    /// Pyre-only cranelift bridge-attach cell addresses.  Returns
    /// `(code_ptr_addr, frame_depth_addr)` — heap-pinned `usize`
    /// addresses cranelift's `emit_attached_bridge_dispatch` bakes
    /// into machine code as immediates.  Per-emission: each emitted
    /// descr (including `ResumeGuardCopiedDescr`) can have a bridge
    /// attached, so the cell addresses must be distinct per emission.
    /// Default `None` for descrs that never carry bridges (singletons
    /// / synthetic FINISH).
    fn bridge_cache_addrs(&self) -> Option<(usize, usize)> {
        None
    }

    /// Pyre-only cranelift bridge code-pointer read.  Returns 0 when
    /// no bridge is attached.  Per-emission classification (see
    /// `bridge_cache_addrs`).
    fn bridge_code_ptr(&self) -> usize {
        0
    }

    /// Pyre-only cranelift bridge cache publish.  Atomic-stores
    /// `(code_ptr, body_ptr)` into the cells whose addresses
    /// `bridge_cache_addrs` reported.  Default panics — only
    /// Resume-family guards carry bridge cache cells.  Callers
    /// gate by `is_resume_guard() || is_resume_guard_copied()`.
    fn store_bridge_caches(&self, _code_ptr: usize, _body_ptr: usize) {
        panic!(
            "store_bridge_caches invoked on a FailDescr that does not \
             carry the per-emission bridge cache cells (only \
             ResumeGuardDescr / ResumeGuardCopiedDescr own them)"
        );
    }

    /// Pyre-only cranelift bridge dispatch payload read (type-erased
    /// raw pointer, backend re-Arcs via `Arc::from_raw`).  Null when
    /// no bridge has been attached.  Default null.
    fn bridge_dispatch_load(&self) -> *mut () {
        std::ptr::null_mut()
    }

    /// Pyre-only cranelift bridge dispatch payload publish.  Atomic-
    /// swaps `new_ptr` in and registers `drop_fn` for cleanup at
    /// descr teardown.  Returns the previous payload so the caller
    /// can reclaim.
    ///
    /// `drop_fn` is `unsafe fn` because it reconstructs an `Arc` from
    /// the raw pointer the caller published; the contract between the
    /// publisher and `drop_fn` is unsafe by construction.
    ///
    /// Default panics — only Resume-family guards own the dispatch
    /// cell.  Silent default would immediately leak `new_ptr`
    /// (caller already transferred ownership by `Arc::into_raw`).
    /// Callers must gate by `is_resume_guard() ||
    /// is_resume_guard_copied()`.
    fn bridge_dispatch_swap(&self, _new_ptr: *mut (), _drop_fn: unsafe fn(*mut ())) -> *mut () {
        panic!(
            "bridge_dispatch_swap invoked on a FailDescr that does not \
             carry the per-emission bridge_dispatch_cell (only \
             ResumeGuardDescr / ResumeGuardCopiedDescr own it); the \
             silent default would leak the supplied new_ptr"
        );
    }

    /// Pyre-only per-emission slot: index of the trace op that produced
    /// this guard at codegen.  Classified per-emission alongside
    /// `history.py:132 AbstractFailDescr._attrs_` `rd_locs` /
    /// `adr_jump_offset` — `assembler.py:279`
    /// `guardtok.faildescr.rd_locs = positions` writes onto the per-
    /// emitted descr without chasing `prev`, and the source op index
    /// shares that classification (one codegen emission = one op).
    /// Default `None` for non-resume FailDescrs.  Overridden on both
    /// `ResumeGuardDescr` and `ResumeGuardCopiedDescr` so the cranelift
    /// backend never has to chase `prev_descr` to find the storage —
    /// the chase would conflate multiple `ResumeGuardCopiedDescr`
    /// siblings sharing a donor (optimizer.py:691 / optimizeopt/mod.rs).
    fn source_op_index(&self) -> Option<usize> {
        None
    }

    /// Pyre-only per-emission slot write.  See `source_op_index` for
    /// the per-emission rationale.  Default panics — only Resume-family
    /// guards own the slot.  Callers must gate by
    /// `is_resume_guard() || is_resume_guard_copied()`.
    fn set_source_op_index(&self, _source_op_index: usize) {
        panic!(
            "set_source_op_index invoked on a FailDescr that does not \
             carry the per-emission source_op_index slot (only \
             ResumeGuardDescr / ResumeGuardCopiedDescr own it)"
        );
    }

    /// Pyre-only per-emission slot: this guard is the eval-breaker word's
    /// back-edge poll rather than a check on traced values.
    ///
    /// Its failures are scheduled exits — the collector armed the word and the
    /// loop has to leave machine code for the request to be serviced — so they
    /// describe when a collection landed, not what the compiled code assumed.
    /// The statistics counter reads this to keep the two apart
    /// (`crate::eval_breaker_word::is_back_edge_poll_guard` decides it).
    /// Default `false` for non-resume FailDescrs.
    fn is_back_edge_poll(&self) -> bool {
        false
    }

    /// Pyre-only per-emission slot write.  See `is_back_edge_poll`.  Default
    /// panics — only Resume-family guards own the slot.  Callers must gate by
    /// `is_resume_guard() || is_resume_guard_copied()`.
    fn set_back_edge_poll(&self) {
        panic!(
            "set_back_edge_poll invoked on a FailDescr that does not \
             carry the per-emission back_edge_poll slot (only \
             ResumeGuardDescr / ResumeGuardCopiedDescr own it)"
        );
    }

    /// `compile.py:683` `AbstractResumeGuardDescr._attrs_ = ('status',)`
    /// — packs `ST_BUSY_FLAG` + type tag + hash on the resume-guard
    /// descr.  `compile.py:741-745` `self.status` read for
    /// `must_compile`.
    fn get_status(&self) -> u64 {
        0
    }

    /// `compile.py:786-788` `start_compiling — self.status |=
    /// ST_BUSY_FLAG`.
    fn start_compiling(&self) {}

    /// `compile.py:790-795` `done_compiling — self.status &=
    /// ~ST_BUSY_FLAG`.
    fn done_compiling(&self) {}

    /// `compile.py:750` check `ST_BUSY_FLAG`.
    fn is_compiling(&self) -> bool {
        false
    }

    /// `compile.py:826-830` `store_hash(metainterp_sd)` — write the
    /// jitcounter hash bits (status with `ST_SHIFT_MASK` applied).
    fn store_hash(&self, _hash: u64) {}

    /// `compile.py:813-824` `make_a_counter_per_value(op, index)` —
    /// pack `type_tag | (index << ST_SHIFT)` into status.
    fn make_a_counter_per_value(&self, _index: u32, _type_tag: u64) {}

    /// history.py:143-147 / schedule.py:654-655 — attach vector resume info
    /// to a guard descriptor. Non-guard fail descriptors ignore this.
    ///
    /// Upstream shape (history.py:143-147):
    /// ```python
    /// def attach_vector_info(self, info):
    ///     info.prev = self.rd_vector_info
    ///     self.rd_vector_info = info
    /// ```
    /// Implementations store the head in their internal
    /// `Option<Box<AccumInfo>>` chain; `AccumInfo.prev` carries the tail.
    fn attach_vector_info(&self, _info: AccumInfo) {}

    /// Read back any attached vector resume info.
    ///
    /// TODO: upstream `descr.rd_vector_info` is a
    /// head-linked singly-linked `Option<AccumInfo>` chain (walk via
    /// `.prev`). Pyre stores the head-linked chain internally
    /// (`AccumInfo.prev` field matches upstream), but this trait
    /// method materializes the walk as a `Vec<AccumInfo>` for
    /// callers that prefer index access. Head is at index 0 (most
    /// recently attached); earlier entries follow via
    /// `entries[i].prev`. Consumers that want strict parity can
    /// walk the chain manually by taking `vector_info().into_iter()
    /// .next()` (the head) and following `prev`.
    fn vector_info(&self) -> Vec<AccumInfo> {
        Vec::new()
    }

    /// `compile.py:869-870 ResumeGuardDescr.copy_all_attributes_from`:
    /// `self.rd_vector_info = other.rd_vector_info.clone()`. Receiver
    /// replaces its own chain in place; identity (descr's
    /// `fail_index` / `status`) is preserved.  `chain` is the donor's
    /// flattened chain (head at index 0).  Implementations rebuild
    /// the linked list and write it through their internal cell.
    /// Non-guard fail descriptors ignore this.
    fn replace_vector_info(&self, _chain: Vec<AccumInfo>) {}
}

/// resume.py:65-85 AccumInfo — metadata attached to guard descriptors
/// so deoptimization can reconstruct vector accumulators.
///
/// Two distinct OpRefs following RPython's separation:
///   - `variable`: resume.py:29/47 — the original scalar accumulator box
///     (used for type inference; getoriginal() returns it).
///   - `location`: resume.py:28 — the register/SSA location where the
///     accumulated vector lives. regalloc.py:350 sets accuminfo.location;
///     the backend reads it for extractlane + reduction at guard exit.
///
/// Field layout matches resume.py:24-85 (VectorInfo base + AccumInfo
/// subclass flattened into one struct, since pyre's vector pass only
/// produces the Accum variant). The sibling `UnpackAtExitInfo` subclass
/// is provided below as a dead reservation matching RPython — RPython
/// itself defines `UnpackAtExitInfo` (resume.py:59-63) but never
/// instantiates it; this is the same cross-cutting pattern as
/// `RawStructPtrInfo` (info.py:452, never instantiated).
#[derive(Debug, Clone)]
pub struct AccumInfo {
    /// resume.py:31/32 prev — next entry in the VectorInfo linked list.
    /// RPython stores a head-linked chain on `descr.rd_vector_info` and
    /// traverses it via `next()` / `clone()`. `None` marks list tail.
    pub prev: Option<Box<AccumInfo>>,
    /// resume.py:27 failargs_pos — index in the guard's fail arguments.
    pub failargs_pos: usize,
    /// resume.py:29 variable — the original scalar variable (getoriginal()).
    pub variable: OpRef,
    /// resume.py:28 location — register/SSA location of the accumulated
    /// vector. regalloc.py:350 sets this.
    pub location: OpRef,
    /// resume.py:66/70 accum_operation — reduction operator ('+', '*', ...).
    pub accum_operation: char,
    /// resume.py:71 scalar — the reduced scalar value. RPython sets this
    /// lazily during backend reduction at guard exit (assembler.py:739).
    /// `OpRef::NONE` = unset (matches RPython `self.scalar = None`).
    pub scalar: OpRef,
}

/// resume.py:59-63 UnpackAtExitInfo — VectorInfo subclass used for
/// scheduling vector value unpack-on-exit at guard failures.
///
/// **Dead reservation** — RPython defines this class but never
/// instantiates it (`grep UnpackAtExitInfo(` in `rpython/jit/` returns
/// only the `instance_clone` self-reference at resume.py:61). The type
/// is preserved here for line-by-line structural parity with
/// `resume.py`; same cross-cutting pattern as `RawStructPtrInfo`
/// (info.py:452-457). Should a future vector pass introduce a
/// producer, the call sites should mirror RPython's `AccumInfo`
/// handling (linked-list `prev` chain + `attach_vector_info`).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct UnpackAtExitInfo {
    /// resume.py:31/32 prev — next entry in the VectorInfo linked list.
    pub prev: Option<Box<UnpackAtExitInfo>>,
    /// resume.py:27 failargs_pos — index in the guard's fail arguments.
    pub failargs_pos: usize,
    /// resume.py:29 variable — the original scalar variable.
    pub variable: OpRef,
    /// resume.py:28 location — register/SSA location of the value.
    pub location: OpRef,
}

fn push_vector_info(head: &mut Option<Box<AccumInfo>>, mut info: AccumInfo) {
    info.prev = head.take();
    *head = Some(Box::new(info));
}

fn flatten_vector_info(head: Option<&AccumInfo>) -> Vec<AccumInfo> {
    let mut result = Vec::new();
    let mut current = head;
    while let Some(info) = current {
        result.push(info.clone());
        current = info.prev.as_deref();
    }
    result
}

/// Frontend decoder from a [`SizeDescr::vtable`] pointer to the canonical
/// `PyObject.w_class` object its instances carry (`get_instantiate(vtable)`).
///
/// Pluggable for the same reason as [`crate::value::set_str_resolver`]: the
/// object model is the frontend's, and the IR layer holds no `PyType` layout.
/// Consulted live rather than cached on the descr — the type objects are
/// installed after the descrs are built, so a value snapshotted at
/// construction time would be null for every descr.
pub type WClassObjFn = fn(vtable: usize) -> Option<i64>;

static W_CLASS_OBJ_RESOLVER: std::sync::OnceLock<WClassObjFn> = std::sync::OnceLock::new();

/// Register the frontend's [`WClassObjFn`].  First call wins, mirroring
/// `OnceLock::set`.  The resolver only ever receives a `vtable` a
/// `SizeDescr` of this frontend reported, so it may assume its own type
/// layout.
pub fn set_w_class_obj_resolver(resolve: WClassObjFn) {
    let _ = W_CLASS_OBJ_RESOLVER.set(resolve);
}

/// Resolve `vtable` through the registered [`WClassObjFn`], or `None` when
/// the vtable is null or no frontend registered one.
pub fn resolve_w_class_obj(vtable: usize) -> Option<i64> {
    if vtable == 0 {
        return None;
    }
    W_CLASS_OBJ_RESOLVER
        .get()
        .and_then(|resolve| resolve(vtable))
}

/// Descriptor for a fixed-size struct/object allocation.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py SizeDescr.
pub trait SizeDescr: Descr {
    /// Total size in bytes.
    fn size(&self) -> usize;

    /// Type ID (for GC header).
    fn type_id(&self) -> u32;

    /// `gc_cache._cache_size[LLType::Struct(key)]` cache key — the
    /// original `path_hash(module_path::Struct)` u64 (not the u32 GC
    /// `type_id` allocated by `gc_cache.init_size_descr`).  Distinct
    /// concepts: `type_id` is the dense sequential gc tid backends use
    /// for `gc.alloc_*_typed`; `cache_key` is the structural identity
    /// the cache slot is keyed on.  PyPy collapses both into the
    /// `STRUCT` lltype object identity; pyre keeps them separate so
    /// the keyed cache resolves `LLType::Struct(cache_key)` regardless
    /// of the dense tid value.  Default 0 — concrete impls that own a
    /// stamp slot should override.  Inverse `bh_size_spec_from_descr`
    /// reader (`assembler.rs`) uses this for the `BhSizeSpec.type_id`
    /// field to keep the analyzer and runtime cache key namespaces
    /// aligned.
    fn cache_key(&self) -> u64 {
        0
    }

    /// Whether this is an immutable object.
    fn is_immutable(&self) -> bool;

    /// Whether this is an object (has vtable).
    fn is_object(&self) -> bool {
        false
    }

    /// Whether the described struct carries a GC header (a `ref - 8`
    /// type-id word).  True for JIT-GC-allocated structs (`new_struct` /
    /// `new_with_vtable`), false for a natively-allocated raw struct
    /// (`register_struct_layout`).  `StructPtrInfo.make_guards` gates
    /// `GUARD_GC_TYPE` on this: a header-less raw struct cannot be
    /// runtime-type-pinned (RPython emits `GUARD_GC_TYPE` only for a real
    /// `GcStruct`).  Default `true` so existing descrs keep their guard;
    /// only the raw-struct path overrides to `false`.  Independent of
    /// `is_object()` because a `new_struct` GC struct also has
    /// `vtable == 0`.
    fn is_gc_managed(&self) -> bool {
        true
    }

    /// Whether `NEW` for this descr should be rewritten to a headerless
    /// nursery allocation.  Separate from `is_gc_managed()`: headerless
    /// allocation changes the rewriter/backend malloc shape, while
    /// `is_gc_managed()` gates runtime type-header guards.
    fn headerless(&self) -> bool {
        false
    }

    /// Whether `NEW` for this descr must allocate in the non-moving old
    /// generation instead of the movable nursery.
    ///
    /// Set it for a type whose objects the host reaches through raw
    /// pointers it does not register as GC roots: a minor collection
    /// inside any residual call would move such an object out from under
    /// those pointers, and the caller would then read and write the dead
    /// pre-move copy.  The flag exists so a JIT-emitted allocation can
    /// match the allocator the host itself uses for that type; a type
    /// whose host allocator is the ordinary nursery one must leave it
    /// `false`, since the old generation is never collected as cheaply.
    fn non_moving(&self) -> bool {
        false
    }

    /// Vtable address, if is_object().
    fn vtable(&self) -> usize {
        0
    }

    /// Pyre object-model: the canonical Python class object pointer
    /// (`PyObject.w_class`) that instances of this type carry, i.e.
    /// `get_instantiate(vtable_type)`. A `new_with_vtable` virtual of a
    /// builtin type inherits this value unless the trace stores an
    /// explicit `w_class` field. OptVirtualize folds `w_class` header
    /// reads to this constant. `None`/`0` for non-pyre size descrs or
    /// before the type objects are initialised.  A frontend whose size
    /// descrs are the generic [`SimpleSizeDescr`] supplies the answer
    /// through [`set_w_class_obj_resolver`] instead of overriding this.
    fn w_class_obj(&self) -> Option<i64> {
        None
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "SizeDescr(size={}, type_id={})",
            self.size(),
            self.type_id()
        )
    }

    /// descr.py:76 `get_all_fielddescrs(self): return self.all_fielddescrs`.
    /// Populated by `heaptracker.all_fielddescrs(gccache, STRUCT)` at
    /// `get_size_descr` construction time (descr.py:125-126).  That walker
    /// is `majit-translate codewriter::heaptracker::all_fielddescrs`, one
    /// crate above this one, so concrete impls thread the list in via a
    /// builder; the default is empty.
    fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &[]
    }

    /// descr.py:71 `self.gc_fielddescrs = gc_fielddescrs`, populated by
    /// `heaptracker.gc_fielddescrs(gccache, STRUCT)` which is
    /// `all_fielddescrs(only_gc=True)` (heaptracker.py:94-95 + :70 filter
    /// `isinstance(FIELD, lltype.Ptr) and FIELD._needsgc()`).  Concrete
    /// impls precompute the subset; the default filter here is for
    /// impls that omit the override.
    fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &[]
    }
}

/// Type-erased marker for `VirtualizableInfo`. Upstream parity:
/// pyjitpl.py:1148-1158 `emit_force_virtualizable` begins with
/// `vinfo = fielddescr.get_vinfo()` — in RPython every field descriptor
/// carries a backreference to the owning `VirtualizableInfo`.
///
/// `VirtualizableInfo` itself lives in `majit-metainterp`, which already
/// depends on `majit-ir`; reversing that dependency would be circular.
/// This trait is the minimal Rust bridge: `majit-metainterp` implements
/// `VinfoMarker for VirtualizableInfo`, and `FieldDescr::get_vinfo()`
/// returns `Option<Arc<dyn VinfoMarker>>` — callers that need the
/// concrete `VirtualizableInfo` downcast via `as_any()`.
pub trait VinfoMarker: std::fmt::Debug + Send + Sync + std::any::Any {
    /// Downcast helper so callers can recover `&VirtualizableInfo`.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Descriptor for a field within a struct.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py FieldDescr.
pub trait FieldDescr: Descr {
    /// descr.py / FieldDescr.get_parent_descr() — the SizeDescr of the
    /// containing struct/object that owns this field. PyPy
    /// Byte offset from the start of the struct.
    fn offset(&self) -> usize;

    /// Size of the field in bytes.
    fn field_size(&self) -> usize;

    /// Type of value stored in this field.
    fn field_type(&self) -> Type;

    /// Whether this is a pointer field (needs GC tracking).
    fn is_pointer_field(&self) -> bool {
        self.field_type() == Type::Ref
    }

    /// Whether this is a float field.
    fn is_float_field(&self) -> bool {
        self.field_type() == Type::Float
    }

    /// Whether reads from this field are signed.
    fn is_field_signed(&self) -> bool {
        true
    }

    /// `descr.py:151 FieldDescr.flag` — `get_type_flag(FIELDTYPE)`
    /// (`descr.py:240-252`).
    ///
    /// The default reconstructs the classification from the IR type the way
    /// `get_type_flag` does, which is exact for the pointer / float / integer
    /// cases; implementations that store the flag verbatim override it and so
    /// also keep `FLAG_STRUCT` and `FLAG_VOID`.
    fn field_flag(&self) -> ArrayFlag {
        match self.field_type() {
            Type::Ref => ArrayFlag::Pointer,
            Type::Float => ArrayFlag::Float,
            Type::Void => ArrayFlag::Void,
            Type::Int if self.is_field_signed() => ArrayFlag::Signed,
            Type::Int => ArrayFlag::Unsigned,
        }
    }

    /// Whether this field is immutable (never written after object creation).
    ///
    /// Immutable field reads from a constant object can be folded to constants,
    /// and their cached values survive cache invalidation by calls/side effects.
    /// Delegates to `Descr::is_always_pure()` by default.
    fn is_immutable(&self) -> bool {
        self.is_always_pure()
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "FieldDescr(offset={}, size={}, type={:?})",
            self.offset(),
            self.field_size(),
            self.field_type()
        )
    }

    /// descr.py: index_in_parent — position within parent struct.
    fn index_in_parent(&self) -> usize {
        0
    }

    /// descr.py: FieldDescr.get_parent_descr() — backreference to the
    /// SizeDescr of the containing struct/object. Required by
    /// `OptContext::ensure_ptr_info_arg0` to dispatch Instance vs Struct
    /// PtrInfo per `optimizer.py:478-484`. Default returns `None`; field
    /// descriptors that don't carry a backreference fall through to the
    /// generic path and the Rust port's `ensure_ptr_info_arg0` panics
    /// rather than installing a malformed PtrInfo.
    fn get_parent_descr(&self) -> Option<DescrRef> {
        None
    }

    /// pyjitpl.py:1148-1149 `vinfo = fielddescr.get_vinfo()` — backreference
    /// to the `VirtualizableInfo` that owns this field. Populated only
    /// for field descriptors built by `VirtualizableInfo::finalize_arc`;
    /// all other descriptors fall through to the default `None` and the
    /// upstream `assert vinfo is not None` is enforced at the call site
    /// (`emit_force_virtualizable`).
    fn get_vinfo(&self) -> Option<Arc<dyn VinfoMarker>> {
        None
    }

    /// descr.py:227 — field name. Format is either:
    /// - `"STRUCT.fieldname"` (from codewriter: descr.py:227)
    /// - `"object.typeptr"` (the vtable field, `make_vtable_field_descr`)
    /// - `""` (unnamed/dynamic field descriptors)
    fn field_name(&self) -> &str {
        ""
    }

    /// descr.py:220-233 cache key (`fieldname`). Defaults to the
    /// display name for descriptor implementations without a separate key.
    fn field_key(&self) -> &str {
        self.field_name()
    }

    /// The class pointer is the object-header field at offset 0
    /// (`rpython/rtyper/rclass.py:167-174`, `OBJECT.typeptr`). Its
    /// descriptors use two field spellings, each bare or struct-qualified:
    /// - `"typeptr"` / `"STRUCT.typeptr"` (codewriter format, descr.py:227)
    /// - `"ob_type"` / `"STRUCT.ob_type"` (e.g. `"PyObject.ob_type"`)
    fn is_typeptr(&self) -> bool {
        let name = self.field_name();
        name == "typeptr"
            || name.ends_with(".typeptr")
            || name == "ob_type"
            || name.ends_with(".ob_type")
    }

    /// Pyre object-model: `PyObject.w_class` (offset 8) carries the
    /// Python-level class identity, distinct from the `typeptr`/vtable
    /// (offset 0). Like `typeptr`, it is a header field — not a value
    /// field that may be indexed by `index_in_parent` against a
    /// virtual's stored fields. Recognised by name so OptVirtualize can
    /// resolve it from the object's class identity instead of colliding
    /// with the first value field.
    ///
    /// Handles both formats:
    /// - `"w_class"` (pyre tracer w_class_descr)
    /// - `"STRUCT.w_class"` (e.g. "PyObject.w_class")
    fn is_w_class(&self) -> bool {
        let name = self.field_name();
        name == "w_class" || name.ends_with(".w_class")
    }

    /// descr.py: sort_key() — for ordering field descriptors.
    fn sort_key(&self) -> usize {
        self.offset()
    }
}

/// RPython: descr.py FLAG_* constants for array element type classification.
///
/// ```python
/// FLAG_POINTER  = 'P'  # GC pointer (Ptr to gc obj)
/// FLAG_FLOAT    = 'F'  # Float or longlong
/// FLAG_UNSIGNED = 'U'  # Unsigned integer
/// FLAG_SIGNED   = 'S'  # Signed integer
/// FLAG_STRUCT   = 'X'  # Inline struct (array-of-structs)
/// FLAG_VOID     = 'V'  # Void
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArrayFlag {
    /// RPython: FLAG_POINTER = 'P'
    Pointer,
    /// RPython: FLAG_FLOAT = 'F'
    Float,
    /// RPython: FLAG_UNSIGNED = 'U'
    Unsigned,
    /// RPython: FLAG_SIGNED = 'S'
    Signed,
    /// RPython: FLAG_STRUCT = 'X'
    Struct,
    /// RPython: FLAG_VOID = 'V'
    Void,
}

impl ArrayFlag {
    /// RPython: get_type_flag(TYPE) (descr.py:241-254).
    ///
    /// When only the IR type is known (no concrete Rust type string),
    /// `Type::Int` maps to `Unsigned` — RPython's default for unknown
    /// integer types (descr.py:254: `return FLAG_UNSIGNED`).
    /// Use `get_type_flag()` in call.rs for precise signed/unsigned
    /// classification from concrete type names.
    pub fn from_item_type(item_type: Type, is_struct: bool) -> Self {
        if is_struct {
            return ArrayFlag::Struct;
        }
        match item_type {
            Type::Ref => ArrayFlag::Pointer,
            Type::Float => ArrayFlag::Float,
            // RPython: default for unresolved integer type is FLAG_UNSIGNED
            // (descr.py:254). Callers with concrete type info should use
            // get_type_flag() for FLAG_SIGNED/FLAG_UNSIGNED distinction.
            Type::Int => ArrayFlag::Unsigned,
            Type::Void => ArrayFlag::Void,
        }
    }

    /// descr.py:241-254: get_type_flag(FIELDTYPE) for FieldDescr.
    ///
    /// For fields, `Type::Int` maps to `Signed` — RPython's default
    /// integer type is `lltype.Signed` which gets FLAG_SIGNED. This
    /// differs from arrays where the default is FLAG_UNSIGNED.
    pub fn from_field_type(field_type: Type) -> Self {
        match field_type {
            Type::Ref => ArrayFlag::Pointer,
            Type::Float => ArrayFlag::Float,
            // RPython: Signed → FLAG_SIGNED (descr.py:248)
            Type::Int => ArrayFlag::Signed,
            Type::Void => ArrayFlag::Void,
        }
    }
}

/// Descriptor for an array type.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py ArrayDescr.
pub trait ArrayDescr: Descr {
    /// Size of the fixed header (before array items).
    fn base_size(&self) -> usize;

    /// Size of each array item in bytes.
    fn item_size(&self) -> usize;

    /// Type ID (for GC header).
    fn type_id(&self) -> u32;

    /// `gc.py:544-549 init_array_descr` post-mint tid setter.  Default
    /// no-op (PyPy's base `ArrayDescr.tid = 0` field is plain assign);
    /// concrete pyre `SimpleArrayDescr` overrides to atomic store so
    /// the analyzer's `arraydescrof` cache-or-mint path can stamp
    /// `path_hash(array_type_id) as u32` onto a shared
    /// `Arc<SimpleArrayDescr>` after `gc_cache.get_array_descr`
    /// resolves.
    fn set_type_id(&self, _id: u32) {}

    /// `gc_cache._cache_array[LLType::Array(key)]` cache key — the
    /// original `path_hash(array_type_id)` u64 (not the u32 GC `type_id`
    /// allocated by `gc_cache.init_array_descr`).  Distinct concepts:
    /// `type_id` is the dense sequential gc tid; `cache_key` is the
    /// structural identity the cache slot is keyed on.  PyPy collapses
    /// both into the `ARRAY` lltype object identity; pyre keeps them
    /// separate so the keyed cache resolves `LLType::Array(cache_key)`
    /// regardless of the dense tid value.  Default 0 — concrete impls
    /// that own a stamp slot should override.  Inverse `BhDescr::Array`
    /// producers (`assembler.rs`, `jitcode.rs`) use this for the
    /// `type_id: u64` field so the runtime descr-back-to-spec round-trip
    /// lands on the same `_cache_array` slot.
    fn cache_key(&self) -> u64 {
        0
    }

    /// Type of each array item.
    fn item_type(&self) -> Type;

    /// Whether items are GC pointers.
    fn is_array_of_pointers(&self) -> bool {
        self.item_type() == Type::Ref
    }

    /// Whether items are floats.
    fn is_array_of_floats(&self) -> bool {
        self.item_type() == Type::Float
    }

    /// Whether integer items should be sign-extended on loads.
    ///
    /// RPython array descriptors distinguish signed from unsigned integer
    /// storage. Backends should ignore this for non-integer item types.
    fn is_item_signed(&self) -> bool {
        true
    }

    /// Descriptor for the length field.
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        None
    }

    /// Whether items are primitive (integer or float, not pointer).
    /// descr.py: is_array_of_primitives()
    fn is_array_of_primitives(&self) -> bool {
        !self.is_array_of_pointers()
    }

    /// Whether items are structs (array-of-structs pattern).
    /// descr.py: is_array_of_structs() → self.flag == FLAG_STRUCT
    fn is_array_of_structs(&self) -> bool {
        false
    }

    /// Whether `NEW_ARRAY` for this descr must allocate in the non-moving
    /// old generation instead of the movable nursery.  Same contract as
    /// [`SizeDescr::non_moving`], for the varsize path.
    fn non_moving(&self) -> bool {
        false
    }

    /// Whether the array is a real GC-managed array (has a GC header so
    /// `GUARD_GC_TYPE` may read a type-id at `ptr - GcHeader::SIZE`).
    /// Default `true` so existing array descrs keep their guard; only a
    /// header-less raw native pointer-array (e.g. a `pool_arrays` base,
    /// minted by `add_ptr_array_descr`) overrides to `false`.  Mirrors
    /// `SizeDescr::is_gc_managed` for the struct path.
    fn is_gc_managed(&self) -> bool {
        true
    }

    /// descr.py:291 ArrayDescr.get_all_fielddescrs() →
    /// all_interiorfielddescrs. For array-of-structs, returns
    /// interior field descriptors.
    fn get_all_interiorfielddescrs(&self) -> Option<&[DescrRef]> {
        None
    }

    /// descr.py:373 `arraydescr.all_interiorfielddescrs = descrs` —
    /// post-construction publish for struct-array interior field
    /// descriptors.  Concrete impls (`SimpleArrayDescr`) override with
    /// a `OnceLock` set-once semantic so `cpu.arraydescrof(ARRAY)`
    /// can return the cached `Arc<dyn ArrayDescr>` first (matching
    /// `Arc::ptr_eq(interior.array_descr, returned_array_descr)` per
    /// `descr.py:388 InteriorFieldDescr.__init__`) and the interior
    /// list back-references the same Arc.  Default no-op for descrs
    /// without an `OnceLock` slot.
    fn set_all_interiorfielddescrs(&self, _descrs: Vec<DescrRef>) {}

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "ArrayDescr(base={}, item={}, type={:?})",
            self.base_size(),
            self.item_size(),
            self.item_type()
        )
    }
}

/// Descriptor for a field within an array element (interior pointer).
///
/// Mirrors rpython/jit/backend/llsupport/descr.py InteriorFieldDescr.
pub trait InteriorFieldDescr: Descr {
    fn array_descr(&self) -> &dyn ArrayDescr;
    fn field_descr(&self) -> &dyn FieldDescr;
}

/// Descriptor for a function call.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py CallDescr.
pub trait CallDescr: Descr {
    /// Types of the arguments.
    fn arg_types(&self) -> &[Type];

    /// Type of the return value.
    fn result_type(&self) -> Type;

    /// Size of the return value in bytes.
    fn result_size(&self) -> usize;

    /// Whether the result is a signed integer.
    fn is_result_signed(&self) -> bool {
        true
    }

    /// Target compiled loop token for `CALL_ASSEMBLER_*`, if this call
    /// descriptor represents a nested JIT-to-JIT call.
    fn call_target_token(&self) -> Option<u64> {
        None
    }

    /// RPython JitDriverSD.index_of_virtualizable for CALL_ASSEMBLER.
    ///
    /// When present, identifies the virtualizable argument inside the
    /// original call_assembler arglist before rewrite.py shrinks it to
    /// `[frame]` or `[frame, virtualizable]`.
    fn call_virtualizable_index(&self) -> Option<usize> {
        None
    }

    /// descr.py:511 `get_extra_info()` — returns the EffectInfo describing
    /// side effects, oopspec classification, and descriptor read/write sets.
    fn get_extra_info(&self) -> &EffectInfo;

    /// Argument class string (RPython encoding: 'i'=int, 'r'=ref, 'f'=float).
    /// descr.py: arg_classes
    fn arg_classes(&self) -> String {
        self.arg_types()
            .iter()
            .map(|t| match t {
                Type::Int => 'i',
                Type::Ref => 'r',
                Type::Float => 'f',
                Type::Void => 'v',
            })
            .collect()
    }

    /// Result type as arg class character.
    fn result_class(&self) -> char {
        match self.result_type() {
            Type::Int => 'i',
            Type::Ref => 'r',
            Type::Float => 'f',
            Type::Void => 'v',
        }
    }

    /// Number of arguments.
    fn num_args(&self) -> usize {
        self.arg_types().len()
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "CallDescr(args={}, result={:?})",
            self.arg_classes(),
            self.result_type()
        )
    }
}

/// Descriptor carrying a sub-jitcode reference (RPython
/// `JitCode(AbstractDescr)`). Emitted by the codewriter for
/// `inline_call_*` opcodes to identify the callee's bytecode body.
///
/// Walker / blackhole consumers retrieve the callee body via
/// `jitcode_index()` indexing into the runtime's all-jitcodes table
/// (cf. `pyre-jit-trace/src/jitcode_runtime.rs::ALL_JITCODES`).
pub trait JitCodeDescr: Descr {
    /// Index of the callee's bytecode body in the runtime's
    /// all-jitcodes table. RPython parity:
    /// `AssemblerDescr::PendingJitCode { jitcode, .. }` → resolved
    /// `BhDescr::JitCode { jitcode_index, .. }` at snapshot time.
    fn jitcode_index(&self) -> usize;
}

/// Descriptor carrying `switch/id` dispatch metadata.
///
/// RPython `jitcode.py:131-143 SwitchDictDescr` stores both
/// `dict: {int: target_pc}` for lookup and `const_keys_in_order`
/// for deterministic miss-path guard generation. Keep both surfaces
/// here instead of reconstructing order from a Rust `HashMap`.
pub trait SwitchDescr: Descr {
    /// RPython `switchdict.dict[search_value]`.
    fn lookup(&self, value: i64) -> Option<usize>;

    /// RPython `switchdict.const_keys_in_order`.
    fn const_keys_in_order(&self) -> &[i64];
}

/// Descriptor carrying a `CALL_ASSEMBLER` loop token.
///
/// RPython routes these ops through `JitCellToken` itself
/// (`rewrite.py:667 assert isinstance(loop_token, JitCellToken)`), so the
/// token-specific queries live outside generic `CallDescr`.
pub trait LoopTokenDescr: Descr {
    /// history.py:443 `JitCellToken.number`.
    fn loop_token_number(&self) -> u64;

    /// rewrite.py:685-689 `jd.index_of_virtualizable`.
    fn call_virtualizable_index(&self) -> Option<usize> {
        None
    }

    /// `compile.py:187 original.record_jump_to(descr)` parity hook.
    ///
    /// Upstream `op.getdescr()` IS a `JitCellToken` object — `record_jump_to`
    /// receives it directly without any number-to-object lookup. majit's
    /// `LoopTokenDescr` trait lives in `majit-ir`, which cannot reference the
    /// `JitCellToken` type defined in `majit-backend`. Implementations that
    /// own an `Arc<JitCellToken>` expose it here as a `&dyn Any` so consumers
    /// in `majit-metainterp` can downcast to recover the owning Arc — matching
    /// upstream's "descr IS the loop token" identity contract without forcing
    /// the trait into a backend dependency cycle.
    ///
    /// Default `None` — only `MetaCallAssemblerDescr` and other production
    /// descriptors that carry a real compiled-loop token override.
    fn token_handle_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

/// Descriptor for `DebugMergePoint` operations — carries source position
/// information at merge points (bytecode boundaries in the traced interpreter).
///
/// Mirrors rpython/jit/metainterp/resoperation.py DebugMergePoint.
/// RPython's meta-interpreter emits these at each bytecode boundary
/// during tracing. They carry:
/// - The JitDriver name (which interpreter generated this trace)
/// - A source-level representation (e.g., "bytecode 42 in function foo")
/// - The call depth (for inlined functions)
///
/// These are used by jitviewer and profiling tools to map compiled code
/// back to the source interpreter's bytecode positions.
#[derive(Clone, Debug)]
pub struct DebugMergePointInfo {
    /// Name of the JitDriver that generated this trace.
    /// E.g., "pypyjit" for PyPy's main interpreter.
    pub jd_name: String,
    /// Source-level representation: a human-readable string identifying
    /// the position in the traced interpreter's code.
    /// E.g., "bytecode LOAD_FAST at offset 12 in function foo".
    pub source_repr: String,
    /// Bytecode index (program counter value) in the traced interpreter.
    pub bytecode_index: i64,
    /// Call depth: 0 for the outermost (root) trace, incremented for
    /// each level of inlined function calls.
    pub call_depth: u32,
}

impl DebugMergePointInfo {
    pub fn new(
        jd_name: impl Into<String>,
        source_repr: impl Into<String>,
        bytecode_index: i64,
        call_depth: u32,
    ) -> Self {
        DebugMergePointInfo {
            jd_name: jd_name.into(),
            source_repr: source_repr.into(),
            bytecode_index,
            call_depth,
        }
    }
}

/// Concrete descriptor wrapping `DebugMergePointInfo` for attachment to IR ops.
#[derive(Debug)]
pub struct DebugMergePointDescr {
    pub info: DebugMergePointInfo,
}

impl DebugMergePointDescr {
    pub fn new(info: DebugMergePointInfo) -> Self {
        DebugMergePointDescr { info }
    }
}

impl Descr for DebugMergePointDescr {
    fn repr(&self) -> String {
        format!(
            "debug_merge_point({}, '{}', pc={}, depth={})",
            self.info.jd_name,
            self.info.source_repr,
            self.info.bytecode_index,
            self.info.call_depth
        )
    }
}

/// `rpython/jit/metainterp/virtualizable.py:73` —
/// `VirtualizableInfo.array_field_descrs[i]` is a `FieldDescr` for the
/// frame field that holds the i-th virtualizable array's pointer.
/// `jtransform.py:1880-1885 do_fixed_list_getitem` and `:1898-1906
/// do_fixed_list_setitem` (vable branches) emit it as the
/// second-to-last operand of `getarrayitem_vable_X` /
/// `setarrayitem_vable_X` and as one of the trailing descrs on
/// `arraylen_vable`.
///
/// Pyre's bytecode jointly encodes the `(array_field_descr, array_descr)`
/// pair as `array_idx:u16`, so this struct stores only the per-array
/// index `i` for the assembler dispatch to recover. Today pyre's
/// `PyFrame` has a single virtualizable array (`locals_cells_stack_w`,
/// idx=0), but the struct is shaped to allow multi-array virtualizables.
#[derive(Debug)]
pub struct VableArrayFieldDescr {
    pub idx: u16,
    /// `effectinfo.py:496` `descr.ei_index = sys.maxint` — singleton, so
    /// the storage lives in the descriptor itself rather than a side
    /// table. Initialised to `u32::MAX` and rewritten by
    /// `compute_bitstrings`.
    ei_index: AtomicU32,
}

impl Descr for VableArrayFieldDescr {
    fn repr(&self) -> String {
        format!("vable_array_field_descr[{}]", self.idx)
    }
    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }
    fn set_ei_index(&self, index: u32) {
        self.ei_index.store(index, Ordering::Relaxed);
    }
}

/// `rpython/jit/metainterp/virtualizable.py:58` —
/// `VirtualizableInfo.array_descrs[i]` is the `ArrayDescr` for the
/// GcArray that the i-th `array_field_descr` points at. Always paired
/// with a `VableArrayFieldDescr` at the same `i` in jtransform's
/// `vable_array_vars` table; the pair appears at the trailing two
/// operand positions of every `getarrayitem_vable_X` /
/// `setarrayitem_vable_X` / `arraylen_vable` SpaceOperation.
#[derive(Debug)]
pub struct VableArrayDescr {
    pub idx: u16,
    /// `effectinfo.py:496` `descr.ei_index = sys.maxint`. See
    /// `VableArrayFieldDescr::ei_index`.
    ei_index: AtomicU32,
}

impl Descr for VableArrayDescr {
    fn repr(&self) -> String {
        format!("vable_array_descr[{}]", self.idx)
    }
    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }
    fn set_ei_index(&self, index: u32) {
        self.ei_index.store(index, Ordering::Relaxed);
    }
}

/// Number of `OnceLock<DescrRef>` slots reserved per array-descr index.
/// `rpython/jit/metainterp/virtualizable.py:58/73` sizes `array_descrs` /
/// `array_field_descrs` to `len(array_fields)`. Pyre's PyFrame has one
/// array (`locals_cells_stack_w`); the state-field JIT may declare several
/// `[int; virt]` arrays. Bump this if a virtualizable grows past it.
const VABLE_ARRAY_DESCR_SLOTS: usize = 8;

/// Singleton accessor for `array_field_descrs[idx]`.
///
/// `rpython/jit/metainterp/virtualizable.py:73` constructs one
/// `FieldDescr` per array field at `VirtualizableInfo.__init__` time
/// and caches it on the `VirtualizableInfo` instance — every later
/// reference (`vable_array_vars[var]` in jtransform, the optimizer,
/// the assembler) uses Python object identity to dedup. Pyre mirrors
/// this with a per-`idx` `OnceLock<DescrRef>` so `Arc::ptr_eq` over two
/// `vable_array_field_descr(idx)` calls with matching `idx` returns true.
pub fn vable_array_field_descr(idx: u16) -> DescrRef {
    static SLOTS: [OnceLock<DescrRef>; VABLE_ARRAY_DESCR_SLOTS] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    let i = idx as usize;
    assert!(
        i < VABLE_ARRAY_DESCR_SLOTS,
        "vable_array_field_descr: idx={idx} exceeds \
         VABLE_ARRAY_DESCR_SLOTS={VABLE_ARRAY_DESCR_SLOTS}",
    );
    SLOTS[i]
        .get_or_init(|| {
            Arc::new(VableArrayFieldDescr {
                idx,
                ei_index: AtomicU32::new(u32::MAX),
            }) as DescrRef
        })
        .clone()
}

/// Singleton accessor for `array_descrs[idx]` — counterpart of
/// `vable_array_field_descr` carrying the GcArray layout descr.
/// Same identity-preservation invariant via a per-`idx` `OnceLock<DescrRef>`.
pub fn vable_array_descr(idx: u16) -> DescrRef {
    static SLOTS: [OnceLock<DescrRef>; VABLE_ARRAY_DESCR_SLOTS] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    let i = idx as usize;
    assert!(
        i < VABLE_ARRAY_DESCR_SLOTS,
        "vable_array_descr: idx={idx} exceeds \
         VABLE_ARRAY_DESCR_SLOTS={VABLE_ARRAY_DESCR_SLOTS}",
    );
    SLOTS[i]
        .get_or_init(|| {
            Arc::new(VableArrayDescr {
                idx,
                ei_index: AtomicU32::new(u32::MAX),
            }) as DescrRef
        })
        .clone()
}

/// `rpython/jit/metainterp/virtualizable.py:71` —
/// `VirtualizableInfo.static_field_descrs[i]` is a `FieldDescr` for
/// the i-th scalar (non-array) field of the virtualizable struct.
/// `jtransform.py:846` (getfield) and `jtransform.py:927` (setfield)
/// emit it as the trailing descr operand of `getfield_vable_<kind>`
/// (after `v_inst`) and `setfield_vable_<kind>` (after `v_inst,
/// v_value`).
///
/// `interp_jit.py:25-30` has 5 scalar fields in fixed order:
/// `[last_instr, pycode, valuestackdepth, debugdata, w_globals]`, so
/// legitimate `idx` values are `0..=4`. The canonical table is
/// `pyre-jit-trace/src/virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`.
/// The struct stores only the per-field
/// index; bytecode emission and runtime field access still go
/// through the field-idx-to-offset table maintained by
/// `virtualizable_spec.rs`.
#[derive(Debug)]
pub struct VableStaticFieldDescr {
    pub idx: u16,
    /// `effectinfo.py:496` `descr.ei_index = sys.maxint`. See
    /// `VableArrayFieldDescr::ei_index`.
    ei_index: AtomicU32,
}

impl Descr for VableStaticFieldDescr {
    fn repr(&self) -> String {
        format!("vable_static_field_descr[{}]", self.idx)
    }
    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }
    fn set_ei_index(&self, index: u32) {
        self.ei_index.store(index, Ordering::Relaxed);
    }
}

/// Number of `OnceLock<DescrRef>` slots reserved for
/// `vable_static_field_descr(idx)` singletons. Matches the exact
/// scalar-field count of PyFrame's virtualizable
/// (`interp_jit.py:25-30`: `last_instr, pycode, valuestackdepth,
/// debugdata, w_globals`), with the canonical table at
/// `pyre-jit-trace/src/virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`.
/// This mirrors
/// `rpython/jit/metainterp/virtualizable.py:71`'s
/// `static_field_descrs = [... for name in static_fields]` which
/// is sized exactly to `len(static_fields)`. Bump this when the
/// PyFrame `_virtualizable_` declaration grows.
const VABLE_STATIC_FIELD_DESCR_SLOTS: usize = 5;

/// Singleton accessor for `static_field_descrs[idx]`.
///
/// `rpython/jit/metainterp/virtualizable.py:71` builds one
/// `FieldDescr` per scalar field eagerly at `VirtualizableInfo
/// .__init__` and caches it; every later jtransform / optimizer /
/// assembler reference uses Python object identity to dedup.  Pyre
/// mirrors this with a per-`idx` `OnceLock<DescrRef>` so
/// `Arc::ptr_eq` over two `vable_static_field_descr(idx)` calls with
/// matching `idx` returns true.
pub fn vable_static_field_descr(idx: u16) -> DescrRef {
    static SLOTS: [OnceLock<DescrRef>; VABLE_STATIC_FIELD_DESCR_SLOTS] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];
    let i = idx as usize;
    assert!(
        i < VABLE_STATIC_FIELD_DESCR_SLOTS,
        "vable_static_field_descr: idx={} exceeds VABLE_STATIC_FIELD_DESCR_SLOTS={}; \
         pyre's PyFrame _virtualizable_ declares only {} static fields \
         (interp_jit.py:25-30)",
        idx,
        VABLE_STATIC_FIELD_DESCR_SLOTS,
        VABLE_STATIC_FIELD_DESCR_SLOTS,
    );
    SLOTS[i]
        .get_or_init(|| {
            Arc::new(VableStaticFieldDescr {
                idx,
                ei_index: AtomicU32::new(u32::MAX),
            }) as DescrRef
        })
        .clone()
}

// EffectInfo / ExtraEffect / OopSpecIndex moved to `crate::effectinfo`
// (mirroring rpython/jit/codewriter/effectinfo.py).
pub use crate::effectinfo::{EffectInfo, ExtraEffect, OopSpecIndex};

// ── Concrete descriptor implementations (descr.py) ──

/// `descr.py:227 name = '%s.%s' % (STRUCT._name, fieldname)` — locate the
/// external `_cache_field[STRUCT][fieldname]` key inside the display name so
/// every descriptor does not need a second owned string. Both the struct name
/// and pyre's synthetic nested-field spelling may contain dots, so deriving the
/// split from the finished display name would be ambiguous; retain the exact
/// byte offset supplied while the two inputs are still separate.
fn field_key_start(name: &str, field_key: &str) -> u32 {
    let start = name
        .len()
        .checked_sub(field_key.len())
        .filter(|&start| name.get(start..) == Some(field_key))
        .expect("FieldDescr display name must end with its cache field key");
    u32::try_from(start).expect("FieldDescr display-name prefix must fit in u32")
}

/// Simple concrete FieldDescr for use by pyre-jit and tests.
/// RPython: `FieldDescr(name, offset, size, flag, index_in_parent, is_pure)`.
#[derive(Debug)]
pub struct SimpleFieldDescr {
    /// Per-trace codewriter slot id (`descr_indices.field_index` from
    /// `CallControl`). Pyre adaptation — PyPy's `FieldDescr` has no
    /// equivalent (PyPy keys raw EI sets on Python `id(descr)` which
    /// Rust models via `Arc::ptr_eq`). Atomic so the cache-or-mint
    /// path (`gc_cache.get_field_descr`) can stamp the analyzer's
    /// `idx` onto a shared `Arc<SimpleFieldDescr>` after a cache hit,
    /// converging analyzer and runtime (`__majit_register_descrs`)
    /// onto the same Arc instance while preserving the analyzer's
    /// per-trace identity for `BhFieldSpec.index` round-trips in
    /// `pyre-jit-trace::state` (line 5879 / 5933).
    index: AtomicU32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    /// `effectinfo.py:496` `descr.ei_index = sys.maxint` — initialised to
    /// `u32::MAX` (sentinel matching `sys.maxint`); rewritten by
    /// `compute_bitstrings` (`effectinfo.py:524-526`).
    ei_index: AtomicU32,
    /// RPython: FieldDescr.name — e.g. "MyStruct.field_name"
    name: Box<str>,
    /// Byte offset of `descr.py:220-233`'s external cache key (`fieldname`)
    /// inside `name`. RPython keeps the key only in `_cache_field`; pyre needs
    /// to expose it for its descriptor-identity census, so retain a compact
    /// slice position instead of duplicating the owned string.
    field_key_start: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_immutable: bool,
    /// RPython `rpython/rtyper/rclass.py` `IR_QUASIIMMUTABLE[_ARRAY]` rank.
    /// Consumed by `FieldDescr::is_quasi_immutable()` (trait default overridden
    /// below) so `rewrite_op_getfield` can emit the `record_quasiimmut_field`
    /// guard pair from `rpython/jit/codewriter/jtransform.py:895-903`.
    is_quasi_immutable: bool,
    /// descr.py:151: FieldDescr.flag — type classification from get_type_flag().
    /// FLAG_POINTER, FLAG_FLOAT, FLAG_SIGNED, FLAG_UNSIGNED, FLAG_STRUCT, FLAG_VOID.
    flag: ArrayFlag,
    virtualizable: bool,
    /// descr.py:158 FieldDescr.index — slot position within the
    /// parent struct's `all_fielddescrs`.
    pub index_in_parent: usize,
    /// descr.py:238 FieldDescr.parent_descr — backreference to the SizeDescr
    /// of the containing struct/object. Required by
    /// `OptContext::ensure_ptr_info_arg0` to dispatch Instance vs Struct
    /// PtrInfo per `optimizer.py:478-484`. Stored as `Weak` to break the
    /// SizeDescr → FieldDescr → SizeDescr Arc cycle introduced by
    /// `make_simple_descr_group`. Interior-mutable because
    /// `descr.py:238` updates the single parent backreference while the
    /// field Arc is shared by `(STRUCT, fieldname)`.
    pub parent_descr: RwLock<Option<Weak<dyn Descr>>>,
    /// pyjitpl.py:1148-1149 `vinfo = fielddescr.get_vinfo()` — backref to
    /// the owning `VirtualizableInfo`. Stored as `Weak<dyn VinfoMarker>`
    /// because the vinfo Arc keeps the descriptor alive (via its
    /// `_static_field_descrs` / `_array_field_descrs` / `vable_token_descr`
    /// slots); a strong back-ref would form a reference cycle. Populated
    /// by `VirtualizableInfo::finalize_arc` via `Arc::new_cyclic`.
    pub vinfo: Option<Weak<dyn VinfoMarker>>,
}

impl Clone for SimpleFieldDescr {
    fn clone(&self) -> Self {
        SimpleFieldDescr {
            index: AtomicU32::new(self.index.load(Ordering::Relaxed)),
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            ei_index: AtomicU32::new(self.ei_index.load(Ordering::Relaxed)),
            name: self.name.clone(),
            field_key_start: self.field_key_start,
            offset: self.offset,
            field_size: self.field_size,
            field_type: self.field_type,
            is_immutable: self.is_immutable,
            is_quasi_immutable: self.is_quasi_immutable,
            flag: self.flag,
            virtualizable: self.virtualizable,
            index_in_parent: self.index_in_parent,
            parent_descr: RwLock::new(self.parent_descr.read().unwrap().clone()),
            vinfo: self.vinfo.clone(),
        }
    }
}

impl SimpleFieldDescr {
    /// Whether this descr already describes the field a `get_field_descr`
    /// caller is asking for.
    ///
    /// `descr.py:218-239` derives every one of these from `(STRUCT,
    /// fieldname)` itself — `symbolic.get_field_token`, `get_type_flag`,
    /// `STRUCT._hints['_immutable_fields_']`,
    /// `heaptracker.get_fielddescr_index_in` — so a cache hit there cannot
    /// describe a different field than the caller means. Pyre takes them as
    /// arguments instead, so two call sites *can* disagree, and the cache
    /// silently keeps whichever minted first. That is never a legitimate
    /// state: `heaptracker.py:76-101 get_fielddescr_index_in` is positional
    /// and `optimizeopt/info.rs force_box` asserts on it, so a disagreeing
    /// `index_in_parent` alone puts the two halves of the descr universe on
    /// different slots of the same object.
    ///
    /// `index` is deliberately excluded — it is the per-trace codewriter slot
    /// id, which the analyzer legitimately restamps onto a shared `Arc`.
    #[expect(
        clippy::too_many_arguments,
        reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
    )]
    pub fn describes_same_field(
        &self,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
        is_quasi_immutable: bool,
        virtualizable: bool,
        index_in_parent: usize,
    ) -> bool {
        self.offset == offset
            && self.field_size == field_size
            && self.field_type == field_type
            && self.is_immutable == is_immutable
            && self.is_quasi_immutable() == is_quasi_immutable
            && self.virtualizable == virtualizable
            && self.index_in_parent == index_in_parent
    }

    pub fn new(
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
    ) -> Self {
        // descr.py:241-254: get_type_flag(FIELDTYPE) — derive flag from IR type.
        // Default: Int→Signed (RPython Signed), Ref→Pointer, Float→Float.
        let flag = ArrayFlag::from_field_type(field_type);
        SimpleFieldDescr {
            index: AtomicU32::new(index),
            descr_index: AtomicI32::new(-1),
            ei_index: AtomicU32::new(u32::MAX),
            name: Box::default(),
            field_key_start: 0,
            offset,
            field_size,
            field_type,
            is_immutable,
            is_quasi_immutable: false,
            flag,
            virtualizable: false,
            index_in_parent: 0,
            parent_descr: RwLock::new(None),
            vinfo: None,
        }
    }

    /// RPython: FieldDescr(name, offset, size, flag, index_in_parent, is_pure).
    /// `name` format: `"STRUCT.fieldname"` (descr.py:227).
    /// `flag`: descr.py:226 get_type_flag(FIELDTYPE).
    #[expect(
        clippy::too_many_arguments,
        reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
    )]
    pub fn new_with_name(
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
        flag: ArrayFlag,
        name: String,
        field_key: String,
    ) -> Self {
        let field_key_start = field_key_start(&name, &field_key);
        SimpleFieldDescr {
            index: AtomicU32::new(index),
            descr_index: AtomicI32::new(-1),
            ei_index: AtomicU32::new(u32::MAX),
            name: name.into_boxed_str(),
            field_key_start,
            offset,
            field_size,
            field_type,
            is_immutable,
            is_quasi_immutable: false,
            flag,
            virtualizable: false,
            index_in_parent: 0,
            parent_descr: RwLock::new(None),
            vinfo: None,
        }
    }

    /// RPython `rpython/rtyper/rclass.py:644-678` — `IR_QUASIIMMUTABLE[_ARRAY]`.
    /// Flipped by the descriptor builder when the field participated in a
    /// quasi-immutable declaration.  Drives
    /// `FieldDescr::is_quasi_immutable()` below.
    pub fn with_quasi_immutable(mut self, is_quasi_immutable: bool) -> Self {
        self.is_quasi_immutable = is_quasi_immutable;
        self
    }

    /// descr.py:151: set flag directly.
    pub fn with_flag(mut self, flag: ArrayFlag) -> Self {
        self.flag = flag;
        self
    }

    /// descr.py:158 `self.index = index_in_parent` — the field's
    /// positional index within its struct, returned by `get_index()`
    /// and read back by `InteriorFieldDescr.get_index()` (descr.py:393).
    pub fn with_index_in_parent(mut self, index_in_parent: usize) -> Self {
        self.index_in_parent = index_in_parent;
        self
    }

    /// Compat shim: with_signed(true) → FLAG_SIGNED, with_signed(false) → FLAG_UNSIGNED.
    pub fn with_signed(mut self, signed: bool) -> Self {
        self.flag = if signed {
            ArrayFlag::Signed
        } else {
            ArrayFlag::Unsigned
        };
        self
    }

    pub fn with_virtualizable(mut self, virtualizable: bool) -> Self {
        self.virtualizable = virtualizable;
        self
    }

    /// Builder: attach a parent SizeDescr backreference + index_in_parent.
    /// Required when the descriptor will be used as the `op.descr` of a
    /// GETFIELD/SETFIELD/QUASIIMMUT_FIELD that flows through
    /// `ensure_ptr_info_arg0` (optimizer.py:478-484).
    pub fn with_parent_descr(mut self, parent: DescrRef, index_in_parent: usize) -> Self {
        self.parent_descr = RwLock::new(Some(Arc::downgrade(&parent)));
        self.index_in_parent = index_in_parent;
        self
    }

    /// descr.py:238 — update this field's single parent backreference.
    pub fn set_parent_descr(&self, parent: &DescrRef) {
        *self.parent_descr.write().unwrap() = Some(Arc::downgrade(parent));
    }

    /// Builder: attach the owning `VirtualizableInfo` backreference that
    /// `FieldDescr::get_vinfo()` returns. `vinfo` is stored as a `Weak`
    /// reference; upgrades succeed for as long as the owning vinfo Arc
    /// is alive. Upstream: pyjitpl.py:1148-1149 `fielddescr.get_vinfo()`.
    pub fn with_vinfo(mut self, vinfo: Weak<dyn VinfoMarker>) -> Self {
        self.vinfo = Some(vinfo);
        self
    }
}

impl Descr for SimpleFieldDescr {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn index(&self) -> u32 {
        self.index.load(Ordering::Relaxed)
    }
    fn set_index(&self, index: u32) {
        self.index.store(index, Ordering::Relaxed);
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }
    fn set_ei_index(&self, ei_index: u32) {
        self.ei_index.store(ei_index, Ordering::Relaxed);
    }
    fn is_always_pure(&self) -> bool {
        // RPython `jtransform.py:895-896`: a quasi-immutable field is *not*
        // always-pure at the descriptor level — the pure-read is protected by
        // `record_quasiimmut_field` + guard.  Only true immutable fields are
        // unconditionally pure.
        self.is_immutable && !self.is_quasi_immutable
    }
    fn is_quasi_immutable(&self) -> bool {
        self.is_quasi_immutable
    }
    fn is_virtualizable(&self) -> bool {
        self.virtualizable
    }
    fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
        Some(self)
    }
}

impl FieldDescr for SimpleFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    /// descr.py:173: is_pointer_field() → self.flag == FLAG_POINTER
    fn is_pointer_field(&self) -> bool {
        self.flag == ArrayFlag::Pointer
    }
    /// descr.py:176: is_float_field() → self.flag == FLAG_FLOAT
    fn is_float_field(&self) -> bool {
        self.flag == ArrayFlag::Float
    }
    /// descr.py:179: is_field_signed() → self.flag == FLAG_SIGNED
    fn is_field_signed(&self) -> bool {
        self.flag == ArrayFlag::Signed
    }
    /// descr.py:151: the stored `FieldDescr.flag`, so `FLAG_STRUCT` /
    /// `FLAG_VOID` survive instead of being re-derived from the IR type.
    fn field_flag(&self) -> ArrayFlag {
        self.flag
    }
    fn is_immutable(&self) -> bool {
        self.is_immutable
    }
    fn field_name(&self) -> &str {
        &self.name
    }
    fn field_key(&self) -> &str {
        &self.name[self.field_key_start as usize..]
    }
    fn index_in_parent(&self) -> usize {
        self.index_in_parent
    }
    fn get_parent_descr(&self) -> Option<DescrRef> {
        self.parent_descr
            .read()
            .unwrap()
            .as_ref()
            .and_then(|p| p.upgrade())
    }
    fn get_vinfo(&self) -> Option<Arc<dyn VinfoMarker>> {
        self.vinfo.as_ref().and_then(|w| w.upgrade())
    }
}

/// Simple concrete SizeDescr.
#[derive(Debug)]
pub struct SimpleSizeDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    size: usize,
    type_id: AtomicU32,
    /// `gc_cache._cache_size[LLType::Struct(cache_key)]` keyed identity
    /// — the original u64 `path_hash(module_path::Struct)` (not the
    /// dense u32 `type_id` GC tid allocated post-mint).  Stamped at
    /// `gc_cache.get_size_descr` cache-miss-mint so the inverse
    /// `bh_size_spec_from_descr` reader can recover the same key the
    /// analyzer-side `bh_size_spec_from_callcontrol` produces (without
    /// this slot, the inverse path returned `type_id` widened to u64,
    /// landing on a different cache slot and breaking round-trip
    /// identity).
    cache_key: u64,
    /// Census marker for a fieldless shell minted by `get_size_descr` before
    /// any producer published the struct's positional field list.
    fieldless_shell_mint: bool,
    /// descr.py:64,112: SizeDescr.immutable_flag
    pub is_immutable: bool,
    vtable: usize,
    /// True when the described struct carries a GC header (allocated by
    /// the JIT GC via `new_struct` / `new_with_vtable`), so a `ref - 8`
    /// type-id word exists for `GuardGcType` to read.  False for a
    /// natively-allocated raw struct registered via
    /// `register_struct_layout` (no header).  Mirrors RPython's
    /// `lltype.GcStruct` vs raw `Struct` distinction: a header-less raw
    /// struct must not be runtime-type-pinned (`StructPtrInfo.make_guards`
    /// emits `GUARD_GC_TYPE` only for a real GC type).  Independent of
    /// `is_object()` (`vtable != 0`) because a `new_struct` GC struct
    /// also has `vtable == 0`.
    is_gc_managed: bool,
    /// Explicit allocation-shape bit for `NEW` rewriting.  Defaults false:
    /// normal JIT-GC structs are headered unless an emitter opts in.
    headerless: bool,
    /// Allocation-generation bit for `NEW` rewriting; see
    /// [`SizeDescr::non_moving`].  Atomic so a frontend can stamp it onto
    /// an already-shared descr, the way `descr_index` is stamped.
    non_moving: AtomicBool,
    /// descr.py:72 `self.all_fielddescrs = all_fielddescrs`.
    all_fielddescrs: Vec<Arc<dyn FieldDescr>>,
    /// descr.py:71 `self.gc_fielddescrs = gc_fielddescrs`.
    /// Precomputed subset of `all_fielddescrs` via `is_pointer_field()`
    /// (heaptracker.py:94-95 `gc_fielddescrs = all_fielddescrs(only_gc=True)`
    /// + heaptracker.py:70 `FIELD._needsgc()` filter).
    gc_fielddescrs: Vec<Arc<dyn FieldDescr>>,
}

impl Clone for SimpleSizeDescr {
    fn clone(&self) -> Self {
        SimpleSizeDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            size: self.size,
            type_id: AtomicU32::new(self.type_id.load(Ordering::Relaxed)),
            cache_key: self.cache_key,
            fieldless_shell_mint: self.fieldless_shell_mint,
            is_immutable: self.is_immutable,
            vtable: self.vtable,
            is_gc_managed: self.is_gc_managed,
            headerless: self.headerless,
            non_moving: AtomicBool::new(self.non_moving.load(Ordering::Relaxed)),
            all_fielddescrs: self.all_fielddescrs.clone(),
            gc_fielddescrs: self.gc_fielddescrs.clone(),
        }
    }
}

impl SimpleSizeDescr {
    pub fn new(index: u32, size: usize, type_id: u32) -> Self {
        SimpleSizeDescr {
            index,
            descr_index: AtomicI32::new(-1),
            size,
            type_id: AtomicU32::new(type_id),
            cache_key: 0,
            fieldless_shell_mint: false,
            is_immutable: false,
            vtable: 0,
            is_gc_managed: true,
            headerless: false,
            non_moving: AtomicBool::new(false),
            all_fielddescrs: Vec::new(),
            gc_fielddescrs: Vec::new(),
        }
    }

    pub fn with_vtable(index: u32, size: usize, type_id: u32, vtable: usize) -> Self {
        SimpleSizeDescr {
            index,
            descr_index: AtomicI32::new(-1),
            size,
            type_id: AtomicU32::new(type_id),
            cache_key: 0,
            fieldless_shell_mint: false,
            is_immutable: false,
            vtable,
            is_gc_managed: true,
            headerless: false,
            non_moving: AtomicBool::new(false),
            all_fielddescrs: Vec::new(),
            gc_fielddescrs: Vec::new(),
        }
    }

    /// Stamp the `gc_cache._cache_size[LLType::Struct(...)]` identity
    /// onto this descr.  Called by `gc_cache.get_size_descr` cache-miss
    /// path after `init_size_descr` allocates the dense GC `type_id`.
    pub fn set_cache_key(&mut self, key: u64) {
        self.cache_key = key;
    }

    fn mark_fieldless_shell_mint(&mut self) {
        self.fieldless_shell_mint = true;
    }

    /// Override the GC-header flag (default `true` from the constructors).
    /// Set `false` for a natively-allocated raw struct registered via
    /// `register_struct_layout` (no `ref - 8` type-id word, so
    /// `GuardGcType` must not be emitted for it).
    pub fn set_gc_managed(&mut self, is_gc_managed: bool) {
        self.is_gc_managed = is_gc_managed;
    }

    /// Override the allocation-shape flag (default `false`).
    pub fn set_headerless(&mut self, headerless: bool) {
        self.headerless = headerless;
    }

    /// Override the allocation-generation flag (default `false`); see
    /// [`SizeDescr::non_moving`].  Takes `&self` so a frontend can stamp
    /// it after the descr is already behind an `Arc` and registered.
    pub fn set_non_moving(&self, non_moving: bool) {
        self.non_moving.store(non_moving, Ordering::Relaxed);
    }

    /// descr.py:123-126 — `get_size_descr` calls
    /// `heaptracker.gc_fielddescrs(...)` / `heaptracker.all_fielddescrs(...)`
    /// and stores both onto the descriptor.  That walker lives one crate
    /// above this one (`majit-translate codewriter::heaptracker`), so
    /// callers thread `all_fielddescrs` in via this builder; the
    /// `gc_fielddescrs` subset is derived by filtering on
    /// `FieldDescr::is_pointer_field()` (heaptracker.py:70).
    pub fn with_all_fielddescrs(mut self, all_fielddescrs: Vec<Arc<dyn FieldDescr>>) -> Self {
        self.gc_fielddescrs = all_fielddescrs
            .iter()
            .filter(|fd| fd.is_pointer_field())
            .cloned()
            .collect();
        self.all_fielddescrs = all_fielddescrs;
        self
    }

    /// Add a GC edge that the positional `all_fielddescrs` list does not
    /// name.  `heaptracker.py:50-73 all_fielddescrs` recurses into the
    /// inherited header so upstream's `gc_fielddescrs` covers it; pyre's
    /// runtime object groups declare only the concrete payload, and the
    /// allocation-clear census still has to see the embedded `PyObject`
    /// pointer edge.  Kept out of `all_fielddescrs` so the positional
    /// indexing above is unaffected.
    pub fn with_extra_gc_fielddescr(mut self, fd: Arc<dyn FieldDescr>) -> Self {
        if !self
            .gc_fielddescrs
            .iter()
            .any(|old| old.offset() == fd.offset())
        {
            self.gc_fielddescrs.push(fd);
        }
        self
    }

    /// gc.py:541: descr.tid = llop.combine_ushort(lltype.Signed, type_id, 0)
    /// Takes `&self` because the runtime layoutbuilder step stamps the
    /// collector id after the baked descriptor table has shared the Arc.
    pub fn set_type_id(&self, type_id: u32) {
        self.type_id.store(type_id, Ordering::Relaxed);
    }
}

impl Descr for SimpleSizeDescr {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn index(&self) -> u32 {
        self.index
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
        Some(self)
    }
}

impl SizeDescr for SimpleSizeDescr {
    fn size(&self) -> usize {
        self.size
    }
    fn type_id(&self) -> u32 {
        self.type_id.load(Ordering::Relaxed)
    }
    fn cache_key(&self) -> u64 {
        self.cache_key
    }
    fn is_immutable(&self) -> bool {
        self.is_immutable
    }
    fn all_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.all_fielddescrs
    }
    fn gc_fielddescrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.gc_fielddescrs
    }
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
    fn is_gc_managed(&self) -> bool {
        self.is_gc_managed
    }
    fn headerless(&self) -> bool {
        self.headerless
    }
    fn non_moving(&self) -> bool {
        self.non_moving.load(Ordering::Relaxed)
    }
    fn vtable(&self) -> usize {
        self.vtable
    }
    /// The generic size descr has no object model of its own, so the
    /// `w_class` identity comes from the frontend's registered
    /// [`WClassObjFn`].  Without it `OptVirtualize` cannot fold a `w_class`
    /// header read off a `new_with_vtable` virtual and forces the virtual
    /// instead, which turns the read into a load of freshly allocated,
    /// uninitialised memory and the guard on it into a per-iteration failure.
    fn w_class_obj(&self) -> Option<i64> {
        resolve_w_class_obj(self.vtable)
    }
}

#[derive(Debug, Clone)]
pub struct SimpleFieldDescrSpec {
    pub index: u32,
    /// descr.py:220-233 cache key (`fieldname`).
    pub field_key: String,
    /// descr.py:227 display name.
    pub name: String,
    pub offset: usize,
    pub field_size: usize,
    pub field_type: Type,
    pub is_immutable: bool,
    /// RPython `rpython/rtyper/rclass.py:644-678` — rank
    /// `IR_QUASIIMMUTABLE` / `IR_QUASIIMMUTABLE_ARRAY`.  Flipped by
    /// `rewrite_op_getfield` consumers to emit
    /// `record_quasiimmut_field` before the pure read
    /// (`rpython/jit/codewriter/jtransform.py:895-903`).
    pub is_quasi_immutable: bool,
    /// descr.py:151: FieldDescr.flag — get_type_flag(FIELDTYPE).
    pub flag: ArrayFlag,
    pub virtualizable: bool,
    pub index_in_parent: usize,
}

#[derive(Debug, Clone)]
pub struct SimpleDescrGroup {
    pub size_descr: Arc<SimpleSizeDescr>,
    pub field_descrs: Vec<Arc<SimpleFieldDescr>>,
}

/// Keyed sibling of [`make_simple_descr_group`] — accepts a u64
/// `cache_key` (= `path_hash(module_path::Struct)`) so the freshly
/// minted descrs land in the keyed `gc_cache._cache_size[key]` and
/// `_cache_field[key][name]` maps in addition to the snapshot order
/// Vecs.  Subsequent `gc_cache.get_size_descr(LLType::Struct(key), ...)`
/// or `get_field_descr(...)` callers (analyzer-side `cc.fielddescrof`,
/// other runtime mint paths) see the same `Arc<dyn Descr>` instead of
/// minting duplicates — restoring PyPy's `cpu.fielddescrof` per-
/// `(STRUCT, name)` identity (`descr.py:218-239`).  First-write wins;
/// the keyed map preserves the original Arc across redundant register
/// calls.
pub fn make_simple_descr_group_keyed(
    index: u32,
    size: usize,
    type_id: u32,
    cache_key: u64,
    vtable: usize,
    is_gc_managed: bool,
    field_specs: &[SimpleFieldDescrSpec],
) -> SimpleDescrGroup {
    make_simple_descr_group_keyed_with_headerless(
        index,
        size,
        type_id,
        cache_key,
        vtable,
        is_gc_managed,
        false,
        field_specs,
        &[],
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
)]
pub fn make_simple_descr_group_keyed_with_headerless(
    index: u32,
    size: usize,
    type_id: u32,
    cache_key: u64,
    vtable: usize,
    is_gc_managed: bool,
    headerless: bool,
    field_specs: &[SimpleFieldDescrSpec],
    extra_gc_fielddescrs: &[Arc<dyn FieldDescr>],
) -> SimpleDescrGroup {
    let struct_key = LLType::struct_key(cache_key);
    // Before `get_field_descr` below can normalise anything: the producer's own
    // `index_in_parent` against the position it hands the field in.
    census_spec_positions(field_specs);
    let mut gc = gc_cache().lock().unwrap();
    // descr.py:218-239 — cache-or-mint each FieldDescr by
    // `(STRUCT, fieldname)` before freezing this producer's positional list.
    let field_descrs: Vec<Arc<SimpleFieldDescr>> = field_specs
        .iter()
        .map(|spec| {
            gc.get_field_descr(
                struct_key.clone(),
                &spec.field_key,
                Some(spec.name.as_str()),
                spec.offset,
                spec.field_size,
                spec.field_type,
                spec.is_immutable,
                spec.is_quasi_immutable,
                spec.flag,
                spec.index,
                spec.virtualizable,
                // `Some` because `SimpleFieldDescrSpec.index_in_parent` is a
                // plain `usize` and CANNOT express the absence, not because
                // every spec resolved a slot: the spec factories in
                // `pyre-jit-trace/src/descr.rs` write literal `index_in_parent:
                // 0` at a dozen sites. So on this path `unresolved_claimed` is
                // an upper bound and `unresolved_placeholder` a lower one. The
                // bound is exact only for callers that hand the `Option` in
                // themselves — the deserialized-`BhDescr::Field` path.
                Some(spec.index_in_parent),
            )
        })
        .collect();
    let all_fielddescrs: Vec<Arc<dyn FieldDescr>> = field_descrs
        .iter()
        .cloned()
        .map(|field_descr| field_descr as Arc<dyn FieldDescr>)
        .collect();
    let mut sd = SimpleSizeDescr::with_vtable(index, size, type_id, vtable);
    sd.set_cache_key(cache_key);
    sd.set_gc_managed(is_gc_managed);
    sd.set_headerless(headerless);
    let mut sd = sd.with_all_fielddescrs(all_fielddescrs);
    for fd in extra_gc_fielddescrs {
        sd = sd.with_extra_gc_fielddescr(fd.clone());
    }
    let size_descr = Arc::new(sd);
    let size_ref = size_descr.clone() as DescrRef;
    gc.register_keyed_size(struct_key.clone(), size_ref);
    let parent = gc
        ._cache_size
        .get(&struct_key)
        .cloned()
        .unwrap_or_else(|| size_descr.clone() as DescrRef);
    // descr.py:238 — each shared field reports the current cached SizeDescr.
    for fd in &field_descrs {
        fd.set_parent_descr(&parent);
    }
    SimpleDescrGroup {
        size_descr,
        field_descrs,
    }
}

/// Inner factory shared between [`make_simple_descr_group`] (no
/// cache key) and [`make_simple_descr_group_keyed`] (registers into
/// the keyed cache map after construction).  `cache_key == 0` is the
/// no-identity sentinel for the non-keyed path.
#[expect(
    clippy::too_many_arguments,
    reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
)]
fn make_simple_descr_group_inner(
    index: u32,
    size: usize,
    type_id: u32,
    cache_key: u64,
    vtable: usize,
    is_gc_managed: bool,
    headerless: bool,
    field_specs: &[SimpleFieldDescrSpec],
) -> SimpleDescrGroup {
    census_spec_positions(field_specs);
    let field_descrs_cell = std::cell::RefCell::new(Vec::<Arc<SimpleFieldDescr>>::new());
    let field_specs = field_specs.to_vec();
    let size_descr = Arc::new_cyclic(|weak_size: &Weak<SimpleSizeDescr>| {
        let parent_descr: Weak<dyn Descr> = weak_size.clone();
        let field_descrs: Vec<Arc<SimpleFieldDescr>> = field_specs
            .iter()
            .map(|spec| {
                let field_key_start = field_key_start(&spec.name, &spec.field_key);
                Arc::new(SimpleFieldDescr {
                    index: AtomicU32::new(spec.index),
                    descr_index: AtomicI32::new(-1),
                    ei_index: AtomicU32::new(u32::MAX),
                    name: spec.name.clone().into_boxed_str(),
                    field_key_start,
                    offset: spec.offset,
                    field_size: spec.field_size,
                    field_type: spec.field_type,
                    is_immutable: spec.is_immutable,
                    is_quasi_immutable: spec.is_quasi_immutable,
                    flag: spec.flag,
                    virtualizable: spec.virtualizable,
                    index_in_parent: spec.index_in_parent,
                    parent_descr: RwLock::new(Some(parent_descr.clone())),
                    vinfo: None,
                })
            })
            .collect();
        *field_descrs_cell.borrow_mut() = field_descrs.clone();
        let all_fielddescrs: Vec<Arc<dyn FieldDescr>> = field_descrs
            .iter()
            .cloned()
            .map(|field_descr| field_descr as Arc<dyn FieldDescr>)
            .collect();
        let mut sd = SimpleSizeDescr::with_vtable(index, size, type_id, vtable);
        // descr.py:108-118 `get_size_descr` cache-miss path stamps the
        // `LLType::Struct(cache_key)` slot onto the descr before Arc
        // wrap; mint sites that bypass `get_size_descr` proper and call
        // this factory must do the same so the inverse
        // `bh_size_spec_from_descr` reader (`assembler.rs`) round-trips
        // through `_cache_size[LLType::Struct(cache_key)]` instead of
        // landing on a stale slot via `type_id` widening.
        sd.set_cache_key(cache_key);
        sd.set_gc_managed(is_gc_managed);
        sd.set_headerless(headerless);
        sd.with_all_fielddescrs(all_fielddescrs)
    });
    let field_descrs = field_descrs_cell.into_inner();
    SimpleDescrGroup {
        size_descr,
        field_descrs,
    }
}

/// No-cache-key variant of [`make_simple_descr_group_keyed`].  Used
/// by legacy mint sites that don't yet plumb the `path_hash` cache
/// key surrogate.  Publishes into snapshot-order Vecs only; the
/// keyed `_cache_*[key]` map stays empty for these descrs so
/// `gc_cache.get_*_descr` lookups for the same struct will mint
/// fresh duplicates.  Callers that have the cache key should prefer
/// `make_simple_descr_group_keyed`.
pub fn make_simple_descr_group(
    index: u32,
    size: usize,
    type_id: u32,
    vtable: usize,
    field_specs: &[SimpleFieldDescrSpec],
) -> SimpleDescrGroup {
    // No-cache legacy mint sites are JIT-allocated structs, so they get the
    // GC-managed, headered shape.  A caller that knows the STRUCT's actual
    // shape passes it through `make_simple_descr_group_with_flags`.
    make_simple_descr_group_with_flags(index, size, type_id, vtable, true, false, field_specs)
}

/// [`make_simple_descr_group`] with the GC-header shape supplied rather than
/// defaulted.
///
/// `descr.py:105-127 get_size_descr` reads both flags off the STRUCT itself,
/// so they describe the parent, not the cache slot — a caller that mints
/// without a cache key still knows them and must not lose them.  Minting a raw
/// or header-less parent as GC-managed and headered would hand the consumer a
/// `GUARD_GC_TYPE` and a header offset the object does not have.
///
/// The mint stays uncached exactly as in [`make_simple_descr_group`]:
/// `cache_key` is 0, so the keyed `_cache_*[key]` maps are untouched and
/// repeated calls yield distinct groups.
pub fn make_simple_descr_group_with_flags(
    index: u32,
    size: usize,
    type_id: u32,
    vtable: usize,
    is_gc_managed: bool,
    headerless: bool,
    field_specs: &[SimpleFieldDescrSpec],
) -> SimpleDescrGroup {
    let group = make_simple_descr_group_inner(
        index,
        size,
        type_id,
        0,
        vtable,
        is_gc_managed,
        headerless,
        field_specs,
    );
    // descr.py:236-247 `get_size_descr` cache-miss branch — snapshot
    // order only.
    crate::descr_registry::register_size(group.size_descr.clone() as DescrRef);
    for fd in &group.field_descrs {
        crate::descr_registry::register_field(fd.clone() as DescrRef);
    }
    group
}

/// Simple concrete ArrayDescr.
#[derive(Debug)]
pub struct SimpleArrayDescr {
    /// Per-trace codewriter slot id. See `SimpleFieldDescr.index` for
    /// the rationale — atomic so the cache-or-mint
    /// (`gc_cache.get_array_descr`) path can stamp the analyzer's
    /// `idx` (from `descr_indices.array_index`) onto a shared
    /// `Arc<SimpleArrayDescr>` after cache resolves.
    index: AtomicU32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    /// `effectinfo.py:496` `descr.ei_index = sys.maxint`. See SimpleFieldDescr.
    ei_index: AtomicU32,
    base_size: usize,
    item_size: usize,
    /// `descr.py:274 ArrayDescr.tid` — u32 GC type id.  Atomic so the
    /// analyzer's arraydescrof cache-or-mint path can stamp
    /// `path_hash(array_type_id) as u32` onto a shared
    /// `Arc<SimpleArrayDescr>` after cache resolves (mirrors the
    /// `gc.py:544-549 init_array_descr` post-mint tid write, which
    /// pyre lifts to a settable atomic for analyzer-side parity).
    type_id: AtomicU32,
    /// `gc_cache._cache_array[LLType::Array(cache_key)]` keyed identity
    /// surrogate — the full `path_hash(array_type_id)` u64 that the
    /// analyzer-side `arraydescrof_concrete` published into the cache.
    /// 0 means "no identity carrier" (legacy non-keyed callers).  The
    /// inverse `BhDescr::Array.type_id` field reads this so round-trips
    /// through `simple_descr_group_from_bh_size` resolve `LLType::Array(
    /// cache_key)` in the same slot.  Stamped at `get_array_descr`
    /// cache-miss-mint when `key == LLType::Array(k)`.
    cache_key: u64,
    item_type: Type,
    /// descr.py:277,286: ArrayDescr.lendescr — length field descriptor, or None.
    pub lendescr: Option<DescrRef>,
    /// descr.py:278: ArrayDescr.flag — element type classification.
    flag: ArrayFlag,
    /// descr.py:288: ArrayDescr._is_pure
    pub is_pure: bool,
    /// descr.py:281,289: ArrayDescr.concrete_type — '\x00' or 'f' for Float.
    pub concrete_type: char,
    /// descr.py:280: ArrayDescr.all_interiorfielddescrs.
    /// For array-of-structs, contains interior field descriptors.
    /// `OnceLock` keeps the field write-once after construction so the
    /// builder can publish a single `Arc<SimpleArrayDescr>` to the
    /// `SimpleInteriorFieldDescr` constructors AND set the interior list
    /// on the same Arc afterwards — RPython
    /// `descr.py:388 InteriorFieldDescr.__init__` carries the exact
    /// arraydescr object, so `interior.array_descr is final.array_descr`
    /// must hold by Arc identity.
    all_interiorfielddescrs: std::sync::OnceLock<Vec<DescrRef>>,
    /// Whether this array carries a GC header (see
    /// `ArrayDescr::is_gc_managed`).  `false` only for a header-less raw
    /// native pointer-array minted by `add_ptr_array_descr`.
    is_gc_managed: bool,
    /// Allocation-generation bit for `NEW_ARRAY` rewriting; see
    /// [`ArrayDescr::non_moving`].  Atomic so a frontend can stamp it onto
    /// an already-shared descr, the way `type_id` is stamped.
    non_moving: AtomicBool,
}

impl Clone for SimpleArrayDescr {
    fn clone(&self) -> Self {
        let interior = std::sync::OnceLock::new();
        if let Some(existing) = self.all_interiorfielddescrs.get() {
            let _ = interior.set(existing.clone());
        }
        SimpleArrayDescr {
            index: AtomicU32::new(self.index.load(Ordering::Relaxed)),
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            ei_index: AtomicU32::new(self.ei_index.load(Ordering::Relaxed)),
            base_size: self.base_size,
            item_size: self.item_size,
            type_id: AtomicU32::new(self.type_id.load(Ordering::Relaxed)),
            cache_key: self.cache_key,
            item_type: self.item_type,
            lendescr: self.lendescr.clone(),
            flag: self.flag,
            is_pure: self.is_pure,
            concrete_type: self.concrete_type,
            all_interiorfielddescrs: interior,
            is_gc_managed: self.is_gc_managed,
            non_moving: AtomicBool::new(self.non_moving.load(Ordering::Relaxed)),
        }
    }
}

impl SimpleArrayDescr {
    pub fn new(
        index: u32,
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
    ) -> Self {
        let flag = ArrayFlag::from_item_type(item_type, false);
        SimpleArrayDescr {
            index: AtomicU32::new(index),
            descr_index: AtomicI32::new(-1),
            ei_index: AtomicU32::new(u32::MAX),
            base_size,
            item_size,
            type_id: AtomicU32::new(type_id),
            cache_key: 0,
            item_type,
            lendescr: None,
            flag,
            is_pure: false,
            concrete_type: '\x00',
            all_interiorfielddescrs: std::sync::OnceLock::new(),
            is_gc_managed: true,
            non_moving: AtomicBool::new(false),
        }
    }

    /// `ArrayDescr` with explicit flag (for struct arrays).
    pub fn with_flag(
        index: u32,
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
        flag: ArrayFlag,
    ) -> Self {
        SimpleArrayDescr {
            index: AtomicU32::new(index),
            descr_index: AtomicI32::new(-1),
            ei_index: AtomicU32::new(u32::MAX),
            base_size,
            item_size,
            type_id: AtomicU32::new(type_id),
            cache_key: 0,
            item_type,
            lendescr: None,
            flag,
            is_pure: false,
            concrete_type: '\x00',
            all_interiorfielddescrs: std::sync::OnceLock::new(),
            is_gc_managed: true,
            non_moving: AtomicBool::new(false),
        }
    }

    /// Override the allocation-generation flag (default `false`); see
    /// [`ArrayDescr::non_moving`].  Takes `&self` so a frontend can stamp
    /// it after the descr is already behind an `Arc` and registered.
    pub fn set_non_moving(&self, non_moving: bool) {
        self.non_moving.store(non_moving, Ordering::Relaxed);
    }

    /// Set the GC-managed flag (`false` for a header-less raw native
    /// pointer-array).  Called on the freshly-built descr by the
    /// `BhDescr::Array` reconstruction / `add_ptr_array_descr` path
    /// before Arc wrapping.
    pub fn set_is_gc_managed(&mut self, is_gc_managed: bool) {
        self.is_gc_managed = is_gc_managed;
    }

    /// Stamp the `gc_cache._cache_array[LLType::Array(...)]` identity
    /// surrogate onto this descr.  Called by `gc_cache.get_array_descr`
    /// cache-miss path before Arc wrap.
    pub fn set_cache_key(&mut self, key: u64) {
        self.cache_key = key;
    }

    /// RPython: arraydescr.all_interiorfielddescrs = descrs
    /// Settable through `&self` so the publication can target the same
    /// `Arc<SimpleArrayDescr>` the `InteriorFieldDescr` constructors
    /// already cloned into their `array_descr` field.  Subsequent calls
    /// silently ignore the new list (matching RPython's "set once" use
    /// at `descr.py:373` inside `get_array_descr`).
    pub fn set_all_interiorfielddescrs(&self, descrs: Vec<DescrRef>) {
        let _ = self.all_interiorfielddescrs.set(descrs);
    }

    /// gc.py:548: descr.tid = llop.combine_ushort(lltype.Signed, type_id, 0)
    /// Called by init_array_descr hook before Arc wrapping.
    pub fn set_type_id(&self, type_id: u32) {
        self.type_id.store(type_id, Ordering::Relaxed);
    }
}

impl Descr for SimpleArrayDescr {
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn index(&self) -> u32 {
        self.index.load(Ordering::Relaxed)
    }
    fn set_index(&self, index: u32) {
        self.index.store(index, Ordering::Relaxed);
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }
    fn set_ei_index(&self, ei_index: u32) {
        self.ei_index.store(ei_index, Ordering::Relaxed);
    }
    fn as_array_descr(&self) -> Option<&dyn ArrayDescr> {
        Some(self)
    }
    /// descr.py:295: ArrayDescr.is_always_pure()
    fn is_always_pure(&self) -> bool {
        self.is_pure
    }
}

impl ArrayDescr for SimpleArrayDescr {
    fn is_gc_managed(&self) -> bool {
        self.is_gc_managed
    }
    fn non_moving(&self) -> bool {
        self.non_moving.load(Ordering::Relaxed)
    }
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id.load(Ordering::Relaxed)
    }
    fn set_type_id(&self, id: u32) {
        self.type_id.store(id, Ordering::Relaxed);
    }
    fn cache_key(&self) -> u64 {
        self.cache_key
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.flag == ArrayFlag::Signed
    }
    /// RPython: descr.py ArrayDescr.is_array_of_pointers()
    fn is_array_of_pointers(&self) -> bool {
        self.flag == ArrayFlag::Pointer
    }
    /// RPython: descr.py ArrayDescr.is_array_of_floats()
    fn is_array_of_floats(&self) -> bool {
        self.flag == ArrayFlag::Float
    }
    /// RPython: descr.py ArrayDescr.is_array_of_structs()
    fn is_array_of_structs(&self) -> bool {
        self.flag == ArrayFlag::Struct
    }
    /// RPython: descr.py ArrayDescr.is_array_of_primitives()
    fn is_array_of_primitives(&self) -> bool {
        matches!(
            self.flag,
            ArrayFlag::Float | ArrayFlag::Signed | ArrayFlag::Unsigned
        )
    }
    /// descr.py:277,286: ArrayDescr.lendescr
    fn len_descr(&self) -> Option<&dyn FieldDescr> {
        self.lendescr.as_ref().and_then(|d| d.as_field_descr())
    }
    /// RPython: descr.py ArrayDescr.get_all_interiorfielddescrs()
    fn get_all_interiorfielddescrs(&self) -> Option<&[DescrRef]> {
        self.all_interiorfielddescrs.get().map(Vec::as_slice)
    }
    /// `descr.py:373 arraydescr.all_interiorfielddescrs = descrs` —
    /// post-construction publish (set-once via `OnceLock`).
    fn set_all_interiorfielddescrs(&self, descrs: Vec<DescrRef>) {
        let _ = self.all_interiorfielddescrs.set(descrs);
    }
}

/// Simple concrete InteriorFieldDescr.
#[derive(Debug)]
pub struct SimpleInteriorFieldDescr {
    /// per-trace analyzer slot id. Stored atomic so the analyzer's
    /// `cc.interiorfielddescrof` cache-or-mint can stamp on a
    /// `Arc<SimpleInteriorFieldDescr>` returned from
    /// `gc_cache._cache_interiorfield` without cloning the Arc, mirroring
    /// the `SimpleFieldDescr.index` / `SimpleArrayDescr.index` stamp
    /// hooks added for `cpu.fielddescrof` / `cpu.arraydescrof` parity.
    index: AtomicU32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    /// `effectinfo.py:496` `descr.ei_index = sys.maxint`. See SimpleFieldDescr.
    ei_index: AtomicU32,
    /// `descr.py:388 InteriorFieldDescr.__init__` carries the
    /// containing `ArrayDescr` object. PyPy duck-types this — any
    /// `ArrayDescr` instance suffices.  Pyre stores `Arc<dyn ArrayDescr>`
    /// over the sole concrete `Arc<SimpleArrayDescr>` (analyzer and
    /// runtime mint), downcasted via `try_downcast_arc` at the analyzer
    /// wrap site.
    array_descr: std::sync::Arc<dyn ArrayDescr>,
    /// `descr.py:388 InteriorFieldDescr.__init__` field descr —
    /// concrete `FieldDescr` in PyPy.
    field_descr: std::sync::Arc<dyn FieldDescr>,
    /// Pyre-side parent SizeDescr backreference; `descr.py` has no
    /// equivalent (PyPy's InteriorFieldDescr derives parent from
    /// `arraydescr`).  Kept for pyre's
    /// `ensure_ptr_info_arg0`-style dispatch paths.
    owner_size_descr: Option<std::sync::Arc<dyn SizeDescr>>,
}

impl Clone for SimpleInteriorFieldDescr {
    fn clone(&self) -> Self {
        SimpleInteriorFieldDescr {
            index: AtomicU32::new(self.index.load(Ordering::Relaxed)),
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            ei_index: AtomicU32::new(self.ei_index.load(Ordering::Relaxed)),
            array_descr: self.array_descr.clone(),
            field_descr: self.field_descr.clone(),
            owner_size_descr: self.owner_size_descr.clone(),
        }
    }
}

impl SimpleInteriorFieldDescr {
    pub fn new(
        index: u32,
        array_descr: std::sync::Arc<dyn ArrayDescr>,
        field_descr: std::sync::Arc<dyn FieldDescr>,
    ) -> Self {
        SimpleInteriorFieldDescr {
            index: AtomicU32::new(index),
            descr_index: AtomicI32::new(-1),
            ei_index: AtomicU32::new(u32::MAX),
            array_descr,
            field_descr,
            owner_size_descr: None,
        }
    }

    pub fn new_with_owner(
        index: u32,
        array_descr: std::sync::Arc<dyn ArrayDescr>,
        field_descr: std::sync::Arc<dyn FieldDescr>,
        owner_size_descr: std::sync::Arc<dyn SizeDescr>,
    ) -> Self {
        SimpleInteriorFieldDescr {
            index: AtomicU32::new(index),
            descr_index: AtomicI32::new(-1),
            ei_index: AtomicU32::new(u32::MAX),
            array_descr,
            field_descr,
            owner_size_descr: Some(owner_size_descr),
        }
    }
}

impl Descr for SimpleInteriorFieldDescr {
    fn index(&self) -> u32 {
        self.index.load(Ordering::Relaxed)
    }
    fn set_index(&self, index: u32) {
        self.index.store(index, Ordering::Relaxed);
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn get_ei_index(&self) -> u32 {
        self.ei_index.load(Ordering::Relaxed)
    }
    fn set_ei_index(&self, ei_index: u32) {
        self.ei_index.store(ei_index, Ordering::Relaxed);
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn as_interior_field_descr(&self) -> Option<&dyn InteriorFieldDescr> {
        Some(self)
    }
}

impl InteriorFieldDescr for SimpleInteriorFieldDescr {
    fn array_descr(&self) -> &dyn ArrayDescr {
        self.array_descr.as_ref()
    }
    fn field_descr(&self) -> &dyn FieldDescr {
        self.field_descr.as_ref()
    }
}

/// Simple concrete CallDescr for non-test use.
/// descr.py:450-493: CallDescr(arg_classes, result_type, result_signed,
///   result_size, extrainfo, ffi_flags).
///
/// `effect` is wrapped in [`EffectInfoCell`] so
/// `effectinfo::compute_bitstrings` can install the compacted
/// bitstrings post-construction. See `Descr::set_effect_bitstrings`.
#[derive(Debug)]
pub struct SimpleCallDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_class: char,
    result_size: usize,
    /// descr.py:453: CallDescr.result_flag — computed from result_type +
    /// result_signed in __init__ (descr.py:478-493).
    result_flag: ArrayFlag,
    effect: Arc<crate::effectinfo::EffectInfoCell>,
}

impl Clone for SimpleCallDescr {
    fn clone(&self) -> Self {
        SimpleCallDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            arg_types: self.arg_types.clone(),
            result_type: self.result_type,
            result_class: self.result_class,
            result_size: self.result_size,
            result_flag: self.result_flag,
            effect: self.effect.clone(),
        }
    }
}

impl SimpleCallDescr {
    /// descr.py:456-493: CallDescr(arg_classes, result_type, result_signed,
    ///   result_size, extrainfo, ffi_flags).
    /// `result_signed` is used to compute `result_flag`.
    pub fn new(
        index: u32,
        arg_types: Vec<Type>,
        result_type: Type,
        result_signed: bool,
        result_size: usize,
        effect: EffectInfo,
    ) -> Self {
        Self::new_with_result_class(
            index,
            arg_types,
            result_type,
            result_class_of(result_type),
            result_signed,
            result_size,
            effect,
        )
    }

    pub fn new_with_result_class(
        index: u32,
        arg_types: Vec<Type>,
        result_type: Type,
        result_class: char,
        result_signed: bool,
        result_size: usize,
        effect: EffectInfo,
    ) -> Self {
        Self::new_with_effect_cell_and_result_class(
            index,
            arg_types,
            result_type,
            result_class,
            result_signed,
            result_size,
            crate::effectinfo::intern_effect_info(effect),
        )
    }

    fn new_with_effect_cell(
        index: u32,
        arg_types: Vec<Type>,
        result_type: Type,
        result_signed: bool,
        result_size: usize,
        effect: Arc<crate::effectinfo::EffectInfoCell>,
    ) -> Self {
        Self::new_with_effect_cell_and_result_class(
            index,
            arg_types,
            result_type,
            result_class_of(result_type),
            result_signed,
            result_size,
            effect,
        )
    }

    fn new_with_effect_cell_and_result_class(
        index: u32,
        arg_types: Vec<Type>,
        result_type: Type,
        result_class: char,
        result_signed: bool,
        result_size: usize,
        effect: Arc<crate::effectinfo::EffectInfoCell>,
    ) -> Self {
        let result_flag = match result_type {
            Type::Void => ArrayFlag::Void,
            Type::Int => {
                if result_signed {
                    ArrayFlag::Signed
                } else {
                    ArrayFlag::Unsigned
                }
            }
            Type::Ref => ArrayFlag::Pointer,
            Type::Float => ArrayFlag::Float,
        };
        SimpleCallDescr {
            index,
            descr_index: AtomicI32::new(-1),
            arg_types,
            result_type,
            result_class,
            result_size,
            result_flag,
            effect,
        }
    }
}

impl Descr for SimpleCallDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
    /// `effectinfo.py:537-538 setattr(ei, 'bitstring_*', …)`. Pyre's
    /// `compute_bitstrings` writeback path reaches every call descr
    /// uniformly via this trait method; without this override pyre
    /// silently mixes index domains (descr.ei_index in the compact
    /// domain, EI bitstring still raw `descr.index()`).
    fn set_effect_bitstrings(
        &self,
        readonly_descrs_fields: Option<Vec<u8>>,
        write_descrs_fields: Option<Vec<u8>>,
        readonly_descrs_arrays: Option<Vec<u8>>,
        write_descrs_arrays: Option<Vec<u8>>,
        readonly_descrs_interiorfields: Option<Vec<u8>>,
        write_descrs_interiorfields: Option<Vec<u8>>,
    ) {
        self.effect.set_bitstrings(
            readonly_descrs_fields,
            write_descrs_fields,
            readonly_descrs_arrays,
            write_descrs_arrays,
            readonly_descrs_interiorfields,
            write_descrs_interiorfields,
        );
    }
}

impl CallDescr for SimpleCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_class(&self) -> char {
        self.result_class
    }
    /// descr.py:537-538: is_result_signed() → result_flag == FLAG_SIGNED
    fn is_result_signed(&self) -> bool {
        self.result_flag == ArrayFlag::Signed
    }
    fn result_size(&self) -> usize {
        self.result_size
    }
    fn get_extra_info(&self) -> &EffectInfo {
        self.effect.get()
    }
}

unsafe extern "C" {
    // llsupport/memcpy.py:3-5 — the host `memcpy` symbol that
    // `gc.py:39 self.memcpy_fn = memcpy_fn` binds via `rffi.llexternal`.
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}

/// llsupport/gc.py:39 `GcLLDescr_framework.memcpy_fn` cast to a Signed
/// via `cast_ptr_to_adr` + `cast_adr_to_int` (rewrite.py:1046-1047).
///
/// Returns the same address on every call so the lowered
/// `CALL_N(memcpy_fn, …)` emitted by `rewrite_copy_str_content` carries
/// a stable ConstInt argument across rewrites — matching upstream where
/// the address is read once into `self.memcpy_fn` per CPU.
pub fn memcpy_fn_addr() -> i64 {
    memcpy as *const () as i64
}

/// llsupport/gc.py:40-43 `GcLLDescr_framework.memcpy_descr`.
///
/// CallDescr used by `rewrite_copy_str_content` for the lowered
/// CALL_N(memcpy_fn, dst, src, n) emitted in place of
/// COPYSTRCONTENT / COPYUNICODECONTENT.  Upstream:
///
/// ```text
/// self.memcpy_descr = get_call_descr(self,
///     [lltype.Signed, lltype.Signed, lltype.Signed], lltype.Void,
///     EffectInfo([], [], [], [], [], [], EffectInfo.EF_CANNOT_RAISE,
///         can_collect=False))
/// ```
///
/// gc.py:40-43 builds a single instance per `GcLLDescription`; pyre keeps
/// the same identity invariant by returning a `OnceLock`-cached `Arc`,
/// so every caller (`GcRewriterImpl::new` per-backend, optimizer
/// virtualstate import, …) sees the same `DescrRef`.
///
/// llsupport/gc.py:33-37 `self.fielddescr_vtable = get_field_descr(
/// self, rclass.OBJECT, 'typeptr')`. FieldDescr describing the
/// `typeptr` slot at the head of every `rclass.OBJECT` (offset 0,
/// `Signed` size). Consumed by `rewrite.py:482-484` to stamp the
/// vtable onto a freshly-allocated NEW_WITH_VTABLE result. None when
/// the translator was built with `gcremovetypeptr=True`; pyre's
/// production backends always emit a typeptr slot, so they install
/// a Some value.
///
/// Cached as a process-wide singleton via `OnceLock` to mirror the
/// "single instance per CPU" semantic; identity stability matters
/// when downstream optimizer caches key on `descr_identity`.
pub fn make_vtable_field_descr() -> DescrRef {
    static VTABLE_DESCR_GROUP: OnceLock<SimpleDescrGroup> = OnceLock::new();
    VTABLE_DESCR_GROUP
        .get_or_init(|| {
            let field_descr_cell = std::cell::RefCell::new(None);
            let size_descr = Arc::new_cyclic(|weak_size: &Weak<SimpleSizeDescr>| {
                let parent_descr: Weak<dyn Descr> = weak_size.clone();
                // rclass.py / objectmodel.py: typeptr lives at object head
                // with `Signed`-sized vtable pointer; is_immutable=true
                // because the class identity does not change after malloc.
                let field_descr = Arc::new(SimpleFieldDescr {
                    index: AtomicU32::new(0x6000_0000),
                    descr_index: AtomicI32::new(-1),
                    ei_index: AtomicU32::new(u32::MAX),
                    name: Box::from("object.typeptr"),
                    field_key_start: field_key_start("object.typeptr", "typeptr"),
                    offset: 0,
                    field_size: std::mem::size_of::<usize>(),
                    field_type: crate::Type::Int,
                    is_immutable: true,
                    is_quasi_immutable: false,
                    flag: ArrayFlag::Signed,
                    virtualizable: false,
                    index_in_parent: 0,
                    parent_descr: RwLock::new(Some(parent_descr)),
                    vinfo: None,
                });
                *field_descr_cell.borrow_mut() = Some(field_descr.clone());
                let all_fielddescrs = vec![field_descr as Arc<dyn FieldDescr>];
                // descr.py:234-238: OBJECT gets a nullptr vtable, so
                // its SizeDescr is not an object descriptor.
                SimpleSizeDescr::new(u32::MAX, std::mem::size_of::<usize>(), 0)
                    .with_all_fielddescrs(all_fielddescrs)
            });
            SimpleDescrGroup {
                size_descr,
                field_descrs: vec![
                    field_descr_cell
                        .into_inner()
                        .expect("Arc::new_cyclic must initialize the typeptr field descriptor"),
                ],
            }
        })
        .field_descrs[0]
        .clone()
}

/// llsupport/gc.py:394 `self.fielddescr_tid = get_field_descr(self,
/// self.GCClass.HDR, 'tid')`. FieldDescr describing the `tid` slot
/// inside the GC header.  Consumed by `rewrite.py:914-918`
/// `gen_initialize_tid` to stamp the type id onto every freshly-allocated
/// framework-GC object's header.  None on Boehm builds (gc.py:157), where
/// `gen_initialize_tid` is a no-op.
///
/// pyre's `GcHeader` is a single u64 word (`tid_and_flags`) split into a
/// lower 32-bit type id and an upper 32-bit flags half (header.rs
/// FLAG_SHIFT = 32).  The descr addresses the *type id* slot only —
/// `offset = 0`, `field_size = 4 bytes`.  Restricting the store width
/// to four bytes is what lets `gen_initialize_tid` overwrite the type
/// id without disturbing flag bits the runtime may already have set on
/// the same word: collector.rs:449 `alloc_in_oldgen` ORs in
/// `TRACK_YOUNG_PTRS` for any object the malloc-nursery slow path
/// promotes to the old gen, and a full-word store would silently wipe
/// it.  Upstream rewrite.py:914-918 has no analogue because
/// `incminimark.HDR.tid` is a single Signed field where `tid` and
/// flags coexist in the same value, and the rewriter never re-stamps
/// tid after a slow malloc — pyre's split layout requires a narrower
/// store at this site instead.  The header sits *before* the object
/// pointer; `gen_initialize_tid` translates the descr's offset by
/// `-HDR_SIZE` to point at the header word.
///
/// Cached as a process-wide singleton via `OnceLock` to mirror gc.py's
/// "single instance per CPU" semantic.
pub fn make_tid_field_descr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static TID_FIELD_DESCR: OnceLock<DescrRef> = OnceLock::new();
    TID_FIELD_DESCR
        .get_or_init(|| {
            // header.rs `GcHeader.tid_and_flags: u64` is split:
            // bits  0..32 — type id (this descr).
            // bits 32..64 — gc flags (TRACK_YOUNG_PTRS / VISITED / PINNED /
            //               HAS_CARDS …).  Owned by the GC, not the JIT.
            // is_immutable=false: incminimark mutates flag bits on mark
            // and rewrites the whole word on forwarding.
            Arc::new(SimpleFieldDescr::new(
                0x7000_0000,
                0,                          // offset within HDR
                std::mem::size_of::<u32>(), // field_size = 4 bytes (lower 32 bits = type id)
                crate::Type::Int,
                false, // is_immutable — flags / forwarding marker mutate
            ))
        })
        .clone()
}

pub fn make_memcpy_calldescr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static MEMCPY_DESCR: OnceLock<DescrRef> = OnceLock::new();
    MEMCPY_DESCR
        .get_or_init(|| {
            let effect = EffectInfo {
                // EF_CANNOT_RAISE — memcpy is leaf; does not raise.
                extraeffect: ExtraEffect::CannotRaise,
                // can_collect=False — regalloc can skip saving GC-ref regs across
                // the call (rewrite.rs `SAVE_DEFAULT_REGS` path).
                can_collect: false,
                ..EffectInfo::default()
            };
            // memcpy returns void.  `result_size=0` / `result_signed=false`
            // are the SimpleCallDescr defaults for a Void return.
            Arc::new(SimpleCallDescr::new(
                0x5000_0000,
                vec![crate::Type::Int, crate::Type::Int, crate::Type::Int],
                crate::Type::Void,
                false,
                0,
                effect,
            ))
        })
        .clone()
}

/// gc.py:45 + gc.py:420-431 generate_function('malloc_array', ...).
/// CallDescr for CALL_R(malloc_array_fn, itemsize, type_id, num_elem) -> Ref.
pub fn make_malloc_array_calldescr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static MALLOC_ARRAY_DESCR: OnceLock<DescrRef> = OnceLock::new();
    MALLOC_ARRAY_DESCR
        .get_or_init(|| {
            Arc::new(SimpleCallDescr::new(
                0x5000_0001,
                vec![crate::Type::Int, crate::Type::Int, crate::Type::Int],
                crate::Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

/// gc.py:45 + gc.py:432-444
/// generate_function('malloc_array_nonstandard', ...).
/// CallDescr for CALL_R(malloc_array_nonstandard_fn,
///                     basesize, itemsize, lengthofs, type_id, num_elem) -> Ref.
pub fn make_malloc_array_nonstandard_calldescr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static MALLOC_ARRAY_NONSTANDARD_DESCR: OnceLock<DescrRef> = OnceLock::new();
    MALLOC_ARRAY_NONSTANDARD_DESCR
        .get_or_init(|| {
            Arc::new(SimpleCallDescr::new(
                0x5000_0002,
                vec![
                    crate::Type::Int,
                    crate::Type::Int,
                    crate::Type::Int,
                    crate::Type::Int,
                    crate::Type::Int,
                ],
                crate::Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

/// gc.py:45 + gc.py:460-467 generate_function('malloc_str', ...).
/// CallDescr for CALL_R(malloc_str_fn, type_id, length) -> Ref.
///
/// TODO: upstream `malloc_str` is generated as
/// `[lltype.Signed]` (length only) and captures `str_type_id` via Python
/// closure scope (gc.py:451 `str_type_id = self.str_descr.tid`).  Rust
/// `extern "C" fn` cannot lexically capture, so the type id is threaded
/// through the call as an explicit Signed arg — same pattern pyre
/// already uses for `malloc_array_nonstandard` (rewrite.py:825-832 +
/// `make_malloc_array_nonstandard_calldescr` below).  The descr's first
/// param is the type id, the second is the requested length.
pub fn make_malloc_str_calldescr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static MALLOC_STR_DESCR: OnceLock<DescrRef> = OnceLock::new();
    MALLOC_STR_DESCR
        .get_or_init(|| {
            Arc::new(SimpleCallDescr::new(
                0x5000_0003,
                vec![crate::Type::Int, crate::Type::Int],
                crate::Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

/// gc.py:45 + gc.py:469-476 generate_function('malloc_unicode', ...).
/// CallDescr for CALL_R(malloc_unicode_fn, type_id, length) -> Ref.
///
/// TODO: see `make_malloc_str_calldescr` — the type
/// id is threaded as an explicit arg because `extern "C" fn` cannot
/// lexically capture `unicode_type_id` the way upstream's
/// `malloc_unicode` closure does (gc.py:455 `unicode_type_id =
/// self.unicode_descr.tid`).
pub fn make_malloc_unicode_calldescr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static MALLOC_UNICODE_DESCR: OnceLock<DescrRef> = OnceLock::new();
    MALLOC_UNICODE_DESCR
        .get_or_init(|| {
            Arc::new(SimpleCallDescr::new(
                0x5000_0004,
                vec![crate::Type::Int, crate::Type::Int],
                crate::Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

/// gc.py:45 + gc.py:481-490 generate_function('malloc_big_fixedsize', ...).
/// CallDescr for CALL_R(malloc_big_fixedsize_fn, size, type_id) -> Ref.
///
/// rewrite.py:778-796 `gen_malloc_fixedsize` framework-GC arm.  The
/// helper allocates a fixed-size object directly in the old gen
/// because the requested size is too large for the nursery
/// (gc.py:478-490 `malloc_big_fixedsize` "Never called as far as I can
/// tell, but there for completeness: allocate a fixed-size object,
/// but not in the nursery, because it is too big.").
pub fn make_malloc_big_fixedsize_calldescr() -> DescrRef {
    use std::sync::{Arc, OnceLock};
    static MALLOC_BIG_FIXEDSIZE_DESCR: OnceLock<DescrRef> = OnceLock::new();
    MALLOC_BIG_FIXEDSIZE_DESCR
        .get_or_init(|| {
            Arc::new(SimpleCallDescr::new(
                0x5000_0005,
                vec![crate::Type::Int, crate::Type::Int],
                crate::Type::Ref,
                false,
                std::mem::size_of::<usize>(),
                EffectInfo::MOST_GENERAL,
            ))
        })
        .clone()
}

/// Simple concrete FailDescr for guard failure descriptors.
#[derive(Debug)]
pub struct SimpleFailDescr {
    index: u32,
    fail_index: u32,
    fail_arg_types: Vec<Type>,
    is_finish: bool,
    trace_id: u64,
    /// schedule.py:654: vector accumulation info attached during vectorization.
    vector_info: std::cell::UnsafeCell<Option<Box<AccumInfo>>>,
}

impl Clone for SimpleFailDescr {
    fn clone(&self) -> Self {
        SimpleFailDescr {
            index: self.index,
            fail_index: self.fail_index,
            fail_arg_types: self.fail_arg_types.clone(),
            is_finish: self.is_finish,
            trace_id: self.trace_id,
            vector_info: std::cell::UnsafeCell::new(unsafe { (&*self.vector_info.get()).clone() }),
        }
    }
}

// Safety: JIT is single-threaded (RPython GIL equivalent). UnsafeCell
// replaces Mutex for rd_vector_info — no concurrent access.
unsafe impl Send for SimpleFailDescr {}
unsafe impl Sync for SimpleFailDescr {}

impl SimpleFailDescr {
    pub fn new(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: false,
            trace_id: 0,
            vector_info: std::cell::UnsafeCell::new(None),
        }
    }

    pub fn finish(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: true,
            trace_id: 0,
            vector_info: std::cell::UnsafeCell::new(None),
        }
    }

    pub fn with_trace_id(mut self, trace_id: u64) -> Self {
        self.trace_id = trace_id;
        self
    }
}

impl Descr for SimpleFailDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for SimpleFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
    fn is_finish(&self) -> bool {
        self.is_finish
    }
    fn trace_id(&self) -> u64 {
        self.trace_id
    }
    fn attach_vector_info(&self, info: AccumInfo) {
        push_vector_info(unsafe { &mut *self.vector_info.get() }, info);
    }
    fn vector_info(&self) -> Vec<AccumInfo> {
        flatten_vector_info(unsafe { (&*self.vector_info.get()).as_deref() })
    }
}

#[cfg(test)]
mod register_keyed_size_authority_tests {
    use super::*;

    #[test]
    fn unresolved_synthetic_struct_layout_is_registered_and_stamped() {
        let mut gc = GcCache::new();
        let cache_key = 0x1234_5678_9abc_def0;
        let fields: Vec<Arc<dyn FieldDescr>> = [0usize, 8]
            .into_iter()
            .enumerate()
            .map(|(index, offset)| {
                Arc::new(SimpleFieldDescr::new_with_name(
                    index as u32,
                    offset,
                    8,
                    Type::Ref,
                    false,
                    ArrayFlag::Pointer,
                    format!("ref{index}"),
                    format!("ref{index}"),
                )) as Arc<dyn FieldDescr>
            })
            .collect();
        let mut sd = SimpleSizeDescr::new(u32::MAX, 16, cache_key as u32);
        sd.set_cache_key(cache_key);
        let descr: DescrRef = Arc::new(sd.with_all_fielddescrs(fields));
        gc.register_keyed_size(LLType::Struct(cache_key), descr.clone());

        let mut observed = Vec::new();
        let count = gc.register_unresolved_struct_tids(|size, offsets| {
            observed.push((size, offsets));
            42
        });

        assert_eq!(count, 1);
        assert_eq!(observed, vec![(16, vec![0, 8])]);
        assert_eq!(descr.as_size_descr().unwrap().type_id(), 42);
        assert_eq!(gc.resolve_struct_tid(cache_key), Some(42));
    }

    fn size_descr_at(type_id: u32, vtable: usize, offsets: &[usize]) -> DescrRef {
        let mut sd = SimpleSizeDescr::with_vtable(u32::MAX, 32, type_id, vtable);
        let fields: Vec<Arc<dyn FieldDescr>> = offsets
            .iter()
            .enumerate()
            .map(|(i, &offset)| {
                Arc::new(SimpleFieldDescr::new_with_name(
                    i as u32,
                    offset,
                    8,
                    Type::Int,
                    false,
                    ArrayFlag::Signed,
                    format!("f{offset}"),
                    format!("f{offset}"),
                )) as Arc<dyn FieldDescr>
            })
            .collect();
        sd = sd.with_all_fielddescrs(fields);
        Arc::new(sd) as DescrRef
    }

    fn size_descr(type_id: u32, vtable: usize, nfields: usize) -> DescrRef {
        let offsets: Vec<usize> = (0..nfields).map(|i| i * 8).collect();
        size_descr_at(type_id, vtable, &offsets)
    }

    /// The analyzer-side producer reports MORE fields and NO vtable, and the
    /// extra entries are the two header words at offsets 0 and 8 — exactly
    /// what `heaptracker.py:64-67 all_fielddescrs` skips. Field count alone
    /// would let that list evict the runtime publish, which holds both the
    /// header-free positional list upstream builds and the only copy of the
    /// vtable `new_with_vtable`'s typeptr store needs.
    #[test]
    fn a_vtable_less_header_bearing_producer_never_displaces_the_runtime_publish() {
        let mut gc = GcCache::new();
        let key = LLType::Struct(0x8463ec159694d229);
        gc.register_keyed_size(key.clone(), size_descr_at(7, 0x1000, &[16, 24, 32]));
        gc.register_keyed_size(
            key.clone(),
            size_descr_at(0x9694d229, 0, &[0, 8, 16, 24, 32]),
        );
        let cached = gc._cache_size.get(&key).unwrap().as_size_descr().unwrap();
        assert_eq!(cached.type_id(), 7, "the real gc tid must survive");
        assert_ne!(cached.vtable(), 0, "the vtable must survive");
        assert_eq!(
            cached
                .all_fielddescrs()
                .iter()
                .map(|f| f.offset())
                .collect::<Vec<_>>(),
            vec![16, 24, 32],
            "the header words must not enter the positional list",
        );
    }

    /// The mirror order. The 18 module-scope groups publish before the
    /// build-time specs, but the per-exception-class groups do not — they
    /// arrive after a `vtable: 0` spec is already cached, and a one-sided rule
    /// would reject their vtable, losing it exactly the way the other order
    /// does.
    #[test]
    fn a_vtable_bearing_publish_still_wins_when_it_arrives_second() {
        let mut gc = GcCache::new();
        let key = LLType::Struct(0x8463ec159694d229);
        gc.register_keyed_size(
            key.clone(),
            size_descr_at(0x9694d229, 0, &[0, 8, 16, 24, 32]),
        );
        gc.register_keyed_size(key.clone(), size_descr_at(7, 0x1000, &[16, 24, 32]));
        let cached = gc._cache_size.get(&key).unwrap().as_size_descr().unwrap();
        assert_eq!(cached.type_id(), 7);
        assert_ne!(cached.vtable(), 0, "the vtable must win from either side");
    }

    /// The two orders above must agree; upstream has one owner per `STRUCT`
    /// and never arbitrates, so an order-dependent answer is a defect by
    /// itself.
    #[test]
    fn the_surviving_descr_does_not_depend_on_registration_order() {
        let runtime = || size_descr_at(7, 0x1000, &[16, 24, 32]);
        let analyzer = || size_descr_at(0x9694d229, 0, &[0, 8, 16, 24, 32]);
        let survivor = |first: DescrRef, second: DescrRef| {
            let mut gc = GcCache::new();
            let key = LLType::Struct(3);
            gc.register_keyed_size(key.clone(), first);
            gc.register_keyed_size(key.clone(), second);
            let sd = gc._cache_size.get(&key).unwrap().as_size_descr().unwrap();
            (sd.type_id(), sd.vtable(), sd.all_fielddescrs().len())
        };
        assert_eq!(
            survivor(runtime(), analyzer()),
            survivor(analyzer(), runtime())
        );
    }

    /// A published parent that lists nothing — what a mint installs when it
    /// calls `get_size_descr(key, size, 0, false)`.
    fn shell_descr(size: usize) -> DescrRef {
        Arc::new(SimpleSizeDescr::new(u32::MAX, size, 0)) as DescrRef
    }

    /// `_cache_field[key]`, spelled the way a mint site spells it: the
    /// `T<type_id>.` stand-in `get_field_descr` falls back to when the caller
    /// carries only the numeric struct identity.
    fn shell_fields(slots: &[(&str, usize)]) -> HashMap<String, Arc<SimpleFieldDescr>> {
        slots
            .iter()
            .enumerate()
            .map(|(i, (name, offset))| {
                let fd = SimpleFieldDescr::new_with_name(
                    i as u32,
                    *offset,
                    8,
                    Type::Int,
                    false,
                    ArrayFlag::Signed,
                    format!("T99.{name}"),
                    (*name).to_string(),
                );
                ((*name).to_string(), Arc::new(fd))
            })
            .collect()
    }

    /// `size_shell_census` must match an aliased twin by SHAPE, never by name.
    ///
    /// The shells come from mint sites that carry only the numeric struct
    /// identity, so their fields get the `T<type_id>.` stand-in spelling. An
    /// owner-name comparison against the real `Owner.field` spellings can only
    /// ever report zero, and a vacuous zero reads exactly like "no aliasing" —
    /// which is how the first version of this census got the answer backwards.
    #[test]
    fn size_shell_census_matches_an_aliased_twin_by_shape_not_by_name() {
        let mut gc = GcCache::new();
        // One key that did get a layout, under real `Owner.field` names.
        gc._cache_size
            .insert(LLType::Struct(1), size_descr_at(7, 0, &[16, 24, 32]));
        // A second key for the same struct: fields known, never listed.
        let shell = LLType::Struct(2);
        gc._cache_size.insert(shell.clone(), shell_descr(32));
        gc._cache_field
            .insert(shell, shell_fields(&[("a", 16), ("b", 24)]));

        assert_eq!(
            gc.size_shell_census(),
            [2, 1, 1, 1, 1],
            "the twin owns both cached slots at the same size, and two slots \
             is more than single-slot evidence",
        );
    }

    /// A shell whose slots no populated list owns is a producer gap, not an
    /// aliasing artifact — the layout was never published anywhere. Conflating
    /// the two would point the fix at the wrong task.
    #[test]
    fn size_shell_census_claims_no_twin_for_a_layout_nobody_published() {
        let mut gc = GcCache::new();
        gc._cache_size
            .insert(LLType::Struct(1), size_descr_at(7, 0, &[16, 24, 32]));
        let shell = LLType::Struct(2);
        gc._cache_size.insert(shell.clone(), shell_descr(32));
        gc._cache_field
            .insert(shell, shell_fields(&[("a", 40), ("b", 48)]));

        assert_eq!(gc.size_shell_census(), [2, 1, 1, 0, 0]);
    }

    /// One cached field matches any same-size struct that happens to own that
    /// slot, so `aliased` alone overstates its case. `aliased_multi` is the
    /// share that does not rest on single-slot evidence.
    #[test]
    fn size_shell_census_separates_single_slot_evidence() {
        let mut gc = GcCache::new();
        gc._cache_size
            .insert(LLType::Struct(1), size_descr_at(7, 0, &[16, 24, 32]));
        let shell = LLType::Struct(2);
        gc._cache_size.insert(shell.clone(), shell_descr(32));
        gc._cache_field.insert(shell, shell_fields(&[("a", 16)]));

        assert_eq!(gc.size_shell_census(), [2, 1, 1, 1, 0]);
    }

    /// A fieldless parent with nothing cached under its key is just a struct
    /// the JIT never indexed. It is `fieldless` but not `shadowing`, because
    /// nothing is being contradicted.
    #[test]
    fn size_shell_census_does_not_call_an_unused_struct_a_shell() {
        let mut gc = GcCache::new();
        gc._cache_size.insert(LLType::Struct(2), shell_descr(32));

        assert_eq!(gc.size_shell_census(), [1, 1, 0, 0, 0]);
    }

    /// A rederived field must survive being looked up twice.
    ///
    /// `get_field_descr` normalises `index_in_parent` against the parent, so the
    /// cached descr holds the derived number while the caller keeps handing in
    /// its own. Validating the cache hit against the raw argument therefore
    /// fires on the SECOND lookup of exactly the fields the normalisation exists
    /// for — and only in debug builds, where `debug_assert!` is live, so the
    /// release gate cannot see it.
    #[test]
    fn a_rederived_field_survives_a_second_lookup() {
        let mut gc = GcCache::new();
        let key = LLType::Struct(5);
        // Parent lists f16, f24, f32 — so `f24` is at index 1.
        gc._cache_size
            .insert(key.clone(), size_descr_at(7, 0, &[16, 24, 32]));
        let lookup = |gc: &mut GcCache| {
            gc.get_field_descr(
                key.clone(),
                "f24",
                None,
                24,
                8,
                Type::Int,
                false,
                false,
                ArrayFlag::Signed,
                0,
                false,
                // A header-counting producer's number, two slots high.
                Some(3),
            )
        };
        let first = lookup(&mut gc);
        assert_eq!(first.index_in_parent, 1, "the parent's own list wins");
        let second = lookup(&mut gc);
        assert_eq!(second.index_in_parent, 1);
    }

    /// The mint path binds a fieldless size-descr shell (`get_size_descr`
    /// populates no `all_fielddescrs`), so `find_index_in_parent` cannot
    /// arbitrate. The runtime published a headerless descr (`index 0`) first;
    /// a later analyzer mint of the SAME field carries a header-counting number
    /// (`index 2`). The cache hit must return the published descr and its canary
    /// must defer to that cached index rather than fire on the caller's
    /// un-arbitrable raw number — the exact `floatval 0-vs-2` /
    /// `locals_cells_stack_w 0-vs-4` panic.
    #[test]
    fn a_shell_parent_defers_to_the_cached_index_not_the_caller() {
        let mut gc = GcCache::new();
        let key = LLType::Struct(9);
        // Fieldless shell — no positional list to rederive against.
        gc._cache_size.insert(key.clone(), size_descr_at(3, 0, &[]));
        // First lookup: the headerless producer's index 0 is cached.
        let first = gc.get_field_descr(
            key.clone(),
            "floatval",
            None,
            16,
            8,
            Type::Float,
            false,
            false,
            ArrayFlag::Float,
            0,
            false,
            Some(0),
        );
        assert_eq!(first.index_in_parent, 0);
        // Second lookup: a header-counting producer hands index 2. No parent
        // arbitrates, so the canary must defer to the cached 0 (no panic) and
        // return the cached descr unchanged.
        let second = gc.get_field_descr(
            key.clone(),
            "floatval",
            None,
            16,
            8,
            Type::Float,
            false,
            false,
            ArrayFlag::Float,
            0,
            false,
            Some(2),
        );
        assert_eq!(second.index_in_parent, 0, "the cached descr wins");
    }

    /// The upgrade rule itself is unchanged when both sides carry a vtable.
    #[test]
    fn a_fuller_layout_still_upgrades_when_both_carry_a_vtable() {
        let mut gc = GcCache::new();
        let key = LLType::Struct(11);
        gc.register_keyed_size(key.clone(), size_descr(7, 0x1000, 1));
        gc.register_keyed_size(key.clone(), size_descr(9, 0x2000, 3));
        let cached = gc._cache_size.get(&key).unwrap().as_size_descr().unwrap();
        assert_eq!(cached.type_id(), 9);
        assert_eq!(cached.all_fielddescrs().len(), 3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_instantiation_suffix_drops_generic_args() {
        // `<…>` is truncated at the first `<`, keeping any module
        // qualifier and dropping a nested argument's own `::` / `<`.
        assert_eq!(strip_instantiation_suffix("Result<bool>"), "Result");
        assert_eq!(
            strip_instantiation_suffix("m::Result<bool, E>"),
            "m::Result"
        );
        assert_eq!(
            strip_instantiation_suffix("Result<core::option::Option<i32>>"),
            "Result"
        );
        // No `<` → unchanged (the behaviour every existing bare-keyed
        // caller relies on).
        assert_eq!(strip_instantiation_suffix("Color"), "Color");
        assert_eq!(strip_instantiation_suffix("module::Color"), "module::Color");
    }

    #[test]
    fn shaped_array_name_requires_a_suffix() {
        assert!(!is_shaped_array_name("Array"));
        assert!(is_shaped_array_name("Array<*mut PyObject;1>"));
        assert!(!is_shaped_array_name("module::Array<i64;2>"));
    }

    #[test]
    fn strip_generic_args_keeps_variant_tail() {
        // The balanced `<…>` group is removed, the trailing `::variant`
        // segment kept — the spelling a per-instantiation variant key
        // (`Result<Tuple>::Ok`) must resolve under the bare template rows.
        assert_eq!(strip_generic_args("Result<Tuple>::Ok"), "Result::Ok");
        assert_eq!(strip_generic_args("Result<Tuple>"), "Result");
        assert_eq!(
            strip_generic_args("m::Result<core::option::Option<i32>>::Ok"),
            "m::Result::Ok"
        );
        // No `<` → borrowed unchanged.
        assert_eq!(strip_generic_args("Result::Ok"), "Result::Ok");
        assert!(matches!(
            strip_generic_args("Result::Ok"),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    #[test]
    fn generic_struct_identity_keeps_monomorphizations_distinct() {
        let template = StructId::from_canonical("option::Option::Some");
        assert_eq!(template.instantiate("<i64>"), template.instantiate("<i64>"));
        assert_ne!(
            template.instantiate("<i64>"),
            template.instantiate("<bool>")
        );
        assert_ne!(template.instantiate("<i64>"), template);
    }

    #[test]
    fn generic_args_span_is_balanced_and_excludes_variant_tail() {
        assert_eq!(
            generic_args_span("Result<i64,PyError>::Ok"),
            Some("<i64,PyError>")
        );
        assert_eq!(
            generic_args_span("Option<Result<i32,E>>::Some"),
            Some("<Result<i32,E>>")
        );
        assert_eq!(generic_args_span("Result::Ok"), None);
        assert_eq!(generic_args_span("Result<i64"), None);
    }

    #[test]
    fn canonical_struct_name_reattaches_generic_suffix() {
        // A leaf with no origin-registry entry reattaches its suffix
        // verbatim (the recursion bottoms out in the ungeneric path).
        // `Tuple` has no `::`, so the leaf stays bare and the `<…>` is
        // preserved exactly — the constructor and the prologue pre-mint
        // pass the identical string and resolve one classdef.
        assert_eq!(canonical_struct_name("Result<Tuple>"), "Result<Tuple>");
        // Already-qualified base passes through verbatim under the suffix.
        assert_eq!(
            canonical_struct_name("core::result::Result<Tuple>"),
            "core::result::Result<Tuple>"
        );
        // Idempotent.
        assert_eq!(
            canonical_struct_name(&canonical_struct_name("Result<Tuple>")),
            canonical_struct_name("Result<Tuple>")
        );
    }

    // ── FFI call surface parity tests (rpython/jit/metainterp/test/test_fficall.py) ──

    /// Concrete CallDescr for testing.
    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
        result_size: usize,
        result_signed: bool,
        effect: EffectInfo,
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            self.result_size
        }
        fn is_result_signed(&self) -> bool {
            self.result_signed
        }
        fn get_extra_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    #[test]
    fn test_simple_fail_descr_vector_info_is_head_linked() {
        let descr = SimpleFailDescr::new(1, 2, vec![Type::Int, Type::Int]);
        descr.attach_vector_info(AccumInfo {
            prev: None,
            failargs_pos: 0,
            variable: OpRef::int_op(10),
            location: OpRef::int_op(20),
            accum_operation: '+',
            scalar: OpRef::NONE,
        });
        descr.attach_vector_info(AccumInfo {
            prev: None,
            failargs_pos: 1,
            variable: OpRef::int_op(11),
            location: OpRef::int_op(21),
            accum_operation: '*',
            scalar: OpRef::NONE,
        });

        let vector_info = descr.vector_info();
        assert_eq!(vector_info.len(), 2);
        assert_eq!(vector_info[0].failargs_pos, 1);
        assert_eq!(vector_info[1].failargs_pos, 0);
        assert_eq!(
            vector_info[0].prev.as_ref().map(|info| info.failargs_pos),
            Some(0)
        );
        assert!(vector_info[1].prev.is_none());

        let cloned = descr.clone();
        let cloned_vector_info = cloned.vector_info();
        assert_eq!(cloned_vector_info.len(), 2);
        assert_eq!(cloned_vector_info[0].failargs_pos, 1);
        assert_eq!(
            cloned_vector_info[0]
                .prev
                .as_ref()
                .map(|info| info.failargs_pos),
            Some(0)
        );
    }

    #[test]
    fn test_call_descr_stores_arg_types_and_result() {
        // Parity with test_simple_call_int: CallDescr correctly stores arg types and result type
        let descr = TestCallDescr {
            arg_types: vec![Type::Int, Type::Int],
            result_type: Type::Int,
            result_size: 8,
            result_signed: true,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.arg_types(), &[Type::Int, Type::Int]);
        assert_eq!(descr.result_type(), Type::Int);
        assert_eq!(descr.result_size(), 8);
        assert!(descr.is_result_signed());
    }

    #[test]
    fn test_call_descr_float_args() {
        // Parity with test_simple_call_float
        let descr = TestCallDescr {
            arg_types: vec![Type::Float, Type::Float],
            result_type: Type::Float,
            result_size: 8,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.arg_types(), &[Type::Float, Type::Float]);
        assert_eq!(descr.result_type(), Type::Float);
    }

    #[test]
    fn test_call_descr_void_result() {
        // Parity with test_returns_none
        let descr = TestCallDescr {
            arg_types: vec![Type::Int, Type::Int],
            result_type: Type::Void,
            result_size: 0,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.result_type(), Type::Void);
        assert_eq!(descr.result_size(), 0);
    }

    #[test]
    fn test_call_descr_many_arguments() {
        // Parity with test_many_arguments: various argument counts
        for count in [0, 6, 20] {
            let arg_types = vec![Type::Int; count];
            let descr = TestCallDescr {
                arg_types,
                result_type: Type::Int,
                result_size: 8,
                result_signed: true,
                effect: EffectInfo::default(),
            };
            assert_eq!(descr.arg_types().len(), count);
        }
    }

    #[test]
    fn test_call_descr_ref_result() {
        let descr = TestCallDescr {
            arg_types: vec![Type::Ref],
            result_type: Type::Ref,
            result_size: 8,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.arg_types(), &[Type::Ref]);
        assert_eq!(descr.result_type(), Type::Ref);
    }

    #[test]
    fn test_call_descr_downcasts_via_trait() {
        let descr: Arc<dyn Descr> = Arc::new(TestCallDescr {
            arg_types: vec![Type::Int],
            result_type: Type::Int,
            result_size: 8,
            result_signed: true,
            effect: EffectInfo::default(),
        });
        let cd = descr.as_call_descr().expect("should downcast to CallDescr");
        assert_eq!(cd.arg_types(), &[Type::Int]);
        assert_eq!(cd.result_type(), Type::Int);
    }

    #[test]
    fn test_call_target_token_default_none() {
        let descr = TestCallDescr {
            arg_types: vec![],
            result_type: Type::Void,
            result_size: 0,
            result_signed: false,
            effect: EffectInfo::default(),
        };
        assert_eq!(descr.call_target_token(), None);
    }

    #[test]
    fn test_effect_info_default_can_raise() {
        let ei = EffectInfo::default();
        assert_eq!(ei.extraeffect, ExtraEffect::CanRaise);
        assert_eq!(ei.oopspecindex, OopSpecIndex::None);
        assert!(ei.check_can_raise(false));
        assert!(!ei.check_is_elidable());
        assert!(ei.extraeffect != ExtraEffect::LoopInvariant);
    }

    #[test]
    fn test_effect_info_elidable_variants() {
        let elidable_effects = [
            ExtraEffect::ElidableCannotRaise,
            ExtraEffect::ElidableOrMemoryError,
            ExtraEffect::ElidableCanRaise,
        ];
        for effect in elidable_effects {
            let ei = EffectInfo {
                extraeffect: effect,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            };
            assert!(ei.check_is_elidable(), "expected elidable for {effect:?}");
        }

        let non_elidable = [
            ExtraEffect::CannotRaise,
            ExtraEffect::CanRaise,
            ExtraEffect::LoopInvariant,
            ExtraEffect::ForcesVirtualOrVirtualizable,
            ExtraEffect::RandomEffects,
        ];
        for effect in non_elidable {
            let ei = EffectInfo {
                extraeffect: effect,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            };
            assert!(
                !ei.check_is_elidable(),
                "expected non-elidable for {effect:?}"
            );
        }
    }

    #[test]
    fn test_effect_info_can_raise_ordering() {
        // ExtraEffect ordering: effects >= ElidableCanRaise can raise
        // effectinfo.py: check_can_raise(ignore_memoryerror=False) is
        // self.extraeffect > EF_CANNOT_RAISE (2)
        let cannot_raise = [
            ExtraEffect::ElidableCannotRaise, // 0
            ExtraEffect::LoopInvariant,       // 1
            ExtraEffect::CannotRaise,         // 2
        ];
        for effect in cannot_raise {
            let ei = EffectInfo {
                extraeffect: effect,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            };
            assert!(
                !ei.check_can_raise(false),
                "expected cannot raise for {effect:?}"
            );
        }

        let can_raise = [
            ExtraEffect::ElidableOrMemoryError,        // 3
            ExtraEffect::ElidableCanRaise,             // 4
            ExtraEffect::CanRaise,                     // 5
            ExtraEffect::ForcesVirtualOrVirtualizable, // 6
            ExtraEffect::RandomEffects,                // 7
        ];
        for effect in can_raise {
            let ei = EffectInfo {
                extraeffect: effect,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            };
            assert!(
                ei.check_can_raise(false),
                "expected can raise for {effect:?}"
            );
        }

        // effectinfo.py: check_can_raise(ignore_memoryerror=True) is
        // self.extraeffect > EF_ELIDABLE_OR_MEMORYERROR (3)
        let cannot_raise_ignoring = [
            ExtraEffect::ElidableCannotRaise,   // 0
            ExtraEffect::LoopInvariant,         // 1
            ExtraEffect::CannotRaise,           // 2
            ExtraEffect::ElidableOrMemoryError, // 3
        ];
        for effect in cannot_raise_ignoring {
            let ei = EffectInfo {
                extraeffect: effect,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            };
            assert!(
                !ei.check_can_raise(true),
                "expected cannot raise (ignoring memoryerror) for {effect:?}"
            );
        }

        let can_raise_ignoring = [
            ExtraEffect::ElidableCanRaise,             // 4
            ExtraEffect::CanRaise,                     // 5
            ExtraEffect::ForcesVirtualOrVirtualizable, // 6
            ExtraEffect::RandomEffects,                // 7
        ];
        for effect in can_raise_ignoring {
            let ei = EffectInfo {
                extraeffect: effect,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            };
            assert!(
                ei.check_can_raise(true),
                "expected can raise (ignoring memoryerror) for {effect:?}"
            );
        }
    }

    #[test]
    fn test_effect_info_loop_invariant() {
        let ei = EffectInfo {
            extraeffect: ExtraEffect::LoopInvariant,
            oopspecindex: OopSpecIndex::None,
            ..Default::default()
        };
        assert!(ei.extraeffect == ExtraEffect::LoopInvariant);
        assert!(!ei.check_is_elidable());
        assert!(!ei.check_can_raise(false));
    }

    #[test]
    fn test_effect_info_libffi_call_oopspec() {
        // FFI calls use LibffiCall oopspec index
        let ei = EffectInfo {
            extraeffect: ExtraEffect::CanRaise,
            oopspecindex: OopSpecIndex::LibffiCall,
            ..Default::default()
        };
        assert_eq!(ei.oopspecindex, OopSpecIndex::LibffiCall);
        assert!(ei.check_can_raise(false));
    }

    #[test]
    fn test_effect_info_forces_virtual() {
        // Parity: calls that force virtualizable objects
        let ei = EffectInfo {
            extraeffect: ExtraEffect::ForcesVirtualOrVirtualizable,
            oopspecindex: OopSpecIndex::JitForceVirtualizable,
            ..Default::default()
        };
        assert!(ei.check_can_raise(false));
        assert!(!ei.check_is_elidable());
        assert_eq!(ei.oopspecindex, OopSpecIndex::JitForceVirtualizable);
    }

    #[test]
    fn test_call_release_gil_opcodes_exist() {
        use crate::resoperation::OpCode;
        // Parity with resoperation.py:1238-1248 call_release_gil_for_descr:
        // 'i' / 'f' / 'v' arms exist; 'r' is commented out as `# no such thing`
        // and is covered by the should_panic test below.
        let int_op = OpCode::call_release_gil_for_type(Type::Int);
        assert_eq!(int_op, OpCode::CallReleaseGilI);

        let float_op = OpCode::call_release_gil_for_type(Type::Float);
        assert_eq!(float_op, OpCode::CallReleaseGilF);

        let void_op = OpCode::call_release_gil_for_type(Type::Void);
        assert_eq!(void_op, OpCode::CallReleaseGilN);
    }

    #[test]
    #[should_panic(expected = "Type::Ref has no upstream counterpart")]
    fn test_call_release_gil_ref_panics() {
        use crate::resoperation::OpCode;
        // resoperation.py:1243-1244: `# no such thing` — `Type::Ref`
        // has no `CALL_RELEASE_GIL_R` arm in upstream; pyre matches by
        // panicking instead of fabricating the missing opcode.
        let _ = OpCode::call_release_gil_for_type(Type::Ref);
    }

    #[test]
    fn test_fail_descr_trait() {
        #[derive(Debug)]
        struct TestFailDescr {
            index: u32,
            arg_types: Vec<Type>,
        }
        impl Descr for TestFailDescr {
            fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
                Some(self)
            }
        }
        impl FailDescr for TestFailDescr {
            fn fail_index(&self) -> u32 {
                self.index
            }
            fn fail_arg_types(&self) -> &[Type] {
                &self.arg_types
            }
        }

        let fd = TestFailDescr {
            index: 7,
            arg_types: vec![Type::Int, Type::Ref],
        };
        assert_eq!(fd.fail_index(), 7);
        assert_eq!(fd.fail_arg_types(), &[Type::Int, Type::Ref]);
        assert!(!fd.is_finish());
        assert_eq!(fd.trace_id(), 0);
        // Ref slot is a GC ref
        assert!(fd.is_gc_ref_slot(1));
        // Int slot is not
        assert!(!fd.is_gc_ref_slot(0));
    }

    #[test]
    fn test_debug_merge_point_descr_repr() {
        let info = DebugMergePointInfo::new("testjit", "bytecode LOAD at 12", 12, 0);
        let descr = DebugMergePointDescr::new(info);
        let repr = descr.repr();
        assert!(repr.contains("testjit"));
        assert!(repr.contains("bytecode LOAD at 12"));
        assert!(repr.contains("pc=12"));
        assert!(repr.contains("depth=0"));
    }
}

// ── Factory functions (descr.py: get_field_descr, get_size_descr, etc.) ──

/// Create a field descriptor with the given layout.
/// Fresh constructor — does NOT go through GcCache.
/// For cached descriptors, use GcCache::get_field_descr().
pub fn make_field_descr(
    offset: usize,
    field_size: usize,
    field_type: Type,
    flag: ArrayFlag,
) -> DescrRef {
    Arc::new(SimpleFieldDescr::new(0, offset, field_size, field_type, false).with_flag(flag))
}

/// Create a field descriptor whose `parent_descr` back-references an existing
/// SizeDescr, for a field whose bare name is absent from that struct's
/// flattened `all_fielddescrs`.  This is the inline-enum-discriminant case: the
/// getfield names the enclosing Rust field (e.g. `strategy`) and bakes the
/// absolute byte offset, whereas the flattened layout only carries the embedded
/// variant's `__discriminant` at its sub-struct-relative offset, so the
/// by-name group lookup misses.  The getfield's own offset/index_in_parent are
/// authoritative; wiring them to the containing struct's SizeDescr restores the
/// `descr.py:238 FieldDescr.parent_descr` invariant that
/// `OptContext::ensure_ptr_info_arg0` (`optimizer.py:478`) requires.
///
/// `parent` must be a cache-owned `DescrRef` (kept alive process-wide) so the
/// stored `Weak` upgrades; a transient group-local Arc would dangle after the
/// caller drops it.
pub fn make_field_descr_with_parent(
    offset: usize,
    field_size: usize,
    field_type: Type,
    flag: ArrayFlag,
    index_in_parent: usize,
    name: String,
    parent: &DescrRef,
) -> DescrRef {
    let parent_weak: Weak<dyn Descr> = Arc::downgrade(parent);
    let mut fd = SimpleFieldDescr::new_with_name(
        index_in_parent as u32,
        offset,
        field_size,
        field_type,
        false,
        flag,
        name.clone(),
        name,
    );
    fd.index_in_parent = index_in_parent;
    fd.parent_descr = RwLock::new(Some(parent_weak));
    Arc::new(fd)
}

/// Resolve the canonical `FieldDescr` a parent `SizeDescr` already owns for a
/// given byte `offset`, rather than minting a partial copy.
///
/// resume.py:597-603 `self.setfields` → `decoder.setfield(struct, fieldnum,
/// fielddescr)` uses the parent's live `FieldDescr` (full immutable /
/// quasi-immutable / name / ei_index). The parent `SizeDescr` is the same live
/// descr the original trace recorded against, so its `all_fielddescrs()` are
/// the canonical field descriptors keyed by `index_in_parent`; return the one
/// whose `offset()` matches instead of reconstructing a lossy partial. Keying
/// by `descr_index()` = `index_in_parent` (a small sequential index) avoids the
/// `stable_field_index` hash fallback that OOMs `ensure_field_descr_slot`.
pub fn field_descr_from_parent_by_offset(parent: &DescrRef, offset: usize) -> DescrRef {
    let sd = parent
        .as_size_descr()
        .expect("field_descr_from_parent_by_offset: parent is not a SizeDescr");
    let fd = sd
        .all_fielddescrs()
        .iter()
        .find(|fd| fd.offset() == offset)
        .unwrap_or_else(|| {
            panic!("FieldDescr offset {offset} is not present in parent SizeDescr all_fielddescrs")
        });
    fd.clone() as DescrRef
}

/// Create a field descriptor with explicit index and immutability.
pub fn make_field_descr_full(
    index: u32,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_immutable: bool,
) -> DescrRef {
    std::sync::Arc::new(SimpleFieldDescr::new(
        index,
        offset,
        field_size,
        field_type,
        is_immutable,
    ))
}

/// Create a size descriptor.
/// Fresh constructor — does NOT go through GcCache.
pub fn make_size_descr(size: usize) -> DescrRef {
    Arc::new(SimpleSizeDescr::new(0, size, 0))
}

/// Create a size descriptor with explicit index and type_id.
pub fn make_size_descr_full(index: u32, size: usize, type_id: u32) -> DescrRef {
    std::sync::Arc::new(SimpleSizeDescr::new(index, size, type_id))
}

/// Create a size descriptor with vtable (for NEW_WITH_VTABLE objects).
pub fn make_size_descr_with_vtable(
    index: u32,
    size: usize,
    type_id: u32,
    vtable: usize,
) -> DescrRef {
    std::sync::Arc::new(SimpleSizeDescr::with_vtable(index, size, type_id, vtable))
}

/// Create an array descriptor.
/// Fresh constructor — does NOT go through GcCache.
pub fn make_array_descr(base_size: usize, item_size: usize, item_type: Type) -> DescrRef {
    Arc::new(SimpleArrayDescr::new(0, base_size, item_size, 0, item_type))
}

/// Create an array descriptor with explicit signedness (`descr.py:241-254
/// get_type_flag`).  Like [`make_array_descr`] but lets the caller force
/// `ArrayFlag::Signed` for `Type::Int` arrays — the default
/// (`from_item_type` second arg `is_struct=false`) maps `Int → Unsigned`
/// per RPython `descr.py:254 FLAG_UNSIGNED` for the unresolved-integer
/// case, which loses the descriptor-level sign distinction the
/// dispatch JitCode opcode-fetch needs.  Used by the trace recorder to
/// keep `(itemsize, is_signed)` paired on the same `ArrayDescr` so
/// `is_item_signed()` round-trips through optimizer / backend reads
/// (`llmodel.py:591 unpack_arraydescr_size + read_int_at_mem(... size,
/// sign)` parity).
pub fn make_array_descr_signed(
    base_size: usize,
    item_size: usize,
    item_type: Type,
    is_signed: bool,
) -> DescrRef {
    let flag = if is_signed && item_type == Type::Int {
        ArrayFlag::Signed
    } else {
        ArrayFlag::from_item_type(item_type, false)
    };
    Arc::new(SimpleArrayDescr::with_flag(
        0, base_size, item_size, 0, item_type, flag,
    ))
}

/// Create an array descriptor with explicit index and type_id.
pub fn make_array_descr_full(
    index: u32,
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
) -> DescrRef {
    std::sync::Arc::new(SimpleArrayDescr::new(
        index, base_size, item_size, type_id, item_type,
    ))
}

/// Build an [`ArrayDescr`] preserving the lltype-discriminant set the
/// codewriter records on `BhDescr::Array`.  Threads `type_id`, the
/// pointer/struct discriminator (which selects the right
/// [`ArrayFlag`]), `lendescr`, and `is_pure` onto the resulting
/// `SimpleArrayDescr`.
///
/// Caller-side caching contract: the resulting Arc identity is
/// stable per-call (each invocation mints a fresh Arc), so any
/// caller that wants `Arc::ptr_eq`-stable identity for repeated
/// calls with the same `BhDescr::Array` shape must memoise on its
/// own.  RPython's `descr.py:348 get_array_descr` keys the cache on
/// the lltype object itself, which encodes ALL of `type_id`,
/// `is_array_of_pointers`, `is_array_of_structs`, `lendescr`,
/// `is_pure`, `interior_fields`, etc; pyre callers MUST use a key
/// that distinguishes every pair of inputs that could resolve to a
/// different `SimpleArrayDescr` shape, otherwise two distinct
/// lltypes will collapse to a single cached Arc and break optimizer
/// invariants that read those fields.  Pyre's only current caller
/// (`pyjitpl::dispatch::dispatch_array_descr_ref`) uses
/// `DispatchArrayDescrKey` which captures the eight `BhDescr::Array`
/// discriminants and pins `lendescr=None` / `is_pure=false` /
/// empty `interior_fields` for the bytecode-array
/// (`program: &[u8]`) opcode-fetch path; future general-purpose
/// callers must extend their cache key to cover whichever inputs
/// they vary.
///
/// `lendescr` mirrors `descr.py:286-287 self.lendescr = lendescr` —
/// the caller passes a pre-built `FieldDescr` for `nolength=False`
/// arrays (the upstream `get_field_arraylen_descr` shape) and `None`
/// for `ARRAY._hints['nolength']` shapes.  `is_pure` mirrors
/// `descr.py:288 self._is_pure = ARRAY._hints.get('immutable',
/// False)` — drives the optimizer's pure-array fold.
///
/// `ei_index` mirrors `effectinfo.py:465 compute_bitstrings()`:
/// publishes the codewriter-side `array_index`
/// (`call.rs::DescrIndexRegistry::array_index`) onto the resulting
/// descr via `set_ei_index` so heap.rs's `force_from_effectinfo`
/// reads the same bitstring slot the producer wrote.
/// `u32::MAX` is the unset sentinel.
///
/// `interior_field_descrs` mirrors `descr.py:372-375 arraydescr.
/// all_interiorfielddescrs` — every interior FieldDescr must carry
/// the SAME `arraydescr` Arc identity the array itself uses.  Pyre's
/// `SimpleArrayDescr::set_all_interiorfielddescrs` writes through a
/// `OnceLock` AFTER the parent Arc is minted; the caller is expected
/// to have pre-built each `SimpleInteriorFieldDescr` referencing this
/// parent.  Pass `Vec::new()` for primitive-item arrays.
///
/// Field provenance (RPython `descr.py:240-289 ArrayDescr.__init__`):
/// - `lendescr`: `descr.py:286-287 self.lendescr = lendescr` — points
///   at the array length field's `FieldDescr` for arrays whose
///   `ARRAY._hints.get('nolength')` is false.  `None` for fixed-size
///   buffers (pyre's `program: &[u8]` opcode-fetch path).
/// - `is_pure`: `descr.py:288 self._is_pure = ARRAY._hints.get(
///   'immutable', False)` — set on `Ptr(rstr.STR)` /
///   `Ptr(rstr.UNICODE)` and any `lltype.Array(... hints={'immutable':
///   True})`.  Drives the optimizer's pure-array fold.
///
#[expect(
    clippy::too_many_arguments,
    reason = "The argument order is the stable JIT IR/descriptor or generated-interpreter ABI shape; grouping it into a Rust-only options object would obscure opcode-field correspondence and macro call-site parity"
)]
pub fn make_array_descr_from_lltype_shape(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    // `len_offset` is the BhDescr-side carry-over; the helper itself
    // reads `lendescr` for the per-array length FieldDescr.  Callers
    // that already have a pre-built `lendescr` pass `None` here and
    // the real FieldDescr below; callers that only have the raw
    // offset can construct an inline `SimpleFieldDescr` at the call
    // site mirroring `descr.py:256-267 get_field_arraylen_descr`.
    _len_offset: Option<usize>,
    item_type: Type,
    is_array_of_pointers: bool,
    is_array_of_structs: bool,
    is_item_signed: bool,
    is_gc_managed: bool,
    lendescr: Option<DescrRef>,
    is_pure: bool,
    ei_index: u32,
    interior_field_descrs: Vec<DescrRef>,
) -> Arc<SimpleArrayDescr> {
    // RPython `descr.py:241-254 get_type_flag` precedence: pointer >
    // struct > primitive.  `is_array_of_pointers` selects FLAG_POINTER,
    // `is_array_of_structs` selects FLAG_STRUCT, otherwise the
    // primitive item_type drives FLAG_FLOAT / FLAG_SIGNED /
    // FLAG_UNSIGNED.  `is_item_signed` only matters for the integer
    // primitive branch (`Type::Int` arrays whose backing lltype is
    // `lltype.Signed`).
    let flag = if is_array_of_pointers {
        ArrayFlag::Pointer
    } else if is_array_of_structs {
        ArrayFlag::Struct
    } else {
        match item_type {
            Type::Float => ArrayFlag::Float,
            Type::Int => {
                if is_item_signed {
                    ArrayFlag::Signed
                } else {
                    ArrayFlag::Unsigned
                }
            }
            Type::Ref => ArrayFlag::Pointer,
            Type::Void => ArrayFlag::Void,
        }
    };
    let mut descr = SimpleArrayDescr::with_flag(0, base_size, item_size, type_id, item_type, flag);
    descr.lendescr = lendescr;
    descr.is_pure = is_pure;
    descr.set_is_gc_managed(is_gc_managed);
    let arc = Arc::new(descr);
    if ei_index != u32::MAX {
        arc.set_ei_index(ei_index);
    }
    if !interior_field_descrs.is_empty() {
        // RPython `descr.py:372-375` populates
        // `arraydescr.all_interiorfielddescrs` after the array descr is
        // minted, using the same arraydescr Arc inside each
        // InteriorFieldDescr (`descr.py:388 InteriorFieldDescr.__init__`).
        // The caller must therefore have built the `interior_field_descrs`
        // with `arc` as their parent — pyre's
        // `SimpleArrayDescr::set_all_interiorfielddescrs` writes through
        // `OnceLock`, leaving the parent Arc identity stable.
        arc.set_all_interiorfielddescrs(interior_field_descrs);
    }
    arc
}

/// Create a call descriptor.
/// Fresh constructor — does NOT go through GcCache.
/// descr.py:647-675: get_call_descr(gccache, ARGS, RESULT, extrainfo).
/// `result_signed` defaults to true for Int results (RPython Signed type),
/// false for all others.
pub fn make_call_descr(arg_types: Vec<Type>, result_type: Type, effect: EffectInfo) -> DescrRef {
    let result_size = match result_type {
        Type::Int | Type::Ref => 8,
        Type::Float => 8,
        Type::Void => 0,
    };
    // descr.py:664: result_signed = get_type_flag(RESULT) == FLAG_SIGNED
    // For Signed (default int), this is true.
    let result_signed = result_type == Type::Int;
    Arc::new(SimpleCallDescr::new(
        0,
        arg_types,
        result_type,
        result_signed,
        result_size,
        effect,
    ))
}

/// Create a call descriptor with explicit index.
pub fn make_call_descr_full(
    index: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_signed: bool,
    result_size: usize,
    effect: EffectInfo,
) -> DescrRef {
    std::sync::Arc::new(SimpleCallDescr::new(
        index,
        arg_types,
        result_type,
        result_signed,
        result_size,
        effect,
    ))
}

/// Create a call descriptor with explicit index and RPython result class.
///
/// `result_type` stays in pyre's coarse IR alphabet (`Int/Ref/Float/Void`);
/// `result_class` preserves the codewriter/backend char alphabet
/// (`i/r/f/L/S/v`) used by RPython `CallDescr.result_type`.
pub fn make_call_descr_full_with_result_class(
    index: u32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_class: char,
    result_signed: bool,
    result_size: usize,
    effect: EffectInfo,
) -> DescrRef {
    std::sync::Arc::new(SimpleCallDescr::new_with_result_class(
        index,
        arg_types,
        result_type,
        result_class,
        result_signed,
        result_size,
        effect,
    ))
}

/// Create a fail descriptor.
pub fn make_fail_descr(fail_index: u32, fail_arg_types: Vec<Type>) -> DescrRef {
    std::sync::Arc::new(SimpleFailDescr::new(0, fail_index, fail_arg_types))
}

/// Create a finish descriptor.
pub fn make_finish_descr(fail_index: u32, fail_arg_types: Vec<Type>) -> DescrRef {
    std::sync::Arc::new(SimpleFailDescr::finish(0, fail_index, fail_arg_types))
}

/// Create a loop TargetToken descriptor.
pub fn make_loop_target_descr(token_id: u64, is_preamble_target: bool) -> DescrRef {
    std::sync::Arc::new(BasicLoopTargetDescr::new(token_id, is_preamble_target))
}

// ── descr.py: unpack helpers ──

/// descr.py: unpack_fielddescr(descr)
/// Extract offset and type from a field descriptor.
pub fn unpack_fielddescr(descr: &DescrRef) -> Option<(usize, usize, Type)> {
    let fd = descr.as_field_descr()?;
    Some((fd.offset(), fd.field_size(), fd.field_type()))
}

/// descr.py: unpack_arraydescr(descr)
/// Extract base size, item size, and type from an array descriptor.
pub fn unpack_arraydescr(descr: &DescrRef) -> Option<(usize, usize, Type)> {
    let ad = descr.as_array_descr()?;
    Some((ad.base_size(), ad.item_size(), ad.item_type()))
}

/// descr.py: unpack_interiorfielddescr(descr)
/// Extract array and field info from an interior field descriptor.
pub fn unpack_interiorfielddescr(descr: &DescrRef) -> Option<(usize, usize, usize, usize, Type)> {
    let ifd = descr.as_interior_field_descr()?;
    let ad = ifd.array_descr();
    let fd = ifd.field_descr();
    Some((
        ad.base_size(),
        ad.item_size(),
        fd.offset(),
        fd.field_size(),
        fd.field_type(),
    ))
}
