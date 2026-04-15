/// Descriptor traits for the JIT IR.
///
/// Translated from rpython/jit/metainterp/history.py (AbstractDescr)
/// and rpython/jit/backend/llsupport/descr.py.
///
/// Descriptors carry type metadata needed by the optimizer and backend
/// for field access, array access, function calls, and guard failures.
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::Weak;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::OpRef;
use crate::value::Type;
use serde::{Deserialize, Serialize};

/// Opaque reference to a descriptor, shared across the JIT pipeline.
pub type DescrRef = Arc<dyn Descr>;

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
    /// Structural key — two calls with the same signature + effects
    /// share one CallDescr.
    Func {
        arg_classes: String,
        result_type: Type,
        /// descr.py:664: result_signed = get_type_flag(RESULT) == FLAG_SIGNED
        result_signed: bool,
        /// descr.py:662: result_size = symbolic.get_size(RESULT_ERASED, tsc)
        result_size: usize,
        extraeffect: u8,
        oopspecindex: u16,
        readonly_descrs_fields: u64,
        write_descrs_fields: u64,
        readonly_descrs_arrays: u64,
        write_descrs_arrays: u64,
        /// effectinfo.py: bitstring_readonly_descrs_interiorfields
        readonly_descrs_interiorfields: u64,
        /// effectinfo.py: bitstring_write_descrs_interiorfields
        write_descrs_interiorfields: u64,
        can_invalidate: bool,
        can_collect: bool,
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
    /// descr.py:665: get_call_descr key tuple.
    pub fn func_key(
        arg_types: &[Type],
        result_type: Type,
        result_signed: bool,
        result_size: usize,
        effect: &EffectInfo,
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
            result_signed,
            result_size,
            extraeffect: effect.extraeffect as u8,
            oopspecindex: effect.oopspecindex as u16,
            readonly_descrs_fields: effect.readonly_descrs_fields,
            write_descrs_fields: effect.write_descrs_fields,
            readonly_descrs_arrays: effect.readonly_descrs_arrays,
            write_descrs_arrays: effect.write_descrs_arrays,
            readonly_descrs_interiorfields: effect.readonly_descrs_interiorfields,
            write_descrs_interiorfields: effect.write_descrs_interiorfields,
            can_invalidate: effect.can_invalidate,
            can_collect: effect.can_collect,
        }
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
    /// descr.py:19: _cache_field[STRUCT][fieldname]
    pub _cache_field: HashMap<LLType, HashMap<String, DescrRef>>,
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

    /// descr.py:109 + gc.py:536 init_size_descr: dense sequential tid.
    /// RPython's `heaptracker.register_TYPE` hands out a fresh small
    /// integer per STRUCT so `guard_class(obj, tid)` can use it as a
    /// direct slot. We generate the same shape here by allocating
    /// monotonically on first `get_size_descr` per key — collision-free
    /// regardless of the caller-supplied LLType key derivation.
    next_struct_tid: u32,
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
            // tid 0 is reserved as "no class" / sentinel.
            next_struct_tid: 1,
        }
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

    /// descr.py:49-50 init_size_descr(self, STRUCT, sizedescr).
    /// Hook for subclass overrides (e.g. GcLLDescr_framework sets tid).
    ///
    /// gc.py:536-542: GcLLDescr_framework.init_size_descr sets
    /// `descr.tid = combine_ushort(type_id, 0)`. Called BEFORE
    /// wrapping in Arc.
    pub fn init_size_descr(&self, _key: &LLType, _sizedescr: &mut SimpleSizeDescr) {
        // Base class does nothing — matches descr.py:50 `pass`.
    }

    /// descr.py:52-54 init_array_descr(self, ARRAY, arraydescr).
    /// Hook for subclass overrides (e.g. GcLLDescr_framework sets tid).
    ///
    /// gc.py:544-549: GcLLDescr_framework.init_array_descr sets
    /// `descr.tid = combine_ushort(type_id, 0)`. The `&mut` reference
    /// allows the hook to mutate the descriptor BEFORE it is wrapped
    /// in Arc, matching RPython's mutable-object semantics.
    pub fn init_array_descr(&self, _key: &LLType, _arraydescr: &mut SimpleArrayDescr) {
        // Base class: assert only — matches descr.py:53-54.
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
    /// The numeric `tid` stored on the returned SizeDescr is allocated
    /// monotonically from `next_struct_tid` (descr.py:109 +
    /// gc.py:536 init_size_descr) — caller does not supply it. This
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
        // Fresh tid per distinct key. See field-doc on `next_struct_tid`.
        let type_id = self.next_struct_tid;
        self.next_struct_tid = self
            .next_struct_tid
            .checked_add(1)
            .expect("GcCache struct tid overflow (u32)");
        // descr.py:117-118: SizeDescr(size, vtable=vtable, immutable_flag=immutable_flag)
        let mut sd = if vtable != 0 {
            SimpleSizeDescr::with_vtable(u32::MAX, size, type_id, vtable)
        } else {
            SimpleSizeDescr::new(u32::MAX, size, type_id)
        };
        sd.is_immutable = immutable_flag;
        // descr.py:119: gccache.init_size_descr(STRUCT, sizedescr)
        // gc.py:536-542: sets descr.tid — must happen BEFORE Arc wrap.
        self.init_size_descr(&key, &mut sd);
        let descr: DescrRef = Arc::new(sd);
        // descr.py:120: cache[STRUCT] = sizedescr
        self._cache_size.insert(key, descr.clone());
        self._cache_size_order.push(descr.clone());
        // descr.py:123-126: gc_fielddescrs / all_fielddescrs
        // populated externally via SimpleSizeDescr::with_all_field_descrs
        // since we lack the heaptracker to auto-discover fields.
        descr
    }

    /// descr.py:218-239 get_field_descr(gccache, STRUCT, fieldname).
    ///
    /// `struct_key`: LLType::Struct — the owning type identity.
    /// `index_in_parent`: descr.py:228 heaptracker.get_fielddescr_index_in(STRUCT, fieldname).
    ///   The structural slot number within the parent struct's field list.
    ///   Caller must provide this — Rust has no heaptracker auto-discovery.
    /// `flag`: descr.py:226 get_type_flag(FIELDTYPE).
    ///
    /// descr.py:234-238: parent_descr = get_size_descr(gccache, STRUCT, vtable).
    /// Looked up from _cache_size[STRUCT]. Caller must ensure get_size_descr
    /// was called first (matches RPython's call at descr.py:238).
    pub fn get_field_descr(
        &mut self,
        struct_key: LLType,
        field_name: &str,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
        flag: ArrayFlag,
        index_in_parent: usize,
    ) -> DescrRef {
        // descr.py:220-221: cache[STRUCT][fieldname]
        if let Some(inner) = self._cache_field.get(&struct_key) {
            if let Some(descr) = inner.get(field_name) {
                return descr.clone();
            }
        }
        // descr.py:227: name = '%s.%s' % (STRUCT._name, fieldname)
        let type_id = match &struct_key {
            LLType::Struct(id) => *id,
            _ => 0,
        };
        let name = format!("T{type_id}.{field_name}");
        // descr.py:234-238: parent_descr = get_size_descr(gccache, STRUCT, vtable)
        let parent = self._cache_size.get(&struct_key).cloned();
        // descr.py:230-231: FieldDescr(name, offset, size, flag, index_in_parent, is_pure)
        let mut fd = SimpleFieldDescr::new_with_name(
            u32::MAX,
            offset,
            field_size,
            field_type,
            is_immutable,
            flag,
            name,
        );
        // descr.py:228: index_in_parent (from heaptracker)
        fd.index_in_parent = index_in_parent;
        // descr.py:238: fielddescr.parent_descr = get_size_descr(gccache, STRUCT, vtable)
        if let Some(ref p) = parent {
            fd.parent_descr = Some(Arc::downgrade(p));
        }
        let descr: DescrRef = Arc::new(fd);
        // descr.py:232-233: cachedict = cache.setdefault(STRUCT, {})
        let inner = self._cache_field.entry(struct_key).or_default();
        inner.insert(field_name.to_string(), descr.clone());
        self._cache_field_order.push(descr.clone());
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
        let key = LLType::func_key(&arg_types, result_type, result_signed, result_size, &effect);
        // descr.py:667-668: cache hit
        if let Some(descr) = self._cache_call.get(&key) {
            return descr.clone();
        }
        // descr.py:670-671: CallDescr(arg_classes, result_type, result_signed,
        //   result_size, extrainfo)
        let descr: DescrRef = Arc::new(SimpleCallDescr::new(
            u32::MAX,
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
    fn target_arglocs(&self) -> Vec<TargetArgLoc>;
    fn set_target_arglocs(&self, arglocs: Vec<TargetArgLoc>);
}

#[derive(Debug, Default)]
struct BasicLoopTargetDescrState {
    ll_loop_code: usize,
    target_arglocs: Vec<TargetArgLoc>,
}

#[derive(Debug)]
struct BasicLoopTargetDescr {
    token_id: u64,
    is_preamble_target: bool,
    state: Mutex<BasicLoopTargetDescrState>,
}

impl BasicLoopTargetDescr {
    fn new(token_id: u64, is_preamble_target: bool) -> Self {
        Self {
            token_id,
            is_preamble_target,
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
        self.state.lock().unwrap().ll_loop_code
    }

    fn set_ll_loop_code(&self, loop_code: usize) {
        self.state.lock().unwrap().ll_loop_code = loop_code;
    }

    fn target_arglocs(&self) -> Vec<TargetArgLoc> {
        self.state.lock().unwrap().target_arglocs.clone()
    }

    fn set_target_arglocs(&self, arglocs: Vec<TargetArgLoc>) {
        self.state.lock().unwrap().target_arglocs = arglocs;
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

    /// Whether this fail descriptor represents a FINISH exit.
    fn is_finish(&self) -> bool {
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
    fn target_descr(&self) -> Option<&DescrRef> {
        None
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

    /// Whether the given exit slot should be treated as a real GC root.
    ///
    /// Backends may override this to distinguish rooted refs from opaque
    /// handles that reuse `Type::Ref`, such as FORCE_TOKEN values.
    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        matches!(self.fail_arg_types().get(slot), Some(Type::Ref))
    }

    /// Exit slot indices that carry opaque force-token handles.
    fn force_token_slots(&self) -> &[usize] {
        &[]
    }

    /// compile.py:741-745: read status for must_compile.
    fn get_status(&self) -> u64 {
        0
    }

    /// compile.py:786-788: start_compiling — set ST_BUSY_FLAG.
    fn start_compiling(&self) {}

    /// compile.py:790-795: done_compiling — clear ST_BUSY_FLAG.
    fn done_compiling(&self) {}

    /// compile.py:750: check ST_BUSY_FLAG.
    fn is_compiling(&self) -> bool {
        false
    }

    /// history.py:143-147 / schedule.py:654-655 — attach vector resume info
    /// to a guard descriptor. Non-guard fail descriptors ignore this.
    fn attach_vector_info(&self, _info: AccumVectorInfo) {}

    /// Read back any attached vector resume info.
    fn vector_info(&self) -> Vec<AccumVectorInfo> {
        Vec::new()
    }
}

/// resume.py:65-80: AccumInfo — metadata attached to guard descriptors
/// so deoptimization can reconstruct vector accumulators.
///
/// Two distinct OpRefs following RPython's separation:
///   - `variable`: original scalar accumulator (resume.py:29 getoriginal(),
///     used for type inference)
///   - `vector_loc`: vector SSA result holding the accumulated vector
///     (regalloc.py:350 accuminfo.location, used by backend for lane reduction)
#[derive(Debug, Clone)]
pub struct AccumVectorInfo {
    pub failargs_pos: usize,
    /// resume.py:29: the original scalar variable (getoriginal()).
    pub variable: OpRef,
    /// regalloc.py:350: vector register/SSA where the accumulated vector lives.
    /// Backend reads this for extractlane + reduction at guard exit.
    pub vector_loc: OpRef,
    pub operator: char,
}

/// Descriptor for a fixed-size struct/object allocation.
///
/// Mirrors rpython/jit/backend/llsupport/descr.py SizeDescr.
pub trait SizeDescr: Descr {
    /// Total size in bytes.
    fn size(&self) -> usize;

    /// Type ID (for GC header).
    fn type_id(&self) -> u32;

    /// Whether this is an immutable object.
    fn is_immutable(&self) -> bool;

    /// Whether this is an object (has vtable).
    fn is_object(&self) -> bool {
        false
    }

    /// Vtable address, if is_object().
    fn vtable(&self) -> usize {
        0
    }

    /// descr.py: repr_of_descr()
    fn repr_of_descr(&self) -> String {
        format!(
            "SizeDescr(size={}, type_id={})",
            self.size(),
            self.type_id()
        )
    }

    /// Field descriptors for fields containing GC pointers.
    fn gc_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        &[]
    }

    /// All field descriptors (not just GC pointer ones).
    /// descr.py: get_all_interiorfielddescrs()
    fn all_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        self.gc_field_descrs() // default: same as gc_field_descrs
    }

    /// Number of fields.
    fn num_fields(&self) -> usize {
        self.all_field_descrs().len()
    }
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

    /// descr.py:227 — field name. Format is either:
    /// - `"STRUCT.fieldname"` (from codewriter: descr.py:227)
    /// - `"typeptr"` (from pyre tracer: ob_type_descr)
    /// - `""` (unnamed/dynamic field descriptors)
    fn field_name(&self) -> &str {
        ""
    }

    /// heaptracker.py:66: `if name == 'typeptr': continue`
    ///
    /// RPython filters typeptr by raw field name BEFORE creating
    /// descriptors (heaptracker.py:60-67). In majit, descriptors are
    /// already created, so we check the name at use time.
    ///
    /// Handles both formats:
    /// - `"typeptr"` (pyre tracer ob_type_descr)
    /// - `"STRUCT.typeptr"` (codewriter format, descr.py:227)
    fn is_typeptr(&self) -> bool {
        let name = self.field_name();
        name == "typeptr" || name.ends_with(".typeptr")
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

    /// descr.py:291 ArrayDescr.get_all_fielddescrs() →
    /// all_interiorfielddescrs. For array-of-structs, returns
    /// interior field descriptors.
    fn get_all_interiorfielddescrs(&self) -> Option<&[DescrRef]> {
        None
    }

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

    /// rewrite.py:665-695 handle_call_assembler: virtualizable expansion info.
    /// When present, the backend expands a single frame reference arg into the
    /// callee's full inputarg layout by reading fields from the frame object.
    fn vable_expansion(&self) -> Option<&VableExpansion> {
        None
    }
}

/// rewrite.py:665-695 handle_call_assembler expansion recipe.
///
/// Describes how to expand a single virtualizable (frame) reference into the
/// full set of inputargs expected by the callee's compiled loop. The backend
/// reads scalar fields and array items from the frame object at the specified
/// byte offsets.
///
/// Layout: `[frame_ref, scalar_0, scalar_1, ..., array_item_0, array_item_1, ...]`
#[derive(Debug, Clone)]
pub struct VableExpansion {
    /// Scalar fields: `[(byte_offset_in_frame, type)]`.
    /// e.g. `[(NI_OFS, Int), (CODE_OFS, Ref), (VSD_OFS, Int), (NS_OFS, Ref)]`
    pub scalar_fields: Vec<(usize, Type)>,
    /// Byte offset of the array struct within the frame object.
    pub array_struct_offset: usize,
    /// Byte offset of the data pointer within the array struct.
    pub array_ptr_offset: usize,
    /// Number of array items to read.
    pub num_array_items: usize,
    /// rewrite.py:674-683 handle_call_assembler arg overrides.
    /// Each `(jitframe_slot, call_assembler_arg_index)` pair tells the
    /// backend: instead of reading from the frame, use CALL_ASSEMBLER
    /// arg[arg_index] for jitframe slot `jitframe_slot`.
    /// jitframe_slot is 0-based index in the items area (0 = frame_ref,
    /// 1 = first scalar, NUM_SCALARS+1 = first array item, etc).
    pub arg_overrides: Vec<(usize, usize)>,
    /// Constant overrides: `(jitframe_slot, value)`.
    /// The backend stores this constant instead of reading from the frame.
    pub const_overrides: Vec<(usize, i64)>,
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

// EffectInfo / ExtraEffect / OopSpecIndex moved to `crate::effectinfo`
// (mirroring rpython/jit/codewriter/effectinfo.py).
pub use crate::effectinfo::{EffectInfo, ExtraEffect, OopSpecIndex};

// ── Concrete descriptor implementations (descr.py) ──

/// Simple concrete FieldDescr for use by pyre-jit and tests.
/// RPython: `FieldDescr(name, offset, size, flag, index_in_parent, is_pure)`.
#[derive(Debug)]
pub struct SimpleFieldDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    /// RPython: FieldDescr.name — e.g. "MyStruct.field_name"
    name: String,
    offset: usize,
    field_size: usize,
    field_type: Type,
    is_immutable: bool,
    /// descr.py:151: FieldDescr.flag — type classification from get_type_flag().
    /// FLAG_POINTER, FLAG_FLOAT, FLAG_SIGNED, FLAG_UNSIGNED, FLAG_STRUCT, FLAG_VOID.
    flag: ArrayFlag,
    virtualizable: bool,
    /// descr.py:158 FieldDescr.index — slot position within the
    /// parent struct's `all_field_descrs`.
    pub index_in_parent: usize,
    /// descr.py:238 FieldDescr.parent_descr — backreference to the SizeDescr
    /// of the containing struct/object. Required by
    /// `OptContext::ensure_ptr_info_arg0` to dispatch Instance vs Struct
    /// PtrInfo per `optimizer.py:478-484`. Stored as `Weak` to break the
    /// SizeDescr → FieldDescr → SizeDescr Arc cycle introduced by
    /// `make_simple_descr_group`.
    pub parent_descr: Option<Weak<dyn Descr>>,
}

impl Clone for SimpleFieldDescr {
    fn clone(&self) -> Self {
        SimpleFieldDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            name: self.name.clone(),
            offset: self.offset,
            field_size: self.field_size,
            field_type: self.field_type,
            is_immutable: self.is_immutable,
            flag: self.flag,
            virtualizable: self.virtualizable,
            index_in_parent: self.index_in_parent,
            parent_descr: self.parent_descr.clone(),
        }
    }
}

impl SimpleFieldDescr {
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
            index,
            descr_index: AtomicI32::new(-1),
            name: String::new(),
            offset,
            field_size,
            field_type,
            is_immutable,
            flag,
            virtualizable: false,
            index_in_parent: 0,
            parent_descr: None,
        }
    }

    /// RPython: FieldDescr(name, offset, size, flag, index_in_parent, is_pure).
    /// `name` format: `"STRUCT.fieldname"` (descr.py:227).
    /// `flag`: descr.py:226 get_type_flag(FIELDTYPE).
    pub fn new_with_name(
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
        is_immutable: bool,
        flag: ArrayFlag,
        name: String,
    ) -> Self {
        SimpleFieldDescr {
            index,
            descr_index: AtomicI32::new(-1),
            name,
            offset,
            field_size,
            field_type,
            is_immutable,
            flag,
            virtualizable: false,
            index_in_parent: 0,
            parent_descr: None,
        }
    }

    /// descr.py:151: set flag directly.
    pub fn with_flag(mut self, flag: ArrayFlag) -> Self {
        self.flag = flag;
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
        self.parent_descr = Some(Arc::downgrade(&parent));
        self.index_in_parent = index_in_parent;
        self
    }
}

impl Descr for SimpleFieldDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
    }
    fn is_always_pure(&self) -> bool {
        self.is_immutable
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
    fn is_immutable(&self) -> bool {
        self.is_immutable
    }
    fn field_name(&self) -> &str {
        &self.name
    }
    fn index_in_parent(&self) -> usize {
        self.index_in_parent
    }
    fn get_parent_descr(&self) -> Option<DescrRef> {
        self.parent_descr.as_ref().and_then(|p| p.upgrade())
    }
}

/// Simple concrete SizeDescr.
#[derive(Debug)]
pub struct SimpleSizeDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    size: usize,
    type_id: u32,
    /// descr.py:64,112: SizeDescr.immutable_flag
    pub is_immutable: bool,
    vtable: usize,
    all_field_descrs: Vec<Arc<dyn FieldDescr>>,
}

impl Clone for SimpleSizeDescr {
    fn clone(&self) -> Self {
        SimpleSizeDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            size: self.size,
            type_id: self.type_id,
            is_immutable: self.is_immutable,
            vtable: self.vtable,
            all_field_descrs: self.all_field_descrs.clone(),
        }
    }
}

impl SimpleSizeDescr {
    pub fn new(index: u32, size: usize, type_id: u32) -> Self {
        SimpleSizeDescr {
            index,
            descr_index: AtomicI32::new(-1),
            size,
            type_id,
            is_immutable: false,
            vtable: 0,
            all_field_descrs: Vec::new(),
        }
    }

    pub fn with_vtable(index: u32, size: usize, type_id: u32, vtable: usize) -> Self {
        SimpleSizeDescr {
            index,
            descr_index: AtomicI32::new(-1),
            size,
            type_id,
            is_immutable: false,
            vtable,
            all_field_descrs: Vec::new(),
        }
    }

    pub fn with_all_field_descrs(mut self, all_field_descrs: Vec<Arc<dyn FieldDescr>>) -> Self {
        self.all_field_descrs = all_field_descrs;
        self
    }

    /// gc.py:541: descr.tid = llop.combine_ushort(lltype.Signed, type_id, 0)
    /// Called by init_size_descr hook before Arc wrapping.
    pub fn set_type_id(&mut self, type_id: u32) {
        self.type_id = type_id;
    }
}

impl Descr for SimpleSizeDescr {
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
        self.type_id
    }
    fn is_immutable(&self) -> bool {
        self.is_immutable
    }
    fn all_field_descrs(&self) -> &[Arc<dyn FieldDescr>] {
        &self.all_field_descrs
    }
    fn is_object(&self) -> bool {
        self.vtable != 0
    }
    fn vtable(&self) -> usize {
        self.vtable
    }
}

#[derive(Debug, Clone)]
pub struct SimpleFieldDescrSpec {
    pub index: u32,
    pub name: String,
    pub offset: usize,
    pub field_size: usize,
    pub field_type: Type,
    pub is_immutable: bool,
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

pub fn make_simple_descr_group(
    index: u32,
    size: usize,
    type_id: u32,
    vtable: usize,
    field_specs: &[SimpleFieldDescrSpec],
) -> SimpleDescrGroup {
    let field_descrs_cell = std::cell::RefCell::new(Vec::<Arc<SimpleFieldDescr>>::new());
    let field_specs = field_specs.to_vec();
    let size_descr = Arc::new_cyclic(|weak_size: &Weak<SimpleSizeDescr>| {
        let parent_descr: Weak<dyn Descr> = weak_size.clone();
        let field_descrs: Vec<Arc<SimpleFieldDescr>> = field_specs
            .iter()
            .map(|spec| {
                Arc::new(SimpleFieldDescr {
                    index: spec.index,
                    descr_index: AtomicI32::new(-1),
                    name: spec.name.clone(),
                    offset: spec.offset,
                    field_size: spec.field_size,
                    field_type: spec.field_type,
                    is_immutable: spec.is_immutable,
                    flag: spec.flag,
                    virtualizable: spec.virtualizable,
                    index_in_parent: spec.index_in_parent,
                    parent_descr: Some(parent_descr.clone()),
                })
            })
            .collect();
        *field_descrs_cell.borrow_mut() = field_descrs.clone();
        let all_field_descrs: Vec<Arc<dyn FieldDescr>> = field_descrs
            .iter()
            .cloned()
            .map(|field_descr| field_descr as Arc<dyn FieldDescr>)
            .collect();
        SimpleSizeDescr {
            index,
            descr_index: AtomicI32::new(-1),
            size,
            type_id,
            is_immutable: false,
            vtable,
            all_field_descrs,
        }
    });
    let field_descrs = field_descrs_cell.into_inner();
    SimpleDescrGroup {
        size_descr,
        field_descrs,
    }
}

/// Simple concrete ArrayDescr.
#[derive(Debug)]
pub struct SimpleArrayDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    base_size: usize,
    item_size: usize,
    type_id: u32,
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
    all_interiorfielddescrs: Option<Vec<DescrRef>>,
}

impl Clone for SimpleArrayDescr {
    fn clone(&self) -> Self {
        SimpleArrayDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            base_size: self.base_size,
            item_size: self.item_size,
            type_id: self.type_id,
            item_type: self.item_type,
            lendescr: self.lendescr.clone(),
            flag: self.flag,
            is_pure: self.is_pure,
            concrete_type: self.concrete_type,
            all_interiorfielddescrs: self.all_interiorfielddescrs.clone(),
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
            index,
            descr_index: AtomicI32::new(-1),
            base_size,
            item_size,
            type_id,
            item_type,
            lendescr: None,
            flag,
            is_pure: false,
            concrete_type: '\x00',
            all_interiorfielddescrs: None,
        }
    }

    /// RPython: ArrayDescr with explicit flag (for struct arrays).
    pub fn with_flag(
        index: u32,
        base_size: usize,
        item_size: usize,
        type_id: u32,
        item_type: Type,
        flag: ArrayFlag,
    ) -> Self {
        SimpleArrayDescr {
            index,
            descr_index: AtomicI32::new(-1),
            base_size,
            item_size,
            type_id,
            item_type,
            lendescr: None,
            flag,
            is_pure: false,
            concrete_type: '\x00',
            all_interiorfielddescrs: None,
        }
    }

    /// RPython: arraydescr.all_interiorfielddescrs = descrs
    pub fn set_all_interiorfielddescrs(&mut self, descrs: Vec<DescrRef>) {
        self.all_interiorfielddescrs = Some(descrs);
    }

    /// gc.py:548: descr.tid = llop.combine_ushort(lltype.Signed, type_id, 0)
    /// Called by init_array_descr hook before Arc wrapping.
    pub fn set_type_id(&mut self, type_id: u32) {
        self.type_id = type_id;
    }
}

impl Descr for SimpleArrayDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
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
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id
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
        self.all_interiorfielddescrs.as_deref()
    }
}

/// Simple concrete InteriorFieldDescr.
#[derive(Debug)]
pub struct SimpleInteriorFieldDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    array_descr: std::sync::Arc<SimpleArrayDescr>,
    field_descr: std::sync::Arc<SimpleFieldDescr>,
    owner_size_descr: Option<std::sync::Arc<SimpleSizeDescr>>,
}

impl Clone for SimpleInteriorFieldDescr {
    fn clone(&self) -> Self {
        SimpleInteriorFieldDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            array_descr: self.array_descr.clone(),
            field_descr: self.field_descr.clone(),
            owner_size_descr: self.owner_size_descr.clone(),
        }
    }
}

impl SimpleInteriorFieldDescr {
    pub fn new(
        index: u32,
        array_descr: std::sync::Arc<SimpleArrayDescr>,
        field_descr: std::sync::Arc<SimpleFieldDescr>,
    ) -> Self {
        SimpleInteriorFieldDescr {
            index,
            descr_index: AtomicI32::new(-1),
            array_descr,
            field_descr,
            owner_size_descr: None,
        }
    }

    pub fn new_with_owner(
        index: u32,
        array_descr: std::sync::Arc<SimpleArrayDescr>,
        field_descr: std::sync::Arc<SimpleFieldDescr>,
        owner_size_descr: std::sync::Arc<SimpleSizeDescr>,
    ) -> Self {
        SimpleInteriorFieldDescr {
            index,
            descr_index: AtomicI32::new(-1),
            array_descr,
            field_descr,
            owner_size_descr: Some(owner_size_descr),
        }
    }
}

impl Descr for SimpleInteriorFieldDescr {
    fn index(&self) -> u32 {
        self.index
    }
    fn get_descr_index(&self) -> i32 {
        self.descr_index.load(Ordering::Relaxed)
    }
    fn set_descr_index(&self, index: i32) {
        self.descr_index.store(index, Ordering::Relaxed);
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
#[derive(Debug)]
pub struct SimpleCallDescr {
    index: u32,
    /// history.py:1092: BackendDescr.descr_index = -1
    descr_index: AtomicI32,
    arg_types: Vec<Type>,
    result_type: Type,
    result_size: usize,
    /// descr.py:453: CallDescr.result_flag — computed from result_type +
    /// result_signed in __init__ (descr.py:478-493).
    result_flag: ArrayFlag,
    effect: EffectInfo,
}

impl Clone for SimpleCallDescr {
    fn clone(&self) -> Self {
        SimpleCallDescr {
            index: self.index,
            descr_index: AtomicI32::new(self.descr_index.load(Ordering::Relaxed)),
            arg_types: self.arg_types.clone(),
            result_type: self.result_type,
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
        // descr.py:478-493: compute result_flag from result_type + result_signed
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
}

impl CallDescr for SimpleCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    /// descr.py:537-538: is_result_signed() → result_flag == FLAG_SIGNED
    fn is_result_signed(&self) -> bool {
        self.result_flag == ArrayFlag::Signed
    }
    fn result_size(&self) -> usize {
        self.result_size
    }
    fn get_extra_info(&self) -> &EffectInfo {
        &self.effect
    }
}

/// resume.py:1124-1132: callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR).
/// Returns a CallDescr for CALL_I(func, size) → int (raw pointer).
pub fn make_raw_malloc_calldescr() -> DescrRef {
    use std::sync::Arc;
    static NEXT_IDX: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x4000_0000);
    let idx = NEXT_IDX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let effect = EffectInfo {
        oopspecindex: OopSpecIndex::RawMallocVarsizeChar,
        ..EffectInfo::default()
    };
    // Raw malloc returns unsigned pointer (not signed)
    Arc::new(SimpleCallDescr::new(
        idx,
        vec![crate::Type::Int],
        crate::Type::Int,
        false,
        8,
        effect,
    ))
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
    vector_info: std::cell::UnsafeCell<Vec<AccumVectorInfo>>,
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
            vector_info: std::cell::UnsafeCell::new(Vec::new()),
        }
    }

    pub fn finish(index: u32, fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        SimpleFailDescr {
            index,
            fail_index,
            fail_arg_types,
            is_finish: true,
            trace_id: 0,
            vector_info: std::cell::UnsafeCell::new(Vec::new()),
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
    fn attach_vector_info(&self, info: AccumVectorInfo) {
        unsafe { &mut *self.vector_info.get() }.push(info);
    }
    fn vector_info(&self) -> Vec<AccumVectorInfo> {
        unsafe { &mut *self.vector_info.get() }.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        // Parity with test_fficall.py: CallReleaseGil opcodes for all return types
        let int_op = OpCode::call_release_gil_for_type(Type::Int);
        assert_eq!(int_op, OpCode::CallReleaseGilI);

        let float_op = OpCode::call_release_gil_for_type(Type::Float);
        assert_eq!(float_op, OpCode::CallReleaseGilF);

        let ref_op = OpCode::call_release_gil_for_type(Type::Ref);
        assert_eq!(ref_op, OpCode::CallReleaseGilR);

        let void_op = OpCode::call_release_gil_for_type(Type::Void);
        assert_eq!(void_op, OpCode::CallReleaseGilN);
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
