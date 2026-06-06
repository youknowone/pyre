//! pypy/objspace/std/mapdict.py
//!
//! Mapdict provides per-instance dict and weakref slots for hasdict /
//! weakrefable types. PyPy stores these inside the mapdict map's "dict"
//! and "weakref" SPECIAL slots; pyre keeps thread-local side tables
//! keyed by object address because pyre has no mapdict.
//!
//! The names below mirror PyPy: `MapdictDictSupport.getdict` →
//! `_obj_getdict`, `MapdictWeakrefSupport.setweakref` →
//! `_mapdict_setweakref`, etc.

use crate::PyError;
use pyre_object::PyObjectRef;

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

// ── attribute shapes (mapdict.py:32-42, 720-732) ──────────────────────

/// mapdict.py:35 `NUM_DIGITS = 4`.
pub const NUM_DIGITS: u32 = 4;
/// mapdict.py:36 `NUM_DIGITS_POW2 = 1 << NUM_DIGITS`.
///
/// Note: upstream multiplies by `NUM_DIGITS_POW2` rather than shifting by
/// `NUM_DIGITS` so the result is known non-negative (mapdict.py:37-38).
pub const NUM_DIGITS_POW2: usize = 1 << NUM_DIGITS;

/// mapdict.py:40-42 — the maximum number of attributes stored in mapdict
/// (afterwards just use a dict).
pub const LIMIT_MAP_ATTRIBUTES: usize = 80;

/// mapdict.py:30 `ALLOW_UNBOXING_INTS = LONG_BIT == 64`. pyre targets
/// 64-bit, so int unboxing is permitted.
pub const ALLOW_UNBOXING_INTS: bool = usize::BITS == 64;

/// mapdict.py:720 `DICT = 0` — attrkind for instance `__dict__` entries.
pub const DICT: u16 = 0;
/// mapdict.py:721 `SPECIAL = 1` — attrkind for the `"dict"` / `"weakref"`
/// special slots.
pub const SPECIAL: u16 = 1;
/// mapdict.py:722 `INVALID = 2` — sentinel attrkind for empty
/// `MapAttrCache` slots.
pub const INVALID: u16 = 2;
/// mapdict.py:723 `SLOTS_STARTING_FROM = 3` — attrkind for `__slots__`
/// slot `i` is `SLOTS_STARTING_FROM + i`.
pub const SLOTS_STARTING_FROM: u16 = 3;

/// mapdict.py:725-732 `attrkind_name`.
///
/// ```python
/// def attrkind_name(attrkind):
///     if attrkind == DICT:
///         return "DICT"
///     if attrkind == SPECIAL:
///         return "SPECIAL"
///     if attrkind == INVALID:
///         return "INVALID"
///     return str(attrkind)
/// ```
pub fn attrkind_name(attrkind: u16) -> String {
    match attrkind {
        DICT => "DICT".to_string(),
        SPECIAL => "SPECIAL".to_string(),
        INVALID => "INVALID".to_string(),
        other => other.to_string(),
    }
}

// ── map nodes (mapdict.py:45-529) ─────────────────────────────────────
//
// AbstractAttribute hierarchy. PyPy uses a class hierarchy
// (AbstractAttribute → Terminator{Dict,NoDict,Devolved} / PlainAttribute,
// mapdict.py:45/304/420). The Rust port models all map nodes with a single
// `MapNode` enum (the enum-vs-hierarchy adaptation explicitly permitted by
// the parity rules) so the recurring `isinstance(self, PlainAttribute)`
// chain tests (mapdict.py:118-122,186) become a cheap `match`, and the three
// Terminator subclasses become a `TerminatorKind` field (mapdict.py:357-418).
//
// Map nodes are interned and shared per type (PyPy interns transitions so the
// same attribute added from the same map yields the same child map); they are
// never freed, so a node is referenced by a raw `*const MapNode` (`MapRef`)
// and the few mutable fields (`ever_mutated`, `allow_unboxing`) use Cell.
//
// AbstractAttribute.space (mapdict.py:47) is omitted: pyre's object space is
// ambient (global helpers) rather than an object threaded through nodes,
// matching the rest of pyre-interpreter.

/// `2 ** methodcachesizeexp` is the MapAttrCache size (pypyoption.py:230,
/// default 11).
pub const METHODCACHESIZEEXP: u32 = 11;

/// A shared, interned, immortal map node (mapdict.py AbstractAttribute).
pub type MapRef = *const MapNode;

/// mapdict.py:357/376/382 — the three Terminator subclasses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TerminatorKind {
    /// `DictTerminator` (mapdict.py:357).
    Dict,
    /// `NoDictTerminator` (mapdict.py:376).
    NoDict,
    /// `DevolvedDictTerminator` (mapdict.py:382).
    Devolved,
}

/// mapdict.py:304 `Terminator(AbstractAttribute)` — the root of a map chain.
pub struct Terminator {
    /// mapdict.py:307 `w_cls`.
    pub w_cls: PyObjectRef,
    /// mapdict.py:308 `allow_unboxing` (quasi-immutable; cleared when an
    /// attribute that was unboxed is reassigned a differently-typed value,
    /// mapdict.py:685).
    pub allow_unboxing: Cell<bool>,
    /// Which Terminator subclass this is.
    pub kind: TerminatorKind,
    /// mapdict.py:360 `DictTerminator.devolved_dict_terminator` (null unless
    /// `kind == Dict`).
    pub devolved_dict_terminator: Cell<MapRef>,
    /// mapdict.py:47 `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`. PyPy lazily inits it
    /// to `{}`; the eager empty map here is equivalent.
    pub cache_attrs: RefCell<HashMap<(String, u16), *const CachedAttributeHolder>>,
    /// mapdict.py:53 `AbstractAttribute.terminator` — a Terminator points to
    /// itself.
    pub terminator: MapRef,
}

/// The unbox type of an `UnboxedPlainAttribute` (mapdict.py:534/547,
/// `space.IntObjectCls` / `space.FloatObjectCls`). pyre exposes no per-type
/// impl-class object, so the int/float distinction is captured by this enum
/// and resolved through `is_int`/`is_float` (the enum-vs-class-object
/// adaptation, parallel to the `MapNode` enum-vs-hierarchy adaptation above).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum UnboxType {
    /// `space.IntObjectCls`.
    Int,
    /// `space.FloatObjectCls`.
    Float,
}

/// mapdict.py:532-563 `UnboxedPlainAttribute`'s extra fields. PyPy models the
/// unboxed attribute as a `PlainAttribute` subclass; pyre keeps it an optional
/// field on `PlainAttribute` so all the shared `PlainAttribute` machinery
/// (find_map_attr / add_attr / reorder / cache) applies unchanged
/// (enum-vs-hierarchy adaptation). When present, the value is stored unboxed in
/// a shared longlong list at `PlainAttribute.storageindex`; `listindex` is the
/// position in that list.
pub struct UnboxedExtra {
    /// mapdict.py:547 `typ`.
    pub typ: UnboxType,
    /// mapdict.py:563 `listindex` — position in the shared longlong list.
    pub listindex: usize,
    /// mapdict.py:544/561 `firstunwrapped` — this attribute is the first
    /// unboxed one to occupy its storage slot (so it allocates the list).
    pub firstunwrapped: bool,
}

/// mapdict.py:420 `PlainAttribute(AbstractAttribute)` — one stored attribute.
pub struct PlainAttribute {
    /// mapdict.py:425 `name` (utf8-encoded).
    pub name: String,
    /// mapdict.py:426 `attrkind`.
    pub attrkind: u16,
    /// mapdict.py:427 `storageindex` (= `back.storage_needed()`).
    pub storageindex: usize,
    /// mapdict.py:428 `_num_attributes` (= `back.num_attributes() + 1`).
    pub num_attributes: usize,
    /// mapdict.py:429 `back`.
    pub back: MapRef,
    /// mapdict.py:430 `ever_mutated` (quasi-immutable).
    pub ever_mutated: Cell<bool>,
    /// mapdict.py:431 `order`.
    pub order: usize,
    /// mapdict.py:47 `AbstractAttribute.cache_attrs` — the per-node transition
    /// cache `(name, attrkind) -> CachedAttributeHolder`.
    pub cache_attrs: RefCell<HashMap<(String, u16), *const CachedAttributeHolder>>,
    /// mapdict.py:53 `AbstractAttribute.terminator` (= `back.terminator`).
    pub terminator: MapRef,
    /// `Some` for an `UnboxedPlainAttribute` (mapdict.py:532); `None` for a
    /// plain boxed attribute.
    pub unboxed: Option<UnboxedExtra>,
}

/// mapdict.py:45 `AbstractAttribute` plus its two concrete subclasses.
pub enum MapNode {
    Terminator(Terminator),
    Plain(PlainAttribute),
}

fn intern_node(node: MapNode) -> MapRef {
    // Immortal: leak the box so the shared node lives for the process.
    Box::into_raw(Box::new(node)) as MapRef
}

/// mapdict.py:306-310 `Terminator.__init__`. `AbstractAttribute.__init__(space,
/// self)` makes the terminator its own `terminator`.
pub fn new_terminator(w_cls: PyObjectRef, kind: TerminatorKind) -> MapRef {
    let raw = Box::into_raw(Box::new(MapNode::Terminator(Terminator {
        w_cls,
        allow_unboxing: Cell::new(true),
        kind,
        devolved_dict_terminator: Cell::new(std::ptr::null()),
        cache_attrs: RefCell::new(HashMap::new()),
        terminator: std::ptr::null(),
    })));
    // Patch the self-referential terminator now that the address is known
    // (still uniquely owned here, before it is shared).
    unsafe {
        if let MapNode::Terminator(t) = &mut *raw {
            t.terminator = raw as MapRef;
        }
    }
    raw as MapRef
}

/// mapdict.py:358-360 `DictTerminator.__init__` — also builds the paired
/// `DevolvedDictTerminator` and links it.
pub fn new_dict_terminator(w_cls: PyObjectRef) -> MapRef {
    let devolved = new_terminator(w_cls, TerminatorKind::Devolved);
    let t = new_terminator(w_cls, TerminatorKind::Dict);
    unsafe {
        (*t).as_terminator().devolved_dict_terminator.set(devolved);
    }
    t
}

/// mapdict.py:423-431 `PlainAttribute.__init__`.
///
/// # Safety
/// `back` must point to a live (immortal) map node.
pub unsafe fn new_plain_attribute(
    name: String,
    attrkind: u16,
    back: MapRef,
    order: usize,
) -> MapRef {
    let back_node = unsafe { &*back };
    intern_node(MapNode::Plain(PlainAttribute {
        name,
        attrkind,
        storageindex: back_node.storage_needed(),
        num_attributes: back_node.num_attributes() + 1,
        back,
        ever_mutated: Cell::new(false),
        order,
        cache_attrs: RefCell::new(HashMap::new()),
        terminator: back_node.terminator(),
        unboxed: None,
    }))
}

/// mapdict.py:534-563 `UnboxedPlainAttribute.__init__` +
/// `_compute_storageindex_listindex`.
///
/// Unlike `PlainAttribute.__init__`, the storage index is shared with the
/// nearest `UnboxedPlainAttribute` ancestor (all unboxed attributes pack their
/// longlong values into one shared list); only the first unboxed attribute in a
/// slot (`firstunwrapped`) allocates a fresh slot.
///
/// # Safety
/// `back` must point to a live (immortal) map node.
pub unsafe fn new_unboxed_plain_attribute(
    name: String,
    attrkind: u16,
    back: MapRef,
    order: usize,
    typ: UnboxType,
) -> MapRef {
    let back_node = unsafe { &*back };
    // _compute_storageindex_listindex (mapdict.py:549-563): walk up looking for
    // an existing UnboxedPlainAttribute to share a storage slot with.
    let mut attr = back;
    let mut shared = None;
    loop {
        match unsafe { &*attr } {
            MapNode::Plain(p) => {
                if let Some(u) = &p.unboxed {
                    shared = Some((p.storageindex, u.listindex + 1));
                    break;
                }
                attr = p.back;
            }
            MapNode::Terminator(_) => break,
        }
    }
    let (storageindex, listindex, firstunwrapped) = match shared {
        Some((storageindex, listindex)) => (storageindex, listindex, false),
        None => (back_node.storage_needed(), 0, true),
    };
    intern_node(MapNode::Plain(PlainAttribute {
        name,
        attrkind,
        storageindex,
        num_attributes: back_node.num_attributes() + 1,
        back,
        ever_mutated: Cell::new(false),
        order,
        cache_attrs: RefCell::new(HashMap::new()),
        terminator: back_node.terminator(),
        unboxed: Some(UnboxedExtra {
            typ,
            listindex,
            firstunwrapped,
        }),
    }))
}

impl MapNode {
    /// Borrow the inner Terminator (panics on PlainAttribute).
    pub fn as_terminator(&self) -> &Terminator {
        match self {
            MapNode::Terminator(t) => t,
            MapNode::Plain(_) => panic!("as_terminator on PlainAttribute"),
        }
    }

    /// Borrow the inner PlainAttribute (panics on Terminator).
    pub fn as_plain(&self) -> &PlainAttribute {
        match self {
            MapNode::Plain(p) => p,
            MapNode::Terminator(_) => panic!("as_plain on Terminator"),
        }
    }

    /// `isinstance(self, PlainAttribute)`.
    pub fn is_plain(&self) -> bool {
        matches!(self, MapNode::Plain(_))
    }

    /// mapdict.py:53,141 `AbstractAttribute.terminator` / `get_terminator`.
    pub fn terminator(&self) -> MapRef {
        match self {
            MapNode::Terminator(t) => t.terminator,
            MapNode::Plain(p) => p.terminator,
        }
    }

    /// mapdict.py:327 (Terminator) / 478 (PlainAttribute) / 565-568
    /// (UnboxedPlainAttribute) `storage_needed`.
    pub fn storage_needed(&self) -> usize {
        match self {
            MapNode::Terminator(_) => 0,
            // mapdict.py:565-568: an unboxed attribute only adds a slot when it
            // is the first unboxed one in its slot (`firstunwrapped`);
            // otherwise it packs into the slot the prior unboxed attribute
            // already reserved, so its size is `back.storage_needed()`.
            MapNode::Plain(p) => match &p.unboxed {
                Some(u) => {
                    if u.firstunwrapped {
                        p.storageindex + 1
                    } else {
                        unsafe { (*p.back).storage_needed() }
                    }
                }
                None => p.storageindex + 1,
            },
        }
    }

    /// mapdict.py:330 (Terminator) / 481 (PlainAttribute) `num_attributes`.
    pub fn num_attributes(&self) -> usize {
        match self {
            MapNode::Terminator(_) => 0,
            MapNode::Plain(p) => p.num_attributes,
        }
    }

    /// mapdict.py:47 `AbstractAttribute.cache_attrs`.
    pub fn cache_attrs(&self) -> &RefCell<HashMap<(String, u16), *const CachedAttributeHolder>> {
        match self {
            MapNode::Terminator(t) => &t.cache_attrs,
            MapNode::Plain(p) => &p.cache_attrs,
        }
    }
}

/// mapdict.py:140,487-490 `AbstractAttribute.search`.
///
/// # Safety
/// `node` and its `back` chain must point to live map nodes.
pub unsafe fn node_search(node: MapRef, attrtype: u16) -> Option<MapRef> {
    match unsafe { &*node } {
        MapNode::Terminator(_) => None,
        MapNode::Plain(p) => {
            if p.attrkind == attrtype {
                Some(node)
            } else {
                unsafe { node_search(p.back, attrtype) }
            }
        }
    }
}

/// mapdict.py:118-122 `AbstractAttribute._find_map_attr` — the uncached chain
/// walk.
///
/// # Safety
/// `node` and its `back` chain must point to live map nodes.
pub unsafe fn find_map_attr_chain(mut node: MapRef, name: &str, attrkind: u16) -> Option<MapRef> {
    while let MapNode::Plain(p) = unsafe { &*node } {
        if attrkind == p.attrkind && name == p.name {
            return Some(node);
        }
        node = p.back;
    }
    None
}

/// mapdict.py:694-715 `MapAttrCache` — the per-space attribute lookup cache
/// behind `find_map_attr`. A null `MapRef` slot means "empty"/"not found".
pub struct MapAttrCache {
    attrs: Vec<MapRef>,
    names: Vec<Option<String>>,
    indexes: Vec<u16>,
    cached_attrs: Vec<MapRef>,
}

impl MapAttrCache {
    fn new() -> Self {
        let size = 1usize << METHODCACHESIZEEXP;
        MapAttrCache {
            attrs: vec![std::ptr::null(); size],
            names: vec![None; size],
            indexes: vec![INVALID; size],
            cached_attrs: vec![std::ptr::null(); size],
        }
    }

    /// mapdict.py:705-713 `clear`.
    pub fn clear(&mut self) {
        for slot in self.attrs.iter_mut() {
            *slot = std::ptr::null();
        }
        for slot in self.names.iter_mut() {
            *slot = None;
        }
        for slot in self.indexes.iter_mut() {
            *slot = INVALID;
        }
        for slot in self.cached_attrs.iter_mut() {
            *slot = std::ptr::null();
        }
    }
}

thread_local! {
    /// `space.fromcache(MapAttrCache)` (mapdict.py:80) — one cache per space;
    /// pyre runs one space per thread.
    static MAP_ATTR_CACHE: RefCell<MapAttrCache> = RefCell::new(MapAttrCache::new());
}

/// Deterministic hash of an attribute name. Only affects `MapAttrCache` bucket
/// distribution — `find_map_attr` always rechecks name+attrkind, so a
/// collision never changes the result. Stands in for
/// `objectmodel.compute_hash(name)` (mapdict.py:91).
fn compute_name_hash(name: &str) -> i64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in name.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h as i64
}

/// mapdict.py:86-117 `AbstractAttribute.find_map_attr` with the method cache
/// always enabled — fuses the dispatcher (mapdict.py:86) and
/// `_find_map_attr_cache` (mapdict.py:100, `@jit.dont_look_inside`). The
/// uncached walk is `find_map_attr_chain` (`_find_map_attr`); the JIT calls
/// that directly rather than tracing this cache path.
///
/// # Safety
/// `self_node` and its `back` chain must point to live map nodes.
pub unsafe fn find_map_attr(self_node: MapRef, name: &str, attrkind: u16) -> Option<MapRef> {
    const SHIFT2: u32 = u64::BITS - METHODCACHESIZEEXP;
    const SHIFT1: u32 = SHIFT2 - 5;
    // current_object_addr_as_int(self) (mapdict.py:88) — the map node address.
    let attrs_as_int = self_node as usize as i64;
    // unrolled hash computation for the 2-tuple (name, attrkind) (mapdict.py:90-95)
    let c1: i64 = 0x34_5678;
    let c2: i64 = 1_000_003;
    let hash_name = compute_name_hash(name);
    let hash_selector = c2.wrapping_mul(c2.wrapping_mul(c1) ^ hash_name) ^ (attrkind as i64);
    let product = attrs_as_int.wrapping_mul(hash_selector) as u64;
    let attr_hash = ((product ^ (product << SHIFT1)) >> SHIFT2) as usize;

    MAP_ATTR_CACHE.with(|cache| {
        {
            let cache = cache.borrow();
            if cache.attrs[attr_hash] == self_node
                && cache.names[attr_hash].as_deref() == Some(name)
                && cache.indexes[attr_hash] == attrkind
            {
                let cached = cache.cached_attrs[attr_hash];
                return if cached.is_null() { None } else { Some(cached) };
            }
        }
        let attr = unsafe { find_map_attr_chain(self_node, name, attrkind) };
        // Populate the cache, gated on `space._side_effects_ok()`
        // (mapdict.py:110). `_side_effects_ok` returns True except under reverse
        // debugging (not ported); the JIT does not trace this write because the
        // cache path is `@jit.dont_look_inside` and the JIT calls
        // `find_map_attr_chain` directly.
        if crate::baseobjspace::side_effects_ok() {
            let mut cache = cache.borrow_mut();
            cache.attrs[attr_hash] = self_node;
            cache.names[attr_hash] = Some(name.to_string());
            cache.indexes[attr_hash] = attrkind;
            cache.cached_attrs[attr_hash] = attr.unwrap_or(std::ptr::null());
        }
        attr
    })
}

// ── obj storage protocol (mapdict.py:904-964 MapdictStorageMixin) ──────
//
// The map-node layer reads and writes attribute values through this trait.
// PyPy mixes `MapdictStorageMixin` into the instance class; pyre's instance
// (W_InstanceObject) implements this trait instead (Slice 2). Storage holds
// `PyObjectRef`, so PyPy's `erase_item`/`unerase_item` (rerased boxing of a
// W_Root into the untyped storage list) are the identity here.

pub trait MapdictObject {
    /// The object's own `W_Root` reference. `DevolvedDictTerminator` reaches the
    /// instance dict through `obj.getdict(space)` (mapdict.py:386,393); pyre
    /// threads the same identity so `_obj_getdict` can key the dict by object
    /// address.
    fn _mapdict_self_ref(&self) -> PyObjectRef;
    /// mapdict.py:905-906 `_get_mapdict_map` (`jit.promote(self.map)`).
    fn _get_mapdict_map(&self) -> MapRef;
    /// mapdict.py:907-908 `_set_mapdict_map`.
    fn _set_mapdict_map(&mut self, map: MapRef);
    /// mapdict.py:914-916 `_mapdict_read_storage`.
    fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef;
    /// mapdict.py:918-919 `_mapdict_write_storage`.
    fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef);
    /// mapdict.py:921-924 `_mapdict_storage_length` (= `self.map.storage_needed()`).
    fn _mapdict_storage_length(&self) -> usize;
    /// mapdict.py:926-939 `_mapdict_pop_attribute`.
    fn _mapdict_pop_attribute(&mut self, map: MapRef);
    /// mapdict.py:942-959 `_set_mapdict_increase_storage1`.
    fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef);
}

/// rerased `erase_unboxed` / `unerase_unboxed` (mapdict.py:38-41) — the
/// unboxed longlong list lives in an otherwise-`PyObjectRef` storage slot.
/// pyre casts a boxed `Vec<i64>` into the slot. The list is heap-leaked here;
/// its GC-managed lifetime arrives with the varsize-instance storage in Slice
/// 2/3 (these nodes/objects are not yet wired into live attribute access).
fn erase_unboxed(list: Box<Vec<i64>>) -> PyObjectRef {
    Box::into_raw(list) as PyObjectRef
}

/// # Safety
/// `slot` must have been produced by `erase_unboxed` and still be live.
unsafe fn unerase_unboxed(slot: PyObjectRef) -> *mut Vec<i64> {
    slot as *mut Vec<i64>
}

/// mapdict.py:571-577 `UnboxedPlainAttribute._unbox`.
///
/// # Safety
/// `w_value` must be of the type named by `typ`.
unsafe fn unbox_value(typ: UnboxType, w_value: PyObjectRef) -> i64 {
    match typ {
        UnboxType::Int => unsafe { pyre_object::w_int_get_value(w_value) },
        // float2longlong (mapdict.py:577).
        UnboxType::Float => unsafe { pyre_object::w_float_get_value(w_value) }.to_bits() as i64,
    }
}

/// mapdict.py:579-584 `UnboxedPlainAttribute._box`.
fn box_value(typ: UnboxType, val: i64) -> PyObjectRef {
    match typ {
        UnboxType::Int => pyre_object::w_int_new(val),
        // longlong2float (mapdict.py:584).
        UnboxType::Float => pyre_object::w_float_new(f64::from_bits(val as u64)),
    }
}

/// `type(w_value) is space.IntObjectCls` (mapdict.py:574,615). `is_int` matches
/// `bool` too (bool subclasses int), but bool's impl class is not the int
/// class, so a bool must not be unboxed — `w_int_new` boxing would discard its
/// bool identity.
///
/// # Safety
/// `w_value` must point to a live object.
unsafe fn is_unboxable_int(w_value: PyObjectRef) -> bool {
    // Early-return control flow rather than `if c { false } else { x }`, which
    // can become a `!c && x` (a `bool_not`) that majit-translate cannot lower.
    if unsafe { pyre_object::is_bool(w_value) } {
        return false;
    }
    unsafe { pyre_object::is_int(w_value) }
}

/// mapdict.py:586-590 `UnboxedPlainAttribute._convert_to_boxed` — rebuild the
/// instance with boxed storage via `obj.copy()` (instance construction), which
/// needs W_InstanceObject and is deferred to Slice 2. Extracted to a
/// normal-returning function so callers stay simple control flow: an inline
/// diverging `else` desugars to an `assert`, which majit-translate cannot
/// lower.
///
/// # Safety
/// `obj` must implement the mapdict storage protocol.
unsafe fn convert_to_boxed<O: MapdictObject>(_obj: &O) {
    unimplemented!(
        "UnboxedPlainAttribute._convert_to_boxed (mapdict.py:586-590): needs obj.copy() (Slice 2)"
    );
}

/// The mutating side of `_convert_to_boxed` plus the subsequent `map.write`
/// (mapdict.py:620-627). Deferred to Slice 2 alongside `obj.copy()`.
///
/// # Safety
/// `obj` must implement the mapdict storage protocol.
unsafe fn convert_to_boxed_and_write<O: MapdictObject>(_obj: &mut O) {
    unimplemented!(
        "UnboxedPlainAttribute._direct_write type change (mapdict.py:620-627): needs obj.copy() (Slice 2)"
    );
}

/// `type(w_value) is self.typ` (mapdict.py:574,615).
///
/// # Safety
/// `w_value` must point to a live object.
unsafe fn value_has_unbox_type(typ: UnboxType, w_value: PyObjectRef) -> bool {
    match typ {
        UnboxType::Int => unsafe { is_unboxable_int(w_value) },
        UnboxType::Float => unsafe { pyre_object::is_float(w_value) },
    }
}

/// mapdict.py:437-444 `PlainAttribute._direct_read` / `_prim_direct_read` /
/// `_pure_direct_read` (identical bodies; the `@jit.elidable` `_pure_direct_read`
/// variant is applied when the read is JIT-wired). `unerase_item` is identity.
/// For an `UnboxedPlainAttribute` this is mapdict.py:592-612.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn plain_direct_read<O: MapdictObject>(attr: MapRef, obj: &O) -> PyObjectRef {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        // mapdict.py:443 — boxed value straight out of the slot.
        None => obj._mapdict_read_storage(p.storageindex),
        Some(u) => {
            // _prim_direct_read (mapdict.py:600-601): box the longlong at
            // (storageindex, listindex).
            let slot = obj._mapdict_read_storage(p.storageindex);
            let raw = unsafe {
                let list: &Vec<i64> = &*unerase_unboxed(slot);
                list[u.listindex]
            };
            let w_res = box_value(u.typ, raw);
            // _direct_read (mapdict.py:592-598): if some other instance of this
            // class turned out not to be type-stable, unboxing was switched off
            // for the whole terminator; convert this instance back to boxed.
            let term = unsafe { (*p.terminator).as_terminator() };
            if term.allow_unboxing.get() {
                // type-stable; nothing to do
            } else {
                // unboxing was switched off for this class because some other
                // instance was not type-stable; convert this one back to boxed.
                unsafe { convert_to_boxed(obj) };
            }
            w_res
        }
    }
}

/// mapdict.py:446-447 `PlainAttribute._direct_write`. `erase_item` is identity.
/// For an `UnboxedPlainAttribute` this is mapdict.py:614-628.
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
pub unsafe fn plain_direct_write<O: MapdictObject>(
    attr: MapRef,
    obj: &mut O,
    w_value: PyObjectRef,
) {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        None => {
            let storageindex = p.storageindex;
            obj._mapdict_write_storage(storageindex, w_value);
        }
        Some(u) => {
            if unsafe { value_has_unbox_type(u.typ, w_value) } {
                // mapdict.py:615-619 — same type: update the longlong in place.
                let val = unsafe { unbox_value(u.typ, w_value) };
                let slot = obj._mapdict_read_storage(p.storageindex);
                unsafe {
                    let list: &mut Vec<i64> = &mut *unerase_unboxed(slot);
                    list[u.listindex] = val;
                }
            } else {
                // mapdict.py:620-627 — type change. Freeze unboxing for the
                // terminator, then convert the instance to boxed storage and
                // rewrite. The conversion uses `obj.copy()` — deferred to
                // Slice 2.
                unsafe { (*p.terminator).as_terminator() }
                    .allow_unboxing
                    .set(false);
                unsafe { convert_to_boxed_and_write(obj) };
            }
        }
    }
}

/// mapdict.py:312-313 `Terminator._read_terminator` and its
/// `DevolvedDictTerminator` override (mapdict.py:387-391). Returns the value or
/// `None` when the attribute is absent.
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn terminator_read<O: MapdictObject>(
    term: MapRef,
    obj: &O,
    name: &str,
    attrkind: u16,
) -> Option<PyObjectRef> {
    let t = unsafe { (*term).as_terminator() };
    match t.kind {
        TerminatorKind::Devolved if attrkind == DICT => {
            // DevolvedDictTerminator._read_terminator (mapdict.py:383-387):
            // `w_dict = obj.getdict(space); return space.finditem_str(w_dict, name)`.
            // `finditem_str` yields NULL (here `None`) when the key is absent.
            let w_dict = _obj_getdict(obj._mapdict_self_ref());
            unsafe { pyre_object::w_dict_getitem_str(w_dict, name) }
        }
        // Terminator / DictTerminator / NoDictTerminator read nothing.
        _ => None,
    }
}

/// mapdict.py:55-66 `AbstractAttribute.read`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_read<O: MapdictObject>(
    self_node: MapRef,
    obj: &O,
    name: &str,
    attrkind: u16,
) -> Option<PyObjectRef> {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        // The `jit.isconstant(attr) and jit.isconstant(obj) and not
        // attr.ever_mutated` guard selects `_pure_direct_read`; both variants
        // have the same body (mapdict.py:60-65).
        Some(attr) => Some(unsafe { plain_direct_read(attr, obj) }),
        None => unsafe { terminator_read((*self_node).terminator(), obj, name, attrkind) },
    }
}

// ── write path (mapdict.py:68-258, 312-321, 668-691) ──────────────────

/// mapdict.py:668-691 `CachedAttributeHolder` — caches the child map produced
/// by adding `(name, attrkind)` to a parent map, so transitions are shared.
/// Interned/immortal like the map nodes it holds.
pub struct CachedAttributeHolder {
    /// mapdict.py:670 `order` (= number of prior children of `back`).
    pub order: usize,
    /// mapdict.py:675 `attr` (quasi-immutable).
    pub attr: Cell<MapRef>,
    /// mapdict.py:676 `typ` (quasi-immutable unbox type, `None` = boxed).
    pub typ: Cell<Option<UnboxType>>,
}

/// mapdict.py:670-676 `CachedAttributeHolder.__init__`.
///
/// # Safety
/// `back` must point to a live map node.
unsafe fn new_cached_attribute_holder(
    name: String,
    attrkind: u16,
    back: MapRef,
    unbox_type: Option<UnboxType>,
) -> *const CachedAttributeHolder {
    let order = unsafe { (*back).cache_attrs() }.borrow().len();
    let attr = match unbox_type {
        None => unsafe { new_plain_attribute(name, attrkind, back, order) },
        Some(typ) => unsafe { new_unboxed_plain_attribute(name, attrkind, back, order, typ) },
    };
    Box::into_raw(Box::new(CachedAttributeHolder {
        order,
        attr: Cell::new(attr),
        typ: Cell::new(unbox_type),
    }))
}

/// mapdict.py:679-691 `CachedAttributeHolder.pick_attr`.
///
/// # Safety
/// `holder` must point to a live `CachedAttributeHolder`.
unsafe fn holder_pick_attr(
    holder: *const CachedAttributeHolder,
    unbox_type: Option<UnboxType>,
) -> MapRef {
    let h = unsafe { &*holder };
    let typ = h.typ.get();
    if typ.is_none() || typ == unbox_type {
        return h.attr.get();
    }
    // The cached attribute was unboxed but the new value has a different type;
    // invalidate unboxing for this terminator and re-box (mapdict.py:682-690).
    h.typ.set(None);
    let attr = h.attr.get();
    let p = unsafe { (*attr).as_plain() };
    unsafe { (*p.terminator).as_terminator() }
        .allow_unboxing
        .set(false);
    let new_attr = unsafe { new_plain_attribute(p.name.clone(), p.attrkind, p.back, h.order) };
    h.attr.set(new_attr);
    new_attr
}

/// mapdict.py:149-156 `AbstractAttribute._get_new_attr`.
///
/// # Safety
/// `self_node` must point to a live map node.
unsafe fn get_new_attr(
    self_node: MapRef,
    name: &str,
    attrkind: u16,
    unbox_type: Option<UnboxType>,
) -> *const CachedAttributeHolder {
    let key = (name.to_string(), attrkind);
    if let Some(&holder) = unsafe { (*self_node).cache_attrs() }.borrow().get(&key) {
        return holder;
    }
    let holder =
        unsafe { new_cached_attribute_holder(name.to_string(), attrkind, self_node, unbox_type) };
    unsafe { (*self_node).cache_attrs() }
        .borrow_mut()
        .insert(key, holder);
    holder
}

/// mapdict.py:170-193 `AbstractAttribute._find_branch_to_move_into`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
unsafe fn find_branch_to_move_into(
    self_node: MapRef,
    name: &str,
    attrkind: u16,
    unbox_type: Option<UnboxType>,
) -> (usize, *const CachedAttributeHolder) {
    let mut current_order = usize::MAX; // sys.maxint
    let mut number_to_readd = 0usize;
    let mut current = self_node;
    let key = (name.to_string(), attrkind);
    loop {
        let holder = unsafe { (*current).cache_attrs() }
            .borrow()
            .get(&key)
            .copied();
        let reached_top = match holder {
            None => true,
            Some(h) => (unsafe { (*h).order }) > current_order,
        };
        if reached_top {
            // didn't find it anywhere yet; if we reached a non-PlainAttribute
            // (the terminator), just add it at the top attribute
            if unsafe { (*current).is_plain() } {
                // keep walking up
            } else {
                return (0, unsafe {
                    get_new_attr(self_node, name, attrkind, unbox_type)
                });
            }
        } else {
            return (number_to_readd, holder.unwrap());
        }
        // not found here, try the parent
        number_to_readd += 1;
        let p = unsafe { (*current).as_plain() };
        current_order = p.order;
        current = p.back;
    }
}

/// mapdict.py:195-202 `AbstractAttribute._pick_unbox_type`.
///
/// Returns the unbox type when the terminator allows unboxing and the value is
/// an unboxable int (only on 64-bit, `ALLOW_UNBOXING_INTS`) or float.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes; `w_value` to a live
/// object.
unsafe fn pick_unbox_type(self_node: MapRef, w_value: PyObjectRef) -> Option<UnboxType> {
    let term = unsafe { (*(*self_node).terminator()).as_terminator() };
    if term.allow_unboxing.get() {
        if ALLOW_UNBOXING_INTS && unsafe { is_unboxable_int(w_value) } {
            return Some(UnboxType::Int);
        } else if unsafe { pyre_object::is_float(w_value) } {
            return Some(UnboxType::Float);
        }
    }
    None
}

/// mapdict.py:449-459 `PlainAttribute._switch_map_and_write_increase_storage1`
/// and the `UnboxedPlainAttribute` override (mapdict.py:629-646).
///
/// # Safety
/// `attr` must point to a live `PlainAttribute` map node.
unsafe fn switch_map_and_write_increase_storage1<O: MapdictObject>(
    attr: MapRef,
    obj: &mut O,
    w_value: PyObjectRef,
) {
    let p = unsafe { (*attr).as_plain() };
    match &p.unboxed {
        None => {
            // mapdict.py:449-459
            if unsafe { (*attr).storage_needed() } > obj._mapdict_storage_length() {
                // erase_item is identity
                obj._set_mapdict_increase_storage1(attr, w_value);
                return;
            }
            // change the map first, then the storage
            obj._set_mapdict_map(attr);
            unsafe { plain_direct_write(attr, obj, w_value) };
        }
        Some(u) => {
            // mapdict.py:629-646
            let val = unsafe { unbox_value(u.typ, w_value) };
            if u.firstunwrapped {
                // a fresh longlong list of one element occupies a new slot
                let unboxed = erase_unboxed(Box::new(vec![val]));
                if unsafe { (*attr).storage_needed() } > obj._mapdict_storage_length() {
                    obj._set_mapdict_increase_storage1(attr, unboxed);
                    return;
                }
                obj._set_mapdict_map(attr);
                obj._mapdict_write_storage(p.storageindex, unboxed);
            } else {
                // append to the existing shared list (a fresh list, matching
                // PyPy's `unboxed + [val]`)
                let slot = obj._mapdict_read_storage(p.storageindex);
                let mut new_list = unsafe {
                    let list: &Vec<i64> = &*unerase_unboxed(slot);
                    list.clone()
                };
                obj._set_mapdict_map(attr);
                debug_assert_eq!(new_list.len(), u.listindex);
                new_list.push(val);
                obj._mapdict_write_storage(p.storageindex, erase_unboxed(Box::new(new_list)));
            }
        }
    }
}

/// mapdict.py:204-258 `AbstractAttribute._reorder_and_add` — the complicated
/// case where a lower-order ancestor already has the attribute, so the
/// attributes passed on the way up must be saved and re-added in order.
///
/// PyPy stores the to-be-readded `(map, value)` pairs in a flat erased array
/// indexed by `stack_index`; the Rust port uses a `Vec<(MapRef, PyObjectRef)>`
/// with push/pop (same LIFO behaviour). `erase_item`/`unerase_item` and
/// `erase_map`/`unerase_map` are the identity / the typed tuple here.
///
/// # Safety
/// `self_node`/`attr` and their chains must point to live map nodes.
unsafe fn reorder_and_add<O: MapdictObject>(
    mut self_node: MapRef,
    obj: &mut O,
    mut number_to_readd: usize,
    mut attr: MapRef,
    mut w_value: PyObjectRef,
) {
    let mut stack: Vec<(MapRef, PyObjectRef)> =
        Vec::with_capacity(unsafe { (*self_node).num_attributes() } * 2);
    loop {
        // we found the attributes further up, need to save the previous
        // values of the attributes we passed
        if number_to_readd != 0 {
            let mut current = self_node;
            for _ in 0..number_to_readd {
                // current is a PlainAttribute
                let w_self_value = unsafe { plain_direct_read(current, obj) };
                stack.push((current, w_self_value));
                current = unsafe { (*current).as_plain() }.back;
                obj._mapdict_pop_attribute(current);
            }
        }
        unsafe { switch_map_and_write_increase_storage1(attr, obj, w_value) };

        // readd the current top of the stack
        match stack.pop() {
            None => return,
            Some((next_map, next_value)) => {
                w_value = next_value;
                let (name, attrkind) = {
                    let p = unsafe { (*next_map).as_plain() };
                    (p.name.clone(), p.attrkind)
                };
                self_node = obj._get_mapdict_map();
                let unbox_type = unsafe { pick_unbox_type(self_node, w_value) };
                let (n, holder) = unsafe {
                    find_branch_to_move_into(self_node, name.as_str(), attrkind, unbox_type)
                };
                number_to_readd = n;
                attr = unsafe { holder_pick_attr(holder, unbox_type) };
            }
        }
    }
}

/// mapdict.py:157-169 `AbstractAttribute.add_attr`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn add_attr<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &str,
    attrkind: u16,
    w_value: PyObjectRef,
) {
    let unbox_type = unsafe { pick_unbox_type(self_node, w_value) };
    let (number_to_readd, holder) =
        unsafe { find_branch_to_move_into(self_node, name, attrkind, unbox_type) };
    let attr = unsafe { holder_pick_attr(holder, unbox_type) };
    if number_to_readd == 0 {
        unsafe { switch_map_and_write_increase_storage1(attr, obj, w_value) };
    } else {
        // the complicated reorder case
        unsafe { reorder_and_add(self_node, obj, number_to_readd, attr, w_value) };
    }
}

/// mapdict.py:312-321 `Terminator._write_terminator` plus the
/// `NoDictTerminator` override (mapdict.py:377-380).
///
/// # Safety
/// `term` must point to a live Terminator map node.
unsafe fn write_terminator<O: MapdictObject>(
    term: MapRef,
    obj: &mut O,
    name: &str,
    attrkind: u16,
    w_value: PyObjectRef,
) -> bool {
    let kind = unsafe { (*term).as_terminator() }.kind;
    match kind {
        // NoDictTerminator: object without __dict__ rejects DICT writes.
        TerminatorKind::NoDict if attrkind == DICT => return false,
        TerminatorKind::Devolved if attrkind == DICT => {
            // DevolvedDictTerminator._write_terminator (mapdict.py:390-395):
            // `w_dict = obj.getdict(space); space.setitem_str(w_dict, name, w_value); return True`.
            let w_dict = _obj_getdict(obj._mapdict_self_ref());
            unsafe { pyre_object::w_dict_setitem_str(w_dict, name, w_value) };
            return true;
        }
        _ => {}
    }
    let map = obj._get_mapdict_map();
    unsafe { add_attr(map, obj, name, attrkind, w_value) };
    if attrkind == DICT
        && unsafe { (*obj._get_mapdict_map()).num_attributes() } >= LIMIT_MAP_ATTRIBUTES
    {
        // mapdict.py:317-320 switches the instance __dict__ from the lazy
        // MapDictStrategy view to an eager UnicodeDictStrategy and devolves the
        // map to DevolvedDictTerminator via materialize_str_dict
        // (`switch_to_text_strategy`, mapdict.py:1148-1155). pyre cannot port
        // this yet: there is no MapDictStrategy (StrategyKind has no Map variant)
        // and no materialize_str_dict / _make_devolved / _set_mapdict_storage_and_map,
        // so `_obj_getdict` returns a separate eager dict rather than a view of
        // the mapdict storage. Switching a strategy here would strand these
        // attributes in mapdict storage with no devolve. Deferred to Slice 8
        // together with the DevolvedDictTerminator devolve transition.
    }
    true
}

/// mapdict.py:68-75 `AbstractAttribute.write`.
///
/// # Safety
/// `self_node` and its chain must point to live map nodes.
pub unsafe fn node_write<O: MapdictObject>(
    self_node: MapRef,
    obj: &mut O,
    name: &str,
    attrkind: u16,
    w_value: PyObjectRef,
) -> bool {
    match unsafe { find_map_attr(self_node, name, attrkind) } {
        None => unsafe {
            write_terminator((*self_node).terminator(), obj, name, attrkind, w_value)
        },
        Some(attr) => {
            let p = unsafe { (*attr).as_plain() };
            if p.ever_mutated.get() {
                // already mutated
            } else {
                p.ever_mutated.set(true);
            }
            unsafe { plain_direct_write(attr, obj, w_value) };
            true
        }
    }
}

thread_local! {
    /// objspace/std/mapdict.py:830 — MapdictDictSupport stores the
    /// instance dict in the "dict" SPECIAL slot of the mapdict map.
    /// pyre keeps a side table of address → W_DictObject because there
    /// is no mapdict; semantically this is the same backing store.
    pub static INSTANCE_DICT: RefCell<HashMap<usize, PyObjectRef>> =
        RefCell::new(HashMap::new());
}

thread_local! {
    /// objspace/std/mapdict.py:780-797 MapdictWeakrefSupport stores the
    /// lifeline in the "weakref" SPECIAL slot of the mapdict map. pyre
    /// keeps a side table because there is no mapdict; semantically
    /// this is the same per-instance lifeline storage.
    pub static WEAKREF_TABLE: RefCell<HashMap<usize, PyObjectRef>> =
        RefCell::new(HashMap::new());
}

// ── MapdictDictSupport ────────────────────────────────────────────────

/// objspace/std/mapdict.py:826-840 _obj_getdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_getdict(self, space):
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     w_dict = self._get_mapdict_map().read(self, "dict", SPECIAL)
///     if w_dict is not None:
///         assert isinstance(w_dict, W_DictMultiObject)
///         return w_dict
///
///     strategy = space.fromcache(MapDictStrategy)
///     storage = strategy.erase(self)
///     w_dict = W_DictObject(space, strategy, storage)
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
///     return w_dict
/// ```
pub fn _obj_getdict(self_ref: PyObjectRef) -> PyObjectRef {
    let existing = INSTANCE_DICT.with(|table| table.borrow().get(&(self_ref as usize)).copied());
    if let Some(w_dict) = existing {
        return w_dict;
    }
    // PyPy stores this in the mapdict "dict" SPECIAL slot. pyre's temporary
    // mapdict adapter is an address-keyed side table; keep the holder
    // GC-managed so a user-held old __dict__ remains traceable after
    // _obj_setdict replaces the side-table entry.
    let w_dict = pyre_object::w_dict_new();
    INSTANCE_DICT.with(|table| {
        table.borrow_mut().insert(self_ref as usize, w_dict);
    });
    w_dict
}

fn current_owner_key(key: usize) -> usize {
    pyre_object::gc_hook::try_gc_current_object_address(key as *mut u8) as usize
}

/// Walk roots held by pyre's temporary mapdict side tables.
///
/// PyPy stores the instance dict and weakref lifeline in mapdict SPECIAL slots,
/// so the translated GC sees them as ordinary object fields. pyre keeps the
/// same logical data in address-keyed side tables until mapdict is ported into
/// the object layout; expose the value slots here so the backend GC can update
/// them when nursery objects move.
pub fn walk_mapdict_roots(mut visitor: impl FnMut(&mut PyObjectRef)) {
    let dict_values = INSTANCE_DICT.with(|table| {
        table
            .borrow()
            .iter()
            .map(|(&key, &dict)| (key, dict))
            .collect::<Vec<_>>()
    });
    // SAFETY: do not hold the RefCell borrow while invoking callbacks. The
    // visitor and w_dict_walk_entries_mut may re-enter mapdict/dict APIs.
    for (key, mut dict) in dict_values {
        visitor(&mut dict);
        let new_key = current_owner_key(key);
        INSTANCE_DICT.with(|table| {
            let mut table = table.borrow_mut();
            if new_key == key {
                if let Some(slot) = table.get_mut(&key) {
                    *slot = dict;
                }
            } else if table.remove(&key).is_some() {
                table.insert(new_key, dict);
            }
        });
        unsafe {
            pyre_object::w_dict_walk_entries_mut(dict, |slot| {
                visitor(slot);
            });
        }
    }
    let weakref_values = WEAKREF_TABLE.with(|table| {
        table
            .borrow()
            .iter()
            .map(|(&key, &value)| (key, value))
            .collect::<Vec<_>>()
    });
    for (key, mut value) in weakref_values {
        visitor(&mut value);
        let new_key = current_owner_key(key);
        WEAKREF_TABLE.with(|table| {
            let mut table = table.borrow_mut();
            if new_key == key {
                if let Some(slot) = table.get_mut(&key) {
                    *slot = value;
                }
            } else if table.remove(&key).is_some() {
                table.insert(new_key, value);
            }
        });
    }
}

/// objspace/std/mapdict.py:842-860 _obj_setdict.
///
/// ```python
/// @objectmodel.dont_inline
/// def _obj_setdict(self, space, w_dict):
///     from pypy.interpreter.error import oefmt
///     terminator = self._get_mapdict_map().terminator
///     assert isinstance(terminator, DictTerminator) or isinstance(terminator, DevolvedDictTerminator)
///     if not space.isinstance_w(w_dict, space.w_dict):
///         raise oefmt(space.w_TypeError, "setting dictionary to a non-dict")
///     assert isinstance(w_dict, W_DictMultiObject)
///     w_olddict = self.getdict(space)
///     ...
///     flag = self._get_mapdict_map().write(self, "dict", SPECIAL, w_dict)
///     assert flag
/// ```
pub fn _obj_setdict(self_ref: PyObjectRef, w_dict: PyObjectRef) -> Result<(), PyError> {
    if !unsafe { pyre_object::is_dict(w_dict) } {
        return Err(PyError::type_error(
            "setting dictionary to a non-dict".to_string(),
        ));
    }
    INSTANCE_DICT.with(|table| {
        table.borrow_mut().insert(self_ref as usize, w_dict);
    });
    Ok(())
}

// ── MapdictWeakrefSupport ─────────────────────────────────────────────

/// objspace/std/mapdict.py:780-787 MapdictWeakrefSupport.getweakref.
///
/// ```python
/// def getweakref(self):
///     from pypy.module._weakref.interp__weakref import WeakrefLifeline
///     lifeline = self._get_mapdict_map().read(self, "weakref", SPECIAL)
///     if lifeline is None:
///         return None
///     assert isinstance(lifeline, WeakrefLifeline)
///     return lifeline
/// ```
pub fn getweakref(self_ref: PyObjectRef) -> Option<PyObjectRef> {
    WEAKREF_TABLE.with(|table| table.borrow().get(&(self_ref as usize)).copied())
}

/// objspace/std/mapdict.py:789-793 MapdictWeakrefSupport.setweakref.
///
/// ```python
/// def setweakref(self, space, weakreflifeline):
///     from pypy.module._weakref.interp__weakref import WeakrefLifeline
///     assert isinstance(weakreflifeline, WeakrefLifeline)
///     self._get_mapdict_map().write(self, "weakref", SPECIAL, weakreflifeline)
/// ```
pub fn setweakref(self_ref: PyObjectRef, weakreflifeline: PyObjectRef) {
    WEAKREF_TABLE.with(|table| {
        table
            .borrow_mut()
            .insert(self_ref as usize, weakreflifeline);
    });
}

/// objspace/std/mapdict.py:795-797 MapdictWeakrefSupport.delweakref.
///
/// ```python
/// def delweakref(self):
///     self._get_mapdict_map().write(self, "weakref", SPECIAL, None)
/// ```
pub fn delweakref(self_ref: PyObjectRef) {
    WEAKREF_TABLE.with(|table| {
        table.borrow_mut().remove(&(self_ref as usize));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    // A map chain is `terminator <- "a"(DICT) <- "b"(DICT)`. The w_cls is a
    // null placeholder: the node layer never dereferences it.
    unsafe fn build_chain() -> (MapRef, MapRef, MapRef) {
        let term = new_dict_terminator(std::ptr::null_mut());
        let a = unsafe { new_plain_attribute("a".to_string(), DICT, term, 0) };
        let b = unsafe { new_plain_attribute("b".to_string(), DICT, a, 1) };
        (term, a, b)
    }

    #[test]
    fn terminator_is_its_own_terminator_with_zero_storage() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            assert_eq!((*term).terminator(), term);
            assert_eq!((*term).storage_needed(), 0);
            assert_eq!((*term).num_attributes(), 0);
            assert_eq!((*term).as_terminator().kind, TerminatorKind::Dict);
            assert!(
                !(*term)
                    .as_terminator()
                    .devolved_dict_terminator
                    .get()
                    .is_null()
            );
        }
    }

    #[test]
    fn plain_attribute_increments_storage_and_keeps_terminator() {
        unsafe {
            let (term, a, b) = build_chain();
            assert_eq!((*a).as_plain().storageindex, 0);
            assert_eq!((*a).storage_needed(), 1);
            assert_eq!((*a).num_attributes(), 1);
            assert_eq!((*b).as_plain().storageindex, 1);
            assert_eq!((*b).storage_needed(), 2);
            assert_eq!((*b).num_attributes(), 2);
            assert_eq!((*a).terminator(), term);
            assert_eq!((*b).terminator(), term);
            assert_eq!((*b).as_plain().back, a);
        }
    }

    #[test]
    fn find_map_attr_chain_walks_back_pointers() {
        unsafe {
            let (_term, a, b) = build_chain();
            assert_eq!(find_map_attr_chain(b, "a", DICT), Some(a));
            assert_eq!(find_map_attr_chain(b, "b", DICT), Some(b));
            assert_eq!(find_map_attr_chain(b, "c", DICT), None);
            // attrkind namespaces are distinct: "a" exists only under DICT.
            assert_eq!(find_map_attr_chain(b, "a", SPECIAL), None);
        }
    }

    #[test]
    fn find_map_attr_cached_matches_uncached_on_hit_and_miss() {
        unsafe {
            MAP_ATTR_CACHE.with(|c| c.borrow_mut().clear());
            let (_term, a, b) = build_chain();
            // first call populates the cache, second hits it
            assert_eq!(find_map_attr(b, "a", DICT), Some(a));
            assert_eq!(find_map_attr(b, "a", DICT), Some(a));
            assert_eq!(find_map_attr(b, "b", DICT), Some(b));
            assert_eq!(find_map_attr(b, "missing", DICT), None);
            assert_eq!(find_map_attr(b, "missing", DICT), None);
        }
    }

    #[test]
    fn search_returns_topmost_match_for_attrkind() {
        unsafe {
            let (term, _a, b) = build_chain();
            // both "a" and "b" are DICT; search from b returns b (topmost)
            assert_eq!(node_search(b, DICT), Some(b));
            assert_eq!(node_search(b, SPECIAL), None);
            assert_eq!(node_search(term, DICT), None);
        }
    }

    // A minimal MapdictObject for read-path tests. Storage holds sentinel
    // pointers that are never dereferenced.
    struct MockObj {
        map: MapRef,
        storage: Vec<PyObjectRef>,
    }

    impl MapdictObject for MockObj {
        fn _mapdict_self_ref(&self) -> PyObjectRef {
            self as *const Self as PyObjectRef
        }
        fn _get_mapdict_map(&self) -> MapRef {
            self.map
        }
        fn _set_mapdict_map(&mut self, map: MapRef) {
            self.map = map;
        }
        fn _mapdict_read_storage(&self, storageindex: usize) -> PyObjectRef {
            self.storage[storageindex]
        }
        fn _mapdict_write_storage(&mut self, storageindex: usize, value: PyObjectRef) {
            self.storage[storageindex] = value;
        }
        fn _mapdict_storage_length(&self) -> usize {
            unsafe { (*self.map).storage_needed() }
        }
        fn _mapdict_pop_attribute(&mut self, map: MapRef) {
            // mapdict.py:926-939. `current_map` is the PlainAttribute being
            // popped; `map` is its parent.
            // `current_map` is the PlainAttribute being popped; `map` is its
            // parent. The unboxed-non-firstunwrapped slot to shrink is computed
            // as a `match` on `firstunwrapped` rather than `!firstunwrapped` so
            // the source walker can lower it.
            let current_map = self.map;
            let unboxed_slot: Option<(usize, usize)> = unsafe {
                match &(*current_map).as_plain().unboxed {
                    Some(u) => match u.firstunwrapped {
                        true => None,
                        false => Some(((*current_map).as_plain().storageindex, u.listindex)),
                    },
                    None => None,
                }
            };
            match unboxed_slot {
                // mapdict.py:931-934: drop the last entry of the shared longlong
                // list (the slot itself stays).
                Some((storageindex, listindex)) => {
                    let slot = self._mapdict_read_storage(storageindex);
                    let new_list: Vec<i64> = unsafe {
                        let list: &Vec<i64> = &*unerase_unboxed(slot);
                        list[..listindex].to_vec()
                    };
                    self._mapdict_write_storage(storageindex, erase_unboxed(Box::new(new_list)));
                }
                // mapdict.py:935-938: truncate storage to the parent map's size.
                None => {
                    let storage_needed = unsafe { (*map).storage_needed() };
                    self.storage.truncate(storage_needed);
                }
            }
            self.map = map;
        }
        fn _set_mapdict_increase_storage1(&mut self, map: MapRef, value: PyObjectRef) {
            // grow storage by one, append value (mapdict.py:942-959)
            self.storage.push(value);
            self.map = map;
        }
    }

    fn sentinel(n: usize) -> PyObjectRef {
        n as PyObjectRef
    }

    // A DictTerminator whose `allow_unboxing` is off, so writes take the boxed
    // PlainAttribute path and never type-inspect the (sentinel) value.
    unsafe fn boxed_dict_terminator() -> MapRef {
        let term = new_dict_terminator(std::ptr::null_mut());
        unsafe { (*term).as_terminator() }.allow_unboxing.set(false);
        term
    }

    #[test]
    fn node_read_returns_stored_value_by_storageindex() {
        unsafe {
            let (_term, _a, b) = build_chain();
            // map b: "a"@storageindex 0, "b"@storageindex 1
            let obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            assert_eq!(node_read(b, &obj, "a", DICT), Some(sentinel(0xa)));
            assert_eq!(node_read(b, &obj, "b", DICT), Some(sentinel(0xb)));
            // absent attribute falls through to the (Dict) terminator → None
            assert_eq!(node_read(b, &obj, "missing", DICT), None);
            assert_eq!(node_read(b, &obj, "a", SPECIAL), None);
            assert_eq!(obj._mapdict_storage_length(), 2);
        }
    }

    #[test]
    fn plain_direct_write_then_read_roundtrips() {
        unsafe {
            let (_term, a, b) = build_chain();
            let mut obj = MockObj {
                map: b,
                storage: vec![sentinel(0xa), sentinel(0xb)],
            };
            plain_direct_write(a, &mut obj, sentinel(0x111));
            assert_eq!(node_read(b, &obj, "a", DICT), Some(sentinel(0x111)));
            assert_eq!(node_read(b, &obj, "b", DICT), Some(sentinel(0xb)));
        }
    }

    #[test]
    fn devolved_terminator_read_write_routes_through_obj_getdict() {
        // mapdict.py:383-395 — a DevolvedDictTerminator reads and writes the
        // DICT attrkind through the instance dict (`_obj_getdict`), not the
        // map storage.  The branch is wired but not yet reachable in
        // production (no MapDictStrategy devolve), so exercise it directly:
        // root a MockObj at the paired devolved terminator and confirm both
        // node_write and node_read go through `_obj_getdict`.
        unsafe {
            let dict_term = new_dict_terminator(std::ptr::null_mut());
            let devolved = (*dict_term).as_terminator().devolved_dict_terminator.get();
            assert!(!devolved.is_null());
            let mut obj = MockObj {
                map: devolved,
                storage: vec![],
            };
            // Write lands in the instance dict, leaving map storage untouched.
            assert!(node_write(devolved, &mut obj, "dk", DICT, sentinel(0x55)));
            assert!(obj.storage.is_empty());
            // The value is in `_obj_getdict`'s dict and node_read reads it back.
            let w_dict = _obj_getdict(obj._mapdict_self_ref());
            assert_eq!(
                pyre_object::w_dict_getitem_str(w_dict, "dk"),
                Some(sentinel(0x55))
            );
            assert_eq!(node_read(devolved, &obj, "dk", DICT), Some(sentinel(0x55)));
            // Absent key and non-DICT attrkind read nothing.
            assert_eq!(node_read(devolved, &obj, "absent_dk", DICT), None);
            assert_eq!(node_read(devolved, &obj, "dk", SPECIAL), None);
        }
    }

    #[test]
    fn add_attr_via_write_grows_map_and_storage() {
        unsafe {
            // start empty: a DictTerminator, no storage
            let term = boxed_dict_terminator();
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            // write two fresh attributes; each takes the common
            // (number_to_readd == 0) append path
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, "x", DICT, sentinel(0x1)));
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, "y", DICT, sentinel(0x2)));

            assert_eq!(obj.storage.len(), 2);
            assert_eq!(unsafe { (*obj.map).num_attributes() }, 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, "x", DICT), Some(sentinel(0x1)));
            assert_eq!(node_read(m, &obj, "y", DICT), Some(sentinel(0x2)));
            assert_eq!(node_read(m, &obj, "z", DICT), None);

            // overwrite an existing attribute → direct write, no growth
            let m = obj._get_mapdict_map();
            assert!(node_write(m, &mut obj, "x", DICT, sentinel(0x9)));
            assert_eq!(obj.storage.len(), 2);
            let m = obj._get_mapdict_map();
            assert_eq!(node_read(m, &obj, "x", DICT), Some(sentinel(0x9)));
        }
    }

    #[test]
    fn add_attr_interns_shared_transition() {
        unsafe {
            // two independent objects adding the same attribute from the same
            // map must converge on the same child map (interned transition)
            let term = boxed_dict_terminator();
            let mut o1 = MockObj {
                map: term,
                storage: vec![],
            };
            let mut o2 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, "p", DICT, sentinel(1));
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, "p", DICT, sentinel(2));
            assert_eq!(o1.map, o2.map);
        }
    }

    #[test]
    fn out_of_order_insert_reorders_to_canonical_map() {
        unsafe {
            let term = boxed_dict_terminator();
            // o1 establishes the canonical insertion order a, b
            let mut o1 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, "a", DICT, sentinel(0xa1));
            let m = o1._get_mapdict_map();
            node_write(m, &mut o1, "b", DICT, sentinel(0xb1));

            // o2 inserts b first, then a — adding "a" must trigger
            // _reorder_and_add (a lower-order ancestor already has "a") and
            // converge on the same canonical map as o1
            let mut o2 = MockObj {
                map: term,
                storage: vec![],
            };
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, "b", DICT, sentinel(0xb2));
            let m = o2._get_mapdict_map();
            node_write(m, &mut o2, "a", DICT, sentinel(0xa2));

            // values preserved through the reorder
            let m = o2._get_mapdict_map();
            assert_eq!(node_read(m, &o2, "a", DICT), Some(sentinel(0xa2)));
            assert_eq!(node_read(m, &o2, "b", DICT), Some(sentinel(0xb2)));
            assert_eq!(o2.storage.len(), 2);
            // reordered to the canonical (insertion-ordered) map shared with o1
            assert_eq!(o2.map, o1.map);
        }
    }

    // ── UnboxedPlainAttribute (mapdict.py:532-665) ────────────────────

    #[test]
    fn unboxed_int_attribute_stores_and_reads() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            // an int written to an unboxing-allowed map becomes an
            // UnboxedPlainAttribute
            assert!(node_write(
                m,
                &mut obj,
                "x",
                DICT,
                pyre_object::w_int_new(42)
            ));
            let p = (*obj.map).as_plain();
            assert!(p.unboxed.is_some());
            assert!(p.unboxed.as_ref().unwrap().firstunwrapped);
            assert_eq!(p.unboxed.as_ref().unwrap().typ, UnboxType::Int);
            // a single storage slot holds the erased longlong list (not a box)
            assert_eq!(obj.storage.len(), 1);
            // reading boxes the longlong back into an int of the same value
            let m = obj._get_mapdict_map();
            let r = node_read(m, &obj, "x", DICT).unwrap();
            assert!(pyre_object::is_int(r));
            assert_eq!(pyre_object::w_int_get_value(r), 42);
        }
    }

    #[test]
    fn two_unboxed_ints_share_one_storage_slot() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "x", DICT, pyre_object::w_int_new(10));
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "y", DICT, pyre_object::w_int_new(20));
            // both pack into a single shared longlong list (storageindex 0,
            // listindex 0 and 1) — storage does not grow for the second one
            assert_eq!(obj.storage.len(), 1);
            let p = (*obj.map).as_plain();
            assert_eq!(p.storageindex, 0);
            assert_eq!(p.unboxed.as_ref().unwrap().listindex, 1);
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, "x", DICT).unwrap()),
                10
            );
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, "y", DICT).unwrap()),
                20
            );
        }
    }

    #[test]
    fn unboxed_int_and_float_share_slot_with_correct_boxing() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "i", DICT, pyre_object::w_int_new(7));
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "f", DICT, pyre_object::w_float_new(2.5));
            // int and float pack into one longlong list (the float as its bits)
            assert_eq!(obj.storage.len(), 1);
            let m = obj._get_mapdict_map();
            let ri = node_read(m, &obj, "i", DICT).unwrap();
            let rf = node_read(m, &obj, "f", DICT).unwrap();
            // each is re-boxed to its own type
            assert!(pyre_object::is_int(ri));
            assert_eq!(pyre_object::w_int_get_value(ri), 7);
            assert!(pyre_object::is_float(rf));
            assert_eq!(pyre_object::w_float_get_value(rf), 2.5);
        }
    }

    #[test]
    fn unboxed_overwrite_same_type_updates_in_place() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "x", DICT, pyre_object::w_int_new(1));
            let map_after_first = obj.map;
            // a same-typed overwrite updates the longlong in place: no map
            // transition, no storage growth
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "x", DICT, pyre_object::w_int_new(2));
            assert_eq!(obj.map, map_after_first);
            assert_eq!(obj.storage.len(), 1);
            let m = obj._get_mapdict_map();
            assert_eq!(
                pyre_object::w_int_get_value(node_read(m, &obj, "x", DICT).unwrap()),
                2
            );
        }
    }

    #[test]
    fn unboxed_float_preserves_bits() {
        unsafe {
            let term = new_dict_terminator(std::ptr::null_mut());
            let mut obj = MockObj {
                map: term,
                storage: vec![],
            };
            let v = -3.141_592_653_589_793_f64;
            let m = obj._get_mapdict_map();
            node_write(m, &mut obj, "f", DICT, pyre_object::w_float_new(v));
            let m = obj._get_mapdict_map();
            let r = node_read(m, &obj, "f", DICT).unwrap();
            assert!(pyre_object::is_float(r));
            assert_eq!(pyre_object::w_float_get_value(r), v);
        }
    }
}
