//! `rpython/rtyper/lltypesystem/llmemory.py` — annotation types for
//! low-level memory addresses.
#![allow(non_snake_case)]

use std::cell::RefCell;

use crate::annotator::model::{KnownType, SomeObjectBase, SomeObjectTrait, SomeValue};
use crate::flowspace::model::ConstValue;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, _wref, ArrayType, GcKind, LowLevelType, cast_pointer, nullptr,
};

/// `class SomeAddress(SomeObject)` (llmemory.py:573-590).
/// Annotation for low-level Address values. `immutable = True`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeAddress {
    pub base: SomeObjectBase,
}

impl SomeAddress {
    pub fn new() -> Self {
        SomeAddress {
            base: SomeObjectBase::new(KnownType::Address, true),
        }
    }

    /// `def is_null_address(self)` (llmemory.py:579-580).
    /// `return self.is_immutable_constant() and not self.const`
    /// — true when the annotation carries a constant that is a falsy
    /// address value (i.e. NULL / fakeaddress(None)).
    pub fn is_null_address(&self) -> bool {
        if !self.is_immutable_constant() {
            return false;
        }
        match &self.base.const_box {
            Some(c) => c.value.is_null_address(),
            None => false,
        }
    }

    /// `def getattr(self, s_attr)` (llmemory.py:582-586).
    /// Returns the annotation for `addr.<access_type>` — the intermediate
    /// value used in `addr.signed[offset]` patterns.
    pub fn annotation_getattr(attr: &str) -> Option<SomeTypedAddressAccess> {
        supported_access_type(attr).map(SomeTypedAddressAccess::new)
    }

    /// `def bool(self)` (llmemory.py:588-589).
    /// `return s_Bool`
    pub fn annotation_bool() -> SomeValue {
        SomeValue::Bool(crate::annotator::model::SomeBool::new())
    }
}

impl Default for SomeAddress {
    fn default() -> Self {
        SomeAddress::new()
    }
}

impl SomeObjectTrait for SomeAddress {
    fn knowntype(&self) -> KnownType {
        KnownType::Address
    }
    fn immutable(&self) -> bool {
        true
    }
    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// llmemory.py:730-735
pub fn supported_access_type(name: &str) -> Option<LowLevelType> {
    match name {
        "signed" => Some(LowLevelType::Signed),
        "unsigned" => Some(LowLevelType::Unsigned),
        "char" => Some(LowLevelType::Char),
        "address" => Some(LowLevelType::Address),
        "float" => Some(LowLevelType::Float),
        _ => None,
    }
}

/// `class SomeTypedAddressAccess(SomeObject)` (llmemory.py:596-605).
/// Annotation for the intermediate value in `addr.signed[offset]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeTypedAddressAccess {
    pub access_type: LowLevelType,
    pub base: SomeObjectBase,
}

impl SomeTypedAddressAccess {
    pub fn new(access_type: LowLevelType) -> Self {
        SomeTypedAddressAccess {
            access_type,
            base: SomeObjectBase::new(KnownType::Object, false),
        }
    }
}

impl SomeObjectTrait for SomeTypedAddressAccess {
    fn knowntype(&self) -> KnownType {
        KnownType::Object
    }
    fn immutable(&self) -> bool {
        false
    }
    fn is_constant(&self) -> bool {
        false
    }
    fn can_be_none(&self) -> bool {
        false
    }
}

/// `class Symbolic` (llmemory.py:11) → `class AddressOffset(Symbolic)`
/// (llmemory.py:19) and its subclasses. The runtime `ref` / `_raw_malloc`
/// / `raw_memcopy` methods operate on the `fakeaddress` simulator, which
/// pyre does not model; what flows through the annotator and rtyper is
/// the rtyping-level structure — variant identity, `known_nonneg`,
/// symbolic arithmetic, and `lltype() == Signed`. `GCHeaderOffset` /
/// `GCHeaderAntiOffset` (llmemory.py:341-386) carry a `gcheaderbuilder`,
/// which belongs to the GC transform pyre does not run, so they are
/// omitted here.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum AddressOffset {
    /// llmemory.py:58 `class ItemOffset(AddressOffset)`.
    ItemOffset { TYPE: LowLevelType, repeat: i64 },
    /// llmemory.py:186 `class FieldOffset(AddressOffset)`.
    FieldOffset { TYPE: LowLevelType, fldname: String },
    /// llmemory.py:225 `class CompositeOffset(AddressOffset)`.
    CompositeOffset(Vec<AddressOffset>),
    /// llmemory.py:278 `class ArrayItemsOffset(AddressOffset)`.
    ArrayItemsOffset(LowLevelType),
    /// llmemory.py:325 `class ArrayLengthOffset(AddressOffset)`.
    ArrayLengthOffset(LowLevelType),
}

impl AddressOffset {
    /// llmemory.py:25 `def lltype(self): return lltype.Signed`.
    pub fn lltype(&self) -> LowLevelType {
        LowLevelType::Signed
    }

    /// llmemory.py:48/77/195/255/286/333 `known_nonneg`.
    pub fn known_nonneg(&self) -> bool {
        match self {
            AddressOffset::ItemOffset { repeat, .. } => *repeat >= 0,
            AddressOffset::FieldOffset { .. } => true,
            AddressOffset::ArrayItemsOffset(_) => true,
            AddressOffset::ArrayLengthOffset(_) => true,
            AddressOffset::CompositeOffset(offsets) => offsets.iter().all(|o| o.known_nonneg()),
        }
    }

    /// llmemory.py:28 `def __add__(self, other): return
    /// CompositeOffset(self, other)`.
    pub fn add(self, other: AddressOffset) -> AddressOffset {
        AddressOffset::composite(vec![self, other])
    }

    /// llmemory.py:67-72 `ItemOffset.__mul__` (`__rmul__ = __mul__`).
    /// Non-`ItemOffset` returns `NotImplemented` upstream → `None` here.
    pub fn mul(self, other: i64) -> Option<AddressOffset> {
        match self {
            AddressOffset::ItemOffset { TYPE, repeat } => Some(AddressOffset::ItemOffset {
                TYPE,
                repeat: repeat * other,
            }),
            _ => None,
        }
    }

    /// llmemory.py:74-75 `ItemOffset.__neg__`; :250-253 `CompositeOffset
    /// .__neg__`. Only those two define `__neg__` upstream; for the other
    /// variants `-offset` raises `TypeError` (no `__neg__`), so `None`.
    pub fn neg(self) -> Option<AddressOffset> {
        match self {
            AddressOffset::ItemOffset { TYPE, repeat } => Some(AddressOffset::ItemOffset {
                TYPE,
                repeat: -repeat,
            }),
            // llmemory.py:250-253 `ofs = [-item for item in self.offsets];
            // ofs.reverse(); return CompositeOffset(*ofs)`. The list
            // comprehension negates every element — if any `-item` raises
            // (FieldOffset/ArrayItemsOffset/ArrayLengthOffset have no
            // `__neg__`), the whole `__neg__` raises. `collect::<Option<_>>`
            // short-circuits to `None` so a non-negatable element is not
            // silently dropped.
            AddressOffset::CompositeOffset(offsets) => {
                let mut ofs = offsets
                    .into_iter()
                    .map(|o| o.neg())
                    .collect::<Option<Vec<AddressOffset>>>()?;
                ofs.reverse();
                Some(AddressOffset::composite(ofs))
            }
            _ => None,
        }
    }

    /// llmemory.py:227-245 `CompositeOffset.__new__` — flatten nested
    /// composites, merge adjacent same-`TYPE` `ItemOffset`s, and collapse
    /// a single-element list to its sole offset.
    pub fn composite(offsets: Vec<AddressOffset>) -> AddressOffset {
        let mut lst: Vec<AddressOffset> = Vec::new();
        for item in offsets {
            match item {
                AddressOffset::CompositeOffset(inner) => lst.extend(inner),
                other => lst.push(other),
            }
        }
        let mut i = lst.len().wrapping_sub(2);
        while (i as isize) >= 0 {
            if let (
                AddressOffset::ItemOffset {
                    TYPE: t0,
                    repeat: r0,
                },
                AddressOffset::ItemOffset {
                    TYPE: t1,
                    repeat: r1,
                },
            ) = (&lst[i], &lst[i + 1])
            {
                if t0 == t1 {
                    let merged = AddressOffset::ItemOffset {
                        TYPE: t0.clone(),
                        repeat: r0 + r1,
                    };
                    lst.splice(i..i + 2, std::iter::once(merged));
                }
            }
            i = i.wrapping_sub(1);
        }
        if lst.len() == 1 {
            lst.pop().unwrap()
        } else {
            AddressOffset::CompositeOffset(lst)
        }
    }

    /// Concrete byte size for code emission. pyre interprets / JITs rather
    /// than emitting C, so a symbolic offset that reaches the assembler is
    /// resolved to its concrete size here — the role RPython's
    /// `rpython/jit/backend/llsupport/symbolic.py` plays for the real
    /// backends (`get_field_token` / `get_size` / `get_array_token`).
    /// Struct field offsets and struct sizes come from `layout`, which the
    /// codewriter (the owner of struct layouts) supplies; primitive item
    /// sizes and the standard length-prefixed array tokens are computed
    /// directly.
    pub fn byte_size(&self, layout: &dyn OffsetLayout) -> Result<i64, String> {
        match self {
            AddressOffset::ItemOffset { TYPE, repeat } => {
                let item = match primitive_byte_size(TYPE) {
                    Ok(sz) => sz,
                    // Non-primitive item (e.g. a struct array element) —
                    // `symbolic.get_size(TYPE)`.
                    Err(_) => match TYPE {
                        LowLevelType::Struct(st) => {
                            layout.struct_size(&st._name).ok_or_else(|| {
                                format!("no struct layout for {} (get_size)", st._name)
                            })?
                        }
                        other => return Err(format!("no byte size for item type {other:?}")),
                    },
                };
                Ok(item * repeat)
            }
            // `symbolic.get_field_token(STRUCT, fldname)[0]`.
            AddressOffset::FieldOffset { TYPE, fldname } => {
                let LowLevelType::Struct(st) = TYPE else {
                    return Err(format!("FieldOffset on non-struct {TYPE:?}"));
                };
                layout.field_offset(&st._name, fldname).ok_or_else(|| {
                    format!(
                        "no field offset for {}.{} (get_field_token)",
                        st._name, fldname
                    )
                })
            }
            AddressOffset::CompositeOffset(offsets) => {
                let mut total = 0;
                for o in offsets {
                    total += o.byte_size(layout)?;
                }
                Ok(total)
            }
            // `symbolic.get_array_token` for a standard length-prefixed
            // array: the length field sits at offset 0 and the items start
            // one word later.
            AddressOffset::ArrayItemsOffset(_) => Ok(WORD),
            AddressOffset::ArrayLengthOffset(_) => Ok(0),
        }
    }
}

/// Word size (length-field width / pointer width).
const WORD: i64 = std::mem::size_of::<usize>() as i64;

/// Resolver for the layout-dependent symbolic offsets, mirroring
/// `rpython/jit/backend/llsupport/symbolic.py`. The codewriter owns the
/// real struct layouts, so it implements this; defining the trait here
/// (rather than depending on the codewriter) keeps `llmemory` below the
/// `jit_codewriter` layer.
pub trait OffsetLayout {
    /// `symbolic.get_field_token(STRUCT, fldname)[0]` — byte offset of the
    /// field within the struct.
    fn field_offset(&self, struct_name: &str, fldname: &str) -> Option<i64>;
    /// `symbolic.get_size(STRUCT)` — total struct byte size.
    fn struct_size(&self, struct_name: &str) -> Option<i64>;
}

/// Byte size of a primitive `LowLevelType` (word = 8 bytes).
fn primitive_byte_size(ty: &LowLevelType) -> Result<i64, String> {
    match ty {
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::Address
        | LowLevelType::Float => Ok(8),
        LowLevelType::Char => Ok(1),
        other => Err(format!("no primitive byte size for {other:?}")),
    }
}

/// `llmemory._sizeof_none(TYPE)` (llmemory.py:391-393) — `ItemOffset(TYPE)`,
/// asserting the type is not varsize.
fn sizeof_none(ty: &LowLevelType) -> Result<AddressOffset, String> {
    if let LowLevelType::Struct(st) = ty
        && st._is_varsize()
    {
        return Err(format!("sizeof: {} is varsize, pass n", st._name));
    }
    Ok(AddressOffset::ItemOffset {
        TYPE: ty.clone(),
        repeat: 1,
    })
}

/// `llmemory.offsetof(TYPE, fldname)` (llmemory.py:426-429) —
/// `FieldOffset(TYPE, fldname)`, asserting the field exists.
pub fn offsetof(struct_ty: &LowLevelType, fldname: &str) -> Result<AddressOffset, String> {
    let LowLevelType::Struct(st) = struct_ty else {
        return Err(format!("offsetof: {struct_ty:?} is not a Struct"));
    };
    if st._flds.get(fldname).is_none() {
        return Err(format!("offsetof: {} has no field {fldname}", st._name));
    }
    Ok(AddressOffset::FieldOffset {
        TYPE: struct_ty.clone(),
        fldname: fldname.to_string(),
    })
}

/// `llmemory.itemoffsetof(TYPE, n=0)` (llmemory.py:438-442) —
/// `ArrayItemsOffset(TYPE)`, plus `ItemOffset(TYPE.OF) * n` when `n != 0`.
fn itemoffsetof(array_ty: &LowLevelType, of: &LowLevelType, n: i64) -> AddressOffset {
    let result = AddressOffset::ArrayItemsOffset(array_ty.clone());
    if n != 0 {
        let item = AddressOffset::ItemOffset {
            TYPE: of.clone(),
            repeat: n,
        };
        result.add(item)
    } else {
        result
    }
}

/// `llmemory._sizeof_int(TYPE, n)` (llmemory.py:400-405) — for a varsize
/// Struct, `offsetof(TYPE, arrayfld) + sizeof(ARRAY, n)`.
fn sizeof_int(struct_ty: &LowLevelType, n: i64) -> Result<AddressOffset, String> {
    let LowLevelType::Struct(st) = struct_ty else {
        return Err(format!("don't know how to take the size of {struct_ty:?}"));
    };
    let fldname = st
        ._arrayfld
        .clone()
        .ok_or_else(|| format!("don't know how to take the size of {}", st._name))?;
    let array_ty = st
        ._flds
        .get(&fldname)
        .ok_or_else(|| format!("sizeof: {} missing array field {fldname}", st._name))?
        .clone();
    Ok(offsetof(struct_ty, &fldname)?.add(sizeof_offset(&array_ty, Some(n))?))
}

/// `llmemory.extra_item_after_alloc(ARRAY)` (llmemory.py:407-409) —
/// `ARRAY._hints.get('extra_item_after_alloc', 0)`. STR's chars array sets it
/// to 1 for the trailing NUL slot (rstr.py:1226-1228).
fn extra_item_after_alloc(arr: &ArrayType) -> i64 {
    match arr._hints.get("extra_item_after_alloc") {
        Some(ConstValue::Int(n)) => *n,
        _ => 0,
    }
}

/// `llmemory.sizeof(TYPE, n=None)` (llmemory.py:411-426). `n=None` sizes a
/// fixed (non-varsize) type; an `Array` is sized as
/// `itemoffsetof(TYPE) + sizeof(TYPE.OF) * (n + extra_item_after_alloc(TYPE))`;
/// a varsize `Struct` defers to [`sizeof_int`].
fn sizeof_offset(ty: &LowLevelType, n: Option<i64>) -> Result<AddressOffset, String> {
    match n {
        None => sizeof_none(ty),
        Some(n) => match ty {
            LowLevelType::Array(arr) => {
                // llmemory.py:422 `n += extra_item_after_alloc(TYPE)`.
                let n = n + extra_item_after_alloc(arr);
                let item = AddressOffset::ItemOffset {
                    TYPE: arr.OF.clone(),
                    repeat: n,
                };
                Ok(itemoffsetof(ty, &arr.OF, 0).add(item))
            }
            _ => sizeof_int(ty, n),
        },
    }
}

/// `llmemory.sizeof(TYPE, n=None)` (llmemory.py:411-426). `inputconst(Signed,
/// sizeof(TYPE))` accepts the result because `AddressOffset.lltype() ==
/// Signed` (matching RPython's `typeOf(Symbolic) -> val.lltype()`).
pub fn sizeof(ty: &LowLevelType, n: Option<i64>) -> Result<ConstValue, String> {
    Ok(ConstValue::AddressOffset(sizeof_offset(ty, n)?))
}

/// `llmemory.dead_wref` (llmemory.py:887) — `_wref(None)._as_ptr()`, a
/// single prebuilt pointer to a dead low-level weakref.
///
/// `_ptr` equality respects container identity (lltype.py:1185-1201), so
/// the value is cached thread-locally to keep every reference to
/// `dead_wref` comparing equal. Thread-local matches the single-threaded
/// translation model (`_ptr` is not `Sync`).
pub fn dead_wref() -> _ptr {
    thread_local! {
        static DEAD_WREF: RefCell<Option<_ptr>> = const { RefCell::new(None) };
    }
    DEAD_WREF.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(_wref::new(None)._as_ptr());
        }
        cell.borrow().as_ref().unwrap().clone()
    })
}

/// `llmemory.weakref_create(ptarget)` (llmemory.py:818-824).
///
/// ```python
/// def weakref_create(ptarget):
///     PTRTYPE = lltype.typeOf(ptarget)
///     assert isinstance(PTRTYPE, lltype.Ptr)
///     assert PTRTYPE.TO._gckind == 'gc'
///     assert ptarget
///     return _wref(ptarget)._as_ptr()
/// ```
///
/// The `_ptr` argument type discharges `isinstance(PTRTYPE, Ptr)`. The
/// target is validated (gc + non-null), then held by a real `_wref`
/// container so [`weakref_deref`] can recover it.
pub fn weakref_create(ptarget: &_ptr) -> Result<_ptr, String> {
    if ptarget._togckind() != GcKind::Gc {
        return Err(format!(
            "weakref_create: target {:?} must be gc-kind",
            ptarget._TYPE
        ));
    }
    if !ptarget.nonzero() {
        return Err("weakref_create: target must be non-null".to_string());
    }
    Ok(_wref::new(Some(ptarget))._as_ptr())
}

/// `llmemory.weakref_deref(PTRTYPE, pwref)` (llmemory.py:835-843).
///
/// ```python
/// def weakref_deref(PTRTYPE, pwref):
///     assert isinstance(PTRTYPE, lltype.Ptr)
///     assert PTRTYPE.TO._gckind == 'gc'
///     assert lltype.typeOf(pwref) == WeakRefPtr
///     p = pwref._obj._dereference()
///     if p is None:
///         return lltype.nullptr(PTRTYPE.TO)
///     else:
///         return cast_any_ptr(PTRTYPE, p)
/// ```
///
/// The referents this port produces are concrete gc-struct pointers whose
/// `typeOf` already equals `PTRTYPE`, so `cast_any_ptr` reduces to the
/// identity branch; a mismatch falls through to [`cast_pointer`]
/// (`cast_opaque_ptr` is not reached for these referents).
pub fn weakref_deref(PTRTYPE: &LowLevelType, pwref: &_ptr) -> Result<_ptr, String> {
    let LowLevelType::Ptr(ptr_t) = PTRTYPE else {
        return Err(format!(
            "weakref_deref: arg 1 must be a Ptr type, got {PTRTYPE:?}"
        ));
    };
    if ptr_t._gckind() != GcKind::Gc {
        return Err(format!(
            "weakref_deref: arg 1 {PTRTYPE:?} must point to a gc container"
        ));
    }
    let _ptr_obj::Wref(wref) = pwref
        ._obj()
        .map_err(|_| "weakref_deref: arg 2 weakref is a delayed pointer".to_string())?
    else {
        return Err("weakref_deref: arg 2 must be a WeakRefPtr".to_string());
    };
    match wref._dereference() {
        None => nullptr(LowLevelType::from((**ptr_t).TO.clone())),
        Some(p) => {
            if LowLevelType::Ptr(Box::new(p._TYPE.clone())) == *PTRTYPE {
                Ok(p)
            } else {
                cast_pointer(ptr_t, &p)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(ty: LowLevelType, repeat: i64) -> AddressOffset {
        AddressOffset::ItemOffset { TYPE: ty, repeat }
    }

    #[test]
    fn lltype_is_signed_for_every_variant() {
        // llmemory.py:25-26 `def lltype(self): return lltype.Signed`.
        assert_eq!(item(LowLevelType::Signed, 1).lltype(), LowLevelType::Signed);
        assert_eq!(
            AddressOffset::FieldOffset {
                TYPE: LowLevelType::Signed,
                fldname: "x".into()
            }
            .lltype(),
            LowLevelType::Signed
        );
    }

    #[test]
    fn item_offset_known_nonneg_tracks_repeat_sign() {
        // llmemory.py:77-78 `return self.repeat >= 0`.
        assert!(item(LowLevelType::Signed, 3).known_nonneg());
        assert!(item(LowLevelType::Signed, 0).known_nonneg());
        assert!(!item(LowLevelType::Signed, -1).known_nonneg());
    }

    #[test]
    fn field_and_array_offsets_are_known_nonneg() {
        // llmemory.py:195/286/333 — FieldOffset/ArrayItemsOffset/
        // ArrayLengthOffset all `known_nonneg() -> True`.
        assert!(
            AddressOffset::FieldOffset {
                TYPE: LowLevelType::Signed,
                fldname: "f".into()
            }
            .known_nonneg()
        );
        assert!(AddressOffset::ArrayItemsOffset(LowLevelType::Signed).known_nonneg());
        assert!(AddressOffset::ArrayLengthOffset(LowLevelType::Signed).known_nonneg());
    }

    #[test]
    fn item_offset_mul_scales_repeat() {
        // llmemory.py:67-70 `ItemOffset.__mul__`.
        assert_eq!(
            item(LowLevelType::Signed, 2).mul(3),
            Some(item(LowLevelType::Signed, 6))
        );
        // Non-ItemOffset `__mul__` returns NotImplemented upstream → None.
        assert_eq!(
            AddressOffset::ArrayItemsOffset(LowLevelType::Signed).mul(3),
            None
        );
    }

    #[test]
    fn item_offset_neg_negates_repeat() {
        // llmemory.py:74-75 `ItemOffset.__neg__`.
        assert_eq!(
            item(LowLevelType::Signed, 4).neg(),
            Some(item(LowLevelType::Signed, -4))
        );
    }

    #[test]
    fn composite_flattens_nested_composites() {
        // llmemory.py:229-233 — nested CompositeOffset is spliced inline.
        let inner = AddressOffset::CompositeOffset(vec![
            AddressOffset::FieldOffset {
                TYPE: LowLevelType::Signed,
                fldname: "a".into(),
            },
            AddressOffset::ArrayItemsOffset(LowLevelType::Char),
        ]);
        let outer = AddressOffset::composite(vec![
            inner,
            AddressOffset::ArrayLengthOffset(LowLevelType::Char),
        ]);
        match outer {
            AddressOffset::CompositeOffset(offsets) => assert_eq!(offsets.len(), 3),
            other => panic!("expected CompositeOffset, got {other:?}"),
        }
    }

    #[test]
    fn composite_merges_adjacent_same_type_item_offsets() {
        // llmemory.py:234-239 — adjacent same-TYPE ItemOffsets merge; a
        // single resulting element collapses (llmemory.py:240-241).
        let merged = AddressOffset::composite(vec![
            item(LowLevelType::Signed, 2),
            item(LowLevelType::Signed, 3),
        ]);
        assert_eq!(merged, item(LowLevelType::Signed, 5));
    }

    #[test]
    fn composite_keeps_distinct_type_item_offsets_separate() {
        let composite = AddressOffset::composite(vec![
            item(LowLevelType::Signed, 2),
            item(LowLevelType::Char, 3),
        ]);
        assert_eq!(
            composite,
            AddressOffset::CompositeOffset(vec![
                item(LowLevelType::Signed, 2),
                item(LowLevelType::Char, 3),
            ])
        );
    }

    #[test]
    fn composite_neg_negates_and_reverses() {
        // llmemory.py:250-253 `CompositeOffset.__neg__`.
        let composite = AddressOffset::CompositeOffset(vec![
            item(LowLevelType::Signed, 2),
            item(LowLevelType::Char, 3),
        ]);
        assert_eq!(
            composite.neg(),
            Some(AddressOffset::CompositeOffset(vec![
                item(LowLevelType::Char, -3),
                item(LowLevelType::Signed, -2),
            ]))
        );
    }

    #[test]
    fn composite_neg_fails_when_an_element_is_not_negatable() {
        // llmemory.py:250 `[-item for item in self.offsets]` raises when an
        // element has no `__neg__` (FieldOffset here) — not silently dropped.
        let composite = AddressOffset::CompositeOffset(vec![
            item(LowLevelType::Signed, 2),
            AddressOffset::FieldOffset {
                TYPE: LowLevelType::Signed,
                fldname: "f".into(),
            },
        ]);
        assert_eq!(composite.neg(), None);
    }

    #[test]
    fn add_builds_composite_and_merges_when_compatible() {
        // llmemory.py:28-31 `__add__ -> CompositeOffset(self, other)`.
        assert_eq!(
            item(LowLevelType::Signed, 2).add(item(LowLevelType::Signed, 5)),
            item(LowLevelType::Signed, 7)
        );
    }

    #[test]
    fn composite_known_nonneg_requires_all_parts() {
        // llmemory.py:255-259.
        assert!(
            AddressOffset::CompositeOffset(vec![
                item(LowLevelType::Signed, 1),
                AddressOffset::FieldOffset {
                    TYPE: LowLevelType::Signed,
                    fldname: "f".into()
                },
            ])
            .known_nonneg()
        );
        assert!(
            !AddressOffset::CompositeOffset(vec![
                item(LowLevelType::Signed, 1),
                item(LowLevelType::Signed, -1),
            ])
            .known_nonneg()
        );
    }

    /// A layout source that knows no struct — used to exercise the
    /// layout-free offset kinds and the missing-layout error paths.
    struct NoLayout;
    impl OffsetLayout for NoLayout {
        fn field_offset(&self, _struct_name: &str, _fldname: &str) -> Option<i64> {
            None
        }
        fn struct_size(&self, _struct_name: &str) -> Option<i64> {
            None
        }
    }

    /// A layout source with one fixed answer for every query.
    struct FakeLayout {
        field: i64,
        size: i64,
    }
    impl OffsetLayout for FakeLayout {
        fn field_offset(&self, _struct_name: &str, _fldname: &str) -> Option<i64> {
            Some(self.field)
        }
        fn struct_size(&self, _struct_name: &str) -> Option<i64> {
            Some(self.size)
        }
    }

    fn struct_ty(name: &str) -> LowLevelType {
        use crate::translator::rtyper::lltypesystem::lltype::StructType;
        LowLevelType::Struct(Box::new(StructType::new(
            name,
            vec![("f".to_string(), LowLevelType::Signed)],
        )))
    }

    #[test]
    fn byte_size_resolves_primitives_and_sums_composites() {
        assert_eq!(item(LowLevelType::Signed, 3).byte_size(&NoLayout), Ok(24));
        assert_eq!(item(LowLevelType::Char, 4).byte_size(&NoLayout), Ok(4));
        let composite = AddressOffset::CompositeOffset(vec![
            item(LowLevelType::Signed, 1),
            item(LowLevelType::Char, 2),
        ]);
        assert_eq!(composite.byte_size(&NoLayout), Ok(10));
    }

    #[test]
    fn byte_size_resolves_array_tokens_without_layout() {
        // Standard length-prefixed array: items one word past the header,
        // length field at offset 0.
        assert_eq!(
            AddressOffset::ArrayItemsOffset(LowLevelType::Signed).byte_size(&NoLayout),
            Ok(WORD)
        );
        assert_eq!(
            AddressOffset::ArrayLengthOffset(LowLevelType::Signed).byte_size(&NoLayout),
            Ok(0)
        );
    }

    #[test]
    fn byte_size_field_and_struct_item_use_layout() {
        let s = struct_ty("S");
        let fo = AddressOffset::FieldOffset {
            TYPE: s.clone(),
            fldname: "f".into(),
        };
        // No layout → get_field_token / get_size unavailable.
        assert!(fo.byte_size(&NoLayout).is_err());
        assert!(item(s.clone(), 2).byte_size(&NoLayout).is_err());
        // With a layout, FieldOffset reads the field offset and a struct
        // ItemOffset reads get_size * repeat.
        let layout = FakeLayout {
            field: 16,
            size: 40,
        };
        assert_eq!(fo.byte_size(&layout), Ok(16));
        assert_eq!(item(s, 2).byte_size(&layout), Ok(80));
    }

    #[test]
    fn byte_size_field_offset_on_non_struct_errors() {
        assert!(
            AddressOffset::FieldOffset {
                TYPE: LowLevelType::Signed,
                fldname: "f".into()
            }
            .byte_size(&NoLayout)
            .is_err()
        );
    }

    #[test]
    fn sizeof_primitive_returns_unit_item_offset() {
        // llmemory.py:412 `sizeof(TYPE) -> ItemOffset(TYPE)`.
        assert_eq!(
            sizeof(&LowLevelType::Signed, None),
            Ok(ConstValue::AddressOffset(item(LowLevelType::Signed, 1)))
        );
    }

    #[test]
    fn sizeof_array_is_items_offset_plus_n_items() {
        use crate::translator::rtyper::lltypesystem::lltype::{ArrayType, FrozenDict, GcKind};
        // llmemory.py:421-423 `sizeof(ARRAY, n) -> itemoffsetof(ARRAY) +
        // sizeof(ARRAY.OF) * n`.
        let array_ty = LowLevelType::Array(Box::new(ArrayType {
            OF: LowLevelType::Signed,
            _hints: FrozenDict::from(Vec::new()),
            _gckind: GcKind::Gc,
        }));
        let expected = AddressOffset::CompositeOffset(vec![
            AddressOffset::ArrayItemsOffset(array_ty.clone()),
            item(LowLevelType::Signed, 3),
        ]);
        assert_eq!(
            sizeof(&array_ty, Some(3)),
            Ok(ConstValue::AddressOffset(expected))
        );
    }

    #[test]
    fn sizeof_array_adds_extra_item_after_alloc() {
        use crate::translator::rtyper::lltypesystem::lltype::ArrayType;
        // llmemory.py:422 `n += extra_item_after_alloc(TYPE)` — an array with
        // `extra_item_after_alloc=1` (e.g. `rstr.STR.chars`) sizes to n+1
        // items so the trailing NUL slot is reserved (rstr.py:1226-1228).
        let array_ty = LowLevelType::Array(Box::new(ArrayType::with_hints(
            LowLevelType::Char,
            vec![("extra_item_after_alloc".into(), ConstValue::Int(1))],
        )));
        let expected = AddressOffset::CompositeOffset(vec![
            AddressOffset::ArrayItemsOffset(array_ty.clone()),
            item(LowLevelType::Char, 4), // 3 requested + 1 extra
        ]);
        assert_eq!(
            sizeof(&array_ty, Some(3)),
            Ok(ConstValue::AddressOffset(expected))
        );
    }

    fn gc_opaque(name: &str) -> _ptr {
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, opaqueptr};
        opaqueptr(LowLevelType::Opaque(Box::new(OpaqueType::gc(name))), "t").unwrap()
    }

    #[test]
    fn weakref_create_on_gc_target_yields_nonzero_gc_weakref() {
        // llmemory.py:818-824 — `_wref(ptarget)._as_ptr()` for a gc target.
        let wref = weakref_create(&gc_opaque("GcThing")).unwrap();
        assert!(wref.nonzero());
        assert_eq!(wref._togckind(), GcKind::Gc);
    }

    #[test]
    fn weakref_create_rejects_null_target() {
        // llmemory.py:823 `assert ptarget`.
        use crate::translator::rtyper::lltypesystem::lltype::{OpaqueType, nullptr};
        let null_gc = nullptr(LowLevelType::Opaque(Box::new(OpaqueType::gc("GcThing")))).unwrap();
        assert!(weakref_create(&null_gc).is_err());
    }

    #[test]
    fn dead_wref_is_a_single_shared_value() {
        // llmemory.py:887 `dead_wref = _wref(None)._as_ptr()` — one prebuilt.
        let a = dead_wref();
        let b = dead_wref();
        assert!(a.nonzero());
        assert_eq!(a, b);
    }

    #[test]
    fn weakref_deref_recovers_the_created_target() {
        // llmemory.py:835-843 — `weakref_deref(PTRTYPE, weakref_create(p))` is p.
        let target = gc_opaque("GcThing");
        let ptrtype = LowLevelType::Ptr(Box::new(target._TYPE.clone()));
        let wref = weakref_create(&target).unwrap();
        let got = weakref_deref(&ptrtype, &wref).unwrap();
        assert_eq!(got, target);
    }

    #[test]
    fn weakref_deref_of_dead_wref_is_null() {
        // llmemory.py:840-841 — a dead wref dereferences to `nullptr(PTRTYPE.TO)`.
        let ptrtype = LowLevelType::Ptr(Box::new(gc_opaque("GcThing")._TYPE.clone()));
        let got = weakref_deref(&ptrtype, &dead_wref()).unwrap();
        assert!(!got.nonzero());
    }
}
