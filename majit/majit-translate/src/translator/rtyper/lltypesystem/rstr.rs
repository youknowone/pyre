//! RPython `rpython/rtyper/lltypesystem/rstr.py` — minimal slice for
//! the `Ptr(STR)` low-level type that `OBJECT_VTABLE.name`
//! (rclass.py:171) and `ClassesPBCRepr.rtype_getattr('__name__')`
//! (rpbc.py:975-981) consume.
//!
//! Upstream rstr.py weighs ~3000 LOC and houses every operation that
//! rtypes Python's `str` / `unicode` types (`StringRepr`,
//! `UnicodeRepr`, `LLHelpers`, `ll_strconcat`, ...). Pyre lands only
//! the data shape this commit needs:
//!
//! ```python
//! STR = GcForwardReference()
//! ...
//! STR.become(GcStruct('rpy_string',
//!                     ('hash', Signed),
//!                     ('chars', Array(Char, hints={'immutable': True,
//!                                     'extra_item_after_alloc': 1})),
//!                     adtmeths={...},
//!                     hints={'remove_hash': True}))
//! ```
//!
//! The full repr port (string operations, comparison, formatting,
//! adtmeths) is deferred to its own follow-on session.
//!
//! ## Why a separate module today
//!
//! Both `OBJECT_VTABLE.name` (rclass.py) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py) want
//! `Ptr(STR)`. Defining `STR` here keeps the dependency direction
//! `rclass / rpbc → rstr → lltype` clean and gives a natural home
//! for the eventual full `StringRepr` port.

use std::sync::LazyLock;

use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, ArrayType, ForwardReference, LowLevelType, LowLevelValue, MallocFlavor, Ptr,
    PtrTarget, StructType, malloc,
};

/// RPython `STR = GcForwardReference()` resolved via
/// `STR.become(GcStruct('rpy_string', ('hash', Signed), ('chars',
/// Array(Char, ...)), hints={'remove_hash': True}))`
/// (rstr.py:32 + rstr.py:1226-1237).
///
/// Pyre lands the data shape — the `adtmeths` table (mallocstr,
/// emptystrfun, copy_string_contents, ll_strhash, ll_length, ll_find,
/// ll_rfind) is deferred. `OBJECT_VTABLE.name`'s `Ptr(STR)` only
/// needs the structural shape today; concrete consumers that emit
/// `getfield` / dereferences will lower through generic `_ptr.getattr`
/// against the resolved Struct.
pub static STR: LazyLock<LowLevelType> = LazyLock::new(|| {
    // Pyre deviation: upstream `Array(Char, hints={'immutable': True,
    // 'extra_item_after_alloc': 1})` carries hints that drive
    // RPython's array layout / immutability tracking. The Rust port
    // omits those hints today — `ArrayType::new` does not accept a
    // hints kwarg, and the structural shape (Array of Char) is the
    // only piece consumed by `ClassesPBCRepr.rtype_getattr('__name__')`
    // and `OBJECT_VTABLE.name`. Adding a `with_hints` constructor is
    // a follow-up alongside the full StringRepr port.
    let chars = ArrayType::new(LowLevelType::Char);
    let body = StructType::gc_with_hints(
        "rpy_string",
        vec![
            ("hash".into(), LowLevelType::Signed),
            ("chars".into(), LowLevelType::Array(Box::new(chars))),
        ],
        vec![(
            "remove_hash".into(),
            crate::flowspace::model::ConstValue::Bool(true),
        )],
    );
    let mut fwd = ForwardReference::gc();
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("STR.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(STR)` — surfaces as the lltype of
/// `OBJECT_VTABLE.name` (rclass.py:171) and the result type of
/// `genop('getfield', [vcls, 'name'], resulttype=Ptr(rstr.STR))` in
/// `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py:980-981).
pub static STRPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let str_t = STR.clone();
    let LowLevelType::ForwardReference(fwd) = str_t else {
        panic!("STR must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

/// RPython `alloc_array_name(name)` (rclass.py:187-188):
///
/// ```python
/// def alloc_array_name(name):
///     return rstr.string_repr.convert_const(name)
/// ```
///
/// Materialises an immortal `_ptr` to a `STR` GcStruct carrying the
/// class name. This is the narrow `StringRepr.convert_const(str)` slice
/// needed by `OBJECT_VTABLE.name`: allocate `STR` with a chars array of
/// `name.len()`, fill the byte chars, and initialise `hash` to 0. The
/// full `StringRepr` hash precomputation and adtmeth table stay in the
/// future full rstr port.
pub fn alloc_array_name(name: &str) -> Result<_ptr, String> {
    let str_body = match STR.clone() {
        LowLevelType::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| "alloc_array_name: STR ForwardReference is not resolved".to_string())?,
        other => other,
    };
    let byte_len = name.len();
    let mut ptr = malloc(str_body, Some(byte_len), MallocFlavor::Gc, true)?;
    ptr.setattr("hash", LowLevelValue::Signed(0))?;
    let Some(obj) = ptr
        ._obj0
        .as_mut()
        .map_err(|_| "alloc_array_name: delayed STR pointer cannot be initialised".to_string())?
    else {
        return Err("alloc_array_name: malloc returned null STR pointer".to_string());
    };
    let _ptr_obj::Struct(s) = obj else {
        return Err("alloc_array_name: STR pointer does not target a Struct".to_string());
    };
    let Some((_, LowLevelValue::Array(chars))) =
        s._fields.iter_mut().find(|(field, _)| field == "chars")
    else {
        return Err("alloc_array_name: STR struct has no chars array".to_string());
    };
    for (i, byte) in name.bytes().enumerate() {
        chars.setitem(i, LowLevelValue::Char(byte as char));
    }
    Ok(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `STR` resolves to a GcStruct named `rpy_string` with the two
    /// upstream fields `(hash: Signed, chars: Array(Char, ...))` and
    /// the `remove_hash` hint that upstream attaches at
    /// rstr.py:1237.
    #[test]
    fn str_resolves_to_gcstruct_rpy_string_with_hash_and_chars() {
        let resolved = match STR.clone() {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().expect("STR ForwardReference must resolve")
            }
            other => panic!("STR must be ForwardReference, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("STR resolves to Struct");
        };
        assert_eq!(body._name, "rpy_string");
        assert!(matches!(body._flds.get("hash"), Some(LowLevelType::Signed)));
        assert!(matches!(
            body._flds.get("chars"),
            Some(LowLevelType::Array(_))
        ));
        // upstream `hints={'remove_hash': True}`.
        assert!(
            body._hints.get("remove_hash").is_some(),
            "remove_hash hint must be present on STR"
        );
    }

    /// `STRPTR` is `Ptr(STR)` — the lltype that `OBJECT_VTABLE.name`
    /// uses and that `ClassesPBCRepr.rtype_getattr('__name__')` would
    /// surface as the resulttype of its `getfield` op.
    #[test]
    fn strptr_is_ptr_to_str_forwardreference() {
        let LowLevelType::Ptr(ptr) = STRPTR.clone() else {
            panic!("STRPTR must be Ptr");
        };
        assert!(matches!(&ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn alloc_array_name_returns_live_str_with_chars() {
        let p = alloc_array_name("Foo").expect("alloc_array_name");
        assert!(p.nonzero());
        let LowLevelValue::Signed(0) = p.getattr("hash").unwrap() else {
            panic!("hash field must be Signed(0)");
        };
        let Some(obj) = p._obj0.as_ref().unwrap().as_ref() else {
            panic!("STR pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("STR pointer must target a Struct");
        };
        let Some((_, LowLevelValue::Array(chars))) =
            s._fields.iter().find(|(field, _)| field == "chars")
        else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), 3);
        assert_eq!(chars.getitem(0), Some(&LowLevelValue::Char('F')));
        assert_eq!(chars.getitem(1), Some(&LowLevelValue::Char('o')));
        assert_eq!(chars.getitem(2), Some(&LowLevelValue::Char('o')));
    }
}
