//! RPython `rpython/rtyper/lltypesystem/rstr.py` — minimal slice for
//! the `Ptr(STR)` / `Ptr(UNICODE)` low-level types that
//! `OBJECT_VTABLE.name` (rclass.py:171) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py:975-981)
//! consume, plus the `UNICODE` data shape that the eventual
//! `UnicodeRepr` port will dereference.
//!
//! Upstream rstr.py weighs ~3000 LOC and houses every operation that
//! rtypes Python's `str` / `unicode` types (`StringRepr`,
//! `UnicodeRepr`, `LLHelpers`, `ll_strconcat`, ...). Pyre lands only
//! the data shape this commit needs:
//!
//! ```python
//! STR = GcForwardReference()
//! UNICODE = GcForwardReference()
//! ...
//! STR.become(GcStruct('rpy_string',
//!                     ('hash', Signed),
//!                     ('chars', Array(Char, hints={'immutable': True,
//!                                     'extra_item_after_alloc': 1})),
//!                     adtmeths={...},
//!                     hints={'remove_hash': True}))
//! UNICODE.become(GcStruct('rpy_unicode',
//!                         ('hash', Signed),
//!                         ('chars', Array(UniChar, hints={'immutable': True})),
//!                         adtmeths={...},
//!                         hints={'remove_hash': True}))
//! ```
//!
//! The full repr port (`AbstractStringRepr`/`AbstractUnicodeRepr`
//! methods, `LLHelpers.ll_*` helpers, comparison, formatting,
//! adtmeths) is deferred to its own follow-on slices — see
//! `~/.claude/projects/.../memory/item3_abstractstringrepr_epic_plan.md`.
//!
//! ## Why a separate module today
//!
//! Both `OBJECT_VTABLE.name` (rclass.py) and
//! `ClassesPBCRepr.rtype_getattr('__name__')` (rpbc.py) want
//! `Ptr(STR)`. Defining `STR` / `UNICODE` here keeps the dependency
//! direction `rclass / rpbc → rstr → lltype` clean and gives a
//! natural home for the eventual full `StringRepr` / `UnicodeRepr`
//! port.

use std::sync::LazyLock;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, _ptr_obj, ArrayType, ForwardReference, LowLevelType, LowLevelValue, MallocFlavor, Ptr,
    PtrTarget, StructType, malloc,
};
use crate::translator::rtyper::rtyper::{
    constant_with_lltype, helper_pygraph_from_graph, variable_with_lltype,
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

/// RPython `UNICODE = GcForwardReference()` resolved via
/// `UNICODE.become(GcStruct('rpy_unicode', ('hash', Signed), ('chars',
/// Array(UniChar, hints={'immutable': True})), hints={'remove_hash':
/// True}))` (rstr.py:33 + rstr.py:1238-1246).
///
/// Mirror of [`STR`] with the `chars` element type swapped from
/// `Char` to `UniChar`. The `extra_item_after_alloc` hint that STR
/// carries (for the trailing NUL slot) is omitted upstream for
/// UNICODE — only `immutable` is set on the array. The Rust port
/// elides both today (see [`STR`] docstring); structural shape is
/// what the eventual `UnicodeRepr` port consumes.
pub static UNICODE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let chars = ArrayType::new(LowLevelType::UniChar);
    let body = StructType::gc_with_hints(
        "rpy_unicode",
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
        .expect("UNICODE.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `Ptr(UNICODE)` — surfaces as the lltype of
/// `UnicodeRepr.lowleveltype` (rstr.py:248) and is the result type of
/// every `gendirectcall(self.ll.ll_*unicode*, ...)` inside
/// `AbstractUnicodeRepr` methods.
pub static UNICODEPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let unicode_t = UNICODE.clone();
    let LowLevelType::ForwardReference(fwd) = unicode_t else {
        panic!("UNICODE must be a ForwardReference");
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

/// `unicode_repr.convert_const(value)` mirror of [`alloc_array_name`]
/// for the `UNICODE` GcStruct. Materialises an immortal `_ptr` to a
/// `UNICODE` carrying the Unicode codepoint sequence of `value` and
/// initialises `hash` to 0. The full `UnicodeRepr.convert_const` and
/// hash precomputation stay in the future full rstr port.
pub fn alloc_array_unicode(value: &str) -> Result<_ptr, String> {
    let unicode_body = match UNICODE.clone() {
        LowLevelType::ForwardReference(fwd) => fwd.resolved().ok_or_else(|| {
            "alloc_array_unicode: UNICODE ForwardReference is not resolved".to_string()
        })?,
        other => other,
    };
    let codepoint_count = value.chars().count();
    let mut ptr = malloc(unicode_body, Some(codepoint_count), MallocFlavor::Gc, true)?;
    ptr.setattr("hash", LowLevelValue::Signed(0))?;
    let Some(obj) = ptr._obj0.as_mut().map_err(|_| {
        "alloc_array_unicode: delayed UNICODE pointer cannot be initialised".to_string()
    })?
    else {
        return Err("alloc_array_unicode: malloc returned null UNICODE pointer".to_string());
    };
    let _ptr_obj::Struct(s) = obj else {
        return Err("alloc_array_unicode: UNICODE pointer does not target a Struct".to_string());
    };
    let Some((_, LowLevelValue::Array(chars))) =
        s._fields.iter_mut().find(|(field, _)| field == "chars")
    else {
        return Err("alloc_array_unicode: UNICODE struct has no chars array".to_string());
    };
    for (i, ch) in value.chars().enumerate() {
        chars.setitem(i, LowLevelValue::UniChar(ch));
    }
    Ok(ptr)
}

// ____________________________________________________________
// LLHelpers — `lltypesystem/rstr.py:307` `class LLHelpers(AbstractLLHelpers)`.
//
// Each helper is synthesised as a single-block low-level graph the
// rtyper inserts via `gendirectcall`. The helpers correspond to
// upstream `LLHelpers.ll_*` static methods.

/// Build the `getsubstruct('chars')` + `getarraysize` op pair shared
/// by `ll_strlen` and `ll_unilen`. The `chars` field is an inline
/// `Array(Char)` / `Array(UniChar)` inside the `STR`/`UNICODE`
/// GcStruct, so RPython emits `getsubstruct` (returning an interior
/// `Ptr(Array(...))`) rather than `getfield`. Compare
/// `rmodel.rs:1290-1306`'s `is_container_type` branch for the same
/// dispatch on the rtyper side.
fn emit_chars_length_ops(
    startblock: &crate::flowspace::model::BlockRef,
    arg: Hlvalue,
    chars_array_ptr_lltype: LowLevelType,
) -> Hlvalue {
    let v_chars = variable_with_lltype("chars", chars_array_ptr_lltype);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getsubstruct",
        vec![
            arg,
            constant_with_lltype(ConstValue::byte_str("chars"), LowLevelType::Void),
        ],
        Hlvalue::Variable(v_chars.clone()),
    ));
    let v_len = variable_with_lltype("length", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "getarraysize",
        vec![Hlvalue::Variable(v_chars)],
        Hlvalue::Variable(v_len.clone()),
    ));
    Hlvalue::Variable(v_len)
}

/// Build the chars Array's `Ptr(Array(...))` lltype for a resolved
/// `STR`/`UNICODE` `ForwardReference`. The result wraps the field's
/// inline `Array(Char|UniChar)` in `Ptr` so the synthesised
/// `getsubstruct` op's result variable carries a valid
/// `SomePtr`-shaped annotation (upstream `lltype.py:1530+`'s
/// `SomePtr.__init__` rejects bare container types — see
/// `llannotation.rs:159-167`).
fn chars_array_ptr_lltype_from_strptr(
    ptr_lltype: &LowLevelType,
) -> Result<LowLevelType, TyperError> {
    let LowLevelType::Ptr(ptr) = ptr_lltype else {
        return Err(TyperError::message(format!(
            "ll_strlen helper expects Ptr(STR/UNICODE), got {ptr_lltype:?}"
        )));
    };
    let body = match &ptr.TO {
        PtrTarget::ForwardReference(fwd) => fwd
            .resolved()
            .ok_or_else(|| TyperError::message("STR/UNICODE ForwardReference not resolved"))?,
        PtrTarget::Struct(s) => LowLevelType::Struct(Box::new(s.clone())),
        other => {
            return Err(TyperError::message(format!(
                "ll_strlen helper Ptr.TO must target STR/UNICODE struct, got {other:?}"
            )));
        }
    };
    let LowLevelType::Struct(body) = body else {
        return Err(TyperError::message(
            "STR/UNICODE ForwardReference must resolve to Struct",
        ));
    };
    let chars_field = body
        ._flds
        .get("chars")
        .cloned()
        .ok_or_else(|| TyperError::message("STR/UNICODE struct has no chars field"))?;
    let LowLevelType::Array(arr) = chars_field else {
        return Err(TyperError::message(
            "STR/UNICODE chars field must be Array(Char|UniChar)",
        ));
    };
    Ok(LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Array(*arr),
    })))
}

/// Synthesise the helper graph for `LLHelpers.ll_strlen` /
/// `LLHelpers.ll_length` (`lltypesystem/rstr.py:351-352, 417-418`):
///
/// ```python
/// @staticmethod
/// def ll_strlen(s):
///     return len(s.chars)
/// ```
///
/// Single-block graph:
/// `getsubstruct(s, 'chars') -> Ptr(Array); getarraysize(...) -> Signed`.
/// `name` is the helper identity (`"ll_strlen"` / `"ll_unilen"`)
/// and `ptr_lltype` selects between `Ptr(STR)` and `Ptr(UNICODE)`.
pub(crate) fn build_ll_strlen_helper_graph(
    name: &str,
    ptr_lltype: LowLevelType,
) -> Result<PyGraph, TyperError> {
    let chars_array_ptr_lltype = chars_array_ptr_lltype_from_strptr(&ptr_lltype)?;
    let arg = variable_with_lltype("s", ptr_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let v_len = emit_chars_length_ops(&startblock, Hlvalue::Variable(arg), chars_array_ptr_lltype);
    startblock.closeblock(vec![
        Link::new(vec![v_len], Some(graph.returnblock.clone()), None).into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["s".to_string()],
        func,
    ))
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

    /// `UNICODE` resolves to a GcStruct named `rpy_unicode` with
    /// the two upstream fields `(hash: Signed, chars: Array(UniChar,
    /// ...))` and the `remove_hash` hint that upstream attaches at
    /// rstr.py:1246. Mirror of
    /// [`str_resolves_to_gcstruct_rpy_string_with_hash_and_chars`]
    /// — verifies the `Array` element type is `UniChar` (not `Char`)
    /// so that an `Ptr(UNICODE).chars[i]` dereference lowers to a
    /// codepoint, not a byte.
    #[test]
    fn unicode_resolves_to_gcstruct_rpy_unicode_with_hash_and_unichar_chars() {
        let resolved = match UNICODE.clone() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("UNICODE ForwardReference must resolve"),
            other => panic!("UNICODE must be ForwardReference, got {other:?}"),
        };
        let LowLevelType::Struct(body) = resolved else {
            panic!("UNICODE resolves to Struct");
        };
        assert_eq!(body._name, "rpy_unicode");
        assert!(matches!(body._flds.get("hash"), Some(LowLevelType::Signed)));
        let Some(LowLevelType::Array(arr)) = body._flds.get("chars") else {
            panic!("chars field must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
        assert!(
            body._hints.get("remove_hash").is_some(),
            "remove_hash hint must be present on UNICODE"
        );
    }

    /// `UNICODEPTR` is `Ptr(UNICODE)` — the lltype that
    /// `UnicodeRepr.lowleveltype` (rstr.py:248) carries.
    #[test]
    fn unicodeptr_is_ptr_to_unicode_forwardreference() {
        let LowLevelType::Ptr(ptr) = UNICODEPTR.clone() else {
            panic!("UNICODEPTR must be Ptr");
        };
        assert!(matches!(&ptr.TO, PtrTarget::ForwardReference(_)));
    }

    /// Round-trip a non-ASCII Unicode string through
    /// `alloc_array_unicode` and verify each codepoint lands as a
    /// distinct `UniChar` cell — confirms that the chars Array
    /// element type is wide enough for the full Unicode range, not
    /// truncated to a single byte.
    #[test]
    fn alloc_array_unicode_returns_live_unicode_with_unichar_codepoints() {
        let p = alloc_array_unicode("αβγ").expect("alloc_array_unicode");
        assert!(p.nonzero());
        let LowLevelValue::Signed(0) = p.getattr("hash").unwrap() else {
            panic!("hash field must be Signed(0)");
        };
        let Some(obj) = p._obj0.as_ref().unwrap().as_ref() else {
            panic!("UNICODE pointer must be live");
        };
        let _ptr_obj::Struct(s) = obj else {
            panic!("UNICODE pointer must target a Struct");
        };
        let Some((_, LowLevelValue::Array(chars))) =
            s._fields.iter().find(|(field, _)| field == "chars")
        else {
            panic!("chars field must be an Array");
        };
        assert_eq!(chars.getlength(), 3);
        assert_eq!(chars.getitem(0), Some(&LowLevelValue::UniChar('α')));
        assert_eq!(chars.getitem(1), Some(&LowLevelValue::UniChar('β')));
        assert_eq!(chars.getitem(2), Some(&LowLevelValue::UniChar('γ')));
    }

    /// `ll_strlen` synthesised against `Ptr(STR)` produces a single
    /// startblock with `getsubstruct('chars')` then `getarraysize`,
    /// and returns `Signed`. Mirrors upstream
    /// `lltypesystem/rstr.py:351-352`:
    /// `def ll_strlen(s): return len(s.chars)`.
    /// `getsubstruct` (not `getfield`) is the correct lltype op for
    /// the inline composite `chars` field — see `rmodel.rs:1290-1306`
    /// `is_container_type` branch.
    #[test]
    fn build_ll_strlen_emits_getsubstruct_chars_then_getarraysize() {
        let helper = build_ll_strlen_helper_graph("ll_strlen", STRPTR.clone())
            .expect("build_ll_strlen_helper_graph");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["getsubstruct", "getarraysize"]);
        // `getsubstruct` first arg is the input `s`, second is the
        // void field-name constant `"chars"`.
        let getsubstruct_args = &startblock.operations[0].args;
        assert_eq!(getsubstruct_args.len(), 2);
        let Hlvalue::Constant(c) = &getsubstruct_args[1] else {
            panic!("getsubstruct's second arg must be a Constant");
        };
        assert!(matches!(c.value, ConstValue::ByteStr(ref b) if b == b"chars"));
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
    }

    /// `ll_unilen` is the `Ptr(UNICODE)` mirror of `ll_strlen` — same
    /// op sequence, just a different input pointer lltype. The chars
    /// Array element lltype derived from the struct body must be
    /// `UniChar`, not `Char`. The `getsubstruct` result is wrapped as
    /// `Ptr(Array(UniChar))` so it carries a valid `SomePtr`-shaped
    /// annotation.
    #[test]
    fn build_ll_unilen_emits_same_op_sequence_with_unichar_chars_lltype() {
        let helper = build_ll_strlen_helper_graph("ll_unilen", UNICODEPTR.clone())
            .expect("build_ll_strlen_helper_graph for UNICODE");
        let inner = helper.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["getsubstruct", "getarraysize"]);
        let getsubstruct_result = &startblock.operations[0].result;
        let Hlvalue::Variable(v) = getsubstruct_result else {
            panic!("getsubstruct result must be a Variable");
        };
        let chars_lltype = v
            .concretetype
            .borrow()
            .clone()
            .expect("chars var must have concretetype");
        let LowLevelType::Ptr(ptr) = chars_lltype else {
            panic!("chars var must be Ptr");
        };
        let PtrTarget::Array(arr) = &ptr.TO else {
            panic!("chars Ptr.TO must be Array");
        };
        assert_eq!(arr.OF, LowLevelType::UniChar);
    }

    /// The synthesised helper graph carries the helper-identity
    /// `name` on both its FunctionGraph and `func` slot, and exposes
    /// the input slot via the startblock's input variable. Required so
    /// `helper_pygraph_from_graph`'s caller cache dedupes by
    /// `(name, args)` rather than by graph identity.
    #[test]
    fn build_ll_strlen_helper_carries_name_and_input_s_variable() {
        let helper = build_ll_strlen_helper_graph("ll_strlen", STRPTR.clone()).unwrap();
        assert_eq!(helper.func.name, "ll_strlen");
        let inner = helper.graph.borrow();
        assert_eq!(inner.name, "ll_strlen");
        let startblock = inner.startblock.borrow();
        assert_eq!(startblock.inputargs.len(), 1);
        let Hlvalue::Variable(v) = &startblock.inputargs[0] else {
            panic!("startblock input must be a Variable");
        };
        assert!(
            v.name().starts_with('s'),
            "input variable name = {:?}",
            v.name()
        );
    }

    /// Passing a non-Ptr lltype should be rejected — upstream's
    /// `len(s.chars)` is only meaningful when `s: Ptr(STR/UNICODE)`.
    /// Guards a future caller from accidentally requesting a
    /// `ll_strlen` against e.g. `LowLevelType::Char`.
    #[test]
    fn build_ll_strlen_rejects_non_ptr_input_lltype() {
        let err = build_ll_strlen_helper_graph("ll_strlen", LowLevelType::Char)
            .expect_err("non-Ptr input must fail");
        assert!(format!("{err:?}").contains("Ptr(STR/UNICODE)"));
    }
}
