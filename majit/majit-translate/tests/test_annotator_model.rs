//! Port of `rpython/annotator/test/test_model.py` (343 LOC, Phase 4 A4.8).
//!
//! Upstream covers ~22 tests but many require `TranslationContext` /
//! `buildannotator` / `rpython.annotator.binaryop` side-effect
//! registration — all of which land in Phase 5.  This file ports the
//! subset that exercises only the model.py primitives already carried
//! by Phase 4:
//!
//!   * `test_equality`
//!   * `test_contains`
//!   * `test_signedness`
//!   * `test_list_union`
//!   * `test_list_contains`
//!   * `test_nan`
//!
//! Upstream tests that depend on Phase 5 (`test_commonbase_simple`,
//! `test_instance_*`, `test_dict_update_*`, `test_iter_*`, the
//! `@given` Hypothesis property tests, and every annotator-driven
//! test) are deferred.

use majit_translate::annotator::model::{
    ListDef, SomeFloat, SomeImpossibleValue, SomeInteger, SomeList, SomeString, SomeTuple,
    SomeType, SomeValue, contains, union, unionof,
};
use majit_translate::flowspace::model::{ConstValue, Constant};

// Mirror of the upstream module-level fixtures (`s1..s6` / `slist`) so
// that each test reads like the Python original.

fn s1() -> SomeValue {
    SomeValue::Type(SomeType::new())
}

fn s2() -> SomeValue {
    SomeValue::Integer(SomeInteger::new(true, false))
}

fn s3() -> SomeValue {
    SomeValue::Integer(SomeInteger::new(false, false))
}

fn listdef_tuple_nonneg() -> ListDef {
    ListDef::new(SomeValue::Tuple(SomeTuple::new(vec![
        SomeValue::Integer(SomeInteger::new(true, false)),
        SomeValue::String(SomeString::default()),
    ])))
}

fn listdef_tuple_signed() -> ListDef {
    ListDef::new(SomeValue::Tuple(SomeTuple::new(vec![
        SomeValue::Integer(SomeInteger::new(false, false)),
        SomeValue::String(SomeString::default()),
    ])))
}

fn s4() -> SomeValue {
    SomeValue::List(SomeList::new(listdef_tuple_nonneg()))
}

fn s5() -> SomeValue {
    SomeValue::List(SomeList::new(listdef_tuple_signed()))
}

fn s6() -> SomeValue {
    // upstream: `s6 = SomeImpossibleValue()`. Our Impossible enum
    // variant and the explicit SomeImpossibleValue struct are two
    // presentations of the same thing; stick with the enum form to
    // match the rest of the port surface.
    SomeValue::Impossible
}

fn const_float(x: f64) -> SomeValue {
    let mut s = SomeFloat::new();
    s.base.const_box = Some(Constant::new(ConstValue::Float(x.to_bits())));
    SomeValue::Float(s)
}

// ---------------------------------------------------------------------------
// test_equality (test_model.py:35-42)
// ---------------------------------------------------------------------------

#[test]
fn test_equality() {
    // upstream binds listdef1/listdef2 once at module scope and reuses
    // them in every `SomeList(...)` construction, so the identity-based
    // same_as holds. Mirror that here — the Phase 5 P5.1 port makes
    // `ListDef.same_as` identity-based, so fresh ListDefs compare
    // unequal even when their element types agree.
    let listdef1 = listdef_tuple_nonneg();
    let listdef2 = listdef_tuple_signed();
    let s1_ = s1();
    let s2_ = s2();
    let s3_ = s3();
    let s4_ = SomeValue::List(SomeList::new(listdef1.clone()));
    let s5_ = SomeValue::List(SomeList::new(listdef2.clone()));
    let s6_ = s6();

    assert_ne!(s1_, s2_);
    assert_ne!(s2_, s3_);
    assert_ne!(s3_, s4_);
    assert_ne!(s4_, s5_);
    assert_ne!(s5_, s6_);

    assert_eq!(s1_, SomeValue::Type(SomeType::new()));
    assert_eq!(s2_, SomeValue::Integer(SomeInteger::new(true, false)));
    assert_eq!(s3_, SomeValue::Integer(SomeInteger::new(false, false)));
    // Shared listdef → same_as via Rc::ptr_eq, matches upstream
    // `s4 == SomeList(listdef1)`.
    assert_eq!(s4_, SomeValue::List(SomeList::new(listdef1.clone())));
    assert_eq!(s5_, SomeValue::List(SomeList::new(listdef2.clone())));
    assert_eq!(s6_, SomeValue::Impossible);
    assert_eq!(SomeValue::Impossible, SomeValue::Impossible);
    let _ = SomeImpossibleValue::new();
}

// ---------------------------------------------------------------------------
// test_contains (test_model.py:44-50)
// ---------------------------------------------------------------------------

#[test]
fn test_contains() {
    // upstream builds a pair table via `[(s, t) for ...]`. Rust port
    // lists the expected True pairs explicitly — anything else is
    // False.
    let slist = [s1(), s2(), s3(), s4(), s6()];
    let expected_true_pairs: &[(usize, usize)] = &[
        (0, 0),
        (0, 4),
        (1, 1),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 4),
        (3, 3),
        (3, 4),
        (4, 4),
    ];
    for (i, s) in slist.iter().enumerate() {
        for (j, t) in slist.iter().enumerate() {
            let expected = expected_true_pairs.contains(&(i, j));
            assert_eq!(
                s.contains(t),
                expected,
                "s{}.contains(s{}) expected {}",
                i + 1,
                j + 1,
                expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// test_signedness (test_model.py:52-54)
// ---------------------------------------------------------------------------

#[test]
fn test_signedness() {
    // upstream:
    // `assert not SomeInteger(unsigned=True).contains(SomeInteger())`
    // `assert SomeInteger(unsigned=True).contains(SomeInteger(nonneg=True))`
    let unsigned = SomeValue::Integer(SomeInteger::new(false, true));
    let signed = SomeValue::Integer(SomeInteger::new(false, false));
    let nonneg = SomeValue::Integer(SomeInteger::new(true, false));

    assert!(!unsigned.contains(&signed));
    assert!(unsigned.contains(&nonneg));
}

// ---------------------------------------------------------------------------
// test_list_union (test_model.py:79-86) — enabled after Phase 5 P5.1.
//
// Upstream body:
//     s1 = SomeList(ListDef('dummy', SomeInteger(nonneg=True)))
//     s2 = SomeList(ListDef('dummy', SomeInteger(nonneg=False)))
//     assert s1 != s2
//     s3 = unionof(s1, s2)
//     assert s1 == s2 == s3
//
// Phase 5 P5.1's `ListDef::union_with` walks `ListItem.itemof`
// backrefs and patches every `ListDefInner::listitem` slot to the
// merged cell, so `s1.listdef.same_as(s2.listdef)` becomes True
// post-merge.
#[test]
fn test_list_union() {
    // upstream uses `ListDef('dummy', ...)` — mutable-bookkeeper
    // path. Mirror with `ListDef::mutable`.
    let s1 = SomeValue::List(SomeList::new(ListDef::mutable(SomeValue::Integer(
        SomeInteger::new(true, false),
    ))));
    let s2 = SomeValue::List(SomeList::new(ListDef::mutable(SomeValue::Integer(
        SomeInteger::new(false, false),
    ))));
    assert_ne!(s1, s2);
    let s3 = unionof([&s1, &s2]).unwrap();
    // upstream post-condition.
    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
}

/// Rust-weaker variant of `test_list_union`. Asserts the invariant
/// that Phase 4's non-mutating union DOES preserve: the returned
/// `SomeList`'s element is the merged SomeInteger bound. Not a
/// line-by-line port of any upstream test; exists so we have
/// coverage for the actual Phase 4 behaviour while the upstream
/// mutation-based assertion is ignored above.
#[test]
fn test_list_union_merged_element_shape_phase4() {
    // Use mutable-bookkeeper ListDef so the merge is allowed.
    let a = SomeValue::List(SomeList::new(ListDef::mutable(SomeValue::Integer(
        SomeInteger::new(true, false),
    ))));
    let b = SomeValue::List(SomeList::new(ListDef::mutable(SomeValue::Integer(
        SomeInteger::new(false, false),
    ))));
    let merged = union(&a, &b).expect("same-shape lists must union");
    let SomeValue::List(list) = merged else {
        panic!("expected merged value to be SomeList");
    };
    let SomeValue::Integer(elem) = list.listdef.s_value() else {
        panic!("expected merged element to be SomeInteger");
    };
    assert!(!elem.nonneg);
    assert!(!elem.unsigned);
}

// ---------------------------------------------------------------------------
// test_list_contains (test_model.py:88-97) — IGNORED in Phase 4.
//
// Upstream asserts BOTH:
//     assert not s2.contains(s1)
//     assert not s1.contains(s2)
//
// The upstream semantics are identity-based: `ListDef.same_as` at
// rpython/annotator/listdef.py:124 delegates to
// `self.listitem.same_as(other.listitem)` which ultimately checks
// `self is other` (Python object identity) after the bookkeeper's
// union-find resolves aliasing. Two independently constructed
// `ListDef`s therefore have DISTINCT `ListItem`s regardless of
// element-type equality, and neither list subsumes the other.
//
// The Rust port's `ListDef::same_as` in
// majit-translate/src/annotator/model.rs:664 collapses to structural
// element comparison (no bookkeeper / no ListItem identity yet), so
// it cannot reproduce upstream's "neither direction contains" result.
// Phase 5's rpython/annotator/listdef.py port wires the real
// identity-based semantics; this test switches back on then.
//
// The previous revision of this file pinned `assert!(contains(&s2, &s1))`
// — which is the OPPOSITE of upstream's assertion. The test is
// gated off until the listdef port lands so we do not ship a test
// whose intent diverges from upstream.

#[test]
fn test_list_contains() {
    // upstream uses `ListDef(None, ...)` — bookkeeper-None path with
    // dont_change_any_more = true. The Rust port's `ListDef::new`
    // matches that shape by default.
    let s1 = SomeValue::List(SomeList::new(ListDef::new(SomeValue::Integer(
        SomeInteger::new(true, false),
    ))));
    let s2 = SomeValue::List(SomeList::new(ListDef::new(SomeValue::Integer(
        SomeInteger::new(false, false),
    ))));
    assert_ne!(s1, s2);
    // `contains` enters a SideEffectFreeGuard, so SomeList's
    // union_with → ListItem::merge raises UnionError → contains
    // returns False. Matches upstream semantics line-by-line.
    assert!(!contains(&s2, &s1));
    assert!(!contains(&s1, &s2));
}

// ---------------------------------------------------------------------------
// test_nan (test_model.py:99-106)
// ---------------------------------------------------------------------------

#[test]
fn test_nan() {
    let f1 = const_float(f64::NAN);
    let f2 = const_float(f64::from_bits(0x7ff8_0000_0000_0001));
    assert!(f1.contains(&f1));
    assert!(f1.contains(&f2));
    assert!(f2.contains(&f1));
}

// ---------------------------------------------------------------------------
// unionof variadic smoke test — upstream tests `unionof(*somevalues)`.
// ---------------------------------------------------------------------------

#[test]
fn test_unionof_multiple_integers() {
    let cells = vec![
        SomeValue::Integer(SomeInteger::new(true, false)),
        SomeValue::Integer(SomeInteger::new(true, false)),
        SomeValue::Integer(SomeInteger::new(false, false)),
    ];
    let u = unionof(&cells).unwrap();
    if let SomeValue::Integer(s) = u {
        assert!(!s.nonneg, "one signed entry widens the fold");
    } else {
        panic!("expected SomeInteger fold result");
    }
}
