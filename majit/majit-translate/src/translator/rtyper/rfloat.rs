//! RPython `rpython/rtyper/rfloat.py` — `FloatRepr` + `SingleFloatRepr`
//! + `LongFloatRepr` + singleton + `SomeFloat/SingleFloat/LongFloat`
//! dispatch.
//!
//! Upstream rfloat.py (174 LOC) covers three Reprs:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class FloatRepr(Repr)` (`rfloat.py:11-64`) | [`FloatRepr`] |
//! | `float_repr = FloatRepr()` singleton (`rfloat.py:65`) | [`float_repr`] |
//! | `class SingleFloatRepr(Repr)` (`rfloat.py:150-158`) | [`SingleFloatRepr`] |
//! | `class LongFloatRepr(Repr)` (`rfloat.py:166-174`) | [`LongFloatRepr`] |
//! | `SomeFloat/SomeSingleFloat/SomeLongFloat rtyper_make*` (`rfloat.py:67-72,144-148,160-164`) | wired in [`super::rmodel::rtyper_makerepr`] + [`super::rmodel::rtyper_makekey`] |
//!
//! ## Deferred to follow-up commits
//!
//! * `pairtype(FloatRepr, FloatRepr)` binary ops + comparisons
//!   (`rfloat.py:75-135`) — `rtype_add/sub/mul/truediv/eq/ne/lt/le/gt/ge`
//!   live on the pairtype, not on `FloatRepr` itself. Upstream
//!   `translate_op_add` (`rtyper.py`) resolves through the pairtype
//!   dispatcher (`tool/pairtype.py`), which is not yet ported; the
//!   binary-op helpers `_rtype_template` / `_rtype_compare_template`
//!   land with the pairtype bridge.
//! * `get_ll_{eq,gt,lt,ge,le}_function` / `get_ll_hash_function`
//!   (`rfloat.py:19-27`) — require a trait slot for the per-Repr
//!   comparator / hasher function pointer. Used by `rdict.py` /
//!   `rordereddict.py` when those land; no current consumer.
//! * `ll_str(self, f)` (`rfloat.py:60-63`) — depends on
//!   `rpython/rlib/rfloat.py:formatd` + `annlowlevel.llstr` which are
//!   not ported.
//! * `_hash_float` (`rlib/objectmodel.py`) — pyre exposes
//!   [`hash_float_stub`] as a placeholder so the `get_ll_hash_function`
//!   slot lands in the same place when the trait gains the hash slot.

use std::sync::Arc;
use std::sync::OnceLock;

use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp};

/// RPython `class FloatRepr(Repr)` (`rfloat.py:11-64`).
///
/// ```python
/// class FloatRepr(Repr):
///     lowleveltype = Float
///
///     def convert_const(self, value):
///         if not isinstance(value, (int, base_int, float)):  # can be bool too
///             raise TyperError("not a float: %r" % (value,))
///         return float(value)
///
///     def rtype_bool(_, hop):
///         vlist = hop.inputargs(Float)
///         return hop.genop('float_is_true', vlist, resulttype=Bool)
///
///     def rtype_neg(_, hop):
///         vlist = hop.inputargs(Float)
///         return hop.genop('float_neg', vlist, resulttype=Float)
///
///     def rtype_pos(_, hop):
///         vlist = hop.inputargs(Float)
///         return vlist[0]
///
///     def rtype_abs(_, hop):
///         vlist = hop.inputargs(Float)
///         return hop.genop('float_abs', vlist, resulttype=Float)
///
///     def rtype_int(_, hop):
///         vlist = hop.inputargs(Float)
///         hop.exception_cannot_occur()
///         return hop.genop('cast_float_to_int', vlist, resulttype=Signed)
///
///     def rtype_float(_, hop):
///         vlist = hop.inputargs(Float)
///         hop.exception_cannot_occur()
///         return vlist[0]
/// ```
#[derive(Debug)]
pub struct FloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl FloatRepr {
    pub fn new() -> Self {
        FloatRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Float,
        }
    }
}

impl Default for FloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for FloatRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FloatRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::FloatRepr
    }

    /// RPython `FloatRepr.convert_const(self, value)` (`rfloat.py:14-17`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, (int, base_int, float)):  # can be bool too
    ///         raise TyperError("not a float: %r" % (value,))
    ///     return float(value)
    /// ```
    ///
    /// `base_int` covers RPython's `r_uint` / `r_longlong` etc. from
    /// `rlib/rarithmetic.py`; pyre maps those to `ConstValue::Int` so
    /// the `int | base_int | float | bool` check collapses to
    /// "is the ConstValue an Int, Float, or Bool".
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let as_float = match value {
            ConstValue::Float(bits) => *bits,
            ConstValue::Int(i) => (*i as f64).to_bits(),
            ConstValue::Bool(b) => (if *b { 1.0_f64 } else { 0.0_f64 }).to_bits(),
            other => {
                return Err(TyperError::message(format!("not a float: {other:?}")));
            }
        };
        Ok(Constant::with_concretetype(
            ConstValue::Float(as_float),
            LowLevelType::Float,
        ))
    }

    /// RPython `FloatRepr.rtype_bool(_, hop)` (`rfloat.py:32-34`):
    /// `return hop.genop('float_is_true', [v], resulttype=Bool)`.
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(hop.genop(
            "float_is_true",
            vlist,
            GenopResult::LLType(LowLevelType::Bool),
        ))
    }

    /// RPython `FloatRepr.rtype_neg(_, hop)` (`rfloat.py:36-38`):
    /// `return hop.genop('float_neg', [v], resulttype=Float)`.
    fn rtype_neg(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(hop.genop("float_neg", vlist, GenopResult::LLType(LowLevelType::Float)))
    }

    /// RPython `FloatRepr.rtype_pos(_, hop)` (`rfloat.py:40-42`):
    /// `return vlist[0]` — identity pass-through after inputargs coerces.
    fn rtype_pos(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(vlist.into_iter().next())
    }

    /// RPython `FloatRepr.rtype_abs(_, hop)` (`rfloat.py:44-46`):
    /// `return hop.genop('float_abs', [v], resulttype=Float)`.
    fn rtype_abs(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        Ok(hop.genop("float_abs", vlist, GenopResult::LLType(LowLevelType::Float)))
    }

    /// RPython `FloatRepr.rtype_int(_, hop)` (`rfloat.py:48-53`):
    ///
    /// ```python
    /// def rtype_int(_, hop):
    ///     vlist = hop.inputargs(Float)
    ///     # int(x) never raises in RPython, you need to use
    ///     # rarithmetic.ovfcheck_float_to_int() if you want this
    ///     hop.exception_cannot_occur()
    ///     return hop.genop('cast_float_to_int', vlist, resulttype=Signed)
    /// ```
    fn rtype_int(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        hop.exception_cannot_occur()?;
        Ok(hop.genop(
            "cast_float_to_int",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    /// RPython `FloatRepr.rtype_float(_, hop)` (`rfloat.py:55-58`):
    /// identity pass-through plus `exception_cannot_occur`.
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        hop.exception_cannot_occur()?;
        Ok(vlist.into_iter().next())
    }
}

/// RPython `float_repr = FloatRepr()` (`rfloat.py:65`).
pub fn float_repr() -> Arc<FloatRepr> {
    static REPR: OnceLock<Arc<FloatRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(FloatRepr::new())).clone()
}

// RPython `FloatRepr.get_ll_hash_function` (`rfloat.py:26-27`) returns
// `rlib.objectmodel._hash_float`. The pyre port lands that helper
// alongside the `get_ll_hash_function` trait slot itself — rdict.py is
// the first consumer. Until then no placeholder is exposed here: a
// mid-port `hash_float_stub` with a hard-coded `return 0` would mask
// the missing wiring instead of surfacing it.

// ____________________________________________________________
// pairtype(FloatRepr, FloatRepr) — rfloat.py:75-135.

pub fn rtype_template(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Float),
        ConvertedTo::LowLevelType(&LowLevelType::Float),
    ])?;
    let opname = format!("float_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Float)))
}

pub fn rtype_compare_template(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Float),
        ConvertedTo::LowLevelType(&LowLevelType::Float),
    ])?;
    let opname = format!("float_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

// ____________________________________________________________
// SingleFloatRepr / LongFloatRepr — `rfloat.py:138-174`.
//
// Upstream creates a fresh `SingleFloatRepr()` / `LongFloatRepr()` per
// `rtyper_makerepr` call (no module-global singleton). Pyre mirrors by
// instantiating a fresh `Arc<SingleFloatRepr>` / `Arc<LongFloatRepr>`
// in [`super::rmodel::rtyper_makerepr`] per SomeSingleFloat /
// SomeLongFloat.

/// RPython `class SingleFloatRepr(Repr)` (`rfloat.py:150-158`).
#[derive(Debug)]
pub struct SingleFloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl SingleFloatRepr {
    pub fn new() -> Self {
        SingleFloatRepr {
            state: ReprState::new(),
            lltype: LowLevelType::SingleFloat,
        }
    }
}

impl Default for SingleFloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for SingleFloatRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SingleFloatRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::SingleFloatRepr
    }

    /// RPython `SingleFloatRepr.rtype_float(self, hop)`
    /// (`rfloat.py:153-158`):
    ///
    /// ```python
    /// def rtype_float(self, hop):
    ///     v, = hop.inputargs(lltype.SingleFloat)
    ///     hop.exception_cannot_occur()
    ///     return hop.genop('cast_primitive', [v], resulttype=lltype.Float)
    /// ```
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::SingleFloat)])?;
        hop.exception_cannot_occur()?;
        Ok(hop.genop(
            "cast_primitive",
            vlist,
            GenopResult::LLType(LowLevelType::Float),
        ))
    }
}

/// RPython `class LongFloatRepr(Repr)` (`rfloat.py:166-174`).
#[derive(Debug)]
pub struct LongFloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl LongFloatRepr {
    pub fn new() -> Self {
        LongFloatRepr {
            state: ReprState::new(),
            lltype: LowLevelType::LongFloat,
        }
    }
}

impl Default for LongFloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for LongFloatRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "LongFloatRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::LongFloatRepr
    }

    /// RPython `LongFloatRepr.rtype_float(self, hop)`
    /// (`rfloat.py:169-174`): symmetrical to [`SingleFloatRepr::rtype_float`]
    /// but over `LongFloat → Float`.
    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::LongFloat)])?;
        hop.exception_cannot_occur()?;
        Ok(hop.genop(
            "cast_primitive",
            vlist,
            GenopResult::LLType(LowLevelType::Float),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn float_repr_lowleveltype_and_repr_string_match_upstream() {
        // rfloat.py:12 — `lowleveltype = Float`.
        let r = FloatRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::Float);
        assert_eq!(r.repr_string(), "<FloatRepr Float>");
        assert_eq!(r.compact_repr(), "FloatR Float");
    }

    #[test]
    fn float_repr_singleton_returns_same_arc() {
        // rfloat.py:65 — `float_repr = FloatRepr()` module-global.
        let a = float_repr();
        let b = float_repr();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn convert_const_accepts_int_float_bool() {
        // rfloat.py:14-17 — `isinstance(value, (int, base_int, float))`
        // admits all numeric kinds. Pyre normalises to `ConstValue::Float`.
        let r = FloatRepr::new();
        let c_int = r.convert_const(&ConstValue::Int(42)).unwrap();
        let ConstValue::Float(bits) = c_int.value else {
            panic!("expected Float after int coercion");
        };
        assert_eq!(f64::from_bits(bits), 42.0);

        let c_bool = r.convert_const(&ConstValue::Bool(true)).unwrap();
        let ConstValue::Float(bits) = c_bool.value else {
            panic!("expected Float after bool coercion");
        };
        assert_eq!(f64::from_bits(bits), 1.0);

        let c_float = r.convert_const(&ConstValue::Float(3.14_f64.to_bits()));
        let c = c_float.unwrap();
        assert_eq!(c.value, ConstValue::Float(3.14_f64.to_bits()));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Float));
    }

    #[test]
    fn convert_const_rejects_non_numeric() {
        // rfloat.py:15-16 — `raise TyperError("not a float: %r" % ...)`.
        let r = FloatRepr::new();
        let err = r
            .convert_const(&ConstValue::Str("pi".to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("not a float"));
    }

    #[test]
    fn single_float_repr_lowleveltype_matches_upstream() {
        // rfloat.py:151 — `lowleveltype = lltype.SingleFloat`.
        let r = SingleFloatRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::SingleFloat);
        assert_eq!(r.repr_string(), "<SingleFloatRepr SingleFloat>");
    }

    #[test]
    fn long_float_repr_lowleveltype_matches_upstream() {
        // rfloat.py:167 — `lowleveltype = lltype.LongFloat`.
        let r = LongFloatRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::LongFloat);
        assert_eq!(r.repr_string(), "<LongFloatRepr LongFloat>");
    }

    #[test]
    fn rtyper_getrepr_on_some_float_returns_the_singleton() {
        // rfloat.py:67-69 — `SomeFloat.rtyper_makerepr` returns
        // `float_repr` (module-global). Verify the full dispatch chain.
        use crate::annotator::model::{SomeFloat, SomeValue};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s_float = SomeValue::Float(SomeFloat::new());
        let r = rtyper.getrepr(&s_float).expect("getrepr(SomeFloat)");
        let expected = float_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r, &expected));

        let r2 = rtyper.getrepr(&s_float).expect("second getrepr(SomeFloat)");
        assert!(Arc::ptr_eq(&r, &r2));
    }

    #[test]
    fn rtyper_getrepr_on_some_single_float_allocates_per_call_repr() {
        // rfloat.py:144-148 — `SomeSingleFloat.rtyper_makerepr`
        // returns `SingleFloatRepr()` (fresh per call — no module-
        // global singleton). But `getrepr` caches the first produced
        // instance per key, so two lookups return the same Arc.
        use crate::annotator::model::{SomeSingleFloat, SomeValue};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s_single = SomeValue::SingleFloat(SomeSingleFloat::new());
        let r = rtyper.getrepr(&s_single).expect("getrepr(SomeSingleFloat)");
        assert_eq!(r.lowleveltype(), &LowLevelType::SingleFloat);
        let r2 = rtyper
            .getrepr(&s_single)
            .expect("second getrepr(SomeSingleFloat)");
        // rtyper.py:54-57: cached Arc identity preserved across lookups.
        assert!(Arc::ptr_eq(&r, &r2));
    }
}
