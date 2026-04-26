//! RPython `rpython/rtyper/rbool.py` — `BoolRepr` + `bool_repr`
//! singleton + `SomeBool` dispatch.
//!
//! Upstream rbool.py (84 LOC) covers three surfaces:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class BoolRepr(IntegerRepr)` (`rbool.py:10-35`) | [`BoolRepr`] |
//! | `bool_repr = BoolRepr()` singleton (`rbool.py:36`) | [`bool_repr`] |
//! | `SomeBool.rtyper_makerepr / rtyper_makekey` (`rbool.py:39-44`) | wired in [`super::rmodel::rtyper_makerepr`] + [`super::rmodel::rtyper_makekey`] |
//!
//! ## Deferred to follow-up commits
//!
//! * `BoolRepr(IntegerRepr)` inheritance (`rbool.py:10`) — upstream
//!   inherits `IntegerRepr` so the bool value participates in
//!   `pairtype(BoolRepr, IntegerRepr)` conversions. `rint.py IntegerRepr`
//!   has not been ported yet; Rust implements [`BoolRepr`] as a
//!   standalone `Repr` and the rint port will wire a shared trait
//!   (`IntegerRepr` sub-trait) when it lands.
//! * `self.as_int = signed_repr` (`rbool.py:14-15`) — Rust does not
//!   carry an explicit field, but `BoolRepr` routes integer coercions
//!   through the same `Signed` repr that upstream stores there.
//! * pairtype conversions `BoolRepr ↔ FloatRepr` / `BoolRepr ↔ IntegerRepr`
//!   (`rbool.py:49-84`) — covered by the pairtype dispatcher port (see
//!   [`super::rnone`] deferral note for the shared plan).

use std::sync::Arc;
use std::sync::OnceLock;

use crate::flowspace::model::{ConstValue, Constant, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{ConvertedTo, GenopResult, HighLevelOp, LowLevelOpList};

/// RPython `class BoolRepr(IntegerRepr)` (`rbool.py:10-35`).
///
/// ```python
/// class BoolRepr(IntegerRepr):
///     lowleveltype = Bool
///     # NB. no 'opprefix' here.  Use 'as_int' systematically.
///     def __init__(self):
///         from rpython.rtyper.rint import signed_repr
///         self.as_int = signed_repr
///
///     def convert_const(self, value):
///         if not isinstance(value, bool):
///             raise TyperError("not a bool: %r" % (value,))
///         return value
///
///     def rtype_bool(_, hop):
///         vlist = hop.inputargs(bool_repr)
///         return vlist[0]
///
///     def rtype_int(_, hop):
///         vlist = hop.inputargs(Signed)
///         hop.exception_cannot_occur()
///         return vlist[0]
///
///     def rtype_float(_, hop):
///         vlist = hop.inputargs(Float)
///         hop.exception_cannot_occur()
///         return vlist[0]
/// ```
///
#[derive(Debug)]
pub struct BoolRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl BoolRepr {
    pub fn new() -> Self {
        BoolRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Bool,
        }
    }
}

impl Default for BoolRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for BoolRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "BoolRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::BoolRepr
    }

    /// RPython `BoolRepr` inherits from `IntegerRepr` (`rbool.py:10`),
    /// picking up `IntegerRepr.get_ll_eq_function` (`rint.py:39-42`)
    /// which returns `None` for non-shortint widths. Bool's lltype
    /// has a non-`None` `_opprefix` (`'int_'`), so the `None` branch
    /// fires; explicit override in Rust since trait dispatch does not
    /// inherit from a concrete type.
    fn get_ll_eq_function(
        &self,
        _rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `BoolRepr` inherits `IntegerRepr.get_ll_hash_function`
    /// (`rint.py:50-54`) returning `ll_hash_int`. Bool's bit-pattern
    /// representation is `Bool` lltype (1 bit / 1 byte); `intmask(b)`
    /// widens it to `Signed` semantically — synthesizes the same
    /// single-block helper graph as IntegerRepr via the shared
    /// [`super::rint::build_ll_hash_int_helper_graph`] builder.
    fn get_ll_hash_function(
        &self,
        rtyper: &super::rtyper::RPythonTyper,
    ) -> Result<Option<super::rtyper::LowLevelFunction>, TyperError> {
        let lltype = LowLevelType::Bool;
        let name = format!("ll_hash_int_{}", lltype.short_name());
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![lltype.clone()],
                LowLevelType::Signed,
                move |_rtyper, args, _result| {
                    super::rint::build_ll_hash_int_helper_graph(&name, &args[0])
                },
            )
            .map(Some)
    }

    /// RPython `BoolRepr.convert_const(self, value)` (`rbool.py:17-20`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, bool):
    ///         raise TyperError("not a bool: %r" % (value,))
    ///     return value
    /// ```
    ///
    /// Upstream returns the raw Python bool; pyre wraps in a typed
    /// `Constant` the same way the base `Repr::convert_const` does
    /// (carrying `concretetype=Bool`).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        if !matches!(value, ConstValue::Bool(_)) {
            return Err(TyperError::message(format!("not a bool: {value:?}")));
        }
        Ok(Constant::with_concretetype(
            value.clone(),
            LowLevelType::Bool,
        ))
    }

    /// RPython `BoolRepr.rtype_bool(_, hop)` (`rbool.py:22-24`):
    ///
    /// ```python
    /// def rtype_bool(_, hop):
    ///     vlist = hop.inputargs(bool_repr)
    ///     return vlist[0]
    /// ```
    ///
    /// `hop.inputargs(bool_repr)` coerces arg 0 to the `bool_repr`
    /// singleton; since the arg's repr already is `bool_repr` (the hop
    /// dispatched to `BoolRepr.rtype_bool` in the first place) upstream
    /// effectively returns `args_v[0]` unchanged. Pyre mirrors the
    /// inputargs+return shape so the pairtype dispatcher sees the same
    /// convertvar call sequence upstream does.
    fn rtype_bool(&self, hop: &HighLevelOp) -> RTypeResult {
        let singleton = bool_repr();
        let vlist = hop.inputargs(vec![ConvertedTo::Repr(singleton.as_ref())])?;
        Ok(vlist.into_iter().next())
    }

    fn rtype_int(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Signed)])?;
        hop.exception_cannot_occur()?;
        Ok(vlist.into_iter().next())
    }

    fn rtype_float(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Float)])?;
        hop.exception_cannot_occur()?;
        Ok(vlist.into_iter().next())
    }
}

/// RPython `bool_repr = BoolRepr()` (`rbool.py:36`).
///
/// Module-global singleton; pyre matches via `Arc<BoolRepr>` cached in
/// `OnceLock` so `is bool_repr` identity comparisons upstream relies on
/// (via `Arc::ptr_eq`) survive the port.
pub fn bool_repr() -> Arc<BoolRepr> {
    static REPR: OnceLock<Arc<BoolRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(BoolRepr::new())).clone()
}

// ____________________________________________________________
// pairtype(BoolRepr, X) conversions — rbool.py:49-84.
//
// Upstream keeps each `class __extend__(pairtype(R_A, R_B))` block as
// a separate metaclass scope, so the four `convert_from_to` methods
// (one per pair) share a name without colliding. Rust file-level
// functions need unique identifiers; the `pair_<left>_<right>_<op>`
// naming convention (also used in rnone.rs / rint.rs) disambiguates
// without renaming the upstream method.

pub fn pair_bool_float_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.lowleveltype() == &LowLevelType::Bool && r_to.lowleveltype() == &LowLevelType::Float {
        return Ok(llops
            .genop(
                "cast_bool_to_float",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Float),
            )
            .map(Hlvalue::Variable));
    }
    Ok(None)
}

pub fn pair_float_bool_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.lowleveltype() == &LowLevelType::Float && r_to.lowleveltype() == &LowLevelType::Bool {
        return Ok(llops
            .genop(
                "float_is_true",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Bool),
            )
            .map(Hlvalue::Variable));
    }
    Ok(None)
}

pub fn pair_bool_integer_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_from.lowleveltype() != &LowLevelType::Bool {
        return Ok(None);
    }
    match r_to.lowleveltype() {
        LowLevelType::Unsigned => Ok(llops
            .genop(
                "cast_bool_to_uint",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Unsigned),
            )
            .map(Hlvalue::Variable)),
        LowLevelType::Signed => Ok(llops
            .genop(
                "cast_bool_to_int",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Signed),
            )
            .map(Hlvalue::Variable)),
        _ => {
            let v_int = llops
                .genop(
                    "cast_bool_to_int",
                    vec![v.clone()],
                    GenopResult::LLType(LowLevelType::Signed),
                )
                .map(Hlvalue::Variable)
                .ok_or_else(|| {
                    TyperError::message("cast_bool_to_int unexpectedly returned Void")
                })?;
            let signed = super::rint::signed_repr();
            llops
                .convertvar(v_int, signed.as_ref(), r_to)
                .map(|converted| Some(converted))
        }
    }
}

pub fn pair_integer_bool_convert_from_to(
    r_from: &dyn Repr,
    r_to: &dyn Repr,
    v: &Hlvalue,
    llops: &mut LowLevelOpList,
) -> Result<Option<Hlvalue>, TyperError> {
    if r_to.lowleveltype() != &LowLevelType::Bool {
        return Ok(None);
    }
    match r_from.lowleveltype() {
        LowLevelType::Unsigned => Ok(llops
            .genop(
                "uint_is_true",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Bool),
            )
            .map(Hlvalue::Variable)),
        LowLevelType::Signed => Ok(llops
            .genop(
                "int_is_true",
                vec![v.clone()],
                GenopResult::LLType(LowLevelType::Bool),
            )
            .map(Hlvalue::Variable)),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::translator::rtyper::rmodel::Setupstate;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    #[test]
    fn bool_repr_lowleveltype_is_bool_and_repr_string_matches_upstream() {
        // rbool.py:11 — `lowleveltype = Bool`.
        let r = BoolRepr::new();
        assert_eq!(r.lowleveltype(), &LowLevelType::Bool);
        // rmodel.py:30 `<%s %s>` formatter — `<BoolRepr Bool>`.
        assert_eq!(r.repr_string(), "<BoolRepr Bool>");
        // rmodel.py:33 compact_repr — "BoolRepr" → "BoolR" replacement
        // followed by short_name.
        assert_eq!(r.compact_repr(), "BoolR Bool");
    }

    #[test]
    fn bool_repr_singleton_returns_same_arc() {
        // rbool.py:36 — `bool_repr = BoolRepr()` module-global.
        let a = bool_repr();
        let b = bool_repr();
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn setup_on_bool_repr_reaches_finished_state() {
        // rmodel.py:35-59 state machine — BoolRepr inherits the default
        // `_setup_repr` (no-op).
        let r = BoolRepr::new();
        assert_eq!(r.state().get(), Setupstate::NotInitialized);
        r.setup().expect("BoolRepr.setup() should succeed");
        assert_eq!(r.state().get(), Setupstate::Finished);
    }

    #[test]
    fn convert_const_accepts_bool_values() {
        // rbool.py:17-20 — convert_const asserts `isinstance(value,
        // bool)` and returns the value (wrapped with Bool lowleveltype
        // via base Repr::convert_const's pyre adaptation).
        let r = BoolRepr::new();
        let c = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c.value, ConstValue::Bool(true));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Bool));
    }

    #[test]
    fn convert_const_rejects_non_bool() {
        // rbool.py:18-19 — `raise TyperError("not a bool: %r" % (value,))`.
        let r = BoolRepr::new();
        let err = r.convert_const(&ConstValue::Int(1)).unwrap_err();
        assert!(err.to_string().contains("not a bool"));
    }

    /// rbool.py inherits `IntegerRepr.get_ll_hash_function` (rint.py:50-54);
    /// for `Bool` lltype the synthesized helper widens via
    /// `cast_bool_to_int` (`intmask(b)` semantics) so the return value
    /// matches the helper's `Signed` return type. Without the cast the
    /// return link would carry a Bool-typed Variable into a
    /// Signed-typed slot and break downstream `int_mul`/`int_xor`
    /// mixing in `gen_hash_function`.
    #[test]
    fn bool_repr_get_ll_hash_function_widens_bool_to_signed_via_cast() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = bool_repr();

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .expect("get_ll_hash_function(Bool)")
            .expect("returns Some helper");
        assert_eq!(llfn.name, "ll_hash_int_Bool");
        assert_eq!(llfn.args, vec![LowLevelType::Bool]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().expect("helper carries a graph");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(
            opnames,
            vec!["cast_bool_to_int"],
            "Bool inputarg must widen to Signed before the helper returns, got {:?}",
            startblock.operations
        );
    }

    #[test]
    fn rtyper_getrepr_on_some_bool_returns_the_singleton() {
        // rbool.py:39-41 — `SomeBool.rtyper_makerepr` returns `bool_repr`.
        // Verify the full dispatch chain keeps singleton identity.
        use crate::annotator::model::{SomeBool, SomeValue};

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let s_bool = SomeValue::Bool(SomeBool::new());
        let r = rtyper.getrepr(&s_bool).expect("getrepr(SomeBool)");
        let expected = bool_repr() as Arc<dyn Repr>;
        assert!(Arc::ptr_eq(&r, &expected));

        // rtyper.py:54-57 cache sanity — second lookup must dedupe to
        // the same Arc.
        let r2 = rtyper.getrepr(&s_bool).expect("second getrepr(SomeBool)");
        assert!(Arc::ptr_eq(&r, &r2));
    }
}
