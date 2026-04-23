//! RPython `rpython/rtyper/rfloat.py` — `FloatRepr` / `SingleFloatRepr` /
//! `LongFloatRepr`.
//!
//! The upstream file defines the base float Repr, two wrapper Reprs for
//! `r_singlefloat` / `r_longfloat`, and a `pairtype(FloatRepr, FloatRepr)`
//! extension block that registers arithmetic / comparison `rtype_*`
//! methods. It also contributes three `__extend__(annmodel.SomeFloat /
//! SomeSingleFloat / SomeLongFloat)` blocks that wire each SomeXxx into
//! the rtyper dispatch.
//!
//! ## Parity scope of this commit
//!
//! Most low-level `rtype_*` methods in the upstream file take a
//! `HighLevelOp` and call `hop.inputargs` / `hop.genop`. Those call
//! sites assume three pieces of infrastructure that **land together in
//! Cascade step 2**:
//!
//! 1. `HighLevelOp::inputargs` / `inputarg` / `inputconst` —
//!    `rtyper.py:655-686`.
//! 2. `RPythonTyper::getprimitiverepr` + `primitive_to_repr` cache —
//!    `rtyper.py:85-93`.
//! 3. `LowLevelOpList::convertvar` + `pairtype(Repr, Repr).convert_from_to`
//!    double-dispatch — `rtyper.py:810-823`.
//!
//! The `SingleFloatRepr.rtype_float` / `LongFloatRepr.rtype_float`
//! wrappers are the one exception: upstream only needs the already
//! landed `inputargs(lltype.X)` + `exception_cannot_occur()` +
//! `genop('cast_primitive', ...)` subset, so pyre can port them
//! directly now. The remaining unary/binary float `rtype_*` bodies stay
//! deferred until the full translate-op dispatch lands.
//!
//! Everything that **does** land in this commit maps one-to-one to
//! upstream:
//!
//! | upstream line | pyre mirror |
//! |---|---|
//! | `class FloatRepr(Repr)` `rfloat.py:11` | [`FloatRepr`] |
//! | `FloatRepr.lowleveltype = Float` `rfloat.py:12` | `FloatRepr::lowleveltype` returns [`LowLevelType::Float`] |
//! | `FloatRepr.convert_const` `rfloat.py:14-17` | [`FloatRepr::convert_const`] |
//! | `FloatRepr.get_ll_*_function` `rfloat.py:19-27` | [`FloatRepr::get_ll_eq_function`] / `get_ll_hash_function` |
//! | `float_repr = FloatRepr()` `rfloat.py:65` | [`float_repr`] singleton |
//! | `class __extend__(annmodel.SomeFloat)` `rfloat.py:67-72` | [`super::rmodel::rtyper_makerepr`] `SomeFloat` arm |
//! | `class SingleFloatRepr(Repr)` `rfloat.py:150-158` | [`SingleFloatRepr`] |
//! | `class __extend__(annmodel.SomeSingleFloat)` `rfloat.py:144-148` | [`super::rmodel::rtyper_makerepr`] `SomeSingleFloat` arm |
//! | `class LongFloatRepr(Repr)` `rfloat.py:166-174` | [`LongFloatRepr`] |
//! | `class __extend__(annmodel.SomeLongFloat)` `rfloat.py:160-164` | [`super::rmodel::rtyper_makerepr`] `SomeLongFloat` arm |
//!
//! The upstream file's `_rtype_template` / `_rtype_compare_template`
//! free functions + the `pairtype(FloatRepr, FloatRepr)` block are
//! reserved for Cascade 2 with a module-level comment pointer so the
//! follow-up commit lands at the exact structural location upstream
//! uses.

use std::sync::{Arc, OnceLock};

use crate::flowspace::model::{ConstValue, Hlvalue};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{LlHelper, Repr, ReprState};
use crate::translator::rtyper::rtyper::{GenopResult, HighLevelOp, ReqType};

// ____________________________________________________________
// RPython `class FloatRepr(Repr)` (rfloat.py:11-63).

/// RPython `class FloatRepr(Repr)` (`rfloat.py:11-63`).
///
/// ```python
/// class FloatRepr(Repr):
///     lowleveltype = Float
///
///     def convert_const(self, value):
///         if not isinstance(value, (int, base_int, float)):  # can be bool too
///             raise TyperError("not a float: %r" % (value,))
///         return float(value)
///     ...
/// ```
///
/// Upstream is the superclass of `IntegerRepr`. In Rust we cannot
/// inherit trait methods from a struct; each concrete Repr reproduces
/// the relevant overrides. The `IntegerRepr` / `BoolRepr` ports in
/// Cascade 1b / 1c land in `rint.rs` / `rbool.rs` with their own
/// impls, each pointing back at the shared `rfloat.rs` class name for
/// parity tracing.
#[derive(Debug)]
pub struct FloatRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl FloatRepr {
    /// Upstream `FloatRepr()` no-arg constructor — all fields come
    /// from class attributes (`rfloat.py:12`).
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "FloatRepr"
    }

    /// RPython `FloatRepr.convert_const` (`rfloat.py:14-17`).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, (int, base_int, float)):  # can be bool too
    ///         raise TyperError("not a float: %r" % (value,))
    ///     return float(value)
    /// ```
    ///
    /// Upstream's `base_int` covers `r_uint`, `r_longlong`, ... — pyre
    /// represents them as [`ConstValue::Int`] today (distinguished by
    /// the source SomeInteger's `KnownType`, not at the constant
    /// level), so the accepting branch matches `Int` / `Float` / `Bool`.
    ///
    /// Upstream `return float(value)` performs the actual numeric cast;
    /// pyre mirrors by normalising Int/Bool to [`ConstValue::Float`]
    /// before wrapping, so the downstream `emit_const_f` consumer never
    /// sees a non-Float `ConstValue` under a `Float` concretetype.
    fn convert_const(
        &self,
        value: &ConstValue,
    ) -> Result<crate::flowspace::model::Constant, TyperError> {
        use crate::flowspace::model::Constant;
        let casted = match value {
            // upstream `isinstance(value, (int, base_int, float))` — pyre
            // lumps int-like values under `ConstValue::Int` and
            // float-like under `ConstValue::Float`. Bool is explicit
            // upstream ("can be bool too" comment) so it passes too.
            ConstValue::Float(_) => value.clone(),
            ConstValue::Int(n) => ConstValue::float(*n as f64),
            ConstValue::Bool(b) => ConstValue::float(if *b { 1.0 } else { 0.0 }),
            other => return Err(TyperError::message(format!("not a float: {other:?}"))),
        };
        Ok(Constant::with_concretetype(casted, self.lltype.clone()))
    }
}

impl FloatRepr {
    /// RPython `FloatRepr.get_ll_eq_function` (`rfloat.py:19-24`).
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return None
    /// get_ll_gt_function = get_ll_eq_function
    /// get_ll_lt_function = get_ll_eq_function
    /// get_ll_ge_function = get_ll_eq_function
    /// get_ll_le_function = get_ll_eq_function
    /// ```
    ///
    /// Upstream returns `None` to signal "no custom eq helper, use the
    /// primitive op". Pyre mirrors with an `Option<()>` sentinel — the
    /// follow-up dict/hash ports will widen this to return a function
    /// pointer when non-default helpers exist (see `IntegerRepr`'s
    /// `ll_eq_shortint` for the first non-None case).
    pub fn get_ll_eq_function(&self) -> Option<()> {
        None
    }

    /// RPython `FloatRepr.get_ll_gt_function` — aliased to `get_ll_eq_function`
    /// upstream.
    pub fn get_ll_gt_function(&self) -> Option<()> {
        self.get_ll_eq_function()
    }

    /// RPython `FloatRepr.get_ll_lt_function`.
    pub fn get_ll_lt_function(&self) -> Option<()> {
        self.get_ll_eq_function()
    }

    /// RPython `FloatRepr.get_ll_ge_function`.
    pub fn get_ll_ge_function(&self) -> Option<()> {
        self.get_ll_eq_function()
    }

    /// RPython `FloatRepr.get_ll_le_function`.
    pub fn get_ll_le_function(&self) -> Option<()> {
        self.get_ll_eq_function()
    }

    /// RPython `FloatRepr.get_ll_hash_function` (`rfloat.py:26-27`).
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return _hash_float
    /// ```
    ///
    /// Upstream returns the `_hash_float` function pointer from
    /// `rpython.rlib.objectmodel`. Pyre returns a typed helper
    /// descriptor so downstream code can switch on helper identity
    /// without stringly-typed matching.
    pub fn get_ll_hash_function(&self) -> LlHelper {
        LlHelper::HashFloat
    }

    // RPython inline comment (`rfloat.py:29-30`):
    //     # no get_ll_fasthash_function: the hash is a bit slow, better cache
    //     # it inside dict entries
}

// ____________________________________________________________
// `pairtype(FloatRepr, FloatRepr).rtype_*` block (rfloat.py:75-125) +
// helper functions `_rtype_template` / `_rtype_compare_template`
// (rfloat.py:129-135).
//
// Cascade 2 ("translate_hl_to_ll + pairtype dispatch") lands the
// per-op rtype_* surface on this file. Upstream defines:
//
// class __extend__(pairtype(FloatRepr, FloatRepr)):
//     # Arithmetic
//     def rtype_add(_, hop):            return _rtype_template(hop, 'add')
//     def rtype_sub(_, hop):            return _rtype_template(hop, 'sub')
//     def rtype_mul(_, hop):            return _rtype_template(hop, 'mul')
//     def rtype_truediv(_, hop):        return _rtype_template(hop, 'truediv')
//     rtype_inplace_add = rtype_add
//     rtype_inplace_sub = rtype_sub
//     rtype_inplace_mul = rtype_mul
//     rtype_inplace_truediv = rtype_truediv
//     rtype_div         = rtype_truediv
//     rtype_inplace_div = rtype_inplace_truediv
//     # Comparisons
//     def rtype_eq(_, hop):             return _rtype_compare_template(hop, 'eq')
//     rtype_is_ = rtype_eq
//     def rtype_ne(_, hop):             return _rtype_compare_template(hop, 'ne')
//     def rtype_lt(_, hop):             return _rtype_compare_template(hop, 'lt')
//     def rtype_le(_, hop):             return _rtype_compare_template(hop, 'le')
//     def rtype_gt(_, hop):             return _rtype_compare_template(hop, 'gt')
//     def rtype_ge(_, hop):             return _rtype_compare_template(hop, 'ge')
//
// def _rtype_template(hop, func):
//     vlist = hop.inputargs(Float, Float)
//     return hop.genop('float_'+func, vlist, resulttype=Float)
//
// def _rtype_compare_template(hop, func):
//     vlist = hop.inputargs(Float, Float)
//     return hop.genop('float_'+func, vlist, resulttype=Bool)
//
// Single-arg FloatRepr rtype_* methods (rfloat.py:32-58): rtype_bool,
// rtype_neg, rtype_pos, rtype_abs, rtype_int, rtype_float. All land
// together in Cascade 2.

/// Singleton accessor — `float_repr = FloatRepr()` (`rfloat.py:65`).
///
/// Upstream treats `float_repr` as a module-global instance; every
/// SomeFloat resolves to this exact repr. Pyre stores in a `OnceLock`
/// so downstream `Arc::ptr_eq(r, float_repr())` matches upstream's
/// `r is float_repr` identity semantics.
pub fn float_repr() -> Arc<FloatRepr> {
    static REPR: OnceLock<Arc<FloatRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(FloatRepr::new())).clone()
}

// ____________________________________________________________
// RPython `class SingleFloatRepr(Repr)` (rfloat.py:150-158).
//
// Upstream uses a dedicated class rather than aliasing FloatRepr
// because `lowleveltype = lltype.SingleFloat` and the only supported
// `rtype_*` is `rtype_float` (cast to Float via cast_primitive).

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

    /// RPython `SingleFloatRepr.rtype_float` (`rfloat.py:153-158`).
    pub fn rtype_float(&self, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        let mut vlist = hop.inputargs(vec![ReqType::LLType(LowLevelType::SingleFloat)])?;
        hop.exception_cannot_occur()?;
        let v = vlist
            .pop()
            .expect("SingleFloatRepr.rtype_float must receive one inputarg");
        let out = hop
            .genop(
                "cast_primitive",
                vec![v],
                GenopResult::LLType(LowLevelType::Float),
            )
            .expect("cast_primitive should produce a Float result");
        Ok(Hlvalue::Variable(out))
    }
}

impl Default for SingleFloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for SingleFloatRepr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "SingleFloatRepr"
    }

    // RPython `SingleFloatRepr` inherits `convert_const` from `Repr`
    // base (`rfloat.py:150-158` has no override). The default
    // `Repr::convert_const` in `rmodel.rs` now forwards to
    // `LowLevelType::contains_value(ConstValue::SingleFloat)`,
    // matching upstream's Primitive-level _enforce.
}

/// Singleton accessor — RPython uses fresh `SingleFloatRepr()` per
/// call (`rfloat.py:145-146`). Pyre caches a single instance so the
/// reprs-cache identity matches upstream; the cache key already
/// collapses identical `rtyper_makekey` tuples so upstream's per-call
/// semantics are behaviour-equivalent.
pub fn single_float_repr() -> Arc<SingleFloatRepr> {
    static REPR: OnceLock<Arc<SingleFloatRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(SingleFloatRepr::new()))
        .clone()
}

// ____________________________________________________________
// RPython `class LongFloatRepr(Repr)` (rfloat.py:166-174).

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

    /// RPython `LongFloatRepr.rtype_float` (`rfloat.py:169-174`).
    pub fn rtype_float(&self, hop: &HighLevelOp) -> Result<Hlvalue, TyperError> {
        let mut vlist = hop.inputargs(vec![ReqType::LLType(LowLevelType::LongFloat)])?;
        hop.exception_cannot_occur()?;
        let v = vlist
            .pop()
            .expect("LongFloatRepr.rtype_float must receive one inputarg");
        let out = hop
            .genop(
                "cast_primitive",
                vec![v],
                GenopResult::LLType(LowLevelType::Float),
            )
            .expect("cast_primitive should produce a Float result");
        Ok(Hlvalue::Variable(out))
    }
}

impl Default for LongFloatRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for LongFloatRepr {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "LongFloatRepr"
    }
}

/// Singleton accessor — RPython uses fresh `LongFloatRepr()` per call
/// (`rfloat.py:161-162`). Same caching rationale as
/// [`single_float_repr`].
pub fn long_float_repr() -> Arc<LongFloatRepr> {
    static REPR: OnceLock<Arc<LongFloatRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(LongFloatRepr::new())).clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeLongFloat, SomeSingleFloat, SomeValue};
    use crate::flowspace::model::{ConstValue, Hlvalue, SpaceOperation, Variable};
    use crate::translator::rtyper::rtyper::{HighLevelOp, LowLevelOpList, RPythonTyper};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn float_repr_is_singleton_with_float_lowleveltype() {
        // rfloat.py:12 lowleveltype = Float. rfloat.py:65 module-level
        // `float_repr = FloatRepr()` — identity matters because the
        // reprs-cache uses `is` comparison upstream (rtyper.py:152).
        let a = float_repr();
        let b = float_repr();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.lowleveltype(), &LowLevelType::Float);
        assert_eq!(a.class_name(), "FloatRepr");
    }

    #[test]
    fn float_repr_convert_const_accepts_int_float_bool() {
        // rfloat.py:14-17 — isinstance(value, (int, base_int, float))
        // passes for int / bool / float (upstream comment "can be bool
        // too").
        let r = FloatRepr::new();
        assert!(r.convert_const(&ConstValue::Int(7)).is_ok());
        assert!(r.convert_const(&ConstValue::Float(0)).is_ok());
        assert!(r.convert_const(&ConstValue::Bool(false)).is_ok());
        // Constants that are not numeric should raise TyperError.
        let err = r.convert_const(&ConstValue::None).unwrap_err();
        assert!(err.to_string().contains("not a float"));
        let err = r.convert_const(&ConstValue::Placeholder).unwrap_err();
        assert!(err.to_string().contains("not a float"));
    }

    #[test]
    fn float_repr_convert_const_casts_int_and_bool_to_float_value() {
        // rfloat.py:17 `return float(value)` — the cast semantics must
        // materialize in the resulting Constant's payload, not only in
        // the `concretetype` label. Before this parity fix the Int/Bool
        // payload was preserved as-is which made downstream `emit_const_f`
        // see a non-Float ConstValue under a Float concretetype.
        let r = FloatRepr::new();
        let c_int = r.convert_const(&ConstValue::Int(7)).unwrap();
        assert_eq!(c_int.value, ConstValue::float(7.0));
        assert_eq!(c_int.concretetype.as_ref(), Some(&LowLevelType::Float));

        let c_true = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c_true.value, ConstValue::float(1.0));

        let c_false = r.convert_const(&ConstValue::Bool(false)).unwrap();
        assert_eq!(c_false.value, ConstValue::float(0.0));

        // Float input is preserved bit-for-bit (no cast needed).
        let c_fl = r.convert_const(&ConstValue::float(2.5)).unwrap();
        assert_eq!(c_fl.value, ConstValue::float(2.5));
    }

    #[test]
    fn float_repr_get_ll_comparison_functions_return_none() {
        // rfloat.py:19-24 all four comparison accessors alias to
        // get_ll_eq_function == None. The Option<()> sentinel holds
        // until the ll helper-fn pointer port lands.
        let r = FloatRepr::new();
        assert!(r.get_ll_eq_function().is_none());
        assert!(r.get_ll_gt_function().is_none());
        assert!(r.get_ll_lt_function().is_none());
        assert!(r.get_ll_ge_function().is_none());
        assert!(r.get_ll_le_function().is_none());
    }

    #[test]
    fn float_repr_hash_function_matches_upstream_import_name() {
        // rfloat.py:26-27 `return _hash_float`.
        let r = FloatRepr::new();
        assert_eq!(r.get_ll_hash_function(), LlHelper::HashFloat);
    }

    #[test]
    fn single_float_repr_uses_single_float_lowleveltype() {
        // rfloat.py:150-151.
        let r = single_float_repr();
        assert_eq!(r.lowleveltype(), &LowLevelType::SingleFloat);
        assert_eq!(r.class_name(), "SingleFloatRepr");
    }

    #[test]
    fn long_float_repr_uses_long_float_lowleveltype() {
        // rfloat.py:166-167.
        let r = long_float_repr();
        assert_eq!(r.lowleveltype(), &LowLevelType::LongFloat);
        assert_eq!(r.class_name(), "LongFloatRepr");
    }

    fn hop_for_unary_float_constant(
        opname: &str,
        arg: Hlvalue,
        annotation: SomeValue,
    ) -> (Rc<RPythonTyper>, HighLevelOp, Rc<RefCell<LowLevelOpList>>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper_rc = Rc::new(RPythonTyper::new(&ann));
        let weak = Rc::downgrade(&rtyper_rc);
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(weak.clone(), None)));
        let spaceop = SpaceOperation::new(
            opname.to_string(),
            vec![arg.clone()],
            Hlvalue::Variable(Variable::new()),
        );
        let hop = HighLevelOp::new(weak, spaceop, Vec::new(), llops.clone());
        hop.args_v.replace(vec![arg]);
        hop.args_s.replace(vec![annotation]);
        hop.args_r.replace(vec![None]);
        (rtyper_rc, hop, llops)
    }

    #[test]
    fn single_float_repr_rtype_float_emits_cast_primitive_to_float() {
        let arg = Hlvalue::Constant(crate::flowspace::model::Constant::new(
            ConstValue::single_float(2.1),
        ));
        let mut s_arg = SomeSingleFloat::new();
        s_arg.base.const_box = match &arg {
            Hlvalue::Constant(c) => Some(c.clone()),
            Hlvalue::Variable(_) => None,
        };
        let (_rtyper, hop, llops) =
            hop_for_unary_float_constant("float", arg, SomeValue::SingleFloat(s_arg));

        let out = SingleFloatRepr::new().rtype_float(&hop).unwrap();
        let Hlvalue::Variable(out_v) = out else {
            panic!("rtype_float should return a Variable");
        };
        assert_eq!(out_v.concretetype.as_ref(), Some(&LowLevelType::Float));
        assert_eq!(llops.borrow().ops.len(), 1);
        assert_eq!(llops.borrow().ops[0].opname, "cast_primitive");
        assert!(llops.borrow()._called_exception_is_here_or_cannot_occur);
    }

    #[test]
    fn long_float_repr_rtype_float_emits_cast_primitive_to_float() {
        let arg = Hlvalue::Constant(crate::flowspace::model::Constant::new(
            ConstValue::long_float(3.25),
        ));
        let mut s_arg = SomeLongFloat::new();
        s_arg.base.const_box = match &arg {
            Hlvalue::Constant(c) => Some(c.clone()),
            Hlvalue::Variable(_) => None,
        };
        let (_rtyper, hop, llops) =
            hop_for_unary_float_constant("float", arg, SomeValue::LongFloat(s_arg));

        let out = LongFloatRepr::new().rtype_float(&hop).unwrap();
        let Hlvalue::Variable(out_v) = out else {
            panic!("rtype_float should return a Variable");
        };
        assert_eq!(out_v.concretetype.as_ref(), Some(&LowLevelType::Float));
        assert_eq!(llops.borrow().ops.len(), 1);
        assert_eq!(llops.borrow().ops[0].opname, "cast_primitive");
        assert!(llops.borrow()._called_exception_is_here_or_cannot_occur);
    }

    #[test]
    fn float_repr_state_starts_notinitialized_and_transitions_to_finished() {
        // Repr base state machine (rmodel.py:35-59) applies to every
        // concrete subclass, including FloatRepr. Verify the scaffold
        // does not short-circuit the state transition.
        use crate::translator::rtyper::rmodel::Setupstate;
        let r = FloatRepr::new();
        assert_eq!(r.state().get(), Setupstate::NotInitialized);
        r.setup().unwrap();
        assert_eq!(r.state().get(), Setupstate::Finished);
    }
}
