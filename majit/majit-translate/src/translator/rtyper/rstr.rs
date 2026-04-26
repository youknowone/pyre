//! RPython `rpython/rtyper/rstr.py` — Char / UniChar Repr (minimal slice).
//!
//! Upstream rstr.py is ~1500 LOC covering `AbstractCharRepr`,
//! `AbstractUniCharRepr`, `AbstractStringRepr`, `AbstractUnicodeRepr`,
//! plus their lltypesystem / ootypesystem realisations and the dense
//! pairtype dispatch surface (eq/ne/lt/le/gt/ge/add/mul/contains/in
//! /str/repr/encode/decode/...). Pyre lands the **minimal slice**
//! required to unblock tuple eq/hash with Char/UniChar items:
//!
//! | upstream | Rust mirror |
//! |---|---|
//! | `class AbstractCharRepr` (`rstr.py:483-541`) | [`CharRepr`] |
//! | `class AbstractUniCharRepr` (`rstr.py:758-775`) | [`UniCharRepr`] |
//! | `AbstractCharRepr.get_ll_eq_function` (`rstr.py:496-497`) | [`Repr::get_ll_eq_function`] impl |
//! | `AbstractCharRepr.get_ll_hash_function` (`rstr.py:499-500`) | [`Repr::get_ll_hash_function`] impl + [`build_ll_char_hash_helper_graph`] |
//! | `AbstractUniCharRepr.get_ll_hash_function` (`rstr.py:767-768`) | [`Repr::get_ll_hash_function`] impl + [`build_ll_unichar_hash_helper_graph`] |
//! | `ll_char_hash` / `ll_unichar_hash` (`rstr.py:937-942`) | helper graph: `cast_char_to_int(ch)` / `cast_unichar_to_int(ch)` |
//! | `SomeChar.rtyper_makerepr` / `SomeUnicodeCodePoint.rtyper_makerepr` (`rstr.py:589-598`) | wired in [`super::rmodel::rtyper_makerepr`] |
//!
//! ## Deferred to follow-up commits
//!
//! * `rtype_str` / `rtype_chr` / `rtype_unichr` / `rtype_int` / `rtype_float`
//!   (rstr.py:516-541, 772-784) — char-side conversions that lower
//!   non-trivial usages and have no caller on the tuple eq/hash path.
//! * `AbstractCharRepr.ll_str` / `AbstractUniCharRepr.ll_str`
//!   (rstr.py:554-562) — chr→str conversion (allocates GC string).
//! * Full `AbstractStringRepr` / `AbstractUnicodeRepr` (rstr.py:67-481,
//!   543-737) — string types are GC-managed Ptr structs; their hash
//!   helper `ll_strhash` mirrors CPython's algorithm. Larger porting
//!   epic.
//! * `ConstValue::Str` byte/unicode tag — pyre's `ConstValue::Str(_)`
//!   does not distinguish Python 2 byte strings from unicode, so
//!   [`CharRepr::convert_const`] / [`UniCharRepr::convert_const`]
//!   currently both accept any `len() == 1` `ConstValue::Str`.
//!   Mirroring rstr.py:491-494 / 757-762 byte/unicode discrimination
//!   requires a separate `ConstValue::ByteStr` / `ConstValue::UniStr`
//!   split (cross-crate infrastructure).

use std::sync::Arc;
use std::sync::OnceLock;

use crate::flowspace::model::{
    Block, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc, Hlvalue, Link,
    SpaceOperation,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{RTypeResult, Repr, ReprState};
use crate::translator::rtyper::rtyper::{
    ConvertedTo, GenopResult, HighLevelOp, LowLevelFunction, RPythonTyper, constant_with_lltype,
    helper_pygraph_from_graph, variable_with_lltype,
};

// ____________________________________________________________
// CharRepr — `rstr.py:483-541` (lltypesystem-bound `AbstractCharRepr`).

/// RPython `class AbstractCharRepr(AbstractStringRepr, AbstractCharRepr_)`
/// (`rstr.py:483-541`) — the lltypesystem `CharRepr` carries `lowleveltype = Char`.
///
/// Pyre lands a single concrete `CharRepr` since the
/// abstract/lltypesystem split is an upstream artefact of supporting
/// both `lltypesystem` and `ootypesystem`; pyre only targets
/// lltypesystem.
#[derive(Debug)]
pub struct CharRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl CharRepr {
    pub fn new() -> Self {
        CharRepr {
            state: ReprState::new(),
            lltype: LowLevelType::Char,
        }
    }
}

impl Default for CharRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for CharRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "CharRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::CharRepr
    }

    /// RPython `BaseCharReprMixin.convert_const(self, value)`
    /// (`rstr.py:491-494`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, str) or len(value) != 1:
    ///         raise TyperError("not a character: %r" % (value,))
    ///     return value
    /// ```
    ///
    /// Pyre maps `Char` lltype to a `ConstValue::Str` of length 1.
    /// Other ConstValue variants and non-1-len strings are rejected.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::Str(s) if s.chars().count() == 1 => Ok(Constant::with_concretetype(
                ConstValue::Str(s.clone()),
                LowLevelType::Char,
            )),
            other => Err(TyperError::message(format!("not a character: {other:?}"))),
        }
    }

    /// RPython `AbstractCharRepr.get_ll_eq_function` (`rstr.py:496-497`):
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     return None
    /// ```
    ///
    /// Returning `None` instructs callers (`gen_eq_function` /
    /// `rtype_contains`) to fall back to the primitive `char_eq`
    /// inline op via `eq_funcs[i] or operator.eq`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `AbstractCharRepr.get_ll_hash_function` (`rstr.py:499-500`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return self.ll.ll_char_hash
    ///
    /// def ll_char_hash(ch):
    ///     return ord(ch)
    /// ```
    ///
    /// Synthesizes the `ll_char_hash(ch) -> Signed` helper graph.
    /// Body: single block, `cast_char_to_int(ch) -> hashed` then close
    /// to returnblock.
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let name = "ll_char_hash".to_string();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![LowLevelType::Char],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| build_ll_char_hash_helper_graph(&name),
            )
            .map(Some)
    }

    /// RPython `BaseCharReprMixin.rtype_len(_, hop)` (`rstr.py:504-505`):
    /// `return hop.inputconst(Signed, 1)`. Single chars always carry
    /// length 1.
    fn rtype_len(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(1))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `BaseCharReprMixin.rtype_bool(_, hop)` (`rstr.py:507-509`):
    /// `assert not hop.args_s[0].can_be_None; return hop.inputconst(Bool, True)`.
    /// Pyre's CharRepr has lltype `Char` (not nullable in the lltype
    /// sense — `NoneRepr` would be a separate static type), so the
    /// `can_be_None` assert is structurally satisfied.
    fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Bool, &ConstValue::Bool(true))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// RPython `BaseCharReprMixin.rtype_ord(_, hop)` (`rstr.py:511-514`):
    ///
    /// ```python
    /// def rtype_ord(_, hop):
    ///     repr = hop.args_r[0].char_repr
    ///     vlist = hop.inputargs(repr)
    ///     return hop.genop('cast_char_to_int', vlist, resulttype=Signed)
    /// ```
    fn rtype_ord(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Char)])?;
        Ok(hop.genop(
            "cast_char_to_int",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    /// RPython `BaseCharReprMixin._rtype_method_isxxx(_, llfn, hop)`
    /// dispatch (`rstr.py:516-538`) — routes by method name to the
    /// per-predicate `ll_char_*` helper graph.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        // Simple inrange predicates (rstr.py:891-922).
        if let Some((llfn_name, lo, hi)) = match method_name {
            "isdigit" => Some(("ll_char_isdigit", 48, 57)),
            "isupper" => Some(("ll_char_isupper", 65, 90)),
            "islower" => Some(("ll_char_islower", 97, 122)),
            _ => None,
        } {
            return char_predicate_inrange_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::Char,
                "cast_char_to_int",
                lo,
                hi,
            );
        }

        // OR-of-conditions predicates (rstr.py:886-912).
        if let Some((llfn_name, conditions)) = match method_name {
            "isspace" => Some(("ll_char_isspace", ISSPACE_CONDITIONS)),
            "isalpha" => Some(("ll_char_isalpha", ISALPHA_CONDITIONS)),
            "isalnum" => Some(("ll_char_isalnum", ISALNUM_CONDITIONS)),
            _ => None,
        } {
            return char_predicate_or_of_conditions_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::Char,
                "cast_char_to_int",
                conditions,
            );
        }

        // ASCII case-folding (rstr.py:542-552 + 925-934). lower: 'A'..='Z'
        // → +32; upper: 'a'..='z' → -32. UniCharRepr does NOT define
        // these — Unicode case-folding semantics are out of scope.
        if let Some((llfn_name, lo, hi, offset)) = match method_name {
            "lower" => Some(("ll_lower_char", 65, 90, 32)),
            "upper" => Some(("ll_upper_char", 97, 122, -32)),
            _ => None,
        } {
            return char_case_fold_method(hop, llfn_name.to_string(), lo, hi, offset);
        }

        Err(TyperError::message(format!(
            "missing CharRepr.rtype_method_{method_name}"
        )))
    }
}

/// RPython `char_repr = CharRepr()` (`rstr.py:1009`) module-global.
pub fn char_repr() -> Arc<CharRepr> {
    static REPR: OnceLock<Arc<CharRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(CharRepr::new())).clone()
}

/// Synthesizes the `ll_char_hash(ch)` helper graph (`rstr.py:937-938`):
/// single block, `cast_char_to_int(ch) -> hashed` then return.
pub(crate) fn build_ll_char_hash_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    build_ll_charlike_hash_helper_graph(name, LowLevelType::Char, "cast_char_to_int")
}

// ____________________________________________________________
// UniCharRepr — `rstr.py:758-775` (lltypesystem-bound `AbstractUniCharRepr`).

/// RPython `class AbstractUniCharRepr(AbstractUnicodeRepr,
/// AbstractCharRepr_)` (`rstr.py:758-775`) — lltypesystem `UniCharRepr`
/// carries `lowleveltype = UniChar`.
#[derive(Debug)]
pub struct UniCharRepr {
    state: ReprState,
    lltype: LowLevelType,
}

impl UniCharRepr {
    pub fn new() -> Self {
        UniCharRepr {
            state: ReprState::new(),
            lltype: LowLevelType::UniChar,
        }
    }
}

impl Default for UniCharRepr {
    fn default() -> Self {
        Self::new()
    }
}

impl Repr for UniCharRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "UniCharRepr"
    }

    fn repr_class_id(&self) -> super::pairtype::ReprClassId {
        super::pairtype::ReprClassId::UniCharRepr
    }

    /// RPython `AbstractUniCharRepr.convert_const` (`rstr.py:759-762`):
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if not isinstance(value, unicode) or len(value) != 1:
    ///         raise TyperError("not a unicode character: %r" % (value,))
    ///     return value
    /// ```
    ///
    /// Pyre maps `UniChar` lltype to a `ConstValue::Str` containing a
    /// single Unicode scalar value (Python 3 collapses `str` and
    /// `unicode`; the distinguisher is the target lltype).
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        match value {
            ConstValue::Str(s) if s.chars().count() == 1 => Ok(Constant::with_concretetype(
                ConstValue::Str(s.clone()),
                LowLevelType::UniChar,
            )),
            other => Err(TyperError::message(format!(
                "not a unicode character: {other:?}"
            ))),
        }
    }

    /// RPython `AbstractUniCharRepr.get_ll_eq_function` (`rstr.py:764-765`):
    /// `return None` — callers fall back to primitive `unichar_eq`.
    fn get_ll_eq_function(
        &self,
        _rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        Ok(None)
    }

    /// RPython `AbstractUniCharRepr.get_ll_hash_function` (`rstr.py:767-768`):
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     return self.ll.ll_unichar_hash
    ///
    /// def ll_unichar_hash(ch):
    ///     return ord(ch)
    /// ```
    fn get_ll_hash_function(
        &self,
        rtyper: &RPythonTyper,
    ) -> Result<Option<LowLevelFunction>, TyperError> {
        let name = "ll_unichar_hash".to_string();
        rtyper
            .lowlevel_helper_function_with_builder(
                name.clone(),
                vec![LowLevelType::UniChar],
                LowLevelType::Signed,
                move |_rtyper, _args, _result| build_ll_unichar_hash_helper_graph(&name),
            )
            .map(Some)
    }

    /// `BaseCharReprMixin.rtype_len` (`rstr.py:504-505`).
    fn rtype_len(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(1))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// `BaseCharReprMixin.rtype_bool` (`rstr.py:507-509`).
    fn rtype_bool(&self, _hop: &HighLevelOp) -> RTypeResult {
        let c = HighLevelOp::inputconst(&LowLevelType::Bool, &ConstValue::Bool(true))?;
        Ok(Some(Hlvalue::Constant(c)))
    }

    /// `BaseCharReprMixin.rtype_ord` (`rstr.py:772-775`) — UniChar variant
    /// uses the `cast_unichar_to_int` op.
    fn rtype_ord(&self, hop: &HighLevelOp) -> RTypeResult {
        let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::UniChar)])?;
        Ok(hop.genop(
            "cast_unichar_to_int",
            vlist,
            GenopResult::LLType(LowLevelType::Signed),
        ))
    }

    /// RPython `BaseCharReprMixin._rtype_method_isxxx(_, llfn, hop)`
    /// dispatch (`rstr.py:516-538`) — UniCharRepr inherits the same
    /// mixin, so the predicate routes go through `cast_unichar_to_int`
    /// per-predicate `ll_unichar_*` helpers.
    fn rtype_method(&self, method_name: &str, hop: &HighLevelOp) -> RTypeResult {
        if let Some((llfn_name, lo, hi)) = match method_name {
            "isdigit" => Some(("ll_unichar_isdigit", 48, 57)),
            "isupper" => Some(("ll_unichar_isupper", 65, 90)),
            "islower" => Some(("ll_unichar_islower", 97, 122)),
            _ => None,
        } {
            return char_predicate_inrange_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                lo,
                hi,
            );
        }

        if let Some((llfn_name, conditions)) = match method_name {
            "isspace" => Some(("ll_unichar_isspace", ISSPACE_CONDITIONS)),
            "isalpha" => Some(("ll_unichar_isalpha", ISALPHA_CONDITIONS)),
            "isalnum" => Some(("ll_unichar_isalnum", ISALNUM_CONDITIONS)),
            _ => None,
        } {
            return char_predicate_or_of_conditions_method(
                hop,
                llfn_name.to_string(),
                LowLevelType::UniChar,
                "cast_unichar_to_int",
                conditions,
            );
        }

        Err(TyperError::message(format!(
            "missing UniCharRepr.rtype_method_{method_name}"
        )))
    }
}

/// RPython `unichar_repr = UniCharRepr()` (`rstr.py:1010`) module-global.
pub fn unichar_repr() -> Arc<UniCharRepr> {
    static REPR: OnceLock<Arc<UniCharRepr>> = OnceLock::new();
    REPR.get_or_init(|| Arc::new(UniCharRepr::new())).clone()
}

/// Synthesizes the `ll_unichar_hash(ch)` helper graph (`rstr.py:941-942`):
/// single block, `cast_unichar_to_int(ch) -> hashed` then return.
pub(crate) fn build_ll_unichar_hash_helper_graph(name: &str) -> Result<PyGraph, TyperError> {
    build_ll_charlike_hash_helper_graph(name, LowLevelType::UniChar, "cast_unichar_to_int")
}

// ____________________________________________________________
// pairtype(AbstractCharRepr, AbstractCharRepr) — `rstr.py:740-746`.
// Six comparison ops (eq/ne/lt/le/gt/ge) all dispatch to the
// per-name lloperation `char_<func>`.

/// RPython `_rtype_compare_template(hop, func)` (`rstr.py:750-753`):
///
/// ```python
/// def _rtype_compare_template(hop, func):
///     vlist = hop.inputargs(char_repr, char_repr)
///     return hop.genop('char_' + func, vlist, resulttype=Bool)
/// ```
pub fn pair_char_char_rtype_compare(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::Char),
        ConvertedTo::LowLevelType(&LowLevelType::Char),
    ])?;
    let opname = format!("char_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

// ____________________________________________________________
// pairtype(AbstractUniCharRepr, AbstractUniCharRepr) — `rstr.py:778-784`.
// `rtype_eq` / `rtype_ne` use the lloperations `unichar_eq` /
// `unichar_ne`; `rtype_lt|le|gt|ge` cast both args through
// `cast_unichar_to_int` and dispatch to `int_<func>`.

/// RPython `_rtype_unchr_compare_template(hop, func)` (`rstr.py:789-792`):
///
/// ```python
/// def _rtype_unchr_compare_template(hop, func):
///     vlist = hop.inputargs(unichar_repr, unichar_repr)
///     return hop.genop('unichar_' + func, vlist, resulttype=Bool)
/// ```
pub fn pair_unichar_unichar_rtype_compare_eqne(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
    ])?;
    let opname = format!("unichar_{func}");
    Ok(hop.genop(&opname, vlist, GenopResult::LLType(LowLevelType::Bool)))
}

/// RPython `_rtype_unchr_compare_template_ord(hop, func)` (`rstr.py:794-800`):
///
/// ```python
/// def _rtype_unchr_compare_template_ord(hop, func):
///     vlist = hop.inputargs(*hop.args_r)
///     vlist2 = []
///     for v in vlist:
///         v = hop.genop('cast_unichar_to_int', [v], resulttype=lltype.Signed)
///         vlist2.append(v)
///     return hop.genop('int_' + func, vlist2, resulttype=Bool)
/// ```
pub fn pair_unichar_unichar_rtype_compare_ord(hop: &HighLevelOp, func: &str) -> RTypeResult {
    let vlist = hop.inputargs(vec![
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
        ConvertedTo::LowLevelType(&LowLevelType::UniChar),
    ])?;
    let mut vlist2 = Vec::with_capacity(vlist.len());
    for v in vlist {
        let casted = hop
            .genop(
                "cast_unichar_to_int",
                vec![v],
                GenopResult::LLType(LowLevelType::Signed),
            )
            .ok_or_else(|| {
                TyperError::message("cast_unichar_to_int genop did not produce a value")
            })?;
        vlist2.push(casted);
    }
    let opname = format!("int_{func}");
    Ok(hop.genop(&opname, vlist2, GenopResult::LLType(LowLevelType::Bool)))
}

// ____________________________________________________________
// Shared single-cast hash helper synthesizer — used by both CharRepr
// and UniCharRepr since their helper graphs are structurally identical
// modulo the cast op + arg lltype.

/// Shared `rtype_method_<predicate>` wrapper for the inrange-pattern
/// helpers (`isdigit/isupper/islower`). RPython
/// `BaseCharReprMixin._rtype_method_isxxx(_, llfn, hop)`
/// (`rstr.py:516-520`):
///
/// ```python
/// def _rtype_method_isxxx(_, llfn, hop):
///     repr = hop.args_r[0].char_repr
///     vlist = hop.inputargs(repr)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(llfn, vlist[0])
/// ```
fn char_predicate_inrange_method(
    hop: &HighLevelOp,
    llfn_name: String,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    lo: i64,
    hi: i64,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&arg_lltype)])?;
    hop.exception_cannot_occur()?;
    let helper_name = llfn_name.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        llfn_name,
        vec![arg_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_charlike_predicate_inrange_helper_graph(
                &helper_name,
                arg_lltype.clone(),
                cast_op,
                lo,
                hi,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// Shared `rtype_method_<predicate>` wrapper for `isspace/isalpha/
/// isalnum` — same call-site shape as `char_predicate_inrange_method`
/// but the helper synthesizer chains an OR of conditions (see
/// [`build_ll_charlike_or_of_conditions_helper_graph`]).
fn char_predicate_or_of_conditions_method(
    hop: &HighLevelOp,
    llfn_name: String,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    conditions: &'static [CharCondition],
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&arg_lltype)])?;
    hop.exception_cannot_occur()?;
    let helper_name = llfn_name.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        llfn_name,
        vec![arg_lltype.clone()],
        LowLevelType::Bool,
        move |_rtyper, _args, _result| {
            build_ll_charlike_or_of_conditions_helper_graph(
                &helper_name,
                arg_lltype.clone(),
                cast_op,
                conditions,
            )
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

/// Shared `rtype_method_<predicate>` wrapper for `lower/upper`. Mirrors
/// `AbstractCharRepr.rtype_method_lower/upper` (`rstr.py:542-552`):
///
/// ```python
/// def rtype_method_lower(self, hop):
///     char_repr = hop.args_r[0].char_repr
///     v_chr, = hop.inputargs(char_repr)
///     hop.exception_cannot_occur()
///     return hop.gendirectcall(self.ll.ll_lower_char, v_chr)
/// ```
fn char_case_fold_method(
    hop: &HighLevelOp,
    llfn_name: String,
    lo: i64,
    hi: i64,
    offset: i64,
) -> RTypeResult {
    let vlist = hop.inputargs(vec![ConvertedTo::LowLevelType(&LowLevelType::Char)])?;
    hop.exception_cannot_occur()?;
    let helper_name = llfn_name.clone();
    let helper = hop.rtyper.lowlevel_helper_function_with_builder(
        llfn_name,
        vec![LowLevelType::Char],
        LowLevelType::Char,
        move |_rtyper, _args, _result| {
            build_ll_char_case_fold_helper_graph(&helper_name, lo, hi, offset)
        },
    )?;
    hop.gendirectcall(&helper, vlist)
}

// rstr.py:886-912 — predicate condition tables.
// `ll_char_isspace`: `c == 32 or 9 <= c <= 13`.
const ISSPACE_CONDITIONS: &[CharCondition] =
    &[CharCondition::Eq(32), CharCondition::InRange(9, 13)];
// `ll_char_isalpha`: `c >= 97 ? c <= 122 : 65 <= c <= 90`.
// Equivalent OR form: `97 <= c <= 122 or 65 <= c <= 90`.
const ISALPHA_CONDITIONS: &[CharCondition] = &[
    CharCondition::InRange(97, 122),
    CharCondition::InRange(65, 90),
];
// `ll_char_isalnum`: digit or upper-alpha or lower-alpha (rstr.py:903-912 nested form).
const ISALNUM_CONDITIONS: &[CharCondition] = &[
    CharCondition::InRange(48, 57),
    CharCondition::InRange(65, 90),
    CharCondition::InRange(97, 122),
];

/// One condition in an OR-of-predicates check (`ll_char_isspace`,
/// `ll_char_isalpha`, `ll_char_isalnum`, ...).
///
/// - `Eq(n)` — `ord(ch) == n`. One check block (1 `int_eq`).
/// - `InRange(lo, hi)` — `lo <= ord(ch) <= hi`. Two check blocks
///   (`int_ge` then `int_le`).
#[derive(Clone, Copy, Debug)]
enum CharCondition {
    Eq(i64),
    InRange(i64, i64),
}

/// Synthesizes a `ll_<charlike>_<predicate>(ch) -> Bool` helper graph
/// whose body is a left-to-right short-circuit `c0 OR c1 OR ...` over
/// per-condition checks against `cast_<arg>_to_int(ch)`. Mirrors
/// RPython source-level short-circuit `or` (`rstr.py:886-912`).
///
/// CFG layout:
/// - **start**: `c = cast(ch)`, then evaluate `predicates[0]`
///   (`int_eq` or `int_ge`).
/// - For each `predicates[i]`, on True branch return `Bool(true)`;
///   on False branch fall through to `predicates[i+1]`'s entry
///   block (taking `c` as link arg) or, for the last predicate,
///   return `Bool(false)`.
/// - `InRange(lo, hi)` predicates have a secondary check block
///   `block_check_hi` for the `int_le(c, hi)` test.
fn build_ll_charlike_or_of_conditions_helper_graph(
    name: &str,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    predicates: &[CharCondition],
) -> Result<PyGraph, TyperError> {
    assert!(
        !predicates.is_empty(),
        "or-of-predicates helper requires at least one condition"
    );

    let arg = variable_with_lltype("ch", arg_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // Compute `c = cast(ch)` once at the entry; carry it as a link
    // arg through to subsequent check blocks.
    let c0 = variable_with_lltype("c", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(c0.clone()),
    ));

    // Pre-create entry blocks for predicates[1..N-1] so each falls
    // through to the next on False.
    let next_entry_blocks: Vec<(crate::flowspace::model::BlockRef, _)> = (1..predicates.len())
        .map(|_| {
            let c_in = variable_with_lltype("c", LowLevelType::Signed);
            (Block::shared(vec![Hlvalue::Variable(c_in.clone())]), c_in)
        })
        .collect();

    let mut current_block = startblock.clone();
    let mut current_c = c0;
    for (i, pred) in predicates.iter().enumerate() {
        let last = i + 1 == predicates.len();
        let false_target = if last {
            graph.returnblock.clone()
        } else {
            next_entry_blocks[i].0.clone()
        };
        let false_link_args = || -> Vec<Hlvalue> {
            if last {
                vec![bool_false()]
            } else {
                vec![Hlvalue::Variable(current_c.clone())]
            }
        };

        match pred {
            CharCondition::Eq(n) => {
                let eq_var = variable_with_lltype("eq", LowLevelType::Bool);
                current_block
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "int_eq",
                        vec![Hlvalue::Variable(current_c.clone()), signed_const(*n)],
                        Hlvalue::Variable(eq_var.clone()),
                    ));
                current_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(eq_var));
                let true_link = Link::new(
                    vec![bool_true()],
                    Some(graph.returnblock.clone()),
                    Some(bool_true()),
                )
                .into_ref();
                let false_link =
                    Link::new(false_link_args(), Some(false_target), Some(bool_false())).into_ref();
                current_block.closeblock(vec![true_link, false_link]);
            }
            CharCondition::InRange(lo, hi) => {
                let ge_var = variable_with_lltype("ge", LowLevelType::Bool);
                current_block
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "int_ge",
                        vec![Hlvalue::Variable(current_c.clone()), signed_const(*lo)],
                        Hlvalue::Variable(ge_var.clone()),
                    ));
                current_block.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge_var));

                let c_for_hi = variable_with_lltype("c", LowLevelType::Signed);
                let block_check_hi = Block::shared(vec![Hlvalue::Variable(c_for_hi.clone())]);
                let ge_true_link = Link::new(
                    vec![Hlvalue::Variable(current_c.clone())],
                    Some(block_check_hi.clone()),
                    Some(bool_true()),
                )
                .into_ref();
                let ge_false_link = Link::new(
                    false_link_args(),
                    Some(false_target.clone()),
                    Some(bool_false()),
                )
                .into_ref();
                current_block.closeblock(vec![ge_true_link, ge_false_link]);

                let le_var = variable_with_lltype("le", LowLevelType::Bool);
                block_check_hi
                    .borrow_mut()
                    .operations
                    .push(SpaceOperation::new(
                        "int_le",
                        vec![Hlvalue::Variable(c_for_hi), signed_const(*hi)],
                        Hlvalue::Variable(le_var.clone()),
                    ));
                block_check_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(le_var));
                let le_true_link = Link::new(
                    vec![bool_true()],
                    Some(graph.returnblock.clone()),
                    Some(bool_true()),
                )
                .into_ref();
                let le_false_link =
                    Link::new(false_link_args(), Some(false_target), Some(bool_false())).into_ref();
                block_check_hi.closeblock(vec![le_true_link, le_false_link]);
            }
        }

        if !last {
            current_block = next_entry_blocks[i].0.clone();
            current_c = next_entry_blocks[i].1.clone();
        }
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

/// Synthesizes the `ll_<charlike>_<predicate>(ch) -> Bool` helper graph
/// for predicates of the form `lo <= ord(ch) <= hi` (RPython
/// `rstr.py:891-922` `ll_char_isdigit/isupper/islower`).
///
/// 3-block CFG (mirrors RPython source-level short-circuit `and`):
/// - **start**: `c = cast_<arg>_to_int(ch); ge = int_ge(c, lo)`. Branches
///   on `ge`: True → `block_check_hi`, False → `returnblock` (carrying
///   `Bool(false)` constant via the link).
/// - **block_check_hi**: `le = int_le(c, hi)`. Closes to `returnblock`
///   carrying `le` as the function result.
fn build_ll_charlike_predicate_inrange_helper_graph(
    name: &str,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
    lo: i64,
    hi: i64,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("ch", arg_lltype);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Bool);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let c_for_hi = variable_with_lltype("c", LowLevelType::Signed);
    let block_check_hi = Block::shared(vec![Hlvalue::Variable(c_for_hi.clone())]);

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // ---- start block: cast then compare against `lo`.
    let c = variable_with_lltype("c", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(c.clone()),
    ));
    let ge = variable_with_lltype("ge", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(c.clone()), signed_const(lo)],
        Hlvalue::Variable(ge.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(c)],
        Some(block_check_hi.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![bool_false()],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // ---- block_check_hi: compare against `hi`, return result.
    let le = variable_with_lltype("le", LowLevelType::Bool);
    block_check_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(c_for_hi), signed_const(hi)],
            Hlvalue::Variable(le.clone()),
        ));
    block_check_hi.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(le)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

/// Synthesizes the `ll_lower_char(ch) -> Char` /
/// `ll_upper_char(ch) -> Char` helper graph (RPython `rstr.py:925-934`):
///
/// ```python
/// def ll_lower_char(ch):
///     if 'A' <= ch <= 'Z':
///         ch = chr(ord(ch) + 32)
///     return ch
/// ```
///
/// 4-block CFG:
/// - **start**: `c = cast_char_to_int(ch); ge = int_ge(c, lo)`. True →
///   `block_check_hi` (link args `[ch, c]`); False → returnblock with
///   original `ch`.
/// - **block_check_hi**: `le = int_le(c, hi)`. True → `block_offset`
///   (link arg `[c]`); False → returnblock with original `ch`.
/// - **block_offset**: `c2 = int_add(c, offset); ch2 = cast_int_to_char(c2)`.
///   Link to returnblock with `ch2`.
fn build_ll_char_case_fold_helper_graph(
    name: &str,
    lo: i64,
    hi: i64,
    offset: i64,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("ch", LowLevelType::Char);
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Char);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let bool_true = || constant_with_lltype(ConstValue::Bool(true), LowLevelType::Bool);
    let bool_false = || constant_with_lltype(ConstValue::Bool(false), LowLevelType::Bool);
    let signed_const = |n: i64| constant_with_lltype(ConstValue::Int(n), LowLevelType::Signed);

    // block_check_hi inputargs: ch_in (carries through unchanged), c_in.
    let ch_for_hi = variable_with_lltype("ch", LowLevelType::Char);
    let c_for_hi = variable_with_lltype("c", LowLevelType::Signed);
    let block_check_hi = Block::shared(vec![
        Hlvalue::Variable(ch_for_hi.clone()),
        Hlvalue::Variable(c_for_hi.clone()),
    ]);

    // block_offset inputargs: c (already in range; offset applied here).
    let c_for_offset = variable_with_lltype("c", LowLevelType::Signed);
    let block_offset = Block::shared(vec![Hlvalue::Variable(c_for_offset.clone())]);

    // ---- start: cast + range-lo check.
    let c = variable_with_lltype("c", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "cast_char_to_int",
        vec![Hlvalue::Variable(arg.clone())],
        Hlvalue::Variable(c.clone()),
    ));
    let ge = variable_with_lltype("ge", LowLevelType::Bool);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![Hlvalue::Variable(c.clone()), signed_const(lo)],
        Hlvalue::Variable(ge.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge));
    let start_true_link = Link::new(
        vec![Hlvalue::Variable(arg.clone()), Hlvalue::Variable(c)],
        Some(block_check_hi.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let start_false_link = Link::new(
        vec![Hlvalue::Variable(arg)],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    startblock.closeblock(vec![start_true_link, start_false_link]);

    // ---- block_check_hi: range-hi check.
    let le = variable_with_lltype("le", LowLevelType::Bool);
    block_check_hi
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![Hlvalue::Variable(c_for_hi.clone()), signed_const(hi)],
            Hlvalue::Variable(le.clone()),
        ));
    block_check_hi.borrow_mut().exitswitch = Some(Hlvalue::Variable(le));
    let hi_true_link = Link::new(
        vec![Hlvalue::Variable(c_for_hi)],
        Some(block_offset.clone()),
        Some(bool_true()),
    )
    .into_ref();
    let hi_false_link = Link::new(
        vec![Hlvalue::Variable(ch_for_hi)],
        Some(graph.returnblock.clone()),
        Some(bool_false()),
    )
    .into_ref();
    block_check_hi.closeblock(vec![hi_true_link, hi_false_link]);

    // ---- block_offset: int_add(c, offset); cast_int_to_char.
    let c2 = variable_with_lltype("c2", LowLevelType::Signed);
    block_offset
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(c_for_offset), signed_const(offset)],
            Hlvalue::Variable(c2.clone()),
        ));
    let ch2 = variable_with_lltype("ch2", LowLevelType::Char);
    block_offset
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "cast_int_to_char",
            vec![Hlvalue::Variable(c2)],
            Hlvalue::Variable(ch2.clone()),
        ));
    block_offset.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(ch2)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

fn build_ll_charlike_hash_helper_graph(
    name: &str,
    arg_lltype: LowLevelType,
    cast_op: &'static str,
) -> Result<PyGraph, TyperError> {
    let arg = variable_with_lltype("ch", arg_lltype.clone());
    let startblock = Block::shared(vec![Hlvalue::Variable(arg.clone())]);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let hashed = variable_with_lltype("hashed", LowLevelType::Signed);
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        cast_op,
        vec![Hlvalue::Variable(arg)],
        Hlvalue::Variable(hashed.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(hashed)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(
        graph,
        vec!["ch".to_string()],
        func,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;

    /// rstr.py:496-500 — `CharRepr.get_ll_eq_function` returns None;
    /// `get_ll_hash_function` returns `ll_char_hash` which casts to int.
    #[test]
    fn char_repr_get_ll_hash_function_emits_cast_char_to_int() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = char_repr();

        assert!(r.get_ll_eq_function(&rtyper).unwrap().is_none());

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("Some helper");
        assert_eq!(llfn.name, "ll_char_hash");
        assert_eq!(llfn.args, vec![LowLevelType::Char]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().unwrap();
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["cast_char_to_int"]);
    }

    /// rstr.py:764-768 — `UniCharRepr.get_ll_eq_function` returns None;
    /// `get_ll_hash_function` returns `ll_unichar_hash` which casts to int.
    #[test]
    fn unichar_repr_get_ll_hash_function_emits_cast_unichar_to_int() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r = unichar_repr();

        assert!(r.get_ll_eq_function(&rtyper).unwrap().is_none());

        let llfn = r
            .get_ll_hash_function(&rtyper)
            .unwrap()
            .expect("Some helper");
        assert_eq!(llfn.name, "ll_unichar_hash");
        assert_eq!(llfn.args, vec![LowLevelType::UniChar]);
        assert_eq!(llfn.result, LowLevelType::Signed);

        let graph = llfn.graph.as_ref().unwrap();
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let opnames: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["cast_unichar_to_int"]);
    }

    /// rstr.py:491-494 / 759-762 — `convert_const` accepts only
    /// 1-character `ConstValue::Str`. Other variants and longer
    /// strings raise TyperError.
    #[test]
    fn char_repr_convert_const_accepts_single_char_only() {
        let r = char_repr();
        let c = r.convert_const(&ConstValue::Str("a".to_string())).unwrap();
        assert_eq!(c.value, ConstValue::Str("a".to_string()));
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::Char));

        let err = r
            .convert_const(&ConstValue::Str("ab".to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("not a character"));

        let err = r.convert_const(&ConstValue::Int(1)).unwrap_err();
        assert!(err.to_string().contains("not a character"));
    }

    #[test]
    fn unichar_repr_convert_const_accepts_single_unicode_only() {
        let r = unichar_repr();
        let c = r.convert_const(&ConstValue::Str("π".to_string())).unwrap();
        assert_eq!(c.concretetype.as_ref(), Some(&LowLevelType::UniChar));

        let err = r
            .convert_const(&ConstValue::Str("πi".to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("not a unicode character"));
    }

    /// `char_repr` / `unichar_repr` are module-global singletons.
    #[test]
    fn char_and_unichar_repr_singletons_dedupe() {
        let a = char_repr();
        let b = char_repr();
        assert!(Arc::ptr_eq(&a, &b));
        let u1 = unichar_repr();
        let u2 = unichar_repr();
        assert!(Arc::ptr_eq(&u1, &u2));
    }

    fn build_pair_compare_hop(
        rtyper: std::rc::Rc<RPythonTyper>,
        llops: std::rc::Rc<std::cell::RefCell<crate::translator::rtyper::rtyper::LowLevelOpList>>,
        opname: &str,
        lltype: LowLevelType,
        repr: Arc<dyn Repr>,
        s_each: crate::annotator::model::SomeValue,
    ) -> HighLevelOp {
        use crate::flowspace::model::Variable;
        let v_left = Variable::new();
        v_left.set_concretetype(Some(lltype.clone()));
        let v_right = Variable::new();
        v_right.set_concretetype(Some(lltype.clone()));
        let v_result = Variable::new();
        v_result.set_concretetype(Some(LowLevelType::Bool));
        let hop = HighLevelOp::new(
            rtyper,
            SpaceOperation::new(
                opname.to_string(),
                vec![Hlvalue::Variable(v_left), Hlvalue::Variable(v_right)],
                Hlvalue::Variable(v_result),
            ),
            Vec::new(),
            llops,
        );
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([s_each.clone(), s_each]);
        hop.args_r
            .borrow_mut()
            .extend([Some(repr.clone()), Some(repr)]);
        hop
    }

    /// rstr.py:740-746 + 750-753 — `pairtype(AbstractCharRepr,
    /// AbstractCharRepr).rtype_<func>` emits `char_<func>` for each
    /// of the six compare operations.
    #[test]
    fn pair_char_char_rtype_compare_emits_char_func_op() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["eq", "ne", "lt", "le", "gt", "ge"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                LowLevelType::Char,
                char_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(
                    false,
                )),
            );
            let result = pair_char_char_rtype_compare(&hop, func)
                .unwrap_or_else(|err| panic!("char {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "char {func}: one llop");
            assert_eq!(ops.ops[0].opname, format!("char_{func}"));
        }
    }

    /// rstr.py:778-780 + 789-792 — `pairtype(AbstractUniCharRepr,
    /// AbstractUniCharRepr).rtype_eq` / `rtype_ne` emit `unichar_eq`
    /// / `unichar_ne` directly.
    #[test]
    fn pair_unichar_unichar_rtype_eqne_emits_unichar_func_op() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["eq", "ne"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                LowLevelType::UniChar,
                unichar_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::UnicodeCodePoint(
                    crate::annotator::model::SomeUnicodeCodePoint::new(false),
                ),
            );
            let result = pair_unichar_unichar_rtype_compare_eqne(&hop, func)
                .unwrap_or_else(|err| panic!("unichar {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "unichar {func}: one llop");
            assert_eq!(ops.ops[0].opname, format!("unichar_{func}"));
        }
    }

    /// rstr.py:781-784 + 794-800 — `pairtype(AbstractUniCharRepr,
    /// AbstractUniCharRepr).rtype_lt|le|gt|ge` cast both args via
    /// `cast_unichar_to_int` then dispatch to `int_<func>`.
    #[test]
    fn pair_unichar_unichar_rtype_ord_casts_then_int_func_op() {
        use crate::translator::rtyper::rtyper::LowLevelOpList;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for func in ["lt", "le", "gt", "ge"] {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let hop = build_pair_compare_hop(
                rtyper.clone(),
                llops.clone(),
                func,
                LowLevelType::UniChar,
                unichar_repr() as Arc<dyn Repr>,
                crate::annotator::model::SomeValue::UnicodeCodePoint(
                    crate::annotator::model::SomeUnicodeCodePoint::new(false),
                ),
            );
            let result = pair_unichar_unichar_rtype_compare_ord(&hop, func)
                .unwrap_or_else(|err| panic!("unichar ord {func}: {err:?}"));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 3, "unichar ord {func}: three llops");
            assert_eq!(ops.ops[0].opname, "cast_unichar_to_int");
            assert_eq!(ops.ops[1].opname, "cast_unichar_to_int");
            assert_eq!(ops.ops[2].opname, format!("int_{func}"));
        }
    }

    /// rstr.py:891-922 ll_char_isdigit / ll_char_isupper /
    /// ll_char_islower bodies are all `lo <= ord(ch) <= hi`. Pyre
    /// synthesizes the same shape as a 3-block CFG: cast → int_ge →
    /// branch (False fallthrough to false-return / True to
    /// block_check_hi → int_le → return).
    #[test]
    fn build_ll_charlike_predicate_inrange_helper_graph_synthesizes_3_block_cfg() {
        let graph = build_ll_charlike_predicate_inrange_helper_graph(
            "ll_char_isdigit",
            LowLevelType::Char,
            "cast_char_to_int",
            48,
            57,
        )
        .expect("synthesize");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["cast_char_to_int", "int_ge"]);
        assert!(startblock.exitswitch.is_some());
        assert_eq!(startblock.exits.len(), 2);

        // True branch leads to a block that runs int_le; False branch
        // leads directly to the returnblock with a Bool(false) link arg.
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let true_target = true_link.borrow().target.as_ref().unwrap().clone();
        let target_block = true_target.borrow();
        let target_ops: Vec<&str> = target_block
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(target_ops, vec!["int_le"]);
    }

    /// rstr.py:886-912 — `ll_char_isspace` body is `c == 32 or 9 <= c
    /// <= 13`. Pyre synthesizes a 4-named-block CFG (start +
    /// inrange_check_hi + next-condition-entry + returnblock).
    #[test]
    fn build_ll_charlike_or_of_conditions_helper_graph_synthesizes_isspace_chain() {
        let graph = build_ll_charlike_or_of_conditions_helper_graph(
            "ll_char_isspace",
            LowLevelType::Char,
            "cast_char_to_int",
            ISSPACE_CONDITIONS,
        )
        .expect("synthesize");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        // start: cast → int_eq(c, 32) → branch on int_eq result.
        assert_eq!(start_ops, vec!["cast_char_to_int", "int_eq"]);
        assert_eq!(startblock.exits.len(), 2);

        // True branch is direct return Bool(true).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link present");
        let true_link_borrow = true_link.borrow();
        let true_first_arg = true_link_borrow
            .args
            .first()
            .and_then(|opt| opt.as_ref())
            .expect("True link first arg present");
        assert!(matches!(
            true_first_arg,
            Hlvalue::Constant(c) if c.value == ConstValue::Bool(true)
        ));
        drop(true_link_borrow);

        // False branch falls through to the InRange(9, 13) entry block,
        // which emits int_ge(c, 9).
        let false_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(false)))
            .expect("False exit link present");
        let inrange_entry = false_link.borrow().target.as_ref().unwrap().clone();
        let entry_borrow = inrange_entry.borrow();
        let entry_ops: Vec<&str> = entry_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(entry_ops, vec!["int_ge"]);
    }

    /// rstr.py:925-934 — `ll_lower_char(ch)` body is the conditional
    /// ASCII offset `if 'A' <= ch <= 'Z': ch = chr(ord(ch) + 32)`.
    /// Pyre synthesizes a 4-block CFG: start (cast + ge check),
    /// block_check_hi (le check), block_offset (int_add +
    /// cast_int_to_char), returnblock.
    #[test]
    fn build_ll_char_case_fold_helper_graph_synthesizes_4_block_cfg() {
        let graph =
            build_ll_char_case_fold_helper_graph("ll_lower_char", 65, 90, 32).expect("synthesize");
        let inner = graph.graph.borrow();
        let startblock = inner.startblock.borrow();
        let start_ops: Vec<&str> = startblock
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(start_ops, vec!["cast_char_to_int", "int_ge"]);
        assert_eq!(startblock.exits.len(), 2);

        // True branch carries [ch, c] to block_check_hi (which runs int_le).
        let true_link = startblock
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("True exit link");
        let check_hi = true_link.borrow().target.as_ref().unwrap().clone();
        let check_hi_borrow = check_hi.borrow();
        let check_hi_ops: Vec<&str> = check_hi_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(check_hi_ops, vec!["int_le"]);

        // block_check_hi True branch leads to block_offset (int_add + cast_int_to_char).
        let hi_true = check_hi_borrow
            .exits
            .iter()
            .find(|l| matches!(l.borrow().exitcase, Some(Hlvalue::Constant(ref c)) if c.value == ConstValue::Bool(true)))
            .expect("hi True exit link");
        let offset_block = hi_true.borrow().target.as_ref().unwrap().clone();
        let offset_borrow = offset_block.borrow();
        let offset_ops: Vec<&str> = offset_borrow
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(offset_ops, vec!["int_add", "cast_int_to_char"]);
    }

    /// rstr.py:516-538 — CharRepr/UniCharRepr inherit
    /// BaseCharReprMixin._rtype_method_isxxx; pyre's `rtype_method`
    /// dispatch routes isdigit/isupper/islower through
    /// `char_predicate_inrange_method`, which emits a single
    /// `direct_call` to the per-arg-type helper graph.
    #[test]
    fn char_unichar_rtype_method_predicates_emit_direct_call_to_per_predicate_helper() {
        use crate::flowspace::model::Variable;
        use crate::translator::rtyper::rtyper::LowLevelOpList;

        struct Case<'a> {
            method: &'a str,
            r: Arc<dyn Repr>,
            arg_lltype: LowLevelType,
            s_each: crate::annotator::model::SomeValue,
            expected_helper: &'a str,
        }
        let s_char = || {
            crate::annotator::model::SomeValue::Char(crate::annotator::model::SomeChar::new(false))
        };
        let s_unichar = || {
            crate::annotator::model::SomeValue::UnicodeCodePoint(
                crate::annotator::model::SomeUnicodeCodePoint::new(false),
            )
        };
        let cases = [
            Case {
                method: "isdigit",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isdigit",
            },
            Case {
                method: "isupper",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isupper",
            },
            Case {
                method: "islower",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_islower",
            },
            Case {
                method: "isspace",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isspace",
            },
            Case {
                method: "isalpha",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isalpha",
            },
            Case {
                method: "isalnum",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_char_isalnum",
            },
            Case {
                method: "isdigit",
                r: unichar_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::UniChar,
                s_each: s_unichar(),
                expected_helper: "ll_unichar_isdigit",
            },
            Case {
                method: "isspace",
                r: unichar_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::UniChar,
                s_each: s_unichar(),
                expected_helper: "ll_unichar_isspace",
            },
            Case {
                method: "lower",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_lower_char",
            },
            Case {
                method: "upper",
                r: char_repr() as Arc<dyn Repr>,
                arg_lltype: LowLevelType::Char,
                s_each: s_char(),
                expected_helper: "ll_upper_char",
            },
        ];

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = std::rc::Rc::new(RPythonTyper::new(&ann));

        for case in cases {
            let llops = std::rc::Rc::new(std::cell::RefCell::new(LowLevelOpList::new(
                rtyper.clone(),
                None,
            )));
            let v_arg = Variable::new();
            v_arg.set_concretetype(Some(case.arg_lltype.clone()));
            let v_result = Variable::new();
            v_result.set_concretetype(Some(LowLevelType::Bool));
            let hop = HighLevelOp::new(
                rtyper.clone(),
                SpaceOperation::new(
                    case.method.to_string(),
                    vec![Hlvalue::Variable(v_arg)],
                    Hlvalue::Variable(v_result),
                ),
                Vec::new(),
                llops.clone(),
            );
            hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
            hop.args_s.borrow_mut().push(case.s_each.clone());
            hop.args_r.borrow_mut().push(Some(case.r.clone()));

            let result = case
                .r
                .rtype_method(case.method, &hop)
                .unwrap_or_else(|err| panic!("{} {}: {err:?}", case.method, case.expected_helper));
            assert!(matches!(result, Some(Hlvalue::Variable(_))));
            let ops = llops.borrow();
            assert_eq!(ops.ops.len(), 1, "{}: one direct_call", case.method);
            assert_eq!(ops.ops[0].opname, "direct_call");
            // First arg of direct_call is the funcptr Constant whose
            // payload identifies the helper graph by name.
            let funcptr_arg = &ops.ops[0].args[0];
            let Hlvalue::Constant(c) = funcptr_arg else {
                panic!("expected Constant funcptr, got {funcptr_arg:?}");
            };
            let dbg = format!("{:?}", c.value);
            assert!(
                dbg.contains(case.expected_helper),
                "{}: expected funcptr '{}' in {dbg}",
                case.method,
                case.expected_helper
            );
        }
    }
}
