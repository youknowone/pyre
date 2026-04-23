//! RPython `rpython/rtyper/rint.py` — `IntegerRepr(FloatRepr)` +
//! per-width integer repr singletons + ll helper functions.
//!
//! ## Parity scope of this commit
//!
//! Like `rfloat.rs`, this commit stops at the Repr-shape boundary:
//! `IntegerRepr` struct, `getintegerrepr(lltype, prefix)` factory,
//! per-width module singletons (`signed_repr`, `unsigned_repr`,
//! `signedlonglong_repr`, …), and the `SomeInteger` arm in
//! [`super::rmodel::rtyper_makerepr`]. `rtype_*` methods + the
//! `pairtype(IntegerRepr, IntegerRepr)` extension block are reserved
//! for Cascade 2 at the upstream-matching structural location.
//!
//! | upstream line | pyre mirror |
//! |---|---|
//! | `class IntegerRepr(FloatRepr)` `rint.py:18-23` | [`IntegerRepr`] |
//! | `IntegerRepr.opprefix` property `rint.py:25-29` | [`IntegerRepr::opprefix`] |
//! | `IntegerRepr.convert_const` `rint.py:31-37` | [`IntegerRepr::convert_const`] |
//! | `IntegerRepr.get_ll_eq_function` `rint.py:39-42` | [`IntegerRepr::get_ll_eq_function`] |
//! | `IntegerRepr.get_ll_hash_function` `rint.py:50-54` | [`IntegerRepr::get_ll_hash_function`] |
//! | `_integer_reprs` cache `rint.py:176` | [`getintegerrepr`] OnceLock-backed singletons |
//! | `getintegerrepr` factory `rint.py:177-183` | [`getintegerrepr`] |
//! | `__extend__(annmodel.SomeInteger)` `rint.py:185-191` | [`super::rmodel::rtyper_makerepr`] `SomeInteger` arm |
//! | `signed_repr` `rint.py:193` | [`signed_repr`] singleton |
//! | `signedlonglong_repr` `rint.py:194` | [`signedlonglong_repr`] |
//! | `signedlonglonglong_repr` `rint.py:195` | [`signedlonglonglong_repr`] |
//! | `unsigned_repr` `rint.py:196` | [`unsigned_repr`] |
//! | `unsignedlonglong_repr` `rint.py:197` | [`unsignedlonglong_repr`] |
//! | `unsignedlonglonglong_repr` `rint.py:198` | [`unsignedlonglonglong_repr`] |
//! | `ll_hash_int` / `ll_hash_long_long` / `ll_eq_shortint` `rint.py:619-627` | [`ll_hash_int`] / [`ll_hash_long_long`] / [`ll_eq_shortint`] |
//! | `ll_check_chr` / `ll_check_unichr` `rint.py:629-640` | [`ll_check_chr`] / [`ll_check_unichr`] |
//!
//! ## ll_*_py_div / ll_*_py_mod family — deferred
//!
//! `rint.py:399-581` defines ~30 variants of `ll_*_py_div` /
//! `ll_*_py_mod` with `_zer` / `_ovf` / `_zer_ovf` / `_nonnegargs`
//! suffix combinations. They are called from `_rtype_call_helper`
//! (upstream rint.py:344-387) which itself is invoked only from the
//! `pairtype(IntegerRepr, IntegerRepr).rtype_floordiv` etc. block.
//! Cascade 2 lands the pairtype block and pulls in this helper family
//! together at the same structural location upstream uses.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::annotator::model::KnownType;
use crate::flowspace::model::{ConstValue, Constant};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::{LlHelper, Repr, ReprState};

// ____________________________________________________________
// RPython `class IntegerRepr(FloatRepr)` (rint.py:18-173).

/// RPython `class IntegerRepr(FloatRepr)` (`rint.py:18-173`).
///
/// ```python
/// class IntegerRepr(FloatRepr):
///     def __init__(self, lowleveltype, opprefix):
///         self.lowleveltype = lowleveltype
///         self._opprefix = opprefix
///         self.as_int = self
/// ```
///
/// Upstream inherits from `FloatRepr` so its `get_ll_hash_function` /
/// comparison-helper accessors can be overridden while sharing the
/// base float semantics. Rust trait inheritance is not available;
/// `IntegerRepr` implements [`Repr`] directly and reproduces the
/// methods that `rfloat.py:FloatRepr` defines via its own
/// implementations pointing back at upstream.
#[derive(Debug)]
pub struct IntegerRepr {
    state: ReprState,
    lltype: LowLevelType,
    /// RPython `self._opprefix` (`rint.py:21`). `None` for integer
    /// widths whose arithmetic is not supported (upstream raises
    /// `TyperError` when accessed; see [`IntegerRepr::opprefix`]).
    _opprefix: Option<&'static str>,
}

impl IntegerRepr {
    /// RPython `IntegerRepr.__init__(lowleveltype, opprefix)`
    /// (`rint.py:19-22`). `as_int = self` is omitted — the Rust
    /// `IntegerRepr::as_int` accessor returns a reference to self.
    pub fn new(lowleveltype: LowLevelType, opprefix: Option<&'static str>) -> Self {
        IntegerRepr {
            state: ReprState::new(),
            lltype: lowleveltype,
            _opprefix: opprefix,
        }
    }

    /// RPython `IntegerRepr.opprefix` property (`rint.py:24-29`).
    ///
    /// ```python
    /// @property
    /// def opprefix(self):
    ///     if self._opprefix is None:
    ///         raise TyperError("arithmetic not supported on %r, its size is too small" %
    ///                          self.lowleveltype)
    ///     return self._opprefix
    /// ```
    pub fn opprefix(&self) -> Result<&'static str, TyperError> {
        self._opprefix.ok_or_else(|| {
            TyperError::message(format!(
                "arithmetic not supported on {}, its size is too small",
                self.lltype.short_name()
            ))
        })
    }

    /// RPython `IntegerRepr.ll_dummy_value = -1` (`rint.py:65`). Class
    /// attribute on upstream; pyre exposes via associated const.
    pub const LL_DUMMY_VALUE: i64 = -1;
}

impl Repr for IntegerRepr {
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
        "IntegerRepr"
    }

    /// RPython `IntegerRepr.convert_const` (`rint.py:31-37`).
    ///
    /// ```python
    /// def convert_const(self, value):
    ///     if isinstance(value, objectmodel.Symbolic):
    ///         return value
    ///     T = typeOf(value)
    ///     if isinstance(T, Number) or T is Bool:
    ///         return cast_primitive(self.lowleveltype, value)
    ///     raise TyperError("not an integer: %r" % (value,))
    /// ```
    ///
    /// * `Symbolic` passthrough is omitted (pyre does not model
    ///   `objectmodel.Symbolic` yet; NEW sentinel lands when the rlib
    ///   port does).
    /// * `typeOf(value)` on pyre's `ConstValue` is approximated by the
    ///   variant tag — `Int` / `Bool` pass, everything else fails.
    /// * `cast_primitive(self.lowleveltype, value)` at `lltype.py:1039`
    ///   normalises the payload to the target int type. Pyre's
    ///   `ConstValue::Int` is a single machine-word i64 covering every
    ///   signed/unsigned width we currently model, so the only payload
    ///   normalisation is `Bool → Int(0/1)`. The `concretetype` stamp
    ///   records the repr's lowleveltype so downstream consumers see a
    ///   consistent Int constant even when the source was a Bool.
    fn convert_const(&self, value: &ConstValue) -> Result<Constant, TyperError> {
        let casted = match value {
            // upstream `Number` accept branch (`rint.py:35-36`) — the
            // Int payload is already in the target word, just re-stamp.
            ConstValue::Int(_) => value.clone(),
            // upstream `T is Bool` accept branch. `cast_primitive` at
            // `lltype.py:1039` coerces Bool → int value 0/1; pyre
            // materialises the Int payload explicitly so downstream
            // `emit_const_i` never sees a Bool under an int concretetype.
            ConstValue::Bool(b) => ConstValue::Int(if *b { 1 } else { 0 }),
            other => return Err(TyperError::message(format!("not an integer: {other:?}"))),
        };
        Ok(Constant::with_concretetype(casted, self.lltype.clone()))
    }
}

impl IntegerRepr {
    /// RPython `IntegerRepr.get_ll_eq_function` (`rint.py:39-42`).
    ///
    /// ```python
    /// def get_ll_eq_function(self):
    ///     if getattr(self, '_opprefix', '?') is None:
    ///         return ll_eq_shortint
    ///     return None
    /// ```
    ///
    /// Upstream returns `ll_eq_shortint` for int types with no
    /// opprefix (i.e. widths too small for arithmetic). Pyre returns
    /// a typed helper descriptor carrying the same helper identity.
    pub fn get_ll_eq_function(&self) -> Option<LlHelper> {
        if self._opprefix.is_none() {
            Some(LlHelper::EqShortInt)
        } else {
            None
        }
    }

    /// RPython `IntegerRepr.get_ll_ge_function` / `get_ll_gt_function`
    /// / `get_ll_lt_function` / `get_ll_le_function` (`rint.py:44-48`).
    ///
    /// ```python
    /// def get_ll_ge_function(self):
    ///     return None
    /// get_ll_gt_function = get_ll_ge_function
    /// get_ll_lt_function = get_ll_ge_function
    /// get_ll_le_function = get_ll_ge_function
    /// ```
    pub fn get_ll_ge_function(&self) -> Option<LlHelper> {
        None
    }

    /// RPython alias of [`Self::get_ll_ge_function`] (`rint.py:46`).
    pub fn get_ll_gt_function(&self) -> Option<LlHelper> {
        self.get_ll_ge_function()
    }

    /// RPython alias of [`Self::get_ll_ge_function`] (`rint.py:47`).
    pub fn get_ll_lt_function(&self) -> Option<LlHelper> {
        self.get_ll_ge_function()
    }

    /// RPython alias of [`Self::get_ll_ge_function`] (`rint.py:48`).
    pub fn get_ll_le_function(&self) -> Option<LlHelper> {
        self.get_ll_ge_function()
    }

    /// RPython `IntegerRepr.get_ll_hash_function` (`rint.py:50-54`).
    ///
    /// ```python
    /// def get_ll_hash_function(self):
    ///     if (sys.maxint == 2147483647 and
    ///         self.lowleveltype in (SignedLongLong, UnsignedLongLong)):
    ///         return ll_hash_long_long
    ///     return ll_hash_int
    /// ```
    ///
    /// On 64-bit pyre `sys.maxint` is always `9223372036854775807`, so
    /// the `sys.maxint == 2**31-1` branch never fires; we always
    /// return `ll_hash_int`. A 32-bit target port would need to check
    /// `cfg!(target_pointer_width = "32")` here.
    pub fn get_ll_hash_function(&self) -> LlHelper {
        if cfg!(target_pointer_width = "32")
            && matches!(
                self.lltype,
                LowLevelType::SignedLongLong | LowLevelType::UnsignedLongLong
            )
        {
            LlHelper::HashLongLong
        } else {
            LlHelper::HashInt
        }
    }

    /// RPython `IntegerRepr.get_ll_fasthash_function = get_ll_hash_function`
    /// (`rint.py:56`). Alias.
    pub fn get_ll_fasthash_function(&self) -> LlHelper {
        self.get_ll_hash_function()
    }

    /// RPython `IntegerRepr.as_int = self` (`rint.py:22`). Mirrors the
    /// Python reflexive-pointer assignment — `BoolRepr` overrides to
    /// return the `signed_repr` singleton (`rbool.py:13-15`).
    pub fn as_int(self: &Arc<Self>) -> Arc<IntegerRepr> {
        self.clone()
    }
}

// ____________________________________________________________
// `_integer_reprs` cache + `getintegerrepr` factory
// (rint.py:176-183).

/// RPython `_integer_reprs = {}` + `getintegerrepr(lltype, prefix=None)`
/// (`rint.py:176-183`).
///
/// ```python
/// _integer_reprs = {}
/// def getintegerrepr(lltype, prefix=None):
///     try:
///         return _integer_reprs[lltype]
///     except KeyError:
///         pass
///     repr = _integer_reprs[lltype] = IntegerRepr(lltype, prefix)
///     return repr
/// ```
///
/// Upstream caches by `lltype` only, ignoring subsequent `prefix`
/// arguments once the repr exists. Mirror that exactly so
/// `getintegerrepr(Signed, None)` after `signed_repr` still returns the
/// same object.
pub fn getintegerrepr(lltype: LowLevelType, prefix: Option<&'static str>) -> Arc<IntegerRepr> {
    static INTEGER_REPRS: OnceLock<Mutex<HashMap<LowLevelType, Arc<IntegerRepr>>>> =
        OnceLock::new();
    let reprs = INTEGER_REPRS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut reprs = reprs.lock().unwrap();
    if let Some(repr) = reprs.get(&lltype) {
        return repr.clone();
    }
    let repr = Arc::new(IntegerRepr::new(lltype.clone(), prefix));
    reprs.insert(lltype, repr.clone());
    repr
}

/// Singleton — `signed_repr = getintegerrepr(Signed, 'int_')`
/// (`rint.py:193`).
pub fn signed_repr() -> Arc<IntegerRepr> {
    getintegerrepr(LowLevelType::Signed, Some("int_"))
}

/// Singleton — `signedlonglong_repr = getintegerrepr(SignedLongLong, 'llong_')`
/// (`rint.py:194`).
pub fn signedlonglong_repr() -> Arc<IntegerRepr> {
    getintegerrepr(LowLevelType::SignedLongLong, Some("llong_"))
}

/// Singleton — `signedlonglonglong_repr = getintegerrepr(SignedLongLongLong,
/// 'lllong_')` (`rint.py:195`).
///
/// Main's `LowLevelType` enum does not yet carry a `SignedLongLongLong`
/// variant. Failing loudly is safer than silently aliasing this to the
/// 64-bit repr.
pub fn signedlonglonglong_repr() -> Arc<IntegerRepr> {
    panic!("SignedLongLongLong repr is not supported until the 128-bit lltype lands")
}

/// Singleton — `unsigned_repr = getintegerrepr(Unsigned, 'uint_')`
/// (`rint.py:196`).
pub fn unsigned_repr() -> Arc<IntegerRepr> {
    getintegerrepr(LowLevelType::Unsigned, Some("uint_"))
}

/// Singleton — `unsignedlonglong_repr = getintegerrepr(UnsignedLongLong, 'ullong_')`
/// (`rint.py:197`).
pub fn unsignedlonglong_repr() -> Arc<IntegerRepr> {
    getintegerrepr(LowLevelType::UnsignedLongLong, Some("ullong_"))
}

/// Singleton — `unsignedlonglonglong_repr = getintegerrepr(UnsignedLongLongLong,
/// 'ulllong_')` (`rint.py:198`).
///
/// See `signedlonglonglong_repr` above.
pub fn unsignedlonglonglong_repr() -> Arc<IntegerRepr> {
    panic!("UnsignedLongLongLong repr is not supported until the 128-bit lltype lands")
}

// ____________________________________________________________
// `SomeInteger.rtyper_makerepr` dispatch — keyed on KnownType.
//
// RPython `rint.py:185-191`:
//
//     class __extend__(annmodel.SomeInteger):
//         def rtyper_makerepr(self, rtyper):
//             lltype = build_number(None, self.knowntype)
//             return getintegerrepr(lltype)
//
//         def rtyper_makekey(self):
//             return self.__class__, self.knowntype
//
// Pyre's `SomeInteger.knowntype` is a `KnownType` enum covering the
// six integer widths upstream `build_number` exposes
// (`rint.py:193-198`). Mapping:
//
//     KnownType::Int            → signed_repr
//     KnownType::Ruint          → unsigned_repr
//     KnownType::LongLong       → signedlonglong_repr
//     KnownType::ULongLong      → unsignedlonglong_repr
// 128-bit `r_longlonglong` / `r_ulonglonglong` land when the
// `rlib/rarithmetic` port adds the matching KnownType variants.

/// Dispatch `SomeInteger.knowntype → IntegerRepr` (`rint.py:186-188`
/// `lltype = build_number(None, self.knowntype); return getintegerrepr(lltype)`).
pub fn integer_repr_for_knowntype(knowntype: KnownType) -> Result<Arc<IntegerRepr>, TyperError> {
    match knowntype {
        KnownType::Int => Ok(signed_repr()),
        KnownType::Ruint => Ok(unsigned_repr()),
        KnownType::LongLong => Ok(signedlonglong_repr()),
        KnownType::ULongLong => Ok(unsignedlonglong_repr()),
        other => Err(TyperError::missing_rtype_operation(format!(
            "SomeInteger.rtyper_makerepr: knowntype {other:?} not yet mapped — \
             port rlib/rarithmetic integer width alias first",
        ))),
    }
}

// ____________________________________________________________
// ll helper functions (rint.py:619-640).
//
// These are called from `rtype_*` / `_rtype_call_helper` in Cascade 2
// but exposed here so the consumer side can register them by name
// today.

/// Runtime-style exception raised by `ll_*` helper functions.
///
/// These helpers are not rtyper diagnostics: upstream raises concrete
/// exceptions like `ValueError`, and the specializing caller wires that
/// into exception edges separately.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlHelperException {
    ValueError,
}

/// RPython `ll_hash_int(n)` (`rint.py:619-620`).
///
/// ```python
/// def ll_hash_int(n):
///     return intmask(n)
/// ```
///
/// On 64-bit hosts `intmask` is identity; on 32-bit it truncates.
pub fn ll_hash_int(n: i64) -> i64 {
    n
}

/// RPython `ll_hash_long_long(n)` (`rint.py:622-623`).
///
/// ```python
/// def ll_hash_long_long(n):
///     return intmask(intmask(n) + 9 * intmask(n >> 32))
/// ```
pub fn ll_hash_long_long(n: i64) -> i32 {
    let low = n as i32;
    let high = (n >> 32) as i32;
    // upstream wraps in intmask — Rust wrapping_add preserves the
    // 2's-complement overflow semantics.
    low.wrapping_add(9i32.wrapping_mul(high))
}

/// RPython `ll_eq_shortint(n, m)` (`rint.py:625-627`).
///
/// ```python
/// def ll_eq_shortint(n, m):
///     return intmask(n) == intmask(m)
/// ll_eq_shortint.no_direct_compare = True
/// ```
pub fn ll_eq_shortint(n: i64, m: i64) -> bool {
    (n as i32) == (m as i32)
}

/// RPython `ll_check_chr(n)` (`rint.py:629-633`).
///
/// ```python
/// def ll_check_chr(n):
///     if 0 <= n <= 255:
///         return
///     else:
///         raise ValueError
/// ```
pub fn ll_check_chr(n: i64) -> Result<(), LlHelperException> {
    if (0..=255).contains(&n) {
        Ok(())
    } else {
        Err(LlHelperException::ValueError)
    }
}

/// RPython `ll_check_unichr(n)` (`rint.py:635-640`).
///
/// ```python
/// def ll_check_unichr(n):
///     from rpython.rlib.runicode import MAXUNICODE
///     if 0 <= n <= MAXUNICODE:
///         return
///     else:
///         raise ValueError
/// ```
///
/// `MAXUNICODE` upstream is `0x10FFFF` on a wide-build CPython and
/// `0xFFFF` on a narrow build. Pyre targets modern Python only, so
/// the wide range is used unconditionally.
pub fn ll_check_unichr(n: i64) -> Result<(), LlHelperException> {
    const MAXUNICODE: i64 = 0x10FFFF;
    if (0..=MAXUNICODE).contains(&n) {
        Ok(())
    } else {
        Err(LlHelperException::ValueError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::rmodel::Setupstate;

    #[test]
    fn signed_repr_is_singleton_with_int_opprefix() {
        // rint.py:193.
        let a = signed_repr();
        let b = signed_repr();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(a.lowleveltype(), &LowLevelType::Signed);
        assert_eq!(a.opprefix().unwrap(), "int_");
    }

    #[test]
    fn unsigned_repr_is_singleton_with_uint_opprefix() {
        // rint.py:196.
        let r = unsigned_repr();
        assert_eq!(r.lowleveltype(), &LowLevelType::Unsigned);
        assert_eq!(r.opprefix().unwrap(), "uint_");
    }

    #[test]
    fn opprefix_returns_typer_error_when_missing() {
        // rint.py:25-29 — arithmetic-not-supported path.
        let r = IntegerRepr::new(LowLevelType::Signed, None);
        let err = r.opprefix().unwrap_err();
        assert!(
            err.to_string().contains("arithmetic not supported"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn convert_const_accepts_int_and_bool_rejects_float() {
        // rint.py:31-37 — upstream accepts (int, base_int, Bool);
        // float raises TyperError.
        let r = signed_repr();
        assert!(r.convert_const(&ConstValue::Int(0)).is_ok());
        assert!(r.convert_const(&ConstValue::Bool(true)).is_ok());
        let err = r.convert_const(&ConstValue::Float(0)).unwrap_err();
        assert!(err.to_string().contains("not an integer"));
    }

    #[test]
    fn convert_const_normalises_bool_to_int_payload() {
        // rint.py:36 `cast_primitive(self.lowleveltype, value)` — when
        // upstream routes a Bool through an IntegerRepr it materialises
        // the int value 0/1. Pyre mirrors so downstream `emit_const_i`
        // never observes `ConstValue::Bool` under an int concretetype.
        let r = signed_repr();
        let c_true = r.convert_const(&ConstValue::Bool(true)).unwrap();
        assert_eq!(c_true.value, ConstValue::Int(1));
        assert_eq!(c_true.concretetype.as_ref(), Some(&LowLevelType::Signed));

        let c_false = r.convert_const(&ConstValue::Bool(false)).unwrap();
        assert_eq!(c_false.value, ConstValue::Int(0));

        // Plain Int payload is preserved bit-for-bit.
        let c_int = r.convert_const(&ConstValue::Int(-42)).unwrap();
        assert_eq!(c_int.value, ConstValue::Int(-42));
    }

    #[test]
    fn get_ll_hash_function_returns_ll_hash_int_on_64bit() {
        // rint.py:50-54. The 64-bit branch always chooses ll_hash_int.
        let r = signed_repr();
        assert_eq!(r.get_ll_hash_function(), LlHelper::HashInt);
    }

    #[test]
    fn get_ll_eq_function_returns_ll_eq_shortint_when_no_opprefix() {
        // rint.py:39-42.
        let r_no_prefix = IntegerRepr::new(LowLevelType::Signed, None);
        assert_eq!(r_no_prefix.get_ll_eq_function(), Some(LlHelper::EqShortInt));

        let r_with_prefix = signed_repr();
        assert_eq!(r_with_prefix.get_ll_eq_function(), None);
    }

    #[test]
    fn as_int_returns_self_on_integer_repr() {
        // rint.py:22 — IntegerRepr.as_int = self.
        let r = signed_repr();
        let asi = r.as_int();
        assert!(Arc::ptr_eq(&r, &asi));
    }

    #[test]
    fn getintegerrepr_dispatches_to_module_singletons() {
        // rint.py:177-183 — cache key is lltype only.
        let s = getintegerrepr(LowLevelType::Signed, Some("int_"));
        assert!(Arc::ptr_eq(&s, &signed_repr()));
        let u = getintegerrepr(LowLevelType::Unsigned, Some("uint_"));
        assert!(Arc::ptr_eq(&u, &unsigned_repr()));
    }

    #[test]
    fn getintegerrepr_ignores_later_prefix_arguments_like_upstream() {
        // Upstream caches by lltype only, so a later call with
        // `prefix=None` still returns the existing repr.
        let seeded = signed_repr();
        let cached = getintegerrepr(LowLevelType::Signed, None);
        assert!(Arc::ptr_eq(&seeded, &cached));
        assert_eq!(cached.opprefix().unwrap(), "int_");
    }

    #[test]
    fn integer_repr_for_knowntype_dispatches_int_and_ruint() {
        // rint.py:186-188 `build_number(None, self.knowntype)`
        // mapping for pyre-supported KnownTypes.
        let r_int = integer_repr_for_knowntype(KnownType::Int).unwrap();
        assert!(Arc::ptr_eq(&r_int, &signed_repr()));
        let r_u = integer_repr_for_knowntype(KnownType::Ruint).unwrap();
        assert!(Arc::ptr_eq(&r_u, &unsigned_repr()));
    }

    #[test]
    fn integer_repr_for_knowntype_dispatches_longlong_widths() {
        // rint.py:193-198 — 64-bit module-level singletons. The 128-bit
        // `r_longlonglong` / `r_ulonglonglong` knowntypes await the
        // `rlib/rarithmetic` port that adds the matching
        // `LowLevelType::SignedLongLongLong` / `UnsignedLongLongLong`
        // enum variants; covered above this point when they land.
        let ll = integer_repr_for_knowntype(KnownType::LongLong).unwrap();
        assert!(Arc::ptr_eq(&ll, &signedlonglong_repr()));
        assert_eq!(ll.lowleveltype(), &LowLevelType::SignedLongLong);

        let ull = integer_repr_for_knowntype(KnownType::ULongLong).unwrap();
        assert!(Arc::ptr_eq(&ull, &unsignedlonglong_repr()));
        assert_eq!(ull.lowleveltype(), &LowLevelType::UnsignedLongLong);
    }

    #[test]
    fn integer_repr_for_knowntype_surface_missing_rtype_on_unmapped() {
        // Any integer KnownType pyre hasn't ported yet should surface a
        // MissingRTypeOperation pointing at the rlib/rarithmetic port
        // as the next step.
        let err = integer_repr_for_knowntype(KnownType::Object).unwrap_err();
        assert!(
            err.is_missing_rtype_operation(),
            "expected MissingRTypeOperation; got {err:?}"
        );
    }

    #[test]
    fn integer_repr_setup_state_matches_base_repr() {
        // Repr state machine applies to IntegerRepr identically.
        let r = signed_repr();
        let _ = r.setup();
        assert_eq!(r.state().get(), Setupstate::Finished);
    }

    #[test]
    fn ll_hash_int_is_identity_on_64bit() {
        // rint.py:619-620 — intmask is identity on 64-bit.
        assert_eq!(ll_hash_int(42), 42);
        assert_eq!(ll_hash_int(-1), -1);
    }

    #[test]
    fn ll_hash_long_long_matches_upstream_mix() {
        // rint.py:622-623 — intmask(intmask(n) + 9 * intmask(n >> 32)).
        let mixed = ll_hash_long_long(0x1234_5678_9ABC_DEF0u64 as i64);
        // The intent test: deterministic + non-zero for a multi-word
        // input. Lock in the specific formula.
        let low = 0x9ABC_DEF0u32 as i32;
        let high = 0x1234_5678i32;
        let expected = low.wrapping_add(9i32.wrapping_mul(high));
        assert_eq!(mixed, expected);
    }

    #[test]
    fn ll_eq_shortint_compares_low_word() {
        // rint.py:625-626.
        assert!(ll_eq_shortint(0xDEAD_BEEF, 0xDEAD_BEEF));
        // High-word differences do not matter for shortint equality.
        assert!(ll_eq_shortint(
            (1i64 << 40) | 0xDEAD_BEEF,
            (1i64 << 44) | 0xDEAD_BEEF
        ));
        assert!(!ll_eq_shortint(1, 2));
    }

    #[test]
    fn ll_check_chr_ranges_match_upstream() {
        // rint.py:629-633 — [0, 255].
        assert!(ll_check_chr(0).is_ok());
        assert!(ll_check_chr(255).is_ok());
        assert_eq!(ll_check_chr(256), Err(LlHelperException::ValueError));
        assert_eq!(ll_check_chr(-1), Err(LlHelperException::ValueError));
    }

    #[test]
    fn ll_check_unichr_ranges_match_upstream_wide_build() {
        // rint.py:635-640 — [0, MAXUNICODE].
        assert!(ll_check_unichr(0).is_ok());
        assert!(ll_check_unichr(0x10FFFF).is_ok());
        assert_eq!(
            ll_check_unichr(0x110000),
            Err(LlHelperException::ValueError)
        );
        assert_eq!(ll_check_unichr(-1), Err(LlHelperException::ValueError));
    }
}
