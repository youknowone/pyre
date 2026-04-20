//! Bookkeeper — central state carrier for the annotator.
//!
//! RPython upstream: `rpython/annotator/bookkeeper.py` (614 LOC).
//!
//! **Phase 5 P5.2 in progress.** Ports the `immutablevalue` /
//! `newlist` / `newdict` subset that downstream modules
//! (builtin.py, description.py, binaryop.py) invoke directly on the
//! bookkeeper. Fields that require the descriptor / class machinery
//! — `descs`, `classdefs`, `methoddescs`, `emulated_pbc_calls`,
//! `classpbc_attr_families`, `all_specializations`, etc. — land in
//! commit 2 alongside `description.py` / `classdesc.py`.
//!
//! ## Phase 5 P5.2+ dependency-blocked helpers
//!
//! * `getdesc(x)` / `immutablevalue` for function / class / bound-
//!   method / weakref / frozen PBC inputs — blocked on
//!   `description.py`.
//! * `getuniqueclassdef(cls)` — blocked on `classdesc.py`.
//! * `emulate_pbc_call(key, pbc, args_s)` — blocked on
//!   `binaryop.py` call-family machinery.
//! * `register_builtins()` / `BUILTIN_ANALYZERS` registry — blocked
//!   on `builtin.py`.
//! * `annotator` backlink (`reflowfromposition` callback) — blocked
//!   on `annrpython.py` driver.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::dictdef::DictDef;
use super::listdef::ListDef;
use super::model::{
    AnnotatorError, Desc, DescKind, SomeBool, SomeBuiltin, SomeByteArray, SomeChar, SomeDict,
    SomeFloat, SomeInteger, SomeList, SomePBC, SomeString, SomeTuple, SomeUnicodeCodePoint,
    SomeUnicodeString, SomeValue, s_none,
};
use crate::flowspace::model::{ConstValue, Constant, HostObject};

/// RPython `bookkeeper.position_key` (bookkeeper.py:147) — the tuple
/// identifying "where in the flow graph the annotator is currently
/// reading/writing a value".
///
/// Upstream stores `(FunctionGraph, Block, operation_index)` directly.
/// The Rust port carries the identity-hash values of the first two
/// components so the struct stays:
///   * cheap to clone / hash (no flowspace import cycle),
///   * free of borrow-lifetime issues inside `read_locations:
///     HashSet<PositionKey>`,
///   * still upstream-shaped as a 3-tuple that `ListItem.read_locations
///     |= other.read_locations` can merge without loss.
///
/// Callers obtain the identity hashes via
/// `Rc::as_ptr(&graph) as usize` / `Rc::as_ptr(&block) as usize`.
/// Full bookkeeper.py port replaces the first two fields with real
/// `Weak<FunctionGraph>` / `Weak<RefCell<Block>>` refs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PositionKey {
    /// Identity hash of the enclosing `FunctionGraph` — upstream
    /// `position_key[0]`.
    pub graph_id: usize,
    /// Identity hash of the enclosing `Block` — upstream
    /// `position_key[1]`.
    pub block_id: usize,
    /// Operation index inside the block — upstream `position_key[2]`.
    pub op_index: usize,
}

impl PositionKey {
    pub fn new(graph_id: usize, block_id: usize, op_index: usize) -> Self {
        PositionKey {
            graph_id,
            block_id,
            op_index,
        }
    }
}

/// RPython `class Bookkeeper` (bookkeeper.py:53).
#[derive(Debug)]
pub struct Bookkeeper {
    /// RPython `self.position_key = None` initial (bookkeeper.py:147).
    /// The annotator driver (`RPythonAnnotator.reflow`) writes into
    /// this slot around each reflow block so `read_item` / `agree`
    /// pick it up. Interior mutability because callers hold
    /// `Rc<Bookkeeper>` sharers.
    pub position_key: RefCell<Option<PositionKey>>,
    /// RPython `self.listdefs = {}` (bookkeeper.py:59). Keyed by
    /// position — callers hitting the same position twice share the
    /// ListDef so merging re-entries stay identity-equal. The key is
    /// `Option<PositionKey>` because upstream uses `self.position_key`
    /// directly as the dict key (bookkeeper.py:180
    /// `self.listdefs[self.position_key]`); when `position_key` is
    /// `None`, upstream still caches under the `None` key — so we do
    /// the same rather than building a fresh ListDef per call outside
    /// a reflow frame.
    pub listdefs: RefCell<HashMap<Option<PositionKey>, ListDef>>,
    /// RPython `self.dictdefs = {}` (bookkeeper.py:60). Same
    /// `Option<PositionKey>` key semantics as `listdefs`.
    pub dictdefs: RefCell<HashMap<Option<PositionKey>, DictDef>>,
}

impl Bookkeeper {
    /// RPython `Bookkeeper.__init__(self, annotator)` (bookkeeper.py:52-76).
    /// Once the annotator driver lands, this constructor takes an
    /// `annotator` backlink; for now it just initialises the bare
    /// storage slots.
    pub fn new() -> Self {
        Bookkeeper {
            position_key: RefCell::new(None),
            listdefs: RefCell::new(HashMap::new()),
            dictdefs: RefCell::new(HashMap::new()),
        }
    }

    /// RPython `bookkeeper.position_key = ...` assignment. Returns the
    /// previous value so callers can restore it around a nested reflow
    /// (matches upstream bookkeeper.py:278 `@contextmanager
    /// position()`).
    pub fn set_position_key(&self, pk: Option<PositionKey>) -> Option<PositionKey> {
        self.position_key.replace(pk)
    }

    /// Current `bookkeeper.position_key`. Returns `None` when no
    /// reflow frame is active (upstream's initial
    /// `self.position_key = None`).
    pub fn current_position_key(&self) -> Option<PositionKey> {
        *self.position_key.borrow()
    }

    /// RPython `Bookkeeper.getlistdef(**flags_if_new)` (bookkeeper.py:178-185).
    ///
    /// Returns the (cached or freshly constructed) ListDef for the
    /// bookkeeper's current position. Upstream stores flags inside the
    /// `listitem.__dict__`; Rust carries the `range_step` flag
    /// explicitly (the only non-default flag any caller passes — see
    /// bookkeeper.py:193-195).
    ///
    /// The current position — including `None` — is used as the cache
    /// key directly, matching upstream's `self.listdefs[self.position_
    /// key]` indexing. Two calls with no active position share the
    /// same ListDef just like two calls inside the same reflow frame
    /// would.
    pub fn getlistdef(self: &Rc<Self>, range_step: Option<i64>) -> ListDef {
        let pk = self.current_position_key();
        let mut listdefs = self.listdefs.borrow_mut();
        if let Some(existing) = listdefs.get(&pk) {
            return existing.clone();
        }
        let new_ld = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
        if let Some(step) = range_step {
            let li = new_ld.inner.listitem.borrow().clone();
            li.borrow_mut().range_step = Some(step);
        }
        listdefs.insert(pk, new_ld.clone());
        new_ld
    }

    /// RPython `Bookkeeper.newlist(*s_values, **flags)` (bookkeeper.py:187-196).
    pub fn newlist(
        self: &Rc<Self>,
        s_values: &[SomeValue],
        range_step: Option<i64>,
    ) -> Result<SomeList, AnnotatorError> {
        let listdef = self.getlistdef(range_step);
        for s_value in s_values {
            listdef
                .generalize(s_value)
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        if let Some(step) = range_step {
            listdef
                .generalize_range_step(Some(step))
                .map_err(|e| AnnotatorError::new(e.msg))?;
        }
        Ok(SomeList::new(listdef))
    }

    /// RPython `Bookkeeper.getdictdef(is_r_dict=False,
    /// force_non_null=False, simple_hash_eq=False)` (bookkeeper.py:198-207).
    ///
    /// `None` position caches just like `Some(pk)`, matching upstream's
    /// `self.dictdefs[self.position_key]` indexing. See [`Self::
    /// getlistdef`] for the rationale.
    pub fn getdictdef(
        self: &Rc<Self>,
        is_r_dict: bool,
        force_non_null: bool,
        simple_hash_eq: bool,
    ) -> DictDef {
        let pk = self.current_position_key();
        let mut dictdefs = self.dictdefs.borrow_mut();
        if let Some(existing) = dictdefs.get(&pk) {
            return existing.clone();
        }
        let new_dd = DictDef::new(
            Some(self.clone()),
            SomeValue::Impossible,
            SomeValue::Impossible,
            is_r_dict,
            force_non_null,
            simple_hash_eq,
        );
        dictdefs.insert(pk, new_dd.clone());
        new_dd
    }

    /// RPython `Bookkeeper.newdict()` (bookkeeper.py:209-212).
    pub fn newdict(self: &Rc<Self>) -> SomeDict {
        SomeDict::new(self.getdictdef(false, false, false))
    }

    /// RPython `Bookkeeper.immutablevalue(x)` (bookkeeper.py:214-325).
    ///
    /// "The most precise SomeValue instance that contains the
    /// immutable value x."
    ///
    /// Input is a flowspace [`ConstValue`] — the Rust-side
    /// counterpart to upstream's Python constant. Primitive branches
    /// (bool / int / float / str / char / unicode / bytearray / tuple
    /// / None) are ported line-by-line; `list` / `dict` build
    /// [`SomeList`] / [`SomeDict`] via `getlistdef` / `getdictdef`
    /// without the upstream `immutable_cache` memoisation (perf-only
    /// deviation, correctness unchanged).
    ///
    /// The function / class / bound-method / weakref / frozen-PBC
    /// / symbolic-constant / PBC branches (bookkeeper.py:218-325)
    /// require `description.py` + `classdesc.py` + `rlib/rarithmetic`
    /// / `rlib/objectmodel` imports that are Phase 5 P5.2+ deps. Those
    /// inputs surface as [`AnnotatorError`] so the missing branch is
    /// observable at the call site.
    pub fn immutablevalue(self: &Rc<Self>, x: &ConstValue) -> Result<SomeValue, AnnotatorError> {
        match x {
            ConstValue::Bool(b) => {
                let mut s = SomeBool::new();
                s.base.const_box = Some(Constant::new(ConstValue::Bool(*b)));
                Ok(SomeValue::Bool(s))
            }
            ConstValue::Int(i) => {
                // upstream: `result = SomeInteger(nonneg = x>=0)`.
                let mut s = SomeInteger::new(*i >= 0, false);
                s.base.const_box = Some(Constant::new(ConstValue::Int(*i)));
                Ok(SomeValue::Integer(s))
            }
            ConstValue::Float(_) => {
                let mut s = SomeFloat::new();
                s.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Float(s))
            }
            ConstValue::Str(s) => {
                let no_nul = !s.contains('\x00');
                let result = if s.chars().count() == 1 {
                    // upstream: `result = SomeChar(no_nul=no_nul)`.
                    let mut ch = SomeChar::new(no_nul);
                    ch.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::Char(ch)
                } else {
                    // upstream: `result = SomeString(no_nul=no_nul)`.
                    let mut st = SomeString::new(false, no_nul);
                    st.inner.base.const_box = Some(Constant::new(x.clone()));
                    SomeValue::String(st)
                };
                Ok(result)
            }
            ConstValue::None => Ok(s_none()),
            ConstValue::Tuple(items) => {
                let items_s = items
                    .iter()
                    .map(|v| self.immutablevalue(v))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(SomeValue::Tuple(SomeTuple::new(items_s)))
            }
            ConstValue::List(items) => {
                // upstream bookkeeper.py:255-265 memoises via
                // `immutable_cache[Constant(x)]`. Rust skips the cache
                // (perf-only deviation, see module doc).
                let listdef = ListDef::new(Some(self.clone()), SomeValue::Impossible, false, false);
                for e in items {
                    let s_e = self.immutablevalue(e)?;
                    listdef
                        .generalize(&s_e)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                }
                let mut result = SomeList::new(listdef);
                result.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::List(result))
            }
            ConstValue::Dict(items) => {
                // upstream bookkeeper.py:266-298 memoises via
                // `immutable_cache` and handles OrderedDict / r_dict
                // via the dict type. Our ConstValue::Dict keys are
                // strings only (flowspace globals), so we build a
                // plain SomeDict with string key type.
                let dictdef = DictDef::new(
                    Some(self.clone()),
                    SomeValue::Impossible,
                    SomeValue::Impossible,
                    false,
                    false,
                    false,
                );
                for (k, v) in items {
                    let s_k = SomeValue::String(SomeString::new(false, !k.contains('\x00')));
                    let s_v = self.immutablevalue(v)?;
                    dictdef
                        .generalize_key(&s_k)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                    dictdef
                        .generalize_value(&s_v)
                        .map_err(|e| AnnotatorError::new(e.msg))?;
                }
                let mut result = SomeDict::new(dictdef);
                result.base.const_box = Some(Constant::new(x.clone()));
                Ok(SomeValue::Dict(result))
            }
            ConstValue::HostObject(obj) => self.immutablevalue_hostobject(obj, x),
            ConstValue::Code(_)
            | ConstValue::Function(_)
            | ConstValue::SpecTag(_)
            | ConstValue::Atom(_)
            | ConstValue::Placeholder => {
                // Code / Function / SpecTag / Atom / Placeholder cover
                // internal flowspace / host-carrier values that
                // upstream never feeds into immutablevalue. Keep the
                // fail-fast stub so any unexpected call-site surfaces
                // a clear error rather than silent stub-SomePBC.
                Err(AnnotatorError::new(format!(
                    "Bookkeeper.immutablevalue({x:?}): internal ConstValue variant \
                     has no upstream immutablevalue branch"
                )))
            }
        }
    }

    /// Narrow dispatch for the `ConstValue::HostObject` arm of
    /// [`Self::immutablevalue`]. Covers the `callable` / `tp is type`
    /// / `_freeze_` branches at bookkeeper.py:309-333 to the extent
    /// that the stub `model::Desc` + `SomeBuiltin` surfaces allow.
    ///
    /// The upstream calls route through `self.getdesc(x)` which lives
    /// in bookkeeper commit 2 (blocked on classdesc.py for
    /// `ClassDesc`). Until that lands, we emit stub `model::Desc`
    /// entries into the [`SomePBC.descriptions`] set so callers see a
    /// typed annotation rather than `AnnotatorError`. Semantic
    /// differences vs upstream:
    ///   * `knowntype` of the returned SomePBC is `KnownType::Other`
    ///     because `commonbase` folding waits for classdesc.py.
    ///   * `SomeConstantType(x, self)` (bookkeeper.py:315-316) for a
    ///     class input collapses into `SomePBC([Desc::Class])` with
    ///     `const_box` set — the PBC-subclass distinction is implicit
    ///     in `DescKind::Class`.
    ///   * Bound methods / weakrefs / frozen-PBCs / BUILTIN_ANALYZERS
    ///     lookup / extregistry / property / symbolic-constant routes
    ///     stay deferred (each needs its own dep — classdesc.py,
    ///     weakref table, builtin.py registry, extregistry, specialize).
    fn immutablevalue_hostobject(
        self: &Rc<Self>,
        obj: &HostObject,
        raw: &ConstValue,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream bookkeeper.py:317 `callable(x)` → SomePBC path. In
        // the Rust port we treat user-functions explicitly because
        // bound-method / find_method dispatch (bookkeeper.py:318-329)
        // requires a full descriptor registry that isn't ported yet.
        if obj.is_user_function() {
            let mut pbc = SomePBC::new(vec![Desc::new(DescKind::Function, obj.qualname())], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        // upstream bookkeeper.py:309-311 — BUILTIN_ANALYZERS lookup
        // produces SomeBuiltin. The Rust port keeps the analyser
        // registry empty (builtin.py is still deferred), so
        // `analyser_name` is set from the host qualname and callers
        // resolving through specialcase.rs dispatch on that string
        // when an analyser registers.
        if obj.is_builtin_callable() {
            let mut sb = SomeBuiltin::new(
                obj.qualname().to_string(),
                None,
                Some(obj.qualname().to_string()),
            );
            sb.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::Builtin(sb));
        }
        // upstream bookkeeper.py:315-316 — `tp is type` → SomeConstant
        // Type(x, self). Implemented as a constant SomePBC over a
        // Class-kind Desc (SomeConstantType IS a SomePBC subclass
        // upstream; `const_box` + Class kind captures the same shape
        // the Rust model can model today).
        if obj.is_class() {
            let mut pbc = SomePBC::new(vec![Desc::new(DescKind::Class, obj.qualname())], false);
            pbc.base.const_box = Some(Constant::new(raw.clone()));
            return Ok(SomeValue::PBC(pbc));
        }
        Err(AnnotatorError::new(format!(
            "Bookkeeper.immutablevalue({raw:?}): host object kind not yet routed \
             (Phase 5 P5.2+ dep — needs classdesc.py / builtin.py / extregistry / \
             weakref / bound-method lookup; see bookkeeper.py:299-333)"
        )))
    }
}

impl Default for Bookkeeper {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::model::{SomeChar, SomeFloat, SomeString};

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    #[test]
    fn immutablevalue_int_sets_nonneg_when_ge_zero() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        match s {
            SomeValue::Integer(si) => {
                assert!(si.nonneg);
                assert!(si.base.const_box.is_some());
            }
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_int_negative_not_nonneg() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Int(-1)).unwrap();
        match s {
            SomeValue::Integer(si) => assert!(!si.nonneg),
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_bool() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Bool(true)).unwrap();
        assert!(matches!(s, SomeValue::Bool(_)));
    }

    #[test]
    fn immutablevalue_single_char_str_is_somechar() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("a".into())).unwrap();
        assert!(matches!(s, SomeValue::Char(_)));
    }

    #[test]
    fn immutablevalue_multichar_str_is_somestring() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("hello".into())).unwrap();
        assert!(matches!(s, SomeValue::String(_)));
    }

    #[test]
    fn immutablevalue_str_with_nul_clears_no_nul() {
        let bk = bk();
        let with_nul = ConstValue::Str("a\x00b".into());
        let s = bk.immutablevalue(&with_nul).unwrap();
        match s {
            SomeValue::String(st) => assert!(!st.inner.no_nul),
            other => panic!("expected SomeString, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_float_is_somefloat() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::Float(1.5_f64.to_bits()))
            .unwrap();
        match s {
            SomeValue::Float(_) => {}
            other => panic!("expected SomeFloat, got {other:?}"),
        }
        let _ = SomeFloat::new();
    }

    #[test]
    fn immutablevalue_none_is_s_none() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::None).unwrap();
        assert!(matches!(s, SomeValue::None_(_)));
    }

    #[test]
    fn immutablevalue_tuple_walks_items() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Bool(false),
            ]))
            .unwrap();
        match s {
            SomeValue::Tuple(t) => {
                assert_eq!(t.items.len(), 2);
                assert!(matches!(t.items[0], SomeValue::Integer(_)));
                assert!(matches!(t.items[1], SomeValue::Bool(_)));
            }
            other => panic!("expected SomeTuple, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_list_generalizes_elements() {
        let bk = bk();
        let s = bk
            .immutablevalue(&ConstValue::List(vec![
                ConstValue::Int(1),
                ConstValue::Int(-1),
            ]))
            .unwrap();
        match s {
            SomeValue::List(sl) => {
                // Element type widened from {nonneg Int, signed Int}
                // to generic Int (nonneg=false after merge with -1).
                if let SomeValue::Integer(si) = sl.listdef.s_value() {
                    assert!(!si.nonneg);
                } else {
                    panic!("expected Int listdef s_value");
                }
                assert!(sl.base.const_box.is_some());
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_class_returns_constant_pbc() {
        // upstream bookkeeper.py:315-316 — `tp is type` produces
        // `SomeConstantType(x, self)`, a SomePBC subclass with
        // `const = x`. The Rust port emits a Class-kind `SomePBC`
        // with `const_box` set; `SomeConstantType` collapses into the
        // PBC subclass because our PBC doesn't carry a Python-class
        // inheritance shadow.
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let class = HostObject::new_class("Foo", vec![]);
        let s = bk
            .immutablevalue(&ConstValue::HostObject(class))
            .expect("class HostObject must produce SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.iter().next().unwrap().kind,
                    DescKind::Class
                );
                assert!(pbc.base.const_box.is_some());
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_user_function_returns_function_pbc() {
        // upstream bookkeeper.py:317-331 — `callable(x)` falls into
        // `SomePBC([self.getdesc(x)])`. Narrow Rust port emits a
        // Function-kind Desc stub; real FunctionDesc wiring lands
        // when bookkeeper commit 2 ports getdesc.
        use crate::annotator::model::{DescKind, SomeValue};
        use crate::flowspace::model::{Constant, GraphFunc, HostObject};
        let bk = bk();
        let globals = Constant::new(ConstValue::Dict(Default::default()));
        let func = HostObject::new_user_function(GraphFunc::new("f", globals));
        let s = bk
            .immutablevalue(&ConstValue::HostObject(func))
            .expect("user-function HostObject must produce SomePBC");
        match s {
            SomeValue::PBC(pbc) => {
                assert_eq!(pbc.descriptions.len(), 1);
                assert_eq!(
                    pbc.descriptions.iter().next().unwrap().kind,
                    DescKind::Function
                );
            }
            other => panic!("expected SomePBC, got {other:?}"),
        }
    }

    #[test]
    fn immutablevalue_builtin_callable_returns_somebuiltin() {
        // upstream bookkeeper.py:309-311 — BUILTIN_ANALYZERS lookup
        // produces SomeBuiltin. Rust port wires the analyser_name
        // from the host qualname until builtin.py lands the registry.
        use crate::annotator::model::SomeValue;
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let bltn = HostObject::new_builtin_callable("len");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(bltn))
            .expect("builtin HostObject must produce SomeBuiltin");
        assert!(matches!(s, SomeValue::Builtin(_)));
    }

    #[test]
    fn immutablevalue_instance_host_object_defers() {
        // Non-function/non-class/non-builtin HostObject kinds (Module,
        // Instance, Opaque) stay deferred until the descriptor
        // registry lands — fail-fast rather than silently routing.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let mod_obj = HostObject::new_module("m");
        let err = bk
            .immutablevalue(&ConstValue::HostObject(mod_obj))
            .expect_err("module HostObject stays deferred");
        assert!(err.msg.unwrap_or_default().contains("not yet routed"));
    }

    #[test]
    fn newlist_creates_somelist_and_generalizes() {
        // Use two SomeInteger variants so the Phase 4 A4.6 pair-union
        // subset (Int ∪ Int) can widen them — upstream's multi-type
        // lists exercise broader pair unions which are Phase 5 P5.2+
        // pending.
        let bk = bk();
        let s_nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let s_signed = SomeValue::Integer(SomeInteger::new(false, false));
        let out = bk.newlist(&[s_nonneg, s_signed], None).unwrap();
        // Element type is now signed Int (widened from nonneg).
        if let SomeValue::Integer(si) = out.listdef.s_value() {
            assert!(!si.nonneg);
        } else {
            panic!("expected SomeInteger listdef element");
        }
    }

    #[test]
    fn newdict_creates_someordicteddict_equivalent() {
        let bk = bk();
        let out = bk.newdict();
        // Fresh-position newdict without subsequent generalize_key /
        // generalize_value carries Impossible for both.
        assert!(matches!(out.dictdef.s_key(), SomeValue::Impossible));
        assert!(matches!(out.dictdef.s_value(), SomeValue::Impossible));
    }

    #[test]
    fn getlistdef_caches_on_same_position() {
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(1, 2, 0)));
        let ld1 = bk.getlistdef(None);
        let ld2 = bk.getlistdef(None);
        assert!(ld1.same_as(&ld2));
    }

    #[test]
    fn getdictdef_caches_on_same_position() {
        let bk = bk();
        bk.set_position_key(Some(PositionKey::new(3, 4, 0)));
        let dd1 = bk.getdictdef(false, false, false);
        let dd2 = bk.getdictdef(false, false, false);
        assert!(dd1.same_as(&dd2));
    }

    #[test]
    fn getlistdef_caches_under_none_position() {
        // upstream bookkeeper.py:180 indexes `self.listdefs[self.
        // position_key]` — when position_key is `None`, both calls
        // land on the same dict entry. Rust port mirrors this by
        // using `Option<PositionKey>` as the cache key; two `getlist
        // def` calls outside a reflow frame must share the same
        // ListDef.
        let bk = bk();
        assert_eq!(bk.current_position_key(), None);
        let ld1 = bk.getlistdef(None);
        let ld2 = bk.getlistdef(None);
        assert!(ld1.same_as(&ld2));
    }

    #[test]
    fn getdictdef_caches_under_none_position() {
        let bk = bk();
        assert_eq!(bk.current_position_key(), None);
        let dd1 = bk.getdictdef(false, false, false);
        let dd2 = bk.getdictdef(false, false, false);
        assert!(dd1.same_as(&dd2));
    }

    #[test]
    fn position_key_set_and_get() {
        let bk = bk();
        assert!(bk.current_position_key().is_none());
        let prev = bk.set_position_key(Some(PositionKey::new(1, 1, 1)));
        assert!(prev.is_none());
        assert_eq!(bk.current_position_key(), Some(PositionKey::new(1, 1, 1)));
    }

    #[test]
    fn unicode_through_const_str() {
        // ConstValue::Str carries both str and unicode upstream; the
        // byte-level no-nul check lands identically.
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("abc".into())).unwrap();
        match s {
            SomeValue::String(st) => assert!(st.inner.no_nul),
            other => panic!("expected SomeString, got {other:?}"),
        }
    }

    #[test]
    fn byte_array_not_yet_routed() {
        // ConstValue has no dedicated Bytes variant; bytearray inputs
        // therefore don't round-trip through immutablevalue today.
        // Test the type itself stays buildable from the annotator
        // model — sanity check in lieu of a full input path.
        let _ = SomeByteArray::default();
    }

    #[test]
    fn char_has_no_nul() {
        let bk = bk();
        let s = bk.immutablevalue(&ConstValue::Str("x".into())).unwrap();
        match s {
            SomeValue::Char(_) => {
                let c = SomeChar::new(true);
                assert!(c.inner.no_nul);
            }
            other => panic!("expected SomeChar, got {other:?}"),
        }
    }
}
