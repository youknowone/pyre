//! RPython `rpython/rtyper/rtyper.py` — `RPythonTyper`, `HighLevelOp`,
//! `LowLevelOpList`.
//!
//! Mirrors upstream module layout: all three types live in the same
//! file (`rtyper.py:47 RPythonTyper`, `rtyper.py:617 HighLevelOp`,
//! `rtyper.py:783 LowLevelOpList`) because the `specialize_block →
//! highlevelops → translate_hl_to_ll` dispatch loop ties them
//! together.
//!
//! Current scope ports the core `HighLevelOp` carrier methods
//! (`setup`, `copy`, `dispatch`, `inputarg`, `inputargs`,
//! `exception_is_here`, `exception_cannot_occur`, `r_s_pop`,
//! `v_s_insertfirstarg`, `swap_fst_snd_args`) and the `LowLevelOpList`
//! buffer methods (`append`, `convertvar`, `genop`) so concrete Repr
//! ports can plug into the same surface as upstream.
//!
//! Deferred methods are documented inline with their upstream line
//! reference.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::model::SomeValue;
use crate::flowspace::argument::Signature;
use crate::flowspace::model::{
    Block, BlockKey, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphFunc,
    GraphRef, HOST_ENV, Hlvalue, Link, LinkRef, SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::llannotation::lltype_to_annotation;
use crate::translator::rtyper::lltypesystem::lltype::{
    _ptr, LowLevelType, PtrTarget, getfunctionptr,
};
use crate::translator::rtyper::rmodel::{
    RTypeResult, Repr, ReprKey, inputconst, inputconst_from_lltype, rtyper_makekey, rtyper_makerepr,
};
use crate::translator::rtyper::rpbc::LLCallTable;

/// RPython `class RPythonTyper(object)` (rtyper.py:42+).
///
/// The full constructor state lands incrementally as the rtyper port
/// progresses. For `simplify.py` parity we need the annotator link and
/// the `already_seen` dict populated by `specialize_more_blocks()`.
pub struct RPythonTyper {
    /// RPython `self.annotator`.
    ///
    /// Rust stores this edge weakly to avoid the
    /// `annotator -> translator -> rtyper -> annotator` cycle that
    /// Python's GC handles upstream.
    pub annotator: Weak<RPythonAnnotator>,
    /// RPython `self.already_seen = {}` assigned in `specialize()`
    /// (rtyper.py:186). Membership is queried by `simplify.py`.
    pub already_seen: RefCell<HashMap<BlockKey, bool>>,
    /// RPython `self.concrete_calltables = {}` assigned in `__init__`
    /// (rtyper.py:57).
    pub concrete_calltables: RefCell<HashMap<usize, (LLCallTable, usize)>>,
    /// RPython `self.reprs = {}` (`rtyper.py:54`) — cache keyed by
    /// `s_obj.rtyper_makekey()`. Pyre stores `Option<Arc<dyn Repr>>`
    /// because upstream pre-inserts `None` before calling
    /// `rtyper_makerepr` to detect recursive `getrepr()` (rtyper.py:156).
    pub reprs: RefCell<HashMap<ReprKey, Option<Arc<dyn Repr>>>>,
    /// RPython `self._reprs_must_call_setup = []` (`rtyper.py:55`).
    pub reprs_must_call_setup: RefCell<Vec<Arc<dyn Repr>>>,
    /// RPython `self._seen_reprs_must_call_setup = {}` (`rtyper.py:56`).
    ///
    /// Pyre stores pointer-identity fingerprints so Arc-equal Reprs
    /// are deduped without requiring `Hash + Eq` on the trait object.
    pub seen_reprs_must_call_setup: RefCell<Vec<*const ()>>,
    /// RPython `self.primitive_to_repr = {}` (`rtyper.py:53`).
    pub primitive_to_repr: RefCell<HashMap<LowLevelType, Arc<dyn Repr>>>,
    /// RPython low-level helper graphs are cached by
    /// `FunctionDesc.cachedgraph()` under `annlowlevel.py`'s
    /// `LowLevelAnnotatorPolicy.lowlevelspecialize`. The Rust port has
    /// no host Python helper function object to hang that cache from, so
    /// the rtyper owns the equivalent graph cache for `ll_*` helpers.
    lowlevel_helper_graphs: RefCell<HashMap<LowLevelHelperKey, Rc<PyGraph>>>,
}

impl RPythonTyper {
    /// RPython `RPythonTyper.__init__(self, annotator, backend=...)`.
    ///
    /// Only the fields required by current simplify parity are seeded
    /// here; additional constructor state lands with the full rtyper
    /// port.
    pub fn new(annotator: &Rc<RPythonAnnotator>) -> Self {
        RPythonTyper {
            annotator: Rc::downgrade(annotator),
            already_seen: RefCell::new(HashMap::new()),
            concrete_calltables: RefCell::new(HashMap::new()),
            reprs: RefCell::new(HashMap::new()),
            reprs_must_call_setup: RefCell::new(Vec::new()),
            seen_reprs_must_call_setup: RefCell::new(Vec::new()),
            primitive_to_repr: RefCell::new(HashMap::new()),
            lowlevel_helper_graphs: RefCell::new(HashMap::new()),
        }
    }

    /// RPython `RPythonTyper.getprimitiverepr(self, lltype)`
    /// (`rtyper.py:85-93`).
    pub fn getprimitiverepr(&self, lltype: &LowLevelType) -> Result<Arc<dyn Repr>, TyperError> {
        if let Some(repr) = self.primitive_to_repr.borrow().get(lltype) {
            return Ok(repr.clone());
        }
        if !is_primitive_lowleveltype(lltype) {
            return Err(TyperError::message(format!(
                "There is no primitive repr for {:?}",
                lltype
            )));
        }
        let repr = self.getrepr(&lltype_to_annotation(lltype.clone()))?;
        self.primitive_to_repr
            .borrow_mut()
            .insert(lltype.clone(), repr.clone());
        Ok(repr)
    }

    /// Test/debug helper mirroring `self.already_seen[block] = True`
    /// in `specialize_more_blocks()` (rtyper.py:225).
    pub fn mark_already_seen(&self, block: &BlockRef) {
        self.already_seen
            .borrow_mut()
            .insert(BlockKey::of(block), true);
    }

    /// RPython `RPythonTyper.getcallable(self, graph)` (rtyper.py:569-581).
    pub fn getcallable(&self, graph: &Rc<PyGraph>) -> Result<_ptr, TyperError> {
        getfunctionptr(&graph.graph, |v| {
            self.bindingrepr(v).map(|r| r.lowleveltype().clone())
        })
    }

    /// RPython `annlowlevel.annotate_lowlevel_helper()` +
    /// `FunctionDesc.cachedgraph()` for low-level helper functions used
    /// by `LowLevelOpList.gendirectcall`.
    pub fn lowlevel_helper_function(
        &self,
        name: impl Into<String>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
    ) -> Result<LowLevelFunction, TyperError> {
        let name = name.into();
        let key = LowLevelHelperKey {
            name: name.clone(),
            args: args.clone(),
            result: result.clone(),
        };
        if let Some(graph) = self.lowlevel_helper_graphs.borrow().get(&key).cloned() {
            return Ok(LowLevelFunction::from_pygraph(name, args, result, graph));
        }

        let graph = Rc::new(lowlevel_helper_graph(self, &name, &args, &result)?);
        self.lowlevel_helper_graphs
            .borrow_mut()
            .insert(key, graph.clone());

        let annotator = self
            .annotator
            .upgrade()
            .ok_or_else(|| TyperError::message("RPythonTyper.annotator weak reference dropped"))?;
        annotator
            .translator
            .graphs
            .borrow_mut()
            .push(graph.graph.clone());
        Ok(LowLevelFunction::from_pygraph(name, args, result, graph))
    }

    /// RPython `RPythonTyper.annotation(self, var)` (rtyper.py:166-168).
    ///
    /// ```python
    /// def annotation(self, var):
    ///     s_obj = self.annotator.annotation(var)
    ///     return s_obj
    /// ```
    pub fn annotation(&self, var: &Hlvalue) -> Option<SomeValue> {
        self.annotator.upgrade().and_then(|ann| ann.annotation(var))
    }

    /// RPython `RPythonTyper.binding(self, var)` (rtyper.py:170-172).
    ///
    /// ```python
    /// def binding(self, var):
    ///     s_obj = self.annotator.binding(var)
    ///     return s_obj
    /// ```
    ///
    /// Panics on missing binding (matching upstream's `KeyError`);
    /// callers handle missing bindings with `annotation()` first.
    pub fn binding(&self, var: &Hlvalue) -> SomeValue {
        let ann = self
            .annotator
            .upgrade()
            .expect("RPythonTyper.annotator weak reference dropped");
        ann.binding(var)
    }

    /// RPython `RPythonTyper.add_pendingsetup(self, repr)`
    /// (rtyper.py:105-111).
    ///
    /// ```python
    /// def add_pendingsetup(self, repr):
    ///     assert isinstance(repr, Repr)
    ///     if repr in self._seen_reprs_must_call_setup:
    ///         return
    ///     self._reprs_must_call_setup.append(repr)
    ///     self._seen_reprs_must_call_setup[repr] = True
    /// ```
    pub fn add_pendingsetup(&self, repr: Arc<dyn Repr>) {
        let fingerprint = Arc::as_ptr(&repr) as *const ();
        let mut seen = self.seen_reprs_must_call_setup.borrow_mut();
        if seen.contains(&fingerprint) {
            return;
        }
        seen.push(fingerprint);
        self.reprs_must_call_setup.borrow_mut().push(repr);
    }

    /// RPython `RPythonTyper.getrepr(self, s_obj)` (rtyper.py:149-164).
    ///
    /// ```python
    /// def getrepr(self, s_obj):
    ///     key = s_obj.rtyper_makekey()
    ///     assert key[0] is s_obj.__class__
    ///     try:
    ///         result = self.reprs[key]
    ///     except KeyError:
    ///         self.reprs[key] = None
    ///         result = s_obj.rtyper_makerepr(self)
    ///         assert not isinstance(result.lowleveltype, ContainerType), ...
    ///         self.reprs[key] = result
    ///         self.add_pendingsetup(result)
    ///     assert result is not None     # recursive getrepr()!
    ///     return result
    /// ```
    pub fn getrepr(&self, s_obj: &SomeValue) -> Result<Arc<dyn Repr>, TyperError> {
        let key = rtyper_makekey(s_obj);
        // Fast-path hit: entry present and Some → return clone.
        if let Some(slot) = self.reprs.borrow().get(&key) {
            return match slot {
                Some(repr) => Ok(repr.clone()),
                None => Err(TyperError::message(format!(
                    "recursive getrepr() for {s_obj:?}"
                ))),
            };
        }
        // First time seeing this key: upstream pre-inserts None as the
        // recursion sentinel, then materialises the Repr.
        self.reprs.borrow_mut().insert(key.clone(), None);
        let result = rtyper_makerepr(s_obj, self)?;
        self.reprs.borrow_mut().insert(key, Some(result.clone()));
        self.add_pendingsetup(result.clone());
        Ok(result)
    }

    /// RPython `RPythonTyper.bindingrepr(self, var)` (rtyper.py:174-175).
    ///
    /// ```python
    /// def bindingrepr(self, var):
    ///     return self.getrepr(self.binding(var))
    /// ```
    pub fn bindingrepr(&self, var: &Hlvalue) -> Result<Arc<dyn Repr>, TyperError> {
        let s_obj = self.binding(var);
        self.getrepr(&s_obj)
    }

    /// RPython `RPythonTyper.call_all_setups(self)` (rtyper.py:243-256).
    ///
    /// ```python
    /// def call_all_setups(self):
    ///     must_setup_more = []
    ///     delayed = []
    ///     while self._reprs_must_call_setup:
    ///         r = self._reprs_must_call_setup.pop()
    ///         if r.is_setup_delayed():
    ///             delayed.append(r)
    ///         else:
    ///             r.setup()
    ///             must_setup_more.append(r)
    ///     for r in must_setup_more:
    ///         r.setup_final()
    ///     self._reprs_must_call_setup.extend(delayed)
    /// ```
    pub fn call_all_setups(&self) -> Result<(), TyperError> {
        let mut must_setup_more: Vec<Arc<dyn Repr>> = Vec::new();
        let mut delayed: Vec<Arc<dyn Repr>> = Vec::new();
        loop {
            let r = self.reprs_must_call_setup.borrow_mut().pop();
            let Some(r) = r else {
                break;
            };
            if r.is_setup_delayed() {
                delayed.push(r);
            } else {
                r.setup()?;
                must_setup_more.push(r);
            }
        }
        for r in &must_setup_more {
            r.setup_final()?;
        }
        self.reprs_must_call_setup.borrow_mut().extend(delayed);
        Ok(())
    }

    /// RPython `HighLevelOp.dispatch` target (`rtyper.py:648-653`).
    pub fn translate_operation(&self, hop: &HighLevelOp) -> RTypeResult {
        match hop.spaceop.opname.as_str() {
            // rtyper.py:497-518 registers `translate_op_<opname>` for
            // unary/binary operations. Unary ops route to `Repr.rtype_*`;
            // binary ops route through the explicit pairtype dispatcher,
            // matching `pair(r_arg1, r_arg2).rtype_*`.
            "getattr" => self.translate_unary_operation(hop, |r, hop| r.rtype_getattr(hop)),
            "setattr" => self.translate_unary_operation(hop, |r, hop| r.rtype_setattr(hop)),
            "len" => self.translate_unary_operation(hop, |r, hop| r.rtype_len(hop)),
            "bool" => self.translate_unary_operation(hop, |r, hop| r.rtype_bool(hop)),
            "simple_call" => self.translate_unary_operation(hop, |r, hop| r.rtype_simple_call(hop)),
            "neg" => self.translate_unary_operation(hop, |r, hop| r.rtype_neg(hop)),
            "neg_ovf" => self.translate_unary_operation(hop, |r, hop| r.rtype_neg_ovf(hop)),
            "pos" => self.translate_unary_operation(hop, |r, hop| r.rtype_pos(hop)),
            "abs" => self.translate_unary_operation(hop, |r, hop| r.rtype_abs(hop)),
            "abs_ovf" => self.translate_unary_operation(hop, |r, hop| r.rtype_abs_ovf(hop)),
            "int" => self.translate_unary_operation(hop, |r, hop| r.rtype_int(hop)),
            "float" => self.translate_unary_operation(hop, |r, hop| r.rtype_float(hop)),
            "invert" => self.translate_unary_operation(hop, |r, hop| r.rtype_invert(hop)),
            "bin" => self.translate_unary_operation(hop, |r, hop| r.rtype_bin(hop)),
            "call_args" => self.translate_unary_operation(hop, |r, hop| r.rtype_call_args(hop)),
            "delattr" => self.translate_unary_operation(hop, |r, hop| r.rtype_delattr(hop)),
            "delslice" => self.translate_unary_operation(hop, |r, hop| r.rtype_delslice(hop)),
            "getslice" => self.translate_unary_operation(hop, |r, hop| r.rtype_getslice(hop)),
            "hash" => self.translate_unary_operation(hop, |r, hop| r.rtype_hash(hop)),
            "hex" => self.translate_unary_operation(hop, |r, hop| r.rtype_hex(hop)),
            "hint" => self.translate_unary_operation(hop, |r, hop| r.rtype_hint(hop)),
            "id" => self.translate_unary_operation(hop, |r, hop| r.rtype_id(hop)),
            "isinstance" => self.translate_unary_operation(hop, |r, hop| r.rtype_isinstance(hop)),
            "issubtype" => self.translate_unary_operation(hop, |r, hop| r.rtype_issubtype(hop)),
            "iter" => self.translate_unary_operation(hop, |r, hop| r.rtype_iter(hop)),
            "long" => self.translate_unary_operation(hop, |r, hop| r.rtype_long(hop)),
            "next" => self.translate_unary_operation(hop, |r, hop| r.rtype_next(hop)),
            "oct" => self.translate_unary_operation(hop, |r, hop| r.rtype_oct(hop)),
            "ord" => self.translate_unary_operation(hop, |r, hop| r.rtype_ord(hop)),
            "repr" => self.translate_unary_operation(hop, |r, hop| r.rtype_repr(hop)),
            "setslice" => self.translate_unary_operation(hop, |r, hop| r.rtype_setslice(hop)),
            "str" => self.translate_unary_operation(hop, |r, hop| r.rtype_str(hop)),
            "type" => self.translate_unary_operation(hop, |r, hop| r.rtype_type(hop)),
            "getitem" | "getitem_idx" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_getitem)
            }
            "setitem" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_setitem),
            "delitem" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_delitem),
            "is_" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_is_),
            "eq" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_eq),
            "ne" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_ne),
            "lt" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_lt),
            "le" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_le),
            "gt" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_gt),
            "ge" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_ge),
            "add" | "inplace_add" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_add)
            }
            "add_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_add_ovf),
            "sub" | "inplace_sub" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_sub)
            }
            "sub_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_sub_ovf),
            "mul" | "inplace_mul" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_mul)
            }
            "mul_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_mul_ovf),
            "truediv" | "inplace_truediv" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_truediv)
            }
            "floordiv" | "inplace_floordiv" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_floordiv)
            }
            "floordiv_ovf" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_floordiv_ovf)
            }
            "div" | "inplace_div" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_div)
            }
            "div_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_div_ovf),
            "mod" | "inplace_mod" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_mod)
            }
            "mod_ovf" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_mod_ovf),
            "xor" | "inplace_xor" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_xor)
            }
            "and_" | "inplace_and" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_and_)
            }
            "or_" | "inplace_or" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_or_)
            }
            "lshift" | "inplace_lshift" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_lshift)
            }
            "lshift_ovf" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_lshift_ovf)
            }
            "rshift" | "inplace_rshift" => {
                self.translate_pair_operation(hop, super::pairtype::pair_rtype_rshift)
            }
            "cmp" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_cmp),
            "coerce" => self.translate_pair_operation(hop, super::pairtype::pair_rtype_coerce),
            _ => self.default_translate_operation(hop),
        }
    }

    fn translate_unary_operation<F>(&self, hop: &HighLevelOp, method: F) -> RTypeResult
    where
        F: Fn(&dyn Repr, &HighLevelOp) -> RTypeResult,
    {
        let r_arg1 = self.hop_arg_repr(hop, 0)?;
        method(r_arg1.as_ref(), hop)
    }

    fn translate_pair_operation(
        &self,
        hop: &HighLevelOp,
        method: fn(&dyn Repr, &dyn Repr, &HighLevelOp) -> RTypeResult,
    ) -> RTypeResult {
        let r_arg1 = self.hop_arg_repr(hop, 0)?;
        let r_arg2 = self.hop_arg_repr(hop, 1)?;
        method(r_arg1.as_ref(), r_arg2.as_ref(), hop)
    }

    fn hop_arg_repr(&self, hop: &HighLevelOp, index: usize) -> Result<Arc<dyn Repr>, TyperError> {
        let args_r = hop.args_r.borrow();
        let r = args_r.get(index).ok_or_else(|| {
            TyperError::message(format!(
                "translate_op_{} missing argument repr {}",
                hop.spaceop.opname, index
            ))
        })?;
        r.clone().ok_or_else(|| {
            TyperError::message(format!(
                "translate_op_{} argument repr {} is not set",
                hop.spaceop.opname, index
            ))
        })
    }

    /// RPython `RPythonTyper.default_translate_operation`
    /// (`rtyper.py:557-558`).
    pub fn default_translate_operation(&self, hop: &HighLevelOp) -> RTypeResult {
        Err(TyperError::message(format!(
            "unimplemented operation: '{}'",
            hop.spaceop.opname
        )))
    }
}

fn is_primitive_lowleveltype(lltype: &LowLevelType) -> bool {
    matches!(
        lltype,
        LowLevelType::Void
            | LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::Bool
            | LowLevelType::Float
            | LowLevelType::SingleFloat
            | LowLevelType::LongFloat
            | LowLevelType::Char
            | LowLevelType::UniChar
    )
}

// ____________________________________________________________
// HighLevelOp — `rtyper.py:617-779`.

/// RPython `HighLevelOp.inputarg(converted_to, arg)` accepts either a
/// `Repr` instance or a primitive low-level type. Rust makes that overload
/// explicit.
pub enum ConvertedTo<'a> {
    Repr(&'a dyn Repr),
    LowLevelType(&'a LowLevelType),
}

impl<'a, T: Repr> From<&'a T> for ConvertedTo<'a> {
    fn from(value: &'a T) -> Self {
        ConvertedTo::Repr(value)
    }
}

impl<'a> From<&'a dyn Repr> for ConvertedTo<'a> {
    fn from(value: &'a dyn Repr) -> Self {
        ConvertedTo::Repr(value)
    }
}

impl<'a> From<&'a Arc<dyn Repr>> for ConvertedTo<'a> {
    fn from(value: &'a Arc<dyn Repr>) -> Self {
        ConvertedTo::Repr(value.as_ref())
    }
}

impl<'a> From<&'a LowLevelType> for ConvertedTo<'a> {
    fn from(value: &'a LowLevelType) -> Self {
        ConvertedTo::LowLevelType(value)
    }
}

enum ResolvedConvertedTo<'a> {
    Borrowed(&'a dyn Repr),
    Owned(Arc<dyn Repr>),
}

impl<'a> ResolvedConvertedTo<'a> {
    fn as_repr(&self) -> &dyn Repr {
        match self {
            ResolvedConvertedTo::Borrowed(repr) => *repr,
            ResolvedConvertedTo::Owned(repr) => repr.as_ref(),
        }
    }
}

/// RPython `class HighLevelOp(object)` (rtyper.py:617-779).
///
/// The per-operation carrier passed to every `translate_op_*` +
/// `Repr.rtype_*` method during `specialize_block`. Fields populated
/// in two stages:
///
/// * Construction (`HighLevelOp.__init__`, rtyper.py:619-623) — stores
///   the rtyper/spaceop/exceptionlinks/llops handles.
/// * `setup()` (rtyper.py:625-633) — materialises `args_v`, `args_s`,
///   `s_result`, `args_r`, `r_result` by querying the annotator and
///   the rtyper. Calls can still surface `MissingRTypeOperation` until
///   each concrete `SomeValue.rtyper_makerepr` arm is ported.
pub struct HighLevelOp {
    /// RPython `self.rtyper = rtyper` (rtyper.py:620).
    pub rtyper: Rc<RPythonTyper>,
    /// RPython `self.spaceop = spaceop` (rtyper.py:621).
    pub spaceop: SpaceOperation,
    /// RPython `self.exceptionlinks = exceptionlinks` (rtyper.py:622).
    /// Set of exceptional successor links collected by
    /// `highlevelops(...)` when a block raises
    /// (`rtyper.py:428-431`).
    pub exceptionlinks: Vec<LinkRef>,
    /// RPython `self.llops = llops` (rtyper.py:623) — shared mutable
    /// low-level op buffer across all hops of the block. Pyre wraps
    /// in `Rc<RefCell<_>>` to mirror Python's by-reference sharing.
    pub llops: Rc<RefCell<LowLevelOpList>>,

    // Fields populated by `setup()` — upstream initialises these
    // lazily (rtyper.py:625-633). Pyre mirrors with `RefCell<Option>`
    // so concrete Reprs can be filled in order without forcing a
    // one-shot `setup()` contract on the Rust side.
    /// RPython `self.args_v = list(spaceop.args)` (rtyper.py:628).
    pub args_v: RefCell<Vec<Hlvalue>>,
    /// RPython `self.args_s = [rtyper.binding(a) for a in spaceop.args]`
    /// (rtyper.py:629).
    pub args_s: RefCell<Vec<SomeValue>>,
    /// RPython `self.s_result = rtyper.binding(spaceop.result)`
    /// (rtyper.py:630).
    pub s_result: RefCell<Option<SomeValue>>,
    /// RPython `self.args_r = [rtyper.getrepr(s_a) for s_a in
    /// self.args_s]` (rtyper.py:631). `None` entries mean the
    /// corresponding Repr has not been materialised yet (upstream
    /// requires all to be filled before `dispatch()` runs).
    pub args_r: RefCell<Vec<Option<Arc<dyn Repr>>>>,
    /// RPython `self.r_result = rtyper.getrepr(self.s_result)`
    /// (rtyper.py:632).
    pub r_result: RefCell<Option<Arc<dyn Repr>>>,
}

impl HighLevelOp {
    /// RPython `HighLevelOp.__init__(self, rtyper, spaceop,
    /// exceptionlinks, llops)` (rtyper.py:619-623).
    pub fn new(
        rtyper: Rc<RPythonTyper>,
        spaceop: SpaceOperation,
        exceptionlinks: Vec<LinkRef>,
        llops: Rc<RefCell<LowLevelOpList>>,
    ) -> Self {
        HighLevelOp {
            rtyper,
            spaceop,
            exceptionlinks,
            llops,
            args_v: RefCell::new(Vec::new()),
            args_s: RefCell::new(Vec::new()),
            s_result: RefCell::new(None),
            args_r: RefCell::new(Vec::new()),
            r_result: RefCell::new(None),
        }
    }

    /// RPython `HighLevelOp.nb_args` property (rtyper.py:636-637).
    pub fn nb_args(&self) -> usize {
        self.args_v.borrow().len()
    }

    /// RPython `HighLevelOp.copy(self)` (`rtyper.py:639-646`).
    pub fn copy(&self) -> Self {
        let result = HighLevelOp::new(
            self.rtyper.clone(),
            self.spaceop.clone(),
            self.exceptionlinks.clone(),
            self.llops.clone(),
        );
        *result.args_v.borrow_mut() = self.args_v.borrow().clone();
        *result.args_s.borrow_mut() = self.args_s.borrow().clone();
        *result.s_result.borrow_mut() = self.s_result.borrow().clone();
        *result.args_r.borrow_mut() = self.args_r.borrow().clone();
        *result.r_result.borrow_mut() = self.r_result.borrow().clone();
        result
    }

    /// RPython `HighLevelOp.dispatch(self)` (`rtyper.py:648-653`).
    pub fn dispatch(&self) -> RTypeResult {
        self.rtyper.translate_operation(self)
    }

    /// RPython `HighLevelOp.setup(self)` (rtyper.py:625-633).
    ///
    /// ```python
    /// def setup(self):
    ///     rtyper = self.rtyper
    ///     spaceop = self.spaceop
    ///     self.args_v   = list(spaceop.args)
    ///     self.args_s   = [rtyper.binding(a) for a in spaceop.args]
    ///     self.s_result = rtyper.binding(spaceop.result)
    ///     self.args_r   = [rtyper.getrepr(s_a) for s_a in self.args_s]
    ///     self.r_result = rtyper.getrepr(self.s_result)
    ///     rtyper.call_all_setups()
    /// ```
    ///
    /// `getrepr` arms for non-Impossible SomeValue variants currently
    /// surface `MissingRTypeOperation` — cascading port work fills
    /// them in (rint.rs, rfloat.rs, rpbc.rs, ...). Callers that hit
    /// the error surface know exactly which upstream module to land
    /// next.
    pub fn setup(&self) -> Result<(), TyperError> {
        let rtyper = &self.rtyper;
        *self.args_v.borrow_mut() = self.spaceop.args.clone();
        let args_s: Vec<SomeValue> = self
            .spaceop
            .args
            .iter()
            .map(|a| rtyper.binding(a))
            .collect();
        *self.s_result.borrow_mut() = Some(rtyper.binding(&self.spaceop.result));
        let mut args_r: Vec<Option<Arc<dyn Repr>>> = Vec::with_capacity(args_s.len());
        for s_a in &args_s {
            args_r.push(Some(rtyper.getrepr(s_a)?));
        }
        *self.args_r.borrow_mut() = args_r;
        let s_result_clone = self.s_result.borrow().clone();
        if let Some(s_r) = s_result_clone {
            *self.r_result.borrow_mut() = Some(rtyper.getrepr(&s_r)?);
        }
        *self.args_s.borrow_mut() = args_s;
        rtyper.call_all_setups()?;
        Ok(())
    }

    fn resolve_converted_to<'a>(
        &self,
        converted_to: ConvertedTo<'a>,
    ) -> Result<ResolvedConvertedTo<'a>, TyperError> {
        match converted_to {
            ConvertedTo::Repr(repr) => Ok(ResolvedConvertedTo::Borrowed(repr)),
            ConvertedTo::LowLevelType(lltype) => Ok(ResolvedConvertedTo::Owned(
                self.rtyper.getprimitiverepr(lltype)?,
            )),
        }
    }

    /// RPython `HighLevelOp.inputarg(self, converted_to, arg)`
    /// (`rtyper.py:655-673`).
    pub fn inputarg<'a>(
        &self,
        converted_to: impl Into<ConvertedTo<'a>>,
        arg: usize,
    ) -> Result<Hlvalue, TyperError> {
        let converted_to = self.resolve_converted_to(converted_to.into())?;
        let v = self.args_v.borrow()[arg].clone();
        if let Hlvalue::Constant(c) = &v {
            return inputconst(converted_to.as_repr(), &c.value).map(Hlvalue::Constant);
        }

        let s_binding = self.args_s.borrow()[arg].clone();
        if let Some(value) = s_binding.const_() {
            return inputconst(converted_to.as_repr(), value).map(Hlvalue::Constant);
        }

        let r_binding = self.args_r.borrow()[arg]
            .clone()
            .ok_or_else(|| TyperError::message("HighLevelOp.inputarg missing source repr"))?;
        self.llops
            .borrow_mut()
            .convertvar(v, r_binding.as_ref(), converted_to.as_repr())
    }

    /// RPython `HighLevelOp.inputconst = staticmethod(inputconst)`
    /// (`rtyper.py:675`).
    pub fn inputconst<'a>(
        converted_to: impl Into<ConvertedTo<'a>>,
        value: &ConstValue,
    ) -> Result<Constant, TyperError> {
        match converted_to.into() {
            ConvertedTo::Repr(repr) => inputconst(repr, value),
            ConvertedTo::LowLevelType(lltype) => inputconst_from_lltype(lltype, value),
        }
    }

    /// RPython `HighLevelOp.inputargs(self, *converted_to)`
    /// (`rtyper.py:677-685`).
    pub fn inputargs<'a>(
        &self,
        converted_to: Vec<ConvertedTo<'a>>,
    ) -> Result<Vec<Hlvalue>, TyperError> {
        if converted_to.len() != self.nb_args() {
            return Err(TyperError::message(format!(
                "operation argument count mismatch:\n'{}' has {} arguments, rtyper wants {}",
                self.spaceop.opname,
                self.nb_args(),
                converted_to.len()
            )));
        }
        let mut vars = Vec::with_capacity(converted_to.len());
        for (i, converted_to) in converted_to.into_iter().enumerate() {
            vars.push(self.inputarg(converted_to, i)?);
        }
        Ok(vars)
    }

    /// RPython `HighLevelOp.genop(self, opname, args_v, resulttype=None)`
    /// (`rtyper.py:687-688`).
    pub fn genop(
        &self,
        opname: &str,
        args_v: Vec<Hlvalue>,
        resulttype: GenopResult,
    ) -> Option<Hlvalue> {
        self.llops
            .borrow_mut()
            .genop(opname, args_v, resulttype)
            .map(Hlvalue::Variable)
    }

    /// RPython `HighLevelOp.gendirectcall(self, ll_function, *args_v)`
    /// (`rtyper.py:690-691`).
    pub fn gendirectcall(
        &self,
        ll_function: &LowLevelFunction,
        args_v: Vec<Hlvalue>,
    ) -> Result<Option<Hlvalue>, TyperError> {
        Ok(self
            .llops
            .borrow_mut()
            .gendirectcall(ll_function, args_v)?
            .map(Hlvalue::Variable))
    }

    /// RPython `HighLevelOp.has_implicit_exception(self, exc_cls)`
    /// (rtyper.py:713-729).
    ///
    pub fn has_implicit_exception(&self, exc_cls_name: &str) -> bool {
        let mut llops = self.llops.borrow_mut();
        if llops.llop_raising_exceptions.is_some() {
            panic!("already generated the llop that raises the exception");
        }
        if self.exceptionlinks.is_empty() {
            return false;
        }
        let exc_cls = HOST_ENV
            .lookup_exception_class(exc_cls_name)
            .unwrap_or_else(|| panic!("unknown exception class {exc_cls_name}"));
        let checked = llops
            .implicit_exceptions_checked
            .get_or_insert_with(Vec::new);
        let mut result = false;
        for link in &self.exceptionlinks {
            let exitcase = link.borrow().exitcase.clone();
            let Some(Hlvalue::Constant(Constant {
                value: ConstValue::HostObject(exit_cls),
                ..
            })) = exitcase
            else {
                continue;
            };
            if exc_cls.is_subclass_of(&exit_cls) {
                checked.push(exit_cls.qualname().to_string());
                result = true;
            }
        }
        result
    }

    /// RPython `HighLevelOp.exception_is_here(self)` (rtyper.py:731-745).
    ///
    /// Stores the index of the current llop as the "raising" llop.
    pub fn exception_is_here(&self) -> Result<(), TyperError> {
        let mut llops = self.llops.borrow_mut();
        llops._called_exception_is_here_or_cannot_occur = true;
        if llops.llop_raising_exceptions.is_some() {
            return Err(TyperError::message(
                "cannot catch an exception at more than one llop",
            ));
        }
        if self.exceptionlinks.is_empty() {
            return Ok(()); // rtyper.py:735-736
        }
        if let Some(_checked) = &llops.implicit_exceptions_checked {
            // upstream sanity check rtyper.py:737-744: every
            // exceptionlink.exitcase must appear in
            // implicit_exceptions_checked. Pyre's `Link.exitcase` is
            // `Option<Hlvalue>` carrying the exception class as a
            // host-object constant; DEFERRED until
            // `rpython/rtyper/exceptiondata.py` ports so the lookup
            // has a canonical `ExceptionData.lltype_of_exception_type`
            // surface to compare against. Parity-marker left in the
            // typed field so the follow-up commit finds the wiring
            // spot.
            let _ = &self.exceptionlinks;
        }
        llops.llop_raising_exceptions = Some(LlopRaisingExceptions::Index(llops.ops.len()));
        Ok(())
    }

    /// RPython `HighLevelOp.exception_cannot_occur(self)`
    /// (rtyper.py:747-753).
    pub fn exception_cannot_occur(&self) -> Result<(), TyperError> {
        let mut llops = self.llops.borrow_mut();
        llops._called_exception_is_here_or_cannot_occur = true;
        if llops.llop_raising_exceptions.is_some() {
            return Err(TyperError::message(
                "cannot catch an exception at more than one llop",
            ));
        }
        if self.exceptionlinks.is_empty() {
            return Ok(());
        }
        llops.llop_raising_exceptions = Some(LlopRaisingExceptions::Removed);
        Ok(())
    }

    /// RPython `HighLevelOp.r_s_pop(self, index=-1)` (rtyper.py:693-696).
    ///
    /// ```python
    /// def r_s_pop(self, index=-1):
    ///     "Return and discard the argument with index position."
    ///     self.args_v.pop(index)
    ///     return self.args_r.pop(index), self.args_s.pop(index)
    /// ```
    pub fn r_s_pop(&self, index: Option<usize>) -> (Option<Arc<dyn Repr>>, SomeValue) {
        let mut v = self.args_v.borrow_mut();
        let mut r = self.args_r.borrow_mut();
        let mut s = self.args_s.borrow_mut();
        let i = index.unwrap_or_else(|| v.len().saturating_sub(1));
        v.remove(i);
        (r.remove(i), s.remove(i))
    }

    /// RPython `HighLevelOp.r_s_popfirstarg(self)` (rtyper.py:698-700).
    pub fn r_s_popfirstarg(&self) -> (Option<Arc<dyn Repr>>, SomeValue) {
        self.r_s_pop(Some(0))
    }

    /// RPython `HighLevelOp.swap_fst_snd_args(self)` (rtyper.py:708-711).
    pub fn swap_fst_snd_args(&self) {
        let mut v = self.args_v.borrow_mut();
        let mut s = self.args_s.borrow_mut();
        let mut r = self.args_r.borrow_mut();
        v.swap(0, 1);
        s.swap(0, 1);
        r.swap(0, 1);
    }

    /// RPython `HighLevelOp.v_s_insertfirstarg(self, v_newfirstarg,
    /// s_newfirstarg)` (rtyper.py:702-706).
    pub fn v_s_insertfirstarg(
        &self,
        v_newfirstarg: Hlvalue,
        s_newfirstarg: SomeValue,
    ) -> Result<(), TyperError> {
        let r_newfirstarg = self.rtyper.getrepr(&s_newfirstarg)?;
        self.args_v.borrow_mut().insert(0, v_newfirstarg);
        self.args_s.borrow_mut().insert(0, s_newfirstarg);
        self.args_r.borrow_mut().insert(0, Some(r_newfirstarg));
        Ok(())
    }
}

// ____________________________________________________________
// LowLevelOpList — `rtyper.py:783-871+`.

/// RPython `class LowLevelOpList(list)` (rtyper.py:783-809) — mutable
/// buffer of `SpaceOperation`s built during specialize_block.
///
/// Upstream subclasses `list`; pyre keeps the list in `ops` and
/// exposes Vec-style operations explicitly.
pub struct LowLevelOpList {
    /// RPython `self.rtyper = rtyper` (rtyper.py:794).
    pub rtyper: Rc<RPythonTyper>,
    /// RPython `self.originalblock = originalblock` (rtyper.py:795).
    pub originalblock: Option<BlockRef>,
    /// RPython `LowLevelOpList.llop_raising_exceptions = None` class
    /// attribute (rtyper.py:790), set by
    /// `HighLevelOp.exception_is_here` / `exception_cannot_occur`.
    pub llop_raising_exceptions: Option<LlopRaisingExceptions>,
    /// RPython `LowLevelOpList.implicit_exceptions_checked = None`
    /// class attribute (rtyper.py:791), managed by
    /// `HighLevelOp.has_implicit_exception`.
    pub implicit_exceptions_checked: Option<Vec<String>>,
    /// Tracks whether the hop has run `exception_is_here` /
    /// `exception_cannot_occur` at least once — used by
    /// `rtyper.py:732,748` bookkeeping.
    pub _called_exception_is_here_or_cannot_occur: bool,
    /// The SpaceOperation buffer itself (upstream stores via
    /// `list.__init__`).
    pub ops: Vec<SpaceOperation>,
}

/// Rust carrier for RPython's `ll_function` argument to
/// `LowLevelOpList.gendirectcall`.
///
/// Upstream receives the actual Python helper function, annotates it with
/// the argument annotations, then obtains a callable graph. The Rust port
/// requires that graph to be present before emitting `direct_call`; a
/// signature-only helper surfaces `MissingRTypeOperation` instead of
/// fabricating a function pointer that upstream would not have produced.
#[derive(Clone, Debug)]
pub struct LowLevelFunction {
    pub name: String,
    pub args: Vec<LowLevelType>,
    pub result: LowLevelType,
    pub graph: Option<Rc<PyGraph>>,
}

impl LowLevelFunction {
    pub fn new(name: impl Into<String>, args: Vec<LowLevelType>, result: LowLevelType) -> Self {
        LowLevelFunction {
            name: name.into(),
            args,
            result,
            graph: None,
        }
    }

    pub fn from_pygraph(
        name: impl Into<String>,
        args: Vec<LowLevelType>,
        result: LowLevelType,
        graph: Rc<PyGraph>,
    ) -> Self {
        LowLevelFunction {
            name: name.into(),
            args,
            result,
            graph: Some(graph),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct LowLevelHelperKey {
    name: String,
    args: Vec<LowLevelType>,
    result: LowLevelType,
}

fn lowlevel_helper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    match name {
        "ll_check_chr" => lowlevel_range_check_helper_graph(name, args, result, 255),
        "ll_check_unichr" => lowlevel_range_check_helper_graph(name, args, result, 0x10ffff),
        "ll_int_abs_ovf" => lowlevel_int_min_unary_ovf_helper_graph(name, args, result, "int_abs"),
        "ll_int_neg_ovf" => lowlevel_int_min_unary_ovf_helper_graph(name, args, result, "int_neg"),
        "ll_int_lshift_ovf" => lowlevel_int_lshift_ovf_helper_graph(name, args, result),
        "ll_int_py_div" => lowlevel_int_py_div_helper_graph(name, args, result, false),
        "ll_int_py_div_nonnegargs" => lowlevel_int_py_div_helper_graph(name, args, result, true),
        "ll_int_py_mod" => lowlevel_int_py_mod_helper_graph(name, args, result, false),
        "ll_int_py_mod_nonnegargs" => lowlevel_int_py_mod_helper_graph(name, args, result, true),
        "ll_int_py_div_ovf" => {
            lowlevel_overflow_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_div")
        }
        "ll_int_py_mod_ovf" => {
            lowlevel_overflow_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_mod")
        }
        "ll_int_py_div_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_div")
        }
        "ll_int_py_mod_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_mod")
        }
        "ll_int_py_div_ovf_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_div_ovf")
        }
        "ll_int_py_mod_ovf_zer" => {
            lowlevel_zero_check_wrapper_graph(rtyper, name, args, result, "ll_int_py_mod_ovf")
        }
        "ll_uint_py_div" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::Unsigned,
            "uint_floordiv",
        ),
        "ll_uint_py_mod" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::Unsigned,
            "uint_mod",
        ),
        "ll_uint_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_uint_py_div",
            LowLevelType::Unsigned,
            "uint_eq",
        ),
        "ll_uint_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_uint_py_mod",
            LowLevelType::Unsigned,
            "uint_eq",
        ),
        "ll_ullong_py_div" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLong,
            "ullong_floordiv",
        ),
        "ll_ullong_py_mod" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLong,
            "ullong_mod",
        ),
        "ll_ullong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ullong_py_div",
            LowLevelType::UnsignedLongLong,
            "ullong_eq",
        ),
        "ll_ullong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ullong_py_mod",
            LowLevelType::UnsignedLongLong,
            "ullong_eq",
        ),
        "ll_ulllong_py_div" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLongLong,
            "ulllong_floordiv",
        ),
        "ll_ulllong_py_mod" => lowlevel_simple_binary_helper_graph(
            name,
            args,
            result,
            LowLevelType::UnsignedLongLongLong,
            "ulllong_mod",
        ),
        "ll_ulllong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ulllong_py_div",
            LowLevelType::UnsignedLongLongLong,
            "ulllong_eq",
        ),
        "ll_ulllong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_ulllong_py_mod",
            LowLevelType::UnsignedLongLongLong,
            "ulllong_eq",
        ),
        "ll_llong_py_div" => lowlevel_signed_wide_py_div_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLong,
            "llong",
            63,
        ),
        "ll_llong_py_mod" => lowlevel_signed_wide_py_mod_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLong,
            "llong",
            63,
        ),
        "ll_llong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_llong_py_div",
            LowLevelType::SignedLongLong,
            "llong_eq",
        ),
        "ll_llong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_llong_py_mod",
            LowLevelType::SignedLongLong,
            "llong_eq",
        ),
        "ll_lllong_py_div" => lowlevel_signed_wide_py_div_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLongLong,
            "lllong",
            127,
        ),
        "ll_lllong_py_mod" => lowlevel_signed_wide_py_mod_helper_graph(
            name,
            args,
            result,
            LowLevelType::SignedLongLongLong,
            "lllong",
            127,
        ),
        "ll_lllong_py_div_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_lllong_py_div",
            LowLevelType::SignedLongLongLong,
            "lllong_eq",
        ),
        "ll_lllong_py_mod_zer" => lowlevel_typed_zero_check_wrapper_graph(
            rtyper,
            name,
            args,
            result,
            "ll_lllong_py_mod",
            LowLevelType::SignedLongLongLong,
            "lllong_eq",
        ),
        _ => Ok(synthetic_lowlevel_helper_graph(name, args, result)),
    }
}

fn variable_with_lltype(name: &str, lltype: LowLevelType) -> Variable {
    let mut var = Variable::named(name);
    var.concretetype = Some(lltype.clone());
    var.annotation
        .replace(Some(Rc::new(lltype_to_annotation(lltype))));
    var
}

fn constant_with_lltype(value: ConstValue, lltype: LowLevelType) -> Hlvalue {
    Hlvalue::Constant(Constant::with_concretetype(value, lltype))
}

fn helper_pygraph_from_graph(
    graph: FunctionGraph,
    argnames: Vec<String>,
    func: GraphFunc,
) -> PyGraph {
    PyGraph {
        graph: Rc::new(RefCell::new(graph)),
        func,
        signature: RefCell::new(Signature::new(argnames, None, None)),
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
    }
}

fn exception_args(exc_name: &str) -> Result<Vec<Hlvalue>, TyperError> {
    let exc_cls = HOST_ENV
        .lookup_exception_class(exc_name)
        .ok_or_else(|| TyperError::message(format!("missing host {exc_name} exception class")))?;
    let exc_instance = HOST_ENV
        .lookup_standard_exception_instance(exc_name)
        .ok_or_else(|| {
            TyperError::message(format!("missing host {exc_name} exception instance"))
        })?;
    Ok(vec![
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(exc_cls))),
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(exc_instance))),
    ])
}

fn lowlevel_range_check_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    upper_bound: i64,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed] || result != &LowLevelType::Void {
        return Err(TyperError::message(format!(
            "{name} expects (Signed) -> Void, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if 0 <= n <= <bound>: return
    //   raise ValueError
    let argnames = vec!["arg0".to_string()];
    let n0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let n1 = variable_with_lltype("n", LowLevelType::Signed);
    let ge0 = variable_with_lltype("ge0", LowLevelType::Bool);
    let lemax = variable_with_lltype("lemax", LowLevelType::Bool);
    let return_var = variable_with_lltype("result", LowLevelType::Void);

    let startblock = Block::shared(vec![Hlvalue::Variable(n0.clone())]);
    let check_upper = Block::shared(vec![Hlvalue::Variable(n1.clone())]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    let exc_args = exception_args("ValueError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ge",
        vec![
            Hlvalue::Variable(n0.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(ge0.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(ge0.clone()));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(n0)],
            Some(check_upper.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args.clone(),
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    check_upper
        .borrow_mut()
        .operations
        .push(SpaceOperation::new(
            "int_le",
            vec![
                Hlvalue::Variable(n1),
                constant_with_lltype(ConstValue::Int(upper_bound), LowLevelType::Signed),
            ],
            Hlvalue::Variable(lemax.clone()),
        ));
    check_upper.borrow_mut().exitswitch = Some(Hlvalue::Variable(lemax));
    check_upper.closeblock(vec![
        Link::new(
            vec![constant_with_lltype(ConstValue::None, LowLevelType::Void)],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_min_unary_ovf_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    opname: &str,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if x == INT_MIN: raise OverflowError
    //   return -x       # ll_int_neg_ovf
    //   return abs(x)   # ll_int_abs_ovf
    let argnames = vec!["arg0".to_string()];
    let x0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let x1 = variable_with_lltype("x", LowLevelType::Signed);
    let is_min = variable_with_lltype("is_min", LowLevelType::Bool);
    let op_result = variable_with_lltype("result", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![Hlvalue::Variable(x0.clone())]);
    let compute = Block::shared(vec![Hlvalue::Variable(x1.clone())]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("OverflowError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(x0.clone()),
            constant_with_lltype(ConstValue::Int(i64::MIN), LowLevelType::Signed),
        ],
        Hlvalue::Variable(is_min.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_min));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0)],
            Some(compute.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    compute.borrow_mut().operations.push(SpaceOperation::new(
        opname,
        vec![Hlvalue::Variable(x1)],
        Hlvalue::Variable(op_result.clone()),
    ));
    compute.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(op_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_lshift_ovf_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   result = x << y
    //   if (result >> y) != x: raise OverflowError
    //   return result
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", LowLevelType::Signed);
    let y = variable_with_lltype("arg1", LowLevelType::Signed);
    let shifted = variable_with_lltype("result", LowLevelType::Signed);
    let shifted_back = variable_with_lltype("shifted_back", LowLevelType::Signed);
    let overflowed = variable_with_lltype("overflowed", LowLevelType::Bool);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("OverflowError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_lshift",
        vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(shifted.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_rshift",
        vec![
            Hlvalue::Variable(shifted.clone()),
            Hlvalue::Variable(y.clone()),
        ],
        Hlvalue::Variable(shifted_back.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_ne",
        vec![Hlvalue::Variable(shifted_back), Hlvalue::Variable(x)],
        Hlvalue::Variable(overflowed.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(overflowed));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(shifted)],
            Some(graph.returnblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_py_div_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    nonnegargs: bool,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py ll_int_py_div:
    //   r = llop.int_floordiv(Signed, x, y)
    //   p = r * y
    //   if y < 0: u = p - x
    //   else:     u = x - p
    //   return r + (u >> INT_BITS_1)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", LowLevelType::Signed);
    let y = variable_with_lltype("arg1", LowLevelType::Signed);
    let r = variable_with_lltype("r", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_floordiv",
        vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));

    if nonnegargs {
        let ok = variable_with_lltype("ok", LowLevelType::Bool);
        let debug_result = variable_with_lltype("debug_result", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(r.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(ok.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![
                Hlvalue::Variable(ok),
                constant_with_lltype(
                    ConstValue::Str("int_py_div_nonnegargs(): one arg is negative".to_string()),
                    LowLevelType::Void,
                ),
            ],
            Hlvalue::Variable(debug_result),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        let p = variable_with_lltype("p", LowLevelType::Signed);
        let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
        let p_neg = variable_with_lltype("p", LowLevelType::Signed);
        let x_neg = variable_with_lltype("x", LowLevelType::Signed);
        let r_neg = variable_with_lltype("r", LowLevelType::Signed);
        let x_pos = variable_with_lltype("x", LowLevelType::Signed);
        let p_pos = variable_with_lltype("p", LowLevelType::Signed);
        let r_pos = variable_with_lltype("r", LowLevelType::Signed);
        let u_neg = variable_with_lltype("u", LowLevelType::Signed);
        let u_pos = variable_with_lltype("u", LowLevelType::Signed);
        let r_join = variable_with_lltype("r", LowLevelType::Signed);
        let u_join = variable_with_lltype("u", LowLevelType::Signed);
        let shifted = variable_with_lltype("shifted", LowLevelType::Signed);
        let div_result = variable_with_lltype("result", LowLevelType::Signed);

        let neg_block = Block::shared(vec![
            Hlvalue::Variable(p_neg.clone()),
            Hlvalue::Variable(x_neg.clone()),
            Hlvalue::Variable(r_neg.clone()),
        ]);
        let pos_block = Block::shared(vec![
            Hlvalue::Variable(x_pos.clone()),
            Hlvalue::Variable(p_pos.clone()),
            Hlvalue::Variable(r_pos.clone()),
        ]);
        let join = Block::shared(vec![
            Hlvalue::Variable(r_join.clone()),
            Hlvalue::Variable(u_join.clone()),
        ]);

        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_mul",
            vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
            Hlvalue::Variable(p.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(y),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(y_is_neg.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
        startblock.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(p.clone()),
                    Hlvalue::Variable(x.clone()),
                    Hlvalue::Variable(r.clone()),
                ],
                Some(neg_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
            Link::new(
                vec![
                    Hlvalue::Variable(x),
                    Hlvalue::Variable(p),
                    Hlvalue::Variable(r),
                ],
                Some(pos_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(false),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
        ]);

        neg_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(p_neg), Hlvalue::Variable(x_neg)],
            Hlvalue::Variable(u_neg.clone()),
        ));
        neg_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r_neg), Hlvalue::Variable(u_neg)],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        pos_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_sub",
            vec![Hlvalue::Variable(x_pos), Hlvalue::Variable(p_pos)],
            Hlvalue::Variable(u_pos.clone()),
        ));
        pos_block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r_pos), Hlvalue::Variable(u_pos)],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_rshift",
            vec![
                Hlvalue::Variable(u_join),
                constant_with_lltype(ConstValue::Int(63), LowLevelType::Signed),
            ],
            Hlvalue::Variable(shifted.clone()),
        ));
        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(r_join), Hlvalue::Variable(shifted)],
            Hlvalue::Variable(div_result.clone()),
        ));
        join.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(div_result)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_int_py_mod_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    nonnegargs: bool,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py ll_int_py_mod:
    //   r = llop.int_mod(Signed, x, y)
    //   if y < 0: u = -r
    //   else:     u = r
    //   return r + (y & (u >> INT_BITS_1))
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", LowLevelType::Signed);
    let y = variable_with_lltype("arg1", LowLevelType::Signed);
    let r = variable_with_lltype("r", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_mod",
        vec![Hlvalue::Variable(x), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));

    if nonnegargs {
        let ok = variable_with_lltype("ok", LowLevelType::Bool);
        let debug_result = variable_with_lltype("debug_result", LowLevelType::Void);
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_ge",
            vec![
                Hlvalue::Variable(r.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(ok.clone()),
        ));
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![
                Hlvalue::Variable(ok),
                constant_with_lltype(
                    ConstValue::Str("int_py_mod_nonnegargs(): one arg is negative".to_string()),
                    LowLevelType::Void,
                ),
            ],
            Hlvalue::Variable(debug_result),
        ));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    } else {
        let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
        let r_neg = variable_with_lltype("r", LowLevelType::Signed);
        let y_neg = variable_with_lltype("y", LowLevelType::Signed);
        let r_pos = variable_with_lltype("r", LowLevelType::Signed);
        let y_pos = variable_with_lltype("y", LowLevelType::Signed);
        let u_neg = variable_with_lltype("u", LowLevelType::Signed);
        let r_join = variable_with_lltype("r", LowLevelType::Signed);
        let y_join = variable_with_lltype("y", LowLevelType::Signed);
        let u_join = variable_with_lltype("u", LowLevelType::Signed);
        let shifted = variable_with_lltype("shifted", LowLevelType::Signed);
        let masked = variable_with_lltype("masked", LowLevelType::Signed);
        let mod_result = variable_with_lltype("result", LowLevelType::Signed);

        let neg_block = Block::shared(vec![
            Hlvalue::Variable(r_neg.clone()),
            Hlvalue::Variable(y_neg.clone()),
        ]);
        let pos_block = Block::shared(vec![
            Hlvalue::Variable(r_pos.clone()),
            Hlvalue::Variable(y_pos.clone()),
        ]);
        let join = Block::shared(vec![
            Hlvalue::Variable(r_join.clone()),
            Hlvalue::Variable(y_join.clone()),
            Hlvalue::Variable(u_join.clone()),
        ]);

        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "int_lt",
            vec![
                Hlvalue::Variable(y.clone()),
                constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
            ],
            Hlvalue::Variable(y_is_neg.clone()),
        ));
        startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
        startblock.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
                Some(neg_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(true),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
            Link::new(
                vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
                Some(pos_block.clone()),
                Some(constant_with_lltype(
                    ConstValue::Bool(false),
                    LowLevelType::Bool,
                )),
            )
            .into_ref(),
        ]);

        neg_block.borrow_mut().operations.push(SpaceOperation::new(
            "int_neg",
            vec![Hlvalue::Variable(r_neg.clone())],
            Hlvalue::Variable(u_neg.clone()),
        ));
        neg_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(r_neg),
                    Hlvalue::Variable(y_neg),
                    Hlvalue::Variable(u_neg),
                ],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        pos_block.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Variable(r_pos.clone()),
                    Hlvalue::Variable(y_pos),
                    Hlvalue::Variable(r_pos),
                ],
                Some(join.clone()),
                None,
            )
            .into_ref(),
        ]);

        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_rshift",
            vec![
                Hlvalue::Variable(u_join),
                constant_with_lltype(ConstValue::Int(63), LowLevelType::Signed),
            ],
            Hlvalue::Variable(shifted.clone()),
        ));
        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_and",
            vec![Hlvalue::Variable(y_join), Hlvalue::Variable(shifted)],
            Hlvalue::Variable(masked.clone()),
        ));
        join.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(r_join), Hlvalue::Variable(masked)],
            Hlvalue::Variable(mod_result.clone()),
        ));
        join.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(mod_result)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
    }

    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_zero_check_wrapper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    callee_name: &str,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if y == 0: raise ZeroDivisionError
    //   return ll_int_py_div(x, y) / ll_int_py_mod(x, y)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let y0 = variable_with_lltype("arg1", LowLevelType::Signed);
    let x1 = variable_with_lltype("x", LowLevelType::Signed);
    let y1 = variable_with_lltype("y", LowLevelType::Signed);
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    let call_result = variable_with_lltype("result", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x0.clone()),
        Hlvalue::Variable(y0.clone()),
    ]);
    let callblock = Block::shared(vec![
        Hlvalue::Variable(x1.clone()),
        Hlvalue::Variable(y1.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("ZeroDivisionError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(y0.clone()),
            constant_with_lltype(ConstValue::Int(0), LowLevelType::Signed),
        ],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0), Hlvalue::Variable(y0)],
            Some(callblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let callee = rtyper.lowlevel_helper_function(
        callee_name,
        vec![LowLevelType::Signed, LowLevelType::Signed],
        LowLevelType::Signed,
    )?;
    callblock
        .borrow_mut()
        .operations
        .push(direct_call_operation(
            rtyper,
            &callee,
            vec![Hlvalue::Variable(x1), Hlvalue::Variable(y1)],
            call_result.clone(),
        )?);
    callblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(call_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_signed_wide_py_div_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    lltype: LowLevelType,
    opprefix: &str,
    bits_minus_one: i64,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    // RPython rtyper/rint.py ll_llong_py_div / ll_lllong_py_div:
    //   r = llop.<prefix>_floordiv(TYPE, x, y)
    //   p = r * y
    //   if y < 0: u = p - x
    //   else:     u = x - p
    //   return r + (u >> <BITS_1>)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", lltype.clone());
    let y = variable_with_lltype("arg1", lltype.clone());
    let r = variable_with_lltype("r", lltype.clone());
    let p = variable_with_lltype("p", lltype.clone());
    let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
    let p_neg = variable_with_lltype("p", lltype.clone());
    let x_neg = variable_with_lltype("x", lltype.clone());
    let r_neg = variable_with_lltype("r", lltype.clone());
    let x_pos = variable_with_lltype("x", lltype.clone());
    let p_pos = variable_with_lltype("p", lltype.clone());
    let r_pos = variable_with_lltype("r", lltype.clone());
    let u_neg = variable_with_lltype("u", lltype.clone());
    let u_pos = variable_with_lltype("u", lltype.clone());
    let r_join = variable_with_lltype("r", lltype.clone());
    let u_join = variable_with_lltype("u", lltype.clone());
    let shifted = variable_with_lltype("shifted", lltype.clone());
    let div_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype.clone());

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let neg_block = Block::shared(vec![
        Hlvalue::Variable(p_neg.clone()),
        Hlvalue::Variable(x_neg.clone()),
        Hlvalue::Variable(r_neg.clone()),
    ]);
    let pos_block = Block::shared(vec![
        Hlvalue::Variable(x_pos.clone()),
        Hlvalue::Variable(p_pos.clone()),
        Hlvalue::Variable(r_pos.clone()),
    ]);
    let join = Block::shared(vec![
        Hlvalue::Variable(r_join.clone()),
        Hlvalue::Variable(u_join.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_floordiv"),
        vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_mul"),
        vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(p.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_lt"),
        vec![
            Hlvalue::Variable(y),
            constant_with_lltype(ConstValue::Int(0), lltype.clone()),
        ],
        Hlvalue::Variable(y_is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(p.clone()),
                Hlvalue::Variable(x.clone()),
                Hlvalue::Variable(r.clone()),
            ],
            Some(neg_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![
                Hlvalue::Variable(x),
                Hlvalue::Variable(p),
                Hlvalue::Variable(r),
            ],
            Some(pos_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    neg_block.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_sub"),
        vec![Hlvalue::Variable(p_neg), Hlvalue::Variable(x_neg)],
        Hlvalue::Variable(u_neg.clone()),
    ));
    neg_block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(r_neg), Hlvalue::Variable(u_neg)],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    pos_block.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_sub"),
        vec![Hlvalue::Variable(x_pos), Hlvalue::Variable(p_pos)],
        Hlvalue::Variable(u_pos.clone()),
    ));
    pos_block.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(r_pos), Hlvalue::Variable(u_pos)],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_rshift"),
        vec![
            Hlvalue::Variable(u_join),
            constant_with_lltype(ConstValue::Int(bits_minus_one), LowLevelType::Signed),
        ],
        Hlvalue::Variable(shifted.clone()),
    ));
    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_add"),
        vec![Hlvalue::Variable(r_join), Hlvalue::Variable(shifted)],
        Hlvalue::Variable(div_result.clone()),
    ));
    join.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(div_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_signed_wide_py_mod_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    lltype: LowLevelType,
    opprefix: &str,
    bits_minus_one: i64,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    // RPython rtyper/rint.py ll_llong_py_mod / ll_lllong_py_mod:
    //   r = llop.<prefix>_mod(TYPE, x, y)
    //   if y < 0: u = -r
    //   else:     u = r
    //   return r + (y & (u >> <BITS_1>))
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", lltype.clone());
    let y = variable_with_lltype("arg1", lltype.clone());
    let r = variable_with_lltype("r", lltype.clone());
    let y_is_neg = variable_with_lltype("y_is_neg", LowLevelType::Bool);
    let r_neg = variable_with_lltype("r", lltype.clone());
    let y_neg = variable_with_lltype("y", lltype.clone());
    let r_pos = variable_with_lltype("r", lltype.clone());
    let y_pos = variable_with_lltype("y", lltype.clone());
    let u_neg = variable_with_lltype("u", lltype.clone());
    let r_join = variable_with_lltype("r", lltype.clone());
    let y_join = variable_with_lltype("y", lltype.clone());
    let u_join = variable_with_lltype("u", lltype.clone());
    let shifted = variable_with_lltype("shifted", lltype.clone());
    let masked = variable_with_lltype("masked", lltype.clone());
    let mod_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype.clone());

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let neg_block = Block::shared(vec![
        Hlvalue::Variable(r_neg.clone()),
        Hlvalue::Variable(y_neg.clone()),
    ]);
    let pos_block = Block::shared(vec![
        Hlvalue::Variable(r_pos.clone()),
        Hlvalue::Variable(y_pos.clone()),
    ]);
    let join = Block::shared(vec![
        Hlvalue::Variable(r_join.clone()),
        Hlvalue::Variable(y_join.clone()),
        Hlvalue::Variable(u_join.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_mod"),
        vec![Hlvalue::Variable(x), Hlvalue::Variable(y.clone())],
        Hlvalue::Variable(r.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_lt"),
        vec![
            Hlvalue::Variable(y.clone()),
            constant_with_lltype(ConstValue::Int(0), lltype.clone()),
        ],
        Hlvalue::Variable(y_is_neg.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(y_is_neg));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y.clone())],
            Some(neg_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(r.clone()), Hlvalue::Variable(y)],
            Some(pos_block.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    neg_block.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_neg"),
        vec![Hlvalue::Variable(r_neg.clone())],
        Hlvalue::Variable(u_neg.clone()),
    ));
    neg_block.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(r_neg),
                Hlvalue::Variable(y_neg),
                Hlvalue::Variable(u_neg),
            ],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    pos_block.closeblock(vec![
        Link::new(
            vec![
                Hlvalue::Variable(r_pos.clone()),
                Hlvalue::Variable(y_pos),
                Hlvalue::Variable(r_pos),
            ],
            Some(join.clone()),
            None,
        )
        .into_ref(),
    ]);

    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_rshift"),
        vec![
            Hlvalue::Variable(u_join),
            constant_with_lltype(ConstValue::Int(bits_minus_one), LowLevelType::Signed),
        ],
        Hlvalue::Variable(shifted.clone()),
    ));
    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_and"),
        vec![Hlvalue::Variable(y_join), Hlvalue::Variable(shifted)],
        Hlvalue::Variable(masked.clone()),
    ));
    join.borrow_mut().operations.push(SpaceOperation::new(
        format!("{opprefix}_add"),
        vec![Hlvalue::Variable(r_join), Hlvalue::Variable(masked)],
        Hlvalue::Variable(mod_result.clone()),
    ));
    join.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(mod_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_simple_binary_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    lltype: LowLevelType,
    opname: &str,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x = variable_with_lltype("arg0", lltype.clone());
    let y = variable_with_lltype("arg1", lltype.clone());
    let op_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype);
    let startblock = Block::shared(vec![
        Hlvalue::Variable(x.clone()),
        Hlvalue::Variable(y.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        opname,
        vec![Hlvalue::Variable(x), Hlvalue::Variable(y)],
        Hlvalue::Variable(op_result.clone()),
    ));
    startblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(op_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_typed_zero_check_wrapper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    callee_name: &str,
    lltype: LowLevelType,
    eq_opname: &str,
) -> Result<PyGraph, TyperError> {
    if args != [lltype.clone(), lltype.clone()] || result != &lltype {
        return Err(TyperError::message(format!(
            "{name} expects ({0}, {0}) -> {0}, got ({args:?}) -> {result:?}",
            lltype.short_name()
        )));
    }

    // RPython rtyper/rint.py unsigned helpers:
    //   if y == 0: raise ZeroDivisionError
    //   return ll_uint_py_div(x, y) / ll_uint_py_mod(x, y)
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x0 = variable_with_lltype("arg0", lltype.clone());
    let y0 = variable_with_lltype("arg1", lltype.clone());
    let x1 = variable_with_lltype("x", lltype.clone());
    let y1 = variable_with_lltype("y", lltype.clone());
    let is_zero = variable_with_lltype("is_zero", LowLevelType::Bool);
    let call_result = variable_with_lltype("result", lltype.clone());
    let return_var = variable_with_lltype("result", lltype.clone());

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x0.clone()),
        Hlvalue::Variable(y0.clone()),
    ]);
    let callblock = Block::shared(vec![
        Hlvalue::Variable(x1.clone()),
        Hlvalue::Variable(y1.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("ZeroDivisionError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        eq_opname,
        vec![
            Hlvalue::Variable(y0.clone()),
            constant_with_lltype(ConstValue::Int(0), lltype.clone()),
        ],
        Hlvalue::Variable(is_zero.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(is_zero));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0), Hlvalue::Variable(y0)],
            Some(callblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let callee = rtyper.lowlevel_helper_function(
        callee_name,
        vec![lltype.clone(), lltype.clone()],
        lltype,
    )?;
    callblock
        .borrow_mut()
        .operations
        .push(direct_call_operation(
            rtyper,
            &callee,
            vec![Hlvalue::Variable(x1), Hlvalue::Variable(y1)],
            call_result.clone(),
        )?);
    callblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(call_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn lowlevel_overflow_check_wrapper_graph(
    rtyper: &RPythonTyper,
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
    callee_name: &str,
) -> Result<PyGraph, TyperError> {
    if args != [LowLevelType::Signed, LowLevelType::Signed] || result != &LowLevelType::Signed {
        return Err(TyperError::message(format!(
            "{name} expects (Signed, Signed) -> Signed, got ({args:?}) -> {result:?}"
        )));
    }

    // RPython rtyper/rint.py:
    //   if (x == -sys.maxint - 1) & (y == -1): raise OverflowError
    //   return ll_int_py_div(x, y) / ll_int_py_mod(x, y)
    //
    // Keep the bitwise `&` shape: RPython intentionally does not
    // short-circuit here so the JIT can see a single combined guard.
    let argnames = vec!["arg0".to_string(), "arg1".to_string()];
    let x0 = variable_with_lltype("arg0", LowLevelType::Signed);
    let y0 = variable_with_lltype("arg1", LowLevelType::Signed);
    let x1 = variable_with_lltype("x", LowLevelType::Signed);
    let y1 = variable_with_lltype("y", LowLevelType::Signed);
    let x_is_min = variable_with_lltype("x_is_min", LowLevelType::Bool);
    let y_is_minus_one = variable_with_lltype("y_is_minus_one", LowLevelType::Bool);
    let overflowed = variable_with_lltype("overflowed", LowLevelType::Bool);
    let call_result = variable_with_lltype("result", LowLevelType::Signed);
    let return_var = variable_with_lltype("result", LowLevelType::Signed);

    let startblock = Block::shared(vec![
        Hlvalue::Variable(x0.clone()),
        Hlvalue::Variable(y0.clone()),
    ]);
    let callblock = Block::shared(vec![
        Hlvalue::Variable(x1.clone()),
        Hlvalue::Variable(y1.clone()),
    ]);
    let mut graph = FunctionGraph::with_return_var(
        name.to_string(),
        startblock.clone(),
        Hlvalue::Variable(return_var),
    );
    let exc_args = exception_args("OverflowError")?;

    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(x0.clone()),
            constant_with_lltype(ConstValue::Int(i64::MIN), LowLevelType::Signed),
        ],
        Hlvalue::Variable(x_is_min.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_eq",
        vec![
            Hlvalue::Variable(y0.clone()),
            constant_with_lltype(ConstValue::Int(-1), LowLevelType::Signed),
        ],
        Hlvalue::Variable(y_is_minus_one.clone()),
    ));
    startblock.borrow_mut().operations.push(SpaceOperation::new(
        "int_and",
        vec![
            Hlvalue::Variable(x_is_min),
            Hlvalue::Variable(y_is_minus_one),
        ],
        Hlvalue::Variable(overflowed.clone()),
    ));
    startblock.borrow_mut().exitswitch = Some(Hlvalue::Variable(overflowed));
    startblock.closeblock(vec![
        Link::new(
            exc_args,
            Some(graph.exceptblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
        Link::new(
            vec![Hlvalue::Variable(x0), Hlvalue::Variable(y0)],
            Some(callblock.clone()),
            Some(constant_with_lltype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            )),
        )
        .into_ref(),
    ]);

    let callee = rtyper.lowlevel_helper_function(
        callee_name,
        vec![LowLevelType::Signed, LowLevelType::Signed],
        LowLevelType::Signed,
    )?;
    callblock
        .borrow_mut()
        .operations
        .push(direct_call_operation(
            rtyper,
            &callee,
            vec![Hlvalue::Variable(x1), Hlvalue::Variable(y1)],
            call_result.clone(),
        )?);
    callblock.closeblock(vec![
        Link::new(
            vec![Hlvalue::Variable(call_result)],
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
    Ok(helper_pygraph_from_graph(graph, argnames, func))
}

fn direct_call_operation(
    rtyper: &RPythonTyper,
    ll_function: &LowLevelFunction,
    args_v: Vec<Hlvalue>,
    result: Variable,
) -> Result<SpaceOperation, TyperError> {
    let graph = ll_function.graph.as_ref().ok_or_else(|| {
        TyperError::missing_rtype_operation(format!(
            "low-level helper {} has no annotated helper graph",
            ll_function.name
        ))
    })?;
    let func_ptr = rtyper.getcallable(graph)?;
    let PtrTarget::Func(func_type) = &func_ptr._TYPE.TO else {
        return Err(TyperError::message(format!(
            "direct_call({}) callable is not a function pointer",
            ll_function.name
        )));
    };
    if func_type.args != ll_function.args || func_type.result != ll_function.result {
        return Err(TyperError::message(format!(
            "direct_call({}) graph type mismatch: got {:?} -> {:?}, expected {:?} -> {:?}",
            ll_function.name,
            func_type.args,
            func_type.result,
            ll_function.args,
            ll_function.result
        )));
    }
    let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
    let c_func = inputconst_from_lltype(&func_ptr_type, &ConstValue::LLPtr(Box::new(func_ptr)))
        .expect("functionptr constant must match Ptr(FuncType)");
    let mut call_args = Vec::with_capacity(args_v.len() + 1);
    call_args.push(Hlvalue::Constant(c_func));
    call_args.extend(args_v);
    Ok(SpaceOperation::new(
        "direct_call",
        call_args,
        Hlvalue::Variable(result),
    ))
}

fn synthetic_lowlevel_helper_graph(
    name: &str,
    args: &[LowLevelType],
    result: &LowLevelType,
) -> PyGraph {
    let argnames: Vec<String> = (0..args.len()).map(|i| format!("arg{i}")).collect();
    let inputargs: Vec<Hlvalue> = argnames
        .iter()
        .zip(args.iter())
        .map(|(argname, lltype)| Hlvalue::Variable(variable_with_lltype(argname, lltype.clone())))
        .collect();
    let startblock = Block::shared(inputargs);
    let return_var = variable_with_lltype("result", result.clone());
    let mut graph =
        FunctionGraph::with_return_var(name.to_string(), startblock, Hlvalue::Variable(return_var));
    let func = GraphFunc::new(
        name.to_string(),
        Constant::new(ConstValue::Dict(Default::default())),
    );
    graph.func = Some(func.clone());
    helper_pygraph_from_graph(graph, argnames, func)
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<&LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype.as_ref(),
        Hlvalue::Constant(c) => c.concretetype.as_ref(),
    }
}

impl LowLevelOpList {
    /// RPython `LowLevelOpList.__init__(self, rtyper=None,
    /// originalblock=None)` (rtyper.py:793-795).
    pub fn new(rtyper: Rc<RPythonTyper>, originalblock: Option<BlockRef>) -> Self {
        LowLevelOpList {
            rtyper,
            originalblock,
            llop_raising_exceptions: None,
            implicit_exceptions_checked: None,
            _called_exception_is_here_or_cannot_occur: false,
            ops: Vec::new(),
        }
    }

    /// RPython `LowLevelOpList.hasparentgraph(self)` (rtyper.py:800-801).
    pub fn hasparentgraph(&self) -> bool {
        self.originalblock.is_some()
    }

    /// RPython `LowLevelOpList.record_extra_call(self, graph)`
    /// (`rtyper.py:803-808`).
    pub fn record_extra_call(&self, callee_graph: &GraphRef) -> Result<(), TyperError> {
        if !self.hasparentgraph() {
            return Ok(());
        }
        let annotator =
            self.rtyper.annotator.upgrade().ok_or_else(|| {
                TyperError::message("RPythonTyper.annotator weak reference dropped")
            })?;
        let Some(parent_block) = &self.originalblock else {
            return Ok(());
        };
        if let Some(Some(caller_graph)) = annotator
            .annotated
            .borrow()
            .get(&BlockKey::of(parent_block))
            .cloned()
        {
            annotator.translator.update_call_graph(
                &caller_graph,
                callee_graph,
                (BlockKey::of(parent_block), self.ops.len()),
            );
        }
        Ok(())
    }

    /// Rust bridge for RPython `record_extra_call(vlist[0].value.graph)`
    /// in `rptr.py:100-101`. Function pointers store graph identity in
    /// `lltype._func.graph`; the live graph object is in
    /// `translator.graphs`.
    pub fn record_extra_call_by_graph_id(&self, graph_id: usize) -> Result<(), TyperError> {
        let annotator =
            self.rtyper.annotator.upgrade().ok_or_else(|| {
                TyperError::message("RPythonTyper.annotator weak reference dropped")
            })?;
        let callee_graph = annotator
            .translator
            .graphs
            .borrow()
            .iter()
            .find(|graph| crate::flowspace::model::GraphKey::of(graph).as_usize() == graph_id)
            .cloned()
            .ok_or_else(|| {
                TyperError::message(format!(
                    "record_extra_call could not find graph id {graph_id}"
                ))
            })?;
        self.record_extra_call(&callee_graph)
    }

    /// RPython `LowLevelOpList.append(self, op)` via `list.append` —
    /// push a `SpaceOperation` onto the buffer.
    pub fn append(&mut self, op: SpaceOperation) {
        self.ops.push(op);
    }

    /// RPython `LowLevelOpList.__len__` via list base.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// RPython `LowLevelOpList.__bool__` via list base.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// RPython `LowLevelOpList.convertvar(self, orig_v, r_from, r_to)`
    /// (`rtyper.py:810-823`).
    ///
    /// Upstream's `pairtype(Repr, Repr).convert_from_to` walks the
    /// `pair_mro` until it finds a registered handler (or exhausts and
    /// raises). Pyre bridges through
    /// [`super::pairtype::pair_convert_from_to`] for the cross-class
    /// lookup and preserves the short-circuit "same repr is identity"
    /// path (upstream `rmodel.py:300`).
    pub fn convertvar(
        &mut self,
        orig_v: Hlvalue,
        r_from: &dyn Repr,
        r_to: &dyn Repr,
    ) -> Result<Hlvalue, TyperError> {
        let same_repr = std::ptr::eq(
            r_from as *const dyn Repr as *const (),
            r_to as *const dyn Repr as *const (),
        );
        if same_repr {
            return Ok(orig_v);
        }
        // Cross-class conversions route through the pairtype
        // dispatcher, which walks `pair_mro(r_from, r_to)` and calls
        // the matching `class __extend__(pairtype(R_A, R_B))` helper.
        if let Some(converted) = super::pairtype::pair_convert_from_to(r_from, r_to, &orig_v, self)?
        {
            if hlvalue_concretetype(&converted) != Some(r_to.lowleveltype()) {
                return Err(TyperError::message(format!(
                    "bug in conversion from {} to {}: returned a {:?}",
                    r_from.repr_string(),
                    r_to.repr_string(),
                    hlvalue_concretetype(&converted)
                )));
            }
            return Ok(converted);
        }
        Err(TyperError::message(format!(
            "don't know how to convert from {} to {}",
            r_from.repr_string(),
            r_to.repr_string()
        )))
    }

    /// RPython `LowLevelOpList.genop(self, opname, args_v,
    /// resulttype=None)` (rtyper.py:825-843).
    ///
    /// ```python
    /// def genop(self, opname, args_v, resulttype=None):
    ///     try:
    ///         for v in args_v:
    ///             v.concretetype
    ///     except AttributeError:
    ///         raise AssertionError("wrong level!  you must call hop.inputargs()"
    ///                              " and pass its result to genop(),"
    ///                              " never hop.args_v directly.")
    ///     vresult = Variable()
    ///     self.append(SpaceOperation(opname, args_v, vresult))
    ///     if resulttype is None:
    ///         vresult.concretetype = Void
    ///         return None
    ///     else:
    ///         if isinstance(resulttype, Repr):
    ///             resulttype = resulttype.lowleveltype
    ///         assert isinstance(resulttype, LowLevelType)
    ///         vresult.concretetype = resulttype
    ///         return vresult
    /// ```
    pub fn genop(
        &mut self,
        opname: &str,
        args_v: Vec<Hlvalue>,
        resulttype: GenopResult,
    ) -> Option<Variable> {
        for v in &args_v {
            // upstream asserts each arg already carries a concretetype
            // (rtyper.py:826-832). Pyre stores `Option<LowLevelType>`
            // on Variable/Constant; None means "not rtyped yet".
            match v {
                Hlvalue::Variable(var) => {
                    assert!(
                        var.concretetype.is_some(),
                        "wrong level! you must call hop.inputargs() and pass \
                         its result to genop(), never hop.args_v directly."
                    );
                }
                Hlvalue::Constant(c) => {
                    assert!(
                        c.concretetype.is_some(),
                        "wrong level! Constant used as genop arg has no concretetype."
                    );
                }
            }
        }
        let mut vresult = Variable::new();
        match resulttype {
            GenopResult::Void => {
                vresult.concretetype = Some(LowLevelType::Void);
                self.ops.push(SpaceOperation::new(
                    opname.to_string(),
                    args_v,
                    Hlvalue::Variable(vresult),
                ));
                None
            }
            GenopResult::LLType(lltype) => {
                vresult.concretetype = Some(lltype);
                let vresult_h = Hlvalue::Variable(vresult.clone());
                self.ops
                    .push(SpaceOperation::new(opname.to_string(), args_v, vresult_h));
                Some(vresult)
            }
            GenopResult::Repr(repr) => {
                vresult.concretetype = Some(repr.lowleveltype().clone());
                let vresult_h = Hlvalue::Variable(vresult.clone());
                self.ops
                    .push(SpaceOperation::new(opname.to_string(), args_v, vresult_h));
                Some(vresult)
            }
        }
    }

    /// RPython `LowLevelOpList.gendirectcall(self, ll_function, *args_v)`
    /// (`rtyper.py:845-882`).
    pub fn gendirectcall(
        &mut self,
        ll_function: &LowLevelFunction,
        args_v: Vec<Hlvalue>,
    ) -> Result<Option<Variable>, TyperError> {
        for (i, (arg, expected)) in args_v.iter().zip(ll_function.args.iter()).enumerate() {
            let got = hlvalue_concretetype(arg).ok_or_else(|| {
                TyperError::message(format!(
                    "gendirectcall argument {i} to {} is missing concretetype",
                    ll_function.name
                ))
            })?;
            if got != expected {
                return Err(TyperError::message(format!(
                    "gendirectcall argument {i} to {} has type {:?}, expected {:?}",
                    ll_function.name, got, expected
                )));
            }
        }
        if args_v.len() != ll_function.args.len() {
            return Err(TyperError::message(format!(
                "gendirectcall({}) argument count mismatch: got {}, expected {}",
                ll_function.name,
                args_v.len(),
                ll_function.args.len()
            )));
        }

        let graph = ll_function.graph.as_ref().ok_or_else(|| {
            TyperError::missing_rtype_operation(format!(
                "low-level helper {} has no annotated helper graph",
                ll_function.name
            ))
        })?;
        self.record_extra_call(&graph.graph)?;

        let func_ptr = self.rtyper.getcallable(graph)?;
        let PtrTarget::Func(func_type) = &func_ptr._TYPE.TO else {
            return Err(TyperError::message(format!(
                "gendirectcall({}) callable is not a function pointer",
                ll_function.name
            )));
        };
        if func_type.args != ll_function.args || func_type.result != ll_function.result {
            return Err(TyperError::message(format!(
                "gendirectcall({}) graph type mismatch: got {:?} -> {:?}, expected {:?} -> {:?}",
                ll_function.name,
                func_type.args,
                func_type.result,
                ll_function.args,
                ll_function.result
            )));
        }
        let graph_result = func_type.result.clone();
        let func_ptr_type = LowLevelType::Ptr(Box::new(func_ptr._TYPE.clone()));
        let c_func = inputconst_from_lltype(&func_ptr_type, &ConstValue::LLPtr(Box::new(func_ptr)))
            .expect("functionptr constant must match Ptr(FuncType)");
        let mut call_args = Vec::with_capacity(args_v.len() + 1);
        call_args.push(Hlvalue::Constant(c_func));
        call_args.extend(args_v);
        Ok(self.genop("direct_call", call_args, GenopResult::LLType(graph_result)))
    }
}

/// RPython `resulttype=None | LowLevelType | Repr` overload type in
/// `LowLevelOpList.genop` (rtyper.py:825-843). Rust needs an explicit
/// enum because the Python overload uses isinstance().
pub enum GenopResult {
    /// `resulttype=None` — upstream emits `vresult.concretetype =
    /// Void` and returns `None`.
    Void,
    /// `resulttype=LowLevelType` — upstream asserts
    /// `isinstance(resulttype, LowLevelType)` and uses it directly.
    LLType(LowLevelType),
    /// `resulttype=Repr` — upstream extracts `.lowleveltype`.
    Repr(Arc<dyn Repr>),
}

/// RPython `self.llop_raising_exceptions` sentinel states (rtyper.py:745,753).
///
/// ```python
/// self.llops.llop_raising_exceptions = len(self.llops)  # :745
/// self.llops.llop_raising_exceptions = "removed"         # :753
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LlopRaisingExceptions {
    /// Index into `LowLevelOpList.ops` of the raising llop.
    Index(usize),
    /// rtyper.py:326 sentinel: the exception cannot actually occur,
    /// all exception links should be removed.
    Removed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::annotator::model::{SomeInteger, SomePtr, SomeValue};
    use crate::flowspace::argument::Signature;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::{BlockKey, ConstValue, Constant, GraphFunc};
    use crate::translator::rtyper::rmodel::{PtrRepr, VoidRepr};

    #[test]
    fn new_rtyper_starts_with_empty_already_seen() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        assert!(rtyper.already_seen.borrow().is_empty());
        assert!(rtyper.concrete_calltables.borrow().is_empty());
        assert!(rtyper.primitive_to_repr.borrow().is_empty());
    }

    #[test]
    fn mark_already_seen_records_block_key() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = ann.translator.entry_point_graph.borrow().clone();
        assert!(block.is_none());

        let graph = crate::flowspace::model::FunctionGraph::new(
            "g",
            std::rc::Rc::new(std::cell::RefCell::new(
                crate::flowspace::model::Block::new(vec![]),
            )),
        );
        let startblock = graph.startblock.clone();
        rtyper.mark_already_seen(&startblock);
        assert!(
            rtyper
                .already_seen
                .borrow()
                .contains_key(&BlockKey::of(&startblock))
        );
    }

    #[test]
    fn getcallable_returns_functionptr_for_graph() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let code = HostCode {
            id: HostCode::fresh_identity(),
            co_name: "f".into(),
            co_filename: "<test>".into(),
            co_firstlineno: 1,
            co_nlocals: 1,
            co_argcount: 1,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: vec!["x".into()],
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(vec!["x".into()], None, None),
        };
        let graph = Rc::new(PyGraph::new(
            GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default()))),
            &code,
        ));
        {
            use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
            let graph_borrow = graph.graph.borrow();
            for arg in graph_borrow.startblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.concretetype = Some(LowLevelType::Signed);
                    v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
                }
            }
            for arg in graph_borrow.returnblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.concretetype = Some(LowLevelType::Void);
                    v.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
                }
            }
        }

        let llfn = rtyper.getcallable(&graph).unwrap();
        use crate::translator::rtyper::lltypesystem::lltype::{_ptr_obj, PtrTarget};
        let PtrTarget::Func(func_t) = &llfn._TYPE.TO else {
            panic!("expected Func ptr, got {:?}", llfn._TYPE.TO);
        };
        assert_eq!(func_t.args.len(), 1);
        let _ptr_obj::Func(func_obj) = llfn._obj().unwrap() else {
            panic!("expected _ptr_obj::Func");
        };
        assert_eq!(func_obj.graph.is_some(), true);
    }

    fn make_rtyper_rc() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    fn make_live_rtyper() -> (Rc<RPythonAnnotator>, Rc<RPythonTyper>) {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        (ann, rtyper)
    }

    fn empty_spaceop(opname: &str) -> SpaceOperation {
        SpaceOperation::new(
            opname.to_string(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        )
    }

    #[test]
    fn highlevelop_constructor_stores_fields_without_running_setup() {
        // rtyper.py:619-623: `__init__` only records references; the
        // args_v / args_s / args_r / s_result / r_result slots remain
        // empty until `setup()` runs (which pyre defers pending the
        // Repr-dispatch chain).
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            empty_spaceop("simple_call"),
            Vec::new(),
            llops,
        );
        assert_eq!(hop.spaceop.opname, "simple_call");
        assert_eq!(hop.nb_args(), 0);
        assert!(hop.exceptionlinks.is_empty());
        assert!(hop.args_v.borrow().is_empty());
        assert!(hop.args_s.borrow().is_empty());
        assert!(hop.args_r.borrow().is_empty());
        assert!(hop.s_result.borrow().is_none());
        assert!(hop.r_result.borrow().is_none());
    }

    #[test]
    fn highlevelop_copy_clones_lists_and_shares_llops() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops.clone());
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(None);

        let copied = hop.copy();
        copied.args_v.borrow_mut().clear();
        assert_eq!(hop.args_v.borrow().len(), 1);
        assert!(Rc::ptr_eq(&copied.llops, &llops));
    }

    #[test]
    fn highlevelop_dispatch_routes_to_default_translate_operation() {
        let (_ann, rtyper) = make_live_rtyper();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            empty_spaceop("not_ported_yet"),
            Vec::new(),
            llops,
        );
        let err = hop.dispatch().unwrap_err();
        assert!(err.to_string().contains("unimplemented operation"));
        assert!(err.to_string().contains("not_ported_yet"));
    }

    #[test]
    fn highlevelop_dispatch_routes_binary_op_to_first_repr_like_rtyper_registeroperations() {
        use crate::translator::rtyper::lltypesystem::lltype::{ArrayType, Ptr, PtrTarget};

        let (_ann, rtyper) = make_live_rtyper();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let ptr = Ptr {
            TO: PtrTarget::Array(ArrayType::new(LowLevelType::Signed)),
        };
        let mut v_array = Variable::new();
        v_array.concretetype = Some(LowLevelType::Ptr(Box::new(ptr.clone())));
        let v_index = Constant::with_concretetype(ConstValue::Int(1), LowLevelType::Signed);
        let mut result = Variable::new();
        result
            .annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::new(
                false, false,
            )))));
        let hop = HighLevelOp::new(
            rtyper.clone(),
            SpaceOperation::new(
                "getitem".to_string(),
                vec![Hlvalue::Variable(v_array), Hlvalue::Constant(v_index)],
                Hlvalue::Variable(result),
            ),
            Vec::new(),
            llops.clone(),
        );
        let r_ptr: Arc<dyn Repr> = Arc::new(PtrRepr::new(ptr.clone()));
        let r_signed = rtyper.getprimitiverepr(&LowLevelType::Signed).unwrap();
        hop.args_v.borrow_mut().extend(hop.spaceop.args.clone());
        hop.args_s.borrow_mut().extend([
            SomeValue::Ptr(SomePtr::new(ptr)),
            SomeValue::Integer(SomeInteger::new(false, false)),
        ]);
        hop.args_r
            .borrow_mut()
            .extend([Some(r_ptr), Some(r_signed.clone())]);
        *hop.r_result.borrow_mut() = Some(r_signed);

        let out = hop.dispatch().unwrap().unwrap();
        assert!(matches!(out, Hlvalue::Variable(_)));
        let ops = llops.borrow();
        assert_eq!(ops.ops.len(), 1);
        assert_eq!(ops.ops[0].opname, "getarrayitem");
    }

    #[test]
    fn highlevelop_inputconst_accepts_repr_and_lowleveltype() {
        let repr = VoidRepr::new();
        let c = HighLevelOp::inputconst(&repr, &ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Void));

        let c = HighLevelOp::inputconst(&LowLevelType::Signed, &ConstValue::Int(7)).unwrap();
        assert_eq!(c.concretetype, Some(LowLevelType::Signed));
    }

    #[test]
    fn highlevelop_inputarg_constant_uses_requested_repr() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Constant(Constant::new(ConstValue::Int(7))));

        let repr = VoidRepr::new();
        let out = hop.inputarg(&repr, 0).unwrap();
        let Hlvalue::Constant(c) = out else {
            panic!("inputarg on Constant should return a Constant");
        };
        assert_eq!(c.value, ConstValue::Int(7));
        assert_eq!(c.concretetype, Some(LowLevelType::Void));
    }

    #[test]
    fn highlevelop_inputarg_variable_uses_convertvar() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        let repr: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let mut var = Variable::new();
        var.concretetype = Some(LowLevelType::Void);
        let arg = Hlvalue::Variable(var.clone());
        hop.args_v.borrow_mut().push(arg.clone());
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(Some(repr.clone()));

        let out = hop.inputarg(&repr, 0).unwrap();
        assert_eq!(out, arg);
    }

    #[test]
    fn highlevelop_inputargs_checks_arity() {
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        let repr = VoidRepr::new();
        let err = hop.inputargs(vec![ConvertedTo::from(&repr)]).unwrap_err();
        assert!(
            err.to_string()
                .contains("operation argument count mismatch")
        );
    }

    #[test]
    fn highlevelop_r_s_pop_removes_trailing_element() {
        // rtyper.py:693-696: pop from args_v + args_r + args_s in lockstep.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_v
            .borrow_mut()
            .push(Hlvalue::Variable(Variable::new()));
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::default()));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_r.borrow_mut().push(None);
        hop.args_r.borrow_mut().push(None);
        let (_r, s) = hop.r_s_pop(None);
        assert!(matches!(s, SomeValue::Impossible));
        assert_eq!(hop.args_v.borrow().len(), 1);
    }

    #[test]
    fn highlevelop_swap_fst_snd_args_swaps_all_three_parallel_vecs() {
        // rtyper.py:708-711.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), Vec::new(), llops);
        let a = Variable::new();
        let b = Variable::new();
        hop.args_v.borrow_mut().push(Hlvalue::Variable(a.clone()));
        hop.args_v.borrow_mut().push(Hlvalue::Variable(b.clone()));
        hop.args_s.borrow_mut().push(SomeValue::Impossible);
        hop.args_s
            .borrow_mut()
            .push(SomeValue::Integer(SomeInteger::default()));
        hop.args_r.borrow_mut().push(None);
        hop.args_r.borrow_mut().push(None);
        hop.swap_fst_snd_args();
        let v = hop.args_v.borrow();
        match (&v[0], &v[1]) {
            (Hlvalue::Variable(first), Hlvalue::Variable(second)) => {
                // Compare by Variable equality (upstream uses `is`);
                // pyre's Variable `PartialEq` matches on identity
                // through the `id` UID allocator.
                assert_eq!(first, &b);
                assert_eq!(second, &a);
            }
            _ => panic!("expected Variable entries"),
        }
        assert!(matches!(hop.args_s.borrow()[0], SomeValue::Integer(_)));
    }

    #[test]
    fn lowlevelop_append_and_len_track_ops_buffer() {
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        assert!(llops.is_empty());
        llops.append(SpaceOperation::new(
            "direct_call".to_string(),
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        ));
        assert_eq!(llops.len(), 1);
        assert_eq!(llops.ops[0].opname, "direct_call");
    }

    #[test]
    fn lowlevelop_convertvar_reuses_value_for_same_repr() {
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let repr: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let value = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ));
        let out = llops
            .convertvar(value.clone(), repr.as_ref(), repr.as_ref())
            .unwrap();
        assert_eq!(out, value);
    }

    #[test]
    fn lowlevelop_convertvar_to_void_returns_void_constant() {
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let from: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let to: Arc<dyn Repr> = Arc::new(VoidRepr::new());
        let value = Hlvalue::Constant(Constant::with_concretetype(
            ConstValue::None,
            LowLevelType::Void,
        ));
        let out = llops.convertvar(value, from.as_ref(), to.as_ref()).unwrap();
        assert_eq!(
            out,
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void
            ))
        );
    }

    #[test]
    fn lowlevelop_genop_void_emits_void_concretetype() {
        // rtyper.py:835-837: resulttype=None → Void + return None.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let result = llops.genop("nop", Vec::new(), GenopResult::Void);
        assert!(result.is_none());
        assert_eq!(llops.ops.len(), 1);
        if let Hlvalue::Variable(v) = &llops.ops[0].result {
            assert_eq!(v.concretetype.as_ref(), Some(&LowLevelType::Void));
        } else {
            panic!("expected Variable result");
        }
    }

    #[test]
    fn lowlevelop_genop_with_lltype_result_sets_concretetype() {
        // rtyper.py:839-843: resulttype=LowLevelType → Variable
        // carries that type.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        // Build a typed input arg so the concretetype assertion passes.
        let mut input = Variable::new();
        input.concretetype = Some(LowLevelType::Signed);
        let result = llops
            .genop(
                "int_add",
                vec![Hlvalue::Variable(input)],
                GenopResult::LLType(LowLevelType::Signed),
            )
            .expect("Signed resulttype must produce a Variable");
        assert_eq!(result.concretetype.as_ref(), Some(&LowLevelType::Signed));
    }

    #[test]
    fn lowlevelop_genop_with_repr_extracts_lowleveltype() {
        // rtyper.py:839-843: resulttype=Repr → upstream does
        // `resulttype = resulttype.lowleveltype`.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let result = llops
            .genop(
                "nop",
                Vec::new(),
                GenopResult::Repr(Arc::new(VoidRepr::new())),
            )
            .expect("VoidRepr produces a Variable with Void concretetype");
        assert_eq!(result.concretetype.as_ref(), Some(&LowLevelType::Void));
    }

    #[test]
    #[should_panic(expected = "wrong level!")]
    fn lowlevelop_genop_rejects_args_without_concretetype() {
        // rtyper.py:826-832: `hop.args_v directly` mistake must
        // assertion-fail.
        let rtyper = make_rtyper_rc();
        let mut llops = LowLevelOpList::new(rtyper, None);
        let untyped = Variable::new();
        assert!(untyped.concretetype.is_none());
        let _ = llops.genop(
            "int_add",
            vec![Hlvalue::Variable(untyped)],
            GenopResult::LLType(LowLevelType::Signed),
        );
    }

    #[test]
    fn exception_cannot_occur_sets_removed_sentinel() {
        // rtyper.py:747-753.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        // Build a dummy exceptionlink so the path executes (upstream
        // early-returns when exceptionlinks is empty).
        let exitblock = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let link = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock),
            None,
        )));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), vec![link], llops.clone());
        hop.exception_cannot_occur().unwrap();
        assert_eq!(
            llops.borrow().llop_raising_exceptions,
            Some(LlopRaisingExceptions::Removed)
        );
    }

    #[test]
    fn has_implicit_exception_records_matching_exception_link() {
        // rtyper.py:713-729: issubclass(exc_cls, link.exitcase) records
        // every matching exception link and returns True.
        let rtyper = make_rtyper_rc();
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let exitblock = Rc::new(RefCell::new(crate::flowspace::model::Block::new(vec![])));
        let catch_exception = HOST_ENV
            .lookup_exception_class("Exception")
            .expect("Exception class");
        let link = Rc::new(RefCell::new(crate::flowspace::model::Link::new(
            vec![],
            Some(exitblock),
            Some(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                catch_exception,
            )))),
        )));
        let hop = HighLevelOp::new(rtyper, empty_spaceop("nop"), vec![link], llops.clone());

        assert!(hop.has_implicit_exception("ValueError"));
        assert_eq!(
            llops.borrow().implicit_exceptions_checked,
            Some(vec!["Exception".to_string()])
        );
    }

    #[test]
    fn lowlevel_helper_function_builds_rint_check_range_graphs() {
        // rint.py:629-638: ll_check_chr / ll_check_unichr are real
        // helper bodies, not signature-only synthetic graphs.
        let (_ann, rtyper) = make_live_rtyper();
        for (name, upper_bound) in [("ll_check_chr", 255), ("ll_check_unichr", 0x10ffff)] {
            let llfn = rtyper
                .lowlevel_helper_function(name, vec![LowLevelType::Signed], LowLevelType::Void)
                .unwrap();
            let pygraph = llfn.graph.expect("helper graph");
            let graph = pygraph.graph.borrow();
            let startblock = graph.startblock.clone();
            let returnblock = graph.returnblock.clone();
            let exceptblock = graph.exceptblock.clone();
            drop(graph);

            let start = startblock.borrow();
            assert_eq!(start.operations.len(), 1);
            assert_eq!(start.operations[0].opname, "int_ge");
            assert_eq!(start.exits.len(), 2);
            let true_link = start
                .exits
                .iter()
                .find(|link| bool_exitcase(link) == Some(true))
                .expect("lower-bound true link");
            let check_upper = true_link
                .borrow()
                .target
                .as_ref()
                .expect("upper-bound block")
                .clone();
            drop(start);

            let upper = check_upper.borrow();
            assert_eq!(upper.operations.len(), 1);
            assert_eq!(upper.operations[0].opname, "int_le");
            assert!(matches!(
                &upper.operations[0].args[1],
                Hlvalue::Constant(Constant {
                    value: ConstValue::Int(n),
                    concretetype: Some(LowLevelType::Signed),
                    ..
                }) if *n == upper_bound
            ));
            assert!(upper.exits.iter().any(|link| {
                bool_exitcase(link) == Some(true)
                    && link
                        .borrow()
                        .target
                        .as_ref()
                        .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&returnblock))
            }));
            assert!(upper.exits.iter().any(|link| {
                bool_exitcase(link) == Some(false)
                    && link
                        .borrow()
                        .target
                        .as_ref()
                        .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&exceptblock))
            }));
        }
    }

    #[test]
    fn lowlevel_helper_function_builds_int_py_div_ovf_wrapper_graph() {
        // rint.py:422-427: the overflow check is intentionally
        // non-short-circuiting `(x == INT_MIN) & (y == -1)`.
        let (_ann, rtyper) = make_live_rtyper();
        let llfn = rtyper
            .lowlevel_helper_function(
                "ll_int_py_div_ovf",
                vec![LowLevelType::Signed, LowLevelType::Signed],
                LowLevelType::Signed,
            )
            .unwrap();
        let pygraph = llfn.graph.expect("helper graph");
        let graph = pygraph.graph.borrow();
        let startblock = graph.startblock.clone();
        let exceptblock = graph.exceptblock.clone();
        drop(graph);

        let start = startblock.borrow();
        let opnames: Vec<_> = start
            .operations
            .iter()
            .map(|op| op.opname.as_str())
            .collect();
        assert_eq!(opnames, vec!["int_eq", "int_eq", "int_and"]);
        assert!(start.exits.iter().any(|link| {
            bool_exitcase(link) == Some(true)
                && link
                    .borrow()
                    .target
                    .as_ref()
                    .is_some_and(|target| BlockKey::of(target) == BlockKey::of(&exceptblock))
        }));
        let callblock = start
            .exits
            .iter()
            .find(|link| bool_exitcase(link) == Some(false))
            .and_then(|link| link.borrow().target.clone())
            .expect("normal path callblock");
        drop(start);

        let call = callblock.borrow();
        assert_eq!(call.operations.len(), 1);
        assert_eq!(call.operations[0].opname, "direct_call");
    }

    fn bool_exitcase(link: &LinkRef) -> Option<bool> {
        match &link.borrow().exitcase {
            Some(Hlvalue::Constant(Constant {
                value: ConstValue::Bool(value),
                ..
            })) => Some(*value),
            _ => None,
        }
    }

    #[test]
    fn rtyper_annotation_returns_none_without_binding() {
        // rtyper.py:166-168 `annotation(var)` returns the annotator's
        // bound value or None for unset variables — matches upstream.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let v = Hlvalue::Variable(Variable::new());
        assert!(rtyper.annotation(&v).is_none());
    }

    #[test]
    fn rtyper_binding_passes_through_annotator_value() {
        // rtyper.py:170-172 `binding(var)` raises KeyError when unset;
        // Rust panics via the underlying annotator. Exercise the
        // positive path here by pre-seeding an annotation.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut var = Variable::new();
        var.annotation
            .replace(Some(Rc::new(SomeValue::Integer(SomeInteger::default()))));
        let v = Hlvalue::Variable(var);
        let binding = rtyper.binding(&v);
        assert!(matches!(binding, SomeValue::Integer(_)));
    }

    #[test]
    #[should_panic(expected = "KeyError: no binding")]
    fn rtyper_binding_panics_on_missing_variable() {
        // rtyper.py:170-172 — upstream raises KeyError when the
        // annotator has no binding. Rust matches via panic.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let v = Hlvalue::Variable(Variable::new());
        let _ = rtyper.binding(&v);
    }

    #[test]
    fn getrepr_caches_impossible_as_shared_singleton() {
        // rtyper.py:149-164: first call materialises, second returns
        // cached entry. For SomeImpossibleValue both arms target the
        // same `impossible_repr` singleton, so pointer-equality via
        // Arc::ptr_eq must hold.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let r1 = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        let r2 = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        assert!(Arc::ptr_eq(&r1, &r2));
        // Cache now carries exactly one entry keyed by Impossible.
        assert_eq!(rtyper.reprs.borrow().len(), 1);
    }

    #[test]
    fn getrepr_surfaces_missing_rtype_operation_for_unported_variants() {
        // rtyper.py:157 — `s_obj.rtyper_makerepr(self)` raises when
        // the variant has no concrete Repr yet. Pyre surfaces
        // `MissingRTypeOperation` so cascading callers can anchor on
        // the upstream module name in the error message.
        //
        // `SomeString` is used here as a still-unported variant; when
        // `rstr.py` lands, rotate to another unported variant
        // (robject.py / rproperty.py / rweakref.py) so this test keeps
        // guarding the error path.
        use crate::annotator::model::SomeString;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let err = rtyper
            .getrepr(&SomeValue::String(SomeString::new(false, false)))
            .unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("rstr.py"));
    }

    #[test]
    fn bindingrepr_passes_through_getrepr() {
        // rtyper.py:174-175.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut var = Variable::new();
        var.annotation.replace(Some(Rc::new(SomeValue::Impossible)));
        let v = Hlvalue::Variable(var);
        let repr = rtyper.bindingrepr(&v).unwrap();
        // impossible_repr lowleveltype is Void.
        assert_eq!(repr.lowleveltype(), &LowLevelType::Void);
    }

    #[test]
    fn getrepr_adds_pendingsetup_once_per_key() {
        // rtyper.py:105-111 + 162: `add_pendingsetup` dedupes by
        // Repr instance; the second getrepr hit must not re-enqueue
        // the cached Arc.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let _ = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        let after_first = rtyper.reprs_must_call_setup.borrow().len();
        let _ = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        // Second call hits cache (rtyper.py:154-155) and skips
        // add_pendingsetup entirely, so pending-setup depth is
        // unchanged.
        assert_eq!(rtyper.reprs_must_call_setup.borrow().len(), after_first);
    }

    #[test]
    fn call_all_setups_drains_pending_reprs() {
        // rtyper.py:243-256.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let _ = rtyper.getrepr(&SomeValue::Impossible).unwrap();
        assert_eq!(rtyper.reprs_must_call_setup.borrow().len(), 1);
        rtyper.call_all_setups().unwrap();
        assert_eq!(rtyper.reprs_must_call_setup.borrow().len(), 0);
    }

    #[test]
    fn highlevelop_setup_fills_args_and_result_reprs() {
        // rtyper.py:625-633.
        // Build a spaceop with one Impossible-typed arg + Impossible
        // result; bind annotations so `rtyper.binding` succeeds.
        let ann_rc = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann_rc));
        let mut arg_var = Variable::new();
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let mut result_var = Variable::new();
        result_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![Hlvalue::Variable(arg_var)],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, spaceop, Vec::new(), llops);
        hop.setup().unwrap();
        assert_eq!(hop.args_v.borrow().len(), 1);
        assert_eq!(hop.args_s.borrow().len(), 1);
        assert_eq!(hop.args_r.borrow().len(), 1);
        assert!(matches!(
            hop.s_result.borrow().as_ref().unwrap(),
            SomeValue::Impossible
        ));
        // Both args_r and r_result resolve to impossible_repr →
        // lowleveltype Void.
        let r_result = hop.r_result.borrow().as_ref().cloned().unwrap();
        assert_eq!(r_result.lowleveltype(), &LowLevelType::Void);
    }

    #[test]
    fn highlevelop_setup_surfaces_missing_rtype_operation_for_unported_arg() {
        // rtyper.py:625-633 → rtyper.py:157 `s_obj.rtyper_makerepr`
        // surfaces MissingRTypeOperation when the argument's
        // SomeValue variant has no concrete Repr yet. Rather than
        // silently fail, the setup path returns the structured
        // TyperError so callers know which upstream module to land.
        //
        // `SomeString` is the still-unported witness variant; when
        // `rstr.py` lands, rotate to another unported variant.
        use crate::annotator::model::SomeString;
        let ann_rc = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann_rc));
        let mut arg_var = Variable::new();
        arg_var
            .annotation
            .replace(Some(Rc::new(SomeValue::String(SomeString::new(
                false, false,
            )))));
        let mut result_var = Variable::new();
        result_var
            .annotation
            .replace(Some(Rc::new(SomeValue::Impossible)));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![Hlvalue::Variable(arg_var)],
            Hlvalue::Variable(result_var),
        );
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(rtyper.clone(), None)));
        let hop = HighLevelOp::new(rtyper, spaceop, Vec::new(), llops);
        let err = hop.setup().unwrap_err();
        assert!(err.is_missing_rtype_operation());
    }
}
