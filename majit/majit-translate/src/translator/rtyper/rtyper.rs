//! RPython `rpython/rtyper/rtyper.py` — `RPythonTyper`, `HighLevelOp`,
//! `LowLevelOpList`.
//!
//! Mirrors upstream module layout: all three types live in the same
//! file (`rtyper.py:47 RPythonTyper`, `rtyper.py:617 HighLevelOp`,
//! `rtyper.py:783 LowLevelOpList`) because the `specialize_block →
//! highlevelops → translate_hl_to_ll` dispatch loop ties them
//! together.
//!
//! Current scope ports struct skeletons + the non-Repr-dispatch
//! methods (`nb_args`, `copy`, `has_implicit_exception`,
//! `exception_is_here`, `exception_cannot_occur`, `r_s_pop`,
//! `v_s_insertfirstarg`, `swap_fst_snd_args` on HighLevelOp; `append`,
//! `genop` on LowLevelOpList) so follow-up commits can layer in
//! `setup()` / `dispatch()` / `inputarg()` / `convertvar` /
//! `gendirectcall` without retrofitting infrastructure.
//!
//! Deferred methods are documented inline with their upstream line
//! reference.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::Arc;

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::model::SomeValue;
use crate::flowspace::model::{
    BlockKey, BlockRef, ConstValue, Constant, Hlvalue, LinkRef, SpaceOperation, Variable,
};
use crate::flowspace::pygraph::PyGraph;
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{_ptr, LowLevelType, getfunctionptr};
use crate::translator::rtyper::rmodel::{
    Repr, ReprKey, inputconst_from_lltype, rtyper_makekey, rtyper_makerepr,
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
    /// Rust uses `Weak` to avoid an Rc cycle with
    /// `RPythonAnnotator.translator -> TranslationContext.rtyper`.
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
        }
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
}

// ____________________________________________________________
// HighLevelOp — `rtyper.py:617-779`.

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
///   the rtyper. This method is DEFERRED pending the
///   [`Repr`]-dispatch chain (specifically
///   `SomeValue.rtyper_makerepr`, which requires per-variant Repr
///   concrete types to be ported).
pub struct HighLevelOp {
    /// RPython `self.rtyper = rtyper` (rtyper.py:620).
    pub rtyper: Weak<RPythonTyper>,
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
        rtyper: Weak<RPythonTyper>,
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
        let rtyper = self
            .rtyper
            .upgrade()
            .ok_or_else(|| TyperError::message("HighLevelOp.rtyper weak reference dropped"))?;
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

    /// RPython `HighLevelOp.has_implicit_exception(self, exc_cls)`
    /// (rtyper.py:713-729).
    ///
    /// DEFERRED: requires pyre's exception class hierarchy (ExceptionData)
    /// to resolve `exitcase.__subclasscheck__`. Scaffolded as a
    /// parity-marker for now; returns `false` until the exception
    /// subsystem is ported.
    pub fn has_implicit_exception(&self, _exc_cls_name: &str) -> bool {
        // TODO (cascading port): implement with
        // `rpython/rtyper/exceptiondata.py ExceptionData`.
        false
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
    ///
    /// `r_newfirstarg` derives from `rtyper.getrepr(s_newfirstarg)` —
    /// DEFERRED; callers must pass the already-computed repr until
    /// `getrepr` lands.
    pub fn v_s_insertfirstarg(
        &self,
        v_newfirstarg: Hlvalue,
        s_newfirstarg: SomeValue,
        r_newfirstarg: Option<Arc<dyn Repr>>,
    ) {
        self.args_v.borrow_mut().insert(0, v_newfirstarg);
        self.args_s.borrow_mut().insert(0, s_newfirstarg);
        self.args_r.borrow_mut().insert(0, r_newfirstarg);
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
    pub rtyper: Weak<RPythonTyper>,
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

impl LowLevelOpList {
    /// RPython `LowLevelOpList.__init__(self, rtyper=None,
    /// originalblock=None)` (rtyper.py:793-795).
    pub fn new(rtyper: Weak<RPythonTyper>, originalblock: Option<BlockRef>) -> Self {
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
    use crate::annotator::model::{SomeInteger, SomeValue};
    use crate::flowspace::argument::Signature;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::{BlockKey, ConstValue, Constant, GraphFunc};
    use crate::translator::rtyper::rmodel::VoidRepr;

    #[test]
    fn new_rtyper_starts_with_empty_already_seen() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        assert!(rtyper.already_seen.borrow().is_empty());
        assert!(rtyper.concrete_calltables.borrow().is_empty());
    }

    #[test]
    fn mark_already_seen_records_block_key() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let block = ann.translator.borrow().entry_point_graph.borrow().clone();
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
                    v.annotation = Some(Rc::new(SomeValue::Impossible));
                }
            }
            for arg in graph_borrow.returnblock.borrow_mut().inputargs.iter_mut() {
                if let crate::flowspace::model::Hlvalue::Variable(v) = arg {
                    v.concretetype = Some(LowLevelType::Void);
                    v.annotation = Some(Rc::new(SomeValue::Impossible));
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

    fn make_rtyper_weak() -> Weak<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        Rc::downgrade(&rtyper)
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
        let rtyper = make_rtyper_weak();
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
    fn highlevelop_r_s_pop_removes_trailing_element() {
        // rtyper.py:693-696: pop from args_v + args_r + args_s in lockstep.
        let rtyper = make_rtyper_weak();
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
        let rtyper = make_rtyper_weak();
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
        let rtyper = make_rtyper_weak();
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
    fn lowlevelop_genop_void_emits_void_concretetype() {
        // rtyper.py:835-837: resulttype=None → Void + return None.
        let rtyper = make_rtyper_weak();
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
        let rtyper = make_rtyper_weak();
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
        let rtyper = make_rtyper_weak();
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
        let rtyper = make_rtyper_weak();
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
        let rtyper = make_rtyper_weak();
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
        var.annotation = Some(Rc::new(SomeValue::Integer(SomeInteger::default())));
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
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let err = rtyper
            .getrepr(&SomeValue::Integer(SomeInteger::default()))
            .unwrap_err();
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("rint.py"));
    }

    #[test]
    fn bindingrepr_passes_through_getrepr() {
        // rtyper.py:174-175.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = RPythonTyper::new(&ann);
        let mut var = Variable::new();
        var.annotation = Some(Rc::new(SomeValue::Impossible));
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
        arg_var.annotation = Some(Rc::new(SomeValue::Impossible));
        let mut result_var = Variable::new();
        result_var.annotation = Some(Rc::new(SomeValue::Impossible));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![Hlvalue::Variable(arg_var)],
            Hlvalue::Variable(result_var),
        );
        let weak = Rc::downgrade(&rtyper);
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(weak.clone(), None)));
        let hop = HighLevelOp::new(weak, spaceop, Vec::new(), llops);
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
        let ann_rc = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann_rc));
        let mut arg_var = Variable::new();
        arg_var.annotation = Some(Rc::new(SomeValue::Integer(SomeInteger::default())));
        let mut result_var = Variable::new();
        result_var.annotation = Some(Rc::new(SomeValue::Impossible));
        let spaceop = SpaceOperation::new(
            "simple_call".to_string(),
            vec![Hlvalue::Variable(arg_var)],
            Hlvalue::Variable(result_var),
        );
        let weak = Rc::downgrade(&rtyper);
        let llops = Rc::new(RefCell::new(LowLevelOpList::new(weak.clone(), None)));
        let hop = HighLevelOp::new(weak, spaceop, Vec::new(), llops);
        let err = hop.setup().unwrap_err();
        assert!(err.is_missing_rtype_operation());
    }
}
