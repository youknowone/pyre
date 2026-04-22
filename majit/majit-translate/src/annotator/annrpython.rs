//! RPython `rpython/annotator/annrpython.py` — `RPythonAnnotator` driver.
//!
//! This file starts as a skeleton holding only the public surface that
//! the `binaryop` / `unaryop` dispatchers immediately consume — the
//! rest of the driver (`build_types`, `complete`, `processblock`,
//! `consider_op`, `flowin`, …) lands with the annrpython porting
//! commits further down the plan (Commit 7 Part A / Commit 8 Part B).
//!
//! Fields and method signatures mirror upstream line-by-line; method
//! bodies that still require un-ported machinery (pendingblocks queue,
//! TranslationContext, policy specialize, …) carry the upstream
//! comment verbatim and a `todo!()` stub so every stub surfaces at
//! runtime rather than silently no-op'ing.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::super::flowspace::model::{
    BlockKey, BlockRef, GraphKey, GraphRef, Hlvalue, LinkKey, LinkRef, Variable, checkgraph,
};
use super::bookkeeper::Bookkeeper;
use super::model::{SomeValue, TLS, UnionError, unionof};
use super::policy::AnnotatorPolicy;
use crate::translator::translator::TranslationContext;

/// RPython `class RPythonAnnotator(object)` (annrpython.py:22).
///
/// "Block annotator for RPython."
pub struct RPythonAnnotator {
    /// RPython `self.translator` (annrpython.py:30-35):
    ///
    /// ```python
    /// if translator is None:
    ///     from rpython.translator.translator import TranslationContext
    ///     translator = TranslationContext()
    ///     translator.annotator = self
    /// self.translator = translator
    /// ```
    ///
    /// Always populated; `new()` defaults it when `None` is supplied.
    /// Upstream also installs a back-reference (`translator.annotator =
    /// self`); Rust requires a `Weak<RPythonAnnotator>` round-trip to
    /// avoid a refcount cycle — that wiring lands with the
    /// bookkeeper/annotator backlink port (reviewer's "pre-existing"
    /// item).
    pub translator: RefCell<TranslationContext>,
    /// RPython `self.genpendingblocks = [{}]` (annrpython.py:36).
    /// Per-generation list of `{block: graph-containing-it}`. The
    /// inner map pairs identity-keyed blocks with the owning graph
    /// reference so `processblock(graph, block)` can recover the graph
    /// later in `complete_pending_blocks()`.
    pub genpendingblocks: RefCell<Vec<HashMap<BlockKey, (BlockRef, GraphRef)>>>,
    /// RPython `self.annotated = {}` (annrpython.py:37).
    ///
    /// Upstream stores `dict[Block, bool | FunctionGraph]`:
    ///   * `block not in self.annotated`: never seen.
    ///   * `self.annotated[block] == False`: inputs bound, awaiting
    ///     `flowin()`.
    ///   * `self.annotated[block] == graph`: done.
    /// Rust port models the two non-absent states as
    /// `Option<GraphRef>`: `None = False`, `Some = graph`.
    pub annotated: RefCell<HashMap<BlockKey, Option<GraphRef>>>,
    /// Rust-side identity→BlockRef index. Upstream's `self.annotated`
    /// dict keys ARE the Python `Block` objects, so iterating the dict
    /// yields block references directly. The Rust `BlockKey` identity
    /// indirection loses that, so we maintain this parallel map —
    /// populated in lockstep with every `annotated.insert` and
    /// `blocked_blocks.insert` — to let `call_sites()` walk "all seen
    /// blocks" without hunting through auxiliary tables.
    pub all_blocks: RefCell<HashMap<BlockKey, BlockRef>>,
    /// RPython `self.added_blocks = None` (annrpython.py:38).
    ///
    /// Upstream sentinel: `None = track nothing`, `{} = start
    /// tracking`, filled by `processblock`. Rust port stores
    /// identity-keyed blocks; the value side is `()` (we only use it
    /// as a set).
    pub added_blocks: RefCell<Option<HashMap<BlockKey, BlockRef>>>,
    /// RPython `self.links_followed = {}` (annrpython.py:39).
    pub links_followed: RefCell<HashSet<LinkKey>>,
    /// RPython `self.notify = {}` (annrpython.py:40).
    pub notify: RefCell<HashMap<BlockKey, HashSet<PositionKey>>>,
    /// RPython `self.fixed_graphs = {}` (annrpython.py:41). Graphs
    /// that have already been rtyped — `addpendingblock` rejects new
    /// pending entries against these.
    pub fixed_graphs: RefCell<HashMap<GraphKey, GraphRef>>,
    /// RPython `self.blocked_blocks = {}` (annrpython.py:42).
    /// `{blocked_block: (graph, opindex)}`. `opindex == None` upstream
    /// → `None` here (block is blocked at entry, not mid-flow).
    pub blocked_blocks: RefCell<HashMap<BlockKey, (BlockRef, GraphRef, Option<usize>)>>,
    /// RPython `self.blocked_graphs = {}` (annrpython.py:44). Records
    /// graphs that have at least one blocked block; the `bool` value
    /// tracks the `blocked_graphs[graph] = True` flip in `complete()`.
    pub blocked_graphs: RefCell<HashMap<GraphKey, (GraphRef, bool)>>,
    /// RPython `self.frozen = False` (annrpython.py:46).
    pub frozen: RefCell<bool>,
    /// RPython `self.policy` (annrpython.py:47-51).
    pub policy: RefCell<AnnotatorPolicy>,
    /// RPython `self.bookkeeper` (annrpython.py:52-54).
    pub bookkeeper: Rc<Bookkeeper>,
    /// RPython `self.keepgoing` (annrpython.py:55).
    pub keepgoing: bool,
    /// RPython `self.failed_blocks = set()` (annrpython.py:56).
    pub failed_blocks: RefCell<HashMap<BlockKey, BlockRef>>,
    /// RPython `self.errors = []` (annrpython.py:57).
    pub errors: RefCell<Vec<String>>,
}

/// Placeholder position key — mirrors the tuple upstream uses as the
/// `position_key` payload passed to `Bookkeeper.at_position`.
pub type PositionKey = super::bookkeeper::PositionKey;

/// RPython `class BlockedInference(Exception)` (annrpython.py:673-693).
///
/// Thrown from `consider_op` / `flowin` to signal "the situation is
/// currently blocked, try other blocks and come back later". The
/// driver catches it in `flowin` / `processblock` and reflects the
/// block into `blocked_blocks` for a later retry.
///
/// Parity deviation (Rust adaptation): upstream uses `raise` /
/// `except BlockedInference`. Rust returns `Result<_, BlockedInference>`
/// from `consider_op`, and `flowin` converts back into the upstream
/// exception flow via match. The struct-level fields match upstream's
/// public attribute names.
pub struct BlockedInference {
    /// RPython `self.op` — the operation that tripped the block (may
    /// be the same raising op at the block tail). Stored as
    /// `HLOperation` because upstream consumers (`flowin`) inspect
    /// `self.op.opname`, which Rust recovers via `op.kind.opname()`.
    pub op: super::super::flowspace::operation::HLOperation,
    /// RPython `self.opindex` — the operation's position in
    /// `block.operations`. `None` for the `-1` sentinel upstream uses
    /// when the op came from a direct `consider_op` call.
    pub opindex: Option<usize>,
    /// RPython `self.break_at` — the bookkeeper's position at the
    /// moment of the block, or `None` if no reflow frame is active.
    pub break_at: Option<PositionKey>,
}

impl BlockedInference {
    /// RPython `BlockedInference.__init__(self, annotator, op, opindex)`
    /// (annrpython.py:677-684).
    pub fn new(
        ann: &RPythonAnnotator,
        op: super::super::flowspace::operation::HLOperation,
        opindex: Option<usize>,
    ) -> Self {
        BlockedInference {
            op,
            opindex,
            break_at: ann.bookkeeper.current_position_key(),
        }
    }
}

impl std::fmt::Debug for BlockedInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockedInference({:?})", self.op.kind)
    }
}

impl std::fmt::Display for BlockedInference {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.break_at {
            Some(pk) => write!(
                f,
                "<BlockedInference break_at graph={} block={} op={} [{:?}]>",
                pk.graph_id, pk.block_id, pk.op_index, self.op.kind
            ),
            None => write!(f, "<BlockedInference break_at ? [{:?}]>", self.op.kind),
        }
    }
}

/// Outcome of [`RPythonAnnotator::flowin_op_loop`] — decides which
/// exits-side dispatch path runs.
///
/// Upstream encodes this via try/except control flow; the Rust port
/// explicits the four outcomes so the exits-side logic can run after
/// all three "except" clauses have agreed on a final `exits` list.
enum OpLoopOutcome {
    /// upstream `else:` — op loop finished without blocks; dead-exit
    /// pruning applies before the exits-side dispatch runs.
    Normal,
    /// upstream `if e.op is block.raising_op:` — exits-side dispatch
    /// runs with `exits = [l for l in block.exits if l.exitcase is not None]`.
    BlockedOnRaisingOp,
    /// upstream `elif e.op.opname in ('simple_call', 'call_args', 'next'):
    /// return` — the block is harmlessly blocked and flowin should
    /// silently return without touching exits/notify.
    HarmlesslySwallowed,
    /// upstream `else: raise` — processblock records the block in
    /// `blocked_blocks`.
    Blocked(BlockedInference),
}

/// The three exception classes `flowin` (annrpython.py:488-537)
/// catches during the op loop, ported as a Rust `Result` error type
/// so the loop can dispatch without `panic!` / `catch_unwind`.
///
/// Upstream structure:
/// ```python
/// try:
///     ...
/// except BlockedInference as e:
///     ...
/// except HarmlesslyBlocked:
///     return
/// except AnnotatorError as e:
///     e.source = gather_error(self, graph, block, i)
///     if self.keepgoing:
///         self.errors.append(e)
///         self.failed_blocks.add(block)
///         return
///     raise
/// ```
pub enum FlowinError {
    /// upstream `BlockedInference` — transient block, retry later.
    Blocked(BlockedInference),
    /// upstream `annmodel.HarmlesslyBlocked` — swallow, return.
    Harmless(crate::annotator::model::HarmlesslyBlocked),
    /// upstream `annmodel.AnnotatorError` (incl. `UnionError`) —
    /// route through gather_error + keepgoing.
    Annotator(crate::annotator::model::AnnotatorError),
}

impl From<BlockedInference> for FlowinError {
    fn from(e: BlockedInference) -> Self {
        FlowinError::Blocked(e)
    }
}

impl From<crate::annotator::model::HarmlesslyBlocked> for FlowinError {
    fn from(e: crate::annotator::model::HarmlesslyBlocked) -> Self {
        FlowinError::Harmless(e)
    }
}

impl From<crate::annotator::model::AnnotatorError> for FlowinError {
    fn from(e: crate::annotator::model::AnnotatorError) -> Self {
        FlowinError::Annotator(e)
    }
}

impl From<crate::annotator::model::AnnotatorException> for FlowinError {
    fn from(e: crate::annotator::model::AnnotatorException) -> Self {
        use crate::annotator::model::AnnotatorException;
        match e {
            AnnotatorException::Annotator(e) => FlowinError::Annotator(e),
            AnnotatorException::Harmless(e) => FlowinError::Harmless(e),
        }
    }
}

/// RPython `gather_error(annotator, graph, block, operindex)`
/// (tool/error.py:67-82) — thin wrapper that binds the
/// [`crate::tool::error::gather_error`] port into annrpython's legacy
/// call shape. The operindex is `Option<usize>` upstream (`None`
/// means "no op, block-level"); callers passing a concrete index wrap
/// it in `Some`.
pub fn gather_error(
    ann: &RPythonAnnotator,
    graph: &GraphRef,
    block: &BlockRef,
    i: usize,
) -> String {
    crate::tool::error::gather_error(ann, graph, block, Some(i))
}

/// RAII guard returned by [`RPythonAnnotator::using_policy`] — mirrors
/// upstream's `@contextmanager` so callers use `let _g =
/// ann.using_policy(...);` in the same shape as `with
/// self.using_policy(policy):`.
pub struct PolicyGuard<'a> {
    ann: &'a RPythonAnnotator,
    saved: Option<AnnotatorPolicy>,
}

impl<'a> Drop for PolicyGuard<'a> {
    fn drop(&mut self) {
        if let Some(saved) = self.saved.take() {
            *self.ann.policy.borrow_mut() = saved;
        }
    }
}

impl RPythonAnnotator {
    /// RPython `RPythonAnnotator.__init__(self, translator=None,
    /// policy=None, bookkeeper=None, keepgoing=False)`
    /// (annrpython.py:26-57).
    ///
    /// Follows upstream: if `bookkeeper` is `None`, construct a fresh
    /// `Bookkeeper(self)`; if `policy` is `None`, install a default
    /// `AnnotatorPolicy()`. The `TLS.bookkeeper = self.bookkeeper`
    /// assignment at upstream line ~81 is folded into
    /// [`Bookkeeper::enter`] — no-op here because we haven't entered a
    /// reflow frame yet.
    pub fn new(
        translator: Option<TranslationContext>,
        policy: Option<AnnotatorPolicy>,
        bookkeeper: Option<Rc<Bookkeeper>>,
        keepgoing: bool,
    ) -> Rc<Self> {
        let policy = policy.unwrap_or_else(AnnotatorPolicy::new);
        let bookkeeper =
            bookkeeper.unwrap_or_else(|| Rc::new(Bookkeeper::new_with_policy(policy.clone())));
        // upstream annrpython.py:30-34 — default to a fresh
        // TranslationContext when caller passes None.
        let translator = translator.unwrap_or_else(TranslationContext::new);
        // Upstream `Bookkeeper.__init__(self, annotator)` stores
        // `self.annotator = annotator` and `RPythonAnnotator.__init__`
        // stores `translator.annotator = self`. Rust uses
        // `Rc::new_cyclic` so both Weak back-references can be
        // installed before any caller observes the final `Rc<Self>`.
        Rc::new_cyclic(|weak: &std::rc::Weak<Self>| {
            bookkeeper.set_annotator(weak.clone());
            translator.set_annotator(weak.clone());
            RPythonAnnotator {
                translator: RefCell::new(translator),
                genpendingblocks: RefCell::new(vec![HashMap::new()]),
                annotated: RefCell::new(HashMap::new()),
                all_blocks: RefCell::new(HashMap::new()),
                added_blocks: RefCell::new(None),
                links_followed: RefCell::new(HashSet::new()),
                notify: RefCell::new(HashMap::new()),
                fixed_graphs: RefCell::new(HashMap::new()),
                blocked_blocks: RefCell::new(HashMap::new()),
                blocked_graphs: RefCell::new(HashMap::new()),
                frozen: RefCell::new(false),
                policy: RefCell::new(policy),
                bookkeeper: Rc::clone(&bookkeeper),
                keepgoing,
                failed_blocks: RefCell::new(HashMap::new()),
                errors: RefCell::new(Vec::new()),
            }
        })
    }

    /// RPython `annotation(self, arg)` (annrpython.py:273-280).
    ///
    /// ```python
    /// def annotation(self, arg):
    ///     "Gives the SomeValue corresponding to the given Variable or Constant."
    ///     if isinstance(arg, Variable):
    ///         return arg.annotation
    ///     elif isinstance(arg, Constant):
    ///         return self.bookkeeper.immutablevalue(arg.value)
    ///     else:
    ///         raise TypeError('Variable or Constant expected, got %r' % (arg,))
    /// ```
    ///
    /// `None` means the variable has not been bound yet (upstream
    /// `arg.annotation` returns `None` on an unseen variable); a
    /// constant whose value cannot be lifted via `immutablevalue`
    /// is an upstream exception — Rust mirrors that by panicking
    /// (callers can always wrap in `std::panic::catch_unwind` when
    /// emulating the upstream `try/except KeyError` style).
    pub fn annotation(&self, arg: &Hlvalue) -> Option<SomeValue> {
        match arg {
            Hlvalue::Variable(v) => v.annotation.as_ref().map(|rc| (**rc).clone()),
            Hlvalue::Constant(c) => match self.bookkeeper.immutableconstant(c) {
                Ok(sv) => Some(sv),
                Err(e) => panic!(
                    "AnnotatorError: immutablevalue({:?}) failed — {}",
                    c.value, e
                ),
            },
        }
    }

    /// RPython `binding(self, arg)` (annrpython.py:282-287).
    ///
    /// ```python
    /// def binding(self, arg):
    ///     s_arg = self.annotation(arg)
    ///     if s_arg is None:
    ///         raise KeyError
    ///     return s_arg
    /// ```
    pub fn binding(&self, arg: &Hlvalue) -> SomeValue {
        self.annotation(arg).expect("KeyError: no binding for arg")
    }

    /// RPython `typeannotation(self, t)` (annrpython.py:289-290).
    pub fn typeannotation(
        &self,
        spec: &super::signature::AnnotationSpec,
    ) -> Result<SomeValue, super::signature::SignatureError> {
        super::signature::annotation(spec, Some(&self.bookkeeper))
    }

    /// RPython `setbinding(self, arg, s_value)` (annrpython.py:292-299).
    ///
    /// ```python
    /// def setbinding(self, arg, s_value):
    ///     s_old = arg.annotation
    ///     if s_old is not None:
    ///         if not s_value.contains(s_old):
    ///             log.WARNING(...)
    ///             assert False
    ///     arg.annotation = s_value
    /// ```
    ///
    /// Rust requires `&mut Variable` to mutate `annotation`. The
    /// driver passes owned `Variable` references while processing a
    /// block, so this is called with `&mut v` there.
    pub fn setbinding(&self, arg: &mut Variable, s_value: SomeValue) {
        if let Some(s_old) = arg.annotation.as_ref() {
            if !s_value.contains(s_old) {
                // upstream: `log.WARNING(...); assert False`.
                // Lattice widening contract — a binding cannot move
                // backwards.
                panic!(
                    "setbinding: new value does not contain old ({:?} ⊄ {:?})",
                    s_value, **s_old
                );
            }
        }
        arg.annotation = Some(Rc::new(s_value));
    }

    /// RPython `warning(self, msg, pos=None)` (annrpython.py:301-...).
    ///
    /// Driver-level logging. Non-ported methods (`build_types`,
    /// `complete`, `processblock`, ...) land with Commit 7 Part A.
    pub fn warning(&self, msg: &str) {
        eprintln!("[annrpython warning] {}", msg);
    }

    /// Install `self.bookkeeper` into `TLS.bookkeeper` — mirrors
    /// upstream annrpython.py:~81 side-effect so `getbookkeeper()`
    /// works during calls that don't go through
    /// `Bookkeeper::at_position`.
    pub fn install_tls_bookkeeper(&self) {
        TLS.with(|state| state.borrow_mut().bookkeeper = Some(Rc::clone(&self.bookkeeper)));
    }

    // ======================================================================
    // RPython `convenience high-level interface` — diagnostics + validation
    // helpers used by the driver/caller side.
    // ======================================================================

    // ======================================================================
    // RPython `convenience high-level interface` (annrpython.py:71-141) —
    // build_types / build_graph_types / annotate_helper entrypoints.
    // ======================================================================

    /// RPython `build_graph_types(self, flowgraph, inputcells,
    /// complete_now=True)` (annrpython.py:130-141).
    ///
    /// ```python
    /// def build_graph_types(self, flowgraph, inputcells, complete_now=True):
    ///     checkgraph(flowgraph)
    ///     nbarg = len(flowgraph.getargs())
    ///     assert len(inputcells) == nbarg
    ///     self.addpendinggraph(flowgraph, inputcells)
    ///     if complete_now:
    ///         self.complete()
    ///     return self.annotation(flowgraph.getreturnvar())
    /// ```
    pub fn build_graph_types(
        &self,
        graph: &GraphRef,
        inputcells: &[SomeValue],
        complete_now: bool,
    ) -> Option<SomeValue> {
        checkgraph(&graph.borrow());
        let nbarg = graph.borrow().getargs().len();
        assert_eq!(
            nbarg,
            inputcells.len(),
            "build_graph_types: wrong number of input cells ({} args vs {} cells)",
            nbarg,
            inputcells.len()
        );
        self.addpendinggraph(graph, inputcells);
        if complete_now {
            self.complete();
        }
        let returnvar = graph.borrow().getreturnvar();
        self.annotation(&returnvar)
    }

    /// RPython `complete_helpers(self)` (annrpython.py:112-120).
    ///
    /// Drives `complete()` inside a scoped `added_blocks` tracker so
    /// the caller can learn which blocks were added for the helper.
    ///
    /// ```python
    /// def complete_helpers(self):
    ///     saved = self.added_blocks
    ///     self.added_blocks = {}
    ///     try:
    ///         self.complete()
    ///         self.simplify(block_subset=self.added_blocks)
    ///     finally:
    ///         self.added_blocks = saved
    /// ```
    pub fn complete_helpers(&self) {
        let saved = self.added_blocks.borrow_mut().replace(HashMap::new());
        self.complete();
        // upstream annrpython.py:118 passes `block_subset=self.added_blocks`
        // unconditionally (the dict we seeded at line 117), so simplify
        // only touches graphs reachable from blocks added during this
        // `complete()`. Passing `Some(&[])` (rather than `None`) when
        // the set is empty matches upstream's empty-dict semantics —
        // simplify then iterates zero graphs and skips
        // `perform_normalizations`.
        let subset_blocks: Vec<BlockRef> = self
            .added_blocks
            .borrow()
            .as_ref()
            .expect("complete_helpers: added_blocks tracker disappeared")
            .values()
            .cloned()
            .collect();
        self.simplify(Some(subset_blocks.as_slice()), None);
        *self.added_blocks.borrow_mut() = saved;
    }

    /// RPython `using_policy(self, policy)` (annrpython.py:122-128).
    ///
    /// A context manager that temporarily swaps `self.policy`. The
    /// Rust port returns a RAII guard that restores the saved policy
    /// on drop, matching the upstream `try ... finally` contract.
    pub fn using_policy(&self, policy: AnnotatorPolicy) -> PolicyGuard<'_> {
        let saved = std::mem::replace(&mut *self.policy.borrow_mut(), policy);
        PolicyGuard {
            ann: self,
            saved: Some(saved),
        }
    }

    // ======================================================================
    // RPython `interface for annotator.bookkeeper` (annrpython.py:313-336).
    // ======================================================================

    /// RPython `recursivecall(self, graph, whence, inputcells)`
    /// (annrpython.py:315-336).
    ///
    /// Used by `bookkeeper.pbc_call` to bring a callee graph into the
    /// pending queue, wire the call site into `self.notify`, and
    /// return the callee's current return annotation. `whence` is the
    /// caller's `(parent_graph, parent_block, parent_index)` position
    /// — passing `None` marks an entry-point call.
    pub fn recursivecall(
        &self,
        graph: &GraphRef,
        whence: Option<(GraphRef, BlockRef, usize)>,
        inputcells: &[SomeValue],
    ) -> SomeValue {
        use super::model::s_impossible_value;
        if let Some((parent_graph, parent_block, parent_index)) = whence.clone() {
            // upstream: `self.translator.update_call_graph(parent_graph, graph, tag)`.
            let tag = (BlockKey::of(&parent_block), parent_index);
            self.translator
                .borrow()
                .update_call_graph(&parent_graph, graph, tag);
            // upstream: `returnpositions = self.notify.setdefault(graph.returnblock, set())`
            let returnblock = graph.borrow().returnblock.clone();
            // upstream: `position_key = (parent_graph, parent_block,
            // parent_index)`. Use `from_refs` so the weak Graph/Block
            // refs are retained for downstream consumers (reflowfromposition).
            let pk = super::bookkeeper::PositionKey::from_refs(
                &parent_graph,
                &parent_block,
                parent_index,
            );
            self.notify
                .borrow_mut()
                .entry(BlockKey::of(&returnblock))
                .or_default()
                .insert(pk);
        }

        // upstream: `self.addpendingblock(graph, graph.startblock, inputcells)`.
        let startblock = graph.borrow().startblock.clone();
        self.addpendingblock(graph, &startblock, inputcells);

        // upstream: `v = graph.getreturnvar(); return self.binding(v)` or
        // s_ImpossibleValue on KeyError.
        let v = graph.borrow().getreturnvar();
        self.annotation(&v).unwrap_or_else(s_impossible_value)
    }

    /// RPython `gettype(self, variable)` (annrpython.py:143-156).
    ///
    /// ```python
    /// def gettype(self, variable):
    ///     if isinstance(variable, Constant):
    ///         return type(variable.value)
    ///     elif isinstance(variable, Variable):
    ///         s_variable = variable.annotation
    ///         if s_variable:
    ///             return s_variable.knowntype
    ///         else:
    ///             return object
    ///     else:
    ///         raise TypeError(...)
    /// ```
    ///
    /// Rust port surfaces the result as [`super::model::KnownType`];
    /// `KnownType::Object` stands in for upstream's `object` fallback.
    /// Constants route through the bookkeeper's `immutablevalue` to
    /// recover the KnownType that the value would receive post-
    /// annotation.
    pub fn gettype(&self, arg: &Hlvalue) -> super::model::KnownType {
        use super::model::{KnownType, SomeObjectTrait};
        match arg {
            Hlvalue::Variable(v) => match v.annotation.as_ref() {
                Some(rc) => (**rc).knowntype(),
                None => KnownType::Object,
            },
            Hlvalue::Constant(c) => self
                .bookkeeper
                .immutableconstant(c)
                .map(|sv| sv.knowntype())
                .unwrap_or(KnownType::Object),
        }
    }

    /// RPython `getuserclassdefinitions(self)` (annrpython.py:158-160):
    /// `return self.bookkeeper.classdefs`.
    pub fn getuserclassdefinitions(
        &self,
    ) -> Vec<Rc<std::cell::RefCell<super::classdesc::ClassDef>>> {
        self.bookkeeper.classdefs.borrow().clone()
    }

    /// RPython `validate(self)` (annrpython.py:269-271):
    /// `self.bookkeeper.check_no_flags_on_instances()`.
    pub fn validate(&self) {
        self.bookkeeper.check_no_flags_on_instances();
    }

    /// Internal helper behind [`Self::call_sites`] that also preserves
    /// the upstream `(graph, block, opindex)` identity as a
    /// [`PositionKey`].
    pub fn call_sites_with_positions(
        &self,
    ) -> Vec<(
        super::super::flowspace::model::SpaceOperation,
        super::bookkeeper::PositionKey,
    )> {
        let mut out = Vec::new();
        let added = self.added_blocks.borrow();
        let blocks: Vec<(GraphRef, BlockRef)> = match added.as_ref() {
            // upstream: `newblocks = self.added_blocks` (dict/set of Blocks).
            Some(set) => {
                let annotated = self.annotated.borrow();
                set.values()
                    .filter_map(|block| {
                        let graph = annotated
                            .get(&BlockKey::of(block))
                            .and_then(|g| g.clone())?;
                        Some((graph, block.clone()))
                    })
                    .collect()
            }
            // upstream: `newblocks = self.annotated  # all of them`.
            // Iterating a Python dict yields keys (Block objects). The
            // Rust port rebuilds those keys through `all_blocks` and
            // keeps the associated graph alongside them so
            // `PositionKey::from_refs` can reconstruct the real call
            // site identity.
            None => {
                let annotated = self.annotated.borrow();
                let all_blocks = self.all_blocks.borrow();
                annotated
                    .iter()
                    .filter_map(|(block_key, graph)| {
                        let graph = graph.clone()?;
                        let block = all_blocks.get(block_key)?.clone();
                        Some((graph, block))
                    })
                    .collect()
            }
        };
        drop(added);
        for (graph, block) in blocks {
            let blk = block.borrow();
            for (i, op) in blk.operations.iter().enumerate() {
                if op.opname == "simple_call" || op.opname == "call_args" {
                    out.push((
                        op.clone(),
                        super::bookkeeper::PositionKey::from_refs(&graph, &block, i),
                    ));
                }
                // upstream: `if op.result.annotation is None: break`.
                if let super::super::flowspace::model::Hlvalue::Variable(v) = &op.result {
                    if v.annotation.is_none() {
                        break;
                    }
                }
            }
        }
        out
    }

    /// RPython `call_sites(self)` (annrpython.py:342-353).
    ///
    /// Yields every `simple_call` / `call_args` operation over the
    /// driver's tracked blocks — `added_blocks` if set, else
    /// everything in `annotated`. Stops early on a block once an
    /// operation with an unannotated result surfaces (matches the
    /// upstream `break` on a partially-annotated block).
    pub fn call_sites(&self) -> Vec<super::super::flowspace::model::SpaceOperation> {
        self.call_sites_with_positions()
            .into_iter()
            .map(|(op, _)| op)
            .collect()
    }

    /// RPython `simplify(self, block_subset=None, extra_passes=None)`
    /// (annrpython.py:357-373).
    ///
    /// Upstream body:
    ///
    /// ```python
    /// transform.transform_graph(self, block_subset=block_subset,
    ///                           extra_passes=extra_passes)
    /// if block_subset is None:
    ///     graphs = self.translator.graphs
    /// else:
    ///     graphs = {}
    ///     for block in block_subset:
    ///         graph = self.annotated.get(block)
    ///         if graph:
    ///             graphs[graph] = True
    /// for graph in graphs:
    ///     simplify.eliminate_empty_blocks(graph)
    /// self.bookkeeper.compute_at_fixpoint()
    /// if block_subset is None:
    ///     perform_normalizations(self)
    /// ```
    ///
    /// The Rust port routes the fixpoint call through the bookkeeper's
    /// `compute_at_fixpoint` and invokes `eliminate_empty_blocks` over
    /// the translator's graph set. `perform_normalizations` remains a
    /// no-op placeholder until `rpython/rtyper/normalizecalls.py` lands.
    pub fn simplify(
        &self,
        block_subset: Option<&[BlockRef]>,
        extra_passes: Option<&[super::super::translator::transform::TransformPass]>,
    ) {
        // upstream: `transform.transform_graph(self, block_subset=..., extra_passes=...)`.
        super::super::translator::transform::transform_graph(self, extra_passes, block_subset);
        // upstream:
        //   if block_subset is None:
        //       graphs = self.translator.graphs
        //   else:
        //       graphs = {}
        //       for block in block_subset:
        //           graph = self.annotated.get(block)
        //           if graph:
        //               graphs[graph] = True
        let graphs: Vec<GraphRef> = match block_subset {
            None => self.translator.borrow().graphs.borrow().clone(),
            Some(blocks) => {
                let mut seen: HashMap<GraphKey, GraphRef> = HashMap::new();
                let annotated = self.annotated.borrow();
                for block in blocks {
                    let key = BlockKey::of(block);
                    if let Some(Some(graph)) = annotated.get(&key) {
                        seen.entry(GraphKey::of(graph)).or_insert(graph.clone());
                    }
                }
                seen.into_values().collect()
            }
        };
        // upstream: `for graph in graphs: simplify.eliminate_empty_blocks(graph)`.
        for graph_ref in graphs {
            let fg = graph_ref.borrow();
            super::super::translator::simplify::eliminate_empty_blocks(&fg);
        }
        // upstream: `self.bookkeeper.compute_at_fixpoint()`.
        self.bookkeeper.compute_at_fixpoint();
        // upstream: `if block_subset is None: perform_normalizations(self)`
        // — perform_normalizations is rtyper-phase; deferred.
        let _ = block_subset;
    }

    /// RPython `apply_renaming(self, s_out, renaming)`
    /// (annrpython.py:448-473).
    ///
    /// Rewrites the `is_type_of` list of a `SomeTypeOf` and the
    /// `knowntypedata` table of a `SomeBool` under the provided
    /// variable-renaming. `renaming` maps each old variable to a
    /// (possibly empty) list of new variables — the annotation is
    /// either rebuilt against the renamed variables or stripped if
    /// no renaming entry exists.
    pub fn apply_renaming(
        &self,
        s_out: SomeValue,
        renaming: &HashMap<Rc<Variable>, Vec<Rc<Variable>>>,
    ) -> SomeValue {
        use super::model::{
            KnownTypeData, SomeBool, SomeTypeOf, SomeValue as SV, add_knowntypedata, typeof_vars,
        };
        // Rebuild SomeTypeOf.is_type_of against the renamed variables.
        // upstream: `for v in s_out.is_type_of: renamed += renaming[v]`.
        let s_out = match s_out {
            SV::TypeOf(t) => {
                let mut renamed_is_type_of: Vec<Rc<Variable>> = Vec::new();
                for v in &t.is_type_of {
                    if let Some(new_vs) = renaming.get(v) {
                        renamed_is_type_of.extend(new_vs.iter().map(Rc::clone));
                    }
                }
                let newcell = typeof_vars(&renamed_is_type_of);
                match newcell {
                    SV::TypeOf(mut nc) => {
                        if t.base.const_box.is_some() {
                            nc.base.const_box = t.base.const_box.clone();
                        }
                        SV::TypeOf(nc)
                    }
                    // typeof_vars returns SomeType when args_v is empty.
                    other => other,
                }
            }
            other => other,
        };
        // Rewrite SomeBool.knowntypedata under the renaming.
        if let SV::Bool(b) = &s_out {
            if let Some(ktd) = &b.knowntypedata {
                let mut renamed: KnownTypeData = HashMap::new();
                for (value, constraints) in ktd {
                    let mut new_inner: HashMap<Rc<Variable>, SomeValue> = HashMap::new();
                    for (v, s) in constraints {
                        if let Some(new_vs) = renaming.get(v) {
                            for new_v in new_vs {
                                new_inner.insert(Rc::clone(new_v), s.clone());
                            }
                        }
                    }
                    if !new_inner.is_empty() {
                        renamed.insert(*value, new_inner);
                    }
                }
                // upstream: `assert isinstance(s_out, SomeBool)` + rebuild.
                let mut newcell = SomeBool::new();
                // SomeObject.is_immutable_constant (model.py:106-107):
                // `self.is_constant() and self.immutable`. SomeBool
                // carries `immutable = True`, so the precondition here
                // is just "const_box populated".
                if b.base.const_box.is_some() {
                    newcell.base.const_box = b.base.const_box.clone();
                }
                newcell.set_knowntypedata(renamed);
                // rebuild add_knowntypedata side-effect: not needed —
                // set_knowntypedata already stores the rebuilt map.
                let _ = add_knowntypedata; // keep the import alive.
                let _ = SomeTypeOf::new(vec![]); // keep the import alive.
                return SV::Bool(newcell);
            }
        }
        s_out
    }

    /// RPython `whereami(self, position_key)` (annrpython.py:476-486).
    ///
    /// ```python
    /// graph, block, i = position_key
    /// ...
    /// return repr(graph) + blk + opid
    /// ```
    ///
    /// Rust `PositionKey` carries the u64-encoded (graph, block, index)
    /// tuple. For log messages we stringify each component.
    pub fn whereami(&self, pk: PositionKey) -> String {
        let opid = if pk.op_index > 0 {
            format!(" op={}", pk.op_index)
        } else {
            String::new()
        };
        format!("graph={} block={}{}", pk.graph_id, pk.block_id, opid)
    }

    /// RPython `complete(self)` (annrpython.py:226-267).
    ///
    /// Drains every pending-block generation until fixpoint, then
    /// validates that every tracked graph has an annotated return
    /// variable. Blocked blocks / failed blocks trigger an
    /// `AnnotatorError` (panic here for now; Commit 7c surfaces the
    /// `format_blocked_annotation_error` payload).
    pub fn complete(&self) {
        use super::model::{SomeObjectTrait, s_impossible_value};
        loop {
            self.complete_pending_blocks();
            // upstream: `self.policy.no_more_blocks_to_annotate(self)`.
            self.policy.borrow().no_more_blocks_to_annotate(self);
            let any_pending = self.genpendingblocks.borrow().iter().any(|d| !d.is_empty());
            if !any_pending {
                break;
            }
        }

        // Collect the set of graphs whose return annotation must be
        // forced after the fixpoint loop.
        let added = self.added_blocks.borrow();
        let (newgraphs, got_blocked) = match added.as_ref() {
            Some(added_set) => {
                // Walk added_set's blocks, resolve annotated[block] →
                // owning graph; treat `None` (= False) entries as
                // "blocked".
                let annotated = self.annotated.borrow();
                let mut graphs: HashMap<GraphKey, GraphRef> = HashMap::new();
                let mut got_blocked = false;
                for bkey in added_set.keys() {
                    match annotated
                        .get(bkey)
                        .expect("complete: added_blocks entry missing from annotated")
                    {
                        Some(graph) => {
                            graphs.insert(GraphKey::of(graph), Rc::clone(graph));
                        }
                        None => got_blocked = true,
                    }
                }
                (graphs.values().cloned().collect::<Vec<_>>(), got_blocked)
            }
            None => {
                let got_blocked = self.annotated.borrow().values().any(|g| g.is_none());
                let graphs: Vec<GraphRef> = self.translator.borrow().graphs.borrow().clone();
                (graphs, got_blocked)
            }
        };
        drop(added);

        if !self.failed_blocks.borrow().is_empty() {
            let errors = self.errors.borrow();
            let text = if errors.is_empty() {
                format!("Annotation failed, {} errors were recorded.", errors.len())
            } else {
                format!(
                    "Annotation failed, {} errors were recorded:\n{}",
                    errors.len(),
                    errors.join("\n-----")
                )
            };
            panic!("AnnotatorError: {}", text);
        }

        if got_blocked {
            // Upstream: flip every blocked graph's flag and construct
            // the multi-line error string via
            // `format_blocked_annotation_error`.
            let mut bg = self.blocked_graphs.borrow_mut();
            for (_k, (_g, flag)) in bg.iter_mut() {
                *flag = true;
            }
            drop(bg);
            let blocked_blocks = self.blocked_blocks.borrow();
            let text = crate::tool::error::format_blocked_annotation_error(self, &blocked_blocks);
            panic!("AnnotatorError: {}", text);
        }

        // Force every return-var annotation to exist.
        for graph in newgraphs {
            let returnvar = graph.borrow().getreturnvar();
            if let Hlvalue::Variable(v) = &returnvar {
                if v.annotation.is_none() {
                    // upstream: `self.setbinding(v, s_ImpossibleValue)`.
                    // `setbinding` mutates the variable; reach through
                    // the graph's startblock to find the mutable slot.
                    let sv = s_impossible_value();
                    let mut graph_mut = graph.borrow_mut();
                    let mut rb = graph_mut.returnblock.borrow_mut();
                    if let Hlvalue::Variable(vm) = &mut rb.inputargs[0] {
                        self.setbinding(vm, sv);
                    }
                    drop(rb);
                    drop(graph_mut);
                }
            }
            // Upstream: `v = graph.exceptblock.inputargs[1]; if
            // v.annotation is not None and v.annotation.can_be_none():
            //     raise AnnotatorError(...)`.
            let excvar = {
                let g = graph.borrow();
                let eb = g.exceptblock.borrow();
                eb.inputargs[1].clone()
            };
            if let Hlvalue::Variable(v) = excvar {
                if let Some(rc) = v.annotation.as_ref() {
                    if rc.can_be_none() {
                        panic!(
                            "AnnotatorError: {:?} is found by annotation to possibly raise None, \
                             but the None was not suppressed by the flow space",
                            graph.borrow().name
                        );
                    }
                }
            }
        }
    }

    // ======================================================================
    // RPython `medium-level interface` (annrpython.py:162-224) — block
    // scheduling and pending-blocks queue.
    // ======================================================================

    /// RPython `addpendinggraph(self, flowgraph, inputcells)`
    /// (annrpython.py:164-165).
    ///
    /// ```python
    /// def addpendinggraph(self, flowgraph, inputcells):
    ///     self.addpendingblock(flowgraph, flowgraph.startblock, inputcells)
    /// ```
    pub fn addpendinggraph(&self, graph: &GraphRef, inputcells: &[SomeValue]) {
        let startblock = graph.borrow().startblock.clone();
        self.addpendingblock(graph, &startblock, inputcells);
    }

    /// RPython `addpendingblock(self, graph, block, cells)`
    /// (annrpython.py:167-191).
    ///
    /// Registers an entry point into `block` with the given input
    /// cells. If `graph` is in `fixed_graphs`, the pass is a
    /// safety-check only: the new annotations must not widen the
    /// existing inputs. Otherwise the block is seeded via
    /// `bindinputargs` (fresh) or `mergeinputargs` (already seen), and
    /// then queued via `schedulependingblock` if the block still needs
    /// flowin work.
    pub fn addpendingblock(&self, graph: &GraphRef, block: &BlockRef, cells: &[SomeValue]) {
        let gkey = GraphKey::of(graph);
        if self.fixed_graphs.borrow().contains_key(&gkey) {
            // Upstream: safety-check path for graphs that have already
            // been rtyped. The new annotations must fit within the
            // existing ones (`unionof(old, new) == old`). We validate
            // positionally against `block.inputargs`.
            let blk = block.borrow();
            for (a, s_newarg) in blk.inputargs.iter().zip(cells) {
                let Hlvalue::Variable(v) = a else { continue };
                let s_old = v
                    .annotation
                    .as_ref()
                    .map(|rc| (**rc).clone())
                    .expect("addpendingblock: fixed graph's inputarg lacks annotation");
                let merged = unionof([&s_old, s_newarg]).unwrap_or_else(|_| {
                    panic!(
                        "AnnotatorError: Late-stage annotation is not allowed to modify the \
                         existing annotation for variable {} ({:?})",
                        v.name(),
                        s_old
                    )
                });
                if merged != s_old {
                    panic!(
                        "AnnotatorError: Late-stage annotation is not allowed to modify the \
                         existing annotation for variable {} ({:?})",
                        v.name(),
                        s_old
                    );
                }
            }
            return;
        }

        // Upstream: `assert not self.frozen`.
        assert!(
            !*self.frozen.borrow(),
            "addpendingblock: annotator is frozen"
        );

        let bkey = BlockKey::of(block);
        let seen_before = self.annotated.borrow().contains_key(&bkey);
        if !seen_before {
            self.bindinputargs(graph, block, cells);
        } else {
            self.mergeinputargs(graph, block, cells);
        }
        // After binding/merging, `annotated[block]` is either
        // `Some(False)` (needs flowin) or `Some(graph)` (already
        // completed). Upstream: `if not self.annotated[block]:
        // self.schedulependingblock(graph, block)`.
        let needs_flow = matches!(self.annotated.borrow().get(&bkey), Some(None));
        if needs_flow {
            self.schedulependingblock(graph, block);
        }
    }

    /// RPython `schedulependingblock(self, graph, block)`
    /// (annrpython.py:193-201).
    ///
    /// ```python
    /// def schedulependingblock(self, graph, block):
    ///     generation = getattr(block, 'generation', 0)
    ///     self.genpendingblocks[generation][block] = graph
    /// ```
    pub fn schedulependingblock(&self, graph: &GraphRef, block: &BlockRef) {
        let generation = block.borrow().generation.unwrap_or(0) as usize;
        let mut pending = self.genpendingblocks.borrow_mut();
        while pending.len() <= generation {
            pending.push(HashMap::new());
        }
        pending[generation].insert(BlockKey::of(block), (Rc::clone(block), Rc::clone(graph)));
    }

    /// RPython `complete_pending_blocks(self)` (annrpython.py:203-224).
    ///
    /// Drains every pending-block generation, bumping each block's
    /// `generation` and calling `processblock(graph, block)`. Because
    /// `processblock` can re-queue blocks into a later generation via
    /// `addpendingblock → schedulependingblock`, the outer loop keeps
    /// scanning until every dict is empty.
    pub fn complete_pending_blocks(&self) {
        loop {
            // Find the first non-empty generation dict.
            let mut generation: usize = 0;
            {
                let pending = self.genpendingblocks.borrow();
                while generation < pending.len() && pending[generation].is_empty() {
                    generation += 1;
                }
                if generation == pending.len() {
                    return; // all empty ⇒ done
                }
            }

            // Upstream: `gen += 1; if len(self.genpendingblocks) == gen:
            //             self.genpendingblocks.append({})`.
            let next_generation_u32 = (generation + 1) as u32;
            {
                let mut pending = self.genpendingblocks.borrow_mut();
                if pending.len() == generation + 1 {
                    pending.push(HashMap::new());
                }
            }

            // Drain the current generation's pending blocks.
            loop {
                let entry = {
                    let mut pending = self.genpendingblocks.borrow_mut();
                    match pending[generation].keys().next().cloned() {
                        Some(k) => pending[generation].remove(&k),
                        None => None,
                    }
                };
                let Some((block, graph)) = entry else { break };
                block.borrow_mut().generation = Some(next_generation_u32);
                self.processblock(&graph, &block);
            }
        }
    }

    /// RPython `reflowpendingblock(self, graph, block)`
    /// (annrpython.py:414-420).
    ///
    /// Re-queues an already-annotated block for a fresh flowin pass.
    /// Used when a dependency tracked in `notify` fires.
    pub fn reflowpendingblock(&self, graph: &GraphRef, block: &BlockRef) {
        assert!(
            !*self.frozen.borrow(),
            "reflowpendingblock: annotator is frozen"
        );
        assert!(
            !self
                .fixed_graphs
                .borrow()
                .contains_key(&GraphKey::of(graph)),
            "reflowpendingblock: graph is fixed"
        );
        self.schedulependingblock(graph, block);
        let bkey = BlockKey::of(block);
        assert!(
            self.annotated.borrow().contains_key(&bkey),
            "reflowpendingblock: block not yet annotated"
        );
        self.annotated.borrow_mut().insert(bkey.clone(), None);
        self.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), Rc::clone(block));
        self.blocked_blocks
            .borrow_mut()
            .insert(bkey, (Rc::clone(block), Rc::clone(graph), None));
    }

    /// RPython `follow_link(self, graph, link, constraints)`
    /// (annrpython.py:579-603).
    ///
    /// ```python
    /// def follow_link(self, graph, link, constraints):
    ///     assert not (isinstance(link.exitcase, (types.ClassType, type)) and
    ///             issubclass(link.exitcase, BaseException))
    ///
    ///     ignore_link = False
    ///     inputs_s = []
    ///     renaming = defaultdict(list)
    ///     for v_out, v_input in zip(link.args, link.target.inputargs):
    ///         renaming[v_out].append(v_input)
    ///
    ///     for v_out in link.args:
    ///         s_out = self.annotation(v_out)
    ///         if v_out in constraints:
    ///             s_constraint = constraints[v_out]
    ///             s_out = pair(s_out, s_constraint).improve()
    ///             # ignore links that try to pass impossible values
    ///             if s_out == s_ImpossibleValue:
    ///                 ignore_link = True
    ///         s_out = self.apply_renaming(s_out, renaming)
    ///         inputs_s.append(s_out)
    ///     if ignore_link:
    ///         return
    ///
    ///     self.links_followed[link] = True
    ///     self.addpendingblock(graph, link.target, inputs_s)
    /// ```
    ///
    /// `constraints` maps variables in `link.args` to the `SomeValue`
    /// they are *known* to carry when the link is taken — the caller
    /// extracts this from the source block's `exitswitch.annotation`
    /// `knowntypedata`.
    pub fn follow_link(
        &self,
        graph: &GraphRef,
        link: &LinkRef,
        constraints: &HashMap<Rc<Variable>, SomeValue>,
    ) {
        use super::binaryop::improve;

        let link_borrow = link.borrow();
        let target_rc = link_borrow
            .target
            .clone()
            .expect("follow_link: link.target must be present");

        // upstream assert: exitcase must not be an exception class.
        // (Exception exits go through follow_raise_link instead.) In the
        // Rust port exception exitcases surface as Hlvalue::Constant
        // with a class HostObject; the caller is responsible for not
        // routing those here.

        // Build renaming map: link.args (source) → link.target.inputargs (dest).
        let target_inputargs = target_rc.borrow().inputargs.clone();
        let mut renaming: HashMap<Rc<Variable>, Vec<Rc<Variable>>> = HashMap::new();
        for (v_out_opt, v_input) in link_borrow.args.iter().zip(&target_inputargs) {
            let Some(Hlvalue::Variable(vo)) = v_out_opt else {
                continue;
            };
            let Hlvalue::Variable(vi) = v_input else {
                continue;
            };
            renaming
                .entry(Rc::new(vo.clone()))
                .or_default()
                .push(Rc::new(vi.clone()));
        }

        let mut ignore_link = false;
        let mut inputs_s: Vec<SomeValue> = Vec::new();
        for v_out_opt in &link_borrow.args {
            let Some(v_out) = v_out_opt else {
                // Transient None in link.args (merge-link undefined-local)
                // — upstream treats absence as impossible; seed the
                // target inputarg with the bottom annotation.
                inputs_s.push(SomeValue::Impossible);
                continue;
            };
            let mut s_out = self.annotation(v_out).unwrap_or(SomeValue::Impossible);
            // upstream: `if v_out in constraints: s_out = pair(s_out, s_c).improve()`.
            if let Hlvalue::Variable(v) = v_out {
                let key = Rc::new(v.clone());
                if let Some(s_constraint) = constraints.get(&key) {
                    s_out = improve(&s_out, s_constraint);
                    if matches!(s_out, SomeValue::Impossible) {
                        ignore_link = true;
                    }
                }
            }
            s_out = self.apply_renaming(s_out, &renaming);
            inputs_s.push(s_out);
        }
        if ignore_link {
            return;
        }

        let lkey = LinkKey::of(link);
        drop(link_borrow);
        self.links_followed.borrow_mut().insert(lkey);
        self.addpendingblock(graph, &target_rc, &inputs_s);
    }

    /// RPython `follow_raise_link(self, graph, link, s_last_exc_value)`
    /// (annrpython.py:605-639).
    ///
    /// ```python
    /// def follow_raise_link(self, graph, link, s_last_exc_value):
    ///     v_last_exc_type = link.last_exception
    ///     v_last_exc_value = link.last_exc_value
    ///
    ///     assert (isinstance(link.exitcase, (types.ClassType, type)) and
    ///             issubclass(link.exitcase, BaseException))
    ///
    ///     assert v_last_exc_type and v_last_exc_value
    ///
    ///     if isinstance(v_last_exc_value, Variable):
    ///         self.setbinding(v_last_exc_value, s_last_exc_value)
    ///
    ///     if isinstance(v_last_exc_type, Variable):
    ///         self.setbinding(v_last_exc_type, typeof([v_last_exc_value]))
    ///
    ///     inputs_s = []
    ///     renaming = defaultdict(list)
    ///     for v_out, v_input in zip(link.args, link.target.inputargs):
    ///         renaming[v_out].append(v_input)
    ///
    ///     for v_out, v_input in zip(link.args, link.target.inputargs):
    ///         if v_out == v_last_exc_type:
    ///             s_out = typeof(renaming[v_last_exc_value])
    ///             if isinstance(v_last_exc_type, Constant):
    ///                 s_out.const = v_last_exc_type.value
    ///             elif v_last_exc_type.annotation.is_constant():
    ///                 s_out.const = v_last_exc_type.annotation.const
    ///             inputs_s.append(s_out)
    ///         else:
    ///             s_out = self.annotation(v_out)
    ///             s_out = self.apply_renaming(s_out, renaming)
    ///             inputs_s.append(s_out)
    ///
    ///     self.links_followed[link] = True
    ///     self.addpendingblock(graph, link.target, inputs_s)
    /// ```
    pub fn follow_raise_link(&self, graph: &GraphRef, link: &LinkRef, s_last_exc_value: SomeValue) {
        use super::model::typeof_vars;

        // Phase 1: mutate link's extravars in place (setbinding). We
        // need &mut access to the Variable slots inside the Link.
        let (target_rc, args_clone, v_last_exc_value_opt, v_last_exc_type_opt): (
            BlockRef,
            Vec<Option<Hlvalue>>,
            Option<Rc<Variable>>,
            Option<Hlvalue>,
        ) = {
            let mut link_mut = link.borrow_mut();
            let target_rc = link_mut
                .target
                .clone()
                .expect("follow_raise_link: link.target must be present");

            // upstream: `assert v_last_exc_type and v_last_exc_value`.
            assert!(
                link_mut.last_exception.is_some(),
                "follow_raise_link: link.last_exception is required"
            );
            assert!(
                link_mut.last_exc_value.is_some(),
                "follow_raise_link: link.last_exc_value is required"
            );

            // if isinstance(v_last_exc_value, Variable): self.setbinding(v_last_exc_value, s_last_exc_value)
            if let Some(Hlvalue::Variable(v)) = link_mut.last_exc_value.as_mut() {
                self.setbinding(v, s_last_exc_value.clone());
            }

            // Capture a snapshot of v_last_exc_value as Rc<Variable>
            // (only the Variable case — the Constant case doesn't enter
            // typeof). typeof_vars needs the updated v (post-setbinding)
            // so we sample after the mutation above.
            let v_last_exc_value_rc: Option<Rc<Variable>> = match link_mut.last_exc_value.as_ref() {
                Some(Hlvalue::Variable(v)) => Some(Rc::new(v.clone())),
                _ => None,
            };

            // if isinstance(v_last_exc_type, Variable): self.setbinding(v_last_exc_type, typeof([v_last_exc_value]))
            if let Some(Hlvalue::Variable(v)) = link_mut.last_exception.as_mut() {
                let args: Vec<Rc<Variable>> = v_last_exc_value_rc
                    .as_ref()
                    .map(|r| vec![Rc::clone(r)])
                    .unwrap_or_default();
                let s_type = typeof_vars(&args);
                self.setbinding(v, s_type);
            }

            // Capture the post-mutation last_exception (used for the
            // v_out == v_last_exc_type equality in phase 2).
            let v_last_exc_type_clone = link_mut.last_exception.clone();

            (
                target_rc,
                link_mut.args.clone(),
                v_last_exc_value_rc,
                v_last_exc_type_clone,
            )
        };

        // Phase 2: build renaming + inputs_s, then addpendingblock.
        let target_inputargs = target_rc.borrow().inputargs.clone();
        let mut renaming: HashMap<Rc<Variable>, Vec<Rc<Variable>>> = HashMap::new();
        for (v_out_opt, v_input) in args_clone.iter().zip(&target_inputargs) {
            let Some(Hlvalue::Variable(vo)) = v_out_opt else {
                continue;
            };
            let Hlvalue::Variable(vi) = v_input else {
                continue;
            };
            renaming
                .entry(Rc::new(vo.clone()))
                .or_default()
                .push(Rc::new(vi.clone()));
        }

        // The exc_type identity for the `v_out == v_last_exc_type` test
        // — only meaningful when v_last_exception is a Variable. Rust's
        // Variable PartialEq is id-based, so direct `==` comparison
        // matches upstream's Python object-identity semantics.
        let exc_type_var: Option<Variable> = v_last_exc_type_opt.as_ref().and_then(|h| match h {
            Hlvalue::Variable(v) => Some(v.clone()),
            _ => None,
        });

        let mut inputs_s: Vec<SomeValue> = Vec::new();
        for v_out_opt in &args_clone {
            let Some(v_out) = v_out_opt else {
                inputs_s.push(SomeValue::Impossible);
                continue;
            };

            // upstream: `if v_out == v_last_exc_type:` — id-based
            // Variable equality (see Variable::PartialEq).
            let is_exc_type = match (v_out, exc_type_var.as_ref()) {
                (Hlvalue::Variable(v), Some(et)) => v == et,
                _ => false,
            };

            if is_exc_type {
                // upstream: s_out = typeof(renaming[v_last_exc_value])
                let renamed: Vec<Rc<Variable>> = v_last_exc_value_opt
                    .as_ref()
                    .and_then(|k| renaming.get(k).cloned())
                    .unwrap_or_default();
                let mut s_out = typeof_vars(&renamed);

                // upstream: `.const` override — Constant case or
                // constant-annotation case.
                match v_last_exc_type_opt.as_ref() {
                    Some(Hlvalue::Constant(c)) => {
                        // `s_out.const = v_last_exc_type.value` — only
                        // meaningful if s_out is a SomeTypeOf.
                        if let SomeValue::TypeOf(t) = &mut s_out {
                            t.base.const_box = Some(c.clone());
                        }
                    }
                    Some(Hlvalue::Variable(v)) => {
                        // `elif v_last_exc_type.annotation.is_constant(): s_out.const = v_last_exc_type.annotation.const`.
                        if let Some(ann) = v.annotation.as_ref() {
                            if let SomeValue::TypeOf(t_prev) = ann.as_ref() {
                                if let Some(c) = t_prev.base.const_box.as_ref() {
                                    if let SomeValue::TypeOf(t) = &mut s_out {
                                        t.base.const_box = Some(c.clone());
                                    }
                                }
                            }
                        }
                    }
                    None => {}
                }
                inputs_s.push(s_out);
            } else {
                let s_out = self.annotation(v_out).unwrap_or(SomeValue::Impossible);
                let s_out = self.apply_renaming(s_out, &renaming);
                inputs_s.push(s_out);
            }
        }

        let lkey = LinkKey::of(link);
        self.links_followed.borrow_mut().insert(lkey);
        self.addpendingblock(graph, &target_rc, &inputs_s);
    }

    /// RPython `reflowfromposition(self, position_key)`
    /// (annrpython.py:338-340).
    ///
    /// ```python
    /// def reflowfromposition(self, position_key):
    ///     graph, block, index = position_key
    ///     self.reflowpendingblock(graph, block)
    /// ```
    ///
    /// `PositionKey` now carries `Weak<FunctionGraph>` + `Weak<Block>`;
    /// if either has been dropped since the position was recorded the
    /// reflow is silently skipped (the block's owner graph is already
    /// gone).
    pub fn reflowfromposition(&self, position_key: &PositionKey) {
        // upstream: `graph, block, index = position_key`
        let Some(graph) = position_key.graph() else {
            return;
        };
        let Some(block) = position_key.block() else {
            return;
        };
        self.reflowpendingblock(&graph, &block);
    }

    /// RPython `bindinputargs(self, graph, block, inputcells)`
    /// (annrpython.py:422-428).
    ///
    /// Creates the initial bindings for the input args of a block:
    /// every `block.inputargs[i]` gets `setbinding(..., inputcells[i])`.
    /// The block is then marked as `annotated[block] = False`
    /// (= awaiting flowin) and registered in `blocked_blocks`.
    pub fn bindinputargs(&self, graph: &GraphRef, block: &BlockRef, inputcells: &[SomeValue]) {
        {
            let mut blk = block.borrow_mut();
            assert_eq!(
                blk.inputargs.len(),
                inputcells.len(),
                "bindinputargs: inputargs/cells arity mismatch"
            );
            for (a, cell) in blk.inputargs.iter_mut().zip(inputcells.iter()) {
                if let Hlvalue::Variable(v) = a {
                    self.setbinding(v, cell.clone());
                }
            }
        }
        let bkey = BlockKey::of(block);
        self.annotated.borrow_mut().insert(bkey.clone(), None);
        self.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), Rc::clone(block));
        self.blocked_blocks
            .borrow_mut()
            .insert(bkey, (Rc::clone(block), Rc::clone(graph), None));
    }

    /// RPython `mergeinputargs(self, graph, block, inputcells)`
    /// (annrpython.py:430-446).
    ///
    /// Widens each of the block's existing input annotations via
    /// `unionof(old, new)`. If any merged cell differs from its old
    /// counterpart, the block is re-seeded via `bindinputargs` so the
    /// widened input triggers a fresh flow pass.
    pub fn mergeinputargs(&self, graph: &GraphRef, block: &BlockRef, inputcells: &[SomeValue]) {
        let blk = block.borrow();
        let oldcells: Vec<SomeValue> = blk
            .inputargs
            .iter()
            .map(|a| match a {
                Hlvalue::Variable(v) => v
                    .annotation
                    .as_ref()
                    .map(|rc| (**rc).clone())
                    .expect("mergeinputargs: inputarg lacks annotation"),
                Hlvalue::Constant(c) => self
                    .bookkeeper
                    .immutableconstant(c)
                    .expect("mergeinputargs: constant immutablevalue failed"),
            })
            .collect();
        drop(blk);

        let unions: Result<Vec<SomeValue>, UnionError> = oldcells
            .iter()
            .zip(inputcells.iter())
            .map(|(c1, c2)| unionof([c1, c2]))
            .collect();

        let unions = match unions {
            Ok(u) => u,
            Err(e) => {
                // Upstream keeps going when `self.keepgoing` is set;
                // otherwise re-raises. Both land cleanly once `errors`
                // carries structured payloads (Commit 7b).
                if self.keepgoing {
                    self.errors.borrow_mut().push(format!("{e}"));
                    self.failed_blocks
                        .borrow_mut()
                        .insert(BlockKey::of(block), Rc::clone(block));
                    return;
                }
                panic!("UnionError in mergeinputargs: {}", e);
            }
        };

        if unions != oldcells {
            self.bindinputargs(graph, block, &unions);
        }
    }

    // ======================================================================
    // RPython `flowing annotations in blocks` (annrpython.py:488-670).
    // ======================================================================

    /// RPython `consider_op(self, op)` (annrpython.py:643-660).
    ///
    /// ```python
    /// def consider_op(self, op):
    ///     for arg in op.args:
    ///         if isinstance(self.annotation(arg), SomeImpossibleValue):
    ///             raise BlockedInference(self, op, -1)
    ///     resultcell = op.consider(self)
    ///     if resultcell is None:
    ///         resultcell = s_ImpossibleValue
    ///     elif resultcell == s_ImpossibleValue:
    ///         raise BlockedInference(self, op, -1)
    ///     assert isinstance(resultcell, annmodel.SomeObject)
    ///     assert isinstance(op.result, Variable)
    ///     self.setbinding(op.result, resultcell)
    /// ```
    ///
    /// Rust returns the result cell (or [`FlowinError`] on block /
    /// harmless swallow / annotator error). The caller owns the block
    /// borrow and is responsible for writing the binding into
    /// `op.result` (which lives behind a `RefCell` on the block).
    pub fn consider_op(
        &self,
        hlop: &super::super::flowspace::operation::HLOperation,
    ) -> Result<SomeValue, FlowinError> {
        // upstream: any arg whose annotation is SomeImpossibleValue
        // means the op cannot run — block.
        for arg in &hlop.args {
            if matches!(self.annotation(arg), Some(SomeValue::Impossible)) {
                return Err(BlockedInference::new(self, hlop.clone(), None).into());
            }
        }
        // upstream `resultcell = op.consider(self)`; AnnotatorError
        // raised inside get_specialization / registered handlers
        // propagates as `FlowinError::Annotator` via the `?`.
        let resultcell = hlop.consider(self)?;
        // upstream: None → s_ImpossibleValue; s_ImpossibleValue → block.
        if matches!(resultcell, SomeValue::Impossible) {
            return Err(BlockedInference::new(self, hlop.clone(), None).into());
        }
        Ok(resultcell)
    }

    /// RPython `get_exception(self, operation)` (annrpython.py:662-670).
    ///
    /// ```python
    /// def get_exception(self, operation):
    ///     """
    ///     Return the annotation for all exceptions that `operation` may raise.
    ///     """
    ///     can_only_throw = operation.get_can_only_throw(self)
    ///     if can_only_throw is None:
    ///         return SomeInstance(self.bookkeeper.getuniqueclassdef(Exception))
    ///     else:
    ///         return self.bookkeeper.new_exception(can_only_throw)
    /// ```
    pub fn get_exception(
        &self,
        hlop: &super::super::flowspace::operation::HLOperation,
    ) -> SomeValue {
        use super::super::flowspace::model::HOST_ENV;
        use super::model::{SomeException, SomeInstance};
        let can_only_throw = hlop.get_can_only_throw(self);
        match can_only_throw {
            // upstream `None` → `SomeInstance(getuniqueclassdef(Exception))`.
            None => {
                let exc_class = HOST_ENV
                    .lookup_exception_class("Exception")
                    .expect("HOST_ENV missing builtin Exception");
                let cdef = self
                    .bookkeeper
                    .getuniqueclassdef(&exc_class)
                    .expect("getuniqueclassdef(Exception) failed");
                SomeValue::Instance(SomeInstance::new(
                    Some(cdef),
                    false,
                    std::collections::BTreeMap::new(),
                ))
            }
            // upstream `self.bookkeeper.new_exception(xs)`.
            Some(xs) => {
                let classes: Vec<_> = xs
                    .iter()
                    .map(|b| {
                        HOST_ENV
                            .lookup_exception_class(b.host_name())
                            .unwrap_or_else(|| panic!("HOST_ENV missing builtin {}", b.host_name()))
                    })
                    .collect();
                let s_exc = self
                    .bookkeeper
                    .new_exception(&classes)
                    .expect("new_exception failed");
                // `SomeException` path — no coercion to SomeInstance;
                // flowin drives intersection/difference on this type.
                let _ = SomeException::new(vec![]); // keep import alive
                SomeValue::Exception(s_exc)
            }
        }
    }

    /// Outcome of the flowin operations-side loop — determines which
    /// exits-side path runs next.
    fn flowin_op_loop(&self, graph: &GraphRef, block: &BlockRef) -> OpLoopOutcome {
        let mut i = 0usize;
        // Re-read the length each iteration because upstream's
        // `new_ops = op.transform(self); block.operations[i:i+1] = new_ops`
        // mutates the list under the loop pointer.
        while i < block.borrow().operations.len() {
            // upstream: `with self.bookkeeper.at_position((graph, block, i)):`
            // Upstream's 3-tuple carries the real graph/block refs; use
            // `from_refs` so `PositionKey::graph()` / `block()` can
            // upgrade them back (reflowfromposition consumes these).
            let pk = super::bookkeeper::PositionKey::from_refs(graph, block, i);
            let _pg = self.bookkeeper.at_position(Some(pk));

            // Reify `SpaceOperation` (Block storage) into an
            // `HLOperation` the dispatcher can consume. Upstream
            // `Block.operations` carries HLOperations directly; the
            // Rust port's flowspace stores the lowered SpaceOperation
            // shape, so we rebuild the HLOperation for the dispatcher
            // at each visit.
            let hlop = {
                let blk = block.borrow();
                let sp = &blk.operations[i];
                let Some(kind) =
                    super::super::flowspace::operation::OpKind::from_opname(&sp.opname)
                else {
                    // Unknown op → skip (upstream never reaches this
                    // branch; the RPython flow object space only emits
                    // ops whose opname is registered).
                    i += 1;
                    continue;
                };
                let args = sp.args.clone();
                let result_var = match &sp.result {
                    Hlvalue::Variable(v) => v.clone(),
                    Hlvalue::Constant(_) => {
                        // upstream: `assert isinstance(op.result, Variable)`.
                        panic!(
                            "flowin: op.result must be Variable (got Constant) for {:?}",
                            sp.opname
                        );
                    }
                };
                super::super::flowspace::operation::HLOperation {
                    kind,
                    args,
                    result: result_var,
                    offset: sp.offset,
                }
            };

            // upstream: `new_ops = op.transform(self)` + block rewrite.
            //
            // ```python
            // new_ops = op.transform(self)
            // if new_ops is not None:
            //     block.operations[i:i+1] = new_ops
            //     if not new_ops:
            //         continue
            //     new_ops[-1].result = op.result
            //     op = new_ops[0]
            // ```
            //
            // `HLOperation.transform` currently returns `None` for every
            // op (empty transform registry) so the branch below is dead
            // until optimizer registrations land — but the control-flow
            // mirror is in place so behaviour tracks upstream once they
            // do.
            let hlop = match hlop.transform(self) {
                None => hlop,
                Some(new_ops) => {
                    let op_result = hlop.result.clone();
                    let new_sps: Vec<super::super::flowspace::model::SpaceOperation> = new_ops
                        .iter()
                        .map(|new_hlop| {
                            super::super::flowspace::model::SpaceOperation::with_offset(
                                new_hlop.kind.opname(),
                                new_hlop.args.clone(),
                                Hlvalue::Variable(new_hlop.result.clone()),
                                new_hlop.offset,
                            )
                        })
                        .collect();
                    {
                        let mut blk = block.borrow_mut();
                        blk.operations.splice(i..=i, new_sps);
                    }
                    // `if not new_ops: continue` — empty replacement
                    // consumes the op.
                    if new_ops.is_empty() {
                        continue;
                    }
                    // `new_ops[-1].result = op.result` — only the last
                    // replacement inherits the original result.
                    {
                        let mut blk = block.borrow_mut();
                        let last_idx = i + new_ops.len() - 1;
                        blk.operations[last_idx].result = Hlvalue::Variable(op_result);
                    }
                    // `op = new_ops[0]`.
                    new_ops[0].clone()
                }
            };

            // Hand off to consider_op.
            let result_cell = match self.consider_op(&hlop) {
                Ok(sv) => sv,
                Err(FlowinError::Blocked(mut e)) => {
                    // upstream flowin:504-526 — BlockedInference handling.
                    // Decide whether the blocked op is the block's
                    // raising_op (in which case exits-side proceeds with
                    // only exception exits) or a harmless call-family
                    // op (swallow) or else propagate.
                    let is_raising_op = {
                        let blk = block.borrow();
                        blk.canraise() && i + 1 == blk.operations.len()
                    };
                    if is_raising_op {
                        e.opindex = Some(i);
                        return OpLoopOutcome::BlockedOnRaisingOp;
                    }
                    match hlop.kind {
                        super::super::flowspace::operation::OpKind::SimpleCall
                        | super::super::flowspace::operation::OpKind::CallArgs
                        | super::super::flowspace::operation::OpKind::Next => {
                            return OpLoopOutcome::HarmlesslySwallowed;
                        }
                        _ => {
                            e.opindex = Some(i);
                            return OpLoopOutcome::Blocked(e);
                        }
                    }
                }
                Err(FlowinError::Harmless(_)) => {
                    // upstream: `except annmodel.HarmlesslyBlocked: return`.
                    return OpLoopOutcome::HarmlesslySwallowed;
                }
                Err(FlowinError::Annotator(mut e)) => {
                    // upstream annrpython.py:531-537:
                    //     except annmodel.AnnotatorError as e:
                    //         e.source = gather_error(self, graph, block, i)
                    //         if self.keepgoing:
                    //             self.errors.append(e)
                    //             self.failed_blocks.add(block)
                    //             return
                    //         raise
                    e.source = Some(gather_error(self, graph, block, i));
                    if self.keepgoing {
                        self.errors.borrow_mut().push(format!("{e}"));
                        self.failed_blocks
                            .borrow_mut()
                            .insert(BlockKey::of(block), Rc::clone(block));
                        return OpLoopOutcome::HarmlesslySwallowed;
                    }
                    // upstream `raise` — no keepgoing safety net. Panic
                    // with a structured message until the outer
                    // AnnotatorError propagation lands.
                    panic!("AnnotatorError (keepgoing off): {e}");
                }
            };

            // upstream: `self.setbinding(op.result, resultcell)`. The
            // Variable we captured above is a snapshot; mutate the
            // live slot inside block.operations[i].result.
            {
                let mut blk = block.borrow_mut();
                if let Hlvalue::Variable(v) = &mut blk.operations[i].result {
                    self.setbinding(v, result_cell);
                }
            }
            i += 1;
        }

        OpLoopOutcome::Normal
    }

    /// RPython `flowin(self, graph, block)` (annrpython.py:488-576).
    ///
    /// Port structure:
    /// 1. Run the op loop via [`flowin_op_loop`]. Its outcome decides
    ///    which `exits` list the exits-side dispatch uses.
    /// 2. Exits-side dispatch (annrpython.py:539-572):
    ///    - `block.canraise` → `get_exception` + per-link intersection /
    ///      difference with `follow_raise_link` for matches and the
    ///      `None` exitcase routed through `follow_link`.
    ///    - Otherwise → knowntypedata-driven `follow_link` per exit.
    /// 3. Notify reflow (annrpython.py:574-576): any position subscribed
    ///    to this block's updates is re-queued.
    fn flowin(&self, graph: &GraphRef, block: &BlockRef) -> Result<(), BlockedInference> {
        use super::super::flowspace::model::LinkRef;
        use super::model::{
            KnownTypeData, SomeException, SomeInstance, SomeObjectTrait, difference, intersection,
            s_impossible_value,
        };

        // Phase 1 — op loop.
        let outcome = self.flowin_op_loop(graph, block);

        // Phase 2 — determine `exits`.
        let exits: Vec<LinkRef> = match outcome {
            OpLoopOutcome::Normal => {
                // upstream else: dead-branch pruning.
                let mut exits = block.borrow().exits.clone();
                // `if isinstance(block.exitswitch, Variable):`
                let exitswitch = block.borrow().exitswitch.clone();
                if let Some(Hlvalue::Variable(_)) = &exitswitch {
                    let s_exitswitch = self.binding(exitswitch.as_ref().unwrap());
                    if s_exitswitch.is_constant() {
                        // upstream: filter by `link.exitcase == s.const`.
                        let s_const = s_exitswitch
                            .const_()
                            .expect("is_constant() but const_() empty")
                            .clone();
                        exits.retain(|link| {
                            let link_b = link.borrow();
                            match &link_b.exitcase {
                                Some(Hlvalue::Constant(c)) => c.value == s_const,
                                Some(Hlvalue::Variable(_)) => false,
                                None => false,
                            }
                        });
                    }
                }
                exits
            }
            OpLoopOutcome::BlockedOnRaisingOp => {
                // upstream: `exits = [link for link in block.exits
                //                     if link.exitcase is not None]`.
                let all = block.borrow().exits.clone();
                all.into_iter()
                    .filter(|l| l.borrow().exitcase.is_some())
                    .collect()
            }
            OpLoopOutcome::HarmlesslySwallowed => {
                // upstream `return` — skip exits & notify reflow.
                return Ok(());
            }
            OpLoopOutcome::Blocked(e) => {
                return Err(e);
            }
        };

        // Phase 3 — exits-side dispatch.
        let canraise = block.borrow().canraise();
        if canraise {
            // upstream: `op = block.raising_op; s_exception = self.get_exception(op)`.
            // Rebuild the raising_op HLOperation from the last
            // SpaceOperation.
            let hlop = {
                let blk = block.borrow();
                let raising = blk
                    .raising_op()
                    .expect("canraise block missing raising_op")
                    .clone();
                let Some(kind) =
                    super::super::flowspace::operation::OpKind::from_opname(&raising.opname)
                else {
                    panic!("flowin: raising_op opname unknown: {:?}", raising.opname);
                };
                let result_var = match &raising.result {
                    Hlvalue::Variable(v) => v.clone(),
                    Hlvalue::Constant(_) => panic!("flowin: raising_op result must be Variable"),
                };
                super::super::flowspace::operation::HLOperation {
                    kind,
                    args: raising.args.clone(),
                    result: result_var,
                    offset: raising.offset,
                }
            };
            let mut s_exception = self.get_exception(&hlop);

            for link in &exits {
                let exitcase = link.borrow().exitcase.clone();
                match exitcase {
                    None => {
                        self.follow_link(graph, link, &HashMap::new());
                    }
                    Some(case_hlv) => {
                        // upstream: `if s_exception == s_ImpossibleValue: break`.
                        if s_exception == s_impossible_value() {
                            break;
                        }
                        // upstream: `s_case = SomeInstance(self.bookkeeper.getuniqueclassdef(case))`.
                        let case_host = match &case_hlv {
                            Hlvalue::Constant(c) => match &c.value {
                                super::super::flowspace::model::ConstValue::HostObject(obj) => {
                                    obj.clone()
                                }
                                other => panic!(
                                    "flowin: canraise exitcase must be a HostObject class \
                                         Constant, got {:?}",
                                    other
                                ),
                            },
                            Hlvalue::Variable(_) => {
                                panic!("flowin: canraise link.exitcase must be a Constant")
                            }
                        };
                        let case_cdef = self
                            .bookkeeper
                            .getuniqueclassdef(&case_host)
                            .expect("getuniqueclassdef for exit case failed");
                        let s_case = SomeValue::Instance(SomeInstance::new(
                            Some(case_cdef),
                            false,
                            std::collections::BTreeMap::new(),
                        ));
                        // upstream: `s_matching_exc = intersection(s_exception, s_case)`.
                        let s_matching_exc = intersection(&s_exception, &s_case);
                        if s_matching_exc != s_impossible_value() {
                            self.follow_raise_link(graph, link, s_matching_exc);
                        }
                        // upstream: `s_exception = difference(s_exception, s_case)`.
                        s_exception = difference(&s_exception, &s_case);
                    }
                }
            }
            // keep imports alive (SomeException used indirectly in
            // get_exception).
            let _ = SomeException::new(vec![]);
        } else {
            // upstream: knowntypedata-driven constraints per exit.
            let exitswitch = block.borrow().exitswitch.clone();
            let knowntypedata: KnownTypeData = match exitswitch {
                Some(Hlvalue::Variable(ref v)) => v
                    .annotation
                    .as_ref()
                    .and_then(|rc| match rc.as_ref() {
                        SomeValue::Bool(b) => b.knowntypedata.clone(),
                        _ => None,
                    })
                    .unwrap_or_default(),
                _ => HashMap::new(),
            };
            for link in &exits {
                let exitcase = link.borrow().exitcase.clone();
                // `knowntypedata.get(link.exitcase, {})` — the outer
                // key is the python-level exitcase truth value. For
                // bool-discriminating exits upstream uses True/False
                // directly; we extract the boolean from the exitcase
                // constant.
                let truth: Option<bool> = match exitcase {
                    Some(Hlvalue::Constant(c)) => match &c.value {
                        super::super::flowspace::model::ConstValue::Bool(b) => Some(*b),
                        super::super::flowspace::model::ConstValue::Int(i) => Some(*i != 0),
                        _ => None,
                    },
                    _ => None,
                };
                let constraints = truth
                    .and_then(|t| knowntypedata.get(&t).cloned())
                    .unwrap_or_default();
                self.follow_link(graph, link, &constraints);
            }
        }

        // Phase 4 — notify reflow.
        // upstream: `if block in self.notify: for position in
        // self.notify[block]: self.reflowfromposition(position)`.
        let positions: Vec<PositionKey> = {
            let bkey = BlockKey::of(block);
            self.notify
                .borrow()
                .get(&bkey)
                .cloned()
                .map(|set| set.into_iter().collect())
                .unwrap_or_default()
        };
        for position in positions {
            // upstream: `self.reflowfromposition(position)`
            self.reflowfromposition(&position);
        }

        Ok(())
    }

    /// RPython `processblock(self, graph, block)` (annrpython.py:378-412).
    ///
    /// ```python
    /// self.annotated[block] = graph
    /// if block in self.failed_blocks:
    ///     return
    /// if block in self.blocked_blocks:
    ///     del self.blocked_blocks[block]
    /// try:
    ///     self.flowin(graph, block)
    /// except BlockedInference as e:
    ///     self.annotated[block] = False
    ///     self.blocked_blocks[block] = (graph, e.opindex)
    /// except Exception as e:
    ///     ...
    ///     raise
    /// if self.added_blocks is not None:
    ///     self.added_blocks[block] = True
    /// ```
    ///
    /// The `flowin` call lands with Commit 8; until then the Rust
    /// port's body contains the bookkeeping (annotated flip, blocked
    /// cleanup, added_blocks tracking) that empty blocks already need.
    /// `BlockedInference` handling is also staged for Commit 8 when
    /// `flowin` actually throws.
    pub fn processblock(&self, graph: &GraphRef, block: &BlockRef) {
        let bkey = BlockKey::of(block);
        // upstream: `self.annotated[block] = graph`.
        self.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(Rc::clone(graph)));
        self.all_blocks
            .borrow_mut()
            .insert(bkey.clone(), Rc::clone(block));
        // upstream: `if block in self.failed_blocks: return`.
        if self.failed_blocks.borrow().contains_key(&bkey) {
            return;
        }
        // upstream: `if block in self.blocked_blocks: del ...`.
        self.blocked_blocks.borrow_mut().remove(&bkey);

        // upstream annrpython.py:397-406:
        //     try:
        //         self.flowin(graph, block)
        //     except BlockedInference as e:
        //         self.annotated[block] = False
        //         self.blocked_blocks[block] = (graph, e.opindex)
        //     except Exception as e:
        //         ...
        //         raise
        match self.flowin(graph, block) {
            Ok(()) => {}
            Err(e) => {
                self.annotated.borrow_mut().insert(bkey.clone(), None);
                self.blocked_blocks.borrow_mut().insert(
                    bkey.clone(),
                    (Rc::clone(block), Rc::clone(graph), e.opindex),
                );
            }
        }

        // upstream: `if self.added_blocks is not None:
        //              self.added_blocks[block] = True`.
        if let Some(added) = self.added_blocks.borrow_mut().as_mut() {
            added.insert(bkey, Rc::clone(block));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::model::{Block, FunctionGraph};
    use super::super::model::SomeInteger;
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    fn mk_graph(name: &str, n_args: usize) -> GraphRef {
        let inputs: Vec<Hlvalue> = (0..n_args)
            .map(|i| Hlvalue::Variable(Variable::named(format!("a{i}"))))
            .collect();
        let startblock = Block::shared(inputs);
        Rc::new(RefCell::new(FunctionGraph::new(name, startblock)))
    }

    #[test]
    fn addpendingblock_seeds_annotated_and_queues_block() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("f", 1);
        let startblock = graph.borrow().startblock.clone();
        let s_int = SomeValue::Integer(SomeInteger::default());
        ann.addpendinggraph(&graph, &[s_int.clone()]);

        // bindinputargs side effect: block appears in `annotated` as
        // `Some(None)` (= False, awaiting flowin).
        let bkey = BlockKey::of(&startblock);
        let annmap = ann.annotated.borrow();
        assert!(annmap.contains_key(&bkey));
        assert!(matches!(annmap.get(&bkey), Some(None)));
        drop(annmap);

        // schedulependingblock side effect: generation-0 dict now
        // carries (block, graph).
        let pending = ann.genpendingblocks.borrow();
        assert_eq!(pending[0].len(), 1);
        assert!(pending[0].contains_key(&bkey));

        // setbinding side effect: the startblock inputarg now has the
        // SomeInteger annotation.
        let bound = {
            let blk = startblock.borrow();
            match &blk.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        assert_eq!(bound, Some(s_int));
    }

    #[test]
    fn mergeinputargs_rebinds_when_union_widens() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("g", 1);
        let startblock = graph.borrow().startblock.clone();

        ann.addpendinggraph(&graph, &[SomeValue::Integer(SomeInteger::default())]);
        // Re-seed with SomeFloat — unionof(Int, Float) = Float, which
        // differs from the old cell, so bindinputargs must re-run.
        ann.addpendingblock(
            &graph,
            &startblock,
            &[SomeValue::Float(super::super::model::SomeFloat::new())],
        );

        let bound = {
            let blk = startblock.borrow();
            match &blk.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        assert!(matches!(bound, Some(SomeValue::Float(_))));
    }

    #[test]
    fn gettype_variable_returns_knowntype_or_object() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let mut v = Variable::named("x");
        assert_eq!(
            ann.gettype(&Hlvalue::Variable(v.clone())),
            super::super::model::KnownType::Object
        );
        ann.setbinding(&mut v, SomeValue::Integer(SomeInteger::default()));
        assert_eq!(
            ann.gettype(&Hlvalue::Variable(v)),
            super::super::model::KnownType::Int
        );
    }

    #[test]
    fn validate_calls_bookkeeper_hook() {
        // No-op stub — just verify it's wired up and callable.
        let ann = RPythonAnnotator::new(None, None, None, false);
        ann.validate();
    }

    #[test]
    fn simplify_invokes_bookkeeper_fixpoint() {
        // No-op stub — exercise the call site with the upstream
        // `block_subset=None, extra_passes=None` shape.
        let ann = RPythonAnnotator::new(None, None, None, false);
        ann.simplify(None, None);
    }

    #[test]
    fn whereami_formats_position_key() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let pk = PositionKey::new(42, 17, 3);
        let s = ann.whereami(pk);
        assert!(s.contains("graph=42"));
        assert!(s.contains("block=17"));
        assert!(s.contains("op=3"));
    }

    #[test]
    fn using_policy_restores_on_drop() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        // Baseline: a default policy. Swap in an alternate policy
        // inside a scope; after the scope ends the default is back.
        let alt = AnnotatorPolicy::new();
        {
            let _g = ann.using_policy(alt);
            // no way to compare policies directly — exercise the Drop.
        }
        // Sanity — another policy swap still works after the first
        // guard has been dropped.
        let _g2 = ann.using_policy(AnnotatorPolicy::new());
        drop(_g2);
    }

    #[test]
    fn build_graph_types_zero_arg_calls_complete() {
        // Minimum valid graph (mirrors upstream test_mingraph): no
        // inputs, startblock closes directly to returnblock with
        // `Constant(1)`. `complete()` binds `returnvar` to
        // `SomeValue::Integer` — this test just verifies the
        // build_graph_types → checkgraph → complete wiring.
        use super::super::super::flowspace::model::{BlockRefExt, ConstValue, Constant, Link};
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let graph = mk_graph("f0", 0);
        {
            let g = graph.borrow();
            let link = Rc::new(RefCell::new(Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::Int(1)))],
                Some(g.returnblock.clone()),
                None,
            )));
            g.startblock.closeblock(vec![link]);
        }
        translator.graphs.borrow_mut().push(Rc::clone(&graph));
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let r = ann.build_graph_types(&graph, &[], true);
        assert!(matches!(r, Some(SomeValue::Integer(_))), "got {:?}", r);
    }

    #[test]
    fn flowin_empty_block_is_noop() {
        // No operations → flowin returns Ok immediately; processblock
        // flips annotated[block] to Some(graph).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("empty", 0);
        let startblock = graph.borrow().startblock.clone();
        ann.processblock(&graph, &startblock);
        assert!(matches!(
            ann.annotated.borrow().get(&BlockKey::of(&startblock)),
            Some(Some(_))
        ));
    }

    #[test]
    fn flowin_consider_op_binds_result() {
        // A single Pos op on a bound Integer should produce a
        // SomeInteger binding on the op.result.
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("posgraph", 1);
        let startblock = graph.borrow().startblock.clone();

        // Seed the input variable with SomeInteger.
        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
            // Construct `pos(a)` — result variable captures the op output.
            let arg = blk.inputargs[0].clone();
            let result = Hlvalue::Variable(Variable::named("r"));
            blk.operations
                .push(super::super::super::flowspace::model::SpaceOperation::new(
                    "pos",
                    vec![arg],
                    result,
                ));
        }

        ann.processblock(&graph, &startblock);
        // After flowin, block.operations[0].result should carry an
        // Integer binding (SomeInteger.pos on SomeInteger = SomeInteger).
        let blk = startblock.borrow();
        if let Hlvalue::Variable(v) = &blk.operations[0].result {
            let sv = v.annotation.as_ref().map(|rc| (**rc).clone());
            assert!(matches!(sv, Some(SomeValue::Integer(_))), "got {:?}", sv);
        } else {
            panic!("op.result is not Variable");
        }
        // annotated[block] == Some(graph).
        assert!(matches!(
            ann.annotated.borrow().get(&BlockKey::of(&startblock)),
            Some(Some(_))
        ));
        // Not queued as blocked.
        assert!(
            !ann.blocked_blocks
                .borrow()
                .contains_key(&BlockKey::of(&startblock))
        );
    }

    #[test]
    fn follow_link_seeds_target_block_with_renamed_args() {
        // Build: source block ---link(v0 -> u0)---> target block.
        // v0 bound SomeInteger; after follow_link, u0 (target's
        // inputarg) should carry SomeInteger via addpendingblock.
        use super::super::super::flowspace::model::{Block, Link};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("link_test", 1);
        // Create a target block with one inputarg u0.
        let u0 = Hlvalue::Variable(Variable::named("u0"));
        let target = Block::shared(vec![u0.clone()]);

        // The source block has the existing startblock with inputarg
        // a0 = Variable v0; bind it to SomeInteger.
        let source = graph.borrow().startblock.clone();
        {
            let mut src = source.borrow_mut();
            if let Hlvalue::Variable(v) = &mut src.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
        }
        let source_a0 = source.borrow().inputargs[0].clone();

        // Construct the link: source.exits = [link(args=[a0], target)].
        let link = Rc::new(RefCell::new(Link::new(
            vec![source_a0],
            Some(target.clone()),
            None,
        )));

        // Drive follow_link with empty constraints.
        ann.follow_link(&graph, &link, &HashMap::new());

        // links_followed should record this link.
        assert!(ann.links_followed.borrow().contains(&LinkKey::of(&link)));
        // target.inputargs[0].annotation should be SomeInteger.
        let bound = {
            let t = target.borrow();
            match &t.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        assert!(
            matches!(bound, Some(SomeValue::Integer(_))),
            "got {:?}",
            bound
        );
    }

    #[test]
    fn follow_link_applies_bool_true_constraint() {
        // source's a0 bound to SomeBool with an unknown value.
        // constraint map says "if this link is taken, a0 is True".
        // improve(SomeBool, s_True) returns s_True (contained). Target
        // inputarg receives the refined SomeBool(const=True) binding.
        use super::super::super::flowspace::model::{Block, Link};
        use super::super::model::{SomeBool, s_true};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("bool_cstrt", 1);
        let u0 = Hlvalue::Variable(Variable::named("u0"));
        let target = Block::shared(vec![u0]);

        let source = graph.borrow().startblock.clone();
        {
            let mut src = source.borrow_mut();
            if let Hlvalue::Variable(v) = &mut src.inputargs[0] {
                ann.setbinding(v, SomeValue::Bool(SomeBool::new()));
            }
        }
        let source_a0 = source.borrow().inputargs[0].clone();
        let link = Rc::new(RefCell::new(Link::new(
            vec![source_a0.clone()],
            Some(target.clone()),
            None,
        )));

        let mut constraints: HashMap<Rc<Variable>, SomeValue> = HashMap::new();
        if let Hlvalue::Variable(v) = &source_a0 {
            constraints.insert(Rc::new(v.clone()), s_true());
        }
        ann.follow_link(&graph, &link, &constraints);

        // target.inputargs[0] should carry the refined bool (const=True).
        let bound = {
            let t = target.borrow();
            match &t.inputargs[0] {
                Hlvalue::Variable(v) => v.annotation.as_ref().map(|rc| (**rc).clone()),
                _ => None,
            }
        };
        match bound {
            Some(SomeValue::Bool(b)) => {
                assert!(b.base.const_box.is_some(), "expected constant bool");
            }
            other => panic!("expected SomeBool(const=True), got {:?}", other),
        }
    }

    #[test]
    fn recursivecall_queues_callee_and_wires_notify() {
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let caller = mk_graph("caller", 0);
        let callee = mk_graph("callee", 1);
        let caller_startblock = caller.borrow().startblock.clone();
        let whence = Some((Rc::clone(&caller), Rc::clone(&caller_startblock), 0));

        let r = ann.recursivecall(
            &callee,
            whence,
            &[SomeValue::Integer(SomeInteger::default())],
        );
        // The callee's return var has no binding yet → s_ImpossibleValue.
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
        // notify[callee.returnblock] should carry the caller's
        // position key.
        let returnblock = callee.borrow().returnblock.clone();
        let notify = ann.notify.borrow();
        let positions = notify
            .get(&BlockKey::of(&returnblock))
            .expect("notify entry missing");
        assert_eq!(positions.len(), 1);
    }

    #[test]
    fn complete_on_empty_schedule_is_noop() {
        // No pending blocks, no translator — complete() should fall
        // through the "all empty" branch and exit cleanly.
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        ann.complete();
    }

    #[test]
    #[should_panic(expected = "complete: added_blocks entry missing from annotated")]
    fn complete_rejects_added_blocks_entries_missing_from_annotated() {
        let translator = super::super::super::translator::translator::TranslationContext::new();
        let ann = RPythonAnnotator::new(Some(translator), None, None, false);
        let graph = mk_graph("broken", 0);
        let block = graph.borrow().startblock.clone();

        ann.added_blocks
            .borrow_mut()
            .replace(HashMap::from([(BlockKey::of(&block), block)]));

        ann.complete();
    }

    #[test]
    fn reflowpendingblock_marks_block_as_needing_reflow() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("h", 1);
        let startblock = graph.borrow().startblock.clone();
        ann.addpendinggraph(&graph, &[SomeValue::Integer(SomeInteger::default())]);

        // Simulate "block was annotated by processblock" by setting
        // annotated[block] = Some(graph).
        let bkey = BlockKey::of(&startblock);
        ann.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(Rc::clone(&graph)));

        ann.reflowpendingblock(&graph, &startblock);

        // Now annotated[block] is None (= False) and the block is
        // re-queued in genpendingblocks and blocked_blocks.
        let annmap = ann.annotated.borrow();
        assert!(matches!(annmap.get(&bkey), Some(None)));
        drop(annmap);
        let pending = ann.genpendingblocks.borrow();
        assert!(pending[0].contains_key(&bkey));
        drop(pending);
        let blocked = ann.blocked_blocks.borrow();
        assert!(blocked.contains_key(&bkey));
    }

    // ------------------------------------------------------------------
    // Dispatch::None consider() — upstream operation.py:534-565.
    // ------------------------------------------------------------------

    #[test]
    fn consider_newtuple_builds_sometuple_from_args() {
        // NewTuple.consider returns SomeTuple(items=[annotator.annotation(a)
        // for a in args]).  Bind two args to SomeInteger / s_str0 and
        // confirm the Tuple carries both.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        use super::super::model::s_str0;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("nt", 2);
        let startblock = graph.borrow().startblock.clone();
        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
            if let Hlvalue::Variable(v) = &mut blk.inputargs[1] {
                ann.setbinding(v, s_str0());
            }
        }
        let blk = startblock.borrow();
        let args = blk.inputargs.clone();
        drop(blk);
        let hlop = HLOperation::new(OpKind::NewTuple, args);
        let got = hlop.consider(&ann).unwrap();
        if let SomeValue::Tuple(t) = got {
            assert_eq!(t.items.len(), 2);
            assert!(matches!(t.items[0], SomeValue::Integer(_)));
            assert!(matches!(t.items[1], SomeValue::String(_)));
        } else {
            panic!("expected SomeValue::Tuple, got {:?}", got);
        }
    }

    #[test]
    fn consider_newlist_routes_through_bookkeeper_newlist() {
        // NewList.consider defers to bookkeeper.newlist(*args_s); with
        // a single SomeInteger arg the returned ListDef.s_value must be
        // SomeInteger.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("nl", 1);
        let startblock = graph.borrow().startblock.clone();
        {
            let mut blk = startblock.borrow_mut();
            if let Hlvalue::Variable(v) = &mut blk.inputargs[0] {
                ann.setbinding(v, SomeValue::Integer(SomeInteger::default()));
            }
        }
        let args = startblock.borrow().inputargs.clone();
        let hlop = HLOperation::new(OpKind::NewList, args);
        let got = hlop.consider(&ann).unwrap();
        if let SomeValue::List(list) = got {
            assert!(matches!(list.listdef.s_value(), SomeValue::Integer(_)));
        } else {
            panic!("expected SomeValue::List, got {:?}", got);
        }
    }

    #[test]
    fn consider_newdict_returns_empty_somedict() {
        // NewDict.consider returns bookkeeper.newdict(), an empty dict
        // whose s_key/s_value start as Impossible (DictDef default).
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let hlop = HLOperation::new(OpKind::NewDict, vec![]);
        let got = hlop.consider(&ann).unwrap();
        assert!(matches!(got, SomeValue::Dict(_)));
    }

    #[test]
    fn consider_newslice_raises_annotator_error() {
        // NewSlice.consider always raises AnnotatorError.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let hlop = HLOperation::new(OpKind::NewSlice, vec![]);
        let err = hlop.consider(&ann).unwrap_err();
        assert!(err.msg.unwrap_or_default().contains("extended slicing"));
    }

    // ------------------------------------------------------------------
    // gather_error / keepgoing — upstream annrpython.py:531-537.
    // ------------------------------------------------------------------

    #[test]
    fn gather_error_returns_location_string() {
        // gather_error formats graph/block/op provenance per upstream
        // error.py:67-82. The Rust wrapper now follows upstream's
        // valid-index contract; when the graph has no source metadata,
        // the formatter must still include the graph provenance and
        // the `no source!` marker.
        use super::super::super::flowspace::model::SpaceOperation;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = mk_graph("gather", 0);
        let startblock = graph.borrow().startblock.clone();
        startblock.borrow_mut().operations.push(SpaceOperation::new(
            "newtuple",
            Vec::new(),
            Hlvalue::Variable(Variable::new()),
        ));
        let msg = gather_error(&ann, &graph, &startblock, 0);
        assert!(
            msg.contains("In <FunctionGraph of gather at 0x"),
            "got {msg:?}"
        );
        assert!(msg.contains("no source!"), "got {msg:?}");
    }

    #[test]
    fn position_key_from_refs_upgrades_graph_and_block() {
        // Priority #4: PositionKey carries Weak refs to
        // FunctionGraph / Block so `reflowfromposition` can recover
        // them. The synthetic `PositionKey::new` path leaves the refs
        // dangling.
        let graph = mk_graph("pkrefs", 0);
        let startblock = graph.borrow().startblock.clone();
        let pk = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 7);
        let g = pk.graph().expect("graph weak ref should upgrade");
        assert!(Rc::ptr_eq(&g, &graph));
        let b = pk.block().expect("block weak ref should upgrade");
        assert!(Rc::ptr_eq(&b, &startblock));
        assert_eq!(pk.op_index, 7);

        // Synthetic constructor leaves refs dangling.
        let pk2 = super::super::bookkeeper::PositionKey::new(1, 2, 3);
        assert!(pk2.graph().is_none());
        assert!(pk2.block().is_none());
    }

    #[test]
    fn position_key_equality_on_identity_hashes() {
        // Two PositionKey values from the same Rc<FunctionGraph> /
        // Rc<Block> must compare equal regardless of how they were
        // constructed — matches Python tuple identity via `id()`.
        let graph = mk_graph("eq", 0);
        let startblock = graph.borrow().startblock.clone();
        let pk_a = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 0);
        let pk_b = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 0);
        assert_eq!(pk_a, pk_b);
        // Different op_index → not equal.
        let pk_c = super::super::bookkeeper::PositionKey::from_refs(&graph, &startblock, 1);
        assert_ne!(pk_a, pk_c);
    }

    #[test]
    fn flowin_error_variants_convert_from() {
        // From impls on FlowinError matter because consider_op uses
        // `?` on all three upstream exception classes.
        use super::super::super::flowspace::operation::{HLOperation, OpKind};
        use super::super::model::{AnnotatorError, HarmlesslyBlocked};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let hlop = HLOperation::new(OpKind::NewDict, vec![]);
        let blocked: FlowinError = BlockedInference::new(&ann, hlop, None).into();
        assert!(matches!(blocked, FlowinError::Blocked(_)));
        let harmless: FlowinError = HarmlesslyBlocked.into();
        assert!(matches!(harmless, FlowinError::Harmless(_)));
        let annerr: FlowinError = AnnotatorError::new("oops").into();
        assert!(matches!(annerr, FlowinError::Annotator(_)));
    }
}
