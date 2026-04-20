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

use super::super::flowspace::model::{Hlvalue, Variable};
use super::bookkeeper::Bookkeeper;
use super::model::{SomeValue, TLS};
use super::policy::AnnotatorPolicy;
use crate::translator::translator::TranslationContext;

/// RPython `class RPythonAnnotator(object)` (annrpython.py:22).
///
/// "Block annotator for RPython."
pub struct RPythonAnnotator {
    /// RPython `self.translator` (annrpython.py:35). Held as an
    /// `Option` because the full `TranslationContext` port has not
    /// landed; `build_types` will create a stub when `None`.
    pub translator: RefCell<Option<TranslationContext>>,
    /// RPython `self.genpendingblocks = [{}]` (annrpython.py:36).
    /// A per-generation list of `{block: graph-containing-it}`.
    pub genpendingblocks: RefCell<Vec<HashMap<BlockKey, GraphKey>>>,
    /// RPython `self.annotated = {}` (annrpython.py:37) — set of
    /// blocks already seen.
    pub annotated: RefCell<HashSet<BlockKey>>,
    /// RPython `self.added_blocks = None` (annrpython.py:38) — see
    /// `processblock()`.
    pub added_blocks: RefCell<Option<HashSet<BlockKey>>>,
    /// RPython `self.links_followed = {}` (annrpython.py:39).
    pub links_followed: RefCell<HashSet<LinkKey>>,
    /// RPython `self.notify = {}` (annrpython.py:40).
    pub notify: RefCell<HashMap<BlockKey, HashSet<PositionKey>>>,
    /// RPython `self.fixed_graphs = {}` (annrpython.py:41).
    pub fixed_graphs: RefCell<HashSet<GraphKey>>,
    /// RPython `self.blocked_blocks = {}` (annrpython.py:42).
    pub blocked_blocks: RefCell<HashMap<BlockKey, (GraphKey, usize)>>,
    /// RPython `self.blocked_graphs = {}` (annrpython.py:44).
    pub blocked_graphs: RefCell<HashSet<GraphKey>>,
    /// RPython `self.frozen = False` (annrpython.py:46).
    pub frozen: RefCell<bool>,
    /// RPython `self.policy` (annrpython.py:47-51).
    pub policy: RefCell<AnnotatorPolicy>,
    /// RPython `self.bookkeeper` (annrpython.py:52-54).
    pub bookkeeper: Rc<Bookkeeper>,
    /// RPython `self.keepgoing` (annrpython.py:55).
    pub keepgoing: bool,
    /// RPython `self.failed_blocks = set()` (annrpython.py:56).
    pub failed_blocks: RefCell<HashSet<BlockKey>>,
    /// RPython `self.errors = []` (annrpython.py:57).
    pub errors: RefCell<Vec<String>>,
}

/// Placeholder identifier for flowspace `Block` objects in driver
/// tables. The real `FunctionGraph.blocks` iterator carries `Block`
/// pointers; until the pendingblocks queue is wired up, this is a
/// plain integer handle.
pub type BlockKey = u64;

/// Placeholder identifier for `FunctionGraph` objects.
pub type GraphKey = u64;

/// Placeholder identifier for `Link` objects.
pub type LinkKey = u64;

/// Placeholder position key — mirrors the tuple upstream uses as the
/// `position_key` payload passed to `Bookkeeper.at_position`.
pub type PositionKey = super::bookkeeper::PositionKey;

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
        Rc::new(RPythonAnnotator {
            translator: RefCell::new(translator),
            genpendingblocks: RefCell::new(vec![HashMap::new()]),
            annotated: RefCell::new(HashSet::new()),
            added_blocks: RefCell::new(None),
            links_followed: RefCell::new(HashSet::new()),
            notify: RefCell::new(HashMap::new()),
            fixed_graphs: RefCell::new(HashSet::new()),
            blocked_blocks: RefCell::new(HashMap::new()),
            blocked_graphs: RefCell::new(HashSet::new()),
            frozen: RefCell::new(false),
            policy: RefCell::new(policy),
            bookkeeper,
            keepgoing,
            failed_blocks: RefCell::new(HashSet::new()),
            errors: RefCell::new(Vec::new()),
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
    pub fn annotation(&self, arg: &Hlvalue) -> Option<SomeValue> {
        match arg {
            Hlvalue::Variable(v) => v.annotation.as_ref().map(|rc| (**rc).clone()),
            Hlvalue::Constant(c) => self.bookkeeper.immutablevalue(&c.value).ok(),
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
}
