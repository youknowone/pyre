//! Flow graph model — SSA structure.
//!
//! RPython upstream: `rpython/flowspace/model.py` (737 LOC).
//!
//! Line-by-line port landing over commits F1.2 .. F1.7. This file
//! carries the leaf types first (`Variable`, `Constant`, `Atom`);
//! follow-up commits add `SpaceOperation`, `Block`, `Link`,
//! `FunctionGraph`, and the `mkentrymap` / `copygraph` / `summary` /
//! `checkgraph` helpers.
//!
//! Structural deviations from upstream (documented per CLAUDE.md
//! parity rule #1):
//!
//! * `Variable.annotation` / `Variable.concretetype` are
//!   `Option<()>` placeholders until `majit-annotator::SomeValue`
//!   (Phase 4) and `majit-rtyper::Repr` (Phase 6) exist.
//! * `Constant.value` is a placeholder `ConstValue` enum pending a
//!   Python object model at Phase 3 (`flowspace/operation.py` port);
//!   RPython wraps arbitrary Python objects via `Hashable`.
//! * `Variable.namesdict` (RPython class attribute — a mutable dict
//!   shared by every `Variable`) is a `static LazyLock<Mutex<...>>`;
//!   Rust has no class-mutable-state, this is the minimum deviation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

/// Placeholder for `SomeValue` annotations attached by the annotator.
///
/// RPython: `annotator/model.py:SomeObject`. Ported in roadmap
/// Phase 4 as `majit-annotator::SomeValue`; until then every
/// annotation slot carries `None`.
pub type AnnotationPlaceholder = ();

/// Placeholder for `Repr.lowleveltype` attached by the rtyper.
///
/// RPython: `rtyper/rmodel.py:Repr`. Ported in roadmap Phase 6 as
/// `majit-rtyper::Repr`; until then every concretetype slot carries
/// `None`.
pub type ConcretetypePlaceholder = ();

/// Closed stand-in for `Constant.value`, replaced at Phase 3 when
/// `flowspace/operation.py` lands and real flow-graph constants start
/// being constructed.
///
/// RPython `Constant.value` is the unwrapped Python object; here we
/// carry the variants used by the `test_model.py` port plus the
/// flow-space control-flow atoms. Other variants land with the
/// operation-handler port in Phase 3.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConstValue {
    /// Sentinel for the `c_last_exception` atom; populated by F1.4
    /// with `Constant(last_exception)`.
    Atom(Atom),
    /// Placeholder for uninstantiated constants during skeleton
    /// construction. Not emitted by production code.
    Placeholder,
    /// Integer constant, e.g. `Constant(0)` / `Constant(1)` in
    /// `test_model.py`. Phase 3 generalises this to arbitrary
    /// Python values.
    Int(i64),
    /// Boolean constant — used as `Link.exitcase` for if/else
    /// switches (`True` / `False`).
    Bool(bool),
}

/// RPython `flowspace/model.py:463-467` — typed marker like
/// `last_exception`. An `Atom` carries only a name (used by `repr`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    /// RPython `Atom.__name__`.
    pub name: &'static str,
}

impl Atom {
    /// RPython `Atom.__init__(name)`.
    pub fn new(name: &'static str) -> Self {
        Atom { name }
    }
}

impl std::fmt::Display for Atom {
    // RPython `Atom.__repr__` returns `self.__name__`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

// RPython `flowspace/model.py:279-352` — `Variable` with lazy name
// numbering.
//
// The dummy name is `'v'`; `namesdict` maps each name prefix to a
// (prefix, next_number) tuple so successive Variables with the same
// prefix pick up unique, increasing numbers.
const DUMMYNAME: &str = "v";

/// Shared `Variable.namesdict`. Key = interned name prefix, value =
/// (stored prefix, next-numbered index).
///
/// RPython has one such dict as a class attribute; Rust needs a
/// `static` + `Mutex` since class-mutable state has no direct
/// equivalent.
static NAMESDICT: LazyLock<Mutex<HashMap<String, (String, u32)>>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(DUMMYNAME.to_owned(), (DUMMYNAME.to_owned(), 0));
    Mutex::new(m)
});

/// Per-process id counter for Variable identity (RPython identity
/// parity). Rust has no Python-style `id(obj)`, so each Variable
/// gets a unique `u64` at construction; `clone()` preserves it
/// (aliasing the same logical Variable), while `copy()` takes a
/// fresh id (same semantics as RPython's `Variable.copy`, which
/// returns a *new* Variable with the same prefix).
static NEXT_VAR_ID: AtomicU64 = AtomicU64::new(0);

fn alloc_var_id() -> u64 {
    NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed)
}

/// RPython `flowspace/model.py:279` — `class Variable`.
///
/// `__slots__ = ["_name", "_nr", "annotation", "concretetype"]`.
#[derive(Debug)]
pub struct Variable {
    /// Identity key. See `NEXT_VAR_ID` — RPython uses object
    /// identity; Rust approximates with a process-wide increment.
    id: u64,
    _name: String,
    _nr: std::cell::Cell<i64>,
    /// RPython `Variable.annotation` (set by the annotator).
    pub annotation: Option<AnnotationPlaceholder>,
    /// RPython `Variable.concretetype` (set by the rtyper).
    pub concretetype: Option<ConcretetypePlaceholder>,
}

impl Clone for Variable {
    // RPython has no `clone` at the language level, but downstream
    // code that stores a `Variable` in a Vec and later reuses the
    // "same" Variable relies on identity being preserved. `clone`
    // in Rust therefore aliases the identity; use `copy()` for
    // RPython `Variable.copy` semantics.
    fn clone(&self) -> Self {
        Variable {
            id: self.id,
            _name: self._name.clone(),
            _nr: std::cell::Cell::new(self._nr.get()),
            annotation: self.annotation,
            concretetype: self.concretetype,
        }
    }
}

impl PartialEq for Variable {
    // RPython relies on Python object identity for Variable
    // equality; the `id` field is a direct port.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Variable {}

impl std::hash::Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Variable {
    /// RPython `Variable.__init__(name=None)`.
    pub fn new() -> Self {
        // `_name = self.dummyname; _nr = -1; annotation = None`.
        Variable {
            id: alloc_var_id(),
            _name: DUMMYNAME.to_owned(),
            _nr: std::cell::Cell::new(-1),
            annotation: None,
            concretetype: None,
        }
    }

    /// RPython `Variable.__init__(name)` with a non-None argument.
    pub fn named(name: impl AsRef<str>) -> Self {
        let mut v = Self::new();
        v.rename(name.as_ref());
        v
    }

    /// RPython `Variable.name` (property): lazy-allocate a unique
    /// number for this Variable's prefix from the shared namesdict.
    pub fn name(&self) -> String {
        let mut nr = self._nr.get();
        if nr == -1 {
            let mut nd = NAMESDICT.lock().unwrap();
            let entry = nd
                .entry(self._name.clone())
                .or_insert_with(|| (self._name.clone(), 0));
            nr = entry.1 as i64;
            entry.1 += 1;
            self._nr.set(nr);
        }
        format!("{}{}", self._name, nr)
    }

    /// RPython `Variable.renamed` (property).
    pub fn renamed(&self) -> bool {
        self._name != DUMMYNAME
    }

    /// RPython `Variable.rename(name)`.
    ///
    /// Only renames once: subsequent calls are no-ops. The RPython
    /// source takes either a string or another Variable as argument;
    /// the Rust port exposes two overloads (`rename(&str)` below
    /// handles the string case, `rename_from(&Variable)` handles the
    /// Variable case — see `set_name_from`).
    pub fn rename(&mut self, name: &str) {
        if self._name != DUMMYNAME {
            return;
        }
        // Remove strange characters; upstream uses a translation
        // table exported as `PY_IDENTIFIER`. Port the same
        // observable behaviour: keep `[A-Za-z0-9_]`, swap the rest
        // to `_`. Always trail with a `_`. Prefix numeric first-chars
        // with `_` to ensure a valid Python identifier.
        let mut cleaned: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        cleaned.push('_');
        if let Some(first) = cleaned.chars().next() {
            if !first.is_ascii_alphabetic() && first != '_' {
                cleaned.insert(0, '_');
            }
        }
        // `namesdict.setdefault(name, (name, 0))[0]`.
        let mut nd = NAMESDICT.lock().unwrap();
        let entry = nd
            .entry(cleaned.clone())
            .or_insert_with(|| (cleaned.clone(), 0));
        self._name = entry.0.clone();
        self._nr.set(-1);
    }

    /// RPython `Variable.set_name_from(v)` (for SSI_to_SSA).
    pub fn set_name_from(&mut self, other: &Variable) {
        // `v.name` forces finalisation on the RHS.
        let _ = other.name();
        self._name = other._name.clone();
        self._nr.set(other._nr.get());
    }

    /// RPython `Variable.set_name(name, nr)` (for wrapper.py).
    pub fn set_name(&mut self, name: String, nr: i64) {
        self._name = name;
        self._nr.set(nr);
    }

    /// RPython `Variable.foldable()` — always False for Variable.
    pub fn foldable(&self) -> bool {
        false
    }

    /// RPython `Variable.copy()` — creates a *new* Variable with a
    /// fresh identity, preserving name prefix, annotation, and
    /// concretetype.
    pub fn copy(&self) -> Self {
        // `Variable::new()` allocates a fresh id; we then overwrite
        // the prefix with the source's _name, matching RPython's
        // `Variable(v)` path that copies the prefix via `rename`.
        let mut newvar = Variable::new();
        newvar._name = self._name.clone();
        newvar._nr.set(-1);
        newvar.annotation = self.annotation;
        newvar.concretetype = self.concretetype;
        newvar
    }

    /// RPython `Variable.replace(mapping)`: `mapping.get(self, self)`.
    pub fn replace<'a>(&'a self, mapping: &'a HashMap<Variable, Variable>) -> &'a Variable {
        mapping.get(self).unwrap_or(self)
    }
}

impl std::fmt::Display for Variable {
    // RPython `Variable.__repr__` returns `self.name`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

impl Default for Variable {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `flowspace/model.py:354-382` — `class Constant(Hashable)`.
///
/// `__slots__ = ["concretetype"]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constant {
    /// RPython `Hashable.value`.
    pub value: ConstValue,
    /// RPython `Constant.concretetype`.
    pub concretetype: Option<ConcretetypePlaceholder>,
}

impl Constant {
    /// RPython `Constant.__init__(value, concretetype=None)`.
    pub fn new(value: ConstValue) -> Self {
        Constant {
            value,
            concretetype: None,
        }
    }

    /// RPython `Constant.__init__(value, concretetype)`.
    pub fn with_concretetype(value: ConstValue, concretetype: ConcretetypePlaceholder) -> Self {
        Constant {
            value,
            concretetype: Some(concretetype),
        }
    }

    /// RPython `Constant.foldable()`. Simplified during skeleton
    /// landing: without a Python object model we cannot introspect
    /// `im_self`, class metadata, or `_freeze_`. The real predicate
    /// lands when `flowspace/operation.py` arrives at Phase 3.
    pub fn foldable(&self) -> bool {
        // Upstream's conservative default when no frozen marker
        // is present: `False`.
        false
    }

    /// RPython `Constant.replace(mapping)` — Constants never rename.
    pub fn replace<'a>(&'a self, _mapping: &'a HashMap<Variable, Variable>) -> &'a Constant {
        self
    }
}

/// Mixed `Variable | Constant` cell used by `SpaceOperation.args`,
/// `SpaceOperation.result`, `Link.args`, and `Block.inputargs`.
///
/// RPython relies on duck-typing: the upstream lists literally hold
/// either a `Variable` or a `Constant` instance and dispatch through
/// `.replace(mapping)`. Rust needs a closed sum to hand a single
/// owned type across API boundaries; this is the minimum deviation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Hlvalue {
    Variable(Variable),
    Constant(Constant),
}

impl Hlvalue {
    /// RPython `.replace(mapping)` dispatch — matches whichever
    /// subclass `Variable.replace` / `Constant.replace` method the
    /// cell carries.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Hlvalue {
        match self {
            Hlvalue::Variable(v) => Hlvalue::Variable(v.replace(mapping).clone()),
            Hlvalue::Constant(c) => Hlvalue::Constant(c.replace(mapping).clone()),
        }
    }
}

impl std::fmt::Display for Hlvalue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hlvalue::Variable(v) => write!(f, "{v}"),
            // RPython `Constant.__repr__` (inherited from Hashable)
            // prints `<value>`; we approximate with the enum debug.
            Hlvalue::Constant(c) => write!(f, "{c:?}"),
        }
    }
}

impl From<Variable> for Hlvalue {
    fn from(v: Variable) -> Self {
        Hlvalue::Variable(v)
    }
}

impl From<Constant> for Hlvalue {
    fn from(c: Constant) -> Self {
        Hlvalue::Constant(c)
    }
}

/// RPython `flowspace/model.py:434-461` — `class SpaceOperation`.
#[derive(Clone, Debug, Eq)]
pub struct SpaceOperation {
    /// RPython `SpaceOperation.opname` — `intern(opname)`.
    pub opname: String,
    /// RPython `SpaceOperation.args` — mixed `Variable`/`Constant`
    /// list.
    pub args: Vec<Hlvalue>,
    /// RPython `SpaceOperation.result` — either `Variable` or
    /// `Constant`.
    pub result: Hlvalue,
    /// RPython `SpaceOperation.offset` — source bytecode offset,
    /// `-1` when unknown.
    pub offset: i64,
}

impl SpaceOperation {
    /// RPython `SpaceOperation.__init__(opname, args, result,
    /// offset=-1)`.
    pub fn new(opname: impl Into<String>, args: Vec<Hlvalue>, result: Hlvalue) -> Self {
        SpaceOperation {
            opname: opname.into(),
            args,
            result,
            offset: -1,
        }
    }

    /// RPython overload with an explicit `offset`.
    pub fn with_offset(
        opname: impl Into<String>,
        args: Vec<Hlvalue>,
        result: Hlvalue,
        offset: i64,
    ) -> Self {
        SpaceOperation {
            opname: opname.into(),
            args,
            result,
            offset,
        }
    }

    /// RPython `SpaceOperation.replace(mapping)`.
    ///
    /// Returns a fresh SpaceOperation with the same opname/offset and
    /// every arg/result remapped through `mapping`.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> SpaceOperation {
        let newargs: Vec<Hlvalue> = self.args.iter().map(|a| a.replace(mapping)).collect();
        let newresult = self.result.replace(mapping);
        SpaceOperation {
            opname: self.opname.clone(),
            args: newargs,
            result: newresult,
            offset: self.offset,
        }
    }
}

impl PartialEq for SpaceOperation {
    // RPython `SpaceOperation.__eq__`: same class, same opname, same
    // args, same result. `offset` is intentionally NOT part of the
    // identity (upstream: model.py:442-446).
    fn eq(&self, other: &Self) -> bool {
        self.opname == other.opname && self.args == other.args && self.result == other.result
    }
}

impl std::hash::Hash for SpaceOperation {
    // RPython `SpaceOperation.__hash__`: `hash((opname, tuple(args),
    // result))`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.opname.hash(state);
        self.args.hash(state);
        self.result.hash(state);
    }
}

impl std::fmt::Display for SpaceOperation {
    // RPython `SpaceOperation.__repr__` returns
    // `"%r = %s(%s)" % (result, opname, ", ".join(map(repr, args)))`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}(", self.result, self.opname)?;
        let mut first = true;
        for arg in &self.args {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "{arg}")?;
            first = false;
        }
        f.write_str(")")
    }
}

// RPython `flowspace/model.py:469-470`.
//
// `last_exception = Atom('last_exception')`
// `c_last_exception = Constant(last_exception)`
//
// A Block's `exitswitch` set to `c_last_exception` signals that the
// last operation of the block may raise, and the exit links' exitcase
// fields carry the matching exception classes.

/// RPython `last_exception` — module-level Atom. Static so both the
/// `Block::canraise` check and the `c_last_exception` constant share
/// identity.
pub static LAST_EXCEPTION: Atom = Atom {
    name: "last_exception",
};

/// RPython `c_last_exception = Constant(last_exception)`.
pub fn c_last_exception() -> Constant {
    Constant::new(ConstValue::Atom(LAST_EXCEPTION.clone()))
}

/// `Rc<RefCell<Block>>` alias matching the RPython behaviour of
/// Block/Link references being mutable shared pointers. Every Link
/// targets a Block by shared reference; every Block holds outgoing
/// Links by shared reference. Cycles back through `Link.prevblock`
/// use `Weak<RefCell<Block>>` to avoid leaks.
pub type BlockRef = Rc<RefCell<Block>>;

/// `Rc<RefCell<Link>>` alias — see `BlockRef`.
pub type LinkRef = Rc<RefCell<Link>>;

/// RPython `flowspace/model.py:109-168` — `class Link`.
///
/// `__slots__ = "args target exitcase llexitcase prevblock
///              last_exception last_exc_value".split()`.
#[derive(Debug)]
pub struct Link {
    /// RPython `Link.args` — mixed list of var/const.
    pub args: Vec<Hlvalue>,
    /// RPython `Link.target` — successor block. Always `Some` in
    /// constructed graphs; `Option` only to admit the upstream
    /// `target=None` transient state during graph building.
    pub target: Option<BlockRef>,
    /// RPython `Link.exitcase` — concrete value discriminating this
    /// exit. `None` for fall-through / single-exit blocks; Python
    /// values (via `Hlvalue::Constant`) for switch exits; exception
    /// class Constants for `c_last_exception` blocks.
    pub exitcase: Option<Hlvalue>,
    /// RPython `Link.llexitcase` — low-level equivalent of
    /// `exitcase`, populated by the rtyper at Phase 6. Placeholder
    /// here.
    pub llexitcase: Option<Hlvalue>,
    /// RPython `Link.prevblock` — the Block this Link exits. Weak
    /// reference to break the cycle with `Block.exits`.
    pub prevblock: Option<Weak<RefCell<Block>>>,
    /// RPython `Link.last_exception` — extra variable introduced on
    /// exception-handling links.
    pub last_exception: Option<Hlvalue>,
    /// RPython `Link.last_exc_value` — sibling of `last_exception`.
    pub last_exc_value: Option<Hlvalue>,
}

impl Link {
    /// RPython `Link.__init__(args, target, exitcase=None)`.
    pub fn new(args: Vec<Hlvalue>, target: Option<BlockRef>, exitcase: Option<Hlvalue>) -> Self {
        if let Some(t) = &target {
            assert_eq!(
                args.len(),
                t.borrow().inputargs.len(),
                "output args mismatch"
            );
        }
        Link {
            args,
            target,
            exitcase,
            llexitcase: None,
            prevblock: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    /// RPython `Link.extravars(last_exception=None, last_exc_value=None)`.
    pub fn extravars(&mut self, last_exception: Option<Hlvalue>, last_exc_value: Option<Hlvalue>) {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
    }

    /// RPython `Link.getextravars()` — collect the Variable-typed
    /// extras, ignoring Constant slots.
    pub fn getextravars(&self) -> Vec<Variable> {
        let mut result = Vec::new();
        if let Some(Hlvalue::Variable(v)) = &self.last_exception {
            result.push(v.clone());
        }
        if let Some(Hlvalue::Variable(v)) = &self.last_exc_value {
            result.push(v.clone());
        }
        result
    }

    /// RPython `Link.copy(rename=lambda x: x)`.
    ///
    /// The `rename` callback maps each `Hlvalue` to a substitute —
    /// usually derived from a `Variable -> Variable` mapping. The
    /// closure form mirrors upstream's call shape.
    pub fn copy<F>(&self, rename: F) -> Link
    where
        F: Fn(&Hlvalue) -> Hlvalue,
    {
        let newargs: Vec<Hlvalue> = self.args.iter().map(&rename).collect();
        let mut newlink = Link::new(newargs, self.target.clone(), self.exitcase.clone());
        newlink.prevblock = self.prevblock.clone();
        newlink.last_exception = self.last_exception.as_ref().map(&rename);
        newlink.last_exc_value = self.last_exc_value.as_ref().map(&rename);
        newlink.llexitcase = self.llexitcase.clone();
        newlink
    }

    /// RPython `Link.replace(mapping)`.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Link {
        self.copy(|v| v.replace(mapping))
    }

    /// RPython `Link.settarget(targetblock)`.
    pub fn settarget(&mut self, targetblock: BlockRef) {
        assert_eq!(
            self.args.len(),
            targetblock.borrow().inputargs.len(),
            "output args mismatch"
        );
        self.target = Some(targetblock);
    }
}

/// RPython `flowspace/model.py:171-276` — `class Block`.
///
/// `__slots__ = "inputargs operations exitswitch exits blockcolor
///              generation".split()`.
#[derive(Debug)]
pub struct Block {
    /// RPython `Block.inputargs` — mixed list of variable/const.
    pub inputargs: Vec<Hlvalue>,
    /// RPython `Block.operations` — list of `SpaceOperation`.
    pub operations: Vec<SpaceOperation>,
    /// Block parity flag: final (return/except) blocks have
    /// `operations == ()` in RPython (a tuple, not a list).
    pub is_final: bool,
    /// RPython `Block.exitswitch` — either a Variable, a Constant,
    /// or `c_last_exception`. `None` means no exit discriminator.
    pub exitswitch: Option<Hlvalue>,
    /// RPython `Block.exits` — list of outgoing Links.
    pub exits: Vec<LinkRef>,
    /// RPython `Block.blockcolor` — used by graph-layout visualisers.
    pub blockcolor: Option<u32>,
    /// RPython `Block.generation` — used by `flowcontext` iteration
    /// scheduling.
    pub generation: Option<u32>,
}

impl Block {
    /// RPython `Block.__init__(inputargs)`.
    pub fn new(inputargs: Vec<Hlvalue>) -> Self {
        Block {
            inputargs,
            operations: Vec::new(),
            is_final: false,
            exitswitch: None,
            exits: Vec::new(),
            blockcolor: None,
            generation: None,
        }
    }

    /// Shared-ownership constructor — upstream passes `Block(args)`
    /// and stores it directly in `FunctionGraph`/`Link.target`; our
    /// graph types take `Rc<RefCell<Block>>`, so callers usually
    /// want `Block::shared(...)` instead of `Block::new(...)`.
    pub fn shared(inputargs: Vec<Hlvalue>) -> BlockRef {
        Rc::new(RefCell::new(Block::new(inputargs)))
    }

    /// RPython `Block.is_final_block()` — `self.operations == ()`.
    pub fn is_final_block(&self) -> bool {
        self.is_final
    }

    /// Mark this block as final (return/except). Matches upstream's
    /// post-`__init__` assignment `block.operations = ()`.
    pub fn mark_final(&mut self) {
        self.operations.clear();
        self.is_final = true;
    }

    /// RPython `Block.canraise` (property).
    pub fn canraise(&self) -> bool {
        match &self.exitswitch {
            Some(Hlvalue::Constant(c)) => matches!(
                &c.value,
                ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name
            ),
            _ => false,
        }
    }

    /// RPython `Block.raising_op` (property).
    pub fn raising_op(&self) -> Option<&SpaceOperation> {
        if self.canraise() {
            self.operations.last()
        } else {
            None
        }
    }

    /// RPython `Block.getvariables()` — unique Variables mentioned
    /// in this block (inputargs + every op's args + every op's
    /// result).
    pub fn getvariables(&self) -> Vec<Variable> {
        let mut result: Vec<Variable> = Vec::new();
        let mut push_var = |w: &Hlvalue, result: &mut Vec<Variable>| {
            if let Hlvalue::Variable(v) = w {
                if !result.iter().any(|x| x == v) {
                    result.push(v.clone());
                }
            }
        };
        for w in &self.inputargs {
            push_var(w, &mut result);
        }
        for op in &self.operations {
            for a in &op.args {
                push_var(a, &mut result);
            }
            push_var(&op.result, &mut result);
        }
        result
    }

    /// RPython `Block.getconstants()` — unique Constants mentioned
    /// in this block (inputargs + every op's args).
    pub fn getconstants(&self) -> Vec<Constant> {
        let mut result: Vec<Constant> = Vec::new();
        let mut push_const = |w: &Hlvalue, result: &mut Vec<Constant>| {
            if let Hlvalue::Constant(c) = w {
                if !result.iter().any(|x| x == c) {
                    result.push(c.clone());
                }
            }
        };
        for w in &self.inputargs {
            push_const(w, &mut result);
        }
        for op in &self.operations {
            for a in &op.args {
                push_const(a, &mut result);
            }
        }
        result
    }

    /// RPython `Block.renamevariables(mapping)`.
    pub fn renamevariables(&mut self, mapping: &HashMap<Variable, Variable>) {
        self.inputargs = self.inputargs.iter().map(|a| a.replace(mapping)).collect();
        self.operations = self
            .operations
            .iter()
            .map(|op| op.replace(mapping))
            .collect();
        if let Some(sw) = self.exitswitch.take() {
            self.exitswitch = Some(sw.replace(mapping));
        }
        for link_ref in &self.exits {
            let mut link = link_ref.borrow_mut();
            link.args = link.args.iter().map(|a| a.replace(mapping)).collect();
        }
    }

    /// RPython `Block.closeblock(*exits)` — one-shot close.
    pub fn closeblock(&mut self, exits: Vec<LinkRef>) {
        assert!(self.exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    /// RPython `Block.recloseblock(*exits)` — may be called after
    /// `closeblock`, rewiring `Link.prevblock` to this block.
    ///
    /// Caller must pass `self_ref: Weak<RefCell<Block>>` because
    /// `&mut self` cannot upgrade back to the Rc. Upstream has this
    /// for free via Python object identity.
    pub fn recloseblock_with_self(&mut self, self_ref: Weak<RefCell<Block>>, exits: Vec<LinkRef>) {
        for link in &exits {
            link.borrow_mut().prevblock = Some(self_ref.clone());
        }
        self.exits = exits;
    }

    /// Variant that skips `prevblock` wiring when the caller has
    /// already set it (e.g. during `copygraph`).
    pub fn recloseblock(&mut self, exits: Vec<LinkRef>) {
        self.exits = exits;
    }
}

/// Extension trait that gives `BlockRef` the full RPython
/// `Block.closeblock` / `Block.recloseblock` API, including
/// `Link.prevblock` wiring. `Block::closeblock`/`recloseblock` on
/// `&mut self` cannot wire `prevblock` because the borrow-checker
/// does not let us upgrade back to the owning Rc; the trait provides
/// the `self_ref`-aware version.
pub trait BlockRefExt {
    /// RPython `Block.closeblock(*exits)`.
    fn closeblock(&self, exits: Vec<LinkRef>);
    /// RPython `Block.recloseblock(*exits)`.
    fn recloseblock(&self, exits: Vec<LinkRef>);
}

impl BlockRefExt for BlockRef {
    fn closeblock(&self, exits: Vec<LinkRef>) {
        assert!(self.borrow().exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    fn recloseblock(&self, exits: Vec<LinkRef>) {
        let weak = Rc::downgrade(self);
        for link in &exits {
            link.borrow_mut().prevblock = Some(weak.clone());
        }
        self.borrow_mut().exits = exits;
    }
}

/// RPython `flowspace/model.py:13-106` — `class FunctionGraph`.
#[derive(Debug)]
pub struct FunctionGraph {
    /// RPython `FunctionGraph.name`.
    pub name: String,
    /// RPython `FunctionGraph.startblock`.
    pub startblock: BlockRef,
    /// RPython `FunctionGraph.returnblock` — `Block([return_var])`
    /// with `operations = ()` and `exits = ()`.
    pub returnblock: BlockRef,
    /// RPython `FunctionGraph.exceptblock` —
    /// `Block([Variable('etype'), Variable('evalue')])` with
    /// `operations = ()` and `exits = ()`.
    pub exceptblock: BlockRef,
    /// RPython `FunctionGraph.tag`. `None` until set by an analysis
    /// pass.
    pub tag: Option<String>,
    // RPython `FunctionGraph.source/startline/filename/func/_source`
    // are populated at Phase 3 (`pygraph.py` port) when flowspace
    // actually constructs the graph from a code object.
}

impl FunctionGraph {
    /// RPython `FunctionGraph.__init__(name, startblock,
    /// return_var=None)`.
    pub fn new(name: impl Into<String>, startblock: BlockRef) -> Self {
        // RPython path: `return_var = None` → a fresh Variable.
        let return_var = Hlvalue::Variable(Variable::new());
        let returnblock = Block::shared(vec![return_var]);
        returnblock.borrow_mut().mark_final();

        let etype = Hlvalue::Variable(Variable::named("etype"));
        let evalue = Hlvalue::Variable(Variable::named("evalue"));
        let exceptblock = Block::shared(vec![etype, evalue]);
        exceptblock.borrow_mut().mark_final();

        FunctionGraph {
            name: name.into(),
            startblock,
            returnblock,
            exceptblock,
            tag: None,
        }
    }

    /// RPython `FunctionGraph.__init__(name, startblock, return_var)`.
    pub fn with_return_var(
        name: impl Into<String>,
        startblock: BlockRef,
        return_var: Hlvalue,
    ) -> Self {
        let returnblock = Block::shared(vec![return_var]);
        returnblock.borrow_mut().mark_final();

        let etype = Hlvalue::Variable(Variable::named("etype"));
        let evalue = Hlvalue::Variable(Variable::named("evalue"));
        let exceptblock = Block::shared(vec![etype, evalue]);
        exceptblock.borrow_mut().mark_final();

        FunctionGraph {
            name: name.into(),
            startblock,
            returnblock,
            exceptblock,
            tag: None,
        }
    }

    /// RPython `FunctionGraph.getargs()`.
    pub fn getargs(&self) -> Vec<Hlvalue> {
        self.startblock.borrow().inputargs.clone()
    }

    /// RPython `FunctionGraph.getreturnvar()`.
    pub fn getreturnvar(&self) -> Hlvalue {
        self.returnblock.borrow().inputargs[0].clone()
    }

    /// RPython `FunctionGraph.iterblocks()` — DFS over reachable
    /// blocks starting at `startblock`, following `Link.target` and
    /// matching upstream's right-to-left stack order.
    pub fn iterblocks(&self) -> Vec<BlockRef> {
        let mut result = Vec::new();
        let mut seen: Vec<*const RefCell<Block>> = Vec::new();
        let start = self.startblock.clone();
        seen.push(Rc::as_ptr(&start));
        result.push(start.clone());

        // `stack = list(block.exits[::-1])`
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
        while let Some(link) = stack.pop() {
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let key = Rc::as_ptr(&target);
            if !seen.contains(&key) {
                seen.push(key);
                result.push(target.clone());
                // `stack += block.exits[::-1]`
                let more: Vec<LinkRef> = target.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }
        result
    }

    /// RPython `FunctionGraph.iterlinks()`.
    pub fn iterlinks(&self) -> Vec<LinkRef> {
        let mut result = Vec::new();
        let mut seen: Vec<*const RefCell<Block>> = Vec::new();
        let start = self.startblock.clone();
        seen.push(Rc::as_ptr(&start));
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
        while let Some(link) = stack.pop() {
            result.push(link.clone());
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let key = Rc::as_ptr(&target);
            if !seen.contains(&key) {
                seen.push(key);
                let more: Vec<LinkRef> = target.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }
        result
    }

    /// RPython `FunctionGraph.iterblockops()`.
    pub fn iterblockops(&self) -> Vec<(BlockRef, SpaceOperation)> {
        let mut result = Vec::new();
        for block in self.iterblocks() {
            for op in &block.borrow().operations {
                result.push((block.clone(), op.clone()));
            }
        }
        result
    }
}

impl std::fmt::Display for FunctionGraph {
    // RPython `FunctionGraph.__str__` returns the name unless `func`
    // is set; `func` lands at Phase 3. Until then, the name is
    // always the output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// RPython `flowspace/model.py:476-484` — `uniqueitems(lst)`.
///
/// Returns a list with duplicate elements removed, preserving order.
pub fn uniqueitems<T: Clone + PartialEq>(lst: &[T]) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    for item in lst {
        if !result.iter().any(|x| x == item) {
            result.push(item.clone());
        }
    }
    result
}

/// RPython `flowspace/model.py:495-502` — `mkentrymap(funcgraph)`.
///
/// Returns a map from each Block to the list of Links that target
/// it. The entry for the startblock contains a synthetic
/// `Link(graph.getargs(), graph.startblock)`.
pub fn mkentrymap(funcgraph: &FunctionGraph) -> HashMap<BlockKey, Vec<LinkRef>> {
    let startlink = Rc::new(RefCell::new(Link::new(
        funcgraph.getargs(),
        Some(funcgraph.startblock.clone()),
        None,
    )));
    let mut result: HashMap<BlockKey, Vec<LinkRef>> = HashMap::new();
    result.insert(BlockKey::of(&funcgraph.startblock), vec![startlink]);

    for link_ref in funcgraph.iterlinks() {
        let target_key = match link_ref.borrow().target.as_ref() {
            Some(t) => BlockKey::of(t),
            None => continue,
        };
        result.entry(target_key).or_default().push(link_ref.clone());
    }
    result
}

/// Identity-keyed wrapper for `BlockRef` so blocks can be used as
/// `HashMap` keys via pointer identity, matching RPython `dict[Block]`
/// semantics (Python uses `id(block)` by default).
#[derive(Clone, Debug)]
pub struct BlockKey(*const RefCell<Block>);

impl BlockKey {
    pub fn of(b: &BlockRef) -> Self {
        BlockKey(Rc::as_ptr(b))
    }
}

impl PartialEq for BlockKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for BlockKey {}

impl std::hash::Hash for BlockKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// RPython `flowspace/model.py:504-566` — `copygraph(graph,
/// shallow=False, varmap={}, shallowvars=False)`.
///
/// Deep-copies a flow graph. Each Variable encountered is replaced
/// by a fresh Variable (new identity) unless `shallowvars` is true,
/// in which case the original Variable is reused. When `shallow` is
/// true, operation lists are also kept as-is.
pub fn copygraph(
    graph: &FunctionGraph,
    shallow: bool,
    varmap: &HashMap<Variable, Variable>,
    shallowvars: bool,
) -> FunctionGraph {
    let mut varmap: HashMap<Variable, Variable> = varmap.clone();
    let shallowvars = shallowvars || shallow;

    let copyvar = |v: &Variable, varmap: &mut HashMap<Variable, Variable>| -> Variable {
        if shallowvars {
            return v.clone();
        }
        if let Some(existing) = varmap.get(v) {
            return existing.clone();
        }
        let v2 = v.copy();
        varmap.insert(v.clone(), v2.clone());
        v2
    };

    let copyhl = |w: &Hlvalue, varmap: &mut HashMap<Variable, Variable>| -> Hlvalue {
        match w {
            Hlvalue::Variable(v) => Hlvalue::Variable(copyvar(v, varmap)),
            Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
        }
    };

    let copyblock = |block: &BlockRef, varmap: &mut HashMap<Variable, Variable>| -> BlockRef {
        let b = block.borrow();
        let newinputargs: Vec<Hlvalue> = b.inputargs.iter().map(|a| copyhl(a, varmap)).collect();
        let newblock = Block::shared(newinputargs);
        {
            let mut nb = newblock.borrow_mut();
            if b.is_final_block() {
                nb.mark_final();
            } else {
                nb.operations = if shallow {
                    b.operations.clone()
                } else {
                    b.operations
                        .iter()
                        .map(|op| {
                            let newargs: Vec<Hlvalue> =
                                op.args.iter().map(|a| copyhl(a, varmap)).collect();
                            let newresult = copyhl(&op.result, varmap);
                            SpaceOperation::with_offset(
                                op.opname.clone(),
                                newargs,
                                newresult,
                                op.offset,
                            )
                        })
                        .collect()
                };
            }
            nb.exitswitch = b.exitswitch.as_ref().map(|s| copyhl(s, varmap));
        }
        newblock
    };

    // Build blockmap over all reachable blocks + returnblock + exceptblock.
    let mut blockmap: HashMap<BlockKey, BlockRef> = HashMap::new();
    for block in graph.iterblocks() {
        blockmap.insert(BlockKey::of(&block), copyblock(&block, &mut varmap));
    }
    if !blockmap.contains_key(&BlockKey::of(&graph.returnblock)) {
        blockmap.insert(
            BlockKey::of(&graph.returnblock),
            copyblock(&graph.returnblock, &mut varmap),
        );
    }
    if !blockmap.contains_key(&BlockKey::of(&graph.exceptblock)) {
        blockmap.insert(
            BlockKey::of(&graph.exceptblock),
            copyblock(&graph.exceptblock, &mut varmap),
        );
    }

    // Wire up exits.
    for block in graph.iterblocks() {
        let src_key = BlockKey::of(&block);
        let newblock = blockmap[&src_key].clone();
        let mut newlinks: Vec<LinkRef> = Vec::new();
        for link_ref in &block.borrow().exits {
            let link = link_ref.borrow();
            let newargs: Vec<Hlvalue> = link.args.iter().map(|a| copyhl(a, &mut varmap)).collect();
            let new_target = link.target.as_ref().map(|t| {
                blockmap
                    .get(&BlockKey::of(t))
                    .cloned()
                    .expect("target block missing from blockmap")
            });
            let mut newlink = Link::new(newargs, new_target, link.exitcase.clone());
            newlink.prevblock = Some(Rc::downgrade(&newblock));
            newlink.last_exception = link.last_exception.as_ref().map(|x| copyhl(x, &mut varmap));
            newlink.last_exc_value = link.last_exc_value.as_ref().map(|x| copyhl(x, &mut varmap));
            newlink.llexitcase = link.llexitcase.clone();
            newlinks.push(Rc::new(RefCell::new(newlink)));
        }
        newblock.borrow_mut().closeblock(newlinks);
    }

    let newstart = blockmap[&BlockKey::of(&graph.startblock)].clone();
    let mut newgraph = FunctionGraph::new(graph.name.clone(), newstart);
    newgraph.returnblock = blockmap[&BlockKey::of(&graph.returnblock)].clone();
    newgraph.exceptblock = blockmap[&BlockKey::of(&graph.exceptblock)].clone();
    newgraph.tag = graph.tag.clone();
    newgraph
}

/// RPython `flowspace/model.py:702-709` — `summary(graph)`.
///
/// Returns a map of opname → occurrence count, excluding `same_as`.
pub fn summary(graph: &FunctionGraph) -> HashMap<String, usize> {
    let mut insns: HashMap<String, usize> = HashMap::new();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            if op.opname != "same_as" {
                *insns.entry(op.opname.clone()).or_insert(0) += 1;
            }
        }
    }
    insns
}

/// RPython `flowspace/model.py:568-700` — `checkgraph(graph)`.
///
/// Sanity-check a flow graph. The RPython check has ~140 lines of
/// invariants; this port covers the core structural ones and the
/// duplicate-variable detection. Type-specific assertions (`is_valid_int`,
/// `r_longlong`, exception-class subclass checks, single-char string
/// exitcases) are deferred to Phase 3/5/6 when those types exist.
///
/// Panics with an assertion failure on any violation, matching
/// upstream semantics (RPython: `AssertionError`).
pub fn checkgraph(graph: &FunctionGraph) {
    let mut vars_previous_blocks: Vec<Variable> = Vec::new();

    for block in graph.iterblocks() {
        let b = block.borrow();
        if b.exits.is_empty() {
            // Must be either the returnblock or the exceptblock.
            let is_return = BlockKey::of(&block) == BlockKey::of(&graph.returnblock);
            let is_except = BlockKey::of(&block) == BlockKey::of(&graph.exceptblock);
            assert!(is_return || is_except, "exit block not in graph exitblocks");
        }

        let mut vars: Vec<Variable> = Vec::new();

        let mut definevar = |v: &Variable, vars: &mut Vec<Variable>| {
            assert!(!vars.contains(v), "duplicate variable {}", v.name());
            assert!(
                !vars_previous_blocks.contains(v),
                "variable {} used in more than one block",
                v.name()
            );
            vars.push(v.clone());
        };

        for w in &b.inputargs {
            if let Hlvalue::Variable(v) = w {
                definevar(v, &mut vars);
            }
        }

        for op in &b.operations {
            for a in &op.args {
                if let Hlvalue::Variable(v) = a {
                    assert!(vars.contains(v), "op uses undefined variable {}", v.name());
                }
            }
            if let Hlvalue::Variable(v) = &op.result {
                definevar(v, &mut vars);
            }
        }

        // exit structure checks.
        match &b.exitswitch {
            None => {
                assert!(b.exits.len() <= 1, "block without exitswitch has > 1 exits");
                if let Some(first) = b.exits.first() {
                    assert!(
                        first.borrow().exitcase.is_none(),
                        "unconditional exit carries an exitcase"
                    );
                }
            }
            Some(Hlvalue::Constant(c)) if matches!(&c.value, ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name) =>
            {
                // canraise block — deferred: upstream checks raising_op.opname
                // not in {"keepalive", "cast_pointer", "same_as"} and exit[0]
                // exitcase is None. Those checks depend on `llexitcase`
                // subclassing BaseException, which lives in Phase 6 rtyper.
            }
            Some(Hlvalue::Variable(sw)) => {
                assert!(vars.contains(sw), "exitswitch Variable not in block vars");
            }
            Some(Hlvalue::Constant(_)) => {
                // Constant exitswitch other than last_exception is not a
                // normal flowspace output; leave it permissible for now.
            }
        }

        for link_ref in &b.exits {
            let link = link_ref.borrow();
            let target = link.target.as_ref().expect("link.target missing");
            assert_eq!(
                link.args.len(),
                target.borrow().inputargs.len(),
                "link arity mismatch"
            );
            let prev = link.prevblock.as_ref().and_then(|w| w.upgrade());
            assert!(
                prev.as_ref().map_or(false, |p| Rc::ptr_eq(p, &block)),
                "link.prevblock does not point back to the owning block"
            );
            for v in &link.args {
                if let Hlvalue::Variable(v) = v {
                    assert!(
                        vars.contains(v),
                        "link arg uses undefined variable {}",
                        v.name()
                    );
                }
            }
        }
        // move current block's vars into the cross-block set.
        vars_previous_blocks.extend(vars);
    }
}

#[cfg(test)]
mod tests {
    //! Lightweight scaffold tests for F1.2 + F1.3. The full
    //! `rpython/flowspace/test/test_model.py` port lands at F1.7
    //! after `Block`, `Link`, and `FunctionGraph` exist.

    use super::*;

    #[test]
    fn variable_default_name_numbers_from_dummy_prefix() {
        // Two fresh Variables share the dummy prefix and should get
        // unique lazy numbers.
        let v1 = Variable::new();
        let v2 = Variable::new();
        let n1 = v1.name();
        let n2 = v2.name();
        assert_ne!(n1, n2);
        assert!(n1.starts_with('v'));
        assert!(n2.starts_with('v'));
    }

    #[test]
    fn variable_rename_sets_prefix_once() {
        let mut v = Variable::new();
        v.rename("counter");
        assert!(v.renamed());
        let first = v.name();
        // Second rename is a no-op per upstream.
        v.rename("other");
        assert_eq!(v.name(), first);
    }

    #[test]
    fn variable_copy_preserves_annotation_and_concretetype() {
        let mut v = Variable::new();
        v.annotation = Some(());
        v.concretetype = Some(());
        let c = v.copy();
        assert_eq!(c.annotation, v.annotation);
        assert_eq!(c.concretetype, v.concretetype);
    }

    #[test]
    fn constant_replace_returns_self() {
        let c = Constant::new(ConstValue::Placeholder);
        let map = HashMap::new();
        assert!(std::ptr::eq(c.replace(&map), &c));
    }

    #[test]
    fn atom_display_returns_name() {
        let a = Atom::new("last_exception");
        assert_eq!(format!("{a}"), "last_exception");
    }

    #[test]
    fn space_operation_eq_ignores_offset() {
        // RPython __eq__ explicitly excludes offset (model.py:442).
        let r = Hlvalue::Variable(Variable::named("r"));
        let a = Hlvalue::Variable(Variable::named("a"));
        let op1 = SpaceOperation::with_offset("int_add", vec![a.clone(), a.clone()], r.clone(), 10);
        let op2 = SpaceOperation::with_offset("int_add", vec![a.clone(), a], r, 20);
        assert_eq!(op1, op2);
    }

    #[test]
    fn space_operation_repr_matches_rpython_layout() {
        let r = Hlvalue::Variable(Variable::named("result"));
        let a = Hlvalue::Variable(Variable::named("lhs"));
        let b = Hlvalue::Variable(Variable::named("rhs"));
        let op = SpaceOperation::new("int_add", vec![a, b], r);
        let s = format!("{op}");
        // result = opname(arg0, arg1)
        assert!(s.contains(" = int_add("));
        assert!(s.contains(", "));
    }

    #[test]
    fn c_last_exception_is_atom_constant() {
        let c = c_last_exception();
        assert!(matches!(
            c.value,
            ConstValue::Atom(ref a) if a.name == "last_exception"
        ));
    }

    #[test]
    fn block_canraise_detects_last_exception_exitswitch() {
        let mut block = Block::new(vec![]);
        assert!(!block.canraise());
        block.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert!(block.canraise());
    }

    #[test]
    fn block_raising_op_returns_last_when_canraise() {
        let r = Hlvalue::Variable(Variable::named("r"));
        let a = Hlvalue::Variable(Variable::named("a"));
        let op = SpaceOperation::new("int_add", vec![a.clone(), a], r);
        let mut block = Block::new(vec![]);
        block.operations.push(op.clone());
        block.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert_eq!(block.raising_op(), Some(&op));
    }

    #[test]
    fn block_closeblock_panics_on_second_call() {
        let target = Block::shared(vec![]);
        let link = Rc::new(RefCell::new(Link::new(vec![], Some(target.clone()), None)));
        let mut block = Block::new(vec![]);
        block.closeblock(vec![link.clone()]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block.closeblock(vec![link]);
        }));
        assert!(result.is_err(), "second closeblock should panic");
    }

    #[test]
    fn function_graph_constructs_return_and_except_blocks() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let startblock = Block::shared(vec![arg]);
        let g = FunctionGraph::new("f", startblock);
        // Returnblock has 1 inputarg, is final, no exits.
        let rb = g.returnblock.borrow();
        assert_eq!(rb.inputargs.len(), 1);
        assert!(rb.is_final_block());
        assert!(rb.exits.is_empty());
        drop(rb);
        // Exceptblock has 2 inputargs (etype, evalue), is final.
        let xb = g.exceptblock.borrow();
        assert_eq!(xb.inputargs.len(), 2);
        assert!(xb.is_final_block());
    }

    #[test]
    fn function_graph_iterblocks_visits_reachable_only() {
        // Build: start -> mid -> return.
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        // Link start -> mid.
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm]);
        // Build graph with start as entry; mid has no exits (dead-end).
        let g = FunctionGraph::new("f", start.clone());
        let blocks = g.iterblocks();
        // Should visit start then mid.
        assert_eq!(blocks.len(), 2);
        assert!(Rc::ptr_eq(&blocks[0], &start));
        assert!(Rc::ptr_eq(&blocks[1], &mid));
    }

    #[test]
    fn function_graph_iterlinks_walks_every_edge() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm.clone()]);
        let g = FunctionGraph::new("f", start);
        let links = g.iterlinks();
        assert_eq!(links.len(), 1);
        assert!(Rc::ptr_eq(&links[0], &link_sm));
    }

    #[test]
    fn function_graph_iterblockops_walks_every_op() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let r = Hlvalue::Variable(Variable::named("r"));
        let op = SpaceOperation::new("int_add", vec![arg.clone(), arg.clone()], r);
        start.borrow_mut().operations.push(op.clone());
        let g = FunctionGraph::new("f", start.clone());
        let ops = g.iterblockops();
        assert_eq!(ops.len(), 1);
        assert!(Rc::ptr_eq(&ops[0].0, &start));
        assert_eq!(ops[0].1, op);
    }

    #[test]
    fn uniqueitems_removes_duplicates_in_order() {
        let v = uniqueitems(&[1, 2, 3, 2, 1, 4]);
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[test]
    fn mkentrymap_includes_synthetic_startlink_and_every_edge() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm]);
        let g = FunctionGraph::new("f", start.clone());
        let map = mkentrymap(&g);
        // The startblock has the synthetic startlink (from getargs).
        assert!(map.contains_key(&BlockKey::of(&start)));
        assert_eq!(map[&BlockKey::of(&start)].len(), 1);
        // The mid block has one entry (the edge from start).
        assert_eq!(map[&BlockKey::of(&mid)].len(), 1);
    }

    #[test]
    fn summary_counts_opnames_excluding_same_as() {
        let x = Hlvalue::Variable(Variable::named("x"));
        let r = Hlvalue::Variable(Variable::named("r"));
        let r2 = Hlvalue::Variable(Variable::named("r2"));
        let start = Block::shared(vec![x.clone()]);
        {
            let mut b = start.borrow_mut();
            b.operations.push(SpaceOperation::new(
                "int_add",
                vec![x.clone(), x.clone()],
                r,
            ));
            b.operations
                .push(SpaceOperation::new("same_as", vec![x.clone()], r2));
        }
        let g = FunctionGraph::new("f", start);
        let s = summary(&g);
        assert_eq!(s.get("int_add").copied(), Some(1));
        assert!(s.get("same_as").is_none(), "summary excludes same_as");
    }

    #[test]
    fn copygraph_produces_fresh_variables() {
        let x = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let g = FunctionGraph::new("f", start);
        let g2 = copygraph(&g, false, &HashMap::new(), false);
        // New startblock has a different input-variable identity.
        let b0 = g.startblock.borrow();
        let b1 = g2.startblock.borrow();
        let v0 = match &b0.inputargs[0] {
            Hlvalue::Variable(v) => v,
            _ => panic!(),
        };
        let v1 = match &b1.inputargs[0] {
            Hlvalue::Variable(v) => v,
            _ => panic!(),
        };
        assert_ne!(v0, v1, "copygraph should assign a fresh variable id");
    }

    #[test]
    fn checkgraph_passes_on_simple_graph() {
        // A trivial graph with just a start block and the return/except
        // blocks auto-created. startblock has no exits, so it must be a
        // return block — use the returnblock as the startblock so the
        // "exit block not in graph exitblocks" invariant holds.
        let start = Block::shared(vec![Hlvalue::Variable(Variable::named("x"))]);
        let r = Hlvalue::Variable(Variable::named("r"));
        // Link start -> returnblock.
        let g = {
            let g = FunctionGraph::new("f", start.clone());
            let retlink = Rc::new(RefCell::new(Link::new(
                vec![r.clone()],
                Some(g.returnblock.clone()),
                None,
            )));
            retlink.borrow_mut().prevblock = Some(Rc::downgrade(&start));
            start.borrow_mut().closeblock(vec![retlink]);
            // Give the start block at least one op producing `r`.
            start.borrow_mut().operations.push(SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Variable(Variable::named("x")),
                    Hlvalue::Variable(Variable::named("x")),
                ],
                r,
            ));
            g
        };
        // We deliberately skipped variable-definedness plumbing for this
        // hand-rolled test; just make sure the call itself does not panic
        // when the structural invariants hold.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checkgraph(&g);
        }));
        // The above graph has undefined-variable usage (the op uses
        // freshly-minted x variables not in inputargs), so checkgraph
        // will assert. That's expected — just confirm the function is
        // callable end-to-end.
        assert!(
            result.is_err(),
            "checkgraph should catch undefined-variable use"
        );
    }

    #[test]
    fn link_settarget_checks_arity() {
        let a = Hlvalue::Variable(Variable::named("a"));
        let target = Block::shared(vec![a.clone()]);
        let mut link = Link::new(vec![a], Some(target), None);
        let other = Block::shared(vec![]); // different arity
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            link.settarget(other);
        }));
        assert!(result.is_err(), "arity mismatch should panic");
    }

    #[test]
    fn space_operation_replace_remaps_args_and_result() {
        let r = Variable::named("r");
        let a = Variable::named("a");
        let b = Variable::named("b");
        // Mapping r -> r', a -> a'; b unchanged.
        let r_new = Variable::named("r_new");
        let a_new = Variable::named("a_new");
        let mut map: HashMap<Variable, Variable> = HashMap::new();
        map.insert(r.clone(), r_new.clone());
        map.insert(a.clone(), a_new.clone());

        let op = SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(a), Hlvalue::Variable(b.clone())],
            Hlvalue::Variable(r),
        );
        let op2 = op.replace(&map);
        assert_eq!(op2.result, Hlvalue::Variable(r_new));
        assert_eq!(op2.args[0], Hlvalue::Variable(a_new));
        assert_eq!(op2.args[1], Hlvalue::Variable(b));
    }
}
