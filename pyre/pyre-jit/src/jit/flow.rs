//! Port of `rpython/flowspace/model.py` for pyre-jit's codewriter.
//!
//! pyre-jit's direct-dispatch codewriter currently emits `SSARepr`
//! instructions inline as it walks the CPython bytecode (one synthetic
//! "block" per Python PC). RPython's pipeline is different:
//!
//!   jtransform → `FunctionGraph { blocks[], startblock }`
//!                      ↓
//!                   regalloc
//!                      ↓
//!                   flatten  → `SSARepr`
//!                      ↓
//!                compute_liveness
//!                      ↓
//!                   assemble
//!
//! The `FunctionGraph` layer makes **link.args ↔ target.inputargs**
//! available to the regalloc (`rpython/tool/algo/regalloc.py:79-112`
//! `coalesce_variables` coalesces those pairs) and to `insert_renamings`
//! (`rpython/jit/codewriter/flatten.py:306-334` emits `%s_copy` /
//! `%s_push` / `%s_pop` when the pairs are not already coalesced).
//!
//! pyre-jit currently substitutes:
//! - An SSARepr-level `ref_copy` scanner (`pyre-jit/src/jit/regalloc.rs`
//!   `coalesce_variables`) in place of RPython's link-level scanner.
//! - No `insert_renamings` at all — pyre-jit's walker pre-breaks cycles
//!   with explicit `obj_tmp*` registers.
//!
//! This module is the RPython-orthodox replacement surface. The data
//! types preserve the same core object shape (Variable, Constant, Link,
//! Block, FunctionGraph, including the special return/except blocks and
//! link exception extras); follow-up slices wire the walker to populate a
//! `FunctionGraph` alongside its current SSARepr emission and eventually
//! drive the regalloc + insert_renamings passes from the graph.
//!
//! Scope of THIS slice: data types only. No walker integration yet.
//! Subsequent slices of Step 6 (see task #205) populate and consume.

use std::cell::{Ref, RefCell, RefMut};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use super::flatten::{IndirectCallTargets, Kind};

/// `rpython/flowspace/model.py:241` `class Variable(object)`.
///
/// Upstream:
/// ```py
/// class Variable(object):
///     def __init__(self, name=None):
///         ...
///         self.concretetype = None
/// ```
///
/// pyre preserves the eventual JIT register kind directly on the
/// `Variable`, but unlike earlier slices the field is now optional:
/// RPython creates `returnblock` / `exceptblock` variables before
/// typing is known, so `kind == None` stands for "concretetype not set
/// yet".
///
/// Identity is by `VariableId` (the `id` field), matching RPython's
/// Python-object-identity semantics used by `Link.args` / `block.inputargs`
/// membership and by `regalloc.py`'s `die_at` dictionary keyed on Variable
/// objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Variable {
    /// Stable per-graph identity. The dispatch walker allocates
    /// monotonically increasing ids in emission order.
    pub id: VariableId,
    pub kind: Option<Kind>,
}

const DUMMYNAME: &str = "v";

#[derive(Debug)]
struct VariableNameState {
    name: String,
    nr: i64,
}

#[derive(Debug)]
struct VariableNames {
    namesdict: HashMap<String, (String, i64)>,
    states: HashMap<VariableId, VariableNameState>,
}

static VARIABLE_NAMES: LazyLock<Mutex<VariableNames>> = LazyLock::new(|| {
    let mut namesdict = HashMap::new();
    namesdict.insert(DUMMYNAME.to_owned(), (DUMMYNAME.to_owned(), 0));
    Mutex::new(VariableNames {
        namesdict,
        states: HashMap::new(),
    })
});

impl Variable {
    pub fn new(id: VariableId, kind: Kind) -> Self {
        Self {
            id,
            kind: Some(kind),
        }
    }

    pub fn new_untyped(id: VariableId) -> Self {
        Self { id, kind: None }
    }

    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Variable {
        mapping.get(self).copied().unwrap_or(*self)
    }

    pub fn name(&self) -> String {
        let mut names = VARIABLE_NAMES.lock().unwrap();
        let (prefix, mut nr) = {
            let state = names
                .states
                .entry(self.id)
                .or_insert_with(|| VariableNameState {
                    name: DUMMYNAME.to_owned(),
                    nr: -1,
                });
            (state.name.clone(), state.nr)
        };
        if nr == -1 {
            let entry = names
                .namesdict
                .entry(prefix.clone())
                .or_insert_with(|| (prefix.clone(), 0));
            nr = entry.1;
            entry.1 += 1;
            names
                .states
                .get_mut(&self.id)
                .expect("name state must exist")
                .nr = nr;
        }
        format!("{}{}", prefix, nr)
    }

    pub fn renamed(&self) -> bool {
        VARIABLE_NAMES
            .lock()
            .unwrap()
            .states
            .get(&self.id)
            .map(|state| state.name != DUMMYNAME)
            .unwrap_or(false)
    }

    pub fn rename(&self, name: &str) {
        let mut names = VARIABLE_NAMES.lock().unwrap();
        let current_name = names
            .states
            .entry(self.id)
            .or_insert_with(|| VariableNameState {
                name: DUMMYNAME.to_owned(),
                nr: -1,
            })
            .name
            .clone();
        if current_name != DUMMYNAME {
            return;
        }

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
        let entry = names
            .namesdict
            .entry(cleaned.clone())
            .or_insert_with(|| (cleaned.clone(), 0));
        let canonical = entry.0.clone();
        let state = names
            .states
            .get_mut(&self.id)
            .expect("name state must exist");
        state.name = canonical;
        state.nr = -1;
    }
}

/// Newtype wrapper so `VariableId` cannot be confused with raw register
/// indices. RPython uses `id(variable_obj)` for the same purpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariableId(pub u32);

/// `rpython/flowspace/model.py:466-471` atom sentinel for exception exits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Atom {
    LastException,
}

impl Atom {
    pub fn name(self) -> &'static str {
        match self {
            Atom::LastException => "last_exception",
        }
    }
}

/// Opaque constant that preserves Python-object-style identity instead of
/// collapsing to a stringly enum.
#[derive(Debug, Clone)]
pub struct OpaqueConstant {
    id: usize,
    repr: String,
}

impl OpaqueConstant {
    pub fn new(repr: impl Into<String>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            repr: repr.into(),
        }
    }

    pub fn repr(&self) -> &str {
        &self.repr
    }
}

impl PartialEq for OpaqueConstant {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for OpaqueConstant {}

impl Hash for OpaqueConstant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialOrd for OpaqueConstant {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpaqueConstant {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

/// `rpython/flowspace/model.py:354` `class Constant(Hashable)`.
///
/// pyre only needs a subset of RPython's full value space today: booleans,
/// signed ints, strings such as `'default'`, the `last_exception` atom, and
/// opaque identity-bearing host objects such as exception type handles.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ConstantValue {
    None,
    Bool(bool),
    Signed(i64),
    Str(String),
    Atom(Atom),
    Opaque(OpaqueConstant),
}

/// RPython `Constant(value, concretetype=None)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constant {
    pub value: ConstantValue,
    pub kind: Option<Kind>,
}

impl Constant {
    pub fn none() -> Self {
        Self::new(ConstantValue::None, Some(Kind::Ref))
    }

    pub fn new(value: ConstantValue, kind: Option<Kind>) -> Self {
        Self { value, kind }
    }

    pub fn bool(value: bool) -> Self {
        Self::new(ConstantValue::Bool(value), Some(Kind::Int))
    }

    pub fn signed(value: i64) -> Self {
        Self::new(ConstantValue::Signed(value), Some(Kind::Int))
    }

    pub fn string(value: impl Into<String>) -> Self {
        Self::new(ConstantValue::Str(value.into()), None)
    }

    pub fn atom(value: Atom) -> Self {
        Self::new(ConstantValue::Atom(value), None)
    }

    pub fn opaque(repr: impl Into<String>, kind: Option<Kind>) -> Self {
        Self::new(ConstantValue::Opaque(OpaqueConstant::new(repr)), kind)
    }

    pub fn replace(&self, _mapping: &HashMap<Variable, Variable>) -> Self {
        self.clone()
    }

    pub fn is_last_exception(&self) -> bool {
        matches!(self.value, ConstantValue::Atom(Atom::LastException))
    }
}

impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Constant {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value
            .cmp(&other.value)
            .then_with(|| kind_rank(self.kind).cmp(&kind_rank(other.kind)))
    }
}

fn kind_rank(kind: Option<Kind>) -> u8 {
    match kind {
        None => 0,
        Some(Kind::Int) => 1,
        Some(Kind::Ref) => 2,
        Some(Kind::Float) => 3,
    }
}

/// `rpython/flowspace/model.py:429-471` `c_last_exception`.
pub fn c_last_exception() -> Constant {
    Constant::atom(Atom::LastException)
}

/// Mixed var/const carrier for `Link.args` / `Block.inputargs`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FlowValue {
    Variable(Variable),
    Constant(Constant),
}

impl FlowValue {
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        match self {
            Self::Variable(v) => v.replace(mapping).into(),
            Self::Constant(c) => c.replace(mapping).into(),
        }
    }

    pub fn rename_with<F>(&self, rename: &mut F) -> Self
    where
        F: FnMut(Variable) -> Variable,
    {
        match self {
            Self::Variable(v) => rename(*v).into(),
            Self::Constant(c) => c.clone().into(),
        }
    }

    pub fn as_variable(&self) -> Option<Variable> {
        match self {
            Self::Variable(v) => Some(*v),
            Self::Constant(_) => None,
        }
    }

    pub fn as_constant(&self) -> Option<&Constant> {
        match self {
            Self::Variable(_) => None,
            Self::Constant(c) => Some(c),
        }
    }

    fn is_last_exception_sentinel(&self) -> bool {
        matches!(self, Self::Constant(c) if c.is_last_exception())
    }
}

impl From<Variable> for FlowValue {
    fn from(value: Variable) -> Self {
        Self::Variable(value)
    }
}

impl From<Constant> for FlowValue {
    fn from(value: Constant) -> Self {
        Self::Constant(value)
    }
}

/// `rpython/jit/codewriter/flatten.py:35-51` `class ListOfKind(object)`,
/// but at the pre-regalloc flow-graph stage where the contents are still
/// Variables / Constants instead of flattened registers.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FlowListOfKind {
    pub kind: Kind,
    pub content: Vec<FlowValue>,
}

impl FlowListOfKind {
    pub fn new(kind: Kind, content: Vec<FlowValue>) -> Self {
        Self { kind, content }
    }

    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        Self {
            kind: self.kind,
            content: self
                .content
                .iter()
                .map(|value| value.replace(mapping))
                .collect(),
        }
    }
}

/// Pointer-identity wrapper for `majit_ir::DescrRef` so a descr can sit
/// inside `SpaceOperationArg` despite `Arc<dyn Descr>` not deriving
/// `Eq`/`Hash`.  Mirrors the treatment in `rpython/jit/metainterp/
/// history.py` where `AbstractDescr` instances are singletons identified
/// by Python `is` — two descrs are "equal" only when they are the same
/// Python object.
#[derive(Debug, Clone)]
pub struct DescrByPtr(pub majit_ir::DescrRef);

impl PartialEq for DescrByPtr {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for DescrByPtr {}

impl std::hash::Hash for DescrByPtr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (std::sync::Arc::as_ptr(&self.0) as *const ()).hash(state);
    }
}

/// Pointer-identity wrapper around `flatten::IndirectCallTargets` —
/// `IndirectCallTargets` carries `Vec<Arc<JitCode>>` which does not
/// derive `Eq`/`Hash`, and two SSARepr sites pointing at the SAME
/// target list must dedup to the same assembler index
/// (`assembler.py:197-206`).  Matches the treatment upstream gives
/// `AbstractDescr` in `SpaceOperation.args` — identity by Python `is`,
/// not structural equality.
#[derive(Debug, Clone)]
pub struct IndirectCallTargetsByPtr(pub Rc<IndirectCallTargets>);

impl PartialEq for IndirectCallTargetsByPtr {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for IndirectCallTargetsByPtr {}

impl std::hash::Hash for IndirectCallTargetsByPtr {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.0) as *const ()).hash(state);
    }
}

/// `rpython/flowspace/model.py:436-439 SpaceOperation.args`.
///
/// Upstream is a duck-typed Python list that mixes `Variable`, `Constant`,
/// `ListOfKind`, `AbstractDescr`, and `IndirectCallTargets` — see
/// `flatten.py:358-370` where the serializer walks `op.args` and
/// dispatches by `isinstance` against those five types.  pyre represents
/// all five: `Value` covers `Variable | Constant`, `ListOfKind` wraps
/// `flow::FlowListOfKind`, `Descr` wraps `majit_ir::DescrRef` (upstream
/// `AbstractDescr`), and `IndirectCallTargets` wraps pyre's
/// `flatten::IndirectCallTargets` (upstream `IndirectCallTargets`).
///
/// PRE-EXISTING-ADAPTATION: Rust needs a concrete sum type.  We cannot
/// simply extend `FlowValue` to cover ListOfKind/Descr/IndirectCallTargets
/// because `FlowValue` is also the element type of `Block.inputargs`,
/// `SpaceOperation.result`, `Link.args`, `Link.exitcase`, and
/// `Link.last_exception` — positions where upstream forbids anything
/// but `Variable` (inputargs/result) or `Variable | Constant` (link
/// args).  Widening `FlowValue` would relax those contracts.  The
/// `SpaceOperationArg` sum is the minimal expression that keeps every
/// other slot typed as upstream demands.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SpaceOperationArg {
    Value(FlowValue),
    ListOfKind(FlowListOfKind),
    Descr(DescrByPtr),
    IndirectCallTargets(IndirectCallTargetsByPtr),
}

impl SpaceOperationArg {
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        match self {
            Self::Value(value) => Self::Value(value.replace(mapping)),
            Self::ListOfKind(list) => Self::ListOfKind(list.replace(mapping)),
            // `flatten.py:365-367` passes AbstractDescr and
            // IndirectCallTargets through unchanged.
            Self::Descr(descr) => Self::Descr(descr.clone()),
            Self::IndirectCallTargets(targets) => Self::IndirectCallTargets(targets.clone()),
        }
    }

    pub fn variables(&self) -> Vec<Variable> {
        match self {
            Self::Value(value) => value.as_variable().into_iter().collect(),
            Self::ListOfKind(list) => list
                .content
                .iter()
                .filter_map(FlowValue::as_variable)
                .collect(),
            Self::Descr(_) | Self::IndirectCallTargets(_) => Vec::new(),
        }
    }

    pub fn constants(&self) -> Vec<Constant> {
        match self {
            Self::Value(value) => value.as_constant().cloned().into_iter().collect(),
            Self::ListOfKind(list) => list
                .content
                .iter()
                .filter_map(|value| value.as_constant().cloned())
                .collect(),
            Self::Descr(_) | Self::IndirectCallTargets(_) => Vec::new(),
        }
    }
}

impl From<majit_ir::DescrRef> for SpaceOperationArg {
    fn from(descr: majit_ir::DescrRef) -> Self {
        Self::Descr(DescrByPtr(descr))
    }
}

impl From<Rc<IndirectCallTargets>> for SpaceOperationArg {
    fn from(targets: Rc<IndirectCallTargets>) -> Self {
        Self::IndirectCallTargets(IndirectCallTargetsByPtr(targets))
    }
}

impl From<FlowValue> for SpaceOperationArg {
    fn from(value: FlowValue) -> Self {
        Self::Value(value)
    }
}

impl From<Variable> for SpaceOperationArg {
    fn from(value: Variable) -> Self {
        Self::Value(value.into())
    }
}

impl From<Constant> for SpaceOperationArg {
    fn from(value: Constant) -> Self {
        Self::Value(value.into())
    }
}

impl From<FlowListOfKind> for SpaceOperationArg {
    fn from(value: FlowListOfKind) -> Self {
        Self::ListOfKind(value)
    }
}

/// `rpython/flowspace/model.py:434` `class SpaceOperation(object)`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SpaceOperation {
    pub opname: String,
    pub args: Vec<SpaceOperationArg>,
    pub result: Option<FlowValue>,
    pub offset: i64,
}

impl SpaceOperation {
    pub fn new(
        opname: impl Into<String>,
        args: Vec<SpaceOperationArg>,
        result: Option<FlowValue>,
        offset: i64,
    ) -> Self {
        Self {
            opname: opname.into(),
            args,
            result,
            offset,
        }
    }

    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        Self {
            opname: self.opname.clone(),
            args: self.args.iter().map(|arg| arg.replace(mapping)).collect(),
            result: self.result.as_ref().map(|result| result.replace(mapping)),
            offset: self.offset,
        }
    }
}

/// `flatten.py:249-259` special-case tuple exitswitch carrier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExitSwitchElement {
    Value(FlowValue),
    Marker(String),
}

impl ExitSwitchElement {
    fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        match self {
            Self::Value(value) => Self::Value(value.replace(mapping)),
            Self::Marker(marker) => Self::Marker(marker.clone()),
        }
    }
}

/// `rpython/flowspace/model.py:179` `block.exitswitch`.
///
/// RPython uses either a normal variable exitswitch, the special
/// `Constant(last_exception)` sentinel, or a tuple introduced by
/// `jtransform.optimize_goto_if_not`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExitSwitch {
    Value(FlowValue),
    Tuple(Vec<ExitSwitchElement>),
}

impl ExitSwitch {
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        match self {
            Self::Value(value) => Self::Value(value.replace(mapping)),
            Self::Tuple(values) => {
                Self::Tuple(values.iter().map(|value| value.replace(mapping)).collect())
            }
        }
    }
}

pub type LinkArg = Option<FlowValue>;

/// pyre-only adaptation: per-arg positional metadata recording which
/// `FrameState.mergeable()` entries fed this `Link.args[j]` and which
/// `target` mergeable entry it is meant to satisfy.
///
/// RPython keeps this correspondence implicitly in
/// `FrameState.getoutputargs()`; pyre's slot-keyed regalloc needs the
/// positions preserved explicitly so it can project `link.args[j] ↔
/// link.target.inputargs[j]` back onto register slots without
/// guessing from `Variable` identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct LinkArgPosition {
    pub source_mergeable_index: Option<usize>,
    pub target_mergeable_index: Option<usize>,
}

/// Shared block identity. RPython gets object identity for free; Rust uses
/// `Rc<RefCell<_>>` and pointer-identity equality.
#[derive(Debug, Clone)]
pub struct BlockRef(Rc<RefCell<Block>>);

impl BlockRef {
    fn new(block: Block) -> Self {
        Self(Rc::new(RefCell::new(block)))
    }

    pub fn borrow(&self) -> Ref<'_, Block> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, Block> {
        self.0.borrow_mut()
    }

    pub fn downgrade(&self) -> Weak<RefCell<Block>> {
        Rc::downgrade(&self.0)
    }

    pub fn input_arity(&self) -> usize {
        self.borrow().inputargs.len()
    }

    pub fn closeblock(&self, exits: Vec<LinkRef>) {
        assert!(self.borrow().exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    pub fn recloseblock(&self, exits: Vec<LinkRef>) {
        let weak = self.downgrade();
        for link in &exits {
            link.borrow_mut().prevblock = Some(weak.clone());
        }
        self.borrow_mut().exits = exits;
    }
}

impl PartialEq for BlockRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for BlockRef {}

impl Hash for BlockRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

/// Shared link identity. Mirrors `BlockRef`.
#[derive(Debug, Clone)]
pub struct LinkRef(Rc<RefCell<Link>>);

impl LinkRef {
    pub fn borrow(&self) -> Ref<'_, Link> {
        self.0.borrow()
    }

    pub fn borrow_mut(&self) -> RefMut<'_, Link> {
        self.0.borrow_mut()
    }
}

impl PartialEq for LinkRef {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for LinkRef {}

impl Hash for LinkRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state);
    }
}

/// `rpython/flowspace/model.py:109` `class Link(object)`.
///
/// Upstream:
/// ```py
/// class Link(object):
///     __slots__ = """args target exitcase llexitcase prevblock
///                 last_exception last_exc_value""".split()
/// ```
#[derive(Debug, Clone)]
pub struct Link {
    /// `link.args` — mixed list of Variables / Constants passed to `target`.
    pub args: Vec<LinkArg>,
    /// pyre-only: preserved `FrameState.getoutputargs()` positions for
    /// each `args[j]`.
    pub arg_positions: Vec<LinkArgPosition>,
    /// `link.target` — successor block, optional during graph construction.
    pub target: Option<BlockRef>,
    /// `link.exitcase`.
    pub exitcase: Option<FlowValue>,
    /// `link.llexitcase`.
    pub llexitcase: Option<FlowValue>,
    /// `link.prevblock`.
    pub prevblock: Option<Weak<RefCell<Block>>>,
    /// Exception-passing variables attached to the edge.
    pub last_exception: Option<Variable>,
    pub last_exc_value: Option<Variable>,
}

impl Link {
    pub fn new(
        args: Vec<FlowValue>,
        target: Option<BlockRef>,
        exitcase: Option<FlowValue>,
    ) -> Self {
        Self::new_mergeable(args.into_iter().map(Some).collect(), target, exitcase)
    }

    pub fn new_mergeable(
        args: Vec<LinkArg>,
        target: Option<BlockRef>,
        exitcase: Option<FlowValue>,
    ) -> Self {
        let arg_len = args.len();
        if let Some(target) = &target {
            assert_eq!(arg_len, target.input_arity(), "output args mismatch");
        }
        Self {
            args,
            arg_positions: vec![LinkArgPosition::default(); arg_len],
            target,
            exitcase,
            llexitcase: None,
            prevblock: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    pub fn with_exitcase(mut self, exitcase: FlowValue) -> Self {
        self.exitcase = Some(exitcase);
        self
    }

    pub fn with_llexitcase(mut self, llexitcase: FlowValue) -> Self {
        self.llexitcase = Some(llexitcase);
        self
    }

    pub fn with_arg_positions(mut self, arg_positions: Vec<LinkArgPosition>) -> Self {
        self.set_arg_positions(arg_positions);
        self
    }

    pub fn set_arg_positions(&mut self, arg_positions: Vec<LinkArgPosition>) {
        assert_eq!(
            self.args.len(),
            arg_positions.len(),
            "link arg positions mismatch"
        );
        self.arg_positions = arg_positions;
    }

    /// `model.py:127-129` `Link.extravars`.
    pub fn extravars(
        &mut self,
        last_exception: Option<Variable>,
        last_exc_value: Option<Variable>,
    ) {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
    }

    /// `model.py:131-138` `Link.getextravars`.
    pub fn getextravars(&self) -> Vec<Variable> {
        let mut result = Vec::new();
        if let Some(v) = self.last_exception {
            result.push(v);
        }
        if let Some(v) = self.last_exc_value {
            result.push(v);
        }
        result
    }

    /// `model.py:140-147` `Link.copy`.
    pub fn copy<F>(&self, mut rename: F) -> Self
    where
        F: FnMut(Variable) -> Variable,
    {
        let mut newlink = Self::new_mergeable(
            self.args
                .iter()
                .map(|arg| arg.as_ref().map(|value| value.rename_with(&mut rename)))
                .collect(),
            self.target.clone(),
            self.exitcase.clone(),
        );
        newlink.arg_positions = self.arg_positions.clone();
        newlink.prevblock = self.prevblock.clone();
        newlink.last_exception = self.last_exception.map(&mut rename);
        newlink.last_exc_value = self.last_exc_value.map(&mut rename);
        newlink.llexitcase = self.llexitcase.clone();
        newlink
    }

    /// `model.py:149-153` `Link.replace`.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Self {
        self.copy(|v| v.replace(mapping))
    }

    /// `model.py:156-159` `Link.settarget`.
    pub fn settarget(&mut self, targetblock: BlockRef) {
        assert_eq!(
            self.args.len(),
            targetblock.borrow().inputargs.len(),
            "output args mismatch"
        );
        self.target = Some(targetblock);
    }

    pub fn into_ref(self) -> LinkRef {
        LinkRef(Rc::new(RefCell::new(self)))
    }
}

/// `rpython/flowspace/model.py:171` `class Block(object)`.
///
#[derive(Debug)]
pub struct Block {
    /// `block.inputargs` — mixed list of variables / constants.
    pub inputargs: Vec<FlowValue>,
    /// `block.operations`.
    pub operations: Vec<SpaceOperation>,
    /// Final (return/except) blocks model RPython's `operations = ()`.
    pub is_final: bool,
    /// `block.exitswitch`.
    pub exitswitch: Option<ExitSwitch>,
    /// `block.exits`.
    pub exits: Vec<LinkRef>,
}

impl Block {
    pub fn new(inputargs: Vec<FlowValue>) -> Self {
        Self {
            inputargs,
            operations: Vec::new(),
            is_final: false,
            exitswitch: None,
            exits: Vec::new(),
        }
    }

    pub fn shared(inputargs: Vec<FlowValue>) -> BlockRef {
        BlockRef::new(Self::new(inputargs))
    }

    /// `rpython/flowspace/model.py:182` `def is_final_block(self): return
    /// self.operations == ()`.
    pub fn is_final_block(&self) -> bool {
        self.is_final
    }

    pub fn mark_final(&mut self) {
        self.operations.clear();
        self.is_final = true;
    }

    /// `model.py:216-217` `Block.canraise`.
    pub fn canraise(&self) -> bool {
        matches!(self.exitswitch, Some(ExitSwitch::Value(ref value)) if value.is_last_exception_sentinel())
    }

    /// `model.py:219-222` `Block.raising_op`.
    pub fn raising_op(&self) -> Option<&SpaceOperation> {
        if self.canraise() {
            self.operations.last()
        } else {
            None
        }
    }

    /// `model.py:224-231` `Block.getvariables`.
    pub fn getvariables(&self) -> Vec<Variable> {
        let mut result: Vec<Variable> = self
            .inputargs
            .iter()
            .filter_map(FlowValue::as_variable)
            .collect();
        for op in &self.operations {
            for arg in &op.args {
                result.extend(arg.variables());
            }
            if let Some(result_value) = &op.result {
                if let Some(variable) = result_value.as_variable() {
                    result.push(variable);
                }
            }
        }
        uniqueitems(result)
    }

    /// `model.py:233-239` `Block.getconstants`.
    pub fn getconstants(&self) -> Vec<Constant> {
        let mut result: Vec<Constant> = self
            .inputargs
            .iter()
            .filter_map(|value| value.as_constant().cloned())
            .collect();
        for op in &self.operations {
            for arg in &op.args {
                result.extend(arg.constants());
            }
        }
        uniqueitems(result)
    }

    /// `model.py:241-247` `Block.renamevariables`.
    pub fn renamevariables(&mut self, mapping: &HashMap<Variable, Variable>) {
        self.inputargs = self
            .inputargs
            .iter()
            .map(|arg| arg.replace(mapping))
            .collect();
        self.operations = self
            .operations
            .iter()
            .map(|op| op.replace(mapping))
            .collect();
        if let Some(exitswitch) = &self.exitswitch {
            self.exitswitch = Some(exitswitch.replace(mapping));
        }
        for link in &self.exits {
            let args = link
                .borrow()
                .args
                .iter()
                .map(|arg| arg.as_ref().map(|value| value.replace(mapping)))
                .collect();
            link.borrow_mut().args = args;
        }
    }
}

/// `rpython/flowspace/model.py:13` `class FunctionGraph(object)`.
///
/// Upstream:
/// ```py
/// class FunctionGraph(object):
///     def __init__(self, name, startblock, return_var=None):
///         self.name = name
///         self.startblock = startblock
///         ...
/// ```
///
/// pyre keeps owner storage for blocks it creates, while the external graph
/// surface uses `BlockRef` object references like RPython.
#[derive(Debug)]
pub struct FunctionGraph {
    pub name: String,
    pub startblock: BlockRef,
    pub returnblock: BlockRef,
    pub exceptblock: BlockRef,
    pub tag: Option<String>,
    /// Monotonic id generator for Variables allocated in this graph.
    next_variable_id: u32,
}

impl FunctionGraph {
    pub fn new(
        name: impl Into<String>,
        startblock: BlockRef,
        return_var: Option<Variable>,
    ) -> Self {
        let mut next_variable_id = 0;
        observe_values_for_next_id(&mut next_variable_id, &startblock.borrow().inputargs);
        if let Some(return_var) = return_var {
            next_variable_id = next_variable_id.max(return_var.id.0 + 1);
        }

        let mut graph = Self {
            name: name.into(),
            startblock: startblock.clone(),
            returnblock: Block::shared(Vec::new()),
            exceptblock: Block::shared(Vec::new()),
            tag: None,
            next_variable_id,
        };

        let return_var = return_var.unwrap_or_else(|| graph.fresh_untyped_variable());
        let except_etype = graph.fresh_untyped_variable();
        let except_evalue = graph.fresh_untyped_variable();

        let returnblock = Block::shared(vec![return_var.into()]);
        returnblock.borrow_mut().mark_final();
        let exceptblock = Block::shared(vec![except_etype.into(), except_evalue.into()]);
        exceptblock.borrow_mut().mark_final();

        graph.returnblock = returnblock;
        graph.exceptblock = exceptblock;
        graph
    }

    /// Allocate a fresh `Variable` of the given kind. Mirrors upstream
    /// `Variable(name=None)` which also creates a new identity.
    pub fn fresh_variable(&mut self, kind: Kind) -> Variable {
        let id = VariableId(self.next_variable_id);
        self.next_variable_id += 1;
        Variable::new(id, kind)
    }

    pub fn fresh_untyped_variable(&mut self) -> Variable {
        let id = VariableId(self.next_variable_id);
        self.next_variable_id += 1;
        Variable::new_untyped(id)
    }

    /// Allocate and own a fresh `Block`.
    pub fn new_block(&mut self, inputargs: Vec<FlowValue>) -> BlockRef {
        observe_values_for_next_id(&mut self.next_variable_id, &inputargs);
        Block::shared(inputargs)
    }

    /// `model.py:28-32` `getargs` / `getreturnvar`.
    pub fn getargs(&self) -> Vec<FlowValue> {
        self.startblock.borrow().inputargs.clone()
    }

    pub fn getreturnvar(&self) -> Variable {
        self.returnblock.borrow().inputargs[0]
            .as_variable()
            .expect("returnblock inputarg should be a Variable")
    }

    /// `model.py:79-82` `iterblockops`.
    pub fn iterblockops(&self) -> Vec<(BlockRef, SpaceOperation)> {
        let mut out = Vec::new();
        for block in self.iterblocks() {
            for op in &block.borrow().operations {
                out.push((block.clone(), op.clone()));
            }
        }
        out
    }

    /// `rpython/flowspace/model.py:66-77` `FunctionGraph.iterblocks`.
    pub fn iterblocks(&self) -> Vec<BlockRef> {
        let start = self.startblock.clone();
        let mut out = vec![start.clone()];
        let mut seen = vec![start.clone()];
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();

        while let Some(link) = stack.pop() {
            let Some(block) = link.borrow().target.clone() else {
                continue;
            };
            if seen.contains(&block) {
                continue;
            }
            seen.push(block.clone());
            out.push(block.clone());
            let more: Vec<LinkRef> = block.borrow().exits.iter().rev().cloned().collect();
            stack.extend(more);
        }

        out
    }

    /// `rpython/flowspace/model.py:78-88` `FunctionGraph.iterlinks`.
    pub fn iterlinks(&self) -> Vec<LinkRef> {
        let start = self.startblock.clone();
        let mut out = Vec::new();
        let mut seen = vec![start.clone()];
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();

        while let Some(link) = stack.pop() {
            let target = link.borrow().target.clone();
            out.push(link);
            let Some(block) = target else {
                continue;
            };
            if !seen.contains(&block) {
                seen.push(block.clone());
                let more: Vec<LinkRef> = block.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }

        out
    }
}

fn observe_values_for_next_id(next_variable_id: &mut u32, values: &[FlowValue]) {
    for value in values {
        if let Some(variable) = value.as_variable() {
            *next_variable_id = (*next_variable_id).max(variable.id.0 + 1);
        }
    }
}

fn uniqueitems<T>(values: Vec<T>) -> Vec<T>
where
    T: Eq + Clone,
{
    let mut result = Vec::new();
    for value in values {
        if !result.contains(&value) {
            result.push(value);
        }
    }
    result
}

fn copy_value(
    value: &FlowValue,
    varmap: &mut HashMap<Variable, Variable>,
    next_variable_id: &mut u32,
    shallowvars: bool,
) -> FlowValue {
    match value {
        FlowValue::Variable(variable) => {
            if shallowvars {
                (*variable).into()
            } else if let Some(mapped) = varmap.get(variable) {
                (*mapped).into()
            } else {
                let new_variable = Variable {
                    id: VariableId(*next_variable_id),
                    kind: variable.kind,
                };
                *next_variable_id += 1;
                varmap.insert(*variable, new_variable);
                new_variable.into()
            }
        }
        FlowValue::Constant(constant) => constant.clone().into(),
    }
}

fn copy_optional_value(
    value: Option<&FlowValue>,
    varmap: &mut HashMap<Variable, Variable>,
    next_variable_id: &mut u32,
    shallowvars: bool,
) -> Option<FlowValue> {
    value.map(|value| copy_value(value, varmap, next_variable_id, shallowvars))
}

fn copy_space_operation_arg(
    arg: &SpaceOperationArg,
    varmap: &mut HashMap<Variable, Variable>,
    next_variable_id: &mut u32,
    shallowvars: bool,
) -> SpaceOperationArg {
    match arg {
        SpaceOperationArg::Value(value) => {
            copy_value(value, varmap, next_variable_id, shallowvars).into()
        }
        SpaceOperationArg::ListOfKind(list) => FlowListOfKind::new(
            list.kind,
            list.content
                .iter()
                .map(|value| copy_value(value, varmap, next_variable_id, shallowvars))
                .collect(),
        )
        .into(),
        // `model.py:231-233` `Block.copy` passes AbstractDescr and
        // IndirectCallTargets through unchanged — both carry
        // graph-external identity and are shared by pointer across
        // clones.
        SpaceOperationArg::Descr(descr) => SpaceOperationArg::Descr(descr.clone()),
        SpaceOperationArg::IndirectCallTargets(targets) => {
            SpaceOperationArg::IndirectCallTargets(targets.clone())
        }
    }
}

fn copy_exitswitch(
    exitswitch: Option<&ExitSwitch>,
    varmap: &mut HashMap<Variable, Variable>,
    next_variable_id: &mut u32,
    shallowvars: bool,
) -> Option<ExitSwitch> {
    exitswitch.map(|exitswitch| match exitswitch {
        ExitSwitch::Value(value) => {
            ExitSwitch::Value(copy_value(value, varmap, next_variable_id, shallowvars))
        }
        ExitSwitch::Tuple(values) => ExitSwitch::Tuple(
            values
                .iter()
                .map(|value| match value {
                    ExitSwitchElement::Value(value) => ExitSwitchElement::Value(copy_value(
                        value,
                        varmap,
                        next_variable_id,
                        shallowvars,
                    )),
                    ExitSwitchElement::Marker(marker) => ExitSwitchElement::Marker(marker.clone()),
                })
                .collect(),
        ),
    })
}

/// `model.py:262-268` `mkentrymap`.
pub fn mkentrymap(funcgraph: &FunctionGraph) -> HashMap<BlockRef, Vec<LinkRef>> {
    let startlink = Link::new(
        funcgraph.getargs(),
        Some(funcgraph.startblock.clone()),
        None,
    )
    .into_ref();
    let mut result = HashMap::from([(funcgraph.startblock.clone(), vec![startlink])]);
    for link in funcgraph.iterlinks() {
        let Some(target) = link.borrow().target.clone() else {
            continue;
        };
        result.entry(target).or_default().push(link);
    }
    result
}

/// `model.py:269-323` `copygraph`.
pub fn copygraph(
    graph: &FunctionGraph,
    shallow: bool,
    varmap: &HashMap<Variable, Variable>,
    shallowvars: bool,
) -> FunctionGraph {
    let mut blockmap: HashMap<BlockRef, BlockRef> = HashMap::new();
    let mut local_varmap = varmap.clone();
    let shallowvars = shallowvars || shallow;
    let mut next_variable_id = graph.next_variable_id;

    let copy_block = |block: &BlockRef,
                      local_varmap: &mut HashMap<Variable, Variable>,
                      next_variable_id: &mut u32,
                      shallowvars: bool| {
        let block_borrow = block.borrow();
        let inputargs = block_borrow
            .inputargs
            .iter()
            .map(|value| copy_value(value, local_varmap, next_variable_id, shallowvars))
            .collect();
        let mut newblock = Block::new(inputargs);
        newblock.is_final = block_borrow.is_final;
        newblock.operations = if shallow {
            block_borrow.operations.clone()
        } else {
            block_borrow
                .operations
                .iter()
                .map(|op| SpaceOperation {
                    opname: op.opname.clone(),
                    args: op
                        .args
                        .iter()
                        .map(|arg| {
                            copy_space_operation_arg(
                                arg,
                                local_varmap,
                                next_variable_id,
                                shallowvars,
                            )
                        })
                        .collect(),
                    result: copy_optional_value(
                        op.result.as_ref(),
                        local_varmap,
                        next_variable_id,
                        shallowvars,
                    ),
                    offset: op.offset,
                })
                .collect()
        };
        newblock.exitswitch = copy_exitswitch(
            block_borrow.exitswitch.as_ref(),
            local_varmap,
            next_variable_id,
            shallowvars,
        );
        let shared = Block::shared(newblock.inputargs.clone());
        {
            let mut shared_borrow = shared.borrow_mut();
            shared_borrow.operations = newblock.operations;
            shared_borrow.is_final = newblock.is_final;
            shared_borrow.exitswitch = newblock.exitswitch;
        }
        shared
    };

    for block in graph.iterblocks() {
        let newblock = copy_block(
            &block,
            &mut local_varmap,
            &mut next_variable_id,
            shallowvars,
        );
        blockmap.insert(block, newblock);
    }

    if !blockmap.contains_key(&graph.returnblock) {
        let newblock = copy_block(
            &graph.returnblock,
            &mut local_varmap,
            &mut next_variable_id,
            shallowvars,
        );
        blockmap.insert(graph.returnblock.clone(), newblock);
    }
    if !blockmap.contains_key(&graph.exceptblock) {
        let newblock = copy_block(
            &graph.exceptblock,
            &mut local_varmap,
            &mut next_variable_id,
            shallowvars,
        );
        blockmap.insert(graph.exceptblock.clone(), newblock);
    }

    for (block, newblock) in &blockmap {
        let old_links = block.borrow().exits.clone();
        let mut newlinks = Vec::with_capacity(old_links.len());
        for link in old_links {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .as_ref()
                .and_then(|target| blockmap.get(target).cloned());
            let mut newlink = Link::new_mergeable(
                link_borrow
                    .args
                    .iter()
                    .map(|arg| {
                        copy_optional_value(
                            arg.as_ref(),
                            &mut local_varmap,
                            &mut next_variable_id,
                            shallowvars,
                        )
                    })
                    .collect(),
                target,
                copy_optional_value(
                    link_borrow.exitcase.as_ref(),
                    &mut local_varmap,
                    &mut next_variable_id,
                    shallowvars,
                ),
            );
            newlink.llexitcase = copy_optional_value(
                link_borrow.llexitcase.as_ref(),
                &mut local_varmap,
                &mut next_variable_id,
                shallowvars,
            );
            newlink.last_exception = link_borrow.last_exception.map(|variable| {
                copy_value(
                    &variable.into(),
                    &mut local_varmap,
                    &mut next_variable_id,
                    shallowvars,
                )
                .as_variable()
                .expect("copied last_exception should stay a Variable")
            });
            newlink.last_exc_value = link_borrow.last_exc_value.map(|variable| {
                copy_value(
                    &variable.into(),
                    &mut local_varmap,
                    &mut next_variable_id,
                    shallowvars,
                )
                .as_variable()
                .expect("copied last_exc_value should stay a Variable")
            });
            newlinks.push(newlink.into_ref());
        }
        newblock.closeblock(newlinks);
    }

    FunctionGraph {
        name: graph.name.clone(),
        startblock: blockmap[&graph.startblock].clone(),
        returnblock: blockmap[&graph.returnblock].clone(),
        exceptblock: blockmap[&graph.exceptblock].clone(),
        tag: graph.tag.clone(),
        next_variable_id,
    }
}

/// `model.py:439-446` `summary`.
pub fn summary(graph: &FunctionGraph) -> HashMap<String, usize> {
    let mut insns = HashMap::new();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            if op.opname != "same_as" {
                *insns.entry(op.opname.clone()).or_insert(0) += 1;
            }
        }
    }
    insns
}

/// Register an operation inside a block.
///
/// Thin wrapper so the eventual walker integration has a stable hook to call.
pub fn push_op(block: &BlockRef, op: SpaceOperation) {
    block.borrow_mut().operations.push(op);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flatten::Kind;

    #[test]
    fn variable_identity_by_id() {
        let v0 = Variable::new(VariableId(0), Kind::Ref);
        let v1 = Variable::new(VariableId(1), Kind::Ref);
        assert_ne!(v0, v1);
        assert_eq!(v0, Variable::new(VariableId(0), Kind::Ref));
    }

    #[test]
    fn function_graph_initializes_start_return_except_blocks() {
        let arg = Variable::new(VariableId(0), Kind::Ref);
        let ret = Variable::new(VariableId(1), Kind::Int);
        let g = FunctionGraph::new("t", Block::shared(vec![arg.into()]), Some(ret));
        assert_eq!(g.getargs(), vec![FlowValue::from(arg)]);
        assert_eq!(g.getreturnvar(), ret);
        assert_eq!(g.startblock.borrow().inputargs, vec![arg.into()]);
        assert_eq!(g.returnblock.borrow().inputargs, vec![ret.into()]);
        assert_eq!(g.exceptblock.borrow().inputargs.len(), 2);
        assert!(
            g.exceptblock
                .borrow()
                .inputargs
                .iter()
                .all(|value| value.as_variable().is_some_and(|v| v.kind.is_none()))
        );
        assert_eq!(g.iterblocks(), vec![g.startblock.clone()]);
        assert!(g.iterlinks().is_empty());
    }

    #[test]
    fn closeblock_sets_prevblock_and_preserves_target() {
        let mut g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let v = g.fresh_variable(Kind::Ref);
        let b = g.new_block(vec![v.into()]);
        let start = g.startblock.clone();
        start.closeblock(vec![
            Link::new(vec![v.into()], Some(b.clone()), None).into_ref(),
        ]);
        let start_borrow = start.borrow();
        let link = start_borrow.exits[0].borrow();
        assert_eq!(link.args, vec![Some(v.into())]);
        assert_eq!(link.arg_positions, vec![LinkArgPosition::default()]);
        assert_eq!(link.target, Some(b));
        assert!(
            link.prevblock
                .as_ref()
                .is_some_and(|prev| Weak::ptr_eq(prev, &start.downgrade()))
        );
    }

    #[test]
    fn link_extravars_round_trip() {
        let v0 = Variable::new(VariableId(0), Kind::Ref);
        let v1 = Variable::new(VariableId(1), Kind::Ref);
        let mut link = Link::new(vec![], None, None);
        link.extravars(Some(v0), Some(v1));
        assert_eq!(link.getextravars(), vec![v0, v1]);
    }

    #[test]
    fn link_exitcase_supports_switch_and_bool_shapes() {
        let link = Link::new(vec![], None, Some(Constant::string("default").into()))
            .with_llexitcase(Constant::signed(7).into());
        assert_eq!(link.exitcase, Some(Constant::string("default").into()));
        assert_eq!(link.llexitcase, Some(Constant::signed(7).into()));
        assert!(Constant::bool(false) < Constant::bool(true));
    }

    #[test]
    fn link_copy_and_replace_preserve_extras() {
        let v0 = Variable::new(VariableId(0), Kind::Ref);
        let v1 = Variable::new(VariableId(1), Kind::Ref);
        let v2 = Variable::new(VariableId(2), Kind::Ref);
        let block_ref = Block::shared(vec![v0.into()]);
        let mut link = Link::new(
            vec![v0.into()],
            Some(block_ref.clone()),
            Some(Constant::bool(true).into()),
        )
        .with_llexitcase(Constant::bool(true).into());
        link.prevblock = Some(block_ref.downgrade());
        link.extravars(Some(v0), Some(v1));

        let copied = link.copy(|v| if v == v0 { v2 } else { v });
        assert_eq!(copied.args, vec![Some(v2.into())]);
        assert_eq!(copied.arg_positions, vec![LinkArgPosition::default()]);
        assert_eq!(copied.last_exception, Some(v2));
        assert_eq!(copied.last_exc_value, Some(v1));
        assert!(
            copied
                .prevblock
                .as_ref()
                .is_some_and(|prev| Weak::ptr_eq(prev, &block_ref.downgrade()))
        );
        assert_eq!(copied.exitcase, Some(Constant::bool(true).into()));
        assert_eq!(copied.llexitcase, Some(Constant::bool(true).into()));

        let replaced = link.replace(&HashMap::from([(v1, v2)]));
        assert_eq!(replaced.last_exception, Some(v0));
        assert_eq!(replaced.last_exc_value, Some(v2));
    }

    #[test]
    #[should_panic(expected = "output args mismatch")]
    fn link_new_arity_mismatch_panics() {
        let mut g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let v = g.fresh_variable(Kind::Ref);
        let b = g.new_block(vec![v.into()]);
        let _ = Link::new(vec![], Some(b), None);
    }

    #[test]
    fn link_args_accept_constants() {
        let mut g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let target = g.new_block(vec![Variable::new(VariableId(9), Kind::Int).into()]);
        let link = Link::new(vec![Constant::signed(42).into()], Some(target), None);
        assert_eq!(link.args, vec![Some(Constant::signed(42).into())]);
        assert_eq!(link.arg_positions, vec![LinkArgPosition::default()]);
    }

    #[test]
    fn link_arg_positions_round_trip() {
        let v0 = Variable::new(VariableId(0), Kind::Ref);
        let target = Block::shared(vec![v0.into()]);
        let link = Link::new(vec![v0.into()], Some(target), None).with_arg_positions(vec![
            LinkArgPosition {
                source_mergeable_index: Some(3),
                target_mergeable_index: Some(5),
            },
        ]);
        assert_eq!(
            link.arg_positions,
            vec![LinkArgPosition {
                source_mergeable_index: Some(3),
                target_mergeable_index: Some(5),
            }],
        );
    }

    #[test]
    fn iterblocks_dfs_order() {
        let mut g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let a = g.startblock.clone();
        let b = g.new_block(vec![]);
        let c = g.new_block(vec![]);
        a.closeblock(vec![
            Link::new(vec![], Some(b.clone()), None).into_ref(),
            Link::new(vec![], Some(c.clone()), None).into_ref(),
        ]);
        b.closeblock(vec![Link::new(vec![], Some(c.clone()), None).into_ref()]);
        assert_eq!(g.iterblocks(), vec![a, b, c]);
    }

    #[test]
    fn iterlinks_follow_rpython_order() {
        let mut g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let a = g.startblock.clone();
        let b = g.new_block(vec![]);
        let c = g.new_block(vec![]);
        a.closeblock(vec![
            Link::new(vec![], Some(b.clone()), None).into_ref(),
            Link::new(vec![], Some(c.clone()), None).into_ref(),
        ]);
        b.closeblock(vec![Link::new(vec![], Some(c.clone()), None).into_ref()]);
        let targets: Vec<BlockRef> = g
            .iterlinks()
            .into_iter()
            .map(|link| {
                link.borrow()
                    .target
                    .clone()
                    .expect("graph links have targets")
            })
            .collect();
        assert_eq!(targets, vec![b, c.clone(), c]);
    }

    #[test]
    fn iterblockops_follows_iterblocks_order() {
        let mut g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let a = g.startblock.clone();
        let b = g.new_block(vec![]);
        a.closeblock(vec![Link::new(vec![], Some(b.clone()), None).into_ref()]);
        push_op(&a, SpaceOperation::new("a", vec![], None, -1));
        push_op(&b, SpaceOperation::new("b", vec![], None, -1));
        let ops: Vec<(BlockRef, String)> = g
            .iterblockops()
            .into_iter()
            .map(|(block, op)| (block, op.opname))
            .collect();
        assert_eq!(ops, vec![(a, "a".to_string()), (b, "b".to_string())]);
    }

    #[test]
    fn mkentrymap_adds_synthetic_start_link() {
        let arg = Variable::new(VariableId(0), Kind::Ref);
        let mut g = FunctionGraph::new("t", Block::shared(vec![arg.into()]), None);
        let next = g.new_block(vec![arg.into()]);
        g.startblock.closeblock(vec![
            Link::new(vec![arg.into()], Some(next.clone()), None).into_ref(),
        ]);

        let entrymap = mkentrymap(&g);
        let start_links = entrymap
            .get(&g.startblock)
            .expect("startblock should have an entry");
        assert_eq!(start_links.len(), 1);
        let start_link = start_links[0].borrow();
        assert_eq!(start_link.args, vec![Some(arg.into())]);
        assert_eq!(start_link.target, Some(g.startblock.clone()));

        let incoming = entrymap
            .get(&next)
            .expect("next block should have incoming links");
        assert_eq!(incoming.len(), 1);
        assert!(
            incoming[0]
                .borrow()
                .prevblock
                .as_ref()
                .is_some_and(|prev| Weak::ptr_eq(prev, &g.startblock.downgrade()))
        );
    }

    #[test]
    fn copygraph_creates_distinct_blocks_links_and_variables() {
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let return_var = Variable::new(VariableId(1), Kind::Ref);
        let mut graph = FunctionGraph::new(
            "copy",
            Block::shared(vec![start_arg.into()]),
            Some(return_var),
        );
        let mid_arg = graph.fresh_variable(Kind::Ref);
        let result = graph.fresh_variable(Kind::Ref);
        let mid = graph.new_block(vec![mid_arg.into()]);
        push_op(
            &mid,
            SpaceOperation::new("same_as", vec![mid_arg.into()], Some(result.into()), 7),
        );
        graph.startblock.closeblock(vec![
            Link::new(vec![start_arg.into()], Some(mid.clone()), None).into_ref(),
        ]);
        mid.closeblock(vec![
            Link::new(vec![result.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let copied = copygraph(&graph, false, &HashMap::new(), false);
        let copied_mid = copied.startblock.borrow().exits[0]
            .borrow()
            .target
            .clone()
            .expect("copied mid block");

        assert_ne!(copied.startblock, graph.startblock);
        assert_ne!(copied_mid, mid);
        assert_ne!(copied.returnblock, graph.returnblock);
        assert_ne!(
            copied.startblock.borrow().inputargs[0].as_variable(),
            Some(start_arg)
        );
        assert_ne!(
            copied.returnblock.borrow().inputargs[0].as_variable(),
            Some(return_var)
        );
        assert_eq!(
            copied_mid.borrow().operations[0].args[0].variables()[0].kind,
            Some(Kind::Ref)
        );
        assert_eq!(
            copied_mid.borrow().operations[0]
                .result
                .as_ref()
                .and_then(FlowValue::as_variable)
                .map(|v| v.kind),
            Some(Some(Kind::Ref))
        );
        assert!(
            copied_mid.borrow().exits[0]
                .borrow()
                .prevblock
                .as_ref()
                .is_some_and(|prev| Weak::ptr_eq(prev, &copied_mid.downgrade()))
        );
    }

    #[test]
    fn summary_counts_ops_except_same_as() {
        let g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let start = g.startblock.clone();
        push_op(&start, SpaceOperation::new("same_as", vec![], None, -1));
        push_op(&start, SpaceOperation::new("int_add", vec![], None, -1));
        push_op(&start, SpaceOperation::new("int_add", vec![], None, -1));
        push_op(&start, SpaceOperation::new("ref_copy", vec![], None, -1));

        let counts = summary(&g);
        assert_eq!(counts.get("int_add"), Some(&2));
        assert_eq!(counts.get("ref_copy"), Some(&1));
        assert!(!counts.contains_key("same_as"));
    }

    #[test]
    fn canraise_and_raising_op_follow_c_last_exception() {
        let mut block = Block::new(vec![]);
        block
            .operations
            .push(SpaceOperation::new("overflowing_add_ovf", vec![], None, -1));
        assert!(!block.canraise());
        assert!(block.raising_op().is_none());

        block.exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));
        assert!(block.canraise());
        assert_eq!(
            block.raising_op().expect("raising op expected").opname,
            "overflowing_add_ovf"
        );
    }

    #[test]
    fn block_getvariables_and_constants_follow_listofkind_args() {
        let v0 = Variable::new(VariableId(0), Kind::Int);
        let v1 = Variable::new(VariableId(1), Kind::Int);
        let block = Block::shared(vec![v0.into()]);
        push_op(
            &block,
            SpaceOperation::new(
                "jit_merge_point",
                vec![
                    FlowListOfKind::new(
                        Kind::Int,
                        vec![v0.into(), Constant::signed(7).into(), v1.into()],
                    )
                    .into(),
                ],
                None,
                0,
            ),
        );

        let block_borrow = block.borrow();
        assert_eq!(block_borrow.getvariables(), vec![v0, v1]);
        assert_eq!(block_borrow.getconstants(), vec![Constant::signed(7)]);
    }

    #[test]
    fn closeblock_and_recloseblock_set_prevblock() {
        let block = Block::shared(vec![]);
        let target1 = Block::shared(vec![]);
        let target2 = Block::shared(vec![]);

        block.closeblock(vec![Link::new(vec![], Some(target1), None).into_ref()]);
        {
            let block_borrow = block.borrow();
            let link = block_borrow.exits[0].borrow();
            assert!(
                link.prevblock
                    .as_ref()
                    .is_some_and(|prev| Weak::ptr_eq(prev, &block.downgrade()))
            );
        }

        block.recloseblock(vec![
            Link::new(vec![], Some(target2.clone()), None).into_ref(),
        ]);
        let block_borrow = block.borrow();
        assert_eq!(block_borrow.exits.len(), 1);
        let link = block_borrow.exits[0].borrow();
        assert_eq!(link.target, Some(target2));
        assert!(
            link.prevblock
                .as_ref()
                .is_some_and(|prev| Weak::ptr_eq(prev, &block.downgrade()))
        );
    }

    #[test]
    fn push_op_appends_to_block() {
        let g = FunctionGraph::new("t", Block::shared(vec![]), None);
        let a = g.startblock.clone();
        push_op(&a, SpaceOperation::new("ref_return", vec![], None, -1));
        assert_eq!(a.borrow().operations.len(), 1);
    }

    #[test]
    fn getvariables_getconstants_and_renamevariables_follow_model_py() {
        let v0 = Variable::new(VariableId(0), Kind::Ref);
        let v1 = Variable::new(VariableId(1), Kind::Ref);
        let v2 = Variable::new(VariableId(2), Kind::Ref);
        let block = Block::shared(vec![v0.into(), Constant::signed(7).into()]);
        push_op(
            &block,
            SpaceOperation::new(
                "ref_copy",
                vec![v0.into(), Constant::signed(3).into()],
                Some(v1.into()),
                12,
            ),
        );
        let exit_link =
            Link::new(vec![v1.into()], Some(Block::shared(vec![v2.into()])), None).into_ref();
        block.closeblock(vec![exit_link.clone()]);

        {
            let block_borrow = block.borrow();
            assert_eq!(block_borrow.getvariables(), vec![v0, v1]);
            assert_eq!(
                block_borrow.getconstants(),
                vec![Constant::signed(7), Constant::signed(3)]
            );
        }

        let mapping = HashMap::from([(v0, v2), (v1, v0)]);
        block.borrow_mut().renamevariables(&mapping);
        let block_borrow = block.borrow();
        assert_eq!(
            block_borrow.inputargs,
            vec![v2.into(), Constant::signed(7).into()]
        );
        assert_eq!(
            block_borrow.operations[0].args,
            vec![v2.into(), Constant::signed(3).into()]
        );
        assert_eq!(block_borrow.operations[0].result, Some(v0.into()));
        assert_eq!(exit_link.borrow().args, vec![Some(v0.into())]);
    }

    #[test]
    fn opaque_constants_preserve_identity() {
        let left = Constant::opaque("OverflowError", None);
        let right = Constant::opaque("OverflowError", None);
        assert_ne!(left, right);
        let cloned = left.clone();
        assert_eq!(left, cloned);
        assert_eq!(
            left.value,
            ConstantValue::Opaque(OpaqueConstant {
                id: match &left.value {
                    ConstantValue::Opaque(value) => value.id,
                    _ => unreachable!(),
                },
                repr: "OverflowError".to_string(),
            })
        );
    }
}
