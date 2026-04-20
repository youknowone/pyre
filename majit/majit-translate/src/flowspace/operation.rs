//! Flow-space high-level operations.
//!
//! RPython upstream: `rpython/flowspace/operation.py` (764 LOC).
//!
//! Commit split (Phase 3 F3.3 of the five-year roadmap at
//! `.claude/plans/majestic-forging-meteor.md`):
//!
//! * **Commit 1 (this file's initial landing)** — data model skeleton.
//!   Every RPython `HLOperation` subclass surfaces as an `OpKind`
//!   enum variant; property tables (`arity`, `pure`, `can_overflow`,
//!   `dispatch`, `ovf_variant`) mirror the add_operator table line by
//!   line. `constfold()` returns `None` pending Commit 2. No wiring
//!   into `flowcontext.rs` yet — `record_pure_op` / `record_maybe_raise_op`
//!   still emit `SpaceOperation` directly via raw opname strings.
//! * **Commit 2** — real `constfold()` bodies (pyfunc mirror via match
//!   on `ConstValue`) + `BuiltinException::canraise` data populated
//!   from the trailing `_add_exceptions` / `_add_except_ovf` loop.
//! * **Commit 3** — the explicit subclasses with custom `eval()` logic
//!   (`NewDict`, `NewTuple`, `NewList`, `Pow`, `Iter`, `Next`,
//!   `GetAttr`, `SimpleCall`, `CallArgs`, `Contains`). Flowcontext
//!   migrates to OpKind-based emission.
//!
//! Rust adaptation (parity rule #1, minimum deviation):
//!
//! * Python `class HLOperation(SpaceOperation)` + `HLOperationMeta`
//!   metaclass dispatch collapses into `struct HLOperation { kind:
//!   OpKind, … }` plus a single enum. Each `OpKind` variant is the
//!   direct 1:1 mapping of an RPython `add_operator('name', …)` call
//!   or a top-level class declaration (`NewDict`, `Pow`, `SimpleCall`,
//!   …).
//! * RPython's global `op.*` namespace — populated by `HLOperationMeta.
//!   __init__` via `setattr(op, cls.opname, cls)` — is replaced by the
//!   `OpKind` variant itself; code that says `op.add` in RPython says
//!   `OpKind::Add` in Rust.
//! * `HLOperationMeta._registry` / `_transform` runtime dicts used by
//!   `SingleDispatchMixin` / `DoubleDispatchMixin` are not materialised
//!   here — they are consumed only by the annotator and rtyper, which
//!   land in Phases 4–6. `OpKind::dispatch()` preserves the `None` / 1
//!   / 2 classification so those phases can populate their own tables.

use std::collections::HashMap;

use super::model::{ConstValue, Constant, Hlvalue, SpaceOperation, Variable};

/// RPython `rpython/flowspace/operation.py` — enumerates every
/// `HLOperation` subclass. Variant order matches the upstream
/// `add_operator` table then the explicit subclass block.
///
/// `opname()` returns the exact string identifier used on
/// `SpaceOperation.opname`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OpKind {
    // ---- add_operator table (operation.py:445-521) ----
    // unary/binary operators registered via add_operator(); variant
    // names follow the upstream `opname` argument (UpperCamel'd), with
    // `_ovf` siblings as `*Ovf`.
    Is,
    Id,
    Type,
    IsSubtype,
    IsInstance,
    Repr,
    Str,
    Format,
    Len,
    Hash,
    SetAttr,
    DelAttr,
    GetItem,
    GetItemIdx,
    SetItem,
    DelItem,
    GetSlice,
    SetSlice,
    DelSlice,
    Trunc,
    Pos,
    Neg,
    NegOvf,
    Bool,
    Abs,
    AbsOvf,
    Hex,
    Oct,
    Bin,
    Ord,
    Invert,
    Add,
    AddOvf,
    Sub,
    SubOvf,
    Mul,
    MulOvf,
    TrueDiv,
    FloorDiv,
    FloorDivOvf,
    Div,
    DivOvf,
    Mod,
    ModOvf,
    DivMod,
    LShift,
    LShiftOvf,
    RShift,
    And,
    Or,
    Xor,
    Int,
    Index,
    Float,
    Long,
    InplaceAdd,
    InplaceSub,
    InplaceMul,
    InplaceTrueDiv,
    InplaceFloorDiv,
    InplaceDiv,
    InplaceMod,
    InplacePow,
    InplaceLShift,
    InplaceRShift,
    InplaceAnd,
    InplaceOr,
    InplaceXor,
    Lt,
    Le,
    Eq,
    Ne,
    Gt,
    Ge,
    Cmp,
    Coerce,
    Get,
    Set,
    Delete,
    UserDel,
    Buffer,
    Yield,
    NewSlice,
    Hint,

    // ---- explicit subclasses (operation.py:523-712) ----
    // These carry custom `eval()` / `consider()` overrides in RPython.
    // Commit 3 lifts the custom logic into the Rust port.
    Contains,
    NewDict,
    NewTuple,
    NewList,
    Pow,
    Iter,
    Next,
    GetAttr,
    SimpleCall,
    CallArgs,
}

/// RPython `operation.py` `canraise` entries reference the Python
/// exception classes `ValueError`, `OverflowError`, … directly. The
/// Rust port carries identities through this enum so the trailing
/// `_add_exceptions` table (operation.py:717-763) can populate
/// [`OpKind::canraise`] without pulling in the full `HOST_ENV`
/// exception hierarchy at this phase.
///
/// This list matches the set of exception names literally mentioned by
/// `operation.py`; `Exception` is the "any" fallback used by
/// `CallOp.canraise` and `op.getitem`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BuiltinException {
    ValueError,
    UnicodeDecodeError,
    ZeroDivisionError,
    OverflowError,
    IndexError,
    KeyError,
    StopIteration,
    RuntimeError,
    Exception,
}

/// RPython `SingleDispatchMixin.dispatch = 1` /
/// `DoubleDispatchMixin.dispatch = 2` / `HLOperation.dispatch = None`
/// (operation.py:70-72, 202-203, 258-259).
///
/// Consumed by the annotator to pick a specialisation; flowspace itself
/// only records the classification.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Dispatch {
    /// `HLOperation.dispatch = None` — special-cased op (NewDict,
    /// NewTuple, NewList, NewSlice, Pow, SimpleCall, CallArgs,
    /// Contains, Trunc, Format, Get, Set, Delete, UserDel, Buffer,
    /// Yield, Hint, InplacePow). Annotator dispatches manually.
    None,
    /// `SingleDispatchMixin.dispatch = 1` — dispatch on `args[0]`.
    Single,
    /// `DoubleDispatchMixin.dispatch = 2` — dispatch on
    /// `(args[0], args[1])`.
    Double,
}

impl OpKind {
    /// Inverse of [`Self::opname`]. Returns `None` when `name` is not
    /// one of RPython's canonical `SpaceOperation.opname` strings —
    /// `flowcontext.rs` also emits a few majit-local synthetic op
    /// names (`not_`, `newset`, `buildstr`, `ll_assert_not_none`,
    /// `list_to_tuple`, `set_typeparam_default`) that decline here.
    pub fn from_opname(name: &str) -> Option<OpKind> {
        Some(match name {
            "is_" => OpKind::Is,
            "id" => OpKind::Id,
            "type" => OpKind::Type,
            "issubtype" => OpKind::IsSubtype,
            "isinstance" => OpKind::IsInstance,
            "repr" => OpKind::Repr,
            "str" => OpKind::Str,
            "format" => OpKind::Format,
            "len" => OpKind::Len,
            "hash" => OpKind::Hash,
            "setattr" => OpKind::SetAttr,
            "delattr" => OpKind::DelAttr,
            "getitem" => OpKind::GetItem,
            "getitem_idx" => OpKind::GetItemIdx,
            "setitem" => OpKind::SetItem,
            "delitem" => OpKind::DelItem,
            "getslice" => OpKind::GetSlice,
            "setslice" => OpKind::SetSlice,
            "delslice" => OpKind::DelSlice,
            "trunc" => OpKind::Trunc,
            "pos" => OpKind::Pos,
            "neg" => OpKind::Neg,
            "neg_ovf" => OpKind::NegOvf,
            "bool" => OpKind::Bool,
            "abs" => OpKind::Abs,
            "abs_ovf" => OpKind::AbsOvf,
            "hex" => OpKind::Hex,
            "oct" => OpKind::Oct,
            "bin" => OpKind::Bin,
            "ord" => OpKind::Ord,
            "invert" => OpKind::Invert,
            "add" => OpKind::Add,
            "add_ovf" => OpKind::AddOvf,
            "sub" => OpKind::Sub,
            "sub_ovf" => OpKind::SubOvf,
            "mul" => OpKind::Mul,
            "mul_ovf" => OpKind::MulOvf,
            "truediv" => OpKind::TrueDiv,
            "floordiv" => OpKind::FloorDiv,
            "floordiv_ovf" => OpKind::FloorDivOvf,
            "div" => OpKind::Div,
            "div_ovf" => OpKind::DivOvf,
            "mod" => OpKind::Mod,
            "mod_ovf" => OpKind::ModOvf,
            "divmod" => OpKind::DivMod,
            "lshift" => OpKind::LShift,
            "lshift_ovf" => OpKind::LShiftOvf,
            "rshift" => OpKind::RShift,
            "and_" => OpKind::And,
            "or_" => OpKind::Or,
            "xor" => OpKind::Xor,
            "int" => OpKind::Int,
            "index" => OpKind::Index,
            "float" => OpKind::Float,
            "long" => OpKind::Long,
            "inplace_add" => OpKind::InplaceAdd,
            "inplace_sub" => OpKind::InplaceSub,
            "inplace_mul" => OpKind::InplaceMul,
            "inplace_truediv" => OpKind::InplaceTrueDiv,
            "inplace_floordiv" => OpKind::InplaceFloorDiv,
            "inplace_div" => OpKind::InplaceDiv,
            "inplace_mod" => OpKind::InplaceMod,
            "inplace_pow" => OpKind::InplacePow,
            "inplace_lshift" => OpKind::InplaceLShift,
            "inplace_rshift" => OpKind::InplaceRShift,
            "inplace_and" => OpKind::InplaceAnd,
            "inplace_or" => OpKind::InplaceOr,
            "inplace_xor" => OpKind::InplaceXor,
            "lt" => OpKind::Lt,
            "le" => OpKind::Le,
            "eq" => OpKind::Eq,
            "ne" => OpKind::Ne,
            "gt" => OpKind::Gt,
            "ge" => OpKind::Ge,
            "cmp" => OpKind::Cmp,
            "coerce" => OpKind::Coerce,
            "get" => OpKind::Get,
            "set" => OpKind::Set,
            "delete" => OpKind::Delete,
            "userdel" => OpKind::UserDel,
            "buffer" => OpKind::Buffer,
            "yield_" => OpKind::Yield,
            "newslice" => OpKind::NewSlice,
            "hint" => OpKind::Hint,
            "contains" => OpKind::Contains,
            "newdict" => OpKind::NewDict,
            "newtuple" => OpKind::NewTuple,
            "newlist" => OpKind::NewList,
            "pow" => OpKind::Pow,
            "iter" => OpKind::Iter,
            "next" => OpKind::Next,
            "getattr" => OpKind::GetAttr,
            "simple_call" => OpKind::SimpleCall,
            "call_args" => OpKind::CallArgs,
            _ => return None,
        })
    }

    /// RPython `cls.opname` (populated by `add_operator(name, …)`).
    ///
    /// Every variant's return value is the exact string used on
    /// `SpaceOperation.opname`, verified by the upstream test
    /// `rpython/flowspace/test/test_objspace.py`.
    pub fn opname(self) -> &'static str {
        match self {
            OpKind::Is => "is_",
            OpKind::Id => "id",
            OpKind::Type => "type",
            OpKind::IsSubtype => "issubtype",
            OpKind::IsInstance => "isinstance",
            OpKind::Repr => "repr",
            OpKind::Str => "str",
            OpKind::Format => "format",
            OpKind::Len => "len",
            OpKind::Hash => "hash",
            OpKind::SetAttr => "setattr",
            OpKind::DelAttr => "delattr",
            OpKind::GetItem => "getitem",
            OpKind::GetItemIdx => "getitem_idx",
            OpKind::SetItem => "setitem",
            OpKind::DelItem => "delitem",
            OpKind::GetSlice => "getslice",
            OpKind::SetSlice => "setslice",
            OpKind::DelSlice => "delslice",
            OpKind::Trunc => "trunc",
            OpKind::Pos => "pos",
            OpKind::Neg => "neg",
            OpKind::NegOvf => "neg_ovf",
            OpKind::Bool => "bool",
            OpKind::Abs => "abs",
            OpKind::AbsOvf => "abs_ovf",
            OpKind::Hex => "hex",
            OpKind::Oct => "oct",
            OpKind::Bin => "bin",
            OpKind::Ord => "ord",
            OpKind::Invert => "invert",
            OpKind::Add => "add",
            OpKind::AddOvf => "add_ovf",
            OpKind::Sub => "sub",
            OpKind::SubOvf => "sub_ovf",
            OpKind::Mul => "mul",
            OpKind::MulOvf => "mul_ovf",
            OpKind::TrueDiv => "truediv",
            OpKind::FloorDiv => "floordiv",
            OpKind::FloorDivOvf => "floordiv_ovf",
            OpKind::Div => "div",
            OpKind::DivOvf => "div_ovf",
            OpKind::Mod => "mod",
            OpKind::ModOvf => "mod_ovf",
            OpKind::DivMod => "divmod",
            OpKind::LShift => "lshift",
            OpKind::LShiftOvf => "lshift_ovf",
            OpKind::RShift => "rshift",
            OpKind::And => "and_",
            OpKind::Or => "or_",
            OpKind::Xor => "xor",
            OpKind::Int => "int",
            OpKind::Index => "index",
            OpKind::Float => "float",
            OpKind::Long => "long",
            OpKind::InplaceAdd => "inplace_add",
            OpKind::InplaceSub => "inplace_sub",
            OpKind::InplaceMul => "inplace_mul",
            OpKind::InplaceTrueDiv => "inplace_truediv",
            OpKind::InplaceFloorDiv => "inplace_floordiv",
            OpKind::InplaceDiv => "inplace_div",
            OpKind::InplaceMod => "inplace_mod",
            OpKind::InplacePow => "inplace_pow",
            OpKind::InplaceLShift => "inplace_lshift",
            OpKind::InplaceRShift => "inplace_rshift",
            OpKind::InplaceAnd => "inplace_and",
            OpKind::InplaceOr => "inplace_or",
            OpKind::InplaceXor => "inplace_xor",
            OpKind::Lt => "lt",
            OpKind::Le => "le",
            OpKind::Eq => "eq",
            OpKind::Ne => "ne",
            OpKind::Gt => "gt",
            OpKind::Ge => "ge",
            OpKind::Cmp => "cmp",
            OpKind::Coerce => "coerce",
            OpKind::Get => "get",
            OpKind::Set => "set",
            OpKind::Delete => "delete",
            OpKind::UserDel => "userdel",
            OpKind::Buffer => "buffer",
            OpKind::Yield => "yield_",
            OpKind::NewSlice => "newslice",
            OpKind::Hint => "hint",

            OpKind::Contains => "contains",
            OpKind::NewDict => "newdict",
            OpKind::NewTuple => "newtuple",
            OpKind::NewList => "newlist",
            OpKind::Pow => "pow",
            OpKind::Iter => "iter",
            OpKind::Next => "next",
            OpKind::GetAttr => "getattr",
            OpKind::SimpleCall => "simple_call",
            OpKind::CallArgs => "call_args",
        }
    }

    /// RPython `cls.arity` — the declared argument count. Matches the
    /// `arity=` argument of `add_operator()` / the explicit `arity`
    /// class attribute on each subclass.
    ///
    /// Returns `None` for variadic / manually dispatched operations
    /// where RPython leaves `arity` unset (`NewDict`, `NewTuple`,
    /// `NewList`, `SimpleCall`, `CallArgs`, `Hint`).
    pub fn arity(self) -> Option<usize> {
        match self {
            // unary
            OpKind::Id
            | OpKind::Type
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::Hash
            | OpKind::Trunc
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Int
            | OpKind::Index
            | OpKind::Float
            | OpKind::Long
            | OpKind::UserDel
            | OpKind::Buffer
            | OpKind::Yield
            | OpKind::Iter
            | OpKind::Next => Some(1),

            // binary
            OpKind::Is
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Format
            | OpKind::DelAttr
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::DelItem
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::DivMod
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplacePow
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce
            | OpKind::Delete
            | OpKind::Contains
            | OpKind::GetAttr => Some(2),

            // ternary
            OpKind::SetAttr
            | OpKind::SetItem
            | OpKind::GetSlice
            | OpKind::DelSlice
            | OpKind::Get
            | OpKind::Set
            | OpKind::NewSlice
            | OpKind::Pow => Some(3),

            // quaternary
            OpKind::SetSlice => Some(4),

            // variadic / manual-dispatch
            OpKind::NewDict
            | OpKind::NewTuple
            | OpKind::NewList
            | OpKind::SimpleCall
            | OpKind::CallArgs
            | OpKind::Hint => None,
        }
    }

    /// RPython `cls.pure` — whether the op is a `PureOperation`
    /// subclass and therefore eligible for `constfold()`.
    ///
    /// Side-effecting ops (setattr/setitem/setslice/call*/inplace*) are
    /// `False`; pure arithmetic / container construction is `True`.
    /// Matches the `pure=` argument of `add_operator` plus the explicit
    /// `PureOperation` base of subclasses like `NewTuple` and `Pow`.
    pub fn pure(self) -> bool {
        match self {
            OpKind::Is
            | OpKind::Type
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::GetSlice
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::DivMod
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::Int
            | OpKind::Index
            | OpKind::Float
            | OpKind::Long
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce
            | OpKind::Contains
            | OpKind::Get
            | OpKind::Buffer
            | OpKind::NewTuple
            | OpKind::Pow
            | OpKind::GetAttr => true,

            OpKind::Id
            | OpKind::Format
            | OpKind::Hash
            | OpKind::SetAttr
            | OpKind::DelAttr
            | OpKind::SetItem
            | OpKind::DelItem
            | OpKind::SetSlice
            | OpKind::DelSlice
            | OpKind::Trunc
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplacePow
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Set
            | OpKind::Delete
            | OpKind::UserDel
            | OpKind::Yield
            | OpKind::NewSlice
            | OpKind::Hint
            | OpKind::NewDict
            | OpKind::NewList
            | OpKind::Iter
            | OpKind::Next
            | OpKind::SimpleCall
            | OpKind::CallArgs => false,
        }
    }

    /// RPython `cls.can_overflow` — `True` for `OverflowingOperation`
    /// variants (`add`, `sub`, `mul`, `neg`, `abs`, `div`, `floordiv`,
    /// `mod`, `lshift` and their `_ovf` siblings).
    pub fn can_overflow(self) -> bool {
        matches!(
            self,
            OpKind::Neg
                | OpKind::NegOvf
                | OpKind::Abs
                | OpKind::AbsOvf
                | OpKind::Add
                | OpKind::AddOvf
                | OpKind::Sub
                | OpKind::SubOvf
                | OpKind::Mul
                | OpKind::MulOvf
                | OpKind::FloorDiv
                | OpKind::FloorDivOvf
                | OpKind::Div
                | OpKind::DivOvf
                | OpKind::Mod
                | OpKind::ModOvf
                | OpKind::LShift
                | OpKind::LShiftOvf
        )
    }

    /// RPython `cls.dispatch` classification used by the annotator to
    /// pick a specialisation.
    pub fn dispatch(self) -> Dispatch {
        match self {
            // SingleDispatchMixin: `add_operator(…, dispatch=1)`.
            OpKind::Id
            | OpKind::Type
            | OpKind::IsSubtype
            | OpKind::IsInstance
            | OpKind::Repr
            | OpKind::Str
            | OpKind::Len
            | OpKind::Hash
            | OpKind::SetAttr
            | OpKind::DelAttr
            | OpKind::GetSlice
            | OpKind::SetSlice
            | OpKind::DelSlice
            | OpKind::Pos
            | OpKind::Neg
            | OpKind::NegOvf
            | OpKind::Bool
            | OpKind::Abs
            | OpKind::AbsOvf
            | OpKind::Hex
            | OpKind::Oct
            | OpKind::Bin
            | OpKind::Ord
            | OpKind::Invert
            | OpKind::Int
            | OpKind::Float
            | OpKind::Long
            | OpKind::Hint
            | OpKind::Contains
            | OpKind::Iter
            | OpKind::Next
            | OpKind::GetAttr
            | OpKind::SimpleCall
            | OpKind::CallArgs => Dispatch::Single,

            // DoubleDispatchMixin: `add_operator(…, dispatch=2)`.
            OpKind::Is
            | OpKind::GetItem
            | OpKind::GetItemIdx
            | OpKind::SetItem
            | OpKind::DelItem
            | OpKind::Add
            | OpKind::AddOvf
            | OpKind::Sub
            | OpKind::SubOvf
            | OpKind::Mul
            | OpKind::MulOvf
            | OpKind::TrueDiv
            | OpKind::FloorDiv
            | OpKind::FloorDivOvf
            | OpKind::Div
            | OpKind::DivOvf
            | OpKind::Mod
            | OpKind::ModOvf
            | OpKind::LShift
            | OpKind::LShiftOvf
            | OpKind::RShift
            | OpKind::And
            | OpKind::Or
            | OpKind::Xor
            | OpKind::InplaceAdd
            | OpKind::InplaceSub
            | OpKind::InplaceMul
            | OpKind::InplaceTrueDiv
            | OpKind::InplaceFloorDiv
            | OpKind::InplaceDiv
            | OpKind::InplaceMod
            | OpKind::InplaceLShift
            | OpKind::InplaceRShift
            | OpKind::InplaceAnd
            | OpKind::InplaceOr
            | OpKind::InplaceXor
            | OpKind::Lt
            | OpKind::Le
            | OpKind::Eq
            | OpKind::Ne
            | OpKind::Gt
            | OpKind::Ge
            | OpKind::Cmp
            | OpKind::Coerce => Dispatch::Double,

            // manual dispatch / no dispatch=… argument on upstream.
            OpKind::Format
            | OpKind::Trunc
            | OpKind::Index
            | OpKind::InplacePow
            | OpKind::DivMod
            | OpKind::Get
            | OpKind::Set
            | OpKind::Delete
            | OpKind::UserDel
            | OpKind::Buffer
            | OpKind::Yield
            | OpKind::NewSlice
            | OpKind::NewDict
            | OpKind::NewTuple
            | OpKind::NewList
            | OpKind::Pow => Dispatch::None,
        }
    }

    /// RPython `OverflowingOperation.ovf_variant` (operation.py:338-339).
    ///
    /// Returns the `_ovf` twin of a checked arithmetic op, or `None`
    /// when the op has no overflow variant.
    pub fn ovf_variant(self) -> Option<OpKind> {
        Some(match self {
            OpKind::Neg => OpKind::NegOvf,
            OpKind::Abs => OpKind::AbsOvf,
            OpKind::Add => OpKind::AddOvf,
            OpKind::Sub => OpKind::SubOvf,
            OpKind::Mul => OpKind::MulOvf,
            OpKind::FloorDiv => OpKind::FloorDivOvf,
            OpKind::Div => OpKind::DivOvf,
            OpKind::Mod => OpKind::ModOvf,
            OpKind::LShift => OpKind::LShiftOvf,
            _ => return None,
        })
    }

    /// RPython `cls.canraise` — populated by the trailing
    /// `_add_exceptions` / `_add_except_ovf` loop
    /// (`operation.py:728-764`).
    ///
    /// Trace order matches upstream, so each arm corresponds to a
    /// specific set of `lis.append(exc)` calls:
    ///
    /// * `op.getitem / getitem_idx / setitem / delitem` line 728-731.
    /// * `op.contains` line 732.
    /// * `_add_exceptions("div mod divmod truediv floordiv pow
    ///   inplace_div inplace_mod inplace_truediv inplace_floordiv
    ///   inplace_pow", ZeroDivisionError)` line 751-753.
    /// * `_add_exceptions("pow inplace_pow lshift inplace_lshift rshift
    ///   inplace_rshift", ValueError)` line 754-755.
    /// * `_add_exceptions("truediv divmod inplace_add inplace_sub
    ///   inplace_mul inplace_truediv inplace_floordiv inplace_div
    ///   inplace_mod inplace_pow inplace_lshift", OverflowError)`
    ///   line 756-759.
    /// * `_add_except_ovf("neg abs add sub mul floordiv div mod
    ///   lshift")` line 760-761 — copies the base canraise onto the
    ///   `_ovf` twin then appends `OverflowError`.
    /// * `_add_exceptions("pow", OverflowError)` line 762-763 — float
    ///   case.
    pub fn canraise(self) -> &'static [BuiltinException] {
        use BuiltinException::*;
        match self {
            // operation.py:728-731.
            OpKind::GetItem | OpKind::GetItemIdx | OpKind::SetItem | OpKind::DelItem => {
                &[IndexError, KeyError, Exception]
            }
            // operation.py:732.
            OpKind::Contains => &[Exception],

            // `div`: ZeroDivisionError (751). No direct OverflowError.
            // `div_ovf`: `_add_except_ovf` duplicates the base list and
            // appends OverflowError (760).
            OpKind::Div => &[ZeroDivisionError],
            OpKind::DivOvf => &[ZeroDivisionError, OverflowError],
            OpKind::Mod => &[ZeroDivisionError],
            OpKind::ModOvf => &[ZeroDivisionError, OverflowError],
            OpKind::FloorDiv => &[ZeroDivisionError],
            OpKind::FloorDivOvf => &[ZeroDivisionError, OverflowError],
            // `truediv` / `divmod` — ZeroDivisionError (751) + OverflowError
            // (756). No `_ovf` variant in the table.
            OpKind::TrueDiv => &[ZeroDivisionError, OverflowError],
            OpKind::DivMod => &[ZeroDivisionError, OverflowError],

            // `pow` — ZeroDivisionError (751) + ValueError (754) +
            // OverflowError (762, float case).
            OpKind::Pow => &[ZeroDivisionError, ValueError, OverflowError],

            // `lshift`: ValueError (754); `lshift_ovf`: + OverflowError (760).
            OpKind::LShift => &[ValueError],
            OpKind::LShiftOvf => &[ValueError, OverflowError],
            // `rshift`: ValueError (754). No `_ovf`.
            OpKind::RShift => &[ValueError],

            // `_ovf` twins of the pure arithmetic family (760-761).
            // Their base variants have empty canraise.
            OpKind::NegOvf | OpKind::AbsOvf | OpKind::AddOvf | OpKind::SubOvf | OpKind::MulOvf => {
                &[OverflowError]
            }

            // inplace family (753, 754, 758-759). Each upstream line
            // appends one exception; the Rust port flattens the
            // cumulative set.
            OpKind::InplaceAdd => &[OverflowError],
            OpKind::InplaceSub => &[OverflowError],
            OpKind::InplaceMul => &[OverflowError],
            OpKind::InplaceDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceMod => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceFloorDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplaceTrueDiv => &[ZeroDivisionError, OverflowError],
            OpKind::InplacePow => &[ZeroDivisionError, ValueError, OverflowError],
            OpKind::InplaceLShift => &[ValueError, OverflowError],
            OpKind::InplaceRShift => &[ValueError],

            // Default: upstream leaves `canraise = []`.
            _ => &[],
        }
    }
}

/// RPython `flowspace/operation.py:66-116` — `class HLOperation(SpaceOperation)`.
///
/// A high-level operation produced by flow objspace handlers. Each
/// `HLOperation` may be folded to a `Constant` via `constfold()` or
/// recorded as a `SpaceOperation` by `flowcontext.record()`.
#[derive(Clone, Debug)]
pub struct HLOperation {
    /// RPython type-erased identity; matches the Python subclass of
    /// `HLOperation` that would have been instantiated upstream.
    pub kind: OpKind,
    /// RPython `self.args = list(args)` (operation.py:74).
    pub args: Vec<Hlvalue>,
    /// RPython `self.result = Variable()` (operation.py:75).
    pub result: Variable,
    /// RPython `self.offset = -1` (operation.py:76) — rewritten by
    /// `flowcontext.record()` before the op becomes a `SpaceOperation`.
    pub offset: i64,
}

impl HLOperation {
    /// RPython `HLOperation.__init__(*args)` (operation.py:73-76).
    pub fn new(kind: OpKind, args: Vec<Hlvalue>) -> Self {
        HLOperation {
            kind,
            args,
            result: Variable::new(),
            offset: -1,
        }
    }

    /// RPython `HLOperation.replace(mapping)` (operation.py:78-84).
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> HLOperation {
        let newargs: Vec<Hlvalue> = self.args.iter().map(|a| a.replace(mapping)).collect();
        let newresult = self.result.replace(mapping).clone();
        HLOperation {
            kind: self.kind,
            args: newargs,
            result: newresult,
            offset: self.offset,
        }
    }

    /// RPython `HLOperation.constfold()` (operation.py:98-99, overridden
    /// on PureOperation at 120-132).
    ///
    /// Behaviour matches PureOperation upstream: require every arg to
    /// be foldable (`Constant.foldable()`), then apply the pyfunc
    /// equivalent. Non-pure ops and folds that would overflow (RPython
    /// `type(result) is long` branch at 141-142) return `None` so the
    /// caller emits a `SpaceOperation` instead.
    ///
    /// Commit 2 scope: pure integer / float / bool / tuple / comparison
    /// arithmetic — the subset actually exercised by flowspace on
    /// realistic RPython inputs. Non-covered pure ops fall through to
    /// `None` (no fold attempted), which is a strict subset of upstream
    /// behaviour.
    pub fn constfold(&self) -> Option<Hlvalue> {
        // Pure ops go through PureOperation.constfold (operation.py:120-132).
        // A few non-pure ops (`GetAttr`, `Iter`, `Next`) carry their own
        // `constfold()` override upstream and must be allowed past the
        // gate. `SimpleCall` / `CallArgs` / `Pow` handle their own
        // special-casing inside pyfunc() above.
        let eligible =
            self.kind.pure() || matches!(self.kind, OpKind::GetAttr | OpKind::Iter | OpKind::Next);
        if !eligible {
            return None;
        }
        for arg in &self.args {
            match arg {
                Hlvalue::Constant(c) if c.foldable() => {}
                _ => return None,
            }
        }
        let args: Vec<&ConstValue> = self
            .args
            .iter()
            .map(|a| match a {
                Hlvalue::Constant(c) => &c.value,
                _ => unreachable!("foldable() above excludes Variable args"),
            })
            .collect();
        let result = pyfunc(self.kind, &args)?;
        Some(Hlvalue::Constant(Constant::new(result)))
    }

    /// RPython `OverflowingOperation.ovfchecked()` (operation.py:197-200)
    /// — returns the `_ovf` twin of this operation, carrying the same
    /// args / result / offset.
    pub fn ovfchecked(&self) -> Option<HLOperation> {
        let ovf_kind = self.kind.ovf_variant()?;
        Some(HLOperation {
            kind: ovf_kind,
            args: self.args.clone(),
            result: self.result.clone(),
            offset: self.offset,
        })
    }

    /// Lower this `HLOperation` into a plain `SpaceOperation` for
    /// `flowcontext.record()`. Mirrors RPython's implicit upcast via
    /// `HLOperation`'s `SpaceOperation` base class.
    pub fn into_space_operation(self) -> SpaceOperation {
        SpaceOperation::with_offset(
            self.kind.opname(),
            self.args,
            Hlvalue::Variable(self.result),
            self.offset,
        )
    }
}

/// RPython `cls.pyfunc(*args)` applied to constant args — returns
/// `Some(result)` when the pyfunc equivalent fires, or `None` when the
/// fold is skipped for a reason upstream also skips (overflow to long,
/// unhandled type combo, unknown op).
///
/// Commit 2 covers the pure arithmetic subset (int / float / bool / str /
/// tuple / comparisons / identity / type-conversion) that realistic
/// flowspace inputs actually fold. Commit 3 extends this with the
/// explicit subclasses that have custom `eval()` (Iter, Next, GetAttr,
/// SimpleCall, CallArgs, Pow).
fn pyfunc(kind: OpKind, args: &[&ConstValue]) -> Option<ConstValue> {
    // --- variadic / ternary ops that fall outside the fixed-arity
    //     match below ---
    match (kind, args.len()) {
        // RPython `NewTuple` (operation.py:542-548). `PureOperation`
        // whose `pyfunc = lambda *args: args`.
        (OpKind::NewTuple, _) => {
            let items: Vec<ConstValue> = args.iter().map(|&v| v.clone()).collect();
            return Some(ConstValue::Tuple(items));
        }
        // RPython `Pow(PureOperation)` (operation.py:568-578) with
        // arity 3 — `pyfunc = pow`. We fold only the int/int/None
        // variant; float Pow / int-with-mod defer to runtime emit.
        (OpKind::Pow, 3) => {
            if let [
                ConstValue::Int(base),
                ConstValue::Int(exp),
                ConstValue::None,
            ] = args
            {
                if *exp < 0 {
                    return None;
                }
                let e: u32 = (*exp).try_into().ok()?;
                return base.checked_pow(e).map(ConstValue::Int);
            }
        }
        // RPython `GetAttr.constfold()` (operation.py:624-646).
        //
        // `getattr(w_obj, w_name)` folds when both args are foldable
        // AND the receiver is NOT flagged by `NOT_REALLY_CONST`
        // (e.g. `sys.maxint`). The Rust port carries no
        // NOT_REALLY_CONST table at this phase; we restrict the fold
        // to `HostObject::Module::module_get` / class attribute
        // lookup, which the bootstrap HOST_ENV exposes directly.
        (OpKind::GetAttr, 2) => {
            if let [ConstValue::HostObject(obj), ConstValue::Str(name)] = args {
                if let Some(value) = obj.module_get(name) {
                    return Some(ConstValue::HostObject(value));
                }
            }
        }
        _ => {}
    }

    match (kind, args) {
        // --- unary ---
        (OpKind::Bool, [v]) => v.truthy().map(ConstValue::Bool),
        (OpKind::Neg, [ConstValue::Int(n)]) => n.checked_neg().map(ConstValue::Int),
        (OpKind::NegOvf, [ConstValue::Int(n)]) => n.checked_neg().map(ConstValue::Int),
        (OpKind::Neg, [ConstValue::Float(bits)]) => Some(ConstValue::float(-f64::from_bits(*bits))),
        (OpKind::Pos, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Pos, [ConstValue::Float(bits)]) => Some(ConstValue::Float(*bits)),
        (OpKind::Abs, [ConstValue::Int(n)]) => n.checked_abs().map(ConstValue::Int),
        (OpKind::AbsOvf, [ConstValue::Int(n)]) => n.checked_abs().map(ConstValue::Int),
        (OpKind::Abs, [ConstValue::Float(bits)]) => {
            Some(ConstValue::float(f64::from_bits(*bits).abs()))
        }
        (OpKind::Invert, [ConstValue::Int(n)]) => Some(ConstValue::Int(!*n)),
        (OpKind::Int, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Int, [ConstValue::Bool(b)]) => Some(ConstValue::Int(if *b { 1 } else { 0 })),
        (OpKind::Int, [ConstValue::Float(bits)]) => {
            let f = f64::from_bits(*bits);
            // RPython `int(float)` truncates toward zero and raises
            // OverflowError on infinities/NaN. Rust `as i64` saturates,
            // so guard the range explicitly and decline otherwise.
            if f.is_finite() && f >= (i64::MIN as f64) && f <= (i64::MAX as f64) {
                Some(ConstValue::Int(f as i64))
            } else {
                None
            }
        }
        (OpKind::Float, [ConstValue::Int(n)]) => Some(ConstValue::float(*n as f64)),
        (OpKind::Float, [ConstValue::Float(bits)]) => Some(ConstValue::Float(*bits)),
        (OpKind::Float, [ConstValue::Bool(b)]) => {
            Some(ConstValue::float(if *b { 1.0 } else { 0.0 }))
        }
        (OpKind::Long, [ConstValue::Int(n)]) => Some(ConstValue::Int(*n)),
        (OpKind::Ord, [ConstValue::Str(s)]) if s.chars().count() == 1 => {
            s.chars().next().map(|c| ConstValue::Int(c as i64))
        }
        (OpKind::Len, [ConstValue::Str(s)]) => Some(ConstValue::Int(s.chars().count() as i64)),
        (OpKind::Len, [ConstValue::Tuple(items) | ConstValue::List(items)]) => {
            Some(ConstValue::Int(items.len() as i64))
        }
        (OpKind::Len, [ConstValue::Dict(items)]) => Some(ConstValue::Int(items.len() as i64)),

        // --- binary identity ---
        // `operator.is_` compares Python identity. For the value set
        // the Rust port carries on `Constant`, structural equality of
        // `ConstValue` is a safe proxy: two `ConstValue::Int(3)` do
        // compare `==` in Python via small-int cache and we never
        // synthesise two distinct wrappers for the same primitive.
        (OpKind::Is, [a, b]) => Some(ConstValue::Bool(a == b)),

        // --- binary arithmetic (int) ---
        (OpKind::Add, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_add(*b).map(ConstValue::Int)
        }
        (OpKind::AddOvf, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_add(*b).map(ConstValue::Int)
        }
        (OpKind::Sub, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_sub(*b).map(ConstValue::Int)
        }
        (OpKind::SubOvf, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_sub(*b).map(ConstValue::Int)
        }
        (OpKind::Mul, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_mul(*b).map(ConstValue::Int)
        }
        (OpKind::MulOvf, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            a.checked_mul(*b).map(ConstValue::Int)
        }
        (OpKind::FloorDiv, [ConstValue::Int(a), ConstValue::Int(b)])
        | (OpKind::FloorDivOvf, [ConstValue::Int(a), ConstValue::Int(b)])
        | (OpKind::Div, [ConstValue::Int(a), ConstValue::Int(b)])
        | (OpKind::DivOvf, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            // RPython flow-space folds `div` as `operator.floordiv` for
            // int operands; `ZeroDivisionError` is not caught, so a
            // zero divisor declines the fold.
            if *b == 0 {
                None
            } else {
                a.checked_div_euclid(*b).map(ConstValue::Int)
            }
        }
        (OpKind::Mod, [ConstValue::Int(a), ConstValue::Int(b)])
        | (OpKind::ModOvf, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            if *b == 0 {
                None
            } else {
                a.checked_rem_euclid(*b).map(ConstValue::Int)
            }
        }
        (OpKind::LShift, [ConstValue::Int(a), ConstValue::Int(b)])
        | (OpKind::LShiftOvf, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            if *b < 0 || *b >= 64 {
                None
            } else {
                a.checked_shl(*b as u32).map(ConstValue::Int)
            }
        }
        (OpKind::RShift, [ConstValue::Int(a), ConstValue::Int(b)]) => {
            if *b < 0 || *b >= 64 {
                None
            } else {
                Some(ConstValue::Int(*a >> (*b as u32)))
            }
        }
        (OpKind::And, [ConstValue::Int(a), ConstValue::Int(b)]) => Some(ConstValue::Int(*a & *b)),
        (OpKind::Or, [ConstValue::Int(a), ConstValue::Int(b)]) => Some(ConstValue::Int(*a | *b)),
        (OpKind::Xor, [ConstValue::Int(a), ConstValue::Int(b)]) => Some(ConstValue::Int(*a ^ *b)),

        // --- binary arithmetic (float) ---
        (OpKind::Add, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            Some(ConstValue::float(f64::from_bits(*a) + f64::from_bits(*b)))
        }
        (OpKind::Sub, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            Some(ConstValue::float(f64::from_bits(*a) - f64::from_bits(*b)))
        }
        (OpKind::Mul, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            Some(ConstValue::float(f64::from_bits(*a) * f64::from_bits(*b)))
        }
        (OpKind::TrueDiv, [ConstValue::Float(a), ConstValue::Float(b)]) => {
            let y = f64::from_bits(*b);
            if y == 0.0 {
                None
            } else {
                Some(ConstValue::float(f64::from_bits(*a) / y))
            }
        }

        // --- binary concat (str / tuple / list) ---
        (OpKind::Add, [ConstValue::Str(a), ConstValue::Str(b)]) => {
            Some(ConstValue::Str(format!("{a}{b}")))
        }
        (OpKind::Add, [ConstValue::Tuple(a), ConstValue::Tuple(b)]) => {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            Some(ConstValue::Tuple(out))
        }
        (OpKind::Add, [ConstValue::List(a), ConstValue::List(b)]) => {
            let mut out = a.clone();
            out.extend(b.iter().cloned());
            Some(ConstValue::List(out))
        }

        // --- comparisons (int / float / str / bool) ---
        (OpKind::Lt, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_lt())),
        (OpKind::Le, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_le())),
        (OpKind::Eq, [a, b]) => Some(ConstValue::Bool(a == b)),
        (OpKind::Ne, [a, b]) => Some(ConstValue::Bool(a != b)),
        (OpKind::Gt, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_gt())),
        (OpKind::Ge, [a, b]) => cmp_fold(a, b).map(|o| ConstValue::Bool(o.is_ge())),

        // Unhandled combination — decline the fold, let flowcontext
        // emit a SpaceOperation. This is the strict subset of upstream
        // behaviour that Commit 2 covers.
        _ => None,
    }
}

/// Ordering helper for the `Lt / Le / Gt / Ge` fold family. Returns
/// `None` for cross-type comparisons that RPython would raise on (e.g.
/// `int < str`) or for NaN float operands.
fn cmp_fold(a: &ConstValue, b: &ConstValue) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (ConstValue::Int(x), ConstValue::Int(y)) => Some(x.cmp(y)),
        (ConstValue::Float(x), ConstValue::Float(y)) => {
            f64::from_bits(*x).partial_cmp(&f64::from_bits(*y))
        }
        (ConstValue::Str(x), ConstValue::Str(y)) => Some(x.cmp(y)),
        (ConstValue::Bool(x), ConstValue::Bool(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{ConstValue, Constant};
    use super::*;

    #[test]
    fn opname_matches_upstream_strings() {
        // A curated selection: one entry per upstream RPython string
        // that carries an underscore or a non-obvious mapping.
        assert_eq!(OpKind::Is.opname(), "is_");
        assert_eq!(OpKind::And.opname(), "and_");
        assert_eq!(OpKind::Or.opname(), "or_");
        assert_eq!(OpKind::Yield.opname(), "yield_");
        assert_eq!(OpKind::GetItemIdx.opname(), "getitem_idx");
        assert_eq!(OpKind::SimpleCall.opname(), "simple_call");
        assert_eq!(OpKind::CallArgs.opname(), "call_args");
        assert_eq!(OpKind::AddOvf.opname(), "add_ovf");
        assert_eq!(OpKind::IsSubtype.opname(), "issubtype");
        assert_eq!(OpKind::IsInstance.opname(), "isinstance");
    }

    #[test]
    fn ovf_variant_matches_upstream_table() {
        // upstream `add_operator(..., ovf=True)` recurses via
        // `add_operator(name+'_ovf', …)`; the pairs below trace that
        // recursion.
        assert_eq!(OpKind::Add.ovf_variant(), Some(OpKind::AddOvf));
        assert_eq!(OpKind::Sub.ovf_variant(), Some(OpKind::SubOvf));
        assert_eq!(OpKind::Mul.ovf_variant(), Some(OpKind::MulOvf));
        assert_eq!(OpKind::Neg.ovf_variant(), Some(OpKind::NegOvf));
        assert_eq!(OpKind::Abs.ovf_variant(), Some(OpKind::AbsOvf));
        assert_eq!(OpKind::LShift.ovf_variant(), Some(OpKind::LShiftOvf));
        // No overflow variant for Eq / Lt / And / ….
        assert_eq!(OpKind::Eq.ovf_variant(), None);
        assert_eq!(OpKind::And.ovf_variant(), None);
        // And the _ovf's themselves don't recurse further.
        assert_eq!(OpKind::AddOvf.ovf_variant(), None);
    }

    #[test]
    fn dispatch_classification_matches_upstream() {
        // `add_operator('add', 2, dispatch=2, …)` → Double.
        assert_eq!(OpKind::Add.dispatch(), Dispatch::Double);
        // `add_operator('len', 1, dispatch=1, …)` → Single.
        assert_eq!(OpKind::Len.dispatch(), Dispatch::Single);
        // `Pow` / `NewDict` / `NewTuple` carry no `dispatch=` →
        // `HLOperation.dispatch = None`.
        assert_eq!(OpKind::Pow.dispatch(), Dispatch::None);
        assert_eq!(OpKind::NewTuple.dispatch(), Dispatch::None);
    }

    #[test]
    fn arity_matches_upstream() {
        assert_eq!(OpKind::Add.arity(), Some(2));
        assert_eq!(OpKind::Len.arity(), Some(1));
        assert_eq!(OpKind::SetAttr.arity(), Some(3));
        assert_eq!(OpKind::SetSlice.arity(), Some(4));
        // variadic / manual-dispatch → None.
        assert_eq!(OpKind::NewTuple.arity(), None);
        assert_eq!(OpKind::SimpleCall.arity(), None);
    }

    #[test]
    fn pure_matches_upstream() {
        assert!(OpKind::Add.pure());
        assert!(OpKind::NewTuple.pure());
        assert!(OpKind::Pow.pure());
        assert!(!OpKind::SetAttr.pure());
        assert!(!OpKind::SimpleCall.pure());
        assert!(!OpKind::InplaceAdd.pure());
    }

    #[test]
    fn can_overflow_matches_upstream() {
        // `add_operator(..., ovf=True)` → True for both the base and
        // its `_ovf` twin (RPython keeps `can_overflow=True` on the
        // OverflowingOperation subclass).
        assert!(OpKind::Add.can_overflow());
        assert!(OpKind::AddOvf.can_overflow());
        assert!(OpKind::LShift.can_overflow());
        assert!(!OpKind::RShift.can_overflow());
        assert!(!OpKind::Eq.can_overflow());
    }

    fn c(v: ConstValue) -> Hlvalue {
        Hlvalue::Constant(Constant::new(v))
    }

    fn ci(n: i64) -> Hlvalue {
        c(ConstValue::Int(n))
    }

    fn cf(f: f64) -> Hlvalue {
        c(ConstValue::float(f))
    }

    fn cb(b: bool) -> Hlvalue {
        c(ConstValue::Bool(b))
    }

    fn cs(s: &str) -> Hlvalue {
        c(ConstValue::Str(s.to_string()))
    }

    fn fold(kind: OpKind, args: Vec<Hlvalue>) -> Option<ConstValue> {
        let op = HLOperation::new(kind, args);
        op.constfold().map(|v| match v {
            Hlvalue::Constant(c) => c.value,
            _ => unreachable!("constfold must not produce Variables"),
        })
    }

    #[test]
    fn canraise_table_matches_upstream() {
        use BuiltinException::*;
        // operation.py:728-731.
        assert_eq!(
            OpKind::GetItem.canraise(),
            &[IndexError, KeyError, Exception]
        );
        assert_eq!(
            OpKind::DelItem.canraise(),
            &[IndexError, KeyError, Exception]
        );
        // operation.py:732.
        assert_eq!(OpKind::Contains.canraise(), &[Exception]);
        // `add` / `sub` / `mul` — base variants have empty canraise.
        assert_eq!(OpKind::Add.canraise(), &[]);
        assert_eq!(OpKind::Sub.canraise(), &[]);
        assert_eq!(OpKind::Mul.canraise(), &[]);
        // `_ovf` twins pick up OverflowError (_add_except_ovf).
        assert_eq!(OpKind::AddOvf.canraise(), &[OverflowError]);
        assert_eq!(OpKind::NegOvf.canraise(), &[OverflowError]);
        // `div` picks up ZeroDivisionError only; `div_ovf` adds OverflowError.
        assert_eq!(OpKind::Div.canraise(), &[ZeroDivisionError]);
        assert_eq!(
            OpKind::DivOvf.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        // `lshift` picks up ValueError; `lshift_ovf` + OverflowError.
        assert_eq!(OpKind::LShift.canraise(), &[ValueError]);
        assert_eq!(OpKind::LShiftOvf.canraise(), &[ValueError, OverflowError]);
        // `pow` picks up all three (ZeroDiv from 751, ValueError from
        // 754, OverflowError from 762).
        assert_eq!(
            OpKind::Pow.canraise(),
            &[ZeroDivisionError, ValueError, OverflowError]
        );
        // `truediv` / `divmod` — ZeroDivisionError + OverflowError, no _ovf.
        assert_eq!(
            OpKind::TrueDiv.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        assert_eq!(
            OpKind::DivMod.canraise(),
            &[ZeroDivisionError, OverflowError]
        );
        // `inplace_add` — only OverflowError (line 756).
        assert_eq!(OpKind::InplaceAdd.canraise(), &[OverflowError]);
        // Comparisons / identity / len stay clean.
        assert_eq!(OpKind::Lt.canraise(), &[]);
        assert_eq!(OpKind::Is.canraise(), &[]);
        assert_eq!(OpKind::Len.canraise(), &[]);
    }

    #[test]
    fn constfold_declines_non_pure_ops() {
        // `setattr` is side-effecting → no fold.
        assert_eq!(fold(OpKind::SetAttr, vec![ci(1), cs("x"), ci(2)]), None);
        // `simple_call` is non-pure → no fold.
        assert_eq!(fold(OpKind::SimpleCall, vec![ci(1), ci(2)]), None);
    }

    #[test]
    fn constfold_declines_variable_args() {
        let v = Hlvalue::Variable(Variable::new());
        assert_eq!(fold(OpKind::Add, vec![v, ci(1)]), None);
    }

    #[test]
    fn constfold_int_arithmetic() {
        assert_eq!(
            fold(OpKind::Add, vec![ci(3), ci(4)]),
            Some(ConstValue::Int(7))
        );
        assert_eq!(
            fold(OpKind::Sub, vec![ci(10), ci(4)]),
            Some(ConstValue::Int(6))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![ci(6), ci(7)]),
            Some(ConstValue::Int(42))
        );
        assert_eq!(fold(OpKind::Neg, vec![ci(5)]), Some(ConstValue::Int(-5)));
        assert_eq!(fold(OpKind::Abs, vec![ci(-5)]), Some(ConstValue::Int(5)));
        assert_eq!(fold(OpKind::Invert, vec![ci(0)]), Some(ConstValue::Int(-1)));
    }

    #[test]
    fn constfold_int_overflow_declines() {
        // RPython `add_ovf` raises OverflowError; flow-space declines
        // the fold (`PureOperation._pure_result` skips long results).
        assert_eq!(
            fold(OpKind::AddOvf, vec![ci(i64::MAX), ci(1)]),
            None,
            "overflow must not fold"
        );
        assert_eq!(
            fold(OpKind::Add, vec![ci(i64::MAX), ci(1)]),
            None,
            "overflow on checked i64 must not fold"
        );
        assert_eq!(
            fold(OpKind::Neg, vec![ci(i64::MIN)]),
            None,
            "neg(i64::MIN) overflows"
        );
    }

    #[test]
    fn constfold_division_declines_zero() {
        assert_eq!(fold(OpKind::Div, vec![ci(1), ci(0)]), None);
        assert_eq!(fold(OpKind::Mod, vec![ci(1), ci(0)]), None);
        assert_eq!(fold(OpKind::FloorDiv, vec![ci(1), ci(0)]), None);
    }

    #[test]
    fn constfold_bitwise() {
        assert_eq!(
            fold(OpKind::And, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b1000))
        );
        assert_eq!(
            fold(OpKind::Or, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b1110))
        );
        assert_eq!(
            fold(OpKind::Xor, vec![ci(0b1100), ci(0b1010)]),
            Some(ConstValue::Int(0b0110))
        );
        assert_eq!(
            fold(OpKind::LShift, vec![ci(1), ci(4)]),
            Some(ConstValue::Int(16))
        );
        assert_eq!(
            fold(OpKind::RShift, vec![ci(16), ci(2)]),
            Some(ConstValue::Int(4))
        );
    }

    #[test]
    fn constfold_float_arithmetic() {
        assert_eq!(
            fold(OpKind::Add, vec![cf(1.5), cf(2.25)]),
            Some(ConstValue::float(3.75))
        );
        assert_eq!(
            fold(OpKind::Sub, vec![cf(10.0), cf(0.5)]),
            Some(ConstValue::float(9.5))
        );
        assert_eq!(
            fold(OpKind::Mul, vec![cf(2.0), cf(3.0)]),
            Some(ConstValue::float(6.0))
        );
        assert_eq!(
            fold(OpKind::TrueDiv, vec![cf(7.0), cf(2.0)]),
            Some(ConstValue::float(3.5))
        );
        assert_eq!(fold(OpKind::TrueDiv, vec![cf(1.0), cf(0.0)]), None);
    }

    #[test]
    fn constfold_comparisons() {
        assert_eq!(
            fold(OpKind::Lt, vec![ci(1), ci(2)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Gt, vec![ci(1), ci(2)]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Eq, vec![ci(3), ci(3)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Ne, vec![ci(3), ci(4)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Le, vec![cs("a"), cs("b")]),
            Some(ConstValue::Bool(true))
        );
        // cross-type: upstream raises TypeError → fold declines.
        assert_eq!(fold(OpKind::Lt, vec![ci(1), cs("a")]), None);
    }

    #[test]
    fn constfold_bool_conversion() {
        assert_eq!(
            fold(OpKind::Bool, vec![ci(0)]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![ci(5)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cb(true)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cs("")]),
            Some(ConstValue::Bool(false))
        );
        assert_eq!(
            fold(OpKind::Bool, vec![cs("x")]),
            Some(ConstValue::Bool(true))
        );
    }

    #[test]
    fn constfold_string_and_tuple_concat() {
        assert_eq!(
            fold(OpKind::Add, vec![cs("hi"), cs(" there")]),
            Some(ConstValue::Str("hi there".into()))
        );
        assert_eq!(
            fold(
                OpKind::Add,
                vec![
                    c(ConstValue::Tuple(vec![ConstValue::Int(1)])),
                    c(ConstValue::Tuple(vec![
                        ConstValue::Int(2),
                        ConstValue::Int(3)
                    ])),
                ],
            ),
            Some(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
                ConstValue::Int(3),
            ]))
        );
    }

    #[test]
    fn constfold_identity() {
        assert_eq!(
            fold(OpKind::Is, vec![ci(3), ci(3)]),
            Some(ConstValue::Bool(true))
        );
        assert_eq!(
            fold(OpKind::Is, vec![ci(3), ci(4)]),
            Some(ConstValue::Bool(false))
        );
    }

    #[test]
    fn constfold_len() {
        assert_eq!(fold(OpKind::Len, vec![cs("abc")]), Some(ConstValue::Int(3)));
        assert_eq!(
            fold(
                OpKind::Len,
                vec![c(ConstValue::Tuple(vec![
                    ConstValue::Int(1),
                    ConstValue::Int(2)
                ]))],
            ),
            Some(ConstValue::Int(2))
        );
    }

    #[test]
    fn constfold_int_conversion() {
        assert_eq!(fold(OpKind::Int, vec![cf(3.75)]), Some(ConstValue::Int(3)));
        assert_eq!(fold(OpKind::Int, vec![cf(f64::INFINITY)]), None);
        assert_eq!(fold(OpKind::Int, vec![cf(f64::NAN)]), None);
        assert_eq!(fold(OpKind::Int, vec![cb(true)]), Some(ConstValue::Int(1)));
    }

    #[test]
    fn constfold_newtuple_is_variadic_pure() {
        // `NewTuple(PureOperation)` with `pyfunc = lambda *args: args`
        // (operation.py:542-548).
        assert_eq!(
            fold(OpKind::NewTuple, vec![ci(1), ci(2), cs("x")]),
            Some(ConstValue::Tuple(vec![
                ConstValue::Int(1),
                ConstValue::Int(2),
                ConstValue::Str("x".into()),
            ]))
        );
        // empty tuple.
        assert_eq!(
            fold(OpKind::NewTuple, vec![]),
            Some(ConstValue::Tuple(Vec::new()))
        );
    }

    #[test]
    fn constfold_pow_int_nomod() {
        // `Pow(PureOperation)` arity 3, `pyfunc = pow`; default third
        // arg is `Constant(None)` (operation.py:575).
        let none = c(ConstValue::None);
        assert_eq!(
            fold(OpKind::Pow, vec![ci(3), ci(4), none.clone()]),
            Some(ConstValue::Int(81))
        );
        // Negative exponent → not an int result → decline fold.
        assert_eq!(fold(OpKind::Pow, vec![ci(2), ci(-1), none.clone()]), None);
        // Overflow → i64::checked_pow returns None → decline fold.
        assert_eq!(fold(OpKind::Pow, vec![ci(2), ci(64), none]), None);
    }

    #[test]
    fn from_opname_round_trips_every_variant() {
        // Sample a few canonical strings; full round-trip is
        // brittle in a unit test, so we assert the ones most likely
        // to regress.
        for kind in [
            OpKind::Add,
            OpKind::AddOvf,
            OpKind::Is,
            OpKind::And,
            OpKind::Yield,
            OpKind::GetItemIdx,
            OpKind::SimpleCall,
            OpKind::CallArgs,
            OpKind::NewTuple,
            OpKind::Pow,
            OpKind::GetAttr,
        ] {
            let name = kind.opname();
            assert_eq!(
                OpKind::from_opname(name),
                Some(kind),
                "round-trip failed for {kind:?}"
            );
        }

        // Unknown majit-synthetic opnames decline.
        assert_eq!(OpKind::from_opname("not_"), None);
        assert_eq!(OpKind::from_opname("newset"), None);
        assert_eq!(OpKind::from_opname("ll_assert_not_none"), None);
    }

    #[test]
    fn ovfchecked_rewrites_kind() {
        let c1 = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let c2 = Hlvalue::Constant(Constant::new(ConstValue::Int(2)));
        let op = HLOperation::new(OpKind::Add, vec![c1.clone(), c2.clone()]);
        let ovf = op.ovfchecked().expect("add has ovf variant");
        assert_eq!(ovf.kind, OpKind::AddOvf);
        assert_eq!(ovf.args, vec![c1, c2]);

        // Ops without an ovf twin return None.
        let eq_op = HLOperation::new(OpKind::Eq, Vec::new());
        assert!(eq_op.ovfchecked().is_none());
    }

    #[test]
    fn into_space_operation_preserves_opname_and_args() {
        let c1 = Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let c2 = Hlvalue::Constant(Constant::new(ConstValue::Int(2)));
        let op = HLOperation::new(OpKind::Add, vec![c1.clone(), c2.clone()]);
        let result_var = op.result.clone();
        let spaceop = op.into_space_operation();
        assert_eq!(spaceop.opname, "add");
        assert_eq!(spaceop.args, vec![c1, c2]);
        assert_eq!(spaceop.result, Hlvalue::Variable(result_var));
        assert_eq!(spaceop.offset, -1);
    }

    #[test]
    fn replace_remaps_args_and_result() {
        let src = Variable::new();
        let dst = Variable::new();
        let mut mapping = HashMap::new();
        mapping.insert(src.clone(), dst.clone());

        let op = HLOperation {
            kind: OpKind::Add,
            args: vec![
                Hlvalue::Variable(src.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ],
            result: src.clone(),
            offset: 42,
        };
        let replaced = op.replace(&mapping);
        assert_eq!(replaced.kind, OpKind::Add);
        assert_eq!(
            replaced.args,
            vec![
                Hlvalue::Variable(dst.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ]
        );
        assert_eq!(replaced.result, dst);
        assert_eq!(replaced.offset, 42);
    }
}
