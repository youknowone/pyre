//! RPython `rpython/annotator/unaryop.py` — single-dispatch
//! (`dispatch == 1`) operation handlers on the `SomeValue` lattice.
//!
//! Commit split (following the plan in
//! `.claude/plans/0-warm-raccoon.md`):
//!
//! * **Commit 6a (this file initial drop)** — `UNARY_OPERATIONS`
//!   listing + `@op.type.register(SomeObject)` +
//!   `@op.contains.register(SomeNone|SomeInteger|SomeFloat|SomeBool)` +
//!   simple-returning `class __extend__(SomeObject)` defaults +
//!   primitive overrides on `SomeFloat` / `SomeInteger` / `SomeBool` +
//!   `class __extend__(SomeTuple)` unary methods.
//! * **Commit 6a2** — `@op.bool.register(SomeObject)` with knowntypedata
//!   branch refinement, `@op.isinstance.register` /
//!   `simple_call` / `call_args` / `issubtype` / `bool_behavior` /
//!   `getattr` / `setattr` / `hash` refinement with `find_method`.
//! * **Commit 6b** — list/dict `method_*` analyzers.
//! * **Commit 6c** — string/bytearray/char/unicode `method_*` analyzers.
//! * **Commit 6d** — Instance / Iterator / Builtin / PBC / None /
//!   Exception / WeakRef handlers.

use std::rc::Rc;

use super::super::flowspace::model::ConstValue;
use super::super::flowspace::model::Constant;
use super::super::flowspace::model::Hlvalue;
use super::super::flowspace::operation::{
    BuiltinException, CanOnlyThrow, HLOperation, OpKind, Specialization, Transformation,
    register_single,
};
use super::annrpython::RPythonAnnotator;
use super::model::{
    AnnotatorError, SomeBool, SomeBuiltinMethod, SomeFloat, SomeInteger, SomeIterator,
    SomeObjectTrait, SomeString, SomeTuple, SomeTypeOf, SomeUnicodeString, SomeValue, SomeValueTag,
    s_impossible_value, s_none,
};

/// RPython `UNARY_OPERATIONS` (unaryop.py:26-28).
///
/// ```python
/// UNARY_OPERATIONS = set([oper.opname for oper in op.__dict__.values()
///                         if oper.dispatch == 1])
/// UNARY_OPERATIONS.remove('contains')
/// ```
pub static UNARY_OPERATIONS: &[OpKind] = &[
    OpKind::Id,
    OpKind::Type,
    OpKind::IsSubtype,
    OpKind::IsInstance,
    OpKind::Repr,
    OpKind::Str,
    OpKind::Len,
    OpKind::Hash,
    OpKind::SetAttr,
    OpKind::DelAttr,
    OpKind::GetSlice,
    OpKind::SetSlice,
    OpKind::DelSlice,
    OpKind::Pos,
    OpKind::Neg,
    OpKind::NegOvf,
    OpKind::Bool,
    OpKind::Abs,
    OpKind::AbsOvf,
    OpKind::Hex,
    OpKind::Oct,
    OpKind::Bin,
    OpKind::Ord,
    OpKind::Invert,
    OpKind::Int,
    OpKind::Float,
    OpKind::Long,
    OpKind::Hint,
    OpKind::Iter,
    OpKind::Next,
    OpKind::GetAttr,
    OpKind::SimpleCall,
    OpKind::CallArgs,
    // `Contains` is removed from UNARY_OPERATIONS upstream line 28 —
    // it's a binary op on the `in` operand semantically. The `@op.contains.register`
    // hooks below still live in unaryop.py, so they land here.
];

/// Module-import-time population of `_REGISTRY_SINGLE` — mirrors the
/// side-effect of RPython importing unaryop.py (which runs all the
/// `@op.X.register(Some_cls)` decorators at import time).
pub fn init(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    init_type_register(reg);
    init_contains_register(reg);
    init_someobject_defaults(reg);
    init_somefloat_overrides(reg);
    init_someinteger_overrides(reg);
    init_somebool_overrides(reg);
    init_sometuple_overrides(reg);
    init_somelist_overrides(reg);
    init_somedict_overrides(reg);
    init_somestring_overrides(reg);
    init_someunicodestring_overrides(reg);
    init_somebytearray_overrides(reg);
    init_somechar_overrides(reg);
    init_someunicodecp_overrides(reg);
    init_someiterator_overrides(reg);
    init_somepbc_overrides(reg);
    init_somenone_overrides(reg);
    init_someweakref_overrides(reg);
}

/// Module-import time side effect mirroring
/// `@op.<name>.register_transform(...)` decorators in unaryop.py. Runs
/// at `_TRANSFORM_SINGLE` thread_local init; one block per upstream
/// `@op.X.register_transform(Y)` decorator.
pub fn init_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    init_instance_single_transform(reg);
    init_instance_attr_transform(reg);
    init_object_call_args_transform(reg);
}

fn register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
    op: OpKind,
    tag: SomeValueTag,
    spec: Specialization,
) {
    reg.entry(op).or_default().insert(tag, spec);
}

fn register_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
    op: OpKind,
    tag: SomeValueTag,
    tx: Transformation,
) {
    reg.entry(op).or_default().insert(tag, tx);
}

// =====================================================================
// unaryop.py:31-33 — @op.type.register(SomeObject)
// =====================================================================
//
//     @op.type.register(SomeObject)
//     def type_SomeObject(annotator, v_arg):
//         return SomeTypeOf([v_arg])

fn init_type_register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    register(
        reg,
        OpKind::Type,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, hl| {
                // upstream unaryop.py:31-33 — `SomeTypeOf([v_arg])`.
                // Constants are ignored: upstream `type(const)` would
                // never enter this dispatcher (it lands on the
                // const-folding path of `Type.constfold`). When the
                // arg happens to be a constant we fall back to an
                // empty is_type_of list.
                let vars: Vec<Rc<super::super::flowspace::model::Variable>> = match &hl.args[0] {
                    super::super::flowspace::model::Hlvalue::Variable(v) => {
                        vec![Rc::new(v.clone())]
                    }
                    super::super::flowspace::model::Hlvalue::Constant(_) => Vec::new(),
                };
                SomeValue::TypeOf(SomeTypeOf::new(vars))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:91-111 — @op.contains.register(SomeObject|SomeNone|…)
// =====================================================================

fn init_contains_register(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:91-94 — SomeObject: returns s_Bool, can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Bool(SomeBool::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:96-104 — SomeNone: returns SomeBool(const=False).
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                let mut s = SomeBool::new();
                s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
                SomeValue::Bool(s)
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:107-111 — SomeInteger/Float/Bool: raise AnnotatorError.
    let number_tags = &[
        SomeValueTag::Integer,
        SomeValueTag::Float,
        SomeValueTag::Bool,
    ];
    for tag in number_tags {
        register(
            reg,
            OpKind::Contains,
            *tag,
            Specialization {
                apply: Box::new(|_ann, _hl| panic!("AnnotatorError: number is not iterable")),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// =====================================================================
// unaryop.py:156-254 — class __extend__(SomeObject) simple-return defaults
// =====================================================================
//
// `class __extend__(SomeObject)` upstream defines the defaults fetched by
// `SingleDispatchMixin.get_specialization` when `getattr(s_arg, opname)`
// hits the base class. Rust port registers one spec per opname on
// `SomeValueTag::Object`; the MRO fallback handles every subclass that
// doesn't override.

fn init_someobject_defaults(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:158-159 — len(self): return SomeInteger(nonneg=True)
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:169-170 — hash: raise AnnotatorError("cannot use hash() in RPython").
    register(
        reg,
        OpKind::Hash,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: cannot use hash() in RPython")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:172-173 — str: SomeString().
    register(
        reg,
        OpKind::Str,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:178-179 — repr: SomeString().
    register(
        reg,
        OpKind::Repr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:181-188 — hex/bin/oct: SomeString().
    for op in &[OpKind::Hex, OpKind::Bin, OpKind::Oct] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| SomeValue::String(SomeString::new(false, false))),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:190-192 — id: raise AnnotatorError.
    register(
        reg,
        OpKind::Id,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                panic!("AnnotatorError: cannot use id() in RPython; see objectmodel.compute_xxx()")
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:194-195 — int: SomeInteger().
    register(
        reg,
        OpKind::Int,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:197-198 — float: SomeFloat().
    register(
        reg,
        OpKind::Float,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:215-229 — getattr(self, s_attr).
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = ann.annotation(&hl.args[0]).expect("getattr: self unbound");
                let s_attr = ann.annotation(&hl.args[1]).expect("getattr: attr unbound");
                let Some(ConstValue::Str(attr)) = s_attr.const_().cloned() else {
                    panic!(
                        "AnnotatorError: getattr({:?}, {:?}) has non-constant argument",
                        s_self, s_attr
                    );
                };
                if let Some(s_method) = s_self.find_method(&attr) {
                    return s_method;
                }
                if s_self.is_immutable_constant() {
                    if let Some(ConstValue::HostObject(pyobj)) = s_self.const_() {
                        if let Ok(value) = crate::flowspace::model::host_getattr(pyobj, &attr) {
                            return ann.bookkeeper.immutablevalue(&value).unwrap_or_else(|err| {
                                panic!("getattr immutablevalue failed: {}", err)
                            });
                        }
                    }
                }
                panic!(
                    "AnnotatorError: Cannot find attribute {:?} on {:?}",
                    attr, s_self
                )
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:200-204 — delattr: warning-only. Returns no value
    // (upstream implicit None). Port as a void specialization returning
    // Impossible.
    register(
        reg,
        OpKind::DelAttr,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Impossible),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:240-241 — hint(self, *args_s): return self.
    register(
        reg,
        OpKind::Hint,
        SomeValueTag::Object,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("hint: self unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:243-250 — getslice/setslice/delslice: s_ImpossibleValue.
    for op in &[OpKind::GetSlice, OpKind::SetSlice, OpKind::DelSlice] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:252-254 — pos/neg/abs/ord/invert/long/iter/next: s_ImpossibleValue.
    //
    //     def pos(self):
    //         return s_ImpossibleValue
    //     neg = abs = ord = invert = long = iter = next = pos
    for op in &[
        OpKind::Pos,
        OpKind::Neg,
        OpKind::Abs,
        OpKind::Ord,
        OpKind::Invert,
        OpKind::Long,
        OpKind::Iter,
        OpKind::Next,
    ] {
        register(
            reg,
            *op,
            SomeValueTag::Object,
            Specialization {
                apply: Box::new(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
}

// =====================================================================
// unaryop.py:257-273 — class __extend__(SomeFloat)
// =====================================================================

fn init_somefloat_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:259-260 — pos(self): return self.
    register(
        reg,
        OpKind::Pos,
        SomeValueTag::Float,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("float.pos: unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:262-265 — neg(self): SomeFloat(). abs = neg.
    for op in &[OpKind::Neg, OpKind::Abs] {
        register(
            reg,
            *op,
            SomeValueTag::Float,
            Specialization {
                apply: Box::new(|_ann, _hl| SomeValue::Float(SomeFloat::new())),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:267-270 — bool(self): SomeBool (const if self is constant).
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Float,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("float.bool: unbound");
                if let SomeValue::Float(f) = &s {
                    if f.is_immutable_constant() {
                        if let Some(c) = &f.base.const_box {
                            if let ConstValue::Float(bits) = c.value {
                                let mut r = SomeBool::new();
                                r.base.const_box = Some(Constant::new(ConstValue::Bool(
                                    f64::from_bits(bits) != 0.0,
                                )));
                                return SomeValue::Bool(r);
                            }
                        }
                    }
                }
                SomeValue::Bool(SomeBool::new())
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:272-273 — len(self): raise AnnotatorError.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Float,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: 'float' has no length")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:275-302 — class __extend__(SomeInteger)
// =====================================================================

fn init_someinteger_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:277-285 — invert/pos/int: SomeInteger(knowntype=self.knowntype).
    //                      can_only_throw = [].
    let same_kt_int = |ann: &RPythonAnnotator, hl: &HLOperation| -> SomeValue {
        let kt = match ann.annotation(&hl.args[0]) {
            Some(SomeValue::Integer(i)) => i.base.knowntype,
            _ => panic!("integer op: arg 0 not SomeInteger"),
        };
        SomeValue::Integer(SomeInteger::new_with_knowntype(false, kt))
    };
    for op in &[OpKind::Invert, OpKind::Pos, OpKind::Int] {
        register(
            reg,
            *op,
            SomeValueTag::Integer,
            Specialization {
                apply: Box::new(same_kt_int),
                can_only_throw: CanOnlyThrow::List(vec![]),
            },
        );
    }
    // unaryop.py:289-293 — neg: SomeInteger(knowntype=self.knowntype). neg_ovf overflows.
    register(
        reg,
        OpKind::Neg,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(same_kt_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::NegOvf,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(same_kt_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py:295-299 — abs: SomeInteger(nonneg=True, knowntype=self.knowntype).
    let abs_int = |ann: &RPythonAnnotator, hl: &HLOperation| -> SomeValue {
        let kt = match ann.annotation(&hl.args[0]) {
            Some(SomeValue::Integer(i)) => i.base.knowntype,
            _ => panic!("integer.abs: arg 0 not SomeInteger"),
        };
        // upstream: SomeInteger(nonneg=True, knowntype=self.knowntype).
        let mut i = SomeInteger::new_with_knowntype(true, kt);
        i.nonneg = true;
        SomeValue::Integer(i)
    };
    register(
        reg,
        OpKind::Abs,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(abs_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::AbsOvf,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(abs_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py:301-302 — len: raise.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Integer,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: 'int' has no length")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:305-330 — class __extend__(SomeBool)
// =====================================================================

fn init_somebool_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:306-307 — bool(self): return self.
    register(
        reg,
        OpKind::Bool,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("bool.bool: unbound")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:309-312 — invert: SomeInteger(). can_only_throw=[].
    register(
        reg,
        OpKind::Invert,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:314-318 — neg: SomeInteger(). neg_ovf overflows.
    register(
        reg,
        OpKind::Neg,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::NegOvf,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::default())),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    // unaryop.py:320-330 — abs / abs_ovf / pos / int: SomeInteger(nonneg=True).
    let nonneg_int = |_ann: &RPythonAnnotator, _hl: &HLOperation| -> SomeValue {
        let mut i = SomeInteger::default();
        i.nonneg = true;
        SomeValue::Integer(i)
    };
    register(
        reg,
        OpKind::Abs,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::AbsOvf,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![BuiltinException::OverflowError]),
        },
    );
    register(
        reg,
        OpKind::Pos,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    register(
        reg,
        OpKind::Int,
        SomeValueTag::Bool,
        Specialization {
            apply: Box::new(nonneg_int),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:332-348 — class __extend__(SomeTuple)
// =====================================================================

fn init_sometuple_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:334-335 — len(self): immutablevalue(len(self.items)).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Tuple,
        Specialization {
            apply: Box::new(|ann, hl| {
                let t = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Tuple(t)) => t,
                    _ => panic!("tuple.len: arg 0 not SomeTuple"),
                };
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(t.items.len() as i64)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:337-339 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Tuple,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("tuple.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:344-348 — getslice((self, s_start, s_stop)): require
    // constant indices, slice items, rebuild SomeTuple. Registered as
    // the primary SomeTuple.getslice handler (the SomeObject fallback
    // would have returned s_ImpossibleValue).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::Tuple,
        Specialization {
            apply: Box::new(|ann, hl| {
                let t = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Tuple(t)) => t,
                    _ => panic!("tuple.getslice: arg 0 not SomeTuple"),
                };
                let start = ann
                    .annotation(&hl.args[1])
                    .expect("tuple.getslice: start unbound");
                let stop = ann
                    .annotation(&hl.args[2])
                    .expect("tuple.getslice: stop unbound");
                let start_c = match &start {
                    SomeValue::Integer(i) if i.is_immutable_constant() => i
                        .base
                        .const_box
                        .as_ref()
                        .and_then(|c| match c.value {
                            ConstValue::Int(v) => Some(v),
                            _ => None,
                        })
                        .expect("tuple.getslice: const_box mismatch"),
                    _ => panic!("AnnotatorError: tuple slicing: needs constants (start not const)"),
                };
                let stop_c = match &stop {
                    SomeValue::Integer(i) if i.is_immutable_constant() => i
                        .base
                        .const_box
                        .as_ref()
                        .and_then(|c| match c.value {
                            ConstValue::Int(v) => Some(v),
                            _ => None,
                        })
                        .expect("tuple.getslice: const_box mismatch"),
                    _ => panic!("AnnotatorError: tuple slicing: needs constants (stop not const)"),
                };
                let len = t.items.len() as i64;
                let clamp = |idx: i64| -> usize {
                    if idx < 0 {
                        (len + idx).max(0) as usize
                    } else {
                        (idx as usize).min(t.items.len())
                    }
                };
                let lo = clamp(start_c);
                let hi = clamp(stop_c).max(lo);
                let items = t.items[lo..hi].to_vec();
                SomeValue::Tuple(SomeTuple::new(items))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:350-443 — @op.contains.register(SomeList) + class __extend__(SomeList)
// =====================================================================

fn init_somelist_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:350-354 — contains: generalize + s_Bool, can_only_throw=[].
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("contains(SomeList): arg 0 not SomeList"),
                };
                let s_element = ann
                    .annotation(&hl.args[1])
                    .expect("contains(SomeList): element unbound");
                s_list
                    .listdef
                    .generalize(&s_element)
                    .expect("contains.SomeList: generalize failed");
                SomeValue::Bool(SomeBool::new())
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:391-396 — len(self): position read_item → 0 if impossible,
    // else SomeObject.len.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.len: arg 0 not SomeList"),
                };
                let position = ann.bookkeeper.current_position_key();
                let s_item = s_list.listdef.read_item(position);
                if matches!(s_item, SomeValue::Impossible) {
                    // upstream: immutablevalue(0) → SomeInteger with const=0.
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                    SomeValue::Integer(i)
                } else {
                    // upstream: SomeObject.len(self) → SomeInteger(nonneg=True).
                    SomeValue::Integer(SomeInteger::new(true, false))
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:398-400 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("list.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:420-423 — getslice((self, s_start, s_stop)):
    //     check_negative_slice(...); return listdef.offspring(bk).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.getslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                let res = s_list
                    .listdef
                    .offspring(&ann.bookkeeper, &[])
                    .expect("listdef.offspring failed");
                SomeValue::List(res)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:425-431 — setslice((self, s_start, s_stop), s_iterable):
    //     check_negative + isinstance(SomeList) + mutate + agree + resize.
    register(
        reg,
        OpKind::SetSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.setslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.setslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.setslice: stop unbound");
                let s_iterable = ann
                    .annotation(&hl.args[3])
                    .expect("list.setslice: iterable unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                let s_other = match s_iterable {
                    SomeValue::List(o) => o,
                    _ => panic!("AnnotatorError: list[start:stop] = x: x must be a list"),
                };
                s_list.listdef.mutate().expect("listdef.mutate failed");
                s_list
                    .listdef
                    .agree(&ann.bookkeeper, &s_other.listdef)
                    .expect("listdef.agree failed");
                s_list.listdef.resize().expect("listdef.resize failed");
                SomeValue::Impossible
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:433-435 — delslice((self, s_start, s_stop)):
    //     check_negative + resize.
    register(
        reg,
        OpKind::DelSlice,
        SomeValueTag::List,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_list = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::List(s)) => s,
                    _ => panic!("list.delslice: arg 0 not SomeList"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("list.delslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("list.delslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                s_list.listdef.resize().expect("listdef.resize failed");
                SomeValue::Impossible
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

/// RPython `check_negative_slice(s_start, s_stop, error="slicing")`
/// (unaryop.py:437-443). Helper shared by list and string slice paths.
fn check_negative_slice(s_start: &SomeValue, s_stop: &SomeValue, error: &str) {
    if let SomeValue::Integer(i) = s_start {
        if !i.nonneg {
            panic!("AnnotatorError: {error}: not proven to have non-negative start");
        }
    }
    if let SomeValue::Integer(i) = s_stop {
        if !i.nonneg {
            // upstream: allow const == -1 as sentinel for "slice to end".
            let is_minus_one = i
                .base
                .const_box
                .as_ref()
                .map(|c| matches!(c.value, ConstValue::Int(-1)))
                .unwrap_or(false);
            if !is_minus_one {
                panic!("AnnotatorError: {error}: not proven to have non-negative stop");
            }
        }
    }
}

// =====================================================================
// unaryop.py:357-418 — SomeList.method_* free functions
// =====================================================================
//
// Upstream these are class methods on `class __extend__(SomeList)`. They
// are NOT directly `@op.X.register` targets; instead `SomeObject.getattr`
// looks up `method_<name>` via `find_method` and wraps them in a
// `SomeBuiltinMethod` for downstream `simple_call`. The Rust port keeps
// the analyzer bodies as free functions and the `find_method` /
// `SomeBuiltinMethod.call` bridge below dispatches to them.

#[allow(dead_code)]
pub fn list_method_append(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:359-361
    s_self.listdef.resize().expect("resize");
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_extend(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_iterable: &SomeValue,
) -> SomeValue {
    // unaryop.py:363-369
    s_self.listdef.resize().expect("resize");
    if let SomeValue::List(other) = s_iterable {
        s_self
            .listdef
            .agree(&ann.bookkeeper, &other.listdef)
            .expect("agree");
    } else {
        // upstream:
        //     s_iter = s_iterable.iter()
        //     self.method_append(s_iter.next())
        // Dispatching back through the registry requires the `iter` op
        // on the iterable's tag and then `next` — land this in Commit 6d
        // alongside the general `find_method`/SomeBuiltinMethod.call
        // wiring.
        panic!(
            "list.method_extend: non-list iterable path requires iter/next dispatch \
             (lands with SomeBuiltinMethod wiring, Commit 6d)"
        );
    }
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_reverse(_ann: &RPythonAnnotator, s_self: &super::model::SomeList) -> SomeValue {
    s_self.listdef.mutate().expect("listdef.mutate");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_insert(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    _s_index: &SomeValue,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:374-375 — delegates to method_append.
    list_method_append(ann, s_self, s_value)
}

#[allow(dead_code)]
pub fn list_method_remove(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:377-379 — resize + generalize.
    s_self.listdef.resize().expect("resize");
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn list_method_pop(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    _s_index: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:381-385 — resize + read_item(position). can_only_throw=[IndexError].
    s_self.listdef.resize().expect("resize");
    let position = ann.bookkeeper.current_position_key();
    s_self.listdef.read_item(position)
}

#[allow(dead_code)]
pub fn list_method_index(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeList,
    s_value: &SomeValue,
) -> SomeValue {
    // unaryop.py:387-389 — generalize + SomeInteger(nonneg=True).
    s_self.listdef.generalize(s_value).expect("generalize");
    SomeValue::Integer(SomeInteger::new(true, false))
}

// =====================================================================
// unaryop.py:446-460 — @op.contains.register(SomeDict) + dict_contains helper
// =====================================================================

/// RPython `dict_contains(s_dct, s_element, position)` (unaryop.py:446-452).
///
/// ```python
/// def dict_contains(s_dct, s_element, position):
///     s_dct.dictdef.generalize_key(s_element)
///     if s_dct._is_empty(position):
///         s_bool = SomeBool(); s_bool.const = False; return s_bool
///     return s_Bool
/// ```
fn dict_contains(
    s_dct: &super::model::SomeDict,
    s_element: &SomeValue,
    position: Option<super::bookkeeper::PositionKey>,
) -> SomeValue {
    s_dct
        .dictdef
        .generalize_key(s_element)
        .expect("dict_contains: generalize_key failed");
    if dict_is_empty(s_dct, position) {
        let mut s = SomeBool::new();
        s.base.const_box = Some(Constant::new(ConstValue::Bool(false)));
        return SomeValue::Bool(s);
    }
    SomeValue::Bool(SomeBool::new())
}

/// RPython `SomeDict._is_empty(self, position)` (unaryop.py:464-468).
fn dict_is_empty(
    s_dct: &super::model::SomeDict,
    position: Option<super::bookkeeper::PositionKey>,
) -> bool {
    let s_key = s_dct.dictdef.read_key(position.clone());
    let s_value = s_dct.dictdef.read_value(position);
    matches!(s_key, SomeValue::Impossible) || matches!(s_value, SomeValue::Impossible)
}

fn init_somedict_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:454-460 — @op.contains.register(SomeDict):
    //     position = annotator.bookkeeper.position_key
    //     return dict_contains(...); can_only_throw = _dict_can_only_throw_nothing.
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::Dict,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_dct = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Dict(d)) => d,
                    _ => panic!("contains(SomeDict): arg 0 not SomeDict"),
                };
                let s_elem = ann
                    .annotation(&hl.args[1])
                    .expect("contains(SomeDict): element unbound");
                let position = ann.bookkeeper.current_position_key();
                dict_contains(&s_dct, &s_elem, position)
            }),
            can_only_throw: CanOnlyThrow::Callable(Box::new(|args_s| {
                // Mirror binaryop::_dict_can_only_throw_nothing (binaryop.py:532-535).
                match args_s.first() {
                    Some(SomeValue::Dict(d)) => {
                        if d.dictdef.custom_eq_hash() {
                            None
                        } else {
                            Some(vec![])
                        }
                    }
                    _ => Some(vec![]),
                }
            })),
        },
    );
    // unaryop.py:470-474 — len(self): 0 if empty, else SomeObject.len.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Dict,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_dct = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::Dict(d)) => d,
                    _ => panic!("dict.len: arg 0 not SomeDict"),
                };
                let position = ann.bookkeeper.current_position_key();
                if dict_is_empty(&s_dct, position) {
                    let mut i = SomeInteger::new(true, false);
                    i.base.const_box = Some(Constant::new(ConstValue::Int(0)));
                    SomeValue::Integer(i)
                } else {
                    SomeValue::Integer(SomeInteger::new(true, false))
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:476-478 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Dict,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("dict.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
}

// =====================================================================
// unaryop.py:462-591 — SomeDict.method_* free functions
// =====================================================================

#[allow(dead_code)]
pub fn dict_method_get(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_dfl: &SomeValue,
) -> SomeValue {
    // unaryop.py:502-506
    let position = ann.bookkeeper.current_position_key();
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key");
    s_self
        .dictdef
        .generalize_value(s_dfl)
        .expect("generalize_value");
    s_self.dictdef.read_value(position)
}

#[allow(dead_code)]
pub fn dict_method_copy(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:510-511 — SomeDict(self.dictdef).
    SomeValue::Dict(super::model::SomeDict::new(s_self.dictdef.clone()))
}

#[allow(dead_code)]
pub fn dict_method_update(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_other: &SomeValue,
) -> SomeValue {
    // unaryop.py:513-516
    //     if s_None.contains(dct2): return SomeImpossibleValue()
    //     dct1.dictdef.union(dct2.dictdef)
    if let SomeValue::None_(_) = s_other {
        return s_impossible_value();
    }
    let other = match s_other {
        SomeValue::Dict(d) => d,
        _ => panic!("dict.update: other not SomeDict or SomeNone"),
    };
    s_self
        .dictdef
        .union_with(&other.dictdef)
        .expect("dictdef.union failed");
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_keys(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:521-523 — bk.newlist(self.dictdef.read_key(bk.position_key)).
    let position = ann.bookkeeper.current_position_key();
    let s_key = s_self.dictdef.read_key(position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_key], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn dict_method_values(ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:525-527
    let position = ann.bookkeeper.current_position_key();
    let s_value = s_self.dictdef.read_value(position);
    let s_list = ann
        .bookkeeper
        .newlist(&[s_value], None)
        .expect("bookkeeper.newlist failed");
    SomeValue::List(s_list)
}

#[allow(dead_code)]
pub fn dict_method_iterkeys(_ann: &RPythonAnnotator, s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:533-534 — SomeIterator(self, 'keys').
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["keys".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_itervalues(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["values".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_iteritems(
    _ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
) -> SomeValue {
    SomeValue::Iterator(SomeIterator::new(
        SomeValue::Dict(s_self.clone()),
        vec!["items".into()],
    ))
}

#[allow(dead_code)]
pub fn dict_method_clear(_ann: &RPythonAnnotator, _s_self: &super::model::SomeDict) -> SomeValue {
    // unaryop.py:548-549 — pass.
    SomeValue::Impossible
}

#[allow(dead_code)]
pub fn dict_method_pop(
    ann: &RPythonAnnotator,
    s_self: &super::model::SomeDict,
    s_key: &SomeValue,
    s_dfl: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:555-560
    s_self
        .dictdef
        .generalize_key(s_key)
        .expect("generalize_key");
    if let Some(s) = s_dfl {
        s_self
            .dictdef
            .generalize_value(s)
            .expect("generalize_value");
    }
    let position = ann.bookkeeper.current_position_key();
    s_self.dictdef.read_value(position)
}

// =====================================================================
// unaryop.py:593-690 — @op.contains.register(SomeString|SomeUnicodeString)
// + class __extend__(SomeString, SomeUnicodeString) shared overrides
// =====================================================================

fn init_somestring_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:593-605 — contains_String:
    //   if const '\0': knowntypedata-based refinement, else SomeObject fallback.
    //   can_only_throw=[].
    //
    // Rust port: the '\0' fast path depends on SomeString::nonnulify()
    // (not yet ported). Land the fallback-only implementation now; the
    // knowntypedata refinement joins Commit 6a2 together with
    // bool_SomeObject / set_knowntypedata plumbing.
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Bool(SomeBool::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:660-662 — iter(self): SomeIterator(self). can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("string.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:684-687 — getslice: check_negative_slice + basestringclass(no_nul).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::String(s)) => s,
                    _ => panic!("string.getslice: arg 0 not SomeString"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("string.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("string.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::String(SomeString::new(false, s_self.inner.no_nul))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:720-725 — len(self): immutablevalue(len(const)) if const, else SomeInteger(nonneg).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::String,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::String(s)) => s,
                    _ => panic!("string.len: arg 0 not SomeString"),
                };
                if s.is_constant() {
                    if let Some(c) = &s.inner.base.const_box {
                        if let ConstValue::Str(v) = &c.value {
                            let mut i = SomeInteger::new(true, false);
                            i.base.const_box =
                                Some(Constant::new(ConstValue::Int(v.chars().count() as i64)));
                            return SomeValue::Integer(i);
                        }
                    }
                }
                SomeValue::Integer(SomeInteger::new(true, false))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_someunicodestring_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:593-605 — contains on SomeUnicodeString (same path).
    register(
        reg,
        OpKind::Contains,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Bool(SomeBool::new())),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:660-662 — iter on SomeUnicodeString.
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: Box::new(|ann, hl| {
                let sv = ann.annotation(&hl.args[0]).expect("unicode.iter: unbound");
                SomeValue::Iterator(SomeIterator::new(sv, vec![]))
            }),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:684-687 — getslice on SomeUnicodeString (basestringclass=SomeUnicodeString).
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::UnicodeString,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_self = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::UnicodeString(s)) => s,
                    _ => panic!("unicode.getslice: arg 0 not SomeUnicodeString"),
                };
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("unicode.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("unicode.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::UnicodeString(SomeUnicodeString::new(false, s_self.inner.no_nul))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_somebytearray_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:693-696 — SomeByteArray.getslice: check + SomeByteArray().
    register(
        reg,
        OpKind::GetSlice,
        SomeValueTag::ByteArray,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s_start = ann
                    .annotation(&hl.args[1])
                    .expect("bytearray.getslice: start unbound");
                let s_stop = ann
                    .annotation(&hl.args[2])
                    .expect("bytearray.getslice: stop unbound");
                check_negative_slice(&s_start, &s_stop, "slicing");
                SomeValue::ByteArray(super::model::SomeByteArray::new(false))
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:771-797 — class __extend__(SomeChar, SomeUnicodeCodePoint) / SomeChar
// =====================================================================

fn init_somechar_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:771-774 — len(self) = immutablevalue(1).
    register(
        reg,
        OpKind::Len,
        SomeValueTag::Char,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:776-779 — ord(self) = SomeInteger(nonneg=True).
    register(
        reg,
        OpKind::Ord,
        SomeValueTag::Char,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

fn init_someunicodecp_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:771-774 — len(self) = immutablevalue(1) on SomeUnicodeCodePoint.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::UnicodeCodePoint,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                let mut i = SomeInteger::new(true, false);
                i.base.const_box = Some(Constant::new(ConstValue::Int(1)));
                SomeValue::Integer(i)
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:799-804 — ord(self): SomeInteger(nonneg=True).
    register(
        reg,
        OpKind::Ord,
        SomeValueTag::UnicodeCodePoint,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Integer(SomeInteger::new(true, false))),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:608-690 / 720-748 / 776-797 — method_* free fns (String/Char)
// =====================================================================

/// RPython "basestringclass" picks SomeString vs SomeUnicodeString based
/// on the actual class of the SomeValue. SomeChar → SomeString,
/// SomeUnicodeCodePoint → SomeUnicodeString.
fn basestring_of(sv: &SomeValue, no_nul: bool) -> SomeValue {
    match sv {
        SomeValue::String(_) | SomeValue::Char(_) => {
            SomeValue::String(SomeString::new(false, no_nul))
        }
        SomeValue::UnicodeString(_) | SomeValue::UnicodeCodePoint(_) => {
            SomeValue::UnicodeString(SomeUnicodeString::new(false, no_nul))
        }
        other => panic!("basestring_of: unexpected {:?}", other),
    }
}

#[allow(dead_code)]
pub fn str_method_startswith(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_frag: &SomeValue,
) -> SomeValue {
    // unaryop.py:611-614 — const+const → immutablevalue(self.const.startswith(frag.const))
    //                       else s_Bool.
    if let (Some(a), Some(b)) = (const_str_of(s_self), const_str_of(s_frag)) {
        let mut r = SomeBool::new();
        r.base.const_box = Some(Constant::new(ConstValue::Bool(a.starts_with(&b))));
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_endswith(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    s_frag: &SomeValue,
) -> SomeValue {
    if let (Some(a), Some(b)) = (const_str_of(s_self), const_str_of(s_frag)) {
        let mut r = SomeBool::new();
        r.base.const_box = Some(Constant::new(ConstValue::Bool(a.ends_with(&b))));
        return SomeValue::Bool(r);
    }
    SomeValue::Bool(SomeBool::new())
}

fn const_str_of(sv: &SomeValue) -> Option<String> {
    let cb = match sv {
        SomeValue::String(s) => &s.inner.base.const_box,
        SomeValue::UnicodeString(s) => &s.inner.base.const_box,
        SomeValue::Char(s) => &s.inner.base.const_box,
        SomeValue::UnicodeCodePoint(s) => &s.inner.base.const_box,
        _ => return None,
    };
    cb.as_ref().and_then(|c| match &c.value {
        ConstValue::Str(v) => Some(v.clone()),
        _ => None,
    })
}

#[allow(dead_code)]
pub fn str_method_find(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:621-623 — check_negative_slice(start,end,"find") → SomeInteger().
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "find");
    }
    SomeValue::Integer(SomeInteger::default())
}

#[allow(dead_code)]
pub fn str_method_rfind(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "rfind");
    }
    SomeValue::Integer(SomeInteger::default())
}

#[allow(dead_code)]
pub fn str_method_count(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _s_frag: &SomeValue,
    s_start: Option<&SomeValue>,
    s_end: Option<&SomeValue>,
) -> SomeValue {
    if let (Some(start), Some(end)) = (s_start, s_end) {
        check_negative_slice(start, end, "count");
    }
    SomeValue::Integer(SomeInteger::new(true, false))
}

#[allow(dead_code)]
pub fn str_method_strip(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    _s_chr: Option<&SomeValue>,
) -> SomeValue {
    // unaryop.py:633-636
    //   if chr is None and isinstance(self, SomeUnicodeString):
    //       raise AnnotatorError("unicode.strip() with no arg is not RPython")
    //   return self.basestringclass(no_nul=self.no_nul)
    if _s_chr.is_none() && matches!(s_self, SomeValue::UnicodeString(_)) {
        panic!("AnnotatorError: unicode.strip() with no arg is not RPython");
    }
    let no_nul = match s_self {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        SomeValue::Char(s) => s.inner.no_nul,
        SomeValue::UnicodeCodePoint(s) => s.inner.no_nul,
        _ => false,
    };
    basestring_of(s_self, no_nul)
}

#[allow(dead_code)]
pub fn str_method_upper(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    // unaryop.py:736-737 / 793-797 — SomeChar.upper returns self; SomeString.upper returns SomeString().
    match _s_self {
        SomeValue::Char(_) => _s_self.clone(),
        _ => SomeValue::String(SomeString::new(false, false)),
    }
}

#[allow(dead_code)]
pub fn str_method_lower(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    match _s_self {
        SomeValue::Char(_) => _s_self.clone(),
        _ => SomeValue::String(SomeString::new(false, false)),
    }
}

#[allow(dead_code)]
pub fn str_method_isdigit(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_isalpha(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_isalnum(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn str_method_replace(
    _ann: &RPythonAnnotator,
    s_self: &SomeValue,
    _s1: &SomeValue,
    s2: &SomeValue,
) -> SomeValue {
    // unaryop.py:681-682 — basestringclass(no_nul=self.no_nul and s2.no_nul).
    let self_no_nul = match s_self {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        _ => false,
    };
    let s2_no_nul = match s2 {
        SomeValue::String(s) => s.inner.no_nul,
        SomeValue::UnicodeString(s) => s.inner.no_nul,
        SomeValue::Char(s) => s.inner.no_nul,
        SomeValue::UnicodeCodePoint(s) => s.inner.no_nul,
        _ => false,
    };
    basestring_of(s_self, self_no_nul && s2_no_nul)
}

#[allow(dead_code)]
pub fn str_method_format(
    _ann: &RPythonAnnotator,
    _s_self: &SomeValue,
    _args: &[SomeValue],
) -> SomeValue {
    // unaryop.py:689-690 — always raises.
    panic!("AnnotatorError: Method format() is not RPython")
}

#[allow(dead_code)]
pub fn char_method_isspace(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn char_method_islower(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

#[allow(dead_code)]
pub fn char_method_isupper(_ann: &RPythonAnnotator, _s_self: &SomeValue) -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

/// RPython `SomeObject.find_method(self, name)` (unaryop.py:206-213).
pub fn find_method(s_self: &SomeValue, name: &str) -> Option<SomeBuiltinMethod> {
    let analyser_name = match s_self {
        SomeValue::List(_) => match name {
            "append" => "list_method_append",
            "extend" => "list_method_extend",
            "reverse" => "list_method_reverse",
            "insert" => "list_method_insert",
            "remove" => "list_method_remove",
            "pop" => "list_method_pop",
            "index" => "list_method_index",
            _ => return None,
        },
        SomeValue::Dict(_) => match name {
            "get" | "setdefault" => "dict_method_get",
            "copy" => "dict_method_copy",
            "update" => "dict_method_update",
            "keys" => "dict_method_keys",
            "values" => "dict_method_values",
            "iterkeys" => "dict_method_iterkeys",
            "itervalues" => "dict_method_itervalues",
            "iteritems" => "dict_method_iteritems",
            "clear" => "dict_method_clear",
            "pop" => "dict_method_pop",
            _ => return None,
        },
        SomeValue::String(_) | SomeValue::UnicodeString(_) => match name {
            "startswith" => "str_method_startswith",
            "endswith" => "str_method_endswith",
            "find" => "str_method_find",
            "rfind" => "str_method_rfind",
            "count" => "str_method_count",
            "strip" => "str_method_strip",
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            "replace" => "str_method_replace",
            "format" => "str_method_format",
            _ => return None,
        },
        SomeValue::Char(_) => match name {
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            "isspace" => "char_method_isspace",
            "islower" => "char_method_islower",
            "isupper" => "char_method_isupper",
            _ => return None,
        },
        SomeValue::UnicodeCodePoint(_) => match name {
            "upper" => "str_method_upper",
            "lower" => "str_method_lower",
            "isdigit" => "str_method_isdigit",
            "isalpha" => "str_method_isalpha",
            "isalnum" => "str_method_isalnum",
            _ => return None,
        },
        _ => return None,
    };
    Some(SomeBuiltinMethod::new(analyser_name, s_self.clone(), name))
}

fn builtin_method_receiver_error(method: &SomeBuiltinMethod) -> AnnotatorError {
    AnnotatorError::new(format!(
        "SomeBuiltinMethod.call(): receiver/type mismatch for {}",
        method.methodname
    ))
}

fn builtin_method_arg_error(method: &SomeBuiltinMethod) -> AnnotatorError {
    AnnotatorError::new(format!(
        "SomeBuiltinMethod.call(): wrong argument shape for {}",
        method.methodname
    ))
}

/// RPython `SomeBuiltinMethod.call(self, args, implicit_init=False)`
/// dispatch helper (unaryop.py:961-967).
pub fn call_builtin_method(
    ann: &RPythonAnnotator,
    method: &SomeBuiltinMethod,
    args: &super::argument::ArgumentsForTranslation,
) -> Result<SomeValue, AnnotatorError> {
    let (args_s, kwds) = args
        .unpack()
        .map_err(|err| AnnotatorError::new(err.getmsg()))?;
    if !kwds.is_empty() {
        return Err(AnnotatorError::new(format!(
            "SomeBuiltinMethod.call(): keyword arguments are not supported for {}",
            method.methodname
        )));
    }

    let result = match method.analyser_name.as_str() {
        "list_method_append" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [s_value] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            list_method_append(ann, s_self, s_value)
        }
        "list_method_extend" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [s_iterable] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            list_method_extend(ann, s_self, s_iterable)
        }
        "list_method_reverse" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            list_method_reverse(ann, s_self)
        }
        "list_method_insert" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [s_index, s_value] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            list_method_insert(ann, s_self, s_index, s_value)
        }
        "list_method_remove" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [s_value] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            list_method_remove(ann, s_self, s_value)
        }
        "list_method_pop" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            match args_s.as_slice() {
                [] => list_method_pop(ann, s_self, None),
                [s_index] => list_method_pop(ann, s_self, Some(s_index)),
                _ => return Err(builtin_method_arg_error(method)),
            }
        }
        "list_method_index" => {
            let SomeValue::List(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [s_value] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            list_method_index(ann, s_self, s_value)
        }
        "dict_method_get" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            match args_s.as_slice() {
                [s_key] => dict_method_get(ann, s_self, s_key, &s_none()),
                [s_key, s_dfl] => dict_method_get(ann, s_self, s_key, s_dfl),
                _ => return Err(builtin_method_arg_error(method)),
            }
        }
        "dict_method_copy" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_copy(ann, s_self)
        }
        "dict_method_update" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [s_other] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_update(ann, s_self, s_other)
        }
        "dict_method_keys" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_keys(ann, s_self)
        }
        "dict_method_values" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_values(ann, s_self)
        }
        "dict_method_iterkeys" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_iterkeys(ann, s_self)
        }
        "dict_method_itervalues" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_itervalues(ann, s_self)
        }
        "dict_method_iteritems" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_iteritems(ann, s_self)
        }
        "dict_method_clear" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            dict_method_clear(ann, s_self)
        }
        "dict_method_pop" => {
            let SomeValue::Dict(s_self) = &*method.s_self else {
                return Err(builtin_method_receiver_error(method));
            };
            match args_s.as_slice() {
                [s_key] => dict_method_pop(ann, s_self, s_key, None),
                [s_key, s_dfl] => dict_method_pop(ann, s_self, s_key, Some(s_dfl)),
                _ => return Err(builtin_method_arg_error(method)),
            }
        }
        "str_method_startswith" => {
            let [s_frag] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_startswith(ann, &method.s_self, s_frag)
        }
        "str_method_endswith" => {
            let [s_frag] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_endswith(ann, &method.s_self, s_frag)
        }
        "str_method_find" => match args_s.as_slice() {
            [s_frag] => str_method_find(ann, &method.s_self, s_frag, None, None),
            [s_frag, s_start] => str_method_find(ann, &method.s_self, s_frag, Some(s_start), None),
            [s_frag, s_start, s_end] => {
                str_method_find(ann, &method.s_self, s_frag, Some(s_start), Some(s_end))
            }
            _ => return Err(builtin_method_arg_error(method)),
        },
        "str_method_rfind" => match args_s.as_slice() {
            [s_frag] => str_method_rfind(ann, &method.s_self, s_frag, None, None),
            [s_frag, s_start] => str_method_rfind(ann, &method.s_self, s_frag, Some(s_start), None),
            [s_frag, s_start, s_end] => {
                str_method_rfind(ann, &method.s_self, s_frag, Some(s_start), Some(s_end))
            }
            _ => return Err(builtin_method_arg_error(method)),
        },
        "str_method_count" => match args_s.as_slice() {
            [s_frag] => str_method_count(ann, &method.s_self, s_frag, None, None),
            [s_frag, s_start] => str_method_count(ann, &method.s_self, s_frag, Some(s_start), None),
            [s_frag, s_start, s_end] => {
                str_method_count(ann, &method.s_self, s_frag, Some(s_start), Some(s_end))
            }
            _ => return Err(builtin_method_arg_error(method)),
        },
        "str_method_strip" => match args_s.as_slice() {
            [] => str_method_strip(ann, &method.s_self, None),
            [s_chr] => str_method_strip(ann, &method.s_self, Some(s_chr)),
            _ => return Err(builtin_method_arg_error(method)),
        },
        "str_method_upper" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_upper(ann, &method.s_self)
        }
        "str_method_lower" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_lower(ann, &method.s_self)
        }
        "str_method_isdigit" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_isdigit(ann, &method.s_self)
        }
        "str_method_isalpha" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_isalpha(ann, &method.s_self)
        }
        "str_method_isalnum" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_isalnum(ann, &method.s_self)
        }
        "str_method_replace" => {
            let [s1, s2] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            str_method_replace(ann, &method.s_self, s1, s2)
        }
        "str_method_format" => str_method_format(ann, &method.s_self, &args_s),
        "char_method_isspace" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            char_method_isspace(ann, &method.s_self)
        }
        "char_method_islower" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            char_method_islower(ann, &method.s_self)
        }
        "char_method_isupper" => {
            let [] = args_s.as_slice() else {
                return Err(builtin_method_arg_error(method));
            };
            char_method_isupper(ann, &method.s_self)
        }
        _ => {
            return Err(AnnotatorError::new(format!(
                "SomeBuiltinMethod.call(): unknown analyser {}",
                method.analyser_name
            )));
        }
    };
    Ok(result)
}

// =====================================================================
// unaryop.py:806-830 — class __extend__(SomeIterator)
// =====================================================================

fn init_someiterator_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:808-810 — iter(self): return self. can_only_throw=[].
    register(
        reg,
        OpKind::Iter,
        SomeValueTag::Iterator,
        Specialization {
            apply: Box::new(|ann, hl| ann.annotation(&hl.args[0]).expect("iterator.iter: unbound")),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:818-830 — next(self):
    //   position = getbookkeeper().position_key
    //   if s_None.contains(self.s_container): return s_ImpossibleValue
    //   if self.variant and self.variant[0] == "enumerate":
    //       s_item = self.s_container.getanyitem(position)
    //       return SomeTuple((SomeInteger(nonneg=True), s_item))
    //   variant = self.variant
    //   if variant == ("reversed",): variant = ()
    //   return self.s_container.getanyitem(position, *variant)
    register(
        reg,
        OpKind::Next,
        SomeValueTag::Iterator,
        Specialization {
            apply: Box::new(|ann, hl| {
                let s = ann.annotation(&hl.args[0]).expect("iterator.next: unbound");
                let it = match s {
                    SomeValue::Iterator(i) => i,
                    _ => panic!("iterator.next: arg 0 not SomeIterator"),
                };
                let position = ann.bookkeeper.current_position_key();
                // upstream: `if s_None.contains(self.s_container): return s_ImpossibleValue`.
                if matches!(&*it.s_container, SomeValue::None_(_)) {
                    return s_impossible_value();
                }
                // upstream: `if self.variant and self.variant[0] == "enumerate": ...`.
                if it.variant.first().map(String::as_str) == Some("enumerate") {
                    let s_item = container_getanyitem(&it.s_container, None, position.clone());
                    return SomeValue::Tuple(SomeTuple::new(vec![
                        SomeValue::Integer(SomeInteger::new(true, false)),
                        s_item,
                    ]));
                }
                // upstream: `if variant == ("reversed",): variant = ()`.
                let variant: Option<&str> = if it.variant.len() == 1 && it.variant[0] == "reversed"
                {
                    None
                } else {
                    it.variant.first().map(String::as_str)
                };
                container_getanyitem(&it.s_container, variant, position)
            }),
            can_only_throw: CanOnlyThrow::Callable(Box::new(|args_s| {
                // upstream `_can_only_throw` (unaryop.py:812-816):
                //   can_throw = [StopIteration]
                //   if isinstance(self.s_container, SomeDict):
                //       can_throw.append(RuntimeError)
                //   return can_throw
                let mut throws = vec![BuiltinException::StopIteration];
                if let Some(SomeValue::Iterator(i)) = args_s.first() {
                    if matches!(&*i.s_container, SomeValue::Dict(_)) {
                        throws.push(BuiltinException::RuntimeError);
                    }
                }
                Some(throws)
            })),
        },
    );
}

/// RPython `container.getanyitem(position, *variant)` dispatch —
/// pulled out into a free function so [`SomeIterator.next`] can call
/// it once for the enumerate branch (with variant forced to `()`) and
/// once for the default branch (variant = `it.variant`).
///
/// Upstream definitions:
/// * SomeTuple (unaryop.py:341-342): `unionof(*self.items)`.
/// * SomeList (unaryop.py:402-403):  `self.listdef.read_item(position)`.
/// * SomeDict (unaryop.py:480-500):  variant-dispatched.
/// * SomeStringOrUnicode (unaryop.py:664-665): `self.basecharclass()`.
fn container_getanyitem(
    s_container: &SomeValue,
    variant: Option<&str>,
    position: Option<super::bookkeeper::PositionKey>,
) -> SomeValue {
    use super::super::flowspace::argument::Signature;
    let _ = Signature::new; // silence dead-imports on cold paths
    match s_container {
        SomeValue::Tuple(t) => {
            if t.items.is_empty() {
                s_impossible_value()
            } else {
                super::model::unionof(t.items.iter()).expect("getanyitem(tuple): unionof failed")
            }
        }
        SomeValue::List(l) => l.listdef.read_item(position),
        SomeValue::Dict(d) => {
            // unaryop.py:480-500 — per-variant dispatch.
            match variant.unwrap_or("keys") {
                "keys" => d.dictdef.read_key(position),
                "values" => d.dictdef.read_value(position),
                "items" => {
                    let s_key = d.dictdef.read_key(position.clone());
                    let s_value = d.dictdef.read_value(position);
                    if matches!(s_key, SomeValue::Impossible)
                        || matches!(s_value, SomeValue::Impossible)
                    {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![s_key, s_value]))
                    }
                }
                "items_with_hash" => {
                    let s_key = d.dictdef.read_key(position.clone());
                    let s_value = d.dictdef.read_value(position);
                    if matches!(s_key, SomeValue::Impossible)
                        || matches!(s_value, SomeValue::Impossible)
                    {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![
                            s_key,
                            s_value,
                            SomeValue::Integer(SomeInteger::default()),
                        ]))
                    }
                }
                "keys_with_hash" => {
                    let s_key = d.dictdef.read_key(position);
                    if matches!(s_key, SomeValue::Impossible) {
                        s_impossible_value()
                    } else {
                        SomeValue::Tuple(SomeTuple::new(vec![
                            s_key,
                            SomeValue::Integer(SomeInteger::default()),
                        ]))
                    }
                }
                other => panic!("getanyitem(dict): unknown variant {other:?}"),
            }
        }
        // unaryop.py:664-665 — `SomeStringOrUnicode.getanyitem` returns
        // `self.basecharclass()`. model.py:326-329 assigns
        //   SomeString.basecharclass = SomeChar
        //   SomeUnicodeString.basecharclass = SomeUnicodeCodePoint
        // and the class is called with no arguments — so the char's
        // `no_nul` defaults to False regardless of the string's.
        SomeValue::String(_) => SomeValue::Char(super::model::SomeChar::new(false)),
        SomeValue::UnicodeString(_) => {
            SomeValue::UnicodeCodePoint(super::model::SomeUnicodeCodePoint::new(false))
        }
        other => panic!("getanyitem: unsupported container {other:?}"),
    }
}

// =====================================================================
// unaryop.py:970-998 — class __extend__(SomePBC)
// =====================================================================

fn init_somepbc_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:982-983 — setattr raises.
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::PBC,
        Specialization {
            apply: Box::new(|_ann, _hl| {
                panic!("AnnotatorError: Cannot modify attribute of a pre-built constant")
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:997-998 — len raises.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::PBC,
        Specialization {
            apply: Box::new(|_ann, _hl| panic!("AnnotatorError: Cannot call len on a pbc")),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:1000-1021 — class __extend__(SomeNone)
// =====================================================================

fn init_somenone_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:1004-1006 — getattr: s_ImpossibleValue, can_only_throw=[].
    register(
        reg,
        OpKind::GetAttr,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::List(vec![]),
        },
    );
    // unaryop.py:1008-1009 — setattr: return None (void → SomeImpossible).
    register(
        reg,
        OpKind::SetAttr,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| SomeValue::Impossible),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
    // unaryop.py:1011-1012 — call / simple_call / call_args: s_ImpossibleValue.
    for op in &[OpKind::SimpleCall, OpKind::CallArgs] {
        register(
            reg,
            *op,
            SomeValueTag::None_,
            Specialization {
                apply: Box::new(|_ann, _hl| s_impossible_value()),
                can_only_throw: CanOnlyThrow::Absent,
            },
        );
    }
    // unaryop.py:1017-1021 — len(self): SomeImpossibleValue.
    register(
        reg,
        OpKind::Len,
        SomeValueTag::None_,
        Specialization {
            apply: Box::new(|_ann, _hl| s_impossible_value()),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// =====================================================================
// unaryop.py:1032-1037 — class __extend__(SomeWeakRef)
// =====================================================================

fn init_someweakref_overrides(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Specialization>,
    >,
) {
    // unaryop.py:1033-1037 — simple_call(self):
    //   if classdef is None: return s_None (dead weakref)
    //   else: SomeInstance(classdef, can_be_None=True)
    register(
        reg,
        OpKind::SimpleCall,
        SomeValueTag::WeakRef,
        Specialization {
            apply: Box::new(|ann, hl| {
                let wr = match ann.annotation(&hl.args[0]) {
                    Some(SomeValue::WeakRef(w)) => w,
                    _ => panic!("weakref.simple_call: arg 0 not SomeWeakRef"),
                };
                match &wr.classdef {
                    None => super::model::s_none(),
                    Some(cd) => SomeValue::Instance(super::model::SomeInstance::new(
                        Some(Rc::clone(cd)),
                        true,
                        std::collections::BTreeMap::new(),
                    )),
                }
            }),
            can_only_throw: CanOnlyThrow::Absent,
        },
    );
}

// Suppress unused warnings for items reserved for the follow-up
// commits in the same family (register_single, SomeUnicodeString,
// SomeBool const-propagation paths, Rc).
#[allow(dead_code)]
fn _keep_imports_live() {
    let _ = register_single;
    let _ = SomeUnicodeString::default;
    let _: Option<Rc<()>> = None;
}

// =====================================================================
// unaryop.py:120-139 — @op.call_args.register_transform(SomeObject)
// =====================================================================
//
// Upstream rewrites `op.call_args(v_func, v_shape, *data_v)` when the
// vararg tail is a `SomeTuple`, unpacking the tuple elements to turn
// the op back into `op.simple_call(...)` (or `op.call_args(...)` if
// keywords remain). The Rust port requires `CallShape` decoding from
// the `ConstValue::Tuple` payload on `v_shape` (mirrors
// `CallSpec.fromshape`).

fn init_object_call_args_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    use super::super::flowspace::argument::CallSpec;

    // unaryop.py:120-139 — @op.call_args.register_transform(SomeObject).
    // ```python
    // def transform_varargs(annotator, v_func, v_shape, *data_v):
    //     callspec = CallSpec.fromshape(v_shape.value, list(data_v))
    //     v_vararg = callspec.w_stararg
    //     if callspec.w_stararg:
    //         s_vararg = annotator.annotation(callspec.w_stararg)
    //         if not isinstance(s_vararg, SomeTuple):
    //             raise AnnotatorError(...)
    //         n_items = len(s_vararg.items)
    //         ops = [op.getitem(v_vararg, const(i)) for i in range(n_items)]
    //         new_args = callspec.arguments_w + [hlop.result for hlop in ops]
    //         if callspec.keywords:
    //             newspec = CallSpec(new_args, callspec.keywords)
    //             shape, data_v = newspec.flatten()
    //             call_op = op.call_args(v_func, const(shape), *data_v)
    //         else:
    //             call_op = op.simple_call(v_func, *new_args)
    //         ops.append(call_op)
    //         return ops
    // ```
    register_transform(
        reg,
        OpKind::CallArgs,
        SomeValueTag::Object,
        Box::new(|ann, args| {
            // args = [v_func, v_shape, *data_v]
            if args.len() < 2 {
                return None;
            }
            let v_func = args[0].clone();
            let v_shape = &args[1];
            // upstream: `CallSpec.fromshape(v_shape.value, list(data_v))`.
            let shape_const = match v_shape {
                Hlvalue::Constant(c) => &c.value,
                _ => return None,
            };
            let shape = decode_call_shape(shape_const)?;
            let data_v: Vec<Hlvalue> = args[2..].to_vec();
            let callspec = CallSpec::fromshape(&shape, data_v);
            // upstream: `v_vararg = callspec.w_stararg; if callspec.w_stararg: ...`.
            let Some(v_vararg) = callspec.w_stararg.clone() else {
                return None;
            };
            // upstream: `s_vararg = annotator.annotation(v_vararg)`.
            let s_vararg = ann.annotation(&v_vararg)?;
            // upstream: `if not isinstance(s_vararg, SomeTuple): raise AnnotatorError`.
            let super::model::SomeValue::Tuple(ref tup) = s_vararg else {
                panic!("AnnotatorError: Calls like f(..., *arg) require 'arg' to be a tuple");
            };
            let n_items = tup.items.len();
            // upstream: `ops = [op.getitem(v_vararg, const(i)) for i in range(n_items)]`.
            let mut ops: Vec<HLOperation> = Vec::with_capacity(n_items + 1);
            let mut unpacked: Vec<Hlvalue> = Vec::with_capacity(n_items);
            for i in 0..n_items {
                let getitem = mk_hlop(
                    OpKind::GetItem,
                    vec![
                        v_vararg.clone(),
                        Hlvalue::Constant(Constant::new(ConstValue::Int(i as i64))),
                    ],
                );
                unpacked.push(Hlvalue::Variable(getitem.result.clone()));
                ops.push(getitem);
            }
            // upstream: `new_args = callspec.arguments_w + [hlop.result for ...]`.
            let mut new_args: Vec<Hlvalue> = callspec.arguments_w.clone();
            new_args.extend(unpacked.into_iter());
            // upstream: `if callspec.keywords: newspec = CallSpec(new_args, callspec.keywords);
            //            shape, data_v = newspec.flatten(); op.call_args(v_func, const(shape), *data_v)`
            //           else: `op.simple_call(v_func, *new_args)`.
            let call_op = if !callspec.keywords.is_empty() {
                let newspec = CallSpec::new(new_args, Some(callspec.keywords), None);
                let (new_shape, new_data_v) = newspec.flatten();
                let mut call_args = vec![
                    v_func,
                    Hlvalue::Constant(Constant::new(encode_call_shape(&new_shape))),
                ];
                call_args.extend(new_data_v.into_iter());
                mk_hlop(OpKind::CallArgs, call_args)
            } else {
                let mut simple_args = vec![v_func];
                simple_args.extend(new_args.into_iter());
                mk_hlop(OpKind::SimpleCall, simple_args)
            };
            ops.push(call_op);
            Some(ops)
        }),
    );
}

/// Decode the `CallShape` triple encoded in a `call_args` shape
/// constant (see `flowcontext::build_call_shape_constant` for the
/// encoder — ConstValue::Tuple([Int(cnt), Tuple([Str(key), …]), Bool(star)])).
fn decode_call_shape(cv: &ConstValue) -> Option<super::super::flowspace::argument::CallShape> {
    let items = match cv {
        ConstValue::Tuple(items) if items.len() == 3 => items,
        _ => return None,
    };
    let shape_cnt = match &items[0] {
        ConstValue::Int(n) => *n as usize,
        _ => return None,
    };
    let shape_keys: Vec<String> = match &items[1] {
        ConstValue::Tuple(keys) => keys
            .iter()
            .map(|k| match k {
                ConstValue::Str(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    let shape_star = match &items[2] {
        ConstValue::Bool(b) => *b,
        _ => return None,
    };
    Some(super::super::flowspace::argument::CallShape {
        shape_cnt,
        shape_keys,
        shape_star,
    })
}

/// Encode a `CallShape` as the `ConstValue::Tuple` payload expected by
/// `call_args` — mirrors `flowcontext::build_call_shape_constant`.
fn encode_call_shape(shape: &super::super::flowspace::argument::CallShape) -> ConstValue {
    ConstValue::Tuple(vec![
        ConstValue::Int(shape.shape_cnt as i64),
        ConstValue::Tuple(
            shape
                .shape_keys
                .iter()
                .cloned()
                .map(ConstValue::Str)
                .collect(),
        ),
        ConstValue::Bool(shape.shape_star),
    ])
}

// =====================================================================
// unaryop.py:867-892 — @op.{len,iter,next,getslice,setslice}.register_transform(SomeInstance)
// =====================================================================

/// Helper — build a new [`HLOperation`] with a fresh result variable,
/// matching upstream `HLOperation.__init__` (operation.py:73-75).
fn mk_hlop(kind: OpKind, args: Vec<Hlvalue>) -> HLOperation {
    HLOperation::new(kind, args)
}

/// RPython `_find_property_meth(s_obj, attr, meth)` (unaryop.py:895-906).
///
/// Walks `s_obj.classdef.getmro()` and collects the `property` descriptor's
/// `fget` / `fset` / `fdel` method reference for `attr`. Returns `None`
/// when the attribute appears in any classdict entry that isn't a
/// `property` (mirrors upstream's `return` without a result).
///
/// `meth` selects which property slot to read — "fget" / "fset" / "fdel" —
/// matching upstream's `getattr(obj.value, meth)`.
fn find_property_meth(
    s_obj: &super::model::SomeInstance,
    attr: &str,
    meth: &str,
) -> Option<Vec<Option<super::super::flowspace::model::HostObject>>> {
    use super::super::flowspace::model::ConstValue;
    let classdef = s_obj.classdef.as_ref()?;
    let mro = super::classdesc::ClassDef::getmro(classdef);
    let mut result: Vec<Option<super::super::flowspace::model::HostObject>> = Vec::new();
    for clsdef in &mro {
        let classdesc = clsdef.borrow().classdesc.clone();
        let entry = classdesc.borrow().classdict.get(attr).cloned();
        let Some(entry) = entry else { continue };
        match entry {
            super::classdesc::ClassDictEntry::Constant(c) => {
                // upstream unaryop.py:902-904:
                //   if (not isinstance(obj, Constant) or
                //           not isinstance(obj.value, property)):
                //       return
                //   result.append(getattr(obj.value, meth))
                let ConstValue::HostObject(host) = &c.value else {
                    return None;
                };
                if !host.is_property() {
                    return None;
                }
                let slot = match meth {
                    "fget" => host.property_fget().cloned(),
                    "fset" => host.property_fset().cloned(),
                    "fdel" => host.property_fdel().cloned(),
                    _ => return None,
                };
                result.push(slot);
            }
            // Non-Constant classdict entries (FunctionDesc etc.) never
            // stand in for a property; upstream's isinstance-check falls
            // through to `return`.
            super::classdesc::ClassDictEntry::Desc(_) => return None,
        }
    }
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

fn init_instance_single_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    // unaryop.py:867-870 — len(v_arg) -> [getattr(v_arg, '__len__'), simple_call(getattr.result)]
    register_transform(
        reg,
        OpKind::Len,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_arg = args[0].clone();
            let get_len = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_arg,
                    Hlvalue::Constant(Constant::new(ConstValue::Str("__len__".into()))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_len.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
            Some(vec![get_len, call])
        }),
    );
    // unaryop.py:872-875 — iter
    register_transform(
        reg,
        OpKind::Iter,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_arg = args[0].clone();
            let get_iter = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_arg,
                    Hlvalue::Constant(Constant::new(ConstValue::Str("__iter__".into()))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_iter.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
            Some(vec![get_iter, call])
        }),
    );
    // unaryop.py:877-880 — next
    register_transform(
        reg,
        OpKind::Next,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_arg = args[0].clone();
            let get_next = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_arg,
                    Hlvalue::Constant(Constant::new(ConstValue::Str("next".into()))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_next.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result]);
            Some(vec![get_next, call])
        }),
    );
    // unaryop.py:882-885 — getslice
    register_transform(
        reg,
        OpKind::GetSlice,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_obj = args[0].clone();
            let v_start = args[1].clone();
            let v_stop = args[2].clone();
            let get_getslice = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_obj,
                    Hlvalue::Constant(Constant::new(ConstValue::Str("__getslice__".into()))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_getslice.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_start, v_stop]);
            Some(vec![get_getslice, call])
        }),
    );
    // unaryop.py:888-892 — setslice
    register_transform(
        reg,
        OpKind::SetSlice,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_obj = args[0].clone();
            let v_start = args[1].clone();
            let v_stop = args[2].clone();
            let v_iterable = args[3].clone();
            let get_setslice = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_obj,
                    Hlvalue::Constant(Constant::new(ConstValue::Str("__setslice__".into()))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_setslice.result.clone());
            let call = mk_hlop(
                OpKind::SimpleCall,
                vec![getattr_result, v_start, v_stop, v_iterable],
            );
            Some(vec![get_setslice, call])
        }),
    );
    // binaryop.py:744-747 — contains. `Contains` is Dispatch::Single
    // upstream; the Rust dispatch table agrees (flowspace/operation.rs
    // OpKind::Contains => Dispatch::Single). Transform body is the same
    // getattr+simple_call pair; the second arg is `v_idx`.
    register_transform(
        reg,
        OpKind::Contains,
        SomeValueTag::Instance,
        Box::new(|_ann, args| {
            let v_ins = args[0].clone();
            let v_idx = args[1].clone();
            let get_contains = mk_hlop(
                OpKind::GetAttr,
                vec![
                    v_ins,
                    Hlvalue::Constant(Constant::new(ConstValue::Str("__contains__".into()))),
                ],
            );
            let getattr_result = Hlvalue::Variable(get_contains.result.clone());
            let call = mk_hlop(OpKind::SimpleCall, vec![getattr_result, v_idx]);
            Some(vec![get_contains, call])
        }),
    );
}

// =====================================================================
// unaryop.py:909-936 — @op.{getattr,setattr}.register_transform(SomeInstance)
// =====================================================================

/// RPython `getattr_SomeInstance` / `setattr_SomeInstance`
/// (unaryop.py:909-936) — property descriptor dispatch.
///
/// Upstream rewrites `getattr(v_obj, 'prop')` into
/// `[getattr(v_obj, 'prop__getter__'), simple_call(getter.result)]`
/// when every classdef in the MRO has `prop` bound as a `property`
/// descriptor, and raises `AnnotatorError` when none of them do (the
/// "mixed property/non-property" case falls through with no rewrite).
/// `setattr` is the setter analogue — three-arg `simple_call(setter,
/// v_value)` instead.
fn init_instance_attr_transform(
    reg: &mut std::collections::HashMap<
        OpKind,
        std::collections::HashMap<SomeValueTag, Transformation>,
    >,
) {
    // unaryop.py:909-921 — getattr
    register_transform(
        reg,
        OpKind::GetAttr,
        SomeValueTag::Instance,
        Box::new(|ann, args| {
            let v_obj = args[0].clone();
            let v_attr = &args[1];
            // upstream: `s_attr = annotator.annotation(v_attr)`.
            let s_attr = ann.annotation(v_attr)?;
            // upstream: `if not s_attr.is_constant() or not isinstance(s_attr.const, str): return`.
            let attr = match s_attr.const_() {
                Some(ConstValue::Str(s)) => s.clone(),
                _ => return None,
            };
            // upstream: `getters = _find_property_meth(annotator.annotation(v_obj), attr, 'fget')`.
            let s_obj = ann.annotation(&v_obj)?;
            let super::model::SomeValue::Instance(ref s_inst) = s_obj else {
                return None;
            };
            let getters = find_property_meth(s_inst, &attr, "fget")?;
            // upstream: `if all(getters): … elif not any(getters): raise`.
            if getters.iter().all(|g| g.is_some()) {
                let getter_name = format!("{attr}__getter__");
                let get_getter = mk_hlop(
                    OpKind::GetAttr,
                    vec![
                        v_obj,
                        Hlvalue::Constant(Constant::new(ConstValue::Str(getter_name))),
                    ],
                );
                let getter_result = Hlvalue::Variable(get_getter.result.clone());
                let call = mk_hlop(OpKind::SimpleCall, vec![getter_result]);
                Some(vec![get_getter, call])
            } else if !getters.iter().any(|g| g.is_some()) {
                panic!("AnnotatorError: Attribute {attr:?} is unreadable");
            } else {
                None
            }
        }),
    );
    // unaryop.py:924-936 — setattr
    register_transform(
        reg,
        OpKind::SetAttr,
        SomeValueTag::Instance,
        Box::new(|ann, args| {
            let v_obj = args[0].clone();
            let v_attr = &args[1];
            let v_value = args[2].clone();
            let s_attr = ann.annotation(v_attr)?;
            let attr = match s_attr.const_() {
                Some(ConstValue::Str(s)) => s.clone(),
                _ => return None,
            };
            let s_obj = ann.annotation(&v_obj)?;
            let super::model::SomeValue::Instance(ref s_inst) = s_obj else {
                return None;
            };
            let setters = find_property_meth(s_inst, &attr, "fset")?;
            if setters.iter().all(|s| s.is_some()) {
                let setter_name = format!("{attr}__setter__");
                let get_setter = mk_hlop(
                    OpKind::GetAttr,
                    vec![
                        v_obj,
                        Hlvalue::Constant(Constant::new(ConstValue::Str(setter_name))),
                    ],
                );
                let setter_result = Hlvalue::Variable(get_setter.result.clone());
                let call = mk_hlop(OpKind::SimpleCall, vec![setter_result, v_value]);
                Some(vec![get_setter, call])
            } else if !setters.iter().any(|s| s.is_some()) {
                panic!("AnnotatorError: Attribute {attr:?} is unwritable");
            } else {
                None
            }
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::model::{ConstValue, Constant, Hlvalue, Variable};
    use super::*;

    fn mk_ann() -> Rc<RPythonAnnotator> {
        RPythonAnnotator::new(None, None, None, false)
    }

    fn hl1(op: OpKind, sv: SomeValue) -> (HLOperation, Rc<RPythonAnnotator>) {
        let ann = mk_ann();
        let mut v = Variable::named("x");
        ann.setbinding(&mut v, sv);
        let hl = HLOperation::new(op, vec![Hlvalue::Variable(v)]);
        (hl, ann)
    }

    #[test]
    fn unary_operations_list_excludes_contains() {
        assert!(!UNARY_OPERATIONS.contains(&OpKind::Contains));
        assert!(UNARY_OPERATIONS.contains(&OpKind::Len));
        assert!(UNARY_OPERATIONS.contains(&OpKind::Bool));
    }

    #[test]
    fn consider_getattr_prefers_find_method_on_lists() {
        let ann = mk_ann();
        let mut v = Variable::named("lst");
        ann.setbinding(
            &mut v,
            SomeValue::List(super::super::model::SomeList::new(
                super::super::listdef::ListDef::new(
                    Some(ann.bookkeeper.clone()),
                    SomeValue::Integer(SomeInteger::default()),
                    false,
                    false,
                ),
            )),
        );
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![
                Hlvalue::Variable(v),
                Hlvalue::Constant(Constant::new(ConstValue::Str("append".into()))),
            ],
        );

        let result = hl.consider(&ann).expect("getattr must succeed");

        let SomeValue::BuiltinMethod(method) = result else {
            panic!("expected SomeBuiltinMethod");
        };
        assert_eq!(method.analyser_name, "list_method_append");
        assert_eq!(method.methodname, "append");
    }

    #[test]
    fn transform_len_someinstance_rewrites_to_getattr_plus_simple_call() {
        // unaryop.py:867-870 — `op.len(v_ins)` → [getattr(v_ins, '__len__'), simple_call(r)].
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v = Variable::named("inst");
        ann.setbinding(
            &mut v,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let hl = HLOperation::new(OpKind::Len, vec![Hlvalue::Variable(v)]);
        let new_ops = hl
            .transform(&ann)
            .expect("Len(SomeInstance) must register_transform");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn transform_call_args_unpacks_tuple_stararg_to_simple_call() {
        // unaryop.py:120-139 — `f(*tup)` where tup has 3 items and no
        // keywords should rewrite to 3 getitem ops + one simple_call.
        use super::super::model::{SomeInteger, SomeTuple};
        let ann = mk_ann();
        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, SomeValue::object());
        let mut v_tup = Variable::named("tup");
        ann.setbinding(
            &mut v_tup,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
        );
        // `f(*tup)` encodes as: call_args(f, shape, tup) where shape =
        // (shape_cnt=0, shape_keys=(), shape_star=True).
        let shape_const = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(0),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(true),
        ])));
        let hl = HLOperation::new(
            OpKind::CallArgs,
            vec![
                Hlvalue::Variable(v_func),
                shape_const,
                Hlvalue::Variable(v_tup),
            ],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("call_args(SomeObject, *SomeTuple) must transform");
        // 3 getitem + 1 simple_call = 4 ops
        assert_eq!(new_ops.len(), 4);
        assert!(matches!(new_ops[0].kind, OpKind::GetItem));
        assert!(matches!(new_ops[1].kind, OpKind::GetItem));
        assert!(matches!(new_ops[2].kind, OpKind::GetItem));
        assert!(matches!(new_ops[3].kind, OpKind::SimpleCall));
        // simple_call receives v_func + 3 unpacked values
        assert_eq!(new_ops[3].args.len(), 4);
    }

    #[test]
    fn transform_call_args_without_stararg_is_noop() {
        // upstream: if callspec.w_stararg is None, transform_varargs
        // falls through without returning ops.
        let ann = mk_ann();
        let mut v_func = Variable::named("f");
        ann.setbinding(&mut v_func, SomeValue::object());
        let shape_const = Hlvalue::Constant(Constant::new(ConstValue::Tuple(vec![
            ConstValue::Int(0),
            ConstValue::Tuple(vec![]),
            ConstValue::Bool(false),
        ])));
        let hl = HLOperation::new(
            OpKind::CallArgs,
            vec![Hlvalue::Variable(v_func), shape_const],
        );
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_getattr_someinstance_without_property_is_noop() {
        // unaryop.py:909-921 — when the target attribute isn't a
        // property descriptor in any classdef, `_find_property_meth`
        // returns None and the transform leaves the op alone.
        use super::super::classdesc::ClassDef;
        use super::super::model::SomeInstance;
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::Str("prop".into())));
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_obj), v_attr_const],
        );
        // Registered but returns None because the classdict carries no
        // `property` descriptor (Rust HostObject property variant not
        // yet modelled).
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_setattr_someinstance_without_property_is_noop() {
        // unaryop.py:924-936 setter analogue.
        use super::super::classdesc::ClassDef;
        use super::super::model::{SomeInstance, SomeInteger};
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let mut v_value = Variable::named("val");
        ann.setbinding(&mut v_value, SomeValue::Integer(SomeInteger::default()));
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::Str("prop".into())));
        let hl = HLOperation::new(
            OpKind::SetAttr,
            vec![
                Hlvalue::Variable(v_obj),
                v_attr_const,
                Hlvalue::Variable(v_value),
            ],
        );
        assert!(hl.transform(&ann).is_none());
    }

    #[test]
    fn transform_getattr_someinstance_with_property_rewrites_via_getter_hidden_fn() {
        // unaryop.py:909-921 — when every classdef in the MRO binds
        // `prop` as a `property` descriptor, the transform rewrites
        // `getattr(v_obj, 'prop')` into `[getattr(v_obj, 'prop__getter__'),
        // simple_call(getter.result)]`.
        use super::super::super::flowspace::model::GraphFunc;
        use super::super::classdesc::{ClassDef, ClassDictEntry};
        use super::super::model::SomeInstance;
        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.Y", None);
        // Build a property(fget, fset) and bind it as the `prop` entry
        // in the classdesc's classdict — simulating what
        // `ClassDesc::add_source_attribute` produces when processing a
        // property descriptor.
        let fget_host =
            super::super::super::flowspace::model::HostObject::new_user_function(GraphFunc::new(
                "get_prop",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
        let fset_host =
            super::super::super::flowspace::model::HostObject::new_user_function(GraphFunc::new(
                "set_prop",
                Constant::new(ConstValue::Dict(Default::default())),
            ));
        let prop_host = super::super::super::flowspace::model::HostObject::new_property(
            "pkg.Y.prop",
            Some(fget_host),
            Some(fset_host),
            None,
        );
        let classdesc = classdef.borrow().classdesc.clone();
        classdesc.borrow_mut().classdict.insert(
            "prop".to_string(),
            ClassDictEntry::constant(ConstValue::HostObject(prop_host)),
        );
        let mut v_obj = Variable::named("obj");
        ann.setbinding(
            &mut v_obj,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let v_attr_const = Hlvalue::Constant(Constant::new(ConstValue::Str("prop".into())));
        let hl = HLOperation::new(
            OpKind::GetAttr,
            vec![Hlvalue::Variable(v_obj), v_attr_const],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("property attr must rewrite to getter+simple_call");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        // The rewritten getattr target is the hidden getter name.
        let target_attr = &new_ops[0].args[1];
        match target_attr {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Str(s) => assert_eq!(s, "prop__getter__"),
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant attr, got {other:?}"),
        }
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn transform_getitem_someinstance_rewrites_to_getattr_plus_simple_call() {
        // binaryop.py:727-730 — `op.getitem(v_ins, v_idx)` → getattr + simple_call.
        use super::super::classdesc::ClassDef;
        use super::super::model::{SomeInstance, SomeInteger};

        let ann = mk_ann();
        let classdef = ClassDef::new_standalone("pkg.X", None);
        let mut v_ins = Variable::named("inst");
        ann.setbinding(
            &mut v_ins,
            SomeValue::Instance(SomeInstance::new(
                Some(classdef),
                false,
                std::collections::BTreeMap::new(),
            )),
        );
        let mut v_idx = Variable::named("i");
        ann.setbinding(&mut v_idx, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::GetItem,
            vec![Hlvalue::Variable(v_ins), Hlvalue::Variable(v_idx)],
        );
        let new_ops = hl
            .transform(&ann)
            .expect("GetItem(SomeInstance, SomeObject) must register_transform");
        assert_eq!(new_ops.len(), 2);
        assert!(matches!(new_ops[0].kind, OpKind::GetAttr));
        assert!(matches!(new_ops[1].kind, OpKind::SimpleCall));
    }

    #[test]
    fn consider_someobject_len_returns_nonneg_integer() {
        // unaryop.py:158-159 — default len returns SomeInteger(nonneg=True).
        let (hl, ann) = hl1(OpKind::Len, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg, "nonneg not set"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somefloat_neg_returns_somefloat() {
        let (hl, ann) = hl1(OpKind::Neg, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somefloat_pos_returns_self() {
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someinteger_abs_is_nonneg() {
        let (hl, ann) = hl1(OpKind::Abs, SomeValue::Integer(SomeInteger::default()));
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somebool_bool_returns_self() {
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somebool_invert_returns_someinteger() {
        let (hl, ann) = hl1(OpKind::Invert, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_sometuple_len_is_constant() {
        // unaryop.py:334-335 — len returns immutablevalue(len(items)).
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Float(SomeFloat::new()),
                SomeValue::Bool(SomeBool::new()),
            ])),
        );
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(3));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_sometuple_iter_returns_someiterator() {
        let (hl, ann) = hl1(OpKind::Iter, SomeValue::Tuple(SomeTuple::new(vec![])));
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_str_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Str, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_hex_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Hex, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_pos_returns_impossible() {
        // unaryop.py:252-254 — default pos returns s_ImpossibleValue.
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::object());
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    fn mk_list_of(item: SomeValue) -> super::super::model::SomeList {
        let ann = mk_ann();
        let listdef = super::super::listdef::ListDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            item,
            false,
            false,
        );
        super::super::model::SomeList::new(listdef)
    }

    #[test]
    fn consider_somelist_contains_returns_bool() {
        let ann = mk_ann();
        let mut v0 = Variable::named("lst");
        let mut v1 = Variable::named("x");
        ann.setbinding(
            &mut v0,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somelist_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somedict_iter_returns_iterator() {
        let ann = mk_ann();
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let mut v = Variable::named("d");
        ann.setbinding(&mut v, SomeValue::Dict(s_dict));
        let hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_len_const_returns_const_int() {
        // unaryop.py:720-725 — len(const "abc") → SomeInteger(const=3).
        let mut s = SomeString::new(false, false);
        s.inner.base.const_box = Some(Constant::new(ConstValue::Str("abc".into())));
        let (hl, ann) = hl1(OpKind::Len, SomeValue::String(s));
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(3));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somechar_len_is_const_1() {
        // unaryop.py:771-774 — SomeChar.len() → immutablevalue(1).
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::Char(super::super::model::SomeChar::new(false)),
        );
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => {
                let c = i.base.const_box.expect("const not propagated");
                assert_eq!(c.value, ConstValue::Int(1));
            }
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somechar_ord_returns_nonneg_integer() {
        let (hl, ann) = hl1(
            OpKind::Ord,
            SomeValue::Char(super::super::model::SomeChar::new(false)),
        );
        let r = hl.consider(&ann).unwrap();
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_someiterator_iter_returns_self() {
        // unaryop.py:808-810 — iter on iterator returns self.
        let ann = mk_ann();
        let it = SomeIterator::new(SomeValue::Tuple(SomeTuple::new(vec![])), vec![]);
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Iter, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_on_tuple_unions_items() {
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ])),
            vec![],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_enumerate_returns_tuple_int_item() {
        // upstream unaryop.py:822-824 — enumerate variant yields
        // SomeTuple((SomeInteger(nonneg=True), getanyitem(container))).
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
            vec!["enumerate".to_string()],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        let SomeValue::Tuple(t) = r else {
            panic!("expected SomeTuple");
        };
        assert_eq!(t.items.len(), 2);
        assert!(matches!(t.items[0], SomeValue::Integer(_)));
        assert!(matches!(t.items[1], SomeValue::Integer(_)));
    }

    #[test]
    fn consider_someiterator_next_reversed_acts_as_default() {
        // upstream unaryop.py:826-827 — `("reversed",)` collapses to ().
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::Tuple(SomeTuple::new(vec![SomeValue::Integer(
                SomeInteger::default(),
            )])),
            vec!["reversed".to_string()],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Integer(_)));
    }

    #[test]
    fn consider_someiterator_next_on_string_returns_char() {
        use super::super::model::SomeString;
        let ann = mk_ann();
        let it = SomeIterator::new(SomeValue::String(SomeString::new(false, true)), vec![]);
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Char(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someiterator_next_on_unicode_returns_codepoint() {
        let ann = mk_ann();
        let it = SomeIterator::new(
            SomeValue::UnicodeString(SomeUnicodeString::new(false, true)),
            vec![],
        );
        let mut v = Variable::named("it");
        ann.setbinding(&mut v, SomeValue::Iterator(it));
        let hl = HLOperation::new(OpKind::Next, vec![Hlvalue::Variable(v)]);
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::UnicodeCodePoint(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somenone_getattr_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::GetAttr,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somenone_len_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::String(SomeString::new(false, false)),
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somedict_contains_position_required() {
        let ann = mk_ann();
        let pk = super::super::bookkeeper::PositionKey::new(3, 0, 0);
        ann.bookkeeper.set_position_key(Some(pk));
        let dictdef = super::super::dictdef::DictDef::new(
            Some(Rc::clone(&ann.bookkeeper)),
            SomeValue::Integer(SomeInteger::default()),
            SomeValue::String(SomeString::new(false, false)),
            false,
            false,
            false,
        );
        let s_dict = super::super::model::SomeDict::new(dictdef);
        let mut v0 = Variable::named("d");
        let mut v1 = Variable::named("k");
        ann.setbinding(&mut v0, SomeValue::Dict(s_dict));
        ann.setbinding(&mut v1, SomeValue::Integer(SomeInteger::default()));
        let hl = HLOperation::new(
            OpKind::Contains,
            vec![Hlvalue::Variable(v0), Hlvalue::Variable(v1)],
        );
        let r = hl.consider(&ann).unwrap();
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }
}
