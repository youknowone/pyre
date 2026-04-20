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
use super::super::flowspace::operation::{
    BuiltinException, CanOnlyThrow, HLOperation, OpKind, Specialization, register_single,
};
use super::annrpython::RPythonAnnotator;
use super::model::{
    SomeBool, SomeFloat, SomeInteger, SomeIterator, SomeObjectTrait, SomeString, SomeTuple,
    SomeTypeOf, SomeUnicodeString, SomeValue, SomeValueTag, s_impossible_value,
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
                // upstream: `SomeTypeOf([v_arg])`.
                let v_name = match &hl.args[0] {
                    super::super::flowspace::model::Hlvalue::Variable(v) => v.name(),
                    super::super::flowspace::model::Hlvalue::Constant(_) => {
                        // `type(const)` — upstream still builds SomeTypeOf
                        // since `args_v` carries the Hlvalue; we preserve
                        // the same shape with an empty-name marker.
                        String::new()
                    }
                };
                SomeValue::TypeOf(SomeTypeOf::new(vec![v_name]))
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
        SomeValue::Integer(SomeInteger::new_with_knowntype(false, false, kt))
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
        let mut i = SomeInteger::new_with_knowntype(false, false, kt);
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
                let position = ann
                    .bookkeeper
                    .current_position_key()
                    .expect("list.len: position_key is None");
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
// `SomeBuiltinMethod` for downstream `simple_call`. Commit 6d wires up
// the `find_method` → `SomeBuiltinMethod.call` dispatch; for now these
// stay as independent free functions that Commit 6d will dispatch to.

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
    let position = ann
        .bookkeeper
        .current_position_key()
        .expect("list.pop: position_key is None");
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
    position: super::bookkeeper::PositionKey,
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
fn dict_is_empty(s_dct: &super::model::SomeDict, position: super::bookkeeper::PositionKey) -> bool {
    let s_key = s_dct.dictdef.read_key(position);
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
                let position = ann
                    .bookkeeper
                    .current_position_key()
                    .expect("contains(SomeDict): position_key is None");
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
                let position = ann
                    .bookkeeper
                    .current_position_key()
                    .expect("dict.len: position_key is None");
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
    let position = ann
        .bookkeeper
        .current_position_key()
        .expect("dict.get: position_key is None");
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
    let position = ann
        .bookkeeper
        .current_position_key()
        .expect("dict.keys: position_key is None");
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
    let position = ann
        .bookkeeper
        .current_position_key()
        .expect("dict.values: position_key is None");
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
    let position = ann
        .bookkeeper
        .current_position_key()
        .expect("dict.pop: position_key is None");
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
    //   StopIteration-throwable iteration step. Full body depends on
    //   SomeValue.getanyitem (List/Dict/Tuple variant-aware), which will
    //   land with the iterator-body commit. Until then register a
    //   minimal implementation that returns the container's canonical
    //   item type — enough for Tuple/List; Dict variants need the
    //   variant-aware branch.
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
                match &*it.s_container {
                    // upstream: `if s_None.contains(self.s_container): return s_ImpossibleValue`.
                    SomeValue::None_(_) => s_impossible_value(),
                    SomeValue::Tuple(t) => {
                        if t.items.is_empty() {
                            s_impossible_value()
                        } else {
                            super::model::unionof(t.items.iter())
                                .expect("iterator.next(tuple): unionof failed")
                        }
                    }
                    SomeValue::List(l) => {
                        let position = ann
                            .bookkeeper
                            .current_position_key()
                            .expect("iterator.next(list): position_key is None");
                        l.listdef.read_item(position)
                    }
                    SomeValue::Dict(d) => {
                        let position = ann
                            .bookkeeper
                            .current_position_key()
                            .expect("iterator.next(dict): position_key is None");
                        // upstream variant-aware branch (keys / values / items /
                        // items_with_hash / keys_with_hash). Minimal port handles
                        // the default (keys) path — the full variant switch
                        // needs SomeDict.getanyitem and lands with the
                        // dict-iterator commit.
                        match it.variant.first().map(String::as_str) {
                            None | Some("keys") => d.dictdef.read_key(position),
                            Some("values") => d.dictdef.read_value(position),
                            Some(other) => panic!(
                                "iterator.next(dict variant={}): lands with dict-iterator commit",
                                other
                            ),
                        }
                    }
                    SomeValue::String(_)
                    | SomeValue::UnicodeString(_)
                    | SomeValue::ByteArray(_) => {
                        panic!("iterator.next(string): lands with string-iterator commit")
                    }
                    other => panic!("iterator.next: unsupported container {:?}", other),
                }
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

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::model::{Hlvalue, Variable};
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
    fn consider_someobject_len_returns_nonneg_integer() {
        // unaryop.py:158-159 — default len returns SomeInteger(nonneg=True).
        let (hl, ann) = hl1(OpKind::Len, SomeValue::object());
        let r = hl.consider(&ann);
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg, "nonneg not set"),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somefloat_neg_returns_somefloat() {
        let (hl, ann) = hl1(OpKind::Neg, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somefloat_pos_returns_self() {
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::Float(SomeFloat::new()));
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Float(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someinteger_abs_is_nonneg() {
        let (hl, ann) = hl1(OpKind::Abs, SomeValue::Integer(SomeInteger::default()));
        let r = hl.consider(&ann);
        match r {
            SomeValue::Integer(i) => assert!(i.nonneg),
            other => panic!("got {:?}", other),
        }
    }

    #[test]
    fn consider_somebool_bool_returns_self() {
        let (hl, ann) = hl1(OpKind::Bool, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somebool_invert_returns_someinteger() {
        let (hl, ann) = hl1(OpKind::Invert, SomeValue::Bool(SomeBool::new()));
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_str_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Str, SomeValue::object());
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_hex_returns_somestring() {
        let (hl, ann) = hl1(OpKind::Hex, SomeValue::object());
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::String(_)), "got {:?}", r);
    }

    #[test]
    fn consider_someobject_pos_returns_impossible() {
        // unaryop.py:252-254 — default pos returns s_ImpossibleValue.
        let (hl, ann) = hl1(OpKind::Pos, SomeValue::object());
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somelist_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::List(mk_list_of(SomeValue::Integer(SomeInteger::default()))),
        );
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Iterator(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_len_const_returns_const_int() {
        // unaryop.py:720-725 — len(const "abc") → SomeInteger(const=3).
        let mut s = SomeString::new(false, false);
        s.inner.base.const_box = Some(Constant::new(ConstValue::Str("abc".into())));
        let (hl, ann) = hl1(OpKind::Len, SomeValue::String(s));
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Integer(_)), "got {:?}", r);
    }

    #[test]
    fn consider_somenone_getattr_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::GetAttr,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somenone_len_returns_impossible() {
        let (hl, ann) = hl1(
            OpKind::Len,
            SomeValue::None_(super::super::model::SomeNone::new()),
        );
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Impossible), "got {:?}", r);
    }

    #[test]
    fn consider_somestring_iter_returns_iterator() {
        let (hl, ann) = hl1(
            OpKind::Iter,
            SomeValue::String(SomeString::new(false, false)),
        );
        let r = hl.consider(&ann);
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
        let r = hl.consider(&ann);
        assert!(matches!(r, SomeValue::Bool(_)), "got {:?}", r);
    }
}
