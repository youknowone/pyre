//! ClassRepr-equivalent vtable extraction.
//!
//! RPython equivalent: `rpython/rtyper/rclass.py` `ClassRepr`.
//!
//! RPython models the per-class vtable as an explicit `Struct` and emits
//! `cast_pointer + getfield(vtable_struct, method_name)` to project a
//! method slot into a `Ptr(FuncType)` variable
//! ([`getclsfield`](https://github.com/pypy/pypy/blob/main/rpython/rtyper/rclass.py#L371-L377)).
//! Pyre cannot model Rust's `dyn Trait` vtable as an IR struct because
//! the layout is compiler-internal (unstable ABI), so this module emits
//! the single condensed op `OpKind::VtableMethodPtr` instead — see the
//! PRE-EXISTING-ADAPTATION block on that variant in `model.rs`.

use std::sync::Arc;

use crate::model::{BlockId, FunctionGraph, OpKind, SpaceOperation, ValueId};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::rmodel::Repr;
use crate::translator::rtyper::rtyper::RPythonTyper;
// `TypeResolutionState` + `ConcreteType` live at
// `jit_codewriter/type_state.rs` — a PRE-EXISTING-ADAPTATION side table
// because pyre's jit_codewriter IR is value-id-based while RPython stores
// `.concretetype` inline on `Variable`. The resolve_types / build_value_kinds
// algorithms still live under translate_legacy until the real rtyper
// replaces them end-to-end.
use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};

/// Insert a `VtableMethodPtr` op at `(block_id, op_index)` and return the
/// produced funcptr ValueId. Updates `type_state` so downstream passes
/// (`build_value_kinds`, regalloc, flatten) see the funcptr as integer
/// kind — matching `int_guard_value(op.args[0])` in
/// `jtransform.py:546`.
///
/// RPython equivalent: `ClassRepr.getclsfield(vcls, attr, llops)`
/// (`rclass.py:371-377`), which appends `cast_pointer + getfield(vtable,
/// mangled_name)` to `llops`.
pub fn class_get_method_ptr(
    graph: &mut FunctionGraph,
    type_state: &mut TypeResolutionState,
    block_id: BlockId,
    op_index: usize,
    receiver: ValueId,
    trait_root: String,
    method_name: String,
) -> ValueId {
    let funcptr = graph.alloc_value();
    let op = SpaceOperation {
        result: Some(funcptr),
        kind: OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        },
    };
    graph.blocks[block_id.0].operations.insert(op_index, op);
    type_state
        .concrete_types
        .insert(funcptr, ConcreteType::Signed);
    funcptr
}

/// RPython `rclass.get_type_repr(rtyper)` (`rpython/rtyper/rclass.py:439-440`).
///
/// ```python
/// def get_type_repr(rtyper):
///     return rtyper.rootclass_repr
/// ```
///
/// Upstream reads `rtyper.rootclass_repr` directly. The Rust port has
/// not yet landed `rootclass_repr` (it depends on the `getclassrepr` /
/// class-tree machinery), so this surface proxies through
/// `ExceptionData.r_exception_type`, which upstream initialises from
/// the same `rtyper.rootclass_repr` value at
/// `exceptiondata.py:18`. The error returned when `ExceptionData` is
/// not yet set points callers at the missing dependency.
pub fn get_type_repr(rtyper: &RPythonTyper) -> Result<Arc<dyn Repr>, TyperError> {
    Ok(rtyper.exceptiondata()?.r_exception_type.clone())
}
