//! Pre-jtransform fold of unit-variant `SyntheticTransparentCtor` calls.
//!
//! RPython parity: `rtyper/rpbc.py::SingleFrozenPBCRepr` resolves a
//! frozen PBC constructor that has no arguments (a unit variant
//! `StepResult::Continue`, `JitAction::Continue`, …) into a singleton
//! `Constant(prebuilt_instance_ptr)` before `jit_codewriter/jtransform`
//! ever sees the call.  See also
//! `rclass.InstanceRepr.get_reusable_prebuilt_instance`.
//!
//! Pyre's frontend (`front/ast.rs:5642`) lowers a unit-variant path
//! expression to `OpKind::Call { target: SyntheticTransparentCtor,
//! args: [] }`.  The companion fold inside
//! `translator/rtyper/flowspace_adapter.rs::legacy_const_define_hlvalue`
//! covers graphs that traverse the Match arm of the dual gate, but
//! per-opcode arm body graphs registered via
//! `register_function_graph` take the Skip arm and bypass that fold.
//! The residual `Call` op then survives into jtransform and is emitted
//! as a `residual_call_r/d>r` wrapper, which blocks
//! `production_walker_handles` activation (Task #333).
//!
//! This pass operates directly on `model::FunctionGraph` after
//! `lower_indirect_calls` and before `Transformer::transform`, so it
//! catches both gate arms.  HostObject identity is interned per pass
//! by qualname so multiple ops referencing the same unit variant
//! within one graph share the same Arc — the assembler's
//! `emit_const_r` dedupes the ref-bank constant pool by
//! `obj.identity_id()`, so per-pass interning is sufficient.

use crate::flowspace::model::HostObject;
use crate::model::{CallTarget, FunctionGraph, OpKind};

/// Rewrite `OpKind::Call { target: SyntheticTransparentCtor, args: [] }`
/// ops whose qualified path matches
/// `front::ast::is_synthetic_unit_variant_path` into
/// `OpKind::ConstRef(prebuilt_instance)`, mirroring
/// `rtyper/rpbc.py::SingleFrozenPBCRepr`.
pub fn fold_unit_variant_ctors(graph: &mut FunctionGraph) {
    let mut interned: Vec<(String, HostObject)> = Vec::new();
    for block in graph.blocks.iter_mut() {
        for op in block.operations.iter_mut() {
            let OpKind::Call {
                target: CallTarget::SyntheticTransparentCtor { name, owner_path },
                args,
                ..
            } = &op.kind
            else {
                continue;
            };
            if !args.is_empty() {
                continue;
            }
            let mut segments = owner_path.clone();
            segments.push(name.clone());
            if !crate::front::ast::is_synthetic_unit_variant_path(&segments) {
                continue;
            }
            let qualname = segments.join(".");
            let instance = match interned.iter().find(|(q, _)| q == &qualname) {
                Some((_, obj)) => obj.clone(),
                None => {
                    let class_obj = HostObject::new_class(qualname.clone(), Vec::new());
                    let Some(instance) = class_obj.reusable_prebuilt_instance() else {
                        continue;
                    };
                    interned.push((qualname, instance.clone()));
                    instance
                }
            };
            op.kind = OpKind::ConstRef(instance);
        }
    }
}
