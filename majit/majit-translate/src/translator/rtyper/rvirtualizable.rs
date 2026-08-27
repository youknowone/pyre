//! RPython `rpython/rtyper/rvirtualizable.py`.
#![allow(non_snake_case)]

use std::collections::HashMap;
use std::rc::Rc;

use crate::flowspace::model::{ConstValue, Constant, FunctionGraph, Hlvalue};
use crate::translator::rtyper::lltypesystem::lltype::{self, _ptr, LowLevelType};

/// RPython `class VirtualizableInstanceRepr(InstanceRepr)`.
///
/// Pyre's `InstanceRepr` is a single Rust struct rather than a Python
/// subclass hierarchy. This carrier records the extra state introduced by
/// `rvirtualizable.py`.
///
/// It has no production caller, and the reason is not a missing piece of
/// `rclass.py`. Upstream discovers the redirected field set by reading
/// `classdesc.get_param('_virtualizable_')` off an RPython class and then
/// injects a `jit_force_virtualizable` per redirected access
/// (`hook_access_field`, rvirtualizable.py) into the FLOWGRAPHS it is
/// about to rtype. Pyre's interpreter is hand-written Rust, so no class
/// carries that parameter — nothing writes the key — and the rtyper's
/// lowered op stream is not what the production jitcode is built from.
/// An injected op would land in a list the production path discards.
///
/// The field set itself is therefore declared out of band, as a data table
/// (`pyre-jit-trace/src/virtualizable_spec.rs`), the same way
/// `_immutable_fields_` is handled.
///
/// That table is also what arms this crate. `pyre-jit-trace`'s build script
/// builds `GraphTransformConfig::vable_fields` and `vable_arrays` out of it
/// (`build/prepass.rs`, behind the default-on `prepass` feature), and that
/// run is the production translation. The emitted artefacts show the whole
/// protocol firing: the instruction table carries `getfield_vable_i/r`,
/// `setfield_vable_i/r`, `getarrayitem_vable_r`, `setarrayitem_vable_r` and
/// `arraylen_vable`, and the descr table carries all five
/// `BhDescr::VableField` indices, which `assembler.rs` mints only from the
/// `VableFieldRead` / `VableFieldWrite` arms.
///
/// This crate's own `generated::build` still passes
/// `GraphTransformConfig::default()`, leaving `vable_fields` and
/// `vable_arrays` empty, so along that path `jtransform.rs
/// check_no_vable_array` returns at its `vable_array_vars.is_empty()` guard
/// and `rewrite_op_getfield`'s `VableFieldRead` arm never fires. That path
/// is a test fixture — the only caller of `generated::with_all_jitcodes` is
/// `tests/test_make_jitcodes_produces_graph_keyed_output.rs` — so it
/// diverges from the production config, not from upstream.
///
/// What has no counterpart either way is the force injection.
/// `hook_access_field` puts a `jit_force_virtualizable` in front of every
/// redirected access while rtyping; pyre instead places the forces by hand
/// at the consumers that need them (`pyre-interpreter` `sys.getframe`,
/// `f_locals`, `f_back`). The marker
/// `executioncontext::jit_force_virtualizable` is what lets
/// `jtransform.rs rewrite_op_jit_force_virtualizable` delete such a call from
/// a looked-inside graph, and it keys on the CALL TARGET's name rather than
/// on `vable_fields`, so the two halves are armed independently of each
/// other.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VirtualizableInstanceRepr {
    pub top_of_virtualizable_hierarchy: bool,
    pub my_redirected_fields: HashMap<String, bool>,
}

impl VirtualizableInstanceRepr {
    pub fn new(top_of_virtualizable_hierarchy: bool) -> Self {
        VirtualizableInstanceRepr {
            top_of_virtualizable_hierarchy,
            my_redirected_fields: HashMap::new(),
        }
    }

    /// RPython `_setup_repr_llfields`.
    pub fn setup_repr_llfields(&self) -> Vec<(String, LowLevelType)> {
        if self.top_of_virtualizable_hierarchy {
            vec![("vable_token".to_string(), lltype::GCREF.clone())]
        } else {
            vec![]
        }
    }

    /// RPython `hook_access_field`: force only redirected fields.
    pub fn should_force_field(&self, mangled_name: &str) -> bool {
        self.my_redirected_fields
            .get(mangled_name)
            .copied()
            .unwrap_or(false)
    }
}

/// RPython `replace_force_virtualizable_with_call(graphs, VTYPEPTR,
/// funcptr)`.
pub fn replace_force_virtualizable_with_call(
    graphs: &[Rc<std::cell::RefCell<FunctionGraph>>],
    VTYPEPTR: &LowLevelType,
    funcptr: &_ptr,
) -> usize {
    let c_funcptr = Hlvalue::Constant(Constant::with_concretetype(
        ConstValue::LLPtr(Box::new(funcptr.clone())),
        LowLevelType::Ptr(Box::new(lltype::typeOf(funcptr))),
    ));
    let mut count = 0;
    for graph in graphs {
        for block in graph.borrow().iterblocks() {
            let mut block = block.borrow_mut();
            let mut newoplist = Vec::with_capacity(block.operations.len());
            for mut op in block.operations.drain(..) {
                if op.opname == "jit_force_virtualizable"
                    && op.args.first().and_then(hlvalue_concretetype).as_ref() == Some(VTYPEPTR)
                {
                    if op.args.last().is_some_and(flags_access_directly) {
                        // `continue` before the append: an `access_directly`
                        // read reaches the virtualizable without going through
                        // the JIT, so the op is dropped rather than rewritten.
                        // Nothing downstream understands the `jit_force_virtualizable`
                        // opname, so leaving it in the block is not a weaker
                        // rewrite but an unlowerable operation.
                        continue;
                    }
                    op.opname = "direct_call".to_string();
                    op.args = vec![c_funcptr.clone(), op.args[0].clone()];
                    count += 1;
                }
                newoplist.push(op);
            }
            block.operations = newoplist;
        }
    }
    count
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

fn flags_access_directly(value: &Hlvalue) -> bool {
    let Hlvalue::Constant(c) = value else {
        return false;
    };
    let ConstValue::Dict(items) = &c.value else {
        return false;
    };
    let key = ConstValue::byte_str("access_directly");
    matches!(items.get(&key), Some(ConstValue::Bool(true)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::Variable;
    use crate::flowspace::model::{Block, SpaceOperation};
    use crate::translator::rtyper::lltypesystem::lltype::{FuncType, functionptr};

    fn force_op(vable: &Variable, access_directly: bool) -> SpaceOperation {
        let mut flags = HashMap::new();
        if access_directly {
            flags.insert(
                ConstValue::byte_str("access_directly"),
                ConstValue::Bool(true),
            );
        }
        let c_name =
            Constant::with_concretetype(ConstValue::byte_str("inst_x"), LowLevelType::Void);
        let c_flags = Constant::with_concretetype(ConstValue::Dict(flags), LowLevelType::Void);
        SpaceOperation::new(
            "jit_force_virtualizable",
            vec![
                Hlvalue::Variable(vable.clone()),
                Hlvalue::Constant(c_name),
                Hlvalue::Constant(c_flags),
            ],
            Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::None,
                LowLevelType::Void,
            )),
        )
    }

    fn graph_with_ops(ops: Vec<SpaceOperation>) -> Rc<std::cell::RefCell<FunctionGraph>> {
        let block = Block::shared(vec![]);
        block.borrow_mut().operations = ops;
        Rc::new(std::cell::RefCell::new(FunctionGraph::new("g", block)))
    }

    #[test]
    fn replace_force_virtualizable_rewrites_matching_non_direct_access() {
        let VTYPEPTR = lltype::GCREF.clone();
        let vable = Variable::named("vable");
        vable.set_concretetype(Some(VTYPEPTR.clone()));
        let graph = graph_with_ops(vec![force_op(&vable, false)]);
        let funcptr = functionptr(
            FuncType {
                args: vec![VTYPEPTR.clone()],
                result: LowLevelType::Void,
            },
            "force",
            None,
            Some("force".to_string()),
        );

        let count = replace_force_virtualizable_with_call(
            std::slice::from_ref(&graph),
            &VTYPEPTR,
            &funcptr,
        );

        assert_eq!(count, 1);
        let graph_ref = graph.borrow();
        let start = graph_ref.startblock.borrow();
        let op = &start.operations[0];
        assert_eq!(op.opname, "direct_call");
        assert_eq!(op.args.len(), 2);
        assert!(matches!(op.args[0], Hlvalue::Constant(_)));
        assert_eq!(op.args[1], Hlvalue::Variable(vable));
    }

    #[test]
    fn replace_force_virtualizable_drops_access_directly_ops() {
        let VTYPEPTR = lltype::GCREF.clone();
        let vable = Variable::named("vable");
        vable.set_concretetype(Some(VTYPEPTR.clone()));
        let graph = graph_with_ops(vec![force_op(&vable, true)]);
        let funcptr = functionptr(
            FuncType {
                args: vec![VTYPEPTR.clone()],
                result: LowLevelType::Void,
            },
            "force",
            None,
            Some("force".to_string()),
        );

        let count = replace_force_virtualizable_with_call(
            std::slice::from_ref(&graph),
            &VTYPEPTR,
            &funcptr,
        );

        assert_eq!(count, 0);
        assert!(graph.borrow().startblock.borrow().operations.is_empty());
    }

    #[test]
    fn setup_repr_llfields_adds_vable_token_only_at_hierarchy_root() {
        assert_eq!(
            VirtualizableInstanceRepr::new(true).setup_repr_llfields(),
            vec![("vable_token".to_string(), lltype::GCREF.clone())]
        );
        assert!(
            VirtualizableInstanceRepr::new(false)
                .setup_repr_llfields()
                .is_empty()
        );
    }
}
