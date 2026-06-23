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
/// `rvirtualizable.py`; wiring it into `rclass::buildinstancerepr` is blocked
/// on the `FieldListAccessor` / `_parse_field_list` surface in `rclass.py`.
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
                    if op.args.last().is_some_and(|arg| flags_access_directly(arg)) {
                        newoplist.push(op);
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

        let count = replace_force_virtualizable_with_call(&[graph.clone()], &VTYPEPTR, &funcptr);

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
    fn replace_force_virtualizable_preserves_access_directly_ops() {
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

        let count = replace_force_virtualizable_with_call(&[graph.clone()], &VTYPEPTR, &funcptr);

        assert_eq!(count, 0);
        assert_eq!(
            graph.borrow().startblock.borrow().operations[0].opname,
            "jit_force_virtualizable"
        );
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
