//! `Result::map_err(result, closure)` → discriminant closure-select.
//!
//! Rust's foreign `core::result` body is opaque in LLBC.  Leaving the method
//! call residual is not executable for an embedded interpreter because there
//! is no monomorphisation-independent host address for its closure type.  The
//! equivalent RPython graph already has the ordinary two exception/value
//! branches (`translator/exceptiontransform.py`); this adapter restores that
//! shape before `result_exc` converts an exception-carrying result to native
//! graph exception edges.

use crate::flowspace::model::Variable;
use crate::front::bool_then::{
    close_goto_mixed, emit_sum_variant, map_source, reproduce_exit_args,
};
use crate::front::option_closure_select::emit_call_once;
use crate::front::option_map_or::emit_narrow;
use crate::model::{
    CallTarget, FieldDescriptor, FunctionGraph, LinkArg, OpKind, SpaceOperation, ValueType,
};

#[derive(Clone)]
pub(crate) struct ResultMapErrSite {
    pub result_var: Variable,
    pub receiver_owner: String,
    pub receiver_ok_owner: String,
    pub receiver_err_owner: String,
    pub result_owner: String,
    pub result_ok_owner: String,
    pub result_err_owner: String,
    pub call_once_owner: String,
    pub ok_ty: ValueType,
    pub ok_class_root: Option<String>,
    pub err_ty: ValueType,
    pub err_class_root: Option<String>,
    pub mapped_err_ty: ValueType,
    pub mapped_err_class_root: Option<String>,
    pub args_tuple_suffix: String,
    /// True only when every captured field recursively needs no destructor.
    /// Other closure environments stay on the fail-closed path until MIR Drop
    /// lowering can preserve their conditional destruction on the Ok arm.
    pub closure_env_is_trivially_dropless: bool,
}

pub(crate) fn rewire_result_map_err_sites(
    graph: &mut FunctionGraph,
    sites: &[ResultMapErrSite],
) -> usize {
    sites
        .iter()
        .filter(|site| rewire_one(graph, site).is_ok())
        .count()
}

fn rewire_one(graph: &mut FunctionGraph, site: &ResultMapErrSite) -> Result<(), String> {
    let name = graph.name.clone();
    if !site.closure_env_is_trivially_dropless {
        return Err(format!(
            "{name}: Result::map_err closure captures values whose Ok-arm destruction is not lowered"
        ));
    }
    let a = graph
        .blocks
        .iter()
        .position(|block| {
            block
                .operations
                .iter()
                .any(|op| op.result.as_ref() == Some(&site.result_var))
        })
        .ok_or_else(|| format!("{name}: Result::map_err result has no producer block"))?;
    let call_idx = graph.blocks[a]
        .operations
        .iter()
        .position(|op| op.result.as_ref() == Some(&site.result_var))
        .ok_or_else(|| format!("{name}: Result::map_err producer vanished"))?;
    if call_idx + 1 != graph.blocks[a].operations.len() {
        return Err(format!(
            "{name}: Result::map_err is not the block's last op"
        ));
    }
    let (receiver, env) = match &graph.blocks[a].operations[call_idx].kind {
        OpKind::Call {
            target: CallTarget::Method { name, .. },
            args,
            ..
        } if name == "map_err" && args.len() == 2 => (args[0].clone(), args[1].clone()),
        other => return Err(format!("{name}: recorded map_err site changed: {other:?}")),
    };
    // When the mapped Result is consumed by `?`, result_exc runs first and
    // changes this call block to LastException exits.  The map_err rewrite
    // then completes that same decision: receiver Ok forwards its payload to
    // the normal edge, receiver Err calls the mapper and raises the mapped
    // value.  A plain map_err consumer keeps the value-encoded Result arms.
    let exception_lowered = matches!(
        graph.blocks[a].exitswitch,
        Some(crate::model::ExitSwitch::LastException)
    );
    let saved_exit = if exception_lowered {
        if graph.blocks[a].exits.len() != 2 {
            return Err(format!(
                "{name}: Result::map_err LastException block does not have two exits"
            ));
        }
        graph.blocks[a]
            .exits
            .iter()
            .find(|exit| exit.exitcase.is_none())
            .cloned()
            .ok_or_else(|| format!("{name}: Result::map_err has no normal exception edge"))?
    } else {
        let [exit] = graph.blocks[a].exits.as_slice() else {
            return Err(format!("{name}: Result::map_err block has multiple exits"));
        };
        if graph.blocks[a].exitswitch.is_some()
            || exit.exitcase.is_some()
            || exit.last_exception.is_some()
            || exit.last_exc_value.is_some()
        {
            return Err(format!(
                "{name}: Result::map_err block exit is not a plain goto"
            ));
        }
        exit.clone()
    };
    let target = saved_exit.target;
    let mut carried = Vec::new();
    for arg in &saved_exit.args {
        if let LinkArg::Value(value) = arg
            && *value != site.result_var
            && !carried.contains(value)
        {
            carried.push(value.clone());
        }
    }

    let mut ok_sources = carried.clone();
    if !ok_sources.contains(&receiver) {
        ok_sources.push(receiver.clone());
    }
    let mut err_sources = ok_sources.clone();
    if !err_sources.contains(&env) {
        err_sources.push(env.clone());
    }
    let (ok_block, ok_inputs) = graph.create_block_with_arg_vars(ok_sources.len());
    let (err_block, err_inputs) = graph.create_block_with_arg_vars(err_sources.len());

    let receiver_ok = map_source(&ok_sources, &ok_inputs, &receiver)
        .ok_or_else(|| format!("{name}: Result receiver not threaded into Ok arm"))?;
    let ok_payload = read_payload(
        graph,
        ok_block,
        receiver_ok,
        &site.receiver_ok_owner,
        site.ok_ty.clone(),
    );
    let ok_payload = emit_narrow(graph, ok_block, ok_payload, &site.ok_class_root);
    let ok_result = if exception_lowered {
        ok_payload
    } else {
        emit_sum_variant(
            graph,
            ok_block,
            &site.result_owner,
            "Ok",
            0,
            Some((&site.result_ok_owner, ok_payload, site.ok_ty.clone())),
        )
    };
    let ok_args = reproduce_exit_args(
        &saved_exit,
        &site.result_var,
        &ok_result,
        &ok_sources,
        &ok_inputs,
        &name,
    )?;
    close_goto_mixed(graph, ok_block, target, ok_args);

    let receiver_err = map_source(&err_sources, &err_inputs, &receiver)
        .ok_or_else(|| format!("{name}: Result receiver not threaded into Err arm"))?;
    let err_payload = read_payload(
        graph,
        err_block,
        receiver_err,
        &site.receiver_err_owner,
        site.err_ty.clone(),
    );
    let err_payload = emit_narrow(graph, err_block, err_payload, &site.err_class_root);
    let env_err = map_source(&err_sources, &err_inputs, &env)
        .ok_or_else(|| format!("{name}: closure env not threaded into Err arm"))?;
    let mapped = emit_call_once(
        graph,
        err_block,
        env_err,
        Some((
            err_payload,
            site.err_ty.clone(),
            site.err_class_root.clone(),
        )),
        &site.call_once_owner,
        site.mapped_err_ty.clone(),
        &site.args_tuple_suffix,
    );
    let mapped = emit_narrow(graph, err_block, mapped, &site.mapped_err_class_root);
    if exception_lowered {
        graph.set_raise_values(err_block, mapped.clone(), mapped);
    } else {
        let err_result = emit_sum_variant(
            graph,
            err_block,
            &site.result_owner,
            "Err",
            1,
            Some((&site.result_err_owner, mapped, site.mapped_err_ty.clone())),
        );
        let err_args = reproduce_exit_args(
            &saved_exit,
            &site.result_var,
            &err_result,
            &err_sources,
            &err_inputs,
            &name,
        )?;
        close_goto_mixed(graph, err_block, target, err_args);
    }

    let a_id = graph.blocks[a].id;
    graph.blocks[a].operations.truncate(call_idx);
    let disc = graph.alloc_value_var();
    graph.block_mut(a_id).operations.push(SpaceOperation {
        result: Some(disc.clone()),
        kind: OpKind::FieldRead {
            base: receiver,
            field: FieldDescriptor {
                name: "__discriminant".to_string(),
                owner_root: Some(site.receiver_owner.clone()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
            },
            ty: ValueType::Int,
            pure: true,
        },
    });
    graph.set_branch(a_id, disc, err_block, err_sources, ok_block, ok_sources);
    Ok(())
}

fn read_payload(
    graph: &mut FunctionGraph,
    block: crate::model::BlockId,
    receiver: Variable,
    owner: &str,
    ty: ValueType,
) -> Variable {
    let payload = graph.alloc_value_var();
    graph.block_mut(block).operations.push(SpaceOperation {
        result: Some(payload.clone()),
        kind: OpKind::FieldRead {
            base: receiver,
            field: FieldDescriptor {
                name: "__pos_0".to_string(),
                owner_root: Some(owner.to_string()),
                owner_id: None,
                base_is_deref: None,
                taken_by_address: false,
            },
            ty,
            pure: true,
        },
    });
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_err_becomes_ok_passthrough_and_err_closure_arms() {
        let mut graph = FunctionGraph::new("map_err_fixture");
        let entry = graph.startblock;
        let receiver = graph.push_op_var(entry, OpKind::ConstInt(1), true).unwrap();
        let env = graph.push_op_var(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op_var(
                entry,
                OpKind::Call {
                    target: CallTarget::method("map_err", Some("Result".into())),
                    args: vec![receiver, env],
                    result_ty: ValueType::Ref(None),
                },
                true,
            )
            .unwrap();
        let (join, _inputs) = graph.create_block_with_arg_vars(1);
        graph.set_return(join, None);
        graph.set_goto(entry, join, vec![result.clone()]);

        let mut site = ResultMapErrSite {
            result_var: result,
            receiver_owner: "core::result::Result<i64,str>".into(),
            receiver_ok_owner: "core::result::Result<i64,str>::Ok".into(),
            receiver_err_owner: "core::result::Result<i64,str>::Err".into(),
            result_owner: "core::result::Result<i64,Error>".into(),
            result_ok_owner: "core::result::Result<i64,Error>::Ok".into(),
            result_err_owner: "core::result::Result<i64,Error>::Err".into(),
            call_once_owner: "fixture::closure".into(),
            ok_ty: ValueType::Int,
            ok_class_root: None,
            err_ty: ValueType::Str,
            err_class_root: None,
            mapped_err_ty: ValueType::Ref(Some("Error".into())),
            mapped_err_class_root: Some("Error".into()),
            args_tuple_suffix: "<str>".into(),
            closure_env_is_trivially_dropless: false,
        };
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site.clone()]), 0);
        assert!(graph.blocks[entry.0].operations.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::Call { target: CallTarget::Method { name, .. }, .. }
                    if name == "map_err"
            )
        }));

        site.closure_env_is_trivially_dropless = true;
        assert_eq!(rewire_result_map_err_sites(&mut graph, &[site]), 1);

        let calls: Vec<&CallTarget> = graph
            .blocks
            .iter()
            .flat_map(|block| &block.operations)
            .filter_map(|op| match &op.kind {
                OpKind::Call { target, .. } => Some(target),
                _ => None,
            })
            .collect();
        assert!(
            !calls.iter().any(
                |target| matches!(target, CallTarget::Method { name, .. } if name == "map_err")
            )
        );
        assert_eq!(
            calls
                .iter()
                .filter(|target| matches!(target, CallTarget::Method { name, .. } if name == "call_once"))
                .count(),
            1,
            "only the Err arm invokes the mapper"
        );
        for variant in ["Ok", "Err"] {
            assert!(calls.iter().any(|target| {
                matches!(target, CallTarget::SyntheticTransparentCtor { name, .. } if name == variant)
            }));
        }
    }
}
