//! Canonical built-in call matching for the graph/pipeline translator.
//!
//! This centralizes the active helper vocabulary so annotate/jtransform/patterns
//! do not each carry their own ad-hoc symbol tables. It is still lighter than
//! RPython's full descriptor/effectinfo model, but it moves majit toward a
//! single source of truth for call semantics.

use crate::graph::CallTarget;
use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum NamespaceAccessKind {
    LoadLocal,
    LoadGlobal,
    StoreLocal,
    StoreGlobal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum BuildCollectionKind {
    List,
    Tuple,
}

// CallDescriptor is now defined in call.rs — re-export for backward compat.
pub use crate::call::CallDescriptor;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallTargetPattern {
    Method {
        name: &'static str,
        receiver_root: Option<&'static str>,
    },
    FunctionPath(&'static [&'static str]),
}

impl CallTargetPattern {
    fn matches(self, target: &CallTarget) -> bool {
        match (self, target) {
            (
                CallTargetPattern::Method {
                    name,
                    receiver_root,
                },
                CallTarget::Method {
                    name: target_name,
                    receiver_root: target_root,
                },
            ) => {
                if target_name != name {
                    return false;
                }
                // If the pattern specifies a receiver type, check it matches.
                // A target receiver that starts with lowercase (e.g. "handler",
                // "self", "executor") is a variable/parameter name, not a type —
                // treat it as matching any pattern receiver.
                receiver_root.is_none_or(|root| {
                    target_root.as_deref() == Some(root)
                        || target_root.as_ref().is_some_and(|r| is_generic_receiver(r))
                })
            }
            (CallTargetPattern::FunctionPath(path), CallTarget::FunctionPath { segments }) => {
                segments.iter().map(String::as_str).eq(path.iter().copied())
            }
            _ => false,
        }
    }
}

/// Detect generic type parameter or variable name used as receiver.
/// Delegates to the canonical implementation in `call.rs`.
pub(crate) fn is_generic_receiver(receiver: &str) -> bool {
    crate::call::is_generic_receiver(receiver)
}

// Keep the old implementation as dead code reference to avoid breaking
// any in-flight references during the transition.
#[allow(dead_code)]
fn _is_generic_receiver_old(receiver: &str) -> bool {
    let mut chars = receiver.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if first.is_lowercase() {
        return true; // variable name: "handler", "self", "executor"
    }
    // Single uppercase letter: "H", "T", "E" — type parameter
    first.is_uppercase() && chars.next().is_none()
}

struct CallDescriptorEntry {
    targets: &'static [CallTargetPattern],
    extra_effect: ExtraEffect,
    oopspec_index: OopSpecIndex,
}

impl CallDescriptorEntry {
    fn effect_info(&self) -> EffectInfo {
        match self.extra_effect {
            ExtraEffect::ElidableCannotRaise => EffectInfo::elidable(),
            extra_effect => EffectInfo::new(extra_effect, self.oopspec_index),
        }
    }
}

const INT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_add"]),
    CallTargetPattern::FunctionPath(&["w_int_sub"]),
    CallTargetPattern::FunctionPath(&["w_int_mul"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_add"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_sub"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_mul"]),
];

const FLOAT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_add"]),
    CallTargetPattern::FunctionPath(&["w_float_sub"]),
];

const LOCAL_READ_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "load_local_value",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "load_local_checked_value",
        receiver_root: None,
    },
];

const LOCAL_WRITE_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::Method {
    name: "store_local_value",
    receiver_root: None,
}];

const FUNCTION_INVOKE_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "call_callable",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "call_function_ex",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "call_kw",
        receiver_root: None,
    },
    // FunctionPath variants — when the impl body calls these as free functions
    CallTargetPattern::FunctionPath(&["call_callable"]),
    CallTargetPattern::FunctionPath(&["call_function_ex"]),
    CallTargetPattern::FunctionPath(&["call_kw"]),
    CallTargetPattern::FunctionPath(&["opcode_call"]),
    // opcode_make_function is NOT here — MAKE_FUNCTION creates a function
    // object, it does not call a function.
];

const TRUTH_CHECK_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "truth_value",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "bool_value_from_truth",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "concrete_truth_as_bool",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "to_bool",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["truth_value"]),
    CallTargetPattern::FunctionPath(&["bool_value_from_truth"]),
    CallTargetPattern::FunctionPath(&["concrete_truth_as_bool"]),
];

// push_value/pop_value are NOT here — they appear in nearly every opcode
// and would classify everything as StackManip. They are bookkeeping,
// not the semantic operation.
const STACK_MANIP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "swap_values",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "copy_value",
        receiver_root: None,
    },
];

const PEEK_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::Method {
    name: "peek_at",
    receiver_root: None,
}];

const LOAD_GLOBAL_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "load_global",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["load_global"]),
];

const STORE_GLOBAL_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "store_global",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["store_global"]),
];

const LOAD_LOCAL_NAMESPACE_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "load_name",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "load_name_value",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "load_from_dict_or_globals",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "load_from_dict_or_deref",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["load_name"]),
    CallTargetPattern::FunctionPath(&["load_name_value"]),
];

const STORE_LOCAL_NAMESPACE_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "store_name",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "store_name_value",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["store_name"]),
    CallTargetPattern::FunctionPath(&["store_name_value"]),
];

// FOR_ITER semantics: call next() and branch on StopIteration.
// GET_ITER (ensure_iter_value, opcode_get_iter) is NOT here — it's
// space.iter() (iterable → iterator conversion), not next/branch.
const RANGE_ITER_NEXT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "iter_next_value",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "for_iter",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["iter_next_value"]),
];

const ITER_CLEANUP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "end_for",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "pop_iter",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "on_iter_exhausted",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["end_for"]),
    CallTargetPattern::FunctionPath(&["on_iter_exhausted"]),
];

const JUMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "set_next_instr",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "fallthrough_target",
        receiver_root: None,
    },
    // Qualified trait calls: ControlFlowOpcodeHandler::set_next_instr(executor, ...)
    CallTargetPattern::FunctionPath(&["ControlFlowOpcodeHandler", "set_next_instr"]),
    CallTargetPattern::FunctionPath(&["set_next_instr"]),
];

const RETURN_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "return_value",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["return_value"]),
    CallTargetPattern::FunctionPath(&["opcode_return_value"]),
];

const BUILD_LIST_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "build_list",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["build_list"]),
    CallTargetPattern::FunctionPath(&["opcode_build_list"]),
];

const BUILD_TUPLE_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "build_tuple",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["build_tuple"]),
    CallTargetPattern::FunctionPath(&["opcode_build_tuple"]),
];

const UNPACK_SEQUENCE_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "unpack_sequence",
        receiver_root: None,
    },
    CallTargetPattern::Method {
        name: "unpack_ex",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["unpack_sequence"]),
    CallTargetPattern::FunctionPath(&["unpack_ex"]),
    CallTargetPattern::FunctionPath(&["opcode_unpack_sequence"]),
];

const SEQUENCE_SETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "store_subscr",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["store_subscr"]),
    CallTargetPattern::FunctionPath(&["opcode_store_subscr"]),
];

const COLLECTION_APPEND_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::Method {
        name: "list_append",
        receiver_root: None,
    },
    CallTargetPattern::FunctionPath(&["list_append"]),
    CallTargetPattern::FunctionPath(&["opcode_list_append"]),
];

const CALL_DESCRIPTOR_TABLE: &[CallDescriptorEntry] = &[
    CallDescriptorEntry {
        targets: INT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
];

fn matches_any(target: &CallTarget, patterns: &[CallTargetPattern]) -> bool {
    patterns
        .iter()
        .copied()
        .any(|pattern| pattern.matches(target))
}

pub(crate) fn is_int_arithmetic_target(target: &CallTarget) -> bool {
    matches_any(target, INT_ARITH_TARGETS)
}

pub(crate) fn is_float_arithmetic_target(target: &CallTarget) -> bool {
    matches_any(target, FLOAT_ARITH_TARGETS)
}

pub(crate) fn is_local_read_target(target: &CallTarget) -> bool {
    matches_any(target, LOCAL_READ_TARGETS)
}

pub(crate) fn is_local_write_target(target: &CallTarget) -> bool {
    matches_any(target, LOCAL_WRITE_TARGETS)
}

pub(crate) fn is_function_invoke_target(target: &CallTarget) -> bool {
    matches_any(target, FUNCTION_INVOKE_TARGETS)
}

pub(crate) fn is_truth_check_target(target: &CallTarget) -> bool {
    matches_any(target, TRUTH_CHECK_TARGETS)
}

pub(crate) fn is_stack_manip_target(target: &CallTarget) -> bool {
    matches_any(target, STACK_MANIP_TARGETS) || matches_any(target, PEEK_TARGETS)
}

pub(crate) fn namespace_access_kind(target: &CallTarget) -> Option<NamespaceAccessKind> {
    if matches_any(target, LOAD_GLOBAL_TARGETS) {
        Some(NamespaceAccessKind::LoadGlobal)
    } else if matches_any(target, STORE_GLOBAL_TARGETS) {
        Some(NamespaceAccessKind::StoreGlobal)
    } else if matches_any(target, LOAD_LOCAL_NAMESPACE_TARGETS) {
        Some(NamespaceAccessKind::LoadLocal)
    } else if matches_any(target, STORE_LOCAL_NAMESPACE_TARGETS) {
        Some(NamespaceAccessKind::StoreLocal)
    } else {
        None
    }
}

pub(crate) fn is_range_iter_next_target(target: &CallTarget) -> bool {
    matches_any(target, RANGE_ITER_NEXT_TARGETS)
}

pub(crate) fn is_iter_cleanup_target(target: &CallTarget) -> bool {
    matches_any(target, ITER_CLEANUP_TARGETS)
}

pub(crate) fn is_jump_target(target: &CallTarget) -> bool {
    matches_any(target, JUMP_TARGETS)
}

pub(crate) fn is_return_target(target: &CallTarget) -> bool {
    matches_any(target, RETURN_TARGETS)
}

pub(crate) fn build_collection_kind(target: &CallTarget) -> Option<BuildCollectionKind> {
    if matches_any(target, BUILD_LIST_TARGETS) {
        Some(BuildCollectionKind::List)
    } else if matches_any(target, BUILD_TUPLE_TARGETS) {
        Some(BuildCollectionKind::Tuple)
    } else {
        None
    }
}

pub(crate) fn is_unpack_sequence_target(target: &CallTarget) -> bool {
    matches_any(target, UNPACK_SEQUENCE_TARGETS)
}

pub(crate) fn is_sequence_setitem_target(target: &CallTarget) -> bool {
    matches_any(target, SEQUENCE_SETITEM_TARGETS)
}

pub(crate) fn is_collection_append_target(target: &CallTarget) -> bool {
    matches_any(target, COLLECTION_APPEND_TARGETS)
}

fn exact_descriptor(target: &CallTarget) -> Option<CallDescriptor> {
    CALL_DESCRIPTOR_TABLE
        .iter()
        .find(|entry| matches_any(target, entry.targets))
        .map(|entry| CallDescriptor::known(target.clone(), entry.effect_info()))
}

pub(crate) fn describe_call(target: &CallTarget) -> Option<CallDescriptor> {
    exact_descriptor(target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_canonical_int_helper_path() {
        let descriptor = describe_call(&CallTarget::function_path(["crate", "math", "w_int_add"]))
            .expect("expected descriptor");
        assert_eq!(
            descriptor.target,
            CallTarget::function_path(["crate", "math", "w_int_add"])
        );
        assert!(is_int_arithmetic_target(&descriptor.target));
        assert!(descriptor.effect_info().is_elidable());
    }

    #[test]
    fn bare_fallback_does_not_describe_pyframe_namespace_store() {
        assert_eq!(
            describe_call(&CallTarget::method(
                "store_name_value",
                Some("PyFrame".into())
            )),
            None
        );
    }

    #[test]
    fn does_not_collapse_multi_segment_function_path_to_leaf() {
        let descriptor = describe_call(&CallTarget::function_path(["foo", "w_int_add"]));
        assert_eq!(descriptor, None);
    }

    #[test]
    fn pyframe_only_method_helpers_require_pyframe_owner() {
        assert_eq!(
            describe_call(&CallTarget::method("load_name", Some("TinyInterp".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("for_iter", Some("TinyInterp".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("unpack_ex", Some("TinyInterp".into()))),
            None
        );
    }

    #[test]
    fn does_not_accept_unused_generic_helper_aliases() {
        assert_eq!(describe_call(&CallTarget::method("finish", None)), None);
        assert_eq!(describe_call(&CallTarget::method("new_list", None)), None);
        assert_eq!(describe_call(&CallTarget::method("new_tuple", None)), None);
        assert_eq!(describe_call(&CallTarget::method("setitem", None)), None);
        assert_eq!(describe_call(&CallTarget::method("append", None)), None);
        assert_eq!(
            describe_call(&CallTarget::function_path(["call_function"])),
            None
        );
        assert_eq!(describe_call(&CallTarget::function_path(["invoke"])), None);
    }

    #[test]
    fn pyframe_method_helpers_do_not_match_as_free_functions() {
        assert_eq!(
            describe_call(&CallTarget::function_path(["call_callable"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["truth_value"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["bool_value_from_truth"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["concrete_truth_as_bool"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["load_name"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["store_name"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["for_iter"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["build_list"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["list_append"])),
            None
        );
        assert_eq!(describe_call(&CallTarget::function_path(["push"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["pop"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["call"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["peek"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["is_true"])), None);
    }

    #[test]
    fn broad_stack_and_truth_method_aliases_are_not_canonical() {
        assert_eq!(
            describe_call(&CallTarget::method("push", Some("TinyInterp".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("pop", Some("TinyInterp".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("call", Some("TinyInterp".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("peek", Some("TinyInterp".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("is_true", Some("TinyInterp".into()))),
            None
        );
    }

    #[test]
    fn broad_collection_and_io_aliases_are_not_canonical() {
        assert_eq!(
            describe_call(&CallTarget::method("is_empty", Some("Vec".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("get", Some("Vec".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("print", Some("Io".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("read", Some("Io".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("read_line", Some("Io".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::method("write", Some("Io".into()))),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["vec_push"])),
            None
        );
        assert_eq!(
            describe_call(&CallTarget::function_path(["crate", "stack", "vec_push"])),
            None
        );
        assert_eq!(describe_call(&CallTarget::function_path(["get"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["print"])), None);
        assert_eq!(
            describe_call(&CallTarget::function_path(["read_exact"])),
            None
        );
    }

    #[test]
    fn operator_and_method_form_arithmetic_aliases_are_not_canonical() {
        assert_eq!(describe_call(&CallTarget::method("+", None)), None);
        assert_eq!(describe_call(&CallTarget::method("-", None)), None);
        assert_eq!(describe_call(&CallTarget::method("*", None)), None);
        assert_eq!(describe_call(&CallTarget::method("len", None)), None);
        assert_eq!(describe_call(&CallTarget::method("size", None)), None);
        assert_eq!(describe_call(&CallTarget::method("w_int_add", None)), None);
        assert_eq!(
            describe_call(&CallTarget::method("w_float_add", None)),
            None
        );
        assert_eq!(describe_call(&CallTarget::function_path(["+"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["len"])), None);
        assert_eq!(describe_call(&CallTarget::function_path(["size"])), None);
    }
}
