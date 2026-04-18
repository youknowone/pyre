//! Build-time fnaddr registry for pyre's traced helper surface.
//!
//! `pyre-jit-trace/build.rs` runs the source-only codewriter. Unlike the
//! proc-macro path, it cannot call `#[jit_module]::__majit_helper_trace_fnaddrs()`
//! on the analyzed sources, so pyre publishes the same shape explicitly here.

fn push_fnaddr(entries: &mut Vec<(&'static str, i64)>, full_path: &'static str, fnptr: *const ()) {
    let fnaddr = fnptr as usize as i64;
    if fnaddr != 0 {
        entries.push((full_path, fnaddr));
    }
}

fn push_alias_pair(
    entries: &mut Vec<(&'static str, i64)>,
    module_path: &'static str,
    root_path: &'static str,
    fnptr: *const (),
) {
    push_fnaddr(entries, module_path, fnptr);
    push_fnaddr(entries, root_path, fnptr);
}

const CALLABLE_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_0",
        "pyre_interpreter::jit_call_callable_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_1",
        "pyre_interpreter::jit_call_callable_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_2",
        "pyre_interpreter::jit_call_callable_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_3",
        "pyre_interpreter::jit_call_callable_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_4",
        "pyre_interpreter::jit_call_callable_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_5",
        "pyre_interpreter::jit_call_callable_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_6",
        "pyre_interpreter::jit_call_callable_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_7",
        "pyre_interpreter::jit_call_callable_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_callable_8",
        "pyre_interpreter::jit_call_callable_8",
    ),
];

const KNOWN_BUILTIN_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_0",
        "pyre_interpreter::jit_call_known_builtin_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_1",
        "pyre_interpreter::jit_call_known_builtin_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_2",
        "pyre_interpreter::jit_call_known_builtin_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_3",
        "pyre_interpreter::jit_call_known_builtin_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_4",
        "pyre_interpreter::jit_call_known_builtin_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_5",
        "pyre_interpreter::jit_call_known_builtin_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_6",
        "pyre_interpreter::jit_call_known_builtin_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_7",
        "pyre_interpreter::jit_call_known_builtin_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_builtin_8",
        "pyre_interpreter::jit_call_known_builtin_8",
    ),
];

const KNOWN_FUNCTION_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_0",
        "pyre_interpreter::jit_call_known_function_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_1",
        "pyre_interpreter::jit_call_known_function_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_2",
        "pyre_interpreter::jit_call_known_function_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_3",
        "pyre_interpreter::jit_call_known_function_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_4",
        "pyre_interpreter::jit_call_known_function_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_5",
        "pyre_interpreter::jit_call_known_function_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_6",
        "pyre_interpreter::jit_call_known_function_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_7",
        "pyre_interpreter::jit_call_known_function_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_call_known_function_8",
        "pyre_interpreter::jit_call_known_function_8",
    ),
];

const LIST_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_list_0",
        "pyre_interpreter::jit_build_list_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_1",
        "pyre_interpreter::jit_build_list_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_2",
        "pyre_interpreter::jit_build_list_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_3",
        "pyre_interpreter::jit_build_list_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_4",
        "pyre_interpreter::jit_build_list_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_5",
        "pyre_interpreter::jit_build_list_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_6",
        "pyre_interpreter::jit_build_list_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_7",
        "pyre_interpreter::jit_build_list_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_list_8",
        "pyre_interpreter::jit_build_list_8",
    ),
];

const TUPLE_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_0",
        "pyre_interpreter::jit_build_tuple_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_1",
        "pyre_interpreter::jit_build_tuple_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_2",
        "pyre_interpreter::jit_build_tuple_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_3",
        "pyre_interpreter::jit_build_tuple_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_4",
        "pyre_interpreter::jit_build_tuple_4",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_5",
        "pyre_interpreter::jit_build_tuple_5",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_6",
        "pyre_interpreter::jit_build_tuple_6",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_7",
        "pyre_interpreter::jit_build_tuple_7",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_tuple_8",
        "pyre_interpreter::jit_build_tuple_8",
    ),
];

const MAP_BUILD_HELPER_PATHS: &[(&str, &str)] = &[
    (
        "pyre_interpreter::runtime_ops::jit_build_map_0",
        "pyre_interpreter::jit_build_map_0",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_1",
        "pyre_interpreter::jit_build_map_1",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_2",
        "pyre_interpreter::jit_build_map_2",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_3",
        "pyre_interpreter::jit_build_map_3",
    ),
    (
        "pyre_interpreter::runtime_ops::jit_build_map_4",
        "pyre_interpreter::jit_build_map_4",
    ),
];

/// Build-time equivalent of `#[jit_module]::__majit_helper_trace_fnaddrs()`.
///
/// The registry includes both the module-qualified path produced by the
/// source analyzer (`runtime_ops::foo`) and the crate-root re-export path
/// (`foo`) that pyre's runtime helper code often calls directly.
pub fn jit_trace_fnaddrs() -> Vec<(&'static str, i64)> {
    let mut entries = Vec::new();

    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_make_function_from_globals",
        "pyre_interpreter::jit_make_function_from_globals",
        crate::runtime_ops::jit_make_function_from_globals as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_load_name_from_namespace",
        "pyre_interpreter::jit_load_name_from_namespace",
        crate::runtime_ops::jit_load_name_from_namespace as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_store_name_to_namespace",
        "pyre_interpreter::jit_store_name_to_namespace",
        crate::runtime_ops::jit_store_name_to_namespace as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_sequence_getitem",
        "pyre_interpreter::jit_sequence_getitem",
        crate::runtime_ops::jit_sequence_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::runtime_ops::jit_range_iter_next_or_null",
        "pyre_interpreter::jit_range_iter_next_or_null",
        crate::runtime_ops::jit_range_iter_next_or_null as *const (),
    );

    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_truth_value",
        "pyre_interpreter::jit_truth_value",
        crate::opcode_ops::jit_truth_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_bool_value_from_truth",
        "pyre_interpreter::jit_bool_value_from_truth",
        crate::opcode_ops::jit_bool_value_from_truth as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_binary_value_from_tag",
        "pyre_interpreter::jit_binary_value_from_tag",
        crate::opcode_ops::jit_binary_value_from_tag as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_compare_value_from_tag",
        "pyre_interpreter::jit_compare_value_from_tag",
        crate::opcode_ops::jit_compare_value_from_tag as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_negative_value",
        "pyre_interpreter::jit_unary_negative_value",
        crate::opcode_ops::jit_unary_negative_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_unary_invert_value",
        "pyre_interpreter::jit_unary_invert_value",
        crate::opcode_ops::jit_unary_invert_value as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_getitem",
        "pyre_interpreter::jit_getitem",
        crate::opcode_ops::jit_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_setitem",
        "pyre_interpreter::jit_setitem",
        crate::opcode_ops::jit_setitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_getattr",
        "pyre_interpreter::jit_getattr",
        crate::opcode_ops::jit_getattr as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_interpreter::opcode_ops::jit_setattr",
        "pyre_interpreter::jit_setattr",
        crate::opcode_ops::jit_setattr as *const (),
    );

    for (nargs, (module_path, root_path)) in CALLABLE_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::callable_call_helper(nargs) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (nargs, (module_path, root_path)) in KNOWN_BUILTIN_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::known_builtin_call_helper(nargs) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (nargs, (module_path, root_path)) in KNOWN_FUNCTION_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::known_function_call_helper(nargs) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in LIST_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::list_build_helper(count) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in TUPLE_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::tuple_build_helper(count) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }
    for (count, (module_path, root_path)) in MAP_BUILD_HELPER_PATHS.iter().enumerate() {
        if let Some(fnptr) = crate::runtime_ops::map_build_helper(count) {
            push_alias_pair(&mut entries, module_path, root_path, fnptr);
        }
    }

    push_alias_pair(
        &mut entries,
        "pyre_object::intobject::jit_w_int_new",
        "pyre_object::jit_w_int_new",
        pyre_object::jit_w_int_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::floatobject::jit_w_float_new",
        "pyre_object::jit_w_float_new",
        pyre_object::jit_w_float_new as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_append",
        "pyre_object::jit_list_append",
        pyre_object::jit_list_append as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_getitem",
        "pyre_object::jit_list_getitem",
        pyre_object::jit_list_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::listobject::jit_list_setitem",
        "pyre_object::jit_list_setitem",
        pyre_object::jit_list_setitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::tupleobject::jit_tuple_getitem",
        "pyre_object::jit_tuple_getitem",
        pyre_object::jit_tuple_getitem as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_concat",
        "pyre_object::jit_str_concat",
        pyre_object::jit_str_concat as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_repeat",
        "pyre_object::jit_str_repeat",
        pyre_object::jit_str_repeat as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_compare",
        "pyre_object::jit_str_compare",
        pyre_object::jit_str_compare as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::strobject::jit_str_is_true",
        "pyre_object::jit_str_is_true",
        pyre_object::jit_str_is_true as *const (),
    );
    push_alias_pair(
        &mut entries,
        "pyre_object::rangeobject::jit_range_iter_new",
        "pyre_object::jit_range_iter_new",
        pyre_object::jit_range_iter_new as *const (),
    );

    entries
}

#[cfg(test)]
mod tests {
    use super::jit_trace_fnaddrs;
    use std::collections::HashMap;

    #[test]
    fn jit_trace_fnaddrs_contains_root_and_module_aliases() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let make_fn =
            crate::runtime_ops::jit_make_function_from_globals as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_make_function_from_globals"],
            make_fn
        );
        assert_eq!(
            bindings["pyre_interpreter::jit_make_function_from_globals"],
            make_fn
        );

        let list_append = pyre_object::jit_list_append as *const () as usize as i64;
        assert_eq!(
            bindings["pyre_object::listobject::jit_list_append"],
            list_append
        );
        assert_eq!(bindings["pyre_object::jit_list_append"], list_append);
    }

    #[test]
    fn jit_trace_fnaddrs_covers_generated_runtime_helper_families() {
        let bindings: HashMap<&'static str, i64> = jit_trace_fnaddrs().into_iter().collect();

        let callable3 =
            crate::runtime_ops::callable_call_helper(3).expect("callable helper") as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_call_callable_3"],
            callable3
        );
        assert_eq!(bindings["pyre_interpreter::jit_call_callable_3"], callable3);

        let tuple2 =
            crate::runtime_ops::tuple_build_helper(2).expect("tuple build helper") as usize as i64;
        assert_eq!(
            bindings["pyre_interpreter::runtime_ops::jit_build_tuple_2"],
            tuple2
        );
        assert_eq!(bindings["pyre_interpreter::jit_build_tuple_2"], tuple2);
    }
}
