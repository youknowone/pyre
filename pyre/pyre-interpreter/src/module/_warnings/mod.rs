//! `_warnings` — structural port of `pypy/module/_warnings/interp_warnings.py`.

use crate::PyError;
use pyre_object::*;

const VERSION_ATTR: &str = "_filters_version_state";

fn pin_root_slot(value: PyObjectRef) -> usize {
    pyre_object::gc_roots::pin_root(value);
    pyre_object::gc_roots::shadow_stack_len() - 1
}

fn module_attr(module: &str, name: &str) -> Option<PyObjectRef> {
    let w_module = crate::importing::get_sys_module(module)?;
    crate::baseobjspace::getattr_str(w_module, name).ok()
}

fn warnings_attr(name: &str) -> Option<PyObjectRef> {
    module_attr("warnings", name)
}

fn import_module(name: &str) -> Result<PyObjectRef, PyError> {
    if let Some(module) = crate::importing::get_sys_module(name) {
        return Ok(module);
    }
    crate::importing::importhook(
        name,
        w_none(),
        w_list_new(vec![w_str_new("*")]),
        0,
        crate::call::getexecutioncontext(),
    )?;
    crate::importing::get_sys_module(name).ok_or_else(|| {
        PyError::new(
            crate::PyErrorKind::ImportError,
            format!("No module named '{name}'"),
        )
    })
}

fn native_attr(name: &str) -> Result<PyObjectRef, PyError> {
    module_attr("_warnings", name)
        .ok_or_else(|| PyError::runtime_error(format!("_warnings.{name} is not initialized")))
}

fn native_store(name: &str, value: PyObjectRef) -> Result<(), PyError> {
    let module = crate::importing::get_sys_module("_warnings")
        .ok_or_else(|| PyError::runtime_error("_warnings is not initialized"))?;
    crate::baseobjspace::setattr_str(module, name, value).map(|_| ())
}

fn get_default_action() -> Result<PyObjectRef, PyError> {
    if let Some(action) = warnings_attr("defaultaction") {
        native_store("_defaultaction", action)?;
        Ok(action)
    } else {
        native_attr("_defaultaction")
    }
}

fn get_once_registry() -> Result<PyObjectRef, PyError> {
    if let Some(registry) = warnings_attr("onceregistry") {
        native_store("_onceregistry", registry)?;
        Ok(registry)
    } else {
        native_attr("_onceregistry")
    }
}

fn create_filter(category: PyObjectRef, action: &str, module: Option<&str>) -> PyObjectRef {
    w_tuple_new(vec![
        w_str_new(action),
        w_none(),
        category,
        module.map(w_str_new).unwrap_or_else(w_none),
        w_int_new(0),
    ])
}

fn warning_class(name: &str) -> PyObjectRef {
    crate::builtins::lookup_exc_class(name)
        .unwrap_or_else(|| panic!("{name} must be installed before _warnings"))
}

fn current_version() -> Result<PyObjectRef, PyError> {
    native_attr(VERSION_ATTR)
}

fn new_version() -> PyObjectRef {
    // PyPy `State.filters_mutated`: `space.call_function(space.w_object)`.
    // The registry compares this opaque sentinel by identity, never value.
    w_instance_new(crate::typedef::w_object())
}

fn filters_mutated_impl() -> Result<(), PyError> {
    native_store(VERSION_ATTR, new_version())
}

fn get_category(message: PyObjectRef, category: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let warning = warning_class("Warning");
    if crate::baseobjspace::isinstance(message, warning)? {
        return crate::typedef::r#type(message)
            .ok_or_else(|| PyError::type_error("warning instance has no type"));
    }
    let category = if category.is_null() || unsafe { is_none(category) } {
        warning_class("UserWarning")
    } else {
        category
    };
    match crate::baseobjspace::issubclass(category, warning) {
        Ok(true) => Ok(category),
        Ok(false) => Err(PyError::type_error("category is not a subclass of Warning")),
        Err(_) => Err(PyError::type_error(format!(
            "category must be a Warning subclass, not '{}'",
            crate::baseobjspace::object_functionstr_type_name(category)
        ))),
    }
}

fn is_internal_frame(frame: *mut crate::PyFrame) -> bool {
    if frame.is_null() {
        return false;
    }
    let filename = unsafe { &*crate::pyframe::pyframe_get_pycode(&*frame) }
        .source_path
        .as_str();
    filename.contains("importlib") && filename.contains("_bootstrap")
}

fn get_frame(mut stacklevel: i64) -> *mut crate::PyFrame {
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return std::ptr::null_mut();
    }
    let mut frame = unsafe { (*ec).gettopframe_nohidden() };
    if stacklevel <= 0 || is_internal_frame(frame) {
        while stacklevel > 1 && !frame.is_null() {
            frame = crate::executioncontext::ExecutionContext::getnextframe_nohidden(frame);
            stacklevel -= 1;
        }
    } else {
        while stacklevel > 1 && !frame.is_null() {
            loop {
                frame = crate::executioncontext::ExecutionContext::getnextframe_nohidden(frame);
                if frame.is_null() || !is_internal_frame(frame) {
                    break;
                }
            }
            stacklevel -= 1;
        }
    }
    frame
}

fn setup_context(stacklevel: i64) -> (PyObjectRef, i64, PyObjectRef, PyObjectRef) {
    let _roots = pyre_object::gc_roots::push_roots();
    let frame = get_frame(stacklevel);
    let (filename, lineno, globals) = if frame.is_null() {
        let globals = crate::importing::get_sys_module("sys")
            .map(|module| unsafe { w_module_get_w_dict(module) })
            .unwrap_or_else(w_dict_new);
        (w_str_new("sys"), 1, globals)
    } else {
        let frame = unsafe { &*frame };
        let code = unsafe { &*crate::pyframe::pyframe_get_pycode(frame) };
        (
            w_str_new(&code.source_path),
            frame.get_last_lineno() as i64,
            frame.get_w_globals(),
        )
    };
    let filename_slot = pin_root_slot(filename);
    let globals_slot = pin_root_slot(globals);
    let registry = unsafe {
        w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
            "__warningregistry__",
        )
    }
    .unwrap_or_else(|| {
        let registry = w_dict_new();
        unsafe {
            w_dict_setitem_str(
                pyre_object::gc_roots::shadow_stack_get(globals_slot),
                "__warningregistry__",
                registry,
            )
        };
        registry
    });
    let registry_slot = pin_root_slot(registry);
    let module = unsafe {
        w_dict_getitem_str(
            pyre_object::gc_roots::shadow_stack_get(globals_slot),
            "__name__",
        )
    }
    .unwrap_or_else(|| w_str_new("<string>"));
    let module_slot = pin_root_slot(module);
    (
        pyre_object::gc_roots::shadow_stack_get(filename_slot),
        lineno,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
    )
}

fn check_matched(filter: PyObjectRef, value: PyObjectRef) -> Result<bool, PyError> {
    if filter.is_null() || unsafe { is_none(filter) } {
        return Ok(true);
    }
    if unsafe { pyre_object::pyobject::is_exact_type(filter, &pyre_object::STR_TYPE) } {
        if unsafe { !is_str(value) } {
            return Ok(false);
        }
        return Ok(unsafe { w_str_get_wtf8(filter) == w_str_get_wtf8(value) });
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let filter_slot = pin_root_slot(filter);
    let value_slot = pin_root_slot(value);
    let matcher = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(filter_slot),
        "match",
    )?;
    let matcher_slot = pin_root_slot(matcher);
    let result = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(matcher_slot),
        &[pyre_object::gc_roots::shadow_stack_get(value_slot)],
    )?;
    let result_slot = pin_root_slot(result);
    crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(result_slot))
}

fn get_filter(
    category: PyObjectRef,
    text: PyObjectRef,
    lineno: i64,
    module: PyObjectRef,
) -> Result<(String, PyObjectRef), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pin_root_slot(category);
    let text_slot = pin_root_slot(text);
    let module_slot = pin_root_slot(module);
    let filters = if let Some(filters) = warnings_attr("filters") {
        native_store("filters", filters)?;
        filters
    } else {
        native_attr("filters")?
    };
    let filters_slot = pin_root_slot(filters);
    // PyPy `space.fixedview(w_filters)` snapshots any iterable before the
    // loop because user code may mutate `warnings.filters` while matching.
    let items =
        crate::baseobjspace::fixedview(pyre_object::gc_roots::shadow_stack_get(filters_slot), -1)?;
    let item_slots: Vec<_> = items.into_iter().map(pin_root_slot).collect();
    for item_slot in item_slots {
        let item = pyre_object::gc_roots::shadow_stack_get(item_slot);
        let fields = crate::baseobjspace::fixedview(item, 5)?;
        let field_slots: Vec<_> = fields.into_iter().map(pin_root_slot).collect();
        let filter_lineno =
            crate::baseobjspace::int_w(pyre_object::gc_roots::shadow_stack_get(field_slots[4]))?;
        if check_matched(
            pyre_object::gc_roots::shadow_stack_get(field_slots[1]),
            pyre_object::gc_roots::shadow_stack_get(text_slot),
        )? && check_matched(
            pyre_object::gc_roots::shadow_stack_get(field_slots[3]),
            pyre_object::gc_roots::shadow_stack_get(module_slot),
        )? && crate::baseobjspace::issubclass(
            pyre_object::gc_roots::shadow_stack_get(category_slot),
            pyre_object::gc_roots::shadow_stack_get(field_slots[2]),
        )? && (filter_lineno == 0 || filter_lineno == lineno)
        {
            return Ok((
                crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(
                    field_slots[0],
                ))?
                .to_string(),
                pyre_object::gc_roots::shadow_stack_get(item_slot),
            ));
        }
    }
    let action = get_default_action()?;
    Ok((crate::baseobjspace::text_w(action)?.to_string(), PY_NULL))
}

fn already_warned(
    registry: PyObjectRef,
    key: PyObjectRef,
    should_set: bool,
) -> Result<bool, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let registry_slot = pin_root_slot(registry);
    let key_slot = pin_root_slot(key);
    let version = current_version()?;
    let version_slot = pin_root_slot(version);
    let registry_version = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
        "version",
    )?;
    if registry_version != Some(pyre_object::gc_roots::shadow_stack_get(version_slot)) {
        let clear = crate::baseobjspace::getattr_str(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            "clear",
        )?;
        let clear_slot = pin_root_slot(clear);
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(clear_slot),
            &[],
        )?;
        let version_key_slot = pin_root_slot(w_str_new("version"));
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            pyre_object::gc_roots::shadow_stack_get(version_key_slot),
            pyre_object::gc_roots::shadow_stack_get(version_slot),
        )?;
    } else if let Some(value) = crate::baseobjspace::finditem(
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
    )? && crate::baseobjspace::is_true(value)?
    {
        return Ok(true);
    }
    if should_set {
        crate::baseobjspace::setitem(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            w_bool_from(true),
        )?;
    }
    Ok(false)
}

fn update_registry(
    registry: PyObjectRef,
    text: PyObjectRef,
    category: PyObjectRef,
) -> Result<bool, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let registry_slot = pin_root_slot(registry);
    let text_slot = pin_root_slot(text);
    let category_slot = pin_root_slot(category);
    let key = w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(text_slot),
        pyre_object::gc_roots::shadow_stack_get(category_slot),
    ]);
    let key_slot = pin_root_slot(key);
    already_warned(
        pyre_object::gc_roots::shadow_stack_get(registry_slot),
        pyre_object::gc_roots::shadow_stack_get(key_slot),
        true,
    )
}

fn show_warning(
    message: PyObjectRef,
    text: PyObjectRef,
    category: PyObjectRef,
    filename: PyObjectRef,
    lineno: i64,
    source_line: PyObjectRef,
    source: PyObjectRef,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let message_slot = pin_root_slot(message);
    let text_slot = pin_root_slot(text);
    let category_slot = pin_root_slot(category);
    let filename_slot = pin_root_slot(filename);
    let source_line_slot = pin_root_slot(source_line);
    let source_slot = pin_root_slot(source);
    if let Some(show) = warnings_attr("_showwarnmsg") {
        if !crate::baseobjspace::callable_w(show) {
            return Err(PyError::type_error(
                "warnings._showwarnmsg() must be set to a callable",
            ));
        }
        let cls = warnings_attr("WarningMessage")
            .ok_or_else(|| PyError::runtime_error("unable to get warnings.WarningMessage"))?;
        let warning_message = crate::call::call_function_impl_result(
            cls,
            &[
                pyre_object::gc_roots::shadow_stack_get(message_slot),
                pyre_object::gc_roots::shadow_stack_get(category_slot),
                pyre_object::gc_roots::shadow_stack_get(filename_slot),
                w_int_new(lineno),
                w_none(),
                w_none(),
                if source.is_null() {
                    w_none()
                } else {
                    pyre_object::gc_roots::shadow_stack_get(source_slot)
                },
            ],
        )?;
        let warning_message_slot = pin_root_slot(warning_message);
        crate::call::call_function_impl_result(
            show,
            &[pyre_object::gc_roots::shadow_stack_get(
                warning_message_slot,
            )],
        )?;
        return Ok(());
    }
    let name = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        "__name__",
    )?;
    let name_slot = pin_root_slot(name);
    let line = format!(
        "{}:{}: {}: {}\n",
        crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(filename_slot))?,
        lineno,
        crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(name_slot))?,
        crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(text_slot))?,
    );
    if let Some(sys) = crate::importing::get_sys_module("sys")
        && let Ok(stderr) = crate::baseobjspace::getattr_str(sys, "stderr")
        && !unsafe { is_none(stderr) }
        && let Ok(write) = crate::baseobjspace::getattr_str(stderr, "write")
    {
        let write_slot = pin_root_slot(write);
        crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(write_slot),
            &[w_str_new(&line)],
        )?;
        let source_line = pyre_object::gc_roots::shadow_stack_get(source_line_slot);
        let source_line = if source_line.is_null() || unsafe { is_none(source_line) } {
            if let Ok(linecache) = import_module("linecache") {
                crate::baseobjspace::getattr_str(linecache, "getline")
                    .and_then(|getline| {
                        crate::call::call_function_impl_result(
                            getline,
                            &[
                                pyre_object::gc_roots::shadow_stack_get(filename_slot),
                                w_int_new(lineno),
                            ],
                        )
                    })
                    .ok()
            } else {
                None
            }
        } else {
            Some(source_line)
        };
        if let Some(source_line) = source_line {
            let source_line_slot = pin_root_slot(source_line);
            let stripped = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(source_line_slot),
                "strip",
            )
            .and_then(|strip| crate::call::call_function_impl_result(strip, &[]))?;
            let stripped = crate::baseobjspace::text_w(stripped)?;
            let visible = stripped.trim_start_matches([' ', '\t', '\u{000c}']);
            if !visible.is_empty() {
                crate::call::call_function_impl_result(
                    pyre_object::gc_roots::shadow_stack_get(write_slot),
                    &[w_str_new(&format!("  {visible}\n"))],
                )?;
            }
        }
    }
    Ok(())
}

fn get_source_line(module_globals: PyObjectRef, lineno: i64) -> Result<PyObjectRef, PyError> {
    if module_globals.is_null() || unsafe { is_none(module_globals) } {
        return Ok(PY_NULL);
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let globals_slot = pin_root_slot(module_globals);
    let Some(loader) = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(globals_slot),
        "__loader__",
    )?
    else {
        return Ok(PY_NULL);
    };
    let loader_slot = pin_root_slot(loader);
    let Some(module_name) = crate::baseobjspace::finditem_str(
        pyre_object::gc_roots::shadow_stack_get(globals_slot),
        "__name__",
    )?
    else {
        return Ok(PY_NULL);
    };
    let module_name_slot = pin_root_slot(module_name);
    let get_source = match crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(loader_slot),
        "get_source",
    ) {
        Ok(get_source) => get_source,
        Err(err) if matches!(err.kind, crate::PyErrorKind::AttributeError) => return Ok(PY_NULL),
        Err(err) => return Err(err),
    };
    let get_source_slot = pin_root_slot(get_source);
    let source = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(get_source_slot),
        &[pyre_object::gc_roots::shadow_stack_get(module_name_slot)],
    )?;
    if unsafe { is_none(source) } {
        return Ok(PY_NULL);
    }
    let source_slot = pin_root_slot(source);
    let splitlines = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(source_slot),
        "splitlines",
    )?;
    let splitlines_slot = pin_root_slot(splitlines);
    let lines = crate::call::call_function_impl_result(
        pyre_object::gc_roots::shadow_stack_get(splitlines_slot),
        &[],
    )?;
    let lines_slot = pin_root_slot(lines);
    match crate::baseobjspace::getitem(
        pyre_object::gc_roots::shadow_stack_get(lines_slot),
        w_int_new(lineno - 1),
    ) {
        Ok(line) => Ok(line),
        Err(err)
            if matches!(
                err.kind,
                crate::PyErrorKind::TypeError | crate::PyErrorKind::IndexError
            ) =>
        {
            Ok(PY_NULL)
        }
        Err(err) => Err(err),
    }
}

fn do_warn_explicit(
    category: PyObjectRef,
    message: PyObjectRef,
    filename: PyObjectRef,
    lineno: i64,
    module: PyObjectRef,
    registry: PyObjectRef,
    source_line: PyObjectRef,
    source: PyObjectRef,
) -> Result<(), PyError> {
    // RPython keeps these values in livevars across every allocating helper.
    // Native Rust locals are invisible to the moving collector, so mirror
    // that lifetime with the established temporary shadow-stack bracket.
    let _roots = pyre_object::gc_roots::push_roots();
    let category_slot = pin_root_slot(category);
    let input_message_slot = pin_root_slot(message);
    let filename_slot = pin_root_slot(filename);
    let input_module_slot = pin_root_slot(module);
    let registry_slot = pin_root_slot(registry);
    let source_line_slot = pin_root_slot(source_line);
    let source_slot = pin_root_slot(source);

    let input_module = pyre_object::gc_roots::shadow_stack_get(input_module_slot);
    let module = if input_module.is_null() || unsafe { is_none(input_module) } {
        let filename = pyre_object::gc_roots::shadow_stack_get(filename_slot);
        if unsafe { !is_str(filename) } {
            // Preserve `space.fsencode_w`'s TypeError for non-text filenames.
            let _ = crate::baseobjspace::text_w(filename)?;
            unreachable!();
        }
        let filename_text = unsafe { w_str_get_wtf8(filename) };
        let filename_bytes = filename_text.as_bytes();
        if filename_bytes.is_empty() {
            w_str_new("<unknown>")
        } else if filename_bytes.ends_with(b".py") {
            let stem = rustpython_wtf8::Wtf8Buf::from_bytes(
                filename_bytes[..filename_bytes.len() - 3].to_vec(),
            )
            .expect("removing an ASCII suffix preserves WTF-8");
            w_str_from_wtf8(stem)
        } else {
            filename
        }
    } else {
        input_module
    };
    let module_slot = pin_root_slot(module);
    let warning = warning_class("Warning");
    let input_message = pyre_object::gc_roots::shadow_stack_get(input_message_slot);
    let category = pyre_object::gc_roots::shadow_stack_get(category_slot);
    let (text, message, category) = if crate::baseobjspace::isinstance(input_message, warning)? {
        (
            crate::builtins::builtin_str(&[pyre_object::gc_roots::shadow_stack_get(
                input_message_slot,
            )])?,
            pyre_object::gc_roots::shadow_stack_get(input_message_slot),
            crate::typedef::r#type(pyre_object::gc_roots::shadow_stack_get(input_message_slot))
                .unwrap_or(category),
        )
    } else {
        let input_message = pyre_object::gc_roots::shadow_stack_get(input_message_slot);
        let text = if unsafe { is_str(input_message) || is_bytes(input_message) } {
            input_message
        } else {
            crate::builtins::builtin_str(&[input_message])?
        };
        let text_slot = pin_root_slot(text);
        let instance = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(category_slot),
            &[pyre_object::gc_roots::shadow_stack_get(input_message_slot)],
        )?;
        let text = pyre_object::gc_roots::shadow_stack_get(text_slot);
        (text, instance, category)
    };
    let text_slot = pin_root_slot(text);
    let message_slot = pin_root_slot(message);
    let category_slot = pin_root_slot(category);
    let key = w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(text_slot),
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        w_int_new(lineno),
    ]);
    let key_slot = pin_root_slot(key);
    let registry = pyre_object::gc_roots::shadow_stack_get(registry_slot);
    let has_registry = !registry.is_null() && unsafe { !is_none(registry) };
    if has_registry
        && already_warned(
            pyre_object::gc_roots::shadow_stack_get(registry_slot),
            pyre_object::gc_roots::shadow_stack_get(key_slot),
            false,
        )?
    {
        return Ok(());
    }
    let (action, item) = get_filter(
        pyre_object::gc_roots::shadow_stack_get(category_slot),
        pyre_object::gc_roots::shadow_stack_get(text_slot),
        lineno,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
    )?;
    if action == "error" {
        return Err(unsafe {
            PyError::from_exc_object(pyre_object::gc_roots::shadow_stack_get(message_slot))
        });
    }
    if action == "ignore" {
        return Ok(());
    }
    let mut warned = false;
    if action != "always" && action != "all" {
        if has_registry {
            crate::baseobjspace::setitem(
                pyre_object::gc_roots::shadow_stack_get(registry_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                w_bool_from(true),
            )?;
        }
        if action == "once" {
            let once = if has_registry {
                pyre_object::gc_roots::shadow_stack_get(registry_slot)
            } else {
                get_once_registry()?
            };
            warned = update_registry(
                once,
                pyre_object::gc_roots::shadow_stack_get(text_slot),
                pyre_object::gc_roots::shadow_stack_get(category_slot),
            )?;
        } else if action == "module" {
            if has_registry {
                warned = update_registry(
                    pyre_object::gc_roots::shadow_stack_get(registry_slot),
                    pyre_object::gc_roots::shadow_stack_get(text_slot),
                    pyre_object::gc_roots::shadow_stack_get(category_slot),
                )?;
            }
        } else if action != "default" {
            return Err(PyError::runtime_error(format!(
                "Unrecognized action ({action}) in warnings.filters: {}",
                if item.is_null() { "???" } else { "filter item" }
            )));
        }
    }
    if !warned {
        show_warning(
            pyre_object::gc_roots::shadow_stack_get(message_slot),
            pyre_object::gc_roots::shadow_stack_get(text_slot),
            pyre_object::gc_roots::shadow_stack_get(category_slot),
            pyre_object::gc_roots::shadow_stack_get(filename_slot),
            lineno,
            pyre_object::gc_roots::shadow_stack_get(source_line_slot),
            pyre_object::gc_roots::shadow_stack_get(source_slot),
        )?;
    }
    Ok(())
}

fn filters_mutated(_: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    filters_mutated_impl()?;
    Ok(w_none())
}

// Python 3.14's `warnings.py` expects the accelerator to provide the lock
// hooks. PyPy's State has no separate lock, and pyre currently executes
// Python on one interpreter thread (`module/thread/mod.rs`), so there is no
// concurrent access to serialize. Keep these process-wide no-ops rather than
// inventing thread-local ownership for shared warning state.
fn lock_noop(_: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(w_none())
}

crate::py_module! {
    "_warnings",
    interpleveldefs: {
        "_warnings_context" => w_none(),
    },
    inline_functions: {
        fn warn(
            message: PyObjectRef,
            #[default(pyre_object::PY_NULL)] category: PyObjectRef,
            #[default(1i64)] stacklevel: i64,
            #[kwonly] #[default(pyre_object::PY_NULL)] source: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let category = get_category(message, category)?;
            let (filename, lineno, module, registry) = setup_context(stacklevel);
            do_warn_explicit(
                category, message, filename, lineno, module, registry,
                pyre_object::PY_NULL, source,
            )?;
            Ok(w_none())
        }

        fn warn_explicit(
            message: PyObjectRef,
            category: PyObjectRef,
            filename: PyObjectRef,
            lineno: i64,
            #[default(pyre_object::PY_NULL)] module: PyObjectRef,
            #[default(pyre_object::PY_NULL)] registry: PyObjectRef,
            #[default(pyre_object::PY_NULL)] module_globals: PyObjectRef,
            #[kwonly] #[default(pyre_object::PY_NULL)] source: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let source_line = get_source_line(module_globals, lineno)?;
            let _roots = pyre_object::gc_roots::push_roots();
            let source_line_slot = pin_root_slot(source_line);
            let category = get_category(message, category)?;
            do_warn_explicit(
                category, message, filename, lineno, module, registry,
                pyre_object::gc_roots::shadow_stack_get(source_line_slot), source,
            )?;
            Ok(w_none())
        }
    },
    functions: {
        "_filters_mutated" / 0 = filters_mutated,
        "_filters_mutated_lock_held" / 0 = filters_mutated,
        "_acquire_lock" / 0 = lock_noop,
        "_release_lock" / 0 = lock_noop,
    },
    extra_init: |ns| {
        let filters = w_list_new(vec![
            create_filter(warning_class("DeprecationWarning"), "default", Some("__main__")),
            create_filter(warning_class("DeprecationWarning"), "ignore", None),
            create_filter(warning_class("PendingDeprecationWarning"), "ignore", None),
            create_filter(warning_class("ImportWarning"), "ignore", None),
            create_filter(warning_class("ResourceWarning"), "ignore", None),
        ]);
        crate::module_ns_store(ns, "filters", filters);
        crate::module_ns_store(ns, "_onceregistry", w_dict_new());
        crate::module_ns_store(ns, "_defaultaction", w_str_new("default"));
        crate::module_ns_store(ns, VERSION_ATTR, new_version());
    },
}
