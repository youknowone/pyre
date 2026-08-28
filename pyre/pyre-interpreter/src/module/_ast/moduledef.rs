//! _ast implementation — PyPy: pypy/module/_ast/moduledef.py +
//! pypy/interpreter/astcompiler/ast.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use pyre_object::PyObjectRef;

const LOCATION_ATTRIBUTES: &[&str] = &["lineno", "col_offset", "end_lineno", "end_col_offset"];

// CPython 3.14 `ast_state.AST_type` / `Load_singleton`.  PyPy's `State` keeps
// the generated AST types process-wide; keep both movable objects in
// registered process-lifetime root slots.
static LOAD_SINGLETON: std::sync::OnceLock<Box<usize>> = std::sync::OnceLock::new();
static AST_TYPE: std::sync::OnceLock<Box<usize>> = std::sync::OnceLock::new();

fn register_ast_type(value: PyObjectRef) {
    let _ = AST_TYPE.get_or_init(|| {
        let mut slot = Box::new(value as usize);
        let root = (&mut *slot) as *mut usize as *mut *mut u8;
        unsafe { pyre_object::gc_hook::try_gc_add_root(root) };
        slot
    });
}

#[majit_macros::dont_look_inside]
fn ast_type() -> PyObjectRef {
    **AST_TYPE
        .get()
        .expect("_ast AST type initialized before AST operations") as PyObjectRef
}

fn register_load_singleton(value: PyObjectRef) {
    let _ = LOAD_SINGLETON.get_or_init(|| {
        let mut slot = Box::new(value as usize);
        let root = (&mut *slot) as *mut usize as *mut *mut u8;
        unsafe { pyre_object::gc_hook::try_gc_add_root(root) };
        slot
    });
}

#[majit_macros::dont_look_inside]
pub(crate) fn load_singleton() -> PyObjectRef {
    **LOAD_SINGLETON
        .get()
        .expect("_ast Load singleton initialized before AST construction")
        as PyObjectRef
}

fn tuple_of_names(names: &[&str]) -> PyObjectRef {
    pyre_object::w_tuple_new(names.iter().map(|name| pyre_object::w_str_new(name)).collect())
}

fn ast_instance_type_name(object: PyObjectRef) -> &'static str {
    // PyPy `W_AST_init` asks `space.type(self)`: every public AST node uses
    // the common `W_ObjectObject` payload, so its payload vtable only says
    // `object`; the heap class lives in `w_class`.
    unsafe {
        pyre_object::w_type_get_name(pyre_object::w_instance_get_type(object))
    }
}

fn ast_fields_owner_name(w_type: PyObjectRef) -> String {
    let name = unsafe { pyre_object::w_type_get_name(w_type) };
    if name == "AST" {
        let bases = unsafe { pyre_object::w_type_get_bases(w_type) };
        if unsafe { pyre_object::w_tuple_len(bases) } == 1
            && unsafe { pyre_object::w_tuple_getitem(bases, 0) }
                == Some(crate::typedef::w_object())
        {
            // CPython's generated root retains the static tp_name `ast.AST`,
            // while PyPy `State.make_new_type` owns it as the heap type `AST`.
            // Keep the PyPy owner/storage and reproduce the 3.14 observable
            // name only at `ast_type_init`'s missing `_fields` diagnostic.
            return "ast.AST".to_owned();
        }
    }
    name.to_owned()
}

fn ast_field_repr(field: PyObjectRef) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    unsafe { crate::display::py_repr_wtf8(field) }
}

fn ast_repr_type_name(object: PyObjectRef) -> String {
    let w_type = crate::typedef::r#type(object)
        .expect("AST instance has a live type")
        .as_ptr();
    if std::ptr::eq(w_type, ast_type()) {
        // CPython's generated root has the static tp_name `ast.AST`; PyPy's
        // W_AST TypeDef owns the same root while generated children use their
        // unqualified ASDL names.
        "ast.AST".to_owned()
    } else {
        unsafe { pyre_object::w_type_get_name(w_type) }.to_owned()
    }
}

fn is_ast_instance(object: PyObjectRef) -> bool {
    let Some(w_type) = crate::typedef::r#type(object) else {
        return false;
    };
    unsafe { pyre_object::w_type_issubtype(w_type.as_ptr(), ast_type()) }
}

// CPython publishes both methods through `AST_type_slots` (`Py_tp_init` /
// `Py_tp_repr`), while PyPy's `TypeDef` turns `interp2app(W_AST.descr_*)` into
// the corresponding wrapper descriptors.  The generated AST root is a heap
// type built after the ordinary builtin-owner table, so bind its owner
// explicitly as it is minted.
static AST_METHOD_OWNER: crate::gateway::MethodOwner = crate::gateway::MethodOwner {
    type_name: "ast.AST",
    is_instance: Some(is_ast_instance),
};

fn ast_slot_wrapper(
    name: &'static str,
    function: crate::gateway::BuiltinCodeFn,
    arity: Option<u16>,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let wrapper = match arity {
        Some(arity) => crate::gateway::make_slot_wrapper_with_arity(name, function, arity),
        None => crate::gateway::make_slot_wrapper(name, function),
    };
    let wrapper_slot = roots.base();
    let _ = roots.pin_root(wrapper);
    let code = unsafe { crate::function::getcode(roots.get(wrapper_slot)) } as PyObjectRef;
    unsafe { crate::gateway::builtin_code_set_owner(code, &AST_METHOD_OWNER) };
    let qualname = pyre_object::w_str_new(&format!("AST.{name}"));
    unsafe {
        crate::function::function_set_qualname(roots.get(wrapper_slot), qualname);
        crate::function::function_set_objclass(roots.get(wrapper_slot), ast_type());
    }
    roots.get(wrapper_slot)
}

/// CPython 3.14 `ast_repr_list`, expressed through PyPy object-space sequence
/// operations.  In particular, list/tuple subclasses keep their `__len__`
/// and `__getitem__` overrides, and only the first and last items are read.
fn ast_repr_list(
    sequence: PyObjectRef,
    depth: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let sequence_slot = roots.base();
    let _ = roots.pin_root(sequence);
    let sequence = || roots.get(sequence_slot);
    let length = crate::baseobjspace::len_w(sequence())?;
    if length == 0 {
        return unsafe { crate::display::py_repr_wtf8(sequence()) };
    }

    // Allocate and root the index before re-reading the movable sequence for
    // the object-space call.
    let first_index_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(pyre_object::w_int_new(0));
    let first = crate::baseobjspace::getitem(sequence(), roots.get(first_index_slot))?;
    let first_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(first);
    let last_slot = if length > 1 {
        let last_index_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_int_new(length - 1));
        let last = crate::baseobjspace::getitem(sequence(), roots.get(last_index_slot))?;
        let slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(last);
        Some(slot)
    } else {
        None
    };

    let is_list = unsafe { pyre_object::is_list(sequence()) };
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    out.push_str(if is_list { "[" } else { "(" });
    for index in 0..std::cmp::min(length, 2) {
        if index > 0 {
            out.push_str(", ");
        }
        let item = if index == 0 {
            roots.get(first_slot)
        } else {
            roots.get(last_slot.expect("second AST sequence item was rooted"))
        };
        if is_ast_instance(item) {
            out.push_wtf8(&ast_repr_max_depth(item, depth - 1)?);
        } else {
            out.push_wtf8(&unsafe { crate::display::py_repr_wtf8(item)? });
        }
        if index == 0 && length > 2 {
            out.push_str(", ...");
        }
    }
    out.push_str(if is_list { "]" } else { ")" });
    Ok(out)
}

/// [3.14-spec] PyPy `W_AST` has no custom repr and no JIT/immutability hint on
/// the definition; the pinned 3.14 `AST_Tests.test_repr` observes the generated
/// `ast_repr_max_depth` surface.  Keep PyPy's W_AST owner, subtype tests and
/// object-space dispatch while porting that bounded formatting walk literally.
fn ast_repr_max_depth(
    object: PyObjectRef,
    depth: i64,
) -> Result<rustpython_wtf8::Wtf8Buf, crate::PyError> {
    let roots = pyre_object::gc_roots::push_roots();
    let object_slot = roots.base();
    let _ = roots.pin_root(object);
    let object = || roots.get(object_slot);
    let type_name = ast_repr_type_name(object());
    if depth <= 0 {
        return Ok(rustpython_wtf8::Wtf8Buf::from_string(format!(
            "{type_name}(...)"
        )));
    }
    let Some(_guard) = crate::display::ReprGuard::enter(object()) else {
        return Ok(rustpython_wtf8::Wtf8Buf::from_string(format!(
            "{type_name}(...)"
        )));
    };

    let w_type = crate::typedef::r#type(object())
        .expect("AST instance has a live type")
        .as_ptr();
    let type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(w_type);
    let fields = crate::baseobjspace::getattr_str(roots.get(type_slot), "_fields")?;
    let fields_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(fields);
    let num_fields = crate::baseobjspace::len_w(roots.get(fields_slot))?;
    if num_fields == 0 {
        return Ok(rustpython_wtf8::Wtf8Buf::from_string(format!(
            "{type_name}()"
        )));
    }

    let mut out = rustpython_wtf8::Wtf8Buf::from_string(format!("{type_name}("));
    for index in 0..num_fields {
        let index_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_int_new(index));
        let name =
            crate::baseobjspace::getitem(roots.get(fields_slot), roots.get(index_slot))?;
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(name);
        let value = crate::baseobjspace::getattr(object(), roots.get(name_slot))?;
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(value);
        let value = roots.get(value_slot);
        let value_repr = if unsafe { pyre_object::is_list(value) || pyre_object::is_tuple(value) }
        {
            ast_repr_list(value, depth)?
        } else if is_ast_instance(value) {
            ast_repr_max_depth(value, depth - 1)?
        } else {
            unsafe { crate::display::py_repr_wtf8(value)? }
        };

        if index > 0 {
            out.push_str(", ");
        }
        out.push_wtf8(crate::baseobjspace::text_wtf8_w(roots.get(name_slot))?);
        out.push_str("=");
        out.push_wtf8(&value_repr);
    }
    out.push_str(")");
    Ok(out)
}

fn ast_repr(args: &[PyObjectRef]) -> crate::PyResult {
    Ok(pyre_object::w_str_from_wtf8_managed(ast_repr_max_depth(
        args[0], 3,
    )?))
}

fn ast_warn(message: rustpython_wtf8::Wtf8Buf) -> Result<(), crate::PyError> {
    crate::warn::warn_category_w(
        pyre_object::w_str_from_wtf8(message),
        "DeprecationWarning",
        2,
    )
}

fn is_ast_init_descriptor(descr: PyObjectRef) -> bool {
    if !unsafe { crate::is_function(descr) } {
        return false;
    }
    let code = unsafe { crate::getcode(descr) } as PyObjectRef;
    (unsafe { crate::is_builtin_code(code) })
        && crate::gateway::builtin_code_fn_eq(
            unsafe { crate::gateway::builtin_code_get(code) },
            ast_init,
        )
}

/// [3.14-spec] CPython `ASTConstructorTests.test_non_str_kwarg` observes that
/// `ast_type_init` receives the raw kwargs dict, while PyPy `Arguments`
/// rejects a non-text key in `_do_combine_starstarargs_wrapped` before
/// `W_AST_init`.  Keep that PyPy path for every ordinary callable.  Only when
/// the live type still owns the generated AST init, inherits `object.__new__`
/// unchanged, and uses the default metaclass do we invoke those same two
/// descriptor bodies with pyre's raw builtin-kwargs carrier.
pub(crate) fn call_type_with_raw_kwargs(
    w_type: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs: PyObjectRef,
) -> Option<crate::PyResult> {
    if !unsafe { pyre_object::is_type(w_type) && pyre_object::is_dict(kwargs) } {
        return None;
    }
    let entries = unsafe { pyre_object::w_dict_items(kwargs) };
    if entries
        .iter()
        .all(|(key, _)| unsafe { pyre_object::is_str(*key) })
    {
        return None;
    }
    if unsafe { pyre_object::w_instance_get_type(w_type) } != crate::typedef::w_type() {
        return None;
    }
    let init_descr = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__init__") }?;
    if !is_ast_init_descriptor(init_descr) {
        return None;
    }
    let new_descr = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") }?;
    let object_new = unsafe {
        crate::baseobjspace::lookup_in_type(crate::typedef::w_object(), "__new__")
    }?;
    if new_descr != object_new {
        return None;
    }

    Some((|| -> crate::PyResult {
        // The type, positionals and kwargs entries all cross object.__new__
        // and ast_type_init, both allocating calls.  Publish the caller-owned
        // slice before packing the raw carrier and rebuild each argument list
        // from the forwarded slots.
        let roots = pyre_object::gc_roots::push_roots();
        let type_slot = roots.publish(&[w_type]);
        let positional_base = roots.publish(positional);
        let kwargs_slot = roots.publish(&[kwargs]);
        roots.normalize(type_slot, positional.len() + 2);
        let current_positional = |index: usize| roots.get(positional_base + index);

        let current_entries = unsafe { pyre_object::w_dict_items(roots.get(kwargs_slot)) };
        let carrier = crate::call::pack_pyre_kwargs(&current_entries);
        let carrier_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(carrier);

        let mut new_args = Vec::with_capacity(positional.len() + 2);
        new_args.push(roots.get(type_slot));
        for index in 0..positional.len() {
            new_args.push(current_positional(index));
        }
        new_args.push(roots.get(carrier_slot));
        let instance = crate::typedef::object_descr_new(&new_args)?;
        let instance_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(instance);

        let mut init_args = Vec::with_capacity(positional.len() + 2);
        init_args.push(roots.get(instance_slot));
        for index in 0..positional.len() {
            init_args.push(current_positional(index));
        }
        init_args.push(roots.get(carrier_slot));
        ast_init(&init_args)?;
        Ok(roots.get(instance_slot))
    })())
}

fn expr_context_type() -> PyObjectRef {
    // CPython `ast_state.expr_context_type`; PyPy's `State` also owns this
    // generated base beside the Load singleton.  Recover that same base from
    // the singleton instead of adding a second semantic side table.
    unsafe {
        let load_type = pyre_object::w_instance_get_type(load_singleton());
        let bases = pyre_object::w_type_get_bases(load_type);
        pyre_object::w_tuple_getitem(bases, 0).expect("Load has expr_context base")
    }
}

/// [3.14-spec] PyPy `W_AST_init` only assigns supplied arguments.  The pinned
/// `ASTConstructorTests` require CPython 3.14 `ast_type_init`'s observable
/// missing-field defaults/deprecations and unexpected-keyword deprecation.
/// Keep PyPy's generated-type owner and general object-space dispatch while
/// porting that constructor decision tree here.
fn ast_init(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let Some((_, values)) = positional.split_first() else {
        return Err(crate::PyError::type_error("AST.__init__() missing self"));
    };
    // `builtin_code_call` hands the body a native slice the collector does not
    // update.  Publish self as well as every value: attribute lookup, equality,
    // repr, warnings and stores all run Python and may collect.
    let roots = pyre_object::gc_roots::push_roots();
    let positional_base = roots.publish(positional);
    let kwargs_slot = kwargs.map(|kwargs| roots.publish(&[kwargs]));
    roots.normalize(
        positional_base,
        positional.len() + usize::from(kwargs.is_some()),
    );
    let zelf = || roots.get(positional_base);
    let w_type_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(unsafe { pyre_object::w_instance_get_type(zelf()) });
    let fields_obj =
        match crate::baseobjspace::findattr_result(roots.get(w_type_slot), "_fields")? {
            Some(fields_obj) => fields_obj,
            None => {
                let owner = ast_fields_owner_name(roots.get(w_type_slot));
                return Err(crate::PyError::attribute_error_with_context(
                    format!("type object '{owner}' has no attribute '_fields'"),
                    roots.get(w_type_slot),
                    "_fields",
                ));
            }
        };
    let fields_obj_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(fields_obj);
    // PyPy `W_AST_init` uses `space.fixedview`, and CPython uses the sequence
    // protocol; user subclasses may therefore replace the generated tuple.
    let fields = crate::baseobjspace::fixedview(roots.get(fields_obj_slot), -1)?;
    let fields_base = roots.publish(&fields);
    roots.normalize(fields_base, fields.len());
    let num_fields = fields.len();
    if values.len() > num_fields {
        return Err(crate::PyError::type_error(format!(
            "{} constructor takes at most {} positional argument{}",
            ast_instance_type_name(zelf()),
            num_fields,
            if num_fields == 1 { "" } else { "s" }
        )));
    }
    // CPython `ast_type_init` uses a set, so duplicate or unhashable custom
    // `_fields` have the same constructor semantics.  Build it through PyPy's
    // set operation rather than a host side-table.
    let remaining_fields = pyre_object::w_set_new();
    let remaining_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(remaining_fields);
    // Build the raw carrier only after `w_set_new` has had its allocation
    // safepoint; `builtin_set_add_items` publishes every member before hashing.
    let rooted_fields: Vec<_> = (0..num_fields)
        .map(|index| roots.get(fields_base + index))
        .collect();
    crate::builtins::builtin_set_add_items(roots.get(remaining_slot), &rooted_fields)?;
    for index in 0..values.len() {
        crate::baseobjspace::setattr(
            zelf(),
            roots.get(fields_base + index),
            roots.get(positional_base + index + 1),
        )?;
        crate::type_methods::set_discard_checked(
            roots.get(remaining_slot),
            roots.get(fields_base + index),
        )?;
    }

    let mut attributes_slot = None;
    if let Some(kwargs_slot) = kwargs_slot {
        let entries = unsafe { pyre_object::w_dict_items(roots.get(kwargs_slot)) };
        let mut entry_objects = Vec::with_capacity(entries.len() * 2);
        for &(key, value) in &entries {
            entry_objects.push(key);
            entry_objects.push(value);
        }
        let entries_base = roots.publish(&entry_objects);
        roots.normalize(entries_base, entry_objects.len());
        for index in 0..entries.len() {
            let key_slot = entries_base + index * 2;
            let value_slot = key_slot + 1;
            let key = roots.get(key_slot);
            if unsafe {
                pyre_object::is_str(key) && pyre_object::w_str_get_value(key) == "__pyre_kw__"
            } {
                continue;
            }

            if crate::baseobjspace::contains(roots.get(fields_obj_slot), roots.get(key_slot))? {
                if !crate::type_methods::set_discard_checked(
                    roots.get(remaining_slot),
                    roots.get(key_slot),
                )? {
                    let key_repr = ast_field_repr(roots.get(key_slot))?;
                    return Err(crate::PyError::type_error(crate::display::wtf8_format!(
                        ast_instance_type_name(zelf()),
                        " got multiple values for argument ",
                        key_repr
                    )));
                }
            } else {
                let attributes = if let Some(slot) = attributes_slot {
                    roots.get(slot)
                } else {
                    let value = crate::baseobjspace::getattr_str(
                        roots.get(w_type_slot),
                        "_attributes",
                    )?;
                    let slot = pyre_object::gc_roots::shadow_stack_len();
                    let _ = roots.pin_root(value);
                    attributes_slot = Some(slot);
                    roots.get(slot)
                };
                if !crate::baseobjspace::contains(attributes, roots.get(key_slot))? {
                    let key_repr = ast_field_repr(roots.get(key_slot))?;
                    ast_warn(crate::display::wtf8_format!(
                        ast_instance_type_name(zelf()),
                        ".__init__ got an unexpected keyword argument ",
                        key_repr,
                        ". Support for arbitrary keyword arguments is deprecated and will be ",
                        "removed in Python 3.15."
                    ))?;
                }
            }
            crate::baseobjspace::setattr(
                zelf(),
                roots.get(key_slot),
                roots.get(value_slot),
            )?;
        }
    }

    if unsafe { pyre_object::setobject::w_set_len(roots.get(remaining_slot)) } == 0 {
        return Ok(pyre_object::w_none());
    }
    let Some(field_types) =
        crate::baseobjspace::findattr_result(roots.get(w_type_slot), "_field_types")?
    else {
        // CPython 3.14 `ast_type_init`: user AST subclasses without generated
        // metadata keep the pre-3.13 behavior and leave omitted fields absent.
        return Ok(pyre_object::w_none());
    };
    let field_types_slot = pyre_object::gc_roots::shadow_stack_len();
    let _ = roots.pin_root(field_types);
    let remaining = unsafe { pyre_object::setobject::w_set_items(roots.get(remaining_slot)) };
    let remaining_base = roots.publish(&remaining);
    roots.normalize(remaining_base, remaining.len());
    for index in 0..remaining.len() {
        let field_slot = remaining_base + index;
        let Some(field_type) = crate::baseobjspace::finditem(
            roots.get(field_types_slot),
            roots.get(field_slot),
        )?
        else {
            let field_repr = ast_field_repr(roots.get(field_slot))?;
            ast_warn(crate::display::wtf8_format!(
                "Field ",
                field_repr,
                " is missing from ",
                ast_instance_type_name(zelf()),
                "._field_types. This will become an error in Python 3.15."
            ))?;
            continue;
        };
        let field_type_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(field_type);
        if unsafe { pyre_object::is_union(roots.get(field_type_slot)) } {
            // Optional fields already have their inherited class-level None.
        } else if unsafe { pyre_object::is_generic_alias(roots.get(field_type_slot)) } {
            let empty = pyre_object::w_list_new(Vec::new());
            let empty_slot = pyre_object::gc_roots::shadow_stack_len();
            let _ = roots.pin_root(empty);
            crate::baseobjspace::setattr(
                zelf(),
                roots.get(field_slot),
                roots.get(empty_slot),
            )?;
        } else if crate::baseobjspace::is_w(roots.get(field_type_slot), expr_context_type()) {
            crate::baseobjspace::setattr(zelf(), roots.get(field_slot), load_singleton())?;
        } else {
            let field_repr = ast_field_repr(roots.get(field_slot))?;
            ast_warn(crate::display::wtf8_format!(
                ast_instance_type_name(zelf()),
                ".__init__ missing 1 required positional argument: ",
                field_repr,
                ". This will become an error in Python 3.15."
            ))?;
        }
    }
    Ok(pyre_object::w_none())
}

/// Optional (`?`) ASDL fields whose PyPy-generated type dictionary supplies
/// `None`.  The location entries live on their ASDL owner and are inherited.
fn node_default_none_fields(name: &str) -> &'static [&'static str] {
    match name {
        "stmt" | "expr" | "excepthandler" => &["end_lineno", "end_col_offset"],
        "FunctionDef" | "AsyncFunctionDef" => &["returns", "type_comment"],
        "Return" => &["value"],
        "Assign" => &["type_comment"],
        "AnnAssign" => &["value"],
        "For" | "AsyncFor" | "With" | "AsyncWith" => &["type_comment"],
        "Raise" => &["exc", "cause"],
        "Assert" => &["msg"],
        "ImportFrom" => &["module", "level"],
        "Yield" => &["value"],
        "FormattedValue" | "Interpolation" => &["format_spec"],
        "Constant" => &["kind"],
        "Slice" => &["lower", "upper", "step"],
        "ExceptHandler" => &["type", "name"],
        "arguments" => &["vararg", "kwarg"],
        "arg" => &["annotation", "type_comment", "end_lineno", "end_col_offset"],
        "keyword" => &["arg", "end_lineno", "end_col_offset"],
        "alias" => &["asname", "end_lineno", "end_col_offset"],
        "withitem" => &["optional_vars"],
        "match_case" => &["guard"],
        "MatchMapping" => &["rest"],
        "MatchStar" => &["name"],
        "MatchAs" => &["pattern", "name"],
        "TypeVar" => &["bound", "default_value"],
        "ParamSpec" | "TypeVarTuple" => &["default_value"],
        _ => &[],
    }
}

fn node_fields(name: &str) -> &'static [&'static str] {
    match name {
        "Module" => &["body", "type_ignores"],
        "Interactive" => &["body"],
        "Expression" => &["body"],
        "FunctionType" => &["argtypes", "returns"],
        "FunctionDef" | "AsyncFunctionDef" => &["name", "args", "body", "decorator_list", "returns", "type_comment", "type_params"],
        "ClassDef" => &["name", "bases", "keywords", "body", "decorator_list", "type_params"],
        "Return" => &["value"],
        "Delete" => &["targets"],
        "Assign" => &["targets", "value", "type_comment"],
        "TypeAlias" => &["name", "type_params", "value"],
        "AugAssign" => &["target", "op", "value"],
        "AnnAssign" => &["target", "annotation", "value", "simple"],
        "For" | "AsyncFor" => &["target", "iter", "body", "orelse", "type_comment"],
        "While" | "If" => &["test", "body", "orelse"],
        "With" | "AsyncWith" => &["items", "body", "type_comment"],
        "Match" => &["subject", "cases"],
        "Raise" => &["exc", "cause"],
        "Try" | "TryStar" => &["body", "handlers", "orelse", "finalbody"],
        "Assert" => &["test", "msg"],
        "Import" => &["names"],
        "ImportFrom" => &["module", "names", "level"],
        "Global" | "Nonlocal" => &["names"],
        "Expr" => &["value"],
        "Pass" | "Break" | "Continue" => &[],
        "BoolOp" => &["op", "values"],
        "NamedExpr" => &["target", "value"],
        "BinOp" => &["left", "op", "right"],
        "UnaryOp" => &["op", "operand"],
        "Lambda" => &["args", "body"],
        "IfExp" => &["test", "body", "orelse"],
        "Dict" => &["keys", "values"],
        "Set" => &["elts"],
        "ListComp" | "SetComp" | "GeneratorExp" => &["elt", "generators"],
        "DictComp" => &["key", "value", "generators"],
        "Await" | "Yield" | "YieldFrom" => &["value"],
        "Compare" => &["left", "ops", "comparators"],
        "Call" => &["func", "args", "keywords"],
        "FormattedValue" => &["value", "conversion", "format_spec"],
        "Interpolation" => &["value", "str", "conversion", "format_spec"],
        "JoinedStr" => &["values"],
        "TemplateStr" => &["values"],
        "Constant" => &["value", "kind"],
        "Attribute" => &["value", "attr", "ctx"],
        "Subscript" => &["value", "slice", "ctx"],
        "Starred" => &["value", "ctx"],
        "Name" => &["id", "ctx"],
        "List" | "Tuple" => &["elts", "ctx"],
        "Slice" => &["lower", "upper", "step"],
        "ExceptHandler" => &["type", "name", "body"],
        "MatchValue" => &["value"],
        "MatchSingleton" => &["value"],
        "MatchSequence" => &["patterns"],
        "MatchMapping" => &["keys", "patterns", "rest"],
        "MatchClass" => &["cls", "patterns", "kwd_attrs", "kwd_patterns"],
        "MatchStar" => &["name"],
        "MatchAs" => &["pattern", "name"],
        "MatchOr" => &["patterns"],
        "TypeIgnore" => &["lineno", "tag"],
        "TypeVar" => &["name", "bound", "default_value"],
        "ParamSpec" | "TypeVarTuple" => &["name", "default_value"],
        "comprehension" => &["target", "iter", "ifs", "is_async"],
        "arguments" => &["posonlyargs", "args", "vararg", "kwonlyargs", "kw_defaults", "kwarg", "defaults"],
        "arg" => &["arg", "annotation", "type_comment"],
        "keyword" => &["arg", "value"],
        "alias" => &["name", "asname"],
        "withitem" => &["context_expr", "optional_vars"],
        "match_case" => &["pattern", "guard", "body"],
        _ => &[],
    }
}

/// CPython 3.14 `add_ast_annotations`: one public field-type spelling per
/// entry in [`node_fields`].  Its generated metadata normalizes the two ASDL
/// `expr?*` fields to `list[expr]`; [`node_signature`] restores the literal
/// spelling only for their docs.  The owner/type creation order remains
/// PyPy's `State.make_new_type` order.
fn node_field_types(name: &str) -> &'static [&'static str] {
    match name {
        "Module" => &["stmt*", "type_ignore*"],
        "Interactive" => &["stmt*"],
        "Expression" => &["expr"],
        "FunctionType" => &["expr*", "expr"],
        "FunctionDef" | "AsyncFunctionDef" => &[
            "identifier", "arguments", "stmt*", "expr*", "expr?", "string?", "type_param*",
        ],
        "ClassDef" => &[
            "identifier", "expr*", "keyword*", "stmt*", "expr*", "type_param*",
        ],
        "Return" => &["expr?"],
        "Delete" => &["expr*"],
        "Assign" => &["expr*", "expr", "string?"],
        "TypeAlias" => &["expr", "type_param*", "expr"],
        "AugAssign" => &["expr", "operator", "expr"],
        "AnnAssign" => &["expr", "expr", "expr?", "int"],
        "For" | "AsyncFor" => &["expr", "expr", "stmt*", "stmt*", "string?"],
        "While" | "If" => &["expr", "stmt*", "stmt*"],
        "With" | "AsyncWith" => &["withitem*", "stmt*", "string?"],
        "Match" => &["expr", "match_case*"],
        "Raise" => &["expr?", "expr?"],
        "Try" | "TryStar" => &["stmt*", "excepthandler*", "stmt*", "stmt*"],
        "Assert" => &["expr", "expr?"],
        "Import" => &["alias*"],
        "ImportFrom" => &["identifier?", "alias*", "int?"],
        "Global" | "Nonlocal" => &["identifier*"],
        "Expr" => &["expr"],
        "Pass" | "Break" | "Continue" => &[],
        "BoolOp" => &["boolop", "expr*"],
        "NamedExpr" => &["expr", "expr"],
        "BinOp" => &["expr", "operator", "expr"],
        "UnaryOp" => &["unaryop", "expr"],
        "Lambda" => &["arguments", "expr"],
        "IfExp" => &["expr", "expr", "expr"],
        "Dict" => &["expr*", "expr*"],
        "Set" => &["expr*"],
        "ListComp" | "SetComp" | "GeneratorExp" => &["expr", "comprehension*"],
        "DictComp" => &["expr", "expr", "comprehension*"],
        "Await" | "YieldFrom" => &["expr"],
        "Yield" => &["expr?"],
        "Compare" => &["expr", "cmpop*", "expr*"],
        "Call" => &["expr", "expr*", "keyword*"],
        "FormattedValue" => &["expr", "int", "expr?"],
        "Interpolation" => &["expr", "constant", "int", "expr?"],
        "JoinedStr" | "TemplateStr" => &["expr*"],
        "Constant" => &["constant", "string?"],
        "Attribute" => &["expr", "identifier", "expr_context"],
        "Subscript" => &["expr", "expr", "expr_context"],
        "Starred" => &["expr", "expr_context"],
        "Name" => &["identifier", "expr_context"],
        "List" | "Tuple" => &["expr*", "expr_context"],
        "Slice" => &["expr?", "expr?", "expr?"],
        "ExceptHandler" => &["expr?", "identifier?", "stmt*"],
        "MatchValue" => &["expr"],
        "MatchSingleton" => &["constant"],
        "MatchSequence" | "MatchOr" => &["pattern*"],
        "MatchMapping" => &["expr*", "pattern*", "identifier?"],
        "MatchClass" => &["expr", "pattern*", "identifier*", "pattern*"],
        "MatchStar" => &["identifier?"],
        "MatchAs" => &["pattern?", "identifier?"],
        "TypeIgnore" => &["int", "string"],
        "TypeVar" => &["identifier", "expr?", "expr?"],
        "ParamSpec" | "TypeVarTuple" => &["identifier", "expr?"],
        "comprehension" => &["expr", "expr", "expr*", "int"],
        "arguments" => &["arg*", "arg*", "arg?", "arg*", "expr*", "arg?", "expr*"],
        "arg" => &["identifier", "expr?", "string?"],
        "keyword" => &["identifier?", "expr"],
        "alias" => &["identifier", "identifier?"],
        "withitem" => &["expr", "expr?"],
        "match_case" => &["pattern", "expr?", "stmt*"],
        _ => &[],
    }
}

fn asdl_base_type(lookup: &dyn Fn(&str) -> Option<PyObjectRef>, name: &str) -> PyObjectRef {
    match name {
        "identifier" | "string" => crate::typedef::gettypeobject(&pyre_object::STR_TYPE),
        "int" => crate::typedef::gettypeobject(&pyre_object::INT_TYPE),
        "constant" => crate::typedef::w_object(),
        _ => lookup(name)
            .unwrap_or_else(|| panic!("_ast ASDL field type {name} was not registered")),
    }
}

fn asdl_field_type(
    lookup: &dyn Fn(&str) -> Option<PyObjectRef>,
    spelling: &str,
) -> crate::PyResult {
    let (base_name, marker) = if let Some(base) = spelling.strip_suffix('*') {
        (base, '*')
    } else if let Some(base) = spelling.strip_suffix('?') {
        (base, '?')
    } else {
        (spelling, ' ')
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base_slot = roots.base();
    let _ = roots.pin_root(asdl_base_type(lookup, base_name));
    match marker {
        '*' => crate::_pypy_generic_alias::make_generic_alias(
            crate::typedef::gettypeobject(&pyre_object::LIST_TYPE),
            roots.get(base_slot),
        ),
        '?' => crate::_pypy_generic_alias::create_union(
            roots.get(base_slot),
            pyre_object::w_none(),
        ),
        _ => Ok(roots.get(base_slot)),
    }
}

fn node_signature(name: &str) -> String {
    let fields = node_fields(name);
    let types = node_field_types(name);
    assert_eq!(fields.len(), types.len(), "ASDL signature fields for {name}");
    if fields.is_empty() {
        name.to_owned()
    } else {
        let fields = types
            .iter()
            .zip(fields)
            .map(|(typ, field)| {
                // `Parser/Python.asdl` permits None placeholders in these two
                // repeated expr fields.  CPython `add_ast_annotations`
                // deliberately publishes `list[expr]`, while the generated
                // docstring retains the literal `expr?*` ASDL spelling.
                let typ = if (name == "Dict" && *field == "keys")
                    || (name == "arguments" && *field == "kw_defaults")
                {
                    "expr?*"
                } else {
                    typ
                };
                format!("{typ} {field}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("{name}({fields})")
    }
}

fn node_doc(name: &str, variants: Option<&[&str]>) -> String {
    let Some(variants) = variants else {
        return node_signature(name);
    };
    let signatures: Vec<_> = variants
        .iter()
        .map(|variant| node_signature(variant))
        .collect();
    if signatures.iter().all(|signature| !signature.contains('(')) {
        format!("{name} = {}", signatures.join(" | "))
    } else {
        let continuation = format!("\n{}| ", " ".repeat(name.len() + 1));
        format!("{name} = {}", signatures.join(&continuation))
    }
}

/// [3.14-spec] PyPy `State.make_new_type` publishes no field-type metadata,
/// while the pinned `lib-python/3/test/test_ast/test_ast.py`
/// `AST_Tests.test_arguments` requires the exact public mapping.  CPython 3.14
/// `add_ast_annotations` attaches it after every ASDL type exists and binds the
/// same dictionary under both names; keep PyPy's generated type owner/order and
/// add only that observable metadata here.
fn install_field_types(lookup: &dyn Fn(&str) -> Option<PyObjectRef>, name: &str) {
    let fields = node_fields(name);
    let types = node_field_types(name);
    assert_eq!(fields.len(), types.len(), "ASDL field/type count for {name}");

    let roots = pyre_object::gc_roots::push_roots();
    let type_slot = roots.base();
    let _ = roots.pin_root(lookup(name).expect("_ast node type registered"));
    let dict_slot = type_slot + 1;
    let _ = roots.pin_root(pyre_object::w_dict_new());
    for (&field, &spelling) in fields.iter().zip(types) {
        let value_roots = pyre_object::gc_roots::push_roots();
        let value_slot = value_roots.base();
        let _ = value_roots.pin_root(
            asdl_field_type(lookup, spelling).expect("construct _ast ASDL field type"),
        );
        let key = pyre_object::w_str_new(field);
        crate::baseobjspace::setitem(
            roots.get(dict_slot),
            key,
            value_roots.get(value_slot),
        )
        .expect("set _ast ASDL field type");
    }
    crate::baseobjspace::setattr_str(
        roots.get(type_slot),
        "_field_types",
        roots.get(dict_slot),
    )
    .expect("set _field_types on _ast node type");
    crate::baseobjspace::setattr_str(
        roots.get(type_slot),
        "__annotations__",
        roots.get(dict_slot),
    )
    .expect("set __annotations__ on _ast node type");
}

fn node_attributes(name: &str) -> Option<&'static [&'static str]> {
    match name {
        "AST" => Some(&[]),
        "stmt" | "expr" | "excepthandler" | "pattern" | "type_param" | "arg" | "keyword"
        | "alias" => Some(LOCATION_ATTRIBUTES),
        _ => None,
    }
}

/// `astcompiler/ast.py State`, reached as `ast.get(space)`:
/// `space.fromcache(State)` builds the node types once and every `_ast`
/// module dict is filled from that one set.  A program that drops
/// `sys.modules['_ast']` and imports the module again is then handed the
/// classes the trees it has already built are instances of.
struct AstState {
    /// Name and class in build order, so a base always precedes the
    /// subclasses that name it.
    ///
    /// A raw run rather than a `Vec`, because a reader and the collector's
    /// walker reach the same slots: neither may mint a Rust reference the
    /// other's would alias.
    entries: *mut (&'static str, PyObjectRef),
    len: usize,
}

/// Published once and never replaced, which is what lets the collector read
/// the table without coordinating with the thread that built it.
static AST_STATE: std::sync::atomic::AtomicPtr<AstState> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

/// Forward the state's classes.  Upstream reaches them through the space's
/// object graph; pyre holds them off-heap, so the collector has to be handed
/// them — between one module dict going away and the next one being filled,
/// this table is their only owner.
pub(crate) fn walk_ast_state_gc(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let ptr = AST_STATE.load(std::sync::atomic::Ordering::Acquire);
    if ptr.is_null() {
        return;
    }
    // The run is complete before the slot names it and its length never
    // changes, so the only word this writes is the class in each entry.
    unsafe {
        let (entries, len) = ((*ptr).entries, (*ptr).len);
        for index in 0..len {
            visitor(&mut (*entries.add(index)).1);
        }
    }
}

/// The one `AstState`, built on the first `_ast` import.
fn ast_state() -> *mut AstState {
    use std::sync::atomic::Ordering;
    let ptr = AST_STATE.load(Ordering::Acquire);
    if !ptr.is_null() {
        return ptr;
    }
    let types = build_ast_types().into_boxed_slice();
    let len = types.len();
    let entries = Box::into_raw(types) as *mut (&'static str, PyObjectRef);
    let created = Box::into_raw(Box::new(AstState { entries, len }));
    match AST_STATE.compare_exchange(
        std::ptr::null_mut(),
        created,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => created,
        Err(existing) => {
            // A loser's classes are ordinary unreachable objects the collector
            // takes; the two host allocations are this call's to release.
            drop(unsafe { Box::from_raw(created) });
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(entries, len)) });
            existing
        }
    }
}

/// The ASDL hierarchy: `AST`, then each abstract group with its concrete
/// members, then the leaf types that descend from `AST` directly.
///
/// Every class is pinned as it is built, because the next `make` allocates and
/// until [`ast_state`] publishes the table this bracket is their only owner.
fn build_ast_types() -> Vec<(&'static str, PyObjectRef)> {
    // `type(name, (base,), {"__module__": "ast"})` — a fresh heap type. The
    // generated AST types report `__module__ == "ast"` (astcompiler/ast.py;
    // the host `_ast.Module.__module__` is likewise `'ast'`).
    let make = |name: &str, base: PyObjectRef, variants: Option<&[&str]>| -> PyObjectRef {
        // PyPy `State.make_new_type` keeps `w_base` and `w_dict` as wrapped
        // objects across every allocation in the body.  Publish both here:
        // the base is also a movable heap type, not a Rust-stable descriptor.
        let roots = pyre_object::gc_roots::push_roots();
        let base_slot = roots.base();
        let _ = roots.pin_root(base);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_dict_new());
        // `make_type` builds ONE tuple of field names and binds it to both
        // `_fields` and `__match_args__`, so `Expr._fields is
        // Expr.__match_args__`. It is pinned beside the namespace dict because
        // the first store allocates a key string and can grow it.
        let fields_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(tuple_of_names(node_fields(name)));
        // The value arrives already built.  Call arguments evaluate left to
        // right, so reading a dict slot inline with an allocating value
        // expression would read the slot first and hand over a pre-move word.
        let put = |target_slot: usize, key: &str, value: PyObjectRef| -> crate::PyResult {
            let value_roots = pyre_object::gc_roots::push_roots();
            let value_slot = value_roots.base();
            let _ = value_roots.pin_root(value);
            let key = pyre_object::w_str_new(key);
            crate::baseobjspace::setitem(roots.get(target_slot), key, value_roots.get(value_slot))
        };
        put(dict_slot, "__module__", pyre_object::w_str_new("ast"))
            .expect("set __module__ on _ast type namespace");
        // `astcompiler/ast.py` `State.make_new_type` publishes one tuple
        // under both names.  `MATCH_CLASS`
        // resolves positional sub-patterns through `__match_args__`, so a
        // `case ast.Expr(expr)` only reaches a field when the node type carries
        // the same field order `_fields` reports.
        put(dict_slot, "_fields", roots.get(fields_slot))
            .expect("set _fields on _ast type namespace");
        // Without this name a class pattern carrying positional sub-patterns
        // raises `TypeError: Expr() accepts 0 positional sub-patterns`.
        // `traceback._extract_caret_anchors_from_line_segment` is written as
        // `case ast.Expr(expr)` over `ast.BinOp` / `ast.Subscript` /
        // `ast.Call`, so every traceback printed through `traceback.py` loses
        // its `~`/`^` anchors without it.
        put(dict_slot, "__match_args__", roots.get(fields_slot))
            .expect("set __match_args__ on _ast type namespace");
        if let Some(attributes) = node_attributes(name) {
            put(dict_slot, "_attributes", tuple_of_names(attributes))
                .expect("set _attributes on _ast type namespace");
        }
        // PyPy `State.make_new_type`: optional ASDL fields are class-level
        // `None` defaults, inherited from the owner for optional attributes.
        for &field in node_default_none_fields(name) {
            put(dict_slot, field, pyre_object::w_none())
                .expect("set optional AST field default");
        }
        // `type_descr_new` reads the metatype off the first argument, the way
        // `descr__new__` takes it as a parameter beside `arguments_w`.  Every
        // other caller reaches it through an attribute lookup that supplies
        // one; this one builds the argument list by hand, so it names `type`.
        let name_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_str_new(name));
        let bases_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = roots.pin_root(pyre_object::w_tuple_new(vec![roots.get(base_slot)]));
        let args = [
            crate::typedef::w_type(),
            roots.get(name_slot),
            roots.get(bases_slot),
            roots.get(dict_slot),
        ];
        let typ = crate::builtins::type_descr_new(&args).expect("_ast heap type creation");
        let type_roots = pyre_object::gc_roots::push_roots();
        let type_slot = type_roots.base();
        let _ = type_roots.pin_root(typ);
        // PyPy `State.make_new_type` assigns its ASDL-generated `doc` after
        // creating each generated type.  The pre-existing `W_AST` root keeps
        // its None docstring.
        if name != "AST" {
            crate::baseobjspace::setattr_str(
                type_roots.get(type_slot),
                "__doc__",
                pyre_object::w_str_new(&node_doc(name, variants)),
            )
            .expect("set generated _ast type doc");
        }
        type_roots.get(type_slot)
    };

    let roots = pyre_object::gc_roots::push_roots();
    let first = roots.base();
    let mut names: Vec<&'static str> = Vec::new();

    // Root: AST(object).
    register_ast_type(make("AST", crate::typedef::w_object(), None));
    let ast_slot = first;
    let _ = roots.pin_root(ast_type());
    names.push("AST");
    // Its own bracket, so the classes below take the slots that follow the
    // root's and `names` stays an index into the run.
    {
        let method_roots = pyre_object::gc_roots::push_roots();
        let init_slot = method_roots.base();
        let _ = method_roots.pin_root(ast_slot_wrapper("__init__", ast_init, None));
        let repr_slot = pyre_object::gc_roots::shadow_stack_len();
        let _ = method_roots.pin_root(ast_slot_wrapper("__repr__", ast_repr, Some(1)));
        crate::baseobjspace::setattr_str(ast_type(), "__init__", method_roots.get(init_slot))
            .expect("set AST.__init__");
        crate::baseobjspace::setattr_str(ast_type(), "__repr__", method_roots.get(repr_slot))
            .expect("set AST.__repr__");
    }

    // Abstract groups (direct AST subclasses) and their concrete members,
    // per the ASDL grammar.
    let groups: &[(&'static str, &[&'static str])] = &[
        ("mod", &["Module", "Interactive", "Expression", "FunctionType"]),
        (
            "stmt",
            &[
                "FunctionDef", "AsyncFunctionDef", "ClassDef", "Return", "Delete", "Assign",
                "TypeAlias", "AugAssign", "AnnAssign", "For", "AsyncFor", "While", "If", "With",
                "AsyncWith", "Match", "Raise", "Try", "TryStar", "Assert", "Import", "ImportFrom",
                "Global", "Nonlocal", "Expr", "Pass", "Break", "Continue",
            ],
        ),
        (
            "expr",
            &[
                "BoolOp", "NamedExpr", "BinOp", "UnaryOp", "Lambda", "IfExp", "Dict", "Set",
                "ListComp", "SetComp", "DictComp", "GeneratorExp", "Await", "Yield", "YieldFrom",
                "Compare", "Call", "FormattedValue", "Interpolation", "JoinedStr",
                "TemplateStr", "Constant", "Attribute", "Subscript", "Starred", "Name", "List",
                "Tuple", "Slice",
            ],
        ),
        ("expr_context", &["Load", "Store", "Del"]),
        ("boolop", &["And", "Or"]),
        (
            "operator",
            &[
                "Add", "Sub", "Mult", "MatMult", "Div", "Mod", "Pow", "LShift", "RShift", "BitOr",
                "BitXor", "BitAnd", "FloorDiv",
            ],
        ),
        ("unaryop", &["Invert", "Not", "UAdd", "USub"]),
        ("cmpop", &["Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn"]),
        ("excepthandler", &["ExceptHandler"]),
        (
            "pattern",
            &[
                "MatchValue", "MatchSingleton", "MatchSequence", "MatchMapping", "MatchClass",
                "MatchStar", "MatchAs", "MatchOr",
            ],
        ),
        ("type_ignore", &["TypeIgnore"]),
        ("type_param", &["TypeVar", "ParamSpec", "TypeVarTuple"]),
    ];
    for (group, members) in groups {
        let group_slot = first + names.len();
        let _ = roots.pin_root(make(group, roots.get(ast_slot), Some(members)));
        names.push(group);
        for m in *members {
            let _ = roots.pin_root(make(m, roots.get(group_slot), None));
            names.push(m);
        }
    }

    // Leaf node types that are direct AST subclasses (no further subclasses).
    let standalone: &[&'static str] = &[
        "comprehension", "arguments", "arg", "keyword", "alias", "withitem", "match_case",
    ];
    for name in standalone {
        let _ = roots.pin_root(make(name, roots.get(ast_slot), None));
        names.push(name);
    }

    // The classes are not in a module dict yet, so the two passes below reach
    // one another by name through the run they were pinned into.
    let lookup = |name: &str| -> Option<PyObjectRef> {
        names
            .iter()
            .position(|built| *built == name)
            .map(|index| roots.get(first + index))
    };
    // CPython's generated `add_ast_annotations` runs only after every type has
    // been created because, for example, FunctionDef refers to `arguments`
    // and `type_param` types declared later in the ASDL traversal.
    for (_, members) in groups {
        for name in *members {
            install_field_types(&lookup, name);
        }
    }
    for name in standalone {
        install_field_types(&lookup, name);
    }

    // CPython 3.14 `ast_state.Load_singleton`: every omitted expression
    // context receives this same object.  Construct it after the ASDL groups
    // have built `Load` and keep it outside the public module namespace.
    let load_roots = pyre_object::gc_roots::push_roots();
    let load_type_slot = load_roots.base();
    let _ = load_roots.pin_root(lookup("Load").expect("_ast.Load built"));
    register_load_singleton(pyre_object::w_instance_new(load_roots.get(load_type_slot)));

    names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, roots.get(first + index)))
        .collect()
}

/// _ast stub — PyPy: pypy/module/_ast/
///
/// Exposes the AST node type hierarchy. The node types are created as **heap
/// types** (via `type(name, bases, {})`) following the ASDL hierarchy
/// (`AST` → abstract group → concrete node), so `ast.py` can subclass them
/// (`class Suite(mod)`) and monkeypatch them (`Tuple.dims = property(...)`),
/// matching CPython where `_ast` types are heap types. Compiler-native Ruff
/// nodes are converted to instances of these public types by `convert.rs`.
pub fn register_module(ns: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
    let state = ast_state();
    // `module_ns_store` allocates the key, so the entry is read back out of the
    // run on each pass rather than held as a view across the stores.
    let (entries, len) = unsafe { ((*state).entries, (*state).len) };
    for index in 0..len {
        let (name, w_type) = unsafe { *entries.add(index) };
        crate::module_ns_store(ns, name, w_type);
    }

    // `compile()` / `ast.parse()` flag bitmasks, used by `lib-python/3/ast.py`
    // (`flags = PyCF_ONLY_AST; flags |= PyCF_TYPE_COMMENTS`). Values are
    // `Include/cpython/compile.h`'s, which `consts.py` matches everywhere but
    // `PyCF_TYPE_COMMENTS`: it kept `PyCF_ASYNC_HACKS` on `0x1000` and moved
    // this one out to `0x40000000`.  3.14 has no `PyCF_ASYNC_HACKS`.
    for (name, value) in &[
        ("PyCF_ONLY_AST", 0x0400i64),
        ("PyCF_ALLOW_TOP_LEVEL_AWAIT", 0x2000),
        ("PyCF_TYPE_COMMENTS", 0x1000),
        // CPython 3.14 Include/cpython/compile.h: requesting an optimized
        // tree necessarily requests an AST result as well.
        ("PyCF_OPTIMIZED_AST", 0x8000 | 0x0400),
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(*value));
    }
    Ok(())
}
