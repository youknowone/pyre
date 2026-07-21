use crate::PyError;
use crate::bytecode::{BinaryOperator, ComparisonOperator};
use pyre_object::{PyObjectRef, w_bool_from};

use crate::{
    CompareOp, add, and_, compare, floordiv, getitem, invert, is_true, lshift, matmul, mod_, mul,
    neg, or_, pow, rshift, sub, truediv, xor,
};

/// Maps an in-place `BinaryOperator` to its special-method name
/// (`__iadd__` etc.), or `None` for non-in-place operators.
fn inplace_dunder_name(op: BinaryOperator) -> Option<&'static str> {
    Some(match op {
        BinaryOperator::InplaceAdd => "__iadd__",
        BinaryOperator::InplaceSubtract => "__isub__",
        BinaryOperator::InplaceMultiply => "__imul__",
        BinaryOperator::InplaceFloorDivide => "__ifloordiv__",
        BinaryOperator::InplaceRemainder => "__imod__",
        BinaryOperator::InplaceTrueDivide => "__itruediv__",
        BinaryOperator::InplacePower => "__ipow__",
        BinaryOperator::InplaceLshift => "__ilshift__",
        BinaryOperator::InplaceMatrixMultiply => "__imatmul__",
        BinaryOperator::InplaceRshift => "__irshift__",
        BinaryOperator::InplaceAnd => "__iand__",
        BinaryOperator::InplaceOr => "__ior__",
        BinaryOperator::InplaceXor => "__ixor__",
        _ => return None,
    })
}

pub fn binary_value(
    a: PyObjectRef,
    b: PyObjectRef,
    op: BinaryOperator,
) -> Result<PyObjectRef, PyError> {
    // descroperation.py:825 `inplace_impl` — consult the in-place
    // special first; fall through to the binary op below when absent or
    // `NotImplemented`.
    if let Some(idunder) = inplace_dunder_name(op) {
        // `seq_bug_compat` applies only to `+=` / `*=`; pass the reflected
        // name so the builtin-sequence rhs-first branch can fire.
        let (rdunder, seq_bug_compat) = match op {
            BinaryOperator::InplaceAdd => (Some("__radd__"), true),
            BinaryOperator::InplaceMultiply => (Some("__rmul__"), true),
            _ => (None, false),
        };
        if let Some(result) = crate::objspace::descroperation::try_inplace_special(
            a,
            b,
            idunder,
            rdunder,
            seq_bug_compat,
        )? {
            return Ok(result);
        }
    }
    match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => add(a, b),
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => sub(a, b),
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => mul(a, b),
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => floordiv(a, b),
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => mod_(a, b),
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => truediv(a, b),
        BinaryOperator::Power | BinaryOperator::InplacePower => pow(a, b),
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => lshift(a, b),
        BinaryOperator::MatrixMultiply | BinaryOperator::InplaceMatrixMultiply => matmul(a, b),
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => rshift(a, b),
        BinaryOperator::And | BinaryOperator::InplaceAnd => and_(a, b),
        // mappingproxy `__ior__` (read-only) raises TypeError, handled
        // above by `try_inplace_special`; both fall through to `or_`.
        BinaryOperator::Or | BinaryOperator::InplaceOr => or_(a, b),
        BinaryOperator::Xor | BinaryOperator::InplaceXor => xor(a, b),
        BinaryOperator::Subscr => getitem(a, b),
    }
}

pub fn binary_value_from_tag(
    a: PyObjectRef,
    b: PyObjectRef,
    op_tag: i64,
) -> Result<PyObjectRef, PyError> {
    // In-place tags (13-24) must consult the in-place special (`__iadd__`
    // etc.) first; route them through `binary_value`.  Tags 0-12 use the
    // plain dispatch below.
    if op_tag > 12 {
        let Some(op) = crate::runtime_ops::binary_op_from_tag(op_tag) else {
            return Err(PyError::type_error(format!(
                "unsupported binary op tag: {op_tag}"
            )));
        };
        return binary_value(a, b, op);
    }
    match op_tag {
        0 => add(a, b),
        1 => sub(a, b),
        2 => mul(a, b),
        3 => floordiv(a, b),
        4 => mod_(a, b),
        5 => truediv(a, b),
        6 => getitem(a, b),
        7 => pow(a, b),
        8 => lshift(a, b),
        9 => rshift(a, b),
        10 => and_(a, b),
        11 => or_(a, b),
        12 => xor(a, b),
        _ => Err(PyError::type_error(format!(
            "unsupported binary op tag: {op_tag}"
        ))),
    }
}

pub fn compare_value(
    a: PyObjectRef,
    b: PyObjectRef,
    op: ComparisonOperator,
) -> Result<PyObjectRef, PyError> {
    let cmp_op = match op {
        ComparisonOperator::Less => CompareOp::Lt,
        ComparisonOperator::LessOrEqual => CompareOp::Le,
        ComparisonOperator::Greater => CompareOp::Gt,
        ComparisonOperator::GreaterOrEqual => CompareOp::Ge,
        ComparisonOperator::Equal => CompareOp::Eq,
        ComparisonOperator::NotEqual => CompareOp::Ne,
    };
    compare(a, b, cmp_op)
}

pub fn compare_value_from_tag(
    a: PyObjectRef,
    b: PyObjectRef,
    op_tag: i64,
) -> Result<PyObjectRef, PyError> {
    // CONTAINS_OP routes through the compare-residual machinery: tag 6 =
    // `in`, tag 7 = `not in`. `a` is the needle, `b` the container (flatten
    // lowers the args as `[item, container]`).
    if op_tag == 6 || op_tag == 7 {
        let found = crate::baseobjspace::contains(b, a)?;
        let result = if op_tag == 7 { !found } else { found };
        return Ok(w_bool_from(result));
    }
    let op = match op_tag {
        0 => CompareOp::Lt,
        1 => CompareOp::Le,
        2 => CompareOp::Gt,
        3 => CompareOp::Ge,
        4 => CompareOp::Eq,
        5 => CompareOp::Ne,
        _ => {
            return Err(PyError::type_error(format!(
                "unsupported compare op tag: {op_tag}"
            )));
        }
    };
    compare(a, b, op)
}

pub fn unary_negative_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    neg(value)
}

pub fn unary_invert_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    invert(value)
}

pub fn unary_positive_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    crate::baseobjspace::pos(value)
}

/// CALL_INTRINSIC_1 ListToTuple — convert a list to a tuple (star
/// unpacking).  Shared by the interpreter's `list_to_tuple` and the JIT
/// residual `bh_list_to_tuple_fn`.  Allocates a fresh tuple.
///
/// pyopcode.py does `space.call_function(space.w_tuple, w_l)` (accepts any
/// iterable); the compiler only ever emits INTRINSIC_LIST_TO_TUPLE right
/// after a `BUILD_LIST` + `LIST_EXTEND` chain, so the operand is always an
/// exact list.  A non-iterable `*arg` is already rejected upstream by
/// LIST_EXTEND ("argument after * must be an iterable"), so the non-list
/// TypeError below is an unreachable defensive guard.
pub fn list_to_tuple_value(value: PyObjectRef) -> Result<PyObjectRef, PyError> {
    unsafe {
        if pyre_object::is_list(value) {
            let items = pyre_object::w_list_items_copy_as_vec(value);
            return Ok(pyre_object::w_tuple_new(items));
        }
    }
    Err(PyError::type_error("expected list for list_to_tuple"))
}
pub fn truth_value(value: PyObjectRef) -> Result<bool, PyError> {
    is_true(value)
}

pub fn bool_value_from_truth(value: bool) -> PyObjectRef {
    w_bool_from(value)
}

/// LOAD_COMMON_CONSTANT — resolve a `CommonConstant` to the object the
/// interpreter pushes.  Shared by the interpreter's `load_common_constant`
/// handler and the JIT residual `bh_load_common_constant_fn` so both
/// resolve identical objects (immortal type/exception classes for the
/// class variants; a freshly built builtin function for `all`/`any`).
pub fn load_common_constant_value(cc: crate::bytecode::CommonConstant) -> PyObjectRef {
    use crate::bytecode::CommonConstant;
    match cc {
        CommonConstant::AssertionError => crate::builtins::lookup_exc_class("AssertionError")
            .unwrap_or_else(|| {
                crate::typedef::gettypeobject(
                    &pyre_object::interp_exceptions::EXC_ASSERTION_ERROR_TYPE,
                )
            }),
        CommonConstant::NotImplementedError => {
            crate::builtins::lookup_exc_class("NotImplementedError").unwrap_or_else(|| {
                crate::make_builtin_function("NotImplementedError", |_args| {
                    Err(crate::PyError::type_error("not implemented"))
                })
            })
        }
        CommonConstant::BuiltinTuple => {
            crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE)
        }
        CommonConstant::BuiltinAll => crate::make_module_builtin_function_with_arity(
            "all",
            crate::builtins::builtin_all_fn,
            1,
        ),
        CommonConstant::BuiltinAny => crate::make_module_builtin_function_with_arity(
            "any",
            crate::builtins::builtin_any_fn,
            1,
        ),
        CommonConstant::BuiltinList => {
            crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
        }
        CommonConstant::BuiltinSet => {
            crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE)
        }
    }
}

/// LIST_EXTEND — extend `list` in place with the items of `iterable`.
/// Shared by the interpreter's `list_extend` handler and the JIT residual
/// `bh_list_extend_fn`.  Mirrors `list.extend`: fast paths for list/tuple
/// sources, generic iterator-protocol fallback otherwise (which surfaces
/// "Value after * must be an iterable, not <T>" when not iterable).
pub fn list_extend_value(list: PyObjectRef, iterable: PyObjectRef) -> Result<(), PyError> {
    // pyopcode.py LIST_EXTEND calls the target list's `extend` method.  The
    // builtin body preserves exact-list/tuple storage fast paths while list
    // and tuple subclasses go through `iter()` so an overridden `__iter__`
    // is observed.
    match crate::type_methods::list_method_extend(&[list, iterable]) {
        Ok(_) => Ok(()),
        Err(error) if crate::baseobjspace::is_iterable(iterable) => Err(error),
        Err(_) => unsafe {
            let type_name = pyre_object::type_name_of(iterable);
            Err(PyError::type_error(format!(
                "Value after * must be an iterable, not {type_name}"
            )))
        },
    }
}

/// SET_ADD — `set.add(value)` (or `list.append` for the list-shaped
/// accumulator).  Shared by the interpreter's `set_add` and the JIT
/// residual `bh_set_add_fn`.  `set` is peeked, mutated in place.
///
/// pyopcode.py uses `space.call_method(w_set, 'add', ...)`; this stores
/// directly by container type instead.  These accumulators only ever touch
/// the container the surrounding BUILD_SET / BUILD_MAP / BUILD_LIST just
/// pushed — a set/dict/list comprehension or display has no syntax to
/// target a user subclass — so the container is always the exact builtin
/// type and the method-dispatch vs direct-store distinction is not
/// observable (SET_UPDATE / DICT_UPDATE below take the same shortcut; the
/// *source* iterable/mapping, which can be arbitrary, still goes through
/// the iterator / mapping protocol).
pub fn set_add_value(set: PyObjectRef, value: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_set_or_frozenset(set) {
            // `try_hash_value` may run a user `__hash__` that allocates and
            // triggers a moving minor collection; `set`/`value` must be
            // shadow-stack roots across it, then reloaded, or the store
            // below can write through a relocated (stale) pointer.  Its
            // digest keys the store, so the element is hashed once.
            let _roots = pyre_object::gc_roots::push_roots();
            let sp = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(set);
            pyre_object::gc_roots::pin_root(value);
            let hash = crate::builtins::try_hash_value(value).map_err(|err| {
                crate::baseobjspace::wrap_set_element_hash_error(
                    pyre_object::gc_roots::shadow_stack_get(sp + 1),
                    err,
                )
            })?;
            let set = pyre_object::gc_roots::shadow_stack_get(sp);
            let value = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            pyre_object::w_set_add_hashed_checked(set, value, hash)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
        } else if pyre_object::is_list(set) {
            pyre_object::w_list_append(set, value);
        }
    }
    Ok(())
}

/// SET_UPDATE — `set.update(iterable)` (or `list.extend` for the
/// list-shaped accumulator).  Shared by the interpreter's `set_update`
/// and the JIT residual `bh_set_update_fn`.  `set` is peeked, mutated in
/// place; a user iterator or element `__hash__` may run Python.
pub fn set_update_value(set: PyObjectRef, iterable: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_set_or_frozenset(set) {
            // `collect_iterable` pops its own shadow-stack roots before
            // returning, so `items` is a bare, unrooted `Vec` — each
            // element (and `set`) needs re-pinning for the duration of
            // this loop, since any iteration's `try_hash_value` can move
            // both `set` and every not-yet-processed item.
            let items = crate::builtins::collect_iterable(iterable)?;
            let _roots = pyre_object::gc_roots::push_roots();
            let sp = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(set);
            let item_base = sp + 1;
            for item in items {
                pyre_object::gc_roots::pin_root(item);
            }
            let item_len = pyre_object::gc_roots::shadow_stack_len() - item_base;
            for i in 0..item_len {
                let item = pyre_object::gc_roots::shadow_stack_get(item_base + i);
                let hash = crate::builtins::try_hash_value(item).map_err(|err| {
                    crate::baseobjspace::wrap_set_element_hash_error(
                        pyre_object::gc_roots::shadow_stack_get(item_base + i),
                        err,
                    )
                })?;
                let set = pyre_object::gc_roots::shadow_stack_get(sp);
                let item = pyre_object::gc_roots::shadow_stack_get(item_base + i);
                pyre_object::w_set_add_hashed_checked(set, item, hash)
                    .map_err(|_| crate::baseobjspace::take_pending_hash_error())?;
            }
        } else if pyre_object::is_list(set) {
            if pyre_object::is_list(iterable) {
                let items = pyre_object::w_list_items_copy_as_vec(iterable);
                for item in items {
                    pyre_object::w_list_append(set, item);
                }
            } else if pyre_object::is_tuple(iterable) {
                for item in pyre_object::w_tuple_items_copy_as_vec(iterable) {
                    pyre_object::w_list_append(set, item);
                }
            }
        }
    }
    Ok(())
}

/// MAP_ADD — `dict[key] = value`.  Shared by the interpreter's `map_add`
/// and the JIT residual `bh_map_add_fn`.  `dict` is peeked, mutated in
/// place; the key is hashed before the raw dict store, so a user `__hash__`
/// may run Python.
pub fn map_add_value(
    dict: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), PyError> {
    unsafe {
        // Root across `try_hash_value` (a possible collection point from a
        // user `__hash__`) and reload before the store, mirroring
        // `set_add_value`.
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(dict);
        pyre_object::gc_roots::pin_root(key);
        pyre_object::gc_roots::pin_root(value);
        let hash = crate::builtins::try_hash_value(key)
            .map_err(|err| crate::baseobjspace::wrap_dict_key_hash_error(key, err))?;
        let dict = pyre_object::gc_roots::shadow_stack_get(sp);
        let key = pyre_object::gc_roots::shadow_stack_get(sp + 1);
        let value = pyre_object::gc_roots::shadow_stack_get(sp + 2);
        pyre_object::w_dict_store_hashed_checked(dict, key, value, hash)
            .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(key))?;
    }
    Ok(())
}

/// DICT_UPDATE — `dict.update(source)` with the `ismapping` gate.  Shared
/// by the interpreter's `dict_update` and the JIT residual
/// `bh_dict_update_fn`.  Non-mapping surfaces "'<T>' object is not a
/// mapping"; a `keys()`/`__getitem__` may run Python.
pub fn dict_update_value(dict: PyObjectRef, source: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if pyre_object::is_dict(source) {
            for (k, v) in pyre_object::w_dict_items(source) {
                pyre_object::w_dict_store(dict, k, v);
            }
            return Ok(());
        }
    }
    let keys_method = match crate::baseobjspace::getattr_str(source, "keys") {
        Ok(m) => m,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            let type_name = unsafe { pyre_object::type_name_of(source) };
            return Err(PyError::type_error(format!(
                "'{type_name}' object is not a mapping"
            )));
        }
        Err(e) => return Err(e),
    };
    let keys_obj = crate::call::call_function_impl_result(keys_method, &[])?;
    let keys = crate::builtins::collect_iterable(keys_obj)?;
    unsafe {
        // A generic mapping's keys are not pre-validated the way a real
        // dict's are (the `is_dict(source)` fast path above skips the hash
        // check for exactly that reason) — `d[key] = mapping[key]` still
        // hashes `key`, so an unhashable one must raise here too.  `dict`,
        // `source`, and every collected key are rooted for the loop's
        // duration: `getitem` (arbitrary `__getitem__`) and
        // `try_hash_value` (arbitrary `__hash__`) both may allocate and
        // move them.
        let _roots = pyre_object::gc_roots::push_roots();
        let sp = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(dict);
        pyre_object::gc_roots::pin_root(source);
        let key_base = sp + 2;
        for key in keys {
            pyre_object::gc_roots::pin_root(key);
        }
        let key_len = pyre_object::gc_roots::shadow_stack_len() - key_base;
        for i in 0..key_len {
            let source = pyre_object::gc_roots::shadow_stack_get(sp + 1);
            let key = pyre_object::gc_roots::shadow_stack_get(key_base + i);
            let val = crate::baseobjspace::getitem(source, key)?;
            let _val_root = pyre_object::gc_roots::push_roots();
            let val_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(val);
            let key = pyre_object::gc_roots::shadow_stack_get(key_base + i);
            let hash = crate::builtins::try_hash_value(key)
                .map_err(|err| crate::baseobjspace::wrap_dict_key_hash_error(key, err))?;
            let dict = pyre_object::gc_roots::shadow_stack_get(sp);
            let key = pyre_object::gc_roots::shadow_stack_get(key_base + i);
            let val = pyre_object::gc_roots::shadow_stack_get(val_slot);
            pyre_object::w_dict_store_hashed_checked(dict, key, val, hash)
                .map_err(|_| crate::baseobjspace::take_pending_dict_key_error(key))?;
        }
    }
    Ok(())
}

/// DICT_MERGE — merge `source` into `dict` with duplicate-key checks.
/// Shared by the interpreter's `dict_merge` and the JIT residual
/// `bh_dict_merge_fn`.  `w_callable` is the peeked callable used only for
/// error-message prefixes; a `keys()`/`__getitem__` may run Python.
pub fn dict_merge_value(
    dict: PyObjectRef,
    source: PyObjectRef,
    w_callable: PyObjectRef,
) -> Result<(), PyError> {
    // pyopcode.py:1979 `_dict_merge`.
    let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
    // `space.isinstance_w(w_dict, space.w_dict)` accepts dict subclasses;
    // a non-dict target is a RuntimeError, not a TypeError.
    if !unsafe { crate::baseobjspace::isinstance_w(dict, w_dict_type) } {
        let type_name = unsafe { (*(*dict).ob_type).name };
        return Err(PyError::new(
            crate::PyErrorKind::RuntimeError,
            format!("expected a dict, got {type_name}"),
        ));
    }
    // `space.len_w` is a generic `__len__` dispatch — a raw `w_dict_len`
    // would be UB on a dict subclass whose layout is not `W_DictObject`.
    let l1 = crate::baseobjspace::len_w(dict)?;
    let source_is_dict = unsafe { crate::baseobjspace::isinstance_w(source, w_dict_type) };
    if !source_is_dict {
        // `if not space.ismapping_w(w_item): raise oefmt(... "%s argument
        // after ** must be a mapping, not %T")`.
        if !crate::baseobjspace::ismapping_w(source) {
            let type_name = unsafe { pyre_object::type_name_of(source) };
            return Err(crate::argument::raise_type_error(
                w_callable,
                format!("argument after ** must be a mapping, not {type_name}"),
            ));
        }
    } else {
        // Dict source fast paths: an empty target merges without the
        // duplicate check (`update1`); an empty source is a no-op.  The raw
        // items walk is exact-dict only — a dict subclass may override
        // `keys()` / `__getitem__`, so it falls through to the generic loop.
        let l2 = crate::baseobjspace::len_w(source)?;
        if l1 == 0 && unsafe { pyre_object::is_dict(source) } {
            unsafe {
                for (k, v) in pyre_object::w_dict_items(source) {
                    pyre_object::w_dict_store(dict, k, v);
                }
            }
            return Ok(());
        }
        if l2 == 0 {
            return Ok(());
        }
    }
    // `_dict_merge_loop`: iterate `iter(w_item.keys())`, look each value up
    // with `space.getitem`, reject a key already present in the target with
    // `"%s got multiple values for keyword argument '%S'"`, then store.
    let keys_method = match crate::baseobjspace::getattr_str(source, "keys") {
        Ok(m) => m,
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
            let type_name = unsafe { (*(*source).ob_type).name };
            return Err(crate::argument::raise_type_error(
                w_callable,
                format!("argument after ** must be a mapping, not {type_name}"),
            ));
        }
        Err(e) => return Err(e),
    };
    let keys_obj = crate::call::call_function_impl_result(keys_method, &[])?;
    let keys = crate::builtins::collect_iterable(keys_obj)?;
    for key in keys {
        let val = crate::baseobjspace::getitem(source, key)?;
        if crate::baseobjspace::contains(dict, key)? {
            let key_str = unsafe { crate::display::py_str(key) }?;
            return Err(crate::argument::raise_type_error(
                w_callable,
                format!("got multiple values for keyword argument '{key_str}'"),
            ));
        }
        unsafe { pyre_object::w_dict_store(dict, key, val) };
    }
    Ok(())
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_truth_value(value: i64) -> i64 {
    match truth_value(value as PyObjectRef) {
        Ok(truth) => truth as i64,
        Err(mut err) => {
            // A raising `__bool__` / `__len__` publishes into the backend
            // exception cells so the trailing GuardNoException deopts and
            // re-raises through the blackhole (llmodel.py:194-199
            // _store_exception).  Return 0 — the guard fires before the
            // truth is consumed.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

// `space.newbool` selects the `w_True` / `w_False` singleton — it neither
// forces a virtualizable nor raises (EF_CANNOT_RAISE), unlike the may-force
// helpers.  Not elidable: the trace consumes the boxed bool as a recorded
// OpRef, so it must not fold to an inline Const.
#[majit_macros::dont_look_inside_cannot_raise]
pub extern "C" fn jit_bool_value_from_truth(value: i64) -> i64 {
    bool_value_from_truth(value != 0) as i64
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_binary_value_from_tag(a: i64, b: i64, op_tag: i64) -> i64 {
    match binary_value_from_tag(a as PyObjectRef, b as PyObjectRef, op_tag) {
        Ok(value) => value as i64,
        Err(mut err) => {
            // llmodel.py:194-199 _store_exception: publish into the backend
            // exception cells so the trailing GuardNoException deopts and
            // re-raises through the blackhole.  Return null — the guard fires
            // before the result is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_compare_value_from_tag(a: i64, b: i64, op_tag: i64) -> i64 {
    match compare_value_from_tag(a as PyObjectRef, b as PyObjectRef, op_tag) {
        Ok(value) => value as i64,
        Err(mut err) => {
            // Publish + null so the trailing GuardNoException deopts and
            // re-raises (llmodel.py:194-199 _store_exception).
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_unary_negative_value(value: i64) -> i64 {
    match unary_negative_value(value as PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            // Publish + null so the trailing GuardNoException deopts and
            // re-raises (llmodel.py:194-199 _store_exception).
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_unary_invert_value(value: i64) -> i64 {
    match unary_invert_value(value as PyObjectRef) {
        Ok(result) => result as i64,
        Err(mut err) => {
            // Publish + null so the trailing GuardNoException deopts and
            // re-raises (llmodel.py:194-199 _store_exception).
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_getitem(obj: i64, index: i64) -> i64 {
    match getitem(obj as PyObjectRef, index as PyObjectRef) {
        Ok(value) => value as i64,
        Err(mut err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after BINARY_SUBSCR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.  Return null —
            // the guard fires before the result ref is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_setitem(obj: i64, index: i64, value: i64) {
    match crate::setitem(
        obj as PyObjectRef,
        index as PyObjectRef,
        value as PyObjectRef,
    ) {
        // STORE_SUBSCR drops `space.setitem`'s result; this void shim does
        // the same so the recorded residual is a void `CALL_N`.
        Ok(_) => {}
        Err(mut err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after STORE_SUBSCR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_getattr(obj: i64, name_ptr: i64, name_len: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    let name = std::str::from_utf8(bytes).expect("invalid attr name in JIT");
    match crate::getattr_str(obj as PyObjectRef, name) {
        Ok(value) => value as i64,
        Err(mut err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after LOAD_ATTR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.  Return null —
            // the guard fires before the result ref is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_setattr(obj: i64, name_ptr: i64, name_len: i64, value: i64) -> i64 {
    let bytes = unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    let name = std::str::from_utf8(bytes).expect("invalid attr name in JIT");
    match crate::setattr_str(obj as PyObjectRef, name, value as PyObjectRef) {
        Ok(_) => 0,
        Err(mut err) => {
            // llmodel.py:194-199 _store_exception: publish the exception into
            // the backend pos_exception cells so the GuardNoException recorded
            // after STORE_ATTR (instruction_may_raise) deopts and re-raises
            // through the blackhole resume instead of crashing.  Return garbage
            // — the guard fires before the result is used.
            crate::runtime_ops::jit_publish_exception(err.to_exc_object());
            0
        }
    }
}

/// C-ABI bridge for the `execute_store_subscr` arm helper consumed by the
/// production walker.  Mirrors RPython's `bh_call_*` calling convention:
/// a single `*mut PyFrame` arg widened to `i64`, success encoded as a
/// non-zero `i64`, errors propagated via
/// `majit_metainterp::blackhole::BH_LAST_EXC_VALUE`.  Required because
/// `crate::execute_store_subscr` itself returns `Result<StepResult<_>,
/// PyError>` whose fat-enum payload does not fit the residual_call's
/// single-register Ref-result slot.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_execute_store_subscr(executor_ptr: i64) -> i64 {
    let executor = unsafe { &mut *(executor_ptr as *mut crate::pyframe::PyFrame) };
    match crate::pyopcode::execute_store_subscr(executor) {
        Ok(_step_result) => 1,
        Err(mut err) => {
            let exc_obj = err.to_exc_object();
            majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
            0
        }
    }
}

/// C-ABI 3-arg `(obj, key, value) → i64` store_subscr helper bound by
/// `pyre-jit::cpu.store_subscr_fn` (`pyre-jit/src/jit/cpu.rs:151`).
/// The codewriter emits a `residual_call_r_v(store_subscr_fn, obj,
/// key, value)` (`codewriter.rs:7042
/// build_store_subscr_fn_residual_call_r_v_insn`); the runtime
/// dispatcher calls this thin wrapper to mutate the heap via
/// `baseobjspace::setitem` — `baseobjspace.py` parity for
/// `ObjSpace.setitem(w_obj, w_key, w_value) → space.descr_setitem(...)`.
///
/// Lives in `pyre-interpreter` so `pyre-jit-trace` can reach the address
/// through `pyre_interpreter::jit_trace_fnaddrs()` without adding a
/// `pyre-jit-trace -> pyre-jit` dependency edge.  `pyre-jit-trace`
/// already depends on pyre-interpreter for the normal recording-time
/// helpers (`jit_setitem`, `jit_getitem`, ...).
///
/// `setitem` may enter a user `__setitem__` (MayForce), so a raise is
/// published into BOTH the backend `_store_exception` cells (via
/// `jit_publish_exception`, so a compiled trace's trailing
/// `GuardNoException` side-exits) AND `BH_LAST_EXC_VALUE` (so the
/// blackhole / full-body walk sees a non-null standing exception). This
/// dual-publish mirrors `bh_store_attr_fn`
/// (`call_jit::publish_residual_call_exception`) and `jit_next`; writing
/// only `BH_LAST_EXC_VALUE` would leave `GuardNoException` reading a
/// stale 0 and silently swallow the raise. The 1/0 return signals
/// raise-vs-success to the walker's residual executor. The walker drains
/// the backend cells after an Err (`jitcode_dispatch.rs` execute-raised
/// arm), so the extra publish does not leak into tracing.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_store_subscr_fn(obj: i64, key: i64, value: i64) -> i64 {
    let obj = obj as pyre_object::PyObjectRef;
    let key = key as pyre_object::PyObjectRef;
    let value = value as pyre_object::PyObjectRef;
    if let Err(mut err) = crate::baseobjspace::setitem(obj, key, value) {
        let exc_obj = err.to_exc_object();
        majit_metainterp::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(exc_obj as i64));
        crate::runtime_ops::jit_publish_exception(exc_obj);
        return 0;
    }
    1
}

/// C-ABI residual bridge for the `dont_look_inside`
/// [`pyre_object::typeobject::w_type_set_uses_object_setattr`]: its
/// `bool` parameter does not match the integer arg slot a residual call
/// supplies, so normalise it from `i64` here before forwarding.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_w_type_set_uses_object_setattr(obj: i64, v: i64) {
    unsafe {
        pyre_object::typeobject::w_type_set_uses_object_setattr(obj as PyObjectRef, v != 0);
    }
}

/// C-ABI residual bridge for the `dont_look_inside`
/// [`pyre_object::interp_exceptions::lookup_exc_class_for_kind`]: its `ExcKind`
/// parameter does not match the integer arg slot a residual call
/// supplies, so reconstruct it from `i64` here before forwarding. The
/// `PyObjectRef` result rides back as `i64` (null = not registered).
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_lookup_exc_class_for_kind(kind_disc: i64) -> i64 {
    use pyre_object::interp_exceptions::{EXC_KIND_COUNT, ExcKind, lookup_exc_class_for_kind};
    if kind_disc < 0 || kind_disc as usize >= EXC_KIND_COUNT {
        return 0;
    }
    // Safety: bounds-checked above; `ExcKind` is `repr(u8)` with
    // contiguous discriminants `0..EXC_KIND_COUNT`.
    let kind: ExcKind = unsafe { std::mem::transmute(kind_disc as u8) };
    lookup_exc_class_for_kind(kind) as i64
}

/// C-ABI residual bridge for `exc_kind_discriminant`: the caught exception
/// value rides in as a `PyObjectRef`; its `kind` discriminant rides back as `i64`.
#[allow(improper_ctypes_definitions)]
pub extern "C" fn bh_w_exception_get_kind(evalue: pyre_object::PyObjectRef) -> i64 {
    pyre_object::interp_exceptions::exc_kind_discriminant(evalue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_hooks::install_hash_hook;
    use pyre_object::{w_bool_get_value, w_dict_len, w_int_get_value, w_int_new, w_set_len};

    #[test]
    fn test_accumulator_helpers_reject_unhashable_values_without_inserting() {
        install_hash_hook();

        let key = pyre_object::w_list_new(vec![]);
        let dict = pyre_object::w_dict_new();
        let map_err = map_add_value(dict, key, w_int_new(1))
            .expect_err("MAP_ADD should reject an unhashable key");
        assert_eq!(map_err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);

        let value = pyre_object::w_list_new(vec![]);
        let set = pyre_object::w_set_new();
        let set_err =
            set_add_value(set, value).expect_err("SET_ADD should reject an unhashable element");
        assert_eq!(set_err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(unsafe { w_set_len(set) }, 0);
    }

    /// SET_UPDATE (`set.update(iterable)`) rejects an unhashable element
    /// mid-iterable without inserting the elements that precede it, and
    /// the rooting added around the per-item `try_hash_value` call does
    /// not disturb the accept path. Exercises `set_update_value`'s SET
    /// branch loop added by `416dc05`.
    #[test]
    fn test_set_update_value_rejects_unhashable_element_without_inserting() {
        install_hash_hook();

        let set = pyre_object::w_set_new();
        let unhashable = pyre_object::w_list_new(vec![]);
        let iterable = pyre_object::w_list_new(vec![w_int_new(1), w_int_new(2), unhashable]);
        let err = set_update_value(set, iterable)
            .expect_err("SET_UPDATE should reject an unhashable element");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        // Reject-before-mutate is per-element (matches CPython's left-to-right
        // partial-update behavior for `set.update`), so the hashable elements
        // preceding the unhashable one are still inserted.
        assert_eq!(unsafe { w_set_len(set) }, 2);

        let set = pyre_object::w_set_new();
        let iterable = pyre_object::w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        set_update_value(set, iterable).expect("SET_UPDATE should accept hashable elements");
        assert_eq!(unsafe { w_set_len(set) }, 3);
    }

    #[test]
    fn test_binary_value_reuses_objspace_dispatch() {
        let result = binary_value(w_int_new(8), w_int_new(3), BinaryOperator::Subtract)
            .expect("binary dispatch should succeed");
        unsafe {
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_compare_value_reuses_objspace_dispatch() {
        let result = compare_value(w_int_new(2), w_int_new(7), ComparisonOperator::Less)
            .expect("compare dispatch should succeed");
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_truth_and_unary_helpers_share_objspace_semantics() {
        assert!(!truth_value(w_int_new(0)).unwrap());
        let neg = unary_negative_value(w_int_new(4)).expect("unary negate should succeed");
        let inv = unary_invert_value(w_int_new(5)).expect("unary invert should succeed");
        let pos = unary_positive_value(w_int_new(6)).expect("unary positive should succeed");
        unsafe {
            assert_eq!(w_int_get_value(neg), -4);
            assert_eq!(w_int_get_value(inv), !5);
            assert_eq!(w_int_get_value(pos), 6);
        }
    }

    #[test]
    fn test_jit_abi_helpers_share_same_objspace_semantics() {
        assert_eq!(jit_truth_value(w_int_new(0) as i64), 0);
        let neg = jit_unary_negative_value(w_int_new(4) as i64) as PyObjectRef;
        let cmp =
            jit_compare_value_from_tag(w_int_new(2) as i64, w_int_new(7) as i64, 0) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(neg), -4);
            assert!(w_bool_get_value(cmp));
        }
    }

    #[test]
    fn test_jit_getitem_and_setitem_share_objspace_semantics() {
        let list = pyre_object::w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let item = jit_getitem(list as i64, w_int_new(1) as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(item), 4);
        }
        jit_setitem(list as i64, w_int_new(0) as i64, w_int_new(9) as i64);
        let updated = jit_getitem(list as i64, w_int_new(0) as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(updated), 9);
        }
    }
}
