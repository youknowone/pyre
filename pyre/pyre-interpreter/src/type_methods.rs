//! Builtin type method implementations.
//!
//! PyPy equivalents:
//!   pypy/objspace/std/listobject.py  (list methods)
//!   pypy/objspace/std/unicodeobject.py  (str methods)
//!   pypy/objspace/std/dictmultiobject.py  (dict methods)
//!   pypy/objspace/std/tupleobject.py  (tuple methods)
//!
//! Separated from space.rs to avoid bloating the hot-path compilation
//! unit. Method functions are registered into TypeDef at startup.

use malachite_bigint::BigInt;
use num_traits::ToPrimitive;
use pyre_object::*;
// `classify`/`case`/`identifier` back the `str` predicate and `casefold`
// methods with the runtime-independent Unicode tables shared with the
// `unicodedata` module.  Those tables track Unicode 17.0.0, one release ahead
// of Python 3.14's 16.0.0, so predicates read a property on the ~500 code
// points release 17 newly assigns (unassigned under 16.0.0) — the same
// data-version skew already noted in `module/unicodedata`, kept consistent
// across `str` and `unicodedata` here.
use rustpython_unicode::{case, classify, identifier};
use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

// ── Arity checks for builtin methods ─────────────────────────────────
// A builtin method registered with `make_builtin_function_with_arity`
// records an arity only as a fast-path dispatch hint — some stubs register
// a hint that differs from their real signature, so the call machinery
// cannot treat it as a contract. Methods therefore validate their own
// argument count and raise instead of asserting. `args[0]` is the bound
// receiver, so the counts in these messages exclude it.

/// TypeError for a method requiring exactly `n` positional arguments after
/// the receiver, called with a different count.
pub(crate) fn arity_exact(
    args: &[PyObjectRef],
    name: &str,
    n: usize,
) -> Result<(), crate::PyError> {
    if args.len() != n + 1 {
        let expected = match n {
            0 => "no arguments".to_string(),
            1 => "exactly one argument".to_string(),
            k => format!("exactly {k} arguments"),
        };
        return Err(crate::PyError::type_error(format!(
            "{name}() takes {expected} ({} given)",
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// TypeError for a method accepting at least `min` positional arguments
/// after the receiver, called with fewer — the PyArg_UnpackTuple
/// "X expected at least N arguments, got M" form (`str.index`, `dict.get`).
pub(crate) fn arity_at_least(
    args: &[PyObjectRef],
    name: &str,
    min: usize,
) -> Result<(), crate::PyError> {
    if args.len() < min + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at least {min} argument{}, got {}",
            if min == 1 { "" } else { "s" },
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// TypeError for a method requiring at least `min` positional arguments
/// after the receiver, called with fewer — the METH_FASTCALL
/// "X() takes at least N positional arguments (M given)" form
/// (`str.replace`, `bytes.translate`).
pub(crate) fn arity_at_least_positional(
    args: &[PyObjectRef],
    name: &str,
    min: usize,
) -> Result<(), crate::PyError> {
    if args.len() < min + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes at least {min} positional argument{} ({} given)",
            if min == 1 { "" } else { "s" },
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// Validate that a str search method's substring (`args[1]`) is a `str`,
/// else raise `TypeError("{method}() argument 1 must be str, not {type}")`.
/// The search path reads `args[1]` as a `W_UnicodeObject`; a non-str would
/// be dereferenced as one. `args[1]` is guaranteed present by the caller's
/// preceding `arity_at_least(_, _, 1)`.
pub(crate) fn require_str_sub(args: &[PyObjectRef], method: &str) -> Result<(), crate::PyError> {
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 1 must be str, not {}",
            arg_type_name(args[1])
        )));
    }
    Ok(())
}

/// TypeError for a method accepting at most `max` positional arguments after
/// the receiver, called with more — the METH_VARARGS "X expected at most N
/// arguments, got M" form (`list.index`, `dict.pop`).
pub(crate) fn arity_at_most(
    args: &[PyObjectRef],
    name: &str,
    max: usize,
) -> Result<(), crate::PyError> {
    if args.len() > max + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at most {max} argument{}, got {}",
            if max == 1 { "" } else { "s" },
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// TypeError for a method requiring exactly `n` positional arguments after
/// the receiver, called with a different count — the PyArg_UnpackTuple
/// min==max form "X expected N arguments, got M" (`list.insert`,
/// `list.__setitem__`).  Unlike `arity_exact`, the message carries neither a
/// trailing "()" nor the "exactly" wording; "argument" is singular when
/// `n == 1`.
pub(crate) fn arity_exact_unpack(
    args: &[PyObjectRef],
    name: &str,
    n: usize,
) -> Result<(), crate::PyError> {
    if args.len() != n + 1 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected {n} argument{}, got {}",
            if n == 1 { "" } else { "s" },
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// TypeError for a slot wrapper requiring exactly `n` positional arguments
/// after the receiver — the `check_num_args` (typeobject.c) form
/// "expected N argument(s), got M", with no method name.  "argument" is
/// singular when `n == 1`.
pub(crate) fn arity_slot(args: &[PyObjectRef], n: usize) -> Result<(), crate::PyError> {
    if args.len() != n + 1 {
        return Err(crate::PyError::type_error(format!(
            "expected {n} argument{}, got {}",
            if n == 1 { "" } else { "s" },
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// TypeError for a METH_NOARGS method called with positional arguments —
/// the "X() takes no arguments (M given)" form (`list.__reversed__`).
pub(crate) fn arity_no_args(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    if args.len() != 1 {
        return Err(crate::PyError::type_error(format!(
            "{name}() takes no arguments ({} given)",
            args.len().saturating_sub(1),
        )));
    }
    Ok(())
}

/// TypeError for the ternary-power slot (`__pow__` / `__rpow__`), which
/// accepts one or two positional arguments after the receiver — the
/// "expected 1 or 2 arguments, got M" form with no method name.
pub(crate) fn arity_pow(args: &[PyObjectRef]) -> Result<(), crate::PyError> {
    let extra = args.len().saturating_sub(1);
    if !(1..=2).contains(&extra) {
        return Err(crate::PyError::type_error(format!(
            "expected 1 or 2 arguments, got {extra}"
        )));
    }
    Ok(())
}

/// TypeError for an unbound method descriptor invoked with no receiver
/// (`list.append()` with zero arguments) — `args` is empty.
pub(crate) fn require_receiver(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' of object needs an argument"
        )));
    }
    Ok(())
}

/// The receiver check normally supplied by PyPy's
/// `interp2app(W_ListObject.descr_*)` gateway.  CPython 3.14 exposes list's
/// slot wrappers and ordinary methods as two descriptor kinds with distinct
/// public mismatch messages, so callers identify which surface they expose.
pub(crate) fn require_list_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method list.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'list' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_list(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!("descriptor '{name}' for 'list' objects doesn't apply to a '{received}' object")
        } else {
            format!("descriptor '{name}' requires a 'list' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_AbstractTupleObject.descr_*)` gateway.  As with list,
/// CPython 3.14 distinguishes slot-wrapper and method-descriptor mismatch
/// messages on the public surface.
pub(crate) fn require_tuple_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method tuple.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'tuple' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_tuple(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'tuple' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a 'tuple' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_SetObject.descr_*)` gateway.  CPython 3.14 exposes set's
/// slot wrappers and ordinary methods as two descriptor kinds with distinct
/// public mismatch messages.
pub(crate) fn require_set_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method set.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'set' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_set(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!("descriptor '{name}' for 'set' objects doesn't apply to a '{received}' object")
        } else {
            format!("descriptor '{name}' requires a 'set' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_FrozensetObject.descr_*)` gateway.  The inherited
/// `W_BaseSetObject` implementation is shared with set, while the gateway
/// still requires a frozenset receiver.
pub(crate) fn require_frozenset_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method frozenset.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'frozenset' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_frozenset(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'frozenset' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a 'frozenset' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// The receiver check supplied by PyPy's
/// `interp2app(W_SetIterObject.descr_*)` gateway.  Python 3.14 exposes
/// `__iter__`/`__next__` as slot wrappers and the remaining operations as
/// method descriptors.
pub(crate) fn require_set_iterator_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method set_iterator.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of 'set_iterator' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    if !unsafe { pyre_object::is_set_iterator(receiver) } {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for 'set_iterator' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!(
                "descriptor '{name}' requires a 'set_iterator' object but received a '{received}'"
            )
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// Receiver validation supplied by `W_AbstractRangeIterator.typedef`'s
/// gateways. Python 3.14 gives the machine-word and arbitrary-precision
/// implementations distinct public owner names even though PyPy shares the
/// abstract typedef implementation.
pub(crate) fn require_range_iterator_receiver(
    args: &[PyObjectRef],
    name: &str,
    method_descriptor: bool,
    long: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let owner = if long {
        "longrange_iterator"
    } else {
        "range_iterator"
    };
    let Some(&receiver) = args.first() else {
        let message = if method_descriptor {
            format!("unbound method {owner}.{name}() needs an argument")
        } else {
            format!("descriptor '{name}' of '{owner}' object needs an argument")
        };
        return Err(crate::PyError::type_error(message));
    };
    let matches = unsafe {
        if long {
            pyre_object::is_long_range_iter(receiver)
        } else {
            pyre_object::is_range_iter(receiver)
        }
    };
    if !matches {
        let received = crate::baseobjspace::object_functionstr_type_name(receiver);
        let message = if method_descriptor {
            format!(
                "descriptor '{name}' for '{owner}' objects doesn't apply to a '{received}' object"
            )
        } else {
            format!("descriptor '{name}' requires a '{owner}' object but received a '{received}'")
        };
        return Err(crate::PyError::type_error(message));
    }
    Ok(receiver)
}

/// Receiver-only arity for `str` methods that take no arguments (`isspace`,
/// `lower`, …).  Rejects a missing receiver and any extra positional argument,
/// matching `str.{name}() takes no arguments (N given)`.
pub(crate) fn require_no_args(args: &[PyObjectRef], name: &str) -> Result<(), crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{name}' of 'str' object needs an argument"
        )));
    }
    if args.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "str.{name}() takes no arguments ({} given)",
            args.len() - 1
        )));
    }
    Ok(())
}

// ── List methods ─────────────────────────────────────────────────────
// All take self (list) as first arg.

pub fn list_method_append(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "append", true)?;
    arity_exact(args, "list.append", 1)?;
    unsafe { w_list_append(args[0], args[1]) };
    Ok(w_none())
}

pub fn list_method_extend(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "extend", true)?;
    arity_exact(args, "list.extend", 1)?;
    let list = args[0];
    let other = args[1];
    unsafe {
        // listobject.py:1019-1033 only takes the storage-copy path when a
        // list/tuple uses its inherited iterator.  An overridden subclass
        // must use the generic incremental iterator path below.
        if is_exact_list(other) {
            let n = w_list_len(other);
            for i in 0..n {
                if let Some(item) = w_list_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        } else if is_exact_tuple(other) {
            let n = w_tuple_len(other);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        } else {
            // listobject.py:1019 _extend_from_iterable: append each yielded
            // value before asking the iterator for the next one.  In
            // particular, an exception from a later `next()` must not roll
            // back the prefix already appended to the receiver.
            let _roots = pyre_object::gc_roots::push_roots();
            let root_base = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(list);
            pyre_object::gc_roots::pin_root(other);
            let iterator =
                crate::baseobjspace::iter(pyre_object::gc_roots::shadow_stack_get(root_base + 1))?;
            pyre_object::gc_roots::pin_root(iterator);
            loop {
                let item = match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(
                    root_base + 2,
                )) {
                    Ok(item) => item,
                    Err(err) if err.kind == crate::PyErrorKind::StopIteration => break,
                    Err(err) => return Err(err),
                };
                let _item_roots = pyre_object::gc_roots::push_roots();
                let item_slot = pyre_object::gc_roots::shadow_stack_len();
                pyre_object::gc_roots::pin_root(item);
                w_list_append(
                    pyre_object::gc_roots::shadow_stack_get(root_base),
                    pyre_object::gc_roots::shadow_stack_get(item_slot),
                );
            }
        }
    }
    Ok(w_none())
}

/// PyPy: listobject.py descr_insert — list.insert(index, item)
pub fn list_method_insert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "insert", true)?;
    arity_exact_unpack(args, "insert", 2)?;
    // `@unwrap_spec(index='index')` → getindex_w(index, OverflowError): coerce
    // through `__index__`; `get_positive_index` then clamps to `[0, len]`
    // inside `w_list_insert`.
    let index = unsafe { crate::baseobjspace::getindex_w_index(args[1])? };
    unsafe { pyre_object::listobject::w_list_insert(args[0], index, args[2]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_pop — list.pop([index])
/// listobject.py:759-772 — empty list raises "pop from empty list",
/// otherwise out-of-range raises "pop index out of range".
pub fn list_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "pop", true)?;
    // `descr_pop` checks arity before touching the list, so `pop(1, 2)` on an
    // empty list reports the surplus argument rather than "pop from empty list".
    arity_at_most(args, "pop", 1)?;
    // `@unwrap_spec(index='index')` → getindex_w(index, OverflowError): coerce
    // through `__index__`.
    let index = if args.len() > 1 {
        unsafe { crate::baseobjspace::getindex_w_index(args[1])? }
    } else {
        -1
    };
    let length = unsafe { pyre_object::w_list_len(args[0]) } as i64;
    if length == 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "pop from empty list",
        ));
    }
    match unsafe { pyre_object::listobject::w_list_pop(args[0], index) } {
        Some(v) => Ok(v),
        None => Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            "pop index out of range",
        )),
    }
}

/// PyPy: listobject.py descr_clear — list.clear()
pub fn list_method_clear(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "clear", true)?;
    unsafe { pyre_object::listobject::w_list_clear(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_copy — list.copy()
pub fn list_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "copy", true)?;
    let list = args[0];
    unsafe {
        let n = w_list_len(list);
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
        Ok(w_list_new(items))
    }
}

/// PyPy: listobject.py descr_reverse — list.reverse()
pub fn list_method_reverse(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "reverse", true)?;
    unsafe { pyre_object::listobject::w_list_reverse(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_sort — list.sort()
pub fn list_method_sort(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "sort", true)?;
    let list = args[0];
    // Keep the argument decoding shared with `sorted()` before changing the
    // receiver's visible storage.
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "sorted() takes at most 1 positional argument ({} given)",
            positional.len()
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["key", "reverse"], "sorted")?;
    let key_fn = crate::builtins::kwarg_get(kwargs, "key")
        .filter(|key| unsafe { !pyre_object::is_none(*key) });
    let reverse = crate::builtins::kwarg_get(kwargs, "reverse")
        .map(|value| crate::baseobjspace::is_true(value))
        .transpose()?
        .unwrap_or(false);

    unsafe {
        // Hold the detached values in shadow-stack slots, not a bare Rust
        // Vec, while key and comparison calls can collect.  The receiver is
        // empty for the whole operation, so user code cannot alter this
        // sorting slice through the visible list.
        let saved = pyre_object::w_list_items_copy_as_vec(list);
        let _roots = pyre_object::gc_roots::push_roots();
        let item_base = pyre_object::gc_roots::shadow_stack_len();
        for item in saved {
            pyre_object::gc_roots::pin_root(item);
        }
        let saved_len = pyre_object::gc_roots::shadow_stack_len() - item_base;
        pyre_object::listobject::w_list_clear(list);

        let sorted = crate::builtins::sort_rooted_items(item_base, saved_len, key_fn, reverse);
        let modified = w_list_len(list) != 0;

        // Discard any visible mutations and always restore the detached
        // values.  On an error from the key function this restores the input
        // order; after a successful sort it installs the sorted permutation.
        pyre_object::listobject::w_list_clear(list);
        match sorted {
            Ok(order) => {
                for index in order {
                    w_list_append(
                        list,
                        pyre_object::gc_roots::shadow_stack_get(item_base + index),
                    );
                }
                if modified {
                    return Err(crate::PyError::new(
                        crate::PyErrorKind::ValueError,
                        "list modified during sort",
                    ));
                }
            }
            Err(err) => {
                for index in 0..saved_len {
                    w_list_append(
                        list,
                        pyre_object::gc_roots::shadow_stack_get(item_base + index),
                    );
                }
                return Err(err);
            }
        }
    }
    Ok(w_none())
}

/// listobject.py:795 `descr_index` — list.index(value[, start[, stop]]).
pub fn list_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "index", true)?;
    arity_at_least(args, "index", 1)?;
    arity_at_most(args, "index", 3)?;
    let list = args[0];
    let value = args[1];
    // listobject.py:799 unwrap_spec defaults: w_start=0 / w_stop=sys.maxint.
    // listobject.py:803 unwrap_start_stop handles negative normalization,
    // __index__ coercion and TypeError for non-index arguments.
    let size = unsafe { pyre_object::w_list_len(list) } as i64;
    let w_start = if args.len() >= 3 {
        args[2]
    } else {
        w_int_new(0)
    };
    let w_stop = if args.len() >= 4 {
        args[3]
    } else {
        w_int_new(i64::MAX)
    };
    let (start, stop) = crate::sliceobject::unwrap_start_stop(size, w_start, w_stop)?;
    match crate::listobject::w_list_find_or_count(list, value, start, stop, false)? {
        crate::listobject::FindOrCountResult::Index(i) => Ok(w_int_new(i)),
        crate::listobject::FindOrCountResult::NotFound => {
            // listobject.c list_index_impl (3.14): a fixed
            // "list.index(x): x not in list" message that does NOT format the
            // value.  gh-100242 dropped the older "%R is not in list" form
            // (which called `repr` on the missing value); PyPy still uses that
            // older form, so this message is the 3.14 target and cannot be
            // asserted against the PyPy check.py oracle.
            Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                "list.index(x): x not in list".to_string(),
            ))
        }
        crate::listobject::FindOrCountResult::Count(_) => {
            unreachable!("find_or_count with count=false never returns Count")
        }
    }
}

/// listobject.py:744 `descr_count` — list.count(value)
pub fn list_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "count", true)?;
    arity_exact(args, "list.count", 1)?;
    let list = args[0];
    let value = args[1];
    match crate::listobject::w_list_find_or_count(list, value, 0, i64::MAX, true)? {
        crate::listobject::FindOrCountResult::Count(n) => Ok(w_int_new(n)),
        crate::listobject::FindOrCountResult::NotFound => Ok(w_int_new(0)),
        crate::listobject::FindOrCountResult::Index(_) => {
            unreachable!("find_or_count with count=true never returns Index")
        }
    }
}

/// listobject.py:782 `descr_remove` — list.remove(value).
pub fn list_method_remove(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_list_receiver(args, "remove", true)?;
    arity_exact(args, "list.remove", 1)?;
    crate::listobject::w_list_remove(args[0], args[1])?;
    Ok(w_none())
}

// ── String methods ───────────────────────────────────────────────────

pub fn str_method_join(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "join", 1)?;
    let sep = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let iterable = args[1];
    let items: Vec<PyObjectRef> = unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            (0..n)
                .filter_map(|i| w_list_getitem(iterable, i as i64))
                .collect()
        } else if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            (0..n)
                .filter_map(|i| w_tuple_getitem(iterable, i as i64))
                .collect()
        } else {
            crate::builtins::collect_iterable(iterable)?
        }
    };
    // pypy/objspace/std/unicodeobject.py:856-872 descr_join — each
    // element must be a str; otherwise TypeError("sequence item N:
    // expected str instance, <T> found"). Silently dropping non-str
    // items lost the error and produced an empty join.
    //
    // A single-element join returns that element (unicode_result_unchanged):
    // an exact str unchanged, a str subclass copied to a base str.
    if items.len() == 1 {
        let item = items[0];
        if unsafe { !is_str(item) } {
            return Err(crate::PyError::type_error(format!(
                "sequence item 0: expected str instance, {} found",
                arg_type_name(item)
            )));
        }
        return Ok(str_result_unchanged(item));
    }
    let mut out = rustpython_wtf8::Wtf8Buf::new();
    for (i, item) in items.iter().enumerate() {
        if unsafe { !is_str(*item) } {
            return Err(crate::PyError::type_error(format!(
                "sequence item {i}: expected str instance, {} found",
                arg_type_name(*item)
            )));
        }
        if i > 0 {
            out.push_wtf8(sep);
        }
        out.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(*item) });
    }
    Ok(pyre_object::w_str_from_wtf8(out))
}

/// `str.split` / `str.rsplit` take `sep` and `maxsplit` positionally or by
/// keyword.  Builtin kwargs arrive as a trailing `__pyre_kw__` dict, so
/// resolve each argument from its positional slot (after the receiver),
/// falling back to the matching keyword.
fn resolve_split_args(
    args: &[PyObjectRef],
    fn_name: &str,
) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::kwarg_reject_unknown(kwargs, &["sep", "maxsplit"], fn_name)?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "sep", pos.get(1).is_some())?;
    crate::builtins::kwarg_reject_duplicate(kwargs, fn_name, "maxsplit", pos.get(2).is_some())?;
    let sep = pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "sep"))
        .unwrap_or(pyre_object::PY_NULL);
    let maxsplit = pos
        .get(2)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "maxsplit"))
        .unwrap_or(pyre_object::PY_NULL);
    Ok((sep, maxsplit))
}

/// Box a code-point slice into a str object.
fn cps_to_str(cps: &[CodePoint]) -> PyObjectRef {
    let mut buf = Wtf8Buf::with_capacity(cps.len());
    for &cp in cps {
        buf.push(cp);
    }
    w_str_from_wtf8(buf)
}

/// A lone surrogate is not whitespace.
fn cp_is_whitespace(cp: CodePoint) -> bool {
    match cp.to_char() {
        Some(c) => classify::is_space(c),
        None => false,
    }
}

/// `str.split()` with no separator: split on runs of whitespace,
/// dropping leading/trailing runs.  When `maxsplit >= 0`, after that
/// many splits the rest (leading whitespace stripped) is one tail token.
fn wtf8_split_whitespace(s: &Wtf8, maxsplit: i64) -> Vec<PyObjectRef> {
    let cps: Vec<CodePoint> = s.code_points().collect();
    let mut out: Vec<PyObjectRef> = Vec::new();
    let mut i = 0usize;
    loop {
        if maxsplit >= 0 && out.len() as i64 >= maxsplit {
            break;
        }
        while i < cps.len() && cp_is_whitespace(cps[i]) {
            i += 1;
        }
        if i == cps.len() {
            break;
        }
        let start = i;
        while i < cps.len() && !cp_is_whitespace(cps[i]) {
            i += 1;
        }
        out.push(cps_to_str(&cps[start..i]));
    }
    while i < cps.len() && cp_is_whitespace(cps[i]) {
        i += 1;
    }
    if i < cps.len() {
        out.push(cps_to_str(&cps[i..]));
    }
    out
}

/// `str.rsplit()` with no separator: like `wtf8_split_whitespace` but
/// scanning from the right, so the tail token is the leading remainder.
fn wtf8_rsplit_whitespace(s: &Wtf8, maxsplit: i64) -> Vec<PyObjectRef> {
    let cps: Vec<CodePoint> = s.code_points().collect();
    let mut tokens: Vec<PyObjectRef> = Vec::new();
    let mut i = cps.len();
    loop {
        if maxsplit >= 0 && tokens.len() as i64 >= maxsplit {
            break;
        }
        while i > 0 && cp_is_whitespace(cps[i - 1]) {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        let end = i;
        while i > 0 && !cp_is_whitespace(cps[i - 1]) {
            i -= 1;
        }
        tokens.push(cps_to_str(&cps[i..end]));
    }
    tokens.reverse();
    let mut prefix_end = i;
    while prefix_end > 0 && cp_is_whitespace(cps[prefix_end - 1]) {
        prefix_end -= 1;
    }
    if prefix_end > 0 {
        let mut out = vec![cps_to_str(&cps[..prefix_end])];
        out.extend(tokens);
        out
    } else {
        tokens
    }
}

pub fn str_method_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "split")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let (sep_arg, maxsplit_arg) = resolve_split_args(args, "split")?;
    let sep = parse_split_sep(sep_arg)?;
    // `unicodeobject.py:972 @unwrap_spec(maxsplit=int) descr_split` —
    // `space.int_w(w_maxsplit)` routes through `__index__`, so any
    // int-like object (subclass, numpy int, etc.) is accepted.
    let maxsplit = parse_split_maxsplit(maxsplit_arg)?;
    let parts: Vec<PyObjectRef> = match sep.as_deref() {
        Some(sep) => {
            // `unicodeobject.py:1028 _split_with_separator` raises
            // ValueError on empty separator before the slow path.
            if sep.as_bytes().is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            if maxsplit < 0 {
                s.split(sep)
                    .map(|p| w_str_from_wtf8(p.to_wtf8_buf()))
                    .collect()
            } else {
                s.splitn((maxsplit as usize) + 1, sep)
                    .map(|p| w_str_from_wtf8(p.to_wtf8_buf()))
                    .collect()
            }
        }
        None => wtf8_split_whitespace(s, maxsplit),
    };
    Ok(w_list_new(parts))
}

/// `pypy/objspace/std/unicodeobject.py:992-994 W_UnicodeObject
/// .convert_arg_to_w_unicode` parity — `sep` must be `None` or a
/// `str`; anything else surfaces a TypeError at the same call site
/// where PyPy's `space.unicode_w` would.
fn parse_split_sep(value: PyObjectRef) -> Result<Option<Wtf8Buf>, crate::PyError> {
    if value.is_null() || unsafe { is_none(value) } {
        return Ok(None);
    }
    if unsafe { is_str(value) } {
        return Ok(Some(unsafe { w_str_get_wtf8(value) }.to_wtf8_buf()));
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(value) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    };
    Err(crate::PyError::type_error(format!(
        "must be str or None, not {tp_name}"
    )))
}

/// `unicodeobject.py:972 @unwrap_spec(maxsplit=int)` parity —
/// `space.int_w(w_maxsplit)` routes through `__index__`, so any
/// int-like object is accepted; an absent maxsplit defaults to -1
/// (unlimited).  An explicit `None` is not int-like and has no
/// `__index__`, so it raises `TypeError` like any other non-integer.
fn parse_split_maxsplit(value: PyObjectRef) -> Result<i64, crate::PyError> {
    if value.is_null() {
        return Ok(-1);
    }
    crate::builtins::space_index_w(value)
}

/// `pypy/objspace/std/unicodeobject.py:993-1024 W_UnicodeObject
/// .descr_rsplit`.  Mirrors `split` semantics in reverse — when
/// `maxsplit` is positive, only the rightmost `maxsplit` separators
/// participate.  Argument validation follows the same
/// `@unwrap_spec(maxsplit=int)` + `convert_arg_to_w_unicode` shape
/// as `descr_split`.
pub fn str_method_rsplit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "rsplit")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let (sep_arg, maxsplit_arg) = resolve_split_args(args, "rsplit")?;
    let sep = parse_split_sep(sep_arg)?;
    let maxsplit = parse_split_maxsplit(maxsplit_arg)?;
    let parts: Vec<PyObjectRef> = match sep.as_deref() {
        Some(sep) => {
            // `unicodeobject.py:1028 _split_with_separator` raises
            // ValueError on empty separator before the slow path —
            // mirrors the forward `split` rejection.
            if sep.as_bytes().is_empty() {
                return Err(crate::PyError::value_error("empty separator"));
            }
            let mut out: Vec<&Wtf8> = if maxsplit < 0 {
                s.rsplit(sep).collect()
            } else {
                s.rsplitn((maxsplit as usize) + 1, sep).collect()
            };
            out.reverse();
            out.into_iter()
                .map(|p| w_str_from_wtf8(p.to_wtf8_buf()))
                .collect()
        }
        None => wtf8_rsplit_whitespace(s, maxsplit),
    };
    Ok(w_list_new(parts))
}

/// `pypy/objspace/std/unicodeobject.py:767-770 W_UnicodeObject.descr_casefold`:
///
/// ```python
/// def descr_casefold(self, space):
///     value = self._value
///     return space.newutf8(unicode_casefold(value), -1)
/// ```
///
/// PyPy delegates to `rpython.rlib.runicode.unicode_casefold` which
/// applies the full Unicode `CaseFolding.txt` mapping (status C +
/// status F: ß → ss, ﬁ → fi, İ → i + combining dot,
/// Lithuanian Į → i̇ǫ, the Greek sigma, etc.).  Pyre routes through
/// `case::casefold_wtf8`, which applies the same
/// `CaseFolding.txt` status-C+F mapping to each scalar code point and
/// passes lone surrogates through unchanged.
pub fn str_method_casefold(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "casefold")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_str_from_wtf8(case::casefold_wtf8(s)))
}

/// `pypy/objspace/std/unicodeobject.py:429-430 W_UnicodeObject
/// .descr_format_map` parity:
///
/// ```python
/// def descr_format_map(self, space, w_mapping):
///     return newformat.format_method(space, self, None, w_mapping, True)
/// ```
///
/// PyPy passes the mapping straight through to `format_method`; each
/// `{name}` field is then resolved by `space.getitem(mapping, w_key)`
/// at format-time, so the mapping is consulted *lazily*.  This
/// matters for mappings with side-effecting `__getitem__`, a
/// `__missing__` hook, or no `keys()` — pre-materialising via
/// `keys()` (the previous implementation) breaks `defaultdict`,
/// custom `Mapping` subclasses, and any object that only implements
/// `__getitem__`.
pub fn str_method_format_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "str.format_map", 1)?;
    let fmt = args[0];
    let mapping = args[1];
    str_method_format_core(fmt, &[], None, Some(mapping))
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject._strip` —
/// `s.strip([chars])`.  When `chars` is missing or None, defaults to
/// the same Unicode whitespace predicate as `str.isspace()`.  When provided, removes
/// any character contained in `chars` from each end (NOT a substring
/// match — `'aabaa'.strip('a') == 'b'`).
fn strip_chars(s: &Wtf8, chars: Option<&Wtf8>, left: bool, right: bool) -> Wtf8Buf {
    let chars_set: Option<Vec<CodePoint>> = chars.map(|c| c.code_points().collect());
    let mut current: &Wtf8 = s;
    if left {
        current = match chars_set.as_ref() {
            Some(set) => current.trim_start_matches(|cp: CodePoint| set.contains(&cp)),
            None => current.trim_start_matches(cp_is_whitespace),
        };
    }
    if right {
        current = match chars_set.as_ref() {
            Some(set) => current.trim_end_matches(|cp: CodePoint| set.contains(&cp)),
            None => current.trim_end_matches(cp_is_whitespace),
        };
    }
    current.to_wtf8_buf()
}

/// `pypy/objspace/std/unicodeobject.py:1464-1473 W_UnicodeObject
/// ._strip` — extract the optional `chars` argument as a `&str`,
/// raising TypeError on non-str non-None arguments rather than
/// silently falling through to the whitespace default.
fn extract_strip_chars(arg: PyObjectRef, fn_name: &str) -> Result<Option<Wtf8Buf>, crate::PyError> {
    if arg.is_null() || unsafe { pyre_object::is_none(arg) } {
        return Ok(None);
    }
    if unsafe { pyre_object::is_str(arg) } {
        return Ok(Some(
            unsafe { pyre_object::w_str_get_wtf8(arg) }.to_wtf8_buf(),
        ));
    }
    Err(crate::PyError::type_error(format!(
        "{fn_name} arg must be None or str"
    )))
}

pub fn str_method_strip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "strip")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let chars = match args.get(1) {
        Some(&a) => extract_strip_chars(a, "strip")?,
        None => None,
    };
    Ok(w_str_from_wtf8(strip_chars(
        s,
        chars.as_deref(),
        true,
        true,
    )))
}

pub fn str_method_lstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "lstrip")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let chars = match args.get(1) {
        Some(&a) => extract_strip_chars(a, "lstrip")?,
        None => None,
    };
    Ok(w_str_from_wtf8(strip_chars(
        s,
        chars.as_deref(),
        true,
        false,
    )))
}

pub fn str_method_rstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "rstrip")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let chars = match args.get(1) {
        Some(&a) => extract_strip_chars(a, "rstrip")?,
        None => None,
    };
    Ok(w_str_from_wtf8(strip_chars(
        s,
        chars.as_deref(),
        false,
        true,
    )))
}

/// `unicodeobject.py descr_startswith` — accepts either a single str
/// prefix or a tuple of str prefixes (CPython parity).
/// unicodeobject.py:848 descr_startswith(self, prefix, start=0, end=sys.maxsize)
pub fn str_method_startswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "startswith", 1)?;
    arity_at_most(args, "startswith", 3)?;
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let Some(slice) = str_slice_args(s, args)? else {
        return validate_prefix_arg(args[1], "startswith").map(|()| w_bool_from(false));
    };
    str_prefix_match(slice, args[1], "startswith", true).map(w_bool_from)
}

pub fn str_method_endswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "endswith", 1)?;
    arity_at_most(args, "endswith", 3)?;
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let Some(slice) = str_slice_args(s, args)? else {
        return validate_prefix_arg(args[1], "endswith").map(|()| w_bool_from(false));
    };
    str_prefix_match(slice, args[1], "endswith", false).map(w_bool_from)
}

/// Apply `startswith`/`endswith`'s optional `start`/`end` bounds to `s`,
/// returning the code-point window as WTF-8. `stringmethods.py:23
/// _convert_idx_params` → `unwrap_start_stop`: each bound runs through
/// `adapt_lower_bound(_eval_slice_index(...))`, so a non-index bound raises a
/// TypeError and a bound is coerced via `__index__`.
///
/// `unicodeobject.py:1319 _unwrap_and_compute_idx_params` then converts the
/// two code-point bounds to byte offsets: a `start` past the end becomes
/// `end_index + 1` rather than being clamped, and `end` is only lowered when
/// it is short of the end. `None` signals the resulting window is inverted,
/// for which the match is always `False` — even for an empty needle, which is
/// why `'abc'.startswith('', 5, 10)` and `''.endswith('', 1, 0)` are `False`.
fn str_slice_args<'a>(
    s: &'a Wtf8,
    args: &[pyre_object::PyObjectRef],
) -> Result<Option<&'a Wtf8>, crate::PyError> {
    let char_len = s.code_points().count() as i64;
    // `None` bounds mean "not provided" (start -> 0, end -> len).
    let start = if args.len() >= 3 && !unsafe { pyre_object::is_none(args[2]) } {
        crate::sliceobject::adapt_lower_bound(char_len, args[2])?
    } else {
        0
    };
    let end = if args.len() >= 4 && !unsafe { pyre_object::is_none(args[3]) } {
        crate::sliceobject::adapt_lower_bound(char_len, args[3])?
    } else {
        char_len
    };
    let bytes = s.as_bytes();
    let index_to_byte = |cp: i64| {
        s.code_point_indices()
            .nth(cp as usize)
            .map_or(bytes.len(), |(i, _)| i)
    };
    let mut end_index = bytes.len();
    if end < char_len {
        end_index = index_to_byte(end);
    }
    let mut start_index = 0usize;
    if start > 0 {
        start_index = if start > char_len {
            end_index + 1
        } else {
            index_to_byte(start)
        };
    }
    if start_index > end_index {
        return Ok(None);
    }
    Ok(Some(unsafe {
        Wtf8::from_bytes_unchecked(&bytes[start_index..end_index])
    }))
}

fn str_prefix_match(
    s: &Wtf8,
    needle: PyObjectRef,
    method: &str,
    start: bool,
) -> Result<bool, crate::PyError> {
    let h = s.as_bytes();
    // WTF-8 is self-synchronizing, so a byte-level prefix/suffix match
    // coincides with a code-point-level one.
    let test = |p: &Wtf8| {
        let p = p.as_bytes();
        if start {
            h.starts_with(p)
        } else {
            h.ends_with(p)
        }
    };
    if unsafe { pyre_object::is_str(needle) } {
        let p = unsafe { pyre_object::w_str_get_wtf8(needle) };
        return Ok(test(p));
    }
    if unsafe { pyre_object::is_tuple(needle) } {
        let n = unsafe { pyre_object::w_tuple_len(needle) };
        for i in 0..n as i64 {
            let item =
                unsafe { pyre_object::w_tuple_getitem(needle, i) }.expect("index is in range");
            if !unsafe { pyre_object::is_str(item) } {
                return Err(crate::PyError::type_error(format!(
                    "tuple for {method} must only contain str, not {}",
                    arg_type_name(item)
                )));
            }
            let p = unsafe { pyre_object::w_str_get_wtf8(item) };
            if test(p) {
                return Ok(true);
            }
        }
        return Ok(false);
    }
    Err(crate::PyError::type_error(format!(
        "{method} first arg must be str or a tuple of str, not {}",
        arg_type_name(needle)
    )))
}

/// Type-check a `startswith`/`endswith` argument (a str, or a tuple whose
/// items are all str) without running the match. Used on the out-of-range
/// window path, where the result is `False` but a bad argument type still
/// raises the same `TypeError` as the in-range path.
fn validate_prefix_arg(needle: PyObjectRef, method: &str) -> Result<(), crate::PyError> {
    if unsafe { pyre_object::is_str(needle) } {
        return Ok(());
    }
    if unsafe { pyre_object::is_tuple(needle) } {
        let n = unsafe { pyre_object::w_tuple_len(needle) };
        for i in 0..n as i64 {
            let item =
                unsafe { pyre_object::w_tuple_getitem(needle, i) }.expect("index is in range");
            if !unsafe { pyre_object::is_str(item) } {
                return Err(crate::PyError::type_error(format!(
                    "tuple for {method} must only contain str, not {}",
                    arg_type_name(item)
                )));
            }
        }
        return Ok(());
    }
    Err(crate::PyError::type_error(format!(
        "{method} first arg must be str or a tuple of str, not {}",
        arg_type_name(needle)
    )))
}

pub fn str_method_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `old` / `new` are positional-only; `count` is positional-or-keyword.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() < 3 {
        return Err(crate::PyError::type_error(format!(
            "replace() takes at least 2 positional arguments ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["count"], "replace")?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "replace", "count", pos.get(3).is_some())?;
    // pypy/objspace/std/unicodeobject.py:1132-1148 descr_replace —
    // both `old` and `new` must be str / W_UnicodeObject; otherwise
    // TypeError("replace() argument N must be str, not ...").
    if !unsafe { pyre_object::is_str(pos[1]) } {
        return Err(crate::PyError::type_error(format!(
            "replace() argument 1 must be str, not {}",
            arg_type_name(pos[1])
        )));
    }
    if !unsafe { pyre_object::is_str(pos[2]) } {
        return Err(crate::PyError::type_error(format!(
            "replace() argument 2 must be str, not {}",
            arg_type_name(pos[2])
        )));
    }
    let s = unsafe { pyre_object::w_str_get_wtf8(pos[0]) };
    let old = unsafe { pyre_object::w_str_get_wtf8(pos[1]) };
    let new = unsafe { pyre_object::w_str_get_wtf8(pos[2]) };
    // Optional `count`: a negative count means "no limit"; 0 leaves the
    // string untouched. Resolved through `__index__`.
    let maxcount = match pos
        .get(3)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "count"))
    {
        Some(w_count) => crate::builtins::space_index_w(w_count)?,
        None => -1,
    };
    Ok(w_str_from_wtf8(wtf8_replace(s, old, new, maxcount)))
}

/// WTF-8 window for the optional `start` / `end` search args: resolve them
/// (PyPy slice semantics via `unwrap_start_stop`) into a byte-offset window
/// `(byte_start, byte_end)` into the WTF-8 backing, indexing by code point so
/// a surrogate-bearing string does not panic in `w_str_get_value`.  Returns
/// `None` when the codepoint window is empty because `start` is past the end
/// or past `end` (the search-miss case shared by count).
fn wtf8_idx_window(
    s: &Wtf8,
    args: &[PyObjectRef],
) -> Result<Option<(usize, usize)>, crate::PyError> {
    let cp_len = s.code_points().count() as i64;
    let w_start = if args.len() >= 3 { args[2] } else { w_none() };
    let w_end = if args.len() >= 4 { args[3] } else { w_none() };
    let (start, end) = crate::sliceobject::unwrap_start_stop(cp_len, w_start, w_end)?;
    if start > cp_len {
        return Ok(None);
    }
    let end = end.min(cp_len);
    if start > end {
        return Ok(None);
    }
    let byte_start = s
        .code_point_indices()
        .nth(start as usize)
        .map_or(s.len(), |(i, _)| i);
    let byte_end = s
        .code_point_indices()
        .nth(end as usize)
        .map_or(s.len(), |(i, _)| i);
    Ok(Some((byte_start, byte_end)))
}

pub fn str_method_find(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "find", 1)?;
    arity_at_most(args, "find", 3)?;
    require_str_sub(args, "find")?;
    Ok(w_int_new(str_unwrap_and_search(args, true)?.unwrap_or(-1)))
}

pub fn str_method_rfind(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "rfind", 1)?;
    arity_at_most(args, "rfind", 3)?;
    require_str_sub(args, "rfind")?;
    Ok(w_int_new(str_unwrap_and_search(args, false)?.unwrap_or(-1)))
}

pub fn str_method_upper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "upper")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_str_from_wtf8(wtf8_map_str_runs(s, str::to_uppercase)))
}

pub fn str_method_lower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "lower")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_str_from_wtf8(wtf8_map_str_runs(s, str::to_lowercase)))
}

/// PyPy: unicodeobject.py descr_format
/// Requires format spec parser — correct for no-arg case only.
/// `str.format(*args)` — PyPy: unicodeobject.py descr_format → newformat.py
pub fn str_method_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "format")?;
    // `pypy/objspace/std/newformat.py Formatter.format` —
    // positional args are slots 1.. of the receiver; keyword args
    // (`{name}` lookups) live in the trailing CALL_KW dict.
    let (positional, kwargs_dict) = crate::builtins::split_builtin_kwargs(&args[1..]);
    str_method_format_core(args[0], positional, kwargs_dict, None)
}

/// Shared core for `str.format` (`{name}` looks up the trailing
/// CALL_KW dict) and `str.format_map` (`{name}` looks up the
/// mapping via `space.getitem(mapping, w_key)`).  PyPy folds both
/// into one `newformat.format_method(space, fmt, args_w, w_kwds, ...)`
/// entry point per `unicodeobject.py:422-430`; pyre splits the
/// keyword-source into "dict snapshot" vs "lazy mapping" so the
/// mapping path stays line-by-line lazy (no pre-materialisation).
fn str_method_format_core(
    fmt_obj: PyObjectRef,
    positional: &[PyObjectRef],
    kwargs_dict: Option<PyObjectRef>,
    mapping: Option<PyObjectRef>,
) -> Result<PyObjectRef, crate::PyError> {
    // Read the template as WTF-8 so a lone surrogate in a literal run (or a
    // surrogate arg spliced in) survives instead of panicking.
    let fmt = unsafe { pyre_object::w_str_get_wtf8(fmt_obj) };
    let mut auto_idx = 0usize;
    // `newformat.py` auto_numbering_state — `None` = ANS_INIT, `Some(true)`
    // = ANS_AUTO (empty `{}` fields), `Some(false)` = ANS_MANUAL (numbered
    // `{0}` fields).  Mixing the two raises ValueError.
    let mut numbering: Option<bool> = None;
    let rendered = format_render(
        fmt,
        positional,
        kwargs_dict,
        mapping,
        &mut auto_idx,
        &mut numbering,
        2,
    )?;
    Ok(pyre_object::w_str_from_wtf8(rendered))
}

/// `newformat.py Formatter.format` rendering pass.  Renders the
/// template `fmt`, threading the auto-/manual-numbering state through
/// the recursive evaluation of nested `{...}` format specs so that
/// `"{:{}}".format(42, ">5")` consumes positional args 0 then 1.
/// `depth` bounds that recursion (the markup recursion limit is 2).
fn format_render(
    fmt: &Wtf8,
    positional: &[PyObjectRef],
    kwargs_dict: Option<PyObjectRef>,
    mapping: Option<PyObjectRef>,
    auto_idx: &mut usize,
    numbering: &mut Option<bool>,
    depth: u32,
) -> Result<Wtf8Buf, crate::PyError> {
    use rustpython_common::format::{
        FieldName, FieldNamePart, FieldType, FormatParseError, FromTemplate,
    };
    let lookup_kwarg = |name: &str| -> Result<Option<PyObjectRef>, crate::PyError> {
        if let Some(m) = mapping {
            // `newformat.format_method(... w_mapping, True)` resolves
            // `{name}` via `space.getitem(mapping, w_key)` per
            // `newformat.py:Template.get_value`; KeyError propagates
            // to the caller (no silent default).
            let w_key = pyre_object::w_str_new(name);
            return crate::baseobjspace::getitem(m, w_key).map(Some);
        }
        if let Some(dict) = kwargs_dict {
            let v = unsafe { pyre_object::w_dict_lookup(dict, pyre_object::w_str_new(name)) };
            return Ok(v);
        }
        Ok(None)
    };

    if depth == 0 {
        return Err(crate::PyError::value_error("Max string recursion exceeded"));
    }
    let parsed = parse_format_parts(fmt)?;
    let mut result = Wtf8Buf::new();
    for part in &parsed {
        let (field_name, conversion_spec, format_spec) = match part {
            PyPyFormatPart::Literal(literal) => {
                result.push_wtf8(literal);
                continue;
            }
            PyPyFormatPart::Field {
                field_name,
                conversion_spec,
                format_spec,
            } => (field_name, conversion_spec, format_spec),
        };

        // `_get_argument` — resolve the base argument named by the field,
        // threading the auto-/manual-numbering state (`None` = uncommitted,
        // `Some(true)` = automatic `{}`, `Some(false)` = manual `{0}`).
        //
        // A missing `]` terminates the field, so it precedes base-argument
        // resolution; every other chain error (empty attribute, empty/`!`
        // item, stray char after `]`) is a resolution error that follows the
        // base lookup. Classify only the head (up to the first `.`/`[`) here,
        // so a missing/out-of-range base raises IndexError/KeyError before a
        // malformed attribute or item chain raises ValueError.
        let full = FieldName::parse(field_name);
        if matches!(full, Err(FormatParseError::MissingRightBracket)) {
            return Err(format_parse_err(FormatParseError::MissingRightBracket, fmt));
        }
        let mut head = Wtf8Buf::new();
        for ch in field_name.code_points() {
            if ch == '.' || ch == '[' {
                break;
            }
            head.push(ch);
        }
        let FieldName { field_type, .. } =
            FieldName::parse(&head).map_err(|e| format_parse_err(e, fmt))?;
        if mapping.is_some() && matches!(field_type, FieldType::Auto | FieldType::Index(_)) {
            return Err(crate::PyError::value_error(
                "Format string contains positional fields",
            ));
        }
        let mut val = match field_type {
            FieldType::Auto => {
                if let Some(false) = *numbering {
                    return Err(crate::PyError::value_error(
                        "cannot switch from manual field specification to automatic \
                         field numbering",
                    ));
                }
                *numbering = Some(true);
                let idx = *auto_idx;
                *auto_idx += 1;
                index_positional(positional, idx)?
            }
            FieldType::Index(idx) => {
                if let Some(true) = *numbering {
                    return Err(crate::PyError::value_error(
                        "cannot switch from automatic field numbering to manual \
                         field specification",
                    ));
                }
                *numbering = Some(false);
                index_positional(positional, idx)?
            }
            FieldType::Keyword(name) => {
                let name_str = name.as_str().unwrap_or("");
                match lookup_kwarg(name_str)? {
                    Some(v) => v,
                    None => {
                        return Err(crate::PyError::key_error_with_key(
                            pyre_object::w_str_from_wtf8(name),
                        ));
                    }
                }
            }
        };

        // Surface any remaining chain parse error now that the base argument
        // has been resolved, then walk the attribute/item chain.
        let FieldName { parts, .. } = full.map_err(|e| format_parse_err(e, fmt))?;

        // `_resolve_lookups` — walk the `.attr` / `[element]` chain; a
        // bracketed all-digit element is an integer index, anything else a
        // string key (already classified by `FieldNamePart`).
        for name_part in &parts {
            val = match name_part {
                FieldNamePart::Attribute(attr) => {
                    crate::baseobjspace::getattr_str(val, attr.as_str().unwrap_or(""))?
                }
                FieldNamePart::Index(idx) => {
                    crate::baseobjspace::getitem(val, pyre_object::w_int_new(*idx as i64))?
                }
                FieldNamePart::StringIndex(key) => {
                    // An all-digit bracket element that reached here overflowed
                    // the integer index — a value that fits is already
                    // classified as `Index` — so reject it like the head does.
                    if key
                        .as_str()
                        .ok()
                        .is_some_and(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
                    {
                        return Err(crate::PyError::value_error(
                            "Too many decimal digits in format string",
                        ));
                    }
                    crate::baseobjspace::getitem(val, pyre_object::w_str_from_wtf8(key.clone()))?
                }
            };
        }

        // A spec containing `{` is itself a template: render it (sharing the
        // numbering state and the recursion budget) before applying it.
        let resolved_spec = if format_spec.as_bytes().contains(&b'{') {
            format_render(
                format_spec,
                positional,
                kwargs_dict,
                mapping,
                auto_idx,
                numbering,
                depth - 1,
            )?
        } else {
            format_spec.clone()
        };

        // `_convert_field` — `!s`/`!r`/`!a` apply str / repr / ascii before
        // the format spec; any other conversion char is an error.  `!s`
        // preserves WTF-8 so a lone surrogate passes through unchanged.
        let converted = match conversion_spec {
            None => val,
            Some(cp) => match cp.to_char_lossy() {
                's' => pyre_object::w_str_from_wtf8(unsafe { crate::py_str_wtf8(val)? }),
                'r' => pyre_object::w_str_new(&unsafe { crate::py_repr(val)? }),
                'a' => pyre_object::w_str_new(&crate::builtins::py_ascii(val)?),
                c => {
                    return Err(crate::PyError::value_error(format!(
                        "Unknown conversion specifier {c}"
                    )));
                }
            },
        };
        let formatted = format_value_dispatch(converted, resolved_spec.as_str().unwrap_or(""))?;
        result.push_wtf8(&formatted);
    }
    Ok(result)
}

enum PyPyFormatPart {
    Literal(Wtf8Buf),
    Field {
        field_name: Wtf8Buf,
        conversion_spec: Option<CodePoint>,
        format_spec: Wtf8Buf,
    },
}

fn codepoint_slice(codepoints: &[CodePoint], start: usize, end: usize) -> Wtf8Buf {
    let mut out = Wtf8Buf::new();
    for &cp in &codepoints[start..end] {
        out.push(cp);
    }
    out
}

/// `newformat.py:TemplateFormatter._do_build_string/_parse_field`.
/// Braces inside the first-part item lookup (`{[{]}`) are ordinary key text;
/// braces after `:`/`!` are recursive format markup.
fn parse_format_parts(fmt: &Wtf8) -> Result<Vec<PyPyFormatPart>, crate::PyError> {
    let s: Vec<CodePoint> = fmt.code_points().collect();
    let end = s.len();
    let mut parts = Vec::new();
    let mut literal = Wtf8Buf::new();
    let mut i = 0usize;
    while i < end {
        let c = s[i];
        if c != '{' && c != '}' {
            literal.push(c);
            i += 1;
            continue;
        }
        if c == '}' {
            if i + 1 == end || s[i + 1] != '}' {
                return Err(crate::PyError::value_error(
                    "Single '}' encountered in format string",
                ));
            }
            literal.push(c);
            i += 2;
            continue;
        }
        if i + 1 == end {
            return Err(crate::PyError::value_error(
                "Single '{' encountered in format string",
            ));
        }
        if s[i + 1] == '{' {
            literal.push(c);
            i += 2;
            continue;
        }
        if !literal.is_empty() {
            parts.push(PyPyFormatPart::Literal(std::mem::take(&mut literal)));
        }

        i += 1;
        let field_start = i;
        let mut nested = 1usize;
        let mut in_second_part = false;
        while i < end {
            let c = s[i];
            if c == '{' {
                nested += 1;
            } else if c == '}' {
                nested -= 1;
                if nested == 0 {
                    break;
                }
            } else if c == '[' && !in_second_part {
                i += 1;
                while i < end && s[i] != ']' {
                    i += 1;
                }
                continue;
            } else if c == ':' || c == '!' {
                in_second_part = true;
            }
            i += 1;
        }
        if nested != 0 {
            return Err(crate::PyError::value_error(
                "expected '}' before end of string",
            ));
        }
        let field_end = i;

        let mut cursor = field_start;
        let mut field_name_end = field_end;
        let mut conversion_spec = None;
        let mut spec_start = field_end;
        while cursor < field_end {
            let c = s[cursor];
            if c == ':' || c == '!' {
                field_name_end = cursor;
                if c == '!' {
                    cursor += 1;
                    if cursor == field_end {
                        return Err(crate::PyError::value_error(
                            "end of string while looking for conversion specifier",
                        ));
                    }
                    conversion_spec = Some(s[cursor]);
                    cursor += 1;
                    if cursor < field_end {
                        if s[cursor] != ':' {
                            return Err(crate::PyError::value_error(
                                "expected ':' after conversion specifier",
                            ));
                        }
                        cursor += 1;
                    }
                } else {
                    cursor += 1;
                }
                spec_start = cursor;
                break;
            } else if c == '[' {
                while cursor + 1 < field_end && s[cursor + 1] != ']' {
                    cursor += 1;
                }
            } else if c == '{' {
                return Err(crate::PyError::value_error("unexpected '{' in field name"));
            }
            cursor += 1;
        }
        parts.push(PyPyFormatPart::Field {
            field_name: codepoint_slice(&s, field_start, field_name_end),
            conversion_spec,
            format_spec: codepoint_slice(&s, spec_start, field_end),
        });
        i += 1;
    }
    if !literal.is_empty() {
        parts.push(PyPyFormatPart::Literal(literal));
    }
    Ok(parts)
}

/// Fetch positional argument `idx`, raising the `str.format` IndexError for
/// an out-of-range replacement index.
fn index_positional(positional: &[PyObjectRef], idx: usize) -> Result<PyObjectRef, crate::PyError> {
    positional.get(idx).copied().ok_or_else(|| {
        crate::PyError::index_error(format!(
            "Replacement index {idx} out of range for positional args tuple"
        ))
    })
}

/// Map a template parse error to the matching `str.format` ValueError.
/// `FormatString` reports `UnmatchedBracket` for both a lone trailing `{`
/// and an opened-but-unclosed field, so the trailing byte disambiguates the
/// two CPython messages.
fn format_parse_err(
    err: rustpython_common::format::FormatParseError,
    fmt: &Wtf8,
) -> crate::PyError {
    use rustpython_common::format::FormatParseError as E;
    let msg = match err {
        E::UnmatchedBracket => {
            if fmt.as_bytes().last() == Some(&b'{') {
                "Single '{' encountered in format string"
            } else {
                "expected '}' before end of string"
            }
        }
        E::MissingStartBracket => "Single '}' encountered in format string",
        E::UnescapedStartBracketInLiteral => "Single '{' encountered in format string",
        E::MissingRightBracket => "expected '}' before end of string",
        E::EmptyAttribute => "Empty attribute in format string",
        E::UnknownConversion => "expected ':' after conversion specifier",
        // The template parser accepts only one bracket layer in a spec and
        // reports a second as `InvalidFormatSpecifier`; that is the markup
        // recursion limit being hit.
        E::InvalidFormatSpecifier => "Max string recursion exceeded",
        E::InvalidCharacterAfterRightBracket => "Only '[' and '.' may follow ']' in format string",
        E::TooManyDecimalDigits => "Too many decimal digits in format string",
    };
    crate::PyError::value_error(msg)
}

/// Geometry of a format spec: `[fill][align][sign][#][0][width][grouping]
/// [.precision][type]`.  Only the presentation types the shared engine
/// cannot format correctly — integer `c` and non-finite floats — read
/// this back; every other case parses through `FormatSpec` directly.
struct ParsedSpec {
    fill: char,
    align: Option<char>,
    sign: Option<char>,
    alt_form: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    ty: char,
}

fn parse_spec(spec: &str) -> ParsedSpec {
    let chars: Vec<char> = spec.chars().collect();
    let mut i = 0;
    let n = chars.len();
    let mut fill = ' ';
    let mut align: Option<char> = None;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        fill = chars[0];
        align = Some(chars[1]);
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = Some(chars[0]);
        i = 1;
    }
    let mut sign: Option<char> = None;
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        sign = Some(chars[i]);
        i += 1;
    }
    let mut alt_form = false;
    if i < n && chars[i] == '#' {
        alt_form = true;
        i += 1;
    }
    if i < n && chars[i] == '0' {
        // The `0` flag implies `fill='0', align='='` when neither was given.
        if align.is_none() {
            align = Some('=');
        }
        if fill == ' ' {
            fill = '0';
        }
        i += 1;
    }
    let mut width = 0usize;
    while i < n && chars[i].is_ascii_digit() {
        width = width * 10 + (chars[i] as u8 - b'0') as usize;
        i += 1;
    }
    let mut grouping: Option<char> = None;
    if i < n && matches!(chars[i], ',' | '_') {
        grouping = Some(chars[i]);
        i += 1;
    }
    let mut precision: Option<usize> = None;
    if i < n && chars[i] == '.' {
        i += 1;
        let mut p = 0usize;
        while i < n && chars[i].is_ascii_digit() {
            p = p * 10 + (chars[i] as u8 - b'0') as usize;
            i += 1;
        }
        precision = Some(p);
    }
    let ty = if i < n { chars[i] } else { '\0' };
    ParsedSpec {
        fill,
        align,
        sign,
        alt_form,
        width,
        grouping,
        precision,
        ty,
    }
}

/// Pad `body` to `width` characters with `fill`, honouring the numeric
/// alignments.  `=` splits a leading sign (and `0x`/`0o`/`0b` base prefix)
/// from the digits and inserts the fill between them.
fn pad_to_width(body: String, fill: char, align: char, width: usize) -> String {
    if body.chars().count() >= width {
        return body;
    }
    let need = width - body.chars().count();
    match align {
        '<' => {
            let mut s = body;
            for _ in 0..need {
                s.push(fill);
            }
            s
        }
        '^' => {
            let left = need / 2;
            let right = need - left;
            let mut s = String::with_capacity(width);
            for _ in 0..left {
                s.push(fill);
            }
            s.push_str(&body);
            for _ in 0..right {
                s.push(fill);
            }
            s
        }
        '=' => {
            let mut chars = body.chars().peekable();
            let mut prefix = String::new();
            if let Some(&c) = chars.peek() {
                if c == '-' || c == '+' || c == ' ' {
                    prefix.push(c);
                    chars.next();
                }
            }
            let rest_so_far: String = chars.clone().collect();
            if rest_so_far.len() >= 2 && rest_so_far.as_bytes()[0] == b'0' {
                let next = rest_so_far.as_bytes()[1];
                if matches!(next, b'x' | b'X' | b'o' | b'b') {
                    prefix.push('0');
                    prefix.push(next as char);
                    chars.next();
                    chars.next();
                }
            }
            let digits: String = chars.collect();
            let mut s = String::with_capacity(width);
            s.push_str(&prefix);
            for _ in 0..need {
                s.push(fill);
            }
            s.push_str(&digits);
            s
        }
        _ => {
            let mut s = String::with_capacity(width);
            for _ in 0..need {
                s.push(fill);
            }
            s.push_str(&body);
            s
        }
    }
}

/// A `&str` paired with its precomputed code-point count, adapting a
/// str body to the `CharLen + Deref<str>` bound `FormatSpec::format_string`
/// requires for width padding.
struct CharLenStr<'a>(&'a str, usize);

impl std::ops::Deref for CharLenStr<'_> {
    type Target = str;
    fn deref(&self) -> &str {
        self.0
    }
}

impl rustpython_common::format::CharLen for CharLenStr<'_> {
    fn char_len(&self) -> usize {
        self.1
    }
}

/// Render a presentation-type code for the "Unknown format code" message:
/// printable ASCII (`0x21..=0x7f`) verbatim, anything else as `\x{hex}`.
fn unknown_code_display(c: char) -> String {
    if ('\u{21}'..='\u{7f}').contains(&c) {
        c.to_string()
    } else {
        format!("\\x{:x}", c as u32)
    }
}

/// Map a `FormatSpec` engine error to the matching Python exception.
/// `spec` and `type_name` supply the value's format spec and type for the
/// messages that name them; `integer` selects the integer-specific text
/// for the `z` presentation code.
fn format_spec_err(
    err: rustpython_common::format::FormatSpecError,
    spec: &str,
    type_name: &str,
    integer: bool,
) -> crate::PyError {
    use rustpython_common::format::FormatSpecError as E;
    match err {
        E::DecimalDigitsTooMany => {
            crate::PyError::value_error("Too many decimal digits in format string")
        }
        // An integer never accepts a precision, so its presence is reported
        // before the value is range-checked — the size never surfaces.
        E::PrecisionTooBig if integer => {
            crate::PyError::value_error("Precision not allowed in integer format specifier")
        }
        E::PrecisionTooBig => crate::PyError::value_error("precision too big"),
        E::InvalidFormatSpecifier => crate::PyError::value_error(format!(
            "Invalid format specifier '{spec}' for object of type '{type_name}'"
        )),
        E::UnspecifiedFormat(c1, c2) => {
            crate::PyError::value_error(format!("Cannot specify '{c1}' with '{c2}'."))
        }
        E::ExclusiveFormat(c1, c2) => {
            crate::PyError::value_error(format!("Cannot specify both '{c1}' and '{c2}'."))
        }
        E::UnknownFormatCode(c, _) if integer && c == 'z' => crate::PyError::value_error(
            "Negative zero coercion (z) not allowed in integer format specifier",
        ),
        E::UnknownFormatCode(c, _) => crate::PyError::value_error(format!(
            "Unknown format code '{}' for object of type '{type_name}'",
            unknown_code_display(c)
        )),
        E::PrecisionNotAllowed => {
            crate::PyError::value_error("Precision not allowed in integer format specifier")
        }
        E::NotAllowed(s) => crate::PyError::value_error(format!(
            "{s} not allowed with integer format specifier 'c'"
        )),
        E::UnableToConvert => crate::PyError::value_error("Unable to convert int to float"),
        E::CodeNotInRange => crate::PyError::overflow_error("%c arg not in range(0x110000)"),
        E::ZeroPadding => {
            crate::PyError::value_error("Zero padding is not allowed in complex format specifier")
        }
        E::AlignmentFlag => crate::PyError::value_error(
            "'=' alignment flag is not allowed in complex format specifier",
        ),
        E::NotImplemented(c, s) => crate::PyError::value_error(format!(
            "Format code '{c}' for object of type '{s}' not implemented yet"
        )),
    }
}

/// Public entry point for the f-string `FormatWithSpec` opcode in
/// `eval.rs::format_with_spec`. Forwards to the same parser used by
/// `str.format` so both surfaces share the spec semantics.
pub fn format_with_spec_public(val: PyObjectRef, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    format_with_spec(val, spec)
}

/// Bind the type-level `__format__` descriptor `meth` to `val` and call it
/// with `spec_obj`, requiring a `str` result.  Dispatching the looked-up
/// `meth` (rather than a fresh instance lookup) keeps `__format__` a
/// type-level special method — an instance-dict `__format__` is ignored —
/// and binds a `@staticmethod` / `@classmethod` / other descriptor override
/// through the descriptor protocol.  The spec is passed through untouched so
/// the type's own `__format__` runs its validation.
pub(crate) fn call_format_dispatch(
    val: PyObjectRef,
    meth: PyObjectRef,
    spec_obj: PyObjectRef,
) -> Result<Wtf8Buf, crate::PyError> {
    unsafe {
        let w_type = crate::typedef::r#type(val).unwrap_or(pyre_object::PY_NULL);
        let result = crate::baseobjspace::get_and_call_function(meth, val, w_type, &[spec_obj])?;
        if !pyre_object::is_str(result) {
            return Err(crate::PyError::type_error(format!(
                "__format__ must return a str, not {}",
                arg_type_name(result)
            )));
        }
        Ok(pyre_object::w_str_get_wtf8(result).to_wtf8_buf())
    }
}

/// `PyObject_Format` — when `val` is a class instance whose type defines
/// `__format__`, dispatch to it (the result must be a `str`); otherwise
/// apply the shared builtin spec parser, with an empty spec collapsing to
/// `str(value)`.  Shared by `format()`, the `FormatSimple`/`FormatWithSpec`
/// f-string opcodes, and `str.format` field formatting.
pub fn format_value_dispatch(val: PyObjectRef, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    // A class instance always dispatches to its `__format__` (its own
    // override or the inherited `object.__format__`).  A builtin subclass
    // dispatches whenever it overrides `__format__` with anything other than
    // the inherited builtin default — a `def`, `@staticmethod`,
    // `@classmethod`, or any non-`BUILTIN_FUNCTION_TYPE` descriptor; the
    // builtin default takes the fast path below, which formats the
    // underlying value directly.  `__format__` is resolved on the type (not
    // the instance) so an instance-dict attribute does not shadow it.
    if let Some(meth) = unsafe { crate::baseobjspace::lookup(val, "__format__") } {
        if unsafe { is_instance(val) }
            || !unsafe { py_type_check(meth, &crate::function::BUILTIN_FUNCTION_TYPE) }
        {
            let spec_obj = pyre_object::w_str_new(spec);
            return call_format_dispatch(val, meth, spec_obj);
        }
    }
    if spec.is_empty() {
        // Empty spec collapses to `str(value)`, preserved in WTF-8 so a
        // str — or an exception whose single argument is a str — keeps
        // its lone surrogates.
        Ok(unsafe { crate::py_str_wtf8(val)? })
    } else {
        Ok(format_with_spec_public(val, spec)?)
    }
}

/// The type name of `obj` for a TypeError message — the `w_class` name
/// for instances, else the storage type name.
pub(crate) fn arg_type_name(obj: PyObjectRef) -> String {
    if obj.is_null() {
        return "object".to_string();
    }
    unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => w_type_get_name(tp).to_string(),
            None => (*(*obj).ob_type).name.to_string(),
        }
    }
}

/// Read a format spec's stored string value. The spec must be a `str`
/// (or subclass); its `__str__` is not consulted, so a raising override
/// does not leak out of formatting.  `arg_desc` names the argument in the
/// `TypeError` raised for a non-`str` spec (`format()` reports `format()
/// argument 2`, a type's `__format__` reports `__format__() argument`).
pub(crate) fn read_format_spec(
    spec_obj: PyObjectRef,
    arg_desc: &str,
) -> Result<String, crate::PyError> {
    if !spec_obj.is_null() && unsafe { is_str(spec_obj) } {
        return Ok(unsafe { w_str_get_value(spec_obj) }.to_string());
    }
    Err(crate::PyError::type_error(format!(
        "{arg_desc} must be str, not {}",
        arg_type_name(spec_obj)
    )))
}

/// `int/float/str/bool.__format__(self, format_spec)` — formats `self`
/// through the shared spec parser without re-dispatching to an instance
/// `__format__` (which `format_value_dispatch` would do for subclasses,
/// risking recursion).  An empty spec collapses to `str(self)`.
pub fn builtin_value_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let spec = if args.len() > 1 {
        read_format_spec(args[1], "__format__() argument")?
    } else {
        String::new()
    };
    if spec.is_empty() {
        // `str(self)` — a str self passes through as WTF-8.
        if unsafe { pyre_object::is_str(args[0]) } {
            return Ok(pyre_object::w_str_from_wtf8(unsafe {
                crate::display::py_str_wtf8(args[0])?
            }));
        }
        return Ok(pyre_object::w_str_new(&unsafe { crate::py_str(args[0])? }));
    }
    Ok(pyre_object::w_str_from_wtf8(format_with_spec_public(
        args[0], &spec,
    )?))
}

fn format_with_spec(val: PyObjectRef, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    use rustpython_common::format::FormatSpec;
    unsafe {
        // `int` / `bool` / `long` share the integer formatter.  The `c`
        // character type is formatted here instead — the shared engine pads
        // by byte length rather than code point — and the float presentation
        // codes (`e`/`f`/`g`/`%`) format the `f64` conversion of the value.
        if pyre_object::is_int(val) || pyre_object::is_bool(val) || pyre_object::is_long(val) {
            let type_name = arg_type_name(val);
            let big = if pyre_object::is_bool(val) {
                BigInt::from(pyre_object::w_bool_get_value(val) as i64)
            } else {
                crate::builtins::obj_to_bigint(val)
            };
            let parsed = FormatSpec::parse(spec);
            if spec.ends_with('c') && parsed.is_ok() {
                return format_char(&big, spec);
            }
            // A float presentation code formats the `f64` conversion (`n` and
            // the integer bases keep full integer precision instead).
            if matches!(parse_spec(spec).ty, 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%') {
                let f = big.to_f64().unwrap_or(f64::INFINITY);
                if f.is_infinite() {
                    return Err(crate::PyError::overflow_error(
                        "int too large to convert to float",
                    ));
                }
                return format_finite_float(f, spec);
            }
            let parsed = parsed.map_err(|e| format_spec_err(e, spec, &type_name, true))?;
            let s = parsed
                .format_int(&big)
                .map_err(|e| format_spec_err(e, spec, &type_name, true))?;
            return Ok(Wtf8Buf::from_string(s));
        }
        if pyre_object::is_float(val) {
            let v = pyre_object::floatobject::w_float_get_value(val);
            // `inf` / `nan` go through the local formatter — the shared engine
            // inserts the digit-grouping separator into their padding.
            if v.is_nan() || v.is_infinite() {
                return format_nonfinite(v, spec);
            }
            return format_finite_float(v, spec);
        }
        if let Some((re, im)) = crate::objspace::descroperation::complex_val(val) {
            let p = parse_spec(spec);
            if p.fill == '0' {
                return Err(crate::PyError::value_error(
                    "Zero padding is not allowed in complex format specifier",
                ));
            }
            if p.align == Some('=') {
                return Err(crate::PyError::value_error(
                    "'=' alignment flag is not allowed in complex format specifier",
                ));
            }
            let align = p.align.unwrap_or('>');
            // No presentation type and no precision: pad str(self), which
            // already carries the parentheses / bare-imaginary form.
            if p.ty == '\0' && p.precision.is_none() {
                let body = Wtf8Buf::from_string(crate::py_str(val)?);
                return Ok(pad_wtf8(&body, p.fill, align, p.width));
            }
            // A presentation type or precision formats the real and imaginary
            // parts as floats and joins them; the imaginary part always
            // carries an explicit sign and the whole ends in `j`.
            let prec = p
                .precision
                .map(|precision| format!(".{precision}"))
                .unwrap_or_default();
            let ty = if p.ty == '\0' { 'f' } else { p.ty };
            let re_spec = format!("{prec}{ty}");
            let im_spec = format!("+{prec}{ty}");
            let mut body = format_finite_float(re, &re_spec)?;
            let im_str = format_finite_float(im, &im_spec)?;
            body.push_wtf8(&im_str);
            body.push_char('j');
            return Ok(pad_wtf8(&body, p.fill, align, p.width));
        }
        if pyre_object::is_str(val) {
            let full = pyre_object::w_str_get_wtf8(val);
            // The shared string formatter rejects grouping and numeric types
            // but not a sign or `=` alignment, which are also disallowed for
            // strings.
            reject_string_sign_align(spec)?;
            // A `0` fill flag at the start of the width pads text with the
            // default-left alignment; the shared formatter treats it as a
            // numeric alignment, so handle it here.
            let sc: Vec<char> = spec.chars().collect();
            let zero_fill = if sc.len() >= 2 && matches!(sc[1], '<' | '>' | '=' | '^') {
                sc.get(2) == Some(&'0')
            } else if sc
                .first()
                .is_some_and(|c| matches!(c, '<' | '>' | '=' | '^'))
            {
                sc.get(1) == Some(&'0')
            } else {
                sc.first() == Some(&'0')
            };
            if zero_fill {
                return format_surrogate_str(full, spec);
            }
            // A valid-UTF-8 body goes through the shared string formatter,
            // which pads by code point.
            if let Ok(valid) = full.as_str() {
                let parsed =
                    FormatSpec::parse(spec).map_err(|e| format_spec_err(e, spec, "str", false))?;
                let s = parsed
                    .format_string(&CharLenStr(valid, valid.chars().count()))
                    .map_err(|e| format_spec_err(e, spec, "str", false))?;
                return Ok(Wtf8Buf::from_string(s));
            }
            // A body carrying a lone surrogate cannot be handed to the
            // `&str`-typed formatter, so pad it by code point here.
            return format_surrogate_str(full, spec);
        }
        // Reached only for the rare builtin type whose `__format__` is the
        // inherited default yet still routes here with a non-empty spec;
        // format its `str()` through the shared string formatter.
        let s = crate::py_str(val)?;
        let parsed = FormatSpec::parse(spec)
            .map_err(|e| format_spec_err(e, spec, &arg_type_name(val), false))?;
        let out = parsed
            .format_string(&CharLenStr(&s, s.chars().count()))
            .map_err(|e| format_spec_err(e, spec, &arg_type_name(val), false))?;
        Ok(Wtf8Buf::from_string(out))
    }
}

/// Reject a sign flag, `=` alignment, or `#` alternate form in a string
/// format spec — all disallowed for `str` values but accepted by the shared
/// formatter.  Only the *explicit* alignment is inspected: a leading `0` flag
/// (which the shared parser normalises to `=`) is a zero fill and stays legal
/// for text, and a `#` used as a fill character precedes the alignment.
fn reject_string_sign_align(spec: &str) -> Result<(), crate::PyError> {
    let chars: Vec<char> = spec.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut align: Option<char> = None;
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        align = Some(chars[1]);
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = Some(chars[0]);
        i = 1;
    }
    if align == Some('=') {
        return Err(crate::PyError::value_error(
            "'=' alignment not allowed in string format specifier",
        ));
    }
    if i < n && matches!(chars[i], '+' | '-' | ' ') {
        return Err(crate::PyError::value_error(if chars[i] == ' ' {
            "Space not allowed in string format specifier"
        } else {
            "Sign not allowed in string format specifier"
        }));
    }
    if i < n && chars[i] == '#' {
        return Err(crate::PyError::value_error(
            "Alternate form (#) not allowed in string format specifier",
        ));
    }
    Ok(())
}

/// Format a `str` value whose WTF-8 body carries a lone surrogate, which
/// the shared `&str`-typed formatter cannot accept.  Only the geometry
/// codes that make sense for text — `[fill][align][width][.precision]`
/// and an optional `s` type — are honoured; a numeric presentation type
/// raises the same "unknown format code" error the valid-UTF-8 path does.
/// A sign / `=` alignment is rejected by the caller.
fn format_surrogate_str(body: &Wtf8, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    let chars: Vec<char> = spec.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut fill = ' ';
    let mut align = '<';
    if n >= 2 && matches!(chars[1], '<' | '>' | '=' | '^') {
        fill = chars[0];
        align = chars[1];
        i = 2;
    } else if n >= 1 && matches!(chars[0], '<' | '>' | '=' | '^') {
        align = chars[0];
        i = 1;
    }
    if i < n && chars[i] == '0' {
        fill = '0';
        i += 1;
    }
    let mut width = 0usize;
    while i < n && chars[i].is_ascii_digit() {
        width = width * 10 + (chars[i] as u8 - b'0') as usize;
        i += 1;
    }
    let mut precision: Option<usize> = None;
    if i < n && chars[i] == '.' {
        i += 1;
        let mut p = 0usize;
        while i < n && chars[i].is_ascii_digit() {
            p = p * 10 + (chars[i] as u8 - b'0') as usize;
            i += 1;
        }
        precision = Some(p);
    }
    if i < n {
        let ty = chars[i];
        if ty != 's' {
            return Err(crate::PyError::value_error(format!(
                "Unknown format code '{ty}' for object of type 'str'"
            )));
        }
        i += 1;
    }
    if i != n {
        return Err(crate::PyError::value_error(format!(
            "Invalid format specifier '{spec}' for object of type 'str'"
        )));
    }
    let body = if let Some(prec) = precision {
        let mut t = Wtf8Buf::new();
        for (k, cp) in body.code_points().enumerate() {
            if k >= prec {
                break;
            }
            t.push(cp);
        }
        t
    } else {
        body.to_wtf8_buf()
    };
    Ok(pad_wtf8(&body, fill, align, width))
}

/// Format an integer through the `c` presentation type: reject the flags
/// that make no sense for a single character, map the value to its code
/// point, then pad by code point.  The caller has already confirmed the
/// spec parses and ends in `c`.
fn format_char(num: &BigInt, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec);
    // Rejection order matches the reference: grouping, then precision, then
    // sign, then alternate form — each beats the value range check.
    if let Some(sep) = p.grouping {
        return Err(crate::PyError::value_error(format!(
            "Cannot specify '{sep}' with 'c'."
        )));
    }
    if p.precision.is_some() {
        return Err(crate::PyError::value_error(
            "Precision not allowed in integer format specifier",
        ));
    }
    if p.sign.is_some() {
        return Err(crate::PyError::value_error(
            "Sign not allowed with integer format specifier 'c'",
        ));
    }
    if p.alt_form {
        return Err(crate::PyError::value_error(
            "Alternate form (#) not allowed with integer format specifier 'c'",
        ));
    }
    let cp = match num.to_i64() {
        None => {
            return Err(crate::PyError::overflow_error(
                "Python int too large to convert to C long",
            ));
        }
        Some(n) => match u32::try_from(n).ok().and_then(CodePoint::from_u32) {
            Some(cp) => cp,
            None => {
                return Err(crate::PyError::overflow_error(
                    "%c arg not in range(0x110000)",
                ));
            }
        },
    };
    let mut body = Wtf8Buf::new();
    body.push(cp);
    // `c` keeps the integer default alignment (right).
    Ok(pad_wtf8(&body, p.fill, p.align.unwrap_or('>'), p.width))
}

/// Format `inf` / `nan`: validate the presentation type, then emit the
/// signed word (`inf` / `nan`, upper-cased for `E`/`F`/`G`, `%`-suffixed
/// for the percentage type) padded to width.  Grouping only validates
/// here — a non-finite value has no digits to separate.
fn format_nonfinite(v: f64, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec);
    validate_float_spec(spec, &p)?;
    let upper = matches!(p.ty, 'E' | 'F' | 'G');
    let word = match (v.is_nan(), upper) {
        (true, false) => "nan",
        (true, true) => "NAN",
        (false, false) => "inf",
        (false, true) => "INF",
    };
    let mut magnitude = String::from(word);
    if p.ty == '%' {
        magnitude.push('%');
    }
    // A NaN carries no meaningful sign, so it takes only an explicit sign
    // flag; an infinity takes its own sign.
    let sign = if v.is_sign_negative() && !v.is_nan() {
        "-"
    } else {
        match p.sign {
            Some('+') => "+",
            Some(' ') => " ",
            _ => "",
        }
    };
    let body = format!("{sign}{magnitude}");
    Ok(Wtf8Buf::from_string(pad_to_width(
        body,
        p.fill,
        p.align.unwrap_or('>'),
        p.width,
    )))
}

/// Validate a float presentation spec against the reference rules,
/// independent of the value.  Structural errors (bad flag order, a
/// precision beyond `i32`, trailing garbage) come from the shared parser
/// for their exact messages; the type and grouping-with-`n` checks are
/// applied on top.
fn validate_float_spec(spec: &str, p: &ParsedSpec) -> Result<(), crate::PyError> {
    rustpython_common::format::FormatSpec::parse(spec)
        .map_err(|e| format_spec_err(e, spec, "float", false))?;
    if !matches!(p.ty, '\0' | 'e' | 'E' | 'f' | 'F' | 'g' | 'G' | 'n' | '%') {
        return Err(crate::PyError::value_error(format!(
            "Unknown format code '{}' for object of type 'float'",
            unknown_code_display(p.ty)
        )));
    }
    if let Some(sep) = p.grouping
        && p.ty == 'n'
    {
        return Err(crate::PyError::value_error(format!(
            "Cannot specify '{sep}' with 'n'."
        )));
    }
    Ok(())
}

/// Format a finite `f64` through `spec`.  Every presentation type
/// (`\0`/`e`/`E`/`f`/`F`/`g`/`G`/`n`/`%`) pads, groups, and rounds through the
/// shared engine.  `validate_float_spec` still supplies the type and
/// grouping-with-`n` messages before delegating.
fn format_finite_float(v: f64, spec: &str) -> Result<Wtf8Buf, crate::PyError> {
    let p = parse_spec(spec);
    validate_float_spec(spec, &p)?;
    let parsed = rustpython_common::format::FormatSpec::parse(spec)
        .map_err(|e| format_spec_err(e, spec, "float", false))?;
    let s = parsed
        .format_float(v)
        .map_err(|e| format_spec_err(e, spec, "float", false))?;
    Ok(Wtf8Buf::from_string(s))
}

/// Pad a WTF-8 string body to `width` code points with `fill`,
/// honouring `<` / `^` / `>` alignment.  String bodies never use the
/// numeric `=` alignment, so any non-`<`/`^` alignment right-aligns.
fn pad_wtf8(body: &Wtf8, fill: char, align: char, width: usize) -> Wtf8Buf {
    let body_len = body.code_points().count();
    if body_len >= width {
        return body.to_wtf8_buf();
    }
    let need = width - body_len;
    let fill_cp = CodePoint::from_char(fill);
    let mut out = Wtf8Buf::with_capacity(body.len() + need * 4);
    match align {
        '<' => {
            out.push_wtf8(body);
            push_cp_repeated(&mut out, fill_cp, need);
        }
        '^' => {
            let left = need / 2;
            let right = need - left;
            push_cp_repeated(&mut out, fill_cp, left);
            out.push_wtf8(body);
            push_cp_repeated(&mut out, fill_cp, right);
        }
        _ => {
            push_cp_repeated(&mut out, fill_cp, need);
            out.push_wtf8(body);
        }
    }
    out
}

/// runicode.py:333 unicode_encode_utf_8 + interp_codecs.py
/// surrogatepass / surrogateescape encode branches.  The WTF-8 backing
/// already stores a lone surrogate as its three-byte sequence, so the
/// surrogate-free common case is a direct byte copy; surrogate code points
/// are routed to the named error handler.  `w_object` is the str being
/// encoded, threaded through so a strict failure can build a structured
/// UnicodeEncodeError carrying it.
fn encode_utf8_with_errors(
    w_object: PyObjectRef,
    err_mode: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let s: &Wtf8 = unsafe { w_str_get_wtf8(w_object) };
    // utf8_encode_utf_8 fast path: no surrogates → already valid UTF-8.
    if let Ok(valid) = s.as_str() {
        return Ok(valid.as_bytes().to_vec());
    }
    let mut out = Vec::with_capacity(s.len());
    let mut buf = [0u8; 4];
    let cps: Vec<CodePoint> = s.code_points().collect();
    let mut i = 0usize;
    while i < cps.len() {
        let cp = cps[i];
        if let Some(c) = cp.to_char() {
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            i += 1;
            continue;
        }
        let code = cp.to_u32();
        let index = i;
        match err_mode {
            // surrogatepass_errors encode branch (interp_codecs.py:455-458):
            // emit the three-byte sequence for the surrogate code point.
            "surrogatepass" => {
                out.push(0xE0 | (code >> 12) as u8);
                out.push(0x80 | ((code >> 6) & 0x3f) as u8);
                out.push(0x80 | (code & 0x3f) as u8);
            }
            // surrogateescape_errors encode branch (interp_codecs.py:528-534):
            // a 0xDC80..0xDCFF surrogate maps back to the byte code-0xDC00;
            // any other surrogate fails.
            "surrogateescape" => {
                if (0xDC80..=0xDCFF).contains(&code) {
                    out.push((code - 0xDC00) as u8);
                } else {
                    return Err(crate::typedef::unicode_encode_error(
                        "utf-8",
                        w_object,
                        index,
                        index + 1,
                        "surrogates not allowed",
                    ));
                }
            }
            "strict" => {
                return Err(crate::typedef::unicode_encode_error(
                    "utf-8",
                    w_object,
                    index,
                    index + 1,
                    "surrogates not allowed",
                ));
            }
            "ignore" => {}
            "replace" => out.push(b'?'),
            "backslashreplace" => out.extend_from_slice(format!("\\u{code:04x}").as_bytes()),
            "xmlcharrefreplace" => out.extend_from_slice(format!("&#{code};").as_bytes()),
            _ => {
                let (rep, newpos) = call_registered_encode_error_handler(
                    err_mode,
                    "utf-8",
                    w_object,
                    cps.len(),
                    index,
                    index + 1,
                    "surrogates not allowed",
                )?;
                match rep {
                    EncodeReplacement::Str(rcps) => {
                        for rc in rcps {
                            if rc >= 0x80 {
                                return Err(crate::typedef::unicode_encode_error(
                                    "utf-8",
                                    w_object,
                                    index,
                                    index + 1,
                                    "surrogates not allowed",
                                ));
                            }
                            out.push(rc as u8);
                        }
                    }
                    EncodeReplacement::Bytes(b) => out.extend_from_slice(&b),
                }
                i = newpos;
                continue;
            }
        }
        i += 1;
    }
    Ok(out)
}

/// PyPy: unicodeobject.py descr_encode → encode_object.
/// For the common 'utf-8' / 'ascii' fast paths, returns the UTF-8 bytes
/// of the string. Other codecs fall through to a best-effort UTF-8 encoding.
pub fn str_method_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "encode")?;
    // `encoding` and `errors` arrive positionally or by keyword; builtin
    // kwargs are packed in a trailing `__pyre_kw__` dict.
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    // `get_encoding_and_errors` unwraps both arguments through
    // `space.text_w`; a present non-string value raises
    // `TypeError("expected str, got X object")` (baseobjspace.py
    // `_typed_unwrap_error`).  An absent argument keeps the default.
    let str_arg = |obj: Option<PyObjectRef>, default: &str| -> Result<String, crate::PyError> {
        match obj {
            None => Ok(default.to_string()),
            Some(o) if o.is_null() => Ok(default.to_string()),
            Some(o) if unsafe { pyre_object::is_str(o) } => {
                Ok(unsafe { w_str_get_value(o) }.to_string())
            }
            Some(o) => {
                let tname = unsafe { (*(*o).ob_type).name };
                Err(crate::PyError::type_error(format!(
                    "expected str, got {tname} object"
                )))
            }
        }
    };
    // `encode(encoding=None, errors=None)` — both positional-or-keyword;
    // the gateway rejects unknown keywords and a value given both ways.
    crate::builtins::kwarg_reject_unknown(kwargs, &["encoding", "errors"], "encode")?;
    let dual =
        |name: &str, p: Option<PyObjectRef>| -> Result<Option<PyObjectRef>, crate::PyError> {
            let kw = crate::builtins::kwarg_get(kwargs, name);
            if p.is_some() && kw.is_some() {
                return Err(crate::PyError::type_error(format!(
                    "got multiple values for argument '{name}'"
                )));
            }
            Ok(p.or(kw))
        };
    let encoding = str_arg(dual("encoding", pos.get(1).copied())?, "utf-8")?;
    let errors = str_arg(dual("errors", pos.get(2).copied())?, "strict")?;
    Ok(pyre_object::w_bytes_from_bytes(&encode_object(
        args[0], &encoding, &errors,
    )?))
}

/// `unicodeobject.py W_UnicodeObject.descr_encode` → `encode_object`.
/// Encodes a str (`w_object`) to bytes with the named codec and error
/// handler.  Shared by `str.encode`, `bytes(str, …)` and
/// `bytearray(str, …)` so all three honour the same codec set and error
/// handlers.  The whole path reads the surrogate-aware WTF-8 view, so a
/// lone surrogate is routed to the error handler rather than crashing.
pub fn encode_object(
    w_object: PyObjectRef,
    encoding: &str,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let enc_lower = encoding.to_ascii_lowercase().replace('_', "-");
    if crate::importing::dev_mode_flag()
        && matches!(
            enc_lower.as_str(),
            "utf-8"
                | "utf8"
                | "u8"
                | "ascii"
                | "us-ascii"
                | "646"
                | "latin-1"
                | "latin1"
                | "iso-8859-1"
                | "8859"
                | "raw-unicode-escape"
                | "utf-16"
                | "utf-16-le"
                | "utf-16-be"
                | "utf-32"
                | "utf-32-le"
                | "utf-32-be"
        )
    {
        crate::module::_codecs::validate_error_handler(errors)?;
    }
    if matches!(enc_lower.as_str(), "utf-8" | "utf8" | "u8") {
        return encode_utf8_with_errors(w_object, errors);
    }
    let s = unsafe { w_str_get_wtf8(w_object) };
    match enc_lower.as_str() {
        "ascii" | "us-ascii" | "646" => encode_narrow(
            s,
            w_object,
            "ascii",
            0x7f,
            "ordinal not in range(128)",
            errors,
        ),
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => encode_narrow(
            s,
            w_object,
            "latin-1",
            0xff,
            "ordinal not in range(256)",
            errors,
        ),
        "raw-unicode-escape" => Ok(encode_raw_unicode_escape(s)),
        _ => match encode_utf16_32(s, &enc_lower, w_object, errors) {
            Some(out) => out,
            None => {
                let encoded =
                    crate::module::_codecs::encode_text_codec(w_object, encoding, errors)?;
                Ok(unsafe { pyre_object::bytesobject::bytes_like_data(encoded) }.to_vec())
            }
        },
    }
}

/// `unicodeobject.c:_PyUnicode_EncodeRawUnicodeEscape` — code points
/// below 0x100 map to a single Latin-1 byte; 0x100..0x10000 become the
/// 6-byte `\uXXXX` form; everything larger becomes `\UXXXXXXXX`.  Unlike
/// `unicode-escape`, the backslash and control characters are not
/// escaped.
pub fn encode_raw_unicode_escape(s: &Wtf8) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    for cp in s.code_points() {
        let v = cp.to_u32();
        if v < 0x100 {
            out.push(v as u8);
        } else if v < 0x10000 {
            out.extend_from_slice(format!("\\u{v:04x}").as_bytes());
        } else {
            out.extend_from_slice(format!("\\U{v:08x}").as_bytes());
        }
    }
    out
}

/// `unicodeobject.c:_PyUnicode_DecodeRawUnicodeEscape` — the inverse of
/// [`encode_raw_unicode_escape`].  A backslash starts a `\uXXXX` /
/// `\UXXXXXXXX` escape; any other byte (including a lone backslash or a
/// malformed escape) is taken as a Latin-1 code point.
pub fn decode_raw_unicode_escape(data: &[u8], errors: &str) -> Result<Wtf8Buf, crate::PyError> {
    let mut out = Wtf8Buf::new();
    // A custom error handler may replace exc.object; decoding then resumes
    // from the new bytes (`buf`).
    let mut buf: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(data);
    let mut i = 0usize;
    while i < buf.len() {
        let b = buf[i];
        if b != b'\\' {
            out.push_char(b as char);
            i += 1;
            continue;
        }
        // Count the run of backslashes; only an odd run can introduce a
        // `\u`/`\U` escape (an even run is literal escaped backslashes,
        // but raw-unicode-escape does not collapse them — each `\` is a
        // literal byte 0x5c).  The escape applies when `\` is followed by
        // `u` or `U` with enough hex digits.
        let kind = buf.get(i + 1).copied();
        let want = match kind {
            Some(b'u') => 4usize,
            Some(b'U') => 8usize,
            _ => 0,
        };
        if want != 0 {
            let escape_start = i;
            let digits_start = i + 2;
            let available_end = (digits_start + want).min(buf.len());
            let mut hex_end = digits_start;
            while hex_end < available_end && buf[hex_end].is_ascii_hexdigit() {
                hex_end += 1;
            }
            let numeric = if available_end == digits_start + want && hex_end == available_end {
                std::str::from_utf8(&buf[digits_start..available_end])
                    .ok()
                    .and_then(|s| u32::from_str_radix(s, 16).ok())
            } else {
                None
            };
            let parsed = numeric.and_then(CodePoint::from_u32);
            if let Some(c) = parsed {
                out.push(c);
                i = available_end;
                continue;
            }
            let error_end = if numeric.is_some() {
                available_end
            } else {
                hex_end
            };
            let reason = if numeric.is_some() {
                "illegal Unicode character"
            } else if want == 4 {
                "truncated \\uXXXX escape"
            } else {
                "truncated \\UXXXXXXXX escape"
            };
            match errors {
                "ignore" => {}
                "replace" => out.push_char('\u{FFFD}'),
                "backslashreplace" => {
                    for &byte in &buf[escape_start..error_end] {
                        out.push_str(&format!("\\x{byte:02x}"));
                    }
                }
                _ => {
                    let (np, nb) = call_registered_decode_error_handler(
                        errors,
                        "rawunicodeescape",
                        &buf,
                        escape_start,
                        error_end,
                        reason,
                        &mut out,
                    )?;
                    if let Some(nb) = nb {
                        buf = std::borrow::Cow::Owned(nb);
                    }
                    i = np;
                    continue;
                }
            }
            i = error_end;
            continue;
        }
        // Not a valid escape — emit both bytes literally as Latin-1.
        out.push_char(b as char);
        if let Some(next) = kind {
            out.push_char(next as char);
            i += 2;
        } else {
            i += 1;
        }
    }
    Ok(out)
}

fn encode_narrow(
    s: &Wtf8,
    source: PyObjectRef,
    enc_name: &str,
    max_cp: u32,
    range_msg: &str,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let cps: Vec<u32> = s.code_points().map(|c| c.to_u32()).collect();
    let mut out: Vec<u8> = Vec::with_capacity(cps.len());
    let mut i = 0usize;
    while i < cps.len() {
        if cps[i] <= max_cp {
            out.push(cps[i] as u8);
            i += 1;
            continue;
        }
        // `surrogateescape` rescues only a 0xDC80..0xDCFF code point, mapping
        // it back to the byte `code-0xDC00` (interp_codecs.py:528-534); any
        // other unencodable code point still raises, so it is handled one at
        // a time rather than over the maximal run.
        if errors == "surrogateescape" && (0xDC80..=0xDCFF).contains(&cps[i]) {
            out.push((cps[i] - 0xDC00) as u8);
            i += 1;
            continue;
        }
        // Maximal run of consecutive unencodable code points — `strict`
        // reports the whole span as one error, like CPython.  A
        // `surrogateescape`-rescuable code point ends the run.
        let start = i;
        let mut end = i;
        while end < cps.len()
            && cps[end] > max_cp
            && !(errors == "surrogateescape" && (0xDC80..=0xDCFF).contains(&cps[end]))
        {
            end += 1;
        }
        match errors {
            // `surrogateescape` reached here only for an unencodable code
            // point outside the rescue range, so it raises like `strict`.
            // `surrogatepass` only rescues surrogates for utf-8/16/32, so a
            // narrow codec re-raises the original UnicodeEncodeError.
            "strict" | "surrogateescape" | "surrogatepass" => {
                return Err(crate::typedef::unicode_encode_error(
                    enc_name, source, start, end, range_msg,
                ));
            }
            "ignore" => {}
            "replace" => out.resize(out.len() + (end - start), b'?'),
            "backslashreplace" => {
                for &cp in &cps[start..end] {
                    let esc = if cp <= 0xff {
                        format!("\\x{cp:02x}")
                    } else if cp <= 0xffff {
                        format!("\\u{cp:04x}")
                    } else {
                        format!("\\U{cp:08x}")
                    };
                    out.extend_from_slice(esc.as_bytes());
                }
            }
            "xmlcharrefreplace" => {
                for &cp in &cps[start..end] {
                    out.extend_from_slice(format!("&#{cp};").as_bytes());
                }
            }
            _ => {
                let (rep, newpos) = call_registered_encode_error_handler(
                    errors,
                    enc_name,
                    source,
                    cps.len(),
                    start,
                    end,
                    range_msg,
                )?;
                match rep {
                    EncodeReplacement::Str(rcps) => {
                        for rc in rcps {
                            if rc > max_cp {
                                return Err(crate::typedef::unicode_encode_error(
                                    enc_name, source, start, end, range_msg,
                                ));
                            }
                            out.push(rc as u8);
                        }
                    }
                    EncodeReplacement::Bytes(b) => out.extend_from_slice(&b),
                }
                i = newpos;
                continue;
            }
        }
        i = end;
    }
    Ok(out)
}

/// Collapse a normalized encoding name to its separator-free form so
/// that `utf-16-le`, `utf16le` and `utf_16_le` all compare equal.
fn compact_codec_name(lower: &str) -> String {
    lower
        .chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .collect()
}

/// Append a 16-bit code unit in the requested byte order.
fn push_unit16(out: &mut Vec<u8>, unit: u16, big_endian: bool) {
    out.extend_from_slice(&if big_endian {
        unit.to_be_bytes()
    } else {
        unit.to_le_bytes()
    });
}

/// Append a 32-bit code unit in the requested byte order.
fn push_unit32(out: &mut Vec<u8>, unit: u32, big_endian: bool) {
    out.extend_from_slice(&if big_endian {
        unit.to_be_bytes()
    } else {
        unit.to_le_bytes()
    });
}

/// Emit a non-surrogate scalar value: utf-32 writes one 32-bit unit,
/// utf-16 writes one BMP unit or a surrogate pair for astral planes.
fn emit_scalar(out: &mut Vec<u8>, cp: u32, is32: bool, big_endian: bool) {
    if is32 {
        push_unit32(out, cp, big_endian);
    } else if cp <= 0xFFFF {
        push_unit16(out, cp as u16, big_endian);
    } else {
        let v = cp - 0x10000;
        push_unit16(out, 0xD800 | (v >> 10) as u16, big_endian);
        push_unit16(out, 0xDC00 | (v & 0x3FF) as u16, big_endian);
    }
}

/// utf-16 / utf-32 encode for the `lower`-normalized codec name, or
/// `None` if `lower` names neither.  The bare `utf-16` / `utf-32` forms
/// emit a little-endian BOM; the `-le` / `-be` forms omit it.  A lone
/// surrogate is routed through `errors` (`surrogatepass` emits its raw
/// code unit; `strict` raises) rather than crashing.
pub fn encode_utf16_32(
    s: &Wtf8,
    lower: &str,
    w_object: PyObjectRef,
    errors: &str,
) -> Option<Result<Vec<u8>, crate::PyError>> {
    // `codec` is the canonical name reported in a UnicodeEncodeError, so
    // a `-le` / `-be` spelling keeps its suffix while `utf16` normalizes
    // to `utf-16`.
    let (is32, big_endian, bom, codec) = match compact_codec_name(lower).as_str() {
        "utf16" | "u16" => (false, false, true, "utf-16"),
        "utf16le" => (false, false, false, "utf-16-le"),
        "utf16be" => (false, true, false, "utf-16-be"),
        "utf32" | "u32" => (true, false, true, "utf-32"),
        "utf32le" => (true, false, false, "utf-32-le"),
        "utf32be" => (true, true, false, "utf-32-be"),
        _ => return None,
    };
    Some(encode_utf16_32_impl(
        s, is32, big_endian, bom, codec, w_object, errors,
    ))
}

fn encode_utf16_32_impl(
    s: &Wtf8,
    is32: bool,
    big_endian: bool,
    bom: bool,
    codec: &str,
    w_object: PyObjectRef,
    errors: &str,
) -> Result<Vec<u8>, crate::PyError> {
    let mut out = Vec::new();
    if bom {
        emit_scalar(&mut out, 0xFEFF, is32, big_endian);
    }
    let cps: Vec<u32> = s.code_points().map(|c| c.to_u32()).collect();
    let mut i = 0usize;
    while i < cps.len() {
        let code = cps[i];
        if !(0xD800..=0xDFFF).contains(&code) {
            emit_scalar(&mut out, code, is32, big_endian);
            i += 1;
            continue;
        }
        let index = i;
        // Lone surrogate — only the utf-8/16/32 surrogatepass branch may
        // emit it, as a raw code unit (interp_codecs.py surrogatepass).
        match errors {
            "surrogatepass" => {
                if is32 {
                    push_unit32(&mut out, code, big_endian);
                } else {
                    push_unit16(&mut out, code as u16, big_endian);
                }
            }
            // surrogateescape rescues a 0xDC80..0xDCFF surrogate to the byte
            // code-0xDC00; any other surrogate still raises.
            "surrogateescape" if (0xDC80..=0xDCFF).contains(&code) => {
                out.push((code - 0xDC00) as u8);
            }
            "ignore" => {}
            "replace" => emit_scalar(&mut out, '?' as u32, is32, big_endian),
            "backslashreplace" => {
                for b in format!("\\u{code:04x}").bytes() {
                    emit_scalar(&mut out, b as u32, is32, big_endian);
                }
            }
            "xmlcharrefreplace" => {
                for b in format!("&#{code};").bytes() {
                    emit_scalar(&mut out, b as u32, is32, big_endian);
                }
            }
            "strict" | "surrogateescape" => {
                return Err(crate::typedef::unicode_encode_error(
                    codec,
                    w_object,
                    index,
                    index + 1,
                    "surrogates not allowed",
                ));
            }
            _ => {
                let (rep, newpos) = call_registered_encode_error_handler(
                    errors,
                    codec,
                    w_object,
                    cps.len(),
                    index,
                    index + 1,
                    "surrogates not allowed",
                )?;
                match rep {
                    EncodeReplacement::Str(rcps) => {
                        for rc in rcps {
                            if rc >= 0x80 {
                                return Err(crate::typedef::unicode_encode_error(
                                    codec,
                                    w_object,
                                    index,
                                    index + 1,
                                    "surrogates not allowed",
                                ));
                            }
                            emit_scalar(&mut out, rc, is32, big_endian);
                        }
                    }
                    EncodeReplacement::Bytes(b) => {
                        let unit = if is32 { 4 } else { 2 };
                        if b.len() % unit != 0 {
                            return Err(crate::typedef::unicode_encode_error(
                                codec,
                                w_object,
                                index,
                                index + 1,
                                "surrogates not allowed",
                            ));
                        }
                        out.extend_from_slice(&b);
                    }
                }
                i = newpos;
                continue;
            }
        }
        i += 1;
    }
    Ok(out)
}

/// utf-16 / utf-32 decode for the `lower`-normalized codec name, or
/// `None` if `lower` names neither.  The bare `utf-16` / `utf-32` forms
/// consume a leading BOM to choose endianness (defaulting to
/// little-endian); the `-le` / `-be` forms are fixed.  A lone surrogate
/// is routed through `err_mode` (`surrogatepass` keeps it as a code
/// point; `strict` raises), so the result is a `Wtf8Buf`.
pub fn decode_utf16_32(
    data: &[u8],
    lower: &str,
    err_mode: &str,
) -> Option<Result<Wtf8Buf, crate::PyError>> {
    // `codec` is the canonical name reported in a UnicodeDecodeError.
    let (is32, fixed_be, codec) = match compact_codec_name(lower).as_str() {
        "utf16" | "u16" => (false, None, "utf-16"),
        "utf16le" => (false, Some(false), "utf-16-le"),
        "utf16be" => (false, Some(true), "utf-16-be"),
        "utf32" | "u32" => (true, None, "utf-32"),
        "utf32le" => (true, Some(false), "utf-32-le"),
        "utf32be" => (true, Some(true), "utf-32-be"),
        _ => return None,
    };
    Some(if is32 {
        decode_utf32_impl(data, fixed_be, codec, err_mode)
    } else {
        decode_utf16_impl(data, fixed_be, codec, err_mode)
    })
}

/// Resolve endianness and the body start offset: a fixed `-le`/`-be`
/// codec ignores any BOM, while the bare form consumes a leading BOM and
/// otherwise defaults to little-endian.
fn resolve_bom(data: &[u8], is32: bool, fixed_be: Option<bool>) -> (bool, usize) {
    match fixed_be {
        Some(be) => (be, 0),
        None if is32 && data.starts_with(&[0xFF, 0xFE, 0x00, 0x00]) => (false, 4),
        None if is32 && data.starts_with(&[0x00, 0x00, 0xFE, 0xFF]) => (true, 4),
        None if !is32 && data.starts_with(&[0xFF, 0xFE]) => (false, 2),
        None if !is32 && data.starts_with(&[0xFE, 0xFF]) => (true, 2),
        None => (false, 0),
    }
}

/// Read one `unit`-byte (2 or 4) code unit at `pos` in the given order.
fn read_code_unit(data: &[u8], pos: usize, unit: usize, big_endian: bool) -> u32 {
    if unit == 2 {
        let arr = [data[pos], data[pos + 1]];
        if big_endian {
            u16::from_be_bytes(arr) as u32
        } else {
            u16::from_le_bytes(arr) as u32
        }
    } else {
        let arr = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
        if big_endian {
            u32::from_be_bytes(arr)
        } else {
            u32::from_le_bytes(arr)
        }
    }
}

/// interp_codecs.py:33-108 `call_errorhandler` (decode branch): invoke a
/// custom handler registered through `_codecs.register_error` for a decode
/// error position.
/// Returns `(newpos, resume)`.  `newpos` is the byte position to resume
/// decoding at, folded against the length of the buffer decoding will
/// continue in.  `resume` is `Some(new_bytes)` when the handler replaced
/// `exc.object` with different bytes (the caller must rebind its working
/// buffer to them) and `None` when the object was left unchanged (the
/// caller keeps decoding the same slice, no allocation).
pub(crate) fn call_registered_decode_error_handler(
    err_mode: &str,
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    out: &mut Wtf8Buf,
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    let w_handler = crate::module::_codecs::lookup_registered_error(err_mode).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{err_mode}'"),
        )
    })?;

    let w_exc =
        crate::typedef::unicode_decode_error(codec, data, start, end, reason).to_exc_object();
    let w_res = crate::baseobjspace::call_function(w_handler, &[w_exc]);
    if w_res.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::type_error("error handler failed")));
    }

    if !unsafe { pyre_object::is_tuple(w_res) } || unsafe { pyre_object::w_tuple_len(w_res) } != 2 {
        return Err(crate::PyError::type_error(
            "decoding error handler must return (str, int) tuple",
        ));
    }
    let w_replace = unsafe { pyre_object::w_tuple_getitem(w_res, 0).unwrap() };
    let w_newpos = unsafe { pyre_object::w_tuple_getitem(w_res, 1).unwrap() };
    if !unsafe { pyre_object::is_str(w_replace) } {
        return Err(crate::PyError::type_error(
            "decoding error handler must return (str, int) tuple",
        ));
    }

    // interp_codecs.py:94-104 — reread exc.object after the handler
    // returns.  The handler may have replaced it, in which case decoding
    // resumes from the new bytes.  A non-bytes object is rejected (the
    // decode C code checks PyBytes_Check on the reread object).
    let w_obj = unsafe { pyre_object::interp_exceptions::w_exception_get_object(w_exc) };
    if !unsafe { pyre_object::bytesobject::is_bytes(w_obj) } {
        return Err(crate::PyError::type_error(
            "UnicodeError 'object' attribute must be a bytes",
        ));
    }
    let new_bytes = unsafe { pyre_object::bytesobject::bytes_like_data(w_obj) };

    // newpos folds against the reread object's length (unicodeobject.c
    // insize), which the loop resumes into.
    let length = new_bytes.len() as i64;
    let mut newpos = match crate::baseobjspace::int_w(w_newpos) {
        Ok(n) => n,
        Err(e) => {
            if e.kind == crate::PyErrorKind::OverflowError {
                -1
            } else {
                return Err(e);
            }
        }
    };
    if newpos < 0 {
        newpos += length;
    }
    if newpos < 0 || newpos > length {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            format!("position {newpos} from error handler out of bounds"),
        ));
    }

    out.push_wtf8(unsafe { pyre_object::w_str_get_wtf8(w_replace) });
    let resume = if new_bytes == data {
        None
    } else {
        Some(new_bytes.to_vec())
    };
    Ok((newpos as usize, resume))
}

/// Replacement returned by a custom encode error handler: either a str
/// (its code points, re-encoded by the codec) or raw bytes (copied
/// verbatim). interp_codecs.py:69-72 rettype 'u' vs 'b'.
enum EncodeReplacement {
    Str(Vec<u32>),
    Bytes(Vec<u8>),
}

/// interp_codecs.py:33-108 `call_errorhandler` (encode branch): invoke a
/// custom handler registered through `_codecs.register_error` for an
/// encode error span. `start`/`end`/`char_len` are CHARACTER (code-point)
/// indices into the source; the returned position is a character index to
/// resume at. The caller re-encodes an `EncodeReplacement::Str` through
/// its own codec (raising the ORIGINAL error if a replacement code point
/// is not encodable) and copies `EncodeReplacement::Bytes` verbatim.
fn call_registered_encode_error_handler(
    err_mode: &str,
    codec: &str,
    source: PyObjectRef,
    char_len: usize,
    start: usize,
    end: usize,
    reason: &str,
) -> Result<(EncodeReplacement, usize), crate::PyError> {
    let w_handler = crate::module::_codecs::lookup_registered_error(err_mode).ok_or_else(|| {
        crate::PyError::new(
            crate::PyErrorKind::LookupError,
            format!("unknown error handler name '{err_mode}'"),
        )
    })?;

    let w_exc =
        crate::typedef::unicode_encode_error(codec, source, start, end, reason).to_exc_object();
    let w_res = crate::baseobjspace::call_function(w_handler, &[w_exc]);
    if w_res.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| crate::PyError::type_error("error handler failed")));
    }

    if !unsafe { pyre_object::is_tuple(w_res) } || unsafe { pyre_object::w_tuple_len(w_res) } != 2 {
        return Err(crate::PyError::type_error(
            "encoding error handler must return (str/bytes, int) tuple",
        ));
    }
    let w_replace = unsafe { pyre_object::w_tuple_getitem(w_res, 0).unwrap() };
    let w_newpos = unsafe { pyre_object::w_tuple_getitem(w_res, 1).unwrap() };

    let replacement = if unsafe { pyre_object::is_str(w_replace) } {
        EncodeReplacement::Str(
            unsafe { pyre_object::w_str_get_wtf8(w_replace) }
                .code_points()
                .map(|c| c.to_u32())
                .collect(),
        )
    } else if unsafe { pyre_object::bytesobject::is_bytes(w_replace) } {
        EncodeReplacement::Bytes(
            unsafe { pyre_object::bytesobject::bytes_like_data(w_replace) }.to_vec(),
        )
    } else {
        return Err(crate::PyError::type_error(
            "encoding error handler must return (str/bytes, int) tuple",
        ));
    };

    // newpos folds against the source CHARACTER length and resumes into the
    // original code-point sequence.
    let length = char_len as i64;
    let mut newpos = match crate::baseobjspace::int_w(w_newpos) {
        Ok(n) => n,
        Err(e) => {
            if e.kind == crate::PyErrorKind::OverflowError {
                -1
            } else {
                return Err(e);
            }
        }
    };
    if newpos < 0 {
        newpos += length;
    }
    if newpos < 0 || newpos > length {
        return Err(crate::PyError::new(
            crate::PyErrorKind::IndexError,
            format!("position {newpos} from error handler out of bounds"),
        ));
    }

    Ok((replacement, newpos as usize))
}

/// Decode error-handler dispatch for utf-16 / utf-32 (interp_codecs.py
/// surrogatepass/surrogateescape branches plus the generic handlers).
/// Appends the replacement to `out` and returns the byte position to
/// resume decoding at.  `unit` is 2 for utf-16, 4 for utf-32.
fn utf16_32_decode_error(
    err_mode: &str,
    codec: &str,
    data: &[u8],
    start: usize,
    end: usize,
    reason: &str,
    big_endian: bool,
    unit: usize,
    out: &mut Wtf8Buf,
) -> Result<(usize, Option<Vec<u8>>), crate::PyError> {
    match err_mode {
        "strict" => Err(crate::typedef::unicode_decode_error(
            codec, data, start, end, reason,
        )),
        "ignore" => Ok((end, None)),
        "replace" => {
            out.push_char('\u{FFFD}');
            Ok((end, None))
        }
        "backslashreplace" => {
            for &b in &data[start..end.min(data.len())] {
                out.push_str(&format!("\\x{b:02x}"));
            }
            Ok((end, None))
        }
        // surrogatepass: reconstruct one surrogate from the unit at
        // `start` and keep it; a non-surrogate value re-raises
        // (interp_codecs.py:476-510).
        "surrogatepass" => {
            if start + unit <= data.len() {
                let ch = read_code_unit(data, start, unit, big_endian);
                if (0xD800..=0xDFFF).contains(&ch) {
                    out.push(CodePoint::from_u32(ch).unwrap());
                    return Ok((start + unit, None));
                }
            }
            Err(crate::typedef::unicode_decode_error(
                codec, data, start, end, reason,
            ))
        }
        // surrogateescape: escape each >=128 byte as 0xdc00+byte, up to 4
        // bytes or the first ASCII byte (interp_codecs.py:536-555).
        "surrogateescape" => {
            let mut consumed = 0usize;
            while consumed < 4 && start + consumed < end {
                let b = data[start + consumed];
                if b < 128 {
                    break;
                }
                out.push(CodePoint::from_u32(0xDC00 + b as u32).unwrap());
                consumed += 1;
            }
            if consumed == 0 {
                return Err(crate::typedef::unicode_decode_error(
                    codec, data, start, end, reason,
                ));
            }
            Ok((start + consumed, None))
        }
        _ => call_registered_decode_error_handler(err_mode, codec, data, start, end, reason, out),
    }
}

/// `unicodehelper.py str_decode_utf_16_helper` (runicode.py:517).
fn decode_utf16_impl(
    data: &[u8],
    fixed_be: Option<bool>,
    codec: &str,
    err_mode: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let (big_endian, mut pos) = resolve_bom(data, false, fixed_be);
    // A custom error handler may replace exc.object; the loop then resumes
    // from the new bytes (`buf`), re-evaluating `len` each iteration.
    let mut buf: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(data);
    let mut len = buf.len();
    let mut out = Wtf8Buf::with_capacity(len / 2);
    // Run a utf-16 error handler and rebind `buf`/`len` when it returns
    // replacement bytes; evaluates to the resume position.
    macro_rules! run16 {
        ($start:expr, $end:expr, $reason:expr) => {{
            let (np, nb) = utf16_32_decode_error(
                err_mode, codec, &buf, $start, $end, $reason, big_endian, 2, &mut out,
            )?;
            if let Some(b) = nb {
                buf = std::borrow::Cow::Owned(b);
                len = buf.len();
            }
            np
        }};
    }
    while pos < len {
        if len - pos < 2 {
            pos = run16!(pos, len, "truncated data");
            if len - pos < 2 {
                break;
            }
            continue;
        }
        let ch = read_code_unit(&buf, pos, 2, big_endian);
        pos += 2;
        if !(0xD800..=0xDFFF).contains(&ch) {
            out.push(CodePoint::from_u32(ch).unwrap());
            continue;
        } else if ch >= 0xDC00 {
            // unexpected lone low surrogate
            pos = run16!(pos - 2, pos, "illegal encoding");
            continue;
        }
        // high surrogate: a low surrogate must follow
        if len - pos < 2 {
            pos -= 2;
            pos = run16!(pos, len, "unexpected end of data");
        } else {
            let ch2 = read_code_unit(&buf, pos, 2, big_endian);
            pos += 2;
            if (0xDC00..=0xDFFF).contains(&ch2) {
                let c = (((ch & 0x3FF) << 10) | (ch2 & 0x3FF)) + 0x10000;
                out.push(CodePoint::from_u32(c).unwrap());
            } else {
                pos = run16!(pos - 4, pos - 2, "illegal UTF-16 surrogate");
            }
        }
    }
    Ok(out)
}

/// `unicodehelper.py str_decode_utf_32_helper` (runicode.py:762).  The
/// public codec rejects surrogates (`allow_surrogates=False`), so a
/// surrogate code point is routed through the error handler.
fn decode_utf32_impl(
    data: &[u8],
    fixed_be: Option<bool>,
    codec: &str,
    err_mode: &str,
) -> Result<Wtf8Buf, crate::PyError> {
    let (big_endian, mut pos) = resolve_bom(data, true, fixed_be);
    // A custom error handler may replace exc.object; the loop then resumes
    // from the new bytes (`buf`), re-evaluating `len` each iteration.
    let mut buf: std::borrow::Cow<[u8]> = std::borrow::Cow::Borrowed(data);
    let mut len = buf.len();
    let mut out = Wtf8Buf::with_capacity(len / 4);
    macro_rules! run32 {
        ($start:expr, $end:expr, $reason:expr) => {{
            let (np, nb) = utf16_32_decode_error(
                err_mode, codec, &buf, $start, $end, $reason, big_endian, 4, &mut out,
            )?;
            if let Some(b) = nb {
                buf = std::borrow::Cow::Owned(b);
                len = buf.len();
            }
            np
        }};
    }
    while pos < len {
        if len - pos < 4 {
            pos = run32!(pos, len, "truncated data");
            if len - pos < 4 {
                break;
            }
            continue;
        }
        let ch = read_code_unit(&buf, pos, 4, big_endian);
        if (0xD800..=0xDFFF).contains(&ch) {
            pos = run32!(
                pos,
                pos + 4,
                "code point in surrogate code point range(0xd800, 0xe000)"
            );
            continue;
        } else if ch >= 0x110000 {
            pos = run32!(pos, len, "code point not in range(0x110000)");
            continue;
        }
        out.push(CodePoint::from_u32(ch).unwrap());
        pos += 4;
    }
    Ok(out)
}

/// Map each scalar code point of `s` through `f`, appending to a
/// `Wtf8Buf`; a lone surrogate passes through unchanged.  Used by the
/// case-mapping methods, which leave surrogates untouched.
/// Apply a whole-string case transform (`str::to_lowercase` / `to_uppercase`)
/// to each maximal valid-UTF-8 run of `s`, passing lone surrogates through
/// unchanged.  Operating on the run rather than each scalar preserves the
/// context rules those transforms encode — notably the Greek Final_Sigma
/// (`Σ` → `ς` word-finally, `σ` elsewhere).
fn wtf8_map_str_runs(s: &Wtf8, f: impl Fn(&str) -> String) -> Wtf8Buf {
    let mut out = Wtf8Buf::with_capacity(s.len());
    let mut run = String::new();
    for cp in s.code_points() {
        match cp.to_char() {
            Some(c) => run.push(c),
            None => {
                if !run.is_empty() {
                    out.push_str(&f(&run));
                    run.clear();
                }
                out.push(cp);
            }
        }
    }
    if !run.is_empty() {
        out.push_str(&f(&run));
    }
    out
}

pub fn str_method_isdigit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isdigit")?;
    // A lone surrogate satisfies no character class, so a non-UTF-8
    // backing is never all-digit (and the empty string is false too).
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_digit),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_isdecimal(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isdecimal")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_decimal),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_isnumeric(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isnumeric")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_numeric),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

pub fn str_method_istitle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "istitle")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let mut cased = false;
    let mut prev_cased = false;
    for cp in s.code_points() {
        // A lone surrogate is uncased, so it resets `prev_cased` like
        // any other non-cased code point (the `None` arm / `else`).
        match cp.to_char() {
            Some(c) => {
                if c.is_uppercase() || case::is_titlecase(c) {
                    if prev_cased {
                        return Ok(w_bool_from(false));
                    }
                    prev_cased = true;
                    cased = true;
                } else if c.is_lowercase() {
                    if !prev_cased {
                        return Ok(w_bool_from(false));
                    }
                    prev_cased = true;
                    cased = true;
                } else {
                    prev_cased = false;
                }
            }
            None => prev_cased = false,
        }
    }
    Ok(w_bool_from(cased))
}

pub fn str_method_isalpha(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isalpha")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_alpha),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isidentifier
pub fn str_method_isidentifier(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isidentifier")?;
    // An identifier cannot contain a lone surrogate, so a non-UTF-8
    // backing is never an identifier.
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => is_identifier(v),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// Check if a string is a valid Python identifier: an `XID_Start` code point
/// (or underscore) followed by `XID_Continue` code points.
/// PyPy: unicodeobject.py `_isidentifier` via `unicodedb.isxidstart`/`isxidcontinue`.
fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    let valid_start = chars.next().is_some_and(identifier::is_start);
    valid_start && chars.all(identifier::is_continue)
}

/// `pypy/objspace/std/unicodeobject.py W_UnicodeObject.descr_zfill`.
/// Pads with leading zeros up to `width`; when the string starts with
/// a sign character (`+`/`-`), the sign stays at the front and zeros
/// fill between it and the digits (`'-42'.zfill(5) == '-0042'`).
pub fn str_method_zfill(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "str.zfill", 1)?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?.max(0) as usize;
    let len = s.code_points().count();
    if len >= width {
        return Ok(str_result_unchanged(args[0]));
    }
    let need = width - len;
    let mut cps = s.code_points();
    let mut out = Wtf8Buf::with_capacity(width);
    let first = cps.clone().next();
    if first == Some(CodePoint::from_char('+')) || first == Some(CodePoint::from_char('-')) {
        out.push(first.unwrap());
        cps.next();
    }
    for _ in 0..need {
        out.push_char('0');
    }
    for cp in cps {
        out.push(cp);
    }
    Ok(w_str_from_wtf8(out))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SearchMode {
    Count,
    Find,
    RFind,
}

const TWOWAY_MAX_SHIFT: usize = 255;
const TWOWAY_TABLE_SIZE: usize = 64;
const TWOWAY_TABLE_MASK: usize = TWOWAY_TABLE_SIZE - 1;

#[inline]
fn rstring_bloom_add(mask: u64, c: u8) -> u64 {
    // RPython `rstring.py:bloom_add`, with LONG_BIT = 64 on this target.
    mask | (1u64 << (c & 63))
}

#[inline]
fn rstring_bloom(mask: u64, c: u8) -> bool {
    // RPython `rstring.py:bloom`.
    (mask & (1u64 << (c & 63))) != 0
}

fn rstring_lex_search(needle: &[u8], len_needle: usize, invert_alphabet: bool) -> (usize, usize) {
    // RPython `rstring.py:_lex_search`.
    let mut max_suffix = 0usize;
    let mut candidate = 1usize;
    let mut k = 0usize;
    let mut period = 1usize;
    while candidate + k < len_needle {
        let a = needle[candidate + k];
        let b = needle[max_suffix + k];
        if if invert_alphabet { b < a } else { a < b } {
            candidate += k + 1;
            k = 0;
            period = candidate - max_suffix;
        } else if a == b {
            if k + 1 != period {
                k += 1;
            } else {
                candidate += period;
                k = 0;
            }
        } else {
            max_suffix = candidate;
            candidate += 1;
            k = 0;
            period = 1;
        }
    }
    (max_suffix, period)
}

fn rstring_factorize(needle: &[u8], len_needle: usize) -> (usize, usize) {
    // RPython `rstring.py:_factorize`.
    let (cut1, period1) = rstring_lex_search(needle, len_needle, false);
    let (cut2, period2) = rstring_lex_search(needle, len_needle, true);
    if cut1 > cut2 {
        (cut1, period1)
    } else {
        (cut2, period2)
    }
}

fn rstring_twoway_preprocess(
    needle: &[u8],
    len_needle: usize,
) -> (usize, usize, usize, bool, [u8; TWOWAY_TABLE_SIZE]) {
    // RPython `rstring.py:_twoway_preprocess`.
    let (cut, mut period) = rstring_factorize(needle, len_needle);
    let mut is_periodic = true;
    let mut i = 0usize;
    while i < cut {
        if needle[i] != needle[period + i] {
            is_periodic = false;
            break;
        }
        i += 1;
    }
    let gap = if is_periodic {
        0
    } else {
        period = cut.max(len_needle - cut) + 1;
        let mut gap = len_needle;
        let last = (needle[len_needle - 1] as usize) & TWOWAY_TABLE_MASK;
        let mut i = len_needle - 1;
        while i > 0 {
            i -= 1;
            if ((needle[i] as usize) & TWOWAY_TABLE_MASK) == last {
                gap = len_needle - 1 - i;
                break;
            }
        }
        gap
    };
    let not_found_shift = len_needle.min(TWOWAY_MAX_SHIFT) as u8;
    let mut table = [not_found_shift; TWOWAY_TABLE_SIZE];
    let mut i = len_needle - not_found_shift as usize;
    while i < len_needle {
        table[(needle[i] as usize) & TWOWAY_TABLE_MASK] = (len_needle - 1 - i) as u8;
        i += 1;
    }
    (cut, period, gap, is_periodic, table)
}

fn rstring_two_way(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    cut: usize,
    mut period: usize,
    gap: usize,
    is_periodic: bool,
    table: &[u8; TWOWAY_TABLE_SIZE],
) -> isize {
    // RPython `rstring.py:_two_way`.
    let haystack_end = base + n;
    let mut window_last = base + m - 1;
    if is_periodic {
        let mut memory = 0usize;
        let mut skip_horspool = false;
        while window_last < haystack_end {
            if !skip_horspool {
                loop {
                    let shift = table[(value[window_last] as usize) & TWOWAY_TABLE_MASK] as usize;
                    window_last += shift;
                    if shift == 0 {
                        break;
                    }
                    if window_last >= haystack_end {
                        return -1;
                    }
                }
            }
            skip_horspool = false;
            let window = window_last + 1 - m;
            let mut i = cut.max(memory);
            let mut mismatch = false;
            while i < m {
                if needle[i] != value[window + i] {
                    window_last += i - cut + 1;
                    memory = 0;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = memory;
            while i < cut {
                if needle[i] != value[window + i] {
                    window_last += period;
                    memory = m - period;
                    if window_last >= haystack_end {
                        return -1;
                    }
                    let shift = table[(value[window_last] as usize) & TWOWAY_TABLE_MASK] as usize;
                    if shift != 0 {
                        let mem_jump = cut.max(memory) - cut + 1;
                        memory = 0;
                        window_last += shift.max(mem_jump);
                    } else {
                        skip_horspool = true;
                    }
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            return (window - base) as isize;
        }
        -1
    } else {
        if period < gap {
            period = gap;
        }
        let gap_jump_end = (cut + gap).min(m);
        while window_last < haystack_end {
            loop {
                let shift = table[(value[window_last] as usize) & TWOWAY_TABLE_MASK] as usize;
                window_last += shift;
                if shift == 0 {
                    break;
                }
                if window_last >= haystack_end {
                    return -1;
                }
            }
            let window = window_last + 1 - m;
            let mut mismatch = false;
            let mut i = cut;
            while i < gap_jump_end {
                if needle[i] != value[window + i] {
                    window_last += gap;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = gap_jump_end;
            while i < m {
                if needle[i] != value[window + i] {
                    window_last += i - cut + 1;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            i = 0;
            while i < cut {
                if needle[i] != value[window + i] {
                    window_last += period;
                    mismatch = true;
                    break;
                }
                i += 1;
            }
            if mismatch {
                continue;
            }
            return (window - base) as isize;
        }
        -1
    }
}

fn rstring_two_way_count(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    cut: usize,
    period: usize,
    gap: usize,
    is_periodic: bool,
    table: &[u8; TWOWAY_TABLE_SIZE],
) -> usize {
    // RPython `rstring.py:_two_way_count`.
    let mut index = 0usize;
    let mut count = 0usize;
    loop {
        let result = rstring_two_way(
            value,
            base + index,
            n - index,
            needle,
            m,
            cut,
            period,
            gap,
            is_periodic,
            table,
        );
        if result == -1 {
            return count;
        }
        count += 1;
        index += result as usize + m;
    }
}

fn rstring_default_find(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    mode: SearchMode,
) -> isize {
    // RPython `rstring.py:_default_find`.
    let w = n - m;
    let mlast = m - 1;
    let mut count = 0usize;
    let mut gap = mlast;
    let last = needle[mlast];
    let mut mask = 0u64;
    let mut j = 0usize;
    while j < mlast {
        mask = rstring_bloom_add(mask, needle[j]);
        if needle[j] == last {
            gap = mlast - j - 1;
        }
        j += 1;
    }
    mask = rstring_bloom_add(mask, last);
    let mut i = 0usize;
    while i <= w {
        if value[base + mlast + i] == last {
            j = 0;
            while j < mlast {
                if value[base + i + j] != needle[j] {
                    break;
                }
                j += 1;
            }
            if j == mlast {
                if mode != SearchMode::Count {
                    return i as isize;
                }
                count += 1;
                i += mlast;
            } else {
                let la = base + mlast + i + 1;
                let c = if la < value.len() { value[la] } else { 0 };
                if !rstring_bloom(mask, c) {
                    i += m;
                } else {
                    i += gap;
                }
            }
        } else {
            let la = base + mlast + i + 1;
            let c = if la < value.len() { value[la] } else { 0 };
            if !rstring_bloom(mask, c) {
                i += m;
            }
        }
        i += 1;
    }
    if mode != SearchMode::Count {
        -1
    } else {
        count as isize
    }
}

fn rstring_adaptive_find(
    value: &[u8],
    base: usize,
    n: usize,
    needle: &[u8],
    m: usize,
    mode: SearchMode,
) -> isize {
    // RPython `rstring.py:_adaptive_find`.
    let w = n - m;
    let mlast = m - 1;
    let mut count = 0usize;
    let mut gap = mlast;
    let mut hits = 0usize;
    let last = needle[mlast];
    let mut mask = 0u64;
    let mut j = 0usize;
    while j < mlast {
        mask = rstring_bloom_add(mask, needle[j]);
        if needle[j] == last {
            gap = mlast - j - 1;
        }
        j += 1;
    }
    mask = rstring_bloom_add(mask, last);
    let mut i = 0usize;
    while i <= w {
        if value[base + mlast + i] == last {
            j = 0;
            while j < mlast {
                if value[base + i + j] != needle[j] {
                    break;
                }
                j += 1;
            }
            if j == mlast {
                if mode != SearchMode::Count {
                    return i as isize;
                }
                count += 1;
                i += mlast;
            } else {
                hits += j + 1;
                if hits > m / 4 && w - i > 2000 {
                    let (cut, period, gap, is_periodic, table) =
                        rstring_twoway_preprocess(needle, m);
                    if mode != SearchMode::Count {
                        let res = rstring_two_way(
                            value,
                            base + i,
                            n - i,
                            needle,
                            m,
                            cut,
                            period,
                            gap,
                            is_periodic,
                            &table,
                        );
                        return if res == -1 { -1 } else { res + i as isize };
                    }
                    let res = rstring_two_way_count(
                        value,
                        base + i,
                        n - i,
                        needle,
                        m,
                        cut,
                        period,
                        gap,
                        is_periodic,
                        &table,
                    );
                    return (res + count) as isize;
                }
                let la = base + mlast + i + 1;
                let c = if la < value.len() { value[la] } else { 0 };
                if !rstring_bloom(mask, c) {
                    i += m;
                } else {
                    i += gap;
                }
            }
        } else {
            let la = base + mlast + i + 1;
            let c = if la < value.len() { value[la] } else { 0 };
            if !rstring_bloom(mask, c) {
                i += m;
            }
        }
        i += 1;
    }
    if mode != SearchMode::Count {
        -1
    } else {
        count as isize
    }
}

fn rstring_search_normal(
    value: &[u8],
    other: &[u8],
    mut start: usize,
    mut end: usize,
    mode: SearchMode,
) -> isize {
    // RPython `rstring.py:_search_normal`, specialized to byte-backed
    // PyPy unicode `_utf8` / pyre WTF-8 storage.
    end = end.min(value.len());
    start = start.min(end);
    let n = end - start;
    let m = other.len();
    if m == 0 {
        return match mode {
            SearchMode::Count => (end - start + 1) as isize,
            SearchMode::RFind => end as isize,
            SearchMode::Find => start as isize,
        };
    }
    let Some(w) = n.checked_sub(m) else {
        return if mode == SearchMode::Count { 0 } else { -1 };
    };
    if mode != SearchMode::RFind {
        let res = if n < 2500 || (m < 100 && n < 30000) || m < 6 {
            rstring_default_find(value, start, n, other, m, mode)
        } else if (m >> 2) * 3 < (n >> 2) {
            let (cut, period, gap, is_periodic, table) = rstring_twoway_preprocess(other, m);
            if mode == SearchMode::Count {
                return rstring_two_way_count(
                    value,
                    start,
                    n,
                    other,
                    m,
                    cut,
                    period,
                    gap,
                    is_periodic,
                    &table,
                ) as isize;
            }
            rstring_two_way(
                value,
                start,
                n,
                other,
                m,
                cut,
                period,
                gap,
                is_periodic,
                &table,
            )
        } else {
            rstring_adaptive_find(value, start, n, other, m, mode)
        };
        if mode == SearchMode::Count {
            res
        } else if res == -1 {
            -1
        } else {
            start as isize + res
        }
    } else {
        // RPython `rstring.py:_search_normal` reverse-find branch.
        let mlast = m - 1;
        let mut skip = mlast;
        let mut mask = rstring_bloom_add(0, other[0]);
        let mut i = mlast;
        while i > 0 {
            mask = rstring_bloom_add(mask, other[i]);
            if other[i] == other[0] {
                skip = i - 1;
            }
            i -= 1;
        }
        let mut i = start + w + 1;
        while i > start {
            i -= 1;
            if value[i] == other[0] {
                let mut matched = true;
                let mut j = mlast;
                while j > 0 {
                    if value[i + j] != other[j] {
                        matched = false;
                        break;
                    }
                    j -= 1;
                }
                if matched {
                    return i as isize;
                }
                if i > 0 && !rstring_bloom(mask, value[i - 1]) {
                    i = i.saturating_sub(m);
                } else {
                    i = i.saturating_sub(skip);
                }
            } else if i > 0 && !rstring_bloom(mask, value[i - 1]) {
                i = i.saturating_sub(m);
            }
        }
        -1
    }
}

/// Number of non-overlapping occurrences of `needle` in `haystack`,
/// scanning over the WTF-8 bytes. PyPy's `unicodeobject.py:1116` delegates
/// to `_utf8.count`; the RPython translation path for non-host strings is
/// `rstring.py:_search_normal(..., SEARCH_COUNT)`.
fn wtf8_count(haystack: &Wtf8, needle: &Wtf8) -> usize {
    if needle.is_empty() {
        return haystack.code_points().count() + 1;
    }
    rstring_search_normal(
        haystack.as_bytes(),
        needle.as_bytes(),
        0,
        haystack.len(),
        SearchMode::Count,
    ) as usize
}

/// First byte offset of `needle` fully within `haystack[lo..hi]`, over
/// WTF-8 bytes. PyPy `unicodeobject.py:_unwrap_and_search` searches the
/// `_utf8` storage after converting codepoint bounds to byte bounds.
fn wtf8_find_bounded(haystack: &[u8], needle: &[u8], lo: usize, hi: usize) -> Option<usize> {
    let res = rstring_search_normal(haystack, needle, lo, hi, SearchMode::Find);
    if res == -1 { None } else { Some(res as usize) }
}

/// Last byte offset of `needle` fully within `haystack[lo..hi]`, over
/// WTF-8 bytes. Mirrors `rstring.py:_search_normal(..., SEARCH_RFIND)`.
fn wtf8_rfind_bounded(haystack: &[u8], needle: &[u8], lo: usize, hi: usize) -> Option<usize> {
    let res = rstring_search_normal(haystack, needle, lo, hi, SearchMode::RFind);
    if res == -1 { None } else { Some(res as usize) }
}

/// PyPy `_unwrap_and_search` (unicodeobject.py:1288-1317) — the shared
/// path for find/rfind/index/rindex. `start`/`end` (args[2]/args[3])
/// are codepoint indices: `unwrap_start_stop` adds the length to a
/// negative value and lower-clamps to 0. The search runs over the
/// WTF-8 bytes inside that window and the byte offset is converted back
/// to a codepoint index. Returns None when not found.
/// `unicodeobject.py:1288 _unwrap_and_search` — the shared path for
/// find/rfind/index/rindex. `start`/`end` (args[2]/args[3]) flow through
/// `unwrap_start_stop`, so `None` / omitted arguments default, any
/// `__index__`-bearing object is accepted, and a `TypeError` propagates.
/// The search runs over the WTF-8 bytes inside the codepoint window and
/// the matching byte offset is mapped back to a codepoint index. Returns
/// `Ok(None)` when not found.
fn str_unwrap_and_search(
    args: &[PyObjectRef],
    forward: bool,
) -> Result<Option<i64>, crate::PyError> {
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let sub = unsafe { pyre_object::w_str_get_wtf8(args[1]) }.as_bytes();
    let h = s.as_bytes();
    // Byte offset of each codepoint, with the trailing length appended,
    // so `cp_offsets[i]` is `_index_to_byte(i)` and a byte offset maps
    // back via its position.
    let mut cp_offsets: Vec<usize> = s.code_point_indices().map(|(i, _)| i).collect();
    cp_offsets.push(h.len());
    let length = (cp_offsets.len() - 1) as i64;

    let w_start = if args.len() >= 3 { args[2] } else { w_none() };
    let w_end = if args.len() >= 4 { args[3] } else { w_none() };
    let (start, end) = crate::sliceobject::unwrap_start_stop(length, w_start, w_end)?;

    let start_index = if start == 0 {
        0
    } else if start > length {
        return Ok(None);
    } else {
        cp_offsets[start as usize]
    };
    let end_index = if end >= length {
        h.len()
    } else {
        cp_offsets[end as usize]
    };
    if start_index > end_index {
        return Ok(None);
    }

    let res_index = if forward {
        wtf8_find_bounded(h, sub, start_index, end_index)
    } else {
        wtf8_rfind_bounded(h, sub, start_index, end_index)
    };
    Ok(res_index.and_then(|ri| cp_offsets.iter().position(|&o| o == ri).map(|i| i as i64)))
}

/// PyPy: unicodeobject.py descr_count
pub fn str_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "count", 1)?;
    arity_at_most(args, "count", 3)?;
    require_str_sub(args, "count")?;
    // Operands read as WTF-8 so lone surrogates do not panic; the optional
    // start / end arguments bound the count window over the code points.
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) };
    let sub = unsafe { pyre_object::w_str_get_wtf8(args[1]) };
    let Some((byte_start, byte_end)) = wtf8_idx_window(s, args)? else {
        return Ok(w_int_new(0));
    };
    let window = rustpython_wtf8::Wtf8::from_bytes(&s.as_bytes()[byte_start..byte_end])
        .expect("code-point boundary slice is valid WTF-8");
    Ok(w_int_new(wtf8_count(window, sub) as i64))
}

/// PyPy: unicodeobject.py descr_index
/// `unicodeobject.py:1006-1010 _descr_index` — missing substring raises
/// "substring not found" (ValueError).
pub fn str_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "index", 1)?;
    arity_at_most(args, "index", 3)?;
    require_str_sub(args, "index")?;
    match str_unwrap_and_search(args, true)? {
        Some(i) => Ok(w_int_new(i)),
        None => Err(crate::PyError::value_error("substring not found")),
    }
}

/// `unicodeobject.py descr_rindex` — like rfind, but raises ValueError
/// when the substring is absent.
/// unicodeobject.py:572 descr_rindex
pub fn str_method_rindex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "rindex", 1)?;
    arity_at_most(args, "rindex", 3)?;
    require_str_sub(args, "rindex")?;
    match str_unwrap_and_search(args, false)? {
        Some(i) => Ok(w_int_new(i)),
        None => Err(crate::PyError::value_error("substring not found")),
    }
}

/// PyPy: unicodeobject.py descr_title
pub fn str_method_title(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "title")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_str_from_wtf8(case::title_wtf8(s)))
}

/// PyPy: unicodeobject.py descr_capitalize
pub fn str_method_capitalize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "capitalize")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_str_from_wtf8(case::capitalize_wtf8(s)))
}

/// PyPy: unicodeobject.py descr_swapcase
pub fn str_method_swapcase(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "swapcase")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_str_from_wtf8(case::swapcase_wtf8(s)))
}

/// PyPy: unicodeobject.py descr_center
/// Resolve the fillchar arg for `center`/`ljust`/`rjust`. Defaults to
/// `' '` when missing; PyPy raises TypeError when the fill string is
/// not exactly one character long (unicodeobject.py:1191-1194
/// _convert_fillchar parity).
fn pad_fillchar(args: &[PyObjectRef], method: &str) -> Result<CodePoint, crate::PyError> {
    if args.len() <= 2 {
        return Ok(CodePoint::from_char(' '));
    }
    if !unsafe { pyre_object::is_str(args[2]) } {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 2 must be a single character"
        )));
    }
    let raw = unsafe { w_str_get_wtf8(args[2]) };
    let mut iter = raw.code_points();
    let first = iter.next();
    if first.is_none() || iter.next().is_some() {
        return Err(crate::PyError::type_error(format!(
            "{method}() argument 2 must be a single character"
        )));
    }
    Ok(first.unwrap())
}

/// Append `cp` to `out`, `n` times.
fn push_cp_repeated(out: &mut Wtf8Buf, cp: CodePoint, n: usize) {
    for _ in 0..n {
        out.push(cp);
    }
}

/// `unicode_result_unchanged`: a str method whose result equals the
/// receiver returns the receiver itself only when it is an exact `str`; a
/// `str` subclass is copied to a fresh base `str`, since str methods never
/// return a subclass instance.
pub(crate) fn str_result_unchanged(obj: PyObjectRef) -> PyObjectRef {
    if unsafe { is_exact_type(obj, &STR_TYPE) } {
        obj
    } else {
        w_str_from_wtf8(unsafe { w_str_get_wtf8(obj) }.to_owned())
    }
}

pub fn str_method_center(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "center", 1)?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?.max(0) as usize;
    let fillchar = pad_fillchar(args, "center")?;
    let s_len = s.code_points().count();
    if s_len >= width {
        return Ok(str_result_unchanged(args[0]));
    }
    // unicodeobject.py:1098 d = (width - len) ; lpad = d//2 + (d & width & 1)
    let d = width - s_len;
    let left = d / 2 + (d & width & 1);
    let right = d - left;
    let mut out = Wtf8Buf::with_capacity(s.len() + (left + right) * 4);
    push_cp_repeated(&mut out, fillchar, left);
    out.push_wtf8(s);
    push_cp_repeated(&mut out, fillchar, right);
    Ok(w_str_from_wtf8(out))
}

/// PyPy: unicodeobject.py descr_ljust
pub fn str_method_ljust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "ljust", 1)?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?.max(0) as usize;
    let fillchar = pad_fillchar(args, "ljust")?;
    let s_len = s.code_points().count();
    if s_len >= width {
        return Ok(str_result_unchanged(args[0]));
    }
    let mut out = Wtf8Buf::with_capacity(s.len() + (width - s_len) * 4);
    out.push_wtf8(s);
    push_cp_repeated(&mut out, fillchar, width - s_len);
    Ok(w_str_from_wtf8(out))
}

/// PyPy: unicodeobject.py descr_rjust
pub fn str_method_rjust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "rjust", 1)?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let width = crate::builtins::space_index_w(args[1])?.max(0) as usize;
    let fillchar = pad_fillchar(args, "rjust")?;
    let s_len = s.code_points().count();
    if s_len >= width {
        return Ok(str_result_unchanged(args[0]));
    }
    let mut out = Wtf8Buf::with_capacity(s.len() + (width - s_len) * 4);
    push_cp_repeated(&mut out, fillchar, width - s_len);
    out.push_wtf8(s);
    Ok(w_str_from_wtf8(out))
}

/// `pypy/objspace/std/unicodeobject.py descr_isprintable` —
///
/// ```python
/// def descr_isprintable(self, space):
///     for ch in self._utf8:
///         if not unicodedb.isprintable(ord(ch)):
///             return space.w_False
///     return space.w_True
/// ```
///
/// Empty string returns True per CPython.  Delegates the per-character
/// category check to `classify::is_printable`.
pub fn str_method_isprintable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isprintable")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    // Empty returns True (vacuous); a lone surrogate is not printable.
    let result = match s.as_str() {
        Ok(v) => v.chars().all(classify::is_printable),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isspace
pub fn str_method_isspace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isspace")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_space),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// True iff `s` has at least one cased (alphabetic) code point and
/// every cased code point matches the requested case.  Lone
/// surrogates are uncased and ignored, so `'ABC\udcff'.isupper()` is
/// still True — unlike the character-class predicates, a surrogate
/// does not force a false result here.
fn wtf8_cased_all(s: &Wtf8, want_upper: bool) -> bool {
    let mut has_cased = false;
    let mut all_match = true;
    for cp in s.code_points() {
        if let Some(c) = cp.to_char() {
            if c.is_alphabetic() {
                has_cased = true;
                let ok = if want_upper {
                    c.is_uppercase()
                } else {
                    c.is_lowercase()
                };
                if !ok {
                    all_match = false;
                }
            }
        }
    }
    has_cased && all_match
}

/// PyPy: unicodeobject.py descr_isupper
pub fn str_method_isupper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isupper")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_bool_from(wtf8_cased_all(s, true)))
}

/// PyPy: unicodeobject.py descr_islower
pub fn str_method_islower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "islower")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    Ok(w_bool_from(wtf8_cased_all(s, false)))
}

/// PyPy: unicodeobject.py descr_isalnum
pub fn str_method_isalnum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isalnum")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => !v.is_empty() && v.chars().all(classify::is_alnum),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// PyPy: unicodeobject.py descr_isascii
pub fn str_method_isascii(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_no_args(args, "isascii")?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let result = match s.as_str() {
        Ok(v) => v.is_ascii(),
        Err(_) => false,
    };
    Ok(w_bool_from(result))
}

/// Builds an owned str from a byte sub-slice of a WTF-8 string. The
/// slice must start and end on code-point boundaries, which holds for
/// the partition cuts below (the separator aligns on boundaries).
fn wtf8_slice_str(bytes: &[u8]) -> PyObjectRef {
    let part = unsafe { Wtf8::from_bytes_unchecked(bytes) };
    w_str_from_wtf8(part.to_wtf8_buf())
}

/// Replaces up to `maxcount` occurrences of `sub` with `by` over the
/// WTF-8 bytes (rstring.py:220-309 `replace_count` isutf8 path). A
/// negative `maxcount` means no limit; an empty `sub` inserts `by` at
/// every code-point boundary, including the ends.
fn wtf8_replace(input: &Wtf8, sub: &Wtf8, by: &Wtf8, maxcount: i64) -> Wtf8Buf {
    if maxcount == 0 {
        return input.to_wtf8_buf();
    }
    let inp = input.as_bytes();
    let sub_b = sub.as_bytes();
    let mut out = Wtf8Buf::new();
    let mut start = 0usize;
    let mut maxcount = maxcount;
    if sub_b.is_empty() {
        let mut indices = input.code_point_indices().map(|(i, _)| i);
        // Skip the leading boundary at 0; it is handled by the first
        // `by` insertion before each code point.
        indices.next();
        loop {
            out.push_wtf8(by);
            maxcount -= 1;
            if start == inp.len() || maxcount == 0 {
                break;
            }
            let next = indices.next().unwrap_or(inp.len());
            out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..next]) });
            start = next;
        }
    } else {
        while maxcount != 0 {
            match wtf8_find_bounded(inp, sub_b, start, inp.len()) {
                Some(next) => {
                    out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..next]) });
                    out.push_wtf8(by);
                    start = next + sub_b.len();
                    maxcount -= 1;
                }
                None => break,
            }
        }
    }
    out.push_wtf8(unsafe { Wtf8::from_bytes_unchecked(&inp[start..]) });
    out
}

/// PyPy: unicodeobject.py descr_partition
pub fn str_method_partition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "str.partition", 1)?;
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "must be str, not {}",
            arg_type_name(args[1])
        )));
    }
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) }.as_bytes();
    let sep = unsafe { pyre_object::w_str_get_wtf8(args[1]) }.as_bytes();
    if sep.is_empty() {
        return Err(crate::PyError::value_error("empty separator"));
    }
    match wtf8_find_bounded(s, sep, 0, s.len()) {
        Some(i) => Ok(w_tuple_new(vec![
            wtf8_slice_str(&s[..i]),
            args[1],
            wtf8_slice_str(&s[i + sep.len()..]),
        ])),
        None => Ok(w_tuple_new(vec![args[0], w_str_new(""), w_str_new("")])),
    }
}

/// PyPy: unicodeobject.py descr_rpartition
pub fn str_method_rpartition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "str.rpartition", 1)?;
    if !unsafe { pyre_object::is_str(args[1]) } {
        return Err(crate::PyError::type_error(format!(
            "must be str, not {}",
            arg_type_name(args[1])
        )));
    }
    let s = unsafe { pyre_object::w_str_get_wtf8(args[0]) }.as_bytes();
    let sep = unsafe { pyre_object::w_str_get_wtf8(args[1]) }.as_bytes();
    if sep.is_empty() {
        return Err(crate::PyError::value_error("empty separator"));
    }
    match wtf8_rfind_bounded(s, sep, 0, s.len()) {
        Some(i) => Ok(w_tuple_new(vec![
            wtf8_slice_str(&s[..i]),
            args[1],
            wtf8_slice_str(&s[i + sep.len()..]),
        ])),
        None => Ok(w_tuple_new(vec![w_str_new(""), w_str_new(""), args[0]])),
    }
}

/// PyPy: unicodeobject.py descr_splitlines.
/// Walks the Unicode line-boundary set explicitly so that
/// `keepends=True` retains the terminator on each emitted line and a
/// trailing `\n` does NOT produce an extra empty entry — matching
/// `'a\nb\n'.splitlines() == ['a', 'b']`.
fn cp_is_linebreak(cp: CodePoint) -> bool {
    matches!(
        cp.to_char(),
        Some(
            '\n' | '\r'
                | '\u{000b}'
                | '\u{000c}'
                | '\u{001c}'
                | '\u{001d}'
                | '\u{001e}'
                | '\u{0085}'
                | '\u{2028}'
                | '\u{2029}'
        )
    )
}

pub fn str_method_splitlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "splitlines")?;
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "splitlines() takes at most 1 argument ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["keepends"], "splitlines")?;
    crate::builtins::kwarg_reject_duplicate(
        kwargs,
        "splitlines",
        "keepends",
        pos.get(1).is_some(),
    )?;
    let cps: Vec<CodePoint> = unsafe { w_str_get_wtf8(pos[0]) }.code_points().collect();
    // keepends is positional-or-keyword.
    let keepends = crate::builtins::kwarg_get(kwargs, "keepends")
        .or_else(|| pos.get(1).copied())
        .map(crate::baseobjspace::is_true)
        .transpose()?
        .unwrap_or(false);
    let mut parts: Vec<PyObjectRef> = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < cps.len() {
        if cp_is_linebreak(cps[i]) {
            let mut term_end = i + 1;
            if cps[i].to_char() == Some('\r')
                && term_end < cps.len()
                && cps[term_end].to_char() == Some('\n')
            {
                term_end += 1;
            }
            let end = if keepends { term_end } else { i };
            parts.push(cps_to_str(&cps[start..end]));
            start = term_end;
            i = term_end;
        } else {
            i += 1;
        }
    }
    if start < cps.len() {
        parts.push(cps_to_str(&cps[start..]));
    }
    Ok(w_list_new(parts))
}

/// PyPy: unicodeobject.py descr_removeprefix (Python 3.9+)
pub fn str_method_removeprefix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "str.removeprefix() takes exactly one argument ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    if !unsafe { pyre_object::is_str(pos[1]) } {
        return Err(crate::PyError::type_error(format!(
            "removeprefix() argument must be str, not {}",
            arg_type_name(pos[1])
        )));
    }
    let s = unsafe { w_str_get_wtf8(pos[0]) };
    let prefix = unsafe { w_str_get_wtf8(pos[1]) };
    match s.strip_prefix(prefix) {
        Some(rest) => Ok(w_str_from_wtf8(rest.to_wtf8_buf())),
        None => Ok(str_result_unchanged(pos[0])),
    }
}

/// PyPy: unicodeobject.py descr_removesuffix (Python 3.9+)
pub fn str_method_removesuffix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, _) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "str.removesuffix() takes exactly one argument ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    if !unsafe { pyre_object::is_str(pos[1]) } {
        return Err(crate::PyError::type_error(format!(
            "removesuffix() argument must be str, not {}",
            arg_type_name(pos[1])
        )));
    }
    let s = unsafe { w_str_get_wtf8(pos[0]) };
    let suffix = unsafe { w_str_get_wtf8(pos[1]) };
    match s.strip_suffix(suffix) {
        Some(rest) => Ok(w_str_from_wtf8(rest.to_wtf8_buf())),
        None => Ok(str_result_unchanged(pos[0])),
    }
}

/// PyPy: unicodeobject.py descr_expandtabs
pub fn str_method_expandtabs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `tabsize` is positional-or-keyword (default 8).
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if pos.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "expandtabs() takes at most 1 argument ({} given)",
            pos.len().saturating_sub(1)
        )));
    }
    crate::builtins::kwarg_reject_unknown(kwargs, &["tabsize"], "expandtabs")?;
    crate::builtins::kwarg_reject_duplicate(kwargs, "expandtabs", "tabsize", pos.get(1).is_some())?;
    let s = unsafe { w_str_get_wtf8(pos[0]) };
    let tabsize = match pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "tabsize"))
    {
        Some(t) => crate::builtins::space_index_w(t)?,
        None => 8,
    };
    // Tabs advance to the next multiple of `tabsize` measured from the
    // start of the current line (the column resets on `\n` / `\r`); a
    // non-positive `tabsize` drops tabs entirely. The expanded length is
    // tracked with checked arithmetic so a pathological `tabsize` raises
    // OverflowError instead of attempting an unbounded allocation.
    let overflow = || crate::PyError::overflow_error("result is too long");
    let mut total: i64 = 0;
    let mut col: i64 = 0;
    for cp in s.code_points() {
        match cp.to_char() {
            Some('\t') => {
                if tabsize > 0 {
                    let incr = tabsize - (col % tabsize);
                    col = col.checked_add(incr).ok_or_else(overflow)?;
                    total = total.checked_add(incr).ok_or_else(overflow)?;
                }
            }
            Some('\n') | Some('\r') => {
                total = total.checked_add(1).ok_or_else(overflow)?;
                col = 0;
            }
            _ => {
                total = total.checked_add(1).ok_or_else(overflow)?;
                col += 1;
            }
        }
    }
    let cap = usize::try_from(total).map_err(|_| overflow())?;
    let mut result = Wtf8Buf::with_capacity(cap);
    let mut col: i64 = 0;
    for cp in s.code_points() {
        match cp.to_char() {
            Some('\t') => {
                if tabsize > 0 {
                    let incr = tabsize - (col % tabsize);
                    col += incr;
                    for _ in 0..incr {
                        result.push_char(' ');
                    }
                }
            }
            Some('\n') | Some('\r') => {
                result.push(cp);
                col = 0;
            }
            _ => {
                result.push(cp);
                col += 1;
            }
        }
    }
    Ok(w_str_from_wtf8(result))
}

/// str.translate(table) — table is a mapping from ordinals (int) to
/// ordinals (int), strings (str), or None (delete).
pub fn str_method_translate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_exact(args, "str.translate", 1)?;
    let s = unsafe { w_str_get_wtf8(args[0]) };
    let table = args[1];
    let mut result = Wtf8Buf::with_capacity(s.len());
    unsafe {
        for cp in s.code_points() {
            let key = w_int_new(cp.to_u32() as i64);
            match crate::baseobjspace::finditem(table, key)? {
                None => result.push(cp),
                Some(val) if is_none(val) => {}
                Some(val) if is_int(val) => {
                    let code = w_int_get_value(val);
                    if let Some(c) = u32::try_from(code).ok().and_then(CodePoint::from_u32) {
                        result.push(c);
                    } else {
                        return Err(crate::PyError::value_error(
                            "character mapping must be in range(0x110000)",
                        ));
                    }
                }
                Some(val) if is_str(val) => {
                    result.push_wtf8(w_str_get_wtf8(val));
                }
                Some(_) => {
                    return Err(crate::PyError::type_error(
                        "character mapping must return integer, None or str",
                    ));
                }
            }
        }
    }
    Ok(w_str_from_wtf8(result))
}

// ── Dict methods ─────────────────────────────────────────────────────

/// Resolve the actual backing W_DictObject for either a plain dict or
/// a dict subclass instance (which stores data in `__dict_data__`).
///
/// PyPy: W_DictMultiObject subclass instances ARE dicts, so no indirection
/// is needed. In pyre, dict subclass instances are W_ObjectObject with a
/// backing dict stored as an attribute.
pub fn resolve_dict_backing(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if is_dict(obj) {
            return obj;
        }
        // `pypy/objspace/std/dictproxyobject.py:75-82 keys_w/values_w/
        // items_w` forward through `space.call_method(self.w_mapping,
        // ...)` — the mapping is unwrapped before any dict-method
        // dispatch.  Surface the same shape here so
        // `dict_method_{keys,values,items,get,copy,update,...}` work
        // on `type.__dict__` without per-method proxy plumbing.
        if pyre_object::is_dict_proxy(obj) {
            let inner = pyre_object::w_dict_proxy_get_mapping(obj);
            if !inner.is_null() && pyre_object::is_dict(inner) {
                return inner;
            }
        }
        if is_instance(obj) {
            if let Ok(backing) = crate::baseobjspace::getattr_str(obj, "__dict_data__") {
                if is_dict(backing) {
                    return backing;
                }
            }
        }
    }
    pyre_object::PY_NULL
}

fn dict_lookup_checked(
    dict: PyObjectRef,
    key: PyObjectRef,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    unsafe {
        pyre_object::dictmultiobject::w_dict_lookup_checked(dict, key)
            .map_err(|_| crate::baseobjspace::take_pending_hash_error())
    }
}

pub(crate) fn dict_store_checked(
    dict: PyObjectRef,
    key: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        pyre_object::dictmultiobject::w_dict_store_checked(dict, key, value)
            .map_err(|_| crate::baseobjspace::take_pending_hash_error())
    }
}

/// Hash `item` with it rooted, then look it up with `probe`.
///
/// A user `__hash__` is a collection point and [`ObjectKey`] caches the element
/// pointer, so hashing inside the key build (`object_key_for_checked`) captures
/// the pre-move pointer — the reason `object_key_hashed` exists. The element is
/// rooted across the hash and reloaded, matching the add path
/// (`builtin_set_add_items`); the set needs no rooting, being an old-gen
/// allocation that keeps its address across a collection.
unsafe fn set_lookup_checked(
    set: PyObjectRef,
    item: PyObjectRef,
    probe: unsafe fn(
        PyObjectRef,
        pyre_object::dictmultiobject::ObjectKey,
    ) -> Result<bool, pyre_object::dictmultiobject::DictKeyError>,
) -> Result<bool, crate::PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let sp = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(item);
    let hash = crate::builtins::try_hash_value(pyre_object::gc_roots::shadow_stack_get(sp))?;
    let key = pyre_object::dictmultiobject::object_key_hashed(
        pyre_object::gc_roots::shadow_stack_get(sp),
        hash,
    );
    probe(set, key).map_err(|_| crate::baseobjspace::take_pending_hash_error())
}

/// Remove an element from a set, hashing it through the protocol.
pub(crate) fn set_discard_checked(
    set: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::PyError> {
    unsafe { set_lookup_checked(set, item, pyre_object::setobject::w_set_discard_key_checked) }
}

/// Test membership in a set, hashing the element through the protocol.
pub(crate) fn set_contains_checked(
    set: PyObjectRef,
    item: PyObjectRef,
) -> Result<bool, crate::PyError> {
    unsafe {
        set_lookup_checked(
            set,
            item,
            pyre_object::setobject::w_set_contains_key_checked,
        )
    }
}

pub fn dict_method_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "get", 1)?;
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    if dict.is_null() {
        return Ok(default);
    }
    Ok(dict_lookup_checked(dict, key)?.unwrap_or(default))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_keys` parity — returns
/// a live `dict_keys` view bound to the source dict, not a snapshot
/// list.  The view's iter / len / contains semantics dispatch back
/// through the source dict (see baseobjspace getattr arm) so
/// mutations on the dict are visible through the view, matching
/// `W_DictViewKeysObject`'s behaviour.
pub fn dict_method_keys(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "keys")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        // Type-erased fallback: the receiver isn't a dict, surface
        // an empty view rather than fabricating a foreign-shaped
        // list (the view's source-dict slot tolerates PY_NULL via
        // the read-side guards).
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Keys,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Keys,
    ))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_values` parity — same
/// shape as `descr_keys`, kind tag `Values`.
pub fn dict_method_values(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "values")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Values,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Values,
    ))
}

/// `pypy/objspace/std/dictmultiobject.py:descr_items` parity — same
/// shape as `descr_keys`, kind tag `Items`.
pub fn dict_method_items(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "items")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(pyre_object::dictmultiobject::w_dict_view_new(
            pyre_object::PY_NULL,
            pyre_object::dictmultiobject::DictViewKind::Items,
        ));
    }
    Ok(pyre_object::dictmultiobject::w_dict_view_new(
        dict,
        pyre_object::dictmultiobject::DictViewKind::Items,
    ))
}

/// Materialise a dict_keys / values / items view's current snapshot
/// as a list of items.  Mirrors the view iteration bodies on
/// `W_DictViewKeysObject` / values / items — pyre's `repr` /
/// `len` / `compare` / set-op paths call this to produce the
/// kind-appropriate list eagerly.
///
/// `__iter__` no longer routes through this helper: it allocates a
/// live `W_BaseDictMultiIterObject` that walks the source dict's entries
/// directly and trips on the dictversion counter, raising
/// `RuntimeError("dictionary changed size during iteration")` when
/// the source mutates mid-iteration.
pub fn dict_view_snapshot(view: PyObjectRef) -> Vec<PyObjectRef> {
    let kind = unsafe { pyre_object::dictmultiobject::w_dict_view_get_kind(view) };
    let dict = unsafe { pyre_object::dictmultiobject::w_dict_view_get_dict(view) };
    if dict.is_null() {
        return Vec::new();
    }
    let items = unsafe { pyre_object::w_dict_items(dict) };
    match kind {
        pyre_object::dictmultiobject::DictViewKind::Keys => {
            items.into_iter().map(|(k, _)| k).collect()
        }
        pyre_object::dictmultiobject::DictViewKind::Values => {
            items.into_iter().map(|(_, v)| v).collect()
        }
        pyre_object::dictmultiobject::DictViewKind::Items => items
            .into_iter()
            .map(|(k, v)| w_tuple_new(vec![k, v]))
            .collect(),
    }
}

/// `pypy/objspace/std/dictmultiobject.py:585-587 descr_copy` —
/// `return w_dict.copy()` which delegates to `strategy.copy(w_dict)`
/// (`:1152 AbstractTypedStrategy.copy`).  Typed strategies preserve
/// their backing shape by cloning the typed storage box and wrapping
/// it in a fresh W_DictObject with the same strategy.  Used by
/// `dict.copy()` and (via `resolve_dict_backing` proxy unwrap) by
/// `mappingproxy.copy()` (`dictproxyobject.py:84 copy_w`).
pub fn dict_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_dict_new());
    }
    let src = resolve_dict_backing(args[0]);
    if src.is_null() {
        return Ok(pyre_object::w_dict_new());
    }
    unsafe { Ok(pyre_object::dictmultiobject::w_dict_copy(src)) }
}

/// PyPy: dictmultiobject.py descr_update — dict.update([other], **kwargs).
///
/// CPython 3.x signature accepts a single optional positional that is
/// either a mapping (uses keys()) or an iterable of (key, value) pairs,
/// followed by arbitrary kwargs that are merged on top.  The trailing
/// `dictmultiobject.py:1378-1398 update1` — merge `w_data` into
/// `w_dict`.  Shared by `dict.__init__` and `dict.update`.
pub(crate) fn dict_update1(w_dict: PyObjectRef, w_data: PyObjectRef) -> Result<(), crate::PyError> {
    let dict = resolve_dict_backing(w_dict);
    if dict.is_null() {
        return Ok(());
    }
    let other_raw = resolve_dict_backing(w_data);
    unsafe {
        let fast_path_eligible = other_raw.is_null() == false
            && pyre_object::is_dict(other_raw)
            && dict_subclass_uses_default_iter(w_data);
        if fast_path_eligible {
            // `dictmultiobject.py:1401-1406 update1_dict_dict`
            let dst_is_empty = pyre_object::dictmultiobject::w_dict_is_regular_empty(dict);
            if dst_is_empty {
                let w_copy = pyre_object::dictmultiobject::w_dict_copy(other_raw);
                pyre_object::dictmultiobject::w_dict_adopt_regular_copy_for_empty_update(
                    dict, w_copy,
                );
            } else {
                for (k, v) in pyre_object::w_dict_items(other_raw) {
                    dict_store_checked(dict, k, v)?;
                }
            }
        } else {
            // `dictmultiobject.py:1388-1398 update1`
            let w_keys_method = match crate::baseobjspace::getattr_str(w_data, "keys") {
                Ok(value) => Some(value),
                Err(e) if e.kind == crate::PyErrorKind::AttributeError => None,
                Err(e) => return Err(e),
            };
            if let Some(w_method) = w_keys_method {
                // `dictmultiobject.py:1421-1424 update1_keys`
                let w_keys_view = crate::call::call_function_impl_result(w_method, &[])?;
                let keys = crate::builtins::collect_iterable(w_keys_view)?;
                for k in keys {
                    let v = crate::baseobjspace::getitem(w_data, k)?;
                    dict_store_checked(dict, k, v)?;
                }
            } else {
                // `dictmultiobject.py:1410-1418 update1_pairs`
                let pairs = crate::builtins::collect_iterable(w_data)?;
                for (idx, pair) in pairs.into_iter().enumerate() {
                    let entries = crate::builtins::collect_iterable(pair)?;
                    if entries.len() != 2 {
                        return Err(crate::PyError::value_error(format!(
                            "dictionary update sequence element #{idx} has length {}; 2 is required",
                            entries.len()
                        )));
                    }
                    dict_store_checked(dict, entries[0], entries[1])?;
                }
            }
        }
    }
    Ok(())
}

/// `dictmultiobject.py:1430-1443 init_or_update` — shared by `dict.__init__`
/// and `dict.update`; `name` selects the error-message verb (`"dict"` vs
/// `"update"`). Stores resolve the subclass backing before writing, so a dict
/// subclass instance is updated through its backing dict rather than its own
/// (uninitialised) strategy slot.
///
/// `__pyre_kw__`-marked dict is the kwargs vehicle pyre's CALL_KW
/// emits for builtin callees (`call.rs:727-744`).
pub fn dict_init_or_update(
    args: &[PyObjectRef],
    name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "update")?;
    let (positional, kwargs_dict) = crate::builtins::split_builtin_kwargs(args);
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "{name} expected at most 1 argument, got {}",
            positional.len() - 1
        )));
    }
    let dict = positional[0];
    if let Some(other) = positional.get(1).copied() {
        dict_update1(dict, other)?;
    }
    let backing = resolve_dict_backing(dict);
    if backing.is_null() {
        // A dict subclass declared with `__slots__` has no attribute storage
        // for its item backing (pyre keeps a dict subclass's items in an
        // instance attribute), so there is nowhere to merge into. Full
        // slotted-dict-subclass support needs intrinsic dict backing.
        return Ok(w_none());
    }
    if let Some(kwargs) = kwargs_dict {
        unsafe {
            for (k, v) in pyre_object::w_dict_items(kwargs) {
                if pyre_object::is_str(k)
                    && pyre_object::w_str_get_wtf8(k).as_str() == Ok("__pyre_kw__")
                {
                    continue;
                }
                dict_store_checked(backing, k, v)?;
            }
        }
    }
    Ok(w_none())
}

/// `dictmultiobject.py:137-139 descr_update` → `init_or_update`; the verb in
/// the arity error is `update`.
pub fn dict_method_update(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "update")?;
    dict_init_or_update(args, "update")
}

/// `dictmultiobject.py:1380-1386 update1` —
///
/// ```python
/// if (isinstance(w_data, W_DictMultiObject) and
///         space.is_w(
///             space.findattr(space.type(w_data), w_st_iter),
///             space.findattr(space.w_dict, w_st_iter))):
///     update1_dict_dict(space, w_dict, w_data)
/// ```
///
/// Returns True when `other` is either a real `dict` (no subclass)
/// or a `dict` subclass that hasn't shadowed `__iter__`.  Pyre's
/// `is_dict` already established the W_DictMultiObject side; this
/// helper performs the `findattr` identity check to keep
/// `__iter__`-overriding subclasses on the slow `keys()` path.
fn dict_subclass_uses_default_iter(other: PyObjectRef) -> bool {
    let Some(other_type) = crate::typedef::r#type(other) else {
        return false;
    };
    let dict_type = crate::typedef::gettypeobject(&pyre_object::DICT_TYPE);
    if dict_type.is_null() {
        // No registered dict typeobject yet (init order quirk) —
        // degrade to "fast path" (treat as plain dict) to preserve
        // current behaviour.
        return true;
    }
    // Real `dict` type — no subclass at all, so __iter__ is by
    // definition unshadowed.
    if std::ptr::eq(other_type as *const _, dict_type as *const _) {
        return true;
    }
    let other_iter = unsafe { crate::baseobjspace::lookup_in_type(other_type, "__iter__") };
    let dict_iter = unsafe { crate::baseobjspace::lookup_in_type(dict_type, "__iter__") };
    match (other_iter, dict_iter) {
        (Some(a), Some(b)) => std::ptr::eq(a, b),
        _ => false,
    }
}

/// `dictmultiobject.py:246-255 descr_pop` →
/// `strategy.pop(self, w_key, w_default)` — single-operation pop
/// via strategy dispatch (one hash).
pub fn dict_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "pop", 1)?;
    arity_at_most(args, "pop", 2)?;
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied();
    if !dict.is_null() {
        unsafe {
            match pyre_object::dictmultiobject::w_dict_pop_checked(dict, key) {
                Ok(Some(val)) => return Ok(val),
                Ok(None) => {}
                Err(_) => return Err(crate::baseobjspace::take_pending_hash_error()),
            }
        }
    }
    default.ok_or_else(|| crate::PyError::key_error_with_key(key))
}

/// `pypy/objspace/std/dictmultiobject.py:1395 W_DictMultiObject.descr_popitem`:
///
/// ```python
/// def descr_popitem(self, space):
///     try:
///         w_key, w_value = self.popitem()
///     except KeyError:
///         raise oefmt(space.w_KeyError, "dictionary is empty")
///     return space.newtuple([w_key, w_value])
/// ```
///
/// In Python 3.7+ `popitem` is LIFO (returns the most recently
/// inserted pair); pyre's `w_dict_items` preserves insertion order
/// so popping the last entry matches the spec.
pub fn dict_method_popitem(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_receiver(args, "popitem")?;
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Err(crate::PyError::key_error("popitem(): dictionary is empty"));
    }
    unsafe {
        if pyre_object::w_dict_len(dict) == 0 {
            return Err(crate::PyError::key_error("popitem(): dictionary is empty"));
        }
        let items = pyre_object::w_dict_items(dict);
        let (k, v) = items
            .last()
            .copied()
            .ok_or_else(|| crate::PyError::key_error("popitem(): dictionary is empty"))?;
        pyre_object::dictmultiobject::w_dict_delitem(dict, k);
        Ok(pyre_object::w_tuple_new(vec![k, v]))
    }
}

/// `dictmultiobject.py:267-269 descr_setdefault` →
/// `self.setdefault(w_key, w_default)` — delegates to
/// `strategy.setdefault` as a single atomic operation (one hash).
pub fn dict_method_setdefault(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    arity_at_least(args, "setdefault", 1)?;
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    if !dict.is_null() {
        unsafe {
            return pyre_object::dictmultiobject::w_dict_setdefault_checked(dict, key, default)
                .map_err(|_| crate::baseobjspace::take_pending_hash_error());
        }
    }
    Ok(default)
}

#[cfg(test)]
mod dict_method_tests {
    use super::*;

    use crate::test_hooks::install_hash_hook;

    fn assert_type_error<T: std::fmt::Debug>(result: Result<T, crate::PyError>) {
        let err = result.expect_err("operation should reject unhashable dict key");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
    }

    #[test]
    fn dict_get_rejects_unhashable_key() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(dict_method_get(&[dict, key]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setitem_rejects_unhashable_key_without_inserting() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(crate::baseobjspace::setitem(dict, key, w_int_new(1)));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_setdefault_rejects_unhashable_key() {
        // `dictmultiobject.py:749-753 EmptyDictStrategy.setdefault`:
        //   self.switch_to_correct_strategy(w_dict, w_key)
        //   w_dict.setitem(w_key, w_default)
        // `w_dict.setitem` hashes the key via the object strategy's
        // `space.hash_w`, so an unhashable key raises TypeError before
        // anything is stored — the dict stays empty.
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        assert_type_error(dict_method_setdefault(&[dict, key, w_int_new(1)]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_pop_empty_returns_default_without_hashing_key() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);
        let default = w_int_new(42);

        let result = dict_method_pop(&[dict, key, default]).expect("default should be returned");
        assert_eq!(result, default);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_pop_empty_without_default_raises_keyerror_not_typeerror() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);

        let err = dict_method_pop(&[dict, key]).expect_err("missing key should raise KeyError");
        assert_eq!(err.kind, crate::PyErrorKind::KeyError);
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }

    #[test]
    fn dict_update_pairs_rejects_unhashable_key() {
        install_hash_hook();
        let dict = w_dict_new();
        let key = w_list_new(vec![]);
        let pair = w_tuple_new(vec![key, w_int_new(1)]);
        let pairs = w_list_new(vec![pair]);

        assert_type_error(dict_method_update(&[dict, pairs]));
        assert_eq!(unsafe { w_dict_len(dict) }, 0);
    }
}

// ── Tuple methods ────────────────────────────────────────────────────

/// PyPy: tupleobject.py descr_index — tuple.index(value)
pub fn tuple_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_tuple_receiver(args, "index", true)?;
    if args.len() < 2 {
        return Err(crate::PyError::type_error(format!(
            "index expected at least 1 argument, got {}",
            args.len().saturating_sub(1)
        )));
    }
    if args.len() > 4 {
        return Err(crate::PyError::type_error(format!(
            "index expected at most 3 arguments, got {}",
            args.len() - 1
        )));
    }
    let tup = args[0];
    let value = args[1];
    let length = unsafe { w_tuple_len(tup) } as i64;
    let w_start = args.get(2).copied().unwrap_or_else(|| w_int_new(0));
    let w_stop = args.get(3).copied().unwrap_or_else(|| w_int_new(i64::MAX));
    let (start, stop) = crate::sliceobject::unwrap_start_stop(length, w_start, w_stop)?;
    for i in start..stop.min(length) {
        if let Some(item) = unsafe { w_tuple_getitem(tup, i) } {
            if crate::baseobjspace::eq_w(item, value)? {
                return Ok(w_int_new(i));
            }
        }
    }
    Err(crate::PyError::value_error(
        "tuple.index(x): x not in tuple",
    ))
}

/// PyPy: tupleobject.py descr_count — tuple.count(value)
pub fn tuple_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    require_tuple_receiver(args, "count", true)?;
    if args.len() != 2 {
        return Err(crate::PyError::type_error(format!(
            "tuple.count() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let tup = args[0];
    let value = args[1];
    let mut count: i64 = 0;
    unsafe {
        let n = w_tuple_len(tup);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(tup, i as i64) {
                if crate::baseobjspace::eq_w(item, value)? {
                    count += 1;
                }
            }
        }
    }
    Ok(w_int_new(count))
}
