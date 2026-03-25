//! Builtin type method implementations.
//!
//! PyPy equivalents:
//!   pypy/objspace/std/listobject.py  (list methods)
//!   pypy/objspace/std/unicodeobject.py  (str methods)
//!   pypy/objspace/std/dictobject.py  (dict methods)
//!   pypy/objspace/std/tupleobject.py  (tuple methods)
//!
//! Separated from space.rs to avoid bloating the hot-path compilation
//! unit. Method functions are registered into TypeDef at startup.

use pyre_object::*;

// ── List methods ─────────────────────────────────────────────────────
// All take self (list) as first arg.

pub fn list_method_append(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2, "append() takes exactly one argument");
    unsafe { w_list_append(args[0], args[1]) };
    w_none()
}

pub fn list_method_extend(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    let list = args[0];
    let other = args[1];
    unsafe {
        if is_list(other) {
            let n = w_list_len(other);
            for i in 0..n {
                if let Some(item) = w_list_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        } else if is_tuple(other) {
            let n = w_tuple_len(other);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        }
    }
    w_none()
}

/// PyPy: listobject.py descr_insert
pub fn list_method_insert(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.insert() not yet implemented");
}

/// PyPy: listobject.py descr_pop
pub fn list_method_pop(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.pop() not yet implemented (requires mutable list removal)");
}

/// PyPy stub — list.clear() not yet implemented
pub fn list_method_clear(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.clear() not yet implemented");
}

pub fn list_method_copy(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let list = args[0];
    unsafe {
        let n = w_list_len(list);
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
        w_list_new(items)
    }
}

/// PyPy stub — list.reverse() not yet implemented
pub fn list_method_reverse(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.reverse() not yet implemented");
}

/// PyPy stub — list.sort() not yet implemented
pub fn list_method_sort(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.sort() not yet implemented");
}

/// PyPy stub — list.index() not yet implemented
pub fn list_method_index(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.index() not yet implemented");
}

/// PyPy stub — list.count() not yet implemented
pub fn list_method_count(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.count() not yet implemented");
}

/// PyPy stub — list.remove() not yet implemented
pub fn list_method_remove(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("list.remove() not yet implemented");
}

// ── String methods ───────────────────────────────────────────────────

pub fn str_method_join(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() == 2);
    let sep = unsafe { w_str_get_value(args[0]) };
    let iterable = args[1];
    let mut parts = Vec::new();
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    if is_str(item) {
                        parts.push(w_str_get_value(item).to_string());
                    }
                }
            }
        } else if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    if is_str(item) {
                        parts.push(w_str_get_value(item).to_string());
                    }
                }
            }
        }
    }
    w_str_new(&parts.join(sep))
}

pub fn str_method_split(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let sep = if args.len() > 1 && !args[1].is_null() && unsafe { !is_none(args[1]) } {
        Some(unsafe { w_str_get_value(args[1]) })
    } else {
        None
    };
    let parts: Vec<PyObjectRef> = match sep {
        Some(sep) => s.split(sep).map(|p| w_str_new(p)).collect(),
        None => s.split_whitespace().map(|p| w_str_new(p)).collect(),
    };
    w_list_new(parts)
}

pub fn str_method_strip(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    w_str_new(unsafe { w_str_get_value(args[0]) }.trim())
}

pub fn str_method_lstrip(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    w_str_new(unsafe { w_str_get_value(args[0]) }.trim_start())
}

pub fn str_method_rstrip(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    w_str_new(unsafe { w_str_get_value(args[0]) }.trim_end())
}

pub fn str_method_startswith(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let prefix = unsafe { w_str_get_value(args[1]) };
    w_bool_from(s.starts_with(prefix))
}

pub fn str_method_endswith(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let suffix = unsafe { w_str_get_value(args[1]) };
    w_bool_from(s.ends_with(suffix))
}

pub fn str_method_replace(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 3);
    let s = unsafe { w_str_get_value(args[0]) };
    let old = unsafe { w_str_get_value(args[1]) };
    let new = unsafe { w_str_get_value(args[2]) };
    w_str_new(&s.replace(old, new))
}

pub fn str_method_find(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sub = unsafe { w_str_get_value(args[1]) };
    w_int_new(s.find(sub).map(|i| i as i64).unwrap_or(-1))
}

pub fn str_method_rfind(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sub = unsafe { w_str_get_value(args[1]) };
    w_int_new(s.rfind(sub).map(|i| i as i64).unwrap_or(-1))
}

pub fn str_method_upper(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    w_str_new(&unsafe { w_str_get_value(args[0]) }.to_uppercase())
}

pub fn str_method_lower(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    w_str_new(&unsafe { w_str_get_value(args[0]) }.to_lowercase())
}

pub fn str_method_format(args: &[PyObjectRef]) -> PyObjectRef {
    // Simplified: return self as-is (format not yet implemented)
    assert!(!args.is_empty());
    args[0]
}

pub fn str_method_encode(args: &[PyObjectRef]) -> PyObjectRef {
    // Simplified: return str as-is (bytes not yet implemented)
    assert!(!args.is_empty());
    args[0]
}

pub fn str_method_isdigit(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    w_bool_from(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
}

pub fn str_method_isalpha(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    w_bool_from(!s.is_empty() && s.chars().all(|c| c.is_alphabetic()))
}

pub fn str_method_zfill(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) } as usize;
    if s.len() >= width {
        return args[0];
    }
    let padding = "0".repeat(width - s.len());
    w_str_new(&format!("{padding}{s}"))
}

// ── Dict methods ─────────────────────────────────────────────────────

pub fn dict_method_get(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let dict = args[0];
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    unsafe {
        if is_int(key) {
            w_dict_lookup(dict, key).unwrap_or(default)
        } else {
            default
        }
    }
}

/// PyPy: dictobject.py descr_keys — returns dict_keys view.
/// Simplified: returns list of int keys from our int-keyed dict.
pub fn dict_method_keys(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let dict = args[0];
    unsafe {
        if is_dict(dict) {
            let d = &*(dict as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let keys: Vec<PyObjectRef> = entries.iter().map(|&(k, _)| k).collect();
            return w_list_new(keys);
        }
    }
    w_list_new(vec![])
}

/// PyPy: dictobject.py descr_values
pub fn dict_method_values(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let dict = args[0];
    unsafe {
        if is_dict(dict) {
            let d = &*(dict as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let values: Vec<PyObjectRef> = entries.iter().map(|&(_, v)| v).collect();
            return w_list_new(values);
        }
    }
    w_list_new(vec![])
}

/// PyPy: dictobject.py descr_items
pub fn dict_method_items(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(!args.is_empty());
    let dict = args[0];
    unsafe {
        if is_dict(dict) {
            let d = &*(dict as *const pyre_object::dictobject::W_DictObject);
            let entries = &*d.entries;
            let items: Vec<PyObjectRef> = entries
                .iter()
                .map(|&(k, v)| w_tuple_new(vec![k, v]))
                .collect();
            return w_list_new(items);
        }
    }
    w_list_new(vec![])
}

/// PyPy stub — dict.update() not yet implemented
pub fn dict_method_update(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("dict.update() not yet implemented");
}

/// PyPy: dictobject.py descr_pop
pub fn dict_method_pop(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2, "dict.pop() takes at least 1 argument");
    let dict = args[0];
    let key = args[1];
    let default = args.get(2).copied();
    unsafe {
        if is_dict(dict) {
            if let Some(val) = w_dict_lookup(dict, key) {
                return val;
            }
        }
    }
    default.unwrap_or_else(|| panic!("KeyError"))
}

pub fn dict_method_setdefault(args: &[PyObjectRef]) -> PyObjectRef {
    assert!(args.len() >= 2);
    let dict = args[0];
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    unsafe {
        if is_dict(dict) {
            if let Some(existing) = w_dict_lookup(dict, key) {
                return existing;
            }
            w_dict_store(dict, key, default);
        }
    }
    default
}

// ── Tuple methods ────────────────────────────────────────────────────

/// PyPy stub — tuple.index() not yet implemented
pub fn tuple_method_index(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("tuple.index() not yet implemented");
}

/// PyPy stub — tuple.count() not yet implemented
pub fn tuple_method_count(_args: &[PyObjectRef]) -> PyObjectRef {
    panic!("tuple.count() not yet implemented");
}
