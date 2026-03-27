//! _sre module — SRE regex engine bridge.
//!
//! Uses sre-engine crate (RustPython's SRE bytecode interpreter).

use crate::{PyNamespace, builtin_code_new, namespace_store};
use pyre_object::*;
use sre_engine::engine::{Request, State, StrDrive};

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "MAGIC", w_int_new(20230612)); // SRE magic number
    namespace_store(ns, "CODESIZE", w_int_new(sre_engine::CODESIZE as i64));
    namespace_store(ns, "MAXREPEAT", w_int_new(sre_engine::MAXREPEAT as i64));
    namespace_store(ns, "MAXGROUPS", w_int_new(sre_engine::MAXGROUPS as i64));
    namespace_store(ns, "compile", builtin_code_new("compile", sre_compile));
    namespace_store(
        ns,
        "ascii_iscased",
        builtin_code_new("ascii_iscased", |args| {
            if args.is_empty() {
                return Ok(w_bool_from(false));
            }
            let ch = unsafe { w_int_get_value(args[0]) } as u8 as char;
            Ok(w_bool_from(ch.is_ascii_alphabetic()))
        }),
    );
    namespace_store(
        ns,
        "unicode_iscased",
        builtin_code_new("unicode_iscased", |args| {
            if args.is_empty() {
                return Ok(w_bool_from(false));
            }
            let ch = char::from_u32(unsafe { w_int_get_value(args[0]) } as u32).unwrap_or('\0');
            Ok(w_bool_from(ch.is_alphabetic()))
        }),
    );
    namespace_store(
        ns,
        "ascii_tolower",
        builtin_code_new("ascii_tolower", |args| {
            if args.is_empty() {
                return Ok(w_int_new(0));
            }
            Ok(w_int_new(
                (unsafe { w_int_get_value(args[0]) } as u8).to_ascii_lowercase() as i64,
            ))
        }),
    );
    namespace_store(
        ns,
        "unicode_tolower",
        builtin_code_new("unicode_tolower", |args| {
            if args.is_empty() {
                return Ok(w_int_new(0));
            }
            let c = char::from_u32(unsafe { w_int_get_value(args[0]) } as u32).unwrap_or('\0');
            Ok(w_int_new(c.to_lowercase().next().unwrap_or(c) as i64))
        }),
    );
    namespace_store(
        ns,
        "getcodesize",
        builtin_code_new("getcodesize", |_| {
            Ok(w_int_new(sre_engine::CODESIZE as i64))
        }),
    );
    namespace_store(
        ns,
        "getlower",
        builtin_code_new("getlower", |args| {
            if args.is_empty() {
                return Ok(w_int_new(0));
            }
            Ok(w_int_new(sre_engine::engine::lower_unicode(
                unsafe { w_int_get_value(args[0]) } as u32
            ) as i64))
        }),
    );
}

/// _sre.compile(pattern, flags, code, groups, groupindex, indexgroup)
fn sre_compile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "_sre.compile() requires at least 3 arguments",
        ));
    }
    let pattern = args[0];
    let flags = args[1];
    let code_list = args[2];
    let groups = if args.len() > 3 {
        args[3]
    } else {
        w_int_new(0)
    };
    let groupindex = if args.len() > 4 {
        args[4]
    } else {
        w_dict_new()
    };
    let indexgroup = if args.len() > 5 {
        args[5]
    } else {
        w_tuple_new(vec![])
    };

    let code_vec = extract_code(code_list)?;
    let code_box = Box::leak(Box::new(code_vec));

    let pat = w_instance_new(crate::typedef::getobjecttype());
    crate::baseobjspace::ATTR_TABLE.with(|t| {
        let mut t = t.borrow_mut();
        let d = t.entry(pat as usize).or_default();
        d.insert("pattern".into(), pattern);
        d.insert("flags".into(), flags);
        d.insert("groups".into(), groups);
        d.insert("groupindex".into(), groupindex);
        d.insert("indexgroup".into(), indexgroup);
        d.insert(
            "_code_ptr".into(),
            w_int_new(code_box.as_ptr() as usize as i64),
        );
        d.insert("_code_len".into(), w_int_new(code_box.len() as i64));
    });

    for (name, func) in [
        ("match", sre_pattern_match as fn(&[PyObjectRef]) -> _),
        ("search", sre_pattern_search),
        ("findall", sre_pattern_findall),
        ("finditer", sre_pattern_finditer),
        ("sub", sre_pattern_sub),
        ("subn", sre_pattern_sub),
        ("split", sre_pattern_split),
        ("fullmatch", sre_pattern_fullmatch),
    ] {
        let _ = crate::baseobjspace::setattr(pat, name, builtin_code_new(name, func));
    }

    Ok(pat)
}

fn extract_code(obj: PyObjectRef) -> Result<Vec<u32>, crate::PyError> {
    let mut code = Vec::new();
    unsafe {
        if is_list(obj) {
            for i in 0..w_list_len(obj) {
                if let Some(v) = w_list_getitem(obj, i as i64) {
                    code.push(w_int_get_value(v) as u32);
                }
            }
        } else if is_tuple(obj) {
            for i in 0..w_tuple_len(obj) {
                if let Some(v) = w_tuple_getitem(obj, i as i64) {
                    code.push(w_int_get_value(v) as u32);
                }
            }
        }
    }
    Ok(code)
}

fn get_code(pat: PyObjectRef) -> Option<&'static [u32]> {
    crate::baseobjspace::ATTR_TABLE.with(|t| {
        let t = t.borrow();
        let d = t.get(&(pat as usize))?;
        let ptr = unsafe { w_int_get_value(*d.get("_code_ptr")?) } as usize as *const u32;
        let len = unsafe { w_int_get_value(*d.get("_code_len")?) } as usize;
        if ptr.is_null() {
            return None;
        }
        Some(unsafe { std::slice::from_raw_parts(ptr, len) })
    })
}

fn do_match(
    args: &[PyObjectRef],
    search: bool,
    match_all: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("requires self and string"));
    }
    let pat = args[0];
    let string = args[1];
    let s = unsafe { w_str_get_value(string) };
    let code = get_code(pat).ok_or_else(|| crate::PyError::type_error("no compiled code"))?;

    let pos = if args.len() > 2 {
        (unsafe { w_int_get_value(args[2]) }) as usize
    } else {
        0
    };
    let endpos = if args.len() > 3 {
        (unsafe { w_int_get_value(args[3]) }) as usize
    } else {
        s.len()
    };

    let req = Request::new(s, pos, endpos, code, match_all);
    let mut state = State::<&str>::default();

    if search {
        state.search(req);
    } else {
        state.pymatch(req);
    }

    if state.has_matched {
        Ok(make_match(pat, string, &state, s))
    } else {
        Ok(w_none())
    }
}

fn make_match(pat: PyObjectRef, string: PyObjectRef, state: &State<&str>, s: &str) -> PyObjectRef {
    let m = w_instance_new(crate::typedef::getobjecttype());
    let start = state.start;
    let end = state.string_position;

    crate::baseobjspace::ATTR_TABLE.with(|t| {
        let mut t = t.borrow_mut();
        let d = t.entry(m as usize).or_default();
        d.insert("string".into(), string);
        d.insert("re".into(), pat);
        d.insert("pos".into(), w_int_new(0));
        d.insert("endpos".into(), w_int_new(s.len() as i64));
        d.insert("_start".into(), w_int_new(start as i64));
        d.insert("_end".into(), w_int_new(end as i64));
        d.insert("lastindex".into(), {
            let li = state.marks.last_index();
            if li >= 0 {
                w_int_new(li as i64)
            } else {
                w_none()
            }
        });
        d.insert("lastgroup".into(), w_none());
    });

    // Group spans
    let num_groups = state.marks.len() / 2;
    let mut spans = vec![w_tuple_new(vec![
        w_int_new(start as i64),
        w_int_new(end as i64),
    ])];
    for gi in 0..num_groups {
        let (gs, ge) = state.marks.get(gi);
        let span = match (gs.into_option(), ge.into_option()) {
            (Some(a), Some(b)) => w_tuple_new(vec![w_int_new(a as i64), w_int_new(b as i64)]),
            _ => w_tuple_new(vec![w_int_new(-1), w_int_new(-1)]),
        };
        spans.push(span);
    }
    let _ = crate::baseobjspace::setattr(m, "_spans", w_list_new(spans));

    for (name, func) in [
        ("group", sre_match_group as fn(&[PyObjectRef]) -> _),
        ("groups", sre_match_groups),
        ("start", sre_match_start),
        ("end", sre_match_end),
        ("span", sre_match_span),
    ] {
        let _ = crate::baseobjspace::setattr(m, name, builtin_code_new(name, func));
    }
    m
}

fn sre_pattern_match(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    do_match(args, false, false)
}
fn sre_pattern_fullmatch(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    do_match(args, false, true)
}
fn sre_pattern_search(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    do_match(args, true, false)
}

fn sre_pattern_findall(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "findall requires self and string",
        ));
    }
    let pat = args[0];
    let s = unsafe { w_str_get_value(args[1]) };
    let code = get_code(pat).ok_or_else(|| crate::PyError::type_error("no code"))?;

    let req = Request::new(s, 0, s.len(), code, false);
    let state = State::<&str>::default();
    let mut results = Vec::new();
    let mut iter = sre_engine::engine::SearchIter { req, state };
    while iter.next().is_some() {
        let matched = &s[iter.state.start..iter.state.string_position];
        results.push(w_str_new(matched));
    }
    Ok(w_list_new(results))
}

fn sre_pattern_finditer(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    sre_pattern_findall(args).and_then(|list| crate::baseobjspace::iter(list))
}

fn sre_pattern_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "sub requires self, repl, string",
        ));
    }
    let pat = args[0];
    let r = unsafe { w_str_get_value(args[1]) };
    let s = unsafe { w_str_get_value(args[2]) };
    let code = get_code(pat).ok_or_else(|| crate::PyError::type_error("no code"))?;

    let req = Request::new(s, 0, s.len(), code, false);
    let state = State::<&str>::default();
    let mut result = String::new();
    let mut last = 0;
    let mut iter = sre_engine::engine::SearchIter { req, state };
    while iter.next().is_some() {
        result.push_str(&s[last..iter.state.start]);
        result.push_str(r);
        last = iter.state.string_position;
    }
    result.push_str(&s[last..]);
    Ok(w_str_new(&result))
}

fn sre_pattern_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Ok(w_list_new(vec![]));
    }
    Ok(w_list_new(vec![args[1]])) // stub
}

fn sre_match_group(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_str_new(""));
    }
    let m = args[0];
    let gi = if args.len() > 1 {
        (unsafe { w_int_get_value(args[1]) }) as usize
    } else {
        0
    };
    let string = crate::baseobjspace::getattr(m, "string")?;
    let spans = crate::baseobjspace::getattr(m, "_spans")?;
    let s = unsafe { w_str_get_value(string) };
    unsafe {
        if let Some(span) = w_list_getitem(spans, gi as i64) {
            if let (Some(so), Some(eo)) = (w_tuple_getitem(span, 0), w_tuple_getitem(span, 1)) {
                let start = w_int_get_value(so) as usize;
                let end = w_int_get_value(eo) as usize;
                if start <= s.len() && end <= s.len() {
                    return Ok(w_str_new(&s[start..end]));
                }
            }
        }
    }
    Ok(w_none())
}

fn sre_match_groups(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_tuple_new(vec![]));
    }
    let m = args[0];
    let string = crate::baseobjspace::getattr(m, "string")?;
    let spans = crate::baseobjspace::getattr(m, "_spans")?;
    let s = unsafe { w_str_get_value(string) };
    let n = unsafe { w_list_len(spans) };
    let mut groups = Vec::new();
    for i in 1..n {
        // skip group 0
        unsafe {
            if let Some(span) = w_list_getitem(spans, i as i64) {
                if let (Some(so), Some(eo)) = (w_tuple_getitem(span, 0), w_tuple_getitem(span, 1)) {
                    let start = w_int_get_value(so);
                    let end = w_int_get_value(eo);
                    if start >= 0
                        && end >= 0
                        && (start as usize) <= s.len()
                        && (end as usize) <= s.len()
                    {
                        groups.push(w_str_new(&s[start as usize..end as usize]));
                        continue;
                    }
                }
            }
        }
        groups.push(w_none());
    }
    Ok(w_tuple_new(groups))
}

fn sre_match_start(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_int_new(0));
    }
    crate::baseobjspace::getattr(args[0], "_start")
}

fn sre_match_end(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_int_new(0));
    }
    crate::baseobjspace::getattr(args[0], "_end")
}

fn sre_match_span(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_tuple_new(vec![w_int_new(0), w_int_new(0)]));
    }
    let s = crate::baseobjspace::getattr(args[0], "_start")?;
    let e = crate::baseobjspace::getattr(args[0], "_end")?;
    Ok(w_tuple_new(vec![s, e]))
}
