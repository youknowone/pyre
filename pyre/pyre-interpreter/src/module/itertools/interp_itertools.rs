//! itertools implementation — PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// itertools stub
pub fn register_module(ns: &mut DictStorage) {
    // chain(*iterables) → flat iterator
    crate::dict_storage_store(
        ns,
        "chain",
        crate::make_builtin_function("chain", |args| {
            let mut items = Vec::new();
            for &arg in args {
                items.extend(crate::builtins::collect_iterable(arg)?);
            }
            let n = items.len();
            let list = pyre_object::w_list_new(items);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // starmap stub
    crate::dict_storage_store(
        ns,
        "starmap",
        crate::make_builtin_function_with_arity(
            "starmap",
            |_| Ok(pyre_object::w_list_new(vec![])),
            2,
        ),
    );
    // count(start=0, step=1) — PyPy: W_Count___new__
    //
    //     def W_Count___new__(space, w_subtype, w_start=0, w_step=1):
    //         return W_Count(space, w_start, w_step)
    crate::dict_storage_store(
        ns,
        "count",
        crate::make_builtin_function("count", |args| {
            let w_start = args.first().copied().unwrap_or(pyre_object::w_int_new(0));
            let w_step = args.get(1).copied().unwrap_or(pyre_object::w_int_new(1));
            Ok(pyre_object::itertoolsmodule::w_count_new(w_start, w_step))
        }),
    );
    // repeat(obj, times=None) — PyPy: W_Repeat___new__
    //
    //     def W_Repeat___new__(space, w_subtype, w_obj, w_times=None):
    //         return W_Repeat(space, w_obj, w_times)
    crate::dict_storage_store(
        ns,
        "repeat",
        crate::make_builtin_function("repeat", |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "repeat() missing 'object' argument",
                ));
            }
            let w_obj = args[0];
            let w_times = if args.len() >= 2 {
                unsafe {
                    if pyre_object::is_int(args[1]) {
                        Some(pyre_object::w_int_get_value(args[1]))
                    } else {
                        None
                    }
                }
            } else {
                None
            };
            Ok(pyre_object::itertoolsmodule::w_repeat_new(w_obj, w_times))
        }),
    );
    // islice
    crate::dict_storage_store(
        ns,
        "islice",
        crate::make_builtin_function("islice", |_| Ok(pyre_object::w_list_new(vec![]))),
    );
    // groupby
    crate::dict_storage_store(
        ns,
        "groupby",
        crate::make_builtin_function("groupby", |_| Ok(pyre_object::w_none())),
    );
    // permutations(iterable, r=None) — PyPy: pypy/module/itertools/interp_itertools.py
    crate::dict_storage_store(
        ns,
        "permutations",
        crate::make_builtin_function("permutations", |args| {
            // `interp_itertools.py W_Permutations.__init__` — iterable
            // is required; missing argument is a TypeError, not an
            // empty result that silently hides call-site bugs.
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "permutations() missing required argument 'iterable'",
                ));
            }
            let pool = crate::builtins::collect_iterable(args[0])?;
            let n = pool.len();
            let r = if args.len() >= 2 {
                unsafe {
                    if pyre_object::is_int(args[1]) {
                        pyre_object::w_int_get_value(args[1]) as usize
                    } else {
                        n
                    }
                }
            } else {
                n
            };
            if r > n {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            // Heap/Lehmer would be clearer; use a recursive closure-free helper.
            fn perms(
                pool: &[pyre_object::PyObjectRef],
                r: usize,
            ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                if r == 0 {
                    return vec![vec![]];
                }
                let mut out = Vec::new();
                for i in 0..pool.len() {
                    let mut rest: Vec<_> = pool.to_vec();
                    let head = rest.remove(i);
                    for mut tail in perms(&rest, r - 1) {
                        let mut v = vec![head];
                        v.append(&mut tail);
                        out.push(v);
                    }
                }
                out
            }
            let all = perms(&pool, r);
            let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // combinations(iterable, r)
    crate::dict_storage_store(
        ns,
        "combinations",
        crate::make_builtin_function_with_arity(
            "combinations",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_list_new(vec![]));
                }
                let pool = crate::builtins::collect_iterable(args[0])?;
                let r = unsafe { pyre_object::w_int_get_value(args[1]) as usize };
                if r > pool.len() {
                    return Ok(pyre_object::w_list_new(vec![]));
                }
                fn combs(
                    pool: &[pyre_object::PyObjectRef],
                    r: usize,
                    start: usize,
                ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                    if r == 0 {
                        return vec![vec![]];
                    }
                    let mut out = Vec::new();
                    for i in start..pool.len() {
                        for mut tail in combs(pool, r - 1, i + 1) {
                            let mut v = vec![pool[i]];
                            v.append(&mut tail);
                            out.push(v);
                        }
                    }
                    out
                }
                let all = combs(&pool, r, 0);
                let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
                let n = tuples.len();
                let list = pyre_object::w_list_new(tuples);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            2,
        ),
    );
    // product(*iterables, repeat=1)
    crate::dict_storage_store(
        ns,
        "product",
        crate::make_builtin_function("product", |args| {
            // `interp_itertools.py W_Product.__init__` —
            // `product(*iterables, repeat=1)`.  The kwarg arrives via
            // the trailing `__pyre_kw__` dict, mirroring how
            // `enumerate`/`zip` extract their kwargs in this module.
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["repeat"], "product")?;
            let repeat = match crate::builtins::kwarg_get(kwargs, "repeat") {
                Some(w) => unsafe {
                    if !pyre_object::is_int(w) {
                        return Err(crate::PyError::type_error(
                            "product() 'repeat' argument must be an integer",
                        ));
                    }
                    pyre_object::w_int_get_value(w)
                },
                None => 1,
            };
            if repeat < 0 {
                return Err(crate::PyError::value_error(
                    "repeat argument cannot be negative",
                ));
            }
            let base_pools: Vec<Vec<_>> = positional
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let mut pools: Vec<Vec<pyre_object::PyObjectRef>> =
                Vec::with_capacity(base_pools.len() * (repeat as usize));
            for _ in 0..repeat {
                for p in &base_pools {
                    pools.push(p.clone());
                }
            }
            let mut result: Vec<Vec<pyre_object::PyObjectRef>> = vec![vec![]];
            for pool in &pools {
                let mut new_result = Vec::with_capacity(result.len() * pool.len());
                for existing in &result {
                    for &item in pool {
                        let mut v = existing.clone();
                        v.push(item);
                        new_result.push(v);
                    }
                }
                result = new_result;
            }
            let tuples: Vec<_> = result.into_iter().map(pyre_object::w_tuple_new).collect();
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // zip_longest(*iterables, fillvalue=None) — interp_itertools.py
    // W_ZipLongest. CALL_KW packs `fillvalue` into the trailing
    // `__pyre_kw__` dict (`call.rs:727-744`); strip it before
    // collecting the iterable pools so the kwarg doesn't surface as
    // an extra positional pool.
    crate::dict_storage_store(
        ns,
        "zip_longest",
        crate::make_builtin_function("zip_longest", |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            // `pypy/module/itertools/interp_itertools.py:685` —
            // W_ZipLongest's `unwrap_spec` only knows about
            // `fillvalue`; any other keyword raises TypeError at the
            // gateway.  Pyre's flat builtin ABI has to enforce this
            // by hand.
            crate::builtins::kwarg_reject_unknown(kwargs, &["fillvalue"], "zip_longest")?;
            let fill =
                crate::builtins::kwarg_get(kwargs, "fillvalue").unwrap_or_else(pyre_object::w_none);
            let pools: Vec<Vec<_>> = positional
                .iter()
                .map(|&a| crate::builtins::collect_iterable(a))
                .collect::<Result<_, _>>()?;
            let max_len = pools.iter().map(|p| p.len()).max().unwrap_or(0);
            let mut tuples = Vec::with_capacity(max_len);
            for i in 0..max_len {
                let row: Vec<_> = pools
                    .iter()
                    .map(|p| if i < p.len() { p[i] } else { fill })
                    .collect();
                tuples.push(pyre_object::w_tuple_new(row));
            }
            let n = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // accumulate(iterable) — sums only, PyPy interp_itertools W_Accumulate.
    crate::dict_storage_store(
        ns,
        "accumulate",
        crate::make_builtin_function("accumulate", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_list_new(vec![]));
            }
            let items = crate::builtins::collect_iterable(args[0])?;
            let mut out = Vec::with_capacity(items.len());
            let mut acc: Option<pyre_object::PyObjectRef> = None;
            for item in items {
                acc = Some(match acc {
                    None => item,
                    Some(prev) => crate::baseobjspace::add(prev, item)?,
                });
                out.push(acc.unwrap());
            }
            let n = out.len();
            let list = pyre_object::w_list_new(out);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // compress(data, selectors)
    crate::dict_storage_store(
        ns,
        "compress",
        crate::make_builtin_function_with_arity(
            "compress",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_list_new(vec![]));
                }
                let data = crate::builtins::collect_iterable(args[0])?;
                let selectors = crate::builtins::collect_iterable(args[1])?;
                let mut out = Vec::new();
                for (d, s) in data.iter().zip(selectors.iter()) {
                    if crate::baseobjspace::is_true(*s) {
                        out.push(*d);
                    }
                }
                let n = out.len();
                let list = pyre_object::w_list_new(out);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            2,
        ),
    );
}
