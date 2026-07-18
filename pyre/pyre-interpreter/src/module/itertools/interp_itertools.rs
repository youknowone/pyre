//! itertools implementation — PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Verbatim move of the inline block previously in importing.rs.


/// itertools stub
pub fn register_module(ns: pyre_object::PyObjectRef) {
    // chain(*iterables) — W_Chain___new__: store `iter(newtuple(args))` as
    // the source-iterables iterator.  W_Chain.next_w (baseobjspace::next)
    // lazily draws each sub-iterable's iterator on demand, so infinite
    // arguments (e.g. `chain([3], repeat(3))`) do not hang at construction.
    let chain_fn = crate::make_builtin_function("chain", |args| {
        let tup = pyre_object::w_tuple_new(args.to_vec());
        let iterables = crate::baseobjspace::iter(tup)?;
        Ok(pyre_object::interp_itertools::w_chain_new(iterables))
    });
    // chain.from_iterable(iterable) — W_Chain.descr_from_iterable: flatten a
    // single iterable of iterables.  Attached as an attribute on the `chain`
    // callable (the classmethod is read straight off the function object, so
    // it is called with just the outer iterable).
    let from_iterable_fn = crate::make_builtin_function("from_iterable", |args| {
        let outer = args.first().copied().ok_or_else(|| {
            crate::PyError::type_error("from_iterable() missing 1 required positional argument")
        })?;
        let iterables = crate::baseobjspace::iter(outer)?;
        Ok(pyre_object::interp_itertools::w_chain_new(iterables))
    });
    crate::setattr_str(chain_fn, "from_iterable", from_iterable_fn)
        .expect("attach itertools.chain.from_iterable");
    crate::module_ns_store(ns, "chain", chain_fn);
    // PyPy exports W_StarMap.typedef itself; its __new__ stores a live source
    // iterator and next_w performs one expanded call at a time.
    crate::module_ns_store(
        ns,
        "starmap",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::STARMAP_TYPE)
            .expect("itertools.starmap TypeDef initialized"),
    );
    // PyPy exposes W_Count.typedef / W_Repeat.typedef themselves from the
    // module, not function-shaped constructor shims.  Their `__new__` slots
    // perform allocation and argument parsing.
    crate::module_ns_store(
        ns,
        "count",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COUNT_TYPE)
            .expect("itertools.count TypeDef initialized"),
    );
    crate::module_ns_store(
        ns,
        "repeat",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::REPEAT_TYPE)
            .expect("itertools.repeat TypeDef initialized"),
    );
    // islice(iterable, stop) | islice(iterable, start, stop[, step]) —
    // PyPy: W_ISlice.__init__.  Pulled lazily from the source iterator so
    // an unbounded input (`count`, `cycle`) is bounded by `stop`.
    crate::module_ns_store(
        ns,
        "islice",
        crate::make_builtin_function("islice", |args| {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(format!(
                    "islice expected at least 2 arguments, got {}",
                    args.len()
                )));
            }
            // `W_ISlice.arg_int_w` — `space.index` then a `>= minimum`
            // gate; a non-integer or out-of-range value is a ValueError
            // carrying the same message.
            fn arg_int(
                w: pyre_object::PyObjectRef,
                minimum: i64,
                msg: &str,
            ) -> Result<i64, crate::PyError> {
                let v = unsafe {
                    // `is_int` is true for a bool (`BOOL_TYPE`), so test `is_bool` first.
                    if pyre_object::is_bool(w) {
                        pyre_object::w_bool_get_value(w) as i64
                    } else if pyre_object::is_int(w) {
                        pyre_object::w_int_get_value(w)
                    } else {
                        return Err(crate::PyError::value_error(msg.to_string()));
                    }
                };
                if v < minimum {
                    return Err(crate::PyError::value_error(msg.to_string()));
                }
                Ok(v)
            }
            let is_none = |w| unsafe { pyre_object::is_none(w) };
            let (start, w_stop, w_step) = if args.len() == 2 {
                (0i64, args[1], None)
            } else if args.len() <= 4 {
                let start = if is_none(args[1]) {
                    0
                } else {
                    arg_int(
                        args[1],
                        0,
                        "Indicies for islice() must be None or non-negative integers",
                    )?
                };
                (start, args[2], args.get(3).copied())
            } else {
                return Err(crate::PyError::type_error(format!(
                    "islice() takes at most 4 arguments ({} given)",
                    args.len() - 2
                )));
            };
            let stop: Option<i64> = if is_none(w_stop) {
                None
            } else {
                Some(
                    arg_int(w_stop, 0, "Stop argument must be a non-negative integer or None.")?
                        .max(start),
                )
            };
            let step = match w_step {
                None => 1,
                Some(w) if is_none(w) => 1,
                Some(w) => arg_int(w, 1, "Step for islice() must be a positive integer or None")?,
            };
            let iterator = crate::baseobjspace::iter(args[0])?;
            let mut out = Vec::new();
            let mut idx: i64 = 0;
            let mut next_target = start;
            loop {
                if let Some(s) = stop {
                    if idx >= s {
                        break;
                    }
                }
                match crate::baseobjspace::next(iterator) {
                    Ok(v) => {
                        if idx == next_target {
                            out.push(v);
                            next_target += step;
                        }
                        idx += 1;
                    }
                    Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                    Err(e) => return Err(e),
                }
            }
            let n = out.len();
            let list = pyre_object::w_list_new(out);
            Ok(pyre_object::w_seq_iter_new(list, n))
        }),
    );
    // groupby
    crate::module_ns_store(
        ns,
        "groupby",
        crate::make_builtin_function("groupby", |_| Ok(pyre_object::w_none())),
    );
    // permutations(iterable, r=None) — PyPy: pypy/module/itertools/interp_itertools.py
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    crate::module_ns_store(
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
    // W_Compress.typedef is exported directly, matching PyPy's dedicated
    // live iterator rather than materializing both inputs into a list.
    crate::module_ns_store(
        ns,
        "compress",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COMPRESS_TYPE)
            .expect("itertools.compress TypeDef initialized"),
    );
    // PyPy exposes these W_Root subclasses through their TypeDefs.  Their
    // `__new__` slots retain the two-argument/subclass-init gateway behavior.
    crate::module_ns_store(
        ns,
        "takewhile",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::TAKEWHILE_TYPE)
            .expect("itertools.takewhile TypeDef initialized"),
    );
    crate::module_ns_store(
        ns,
        "dropwhile",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::DROPWHILE_TYPE)
            .expect("itertools.dropwhile TypeDef initialized"),
    );
    crate::module_ns_store(
        ns,
        "filterfalse",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::FILTERFALSE_TYPE)
            .expect("itertools.filterfalse TypeDef initialized"),
    );
    // pairwise(iterable) — W_Pairwise__new__: store `space.iter(w_iterable)`;
    // pairs are produced lazily by W_Pairwise.next_w (baseobjspace::next).
    crate::module_ns_store(
        ns,
        "pairwise",
        crate::make_builtin_function_with_arity(
            "pairwise",
            |args| {
                let iterator = crate::baseobjspace::iter(args[0])?;
                Ok(pyre_object::interp_itertools::w_pairwise_new(iterator))
            },
            1,
        ),
    );
    // cycle(iterable) — W_Cycle___new__: store `space.iter(w_iterable)` and an
    // empty `saved` list.  W_Cycle.next_w (baseobjspace::next) pulls from the
    // source on the first pass, saving each element, then replays `saved`
    // forever.
    crate::module_ns_store(
        ns,
        "cycle",
        crate::make_builtin_function_with_arity(
            "cycle",
            |args| {
                let iterator = crate::baseobjspace::iter(args[0])?;
                Ok(pyre_object::interp_itertools::w_cycle_new(iterator))
            },
            1,
        ),
    );
    // batched(iterable, n, *, strict=False) — CPython 3.13 itertools.batched.
    // Batches the input into tuples of length `n`; the last tuple may be
    // shorter unless `strict` is set, in which case a short final batch
    // raises ValueError.  Materialized eagerly like the other builtins here.
    crate::module_ns_store(
        ns,
        "batched",
        crate::make_builtin_function("batched", |args| {
            let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
            crate::builtins::kwarg_reject_unknown(kwargs, &["strict"], "batched")?;
            if positional.len() < 2 {
                return Err(crate::PyError::type_error(format!(
                    "batched expected at least 2 arguments, got {}",
                    positional.len()
                )));
            }
            // `n` goes through `space.index` (`__index__`); a non-integer
            // raises "'X' object cannot be interpreted as an integer".
            let n = crate::builtins::space_index_w(positional[1])?;
            if n < 1 {
                return Err(crate::PyError::value_error("n must be at least one"));
            }
            let strict = match crate::builtins::kwarg_get(kwargs, "strict") {
                Some(w) => crate::baseobjspace::is_true(w)?,
                None => false,
            };
            let n = n as usize;
            let items = crate::builtins::collect_iterable(positional[0])?;
            let mut tuples = Vec::with_capacity(items.len().div_ceil(n));
            let mut i = 0usize;
            while i < items.len() {
                let end = (i + n).min(items.len());
                let chunk: Vec<_> = items[i..end].to_vec();
                if strict && chunk.len() != n {
                    return Err(crate::PyError::value_error("batched(): incomplete batch"));
                }
                tuples.push(pyre_object::w_tuple_new(chunk));
                i = end;
            }
            let count = tuples.len();
            let list = pyre_object::w_list_new(tuples);
            Ok(pyre_object::w_seq_iter_new(list, count))
        }),
    );
}
