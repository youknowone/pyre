//! itertools implementation — PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Verbatim move of the inline block previously in importing.rs.


/// groupby(iterable, key=None) — itertools-docs pure-Python equivalent.
const GROUPBY_SRC: &str = r#"
class groupby:
    __module__ = 'itertools'
    def __init__(self, iterable, key=None):
        if key is None:
            key = lambda x: x
        self.keyfunc = key
        self.it = iter(iterable)
        self.tgtkey = self.currkey = self.currvalue = object()
    def __iter__(self):
        return self
    def __next__(self):
        self.id = object()
        while self.currkey == self.tgtkey:
            self.currvalue = next(self.it)
            self.currkey = self.keyfunc(self.currvalue)
        self.tgtkey = self.currkey
        return (self.currkey, self._grouper(self.tgtkey, self.id))
    def _grouper(self, tgtkey, id):
        while self.id is id and self.currkey == tgtkey:
            yield self.currvalue
            try:
                self.currvalue = next(self.it)
            except StopIteration:
                return
            self.currkey = self.keyfunc(self.currvalue)
"#;

/// tee(iterable, n=2) — itertools-docs pure-Python equivalent.  Each `_tee`
/// keeps its own deque; when a deque runs dry the shared source iterator is
/// advanced once and the new value is fanned out to every deque, so the copies
/// stay independent and an unbounded source is drawn lazily.
const TEE_SRC: &str = r#"
import collections
import operator

class _tee:
    __module__ = 'itertools'
    def __init__(self, it, deques, mydeque):
        self._it = it
        self._deques = deques
        self._mydeque = mydeque
    def __iter__(self):
        return self
    def __copy__(self):
        # W_TeeIterable.copy_w: the clone starts at this iterator's current
        # node while sharing the same source and future buffer fan-out.
        mydeque = collections.deque(self._mydeque)
        self._deques.append(mydeque)
        return _tee(self._it, self._deques, mydeque)
    def __next__(self):
        if not self._mydeque:
            newval = next(self._it)
            for d in self._deques:
                d.append(newval)
        return self._mydeque.popleft()

def tee(iterable, n=2):
    n = operator.index(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    it = iter(iterable)
    if hasattr(it, '__copy__'):
        return tuple(it if i == 0 else it.__copy__() for i in range(n))
    deques = [collections.deque() for _ in range(n)]
    return tuple(_tee(it, deques, d) for d in deques)
"#;

pub fn register_module(ns: pyre_object::PyObjectRef) {
    // PyPy exports W_Chain.typedef itself. Its __new__ and classmethod
    // from_iterable both preserve lazy traversal of the outer iterable.
    crate::module_ns_store(
        ns,
        "chain",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::CHAIN_TYPE)
            .expect("itertools.chain TypeDef initialized")
            .as_ptr(),
    );
    // PyPy exports W_StarMap.typedef itself; its __new__ stores a live source
    // iterator and next_w performs one expanded call at a time.
    crate::module_ns_store(
        ns,
        "starmap",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::STARMAP_TYPE).expect("itertools.starmap TypeDef initialized").as_ptr(),
    );
    // PyPy exposes W_Count.typedef / W_Repeat.typedef themselves from the
    // module, not function-shaped constructor shims.  Their `__new__` slots
    // perform allocation and argument parsing.
    crate::module_ns_store(
        ns,
        "count",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COUNT_TYPE).expect("itertools.count TypeDef initialized").as_ptr(),
    );
    crate::module_ns_store(
        ns,
        "repeat",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::REPEAT_TYPE).expect("itertools.repeat TypeDef initialized").as_ptr(),
    );
    // PyPy exports W_ISlice.typedef itself.  The native object retains its
    // source iterator plus count/next/stop/step cursor state and therefore
    // skips and yields incrementally rather than materializing the result.
    crate::module_ns_store(
        ns,
        "islice",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::ISLICE_TYPE)
            .expect("itertools.islice TypeDef initialized")
            .as_ptr(),
    );
    // groupby(iterable, key=None) — the itertools-docs pure-Python
    // equivalent.  The parent and each group share the `currkey/currvalue`
    // cursor plus an `id` token that invalidates a group once the parent
    // advances; expressing that shared state directly in Python avoids a
    // second native iterator type.
    crate::importing::appleveldef_install(ns, GROUPBY_SRC, "<inline>", &["groupby"]);
    // tee(iterable, n=2) — the itertools-docs pure-Python equivalent.  A native
    // dataobject type would only save buffer copies; the deque-per-copy recipe
    // keeps the copies lazy and independent, which is what callers observe.
    crate::importing::appleveldef_install(ns, TEE_SRC, "<inline>", &["tee"]);
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
                if unsafe { pyre_object::is_none(args[1]) } {
                    n
                } else {
                    let r = crate::builtins::space_index_w(args[1])?;
                    if r < 0 {
                        return Err(crate::PyError::value_error("r must be non-negative"));
                    }
                    r as usize
                }
            } else {
                n
            };
            if r > n {
                let list = pyre_object::w_list_new(vec![]);
                return Ok(pyre_object::w_seq_iter_new(list, 0));
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
    // PyPy W_Combinations: retain pool/index/result state and advance one
    // lexicographic combination at a time.
    crate::module_ns_store(
        ns,
        "combinations",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COMBINATIONS_TYPE)
            .expect("itertools.combinations TypeDef initialized")
            .as_ptr(),
    );
    // combinations_with_replacement(iterable, r) — like combinations, but an
    // element may repeat, so the recursion re-enters at `i` rather than `i + 1`
    // and `r` may exceed the pool length.  `r` is taken through `__index__`
    // before the iterable is drawn, matching the argument-clinic evaluation
    // order, and a negative `r` is a ValueError.
    crate::module_ns_store(
        ns,
        "combinations_with_replacement",
        crate::make_builtin_function_with_arity_and_maybe_sig(
            "combinations_with_replacement",
            |args| {
                let missing = match args.len() {
                    0 => Some("iterable"),
                    1 => Some("r"),
                    _ => None,
                };
                if let Some(name) = missing {
                    return Err(crate::PyError::type_error(format!(
                        "combinations_with_replacement() missing required argument '{name}'"
                    )));
                }
                let r = crate::builtins::space_index_w(args[1])?;
                if r < 0 {
                    return Err(crate::PyError::value_error("r must be non-negative"));
                }
                let r = r as usize;
                let pool = crate::builtins::collect_iterable(args[0])?;
                fn cwr(
                    pool: &[pyre_object::PyObjectRef],
                    r: usize,
                    start: usize,
                ) -> Vec<Vec<pyre_object::PyObjectRef>> {
                    if r == 0 {
                        return vec![vec![]];
                    }
                    let mut out = Vec::new();
                    for i in start..pool.len() {
                        for mut tail in cwr(pool, r - 1, i) {
                            let mut v = vec![pool[i]];
                            v.append(&mut tail);
                            out.push(v);
                        }
                    }
                    out
                }
                let all = cwr(&pool, r, 0);
                let tuples: Vec<_> = all.into_iter().map(pyre_object::w_tuple_new).collect();
                let n = tuples.len();
                let list = pyre_object::w_list_new(tuples);
                Ok(pyre_object::w_seq_iter_new(list, n))
            },
            2,
            Some(crate::gateway::Signature::new(
                vec!["iterable", "r"],
                None,
                None,
                0,
                0,
            )),
        ),
    );
    // PyPy W_Product: retain only the pool snapshots and odometer state rather
    // than eagerly materializing the Cartesian result.
    crate::module_ns_store(
        ns,
        "product",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::PRODUCT_TYPE)
            .expect("itertools.product TypeDef initialized")
            .as_ptr(),
    );
    // PyPy exports W_ZipLongest.typedef.  Construction keeps each source as a
    // live iterator, so unbounded inputs remain lazy.
    crate::module_ns_store(
        ns,
        "zip_longest",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::ZIP_LONGEST_TYPE).expect("itertools.zip_longest TypeDef initialized").as_ptr(),
    );
    // PyPy exports the live W_Accumulate iterator TypeDef.  Its running total,
    // optional function, and initial value stay on the object and next_w
    // advances the source lazily.
    crate::module_ns_store(
        ns,
        "accumulate",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::ACCUMULATE_TYPE).expect("itertools.accumulate TypeDef initialized").as_ptr(),
    );
    // W_Compress.typedef is exported directly, matching PyPy's dedicated
    // live iterator rather than materializing both inputs into a list.
    crate::module_ns_store(
        ns,
        "compress",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::COMPRESS_TYPE).expect("itertools.compress TypeDef initialized").as_ptr(),
    );
    // PyPy exposes these W_Root subclasses through their TypeDefs.  Their
    // `__new__` slots retain the two-argument/subclass-init gateway behavior.
    crate::module_ns_store(
        ns,
        "takewhile",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::TAKEWHILE_TYPE).expect("itertools.takewhile TypeDef initialized").as_ptr(),
    );
    crate::module_ns_store(
        ns,
        "dropwhile",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::DROPWHILE_TYPE).expect("itertools.dropwhile TypeDef initialized").as_ptr(),
    );
    crate::module_ns_store(
        ns,
        "filterfalse",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::FILTERFALSE_TYPE).expect("itertools.filterfalse TypeDef initialized").as_ptr(),
    );
    // PyPy exports the native W_Pairwise / W_Cycle TypeDefs. Their __new__
    // slots allocate the requested subtype and retain a live source iterator.
    crate::module_ns_store(
        ns,
        "pairwise",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::PAIRWISE_TYPE)
            .expect("itertools.pairwise TypeDef initialized")
            .as_ptr(),
    );
    crate::module_ns_store(
        ns,
        "cycle",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::CYCLE_TYPE)
            .expect("itertools.cycle TypeDef initialized")
            .as_ptr(),
    );
    // CPython 3.14 exports the live `batched` iterator TypeDef.  The source is
    // consumed only when `__next__` requests one batch.
    crate::module_ns_store(
        ns,
        "batched",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::BATCHED_TYPE)
            .expect("itertools.batched TypeDef initialized")
            .as_ptr(),
    );
}
