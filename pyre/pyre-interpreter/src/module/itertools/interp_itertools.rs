//! itertools implementation — PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Verbatim move of the inline block previously in importing.rs.

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
    // PyPy W_GroupBy and W_GroupByIterator share the live source cursor.
    crate::module_ns_store(
        ns,
        "groupby",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::GROUPBY_TYPE)
            .expect("itertools.groupby TypeDef initialized")
            .as_ptr(),
    );
    crate::module_ns_store(
        ns,
        "_grouper",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::GROUPBY_ITERATOR_TYPE)
            .expect("itertools._grouper TypeDef initialized")
            .as_ptr(),
    );
    // tee(iterable, n=2) — the itertools-docs pure-Python equivalent.  A native
    // dataobject type would only save buffer copies; the deque-per-copy recipe
    // keeps the copies lazy and independent, which is what callers observe.
    crate::importing::appleveldef_install(ns, TEE_SRC, "<inline>", &["tee"]);
    // PyPy W_Permutations: retain the pool, indices, and rollover cycles,
    // yielding one permutation at a time.
    crate::module_ns_store(
        ns,
        "permutations",
        crate::typedef::gettypefor(&pyre_object::interp_itertools::PERMUTATIONS_TYPE)
            .expect("itertools.permutations TypeDef initialized")
            .as_ptr(),
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
    // PyPy W_CombinationsWithReplacement: retain pool/index/result state and
    // advance one repeated combination at a time.
    crate::module_ns_store(
        ns,
        "combinations_with_replacement",
        crate::typedef::gettypefor(
            &pyre_object::interp_itertools::COMBINATIONS_WITH_REPLACEMENT_TYPE,
        )
        .expect("itertools.combinations_with_replacement TypeDef initialized")
        .as_ptr(),
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
