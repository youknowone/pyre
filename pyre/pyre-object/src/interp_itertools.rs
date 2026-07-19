//! itertools module — iterator objects.
//!
//! PyPy: pypy/module/itertools/interp_itertools.py
//!
//! Line-by-line port of the W_Count / W_Repeat / W_Cycle / W_Chain etc.
//! classes. Each class becomes a `#[repr(C)]` struct with a static PyType.

use crate::pyobject::*;
use pyre_macros::pyre_class;

// ── W_Count — pypy/module/itertools/interp_itertools.py:class W_Count ──
//
// ```python
// class W_Count(W_Root):
//     def __init__(self, space, w_firstval, w_step):
//         self.space = space
//         self.w_c = w_firstval
//         self.w_step = w_step
//
//     def iter_w(self):
//         return self
//
//     def next_w(self):
//         w_c = self.w_c
//         self.w_c = self.space.add(w_c, self.w_step)
//         return w_c
// ```
//
// The receiver stores `w_c` (current value) and `w_step` which are both
// PyObjectRef so that count(1.5, 0.5) works for float too.

#[pyre_class("itertools.count", type_id = 24, static_name = "COUNT")]
pub struct W_Count {
    pub w_c: PyObjectRef,
    pub w_step: PyObjectRef,
}

pub fn w_count_new(w_firstval: PyObjectRef, w_step: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_firstval);
    crate::gc_roots::pin_root(w_step);
    W_Count::allocate(W_Count {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_c: w_firstval,
        w_step,
    })
}

/// Check if an object is a `W_Count`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_count(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COUNT_TYPE) }
}

/// Read the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_c(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_c }
}

/// Write the current `w_c` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_set_c(obj: PyObjectRef, v: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Count)).w_c = v;
    }
}

/// Read the `w_step` field.
///
/// # Safety
/// `obj` must point to a valid `W_Count`.
pub unsafe fn w_count_get_step(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Count)).w_step }
}

// ── W_Repeat — pypy/module/itertools/interp_itertools.py:class W_Repeat ──
//
// ```python
// class W_Repeat(W_Root):
//     def __init__(self, space, w_obj, w_times):
//         self.space = space
//         self.w_obj = w_obj
//         if w_times is None:
//             self.counting = False
//             self.count = 0
//         else:
//             self.counting = True
//             self.count = max(self.space.int_w(w_times), 0)
//
//     def next_w(self):
//         if self.counting:
//             if self.count <= 0:
//                 raise OperationError(self.space.w_StopIteration, self.space.w_None)
//             self.count -= 1
//         return self.w_obj
// ```

#[pyre_class("itertools.repeat", type_id = 25, static_name = "REPEAT")]
pub struct W_Repeat {
    pub w_obj: PyObjectRef,
    pub counting: bool,
    pub count: i64,
}

pub fn w_repeat_new(w_obj: PyObjectRef, w_times: Option<i64>) -> PyObjectRef {
    let (counting, count) = match w_times {
        None => (false, 0),
        Some(n) => (true, n.max(0)),
    };
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_obj);
    W_Repeat::allocate(W_Repeat {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_obj,
        counting,
        count,
    })
}

/// Check if an object is a `W_Repeat`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_repeat(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &REPEAT_TYPE) }
}

/// Read the `w_obj` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_obj(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Repeat)).w_obj }
}

/// Read the `counting` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_counting(obj: PyObjectRef) -> bool {
    unsafe { (*(obj as *const W_Repeat)).counting }
}

/// Read the `count` field.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_get_count(obj: PyObjectRef) -> i64 {
    unsafe { (*(obj as *const W_Repeat)).count }
}

/// Decrement the `count` field by 1.
///
/// # Safety
/// `obj` must point to a valid `W_Repeat`.
pub unsafe fn w_repeat_dec_count(obj: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Repeat)).count -= 1;
    }
}

// ── W_TakeWhile — pypy/module/itertools/interp_itertools.py:class W_TakeWhile ──
//
// ```python
// class W_TakeWhile(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
//         self.stopped = False
// ```
//
// `next_w` lives in the interpreter (`baseobjspace::next`) because it
// calls the predicate.

#[pyre_class("itertools.takewhile", type_id = 54, static_name = "TAKEWHILE")]
pub struct W_TakeWhile {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
    pub stopped: bool,
}

/// `w_iterable` must already be an iterator (`space.iter` applied by the
/// caller, matching `W_TakeWhile.__init__`).
pub fn w_takewhile_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_predicate);
    crate::gc_roots::pin_root(w_iterable);
    W_TakeWhile::allocate(W_TakeWhile {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
        stopped: false,
    })
}

/// Check if an object is a `W_TakeWhile`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_takewhile(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &TAKEWHILE_TYPE) }
}

// ── W_DropWhile — pypy/module/itertools/interp_itertools.py:class W_DropWhile ──
//
// ```python
// class W_DropWhile(W_Root):
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
//         self.started = False
// ```

#[pyre_class("itertools.dropwhile", type_id = 55, static_name = "DROPWHILE")]
pub struct W_DropWhile {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
    pub started: bool,
}

/// `w_iterable` must already be an iterator (`space.iter` applied by the
/// caller, matching `W_DropWhile.__init__`).
pub fn w_dropwhile_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_predicate);
    crate::gc_roots::pin_root(w_iterable);
    W_DropWhile::allocate(W_DropWhile {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
        started: false,
    })
}

/// Check if an object is a `W_DropWhile`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_dropwhile(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DROPWHILE_TYPE) }
}

// ── W_FilterFalse — pypy/module/itertools/interp_itertools.py:class W_FilterFalse ──
//
// Subclass of `W_Filter` (`pypy/module/__builtin__/functional.py:916`)
// with `reverse = True`:
//
// ```python
// class W_Filter(W_Root):
//     reverse = False # set to True in itertools
//     def __init__(self, space, w_predicate, w_iterable):
//         self.space = space
//         if space.is_w(w_predicate, space.w_None):
//             self.w_predicate = None
//         else:
//             self.w_predicate = w_predicate
//         self.w_iterable = space.iter(w_iterable)
// ```
//
// `w_predicate` is PY_NULL when the Python-level predicate was None.

#[pyre_class("itertools.filterfalse", type_id = 56, static_name = "FILTERFALSE")]
pub struct W_FilterFalse {
    pub w_predicate: PyObjectRef,
    pub w_iterable: PyObjectRef,
}

/// `w_iterable` must already be an iterator; `w_predicate` is PY_NULL
/// for a None predicate (`W_Filter.__init__`).
pub fn w_filterfalse_new(w_predicate: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    if !w_predicate.is_null() {
        crate::gc_roots::pin_root(w_predicate);
    }
    crate::gc_roots::pin_root(w_iterable);
    W_FilterFalse::allocate(W_FilterFalse {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_predicate,
        w_iterable,
    })
}

/// Check if an object is a `W_FilterFalse`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_filterfalse(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FILTERFALSE_TYPE) }
}

// ── W_Compress — pypy/module/itertools/interp_itertools.py:W_Compress ──
//
// ```python
// class W_Compress(W_Root):
//     def __init__(self, space, w_data, w_selectors):
//         self.space = space
//         self.w_data = space.iter(w_data)
//         self.w_selectors = space.iter(w_selectors)
// ```
//
// `next_w` lives in the interpreter because it invokes both iterators and
// selector truth testing.  The two live iterators are traced GC edges.

#[pyre_class("itertools.compress", static_name = "COMPRESS")]
pub struct W_Compress {
    pub w_data: PyObjectRef,
    pub w_selectors: PyObjectRef,
}

/// Both arguments must already have had `space.iter` applied, matching
/// `W_Compress.__init__`.
pub fn w_compress_new(w_data: PyObjectRef, w_selectors: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_data);
    crate::gc_roots::pin_root(w_selectors);
    W_Compress::allocate(W_Compress {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_data,
        w_selectors,
    })
}

/// Check if an object is a `W_Compress`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_compress(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COMPRESS_TYPE) }
}

// ── W_StarMap — pypy/module/itertools/interp_itertools.py:W_StarMap ──
//
// ```python
// class W_StarMap(W_Root):
//     def __init__(self, space, w_fun, w_iterable):
//         self.space = space
//         self.w_fun = w_fun
//         self.w_iterable = self.space.iter(w_iterable)
// ```
//
// `next_w` lives in the interpreter because it expands the next object into
// call arguments and invokes `w_fun`.  Both fields are traced GC edges.

#[pyre_class("itertools.starmap", static_name = "STARMAP")]
pub struct W_StarMap {
    pub w_fun: PyObjectRef,
    pub w_iterable: PyObjectRef,
}

/// `w_iterable` must already have had `space.iter` applied.
pub fn w_starmap_new(w_fun: PyObjectRef, w_iterable: PyObjectRef) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_fun);
    crate::gc_roots::pin_root(w_iterable);
    W_StarMap::allocate(W_StarMap {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_fun,
        w_iterable,
    })
}

/// Check if an object is a `W_StarMap`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_starmap(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &STARMAP_TYPE) }
}

// ── W_Accumulate — pypy/module/itertools/interp_itertools.py:W_Accumulate ──
//
// PyPy keeps the live iterator, optional binary function, running total, and
// pending initial value on the iterator object.  PY_NULL represents PyPy's
// internal `None` sentinel for `w_func` and `w_total`; `w_initial` is the
// Python-level value and is reset to Python None after it is yielded.

#[pyre_class("itertools.accumulate", static_name = "ACCUMULATE")]
pub struct W_Accumulate {
    pub w_iterable: PyObjectRef,
    pub w_func: PyObjectRef,
    pub w_total: PyObjectRef,
    pub w_initial: PyObjectRef,
}

pub fn w_accumulate_new(
    w_iterable: PyObjectRef,
    w_func: PyObjectRef,
    w_initial: PyObjectRef,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterable);
    if !w_func.is_null() {
        crate::gc_roots::pin_root(w_func);
    }
    crate::gc_roots::pin_root(w_initial);
    W_Accumulate::allocate(W_Accumulate {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterable,
        w_func,
        w_total: std::ptr::null_mut(),
        w_initial,
    })
}

#[inline]
pub unsafe fn is_accumulate(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ACCUMULATE_TYPE) }
}

#[inline]
pub unsafe fn w_accumulate_set_total(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Accumulate)).w_total = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

#[inline]
pub unsafe fn w_accumulate_set_initial(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Accumulate)).w_initial = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

// ── W_ZipLongest — pypy/module/itertools/interp_itertools.py:W_ZipLongest ──

#[pyre_class("itertools.zip_longest", static_name = "ZIP_LONGEST")]
pub struct W_ZipLongest {
    /// A Python list of live iterators; exhausted entries become Python None.
    pub w_iterators: PyObjectRef,
    pub w_fillvalue: PyObjectRef,
    pub active: i64,
}

pub fn w_zip_longest_new(
    w_iterators: PyObjectRef,
    w_fillvalue: PyObjectRef,
    active: i64,
) -> PyObjectRef {
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterators);
    crate::gc_roots::pin_root(w_fillvalue);
    W_ZipLongest::allocate(W_ZipLongest {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterators,
        w_fillvalue,
        active,
    })
}

#[inline]
pub unsafe fn is_zip_longest(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ZIP_LONGEST_TYPE) }
}

// ── W_Pairwise — pypy/module/itertools/interp_itertools.py:class W_Pairwise ──
//
// ```python
// class W_Pairwise(W_Root):
//     def __init__(self, space, w_iterator):
//         self.space = space
//         self.w_iterator = w_iterator
//         self.w_prev = None
// ```
//
// `w_prev` is PY_NULL until the first `next_w`.

#[pyre_class("itertools.pairwise", type_id = 57, static_name = "PAIRWISE")]
pub struct W_Pairwise {
    pub w_iterator: PyObjectRef,
    pub w_prev: PyObjectRef,
}

/// `w_iterator` must already be an iterator (`W_Pairwise__new__` applies
/// `space.iter`).
pub fn w_pairwise_new(w_iterator: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterator);
    W_Pairwise::allocate(W_Pairwise {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterator,
        w_prev: std::ptr::null_mut(),
    })
}

/// Check if an object is a `W_Pairwise`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_pairwise(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &PAIRWISE_TYPE) }
}

// ── W_Cycle — pypy/module/itertools/interp_itertools.py:class W_Cycle ──
//
// ```python
// class W_Cycle(W_Root):
//     def __init__(self, space, w_iterable):
//         self.space = space
//         self.saved_w = []
//         self.w_iterable = space.iter(w_iterable)
//         self.index = 0    # 0 during the first iteration; > 0 afterwards
// ```
//
// `next_w` (in `baseobjspace::next`) pulls from `w_iterable` on the first
// pass, appending each element to `saved`; once the source is exhausted it
// replays `saved` forever.  `index` is 0 during the first pass and > 0 once
// cycling.  Unlike the predicate iterators (whose referents are rooted by
// the live predicate/iterable), `saved` is owned solely by the W_Cycle, so
// the GC must trace it — the type is registered in the JIT GC driver
// (`pyre-jit/src/eval.rs`) via `register_pyre_class` in AUTO-ID mode.
#[pyre_class("itertools.cycle", static_name = "CYCLE")]
pub struct W_Cycle {
    pub w_iterable: PyObjectRef,
    pub saved: PyObjectRef,
    pub index: i64,
}

/// `w_iterable` must already be an iterator (`cycle`'s registrar applies
/// `space.iter`).  Allocates an empty `saved` list.
pub fn w_cycle_new(w_iterable: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterable);
    let saved = crate::listobject::w_list_new(Vec::new());
    crate::gc_roots::pin_root(saved);
    W_Cycle::allocate(W_Cycle {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterable,
        saved,
        index: 0,
    })
}

/// Check if an object is a `W_Cycle`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_cycle(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CYCLE_TYPE) }
}

// ── W_Chain — pypy/module/itertools/interp_itertools.py:class W_Chain ──
//
// ```python
// class W_Chain(W_Root):
//     def __init__(self, space, w_iterables):
//         self.space = space
//         self.w_iterables = w_iterables
//         self.w_it = None
//
//     def _advance(self):
//         self.w_it = self.space.iter(self.space.next(self.w_iterables))
//
//     def next_w(self):
//         if not self.w_it:
//             self._advance()     # may raise StopIteration
//         while True:
//             try:
//                 return self.space.next(self.w_it)
//             except OperationError as e:
//                 if e.match(self.space, self.space.w_StopIteration):
//                     self.w_it = None
//                     self._advance()
//                 else:
//                     raise
// ```
//
// `w_iterables` is an iterator over the source iterables; `w_it` is the
// current active sub-iterator (PY_NULL until the first `next_w`, and reset
// to PY_NULL each time a sub-iterator is exhausted).  Both pointer fields
// are owned solely by the W_Chain, so the GC must trace them — the type is
// registered in the JIT GC driver (`pyre-jit/src/eval.rs`) via
// `register_pyre_class` in AUTO-ID mode.
#[pyre_class("itertools.chain", static_name = "CHAIN")]
pub struct W_Chain {
    pub w_iterables: PyObjectRef,
    pub w_it: PyObjectRef,
}

/// `w_iterables` must already be an iterator over the source iterables
/// (`chain` / `chain.from_iterable` apply `space.iter`).  `w_it` starts
/// PY_NULL (no active sub-iterator yet).
pub fn w_chain_new(w_iterables: PyObjectRef) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`).
    let _roots = crate::gc_roots::push_roots();
    crate::gc_roots::pin_root(w_iterables);
    W_Chain::allocate(W_Chain {
        ob: PyObject {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        },
        w_iterables,
        w_it: std::ptr::null_mut(),
    })
}

/// Check if an object is a `W_Chain`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_chain(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CHAIN_TYPE) }
}

/// Read the `w_iterables` field of a `W_Chain`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_get_iterables(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Chain)).w_iterables }
}

/// Read the `w_it` field of a `W_Chain`.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_get_it(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const W_Chain)).w_it }
}

/// Store the `w_iterables` field of a `W_Chain`.  Reassigning a pointer
/// field can record an old→young edge, so the GC write barrier runs.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_set_iterables(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Chain)).w_iterables = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

/// Store the `w_it` field of a `W_Chain`.  Reassigning a pointer field can
/// record an old→young edge, so the GC write barrier runs.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `W_Chain`.
#[inline]
pub unsafe fn w_chain_set_it(obj: PyObjectRef, w_value: PyObjectRef) {
    unsafe {
        (*(obj as *mut W_Chain)).w_it = w_value;
        crate::gc_hook::try_gc_write_barrier(obj as *mut u8);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w_count_gc_type_id_matches_descr() {
        assert_eq!(W_COUNT_GC_TYPE_ID, 24);
        assert_eq!(
            <W_Count as crate::lltype::GcType>::type_id(),
            W_COUNT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Count as crate::lltype::GcType>::SIZE,
            W_COUNT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_repeat_gc_type_id_matches_descr() {
        assert_eq!(W_REPEAT_GC_TYPE_ID, 25);
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::type_id(),
            W_REPEAT_GC_TYPE_ID
        );
        assert_eq!(
            <W_Repeat as crate::lltype::GcType>::SIZE,
            W_REPEAT_OBJECT_SIZE
        );
    }

    #[test]
    fn w_takewhile_gc_type_id_matches_descr() {
        assert_eq!(W_TAKEWHILE_GC_TYPE_ID, 54);
        assert_eq!(
            <W_TakeWhile as crate::lltype::GcType>::type_id(),
            W_TAKEWHILE_GC_TYPE_ID
        );
        assert_eq!(
            <W_TakeWhile as crate::lltype::GcType>::SIZE,
            W_TAKEWHILE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_dropwhile_gc_type_id_matches_descr() {
        assert_eq!(W_DROPWHILE_GC_TYPE_ID, 55);
        assert_eq!(
            <W_DropWhile as crate::lltype::GcType>::type_id(),
            W_DROPWHILE_GC_TYPE_ID
        );
        assert_eq!(
            <W_DropWhile as crate::lltype::GcType>::SIZE,
            W_DROPWHILE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_filterfalse_gc_type_id_matches_descr() {
        assert_eq!(W_FILTERFALSE_GC_TYPE_ID, 56);
        assert_eq!(
            <W_FilterFalse as crate::lltype::GcType>::type_id(),
            W_FILTERFALSE_GC_TYPE_ID
        );
        assert_eq!(
            <W_FilterFalse as crate::lltype::GcType>::SIZE,
            W_FILTERFALSE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_compress_gc_descriptor_traces_both_iterator_fields() {
        assert_eq!(W_COMPRESS_GC_PTR_OFFSETS.len(), 2);
        assert_eq!(
            W_COMPRESS_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_Compress, w_data)
        );
        assert_eq!(
            W_COMPRESS_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_Compress, w_selectors)
        );
        assert_eq!(
            <W_Compress as crate::lltype::GcType>::SIZE,
            W_COMPRESS_OBJECT_SIZE
        );
    }

    #[test]
    fn w_starmap_gc_descriptor_traces_function_and_iterator() {
        assert_eq!(W_STARMAP_GC_PTR_OFFSETS.len(), 2);
        assert_eq!(
            W_STARMAP_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_StarMap, w_fun)
        );
        assert_eq!(
            W_STARMAP_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_StarMap, w_iterable)
        );
        assert_eq!(
            <W_StarMap as crate::lltype::GcType>::SIZE,
            W_STARMAP_OBJECT_SIZE
        );
    }

    #[test]
    fn w_accumulate_gc_descriptor_traces_live_state() {
        assert_eq!(W_ACCUMULATE_GC_PTR_OFFSETS.len(), 4);
        assert_eq!(
            W_ACCUMULATE_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_Accumulate, w_iterable),
                std::mem::offset_of!(W_Accumulate, w_func),
                std::mem::offset_of!(W_Accumulate, w_total),
                std::mem::offset_of!(W_Accumulate, w_initial),
            ]
        );
        assert_eq!(
            <W_Accumulate as crate::lltype::GcType>::SIZE,
            W_ACCUMULATE_OBJECT_SIZE
        );
    }

    #[test]
    fn w_zip_longest_gc_descriptor_traces_iterators_and_fillvalue() {
        assert_eq!(W_ZIP_LONGEST_GC_PTR_OFFSETS.len(), 2);
        assert_eq!(
            W_ZIP_LONGEST_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(W_ZipLongest, w_iterators),
                std::mem::offset_of!(W_ZipLongest, w_fillvalue),
            ]
        );
        assert_eq!(
            <W_ZipLongest as crate::lltype::GcType>::SIZE,
            W_ZIP_LONGEST_OBJECT_SIZE
        );
    }

    #[test]
    fn w_pairwise_gc_type_id_matches_descr() {
        assert_eq!(W_PAIRWISE_GC_TYPE_ID, 57);
        assert_eq!(
            <W_Pairwise as crate::lltype::GcType>::type_id(),
            W_PAIRWISE_GC_TYPE_ID
        );
        assert_eq!(
            <W_Pairwise as crate::lltype::GcType>::SIZE,
            W_PAIRWISE_OBJECT_SIZE
        );
    }

    // W_Cycle is registered in AUTO-ID mode (no `type_id = N`), so the GC
    // tid is stamped at JIT-driver init rather than asserted against a
    // constant.  What must hold is that both traced edges — the source
    // `w_iterable` and the owned `saved` replay buffer — are reported to
    // the collector, and that the descriptor reflects the struct's size.
    #[test]
    fn w_cycle_gc_descriptor_traces_both_pointer_fields() {
        assert_eq!(W_CYCLE_GC_PTR_OFFSETS.len(), 2);
        assert_eq!(
            W_CYCLE_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_Cycle, w_iterable)
        );
        assert_eq!(
            W_CYCLE_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_Cycle, saved)
        );
        assert_eq!(
            <W_Cycle as crate::lltype::GcType>::SIZE,
            W_CYCLE_OBJECT_SIZE
        );
    }

    // W_Chain is registered in AUTO-ID mode (no `type_id = N`), so the GC
    // tid is stamped at JIT-driver init rather than asserted against a
    // constant.  What must hold is that both traced edges — the source
    // `w_iterables` iterator and the current sub-iterator `w_it` — are
    // reported to the collector, and that the descriptor reflects the
    // struct's size.
    #[test]
    fn w_chain_gc_descriptor_traces_both_pointer_fields() {
        assert_eq!(W_CHAIN_GC_PTR_OFFSETS.len(), 2);
        assert_eq!(
            W_CHAIN_GC_PTR_OFFSETS[0],
            std::mem::offset_of!(W_Chain, w_iterables)
        );
        assert_eq!(
            W_CHAIN_GC_PTR_OFFSETS[1],
            std::mem::offset_of!(W_Chain, w_it)
        );
        assert_eq!(
            <W_Chain as crate::lltype::GcType>::SIZE,
            W_CHAIN_OBJECT_SIZE
        );
    }
}
