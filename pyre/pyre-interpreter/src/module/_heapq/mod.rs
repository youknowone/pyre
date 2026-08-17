//! `_heapq` accelerator module — `Modules/_heapqmodule.c`.
//!
//! PyPy has no interpreter-level `_heapq`: `heapq.py`'s `from _heapq import *`
//! finds nothing there and the app-level algorithms stand on their own. So
//! this module has no PyPy owner to follow and the C file is the model, the
//! same footing `_bisect` sits on.
//!
//! Every comparison here runs arbitrary Python, and so does every element
//! read — a list under the integer strategy boxes on `getitem`. Both can
//! collect and relocate a young object, and `__lt__` can resize the heap on
//! top of that. The C file answers the resize with a size re-check after each
//! comparison; pyre answers the relocation the way `_bisect` does, by holding
//! the heap and any element that must survive a call in a shadow-stack slot.
//! Nothing is carried across a call in a Rust local.

use pyre_object::gc_roots::{pin_root, push_roots, shadow_stack_get, shadow_stack_len};
use pyre_object::*;

/// The one thing the min-heap and max-heap variants differ by: which way
/// round the two operands of `<` go. `_heapqmodule.c` spells this as a second
/// copy of each function (`siftdown_max`, `siftup_max`); the copies are
/// otherwise identical.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Order {
    Min,
    Max,
}

impl Order {
    /// True when `first` belongs above `second` in this heap's ordering.
    /// Min asks `first < second`, max asks `second < first`.
    fn precedes(self, first: PyObjectRef, second: PyObjectRef) -> Result<bool, crate::PyError> {
        match self {
            Order::Min => less_than(first, second),
            Order::Max => less_than(second, first),
        }
    }
}

/// `PyObject_RichCompareBool(left, right, Py_LT)`.
fn less_than(left: PyObjectRef, right: PyObjectRef) -> Result<bool, crate::PyError> {
    let result = crate::objspace::descroperation::compare(
        left,
        right,
        crate::objspace::descroperation::CompareOp::Lt,
    )?;
    crate::baseobjspace::is_true(result)
}

/// Pin `value` for the enclosing `push_roots` scope and return the slot that
/// owns it from here on — the pin queries the collector, so the slot, not
/// this argument, is what holds the forwarded object afterwards.
fn pin(value: PyObjectRef) -> usize {
    let slot = shadow_stack_len();
    pin_root(value);
    slot
}

/// The heap argument, held as a shadow-stack slot.
#[derive(Clone, Copy)]
struct Heap(usize);

impl Heap {
    fn get(self) -> PyObjectRef {
        shadow_stack_get(self.0)
    }

    fn len(self) -> i64 {
        unsafe { w_list_len(self.get()) as i64 }
    }

    fn item(self, index: i64) -> Result<PyObjectRef, crate::PyError> {
        unsafe { w_list_getitem(self.get(), index) }.ok_or_else(index_out_of_range)
    }

    fn set_item(self, index: i64, value: PyObjectRef) -> Result<(), crate::PyError> {
        if unsafe { w_list_setitem(self.get(), index, value) } {
            Ok(())
        } else {
            Err(index_out_of_range())
        }
    }

    /// Exchange two elements, re-reading both first: the caller has run a
    /// comparison since it last looked, and `__lt__` may have replaced either.
    /// Both are rooted across the writes — a store that promotes the list off
    /// its integer strategy allocates.
    fn swap(self, left: i64, right: i64) -> Result<(), crate::PyError> {
        let _roots = push_roots();
        let left_item = pin(self.item(left)?);
        let right_item = pin(self.item(right)?);
        self.set_item(left, shadow_stack_get(right_item))?;
        self.set_item(right, shadow_stack_get(left_item))
    }

    /// Read two elements and ask `order` which of them comes first. The left
    /// one is rooted because reading the right one can box, and boxing can
    /// collect.
    fn compare_items(self, left: i64, right: i64, order: Order) -> Result<bool, crate::PyError> {
        let _roots = push_roots();
        let left_item = pin(self.item(left)?);
        let right_item = self.item(right)?;
        order.precedes(shadow_stack_get(left_item), right_item)
    }
}

fn index_out_of_range() -> crate::PyError {
    crate::PyError::index_error("index out of range")
}

fn changed_size() -> crate::PyError {
    crate::PyError::runtime_error("list changed size during iteration")
}

/// The `heap: object(subclass_of='&PyList_Type')` converter. `position` names
/// the argument the way Argument Clinic does: by number when the function
/// takes more than one, and not at all when `heap` is the only one.
fn heap_argument(
    heap: PyObjectRef,
    function: &str,
    position: Option<usize>,
) -> Result<PyObjectRef, crate::PyError> {
    if unsafe { crate::baseobjspace::isinstance_list_w(heap) } {
        return Ok(heap);
    }
    let argument = match position {
        Some(position) => format!("argument {position}"),
        None => "argument".to_string(),
    };
    Err(crate::PyError::type_error(format!(
        "{function}() {argument} must be list, not {}",
        crate::type_methods::arg_type_name(heap)
    )))
}

/// `siftdown` / `siftdown_max`: follow the path to the root, moving parents
/// down until finding a place the item at `pos` fits.
fn siftdown(heap: Heap, startpos: i64, pos: i64, order: Order) -> Result<(), crate::PyError> {
    let size = heap.len();
    if pos >= size {
        return Err(index_out_of_range());
    }
    let mut pos = pos;
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        let fits = heap.compare_items(pos, parentpos, order)?;
        if size != heap.len() {
            return Err(changed_size());
        }
        if !fits {
            break;
        }
        heap.swap(parentpos, pos)?;
        pos = parentpos;
    }
    Ok(())
}

/// `siftup` / `siftup_max`: bubble the preceding child up until hitting a
/// leaf, then bubble the displaced item to its final resting place by sifting
/// its parents down.
fn siftup(heap: Heap, pos: i64, order: Order) -> Result<(), crate::PyError> {
    let endpos = heap.len();
    let startpos = pos;
    if pos >= endpos {
        return Err(index_out_of_range());
    }
    let mut pos = pos;
    let limit = endpos >> 1; /* smallest pos that has no child */
    while pos < limit {
        /* Set childpos to the index of the child that goes up. */
        let mut childpos = 2 * pos + 1; /* leftmost child position */
        if childpos + 1 < endpos {
            if !heap.compare_items(childpos, childpos + 1, order)? {
                childpos += 1;
            }
            if endpos != heap.len() {
                return Err(changed_size());
            }
        }
        /* Move that child up. */
        heap.swap(childpos, pos)?;
        pos = childpos;
    }
    siftdown(heap, startpos, pos, order)
}

fn keep_top_bit(mut n: i64) -> i64 {
    let mut i = 0;
    while n > 1 {
        n >>= 1;
        i += 1;
    }
    n << i
}

/* Cache friendly version of heapify()
   -----------------------------------

   Build-up a heap in O(n) time by performing siftup() operations
   on nodes whose children are already heaps.

   The simplest way is to sift the nodes in reverse order from
   n//2-1 to 0 inclusive.  The downside is that children may be
   out of cache by the time their parent is reached.

   A better way is to not wait for the children to go out of cache.
   Once a sibling pair of child nodes have been sifted, immediately
   sift their parent node (while the children are still in cache).

   Both ways build child heaps before their parents, so both ways
   do the exact same number of comparisons and produce exactly
   the same heap.  The only difference is that the traversal
   order is optimized for cache efficiency.
*/
fn cache_friendly_heapify(heap: Heap, order: Order) -> Result<(), crate::PyError> {
    let m = heap.len() >> 1; /* index of first childless node */
    let leftmost = keep_top_bit(m + 1) - 1; /* leftmost node in row of m */
    let mhalf = m >> 1; /* parent of first childless node */

    let sift_ancestors = |start: i64| -> Result<(), crate::PyError> {
        let mut j = start;
        loop {
            siftup(heap, j, order)?;
            if j & 1 == 0 {
                return Ok(());
            }
            j >>= 1;
        }
    };
    for i in (mhalf..leftmost).rev() {
        sift_ancestors(i)?;
    }
    for i in (leftmost..m).rev() {
        sift_ancestors(i)?;
    }
    Ok(())
}

fn heapify_internal(heap: Heap, order: Order) -> Result<(), crate::PyError> {
    /* For heaps likely to be bigger than L1 cache, we use the cache
       friendly heapify function.  For smaller heaps that fit entirely
       in cache, we prefer the simpler algorithm with less branching.
    */
    let n = heap.len();
    if n > 2500 {
        return cache_friendly_heapify(heap, order);
    }
    /* Transform bottom-up.  The largest index there's any point to
       looking at is the largest with a child index in-range, so must
       have 2*i + 1 < n, or i < (n-1)/2, which is n//2 - 1.
    */
    for i in (0..(n >> 1)).rev() {
        siftup(heap, i, order)?;
    }
    Ok(())
}

fn heappush_impl(
    args: &[PyObjectRef],
    function: &str,
    order: Order,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = push_roots();
    let heap = Heap(pin(heap_argument(args[0], function, Some(1))?));
    let item = pin(args[1]);
    unsafe { w_list_append(heap.get(), shadow_stack_get(item)) };
    siftdown(heap, 0, heap.len() - 1, order)?;
    Ok(w_none())
}

fn heappop_impl(
    args: &[PyObjectRef],
    function: &str,
    order: Order,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = push_roots();
    let heap = Heap(pin(heap_argument(args[0], function, None)?));
    /* raises IndexError if the heap is empty */
    if heap.len() == 0 {
        return Err(index_out_of_range());
    }
    let lastelt = pin(unsafe { w_list_pop_end(heap.get()) }.ok_or_else(index_out_of_range)?);
    if heap.len() == 0 {
        return Ok(shadow_stack_get(lastelt));
    }
    let returnitem = pin(heap.item(0)?);
    heap.set_item(0, shadow_stack_get(lastelt))?;
    siftup(heap, 0, order)?;
    Ok(shadow_stack_get(returnitem))
}

fn heapreplace_impl(
    args: &[PyObjectRef],
    function: &str,
    order: Order,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = push_roots();
    let heap = Heap(pin(heap_argument(args[0], function, Some(1))?));
    let item = pin(args[1]);
    if heap.len() == 0 {
        return Err(index_out_of_range());
    }
    let returnitem = pin(heap.item(0)?);
    heap.set_item(0, shadow_stack_get(item))?;
    siftup(heap, 0, order)?;
    Ok(shadow_stack_get(returnitem))
}

fn heappushpop_impl(
    args: &[PyObjectRef],
    function: &str,
    order: Order,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = push_roots();
    let heap = Heap(pin(heap_argument(args[0], function, Some(1))?));
    let item = pin(args[1]);
    if heap.len() == 0 {
        return Ok(shadow_stack_get(item));
    }
    let top = pin(heap.item(0)?);
    // The pushed item stays out of the heap unless the top comes out first.
    if !order.precedes(shadow_stack_get(top), shadow_stack_get(item))? {
        return Ok(shadow_stack_get(item));
    }
    // bpo-39421: that comparison could have emptied the heap.
    if heap.len() == 0 {
        return Err(index_out_of_range());
    }
    let returnitem = pin(heap.item(0)?);
    heap.set_item(0, shadow_stack_get(item))?;
    siftup(heap, 0, order)?;
    Ok(shadow_stack_get(returnitem))
}

fn heapify_impl(
    args: &[PyObjectRef],
    function: &str,
    order: Order,
) -> Result<PyObjectRef, crate::PyError> {
    let _roots = push_roots();
    let heap = Heap(pin(heap_argument(args[0], function, None)?));
    heapify_internal(heap, order)?;
    Ok(w_none())
}

crate::py_module! {
    "_heapq",
    functions: {
        "heappush"        / 2 = |args| heappush_impl(args, "heappush", Order::Min),
        "heappushpop"     / 2 = |args| heappushpop_impl(args, "heappushpop", Order::Min),
        "heappop"         / 1 = |args| heappop_impl(args, "heappop", Order::Min),
        "heapreplace"     / 2 = |args| heapreplace_impl(args, "heapreplace", Order::Min),
        "heapify"         / 1 = |args| heapify_impl(args, "heapify", Order::Min),

        "heappush_max"    / 2 = |args| heappush_impl(args, "heappush_max", Order::Max),
        "heappushpop_max" / 2 = |args| heappushpop_impl(args, "heappushpop_max", Order::Max),
        "heappop_max"     / 1 = |args| heappop_impl(args, "heappop_max", Order::Max),
        "heapreplace_max" / 2 = |args| heapreplace_impl(args, "heapreplace_max", Order::Max),
        "heapify_max"     / 1 = |args| heapify_impl(args, "heapify_max", Order::Max),
    },
}
