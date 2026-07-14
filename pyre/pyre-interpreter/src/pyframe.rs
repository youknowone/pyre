//! PyFrame — execution frame for Python bytecode.
#![allow(non_snake_case)]
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::rc::Rc;

use crate::{CodeFlags, CodeObject};
use crate::{DictStorage, PyExecutionContext};
use pyre_object::FixedObjectArray;
use pyre_object::*;

// Ensure *const PyExecutionContext and Rc<PyExecutionContext> have the same
// size so that PyFrame field offsets are preserved after the switch.
const _: () = assert!(
    std::mem::size_of::<*const PyExecutionContext>()
        == std::mem::size_of::<Rc<PyExecutionContext>>()
);

/// `types.FrameType` — the PyType every `PyFrame`'s `ob_header` points at.
/// pyframe.py `class PyFrame(W_Root)` with `typedef.py:736 PyFrame.typedef
/// = TypeDef("frame", ...)`. The descriptors (`f_back`, `f_locals`, …) are
/// attached to this type by the `frame` typedef; until then it is a bare
/// identity tag so a frame carries a valid `ob_type` like every other
/// W_Root (mirrors `pytraceback::PYTRACEBACK_TYPE`).
pub static FRAME_TYPE: PyType = new_pytype("frame");

/// Build the `ob_header` for a freshly-created `PyFrame` — `ob_type`
/// pinned to [`FRAME_TYPE`], `w_class` the cached `W_TypeObject`
/// (null during bootstrap before `init_typeobjects`). Mirrors
/// `pytraceback::pytraceback_new`'s header construction.
#[inline]
fn frame_ob_header() -> PyObject {
    PyObject {
        ob_type: &FRAME_TYPE as *const PyType,
        w_class: get_instantiate(&FRAME_TYPE),
    }
}

/// Execution frame for a single Python code block.
///
/// Unified `locals_cells_stack_w` array layout:
///   - indices `0..nlocals` — local variables
///   - indices `nlocals..nlocals+ncells` — cell/free variable slots
///   - indices `nlocals+ncells..` — operand stack
///
/// `valuestackdepth` is the absolute index into this array; it starts at
/// `nlocals + ncells` (empty stack) and grows upward on push.
///
/// The JIT's Virtualize pass keeps `locals_cells_stack_w` slots in CPU
/// registers during compiled code execution, eliminating heap reads/writes
/// for the hottest interpreter state.
///
/// The `vable_token` field coordinates ownership: when JIT code is
/// running, the token is nonzero and the canonical field values live
/// in registers. A "force" flushes them back to the heap.
#[repr(C)]
pub struct PyFrame {
    /// `PyObject` prefix (`ob_type` / `w_class`) making the frame a
    /// Python-visible `W_Root` — `pyframe.py class PyFrame(W_Root)`.
    /// `ob_type` points at [`FRAME_TYPE`]. Kept at offset 0 like every
    /// other `PyObject`-layout struct so `ob_type` reads land on the
    /// typeptr the JIT `GuardClass` / `type()` expect.
    pub ob_header: PyObject,
    /// Raw pointer to the shared execution context.
    /// The top-level frame leaks the Rc via `Rc::into_raw`.
    /// Callee frames just copy the pointer (no atomic refcount ops).
    pub execution_context: *const PyExecutionContext,
    /// Pointer to the Code object (PyCode).
    ///
    /// PyPy: pyframe.py `self.pycode = code` — stores the PyCode instance.
    /// Same pointer as `func.getcode()`, so `getcode(func) == frame.pycode`.
    pub pycode: *const (),
    /// pypy/interpreter/pyframe.py:84,110-112 locals_cells_stack_w
    /// `[None] * size; make_sure_not_resized(...)` → fixed-length GcArray.
    pub locals_cells_stack_w: *mut FixedObjectArray,
    /// Absolute index into `locals_cells_stack_w` marking the top of the
    /// operand stack. Starts at `nlocals + ncells` (empty stack), grows upward.
    pub valuestackdepth: usize,
    /// `pyframe.py:72 last_instr` — index of the currently-dispatching (or
    /// just-dispatched) instruction. Initialized to `-1` (frame not yet
    /// entered). `get_last_lineno` uses this for `offset2lineno`.
    ///
    /// Storage semantic matches RPython.  `pypy/interpreter/pyopcode.py:172
    /// self.last_instr = intmask(next_instr)` writes the to-be-dispatched
    /// byte-offset at the top of every dispatch iteration, and pyre's
    /// `eval.rs::eval_loop` writes `frame.last_instr = pc` at the same
    /// point.  At any inspection boundary (between opcodes, at a guard,
    /// during a handler) both sides hold "current/just-completed opcode"
    /// modulo bytes-vs-instruction units.
    ///
    /// TODO(pattern-level, not semantic): RPython's
    /// dispatch carries the next-to-execute pc as a separate `next_instr`
    /// local variable advanced by `next_instr += 2` after the opcode read.
    /// pyre packs the same information into `last_instr` plus the
    /// `next_instr()` accessor below, which returns `last_instr + 1`.
    /// `set_last_instr_from_next_instr` is the inverse setter, and the
    /// JIT vable shadow `vable_last_instr` mirrors the same field. The
    /// `±1` arithmetic on either side of the accessor cancels — the
    /// runtime never observes a divergence — so this remains a structural
    /// adaptation, not a parity bug.
    pub last_instr: isize,
    /// pyframe.py:80 escaped — see mark_as_escaped()
    pub escaped: bool,
    /// pyframe.py:82 debugdata — lazily allocated tracing/debug payload.
    /// Virtualizable static field (interp_jit.py:28).
    pub debugdata: *mut FrameDebugData,
    /// pyframe.py:86 lastblock — head of the FrameBlock linked list.
    /// Virtualizable static field (interp_jit.py:29).
    pub lastblock: *mut FrameBlock,
    /// Virtualizable token — set by JIT when this frame is virtualized.
    /// 0 = not virtualized, nonzero = pointer to JIT state.
    pub vable_token: usize,
    /// PyPy: `frame_finished_execution = False`.
    pub frame_finished_execution: bool,
    /// PyPy: `f_generator_nowref = None`.
    pub f_generator_nowref: PyObjectRef,
    /// PyPy: `w_yielding_from = None`.
    pub w_yielding_from: PyObjectRef,
    /// PyPy: `f_backref = jit.vref_None`.
    pub f_backref: *mut PyFrame,
    /// pyframe.py:115-116 — `self.builtin = space.builtin
    /// .pick_builtin(w_globals)`, gated under `honor__builtins__`
    /// (`baseobjspace::HONOR_BUILTINS`, default False).  With the option
    /// off (the default) `frame_builtin*` returns `space.builtin`, ignoring
    /// a custom `__builtins__` in globals; with it on the `pick_builtin*`
    /// family honors `globals['__builtins__']`.  Consulted by LOAD_GLOBAL's
    /// builtins fallback (`pyopcode.py:918-927`) via
    /// `space.finditem_str(self.w_builtin.w_dict, name)` — honouring
    /// dict subclass `__getitem__` overrides (`moduledef.py:102-103`)
    /// without an extra storage-keyed fast path.  `frame.get_builtin()`
    /// returns this directly so callers (`exec`'s
    /// `setdefault('__builtins__', self.get_builtin())` at
    /// `pyopcode.py:773-774`) see the picked Module, not the EC's
    /// default builtin.
    pub w_builtin: PyObjectRef,
    /// `pypy/interpreter/pyframe.py:49 self.w_globals = w_globals`
    /// — the canonical W_DictObject paired with this frame's globals.
    ///
    /// **Population**: eagerly resolved at frame construction by calling
    /// `dict_storage_to_dict(w_globals)` on the raw storage handed to the
    /// constructor (or threaded through directly by the obj-taking
    /// constructors), so every reader after construction observes the same
    /// W_DictObject across the frame's lifetime.  This is the source of
    /// truth: `get_w_globals()` and `get_w_globals_storage()` both return it
    /// directly.  Synthetic test stubs that hand-build PyFrame without a real
    /// globals leave it `PY_NULL`.
    pub w_globals: PyObjectRef,
}

/// GC type id for `PyFrame`. Reserved ahead of any callsite that allocates
/// frames through `NewWithVtable` / `New` in trace IR — the GC consults
/// the registered `TypeInfo` to write the type tag into the header. Today
/// every `PyFrame` is heap-allocated outside the nursery (`std::alloc`
/// + a leaked `Box`) and roots are visited by the custom walker
/// (`pyre-interpreter::eval::walk_pyframe_roots`); this id is therefore
/// metadata-only at registration time.
/// `emit_new_pyframe_inline_self_recursive` will be the first writer.
///
/// Asserts the same id is returned by `gc.register_type(...)` so any
/// drift panics on startup.
pub const PYFRAME_GC_TYPE_ID: u32 = 37;

/// GC type ids appended after the existing runtime registration census.
/// `FrameDebugData` is stationary old-gen for a GC-owned frame; block-stack
/// nodes are ordinary nursery objects.  Keep these at the tail of
/// `pyre-jit::eval::build_gc` so older type ids never shift.
pub const FRAME_DEBUG_DATA_GC_TYPE_ID: u32 = 103;
pub const FRAME_BLOCK_GC_TYPE_ID: u32 = 104;

/// GC header size in bytes — single source of truth is
/// [`majit_gc::header::GcHeader::SIZE`]. Every `FixedObjectArray` and
/// `PyFrame` allocation prepends this many zero bytes so RPython-style
/// write barriers (`*(obj + wb_byteofs) & mask`) read a valid header with
/// `TRACK_YOUNG_PTRS=0` and skip the slow path. Scalar `value`
/// allocations route through [`majit_gc::header::alloc_with_gc_header`].
pub const GC_HEADER_SIZE: usize = majit_gc::header::GcHeader::SIZE;

/// Ownership selected by the caller that decides a frame's lifetime.
/// `FrameBox::new` call frames use `OldGenGc`; tracer-private snapshots use
/// `StdAlloc` so their locals remain valid until deterministic `Drop`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameLocalsArrayAllocation {
    OldGenGc,
    StdAlloc,
}

/// Allocation size (in bytes, including the GC header) for a
/// `FixedObjectArray` of the given length.
#[inline]
fn fixed_array_alloc_size(len: usize) -> usize {
    GC_HEADER_SIZE
        + pyre_object::FIXED_ARRAY_ITEMS_OFFSET
        + len * std::mem::size_of::<pyre_object::PyObjectRef>()
}

#[inline]
fn fixed_array_layout(len: usize) -> std::alloc::Layout {
    // Track the header alignment (the block leads with a GcHeader the barrier
    // reads at `obj - SIZE`), not a hardcoded 8.
    let align = std::mem::align_of::<majit_gc::header::GcHeader>()
        .max(std::mem::align_of::<FixedObjectArray>());
    std::alloc::Layout::from_size_align(fixed_array_alloc_size(len), align).unwrap()
}

/// Allocate a fixed-length GcArray-shaped `FixedObjectArray` with all
/// slots initialised to `fill`. The allocation is prefixed with a
/// zeroed GC header ([`GC_HEADER_SIZE`] bytes) so the write barrier
/// fast-path sees `TRACK_YOUNG_PTRS=0`.
///
/// The returned pointer points at the length prefix; items follow
/// immediately in memory at `FIXED_ARRAY_ITEMS_OFFSET` — matching
/// RPython `Ptr(GcArray(PyObjectRef))`.
pub unsafe fn alloc_fixed_array_with_header(
    len: usize,
    fill: pyre_object::PyObjectRef,
) -> *mut FixedObjectArray {
    unsafe {
        let layout = fixed_array_layout(len);
        let raw = std::alloc::alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let arr = raw.add(GC_HEADER_SIZE) as *mut FixedObjectArray;
        (*arr).len = len;
        let items = (arr as *mut u8).add(pyre_object::FIXED_ARRAY_ITEMS_OFFSET)
            as *mut pyre_object::PyObjectRef;
        for i in 0..len {
            items.add(i).write(fill);
        }
        arr
    }
}

/// Allocate a frame locals array in the lifetime regime selected by its owner.
/// The old-gen form is the type-9 GcArray layout `[GcHeader | len | items]`.
unsafe fn alloc_frame_locals_array(
    len: usize,
    fill: pyre_object::PyObjectRef,
    allocation: FrameLocalsArrayAllocation,
) -> *mut FixedObjectArray {
    if allocation == FrameLocalsArrayAllocation::OldGenGc {
        let payload = pyre_object::FIXED_ARRAY_ITEMS_OFFSET
            + len * std::mem::size_of::<pyre_object::PyObjectRef>();
        let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
            pyre_object::PY_OBJECT_ARRAY_GC_TYPE_ID,
            payload,
        );
        if !raw.is_null() {
            let arr = raw as *mut FixedObjectArray;
            unsafe {
                (*arr).len = len;
                let items = (*arr).items_mut_ptr();
                for i in 0..len {
                    items.add(i).write(fill);
                }
            }
            return arr;
        }
    }
    unsafe { alloc_fixed_array_with_header(len, fill) }
}

#[inline]
fn remember_frame_locals_array(array: *mut FixedObjectArray) {
    if pyre_object::gc_hook::try_gc_owns_object(array as *mut u8) {
        pyre_object::gc_hook::try_gc_write_barrier(array as *mut u8);
    }
}

/// Allocate a `FixedObjectArray` pre-populated from `values`. The
/// resulting array has `values.len()` slots; allocation layout matches
/// [`alloc_fixed_array_with_header`].
pub unsafe fn alloc_fixed_array_from_vec(
    values: Vec<pyre_object::PyObjectRef>,
) -> *mut FixedObjectArray {
    unsafe {
        let len = values.len();
        let layout = fixed_array_layout(len);
        let raw = std::alloc::alloc_zeroed(layout);
        if raw.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        let arr = raw.add(GC_HEADER_SIZE) as *mut FixedObjectArray;
        (*arr).len = len;
        let items = (arr as *mut u8).add(pyre_object::FIXED_ARRAY_ITEMS_OFFSET)
            as *mut pyre_object::PyObjectRef;
        for (i, v) in values.into_iter().enumerate() {
            items.add(i).write(v);
        }
        arr
    }
}

/// Deallocate a `FixedObjectArray` allocated with
/// [`alloc_fixed_array_with_header`] or [`alloc_fixed_array_from_vec`].
pub unsafe fn dealloc_array_with_gc_header(ptr: *mut FixedObjectArray) {
    if ptr.is_null() {
        return;
    }
    debug_assert!(!pyre_object::gc_hook::try_gc_owns_object(ptr as *mut u8));
    unsafe {
        let len = (*ptr).len;
        let raw = (ptr as *mut u8).sub(GC_HEADER_SIZE);
        std::alloc::dealloc(raw, fixed_array_layout(len));
    }
}

/// Heap layout for a managed `PyFrame`: a leading GC header followed by
/// the frame body. The header occupies [`GC_HEADER_SIZE`] bytes and stays
/// zeroed (`type_id = 0`, `flags = 0`) so the write-barrier fast-path reads
/// `TRACK_YOUNG_PTRS = 0` at `frame - GC_HEADER_SIZE` and skips the slow
/// path. Mirrors `FixedObjectArray`'s header prefix and the callee-frame
/// allocation in `pyre-jit::call_jit`, realising the documented invariant
/// that every heap `PyFrame` is a header-bearing GC object.
#[repr(C)]
struct GcFramePrefix {
    gc_header: u64,
    frame: PyFrame,
}

// The frame must sit exactly GC_HEADER_SIZE bytes into the prefix: the write
// barrier reads the header at `frame - GC_HEADER_SIZE`, and FrameBox::drop
// reconstructs the owning Box from that same address. A future PyFrame field
// with alignment greater than the header word would make `#[repr(C)]` pad the
// frame to a larger offset and silently break both, so pin the offset here.
const _: () = assert!(std::mem::offset_of!(GcFramePrefix, frame) == GC_HEADER_SIZE);

/// Owning handle to a heap [`PyFrame`] allocated with a leading GC header.
/// Dereferences to the inner `PyFrame`; the header lives at
/// `self.ptr - GC_HEADER_SIZE`. Replaces the bare `Box<PyFrame>` so the
/// frame the JIT treats as the virtualizable is a real GC object with a
/// valid header at `frame - GC_HEADER_SIZE`.
pub struct FrameBox {
    ptr: *mut PyFrame,
}

impl FrameBox {
    /// Move `frame` onto the heap behind a GC header.
    ///
    /// `pyframe.py class PyFrame(W_Root)` — an executing frame is a normal
    /// GC object whose lifetime is its reachability.  When the GC hook is
    /// installed this allocates a non-moving old-gen `PYFRAME_GC_TYPE_ID`
    /// block (the same `try_gc_alloc_stable` path every `W_*` uses, e.g.
    /// `function.rs:373`); the collector reclaims the frame and its
    /// GC-managed locals, debug data, and block stack once no root
    /// (`walk_pyframe_roots` over the `CURRENT_FRAME` / `f_backref` chain)
    /// reaches it, so `Drop` performs no manual free
    /// (`executioncontext.py:91-107 leave` frees nothing either).
    ///
    /// Before the hook is wired (bootstrap, tests) `try_gc_alloc_stable`
    /// returns `None`; fall back to the `std::alloc` `GcFramePrefix` box,
    /// which `Drop` frees manually.  The two regimes share one memory
    /// layout — an 8-byte GC header immediately before the frame body — so
    /// every reader (`frame - GC_HEADER_SIZE` write-barrier header,
    /// `pyframe_object_custom_trace`) is regime-independent; `Drop`
    /// distinguishes them with `try_gc_owns_object`.
    pub fn new(frame: PyFrame) -> Self {
        let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
            PYFRAME_GC_TYPE_ID,
            std::mem::size_of::<PyFrame>(),
        );
        if !raw.is_null() {
            debug_assert!(pyre_object::gc_hook::try_gc_owns_object(
                frame.locals_cells_stack_w as *mut u8
            ));
            pyre_object::gc_interp::note_alloc();
            let ptr = raw as *mut PyFrame;
            unsafe {
                std::ptr::write(ptr, frame);
            }
            // The old-gen frame may hold pointers to freshly nursery-born
            // argument / locals objects; remember it for the next minor
            // tracer, exactly as `generator.rs:72` does for a stable
            // generator wrapping young frame contents.
            pyre_object::gc_hook::try_gc_write_barrier(raw);
            return FrameBox { ptr };
        }
        debug_assert!(!pyre_object::gc_hook::try_gc_owns_object(
            frame.locals_cells_stack_w as *mut u8
        ));
        FrameBox::new_boxed(frame)
    }

    /// Allocate a frame that is NOT GC-managed even when the GC hook is
    /// installed — a plain `std::alloc` `GcFramePrefix` box reclaimed by
    /// `Drop`.  Used for tracer snapshots (`snapshot_for_tracing`), which
    /// a tracer holds off the `CURRENT_FRAME` chain across an entire
    /// `trace_bytecode` walk: a major cycle can complete mid-walk, and no
    /// root reaches the snapshot, so GC lifetime would reclaim it while the
    /// tracer still reads it.  A deterministic scope-end free is correct
    /// for these transient, tracer-private copies.
    pub fn new_boxed(frame: PyFrame) -> Self {
        let raw = Box::into_raw(Box::new(GcFramePrefix {
            gc_header: 0,
            frame,
        }));
        let ptr = unsafe { std::ptr::addr_of_mut!((*raw).frame) };
        FrameBox { ptr }
    }

    /// Relinquish ownership, returning the inner-frame pointer. The header
    /// remains at `ptr - GC_HEADER_SIZE`; reclaim via [`FrameBox::from_raw`].
    pub fn into_raw(self) -> *mut PyFrame {
        let ptr = self.ptr;
        std::mem::forget(self);
        ptr
    }

    /// Reconstruct ownership from a pointer previously produced by
    /// [`FrameBox::into_raw`].
    ///
    /// # Safety
    /// `ptr` must originate from [`FrameBox::into_raw`] / [`FrameBox::new`]
    /// and must not have been freed.
    pub unsafe fn from_raw(ptr: *mut PyFrame) -> Self {
        FrameBox { ptr }
    }

    /// Raw pointer to the inner frame (header at `ptr - GC_HEADER_SIZE`).
    pub fn as_mut_ptr(&mut self) -> *mut PyFrame {
        self.ptr
    }

    /// pyframe.py:259 initialize_as_generator — wrap this frame in a generator
    /// object that takes ownership of it. PyPy does `GeneratorIterator(self)`:
    /// the generator references the same frame, no copy. FrameBox already holds
    /// the heap, header-bearing frame, so ownership transfers straight through
    /// `into_raw` without the snapshot copy `PyFrame::initialize_as_generator`
    /// needs for the borrowed-`&mut self` case.
    pub fn into_generator(mut self) -> crate::PyResult {
        self.fix_array_ptrs();
        let register_final = !self.code().exceptiontable.is_empty();
        // A suspended generator frame is off the call chain — `f_back` is
        // None until a resume re-links it (`executioncontext.py enter`
        // rebinds `f_backref = topframeref`; pyre does the same at
        // `execute_frame`).  Null it now so the generator's custom trace,
        // which greys the frame block and recurses `f_backref`, never greys
        // the (possibly already-freed) caller frame captured at suspend.
        self.f_backref = std::ptr::null_mut();
        let frame_ptr = self.into_raw();
        // `w_generator_new` allocates and may trigger a collection. Until the
        // generator owns `frame_ptr` (and its custom trace greys the frame
        // block), the frame's locals/args live only in its
        // `locals_cells_stack_w` — the caller has already dropped them from
        // its own stack — so root that slot across the allocation. The frame
        // block is non-moving (old-gen when GC-managed, `std::alloc`
        // otherwise), so only the locals array needs protecting.
        let _root = FrameLocalsRoot::new(frame_ptr);
        let generator = pyre_object::generator::w_generator_new(frame_ptr as *mut u8);
        unsafe {
            (*frame_ptr).f_generator_nowref = generator;
        }
        if register_final {
            crate::executioncontext::register_generator_finalizer(generator);
        }
        Ok(generator)
    }
}

impl std::ops::Deref for FrameBox {
    type Target = PyFrame;
    #[inline]
    fn deref(&self) -> &PyFrame {
        unsafe { &*self.ptr }
    }
}

impl std::ops::DerefMut for FrameBox {
    #[inline]
    fn deref_mut(&mut self) -> &mut PyFrame {
        unsafe { &mut *self.ptr }
    }
}

impl Drop for FrameBox {
    fn drop(&mut self) {
        // GC-managed (old-gen) frames are reclaimed by a major mark-sweep
        // when no root reaches them (`pyframe.py class PyFrame(W_Root)`;
        // `executioncontext.py:91-107 leave` frees nothing).  Their
        // The collector reclaims their GC-managed locals array, debug data,
        // and block chain with the frame, so no manual cleanup runs here.
        // Only a `std::alloc` snapshot / bootstrap fallback box is freed
        // manually — reconstruct and drop it, which runs `PyFrame::drop`.
        if pyre_object::gc_hook::try_gc_owns_object(self.ptr as *mut u8) {
            return;
        }
        unsafe {
            let prefix = (self.ptr as *mut u8).sub(GC_HEADER_SIZE) as *mut GcFramePrefix;
            drop(Box::from_raw(prefix));
        }
    }
}

/// RAII guard registering a frame's `locals_cells_stack_w` slot as a GC root
/// for the duration of an allocating frame-setup step. Mirrors the callee-locals
/// root the eval path holds in `call.rs`: during setup the freshly-installed
/// locals/cells live only in that array, so an intervening collection would
/// drop or mis-forward them unless the slot is rooted.
pub struct FrameLocalsRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl FrameLocalsRoot {
    pub fn new(frame_ptr: *mut PyFrame) -> Self {
        let slot =
            unsafe { std::ptr::addr_of_mut!((*frame_ptr).locals_cells_stack_w) as *mut *mut u8 };
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for FrameLocalsRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

#[inline]
fn remember_frame_debug_data(debugdata: *mut FrameDebugData) {
    if pyre_object::gc_hook::try_gc_owns_object(debugdata as *mut u8) {
        pyre_object::gc_hook::try_gc_write_barrier(debugdata as *mut u8);
    }
}

unsafe fn clone_debugdata_ptr(
    ptr: *mut FrameDebugData,
    allocation: FrameLocalsArrayAllocation,
) -> *mut FrameDebugData {
    unsafe {
        if ptr.is_null() {
            std::ptr::null_mut()
        } else if allocation == FrameLocalsArrayAllocation::OldGenGc {
            let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
                FRAME_DEBUG_DATA_GC_TYPE_ID,
                std::mem::size_of::<FrameDebugData>(),
            );
            if !raw.is_null() {
                std::ptr::write(raw as *mut FrameDebugData, (*ptr).clone());
                // The clone may carry young locals / trace callback refs.
                remember_frame_debug_data(raw as *mut FrameDebugData);
                return raw as *mut FrameDebugData;
            }
            pyre_object::lltype::malloc_raw((*ptr).clone())
        } else {
            pyre_object::lltype::malloc_raw((*ptr).clone())
        }
    }
}

unsafe fn clear_debugdata_ptr(ptr: &mut *mut FrameDebugData) {
    unsafe {
        if !(*ptr).is_null() {
            if !pyre_object::gc_hook::try_gc_owns_object(*ptr as *mut u8) {
                drop(Box::from_raw(*ptr));
            }
            *ptr = std::ptr::null_mut();
        }
    }
}

struct FrameBlockRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl FrameBlockRoot {
    unsafe fn new(block: &mut *mut FrameBlock) -> Self {
        let slot = block as *mut *mut FrameBlock as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for FrameBlockRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

unsafe fn alloc_frame_block(
    block: FrameBlock,
    allocation: FrameLocalsArrayAllocation,
) -> *mut FrameBlock {
    if allocation == FrameLocalsArrayAllocation::OldGenGc {
        if let Some(raw) = pyre_object::gc_hook::try_gc_alloc(
            FRAME_BLOCK_GC_TYPE_ID,
            std::mem::size_of::<FrameBlock>(),
        )
        .filter(|raw| !raw.is_null())
        {
            unsafe { std::ptr::write(raw as *mut FrameBlock, block) };
            return raw as *mut FrameBlock;
        }
    }
    pyre_object::lltype::malloc_raw(block)
}

#[inline]
fn remember_frame_block_node(node: *mut FrameBlock) {
    if pyre_object::gc_hook::try_gc_owns_object(node as *mut u8) {
        pyre_object::gc_hook::try_gc_write_barrier(node as *mut u8);
    }
}

unsafe fn clone_block_chain(
    ptr: *mut FrameBlock,
    allocation: FrameLocalsArrayAllocation,
) -> *mut FrameBlock {
    // `previous` is assigned only while a node is constructed and points to
    // a strictly older node. Rebuild oldest-to-newest, but barrier each store:
    // a nursery-full allocation fallback can make a node old while its
    // predecessor is still young.
    let mut source = Vec::new();
    let mut current = ptr;
    while !current.is_null() {
        unsafe {
            source.push(FrameBlock {
                handlerposition: (*current).handlerposition,
                valuestackdepth: (*current).valuestackdepth,
                previous: std::ptr::null_mut(),
            });
            current = (*current).previous;
        }
    }

    let mut cloned = std::ptr::null_mut();
    for mut block in source.into_iter().rev() {
        let _root = unsafe { FrameBlockRoot::new(&mut cloned) };
        block.previous = std::ptr::null_mut();
        let node = unsafe { alloc_frame_block(block, allocation) };
        unsafe { (*node).previous = cloned };
        remember_frame_block_node(node);
        cloned = node;
    }
    cloned
}

unsafe fn clear_block_chain(ptr: &mut *mut FrameBlock) {
    unsafe {
        if !(*ptr).is_null() && pyre_object::gc_hook::try_gc_owns_object(*ptr as *mut u8) {
            // A GC chain is uniformly managed by its owning GC frame.  The
            // collector traces `previous` and reclaims the nodes itself.
            *ptr = std::ptr::null_mut();
            return;
        }
        let mut current = *ptr;
        while !current.is_null() {
            debug_assert!(!pyre_object::gc_hook::try_gc_owns_object(
                current as *mut u8
            ));
            let block = Box::from_raw(current);
            current = block.previous;
        }
        *ptr = std::ptr::null_mut();
    }
}

impl Drop for PyFrame {
    fn drop(&mut self) {
        // Reached only for a `std::alloc`-backed frame (the `FrameBox`
        // fallback box, or a bare stack `PyFrame`): it owns `std::alloc`
        // resources, so free them here.  GC-managed frames never run
        // `PyFrame::drop`; the collector reclaims their frame-owned resources.
        unsafe { self.free_owned_contents(true) };
    }
}

impl PyFrame {
    #[inline]
    fn aux_allocation(&self) -> FrameLocalsArrayAllocation {
        if pyre_object::gc_hook::try_gc_owns_object(self as *const Self as *mut u8) {
            FrameLocalsArrayAllocation::OldGenGc
        } else {
            FrameLocalsArrayAllocation::StdAlloc
        }
    }

    /// Free the `std::alloc` resources owned by a snapshot, fallback, or bare
    /// stack frame: its `locals_cells_stack_w` array, `FrameDebugData` box,
    /// and `FrameBlock` chain.  This is reached only through `Drop for
    /// PyFrame`; GC-managed frames and all of their corresponding resources
    /// are reclaimed by the collector.
    ///
    /// `free_locals_array` gates freeing `locals_cells_stack_w`: it is a
    /// `std::alloc` block for this Drop-only regime (free it), but can remain
    /// false if a future regime-mixup supplies a GC-managed array.  The
    /// `try_gc_owns_object` checks in this cleanup path are retained as a
    /// guard against freeing collector-owned storage.
    ///
    /// # Safety
    /// Runs at most once per frame — the pointers are nulled as they are
    /// freed so a second call is a no-op.
    pub unsafe fn free_owned_contents(&mut self, free_locals_array: bool) {
        if free_locals_array && !self.locals_cells_stack_w.is_null() {
            unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
            self.locals_cells_stack_w = std::ptr::null_mut();
        }
        unsafe {
            clear_debugdata_ptr(&mut self.debugdata);
            clear_block_chain(&mut self.lastblock);
        }
    }

    /// Access locals_cells_stack_w (deref the pointer).
    #[inline]
    pub fn locals_w(&self) -> &FixedObjectArray {
        unsafe { &*self.locals_cells_stack_w }
    }

    /// Mutably access locals_cells_stack_w.
    #[inline]
    pub fn locals_w_mut(&mut self) -> &mut FixedObjectArray {
        unsafe { &mut *self.locals_cells_stack_w }
    }

    /// Restore the per-trace mutable frame state — instruction pointer, value
    /// stack depth, and `locals_cells_stack_w` contents — from a snapshot
    /// taken before a bridge trace.  `trace_and_compile_from_bridge` uses this
    /// to undo any concrete-execution mutation the full-body walker applied to
    /// the live frame during the walk, so a `BridgeCompiled` resume re-enters
    /// at the exact guard state instead of mid-body / past a stepped loop
    /// counter.  `src` is a `snapshot_for_tracing` of this same frame, so the
    /// arrays share length; the `min` is defensive.
    ///
    /// These three are exactly the mutable virtualizable fields the walk can
    /// touch (may-force residual calls advance `last_instr` and the value
    /// stack).  The remaining `_virtualizable_` fields — `pycode`, `w_globals`,
    /// `debugdata` — are deliberately NOT restored: `pycode`/`w_globals` are
    /// frame-invariant and `debugdata` is debug-only; none are written on the
    /// live frame during tracing (the walk writes only the symbolic `PyreSym`
    /// shadow), so restoring them would be dead.
    pub fn restore_resume_state_from(&mut self, src: &PyFrame) {
        self.last_instr = src.last_instr;
        self.valuestackdepth = src.valuestackdepth;
        let src_vals = src.locals_w().as_slice().to_vec();
        let dst = self.locals_w_mut();
        let n = src_vals.len().min(dst.as_slice().len());
        for (i, &v) in src_vals.iter().take(n).enumerate() {
            dst[i] = v;
        }
    }
}

/// Extract raw CodeObject from frame's PyCode.
///
/// PyPy: `frame.pycode` gives `PyCode` which IS the code object.
/// pyre: PyCode wraps a raw CodeObject — this extracts it.
///
/// `@jit.elidable` (`rlib/jit.py:13`): deterministic, no allocation,
/// no raise — pure pointer cast through `w_code_get_ptr`.
/// Mapped to `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`) so the
/// metainterp can execute it at trace time and the walker can read
/// the concrete result back into the Ref bank.
#[majit_macros::elidable_cannot_raise]
#[inline]
pub unsafe fn pyframe_get_pycode(frame: &PyFrame) -> *const CodeObject {
    unsafe { crate::w_code_get_ptr(frame.pycode as pyre_object::PyObjectRef) as *const CodeObject }
}

#[repr(C)]
#[derive(Clone)]
pub struct FrameDebugData {
    /// pyframe.py:44 — the frame's locals mapping (`self.w_locals`).
    /// At module scope it is the `w_globals` dict; in a class body it is
    /// the class namespace; for a function it is the dict lazily
    /// materialised by `fast2locals`; `exec(src, g, mapping)` binds an
    /// arbitrary `__getitem__` mapping here.  `STORE/LOAD/DELETE_NAME`
    /// route through `space.setitem/getitem/delitem(w_locals, ...)`.
    pub w_locals: PyObjectRef,
    /// pyframe.py:37
    pub w_f_trace: PyObjectRef,
    /// pyframe.py:40
    pub is_being_profiled: bool,
    /// pyframe.py:41
    pub is_in_line_tracing: bool,
    /// pyframe.py:42
    pub f_trace_lines: bool,
    /// pyframe.py:43
    pub f_trace_opcodes: bool,
    /// pyframe.py:38
    pub instr_prev_plus_one: isize,
    /// pyframe.py:39
    pub f_lineno: isize,
    /// pyframe.py:45
    pub hidden_operationerr: PyObjectRef,
}

impl FrameDebugData {
    // `_pycode` keeps the constructor shape of `pyframe.py:48
    // FrameDebugData.__init__(self, pycode, init_lineno)`.  The frame's
    // globals object now lives in `PyFrame.w_globals`, so debugdata no
    // longer snapshots `pycode.w_globals`.
    pub fn new(_pycode: *const (), init_lineno: isize) -> Self {
        Self {
            w_locals: pyre_object::PY_NULL,
            w_f_trace: pyre_object::PY_NULL,
            is_being_profiled: false,
            is_in_line_tracing: false,
            f_trace_lines: true,
            f_trace_opcodes: false,
            instr_prev_plus_one: 0,
            f_lineno: init_lineno,
            hidden_operationerr: pyre_object::PY_NULL,
        }
    }
}

impl Default for FrameDebugData {
    fn default() -> Self {
        Self::new(std::ptr::null(), -1)
    }
}

/// pyopcode.py:1875-1897 FrameBlock — linked list node for the block stack.
/// `previous` forms a singly-linked list; `lastblock` in PyFrame is the head.
/// It is assigned only during construction and targets a strictly older node,
/// so an old-gen node never needs a write barrier for it.
#[derive(Debug, Clone, Copy)]
pub struct FrameBlock {
    /// pyopcode.py:1883
    pub valuestackdepth: usize,
    /// pyopcode.py:1882
    pub handlerposition: usize,
    /// pyopcode.py:1884 — pointer to the previous FrameBlock (null = None).
    pub previous: *mut FrameBlock,
}

impl FrameBlock {
    /// pyopcode.py:1886-1887
    #[inline]
    pub fn cleanupstack(&self, frame: &mut PyFrame) {
        frame.dropvaluesuntil(self.valuestackdepth);
    }
}

#[inline]
pub fn get_block_class(opname: &str) -> &'static str {
    match opname {
        "SETUP_LOOP" | "SETUP_EXCEPT" | "SETUP_FINALLY" | "SETUP_WITH" => "FrameBlock",
        _ => "FrameBlock",
    }
}

#[inline]
pub fn unpickle_block(_space: PyObjectRef, w_tup: PyObjectRef) -> FrameBlock {
    let _ = _space;
    let handlerposition = unsafe {
        w_tuple_getitem(w_tup, 0).and_then(|v| {
            if is_int(v) {
                Some(w_int_get_value(v) as usize)
            } else {
                None
            }
        })
    }
    .unwrap_or(0);
    let valuestackdepth = unsafe {
        w_tuple_getitem(w_tup, 2).and_then(|v| {
            if is_int(v) {
                Some(w_int_get_value(v) as usize)
            } else {
                None
            }
        })
    }
    .unwrap_or(0);
    FrameBlock {
        handlerposition,
        valuestackdepth,
        previous: std::ptr::null_mut(),
    }
}

// ── Virtualizable field offsets ───────────────────────────────────────
//
// These constants tell the JIT where each virtualizable field lives
// inside a PyFrame, so it can read/write them via raw pointer arithmetic.
// Equivalent to PyPy's `_virtualizable_` descriptor on pyframe.py.

/// Byte offset of `pycode` in `PyFrame`.
pub const PYFRAME_PYCODE_OFFSET: usize = std::mem::offset_of!(PyFrame, pycode);

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrame, vable_token);

/// Byte offset of `last_instr` in `PyFrame`.
pub const PYFRAME_LAST_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrame, last_instr);

/// Byte offset of `valuestackdepth` in `PyFrame`.
pub const PYFRAME_VALUESTACKDEPTH_OFFSET: usize = std::mem::offset_of!(PyFrame, valuestackdepth);

/// Byte offset of `locals_cells_stack_w` in `PyFrame`.
pub const PYFRAME_LOCALS_CELLS_STACK_OFFSET: usize =
    std::mem::offset_of!(PyFrame, locals_cells_stack_w);

/// Byte offset of `debugdata` in `PyFrame`.
pub const PYFRAME_DEBUGDATA_OFFSET: usize = std::mem::offset_of!(PyFrame, debugdata);

/// Byte offset of `lastblock` in `PyFrame`.
pub const PYFRAME_LASTBLOCK_OFFSET: usize = std::mem::offset_of!(PyFrame, lastblock);

/// Byte offset of `f_generator_nowref` in `PyFrame`.
/// `PyObjectRef` slot — points into the GC heap (possibly nursery).
pub const PYFRAME_F_GENERATOR_NOWREF_OFFSET: usize =
    std::mem::offset_of!(PyFrame, f_generator_nowref);

/// Byte offset of `w_yielding_from` in `PyFrame`.
/// `PyObjectRef` slot — points into the GC heap (possibly nursery).
pub const PYFRAME_W_YIELDING_FROM_OFFSET: usize = std::mem::offset_of!(PyFrame, w_yielding_from);

/// Byte offset of `f_backref` in `PyFrame`.
/// `*mut PyFrame` — once `NewWithVtable(PyFrame)` lands in trace IR,
/// chained recursive callees may have a nursery-allocated parent frame
/// reachable through this pointer.
pub const PYFRAME_F_BACKREF_OFFSET: usize = std::mem::offset_of!(PyFrame, f_backref);

/// Byte offset of `w_builtin` in `PyFrame` (the picked builtin Module).
/// Read by the descr GC walker so a collection survives across the
/// guard exit / re-entry edge.
pub const PYFRAME_W_BUILTIN_OFFSET: usize = std::mem::offset_of!(PyFrame, w_builtin);

/// Byte offset of `w_globals` in `PyFrame` — the canonical
/// W_DictObject paired with the storage in `w_globals`.  Registered
/// as a GC-traceable slot so a minor collection forwards the pointer
/// when the dict survives.  The slot is lazy: `PY_NULL` until
/// `get_w_globals` resolves it.
pub const PYFRAME_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(PyFrame, w_globals);

// Backward-compat aliases used by JIT code.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = PYFRAME_VALUESTACKDEPTH_OFFSET;
pub const PYFRAME_LOCALS_OFFSET: usize = PYFRAME_LOCALS_CELLS_STACK_OFFSET;

/// pytraceback.py offset2lineno(c, stopat) — convert instruction index to line number.
/// Matches RPython: negative `stopat` means "frame not yet started", returns
/// first-line.
#[inline]
pub fn offset2lineno(code: &CodeObject, stopat: isize) -> usize {
    let lineno = code.first_line_number.map(|n| n.get()).unwrap_or(1);
    if stopat < 0 {
        return lineno;
    }
    code.locations
        .get(stopat as usize)
        .map(|(start, _)| start.line.get())
        .unwrap_or(lineno)
}

// ─────────────────────────────────────────────────────────────────────
// `f_lineno` jump validation — a port of CPython 3.14's `mark_stacks`
// (`Objects/frameobject.c`).
//
// PyPy's `fset_f_lineno` (`pyframe.py`) validates a debugger line-jump
// against the 3.11-era block model: it scans for `SETUP_LOOP` /
// `SETUP_FINALLY` / `END_FINALLY` / `POP_BLOCK` and walks `co_lnotab`.
// pyre runs 3.14-structural bytecode where those block-setup opcodes are
// pseudo-ops the compiler lowers into `co_exceptiontable`; there is no
// `co_lnotab`.  A line-by-line PyPy port is therefore impossible — the
// opcodes it inspects don't exist here.  The behaviour-correct 3.14
// equivalent is `mark_stacks`: a fixpoint reachability analysis that
// abstracts each operand-stack slot to a `Kind`, so a jump is admitted
// only when the source and target abstract stacks are compatible.
// ─────────────────────────────────────────────────────────────────────

/// `frameobject.c` `Kind` — the abstract contents of one operand-stack
/// slot.  Packed 3 bits per slot into an `i64` (`BITS_PER_BLOCK = 3`),
/// so the abstract stack of up to `MAX_STACK_ENTRIES = 21` slots is a
/// single integer that can be compared for equality across paths.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
enum StackKind {
    Iterator = 1,
    Except = 2,
    Object = 3,
    Null = 4,
    Lasti = 5,
}

const MARK_BITS_PER_BLOCK: i64 = 3;
const MARK_MASK: i64 = (1 << MARK_BITS_PER_BLOCK) - 1;
/// `1 << (63 - BITS_PER_BLOCK)` — a stack whose unsigned value is at or
/// above this would lose its top slot when shifted left.  The negative
/// sentinels (`OVERFLOWED` / `UNINITIALIZED`) reinterpret as huge
/// unsigned values, so this test also propagates them.
const MARK_WILL_OVERFLOW: u64 = 1u64 << (63 - MARK_BITS_PER_BLOCK);

/// `frameobject.c` sentinels for a per-instruction abstract stack.
const MARK_UNINITIALIZED: i64 = -2;
const MARK_OVERFLOWED: i64 = -1;
const MARK_EMPTY_STACK: i64 = 0;

#[inline]
fn mark_push_value(stack: i64, kind: StackKind) -> i64 {
    mark_push_kind_bits(stack, kind as i64)
}

/// `push_value` with the kind supplied as raw bits — used by `COPY`,
/// which re-pushes a slot peeked from the stack (whose kind is not
/// statically one of the [`StackKind`] variants).
#[inline]
fn mark_push_kind_bits(stack: i64, kind_bits: i64) -> i64 {
    if (stack as u64) >= MARK_WILL_OVERFLOW {
        MARK_OVERFLOWED
    } else {
        (stack << MARK_BITS_PER_BLOCK) | kind_bits
    }
}

#[inline]
fn mark_pop_value(stack: i64) -> i64 {
    // Arithmetic right shift preserves the OVERFLOWED / UNINITIALIZED
    // sentinels (both negative).
    stack >> MARK_BITS_PER_BLOCK
}

#[inline]
fn mark_top_of_stack(stack: i64) -> i64 {
    stack & MARK_MASK
}

#[inline]
fn mark_peek(stack: i64, n: i64) -> i64 {
    (stack >> (MARK_BITS_PER_BLOCK * (n - 1))) & MARK_MASK
}

#[inline]
fn mark_stack_swap(stack: i64, n: i64) -> i64 {
    let top = mark_top_of_stack(stack);
    let nth = mark_peek(stack, n);
    let shift = MARK_BITS_PER_BLOCK * (n - 1);
    let stack = stack & !(MARK_MASK << shift) | (top << shift);
    (stack & !MARK_MASK) | nth
}

/// `frameobject.c pop_to_level` — pop the abstract stack down to `level`
/// slots.
#[inline]
fn mark_pop_to_level(mut stack: i64, level: i32) -> i64 {
    let mut depth = 0i64;
    let mut s = stack;
    while s > MARK_EMPTY_STACK {
        s = mark_pop_value(s);
        depth += 1;
    }
    while depth > level as i64 {
        stack = mark_pop_value(stack);
        depth -= 1;
    }
    stack
}

/// True when a jump from `from` kind to `to` kind is admissible
/// (`frameobject.c compatible_kind`).
#[inline]
fn mark_compatible_kind(from: i64, to: i64) -> bool {
    if to == 0 {
        return false;
    }
    if to == StackKind::Object as i64 {
        return from != StackKind::Null as i64;
    }
    if to == StackKind::Null as i64 {
        return true;
    }
    from == to
}

/// `frameobject.c compatible_stack` — pop `from_stack` to the target
/// depth, then compare each slot's kind.
fn mark_compatible_stack(mut from_stack: i64, target_stack: i64) -> bool {
    if from_stack < 0 || target_stack < 0 {
        return false;
    }
    let mut to = target_stack;
    // Depth of each packed stack.
    let depth = |mut s: i64| -> i64 {
        let mut d = 0;
        while s > MARK_EMPTY_STACK {
            s = mark_pop_value(s);
            d += 1;
        }
        d
    };
    while depth(from_stack) > depth(to) {
        from_stack = mark_pop_value(from_stack);
    }
    while to > MARK_EMPTY_STACK {
        if from_stack <= MARK_EMPTY_STACK {
            return false;
        }
        if !mark_compatible_kind(mark_top_of_stack(from_stack), mark_top_of_stack(to)) {
            return false;
        }
        from_stack = mark_pop_value(from_stack);
        to = mark_pop_value(to);
    }
    from_stack == MARK_EMPTY_STACK
}

/// Diagnostic for an incompatible target stack
/// (`frameobject.c explain_incompatible_stack`).
fn mark_explain_incompatible_stack(target_stack: i64) -> &'static str {
    if target_stack == MARK_OVERFLOWED {
        return "stack too deep to analyze";
    }
    if target_stack == MARK_UNINITIALIZED {
        return "can't jump into an exception handler, or code may be unreachable";
    }
    let top = mark_top_of_stack(target_stack);
    if top == StackKind::Except as i64 {
        "can't jump into an 'except' block as there's no exception"
    } else if top == StackKind::Lasti as i64 {
        "can't jump into a re-raising block as there's no location"
    } else if top == StackKind::Object as i64 || top == StackKind::Null as i64 {
        "incompatible stacks"
    } else if top == StackKind::Iterator as i64 {
        "can't jump into the body of a for loop"
    } else {
        "incompatible stacks"
    }
}

/// `frameobject.c marklines` — the source line that starts at each
/// instruction-unit index (`-1` where no line change begins).
fn mark_lines(code: &CodeObject, len: usize) -> Vec<i32> {
    let mut lines = vec![-1i32; len];
    let first = code.first_line_number.map(|n| n.get() as i32).unwrap_or(1);
    let mut last_line = -1i32;
    for i in 0..len {
        // `locations[i].0` is the start SourceLocation of unit `i`.
        let line = code
            .locations
            .get(i)
            .map(|(start, _)| start.line.get() as i32)
            .unwrap_or(first);
        if line != last_line && line != -1 {
            lines[i] = line;
            last_line = line;
        }
    }
    lines
}

/// `frameobject.c first_line_not_before` — the smallest line `>= line`
/// present in `lines`, or `-1`.
fn mark_first_line_not_before(lines: &[i32], line: i32) -> i32 {
    let mut result = i32::MAX;
    for &l in lines {
        if l < result && l >= line {
            result = l;
        }
    }
    if result == i32::MAX { -1 } else { result }
}

/// `frameobject.c mark_stacks` — fixpoint abstract-interpretation of the
/// operand stack.  Returns `stacks[i]` = the packed abstract stack on
/// entry to unit `i` (`UNINITIALIZED` where unreachable), length
/// `len + 1`.
fn mark_stacks(code: &CodeObject, len: usize) -> Vec<i64> {
    use crate::bytecode::Instruction;

    let mut stacks = vec![MARK_UNINITIALIZED; len + 1];
    stacks[0] = MARK_EMPTY_STACK;

    let mut todo = true;
    while todo {
        todo = false;
        // ── Scan instructions ──
        let mut i = 0usize;
        while i < len {
            let mut next_stack = stacks[i];
            let Some((opcode, op_arg)) = crate::pyopcode::decode_instruction_at(code, i) else {
                i += 1;
                continue;
            };
            // `decode_instruction_at` already accumulates EXTENDED_ARG
            // prefixes into `op_arg` at the real opcode's index.  Mirror
            // CPython's inner `while opcode == EXTENDED_ARG` loop by
            // propagating this unit's entry stack onto the following unit
            // (the next prefix or the real opcode), so an instruction with
            // an EXTENDED_ARG prefix inherits the prefix's entry stack.
            if matches!(opcode, Instruction::ExtendedArg) {
                if i + 1 < stacks.len() {
                    stacks[i + 1] = next_stack;
                }
                i += 1;
                continue;
            }
            let raw_arg: u32 = op_arg.into();
            let caches = opcode.cache_entries();
            let next_i = i + caches + 1;

            if next_stack == MARK_UNINITIALIZED {
                i = next_i;
                continue;
            }

            // Relative jump targets are measured from `next_i` (past the
            // opcode's cache slots), matching `absolutize_jump_target`.
            match opcode {
                Instruction::PopJumpIfFalse { delta }
                | Instruction::PopJumpIfTrue { delta }
                | Instruction::PopJumpIfNone { delta }
                | Instruction::PopJumpIfNotNone { delta } => {
                    let j = next_i + delta.get(op_arg).as_usize();
                    next_stack = mark_pop_value(next_stack);
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::Send { delta } => {
                    let j = next_i + delta.get(op_arg).as_usize();
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::JumpForward { delta } => {
                    let j = next_i + delta.get(op_arg).as_usize();
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                }
                Instruction::JumpBackward { delta }
                | Instruction::JumpBackwardNoInterrupt { delta } => {
                    let j = next_i.saturating_sub(delta.get(op_arg).as_usize());
                    if stacks[j] == MARK_UNINITIALIZED && j < i {
                        todo = true;
                    }
                    stacks[j] = next_stack;
                }
                Instruction::GetIter | Instruction::GetAiter => {
                    next_stack = mark_push_value(mark_pop_value(next_stack), StackKind::Iterator);
                    stacks[next_i] = next_stack;
                }
                Instruction::ForIter { delta } => {
                    // Fall-through: iterator stays on the stack, the loop
                    // variable (Object) is pushed above it.  Exhaustion
                    // branch: the iterator stays on the stack (net 0) — it
                    // is popped later by `POP_ITER`, matching the runtime
                    // stack effect (liveness.rs ForIter `(d+1, d)`).  No
                    // loop variable is pushed on that branch.
                    let target_stack = mark_push_value(next_stack, StackKind::Object);
                    stacks[next_i] = target_stack;
                    let j = next_i + delta.get(op_arg).as_usize();
                    if j <= len {
                        stacks[j] = next_stack;
                    }
                }
                Instruction::EndAsyncFor => {
                    next_stack = mark_pop_value(mark_pop_value(next_stack));
                    stacks[next_i] = next_stack;
                }
                Instruction::PushExcInfo => {
                    // Runtime (codewriter PushExcInfo): pop the new
                    // exception, push the previous exception (Except slot),
                    // then push the new exception back (Object).  Net +1,
                    // but the shape is `[.., Except, Object]` — model both
                    // slots so `f_lineno` jump validation matches the live
                    // handler stack.
                    let below = mark_pop_value(next_stack);
                    next_stack = mark_push_value(below, StackKind::Except);
                    next_stack = mark_push_value(next_stack, StackKind::Object);
                    stacks[next_i] = next_stack;
                }
                Instruction::PopExcept => {
                    next_stack = mark_pop_value(next_stack);
                    stacks[next_i] = next_stack;
                }
                Instruction::ReturnValue => {
                    // End of a path.
                }
                Instruction::RaiseVarargs { .. } | Instruction::Reraise { .. } => {
                    // End of a path.
                }
                Instruction::PushNull => {
                    next_stack = mark_push_value(next_stack, StackKind::Null);
                    stacks[next_i] = next_stack;
                }
                Instruction::LoadGlobal { .. } => {
                    next_stack = mark_push_value(next_stack, StackKind::Object);
                    if raw_arg & 1 != 0 {
                        next_stack = mark_push_value(next_stack, StackKind::Null);
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::LoadAttr { .. } => {
                    if raw_arg & 1 != 0 {
                        next_stack = mark_pop_value(next_stack);
                        next_stack = mark_push_value(next_stack, StackKind::Object);
                        next_stack = mark_push_value(next_stack, StackKind::Null);
                    }
                    stacks[next_i] = next_stack;
                }
                Instruction::Swap { .. } => {
                    next_stack = mark_stack_swap(next_stack, raw_arg as i64);
                    stacks[next_i] = next_stack;
                }
                Instruction::Copy { .. } => {
                    next_stack =
                        mark_push_kind_bits(next_stack, mark_peek(next_stack, raw_arg as i64));
                    stacks[next_i] = next_stack;
                }
                _ => {
                    // `PyCompile_OpcodeStackEffect(opcode, oparg)` — the
                    // net slot delta, with every pushed slot abstracted to
                    // Object.
                    let mut delta = opcode.stack_effect(raw_arg);
                    while delta < 0 {
                        next_stack = mark_pop_value(next_stack);
                        delta += 1;
                    }
                    while delta > 0 {
                        next_stack = mark_push_value(next_stack, StackKind::Object);
                        delta -= 1;
                    }
                    stacks[next_i] = next_stack;
                }
            }
            i = next_i;
        }

        // ── Scan the exception table ──
        for entry in crate::pycode::decode_exceptiontable(&code.exceptiontable) {
            let start_offset = (entry.start / 2) as usize;
            let handler = (entry.target / 2) as usize;
            let level = entry.depth as i32;
            let lasti = entry.lasti;
            if start_offset >= stacks.len() || handler >= stacks.len() {
                continue;
            }
            if stacks[start_offset] != MARK_UNINITIALIZED && stacks[handler] == MARK_UNINITIALIZED {
                todo = true;
                let mut target_stack = mark_pop_to_level(stacks[start_offset], level);
                if lasti {
                    target_stack = mark_push_value(target_stack, StackKind::Lasti);
                }
                target_stack = mark_push_value(target_stack, StackKind::Except);
                stacks[handler] = target_stack;
            }
        }
    }
    stacks
}

/// pyframe.py:105-106 — cell + free variable slot count.
///
/// Returns the number of *extra* slots beyond `varnames` needed for
/// cellvars and freevars. CPython 3.11+ unified the slot layout
/// (`co_localsplusnames`): a cellvar that also appears in varnames
/// (i.e. a parameter captured by an inner function) shares its
/// varname slot via `MAKE_CELL`. Only cellvars NOT in varnames need
/// a fresh slot. Without this overlap filtering, freevar indices
/// shift by the overlap count and `LOAD_DEREF`/`LOAD_FAST` for
/// freevars reads the wrong slot — see decorator-with-args test
/// `def repeat(n): def wrap(fn): def inner(): return (n, fn)`
/// where `n` resolved to `fn` because of the off-by-one slot.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn npure_cellvars(code: &CodeObject) -> usize {
    let mut count = 0;
    for c in code.cellvars.iter() {
        let cs: &str = c.as_ref();
        let mut overlaps = false;
        for v in code.varnames.iter() {
            let vs: &str = v.as_ref();
            if vs == cs {
                overlaps = true;
                break;
            }
        }
        if !overlaps {
            count += 1;
        }
    }
    count
}

#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn ncells(code: &CodeObject) -> usize {
    npure_cellvars(code) + code.freevars.len()
}

/// True when localsplus slot `idx` carries `CO_FAST_HIDDEN`, i.e. an inlined
/// comprehension's iteration variable (PEP 709). These slots are workspace
/// only and stay invisible to `locals()` / the fast↔locals sync.
#[inline]
#[majit_macros::elidable_cannot_raise]
pub fn hidden_local(code: &CodeObject, idx: usize) -> bool {
    code.localspluskinds
        .get(idx)
        .is_some_and(|&kind| kind & crate::bytecode::CO_FAST_HIDDEN != 0)
}

/// `LOAD_DEREF` unbound-variable error for the unified deref slot `idx`,
/// shared by the interpreter (`load_deref`) and the JIT residual
/// (`bh_load_deref_value_fn`).
///
/// pyopcode.py `_load_deref_unbound`: a cell variable (a captured local whose
/// deref slot shares its `varnames` index via `MAKE_CELL`, or a pure cellvar
/// in the `[nvarnames, nvarnames + npure_cellvars)` band) reports "local
/// variable '{name}' referenced before assignment"; a free variable reports
/// "free variable '{name}' referenced before assignment in enclosing scope".
/// pyre has no `UnboundLocalError`, so — like `load_local_checked_value` —
/// both use `NameError`.
///
/// `idx` follows the `npure_cellvars` slot layout: `varnames` occupy
/// `[0, nvarnames)`, pure cellvars (those not also varnames) the next
/// `npure_cellvars` slots, then freevars.
/// The name and free/cell kind of the local / cell / free variable at
/// localsplus index `idx`, following the `[varnames, pure cellvars, freevars]`
/// layout.  Used to resolve the name behind a `*_DEREF` oparg; an out-of-range
/// index yields an empty name rather than panicking.
pub fn deref_name_and_kind(code: &CodeObject, idx: usize) -> (&str, bool) {
    let nvarnames = code.varnames.len();
    if idx < nvarnames {
        return (code.varnames[idx].as_ref(), false);
    }
    let cell_slot = idx - nvarnames;
    let npure = npure_cellvars(code);
    if cell_slot < npure {
        let name = code
            .cellvars
            .iter()
            .filter(|c| {
                let cs: &str = c.as_ref();
                !code.varnames.iter().any(|v| {
                    let vs: &str = v.as_ref();
                    vs == cs
                })
            })
            .nth(cell_slot)
            .map(|c| c.as_ref())
            .unwrap_or("");
        (name, false)
    } else {
        let name = code
            .freevars
            .get(cell_slot - npure)
            .map(|f| f.as_ref())
            .unwrap_or("");
        (name, true)
    }
}

pub fn deref_unbound_error(code: &CodeObject, idx: usize) -> crate::PyError {
    let (name, is_free) = deref_name_and_kind(code, idx);
    let message = if is_free {
        format!("free variable '{name}' referenced before assignment in enclosing scope")
    } else {
        format!("local variable '{name}' referenced before assignment")
    };
    crate::PyError::name_error_with_name(message, name)
}

/// Whether calling a code object with these flags produces a suspended
/// frame object (generator / coroutine / async generator) rather than
/// running the body eagerly.
/// pyframe.py:246 `_is_generator_or_coroutine`: CO_COROUTINE |
/// CO_GENERATOR | CO_ASYNC_GENERATOR.  CO_ITERABLE_COROUTINE is a marker
/// layered on a generator function (it always co-occurs with
/// CO_GENERATOR), so it is not tested here.
#[inline]
pub fn code_flags_make_generator(flags: crate::CodeFlags) -> bool {
    flags.intersects(
        crate::CodeFlags::GENERATOR
            | crate::CodeFlags::COROUTINE
            | crate::CodeFlags::ASYNC_GENERATOR,
    )
}

impl PyFrame {
    /// pyframe.py:121 getdebug → self.debugdata
    #[inline]
    fn getdebug_data(&self) -> Option<&FrameDebugData> {
        (!self.debugdata.is_null()).then(|| unsafe { &*self.debugdata })
    }

    /// pyframe.py:124 getorcreatedebug
    #[inline]
    fn getorcreate_debug_data(&mut self, init_lineno: isize) -> &mut FrameDebugData {
        if self.debugdata.is_null() {
            let allocation = self.aux_allocation();
            let value = FrameDebugData::new(self.pycode, init_lineno);
            self.debugdata = if allocation == FrameLocalsArrayAllocation::OldGenGc {
                let raw = pyre_object::gc_hook::try_gc_alloc_stable_raw(
                    FRAME_DEBUG_DATA_GC_TYPE_ID,
                    std::mem::size_of::<FrameDebugData>(),
                );
                if !raw.is_null() {
                    unsafe { std::ptr::write(raw as *mut FrameDebugData, value) };
                    raw as *mut FrameDebugData
                } else {
                    pyre_object::lltype::malloc_raw(value)
                }
            } else {
                pyre_object::lltype::malloc_raw(value)
            };
        }
        // Callers mutate the returned payload directly. Remember the container
        // immediately before exposing it so old debugdata keeps any young refs
        // written by the caller visible to the next minor collection.
        remember_frame_debug_data(self.debugdata);
        unsafe { &mut *self.debugdata }
    }

    /// PyPy-compatible `getdebug()`.
    #[inline]
    pub fn getdebug(&self) -> Option<&FrameDebugData> {
        self.getdebug_data()
    }

    /// PyPy-compatible `getorcreatedebug()`.
    #[inline]
    pub fn getorcreatedebug(&mut self, init_lineno: isize) -> &mut FrameDebugData {
        self.getorcreate_debug_data(init_lineno)
    }

    /// PyPy-compatible alias for `code()`.
    #[inline]
    pub fn getcode(&self) -> &CodeObject {
        self.code()
    }

    /// PyPy-compatible `fget_code`.
    #[inline]
    pub fn fget_code(&self) -> &CodeObject {
        self.code()
    }

    /// pyframe.py:129-133 get_w_globals_storage — return the frame's canonical
    /// W_DictObject.  `pyframe.py:49 self.w_globals = w_globals` keeps that
    /// object as the single globals field.
    #[inline]
    pub fn get_w_globals_storage(&self) -> PyObjectRef {
        self.w_globals
    }

    /// The canonical W_DictObject for this frame's globals
    /// (`pyframe.py:49 self.w_globals = w_globals`).  Every frame
    /// constructor seeds `w_globals` eagerly, so this is a plain
    /// field read; callers wanting object identity
    /// (`function.__globals__ is frame.f_globals`, `globals() is
    /// module.__dict__`, etc.) read it directly.
    ///
    /// Returns `PY_NULL` when the frame has no globals (test stubs);
    /// callers that expect a dict should null-check before dereferencing.
    #[inline]
    pub fn get_w_globals(&self) -> PyObjectRef {
        self.w_globals
    }

    /// pyframe.py:135 get_w_f_trace
    #[inline]
    pub fn get_w_f_trace(&self) -> PyObjectRef {
        self.getdebug_data()
            .and_then(|data| (!data.w_f_trace.is_null()).then_some(data.w_f_trace))
            .unwrap_or(pyre_object::PY_NULL)
    }

    /// pyframe.py:141 get_is_being_profiled
    #[inline]
    pub fn get_is_being_profiled(&self) -> bool {
        self.getdebug_data()
            .map_or(false, |data| data.is_being_profiled)
    }

    /// `getorcreatedebug().w_locals` — the STORE_NAME / DELETE_NAME target
    /// (`pyopcode.py:855-865`): the class namespace, or the globals dict at
    /// module scope. Lazily allocates an empty dict if the frame has none.
    /// Unlike `getdictscope` it performs no `fast2locals` materialization, so
    /// it never disturbs `CO_FAST_HIDDEN` slots.
    #[inline]
    pub fn get_or_create_w_locals(&mut self) -> PyObjectRef {
        let existing = self.get_w_locals();
        if !existing.is_null() {
            return existing;
        }
        // A frame that reaches STORE_NAME / SETUP_ANNOTATIONS without a
        // bound locals mapping is degenerate (module / class / exec all
        // bind one in `initialize_frame_scopes` / `setdictscope`).
        // Allocate a fresh dict so the write still lands somewhere
        // observable instead of faulting.
        let w_locals = unsafe { pyre_object::w_dict_new() };
        self.getorcreate_debug_data(-1).w_locals = w_locals;
        w_locals
    }

    /// PyPy-compatible `__init__` hook.
    #[inline]
    pub fn __init__(
        &mut self,
        code: *const (),
        w_globals_storage: *mut DictStorage,
        outer_func: PyObjectRef,
    ) {
        let _ = outer_func;
        let allocation = self.aux_allocation();
        self.pycode = code;
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        if !self.locals_cells_stack_w.is_null()
            && !pyre_object::gc_hook::try_gc_owns_object(self.locals_cells_stack_w as *mut u8)
        {
            unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
        }
        self.locals_cells_stack_w = unsafe {
            alloc_frame_locals_array(
                (&*raw).varnames.len() + ncells(&*raw) + (&*raw).max_stackdepth as usize,
                PY_NULL,
                allocation,
            )
        };
        self.valuestackdepth = unsafe { (&*raw).varnames.len() + ncells(&*raw) };
        self.last_instr = -1;
        self.escaped = false;
        self.frame_finished_execution = false;
        self.f_generator_nowref = PY_NULL;
        self.w_yielding_from = PY_NULL;
        self.f_backref = std::ptr::null_mut();
        unsafe {
            clear_debugdata_ptr(&mut self.debugdata);
            clear_block_chain(&mut self.lastblock);
        }
        // `pyframe.py:114-115` — `self.builtin = space.builtin.pick_builtin(
        // w_globals)`.  pyre keeps the picked builtin and the canonical
        // `w_globals` W_DictObject (the `get_w_globals_storage` resolution of
        // `pyframe.py:128-132`) in the adjacent `w_builtin` / `w_globals`
        // slots, resolved eagerly here exactly as `new_minimal` does, so a
        // frame built through this hook is left in the same state as one
        // built by `createframe`.
        let w_globals = if w_globals_storage.is_null() {
            PY_NULL
        } else {
            crate::baseobjspace::dict_storage_to_dict(w_globals_storage)
        };
        self.w_builtin = crate::baseobjspace::frame_builtin_obj(w_globals, self.execution_context);
        self.w_globals = w_globals;
        // pyframe.py:103 — stamp `pycode.w_globals` (the first-globals cache
        // the LOAD_GLOBAL fast path keys on); side effect only, since the
        // gated debugdata snapshot retired in favour of `w_globals`.
        unsafe {
            crate::w_code_frame_stores_global(code as PyObjectRef, self.w_globals);
        }
        // pyframe.py:118 — final step of __init__.
        self.initialize_frame_scopes(outer_func, code).expect(
            "PyFrame::__init__: initialize_frame_scopes raised — caller should use createframe",
        );
        remember_frame_locals_array(self.locals_cells_stack_w);
    }

    /// PyPy-compatible `__repr__`.
    #[inline]
    pub fn __repr__(&self) -> String {
        format!(
            "<{}.{} executing {} at line {}>",
            module_path!(),
            std::any::type_name::<Self>()
                .rsplit("::")
                .next()
                .unwrap_or("PyFrame"),
            self.code().obj_name.as_str(),
            self.get_last_lineno()
        )
    }

    /// PyPy-compatible `fget_w_globals_storage`.
    #[inline]
    pub fn fget_w_globals_storage(&self) -> PyObjectRef {
        self.get_w_globals_storage()
    }

    /// PyPy-compatible `_getcell`.
    #[inline]
    pub fn _getcell(&self, varindex: usize) -> PyObjectRef {
        self.locals_w()
            .as_slice()
            .get(self.nlocals() + varindex)
            .copied()
            .unwrap_or(PY_NULL)
    }

    /// PyPy-compatible `getclosure`.
    #[inline]
    pub fn getclosure(&self) -> PyObjectRef {
        PY_NULL
    }

    /// pyframe.py:223-261 initialize_frame_scopes.
    ///
    /// Errors mirror pyframe.py:242-246 (TypeError "directly executed code
    /// object may not contain free variables") and pyframe.py:251-253
    /// (ValueError "code object received a closure with an unexpected
    /// number of free variables") so callers can surface them through
    /// PyPy's OperationError-equivalent path instead of panicking.
    #[inline]
    pub fn initialize_frame_scopes(
        &mut self,
        outer_func: PyObjectRef,
        _code: *const (),
    ) -> Result<(), crate::PyError> {
        let code = unsafe { &*pyframe_get_pycode(self) };
        let flags = code.flags;
        if !flags.contains(CodeFlags::OPTIMIZED) {
            if flags.contains(CodeFlags::NEWLOCALS) {
                // pyframe.py:213 — class body binds a fresh locals namespace
                // dict object (`space.newdict(module=True)`).  `build_class`
                // replaces it with the `__prepare__` namespace via
                // `setdictscope`; an orphan NEWLOCALS frame still has a
                // usable mapping.
                let w_locals = unsafe { pyre_object::w_dict_new() };
                self.getorcreate_debug_data(-1).w_locals = w_locals;
            } else {
                // pyframe.py:216-218 — module scope binds `w_locals = w_globals`.
                // Bind the canonical W_DictObject so STORE_NAME / LOAD_NAME /
                // DELETE_NAME and `locals()` route through the object instead of
                // the raw DictStorage proxy.
                let w_globals = self.get_w_globals();
                self.getorcreate_debug_data(-1).w_locals = w_globals;
            }
        }

        let npure = npure_cellvars(code);
        let nfreevars = code.freevars.len();
        if npure == 0 && nfreevars == 0 {
            return Ok(());
        }

        let closure = if !outer_func.is_null() && unsafe { crate::is_function(outer_func) } {
            unsafe { crate::function_get_closure(outer_func) }
        } else {
            PY_NULL
        };
        let closure_size = if closure.is_null() {
            0
        } else {
            unsafe { w_tuple_len(closure) }
        };
        if nfreevars > 0 && outer_func.is_null() {
            return Err(crate::PyError::type_error(
                "directly executed code object may not contain free variables",
            ));
        }
        if closure_size != nfreevars {
            return Err(crate::PyError::value_error(format!(
                "code object received a closure with an unexpected number of free variables"
            )));
        }

        // CPython 3.11+ unified slot layout: only cellvars NOT in
        // varnames take a fresh slot.  See `npure_cellvars` and
        // `new_for_call_with_closure` for the call-site mirror.
        let mut index = code.varnames.len();
        for _ in 0..npure {
            self.locals_w_mut()[index] = pyre_object::w_cell_new(PY_NULL);
            index += 1;
        }
        for i in 0..nfreevars {
            self.locals_w_mut()[index] =
                unsafe { w_tuple_getitem(closure, i as i64).unwrap_or(PY_NULL) };
            index += 1;
        }
        Ok(())
    }

    /// pyframe.py:547-552 setdictscope(w_locals, skip_free_vars=False) —
    /// install `w_locals` as the frame's locals mapping and reflect
    /// its entries into the fastlocals via `locals2fast`.
    ///
    /// `pypy/interpreter/pyopcode.py:2003-2013 ensure_ns` admits any object
    /// exposing `__getitem__` as locals, so the mapping may be a plain dict
    /// (module / class / `exec(src, g, l)`) or an arbitrary mapping; both
    /// share this path and `STORE_NAME` / `LOAD_NAME` / `DELETE_NAME` route
    /// through `space.setitem` / `space.getitem` / `space.delitem` on it.
    #[inline]
    pub fn setdictscope(&mut self, w_locals: PyObjectRef) -> Result<(), crate::PyError> {
        self.getorcreate_debug_data(-1).w_locals = w_locals;
        self.locals2fast(false)
    }

    /// Read the frame's locals mapping registered by `setdictscope`
    /// (or `initialize_frame_scopes`).  Returns `PY_NULL` when the frame has
    /// no locals bound yet (a function before its first `fast2locals`).
    #[inline]
    pub fn get_w_locals(&self) -> PyObjectRef {
        self.getdebug_data()
            .map_or(pyre_object::PY_NULL, |data| data.w_locals)
    }

    /// pyframe.py:540-545 getdictscope — runs `fast2locals` then returns
    /// `self.debugdata.w_locals` (the locals mapping object).
    ///
    /// `fast2locals` lazily materialises a fresh dict for a function frame
    /// (pyframe.py:557 `space.newdict(instance=True)`) and caches it in
    /// `w_locals`, so repeated calls return the same object —
    /// `frame.f_locals is frame.f_locals` holds.
    #[inline]
    pub fn getdictscope(&mut self) -> Result<PyObjectRef, crate::PyError> {
        self.fast2locals()?;
        Ok(self.get_w_locals())
    }

    /// Create a minimal frame stub for passing to call dispatch.
    /// Used by MIFrame Box tracking when concrete_frame is unavailable.
    pub fn new_minimal(
        code: *const (),
        w_globals_storage: *mut crate::DictStorage,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        let nlocals = unsafe { (&*raw).varnames.len() };
        let ncells = unsafe { ncells(&*raw) };
        let size = nlocals + ncells + 16; // small stack
        // `pyframe.py:98 __init__(self, space, code, w_globals, ...)`
        // stores `w_globals` as the canonical W_DictObject directly.
        // Pyre carries both the raw storage pointer (`w_globals_storage`) and
        // the W_DictObject (`w_globals`); eager resolution at
        // frame construction matches PyPy's `self.w_globals = w_globals`
        // identity invariant without waiting for the first reader.
        let w_globals = if w_globals_storage.is_null() {
            PY_NULL
        } else {
            crate::baseobjspace::dict_storage_to_dict(w_globals_storage)
        };
        let w_builtin = crate::baseobjspace::frame_builtin_obj(w_globals, execution_context);
        // pyframe.py:103 — stamp `pycode.w_globals`; side effect only (the
        // gated debugdata snapshot retired in favour of `w_globals`).
        unsafe {
            crate::w_code_frame_stores_global(code as PyObjectRef, w_globals);
        }
        let mut frame = PyFrame {
            ob_header: frame_ob_header(),
            execution_context,
            pycode: code,
            locals_cells_stack_w: unsafe {
                alloc_fixed_array_from_vec(vec![pyre_object::PY_NULL; size])
            },
            valuestackdepth: nlocals + ncells,
            last_instr: -1,
            escaped: false,
            debugdata: std::ptr::null_mut(),
            lastblock: std::ptr::null_mut(),
            vable_token: 0,
            frame_finished_execution: false,
            f_generator_nowref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            w_builtin,
            w_globals,
        };
        frame
    }

    /// Test-helper constructor — creates a frame with a fresh execution
    /// context.
    ///
    /// TODO: PyPy has no equivalent — tests there call
    /// `space.appexec` or build the context explicitly before invoking
    /// `space.createframe`.  Pyre keeps `PyFrame::new` for test
    /// ergonomics (~67 callers across `pyre-interpreter`, `pyre-jit`,
    /// `pyre-jit-trace` test modules) but routes the body through
    /// `createframe` (PyPy `baseobjspace.py:796`) so every heap-allocated
    /// `PyFrame` flows through the canonical entry point.
    pub fn new(code: CodeObject) -> FrameBox {
        Self::new_with_context(code, Rc::new(PyExecutionContext::default()))
            .expect("PyFrame::new: test entry code must not carry freevars")
    }

    /// Module-entry adapter for `createframe` — leaks owned arguments
    /// into the raw-pointer shape that `createframe` expects, builds
    /// the module-entry `__dict__` (with `__name__ = "__main__"`), and
    /// returns the resulting heap-allocated frame.
    ///
    /// TODO: PyPy's `space.createframe(code, w_globals,
    /// outer_func)` (`baseobjspace.py:796`) takes already-constructed
    /// `code` and `w_globals` Python objects and reads execution context
    /// from `space.threadlocals`. PyPy callers (`pypy/interpreter/main.py
    /// run_module`, etc.) build `w_globals` and set `__name__` themselves
    /// before calling `createframe`. Pyre's adapter exists because:
    /// (1) Rust has no GC, so `code`/`w_globals`/`ec` lifetimes must be
    /// leaked into raw pointers manually; (2) pyre's `ExecutionContext`
    /// is per-frame (not per-space), so callers pass `Rc<EC>` explicitly.
    /// Three production callers (`pyrex/lib.rs`, `pyre-wasm/lib.rs`,
    /// `pyre-wasm-test/main.rs`) plus `PyFrame::new` (test helper)
    /// invoke this; without the adapter each would duplicate the 9-line
    /// leak setup.
    ///
    /// Returns `Result<Box<Self>, PyError>` so createframe's freevar /
    /// closure errors surface as interpreter errors instead of panics.
    pub fn new_with_context(
        code: CodeObject,
        execution_context: Rc<PyExecutionContext>,
    ) -> Result<FrameBox, crate::PyError> {
        // `fresh_module_globals` seeds `__builtins__ = space.builtin`
        // (PyPy `main.py:45 / Module.__init__` parity) into a proxy-less
        // celldict.  Just set `__name__` on top.
        let w_globals = execution_context.fresh_module_globals();
        // Root the fresh globals across the `__name__` store, code/object
        // allocations, and frame construction; `createframe_obj` stores
        // `w_globals` into the frame (and `w_code_set_w_globals` into the
        // code), both of which root it once they return.
        let _root = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_globals);
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                w_globals,
                "__name__",
                pyre_object::w_str_new("__main__"),
            );
        }
        let code_ptr = Box::into_raw(Box::new(code));
        let w_code = crate::w_code_new(code_ptr as *const ());
        unsafe {
            crate::w_code_set_w_globals(w_code, w_globals);
        }
        let ctx_ptr = Rc::into_raw(execution_context);
        crate::createframe_obj(w_code as *const (), w_globals, ctx_ptr, None)
    }

    /// PyFrame constructor body called from `createframe` (PyPy
    /// `baseobjspace.py:796`) when `outer_func` is `None` — sets up the
    /// fixed-array stack, debug data, w_globals binding, and module-level
    /// `w_locals = w_globals` semantics.  Crate-private helper kept as the
    /// namespace constructor shape even when call sites allocate directly.
    pub(crate) fn new_with_namespace(
        code: *const (),
        execution_context: *const PyExecutionContext,
        w_globals_storage: *mut DictStorage,
    ) -> Self {
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        let code_ref = unsafe { &*raw };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        // `pyframe.py:98 __init__` — `self.w_globals = w_globals` stores
        // the W_DictObject directly; eager `dict_storage_to_dict`
        // mirrors the identity invariant on pyre's split layout.
        let w_globals = if w_globals_storage.is_null() {
            PY_NULL
        } else {
            crate::baseobjspace::dict_storage_to_dict(w_globals_storage)
        };
        let w_builtin = crate::baseobjspace::frame_builtin_obj(w_globals, execution_context);
        // pyframe.py:103 — stamp `pycode.w_globals`; side effect only (the
        // gated debugdata snapshot retired in favour of `w_globals`).
        unsafe {
            crate::w_code_frame_stores_global(code as PyObjectRef, w_globals);
        }
        let mut frame = PyFrame {
            ob_header: frame_ob_header(),
            execution_context,
            pycode: code,
            locals_cells_stack_w: unsafe {
                alloc_fixed_array_with_header(num_locals + num_cells + max_stack, PY_NULL)
            },
            valuestackdepth: num_locals + num_cells,
            last_instr: -1,
            escaped: false,
            debugdata: std::ptr::null_mut(),
            lastblock: std::ptr::null_mut(),
            vable_token: 0,
            frame_finished_execution: false,
            f_generator_nowref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            w_builtin,
            w_globals,
        };
        // Module-level w_locals = w_globals binding flows naturally
        // through `createframe → initialize_frame_scopes` since RustPython
        // codegen emits empty flags for the module seed CodeInfo
        // (pyframe.py:216-218).  This constructor bypasses
        // initialize_frame_scopes, so still bind w_locals to w_globals
        // explicitly to match what `createframe` would observe — in the
        // object form (the canonical W_DictObject), not the raw storage.
        frame.getorcreate_debug_data(-1).w_locals = w_globals;
        frame
    }

    /// RPython MetaInterp traces against its own MIFrame stack instead of
    /// mutating the live interpreter frame in place. pyre still executes
    /// bytecodes concretely during tracing, so use an owned snapshot when
    /// recording a trace to keep the real frame state unchanged until the
    /// interpreter actually executes the same path.
    pub fn snapshot_for_tracing(&self) -> FrameBox {
        // A tracer holds this snapshot off the `CURRENT_FRAME` chain across
        // the whole `trace_bytecode` walk, during which a major GC cycle can
        // complete; no root reaches it, so it must NOT have GC lifetime —
        // `new_boxed` gives it a deterministic scope-end free.
        let mut frame =
            FrameBox::new_boxed(self.build_snapshot_frame(FrameLocalsArrayAllocation::StdAlloc));
        // fix_array_ptrs AFTER Box allocation: inline_buf ptr must
        // point to the heap-allocated frame, not a stale stack address.
        frame.fix_array_ptrs();
        frame
    }

    /// Build the copied `PyFrame` value shared by the tracer snapshot and
    /// the generator snapshot.  Frame-LOCAL state (`locals_cells_stack_w` /
    /// `valuestackdepth` / `last_instr`) is COPIED, so snapshot mutations to
    /// locals/stack are discarded — the abort-safety the snapshot exists
    /// for.  `w_globals` is the SAME dict ptr, so a concrete shared-heap
    /// write during recording would leak to the real heap and double-apply
    /// on the compiled loop's re-run; Gap 10 removed that path (inline-frame
    /// STORE_GLOBAL records as deferred IR, applied exactly once).
    fn build_snapshot_frame(&self, allocation: FrameLocalsArrayAllocation) -> PyFrame {
        PyFrame {
            ob_header: frame_ob_header(),
            execution_context: self.execution_context,
            pycode: self.pycode,
            locals_cells_stack_w: unsafe {
                let values = self.locals_w().to_vec();
                let array = alloc_frame_locals_array(values.len(), PY_NULL, allocation);
                for (i, value) in values.into_iter().enumerate() {
                    (*array).items_mut_ptr().add(i).write(value);
                }
                remember_frame_locals_array(array);
                array
            },
            valuestackdepth: self.valuestackdepth,
            last_instr: self.last_instr,
            escaped: self.escaped,
            debugdata: unsafe { clone_debugdata_ptr(self.debugdata, allocation) },
            lastblock: unsafe { clone_block_chain(self.lastblock, allocation) },
            vable_token: self.vable_token,
            frame_finished_execution: self.frame_finished_execution,
            f_generator_nowref: self.f_generator_nowref,
            w_yielding_from: self.w_yielding_from,
            f_backref: self.f_backref,
            w_builtin: self.w_builtin,
            w_globals: self.w_globals,
        }
    }

    /// Snapshot a borrowed frame into a GC-managed owned frame for
    /// `initialize_as_generator`.  Unlike `snapshot_for_tracing` this uses
    /// `FrameBox::new` (GC lifetime): a generator's suspended frame lives as
    /// long as the generator object reaches it (`generator.py` holds the
    /// frame), and the generator's custom trace greys the frame block.
    pub fn snapshot_for_generator(&self) -> FrameBox {
        // `build_snapshot_frame` finishes by cloning `lastblock`; no later
        // field construction allocates. `FrameBox::new` then uses the
        // non-collecting `alloc_in_oldgen` path and write-barriers the frame
        // before this returns, so any nursery block chain is reached from the
        // remembered set on the next minor collection.
        let mut frame =
            FrameBox::new(self.build_snapshot_frame(FrameLocalsArrayAllocation::OldGenGc));
        frame.fix_array_ptrs();
        frame
    }

    /// Number of local variable slots (from code object).
    ///
    /// `@jit.elidable`: read-only access to `varnames.len()`, no
    /// allocation, no raise.  `EF_ELIDABLE_CANNOT_RAISE` parity.
    #[majit_macros::elidable_cannot_raise]
    #[inline]
    pub fn nlocals(&self) -> usize {
        unsafe { (&*pyframe_get_pycode(self)).varnames.len() }
    }

    /// Number of cell + free variable slots.
    #[majit_macros::elidable_cannot_raise]
    #[inline]
    pub fn ncells(&self) -> usize {
        // `npure_cellvars` (the O(cellvars × varnames) overlap count) is
        // code-invariant and cached on the `PyCode` wrapper; `freevars.len()`
        // is a direct read.  Falls back to the full walk for stub frames
        // whose `pycode` wrapper carries the `u32::MAX` sentinel.
        let code = unsafe { &*pyframe_get_pycode(self) };
        match unsafe {
            crate::pycode::w_code_npure_cellvars(self.pycode as pyre_object::PyObjectRef)
        } {
            Some(npure) => npure + code.freevars.len(),
            None => ncells(code),
        }
    }

    /// First index of the operand stack (after locals and cells).
    #[inline]
    pub fn stack_base(&self) -> usize {
        self.nlocals() + self.ncells()
    }

    // ── Stack operations ──────────────────────────────────────────────

    #[inline]
    pub fn push(&mut self, value: PyObjectRef) {
        self.assert_stack_index(self.valuestackdepth);
        let idx = self.valuestackdepth;
        self.locals_w_mut()[idx] = value;
        self.valuestackdepth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        assert!(self.valuestackdepth > self.stack_base());
        let depth = self.valuestackdepth - 1;
        let value = self.locals_w()[depth];
        self.locals_w_mut()[depth] = PY_NULL;
        self.valuestackdepth = depth;
        value
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        self.locals_w()[self.valuestackdepth - 1]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        self.locals_w()[self.valuestackdepth - 1 - depth]
    }

    /// PyPy-compatible stack operation aliases.
    #[inline]
    pub fn pushvalue(&mut self, value: PyObjectRef) {
        self.push(value)
    }

    /// pyframe.py:304-307 pushvalue_none
    #[inline]
    pub fn pushvalue_none(&mut self) {
        let depth = self.valuestackdepth;
        debug_assert!(self.locals_w()[depth].is_null());
        self.valuestackdepth = depth + 1;
    }

    /// PyPy-compatible stack index guard.
    #[inline]
    pub fn assert_stack_index(&self, index: usize) {
        debug_assert!(self._check_stack_index(index));
    }

    /// PyPy-compatible stack index validator.
    ///
    /// Asserts both lower and upper bounds: a valid stack write goes to
    /// `stack_base() <= index < locals_cells_stack_w.len()`. Pyre's
    /// `PyObjectArray` is allocated with `nlocals + ncells +
    /// max_stackdepth` slots (pyframe.rs:1091), so writing at or past
    /// `array_len` overruns the heap buffer — catastrophic in release
    /// mode, where `PyObjectArray` indexing is unchecked. This guard
    /// converts the heap overrun into a debug-mode assertion failure
    /// that surfaces tracer/JIT vsd miscalculations at the source
    /// rather than as silent malloc corruption later.
    #[inline]
    pub fn _check_stack_index(&self, index: usize) -> bool {
        index >= self.stack_base() && index < self.locals_w().len()
    }

    /// pyframe.py:313-314 popvalue
    #[inline]
    pub fn popvalue(&mut self) -> PyObjectRef {
        let value = self.popvalue_maybe_none();
        assert!(!value.is_null());
        value
    }

    /// pyframe.py:316-322 popvalue_maybe_none
    #[inline]
    pub fn popvalue_maybe_none(&mut self) -> PyObjectRef {
        let depth = self.valuestackdepth - 1;
        self.assert_stack_index(depth);
        let w_object = self.locals_w()[depth];
        self.locals_w_mut()[depth] = PY_NULL;
        self.valuestackdepth = depth;
        w_object
    }

    /// PyPy `PyFrame._new_popvalues` factory.
    #[inline]
    pub fn _new_popvalues() -> fn(&mut Self, usize) -> Vec<PyObjectRef> {
        Self::popvalues
    }

    /// PyPy-compatible pop-values helper.
    #[inline]
    pub fn popvalues(&mut self, n: usize) -> Vec<PyObjectRef> {
        let mut out = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            out[idx] = self.popvalue();
        }
        out
    }

    /// PyPy-compatible `popvalues_mutable`.
    #[inline]
    pub fn popvalues_mutable(&mut self, n: usize) -> Vec<PyObjectRef> {
        self.popvalues(n)
    }

    /// pyframe.py:337-345 peekvalues
    #[inline]
    pub fn peekvalues(&self, n: usize) -> Vec<PyObjectRef> {
        let base = self.valuestackdepth - n;
        // Reads cover `[base, valuestackdepth)`; the highest index is
        // `valuestackdepth - 1 < len`, so only the lower `base` bound needs
        // guarding. When `n == 0` nothing is read and `base ==
        // valuestackdepth` may equal `len` at peak depth, so skip the
        // upper-bound `assert_stack_index` for the empty peek.
        if n > 0 {
            self.assert_stack_index(base);
        } else {
            debug_assert!(base >= self.stack_base());
        }
        let mut values_w = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            values_w[idx] = self.locals_w()[base + idx];
        }
        values_w
    }

    /// pyframe.py:348-355 dropvalues
    #[inline]
    pub fn dropvalues(&mut self, n: usize) {
        let finaldepth = self.valuestackdepth - n;
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.locals_w_mut()[idx] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// PyPy-compatible `pushrevvalues`.
    #[inline]
    pub fn pushrevvalues(&mut self, _n: usize, values_w: &[PyObjectRef]) {
        let n = if _n == 0 { values_w.len() } else { _n };
        assert!(n <= values_w.len());
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            self.push(values_w[idx]);
        }
    }

    /// PyPy-compatible `dupvalues`.
    #[inline]
    pub fn dupvalues(&mut self, n: usize) {
        let values = self.peekvalues(n);
        for value in values {
            self.push(value);
        }
    }

    /// PyPy-compatible `peekvalue()`.
    #[inline]
    pub fn peekvalue(&self, index_from_top: usize) -> PyObjectRef {
        self.peek_at(index_from_top)
    }

    /// PyPy-compatible `peekvalue_maybe_none()`.
    #[inline]
    pub fn peekvalue_maybe_none(&self, index_from_top: usize) -> PyObjectRef {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(usize::MAX);
        if index == usize::MAX || index < self.stack_base() {
            return PY_NULL;
        }
        self.locals_w()[index]
    }

    /// PyPy-compatible `settopvalue()`.
    #[inline]
    pub fn settopvalue(&mut self, value: PyObjectRef, index_from_top: usize) {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(0);
        self.assert_stack_index(index);
        assert!(index < self.valuestackdepth);
        self.locals_w_mut()[index] = value;
    }

    /// PyPy-compatible `dropvaluesuntil()`.
    #[inline]
    pub fn dropvaluesuntil(&mut self, finaldepth: usize) {
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.locals_w_mut()[idx] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// pyframe.py:186 append_block
    #[inline]
    pub fn append_block(&mut self, mut block: FrameBlock) {
        let allocation = self.aux_allocation();
        let mut previous = self.lastblock;
        let _root = unsafe { FrameBlockRoot::new(&mut previous) };
        block.previous = std::ptr::null_mut();
        let node = unsafe { alloc_frame_block(block, allocation) };
        // Both links need barriers: a nursery-full allocation fallback can
        // make `node` old while `previous` is young, and this old frame usually
        // receives a young `node`.
        unsafe { (*node).previous = previous };
        remember_frame_block_node(node);
        self.lastblock = node;
        if pyre_object::gc_hook::try_gc_owns_object(self as *mut PyFrame as *mut u8) {
            pyre_object::gc_hook::try_gc_write_barrier(self as *mut PyFrame as *mut u8);
        }
    }

    /// pyframe.py:190 pop_block
    #[inline]
    pub fn pop_block(&mut self) -> Option<FrameBlock> {
        if self.lastblock.is_null() {
            return None;
        }
        unsafe {
            let current = self.lastblock;
            if pyre_object::gc_hook::try_gc_owns_object(current as *mut u8) {
                let mut result = *current;
                self.lastblock = result.previous;
                result.previous = std::ptr::null_mut();
                Some(result)
            } else {
                let block = Box::from_raw(current);
                self.lastblock = block.previous;
                let mut result = *block;
                result.previous = std::ptr::null_mut();
                Some(result)
            }
        }
    }

    /// pyframe.py:195 blockstack_non_empty
    #[inline]
    pub fn blockstack_non_empty(&self) -> bool {
        !self.lastblock.is_null()
    }

    /// PyPy-compatible exception-info unwind helper.
    #[inline]
    pub fn _exc_info_unroll(&self, _for_hidden: bool) -> PyObjectRef {
        let _ = _for_hidden;
        pyre_object::w_none()
    }

    /// PyPy-compatible unexpected-exception converter.
    #[inline]
    pub fn _convert_unexpected_exception(&self, _e: PyObjectRef) -> PyObjectRef {
        let _ = _e;
        pyre_object::w_none()
    }

    /// PyPy-compatible pickle state helper.
    #[inline]
    pub fn _reduce_state(&self) -> PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::w_int_new(self.last_instr as i64),
            pyre_object::w_int_new(self.valuestackdepth as i64),
        ])
    }

    /// PyPy-compatible `descr__reduce__`.
    #[inline]
    pub fn descr__reduce__(&self) -> PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            self._reduce_state(),
        ])
    }

    /// PyPy-compatible `descr__setstate__`.
    #[inline]
    pub fn descr__setstate__(&mut self, _state: PyObjectRef) {
        let _ = _state;
    }

    /// pyframe.py:198 get_blocklist — walk linked list, return in reverse order.
    #[inline]
    pub fn get_blocklist(&self) -> Vec<FrameBlock> {
        let mut lst = Vec::new();
        let mut block = self.lastblock;
        while !block.is_null() {
            unsafe {
                let mut entry = *block;
                entry.previous = std::ptr::null_mut();
                lst.push(entry);
                block = (*block).previous;
            }
        }
        lst
    }

    /// pyframe.py:207 set_blocklist — rebuild linked list from slice.
    #[inline]
    pub fn set_blocklist(&mut self, lst: &[FrameBlock]) {
        unsafe { clear_block_chain(&mut self.lastblock) };
        let mut i = lst.len();
        while i > 0 {
            i -= 1;
            self.append_block(lst[i]);
        }
    }

    /// PyPy-compatible execution entrypoint.
    #[inline]
    pub fn run(&mut self) -> crate::PyResult {
        if self._is_generator_or_coroutine() {
            self.initialize_as_generator()
        } else {
            self.execute_frame(None, None)
        }
    }

    /// `run`, but dispatching the non-generator body through the registered
    /// eval function (`call::get_eval_fn`) instead of the plain interpreter.
    ///
    /// `interp_jit.py:81-99` applies the jitdriver to every frame uniformly,
    /// so module / class / exec'd code reaches `jit_merge_point` exactly like
    /// a called function does.  `run` hardcodes the plain interpreter, which
    /// keeps these entry frames off the portal; this variant restores the
    /// uniform routing for the run-sites that want it (exec / eval / import /
    /// class body), matching how `call_user_function` reaches the portal via
    /// `get_eval_fn`.
    ///
    /// `settrace` is honored without forcing plain eval: installing a tracefunc
    /// does not set `FORCE_PLAIN_EVAL` (that flag is a blackhole / `force_fn`
    /// re-entry guard, unrelated to tracing).  The JIT eval override
    /// (`eval_with_jit`) instead reads `ec.w_tracefunc` inline each bytecode
    /// for line events and routes non-JIT-eligible frames through
    /// `execute_frame` (so `call_trace` / `return_trace` frame events still
    /// fire), exactly as a normal call reaches the portal via `get_eval_fn`;
    /// and when the portal declines a frame it falls back to `execute_frame`
    /// (plain), so this never re-enters the portal.
    #[inline]
    pub fn run_with_jit(&mut self) -> crate::PyResult {
        if self._is_generator_or_coroutine() {
            self.initialize_as_generator()
        } else {
            crate::call::get_eval_fn()(self)
        }
    }

    /// pyframe.py:300 resume_execute_frame (send-path only).
    ///
    /// A suspended delegate is resumed by `generator_send_ex` before this
    /// method runs.  Once that delegate completes, this path receives the
    /// normal outer-frame input again.
    #[inline]
    pub fn resume_execute_frame(
        &mut self,
        w_arg_or_err: PyObjectRef,
    ) -> Result<usize, crate::PyError> {
        if self.last_instr != -1 {
            self.pushvalue(w_arg_or_err);
            Ok(self.last_instr as usize + 1)
        } else {
            Ok(0)
        }
    }

    /// PyPy-compatible execution entrypoint with optional inbound values.
    #[inline]
    #[allow(unused_variables)]
    pub fn execute_frame(
        &mut self,
        w_inputvalue: Option<PyObjectRef>,
        operr: Option<crate::PyError>,
    ) -> crate::PyResult {
        if operr.is_none() {
            if let Some(w_arg_or_err) = w_inputvalue {
                let _ = self.resume_execute_frame(w_arg_or_err)?;
            }
        }
        crate::eval::eval_frame_plain_with_operr(self, operr)
    }

    /// pyframe.py:521-522 `hide(self): return self.pycode.hidden_applevel`.
    ///
    /// PyPy creates a `PyFrame` for every callable that has a Code
    /// object — including gateway builtins (`BuiltinCode`,
    /// `gateway.py:743 hidden_applevel = True`) and the
    /// `app_main.py`-style internal frames
    /// (`pycompiler.compile(..., hidden_applevel=True)`).  The
    /// `hide()` flag lets `_trace` skip those internal frames so
    /// user-visible callbacks never see the gateway machinery.
    ///
    /// Pyre does not allocate a `PyFrame` for builtin calls — the
    /// `dispatch_callable` builtin closure (`runtime_ops.rs:275`)
    /// invokes the BuiltinCode function pointer directly, with no
    /// frame attached, so the trace path never observes builtin
    /// frames.  Hidden-applevel user frames (the `app_main.py`
    /// case) are not reachable in pyre yet either — pyre has no
    /// `pycompiler.compile(hidden_applevel=True)` call site.  The
    /// field still lives on PyCode, and this accessor mirrors
    /// PyPy's object-field read.
    #[inline]
    pub fn hide(&self) -> bool {
        unsafe { crate::w_code_hidden_applevel(self.pycode as PyObjectRef) }
    }

    /// pyframe.py:183 mark_as_escaped
    #[inline]
    pub fn mark_as_escaped(&mut self) {
        self.escaped = true;
    }

    /// pyframe.py:216-220 `get_builtin` — returns `self.builtin` (the
    /// per-frame picked builtin Module, set at frame creation by
    /// `pick_builtin(w_globals)`).  Falls back to the EC's default
    /// builtin when the frame was constructed without globals (e.g.
    /// `PyFrame::new_minimal` stub frames).
    #[inline]
    pub fn get_builtin(&self) -> PyObjectRef {
        if !self.w_builtin.is_null() {
            return self.w_builtin;
        }
        if self.execution_context.is_null() {
            return pyre_object::PY_NULL;
        }
        unsafe { (*self.execution_context).get_builtin() }
    }

    /// PyPy-compatible `get_f_back`.
    #[inline]
    pub fn get_f_back(&self) -> *mut PyFrame {
        self.f_backref
    }

    /// pyframe.py:768-771 `fget_f_builtins` — `self.get_builtin()
    /// .getdict(space)`, which `module.py:20` defines as `self.w_dict`.
    /// Pyre's `w_module_new` constructs `w_dict` at allocation time so
    /// every Module surfaces a stable, non-null identity here.
    #[inline]
    pub fn fget_f_builtins(&self) -> PyObjectRef {
        let w_builtin = self.get_builtin();
        if w_builtin.is_null() {
            return pyre_object::PY_NULL;
        }
        if unsafe { pyre_object::is_module(w_builtin) } {
            return unsafe { pyre_object::w_module_get_w_dict(w_builtin) };
        }
        w_builtin
    }

    /// pyframe.py:764 `fget_f_back` — the next non-hidden frame, i.e.
    /// `ExecutionContext.getnextframe_nohidden(self)`, skipping
    /// `hidden_applevel` gateway / bridge frames.  The plain
    /// `get_f_back()` accessor (used internally, including by the
    /// nohidden walker itself) returns the raw `f_backref` link.
    #[inline]
    pub fn fget_f_back(&self) -> *mut PyFrame {
        crate::executioncontext::ExecutionContext::getnextframe_nohidden(
            self as *const PyFrame as *mut PyFrame,
        )
    }

    /// pyframe.py:641-642 fget_code → self.getcode().  Returns the `PyCode`
    /// wrapper object (`self.pycode`) itself, which is what `frame.f_code`
    /// yields to Python — not the inner `CodeObject`.
    #[inline]
    pub fn fget_f_code(&self) -> PyObjectRef {
        self.pycode as PyObjectRef
    }

    /// pyframe.py:849-853 descr_repr — `<frame at 0x…, file '…', line …,
    /// code …>` via `getrepr(space, "frame", moreinfo)`.
    pub fn descr_repr(&self) -> String {
        let code = self.code();
        format!(
            "<frame at {:p}, file '{}', line {}, code {}>",
            self as *const PyFrame,
            code.source_path.as_str(),
            self.get_last_lineno(),
            code.obj_name.as_str(),
        )
    }

    /// `frame_clear` (`frame.clear()`): clear most references held by the
    /// frame.  Refuses on an executing (non-generator) frame or a running
    /// generator (`"cannot clear an executing frame"`) and on a generator
    /// frame suspended at a `yield` (`"cannot clear a suspended frame"`);
    /// a not-yet-started or already-exhausted generator is finalized
    /// (marked exhausted).  Otherwise clears `w_f_trace`, resets
    /// `w_locals` to a fresh dict, and replaces every local / cell / free
    /// var / stack slot (cells are rebound to fresh empty cells so a
    /// shared inner/outer cell is not mutated).
    pub fn descr_clear(&mut self) -> Result<(), crate::PyError> {
        if !self.frame_finished_execution {
            if !self._is_generator_or_coroutine() {
                return Err(crate::PyError::runtime_error(
                    "cannot clear an executing frame",
                ));
            }
            let w_gen = self.get_generator();
            if !w_gen.is_null() {
                if unsafe { pyre_object::generator::w_generator_is_running(w_gen) } {
                    return Err(crate::PyError::runtime_error(
                        "cannot clear an executing frame",
                    ));
                }
                // A generator started but not exhausted is suspended at a
                // `yield`; clearing it would drop the live operand stack
                // out from under a later `resume`, so refuse.
                let suspended = unsafe {
                    pyre_object::generator::w_generator_is_started(w_gen)
                        && !pyre_object::generator::w_generator_is_exhausted(w_gen)
                };
                if suspended {
                    return Err(crate::PyError::runtime_error(
                        "cannot clear a suspended frame",
                    ));
                }
                // Not started or already exhausted: finalize.
                unsafe { pyre_object::generator::w_generator_set_exhausted(w_gen) };
            }
        }

        if let Some(debug) = self.getdebug() {
            let had_locals = !debug.w_locals.is_null();
            // Allocate before remembering debugdata: a minor can happen here.
            let w_locals = had_locals.then(|| unsafe { pyre_object::w_dict_new() });
            let d = self.getorcreate_debug_data(-1);
            d.w_f_trace = pyre_object::PY_NULL;
            if let Some(w_locals) = w_locals {
                d.w_locals = w_locals;
            }
        }

        // Clear locals, cell/free vars, and the stack.  A cell slot is
        // rebound to a fresh empty cell (not mutated in place, since it may
        // still be shared by an inner/outer function).
        let len = self.locals_w().len();
        for i in 0..len {
            let w_oldvalue = self.locals_w()[i];
            let w_newvalue = if !w_oldvalue.is_null() && unsafe { pyre_object::is_cell(w_oldvalue) }
            {
                pyre_object::w_cell_new(pyre_object::PY_NULL)
            } else {
                pyre_object::PY_NULL
            };
            self.locals_w_mut()[i] = w_newvalue;
        }
        self.valuestackdepth = 0;
        Ok(())
    }

    /// pyframe.py:773 fget_f_lasti → space.newint(self.last_instr)
    #[inline]
    pub fn fget_f_lasti(&self) -> isize {
        self.last_instr
    }

    /// PyPy-compatible `fget_f_trace`.
    #[inline]
    pub fn fget_f_trace(&self) -> PyObjectRef {
        self.get_w_f_trace()
    }

    /// pyframe.py:785-791 fset_f_trace
    #[inline]
    pub fn fset_f_trace(&mut self, w_trace: PyObjectRef) {
        if w_trace.is_null() || w_trace == pyre_object::w_none() {
            self.getorcreate_debug_data(-1).w_f_trace = pyre_object::PY_NULL;
        } else {
            let lineno = self.get_last_lineno();
            let d = self.getorcreate_debug_data(-1);
            d.w_f_trace = w_trace;
            d.f_lineno = lineno;
        }
    }

    /// pyframe.py:793-794 fdel_f_trace
    #[inline]
    pub fn fdel_f_trace(&mut self) {
        self.getorcreate_debug_data(-1).w_f_trace = pyre_object::PY_NULL;
    }

    /// pyframe.py:153-157 get_f_trace_lines
    #[inline]
    pub fn get_f_trace_lines(&self) -> bool {
        self.getdebug_data().map_or(true, |d| d.f_trace_lines)
    }

    /// pyframe.py:159-163 get_f_trace_opcodes
    #[inline]
    pub fn get_f_trace_opcodes(&self) -> bool {
        self.getdebug_data().map_or(false, |d| d.f_trace_opcodes)
    }

    /// pyframe.py:796-797 fget_f_trace_lines
    #[inline]
    pub fn fget_f_trace_lines(&self) -> bool {
        self.get_f_trace_lines()
    }

    /// pyframe.py:799-800 fset_f_trace_lines
    #[inline]
    pub fn fset_f_trace_lines(&mut self, value: bool) {
        self.getorcreate_debug_data(-1).f_trace_lines = value;
    }

    /// pyframe.py:802-803 fget_f_trace_opcodes
    #[inline]
    pub fn fget_f_trace_opcodes(&self) -> bool {
        self.get_f_trace_opcodes()
    }

    /// pyframe.py:805-806 fset_f_trace_opcodes
    #[inline]
    pub fn fset_f_trace_opcodes(&mut self, value: bool) {
        self.getorcreate_debug_data(-1).f_trace_opcodes = value;
    }

    /// PyPy-compatible `fget_f_exc_type`.
    #[inline]
    pub fn fget_f_exc_type(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_value`.
    #[inline]
    pub fn fget_f_exc_value(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_traceback`.
    #[inline]
    pub fn fget_f_exc_traceback(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_restricted`.
    #[inline]
    pub fn fget_f_restricted(&self) -> bool {
        false
    }

    /// pyframe.py:861-863 get_last_lineno → pytraceback.offset2lineno(pycode, last_instr)
    #[inline]
    pub fn get_last_lineno(&self) -> isize {
        offset2lineno(self.code(), self.last_instr) as isize
    }

    /// pyframe.py:660-671 fget_f_lineno
    #[inline]
    pub fn fget_f_lineno(&self) -> isize {
        if self.get_w_f_trace().is_null() {
            self.get_last_lineno()
        } else {
            let f_lineno = self.getdebug_data().map_or(-1, |dd| dd.f_lineno);
            if f_lineno == -1 {
                self.code()
                    .first_line_number
                    .map_or(-1, |n| n.get() as isize)
            } else {
                f_lineno
            }
        }
    }

    /// `frameobject.c frame_lineno_set` — set the line the frame will
    /// resume at, validating the jump against [`mark_stacks`].
    ///
    /// Only a trace function may jump (`get_w_f_trace()` non-null); the
    /// jump target must land on a real source line, must not enter an
    /// exception handler / for-loop body / re-raise block, and the source
    /// and target abstract operand stacks must be compatible.  On success
    /// the operand stack is unwound to the target depth (closing dropped
    /// values, restoring `exc_info` for popped `Except` slots) and
    /// `last_instr` is repointed.
    ///
    /// Deviation: `frame_lineno_set` also binds `None` to any local the
    /// compiler proved live-but-unbound at the target (the `PyStackRef_None`
    /// pass over `co_localspluskinds`).  pyre keeps unbound locals as
    /// `PY_NULL` guarded by `LOAD_FAST_CHECK`, so that pass is omitted; the
    /// only observable difference is an `UnboundLocalError` where the
    /// jumped-to code reads such a local before assigning it, versus
    /// silently seeing `None`.
    pub fn fset_f_lineno(&mut self, new_f_lineno: isize) -> Result<(), crate::PyError> {
        // frame_lineno_set: you can only jump from within a trace
        // function, not via `_getframe` / similar hackery.
        if self.get_w_f_trace().is_null() {
            return Err(crate::PyError::value_error(
                "f_lineno can only be set by a trace function.",
            ));
        }
        // A newly-entered frame (call event) has no dispatched
        // instruction yet.
        if self.last_instr == -1 {
            return Err(crate::PyError::value_error(
                "can't jump from the 'call' trace event of a new frame",
            ));
        }
        // Jumps are allowed from a `line` trace event, and — outside a
        // line event — only when the frame is suspended at `YIELD_VALUE`
        // (`frame_lineno_set_impl` lists `PY_MONITORING_EVENT_PY_YIELD`
        // in the same allowed group as line/jump events).  The pending
        // yield value is accounted for by the `is_suspended` unwind
        // below.  pyframe.py:685-689:
        //     if not d.is_in_line_tracing:
        //         if ord(code[self.last_instr]) != YIELD_VALUE:
        //             raise "can only jump from a 'line' trace event"
        if !self.getdebug().map_or(false, |d| d.is_in_line_tracing) {
            let at_yield = matches!(
                crate::pyopcode::decode_instruction_at(self.code(), self.last_instr as usize),
                Some((crate::bytecode::Instruction::YieldValue { .. }, _))
            );
            if !at_yield {
                return Err(crate::PyError::value_error(
                    "can only jump from a 'line' trace event",
                ));
            }
        }
        // `frame_is_suspended` — a generator/coroutine frame that has
        // started, is not currently running, and is not exhausted has a
        // pending `yield` value on the stack that the resume will pop, so
        // the unwind must account for it.
        let is_suspended = if self._is_generator_or_coroutine() {
            let w_gen = self.get_generator();
            !w_gen.is_null()
                && unsafe {
                    pyre_object::generator::w_generator_is_started(w_gen)
                        && !pyre_object::generator::w_generator_is_running(w_gen)
                        && !pyre_object::generator::w_generator_is_exhausted(w_gen)
                }
        } else {
            false
        };

        let code = self.code();
        let len = code.instructions.len();
        let first_line = code.first_line_number.map(|n| n.get() as i32).unwrap_or(1);

        let mut new_lineno = new_f_lineno as i32;
        if new_lineno < first_line {
            return Err(crate::PyError::value_error(format!(
                "line {new_lineno} comes before the current code block"
            )));
        }

        let lines = mark_lines(code, len);
        new_lineno = mark_first_line_not_before(&lines, new_lineno);
        if new_lineno < 0 {
            return Err(crate::PyError::value_error(format!(
                "line {new_f_lineno} comes after the current code block"
            )));
        }

        let stacks = mark_stacks(code, len);
        let last_instr = self.last_instr as usize;
        let start_stack = *stacks.get(last_instr).unwrap_or(&MARK_UNINITIALIZED);

        let mut best_stack = MARK_OVERFLOWED;
        let mut best_addr: isize = -1;
        let mut err: i32 = -1;
        let mut msg: String = "cannot find bytecode for specified line".to_string();
        for i in 0..len {
            if lines[i] != new_lineno {
                continue;
            }
            let target_stack = stacks[i];
            if mark_compatible_stack(start_stack, target_stack) {
                err = 0;
                if target_stack > best_stack {
                    best_stack = target_stack;
                    best_addr = i as isize;
                }
            } else if err < 0 {
                if start_stack == MARK_OVERFLOWED {
                    msg = "stack too deep to analyze".to_string();
                } else if start_stack == MARK_UNINITIALIZED {
                    msg = "can't jump from unreachable code".to_string();
                } else {
                    msg = mark_explain_incompatible_stack(target_stack).to_string();
                    err = 1;
                }
            }
        }
        if err != 0 {
            return Err(crate::PyError::value_error(msg));
        }

        // Unwind the operand stack from `start_stack` down to
        // `best_stack`, closing dropped values.  A dropped `Except` slot
        // restores the previous `exc_info` (pyre keeps the active
        // exception in the TLS current-exception slot, saved beneath the
        // handler on the value stack — see `push_exc_info`).
        let mut cur_stack = start_stack;
        if is_suspended {
            // Account for the value popped by yield.
            cur_stack = mark_pop_value(cur_stack);
        }
        while cur_stack > best_stack {
            let popped = self.popvalue();
            if mark_top_of_stack(cur_stack) == StackKind::Except as i64 {
                // The popped value is the saved previous exception; make
                // it current again.
                crate::eval::set_current_exception(popped);
            }
            cur_stack = mark_pop_value(cur_stack);
        }

        self.getorcreate_debug_data(-1).f_lineno = new_lineno as isize;
        self.last_instr = best_addr;
        Ok(())
    }

    /// PyPy-compatible `setfastscope`.
    #[inline]
    pub fn setfastscope(&mut self, scope_w: &[PyObjectRef]) {
        assert!(scope_w.len() <= self.nlocals());
        for (index, value) in scope_w.iter().copied().enumerate() {
            self.locals_w_mut()[index] = value;
        }
        // In this port, cell initialization is performed as part of scope load.
        self.init_cells();
    }

    /// pyframe.py:601-636 locals2fast(skip_free_vars=False) — reflect the
    /// locals mapping back into the fastlocals.  Reads each varname / cellvar
    /// / freevar from the mapping via `space.finditem_str` (KeyError →
    /// missing); a frame with no locals bound has nothing to copy.
    pub fn locals2fast(&mut self, skip_free_vars: bool) -> Result<(), crate::PyError> {
        let w_locals = self.get_w_locals();
        if w_locals.is_null() {
            return Ok(());
        }
        let code_ptr = unsafe { pyframe_get_pycode(self) };
        let code = unsafe { &*code_ptr };
        let numlocals = code.varnames.len();

        let mut new_fastlocals_w = vec![PY_NULL; numlocals];
        for i in 0..numlocals {
            // CO_FAST_HIDDEN slots are not reflected in the locals mapping —
            // preserve the current fast value instead of clearing it.
            if hidden_local(code, i) {
                new_fastlocals_w[i] = self.locals_w()[i];
                continue;
            }
            let name = &code.varnames[i];
            if let Some(w_value) = finditem_str_object(w_locals, name)? {
                new_fastlocals_w[i] = w_value;
            }
        }
        self.setfastscope(&new_fastlocals_w);

        let pure_cells: Vec<&_> = code
            .cellvars
            .iter()
            .filter(|c| {
                let cs: &str = c.as_ref();
                !code.varnames.iter().any(|v| {
                    let vs: &str = v.as_ref();
                    vs == cs
                })
            })
            .collect();
        let npure = pure_cells.len();
        let include_freevars = code.flags.contains(CodeFlags::OPTIMIZED) && !skip_free_vars;
        let freevarnames_len = if include_freevars {
            npure + code.freevars.len()
        } else {
            npure
        };
        for i in 0..freevarnames_len {
            let name: &str = if i < npure {
                pure_cells[i].as_ref()
            } else {
                code.freevars[i - npure].as_ref()
            };
            let idx = numlocals + i;
            if idx < self.locals_w().len() {
                let w_value = finditem_str_object(w_locals, name)?.unwrap_or(PY_NULL);
                let slot = self.locals_w()[idx];
                if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_set(slot, w_value) };
                } else {
                    self.locals_w_mut()[idx] = w_value;
                }
            }
        }
        Ok(())
    }

    /// pyframe.py:640-651 init_cells.
    ///
    /// In the CPython 3.11+ unified slot layout that pyre adopted via
    /// `npure_cellvars`, every cellvar that also appears in varnames
    /// shares its varname slot — and the bytecode walker emits
    /// `MAKE_CELL i` to wrap that local into a cell at function entry
    /// (see `RustPython` 0.5 compiler `bytecode/instruction.rs:233`).
    /// That removes the need for the legacy `_compute_args_as_cellvars`
    /// arg-to-cell copy: there is no separate cellvar slot to copy
    /// into.  Pure cellvars (cellvars not in varnames) are pre-
    /// initialised as empty cells in `new_for_call_with_closure`.  This
    /// function therefore has nothing to do; keep the entry point so
    /// other call sites that already invoke it remain valid.
    #[inline]
    pub fn init_cells(&mut self) {}

    /// pyframe.py:554-598 fast2locals — copy the fastlocals into the locals
    /// mapping via `space.setitem_str` (`pyframe.py:568`), using `space.delitem`
    /// for missing slots (`pyframe.py:571-574`; `delitem`'s `KeyError` is
    /// silently dropped).  A function frame with no locals bound yet lazily
    /// allocates a fresh dict (pyframe.py:557 `self.space.newdict(instance=True)`)
    /// and caches it, so `locals() is locals()` holds.  Errors propagate.
    pub fn fast2locals(&mut self) -> Result<(), crate::PyError> {
        let w_locals = self.get_or_create_w_locals();
        let code_ptr = unsafe { pyframe_get_pycode(self) };
        let code = unsafe { &*code_ptr };
        let varnames = &code.varnames;
        let numlocals = varnames.len();

        for i in 0..numlocals {
            // CO_FAST_HIDDEN slots are not user-visible — see fast2locals.
            if hidden_local(code, i) {
                continue;
            }
            let name = &varnames[i];
            let w_value = self.locals_w()[i];
            if !w_value.is_null() {
                setitem_str_object(w_locals, name, w_value)?;
            } else {
                delitem_str_object(w_locals, name)?;
            }
        }

        let pure_cells: Vec<&_> = code
            .cellvars
            .iter()
            .filter(|c| {
                let cs: &str = c.as_ref();
                !varnames.iter().any(|v| {
                    let vs: &str = v.as_ref();
                    vs == cs
                })
            })
            .collect();
        let npure = pure_cells.len();
        let include_freevars = code.flags.contains(CodeFlags::OPTIMIZED);
        let freevarnames_len = if include_freevars {
            npure + code.freevars.len()
        } else {
            npure
        };
        for i in 0..freevarnames_len {
            let name: &str = if i < npure {
                pure_cells[i].as_ref()
            } else {
                code.freevars[i - npure].as_ref()
            };
            let idx = numlocals + i;
            if idx < self.locals_w().len() {
                let slot = self.locals_w()[idx];
                let w_value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
                    unsafe { pyre_object::w_cell_get(slot) }
                } else {
                    slot
                };
                if !w_value.is_null() {
                    setitem_str_object(w_locals, name, w_value)?;
                } else if code.flags.contains(CodeFlags::OPTIMIZED) {
                    // Optimized (function) frames own their cellvars in the
                    // cell, so an empty cell means the local is unbound and its
                    // locals entry must be removed.  Module/class frames are
                    // namespace-authoritative: a cellvar can hold its binding in
                    // `w_locals` via STORE_NAME while its cell stays empty
                    // (`__conditional_annotations__`), so an empty cell there
                    // must not erase the STORE_NAME binding.
                    delitem_str_object(w_locals, name)?;
                }
            }
        }
        Ok(())
    }

    /// pyframe.py:208-218 `make_arguments` — build Arguments from the value
    /// stack. `methodcall` and `w_function` are stored on the resulting
    /// Arguments for diagnostic use (better error messages on argument
    /// mismatch); pyre's call path currently passes only the positional
    /// vector, so the parameters are recorded as part of the parity contract
    /// without downstream consumers yet.
    #[inline]
    pub fn make_arguments(
        &self,
        nargs: usize,
        _methodcall: bool,
        _w_function: PyObjectRef,
    ) -> Vec<PyObjectRef> {
        self.peekvalues(nargs)
    }

    /// PyPy-compatible argument list builder.
    #[inline]
    #[allow(unused_variables)]
    pub fn argument_factory(
        &self,
        _arguments: &[PyObjectRef],
        _keywords: &[PyObjectRef],
        _keywords_w: &[PyObjectRef],
        _w_star: PyObjectRef,
        _w_starstar: PyObjectRef,
        _methodcall: bool,
    ) -> Vec<PyObjectRef> {
        let mut args = Vec::new();
        args.extend_from_slice(_arguments);
        args.extend_from_slice(_keywords);
        args.extend_from_slice(_keywords_w);
        if !_w_star.is_null() {
            args.push(_w_star);
        }
        if !_w_starstar.is_null() {
            args.push(_w_starstar);
        }
        args
    }

    /// Create a new frame for a function call.
    ///
    /// The `globals` pointer is shared from the function object -- no clone.
    /// The `code` pointer is shared from the function object -- no clone.
    /// `closure` is a tuple of cell objects from the enclosing scope,
    /// or PY_NULL if the function has no free variables.
    pub fn new_for_call(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure(code, args, globals, execution_context, PY_NULL)
    }

    /// JIT warm-entry frame builder. The only callers are the `call_jit.rs`
    /// residual-frame helpers (`create_callee_frame_impl*`,
    /// `jit_create_self_recursive_callee_frame_1`, `jit_force_callee_frame`),
    /// which run from compiled code through an `i64`-pointer ABI with no error
    /// channel, so this stays infallible. PRE-EXISTING-ADAPTATION: a compiled
    /// trace sets the virtualizable frame fields directly (pyjitpl frame
    /// setup) instead of re-running `__init__` as a residual call, so there is
    /// no fallible `createframe` residual to mirror and `pick_builtin` cannot
    /// newly raise here — `globals` carries an `__builtins__` already validated
    /// when the interpreter first built the frame.
    pub fn new_for_call_with_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure_and_globals_obj(
            code,
            args,
            globals,
            w_globals,
            execution_context,
            PY_NULL,
            FrameLocalsArrayAllocation::StdAlloc,
        )
    }

    /// Create a new frame for a function call with a closure.
    pub fn new_for_call_with_closure(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
    ) -> Self {
        let w_globals = if globals.is_null() {
            PY_NULL
        } else {
            crate::baseobjspace::dict_storage_to_dict(globals)
        };
        Self::new_for_call_with_closure_and_globals_obj(
            code,
            args,
            globals,
            w_globals,
            execution_context,
            closure,
            FrameLocalsArrayAllocation::StdAlloc,
        )
    }

    /// pyframe.py:114 `Frame.__init__` resolves `self.builtin` through
    /// `pick_builtin(w_globals)`, which raises a non-KeyError
    /// `OperationError` straight out of `__init__`.  Fallible frame builder
    /// mirroring that path.
    pub fn try_new_for_call_with_closure_and_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
        allocation: FrameLocalsArrayAllocation,
    ) -> Result<Self, crate::PyError> {
        let w_globals = if w_globals.is_null() && !globals.is_null() {
            crate::baseobjspace::dict_storage_to_dict(globals)
        } else {
            w_globals
        };
        let w_builtin =
            crate::baseobjspace::frame_builtin_obj_checked(w_globals, execution_context)?;
        Ok(Self::finish_for_call_with_globals_obj(
            code,
            args,
            w_globals,
            execution_context,
            closure,
            w_builtin,
            allocation,
        ))
    }

    /// PyPy `space.createframe(code, function.w_func_globals, function)`.
    /// The semantic globals carrier is the dict object.  `globals` is kept
    /// only as pyre's legacy raw-storage side channel for code paths that
    /// have not yet migrated off `PyFrame.w_globals`.
    ///
    /// Infallible sibling of `try_new_for_call_with_closure_and_globals_obj`.
    /// All interpreter and tracer entry points use the `try_` variant; this
    /// one survives only for the JIT warm-entry path via
    /// `new_for_call_with_globals_obj` (see that method for why fallibility has
    /// no upstream counterpart there). `pick_builtin_obj` drops a non-KeyError
    /// `__builtins__` lookup, which is unreachable on that path.
    pub fn new_for_call_with_closure_and_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut DictStorage,
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
        allocation: FrameLocalsArrayAllocation,
    ) -> Self {
        let w_globals = if w_globals.is_null() && !globals.is_null() {
            crate::baseobjspace::dict_storage_to_dict(globals)
        } else {
            w_globals
        };
        let w_builtin = crate::baseobjspace::frame_builtin_obj(w_globals, execution_context);
        Self::finish_for_call_with_globals_obj(
            code,
            args,
            w_globals,
            execution_context,
            closure,
            w_builtin,
            allocation,
        )
    }

    /// Common tail of the frame builders: everything in `Frame.__init__`
    /// except the `pick_builtin` resolution, which is lifted to the callers
    /// so the fallible variant can propagate its error.
    fn finish_for_call_with_globals_obj(
        code: *const (),
        args: &[PyObjectRef],
        w_globals: PyObjectRef,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
        w_builtin: PyObjectRef,
        allocation: FrameLocalsArrayAllocation,
    ) -> Self {
        let code_ref = unsafe {
            &*(crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject)
        };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        let locals_cells_stack_w = unsafe {
            alloc_frame_locals_array(num_locals + num_cells + max_stack, PY_NULL, allocation)
        };

        {
            // Populate the freshly-allocated array via its mutable slice.
            let arr = unsafe { &mut *locals_cells_stack_w };

            // Bind positional arguments directly -- no intermediate Vec.
            let nargs = args.len().min(num_locals);
            for i in 0..nargs {
                arr[i] = args[i];
            }

            // CPython 3.11+ `co_localsplusnames` unified slot layout:
            // each cellvar that ALSO appears in varnames shares its
            // varname slot (MAKE_CELL wraps the local). Only cellvars
            // NOT in varnames take a fresh slot in the cell region.
            // Allocating cells for the overlap would shift freevar
            // indices and break LOAD_DEREF on `def repeat(n): def
            // wrap(fn): def inner(): return (n, fn)` style closures.
            let npure = npure_cellvars(code_ref);
            for i in 0..npure {
                arr[num_locals + i] = pyre_object::w_cell_new(PY_NULL);
            }
            if !closure.is_null() {
                let nfreevars = code_ref.freevars.len();
                for i in 0..nfreevars {
                    let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap() };
                    arr[num_locals + npure + i] = cell;
                }
            }
        }

        // Stable frame-locals arrays are filled before their owning frame is
        // published. `w_cell_new` uses the non-collecting old-gen allocator;
        // remember the completed array before the next allocating operation.
        remember_frame_locals_array(locals_cells_stack_w);

        // pyframe.py:103 — stamp `pycode.w_globals`; side effect only (the
        // gated debugdata snapshot retired in favour of `w_globals`).
        unsafe {
            crate::w_code_frame_stores_global(code as PyObjectRef, w_globals);
        }

        let mut frame = PyFrame {
            ob_header: frame_ob_header(),
            execution_context,
            pycode: code,
            locals_cells_stack_w,
            valuestackdepth: num_locals + num_cells,
            last_instr: -1,
            escaped: false,
            debugdata: std::ptr::null_mut(),
            lastblock: std::ptr::null_mut(),
            vable_token: 0,
            frame_finished_execution: false,
            f_generator_nowref: PY_NULL,
            w_yielding_from: PY_NULL,
            f_backref: std::ptr::null_mut(),
            w_builtin,
            w_globals,
        };
        frame.init_cells();
        frame
    }

    /// Borrow the raw CodeObject.
    #[inline]
    pub fn code(&self) -> &CodeObject {
        unsafe { &*pyframe_get_pycode(self) }
    }

    #[inline]
    pub fn _is_generator_or_coroutine(&self) -> bool {
        code_flags_make_generator(self.code().flags)
    }

    /// pyframe.py:276 initialize_as_generator
    ///
    /// Adaptation: pyre builds the caller's PyFrame on the interpreter stack,
    /// so we snapshot it onto the heap before handing ownership to the
    /// generator object. The backref (`f_generator_nowref`) is set on that
    /// heap-owned snapshot — not on the temporary caller frame — so later
    /// `get_generator()` calls through the surviving frame pointer return
    /// the right object.
    #[inline]
    pub fn initialize_as_generator(&mut self) -> crate::PyResult {
        // pyframe.py:259 wraps `self` directly. A borrowed `&mut self` cannot
        // hand ownership to the generator, so snapshot into an owned FrameBox
        // first. Callers that already own a FrameBox should use
        // `FrameBox::into_generator` to skip this copy.  Use the GC-managed
        // snapshot: the generator owns the frame for its whole life, off the
        // `CURRENT_FRAME` chain, so its lifetime is GC reachability.
        self.snapshot_for_generator().into_generator()
    }

    #[inline]
    pub fn get_generator(&self) -> PyObjectRef {
        self.f_generator_nowref
    }

    /// Returns RPython's `last_instr` (= `next_instr`) value, derived from
    /// pyre's CPython-aligned storage. See the field comment on `last_instr`
    /// for the convention difference. PyPy `pyopcode.py:172` would read this
    /// field directly; pyre derives it because storage matches `f_lasti`.
    #[inline]
    pub fn next_instr(&self) -> usize {
        if self.last_instr < 0 {
            0
        } else {
            self.last_instr as usize + 1
        }
    }

    /// Inverse of `next_instr()`. Stores `next_instr - 1` so the field
    /// continues to satisfy `f_lasti = last_instr` (CPython convention).
    /// RPython would write `next_instr` directly here.
    #[inline]
    pub fn set_last_instr_from_next_instr(&mut self, next_instr: usize) {
        self.last_instr = next_instr as isize - 1;
    }

    /// Repoint internal array pointers after a struct move.
    ///
    /// `PyObjectArray` with small-buffer optimization stores an inline
    /// buffer whose address changes on move. Call this once after the
    /// frame is at its final stack location.
    #[inline]
    pub fn fix_array_ptrs(&mut self) {
        // locals_cells_stack_w is now a heap-allocated pointer;
        // fix_ptr on PyObjectArray is a no-op.
    }

    /// Load a constant from the code object by raw index.
    /// Used by the blackhole interpreter's bh_load_const_fn.
    pub fn load_const_pyobj(&self, idx: usize) -> PyObjectRef {
        let code = self.code();
        // RPython: constants are in JitCode.constants_r. In pyre, we resolve
        // from the CodeObject's constant table at runtime.
        let constants = code_constants(code);
        if idx >= constants.len() {
            return pyre_object::w_none();
        }
        pyobject_from_constant(&constants[idx])
    }
}

/// Load a constant from a CodeObject without a PyFrame.
/// Used by the blackhole's bh_load_const_fn when the code pointer
/// comes from a virtualizable field read.
pub fn load_const_from_code(code: &CodeObject, idx: usize) -> PyObjectRef {
    let constants = code_constants(code);
    if idx >= constants.len() {
        return pyre_object::w_none();
    }
    pyobject_from_constant(&constants[idx])
}

/// `co_names[idx]` — the global/attribute name at `idx`, used by the JIT
/// full-body walker to resolve a `LOAD_GLOBAL` namei to the name string for
/// the module-dict cell-cache lookup.  `None` when `idx` is out of range.
pub fn load_name_from_code(code: &CodeObject, idx: usize) -> Option<&str> {
    code.names.get(idx).map(|s| s.as_str())
}

pub(crate) fn code_constants(code: &CodeObject) -> &[crate::bytecode::ConstantData] {
    unsafe {
        std::slice::from_raw_parts(
            code.constants.as_ptr() as *const crate::bytecode::ConstantData,
            code.constants.len(),
        )
    }
}

/// Materialise a single `ConstantData` into a `PyObjectRef`.
///
/// Line-by-line port of the `ConstantOpcodeHandler for PyFrame` impl
/// (`eval.rs:1300-1352`) routed through `load_const_value`
/// (`pyopcode.rs:343-394`). The blackhole's `bh_load_const_fn` lacks a
/// `&mut PyFrame` to dispatch through the trait, so this free function
/// mirrors each `*_constant` body directly. Variant order matches
/// `pyopcode.rs::load_const_value` so future additions stay in sync.
fn pyobject_from_constant(constant: &crate::bytecode::ConstantData) -> PyObjectRef {
    use crate::bytecode::ConstantData;
    use num_traits::ToPrimitive;
    match constant {
        // `pyopcode.rs:347-353` — promote bigints to W_LongObject just
        // like `load_const_value` does before invoking the trait.
        ConstantData::Integer { value } => {
            if pyre_object::longobject::jit_bigint_to_i64_fits(value) != 0 {
                pyre_object::intobject::w_int_new(pyre_object::longobject::jit_bigint_to_i64_value(
                    value,
                ))
            } else {
                pyre_object::longobject::w_long_new(value.clone())
            }
        }
        // `eval.rs:1309-1311 float_constant`.
        ConstantData::Float { value } => pyre_object::floatobject::w_float_new(*value),
        // `eval.rs:1313-1315 bool_constant` — bools must surface as
        // W_BoolObject (`is space.w_True/w_False`), not W_IntObject.
        ConstantData::Boolean { value } => pyre_object::w_bool_from(*value),
        // `eval.rs:1317-1319 str_constant` — `box_str_constant` interns
        // matching `space.newtext` per `unicodeobject.py wrapunicode`.
        ConstantData::Str { value } => pyre_object::unicodeobject::box_str_constant(value),
        // `eval.rs:1321-1323 bytes_constant`.
        ConstantData::Bytes { value } => pyre_object::bytesobject::w_bytes_from_bytes(value),
        // Reached only for a code constant nested inside a container constant;
        // top-level `LOAD_CONST` routes through `co_consts_w` in `bh_load_const_fn`
        // so the blackhole shares the interpreter's wrapper.
        ConstantData::Code { code } => crate::pycode::box_code_constant(code),
        // `eval.rs:1333-1335 none_constant`.
        ConstantData::None => pyre_object::w_none(),
        // `eval.rs:1337-1339 ellipsis_constant`.
        ConstantData::Ellipsis => pyre_object::special::w_ellipsis(),
        // `pyopcode.rs:360-366` — recurse + delegate to the default
        // `build_tuple` body (`eval.rs:767 build_tuple_from_refs`).
        ConstantData::Tuple { elements } => {
            let items: Vec<PyObjectRef> = elements.iter().map(pyobject_from_constant).collect();
            crate::runtime_ops::build_tuple_from_refs(&items)
        }
        // `pyopcode.rs:382-393` — recurse over `[start, stop, step]`
        // before invoking `slice_constant` (`eval.rs:1341-1348`).
        ConstantData::Slice { elements } => {
            let start = pyobject_from_constant(&elements[0]);
            let stop = pyobject_from_constant(&elements[1]);
            let step = pyobject_from_constant(&elements[2]);
            pyre_object::w_slice_new(start, stop, step)
        }
        // `pyopcode.rs:375-381` — recurse + delegate to
        // `frozenset_constant` (`eval.rs:1350-1352`).
        ConstantData::Frozenset { elements } => {
            let items: Vec<PyObjectRef> = elements.iter().map(pyobject_from_constant).collect();
            pyre_object::w_frozenset_from_items(&items)
        }
        ConstantData::Complex { value } => {
            pyre_object::complexobject::w_complex_new(value.re, value.im)
        }
    }
}

// Virtualizable configuration is in jit/frame_layout.rs

/// pypy/interpreter/baseobjspace.py:796-798 `createframe`.
///
/// ```python
/// def createframe(self, code, w_globals, outer_func=None):
///     "Create an empty PyFrame suitable for this code object."
///     return self.FrameClass(self, code, w_globals, outer_func)
/// ```
///
/// Returns `Box<PyFrame>` matching PyPy's heap-allocated PyFrame (RPython
/// class instance — `pyframe.py:51 class PyFrame(W_Root)`).  The Box keeps
/// every `PyFrame` heap-resident.
///
/// The body inlines `pyframe.py:98-119 PyFrame.__init__` line-by-line:
/// allocate `locals_cells_stack_w` of size `nlocals + ncellvars + nfreevars
/// + stacksize`, set `valuestackdepth`, optionally bind debug `w_globals`
/// when `code.frame_stores_global(w_globals)`, then call
/// `self.initialize_frame_scopes(outer_func, code)` (`pyframe.py:223`)
/// which performs cell init, freevar copy from `outer_func.closure`, and
/// raises on freevar/closure-size mismatch.  No constructor switch — both
/// branches share the same allocation + scope-init path so the cell/freevar
/// invariants of `pyframe.py:223-261` hold uniformly.
///
/// `outer_func` carries the closure-providing function reference per PyPy
/// (`function.py:126-127, 208-209, 219-220`).  `None` for module / exec /
/// REPL frames where freevars must be empty (PyPy raises TypeError
/// "directly executed code object may not contain free variables" in
/// `pyframe.py:242-246`).  `Some(func)` for function calls AND class-body
/// execution (`pypy/module/__builtin__/compiling.py:208`), where the
/// function's closure tuple seeds the freevar slots via
/// `function_get_closure`.  Pyre adds `execution_context` and `w_globals`
/// as explicit parameters because pyre lacks PyPy's `space` implicit
/// carrier.
///
/// **Args binding** (positional argument values into
/// `locals_cells_stack_w[0..nargs]`) is **caller-side** per PyPy
/// `pycode.py:241-249 funcrun`: caller invokes
/// `space.createframe(...) → args.parse_into_scope(...) → frame.init_cells()
/// → frame.run(...)`.  createframe itself never binds args and never calls
/// `init_cells()`.
///
/// # Safety
/// `code`, `w_globals`, and `execution_context` must be valid pointers
/// for the duration the returned `Box<PyFrame>` is alive.  `outer_func`,
/// when `Some`, must be a valid Function `PyObjectRef`.
pub fn createframe(
    code: *const (),
    w_globals: *mut DictStorage,
    execution_context: *const PyExecutionContext,
    outer_func: Option<PyObjectRef>,
) -> Result<FrameBox, crate::PyError> {
    // Legacy raw-storage entry: resolve the canonical W_DictObject sibling
    // and delegate to `createframe_obj`.  `dict_storage_to_dict` and
    // `w_dict_get_dict_storage_proxy` are mutual inverses under the
    // `mirror_target` invariant, so the round-trip preserves identity.
    let w_globals = if w_globals.is_null() {
        PY_NULL
    } else {
        crate::baseobjspace::dict_storage_to_dict(w_globals)
    };
    createframe_obj(code, w_globals, execution_context, outer_func)
}

/// `baseobjspace.py:796 createframe` with the globals passed as the dict
/// OBJECT (`pyframe.py:49 self.w_globals = w_globals` stores the object).
pub fn createframe_obj(
    code: *const (),
    w_globals: PyObjectRef,
    execution_context: *const PyExecutionContext,
    outer_func: Option<PyObjectRef>,
) -> Result<FrameBox, crate::PyError> {
    // pyframe.py:98-119 PyFrame.__init__ — line-by-line.
    //   self.space = space               (pyre: implicit, no field)
    //   self.pycode = code               (pycode field below)
    //   if code.frame_stores_global(w_globals):
    //       self.getorcreatedebug().w_globals = w_globals
    //   ncellvars = len(code.co_cellvars)
    //   nfreevars = len(code.co_freevars)
    //   size = code.co_nlocals + ncellvars + nfreevars + code.co_stacksize
    //   self.locals_cells_stack_w = [None] * size
    //   self.valuestackdepth = code.co_nlocals + ncellvars + nfreevars
    //   ...
    //   self.initialize_frame_scopes(outer_func, code)
    //
    // Normalize a dict-subclass globals (`exec(src, G())`) to its inner
    // `__dict_data__` dict.  The storage proxy that LOAD_GLOBAL reads lives
    // on the `W_DictObject`, and downstream readers key off this object —
    // `get_w_globals_storage`, MAKE_FUNCTION's `function.__globals__`, and the
    // JIT inline / callee globals readers — so the frame must hold the
    // `W_DictObject`, not the subclass instance whose layout has no proxy
    // slot.  No-op for a plain dict / module dict (`resolve_dict_backing`
    // returns the argument).
    let w_globals = if w_globals.is_null() {
        w_globals
    } else {
        let backing = crate::type_methods::resolve_dict_backing(w_globals);
        if backing.is_null() {
            w_globals
        } else {
            backing
        }
    };
    let raw = unsafe { crate::w_code_get_ptr(code as PyObjectRef) as *const CodeObject };
    let code_ref = unsafe { &*raw };
    let num_locals = code_ref.varnames.len();
    let num_cells = ncells(code_ref);
    let max_stack = code_ref.max_stackdepth as usize;
    // pyframe.py:103 — stamp `pycode.w_globals`; side effect only (the gated
    // debugdata snapshot retired in favour of `w_globals`).
    unsafe {
        crate::w_code_frame_stores_global(code as PyObjectRef, w_globals);
    }

    let size = num_locals + num_cells + max_stack;
    let w_builtin = crate::baseobjspace::frame_builtin_obj(w_globals, execution_context);
    let mut frame = FrameBox::new(PyFrame {
        ob_header: frame_ob_header(),
        execution_context,
        pycode: code,
        locals_cells_stack_w: unsafe {
            alloc_frame_locals_array(size, PY_NULL, FrameLocalsArrayAllocation::OldGenGc)
        },
        valuestackdepth: num_locals + num_cells,
        last_instr: -1,
        escaped: false,
        debugdata: std::ptr::null_mut(),
        lastblock: std::ptr::null_mut(),
        vable_token: 0,
        frame_finished_execution: false,
        f_generator_nowref: PY_NULL,
        w_yielding_from: PY_NULL,
        f_backref: std::ptr::null_mut(),
        w_builtin,
        w_globals,
    });
    // pyframe.py:119 — final step of __init__.  PY_NULL plays the role of
    // Python `None` per the existing `initialize_frame_scopes` convention
    // (pyframe.rs:664).  Top-level module / interactive / expression code
    // arrives here without CO_NEWLOCALS — RustPython codegen emits empty
    // flags for the seed CodeInfo (`crates/codegen/src/compile.rs Compiler::new`)
    // so initialize_frame_scopes selects the `!OPTIMIZED && !NEWLOCALS`
    // arm and binds `w_locals = w_globals` per pyframe.py:233-235.
    let outer_ref = outer_func.unwrap_or(PY_NULL);
    // initialize_frame_scopes allocates (the locals dict and `w_cell_new`
    // cells) and stores the cells into `locals_cells_stack_w`; root the slot so
    // a collection mid-init can't drop the cells already written there.
    {
        let _root = FrameLocalsRoot::new(frame.as_mut_ptr());
        frame.initialize_frame_scopes(outer_ref, code)?;
    }
    remember_frame_locals_array(frame.locals_cells_stack_w);

    Ok(frame)
}

/// `space.finditem_str(w_obj, key)` — `space.getitem` with KeyError
/// remapped to `None`.  Non-KeyError errors propagate unchanged so
/// `fast2locals`/`locals2fast` raise as PyPy does at `pyframe.py:613` /
/// `pyframe.py:632` (`pypy/objspace/std/objspace.py finditem_str` re-
/// raises everything except `KeyError`).
fn finditem_str_object(
    w_obj: PyObjectRef,
    name: &str,
) -> Result<Option<PyObjectRef>, crate::PyError> {
    let key = unsafe { pyre_object::w_str_new(name) };
    match crate::baseobjspace::getitem(w_obj, key) {
        Ok(v) if !v.is_null() => Ok(Some(v)),
        Ok(_) => Ok(None),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(None),
        Err(e) => Err(e),
    }
}

/// `space.setitem(w_obj, w_str_new(name), value)` — every error
/// propagates (PyPy `pyframe.py:568` `space.setitem_str` does not
/// swallow exceptions).
fn setitem_str_object(
    w_obj: PyObjectRef,
    name: &str,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    let key = unsafe { pyre_object::w_str_new(name) };
    crate::baseobjspace::setitem(w_obj, key, value).map(|_| ())
}

/// `space.delitem(w_obj, w_str_new(name))` — `pyframe.py:571-574 /
/// 589-593` ignores `KeyError` only; other errors propagate.
fn delitem_str_object(w_obj: PyObjectRef, name: &str) -> Result<(), crate::PyError> {
    let key = unsafe { pyre_object::w_str_new(name) };
    match crate::baseobjspace::delitem(w_obj, key) {
        Ok(_) => Ok(()),
        Err(e) if e.kind == crate::PyErrorKind::KeyError => Ok(()),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::load_const_from_code;

    #[test]
    fn load_const_from_code_returns_ellipsis_singleton() {
        let code = crate::compile_eval("...");
        let code = code.expect("compile should succeed");
        let ellipsis_index = super::code_constants(&code)
            .iter()
            .position(|constant| matches!(constant, crate::bytecode::ConstantData::Ellipsis))
            .expect("compiled code should contain an Ellipsis constant");

        let loaded = load_const_from_code(&code, ellipsis_index);
        assert_eq!(loaded, pyre_object::special::w_ellipsis());
    }

    // ── mark_stacks (f_lineno jump validation) ──

    use super::{
        MARK_EMPTY_STACK, MARK_UNINITIALIZED, StackKind, mark_compatible_stack,
        mark_first_line_not_before, mark_lines, mark_stacks,
    };

    #[test]
    fn mark_stacks_entry_is_empty_and_reachable() {
        let code = crate::compile_exec("x = 1\ny = x + 1\n").expect("compile");
        let len = code.instructions.len();
        let stacks = mark_stacks(&code, len);
        // Entry to the first instruction is the empty stack.
        assert_eq!(stacks[0], MARK_EMPTY_STACK);
        // Every real (decodable, non-cache) instruction is reachable in
        // straight-line code — none stay UNINITIALIZED.
        for pc in 0..len {
            if let Some((instr, _)) = crate::pyopcode::decode_instruction_at(&code, pc) {
                if matches!(instr, crate::bytecode::Instruction::Cache) {
                    continue;
                }
                assert_ne!(
                    stacks[pc], MARK_UNINITIALIZED,
                    "pc {pc} ({instr:?}) should be reachable"
                );
            }
        }
    }

    #[test]
    fn mark_stacks_for_iter_target_carries_iterator() {
        // The FOR_ITER body sees the iterator underneath the loop var, so
        // the abstract stack at the body has an Iterator slot at bottom.
        let code = crate::compile_exec("t = 0\nfor i in range(3):\n    t += i\n").expect("compile");
        let len = code.instructions.len();
        let stacks = mark_stacks(&code, len);
        // Find a GET_ITER; the slot it produces is Iterator and stays on
        // the stack through the loop body.
        let mut saw_iterator_slot = false;
        for pc in 0..len {
            if stacks[pc] > MARK_EMPTY_STACK
                && (stacks[pc] & 0b111 == StackKind::Iterator as i64
                    || (stacks[pc] >> 3) & 0b111 == StackKind::Iterator as i64)
            {
                saw_iterator_slot = true;
                break;
            }
        }
        assert!(
            saw_iterator_slot,
            "a for-loop's abstract stacks should contain an Iterator slot"
        );
    }

    #[test]
    fn mark_lines_and_first_line_not_before() {
        let code = crate::compile_exec("a = 1\nb = 2\nc = 3\n").expect("compile");
        let len = code.instructions.len();
        let lines = mark_lines(&code, len);
        let first = code.first_line_number.map(|n| n.get() as i32).unwrap_or(1);
        // The earliest recorded line is the first source line.
        let min_line = lines.iter().copied().filter(|&l| l >= 0).min().unwrap();
        assert_eq!(min_line, first);
        // A request before the code resolves to the first line.
        assert_eq!(mark_first_line_not_before(&lines, first), first);
        // A request past the end resolves to -1.
        assert_eq!(mark_first_line_not_before(&lines, first + 1000), -1);
    }

    #[test]
    fn compatible_stack_same_and_incompatible() {
        // Identical stacks are compatible.
        assert!(mark_compatible_stack(MARK_EMPTY_STACK, MARK_EMPTY_STACK));
        // An Object target accepts an Iterator source popped to depth, but
        // an Iterator target rejects an Object source (can't fabricate a
        // loop iterator).
        let obj = super::mark_push_value(MARK_EMPTY_STACK, StackKind::Object);
        let iter = super::mark_push_value(MARK_EMPTY_STACK, StackKind::Iterator);
        assert!(
            mark_compatible_stack(iter, obj),
            "Object target accepts any non-Null"
        );
        assert!(
            !mark_compatible_stack(obj, iter),
            "Iterator target rejects Object source"
        );
    }
}
