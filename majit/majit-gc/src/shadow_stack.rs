/// Shadow stack for GC root tracking in compiled JIT code.
///
/// RPython reference: rpython/jit/backend/llsupport/gc.py GcRootMap_shadowstack
///
/// Two stacks:
/// 1. GcRef shadow stack — individual GC refs (legacy, for non-jitframe roots)
/// 2. JitFrame shadow stack — jitframe pointers (RPython _call_header_shadowstack)
///
/// Protocol for jitframe shadow stack (assembler.py:1122-1136):
///   Entry: inline MOVs push [is_minor=1, jf_ptr] to root stack
///   Per-call: push_gcmap writes jf_gcmap; pop_gcmap clears it
///   GC: walk_jf_roots → read jf_gcmap → trace ref slots
///   Exit: pop_jf_to(depth)   — _call_footer_shadowstack
///
/// The jitframe shadow stack uses a per-thread flat memory array with a
/// root_stack_top pointer, matching RPython's per-thread ShadowStackPool.
/// Compiled code manipulates the current thread's root_stack_top with inline
/// load/store instructions (no function calls), exactly as in
/// assembler.py:1122-1136.
use std::cell::{Cell, RefCell};
use std::sync::{Mutex, OnceLock, RwLock};

use majit_ir::GcRef;

/// shadowstack.py:281 default root_stack_depth. RPython allocates
/// `163840 * sizeofaddr` bytes for the initial shadow stack pool.
const DEFAULT_SHADOW_STACK_DEPTH: usize = 163840;

/// WORD size (bytes). RPython: arch.py WORD.
const WORD: usize = std::mem::size_of::<usize>();

// ── Heap-allocated growable jitframe shadow stack (ShadowStackPool parity) ─
//
// RPython's ShadowStackPool starts with `DEFAULT_ROOT_STACK_DEPTH`
// entries and grows via `increase_root_stack_depth` which
// `raw_malloc`s a larger buffer and copies the used portion
// (rpython/memory/gctransform/shadowstack.py:351). Pyre mirrors this
// with one heap-allocated `Box<[usize]>` per OS thread. Compiled JIT code
// reads/writes that thread's `root_stack_top` cell (a stable TLS address)
// and dereferences it just like RPython.
//
// Each entry is [is_minor_marker: WORD, jf_ptr: WORD] = 2*WORD bytes.
// root_stack_top points past the last entry (next free slot).
//
// assembler.py:1122-1128 _call_header_shadowstack:
//   MOV ebx, [root_stack_top_addr]    // load top pointer
//   MOV [ebx], 1                      // is_minor marker
//   MOV [ebx + WORD], ebp             // jf_ptr
//   ADD ebx, 2*WORD                   // advance
//   MOV [root_stack_top_addr], ebx    // store new top
//
// assembler.py:1130-1136 _call_footer_shadowstack:
//   SUB [root_stack_top_addr], 2*WORD  // decrement top

struct JitFrameShadowStack {
    /// Base pointer of the currently allocated backing buffer. 0 until
    /// first use. The buffer is owned by `owner`, so this address remains
    /// valid until the next grow.
    base: usize,
    /// Current top pointer. Compiled code embeds the address of this cell and
    /// mutates its inner usize directly with inline loads/stores.
    top: Cell<usize>,
    /// One-past-the-end address of the current backing buffer. Compiled code
    /// embeds this cell's address, never its value: `grow()` replaces `owner`.
    limit: Cell<usize>,
    /// Current capacity in entries (each entry is two usize words).
    capacity: usize,
    owner: Option<Box<[usize]>>,
}

impl JitFrameShadowStack {
    fn new() -> Self {
        Self {
            base: 0,
            top: Cell::new(0),
            limit: Cell::new(0),
            capacity: 0,
            owner: None,
        }
    }

    fn ensure_init(&mut self) {
        if self.base == 0 {
            self.grow(DEFAULT_SHADOW_STACK_DEPTH);
        }
    }

    /// rpython/memory/gctransform/shadowstack.py:351
    /// `increase_root_stack_depth` parity: allocate new, copy used
    /// portion, update pointers, free old. RPython also handles all live
    /// thread shadow stacks; this Rust port keeps the same shape by giving
    /// each OS thread its own backing buffer.
    fn grow(&mut self, new_capacity: usize) {
        if new_capacity <= self.capacity && self.owner.is_some() {
            return;
        }
        let used_bytes = self.top.get().saturating_sub(self.base);
        let mut new_buf: Box<[usize]> = vec![0usize; new_capacity * 2].into_boxed_slice();
        let new_ptr = new_buf.as_mut_ptr() as usize;

        if self.base != 0 && used_bytes > 0 {
            // SAFETY: `self.base` points into the previous buffer held by
            // `self.owner`; the destination is a fresh allocation. This grow
            // operation is only called from the owning thread.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.base as *const u8,
                    new_ptr as *mut u8,
                    used_bytes,
                );
            }
        }

        self.base = new_ptr;
        self.top.set(new_ptr + used_bytes);
        self.capacity = new_capacity;
        self.limit.set(new_ptr + new_capacity * 2 * WORD);
        self.owner = Some(new_buf);
    }

    fn top_addr(&self) -> usize {
        self.top.as_ptr() as usize
    }

    fn limit_addr(&self) -> usize {
        self.limit.as_ptr() as usize
    }
}

/// Callback type for tracing a libc-allocated jitframe's interior.
/// The jitframe lives in malloc memory (not nursery, not oldgen), so
/// `trace_and_update_object` can't reach it via `self.types` lookup.
/// The host crate registers a tracer that knows the JitFrame layout
/// and walks `jf_gcmap` bits to expose Ref slots.
///
/// The callback receives the jitframe payload address and a closure
/// that maps a nursery-pointing slot to its new forwarded address.
pub type LibcJitframeTracer = unsafe fn(obj_addr: usize, update: &mut dyn FnMut(*mut GcRef));

static LIBC_JF_TRACER: std::sync::OnceLock<LibcJitframeTracer> = std::sync::OnceLock::new();

/// Register the host's libc-jitframe tracer. Call once at GC init.
/// Subsequent calls are ignored (OnceLock semantics).
pub fn register_libc_jitframe_tracer(tracer: LibcJitframeTracer) {
    let _ = LIBC_JF_TRACER.set(tracer);
}

// Track which pointers refer to libc-allocated jitframes so the GC
// visitor can safely dispatch to the registered tracer. Without this
// set the visitor cannot tell a libc-alloc'd jitframe from an
// unrelated foreign pointer that happens to sit on the shadow stack.
static LIBC_JF_REGISTRY: OnceLock<RwLock<indexmap::IndexSet<usize>>> = OnceLock::new();

fn libc_jf_registry() -> &'static RwLock<indexmap::IndexSet<usize>> {
    LIBC_JF_REGISTRY.get_or_init(|| RwLock::new(indexmap::IndexSet::new()))
}

/// Register a libc-allocated jitframe payload address. Must be called
/// before pushing the jitframe onto the JF shadow stack.
pub fn register_libc_jitframe(addr: usize) {
    libc_jf_registry().write().unwrap().insert(addr);
}

/// Unregister a libc-allocated jitframe address. Call once the
/// jitframe memory is about to be freed.
pub fn unregister_libc_jitframe(addr: usize) {
    libc_jf_registry().write().unwrap().swap_remove(&addr);
}

/// Check whether an address was registered as a libc-allocated jitframe.
pub fn is_libc_jitframe(addr: usize) -> bool {
    libc_jf_registry().read().unwrap().contains(&addr)
}

/// Invoke the registered tracer if any. Returns true if tracer ran.
pub fn trace_libc_jitframe(obj_addr: usize, update: &mut dyn FnMut(*mut GcRef)) -> bool {
    if let Some(tracer) = LIBC_JF_TRACER.get() {
        unsafe { tracer(obj_addr, update) };
        true
    } else {
        false
    }
}

/// gc.py:255-257 get_root_stack_top_addr()
/// Returns the ADDRESS of the current thread's root_stack_top variable
/// (not its value). Compiled code uses this to emit inline loads/stores.
pub fn get_root_stack_top_addr() -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        stack.top_addr()
    })
}

/// Address of this thread's grow-synchronised root-stack limit cell.
///
/// Compiled code loads the value for every inline push. `grow()` updates it
/// after replacing the backing buffer, while the cell address itself remains
/// stable for the lifetime of the thread-local shadow-stack object.
pub fn get_root_stack_limit_addr() -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        stack.limit_addr()
    })
}

thread_local! {
    /// shadowstack.py:287 `root_stack_depth`. Growable via
    /// `increase_root_stack_depth`; can never shrink.
    static MAX_SHADOW_STACK_DEPTH: Cell<usize> = const { Cell::new(DEFAULT_SHADOW_STACK_DEPTH) };

    /// Thread-local shadow stack for individual GcRef roots.
    static SHADOW_STACK: RefCell<ShadowStack> = RefCell::new(ShadowStack::new());

    /// Thread-local flat jitframe root stack. Rust tests run multiple
    /// JIT/GC tests in parallel, so using one process-global root stack lets
    /// one GC walk another test's jitframe. RPython's root stack is per
    /// thread; mirror that here while preserving the inline root_stack_top
    /// protocol for compiled code.
    static JF_ROOT_STACK: RefCell<JitFrameShadowStack> =
        RefCell::new(JitFrameShadowStack::new());

    /// Thread-local stack of blackhole interpreter register banks.
    /// blackhole.py BlackholeInterpreter.registers_r parity: each active
    /// blackhole frame's ref register file is a GC root range. The GC must
    /// trace these slots during minor collection so nursery objects held only
    /// by a blackhole register survive across collecting calls.
    static BH_REGS_STACK: RefCell<Vec<BhRegsEntry>> = RefCell::new(Vec::with_capacity(16));

    /// Thread-local stack of ref slices live only during blackhole resume
    /// construction (`resume.py:1312 blackhole_from_resumedata`).  The
    /// `virtuals_cache` and each frame's `registers_r` are filled by lazily
    /// materializing virtuals (`getvirtual_ptr` → allocator); a minor
    /// collection triggered while materializing a later virtual relocates
    /// the earlier ones, but the unrooted raw `Vec` copies are not forwarded
    /// until `run()` re-roots `registers_r` via `push_bh_regs`.  RPython
    /// traces these through the GC-managed reader/blackhole objects; pyre
    /// stores raw `i64`, so the slices are registered here for the
    /// construction window and popped when it ends.
    static RESUME_REF_ROOTS_STACK: RefCell<Vec<(*mut i64, usize)>> =
        RefCell::new(Vec::with_capacity(16));
}

/// Root structures owned by one registered mutator thread.
///
/// The pointers remain valid until `unregister_mutator` runs on the owning
/// thread. Foreign access is only permitted while gc_sync has quiesced every
/// registered mutator, so none of the underlying TLS structures can move or
/// be mutably accessed by its owner during a walk.
struct MutatorEntry {
    thread_id: std::thread::ThreadId,
    shadow_stack: *const RefCell<ShadowStack>,
    jf_root_stack: *const RefCell<JitFrameShadowStack>,
    bh_regs_stack: *const RefCell<Vec<BhRegsEntry>>,
    resume_ref_roots_stack: *const RefCell<Vec<(*mut i64, usize)>>,
    extra_areas: Vec<MutatorExtraArea>,
}

/// Walker for one opaque root area owned by a registered mutator.
///
/// The callback runs on the collecting thread. It must derive every
/// thread-specific address from `data` and must not consult caller TLS.
pub type MutatorExtraWalkFn = unsafe fn(*const (), &mut dyn FnMut(&mut GcRef));

#[derive(Clone, Copy)]
struct MutatorExtraArea {
    walk: MutatorExtraWalkFn,
    data: *const (),
}

// The raw pointers refer to TLS owned by `thread_id`. The registry only moves
// pointer values between threads; dereferencing them requires the STW
// quiescence established by gc_sync.
unsafe impl Send for MutatorEntry {}

static MUTATOR_REGISTRY: Mutex<Vec<MutatorEntry>> = Mutex::new(Vec::new());

/// Register the current thread's four TLS root structures for STW root walks.
/// Unregistration is supplied by the caller's RAII destructor (pyre-jit's `GcMutatorRegistration` thread-local, armed in `init_gc_subsystem`, whose `Drop` calls [`unregister_mutator`]); callers must arm that pairing, while an API-level return guard is a tracked follow-up.
pub fn register_mutator() {
    let thread_id = std::thread::current().id();
    let shadow_stack = SHADOW_STACK.with(|stack| stack as *const _);
    let jf_root_stack = JF_ROOT_STACK.with(|stack| stack as *const _);
    let bh_regs_stack = BH_REGS_STACK.with(|stack| stack as *const _);
    let resume_ref_roots_stack = RESUME_REF_ROOTS_STACK.with(|stack| stack as *const _);

    let mut registry = MUTATOR_REGISTRY.lock().unwrap();
    assert!(
        !registry.iter().any(|entry| entry.thread_id == thread_id),
        "mutator thread registered twice"
    );
    registry.push(MutatorEntry {
        thread_id,
        shadow_stack,
        jf_root_stack,
        bh_regs_stack,
        resume_ref_roots_stack,
        extra_areas: Vec::new(),
    });
}

/// Append an opaque root area to the current registered mutator.
///
/// # Safety
///
/// The caller must keep `data` valid until [`unregister_mutator`] runs on this
/// thread. `walk` must derive every address it dereferences from `data`, never
/// from caller TLS, because a foreign collecting thread invokes `walk` during
/// STW.
pub unsafe fn register_mutator_extra_area(walk: MutatorExtraWalkFn, data: *const ()) {
    let thread_id = std::thread::current().id();
    let mut registry = MUTATOR_REGISTRY.lock().unwrap();
    let entry = registry
        .iter_mut()
        .find(|entry| entry.thread_id == thread_id)
        .expect("register_mutator_extra_area called before register_mutator");
    entry.extra_areas.push(MutatorExtraArea { walk, data });
}

/// Walk every registered mutator's opaque extra root areas during STW.
pub fn walk_all_extra_areas(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_extra_areas walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = MUTATOR_REGISTRY.lock().unwrap();
    for mutator in registry.iter() {
        for area in mutator.extra_areas.iter() {
            // SAFETY: gc_sync has quiesced every registered owner, and each
            // area remains valid until its MutatorEntry is removed.
            unsafe { (area.walk)(area.data, &mut visitor) };
        }
    }
}

/// Walk the current mutator's opaque extra root areas.
///
/// This is the single-thread collection path; callers without a registered
/// mutator have no per-thread areas and are a no-op.
pub fn walk_my_extra_areas(mut visitor: impl FnMut(&mut GcRef)) {
    let thread_id = std::thread::current().id();
    let registry = MUTATOR_REGISTRY.lock().unwrap();
    let Some(mutator) = registry.iter().find(|entry| entry.thread_id == thread_id) else {
        return;
    };
    for area in mutator.extra_areas.iter() {
        // SAFETY: this is the owning thread's synchronous collection path.
        unsafe { (area.walk)(area.data, &mut visitor) };
    }
}

/// Remove the current thread from the all-thread root registry.
///
/// Call this before gc_sync::unregister_thread so RUNNING cannot reach zero
/// while an entry whose TLS is being destroyed remains visible to the
/// collector.
pub fn unregister_mutator() {
    let thread_id = std::thread::current().id();
    let mut registry = MUTATOR_REGISTRY.lock().unwrap();
    let index = registry
        .iter()
        .position(|entry| entry.thread_id == thread_id)
        .expect("unregistering an unregistered mutator thread");
    registry.swap_remove(index);
}

/// The shadow stack itself.
struct ShadowStack {
    entries: Vec<GcRef>,
}

impl ShadowStack {
    fn new() -> Self {
        ShadowStack {
            entries: Vec::with_capacity(64),
        }
    }
}

// ── GcRef shadow stack (individual refs) ─────────────────────────

/// Push a GC reference onto the shadow stack.
pub fn push(gcref: GcRef) -> usize {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.entries.len();
        let max = MAX_SHADOW_STACK_DEPTH.with(|c| c.get());
        assert!(depth < max, "shadow stack overflow");
        ss.entries.push(gcref);
        depth
    })
}

/// Pop entries from the shadow stack back to the given depth.
///
/// TODO: Rust drops thread-locals in reverse order on
/// thread exit, and a TLS-owned `Drop` (e.g. `JitDriver`'s) may call
/// this during its own teardown. If
/// `SHADOW_STACK`'s destructor has already fired, `.with()` panics with
/// `AccessError`. RPython has no analogous hazard — the GIL thread does
/// not tear down TLS mid-run. Silently return an empty Vec so the
/// exiting thread proceeds.
pub fn pop_to(depth: usize) -> Vec<GcRef> {
    SHADOW_STACK
        .try_with(|ss| {
            let mut ss = ss.borrow_mut();
            ss.entries.split_off(depth)
        })
        .unwrap_or_default()
}

/// Like `pop_to` but silently no-ops if the shadow stack thread-local
/// is already torn down (program shutdown). Drop callers should use this
/// because the TLS drop order between `JitDriver` and `SHADOW_STACK` is
/// not deterministic in Rust.
pub fn try_pop_to(depth: usize) {
    let _ = SHADOW_STACK.try_with(|ss| {
        let mut ss = ss.borrow_mut();
        ss.entries.truncate(depth);
    });
}

/// Get a GcRef at the given index.
pub fn get(index: usize) -> GcRef {
    SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        ss.entries[index]
    })
}

/// Walk all entries on the GcRef shadow stack.
pub fn walk_roots(mut visitor: impl FnMut(&mut GcRef)) {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        for entry in ss.entries.iter_mut() {
            if !entry.is_null() {
                visitor(entry);
            }
        }
    });
}

/// Walk every registered mutator's GcRef shadow stack during STW.
///
/// This deliberately bypasses `RefCell::borrow_mut`: a RefCell borrow is a
/// same-thread runtime check and must not be used as the synchronization
/// mechanism for foreign TLS. gc_sync's quiescence is what makes the raw
/// dereferences and in-place forwarding sound.
pub fn walk_all_roots(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_roots walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = MUTATOR_REGISTRY.lock().unwrap();
    for mutator in registry.iter() {
        // SAFETY: every registered mutator is quiesced, and registry removal
        // precedes the owner's RUNNING decrement and TLS destruction.
        let ss = unsafe { &mut *(*mutator.shadow_stack).as_ptr() };
        for entry in ss.entries.iter_mut() {
            if !entry.is_null() {
                visitor(entry);
            }
        }
    }
}

/// Current depth of the GcRef shadow stack.
///
/// TODO: see `pop_to` for the TLS-teardown rationale.
/// Returns 0 when the TLS has been destroyed; callers running under
/// Drop (e.g. `ExportedState::release_roots`) observe an empty
/// stack instead of panicking on the destroyed key.
pub fn depth() -> usize {
    SHADOW_STACK
        .try_with(|ss| ss.borrow().entries.len())
        .unwrap_or(0)
}

/// rpython/memory/gctransform/shadowstack.py:351
/// `increase_root_stack_depth` parity. Grows BOTH:
///   * the per-thread safety cap for the generic GcRef shadow stack
///     (`MAX_SHADOW_STACK_DEPTH`); and
///   * the jitframe shadow stack's backing buffer (reallocates via
///     the per-thread backing buffer so compiled code can push up to
///     `new_depth` jitframes without running off the end).
///
/// Never shrinks. Called from `sys.setrecursionlimit` with
/// `int(new_limit * 0.001 * 163840)` (pypy/module/sys/vm.py:70).
pub fn increase_root_stack_depth(new_depth: usize) {
    MAX_SHADOW_STACK_DEPTH.with(|c| {
        if new_depth > c.get() {
            c.set(new_depth);
        }
    });
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        stack.grow(new_depth);
    });
}

/// Clear both shadow stacks.
pub fn clear() {
    SHADOW_STACK.with(|ss| ss.borrow_mut().entries.clear());
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        // Reset root_stack_top to base of the current backing buffer.
        stack.top.set(stack.base);
    });
}

// ── JitFrame shadow stack (assembler.py:1122-1136) ───────────────
//
// Flat memory array with entries: [is_minor(usize), jf_ptr(usize)].
// root_stack_top points past the last entry. Compiled code manipulates
// root_stack_top directly with inline load/store instructions.

/// Push a jitframe GcRef onto the flat shadow stack.
///
/// assembler.py:1122-1128 _call_header_shadowstack:
///   MOV [top], 1           // is_minor marker
///   MOV [top + WORD], ebp  // jf_ptr
///   ADD top, 2*WORD        // advance
pub fn push_jf(jf_ptr: GcRef) -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let top = stack.top.get() as *mut usize;
            let depth = (top as usize - base as usize) / (2 * WORD);
            assert!(
                depth < stack.capacity,
                "jf shadow stack overflow — capacity {}, depth {}",
                stack.capacity,
                depth
            );
            // assembler.py:1125: MOV [ebx], 1
            *top = 1;
            // assembler.py:1126: MOV [ebx + WORD], ebp
            *top.add(1) = jf_ptr.0;
            // assembler.py:1127: ADD ebx, 2*WORD
            let new_top = top.add(2);
            stack.top.set(new_top as usize);
            depth
        }
    })
}

/// Read the GC-updated jf_ptr from the shadow stack at the given depth.
///
/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]
///   MOV ebp, [ecx - WORD]
pub fn peek_jf(depth: usize) -> GcRef {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *const usize;
            // entry at `depth`: base[depth*2] = is_minor, base[depth*2+1] = jf_ptr
            GcRef(*base.add(depth * 2 + 1))
        }
    })
}

/// Pop jitframe entries back to the given depth.
///
/// assembler.py:1130-1136 _call_footer_shadowstack:
///   SUB [rootstacktop], 2*WORD
pub fn pop_jf_to(depth: usize) {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let new_top = base.add(depth * 2);
            stack.top.set(new_top as usize);
        }
    });
}

/// Pop the single top jitframe entry, if any.
///
/// assembler.py:1130-1136 _call_footer_shadowstack:
///   SUB [rootstacktop], 2*WORD
/// Single-access fusion of `jf_depth()` + `pop_jf_to(depth - 1)` for
/// per-call hot paths (the wasm CA return leg pops one frame per call);
/// the two-step form pays the thread-local + RefCell round-trip twice.
pub fn pop_jf_top() {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        let top = stack.top.get();
        if top > stack.base {
            stack.top.set(top - 2 * WORD);
        }
    });
}

/// Current depth of the jitframe shadow stack.
pub fn jf_depth() -> usize {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        (stack.top.get() - stack.base) / (2 * WORD)
    })
}

/// Walk jitframe shadow stack entries as GC roots.
///
/// Each jf_ptr is exposed as `&mut GcRef`. The GC treats it like any
/// other root: if it points into the nursery, the jitframe is copied
/// to old gen and the GcRef is updated in place.
///
/// The jitframe's internal ref slots are NOT traced here — that is
/// handled by `jitframe_custom_trace` via Phase 2 (remembered set +
/// custom_trace), exactly as in RPython where `root_walker.walk_roots()`
/// copies the jitframe, and then `jitframe_trace` (custom_trace hook)
/// traces the gcmap-indicated ref slots during Phase 2.
pub fn walk_jf_roots(mut visitor: impl FnMut(&mut GcRef)) {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let top = stack.top.get() as *mut usize;
            let mut ptr = base;
            while ptr < top {
                // ptr[0] = is_minor marker, ptr[1] = jf_ptr
                let jf_ref = &mut *(ptr.add(1) as *mut GcRef);
                if !jf_ref.is_null() {
                    visitor(jf_ref);
                }
                ptr = ptr.add(2);
            }
        }
    });
}

/// Walk every registered mutator's jitframe shadow stack during STW.
pub fn walk_all_jf_roots(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_jf_roots walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = MUTATOR_REGISTRY.lock().unwrap();
    for mutator in registry.iter() {
        // SAFETY: the owning mutator is quiesced and cannot change the stack
        // or its backing allocation until the STW guard resumes it.
        let stack = unsafe { &mut *(*mutator.jf_root_stack).as_ptr() };
        stack.ensure_init();
        unsafe {
            let base = stack.base as *mut usize;
            let top = stack.top.get() as *mut usize;
            let mut ptr = base;
            while ptr < top {
                let jf_ref = &mut *(ptr.add(1) as *mut GcRef);
                if !jf_ref.is_null() {
                    visitor(jf_ref);
                }
                ptr = ptr.add(2);
            }
        }
    }
}

/// Read the jf_ptr of the top shadow stack entry.
///
/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]
///   MOV ebp, [ecx - WORD]
///
/// After a collecting call, the GC may have copied the jitframe. The
/// shadow stack entry has been updated. Compiled code reloads jf_ptr
/// from here to get the (possibly new) address.
pub fn jf_top_ptr() -> GcRef {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let top = stack.top.get();
            if top <= stack.base {
                return GcRef::NULL;
            }
            // top points past the last entry; jf_ptr is at top - WORD
            let jf_ptr_addr = (top - WORD) as *const usize;
            GcRef(*jf_ptr_addr)
        }
    })
}

/// Read the jf_ptr one entry below the top jitframe shadow-stack entry.
///
/// While a CALL_ASSEMBLER callee is pushed, this is its caller's frame. A
/// collecting allocation performed before the callee push can update this
/// shadow-stack slot, so compiled wasm reloads its own frame from here before
/// addressing local-0-relative frame homes.
pub fn jf_under_top_ptr() -> GcRef {
    JF_ROOT_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        stack.ensure_init();
        unsafe {
            let top = stack.top.get();
            if top <= stack.base + 2 * WORD {
                return GcRef::NULL;
            }
            // Each entry is [is_minor, jf_ptr]. The caller's jf_ptr is three
            // words below top: top[-1] is this callee, top[-3] its caller.
            let jf_ptr_addr = (top - 3 * WORD) as *const usize;
            GcRef(*jf_ptr_addr)
        }
    })
}

// ── Blackhole register bank shadow stack ────────────────────────
//
// blackhole.py:840 BlackholeInterpreter.registers_r parity:
// the ref register file is part of the GC's root set during the
// blackhole interpreter's lifetime. RPython's GC scans the
// blackhole interpreter's RPython-managed Box arrays directly;
// pyre stores raw i64 ptrs in Vec<i64> so we register the active
// register banks with the collector explicitly.

/// One active blackhole register bank.
///
/// `regs_ptr` points at the start of a `Vec<i64>` slot range; `regs_len`
/// is the number of slots (working registers AND constants — constants
/// are pre-existing old-gen pointers and pass through `copy_nursery_object`
/// untouched). `tmpreg_ptr` points at the temporary ref register that
/// holds in-flight call return values.
#[derive(Clone, Copy)]
struct BhRegsEntry {
    regs_ptr: *mut i64,
    regs_len: usize,
    tmpreg_ptr: *mut i64,
}

// `*mut i64` is not `Send` by default. The thread-local storage means we
// only ever access entries from the thread that pushed them, so this
// `unsafe impl` is sound.
unsafe impl Send for BhRegsEntry {}

/// Push a blackhole interpreter's register bank onto the GC root stack.
///
/// The caller must ensure the slice remains valid until the matching
/// `pop_bh_regs_to(depth)` is called. Returns the previous depth so the
/// caller can restore it on exit (RPython _call_header_shadowstack /
/// _call_footer_shadowstack pattern).
///
/// # Safety
/// `regs` must remain alive and pinned until pop.
pub unsafe fn push_bh_regs(regs: &mut [i64], tmpreg: &mut i64) -> usize {
    BH_REGS_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        let depth = ss.len();
        let max = MAX_SHADOW_STACK_DEPTH.with(|c| c.get());
        assert!(depth < max, "blackhole regs stack overflow");
        ss.push(BhRegsEntry {
            regs_ptr: regs.as_mut_ptr(),
            regs_len: regs.len(),
            tmpreg_ptr: tmpreg as *mut i64,
        });
        depth
    })
}

/// Pop blackhole register entries back to the given depth.
pub fn pop_bh_regs_to(depth: usize) {
    let _ = BH_REGS_STACK.try_with(|ss| {
        ss.borrow_mut().truncate(depth);
    });
}

/// Walk all active blackhole register banks as GC roots.
///
/// Each i64 slot is exposed as `&mut GcRef`. Slots holding non-nursery
/// pointers (constants, old-gen, NULL) are passed through unchanged by
/// `copy_nursery_object` semantics in the visitor.
pub fn walk_bh_regs(mut visitor: impl FnMut(&mut GcRef)) {
    BH_REGS_STACK.with(|ss| {
        let ss = ss.borrow();
        for entry in ss.iter() {
            // SAFETY: the BlackholeInterpreter that pushed this entry is on
            // the call stack above us (we are inside its run() body via a
            // collecting call). The Vec<i64> backing storage is pinned for
            // the lifetime of that frame.
            let slots = unsafe { std::slice::from_raw_parts_mut(entry.regs_ptr, entry.regs_len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
            // tmpreg_r holds in-flight call return values between
            // `let result = call_int_function(...)` and the subsequent
            // `self.registers_r[dst] = result;` store. RPython's blackhole
            // interpreter holds this in an RPython-managed slot so it is
            // automatically a root.
            let tmp = unsafe { &mut *(entry.tmpreg_ptr as *mut GcRef) };
            visitor(tmp);
        }
    });
}

/// Walk every registered mutator's blackhole register roots during STW.
pub fn walk_all_bh_regs(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_bh_regs walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = MUTATOR_REGISTRY.lock().unwrap();
    for mutator in registry.iter() {
        // SAFETY: all owners are quiesced, and each registered slice remains
        // pinned until its owning blackhole frame pops the entry after resume.
        let entries = unsafe { &*(*mutator.bh_regs_stack).as_ptr() };
        for entry in entries.iter() {
            let slots = unsafe { std::slice::from_raw_parts_mut(entry.regs_ptr, entry.regs_len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
            let tmp = unsafe { &mut *(entry.tmpreg_ptr as *mut GcRef) };
            visitor(tmp);
        }
    }
}

/// Current depth of the blackhole register bank stack.
pub fn bh_regs_depth() -> usize {
    BH_REGS_STACK.with(|ss| ss.borrow().len())
}

/// Current depth of the resume-construction ref-slice root stack.
pub fn resume_ref_roots_depth() -> usize {
    RESUME_REF_ROOTS_STACK.with(|ss| ss.borrow().len())
}

/// Register a ref slice as a GC root for the blackhole resume
/// construction window (`resume.py:1312 blackhole_from_resumedata`).
///
/// # Safety
/// `slice` must remain alive and at a fixed address until the matching
/// `pop_resume_ref_roots_to` runs.  Both the `virtuals_cache` and a
/// blackhole frame's `registers_r` are stable after allocation
/// (`prepare_virtuals` / `setposition` size them once; later fills only
/// index), so the captured pointer stays valid for the whole window.
pub unsafe fn push_resume_ref_roots(slice: &mut [i64]) {
    RESUME_REF_ROOTS_STACK.with(|ss| {
        ss.borrow_mut().push((slice.as_mut_ptr(), slice.len()));
    });
}

/// Pop resume-construction ref-slice roots back to the given depth.
pub fn pop_resume_ref_roots_to(depth: usize) {
    let _ = RESUME_REF_ROOTS_STACK.try_with(|ss| {
        ss.borrow_mut().truncate(depth);
    });
}

/// Walk all registered resume-construction ref slices as GC roots.
///
/// Each `i64` slot is exposed as `&mut GcRef`; non-nursery values
/// (unmaterialized `0` cache slots, old-gen pointers) pass through the
/// visitor's `copy_nursery_object` guard unchanged.
pub fn walk_resume_ref_roots(mut visitor: impl FnMut(&mut GcRef)) {
    RESUME_REF_ROOTS_STACK.with(|ss| {
        for &(ptr, len) in ss.borrow().iter() {
            let slots = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
        }
    });
}

/// Walk every registered mutator's resume-construction roots during STW.
pub fn walk_all_resume_ref_roots(mut visitor: impl FnMut(&mut GcRef)) {
    debug_assert!(
        crate::gc_sync::mutators_quiesced(),
        "walk_all_resume_ref_roots walks foreign mutator TLS; caller must own collector-side STW",
    );
    let registry = MUTATOR_REGISTRY.lock().unwrap();
    for mutator in registry.iter() {
        // SAFETY: the owner is quiesced and the registered slices stay pinned
        // for the complete resume-construction window.
        let entries = unsafe { &*(*mutator.resume_ref_roots_stack).as_ptr() };
        for &(ptr, len) in entries.iter() {
            let slots = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            for slot in slots.iter_mut() {
                let gcref = unsafe { &mut *(slot as *mut i64 as *mut GcRef) };
                visitor(gcref);
            }
        }
    }
}

// ── Extra root walkers registered by the embedder ───────────────
//
// rpython/memory/gctransform/framework.py `root_walker.walk_roots` parity:
// RPython registers GcRootMap sources (shadow stack, thread-local refs,
// jitframes, stack roots) once during setup; at collection time, the GC
// iterates all registered sources.
//
// pyre's interpreter holds live GC refs in `PyFrame.locals_cells_stack_w`
// and must register a walker for that storage from outside majit-gc
// (a crate-level upward dependency would cycle). The hook lets any
// embedder (pyre-interpreter, pyre-jit) plug a callback that the
// collector will call during `Phase 1e` of `do_collect_nursery`.

/// Signature expected by `register_extra_root_walker`.
///
/// The registered function receives an opaque visitor that must be
/// invoked once per GC root slot. Each slot is exposed as
/// `&mut GcRef`; writing back through the reference forwards the
/// root if the GC moved the referenced object.
pub type ExtraRootWalkerFn = fn(&mut dyn FnMut(&mut GcRef));

/// Which collection is currently driving the extra-root walk.
///
/// incminimark distinguishes the two: a minor collection scans an
/// old/prebuilt object only when the write barrier recorded a store into it
/// (`old_objects_pointing_to_young`, incminimark.py:339-344), while a major
/// collection always traces `prebuilt_root_objects` (incminimark.py:355).
/// Walkers that mirror prebuilt-object scanning read this to apply the same
/// minor-skip; the default is [`ExtraRootWalkKind::Major`] (scan everything)
/// so an unset call site stays conservative.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExtraRootWalkKind {
    Minor,
    Major,
}

thread_local! {
    static EXTRA_ROOT_WALK_KIND: std::cell::Cell<ExtraRootWalkKind> =
        const { std::cell::Cell::new(ExtraRootWalkKind::Major) };
}

/// Set by the collector around its root-walk phases; see
/// [`ExtraRootWalkKind`].
pub fn set_extra_root_walk_kind(kind: ExtraRootWalkKind) {
    EXTRA_ROOT_WALK_KIND.with(|k| k.set(kind));
}

/// The collection kind driving the current extra-root walk.
pub fn extra_root_walk_kind() -> ExtraRootWalkKind {
    EXTRA_ROOT_WALK_KIND.with(|k| k.get())
}

// Walkers registered today: eval.rs registers twelve (rd_consts, partial
// trace, active trace, compile snapshot, jitcode constants, fbw store
// journal, fbw finish concrete, pyre interpreter side tables, signal
// handlers, weakref-box inner, jit callee frames, and pyre objects), plus
// gcreftracer.rs registers the per-loop gc_table walker — thirteen total.
// Leave headroom above that so a future root source does not overflow the
// array and poison the registry lock with a "capacity exceeded" panic on
// first use.
const MAX_EXTRA_ROOT_WALKERS: usize = 16;

/// Registered root walkers. Each slot is either `None` or a function
/// pointer. We cap the count to keep the set stack-allocatable and
/// avoid dynamic allocation in the GC hot path.
static EXTRA_ROOT_WALKERS: std::sync::RwLock<[Option<ExtraRootWalkerFn>; MAX_EXTRA_ROOT_WALKERS]> =
    std::sync::RwLock::new([None; MAX_EXTRA_ROOT_WALKERS]);

/// Register an additional root walker.
///
/// Called at process start (or module init) by the embedder once per
/// root source. Duplicate registrations are tolerated — the walker is
/// only appended if not already present.
pub fn register_extra_root_walker(walker: ExtraRootWalkerFn) {
    let mut guard = EXTRA_ROOT_WALKERS.write().unwrap();
    for slot in guard.iter_mut() {
        match slot {
            Some(existing) if std::ptr::fn_addr_eq(*existing, walker) => return,
            None => {
                *slot = Some(walker);
                return;
            }
            _ => {}
        }
    }
    panic!(
        "register_extra_root_walker: capacity exceeded ({} walkers already registered)",
        MAX_EXTRA_ROOT_WALKERS
    );
}

/// Invoke every registered extra root walker with the given visitor.
///
/// Called by `MiniMarkGC::do_collect_nursery` (Phase 1e).
pub fn walk_extra_roots(mut visitor: impl FnMut(&mut GcRef)) {
    // Snapshot the walker list under a read guard so a walker that
    // triggers further allocation (and recursively a collection) does
    // not observe the lock held in write mode.
    let walkers = {
        let guard = EXTRA_ROOT_WALKERS.read().unwrap();
        *guard
    };
    for slot in walkers.iter() {
        if let Some(walker) = slot {
            walker(&mut visitor);
        }
    }
}

// ── Extern "C" interface for compiled code ──────────────────────

/// Push a GC reference from compiled code.
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_push(gcref_raw: i64) -> i64 {
    let gcref = GcRef(gcref_raw as usize);
    push(gcref) as i64
}

/// Pop shadow stack to depth and return the value at given index.
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_pop_and_get(_depth: i64, index: i64) -> i64 {
    SHADOW_STACK.with(|ss| {
        let ss = ss.borrow();
        let idx = index as usize;
        if idx < ss.entries.len() {
            ss.entries[idx].0 as i64
        } else {
            0
        }
    })
}

/// Set shadow stack depth (truncate).
#[unsafe(no_mangle)]
pub extern "C" fn majit_shadow_stack_set_depth(new_depth: i64) {
    SHADOW_STACK.with(|ss| {
        let mut ss = ss.borrow_mut();
        ss.entries.truncate(new_depth as usize);
    });
}

/// Read the top jf_ptr from the jitframe shadow stack.
///
/// assembler.py:1369-1377 _reload_frame_if_necessary:
///   MOV ecx, [rootstacktop]; MOV ebp, [ecx - WORD]
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_get_top_jf_ptr() -> i64 {
    jf_top_ptr().0 as i64
}

/// _call_header_shadowstack (assembler.py:1122-1128).
/// Non-compiled callers use this; compiled code uses inline MOVs.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_push(jf_ptr_raw: i64) -> i64 {
    push_jf(crate::GcRef(jf_ptr_raw as usize)) as i64
}

/// _call_footer_shadowstack (assembler.py:1130-1136).
/// Non-compiled callers use this; compiled code uses inline SUB.
#[unsafe(no_mangle)]
pub extern "C" fn majit_jf_shadow_stack_pop_to(depth: i64) {
    pop_jf_to(depth as usize);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Keep these tests serialized because some of them intentionally shrink
    // the current thread's capacity and then re-raise an expected panic.
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn jf_root_stack_base_for_test() -> usize {
        JF_ROOT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.ensure_init();
            stack.base
        })
    }

    fn jf_root_stack_capacity_for_test() -> usize {
        JF_ROOT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.ensure_init();
            stack.capacity
        })
    }

    fn set_jf_root_stack_capacity_for_test(capacity: usize) {
        JF_ROOT_STACK.with(|stack| {
            let mut stack = stack.borrow_mut();
            stack.ensure_init();
            stack.capacity = capacity;
        });
    }

    #[test]
    fn test_push_pop_roundtrip() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        let a = GcRef(0x1000);
        let b = GcRef(0x2000);
        let depth = push(a);
        assert_eq!(depth, 0);
        push(b);
        assert_eq!(super::depth(), 2);

        let popped = pop_to(depth);
        assert_eq!(popped.len(), 2);
        assert_eq!(popped[0], a);
        assert_eq!(popped[1], b);
        assert_eq!(super::depth(), 0);
    }

    #[test]
    fn test_walk_roots_updates() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        push(GcRef(0x1000));
        push(GcRef(0x2000));

        // Simulate GC moving objects
        walk_roots(|gcref| {
            gcref.0 += 0x100;
        });

        assert_eq!(get(0), GcRef(0x1100));
        assert_eq!(get(1), GcRef(0x2100));
        clear();
    }

    #[test]
    fn test_mutator_extra_area_walks_current_thread() {
        unsafe fn walk_cell(data: *const (), visitor: &mut dyn FnMut(&mut GcRef)) {
            let slot = unsafe { &mut *(data as *mut GcRef) };
            visitor(slot);
        }

        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        let mut root = GcRef(0x1000);
        register_mutator();
        // SAFETY: `root` remains valid until unregistration, and `walk_cell`
        // derives its sole dereference from the supplied `data` pointer.
        unsafe {
            register_mutator_extra_area(walk_cell, &mut root as *mut GcRef as *const ());
        }
        walk_my_extra_areas(|gcref| gcref.0 += 0x100);
        unregister_mutator();
        assert_eq!(root, GcRef(0x1100));
    }

    #[test]
    fn test_extern_c_interface() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        let depth = majit_shadow_stack_push(0x3000);
        assert_eq!(depth, 0);
        majit_shadow_stack_push(0x4000);

        let val = majit_shadow_stack_pop_and_get(0, 0);
        assert_eq!(val, 0x3000);
        let val = majit_shadow_stack_pop_and_get(0, 1);
        assert_eq!(val, 0x4000);

        majit_shadow_stack_set_depth(0);
        assert_eq!(super::depth(), 0);
    }

    #[test]
    fn test_jf_shadow_stack_push_pop() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        assert_eq!(jf_depth(), 0);
        let depth = push_jf(GcRef(0x1000));
        assert_eq!(depth, 0);
        assert_eq!(jf_depth(), 1);
        assert_eq!(peek_jf(0), GcRef(0x1000));
        push_jf(GcRef(0x2000));
        assert_eq!(jf_depth(), 2);
        assert_eq!(peek_jf(1), GcRef(0x2000));
        pop_jf_to(depth);
        assert_eq!(jf_depth(), 0);
    }

    #[test]
    fn test_walk_jf_roots_updates_gcref() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        push_jf(GcRef(0x1000));
        push_jf(GcRef(0x2000));

        // Simulate GC moving jitframes (root_walker semantics)
        walk_jf_roots(|gcref| {
            gcref.0 += 0x100;
        });

        // After walk, entries should be updated in-place
        assert_eq!(peek_jf(0), GcRef(0x1100));
        assert_eq!(jf_top_ptr(), GcRef(0x2100));
        pop_jf_to(0);
    }

    #[test]
    fn test_jf_top_ptr_reload() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        push_jf(GcRef(0xABCD));
        assert_eq!(jf_top_ptr(), GcRef(0xABCD));
        assert_eq!(majit_jf_shadow_stack_get_top_jf_ptr(), 0xABCD);

        // Simulate GC updating the entry
        walk_jf_roots(|gcref| {
            gcref.0 = 0xDEAD;
        });
        assert_eq!(jf_top_ptr(), GcRef(0xDEAD));
        assert_eq!(majit_jf_shadow_stack_get_top_jf_ptr(), 0xDEAD as i64);
        pop_jf_to(0);
    }

    #[test]
    fn test_jf_flat_array_layout() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        // Verify the flat array layout matches RPython's [is_minor, jf_ptr] pairs
        clear();
        push_jf(GcRef(0xAAAA));
        push_jf(GcRef(0xBBBB));
        unsafe {
            let base = jf_root_stack_base_for_test() as *const usize;
            // Entry 0: [1, 0xAAAA]
            assert_eq!(*base, 1);
            assert_eq!(*base.add(1), 0xAAAA);
            // Entry 1: [1, 0xBBBB]
            assert_eq!(*base.add(2), 1);
            assert_eq!(*base.add(3), 0xBBBB);
        }
        pop_jf_to(0);
    }

    /// rpython/memory/gctransform/shadowstack.py:351 parity:
    /// sys.setrecursionlimit(>1000) must resize the jitframe
    /// shadow-stack backing buffer, not just raise a limit.
    #[test]
    fn test_increase_root_stack_depth_grows_backing_buffer() {
        let _lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        // Populate a known jf_ptr that must survive the resize.
        push_jf(GcRef(0xDEADBEEF));
        let base_before = jf_root_stack_base_for_test();
        let capacity_before = jf_root_stack_capacity_for_test();
        let limit_addr = get_root_stack_limit_addr();
        let limit_before = unsafe { *(limit_addr as *const usize) };
        assert_eq!(capacity_before, DEFAULT_SHADOW_STACK_DEPTH);

        // Grow to 2x the default. RPython's resize copies the used
        // portion to the new buffer.
        let new_cap = DEFAULT_SHADOW_STACK_DEPTH * 2;
        increase_root_stack_depth(new_cap);

        let base_after = jf_root_stack_base_for_test();
        let capacity_after = jf_root_stack_capacity_for_test();
        assert_eq!(get_root_stack_limit_addr(), limit_addr);
        assert_ne!(unsafe { *(limit_addr as *const usize) }, limit_before);
        assert_eq!(
            unsafe { *(limit_addr as *const usize) },
            base_after + new_cap * 2 * WORD,
            "the stable limit cell must track the reallocated backing buffer"
        );
        assert_ne!(
            base_before, base_after,
            "resize must reallocate the backing buffer"
        );
        assert_eq!(
            capacity_after, new_cap,
            "capacity atomic must reflect the new size"
        );
        // The copied jf_ptr must still be reachable via peek_jf.
        assert_eq!(peek_jf(0), GcRef(0xDEADBEEF));
        // Pushing up to the new capacity must not overflow.
        for i in 1..new_cap {
            push_jf(GcRef(0x1000 + i));
        }
        assert_eq!(jf_depth(), new_cap);
        pop_jf_to(0);
    }

    /// Pushing exactly `capacity` entries is OK; push #{capacity+1}
    /// must trip the overflow assert, not silently corrupt heap.
    #[test]
    #[should_panic(expected = "jf shadow stack overflow")]
    fn test_jf_shadow_stack_overflow_panics() {
        let lock = TEST_MUTEX.lock().unwrap_or_else(|p| p.into_inner());
        clear();
        // Pre-shrink: the capacity atomic drives the assert; the
        // backing HEAP buffer is larger than 4 entries, so the
        // overflow fires before any OOB write.
        let original_cap = jf_root_stack_capacity_for_test();
        set_jf_root_stack_capacity_for_test(4);
        let result = std::panic::catch_unwind(|| {
            for i in 0..4 {
                push_jf(GcRef(i));
            }
            push_jf(GcRef(99)); // MUST panic
        });
        // Restore state before dropping the lock so the next test
        // sees the default capacity. Then release the lock cleanly
        // before re-raising, so the Mutex doesn't poison.
        pop_jf_to(0);
        set_jf_root_stack_capacity_for_test(original_cap);
        drop(lock);
        match result {
            Err(payload) => std::panic::resume_unwind(payload),
            Ok(_) => panic!("jf shadow stack overflow did not fire"),
        }
    }
}
