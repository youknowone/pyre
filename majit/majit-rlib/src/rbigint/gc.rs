//! The GC boundary of `rbigint` — everything `rpython/rlib/rbigint.py` does
//! not contain because RPython's translator supplies it.
//!
//! Upstream's rbigint names no malloc, no root, and no write barrier: the GC
//! transform inserts shadow-stack pushes and barriers into the final graphs
//! (`rpython/memory/gctransform/framework.py`), two pipeline stages after the
//! codewriter (`rpython/translator/driver.py:348` vs `:408`). pyre's binary is
//! produced by rustc, which has no such stage, so this code is written by hand
//! — but it is written *here*, in the layer that corresponds to the one that
//! generates it, and not in the port of the algorithm file.
//!
//! What belongs here: root guards for handles a Rust local holds across a
//! collecting call, the payload/`tuple2` allocators and the tier they select,
//! creation write barriers, and the runtime GC type ids. What does not:
//! anything with a line in `rbigint.py`.

use super::*;
use majit_gc::GcAllocOutcome;

// ---- RBigIntGcRoot ----
/// Address-stable GC root for a host-side, by-value `RBigInt`.
///
/// RPython's GC transform puts an `rbigint` local's `_digits` edge in the
/// shadow stack whenever that local is live across a call that can collect.
/// Rust locals have no generated stack map, so interpreter-level consumers
/// that retain an unboxed `RBigInt` across a Python callback use this exact
/// analogue.  The `Box` is required: moving this guard must not move the slot
/// registered with MiniMark.
pub struct RBigIntGcRoot {
    value: Box<RBigInt>,
    slot: *mut majit_ir::GcRef,
    registered: bool,
}

impl RBigIntGcRoot {
    pub fn new(value: RBigInt) -> Self {
        let mut value = Box::new(value);
        let slot = (&mut value._digits as *mut *mut TypedItemsBlock).cast::<majit_ir::GcRef>();
        let registered = unsafe { majit_gc::gc_add_root(slot) };
        Self {
            value,
            slot,
            registered,
        }
    }
}

impl std::ops::Deref for RBigIntGcRoot {
    type Target = RBigInt;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl std::ops::DerefMut for RBigIntGcRoot {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl Drop for RBigIntGcRoot {
    fn drop(&mut self) {
        if self.registered {
            majit_gc::gc_remove_root(self.slot);
        }
    }
}

// ---- payload offsets, type id, prebuilt identity, payload allocators ----
/// Offset used when registering the raw RBigInt payload with MiniMark.
pub const RBIGINT_DIGITS_OFFSET: usize = std::mem::offset_of!(RBigInt, _digits);
pub const RBIGINT_PAYLOAD_SIZE: usize = std::mem::size_of::<RBigInt>();

/// Runtime GC id for the plain `rbigint` object.  The payload has one GC edge,
/// `_digits`, to its `GcArray(Signed)` and no destructor or external storage.
static RBIGINT_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_rbigint_gc_type_id(id: u32) {
    RBIGINT_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[majit_macros::dont_look_inside]
pub fn rbigint_gc_type_id() -> u32 {
    RBIGINT_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Return the translated prebuilt object's immortal payload address when
/// `value` already aliases one of its digit arrays. Identity of the digit
/// slot, not numeric equality, is intentional: upstream has a few internal
/// zero results that must remain fresh because their digits are filled later.
pub(super) fn prebuilt_payload_pointer(value: &RBigInt) -> Option<*mut RBigInt> {
    // All four are single-digit objects: `zero()` carries `_size == 0` and the
    // other three `|_size| == 1` (rbigint.py). Their digit arrays are
    // one element long, so a value holding two or more digits cannot alias any
    // of them — and `_normalize` (rbigint.py) is the only route by
    // which a computed result reaches the zero form, where it assigns
    // `self._digits = NULLDIGITS` itself. Deciding that from the size leaves
    // the table for the values that can actually match.
    if !(-1..=1).contains(&value._size) {
        return None;
    }
    let digits = value._digits as usize;
    for (index, slot) in PREBUILT_DIGITS.iter().enumerate() {
        // An unpublished slot reads 0, which no live digit array can equal.
        if slot.load(std::sync::atomic::Ordering::Relaxed) != digits {
            continue;
        }
        let Some(&raw) = prebuilt_slots()[index].get() else {
            continue;
        };
        let prebuilt = unsafe { &*(raw as *const RBigInt) };
        if value._size == prebuilt._size {
            return Some(raw as *mut RBigInt);
        }
    }
    None
}

#[inline]
pub(crate) fn alloc_rbigint_nursery_impl(
    value: RBigInt,
    canonicalize_prebuilt: bool,
) -> *mut RBigInt {
    if canonicalize_prebuilt && let Some(prebuilt) = prebuilt_payload_pointer(&value) {
        return prebuilt;
    }
    let tid = rbigint_gc_type_id();
    let mut needs_write_barrier = true;
    // A `Some(null)` here means the GC owns the heap and could not satisfy the
    // request; `malloc_raw` below would then leave `_digits` — this payload's
    // one traced edge — unreachable to the collector.
    if tid != 0
        && let Some(raw) = GcAllocOutcome::classify(unsafe {
            majit_gc::alloc_fast_nursery_typed_with_placement(
                tid,
                RBIGINT_PAYLOAD_SIZE,
                &mut needs_write_barrier,
            )
        })
        .allocated_or_abort(RBIGINT_PAYLOAD_SIZE)
    {
        unsafe {
            std::ptr::write(raw as *mut RBigInt, value);
        }
        // framework.py `propagate_no_write_barrier_needed` removes
        // GC-pointer field barriers while initializing a fresh fixed-size
        // nursery allocation. The no-collect allocator reports the exceptional
        // old-gen spill, where `_digits` can still be young.
        if needs_write_barrier {
            majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
        }
        return raw as *mut RBigInt;
    }
    crate::malloc_raw(value)
}

#[inline]
pub fn alloc_rbigint_nursery(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_impl(value, true)
}

/// No-collect twin of [`alloc_rbigint_clone_nursery_collecting`]: allocate a
/// fresh handle for a shallow copy whose digit array happens to be a prebuilt
/// value's, without canonicalizing it back onto that prebuilt payload.
#[inline]
pub fn alloc_rbigint_clone_nursery(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_impl(value, false)
}

#[inline]
fn alloc_rbigint_nursery_collecting_impl(
    mut value: RBigInt,
    canonicalize_prebuilt: bool,
) -> *mut RBigInt {
    if canonicalize_prebuilt && let Some(prebuilt) = prebuilt_payload_pointer(&value) {
        return prebuilt;
    }
    let tid = rbigint_gc_type_id();
    if tid != 0 {
        // RPython's stack map exposes this freshly-computed rbigint's sole GC
        // edge only when malloc reaches collect_and_reserve. The rooted
        // collecting hook preserves that shape: the common nursery bump does
        // no dynamic root-set mutation, while the nursery-full slow path
        // temporarily registers and forwards this exact digit slot.
        //
        // The rbigint payload registers no destructor and is not a WEAKREF —
        // its one traced edge is `_digits` — so this malloc site is one of the
        // `malloc_fast` sites `gct_fv_gc_malloc` (`framework.py`)
        // selects.
        let digit_slot =
            (&mut value._digits as *mut *mut TypedItemsBlock).cast::<majit_ir::GcRef>();
        let mut needs_write_barrier = true;
        // `NoRoute` falls through to the no-collect path below, which has its
        // own hook to try. A failure does not: this allocation already ran a
        // minor collection, so retrying the no-collect path would only reach
        // its `malloc_raw` fallback and hide the failure behind an untraced
        // payload.
        let raw = GcAllocOutcome::classify(unsafe {
            majit_gc::alloc_fast_nursery_collecting_typed_rooted(
                tid,
                RBIGINT_PAYLOAD_SIZE,
                digit_slot,
                &mut needs_write_barrier,
            )
        })
        .allocated_or_abort(RBIGINT_PAYLOAD_SIZE);
        if let Some(raw) = raw {
            unsafe {
                std::ptr::write(raw as *mut RBigInt, value);
            }
            // framework.py `propagate_no_write_barrier_needed` removes
            // GC-pointer field barriers while initializing a fresh fixed-size
            // nursery allocation. Retain it only for collectors that satisfy
            // the request in old-gen.
            if needs_write_barrier {
                majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
            }
            return raw as *mut RBigInt;
        }
    }
    alloc_rbigint_nursery_impl(value, canonicalize_prebuilt)
}

#[inline]
pub fn alloc_rbigint_nursery_collecting(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_collecting_impl(value, true)
}

/// Let the collector reclaim the conversion's temporaries before the nursery
/// is exhausted.
///
/// `rbigint.py`'s loop reaches `x.divmod(...)` on every iteration,
/// and upstream that is an ordinary malloc: `[NULLDIGIT] * size` lowers to
/// `ll_newlist`, the inlined `malloc_fast` copy of `malloc_fixedsize`
/// (framework.py:366-373), whose nursery bump reaches `collect_and_reserve`
/// the moment it overflows (incminimark.py:676-680). So the quotients and
/// remainders of earlier iterations — dead the moment the next one starts —
/// never leave the nursery.
///
/// pyre spells that same allocation as `Digits::new`, which routes to
/// `alloc_fast_nursery_typed` — `malloc_fast`'s *non*-collecting twin. It has
/// to: a collection there would move digit blocks out from under the unboxed
/// `RBigInt` handles every arithmetic graph in this file holds across it, and
/// nothing roots them. It spills to old-gen instead, so a long conversion
/// fills the nursery once and then promotes every later block, and the next
/// allocation *after* the conversion pays for all of them in one collection.
///
/// Routing `Digits::new` to the collecting allocator is what would close this
/// deviation outright, and it is not a change this function makes: it would
/// oblige every caller in the file to root what it holds.
///
/// Minting one payload through the collecting allocator restores the upstream
/// shape at the one point in that loop where the frame's only live edge is
/// already rooted. The bump fails exactly when the nursery is exhausted, which
/// is exactly when spilling would otherwise begin; the collection that follows
/// hands the loop a fresh nursery, so at most one iteration's blocks can be
/// promoted instead of the whole conversion's.
///
/// The minted handle is immediately dead. That is the point: it is a request
/// for the collection, not a value.
///
/// # Safety
///
/// The caller must keep every `RBigInt` it holds live across this call rooted
/// — `x` in the two `_format_recursive_*` graphs is held in an
/// [`RBigIntGcRoot`] for exactly this reason — and must not read an `&RBigInt`
/// derived before the call afterwards.
#[inline]
pub unsafe fn format_recursion_safepoint(rooted: &RBigInt) {
    let _ = alloc_rbigint_clone_nursery_collecting(rooted.clone());
}

/// Allocate the fresh translated GC handle required by `RBigInt::clone`.
///
/// RPython's `rbigint.neg`/`abs` shallow-copy the immutable digit list and
/// then update the new rbigint object's sign. Rust represents that intermediate
/// object as a by-value handle, so the clone residual must preserve the shared
/// digit array while bypassing the ordinary prebuilt-payload canonicalization.
#[inline]
pub fn alloc_rbigint_clone_nursery_collecting(value: RBigInt) -> *mut RBigInt {
    alloc_rbigint_nursery_collecting_impl(value, false)
}

#[inline]
pub fn alloc_rbigint_stable(value: RBigInt) -> *mut RBigInt {
    if let Some(prebuilt) = prebuilt_payload_pointer(&value) {
        return prebuilt;
    }
    let tid = rbigint_gc_type_id();
    if tid != 0 {
        // `NoRoute` leaves `raw` null and falls through to `malloc_raw`; a
        // `Failed` aborts inside `allocated_or_abort`.
        let raw = GcAllocOutcome::classify(majit_gc::alloc_oldgen_typed(tid, RBIGINT_PAYLOAD_SIZE))
            .allocated_or_abort(RBIGINT_PAYLOAD_SIZE)
            .unwrap_or(std::ptr::null_mut());
        if !raw.is_null() {
            unsafe {
                std::ptr::write(raw as *mut RBigInt, value);
            }
            // `raw` is old-gen while `value._digits` is the RPython-style
            // nursery GcArray(Signed).  Without this creation barrier a minor
            // collection never visits the payload and reclaims/moves the live
            // digit array behind W_LongObject.
            majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
            return raw as *mut RBigInt;
        }
    }
    crate::malloc_raw(value)
}

// ---- pair type id and pair allocators ----
/// Runtime GC id for the `tuple2` struct. Both fields are traced edges; a pair
/// allocated before the id is published (bare tests, pre-init bootstrap) falls
/// back to a leaked raw allocation, like the payload helpers above.
static RBIGINT_PAIR_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

pub fn set_rbigint_pair_gc_type_id(id: u32) {
    RBIGINT_PAIR_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

#[majit_macros::dont_look_inside]
pub fn rbigint_pair_gc_type_id() -> u32 {
    RBIGINT_PAIR_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Explicit root for an already-allocated GC pointer held in a host local.
///
/// RPython's stack map covers `div` and `mod` from the moment each is allocated
/// until the `tuple2` malloc stores them. A Rust local has no generated map, so
/// each payload pointer registers its own slot; the collector forwards through
/// it exactly as it would through a shadow-stack entry.
struct PendingPairItemRoot {
    slot: *mut majit_ir::GcRef,
    registered: bool,
}

impl PendingPairItemRoot {
    /// `slot` must outlive the guard and must not move while it is registered.
    unsafe fn new(slot: *mut *mut RBigInt) -> Self {
        let slot = slot.cast::<majit_ir::GcRef>();
        let registered = unsafe { majit_gc::gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for PendingPairItemRoot {
    fn drop(&mut self) {
        if self.registered {
            majit_gc::gc_remove_root(self.slot);
        }
    }
}

/// Allocate both halves and the `tuple2` that owns them.
///
/// The order is upstream's: `div` and `mod` become GC objects first, the pair
/// last, so no store outlives the allocation that could move its target. Each
/// step keeps everything already allocated reachable — the by-value handles via
/// their `_digits` roots inside the payload allocator, the payload pointers via
/// the guards here.
pub fn alloc_rbigint_pair_nursery_collecting(item0: RBigInt, item1: RBigInt) -> *mut RBigIntPair {
    // Until a half has its own payload it is only a by-value handle, and the
    // payload allocator roots the digits of the handle it was given — not the
    // other one. Root both up front, as `_int_divmod`'s caller does around its
    // two `newlong` calls.
    let item0 = RBigIntGcRoot::new(item0);
    let item1 = RBigIntGcRoot::new(item1);

    let mut item0 = alloc_rbigint_nursery_collecting(item0.translated_alias());
    let _item0_root = unsafe { PendingPairItemRoot::new(&mut item0) };
    let mut item1 = alloc_rbigint_nursery_collecting(item1.translated_alias());
    let _item1_root = unsafe { PendingPairItemRoot::new(&mut item1) };

    let tid = rbigint_pair_gc_type_id();
    if tid != 0 {
        // The collecting hook is the one the JIT residual wants; backends
        // without it fall through to the no-collect allocator, and only a
        // pre-init heap reaches `malloc_raw`. An untraced pair would be an
        // invisible edge to two GC-managed payloads, so this chain must not end
        // in `malloc_raw` while the payloads themselves are GC-managed.
        let raw = GcAllocOutcome::classify(majit_gc::alloc_nursery_collecting_typed(
            tid,
            RBIGINT_PAIR_SIZE,
        ))
        .allocated_or_abort(RBIGINT_PAIR_SIZE)
        .or_else(|| {
            GcAllocOutcome::classify(majit_gc::alloc_nursery_typed(tid, RBIGINT_PAIR_SIZE))
                .allocated_or_abort(RBIGINT_PAIR_SIZE)
        });
        if let Some(raw) = raw {
            unsafe {
                // Any collection the allocation above ran forwarded both roots,
                // so these reads take the post-collection addresses.
                std::ptr::write(raw as *mut RBigIntPair, RBigIntPair { item0, item1 });
            }
            // A nursery-full allocation can satisfy the pair from old-gen while
            // both payloads stay young, so this is not the fresh fixed-size
            // initialization whose field barriers framework.py:28-61
            // `propagate_no_write_barrier_needed` removes.
            majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
            return raw as *mut RBigIntPair;
        }
    }
    crate::malloc_raw(RBigIntPair { item0, item1 })
}

/// Build the `tuple2` over two payloads that are already reachable.
///
/// The walker needs a concrete pair to attach to the `CallR` it records, but it
/// runs on the host stack with no gcmap over its live set — which is the one
/// thing [`alloc_rbigint_pair_nursery_collecting`] requires of its caller. This
/// allocation therefore cannot collect, so the caller's live payloads keep the
/// addresses it read them at.
pub fn alloc_rbigint_pair_no_collect(item0: *mut RBigInt, item1: *mut RBigInt) -> *mut RBigIntPair {
    let tid = rbigint_pair_gc_type_id();
    if tid != 0
        && let Some(raw) =
            GcAllocOutcome::classify(majit_gc::alloc_nursery_typed(tid, RBIGINT_PAIR_SIZE))
                .allocated_or_abort(RBIGINT_PAIR_SIZE)
    {
        unsafe {
            std::ptr::write(raw as *mut RBigIntPair, RBigIntPair { item0, item1 });
        }
        majit_gc::gc_write_barrier(majit_ir::GcRef(raw as usize));
        return raw as *mut RBigIntPair;
    }
    crate::malloc_raw(RBigIntPair { item0, item1 })
}

// ---- PendingPartsCacheDigitRoot ----
/// Explicit root for a cached rbigint that has been computed but is not yet
/// reachable from the translated module-global `_parts_cache` graph.
///
/// RPython's GC transform roots this local automatically across publication.
/// In pyre another mutator may collect while this thread is allocating the
/// host-side snapshot vector, so the cached value's movable GcArray(Signed)
/// slot must be registered until either the shared list owns it or it is
/// discarded after losing a concurrent append race.
pub(super) struct PendingPartsCacheDigitRoot {
    slot: *mut majit_ir::GcRef,
    registered: bool,
}

impl PendingPartsCacheDigitRoot {
    /// `value`'s Arc allocation keeps the slot address stable for this
    /// guard's lifetime.
    pub(super) unsafe fn new(value: &std::sync::Arc<RBigInt>) -> Self {
        let value = std::sync::Arc::as_ptr(value) as *mut RBigInt;
        let slot = unsafe { std::ptr::addr_of_mut!((*value)._digits).cast::<majit_ir::GcRef>() };
        let registered = unsafe { majit_gc::gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for PendingPartsCacheDigitRoot {
    fn drop(&mut self) {
        if self.registered {
            majit_gc::gc_remove_root(self.slot);
        }
    }
}

// ---- walk_rbigint_cache_digit_slots ----
/// Visit the `_digits` GC slots held by the process-global formatter cache.
/// PyPy's module-global `_parts_cache` is part of the translated prebuilt root
/// graph; pyre's embedder adapts these raw slots to its `GcRef` root visitor.
pub fn walk_rbigint_cache_digit_slots(mut visitor: impl FnMut(&mut *mut u8)) {
    // rbigint.py's NULLRBIGINT / ONERBIGINT / ONENEGATIVERBIGINT /
    // FIVERBIGINT are translated prebuilt roots.  Do not initialize a
    // previously-unused constant from inside the collector; visit only slots
    // already published by ordinary execution.
    for slot in prebuilt_slots() {
        if let Some(&raw) = slot.get() {
            let value = unsafe { &mut *(raw as *mut RBigInt) };
            visitor(unsafe {
                &mut *(&mut value._digits as *mut *mut TypedItemsBlock as *mut *mut u8)
            });
        }
    }

    let all = PARTS_CACHE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for cache in all.iter().flatten() {
        let parts = cache
            .parts_cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for value in parts.iter() {
            // Every published snapshot is a monotonic extension and shares
            // these exact Arc<RBigInt> objects with older reader snapshots.
            // The collector runs this callback at STW, so forwarding the
            // shared object's `_digits` slot updates every reader without an
            // aliasing data race.
            let value = unsafe { &mut *(std::sync::Arc::as_ptr(value) as *mut RBigInt) };
            visitor(unsafe {
                &mut *(&mut value._digits as *mut *mut TypedItemsBlock as *mut *mut u8)
            });
        }
    }
}
