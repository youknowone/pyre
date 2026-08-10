pub use gcreftracer::{GcTable, install_gc_table_walker};
/// GC traits and interfaces for the JIT.
///
/// The GC subsystem provides:
/// 1. Object allocation (nursery bump-pointer + old gen)
/// 2. Write barrier insertion
/// 3. GC-aware IR rewriting (NEW_* → inline nursery allocation)
/// 4. Stack maps for compiled code
///
/// Reference: rpython/memory/gc/incminimark.py, rpython/jit/backend/llsupport/gc.py
use majit_ir::{Const, ConstMap, GcRef, Op};
pub use trace::{ClassTypeLayout, TypeEntry, TypeInfo, TypeInfoLayout};

mod address_dict;
pub mod collector;
pub mod gc_sync;
pub mod gcreftracer;
pub mod header;
pub mod hook;
pub mod hook_cell;
pub mod minimarkpage;
pub mod nursery;
pub mod oldgen;
pub mod rewrite;
pub mod rgil;
pub mod shadow_stack;
pub mod trace;
pub mod weakref;

/// GC flags stored in object headers.
///
/// From incminimark.py GCFLAG_* constants.
pub mod flags {
    // incminimark.py GCFLAG_* — bit positions must match RPython exactly.
    // first_gcflag = 1 << 32; each constant below is (first_gcflag << N)
    // expressed as the unshifted bit index N.
    /// GCFLAG_TRACK_YOUNG_PTRS (bit 0)
    pub const TRACK_YOUNG_PTRS: u64 = 1 << 0;
    /// GCFLAG_NO_HEAP_PTRS (bit 1)
    pub const NO_HEAP_PTRS: u64 = 1 << 1;
    /// GCFLAG_VISITED (bit 2)
    pub const VISITED: u64 = 1 << 2;
    /// GCFLAG_HAS_SHADOW (bit 3)
    pub const HAS_SHADOW: u64 = 1 << 3;
    /// GCFLAG_FINALIZATION_ORDERING (bit 4)
    pub const FINALIZATION_ORDERING: u64 = 1 << 4;
    /// GCFLAG_EXTRA (bit 5) — reserved
    pub const EXTRA: u64 = 1 << 5;
    /// GCFLAG_HAS_CARDS (bit 6)
    pub const HAS_CARDS: u64 = 1 << 6;
    /// GCFLAG_CARDS_SET (bit 7) — MSB of the byte containing TRACK_YOUNG_PTRS.
    /// The x86 backend relies on this being -0x80 as a signed byte.
    pub const CARDS_SET: u64 = 1 << 7;
    /// GCFLAG_VISITED_RMY (bit 8)
    pub const VISITED_RMY: u64 = 1 << 8;
    /// GCFLAG_PINNED (bit 9)
    pub const PINNED: u64 = 1 << 9;
    /// GCFLAG_IGNORE_FINALIZER (bit 10)
    pub const IGNORE_FINALIZER: u64 = 1 << 10;
    /// GCFLAG_SHADOW_INITIALIZED (bit 11)
    pub const SHADOW_INITIALIZED: u64 = 1 << 11;
    /// GCFLAG_DUMMY (bit 12)
    pub const DUMMY: u64 = 1 << 12;
    /// The object is already on a finalizer queue (bit 13).
    ///
    /// Not an incminimark flag — bit 13 is `_GCFLAG_FIRST_UNUSED`
    /// (incminimark.py:169) there, so this claims the first free bit and
    /// disturbs no RPython position. incminimark keeps the registered set
    /// purely in its two deques and never asks the question, because
    /// `register_finalizer` is contracted to be called at most once per
    /// object. That contract is checked only untranslated (`rgc.py:648-649`
    /// `assert not self._already_registered(obj)`); translated, a second
    /// registration silently appends a second deque entry and the finalizer
    /// runs again on the following major collection. This flag lets the
    /// queue enforce the contract for callers that cannot establish single
    /// registration statically.
    pub const FINALIZER_REGISTERED: u64 = 1 << 13;
}

/// Low-level trigger stored in an RPython finalizer handler.  It must only
/// schedule app-level work; finalizers themselves run after collection.
pub type FinalizerTriggerFn = fn();

/// True when the `gc_stress` test feature is compiled in: every allocation
/// may then run a full collection inside `alloc_with_type`, so JIT fast
/// paths that bypass it (inline nursery bump) must stay disabled or the
/// stress coverage silently shrinks to non-JIT allocations.
pub fn gc_stress_enabled() -> bool {
    cfg!(feature = "gc_stress")
}

/// `MAJIT_GC_LIFETIME_LOG` — trace remembered-set adds and old-gen frees.
///
/// Read once.  The gate sits in the write barrier and the old-gen sweep, and
/// `std::env::var_os` takes the environment lock and scans it linearly on every
/// call, so asking per event costs whether or not the variable is set.  Same
/// shape as `majit_metainterp::majit_log_enabled`.
pub fn gc_lifetime_log_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_GC_LIFETIME_LOG").is_some());
    *ENABLED
}

/// `MAJIT_LOG`, read once — the same gate `majit_metainterp::majit_log_enabled`
/// caches, for the collector's own per-collection sites.
pub fn majit_log_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("MAJIT_LOG").is_some());
    *ENABLED
}

/// `MAJIT_GC_DRAIN_CENSUS` — aggregate how much each minor collection drains.
///
/// A `PYPY_GC_NURSERY` sweep can show that run time barely moves across an 8x
/// range of nursery sizes without saying why.  Two very different causes have
/// that shape: the nursery is simply too small for the allocation rate, or
/// collections promote nearly everything they scan and therefore free nearly
/// nothing, so the next allocation collects again no matter how large the
/// nursery is.  The distinguishing number is the fraction of scanned bytes a
/// collection leaves behind, which is what this census records.
///
/// Read once — same shape as [`gc_lifetime_log_enabled`].
pub fn drain_census_enabled() -> bool {
    drain_census_dump_interval().is_some()
}

/// Set `MAJIT_GC_DRAIN_CENSUS` to a positive integer to also dump the running
/// summary every that many collections. The end-of-run summary is unreachable
/// for the runs this census is most needed on — a collection storm that has to
/// be killed rather than waited out — so those need the periodic line.
fn drain_census_dump_interval() -> Option<u64> {
    static INTERVAL: std::sync::LazyLock<Option<u64>> = std::sync::LazyLock::new(|| {
        let value = std::env::var_os("MAJIT_GC_DRAIN_CENSUS")?;
        Some(value.to_str().and_then(|v| v.parse().ok()).unwrap_or(0))
    });
    *INTERVAL
}

/// Survival-rate deciles: bucket `i` counts the collections that promoted
/// `[i*10, (i+1)*10)` percent of the bytes they found in the nursery, with a
/// fully-surviving collection landing in the last bucket.  Averages alone hide
/// the case this census exists to detect — a run split between cheap
/// collections and ones that copy the whole nursery.
const DRAIN_DECILES: usize = 10;

#[derive(Default)]
struct DrainCensus {
    minors: u64,
    nursery_size: usize,
    used_before: u64,
    promoted: u64,
    pinned: u64,
    deciles: [u64; DRAIN_DECILES],
}

static DRAIN_CENSUS: std::sync::Mutex<Option<DrainCensus>> = std::sync::Mutex::new(None);

/// Record one minor collection: the nursery bytes it found (`used_before`),
/// the bytes it copied out to the old generation (`promoted`), and how many
/// objects stayed behind pinned.
pub fn drain_census_record(
    used_before: usize,
    promoted: usize,
    pinned: usize,
    nursery_size: usize,
) {
    let Ok(mut guard) = DRAIN_CENSUS.lock() else {
        return;
    };
    let census = guard.get_or_insert_with(DrainCensus::default);
    census.minors += 1;
    census.nursery_size = nursery_size;
    census.used_before += used_before as u64;
    census.promoted += promoted as u64;
    census.pinned += pinned as u64;
    // A collection that found nothing drained nothing; count it as 0% rather
    // than dividing by zero.
    let percent = if used_before == 0 {
        0
    } else {
        // Widen before scaling: on wasm32 a nursery past ~43 MB makes
        // `used_before * 100` wrap a 32-bit `usize`, which buckets the sample
        // wrong in release and trips the overflow check in debug.
        ((promoted.min(used_before) as u64 * 100) / used_before as u64) as usize
    };
    census.deciles[(percent / 10).min(DRAIN_DECILES - 1)] += 1;
    if drain_census_dump_interval()
        .is_some_and(|interval| interval > 0 && census.minors.is_multiple_of(interval))
    {
        eprintln!("[gc-drain] {}", census.summary());
    }
}

impl DrainCensus {
    fn summary(&self) -> String {
        let minors = self.minors;
        let survived = if self.used_before == 0 {
            0.0
        } else {
            (self.promoted as f64 * 100.0) / self.used_before as f64
        };
        let deciles = self
            .deciles
            .iter()
            .map(|count| count.to_string())
            .collect::<Vec<_>>()
            .join("/");
        format!(
            "gc_drain minors={minors} nursery={} used_avg={} promoted_avg={} \
             freed_avg={} survived={survived:.1}% pinned_avg={:.2} deciles={deciles}",
            self.nursery_size,
            self.used_before / minors,
            self.promoted / minors,
            self.used_before.saturating_sub(self.promoted) / minors,
            self.pinned as f64 / minors as f64,
        )
    }
}

/// One-line summary of [`drain_census_record`], or a note that nothing was
/// recorded.
pub fn drain_census_summary() -> String {
    let Ok(guard) = DRAIN_CENSUS.lock() else {
        return "gc_drain <poisoned>".to_string();
    };
    let Some(census) = guard.as_ref().filter(|census| census.minors > 0) else {
        return "gc_drain minors=0".to_string();
    };
    census.summary()
}

/// Write barrier descriptor — information the JIT needs to emit write barrier checks.
///
/// From rpython/jit/backend/llsupport/gc.py WriteBarrierDescr.
#[derive(Debug, Clone)]
pub struct WriteBarrierDescr {
    /// gc.py:268: GCClass.JIT_WB_IF_FLAG
    pub jit_wb_if_flag: u64,
    /// gc.py:269: extract_flag_byte(jit_wb_if_flag) → byteofs
    /// Object-relative (negative = before object start, in header).
    pub jit_wb_if_flag_byteofs: i32,
    /// gc.py:269: extract_flag_byte(jit_wb_if_flag) → singlebyte
    pub jit_wb_if_flag_singlebyte: u8,
    /// gc.py:273: GCClass.JIT_WB_CARDS_SET (0 if no card marking)
    pub jit_wb_cards_set: u64,
    /// gc.py:274: GCClass.JIT_WB_CARD_PAGE_SHIFT
    pub jit_wb_card_page_shift: u32,
    /// gc.py:275: extract_flag_byte(jit_wb_cards_set) → byteofs
    pub jit_wb_cards_set_byteofs: i32,
    /// gc.py:275: extract_flag_byte(jit_wb_cards_set) → singlebyte
    /// Must equal -0x80 (signed) per gc.py:281 assert.
    pub jit_wb_cards_set_singlebyte: i8,
}

impl WriteBarrierDescr {
    /// gc.py:285-293 extract_flag_byte: find the non-zero byte in the
    /// header-shifted flag word and return (obj_relative_byteofs, singlebyte).
    ///
    /// The returned offset is relative to the **object pointer** (not the
    /// header), matching RPython's convention where the JIT emits
    /// `load [obj + byteofs]`.  Since our header sits at `obj - GcHeader::SIZE`,
    /// the conversion is `obj_ofs = header_ofs - GcHeader::SIZE`.
    pub fn extract_flag_byte(flag: u64) -> (i32, i8) {
        let shifted = flag << crate::header::FLAG_SHIFT;
        let bytes = shifted.to_le_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if b != 0 {
                let obj_ofs = i as i32 - crate::header::GcHeader::SIZE as i32;
                return (obj_ofs, b as i8);
            }
        }
        (0, 0)
    }

    /// Build a descriptor with correct byte offsets for the current
    /// header layout. gc.py:259-293 WriteBarrierDescr.__init__.
    pub fn for_current_gc() -> Self {
        let (if_flag_byteofs, if_flag_singlebyte) =
            Self::extract_flag_byte(flags::TRACK_YOUNG_PTRS);
        let (cards_set_byteofs, cards_set_singlebyte) = Self::extract_flag_byte(flags::CARDS_SET);
        // gc.py:280-281: the x86 backend relies on these two facts
        // to avoid one instruction in _write_barrier_fastpath.
        debug_assert_eq!(
            cards_set_byteofs, if_flag_byteofs,
            "CARDS_SET and TRACK_YOUNG_PTRS must be in the same byte"
        );
        debug_assert_eq!(
            cards_set_singlebyte, -0x80i8,
            "CARDS_SET must be the MSB of its byte (-0x80)"
        );
        WriteBarrierDescr {
            jit_wb_if_flag: flags::TRACK_YOUNG_PTRS,
            jit_wb_if_flag_byteofs: if_flag_byteofs,
            jit_wb_if_flag_singlebyte: if_flag_singlebyte as u8,
            jit_wb_cards_set: flags::CARDS_SET,
            jit_wb_card_page_shift: crate::collector::DEFAULT_CARD_PAGE_SHIFT,
            jit_wb_cards_set_byteofs: cards_set_byteofs,
            jit_wb_cards_set_singlebyte: cards_set_singlebyte,
        }
    }
}

/// GC allocator interface.
///
/// Provides allocation and collection primitives.
/// One `incminimark.py:810-822 collect_step` state transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GcStepTransition {
    pub old_state: u8,
    pub new_state: u8,
}

impl GcStepTransition {
    pub const SCANNING: u8 = 0;
    pub const MARKING: u8 = 1;
    pub const SWEEPING: u8 = 2;
    pub const FINALIZING: u8 = 3;

    /// `rgc.py:52-60 is_done__states`: a major collection is done when the
    /// step ended in the starting state *and* did not begin there. A step that
    /// found no work reports `(SCANNING, SCANNING)`, which completes nothing.
    pub const fn is_done(self) -> bool {
        self.old_state != Self::SCANNING && self.new_state == Self::SCANNING
    }
}

/// `incminimark.py:3128-3154 get_stats` values owned by the collector.
/// Every byte count includes the nursery where upstream's corresponding
/// `rgc.get_stats` selector does.  JIT assembler accounting is deliberately
/// outside this struct, as it is upstream (`jit_hooks.stats_asmmemmgr_*`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcMemoryStats {
    pub total_gc_memory: usize,
    pub total_allocated_memory: usize,
    pub peak_memory: usize,
    pub peak_allocated_memory: usize,
    pub total_arena_memory: usize,
    pub total_rawmalloced_memory: usize,
    pub peak_arena_memory: usize,
    pub peak_rawmalloced_memory: usize,
    pub nursery_size: usize,
    pub total_gc_time_ms: usize,
}

pub trait GcAllocator: Send {
    fn debug_validate_oldgen_freeblocks(&self, _site: &str) {}

    /// Allocate a fixed-size object in the nursery.
    fn alloc_nursery(&mut self, size: usize) -> GcRef;

    /// Allocate a HEADERLESS nursery object of exactly `size` bytes and return
    /// the RAW base (no GcHeader, no type word), matching the JIT headerless
    /// fast-path contract. Only a headerless-aware GC box may serve this.
    /// `MiniMarkGC` is a headered collector: its nursery walk reads a GcHeader
    /// at `obj - GcHeader::SIZE`, which a headerless object lacks, so it must
    /// never serve a headerless allocation. The default panics to turn a future
    /// misconfiguration (an interpreter that declares `headerless_structs` while
    /// leaving MiniMarkGC as the active dynasm GC) into a loud failure instead of
    /// a silently mis-based pointer + untraceable object.
    fn alloc_nursery_headerless(&mut self, _size: usize) -> GcRef {
        panic!(
            "alloc_nursery_headerless called on a headerless-unaware GcAllocator \
             (e.g. MiniMarkGC); headerless nursery allocation requires a \
             headerless-aware active dynasm GC box"
        );
    }

    /// [`Self::alloc_nursery_headerless`] for a caller that cannot survive a
    /// collection: the metainterp's jitcode tracer executes `NEW` while holding
    /// raw object pointers in its own register bank, which is not part of any
    /// root set, so a moving collection under this call would strand them. The
    /// allocator must grow instead of evacuating. The default forwards to the
    /// collecting form, which is correct for a non-moving collector.
    ///
    /// The forward is not itself a guard, so the obligation is on the
    /// implementor: a MOVING headerless-aware allocator must override this
    /// rather than inherit the forward. No current one does, and the two shapes
    /// that reach here are both safe — [`MiniMarkGC`](crate::collector::MiniMarkGC)
    /// moves but is headered, so `alloc_nursery_headerless`'s own default panics
    /// before any collection; the headerless boxes the backends install grow
    /// their pool instead of evacuating, which is exactly what this method asks
    /// for. Adding a moving headerless box without an override would reintroduce
    /// the silent stranding.
    fn alloc_nursery_headerless_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_nursery_headerless(size)
    }

    /// Allocate a fixed-size object with a known GC type id.
    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let _ = type_id;
        self.alloc_nursery(size)
    }

    /// Allocate a fixed-size typed object, collecting the nursery when full,
    /// while keeping one caller-owned GC slot live across the slow path.
    ///
    /// RPython's shadow-stack/stack-map transform exposes such a local only
    /// when the allocation reaches `collect_and_reserve`.  The default keeps
    /// the slot registered around the whole allocation for collectors without
    /// a nursery fast-path override. MiniMark overrides this so a successful
    /// bump allocation performs no dynamic root registration.
    ///
    /// # Safety
    /// `root` must point to a valid mutable `GcRef` slot for the duration of
    /// this call. `needs_write_barrier` must point to a valid mutable `bool`
    /// slot; the allocator writes whether initializing the result with `root`
    /// requires an old-to-young creation barrier.
    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        // Collectors without placement information keep the conservative
        // creation barrier. MiniMark overrides this and clears the flag for
        // the ordinary nursery result.
        unsafe { *needs_write_barrier = true };
        unsafe { self.add_root(root) };
        let result = self.alloc_nursery_typed(type_id, size);
        self.remove_root(root);
        result
    }

    /// [`Self::alloc_nursery_collecting_typed_rooted`] for a type that carries
    /// neither a finalizer nor the weakref flag.
    ///
    /// `gct_fv_gc_malloc` (`framework.py:820-838`) resolves both properties at
    /// transformation time and calls `malloc_fast` — the `inline=True` copy of
    /// `malloc_fixedsize` annotated `s_False, s_False, s_False`
    /// (`framework.py:361-382`) — whenever they are both false, which is the
    /// case for every fixed-size malloc of a plain struct. The default forwards
    /// to the general form, which is what a collector that does not distinguish
    /// the two bodies has.
    ///
    /// # Safety
    /// Same contract as [`Self::alloc_nursery_collecting_typed_rooted`], plus
    /// `type_id` must name a type with no destructor and no weakref flag.
    unsafe fn alloc_fast_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.alloc_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
        }
    }

    /// Allocate a fixed-size object without triggering collection.
    ///
    /// Implementations may fall back to old-gen allocation when the nursery
    /// cannot satisfy the request.
    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef;

    /// Allocate a variable-size object (array/string).
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef;

    /// Allocate a variable-size object with a known GC type id.
    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let _ = type_id;
        self.alloc_varsize(base_size, item_size, length)
    }

    /// Allocate a fixed-size object with type id without triggering collection.
    ///
    /// Falls back to old-gen when nursery is full. Used for jitframe
    /// allocation where input refs on the Rust stack are not yet protected
    /// by the shadow stack (Rust stack is not traced by GC, unlike RPython
    /// stack where `lltype.malloc` can safely trigger GC).
    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let _ = type_id;
        self.alloc_nursery_no_collect(size)
    }

    /// Fallible host-side form of `alloc_nursery_no_collect_typed`.
    ///
    /// MiniMark overrides this so rawmalloc failure is returned as NULL.
    /// Collectors without a distinct fallible path retain their existing
    /// allocation behavior.
    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_nursery_no_collect_typed(type_id, size)
    }

    /// [`Self::try_alloc_nursery_no_collect_typed`] for a type that carries
    /// neither a finalizer nor the weakref flag: `malloc_fast`
    /// (`framework.py:361-382`).
    ///
    /// # Safety
    /// `type_id` must name a type with no destructor and no weakref flag.
    unsafe fn try_alloc_fast_nursery_no_collect_typed(
        &mut self,
        type_id: u32,
        size: usize,
    ) -> GcRef {
        self.try_alloc_nursery_no_collect_typed(type_id, size)
    }

    /// Fallible no-collect allocation that also reports whether initializing
    /// the fresh result with a young GC reference needs a creation barrier.
    ///
    /// Collectors without placement information conservatively report
    /// `true`. MiniMark clears the flag for an ordinary nursery bump and keeps
    /// it set when the allocation spills to old-gen.
    ///
    /// # Safety
    /// `needs_write_barrier` must point to a valid mutable `bool` slot for the
    /// duration of this call.
    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe { *needs_write_barrier = true };
        self.try_alloc_nursery_no_collect_typed(type_id, size)
    }

    /// [`Self::try_alloc_nursery_no_collect_typed_with_placement`] for a type
    /// that carries neither a finalizer nor the weakref flag: `malloc_fast`
    /// (`framework.py:361-382`).
    ///
    /// # Safety
    /// Same contract as
    /// [`Self::try_alloc_nursery_no_collect_typed_with_placement`], plus
    /// `type_id` must name a type with no destructor and no weakref flag.
    unsafe fn try_alloc_fast_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        unsafe {
            self.try_alloc_nursery_no_collect_typed_with_placement(
                type_id,
                size,
                needs_write_barrier,
            )
        }
    }

    /// Allocate a variable-size object without triggering collection.
    ///
    /// Implementations may fall back to old-gen allocation when the nursery
    /// cannot satisfy the request.
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef;

    /// Allocate a stable-address object directly in old-gen.
    ///
    /// Used by host-side allocators (e.g. pyre-object `w_int_new`
    /// non-cached path) that return a raw pointer the caller holds on
    /// the Rust stack before it can be stored into a GC-tracked slot.
    /// Old-gen objects never move in MiniMark mark-sweep collection,
    /// so a subsequent minor collection cannot invalidate the pointer.
    ///
    /// Default implementation routes to
    /// `alloc_nursery_no_collect_typed` so backends without a
    /// distinct old-gen still compile; backends with a real old-gen
    /// override to force placement there.
    ///
    /// Non-collecting, unlike its structural counterpart
    /// `external_malloc` (incminimark.py:987-994), which tests
    /// `threshold_reached(raw_malloc_usage(totalsize))` before allocating and
    /// drives `minor_collection_with_major_progress` when it holds. That check
    /// cannot be made here: an RPython caller's locals are shadow-stack roots,
    /// so upstream may collect mid-allocation, while the callers this entry
    /// exists for are holding the raw pointer on the Rust stack precisely
    /// because it is not a root. Old-gen growth is instead answered by the
    /// interpreter safepoint's `threshold_reached` poll, which runs where the
    /// root set is known (`pyre-object`'s `gc_interp::safepoint`).
    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_nursery_no_collect_typed(type_id, size)
    }

    /// incminimark.py:1569: jit_remember_young_pointer(obj)
    /// Perform a write barrier check on `obj`.
    /// Must be called before storing a GC reference into `obj`.
    fn write_barrier(&mut self, obj: GcRef);

    /// Write barrier for an object the caller has already proved belongs to
    /// this collector.  Fresh results from `alloc_oldgen_typed` and the
    /// collector's no-collect allocator satisfy this contract.  Collectors
    /// may skip their defensive hybrid-heap membership query; the default
    /// preserves the safe entry point for backends without that distinction.
    fn write_barrier_managed(&mut self, obj: GcRef) {
        self.write_barrier(obj);
    }

    /// incminimark.py:1606 jit_remember_young_pointer_from_array:
    /// Called by JIT when TRACK_YOUNG_PTRS set but CARDS_SET not.
    /// Tries to set CARDS_SET if HAS_CARDS; else generic barrier.
    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef);

    /// incminimark.py:1557 remember_young_pointer_from_array2:
    /// Full card-marking barrier with index. Called when marking a
    /// specific card after CARDS_SET is already established.
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    );

    /// Trigger a minor (nursery) collection.
    fn collect_nursery(&mut self);

    /// Trigger a full collection.
    fn collect_full(&mut self);

    /// `incminimark.py:810-822 collect_step`: perform one minor collection
    /// and exactly one major-collection state transition, independently of
    /// the automatic-collection enabled flag.
    ///
    /// The default is `rgc.py:20-31`'s non-incremental answer: a collector
    /// with no state machine does the whole collection at once and reports
    /// `_encode_states(1, 0)`, a transition [`GcStepTransition::is_done`]
    /// accepts. Reporting the starting state on both sides would instead say
    /// the collection is still in progress, so a caller stepping until done
    /// would run a full collection on every iteration and never stop.
    fn collect_step(&mut self) -> GcStepTransition {
        self.collect_full();
        GcStepTransition {
            old_state: GcStepTransition::MARKING,
            new_state: GcStepTransition::SCANNING,
        }
    }

    /// `rpython/rlib/rgc.py:1224 do_get_objects`: walk every object reachable
    /// from the GC roots and visit the `rclass.OBJECT` instances selected by
    /// the generation argument. The visitor runs before the collector releases
    /// its inspection pause, so callers can root every returned object without
    /// exposing unrooted raw addresses to another collection. Collectors
    /// without an inspector visit no objects.
    fn get_objects(&mut self, _generation: i8, _visitor: &mut dyn FnMut(GcRef)) {}

    /// `pypy/module/gc/referents.py:53-78 _list_w_obj_referents`: visit the
    /// app-level objects `obj` refers to directly, expanding the
    /// interpreter-internal structs in between. Same rooting contract as
    /// [`GcAllocator::get_objects`]. Collectors without an inspector visit no
    /// objects.
    fn get_referents(&mut self, _obj: GcRef, _visitor: &mut dyn FnMut(GcRef)) {}

    /// Whether the collector traverses references out of `obj` — what
    /// `gc.is_tracked` reports. Collectors without an inspector track nothing.
    fn is_tracked(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// `inspector.py:get_rpy_memory_usage`: size of the object's translated
    /// payload, excluding the GC header and anything reachable from it.
    /// `None` is the untranslated equivalent of inspector.py returning `-1`
    /// for a collector that does not implement the operation.
    fn get_rpy_memory_usage(&mut self, _obj: GcRef) -> Option<usize> {
        None
    }

    /// `inspector.py:get_rpy_type_index`: positive member index in the
    /// translated type-info group.  Index zero is deliberately unused by
    /// RPython's `TypeLayoutBuilder.make_type_info_group`.
    fn get_rpy_type_index(&mut self, _obj: GcRef) -> Option<usize> {
        None
    }

    /// `inspector.py:get_rpy_roots`: visit the collector's raw roots without
    /// expanding interpreter-internal objects.  `false` means the collector
    /// does not implement the inspector operation.
    fn get_rpy_roots(&mut self, _visitor: &mut dyn FnMut(GcRef)) -> bool {
        false
    }

    /// `inspector.py:get_rpy_referents`: visit the raw pointers traced from
    /// one object, without the app-level expansion performed by
    /// [`GcAllocator::get_referents`].
    fn get_rpy_referents(&mut self, _obj: GcRef, _visitor: &mut dyn FnMut(GcRef)) -> bool {
        false
    }

    /// `inspector.py:265-272 dump_rpy_heap`. The fd receives native signed
    /// machine words; `Ok(false)` denotes a collector without an inspector.
    fn dump_rpy_heap(&mut self, _fd: i32) -> Result<bool, i32> {
        Ok(false)
    }

    /// Translation-time type-info metadata returned by `get_typeids_z` and
    /// `get_typeids_list`. `None` denotes a collector without this operation.
    fn get_typeids_text(&self) -> Option<Vec<u8>> {
        None
    }

    fn get_typeids_list(&self) -> Option<Vec<usize>> {
        None
    }

    /// `incminimark.raw_malloc_memory_pressure(size, adr)`. When `object` is
    /// non-null, the translated `special_memory_pressure` field on that
    /// object is set to `size` before the collection threshold is adjusted.
    fn add_memory_pressure(&mut self, _size: isize, _object: GcRef) {}

    /// `inspector.count_memory_pressure`: sum the pressure fields on all
    /// root-reachable objects. Collectors without translated type metadata
    /// report zero.
    fn total_memory_pressure(&mut self) -> isize {
        0
    }

    /// Whether a raw inspector node is an app-level `W_Root` rather than an
    /// interpreter-internal GC struct that must be wrapped in `GcRef`.
    fn is_app_level_object(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// Trigger a non-moving old-gen-only major collection (sweep dead old-gen
    /// objects without moving the nursery). The default no-ops so a backend
    /// with no incremental old-gen lacks no method; `MiniMarkGC` overrides it.
    fn collect_oldgen_nonmoving(&mut self) {}

    /// Toggle automatic major-collection progress. Backends without
    /// incremental major collection ignore it.
    fn enable(&mut self) {}

    fn disable(&mut self) {}

    fn isenabled(&self) -> bool {
        true
    }

    /// rgc.py `FinalizerQueue.register_finalizer` /
    /// framework.py `gc_fq_register`: register `obj` with one translated
    /// finalizer handler.  Backends without finalizer queues ignore it.
    fn register_finalizer(&mut self, _fq_index: usize, _obj: GcRef, _trigger: FinalizerTriggerFn) {}

    /// rgc.py `FinalizerQueue.next_dead` / framework.py `gc_fq_next_dead`.
    fn finalizer_next_dead(&mut self, _fq_index: usize) -> Option<GcRef> {
        None
    }

    /// minimark.py:1900-1915 `id_or_identityhash(gcobj)`.
    /// Return a stable address for the object that does not change
    /// across GC moves.  For nursery objects, allocates a shadow in
    /// old-gen and returns its address.  For old-gen objects, returns
    /// the object's own address.
    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        obj_addr
    }

    /// `gc.py:401 self.write_barrier_descr = WriteBarrierDescr(self)`:
    /// the descriptor for the write barrier check. Defaulting to `None` is
    /// `gc.py:156 GcLLDescr_boehm.write_barrier_descr = None` — a collector
    /// that needs no write barrier. Upstream reads the attribute directly and
    /// has no `get_write_barrier_descr` accessor.
    fn get_write_barrier_descr(&self) -> Option<WriteBarrierDescr> {
        None
    }

    /// Register a stack/root slot that contains a `GcRef`.
    ///
    /// The pointer must remain valid until removed. Backends use this to
    /// expose shadow-root buffers around collecting helper calls.
    ///
    /// # Safety
    /// The caller must ensure the slot remains valid for the duration of the
    /// registration.
    unsafe fn add_root(&mut self, _root: *mut GcRef) {}

    /// Remove a previously-registered root slot.
    fn remove_root(&mut self, _root: *mut GcRef) {}

    /// Whether `addr` lies inside this GC's managed heap (nursery or
    /// old-gen). Used by host-side allocators to discriminate
    /// GC-allocated blocks from `std::alloc`-backed ones during the
    /// L1/L2 stepping-stone window — `dealloc_items_block` must
    /// early-return for GC-managed pointers (the GC sweeps them) and
    /// fall through to `std::alloc::dealloc` for non-managed ones.
    /// Default `false` matches stub allocators (no managed heap).
    fn is_managed_heap_object(&self, _addr: usize) -> bool {
        false
    }

    /// Whether `addr` is a live object in the moving nursery.  Custom trace
    /// hooks use this to distinguish a nursery child whose own type walker
    /// will scan its items from an old-gen child whose owner must scan an
    /// off-barrier payload during a minor collection.
    fn is_nursery_object(&self, _addr: usize) -> bool {
        false
    }

    /// The nursery's `[start, end)`, when it is a fixed range for the life of
    /// this allocator.  `None` (the default) keeps callers on the full
    /// `is_nursery_object` query.
    fn nursery_bounds(&self) -> Option<(usize, usize)> {
        None
    }

    /// `gc/base.py:380-383 is_valid_gc_object`'s tagged-immediate setting
    /// (`translationoption.py:185 taggedpointers`, default off).
    fn taggedpointers(&self) -> bool {
        false
    }

    /// Current nursery free pointer.
    fn nursery_free(&self) -> *mut u8;

    /// gc.py:525-531 get_nursery_free_addr parity.
    /// Address of the mutable nursery_free field that JIT code updates.
    fn nursery_free_addr(&self) -> usize;

    /// Nursery top (end) pointer.
    fn nursery_top(&self) -> *const u8;

    /// gc.py:525-531 get_nursery_top_addr parity.
    /// Address of the published nursery_top slot that JIT code reads.
    fn nursery_top_addr(&self) -> usize;

    /// Maximum size for nursery allocation (larger objects go to old gen directly).
    fn max_nursery_object_size(&self) -> usize;

    /// incminimark.py: card_page_indices → JIT_WB_CARD_PAGE_SHIFT.
    /// Log2 of the card page size. 0 if card marking is disabled.
    fn card_page_shift(&self) -> u32 {
        0
    }

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Adds the object directly to the remembered set. The JIT has already
    /// performed the inline flag test (COND_CALL_GC_WB) and determined
    /// that the barrier is needed.
    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.write_barrier(obj);
    }

    /// Whether the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT emits COND_CALL_GC_WB (inline flag test +
    /// conditional call) instead of a full barrier call.
    fn can_optimize_cond_call(&self) -> bool {
        false
    }

    /// Perform one incremental GC step at a JIT safepoint.
    /// Returns true if any GC work was done.
    fn gc_step(&mut self) -> bool {
        false
    }

    /// Free memory associated with invalidated JIT compiled code.
    fn jit_free(&mut self, _code_ptr: usize, _size: usize) {}

    /// Pin a nursery object so it won't move during minor collection.
    /// Returns true if pinning succeeded.
    fn pin(&mut self, _obj: GcRef) -> bool {
        false
    }

    /// Unpin a previously pinned object.
    fn unpin(&mut self, _obj: GcRef) {}

    /// Check if an object is pinned.
    fn is_pinned(&self, _obj: GcRef) -> bool {
        false
    }

    /// Register a GC type descriptor and return its type id.
    ///
    /// RPython parity: `rgc.register_custom_trace_hook(TYPE, trace_fn)`.
    fn register_type(&mut self, _info: TypeInfo) -> u32 {
        0
    }

    /// Number of registered GC types.
    fn type_count(&self) -> usize {
        0
    }

    /// Diagnostic only: `(oldgen_total_bytes, nursery_used_bytes)`.
    /// `oldgen_total_bytes` is `get_total_memory_used` (promoted + raw/large
    /// old-gen objects, NOT the nursery); `nursery_used_bytes` is the current
    /// nursery bump-pointer fill. Used to split GC-retained memory from
    /// host-heap allocations when diagnosing growth. Default `(0, 0)` for stub
    /// allocators with no byte accounting.
    fn heap_byte_stats(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Full incminimark memory/time statistics. Stub allocators report zeros.
    fn gc_memory_stats(&self) -> GcMemoryStats {
        GcMemoryStats::default()
    }

    /// incminimark.py:1288-1290 `threshold_reached(0)`: whether the memory the
    /// collector is responsible for has caught up to the threshold it set for
    /// the next major collection. Everything that shapes that threshold — the
    /// growth ratio, `growth_rate_max`, `max_delta`, `min_heap_size`,
    /// `max_heap_size` — lives behind this answer, so a caller that cannot
    /// drive collection from the allocator can ask here instead of modelling
    /// heap growth itself. Default `false` for stub allocators with no
    /// threshold accounting.
    fn major_threshold_reached(&self) -> bool {
        false
    }

    /// Diagnostic only: `(minor_collections, major_collections)` run so far.
    /// Used to attribute run time to collection cadence (e.g. old-gen churn
    /// driving repeated majors). Default `(0, 0)` for stub allocators.
    fn collection_counts(&self) -> (usize, usize) {
        (0, 0)
    }

    /// Whether a JIT inline nursery bump of `type_id` is equivalent to
    /// `alloc_with_type`'s fast path: the type registers no destructor and is
    /// not a weakref (either would need a side-list push at allocation, i.e.
    /// the slow path). Mirrors rewrite.py's malloc fast-path eligibility
    /// (types with finalizers/weakrefs keep the call). Default `false` so
    /// stub allocators keep the helper path.
    fn type_alloc_is_plain(&self, _type_id: u32) -> bool {
        false
    }

    /// Look up the fixed-object size for a registered GC type.
    ///
    /// RPython parity: this matches `cpu.bh_new(typedescr)` reading
    /// `typedescr.size` (llmodel.py / descr.py).  Default `None` keeps
    /// stub allocators (e.g. wasm/dynasm) from claiming knowledge.
    fn type_size(&self, _type_id: u32) -> Option<usize> {
        None
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Maps a vtable pointer to its registered GC type id. RPython
    /// computes this arithmetically from the GC type_info_group base
    /// (gc.py:584-589); pyre's GC keeps an explicit vtable→type_id table
    /// because pyre frontends register vtables independently from the
    /// translator pipeline.
    ///
    /// Default `None` matches a GC layer with no installed mapping
    /// (e.g. dynasm/wasm stubs). The cmp_guard_class fallback panics
    /// instead of silently producing wrong code.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, _classptr: usize) -> Option<u32> {
        None
    }

    /// Register a vtable pointer as the canonical class for a type id.
    /// Frontends call this once per type after `register_type`, mirroring
    /// how RPython's translator emits the vtable→typeid pair into the
    /// GC type_info_group.
    fn register_vtable_for_type(&mut self, _vtable: usize, _type_id: u32) {}

    /// `gctypelayout.encode_type_shapes_now` parity
    /// (gctypelayout.py:393-398): closes the type-registration phase.
    /// After freeze, `register_type` is forbidden, the
    /// `type_info_group` base address is stable, and every
    /// `is_object` type's `subclassrange_{min,max}` reflects the
    /// preorder of its inheritance chain (`assign_inheritance_ids`,
    /// rtyper/normalizecalls.py:373-389).
    ///
    /// Backends call this from `set_gc_allocator` so the embedded
    /// codegen-time pointers and bounds are immutable thereafter.
    /// Default no-op for stub allocators with no type table.
    fn freeze_types(&mut self) {}

    /// llsupport/gc.py:162 / gc.py:318 `supports_guard_gc_type` flag.
    /// `GcLLDescr_boehm` sets it to `False`; `GcLLDescr_framework` sets
    /// it to `True`. Relayed to `cpu.supports_guard_gc_type` via
    /// `llmodel.py:63`. Gates the backend's `genop_guard_guard_gc_type`,
    /// `genop_guard_guard_is_object`, and `genop_guard_guard_subclass`
    /// (x86/assembler.py:1896, 1925, 1946 `assert`) and
    /// `ConstPtrInfo.get_known_class(cpu)` at info.py:766. The default
    /// `false` matches `AbstractCPU.supports_guard_gc_type` in
    /// `rpython/jit/backend/model.py:21` and keeps backends without an
    /// installed TYPE_INFO table from emitting the guards.
    fn supports_guard_gc_type(&self) -> bool {
        false
    }

    /// llsupport/gc.py:631-642 `check_is_object` parity. Reads the
    /// typeid for `gcref` (gc.py:623-629 `get_actual_typeid`) and
    /// returns whether that type has `rclass.OBJECT` layout — i.e.
    /// whether `T_IS_RPYTHON_INSTANCE` is set in its infobits (gc.py:
    /// 631-642 walks the TYPE_INFO table to test that bit).
    ///
    /// Exposed on `cpu.check_is_object(gcptr)` via llmodel.py:541-546,
    /// which asserts `supports_guard_gc_type` before delegating. The
    /// optimizer consults this through info.py:766 inside
    /// `ConstPtrInfo.get_known_class(cpu)` to decide whether reading
    /// offset 0 of a constant gcref is safe.
    ///
    /// Returns `false` for null pointers and for backends without a
    /// type registry (matching `GcLLDescr_boehm`, which does not
    /// define `check_is_object`).
    fn check_is_object(&self, _gcref: GcRef) -> bool {
        false
    }

    /// gc/base.py:380-383 `is_valid_gc_object` tagged-immediate test:
    /// `config.taggedpointers && (addr & 1 == 1)`. Backends with no
    /// tagged-immediate support (the default) return `false`.
    fn is_tagged_immediate(&self, _addr: usize) -> bool {
        false
    }

    /// `rpython/rlib/rgc.py:229` `can_move(p)` — whether the GC object
    /// `gcref` sits at an address that may still move. "With non-moving
    /// GCs, it is always False; with moving GCs it can be True for some
    /// time, then False once the object is sure not to move." The default
    /// is `false`, matching a non-moving GC (and the no-GC case).
    fn can_move(&self, _gcref: GcRef) -> bool {
        false
    }

    /// llsupport/gc.py:592 `get_translated_info_for_typeinfo`.
    /// Returns `(type_info_group_base, shift_by, sizeof_ti)`:
    ///  * `type_info_group_base` — base address of the `TYPE_INFO` table
    ///    (`llop.gc_get_type_info_group`).
    ///  * `shift_by` — `2` on 32-bit, `0` on 64-bit (gc.py:596-599).
    ///  * `sizeof_ti` — `rffi.sizeof(GCData.TYPE_INFO)`.
    /// Called by `genop_guard_guard_is_object` (x86/assembler.py:1934)
    /// and `genop_guard_guard_subclass` (x86/assembler.py:1965).
    ///
    /// Default panics to match RPython: `GcLLDescr_boehm` does not
    /// define the method, and calling it when
    /// `supports_guard_gc_type = False` is a precondition violation.
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        panic!(
            "GcAllocator::get_translated_info_for_typeinfo called but the \
             GC has not installed a TYPE_INFO layout (see llsupport/gc.py:\
             592); callers must first check supports_guard_gc_type"
        )
    }

    /// llsupport/gc.py:619 `get_translated_info_for_guard_is_object`.
    /// Returns `(infobits_offset, T_IS_RPYTHON_INSTANCE_BYTE)` used by
    /// `genop_guard_guard_is_object` to locate the `infobits` byte in
    /// the `TYPE_INFO` entry and the bitmask for the
    /// `T_IS_RPYTHON_INSTANCE` flag.
    ///
    /// Default panics — same rationale as
    /// `get_translated_info_for_typeinfo`.
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        panic!(
            "GcAllocator::get_translated_info_for_guard_is_object called \
             but the GC has not installed a TYPE_INFO layout (see \
             llsupport/gc.py:619); callers must first check \
             supports_guard_gc_type"
        )
    }

    /// x86/assembler.py:1951 `cpu.subclassrange_min_offset`.
    /// Byte offset of the `subclassrange_min` field inside
    /// `rclass.CLASSTYPE`. `genop_guard_guard_subclass` uses it twice:
    /// once to read the subclassrange minimum from the object's
    /// vtable (x86/assembler.py:1956) and once to locate the same
    /// field inside a `TYPE_INFO` entry (x86/assembler.py:1968-1969).
    ///
    /// Default panics — same rationale as the other TYPE_INFO helpers.
    fn subclassrange_min_offset(&self) -> usize {
        panic!(
            "GcAllocator::subclassrange_min_offset called but the GC has \
             not installed an rclass.CLASSTYPE layout (see x86/\
             assembler.py:1951); callers must first check \
             supports_guard_gc_type"
        )
    }

    /// x86/assembler.py:1971-1974 bounds lookup at codegen time:
    ///     vtable_ptr = loc_check_against_class.getint()
    ///     vtable_ptr = rffi.cast(rclass.CLASSTYPE, vtable_ptr)
    ///     check_min = vtable_ptr.subclassrange_min
    ///     check_max = vtable_ptr.subclassrange_max
    /// Returns `(subclassrange_min, subclassrange_max)` for the class
    /// whose pointer is given, or `None` if no entry exists.
    ///
    /// Default `None` keeps backends without an installed
    /// `rclass.CLASSTYPE` layout from emitting a wrong bounds check;
    /// `genop_guard_guard_subclass` callers panic loudly when the
    /// lookup misses.
    fn subclass_range(&self, _classptr: usize) -> Option<(i64, i64)> {
        None
    }

    /// Companion to `subclass_range` keyed by typeid instead of
    /// classptr. Used by the executor's `GuardSubclass` arm after it
    /// resolves `value.typeptr` via `get_actual_typeid`
    /// (llgraph/runner.py:1271-1281). Default `None`.
    fn typeid_subclass_range(&self, _typeid: u32) -> Option<(i64, i64)> {
        None
    }

    /// gc.py:624-629 `get_actual_typeid` parity. Reads the typeid
    /// from the GC header half-word for managed objects, or resolves
    /// the foreign object's classptr through `vtable_to_type_id` for
    /// backends that register a seam (e.g. pyre's PyObject layout).
    /// Default `None` for stubs without a type registry.
    fn get_actual_typeid(&self, _gcref: GcRef) -> Option<u32> {
        None
    }

    /// Companion to `check_is_object` keyed by typeid. Returns
    /// whether the typeid carries `T_IS_RPYTHON_INSTANCE` in its
    /// TYPE_INFO entry (gctypelayout.py:642). Default `None`.
    fn typeid_is_object(&self, _typeid: u32) -> Option<bool> {
        None
    }
}

/// Forwarding handle to the process-global GC singleton via `gc_sync`.
///
/// `&mut self` methods route through `gc_sync::gc_op`; `&self` read-only
/// queries route through `gc_sync::gc_query_reentrant` so they stay correct
/// when a collection-time extra-root walker re-enters the GC (ownership / type
/// queries) while this thread already holds the `&mut`, which a `&mut` path
/// would alias. No raw pointer at this layer.
///
/// # Thread safety
///
/// Exclusion comes from the GIL (`rgil`), which the caller holds for
/// as long as it runs pyre code, so these methods take no lock of their own.
/// Collection uses STW safepoint protocol (`gc_sync::request_stw`).
/// See gh#396 for the full free-threading GC design.
pub struct GcHandle;

// GcHandle is a zero-size marker; Send is trivially safe.
unsafe impl Send for GcHandle {}

impl GcAllocator for GcHandle {
    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery(size))
    }
    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_typed(type_id, size))
    }
    unsafe fn alloc_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.alloc_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
        })
    }
    unsafe fn alloc_fast_nursery_collecting_typed_rooted(
        &mut self,
        type_id: u32,
        size: usize,
        root: *mut GcRef,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.alloc_fast_nursery_collecting_typed_rooted(type_id, size, root, needs_write_barrier)
        })
    }
    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_no_collect(size))
    }
    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize(base_size, item_size, length))
    }
    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize_typed(type_id, base_size, item_size, length))
    }
    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_nursery_no_collect_typed(type_id, size))
    }
    fn try_alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.try_alloc_nursery_no_collect_typed(type_id, size))
    }
    unsafe fn try_alloc_fast_nursery_no_collect_typed(
        &mut self,
        type_id: u32,
        size: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe { gc.try_alloc_fast_nursery_no_collect_typed(type_id, size) })
    }
    unsafe fn try_alloc_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.try_alloc_nursery_no_collect_typed_with_placement(type_id, size, needs_write_barrier)
        })
    }
    unsafe fn try_alloc_fast_nursery_no_collect_typed_with_placement(
        &mut self,
        type_id: u32,
        size: usize,
        needs_write_barrier: *mut bool,
    ) -> GcRef {
        gc_sync::gc_op(|gc| unsafe {
            gc.try_alloc_fast_nursery_no_collect_typed_with_placement(
                type_id,
                size,
                needs_write_barrier,
            )
        })
    }
    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_varsize_no_collect(base_size, item_size, length))
    }
    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        gc_sync::gc_op(|gc| gc.alloc_oldgen_typed(type_id, size))
    }
    fn write_barrier(&mut self, obj: GcRef) {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.write_barrier(obj))
    }
    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.jit_remember_young_pointer_from_array(obj))
    }
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        gc_sync::gc_op_with_root(obj, |gc, obj| {
            gc.remember_young_pointer_from_array2(obj, index, card_page_shift)
        })
    }
    fn collect_nursery(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_nursery())
    }
    fn collect_full(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_full())
    }
    fn collect_step(&mut self) -> GcStepTransition {
        gc_sync::gc_op(|gc| gc.collect_step())
    }
    fn get_objects(&mut self, generation: i8, visitor: &mut dyn FnMut(GcRef)) {
        gc_sync::gc_op(|gc| gc.get_objects(generation, visitor))
    }
    fn get_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) {
        // Rooted like `write_barrier`: a contended `gc_op` parks this mutator,
        // so another thread's minor collection can forward `obj` while the call
        // waits. Reading the raw argument afterwards would hand the collector a
        // forwarding stub, which `do_get_referents` rejects — reporting a live
        // object as having no referents at all.
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_referents(obj, visitor))
    }
    fn is_tracked(&mut self, obj: GcRef) -> bool {
        // Same exposure as `get_referents`: a forwarded address is not a
        // managed-heap object of the type the answer is read out of.
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.is_tracked(obj))
    }
    fn get_rpy_memory_usage(&mut self, obj: GcRef) -> Option<usize> {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_rpy_memory_usage(obj))
    }
    fn get_rpy_type_index(&mut self, obj: GcRef) -> Option<usize> {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_rpy_type_index(obj))
    }
    fn get_rpy_roots(&mut self, visitor: &mut dyn FnMut(GcRef)) -> bool {
        gc_sync::gc_op(|gc| gc.get_rpy_roots(visitor))
    }
    fn get_rpy_referents(&mut self, obj: GcRef, visitor: &mut dyn FnMut(GcRef)) -> bool {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.get_rpy_referents(obj, visitor))
    }
    fn dump_rpy_heap(&mut self, fd: i32) -> Result<bool, i32> {
        gc_sync::gc_op(|gc| gc.dump_rpy_heap(fd))
    }
    fn get_typeids_text(&self) -> Option<Vec<u8>> {
        gc_sync::gc_query_reentrant(|gc| gc.get_typeids_text())
    }
    fn get_typeids_list(&self) -> Option<Vec<usize>> {
        gc_sync::gc_query_reentrant(|gc| gc.get_typeids_list())
    }
    fn add_memory_pressure(&mut self, size: isize, object: GcRef) {
        if object.is_null() {
            gc_sync::gc_op(|gc| gc.add_memory_pressure(size, object));
        } else {
            gc_sync::gc_op_with_root(object, |gc, object| gc.add_memory_pressure(size, object));
        }
    }
    fn total_memory_pressure(&mut self) -> isize {
        gc_sync::gc_op(|gc| gc.total_memory_pressure())
    }
    fn is_app_level_object(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op_with_root(obj, |gc, obj| gc.is_app_level_object(obj))
    }
    fn collect_oldgen_nonmoving(&mut self) {
        gc_sync::gc_op(|gc| gc.collect_oldgen_nonmoving())
    }
    fn enable(&mut self) {
        gc_sync::gc_op(|gc| gc.enable())
    }
    fn disable(&mut self) {
        gc_sync::gc_op(|gc| gc.disable())
    }
    fn isenabled(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.isenabled())
    }
    fn register_finalizer(&mut self, fq_index: usize, obj: GcRef, trigger: FinalizerTriggerFn) {
        gc_sync::gc_op(|gc| gc.register_finalizer(fq_index, obj, trigger))
    }
    fn finalizer_next_dead(&mut self, fq_index: usize) -> Option<GcRef> {
        gc_sync::gc_op(|gc| gc.finalizer_next_dead(fq_index))
    }
    fn id_or_identityhash(&mut self, obj_addr: usize) -> usize {
        gc_sync::gc_op(|gc| gc.id_or_identityhash(obj_addr))
    }
    fn get_write_barrier_descr(&self) -> Option<WriteBarrierDescr> {
        gc_sync::gc_query_reentrant(|gc| gc.get_write_barrier_descr())
    }
    unsafe fn add_root(&mut self, root: *mut GcRef) {
        unsafe { gc_sync::gc_op_add_root(root) }
    }
    fn remove_root(&mut self, root: *mut GcRef) {
        gc_sync::gc_op(|gc| gc.remove_root(root))
    }
    fn is_managed_heap_object(&self, addr: usize) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_managed_heap_object(addr))
    }
    fn is_nursery_object(&self, addr: usize) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_nursery_object(addr))
    }
    fn nursery_free(&self) -> *mut u8 {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_free())
    }
    fn nursery_free_addr(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_free_addr())
    }
    fn nursery_top(&self) -> *const u8 {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_top())
    }
    fn nursery_top_addr(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.nursery_top_addr())
    }
    fn max_nursery_object_size(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.max_nursery_object_size())
    }
    fn card_page_shift(&self) -> u32 {
        gc_sync::gc_query_reentrant(|gc| gc.card_page_shift())
    }
    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.jit_remember_young_pointer(obj))
    }
    fn can_optimize_cond_call(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.can_optimize_cond_call())
    }
    fn gc_step(&mut self) -> bool {
        gc_sync::gc_op(|gc| gc.gc_step())
    }
    fn jit_free(&mut self, code_ptr: usize, size: usize) {
        gc_sync::gc_op(|gc| gc.jit_free(code_ptr, size))
    }
    fn pin(&mut self, obj: GcRef) -> bool {
        gc_sync::gc_op(|gc| gc.pin(obj))
    }
    fn unpin(&mut self, obj: GcRef) {
        gc_sync::gc_op(|gc| gc.unpin(obj))
    }
    fn is_pinned(&self, obj: GcRef) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_pinned(obj))
    }
    fn register_type(&mut self, info: trace::TypeInfo) -> u32 {
        gc_sync::gc_op(|gc| gc.register_type(info))
    }
    fn type_count(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.type_count())
    }
    fn heap_byte_stats(&self) -> (usize, usize) {
        gc_sync::gc_query_reentrant(|gc| gc.heap_byte_stats())
    }
    fn gc_memory_stats(&self) -> GcMemoryStats {
        gc_sync::gc_query_reentrant(|gc| gc.gc_memory_stats())
    }
    fn major_threshold_reached(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.major_threshold_reached())
    }
    fn collection_counts(&self) -> (usize, usize) {
        gc_sync::gc_query_reentrant(|gc| gc.collection_counts())
    }
    fn type_alloc_is_plain(&self, type_id: u32) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.type_alloc_is_plain(type_id))
    }
    fn type_size(&self, type_id: u32) -> Option<usize> {
        gc_sync::gc_query_reentrant(|gc| gc.type_size(type_id))
    }
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        gc_sync::gc_query_reentrant(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
    }
    fn register_vtable_for_type(&mut self, vtable: usize, type_id: u32) {
        gc_sync::gc_op(|gc| gc.register_vtable_for_type(vtable, type_id))
    }
    fn freeze_types(&mut self) {
        gc_sync::gc_op(|gc| gc.freeze_types())
    }
    fn supports_guard_gc_type(&self) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.supports_guard_gc_type())
    }
    fn check_is_object(&self, gcref: GcRef) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.check_is_object(gcref))
    }
    fn is_tagged_immediate(&self, addr: usize) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.is_tagged_immediate(addr))
    }
    fn can_move(&self, gcref: GcRef) -> bool {
        gc_sync::gc_query_reentrant(|gc| gc.can_move(gcref))
    }
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        gc_sync::gc_query_reentrant(|gc| gc.get_translated_info_for_typeinfo())
    }
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        gc_sync::gc_query_reentrant(|gc| gc.get_translated_info_for_guard_is_object())
    }
    fn subclassrange_min_offset(&self) -> usize {
        gc_sync::gc_query_reentrant(|gc| gc.subclassrange_min_offset())
    }
    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        gc_sync::gc_query_reentrant(|gc| gc.subclass_range(classptr))
    }
    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        gc_sync::gc_query_reentrant(|gc| gc.typeid_subclass_range(typeid))
    }
    fn get_actual_typeid(&self, gcref: GcRef) -> Option<u32> {
        gc_sync::gc_query_reentrant(|gc| gc.get_actual_typeid(gcref))
    }
    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        gc_sync::gc_query_reentrant(|gc| gc.typeid_is_object(typeid))
    }
}

/// GC rewriter — transforms IR operations for GC integration.
///
/// Converts high-level NEW_*/SETFIELD_GC operations into:
/// - Inline nursery bump-pointer allocation (CALL_MALLOC_NURSERY)
/// - Write barrier conditional calls (COND_CALL_GC_WB)
///
/// Reference: rpython/jit/backend/llsupport/rewrite.py GcRewriterAssembler.
pub trait GcRewriter: Send {
    /// Rewrite a list of operations, inserting GC-aware code.
    fn rewrite_for_gc(&self, ops: &[Op]) -> Vec<Op>;
    /// Rewrite with access to the constant pool.
    /// Returns (rewritten ops, merged constants, gc_table gcrefs). Each
    /// `Const` box carries its own type via `Const::get_type`, so a separate
    /// type side-table is no longer threaded through the return. The third
    /// element is the per-loop reference-constant list collected by
    /// `remove_constptr` (rewrite.py:1033-1043 `gcrefs_output_list`); the
    /// backend builds a `GcTable` from it and bakes its base address into the
    /// `LoadFromGcTable` loads.
    ///
    /// The default impl forwards to `rewrite_for_gc` and preserves the
    /// caller's constants verbatim. `rewrite_for_gc` may leave `Const*`
    /// operands untouched, so returning an empty map would silently strand
    /// downstream readers that resolve `ConstInt.raw()` against this table.
    fn rewrite_for_gc_with_constants(
        &self,
        ops: &[Op],
        constants: &ConstMap<Const>,
    ) -> (Vec<Op>, ConstMap<Const>, Vec<GcRef>) {
        (self.rewrite_for_gc(ops), constants.clone(), Vec::new())
    }
}

/// Stack map — records which frame slots contain GC references at a safepoint.
///
/// At each guard (potential GC safepoint), the backend records a stack map
/// so the GC can find all live references in compiled code.
#[derive(Debug, Clone)]
pub struct GcMap {
    /// Bitmap: bit N is set if frame slot N contains a GC reference.
    pub ref_bitmap: Vec<u64>,
}

impl GcMap {
    pub fn new() -> Self {
        GcMap {
            ref_bitmap: Vec::new(),
        }
    }

    pub fn set_ref(&mut self, slot: usize) {
        let word = slot / 64;
        let bit = slot % 64;
        if word >= self.ref_bitmap.len() {
            self.ref_bitmap.resize(word + 1, 0);
        }
        self.ref_bitmap[word] |= 1u64 << bit;
    }

    pub fn is_ref(&self, slot: usize) -> bool {
        let word = slot / 64;
        let bit = slot % 64;
        if word >= self.ref_bitmap.len() {
            return false;
        }
        (self.ref_bitmap[word] >> bit) & 1 != 0
    }
}

impl Default for GcMap {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Process-global active GC allocator hooks
// ─────────────────────────────────────────────────────────────────────
//
// The metainterp / optimizer layer needs a backend-agnostic way to query
// the current CPU's GC type registry (llmodel.py:541-546
// `cpu.check_is_object(gcptr)`). In RPython the optimizer reaches it via
// `self.optimizer.cpu`, which holds a reference to the backend-provided
// CPU object. majit has no such field; instead the live backends register
// a callback here that the metainterp can invoke without taking a
// backend dependency.

/// Process-global callback that answers `cpu.check_is_object(gcptr)` for
/// the currently active backend. Set by the backend when it installs a
/// GC runtime; cleared when the runtime is
/// unregistered.
pub type CheckIsObjectFn = fn(GcRef) -> bool;

/// Process-global callback that answers the collector's tagged-immediate
/// test (gc/base.py:380-383 `is_valid_gc_object`) for the currently
/// active backend's GC: `config.taggedpointers && (addr & 1 == 1)`.
/// Lets a backend-agnostic caller decide that an odd-valued constant
/// address is an unboxed immediate rather than a heap object, without
/// reading an object header at offset 0.
pub type IsTaggedImmediateFn = fn(usize) -> bool;

/// Process-global callback that answers `gc_ll_descr.get_actual_typeid`
/// (gc.py:624-629) for the currently active backend. Returns the
/// `rffi.cast(HDRPTR, gcptr).tid` half-word for managed objects and
/// `vtable_to_type_id` for the foreign-object seam pyre uses. Paired
/// with `ACTIVE_CHECK_IS_OBJECT`; both are installed together so the
/// metainterp's guard interpretation stays consistent with the GC's
/// runtime layout assumptions.
pub type GetActualTypeidFn = fn(GcRef) -> Option<u32>;

/// Process-global callback that answers the codegen-time bounds lookup
/// `rffi.cast(rclass.CLASSTYPE, vtable_ptr).subclassrange_{min,max}`
/// from x86/assembler.py:1971-1974. Used by the executor's
/// `GuardSubclass` arm to evaluate bridges interpretively.
pub type SubclassRangeFn = fn(classptr: usize) -> Option<(i64, i64)>;

/// Process-global callback that answers the `value.typeptr
/// .subclassrange_min/max` lookup from llgraph/runner.py:1271-1281
/// directly by typeid. The backend installs this alongside
/// `subclass_range` so the executor can recover the object side of
/// `execute_guard_subclass` without going through a vtable pointer —
/// managed objects carry only a typeid in their GC header, and the
/// TYPE_INFO table already stores the preorder bounds in its paired
/// `CLASSTYPE` entry (gctypelayout.py:359-374).
pub type TypeidSubclassRangeFn = fn(typeid: u32) -> Option<(i64, i64)>;

/// Process-global callback that answers `rclass.OBJECT`-layout queries
/// by typeid — "does this typeid carry `T_IS_RPYTHON_INSTANCE` in its
/// TYPE_INFO entry" (gctypelayout.py:642). The executor's
/// `GuardIsObject` arm calls this after resolving the object's typeid
/// via the `get_actual_typeid` seam, avoiding a second indirection
/// through `check_is_object` (which would re-resolve the typeid).
pub type TypeidIsObjectFn = fn(typeid: u32) -> Option<bool>;
pub type ExtraRootWalkerFn = fn(&mut dyn FnMut(&mut GcRef));

/// Process-global callback that answers `rgc.can_move(gcref)`
/// (rpython/rlib/rgc.py:229) for the currently active backend's GC. The
/// const-baking site (`x86/regalloc.py:58-61 convert_to_imm`) consults
/// this before baking a `ConstPtr` immediate.
pub type CanMoveFn = fn(GcRef) -> bool;

global_hook!(static ACTIVE_CHECK_IS_OBJECT: CheckIsObjectFn);
global_hook!(static ACTIVE_IS_TAGGED_IMMEDIATE: IsTaggedImmediateFn);
global_hook!(static ACTIVE_GET_ACTUAL_TYPEID: GetActualTypeidFn);
global_hook!(static ACTIVE_SUBCLASS_RANGE: SubclassRangeFn);
global_hook!(static ACTIVE_TYPEID_SUBCLASS_RANGE: TypeidSubclassRangeFn);
global_hook!(static ACTIVE_TYPEID_IS_OBJECT: TypeidIsObjectFn);
global_hook!(static ACTIVE_CAN_MOVE: CanMoveFn);
static ACTIVE_SUPPORTS_GUARD_GC_TYPE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
global_hook!(static ACTIVE_EXTRA_ROOT_WALKER: ExtraRootWalkerFn);

/// Bundle of callbacks the metainterp / executor can reach through
/// process-global cells. Mirrors the fan-out of methods RPython's optimizer
/// and blackhole reach via `self.cpu` / `self.cpu.gc_ll_descr`; majit
/// installs them together so a backend swap is a single call.
#[derive(Clone, Copy, Default)]
pub struct ActiveGcGuardHooks {
    pub check_is_object: Option<CheckIsObjectFn>,
    pub is_tagged_immediate: Option<IsTaggedImmediateFn>,
    pub get_actual_typeid: Option<GetActualTypeidFn>,
    pub subclass_range: Option<SubclassRangeFn>,
    pub typeid_subclass_range: Option<TypeidSubclassRangeFn>,
    pub typeid_is_object: Option<TypeidIsObjectFn>,
    pub can_move: Option<CanMoveFn>,
    pub supports_guard_gc_type: bool,
}

/// Install the active backend's GC-guard callbacks.
/// Called by backends when they enter a JIT region. Pass a default
/// `ActiveGcGuardHooks` with every field set to `None` / `false` to
/// clear. Mirrors how RPython's `cpu` field lets the optimizer and
/// executor reach `cpu.check_is_object`, `gc_ll_descr
/// .get_actual_typeid`, and the codegen-time bounds lookup; majit
/// bundles them here so a backend install is a single call.
pub fn set_active_gc_guard_hooks(hooks: ActiveGcGuardHooks) {
    ACTIVE_CHECK_IS_OBJECT.set(hooks.check_is_object);
    ACTIVE_IS_TAGGED_IMMEDIATE.set(hooks.is_tagged_immediate);
    ACTIVE_GET_ACTUAL_TYPEID.set(hooks.get_actual_typeid);
    ACTIVE_SUBCLASS_RANGE.set(hooks.subclass_range);
    ACTIVE_TYPEID_SUBCLASS_RANGE.set(hooks.typeid_subclass_range);
    ACTIVE_TYPEID_IS_OBJECT.set(hooks.typeid_is_object);
    ACTIVE_CAN_MOVE.set(hooks.can_move);
    ACTIVE_SUPPORTS_GUARD_GC_TYPE.store(
        hooks.supports_guard_gc_type,
        std::sync::atomic::Ordering::Release,
    );
}

/// Snapshot of the current guard-hook cells (test-only helper).
#[allow(dead_code)]
fn current_gc_guard_hooks() -> ActiveGcGuardHooks {
    ActiveGcGuardHooks {
        check_is_object: ACTIVE_CHECK_IS_OBJECT.get(),
        is_tagged_immediate: ACTIVE_IS_TAGGED_IMMEDIATE.get(),
        get_actual_typeid: ACTIVE_GET_ACTUAL_TYPEID.get(),
        subclass_range: ACTIVE_SUBCLASS_RANGE.get(),
        typeid_subclass_range: ACTIVE_TYPEID_SUBCLASS_RANGE.get(),
        typeid_is_object: ACTIVE_TYPEID_IS_OBJECT.get(),
        can_move: ACTIVE_CAN_MOVE.get(),
        supports_guard_gc_type: supports_guard_gc_type(),
    }
}

static GUARD_HOOKS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Test-only: install `hooks` while holding a process-global lock, restoring
/// the previous guard-hook values when the returned guard drops. Serializes
/// hook-overriding tests so the now-global cells do not race. `pub` because
/// the sole caller lives in the `majit-metainterp` crate's tests.
pub struct GuardHooksTestGuard {
    prev: ActiveGcGuardHooks,
    _lock: std::sync::MutexGuard<'static, ()>,
}
impl Drop for GuardHooksTestGuard {
    fn drop(&mut self) {
        set_active_gc_guard_hooks(self.prev);
    }
}
pub fn override_gc_guard_hooks_for_test(hooks: ActiveGcGuardHooks) -> GuardHooksTestGuard {
    let lock = GUARD_HOOKS_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let prev = current_gc_guard_hooks();
    set_active_gc_guard_hooks(hooks);
    GuardHooksTestGuard { prev, _lock: lock }
}

/// Install a process-global callback that exposes non-shadow-stack roots
/// owned by the embedding runtime.
pub fn set_active_extra_root_walker(walker: Option<ExtraRootWalkerFn>) {
    ACTIVE_EXTRA_ROOT_WALKER.set(walker);
}

/// Walk the active runtime's extra GC roots.
pub fn walk_active_extra_roots(visitor: &mut dyn FnMut(&mut GcRef)) {
    if let Some(f) = ACTIVE_EXTRA_ROOT_WALKER.get() {
        f(visitor);
    }
}

/// llmodel.py:541-546 `cpu.check_is_object(gcptr)` shim. Returns whether
/// `gcref` is a `T_IS_RPYTHON_INSTANCE` (has `typeptr` at offset 0). When
/// no backend has installed a callback, returns `false`.
pub fn check_is_object(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    match ACTIVE_CHECK_IS_OBJECT.get() {
        Some(f) => f(gcref),
        None => false,
    }
}

/// gc/base.py:380-383 `is_valid_gc_object` tagged-immediate test shim.
/// Delegates to the active backend's installed callback, which reads its
/// GC's `config.taggedpointers`. Returns `false` for null and when no
/// backend is installed — same absent-backend semantics as
/// `check_is_object`, so flag-off / no-GC paths are unaffected.
pub fn is_tagged_immediate(addr: usize) -> bool {
    match ACTIVE_IS_TAGGED_IMMEDIATE.get() {
        Some(f) => f(addr),
        None => false,
    }
}

/// Whether the active backend's GC has `config.taggedpointers` enabled
/// (translationoption.py:185). The installed `is_tagged_immediate` callback
/// answers `config.taggedpointers && (addr & 1 == 1)`, so probing it with an
/// odd sentinel address isolates the config flag without a live pointer.
/// Returns `false` when no backend is installed — same absent-backend
/// semantics as [`is_tagged_immediate`], so flag-off paths are unaffected.
pub fn taggedpointers_enabled() -> bool {
    is_tagged_immediate(1)
}

/// gc.py:624-629 `gc_ll_descr.get_actual_typeid(gcptr)` shim.
/// Delegates to the active backend's installed callback; returns
/// `None` when no backend is installed, which mirrors
/// `llgraph/runner.py:1263-1269` skip semantics (the interpretive
/// guard treats an unresolved object as passing).
pub fn get_actual_typeid(gcref: GcRef) -> Option<u32> {
    if gcref.is_null() {
        return None;
    }
    match ACTIVE_GET_ACTUAL_TYPEID.get() {
        Some(f) => f(gcref),
        None => None,
    }
}

/// `rgc.can_move(gcref)` shim (rpython/rlib/rgc.py:229). Delegates to the
/// active backend's installed callback. Returns `false` for null pointers
/// and when no backend is installed — i.e. a non-moving / absent GC, where
/// every object address is stable (rgc.py:231 "with non-moving GCs, it is
/// always False").
pub fn can_move(gcref: GcRef) -> bool {
    if gcref.is_null() {
        return false;
    }
    match ACTIVE_CAN_MOVE.get() {
        Some(f) => f(gcref),
        None => false,
    }
}

/// x86/assembler.py:1971-1974 codegen-time bounds lookup shim used by
/// the interpretive `GuardSubclass`. Returns
/// `(subclassrange_min, subclassrange_max)` for the class whose vtable
/// pointer is given, or `None` when no backend is installed.
pub fn subclass_range(classptr: usize) -> Option<(i64, i64)> {
    ACTIVE_SUBCLASS_RANGE.get().and_then(|f| f(classptr))
}

/// Companion to `subclass_range` keyed by typeid instead of classptr.
/// Resolves `value.typeptr.subclassrange_min/max` from
/// llgraph/runner.py:1271-1281 when the executor only has a typeid in
/// hand (e.g. after calling `get_actual_typeid` on an object whose
/// classptr is known only to the GC). Returns `None` when no backend
/// is installed.
pub fn typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    ACTIVE_TYPEID_SUBCLASS_RANGE.get().and_then(|f| f(typeid))
}

/// Companion to `check_is_object` keyed by typeid. Called by the
/// executor's `GuardIsObject` arm after resolving the object to a
/// typeid via `get_actual_typeid`. Returns `None` when no backend is
/// installed.
pub fn typeid_is_object(typeid: u32) -> Option<bool> {
    ACTIVE_TYPEID_IS_OBJECT.get().and_then(|f| f(typeid))
}

/// llmodel.py:63 `supports_guard_gc_type` shim. Mirrors the active
/// backend's capability flag. `false` when no backend has been installed.
pub fn supports_guard_gc_type() -> bool {
    ACTIVE_SUPPORTS_GUARD_GC_TYPE.load(std::sync::atomic::Ordering::Acquire)
}

/// Set once any thread has installed a per-thread GC box on any backend. That
/// path is test-only: production installs the backend standalone, leaving the
/// process-global [`gc_sync`] singleton as the sole allocator on every thread.
///
/// Set-only, never cleared: a backend can drop one thread's box while another
/// thread still owns one, and clearing the flag would route past that live box.
///
/// RPython has no per-thread allocator — the GC transformer weaves every
/// `malloc` against the single translation-time `gcdata`
/// (`gctransform/framework.py`) — so the box is pyre-side test scaffolding, and
/// the queries below let the allocation path it does not serve skip it.
///
/// Only compiled when a box can exist at all, i.e. under `gc_box`; see
/// [`gc_box_installed`] for what the other build spells instead.
#[cfg(any(test, feature = "gc_box"))]
static GC_BOX_INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Record that a backend has installed a per-thread GC box. Called from each
/// backend's box-installing entry point.
#[cfg(any(test, feature = "gc_box"))]
pub fn note_gc_box_installed() {
    GC_BOX_INSTALLED.store(true, std::sync::atomic::Ordering::Release);
}

/// A build without `gc_box` answers every [`gc_box_installed`] query with a
/// constant `false`, so a box installed here would be invisible to all of them
/// and its owner would silently allocate out of the singleton instead. Refuse
/// loudly rather than let that read as a pass.
#[cfg(not(any(test, feature = "gc_box")))]
pub fn note_gc_box_installed() {
    panic!(
        "a per-thread GC box was installed in a build without the `gc_box` feature; \
         production installs the backend standalone (`install_gc_standalone`), and a \
         test that needs a box must depend on `majit-gc` with `features = [\"gc_box\"]`"
    );
}

/// Whether any thread may own a per-thread GC box. See [`GC_BOX_INSTALLED`].
#[cfg(any(test, feature = "gc_box"))]
#[inline]
pub fn gc_box_installed() -> bool {
    GC_BOX_INSTALLED.load(std::sync::atomic::Ordering::Acquire)
}

/// No backend can hold a box in this build, so every `gc_box_installed() && …`
/// probe folds away and the allocation, write-barrier and query trampolines
/// call `gc_sync` directly — which is the whole shape RPython has.
#[cfg(not(any(test, feature = "gc_box")))]
#[inline(always)]
pub fn gc_box_installed() -> bool {
    false
}

// ── Host-side nursery allocation hook ───────────────────────────────
//
// Separate from `ActiveGcGuardHooks` because allocation is not a
// guard-time concern. The backend installs one function pointer here
// so host-side allocators (pyre-object's `w_int_new`, `w_float_new`,
// …) can route through the real GC without taking a backend-specific
// dependency. Mirrors how RPython host code reaches `gc.malloc(TYPE)`
// through the global GC instance.
//
// When no thread owns a per-thread GC box, a backend's hook does nothing but
// forward to `gc_sync` — the same process-global singleton this crate owns. The
// `standalone_*` functions below are that forwarding, spelled once here and
// called from both sides: the backend trampolines fall through to them, and the
// entry points take them directly instead of going out through the hook. Same
// work, one fewer indirect call and stack frame per allocation. The hook is
// still consulted for its presence, so "no backend installed" keeps returning
// null exactly as before.

/// The nursery allocation a backend's `AllocNurseryTypedFn` performs once no
/// per-thread box owns the request.
///
/// Non-collecting on purpose: the caller holds a raw pointer that is not a
/// registered GC root, so a collection here would move the fresh object out
/// from under it. Falling back to old-gen on a full nursery keeps the result
/// stable across any minor collection before the caller stores it into a
/// tracked slot.
#[inline]
pub fn standalone_alloc_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    gc_sync::gc_op(|g| g.try_alloc_nursery_no_collect_typed(type_id, payload_size))
}

/// [`standalone_alloc_nursery_typed`] for a type that carries neither a
/// finalizer nor the weakref flag: `malloc_fast` (`framework.py:361-382`).
///
/// # Safety
/// `type_id` must name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn standalone_alloc_fast_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    gc_sync::gc_op(|g| unsafe { g.try_alloc_fast_nursery_no_collect_typed(type_id, payload_size) })
}

/// The rooted collecting nursery allocation a backend's
/// `AllocNurseryCollectingTypedRootedFn` performs once no per-thread box owns
/// the request.
///
/// # Safety
/// Same contract as [`alloc_nursery_collecting_typed_rooted`]: `root` and
/// `needs_write_barrier` must stay valid for the call.
#[inline]
pub unsafe fn standalone_alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    gc_sync::gc_op(|g| unsafe {
        g.alloc_nursery_collecting_typed_rooted(type_id, payload_size, root, needs_write_barrier)
    })
}

/// [`standalone_alloc_nursery_collecting_typed_rooted`] for a type that carries
/// neither a finalizer nor the weakref flag: `malloc_fast`
/// (`framework.py:361-382`).
///
/// # Safety
/// Same contract as [`standalone_alloc_nursery_collecting_typed_rooted`], plus
/// `type_id` must name a type with no destructor and no weakref flag.
#[inline]
pub unsafe fn standalone_alloc_fast_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    gc_sync::gc_op(|g| unsafe {
        g.alloc_fast_nursery_collecting_typed_rooted(
            type_id,
            payload_size,
            root,
            needs_write_barrier,
        )
    })
}

/// Process-global callback that performs a nursery allocation for the
/// currently active backend. The callback returns `GcRef(0)` (i.e.
/// null) on allocation failure so callers can fall back to a
/// non-GC allocator.
pub type AllocNurseryTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_NURSERY_TYPED: AllocNurseryTypedFn);

/// Install the active backend's nursery allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_nursery_typed(hook: Option<AllocNurseryTypedFn>) {
    ACTIVE_ALLOC_NURSERY_TYPED.set(hook);
}

/// Whether a backend owns the heap.
///
/// The allocation entry points below answer both "no backend" and "the GC
/// refused" with `GcRef(0)`, and the two are not interchangeable. Nothing owns
/// the heap during a bare unit test or the pre-`init_gc_subsystem` bootstrap,
/// so a caller's own `malloc(flavor='raw')` path *is* the whole heap there and
/// taking it keeps the object graph consistent. Once a backend is installed
/// that same fallback would leave a raw pointer in a field the collector traces
/// as a GC reference, and the caller must fail instead.
///
/// `set_gc_allocator` publishes every allocation hook in one block
/// (`majit-backend-dynasm/src/runner.rs:310-323` and its cranelift/wasm twins),
/// so this one cell answers for all of them.
///
/// Upstream has neither state: the GC is a prebuilt constant
/// (`rpython/memory/gctransform/framework.py:254`) and a nursery that cannot
/// satisfy a request reaches `collect_and_reserve`
/// (`rpython/memory/gc/incminimark.py:981-985`), which raises MemoryError.
pub fn gc_allocator_installed() -> bool {
    ACTIVE_ALLOC_NURSERY_TYPED.get().is_some()
}

/// What an allocation answered, with the two non-pointer states kept apart.
///
/// `malloc_fixedsize` (incminimark.py:640-693) has neither state. The GC is a
/// prebuilt constant (framework.py:254), so a route always exists, and a
/// nursery that cannot satisfy the request reaches `collect_and_reserve`
/// (incminimark.py:981-985), which raises MemoryError rather than handing a
/// null back. Both states are this port's own, and they are not
/// interchangeable:
///
/// * [`NoRoute`](Self::NoRoute) — nothing owns the heap yet: a bare unit
///   test, the pre-`init_gc_subsystem` bootstrap, or a build with no backend.
///   The caller's `malloc_raw` path *is* the whole heap there, so taking it
///   keeps the object graph consistent.
/// * [`Failed`](Self::Failed) — a GC owns the heap and could not satisfy the
///   request. `malloc_raw` is the wrong answer now: the object would hold
///   managed references the collector never traces or forwards, and its
///   missing header lets a type-id witness misread the words before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GcAllocOutcome {
    Allocated(*mut u8),
    Failed,
    NoRoute,
}

impl GcAllocOutcome {
    /// Classify a hook result: `None` is [`NoRoute`](Self::NoRoute),
    /// `Some(null)` is [`Failed`](Self::Failed).
    #[inline]
    pub fn from_hook(result: Option<*mut u8>) -> Self {
        match result {
            Some(raw) if !raw.is_null() => Self::Allocated(raw),
            Some(_) => Self::Failed,
            None => Self::NoRoute,
        }
    }

    /// Classify what an allocation entry point in this module returned. Those
    /// answer both states with `GcRef(0)`, so the null is read against
    /// [`gc_allocator_installed`] to tell them apart.
    #[inline]
    pub fn classify(raw: GcRef) -> Self {
        if raw.0 != 0 {
            Self::Allocated(raw.0 as *mut u8)
        } else if gc_allocator_installed() {
            Self::Failed
        } else {
            Self::NoRoute
        }
    }

    /// The allocated pointer, or `None` for [`NoRoute`](Self::NoRoute) so the
    /// caller takes its own non-GC path. A [`Failed`](Self::Failed) does not
    /// return: see [`gc_alloc_failed`].
    #[inline]
    pub fn allocated_or_abort(self, payload_size: usize) -> Option<*mut u8> {
        match self {
            Self::Allocated(raw) => Some(raw),
            Self::Failed => gc_alloc_failed(payload_size),
            Self::NoRoute => None,
        }
    }
}

/// A GC that owns the heap could not satisfy an allocation.
///
/// `collect_and_reserve` (incminimark.py:981-985) raises MemoryError at this
/// point, so no caller of `malloc_fixedsize` observes a null. The callers here
/// return a bare pointer and run under JIT frames that cannot unwind, so the
/// failure aborts instead — the answer `alloc_typed_items_block_nursery`
/// already gives for a digit array whose allocation fails.
#[cold]
#[inline(never)]
pub fn gc_alloc_failed(payload_size: usize) -> ! {
    let layout = std::alloc::Layout::from_size_align(payload_size, std::mem::align_of::<usize>())
        .unwrap_or_else(|_| std::alloc::Layout::new::<usize>());
    std::alloc::handle_alloc_error(layout)
}

/// Allocate through the active backend's GC. Returns `GcRef(0)` when
/// no backend has installed a hook (callers treat this as a
/// null pointer and fall back to their non-GC path).
pub fn alloc_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED.get() {
        Some(_) if !gc_box_installed() => standalone_alloc_nursery_typed(type_id, payload_size),
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// [`alloc_nursery_typed`] for a type that carries neither a finalizer nor the
/// weakref flag: `malloc_fast` (`framework.py:361-382`), the copy
/// `gct_fv_gc_malloc` (`framework.py:820-838`) selects for exactly that case.
///
/// Only the direct path folds the two registrations out. A per-thread GC box is
/// a backend-test configuration, so its `AllocNurseryTypedFn` keeps running the
/// general body — always a correct implementation of the fast one, just with
/// the two constant-false tests still present.
///
/// # Safety
/// `type_id` must name a type with no destructor and no weakref flag.
pub unsafe fn alloc_fast_nursery_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED.get() {
        Some(_) if !gc_box_installed() => unsafe {
            standalone_alloc_fast_nursery_typed(type_id, payload_size)
        },
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// Process-global callback for a headerless, non-collecting nursery
/// allocation. Returns `GcRef(0)` when no backend has installed a hook, so
/// the caller can keep its own non-GC path.
pub type AllocNurseryHeaderlessNoCollectFn = fn(size: usize) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_HEADERLESS_NO_COLLECT: AllocNurseryHeaderlessNoCollectFn
);

/// Install the active backend's headerless no-collect allocator callback.
/// Pass `None` to clear.
pub fn set_active_alloc_nursery_headerless_no_collect(
    hook: Option<AllocNurseryHeaderlessNoCollectFn>,
) {
    ACTIVE_ALLOC_NURSERY_HEADERLESS_NO_COLLECT.set(hook);
}

/// Allocate a headerless object through the active backend's GC without
/// letting it collect. Returns `GcRef(0)` when no backend installed a hook.
///
/// The metainterp's jitcode tracer reaches its `NEW` allocation through here:
/// an interpreter that declares `headerless_structs` owns those objects in its
/// own collected pool, so a plain host-heap block there would be invisible to
/// its collector for as long as the object stays reachable.
pub fn alloc_nursery_headerless_no_collect(size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_HEADERLESS_NO_COLLECT.get() {
        Some(f) => f(size),
        None => GcRef(0),
    }
}

/// Placement-reporting companion of [`AllocNurseryTypedFn`] for fresh-object
/// initialization. The allocation remains no-collect; the out-parameter is
/// `false` for a nursery result and `true` for an old-gen spill.
pub type AllocNurseryTypedWithPlacementFn =
    unsafe fn(type_id: u32, payload_size: usize, needs_write_barrier: *mut bool) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT:
        AllocNurseryTypedWithPlacementFn
);

pub fn set_active_alloc_nursery_typed_with_placement(
    hook: Option<AllocNurseryTypedWithPlacementFn>,
) {
    ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT.set(hook);
}

/// Allocate through the active backend's no-collect nursery allocator while
/// reporting the result placement.
///
/// # Safety
/// `needs_write_barrier` must remain a valid mutable `bool` slot until this
/// call returns.
pub unsafe fn alloc_nursery_typed_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT.get() {
        Some(f) => unsafe { f(type_id, payload_size, needs_write_barrier) },
        None => {
            // No backend installed placement reporting, so keep the
            // conservative creation barrier the sibling fallbacks report.
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// [`alloc_nursery_typed_with_placement`] for a type that carries neither a
/// finalizer nor the weakref flag: `malloc_fast` (`framework.py:361-382`).
///
/// Like [`alloc_fast_nursery_typed`], only the direct path folds the two
/// registrations out; a per-thread box keeps the general body.
///
/// # Safety
/// Same contract as [`alloc_nursery_typed_with_placement`], plus `type_id` must
/// name a type with no destructor and no weakref flag.
pub unsafe fn alloc_fast_nursery_typed_with_placement(
    type_id: u32,
    payload_size: usize,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_TYPED_WITH_PLACEMENT.get() {
        Some(_) if !gc_box_installed() => gc_sync::gc_op(|g| unsafe {
            g.try_alloc_fast_nursery_no_collect_typed_with_placement(
                type_id,
                payload_size,
                needs_write_barrier,
            )
        }),
        Some(f) => unsafe { f(type_id, payload_size, needs_write_barrier) },
        None => {
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// Process-global callback that performs a stable-address old-gen
/// allocation for the currently active backend. Used by host-side
/// allocators whose callers hold the returned pointer on the Rust
/// stack without registering it as a GC root. MiniMark's
/// old-gen is mark-sweep (non-moving), so a subsequent minor
/// collection cannot invalidate the pointer. The callback returns
/// `GcRef(0)` on allocation failure.
pub type AllocOldgenTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_OLDGEN_TYPED: AllocOldgenTypedFn);

/// Install the active backend's old-gen allocator callback. Pass
/// `None` to clear.
pub fn set_active_alloc_oldgen_typed(hook: Option<AllocOldgenTypedFn>) {
    ACTIVE_ALLOC_OLDGEN_TYPED.set(hook);
}

/// Allocate a stable-address (old-gen) object through the active
/// backend's GC. Returns `GcRef(0)` when no backend has installed a hook.
pub fn alloc_oldgen_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_OLDGEN_TYPED.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// Process-global callback for a *collecting* nursery allocation — unlike
/// [`alloc_nursery_typed`] (which the backends install as the no-collect
/// variant), this one runs a minor collection when the nursery is full instead
/// of spilling to old-gen. Only safe for callers that hold no unrooted GC
/// pointer across the allocation AND run at a JIT safepoint whose gcmap roots
/// the live set (e.g. the elidable bigint payload helpers, invoked from a
/// gcmap-carrying residual CallR). Returns `GcRef(0)` when no backend installed.
pub type AllocNurseryCollectingTypedFn = fn(type_id: u32, payload_size: usize) -> GcRef;

global_hook!(static ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED: AllocNurseryCollectingTypedFn);

/// Install the active backend's collecting-nursery allocator callback. Pass
/// `None` to clear. Backends that do not install one leave callers to fall back
/// to the no-collect path.
pub fn set_active_alloc_nursery_collecting_typed(hook: Option<AllocNurseryCollectingTypedFn>) {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.set(hook);
}

/// Allocate through the active backend's collecting nursery allocator. Returns
/// `GcRef(0)` when no backend (or no collecting hook) is installed (callers
/// treat this as null and fall back to the no-collect path).
pub fn alloc_nursery_collecting_typed(type_id: u32, payload_size: usize) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED.get() {
        Some(f) => f(type_id, payload_size),
        None => GcRef(0),
    }
}

/// Process-global rooted companion of
/// [`AllocNurseryCollectingTypedFn`]. The caller supplies the one GC slot that
/// is live only on the native Rust stack while the allocation may collect.
pub type AllocNurseryCollectingTypedRootedFn = unsafe fn(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef;

global_hook!(
    static ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED:
        AllocNurseryCollectingTypedRootedFn
);

pub fn set_active_alloc_nursery_collecting_typed_rooted(
    hook: Option<AllocNurseryCollectingTypedRootedFn>,
) {
    ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED.set(hook);
}

/// Allocate through the active rooted collecting allocator.
///
/// # Safety
/// `root` must remain a valid mutable `GcRef` slot until this call returns.
/// `needs_write_barrier` must remain a valid mutable `bool` slot.
pub unsafe fn alloc_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED.get() {
        Some(_) if !gc_box_installed() => unsafe {
            standalone_alloc_nursery_collecting_typed_rooted(
                type_id,
                payload_size,
                root,
                needs_write_barrier,
            )
        },
        Some(f) => unsafe { f(type_id, payload_size, root, needs_write_barrier) },
        None => {
            // No backend installed placement reporting, so keep the
            // conservative creation barrier the sibling fallbacks report.
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// [`alloc_nursery_collecting_typed_rooted`] for a type that carries neither a
/// finalizer nor the weakref flag: `malloc_fast` (`framework.py:361-382`).
///
/// Like [`alloc_fast_nursery_typed`], only the direct path folds the two
/// registrations out; a per-thread box keeps the general body.
///
/// # Safety
/// Same contract as [`alloc_nursery_collecting_typed_rooted`], plus `type_id`
/// must name a type with no destructor and no weakref flag.
pub unsafe fn alloc_fast_nursery_collecting_typed_rooted(
    type_id: u32,
    payload_size: usize,
    root: *mut GcRef,
    needs_write_barrier: *mut bool,
) -> GcRef {
    match ACTIVE_ALLOC_NURSERY_COLLECTING_TYPED_ROOTED.get() {
        Some(_) if !gc_box_installed() => unsafe {
            standalone_alloc_fast_nursery_collecting_typed_rooted(
                type_id,
                payload_size,
                root,
                needs_write_barrier,
            )
        },
        Some(f) => unsafe { f(type_id, payload_size, root, needs_write_barrier) },
        None => {
            unsafe { *needs_write_barrier = true };
            GcRef(0)
        }
    }
}

/// Process-global callback that runs a full mark-sweep collection cycle
/// on the active backend's GC (`GcAllocator::collect_full`). Used by
/// `pypy/module/gc/interp_gc.py:7-26 collect` ports — i.e. user-level
/// `gc.collect()` reaches the live GC through this trampoline. Returns
/// silently when no backend has installed a hook (callers treat
/// it as a no-op).
pub type CollectFullFn = fn();

global_hook!(static ACTIVE_COLLECT_FULL: CollectFullFn);

/// Install the active backend's full-collection trampoline. Pass
/// `None` to clear.
pub fn set_active_collect_full(hook: Option<CollectFullFn>) {
    ACTIVE_COLLECT_FULL.set(hook);
}

/// Trigger a full mark-sweep collection on the active backend's GC.
/// No-op when no backend has installed a hook.
pub fn collect_full() {
    if let Some(f) = ACTIVE_COLLECT_FULL.get() {
        f();
    }
}

/// Active-backend trampoline for `incminimark.py:810-822 collect_step`.
pub type CollectStepFn = fn() -> GcStepTransition;

global_hook!(static ACTIVE_COLLECT_STEP: CollectStepFn);

pub fn set_active_collect_step(hook: Option<CollectStepFn>) {
    ACTIVE_COLLECT_STEP.set(hook);
}

pub fn collect_step() -> GcStepTransition {
    ACTIVE_COLLECT_STEP.get().map_or(
        GcStepTransition {
            old_state: GcStepTransition::SCANNING,
            new_state: GcStepTransition::SCANNING,
        },
        |step| step(),
    )
}

/// Active-backend trampoline for `rpython/rlib/rgc.py:1224
/// do_get_objects`. The generation values are CPython 3.14's public
/// `gc.get_objects` convention. The backend must invoke the visitor while its
/// inspection pause is still held.
pub type GetObjectsVisitorFn = fn(GcRef);
pub type GetObjectsFn = fn(i8, GetObjectsVisitorFn);

global_hook!(static ACTIVE_GET_OBJECTS: GetObjectsFn);

pub fn set_active_get_objects(hook: Option<GetObjectsFn>) {
    ACTIVE_GET_OBJECTS.set(hook);
}

pub fn get_objects(generation: i8, visitor: GetObjectsVisitorFn) {
    if let Some(f) = ACTIVE_GET_OBJECTS.get() {
        f(generation, visitor);
    }
}

/// Active-backend trampoline for `pypy/module/gc/referents.py:53-78
/// _list_w_obj_referents`. Same rooting contract as [`get_objects`]: the
/// backend invokes the visitor while its inspection pause is still held.
pub type GetReferentsFn = fn(GcRef, GetObjectsVisitorFn);

global_hook!(static ACTIVE_GET_REFERENTS: GetReferentsFn);

pub fn set_active_get_referents(hook: Option<GetReferentsFn>) {
    ACTIVE_GET_REFERENTS.set(hook);
}

pub fn get_referents(obj: GcRef, visitor: GetObjectsVisitorFn) {
    if let Some(f) = ACTIVE_GET_REFERENTS.get() {
        f(obj, visitor);
    }
}

/// Active-backend trampoline for `gc.is_tracked`: whether the collector
/// traverses references out of the object.
pub type IsTrackedFn = fn(GcRef) -> bool;

global_hook!(static ACTIVE_IS_TRACKED: IsTrackedFn);

pub fn set_active_is_tracked(hook: Option<IsTrackedFn>) {
    ACTIVE_IS_TRACKED.set(hook);
}

pub fn is_tracked(obj: GcRef) -> bool {
    match ACTIVE_IS_TRACKED.get() {
        Some(f) => f(obj),
        None => false,
    }
}

/// Active-backend trampolines for `inspector.py:get_rpy_memory_usage` and
/// `get_rpy_type_index`.  The `Option` is the Rust spelling of upstream's
/// negative result for a missing collector operation.
pub type GetRpyIntrospectionFn = fn(GcRef) -> Option<usize>;

global_hook!(static ACTIVE_GET_RPY_MEMORY_USAGE: GetRpyIntrospectionFn);
global_hook!(static ACTIVE_GET_RPY_TYPE_INDEX: GetRpyIntrospectionFn);

pub fn set_active_get_rpy_memory_usage(hook: Option<GetRpyIntrospectionFn>) {
    ACTIVE_GET_RPY_MEMORY_USAGE.set(hook);
}

pub fn set_active_get_rpy_type_index(hook: Option<GetRpyIntrospectionFn>) {
    ACTIVE_GET_RPY_TYPE_INDEX.set(hook);
}

pub fn get_rpy_memory_usage(obj: GcRef) -> Option<usize> {
    ACTIVE_GET_RPY_MEMORY_USAGE.get().and_then(|f| f(obj))
}

pub fn get_rpy_type_index(obj: GcRef) -> Option<usize> {
    ACTIVE_GET_RPY_TYPE_INDEX.get().and_then(|f| f(obj))
}

pub type GetRpyRootsFn = fn(GetObjectsVisitorFn) -> bool;
pub type GetRpyReferentsFn = fn(GcRef, GetObjectsVisitorFn) -> bool;
pub type IsAppLevelObjectFn = fn(GcRef) -> bool;

global_hook!(static ACTIVE_GET_RPY_ROOTS: GetRpyRootsFn);
global_hook!(static ACTIVE_GET_RPY_REFERENTS: GetRpyReferentsFn);
global_hook!(static ACTIVE_IS_APP_LEVEL_OBJECT: IsAppLevelObjectFn);

pub fn set_active_get_rpy_roots(hook: Option<GetRpyRootsFn>) {
    ACTIVE_GET_RPY_ROOTS.set(hook);
}

pub fn set_active_get_rpy_referents(hook: Option<GetRpyReferentsFn>) {
    ACTIVE_GET_RPY_REFERENTS.set(hook);
}

pub fn set_active_is_app_level_object(hook: Option<IsAppLevelObjectFn>) {
    ACTIVE_IS_APP_LEVEL_OBJECT.set(hook);
}

pub fn get_rpy_roots(visitor: GetObjectsVisitorFn) -> bool {
    ACTIVE_GET_RPY_ROOTS.get().is_some_and(|f| f(visitor))
}

pub fn get_rpy_referents(obj: GcRef, visitor: GetObjectsVisitorFn) -> bool {
    ACTIVE_GET_RPY_REFERENTS
        .get()
        .is_some_and(|f| f(obj, visitor))
}

pub fn is_app_level_object(obj: GcRef) -> bool {
    ACTIVE_IS_APP_LEVEL_OBJECT.get().is_some_and(|f| f(obj))
}

pub type DumpRpyHeapFn = fn(i32) -> Result<bool, i32>;
pub type GetTypeidsTextFn = fn() -> Option<Vec<u8>>;
pub type GetTypeidsListFn = fn() -> Option<Vec<usize>>;

global_hook!(static ACTIVE_DUMP_RPY_HEAP: DumpRpyHeapFn);
global_hook!(static ACTIVE_GET_TYPEIDS_TEXT: GetTypeidsTextFn);
global_hook!(static ACTIVE_GET_TYPEIDS_LIST: GetTypeidsListFn);

pub fn set_active_dump_rpy_heap(hook: Option<DumpRpyHeapFn>) {
    ACTIVE_DUMP_RPY_HEAP.set(hook);
}

pub fn set_active_get_typeids_text(hook: Option<GetTypeidsTextFn>) {
    ACTIVE_GET_TYPEIDS_TEXT.set(hook);
}

pub fn set_active_get_typeids_list(hook: Option<GetTypeidsListFn>) {
    ACTIVE_GET_TYPEIDS_LIST.set(hook);
}

pub fn dump_rpy_heap(fd: i32) -> Result<bool, i32> {
    ACTIVE_DUMP_RPY_HEAP
        .get()
        .map_or(Ok(false), |hook| hook(fd))
}

pub fn get_typeids_text() -> Option<Vec<u8>> {
    ACTIVE_GET_TYPEIDS_TEXT.get().and_then(|hook| hook())
}

pub fn get_typeids_list() -> Option<Vec<usize>> {
    ACTIVE_GET_TYPEIDS_LIST.get().and_then(|hook| hook())
}

/// Active-backend trampolines for `rgc.add_memory_pressure` and
/// `inspector.count_memory_pressure`. The object-bearing form is retained for
/// translated callers even though `__pypy__.add_memory_pressure` passes NULL.
pub type AddMemoryPressureFn = fn(isize, GcRef);
pub type TotalMemoryPressureFn = fn() -> isize;

global_hook!(static ACTIVE_ADD_MEMORY_PRESSURE: AddMemoryPressureFn);
global_hook!(static ACTIVE_TOTAL_MEMORY_PRESSURE: TotalMemoryPressureFn);

pub fn set_active_add_memory_pressure(hook: Option<AddMemoryPressureFn>) {
    ACTIVE_ADD_MEMORY_PRESSURE.set(hook);
}

pub fn set_active_total_memory_pressure(hook: Option<TotalMemoryPressureFn>) {
    ACTIVE_TOTAL_MEMORY_PRESSURE.set(hook);
}

pub fn add_memory_pressure(size: isize, object: GcRef) {
    if let Some(hook) = ACTIVE_ADD_MEMORY_PRESSURE.get() {
        hook(size, object);
    }
}

pub fn total_memory_pressure() -> isize {
    ACTIVE_TOTAL_MEMORY_PRESSURE.get().map_or(0, |hook| hook())
}

/// Process-global callback running a non-moving old-gen-only major collection
/// (`GcAllocator::collect_oldgen_nonmoving`). The interpreter GC safepoint
/// reaches it to reclaim stable-allocated interp int/float without moving the
/// nursery — so it can fire under an active JIT (nursery non-empty), unlike
/// the moving `collect_full`. No-op when no backend is installed.
pub type CollectOldgenFn = fn();

global_hook!(static ACTIVE_COLLECT_OLDGEN: CollectOldgenFn);

/// Install the active backend's non-moving-major trampoline. Pass `None` to
/// clear.
pub fn set_active_collect_oldgen(hook: Option<CollectOldgenFn>) {
    ACTIVE_COLLECT_OLDGEN.set(hook);
}

/// Trigger a non-moving old-gen-only major collection on the active backend's
/// GC. No-op when no backend has installed a hook.
pub fn collect_oldgen_nonmoving() {
    if let Some(f) = ACTIVE_COLLECT_OLDGEN.get() {
        f();
    }
}

/// Process-global callback reporting the active GC's `heap_byte_stats`
/// (`(oldgen_total, nursery_used)`). Diagnostic: it lets a host runner split
/// GC-retained memory from host-heap growth. Deciding when to collect is not
/// among its uses — that question is [`active_major_threshold_reached`], which
/// answers it from the collector's own threshold rather than from a number a
/// caller would have to compare against a threshold of its own.
pub type HeapStatsFn = fn() -> (usize, usize);

global_hook!(static ACTIVE_HEAP_STATS: HeapStatsFn);

/// Install the active backend's `heap_byte_stats` trampoline.
pub fn set_active_heap_stats(hook: Option<HeapStatsFn>) {
    ACTIVE_HEAP_STATS.set(hook);
}

/// Report `(oldgen_total, nursery_used)` from the active backend's GC.
/// `(0, 0)` when no backend has installed a hook.
pub fn active_heap_stats() -> (usize, usize) {
    match ACTIVE_HEAP_STATS.get() {
        Some(f) => f(),
        None => (0, 0),
    }
}

/// Active-backend trampoline for `incminimark.py:get_stats`.
pub type GcMemoryStatsFn = fn() -> GcMemoryStats;

global_hook!(static ACTIVE_GC_MEMORY_STATS: GcMemoryStatsFn);

pub fn set_active_gc_memory_stats(hook: Option<GcMemoryStatsFn>) {
    ACTIVE_GC_MEMORY_STATS.set(hook);
}

pub fn active_gc_memory_stats() -> GcMemoryStats {
    match ACTIVE_GC_MEMORY_STATS.get() {
        Some(f) => f(),
        None => GcMemoryStats::default(),
    }
}

/// Active JIT CPU's `AsmMemoryManager.get_stats()` pair, exposed here so the
/// interpreter-level `gc` module does not acquire a dependency on a concrete
/// machine-code backend (`pypy/module/gc/referents.py:_get_stats`).
pub type JitBackendMemoryStatsFn = fn() -> (usize, usize);

global_hook!(static ACTIVE_JIT_BACKEND_MEMORY_STATS: JitBackendMemoryStatsFn);

pub fn set_active_jit_backend_memory_stats(hook: Option<JitBackendMemoryStatsFn>) {
    ACTIVE_JIT_BACKEND_MEMORY_STATS.set(hook);
}

pub fn active_jit_backend_memory_stats() -> (usize, usize) {
    ACTIVE_JIT_BACKEND_MEMORY_STATS
        .get()
        .map_or((0, 0), |hook| hook())
}

/// Process-global callback reporting the active GC's `major_threshold_reached`
/// (incminimark.py:1288-1290). The interpreter GC safepoint
/// (`pyre_object::gc_interp`) collects when this says the collector wants a
/// major, instead of keeping a second, poorer model of heap growth beside the
/// collector's own.
pub type MajorThresholdReachedFn = fn() -> bool;

global_hook!(static ACTIVE_MAJOR_THRESHOLD_REACHED: MajorThresholdReachedFn);

/// Install the active backend's `major_threshold_reached` trampoline.
pub fn set_active_major_threshold_reached(hook: Option<MajorThresholdReachedFn>) {
    ACTIVE_MAJOR_THRESHOLD_REACHED.set(hook);
}

/// Whether the active backend's GC has reached its next-major threshold.
/// `false` when no backend has installed a hook, so a caller with no collector
/// behind it never collects.
pub fn active_major_threshold_reached() -> bool {
    match ACTIVE_MAJOR_THRESHOLD_REACHED.get() {
        Some(f) => f(),
        None => false,
    }
}

/// Process-global callback that reports whether a raw address is owned
/// by the active backend's GC heap. Used by host-side allocators
/// (`pyre-object`'s `dealloc_items_block`) to discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks during the L1/L2 stepping-stone window:
/// `dealloc` must early-return for GC-managed pointers (the GC
/// sweeps them) and fall through to `std::alloc::dealloc` for
/// `std::alloc`-allocated ones.
pub type GcOwnsObjectFn = fn(addr: usize) -> bool;
pub type GcIsNurseryObjectFn = fn(addr: usize) -> bool;

global_hook!(static ACTIVE_GC_OWNS_OBJECT: GcOwnsObjectFn);
global_hook!(static ACTIVE_GC_IS_NURSERY_OBJECT: GcIsNurseryObjectFn);

/// Install the active backend's `is_managed_heap_object` trampoline.
pub fn set_active_gc_owns_object(hook: Option<GcOwnsObjectFn>) {
    ACTIVE_GC_OWNS_OBJECT.set(hook);
}

/// Install the active backend's nursery-membership predicate.
pub fn set_active_gc_is_nursery_object(hook: Option<GcIsNurseryObjectFn>) {
    ACTIVE_GC_IS_NURSERY_OBJECT.set(hook);
}

/// minimark.py:1900-1915 `id_or_identityhash` hook.
pub type GcIdOrIdentityHashFn = fn(addr: usize) -> usize;

global_hook!(static ACTIVE_GC_ID_OR_IDENTITYHASH: GcIdOrIdentityHashFn);

pub fn set_active_gc_id_or_identityhash(hook: Option<GcIdOrIdentityHashFn>) {
    ACTIVE_GC_ID_OR_IDENTITYHASH.set(hook);
}

/// Return a GC-move-stable address for identity hashing.
/// Falls back to `addr` when no backend is installed.
pub fn gc_id_or_identityhash(addr: usize) -> usize {
    match ACTIVE_GC_ID_OR_IDENTITYHASH.get() {
        Some(f) => f(addr),
        None => addr,
    }
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no backend has installed a hook —
/// callers treat that as "no GC owns this pointer" and fall through
/// to their non-GC dealloc path.
pub fn gc_owns_object(addr: usize) -> bool {
    match ACTIVE_GC_OWNS_OBJECT.get() {
        Some(f) => f(addr),
        None => false,
    }
}

/// Published `[start, end)` of the `gc_sync` singleton's nursery, plus the
/// tagged-pointer setting `is_valid_gc_object` consults.
///
/// `is_nursery_object_start` is `addr != 0 && !tagged && nursery.contains(addr)`
/// — three loads and two compares against fields that never move for the life
/// of the GC instance (`reset` rewinds `free`, never `start`/`size`).  Reaching
/// them through the hook chain instead costs a thread-local lookup, a `RefCell`
/// borrow and a trait-object dispatch on every root pin and every root reload,
/// which is where the query is actually hot.  Publishing them lets the common
/// process shape — one singleton, no per-thread allocator box — answer inline.
///
/// Armed only for the singleton.  `install_gc_box` gives a thread its own
/// allocator with its own nursery, which a single published pair cannot
/// describe, so that path disarms this permanently
/// ([`disarm_published_nursery`]).
static PUBLISHED_NURSERY_START: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PUBLISHED_NURSERY_END: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PUBLISHED_NURSERY_TAGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static PUBLISHED_NURSERY_ARMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static PUBLISHED_NURSERY_REFUSED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Publish the singleton's nursery bounds.  A `None` from `nursery_bounds`
/// (a stub allocator, or one whose nursery is not a fixed range) leaves the
/// fast path disarmed.
///
/// The disarm-store-rearm here is not atomic against a concurrent query, so a
/// reader can pair a stale "armed" with the incoming bounds.  Both writers are
/// the singleton installers, which run before the GC has a mutator to answer
/// for; the hook path they replace is no more synchronized, since the swap
/// behind it is not atomic against a query either.
pub fn publish_singleton_nursery(gc: &dyn GcAllocator) {
    PUBLISHED_NURSERY_ARMED.store(false, std::sync::atomic::Ordering::Release);
    if PUBLISHED_NURSERY_REFUSED.load(std::sync::atomic::Ordering::Acquire) {
        return;
    }
    let Some((start, end)) = gc.nursery_bounds() else {
        return;
    };
    PUBLISHED_NURSERY_START.store(start, std::sync::atomic::Ordering::Relaxed);
    PUBLISHED_NURSERY_END.store(end, std::sync::atomic::Ordering::Relaxed);
    PUBLISHED_NURSERY_TAGGED.store(gc.taggedpointers(), std::sync::atomic::Ordering::Relaxed);
    PUBLISHED_NURSERY_ARMED.store(true, std::sync::atomic::Ordering::Release);
}

/// Permanently retire the published fast path: a per-thread allocator box is
/// in play, so "the nursery" is no longer a single process-wide range.
pub fn disarm_published_nursery() {
    PUBLISHED_NURSERY_REFUSED.store(true, std::sync::atomic::Ordering::Release);
    PUBLISHED_NURSERY_ARMED.store(false, std::sync::atomic::Ordering::Release);
}

/// Whether `addr` is a live object in the active backend's nursery.
///
/// `#[inline]`: the armed path is a two-word range compare, and the callers
/// that matter are the per-pin / per-barrier queries next door.
#[inline]
pub fn gc_is_nursery_object(addr: usize) -> bool {
    if PUBLISHED_NURSERY_ARMED.load(std::sync::atomic::Ordering::Acquire) {
        // `is_nursery_object_start` = `is_valid_gc_object && is_in_nursery`.
        if addr == 0
            || (PUBLISHED_NURSERY_TAGGED.load(std::sync::atomic::Ordering::Relaxed)
                && addr & 1 == 1)
        {
            return false;
        }
        return addr >= PUBLISHED_NURSERY_START.load(std::sync::atomic::Ordering::Relaxed)
            && addr < PUBLISHED_NURSERY_END.load(std::sync::atomic::Ordering::Relaxed);
    }
    match ACTIVE_GC_IS_NURSERY_OBJECT.get() {
        Some(f) => f(addr),
        None if gc_sync::is_initialized() => {
            gc_sync::gc_query_reentrant(|gc| gc.is_nursery_object(addr))
        }
        None => false,
    }
}

/// Return the current address for a managed object without treating it as a
/// root. During a minor collection this follows an already-installed nursery
/// forwarding pointer; otherwise it returns `addr` unchanged.
#[inline]
pub fn gc_current_object_address(addr: usize) -> usize {
    // Only a nursery address can carry a forwarding stub: `_trace_drag_out`
    // installs one at the young object's old address, and the major collection
    // is mark-and-sweep, so nothing outside the nursery ever moves.  The
    // nursery test is a two-word range compare, where the full ownership query
    // also walks the old-generation arena index and the rawmalloc set — and
    // every root pin and every root reload asks this question.
    if addr == 0 || !gc_is_nursery_object(addr) {
        return addr;
    }
    let hdr = unsafe { header::header_of(addr) };
    if unsafe { (*hdr).is_forwarded() } {
        unsafe { header::GcHeader::forwarding_address(hdr) }
    } else {
        addr
    }
}

/// Process-global callbacks for registering/removing a Rust-stack slot
/// as a GC root with the currently active backend. Used by host-side
/// allocators whose callers need to keep a
/// just-allocated nursery pointer alive across a subsequent
/// potentially-collecting allocation.
///
/// RPython accomplishes the same thing automatically via its GC
/// transform pass (shadowstack save/restore around safepoints). pyre
/// lacks that pass, so root registration is explicit at the call
/// site. This is a documented TODO.
pub type AddRootFn = unsafe fn(slot: *mut GcRef);
pub type RemoveRootFn = fn(slot: *mut GcRef);

global_hook!(static ACTIVE_ADD_ROOT: AddRootFn);
global_hook!(static ACTIVE_REMOVE_ROOT: RemoveRootFn);

/// Install the active backend's root-register callbacks. Pass `None`
/// to clear.
pub fn set_active_root_hooks(add: Option<AddRootFn>, remove: Option<RemoveRootFn>) {
    if add.is_some() {
        // Publish remove before add so every newly registered root can be removed.
        ACTIVE_REMOVE_ROOT.set(remove);
        ACTIVE_ADD_ROOT.set(add);
    } else {
        // Withdraw add before remove so no root can be registered without removal.
        ACTIVE_ADD_ROOT.set(add);
        ACTIVE_REMOVE_ROOT.set(remove);
    }
}

/// Register a stack slot as a GC root with the active backend. No-op
/// when no backend has installed a hook.
///
/// Returns whether the hook ran, so a guard that pairs this with
/// [`gc_remove_root`] on drop can skip the removal of a root it never
/// registered.
///
/// # Safety
/// The caller must ensure the slot remains valid until
/// [`gc_remove_root`] is called with the same pointer.
pub unsafe fn gc_add_root(slot: *mut GcRef) -> bool {
    match ACTIVE_ADD_ROOT.get() {
        Some(f) => {
            unsafe { f(slot) };
            true
        }
        None => false,
    }
}

/// Remove a previously-registered root slot from the active backend.
/// No-op when no backend has installed a hook. Returns whether the hook ran.
pub fn gc_remove_root(slot: *mut GcRef) -> bool {
    match ACTIVE_REMOVE_ROOT.get() {
        Some(f) => {
            f(slot);
            true
        }
        None => false,
    }
}

/// rgc.py `FinalizerQueue.register_finalizer` - routed to the active backend GC.
pub type RegisterFinalizerFn = fn(fq_index: usize, obj: GcRef, trigger: FinalizerTriggerFn);
/// rgc.py `FinalizerQueue.next_dead` - routed to the active backend GC.
pub type FinalizerNextDeadFn = fn(fq_index: usize) -> Option<GcRef>;

global_hook!(static ACTIVE_REGISTER_FINALIZER: RegisterFinalizerFn);
global_hook!(static ACTIVE_FINALIZER_NEXT_DEAD: FinalizerNextDeadFn);

/// Install the active backend's finalizer trampolines. Pass `None` to clear.
pub fn set_active_finalizer_hooks(
    register: Option<RegisterFinalizerFn>,
    next_dead: Option<FinalizerNextDeadFn>,
) {
    ACTIVE_REGISTER_FINALIZER.set(register);
    ACTIVE_FINALIZER_NEXT_DEAD.set(next_dead);
}

/// Register an object with an RPython-style finalizer queue on the active GC.
pub fn gc_register_finalizer(fq_index: usize, obj: GcRef, trigger: FinalizerTriggerFn) {
    match ACTIVE_REGISTER_FINALIZER.get() {
        Some(f) => f(fq_index, obj, trigger),
        None => gc_sync::gc_op(|gc| gc.register_finalizer(fq_index, obj, trigger)),
    }
}

/// Pop one object from the active GC's RPython-style finalizer death queue.
pub fn gc_fq_next_dead(fq_index: usize) -> Option<GcRef> {
    match ACTIVE_FINALIZER_NEXT_DEAD.get() {
        Some(f) => f(fq_index),
        None => gc_sync::gc_op(|gc| gc.finalizer_next_dead(fq_index)),
    }
}

/// rgc.enable / rgc.disable — toggle automatic major-collection progress
/// on the process-global GC.
pub fn gc_set_enabled(enabled: bool) {
    gc_sync::gc_op(|gc| if enabled { gc.enable() } else { gc.disable() })
}

/// Process-global callback that performs a host-side write barrier through
/// the currently active backend GC.
pub type WriteBarrierFn = fn(obj: GcRef);

global_hook!(static ACTIVE_WRITE_BARRIER: WriteBarrierFn);
global_hook!(static ACTIVE_WRITE_BARRIER_MANAGED: WriteBarrierFn);

/// Install the active backend's write-barrier callback. Pass `None` to clear.
pub fn set_active_write_barrier(hook: Option<WriteBarrierFn>) {
    ACTIVE_WRITE_BARRIER.set(hook);
}

/// Install the barrier entry for objects already known to be GC-managed.
pub fn set_active_write_barrier_managed(hook: Option<WriteBarrierFn>) {
    ACTIVE_WRITE_BARRIER_MANAGED.set(hook);
}

/// Perform a write barrier through the active backend.
///
/// Calling convention: callers must invoke this before storing a GC reference
/// into `obj`, matching [`GcAllocator::write_barrier`]. The active callback is
/// a process-global cell installed with [`set_active_write_barrier`] as a
/// [`WriteBarrierFn`]; this is a no-op when no barrier is installed.
pub fn gc_write_barrier(obj: GcRef) {
    if let Some(f) = ACTIVE_WRITE_BARRIER.get() {
        f(obj)
    }
}

/// Perform a write barrier without repeating the hybrid-heap ownership query.
/// Falls back to the safe barrier when a backend has no specialized entry.
pub fn gc_write_barrier_managed(obj: GcRef) {
    if let Some(f) = ACTIVE_WRITE_BARRIER_MANAGED.get() {
        f(obj)
    } else {
        gc_write_barrier(obj)
    }
}

// ── TEMPORARY DIAGNOSTIC: blackhole-materialized object registry ──
//
// Enabled by `MAJIT_GC_BH_PROBE`. Records every block a backend's `bh_new*`
// hands out so `do_collect_nursery` can check, at the end of a minor, that
// none of them still names a nursery address. Remove with the investigation.

thread_local! {
    /// (address, payload size, origin) — origin 1 = born old, 2 = promoted.
    static BH_PROBE_OBJECTS: std::cell::RefCell<Vec<(usize, usize, u8)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// One distinct `(type, offset, origin)` class of post-minor nursery reference.
pub struct BhProbeViolation {
    pub minor: usize,
    pub origin: u8,
    pub holder: usize,
    pub tid: u32,
    pub payload_size: usize,
    pub offset: usize,
    pub value: usize,
    pub forwarded: bool,
    pub type_size: usize,
    pub item_size: usize,
    pub length_offset: usize,
    pub gc_ptr_offsets: Vec<usize>,
    pub custom_trace: bool,
    pub is_object: bool,
    pub track_young_ptrs: bool,
    pub remembered: bool,
    pub barriered_ever: bool,
    pub traced_this_minor: bool,
    pub store_sites: u32,
}

thread_local! {
    static BH_PROBE_VIOLATIONS: std::cell::RefCell<Vec<BhProbeViolation>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Whether this `(type, offset, origin)` class was already recorded.
pub fn bh_probe_violation_seen(tid: u32, offset: usize, origin: u8) -> bool {
    BH_PROBE_VIOLATIONS
        .try_with(|v| {
            v.borrow()
                .iter()
                .any(|e| e.tid == tid && e.offset == offset && e.origin == origin)
        })
        .unwrap_or(false)
}

/// Append newly seen classes; returns the total class count.
pub fn bh_probe_record_violations(fresh: Vec<BhProbeViolation>) -> usize {
    BH_PROBE_VIOLATIONS
        .try_with(|v| {
            let mut v = v.borrow_mut();
            v.extend(fresh);
            v.len()
        })
        .unwrap_or(0)
}

pub fn bh_probe_violation_report(minor: usize) -> String {
    use std::fmt::Write as _;
    let mut out = format!("BH PROBE: post-minor nursery references, through minor #{minor}\n");
    let _ = BH_PROBE_VIOLATIONS.try_with(|v| {
        for e in v.borrow().iter() {
            let _ = write!(
                out,
                "  minor#{} {} holder={:#x} tid={} payload={} offset={}({}) value={:#x} \
                 forwarded={} | type: size={} item_size={} length_offset={} gc_ptr_offsets={:?} \
                 custom_trace={} is_object={} | holder: track_young={} remembered={} \
                 barriered_ever={} traced_this_minor={} store_sites={:#b} | layout={:?}\n",
                e.minor,
                if e.origin == BH_PROBE_ORIGIN_PROMOTED {
                    "promoted"
                } else {
                    "born-old"
                },
                e.holder,
                e.tid,
                e.payload_size,
                e.offset,
                bh_probe_field_name(e.tid, e.offset),
                e.value,
                e.forwarded,
                e.type_size,
                e.item_size,
                e.length_offset,
                e.gc_ptr_offsets,
                e.custom_trace,
                e.is_object,
                e.track_young_ptrs,
                e.remembered,
                e.barriered_ever,
                e.traced_this_minor,
                e.store_sites,
                bh_probe_layout(e.tid),
            );
        }
    });
    out
}

/// Type ids the post-minor scan must not read as a plain word array.
static BH_PROBE_IGNORED_TIDS: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

/// Exempt a type from the post-minor nursery-reference scan.
///
/// A `JitFrame`'s slots are typed by its own `jf_gcmap`, not by the type table,
/// so a slot outside that map legitimately holds a stale scratch word that can
/// fall inside the nursery range.
pub fn bh_probe_ignore_tid(type_id: u32) {
    if let Ok(mut v) = BH_PROBE_IGNORED_TIDS.lock() {
        if !v.contains(&type_id) {
            v.push(type_id);
        }
    }
}

pub fn bh_probe_tid_ignored(type_id: u32) -> bool {
    BH_PROBE_IGNORED_TIDS
        .lock()
        .map(|v| v.contains(&type_id))
        .unwrap_or(false)
}

/// Origin tag for a block born straight into the old generation.
pub const BH_PROBE_ORIGIN_BORN_OLD: u8 = 1;
/// Origin tag for a nursery survivor the minor promoted.
pub const BH_PROBE_ORIGIN_PROMOTED: u8 = 2;

/// Whether the blackhole-object probe is enabled.
pub fn bh_probe_enabled() -> bool {
    // wasm32-unknown-unknown has no process environment, so the guest cannot
    // read MAJIT_GC_BH_PROBE; the compile-time feature is its gate instead.
    if cfg!(target_arch = "wasm32") {
        return cfg!(feature = "bh_probe");
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MAJIT_GC_BH_PROBE").is_some())
}

/// Record an old-generation block and its payload size.
pub fn note_bh_object(addr: usize, payload_size: usize, origin: u8) {
    if addr == 0 || !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_OBJECTS.try_with(|v| v.borrow_mut().push((addr, payload_size, origin)));
}

/// Visit every recorded old-generation block.
pub fn with_bh_objects<R>(f: impl FnOnce(&[(usize, usize, u8)]) -> R) -> Option<R> {
    BH_PROBE_OBJECTS.try_with(|v| f(&v.borrow())).ok()
}

thread_local! {
    /// Objects `remember_young_pointer` has ever put on the remembered set.
    static BH_PROBE_BARRIERED: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
    /// Objects `trace_and_update_object` visited during the current minor.
    static BH_PROBE_TRACED: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

pub fn bh_probe_note_barriered(addr: usize) {
    if !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_BARRIERED.try_with(|s| s.borrow_mut().insert(addr));
}

pub fn bh_probe_note_traced(addr: usize) {
    if !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_TRACED.try_with(|s| s.borrow_mut().insert(addr));
}

pub fn bh_probe_clear_traced() {
    if !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_TRACED.try_with(|s| s.borrow_mut().clear());
}

pub fn bh_probe_was_barriered(addr: usize) -> bool {
    BH_PROBE_BARRIERED
        .try_with(|s| s.borrow().contains(&addr))
        .unwrap_or(false)
}

pub fn bh_probe_was_traced(addr: usize) -> bool {
    BH_PROBE_TRACED
        .try_with(|s| s.borrow().contains(&addr))
        .unwrap_or(false)
}

thread_local! {
    /// (object, offset) -> bitmask of instrumented store sites that wrote it.
    static BH_PROBE_STORES: std::cell::RefCell<
        std::collections::HashMap<(usize, usize), u32>,
    > = std::cell::RefCell::new(std::collections::HashMap::new());
}

pub fn bh_probe_note_store(obj: usize, offset: usize, site: u32) {
    if obj == 0 || !bh_probe_enabled() {
        return;
    }
    let _ = BH_PROBE_STORES.try_with(|m| {
        *m.borrow_mut().entry((obj, offset)).or_insert(0) |= 1u32 << site;
    });
}

pub fn bh_probe_store_sites(obj: usize, offset: usize) -> u32 {
    BH_PROBE_STORES
        .try_with(|m| m.borrow().get(&(obj, offset)).copied().unwrap_or(0))
        .unwrap_or(0)
}

/// Published nursery bounds so the probe can scan without reaching the GC.
pub static BH_PROBE_NURSERY_LO: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
pub static BH_PROBE_NURSERY_HI: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    static BH_PROBE_PHASE: std::cell::Cell<&'static str> = const { std::cell::Cell::new("interp") };
}

pub struct BhProbePhase(&'static str);

impl BhProbePhase {
    pub fn enter(name: &'static str) -> Self {
        let prev = BH_PROBE_PHASE
            .try_with(|c| c.replace(name))
            .unwrap_or("interp");
        BhProbePhase(prev)
    }
}

impl Drop for BhProbePhase {
    fn drop(&mut self) {
        let _ = BH_PROBE_PHASE.try_with(|c| c.set(self.0));
    }
}

pub fn bh_probe_phase() -> &'static str {
    BH_PROBE_PHASE.try_with(|c| c.get()).unwrap_or("interp")
}

/// Walk every recorded blackhole object looking for a word that names the
/// nursery, and panic naming the phase that produced it.
pub fn bh_probe_scan(reason: &str) {
    if !bh_probe_enabled() {
        return;
    }
    let lo = BH_PROBE_NURSERY_LO.load(std::sync::atomic::Ordering::Relaxed);
    let hi = BH_PROBE_NURSERY_HI.load(std::sync::atomic::Ordering::Relaxed);
    if lo == 0 {
        return;
    }
    let hit = with_bh_objects(|objects| {
        for &(addr, payload_size, origin) in objects {
            // Promoted blocks are the minor's own output; this scan is about
            // what a blackhole store just left behind in a born-old block.
            if origin != BH_PROBE_ORIGIN_BORN_OLD || (addr >= lo && addr < hi) {
                continue;
            }
            // TRACK_YOUNG_PTRS clear means the object is already on the
            // remembered set, so a young field in it is legal and will be
            // traced.  Only a flagged object naming the nursery is a lost edge.
            let hdr = (addr - crate::header::GcHeader::SIZE) as *const crate::header::GcHeader;
            if unsafe { !(*hdr).has_flag(crate::flags::TRACK_YOUNG_PTRS) } {
                continue;
            }
            for offset in (0..payload_size).step_by(std::mem::size_of::<usize>()) {
                let word = unsafe { *((addr + offset) as *const usize) };
                if word >= lo && word < hi {
                    return Some((addr, offset, word));
                }
            }
        }
        None
    });
    if let Some(Some((addr, offset, word))) = hit {
        panic!(
            "BH SCAN: blackhole object named the nursery — reason={} phase={} \
             holder={:#x} offset={}({}) value={:#x} sites={:#b}",
            reason,
            bh_probe_phase(),
            addr,
            offset,
            bh_probe_field_name(37, offset),
            word,
            bh_probe_store_sites(addr, offset),
        );
    }
}

/// Offset -> field-name table for the probe's diagnostics, published by the
/// crate that owns the layout so the probe does not have to guess.
static BH_PROBE_FIELD_NAMES: std::sync::Mutex<Vec<(u32, usize, &'static str)>> =
    std::sync::Mutex::new(Vec::new());

pub fn bh_probe_set_field_names(entries: &[(u32, usize, &'static str)]) {
    if let Ok(mut v) = BH_PROBE_FIELD_NAMES.lock() {
        v.extend_from_slice(entries);
    }
}

pub fn bh_probe_layout(type_id: u32) -> Vec<(usize, &'static str)> {
    let mut rows: Vec<(usize, &'static str)> = BH_PROBE_FIELD_NAMES
        .lock()
        .map(|v| {
            v.iter()
                .filter(|&&(t, _, _)| t == type_id)
                .map(|&(_, o, n)| (o, n))
                .collect()
        })
        .unwrap_or_default();
    rows.sort_unstable();
    rows
}

pub fn bh_probe_field_name(type_id: u32, offset: usize) -> &'static str {
    BH_PROBE_FIELD_NAMES
        .lock()
        .ok()
        .and_then(|v| {
            v.iter()
                .find(|&&(t, o, _)| t == type_id && o == offset)
                .map(|&(_, _, n)| n)
        })
        .unwrap_or("?")
}

#[cfg(test)]
mod headerless_no_collect_tests {
    use super::*;

    /// A `GcAllocator` that only implements the collecting headerless form, so
    /// the default `alloc_nursery_headerless_no_collect` has to forward to it.
    struct ForwardingGc {
        headerless_calls: usize,
    }

    impl GcAllocator for ForwardingGc {
        fn alloc_nursery(&mut self, _size: usize) -> GcRef {
            GcRef(0x1000)
        }
        fn alloc_nursery_headerless(&mut self, _size: usize) -> GcRef {
            self.headerless_calls += 1;
            GcRef(0x2000)
        }
        fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
            self.alloc_nursery(size)
        }
        fn alloc_varsize(&mut self, base: usize, item: usize, len: usize) -> GcRef {
            self.alloc_nursery(base + item * len)
        }
        fn alloc_varsize_no_collect(&mut self, base: usize, item: usize, len: usize) -> GcRef {
            self.alloc_varsize(base, item, len)
        }
        fn write_barrier(&mut self, _obj: GcRef) {}
        fn jit_remember_young_pointer_from_array(&mut self, _obj: GcRef) {}
        fn remember_young_pointer_from_array2(
            &mut self,
            _obj: GcRef,
            _index: usize,
            _card_page_shift: u32,
        ) {
        }
        fn collect_nursery(&mut self) {}
        fn collect_full(&mut self) {}
        fn nursery_free(&self) -> *mut u8 {
            std::ptr::null_mut()
        }
        fn nursery_free_addr(&self) -> usize {
            0
        }
        fn nursery_top(&self) -> *const u8 {
            std::ptr::null()
        }
        fn nursery_top_addr(&self) -> usize {
            0
        }
        fn max_nursery_object_size(&self) -> usize {
            0
        }
    }

    #[test]
    fn no_collect_default_forwards_to_the_collecting_headerless_form() {
        let mut gc = ForwardingGc {
            headerless_calls: 0,
        };
        assert_eq!(gc.alloc_nursery_headerless_no_collect(16), GcRef(0x2000));
        assert_eq!(gc.headerless_calls, 1);
    }

    fn stub_alloc(size: usize) -> GcRef {
        GcRef(0x4000 + size)
    }

    #[test]
    fn active_hook_round_trips_and_is_null_when_absent() {
        // Uninstalled: callers must see null so they can keep their own path.
        set_active_alloc_nursery_headerless_no_collect(None);
        assert_eq!(alloc_nursery_headerless_no_collect(16), GcRef(0));

        set_active_alloc_nursery_headerless_no_collect(Some(stub_alloc));
        assert_eq!(alloc_nursery_headerless_no_collect(16), GcRef(0x4010));

        set_active_alloc_nursery_headerless_no_collect(None);
        assert_eq!(alloc_nursery_headerless_no_collect(16), GcRef(0));
    }
}
