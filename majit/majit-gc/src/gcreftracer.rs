//! Per-loop GC table for reference constants baked into compiled traces.
//!
//! Counterpart of `rpython/jit/backend/llsupport/gcreftracer.py`. A
//! compiled loop/bridge that references constant GC objects must keep
//! those references alive and up to date across moving collections.
//! Instead of baking the raw `GcRef` value as a machine-code immediate
//! (which a moving GC cannot find or update), the backend bakes the
//! address of a per-loop array of reference slots and emits a
//! `LoadFromGcTable(index)` load (`x86/assembler.py:1545`
//! `genop_load_from_gc_table`). Each slot is a GC root: the collector
//! forwards it in place during a stop-the-world collection, so the next
//! load observes the relocated object.
//!
//! Upstream models the array with a `GCREFTRACER` `GcStruct`
//! (`gcreftracer.py:7-11`) carrying `array_base_addr` + `array_length`,
//! registered with the GC via a custom trace hook
//! (`register_custom_trace_hook`, `gcreftracer.py`) and *reached* by
//! the collector through the loop token's `asmmemmgr_gcreftracers`, which
//! in RPython is a GC-managed list (`x86/assembler.py:823`,
//! `model.py:294`) so the tracer header sits in the live object graph.
//!
//! pyre has the custom-trace-hook facility — `TypeInfo::custom_trace`
//! (`trace.rs`, `register_custom_trace_hook` parity), already used by
//! `JITFRAME` (`jitframe.rs`), the structural twin of `GCREFTRACER` —
//! so a managed `GCREFTRACER` GcStruct with a slot-tracing hook is
//! portable *in isolation*. What is not yet portable is the
//! *reachability*: pyre's `CompiledLoopToken` is a Rust-owned struct and
//! its `asmmemmgr_gcreftracers` is a `Mutex<Vec<Arc<dyn Any>>>` keepalive
//! (`CompiledLoopToken` in `majit-backend/src/lib.rs`), not a GC edge — the
//! collector has no
//! object-graph path to a managed header. A GcStruct header would still
//! need a GC root to be reached, and that root would be a walker over the
//! Rust-owned keepalive — a header allocation, a hook, and a mark step
//! added without removing the walker.
//!
//! So pyre forwards the slots directly from an extra root walker
//! (`shadow_stack::register_extra_root_walker`, walked at
//! `collector.rs`'s `do_collect_nursery` and
//! `rescan_major_nonstack_roots_and_drain`) over
//! the live-table registry — the analog of RPython reaching the tracer
//! through the CLT keepalive. Convergence path: once
//! `asmmemmgr_gcreftracers` (or `CompiledLoopToken` itself) is GC-traced,
//! this collapses to a managed `GCREFTRACER` GcStruct + custom trace
//! hook, matching upstream exactly.
//!
//! One upstream duty does not carry over. `free_loop_and_bridges` calls
//! `clear_gcref_tracer`, which zeroes `array_length`, because upstream's
//! slot array is reserved inside the code block (`reserve_gcref_table`)
//! and is freed with it — a tracer outliving the block would otherwise
//! hand the collector freed memory. Here the slots are the table's own
//! `Box`, allocated and released with it, so the array cannot outlive its
//! storage and there is nothing to turn off.

use parking_lot::RwLock;
use std::cell::Cell;
use std::sync::{Arc, Weak};

use majit_ir::GcRef;

/// `llsupport/symbolic.py` `WORD`.
const WORD: usize = core::mem::size_of::<usize>();

/// A per-loop array of reference-constant slots.
///
/// `gcreftracer.py` `GCREFTRACER` stores `array_base_addr` +
/// `array_length`. The array itself sits at the start of the loop's
/// machine-code block (`assembler.py reserve_gcref_table`), where the
/// `LoadFromGcTable` genop reaches it PC-relative; a backend that keeps the
/// array on the Rust heap instead ([`from_gcrefs`](GcTable::from_gcrefs))
/// hands out the heap address as `array_base_addr` and bakes it as an
/// absolute immediate. Either address is stable for the table's whole life.
pub struct GcTable {
    array_base_addr: usize,
    array_length: usize,
    /// The heap-owned array, when the slots do not live in a code block.
    _owned: Option<Box<[Cell<GcRef>]>>,
}

// SAFETY: a `GcTable`'s slots are only mutated through `trace`, which
// runs exclusively during a stop-the-world collection
// (`collector.rs` `do_collect_nursery` / major), when no JIT or
// interpreter thread is reading the slots. Construction fills the slots
// before the `Arc` is shared, and they are never written again outside
// `trace`. The `Cell` provides interior mutability for in-place
// forwarding; the `Send`/`Sync` bounds let the `Arc<GcTable>` live on
// `CompiledLoopToken.asmmemmgr_gcreftracers` and a `Weak<GcTable>` in
// the global registry.
unsafe impl Send for GcTable {}
unsafe impl Sync for GcTable {}

/// Live per-loop tables, walked as GC roots. A strong reference is held by
/// `CompiledLoopToken.asmmemmgr_gcreftracers` (parity
/// `gcreftracers.append(tracer)`, `x86/assembler.py`); the registry keeps
/// only a `Weak`, and a `Weak` that stops upgrading is the whole of
/// deregistration. No `free_loop` clears `asmmemmgr_gcreftracers`, and
/// nothing dispatches `Backend::free_loop` at all — the release is `Arc`
/// drop, driven by the memory manager retiring a token in
/// `try_to_free_some_loops`.
///
/// That drop is not always the CLT's. On cranelift a bridge's table is
/// pinned a second time by every `BridgeData` that can dispatch to the
/// bridge, so it outlives the token it was registered against for as long
/// as a fail descr in another token holds one. The table dies with its
/// last strong holder, whichever that is; on dynasm the CLT is the only
/// one.
static LIVE_GC_TABLES: RwLock<Vec<Weak<GcTable>>> = RwLock::new(Vec::new());

/// Test-only lock modeling the stop-the-world invariant that no table is
/// dropped while a walk is in flight. The harness runs tests in parallel, so
/// a collector test's collection — which walks this registry through the
/// globally-registered [`gc_table_extra_root_walker`] once any table has
/// existed — can call [`walk_all_gc_tables`] concurrently with a registry
/// test's table drop, transiently upgrading a `Weak` the dropping test
/// expects to be dead. A registry test takes the write side to exclude every
/// walk across its drop/observe window; each walk takes the read side.
/// Compiled out in production, where the STW collector already guarantees no
/// concurrent drop.
#[cfg(test)]
static GC_TABLE_WALK_LOCK: RwLock<()> = RwLock::new(());

/// Tables built since the last minor collection. `GCREFTRACER` is an
/// ordinary old object upstream: writing its slots at construction puts it
/// in MiniMark's `old_objects_pointing_to_young` for exactly one minor
/// collection, which promotes every referent it holds, and no later minor
/// visits it again because its slots are never written afterwards. A major
/// collection marks through every live tracer regardless. This list is that
/// remembered set: [`walk_all_gc_tables_inner`] drains it on a minor walk and
/// walks the whole registry on a major one.
static PENDING_MINOR_TABLES: parking_lot::Mutex<Vec<Weak<GcTable>>> =
    parking_lot::Mutex::new(Vec::new());

impl GcTable {
    /// Build a per-loop table from the rewrite's gcref output list and
    /// register it for GC forwarding.
    ///
    /// `gcreftracer.py` `make_framework_tracer` warns that the
    /// tracer allocation can itself trigger a GC, so it writes the
    /// gcrefs into the raw array only afterwards. Here the
    /// `Box<[Cell<GcRef>]>` is a plain Rust heap allocation that does not
    /// go through the moving GC, so that hazard cannot arise; the slots
    /// are filled before the `Arc` exists, and registration happens last,
    /// so no collection can observe a half-filled, registered table.
    pub fn from_gcrefs(gcrefs: &[GcRef]) -> Arc<GcTable> {
        // Ensure the forwarding walker is installed before the table can
        // be observed by a collection. `register_extra_root_walker`
        // dedups by fn address, so this is idempotent across every
        // compiled loop; before the first table exists there is nothing
        // to walk, so installing lazily here is equivalent to a one-time
        // backend init without depending on a separate init call site.
        install_gc_table_walker();
        let slots: Box<[Cell<GcRef>]> = gcrefs.iter().map(|g| Cell::new(*g)).collect();
        let table = GcTable {
            array_base_addr: slots.as_ptr() as usize,
            array_length: slots.len(),
            _owned: Some(slots),
        };
        Self::register(table)
    }

    /// `gcreftracer.py` `make_framework_tracer`: write the gcrefs into the
    /// raw array reserved at `array_base_addr` inside a machine-code block
    /// (`assembler.py patch_gcref_table`) and register the tracer. The
    /// block is written under the assembler-writing bracket
    /// (`rmmap.enter_assembler_writing`).
    ///
    /// # Safety
    /// `array_base_addr` must address `gcrefs.len()` word-aligned words
    /// inside a JIT mapping that outlives the returned table.
    #[cfg(not(target_arch = "wasm32"))]
    pub unsafe fn in_code(array_base_addr: usize, gcrefs: &[GcRef]) -> Arc<GcTable> {
        install_gc_table_walker();
        {
            let _writing = crate::rmmap::AssemblerWriting::enter();
            for (i, g) in gcrefs.iter().enumerate() {
                unsafe { *((array_base_addr + i * WORD) as *mut GcRef) = *g };
            }
        }
        Self::register(GcTable {
            array_base_addr,
            array_length: gcrefs.len(),
            _owned: None,
        })
    }

    fn register(table: GcTable) -> Arc<GcTable> {
        let table = Arc::new(table);
        register_table(&table);
        PENDING_MINOR_TABLES.lock().push(Arc::downgrade(&table));
        table
    }

    /// Raw base address of the slot array, baked by the backend genop as
    /// the `LoadFromGcTable` base immediate. `gcreftracer.py:9`
    /// `array_base_addr`.
    pub fn base_addr(&self) -> usize {
        self.array_base_addr
    }

    /// Number of reference-constant slots. `gcreftracer.py:10`
    /// `array_length`.
    pub fn len(&self) -> usize {
        self.array_length
    }

    pub fn is_empty(&self) -> bool {
        self.array_length == 0
    }

    /// Read slot `i`.
    pub fn slot(&self, i: usize) -> GcRef {
        assert!(i < self.array_length);
        // SAFETY: `array_base_addr + i*WORD` is a slot of a live array
        // (see the struct invariant); slots are read outside a collection
        // and written only by `trace` inside one.
        unsafe { *((self.array_base_addr + i * WORD) as *const GcRef) }
    }

    /// Forward every slot in place. `gcreftracer.py:13-23`
    /// `gcrefs_trace`: each `array_base_addr + i*WORD` slot is handed to
    /// the GC as a root; writing back through the visitor forwards the
    /// constant if the moving GC relocated the referenced object.
    pub fn trace(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        // `gcrefs_trace` brackets the walk with
        // `rmmap.enter_assembler_writing()`: an in-code array is part of a
        // JIT mapping.
        #[cfg(not(target_arch = "wasm32"))]
        let _writing = self
            ._owned
            .is_none()
            .then(crate::rmmap::AssemblerWriting::enter);
        for i in 0..self.array_length {
            let p = (self.array_base_addr + i * WORD) as *mut GcRef;
            // SAFETY: see `slot`; this runs inside a stop-the-world
            // collection, the only writer.
            unsafe {
                let mut r = *p;
                visitor(&mut r);
                *p = r;
            }
        }
    }
}

/// Append a live table to the registry, sweeping out tables whose loop
/// tokens have already been freed.
fn register_table(table: &Arc<GcTable>) {
    let mut guard = LIVE_GC_TABLES.write();
    guard.retain(|w| w.strong_count() > 0);
    guard.push(Arc::downgrade(table));
}

/// Forward the slots of every live per-loop table. Registered once via
/// [`install_gc_table_walker`]; fires at both the minor
/// (`do_collect_nursery`) and major
/// (`rescan_major_nonstack_roots_and_drain`) collection
/// phases.
fn walk_all_gc_tables(visitor: &mut dyn FnMut(&mut GcRef)) {
    // In test builds, hold the read side of the walk lock so a registry
    // test's drop/observe window (which takes the write side) is never
    // interleaved with a walk. Compiles out in production.
    #[cfg(test)]
    let _walk = GC_TABLE_WALK_LOCK.read();
    walk_all_gc_tables_inner(visitor);
}

/// The walk itself, without the test-only lock, so a registry test that
/// already holds the write side observes the registry without re-entering
/// the lock.
fn walk_all_gc_tables_inner(visitor: &mut dyn FnMut(&mut GcRef)) {
    // A minor collection reaches only the tables the remembered set names
    // (see `PENDING_MINOR_TABLES`); every other live table already holds
    // promoted referents, which a minor collection does not move.
    if crate::shadow_stack::extra_root_walk_kind() == crate::shadow_stack::ExtraRootWalkKind::Minor
    {
        let pending = std::mem::take(&mut *PENDING_MINOR_TABLES.lock());
        for table in pending.iter().filter_map(Weak::upgrade) {
            table.trace(visitor);
        }
        return;
    }
    // Snapshot the live tables under a read guard, then release the lock
    // before tracing (same snapshot-then-iterate discipline as
    // `walk_extra_roots`, `shadow_stack.rs`). Dead `Weak`s are
    // filtered by `upgrade`.
    let live: Vec<Arc<GcTable>> = {
        let guard = LIVE_GC_TABLES.read();
        guard.iter().filter_map(|w| w.upgrade()).collect()
    };
    for table in &live {
        table.trace(visitor);
    }
}

/// Extra-root-walker entry point. A plain `fn` so it can be deduped by
/// address in `register_extra_root_walker`.
fn gc_table_extra_root_walker(visitor: &mut dyn FnMut(&mut GcRef)) {
    walk_all_gc_tables(visitor);
}

/// Install the per-loop gc_table forwarding walker. Idempotent —
/// `register_extra_root_walker` dedups by fn address
/// (`shadow_stack.rs`). Call once at backend init.
pub fn install_gc_table_walker() {
    crate::shadow_stack::register_extra_root_walker(gc_table_extra_root_walker);
}

#[cfg(test)]
mod tests {
    use super::*;

    // `LIVE_GC_TABLES` is a process-global registry; in production it is
    // only mutated outside a collection (table build at compile time) and
    // only read inside a stop-the-world collection, so there is never a
    // concurrent build-vs-walk. The test harness runs tests in parallel, so
    // besides serializing the table-touching tests against each other, the
    // write side of [`GC_TABLE_WALK_LOCK`] also excludes any concurrent
    // collector-test collection whose walk would otherwise transiently
    // resurrect a table this test is dropping — modeling the STW invariant.

    #[test]
    fn trace_forwards_slots_in_place() {
        let _serialize = GC_TABLE_WALK_LOCK.write();
        let table = GcTable::from_gcrefs(&[GcRef(0x1000), GcRef(0x2000)]);
        // A moving collection relocates 0x1000 -> 0x9000.
        table.trace(&mut |r| {
            if r.0 == 0x1000 {
                r.0 = 0x9000;
            }
        });
        assert_eq!(table.slot(0), GcRef(0x9000));
        assert_eq!(table.slot(1), GcRef(0x2000));
        assert_eq!(
            table.base_addr(),
            table._owned.as_ref().unwrap().as_ptr() as usize
        );
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn baked_load_reads_forwarded_ref_after_move() {
        // End-to-end model of a ref constant surviving across compilation and
        // later execution. The backend bakes a `LoadFromGcTable` as a load of
        // `*(base + index*WORD)` (rewrite.py `remove_constptr`), with
        // `base` fixed at compile time. A moving collection forwards the slot
        // value in place (gcreftracer.py `trace`), so a later execution of the
        // baked load observes the relocated address — no stale immediate. This
        // is the shared dynasm/cranelift `LoadFromGcTable` contract; wasm never
        // runs the GC rewrite (loud-panic), so it has no moving-GC ref-const
        // path to cover.
        let _serialize = GC_TABLE_WALK_LOCK.write();
        let table = GcTable::from_gcrefs(&[GcRef(0x1000), GcRef(0x2000)]);
        // `base` is the value baked into the trace at compile time.
        let base = table.base_addr();
        // `GcRef` is `#[repr(transparent)]` over `usize` and `Cell<GcRef>`
        // shares that layout, so `base + index*WORD` addresses `slots[index]`
        // exactly as the emitted machine load does.
        let baked_load = |index: usize| -> usize {
            // SAFETY: `index < len`; the slot is a live `Cell<GcRef>` (usize).
            unsafe { *((base + index * core::mem::size_of::<usize>()) as *const usize) }
        };
        assert_eq!(
            baked_load(0),
            0x1000,
            "pre-move load reads the original ref"
        );
        assert_eq!(baked_load(1), 0x2000);
        // A moving collection relocates 0x1000 -> 0x9000.
        table.trace(&mut |r| {
            if r.0 == 0x1000 {
                r.0 = 0x9000;
            }
        });
        assert_eq!(
            table.base_addr(),
            base,
            "baked base address is stable across the collection"
        );
        assert_eq!(
            baked_load(0),
            0x9000,
            "post-move load via the baked base reads the forwarded ref"
        );
        assert_eq!(baked_load(1), 0x2000, "unmoved ref unchanged");
    }

    #[test]
    fn dropping_table_deregisters_from_walk() {
        let _serialize = GC_TABLE_WALK_LOCK.write();
        // A sentinel unlikely to collide with any other test's table.
        const SENTINEL: GcRef = GcRef(0x0000_DEAD_BEEF);
        // The write side is already held, so count through the unlocked
        // walk to avoid re-entering the lock.
        let count_sentinels = || {
            let mut n = 0usize;
            walk_all_gc_tables_inner(&mut |r| {
                if *r == SENTINEL {
                    n += 1;
                }
            });
            n
        };
        {
            let _table = GcTable::from_gcrefs(&[SENTINEL]);
            assert_eq!(count_sentinels(), 1, "live table must be walked");
        }
        // The Arc dropped; the registry's Weak no longer upgrades.
        assert_eq!(count_sentinels(), 0, "freed table must not be walked");
    }
}
