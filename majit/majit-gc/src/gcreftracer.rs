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
//! (`register_custom_trace_hook`, `gcreftracer.py:30`). pyre has no
//! custom-trace-hook facility; the established equivalent is the extra
//! root walker registry (`shadow_stack::register_extra_root_walker`,
//! walked at `collector.rs:668` minor and `collector.rs:1185` major), so
//! a single module walker forwards every live table's slots.

use std::cell::Cell;
use std::sync::{Arc, RwLock, Weak};

use majit_ir::GcRef;

/// A per-loop array of reference-constant slots.
///
/// `gcreftracer.py:7-11` `GCREFTRACER` stores `array_base_addr` +
/// `array_length`; here the `Box<[Cell<GcRef>]>` *is* that array:
/// [`base_addr`](GcTable::base_addr) is `array_base_addr` and
/// [`len`](GcTable::len) is `array_length`. The `Box` is a stable Rust
/// heap allocation (never relocated by the moving GC), so its base
/// address is valid for the table's whole life — that is what the
/// backend bakes as the `LoadFromGcTable` base immediate.
pub struct GcTable {
    slots: Box<[Cell<GcRef>]>,
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

/// Live per-loop tables, walked as GC roots. The strong reference is
/// held by `CompiledLoopToken.asmmemmgr_gcreftracers` (parity
/// `gcreftracers.append(tracer)`, `x86/assembler.py:823`); the registry
/// keeps only a `Weak`, so when the loop token is freed the table drops
/// and its registry entry becomes dangling — deregistration needs no
/// explicit `free_loop` hook (neither backend's `free_loop` clears
/// `asmmemmgr_gcreftracers`; both rely on `Arc<CompiledLoopToken>`
/// drop).
static LIVE_GC_TABLES: RwLock<Vec<Weak<GcTable>>> = RwLock::new(Vec::new());

impl GcTable {
    /// Build a per-loop table from the rewrite's gcref output list and
    /// register it for GC forwarding.
    ///
    /// `gcreftracer.py:26-43` `make_framework_tracer` warns that the
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
        let table = Arc::new(GcTable { slots });
        register_table(&table);
        table
    }

    /// Raw base address of the slot array, baked by the backend genop as
    /// the `LoadFromGcTable` base immediate. `gcreftracer.py:9`
    /// `array_base_addr`.
    pub fn base_addr(&self) -> usize {
        self.slots.as_ptr() as usize
    }

    /// Number of reference-constant slots. `gcreftracer.py:10`
    /// `array_length`.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Forward every slot in place. `gcreftracer.py:13-23`
    /// `gcrefs_trace`: each `array_base_addr + i*WORD` slot is handed to
    /// the GC as a root; writing back through the visitor forwards the
    /// constant if the moving GC relocated the referenced object.
    pub fn trace(&self, visitor: &mut dyn FnMut(&mut GcRef)) {
        for cell in self.slots.iter() {
            let mut r = cell.get();
            visitor(&mut r);
            cell.set(r);
        }
    }
}

/// Append a live table to the registry, sweeping out tables whose loop
/// tokens have already been freed.
fn register_table(table: &Arc<GcTable>) {
    let mut guard = LIVE_GC_TABLES.write().unwrap();
    guard.retain(|w| w.strong_count() > 0);
    guard.push(Arc::downgrade(table));
}

/// Forward the slots of every live per-loop table. Registered once via
/// [`install_gc_table_walker`]; fires at both the minor
/// (`collector.rs:668`) and major (`collector.rs:1185`) collection
/// phases.
fn walk_all_gc_tables(visitor: &mut dyn FnMut(&mut GcRef)) {
    // Snapshot the live tables under a read guard, then release the lock
    // before tracing (same snapshot-then-iterate discipline as
    // `walk_extra_roots`, `shadow_stack.rs:622`). Dead `Weak`s are
    // filtered by `upgrade`.
    let live: Vec<Arc<GcTable>> = {
        let guard = LIVE_GC_TABLES.read().unwrap();
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
/// (`shadow_stack.rs:602`). Call once at backend init.
pub fn install_gc_table_walker() {
    crate::shadow_stack::register_extra_root_walker(gc_table_extra_root_walker);
}

#[cfg(test)]
mod tests {
    use super::*;

    // `LIVE_GC_TABLES` is a process-global registry; in production it is
    // only mutated outside a collection (table build at compile time) and
    // only read inside a stop-the-world collection, so there is never a
    // concurrent build-vs-walk. The test harness runs tests in parallel,
    // which would let one table-building test's registry mutation race
    // another's global `walk_all_gc_tables` assertion. Serialize the
    // table-touching tests against each other to model the STW invariant.
    static TEST_REGISTRY_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn trace_forwards_slots_in_place() {
        let _serialize = TEST_REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let table = GcTable::from_gcrefs(&[GcRef(0x1000), GcRef(0x2000)]);
        // A moving collection relocates 0x1000 -> 0x9000.
        table.trace(&mut |r| {
            if r.0 == 0x1000 {
                r.0 = 0x9000;
            }
        });
        assert_eq!(table.slots[0].get(), GcRef(0x9000));
        assert_eq!(table.slots[1].get(), GcRef(0x2000));
        assert_eq!(table.base_addr(), table.slots.as_ptr() as usize);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn dropping_table_deregisters_from_walk() {
        let _serialize = TEST_REGISTRY_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // A sentinel unlikely to collide with any other test's table.
        const SENTINEL: GcRef = GcRef(0x0DEAD_BEEF);
        let count_sentinels = || {
            let mut n = 0usize;
            walk_all_gc_tables(&mut |r| {
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
