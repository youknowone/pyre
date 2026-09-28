//! Process-lifetime owner of `ConstPtr.value`.
//!
//! `history.py` `ConstPtr` is one GC object. Holders store the box; the
//! collector writes `value`. `OpRef` is a `Copy` word, so the box's
//! address cannot live in the word: a minor would stale every copy.
//! The word is an index into this table. The table is the single
//! mutable `value` slot, registered as an extra root
//! (`gcreftracer.GcTable` is the same shape for compiled-code constants).
//!
//! Intern key is `gc_id_or_identityhash` (`minimark.py`
//! `id_or_identityhash`), not the address. The key is recorded at
//! intern time and is not recomputed during a collection.

use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Mutex;

use crate::value::{GcRef, gc_id_or_identityhash};

struct Table {
    /// Index 0 is null and is never interned.
    slots: Vec<GcRef>,
    /// `(identity hash, index)`, sorted by hash then index.
    by_hash: Vec<(u64, u32)>,
}

impl Table {
    fn new() -> Self {
        Self {
            slots: vec![GcRef::NULL],
            by_hash: Vec::new(),
        }
    }
}

static TABLE: Mutex<Table> = Mutex::new(Table {
    slots: Vec::new(),
    by_hash: Vec::new(),
});
static READY: AtomicBool = AtomicBool::new(false);
static WALKING: AtomicBool = AtomicBool::new(false);

fn table() -> parking_lot::MutexGuard<'static, Table> {
    let mut guard = TABLE.lock();
    if !READY.load(Ordering::Acquire) {
        if guard.slots.is_empty() {
            *guard = Table::new();
        }
        READY.store(true, Ordering::Release);
    }
    guard
}

/// Index of `addr` in the table. Null is 0. Same referent, same index.
pub fn intern(addr: GcRef) -> u32 {
    if addr.is_null() {
        return 0;
    }
    let hash = gc_id_or_identityhash(addr.0) as u64;
    let mut guard = table();
    if let Some(idx) = find(&guard, hash, addr.0) {
        return idx;
    }
    let idx = guard.slots.len() as u32;
    guard.slots.push(addr);
    let pos = guard
        .by_hash
        .partition_point(|&(h, i)| (h, i) < (hash, idx));
    guard.by_hash.insert(pos, (hash, idx));
    idx
}

/// Write the forwarded address of an existing index.
///
/// `MIFrame.set_forwarded_ref_value` publishes one move into the box.
/// The identity hash recorded at intern time does not change.
pub fn set_slot(index: u32, addr: GcRef) {
    if index == 0 {
        return;
    }
    let mut guard = table();
    if let Some(slot) = guard.slots.get_mut(index as usize) {
        *slot = addr;
    }
}

/// Current address of index `index`.
pub fn resolve(index: u32) -> GcRef {
    if index == 0 {
        return GcRef::NULL;
    }
    let guard = table();
    guard
        .slots
        .get(index as usize)
        .copied()
        .unwrap_or(GcRef::NULL)
}

/// Forward every non-null slot. Nested walk (a visitor that collects)
/// is a no-op: the outer walk is already updating the slots.
pub fn walk(visitor: &mut dyn FnMut(&mut GcRef)) {
    if WALKING.swap(true, Ordering::AcqRel) {
        return;
    }
    struct Clear;
    impl Drop for Clear {
        fn drop(&mut self) {
            WALKING.store(false, Ordering::Release);
        }
    }
    let _clear = Clear;
    let mut guard = table();
    for slot in guard.slots.iter_mut().skip(1) {
        if !slot.is_null() {
            visitor(slot);
        }
    }
}

fn find(table: &Table, hash: u64, addr: usize) -> Option<u32> {
    let start = table.by_hash.partition_point(|&(h, _)| h < hash);
    for &(h, idx) in &table.by_hash[start..] {
        if h != hash {
            break;
        }
        if table.slots.get(idx as usize).map(|s| s.0) == Some(addr) {
            return Some(idx);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intern_is_stable_and_null_is_zero() {
        assert_eq!(intern(GcRef::NULL), 0);
        assert!(resolve(0).is_null());
        let a = GcRef(0x1111_0000);
        let i = intern(a);
        assert_eq!(intern(a), i);
        assert_eq!(resolve(i), a);
        let b = GcRef(0x2222_0000);
        assert_ne!(intern(b), i);
    }

    #[test]
    fn walk_forwards_the_slot_the_index_names() {
        let addr = GcRef(0x3333_0000);
        let idx = intern(addr);
        walk(&mut |slot| {
            if slot.0 == addr.0 {
                *slot = GcRef(0x4444_0000);
            }
        });
        assert_eq!(resolve(idx), GcRef(0x4444_0000));
    }
}
