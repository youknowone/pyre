//! `DICTENTRYARRAY = lltype.GcArray(DICTENTRY)` (`rordereddict.py get_ll_dict`).
//!
//! The array is allocated with `_ll_malloc_entries` (`malloc(ENTRIES, n, zero=True)`).
//! A zero item is an invalid entry: `f_valid` is false and the key and value are
//! the all-zero [`EntryDummy`].

use std::alloc::Layout;
use std::sync::atomic::{AtomicU32, Ordering};

/// One `d.entries` element — `odictentry` (`get_ll_dict` `entryfields`):
/// `key`, `f_valid`, `value`, `f_hash`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Entry<K, V> {
    pub key: K,
    pub f_valid: bool,
    pub value: V,
    pub f_hash: u64,
}

/// The value a deleted entry's key or value is reset to
/// (`must_clear_key` / `must_clear_value` store `nullptr`).
pub trait EntryDummy {
    fn dummy() -> Self;
}

impl EntryDummy for u64 {
    fn dummy() -> Self {
        0
    }
}

impl EntryDummy for i64 {
    fn dummy() -> Self {
        0
    }
}

impl EntryDummy for () {
    fn dummy() -> Self {}
}

/// `DICTENTRYARRAY`. `length` is `len(d.entries)`: the allocated item count.
/// The collector reads it as the varsize length and visits every item below it.
#[repr(C)]
pub struct GcEntries<K, V> {
    pub length: usize,
    /// Varsize tail. The allocation extends `length` items past this field.
    pub items: [Entry<K, V>; 0],
}

/// Byte offsets of the GC references inside one value of `Self`
/// (`gctypelayout` `varofstoptrs` contribution).
pub trait GcRefOffsets {
    const GC_REF_OFFSETS: &'static [usize];
}

impl GcRefOffsets for crate::pyobject::PyObjectRef {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for crate::dictmultiobject::ObjectKey {
    const GC_REF_OFFSETS: &'static [usize] =
        &[std::mem::offset_of!(crate::dictmultiobject::ObjectKey, obj)];
}

impl GcRefOffsets for crate::identitydict::IdentityKey {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for crate::dictmultiobject::BytesKey {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for crate::celldict::StrKey {
    const GC_REF_OFFSETS: &'static [usize] = &[0];
}

impl GcRefOffsets for i64 {
    const GC_REF_OFFSETS: &'static [usize] = &[];
}

impl GcRefOffsets for u64 {
    const GC_REF_OFFSETS: &'static [usize] = &[];
}

impl GcRefOffsets for () {
    const GC_REF_OFFSETS: &'static [usize] = &[];
}

/// Per-`(K, V)` GC type id of [`GcEntries`].
pub trait GcEntriesType {
    fn entries_gc_type_id() -> u32;
}

macro_rules! entries_gc_type_id {
    ($atomic:ident, $setter:ident, $getter:ident, $k:ty, $v:ty) => {
        static $atomic: AtomicU32 = AtomicU32::new(0);

        pub fn $setter(id: u32) {
            $atomic.store(id, Ordering::Relaxed);
        }

        #[majit_macros::dont_look_inside]
        pub fn $getter() -> u32 {
            $atomic.load(Ordering::Relaxed)
        }

        impl GcEntriesType for ($k, $v) {
            fn entries_gc_type_id() -> u32 {
                $getter()
            }
        }
    };
}

entries_gc_type_id!(
    OBJECT_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_object_key_pyobject_entries_gc_type_id,
    object_key_pyobject_entries_gc_type_id,
    crate::dictmultiobject::ObjectKey,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    I64_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_i64_pyobject_entries_gc_type_id,
    i64_pyobject_entries_gc_type_id,
    i64,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    BYTES_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_bytes_key_pyobject_entries_gc_type_id,
    bytes_key_pyobject_entries_gc_type_id,
    crate::dictmultiobject::BytesKey,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    OBJECT_KEY_UNIT_ENTRIES_GC_TYPE_ID,
    set_object_key_unit_entries_gc_type_id,
    object_key_unit_entries_gc_type_id,
    crate::dictmultiobject::ObjectKey,
    ()
);
entries_gc_type_id!(
    IDENTITY_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_identity_key_pyobject_entries_gc_type_id,
    identity_key_pyobject_entries_gc_type_id,
    crate::identitydict::IdentityKey,
    crate::pyobject::PyObjectRef
);
entries_gc_type_id!(
    STR_KEY_PYOBJECT_ENTRIES_GC_TYPE_ID,
    set_str_key_pyobject_entries_gc_type_id,
    str_key_pyobject_entries_gc_type_id,
    crate::celldict::StrKey,
    crate::pyobject::PyObjectRef
);

/// GC-reference offsets inside one [`Entry`], relative to the item start.
pub fn entry_gc_ref_offsets<K: GcRefOffsets, V: GcRefOffsets>() -> Vec<usize> {
    let mut offsets = Vec::new();
    let key_base = std::mem::offset_of!(Entry<K, V>, key);
    for off in K::GC_REF_OFFSETS {
        offsets.push(key_base + off);
    }
    let value_base = std::mem::offset_of!(Entry<K, V>, value);
    for off in V::GC_REF_OFFSETS {
        offsets.push(value_base + off);
    }
    offsets
}

/// First item of a `DICTENTRYARRAY`. Null stays null.
pub fn entries_item_ptr<K, V>(entries: *mut GcEntries<K, V>) -> *mut Entry<K, V> {
    if entries.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe { std::ptr::addr_of_mut!((*entries).items).cast() }
    }
}

/// `len(d.entries)`. A null array is length 0.
pub fn entries_allocated_len<K, V>(entries: *const GcEntries<K, V>) -> usize {
    if entries.is_null() {
        0
    } else {
        unsafe { (*entries).length }
    }
}

fn entries_payload_bytes<K, V>(n: usize) -> usize {
    std::mem::offset_of!(GcEntries<K, V>, items)
        .checked_add(
            n.checked_mul(std::mem::size_of::<Entry<K, V>>())
                .expect("entries allocation size"),
        )
        .expect("entries allocation size")
}

/// `_ll_malloc_entries`: `malloc(ENTRIES, n, zero=True)`.
///
/// `n == 0` is the empty array, represented by null (an empty `RDict` holds
/// no array until the first insert).
pub fn alloc_entries<K, V>(n: usize) -> *mut GcEntries<K, V>
where
    (K, V): GcEntriesType,
{
    if n == 0 {
        return std::ptr::null_mut();
    }
    let bytes = entries_payload_bytes::<K, V>(n);
    let tid = <(K, V)>::entries_gc_type_id();
    if tid != 0 {
        let raw = crate::gc_hook::try_gc_alloc_young_nonmoving_no_collect_raw(tid, bytes);
        if !raw.is_null() {
            // pyre adaptation of `_ll_malloc_entries`: Rust hands out `&K` / `&V`
            // into the array, so the block is allocated on the non-moving tier,
            // young (`external_malloc(..., alloc_young=True)`) so that a dropped
            // dict gives its array back on the next minor as the nursery
            // `DICTENTRYARRAY` does.  The no-collect entry zero-fills and never
            // collects; the barrier is for a refused young birth, which lands
            // in the old generation and may be filled with young items before
            // the next minor.
            let entries = raw as *mut GcEntries<K, V>;
            unsafe {
                (*entries).length = n;
            }
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            return entries;
        }
    }
    // `tid == 0` or no hook (unit tests, `pyre-wasm-test`): immortal
    // `alloc_zeroed`, the same contract as `malloc_typed`. Never freed.
    let layout = Layout::from_size_align(bytes, std::mem::align_of::<GcEntries<K, V>>())
        .expect("entries layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let entries = raw as *mut GcEntries<K, V>;
    unsafe {
        (*entries).length = n;
    }
    entries
}

#[cfg(test)]
impl GcEntriesType for (u64, u64) {
    fn entries_gc_type_id() -> u32 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_dummy_is_zero<T: EntryDummy + Copy>() {
        let value = T::dummy();
        let bytes = unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
        };
        assert!(
            bytes.iter().all(|byte| *byte == 0),
            "{} dummy is not all-zero",
            std::any::type_name::<T>()
        );
    }

    #[test]
    fn entry_dummies_are_all_zero_bits() {
        assert_dummy_is_zero::<u64>();
        assert_dummy_is_zero::<i64>();
        assert_dummy_is_zero::<()>();
        assert_dummy_is_zero::<crate::pyobject::PyObjectRef>();
        assert_dummy_is_zero::<crate::dictmultiobject::ObjectKey>();
        assert_dummy_is_zero::<crate::dictmultiobject::BytesKey>();
        assert_dummy_is_zero::<crate::identitydict::IdentityKey>();
        assert_dummy_is_zero::<crate::setobject::IdentitySetKey>();
        assert_dummy_is_zero::<crate::celldict::StrKey>();
    }

    #[test]
    fn fresh_entries_are_zero_filled() {
        let entries = alloc_entries::<u64, u64>(4);
        assert!(!entries.is_null());
        unsafe {
            assert_eq!((*entries).length, 4);
            let item = entries_item_ptr(entries);
            for i in 0..4 {
                let entry = &*item.add(i);
                assert!(!entry.f_valid);
                assert_eq!(entry.key, 0);
                assert_eq!(entry.value, 0);
                assert_eq!(entry.f_hash, 0);
            }
        }
    }

    #[test]
    fn object_key_entry_offsets_match_the_field_layout() {
        type Key = crate::dictmultiobject::ObjectKey;
        type Value = crate::pyobject::PyObjectRef;
        let key_base = std::mem::offset_of!(Entry<Key, Value>, key);
        let value_base = std::mem::offset_of!(Entry<Key, Value>, value);
        assert_eq!(
            entry_gc_ref_offsets::<Key, Value>(),
            vec![key_base + std::mem::offset_of!(Key, obj), value_base + 0,]
        );

        let unit_key = std::mem::offset_of!(Entry<Key, ()>, key);
        assert_eq!(
            entry_gc_ref_offsets::<Key, ()>(),
            vec![unit_key + std::mem::offset_of!(Key, obj)]
        );
        assert!(<()>::GC_REF_OFFSETS.is_empty());
    }
}
