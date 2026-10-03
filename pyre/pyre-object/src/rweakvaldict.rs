//! `rpython/rlib/_rweakvaldict.py` `WeakValueDictRepr`.
//!
//! `WEAKDICT` is `num_items`, `resize_counter`, `entries`. Each
//! `WEAKDICTENTRY` is `(key, WeakRefPtr)` — no `f_valid`, no `f_hash`.
//! `ll_valid` is a live `weakref_deref`; `ll_everused` is a non-null value
//! word, so a dead weakref stays until `ll_weakdict_resize`. Lookup and
//! resize are `rdict.py` `ll_dict_lookup` / `ll_dict_resize` (`DICT_INITSIZE`
//! is 8). `paranoia` is false.
//!
//! The entries array is the non-moving tier, same as `_ll_malloc_entries`
//! here: a raw pointer into the array must stay put across a later store.
//! `_ll_free_entries` is a no-op, so a replaced array is left for the
//! collector.

use std::sync::atomic::{AtomicU32, Ordering};

use crate::celldict::StrKey;
use crate::weakref::Weakref;

const PERTURB_SHIFT: u32 = 5;
const DICT_INITSIZE: usize = 8;
const HIGHEST_BIT: u64 = 1_u64 << 63;

static WEAKDICT_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);
static WEAKDICT_ENTRIES_GC_TYPE_ID: AtomicU32 = AtomicU32::new(0);

pub fn set_weakdict_gc_type_id(id: u32) {
    WEAKDICT_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

pub fn set_weakdict_entries_gc_type_id(id: u32) {
    WEAKDICT_ENTRIES_GC_TYPE_ID.store(id, Ordering::Relaxed);
}

pub fn weakdict_gc_type_id() -> u32 {
    WEAKDICT_GC_TYPE_ID.load(Ordering::Relaxed)
}

fn weakdict_entries_gc_type_id() -> u32 {
    WEAKDICT_ENTRIES_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `WEAKDICTENTRY`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeakDictEntry {
    pub key: StrKey,
    pub value: *mut Weakref,
}

/// `WEAKDICTENTRYARRAY`. `length` is `len(entries)`.
#[repr(C)]
pub struct WeakDictEntries {
    pub length: usize,
    pub items: [WeakDictEntry; 0],
}

/// `WEAKDICT` / `weakvaldict`.
#[repr(C)]
pub struct WeakDict {
    pub num_items: isize,
    pub resize_counter: isize,
    pub entries: *mut WeakDictEntries,
}

fn alloc_raw(tid: u32, bytes: usize) -> *mut u8 {
    if tid != 0 {
        let raw = crate::gc_hook::try_gc_alloc_stable_raw(tid, bytes);
        if !raw.is_null() {
            crate::gc_hook::try_gc_write_barrier_managed(raw);
            return raw;
        }
    }
    let layout = std::alloc::Layout::from_size_align(bytes, std::mem::align_of::<usize>())
        .expect("weakdict layout");
    let raw = unsafe { std::alloc::alloc_zeroed(layout) };
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    raw
}

fn entries_bytes(n: usize) -> usize {
    std::mem::offset_of!(WeakDictEntries, items)
        .checked_add(
            n.checked_mul(std::mem::size_of::<WeakDictEntry>())
                .expect("weakdict entries size"),
        )
        .expect("weakdict entries size")
}

fn alloc_entries(n: usize) -> *mut WeakDictEntries {
    let raw = alloc_raw(weakdict_entries_gc_type_id(), entries_bytes(n));
    let entries = raw as *mut WeakDictEntries;
    unsafe { (*entries).length = n };
    entries
}

fn items(entries: *mut WeakDictEntries) -> *mut WeakDictEntry {
    unsafe { std::ptr::addr_of_mut!((*entries).items).cast() }
}

fn entry(entries: *mut WeakDictEntries, i: usize) -> *mut WeakDictEntry {
    unsafe { items(entries).add(i) }
}

fn everused(entries: *mut WeakDictEntries, i: usize) -> bool {
    unsafe { !(*entry(entries, i)).value.is_null() }
}

fn valid(entries: *mut WeakDictEntries, i: usize) -> bool {
    let value = unsafe { (*entry(entries, i)).value };
    if value.is_null() {
        return false;
    }
    // `ll_valid`: `bool(value) and bool(weakref_deref(...))`.
    !unsafe { crate::weakref::w_weakref_deref(value) }.is_null()
}

/// `ll_strhash` over the key block. A missing str-hash hook (object-crate
/// tests) uses a non-zero FNV so the open-addressing probe still agrees
/// with itself. `0` in the STR hash slot means "not memoized".
fn key_hash(key: StrKey) -> u64 {
    if key.0.is_null() {
        return 0;
    }
    let cached = unsafe { (*key.0).hash };
    if cached != 0 {
        return cached as u64;
    }
    let bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key.0) };
    let hash = bytes_hash(bytes);
    if hash != 0 {
        unsafe { (*key.0).hash = hash as usize };
    }
    hash
}

fn bytes_hash(bytes: &[u8]) -> u64 {
    if let Some(hash) = crate::dict_eq_hook::try_hash_str(bytes) {
        return hash as u64;
    }
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    if h == 0 { 1 } else { h }
}

fn key_eq(key: StrKey, bytes: &[u8]) -> bool {
    if key.0.is_null() {
        return bytes.is_empty();
    }
    unsafe { crate::unicodeobject::utf8_payload_bytes(key.0) == bytes }
}

fn barrier_entries(entries: *mut WeakDictEntries) {
    if !entries.is_null() {
        crate::gc_hook::try_gc_write_barrier(entries as *mut u8);
    }
}

fn barrier_dict(d: *mut WeakDict) {
    if !d.is_null() {
        crate::gc_hook::try_gc_write_barrier(d as *mut u8);
    }
}

impl WeakDict {
    /// `ll_new_weakdict`.
    pub fn ll_new() -> *mut WeakDict {
        let raw = alloc_raw(weakdict_gc_type_id(), std::mem::size_of::<WeakDict>());
        let d = raw as *mut WeakDict;
        unsafe {
            (*d).entries = alloc_entries(DICT_INITSIZE);
            (*d).num_items = 0;
            (*d).resize_counter = (DICT_INITSIZE * 2) as isize;
        }
        barrier_dict(d);
        d
    }

    /// `ll_get`. `None` is the null object pointer.
    #[majit_macros::dont_look_inside]
    pub fn ll_get(&self, bytes: &[u8]) -> Option<crate::pyobject::PyObjectRef> {
        let hash = bytes_hash(bytes);
        let i = self.ll_dict_lookup(bytes, hash) & (HIGHEST_BIT - 1);
        let valueref = unsafe { (*entry(self.entries, i as usize)).value };
        if valueref.is_null() {
            None
        } else {
            let obj = unsafe { crate::weakref::w_weakref_deref(valueref) };
            if obj.is_null() { None } else { Some(obj) }
        }
    }

    /// `ll_set_nonnull` after `weakref_create`. The caller creates the
    /// weakref first: this table's mutex is also taken by the root walk,
    /// and `weakref_create` may collect.
    #[majit_macros::dont_look_inside]
    pub fn ll_set_nonnull(&mut self, key: StrKey, valueref: *mut Weakref) {
        debug_assert!(!valueref.is_null());
        let hash = key_hash(key);
        let bytes = unsafe { crate::unicodeobject::utf8_payload_bytes(key.0) };
        let found = self.ll_dict_lookup(bytes, hash);
        let i = (found & (HIGHEST_BIT - 1)) as usize;
        let was_everused = everused(self.entries, i);
        unsafe {
            (*entry(self.entries, i)).key = key;
            (*entry(self.entries, i)).value = valueref;
        }
        barrier_entries(self.entries);
        if !was_everused {
            self.resize_counter -= 3;
            if self.resize_counter <= 0 {
                self.ll_weakdict_resize();
            }
        }
    }

    /// The value word at `bytes`, including a dead weakref. `None` when
    /// the key was never stored.
    pub fn stored_word(&self, bytes: &[u8]) -> Option<usize> {
        let hash = bytes_hash(bytes);
        let found = self.ll_dict_lookup(bytes, hash);
        if found & HIGHEST_BIT != 0 {
            return None;
        }
        let value = unsafe { (*entry(self.entries, found as usize)).value };
        if value.is_null() { None } else { Some(value as usize) }
    }

    pub fn count_valid(&self) -> usize {
        if self.entries.is_null() {
            return 0;
        }
        let n = unsafe { (*self.entries).length };
        (0..n).filter(|&i| valid(self.entries, i)).count()
    }

    /// `ll_dict_lookup`. The high bit marks a miss; the low bits are the
    /// slot (`MASK` keeps them, and a miss slot may be a deleted one).
    fn ll_dict_lookup(&self, bytes: &[u8], hash: u64) -> u64 {
        let entries = self.entries;
        let mask = (unsafe { (*entries).length } - 1) as u64;
        let mut i = hash & mask;
        if valid(entries, i as usize) {
            let checking = unsafe { (*entry(entries, i as usize)).key };
            if key_hash(checking) == hash && key_eq(checking, bytes) {
                return i;
            }
        } else if !everused(entries, i as usize) {
            return i | HIGHEST_BIT;
        }
        let mut freeslot: i64 = if everused(entries, i as usize) && !valid(entries, i as usize) {
            i as i64
        } else {
            -1
        };
        let mut perturb = hash;
        loop {
            i = (i << 2).wrapping_add(i).wrapping_add(perturb).wrapping_add(1) & mask;
            if !everused(entries, i as usize) {
                if freeslot == -1 {
                    freeslot = i as i64;
                }
                return (freeslot as u64) | HIGHEST_BIT;
            } else if valid(entries, i as usize) {
                let checking = unsafe { (*entry(entries, i as usize)).key };
                if key_hash(checking) == hash && key_eq(checking, bytes) {
                    return i;
                }
            } else if freeslot == -1 {
                freeslot = i as i64;
            }
            perturb >>= PERTURB_SHIFT;
        }
    }

    fn ll_dict_lookup_clean(&self, hash: u64) -> usize {
        let entries = self.entries;
        let mask = (unsafe { (*entries).length } - 1) as u64;
        let mut i = hash & mask;
        let mut perturb = hash;
        while everused(entries, i as usize) {
            i = (i << 2).wrapping_add(i).wrapping_add(perturb).wrapping_add(1) & mask;
            perturb >>= PERTURB_SHIFT;
        }
        i as usize
    }

    /// `ll_weakdict_resize` then `ll_dict_resize`.
    fn ll_weakdict_resize(&mut self) {
        let entries = self.entries;
        let n = unsafe { (*entries).length };
        let mut num_items = 0isize;
        for i in 0..n {
            if valid(entries, i) {
                num_items += 1;
            }
        }
        self.num_items = num_items;
        self.ll_dict_resize();
    }

    fn ll_dict_resize(&mut self) {
        let num_extra = std::cmp::min(self.num_items + 1, 30000);
        let new_estimate = (self.num_items + num_extra) * 2;
        let mut new_size = DICT_INITSIZE;
        while (new_size as isize) <= new_estimate {
            new_size *= 2;
        }
        let old_entries = self.entries;
        let old_size = unsafe { (*old_entries).length };
        self.entries = alloc_entries(new_size);
        self.num_items = 0;
        self.resize_counter = (new_size * 2) as isize;
        barrier_dict(self as *mut WeakDict);
        for i in 0..old_size {
            if !valid(old_entries, i) {
                continue;
            }
            let item = unsafe { *entry(old_entries, i) };
            let hash = key_hash(item.key);
            self.ll_dict_insertclean(item.key, item.value, hash);
        }
    }

    fn ll_dict_insertclean(&mut self, key: StrKey, value: *mut Weakref, hash: u64) {
        let i = self.ll_dict_lookup_clean(hash);
        unsafe {
            (*entry(self.entries, i)).key = key;
            (*entry(self.entries, i)).value = value;
        }
        self.num_items += 1;
        self.resize_counter -= 3;
        barrier_entries(self.entries);
    }
}

/// Copy live entries into `dst`. Used when an immortal pre-GC table is
/// replaced by a collector-owned `WEAKDICT`.
pub fn copy_valid(src: *mut WeakDict, dst: *mut WeakDict) {
    if src.is_null() || dst.is_null() {
        return;
    }
    unsafe {
        let entries = (*src).entries;
        if entries.is_null() {
            return;
        }
        let n = (*entries).length;
        for i in 0..n {
            if !valid(entries, i) {
                continue;
            }
            let item = *entry(entries, i);
            (*dst).ll_set_nonnull(item.key, item.value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn miss_returns_the_stored_referent_and_a_second_key_does_not_clobber_it() {
        let d = unsafe { &mut *WeakDict::ll_new() };
        let key = StrKey(crate::unicodeobject::alloc_utf8_payload(b"abc", false));
        let obj = 0x1111 as crate::pyobject::PyObjectRef;
        let wref = unsafe { crate::weakref::w_weakref_new(obj) };
        d.ll_set_nonnull(key, wref);
        assert_eq!(d.ll_get(b"abc"), Some(obj));
        assert!(d.ll_get(b"abd").is_none());
        assert_eq!(d.stored_word(b"abc"), Some(wref as usize));
        assert_eq!(d.count_valid(), 1);
    }

    /// Six new slots drive `resize_counter` from `DICT_INITSIZE * 2` through
    /// zero (`ll_set_nonnull` subtracts 3). Every key must still be found.
    #[test]
    fn resize_keeps_live_weak_values() {
        let d = unsafe { &mut *WeakDict::ll_new() };
        let mut objs = Vec::new();
        for n in 0..6 {
            let bytes = format!("k{n}");
            let key = StrKey(crate::unicodeobject::alloc_utf8_payload(bytes.as_bytes(), false));
            let obj = (0x2000 + n) as crate::pyobject::PyObjectRef;
            let wref = unsafe { crate::weakref::w_weakref_new(obj) };
            d.ll_set_nonnull(key, wref);
            objs.push((bytes, obj));
        }
        assert!(unsafe { (*d.entries).length } > DICT_INITSIZE);
        for (bytes, obj) in objs {
            assert_eq!(d.ll_get(bytes.as_bytes()), Some(obj));
        }
    }
}
