//! RPython `rpython/rtyper/lltypesystem/llheap.py` parity module.
//!
//! Upstream `llheap.py` is the LLInterpreter-facing heap facade over
//! `lltype`, `llmemory`, and a handful of `rgc` helpers.  This module
//! exposes the subset whose underlying Rust implementation already exists.

pub use crate::translator::rtyper::lltypesystem::llmemory::{weakref_create, weakref_deref};
pub use crate::translator::rtyper::lltypesystem::lltype::{MallocFlavor, malloc, typeOf};

use crate::translator::rtyper::lltypesystem::lltype::{_ptr, _ptr_obj, GcKind};

/// RPython `llheap.free = lltype.free` (`llheap.py:3`).
///
/// Mirrors the raw-container checks from `lltype.py:2246-2254`.  The
/// `track_allocation` flag is accepted for signature parity; pyre has no
/// leakfinder side-channel here, so freeing is just the container `_free`.
pub fn free(p: &_ptr, flavor: MallocFlavor, _track_allocation: bool) -> Result<(), String> {
    if flavor == MallocFlavor::Gc {
        return Err("gc flavor free".to_string());
    }
    if p._togckind() != GcKind::Raw {
        return Err("free(): only for pointers to non-gc containers".to_string());
    }
    match p
        ._obj()
        .map_err(|_| "free(): delayed pointer has no concrete object".to_string())?
    {
        _ptr_obj::Struct(obj) => obj._free(),
        _ptr_obj::Array(obj) => obj._free(),
        _ptr_obj::Opaque(obj) => obj._free(),
        _ => return Err("free(): only for pointers to non-gc containers".to_string()),
    }
    Ok(())
}

/// RPython `weakref_create_getlazy(objgetter)` (`llheap.py:15-16`).
pub fn weakref_create_getlazy<F>(objgetter: F) -> Result<_ptr, String>
where
    F: FnOnce() -> _ptr,
{
    let obj = objgetter();
    weakref_create(&obj)
}

/// RPython `shrink_array(p, smallersize)` (`llheap.py:19-20`).
pub fn shrink_array(_p: &_ptr, _smallersize: usize) -> bool {
    false
}

/// RPython `thread_run()` (`llheap.py:23-24`).
pub fn thread_run() {}

/// RPython `thread_start()` (`llheap.py:26-27`).
pub fn thread_start() {}

/// RPython `thread_die()` (`llheap.py:29-30`).
pub fn thread_die() {}

/// RPython `pin(obj)` (`llheap.py:32-33`).
pub fn pin<T>(_obj: &T) -> bool {
    false
}

/// RPython `unpin(obj)` (`llheap.py:35-37`).
pub fn unpin<T>(_obj: &T) {
    panic!("pin() always returns False, so unpin() should not be called");
}

/// RPython `_is_pinned(obj)` (`llheap.py:39-40`).
pub fn _is_pinned<T>(_obj: &T) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, StructType};

    #[test]
    fn llheap_exposes_lltype_and_weakref_heap_facade() {
        let raw_t = LowLevelType::Struct(Box::new(StructType::new("RAW", vec![])));
        let raw_p = malloc(raw_t, None, MallocFlavor::Raw, false).unwrap();
        assert_eq!(typeOf(&raw_p)._gckind(), GcKind::Raw);
        assert!(!raw_p._was_freed().unwrap());

        free(&raw_p, MallocFlavor::Raw, true).unwrap();
        assert!(raw_p._was_freed().unwrap());

        let gc_t = LowLevelType::Struct(Box::new(StructType::gc("GC", vec![])));
        let gc_p = malloc(gc_t, None, MallocFlavor::Gc, false).unwrap();
        let wref = weakref_create_getlazy(|| gc_p.clone()).unwrap();
        let deref = weakref_deref(&LowLevelType::Ptr(Box::new(typeOf(&gc_p))), &wref).unwrap();
        assert!(deref.nonzero());
    }

    #[test]
    fn llheap_small_rgc_facade_matches_upstream_defaults() {
        let value = 42_i64;
        assert!(!shrink_array(
            &malloc(
                LowLevelType::Struct(Box::new(StructType::new("RAW2", vec![]))),
                None,
                MallocFlavor::Raw,
                false,
            )
            .unwrap(),
            1,
        ));
        thread_run();
        thread_start();
        thread_die();
        assert!(!pin(&value));
        assert!(!_is_pinned(&value));
    }
}
