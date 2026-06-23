//! RPython `rpython/rtyper/lltypesystem/llarena.py` parity module.
//!
//! Upstream `llarena.py` has two halves: a fake arena model used by
//! llinterp and translation support registered through `rffi` /
//! `extfunc`. This slice lands the public fake-arena names and the
//! standalone alignment helper. Object reservation and raw-memory
//! registration remain pending with the full `rffi` / `extfunc`
//! integration.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::llmemory::AddressOffset;

static COUNT_ARENAS: AtomicUsize = AtomicUsize::new(0);

/// RPython `class ArenaError(Exception)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArenaError {
    message: String,
}

impl ArenaError {
    pub fn new(message: impl Into<String>) -> Self {
        ArenaError {
            message: message.into(),
        }
    }
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ArenaError {}

/// RPython `class Arena(object)`.
#[derive(Debug)]
pub struct Arena {
    pub arena_index: usize,
    pub nbytes: i64,
    pub freed: bool,
    pub protect_inaccessible: Option<bool>,
    zero: i64,
}

impl Arena {
    pub fn new(nbytes: i64, zero: i64) -> Result<Self, ArenaError> {
        if nbytes < 0 {
            return Err(ArenaError::new("arena size must be non-negative"));
        }
        Ok(Arena {
            arena_index: COUNT_ARENAS.fetch_add(1, Ordering::Relaxed) + 1,
            nbytes,
            freed: false,
            protect_inaccessible: None,
            zero,
        })
    }

    pub fn check(&self) -> Result<(), ArenaError> {
        if self.freed {
            return Err(ArenaError::new("arena was already freed"));
        }
        if self.protect_inaccessible.is_some() {
            return Err(ArenaError::new("arena is currently arena_protect()ed"));
        }
        Ok(())
    }

    pub fn reset(&mut self, zero: i64, start: i64, size: Option<i64>) -> Result<(), ArenaError> {
        self.check()?;
        let stop = size.map_or(self.nbytes, |size| start + size);
        if !(0 <= start && start <= stop && stop <= self.nbytes) {
            return Err(ArenaError::new("arena reset range is outside the arena"));
        }
        self.zero = zero;
        Ok(())
    }

    pub fn mark_freed(&mut self) {
        self.freed = true;
    }
}

impl fmt::Display for Arena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "<Arena #{} [{} bytes]>", self.arena_index, self.nbytes)
    }
}

/// RPython `class fakearenaaddress(llmemory.fakeaddress)`.
#[derive(Clone, Debug)]
pub struct FakeArenaAddress {
    pub arena: Arc<Mutex<Arena>>,
    pub offset: i64,
}

impl FakeArenaAddress {
    pub fn new(arena: Arc<Mutex<Arena>>, offset: i64) -> Result<Self, ArenaError> {
        let nbytes = arena.lock().unwrap().nbytes;
        if !(0 <= offset && offset <= nbytes) {
            return Err(ArenaError::new("Address offset is outside the arena"));
        }
        Ok(FakeArenaAddress { arena, offset })
    }
}

impl fmt::Display for FakeArenaAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arena = self.arena.lock().unwrap();
        write!(f, "<arenaaddr {} + {}>", *arena, self.offset)
    }
}

/// Upstream spelling preserved for code that mirrors
/// `isinstance(addr, fakearenaaddress)`.
#[allow(non_camel_case_types)]
pub type fakearenaaddress = FakeArenaAddress;

/// RPython `class RoundedUpForAllocation(llmemory.AddressOffset)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RoundedUpForAllocation {
    pub basesize: AddressOffset,
    pub minsize: Option<AddressOffset>,
}

impl RoundedUpForAllocation {
    pub fn known_nonneg(&self) -> bool {
        self.basesize.known_nonneg()
    }
}

/// RPython `arena_malloc(nbytes, zero)`.
pub fn arena_malloc(nbytes: i64, zero: i64) -> Result<FakeArenaAddress, ArenaError> {
    let arena = Arc::new(Mutex::new(Arena::new(nbytes, zero)?));
    FakeArenaAddress::new(arena, 0)
}

/// RPython `arena_free(arena_addr)`.
pub fn arena_free(arena_addr: &FakeArenaAddress) -> Result<(), ArenaError> {
    if arena_addr.offset != 0 {
        return Err(ArenaError::new("arena_free expects the arena base address"));
    }
    let mut arena = arena_addr.arena.lock().unwrap();
    arena.reset(0, 0, None)?;
    arena.mark_freed();
    Ok(())
}

/// RPython `arena_reset(arena_addr, size, zero)`.
pub fn arena_reset(arena_addr: &FakeArenaAddress, size: i64, zero: i64) -> Result<(), ArenaError> {
    arena_addr
        .arena
        .lock()
        .unwrap()
        .reset(zero, arena_addr.offset, Some(size))
}

/// RPython `getfakearenaaddress(addr)`.
pub fn getfakearenaaddress(addr: &FakeArenaAddress) -> FakeArenaAddress {
    addr.clone()
}

fn llarena_runtime_deferred(name: &str) -> TyperError {
    TyperError::missing_rtype_operation(format!(
        "llarena.{name} requires raw arena/runtime storage integration"
    ))
}

pub fn _oldobj_to_address() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("_oldobj_to_address"))
}

pub fn arena_reserve() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("arena_reserve"))
}

pub fn arena_shrink_obj() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("arena_shrink_obj"))
}

/// RPython `round_up_for_allocation(size, minsize=0)`.
pub fn round_up_for_allocation(
    basesize: AddressOffset,
    minsize: Option<AddressOffset>,
) -> RoundedUpForAllocation {
    RoundedUpForAllocation { basesize, minsize }
}

/// RPython `arena_new_view(ptr)`.
pub fn arena_new_view(ptr: &FakeArenaAddress) -> Result<FakeArenaAddress, ArenaError> {
    let nbytes = ptr.arena.lock().unwrap().nbytes;
    arena_malloc(nbytes, 0)
}

pub fn madvise_arena_free() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("madvise_arena_free"))
}

pub fn llimpl_malloc() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_malloc"))
}

pub fn llimpl_calloc() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_calloc"))
}

pub fn llimpl_free() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_free"))
}

pub fn llimpl_arena_malloc() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_arena_malloc"))
}

pub fn llimpl_arena_reset() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_arena_reset"))
}

pub fn llimpl_arena_reserve() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_arena_reserve"))
}

pub fn llimpl_arena_shrink_obj() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_arena_shrink_obj"))
}

pub fn llimpl_arena_new_view(addr: &FakeArenaAddress) -> FakeArenaAddress {
    addr.clone()
}

pub fn llimpl_arena_protect() -> Result<(), TyperError> {
    Err(llarena_runtime_deferred("llimpl_arena_protect"))
}

pub fn llimpl_getfakearenaaddress(addr: &FakeArenaAddress) -> FakeArenaAddress {
    getfakearenaaddress(addr)
}

/// RPython `llimpl_round_up_for_allocation(size, minsize)`.
///
/// Upstream uses module global `MEMORY_ALIGNMENT`; the Rust port takes
/// it explicitly so tests and future platform wiring can provide the
/// value computed by `rffi_platform.memory_alignment`.
pub fn llimpl_round_up_for_allocation(size: i64, minsize: i64, memory_alignment: i64) -> i64 {
    assert!(memory_alignment > 0);
    let base = size.max(minsize);
    (base + (memory_alignment - 1)) & !(memory_alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::{
        arena_free, arena_malloc, arena_new_view, arena_reserve, arena_reset, getfakearenaaddress,
        llimpl_arena_new_view, llimpl_getfakearenaaddress, llimpl_malloc,
        llimpl_round_up_for_allocation,
    };

    #[test]
    fn arena_malloc_returns_base_fakearenaaddress() {
        let addr = arena_malloc(64, 1).expect("arena_malloc");
        assert_eq!(addr.offset, 0);
        assert_eq!(addr.arena.lock().unwrap().nbytes, 64);
    }

    #[test]
    fn arena_free_marks_arena_as_freed() {
        let addr = arena_malloc(8, 0).expect("arena_malloc");
        arena_free(&addr).expect("arena_free");
        assert!(addr.arena.lock().unwrap().freed);
    }

    #[test]
    fn arena_reset_rejects_out_of_range_subrange() {
        let addr = arena_malloc(8, 0).expect("arena_malloc");
        assert!(arena_reset(&addr, 9, 0).is_err());
    }

    #[test]
    fn arena_address_identity_helpers_match_fakearena_fast_path() {
        let addr = arena_malloc(16, 0).expect("arena_malloc");
        assert_eq!(getfakearenaaddress(&addr).offset, addr.offset);
        assert_eq!(llimpl_getfakearenaaddress(&addr).offset, addr.offset);
        assert_eq!(llimpl_arena_new_view(&addr).offset, addr.offset);
    }

    #[test]
    fn arena_new_view_returns_fresh_arena_with_same_size() {
        let addr = arena_malloc(16, 0).expect("arena_malloc");
        let view = arena_new_view(&addr).expect("arena_new_view");
        assert_eq!(view.offset, 0);
        assert_eq!(view.arena.lock().unwrap().nbytes, 16);
        assert_ne!(
            view.arena.lock().unwrap().arena_index,
            addr.arena.lock().unwrap().arena_index
        );
    }

    #[test]
    fn raw_runtime_helpers_are_explicitly_deferred() {
        let err = arena_reserve().expect_err("arena reserve is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("arena_reserve"));

        let err = llimpl_malloc().expect_err("raw malloc is deferred");
        assert!(err.is_missing_rtype_operation());
        assert!(err.to_string().contains("llimpl_malloc"));
    }

    #[test]
    fn llimpl_round_up_for_allocation_matches_rpython_formula() {
        assert_eq!(llimpl_round_up_for_allocation(1, 0, 8), 8);
        assert_eq!(llimpl_round_up_for_allocation(8, 0, 8), 8);
        assert_eq!(llimpl_round_up_for_allocation(9, 0, 8), 16);
        assert_eq!(llimpl_round_up_for_allocation(1, 17, 8), 24);
    }
}
