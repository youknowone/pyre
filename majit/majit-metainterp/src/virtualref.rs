//! Virtual references: lazy materialization of virtualized objects.
//!
//! When JIT-compiled code virtualizes an object (e.g., an interpreter frame),
//! external code may still hold a reference to it. A virtual reference is a
//! lightweight wrapper that defers materialization until someone actually
//! accesses (forces) it.
//!
//! The `JitVirtualRef` struct has two fields:
//! - `virtual_token`: while the JIT is running, this is the force_token
//!   (i.e., the JIT frame address). Set to TOKEN_NONE when forced or
//!   when the frame is no longer active.
//! - `forced`: once forced, this points to the materialized object.
//!
//! Mirrors `rpython/jit/metainterp/virtualref.py`.

use std::sync::atomic::{AtomicU32, Ordering};

/// GC type id for JitVirtualRef, set by `set_vref_gc_type_id()` at startup.
/// RPython registers JIT_VIRTUAL_REF as a real GC type; pyre does the same
/// via `gc.register_type(TypeInfo::with_gc_ptrs(...))` in eval.rs.
static VREF_GC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);

/// Set the GC type id for JitVirtualRef. Called once at startup after
/// `gc.register_type()` returns the assigned id.
pub fn set_vref_gc_type_id(type_id: u32) {
    VREF_GC_TYPE_ID.store(type_id, Ordering::Relaxed);
}

/// Get the registered GC type id for JitVirtualRef.
pub fn vref_gc_type_id() -> u32 {
    VREF_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `rpython/rtyper/rclass.py:OBJECT` — RPython's GC
/// object header.  Every `rclass.OBJECT` subclass starts with a
/// `typeptr` field (a vtable pointer) used for runtime type identity
/// checks (`inst.typeptr == some_vtable`).  Pyre's analogue: a u64
/// `typeptr` slot at offset 0 carrying the type-id constant for
/// each registered GC type.
#[repr(C)]
pub struct ObjectHeader {
    /// `rclass.OBJECT.typeptr` — runtime type identity.  RPython
    /// stores a pointer to the per-class `OBJECT_VTABLE` instance;
    /// pyre stores the type-id constant directly (e.g.
    /// `JIT_VIRTUAL_REF_VTABLE` for `JitVirtualRef`).
    pub typeptr: u64,
}

/// `rpython/rlib/jit.py JitVirtualRef`: heap-allocated virtual
/// reference.  Direct port of `virtualref.py:17-20`:
/// ```python
/// self.JIT_VIRTUAL_REF = lltype.GcStruct('JitVirtualRef',
///     ('super', rclass.OBJECT),
///     ('virtual_token', llmemory.GCREF),
///     ('forced', rclass.OBJECTPTR))
/// ```
/// The `super: ObjectHeader` field is the Rust analogue of upstream's
/// `('super', rclass.OBJECT)` inheritance link — every JitVirtualRef
/// starts with the typeptr slot at offset 0, matching the runtime
/// layout `is_virtual_ref` reads through.
///
/// `virtual_token` and `forced` are both `llmemory.GCREF`/`OBJECTPTR`
/// upstream — pointer fields, not int.  Pyre stores them as `*mut
/// u8` so the runtime layout and the optimizer-side
/// `make_vref_field_descr` (`Type::Ref` per
/// `optimizeopt/virtualize.rs`) agree on the slot type.
///
/// TODO (GC trace).  Upstream traces both fields as
/// real GC pointers; pyre traces only `forced`.  `eval.rs:241-247`
/// registers JIT_VIRTUAL_REF with `gc_ptr_offsets = [16]` (forced
/// only).  The reason is that every value `virtual_token` ever holds
/// at runtime falls outside the GC heap:
///   - `TOKEN_NONE` — null, safe to walk.
///   - `token_tracing_rescall()` — program-lifetime leaked
///     `Box<ObjectHeader>` (see `allocate_tracing_rescall_dummy` /
///     `TRACING_RESCALL_DUMMY_PTR` below), host-heap allocated and
///     never freed; not a GC-allocated `_dummy` GcStruct.
///   - an active JITFRAME address — `libc::calloc`'d on a host-side
///     pool (eval.rs:232-240), not nursery/oldgen.
/// Routing it through `trace_and_update_object` would either be a
/// no-op or trip a poison-address check.  The optimizer-side
/// `Type::Ref` is intentionally retained so that
/// `setfield_gc_r` / `getfield_gc_r` ops emit correctly during
/// tracing; only the collector's view of the slot diverges.
/// Convergence path: would require allocating `_dummy` via the GC
/// AND routing JITFRAMEs through GC-managed allocation, both outside
/// the current parity scope.
#[repr(C)]
pub struct JitVirtualRef {
    /// `('super', rclass.OBJECT)` — typeptr slot at offset 0.
    pub super_: ObjectHeader,
    pub virtual_token: *mut u8,
    pub forced: *mut u8,
}

/// `virtualref.py:21-23` `jit_virtual_ref_vtable = lltype.malloc(
/// rclass.OBJECT_VTABLE, ..., immortal=True)` — the per-class vtable
/// instance that `is_virtual_ref` compares `inst.typeptr` against.
/// Pyre stores the type-id as a u64 magic constant rather than a
/// real OBJECT_VTABLE pointer; the comparison `header.typeptr ==
/// JIT_VIRTUAL_REF_VTABLE` is the structural equivalent of
/// upstream's `inst.typeptr == self.jit_virtual_ref_vtable`.
pub const JIT_VIRTUAL_REF_VTABLE: u64 = 0x4A49_5456_5245_4621; // "JITVREF!"

/// `rpython/rlib/jit.py:487 class InvalidVirtualRef(Exception)` —
/// `force_virtual` raises this when `virtual_token == TOKEN_NONE`
/// but `forced` is null (`virtualref.py:174-176`).  Pyre's single
/// canonical definition lives in `crate::jit::InvalidVirtualRef`
/// (mirrors `virtualref.py:9 from rpython.rlib.jit import
/// InvalidVirtualRef`); re-exported here for convenience.
pub use crate::jit::InvalidVirtualRef;

/// Allocate a concrete JitVirtualRef on the heap.
/// `virtualref.py:85-91 virtual_ref_during_tracing(real_object)`.
/// Initializes virtual_token = TOKEN_NONE, forced = real_object.
/// Returns raw pointer; caller owns the allocation.
pub fn alloc_virtual_ref(real_object: *mut u8) -> *mut u8 {
    let vref = Box::new(JitVirtualRef {
        super_: ObjectHeader {
            typeptr: JIT_VIRTUAL_REF_VTABLE,
        },
        virtual_token: TOKEN_NONE,
        forced: real_object,
    });
    Box::into_raw(vref) as *mut u8
}

/// Token value indicating no JIT frame is active.
/// `virtualizable.py:329 TOKEN_NONE = lltype.nullptr(llmemory.GCREF.TO)`.
pub const TOKEN_NONE: *mut u8 = std::ptr::null_mut();

/// `virtualizable.py:326`:
/// ```python
/// _DUMMY = lltype.GcStruct('JITFRAME_DUMMY')
/// ```
/// Pyre stores the corresponding type-id as a u64 magic constant —
/// the typeptr written into the `super_.typeptr` slot of the
/// allocated `_dummy` instance below.
pub const JITFRAME_DUMMY_VTABLE: u64 = 0x4A46_4D44_554D_4D59; // "JFMDUMMY"

/// Lazy initialisation of the `_dummy` address.  `OnceLock<usize>`
/// (instead of `OnceLock<*mut u8>`) so the cell is `Sync` —
/// raw-pointer types are not.
static TRACING_RESCALL_DUMMY_PTR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// `virtualizable.py:327 _dummy = lltype.malloc(_DUMMY)` — allocate
/// the singleton dummy `JITFRAME_DUMMY` object whose address serves
/// as the tracing sentinel.  Pyre's `Box::into_raw(Box::new(...))`
/// produces a heap-allocated, stable, non-null address; the
/// `Box::leak` semantic (the box is intentionally never freed)
/// matches upstream's `immortal=True`-equivalent lifetime — `_dummy`
/// is allocated once at first use and stays live for the rest of
/// the program.
fn allocate_tracing_rescall_dummy() -> *mut u8 {
    let header = Box::new(ObjectHeader {
        typeptr: JITFRAME_DUMMY_VTABLE,
    });
    Box::into_raw(header) as *mut u8
}

/// Token value used during tracing when a residual call is in progress.
/// `virtualizable.py:330`:
/// ```python
/// TOKEN_TRACING_RESCALL = lltype.cast_opaque_ptr(llmemory.GCREF, _dummy)
/// ```
/// Pyre returns the address of a real heap-allocated `ObjectHeader`
/// initialised lazily on first call; subsequent calls return the
/// same address (program-lifetime immortal).
///
/// TODO (GC registration).  Upstream's `_dummy`
/// is a real GcStruct that the collector knows about; pyre's leaked
/// `Box<ObjectHeader>` is host-allocated memory the GC has no
/// record of.  The adaptation is internally consistent because
/// `virtual_token` is not GC-traced either (see `JitVirtualRef`
/// doc-comment); if the slot ever becomes GC-traced, the `_dummy`
/// allocation must move to the GC heap and a JITFRAME_DUMMY type
/// must be registered with the collector.
#[inline]
pub fn token_tracing_rescall() -> *mut u8 {
    *TRACING_RESCALL_DUMMY_PTR.get_or_init(|| allocate_tracing_rescall_dummy() as usize) as *mut u8
}

/// Virtual reference state for a single reference.
///
/// A virtual reference wraps a virtualizable object during JIT execution.
/// When code outside the JIT tries to access the virtual object, the
/// reference is "forced" -- the virtual object is materialized on the heap.
///
/// `virtualref.py:32-42` parity: `descr` / `descr_virtual_token` /
/// `descr_forced` hold the live `cpu.sizeof(JIT_VIRTUAL_REF)` and
/// `cpu.fielddescrof(JIT_VIRTUAL_REF, 'virtual_token' | 'forced')`
/// Arcs.  Pyre's underlying generators (`vref_size_descr()` /
/// `make_vref_field_descr_typed(...)`) cache the Arc identity at
/// module level so every read of these fields returns the same Arc
/// that the optimizer-emit sites stamp into `op.descr`
/// (`virtualize.rs:1520 / 1527`) — `Arc::ptr_eq` parity with
/// `history.py:125 cpu.get_latest_descr() is op.getdescr()`.
#[derive(Debug, Clone)]
pub struct VirtualRefInfo {
    /// `virtualref.py:32-33` `self.descr = cpu.sizeof(JIT_VIRTUAL_REF, ...)`
    pub descr: majit_ir::DescrRef,
    /// `virtualref.py:40-41` `self.descr_virtual_token = cpu.fielddescrof(..., 'virtual_token')`
    pub descr_virtual_token: majit_ir::DescrRef,
    /// `virtualref.py:42`    `self.descr_forced        = cpu.fielddescrof(..., 'forced')`
    pub descr_forced: majit_ir::DescrRef,
}

impl Default for VirtualRefInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::resume::VRefInfo for VirtualRefInfo {
    /// virtualref.py:122-129 continue_tracing(gcref, real_object)
    ///
    /// Mirrors RPython:
    ///   if not self.is_virtual_ref(gcref): return
    ///   assert real_object
    ///   vref = ...; assert vref.virtual_token != TOKEN_TRACING_RESCALL
    ///   vref.virtual_token = TOKEN_NONE
    ///   vref.forced = real_object
    fn continue_tracing(&self, vref: i64, virtual_ref: i64) {
        // `virtualref.py:122-129 continue_tracing(gcref, real_object)`
        // — delegate to the inherent helper which carries the
        // `is_virtual_ref` guard, `assert real_object`, and
        // `assert vref.virtual_token != TOKEN_TRACING_RESCALL`
        // post-conditions.
        unsafe {
            self.continue_tracing(vref as *mut u8, virtual_ref as *mut u8);
        }
    }
}

// Project convention: RPython's plain `assert ...` ports to
// `debug_assert!(...)`.  Both elide under the production translation
// (RPython `-O`, pyre `--release`); both fire under `assert ...` /
// `cargo test` (RPython untranslated, pyre debug build).  Where
// upstream uses `if not we_are_translated(): assert ...` the port is
// `#[cfg(debug_assertions)] { assert!(...) }` (always-on under
// debug_assertions, elided in release).  See `~/.claude/projects/.../
// memory/feedback_rpython_assert_parity_always_on_2026_05_05.md`.
//
// The `debug_assert!` calls below correspond to upstream `assert`
// statements at:
//   virtualref.py:104  (assert vref.virtual_token == TOKEN_NONE)
//   virtualref.py:111  (assert vref.forced)
//   virtualref.py:115  (assert vref.virtual_token == TOKEN_TRACING_RESCALL)
//   virtualref.py:125  (assert real_object)
//   virtualref.py:127  (assert vref.virtual_token != TOKEN_TRACING_RESCALL)
//   virtualref.py:166  (assert vref.forced)
//   virtualref.py:169  (assert not vref.forced)
//   virtualref.py:172  (assert vref.virtual_token == TOKEN_NONE)
//   virtualref.py:173  (assert vref.forced)
impl VirtualRefInfo {
    /// Create a VirtualRefInfo.
    pub fn new() -> Self {
        use crate::optimizeopt::virtualize::{
            VREF_FORCED_FIELD_INDEX, VREF_VIRTUAL_TOKEN_FIELD_INDEX, make_vref_field_descr_pub,
            vref_size_descr,
        };
        VirtualRefInfo {
            descr: vref_size_descr(),
            descr_virtual_token: make_vref_field_descr_pub(VREF_VIRTUAL_TOKEN_FIELD_INDEX),
            descr_forced: make_vref_field_descr_pub(VREF_FORCED_FIELD_INDEX),
        }
    }

    /// `virtualref.py:157-177 force_virtual(inst)` — force a virtual
    /// reference: materialise the virtual object if needed.
    ///
    /// Returns `Err(InvalidVirtualRef)` when `virtual_token ==
    /// TOKEN_NONE` and `forced` is null — the vref was tracked but
    /// never properly initialised (`virtualref.py:174-176`).
    ///
    /// `force_now` mirrors upstream's
    /// `ResumeGuardForcedDescr.force_now(cpu, token)` contract
    /// (`compile.py:966-985`):
    ///
    /// ```python
    /// @staticmethod
    /// def force_now(cpu, token):
    ///     deadframe = cpu.force(token)
    ///     faildescr = cpu.get_latest_descr(deadframe)
    ///     assert isinstance(faildescr, ResumeGuardForcedDescr)
    ///     faildescr.handle_async_forcing(deadframe)
    /// ```
    ///
    /// `force_now` MUST mutate the vref through the writeback the
    /// JIT frame performs during `cpu.force(token)` —
    /// `vref.virtual_token` becomes `TOKEN_NONE` and `vref.forced`
    /// becomes the materialised object.  The post-call assertions
    /// (`virtualref.py:172-173`) verify these writebacks; pyre
    /// mirrors them as `debug_assert!`s after `force_now` returns.
    ///
    /// TODO (dependency injection).  Upstream
    /// passes `cpu` directly because `ResumeGuardForcedDescr` lives
    /// alongside the runner; pyre's `VirtualRefInfo` cannot import
    /// `dyn Runner` without a crate-cycle (the runner trait lives
    /// in the backend crate).  The closure parameter is the Rust
    /// adaptation: callers (which have both `cpu` and
    /// `MetaInterp::handle_async_forcing` in scope) bind the
    /// closure to `|vref| { let df = cpu.force(...); ... }` — same
    /// effect, opposite control direction.  Convergence path:
    /// promote `Runner` to a metainterp-visible trait (or pull the
    /// resume-data handling into a separate crate without runner
    /// dependency) so `force_now` can call `cpu.force` directly.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn force_virtual(
        &self,
        vref_ptr: *mut u8,
        force_now: impl FnOnce(*mut JitVirtualRef),
    ) -> Result<*mut u8, InvalidVirtualRef> {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            let token = vref.virtual_token;
            if token != TOKEN_NONE {
                if token == token_tracing_rescall() {
                    // `virtualref.py:161-167`: "virtual" is not a
                    // virtual at all during tracing; reset token as
                    // the marker that this "virtual" escapes.
                    debug_assert!(!vref.forced.is_null());
                    vref.virtual_token = TOKEN_NONE;
                } else {
                    // `virtualref.py:168-173`: active-frame token —
                    // delegate to the cpu/descr force_now sequence,
                    // then assert post-conditions on the writeback.
                    debug_assert!(vref.forced.is_null());
                    force_now(vref);
                    debug_assert_eq!(vref.virtual_token, TOKEN_NONE);
                    debug_assert!(!vref.forced.is_null());
                }
            } else if vref.forced.is_null() {
                // `virtualref.py:174-176`: token == TOKEN_NONE and
                // the vref was not forced — invalid.
                return Err(InvalidVirtualRef);
            }
            Ok(vref.forced)
        }
    }

    /// Create a virtual reference during tracing.
    ///
    /// virtualref.py:85-91: `virtual_ref_during_tracing(real_object)`
    ///
    /// Allocates a concrete JitVirtualRef on the heap with
    /// virtual_token = TOKEN_NONE, forced = real_object.
    pub fn virtual_ref_during_tracing(&self, real_object: *mut u8) -> *mut u8 {
        alloc_virtual_ref(real_object)
    }

    /// `virtualref.py:100-105 tracing_before_residual_call(gcref)` —
    /// sets token to `TOKEN_TRACING_RESCALL` so that if the callee
    /// forces the vref, we detect it after the call.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns silently
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose first 8 bytes are the type-tag word.
    pub unsafe fn tracing_before_residual_call(&self, vref_ptr: *mut u8) {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return;
            }
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:104 assert vref.virtual_token == TOKEN_NONE`
            debug_assert_eq!(vref.virtual_token, TOKEN_NONE);
            vref.virtual_token = token_tracing_rescall();
        }
    }

    /// `virtualref.py:107-120 tracing_after_residual_call(gcref)` —
    /// returns `true` if the vref was forced during the residual
    /// call.  If not forced, resets token to `TOKEN_NONE`.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns `False`
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose first 8 bytes are the type-tag word.
    pub unsafe fn tracing_after_residual_call(&self, vref_ptr: *mut u8) -> bool {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return false;
            }
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:111 assert vref.forced` — by construction
            // a JitVirtualRef has its `forced` field initialised to a
            // non-null real_object at allocation time.
            debug_assert!(!vref.forced.is_null());
            if vref.virtual_token != TOKEN_NONE {
                // `virtualref.py:113-117` not modified by the residual
                // call; assert it is still TOKEN_TRACING_RESCALL and
                // clear it.
                debug_assert_eq!(vref.virtual_token, token_tracing_rescall());
                vref.virtual_token = TOKEN_NONE;
                false
            } else {
                // `virtualref.py:118-120` token was cleared by the
                // callee — "modified during residual call" marker.
                true
            }
        }
    }

    /// `virtualref.py:122-129 continue_tracing(gcref, real_object)` —
    /// updates the `forced` field and clears the token.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns silently
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose first 8 bytes are the type-tag word.
    pub unsafe fn continue_tracing(&self, vref_ptr: *mut u8, real_object: *mut u8) {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return;
            }
            // `virtualref.py:125 assert real_object` — caller must
            // supply a non-null materialised object.
            debug_assert!(!real_object.is_null());
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:127 assert vref.virtual_token != TOKEN_TRACING_RESCALL`
            debug_assert_ne!(vref.virtual_token, token_tracing_rescall());
            vref.virtual_token = TOKEN_NONE;
            vref.forced = real_object;
        }
    }

    /// Check if a virtual reference is currently active (has a JIT frame token).
    ///
    /// virtualref.py: token != TOKEN_NONE and token != TOKEN_TRACING_RESCALL
    pub fn is_active(token: *mut u8) -> bool {
        token != TOKEN_NONE && token != token_tracing_rescall()
    }

    /// Check if a virtual reference is forced (token == TOKEN_NONE).
    pub fn is_forced(token: *mut u8) -> bool {
        token == TOKEN_NONE
    }

    /// Check if a virtual reference is in a residual call.
    pub fn is_in_residual_call(token: *mut u8) -> bool {
        token == token_tracing_rescall()
    }

    /// `virtualref.py:94-98 is_virtual_ref(gcref)`:
    /// ```python
    /// def is_virtual_ref(self, gcref):
    ///     if not gcref:
    ///         return False
    ///     inst = lltype.cast_opaque_ptr(rclass.OBJECTPTR, gcref)
    ///     return inst.typeptr == self.jit_virtual_ref_vtable
    /// ```
    /// Pyre reads `super_.typeptr` (offset 0, the
    /// `('super', rclass.OBJECT)` slot) and compares it against
    /// `JIT_VIRTUAL_REF_VTABLE`.
    ///
    /// # Safety
    /// `ptr` must point to a valid GCREF object whose first 8 bytes
    /// are the typeptr word, or be null.
    pub unsafe fn is_virtual_ref(&self, ptr: *const u8) -> bool {
        unsafe {
            if ptr.is_null() {
                return false;
            }
            let header = ptr as *const ObjectHeader;
            (*header).typeptr == JIT_VIRTUAL_REF_VTABLE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_constants() {
        assert!(
            TOKEN_NONE.is_null(),
            "TOKEN_NONE = nullptr(GCREF) per virtualizable.py:329"
        );
        let rescall = token_tracing_rescall();
        assert!(
            !rescall.is_null(),
            "token_tracing_rescall() = cast_opaque_ptr(_dummy) per virtualizable.py:330 — non-null sentinel",
        );
        assert_ne!(
            TOKEN_NONE, rescall,
            "TOKEN_NONE and token_tracing_rescall() must be distinguishable",
        );
        // Stable across calls — `_dummy = lltype.malloc(_DUMMY)` is a single
        // program-lifetime allocation, so repeated reads must yield the
        // same address.
        assert_eq!(rescall, token_tracing_rescall());
        // Address must point at memory we own (the heap-allocated
        // `ObjectHeader` for `_dummy`), not a poison value like
        // `usize::MAX as *mut u8`.  Read-back of the typeptr field
        // probes that a real `_dummy` allocation backs the token —
        // matches `JITFRAME_DUMMY_VTABLE`.
        unsafe {
            let header = rescall as *const ObjectHeader;
            assert_eq!((*header).typeptr, JITFRAME_DUMMY_VTABLE);
        }
    }

    /// `virtualref.py:174-176`: `force_virtual` raises
    /// `InvalidVirtualRef` when `virtual_token == TOKEN_NONE` and
    /// `forced` is null.  pyre surfaces the same shape as
    /// `Err(InvalidVirtualRef)`.
    #[test]
    fn force_virtual_invalid_when_token_none_and_forced_null() {
        let info = VirtualRefInfo::new();
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: TOKEN_NONE,
            forced: std::ptr::null_mut(),
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        let err = unsafe {
            info.force_virtual(vref_ptr, |_| {
                panic!("force_now must not be called when token == TOKEN_NONE")
            })
        }
        .expect_err(
            "TOKEN_NONE + null forced must surface Err(InvalidVirtualRef) per virtualref.py:174-176",
        );
        assert_eq!(err, InvalidVirtualRef);
    }

    /// `virtualref.py:160-167`: when `virtual_token ==
    /// TOKEN_TRACING_RESCALL`, `force_virtual` resets the token to
    /// `TOKEN_NONE` and returns the existing `forced` value
    /// without calling the materialiser.
    #[test]
    fn force_virtual_tracing_rescall_resets_token() {
        let info = VirtualRefInfo::new();
        let real_obj: *mut u8 = 0xCAFE_F00Du64 as *mut u8;
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: token_tracing_rescall(),
            forced: real_obj,
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        let result = unsafe {
            info.force_virtual(vref_ptr, |_| {
                panic!("force_now must not be called for TOKEN_TRACING_RESCALL branch")
            })
        }
        .expect("tracing-rescall branch must succeed");
        assert_eq!(result, real_obj);
        assert!(
            vref.virtual_token.is_null(),
            "token must be reset to TOKEN_NONE"
        );
    }

    /// `virtualref.py:168-173`: when `virtual_token != TOKEN_NONE`
    /// and != `TOKEN_TRACING_RESCALL`, `force_virtual` invokes
    /// `force_now`, which (mirroring upstream's `cpu.force(token)`
    /// + `handle_async_forcing`) must mutate the vref so that
    /// `virtual_token` becomes `TOKEN_NONE` and `forced` becomes
    /// the materialised object.  pyre's closure receives the
    /// `*mut JitVirtualRef` and performs the writeback in-place.
    #[test]
    fn force_virtual_active_frame_invokes_force_now_with_vref_mutation() {
        let info = VirtualRefInfo::new();
        let active_token: *mut u8 = 0xABCDu64 as *mut u8;
        let materialized: *mut u8 = 0xCAFEu64 as *mut u8;
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: active_token,
            forced: std::ptr::null_mut(),
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        let result = unsafe {
            info.force_virtual(vref_ptr, |v| {
                // Equivalent of `cpu.force(token)` writeback +
                // `faildescr.handle_async_forcing(deadframe)`:
                // populate forced and clear the token.
                (*v).forced = materialized;
                (*v).virtual_token = TOKEN_NONE;
            })
        }
        .expect("active-frame branch must succeed");
        assert_eq!(result, materialized);
        assert!(vref.virtual_token.is_null());
        assert_eq!(vref.forced, materialized);
    }

    /// `virtualref.py:101-102`: `tracing_before_residual_call`
    /// returns silently when the gcref is not a JitVirtualRef.
    /// pyre's `is_virtual_ref` guard mirrors the shape — a
    /// non-vref pointer (here a `u64` whose first 8 bytes are NOT
    /// `JIT_VIRTUAL_REF_VTABLE`) must be left untouched.
    #[test]
    fn tracing_before_residual_call_skips_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = &mut not_a_vref as *mut u64 as *mut u8;
        unsafe {
            info.tracing_before_residual_call(ptr);
        }
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// `virtualref.py:108-109`: `tracing_after_residual_call`
    /// returns `false` for a non-vref input.
    #[test]
    fn tracing_after_residual_call_returns_false_for_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = &mut not_a_vref as *mut u64 as *mut u8;
        let forced = unsafe { info.tracing_after_residual_call(ptr) };
        assert!(!forced, "non-vref input must surface as not-forced");
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// `virtualref.py:123-124`: `continue_tracing` returns silently
    /// when the gcref is not a JitVirtualRef.
    #[test]
    fn continue_tracing_skips_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = &mut not_a_vref as *mut u64 as *mut u8;
        let real_obj: *mut u8 = 0xDEADBEEFu64 as *mut u8;
        unsafe {
            info.continue_tracing(ptr, real_obj);
        }
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// Sanity: a real `JitVirtualRef` IS recognised by
    /// `is_virtual_ref` and the tracing helpers DO mutate it
    /// (`virtualref.py:103-105` happy path).
    #[test]
    fn tracing_before_residual_call_mutates_real_vref() {
        let info = VirtualRefInfo::new();
        let real_obj: *mut u8 = 0xCAFE_F00Du64 as *mut u8;
        let mut vref = JitVirtualRef {
            super_: ObjectHeader {
                typeptr: JIT_VIRTUAL_REF_VTABLE,
            },
            virtual_token: TOKEN_NONE,
            forced: real_obj,
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        unsafe {
            info.tracing_before_residual_call(vref_ptr);
        }
        assert_eq!(
            vref.virtual_token,
            token_tracing_rescall(),
            "real vref must transition to TOKEN_TRACING_RESCALL",
        );
    }
}
