//! Runtime GC type-id registry for the `StringBuilder` (rbuilder) value.
//!
//! The JIT models RPython's `StringBuilder` as a bare `GcStruct` (no `w_class`)
//! allocated by the ordinary `new` operation. `current_buf` is a GC pointer to
//! the low-level `STR`, and `extra_pieces` is a GC pointer to the linked
//! `STRINGPIECE` nodes. This mirrors `rpython/rtyper/lltypesystem/rbuilder.py`:
//! ownership is expressed entirely by traced fields rather than Rust drop glue.
//! The runtime type id is published here so `pyre-jit-trace` can stamp it into
//! the allocation descriptor.

/// Field byte offsets and total body size of the `StringBuilder` bare GcStruct.
/// Single source of truth shared by the runtime `StringBuilderBox` layout
/// (`pyre-jit::eval`, where `offset_of!` const asserts pin the struct to these)
/// and the size descriptor (`pyre-jit-trace::descr`, which stamps them into the
/// allocation shape). Reordering the runtime struct without updating these fails
/// the eval const asserts at compile time, so the descriptor can never reserve a
/// body that disagrees with the struct.
pub const STRINGBUILDER_SIZE: usize = 40;
pub const STRINGBUILDER_CURRENT_BUF_OFFSET: usize = 0;
pub const STRINGBUILDER_CURRENT_POS_OFFSET: usize = 8;
pub const STRINGBUILDER_CURRENT_END_OFFSET: usize = 16;
pub const STRINGBUILDER_TOTAL_SIZE_OFFSET: usize = 24;
pub const STRINGBUILDER_EXTRA_PIECES_OFFSET: usize = 32;

/// Field byte offsets and total body size of the `StringPiece` chain node — the
/// same single-source-of-truth contract as [`STRINGBUILDER_SIZE`] et al.
pub const STRINGPIECE_SIZE: usize = 16;
pub const STRINGPIECE_BUF_OFFSET: usize = 0;
pub const STRINGPIECE_PREV_PIECE_OFFSET: usize = 8;

/// Runtime-assigned GC type id for the `StringBuilder` box. Published by
/// `pyre-jit::eval` at the tail of `build_gc`; read by the size descriptor in
/// `pyre-jit-trace::descr` and by `bh_new`.
static STRINGBUILDER_GC_TYPE_ID: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the `StringBuilder` box. `Release` so
/// the `gc.register_type` entry `pyre-jit::eval build_gc` filled before this
/// store (including its `gc_ptr_offsets`) is visible to any `Acquire` reader.
pub fn set_stringbuilder_gc_type_id(id: u32) {
    debug_assert_ne!(id, 0, "0 is the unpublished sentinel");
    STRINGBUILDER_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
}

/// Read the runtime-assigned GC type id for the `StringBuilder` box. `Acquire`
/// pairs with the `Release` store in `set_stringbuilder_gc_type_id` so the
/// `gc.register_type` registration is visible here. A 0 result is the
/// pre-publish sentinel: `build_gc` publishes the real tid before any JIT
/// allocation, so 0 only occurs pre-init or in a unit test, where
/// `gc_alloc_storage_box` treats it as the documented `malloc_raw` fallback.
#[majit_macros::dont_look_inside]
pub fn stringbuilder_gc_type_id() -> u32 {
    STRINGBUILDER_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}

/// Runtime-assigned GC type id for a `StringPiece` chain node — the
/// `extra_pieces` chain the builder grows when its `current_buf` fills. Each
/// node is a bare `GcStruct("stringpiece", {buf, prev_piece})`; both fields are
/// traced GC references, matching the RPython definition. Registered by
/// `pyre-jit::eval` right after the builder tid; read by `pyre-jit-trace`'s size
/// descriptor and by the grow primitive's `malloc(STRINGPIECE)`.
static STRINGPIECE_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the `StringPiece` node. `Release` so
/// the `gc.register_type` entry `pyre-jit::eval build_gc` filled before this
/// store (including its `gc_ptr_offsets`) is visible to any `Acquire` reader.
pub fn set_stringpiece_gc_type_id(id: u32) {
    debug_assert_ne!(id, 0, "0 is the unpublished sentinel");
    STRINGPIECE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Release);
}

/// Read the runtime-assigned GC type id for the `StringPiece` node. `Acquire`
/// pairs with the `Release` store in `set_stringpiece_gc_type_id` so the
/// `gc.register_type` registration is visible here. A 0 result is the
/// pre-publish sentinel: `build_gc` publishes the real tid before any JIT
/// allocation, so 0 only occurs pre-init or in a unit test, where
/// `gc_alloc_storage_box` treats it as the documented `malloc_raw` fallback.
#[majit_macros::dont_look_inside]
pub fn stringpiece_gc_type_id() -> u32 {
    STRINGPIECE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Acquire)
}
