//! Runtime GC type-id registry for the `StringBuilder` (rbuilder) value.
//!
//! The JIT models RPython's `StringBuilder` as a bare `GcStruct` (no `w_class`)
//! that the `New{"stringbuilder"}` opcode allocates. `current_buf` points at a
//! raw `std::alloc` low-level string (pyre's `rstr.STR` equivalent), off-GC and
//! immobile, so the tid carries drop glue (`StringBuilderBox::drop` in
//! `pyre-jit::eval`) to free it on sweep. `extra_pieces` is a **GC edge**: the
//! grow path allocates each chain node with `malloc(STRINGPIECE)` (a GC alloc),
//! so the collector traces that field (the tid registers `gc_ptr_offsets` for
//! it) and reclaims the STRINGPIECE nodes itself. The tid is assigned at runtime
//! by `pyre-jit::eval` after the fixed-constant type registrations and published
//! here so `pyre-jit-trace`'s size descriptor can stamp it into the allocation
//! shape.

/// Runtime-assigned GC type id for the `StringBuilder` box. Published by
/// `pyre-jit::eval` at the tail of `build_gc`; read by the size descriptor in
/// `pyre-jit-trace::descr` and by `bh_new`.
static STRINGBUILDER_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the `StringBuilder` box.
pub fn set_stringbuilder_gc_type_id(id: u32) {
    STRINGBUILDER_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the `StringBuilder` box.
#[majit_macros::dont_look_inside]
pub fn stringbuilder_gc_type_id() -> u32 {
    STRINGBUILDER_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}

/// Runtime-assigned GC type id for a `StringPiece` chain node — the
/// `extra_pieces` chain the builder grows when its `current_buf` fills. Each
/// node is a bare `GcStruct("stringpiece", {buf, prev_piece})`: `buf` is a raw
/// off-GC low-level string (drop glue frees it), `prev_piece` is a `Ref` edge to
/// the previous node (traced). Registered by `pyre-jit::eval` right after the
/// builder tid; read by `pyre-jit-trace`'s size descriptor and by the grow
/// primitive's `malloc(STRINGPIECE)`.
static STRINGPIECE_GC_TYPE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Record the GC type id registered for the `StringPiece` node.
pub fn set_stringpiece_gc_type_id(id: u32) {
    STRINGPIECE_GC_TYPE_ID.store(id, std::sync::atomic::Ordering::Relaxed);
}

/// Read the runtime-assigned GC type id for the `StringPiece` node.
#[majit_macros::dont_look_inside]
pub fn stringpiece_gc_type_id() -> u32 {
    STRINGPIECE_GC_TYPE_ID.load(std::sync::atomic::Ordering::Relaxed)
}
