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

/// Runtime realization of the RPython `StringBuilder` GcStruct (rbuilder epic
/// task #43). A bare (no `w_class`) but *headered* GcStruct carrying two traced
/// fields:
/// - `current_buf` (offset 0) is RPython's `Ptr(STR)` GC edge.
/// - `extra_pieces` (offset 32) is a **GC edge**: the grow path allocates each
///   chain node with `malloc(STRINGPIECE)` (a `New{"stringpiece"}` GC alloc), so
///   the collector traces and reclaims the chain.
/// `#[repr(C)]` with five `i64` fields fixes the body layout the analyzer's
/// `New{"stringbuilder"}` size descriptor mirrors: current_buf@0, current_pos@8,
/// current_end@16, total_size@24, extra_pieces@32, size 40.
// Built via an `alloc_builder` struct literal in the `ll_new` ctor and handed to
// `gc_alloc_storage_box` (task #48); `current_pos`/`current_end`/`total_size` are
// then written in place by the ctor/append/grow primitives. The layout slots are
// load-bearing, so silence the unread-field lints.
#[allow(dead_code)]
#[repr(C)]
struct StringBuilderBox {
    current_buf: i64,
    current_pos: i64,
    current_end: i64,
    total_size: i64,
    extra_pieces: i64,
}

// Body size and field offsets are the single source of truth in
// `pyre_object::rbuilder`, which the size descriptor (`pyre-jit-trace::descr`)
// also reads. Pin the struct to them so a reorder here fails to compile rather
// than silently disagreeing with the allocation shape.
const _: () = assert!(
    std::mem::size_of::<StringBuilderBox>() == STRINGBUILDER_SIZE,
    "StringBuilderBox body must match the stringbuilder size descriptor",
);
const _: () = assert!(
    std::mem::offset_of!(StringBuilderBox, current_buf) == STRINGBUILDER_CURRENT_BUF_OFFSET
);
const _: () = assert!(
    std::mem::offset_of!(StringBuilderBox, current_pos) == STRINGBUILDER_CURRENT_POS_OFFSET
);
const _: () = assert!(
    std::mem::offset_of!(StringBuilderBox, current_end) == STRINGBUILDER_CURRENT_END_OFFSET
);
const _: () =
    assert!(std::mem::offset_of!(StringBuilderBox, total_size) == STRINGBUILDER_TOTAL_SIZE_OFFSET);
const _: () = assert!(
    std::mem::offset_of!(StringBuilderBox, extra_pieces) == STRINGBUILDER_EXTRA_PIECES_OFFSET
);

/// One node of a [`StringBuilderBox`] `extra_pieces` chain (rbuilder task #48).
/// RPython `rbuilder.py` STRINGPIECE `GcStruct("stringpiece", {buf, prev_piece})`:
/// - `buf` (offset 0) is a traced `Ptr(STR)` handed off when the current buffer
///   filled.
/// - `prev_piece` (offset 8) is a **GC edge** to the previous node (`0` at the
///   chain end); the collector traces and reclaims the previous node.
/// A bare (no `w_class`) headered GcStruct: `bh_new`/`malloc(STRINGPIECE)`
/// allocates it; the item size for `buf` is the STR width (1).
// Built via an `alloc_piece` struct literal in the `ll_grow_by` grow primitive
// and handed to `gc_alloc_storage_box` (task #48).
#[allow(dead_code)]
#[repr(C)]
struct StringPieceBox {
    buf: i64,
    prev_piece: i64,
}

const _: () = assert!(
    std::mem::size_of::<StringPieceBox>() == STRINGPIECE_SIZE,
    "StringPieceBox body must match the stringpiece size descriptor",
);
const _: () = assert!(std::mem::offset_of!(StringPieceBox, buf) == STRINGPIECE_BUF_OFFSET);
const _: () =
    assert!(std::mem::offset_of!(StringPieceBox, prev_piece) == STRINGPIECE_PREV_PIECE_OFFSET);

/// Native runtime realization of the RPython `StringBuilder` helper graphs
/// (`rpython/rtyper/lltypesystem/rbuilder.py`) — rbuilder epic task #48b.
///
/// The `New{"stringbuilder"}`/`New{"stringpiece"}` + FieldRead/FieldWrite the
/// analyzer drains manipulate the [`StringBuilderBox`]/[`StringPieceBox`] layouts
/// registered in `pyre-jit::eval build_gc`; these are the leaf primitives those
/// graphs call (`mallocfn` / `copy_string_contents` / `ll_shrink_array` /
/// `malloc(STRINGPIECE)`). The `dont_look_inside` residual append targets
/// (`jit_ll_append_res0`/`jit_ll_append_res_slice`) enter here from the codewriter
/// when an append cannot be inlined.
///
/// String width only (`item_size == 1`) — the `STR` specialization the analyzer
/// drains; the `UNICODE` builder is a future parallel set.
///
/// `current_buf` and every STRINGPIECE `buf` are ordinary GC references, so the
/// aliasing and reuse paths below follow `rbuilder.py` directly.
#[allow(dead_code)]
pub mod rbuilder_runtime {
    use super::{StringBuilderBox, StringPieceBox};
    use crate::lowlevel_string::{
        LOWLEVEL_STR_BASE_SIZE, LOWLEVEL_UNICODE_BASE_SIZE, bh_alloc_lowlevel_string,
        bh_free_lowlevel_string, bh_lowlevel_chars_offset, bh_lowlevel_string_len,
        bh_write_lowlevel_char,
    };

    /// `StringBuilderRepr.basetp = STR` ⇒ one byte per char.
    pub const STR_ITEM_SIZE: usize = 1;
    /// `ll_new` clamps `init_size` into `[0, 1280]`.
    const INIT_SIZE_MAX: i64 = 1280;

    #[inline]
    fn str_base_size(item_size: usize) -> usize {
        if item_size == 1 {
            LOWLEVEL_STR_BASE_SIZE
        } else {
            LOWLEVEL_UNICODE_BASE_SIZE
        }
    }

    /// `rstr.copy_string_contents(src, dst, srcstart, dststart, length)`: copy
    /// `count` chars from `src[src_start..]` into `dst[dst_start..]`. `src` and
    /// `dst` are always distinct allocations on the builder paths.
    fn copy_string_contents(
        src: i64,
        dst: i64,
        src_start: usize,
        dst_start: usize,
        count: usize,
        item_size: usize,
    ) {
        if src == 0 || dst == 0 || count == 0 {
            return;
        }
        let chars_off = bh_lowlevel_chars_offset(item_size);
        // SAFETY: the builder's size arithmetic guarantees both ranges fit; the
        // two low-level strings are distinct allocations.
        unsafe {
            let s = (src as *const u8).add(chars_off + src_start * item_size);
            let d = (dst as *mut u8).add(chars_off + dst_start * item_size);
            std::ptr::copy_nonoverlapping(s, d, count * item_size);
        }
    }

    /// `rgc.ll_shrink_array(buf, new_len)`: allocate a fresh `new_len` buffer
    /// when the collector cannot shrink this varsize leaf in place.
    fn ll_shrink_array(buf: i64, new_len: usize, item_size: usize) -> i64 {
        if buf == 0 {
            return 0;
        }
        let base = str_base_size(item_size);
        let new_buf = bh_alloc_lowlevel_string(new_len, base, item_size);
        if new_buf == 0 {
            return 0;
        }
        copy_string_contents(buf, new_buf, 0, 0, new_len, item_size);
        bh_free_lowlevel_string(buf, base, item_size);
        new_buf
    }

    #[inline]
    fn alloc_builder(value: StringBuilderBox) -> i64 {
        let tid = pyre_object::rbuilder::stringbuilder_gc_type_id();
        pyre_object::gc_storage::gc_alloc_storage_box::<StringBuilderBox>(value, tid) as i64
    }

    #[inline]
    fn alloc_piece(value: StringPieceBox) -> i64 {
        let tid = pyre_object::rbuilder::stringpiece_gc_type_id();
        pyre_object::gc_storage::gc_alloc_storage_box::<StringPieceBox>(value, tid) as i64
    }

    /// `StringBuilderRepr.ll_new(init_size)` (`rbuilder.py`).
    pub fn ll_new(init_size: i64, item_size: usize) -> i64 {
        // This runtime entry is the StringBuilder specialization; the Unicode
        // builder uses the parallel translated repr.
        assert_eq!(
            item_size, 1,
            "StringBuilder is STR-only until the destructors widen"
        );
        // `intmask(min(r_uint(init_size), r_uint(1280)))` — negatives (huge as
        // unsigned) and anything over 1280 clamp to 1280.
        let init = if !(0..=INIT_SIZE_MAX).contains(&init_size) {
            INIT_SIZE_MAX as usize
        } else {
            init_size as usize
        };
        let base = str_base_size(item_size);
        let current_buf = bh_alloc_lowlevel_string(init, base, item_size);
        if current_buf == 0 {
            return 0;
        }
        alloc_builder(StringBuilderBox {
            current_buf,
            current_pos: 0,
            current_end: init as i64,
            total_size: init as i64,
            extra_pieces: 0,
        })
    }

    /// `rbuilder.py ll_grow_by` growth arithmetic — the three `ovfcheck`s that
    /// add the requested growth to `total_size` and round it up to a multiple of
    /// 64. Upstream any overflow is `raise MemoryError`; this returns `None` so
    /// the caller fails cleanly instead of wrapping to a small allocation.
    /// Returns `(needed, new_total)`.
    fn grow_by_sizes(needed: i64, total_size: i64) -> Option<(i64, i64)> {
        let needed = needed.checked_add(total_size)?;
        let needed = needed.checked_add(63)? & !63;
        let new_total = total_size.checked_add(needed)?;
        Some((needed, new_total))
    }

    /// `rbuilder.py ll_grow_by`: round the growth up to a multiple of 64,
    /// chain the filled `current_buf` into a fresh STRINGPIECE, install a new
    /// empty `current_buf`. The growth `ovfcheck`s are ported via
    /// [`grow_by_sizes`]; upstream raises `MemoryError` on overflow, which the
    /// residual shim cannot, so it aborts loudly instead. (The JIT
    /// graph-builder's `int_add_ovf` exception edge is a separate deferred half.)
    fn ll_grow_by(builder: i64, needed: i64, item_size: usize) {
        // SAFETY: `builder` is a live StringBuilderBox body pointer.
        let (old_buf, old_extra, total_size) = {
            let b = unsafe { &*(builder as *const StringBuilderBox) };
            (b.current_buf, b.extra_pieces, b.total_size)
        };
        let (needed, new_total) = grow_by_sizes(needed, total_size).unwrap_or_else(|| {
            // Upstream `except OverflowError: raise MemoryError`; abort loudly
            // rather than wrap to a small allocation and corrupt the build.
            panic!("StringBuilder grow size overflow; MemoryError propagation is not ported yet")
        });
        let needed = needed as usize;
        let base = str_base_size(item_size);
        let new_string = bh_alloc_lowlevel_string(needed, base, item_size);
        // A null buffer with a non-zero `current_end` would make later appends
        // silently drop their data (`copy_string_contents` / `bh_write_lowlevel_char`
        // no-op on a null pointer) while `ll_getlength` keeps counting it — a wrong
        // string instead of a failure. MemoryError propagation is not ported yet,
        // so fail loudly.
        assert!(
            new_string != 0,
            "StringBuilder grow failed to allocate {needed} bytes; MemoryError propagation is not ported yet",
        );
        let old_piece = alloc_piece(StringPieceBox {
            buf: old_buf,
            prev_piece: old_extra,
        });
        let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
        b.current_buf = new_string;
        b.current_pos = 0;
        b.current_end = needed as i64;
        b.total_size = new_total;
        b.extra_pieces = old_piece;
    }

    /// `rbuilder.py ll_grow_and_append`.
    fn ll_grow_and_append(builder: i64, ll_str: i64, start: i64, size: i64, item_size: usize) {
        let (current_pos, total_size, old_extra) = {
            let b = unsafe { &*(builder as *const StringBuilderBox) };
            (b.current_pos, b.total_size, b.extra_pieces)
        };
        if size > 1280
            && current_pos == 0
            && start == 0
            && size == bh_lowlevel_string_len(ll_str) as i64
            && let Some(total_size) = total_size.checked_add(size)
        {
            let old_piece = alloc_piece(StringPieceBox {
                buf: ll_str,
                prev_piece: old_extra,
            });
            let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
            b.total_size = total_size;
            b.extra_pieces = old_piece;
            return;
        }

        let (part1, cur_pos, cur_buf) = {
            let b = unsafe { &*(builder as *const StringBuilderBox) };
            (b.current_end - b.current_pos, b.current_pos, b.current_buf)
        };
        // First, the part that still fits in the current piece.
        copy_string_contents(
            ll_str,
            cur_buf,
            start as usize,
            cur_pos as usize,
            part1 as usize,
            item_size,
        );
        let start = start + part1;
        let size = size - part1;
        // Allocate the new piece, then copy the remainder into the fresh buffer.
        ll_grow_by(builder, size, item_size);
        let new_buf = {
            let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
            b.current_pos = size;
            b.current_buf
        };
        copy_string_contents(ll_str, new_buf, start as usize, 0, size as usize, item_size);
    }

    /// `rbuilder.py _ll_append`.
    pub fn ll_append(builder: i64, ll_str: i64, start: i64, size: i64, item_size: usize) {
        // STR-only — see [`ll_new`].
        assert_eq!(
            item_size, 1,
            "StringBuilder is STR-only until the destructors widen"
        );
        let (pos, end, cur_buf) = {
            let b = unsafe { &*(builder as *const StringBuilderBox) };
            (b.current_pos, b.current_end, b.current_buf)
        };
        if (end - pos) < size {
            ll_grow_and_append(builder, ll_str, start, size, item_size);
        } else {
            let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
            b.current_pos = pos + size;
            copy_string_contents(
                ll_str,
                cur_buf,
                start as usize,
                pos as usize,
                size as usize,
                item_size,
            );
        }
    }

    /// `rbuilder.py ll_append_char`.
    pub fn ll_append_char(builder: i64, char: i64, item_size: usize) {
        // STR-only — see [`ll_new`].
        assert_eq!(
            item_size, 1,
            "StringBuilder is STR-only until the destructors widen"
        );
        let full = {
            let b = unsafe { &*(builder as *const StringBuilderBox) };
            b.current_pos == b.current_end
        };
        if full {
            ll_grow_by(builder, 1, item_size);
        }
        let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
        let pos = b.current_pos;
        b.current_pos = pos + 1;
        bh_write_lowlevel_char(b.current_buf, pos as usize, char, item_size);
    }

    /// `rbuilder.py ll_getlength`.
    pub fn ll_getlength(builder: i64) -> i64 {
        let b = unsafe { &*(builder as *const StringBuilderBox) };
        b.total_size - (b.current_end - b.current_pos)
    }

    /// `rbuilder.py ll_shrink_final`.
    fn ll_shrink_final(builder: i64, item_size: usize) {
        let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
        let final_size = b.current_pos;
        // The old buffer remains collector-owned after replacement.
        b.current_buf = ll_shrink_array(b.current_buf, final_size as usize, item_size);
        b.current_end = final_size;
        b.total_size = final_size;
    }

    /// `rbuilder.py ll_fold_pieces`: concatenate `current_buf` and the
    /// `extra_pieces` chain (newest-first ⇒ filled back-to-front) into one buffer.
    fn ll_fold_pieces(builder: i64, item_size: usize) {
        let final_size = ll_getlength(builder);
        let (extra, current_pos, current_buf) = {
            let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
            let e = b.extra_pieces;
            b.extra_pieces = 0;
            (e, b.current_pos, b.current_buf)
        };
        let extra_box = unsafe { &*(extra as *const StringPieceBox) };
        if current_pos == 0 && extra_box.prev_piece == 0 {
            let piece = extra_box.buf;
            debug_assert_eq!(final_size, bh_lowlevel_string_len(piece) as i64);
            let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
            b.total_size = final_size;
            b.current_buf = piece;
            b.current_pos = final_size;
            b.current_end = final_size;
            return;
        }
        // Allocate the result and copy every piece back-to-front.
        let base = str_base_size(item_size);
        let result = bh_alloc_lowlevel_string(final_size as usize, base, item_size);
        {
            let b = unsafe { &mut *(builder as *mut StringBuilderBox) };
            b.total_size = final_size;
            b.current_buf = result;
            b.current_pos = final_size;
            b.current_end = final_size;
        }
        let mut piece = current_buf;
        let mut piece_lgt = current_pos;
        let mut extra = extra;
        let mut dst = final_size;
        loop {
            dst -= piece_lgt;
            copy_string_contents(
                piece,
                result,
                0,
                dst as usize,
                piece_lgt as usize,
                item_size,
            );
            if extra == 0 {
                break;
            }
            let eb = unsafe { &*(extra as *const StringPieceBox) };
            piece = eb.buf;
            piece_lgt = bh_lowlevel_string_len(piece) as i64;
            extra = eb.prev_piece;
        }
    }

    /// `rbuilder.py ll_build`: consolidate to a single buffer and return the
    /// builder's GC reference. The builder intentionally keeps the same reference.
    pub fn ll_build(builder: i64, item_size: usize) -> i64 {
        // STR-only — see [`ll_new`].
        assert_eq!(
            item_size, 1,
            "StringBuilder is STR-only until the destructors widen"
        );
        let (extra_pieces, current_pos, total_size) = {
            let b = unsafe { &*(builder as *const StringBuilderBox) };
            (b.extra_pieces, b.current_pos, b.total_size)
        };
        if extra_pieces != 0 {
            ll_fold_pieces(builder, item_size);
        } else if current_pos != total_size {
            ll_shrink_final(builder, item_size);
        }
        unsafe { (*(builder as *const StringBuilderBox)).current_buf }
    }

    /// `ll_append_res0(ll_builder, ll_str)` = `_ll_append(ll_builder, ll_str, 0,
    /// len(ll_str.chars))` (`rbuilder.py`). The `dont_look_inside` residual
    /// target the codewriter binds when an append cannot be inlined; mutates the
    /// builder in place and returns void.
    pub extern "C" fn jit_ll_append_res0(builder: i64, ll_str: i64) {
        let size = bh_lowlevel_string_len(ll_str) as i64;
        ll_append(builder, ll_str, 0, size, STR_ITEM_SIZE);
    }

    /// `ll_append_res_slice(ll_builder, ll_str, start, end)` =
    /// `_ll_append(ll_builder, ll_str, start, end - start)` (`rbuilder.py`). The
    /// slice helper takes `end`; `_ll_append` takes the count, so convert here.
    pub extern "C" fn jit_ll_append_res_slice(builder: i64, ll_str: i64, start: i64, end: i64) {
        ll_append(builder, ll_str, start, end - start, STR_ITEM_SIZE);
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::lowlevel_string::bh_read_lowlevel_string;

        fn make_str(bytes: &[u8]) -> i64 {
            let s = bh_alloc_lowlevel_string(bytes.len(), LOWLEVEL_STR_BASE_SIZE, STR_ITEM_SIZE);
            for (i, &b) in bytes.iter().enumerate() {
                bh_write_lowlevel_char(s, i, b as i64, STR_ITEM_SIZE);
            }
            s
        }

        fn read_str(s: i64) -> Vec<u8> {
            bh_read_lowlevel_string(s, STR_ITEM_SIZE)
                .into_iter()
                .map(|c| c as u8)
                .collect()
        }

        // These tests exercise the append/grow/build LOGIC + buffer arithmetic
        // over the real low-level-string primitives. The GC is not initialized in
        // a unit test, so `gc_alloc_storage_box` falls back to `malloc_raw` (no
        // sweep destructor) — GC tracing/reclamation of the builder + chain is
        // validated by the end-to-end fixture (task #50). STRINGPIECE nodes the
        // fold detaches therefore leak here; that is expected without a collector.

        #[test]
        fn ll_build_single_buffer_shrinks_and_roundtrips() {
            let item = STR_ITEM_SIZE;
            let builder = ll_new(100, item);
            let hello = make_str(b"hello");
            ll_append(builder, hello, 0, 5, item);
            assert_eq!(ll_getlength(builder), 5);
            // current_pos (5) != total_size (100) ⇒ ll_shrink_final.
            let result = ll_build(builder, item);
            assert_eq!(read_str(result), b"hello");
            assert_eq!(ll_getlength(builder), 5);
        }

        #[test]
        fn ll_build_across_grow_folds_pieces_in_order() {
            let item = STR_ITEM_SIZE;
            // init 4 forces a grow when appending 10 chars.
            let builder = ll_new(4, item);
            let s = make_str(b"abcdefghij");
            ll_append(builder, s, 0, 10, item);
            // Must have chained a piece.
            let extra = {
                let b = unsafe { &*(builder as *const StringBuilderBox) };
                b.extra_pieces
            };
            assert_ne!(extra, 0, "append across init size must chain a piece");
            ll_append_char(builder, b'!' as i64, item);
            assert_eq!(ll_getlength(builder), 11);
            // extra_pieces present ⇒ ll_fold_pieces.
            let result = ll_build(builder, item);
            assert_eq!(read_str(result), b"abcdefghij!");
            assert_eq!(ll_getlength(builder), 11);
        }

        #[test]
        fn ll_build_across_two_grows_folds_multi_piece_chain() {
            let item = STR_ITEM_SIZE;
            // init 2 + a 10-char then a 60-char append forces two grows, so the
            // fold walks a chain of two STRINGPIECE nodes plus the current buffer
            // (the single-grow test above only exercises one node).
            let builder = ll_new(2, item);
            let bytes: Vec<u8> = (0..70u8).map(|i| b'A' + (i % 26)).collect();
            let s = make_str(&bytes);
            ll_append(builder, s, 0, 10, item);
            ll_append(builder, s, 10, 60, item);
            // Two grows ⇒ a two-node chain: the head node's prev_piece is another
            // node, not the chain-end sentinel.
            let head = {
                let b = unsafe { &*(builder as *const StringBuilderBox) };
                b.extra_pieces
            };
            assert_ne!(head, 0, "two appends past init must chain pieces");
            let head_prev = unsafe { &*(head as *const StringPieceBox) }.prev_piece;
            assert_ne!(head_prev, 0, "two grows must chain at least two nodes");
            assert_eq!(ll_getlength(builder), 70);
            // extra_pieces present ⇒ ll_fold_pieces walks the whole chain.
            let result = ll_build(builder, item);
            assert_eq!(read_str(result), bytes);
        }

        #[test]
        fn ll_append_char_grows_when_full() {
            let item = STR_ITEM_SIZE;
            let builder = ll_new(2, item);
            for &c in b"xyz" {
                ll_append_char(builder, c as i64, item);
            }
            assert_eq!(ll_getlength(builder), 3);
            let result = ll_build(builder, item);
            assert_eq!(read_str(result), b"xyz");
        }

        #[test]
        fn grow_by_sizes_rounds_up_and_reports_overflow() {
            // (needed + total_size) rounded up to a multiple of 64, then
            // new_total = total_size + that.
            assert_eq!(grow_by_sizes(10, 4), Some((64, 68)));
            assert_eq!(grow_by_sizes(0, 0), Some((0, 0)));
            assert_eq!(grow_by_sizes(64, 0), Some((64, 64)));
            // Each of the three `ovfcheck` points overflows near i64::MAX.
            assert_eq!(grow_by_sizes(i64::MAX, 1), None); // needed + total_size
            assert_eq!(grow_by_sizes(i64::MAX - 10, 0), None); // + 63
            assert_eq!(grow_by_sizes(0, i64::MAX - 100), None); // total_size + needed
        }
    }
}
