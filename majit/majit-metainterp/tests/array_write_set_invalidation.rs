//! Array-element write-sets must be visible on the macro / `JitDriver` path.
//!
//! `residual_writes` restores the heapcache invalidation an inline
//! `setfield_gc` would have performed, for residuals that mutate a host struct
//! through opaque code.  These tests pin the array-element half of that
//! contract, which the node-chain storage layout never exercised: an element
//! used to live at `Node.value`, reached through a base that was a different
//! node ref on every iteration, so invalidating the `head` field handed the
//! next load a fresh base and a guaranteed cache miss.  With a buffer the base
//! is the SAME pointer across the residual, so a missing array invalidation is
//! a stale read on the common path rather than a latent one.
//!
//! `OptHeap::force_from_effectinfo` resolves a cached container's bitstring
//! slot through `descr.get_ei_index()`.  On the `#[jit_interp]` path
//! `compute_bitstrings` never runs, so that slot is the unset `u32::MAX`
//! sentinel and every bitstring lookup misses.  The FIELD loop already covers
//! this with a raw-set identity fallback gated on `!compute_bitstrings_has_run()`
//! (`optimizeopt/heap.rs`, `writes_field_descr_by_identity`).  The ARRAY loop
//! has no counterpart, which is what these tests hold it to.

use majit_ir::Type;
use majit_ir::descr::make_array_descr_from_lltype_shape;
use majit_ir::{DescrRef, EffectInfo, ExtraEffect, OopSpecIndex};

/// The shape `Assembler::add_raw_int_array_descr(item_size, is_item_signed)`
/// interns for an `array_fields` element access: a header-less buffer pointer
/// (`base_size = 0`, no length word, not GC-managed) over `item_size` items.
fn mint_element_array_descr_of(item_size: usize, is_item_signed: bool) -> DescrRef {
    make_array_descr_from_lltype_shape(
        /* type_id */ 0,
        /* base_size */ 0,
        item_size,
        /* len_offset */ None,
        Type::Int,
        /* is_array_of_pointers */ false,
        /* is_array_of_structs */ false,
        is_item_signed,
        /* is_gc_managed */ false,
        /* lendescr */ None,
        /* is_pure */ false,
        /* ei_index */ u32::MAX,
        /* interior_field_descrs */ Vec::new(),
    )
}

/// The signed 8-byte element array the node-chain storage reads.
fn mint_element_array_descr() -> DescrRef {
    mint_element_array_descr_of(8, true)
}

/// An `EffectInfo` shaped the way a `residual_writes`-declared residual's would
/// be once the array vocabulary exists: it names `descr` in its raw array write
/// set and nothing else.
fn ei_writing_array(descr: &DescrRef) -> EffectInfo {
    let mut ei = EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);
    ei._write_descrs_arrays = Some(vec![descr.clone()]);
    ei
}

/// Precondition for the test below, asserted separately so a failure there is
/// not misread as the defect under test: on this path `compute_bitstrings` has
/// not run, so no descr carries a real bitstring slot.
#[test]
fn macro_path_leaves_the_array_descr_at_the_unset_sentinel() {
    let descr = mint_element_array_descr();
    assert!(
        !majit_ir::effectinfo::compute_bitstrings_has_run(),
        "these tests characterise the `#[jit_interp]` path, on which \
         `compute_bitstrings` never runs"
    );
    assert_eq!(
        descr.get_ei_index(),
        u32::MAX,
        "an element array descr should carry the unset slot on this path"
    );
}

/// The defect: an EI that names an array descr in its raw write set is
/// invisible to `force_from_effectinfo`, because the array loop consults only
/// the bitstring and the bitstring slot is the sentinel.
///
/// The two sides of the `assert_eq!` are deliberate — `named_by_identity` is
/// what the EI actually declares, `visible_to_optimizer` is what the array loop
/// in `force_from_effectinfo` can currently see. They must agree.
///
/// When the array-side identity fallback lands, `visible_to_optimizer` becomes
/// a call to the real predicate (the array twin of
/// `EffectInfo::writes_field_descr_by_identity`) rather than the bitstring
/// check alone, and this test turns green without its subject changing.
#[test]
fn array_write_set_is_visible_to_force_from_effectinfo() {
    let descr = mint_element_array_descr();
    let ei = ei_writing_array(&descr);

    let named_by_identity = ei
        ._write_descrs_arrays
        .as_deref()
        .is_some_and(|set| set.iter().any(|d| std::sync::Arc::ptr_eq(d, &descr)));
    assert!(
        named_by_identity,
        "test setup: the EI must name the descr in its raw array write set"
    );

    // Exactly what `OptHeap::force_from_effectinfo`'s array loop evaluates.
    let visible_to_optimizer = ei.check_write_descr_array(descr.get_ei_index())
        || (!majit_ir::effectinfo::compute_bitstrings_has_run()
            && ei.writes_array_descr_by_shape(&descr));

    assert_eq!(
        named_by_identity, visible_to_optimizer,
        "a residual that declares it writes this array must invalidate the \
         cached element load. The field loop recovers exactly this case via \
         `writes_field_descr_by_identity` when `compute_bitstrings` has not \
         run; the array loop has no such fallback, so the declared write is \
         silently dropped and the trace keeps serving a stale \
         `getarrayitem_gc_i` from before the call"
    );
}

/// The shape the opcode-fetch path interns: `program: &[u8]`, one-byte
/// unsigned items. Signedness matters — a signed byte descr makes the backend
/// emit `movsx` and corrupt dispatch — so this is a genuinely distinct array.
fn mint_program_byte_array_descr() -> DescrRef {
    make_array_descr_from_lltype_shape(
        0,
        0,
        1,
        None,
        Type::Int,
        false,
        false,
        /* is_item_signed */ false,
        true,
        None,
        false,
        u32::MAX,
        Vec::new(),
    )
}

/// The write set is built during jitcode assembly and the cached read descr is
/// minted at trace time from a different cache, so the two are NEVER the same
/// `Arc`. Matching has to survive that, which is why the array predicate keys
/// on shape rather than pointer identity.
#[test]
fn a_distinct_arc_of_the_same_shape_still_matches() {
    let cached = mint_element_array_descr();
    let declared = mint_element_array_descr();
    assert!(
        !std::sync::Arc::ptr_eq(&cached, &declared),
        "test setup: these must be distinct Arcs, or this proves nothing"
    );

    let ei = ei_writing_array(&declared);
    assert!(
        ei.writes_array_descr_by_shape(&cached),
        "a residual's declared array write must reach the cached read even \
         though the two descrs were minted separately"
    );
}

/// …and it must not match everything, or the predicate is vacuous and the test
/// above passes for the wrong reason. A write to the `Val` element array must
/// not invalidate the opcode-fetch byte array.
#[test]
fn a_different_element_shape_does_not_match() {
    let ei = ei_writing_array(&mint_element_array_descr());
    assert!(
        !ei.writes_array_descr_by_shape(&mint_program_byte_array_descr()),
        "declaring a write to the 8-byte signed element array must not \
         invalidate the 1-byte unsigned opcode array"
    );
}

/// The macro's emitted expression, evaluated directly: a `field[]` declaration
/// lowers to one `(size_of::<Elem>(), true)` array spec, and the descr
/// `residual_write_effect_info` mints from it must carry the shape the reads
/// intern.  This is the join between the two halves — the macro picks the
/// numbers, this function turns them into a descr, and the shape predicate
/// reads it back.
#[test]
fn the_emitted_array_spec_mints_the_shape_the_reads_intern() {
    let ei = majit_metainterp::residual_write_effect_info(
        &[],
        // What `residual_write_effect_info_tokens` emits for a `Val` element
        // array: `(::core::mem::size_of::<Val>(), true)`.
        &[(8, true)],
        true,
    );
    assert!(
        ei.writes_array_descr_by_shape(&mint_element_array_descr()),
        "the descr minted from the emitted spec must match the one \
         `add_gc_int_array_descr(8, true)` interns for the read"
    );
    assert!(
        !ei.writes_array_descr_by_shape(&mint_program_byte_array_descr()),
        "and must not reach an array of a different element shape"
    );
}

/// Signedness is part of the shape key, so a `u8` element declaration has to
/// mint an UNSIGNED write descr — the same value the read derives from
/// `(<u8>::MIN as i128) < 0`. The write spec used to be a hard-coded `true`,
/// which matched only because the read hard-coded `true` as well; deriving the
/// read alone leaves the two halves disagreeing and the declaration inert.
///
/// The second assertion is what makes this test able to fail: it evaluates the
/// old spelling on the same input and requires it to miss.
#[test]
fn an_unsigned_element_declaration_mints_an_unsigned_write_descr() {
    let unsigned_read = mint_element_array_descr_of(1, false);

    let derived = majit_metainterp::residual_write_effect_info(&[], &[(1, false)], true);
    assert!(
        derived.writes_array_descr_by_shape(&unsigned_read),
        "a `u8` element write declaration must invalidate the `u8` element \
         load the same declaration's reads intern"
    );

    let hard_coded_signed = majit_metainterp::residual_write_effect_info(&[], &[(1, true)], true);
    assert!(
        !hard_coded_signed.writes_array_descr_by_shape(&unsigned_read),
        "control: a signed write spec must NOT reach the unsigned read, or \
         this test could not have caught the mismatch it exists for"
    );
}

/// A fields-only caller must be unaffected by the array half existing: an empty
/// `arrays` has to leave the same affirmative "writes no arrays" that
/// `EffectInfo::const_new` installs, not an unknown-writes `None`.
#[test]
fn an_empty_array_spec_list_still_declares_that_nothing_is_written() {
    let ei = majit_metainterp::residual_write_effect_info(&[], &[], true);
    assert_eq!(
        ei._write_descrs_arrays.as_deref().map(<[_]>::len),
        Some(0),
        "an empty array spec list must stay an affirmative empty write set"
    );
    assert!(
        !ei.writes_array_descr_by_shape(&mint_element_array_descr()),
        "a residual that declares no array writes must not invalidate one"
    );
}

/// The field half of the same contract, as the control: it must already pass.
/// If this ever goes red the fallback mechanism itself moved, and the array
/// test above is measuring something other than the missing array twin.
#[test]
fn field_write_set_identity_fallback_is_the_working_control() {
    let descr = mint_element_array_descr();
    let mut ei = EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);
    ei._write_descrs_fields = Some(vec![descr.clone()]);

    assert!(
        ei.writes_field_descr_by_identity(&descr),
        "the field raw-set identity probe is the precedent the array side must \
         mirror; if this fails the precedent is gone"
    );
}
