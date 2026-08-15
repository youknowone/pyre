//! Two emit sites may describe one word differently, and nothing said so.
//!
//! `register_struct_layout` accumulates a struct's layout across the sites that
//! touch it and dedups on the byte offset. An offset does not identify a field
//! — a flattened inline aggregate and its first leaf share an address — so a
//! second registration at a known offset is one of three things and only the
//! first is benign: the same field again, a different field that happens to sit
//! there, or the same field described differently.
//!
//! The first is the common case: every site re-registers what it accesses, so
//! the same few type ids arrive hundreds of times. The other two were dropped
//! on the floor, and both are silent in a way that outlives the build. A
//! dropped sibling leaves the spec short an entry that `add_struct_field_descr`
//! resolves `index_in_parent` and `name` against; a redescribed word gets the
//! GC-pointer flag of whichever site emitted first, which is a property of
//! emission order rather than of the struct.
//!
//! No compile-time witness can catch the second one. Stable Rust has no
//! negative trait bound, so a declaration can state that a field it *names* is
//! a pointer but not that a field it omits is not — `ref_field_witness_tokens`
//! covers exactly the declared half. This is the mechanism for the other half,
//! and it works by disagreement rather than by declaration.

use majit_metainterp::{JitCodeBuilder, StructLayoutConflictKind};

/// Distinct per test: the registry is process-global, so two tests sharing a
/// type id would read each other's conflicts.
const TID_AGREE: u64 = 0x4147_5245_4531;
const TID_REDESCRIBED: u64 = 0x5245_4445_5343;
const TID_SIBLING: u64 = 0x5349_424C_494E;

fn conflicts_for(type_id: u64) -> Vec<majit_metainterp::StructLayoutConflict> {
    majit_metainterp::struct_layout_conflicts()
        .into_iter()
        .filter(|c| c.type_id == type_id)
        .collect()
}

/// The negative control, and it has to come with its own denominator.
///
/// "No conflict was reported" is what a check that never ran also produces, and
/// a field at an offset nobody registered yet is appended without comparing
/// anything. So this asserts the comparison count moved: the re-registration
/// below really did meet the first one.
#[test]
fn a_site_re_registering_what_it_already_declared_is_not_a_conflict() {
    let before = majit_metainterp::struct_layout_comparisons();
    let mut builder = JitCodeBuilder::new();
    let layout = &[(0, true, "head", 8, false), (8, false, "size", 8, true)][..];
    // Two access sites on one struct, each re-declaring the whole layout —
    // the shape `#[jit_interp]` emits, one call per getfield/setfield.
    builder.register_struct_layout(16, TID_AGREE, false, false, layout);
    builder.register_struct_layout(16, TID_AGREE, false, false, layout);
    let _ = builder.finish();

    assert!(
        majit_metainterp::struct_layout_comparisons() >= before + 2,
        "the second registration must have been compared against the first, or \
         an empty conflict list below is vacuous",
    );
    assert_eq!(
        conflicts_for(TID_AGREE),
        vec![],
        "identical re-registration is the common case and must stay silent",
    );
}

/// A pointer word registered as a scalar by a second site.
///
/// This is what a field left out of one declaration's ref-field map produces.
/// Both sites compile, both emit, and the collector traces through the word or
/// does not depending on which site the assembler saw first.
#[test]
fn one_word_declared_a_pointer_and_a_scalar_is_a_conflict() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(
        16,
        TID_REDESCRIBED,
        false,
        false,
        &[(0, true, "head", 8, false)],
    );
    builder.register_struct_layout(
        16,
        TID_REDESCRIBED,
        false,
        false,
        &[(0, false, "head", 8, true)],
    );
    let _ = builder.finish();

    let conflicts = conflicts_for(TID_REDESCRIBED);
    assert_eq!(conflicts.len(), 1, "one disagreement: {conflicts:#?}");
    let conflict = &conflicts[0];
    assert_eq!(conflict.kind, StructLayoutConflictKind::Redescribed);
    assert_eq!(conflict.offset, 0);
    assert!(
        conflict.kept.is_ref && !conflict.dropped.is_ref,
        "the first registration is what the spec keeps: {conflict:#?}",
    );
}

/// A real sibling at an occupied offset, which the offset-keyed dedup discards.
///
/// Registering both in ONE call keeps both — `field_specs_from_layout` does not
/// dedup within a call, which is how a flattened aggregate and its first leaf
/// both survive. Split across two calls, the second is dropped, and this is the
/// only place that says so.
#[test]
fn a_sibling_at_an_occupied_offset_is_reported_as_dropped() {
    let mut builder = JitCodeBuilder::new();
    builder.register_struct_layout(24, TID_SIBLING, false, false, &[(8, false, "agg", 8, true)]);
    builder.register_struct_layout(
        24,
        TID_SIBLING,
        false,
        false,
        &[(8, true, "leaf", 8, false)],
    );
    let jitcode = builder.finish();
    let _ = jitcode;

    let conflicts = conflicts_for(TID_SIBLING);
    assert_eq!(conflicts.len(), 1, "one dropped sibling: {conflicts:#?}");
    let conflict = &conflicts[0];
    assert_eq!(conflict.kind, StructLayoutConflictKind::DroppedSibling);
    assert_eq!(conflict.kept.name, "agg");
    assert_eq!(conflict.dropped.name, "leaf");
}

/// The gate's own two failures must read differently.
///
/// `assert_no_struct_layout_conflicts` is what a consumer calls, and it fails
/// on an empty registry as well as on a populated one — an empty registry is
/// also what a process that built no jitcodes produces. This test runs last in
/// the file's order by name, but order is not what makes it correct: it asserts
/// on the message, and the un-exercised message can only appear before any
/// comparison has happened anywhere in the process.
#[test]
fn the_gate_reports_a_conflict_rather_than_passing_on_it() {
    let mut builder = JitCodeBuilder::new();
    const TID: u64 = 0x4741_5445_4441;
    builder.register_struct_layout(16, TID, false, false, &[(0, true, "head", 8, false)]);
    builder.register_struct_layout(16, TID, false, false, &[(0, false, "head", 8, true)]);
    let _ = builder.finish();

    let failure = std::panic::catch_unwind(majit_metainterp::assert_no_struct_layout_conflicts)
        .expect_err("this fixture registers one word two ways on purpose");
    let message = failure
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| failure.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default();
    assert!(
        message.contains("disagreed with the layout already recorded"),
        "the gate must fail on the disagreement, not on its denominator; \
         got {message:?}",
    );
}
