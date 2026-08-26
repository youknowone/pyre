//! One declaration, two front ends.
//!
//! `#[jit_immutable_fields]` was harvested only out of Charon's `global_decls`,
//! which the proc-macro path never produces.  Rather than add a second place to
//! write the list, the attribute publishes the same string twice: the marker
//! const the LLBC harvester reads, and an inherent associated const the layout
//! emit sites read.
//!
//! The second one has to answer for a struct that never declared anything —
//! an emit site names its struct without knowing — so the reads below cover
//! both halves.  Reading through the trait explicitly is deliberately *not*
//! tested as equivalent: it is the spelling generated code must avoid.

use majit_metainterp::MajitImmutableFields as _;

#[majit_macros::jit_immutable_fields("left", "right", "kind", "ch")]
#[repr(C)]
struct Declared {
    kind: u8,
    ch: u8,
    marked: u8,
    left: usize,
    right: usize,
}

#[repr(C)]
struct OptedOut {
    x: usize,
}

#[test]
fn a_declaring_struct_exposes_its_ranks_by_inherent_const() {
    assert_eq!(
        <Declared>::__MAJIT_IMMUTABLE_FIELDS,
        "left,right,kind,ch",
        "the inherent const must shadow the blanket-trait default",
    );
}

#[test]
fn a_struct_that_never_declared_falls_back_to_empty() {
    assert_eq!(
        <OptedOut>::__MAJIT_IMMUTABLE_FIELDS,
        "",
        "an emit site names every struct it registers a layout for, including \
         ones that opted out; the read must resolve rather than fail to compile",
    );
}

#[test]
fn the_marker_const_the_llbc_harvester_reads_still_exists() {
    // `harvest_immutable_fields_from_llbcs` reads this one.  Publishing the
    // declaration a second way must not disturb the first.
    assert_eq!(_immutable_fields_Declared, "left,right,kind,ch");
}

#[test]
fn the_two_publications_carry_the_same_list() {
    assert_eq!(
        _immutable_fields_Declared,
        <Declared>::__MAJIT_IMMUTABLE_FIELDS
    );
}

#[test]
fn the_struct_definition_is_left_intact() {
    // The attribute documents that it does not rewrite the struct.  A field
    // list that survived expansion is what every downstream `offset_of!` in a
    // layout emit site depends on.
    let node = Declared {
        kind: 0,
        ch: b'a',
        marked: 1,
        left: 0,
        right: 0,
    };
    assert_eq!(node.kind, 0);
    assert_eq!(node.ch, b'a');
    assert_eq!(node.marked, 1);
    assert_eq!(node.left, 0);
    assert_eq!(node.right, 0);
}
