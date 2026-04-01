//! Visitor trait for walking virtual object structures.
//!
//! Provides consistent traversal of all virtual object types during
//! resume data construction, unrolling, and other analyses.
//!
//! Translated from rpython/jit/metainterp/walkvirtual.py.

use majit_ir::OpRef;

use crate::optimizeopt::info::{
    PtrInfo, VirtualArrayInfo, VirtualArrayStructInfo, VirtualInfo, VirtualRawBufferInfo,
    VirtualStructInfo, VirtualizableFieldState,
};

/// Visitor for virtual object structures.
///
/// Implement this trait to process all virtual object types uniformly.
/// Default implementations do nothing, so visitors can override only
/// the types they care about.
pub trait VirtualVisitor {
    /// Visit a virtual object (from NEW_WITH_VTABLE).
    fn visit_virtual(&mut self, _opref: OpRef, _info: &VirtualInfo) {}

    /// Visit a virtual array (from NEW_ARRAY).
    fn visit_varray(&mut self, _opref: OpRef, _info: &VirtualArrayInfo) {}

    /// Visit a virtual struct without vtable (from NEW).
    fn visit_vstruct(&mut self, _opref: OpRef, _info: &VirtualStructInfo) {}

    /// Visit a virtual array of structs (interior field access).
    fn visit_varraystruct(&mut self, _opref: OpRef, _info: &VirtualArrayStructInfo) {}

    /// Visit a virtual raw buffer.
    fn visit_vrawbuffer(&mut self, _opref: OpRef, _info: &VirtualRawBufferInfo) {}

    /// Visit a virtualizable object (interpreter frame).
    fn visit_virtualizable(&mut self, _opref: OpRef, _info: &VirtualizableFieldState) {}
}

/// Walk all virtuals in a `PtrInfo` slice and dispatch to the visitor.
///
/// Each entry is `(opref, ptr_info)`. Non-virtual entries (NonNull,
/// Constant, KnownClass, Virtualizable) are skipped.
pub fn walk_virtuals(virtuals: &[(OpRef, PtrInfo)], visitor: &mut impl VirtualVisitor) {
    for (opref, info) in virtuals {
        match info {
            PtrInfo::Virtual(v) => visitor.visit_virtual(*opref, v),
            PtrInfo::VirtualArray(v) => visitor.visit_varray(*opref, v),
            PtrInfo::VirtualStruct(v) => visitor.visit_vstruct(*opref, v),
            PtrInfo::VirtualArrayStruct(v) => visitor.visit_varraystruct(*opref, v),
            PtrInfo::VirtualRawBuffer(v) => visitor.visit_vrawbuffer(*opref, v),
            PtrInfo::Virtualizable(v) => visitor.visit_virtualizable(*opref, v),
            PtrInfo::NonNull { .. }
            | PtrInfo::Constant(_)
            | PtrInfo::KnownClass { .. }
            | PtrInfo::Instance(_)
            | PtrInfo::Struct(_)
            | PtrInfo::Array(_)
            | PtrInfo::Str(_) => {}
        }
    }
}

/// walkvirtual.py: walk_virtuals_and_count — count total number of virtual items.
pub fn count_virtual_items(virtuals: &[(OpRef, PtrInfo)]) -> usize {
    let mut count = 0;
    for (_, info) in virtuals {
        count += info.num_fields();
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use majit_ir::{Descr, DescrRef, GcRef, OpRef};

    #[derive(Debug)]
    struct TestDescr;
    impl Descr for TestDescr {}

    /// Counting visitor for testing.
    #[derive(Default)]
    struct CountingVisitor {
        virtual_count: usize,
        varray_count: usize,
        vstruct_count: usize,
        varraystruct_count: usize,
        vrawbuffer_count: usize,
    }

    impl VirtualVisitor for CountingVisitor {
        fn visit_virtual(&mut self, _opref: OpRef, _info: &VirtualInfo) {
            self.virtual_count += 1;
        }
        fn visit_varray(&mut self, _opref: OpRef, _info: &VirtualArrayInfo) {
            self.varray_count += 1;
        }
        fn visit_vstruct(&mut self, _opref: OpRef, _info: &VirtualStructInfo) {
            self.vstruct_count += 1;
        }
        fn visit_varraystruct(&mut self, _opref: OpRef, _info: &VirtualArrayStructInfo) {
            self.varraystruct_count += 1;
        }
        fn visit_vrawbuffer(&mut self, _opref: OpRef, _info: &VirtualRawBufferInfo) {
            self.vrawbuffer_count += 1;
        }
    }

    /// Collector visitor that records field OpRefs from all virtual types.
    #[derive(Default)]
    struct FieldCollector {
        collected: Vec<OpRef>,
    }

    impl VirtualVisitor for FieldCollector {
        fn visit_virtual(&mut self, _opref: OpRef, info: &VirtualInfo) {
            for (_, field_ref) in &info.fields {
                self.collected.push(*field_ref);
            }
        }
        fn visit_varray(&mut self, _opref: OpRef, info: &VirtualArrayInfo) {
            for item in &info.items {
                self.collected.push(*item);
            }
        }
        fn visit_vstruct(&mut self, _opref: OpRef, info: &VirtualStructInfo) {
            for (_, field_ref) in &info.fields {
                self.collected.push(*field_ref);
            }
        }
        fn visit_vrawbuffer(&mut self, _opref: OpRef, info: &VirtualRawBufferInfo) {
            for (_, _, val_ref) in &info.entries {
                self.collected.push(*val_ref);
            }
        }
    }

    fn descr_ref() -> DescrRef {
        Arc::new(TestDescr)
    }

    #[test]
    fn walk_virtual_counts_all_types() {
        let virtuals: Vec<(OpRef, PtrInfo)> = vec![
            (
                OpRef(0),
                PtrInfo::Virtual(VirtualInfo {
                    descr: descr_ref(),
                    known_class: None,
                    fields: vec![(0, OpRef(10))],
                    field_descrs: Vec::new(),
                    last_guard_pos: -1,
                }),
            ),
            (
                OpRef(1),
                PtrInfo::VirtualArray(VirtualArrayInfo {
                    descr: descr_ref(),
                    clear: false,
                    items: vec![OpRef(20), OpRef(21)],
                    last_guard_pos: -1,
                }),
            ),
            (
                OpRef(2),
                PtrInfo::VirtualStruct(VirtualStructInfo {
                    descr: descr_ref(),
                    fields: vec![(0, OpRef(30))],
                    field_descrs: Vec::new(),
                    last_guard_pos: -1,
                }),
            ),
            (
                OpRef(3),
                PtrInfo::VirtualArrayStruct(VirtualArrayStructInfo {
                    descr: descr_ref(),
                    element_fields: vec![vec![(0, OpRef(40))]],
                    last_guard_pos: -1,
                }),
            ),
            (
                OpRef(4),
                PtrInfo::VirtualRawBuffer(VirtualRawBufferInfo {
                    size: 16,
                    entries: vec![(0, 8, OpRef(50))],
                    last_guard_pos: -1,
                }),
            ),
            // Non-virtual entries should be skipped.
            (OpRef(5), PtrInfo::nonnull()),
            (OpRef(6), PtrInfo::Constant(GcRef::NULL)),
            (
                OpRef(7),
                PtrInfo::KnownClass {
                    class_ptr: GcRef::NULL,
                    is_nonnull: true,
                    last_guard_pos: -1,
                },
            ),
        ];

        let mut visitor = CountingVisitor::default();
        walk_virtuals(&virtuals, &mut visitor);

        assert_eq!(visitor.virtual_count, 1);
        assert_eq!(visitor.varray_count, 1);
        assert_eq!(visitor.vstruct_count, 1);
        assert_eq!(visitor.varraystruct_count, 1);
        assert_eq!(visitor.vrawbuffer_count, 1);
    }

    #[test]
    fn walk_empty_list() {
        let virtuals: Vec<(OpRef, PtrInfo)> = vec![];
        let mut visitor = CountingVisitor::default();
        walk_virtuals(&virtuals, &mut visitor);

        assert_eq!(visitor.virtual_count, 0);
        assert_eq!(visitor.varray_count, 0);
    }

    #[test]
    fn walk_collects_fields() {
        let virtuals: Vec<(OpRef, PtrInfo)> = vec![
            (
                OpRef(0),
                PtrInfo::Virtual(VirtualInfo {
                    descr: descr_ref(),
                    known_class: None,
                    fields: vec![(0, OpRef(10)), (1, OpRef(11))],
                    field_descrs: Vec::new(),
                    last_guard_pos: -1,
                }),
            ),
            (
                OpRef(1),
                PtrInfo::VirtualArray(VirtualArrayInfo {
                    descr: descr_ref(),
                    clear: false,
                    items: vec![OpRef(20)],
                    last_guard_pos: -1,
                }),
            ),
            (
                OpRef(2),
                PtrInfo::VirtualRawBuffer(VirtualRawBufferInfo {
                    size: 16,
                    entries: vec![(0, 8, OpRef(30)), (8, 8, OpRef(31))],
                    last_guard_pos: -1,
                }),
            ),
        ];

        let mut collector = FieldCollector::default();
        walk_virtuals(&virtuals, &mut collector);

        assert_eq!(
            collector.collected,
            vec![OpRef(10), OpRef(11), OpRef(20), OpRef(30), OpRef(31)]
        );
    }

    #[test]
    fn walk_skips_non_virtual() {
        let virtuals: Vec<(OpRef, PtrInfo)> = vec![
            (OpRef(0), PtrInfo::nonnull()),
            (OpRef(1), PtrInfo::Constant(GcRef::NULL)),
            (
                OpRef(2),
                PtrInfo::KnownClass {
                    class_ptr: GcRef::NULL,
                    is_nonnull: false,
                    last_guard_pos: -1,
                },
            ),
        ];

        let mut visitor = CountingVisitor::default();
        walk_virtuals(&virtuals, &mut visitor);

        assert_eq!(visitor.virtual_count, 0);
        assert_eq!(visitor.varray_count, 0);
        assert_eq!(visitor.vstruct_count, 0);
        assert_eq!(visitor.varraystruct_count, 0);
        assert_eq!(visitor.vrawbuffer_count, 0);
    }
}
