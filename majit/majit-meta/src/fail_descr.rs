use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use majit_ir::{DescrRef, FailDescr, Type};

use crate::resume::ResumeData;

/// Global counter for unique fail_index allocation.
///
/// Mirrors RPython's ResumeGuardDescr numbering — each guard in every
/// compiled trace receives a unique fail_index so the backend can
/// report exactly which guard failed.
static NEXT_FAIL_INDEX: AtomicU32 = AtomicU32::new(1);

/// Reset the global fail_index counter (for testing).
#[cfg(test)]
pub fn reset_fail_index_counter() {
    NEXT_FAIL_INDEX.store(1, Ordering::SeqCst);
}

/// Allocate the next unique fail_index.
fn alloc_fail_index() -> u32 {
    NEXT_FAIL_INDEX.fetch_add(1, Ordering::SeqCst)
}

/// Per-guard FailDescr carrying a unique index and type information.
///
/// Mirrors RPython's ResumeGuardDescr — each guard operation gets its
/// own descriptor with a unique `fail_index` so the backend can identify
/// exactly which guard failed after execution.
#[derive(Debug)]
struct MetaFailDescr {
    fail_index: u32,
    types: Vec<Type>,
}

impl majit_ir::Descr for MetaFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for MetaFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.types
    }
}

/// Per-guard FailDescr that also carries resume data for deoptimization.
///
/// Mirrors RPython's ResumeGuardDescr with snapshot information.
/// When a guard fails, the backend uses the resume data to reconstruct
/// the interpreter state (virtual objects, frame variables, etc.).
#[derive(Debug)]
#[allow(dead_code)]
struct ResumeGuardDescr {
    fail_index: u32,
    types: Vec<Type>,
    resume_data: ResumeData,
}

impl majit_ir::Descr for ResumeGuardDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
}

impl FailDescr for ResumeGuardDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.types
    }
}

impl ResumeGuardDescr {
    /// Access the resume data for deoptimization.
    #[allow(dead_code)]
    pub fn resume_data(&self) -> &ResumeData {
        &self.resume_data
    }
}

/// Create a FailDescr for `num_live` integer values with an auto-assigned
/// unique fail_index.
///
/// Each call produces a distinct fail_index so the backend can identify
/// which guard failed.
pub fn make_fail_descr(num_live: usize) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index: alloc_fail_index(),
        types: vec![Type::Int; num_live],
    })
}

/// Create a FailDescr with an explicit fail_index.
///
/// Used when the caller needs to control the fail_index (e.g., for
/// bridge compilation where the fail_index must match the original guard).
#[allow(dead_code)]
pub fn make_fail_descr_with_index(fail_index: u32, num_live: usize) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index,
        types: vec![Type::Int; num_live],
    })
}

/// Create a FailDescr with explicit types and auto-assigned fail_index.
pub fn make_fail_descr_typed(types: Vec<Type>) -> DescrRef {
    Arc::new(MetaFailDescr {
        fail_index: alloc_fail_index(),
        types,
    })
}

/// Create a FailDescr with explicit types and an explicit fail_index.
pub fn make_fail_descr_typed_with_index(fail_index: u32, types: Vec<Type>) -> DescrRef {
    Arc::new(MetaFailDescr { fail_index, types })
}

/// Create a ResumeGuardDescr — a FailDescr that carries resume data
/// for reconstructing interpreter state on guard failure.
///
/// Mirrors RPython's ResumeGuardDescr which attaches snapshot data
/// to each guard so the meta-interpreter can reconstruct the full
/// interpreter state (including virtual objects) when deoptimizing.
#[allow(dead_code)]
pub fn make_resume_guard_descr(num_live: usize, resume_data: ResumeData) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index: alloc_fail_index(),
        types: vec![Type::Int; num_live],
        resume_data,
    })
}

/// Create a ResumeGuardDescr with an explicit fail_index.
#[allow(dead_code)]
pub fn make_resume_guard_descr_with_index(
    fail_index: u32,
    num_live: usize,
    resume_data: ResumeData,
) -> DescrRef {
    Arc::new(ResumeGuardDescr {
        fail_index,
        types: vec![Type::Int; num_live],
        resume_data,
    })
}

/// Extract resume data from a guard's FailDescr + MetaInterp's resume_data map.
///
/// The recommended pattern for resume data lookup:
/// 1. The guard's FailDescr carries a unique `fail_index`
/// 2. The MetaInterp stores `ResumeData` in a `HashMap<u32, ResumeData>`
///    keyed by `fail_index`
/// 3. On guard failure, look up `fail_index` in the map
///
/// This matches RPython's approach where `ResumeGuardDescr` points to
/// snapshot data stored alongside the compiled loop.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fail_descr_unique_indices() {
        reset_fail_index_counter();
        let d1 = make_fail_descr(2);
        let d2 = make_fail_descr(3);
        let d3 = make_fail_descr(1);

        let fi1 = d1.as_fail_descr().unwrap().fail_index();
        let fi2 = d2.as_fail_descr().unwrap().fail_index();
        let fi3 = d3.as_fail_descr().unwrap().fail_index();

        // All indices must be unique
        assert_ne!(fi1, fi2);
        assert_ne!(fi2, fi3);
        assert_ne!(fi1, fi3);
    }

    #[test]
    fn test_fail_descr_with_explicit_index() {
        let d = make_fail_descr_with_index(42, 3);
        assert_eq!(d.as_fail_descr().unwrap().fail_index(), 42);
        assert_eq!(d.as_fail_descr().unwrap().fail_arg_types().len(), 3);
    }

    #[test]
    fn test_fail_descr_typed() {
        let types = vec![Type::Int, Type::Ref, Type::Float];
        let d = make_fail_descr_typed(types.clone());
        assert_eq!(d.as_fail_descr().unwrap().fail_arg_types(), &types);
    }

    #[test]
    fn test_resume_guard_descr() {
        let rd = ResumeData::simple(42, 3);
        let d = make_resume_guard_descr(3, rd);

        let fd = d.as_fail_descr().unwrap();
        assert!(fd.fail_index() > 0);
        assert_eq!(fd.fail_arg_types().len(), 3);
    }

    #[test]
    fn test_resume_guard_descr_with_explicit_index() {
        let rd = ResumeData::simple(99, 2);
        let d = make_resume_guard_descr_with_index(7, 2, rd);

        let fd = d.as_fail_descr().unwrap();
        assert_eq!(fd.fail_index(), 7);
        assert_eq!(fd.fail_arg_types().len(), 2);
    }
}
