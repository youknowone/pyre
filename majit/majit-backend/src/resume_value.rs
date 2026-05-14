//! Resume-value tagged sources used by `ResumeData` (and ultimately
//! `ResumeGuardDescr`) — moved here from `majit-metainterp::resume` as
//! part of the Phase C-1 cascade toward backend struct deletion.
//!
//! `compile.py:855 AbstractResumeGuardDescr._attrs_` resume payload
//! references these tags through `rd_virtuals` / pending field
//! sources; placing them in a backend-accessible crate lets the
//! eventual `ResumeGuardDescr` definition live where the backend
//! codegen can instantiate it directly.

use majit_ir::resumedata::{ResumeValueKind, ResumeValueLayoutSummary};
use majit_ir::{Const, Type};

use crate::ExitValueSourceLayout;

/// Tagged source for a value that must be reconstructed on resume.
///
/// This is the majit equivalent of the tagged numbering used by
/// `rpython/jit/metainterp/resume.py`. Each `Constant` entry carries a
/// full `majit_ir::Const` (Int/Float/Ref) so the encoder's `getconst`
/// dispatch (resume.py:157-188) can route through the shared pool
/// (`ResumeDataLoopMemo.consts`) without losing type information.
#[derive(Debug, Clone, PartialEq)]
pub enum ResumeValueSource {
    /// Value comes from the deadframe fail-args array.
    FailArg(usize),
    /// Value is a compile-time constant — carries the Const so that the
    /// type survives encoding, matching RPython's `Const` object.
    Constant(Const),
    /// Value is a virtual object that must be materialized on resume.
    Virtual(usize),
    /// Value exists conceptually but is still uninitialized.
    ///
    /// Mirrors RPython's `UNINITIALIZED` tag used for string/unicode content.
    Uninitialized,
    /// Slot is not live at this guard.
    Unavailable,
}

impl ResumeValueSource {
    pub fn kind(&self) -> ResumeValueKind {
        match self {
            ResumeValueSource::FailArg(_) => ResumeValueKind::FailArg,
            ResumeValueSource::Constant(_) => ResumeValueKind::Constant,
            ResumeValueSource::Virtual(_) => ResumeValueKind::Virtual,
            ResumeValueSource::Uninitialized => ResumeValueKind::Uninitialized,
            ResumeValueSource::Unavailable => ResumeValueKind::Unavailable,
        }
    }

    pub fn layout_summary(&self) -> ResumeValueLayoutSummary {
        match self {
            ResumeValueSource::FailArg(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::FailArg,
                fail_arg_index: *index,
                raw_fail_arg_position: Some(*index),
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ResumeValueSource::Constant(c) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Constant,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: Some(c.as_raw_i64()),
                constant_type: Some(c.get_type()),
                virtual_index: None,
            },
            ResumeValueSource::Virtual(index) => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Virtual,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: Some(*index),
            },
            ResumeValueSource::Uninitialized => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Uninitialized,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
            ResumeValueSource::Unavailable => ResumeValueLayoutSummary {
                kind: ResumeValueKind::Unavailable,
                fail_arg_index: 0,
                raw_fail_arg_position: None,
                constant: None,
                constant_type: None,
                virtual_index: None,
            },
        }
    }
}

/// Cross-crate impl methods for `ResumeValueLayoutSummary` — these
/// reference both `ResumeValueSource` (this module) and
/// `ExitValueSourceLayout` (also in `majit-backend`), so they live
/// alongside the moved enum.
pub trait ResumeValueLayoutSummaryExt {
    /// `resume.py:226` raw fail-arg position lookup — falls back to
    /// `fail_arg_index` when the explicit `raw_fail_arg_position`
    /// override is absent.
    fn raw_fail_arg_position_or_fallback(&self) -> usize;
    fn to_resume_source(&self) -> ResumeValueSource;
    fn to_exit_source(&self, virtual_offset: usize) -> ExitValueSourceLayout;
}

impl ResumeValueLayoutSummaryExt for ResumeValueLayoutSummary {
    fn raw_fail_arg_position_or_fallback(&self) -> usize {
        self.raw_fail_arg_position.unwrap_or(self.fail_arg_index)
    }

    fn to_resume_source(&self) -> ResumeValueSource {
        match self.kind {
            ResumeValueKind::FailArg => {
                ResumeValueSource::FailArg(self.raw_fail_arg_position_or_fallback())
            }
            ResumeValueKind::Constant => {
                let raw = self.constant.expect("missing constant value");
                let tp = self.constant_type.expect("missing constant type");
                ResumeValueSource::Constant(Const::from_raw_i64(raw, tp))
            }
            ResumeValueKind::Virtual => {
                ResumeValueSource::Virtual(self.virtual_index.expect("missing virtual index"))
            }
            ResumeValueKind::Uninitialized => ResumeValueSource::Uninitialized,
            ResumeValueKind::Unavailable => ResumeValueSource::Unavailable,
        }
    }

    fn to_exit_source(&self, virtual_offset: usize) -> ExitValueSourceLayout {
        match self.kind {
            ResumeValueKind::FailArg => {
                ExitValueSourceLayout::ExitValue(self.raw_fail_arg_position_or_fallback())
            }
            ResumeValueKind::Constant => {
                ExitValueSourceLayout::Constant(self.constant.expect("missing constant value"))
            }
            ResumeValueKind::Virtual => ExitValueSourceLayout::Virtual(
                self.virtual_index.expect("missing virtual index") + virtual_offset,
            ),
            ResumeValueKind::Uninitialized => ExitValueSourceLayout::Uninitialized,
            ResumeValueKind::Unavailable => ExitValueSourceLayout::Unavailable,
        }
    }
}

/// Free function constructor (replaces the moved
/// `ResumeValueLayoutSummary::from_exit_value_source` inherent method
/// — cross-crate orphan rule prevents defining inherent impls on a
/// foreign type, so the conversion lives as a `pub fn`).
pub fn resume_value_layout_summary_from_exit_value_source(
    source: &ExitValueSourceLayout,
) -> ResumeValueLayoutSummary {
    match source {
        ExitValueSourceLayout::ExitValue(index) => ResumeValueLayoutSummary {
            kind: ResumeValueKind::FailArg,
            fail_arg_index: *index,
            raw_fail_arg_position: Some(*index),
            constant: None,
            constant_type: None,
            virtual_index: None,
        },
        ExitValueSourceLayout::Constant(value) => ResumeValueLayoutSummary {
            kind: ResumeValueKind::Constant,
            fail_arg_index: 0,
            raw_fail_arg_position: None,
            constant: Some(*value),
            constant_type: Some(Type::Int),
            virtual_index: None,
        },
        ExitValueSourceLayout::Virtual(index) => ResumeValueLayoutSummary {
            kind: ResumeValueKind::Virtual,
            fail_arg_index: 0,
            raw_fail_arg_position: None,
            constant: None,
            constant_type: None,
            virtual_index: Some(*index),
        },
        ExitValueSourceLayout::Uninitialized => ResumeValueLayoutSummary {
            kind: ResumeValueKind::Uninitialized,
            fail_arg_index: 0,
            raw_fail_arg_position: None,
            constant: None,
            constant_type: None,
            virtual_index: None,
        },
        ExitValueSourceLayout::Unavailable => ResumeValueLayoutSummary {
            kind: ResumeValueKind::Unavailable,
            fail_arg_index: 0,
            raw_fail_arg_position: None,
            constant: None,
            constant_type: None,
            virtual_index: None,
        },
    }
}
