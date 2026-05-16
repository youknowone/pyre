//! Re-export of `majit_ir::vec_assoc::VecAssoc` for paths that
//! historically routed through `crate::optimizeopt::vec_assoc`.
//!
//! Shared with `majit-translate` so cross-crate state (e.g.
//! `JitCodeBody.resulttypes`) can use the same Vec-backed container.

pub use majit_ir::vec_assoc::VecAssoc;
