//! `translator/goal/` — port of `rpython/translator/goal/`.
//!
//! Upstream is the package that hosts `targetstandalone.py`,
//! `translate.py`, `unixcheckpoint.py` and `query.py`. Only modules
//! consumed by the already-ported driver land here:
//!
//! | upstream                              | local              |
//! |---------------------------------------|--------------------|
//! | `rpython/translator/goal/query.py`    | [`query`]          |
//!
//! The other goal-package modules (`targetstandalone`, `translate`,
//! `unixcheckpoint`) are c-backend / fork-checkpoint concerns and are
//! deferred to the corresponding upstream-leaf-driven ports.

pub mod query;
pub mod unixcheckpoint;
