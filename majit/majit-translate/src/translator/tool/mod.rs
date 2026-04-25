//! `translator/tool/` — Rust port of `rpython/translator/tool/`.
//!
//! Only the subset required by the annotator / driver ports lands;
//! new submodules arrive as their dependencies are pulled in.
//!
//! | majit                 | rpython/translator/tool/      |
//! |-----------------------|-------------------------------|
//! | `taskengine`          | `taskengine.py`               |
//!
//! Deferred (no current consumer): `cbuild.py`, `graphpage.py`,
//! `lltracker.py`, `make_dot.py`, `pdbplus.py`, `reftracker.py`,
//! `staticsizereport.py`, `stdoutcapture.py`. They land alongside
//! the consumers that require them.

pub mod taskengine;
