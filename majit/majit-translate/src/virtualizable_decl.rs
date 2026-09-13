//! The `_virtualizable_` class declaration, as a registered set of roots.
//!
//! RPython puts it on the class: `_virtualizable_ = ['x', 'y[*]']` is read
//! back through `classdesc.get_param('_virtualizable_')`, and
//! `rlib/jit.py`'s `hint` entry consults exactly that before it mints
//! `access_directly` on a `SomeInstance` — a value whose class does not
//! declare it has the flags deleted instead.
//!
//! Pyre's interpreter is hand-written Rust and no struct carries that
//! parameter, so the declaration is supplied out of band by the consumer, the
//! same way the codewriter's `GraphTransformConfig::vable_fields` is
//! (`rvirtualizable.rs` records why). This module is where the front end
//! reads it back, so the minter's class test can run before there is a
//! `ClassDesc` to ask.
//!
//! Upstream the declaration is per-CLASS Bookkeeper state, so it is neither
//! thread- nor invocation-scoped: `get_param` asks the class every time.
//! Scoping it to an invocation is a pyre deviation forced by having no
//! `ClassDesc` at this point, and the way it is kept honest is that the
//! pipeline re-seeds this registry from its own `AnalyzeConfig` on every run
//! (`lib.rs analyze_pipeline_from_module_paths`), deriving the roots from the
//! `owner_root` the same config already puts on every
//! `VirtualizableFieldDescriptor`. One declaration channel, refreshed per
//! invocation — the shape `local_crates` has.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::flowspace::model::{ConstValue, HostObject};

/// `pypy/module/pypyjit/interp_jit.py` `PyFrame._virtualizable_`.
pub const PYFRAME_VIRTUALIZABLE: &[&str] = &[
    "last_instr",
    "pycode",
    "valuestackdepth",
    "locals_cells_stack_w[*]",
    "debugdata",
    "w_globals",
];

thread_local! {
    /// Per-pipeline-invocation `_virtualizable_` roots, seeded by the
    /// consumer before it builds a program and read back by
    /// `front::semantic::propagate_access_directly` during the SAME
    /// invocation.
    ///
    /// Thread-local for the reason `local_crates.rs` spells out: a translate
    /// pipeline runs start-to-finish on one thread, and a process-global
    /// would let a parallel `cargo test` pipeline overwrite this run's
    /// declaration between its own seed and read.
    static REGISTERED: RefCell<HashMap<String, Vec<String>>> = RefCell::new(HashMap::new());
}

/// Replace this thread's `_virtualizable_` root set. A later invocation on
/// the same thread overwrites, matching the per-invocation semantics of
/// `local_crates::register_local_crate_roots`.
///
/// The production caller is the pipeline itself, which derives the set from
/// the `AnalyzeConfig` it was handed. Public for a consumer that builds a
/// program without going through `analyze_pipeline_from_module_paths`; such a
/// consumer owns the ordering, since the read happens during the build.
pub fn register_virtualizable_roots(roots: impl IntoIterator<Item = String>) {
    REGISTERED.with(|registered| {
        *registered.borrow_mut() = roots.into_iter().map(|root| (root, Vec::new())).collect();
    });
}

/// The registered roots for the current pipeline invocation. Empty when the
/// consumer declared none, which makes the minter's class test fail closed —
/// upstream's erasing branch.
pub(crate) fn virtualizable_roots() -> HashSet<String> {
    REGISTERED.with(|registered| registered.borrow().keys().cloned().collect())
}

/// `interp_jit.py` `PyFrame._virtualizable_ = [...]` — stamp the class
/// attribute on a newly interned host when this invocation declared it.
pub fn stamp_host_virtualizable(host: &HostObject, class_key: &str) {
    let Some(fields) = field_names_for(class_key) else {
        return;
    };
    let items = fields
        .into_iter()
        .map(ConstValue::byte_str)
        .collect::<Vec<_>>();
    host.class_set("_virtualizable_", ConstValue::List(items));
}

fn field_names_for(class_key: &str) -> Option<Vec<String>> {
    if is_pyframe(class_key) && is_registered(class_key) {
        return Some(
            PYFRAME_VIRTUALIZABLE
                .iter()
                .map(|name| (*name).to_string())
                .collect(),
        );
    }
    let fields = REGISTERED.with(|registered| lookup(&registered.borrow(), class_key).cloned())?;
    if fields.is_empty() {
        return None;
    }
    Some(fields)
}

fn is_registered(class_key: &str) -> bool {
    REGISTERED.with(|registered| lookup(&registered.borrow(), class_key).is_some())
}

fn is_pyframe(class_key: &str) -> bool {
    class_key == "PyFrame" || class_key.ends_with("::PyFrame")
}

fn lookup<'a>(
    registered: &'a HashMap<String, Vec<String>>,
    class_key: &str,
) -> Option<&'a Vec<String>> {
    if let Some(fields) = registered.get(class_key) {
        return Some(fields);
    }
    registered
        .iter()
        .find_map(|(key, fields)| class_key.ends_with(&format!("::{key}")).then_some(fields))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_host_virtualizable_writes_interp_jit_list_on_pyframe() {
        register_virtualizable_roots(["PyFrame".to_string()]);
        let host = HostObject::new_class("PyFrame", vec![]);
        stamp_host_virtualizable(&host, "PyFrame");
        let ConstValue::List(items) = host
            .class_get("_virtualizable_")
            .expect("interp_jit.py assigns PyFrame._virtualizable_")
        else {
            panic!("_virtualizable_ must be a list");
        };
        let names: Vec<String> = items
            .iter()
            .map(|item| match item {
                ConstValue::ByteStr(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            names,
            PYFRAME_VIRTUALIZABLE
                .iter()
                .map(|name| (*name).to_string())
                .collect::<Vec<_>>()
        );
        register_virtualizable_roots(std::iter::empty::<String>());
    }

    #[test]
    fn stamp_host_virtualizable_skips_undeclared_classes() {
        register_virtualizable_roots(std::iter::empty::<String>());
        let host = HostObject::new_class("Plain", vec![]);
        stamp_host_virtualizable(&host, "Plain");
        assert!(host.class_get("_virtualizable_").is_none());
    }

    #[test]
    fn stamp_host_virtualizable_skips_empty_non_pyframe_lists() {
        register_virtualizable_roots(["OtherFrame".to_string()]);
        let host = HostObject::new_class("OtherFrame", vec![]);
        stamp_host_virtualizable(&host, "OtherFrame");
        assert!(host.class_get("_virtualizable_").is_none());
        register_virtualizable_roots(std::iter::empty::<String>());
    }

    #[test]
    fn lookup_does_not_share_a_qualified_declaration_across_leaf_names() {
        REGISTERED.with(|registered| {
            *registered.borrow_mut() = [("a::Frame".to_string(), vec!["x".to_string()])]
                .into_iter()
                .collect();
        });
        let host = HostObject::new_class("Frame", vec![]);
        stamp_host_virtualizable(&host, "b::Frame");
        assert!(host.class_get("_virtualizable_").is_none());
        stamp_host_virtualizable(&host, "mod::a::Frame");
        let ConstValue::List(items) = host
            .class_get("_virtualizable_")
            .expect("qualified suffix of the registered key stamps")
        else {
            panic!("_virtualizable_ must be a list");
        };
        assert_eq!(items, vec![ConstValue::byte_str("x")]);
        register_virtualizable_roots(std::iter::empty::<String>());
    }
}
