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

use crate::codewriter::jtransform::{GraphTransformConfig, VirtualizableFieldDescriptor};
use crate::flowspace::model::{ConstValue, HostObject};

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

/// Replace this thread's `_virtualizable_` declarations: each root with the
/// list its class declares (`name`, or `name[*]` for an array). A later
/// invocation on the same thread overwrites, matching the per-invocation
/// semantics of `local_crates::register_local_crate_roots`.
///
/// The production caller is the pipeline itself, which derives the set from
/// the `AnalyzeConfig` it was handed ([`declarations_from_config`]). Public for
/// a consumer that builds a program without going through
/// `analyze_pipeline_from_module_paths`; such a consumer owns the ordering,
/// since the read happens during the build.
pub fn register_virtualizable_declarations(
    declarations: impl IntoIterator<Item = (String, Vec<String>)>,
) {
    REGISTERED.with(|registered| {
        let mut registered = registered.borrow_mut();
        registered.clear();
        for (root, fields) in declarations {
            registered.entry(root).or_default().extend(fields);
        }
    });
}

/// The `_virtualizable_` list each owner root declares in `config`: static
/// fields, then arrays spelled `name[*]`, each in declared index order.
pub fn declarations_from_config(config: &GraphTransformConfig) -> Vec<(String, Vec<String>)> {
    let mut declarations: Vec<(String, Vec<String>)> = Vec::new();
    let mut add = |fields: &[VirtualizableFieldDescriptor], suffix: &str| {
        let mut fields: Vec<&VirtualizableFieldDescriptor> = fields.iter().collect();
        fields.sort_by_key(|field| field.index);
        for field in fields {
            let Some(root) = field.owner_root.as_ref() else {
                continue;
            };
            let name = format!("{}{suffix}", field.name);
            match declarations.iter_mut().find(|(known, _)| known == root) {
                Some((_, names)) => names.push(name),
                None => declarations.push((root.clone(), vec![name])),
            }
        }
    };
    add(&config.vable_fields, "");
    add(&config.vable_arrays, "[*]");
    declarations
}

/// The registered roots for the current pipeline invocation. Empty when the
/// consumer declared none, which makes the minter's class test fail closed —
/// upstream's erasing branch.
pub(crate) fn virtualizable_roots() -> HashSet<String> {
    REGISTERED.with(|registered| registered.borrow().keys().cloned().collect())
}

/// `cls._virtualizable_ = [...]` (`interp_jit.py` assigns it on the frame
/// class) — stamp the class attribute on a newly interned host when this
/// invocation declared it.
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
    let fields = REGISTERED.with(|registered| lookup(&registered.borrow(), class_key).cloned())?;
    if fields.is_empty() {
        return None;
    }
    Some(fields)
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
    fn stamp_host_virtualizable_writes_the_declared_list() {
        register_virtualizable_declarations([(
            "Frame".to_string(),
            vec!["pc".to_string(), "stack[*]".to_string()],
        )]);
        let host = HostObject::new_class("Frame", vec![]);
        stamp_host_virtualizable(&host, "Frame");
        let ConstValue::List(items) = host
            .class_get("_virtualizable_")
            .expect("the declared class carries _virtualizable_")
        else {
            panic!("_virtualizable_ must be a list");
        };
        assert_eq!(
            items,
            vec![ConstValue::byte_str("pc"), ConstValue::byte_str("stack[*]")]
        );
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }

    #[test]
    fn declarations_from_config_lists_statics_then_arrays_per_root() {
        let config = GraphTransformConfig {
            vable_fields: vec![
                VirtualizableFieldDescriptor::new("b", Some("Frame".into()), 1),
                VirtualizableFieldDescriptor::new("a", Some("Frame".into()), 0),
                VirtualizableFieldDescriptor::new("loose", None, 2),
            ],
            vable_arrays: vec![VirtualizableFieldDescriptor::new(
                "items",
                Some("Frame".into()),
                0,
            )],
            ..GraphTransformConfig::default()
        };
        assert_eq!(
            declarations_from_config(&config),
            vec![(
                "Frame".to_string(),
                vec!["a".to_string(), "b".to_string(), "items[*]".to_string()]
            )]
        );
    }

    #[test]
    fn stamp_host_virtualizable_skips_undeclared_classes() {
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
        let host = HostObject::new_class("Plain", vec![]);
        stamp_host_virtualizable(&host, "Plain");
        assert!(host.class_get("_virtualizable_").is_none());
    }

    #[test]
    fn stamp_host_virtualizable_skips_a_root_with_no_fields() {
        register_virtualizable_declarations([("OtherFrame".to_string(), Vec::new())]);
        let host = HostObject::new_class("OtherFrame", vec![]);
        stamp_host_virtualizable(&host, "OtherFrame");
        assert!(host.class_get("_virtualizable_").is_none());
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
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
        register_virtualizable_declarations(std::iter::empty::<(String, Vec<String>)>());
    }
}
