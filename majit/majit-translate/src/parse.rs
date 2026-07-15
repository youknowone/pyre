//! Call-path and inherent-method metadata carriers shared across the
//! front-end and codewriter.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallPath {
    pub segments: Vec<String>,
}

impl CallPath {
    pub fn from_segments<I, S>(segments: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: segments.into_iter().map(Into::into).collect(),
        }
    }

    /// Build the canonical CallPath for an inherent / trait-impl method.
    ///
    /// `impl_type_joined` may be a single segment (`"Foo"`) or a
    /// `::`-joined type path (`"a::Foo"`, `"mod::Outer::Inner"`). The
    /// impl_type is split into its individual segments and concatenated
    /// with the method name so that the resulting CallPath is uniform
    /// with free-fn paths (`["a", "b", "f"]`) — both the
    /// type-qualified prefix and the method name live at the same
    /// segment granularity. Previously impl methods were stored as
    /// 2-segment `[impl_type_joined, method]`, which diverged from
    /// free-fn shape and forced macro-side heuristics; this form
    /// restores uniformity (RPython parity: `getfunctionptr(graph)` is
    /// string-free and does not distinguish the two shapes
    /// `rpython/jit/codewriter/call.py:174-187`).
    ///
    // Structural adaptation: Rust `::` ↔ PyPy `.` path separator.
    // `impl_type_joined` may arrive in either spelling — Rust extraction
    // emits `module::Type`, while `ClassDef.name` mirrors classdesc.py
    // `cls.__module__ + '.' + cls.__name__` (a `.`-joined `module.Class`).
    // Split on both so the segment granularity is independent of which
    // caller minted the string: callers in lib.rs / call.rs /
    // codewriter.rs do not all route through a `.`→`::` normalization
    // boundary, so accepting both keeps the invariant statically true.
    pub fn for_impl_method(impl_type_joined: &str, method: &str) -> Self {
        let mut segments: Vec<String> = impl_type_joined
            .split("::")
            .flat_map(|s| s.split('.'))
            .filter(|seg| !seg.is_empty())
            .map(|seg| seg.to_string())
            .collect();
        segments.push(method.to_string());
        Self { segments }
    }

    pub fn last_segment(&self) -> Option<&str> {
        self.segments.last().map(String::as_str)
    }

    pub fn canonical_key(&self) -> String {
        self.segments.join("::")
    }

    /// For a path built by `for_impl_method`, extract the impl type
    /// portion (all segments except the trailing method name).
    pub fn impl_type_prefix(&self) -> String {
        if self.segments.len() >= 2 {
            self.segments[..self.segments.len() - 1].join("::")
        } else {
            self.segments.join("::")
        }
    }
}

/// Strip the module prefix and return the trailing identifier.
///
/// Accepts both spellings: a `::`-joined Rust path and the `.`-joined
/// `ClassDef.name` form (classdesc.py `cls.__module__ + '.' +
/// cls.__name__`). A plain `rsplit('.')` misses Rust-rooted values and a
/// plain `rsplit("::")` misses Python-rooted values, so strip the longer
/// `::` prefix first and then any residual `.` prefix — the final
/// identifier is recovered regardless of which separator the caller used.
pub fn canonical_leaf(name: &str) -> &str {
    let after_colon = name.rsplit("::").next().unwrap_or(name);
    after_colon.rsplit('.').next().unwrap_or(after_colon)
}

#[derive(Debug, Clone)]
pub struct InherentMethodInfo {
    pub for_type: String,
    pub self_ty_root: Option<String>,
    pub name: String,
    pub graph: crate::model::FunctionGraph,
    /// RPython: op.result.concretetype — return type for array identity.
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    pub hints: Vec<String>,
}
