//! Source parsing: file-level metadata (use-imports, glob re-exports)
//! and opcode-dispatch selector types.

use serde::{Deserialize, Serialize};
use syn::{File, Item};

/// Raw opcode-dispatch arm extracted from the interpreter match.
///
/// This is the canonical parse/front-end view of opcode dispatch before
/// graph/pipeline classification is attached.
#[derive(Debug, Clone)]
pub struct ExtractedOpcodeArm {
    pub selector: OpcodeDispatchSelector,
    /// Semantic graph of the match arm body.
    /// This is the handler's own graph — the primary input for
    /// jtransform/flatten.
    pub body_graph: Option<crate::model::FunctionGraph>,
    /// Set when the arm body is a single tail-call to a lifted
    /// per-opcode handler free fn (`execute_<op>(dispatcher params)`).
    /// In that case `body_graph` is the mechanically synthesized
    /// dispatcher-shaped wrapper, and this records the handler's
    /// [`CallPath`] — the seam that lets the JIT resolve the Charon/MIR
    /// handler graph by name instead of re-lowering the arm body.
    pub mir_handler_path: Option<CallPath>,
}

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OpcodeDispatchSelector {
    Path(CallPath),
    Wildcard,
    Or(Vec<OpcodeDispatchSelector>),
    Unsupported,
}

impl OpcodeDispatchSelector {
    pub fn canonical_key(&self) -> String {
        match self {
            Self::Path(path) => path.canonical_key(),
            Self::Wildcard => "_".into(),
            Self::Or(cases) => cases
                .iter()
                .map(Self::canonical_key)
                .collect::<Vec<_>>()
                .join(" | "),
            Self::Unsupported => "<unsupported>".into(),
        }
    }
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

/// Parsed representation of an interpreter source file.
pub struct ParsedInterpreter {
    pub file: File,
    /// Crate-stripped module path of this source file
    /// (e.g. `"intobject"` for `pyre_object/src/intobject.rs`).
    /// Empty when the caller did not supply one — top-level items
    /// remain at simple-name registration.
    pub module_path: String,
    /// `use` declarations resolved into an alias → fully-qualified-path
    /// table, populated by [`collect_use_imports`].  Mirrors PyPy's
    /// `annotator.bookkeeper` import-resolution step: when the AST
    /// references a bare type name `Foo` that this file pulled in via
    /// `use other_mod::Foo;` (or `use other_mod::Foo as Q;`), the
    /// canonical fully-qualified path lives under the in-scope alias
    /// here so `qualify_to_canonical_struct` can resolve cross-module
    /// type identity without re-walking the source tree.
    pub use_imports: std::collections::HashMap<String, String>,
    /// `pub use <path>::*` glob re-exports at the file root.  Each
    /// entry is the source-path segments (after `crate::` /
    /// `self::` / `super::` are stripped), without the trailing
    /// glob.  Example: `pub use crate::objspace::descroperation::*;`
    /// in `baseobjspace.rs` (module path `"baseobjspace"`) records
    /// `["objspace", "descroperation"]`.  Consumed at
    /// `lib.rs::analyze_pipeline_from_parsed` to emit extra alias
    /// paths so callers writing `crate::baseobjspace::pos(...)`
    /// (Rust-resolved through the re-export) resolve to the same
    /// function graph the walker registered under
    /// `crate::objspace::descroperation::pos`.  Non-pub `use` globs
    /// AND pub uses of non-`crate::` paths (external crates, stdlib)
    /// are skipped.
    pub pub_use_globs: Vec<Vec<String>>,
    /// Non-`pub` `use <path>::*` glob imports at the file root.  Each
    /// entry is the source-path segments after `crate::` / `self::` /
    /// pyre-internal-crate-alias roots are stripped, without the
    /// trailing glob.  Example: `use crate::pyobject::*;` in
    /// `excobject.rs` records `["pyobject"]`.  Each glob root is meant
    /// to expand into explicit `(alias → full_path)` entries on the
    /// per-file `use_imports` map — mirroring Python's import-resolution
    /// step that binds glob-imported names into the importing module's
    /// namespace at module-load time — so bare names resolve through the
    /// primary `use_imports` lookup without a separate glob fallback.
    pub use_globs: Vec<Vec<String>>,
}

pub fn parse_source(source: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    let use_imports = collect_use_imports(&file.items);
    let pub_use_globs = collect_pub_use_globs(&file.items);
    let use_globs = collect_use_globs(&file.items);
    ParsedInterpreter {
        file,
        module_path: String::new(),
        use_imports,
        pub_use_globs,
        use_globs,
    }
}

/// Parse a bundled Rust source file with its crate-stripped module
/// path.  e.g. `parse_source_with_module(src, "intobject")` for
/// `pyre_object/src/intobject.rs` — aligns analyzer-side
/// `path_hash(canonical_struct_name)` with the runtime's
/// dual-published `path_hash(strip_crate(module_path!())::Name)` slot
/// in `gc_cache._cache_size` (PyPy `cache[STRUCT]` lltype-object
/// identity, descr.py:108-118).
pub fn parse_source_with_module(source: &str, module_path: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    let use_imports = collect_use_imports(&file.items);
    let pub_use_globs = collect_pub_use_globs(&file.items);
    let use_globs = collect_use_globs(&file.items);
    ParsedInterpreter {
        file,
        module_path: module_path.to_string(),
        use_imports,
        pub_use_globs,
        use_globs,
    }
}

/// Walk every `Item::Use` at the file root and recursively expand the
/// use tree into an `{alias → full_path}` table.
///
/// Handles `UseTree::Path`, `UseTree::Name`, `UseTree::Rename` (`use X
/// as Y`), and `UseTree::Group` (`use X::{A, B}`).  `UseTree::Glob`
/// (`use X::*`) is recorded as a no-op: pyre cannot resolve glob
/// exports without re-parsing the target module, so glob-imported
/// bare names fall back to the same-module-default qualification.
///
/// Restricted to file-root use statements: PyPy's resolver also only
/// honours module-level imports (`annrpython.py` bookkeeper); function-
/// local `use` clauses are out of scope.
pub(crate) fn collect_use_imports(items: &[Item]) -> std::collections::HashMap<String, String> {
    let mut imports = std::collections::HashMap::new();
    for item in items {
        if let Item::Use(u) = item {
            walk_use_tree(&u.tree, &mut Vec::new(), &mut imports);
        }
    }
    imports
}

/// Walk every file-root `pub use <path>::*;` and collect the source
/// path segments (after `crate::` / `self::` are stripped, with
/// internal pyre-crate roots also stripped to mirror the rest of the
/// analyzer's namespace normalisation).  Non-`pub` use globs are
/// excluded — those are private imports only the file itself can
/// resolve through, never visible to external callers.
///
/// `pub use foo::*` (no `crate::`) and globs rooted in external
/// crates (`std`, `core`, `alloc`, well-known external workspace
/// crates) are also skipped: pyre cannot synthesise sensible aliases
/// for them without re-walking the target module, and the function-
/// registry only carries pyre-source paths in the first place.
pub(crate) fn collect_pub_use_globs(items: &[Item]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for item in items {
        let Item::Use(u) = item else {
            continue;
        };
        if !matches!(u.vis, syn::Visibility::Public(_)) {
            continue;
        }
        walk_pub_use_for_globs(&u.tree, &mut Vec::new(), &mut out);
    }
    out
}

/// Walk every file-root `use <path>::*` (non-`pub`) statement and
/// collect the source-path segments (after `crate::` / `self::` /
/// pyre-internal-crate-alias roots are stripped).  Mirrors
/// [`collect_pub_use_globs`] but for plain `use` instead of `pub use`.
/// The result is stored on [`ParsedInterpreter::use_globs`], where each
/// glob root is meant to expand into explicit `use_imports` entries at
/// semantic build time.
pub(crate) fn collect_use_globs(items: &[Item]) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for item in items {
        let Item::Use(u) = item else {
            continue;
        };
        if matches!(u.vis, syn::Visibility::Public(_)) {
            continue;
        }
        walk_pub_use_for_globs(&u.tree, &mut Vec::new(), &mut out);
    }
    out
}

fn walk_pub_use_for_globs(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    out: &mut Vec<Vec<String>>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            walk_pub_use_for_globs(&p.tree, prefix, out);
            prefix.pop();
        }
        syn::UseTree::Glob(_) => {
            let stripped = strip_glob_root(prefix);
            if !stripped.is_empty() {
                out.push(stripped);
            }
        }
        syn::UseTree::Group(g) => {
            for sub in &g.items {
                walk_pub_use_for_globs(sub, prefix, out);
            }
        }
        syn::UseTree::Name(_) | syn::UseTree::Rename(_) => {
            // Named re-exports (`pub use crate::M::name`) are handled
            // by the regular `walk_use_tree` namespace machinery via
            // `use_imports` — only Glob entries need the per-source
            // alias-fan-out at registration time.
        }
    }
}

/// Strip the leading namespace-root token (`crate`, `self`, or a
/// well-known pyre crate alias) so the stored segments match the
/// `func.module_path` shape used downstream (`pyre-interpreter/src/
/// foo.rs` → module path `"foo"`, segments `["foo"]`).
fn strip_glob_root(prefix: &[String]) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let first = prefix[0].as_str();
    let is_local_root = matches!(first, "crate" | "self");
    let is_pyre_alias = PYRE_INTERNAL_CRATES.contains(&first);
    if is_local_root || is_pyre_alias {
        prefix[1..].iter().cloned().collect()
    } else {
        // External crates / stdlib — skip.  Returning empty signals
        // the caller to drop this entry.
        Vec::new()
    }
}

fn walk_use_tree(
    tree: &syn::UseTree,
    prefix: &mut Vec<String>,
    imports: &mut std::collections::HashMap<String, String>,
) {
    match tree {
        syn::UseTree::Path(p) => {
            prefix.push(p.ident.to_string());
            walk_use_tree(&p.tree, prefix, imports);
            prefix.pop();
        }
        syn::UseTree::Name(n) => {
            let alias = n.ident.to_string();
            prefix.push(alias.clone());
            imports.insert(alias, joined_use_path(prefix));
            prefix.pop();
        }
        syn::UseTree::Rename(r) => {
            prefix.push(r.ident.to_string());
            imports.insert(r.rename.to_string(), joined_use_path(prefix));
            prefix.pop();
        }
        syn::UseTree::Glob(_) => {
            // No exposed names — caller falls back to local-module qualification.
        }
        syn::UseTree::Group(g) => {
            for sub in &g.items {
                walk_use_tree(sub, prefix, imports);
            }
        }
    }
}

/// Crate-root names that the analyzer treats as the local namespace —
/// stripped from `use` paths the same way `crate::` is stripped.  This
/// list aligns the analyzer's `path_hash(canonical_struct_name)`
/// namespace with the runtime's `module_path!()`-stripped namespace and
/// keeps cross-crate impl-method receiver spelling identical to the
/// crate-stripped `module_path_from_source_file` form used by the
/// production `analyze_multiple_pipeline_with_modules` entries.
pub(crate) const PYRE_INTERNAL_CRATES: &[&str] = &[
    "pyre_interpreter",
    "pyre_jit",
    "pyre_jit_trace",
    "pyre_object",
    "majit_ir",
    "majit_metainterp",
    "majit_translate",
    "majit_gc",
    "majit_backend_dynasm",
    "majit_backend_cranelift",
];

/// Join the accumulated `use` path segments and drop the leading
/// `crate::` keyword (or any analyzer-internal crate root in
/// [`PYRE_INTERNAL_CRATES`]) when present.  Runtime `#[jit_struct]`
/// hashes types through `majit_ir::descr::path_hash_stripped_crate`,
/// which strips the leading `module_path!()` segment (the crate root)
/// before hashing.  Analyzer-side `path_hash` must see the same
/// namespace, so the `crate::` syntactic marker (and the equivalent
/// crate-root segment for cross-crate `use foo_crate::bar::T` imports
/// inside the analyzer's source set) is dropped here at collection
/// time rather than at every consumer.  `use other_crate::Foo` paths
/// from crates outside the analyzer's source set are kept verbatim.
fn joined_use_path(segments: &[String]) -> String {
    if let Some(first) = segments.first().map(String::as_str) {
        if first == "crate" || PYRE_INTERNAL_CRATES.contains(&first) {
            return segments[1..].join("::");
        }
    }
    segments.join("::")
}

pub(crate) fn reject_duplicate_opcode_selectors(
    arms: Vec<ExtractedOpcodeArm>,
) -> Vec<ExtractedOpcodeArm> {
    let mut seen = std::collections::HashMap::new();
    for (idx, arm) in arms.iter().enumerate() {
        let key = arm.selector.canonical_key();
        if let Some(first_idx) = seen.insert(key.clone(), idx) {
            panic!(
                "duplicate opcode dispatch selector `{key}` at arm {} and arm {}",
                first_idx + 1,
                idx + 1
            );
        }
    }
    arms
}
