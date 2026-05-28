//! Source parsing: extract opcode dispatch and trait impls.

use crate::{MethodInfo, TraitImplInfo};
use serde::{Deserialize, Serialize};
use syn::{ExprMatch, File, Item, ItemFn, Pat, Path, visit::Visit};

/// Raw opcode-dispatch arm extracted from the interpreter match.
///
/// This is the canonical parse/front-end view of opcode dispatch before
/// graph/pipeline classification is attached.
#[derive(Debug, Clone)]
pub struct ExtractedOpcodeArm {
    pub selector: OpcodeDispatchSelector,
    pub handler_calls: Vec<ExtractedHandlerCall>,
    /// Semantic graph of the match arm body.
    /// This is the handler's own graph — the primary input for
    /// jtransform/flatten. handler_calls are metadata only.
    pub body_graph: Option<crate::model::FunctionGraph>,
}

#[derive(Debug, Clone, Default)]
pub struct ReceiverTraitBindings {
    pub traits_by_receiver: std::collections::HashMap<String, Vec<String>>,
    pub type_root_by_receiver: std::collections::HashMap<String, String>,
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
    // Both are accepted because ClassDef.name mirrors classdesc.py
    // `cls.__module__ + '.' + cls.__name__` while Rust extraction
    // emits `module::Type`.
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

/// Strip every `::` and `.` prefix and return the trailing segment.
///
/// Type-name strings traversing the codewriter boundary carry one of
/// two separator conventions: `module.Class` (RPython parity:
/// `classdesc.py:500-502 cls.__module__ + '.' + cls.__name__`) or
/// `module::Class` (Rust path extraction at `parse.rs:632-635
/// self_ty_root_qualified`). Comparators that want the bare leaf
/// (override pattern matchers, debug printers) must accept both — a
/// plain `rsplit('.')` misses Rust-rooted values and a plain
/// `rsplit("::")` misses Python-rooted values. Strip the longer
/// `::` first, then the single-char `.`, so the returned slice is
/// the final identifier regardless of mix.
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtractedHandlerCall {
    Method {
        name: String,
        receiver_root: Option<String>,
    },
    FunctionPath(CallPath),
    UnsupportedFunctionExpr,
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

/// `pub const NAME: TY = ...;` vs `pub static NAME: TY = ...;`.
/// `Const` values are compile-time inlined; `Static` values have a
/// stable memory location and are addressable as `&NAME` / `&mut NAME`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModuleStaticKind {
    Const,
    Static,
}

/// Statically-known literal value attached to a `pub const` /
/// `pub static` declaration whose initialiser is a single
/// `syn::Lit::{Bool, Int, Float}` expression.  Mirrors PyPy
/// `flowspace`'s `Constant(value)` shape — the value is known at
/// compile time and lowers to a typed `OpKind::Const*` node
/// directly, sidestepping the body-`Input` mis-classification path.
///
/// `Float` stores `f64::to_bits()` matching the `OpKind::ConstFloat`
/// payload format used by the `syn::Lit::Float` arm at
/// `front/ast.rs:5030`.
#[derive(Debug, Clone, Copy)]
pub enum ModuleStaticLiteral {
    Bool(bool),
    Int(i64),
    Float(u64),
}

/// Module-level `const` / `static` declaration collected at parse time.
///
/// Records enough information for the front-end `Expr::Path` arm and
/// the annotator to resolve a bare/qualified reference to a known
/// global without falling through to a body-`Input` mis-classification.
///
/// PyPy parity: mirrors `flowspace/flowcontext.py:858 LOAD_GLOBAL`
/// resolving a statically known module attribute to a
/// `Constant(value)` node.  The Python flowspace reaches the value
/// via attribute access on the function's `__globals__` dict; the
/// Rust port collects the same data at parse time so the
/// `Expr::Path` arm does not have to re-walk the source tree.
#[derive(Debug, Clone)]
pub struct ModuleStaticDecl {
    /// `const` or `static`.
    pub kind: ModuleStaticKind,
    /// Type root identifier (last segment of a `syn::Type::Path`),
    /// e.g. `"PyType"` for `pub static INT_TYPE: PyType = ...`.
    /// `None` for non-path types (tuple / function pointer / array
    /// / etc.) — the diagnostic [`Self::type_str`] still records the
    /// full type text in that case.
    pub type_root: Option<String>,
    /// Full type text rendered via `quote::ToTokens`.  Diagnostic
    /// only; the front-end resolver should key off `type_root`.
    pub type_str: String,
    /// Compile-time literal value, when the initialiser expression is
    /// in the small module-global subset handled by
    /// [`literal_value_from_expr_with_bindings`].  `None` for
    /// non-literal initialisers (e.g. `new_pytype("int")`,
    /// `std::ptr::null_mut()`, composite struct literal) — those need
    /// addressable-host-value annotator wiring instead.
    pub literal: Option<ModuleStaticLiteral>,
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
    /// `pub const` / `pub static` declarations, keyed by
    /// `(nested_mod_chain, bare_name)`.  File-root decls use an empty
    /// `nested_mod_chain`; decls inside `mod foo { ... }` use
    /// `"foo"`, and `mod foo { mod bar { ... } }` uses `"foo::bar"`.
    /// Populated by [`collect_module_statics`].
    ///
    /// Inline-module recursion mirrors the front-end's
    /// `GraphBuildContext.module_prefix` namespacing
    /// (`front/ast.rs:1453-1471 build_graphs_from_items` Item::Mod arm)
    /// so a `pub const FOO` declared inside `mod foo { ... }` is
    /// addressable from a path expression inside the same `mod foo`
    /// scope as bare `FOO`, and from outside as `foo::FOO`.
    pub module_statics: std::collections::HashMap<(String, String), ModuleStaticDecl>,
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
    /// `excobject.rs` records `["pyobject"]`.  Consumed at semantic
    /// build time by
    /// [`crate::front::ast::build_semantic_program_from_parsed_files_with_options`]
    /// which expands each glob into explicit `(alias → full_path)`
    /// entries on the per-file `use_imports` map — mirroring Python's
    /// import-resolution step that binds glob-imported names into the
    /// importing module's namespace at module-load time.  The
    /// front-end `Expr::Path` arm then resolves bare names through
    /// the primary `use_imports` lookup without a separate glob
    /// fallback.
    pub use_globs: Vec<Vec<String>>,
}

pub fn parse_source(source: &str) -> ParsedInterpreter {
    let file = syn::parse_file(source).expect("failed to parse bundled source");
    let use_imports = collect_use_imports(&file.items);
    let module_statics = collect_module_statics(&file.items);
    let pub_use_globs = collect_pub_use_globs(&file.items);
    let use_globs = collect_use_globs(&file.items);
    ParsedInterpreter {
        file,
        module_path: String::new(),
        use_imports,
        module_statics,
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
    let module_statics = collect_module_statics(&file.items);
    let pub_use_globs = collect_pub_use_globs(&file.items);
    let use_globs = collect_use_globs(&file.items);
    ParsedInterpreter {
        file,
        module_path: module_path.to_string(),
        use_imports,
        module_statics,
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
/// The result is stored on [`ParsedInterpreter::use_globs`] and
/// consumed by `front::ast::build_semantic_program_*_with_options`,
/// which expands each glob root into explicit `use_imports` entries
/// at semantic build time.
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

/// Walk every `Item::Const` and `Item::Static` and collect
/// their declared name + type into a `{(nested_mod_chain, name) →
/// ModuleStaticDecl}` table.  Recurses into `Item::Mod` blocks so
/// inline-module decls are addressable under their nested chain —
/// matching the front-end's `module_prefix` namespacing
/// (`front/ast.rs:1453-1471 build_graphs_from_items` Item::Mod arm),
/// which is the read side that consumes this table.  Decls inside
/// functions or impl blocks remain out of scope (they are not
/// addressable as bare path-expressions from a sibling lowering
/// site).
///
/// `pub` is not required.  Pyre's downstream resolver does not enforce
/// visibility; the per-file `use_imports` table is the authoritative
/// gate for cross-file lookups.
pub(crate) fn collect_module_statics(
    items: &[Item],
) -> std::collections::HashMap<(String, String), ModuleStaticDecl> {
    let mut statics = std::collections::HashMap::new();
    let mut initializers = indexmap::IndexMap::new();
    collect_module_statics_into(items, "", &mut statics, &mut initializers);

    let mut memo = std::collections::HashMap::new();
    for key in initializers.keys().cloned().collect::<Vec<_>>() {
        let literal = resolve_module_static_literal(
            &key,
            &initializers,
            &mut memo,
            &mut std::collections::HashSet::new(),
        );
        if let Some(decl) = statics.get_mut(&key) {
            decl.literal = literal;
        }
    }
    statics
}

fn collect_module_statics_into(
    items: &[Item],
    prefix: &str,
    statics: &mut std::collections::HashMap<(String, String), ModuleStaticDecl>,
    initializers: &mut indexmap::IndexMap<(String, String), syn::Expr>,
) {
    use quote::ToTokens;
    for item in items {
        match item {
            Item::Const(c) => {
                let type_str = c.ty.to_token_stream().to_string();
                let type_root = type_root_ident(&c.ty);
                let key = (prefix.to_string(), c.ident.to_string());
                initializers.insert(key.clone(), c.expr.as_ref().clone());
                statics.insert(
                    key,
                    ModuleStaticDecl {
                        kind: ModuleStaticKind::Const,
                        type_root,
                        type_str,
                        literal: None,
                    },
                );
            }
            // `static mut FOO` is runtime-mutable storage; constfolding
            // its initial value is unsound because LOAD_GLOBAL must read
            // the current `module.__dict__` value (PyPy
            // `flowspace/flowcontext.py:845`).  The rust_source walker
            // already excludes mutable statics (`register.rs:935-936`
            // documented at `:488-494`); the parse-side snapshot must
            // apply the same gate so the front-end `Expr::Path` arm at
            // `front/ast.rs::lookup_module_static_literal` does not
            // const-fold a `static mut` initialiser.
            Item::Static(s) if matches!(s.mutability, syn::StaticMutability::None) => {
                let type_str = s.ty.to_token_stream().to_string();
                let type_root = type_root_ident(&s.ty);
                let key = (prefix.to_string(), s.ident.to_string());
                initializers.insert(key.clone(), s.expr.as_ref().clone());
                statics.insert(
                    key,
                    ModuleStaticDecl {
                        kind: ModuleStaticKind::Static,
                        type_root,
                        type_str,
                        literal: None,
                    },
                );
            }
            Item::Mod(m) => {
                if let Some((_, ref nested_items)) = m.content {
                    let nested_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_module_statics_into(
                        nested_items,
                        &nested_prefix,
                        statics,
                        initializers,
                    );
                }
            }
            _ => {}
        }
    }
}

fn resolve_module_static_literal(
    key: &(String, String),
    initializers: &indexmap::IndexMap<(String, String), syn::Expr>,
    memo: &mut std::collections::HashMap<(String, String), Option<ModuleStaticLiteral>>,
    visiting: &mut std::collections::HashSet<(String, String)>,
) -> Option<ModuleStaticLiteral> {
    if let Some(value) = memo.get(key) {
        return *value;
    }
    let expr = initializers.get(key)?;
    if !visiting.insert(key.clone()) {
        return None;
    }
    let value = literal_value_from_expr_with_bindings(expr, &key.0, initializers, memo, visiting);
    visiting.remove(key);
    memo.insert(key.clone(), value);
    value
}

/// Decode a const / static initialiser expression into a
/// [`ModuleStaticLiteral`] when it is in pyre's conservative
/// module-global subset: primitive bool/int/float literals, unary
/// `-`/`!`, integer binary operators, representable primitive casts,
/// parens/groups, and bare names present in the same inline-module
/// global table regardless of declaration order.  Returns `None` for
/// calls, references, composite literals, multi-segment paths, casts
/// whose result cannot be represented by [`ModuleStaticLiteral`], and
/// anything else that needs real host-value annotator wiring.
fn literal_value_from_expr_with_bindings(
    expr: &syn::Expr,
    current_prefix: &str,
    initializers: &indexmap::IndexMap<(String, String), syn::Expr>,
    memo: &mut std::collections::HashMap<(String, String), Option<ModuleStaticLiteral>>,
    visiting: &mut std::collections::HashSet<(String, String)>,
) -> Option<ModuleStaticLiteral> {
    let mut recurse = |e: &syn::Expr| {
        literal_value_from_expr_with_bindings(e, current_prefix, initializers, memo, visiting)
    };
    match expr {
        syn::Expr::Lit(lit) => literal_value_from_lit(&lit.lit, false),
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Neg(_),
            expr,
            ..
        }) => {
            if let syn::Expr::Lit(inner) = &**expr {
                literal_value_from_lit(&inner.lit, true)
            } else {
                match recurse(expr)? {
                    ModuleStaticLiteral::Int(n) => Some(ModuleStaticLiteral::Int(-n)),
                    ModuleStaticLiteral::Float(bits) => Some(ModuleStaticLiteral::Float(
                        (-f64::from_bits(bits)).to_bits(),
                    )),
                    _ => None,
                }
            }
        }
        syn::Expr::Unary(syn::ExprUnary {
            op: syn::UnOp::Not(_),
            expr,
            ..
        }) => match recurse(expr)? {
            ModuleStaticLiteral::Bool(b) => Some(ModuleStaticLiteral::Bool(!b)),
            ModuleStaticLiteral::Int(n) => Some(ModuleStaticLiteral::Int(!n)),
            _ => None,
        },
        syn::Expr::Binary(syn::ExprBinary {
            left, op, right, ..
        }) => {
            let lhs = recurse(left)?;
            let rhs = recurse(right)?;
            match (lhs, rhs) {
                (ModuleStaticLiteral::Int(a), ModuleStaticLiteral::Int(b)) => {
                    let result = match op {
                        syn::BinOp::BitOr(_) => a | b,
                        syn::BinOp::BitAnd(_) => a & b,
                        syn::BinOp::BitXor(_) => a ^ b,
                        syn::BinOp::Shl(_) => a.checked_shl(b as u32)?,
                        syn::BinOp::Shr(_) => a.checked_shr(b as u32)?,
                        syn::BinOp::Add(_) => a.checked_add(b)?,
                        syn::BinOp::Sub(_) => a.checked_sub(b)?,
                        syn::BinOp::Mul(_) => a.checked_mul(b)?,
                        syn::BinOp::Div(_) => a.checked_div(b)?,
                        syn::BinOp::Rem(_) => a.checked_rem(b)?,
                        _ => return None,
                    };
                    Some(ModuleStaticLiteral::Int(result))
                }
                _ => None,
            }
        }
        syn::Expr::Path(syn::ExprPath {
            qself: None, path, ..
        }) if path.segments.iter().all(|s| s.arguments.is_none()) => {
            let segs: Vec<String> = path.segments.iter().map(|s| s.ident.to_string()).collect();
            let key = resolve_path_to_key(current_prefix, &segs, initializers);
            key.and_then(|k| resolve_module_static_literal(&k, initializers, memo, visiting))
        }
        syn::Expr::Cast(syn::ExprCast { expr, ty, .. }) => literal_value_cast(recurse(expr)?, ty),
        syn::Expr::Group(g) => recurse(&g.expr),
        syn::Expr::Paren(p) => recurse(&p.expr),
        _ => None,
    }
}

fn resolve_path_to_key(
    current_prefix: &str,
    segments: &[String],
    initializers: &indexmap::IndexMap<(String, String), syn::Expr>,
) -> Option<(String, String)> {
    if segments.len() == 1 {
        return Some((current_prefix.to_string(), segments[0].clone()));
    }
    // Handle `super::NAME` — resolve to parent module prefix.
    if segments[0] == "super" {
        let parent = if let Some(idx) = current_prefix.rfind("::") {
            &current_prefix[..idx]
        } else {
            ""
        };
        return resolve_path_to_key(parent, &segments[1..], initializers);
    }
    // Handle `self::NAME` — stays in current prefix.
    if segments[0] == "self" {
        return resolve_path_to_key(current_prefix, &segments[1..], initializers);
    }
    // Handle `crate::...` — strip crate prefix.
    if segments[0] == "crate" || PYRE_INTERNAL_CRATES.contains(&segments[0].as_str()) {
        let module = segments[1..segments.len() - 1].join("::");
        let name = segments.last()?.clone();
        let key = (module, name);
        if initializers.contains_key(&key) {
            return Some(key);
        }
        return None;
    }
    // Multi-segment relative: `nested::NAME` → (current_prefix::nested, NAME)
    let module_part = segments[..segments.len() - 1].join("::");
    let name = segments.last()?.clone();
    let qualified = if current_prefix.is_empty() {
        module_part.clone()
    } else {
        format!("{}::{}", current_prefix, module_part)
    };
    let key = (qualified, name.clone());
    if initializers.contains_key(&key) {
        return Some(key);
    }
    // Fallback: try module_part alone (top-level nested mod).
    let key = (module_part, name);
    if initializers.contains_key(&key) {
        return Some(key);
    }
    None
}

fn literal_value_cast(value: ModuleStaticLiteral, ty: &syn::Type) -> Option<ModuleStaticLiteral> {
    let target = primitive_cast_target(ty)?;
    match (value, target) {
        (ModuleStaticLiteral::Bool(b), PrimitiveCastTarget::Int(t)) => {
            cast_i64_to_int(if b { 1 } else { 0 }, t)
        }
        (ModuleStaticLiteral::Int(n), PrimitiveCastTarget::Int(t)) => cast_i64_to_int(n, t),
        (ModuleStaticLiteral::Int(n), PrimitiveCastTarget::Float(FloatCastTarget::F32)) => {
            Some(ModuleStaticLiteral::Float(((n as f32) as f64).to_bits()))
        }
        (ModuleStaticLiteral::Int(n), PrimitiveCastTarget::Float(FloatCastTarget::F64)) => {
            Some(ModuleStaticLiteral::Float((n as f64).to_bits()))
        }
        (ModuleStaticLiteral::Float(bits), PrimitiveCastTarget::Int(t)) => {
            cast_f64_to_int(f64::from_bits(bits), t)
        }
        (ModuleStaticLiteral::Float(bits), PrimitiveCastTarget::Float(FloatCastTarget::F32)) => {
            Some(ModuleStaticLiteral::Float(
                ((f64::from_bits(bits) as f32) as f64).to_bits(),
            ))
        }
        (ModuleStaticLiteral::Float(bits), PrimitiveCastTarget::Float(FloatCastTarget::F64)) => {
            Some(ModuleStaticLiteral::Float(bits))
        }
        // Rust does not support arbitrary numeric/float casts to bool via `as`.
        (_, PrimitiveCastTarget::Bool) => None,
        // Rust supports bool -> integer, but not bool -> float via `as`.
        (ModuleStaticLiteral::Bool(_), PrimitiveCastTarget::Float(_)) => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrimitiveCastTarget {
    Int(IntCastTarget),
    Float(FloatCastTarget),
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntCastTarget {
    I8,
    I16,
    I32,
    I64,
    Isize,
    U8,
    U16,
    U32,
    U64,
    Usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatCastTarget {
    F32,
    F64,
}

fn primitive_cast_target(ty: &syn::Type) -> Option<PrimitiveCastTarget> {
    match ty {
        syn::Type::Path(path) if path.qself.is_none() => {
            let segment = path.path.segments.last()?;
            if !segment.arguments.is_empty() {
                return None;
            }
            match segment.ident.to_string().as_str() {
                "i8" => Some(PrimitiveCastTarget::Int(IntCastTarget::I8)),
                "i16" => Some(PrimitiveCastTarget::Int(IntCastTarget::I16)),
                "i32" => Some(PrimitiveCastTarget::Int(IntCastTarget::I32)),
                "i64" => Some(PrimitiveCastTarget::Int(IntCastTarget::I64)),
                "isize" => Some(PrimitiveCastTarget::Int(IntCastTarget::Isize)),
                "u8" => Some(PrimitiveCastTarget::Int(IntCastTarget::U8)),
                "u16" => Some(PrimitiveCastTarget::Int(IntCastTarget::U16)),
                "u32" => Some(PrimitiveCastTarget::Int(IntCastTarget::U32)),
                "u64" => Some(PrimitiveCastTarget::Int(IntCastTarget::U64)),
                "usize" => Some(PrimitiveCastTarget::Int(IntCastTarget::Usize)),
                "f32" => Some(PrimitiveCastTarget::Float(FloatCastTarget::F32)),
                "f64" => Some(PrimitiveCastTarget::Float(FloatCastTarget::F64)),
                "bool" => Some(PrimitiveCastTarget::Bool),
                _ => None,
            }
        }
        syn::Type::Paren(paren) => primitive_cast_target(&paren.elem),
        syn::Type::Group(group) => primitive_cast_target(&group.elem),
        _ => None,
    }
}

fn cast_i64_to_int(n: i64, target: IntCastTarget) -> Option<ModuleStaticLiteral> {
    let value = match target {
        IntCastTarget::I8 => (n as i8) as i64,
        IntCastTarget::I16 => (n as i16) as i64,
        IntCastTarget::I32 => (n as i32) as i64,
        IntCastTarget::I64 => n,
        IntCastTarget::Isize => (n as isize) as i64,
        IntCastTarget::U8 => (n as u8) as i64,
        IntCastTarget::U16 => (n as u16) as i64,
        IntCastTarget::U32 => (n as u32) as i64,
        IntCastTarget::U64 => {
            let u = n as u64;
            if u > i64::MAX as u64 {
                return None;
            }
            u as i64
        }
        IntCastTarget::Usize => {
            let u = n as usize;
            if (u as u128) > (i64::MAX as u128) {
                return None;
            }
            u as i64
        }
    };
    Some(ModuleStaticLiteral::Int(value))
}

fn cast_f64_to_int(n: f64, target: IntCastTarget) -> Option<ModuleStaticLiteral> {
    let value = match target {
        IntCastTarget::I8 => (n as i8) as i64,
        IntCastTarget::I16 => (n as i16) as i64,
        IntCastTarget::I32 => (n as i32) as i64,
        IntCastTarget::I64 => n as i64,
        IntCastTarget::Isize => (n as isize) as i64,
        IntCastTarget::U8 => (n as u8) as i64,
        IntCastTarget::U16 => (n as u16) as i64,
        IntCastTarget::U32 => (n as u32) as i64,
        IntCastTarget::U64 => {
            let u = n as u64;
            if u > i64::MAX as u64 {
                return None;
            }
            u as i64
        }
        IntCastTarget::Usize => {
            let u = n as usize;
            if (u as u128) > (i64::MAX as u128) {
                return None;
            }
            u as i64
        }
    };
    Some(ModuleStaticLiteral::Int(value))
}

fn literal_value_from_lit(lit: &syn::Lit, negative: bool) -> Option<ModuleStaticLiteral> {
    match lit {
        syn::Lit::Bool(b) => {
            // `!false = true` is not the same as `-false`; the negation
            // path only applies to numeric literals.  Bool reaches here
            // only when `negative=false` because there is no `-bool`.
            if negative {
                return None;
            }
            Some(ModuleStaticLiteral::Bool(b.value))
        }
        syn::Lit::Int(i) => {
            let v: i64 = i.base10_parse().ok()?;
            Some(ModuleStaticLiteral::Int(if negative { -v } else { v }))
        }
        syn::Lit::Float(f) => {
            let v: f64 = f.base10_parse().ok()?;
            let bits = (if negative { -v } else { v }).to_bits();
            Some(ModuleStaticLiteral::Float(bits))
        }
        _ => None,
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

/// Find a top-level function by exact name in the parsed source.
pub(crate) fn find_function<'a>(parsed: &'a ParsedInterpreter, name: &str) -> Option<&'a ItemFn> {
    find_function_in_file(&parsed.file, name)
}

/// Find a top-level function by exact name in a parsed file.
pub(crate) fn find_function_in_file<'a>(file: &'a File, name: &str) -> Option<&'a ItemFn> {
    file.items.iter().find_map(|item| {
        if let Item::Fn(func) = item {
            (func.sig.ident == name).then_some(func)
        } else {
            None
        }
    })
}

/// Find an opcode-dispatch `match` expression within a function.
fn find_opcode_match(func: &ItemFn) -> Option<&ExprMatch> {
    struct Finder<'a> {
        result: Option<&'a ExprMatch>,
    }

    impl<'ast> Visit<'ast> for Finder<'ast> {
        fn visit_expr_match(&mut self, node: &'ast ExprMatch) {
            if self.result.is_none() && node.arms.first().is_some_and(is_opcode_pattern) {
                self.result = Some(node);
                return;
            }
            syn::visit::visit_expr_match(self, node);
        }
    }

    let mut finder = Finder { result: None };
    finder.visit_item_fn(func);
    finder.result
}

/// Find the canonical opcode-dispatch match in a parsed interpreter source.
///
/// This is the public parse/front-end helper for consumers that still need the
/// raw `match` AST rather than the extracted `ExtractedOpcodeArm` view.
pub fn find_opcode_dispatch_match(parsed: &ParsedInterpreter) -> Option<&ExprMatch> {
    find_function(parsed, "mainloop").and_then(find_opcode_match)
}

/// Extract trait implementations AND trait default methods from the parsed source.
/// Recurses into `Item::Mod` for whole-program visibility (RPython parity).
///
/// RPython `flowspace/objspace.py:49` + `flowcontext.py:417` —
/// `build_flow()` / `buildflowgraph()` re-raise `FlowingError`, and the
/// translator observes unsupported constructs as a hard failure.  The
/// extractor mirrors that: if any trait method's body hits an
/// unsupported construct, the whole extraction aborts with Err rather
/// than silently recording a `MethodInfo.graph = None` that dispatch
/// could later route through without a semantic graph.
pub fn extract_trait_impls(
    parsed: &ParsedInterpreter,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
) -> Result<Vec<TraitImplInfo>, crate::front::ast::FlowingError> {
    let mut impls = Vec::new();
    let mut known_trait_names = std::collections::HashSet::new();
    collect_trait_names(&parsed.file.items, "", &mut known_trait_names);
    collect_trait_impls_from_items(
        &parsed.file.items,
        "",
        struct_fields,
        fn_return_types,
        &parsed.use_imports,
        known_struct_names,
        &known_trait_names,
        &mut impls,
    )?;
    Ok(impls)
}

fn collect_trait_impls_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    impls: &mut Vec<TraitImplInfo>,
) -> Result<(), crate::front::ast::FlowingError> {
    for item in items {
        match item {
            // Concrete trait impls (impl Trait for Type)
            Item::Impl(impl_block) => {
                if let Some((_, trait_path, _)) = &impl_block.trait_ {
                    let trait_name =
                        canonical_trait_path_name(trait_path, prefix, known_trait_names);
                    let self_ty = &impl_block.self_ty;
                    let for_type = canonical_type_name(self_ty);
                    // Qualify bare type name with module prefix (RPython: unique type identity).
                    // Route through `qualify_type_name_with_imports` with the same
                    // `parsed.use_imports` map graph build threads into
                    // `GraphBuildContext` so trait-impl registration keys align
                    // with use-site lookups when the receiver type is referenced
                    // via a `use <path> as alias` form.
                    let self_ty_root = type_root_ident(self_ty).map(|t| {
                        crate::front::ast::qualify_type_name_with_imports(&t, prefix, use_imports)
                    });
                    let mut methods: Vec<MethodInfo> = Vec::new();
                    for item in &impl_block.items {
                        if let syn::ImplItem::Fn(method) = item {
                            let fake_fn = syn::ItemFn {
                                attrs: method.attrs.clone(),
                                vis: syn::Visibility::Inherited,
                                sig: method.sig.clone(),
                                block: Box::new(method.block.clone()),
                            };
                            // jit.py:184-201 — `@elidable_promote` on a
                            // trait-impl method installs two callables
                            // (orig + wrapper); `synthesize_or_passthrough`
                            // makes this lowering pass see both.  Method
                            // signature parity (return type, hints) is
                            // taken from the synthesized `ItemFn`, not the
                            // original `method`, so the orig's
                            // `_orig_<NAME>_unlikely_name` identity
                            // survives downstream.
                            // `?` propagates `FlowingError` out of the
                            // extractor (RPython re-raise at
                            // `flowspace/flowcontext.py:417`).  The
                            // qualified `self_ty_root` is threaded into
                            // synthesis so the wrapper's tail call uses
                            // the same `<ImplType>::_orig_<name>_unlikely_name`
                            // path that `lib.rs:531-537` registers via
                            // `CallPath::for_impl_method`.
                            for synth in crate::front::ast::synthesize_or_passthrough(
                                fake_fn,
                                self_ty_root.as_deref(),
                            ) {
                                let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                    &synth,
                                    self_ty_root.clone(),
                                    struct_fields,
                                    fn_return_types,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )?;
                                let return_type = match &synth.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::ast::qualified_full_type_string(
                                            ty,
                                            prefix,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    syn::ReturnType::Default => Some("()".to_string()),
                                };
                                methods.push(MethodInfo {
                                    name: synth.sig.ident.to_string(),
                                    graph: Some(sf.graph),
                                    return_type,
                                    hints: sf.hints,
                                });
                            }
                        }
                    }
                    impls.push(TraitImplInfo {
                        trait_name,
                        for_type,
                        self_ty_root,
                        methods,
                    });
                }
            }
            // Trait definitions with default methods
            Item::Trait(trait_def) => {
                let trait_name = qualify_known_trait_name(
                    &trait_def.ident.to_string(),
                    prefix,
                    known_trait_names,
                );
                let mut methods: Vec<MethodInfo> = Vec::new();
                for item in &trait_def.items {
                    if let syn::TraitItem::Fn(method) = item {
                        if let Some(block) = &method.default {
                            let fake_fn = syn::ItemFn {
                                attrs: method.attrs.clone(),
                                vis: syn::Visibility::Inherited,
                                sig: method.sig.clone(),
                                block: Box::new(block.clone()),
                            };
                            // jit.py:184-201 — trait default methods get
                            // the same wrapper/orig synthesis.  The
                            // concrete `Self` type is not known until
                            // a `for <T>` impl resolves the trait, so
                            // the synthesizer emits a bare-path tail
                            // call (`None`); a downstream `impl Trait
                            // for S` block would re-emit the method
                            // with `self_ty_root = "S"` and lower the
                            // wrapper through this same path.
                            for synth in crate::front::ast::synthesize_or_passthrough(fake_fn, None)
                            {
                                let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                    &synth,
                                    None,
                                    struct_fields,
                                    fn_return_types,
                                    prefix,
                                    use_imports,
                                    known_struct_names,
                                    known_trait_names,
                                )?;
                                let return_type = match &synth.sig.output {
                                    syn::ReturnType::Type(_, ty) => {
                                        crate::front::ast::qualified_full_type_string(
                                            ty,
                                            prefix,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    syn::ReturnType::Default => Some("()".to_string()),
                                };
                                methods.push(MethodInfo {
                                    name: synth.sig.ident.to_string(),
                                    graph: Some(sf.graph),
                                    return_type,
                                    hints: sf.hints,
                                });
                            }
                        }
                    }
                }
                if !methods.is_empty() {
                    impls.push(TraitImplInfo {
                        trait_name: trait_name.clone(),
                        for_type: format!("<default methods of {}>", trait_name),
                        self_ty_root: None,
                        methods,
                    });
                }
            }
            // Recurse into module blocks with qualified prefix.
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_impls_from_items(
                        sub_items,
                        &mod_prefix,
                        struct_fields,
                        fn_return_types,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                        impls,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn collect_trait_names(
    items: &[Item],
    prefix: &str,
    known_trait_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            Item::Trait(trait_def) => {
                let bare_name = trait_def.ident.to_string();
                known_trait_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_trait_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_names(sub_items, &mod_prefix, known_trait_names);
                }
            }
            _ => {}
        }
    }
}

/// Extract inherent impl methods (impl Type { ... }) as canonical call targets.
/// Recurses into `Item::Mod` for whole-program visibility (RPython parity).
///
/// RPython `flowspace/objspace.py:49` — `build_flow()` re-raises
/// `FlowingError` rather than silently skipping a function.  Pyre's
/// inherent-method extractor mirrors that: `FlowingError` in any
/// method body aborts the whole extraction so dispatch resolution
/// never sees a silently-dropped method.
pub fn extract_inherent_impl_methods(
    parsed: &ParsedInterpreter,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
) -> Result<Vec<InherentMethodInfo>, crate::front::ast::FlowingError> {
    let mut methods = Vec::new();
    // Feed `parsed.module_path` so the inherent-impl receiver-root
    // qualification agrees with the caller-side
    // `qualify_type_name_with_imports` result, which `analyze_pipeline_from_parsed`
    // routes through `STRUCT_ORIGIN_REGISTRY` populated by
    // `collect_struct_origins` over the same module path.  Empty
    // `module_path` (fixtures using `parse_source`) falls through to
    // the bare-name registration path.
    collect_inherent_methods_from_items(
        &parsed.file.items,
        &parsed.module_path,
        struct_fields,
        fn_return_types,
        &parsed.use_imports,
        known_struct_names,
        &mut methods,
    )?;
    Ok(methods)
}

fn collect_inherent_methods_from_items(
    items: &[Item],
    prefix: &str,
    struct_fields: &crate::front::StructFieldRegistry,
    fn_return_types: &std::collections::HashMap<String, String>,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    methods: &mut Vec<InherentMethodInfo>,
) -> Result<(), crate::front::ast::FlowingError> {
    for item in items {
        match item {
            Item::Impl(impl_block) => {
                if impl_block.trait_.is_some() {
                    continue;
                }
                let for_type = canonical_type_name(&impl_block.self_ty);
                // Two `self_ty_root` forms run side by side:
                //  - `self_ty_root_bare` keeps the raw `impl` type ident
                //    (e.g. `"PyFrame"`) and is fed into the graph build so
                //    inside-the-graph `self.field` accesses carry the
                //    same `owner_root` spelling the virtualizable spec
                //    (`virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT`,
                //    plain `"PyFrame"`) matches against
                //    (`jit_codewriter/jtransform.rs::VirtualizableFieldDescriptor::matches`).
                //  - `self_ty_root_qualified` runs the bare name through
                //    `qualify_type_name_with_imports` so the inherent-method
                //    registration `CallPath::for_impl_method` agrees with
                //    the caller-side receiver-type spelling
                //    (`receiver_type_root` → `local_type_roots`, also fed
                //    through `qualify_type_name_with_imports`).  PyPy
                //    `bookkeeper.getdesc(value).graph` single-source
                //    identity uses the same lookup at both ends.
                let self_ty_root_bare = type_root_ident(&impl_block.self_ty);
                let self_ty_root_qualified = self_ty_root_bare.as_ref().map(|t| {
                    crate::front::ast::qualify_type_name_with_imports(t, prefix, use_imports)
                });
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let fake_fn = syn::ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        // jit.py:184-201 — inherent-impl methods get the
                        // same wrapper/orig synthesis as free fns;
                        // `?` propagates `FlowingError` per
                        // `flowspace/flowcontext.py:417`.  `synthesize_or_passthrough`
                        // and `build_function_graph_with_self_ty_pub`
                        // both receive the bare spelling so the wrapper's
                        // self-typed tail call and the body's `self.field`
                        // accesses share the same `owner_root` spelling
                        // the vable spec asserts on.
                        for synth in crate::front::ast::synthesize_or_passthrough(
                            fake_fn,
                            self_ty_root_bare.as_deref(),
                        ) {
                            let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                                &synth,
                                self_ty_root_bare.clone(),
                                struct_fields,
                                fn_return_types,
                                prefix,
                                use_imports,
                                known_struct_names,
                                &std::collections::HashSet::new(),
                            )?;
                            let return_type = match &synth.sig.output {
                                syn::ReturnType::Type(_, ty) => {
                                    crate::front::ast::qualified_full_type_string(
                                        ty,
                                        prefix,
                                        known_struct_names,
                                        &std::collections::HashSet::new(),
                                    )
                                }
                                syn::ReturnType::Default => Some("()".to_string()),
                            };
                            methods.push(InherentMethodInfo {
                                for_type: for_type.clone(),
                                self_ty_root: self_ty_root_qualified.clone(),
                                name: synth.sig.ident.to_string(),
                                graph: sf.graph,
                                return_type,
                                hints: sf.hints,
                            });
                        }
                    }
                }
            }
            // Recurse into module blocks with qualified prefix.
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_inherent_methods_from_items(
                        sub_items,
                        &mod_prefix,
                        struct_fields,
                        fn_return_types,
                        use_imports,
                        known_struct_names,
                        methods,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Extract canonical opcode dispatch arms from `execute_opcode_step`.
///
/// This preserves source-level match structure and handler calls so canonical
/// graph/pipeline consumers can resolve and classify these arms directly.
///
/// Duplicate opcode selectors are rejected. Silently keeping the first arm
/// would hide dispatch drift in the interpreter source.
pub fn extract_opcode_dispatch_arms(
    parsed: &ParsedInterpreter,
    fn_return_types: &std::collections::HashMap<String, String>,
) -> Vec<ExtractedOpcodeArm> {
    let Some(func) = find_function(parsed, "execute_opcode_step") else {
        return Vec::new();
    };
    let Some(opcode_match) = find_opcode_match(func) else {
        return Vec::new();
    };
    reject_duplicate_opcode_selectors(extract_match_arms(opcode_match, &func.sig, fn_return_types))
}

/// Extract receiver -> trait bounds for `execute_opcode_step`.
///
/// This lets canonical dispatch resolution follow generic receiver methods
/// through the trait that actually defines their default bodies.
pub fn extract_opcode_dispatch_receiver_traits(
    parsed: &ParsedInterpreter,
) -> ReceiverTraitBindings {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            if func.sig.ident == "execute_opcode_step" {
                return extract_receiver_trait_bindings(func);
            }
        }
    }
    ReceiverTraitBindings::default()
}

/// Collect canonical function names and graphs for the active pipeline path.
///
/// Test-only helper.  RPython `flowspace/objspace.py:49` re-raise
/// semantics — `FlowingError` propagates out rather than silently
/// dropping the graph.
///
/// `metadata` carries the whole-program registries
/// (`struct_fields` / `fn_return_types` / `known_struct_names` /
/// `known_trait_names`) the per-function `build_function_graph_*`
/// call needs; callers build it once across all parsed files so a
/// callsite in one file can resolve a free function defined in
/// another (RPython `annrpython.py:103-150 build_types` is a single
/// whole-program pass before per-function graph build).
#[cfg(test)]
pub fn collect_function_graphs(
    parsed: &ParsedInterpreter,
    metadata: &crate::front::ast::ProgramMetadata,
    graphs: &mut std::collections::HashMap<CallPath, crate::model::FunctionGraph>,
) -> Result<(), crate::front::ast::FlowingError> {
    for item in &parsed.file.items {
        if let Item::Fn(func) = item {
            let name = func.sig.ident.to_string();
            let sf = crate::front::ast::build_function_graph_with_self_ty_pub(
                func,
                None,
                &metadata.struct_fields,
                &metadata.fn_return_types,
                "",
                &parsed.use_imports,
                &metadata.known_struct_names,
                &metadata.known_trait_names,
            )?;
            graphs.insert(CallPath::from_segments([name.clone()]), sf.graph.clone());
            graphs.insert(CallPath::from_segments(["crate", name.as_str()]), sf.graph);
        }
    }
    Ok(())
}

/// Extract opcode-dispatch arms from the canonical dispatch match only.
///
/// RPython `flowspace/objspace.py:49` + `flowcontext.py:417` —
/// `FlowingError` propagates out of `build_flow()`, making unsupported
/// constructs a hard failure.  Pyre's dispatch extractor mirrors that:
/// an arm body that hits an unsupported construct aborts the walk with
/// a panic rather than silently dropping the arm's graph.  Silently
/// dropping would let a `PipelineOpcodeArm` reach the codewriter
/// without the semantic graph the downstream jitcode path depends on.
fn extract_match_arms(
    expr: &ExprMatch,
    sig: &syn::Signature,
    fn_return_types: &std::collections::HashMap<String, String>,
) -> Vec<ExtractedOpcodeArm> {
    expr.arms
        .iter()
        .map(|arm| {
            let handler_calls = extract_handler_calls(&arm.body);
            let selector = extract_opcode_dispatch_selector(&arm.pat);
            let name = selector.canonical_key();
            let mut graph = crate::model::FunctionGraph::new(name.clone());
            // Pre-bind `execute_opcode_step`'s formal parameters as
            // startblock inputargs so the arm body's `Expr::Path`
            // references (e.g. `frame`, `instruction`, `executor`)
            // resolve to those inputargs instead of falling through
            // to the naked body-`Input` emit that the flowspace
            // adapter rejects as "adapter cross-block body Input"
            // (the dominant Cat 2.1 Skip family).  PyPy/RPython parity:
            // each per-opcode handler method receives the
            // dispatcher's parameters in its formal signature
            // (`pypy/interpreter/pyopcode.py:519`,
            // `rpython/flowspace/model.py:28 startblock.inputargs`).
            crate::front::ast::lower_expr_into_graph_with_signature(
                &mut graph,
                &arm.body,
                Some(sig),
                fn_return_types,
            )
            .unwrap_or_else(|e| {
                panic!("opcode dispatch arm `{name}` must lower without FlowingError: {e:?}")
            });
            ExtractedOpcodeArm {
                selector,
                handler_calls,
                body_graph: Some(graph),
            }
        })
        .collect()
}

fn reject_duplicate_opcode_selectors(arms: Vec<ExtractedOpcodeArm>) -> Vec<ExtractedOpcodeArm> {
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

fn is_opcode_pattern(arm: &syn::Arm) -> bool {
    pattern_is_opcode_dispatch(&arm.pat)
}

fn pattern_is_opcode_dispatch(pat: &Pat) -> bool {
    match pat {
        Pat::Ident(pat) => pat.ident.to_string().starts_with("OP_"),
        Pat::Path(path) => path_is_opcode_dispatch(&path.path),
        Pat::Struct(pat) => path_is_opcode_dispatch(&pat.path),
        Pat::TupleStruct(pat) => path_is_opcode_dispatch(&pat.path),
        Pat::Tuple(pat) => pat.elems.iter().any(pattern_is_opcode_dispatch),
        Pat::Or(pat) => pat.cases.iter().any(pattern_is_opcode_dispatch),
        _ => false,
    }
}

fn path_is_opcode_dispatch(path: &Path) -> bool {
    let last = path
        .segments
        .last()
        .map(|segment| segment.ident.to_string());
    if let Some(last) = last {
        if last.starts_with("OP_") {
            return true;
        }
    }
    path.segments
        .iter()
        .any(|segment| segment.ident == "Instruction")
}

fn extract_opcode_dispatch_selector(pat: &Pat) -> OpcodeDispatchSelector {
    match pat {
        Pat::Ident(pat) => {
            OpcodeDispatchSelector::Path(CallPath::from_segments([pat.ident.to_string()]))
        }
        Pat::Path(path) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            path.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::Struct(pat) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            pat.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::TupleStruct(pat) => OpcodeDispatchSelector::Path(CallPath::from_segments(
            pat.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        Pat::Or(pat) => OpcodeDispatchSelector::Or(
            pat.cases
                .iter()
                .map(extract_opcode_dispatch_selector)
                .collect(),
        ),
        Pat::Wild(_) => OpcodeDispatchSelector::Wildcard,
        _ => OpcodeDispatchSelector::Unsupported,
    }
}

/// Extract handler call identities from an expression.
fn extract_handler_calls(expr: &syn::Expr) -> Vec<ExtractedHandlerCall> {
    let mut calls = Vec::new();
    let mut collector = CallCollector { calls: &mut calls };
    syn::visit::visit_expr(&mut collector, expr);
    calls
}

struct CallCollector<'a> {
    calls: &'a mut Vec<ExtractedHandlerCall>,
}

impl<'ast, 'a> Visit<'ast> for CallCollector<'a> {
    fn visit_expr_method_call(&mut self, call: &'ast syn::ExprMethodCall) {
        self.calls.push(ExtractedHandlerCall::Method {
            name: call.method.to_string(),
            receiver_root: expr_root_ident(&call.receiver),
        });
        syn::visit::visit_expr_method_call(self, call);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        self.calls.push(extract_function_call_identity(&call.func));
        syn::visit::visit_expr_call(self, call);
    }
}

fn extract_function_call_identity(expr: &syn::Expr) -> ExtractedHandlerCall {
    match expr {
        syn::Expr::Path(path) => ExtractedHandlerCall::FunctionPath(CallPath::from_segments(
            path.path.segments.iter().map(|seg| seg.ident.to_string()),
        )),
        _ => ExtractedHandlerCall::UnsupportedFunctionExpr,
    }
}

fn expr_root_ident(expr: &syn::Expr) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path.path.get_ident().map(|ident| ident.to_string()),
        syn::Expr::Reference(r) => expr_root_ident(&r.expr),
        syn::Expr::Paren(p) => expr_root_ident(&p.expr),
        syn::Expr::Field(field) => expr_root_ident(&field.base),
        syn::Expr::Index(index) => expr_root_ident(&index.expr),
        _ => None,
    }
}

fn extract_receiver_trait_bindings(func: &ItemFn) -> ReceiverTraitBindings {
    let mut generic_bounds: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    let push_bounds = |generic_bounds: &mut std::collections::HashMap<String, Vec<String>>,
                       name: String,
                       bounds: Vec<String>| {
        if bounds.is_empty() {
            return;
        }
        let entry = generic_bounds.entry(name).or_default();
        for b in bounds {
            if !entry.iter().any(|existing| existing == &b) {
                entry.push(b);
            }
        }
    };

    let collect_trait_names =
        |bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>| {
            bounds
                .iter()
                .filter_map(|bound| match bound {
                    syn::TypeParamBound::Trait(trait_bound) => {
                        Some(canonical_path_name(&trait_bound.path))
                    }
                    _ => None,
                })
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>()
        };

    for param in &func.sig.generics.params {
        if let syn::GenericParam::Type(ty) = param {
            let bounds = collect_trait_names(&ty.bounds);
            push_bounds(&mut generic_bounds, ty.ident.to_string(), bounds);
        }
    }

    if let Some(where_clause) = &func.sig.generics.where_clause {
        for predicate in &where_clause.predicates {
            let syn::WherePredicate::Type(pred) = predicate else {
                continue;
            };
            let Some(name) = type_root_ident(&pred.bounded_ty) else {
                continue;
            };
            let bounds = collect_trait_names(&pred.bounds);
            push_bounds(&mut generic_bounds, name, bounds);
        }
    }

    let mut traits_by_receiver = std::collections::HashMap::new();
    let mut type_root_by_receiver = std::collections::HashMap::new();
    for arg in &func.sig.inputs {
        let syn::FnArg::Typed(arg) = arg else {
            continue;
        };
        let syn::Pat::Ident(pat_ident) = &*arg.pat else {
            continue;
        };
        if let Some(type_name) = type_root_ident(&arg.ty) {
            type_root_by_receiver.insert(pat_ident.ident.to_string(), type_name.clone());
            if let Some(bounds) = generic_bounds.get(&type_name) {
                traits_by_receiver.insert(pat_ident.ident.to_string(), bounds.clone());
            }
        }
    }

    ReceiverTraitBindings {
        traits_by_receiver,
        type_root_by_receiver,
    }
}

/// Returns full type path — all segments joined by "::".
/// RPython: lltype.Struct has globally unique identity.
fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments.is_empty() {
                None
            } else {
                Some(segments.join("::"))
            }
        }
        syn::Type::Reference(reference) => type_root_ident(&reference.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        _ => None,
    }
}

fn canonical_path_name(path: &syn::Path) -> String {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect::<Vec<_>>()
        .join("::")
}

fn qualify_known_trait_name(
    bare: &str,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let qualified = if prefix.is_empty() || bare.contains("::") {
        None
    } else {
        Some(format!("{}::{}", prefix, bare))
    };
    if let Some(qualified) = qualified {
        if known_trait_names.contains(&qualified) {
            qualified
        } else {
            bare.to_string()
        }
    } else {
        bare.to_string()
    }
}

fn canonical_trait_path_name(
    path: &syn::Path,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let canonical = canonical_path_name(path);
    if path.segments.len() == 1 {
        qualify_known_trait_name(&canonical, prefix, known_trait_names)
    } else {
        canonical
    }
}

fn canonical_type_name(ty: &syn::Type) -> String {
    match ty {
        syn::Type::Path(path) => canonical_path_name(&path.path),
        syn::Type::Reference(reference) => canonical_type_name(&reference.elem),
        syn::Type::Paren(paren) => canonical_type_name(&paren.elem),
        syn::Type::Group(group) => canonical_type_name(&group.elem),
        syn::Type::Ptr(ptr) => canonical_type_name(&ptr.elem),
        syn::Type::Slice(slice) => format!("[{}]", canonical_type_name(&slice.elem)),
        syn::Type::Array(array) => format!("[{}]", canonical_type_name(&array.elem)),
        _ => "<unsupported-type>".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collect_module_statics_records_const_and_static_decls_at_file_root() {
        let parsed = parse_source(
            r#"
            pub const PY_NULL: *mut PyObject = std::ptr::null_mut();
            pub static INT_TYPE: PyType = new_pytype("int");
            pub const WITHPREBUILTINT: bool = false;
            // function-local + nested-module decls must not appear:
            fn helper() {
                const LOCAL: i32 = 7;
            }
            mod nested {
                pub const NESTED: u8 = 1;
            }
        "#,
        );

        let py_null = parsed
            .module_statics
            .get(&(String::new(), "PY_NULL".to_string()))
            .expect("PY_NULL collected");
        assert_eq!(py_null.kind, ModuleStaticKind::Const);
        // `*mut PyObject` is a `syn::Type::Ptr`; `type_root_ident`
        // intentionally returns `None` for non-`Path`/`Reference`
        // types (consistent with the existing call sites at
        // `parse.rs:1004` parameter-type extraction).  The full type
        // text remains available via `type_str` for diagnostics.
        assert!(py_null.type_root.is_none());
        assert!(py_null.type_str.contains("PyObject"));

        let int_type = parsed
            .module_statics
            .get(&(String::new(), "INT_TYPE".to_string()))
            .expect("INT_TYPE collected");
        assert_eq!(int_type.kind, ModuleStaticKind::Static);
        assert_eq!(int_type.type_root.as_deref(), Some("PyType"));

        let prebuiltint = parsed
            .module_statics
            .get(&(String::new(), "WITHPREBUILTINT".to_string()))
            .expect("WITHPREBUILTINT collected");
        assert_eq!(prebuiltint.kind, ModuleStaticKind::Const);
        assert_eq!(prebuiltint.type_root.as_deref(), Some("bool"));
        assert!(matches!(
            prebuiltint.literal,
            Some(ModuleStaticLiteral::Bool(false))
        ));

        // `INT_TYPE = new_pytype("int")` is a call, not a literal —
        // no `literal` payload.
        assert!(int_type.literal.is_none());
        // `PY_NULL = std::ptr::null_mut()` likewise.
        assert!(py_null.literal.is_none());

        // Function-local decls remain excluded — the walker only
        // descends into `Item::Mod` blocks, not function bodies.
        assert!(
            parsed
                .module_statics
                .get(&(String::new(), "LOCAL".to_string()))
                .is_none()
        );
        // Nested-module decl is recorded under its inline-mod prefix
        // (`"nested"` here), matching the front-end's `module_prefix`
        // namespacing.
        let nested = parsed
            .module_statics
            .get(&("nested".to_string(), "NESTED".to_string()))
            .expect("NESTED collected under nested mod prefix");
        assert_eq!(nested.kind, ModuleStaticKind::Const);
        assert_eq!(nested.type_root.as_deref(), Some("u8"));
        assert!(
            parsed
                .module_statics
                .get(&(String::new(), "NESTED".to_string()))
                .is_none()
        );
    }

    #[test]
    fn collect_module_statics_records_primitive_literal_values() {
        let parsed = parse_source(
            r#"
            pub const TRUE_FLAG: bool = true;
            pub const FALSE_FLAG: bool = false;
            pub const POS_INT: i64 = 42;
            pub const NEG_INT: i32 = -7;
            pub const POS_FLOAT: f64 = 3.14;
            pub const NEG_FLOAT: f64 = -2.5;
            // Non-literal initialisers must NOT carry a literal payload:
            pub const FROM_FN: u32 = compute();
            pub const SHIFT_EXPR: u64 = 1u64 << 32;
            pub const ACTUAL_CAST: i64 = 1 as i64;
        "#,
        );

        let get = |name: &str| {
            parsed
                .module_statics
                .get(&(String::new(), name.to_string()))
                .unwrap_or_else(|| panic!("{name} collected"))
        };
        assert!(matches!(
            get("TRUE_FLAG").literal,
            Some(ModuleStaticLiteral::Bool(true))
        ));
        assert!(matches!(
            get("FALSE_FLAG").literal,
            Some(ModuleStaticLiteral::Bool(false))
        ));
        assert!(matches!(
            get("POS_INT").literal,
            Some(ModuleStaticLiteral::Int(42))
        ));
        assert!(matches!(
            get("NEG_INT").literal,
            Some(ModuleStaticLiteral::Int(-7))
        ));
        match get("POS_FLOAT").literal {
            Some(ModuleStaticLiteral::Float(bits)) => {
                assert_eq!(f64::from_bits(bits), 3.14);
            }
            other => panic!("expected POS_FLOAT to carry Float literal, got {other:?}"),
        }
        match get("NEG_FLOAT").literal {
            Some(ModuleStaticLiteral::Float(bits)) => {
                assert_eq!(f64::from_bits(bits), -2.5);
            }
            other => panic!("expected NEG_FLOAT to carry Float literal, got {other:?}"),
        }
        assert!(get("FROM_FN").literal.is_none());
        assert!(
            matches!(get("SHIFT_EXPR").literal, Some(ModuleStaticLiteral::Int(v)) if v == 1i64 << 32)
        );
        assert!(matches!(
            get("ACTUAL_CAST").literal,
            Some(ModuleStaticLiteral::Int(1))
        ));
    }

    #[test]
    fn collect_module_statics_folds_representable_primitive_casts() {
        let parsed = parse_source(
            r#"
            pub const IDENTITY: i64 = 1 as i64;
            pub const WRAP_U8: u8 = 256 as u8;
            pub const NEG_TO_U8: u8 = -1 as u8;
            pub const BOOL_TO_INT: u8 = true as u8;
            pub const INT_TO_FLOAT: f64 = 3 as f64;
            pub const FLOAT_TO_INT: i64 = 3.9 as i64;
            pub const FLOAT_TO_F32: f64 = 1.1 as f32 as f64;
            pub const UNREPRESENTABLE_U64: u64 = -1 as u64;
            pub const INVALID_BOOL_CAST: bool = 1 as bool;
        "#,
        );

        let get = |name: &str| {
            parsed
                .module_statics
                .get(&(String::new(), name.to_string()))
                .unwrap_or_else(|| panic!("{name} collected"))
        };
        assert!(matches!(
            get("IDENTITY").literal,
            Some(ModuleStaticLiteral::Int(1))
        ));
        assert!(matches!(
            get("WRAP_U8").literal,
            Some(ModuleStaticLiteral::Int(0))
        ));
        assert!(matches!(
            get("NEG_TO_U8").literal,
            Some(ModuleStaticLiteral::Int(255))
        ));
        assert!(matches!(
            get("BOOL_TO_INT").literal,
            Some(ModuleStaticLiteral::Int(1))
        ));
        match get("INT_TO_FLOAT").literal {
            Some(ModuleStaticLiteral::Float(bits)) => assert_eq!(f64::from_bits(bits), 3.0),
            other => panic!("expected INT_TO_FLOAT to carry Float literal, got {other:?}"),
        }
        assert!(matches!(
            get("FLOAT_TO_INT").literal,
            Some(ModuleStaticLiteral::Int(3))
        ));
        match get("FLOAT_TO_F32").literal {
            Some(ModuleStaticLiteral::Float(bits)) => {
                assert_eq!(f64::from_bits(bits), 1.1f32 as f64);
            }
            other => panic!("expected FLOAT_TO_F32 to carry Float literal, got {other:?}"),
        }
        assert!(get("UNREPRESENTABLE_U64").literal.is_none());
        assert!(get("INVALID_BOOL_CAST").literal.is_none());
    }

    #[test]
    fn collect_module_statics_resolves_module_global_literal_bindings() {
        let parsed = parse_source(
            r#"
            pub const BASE: i64 = 40;
            pub const DERIVED: i64 = BASE + 2;
            pub static IMMUTABLE_BASE: i64 = 5;
            pub const FROM_STATIC: i64 = IMMUTABLE_BASE * 3;
            pub const FORWARD: i64 = LATER + 1;
            pub const FORWARD_CHAIN: i64 = FORWARD + 1;
            pub const LATER: i64 = 9;
            mod nested {
                pub const INNER_BASE: i64 = 7;
                pub const INNER_DERIVED: i64 = INNER_BASE + 1;
                pub const INNER_FORWARD: i64 = INNER_LATER + 1;
                pub const INNER_LATER: i64 = 11;
                pub const FROM_OUTER: i64 = BASE + 1;
                pub const MULTI_SEGMENT: i64 = super::BASE + 1;
            }
        "#,
        );

        let get = |prefix: &str, name: &str| {
            parsed
                .module_statics
                .get(&(prefix.to_string(), name.to_string()))
                .unwrap_or_else(|| panic!("{prefix}::{name} collected"))
        };
        assert!(matches!(
            get("", "DERIVED").literal,
            Some(ModuleStaticLiteral::Int(42))
        ));
        assert!(matches!(
            get("", "FROM_STATIC").literal,
            Some(ModuleStaticLiteral::Int(15))
        ));
        assert!(matches!(
            get("", "FORWARD").literal,
            Some(ModuleStaticLiteral::Int(10))
        ));
        assert!(matches!(
            get("", "FORWARD_CHAIN").literal,
            Some(ModuleStaticLiteral::Int(11))
        ));
        assert!(matches!(
            get("nested", "INNER_DERIVED").literal,
            Some(ModuleStaticLiteral::Int(8))
        ));
        assert!(matches!(
            get("nested", "INNER_FORWARD").literal,
            Some(ModuleStaticLiteral::Int(12))
        ));
        assert!(get("nested", "FROM_OUTER").literal.is_none());
        assert!(matches!(
            get("nested", "MULTI_SEGMENT").literal,
            Some(ModuleStaticLiteral::Int(41))
        ));
    }

    /// PyPy parity regression guard: `static mut` storage carries a
    /// runtime-mutable slot, and `LOAD_GLOBAL` reads the current
    /// `module.__dict__` value (PyPy `flowspace/flowcontext.py:845`).
    /// Snapshotting the initial-value literal would let the frontend
    /// const-fold reads of a name whose live value has drifted, which
    /// is unsound.  The rust_source walker already excludes mutable
    /// statics (`register.rs:935-936` documented at `:488-494`); the
    /// parse-side snapshot must apply the same gate.
    #[test]
    fn collect_module_statics_skips_static_mut_initial_values() {
        let parsed = parse_source(
            r#"
            pub static IMMUTABLE_FLAG: bool = true;
            pub static mut RUNTIME_COUNTER: i64 = 0;
            pub static mut LIVE_BUFFER_LEN: usize = 16;
        "#,
        );
        assert!(
            parsed
                .module_statics
                .get(&(String::new(), "IMMUTABLE_FLAG".to_string()))
                .is_some(),
            "immutable `static` must still snapshot",
        );
        assert!(
            parsed
                .module_statics
                .get(&(String::new(), "RUNTIME_COUNTER".to_string()))
                .is_none(),
            "`static mut RUNTIME_COUNTER` must not snapshot — runtime mutation \
             breaks const-fold soundness (PyPy flowcontext.py:845 LOAD_GLOBAL)",
        );
        assert!(
            parsed
                .module_statics
                .get(&(String::new(), "LIVE_BUFFER_LEN".to_string()))
                .is_none(),
            "`static mut LIVE_BUFFER_LEN` must not snapshot",
        );
    }

    #[test]
    fn extract_function_call_identity_preserves_path_segments() {
        let expr: syn::Expr = syn::parse_quote!(crate::runtime::exec_build_list(frame, 1));
        let call = match expr {
            syn::Expr::Call(call) => call,
            _ => panic!("expected call expr"),
        };

        let identity = extract_function_call_identity(&call.func);
        assert_eq!(
            identity,
            ExtractedHandlerCall::FunctionPath(CallPath::from_segments([
                "crate",
                "runtime",
                "exec_build_list",
            ]))
        );
    }

    #[test]
    fn extract_receiver_trait_bindings_from_execute_opcode_step() {
        let parsed = parse_source(
            r#"
            pub trait OpcodeStepExecutor { fn load_fast_checked(&mut self, idx: usize); }
            pub fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, idx: usize) {
                executor.load_fast_checked(idx);
            }
        "#,
        );

        let bindings = extract_opcode_dispatch_receiver_traits(&parsed);
        assert_eq!(
            bindings.traits_by_receiver.get("executor"),
            Some(&vec!["OpcodeStepExecutor".to_string()])
        );
    }

    /// The pyre-interpreter `execute_opcode_step` signature uses a single
    /// direct bound (`<E: OpcodeStepExecutor>`) plus a `where E: Trait + Trait
    /// + ...` clause listing every handler trait. The extractor must collect
    /// both sources so downstream resolution (callcontrol) can map trait
    /// method calls to impl graphs.
    #[test]
    fn extract_receiver_trait_bindings_handles_where_clause_bounds() {
        let parsed = parse_source(
            r#"
            pub trait OpcodeStepExecutor {}
            pub trait SharedOpcodeHandler {}
            pub trait ConstantOpcodeHandler {}
            pub trait LocalOpcodeHandler {}
            pub trait NamespaceOpcodeHandler {}
            pub trait StackOpcodeHandler {}
            pub trait IterOpcodeHandler {}
            pub trait TruthOpcodeHandler {}
            pub trait ControlFlowOpcodeHandler {}
            pub trait BranchOpcodeHandler {}
            pub trait ArithmeticOpcodeHandler {}

            pub fn execute_opcode_step<E: OpcodeStepExecutor>(
                executor: &mut E,
            ) -> ()
            where
                E: SharedOpcodeHandler
                    + ConstantOpcodeHandler
                    + LocalOpcodeHandler
                    + NamespaceOpcodeHandler
                    + StackOpcodeHandler
                    + IterOpcodeHandler
                    + TruthOpcodeHandler
                    + ControlFlowOpcodeHandler
                    + BranchOpcodeHandler
                    + ArithmeticOpcodeHandler,
            {
                let _ = executor;
            }
        "#,
        );

        let bindings = extract_opcode_dispatch_receiver_traits(&parsed);
        let traits = bindings
            .traits_by_receiver
            .get("executor")
            .expect("executor receiver binding");
        let expected = [
            "OpcodeStepExecutor",
            "SharedOpcodeHandler",
            "ConstantOpcodeHandler",
            "LocalOpcodeHandler",
            "NamespaceOpcodeHandler",
            "StackOpcodeHandler",
            "IterOpcodeHandler",
            "TruthOpcodeHandler",
            "ControlFlowOpcodeHandler",
            "BranchOpcodeHandler",
            "ArithmeticOpcodeHandler",
        ];
        for name in expected {
            assert!(
                traits.iter().any(|t| t == name),
                "expected trait `{}` in executor bindings, got {:?}",
                name,
                traits
            );
        }
    }

    #[test]
    fn extract_trait_impls_qualify_nested_trait_names() {
        let parsed = parse_source(
            r#"
            mod a {
                pub trait Handler {
                    fn run(&mut self) {}
                }
                pub struct A;
                impl Handler for A {}
            }
            mod b {
                pub trait Handler {
                    fn run(&mut self) {}
                }
                pub struct B;
                impl Handler for B {}
            }
        "#,
        );
        let impls = extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let trait_names: std::collections::HashSet<&str> =
            impls.iter().map(|imp| imp.trait_name.as_str()).collect();
        assert!(trait_names.contains("a::Handler"));
        assert!(trait_names.contains("b::Handler"));
    }

    #[test]
    fn extract_trait_default_methods_include_graphs() {
        let parsed = parse_source(
            r#"
            trait Foo {
                fn helper(&mut self, x: i64) -> i64 {
                    x + 1
                }
            }
        "#,
        );
        let impls = extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let helper = impls[0]
            .methods
            .iter()
            .find(|m| m.name == "helper")
            .expect("helper method");
        assert!(
            helper.graph.is_some(),
            "trait default method should carry graph"
        );
    }

    #[test]
    fn extract_opcode_dispatch_selector_uses_exact_variant_path() {
        let pat: syn::Pat = syn::parse_quote!(Instruction::LoadFast { var_num });
        let selector = extract_opcode_dispatch_selector(&pat);
        assert_eq!(
            selector,
            OpcodeDispatchSelector::Path(CallPath::from_segments(["Instruction", "LoadFast",]))
        );
        assert_eq!(selector.canonical_key(), "Instruction::LoadFast");
    }

    #[test]
    fn extract_opcode_dispatch_selector_preserves_or_cases() {
        let pat: syn::Pat = syn::parse_quote!(Instruction::LoadFast | Instruction::StoreFast);
        let selector = extract_opcode_dispatch_selector(&pat);
        assert_eq!(
            selector.canonical_key(),
            "Instruction::LoadFast | Instruction::StoreFast"
        );
    }

    #[test]
    fn find_function_uses_canonical_parse_surface() {
        let parsed = parse_source(
            r#"
            fn helper() {}
            fn mainloop() {}
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        assert_eq!(func.sig.ident, "mainloop");
    }

    #[test]
    fn find_opcode_match_uses_canonical_parse_surface() {
        let parsed = parse_source(
            r#"
            fn mainloop() {
                match op {
                    OP_ADD => {},
                    _ => {},
                }
            }
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 2);
    }

    #[test]
    fn find_opcode_match_finds_nested_dispatch() {
        let parsed = parse_source(
            r#"
            fn mainloop() {
                loop {
                    match op {
                        OP_ADD => {},
                        OP_SUB => {},
                        _ => {},
                    }
                }
            }
        "#,
        );
        let func = find_function(&parsed, "mainloop").expect("mainloop");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 3);
    }

    #[test]
    fn find_opcode_match_accepts_instruction_enum_dispatch() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction) {
                match inst {
                    Instruction::LoadConst { idx } => {}
                    Instruction::Add => {}
                    _ => {}
                }
            }
        "#,
        );
        let func = find_function(&parsed, "execute_opcode_step").expect("execute_opcode_step");
        let opcode_match = find_opcode_match(func).expect("opcode match");
        assert_eq!(opcode_match.arms.len(), 3);
    }

    #[test]
    fn extract_opcode_dispatch_arms_ignores_nested_matches_in_arm_bodies() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction, x: i64) {
                match inst {
                    Instruction::Copy => {
                        match x {
                            Instruction::Copy => {}
                            _ => {}
                        }
                    }
                    Instruction::YieldValue => {}
                    _ => {}
                }
            }
        "#,
        );
        let arms = extract_opcode_dispatch_arms(&parsed, &std::collections::HashMap::new());
        let selectors: Vec<String> = arms
            .iter()
            .map(|arm| arm.selector.canonical_key())
            .collect();
        assert_eq!(
            selectors,
            vec![
                "Instruction::Copy".to_string(),
                "Instruction::YieldValue".to_string(),
                "_".to_string(),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "duplicate opcode dispatch selector `Instruction::Copy`")]
    fn extract_opcode_dispatch_arms_rejects_duplicate_selectors() {
        let parsed = parse_source(
            r#"
            fn execute_opcode_step(inst: Instruction) {
                match inst {
                    Instruction::Copy => {}
                    Instruction::Copy => {}
                    _ => {}
                }
            }
        "#,
        );
        let _ = extract_opcode_dispatch_arms(&parsed, &std::collections::HashMap::new());
    }
}
