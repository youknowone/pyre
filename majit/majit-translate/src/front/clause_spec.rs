//! One graph per concrete generic instantiation.
//!
//! `rpython/annotator/specialize.py` `specialize_argtype` /
//! `default_specialize` and `description.py` `FunctionDesc.cachedgraph`
//! build a distinct flow graph for each instantiation, named
//! `func__<key>`. Charon extracts with `monomorphize: false`, so a
//! generic body arrives once and its `CallKind::Trait` obligations are
//! `Clause` refs. A call whose `generics.trait_refs` are already
//! `TraitImpl` ids supplies the binding: the copy substitutes each
//! depth-0 `Clause` with that ref, and a `ParentClause` of a resolved
//! impl reads that impl's `implied_trait_refs`. The unspecialized body
//! is left unchanged.

use std::collections::{HashMap, VecDeque};

use majit_charon_reader::ullbc::{Signature, TyRef};
use majit_charon_reader::{FunDecl, Llbc, Unstructured};
use serde::Deserialize;
use serde_json::Value;

/// One specialized copy waiting to be lowered.
#[derive(Clone)]
pub(crate) struct SpecRequest {
    pub fn_id: u64,
    /// Last path segment. Carries the full instantiation key because a
    /// jitcode name keeps only that segment.
    pub leaf: String,
    pub trait_refs: Vec<Value>,
    /// `generics.types` of the call. Depth-0 `TypeVar`s are already rejected.
    pub types: Vec<Value>,
    /// `generics.const_generics` of the call.
    pub const_generics: Vec<Value>,
}

/// Instantiations reached from concrete call sites. The set is the
/// cache: a key is recorded once, when the call is first seen.
pub(crate) struct SpecQueue {
    pending: VecDeque<SpecRequest>,
    /// First request that claimed each leaf. A later request for the same
    /// leaf is the same instantiation; one from another function is a clash.
    seen: HashMap<String, SpecRequest>,
    /// `def_id`s whose body contains a depth-0 `Clause`.
    clause_body: std::collections::HashMap<u64, bool>,
}

impl SpecQueue {
    pub(crate) fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            seen: HashMap::new(),
            clause_body: std::collections::HashMap::new(),
        }
    }

    /// True when copying the body can bind a `Clause`. A generic callee
    /// whose body never names one stays on the unspecialized graph.
    pub(crate) fn body_has_own_clause(&mut self, fd: &FunDecl, llbc: &Llbc) -> bool {
        if let Some(known) = self.clause_body.get(&fd.def_id) {
            return *known;
        }
        let has = fd.body.as_ref().is_some_and(|raw| {
            let Ok(value) = serde_json::from_str::<Value>(raw.get()) else {
                return false;
            };
            let Some(body) = value.get("Unstructured") else {
                return false;
            };
            mentions_own_clause(body, llbc)
        });
        self.clause_body.insert(fd.def_id, has);
        has
    }

    pub(crate) fn enqueue(&mut self, req: SpecRequest) -> bool {
        // The leaf carries the hash of the rendered key, so a second request
        // under it is the same instantiation even when its JSON spells a type
        // through a different dedup id. Only a different function is a clash.
        if let Some(first) = self.seen.get(&req.leaf) {
            if first.fn_id != req.fn_id {
                panic!(
                    "specialized leaf {} claimed by two functions ({} and {})",
                    req.leaf, first.fn_id, req.fn_id
                );
            }
            return false;
        }
        self.seen.insert(req.leaf.clone(), req.clone());
        self.pending.push_back(req);
        true
    }

    pub(crate) fn pop(&mut self) -> Option<SpecRequest> {
        self.pending.pop_front()
    }
}

/// The declaration carries an `Unstructured` body in this LLBC, so there
/// is a graph to copy. An opaque declaration has none.
pub(crate) fn decl_has_unstructured_body(fd: &FunDecl) -> bool {
    fd.body.as_ref().is_some_and(|raw| {
        raw.get()
            .trim_start()
            .strip_prefix('{')
            .is_some_and(|rest| rest.trim_start().starts_with("\"Unstructured\""))
    })
}

/// The declaration has type parameters or trait clauses, so one shared
/// body is not one instantiation.
pub(crate) fn decl_is_generic(fd: &FunDecl) -> bool {
    let Some(generics) = fd.generics.as_ref().and_then(Value::as_object) else {
        return false;
    };
    let nonempty = |key: &str| {
        generics
            .get(key)
            .and_then(Value::as_array)
            .is_some_and(|items| !items.is_empty())
    };
    nonempty("types") || nonempty("trait_clauses") || nonempty("const_generics")
}

/// `generics.trait_refs` when every ref is resolved (no `Clause`),
/// including an empty list. Used while lowering a spec copy, where a
/// callee may have no trait clauses.
pub(crate) fn concrete_trait_refs_or_empty(generics: &Value, llbc: &Llbc) -> Option<Vec<Value>> {
    let refs = generics
        .get("trait_refs")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if refs
        .iter()
        .any(|tref| matches!(ref_class(tref, llbc, 0), RefClass::Clause))
    {
        return None;
    }
    Some(refs)
}

/// `generics.trait_refs` when every ref is resolved (no `Clause`) and at
/// least one is a `TraitImpl`. `None` leaves the callee unspecialized.
pub(crate) fn concrete_trait_refs(generics: &Value, llbc: &Llbc) -> Option<Vec<Value>> {
    let refs = generics.get("trait_refs")?.as_array()?;
    if refs.is_empty() {
        return None;
    }
    let mut saw_impl = false;
    for tref in refs {
        match ref_class(tref, llbc, 0) {
            RefClass::Clause => return None,
            RefClass::TraitImpl => saw_impl = true,
            RefClass::Other => {}
        }
    }
    if !saw_impl {
        return None;
    }
    Some(refs.clone())
}

/// `{leaf}__spec_{readable}_{hash}`. `FunctionDesc.cachedgraph` names the
/// copy `name__valid_identifier(nameof(key))` from the key's names.
/// `readable` is those type-argument leaves; `hash` is FNV-1a of the
/// function path, trait refs, types and const generics, so the spelling
/// carries no extraction-local id.
pub(crate) fn spec_leaf(leaf: &str, fn_id: u64, generics: &Value, llbc: &Llbc) -> String {
    let fn_name = llbc
        .fn_by_id(fn_id)
        .map(|fd| spec_fn_name(&fd.item_meta, llbc))
        .unwrap_or_else(|| leaf.to_string());
    let trait_refs = generics
        .get("trait_refs")
        .and_then(Value::as_array)
        .map(|refs| {
            refs.iter()
                .map(|tref| spec_trait_ref_name(tref, llbc))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let types = generics
        .get("types")
        .and_then(Value::as_array)
        .map(|types| {
            types
                .iter()
                .map(|ty| spec_type_name(ty, llbc, 0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let const_generics = generics
        .get("const_generics")
        .and_then(Value::as_array)
        .map(|consts| {
            consts
                .iter()
                .map(|cg| spec_const_name(cg, llbc, 0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let key = canon_key(&fn_name, &trait_refs, &types, &const_generics);
    let hash = fnv1a64(key.as_bytes());
    let readable = readable_type_leaves(&types);
    if readable.is_empty() {
        format!("{leaf}__spec_{hash:016x}")
    } else {
        format!("{leaf}__spec_{readable}_{hash:016x}")
    }
}

/// The leaf `spec_leaf` was given. A specialized graph is a copy of the
/// same function (`FunctionDesc.cachedgraph`), so a pass that recognizes a
/// callee by its leaf sees through the `__spec_` marker.
pub(crate) fn unspecialized_leaf(leaf: &str) -> &str {
    match leaf.find("__spec_") {
        Some(at) => &leaf[..at],
        None => leaf,
    }
}

/// `generics.types` / `generics.const_generics` when neither list contains
/// a depth-0 type or const-generic variable. `None` leaves the callee
/// unspecialized.
pub(crate) fn concrete_type_args(
    generics: &Value,
    llbc: &Llbc,
) -> Option<(Vec<Value>, Vec<Value>)> {
    let types = generics
        .get("types")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let const_generics = generics
        .get("const_generics")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let open = |items: &[Value]| items.iter().any(|item| contains_depth0_var(item, llbc, 0));
    if open(&types) || open(&const_generics) {
        return None;
    }
    Some((types, const_generics))
}

/// Copy `fd`'s unstructured body with each depth-0 `Clause` replaced by
/// `trait_refs[index]` and each depth-0 type / const-generic variable
/// replaced by the call's arguments.
pub(crate) fn substituted_unstructured(
    fd: &FunDecl,
    llbc: &Llbc,
    trait_refs: &[Value],
    types: &[Value],
    const_generics: &[Value],
) -> Option<Unstructured> {
    let raw = fd.body.as_ref()?.get();
    let mut value: Value = serde_json::from_str(raw).ok()?;
    let body = value.get_mut("Unstructured")?;
    // Type variables first. `subst_vars` replaces a wrapper with a plain
    // copy of its body; doing that after the clause pass would restore
    // the shared dedup body and put the `Clause` back.
    substitute_type_vars(body, llbc, types, const_generics);
    substitute_clauses(body, llbc, trait_refs);
    #[derive(Deserialize)]
    struct Proj {
        #[serde(rename = "Unstructured")]
        unstructured: Unstructured,
    }
    serde_json::from_value::<Proj>(value)
        .ok()
        .map(|proj| proj.unstructured)
}

/// Trait-impl id of a resolved trait call. `None` for a `Clause` ref.
pub(crate) fn resolved_trait_impl_id(payload: &Value, llbc: &Llbc) -> Option<u64> {
    let arr = payload.as_array()?;
    let resolved = resolve_trait_ref(arr.first()?, llbc, 0)?;
    trait_impl_id(&resolved, llbc, 0)
}

/// The impl method a resolved trait call names, plus the `trait_refs`
/// that instantiate that method. `None` when the ref is still a clause.
pub(crate) fn trait_impl_method(payload: &Value, llbc: &Llbc) -> Option<(u64, Value)> {
    let arr = payload.as_array()?;
    let resolved = resolve_trait_ref(arr.first()?, llbc, 0)?;
    let impl_id = trait_impl_id(&resolved, llbc, 0)?;
    let decl_id = arr.get(2)?.as_u64()?;
    let method_idx = arr.get(1)?.as_u64()?;
    let fn_id = impl_method_fn_id(llbc, impl_id, decl_id, method_idx)?;
    let generics = resolved
        .pointer("/kind/TraitImpl/generics")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    Some((fn_id, generics))
}

fn impl_method_fn_id(llbc: &Llbc, impl_id: u64, decl_id: u64, method_idx: u64) -> Option<u64> {
    let methods = llbc
        .trait_impls_raw()
        .get(impl_id as usize)?
        .get("methods")?
        .as_array()?;
    for method in methods {
        let Some(kind) = method
            .get("kind")
            .and_then(|kind| kind.get("TraitMethod"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        let Some(id) = kind.first().and_then(Value::as_u64) else {
            continue;
        };
        if id == decl_id {
            return method.get("skip_binder")?.get("id")?.as_u64();
        }
    }
    methods
        .get(method_idx as usize)?
        .get("skip_binder")?
        .get("id")?
        .as_u64()
}

enum RefClass {
    Clause,
    TraitImpl,
    Other,
}

fn ref_class(v: &Value, llbc: &Llbc, depth: usize) -> RefClass {
    if depth > 8 {
        return RefClass::Other;
    }
    let Some(obj) = unwrap_ref(v, llbc, depth) else {
        return RefClass::Other;
    };
    let Some(kind) = obj.get("kind") else {
        return RefClass::Other;
    };
    if kind.get("Clause").is_some() {
        return RefClass::Clause;
    }
    if kind.get("TraitImpl").is_some() {
        return RefClass::TraitImpl;
    }
    if kind.get("ParentClause").is_some() {
        return match resolve_trait_ref(v, llbc, depth) {
            Some(resolved) => ref_class(&resolved, llbc, depth + 1),
            None => RefClass::Other,
        };
    }
    RefClass::Other
}

fn trait_impl_id(v: &Value, llbc: &Llbc, depth: usize) -> Option<u64> {
    if depth > 8 {
        return None;
    }
    let obj = unwrap_ref(v, llbc, depth)?;
    obj.get("kind")?.get("TraitImpl")?.get("id")?.as_u64()
}

fn resolve_trait_ref(v: &Value, llbc: &Llbc, depth: usize) -> Option<Value> {
    if depth > 8 {
        return None;
    }
    let obj = unwrap_ref(v, llbc, depth)?.clone();
    let Some(parent) = obj.get("kind").and_then(|kind| kind.get("ParentClause")) else {
        return Some(obj);
    };
    let pair = parent.as_array()?;
    let parent_ref = resolve_trait_ref(pair.first()?, llbc, depth + 1)?;
    let index = pair.get(1)?.as_u64()? as usize;
    let impl_id = trait_impl_id(&parent_ref, llbc, 0)?;
    let implied = llbc
        .trait_impls_raw()
        .get(impl_id as usize)?
        .get("implied_trait_refs")?
        .as_array()?;
    resolve_trait_ref(implied.get(index)?, llbc, depth + 1)
}

fn unwrap_ref<'a>(v: &'a Value, llbc: &'a Llbc, depth: usize) -> Option<&'a Value> {
    if depth > 8 {
        return None;
    }
    let obj = v.as_object()?;
    if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
        return unwrap_ref(llbc.dedup_body(id)?, llbc, depth + 1);
    }
    if let Some(arr) = obj.get("HashConsedValue").and_then(Value::as_array)
        && arr.len() == 2
    {
        return unwrap_ref(&arr[1], llbc, depth + 1);
    }
    Some(v)
}

fn clause_index(v: &Value, llbc: &Llbc) -> Option<usize> {
    let obj = unwrap_ref(v, llbc, 0)?;
    let bound = obj.get("kind")?.get("Clause")?.get("Bound")?.as_array()?;
    if bound.first()?.as_u64()? != 0 {
        return None;
    }
    Some(bound.get(1)?.as_u64()? as usize)
}

fn mentions_own_clause(v: &Value, llbc: &Llbc) -> bool {
    mentions_own_clause_at(v, llbc, 0)
}

fn mentions_own_clause_at(v: &Value, llbc: &Llbc, depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    if clause_index(v, llbc).is_some() {
        return true;
    }
    if let Some(body) = indirect_body(v, llbc) {
        return mentions_own_clause_at(&body, llbc, depth + 1);
    }
    match v {
        Value::Array(items) => items
            .iter()
            .any(|item| mentions_own_clause_at(item, llbc, depth + 1)),
        Value::Object(map) => map
            .values()
            .any(|item| mentions_own_clause_at(item, llbc, depth + 1)),
        _ => false,
    }
}

/// Replace depth-0 type variables with `types[i]` and depth-0 const-generic
/// variables with `const_generics[i]`. A `Deduplicated` / `HashConsedValue`
/// node whose resolved body contains such a variable is replaced by the
/// substituted plain node. The shared dedup table is not written.
pub(crate) fn substitute_type_vars(
    body: &mut Value,
    llbc: &Llbc,
    types: &[Value],
    const_generics: &[Value],
) {
    subst_vars(body, llbc, types, const_generics, 0);
}

/// `fd.signature` with the same depth-0 substitution as the copied body.
pub(crate) fn substituted_signature(
    sig: &Signature,
    llbc: &Llbc,
    types: &[Value],
    const_generics: &[Value],
) -> Signature {
    Signature {
        is_unsafe: sig.is_unsafe,
        inputs: sig
            .inputs
            .iter()
            .map(|ty| subst_tyref(ty, llbc, types, const_generics))
            .collect(),
        output: subst_tyref(&sig.output, llbc, types, const_generics),
    }
}

fn subst_tyref(ty: &TyRef, llbc: &Llbc, types: &[Value], const_generics: &[Value]) -> TyRef {
    let mut value = match ty {
        TyRef::Dedup { id } => serde_json::json!({ "Deduplicated": id }),
        TyRef::Inline { value: (id, body) } => {
            serde_json::json!({ "HashConsedValue": [id, body] })
        }
        TyRef::Other(body) => body.clone(),
    };
    substitute_type_vars(&mut value, llbc, types, const_generics);
    serde_json::from_value(value.clone()).unwrap_or(TyRef::Other(value))
}

fn subst_vars(v: &mut Value, llbc: &Llbc, types: &[Value], const_generics: &[Value], depth: usize) {
    if depth > 64 {
        return;
    }
    if let Some(index) = type_var_index(v)
        && let Some(replacement) = types.get(index)
    {
        *v = replacement.clone();
        return;
    }
    if let Some(index) = const_var_index(v)
        && let Some(replacement) = const_generics.get(index)
    {
        *v = replacement.clone();
        return;
    }
    if let Some(mut plain) = indirect_body_with_var(v, llbc) {
        subst_vars(&mut plain, llbc, types, const_generics, depth + 1);
        *v = plain;
        return;
    }
    match v {
        Value::Array(items) => {
            for item in items {
                subst_vars(item, llbc, types, const_generics, depth + 1);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                subst_vars(item, llbc, types, const_generics, depth + 1);
            }
        }
        _ => {}
    }
}

/// Depth-0 `TypeVar` index: `{"TypeVar":{"Bound":[0, i]}}` or
/// `{"TypeVar":{"Free": i}}`.
fn type_var_index(v: &Value) -> Option<usize> {
    let var = v.as_object()?.get("TypeVar")?;
    if let Some(bound) = var.get("Bound").and_then(Value::as_array) {
        if bound.first()?.as_u64()? != 0 {
            return None;
        }
        return Some(bound.get(1)?.as_u64()? as usize);
    }
    var.get("Free")
        .and_then(Value::as_u64)
        .map(|index| index as usize)
}

/// Depth-0 const-generic variable index:
/// `{"kind":{"Var":{"Bound":[0, i]}}}` or `{"kind":{"Var":{"Free": i}}}`.
fn const_var_index(v: &Value) -> Option<usize> {
    let var = v.as_object()?.get("kind")?.get("Var")?;
    if let Some(bound) = var.get("Bound").and_then(Value::as_array) {
        if bound.first()?.as_u64()? != 0 {
            return None;
        }
        return Some(bound.get(1)?.as_u64()? as usize);
    }
    var.get("Free")
        .and_then(Value::as_u64)
        .map(|index| index as usize)
}

fn contains_depth0_var(v: &Value, llbc: &Llbc, depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    if type_var_index(v).is_some() || const_var_index(v).is_some() {
        return true;
    }
    if let Some(body) = indirect_body(v, llbc) {
        return contains_depth0_var(&body, llbc, depth + 1);
    }
    match v {
        Value::Array(items) => items
            .iter()
            .any(|item| contains_depth0_var(item, llbc, depth + 1)),
        Value::Object(map) => map
            .values()
            .any(|item| contains_depth0_var(item, llbc, depth + 1)),
        _ => false,
    }
}

/// Resolved body of a dedup wrapper when that body contains a depth-0
/// variable. `None` when `v` is not a wrapper or the body has no such variable.
fn indirect_body_with_var(v: &Value, llbc: &Llbc) -> Option<Value> {
    let body = indirect_body(v, llbc)?;
    if contains_depth0_var(&body, llbc, 0) {
        Some(body)
    } else {
        None
    }
}

fn indirect_body(v: &Value, llbc: &Llbc) -> Option<Value> {
    let obj = v.as_object()?;
    if obj.len() != 1 {
        return None;
    }
    if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
        return llbc.dedup_body(id).cloned();
    }
    let arr = obj.get("HashConsedValue").and_then(Value::as_array)?;
    if arr.len() != 2 {
        return None;
    }
    if let Some(id) = arr.first().and_then(Value::as_u64)
        && let Some(body) = llbc.dedup_body(id)
    {
        return Some(body.clone());
    }
    Some(arr[1].clone())
}

fn substitute_clauses(v: &mut Value, llbc: &Llbc, trait_refs: &[Value]) {
    substitute_clauses_at(v, llbc, trait_refs, 0);
}

/// Same shape as [`subst_vars`]: a depth-0 `Clause` is replaced in place;
/// a `Deduplicated` / `HashConsedValue` wrapper whose resolved body
/// mentions one is replaced by a plain copy of that body (the shared
/// dedup table is not written) and the copy is walked. Anything else is
/// walked through its children.
fn substitute_clauses_at(v: &mut Value, llbc: &Llbc, trait_refs: &[Value], depth: usize) {
    if depth > 64 {
        return;
    }
    if let Some(index) = clause_index(v, llbc)
        && let Some(replacement) = trait_refs.get(index)
    {
        *v = replacement.clone();
        return;
    }
    if let Some(mut plain) = indirect_body_with_clause(v, llbc) {
        substitute_clauses_at(&mut plain, llbc, trait_refs, depth + 1);
        *v = plain;
        return;
    }
    match v {
        Value::Array(items) => {
            for item in items {
                substitute_clauses_at(item, llbc, trait_refs, depth + 1);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                substitute_clauses_at(item, llbc, trait_refs, depth + 1);
            }
        }
        _ => {}
    }
}

/// Resolved body of a dedup wrapper when that body mentions a depth-0
/// `Clause`. `None` when `v` is not a wrapper or the body has no such clause.
fn indirect_body_with_clause(v: &Value, llbc: &Llbc) -> Option<Value> {
    let body = indirect_body(v, llbc)?;
    if mentions_own_clause(&body, llbc) {
        Some(body)
    } else {
        None
    }
}

/// Charon type expression spelled from declaration names. Extraction-local
/// ids are followed (`Deduplicated`, `HashConsedValue`) or replaced by
/// `name_path`, never printed.
pub(crate) fn spec_type_name(v: &Value, llbc: &Llbc, depth: usize) -> String {
    if depth > 32 {
        return "deep".to_string();
    }
    if let Some(obj) = v.as_object()
        && obj.len() == 1
    {
        if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
            return match llbc.dedup_body(id) {
                Some(body) => spec_type_name(body, llbc, depth + 1),
                None => "?dedup".to_string(),
            };
        }
        if let Some(arr) = obj.get("HashConsedValue").and_then(Value::as_array)
            && arr.len() == 2
        {
            return spec_type_name(&arr[1], llbc, depth + 1);
        }
    }
    if let Some(index) = v.pointer("/TypeVar/Bound/1").and_then(Value::as_u64) {
        return format!("v{index}");
    }
    if let Some(index) = type_var_index(v) {
        return format!("v{index}");
    }
    let Some(obj) = v.as_object() else {
        return canonical_type_json(v, llbc, depth);
    };
    if let Some(lit) = obj.get("Literal") {
        return spec_literal(lit);
    }
    if let Some(r) = obj.get("Ref") {
        return spec_ref(r, llbc, depth);
    }
    if let Some(rp) = obj.get("RawPtr") {
        return spec_raw_ptr(rp, llbc, depth);
    }
    if let Some(adt) = obj.get("Adt").and_then(Value::as_object) {
        return spec_adt(adt, llbc, depth);
    }
    if let Some(arr) = obj.get("Array").and_then(Value::as_array) {
        return spec_array_pair(arr, llbc, depth);
    }
    if let Some(elem) = obj.get("Slice") {
        return format!("[{}]", spec_type_name(elem, llbc, depth + 1));
    }
    if let Some(fnptr) = obj.get("FnPtr") {
        return spec_fn_ptr(fnptr, llbc, depth);
    }
    canonical_type_json(v, llbc, depth)
}

fn spec_const_name(v: &Value, llbc: &Llbc, depth: usize) -> String {
    if let Some(n) = scalar_decimal(v) {
        return n;
    }
    spec_type_name(v, llbc, depth)
}

fn spec_trait_ref_name(v: &Value, llbc: &Llbc) -> String {
    match ref_class(v, llbc, 0) {
        RefClass::TraitImpl => {
            let Some(resolved) = resolve_trait_ref(v, llbc, 0) else {
                return "x".to_string();
            };
            let Some(id) = trait_impl_id(&resolved, llbc, 0) else {
                return "x".to_string();
            };
            render_trait_impl(llbc, id).unwrap_or_else(|| "x".to_string())
        }
        RefClass::Clause | RefClass::Other => "b".to_string(),
    }
}

/// `impl_trait`: the trait's `name_path` and its `<types>`. Self is one of
/// those generics.
/// The function's name with each impl segment spelled by what it
/// implements. `name_path` renders every impl block as `<Impl>`, so two
/// methods of the same name in sibling impls share it.
fn spec_fn_name(meta: &majit_charon_reader::ullbc::ItemMeta, llbc: &Llbc) -> String {
    use majit_charon_reader::ullbc::NameSeg;
    let segs = meta
        .name
        .iter()
        .map(|seg| match seg {
            NameSeg::Ident {
                ident: (name, disambiguator),
            } => {
                if *disambiguator > 0 {
                    format!("{name}#{disambiguator}")
                } else {
                    name.clone()
                }
            }
            NameSeg::Other(v) => {
                if let Some(id) = v.pointer("/Impl/Trait").and_then(Value::as_u64) {
                    let name = render_trait_impl(llbc, id).unwrap_or_else(|| "?".to_string());
                    return format!("<impl {name}>");
                }
                if let Some(ty) = v
                    .pointer("/Impl/Ty/skip_binder")
                    .or_else(|| v.pointer("/Impl/Ty/value"))
                {
                    return format!("<impl {}>", spec_type_name(ty, llbc, 0));
                }
                canonical_type_json(v, llbc, 0)
            }
        })
        .collect::<Vec<_>>();
    segs.join("::")
}

fn render_trait_impl(llbc: &Llbc, impl_id: u64) -> Option<String> {
    let row = llbc.trait_impls_raw().get(impl_id as usize)?;
    let impl_trait = row.get("impl_trait")?;
    let trait_id = impl_trait.get("id")?.as_u64()?;
    let name = llbc.trait_by_id(trait_id)?.item_meta.name_path();
    let types = impl_trait
        .pointer("/generics/types")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|ty| spec_type_name(ty, llbc, 0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let consts = impl_trait
        .pointer("/generics/const_generics")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|cg| spec_const_name(cg, llbc, 0))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(format!("{name}{}", angle_args(&types, &consts)))
}

fn spec_literal(lit: &Value) -> String {
    if let Some(atom) = lit.as_str() {
        return match atom {
            "Bool" => "bool".to_string(),
            "Char" => "char".to_string(),
            other => format!("lit_{other}"),
        };
    }
    if let Some(obj) = lit.as_object() {
        if let Some(int) = obj
            .get("Int")
            .or_else(|| obj.get("UInt"))
            .or_else(|| obj.get("Integer"))
            .and_then(Value::as_str)
        {
            return int.to_ascii_lowercase();
        }
        if let Some(float) = obj.get("Float").and_then(Value::as_str) {
            return match float {
                "F16" => "f16".to_string(),
                "F32" => "f32".to_string(),
                "F64" => "f64".to_string(),
                "F128" => "f128".to_string(),
                other => format!("float_{other}"),
            };
        }
    }
    lit.to_string()
}

fn spec_ref(r: &Value, llbc: &Llbc, depth: usize) -> String {
    let (ty, kind) = if let Some(arr) = r.as_array() {
        (arr.get(1), arr.get(2).and_then(Value::as_str))
    } else if let Some(obj) = r.as_object() {
        (obj.get("ty"), obj.get("kind").and_then(Value::as_str))
    } else {
        return canonical_type_json(r, llbc, depth);
    };
    let Some(ty) = ty else {
        return "ref".to_string();
    };
    let inner = spec_type_name(ty, llbc, depth + 1);
    if kind.is_some_and(|k| k.eq_ignore_ascii_case("Mut")) {
        format!("&mut {inner}")
    } else {
        format!("&{inner}")
    }
}

fn spec_raw_ptr(rp: &Value, llbc: &Llbc, depth: usize) -> String {
    let (ty, kind) = if let Some(arr) = rp.as_array().filter(|arr| arr.len() == 2) {
        (arr.first(), arr.get(1).and_then(Value::as_str))
    } else if let Some(obj) = rp.as_object() {
        (
            obj.get("ty").or_else(|| obj.get("inner")),
            obj.get("kind")
                .or_else(|| obj.get("mutability"))
                .and_then(Value::as_str),
        )
    } else {
        return canonical_type_json(rp, llbc, depth);
    };
    let Some(ty) = ty else {
        return "rawptr".to_string();
    };
    let inner = spec_type_name(ty, llbc, depth + 1);
    if kind.is_some_and(|k| k.eq_ignore_ascii_case("Mut")) {
        format!("*mut {inner}")
    } else {
        format!("*const {inner}")
    }
}

fn spec_adt(adt: &serde_json::Map<String, Value>, llbc: &Llbc, depth: usize) -> String {
    let types = generic_items(adt, "types")
        .into_iter()
        .map(|ty| spec_type_name(ty, llbc, depth + 1))
        .collect::<Vec<_>>();
    let consts = generic_items(adt, "const_generics")
        .into_iter()
        .map(|cg| spec_const_name(cg, llbc, depth + 1))
        .collect::<Vec<_>>();
    let id = adt.get("id");
    if let Some(atom) = id.and_then(Value::as_str) {
        if atom == "Tuple" {
            return match types.as_slice() {
                [] => "()".to_string(),
                [one] => format!("({one},)"),
                many => format!("({})", many.join(",")),
            };
        }
        return format!("adt_{atom}{}", angle_args(&types, &consts));
    }
    if let Some(id_obj) = id.and_then(Value::as_object) {
        if let Some(def_id) = id_obj.get("Adt").and_then(Value::as_u64) {
            let name = llbc
                .type_by_id(def_id)
                .map(|td| td.item_meta.name_path())
                .unwrap_or_else(|| "?adt".to_string());
            return format!("{name}{}", angle_args(&types, &consts));
        }
        if let Some(builtin) = id_obj.get("Builtin") {
            return spec_builtin(builtin, &types, &consts);
        }
    }
    canonical_type_json(&Value::Object(adt.clone()), llbc, depth)
}

fn spec_builtin(builtin: &Value, types: &[String], consts: &[String]) -> String {
    let name = builtin.as_str().or_else(|| {
        builtin
            .as_object()
            .and_then(|map| map.keys().next().map(String::as_str))
    });
    match name {
        Some("Box") => format!("Box{}", angle_args(types, consts)),
        Some("Slice") => match types.first() {
            Some(inner) => format!("[{inner}]"),
            None => "slice".to_string(),
        },
        Some("Str") => "str".to_string(),
        Some("Array") => {
            let elem = types.first().map(String::as_str).unwrap_or("");
            let len = consts.first().map(String::as_str).unwrap_or("N");
            format!("[{elem};{len}]")
        }
        Some(other) => format!("builtin_{other}{}", angle_args(types, consts)),
        None => format!("builtin{}", angle_args(types, consts)),
    }
}

fn spec_array_pair(arr: &[Value], llbc: &Llbc, depth: usize) -> String {
    if arr.len() == 2 {
        let elem = spec_type_name(&arr[0], llbc, depth + 1);
        let len = spec_const_name(&arr[1], llbc, depth + 1);
        format!("[{elem};{len}]")
    } else {
        canonical_type_json(&Value::Array(arr.to_vec()), llbc, depth)
    }
}

fn spec_fn_ptr(fnptr: &Value, llbc: &Llbc, depth: usize) -> String {
    let Some(sig) = fnptr.get("skip_binder").unwrap_or(fnptr).as_object() else {
        return "fn".to_string();
    };
    let inputs = sig
        .get("inputs")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(|ty| spec_type_name(ty, llbc, depth + 1))
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let output = sig
        .get("output")
        .map(|ty| spec_type_name(ty, llbc, depth + 1))
        .unwrap_or_else(|| "()".to_string());
    if sig.get("is_unsafe").and_then(Value::as_bool) == Some(true) {
        format!("unsafe fn({inputs}) -> {output}")
    } else {
        format!("fn({inputs}) -> {output}")
    }
}

fn generic_items<'a>(adt: &'a serde_json::Map<String, Value>, key: &str) -> Vec<&'a Value> {
    adt.get("generics")
        .and_then(Value::as_object)
        .and_then(|generics| generics.get(key))
        .and_then(Value::as_array)
        .map(|items| items.iter().collect())
        .unwrap_or_default()
}

fn angle_args(types: &[String], consts: &[String]) -> String {
    if types.is_empty() && consts.is_empty() {
        return String::new();
    }
    let mut args = types.to_vec();
    args.extend(consts.iter().cloned());
    format!("<{}>", args.join(","))
}

fn scalar_decimal(v: &Value) -> Option<String> {
    fn from_scalar(scalar: &Value) -> Option<String> {
        let obj = scalar.as_object()?;
        for key in ["Unsigned", "Signed"] {
            let parts = obj.get(key)?.as_array()?;
            let n = parts.last()?;
            if let Some(text) = n.as_str() {
                return Some(text.to_string());
            }
            if let Some(n) = n.as_u64() {
                return Some(n.to_string());
            }
        }
        None
    }
    if let Some(scalar) = v.get("Scalar") {
        return from_scalar(scalar);
    }
    if let Some(scalar) = v.pointer("/Value/Scalar") {
        return from_scalar(scalar);
    }
    if let Some(scalar) = v.pointer("/kind/Literal/Scalar") {
        return from_scalar(scalar);
    }
    v.pointer("/kind/Value/Scalar").and_then(from_scalar)
}

fn canonical_type_json(v: &Value, llbc: &Llbc, depth: usize) -> String {
    resolve_decl_ids(v, llbc, depth).to_string()
}

/// Compact JSON with `Deduplicated` / `HashConsedValue` followed and every
/// ADT, trait, impl and fun id replaced by that declaration's `name_path`.
fn resolve_decl_ids(v: &Value, llbc: &Llbc, depth: usize) -> Value {
    if depth > 64 {
        return Value::String("deep".into());
    }
    if let Some(obj) = v.as_object()
        && obj.len() == 1
    {
        if let Some(id) = obj.get("Deduplicated").and_then(Value::as_u64) {
            return match llbc.dedup_body(id) {
                Some(body) => resolve_decl_ids(body, llbc, depth + 1),
                None => Value::String("?dedup".into()),
            };
        }
        if let Some(arr) = obj.get("HashConsedValue").and_then(Value::as_array)
            && arr.len() == 2
        {
            return resolve_decl_ids(&arr[1], llbc, depth + 1);
        }
        if let Some(id) = obj.get("Adt").and_then(Value::as_u64) {
            return Value::String(type_path(llbc, id));
        }
        if let Some(id) = obj.get("Fun").and_then(Value::as_u64) {
            return Value::String(fun_path(llbc, id));
        }
    }
    match v {
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| resolve_decl_ids(item, llbc, depth + 1))
                .collect(),
        ),
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (key, child) in map {
                let replaced = match key.as_str() {
                    "TraitImpl" => resolve_trait_impl_value(child, llbc, depth),
                    "id" if child.as_u64().is_some() => {
                        Value::String(decl_path(llbc, child.as_u64().unwrap()))
                    }
                    "Adt" if child.as_u64().is_some() => {
                        Value::String(type_path(llbc, child.as_u64().unwrap()))
                    }
                    "Fun" | "Regular" if child.as_u64().is_some() => {
                        Value::String(fun_path(llbc, child.as_u64().unwrap()))
                    }
                    "Trait" if child.as_u64().is_some() => {
                        let id = child.as_u64().unwrap();
                        Value::String(
                            llbc.trait_by_id(id)
                                .map(|td| td.item_meta.name_path())
                                .or_else(|| render_trait_impl(llbc, id))
                                .unwrap_or_else(|| "?".to_string()),
                        )
                    }
                    "trait_decl_id" if child.as_u64().is_some() => Value::String(
                        llbc.trait_by_id(child.as_u64().unwrap())
                            .map(|td| td.item_meta.name_path())
                            .unwrap_or_else(|| "?".to_string()),
                    ),
                    _ => resolve_decl_ids(child, llbc, depth + 1),
                };
                out.insert(key.clone(), replaced);
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

fn type_path(llbc: &Llbc, id: u64) -> String {
    llbc.type_by_id(id)
        .map(|td| td.item_meta.name_path())
        .unwrap_or_else(|| "?".to_string())
}

fn fun_path(llbc: &Llbc, id: u64) -> String {
    llbc.fn_by_id(id)
        .map(|fd| fd.item_meta.name_path())
        .unwrap_or_else(|| "?".to_string())
}

fn decl_path(llbc: &Llbc, id: u64) -> String {
    if let Some(td) = llbc.trait_by_id(id) {
        return td.item_meta.name_path();
    }
    if let Some(td) = llbc.type_by_id(id) {
        return td.item_meta.name_path();
    }
    if let Some(fd) = llbc.fn_by_id(id) {
        return fd.item_meta.name_path();
    }
    render_trait_impl(llbc, id).unwrap_or_else(|| "?".to_string())
}

fn canon_key(fn_name: &str, traits: &[String], types: &[String], consts: &[String]) -> String {
    let mut out = String::new();
    let fn_part = [fn_name.to_string()];
    push_group(&mut out, "fn", &fn_part);
    push_group(&mut out, "tr", traits);
    push_group(&mut out, "ty", types);
    push_group(&mut out, "cg", consts);
    out
}

fn resolve_trait_impl_value(v: &Value, llbc: &Llbc, depth: usize) -> Value {
    let Some(obj) = v.as_object() else {
        return resolve_decl_ids(v, llbc, depth + 1);
    };
    let mut out = serde_json::Map::new();
    for (key, child) in obj {
        if key == "id"
            && let Some(id) = child.as_u64()
        {
            let name = render_trait_impl(llbc, id).unwrap_or_else(|| "?".to_string());
            out.insert(key.clone(), Value::String(name));
            continue;
        }
        out.insert(key.clone(), resolve_decl_ids(child, llbc, depth + 1));
    }
    Value::Object(out)
}

fn push_group(out: &mut String, label: &str, parts: &[String]) {
    out.push_str(label);
    out.push('#');
    out.push_str(&parts.len().to_string());
    for part in parts {
        out.push('\u{1f}');
        out.push_str(&part.len().to_string());
        out.push(':');
        out.push_str(part);
    }
    out.push('\u{1e}');
}

fn readable_type_leaves(rendered: &[String]) -> String {
    if rendered.is_empty() {
        return String::new();
    }
    let joined = rendered
        .iter()
        .map(|name| type_leaf(name))
        .collect::<Vec<_>>()
        .join("_");
    if joined.is_empty() {
        return String::new();
    }
    let mut readable = crate::tool::sourcetools::valid_identifier(&joined);
    readable.truncate(48);
    readable
}

/// Last path segment of a rendered type, with a trailing generic list dropped.
fn type_leaf(rendered: &str) -> String {
    let segment = rendered.rsplit("::").next().unwrap_or(rendered);
    match segment.find('<') {
        Some(at) => segment[..at].to_string(),
        None => segment.to_string(),
    }
}

/// 64-bit FNV-1a. Not `DefaultHasher`: the seed must not depend on the process.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_llbc() -> Llbc {
        Llbc::from_slice(
            br#"{"charon_version":"t","has_errors":false,"translated":{"crate_name":"c","fun_decls":[],"files":[]}}"#,
        )
        .expect("empty llbc")
    }

    fn usize_const(n: &str) -> Value {
        json!({"Value": {"Scalar": {"Unsigned": ["Usize", n]}}})
    }

    /// `fn f<const N: usize>()` at `N = 4` and `N = 8` is two graphs.
    #[test]
    fn spec_leaf_distinguishes_const_generic_values() {
        let llbc = empty_llbc();
        let generics = |n: &str| json!({"regions": [], "types": [], "const_generics": [usize_const(n)], "trait_refs": []});
        let four = spec_leaf("f", 7, &generics("4"), &llbc);
        let eight = spec_leaf("f", 7, &generics("8"), &llbc);
        assert_ne!(four, eight);
        assert_eq!(four, spec_leaf("f", 7, &generics("4"), &llbc));
    }

    #[test]
    fn spec_leaf_without_const_generics_keeps_its_name() {
        let llbc = empty_llbc();
        let generics = json!({"regions": [], "types": [], "const_generics": [], "trait_refs": []});
        let name = spec_leaf("f", 7, &generics, &llbc);
        assert!(name.starts_with("f__spec_"), "{name}");
        let hash = name.strip_prefix("f__spec_").unwrap();
        assert_eq!(hash.len(), 16, "{name}");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "{name}");
        assert_eq!(name, spec_leaf("f", 7, &generics, &llbc));
    }

    #[test]
    fn unspecialized_leaf_strips_the_spec_key_only() {
        let llbc = empty_llbc();
        let generics = json!({"regions": [], "types": [], "const_generics": [usize_const("4")], "trait_refs": []});
        let leaf = spec_leaf("zero_division", 42, &generics, &llbc);
        assert!(leaf.starts_with("zero_division__spec_"), "{leaf}");
        assert_eq!(unspecialized_leaf(&leaf), "zero_division");
        assert_eq!(unspecialized_leaf("zero_division"), "zero_division");
        assert_eq!(unspecialized_leaf("a__sb_c"), "a__sb_c");
    }

    /// The same instantiation through a `Deduplicated` id and inline is one leaf.
    #[test]
    fn spec_leaf_dedup_matches_inline_body() {
        let inline = json!({"Literal": {"Int": "I64"}});
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [{
                    "def_id": 1,
                    "item_meta": {
                        "name": [{"Ident": ["fixture", 0]}, {"Ident": ["f", 0]}],
                        "span": {"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}},
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "signature": {
                        "is_unsafe": false,
                        "inputs": [],
                        "output": {"HashConsedValue": [7, inline]}
                    },
                    "body": null
                }],
                "files": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture");
        let dedup = json!({"Deduplicated": 7});
        assert_eq!(
            spec_type_name(&inline, &llbc, 0),
            spec_type_name(&dedup, &llbc, 0)
        );
        let inline_g = json!({"types": [inline], "trait_refs": [], "const_generics": []});
        let dedup_g = json!({"types": [dedup], "trait_refs": [], "const_generics": []});
        assert_eq!(
            spec_leaf("f", 1, &inline_g, &llbc),
            spec_leaf("f", 1, &dedup_g, &llbc)
        );
    }

    /// `&T` stays distinct from `T`.
    #[test]
    fn spec_leaf_keeps_ref_distinct_from_referent() {
        let llbc = empty_llbc();
        let ty = json!({"Literal": {"Int": "I64"}});
        let shared = json!({"Ref": ["Erased", ty, "Shared"]});
        let bare = json!({"types": [ty], "trait_refs": [], "const_generics": []});
        let reference = json!({"types": [shared], "trait_refs": [], "const_generics": []});
        assert_ne!(
            spec_type_name(&ty, &llbc, 0),
            spec_type_name(&shared, &llbc, 0)
        );
        assert_ne!(
            spec_leaf("f", 7, &bare, &llbc),
            spec_leaf("f", 7, &reference, &llbc)
        );
    }

    /// `fn f<const N: usize>()` is generic even with no type params or clauses,
    /// so `f::<4>` and `f::<8>` are two spec copies.
    #[test]
    fn const_only_generic_is_specialized() {
        let span = json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [{
                    "def_id": 0,
                    "item_meta": {
                        "name": [{"Ident": ["f", 0]}],
                        "span": span,
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "signature": {"is_unsafe": false, "inputs": [], "output": {"Literal": {"Int": "Usize"}}},
                    "generics": {
                        "regions": [],
                        "types": [],
                        "const_generics": [{"index": 0, "name": "N", "ty": {"Literal": "Usize"}}],
                        "trait_clauses": []
                    },
                    "body": null
                }],
                "files": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture");
        let fd = llbc.fn_by_id(0).expect("f");
        assert!(decl_is_generic(fd));
        let generics = |n: &str| json!({"regions": [], "types": [], "const_generics": [usize_const(n)], "trait_refs": []});
        assert_ne!(
            spec_leaf("f", 0, &generics("4"), &llbc),
            spec_leaf("f", 0, &generics("8"), &llbc)
        );
    }

    /// A wrapper whose shared body holds a depth-0 `TypeVar` and a depth-0
    /// `Clause` is substituted in the copy. The dedup table stays shared.
    #[test]
    fn clause_subst_follows_dedup_and_hash_cons_without_writing_the_table() {
        let span = json!({"data": {"file_id": 0, "beg": {"line": 1, "col": 0}, "end": {"line": 1, "col": 1}}});
        let generics = json!({
            "regions": [],
            "types": [{"TypeVar": {"Bound": [0, 0]}}],
            "const_generics": [],
            "trait_refs": [{"kind": {"Clause": {"Bound": [0, 0]}}}]
        });
        let i64_ty = json!({"Literal": {"Int": "I64"}});
        let impl_ref = json!({"kind": {"TraitImpl": {"id": 0, "generics": {"regions": [], "types": [i64_ty], "const_generics": [], "trait_refs": []}}}});
        let body = json!({
            "Unstructured": {
                "span": span,
                "locals": {
                    "arg_count": 0,
                    "locals": [
                        {"index": 0, "name": null, "span": span, "ty": {"Deduplicated": 11}},
                        {"index": 1, "name": null, "span": span, "ty": {"HashConsedValue": [11, generics]}}
                    ]
                },
                "body": [{"statements": [], "terminator": {"span": span, "kind": "Return"}}]
            }
        });
        let file = json!({
            "charon_version": "t",
            "has_errors": false,
            "translated": {
                "crate_name": "c",
                "fun_decls": [{
                    "def_id": 0,
                    "item_meta": {
                        "name": [{"Ident": ["f", 0]}],
                        "span": span,
                        "source_text": null,
                        "attr_info": {"attributes": [], "inline": null, "rename": null, "public": true},
                        "is_local": true
                    },
                    "signature": {"is_unsafe": false, "inputs": [], "output": {"HashConsedValue": [11, generics]}},
                    "body": body
                }],
                "files": []
            }
        });
        let llbc = Llbc::from_slice(file.to_string().as_bytes()).expect("fixture");
        let shared = llbc.dedup_body(11).expect("dedup 11").clone();
        assert!(mentions_own_clause(&shared, &llbc));
        assert!(contains_depth0_var(&shared, &llbc, 0));
        let fd = llbc.fn_by_id(0).expect("f");
        let copied =
            substituted_unstructured(fd, &llbc, &[impl_ref.clone()], &[i64_ty.clone()], &[])
                .expect("substituted body");
        for local in &copied.locals.locals {
            let majit_charon_reader::ullbc::TyRef::Other(ty) = &local.ty else {
                panic!("wrapper survived in {:?}", local.ty);
            };
            assert!(
                !value_has_depth0_type_var(ty) && !value_has_depth0_clause(ty),
                "depth-0 var or clause survived: {ty}"
            );
        }
        assert_eq!(llbc.dedup_body(11), Some(&shared));
        assert!(value_has_depth0_type_var(&shared) && value_has_depth0_clause(&shared));
    }

    fn value_has_depth0_type_var(v: &Value) -> bool {
        if type_var_index(v).is_some() {
            return true;
        }
        match v {
            Value::Array(items) => items.iter().any(value_has_depth0_type_var),
            Value::Object(map) => map.values().any(value_has_depth0_type_var),
            _ => false,
        }
    }

    fn value_has_depth0_clause(v: &Value) -> bool {
        if v.pointer("/kind/Clause/Bound/0").and_then(Value::as_u64) == Some(0) {
            return true;
        }
        match v {
            Value::Array(items) => items.iter().any(value_has_depth0_clause),
            Value::Object(map) => map.values().any(value_has_depth0_clause),
            _ => false,
        }
    }
}
