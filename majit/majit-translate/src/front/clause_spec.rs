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

use std::collections::{HashSet, VecDeque};

use majit_charon_reader::ullbc::{Signature, TyRef};
use majit_charon_reader::{FunDecl, Llbc, Unstructured};
use serde::Deserialize;
use serde_json::Value;

/// One specialized copy waiting to be lowered.
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
    seen: HashSet<String>,
    /// `def_id`s whose body contains a depth-0 `Clause` and no closure call.
    clause_body: std::collections::HashMap<u64, bool>,
}

impl SpecQueue {
    pub(crate) fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            seen: HashSet::new(),
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
            mentions_own_clause(body, llbc) && !body_calls_closure(body, llbc)
        });
        self.clause_body.insert(fd.def_id, has);
        has
    }

    pub(crate) fn enqueue(&mut self, req: SpecRequest) -> bool {
        if !self.seen.insert(req.leaf.clone()) {
            return false;
        }
        self.pending.push_back(req);
        true
    }

    pub(crate) fn pop(&mut self) -> Option<SpecRequest> {
        self.pending.pop_front()
    }
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
    nonempty("types") || nonempty("trait_clauses")
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

/// `name__s<fn>_<impl ids>_<type keys>`. Distinct from the bare leaf and
/// from every other instantiation of the same `fn`.
pub(crate) fn spec_leaf(leaf: &str, fn_id: u64, generics: &Value, llbc: &Llbc) -> String {
    let impls = generics
        .get("trait_refs")
        .and_then(Value::as_array)
        .map(|refs| {
            refs.iter()
                .map(|tref| match ref_class(tref, llbc, 0) {
                    RefClass::TraitImpl => trait_impl_id(tref, llbc, 0)
                        .map(|id| id.to_string())
                        .unwrap_or_else(|| "x".to_string()),
                    _ => "b".to_string(),
                })
                .collect::<Vec<_>>()
                .join("_")
        })
        .unwrap_or_default();
    let types = generics
        .get("types")
        .and_then(Value::as_array)
        .map(|types| types.iter().map(type_key).collect::<Vec<_>>().join("_"))
        .unwrap_or_default();
    format!("{leaf}__s{fn_id}_{impls}_{types}")
}

/// `generics.types` / `generics.const_generics` when neither list contains
/// a depth-0 `TypeVar`. `None` leaves the callee unspecialized.
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
    let open = |items: &[Value]| items.iter().any(|item| contains_depth0_type_var(item, llbc, 0));
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
    substitute_clauses(body, llbc, trait_refs);
    substitute_type_vars(body, llbc, types, const_generics);
    #[derive(Deserialize)]
    struct Proj {
        #[serde(rename = "Unstructured")]
        unstructured: Unstructured,
    }
    serde_json::from_value::<Proj>(value)
        .ok()
        .map(|proj| proj.unstructured)
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
        let kind = method.get("kind")?.get("TraitMethod")?.as_array()?;
        if kind.first()?.as_u64()? == decl_id {
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
    obj.get("kind")?
        .get("TraitImpl")?
        .get("id")?
        .as_u64()
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

fn body_calls_closure(v: &Value, llbc: &Llbc) -> bool {
    match v {
        Value::Object(map) => {
            if let Some(id) = map.get("Regular").and_then(Value::as_u64)
                && llbc.fn_by_id(id).is_some_and(|fd| fd.item_meta.name_path().contains("closure"))
            {
                return true;
            }
            map.values().any(|item| body_calls_closure(item, llbc))
        }
        Value::Array(items) => items.iter().any(|item| body_calls_closure(item, llbc)),
        _ => false,
    }
}

fn mentions_own_clause(v: &Value, llbc: &Llbc) -> bool {
    if clause_index(v, llbc).is_some() {
        return true;
    }
    match v {
        Value::Array(items) => items.iter().any(|item| mentions_own_clause(item, llbc)),
        Value::Object(map) => map.values().any(|item| mentions_own_clause(item, llbc)),
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

fn subst_vars(
    v: &mut Value,
    llbc: &Llbc,
    types: &[Value],
    const_generics: &[Value],
    depth: usize,
) {
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
    var.get("Free").and_then(Value::as_u64).map(|index| index as usize)
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
    var.get("Free").and_then(Value::as_u64).map(|index| index as usize)
}

fn contains_depth0_type_var(v: &Value, llbc: &Llbc, depth: usize) -> bool {
    if depth > 64 {
        return false;
    }
    if type_var_index(v).is_some() {
        return true;
    }
    if let Some(body) = indirect_body(v, llbc) {
        return contains_depth0_type_var(&body, llbc, depth + 1);
    }
    match v {
        Value::Array(items) => items
            .iter()
            .any(|item| contains_depth0_type_var(item, llbc, depth + 1)),
        Value::Object(map) => map
            .values()
            .any(|item| contains_depth0_type_var(item, llbc, depth + 1)),
        _ => false,
    }
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
    if let Some(index) = clause_index(v, llbc)
        && let Some(replacement) = trait_refs.get(index)
    {
        *v = replacement.clone();
        return;
    }
    match v {
        Value::Array(items) => {
            for item in items {
                substitute_clauses(item, llbc, trait_refs);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                substitute_clauses(item, llbc, trait_refs);
            }
        }
        _ => {}
    }
}

fn type_key(v: &Value) -> String {
    if let Some(id) = v
        .as_object()
        .and_then(|obj| obj.get("Deduplicated"))
        .and_then(Value::as_u64)
    {
        return format!("d{id}");
    }
    if let Some(id) = v
        .as_object()
        .and_then(|obj| obj.get("HashConsedValue"))
        .and_then(Value::as_array)
        .and_then(|arr| arr.first())
        .and_then(Value::as_u64)
    {
        return format!("d{id}");
    }
    if let Some(index) = v
        .pointer("/TypeVar/Bound/1")
        .and_then(Value::as_u64)
    {
        return format!("v{index}");
    }
    let text = v.to_string();
    let hash = text.bytes().fold(0u64, |acc, byte| {
        acc.wrapping_mul(31).wrapping_add(u64::from(byte))
    });
    format!("h{hash:x}")
}
