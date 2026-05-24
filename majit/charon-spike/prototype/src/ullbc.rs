// Many fields below exist solely so serde can decode the JSON shape;
// they are intentionally unread in the spike — the canonical text form
// projects through accessor helpers.
#![allow(dead_code)]
//! Minimal serde decode of the subset of Charon ULLBC the prototype needs.
//!
//! Schema is reverse-engineered from `corpus.ullbc` (charon 0.1.196,
//! `--ullbc` output).  Many enums in the real schema are wider; we only
//! pull in the variants the corpus actually exercises.  Unknown variants
//! are accepted as `serde_json::Value` so the prototype keeps loading
//! when Charon adds new kinds — the lowering pass then fails loudly on
//! a kind it does not understand.

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub struct LlbcFile {
    pub charon_version: String,
    pub has_errors: bool,
    pub translated: Translated,
}

#[derive(Debug, Deserialize)]
pub struct Translated {
    pub crate_name: String,
    pub fun_decls: Vec<Option<FunDecl>>,
}

#[derive(Debug, Deserialize)]
pub struct FunDecl {
    pub def_id: u64,
    pub item_meta: ItemMeta,
    pub signature: Signature,
    /// Charon writes `body: null` for opaque references and either
    /// `{"Unstructured": {...}}`, `{"Structured": {...}}`, or
    /// `{"Error": {...}}` otherwise.  We keep it untyped here and
    /// project to [`Unstructured`] via [`FunDecl::unstructured`] so
    /// schema drift in the other variants doesn't break parsing.
    pub body: Option<Value>,
}

impl FunDecl {
    pub fn unstructured(&self) -> Option<Unstructured> {
        let body = self.body.as_ref()?;
        let inner = body.as_object()?.get("Unstructured")?;
        match serde_json::from_value::<Unstructured>(inner.clone()) {
            Ok(u) => Some(u),
            Err(e) => {
                eprintln!(
                    "warning: failed to parse Unstructured body of {}: {}",
                    self.item_meta.name_path(),
                    e
                );
                None
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ItemMeta {
    pub name: Vec<NameSeg>,
    pub span: Span,
    pub source_text: Option<String>,
}

impl ItemMeta {
    pub fn name_path(&self) -> String {
        let mut out = String::new();
        for (i, seg) in self.name.iter().enumerate() {
            if i > 0 {
                out.push_str("::");
            }
            match seg {
                NameSeg::Ident { ident: (s, _) } => out.push_str(s),
                NameSeg::Other(_) => out.push_str("<?>"),
            }
        }
        out
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum NameSeg {
    Ident {
        #[serde(rename = "Ident")]
        ident: (String, u64),
    },
    Other(Value),
}

#[derive(Debug, Deserialize)]
pub struct Span {
    pub data: SpanData,
}

#[derive(Debug, Deserialize)]
pub struct SpanData {
    pub file_id: u64,
    pub beg: Loc,
    pub end: Loc,
}

#[derive(Debug, Deserialize)]
pub struct Loc {
    pub line: u64,
    pub col: u64,
}

#[derive(Debug, Deserialize)]
pub struct Signature {
    pub is_unsafe: bool,
    pub inputs: Vec<TyRef>,
    pub output: TyRef,
}

/// Type reference. Real schema has more variants (HashConsedValue inline,
/// literal forms, ADT, etc.); we accept anything as opaque JSON and only
/// pull the `Deduplicated` id when we need to print a stable label.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TyRef {
    Deduplicated { #[serde(rename = "Deduplicated")] id: u64 },
    Other(Value),
}

impl TyRef {
    pub fn label(&self) -> String {
        match self {
            TyRef::Deduplicated { id } => format!("ty#{id}"),
            TyRef::Other(v) => {
                if let Some(obj) = v.as_object() {
                    let key = obj.keys().next().map(String::as_str).unwrap_or("?");
                    format!("ty<{key}>")
                } else {
                    "ty<?>".into()
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Unstructured {
    pub locals: Locals,
    pub body: Vec<BasicBlock>,
}

#[derive(Debug, Deserialize)]
pub struct Locals {
    pub arg_count: u64,
    pub locals: Vec<Local>,
}

#[derive(Debug, Deserialize)]
pub struct Local {
    pub index: u64,
    pub name: Option<String>,
    pub ty: TyRef,
}

#[derive(Debug, Deserialize)]
pub struct BasicBlock {
    pub statements: Vec<Statement>,
    /// Raw terminator JSON.  We project to [`TermKind`] via
    /// [`BasicBlock::term`] so a parse error on one terminator does not
    /// prevent the rest of the function from loading.
    pub terminator: Value,
}

impl BasicBlock {
    pub fn term(&self) -> Result<TermKind, String> {
        let kind = self
            .terminator
            .as_object()
            .and_then(|m| m.get("kind"))
            .ok_or_else(|| "terminator has no kind".to_string())?;
        serde_json::from_value::<TermKind>(kind.clone())
            .map_err(|e| format!("{e} (raw kind: {kind})"))
    }
}

#[derive(Debug, Deserialize)]
pub struct Statement {
    /// Raw statement-kind JSON.  Project to [`StmtKind`] via
    /// [`Statement::stmt_kind`] so the prototype keeps loading even when
    /// a statement variant is new.
    pub kind: Value,
}

impl Statement {
    pub fn stmt_kind(&self) -> Result<StmtKind, String> {
        serde_json::from_value::<StmtKind>(self.kind.clone())
            .map_err(|e| format!("{e} (raw kind: {})", self.kind))
    }
}

#[derive(Debug, Deserialize)]
pub enum StmtKind {
    StorageLive(u64),
    StorageDead(u64),
    Assign(Place, Rvalue),
    Assert(Value),
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub struct Place {
    pub kind: PlaceKind,
    pub ty: TyRef,
}

#[derive(Debug, Deserialize)]
pub enum PlaceKind {
    Local(u64),
    Projection(Box<Place>, Value),
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub enum Rvalue {
    Use(Operand),
    BinaryOp(String, Operand, Operand),
    UnaryOp(Value, Operand),
    Ref {
        place: Place,
        kind: Value,
        ptr_metadata: Value,
    },
    /// `Aggregate(kind, operands)` — constructs tuples, structs, enum
    /// variants, arrays.  We only inspect the variant tag for canonical
    /// printing.
    Aggregate(Value, Vec<Operand>),
    Discriminant(Place),
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Const(Value),
}


#[derive(Debug, Deserialize)]
pub enum TermKind {
    Goto { target: u64 },
    Switch { discr: Operand, targets: SwitchTargets },
    Call(Value),
    Assert {
        assert: Value,
        target: u64,
        on_unwind: u64,
    },
    Drop {
        target: u64,
        on_unwind: u64,
        #[serde(flatten)]
        rest: Value,
    },
    Return,
    Abort(Value),
    UnwindResume,
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
pub enum SwitchTargets {
    If(u64, u64),
    SwitchInt(Value, Vec<(Value, u64)>, u64),
}

impl LlbcFile {
    pub fn local_fn<'a>(&'a self, name: &str) -> Option<&'a FunDecl> {
        for f in self.translated.fun_decls.iter().flatten() {
            let path = f.item_meta.name_path();
            if path.ends_with(&format!("::{name}")) || path == name {
                return Some(f);
            }
        }
        None
    }
}
