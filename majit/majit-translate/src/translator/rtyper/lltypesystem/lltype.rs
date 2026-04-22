//! RPython `rpython/rtyper/lltypesystem/lltype.py`.
//!
//! Currently ports two surfaces:
//! * Function-pointer surface consumed by `translator/simplify.py:get_graph`:
//!   [`_ptr`], [`_func`], [`FuncType`], [`functionptr`], [`getfunctionptr`],
//!   [`_getconcretetype`].
//! * [`LowLevelType`] primitive enum consumed by `rmodel.py`'s
//!   `Repr.lowleveltype` attribute and by `inputconst(reqtype, value)`.
//!   The Rust adaptation collapses upstream's class hierarchy
//!   (`LowLevelType` → `Primitive` / `Number` / `Ptr` / `Struct` / `Array`
//!   at `lltype.py:98,642,665,721,...`) into an enum so `Repr`
//!   implementations can pattern-match on kind without Rust trait-object
//!   downcasts. The three variants currently populated (`Void`, `Bool`,
//!   `Signed`, `Float`, `Char`, `UniChar`, `Unsigned`, `SingleFloat`,
//!   `LongFloat`, `SignedLongLong`, `UnsignedLongLong`, `Ptr`) cover every
//!   type used by `rpbc.py FunctionRepr` / `rclass.py InstanceRepr` /
//!   `FunctionReprBase.call` — additional container kinds (`Struct`,
//!   `Array`, `ForwardReference`) land with the commit that consumes
//!   them.

use std::hash::{Hash, Hasher};
use std::rc::Rc;

use crate::flowspace::model::{ConcretetypePlaceholder, ConstValue, GraphKey, GraphRef, Hlvalue};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DelayedPointer;

/// RPython `class LowLevelType(object)` at `lltype.py:98` + its primitive
/// subclasses at `lltype.py:642-721`.
///
/// Rust adaptation (parity rule #1 — smallest possible deviation):
///
/// * Upstream uses a class hierarchy rooted at `LowLevelType` with
///   `Primitive` / `Number` / `Ptr` / `Struct` / `Array` / etc.
///   subclasses, identified by `__class__` and distinguished by
///   `.__name__` fields (`lltype.py:701-718`: `Signed`, `Float`, `Bool`,
///   `Void`, `Char`, `UniChar`, ...). Each is an instance singleton
///   cached in module-global variables.
/// * Rust collapses the hierarchy into an enum and re-exposes the
///   singletons via `LowLevelType::Void` / `::Bool` / `::Signed` / ...
///   plus module-level constants (see [`VOID`], [`BOOL`], [`SIGNED`],
///   etc.) matching upstream's `lltype.Void` / `lltype.Bool` /
///   `lltype.Signed` import surface.
///
/// The three container variants (`Struct`, `Array`, `ForwardReference`)
/// lands with commits that consume them; today only [`LowLevelType::Ptr`]
/// pointing at a [`FuncType`] is used (pyre `rtyper::getcallable` →
/// `getfunctionptr` surface).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LowLevelType {
    /// RPython `Void = Primitive("Void", None)` (lltype.py:717).
    Void,
    /// RPython `Bool = Primitive("Bool", False)` (lltype.py:716).
    Bool,
    /// RPython `Signed = build_number("Signed", int)` (lltype.py:703).
    /// Fixed-width host-signed integer (`i64` in Rust).
    Signed,
    /// RPython `Unsigned = build_number("Unsigned", r_uint)` (lltype.py:704).
    Unsigned,
    /// RPython `SignedLongLong` (lltype.py:705).
    SignedLongLong,
    /// RPython `UnsignedLongLong` (lltype.py:707).
    UnsignedLongLong,
    /// RPython `Float = Primitive("Float", 0.0)` (lltype.py:710) — C `double`.
    Float,
    /// RPython `SingleFloat` (lltype.py:711) — C `float`.
    SingleFloat,
    /// RPython `LongFloat` (lltype.py:712) — C `long double`.
    LongFloat,
    /// RPython `Char = Primitive("Char", '\x00')` (lltype.py:715).
    Char,
    /// RPython `UniChar = Primitive("UniChar", u'\x00')` (lltype.py:718).
    UniChar,
    /// RPython `class Ptr(LowLevelType)` (lltype.py:721-759) where the
    /// pointee is a function type. Other pointee variants (`Struct`,
    /// `Array`) land with consuming commits.
    Ptr(Rc<FuncType>),
}

/// RPython `lltype.Void` singleton surface. Pure re-export of the enum
/// variant so `rmodel.rs` / `rpbc.rs` / `rclass.rs` ports can mirror
/// upstream `from rpython.rtyper.lltypesystem.lltype import Void` reads.
pub const VOID: LowLevelType = LowLevelType::Void;
/// RPython `lltype.Bool` singleton surface.
pub const BOOL: LowLevelType = LowLevelType::Bool;
/// RPython `lltype.Signed` singleton surface.
pub const SIGNED: LowLevelType = LowLevelType::Signed;
/// RPython `lltype.Float` singleton surface.
pub const FLOAT: LowLevelType = LowLevelType::Float;
/// RPython `lltype.Char` singleton surface.
pub const CHAR: LowLevelType = LowLevelType::Char;
/// RPython `lltype.UniChar` singleton surface.
pub const UNICHAR: LowLevelType = LowLevelType::UniChar;

impl LowLevelType {
    /// RPython `LowLevelType._contains_value(value)` — used by
    /// `Repr.convert_const` (`rmodel.py:122`) and by `inputconst`
    /// (`rmodel.py:390`) as the "does this low-level type admit this
    /// Python value as a prebuilt constant" check.
    ///
    /// Upstream dispatches through each subclass's `_enforce` /
    /// `_contains_value` implementation; the Rust port pattern-matches
    /// on variant + [`ConstValue`]. Returns `true` if `value` is a
    /// valid constant of kind `self`. Unsupported variants (e.g. rich
    /// container wrappers outside the covered set) conservatively
    /// accept [`ConstValue::Placeholder`] and reject everything else,
    /// matching upstream's TyperError raising surface downstream.
    pub fn contains_value(&self, value: &ConstValue) -> bool {
        // Upstream special-cases `Placeholder` (used by normalizecalls
        // sentinel `description.NODEFAULT`) as a universally acceptable
        // constant while its holder recomputes the real type. Mirror
        // that tolerance so the normalizecalls rewrite branch does not
        // trip convert_const validation during mid-pipeline rewrites.
        if matches!(value, ConstValue::Placeholder) {
            return true;
        }
        match self {
            // upstream `Void = Primitive("Void", None)` only admits
            // Python `None`.
            LowLevelType::Void => matches!(value, ConstValue::None),
            // upstream `Bool = Primitive("Bool", False)`.
            LowLevelType::Bool => matches!(value, ConstValue::Bool(_)),
            // upstream `Signed` / `Unsigned` / `SignedLongLong` /
            // `UnsignedLongLong` all accept Python `int` (with range
            // checking upstream; pyre's `ConstValue::Int` is already i64
            // so the only check left is category match).
            LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::UnsignedLongLong => matches!(value, ConstValue::Int(_)),
            // upstream `Float` / `SingleFloat` / `LongFloat` accept
            // Python `float`.
            LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat => {
                matches!(value, ConstValue::Float(_))
            }
            // upstream `Char` accepts a single-byte Python str; pyre
            // represents both as `ConstValue::Str` so additional length
            // validation belongs to convert_const callers (rmodel.py
            // does not tighten here either — TyperError triggers
            // downstream via _enforce on malformed constants).
            LowLevelType::Char => matches!(value, ConstValue::Str(_)),
            LowLevelType::UniChar => matches!(value, ConstValue::Str(_)),
            // upstream `Ptr(FuncType)` accepts `_ptr` instances — pyre's
            // `ConstValue::LLPtr` is the direct equivalent.
            LowLevelType::Ptr(_) => matches!(value, ConstValue::LLPtr(_)),
        }
    }

    /// RPython `LowLevelType.__str__` (`lltype.py:648` Primitive,
    /// `lltype.py:745` Ptr). Used by Repr's diagnostic messages
    /// (`rmodel.py:30,123`).
    pub fn short_name(&self) -> String {
        match self {
            LowLevelType::Void => "Void".to_string(),
            LowLevelType::Bool => "Bool".to_string(),
            LowLevelType::Signed => "Signed".to_string(),
            LowLevelType::Unsigned => "Unsigned".to_string(),
            LowLevelType::SignedLongLong => "SignedLongLong".to_string(),
            LowLevelType::UnsignedLongLong => "UnsignedLongLong".to_string(),
            LowLevelType::Float => "Float".to_string(),
            LowLevelType::SingleFloat => "SingleFloat".to_string(),
            LowLevelType::LongFloat => "LongFloat".to_string(),
            LowLevelType::Char => "Char".to_string(),
            LowLevelType::UniChar => "UniChar".to_string(),
            // upstream `Ptr.__str__ = "* %s" % (self.TO,)`. pyre's
            // FuncType Debug-prints the arg/result ConcretetypePlaceholder
            // list, which is coarse until the concretetype migration
            // lands. Use a stable shape so downstream tests can pattern
            // match without depending on full Debug repr.
            LowLevelType::Ptr(to) => format!("* FuncType({} args)", to.args.len()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub args: Vec<ConcretetypePlaceholder>,
    pub result: ConcretetypePlaceholder,
}

#[derive(Clone, Debug)]
pub struct _func {
    pub TYPE: FuncType,
    pub _name: String,
    pub graph: Option<usize>,
    pub _callable: Option<String>,
}

impl PartialEq for _func {
    fn eq(&self, other: &Self) -> bool {
        self.TYPE == other.TYPE
            && self._name == other._name
            && self._callable == other._callable
            && self.graph == other.graph
    }
}

impl Eq for _func {}

impl Hash for _func {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.TYPE.hash(state);
        self._name.hash(state);
        self._callable.hash(state);
        match &self.graph {
            Some(graph) => {
                true.hash(state);
                graph.hash(state);
            }
            None => false.hash(state),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct _ptr {
    pub _TYPE: FuncType,
    pub _obj0: Result<_func, DelayedPointer>,
}

impl _ptr {
    pub fn _obj(&self) -> Result<&_func, DelayedPointer> {
        self._obj0.as_ref().map_err(|_| DelayedPointer)
    }
}

pub fn functionptr(
    TYPE: FuncType,
    name: &str,
    graph: Option<usize>,
    _callable: Option<String>,
) -> _ptr {
    _ptr {
        _TYPE: TYPE.clone(),
        _obj0: Ok(_func {
            TYPE,
            _name: name.to_string(),
            graph,
            _callable,
        }),
    }
}

pub fn _getconcretetype(v: &Hlvalue) -> ConcretetypePlaceholder {
    match v {
        Hlvalue::Variable(v) => v.concretetype.unwrap_or(()),
        Hlvalue::Constant(c) => c.concretetype.unwrap_or(()),
    }
}

pub fn getfunctionptr(
    graph: &GraphRef,
    getconcretetype: fn(&Hlvalue) -> ConcretetypePlaceholder,
) -> _ptr {
    let graph_b = graph.borrow();
    let llinputs = graph_b.getargs().iter().map(getconcretetype).collect();
    let lloutput = getconcretetype(&graph_b.getreturnvar());
    let ft = FuncType {
        args: llinputs,
        result: lloutput,
    };
    let name = graph_b.name.clone();
    let callable = graph_b.func.as_ref().map(|func| func.name.clone());
    drop(graph_b);
    functionptr(ft, &name, Some(GraphKey::of(graph).as_usize()), callable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{Block, FunctionGraph};
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn functionptr_keeps_graph_on_funcobj() {
        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", start)));
        let ptr = getfunctionptr(&graph, _getconcretetype);
        let funcobj = ptr._obj().unwrap();
        assert_eq!(funcobj.graph, Some(GraphKey::of(&graph).as_usize()));
    }

    #[test]
    fn getfunctionptr_calls_getconcretetype_for_args_and_result() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLS: AtomicUsize = AtomicUsize::new(0);

        fn counting_getconcretetype(v: &Hlvalue) -> ConcretetypePlaceholder {
            let _ = v;
            CALLS.fetch_add(1, Ordering::Relaxed);
        }

        let start = Rc::new(RefCell::new(Block::new(vec![
            Hlvalue::Variable(crate::flowspace::model::Variable::new()),
            Hlvalue::Variable(crate::flowspace::model::Variable::new()),
        ])));
        let graph = Rc::new(RefCell::new(FunctionGraph::new("f", start)));
        CALLS.store(0, Ordering::Relaxed);

        let _ = getfunctionptr(&graph, counting_getconcretetype);

        assert_eq!(CALLS.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn lowleveltype_primitive_contains_value_matches_upstream_enforce() {
        use crate::flowspace::model::ConstValue;

        // RPython `Void = Primitive("Void", None)` only accepts None
        // (`lltype.py:654` — "if self is not Void and initialization
        // != 'example': return _uninitialized(self)").
        assert!(LowLevelType::Void.contains_value(&ConstValue::None));
        assert!(!LowLevelType::Void.contains_value(&ConstValue::Bool(true)));

        // RPython `Bool` Primitive accepts bools.
        assert!(LowLevelType::Bool.contains_value(&ConstValue::Bool(false)));
        assert!(LowLevelType::Bool.contains_value(&ConstValue::Bool(true)));
        assert!(!LowLevelType::Bool.contains_value(&ConstValue::Int(0)));

        // RPython `Signed` / `Unsigned` / `SignedLongLong` /
        // `UnsignedLongLong` accept Python int.
        assert!(LowLevelType::Signed.contains_value(&ConstValue::Int(42)));
        assert!(LowLevelType::Unsigned.contains_value(&ConstValue::Int(42)));
        assert!(LowLevelType::SignedLongLong.contains_value(&ConstValue::Int(42)));
        assert!(!LowLevelType::Signed.contains_value(&ConstValue::Bool(true)));

        // RPython `Float` / `SingleFloat` / `LongFloat` accept Python float.
        // pyre stores floats as u64 bit-patterns in ConstValue::Float.
        assert!(LowLevelType::Float.contains_value(&ConstValue::Float(0)));
        assert!(LowLevelType::SingleFloat.contains_value(&ConstValue::Float(0)));
        assert!(LowLevelType::LongFloat.contains_value(&ConstValue::Float(0)));
        assert!(!LowLevelType::Float.contains_value(&ConstValue::Int(0)));
    }

    #[test]
    fn lowleveltype_placeholder_value_is_universally_accepted() {
        use crate::flowspace::model::ConstValue;

        // Placeholder sentinel (`description.NODEFAULT` upstream) must
        // pass `_contains_value` so the normalizecalls rewrite branch
        // can stash it as a transient row-level padding without
        // tripping convert_const validation. See rmodel.rs's
        // inputconst port for the load-bearing use.
        for lltype in [
            LowLevelType::Void,
            LowLevelType::Bool,
            LowLevelType::Signed,
            LowLevelType::Float,
            LowLevelType::Char,
            LowLevelType::UniChar,
            LowLevelType::Ptr(Rc::new(FuncType {
                args: vec![],
                result: (),
            })),
        ] {
            assert!(
                lltype.contains_value(&ConstValue::Placeholder),
                "Placeholder must be universally acceptable (lltype={lltype:?})"
            );
        }
    }

    #[test]
    fn lowleveltype_primitive_short_name_matches_upstream_class_name() {
        // rmodel.py:30 `<%s %s>` formatter and rmodel.py:33
        // `compact_repr` both consume `lowleveltype._short_name()`
        // (Primitive) or `lowleveltype.__name__` (Ptr). Lock in the
        // upstream strings.
        assert_eq!(LowLevelType::Void.short_name(), "Void");
        assert_eq!(LowLevelType::Bool.short_name(), "Bool");
        assert_eq!(LowLevelType::Signed.short_name(), "Signed");
        assert_eq!(LowLevelType::Unsigned.short_name(), "Unsigned");
        assert_eq!(LowLevelType::SignedLongLong.short_name(), "SignedLongLong");
        assert_eq!(
            LowLevelType::UnsignedLongLong.short_name(),
            "UnsignedLongLong"
        );
        assert_eq!(LowLevelType::Float.short_name(), "Float");
        assert_eq!(LowLevelType::SingleFloat.short_name(), "SingleFloat");
        assert_eq!(LowLevelType::LongFloat.short_name(), "LongFloat");
        assert_eq!(LowLevelType::Char.short_name(), "Char");
        assert_eq!(LowLevelType::UniChar.short_name(), "UniChar");
    }

    #[test]
    fn lowleveltype_ptr_short_name_follows_upstream_prefix() {
        // upstream Ptr.__str__: `'* %s' % self.TO`. pyre stabilises the
        // FuncType body into "FuncType(N args)" for test-friendly
        // substring matching until a richer Debug impl lands.
        let ft = Rc::new(FuncType {
            args: vec![(), ()],
            result: (),
        });
        let ptr = LowLevelType::Ptr(ft);
        assert!(
            ptr.short_name().starts_with("* "),
            "Ptr short_name should use upstream '* {{TO}}' prefix; got {:?}",
            ptr.short_name()
        );
        assert!(
            ptr.short_name().contains("2 args"),
            "Ptr short_name should carry FuncType arity hint; got {:?}",
            ptr.short_name()
        );
    }

    #[test]
    fn lowleveltype_module_constants_match_variant_singletons() {
        // `VOID` / `BOOL` / `SIGNED` / `FLOAT` / `CHAR` / `UNICHAR` are
        // the pyre re-exports of upstream's `lltype.Void` / `lltype.Bool`
        // / ... (which are instance singletons). Lock the identity so
        // downstream `use lltype::{Void, Bool, ...}` imports see a
        // stable match source.
        assert_eq!(VOID, LowLevelType::Void);
        assert_eq!(BOOL, LowLevelType::Bool);
        assert_eq!(SIGNED, LowLevelType::Signed);
        assert_eq!(FLOAT, LowLevelType::Float);
        assert_eq!(CHAR, LowLevelType::Char);
        assert_eq!(UNICHAR, LowLevelType::UniChar);
    }

    #[test]
    fn delayed_pointer_raises_on_obj_access() {
        let ptr = _ptr {
            _TYPE: FuncType {
                args: vec![],
                result: (),
            },
            _obj0: Err(DelayedPointer),
        };
        assert_eq!(ptr._obj(), Err(DelayedPointer));
    }
}
