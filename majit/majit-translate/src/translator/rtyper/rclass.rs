//! RPython `rpython/rtyper/rclass.py`.
//!
//! Current port scope:
//!
//! * Module-level type constants `OBJECT_VTABLE`, `CLASSTYPE`, `OBJECT`,
//!   `OBJECTPTR`, `NONGCOBJECT`, `NONGCOBJECTPTR` (rclass.py:160-180).
//!   Backed by `LazyLock<LowLevelType>` to preserve the upstream
//!   module-level singleton semantics (module import = first Lazy deref).
//! * `RootClassRepr` (rclass.py:420-437), `ClassRepr` (rclass.py:191-284),
//!   `InstanceRepr` (rclass.py:467-558) — `__init__` for all three and
//!   the `_setup_repr` vtable-super / object_type-super chains landed,
//!   including readonly/non-readonly attr iteration, `prepare_method`,
//!   `extra_access_sets`, and field reordering by `attr_reverse_size`.
//! * `getclassrepr` / `getinstancerepr` / `buildinstancerepr`
//!   (rclass.py:67-119) — both `classdef is None` and
//!   `classdef != None` arms dispatch. `getclassrepr` caches through
//!   upstream's `classdef.repr` slot, and `getinstancerepr` reads
//!   `_alloc_flavor_` through `classdesc.get_param(...)`.
//!
//! Deferred:
//!
//! * `_check_for_immutable_hints` (rclass.py:560-581),
//!   `special_memory_pressure` + `mutate_*` quasi-immutable fields
//!   (rclass.py:534-546). Phase R2-D, gated on `classdesc.get_param`
//!   / bookkeeper `memory_pressure_types`.
//! * `ClassRepr.init_vtable` + `fill_vtable_root` + `setup_vtable`
//!   (rclass.py:296-418) — needs `lltype.malloc(immortal=True)`,
//!   `attachRuntimeTypeInfo`, `RuntimeTypeInfo`. Phase R3.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::{Arc, LazyLock};

use crate::annotator::classdesc::ClassDef;
use crate::annotator::description::{ClassDefKey, DescEntry};
use crate::annotator::model::{DescKind, SomePBC, SomeValue};
use crate::flowspace::model::{ConstValue, HostObject};
use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
use crate::model::{BlockId, FunctionGraph, OpKind, SpaceOperation, ValueId};
use crate::translator::rtyper::error::TyperError;
use crate::translator::rtyper::lltypesystem::lltype::{
    ForwardReference, GcKind, LowLevelType, Ptr, PtrTarget, RUNTIME_TYPE_INFO, StructType,
};
use crate::translator::rtyper::pairtype::ReprClassId;
use crate::translator::rtyper::rmodel::{Repr, ReprState, mangle};
use crate::translator::rtyper::rtyper::RPythonTyper;

// ---------------------------------------------------------------------
// VtableMethodPtr helper (carried over from the pre-R1 rclass.rs scaffold).
// ---------------------------------------------------------------------

/// Insert a `VtableMethodPtr` op at `(block_id, op_index)` and return the
/// produced funcptr ValueId. Updates `type_state` so downstream passes
/// (`build_value_kinds`, regalloc, flatten) see the funcptr as integer
/// kind — matching `int_guard_value(op.args[0])` in
/// `jtransform.py:546`.
///
/// RPython equivalent: `ClassRepr.getclsfield(vcls, attr, llops)`
/// (`rclass.py:371-377`), which appends `cast_pointer + getfield(vtable,
/// mangled_name)` to `llops`. The full `getclsfield` ports with Phase R2
/// `ClassRepr._setup_repr`; this helper stays as a PRE-EXISTING-ADAPTATION
/// bridge for the pyre IR representation of vtable method slots (see the
/// `OpKind::VtableMethodPtr` comment block in `model.rs`).
pub fn class_get_method_ptr(
    graph: &mut FunctionGraph,
    type_state: &mut TypeResolutionState,
    block_id: BlockId,
    op_index: usize,
    receiver: ValueId,
    trait_root: String,
    method_name: String,
) -> ValueId {
    let funcptr = graph.alloc_value();
    let op = SpaceOperation {
        result: Some(funcptr),
        kind: OpKind::VtableMethodPtr {
            receiver,
            trait_root,
            method_name,
        },
    };
    graph.blocks[block_id.0].operations.insert(op_index, op);
    type_state
        .concrete_types
        .insert(funcptr, ConcreteType::Signed);
    funcptr
}

// ---------------------------------------------------------------------
// rclass.py:25-65 — FieldListAccessor / ImmutableRanking / IR_* constants.
// ---------------------------------------------------------------------

/// RPython `class ImmutableRanking` (`rclass.py:45-54`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ImmutableRanking {
    pub name: &'static str,
    pub is_immutable: bool,
}

/// RPython `IR_MUTABLE = ImmutableRanking('mutable', False)` (rclass.py:56).
pub const IR_MUTABLE: ImmutableRanking = ImmutableRanking {
    name: "mutable",
    is_immutable: false,
};

/// RPython `IR_IMMUTABLE = ImmutableRanking('immutable', True)` (rclass.py:57).
pub const IR_IMMUTABLE: ImmutableRanking = ImmutableRanking {
    name: "immutable",
    is_immutable: true,
};

// IR_IMMUTABLE_ARRAY / IR_QUASIIMMUTABLE / IR_QUASIIMMUTABLE_ARRAY defer to
// Phase R2 when `ClassRepr._setup_repr` starts consuming field rankings.

// ---------------------------------------------------------------------
// rclass.py:160-180 — OBJECT_VTABLE / OBJECT / NONGCOBJECT module constants.
// ---------------------------------------------------------------------

/// RPython module-level `OBJECT_VTABLE = lltype.ForwardReference()` resolved
/// via `.become(Struct('object_vtable', ...))` (rclass.py:160, 167-174).
///
/// Resolves the forward reference to `object_vtable` with the three
/// already-landed head fields `subclassrange_min` / `subclassrange_max`
/// / `rtti`. The remaining two upstream fields (`name: Ptr(rstr.STR)`,
/// `instantiate: Ptr(FuncType([], OBJECTPTR))`) stay deferred until
/// their dependencies (`rstr.STR` string repr, `my_instantiate_graph`
/// from `create_instantiate_functions`) port. Structural equality of
/// two independently-built `OBJECT_VTABLE` clones is preserved by
/// `LowLevelType`'s field-wise `PartialEq`, and the shared
/// `Arc<Mutex<_>>` inside the resolved `ForwardReference` keeps the
/// identity-level singleton semantics that `cast_vtable_to_typeptr`
/// relies on.
pub static OBJECT_VTABLE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let mut fwd = ForwardReference::new();
    // Upstream `Ptr(RuntimeTypeInfo)` (rclass.py:171). The Rust port
    // reconstructs it via the ported `Ptr::from_container_type` helper
    // on the `RUNTIME_TYPE_INFO` opaque singleton.
    let rtti_ptr_type = LowLevelType::Ptr(Box::new(
        Ptr::from_container_type(RUNTIME_TYPE_INFO.clone()).expect(
            "Ptr(RuntimeTypeInfo) must be constructible from the RUNTIME_TYPE_INFO singleton",
        ),
    ));
    let body = StructType::with_hints(
        "object_vtable",
        vec![
            ("subclassrange_min".into(), LowLevelType::Signed),
            ("subclassrange_max".into(), LowLevelType::Signed),
            ("rtti".into(), rtti_ptr_type),
        ],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("static_immutable".into(), ConstValue::Bool(true)),
        ],
    );
    fwd.r#become(LowLevelType::Struct(Box::new(body)))
        .expect("OBJECT_VTABLE.become should succeed");
    LowLevelType::ForwardReference(Box::new(fwd))
});

/// RPython `CLASSTYPE = Ptr(OBJECT_VTABLE)` (rclass.py:161).
pub static CLASSTYPE: LazyLock<LowLevelType> = LazyLock::new(|| {
    let vtable = OBJECT_VTABLE.clone();
    let LowLevelType::ForwardReference(fwd) = vtable else {
        panic!("OBJECT_VTABLE must be a ForwardReference");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::ForwardReference(*fwd),
    }))
});

/// RPython `OBJECT = GcStruct('object', ('typeptr', CLASSTYPE),
/// hints={...}, rtti=True)` (rclass.py:162-165). `rtti=True` funnels
/// through `RttiStruct._install_extras` and mints a
/// `RuntimeTypeInfo` opaque stored on `_runtime_type_info`, so
/// `getRuntimeTypeInfo(OBJECT)` succeeds once R3 consumers
/// (`fill_vtable_root`) land.
pub static OBJECT: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(StructType::gc_rtti_with_hints(
        "object",
        vec![("typeptr".into(), CLASSTYPE.clone())],
        vec![
            ("immutable".into(), ConstValue::Bool(true)),
            ("shouldntbenull".into(), ConstValue::Bool(true)),
            ("typeptr".into(), ConstValue::Bool(true)),
        ],
    )))
});

/// RPython `OBJECTPTR = Ptr(OBJECT)` (rclass.py:166).
pub static OBJECTPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let LowLevelType::Struct(body) = OBJECT.clone() else {
        panic!("OBJECT must be a Struct");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*body),
    }))
});

/// RPython `NONGCOBJECT = Struct('nongcobject', ('typeptr', CLASSTYPE))`
/// (rclass.py:176).
pub static NONGCOBJECT: LazyLock<LowLevelType> = LazyLock::new(|| {
    LowLevelType::Struct(Box::new(StructType::new(
        "nongcobject",
        vec![("typeptr".into(), CLASSTYPE.clone())],
    )))
});

/// RPython `NONGCOBJECTPTR = Ptr(NONGCOBJECT)` (rclass.py:177).
pub static NONGCOBJECTPTR: LazyLock<LowLevelType> = LazyLock::new(|| {
    let LowLevelType::Struct(body) = NONGCOBJECT.clone() else {
        panic!("NONGCOBJECT must be a Struct");
    };
    LowLevelType::Ptr(Box::new(Ptr {
        TO: PtrTarget::Struct(*body),
    }))
});

/// RPython `Flavor` as stored in `getgcflavor(classdef)` return values /
/// `default_flavor` kwargs. Upstream uses strings `'gc'` / `'raw'` /
/// `'stack'` (rclass.py:180 `LLFLAVOR = {'gc':'gc','raw':'raw','stack':'raw'}`).
/// Rust uses a `Copy + Hash + Eq` enum so the `instance_reprs` cache key
/// `(Option<ClassDefKey>, Flavor)` stays pointer-identity friendly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flavor {
    Gc,
    Raw,
}

impl Flavor {
    /// RPython `LLFLAVOR[flavor]` (rclass.py:180). `'stack'` folds to
    /// `'raw'` upstream; Rust surfaces it via [`Flavor::from_str_strict`]
    /// when parity of that arm matters.
    pub fn llflavor(self) -> &'static str {
        match self {
            Flavor::Gc => "gc",
            Flavor::Raw => "raw",
        }
    }

    fn from_alloc_flavor(value: &str) -> Result<Self, TyperError> {
        match value {
            "gc" => Ok(Flavor::Gc),
            "raw" | "stack" => Ok(Flavor::Raw),
            other => Err(TyperError::message(format!(
                "unsupported _alloc_flavor_ {other:?} in rclass.getgcflavor"
            ))),
        }
    }
}

fn const_truthy(value: &ConstValue) -> bool {
    match value {
        ConstValue::None => false,
        ConstValue::Placeholder => false,
        ConstValue::Bool(flag) => *flag,
        ConstValue::Int(value) => *value != 0,
        ConstValue::Float(bits) => f64::from_bits(*bits) != 0.0,
        ConstValue::Str(text) => !text.is_empty(),
        ConstValue::Tuple(items) => !items.is_empty(),
        ConstValue::List(items) => !items.is_empty(),
        ConstValue::Dict(items) => !items.is_empty(),
        ConstValue::Atom(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::LLPtr(_)
        | ConstValue::SpecTag(_)
        | ConstValue::HostObject(_) => true,
    }
}

fn host_is_unboxed_value_subclass(pyobj: &HostObject) -> bool {
    pyobj.mro().is_some_and(|mro| {
        mro.iter().any(|cls| {
            let qualname = cls.qualname();
            qualname == "UnboxedValue" || qualname.ends_with(".UnboxedValue")
        })
    })
}

fn classdesc_get_param(
    classdef: &Rc<RefCell<ClassDef>>,
    name: &str,
    default: ConstValue,
    inherit: bool,
) -> ConstValue {
    let mut cdesc = Some(classdef.borrow().classdesc.clone());
    while let Some(cdesc_rc) = cdesc {
        let (value, basedesc) = {
            let cdesc_ref = cdesc_rc.borrow();
            (cdesc_ref.pyobj.class_get(name), cdesc_ref.basedesc.clone())
        };
        if let Some(value) = value {
            return value;
        }
        if !inherit {
            break;
        }
        cdesc = basedesc;
    }
    default
}

fn getgcflavor(classdef: &Rc<RefCell<ClassDef>>) -> Result<Flavor, TyperError> {
    let alloc_flavor = classdesc_get_param(
        classdef,
        "_alloc_flavor_",
        ConstValue::Str("gc".to_string()),
        true,
    );
    let ConstValue::Str(alloc_flavor) = alloc_flavor else {
        return Err(TyperError::message(
            "classdesc.get_param('_alloc_flavor_') must return a string",
        ));
    };
    Flavor::from_alloc_flavor(&alloc_flavor)
}

fn lowleveltype_size_hint(lltype: &LowLevelType) -> Option<usize> {
    match lltype {
        LowLevelType::Void => None,
        LowLevelType::Bool | LowLevelType::Char => Some(1),
        LowLevelType::UniChar | LowLevelType::SingleFloat => Some(4),
        LowLevelType::Signed
        | LowLevelType::Unsigned
        | LowLevelType::SignedLongLong
        | LowLevelType::UnsignedLongLong
        | LowLevelType::Float
        | LowLevelType::LongFloat
        | LowLevelType::Ptr(_)
        | LowLevelType::InteriorPtr(_) => Some(std::mem::size_of::<usize>()),
        LowLevelType::SignedLongLongLong | LowLevelType::UnsignedLongLongLong => Some(16),
        LowLevelType::Struct(_)
        | LowLevelType::Array(_)
        | LowLevelType::FixedSizeArray(_)
        | LowLevelType::Func(_)
        | LowLevelType::Opaque(_) => None,
        LowLevelType::ForwardReference(fwd) => {
            fwd.resolved().as_ref().and_then(lowleveltype_size_hint)
        }
    }
}

fn attr_reverse_size((_, lltype): &(String, LowLevelType)) -> Option<i64> {
    lowleveltype_size_hint(lltype).map(|size| -(size as i64))
}

fn sort_llfields(llfields: &mut Vec<(String, LowLevelType)>) {
    llfields.sort_by(|left, right| left.0.cmp(&right.0));
    llfields.sort_by_key(attr_reverse_size);
}

fn class_attr_family_key(
    access_set: &Rc<RefCell<crate::annotator::description::ClassAttrFamily>>,
) -> usize {
    Rc::as_ptr(access_set) as usize
}

/// RPython `OBJECT_BY_FLAVOR[LLFLAVOR[gcflavor]]` (rclass.py:179-180,
/// consumed at rclass.py:472). Returns the underlying `LowLevelType` for
/// the root `object_type` of an `InstanceRepr` with `classdef is None`.
pub fn object_by_flavor(flavor: Flavor) -> LowLevelType {
    match flavor {
        Flavor::Gc => OBJECT.clone(),
        Flavor::Raw => NONGCOBJECT.clone(),
    }
}

// ---------------------------------------------------------------------
// rclass.py:191-418 — ClassRepr (classdef != None flavour).
// ---------------------------------------------------------------------

/// RPython `class ClassRepr(Repr)` (rclass.py:191-418).
///
/// Readonly class attrs populate `clsfields` / `allmethods`, and
/// `extra_access_sets` populate `pbcfields`, matching the attr walk in
/// rclass.py:252-271. Method-valued attrs flow through
/// `prepare_method()` before repr lookup so method-descriptor PBCs are
/// rewritten to function PBCs like upstream.
#[derive(Debug)]
pub struct ClassRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:193). Weak because
    /// `ClassDef.repr` holds the strong repr cache entry, and the
    /// back-edge mirrors R1's `Weak<RPythonAnnotator>` on
    /// `RPythonTyper.annotator`.
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef = classdef` (rclass.py:194). Always
    /// non-None for `ClassRepr`; `classdef is None` routes through
    /// [`RootClassRepr`] per upstream's `class
    /// RootClassRepr(ClassRepr)` override at rclass.py:420.
    classdef: Rc<RefCell<ClassDef>>,
    /// RPython `self.vtable_type = lltype.ForwardReference()`
    /// (rclass.py:195). Stored as `LowLevelType::ForwardReference`; the
    /// inner `Arc<Mutex<Option<LowLevelType>>>` target is shared with
    /// [`ClassRepr::lowleveltype`]'s Ptr target so resolving one also
    /// resolves the other.
    vtable_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.vtable_type)`
    /// (rclass.py:196).
    lowleveltype: LowLevelType,
    /// RPython `self.clsfields = clsfields` (rclass.py:281). `{name:
    /// (mangled_name, r)}` dict holding the class-level attribute
    /// reprs, populated by `_setup_repr`.
    clsfields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.pbcfields = pbcfields` (rclass.py:282).
    /// Upstream keys by `(access_set, attr)`; Rust stores the
    /// access-set identity pointer alongside `attr`.
    pbcfields: RefCell<HashMap<(usize, String), (String, Arc<dyn Repr>)>>,
    /// RPython `self.allmethods = allmethods` (rclass.py:283).
    /// `{name: True}` set-as-dict upstream; Rust keeps the dict shape
    /// to remain line-by-line portable even though `HashSet<String>`
    /// would be idiomatic.
    allmethods: RefCell<HashMap<String, bool>>,
    /// RPython `self.rbase = getclassrepr(...)` (rclass.py:273).
    rbase: RefCell<Option<ClassReprArc>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl ClassRepr {
    /// RPython `ClassRepr.__init__(self, rtyper, classdef)`
    /// (rclass.py:192-196).
    ///
    /// Allocates a fresh `ForwardReference` for the vtable container and
    /// stores `Ptr(self.vtable_type)` as the Repr's low-level type. The
    /// `ForwardReference`'s `target: Arc<Mutex<_>>` is cloned into the
    /// `Ptr.TO` variant so both fields observe the same resolution once
    /// `_setup_repr` lands the Struct body.
    pub fn new(rtyper: &Rc<RPythonTyper>, classdef: &Rc<RefCell<ClassDef>>) -> Self {
        let fwd = ForwardReference::new();
        let fwd_for_ptr = fwd.clone();
        let vtable_type = LowLevelType::ForwardReference(Box::new(fwd));
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd_for_ptr),
        }));
        ClassRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: classdef.clone(),
            vtable_type,
            lowleveltype,
            clsfields: RefCell::new(HashMap::new()),
            pbcfields: RefCell::new(HashMap::new()),
            allmethods: RefCell::new(HashMap::new()),
            rbase: RefCell::new(None),
            state: ReprState::new(),
        }
    }

    /// RPython `ClassRepr.classdef` (rclass.py:194).
    pub fn classdef(&self) -> Rc<RefCell<ClassDef>> {
        self.classdef.clone()
    }

    /// RPython `ClassRepr.vtable_type` (rclass.py:195). Exposed so
    /// `ClassRepr._setup_repr` (on a child class) can borrow the parent
    /// vtable_type when building its `('super', rbase.vtable_type)`
    /// Struct entry.
    pub fn vtable_type(&self) -> &LowLevelType {
        &self.vtable_type
    }

    /// Read-only view of the class-level `clsfields` dict populated by
    /// `_setup_repr`. RPython `self.clsfields` (rclass.py:281).
    pub fn clsfields(&self) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.clsfields.borrow()
    }

    /// Read-only view of the class-level `allmethods` dict. RPython
    /// `self.allmethods` (rclass.py:283).
    pub fn allmethods(&self) -> std::cell::Ref<'_, HashMap<String, bool>> {
        self.allmethods.borrow()
    }

    fn prepare_method(&self, s_value: &SomeValue) -> Result<Option<SomeValue>, TyperError> {
        let SomeValue::PBC(s_pbc) = s_value else {
            return Ok(None);
        };
        let kind = s_pbc
            .get_kind()
            .map_err(|err| TyperError::message(err.to_string()))?;
        if kind != DescKind::Method {
            return Ok(None);
        }
        let s_unbound = ClassDef::lookup_filter(&self.classdef, s_pbc, None, &BTreeMap::new())
            .map_err(|err| TyperError::message(err.to_string()))?;
        let SomeValue::PBC(s_unbound) = s_unbound else {
            return Err(TyperError::message(
                "ClassRepr.prepare_method expected lookup_filter() to return SomePBC",
            ));
        };
        let funcdescs: Vec<DescEntry> = s_unbound
            .descriptions
            .values()
            .map(|entry| {
                let method = entry.as_method().ok_or_else(|| {
                    TyperError::message(
                        "ClassRepr.prepare_method expected MethodDesc entries after lookup_filter",
                    )
                })?;
                Ok(DescEntry::Function(method.borrow().funcdesc.clone()))
            })
            .collect::<Result<_, TyperError>>()?;
        // rclass.py:236 — `return annmodel.SomePBC(funcdescs)`. The
        // upstream constructor uses the `can_be_None=False` default; do
        // not propagate `s_unbound.can_be_none` (would alter the
        // nullable-method-PBC repr/key shape).
        Ok(Some(SomeValue::PBC(SomePBC::new(funcdescs, false))))
    }
}

impl Repr for ClassRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ClassRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// RPython `ClassRepr._setup_repr` (rclass.py:242-284).
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("ClassRepr._setup_repr: RPythonTyper weak ref expired")
        })?;
        let (name, basedef) = {
            let cd = self.classdef.borrow();
            (cd.name.clone(), cd.basedef.clone())
        };

        // rclass.py:247-249 — `clsfields = {}; pbcfields = {};
        // allmethods = {}`. Populate local copies first (upstream's
        // "don't store mutable objects on self before they are fully
        // built" rule at rclass.py:243-245).
        let mut clsfields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let mut pbcfields: HashMap<(usize, String), (String, Arc<dyn Repr>)> = HashMap::new();
        let mut allmethods: HashMap<String, bool> = HashMap::new();
        let mut llfields: Vec<(String, LowLevelType)> = Vec::new();

        // rclass.py:252-262 — attrs iteration with readonly filter.
        let attrs_sorted = {
            let cd = self.classdef.borrow();
            cd.attrs
                .iter()
                .map(|(k, v)| (k.clone(), (v.s_value.clone(), v.readonly)))
                .collect::<BTreeMap<String, (SomeValue, bool)>>()
        };
        for (attr_name, (s_value, readonly)) in &attrs_sorted {
            if !readonly {
                continue;
            }
            let mut s_value = s_value.clone();
            if let Some(s_unboundmethod) = self.prepare_method(&s_value)? {
                allmethods.insert(attr_name.clone(), true);
                s_value = s_unboundmethod;
            }
            let r = rtyper.getrepr(&s_value)?;
            let mangled_name = format!("cls_{attr_name}");
            clsfields.insert(attr_name.clone(), (mangled_name.clone(), r.clone()));
            llfields.push((mangled_name, r.lowleveltype().clone()));
        }
        for (access_set, attr_name, counter) in
            self.classdef.borrow().extra_access_sets.values().cloned()
        {
            let r = rtyper.getrepr(&access_set.borrow().s_value)?;
            let mangled_name = mangle(&format!("pbc{counter}"), &attr_name);
            pbcfields.insert(
                (class_attr_family_key(&access_set), attr_name.clone()),
                (mangled_name.clone(), r.clone()),
            );
            llfields.push((mangled_name, r.lowleveltype().clone()));
        }
        sort_llfields(&mut llfields);

        // rclass.py:273 — `self.rbase = getclassrepr(self.rtyper,
        // self.classdef.basedef)`.
        let rbase = getclassrepr_arc(&rtyper, basedef.as_ref())?;
        // rclass.py:274 — `self.rbase.setup()`.
        Repr::setup(rbase.as_repr().as_ref())?;

        // rclass.py:275-278 — `kwds = {'hints': {...}}; vtable_type =
        // Struct('%s_vtable' % name, ('super', rbase.vtable_type),
        // *llfields, **kwds)`.
        let super_field_type = rbase.vtable_type().clone();
        let mut fields = Vec::with_capacity(1 + llfields.len());
        fields.push(("super".into(), super_field_type));
        fields.extend(llfields);
        let vtable_body = StructType::with_hints(
            &format!("{name}_vtable"),
            fields,
            vec![
                ("immutable".into(), ConstValue::Bool(true)),
                ("static_immutable".into(), ConstValue::Bool(true)),
            ],
        );
        // rclass.py:279 — `self.vtable_type.become(vtable_type)`. The
        // inner `Arc<Mutex<_>>` propagates to `self.lowleveltype`'s Ptr
        // target via the clone in `new()`.
        let LowLevelType::ForwardReference(fwd) = &self.vtable_type else {
            return Err(TyperError::message(
                "ClassRepr.vtable_type must be LowLevelType::ForwardReference",
            ));
        };
        fwd.r#become(LowLevelType::Struct(Box::new(vtable_body)))
            .map_err(TyperError::message)?;

        // rclass.py:280 — `allmethods.update(self.rbase.allmethods)`.
        // Merge whichever parent methods upstream already recorded.
        if let ClassReprArc::Inst(parent_inst) = &rbase {
            for (method_name, _) in parent_inst.allmethods.borrow().iter() {
                allmethods.insert(method_name.clone(), true);
            }
        }

        // rclass.py:281-284 — publish the populated dicts onto `self`.
        *self.clsfields.borrow_mut() = clsfields;
        *self.pbcfields.borrow_mut() = pbcfields;
        *self.allmethods.borrow_mut() = allmethods;
        *self.rbase.borrow_mut() = Some(rbase);
        // `self.vtable = None` is represented by the absence of any R3
        // vtable slot on this struct.
        Ok(())
    }
}

/// Typed return for [`getclassrepr_arc`]: the internal recursion in
/// [`ClassRepr::_setup_repr`] needs direct access to `rbase.vtable_type`
/// and the Repr `setup()` entry-point, both of which are awkward to
/// reach through the public `Arc<dyn Repr>`. The enum branches mirror
/// upstream's class hierarchy (`ClassRepr` vs its `RootClassRepr`
/// subclass at rclass.py:420) without requiring runtime type reflection.
#[derive(Clone, Debug)]
pub enum ClassReprArc {
    Root(Arc<RootClassRepr>),
    Inst(Arc<ClassRepr>),
}

impl ClassReprArc {
    /// RPython `rbase.vtable_type` reader (rclass.py:277). Returns the
    /// `LowLevelType` that `ClassRepr._setup_repr` plugs into the
    /// `('super', ...)` field of the child vtable Struct.
    pub fn vtable_type(&self) -> &LowLevelType {
        match self {
            Self::Root(r) => r.vtable_type(),
            Self::Inst(r) => r.vtable_type(),
        }
    }

    /// Upcast to `Arc<dyn Repr>` so callers can invoke `Repr::setup` or
    /// hand the value to code that only needs the Repr surface (e.g.
    /// [`ExceptionData::r_exception_type`]).
    pub fn as_repr(&self) -> Arc<dyn Repr> {
        match self {
            Self::Root(r) => r.clone() as Arc<dyn Repr>,
            Self::Inst(r) => r.clone() as Arc<dyn Repr>,
        }
    }
}

// ---------------------------------------------------------------------
// rclass.py:420-438 — RootClassRepr (classdef = None flavour).
// ---------------------------------------------------------------------

/// RPython `class RootClassRepr(ClassRepr)` (rclass.py:420-437).
///
/// Upstream inherits all `ClassRepr` methods and overrides `__init__` /
/// `_setup_repr` / `init_vtable`. The Rust port keeps `RootClassRepr` as
/// a sibling struct sharing the [`ClassReprArc`] dispatch enum because
/// Rust has no structural inheritance. [`RootClassRepr::_setup_repr`] is
/// a no-op (clsfields/pbcfields/allmethods/vtable all start empty) until
/// R2-B / R3 add their storage.
#[derive(Debug)]
pub struct RootClassRepr {
    /// RPython `self.classdef = None` (rclass.py:422).
    classdef: Option<ClassDefKey>,
    /// RPython `self.vtable_type = OBJECT_VTABLE` (rclass.py:426).
    vtable_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.vtable_type)` (rclass.py:427).
    lowleveltype: LowLevelType,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
    // clsfields / pbcfields / allmethods / vtable fields are initialised by
    // `_setup_repr` to empty / None. They'll carry real data once Phase R2
    // / R3 port `setup_vtable` / `init_vtable`.
}

impl RootClassRepr {
    /// RPython `RootClassRepr.__init__(self, rtyper)` (rclass.py:424-427).
    ///
    /// Upstream stores `self.rtyper = rtyper`; the Rust port does not
    /// yet read it from [`RootClassRepr`], so it is omitted until a
    /// concrete method (`convert_desc`, `init_vtable`, ...) needs it.
    pub fn new() -> Self {
        RootClassRepr {
            classdef: None,
            vtable_type: OBJECT_VTABLE.clone(),
            lowleveltype: CLASSTYPE.clone(),
            state: ReprState::new(),
        }
    }

    /// RPython `RootClassRepr.classdef` (`classdef = None`, rclass.py:422).
    pub fn classdef(&self) -> Option<ClassDefKey> {
        self.classdef
    }

    /// RPython `RootClassRepr.vtable_type` (rclass.py:426).
    pub fn vtable_type(&self) -> &LowLevelType {
        &self.vtable_type
    }
}

impl Repr for RootClassRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        // Upstream `ClassRepr.__repr__` (rclass.py:198-203) emits
        // `"<ClassRepr for object>"`; the pyre tag follows the class name
        // matching the `.__class__.__name__` of RootClassRepr.
        "RootClassRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// RPython `RootClassRepr._setup_repr(self)` (rclass.py:429-433).
    ///
    /// ```python
    /// def _setup_repr(self):
    ///     self.clsfields = {}
    ///     self.pbcfields = {}
    ///     self.allmethods = {}
    ///     self.vtable = None
    /// ```
    ///
    /// All four assignments are currently no-ops because Phase R1 holds
    /// no storage for them — the fields only exist upstream to keep
    /// `ClassRepr` method dispatch consistent, and the methods that read
    /// them (`getvtable`, `getclsfield`, `get_field`) land with Phase R2.
    fn _setup_repr(&self) -> Result<(), TyperError> {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// rclass.py:467-558 — InstanceRepr (classdef=None + classdef!=None
// scaffolding branches).
// ---------------------------------------------------------------------

/// RPython `class InstanceRepr(Repr)` (rclass.py:467-558).
///
/// R2-C brings in the attrs iteration for `readonly = False` instance
/// attributes: `fields` / `allinstancefields` populate via
/// `rtyper.getrepr(attrdef.s_value)` + `mangled_name = 'inst_' + name`
/// (rclass.py:503-509), and the `object_type` Struct body includes the
/// `(mangled_name, r.lowleveltype)` pairs alongside `('super',
/// rbase.object_type)`. The `special_memory_pressure` /
/// `mutate_<name>` quasi-immutable fields (rclass.py:534-546), the
/// `_check_for_immutable_hints` branch (rclass.py:560-581), and the
/// `rtti=True` kwarg (rclass.py:531-532 `attachRuntimeTypeInfo` side
/// effect) remain R2-D / R3 scope.
#[derive(Debug)]
pub struct InstanceRepr {
    /// RPython `self.rtyper = rtyper` (rclass.py:469).
    rtyper: Weak<RPythonTyper>,
    /// RPython `self.classdef` (rclass.py:470). `None` mirrors
    /// upstream's sentinel for the root instance that carries
    /// `object_type = OBJECT_BY_FLAVOR[...]` directly.
    classdef: Option<Rc<RefCell<ClassDef>>>,
    /// RPython `self.object_type = OBJECT_BY_FLAVOR[LLFLAVOR[gcflavor]]`
    /// when classdef is None (rclass.py:472), otherwise a fresh
    /// `ForwardReference` (rclass.py:474-475). The FR target resolves
    /// during `_setup_repr` once the `MkStruct('name', ('super',
    /// rbase.object_type), ...)` body is built (rclass.py:548-554).
    object_type: LowLevelType,
    /// RPython `self.lowleveltype = Ptr(self.object_type)` (rclass.py:477).
    lowleveltype: LowLevelType,
    /// RPython `self.gcflavor` (rclass.py:478).
    gcflavor: Flavor,
    /// RPython `self.rclass = getclassrepr(...)` (rclass.py:494).
    rclass: RefCell<Option<ClassReprArc>>,
    /// RPython `self.fields = fields` (rclass.py:557). Instance-level
    /// attributes keyed by unmangled attr name; populated by
    /// `_setup_repr`.
    fields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.rbase = getinstancerepr(...)` (rclass.py:517).
    rbase: RefCell<Option<Arc<InstanceRepr>>>,
    /// RPython `self.allinstancefields = allinstancefields`
    /// (rclass.py:558). Accumulates rbase.allinstancefields ∪ fields.
    allinstancefields: RefCell<HashMap<String, (String, Arc<dyn Repr>)>>,
    /// RPython `self.immutable_field_set = set()` (rclass.py:493) for
    /// classdef=None, overwritten by `_check_for_immutable_hints` for
    /// classdef!=None (rclass.py:576). R2-C keeps it as an empty set —
    /// the immutable-hint derivation lands in R2-D.
    immutable_field_set: RefCell<HashSet<String>>,
    /// RPython `Repr._initialized` state machine.
    state: ReprState,
}

impl InstanceRepr {
    /// RPython `InstanceRepr.__init__(self, rtyper, classdef,
    /// gcflavor='gc')` (rclass.py:468-478).
    ///
    /// Covers both branches: `classdef is None` (rclass.py:471-472) and
    /// `classdef is not None` (rclass.py:473-475). In the latter case a
    /// fresh `ForwardReference` / `GcForwardReference` is allocated with
    /// its `Arc<Mutex<_>>` target shared into the `lowleveltype` Ptr, so
    /// [`InstanceRepr::_setup_repr`] can resolve it once.
    pub fn new(
        rtyper: &Rc<RPythonTyper>,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        gcflavor: Flavor,
    ) -> Self {
        let (object_type, lowleveltype) = build_instance_types(classdef, gcflavor);
        InstanceRepr {
            rtyper: Rc::downgrade(rtyper),
            classdef: classdef.cloned(),
            object_type,
            lowleveltype,
            gcflavor,
            rclass: RefCell::new(None),
            fields: RefCell::new(HashMap::new()),
            rbase: RefCell::new(None),
            allinstancefields: RefCell::new(HashMap::new()),
            immutable_field_set: RefCell::new(HashSet::new()),
            state: ReprState::new(),
        }
    }

    /// Convenience constructor for the classdef=None root instance
    /// (rclass.py:471-472). Keeps the live `rtyper` backref so
    /// `_setup_repr` can populate the root `__class__` field exactly
    /// like upstream.
    pub fn new_rootinstance(rtyper: &Rc<RPythonTyper>, gcflavor: Flavor) -> Self {
        Self::new(rtyper, None, gcflavor)
    }

    /// RPython `InstanceRepr.classdef` (rclass.py:470). Returns
    /// `None` for root instances, `Some(Rc<RefCell<ClassDef>>)` otherwise.
    pub fn classdef(&self) -> Option<Rc<RefCell<ClassDef>>> {
        self.classdef.clone()
    }

    /// RPython `InstanceRepr.object_type` (rclass.py:472,475).
    pub fn object_type(&self) -> &LowLevelType {
        &self.object_type
    }

    /// RPython `InstanceRepr.gcflavor` (rclass.py:478).
    pub fn gcflavor(&self) -> Flavor {
        self.gcflavor
    }

    /// Read-only view of the instance-level `fields` dict populated by
    /// `_setup_repr`. RPython `self.fields` (rclass.py:557).
    pub fn fields(&self) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.fields.borrow()
    }

    /// Read-only view of the `allinstancefields` dict. RPython
    /// `self.allinstancefields` (rclass.py:558).
    pub fn allinstancefields(
        &self,
    ) -> std::cell::Ref<'_, HashMap<String, (String, Arc<dyn Repr>)>> {
        self.allinstancefields.borrow()
    }
}

/// Build the `(object_type, lowleveltype)` pair for an InstanceRepr
/// per rclass.py:471-477.
fn build_instance_types(
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    gcflavor: Flavor,
) -> (LowLevelType, LowLevelType) {
    if classdef.is_some() {
        // rclass.py:474-475 — `ForwardRef = lltype.FORWARDREF_BY_FLAVOR[
        // LLFLAVOR[gcflavor]]; self.object_type = ForwardRef()`.
        let fwd = match gcflavor {
            Flavor::Gc => ForwardReference::gc(),
            Flavor::Raw => ForwardReference::new(),
        };
        let fwd_for_ptr = fwd.clone();
        let object_type = LowLevelType::ForwardReference(Box::new(fwd));
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: PtrTarget::ForwardReference(fwd_for_ptr),
        }));
        (object_type, lowleveltype)
    } else {
        // rclass.py:471-472 — classdef is None path.
        let object_type = object_by_flavor(gcflavor);
        let lowleveltype = LowLevelType::Ptr(Box::new(Ptr {
            TO: match &object_type {
                LowLevelType::Struct(body) => PtrTarget::Struct((**body).clone()),
                other => panic!("object_by_flavor returned non-Struct {other:?}"),
            },
        }));
        (object_type, lowleveltype)
    }
}

impl Repr for InstanceRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "InstanceRepr"
    }

    fn repr_class_id(&self) -> ReprClassId {
        ReprClassId::Repr
    }

    /// RPython `InstanceRepr._setup_repr` (rclass.py:487-558).
    fn _setup_repr(&self) -> Result<(), TyperError> {
        let rtyper = self.rtyper.upgrade().ok_or_else(|| {
            TyperError::message("InstanceRepr._setup_repr: RPythonTyper weak ref expired")
        })?;

        // rclass.py:494 — `self.rclass = getclassrepr(self.rtyper,
        // self.classdef)`.
        let rclass = getclassrepr_arc(&rtyper, self.classdef.as_ref())?;
        *self.rclass.borrow_mut() = Some(rclass);

        // rclass.py:495-496 — `fields = {}; allinstancefields = {}`.
        // Build local dicts first (rclass.py:488-491 "don't store
        // mutable objects on self before they are fully built").
        let mut fields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let mut allinstancefields: HashMap<String, (String, Arc<dyn Repr>)> = HashMap::new();
        let Some(classdef_rc) = self.classdef.as_ref() else {
            fields.insert(
                "__class__".to_string(),
                ("typeptr".to_string(), get_type_repr(rtyper.as_ref())?),
            );
            allinstancefields.extend(fields.clone());
            *self.fields.borrow_mut() = fields;
            *self.allinstancefields.borrow_mut() = allinstancefields;
            return Ok(());
        };
        let mut myllfields: Vec<(String, LowLevelType)> = Vec::new();

        // rclass.py:501-509 — attrs iteration with readonly filter.
        let attrs_sorted = {
            let cd = classdef_rc.borrow();
            cd.attrs
                .iter()
                .map(|(k, v)| (k.clone(), (v.s_value.clone(), v.readonly)))
                .collect::<BTreeMap<String, (SomeValue, bool)>>()
        };
        for (attr_name, (s_value, readonly)) in &attrs_sorted {
            if *readonly {
                continue;
            }
            let r = rtyper.getrepr(s_value)?;
            let mangled_name = format!("inst_{attr_name}");
            fields.insert(attr_name.clone(), (mangled_name.clone(), r.clone()));
            myllfields.push((mangled_name, r.lowleveltype().clone()));
        }
        sort_llfields(&mut myllfields);

        // rclass.py:517-519 — `self.rbase = getinstancerepr(
        // self.rtyper, self.classdef.basedef, self.gcflavor);
        // self.rbase.setup()`.
        let basedef = classdef_rc.borrow().basedef.clone();
        let rbase = getinstancerepr(&rtyper, basedef.as_ref(), self.gcflavor)?;
        Repr::setup(rbase.as_ref())?;
        *self.rbase.borrow_mut() = Some(rbase.clone());

        // rclass.py:548-554 — `object_type = MkStruct(classdef.name,
        // ('super', rbase.object_type), hints=hints, adtmeths=adtmeths,
        // *llfields, **kwds)` + `self.object_type.become(object_type)`.
        // Gc-flavor also passes `rtti=True` (rclass.py:531-532) so the
        // struct carries an `_runtime_type_info` opaque consumable by
        // `fill_vtable_root` via `getRuntimeTypeInfo`. Immutable /
        // special_memory_pressure hints stay unported (R2-D).
        let name = classdef_rc.borrow().name.clone();
        let mut struct_fields = Vec::with_capacity(1 + myllfields.len());
        struct_fields.push(("super".into(), rbase.object_type().clone()));
        struct_fields.extend(myllfields);
        let body = match self.gcflavor {
            Flavor::Gc => StructType::gc_rtti(&name, struct_fields),
            Flavor::Raw => StructType::with_hints(&name, struct_fields, vec![]),
        };
        let LowLevelType::ForwardReference(fwd) = &self.object_type else {
            return Err(TyperError::message(
                "InstanceRepr.object_type must be LowLevelType::ForwardReference \
                 when classdef is not None",
            ));
        };
        fwd.r#become(LowLevelType::Struct(Box::new(body)))
            .map_err(TyperError::message)?;

        // rclass.py:555-558 — `allinstancefields.update(
        // self.rbase.allinstancefields); allinstancefields.update(
        // fields); self.fields = fields;
        // self.allinstancefields = allinstancefields`.
        for (k, v) in rbase.allinstancefields.borrow().iter() {
            allinstancefields.insert(k.clone(), v.clone());
        }
        for (k, v) in fields.iter() {
            allinstancefields.insert(k.clone(), v.clone());
        }
        *self.fields.borrow_mut() = fields;
        *self.allinstancefields.borrow_mut() = allinstancefields;
        Ok(())
    }
}

// ---------------------------------------------------------------------
// rclass.py:67-88, 91-119, 439-440 — module-level accessors.
// ---------------------------------------------------------------------

/// RPython `get_type_repr(rtyper)` (`rclass.py:439-440`).
///
/// ```python
/// def get_type_repr(rtyper):
///     return rtyper.rootclass_repr
/// ```
pub fn get_type_repr(rtyper: &RPythonTyper) -> Result<Arc<dyn Repr>, TyperError> {
    rtyper
        .rootclass_repr
        .borrow()
        .clone()
        .map(|r| r as Arc<dyn Repr>)
        .ok_or_else(|| {
            TyperError::message(
                "rtyper.rootclass_repr is not set — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })
}

/// RPython `getclassrepr(rtyper, classdef)` (`rclass.py:67-74`).
///
/// ```python
/// def getclassrepr(rtyper, classdef):
///     if classdef is None:
///         return rtyper.rootclass_repr
///     result = classdef.repr
///     if result is None:
///         result = classdef.repr = ClassRepr(rtyper, classdef)
///         rtyper.add_pendingsetup(result)
///     return result
/// ```
///
/// The `classdef != None` arm caches through `ClassDef.repr`, matching
/// upstream's per-classdef storage.
pub fn getclassrepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
) -> Result<Arc<dyn Repr>, TyperError> {
    Ok(getclassrepr_arc(rtyper, classdef)?.as_repr())
}

/// Internal-recursion companion to [`getclassrepr`]: returns the
/// typed [`ClassReprArc`] enum so [`ClassRepr::_setup_repr`] can read
/// `rbase.vtable_type` directly without upcasting through
/// `Arc<dyn Repr>`.
pub(crate) fn getclassrepr_arc(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
) -> Result<ClassReprArc, TyperError> {
    let Some(classdef_rc) = classdef else {
        // rclass.py:68-69 — `if classdef is None: return rtyper.rootclass_repr`.
        let root = rtyper.rootclass_repr.borrow().clone().ok_or_else(|| {
            TyperError::message(
                "rtyper.rootclass_repr is not set — call \
                 RPythonTyper::initialize_exceptiondata() after construction",
            )
        })?;
        return Ok(ClassReprArc::Root(root));
    };
    // rclass.py:70-73 — `classdef.repr` cache, fresh-build path.
    if let Some(existing) = classdef_rc.borrow().repr.clone() {
        return Ok(ClassReprArc::Inst(existing));
    }
    let repr = Arc::new(ClassRepr::new(rtyper, classdef_rc));
    classdef_rc.borrow_mut().repr = Some(repr.clone());
    rtyper.add_pendingsetup(repr.clone() as Arc<dyn Repr>);
    Ok(ClassReprArc::Inst(repr))
}

/// RPython `getinstancerepr(rtyper, classdef, default_flavor='gc')`
/// (`rclass.py:76-88`).
///
/// ```python
/// def getinstancerepr(rtyper, classdef, default_flavor='gc'):
///     if classdef is None:
///         flavor = default_flavor
///     else:
///         flavor = getgcflavor(classdef)
///     try:
///         result = rtyper.instance_reprs[classdef, flavor]
///     except KeyError:
///         result = buildinstancerepr(rtyper, classdef, gcflavor=flavor)
///         rtyper.instance_reprs[classdef, flavor] = result
///         rtyper.add_pendingsetup(result)
///     return result
/// ```
///
pub fn getinstancerepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    default_flavor: Flavor,
) -> Result<Arc<InstanceRepr>, TyperError> {
    let flavor = match classdef {
        None => default_flavor,
        Some(classdef_rc) => getgcflavor(classdef_rc)?,
    };
    let classdef_key = classdef.map(ClassDefKey::from_classdef);
    let key = (classdef_key, flavor);
    if let Some(existing) = rtyper.instance_reprs.borrow().get(&key) {
        return Ok(existing.clone());
    }
    let result = Arc::new(buildinstancerepr(rtyper, classdef, flavor)?);
    rtyper
        .instance_reprs
        .borrow_mut()
        .insert(key, result.clone());
    rtyper.add_pendingsetup(result.clone());
    Ok(result)
}

/// RPython `buildinstancerepr(rtyper, classdef, gcflavor='gc')`
/// (`rclass.py:91-119`).
///
pub fn buildinstancerepr(
    rtyper: &Rc<RPythonTyper>,
    classdef: Option<&Rc<RefCell<ClassDef>>>,
    gcflavor: Flavor,
) -> Result<InstanceRepr, TyperError> {
    // rclass.py:94-103 — `classdef is None` short-circuits `unboxed =
    // []`, `virtualizable = False`; otherwise scan all subdefs for
    // `UnboxedValue`-derived classes and read `_virtualizable_`. The
    // `usetagging` gate is `len(unboxed) != 0 and
    // config.translation.taggedpointers`, so when taggedpointers is off
    // the presence of an `UnboxedValue` subclass falls through to a
    // regular `InstanceRepr`.
    let (unboxed, virtualizable) = match classdef {
        None => (Vec::new(), false),
        Some(classdef_rc) => {
            let unboxed: Vec<Rc<RefCell<ClassDef>>> = ClassDef::getallsubdefs(classdef_rc)
                .into_iter()
                .filter(|subdef| {
                    host_is_unboxed_value_subclass(&subdef.borrow().classdesc.borrow().pyobj)
                })
                .collect();
            let virtualizable = const_truthy(&classdesc_get_param(
                classdef_rc,
                "_virtualizable_",
                ConstValue::Bool(false),
                true,
            ));
            (unboxed, virtualizable)
        }
    };
    // `rtyper.annotator` is weak because the annotator owns a strong
    // `Rc<RPythonTyper>` (rtyper.py `annotator.translator.rtyper` +
    // `rtyper.annotator` form a Python-side cycle; Rust breaks it by
    // keeping this edge weak). Tests that construct an RPythonTyper from
    // a non-retained `Rc<RPythonAnnotator>` observe the weak drop here.
    // Upstream `config.translation.taggedpointers` defaults to `False`,
    // so on upgrade failure we use the default — matching upstream
    // semantics for any caller that has not explicitly opted in to
    // tagged pointers.
    let taggedpointers = rtyper
        .annotator
        .upgrade()
        .map(|ann| ann.translator.config.translation.taggedpointers)
        .unwrap_or(false);
    let usetagging = !unboxed.is_empty() && taggedpointers;

    // rclass.py:105-108 — virtualizable path.
    if virtualizable {
        assert!(
            unboxed.is_empty(),
            "_virtualizable_ class must not have UnboxedValue subclasses"
        );
        assert_eq!(gcflavor, Flavor::Gc, "_virtualizable_ requires gc flavor");
        return Err(TyperError::message(
            "buildinstancerepr: VirtualizableInstanceRepr parity requires \
             rpython/rtyper/rvirtualizable.py (not yet ported)",
        ));
    }
    // rclass.py:109-117 — tagged-pointer path.
    if usetagging {
        if unboxed.len() != 1 {
            return Err(TyperError::message(format!(
                "{:?} has several UnboxedValue subclasses",
                classdef.map(|cd| cd.borrow().name.clone())
            )));
        }
        assert_eq!(
            gcflavor,
            Flavor::Gc,
            "UnboxedValue tagging requires gc flavor"
        );
        return Err(TyperError::message(
            "buildinstancerepr: TaggedInstanceRepr parity requires \
             rpython/rtyper/lltypesystem/rtagged.py (not yet ported)",
        ));
    }
    // rclass.py:118-119 — default `InstanceRepr`. An `UnboxedValue`
    // subclass without `config.translation.taggedpointers` falls through
    // here upstream too, so this arm owns that parity case.
    match classdef {
        None => Ok(InstanceRepr::new_rootinstance(rtyper, gcflavor)),
        Some(classdef_rc) => Ok(InstanceRepr::new(rtyper, Some(classdef_rc), gcflavor)),
    }
}

/// Cache key shape `(Option<ClassDefKey>, Flavor)` used by
/// [`RPythonTyper::instance_reprs`]. Exposed so consumers outside this
/// module can build matching queries without importing `Flavor` + the
/// field directly.
pub type InstanceReprKey = (Option<ClassDefKey>, Flavor);

// ---------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn object_vtable_resolves_to_struct_with_subclassrange_fields() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!("OBJECT_VTABLE must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("OBJECT_VTABLE should be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("OBJECT_VTABLE must resolve to a Struct");
        };
        assert_eq!(body._name, "object_vtable");
        assert!(body._flds.get("subclassrange_min").is_some());
        assert!(body._flds.get("subclassrange_max").is_some());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("static_immutable"),
            Some(&ConstValue::Bool(true))
        );
    }

    #[test]
    fn object_vtable_carries_rtti_ptr_to_runtime_type_info() {
        let LowLevelType::ForwardReference(fwd) = OBJECT_VTABLE.clone() else {
            panic!();
        };
        let LowLevelType::Struct(body) = fwd.resolved().expect("resolved") else {
            panic!();
        };
        let rtti_type = body
            ._flds
            .get("rtti")
            .expect("rtti field must be present (rclass.py:171)");
        let LowLevelType::Ptr(ptr) = rtti_type else {
            panic!("rtti must be a Ptr, got {rtti_type:?}");
        };
        let PtrTarget::Opaque(opaque) = &ptr.TO else {
            panic!("rtti must point to an opaque, got {:?}", ptr.TO);
        };
        assert_eq!(opaque.tag, "RuntimeTypeInfo");
    }

    #[test]
    fn classtype_is_ptr_to_object_vtable_forward_reference() {
        let LowLevelType::Ptr(ptr) = CLASSTYPE.clone() else {
            panic!("CLASSTYPE must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn object_is_gcstruct_with_typeptr_field_and_hints() {
        let LowLevelType::Struct(body) = OBJECT.clone() else {
            panic!("OBJECT must be Struct");
        };
        assert_eq!(body._name, "object");
        assert_eq!(body._gckind, GcKind::Gc);
        assert!(body._flds.get("typeptr").is_some());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("shouldntbenull"),
            Some(&ConstValue::Bool(true))
        );
        assert_eq!(body._hints.get("typeptr"), Some(&ConstValue::Bool(true)));
    }

    #[test]
    fn object_carries_rtti_opaque_under_runtime_type_info() {
        // rclass.py:162-165 sets `rtti=True` on OBJECT so
        // `getRuntimeTypeInfo(OBJECT)` resolves. R3 consumers
        // (`fill_vtable_root`) rely on this.
        let LowLevelType::Struct(body) = OBJECT.clone() else {
            panic!("OBJECT must be Struct");
        };
        let opaque = body
            ._runtime_type_info
            .as_ref()
            .expect("OBJECT must carry an _runtime_type_info opaque");
        assert_eq!(opaque._name.as_deref(), Some("object"));
    }

    #[test]
    fn nongcobject_is_raw_struct_with_typeptr_field() {
        let LowLevelType::Struct(body) = NONGCOBJECT.clone() else {
            panic!("NONGCOBJECT must be Struct");
        };
        assert_eq!(body._name, "nongcobject");
        assert_eq!(body._gckind, GcKind::Raw);
    }

    #[test]
    fn object_by_flavor_returns_gc_and_raw_forms() {
        assert_eq!(object_by_flavor(Flavor::Gc), OBJECT.clone());
        assert_eq!(object_by_flavor(Flavor::Raw), NONGCOBJECT.clone());
    }

    #[test]
    fn root_class_repr_lowleveltype_is_classtype() {
        let repr = RootClassRepr::new();
        assert_eq!(repr.lowleveltype(), &CLASSTYPE.clone());
        assert_eq!(repr.vtable_type(), &OBJECT_VTABLE.clone());
        assert_eq!(repr.classdef(), None);
    }

    #[test]
    fn root_class_repr_setup_is_noop_idempotent() {
        let repr = RootClassRepr::new();
        Repr::setup(&repr).expect("first setup");
        Repr::setup(&repr).expect("second setup re-enters Finished branch");
    }

    #[test]
    fn instance_repr_none_classdef_carries_gc_object_type() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        assert!(repr.classdef().is_none());
        assert_eq!(repr.gcflavor(), Flavor::Gc);
        assert_eq!(repr.object_type(), &OBJECT.clone());
        let LowLevelType::Ptr(ptr) = repr.lowleveltype().clone() else {
            panic!("InstanceRepr.lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::Struct(_)));
    }

    #[test]
    fn instance_repr_none_classdef_setup_ok() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let repr = InstanceRepr::new_rootinstance(&rtyper, Flavor::Gc);
        Repr::setup(&repr).expect("setup classdef=None path");
    }

    // -----------------------------------------------------------------
    // R2-A — ClassRepr scaffold + getclassrepr(classdef != None).
    // -----------------------------------------------------------------

    fn fresh_rtyper() -> Rc<RPythonTyper> {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        rtyper
    }

    #[test]
    fn instance_repr_none_classdef_populates___class___field() {
        let rtyper = fresh_rtyper();
        let repr = getinstancerepr(&rtyper, None, Flavor::Gc).expect("root instance repr");
        Repr::setup(repr.as_ref()).expect("setup root instance repr");

        let fields = repr.fields();
        let (mangled, field_repr) = fields
            .get("__class__")
            .expect("root instance must expose __class__");
        assert_eq!(mangled, "typeptr");
        assert_eq!(field_repr.lowleveltype(), &CLASSTYPE.clone());
        assert!(repr.allinstancefields().contains_key("__class__"));
    }

    #[test]
    fn classrepr_new_stores_classdef_and_unresolved_vtable_forward_reference() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = ClassRepr::new(&rtyper, &classdef);

        // rclass.py:194 — `self.classdef = classdef`.
        assert!(Rc::ptr_eq(&repr.classdef(), &classdef));

        // rclass.py:195 — vtable_type is a fresh ForwardReference (no
        // become() yet).
        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!("ClassRepr.vtable_type must be ForwardReference before setup");
        };
        assert!(
            fwd.resolved().is_none(),
            "vtable_type should not be resolved until _setup_repr runs"
        );

        // rclass.py:196 — `lowleveltype = Ptr(self.vtable_type)`.
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("ClassRepr.lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn classrepr_setup_resolves_vtable_struct_with_super_pointing_at_root_vtable() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr_arc =
            match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr_arc(classdef)") {
                ClassReprArc::Inst(r) => r,
                _ => panic!("classdef != None should return ClassReprArc::Inst"),
            };

        Repr::setup(repr_arc.as_ref()).expect("ClassRepr._setup_repr");

        // rclass.py:276-279 — vtable_type resolves to `<name>_vtable`
        // Struct with a single ('super', rbase.vtable_type) field.
        let LowLevelType::ForwardReference(fwd) = repr_arc.vtable_type() else {
            panic!("vtable_type must be ForwardReference");
        };
        let resolved = fwd.resolved().expect("vtable_type must be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("vtable_type must resolve to Struct");
        };
        assert_eq!(body._name, "pkg.C_vtable");
        let super_field = body
            ._flds
            .get("super")
            .expect("vtable Struct must carry a 'super' field");

        // rbase for classdef with no basedef = RootClassRepr, whose
        // vtable_type is OBJECT_VTABLE (via rclass.py:69 + rclass.py:426).
        assert_eq!(*super_field, OBJECT_VTABLE.clone());
        assert_eq!(body._hints.get("immutable"), Some(&ConstValue::Bool(true)));
        assert_eq!(
            body._hints.get("static_immutable"),
            Some(&ConstValue::Bool(true))
        );
    }

    #[test]
    fn getclassrepr_caches_second_lookup_returns_same_arc() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let first = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("first") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        let second = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("second") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        assert!(
            Arc::ptr_eq(&first, &second),
            "getclassrepr must cache by ClassDefKey identity"
        );
    }

    #[test]
    fn classrepr_basedef_recursion_threads_parent_vtable_type_into_super_field() {
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));

        // Force both reprs to exist so we can compare their vtable_types.
        let base_arc = match getclassrepr_arc(&rtyper, Some(&base)).expect("base") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        let child_arc = match getclassrepr_arc(&rtyper, Some(&child)).expect("child") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };

        // Setup child: internally triggers base.setup() via rclass.py:274.
        Repr::setup(child_arc.as_ref()).expect("child setup");

        // Both reprs must now be FINISHED.
        let resolved_base = match base_arc.vtable_type() {
            LowLevelType::ForwardReference(fwd) => fwd
                .resolved()
                .expect("base vtable_type must be resolved via child setup"),
            _ => panic!(),
        };
        let resolved_child = match child_arc.vtable_type() {
            LowLevelType::ForwardReference(fwd) => {
                fwd.resolved().expect("child vtable_type must be resolved")
            }
            _ => panic!(),
        };

        // Child's 'super' field type must equal base's vtable_type
        // (resolved LowLevelType::ForwardReference that compares equal
        // via inner Arc<Mutex> target).
        let LowLevelType::Struct(child_body) = resolved_child else {
            panic!("child vtable must be Struct");
        };
        let super_field = child_body
            ._flds
            .get("super")
            .expect("child vtable must have 'super'");
        // rclass.py:277 passes `rbase.vtable_type` (a ForwardReference
        // LowLevelType wrapper around the same Arc<Mutex> that the
        // base repr holds). We stored a clone in the 'super' slot, so
        // both should resolve to the same inner Struct body.
        let LowLevelType::ForwardReference(super_fwd) = super_field else {
            panic!("'super' field type must be ForwardReference to base vtable");
        };
        assert_eq!(
            super_fwd
                .resolved()
                .expect("super field's ForwardReference resolved"),
            resolved_base,
        );
    }

    #[test]
    fn getclassrepr_arc_none_classdef_returns_rootclass_repr() {
        let rtyper = fresh_rtyper();
        let arc = getclassrepr_arc(&rtyper, None).expect("None classdef");
        match arc {
            ClassReprArc::Root(r) => assert_eq!(r.classdef(), None),
            _ => panic!("None classdef must route to Root"),
        }
    }

    #[test]
    fn getclassrepr_without_rootclass_repr_returns_typererror_for_none() {
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        // No initialize_exceptiondata — rootclass_repr stays None.
        let err =
            getclassrepr_arc(&rtyper, None).expect_err("should surface missing rootclass_repr");
        assert!(
            err.to_string().contains("initialize_exceptiondata"),
            "error must point at initialize_exceptiondata, got: {err}",
        );
    }

    #[test]
    fn classrepr_prepare_method_rewrites_method_pbc_to_function_pbc() {
        use crate::annotator::bookkeeper::Bookkeeper;
        use crate::annotator::classdesc::ClassDesc;
        use crate::annotator::description::{FunctionDesc, MethodDesc};
        use crate::flowspace::argument::Signature;

        let rtyper = fresh_rtyper();
        let bk = Rc::new(Bookkeeper::new());
        let pyobj = HostObject::new_class("pkg.C", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            pyobj,
            "pkg.C".into(),
        )));
        let classdef = ClassDef::new(&bk, &desc);
        bk.register_classdef(classdef.clone());
        let funcdesc = Rc::new(RefCell::new(FunctionDesc::new(
            bk.clone(),
            None,
            "pkg.C.method",
            Signature::new(vec!["self".to_string()], None, None),
            None,
            None,
        )));
        let method = Rc::new(RefCell::new(MethodDesc::new(
            bk,
            funcdesc.clone(),
            ClassDefKey::from_classdef(&classdef),
            None,
            "method",
            BTreeMap::new(),
        )));
        let s_value = SomeValue::PBC(SomePBC::new(vec![DescEntry::Method(method)], true));

        let repr = ClassRepr::new(&rtyper, &classdef);
        let prepared = repr
            .prepare_method(&s_value)
            .expect("prepare_method")
            .expect("method SomePBC should be rewritten");
        let SomeValue::PBC(prepared_pbc) = prepared else {
            panic!("prepare_method must keep a SomePBC");
        };
        assert_eq!(
            prepared_pbc.get_kind().expect("PBC kind"),
            DescKind::Function
        );
        assert!(
            !prepared_pbc.can_be_none,
            "prepare_method must drop nullable method-PBC shape like upstream"
        );
        assert_eq!(prepared_pbc.descriptions.len(), 1);
        assert!(
            prepared_pbc
                .descriptions
                .values()
                .next()
                .expect("rewritten PBC entry")
                .is_function()
        );
    }

    #[test]
    fn classrepr_setup_materializes_extra_access_sets() {
        use crate::annotator::description::{ClassAttrFamily, DescKey};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let access_set = Rc::new(RefCell::new(ClassAttrFamily::new(DescKey::from_rc(
            &classdef.borrow().classdesc,
        ))));
        access_set.borrow_mut().s_value = SomeValue::Bool(crate::annotator::model::SomeBool::new());
        let access_key = class_attr_family_key(&access_set);
        classdef
            .borrow_mut()
            .extra_access_sets
            .insert(access_key, (access_set, "x".to_string(), 0));

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("extra_access_sets setup");
        let pbcfields = repr.pbcfields.borrow();
        assert!(pbcfields.contains_key(&(access_key, "x".to_string())));
    }

    // -----------------------------------------------------------------
    // R2-B — InstanceRepr classdef != None scaffolding.
    // -----------------------------------------------------------------

    #[test]
    fn instance_repr_new_with_classdef_stores_forward_reference_object_type() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = InstanceRepr::new(&rtyper, Some(&classdef), Flavor::Gc);

        // rclass.py:470 — classdef field.
        let stored = repr.classdef().expect("classdef must be Some");
        assert!(Rc::ptr_eq(&stored, &classdef));

        // rclass.py:474-475 — object_type is a fresh GcForwardReference.
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference when classdef != None");
        };
        assert!(fwd.resolved().is_none());

        // rclass.py:477 — lowleveltype is Ptr(self.object_type).
        let LowLevelType::Ptr(ptr) = repr.lowleveltype() else {
            panic!("lowleveltype must be Ptr");
        };
        assert!(matches!(ptr.TO, PtrTarget::ForwardReference(_)));
    }

    #[test]
    fn buildinstancerepr_unboxedvalue_without_taggedpointers_falls_back_to_plain_instance_repr() {
        use crate::annotator::annrpython::RPythonAnnotator;

        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        rtyper
            .initialize_exceptiondata()
            .expect("initialize_exceptiondata");
        let classdef = ClassDef::new_standalone("pkg.UnboxedValue", None);

        let repr =
            buildinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("plain InstanceRepr");

        let stored = repr.classdef().expect("classdef must be Some");
        assert!(Rc::ptr_eq(&stored, &classdef));
        Repr::setup(&repr).expect("plain InstanceRepr setup");
    }

    #[test]
    fn instance_repr_setup_resolves_object_type_to_struct_with_super_pointing_at_root_object() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("getinstancerepr(classdef)");

        Repr::setup(repr.as_ref()).expect("InstanceRepr._setup_repr");

        // rclass.py:548-554 — object_type resolves to Struct('pkg.C',
        // ('super', rbase.object_type)) where rbase is the root
        // InstanceRepr's OBJECT (classdef=None).
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must still be ForwardReference wrapper");
        };
        let resolved = fwd.resolved().expect("object_type must be resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        assert_eq!(body._name, "pkg.C");
        assert_eq!(body._gckind, GcKind::Gc);
        let super_field = body
            ._flds
            .get("super")
            .expect("object_type Struct must carry 'super' field");
        assert_eq!(*super_field, OBJECT.clone());
    }

    #[test]
    fn instance_repr_gc_setup_attaches_runtime_type_info_opaque_on_body_struct() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RttiC", None);
        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc)
            .expect("getinstancerepr(classdef)");

        Repr::setup(repr.as_ref()).expect("InstanceRepr._setup_repr");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!("object_type must be ForwardReference wrapper");
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!("object_type must resolve to Struct");
        };
        // rclass.py:531-532 — gc-flavor instance reprs pass rtti=True
        // so `getRuntimeTypeInfo(body)` succeeds after setup.
        let rtti = body
            ._runtime_type_info
            .as_ref()
            .expect("gc-flavor instance body must carry _runtime_type_info");
        assert_eq!(rtti.TYPE.tag, "RuntimeTypeInfo");
        assert_eq!(rtti._name.as_deref(), Some("pkg.RttiC"));
    }

    #[test]
    fn instance_repr_raw_setup_does_not_attach_runtime_type_info_opaque() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RawC", None);
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::Str("raw".to_string()));
        let repr =
            getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr raw");
        assert_eq!(repr.gcflavor(), Flavor::Raw);
        Repr::setup(repr.as_ref()).expect("setup raw");
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("raw resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert!(body._runtime_type_info.is_none());
    }

    #[test]
    fn getinstancerepr_caches_by_classdef_identity() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        let first = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("first");
        let second = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("second");
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn getinstancerepr_reads_alloc_flavor_from_classdesc() {
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.RawC", None);
        classdef
            .borrow()
            .classdesc
            .borrow()
            .pyobj
            .class_set("_alloc_flavor_", ConstValue::Str("raw".to_string()));

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("raw repr");
        assert_eq!(repr.gcflavor(), Flavor::Raw);
    }

    #[test]
    fn instance_repr_basedef_recursion_threads_parent_object_type_into_super_field() {
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));
        let base_repr = getinstancerepr(&rtyper, Some(&base), Flavor::Gc).expect("base");
        let child_repr = getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("child");

        Repr::setup(child_repr.as_ref()).expect("child setup");

        // Both reprs must now be FINISHED.
        let LowLevelType::ForwardReference(child_fwd) = child_repr.object_type() else {
            panic!();
        };
        let child_resolved = child_fwd.resolved().expect("child resolved");
        let LowLevelType::ForwardReference(base_fwd) = base_repr.object_type() else {
            panic!();
        };
        let base_resolved = base_fwd.resolved().expect("base resolved via child");

        let LowLevelType::Struct(child_body) = child_resolved else {
            panic!();
        };
        let super_field = child_body
            ._flds
            .get("super")
            .expect("child object_type must have 'super'");
        let LowLevelType::ForwardReference(super_fwd) = super_field else {
            panic!("child 'super' field must be a ForwardReference pointing at base");
        };
        assert_eq!(
            super_fwd.resolved().expect("super field resolved"),
            base_resolved
        );
    }

    // -----------------------------------------------------------------
    // R2-C — attrs iteration on ClassRepr + InstanceRepr.
    // -----------------------------------------------------------------

    fn attach_attr(
        classdef: &Rc<RefCell<ClassDef>>,
        name: &str,
        s_value: SomeValue,
        readonly: bool,
    ) {
        use crate::annotator::classdesc::Attribute;
        let mut attr = Attribute::new(name);
        attr.s_value = s_value;
        attr.readonly = readonly;
        classdef.borrow_mut().attrs.insert(name.to_string(), attr);
    }

    #[test]
    fn classrepr_setup_populates_clsfields_for_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "LIMIT",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        let clsfields = repr.clsfields();
        let (mangled, field_repr) = clsfields
            .get("LIMIT")
            .expect("readonly attr LIMIT should land in clsfields");
        assert_eq!(mangled, "cls_LIMIT");
        // rclass.py:262 — llfields append uses r.lowleveltype; Integer's
        // Repr lowleveltype is lltype.Signed.
        assert_eq!(field_repr.lowleveltype(), &LowLevelType::Signed);

        // rclass.py:275-278 — the vtable Struct body must carry the
        // `(mangled_name, r.lowleveltype)` pair after 'super'.
        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("vtable_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        let limit_field = body
            ._flds
            .get("cls_LIMIT")
            .expect("vtable must carry cls_LIMIT field");
        assert_eq!(*limit_field, LowLevelType::Signed);
    }

    #[test]
    fn classrepr_setup_skips_non_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false, // instance attr, not class attr
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        assert!(
            repr.clsfields().is_empty(),
            "non-readonly attrs must not leak into clsfields"
        );
    }

    #[test]
    fn classrepr_sorts_vtable_fields_by_reverse_size_after_name_order() {
        use crate::annotator::model::{SomeBool, SomeInteger};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(&classdef, "a_bool", SomeValue::Bool(SomeBool::new()), true);
        attach_attr(
            &classdef,
            "z_int",
            SomeValue::Integer(SomeInteger::new(false, false)),
            true,
        );

        let repr = match getclassrepr_arc(&rtyper, Some(&classdef)).expect("getclassrepr") {
            ClassReprArc::Inst(r) => r,
            _ => panic!(),
        };
        Repr::setup(repr.as_ref()).expect("setup");

        let LowLevelType::ForwardReference(fwd) = repr.vtable_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("vtable_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert_eq!(body._names[0], "super");
        assert_eq!(body._names[1], "cls_z_int");
        assert_eq!(body._names[2], "cls_a_bool");
    }

    #[test]
    fn instance_repr_setup_populates_fields_for_non_readonly_attrs() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(
            &classdef,
            "x",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(repr.as_ref()).expect("setup");

        let fields = repr.fields();
        let (mangled, field_repr) = fields.get("x").expect("instance attr x in fields");
        assert_eq!(mangled, "inst_x");
        assert_eq!(field_repr.lowleveltype(), &LowLevelType::Signed);

        // rclass.py:548-554 — object_type Struct body has inst_x pair.
        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        let x_field = body
            ._flds
            .get("inst_x")
            .expect("object_type must carry inst_x field");
        assert_eq!(*x_field, LowLevelType::Signed);
    }

    #[test]
    fn instance_repr_allinstancefields_unions_parent_chain() {
        use crate::annotator::model::SomeInteger;
        let rtyper = fresh_rtyper();
        let base = ClassDef::new_standalone("pkg.Base", None);
        attach_attr(
            &base,
            "shared",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );
        let child = ClassDef::new_standalone("pkg.Child", Some(&base));
        attach_attr(
            &child,
            "extra",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let child_repr =
            getinstancerepr(&rtyper, Some(&child), Flavor::Gc).expect("getinstancerepr child");
        Repr::setup(child_repr.as_ref()).expect("child setup");

        let all = child_repr.allinstancefields();
        assert!(
            all.contains_key("shared"),
            "parent 'shared' attr must propagate into child.allinstancefields"
        );
        assert!(
            all.contains_key("extra"),
            "child's own 'extra' attr must appear in allinstancefields"
        );

        // Fields-proper stays minimal (only child's own).
        let fields = child_repr.fields();
        assert!(!fields.contains_key("shared"));
        assert!(fields.contains_key("extra"));
    }

    #[test]
    fn instance_repr_sorts_fields_by_reverse_size_after_name_order() {
        use crate::annotator::model::{SomeBool, SomeInteger};

        let rtyper = fresh_rtyper();
        let classdef = ClassDef::new_standalone("pkg.C", None);
        attach_attr(&classdef, "a_bool", SomeValue::Bool(SomeBool::new()), false);
        attach_attr(
            &classdef,
            "z_int",
            SomeValue::Integer(SomeInteger::new(false, false)),
            false,
        );

        let repr = getinstancerepr(&rtyper, Some(&classdef), Flavor::Gc).expect("getinstancerepr");
        Repr::setup(repr.as_ref()).expect("setup");

        let LowLevelType::ForwardReference(fwd) = repr.object_type() else {
            panic!();
        };
        let resolved = fwd.resolved().expect("object_type resolved");
        let LowLevelType::Struct(body) = resolved else {
            panic!();
        };
        assert_eq!(body._names[0], "super");
        assert_eq!(body._names[1], "inst_z_int");
        assert_eq!(body._names[2], "inst_a_bool");
    }
}
