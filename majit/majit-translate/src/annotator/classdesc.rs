//! Type inference for user-defined classes.
//!
//! RPython upstream: `rpython/annotator/classdesc.py` (968 LOC).
//!
//! This commit ports the first slice of classdesc.py:
//!
//! | upstream | Rust |
//! |---|---|
//! | `Attribute` (classdesc.py:72-134) | [`Attribute`] |
//! | `ClassDef` data + structural methods (classdesc.py:136-431) | [`ClassDef`] |
//! | `InstanceSource` (classdesc.py:435-464) | [`InstanceSource`] |
//! | `NoSuchAttrError` (classdesc.py:466-468) | [`NoSuchAttrError`] |
//! | `is_mixin` (classdesc.py:471) | [`is_mixin`] |
//! | `is_primitive_type` (classdesc.py:474) | [`is_primitive_type`] |
//! | `BuiltinTypeDesc` (classdesc.py:479-485) | [`BuiltinTypeDesc`] |
//! | `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.py:957-961) | [`force_attributes_into_classes`] |
//! | `ClassDesc` (classdesc.py:488-600) | [`ClassDesc`] |
//!
//! ## PRE-EXISTING-ADAPTATION: cyclic ClassDef ↔ ClassDesc
//!
//! Upstream stores `classdef.classdesc: ClassDesc` and `classdesc.classdef:
//! ClassDef` as mutual Python references — cycle collection handles the
//! graph. The Rust port carries both as
//! `Rc<RefCell<classdesc::{ClassDef,ClassDesc}>>` with one side down-graded
//! to `Weak<...>` to avoid retain cycles. Direction: `ClassDef.classdesc`
//! is strong (ClassDef owns its ClassDesc); `ClassDesc.classdef` is
//! `Weak` (back-reference for the `getuniqueclassdef` cache slot). This
//! flips upstream's natural "ClassDesc owns ClassDef" direction — see
//! the note at [`ClassDef::new`] — but is load-bearing in Rust because
//! test ClassDefs are constructed standalone (no live bookkeeper to
//! retain ClassDesc). When the full bookkeeper lands with c2 the
//! bookkeeper retains ClassDesc through `descs[pyobj]` and this
//! detail becomes invisible.
//!
//! `subdefs` / `parentdefs` are `Weak` because they point across the
//! subclass graph.
//!
//! ## PRE-EXISTING-ADAPTATION: identity equality
//!
//! Upstream `ClassDef` equality is Python object identity (`is`): two
//! `ClassDef` structs with the same `name` are still distinct if they
//! came from different `ClassDesc`s. The Rust port carries
//! `Rc<RefCell<ClassDef>>` and equality on the `Rc` is `Rc::ptr_eq`,
//! matching upstream. Name-based comparisons must walk
//! `borrow().name`.
//!
//! ## PRE-EXISTING-ADAPTATION: HostObject class-dict reflection
//!
//! [`HostObject::Class`] exposes `class_get(name)` /
//! `class_contains(name)` as the reflection surface upstream reads via
//! `cls.__dict__.get(…)`. `ClassDesc::__init__` uses these to detect
//! `_mixin_`, `_immutable_fields_`, `__slots__`, `_attrs_`,
//! `__NOT_RPYTHON__`, and `_annspecialcase_`.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};
use std::sync::Arc;

use super::bookkeeper::{Bookkeeper, EmulatedPbcCallKey, PositionKey};
use super::description::ClassAttrFamily;
use super::model::{AnnotatorError, DescKind, SomeInteger, SomeString, SomeValue, union};
use crate::flowspace::model::{
    ConstValue, Constant, HOST_ENV, HostGetAttrError, HostObject, host_getattr,
};
use crate::tool::flattenrec::FlattenRecursion;
use crate::translator::rtyper::rclass::ClassRepr;

thread_local! {
    /// RPython `ClassDef._see_instance_flattenrec = FlattenRecursion()`
    /// (classdesc.py:402). Upstream's `FlattenRecursion` inherits
    /// `TlsClass` (tool/flattenrec.py:7-10) so the class-attribute is
    /// effectively one-per-thread; the Rust port mirrors the scope
    /// with `thread_local!`.
    static SEE_INSTANCE_FLATTENREC: FlattenRecursion<AnnotatorError> =
        FlattenRecursion::new();
}

// ---------------------------------------------------------------------------
// ClassDictEntry (classdesc.py:506 `{attr: Constant-or-Desc}`).
// ---------------------------------------------------------------------------

/// RPython `classdict[attr]` value type — upstream comment
/// (classdesc.py:506) states `{attr: Constant-or-Desc}`. Upstream relies
/// on Python `isinstance(obj, Constant)` / `isinstance(obj, Desc)`
/// dispatch at `s_get_value` (classdesc.py:786-802). The Rust port
/// closes the sum explicitly so callers match on the variant.
///
/// The `Desc` variant holds a [`super::description::DescEntry`] — this
/// is the carrier `add_source_attribute` stores when cloning a
/// `FunctionType` into a mixin class (classdesc.py:604-613).
#[derive(Clone, Debug)]
pub enum ClassDictEntry {
    /// upstream `Constant(value)` — prebuilt values.
    Constant(Constant),
    /// upstream `self.bookkeeper.newfuncdesc(value)` stored directly —
    /// preserves mixin-specific FunctionDesc identity (classdesc.py:611).
    Desc(super::description::DescEntry),
}

impl ClassDictEntry {
    /// Convenience — `Constant` cases wrap a `Constant::new(value)`.
    pub fn constant(value: ConstValue) -> Self {
        ClassDictEntry::Constant(Constant::new(value))
    }
}

// ---------------------------------------------------------------------------
// Error marker (classdesc.py:466-468).
// ---------------------------------------------------------------------------

/// RPython `class NoSuchAttrError(AnnotatorError)` (classdesc.py:466-468).
///
/// Raised by `Attribute.validate` when `__slots__` / `_attrs_` forbids
/// the name, and by `Attribute.modified` when `attr_allowed=False`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoSuchAttrError(pub AnnotatorError);

impl NoSuchAttrError {
    pub fn new(msg: impl Into<String>) -> Self {
        NoSuchAttrError(AnnotatorError::new(msg))
    }
}

impl From<NoSuchAttrError> for AnnotatorError {
    fn from(err: NoSuchAttrError) -> Self {
        err.0
    }
}

impl std::fmt::Display for NoSuchAttrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for NoSuchAttrError {}

// ---------------------------------------------------------------------------
// BuiltinTypeDesc (classdesc.py:479-485).
// ---------------------------------------------------------------------------

/// RPython `class BuiltinTypeDesc(object)` (classdesc.py:479-485).
///
/// Represents a primitive or builtin type object (e.g. `int`, `str`).
/// Stored in the bookkeeper alongside `ClassDesc` entries; upstream
/// uses it only to answer `issubclass` queries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuiltinTypeDesc {
    pub pyobj: HostObject,
}

impl BuiltinTypeDesc {
    pub fn new(pyobj: HostObject) -> Self {
        BuiltinTypeDesc { pyobj }
    }

    /// RPython `BuiltinTypeDesc.issubclass(self, other)` (classdesc.py:484-485).
    pub fn issubclass(&self, other: &BuiltinTypeDesc) -> bool {
        self.pyobj.is_subclass_of(&other.pyobj)
    }
}

// ---------------------------------------------------------------------------
// is_mixin / is_primitive_type helpers (classdesc.py:471-476).
// ---------------------------------------------------------------------------

/// RPython `is_mixin(cls)` (classdesc.py:471-472).
///
/// Upstream: `cls.__dict__.get('_mixin_', False)`. Rust port reads
/// through [`HostObject::class_get`] (populated by
/// `new_class_with_members`) and accepts `ConstValue::Bool(true)` as
/// the only truthy form.
pub fn is_mixin(cls: &HostObject) -> bool {
    matches!(cls.class_get("_mixin_"), Some(ConstValue::Bool(true)))
}

/// RPython `is_primitive_type(cls)` (classdesc.py:474-476).
///
/// Upstream checks `cls.__module__ == '__builtin__'` or
/// `issubclass(cls, base_int)`. The Rust port approximates by
/// inspecting the `qualname` prefix (`__builtin__.` or `builtins.`);
/// `base_int` subclasses are accepted via an explicit qualname-prefix
/// allow-list because we do not yet materialise `base_int` as a
/// [`HostObject`].
pub fn is_primitive_type(cls: &HostObject) -> bool {
    let qn = cls.qualname();
    qn.starts_with("__builtin__.")
        || qn.starts_with("builtins.")
        || matches!(
            qn,
            "int"
                | "bool"
                | "float"
                | "str"
                | "bytes"
                | "unicode"
                | "type"
                | "object"
                | "NoneType"
                | "tuple"
                | "list"
                | "dict"
        )
}

// ---------------------------------------------------------------------------
// FORCE_ATTRIBUTES_INTO_CLASSES (classdesc.py:957-961).
// ---------------------------------------------------------------------------

/// RPython `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.py:957-968).
///
/// Maps exception-class qualnames to the attribute annotations that
/// `ClassDesc._init_classdef` must eagerly install. Upstream keys on
/// the live Python class object; the Rust port keys on the qualname
/// because that's the only identity we can carry without Python
/// runtime. The `WindowsError` entry is omitted (classdesc.py:963-968)
/// because the Rust port targets a portable host.
pub fn force_attributes_into_classes() -> HashMap<&'static str, HashMap<&'static str, SomeValue>> {
    let mut map = HashMap::new();
    let mut env_error = HashMap::new();
    env_error.insert("errno", SomeValue::Integer(SomeInteger::default()));
    env_error.insert("strerror", SomeValue::String(SomeString::new(true, false)));
    env_error.insert("filename", SomeValue::String(SomeString::new(true, false)));
    map.insert("EnvironmentError", env_error);
    map
}

// ---------------------------------------------------------------------------
// AttrSource (classdesc.py: implicit via duck-typing on InstanceSource /
// ClassDesc).
// ---------------------------------------------------------------------------

/// RPython dispatch between class-level sources and prebuilt-instance
/// sources inside `Attribute.add_constant_source` (classdesc.py:87-93)
/// and `ClassDef.add_source_for_attribute` (classdesc.py:189-220).
///
/// Upstream both `ClassDesc` and `InstanceSource` expose
/// `.instance_level` + `.s_get_value(classdef, name)`; the Rust port
/// closes the set explicitly here so callers pattern-match on the
/// source kind.
#[derive(Clone, Debug)]
pub enum AttrSource {
    /// `classsources[attr] = self` (classdesc.py:687-688). The source
    /// is the owning `ClassDesc`; `instance_level = False`.
    Class(Weak<RefCell<ClassDesc>>),
    /// `InstanceSource(bookkeeper, obj)` (classdesc.py:411). The source
    /// is a prebuilt instance; `instance_level = True`.
    Instance(InstanceSource),
}

impl AttrSource {
    pub fn instance_level(&self) -> bool {
        matches!(self, AttrSource::Instance(_))
    }

    /// Dispatch to `ClassDesc.s_get_value` / `InstanceSource.s_get_value`.
    ///
    /// Class-level sources delegate to [`ClassDesc::s_get_value`]
    /// (classdesc.py:784-802). Instance-level sources still require
    /// live instance reflection that isn't available in the Rust port.
    pub fn s_get_value(
        &self,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        name: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        match self {
            AttrSource::Class(w) => {
                let cdesc = w.upgrade().ok_or_else(|| {
                    AnnotatorError::new("AttrSource::Class: ClassDesc backlink dropped")
                })?;
                ClassDesc::s_get_value(&cdesc, classdef, name)
            }
            AttrSource::Instance(inst) => inst.s_get_value(classdef, name),
        }
    }
}

// ---------------------------------------------------------------------------
// InstanceSource (classdesc.py:435-464).
// ---------------------------------------------------------------------------

/// RPython `class InstanceSource(object)` (classdesc.py:435-464).
///
/// Wraps a prebuilt Python instance seen by the annotator. The Rust
/// port stores the `Bookkeeper` as `Weak` to match upstream's Python
/// reference and captures the live object via its [`HostObject`]
/// carrier.
#[derive(Clone, Debug)]
pub struct InstanceSource {
    pub bookkeeper: Weak<Bookkeeper>,
    pub obj: HostObject,
}

impl InstanceSource {
    /// RPython `instance_level = True` (classdesc.py:436).
    pub const INSTANCE_LEVEL: bool = true;

    pub fn new(bookkeeper: &Rc<Bookkeeper>, obj: HostObject) -> Self {
        InstanceSource {
            bookkeeper: Rc::downgrade(bookkeeper),
            obj,
        }
    }

    /// RPython `InstanceSource.s_get_value(classdef, name)`
    /// (classdesc.py:442-451).
    ///
    pub fn s_get_value(
        &self,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        name: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        let value = match host_getattr(&self.obj, name) {
            Ok(value) => value,
            Err(HostGetAttrError::Missing) => {
                if let Some(classdef) = classdef {
                    let classdef = classdef.borrow();
                    let all_enforced_attrs = &classdef.classdesc.borrow().all_enforced_attrs;
                    if all_enforced_attrs
                        .as_ref()
                        .is_some_and(|attrs| attrs.contains(name))
                    {
                        return Ok(SomeValue::Impossible);
                    }
                }
                return Err(AnnotatorError::new(format!(
                    "AttributeError: {:?} object has no attribute {:?}",
                    self.obj.qualname(),
                    name
                )));
            }
            Err(HostGetAttrError::Unsupported) => {
                return Err(AnnotatorError::new(format!(
                    "InstanceSource.s_get_value({:?}): host getattr is unsupported",
                    self.obj.qualname()
                )));
            }
        };
        let bk = self
            .bookkeeper
            .upgrade()
            .ok_or_else(|| AnnotatorError::new("InstanceSource.s_get_value: Bookkeeper dropped"))?;
        bk.immutablevalue(&value)
    }

    /// RPython `InstanceSource.all_instance_attributes(self)`
    /// (classdesc.py:453-464).
    pub fn all_instance_attributes(&self) -> Result<Vec<String>, AnnotatorError> {
        let mut result = self.obj.instance_dict_keys();
        let Some(tp) = self.obj.instance_class() else {
            return Ok(result);
        };
        if !tp.is_class() {
            return Ok(result);
        }
        let Some(mro) = tp.mro() else {
            return Ok(result);
        };
        for basetype in mro {
            let Some(slots) = basetype.class_get("__slots__") else {
                continue;
            };
            if matches!(slots.truthy(), Some(false)) {
                continue;
            }
            match slots {
                ConstValue::Str(slot) => result.push(slot),
                ConstValue::Tuple(items) | ConstValue::List(items) => {
                    for item in items {
                        match item {
                            ConstValue::Str(slot) => result.push(slot),
                            _ => {
                                return Err(AnnotatorError::new(
                                    "__slots__ must be a sequence of strings",
                                ));
                            }
                        }
                    }
                }
                _ => {
                    return Err(AnnotatorError::new(
                        "__slots__ must be a string or a sequence of strings",
                    ));
                }
            }
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Attribute (classdesc.py:72-134).
// ---------------------------------------------------------------------------

/// RPython `class Attribute(object)` (classdesc.py:72-134).
///
/// Records the merged `SomeValue` annotation for one attribute name
/// together with the set of read positions. Two invariants from the
/// upstream docstring (classdesc.py:23-41) are carried by
/// [`ClassDef::_generalize_attr`]:
///
/// * (A) if `x.attr` is read/written on an instance of class A, then A
///   or a parent class of A owns an `Attribute` for `attr`.
/// * (I) if B subclasses A, at most one of them has an `Attribute` for
///   `attr` — subclass merges hoist into the parent.
#[derive(Clone, Debug)]
pub struct Attribute {
    /// RPython `self.name` (classdesc.py:81).
    pub name: String,
    /// RPython `self.s_value = s_ImpossibleValue` (classdesc.py:82).
    pub s_value: SomeValue,
    /// RPython `self.readonly = True` (classdesc.py:83).
    pub readonly: bool,
    /// RPython `self.attr_allowed = True` (classdesc.py:84).
    pub attr_allowed: bool,
    /// RPython `self.read_locations = set()` (classdesc.py:85).
    ///
    /// Upstream stores positions as tuples of `(FunctionGraph, Block,
    /// op_index)`; the Rust port uses the ported [`PositionKey`]
    /// identity.
    pub read_locations: HashSet<PositionKey>,
}

impl Attribute {
    /// RPython `Attribute.__init__(self, name)` (classdesc.py:79-85).
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        assert!(
            name != "__class__",
            "Attribute name must not be '__class__' (classdesc.py:80)"
        );
        Attribute {
            name,
            s_value: SomeValue::Impossible,
            readonly: true,
            attr_allowed: true,
            read_locations: HashSet::new(),
        }
    }

    /// RPython `Attribute.add_constant_source(self, classdef, source)`
    /// (classdesc.py:87-93).
    pub fn add_constant_source(
        &mut self,
        classdef: &Rc<RefCell<ClassDef>>,
        source: &AttrSource,
    ) -> Result<(), AnnotatorError> {
        let s_value = source.s_get_value(Some(classdef), &self.name)?;
        if source.instance_level() {
            // upstream: "a prebuilt instance source forces readonly=False".
            self.modified(Some(classdef))?;
        }
        let s_new_value =
            union(&self.s_value, &s_value).map_err(|e| AnnotatorError::new(e.to_string()))?;
        self.s_value = s_new_value;
        Ok(())
    }

    /// RPython `Attribute.merge(self, other, classdef)`
    /// (classdesc.py:95-101).
    pub fn merge(
        &mut self,
        other: &Attribute,
        classdef: &Rc<RefCell<ClassDef>>,
    ) -> Result<(), AnnotatorError> {
        assert_eq!(self.name, other.name, "Attribute.merge name mismatch");
        let s_new_value =
            union(&self.s_value, &other.s_value).map_err(|e| AnnotatorError::new(e.to_string()))?;
        self.s_value = s_new_value;
        if !other.readonly {
            self.modified(Some(classdef))?;
        }
        self.read_locations
            .extend(other.read_locations.iter().cloned());
        Ok(())
    }

    /// RPython `Attribute.validate(self, homedef)` (classdesc.py:103-120).
    ///
    /// The `SomePBC` / `MethodDesc` branch triggers
    /// [`ClassDef::check_missing_attribute_update`], which is a c3
    /// dependency; this implementation skips that branch with a
    /// comment. The `all_enforced_attrs` branch is handled here in
    /// full.
    pub fn validate(&mut self, homedef: &Rc<RefCell<ClassDef>>) -> Result<(), NoSuchAttrError> {
        // upstream classdesc.py:106-111 — SomePBC / MethodDesc branch.
        if let SomeValue::PBC(pbc) = &self.s_value {
            if let Ok(DescKind::Method) = pbc.get_kind() {
                // upstream:
                //     if homedef.classdesc.read_attribute(attr, None) is None:
                //         homedef.check_missing_attribute_update(attr)
                //
                // classdesc.read_attribute + ClassDef.check_missing_attribute_
                // update land with c3; skip silently here. The call site is
                // preserved as a comment above so the follow-up patch is pure
                // addition.
            }
        }

        // upstream classdesc.py:113-120 — __slots__ / _attrs_ enforcement.
        let homedef_ref = homedef.borrow();
        let classdesc_rc = homedef_ref.classdesc.clone();
        let classdesc_ref = classdesc_rc.borrow();
        if let Some(enforced) = &classdesc_ref.all_enforced_attrs {
            if !enforced.contains(&self.name) {
                self.attr_allowed = false;
                if !self.readonly {
                    return Err(NoSuchAttrError::new(format!(
                        "the attribute {:?} goes here to {:?}, but it is \
                         forbidden here",
                        self.name, homedef_ref.name
                    )));
                }
            }
        }
        Ok(())
    }

    /// RPython `Attribute.modified(self, classdef='?')` (classdesc.py:122-134).
    ///
    /// The `attr_allowed=False` path invokes `bookkeeper.getattr_locations`
    /// to render the list of read locations in the `NoSuchAttrError`
    /// message; when the backlink is missing the error is returned with
    /// an empty location list.
    pub fn modified(
        &mut self,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
    ) -> Result<(), AnnotatorError> {
        self.readonly = false;
        if !self.attr_allowed {
            // upstream:
            //   from rpython.annotator.bookkeeper import getbookkeeper
            //   bk = getbookkeeper()
            //   classdesc = classdef.classdesc
            //   locations = bk.getattr_locations(classdesc, self.name)
            //   raise NoSuchAttrError(...formatted message...)
            let name = classdef
                .map(|c| c.borrow().name.clone())
                .unwrap_or_else(|| "?".to_string());
            let locations_str = match classdef {
                Some(cd) => {
                    let bk_opt = cd.borrow().bookkeeper.upgrade();
                    let classdesc = cd.borrow().classdesc.clone();
                    match bk_opt {
                        Some(bk) => match bk.getattr_locations(&classdesc, &self.name) {
                            Ok(locs) => locs
                                .iter()
                                .map(|l| format!("{:?}", l))
                                .collect::<Vec<_>>()
                                .join("\n"),
                            Err(_) => String::new(),
                        },
                        None => String::new(),
                    }
                }
                None => String::new(),
            };
            return Err(AnnotatorError::new(format!(
                "Attribute {:?} on {:?} should be read-only.\n\
                 This error can be caused by another 'getattr' that promoted\n\
                 the attribute here; the list of read locations is:\n{}",
                self.name, name, locations_str
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ClassDesc (classdesc.py:488-600).
// ---------------------------------------------------------------------------

/// RPython `class ClassDesc(Desc)` (classdesc.py:488-600).
///
/// Structural port of the annotator-visible surface: `__init__` body
/// (mixin resolution, slots / `_attrs_` detection,
/// `is_builtin_exception_class` pre-population, `_immutable_fields_`
/// enforcement), `lookup`, `read_attribute`, `s_read_attribute`,
/// `s_get_value`, `add_source_attribute`, `add_mixins`,
/// `add_sources_for_class`, `getclassdef`, `getuniqueclassdef`,
/// `pycall`, `consider_call_site` (with `getallbases` /
/// `getcommonbase` / `MethodDesc` initdesc recursion),
/// `getattrfamily`, `mergeattrfamilies`, `maybe_return_immutable_list`,
/// and exception-class predicates.
#[derive(Debug)]
pub struct ClassDesc {
    /// RPython `self.pyobj` — the live Python class object.
    pub pyobj: HostObject,
    /// RPython `self.name` — upstream `cls.__module__ + '.' + cls.__name__`.
    pub name: String,
    /// RPython `self.basedesc` — the base `ClassDesc` or `None`.
    pub basedesc: Option<Rc<RefCell<ClassDesc>>>,
    /// RPython `self.classdef` — back-reference to the unique
    /// `ClassDef`. Populated lazily by `getuniqueclassdef`. Stored as
    /// `Weak` so the ClassDef strongly owning this ClassDesc doesn't
    /// retain itself through this back-reference (see module doc).
    pub classdef: Option<Weak<RefCell<ClassDef>>>,
    /// RPython `self.all_enforced_attrs` — `None` or an attribute name
    /// set. `None` means attribute accesses are unconstrained;
    /// `Some(set)` enforces `__slots__` / `_attrs_`.
    pub all_enforced_attrs: Option<HashSet<String>>,
    /// RPython `self.classdict` — `{attr: Constant-or-Desc}`
    /// (classdesc.py:506). The [`ClassDictEntry`] sum carries either
    /// constant values or stored `DescEntry`s produced by
    /// `add_source_attribute`'s mixin-FunctionType branch.
    pub classdict: HashMap<String, ClassDictEntry>,
    /// RPython `self.immutable_fields` — `set(cls._immutable_fields_)`.
    pub immutable_fields: HashSet<String>,
    /// RPython class attribute `instance_level = False`
    /// (classdesc.py:490).
    pub instance_level: bool,
    /// RPython `self._detect_invalid_attrs` (classdesc.py:826) — lazily
    /// populated attribute-name set used by
    /// [`Self::maybe_return_immutable_list`] to detect subclass-declared
    /// `_immutable_fields_` entries that were migrated to a superclass.
    pub detect_invalid_attrs: Option<HashSet<String>>,
    /// Back-reference to the bookkeeper. Weak to avoid cycles.
    pub bookkeeper: Weak<Bookkeeper>,
}

impl ClassDesc {
    /// Data-only shell used by unit tests that need a bare `ClassDesc`
    /// without running the full `__init__` (classdesc.py:494-588)
    /// mixin / slots / exception-pre-population body. Production
    /// callers use [`Self::new`].
    pub fn new_shell(bookkeeper: &Rc<Bookkeeper>, pyobj: HostObject, name: String) -> Self {
        ClassDesc {
            pyobj,
            name,
            basedesc: None,
            classdef: None,
            all_enforced_attrs: None,
            classdict: HashMap::new(),
            immutable_fields: HashSet::new(),
            instance_level: false,
            detect_invalid_attrs: None,
            bookkeeper: Rc::downgrade(bookkeeper),
        }
    }

    /// RPython `ClassDesc.issubclass(self, other)` (classdesc.py:746-747).
    pub fn issubclass(&self, other: &ClassDesc) -> bool {
        self.pyobj.is_subclass_of(&other.pyobj)
    }

    /// RPython `ClassDesc.is_exception_class(self)` (classdesc.py:735-736).
    ///
    /// Upstream: `issubclass(self.pyobj, BaseException)`. The Rust port
    /// looks up `BaseException` via [`HOST_ENV`] and calls
    /// [`HostObject::is_subclass_of`] — same semantics.
    pub fn is_exception_class(&self) -> bool {
        match HOST_ENV.lookup_builtin("BaseException") {
            Some(base_exc) => self.pyobj.is_subclass_of(&base_exc),
            None => false,
        }
    }

    /// RPython `ClassDesc.is_builtin_exception_class(self)`
    /// (classdesc.py:738-744).
    ///
    /// Upstream:
    /// ```python
    /// if self.is_exception_class():
    ///     if self.pyobj.__module__ == 'exceptions':
    ///         return True
    ///     if issubclass(self.pyobj, AssertionError):
    ///         return True
    /// return False
    /// ```
    /// Rust port: `__module__ == 'exceptions'` becomes qualname with no
    /// dot (HOST_ENV stores `BaseException`, `TypeError`, … under their
    /// short names; user classes are `pkg.Cls`). AssertionError check
    /// routes through HOST_ENV.
    pub fn is_builtin_exception_class(&self) -> bool {
        if !self.is_exception_class() {
            return false;
        }
        if !self.pyobj.qualname().contains('.') {
            return true;
        }
        if let Some(assert_err) = HOST_ENV.lookup_builtin("AssertionError") {
            if self.pyobj.is_subclass_of(&assert_err) {
                return true;
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Full __init__ body + add_source_attribute / add_mixins /
    // add_sources_for_class / _init_classdef / getuniqueclassdef —
    // c2 step 2c-ii.
    // -----------------------------------------------------------------------

    /// RPython `ClassDesc.__init__(self, bookkeeper, cls, name=None,
    /// basedesc=None, classdict=None)` (classdesc.py:494-588).
    ///
    /// Returns the fully-initialised classdesc as `Rc<RefCell<Self>>`
    /// because mixin resolution + `add_source_attribute` recursion must
    /// mutate the classdict through interior mutability, and the
    /// recursive `bookkeeper.getdesc(base)` call needs to retain the
    /// fresh `Rc` identity for `me.basedesc`.
    pub fn new(
        bookkeeper: &Rc<Bookkeeper>,
        cls: HostObject,
        name: Option<String>,
        basedesc: Option<Rc<RefCell<ClassDesc>>>,
        classdict: Option<HashMap<String, ClassDictEntry>>,
    ) -> Result<Rc<RefCell<Self>>, AnnotatorError> {
        // classdesc.py:497-498 — __NOT_RPYTHON__ guard.
        if cls.class_has("__NOT_RPYTHON__") {
            return Err(AnnotatorError::new(format!(
                "Bad class {:?} (carries __NOT_RPYTHON__ marker)",
                cls.qualname()
            )));
        }

        // classdesc.py:500-502 — name default = cls.__module__ + '.' + cls.__name__.
        let name = name.unwrap_or_else(|| cls.qualname().to_string());

        // classdesc.py:507-510 — _annspecialcase_ is unsupported.
        if let Some(tag) = cls.class_get("_annspecialcase_") {
            let truthy = match &tag {
                ConstValue::Str(s) => !s.is_empty(),
                ConstValue::Bool(b) => *b,
                ConstValue::None => false,
                _ => true,
            };
            if truthy {
                return Err(AnnotatorError::new(
                    "Class specialization has been removed. The \
                     '_annspecialcase_' class tag is now unsupported.",
                ));
            }
        }

        // classdesc.py:513-515 — is_mixin guard on the class itself.
        if is_mixin(&cls) {
            return Err(AnnotatorError::new(format!(
                "cannot use directly the class {:?} because it is a _mixin_",
                cls.qualname()
            )));
        }

        // classdesc.py:518-526 — baselist + BaseException/Exception hack.
        let object_cls = HOST_ENV.lookup_builtin("object");
        let exc_cls = HOST_ENV.lookup_builtin("Exception");
        let base_exc_cls = HOST_ENV.lookup_builtin("BaseException");

        let mut baselist: Vec<HostObject> =
            cls.class_bases().map(|b| b.to_vec()).unwrap_or_default();
        if let Some(ref exc) = exc_cls {
            if &cls == exc {
                baselist.clear();
            } else if let Some(ref base_exc) = base_exc_cls {
                if baselist.len() == 1 && &baselist[0] == base_exc {
                    baselist = vec![exc.clone()];
                }
            }
        }

        // classdesc.py:528-535 — _immutable_fields_ extraction (reject str).
        let immutable_fields = Self::extract_immutable_fields(&cls)?;

        // classdesc.py:537-554 — mixin / base partitioning.
        let mut mixins_before: Vec<HostObject> = Vec::new();
        let mut mixins_after: Vec<HostObject> = Vec::new();
        let mut base: Option<HostObject> = None; // None ≡ `object`.
        for b1 in &baselist {
            if object_cls.as_ref() == Some(b1) {
                continue;
            }
            if is_mixin(b1) {
                if base.is_none() {
                    mixins_before.push(b1.clone());
                } else {
                    mixins_after.push(b1.clone());
                }
            } else {
                if base.is_some() {
                    return Err(AnnotatorError::new(format!(
                        "multiple inheritance only supported with _mixin_: {:?}",
                        cls.qualname()
                    )));
                }
                base = Some(b1.clone());
            }
        }
        if !mixins_before.is_empty() && !mixins_after.is_empty() {
            return Err(AnnotatorError::new(format!(
                "unsupported: class {:?} has mixin bases both before and \
                 after the regular base",
                cls.qualname()
            )));
        }

        // Build `self` — subsequent mutation flows through interior mutability.
        let me = Rc::new(RefCell::new(ClassDesc {
            pyobj: cls.clone(),
            name,
            basedesc,
            classdef: None,
            all_enforced_attrs: None,
            classdict: classdict.unwrap_or_default(),
            immutable_fields,
            instance_level: false,
            detect_invalid_attrs: None,
            bookkeeper: Rc::downgrade(bookkeeper),
        }));

        // classdesc.py:555-557 — add_mixins(after, check_not_in=base); add_mixins(before); add_sources_for_class(cls).
        let check_not_in_after = base.clone().or_else(|| object_cls.clone());
        Self::add_mixins(&me, &mixins_after, check_not_in_after.as_ref())?;
        Self::add_mixins(&me, &mixins_before, object_cls.as_ref())?;
        Self::add_sources_for_class(&me, &cls)?;

        // classdesc.py:559-560 — resolve real base via bookkeeper.getdesc.
        if let Some(base_host) = base.as_ref() {
            if object_cls.as_ref() != Some(base_host) {
                let base_entry = bookkeeper.getdesc(base_host)?;
                let base_cd = base_entry.as_class().ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "ClassDesc.__init__: base class {:?} not routed as ClassDesc",
                        base_host.qualname()
                    ))
                })?;
                me.borrow_mut().basedesc = Some(base_cd);
            }
        }

        // classdesc.py:562-575 — __slots__ / _attrs_ → all_enforced_attrs.
        if cls.class_has("__slots__") || cls.class_has("_attrs_") {
            let mut attrs: HashSet<String> = HashSet::new();
            for decl_name in ["__slots__", "_attrs_"] {
                if let Some(decl) = cls.class_get(decl_name) {
                    let names = Self::extract_name_sequence(decl_name, &decl)?;
                    for n in names {
                        attrs.insert(n);
                    }
                }
            }
            let basedesc_opt = me.borrow().basedesc.clone();
            if let Some(basedesc_rc) = basedesc_opt {
                let b = basedesc_rc.borrow();
                match b.all_enforced_attrs.as_ref() {
                    Some(base_attrs) => {
                        for n in base_attrs {
                            attrs.insert(n.clone());
                        }
                    }
                    None => {
                        return Err(AnnotatorError::new(format!(
                            "{:?} has slots or _attrs_, but not its base class",
                            cls.qualname()
                        )));
                    }
                }
            }
            me.borrow_mut().all_enforced_attrs = Some(attrs);
        }

        // classdesc.py:577-580 — builtin exception class → empty enforced attrs,
        // unless FORCE_ATTRIBUTES_INTO_CLASSES whitelists the class.
        let need_force_empty = {
            let self_ref = me.borrow();
            self_ref.is_builtin_exception_class() && self_ref.all_enforced_attrs.is_none()
        };
        if need_force_empty {
            let force_map = force_attributes_into_classes();
            if !force_map.contains_key(cls.qualname()) {
                me.borrow_mut().all_enforced_attrs = Some(HashSet::new());
            }
        }

        // classdesc.py:582-588 — _must_be_light_finalizer_ check. The
        // inner `getattr(cls.__del__, '_must_be_light_finalizer_',
        // False)` probe walks a method-level attribute the HostObject
        // model does not expose; the class-level side is kept so the
        // method-level guard drops in once method-attr reflection lands.
        if cls.class_get("_must_be_light_finalizer_").is_some() && cls.class_has("__del__") {
            // method-attr reflection not modelled on HostObject yet.
        }

        Ok(me)
    }

    /// RPython `ClassDesc.add_source_attribute(self, name, value,
    /// mixin=False)` (classdesc.py:590-634).
    ///
    /// Property branch (classdesc.py:591-602) is a line-by-line port:
    /// fget / fset are stored as `name__getter__` / `name__setter__`
    /// hidden functions before the property object itself is retained
    /// under `name`. `staticmethod` / `classmethod` descriptors are
    /// carried through the `HostObject` wrapper variants so
    /// `s_get_value` can preserve the upstream binding rules.
    /// FunctionType + mixin uses
    /// [`Bookkeeper::newfuncdesc`] to preserve mixin-specific
    /// FunctionDesc identity (classdesc.py:608-613).
    pub fn add_source_attribute(
        this: &Rc<RefCell<Self>>,
        name: &str,
        value: ConstValue,
        mixin: bool,
    ) -> Result<(), AnnotatorError> {
        // classdesc.py:591-602 — property branch.
        if let ConstValue::HostObject(ref host) = value {
            if host.is_property() {
                // upstream: `if value.fget is not None:`
                //              `newname = name + '__getter__'`
                //              `func = func_with_new_name(value.fget, newname)`
                //              `self.add_source_attribute(newname, func, mixin)`.
                if let Some(fget) = host.property_fget() {
                    let newname = format!("{name}__getter__");
                    let renamed = fget
                        .renamed_user_function(&newname)
                        .unwrap_or_else(|| fget.clone());
                    Self::add_source_attribute(
                        this,
                        &newname,
                        ConstValue::HostObject(renamed),
                        mixin,
                    )?;
                }
                if let Some(fset) = host.property_fset() {
                    let newname = format!("{name}__setter__");
                    let renamed = fset
                        .renamed_user_function(&newname)
                        .unwrap_or_else(|| fset.clone());
                    Self::add_source_attribute(
                        this,
                        &newname,
                        ConstValue::HostObject(renamed),
                        mixin,
                    )?;
                }
                // upstream: `self.classdict[name] = Constant(value)` —
                // the property object itself is retained for
                // `_find_property_meth` to pattern-match at transform
                // time.
                this.borrow_mut()
                    .classdict
                    .insert(name.to_string(), ClassDictEntry::constant(value));
                return Ok(());
            }
        }

        // classdesc.py:604-618 — FunctionType branch.
        if let ConstValue::HostObject(ref host) = value {
            if host.is_user_function() {
                // upstream: if not hasattr(value, 'class_'): value.class_ =
                // self.pyobj (debug-only attribute; Rust port skips —
                // HostObject is immutable-keyed on the inner Arc).
                if mixin {
                    let bk = this.borrow().bookkeeper.upgrade().ok_or_else(|| {
                        AnnotatorError::new("ClassDesc.add_source_attribute: bookkeeper dropped")
                    })?;
                    let funcdesc = bk.newfuncdesc(host)?;
                    let entry = super::description::DescEntry::Function(funcdesc);
                    this.borrow_mut()
                        .classdict
                        .insert(name.to_string(), ClassDictEntry::Desc(entry));
                    return Ok(());
                }
                // non-mixin: falls through to Constant storage.
            }
        }

        // classdesc.py:619-622 — staticmethod + mixin. Upstream clones
        // the wrapped function so each mixed-in class gets a distinct
        // function object identity. Mirror that by cloning the inner
        // UserFunction host when possible, then re-wrapping it in a
        // fresh staticmethod host object.
        let value = if let ConstValue::HostObject(ref host) = value {
            if host.is_staticmethod() && mixin {
                let wrapped = host.staticmethod_func().ok_or_else(|| {
                    AnnotatorError::new("ClassDesc.add_source_attribute: malformed staticmethod")
                })?;
                let cloned_wrapped = if let Some(graph_func) = wrapped.user_function() {
                    HostObject::new_user_function(graph_func.clone())
                } else {
                    wrapped.clone()
                };
                ConstValue::HostObject(HostObject::new_staticmethod(
                    host.qualname().to_string(),
                    cloned_wrapped,
                ))
            } else {
                value
            }
        } else {
            value
        };

        // classdesc.py:624-626 — MemberDescriptor skip. HostObject does
        // not emit MemberDescriptors so nothing to guard against.

        // classdesc.py:627-633 — __init__ on a builtin exception class
        // is dropped unless the method itself is registered in
        // `BUILTIN_ANALYZERS`. Upstream:
        //
        //     if name == '__init__' and self.is_builtin_exception_class():
        //         from rpython.annotator.builtin import BUILTIN_ANALYZERS
        //         value = getattr(value, 'im_func', value)
        //         if value not in BUILTIN_ANALYZERS:
        //             return
        //
        // Rust keys the analyser registry on HostObject qualname (see
        // `super::builtin` module header), so the membership test
        // becomes `super::builtin::is_registered(host.qualname())`.
        if name == "__init__" {
            let is_builtin_exc = this.borrow().is_builtin_exception_class();
            if is_builtin_exc {
                let registered = match &value {
                    ConstValue::HostObject(host) => super::builtin::is_registered(host.qualname()),
                    _ => false,
                };
                if !registered {
                    return Ok(());
                }
            }
        }

        this.borrow_mut()
            .classdict
            .insert(name.to_string(), ClassDictEntry::constant(value));
        Ok(())
    }

    /// RPython `ClassDesc.add_mixins(self, mixins, check_not_in=object)`
    /// (classdesc.py:636-662).
    ///
    /// Synthesises a throwaway `HostObject::new_class("__tmp_mixin_mro__",
    /// mixins + object)` and reads `.mro()` to recover the C3-ordered
    /// mixin chain. Upstream does the same with
    /// `type('tmp', tuple(mixins) + (object,), {}).__mro__`.
    pub fn add_mixins(
        this: &Rc<RefCell<Self>>,
        mixins: &[HostObject],
        check_not_in: Option<&HostObject>,
    ) -> Result<(), AnnotatorError> {
        if mixins.is_empty() {
            return Ok(());
        }
        let object_cls = HOST_ENV
            .lookup_builtin("object")
            .ok_or_else(|| AnnotatorError::new("HOST_ENV missing builtin `object`"))?;
        let mut tmp_bases: Vec<HostObject> = mixins.to_vec();
        tmp_bases.push(object_cls.clone());
        let tmp = HostObject::new_class("__tmp_mixin_mro__", tmp_bases);
        let full_mro = tmp.mro().ok_or_else(|| {
            AnnotatorError::new("add_mixins: C3 linearisation conflict among mixin bases")
        })?;
        // strip first (`tmp`) and last (`object`) — upstream classdesc.py:641-642.
        let end = full_mro.len().saturating_sub(1);
        let mixin_mro: &[HostObject] = if end > 1 { &full_mro[1..end] } else { &[] };

        // classdesc.py:644-651 — skip = {names from check_not_in's MRO dicts}.
        let mut skip: HashSet<String> = HashSet::new();
        if let Some(cni) = check_not_in {
            Self::collect_mro_dict_keys(cni, &mut skip);
        }

        // classdesc.py:653-662 — reversed(mro) iterate → add_source_attribute.
        for base in mixin_mro.iter().rev() {
            if !is_mixin(base) {
                return Err(AnnotatorError::new(format!(
                    "Mixin class {:?} has non mixin base class {:?}",
                    mixins
                        .iter()
                        .map(|m| m.qualname().to_string())
                        .collect::<Vec<_>>(),
                    base.qualname()
                )));
            }
            for (name, value) in base.class_dict_items() {
                if skip.contains(&name) {
                    continue;
                }
                Self::add_source_attribute(this, &name, value, true)?;
            }
            if base.class_has("_immutable_fields_") {
                if let Ok(fields) = Self::extract_immutable_fields(base) {
                    for f in fields {
                        this.borrow_mut().immutable_fields.insert(f);
                    }
                }
            }
        }
        Ok(())
    }

    /// RPython `ClassDesc.add_sources_for_class(self, cls)`
    /// (classdesc.py:665-667).
    pub fn add_sources_for_class(
        this: &Rc<RefCell<Self>>,
        cls: &HostObject,
    ) -> Result<(), AnnotatorError> {
        for (name, value) in cls.class_dict_items() {
            Self::add_source_attribute(this, &name, value, false)?;
        }
        Ok(())
    }

    /// RPython `ClassDesc._init_classdef(self)` (classdesc.py:672-697).
    ///
    /// The final `__del__` branch mirrors upstream's
    /// `bookkeeper.emulate_pbc_call(classdef, s_func, args_s)` exactly,
    /// keyed by the unique [`ClassDef`].
    pub fn _init_classdef(
        this: &Rc<RefCell<Self>>,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        let bk =
            this.borrow().bookkeeper.upgrade().ok_or_else(|| {
                AnnotatorError::new("ClassDesc._init_classdef: bookkeeper dropped")
            })?;
        // upstream classdesc.py:673 — ClassDef(bookkeeper, self).
        let classdef = ClassDef::new(&bk, this);
        // upstream classdesc.py:674 — bookkeeper.classdefs.append(classdef).
        bk.register_classdef(classdef.clone());
        // upstream classdesc.py:675 — self.classdef = classdef. Stored as
        // Weak inside ClassDef::new; re-install here to be explicit.
        this.borrow_mut().classdef = Some(Rc::downgrade(&classdef));

        // classdesc.py:679-682 — FORCE_ATTRIBUTES_INTO_CLASSES override.
        let force_map = force_attributes_into_classes();
        let qualname = this.borrow().pyobj.qualname().to_string();
        if let Some(overrides) = force_map.get(qualname.as_str()) {
            for (attr_name, s_value) in overrides {
                ClassDef::generalize_attr(&classdef, attr_name, Some(s_value.clone()))?;
                // upstream: classdef.find_attribute(name).modified(classdef)
                let owner = ClassDef::locate_attribute(&classdef, attr_name)?;
                let mut owner_mut = owner.borrow_mut();
                let attr = owner_mut.attrs.get_mut(*attr_name).ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "_init_classdef: attribute {:?} missing after generalize",
                        attr_name
                    ))
                })?;
                attr.modified(Some(&classdef))?;
            }
        }

        // classdesc.py:686-689 — classsources = {attr: self for attr in classdict}.
        let source = AttrSource::Class(Rc::downgrade(this));
        let attr_names: Vec<String> = this.borrow().classdict.keys().cloned().collect();
        let mut classsources: HashMap<String, AttrSource> = HashMap::new();
        for n in attr_names {
            classsources.insert(n, source.clone());
        }
        ClassDef::setup(&classdef, classsources)?;

        // classdesc.py:691-696 — annotate `__del__` if present.
        if this.borrow().classdict.contains_key("__del__") {
            let s_func = Self::s_read_attribute(this, "__del__")?;
            let args_s = [SomeValue::Instance(super::model::SomeInstance::new(
                Some(classdef.clone()),
                false,
                std::collections::BTreeMap::new(),
            ))];
            let s = bk.emulate_pbc_call(
                EmulatedPbcCallKey::ClassDef(super::description::ClassDefKey::from_classdef(
                    &classdef,
                )),
                &s_func,
                &args_s,
                &[],
                None,
            )?;
            assert!(super::model::s_none().contains(&s));
        }
        Ok(classdef)
    }

    /// RPython `ClassDesc.getuniqueclassdef(self)` (classdesc.py:699-702).
    pub fn getuniqueclassdef(
        this: &Rc<RefCell<Self>>,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        let existing = this.borrow().classdef.as_ref().and_then(|w| w.upgrade());
        match existing {
            Some(cd) => Ok(cd),
            None => Self::_init_classdef(this),
        }
    }

    /// RPython `ClassDesc.pycall(self, whence, args, s_previous_result, op=None)`
    /// (classdesc.py:704-733).
    ///
    /// ```python
    /// def pycall(self, whence, args, s_previous_result, op=None):
    ///     classdef = self.getuniqueclassdef()
    ///     s_instance = SomeInstance(classdef)
    ///     s_init = self.s_read_attribute('__init__')
    ///     if isinstance(s_init, SomeImpossibleValue):
    ///         if not self.is_exception_class():
    ///             try:
    ///                 args.fixedunpack(0)
    ///             except ValueError:
    ///                 raise AnnotatorError("default __init__ takes no argument"
    ///                                      " (class %s)" % (self.name,))
    ///         elif self.pyobj is Exception:
    ///             try:
    ///                 [s_arg] = args.fixedunpack(1)
    ///             except ValueError:
    ///                 pass
    ///             else:
    ///                 from rpython.rtyper.llannotation import SomePtr
    ///                 assert not isinstance(s_arg, SomePtr)
    ///     else:
    ///         args = args.prepend(s_instance)
    ///         s_init.call(args)
    ///     return s_instance
    /// ```
    pub fn pycall(
        this: &Rc<RefCell<Self>>,
        _whence: Option<(
            crate::flowspace::model::GraphRef,
            crate::flowspace::model::BlockRef,
            usize,
        )>,
        args: &super::argument::ArgumentsForTranslation,
        _s_previous_result: &SomeValue,
        _op_key: Option<super::bookkeeper::PositionKey>,
    ) -> Result<SomeValue, AnnotatorError> {
        use super::model::{SomeInstance, SomeValue};
        // upstream: `classdef = self.getuniqueclassdef()`.
        let classdef = Self::getuniqueclassdef(this)?;
        // upstream: `s_instance = SomeInstance(classdef)`.
        let s_instance = SomeValue::Instance(SomeInstance::new(
            Some(classdef.clone()),
            false,
            std::collections::BTreeMap::new(),
        ));
        // upstream: `s_init = self.s_read_attribute('__init__')`.
        let s_init = Self::s_read_attribute(this, "__init__")?;
        let is_impossible = matches!(s_init, SomeValue::Impossible);
        if is_impossible {
            // upstream: `if not self.is_exception_class(): args.fixedunpack(0)`.
            let is_exc = this.borrow().is_exception_class();
            let name = this.borrow().name.clone();
            if !is_exc {
                args.fixedunpack(0).map_err(|_e| {
                    AnnotatorError::new(format!(
                        "default __init__ takes no argument (class {})",
                        name
                    ))
                })?;
            }
            // upstream: `elif self.pyobj is Exception: try: [s_arg] = args.fixedunpack(1)
            //                 except ValueError: pass else: assert not isinstance(s_arg, SomePtr)`.
            if this.borrow().pyobj.qualname() == "Exception"
                && let Ok(unpacked) = args.fixedunpack(1)
            {
                let [s_arg] = unpacked.as_slice() else {
                    unreachable!("fixedunpack(1) must return exactly one argument");
                };
                assert!(!matches!(
                    s_arg,
                    SomeValue::Ptr(_) | SomeValue::InteriorPtr(_)
                ));
            }
        } else {
            // upstream: `args = args.prepend(s_instance); s_init.call(args)`.
            let bound_args = args.prepend(s_instance.clone());
            s_init.call(&bound_args)?;
        }
        Ok(s_instance)
    }

    /// RPython `ClassDesc.lookup(self, name)` (classdesc.py:749-756).
    ///
    /// ```python
    /// def lookup(self, name):
    ///     cdesc = self
    ///     while name not in cdesc.classdict:
    ///         cdesc = cdesc.basedesc
    ///         if cdesc is None:
    ///             return None
    ///     return cdesc
    /// ```
    pub fn lookup(this: &Rc<RefCell<Self>>, name: &str) -> Option<Rc<RefCell<Self>>> {
        let mut cdesc: Rc<RefCell<Self>> = this.clone();
        loop {
            if cdesc.borrow().classdict.contains_key(name) {
                return Some(cdesc);
            }
            let next = cdesc.borrow().basedesc.clone();
            match next {
                Some(base) => cdesc = base,
                None => return None,
            }
        }
    }

    /// RPython `ClassDesc.get_param(self, name, default=None, inherit=True)`
    /// (classdesc.py:758-763).
    pub fn get_param(&self, name: &str, default: Option<ConstValue>, inherit: bool) -> ConstValue {
        if inherit {
            match host_getattr(&self.pyobj, name) {
                Ok(value) => value,
                Err(HostGetAttrError::Missing) | Err(HostGetAttrError::Unsupported) => {
                    default.unwrap_or(ConstValue::None)
                }
            }
        } else {
            self.pyobj
                .class_get(name)
                .unwrap_or_else(|| default.unwrap_or(ConstValue::None))
        }
    }

    /// RPython `ClassDesc.getallbases(self)` (classdesc.py:900-904).
    ///
    /// Walks `basedesc` chain from `self` outward and yields every
    /// intermediate ClassDesc. Upstream returns a generator; Rust
    /// collects into a Vec.
    pub fn getallbases(this: &Rc<RefCell<Self>>) -> Vec<Rc<RefCell<Self>>> {
        let mut out = vec![this.clone()];
        let mut cur = this.borrow().basedesc.clone();
        while let Some(d) = cur {
            let next = d.borrow().basedesc.clone();
            out.push(d);
            cur = next;
        }
        out
    }

    /// RPython `ClassDesc.getcommonbase(descs)` (classdesc.py:906-915).
    ///
    /// ```python
    /// commondesc = descs[0]
    /// for desc in descs[1:]:
    ///     allbases = set(commondesc.getallbases())
    ///     while desc not in allbases:
    ///         assert desc is not None, "no common base for %r" % (descs,)
    ///         desc = desc.basedesc
    ///     commondesc = desc
    /// return commondesc
    /// ```
    pub fn getcommonbase(descs: &[Rc<RefCell<Self>>]) -> Result<Rc<RefCell<Self>>, AnnotatorError> {
        if descs.is_empty() {
            return Err(AnnotatorError::new("getcommonbase: empty desc list"));
        }
        let mut commondesc = descs[0].clone();
        for desc in descs.iter().skip(1) {
            let allbases = Self::getallbases(&commondesc);
            let mut cur: Rc<RefCell<Self>> = desc.clone();
            loop {
                if allbases.iter().any(|b| Rc::ptr_eq(b, &cur)) {
                    break;
                }
                let next = cur.borrow().basedesc.clone();
                match next {
                    Some(n) => cur = n,
                    None => {
                        return Err(AnnotatorError::new(format!(
                            "no common base for {:?}",
                            descs
                                .iter()
                                .map(|d| d.borrow().name.clone())
                                .collect::<Vec<_>>()
                        )));
                    }
                }
            }
            commondesc = cur;
        }
        Ok(commondesc)
    }

    /// RPython `ClassDesc.read_attribute(self, name, default=NODEFAULT)`
    /// (classdesc.py:765-773).
    ///
    /// ```python
    /// def read_attribute(self, name, default=NODEFAULT):
    ///     cdesc = self.lookup(name)
    ///     if cdesc is None:
    ///         if default is NODEFAULT:
    ///             raise AttributeError
    ///         return default
    ///     return cdesc.classdict[name]
    /// ```
    ///
    /// Rust port returns `Option<ClassDictEntry>`; upstream's
    /// AttributeError/NODEFAULT/default triad collapses to `None` +
    /// caller-supplied default via `Option::or`.
    pub fn read_attribute(this: &Rc<RefCell<Self>>, name: &str) -> Option<ClassDictEntry> {
        let cdesc = Self::lookup(this, name)?;
        let entry = cdesc.borrow().classdict.get(name).cloned();
        entry
    }

    /// RPython `ClassDesc.s_read_attribute(self, name)` (classdesc.py:775-782).
    ///
    /// ```python
    /// def s_read_attribute(self, name):
    ///     cdesc = self.lookup(name)
    ///     if cdesc is None:
    ///         return s_ImpossibleValue
    ///     return cdesc.s_get_value(None, name)
    /// ```
    pub fn s_read_attribute(
        this: &Rc<RefCell<Self>>,
        name: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        match Self::lookup(this, name) {
            None => Ok(super::model::s_impossible_value()),
            Some(cdesc) => Self::s_get_value(&cdesc, None, name),
        }
    }

    /// RPython `ClassDesc.getattrfamily(self, attrname)`
    /// (classdesc.py:920-924).
    ///
    /// ```python
    /// def getattrfamily(self, attrname):
    ///     access_sets = self.bookkeeper.get_classpbc_attr_families(attrname)
    ///     _, _, attrfamily = access_sets.find(self)
    ///     return attrfamily
    /// ```
    pub fn getattrfamily(
        this: &Rc<RefCell<Self>>,
        attrname: &str,
    ) -> Option<Rc<RefCell<super::description::ClassAttrFamily>>> {
        use super::description::DescKey;
        let key = DescKey::from_rc(this);
        let bk = this.borrow().bookkeeper.upgrade()?;
        Some(bk.with_classpbc_attr_families(attrname, |uf| {
            let rep = uf.find_rep(key);
            uf.get(&rep)
                .cloned()
                .expect("UnionFind.find_rep() must materialise a ClassAttrFamily")
        }))
    }

    /// RPython `ClassDesc.queryattrfamily(self, attrname)`
    /// (classdesc.py:926-933).
    ///
    /// ```python
    /// def queryattrfamily(self, attrname):
    ///     access_sets = self.bookkeeper.get_classpbc_attr_families(attrname)
    ///     try:
    ///         return access_sets[self]
    ///     except KeyError:
    ///         return None
    /// ```
    pub fn queryattrfamily(
        this: &Rc<RefCell<Self>>,
        attrname: &str,
    ) -> Option<Rc<RefCell<super::description::ClassAttrFamily>>> {
        use super::description::DescKey;
        let key = DescKey::from_rc(this);
        let bk = this.borrow().bookkeeper.upgrade()?;
        bk.with_classpbc_attr_families(attrname, |uf| {
            if !uf.contains(&key) {
                return None;
            }
            uf.get(&key).cloned()
        })
    }

    /// RPython `ClassDesc.mergeattrfamilies(self, others, attrname)`
    /// (classdesc.py:935-942).
    ///
    /// ```python
    /// def mergeattrfamilies(self, others, attrname):
    ///     access_sets = self.bookkeeper.get_classpbc_attr_families(attrname)
    ///     changed, rep, attrfamily = access_sets.find(self)
    ///     for desc in others:
    ///         changed1, rep, attrfamily = access_sets.union(rep, desc)
    ///         changed = changed or changed1
    ///     return changed
    /// ```
    pub fn mergeattrfamilies(
        this: &Rc<RefCell<Self>>,
        others: &[Rc<RefCell<Self>>],
        attrname: &str,
    ) -> bool {
        use super::description::DescKey;
        let Some(bk) = this.borrow().bookkeeper.upgrade() else {
            return false;
        };
        let head_key = DescKey::from_rc(this);
        bk.with_classpbc_attr_families(attrname, |uf| {
            let mut rep = uf.find_rep(head_key);
            let mut changed = false;
            for desc in others {
                let other_key = DescKey::from_rc(desc);
                let (c, new_rep) = uf.union(rep, other_key);
                changed |= c;
                rep = new_rep;
            }
            changed
        })
    }

    /// RPython `ClassDesc.s_get_value(self, classdef, name)`
    /// (classdesc.py:784-802).
    ///
    /// ```python
    /// def s_get_value(self, classdef, name):
    ///     obj = self.classdict[name]
    ///     if isinstance(obj, Constant):
    ///         value = obj.value
    ///         if isinstance(value, staticmethod):
    ///             value = value.__get__(42)
    ///             classdef = None
    ///         elif isinstance(value, classmethod):
    ///             raise AnnotatorError("classmethods are not supported")
    ///         s_value = self.bookkeeper.immutablevalue(value)
    ///         if classdef is not None:
    ///             s_value = s_value.bind_callables_under(classdef, name)
    ///     elif isinstance(obj, Desc):
    ///         if classdef is not None:
    ///             obj = obj.bind_under(classdef, name)
    ///         s_value = SomePBC([obj])
    ///     else:
    ///         raise TypeError("classdict should not contain %r" % (obj,))
    ///     return s_value
    /// ```
    ///
    pub fn s_get_value(
        this: &Rc<RefCell<Self>>,
        classdef: Option<&Rc<RefCell<ClassDef>>>,
        name: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream: `obj = self.classdict[name]`.
        let entry = this.borrow().classdict.get(name).cloned().ok_or_else(|| {
            AnnotatorError::new(format!(
                "ClassDesc.s_get_value({}): no classdict entry {:?}",
                this.borrow().name,
                name
            ))
        })?;
        let bookkeeper = this.borrow().bookkeeper.upgrade().ok_or_else(|| {
            AnnotatorError::new("ClassDesc.s_get_value: Bookkeeper backlink dropped")
        })?;
        match entry {
            ClassDictEntry::Constant(c) => {
                let mut bind_classdef = classdef.cloned();
                let value = match &c.value {
                    ConstValue::HostObject(host) if host.is_staticmethod() => {
                        bind_classdef = None;
                        ConstValue::HostObject(
                            host.staticmethod_func()
                                .ok_or_else(|| {
                                    AnnotatorError::new(
                                        "ClassDesc.s_get_value: malformed staticmethod wrapper",
                                    )
                                })?
                                .clone(),
                        )
                    }
                    ConstValue::HostObject(host) if host.is_classmethod() => {
                        return Err(AnnotatorError::new("classmethods are not supported"));
                    }
                    other => other.clone(),
                };
                let s_value = bookkeeper.immutablevalue(&value)?;
                if let Some(cd) = bind_classdef.as_ref() {
                    Ok(super::model::bind_callables_under(&s_value, cd, name))
                } else {
                    Ok(s_value)
                }
            }
            ClassDictEntry::Desc(desc_entry) => {
                // upstream: `if classdef is not None: obj = obj.bind_under(classdef, name)`.
                let bound_entry = if let Some(cd) = classdef {
                    desc_entry.bind_under(cd, name)
                } else {
                    desc_entry
                };
                // upstream: `s_value = SomePBC([obj])`.
                Ok(SomeValue::PBC(super::model::SomePBC::new(
                    [bound_entry],
                    false,
                )))
            }
        }
    }

    /// RPython `ClassDesc.getclassdef(self, key)` (classdesc.py:669-670).
    pub fn getclassdef(
        this: &Rc<RefCell<Self>>,
        _key: (),
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        Self::getuniqueclassdef(this)
    }

    /// RPython `ClassDesc.find_source_for(self, name)` (classdesc.py:808-817).
    ///
    /// ```python
    /// def find_source_for(self, name):
    ///     if name in self.classdict:
    ///         return self
    ///     cls = self.pyobj
    ///     if name in cls.__dict__:
    ///         self.add_source_attribute(name, cls.__dict__[name])
    ///         if name in self.classdict:
    ///             return self
    ///     return None
    /// ```
    ///
    /// Returns an [`AttrSource::Class`] pointing at `this` when the
    /// attribute lives in (or was just imported into) `classdict`.
    pub fn find_source_for(
        this: &Rc<RefCell<Self>>,
        name: &str,
    ) -> Result<Option<AttrSource>, AnnotatorError> {
        // upstream: `if name in self.classdict: return self`
        if this.borrow().classdict.contains_key(name) {
            return Ok(Some(AttrSource::Class(Rc::downgrade(this))));
        }
        // upstream: `cls = self.pyobj; if name in cls.__dict__: ...`
        let pyobj = this.borrow().pyobj.clone();
        if let Some(value) = pyobj.class_get(name) {
            // upstream: `self.add_source_attribute(name, cls.__dict__[name])`.
            Self::add_source_attribute(this, name, value, false)?;
            if this.borrow().classdict.contains_key(name) {
                return Ok(Some(AttrSource::Class(Rc::downgrade(this))));
            }
        }
        Ok(None)
    }

    /// RPython `ClassDesc.consider_call_site(descs, args, s_result, op)`
    /// RPython `ClassDesc.maybe_return_immutable_list(self, attr, s_result)`
    /// (classdesc.py:819-850).
    ///
    /// ```python
    /// def maybe_return_immutable_list(self, attr, s_result):
    ///     if self._detect_invalid_attrs and attr in self._detect_invalid_attrs:
    ///         raise AnnotatorError("field %r was migrated to %r from a subclass...")
    ///     search1 = '%s[*]' % (attr,)
    ///     search2 = '%s?[*]' % (attr,)
    ///     cdesc = self
    ///     while cdesc is not None:
    ///         immutable_fields = cdesc.immutable_fields
    ///         if immutable_fields:
    ///             if (search1 in immutable_fields or search2 in immutable_fields):
    ///                 s_result.listdef.never_resize()
    ///                 s_copy = s_result.listdef.offspring(self.bookkeeper)
    ///                 s_copy.listdef.mark_as_immutable()
    ///                 cdesc = cdesc.basedesc
    ///                 while cdesc is not None:
    ///                     if cdesc._detect_invalid_attrs is None:
    ///                         cdesc._detect_invalid_attrs = set()
    ///                     cdesc._detect_invalid_attrs.add(attr)
    ///                     cdesc = cdesc.basedesc
    ///                 return s_copy
    ///         cdesc = cdesc.basedesc
    ///     return s_result
    /// ```
    ///
    /// "hack: `x.lst` where `lst` is listed in `_immutable_fields_` as
    /// either `lst[*]` or `lst?[*]` should really return an immutable
    /// list as a result."
    pub fn maybe_return_immutable_list(
        this: &Rc<RefCell<Self>>,
        attr: &str,
        s_result: &SomeValue,
    ) -> Result<SomeValue, AnnotatorError> {
        // upstream: `if self._detect_invalid_attrs and attr in ...: raise`.
        {
            let borrowed = this.borrow();
            if let Some(detect) = &borrowed.detect_invalid_attrs {
                if detect.contains(attr) {
                    return Err(AnnotatorError::new(format!(
                        "field {:?} was migrated to {:?} from a subclass in \
                         which it was declared as _immutable_fields_",
                        attr, borrowed.pyobj
                    )));
                }
            }
        }

        // upstream: `search1 = '%s[*]' % (attr,); search2 = '%s?[*]' % (attr,)`.
        let search1 = format!("{attr}[*]");
        let search2 = format!("{attr}?[*]");

        // upstream: `cdesc = self; while cdesc is not None: ...`.
        let SomeValue::List(s_list) = s_result else {
            // maybe_return_immutable_list is only called by
            // ClassDef::s_getattr on a SomeList. If we somehow land
            // here with a non-list, keep upstream's pass-through.
            return Ok(s_result.clone());
        };

        let mut cdesc: Option<Rc<RefCell<Self>>> = Some(this.clone());
        while let Some(cd) = cdesc {
            let is_match = {
                let b = cd.borrow();
                !b.immutable_fields.is_empty()
                    && (b.immutable_fields.contains(&search1)
                        || b.immutable_fields.contains(&search2))
            };
            if is_match {
                // upstream: `s_result.listdef.never_resize();
                //            s_copy = s_result.listdef.offspring(self.bookkeeper);
                //            s_copy.listdef.mark_as_immutable()`.
                s_list.listdef.never_resize().map_err(|e| {
                    AnnotatorError::new(format!("maybe_return_immutable_list({attr}): {}", e.0))
                })?;
                let bookkeeper = this.borrow().bookkeeper.upgrade().ok_or_else(|| {
                    AnnotatorError::new(
                        "ClassDesc.maybe_return_immutable_list: bookkeeper backlink dropped",
                    )
                })?;
                let s_copy = s_list.listdef.offspring(&bookkeeper, &[])?;
                s_copy.listdef.mark_as_immutable().map_err(|e| {
                    AnnotatorError::new(format!("maybe_return_immutable_list({attr}): {}", e.0))
                })?;

                // upstream inner loop: walk basedesc chain marking
                // `_detect_invalid_attrs`.
                let mut inner = cd.borrow().basedesc.clone();
                while let Some(base_cd) = inner {
                    {
                        let mut b = base_cd.borrow_mut();
                        b.detect_invalid_attrs
                            .get_or_insert_with(HashSet::new)
                            .insert(attr.to_string());
                    }
                    inner = base_cd.borrow().basedesc.clone();
                }
                return Ok(SomeValue::List(s_copy));
            }
            cdesc = cd.borrow().basedesc.clone();
        }
        // upstream: `return s_result`.
        Ok(s_result.clone())
    }

    /// (classdesc.py:853-902).
    ///
    /// Phase 1: `descs[0].getcallfamily(); descs[0].mergecallfamilies(*descs[1:])`
    /// — keeps the PBC call-family UnionFind consistent.
    ///
    /// Phase 2: compute the `__init__` MethodDesc set via
    /// `desc.s_read_attribute('__init__')` and recurse into
    /// `MethodDesc.consider_call_site(initdescs, args, s_None, op)`.
    pub fn consider_call_site(
        descs: &[Rc<RefCell<ClassDesc>>],
        args: &super::argument::ArgumentsForTranslation,
        s_result: &super::model::SomeValue,
        op_key: Option<super::bookkeeper::PositionKey>,
    ) -> Result<(), super::model::AnnotatorError> {
        use super::description::DescKey;
        use super::model::SomeValue;
        if descs.is_empty() {
            return Ok(());
        }
        // ClassDesc doesn't embed the Rust `Desc` struct (the Python
        // `class ClassDesc(Desc)` inheritance collapses into a bare
        // struct here), so `getcallfamily` / `mergecallfamilies` are
        // re-derived against the bookkeeper's pbc_maximal_call_families
        // UnionFind keyed by `DescKey::from_rc(&classdesc)` — the same
        // pointer identity upstream `id(desc)` provides.
        let Some(bk) = descs[0].borrow().bookkeeper.upgrade() else {
            return Ok(());
        };
        let head_key = DescKey::from_rc(&descs[0]);
        // upstream: `descs[0].getcallfamily()` — force the family slot
        // to exist so merge has a representative.
        {
            let mut families = bk.pbc_maximal_call_families.borrow_mut();
            let _ = families.find_rep(head_key);
        }
        // upstream: `descs[0].mergecallfamilies(*descs[1:])`.
        {
            let mut families = bk.pbc_maximal_call_families.borrow_mut();
            let mut rep = families.find_rep(head_key);
            for other in descs.iter().skip(1) {
                let other_key = DescKey::from_rc(other);
                let (_changed, new_rep) = families.union(rep, other_key);
                rep = new_rep;
            }
            let _ = rep;
        }

        // Phase 2 (classdesc.py:856-902).
        //
        // upstream:
        //   if len(descs) == 1:
        //       if not isinstance(s_result, SomeInstance):
        //           raise AnnotatorError("calling a class didn't return an instance??")
        //       classdefs = [s_result.classdef]
        //   else:
        //       classdefs = [desc.getuniqueclassdef() for desc in descs]
        //       has_init = any(isinstance(desc.s_read_attribute('__init__'), SomePBC) for desc in descs)
        //       basedesc = ClassDesc.getcommonbase(descs)
        //       s_init = basedesc.s_read_attribute('__init__')
        //       parent_has_init = isinstance(s_init, SomePBC)
        //       if has_init and not parent_has_init:
        //           raise AnnotatorError(...)
        let classdefs: Vec<Rc<RefCell<ClassDef>>> = if descs.len() == 1 {
            // upstream: single-class call; `s_result` is the instance
            // annotation carrying the (possibly specialised) classdef.
            match s_result {
                SomeValue::Instance(si) => match &si.classdef {
                    Some(cd) => vec![cd.clone()],
                    None => {
                        return Err(AnnotatorError::new(
                            "calling a class didn't return an instance??",
                        ));
                    }
                },
                _ => {
                    return Err(AnnotatorError::new(
                        "calling a class didn't return an instance??",
                    ));
                }
            }
        } else {
            let mut cds = Vec::with_capacity(descs.len());
            for d in descs {
                cds.push(Self::getuniqueclassdef(d)?);
            }
            // has_init check across descs.
            let mut has_init = false;
            for d in descs {
                let s_init = Self::s_read_attribute(d, "__init__")?;
                if matches!(s_init, SomeValue::PBC(_)) {
                    has_init = true;
                }
            }
            let basedesc = Self::getcommonbase(descs)?;
            let s_parent_init = Self::s_read_attribute(&basedesc, "__init__")?;
            let parent_has_init = matches!(s_parent_init, SomeValue::PBC(_));
            if has_init && !parent_has_init {
                return Err(AnnotatorError::new(format!(
                    "some subclasses among {:?} declare __init__(), but not the common parent class",
                    descs
                        .iter()
                        .map(|d| d.borrow().name.clone())
                        .collect::<Vec<_>>()
                )));
            }
            cds
        };

        // upstream:
        //   initdescs = []
        //   for desc, classdef in zip(descs, classdefs):
        //       s_init = desc.s_read_attribute('__init__')
        //       if isinstance(s_init, SomePBC):
        //           assert len(s_init.descriptions) == 1
        //           initfuncdesc, = s_init.descriptions
        //           if isinstance(initfuncdesc, FunctionDesc):
        //               initmethdesc = getbookkeeper().getmethoddesc(
        //                   initfuncdesc, classdef, classdef, '__init__')
        //               initdescs.append(initmethdesc)
        let mut initdescs: Vec<Rc<RefCell<super::description::MethodDesc>>> = Vec::new();
        for (desc, classdef) in descs.iter().zip(classdefs.iter()) {
            let s_init = Self::s_read_attribute(desc, "__init__")?;
            let SomeValue::PBC(pbc) = s_init else {
                continue;
            };
            if pbc.descriptions.len() != 1 {
                return Err(AnnotatorError::new(
                    "unexpected dynamic __init__?".to_string(),
                ));
            }
            let Some(entry) = pbc.descriptions.values().next() else {
                continue;
            };
            let super::description::DescEntry::Function(initfuncdesc) = entry else {
                continue;
            };
            let classdef_key = super::description::ClassDefKey::from_classdef(classdef);
            let initmethdesc = bk.getmethoddesc(
                initfuncdesc,
                classdef_key,
                Some(classdef_key),
                "__init__",
                std::collections::BTreeMap::new(),
            );
            initdescs.push(initmethdesc);
        }

        // upstream:
        //   if initdescs:
        //       initdescs[0].mergecallfamilies(*initdescs[1:])
        //       MethodDesc.consider_call_site(initdescs, args, s_None, op)
        if !initdescs.is_empty() {
            let head_entry = super::description::DescEntry::Method(initdescs[0].clone());
            let borrowed: Vec<_> = initdescs.iter().skip(1).map(|d| d.borrow()).collect();
            let others: Vec<&super::description::Desc> = borrowed.iter().map(|d| &d.base).collect();
            // Route the mergecallfamilies through the head MethodDesc's
            // Desc base — MethodDesc::consider_call_site will redo this
            // internally, but upstream calls both explicitly.
            let _ = head_entry;
            initdescs[0].borrow().base.mergecallfamilies(&others)?;
            let s_none = SomeValue::None_(super::model::SomeNone::new());
            super::description::MethodDesc::consider_call_site(&initdescs, args, &s_none, op_key)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Private helpers.
    // -----------------------------------------------------------------------

    /// Upstream classdesc.py:528-535 — parse `_immutable_fields_` as a
    /// sequence of names, rejecting str inputs.
    fn extract_immutable_fields(cls: &HostObject) -> Result<HashSet<String>, AnnotatorError> {
        let v = match cls.class_get("_immutable_fields_") {
            Some(v) => v,
            None => return Ok(HashSet::new()),
        };
        if let ConstValue::Str(_) = &v {
            return Err(AnnotatorError::new(format!(
                "In class {:?}, '_immutable_fields_' must be a sequence of \
                 attribute names, not a string.",
                cls.qualname()
            )));
        }
        let names = Self::extract_name_sequence("_immutable_fields_", &v)?;
        Ok(names.into_iter().collect())
    }

    /// Upstream's `__slots__` / `_attrs_` / `_immutable_fields_` share a
    /// sequence-of-strings shape. Accepts both tuples and lists. For
    /// `__slots__` upstream auto-wraps a str into `(str,)` — mirrored
    /// here. `_immutable_fields_` rejects str ahead of calling this.
    fn extract_name_sequence(
        decl_name: &str,
        v: &ConstValue,
    ) -> Result<Vec<String>, AnnotatorError> {
        let items: &[ConstValue] = match v {
            ConstValue::Tuple(items) | ConstValue::List(items) => items,
            ConstValue::Str(s) => return Ok(vec![s.clone()]),
            _ => {
                return Err(AnnotatorError::new(format!(
                    "{:?} must be a sequence of names",
                    decl_name
                )));
            }
        };
        items
            .iter()
            .map(|item| match item {
                ConstValue::Str(s) => Ok(s.clone()),
                _ => Err(AnnotatorError::new(format!(
                    "{:?} must be a sequence of strings",
                    decl_name
                ))),
            })
            .collect()
    }

    /// Walks `cls.__mro__` and collects every class-dict key. Used by
    /// [`Self::add_mixins`] to build the skip set that prevents mixin
    /// attributes from shadowing the real base class.
    fn collect_mro_dict_keys(cls: &HostObject, out: &mut HashSet<String>) {
        let mro = cls.mro().unwrap_or_else(|| vec![cls.clone()]);
        for c in mro {
            for name in c.class_dict_keys() {
                out.insert(name);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ClassDef (classdesc.py:136-431).
// ---------------------------------------------------------------------------

/// RPython `class ClassDef(object)` — "Wraps a user class."
/// (classdesc.py:136-431).
///
/// Data + structural methods are ported here; `s_getattr`,
/// `lookup_filter`, `check_missing_attribute_update`, `check_attr_here`,
/// and `see_instance` land with c3 because they depend on the
/// bookkeeper's update_attr / MethodDesc.bind_self / live-reflection
/// hooks.
#[derive(Debug)]
pub struct ClassDef {
    /// RPython `self.bookkeeper`. Weak to avoid cycles with the bookkeeper's
    /// `classdefs` Vec.
    pub bookkeeper: Weak<Bookkeeper>,
    /// RPython `self.attrs = {}` — `{name: Attribute}`.
    pub attrs: HashMap<String, Attribute>,
    /// RPython `self.classdesc = classdesc`. Strong reference —
    /// ClassDef owns its ClassDesc; the cycle is broken on the
    /// ClassDesc side where `classdef` is `Weak` (see module doc).
    pub classdesc: Rc<RefCell<ClassDesc>>,
    /// RPython `self.name = self.classdesc.name`.
    pub name: String,
    /// RPython `self.shortname = self.name.split('.')[-1]`.
    pub shortname: String,
    /// RPython `self.subdefs = []` — Weak to avoid subclass retain cycles.
    pub subdefs: Vec<Weak<RefCell<ClassDef>>>,
    /// RPython `self.attr_sources = {}` — `{name: list-of-sources}`.
    pub attr_sources: HashMap<String, Vec<AttrSource>>,
    /// RPython `self.read_locations_of__class__ = {}`.
    pub read_locations_of_class: HashMap<PositionKey, bool>,
    /// RPython `self.repr = None`. Populated by `rclass.getclassrepr()`
    /// with the cached `ClassRepr` for this classdef.
    pub repr: Option<Arc<ClassRepr>>,
    /// RPython `self.extra_access_sets = {}`.
    ///
    /// Keyed by access-set identity; values carry the original family
    /// object alongside the `(attrname, counter)` payload stored
    /// upstream.
    pub extra_access_sets: HashMap<usize, (Rc<RefCell<ClassAttrFamily>>, String, usize)>,
    /// RPython `self.instances_seen = set()`. Identity hashes of the
    /// prebuilt instances already absorbed by `see_instance`.
    pub instances_seen: HashSet<usize>,
    /// RPython `self.basedef` — strong reference; the superclass owns
    /// the weak `subdefs` back-ref.
    pub basedef: Option<Rc<RefCell<ClassDef>>>,
    /// RPython `self.parentdefs = dict.fromkeys(self.getmro())`. Stored
    /// as identity pointers so `.contains` works with `Rc::as_ptr`.
    pub parentdefs: HashSet<usize>,
    /// Stable classdef identity used by
    /// `normalizecalls.get_unique_cdef_id()` (normalizecalls.py:393-399)
    /// to build reversed-MRO inheritance-order witnesses.
    pub unique_cdef_id: Option<usize>,
    /// RPython `classdef.minid = TotalOrderSymbolic(witness, lst)` set by
    /// `rpython.rtyper.normalizecalls.assign_inheritance_ids`
    /// (normalizecalls.py:385-389). Bracketing ID for this classdef's
    /// subtree: `self.minid < desc.minid < self.maxid` for every proper
    /// descendant `desc`. Upstream is a deferred `TotalOrderSymbolic`;
    /// the Rust port stores the computed integer position of the
    /// classdef's start marker in the current reversed-MRO witness
    /// ordering. `None` until `assign_inheritance_ids` runs.
    pub minid: Option<i64>,
    /// RPython `classdef.maxid = TotalOrderSymbolic(witness + [MAX], lst)`
    /// (normalizecalls.py:389). Upper bracket of this classdef's subtree
    /// — strictly greater than every descendant's `minid`. `None` until
    /// `assign_inheritance_ids` runs.
    pub maxid: Option<i64>,
    /// RPython `classdef.my_instantiate_graph = graph`
    /// (normalizecalls.py:294). Set by
    /// [`crate::translator::rtyper::normalizecalls::create_instantiate_function`]
    /// for every classdef whose constructor participates in a
    /// polymorphic `ClassesPBCRepr.call()` — consumed by
    /// `ClassRepr.fill_vtable_root` (rclass.py:356-358) to emit the
    /// `vtable.instantiate` slot. Upstream tests this with
    /// `hasattr(classdef, 'my_instantiate_graph')`; the Rust port uses
    /// `Option::is_some` instead.
    pub my_instantiate_graph: Option<crate::flowspace::model::GraphRef>,
}

impl ClassDef {
    /// RPython `ClassDef.__init__(self, bookkeeper, classdesc)`
    /// (classdesc.py:139-159).
    ///
    /// Builds an `Rc<RefCell<Self>>` directly because the constructor
    /// needs to register `self` in the base class's `subdefs`, which
    /// requires an `Rc` identity.
    pub fn new(
        bookkeeper: &Rc<Bookkeeper>,
        classdesc: &Rc<RefCell<ClassDesc>>,
    ) -> Rc<RefCell<ClassDef>> {
        let (name, basedesc_opt) = {
            let cd = classdesc.borrow();
            (cd.name.clone(), cd.basedesc.clone())
        };
        let shortname = name.rsplit('.').next().unwrap_or(name.as_str()).to_string();

        // upstream: if classdesc.basedesc: basedef = basedesc.getuniqueclassdef()
        let basedef = basedesc_opt.as_ref().map(|base| {
            ClassDesc::getuniqueclassdef(base).unwrap_or_else(|err| {
                panic!(
                    "ClassDef::new: basedesc.getuniqueclassdef() failed for {:?}: {}",
                    base.borrow().name,
                    err
                )
            })
        });

        let me = Rc::new(RefCell::new(ClassDef {
            bookkeeper: Rc::downgrade(bookkeeper),
            attrs: HashMap::new(),
            classdesc: classdesc.clone(),
            name,
            shortname,
            subdefs: Vec::new(),
            attr_sources: HashMap::new(),
            read_locations_of_class: HashMap::new(),
            repr: None,
            extra_access_sets: HashMap::new(),
            instances_seen: HashSet::new(),
            basedef: basedef.clone(),
            parentdefs: HashSet::new(),
            unique_cdef_id: None,
            minid: None,
            maxid: None,
            my_instantiate_graph: None,
        }));

        if let Some(base) = basedef.as_ref() {
            base.borrow_mut().subdefs.push(Rc::downgrade(&me));
            // upstream: self.basedef.see_new_subclass(self)
            // see_new_subclass writes into read_locations_of__class__ and
            // reflows — no work to do when that dict is empty on init.
            Self::see_new_subclass_recursive(base, &me);
        }

        // upstream: self.parentdefs = dict.fromkeys(self.getmro()).
        let mro = Self::getmro(&me);
        let mut parents = HashSet::new();
        for cd in mro {
            parents.insert(Rc::as_ptr(&cd) as usize);
        }
        me.borrow_mut().parentdefs = parents;

        // upstream `ClassDesc._init_classdef` (classdesc.py:672-697) writes
        // `self.classdef = classdef`. Stored as Weak in Rust so ClassDef
        // strongly owns ClassDesc without cycling through this back-ref.
        classdesc.borrow_mut().classdef = Some(Rc::downgrade(&me));

        me
    }

    /// Test-only helper — constructs a standalone [`ClassDef`] with a
    /// fresh bookkeeper + ClassDesc shell under the given name, with
    /// optional base class. Mirrors upstream's pattern of creating
    /// classdefs through `bookkeeper.getdesc(cls).getuniqueclassdef()`
    /// without requiring the full Python runtime.
    pub fn new_standalone(
        name: impl Into<String>,
        base: Option<&Rc<RefCell<ClassDef>>>,
    ) -> Rc<RefCell<ClassDef>> {
        let name: String = name.into();
        let bk = Rc::new(Bookkeeper::new());
        let base_desc = base.map(|b| b.borrow().classdesc.clone());
        let base_host_list = base_desc
            .as_ref()
            .map(|cd| vec![cd.borrow().pyobj.clone()])
            .unwrap_or_default();
        let pyobj = HostObject::new_class(&name, base_host_list);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(&bk, pyobj, name)));
        desc.borrow_mut().basedesc = base_desc;
        ClassDef::new(&bk, &desc)
    }

    // -----------------------------------------------------------------------
    // Read-only structural methods.
    // -----------------------------------------------------------------------

    /// RPython `ClassDef.__repr__` (classdesc.py:242-243).
    pub fn repr_str(&self) -> String {
        format!("<ClassDef '{}'>", self.name)
    }

    /// RPython `ClassDef.getmro(self)` (classdesc.py:256-259).
    ///
    /// Walks `basedef` chain bottom-up. Returns owning `Rc`s so
    /// downstream callers can identity-compare.
    pub fn getmro(start: &Rc<RefCell<ClassDef>>) -> Vec<Rc<RefCell<ClassDef>>> {
        let mut out = Vec::new();
        let mut cur = Some(start.clone());
        while let Some(cd) = cur {
            let next = cd.borrow().basedef.clone();
            out.push(cd);
            cur = next;
        }
        out
    }

    /// RPython `ClassDef.issubclass(self, other)` (classdesc.py:261-262).
    ///
    /// Delegates to `classdesc.issubclass` (strong Rc — always live).
    pub fn issubclass(&self, other: &Rc<RefCell<ClassDef>>) -> bool {
        let self_desc = self.classdesc.clone();
        let other_desc = other.borrow().classdesc.clone();
        self_desc.borrow().issubclass(&other_desc.borrow())
    }

    /// RPython `ClassDef.getallsubdefs(self)` (classdesc.py:264-272).
    pub fn getallsubdefs(start: &Rc<RefCell<ClassDef>>) -> Vec<Rc<RefCell<ClassDef>>> {
        let mut out = Vec::new();
        let mut pending = vec![start.clone()];
        let mut seen: HashSet<usize> = HashSet::new();
        seen.insert(Rc::as_ptr(start) as usize);
        while let Some(cd) = pending.pop() {
            out.push(cd.clone());
            for sub in cd.borrow().subdefs.iter() {
                if let Some(sub_rc) = sub.upgrade() {
                    let key = Rc::as_ptr(&sub_rc) as usize;
                    if !seen.contains(&key) {
                        seen.insert(key);
                        pending.push(sub_rc);
                    }
                }
            }
        }
        out
    }

    /// RPython `ClassDef.commonbase(self, other)` (classdesc.py:251-254).
    ///
    /// Free-function form so callers don't need to split `self` from
    /// the `Rc`.
    pub fn commonbase(
        a: &Rc<RefCell<ClassDef>>,
        b: &Rc<RefCell<ClassDef>>,
    ) -> Option<Rc<RefCell<ClassDef>>> {
        let mut cur = Some(b.clone());
        while let Some(cd) = cur {
            if a.borrow().issubclass(&cd) {
                return Some(cd);
            }
            let next = cd.borrow().basedef.clone();
            cur = next;
        }
        None
    }

    /// RPython `ClassDef.has_no_attrs(self)` (classdesc.py:245-249).
    pub fn has_no_attrs(start: &Rc<RefCell<ClassDef>>) -> bool {
        for cd in Self::getmro(start) {
            if !cd.borrow().attrs.is_empty() {
                return false;
            }
        }
        true
    }

    /// RPython `ClassDef.get_owner(self, attrname)` (classdesc.py:222-228).
    pub fn get_owner(
        start: &Rc<RefCell<ClassDef>>,
        attrname: &str,
    ) -> Option<Rc<RefCell<ClassDef>>> {
        for cd in Self::getmro(start) {
            if cd.borrow().attrs.contains_key(attrname) {
                return Some(cd);
            }
        }
        None
    }

    /// RPython `ClassDef.about_attribute(self, name)` (classdesc.py:324-334).
    pub fn about_attribute(start: &Rc<RefCell<ClassDef>>, name: &str) -> Option<SomeValue> {
        for cd in Self::getmro(start) {
            let cd_ref = cd.borrow();
            if let Some(attr) = cd_ref.attrs.get(name) {
                if !matches!(attr.s_value, SomeValue::Impossible) {
                    return Some(attr.s_value.clone());
                }
                return None;
            }
        }
        None
    }

    /// RPython `ClassDef._freeze_(self)` (classdesc.py:429-431).
    ///
    /// Upstream raises to prevent `immutablevalue(classdef)` from
    /// silently storing a ClassDef as a constant. The Rust port
    /// returns an [`AnnotatorError`] with the same message.
    pub fn _freeze_() -> Result<(), AnnotatorError> {
        Err(AnnotatorError::new(
            "ClassDefs are used as knowntype for instances but cannot be \
             used as immutablevalue arguments directly",
        ))
    }

    // -----------------------------------------------------------------------
    // Mutable structural methods.
    // -----------------------------------------------------------------------

    /// RPython `ClassDef.setup(self, sources)` (classdesc.py:161-166).
    pub fn setup(
        this: &Rc<RefCell<ClassDef>>,
        sources: HashMap<String, AttrSource>,
    ) -> Result<(), AnnotatorError> {
        for (name, source) in sources {
            Self::add_source_for_attribute(this, &name, source)?;
        }
        // upstream: bookkeeper.event('classdef_setup', self)
        // bookkeeper.event hook lands with c3; no-op here matches the
        // default empty event body (classdesc.py:165-166).
        Ok(())
    }

    /// RPython `ClassDef.add_source_for_attribute(self, attr, source)`
    /// (classdesc.py:189-220).
    ///
    /// The `bookkeeper.update_attr(cdef, attrdef)` call site drives
    /// [`Bookkeeper::update_attr`] for reflow + validation.
    pub fn add_source_for_attribute(
        this: &Rc<RefCell<ClassDef>>,
        attr: &str,
        source: AttrSource,
    ) -> Result<(), AnnotatorError> {
        // upstream: for cdef in self.getmro(): if attr in cdef.attrs: ...
        for cdef in Self::getmro(this) {
            let has_attr = cdef.borrow().attrs.contains_key(attr);
            if has_attr {
                let s_prev = {
                    let cdef_ref = cdef.borrow();
                    cdef_ref.attrs.get(attr).unwrap().s_value.clone()
                };
                {
                    let mut cdef_mut = cdef.borrow_mut();
                    let attrdef = cdef_mut.attrs.get_mut(attr).unwrap();
                    attrdef.add_constant_source(this, &source)?;
                }
                let s_new = {
                    let cdef_ref = cdef.borrow();
                    cdef_ref.attrs.get(attr).unwrap().s_value.clone()
                };
                // upstream: if attrdef.s_value != s_prev_value:
                //     self.bookkeeper.update_attr(cdef, attrdef)
                if s_new != s_prev {
                    if let Some(bk) = cdef.borrow().bookkeeper.upgrade() {
                        bk.update_attr(&cdef, attr)?;
                    }
                }
                return Ok(());
            }
        }

        // No existing Attribute: remember in attr_sources.
        this.borrow_mut()
            .attr_sources
            .entry(attr.to_string())
            .or_default()
            .push(source.clone());

        // upstream: if not source.instance_level: for subdef in
        // self.getallsubdefs(): if attr in subdef.attrs: ...
        if !source.instance_level() {
            for subdef in Self::getallsubdefs(this) {
                let has_attr = subdef.borrow().attrs.contains_key(attr);
                if has_attr {
                    let s_prev = {
                        let s_ref = subdef.borrow();
                        s_ref.attrs.get(attr).unwrap().s_value.clone()
                    };
                    {
                        let mut subdef_mut = subdef.borrow_mut();
                        let attrdef = subdef_mut.attrs.get_mut(attr).unwrap();
                        attrdef.add_constant_source(this, &source)?;
                    }
                    let s_new = {
                        let s_ref = subdef.borrow();
                        s_ref.attrs.get(attr).unwrap().s_value.clone()
                    };
                    if s_new != s_prev {
                        if let Some(bk) = subdef.borrow().bookkeeper.upgrade() {
                            bk.update_attr(&subdef, attr)?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// RPython `ClassDef.locate_attribute(self, attr)` (classdesc.py:231-237).
    pub fn locate_attribute(
        this: &Rc<RefCell<ClassDef>>,
        attr: &str,
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        if let Some(cdef) = Self::get_owner(this, attr) {
            return Ok(cdef);
        }
        Self::generalize_attr_internal(this, attr, None)?;
        Ok(this.clone())
    }

    /// RPython `ClassDef.find_attribute(self, attr)` (classdesc.py:239-240).
    pub fn find_attribute(
        this: &Rc<RefCell<ClassDef>>,
        attr: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        let owner = Self::locate_attribute(this, attr)?;
        let attr_value = owner
            .borrow()
            .attrs
            .get(attr)
            .map(|a| a.s_value.clone())
            .ok_or_else(|| {
                AnnotatorError::new(format!(
                    "ClassDef.find_attribute {:?} missing after locate_attribute",
                    attr
                ))
            })?;
        Ok(attr_value)
    }

    /// RPython `ClassDef.lookup_filter(self, pbc, name=None, flags={})`
    /// (classdesc.py:336-374).
    ///
    /// "Selects the methods in the pbc that could possibly be seen by
    /// a lookup performed on an instance of 'self', removing the ones
    /// that cannot appear."
    pub fn lookup_filter(
        this: &Rc<RefCell<ClassDef>>,
        pbc: &super::model::SomePBC,
        _name: Option<&str>,
        flags: &std::collections::BTreeMap<String, bool>,
    ) -> Result<SomeValue, AnnotatorError> {
        let mut d: Vec<super::description::DescEntry> = Vec::new();
        let mut uplookup: Option<Rc<RefCell<ClassDef>>> = None;
        let mut updesc: Option<Rc<RefCell<super::description::MethodDesc>>> = None;

        let bookkeeper = this.borrow().bookkeeper.upgrade().ok_or_else(|| {
            AnnotatorError::new("ClassDef.lookup_filter: bookkeeper backlink dropped")
        })?;

        for entry in pbc.descriptions.values() {
            // upstream: `if isinstance(desc, MethodDesc) and desc.selfclassdef is None:`
            let method = match entry {
                super::description::DescEntry::Method(md) => {
                    if md.borrow().selfclassdef.is_none() {
                        Some(md.clone())
                    } else {
                        None
                    }
                }
                _ => None,
            };

            let Some(desc) = method else {
                // upstream: `d.append(desc)` for non-method or
                // already-bound methods.
                d.push(entry.clone());
                continue;
            };

            // upstream: `methclassdef = desc.originclassdef`.
            let methclassdef_key = desc.borrow().originclassdef;
            let Some(methclassdef) = bookkeeper.lookup_classdef(methclassdef_key) else {
                // Missing classdef — skip this desc (upstream never hits
                // this because the UnionFind keeps classdefs alive).
                continue;
            };

            let is_same = Rc::ptr_eq(&methclassdef, this);
            let is_subclass_of_self = !is_same && methclassdef.borrow().issubclass(this);
            let self_subclass_of_meth = this.borrow().issubclass(&methclassdef);

            if is_subclass_of_self {
                // upstream: `pass  # subclasses methods are always candidates`.
            } else if self_subclass_of_meth {
                // upstream: `if uplookup is None or methclassdef.issubclass(uplookup):`
                let replace = match &uplookup {
                    None => true,
                    Some(up) => methclassdef.borrow().issubclass(up),
                };
                if replace {
                    uplookup = Some(methclassdef.clone());
                    updesc = Some(desc.clone());
                }
                continue;
            } else {
                // upstream: `continue  # not matching`.
                continue;
            }

            // upstream: `desc = desc.bind_self(methclassdef, flags)`.
            let bound = desc.borrow().bind_self(methclassdef_key, flags.clone())?;
            d.push(super::description::DescEntry::Method(bound));
        }

        // upstream: `if uplookup is not None: d.append(updesc.bind_self(self, flags))`.
        if let (Some(up), Some(ud)) = (uplookup.as_ref(), updesc.as_ref()) {
            let up_key = super::description::ClassDefKey::from_classdef(up);
            // NB. upstream uses `self` — bind_self to the caller's own
            // classdef, not `up`. Matches classdesc.py:367.
            let self_key = super::description::ClassDefKey::from_classdef(this);
            let _ = up_key;
            let bound = ud.borrow().bind_self(self_key, flags.clone())?;
            d.push(super::description::DescEntry::Method(bound));
        }

        // upstream: `if d: return SomePBC(d, can_be_None=pbc.can_be_None)`.
        if !d.is_empty() {
            let new_pbc = super::model::SomePBC::with_subset(d, pbc.can_be_none, None);
            return Ok(SomeValue::PBC(new_pbc));
        }
        // upstream: `elif pbc.can_be_None: return s_None`.
        if pbc.can_be_none {
            return Ok(super::model::s_none());
        }
        // upstream: `else: return s_ImpossibleValue`.
        Ok(super::model::s_impossible_value())
    }

    /// RPython `ClassDef.s_getattr(self, attrname, flags)` (classdesc.py:168-187).
    ///
    /// ```python
    /// def s_getattr(self, attrname, flags):
    ///     attrdef = self.find_attribute(attrname)
    ///     s_result = attrdef.s_value
    ///     if isinstance(s_result, SomePBC):
    ///         s_result = self.lookup_filter(s_result, attrname, flags)
    ///     elif isinstance(s_result, SomeImpossibleValue):
    ///         self.check_missing_attribute_update(attrname)
    ///         for basedef in self.getmro():
    ///             if basedef.classdesc.all_enforced_attrs is not None:
    ///                 if attrname in basedef.classdesc.all_enforced_attrs:
    ///                     raise HarmlesslyBlocked("get enforced attr")
    ///     elif isinstance(s_result, SomeList):
    ///         s_result = self.classdesc.maybe_return_immutable_list(
    ///             attrname, s_result)
    ///     return s_result
    /// ```
    pub fn s_getattr(
        this: &Rc<RefCell<ClassDef>>,
        attrname: &str,
        flags: &std::collections::BTreeMap<String, bool>,
    ) -> Result<SomeValue, super::model::AnnotatorException> {
        // upstream: `attrdef = self.find_attribute(attrname);
        //            s_result = attrdef.s_value`.
        let s_result = Self::find_attribute(this, attrname)?;
        match &s_result {
            SomeValue::PBC(pbc) => Ok(Self::lookup_filter(this, pbc, Some(attrname), flags)?),
            SomeValue::Impossible => {
                // upstream: `self.check_missing_attribute_update(attrname)`.
                Self::check_missing_attribute_update(this, attrname)?;
                // upstream: `for basedef in self.getmro(): if
                //            basedef.classdesc.all_enforced_attrs is not None
                //            and attrname in all_enforced_attrs:
                //            raise HarmlesslyBlocked("get enforced attr")`.
                for basedef in Self::getmro(this) {
                    let cdesc = basedef.borrow().classdesc.clone();
                    let all_enforced = cdesc.borrow().all_enforced_attrs.clone();
                    if let Some(attrs) = all_enforced {
                        if attrs.contains(attrname) {
                            return Err(super::model::HarmlesslyBlocked.into());
                        }
                    }
                }
                Ok(s_result)
            }
            SomeValue::List(_) => {
                // upstream: `s_result = self.classdesc.maybe_return_immutable_list(
                //                          attrname, s_result)`.
                let classdesc = this.borrow().classdesc.clone();
                Ok(ClassDesc::maybe_return_immutable_list(
                    &classdesc, attrname, &s_result,
                )?)
            }
            _ => Ok(s_result),
        }
    }

    /// RPython `ClassDef.check_attr_here(self, name)` (classdesc.py:389-400).
    ///
    /// ```python
    /// def check_attr_here(self, name):
    ///     source = self.classdesc.find_source_for(name)
    ///     if source is not None:
    ///         self.add_source_for_attribute(name, source)
    ///         for subdef in self.getallsubdefs():
    ///             if subdef is not self:
    ///                 subdef.check_attr_here(name)
    ///         return True
    ///     else:
    ///         return False
    /// ```
    pub fn check_attr_here(
        this: &Rc<RefCell<ClassDef>>,
        name: &str,
    ) -> Result<bool, AnnotatorError> {
        let classdesc = this.borrow().classdesc.clone();
        let source = ClassDesc::find_source_for(&classdesc, name)?;
        let Some(source) = source else {
            return Ok(false);
        };
        Self::add_source_for_attribute(this, name, source)?;
        for subdef in Self::getallsubdefs(this) {
            if !Rc::ptr_eq(&subdef, this) {
                Self::check_attr_here(&subdef, name)?;
            }
        }
        Ok(true)
    }

    /// RPython `ClassDef.check_missing_attribute_update(self, name)`
    /// (classdesc.py:376-387).
    ///
    /// ```python
    /// def check_missing_attribute_update(self, name):
    ///     found = False
    ///     parents = list(self.getmro())
    ///     parents.reverse()
    ///     for base in parents:
    ///         if base.check_attr_here(name):
    ///             found = True
    ///     return found
    /// ```
    pub fn check_missing_attribute_update(
        this: &Rc<RefCell<ClassDef>>,
        name: &str,
    ) -> Result<bool, AnnotatorError> {
        let mut found = false;
        let mut parents = Self::getmro(this);
        parents.reverse();
        for base in parents {
            if Self::check_attr_here(&base, name)? {
                found = true;
            }
        }
        Ok(found)
    }

    /// RPython `ClassDef.see_instance(self, x)` (classdesc.py:402-416).
    ///
    /// ```python
    /// _see_instance_flattenrec = FlattenRecursion()
    ///
    /// def see_instance(self, x):
    ///     assert isinstance(x, self.classdesc.pyobj)
    ///     key = Hashable(x)
    ///     if key in self.instances_seen:
    ///         return
    ///     self.instances_seen.add(key)
    ///     self.bookkeeper.event('mutable', x)
    ///     source = InstanceSource(self.bookkeeper, x)
    ///     def delayed():
    ///         for attr in source.all_instance_attributes():
    ///             self.add_source_for_attribute(attr, source)
    ///             # ^^^ can trigger reflowing
    ///     self._see_instance_flattenrec(delayed)
    /// ```
    ///
    /// `_see_instance_flattenrec` is a class-attribute
    /// [`FlattenRecursion`] — upstream inherits `TlsClass`, so one
    /// per thread. Rust port stores the flattener in a `thread_local!`
    /// to match the TLS scope. `bookkeeper.event` is the default
    /// empty hook (bookkeeper.py:78-79) so the dispatch is omitted.
    pub fn see_instance(
        this: &Rc<RefCell<ClassDef>>,
        x: &HostObject,
    ) -> Result<(), AnnotatorError> {
        // upstream: `key = Hashable(x); if key in self.instances_seen: return`.
        let key = x.identity_id();
        {
            let mut borrowed = this.borrow_mut();
            if !borrowed.instances_seen.insert(key) {
                return Ok(());
            }
        }
        // upstream: `source = InstanceSource(self.bookkeeper, x)`.
        let bookkeeper = this.borrow().bookkeeper.upgrade().ok_or_else(|| {
            AnnotatorError::new("ClassDef.see_instance: bookkeeper backlink dropped")
        })?;
        let inst_source = InstanceSource::new(&bookkeeper, x.clone());
        // upstream: `def delayed(): for attr in source.all_instance_attributes():
        //                             self.add_source_for_attribute(attr, source)`.
        let this_cloned = this.clone();
        let delayed = Box::new(move || -> Result<(), AnnotatorError> {
            let attrs = inst_source.all_instance_attributes()?;
            for attr in attrs {
                Self::add_source_for_attribute(
                    &this_cloned,
                    &attr,
                    AttrSource::Instance(inst_source.clone()),
                )?;
            }
            Ok(())
        });
        // upstream: `self._see_instance_flattenrec(delayed)`.
        SEE_INSTANCE_FLATTENREC.with(|flat| flat.call(delayed))
    }

    /// RPython `ClassDef.generalize_attr(self, attr, s_value=None)`
    /// (classdesc.py:315-322).
    pub fn generalize_attr(
        this: &Rc<RefCell<ClassDef>>,
        attr: &str,
        s_value: Option<SomeValue>,
    ) -> Result<(), AnnotatorError> {
        if let Some(cdef) = Self::get_owner(this, attr) {
            Self::generalize_attr_internal(&cdef, attr, s_value)
        } else {
            Self::generalize_attr_internal(this, attr, s_value)
        }
    }

    /// RPython `ClassDef._generalize_attr(self, attr, s_value)`
    /// (classdesc.py:274-313).
    fn generalize_attr_internal(
        this: &Rc<RefCell<ClassDef>>,
        attr: &str,
        s_value: Option<SomeValue>,
    ) -> Result<(), AnnotatorError> {
        let mut newattr = Attribute::new(attr);
        if let Some(sv) = s_value {
            newattr.s_value = sv;
        }

        let mut constant_sources: Vec<(Rc<RefCell<ClassDef>>, AttrSource)> = Vec::new();

        // upstream: remove attribute from subclasses, merging into newattr.
        for subdef in Self::getallsubdefs(this) {
            let existing = subdef.borrow_mut().attrs.remove(attr);
            if let Some(subattr) = existing {
                newattr.merge(&subattr, this)?;
            }
            // accumulate attr_sources from all subclasses.
            let taken = subdef.borrow_mut().attr_sources.remove(attr);
            if let Some(sources) = taken {
                for source in sources {
                    constant_sources.push((subdef.clone(), source));
                }
            }
        }

        // upstream: accumulate attr_sources from all parents.
        for superdef in Self::getmro(this) {
            let sources_here: Vec<AttrSource> = superdef
                .borrow()
                .attr_sources
                .get(attr)
                .cloned()
                .unwrap_or_default();
            for source in sources_here {
                if !source.instance_level() {
                    constant_sources.push((superdef.clone(), source));
                }
            }
        }

        // install the new Attribute.
        this.borrow_mut().attrs.insert(attr.to_string(), newattr);

        // feed the pending constant sources.
        for (origin_classdef, source) in constant_sources {
            let mut this_mut = this.borrow_mut();
            let attrdef = this_mut.attrs.get_mut(attr).unwrap();
            attrdef.add_constant_source(&origin_classdef, &source)?;
        }

        // upstream: self.bookkeeper.update_attr(self, newattr).
        if let Some(bk) = this.borrow().bookkeeper.upgrade() {
            bk.update_attr(this, attr)?;
        }
        Ok(())
    }

    /// RPython `ClassDef.see_new_subclass(self, classdef)`
    /// (classdesc.py:418-422).
    ///
    /// RPython `ClassDef.read_attr__class__(self)` (classdesc.py:424-427).
    ///
    /// ```python
    /// def read_attr__class__(self):
    ///     position = self.bookkeeper.position_key
    ///     self.read_locations_of__class__[position] = True
    ///     return SomePBC([subdef.classdesc for subdef in self.getallsubdefs()])
    /// ```
    #[allow(non_snake_case)]
    pub fn read_attr__class__(this: &Rc<RefCell<ClassDef>>) -> SomeValue {
        // upstream: position = self.bookkeeper.position_key
        let pk_opt = this
            .borrow()
            .bookkeeper
            .upgrade()
            .and_then(|bk| bk.current_position_key());
        // upstream: self.read_locations_of__class__[position] = True
        if let Some(pk) = pk_opt {
            this.borrow_mut().read_locations_of_class.insert(pk, true);
        }
        // upstream: return SomePBC([subdef.classdesc for subdef in self.getallsubdefs()])
        let subdefs = Self::getallsubdefs(this);
        let descriptions: Vec<super::description::DescEntry> = subdefs
            .into_iter()
            .map(|sd| {
                let cd = sd.borrow().classdesc.clone();
                super::description::DescEntry::Class(cd)
            })
            .collect();
        SomeValue::PBC(super::model::SomePBC::new(descriptions, false))
    }

    /// Walks `base` up through its `basedef` chain, firing
    /// `bookkeeper.annotator.reflowfromposition(position)` for every
    /// position previously recorded in `read_locations_of__class__`.
    /// Called from `ClassDef::__init__` after a fresh subclass attaches
    /// itself via `basedef.subdefs.append(self)`.
    fn see_new_subclass_recursive(base: &Rc<RefCell<ClassDef>>, child: &Rc<RefCell<ClassDef>>) {
        // upstream: for position in self.read_locations_of__class__:
        //     self.bookkeeper.annotator.reflowfromposition(position)
        let (positions, bk_opt) = {
            let base_ref = base.borrow();
            let positions: Vec<PositionKey> =
                base_ref.read_locations_of_class.keys().cloned().collect();
            let bk_opt = base_ref.bookkeeper.upgrade();
            (positions, bk_opt)
        };
        if let Some(bk) = bk_opt {
            if let Some(ann) = bk.annotator.borrow().upgrade() {
                for position in positions {
                    ann.reflowfromposition(&position);
                }
            }
        }
        let parent = base.borrow().basedef.clone();
        if let Some(parent_rc) = parent {
            Self::see_new_subclass_recursive(&parent_rc, child);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn compiled_graph_func(src: &str) -> crate::flowspace::model::GraphFunc {
        use rustpython_compiler::{Mode, compile as rp_compile};
        use rustpython_compiler_core::bytecode::ConstantData;

        let code = rp_compile(src, Mode::Exec, "<test>".into(), Default::default())
            .expect("compile should succeed");
        let inner = code
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body should be a code constant");
        crate::flowspace::model::GraphFunc::from_host_code(
            crate::flowspace::bytecode::HostCode::from_code(inner),
            Constant::new(ConstValue::Dict(Default::default())),
            Vec::new(),
        )
    }

    fn make_classdesc(bk: &Rc<Bookkeeper>, name: &str) -> Rc<RefCell<ClassDesc>> {
        let pyobj = HostObject::new_class(name, vec![]);
        Rc::new(RefCell::new(ClassDesc::new_shell(bk, pyobj, name.into())))
    }

    #[test]
    fn attribute_init_rejects_dunder_class() {
        let result = std::panic::catch_unwind(|| Attribute::new("__class__"));
        assert!(result.is_err());
    }

    #[test]
    fn attribute_merge_unions_values() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        let mut a = Attribute::new("x");
        a.s_value = SomeValue::Integer(SomeInteger::new(true, false));
        let mut b = Attribute::new("x");
        b.s_value = SomeValue::Integer(SomeInteger::default());
        a.merge(&b, &cd).expect("merge must succeed for int pair");
        assert!(matches!(a.s_value, SomeValue::Integer(_)));
    }

    #[test]
    fn attribute_modified_sets_readonly_false() {
        let mut a = Attribute::new("x");
        assert!(a.readonly);
        a.modified(None)
            .expect("modified must succeed when attr_allowed");
        assert!(!a.readonly);
    }

    #[test]
    fn attribute_modified_with_forbid_raises() {
        let mut a = Attribute::new("x");
        a.attr_allowed = false;
        a.readonly = false; // already modified path
        let err = a.modified(None).unwrap_err();
        let msg = err.msg.as_deref().unwrap();
        // upstream classdesc.py:129-134 error prefix:
        //   "Attribute %r on %r should be read-only."
        assert!(msg.contains("should be read-only"), "got: {msg}");
        assert!(msg.contains("\"x\""), "got: {msg}");
    }

    #[test]
    fn classdef_read_attr_class_returns_subclass_pbc() {
        use crate::annotator::model::{DescKind, SomeValue};
        let bk = make_bk();
        let parent = make_classdesc(&bk, "pkg.Parent");
        let parent_cd = ClassDef::new(&bk, &parent);
        let child = make_classdesc(&bk, "pkg.Child");
        child.borrow_mut().basedesc = Some(parent.clone());
        let _child_cd = ClassDef::new(&bk, &child);
        // upstream: SomePBC([subdef.classdesc for subdef in self.getallsubdefs()])
        let result = ClassDef::read_attr__class__(&parent_cd);
        match &result {
            SomeValue::PBC(pbc) => {
                let names: Vec<String> = pbc
                    .descriptions
                    .values()
                    .filter_map(|d| d.as_class())
                    .map(|cd| cd.borrow().name.clone())
                    .collect();
                assert!(names.iter().any(|n| n == "pkg.Parent"));
                assert!(names.iter().any(|n| n == "pkg.Child"));
                assert!(
                    pbc.descriptions
                        .values()
                        .all(|d| d.kind() == DescKind::Class)
                );
            }
            _ => panic!("read_attr__class__ must return a SomePBC"),
        }
    }

    #[test]
    fn classdef_init_sets_name_and_shortname() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Sub.Foo");
        let cd = ClassDef::new(&bk, &desc);
        let r = cd.borrow();
        assert_eq!(r.name, "pkg.Sub.Foo");
        assert_eq!(r.shortname, "Foo");
        assert!(r.basedef.is_none());
        assert!(r.repr.is_none());
    }

    #[test]
    fn classdef_getmro_single_returns_self() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        let mro = ClassDef::getmro(&cd);
        assert_eq!(mro.len(), 1);
        assert!(Rc::ptr_eq(&mro[0], &cd));
    }

    #[test]
    fn classdef_getallsubdefs_single_returns_self() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        let subs = ClassDef::getallsubdefs(&cd);
        assert_eq!(subs.len(), 1);
        assert!(Rc::ptr_eq(&subs[0], &cd));
    }

    #[test]
    fn classdef_about_attribute_empty_returns_none() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        assert!(ClassDef::about_attribute(&cd, "missing").is_none());
    }

    #[test]
    fn classdef_has_no_attrs_true_initially() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        assert!(ClassDef::has_no_attrs(&cd));
    }

    #[test]
    fn classdef_freeze_returns_error() {
        let err = ClassDef::_freeze_().unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap()
                .contains("cannot be used as immutablevalue")
        );
    }

    #[test]
    fn instance_source_all_instance_attributes_collects_instance_dict_and_slots() {
        let bk = make_bk();
        let base = HostObject::new_class("pkg.Base", vec![]);
        base.class_set("__slots__", ConstValue::Str("base_slot".into()));
        let cls = HostObject::new_class("pkg.X", vec![base]);
        cls.class_set(
            "__slots__",
            ConstValue::Tuple(vec![ConstValue::Str("slot_a".into())]),
        );
        let obj = HostObject::new_instance(cls, vec![]);
        obj.instance_set("dyn", ConstValue::Int(1));
        let src = InstanceSource::new(&bk, obj);
        let attrs = src.all_instance_attributes().unwrap();
        assert!(attrs.iter().any(|attr| attr == "dyn"));
        assert!(attrs.iter().any(|attr| attr == "slot_a"));
        assert!(attrs.iter().any(|attr| attr == "base_slot"));
    }

    #[test]
    fn attr_source_s_get_value_reads_instance_and_enforced_missing() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.X", vec![]);
        cls.class_set("klass", ConstValue::Int(9));
        let obj = HostObject::new_instance(cls.clone(), vec![]);
        obj.instance_set("x", ConstValue::Int(4));
        let src = AttrSource::Instance(InstanceSource::new(&bk, obj));

        let s_x = src.s_get_value(None, "x").unwrap();
        assert!(matches!(s_x, SomeValue::Integer(_)));

        let s_klass = src.s_get_value(None, "klass").unwrap();
        assert!(matches!(s_klass, SomeValue::Integer(_)));

        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(&bk, cls, "pkg.X".into())));
        desc.borrow_mut()
            .all_enforced_attrs
            .replace(HashSet::from([String::from("must_exist")]));
        let classdef = ClassDef::new(&bk, &desc);
        let s_missing = src.s_get_value(Some(&classdef), "must_exist").unwrap();
        assert!(matches!(s_missing, SomeValue::Impossible));
    }

    #[test]
    fn s_getattr_enforced_attr_raises_harmlessly_blocked() {
        // upstream classdesc.py:181-183: SomeImpossibleValue +
        // attrname ∈ all_enforced_attrs → raise HarmlesslyBlocked.
        use super::super::model::AnnotatorException;
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.X", vec![]);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(&bk, cls, "pkg.X".into())));
        desc.borrow_mut()
            .all_enforced_attrs
            .replace(HashSet::from([String::from("blocked_attr")]));
        let classdef = ClassDef::new(&bk, &desc);
        let flags = std::collections::BTreeMap::new();
        let err = ClassDef::s_getattr(&classdef, "blocked_attr", &flags)
            .expect_err("enforced attr getattr must raise");
        assert!(matches!(err, AnnotatorException::Harmless(_)));
    }

    #[test]
    fn builtin_type_desc_issubclass() {
        let base = HostObject::new_class("BaseException", vec![]);
        let exc = HostObject::new_class("Exception", vec![base.clone()]);
        let exc_desc = BuiltinTypeDesc::new(exc);
        let base_desc = BuiltinTypeDesc::new(base);
        assert!(exc_desc.issubclass(&base_desc));
        assert!(!base_desc.issubclass(&exc_desc));
    }

    #[test]
    fn is_primitive_type_detects_builtins() {
        let int_cls = HostObject::new_class("int", vec![]);
        assert!(is_primitive_type(&int_cls));
        let user_cls = HostObject::new_class("pkg.UserFoo", vec![]);
        assert!(!is_primitive_type(&user_cls));
    }

    #[test]
    fn is_mixin_reads_class_dict_flag() {
        let plain = HostObject::new_class("pkg.Plain", vec![]);
        assert!(!is_mixin(&plain));
        let mixin_cls = HostObject::new_class("pkg.MixinA", vec![]);
        mixin_cls.class_set("_mixin_", ConstValue::Bool(true));
        assert!(is_mixin(&mixin_cls));
    }

    #[test]
    fn is_mixin_rejects_false_value() {
        // upstream treats truthy values only; `_mixin_ = False` stays non-mixin.
        let cls = HostObject::new_class("pkg.NotMixin", vec![]);
        cls.class_set("_mixin_", ConstValue::Bool(false));
        assert!(!is_mixin(&cls));
    }

    #[test]
    fn classdesc_is_exception_class_via_host_env() {
        // Exception is bootstrapped in HOST_ENV → subclass path works.
        let bk = make_bk();
        let base_exc = HOST_ENV
            .lookup_builtin("BaseException")
            .expect("HOST_ENV BaseException bootstrap");
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            base_exc,
            "BaseException".into(),
        )));
        assert!(desc_rc.borrow().is_exception_class());
        // plain user class is NOT an exception class.
        let plain = make_classdesc(&bk, "pkg.Foo");
        assert!(!plain.borrow().is_exception_class());
    }

    #[test]
    fn classdesc_is_builtin_exception_class_true_for_bootstrapped_names() {
        let bk = make_bk();
        let type_err = HOST_ENV
            .lookup_builtin("TypeError")
            .expect("HOST_ENV TypeError bootstrap");
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            type_err,
            "TypeError".into(),
        )));
        assert!(desc_rc.borrow().is_builtin_exception_class());
    }

    #[test]
    fn classdesc_is_builtin_exception_class_false_for_user_class() {
        let bk = make_bk();
        let plain = make_classdesc(&bk, "pkg.NotExc");
        assert!(!plain.borrow().is_builtin_exception_class());
    }

    // --- c2 step 2c-ii: ClassDesc::new + add_source_attribute + add_mixins ---

    #[test]
    fn classdesc_new_rejects_not_rpython_marker() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.Bad", vec![]);
        cls.class_set("__NOT_RPYTHON__", ConstValue::Bool(true));
        let err = ClassDesc::new(&bk, cls, None, None, None).unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("__NOT_RPYTHON__"));
    }

    #[test]
    fn classdesc_new_rejects_annspecialcase() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.OldSpec", vec![]);
        cls.class_set("_annspecialcase_", ConstValue::Str("memo".into()));
        let err = ClassDesc::new(&bk, cls, None, None, None).unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("_annspecialcase_"));
    }

    #[test]
    fn classdesc_new_rejects_mixin_used_directly() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.MixinDirect", vec![]);
        cls.class_set("_mixin_", ConstValue::Bool(true));
        let err = ClassDesc::new(&bk, cls, None, None, None).unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("_mixin_"));
    }

    #[test]
    fn classdesc_new_populates_classdict_from_pyobj() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.WithAttr", vec![]);
        cls.class_set("x", ConstValue::Int(42));
        cls.class_set("y", ConstValue::Bool(false));
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let cd = desc_rc.borrow();
        assert!(cd.classdict.contains_key("x"));
        assert!(cd.classdict.contains_key("y"));
        match cd.classdict.get("x").unwrap() {
            ClassDictEntry::Constant(c) => match &c.value {
                ConstValue::Int(42) => {}
                other => panic!("expected Int(42), got {other:?}"),
            },
            other => panic!("expected Constant entry, got {other:?}"),
        }
    }

    #[test]
    fn classdesc_new_extracts_immutable_fields() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.Immut", vec![]);
        cls.class_set(
            "_immutable_fields_",
            ConstValue::Tuple(vec![
                ConstValue::Str("a".into()),
                ConstValue::Str("b".into()),
            ]),
        );
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let cd = desc_rc.borrow();
        assert!(cd.immutable_fields.contains("a"));
        assert!(cd.immutable_fields.contains("b"));
    }

    #[test]
    fn classdesc_new_rejects_str_immutable_fields() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.ImmutStr", vec![]);
        cls.class_set("_immutable_fields_", ConstValue::Str("justone".into()));
        let err = ClassDesc::new(&bk, cls, None, None, None).unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("not a string"));
    }

    #[test]
    fn classdesc_new_resolves_basedesc_via_bookkeeper() {
        let bk = make_bk();
        let base = HostObject::new_class("pkg.Base", vec![]);
        let child = HostObject::new_class("pkg.Child", vec![base.clone()]);
        let child_desc = ClassDesc::new(&bk, child, None, None, None).unwrap();
        // The recursive bookkeeper.getdesc(base) must produce a real
        // ClassDesc and install it as basedesc.
        let basedesc_opt = child_desc.borrow().basedesc.clone();
        let basedesc_rc = basedesc_opt.expect("basedesc must be set via getdesc");
        assert_eq!(basedesc_rc.borrow().name, "pkg.Base");
        // base is also cached in bookkeeper.descs (identity-shared).
        let cached = bk.getdesc(&base).unwrap();
        let cached_rc = cached.as_class().unwrap();
        assert!(Rc::ptr_eq(&basedesc_rc, &cached_rc));
    }

    #[test]
    fn classdesc_new_object_base_stays_none() {
        let bk = make_bk();
        let object_cls = HOST_ENV.lookup_builtin("object").unwrap();
        let cls = HostObject::new_class("pkg.DirectObject", vec![object_cls.clone()]);
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        // `object` is skipped during partitioning — no basedesc installed.
        assert!(desc_rc.borrow().basedesc.is_none());
    }

    #[test]
    fn classdesc_new_slots_enforce_attrs() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.Slotted", vec![]);
        cls.class_set(
            "__slots__",
            ConstValue::Tuple(vec![
                ConstValue::Str("x".into()),
                ConstValue::Str("y".into()),
            ]),
        );
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let cd = desc_rc.borrow();
        let enforced = cd.all_enforced_attrs.as_ref().expect("__slots__ enforced");
        assert!(enforced.contains("x"));
        assert!(enforced.contains("y"));
    }

    #[test]
    fn classdesc_add_mixins_brings_in_mixin_dict() {
        let bk = make_bk();
        // mixin class M with attr `m_attr`.
        let mixin_cls = HostObject::new_class("pkg.M", vec![]);
        mixin_cls.class_set("_mixin_", ConstValue::Bool(true));
        mixin_cls.class_set("m_attr", ConstValue::Int(100));
        // User class C inherits from M (and implicitly from object).
        let user_cls = HostObject::new_class("pkg.C", vec![mixin_cls.clone()]);
        let desc_rc = ClassDesc::new(&bk, user_cls, None, None, None).unwrap();
        let cd = desc_rc.borrow();
        // mixin attr was copied into the user class's classdict.
        assert!(cd.classdict.contains_key("m_attr"));
        // _mixin_ marker is also copied (upstream behaviour — skip only
        // uses check_not_in's MRO dict, not a fixed blocklist).
    }

    #[test]
    fn classdesc_add_mixins_skip_set_respects_base() {
        let bk = make_bk();
        // mixin class M with attr `shared`.
        let mixin_cls = HostObject::new_class("pkg.M", vec![]);
        mixin_cls.class_set("_mixin_", ConstValue::Bool(true));
        mixin_cls.class_set("shared", ConstValue::Int(100));
        // Base class B with attr `shared` too.
        let base_cls = HostObject::new_class("pkg.B", vec![]);
        base_cls.class_set("shared", ConstValue::Int(200));
        // Upstream partitioning walks bases in order: real base FIRST,
        // mixin AFTER → mixin lands in `mixins_after` with
        // check_not_in=base → skip set from B's MRO protects `shared`.
        let user_cls = HostObject::new_class("pkg.C2", vec![base_cls.clone(), mixin_cls.clone()]);
        let desc_rc = ClassDesc::new(&bk, user_cls, None, None, None).unwrap();
        let cd = desc_rc.borrow();
        // C2's own classdict must NOT carry `shared` from the mixin —
        // base B already provides it, so the skip set protected it.
        assert!(!cd.classdict.contains_key("shared"));
    }

    #[test]
    fn classdesc_add_source_attribute_mixin_function_stores_desc() {
        use crate::flowspace::model::GraphFunc;
        let bk = make_bk();
        // Build a target classdesc.
        let cls = HostObject::new_class("pkg.HasMixed", vec![]);
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            cls,
            "pkg.HasMixed".into(),
        )));
        // Simulate add_source_attribute storing a FunctionType/mixin=true.
        let gf = GraphFunc::new("f", Constant::new(ConstValue::Dict(Default::default())));
        let fn_host = HostObject::new_user_function(gf);
        ClassDesc::add_source_attribute(&desc_rc, "foo", ConstValue::HostObject(fn_host), true)
            .unwrap();
        let cd = desc_rc.borrow();
        match cd.classdict.get("foo").unwrap() {
            ClassDictEntry::Desc(_) => {}
            other => panic!("expected Desc entry for mixin FunctionType, got {other:?}"),
        }
    }

    #[test]
    fn classdesc_add_source_attribute_property_expands_getter_setter() {
        use crate::flowspace::model::GraphFunc;
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.WithProp", vec![]);
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            cls,
            "pkg.WithProp".into(),
        )));
        // Build a property(fget, fset) with real UserFunction hosts.
        let fget_host = HostObject::new_user_function(GraphFunc::new(
            "get_x",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        let fset_host = HostObject::new_user_function(GraphFunc::new(
            "set_x",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        let prop_host = HostObject::new_property(
            "pkg.WithProp.x",
            Some(fget_host.clone()),
            Some(fset_host.clone()),
            None,
        );
        ClassDesc::add_source_attribute(
            &desc_rc,
            "x",
            ConstValue::HostObject(prop_host.clone()),
            false,
        )
        .unwrap();
        let cd = desc_rc.borrow();
        // upstream classdesc.py:591-602 — property itself lives under
        // `name`, getter/setter aliases under `name__getter__` /
        // `name__setter__`.
        assert!(cd.classdict.contains_key("x"));
        assert!(cd.classdict.contains_key("x__getter__"));
        assert!(cd.classdict.contains_key("x__setter__"));
        // The `x` slot carries the property Constant, not a Desc.
        match cd.classdict.get("x").unwrap() {
            ClassDictEntry::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => assert!(h.is_property()),
                other => panic!("expected property HostObject, got {other:?}"),
            },
            other => panic!("expected Constant entry for property, got {other:?}"),
        }
        match cd.classdict.get("x__getter__").unwrap() {
            ClassDictEntry::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => {
                    let func = h.user_function().expect("getter must stay a user function");
                    assert_eq!(func.name, "x__getter__");
                }
                other => panic!("expected getter HostObject, got {other:?}"),
            },
            other => panic!("expected Constant getter entry, got {other:?}"),
        }
        match cd.classdict.get("x__setter__").unwrap() {
            ClassDictEntry::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => {
                    let func = h.user_function().expect("setter must stay a user function");
                    assert_eq!(func.name, "x__setter__");
                }
                other => panic!("expected setter HostObject, got {other:?}"),
            },
            other => panic!("expected Constant setter entry, got {other:?}"),
        }
    }

    #[test]
    fn classdesc_add_source_attribute_mixin_staticmethod_clones_wrapper_function() {
        use crate::flowspace::model::GraphFunc;
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.WithStatic", vec![]);
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            cls,
            "pkg.WithStatic".into(),
        )));
        let fn_host = HostObject::new_user_function(GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        let static_host = HostObject::new_staticmethod("pkg.WithStatic.f", fn_host.clone());
        ClassDesc::add_source_attribute(&desc_rc, "f", ConstValue::HostObject(static_host), true)
            .unwrap();
        let cd = desc_rc.borrow();
        match cd.classdict.get("f").unwrap() {
            ClassDictEntry::Constant(c) => match &c.value {
                ConstValue::HostObject(h) => {
                    assert!(h.is_staticmethod());
                    let stored_func = h.staticmethod_func().expect("wrapped function");
                    assert!(
                        !stored_func.is_classmethod() && !stored_func.is_staticmethod(),
                        "staticmethod must wrap the bare function object"
                    );
                    assert_ne!(
                        stored_func.identity_id(),
                        fn_host.identity_id(),
                        "mixin staticmethod should clone the wrapped function identity"
                    );
                }
                other => panic!("expected staticmethod HostObject, got {other:?}"),
            },
            other => panic!("expected Constant entry for staticmethod, got {other:?}"),
        }
    }

    #[test]
    fn classdesc_add_source_attribute_skips_builtin_exc_init() {
        // upstream classdesc.py:627-633 — `if __init__ and
        // is_builtin_exception_class() and value not in BUILTIN_ANALYZERS:
        // return`. A non-HostObject ConstValue (here an Int sentinel) is
        // trivially unregistered, so the assignment drops.
        let bk = make_bk();
        let base_exc = HOST_ENV.lookup_builtin("BaseException").unwrap();
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            base_exc,
            "BaseException".into(),
        )));
        ClassDesc::add_source_attribute(&desc_rc, "__init__", ConstValue::Int(0), false).unwrap();
        assert!(!desc_rc.borrow().classdict.contains_key("__init__"));
    }

    #[test]
    fn classdesc_add_source_attribute_keeps_registered_builtin_exc_init() {
        // upstream classdesc.py:627-633 positive side — when the value
        // IS in BUILTIN_ANALYZERS (e.g. `object.__init__` registered by
        // `builtin.py:198`), the assignment survives even on a builtin
        // exception class.
        let bk = make_bk();
        let base_exc = HOST_ENV.lookup_builtin("BaseException").unwrap();
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            base_exc,
            "BaseException".into(),
        )));
        let init_host = HostObject::new_builtin_callable("object.__init__");
        ClassDesc::add_source_attribute(
            &desc_rc,
            "__init__",
            ConstValue::HostObject(init_host),
            false,
        )
        .unwrap();
        assert!(desc_rc.borrow().classdict.contains_key("__init__"));
    }

    #[test]
    fn maybe_return_immutable_list_marks_copy_when_attr_matches() {
        // upstream classdesc.py:836-839 — `x.lst` with lst[*] in
        // `_immutable_fields_` must return a marked-immutable list copy.
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.C", vec![]);
        cls.class_set(
            "_immutable_fields_",
            ConstValue::Tuple(vec![ConstValue::Str("lst[*]".to_string())]),
        );
        let desc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let s_list = SomeValue::List(SomeList::new(ListDef::new(
            Some(bk.clone()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        )));
        let out = ClassDesc::maybe_return_immutable_list(&desc, "lst", &s_list)
            .expect("attr matches lst[*] so must return a marked copy");
        let SomeValue::List(marked) = out else {
            panic!("expected SomeList copy");
        };
        assert!(marked.listdef.listitem_rc().borrow().immutable);
    }

    #[test]
    fn maybe_return_immutable_list_passes_through_when_no_match() {
        use crate::annotator::annrpython::RPythonAnnotator;
        use crate::annotator::listdef::ListDef;
        use crate::annotator::model::{SomeInteger, SomeList};
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.C", vec![]);
        // _immutable_fields_ omitted — attr doesn't match.
        let desc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let s_list = SomeValue::List(SomeList::new(ListDef::new(
            Some(bk.clone()),
            SomeValue::Integer(SomeInteger::default()),
            false,
            false,
        )));
        let out = ClassDesc::maybe_return_immutable_list(&desc, "other", &s_list).unwrap();
        // upstream: returns s_result unchanged ("common case").
        assert_eq!(out, s_list);
    }

    #[test]
    fn classdesc_getuniqueclassdef_lazy_inits_and_caches() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.Lazy", vec![]);
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let cd1 = ClassDesc::getuniqueclassdef(&desc_rc).unwrap();
        let cd2 = ClassDesc::getuniqueclassdef(&desc_rc).unwrap();
        // Second call returns the same Rc (identity-cached via Weak).
        assert!(Rc::ptr_eq(&cd1, &cd2));
        // bookkeeper.classdefs retains the classdef.
        let snap = bk.classdef_snapshot();
        assert_eq!(snap.len(), 1);
        assert!(Rc::ptr_eq(&snap[0], &cd1));
    }

    #[test]
    fn classdesc_init_classdef_emulates_del_call() {
        // upstream classdesc.py:691-696 — `_init_classdef` dispatches
        // `__del__` through `bookkeeper.emulate_pbc_call`. The new
        // emulate_pbc_call path delegates to real `FunctionDesc.pycall`
        // (bookkeeper.py:570-572 `self.pbc_call(...)`), which requires a
        // live annotator backlink — so the test wires up the full
        // `RPythonAnnotator` rather than a bare Bookkeeper.
        use crate::annotator::annrpython::RPythonAnnotator;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let bk = ann.bookkeeper.clone();
        let cls = HostObject::new_class("pkg.HasDel", vec![]);
        cls.class_set(
            "__del__",
            ConstValue::HostObject(HostObject::new_user_function(compiled_graph_func(
                "def __del__(self):\n    return None\n",
            ))),
        );
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let classdef = ClassDesc::_init_classdef(&desc_rc).unwrap();
        let key = EmulatedPbcCallKey::ClassDef(
            super::super::description::ClassDefKey::from_classdef(&classdef),
        );
        assert!(bk.emulated_pbc_calls.borrow().contains_key(&key));
    }

    #[test]
    fn bookkeeper_getdesc_class_returns_full_classdesc_with_classdict() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.Full", vec![]);
        cls.class_set("x", ConstValue::Int(1));
        cls.class_set("y", ConstValue::Bool(true));
        let entry = bk.getdesc(&cls).unwrap();
        let cd = entry.as_class().unwrap();
        // Full __init__ body ran — classdict has x/y via add_sources_for_class.
        assert!(cd.borrow().classdict.contains_key("x"));
        assert!(cd.borrow().classdict.contains_key("y"));
    }

    #[test]
    fn classdict_entry_constant_and_desc_variants() {
        let c = ClassDictEntry::constant(ConstValue::Int(7));
        assert!(matches!(c, ClassDictEntry::Constant(_)));
        // Desc variant is constructable from a DescEntry handle — use a
        // Function entry via the DescEntry::is_function helper to avoid
        // re-porting bookkeeper.newfuncdesc in the test.
        use crate::flowspace::model::GraphFunc;
        let bk = make_bk();
        let host = HostObject::new_user_function(GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        let entry = bk.getdesc(&host).unwrap();
        let stored = ClassDictEntry::Desc(entry);
        assert!(matches!(stored, ClassDictEntry::Desc(_)));
    }

    #[test]
    fn force_attributes_into_classes_has_environment_error() {
        let map = force_attributes_into_classes();
        let env = map
            .get("EnvironmentError")
            .expect("EnvironmentError entry present");
        assert!(env.contains_key("errno"));
        assert!(env.contains_key("strerror"));
        assert!(env.contains_key("filename"));
    }

    #[test]
    fn classdef_generalize_attr_installs_attribute() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        ClassDef::generalize_attr(&cd, "x", Some(SomeValue::Integer(SomeInteger::default())))
            .expect("generalize_attr must succeed without sources");
        assert!(cd.borrow().attrs.contains_key("x"));
    }

    #[test]
    fn classdef_find_attribute_creates_on_miss() {
        let bk = make_bk();
        let desc = make_classdesc(&bk, "pkg.Foo");
        let cd = ClassDef::new(&bk, &desc);
        let sv = ClassDef::find_attribute(&cd, "y").expect("find_attribute must succeed");
        // New attribute starts at SomeValue::Impossible because no source was supplied.
        assert!(matches!(sv, SomeValue::Impossible));
    }

    #[test]
    fn classdesc_consider_call_site_merges_call_families() {
        use super::super::argument::simple_args;
        use super::super::description::DescKey;
        use super::super::model::SomeValue;
        let bk = make_bk();
        // Phase 2 requires a common base — upstream classdesc.py:912
        // raises `AssertionError("no common base")` when multi-class
        // call sites have no shared ancestor.
        let common = make_classdesc(&bk, "pkg.Common");
        // getuniqueclassdef populates Common's classdef so child
        // classdefs can `upgrade` the Weak backref when initialised.
        let _common_cd = ClassDesc::getuniqueclassdef(&common).unwrap();
        let a = make_classdesc(&bk, "pkg.A");
        a.borrow_mut().basedesc = Some(common.clone());
        let b = make_classdesc(&bk, "pkg.B");
        b.borrow_mut().basedesc = Some(common.clone());
        let a_key = DescKey::from_rc(&a);
        let b_key = DescKey::from_rc(&b);
        // Pre-condition: A and B belong to distinct CallFamilies.
        {
            let mut families = bk.pbc_maximal_call_families.borrow_mut();
            let rep_a = families.find_rep(a_key);
            let rep_b = families.find_rep(b_key);
            assert_ne!(rep_a, rep_b);
        }
        // Run consider_call_site — should union the two families.
        let args = simple_args(vec![]);
        ClassDesc::consider_call_site(&[a.clone(), b.clone()], &args, &SomeValue::Impossible, None)
            .expect("consider_call_site must succeed");
        // Post-condition: A and B share a CallFamily.
        let mut families = bk.pbc_maximal_call_families.borrow_mut();
        let rep_a = families.find_rep(a_key);
        let rep_b = families.find_rep(b_key);
        assert_eq!(rep_a, rep_b);
    }

    #[test]
    fn classdesc_consider_call_site_rejects_no_common_base() {
        // classdesc.py:912 assertion: multi-class PBC call site with no
        // shared ancestor raises AnnotatorError.
        use super::super::argument::simple_args;
        use super::super::model::SomeValue;
        let bk = make_bk();
        let a = make_classdesc(&bk, "pkg.Standalone1");
        let b = make_classdesc(&bk, "pkg.Standalone2");
        let args = simple_args(vec![]);
        let err = ClassDesc::consider_call_site(
            &[a.clone(), b.clone()],
            &args,
            &SomeValue::Impossible,
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("no common base"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn classdesc_consider_call_site_single_class_requires_instance_result() {
        use super::super::argument::simple_args;
        use super::super::model::SomeValue;
        let bk = make_bk();
        let common = make_classdesc(&bk, "pkg.Common");
        let _ = ClassDesc::getuniqueclassdef(&common).unwrap();
        let args = simple_args(vec![]);
        let err = ClassDesc::consider_call_site(&[common], &args, &SomeValue::Impossible, None)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("calling a class didn't return an instance??"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn classdesc_getcommonbase_walks_chain() {
        // classdesc.py:906-915: getcommonbase([A,B]) returns the nearest
        // shared ancestor. When A and B share a direct parent P, the
        // common base is P.
        let bk = make_bk();
        let p = make_classdesc(&bk, "pkg.P");
        let a = make_classdesc(&bk, "pkg.A");
        a.borrow_mut().basedesc = Some(p.clone());
        let b = make_classdesc(&bk, "pkg.B");
        b.borrow_mut().basedesc = Some(p.clone());
        let common = ClassDesc::getcommonbase(&[a.clone(), b.clone()]).unwrap();
        assert!(Rc::ptr_eq(&common, &p));
    }

    #[test]
    fn classdesc_lookup_walks_basedesc_chain() {
        use super::super::super::flowspace::model::ConstValue;
        let bk = make_bk();
        let parent = make_classdesc(&bk, "pkg.Parent");
        parent
            .borrow_mut()
            .classdict
            .insert("x".into(), ClassDictEntry::constant(ConstValue::Int(7)));
        let child = make_classdesc(&bk, "pkg.Child");
        child.borrow_mut().basedesc = Some(parent.clone());

        // Direct hit on parent — child lookup walks the chain.
        let found = ClassDesc::lookup(&child, "x").expect("x should be found on parent");
        assert!(Rc::ptr_eq(&found, &parent));

        // Missing name — None.
        assert!(ClassDesc::lookup(&child, "nonexistent").is_none());

        // read_attribute returns the same constant entry.
        let entry = ClassDesc::read_attribute(&child, "x").expect("x should be readable");
        match entry {
            ClassDictEntry::Constant(c) => assert_eq!(c.value, ConstValue::Int(7)),
            _ => panic!("expected Constant entry"),
        }
    }

    #[test]
    fn classdesc_s_read_attribute_returns_impossible_for_missing() {
        let bk = make_bk();
        let cdesc = make_classdesc(&bk, "pkg.Empty");
        let sv = ClassDesc::s_read_attribute(&cdesc, "missing")
            .expect("s_read_attribute must succeed with Impossible");
        assert!(matches!(sv, SomeValue::Impossible));
    }

    #[test]
    fn classdesc_s_get_value_unwraps_constants() {
        use super::super::super::flowspace::model::ConstValue;
        use super::super::model::SomeValue;
        let bk = make_bk();
        let cdesc = make_classdesc(&bk, "pkg.Holder");
        cdesc
            .borrow_mut()
            .classdict
            .insert("n".into(), ClassDictEntry::constant(ConstValue::Int(42)));
        let sv = ClassDesc::s_read_attribute(&cdesc, "n").expect("immutablevalue path");
        // immutablevalue(Int(42)) is a constant-bound SomeInteger.
        match sv {
            SomeValue::Integer(_) => {}
            other => panic!("expected SomeInteger from immutablevalue, got {:?}", other),
        }
    }

    #[test]
    fn classdesc_s_get_value_unwraps_staticmethod_without_binding() {
        use crate::flowspace::model::GraphFunc;

        let bk = make_bk();
        let cdesc = make_classdesc(&bk, "pkg.StaticHolder");
        let classdef = ClassDesc::getuniqueclassdef(&cdesc).unwrap();
        let fn_host = HostObject::new_user_function(GraphFunc::new(
            "f",
            Constant::new(ConstValue::Dict(Default::default())),
        ));
        cdesc.borrow_mut().classdict.insert(
            "f".into(),
            ClassDictEntry::constant(ConstValue::HostObject(HostObject::new_staticmethod(
                "pkg.StaticHolder.f",
                fn_host,
            ))),
        );

        let sv = ClassDesc::s_get_value(&cdesc, Some(&classdef), "f").unwrap();

        let SomeValue::PBC(pbc) = sv else {
            panic!("expected SomePBC from staticmethod lookup");
        };
        assert_eq!(pbc.descriptions.len(), 1);
        let desc = pbc.descriptions.values().next().unwrap();
        assert!(desc.is_function(), "staticmethod lookup must stay unbound");
    }

    #[test]
    fn classdesc_s_get_value_rejects_classmethod() {
        use crate::flowspace::model::GraphFunc;

        let bk = make_bk();
        let cdesc = make_classdesc(&bk, "pkg.ClassMethodHolder");
        cdesc.borrow_mut().classdict.insert(
            "f".into(),
            ClassDictEntry::constant(ConstValue::HostObject(HostObject::new_classmethod(
                "pkg.ClassMethodHolder.f",
                HostObject::new_user_function(GraphFunc::new(
                    "f",
                    Constant::new(ConstValue::Dict(Default::default())),
                )),
            ))),
        );

        let err = ClassDesc::s_get_value(&cdesc, None, "f").unwrap_err();
        assert!(
            err.msg
                .unwrap_or_default()
                .contains("classmethods are not supported")
        );
    }

    #[test]
    fn classdesc_pycall_rejects_non_callable_init() {
        let bk = make_bk();
        let cdesc = make_classdesc(&bk, "pkg.BadInit");
        cdesc.borrow_mut().classdict.insert(
            "__init__".into(),
            ClassDictEntry::constant(ConstValue::Int(7)),
        );
        let args = super::super::argument::ArgumentsForTranslation::new(vec![], None, None);

        let err = ClassDesc::pycall(&cdesc, None, &args, &SomeValue::Impossible, None).unwrap_err();

        assert!(
            err.msg
                .unwrap_or_default()
                .contains("Cannot prove that the object is callable")
        );
    }

    #[test]
    #[should_panic]
    fn classdesc_pycall_exception_rejects_someptr_arg() {
        use crate::annotator::model::SomePtr;
        use crate::flowspace::model::{Block, FunctionGraph, Hlvalue, Variable};
        use crate::translator::rtyper::lltypesystem::lltype;

        let bk = make_bk();
        let exc = HOST_ENV
            .lookup_builtin("Exception")
            .expect("HOST_ENV must bootstrap Exception");
        let cdesc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            exc,
            "Exception".into(),
        )));

        let start = Rc::new(RefCell::new(Block::new(vec![])));
        let ret = Variable::new();
        ret.set_concretetype(Some(lltype::LowLevelType::Void));
        let graph = Rc::new(RefCell::new(FunctionGraph::with_return_var(
            "f",
            start,
            Hlvalue::Variable(ret),
        )));
        let ptr = lltype::getfunctionptr(&graph, lltype::_getconcretetype).unwrap();
        let s_arg = SomeValue::Ptr(SomePtr::new(lltype::typeOf(&ptr)));
        let args = super::super::argument::ArgumentsForTranslation::new(vec![s_arg], None, None);

        let _ = ClassDesc::pycall(&cdesc, None, &args, &SomeValue::Impossible, None);
    }
}
