//! Type inference for user-defined classes.
//!
//! RPython upstream: `rpython/annotator/classdesc.py` (968 LOC).
//!
//! This commit ports the first slice of classdesc.py:
//!
//! | upstream | Rust |
//! |---|---|
//! | `Attribute` (classdesc.py:72-134) | [`Attribute`] |
//! | `ClassDef` data + structural methods (classdesc.py:136-431) | [`ClassDef`] (c3 hooks deferred) |
//! | `InstanceSource` (classdesc.py:435-464) | [`InstanceSource`] |
//! | `NoSuchAttrError` (classdesc.py:466-468) | [`NoSuchAttrError`] |
//! | `is_mixin` (classdesc.py:471) | [`is_mixin`] |
//! | `is_primitive_type` (classdesc.py:474) | [`is_primitive_type`] |
//! | `BuiltinTypeDesc` (classdesc.py:479-485) | [`BuiltinTypeDesc`] |
//! | `FORCE_ATTRIBUTES_INTO_CLASSES` (classdesc.py:957-961) | [`force_attributes_into_classes`] |
//! | `ClassDesc` stub shell (classdesc.py:488-600) | [`ClassDesc`] |
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
//! ## PRE-EXISTING-ADAPTATION: deferred c3 methods
//!
//! `ClassDef::{s_getattr, lookup_filter, check_missing_attribute_update,
//! check_attr_here, see_instance}` surface as fail-fast
//! [`AnnotatorError`] until c3 lands the bookkeeper + MethodDesc.bind_self
//! machinery they depend on. Structural data/method parity is preserved so
//! the c3 follow-up is a pure additive patch.
//!
//! ## PRE-EXISTING-ADAPTATION: HostObject class-dict reflection
//!
//! Upstream reflection such as `cls.__dict__.get('_mixin_')` is not
//! available on [`HostObject`] yet; [`is_mixin`] / [`is_primitive_type`]
//! / [`ClassDesc::new`] carry stub behaviour that defaults to "not a
//! mixin" / "not primitive" with a comment citing the upstream line.
//! Full reflection lands alongside c2 where `ClassDesc.__init__` body
//! is ported.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::{Rc, Weak};

use super::bookkeeper::{Bookkeeper, PositionKey};
use super::model::{AnnotatorError, DescKind, SomeInteger, SomeString, SomeValue, union};
use crate::flowspace::model::{ConstValue, Constant, HOST_ENV, HostObject};

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
    /// Class-level sources require the ClassDesc to be live (strong
    /// `upgrade`) and the `s_get_value` body which lives in c2. This
    /// returns [`AnnotatorError`] citing Phase 5 P5.2+ until c2 lands.
    pub fn s_get_value(
        &self,
        _classdef: Option<&Rc<RefCell<ClassDef>>>,
        _name: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        match self {
            AttrSource::Class(_) => Err(AnnotatorError::new(
                "AttrSource::Class::s_get_value requires ClassDesc.s_get_value \
                 (Phase 5 P5.2 classdesc c2 dep — see classdesc.py:784-802)",
            )),
            AttrSource::Instance(_) => Err(AnnotatorError::new(
                "AttrSource::Instance::s_get_value requires live instance \
                 attribute reflection (Phase 5 P5.2+ dep — see \
                 classdesc.py:442-451)",
            )),
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
/// carrier. The `s_get_value` / `all_instance_attributes` bodies call
/// into Python reflection that isn't available here; both surface as
/// [`AnnotatorError`] with a Phase 5 P5.2+ citation.
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
    /// Defers to live instance reflection; returns [`AnnotatorError`]
    /// citing the dep until c2/c3 wires instance-attribute reads.
    pub fn s_get_value(
        &self,
        _classdef: Option<&Rc<RefCell<ClassDef>>>,
        _name: &str,
    ) -> Result<SomeValue, AnnotatorError> {
        Err(AnnotatorError::new(
            "InstanceSource.s_get_value requires live instance attribute \
             reflection (Phase 5 P5.2+ dep — see classdesc.py:442-451)",
        ))
    }

    /// RPython `InstanceSource.all_instance_attributes(self)`
    /// (classdesc.py:453-464).
    ///
    /// Walks `obj.__dict__` + `tp.__mro__` `__slots__`. Deferred as
    /// above.
    pub fn all_instance_attributes(&self) -> Result<Vec<String>, AnnotatorError> {
        Err(AnnotatorError::new(
            "InstanceSource.all_instance_attributes requires live __dict__ / \
             __slots__ reflection (Phase 5 P5.2+ dep — see classdesc.py:453-464)",
        ))
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
    /// The `attr_allowed=False` path needs `bookkeeper.getattr_locations`,
    /// which is a Phase 5 P5.2+ bookkeeper-commit-2 dependency; when
    /// that branch fires we surface an [`AnnotatorError`] citing the
    /// missing dep rather than silently continuing.
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
// ClassDesc stub (classdesc.py:488-600 — data shell only; c2 fills body).
// ---------------------------------------------------------------------------

/// RPython `class ClassDesc(Desc)` (classdesc.py:488-600).
///
/// **c1 stub shell.** Carries only the minimum fields needed by
/// [`ClassDef::new`] + [`Attribute::validate`] so the structural port
/// of `ClassDef` can land. `ClassDesc::__init__` (classdesc.py:494-588)
/// body — mixin resolution, slots / _attrs_ detection,
/// `is_builtin_exception_class` pre-population, `_immutable_fields_`
/// enforcement — lands with c2. Downstream methods (`lookup`,
/// `read_attribute`, `s_read_attribute`, `s_get_value`,
/// `add_source_attribute`, `getclassdef`, `pycall`,
/// `consider_call_site`, ...) are not declared yet; c2 adds them.
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
    /// Back-reference to the bookkeeper. Weak to avoid cycles.
    pub bookkeeper: Weak<Bookkeeper>,
}

impl ClassDesc {
    /// RPython `ClassDesc.__init__` stub (classdesc.py:494-588).
    ///
    /// c1 skips the mixin / slots / exception detection so the
    /// structural port compiles. c2 replaces this with the full body.
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
        // model does not expose; the check stays deferred. Structural
        // shape preserved so c3 flips this into a full guard with a
        // single `add_method_attr_lookup` helper.
        if cls.class_get("_must_be_light_finalizer_").is_some() && cls.class_has("__del__") {
            // Deferred — c3 dep on method-level `_must_be_light_finalizer_`.
        }

        Ok(me)
    }

    /// RPython `ClassDesc.add_source_attribute(self, name, value,
    /// mixin=False)` (classdesc.py:590-634).
    ///
    /// Property / staticmethod / MemberDescriptor branches are
    /// structural ports deferred because [`HostObject`] does not yet
    /// carry those Python descriptor kinds; skips are annotated
    /// inline. FunctionType + mixin uses
    /// [`Bookkeeper::newfuncdesc`] to preserve mixin-specific
    /// FunctionDesc identity (classdesc.py:608-613).
    pub fn add_source_attribute(
        this: &Rc<RefCell<Self>>,
        name: &str,
        value: ConstValue,
        mixin: bool,
    ) -> Result<(), AnnotatorError> {
        // classdesc.py:591-602 — property branch. HostObject has no
        // property carrier; fall through to the default Constant
        // assignment. Structural parity lands when HostObject gains a
        // Property variant.

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

        // classdesc.py:619-622 — staticmethod + mixin. HostObject does
        // not model staticmethod today; stay with Constant storage.
        // classdesc.py:624-626 — MemberDescriptor skip. HostObject does
        // not emit MemberDescriptors so nothing to guard against.

        // classdesc.py:627-633 — __init__ on builtin exception class
        // routes through BUILTIN_ANALYZERS (builtin.py). Until builtin.py
        // lands, mirror upstream's "skip store" effect conservatively —
        // the BUILTIN_ANALYZERS registry would normally decide whether
        // to keep the assignment; an empty registry means no whitelist,
        // so upstream drops it.
        if name == "__init__" {
            let is_builtin_exc = this.borrow().is_builtin_exception_class();
            if is_builtin_exc {
                return Ok(());
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
    /// The `bookkeeper.emulate_pbc_call(__del__, …)` tail
    /// (classdesc.py:691-696) requires the call-family machinery ported
    /// in `binaryop.py` + `annrpython.py` driver. Presence of `__del__`
    /// on a classdict therefore surfaces as [`AnnotatorError`] so
    /// `__del__`-bearing classes abort until the dep lands.
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

        // classdesc.py:691-696 — __del__ emulate_pbc_call path.
        if this.borrow().classdict.contains_key("__del__") {
            return Err(AnnotatorError::new(
                "ClassDesc._init_classdef: __del__ triggers bookkeeper.emulate_pbc_call \
                 (Phase 5 P5.2+ binaryop.py + annrpython.py dep — see classdesc.py:691-696)",
            ));
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

    /// RPython `ClassDesc.getclassdef(self, key)` (classdesc.py:669-670).
    pub fn getclassdef(
        this: &Rc<RefCell<Self>>,
        _key: (),
    ) -> Result<Rc<RefCell<ClassDef>>, AnnotatorError> {
        Self::getuniqueclassdef(this)
    }

    /// RPython `ClassDesc.consider_call_site(descs, args, s_result, op)`
    /// (classdesc.py:853-902).
    ///
    /// Phase 1 (this port): `descs[0].getcallfamily(); descs[0].mergecallfamilies(*descs[1:])`
    /// — keeps the PBC call-family UnionFind consistent.
    ///
    /// Phase 2 (deferred) computes the `__init__` MethodDesc set via
    /// `desc.s_read_attribute('__init__')` and recurses into
    /// `MethodDesc.consider_call_site(initdescs, args, s_None, op)`.
    /// That hinges on `ClassDesc::s_read_attribute` + `getcommonbase`,
    /// neither ported yet — leaving the recursion as a TODO keeps the
    /// call-family side (phase 1) working and surfaces the missing
    /// piece explicitly rather than silently no-op'ing the entire
    /// branch.
    pub fn consider_call_site(
        descs: &[Rc<RefCell<ClassDesc>>],
        _args: &super::argument::ArgumentsForTranslation,
        _s_result: &super::model::SomeValue,
    ) -> Result<(), super::model::AnnotatorError> {
        use super::description::DescKey;
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
        let mut families = bk.pbc_maximal_call_families.borrow_mut();
        let mut rep = families.find_rep(head_key);
        for other in descs.iter().skip(1) {
            let other_key = DescKey::from_rc(other);
            let (_changed, new_rep) = families.union(rep, other_key);
            rep = new_rep;
        }
        drop(families);
        // Phase 2 (classdesc.py:856-902): __init__ initdescs recursion
        // deferred — requires ClassDesc::s_read_attribute +
        // getcommonbase (not yet ported). The phase-1 call-family
        // bookkeeping is sufficient to dedupe class-PBC call sites in
        // the family UnionFind; __init__ specialization lands with
        // a later commit.
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
    /// RPython `self.repr = None`. Populated by rtyper; stays `None`
    /// at annotator time.
    pub repr_ready: bool,
    /// RPython `self.extra_access_sets = {}`.
    ///
    /// Keyed on attribute name; values are opaque tags in c1 (real
    /// access-set machinery lands with classdesc c3's
    /// `getattrfamily`).
    pub extra_access_sets: HashMap<String, ()>,
    /// RPython `self.instances_seen = set()`. Identity hashes of the
    /// prebuilt instances already absorbed by `see_instance`.
    pub instances_seen: HashSet<usize>,
    /// RPython `self.basedef` — strong reference; the superclass owns
    /// the weak `subdefs` back-ref.
    pub basedef: Option<Rc<RefCell<ClassDef>>>,
    /// RPython `self.parentdefs = dict.fromkeys(self.getmro())`. Stored
    /// as identity pointers so `.contains` works with `Rc::as_ptr`.
    pub parentdefs: HashSet<usize>,
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
            let weak = base.borrow().classdef.clone();
            match weak.and_then(|w| w.upgrade()) {
                Some(cd) => cd,
                None => panic!(
                    "ClassDef::new requires basedesc.classdef to upgrade to a \
                     live ClassDef; ClassDesc.getuniqueclassdef lazy-init \
                     lands with classdesc c2 (classdesc.py:699-702)."
                ),
            }
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
            repr_ready: false,
            extra_access_sets: HashMap::new(),
            instances_seen: HashSet::new(),
            basedef: basedef.clone(),
            parentdefs: HashSet::new(),
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
    ///
    /// The `bookkeeper.update_attr` reflow at the tail is a Phase 5
    /// P5.2+ bookkeeper-c2 dependency; c1 preserves the new-Attribute
    /// installation + source merge.
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
    fn instance_source_defers_with_phase5_citation() {
        let bk = make_bk();
        let obj = HostObject::new_instance(HostObject::new_class("pkg.X", vec![]), vec![]);
        let src = InstanceSource::new(&bk, obj);
        let err = src.all_instance_attributes().unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("Phase 5 P5.2+"));
    }

    #[test]
    fn attr_source_s_get_value_defers() {
        let bk = make_bk();
        let obj = HostObject::new_instance(HostObject::new_class("pkg.X", vec![]), vec![]);
        let src = AttrSource::Instance(InstanceSource::new(&bk, obj));
        let err = src.s_get_value(None, "x").unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("live instance"));
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
    fn classdesc_add_source_attribute_skips_builtin_exc_init() {
        let bk = make_bk();
        let base_exc = HOST_ENV.lookup_builtin("BaseException").unwrap();
        let desc_rc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &bk,
            base_exc,
            "BaseException".into(),
        )));
        // __init__ on a builtin exception class is skipped (upstream
        // BUILTIN_ANALYZERS dep — defer to builtin.py port).
        ClassDesc::add_source_attribute(&desc_rc, "__init__", ConstValue::Int(0), false).unwrap();
        assert!(!desc_rc.borrow().classdict.contains_key("__init__"));
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
    fn classdesc_init_classdef_rejects_del_with_citation() {
        let bk = make_bk();
        let cls = HostObject::new_class("pkg.HasDel", vec![]);
        cls.class_set("__del__", ConstValue::Int(0));
        let desc_rc = ClassDesc::new(&bk, cls, None, None, None).unwrap();
        let err = ClassDesc::_init_classdef(&desc_rc).unwrap_err();
        assert!(err.msg.as_deref().unwrap().contains("emulate_pbc_call"));
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
        use crate::annotator::description::DescEntry;
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
        let a = make_classdesc(&bk, "pkg.A");
        let b = make_classdesc(&bk, "pkg.B");
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
        ClassDesc::consider_call_site(&[a.clone(), b.clone()], &args, &SomeValue::Impossible)
            .expect("phase-1 consider_call_site must succeed");
        // Post-condition: A and B share a CallFamily.
        let mut families = bk.pbc_maximal_call_families.borrow_mut();
        let rep_a = families.find_rep(a_key);
        let rep_b = families.find_rep(b_key);
        assert_eq!(rep_a, rep_b);
    }
}
