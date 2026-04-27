//! Port of `rpython/rtyper/extregistry.py`.
//!
//! Upstream provides two parallel registries that
//! `Bookkeeper.immutablevalue(x)` (`bookkeeper.py:312-314`) and
//! `signature.annotationoftype(t)` (`signature.py:98-100`) consult
//! before falling back to the generic callable / instance branches:
//!
//! * **Value-level** — `EXT_REGISTRY_BY_VALUE` keyed on the Python
//!   value itself. Driven by `class Entry(ExtRegistryEntry): _about_ =
//!   <value>` registrations (extregistry.py:11). [`is_registered`] /
//!   [`lookup`] expose this path; the bookkeeper consumes it.
//! * **Type-level** — `EXT_REGISTRY_BY_TYPE` keyed on a Python type
//!   object. Driven by `class Entry(ExtRegistryEntry): _type_ = <type>`
//!   registrations (extregistry.py:14). [`is_registered_type`] /
//!   [`lookup_type`] expose this path; the signature module consumes
//!   it.
//!
//! ## Rust language adaptation (parity rule #1)
//!
//! Upstream populates both registries lazily through the
//! `AutoRegisteringType` metaclass — every `class Entry(...)` triggers
//! `__init__` which inserts `cls` into the dict keyed on the
//! `_about_` / `_type_` slot. Rust has no metaclass; the
//! [`ExtRegistryEntry`] enum below carries one variant per upstream
//! `_about_` / `_type_` registration, and [`is_registered`] /
//! [`lookup`] match on the carrier shape directly. Adding a new
//! upstream entry means adding a variant + match arm here.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Mutex, OnceLock};

use crate::annotator::bookkeeper::Bookkeeper;
use crate::annotator::model::{AnnotatorError, SomeBuiltin, SomeValue};
use crate::flowspace::model::{ConstValue, HostObject};

use super::error::TyperError;
use super::lltypesystem::lltype;
use super::rbuiltin::BuiltinTyperFn;

/// One arm per upstream `class Entry(ExtRegistryEntry)` subclass that
/// the Rust port currently consults.
#[derive(Clone, Debug)]
pub enum ExtRegistryEntry {
    /// Upstream `lltype.py:1513-1518`:
    ///
    /// ```python
    /// class _ptrEntry(ExtRegistryEntry):
    ///     _type_ = _ptr
    ///     def compute_annotation(self):
    ///         from rpython.rtyper.llannotation import SomePtr
    ///         return SomePtr(typeOf(self.instance))
    /// ```
    ///
    /// The `Option<_ptr>` holds the value-level instance when the
    /// entry was constructed via [`lookup`]; `None` when constructed
    /// via [`lookup_type`] (matching upstream `Entry(tp)` /
    /// `Entry(type=tp, instance=None)` at extregistry.py:124-126).
    /// Upstream `_ptrEntry.compute_annotation` reads `self.instance`,
    /// so the type-level path raises `AssertionError` upstream — the
    /// Rust [`Self::compute_annotation`] surfaces the same failure as
    /// an [`AnnotatorError`].
    Ptr(Option<lltype::_ptr>),
    /// Generic HostObject value-level entry mirroring the base
    /// `ExtRegistryEntry.compute_annotation` implementation:
    /// construct `SomeBuiltin(analyser, methodname=...)` for the
    /// registered host callable.
    HostValueBuiltin {
        instance: HostObject,
        analyser_name: String,
        methodname: Option<String>,
    },
    /// HostObject type-level entry whose subclass override returns a
    /// fixed annotation. This mirrors `_type_ = T` entries that define
    /// `compute_annotation` instead of using the base builtin-function
    /// implementation.
    HostTypeAnnotation {
        type_obj: HostObject,
        instance: Option<HostObject>,
        annotation: RegisteredAnnotation,
    },
}

/// Small Send-able annotation payload for static HostObject type
/// registry entries. Add variants as concrete upstream `_type_`
/// entries land.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RegisteredAnnotation {
    Bool,
    Int,
    Str,
}

/// Cache-key identity for `SomeBuiltin.rtyper_makekey` after the
/// upstream `extregistry.lookup(const)` remap.
///
/// RPython `ExtRegistryEntry.__hash__` / `__eq__`
/// (extregistry.py:43-48) key on `(self.__class__, self.type,
/// self.instance)`. Rust has one enum variant per entry subclass, so
/// the variant tag supplies `self.__class__`; these fields carry the
/// type/instance identities.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExtRegistryEntryKey {
    Ptr {
        instance_identity: Option<u64>,
    },
    HostValueBuiltin {
        instance_identity: usize,
    },
    HostTypeAnnotation {
        type_identity: usize,
        instance_identity: Option<usize>,
    },
}

impl ExtRegistryEntry {
    pub fn makekey(&self) -> ExtRegistryEntryKey {
        match self {
            ExtRegistryEntry::Ptr(instance) => ExtRegistryEntryKey::Ptr {
                instance_identity: instance.as_ref().map(lltype::_ptr::_hashable_identity),
            },
            ExtRegistryEntry::HostValueBuiltin { instance, .. } => {
                ExtRegistryEntryKey::HostValueBuiltin {
                    instance_identity: instance.identity_id(),
                }
            }
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj, instance, ..
            } => ExtRegistryEntryKey::HostTypeAnnotation {
                type_identity: type_obj.identity_id(),
                instance_identity: instance.as_ref().map(HostObject::identity_id),
            },
        }
    }

    /// RPython `ExtRegistryEntry.compute_annotation_bk(self, bk)`
    /// (extregistry.py:54-56). Stashes the bookkeeper on `self` and
    /// dispatches to `compute_annotation`. The Rust port forwards the
    /// bookkeeper into `compute_annotation` directly via the same `&Rc`
    /// — no per-entry mutable state needed.
    pub fn compute_annotation_bk(
        &self,
        _bookkeeper: &Rc<Bookkeeper>,
    ) -> Result<SomeValue, AnnotatorError> {
        self.compute_annotation()
    }

    /// RPython `ExtRegistryEntry.compute_annotation(self)`
    /// (extregistry.py:58-67). Subclass override per variant.
    pub fn compute_annotation(&self) -> Result<SomeValue, AnnotatorError> {
        match self {
            // upstream lltype.py:1517-1518:
            //     def compute_annotation(self):
            //         from rpython.rtyper.llannotation import SomePtr
            //         return SomePtr(typeOf(self.instance))
            ExtRegistryEntry::Ptr(Some(instance)) => Ok(super::llannotation::lltype_to_annotation(
                lltype::typeOf(instance),
            )),
            // upstream `_ptrEntry.compute_annotation` reads
            // `self.instance`; with `instance=None` (type-level path)
            // it raises AttributeError. Surface the same failure here.
            ExtRegistryEntry::Ptr(None) => Err(AnnotatorError::new(
                "ExtRegistryEntry::Ptr.compute_annotation: type-level lookup has no instance \
                 (upstream _ptrEntry.compute_annotation requires self.instance)",
            )),
            ExtRegistryEntry::HostValueBuiltin {
                instance,
                analyser_name,
                methodname,
            } => {
                let mut result = SomeBuiltin::new(
                    analyser_name.clone(),
                    None,
                    methodname
                        .clone()
                        .or_else(|| Some(instance.qualname().to_string())),
                );
                result.base.const_box = Some(crate::flowspace::model::Constant::new(
                    ConstValue::HostObject(instance.clone()),
                ));
                Ok(SomeValue::Builtin(result))
            }
            ExtRegistryEntry::HostTypeAnnotation { annotation, .. } => match annotation {
                RegisteredAnnotation::Bool => Ok(SomeValue::Bool(Default::default())),
                RegisteredAnnotation::Int => Ok(SomeValue::Integer(Default::default())),
                RegisteredAnnotation::Str => Ok(SomeValue::String(Default::default())),
            },
        }
    }

    /// rbuiltin.py:81 `entry.specialize_call` attribute lookup.
    ///
    /// Upstream `ExtRegistryEntry` (extregistry.py:33-72) does not
    /// define a base `specialize_call` method; subclasses that override
    /// it return a typer callable that `findbltintyper` returns
    /// directly, while subclasses that do not override it raise
    /// `AttributeError` at the attribute lookup site. The Rust enum
    /// mirrors per-variant: each arm whose upstream subclass overrides
    /// `specialize_call` returns `Ok(typer_fn)`; arms whose upstream
    /// subclass does not override it surface the AttributeError as a
    /// `TyperError` so the rtyper fails closed at the same point.
    pub fn specialize_call(&self) -> Result<BuiltinTyperFn, TyperError> {
        match self {
            // lltype.py:1513-1518 `_ptrEntry(ExtRegistryEntry)` defines
            // only `_type_` and `compute_annotation` — no
            // `specialize_call`. Upstream raises AttributeError at
            // `entry.specialize_call`.
            ExtRegistryEntry::Ptr(_) => Err(TyperError::message(
                "'ExtRegistryEntry' object has no attribute 'specialize_call'",
            )),
            ExtRegistryEntry::HostValueBuiltin { .. }
            | ExtRegistryEntry::HostTypeAnnotation { .. } => Err(TyperError::message(
                "'ExtRegistryEntry' object has no attribute 'specialize_call'",
            )),
        }
    }
}

// RPython stores these as EXT_REGISTRY_BY_VALUE / EXT_REGISTRY_BY_TYPE dicts
// (extregistry.py:115-116). HashMap is therefore the literal Rust analogue,
// not a side table invented for convenience.
static HOST_VALUE_REGISTRY: OnceLock<Mutex<HashMap<HostObject, ExtRegistryEntry>>> =
    OnceLock::new();
static HOST_TYPE_REGISTRY: OnceLock<Mutex<HashMap<HostObject, ExtRegistryEntry>>> = OnceLock::new();

fn host_value_registry() -> &'static Mutex<HashMap<HostObject, ExtRegistryEntry>> {
    HOST_VALUE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn host_type_registry() -> &'static Mutex<HashMap<HostObject, ExtRegistryEntry>> {
    HOST_TYPE_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Rust equivalent of `AutoRegisteringType._register_value`.
pub fn register_host_value(host: HostObject, entry: ExtRegistryEntry) -> Result<(), TyperError> {
    let mut registry = host_value_registry().lock().unwrap();
    if registry.contains_key(&host) {
        return Err(TyperError::message(format!(
            "duplicate extregistry value registration for {}",
            host.qualname()
        )));
    }
    registry.insert(host, entry);
    Ok(())
}

/// Rust equivalent of `AutoRegisteringType._register_type`.
pub fn register_host_type(
    host_type: HostObject,
    entry: ExtRegistryEntry,
) -> Result<(), TyperError> {
    let mut registry = host_type_registry().lock().unwrap();
    if registry.contains_key(&host_type) {
        return Err(TyperError::message(format!(
            "duplicate extregistry type registration for {}",
            host_type.qualname()
        )));
    }
    registry.insert(host_type, entry);
    Ok(())
}

fn bind_lookup_instance(entry: ExtRegistryEntry, instance: &HostObject) -> ExtRegistryEntry {
    match entry {
        ExtRegistryEntry::HostTypeAnnotation {
            type_obj,
            annotation,
            ..
        } => ExtRegistryEntry::HostTypeAnnotation {
            type_obj,
            instance: Some(instance.clone()),
            annotation,
        },
        other => other,
    }
}

fn lookup_host_object(host: &HostObject) -> Option<ExtRegistryEntry> {
    if let Some(entry) = host_value_registry().lock().unwrap().get(host).cloned() {
        return Some(entry);
    }
    let cls = host.instance_class()?;
    host_type_registry()
        .lock()
        .unwrap()
        .get(cls)
        .cloned()
        .map(|entry| bind_lookup_instance(entry, host))
}

// ---------------------------------------------------------------------------
// Value-level registry — RPython `EXT_REGISTRY_BY_VALUE` (extregistry.py:115).
// ---------------------------------------------------------------------------

/// RPython `extregistry.is_registered(instance)` (extregistry.py:141-146).
///
/// Upstream:
///
/// ```python
/// def _lookup_cls(instance):
///     try:
///         return EXT_REGISTRY_BY_VALUE[instance]
///     except (KeyError, TypeError):
///         return _lookup_type_cls(type(instance))
///
/// def is_registered(instance):
///     try:
///         _lookup_cls(instance)
///     except KeyError:
///         return False
///     return True
/// ```
///
/// Upstream falls back to type-level lookup when the value-level
/// dictionary misses; the Rust port mirrors that for HostObject
/// instances via [`HostObject::instance_class`]. `LLPtr` matches the
/// upstream `_ptrEntry._type_=_ptr` fallback (every `_ptr` value's
/// `type(instance)` is `_ptr`).
pub fn is_registered(instance: &ConstValue) -> bool {
    match instance {
        ConstValue::LLPtr(_) => true,
        ConstValue::HostObject(host) => lookup_host_object(host).is_some(),
        _ => false,
    }
}

/// RPython `extregistry.lookup(instance)` (extregistry.py:137-139).
///
/// Upstream:
///
/// ```python
/// def lookup(instance):
///     Entry = _lookup_cls(instance)
///     return Entry(type(instance), instance)
/// ```
///
/// The Rust port constructs the [`ExtRegistryEntry`] variant
/// corresponding to the carrier shape, with the value-level instance
/// embedded so `compute_annotation` can read it.
pub fn lookup(instance: &ConstValue) -> Option<ExtRegistryEntry> {
    match instance {
        ConstValue::LLPtr(ptr) => Some(ExtRegistryEntry::Ptr(Some((**ptr).clone()))),
        ConstValue::HostObject(host) => lookup_host_object(host),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Type-level registry — RPython `EXT_REGISTRY_BY_TYPE` (extregistry.py:116).
// ---------------------------------------------------------------------------

/// RPython `extregistry.is_registered_type(tp)` (extregistry.py:128-129).
///
/// Upstream consults `EXT_REGISTRY_BY_TYPE` directly. The Rust port
/// first checks the HostObject registry dict populated by
/// [`register_host_type`].
pub fn is_registered_type(host: &HostObject) -> bool {
    host_type_registry().lock().unwrap().contains_key(host)
}

/// RPython `extregistry.lookup_type(tp)` (extregistry.py:124-126).
///
/// Upstream:
///
/// ```python
/// def lookup_type(tp):
///     Entry = _lookup_type_cls(tp)
///     return Entry(tp)  # instance=None
/// ```
///
/// Constructed entries carry `instance=None` so subclasses whose
/// `compute_annotation` reads `self.instance` (e.g. `_ptrEntry`) raise
/// the same AttributeError upstream produces. The Rust port encodes
/// this via the `Option<_ptr>` slot on [`ExtRegistryEntry::Ptr`].
pub fn lookup_type(host: &HostObject) -> Option<ExtRegistryEntry> {
    host_type_registry().lock().unwrap().get(host).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::ConstValue;
    use crate::translator::rtyper::lltypesystem::lltype::{RUNTIME_TYPE_INFO, nullptr};

    fn sample_ptr() -> crate::translator::rtyper::lltypesystem::lltype::_ptr {
        nullptr(RUNTIME_TYPE_INFO.clone()).expect("RUNTIME_TYPE_INFO is a container type")
    }

    /// upstream extregistry.py:141-146 — `is_registered` returns True
    /// for any value whose type is registered (here, `_ptr`).
    #[test]
    fn is_registered_matches_llptr_value() {
        let ptr = sample_ptr();
        let cv = ConstValue::LLPtr(Box::new(ptr));
        assert!(is_registered(&cv));
    }

    /// `lookup` returns the matching entry with the value-level
    /// instance embedded (upstream `Entry(type, instance)` construction).
    #[test]
    fn lookup_returns_ptr_entry_with_instance() {
        let ptr = sample_ptr();
        let cv = ConstValue::LLPtr(Box::new(ptr));
        let entry = lookup(&cv).expect("LLPtr must surface as Ptr entry");
        match entry {
            ExtRegistryEntry::Ptr(Some(_)) => {}
            ExtRegistryEntry::Ptr(None) => panic!("value-level lookup must carry instance"),
            other => panic!("LLPtr lookup returned wrong entry: {other:?}"),
        }
    }

    /// Unregistered HostObject values still miss, matching upstream
    /// `KeyError` from both value and type registries.
    #[test]
    fn is_registered_misses_unregistered_hostobject() {
        let host = HostObject::new_class("UserClass", vec![]);
        let cv = ConstValue::HostObject(host);
        assert!(!is_registered(&cv));
        assert!(lookup(&cv).is_none());
    }

    /// Type-level lookup returns None for unregistered host classes.
    #[test]
    fn is_registered_type_misses_unregistered_host_class() {
        let host = HostObject::new_class("UserClass", vec![]);
        assert!(!is_registered_type(&host));
        assert!(lookup_type(&host).is_none());
    }

    #[test]
    fn host_value_registry_hits_registered_hostobject() {
        let host = HostObject::new_builtin_callable("pkg.extregistry.value");
        register_host_value(
            host.clone(),
            ExtRegistryEntry::HostValueBuiltin {
                instance: host.clone(),
                analyser_name: "pkg.extregistry.value".into(),
                methodname: Some("value".into()),
            },
        )
        .expect("register host value");
        let cv = ConstValue::HostObject(host);
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("registered HostObject value must lookup");
        let s = entry
            .compute_annotation()
            .expect("registered host builtin computes annotation");
        assert!(matches!(s, SomeValue::Builtin(_)));
    }

    #[test]
    fn host_type_registry_hits_registered_host_class() {
        let host = HostObject::new_class("pkg.extregistry.Type", vec![]);
        register_host_type(
            host.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: host.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Int,
            },
        )
        .expect("register host type");
        assert!(is_registered_type(&host));
        let entry = lookup_type(&host).expect("registered HostObject type must lookup");
        let s = entry
            .compute_annotation()
            .expect("registered host type computes annotation");
        assert!(matches!(s, SomeValue::Integer(_)));
    }

    #[test]
    fn host_value_registry_rejects_duplicate_registration() {
        let host = HostObject::new_builtin_callable("pkg.extregistry.duplicate_value");
        register_host_value(
            host.clone(),
            ExtRegistryEntry::HostValueBuiltin {
                instance: host.clone(),
                analyser_name: "first".into(),
                methodname: None,
            },
        )
        .expect("first registration must succeed");

        let err = register_host_value(
            host.clone(),
            ExtRegistryEntry::HostValueBuiltin {
                instance: host.clone(),
                analyser_name: "second".into(),
                methodname: None,
            },
        )
        .expect_err("duplicate _about_ registration must fail like RPython");
        assert!(format!("{err:?}").contains("duplicate extregistry value registration"));
    }

    #[test]
    fn host_type_registry_rejects_duplicate_registration() {
        let host = HostObject::new_class("pkg.extregistry.DuplicateType", vec![]);
        register_host_type(
            host.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: host.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Int,
            },
        )
        .expect("first registration must succeed");

        let err = register_host_type(
            host.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: host.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Bool,
            },
        )
        .expect_err("duplicate _type_ registration must fail like RPython");
        assert!(format!("{err:?}").contains("duplicate extregistry type registration"));
    }

    #[test]
    fn host_value_lookup_falls_back_to_registered_instance_class() {
        let cls = HostObject::new_class("pkg.extregistry.FallbackType", vec![]);
        register_host_type(
            cls.clone(),
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj: cls.clone(),
                instance: None,
                annotation: RegisteredAnnotation::Str,
            },
        )
        .expect("register host type");

        let inst = HostObject::new_instance(cls.clone(), vec![]);
        let cv = ConstValue::HostObject(inst.clone());
        assert!(is_registered(&cv));
        let entry = lookup(&cv).expect("registered instance class must lookup");
        match entry {
            ExtRegistryEntry::HostTypeAnnotation {
                type_obj,
                instance,
                annotation: RegisteredAnnotation::Str,
            } => {
                assert_eq!(type_obj.identity_id(), cls.identity_id());
                assert_eq!(
                    instance.as_ref().map(HostObject::identity_id),
                    Some(inst.identity_id())
                );
            }
            other => panic!("wrong fallback entry: {other:?}"),
        }
    }

    /// Type-level entry construction carries `instance=None` upstream;
    /// the Rust port's `Ptr(None)` mirrors this. `compute_annotation`
    /// surfaces AnnotatorError because `_ptrEntry` reads `self.instance`.
    #[test]
    fn ptr_entry_without_instance_errors_at_compute_annotation() {
        let entry = ExtRegistryEntry::Ptr(None);
        let err = entry
            .compute_annotation()
            .expect_err("type-level _ptrEntry must error without instance");
        assert!(err.to_string().contains("requires self.instance"));
    }
}
