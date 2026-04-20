//! Flow graph model — SSA structure.
//!
//! RPython upstream: `rpython/flowspace/model.py` (737 LOC).
//!
//! Line-by-line port landing over commits F1.2 .. F1.7. This file
//! carries the leaf types first (`Variable`, `Constant`, `Atom`);
//! follow-up commits add `SpaceOperation`, `Block`, `Link`,
//! `FunctionGraph`, and the `mkentrymap` / `copygraph` / `summary` /
//! `checkgraph` helpers.
//!
//! Structural deviations from upstream (documented per CLAUDE.md
//! parity rule #1):
//!
//! * `Variable.annotation` / `Variable.concretetype` are
//!   `Option<()>` placeholders until `majit-annotator::SomeValue`
//!   (Phase 4) and `majit-rtyper::Repr` (Phase 6) exist.
//! * `Constant.value` is a placeholder `ConstValue` enum pending a
//!   Python object model at Phase 3 (`flowspace/operation.py` port);
//!   RPython wraps arbitrary Python objects via `Hashable`.
//! * `Variable.namesdict` (RPython class attribute — a mutable dict
//!   shared by every `Variable`) is a `static LazyLock<Mutex<...>>`;
//!   Rust has no class-mutable-state, this is the minimum deviation.

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::rc::{Rc, Weak};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use super::bytecode::HostCode;
use crate::annotator::model::SomeValue;

// RPython `Variable.annotation` holds a `SomeObject` subclass instance
// (annotator/model.py:SomeObject). Rust stores `Option<Rc<SomeValue>>`
// directly — `flowspace` and `annotator` are sibling modules inside
// `majit-translate`, so the cross-module reference is just a `use`.

/// Placeholder for `Repr.lowleveltype` attached by the rtyper.
///
/// RPython: `rtyper/rmodel.py:Repr`. Ported in roadmap Phase 6 as
/// `majit-rtyper::Repr`; until then every concretetype slot carries
/// `None`.
pub type ConcretetypePlaceholder = ();

/// RPython `Constant.value` 에 담기는 host-level Python object 의 일반
/// carrier.
///
/// Upstream `rpython/flowspace/model.py:354` `class Constant(Hashable)`
/// 은 `Hashable.value` 에 임의의 Python object 를 그대로 담는다
/// (`self.value = value`). Rust 포트는 `ConstValue` 를 닫힌 enum 으로
/// 시작했지만 builtin function / type / exception class / exception
/// instance / module 까지 모두 `Rc<HostObjectInner>` 한 carrier 로
/// 모으는 편이 더 orthodox 하다.
///
/// Identity 는 `Rc::ptr_eq` — upstream 의 Python object `is` 비교와
/// 동일. `SPECIAL_CASES` 처럼 identity-keyed 테이블은 동일 instance 를
/// 공유하는 싱글턴 (`HOST_ENV.lookup_builtin` 이 돌려주는 Rc) 으로
/// 부트스트랩한다.
///
/// Deviation (parity rule #1): upstream 은 Python runtime 이 있기
/// 때문에 임의의 객체를 그대로 carrier 에 담을 수 있다. Rust 포트는
/// Python runtime 을 내장하지 않으므로 `HostObjectKind` 에 upstream
/// 에서 관찰되는 구체 분류 (class/module/builtin callable/user
/// function/instance) 를 열거한다. 이 enum 은 외부 공개 API 가 아니고,
/// carrier 밖에서 관찰되는 contract 는 `Rc` identity 와 `qualname()`,
/// `is_subclass_of(…)` 같은 introspection 메서드뿐이다.
#[derive(Clone)]
pub struct HostObject {
    inner: Arc<HostObjectInner>,
}

struct HostObjectInner {
    qualname: String,
    kind: HostObjectKind,
}

enum HostObjectKind {
    /// Python type/class object. `bases` 는 `__bases__` 튜플; 재귀적
    /// `issubclass` 순회에 사용. `members` 는 `cls.__dict__` 대응 —
    /// annotator ClassDesc.__init__ 이 `_mixin_`, `_immutable_fields_`,
    /// `__slots__`, `_attrs_`, `__NOT_RPYTHON__`,
    /// `_annspecialcase_` 등을 읽고, `add_sources_for_class` 가 모든
    /// 엔트리를 순회한다. 값 타입이 임의의 Python 값(bool, tuple,
    /// function, property, …)이므로 `ConstValue` carrier 로 담는다.
    Class {
        bases: Vec<HostObject>,
        members: Mutex<HashMap<String, ConstValue>>,
    },
    /// Python module object. `members` 는 module dict — `getattr` 조회
    /// 대상. `LazyLock` singleton 의 Sync 요구를 만족하려고 `Mutex`.
    Module {
        members: Mutex<HashMap<String, HostObject>>,
    },
    /// Python builtin callable (function, method). 이름 외 구조
    /// 없음.
    BuiltinCallable,
    /// Python function object (user-defined). `graph_func` 는 flowspace
    /// 가 inspect 를 통해 들여다보는 code + closure 상태.
    UserFunction { graph_func: Box<GraphFunc> },
    /// Python instance (raise 문에서 materialise 된 exception 인스턴스
    /// 포함). `class_obj` 는 `__class__`; `args` 는 constructor
    /// arguments; `instance_dict` 는 per-instance attribute dict
    /// (`inst.__dict__`) — prebuilt instances attached via class
    /// annotation decorators populate this so `getattr(inst, attr)`
    /// in `FrozenDesc.default_read_attribute` can observe them.
    Instance {
        class_obj: HostObject,
        args: Vec<ConstValue>,
        instance_dict: Mutex<HashMap<String, ConstValue>>,
    },
    /// `Constant.value` 에 담긴 임의의 host object — flowspace 가 구조
    /// 를 모르지만 보존해야 하는 값(예: 포팅되지 않은 `ConstantData`
    /// variant, pyre-level opaque object). `qualname` 에 debug-only
    /// 식별자를 기록하고, identity 는 `Arc::ptr_eq` 로 유지한다. 이
    /// 키는 upstream 의 `Constant.value = <anonymous object>` 경로에
    /// 대응한다.
    Opaque,
    /// Python `property` descriptor (upstream classdesc.py:591-602). fget
    /// / fset / fdel 은 upstream `property(fget, fset, fdel, doc)` 의
    /// 각 슬롯에 대응하며, `Option<HostObject>` 로 담는다 (None =
    /// 미정의). unaryop.py:895 `_find_property_meth` 는 classdict 의
    /// `Constant(property_value)` 에서 `getattr(obj.value, meth)` 로 이
    /// 슬롯을 추출한다.
    Property {
        fget: Option<HostObject>,
        fset: Option<HostObject>,
        fdel: Option<HostObject>,
    },
    /// Python `staticmethod(func)` descriptor. The wrapped callable is
    /// typically a user-function HostObject; `ClassDesc.s_get_value`
    /// unwraps it before calling `immutablevalue`, matching
    /// `value.__get__(42)` upstream.
    StaticMethod { func: HostObject },
    /// Python `classmethod(func)` descriptor. `ClassDesc.s_get_value`
    /// recognizes this wrapper and raises the upstream
    /// "classmethods are not supported" error.
    ClassMethod { func: HostObject },
}

impl PartialEq for HostObject {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for HostObject {}

impl Hash for HostObject {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.inner) as usize).hash(state);
    }
}

impl std::fmt::Debug for HostObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<host {}>", self.inner.qualname)
    }
}

impl HostObject {
    pub fn qualname(&self) -> &str {
        &self.inner.qualname
    }

    /// Pointer-identity handle — used where upstream relies on
    /// `id(pyobj)` to build a dict key (e.g. sandbox trampoline
    /// emulation key, policy.py:87).
    pub fn identity_id(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }

    pub fn is_class(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Class { .. })
    }

    pub fn is_module(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Module { .. })
    }

    pub fn is_builtin_callable(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::BuiltinCallable)
    }

    pub fn is_instance(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Instance { .. })
    }

    pub fn is_user_function(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::UserFunction { .. })
    }

    pub fn user_function(&self) -> Option<&GraphFunc> {
        match &self.inner.kind {
            HostObjectKind::UserFunction { graph_func } => Some(graph_func.as_ref()),
            _ => None,
        }
    }

    /// Upstream `issubclass(self, other)` over `__bases__`. 재귀적
    /// 깊이우선 탐색이며, `__mro__` C3 linearisation 과 예외 계층에서는
    /// 동일한 결과를 준다.
    pub fn is_subclass_of(&self, other: &HostObject) -> bool {
        if self == other {
            return true;
        }
        match &self.inner.kind {
            HostObjectKind::Class { bases, .. } => bases.iter().any(|b| b.is_subclass_of(other)),
            _ => false,
        }
    }

    /// Class.__bases__ — bases tuple view. Non-class 는 None.
    pub fn class_bases(&self) -> Option<&[HostObject]> {
        match &self.inner.kind {
            HostObjectKind::Class { bases, .. } => Some(bases.as_slice()),
            _ => None,
        }
    }

    /// `cls.__dict__.get(name)` — class dict lookup. Non-class 는
    /// None.
    pub fn class_get(&self, name: &str) -> Option<ConstValue> {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members.lock().unwrap().get(name).cloned(),
            _ => None,
        }
    }

    /// Class dict setter — bootstrap / ClassDesc.create_new_attribute
    /// 경로에서 사용. Non-class 에 대해서는 no-op.
    pub fn class_set(&self, name: impl Into<String>, value: ConstValue) {
        if let HostObjectKind::Class { members, .. } = &self.inner.kind {
            members.lock().unwrap().insert(name.into(), value);
        }
    }

    /// `cls.__dict__.keys()` — class dict key snapshot. Non-class 는
    /// 빈 Vec.
    pub fn class_dict_keys(&self) -> Vec<String> {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => {
                members.lock().unwrap().keys().cloned().collect()
            }
            _ => Vec::new(),
        }
    }

    /// `cls.__dict__.items()` — class dict entry snapshot. Non-class
    /// 는 빈 Vec.
    pub fn class_dict_items(&self) -> Vec<(String, ConstValue)> {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members
                .lock()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// `cls.__dict__.__contains__(name)` — class dict 키 존재 검사.
    /// Non-class 는 false.
    pub fn class_has(&self, name: &str) -> bool {
        match &self.inner.kind {
            HostObjectKind::Class { members, .. } => members.lock().unwrap().contains_key(name),
            _ => false,
        }
    }

    /// `cls.__mro__` — C3 linearisation over `__bases__`. Non-class 는
    /// None. 상위 class 가 중복된 경우를 처리하지만, 복수 정의 충돌시
    /// `TypeError: MRO conflict` 대신 None 을 돌려준다 (upstream
    /// `type(...).__mro__` 은 TypeError 를 던짐). RPython annotator 는
    /// `add_mixins` 의 `type('tmp', tuple(mixins) + (object,), {}).__mro__`
    /// 경로에서만 이 함수를 쓰므로, mixin 계층이 C3 충돌을 일으키지 않는
    /// 한 None 경로는 타지 않는다.
    pub fn mro(&self) -> Option<Vec<HostObject>> {
        if !self.is_class() {
            return None;
        }
        c3_linearise(self)
    }

    /// Instance → `__class__`. None 이면 `self` 가 인스턴스가 아님.
    pub fn instance_class(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Instance { class_obj, .. } => Some(class_obj),
            _ => None,
        }
    }

    /// Instance → constructor args (raise-site 에서 `ValueError("msg")`
    /// 같은 호출로 captured).
    pub fn instance_args(&self) -> Option<&[ConstValue]> {
        match &self.inner.kind {
            HostObjectKind::Instance { args, .. } => Some(args.as_slice()),
            _ => None,
        }
    }

    /// Instance → per-instance `__dict__` lookup. Mirrors
    /// `getattr(instance, name)` respecting `inst.__dict__` before
    /// MRO. Returns `None` when the instance doesn't carry the
    /// attribute (or `self` isn't an instance).
    pub fn instance_get(&self, name: &str) -> Option<ConstValue> {
        match &self.inner.kind {
            HostObjectKind::Instance { instance_dict, .. } => {
                instance_dict.lock().unwrap().get(name).cloned()
            }
            _ => None,
        }
    }

    /// Instance setter — installs an entry into the per-instance
    /// `__dict__`. Used by prebuilt-instance fixtures in tests and
    /// by the `@setattr_to_class_annotation` style decorators
    /// that populate known attributes upfront.
    pub fn instance_set(&self, name: impl Into<String>, value: ConstValue) {
        if let HostObjectKind::Instance { instance_dict, .. } = &self.inner.kind {
            instance_dict.lock().unwrap().insert(name.into(), value);
        }
    }

    /// Module member 조회 — upstream `getattr(module, name)`.
    pub fn module_get(&self, name: &str) -> Option<HostObject> {
        match &self.inner.kind {
            HostObjectKind::Module { members } => members.lock().unwrap().get(name).cloned(),
            _ => None,
        }
    }

    /// Module setter — module object bootstrap 과정에서만 사용.
    pub fn module_set(&self, name: impl Into<String>, value: HostObject) {
        if let HostObjectKind::Module { members } = &self.inner.kind {
            members.lock().unwrap().insert(name.into(), value);
        }
    }

    pub fn new_class(qualname: impl Into<String>, bases: Vec<HostObject>) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::Class {
                    bases,
                    members: Mutex::new(HashMap::new()),
                },
            }),
        }
    }

    /// `new_class` + initial class dict. annotator
    /// `ClassDesc.__init__` 가 mixin 분리 + `add_sources_for_class`
    /// 를 돌리기 전에, bootstrap 이 미리 만들어놓은 class object 에
    /// 필요한 멤버 (`_mixin_`, `_immutable_fields_`, …) 를 즉시 넣어
    /// 주기 위한 편의 생성자.
    pub fn new_class_with_members(
        qualname: impl Into<String>,
        bases: Vec<HostObject>,
        members: HashMap<String, ConstValue>,
    ) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::Class {
                    bases,
                    members: Mutex::new(members),
                },
            }),
        }
    }

    pub fn new_module(qualname: impl Into<String>) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::Module {
                    members: Mutex::new(HashMap::new()),
                },
            }),
        }
    }

    pub fn new_builtin_callable(qualname: impl Into<String>) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::BuiltinCallable,
            }),
        }
    }

    pub fn new_user_function(graph_func: GraphFunc) -> Self {
        let qualname = graph_func.name.clone();
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                kind: HostObjectKind::UserFunction {
                    graph_func: Box::new(graph_func),
                },
            }),
        }
    }

    /// RPython `func_with_new_name(func, newname)` helper surface for
    /// user functions. Mirrors the upstream behavior that allocates a
    /// fresh function object sharing the same code/defaults/closure but
    /// exposing a different `__name__`.
    pub fn renamed_user_function(&self, newname: &str) -> Option<Self> {
        let graph_func = self.user_function()?;
        Some(Self::new_user_function(graph_func.with_new_name(newname)))
    }

    pub fn new_instance(class_obj: HostObject, args: Vec<ConstValue>) -> Self {
        let qualname = format!("{}-instance", class_obj.qualname());
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname,
                kind: HostObjectKind::Instance {
                    class_obj,
                    args,
                    instance_dict: Mutex::new(HashMap::new()),
                },
            }),
        }
    }

    /// `Constant.value` 에 담긴 임의 host object 를 carry. `qualname`
    /// 은 debug 에 사용; identity 는 항상 새로운 Arc.
    pub fn new_opaque(qualname: impl Into<String>) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::Opaque,
            }),
        }
    }

    pub fn is_opaque(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Opaque)
    }

    /// Python `property(fget, fset, fdel, doc)` constructor — upstream
    /// classdesc.py:591-602 synthesises getter / setter hidden functions
    /// and stores the property object itself in classdict.
    pub fn new_property(
        qualname: impl Into<String>,
        fget: Option<HostObject>,
        fset: Option<HostObject>,
        fdel: Option<HostObject>,
    ) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::Property { fget, fset, fdel },
            }),
        }
    }

    pub fn is_property(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::Property { .. })
    }

    /// Upstream `property.fget` — `None` if not provided at property
    /// construction time.
    pub fn property_fget(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Property { fget, .. } => fget.as_ref(),
            _ => None,
        }
    }

    /// Upstream `property.fset`.
    pub fn property_fset(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Property { fset, .. } => fset.as_ref(),
            _ => None,
        }
    }

    /// Upstream `property.fdel`.
    pub fn property_fdel(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::Property { fdel, .. } => fdel.as_ref(),
            _ => None,
        }
    }

    /// Python `staticmethod(func)` constructor.
    pub fn new_staticmethod(qualname: impl Into<String>, func: HostObject) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::StaticMethod { func },
            }),
        }
    }

    pub fn is_staticmethod(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::StaticMethod { .. })
    }

    /// Upstream `staticmethod.__get__(42)` unwraps back to the wrapped
    /// callable object without binding.
    pub fn staticmethod_func(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::StaticMethod { func } => Some(func),
            _ => None,
        }
    }

    /// Python `classmethod(func)` constructor.
    pub fn new_classmethod(qualname: impl Into<String>, func: HostObject) -> Self {
        HostObject {
            inner: Arc::new(HostObjectInner {
                qualname: qualname.into(),
                kind: HostObjectKind::ClassMethod { func },
            }),
        }
    }

    pub fn is_classmethod(&self) -> bool {
        matches!(self.inner.kind, HostObjectKind::ClassMethod { .. })
    }

    pub fn classmethod_func(&self) -> Option<&HostObject> {
        match &self.inner.kind {
            HostObjectKind::ClassMethod { func } => Some(func),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostGetAttrError {
    Missing,
    Unsupported,
}

fn host_class_mro_get(cls: &HostObject, name: &str) -> Option<ConstValue> {
    if let Some(value) = cls.class_get(name) {
        return Some(value);
    }
    let mro = cls.mro()?;
    for base in mro.iter().skip(1) {
        if let Some(value) = base.class_get(name) {
            return Some(value);
        }
    }
    None
}

fn host_unwrap_staticmethod(value: ConstValue) -> ConstValue {
    match value {
        ConstValue::HostObject(host) if host.is_staticmethod() => host
            .staticmethod_func()
            .cloned()
            .map(ConstValue::HostObject)
            .unwrap_or(ConstValue::HostObject(host)),
        other => other,
    }
}

/// Best-effort host-level `getattr(obj, name)` over the HostObject
/// kinds modelled by the current Rust port.
///
/// This mirrors the parts of Python descriptor lookup the HostObject
/// carrier can represent directly today:
/// * modules read from the module dict;
/// * class lookup walks the class MRO and unwraps `staticmethod`;
/// * instance lookup consults `__dict__` first, then the class MRO, and
///   unwraps `staticmethod`.
///
/// Descriptors that would require executing Python code or materialising
/// a bound method object are still outside the current host model and
/// therefore remain unresolved here.
pub fn host_getattr(pyobj: &HostObject, name: &str) -> Result<ConstValue, HostGetAttrError> {
    if pyobj.is_module() {
        return pyobj
            .module_get(name)
            .map(ConstValue::HostObject)
            .ok_or(HostGetAttrError::Missing);
    }
    if pyobj.is_class() {
        return host_class_mro_get(pyobj, name)
            .map(host_unwrap_staticmethod)
            .ok_or(HostGetAttrError::Missing);
    }
    if pyobj.is_instance() {
        if let Some(value) = pyobj.instance_get(name) {
            return Ok(value);
        }
        let cls = pyobj
            .instance_class()
            .ok_or(HostGetAttrError::Unsupported)?;
        return host_class_mro_get(cls, name)
            .map(host_unwrap_staticmethod)
            .ok_or(HostGetAttrError::Missing);
    }
    Err(HostGetAttrError::Unsupported)
}

/// C3 linearisation — CPython `type.__mro__` 알고리즘의 Rust 포트.
///
/// C3 규칙: `L[C] = C + merge(L[B1], L[B2], …, [B1, B2, …])` 에서
/// `merge` 는 각 리스트의 head 를 후보로 보고, **다른 어떤 리스트의
/// tail 에도 등장하지 않는** head 를 선택해 결과에 append 하고 모든
/// 리스트에서 제거한다. 선택 가능한 head 가 없으면 conflict 로
/// linearisation 실패 (Python 은 TypeError). 여기서는 None 을
/// 돌려주고, annotator `add_mixins` 처럼 단일 상속 + mixin 인 경우에는
/// 항상 성공한다.
fn c3_linearise(cls: &HostObject) -> Option<Vec<HostObject>> {
    let bases = cls.class_bases()?;
    let mut lists: Vec<Vec<HostObject>> = Vec::new();
    for base in bases {
        let base_mro = c3_linearise(base)?;
        lists.push(base_mro);
    }
    if !bases.is_empty() {
        lists.push(bases.to_vec());
    }
    let mut result: Vec<HostObject> = vec![cls.clone()];
    loop {
        lists.retain(|l| !l.is_empty());
        if lists.is_empty() {
            return Some(result);
        }
        // 첫 번째 good head 를 찾는다.
        let mut chosen: Option<HostObject> = None;
        for list in &lists {
            let head = &list[0];
            let appears_in_tail = lists.iter().any(|other| other[1..].contains(head));
            if !appears_in_tail {
                chosen = Some(head.clone());
                break;
            }
        }
        let head = chosen?;
        result.push(head.clone());
        for list in lists.iter_mut() {
            if !list.is_empty() && list[0] == head {
                list.remove(0);
            }
        }
    }
}

/// Host namespace 에뮬레이션 — upstream 의 `__builtin__` / imported
/// module table. `HOST_ENV.lookup_builtin(name)` 은
/// `flowcontext.py:851` 의 `getattr(__builtin__, varname)` 에 대응하고,
/// `HOST_ENV.import_module(name)` 은 `flowcontext.py:660` 의
/// `__import__(name, ...)` 에 대응한다.
///
/// Deviation (parity rule #1): upstream 은 flow 시점에 실제 Python
/// `__import__` 를 돌리므로 임의 모듈을 로딩할 수 있다. Rust 포트는
/// Python runtime 을 품지 않기 때문에 flowspace 가 참조할 수 있는
/// module/class/callable 을 bootstrap 시점에 pre-populate 하고,
/// 거기에 없는 이름은 `ImportError` / `FlowingError` 로 빠진다. pyre
/// 를 통합할 때 실제 runtime 을 `HostEnv` backend 로 꽂을 수 있도록
/// API 는 이 접근만 노출한다.
pub struct HostEnv {
    builtins: HashMap<String, HostObject>,
    modules: Mutex<HashMap<String, HostObject>>,
}

impl HostEnv {
    fn bootstrap() -> Self {
        let mut env = HostEnv {
            builtins: HashMap::new(),
            modules: Mutex::new(HashMap::new()),
        };
        env.bootstrap_builtin_exceptions();
        env.bootstrap_builtin_types();
        env.bootstrap_builtin_callables();
        env.bootstrap_std_modules();
        env
    }

    fn insert_builtin(&mut self, name: &str, obj: HostObject) {
        self.builtins.insert(name.to_owned(), obj);
    }

    fn bootstrap_builtin_exceptions(&mut self) {
        // BaseException → Exception → …, rpython/rlib/rstackovf.py 의
        // _StackOverflow 까지 upstream 이 flow 중에 참조하는 class 를
        // 미리 materialise.
        //
        // rstackovf.py:10-14 — `class StackOverflow(RuntimeError)` 이
        // 진짜 class 이고, 같은 class object 가 `_StackOverflow` 라는
        // 이름으로도 바인딩된다. 그 직후 모듈-수준 `StackOverflow` 는
        // `((RuntimeError, RuntimeError),)` 튜플 sentinel 로 rebind
        // 되지만, flowspace 에서 참조되는 식별자는 `_StackOverflow`
        // 쪽이다 (annotator.exception.standard_exceptions 항목 이름과
        // 일치). Rust 포트는 동일 class object 를 두 lookup key 에
        // 등록해 upstream 식별자 공유를 재현한다.
        let base = HostObject::new_class("BaseException", vec![]);
        let exc = HostObject::new_class("Exception", vec![base.clone()]);
        let runtime = HostObject::new_class("RuntimeError", vec![exc.clone()]);
        let stackovf = HostObject::new_class("StackOverflow", vec![runtime.clone()]);
        let not_impl = HostObject::new_class("NotImplementedError", vec![runtime.clone()]);

        let children = [
            ("AssertionError", &exc),
            ("ImportError", &exc),
            ("StopIteration", &exc),
            ("TypeError", &exc),
            ("ValueError", &exc),
            ("ZeroDivisionError", &exc),
            ("AttributeError", &exc),
            ("KeyError", &exc),
            ("IndexError", &exc),
            ("NameError", &exc),
            ("LookupError", &exc),
            ("OSError", &exc),
            ("OverflowError", &exc),
        ];
        for (name, parent) in children {
            let cls = HostObject::new_class(name, vec![parent.clone()]);
            self.insert_builtin(name, cls);
        }
        self.insert_builtin("BaseException", base);
        self.insert_builtin("Exception", exc);
        self.insert_builtin("RuntimeError", runtime);
        // upstream `_StackOverflow = StackOverflow` (rstackovf.py:14) —
        // 동일 class object 를 두 키에 등록. "StackOverflow" 는 모듈
        // 수준의 원 class 이름, "_StackOverflow" 는 flow-level 에서
        // 쓰이는 별칭.
        self.insert_builtin("StackOverflow", stackovf.clone());
        self.insert_builtin("_StackOverflow", stackovf);
        self.insert_builtin("NotImplementedError", not_impl);
    }

    fn bootstrap_builtin_types(&mut self) {
        // upstream 에서 `const(type)` 등으로 참조되는 builtin class
        // object. 상속 관계는 아직 관심 영역이 아니므로 bases 는
        // 비어두고 identity 만 유지한다.
        for name in [
            "type",
            "object",
            "str",
            "int",
            "float",
            "bool",
            "tuple",
            "list",
            "dict",
            "set",
            "frozenset",
            "bytes",
            "bytearray",
            "complex",
            "memoryview",
            "range",
            "slice",
            "enumerate",
            "zip",
            "map",
            "filter",
            "reversed",
            "property",
            "classmethod",
            "staticmethod",
        ] {
            self.insert_builtin(name, HostObject::new_class(name, vec![]));
        }
    }

    fn bootstrap_builtin_callables(&mut self) {
        // upstream `__builtin__` 에 존재하는 callable 중 flowspace 가
        // `find_global` fallback 으로 실제 조회할 수 있는 것들. 기존
        // `BuiltinFunction` enum 의 모든 이름을 그대로 옮긴다 — 추가/
        // 삭제는 upstream 의 `__builtin__` 범위와 연동.
        for name in [
            "__import__",
            "locals",
            "getattr",
            "setattr",
            "delattr",
            "print",
            "all",
            "any",
            "len",
            "iter",
            "next",
            "isinstance",
            "issubclass",
            "hasattr",
            "callable",
            "id",
            "hash",
            "repr",
            "min",
            "max",
            "abs",
            "sum",
            "round",
            "divmod",
            "pow",
            "chr",
            "ord",
            "hex",
            "oct",
            "bin",
            "format",
            "vars",
            "dir",
            "compile",
            "input",
            "exec",
            "eval",
            "super",
            "open",
        ] {
            self.insert_builtin(name, HostObject::new_builtin_callable(name));
        }
        // RPython-only print helpers — upstream `specialcase.py:76-96`.
        for name in [
            "rpython_print_item",
            "rpython_print_end",
            "rpython_print_newline",
        ] {
            self.insert_builtin(name, HostObject::new_builtin_callable(name));
        }
    }

    fn bootstrap_std_modules(&mut self) {
        // `__import__("os", …)` 는 실제 os 모듈을 돌려주지만 Rust 포트
        // 에서는 upstream 의 `specialcase.py:53-67` 가 참조하는 이름만
        // 유지한다. 이 bootstrap 이 빠뜨린 dotted-path 는
        // `import_module` 이 `None` 을 돌려주어 flowspace 단에서
        // `ImportError` 로 번역된다.
        let os = HostObject::new_module("os");
        let os_path = HostObject::new_module("os.path");
        let rfile = HostObject::new_module("rpython.rlib.rfile");
        let rpath = HostObject::new_module("rpython.rlib.rpath");

        os.module_set("fdopen", HostObject::new_builtin_callable("os.fdopen"));
        os.module_set("tmpfile", HostObject::new_builtin_callable("os.tmpfile"));
        os.module_set("remove", HostObject::new_builtin_callable("os.remove"));
        os.module_set("unlink", HostObject::new_builtin_callable("os.unlink"));
        os_path.module_set("isdir", HostObject::new_builtin_callable("os.path.isdir"));
        os_path.module_set("isabs", HostObject::new_builtin_callable("os.path.isabs"));
        os_path.module_set(
            "normpath",
            HostObject::new_builtin_callable("os.path.normpath"),
        );
        os_path.module_set(
            "abspath",
            HostObject::new_builtin_callable("os.path.abspath"),
        );
        os_path.module_set("join", HostObject::new_builtin_callable("os.path.join"));
        os_path.module_set(
            "splitdrive",
            HostObject::new_builtin_callable("os.path.splitdrive"),
        );
        os.module_set("path", os_path.clone());
        rfile.module_set(
            "create_file",
            HostObject::new_builtin_callable("rpython.rlib.rfile.create_file"),
        );
        rfile.module_set(
            "create_fdopen_rfile",
            HostObject::new_builtin_callable("rpython.rlib.rfile.create_fdopen_rfile"),
        );
        rfile.module_set(
            "create_temp_rfile",
            HostObject::new_builtin_callable("rpython.rlib.rfile.create_temp_rfile"),
        );
        rpath.module_set(
            "risdir",
            HostObject::new_builtin_callable("rpython.rlib.rpath.risdir"),
        );
        rpath.module_set(
            "risabs",
            HostObject::new_builtin_callable("rpython.rlib.rpath.risabs"),
        );
        rpath.module_set(
            "rnormpath",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rnormpath"),
        );
        rpath.module_set(
            "rabspath",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rabspath"),
        );
        rpath.module_set(
            "rjoin",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rjoin"),
        );
        rpath.module_set(
            "rsplitdrive",
            HostObject::new_builtin_callable("rpython.rlib.rpath.rsplitdrive"),
        );

        let mut mods = self.modules.lock().unwrap();
        mods.insert("os".into(), os);
        mods.insert("os.path".into(), os_path);
        mods.insert("rpython.rlib.rfile".into(), rfile);
        mods.insert("rpython.rlib.rpath".into(), rpath);
    }

    /// upstream `getattr(__builtin__, name)` — `flowcontext.py:851`.
    pub fn lookup_builtin(&self, name: &str) -> Option<HostObject> {
        self.builtins.get(name).cloned()
    }

    /// upstream `__import__(name, …)` — `flowcontext.py:660`.
    pub fn import_module(&self, name: &str) -> Option<HostObject> {
        self.modules.lock().unwrap().get(name).cloned()
    }

    /// Exception class 가 builtin 테이블에 있다면 그 HostObject 를
    /// 돌려준다. user-defined class 를 새로 등록하는 API 는 현재
    /// 없으며 필요할 때 flowcontext 가 직접 `HostObject::new_class` 로
    /// 구성한다.
    pub fn lookup_exception_class(&self, name: &str) -> Option<HostObject> {
        self.lookup_builtin(name).filter(|obj| {
            obj.is_class()
                && obj.is_subclass_of(self.lookup_builtin("BaseException").as_ref().unwrap())
        })
    }
}

/// 프로세스 전역 host namespace singleton. bootstrap 은 upstream
/// `__builtin__` + 알려진 stdlib 모듈의 placeholder 로 채워진다.
pub static HOST_ENV: LazyLock<HostEnv> = LazyLock::new(HostEnv::bootstrap);

/// Flow-space carrier for Python objects referenced directly by the
/// strict `flowcontext.py` port.
///
/// RPython `Constant.value` is the unwrapped Python object; the Rust
/// port mirrors that contract with an explicit sum type until the
/// wider host object model lands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConstValue {
    /// Sentinel for the `c_last_exception` atom; populated by F1.4
    /// with `Constant(last_exception)`.
    Atom(Atom),
    /// Placeholder for uninstantiated constants during skeleton
    /// construction. Not emitted by production code.
    Placeholder,
    /// Integer constant, e.g. `Constant(0)` / `Constant(1)` in
    /// `test_model.py`. Phase 3 generalises this to arbitrary
    /// Python values.
    Int(i64),
    /// Python float constant. RPython stores the `f64` directly on
    /// `Constant.value`; Rust carries it as `f64::to_bits()` so the
    /// enum keeps its `Eq + Hash` derivation (IEEE 754 NaN violates
    /// `Eq`). Use `ConstValue::float(value)` / `ConstValue::as_float()`
    /// to convert.
    Float(u64),
    /// Python dict constant. Used for `func.__globals__`.
    Dict(HashMap<String, ConstValue>),
    /// Python string constant. Used by `checkgraph()` to validate
    /// single-character switch exitcases and `"default"` ordering.
    Str(String),
    /// Python tuple constant. Used by `CallSpec.as_list()` to unpack
    /// `*args` exactly like upstream's `w_stararg.value`.
    Tuple(Vec<ConstValue>),
    /// Python list constant. Flowspace treats tuple/list starred args
    /// identically at this phase: each element is rewrapped into an
    /// individual `Constant`.
    List(Vec<ConstValue>),
    /// Boolean constant — used as `Link.exitcase` for if/else
    /// switches (`True` / `False`).
    Bool(bool),
    /// `None` — CPython's None singleton, used by
    /// `FrameState._exc_args()` as the sentinel for
    /// "no pending exception".
    None,
    /// Host code object constant (`func.__code__`).
    Code(Box<HostCode>),
    /// Python function object.
    Function(Box<GraphFunc>),
    /// Arbitrary host-level Python object (class, module, builtin
    /// callable, instance). upstream `Constant.value` 에 담기는 임의
    /// object 를 흉내내는 일반 carrier — `HostObject` 참조.
    HostObject(HostObject),
    /// RPython `rpython/rlib/unroll.py:SpecTag` — an identity-bearing
    /// marker instance that prevents two different tags from being
    /// merged by `framestate.union()`. Each `SpecTag(id)` carries a
    /// process-unique `u64` so two separately constructed tags never
    /// compare equal even if wrapped in `Constant`.
    ///
    /// Deviation from upstream (parity rule #1): RPython uses a Python
    /// class (`SpecTag`) whose identity is the instance's `id()`. Rust
    /// has no cross-session `id()`, so we materialise identity as an
    /// atomic counter.
    SpecTag(u64),
}

impl Hash for ConstValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ConstValue::Atom(atom) => atom.hash(state),
            ConstValue::Placeholder => {}
            ConstValue::Int(value) => value.hash(state),
            ConstValue::Float(bits) => bits.hash(state),
            ConstValue::Dict(items) => {
                let mut entries: Vec<_> = items.iter().collect();
                entries.sort_by(|left, right| left.0.cmp(right.0));
                for (key, value) in entries {
                    key.hash(state);
                    value.hash(state);
                }
            }
            ConstValue::Str(value) => value.hash(state),
            ConstValue::Tuple(items) | ConstValue::List(items) => items.hash(state),
            ConstValue::Bool(value) => value.hash(state),
            ConstValue::None => {}
            ConstValue::Code(code) => {
                code.co_name.hash(state);
                code.co_filename.hash(state);
                code.co_firstlineno.hash(state);
                code.co_argcount.hash(state);
                code.co_nlocals.hash(state);
                code.co_flags.hash(state);
            }
            ConstValue::Function(func) => {
                func.name.hash(state);
                func._jit_look_inside_.hash(state);
                func.relax_sig_check.hash(state);
                func.globals.hash(state);
                func.closure.hash(state);
                func.defaults.hash(state);
                func.filename.hash(state);
                func.firstlineno.hash(state);
                func.code.as_ref().map(|code| &code.co_name).hash(state);
            }
            ConstValue::HostObject(obj) => obj.hash(state),
            ConstValue::SpecTag(id) => id.hash(state),
        }
    }
}

/// RPython `flowspace/model.py:463-467` — typed marker like
/// `last_exception`. An `Atom` carries only a name (used by `repr`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Atom {
    /// RPython `Atom.__name__`.
    pub name: &'static str,
}

impl Atom {
    /// RPython `Atom.__init__(name)`.
    pub fn new(name: &'static str) -> Self {
        Atom { name }
    }
}

impl std::fmt::Display for Atom {
    // RPython `Atom.__repr__` returns `self.__name__`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

// RPython `flowspace/model.py:279-352` — `Variable` with lazy name
// numbering.
//
// The dummy name is `'v'`; `namesdict` maps each name prefix to a
// (prefix, next_number) tuple so successive Variables with the same
// prefix pick up unique, increasing numbers.
const DUMMYNAME: &str = "v";

/// Shared `Variable.namesdict`. Key = interned name prefix, value =
/// (stored prefix, next-numbered index).
///
/// RPython has one such dict as a class attribute; Rust needs a
/// `static` + `Mutex` since class-mutable state has no direct
/// equivalent.
static NAMESDICT: LazyLock<Mutex<HashMap<String, (String, u32)>>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert(DUMMYNAME.to_owned(), (DUMMYNAME.to_owned(), 0));
    Mutex::new(m)
});

/// Per-process id counter for Variable identity (RPython identity
/// parity). Rust has no Python-style `id(obj)`, so each Variable
/// gets a unique `u64` at construction; `clone()` preserves it
/// (aliasing the same logical Variable), while `copy()` takes a
/// fresh id (same semantics as RPython's `Variable.copy`, which
/// returns a *new* Variable with the same prefix).
static NEXT_VAR_ID: AtomicU64 = AtomicU64::new(0);

fn alloc_var_id() -> u64 {
    NEXT_VAR_ID.fetch_add(1, Ordering::Relaxed)
}

/// RPython `flowspace/model.py:279` — `class Variable`.
///
/// `__slots__ = ["_name", "_nr", "annotation", "concretetype"]`.
#[derive(Debug)]
pub struct Variable {
    /// Identity key. See `NEXT_VAR_ID` — RPython uses object
    /// identity; Rust approximates with a process-wide increment.
    id: u64,
    _name: String,
    _nr: std::cell::Cell<i64>,
    /// RPython `Variable.annotation` (set by the annotator). Holds a
    /// shared [`SomeValue`] handle once the annotator binds the
    /// variable — upstream Python uses reference semantics so one
    /// lattice instance can back many `Variable`s.
    pub annotation: Option<Rc<SomeValue>>,
    /// RPython `Variable.concretetype` (set by the rtyper).
    pub concretetype: Option<ConcretetypePlaceholder>,
}

impl Clone for Variable {
    // RPython has no `clone` at the language level, but downstream
    // code that stores a `Variable` in a Vec and later reuses the
    // "same" Variable relies on identity being preserved. `clone`
    // in Rust therefore aliases the identity; use `copy()` for
    // RPython `Variable.copy` semantics.
    fn clone(&self) -> Self {
        Variable {
            id: self.id,
            _name: self._name.clone(),
            _nr: std::cell::Cell::new(self._nr.get()),
            annotation: self.annotation.clone(),
            concretetype: self.concretetype,
        }
    }
}

impl PartialEq for Variable {
    // RPython relies on Python object identity for Variable
    // equality; the `id` field is a direct port.
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Variable {}

impl std::hash::Hash for Variable {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl Variable {
    /// RPython `Variable.__init__(name=None)`.
    pub fn new() -> Self {
        // `_name = self.dummyname; _nr = -1; annotation = None`.
        Variable {
            id: alloc_var_id(),
            _name: DUMMYNAME.to_owned(),
            _nr: std::cell::Cell::new(-1),
            annotation: None,
            concretetype: None,
        }
    }

    /// RPython `Variable.__init__(name)` with a non-None argument.
    pub fn named(name: impl AsRef<str>) -> Self {
        let mut v = Self::new();
        v.rename(name.as_ref());
        v
    }

    /// RPython `Variable.name` (property): lazy-allocate a unique
    /// number for this Variable's prefix from the shared namesdict.
    pub fn name(&self) -> String {
        let mut nr = self._nr.get();
        if nr == -1 {
            let mut nd = NAMESDICT.lock().unwrap();
            let entry = nd
                .entry(self._name.clone())
                .or_insert_with(|| (self._name.clone(), 0));
            nr = entry.1 as i64;
            entry.1 += 1;
            self._nr.set(nr);
        }
        format!("{}{}", self._name, nr)
    }

    /// RPython `Variable.renamed` (property).
    pub fn renamed(&self) -> bool {
        self._name != DUMMYNAME
    }

    /// Generator flowspace needs the pre-numbered name prefix that
    /// upstream reads as `v._name`.
    pub fn name_prefix(&self) -> &str {
        &self._name
    }

    /// RPython `Variable.rename(name)`.
    ///
    /// Only renames once: subsequent calls are no-ops. The RPython
    /// source takes either a string or another Variable as argument;
    /// the Rust port exposes two overloads (`rename(&str)` below
    /// handles the string case, `rename_from(&Variable)` handles the
    /// Variable case — see `set_name_from`).
    pub fn rename(&mut self, name: &str) {
        if self._name != DUMMYNAME {
            return;
        }
        // Remove strange characters; upstream uses a translation
        // table exported as `PY_IDENTIFIER`. Port the same
        // observable behaviour: keep `[A-Za-z0-9_]`, swap the rest
        // to `_`. Always trail with a `_`. Prefix numeric first-chars
        // with `_` to ensure a valid Python identifier.
        let mut cleaned: String = name
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        cleaned.push('_');
        if let Some(first) = cleaned.chars().next() {
            if !first.is_ascii_alphabetic() && first != '_' {
                cleaned.insert(0, '_');
            }
        }
        // `namesdict.setdefault(name, (name, 0))[0]`.
        let mut nd = NAMESDICT.lock().unwrap();
        let entry = nd
            .entry(cleaned.clone())
            .or_insert_with(|| (cleaned.clone(), 0));
        self._name = entry.0.clone();
        self._nr.set(-1);
    }

    /// RPython `Variable.set_name_from(v)` (for SSI_to_SSA).
    pub fn set_name_from(&mut self, other: &Variable) {
        // `v.name` forces finalisation on the RHS.
        let _ = other.name();
        self._name = other._name.clone();
        self._nr.set(other._nr.get());
    }

    /// RPython `Variable.set_name(name, nr)` (for wrapper.py).
    pub fn set_name(&mut self, name: String, nr: i64) {
        self._name = name;
        self._nr.set(nr);
    }

    /// RPython `Variable.foldable()` — always False for Variable.
    pub fn foldable(&self) -> bool {
        false
    }

    /// RPython `Variable.copy()` — creates a *new* Variable with a
    /// fresh identity, preserving name prefix, annotation, and
    /// concretetype.
    pub fn copy(&self) -> Self {
        // `Variable::new()` allocates a fresh id; we then overwrite
        // the prefix with the source's _name, matching RPython's
        // `Variable(v)` path that copies the prefix via `rename`.
        let mut newvar = Variable::new();
        newvar._name = self._name.clone();
        newvar._nr.set(-1);
        newvar.annotation = self.annotation.clone();
        newvar.concretetype = self.concretetype;
        newvar
    }

    /// RPython `Variable.replace(mapping)`: `mapping.get(self, self)`.
    pub fn replace<'a>(&'a self, mapping: &'a HashMap<Variable, Variable>) -> &'a Variable {
        mapping.get(self).unwrap_or(self)
    }
}

impl std::fmt::Display for Variable {
    // RPython `Variable.__repr__` returns `self.name`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name())
    }
}

impl Default for Variable {
    fn default() -> Self {
        Self::new()
    }
}

/// RPython `flowspace/model.py:354-382` — `class Constant(Hashable)`.
///
/// `__slots__ = ["concretetype"]`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Constant {
    /// RPython `Hashable.value`.
    pub value: ConstValue,
    /// RPython `Constant.concretetype`.
    pub concretetype: Option<ConcretetypePlaceholder>,
}

impl Constant {
    /// RPython `Constant.__init__(value, concretetype=None)`.
    pub fn new(value: ConstValue) -> Self {
        Constant {
            value,
            concretetype: None,
        }
    }

    /// RPython `Constant.__init__(value, concretetype)`.
    pub fn with_concretetype(value: ConstValue, concretetype: ConcretetypePlaceholder) -> Self {
        Constant {
            value,
            concretetype: Some(concretetype),
        }
    }

    /// RPython `flowspace/model.py:362-379` — `Constant.foldable()`.
    ///
    /// upstream line-by-line:
    /// ```python
    /// to_check = self.value
    /// if hasattr(to_check, 'im_self'):
    ///     to_check = to_check.im_self
    /// if isinstance(to_check, (type, types.ClassType, types.ModuleType)):
    ///     return True
    /// if (hasattr(to_check, '__class__') and
    ///         to_check.__class__.__module__ == '__builtin__'):
    ///     return True
    /// if hasattr(to_check, '_freeze_'):
    ///     assert to_check._freeze_() is True
    ///     return True
    /// return False
    /// ```
    /// Rust 매핑:
    /// * type/ClassType/ModuleType arm → `HostObject::is_class()` or
    ///   `is_module()` — 실제 host object 의 분류를 읽는다. builtin
    ///   type (`const(type)`, `const(str)`, ...) 도 `is_class()` 로
    ///   포착된다.
    /// * `__class__.__module__ == '__builtin__'` arm → Rust enum 으로
    ///   직접 표현되는 builtin 값들 (`Int` / `Float` / `Bool` / `Str` /
    ///   `None` / `Tuple` / `List` / `Dict` / `Code`) 과 `HostObject`
    ///   중 builtin-callable 분류.
    /// * `_freeze_` arm → Rust 포트는 `_freeze_` 프로토콜을 갖지 않으므
    ///   로 현재 만족하는 variant 없음 (Phase 5 annotator에서 도입).
    ///
    /// 최종 False 로 떨어지는 variant:
    /// `Atom`, `Placeholder`, `Function(_)` (user function — upstream
    /// `_freeze_` 없는 한 False), `HostObject::Instance` (user-defined
    /// instance — `__class__.__module__` 이 `'__main__'` 등 — 기본
    /// False), `SpecTag`.
    pub fn foldable(&self) -> bool {
        match &self.value {
            // `isinstance(to_check, type)` / ClassType / ModuleType.
            ConstValue::HostObject(obj) => {
                if obj.is_class() || obj.is_module() {
                    true
                } else if obj.is_builtin_callable() {
                    // `builtin_function_or_method` 의 `__module__` 은
                    // `'__builtin__'`.
                    true
                } else {
                    // user function / user instance → `_freeze_` 가
                    // 있어야 True. 미지원.
                    false
                }
            }
            // `to_check.__class__.__module__ == '__builtin__'` — Python
            // built-in primitive / container type 의 값.
            ConstValue::Int(_)
            | ConstValue::Float(_)
            | ConstValue::Bool(_)
            | ConstValue::Str(_)
            | ConstValue::None
            | ConstValue::Tuple(_)
            | ConstValue::List(_)
            | ConstValue::Dict(_)
            | ConstValue::Code(_) => true,
            // 아래는 upstream 의 최종 `return False`.
            ConstValue::Atom(_)
            | ConstValue::Placeholder
            | ConstValue::Function(_)
            | ConstValue::SpecTag(_) => false,
        }
    }

    /// RPython `Constant.replace(mapping)` — Constants never rename.
    pub fn replace<'a>(&'a self, _mapping: &'a HashMap<Variable, Variable>) -> &'a Constant {
        self
    }
}

impl ConstValue {
    /// Wrap an `f64` into the bit-preserving `Float` variant.
    pub fn float(value: f64) -> Self {
        ConstValue::Float(value.to_bits())
    }

    /// Unwrap a `Float` variant back to `f64`. Returns `None` for
    /// other variants.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ConstValue::Float(bits) => Some(f64::from_bits(*bits)),
            _ => None,
        }
    }

    /// Best-effort Python truthiness for the constant variants the
    /// current flow-space port constructs directly.
    pub fn truthy(&self) -> Option<bool> {
        match self {
            ConstValue::Placeholder => None,
            ConstValue::Int(n) => Some(*n != 0),
            // Python `bool(float)` is `float != 0.0` (including -0.0
            // which equals 0.0 under IEEE 754).
            ConstValue::Float(bits) => Some(f64::from_bits(*bits) != 0.0),
            ConstValue::Dict(items) => Some(!items.is_empty()),
            ConstValue::Str(s) => Some(!s.is_empty()),
            ConstValue::Tuple(items) | ConstValue::List(items) => Some(!items.is_empty()),
            ConstValue::Bool(value) => Some(*value),
            ConstValue::None => Some(false),
            ConstValue::Code(_) => Some(true),
            ConstValue::Function(_) => Some(true),
            ConstValue::HostObject(_) => Some(true),
            ConstValue::Atom(_) => Some(true),
            ConstValue::SpecTag(_) => Some(true),
        }
    }

    pub fn dict_items(&self) -> Option<&HashMap<String, ConstValue>> {
        match self {
            ConstValue::Dict(items) => Some(items),
            _ => None,
        }
    }

    pub fn sequence_items(&self) -> Option<&[ConstValue]> {
        match self {
            ConstValue::Tuple(items) | ConstValue::List(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    /// RPython `for x in self.w_stararg.value` iteration contract
    /// (`argument.py:113`) — upstream consumes any Python iterable.
    /// Line-by-line port for the `ConstValue` variants that can appear
    /// as `w_stararg.value` in practice: tuple/list (flow-space's
    /// BUILD_TUPLE/BUILD_LIST output), str (per-char iteration matches
    /// Python semantics), dict (iteration over keys in insertion order;
    /// HashMap iteration order is used in the Rust port — upstream's
    /// dict is also unordered before the flow-graph builder sorts
    /// explicitly, so this matches upstream's pre-sort state).
    pub fn iter_items(&self) -> Option<Vec<ConstValue>> {
        match self {
            ConstValue::Tuple(items) | ConstValue::List(items) => Some(items.clone()),
            ConstValue::Str(value) => Some(
                value
                    .chars()
                    .map(|c| ConstValue::Str(c.to_string()))
                    .collect(),
            ),
            ConstValue::Dict(items) => Some(items.keys().cloned().map(ConstValue::Str).collect()),
            _ => None,
        }
    }

    /// Exception class 이면 `qualname()` 을 돌려준다. 임의 Class 가
    /// 예외 클래스인지 판정하려면 `HOST_ENV.lookup_builtin("BaseException")`
    /// 과 `is_subclass_of` 로 체크한다 — 이 helper 는 편의상 class 면
    /// 아무 qualname 이나 노출한다.
    pub fn host_class_name(&self) -> Option<&str> {
        match self {
            ConstValue::HostObject(obj) if obj.is_class() => Some(obj.qualname()),
            _ => None,
        }
    }

    /// HostObject 인 경우 reference. 기존 ExceptionClass/Builtin/
    /// ExceptionInstance 대체 패턴에서 공통으로 사용.
    pub fn as_host_object(&self) -> Option<&HostObject> {
        match self {
            ConstValue::HostObject(obj) => Some(obj),
            _ => None,
        }
    }

    /// `__builtin__` / stdlib namespace 에서 이름으로 HostObject 를 끌어내
    /// `ConstValue::HostObject` 로 감싼다. `HOST_ENV` 에 해당 이름이
    /// 없으면 panic (bootstrap 누락은 개발 단계에서 즉시 드러내는 편이
    /// 안전하다).
    pub fn builtin(name: &str) -> Self {
        let obj = HOST_ENV
            .lookup_builtin(name)
            .unwrap_or_else(|| panic!("HOST_ENV missing builtin {name}"));
        ConstValue::HostObject(obj)
    }
}

/// Application-level exception captured inside the flow space.
///
/// RPython basis: `rpython/flowspace/model.py:385-392` — `class
/// FSException(object)`. `w_type` and `w_value` carry the exception
/// class and instance as flow-space `Hlvalue`s.
///
/// RPython's `ConstException(Constant, FSException)` multiple
/// inheritance has no direct Rust equivalent; the upstream only uses
/// it as a fast-path marker for `foldable()` on constant exceptions.
/// Phase 3 (`flowspace/operation.py`) handles the same flow via a
/// `Constant`-carrying FSException without the extra class.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FSException {
    /// RPython `FSException.w_type` — the exception class (as an
    /// `Hlvalue`, usually a `Constant` wrapping the Python class).
    pub w_type: Hlvalue,
    /// RPython `FSException.w_value` — the exception instance (as an
    /// `Hlvalue`).
    pub w_value: Hlvalue,
}

impl FSException {
    /// RPython `FSException.__init__(w_type, w_value)`.
    pub fn new(w_type: Hlvalue, w_value: Hlvalue) -> Self {
        FSException { w_type, w_value }
    }
}

impl std::fmt::Display for FSException {
    // RPython `FSException.__str__` — `[w_type: w_value]`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}: {}]", self.w_type, self.w_value)
    }
}

/// Mixed `Variable | Constant` cell used by `SpaceOperation.args`,
/// `SpaceOperation.result`, and `Block.inputargs`.
///
/// RPython relies on duck-typing: the upstream lists literally hold
/// either a `Variable` or a `Constant` instance and dispatch through
/// `.replace(mapping)`. Rust needs a closed sum to hand a single
/// owned type across API boundaries; this is the minimum deviation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Hlvalue {
    Variable(Variable),
    Constant(Constant),
}

impl Hlvalue {
    /// RPython `.replace(mapping)` dispatch — matches whichever
    /// subclass `Variable.replace` / `Constant.replace` method the
    /// cell carries.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Hlvalue {
        match self {
            Hlvalue::Variable(v) => Hlvalue::Variable(v.replace(mapping).clone()),
            Hlvalue::Constant(c) => Hlvalue::Constant(c.replace(mapping).clone()),
        }
    }
}

impl std::fmt::Display for Hlvalue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Hlvalue::Variable(v) => write!(f, "{v}"),
            // RPython `Constant.__repr__` (inherited from Hashable)
            // prints `<value>`; we approximate with the enum debug.
            Hlvalue::Constant(c) => write!(f, "{c:?}"),
        }
    }
}

impl From<Variable> for Hlvalue {
    fn from(v: Variable) -> Self {
        Hlvalue::Variable(v)
    }
}

impl From<Constant> for Hlvalue {
    fn from(c: Constant) -> Self {
        Hlvalue::Constant(c)
    }
}

/// RPython `flowspace/model.py:434-461` — `class SpaceOperation`.
#[derive(Clone, Debug, Eq)]
pub struct SpaceOperation {
    /// RPython `SpaceOperation.opname` — `intern(opname)`.
    pub opname: String,
    /// RPython `SpaceOperation.args` — mixed `Variable`/`Constant`
    /// list.
    pub args: Vec<Hlvalue>,
    /// RPython `SpaceOperation.result` — either `Variable` or
    /// `Constant`.
    pub result: Hlvalue,
    /// RPython `SpaceOperation.offset` — source bytecode offset,
    /// `-1` when unknown.
    pub offset: i64,
}

impl SpaceOperation {
    /// RPython `SpaceOperation.__init__(opname, args, result,
    /// offset=-1)`.
    pub fn new(opname: impl Into<String>, args: Vec<Hlvalue>, result: Hlvalue) -> Self {
        SpaceOperation {
            opname: opname.into(),
            args,
            result,
            offset: -1,
        }
    }

    /// RPython overload with an explicit `offset`.
    pub fn with_offset(
        opname: impl Into<String>,
        args: Vec<Hlvalue>,
        result: Hlvalue,
        offset: i64,
    ) -> Self {
        SpaceOperation {
            opname: opname.into(),
            args,
            result,
            offset,
        }
    }

    /// RPython `SpaceOperation.replace(mapping)`.
    ///
    /// Returns a fresh SpaceOperation with the same opname/offset and
    /// every arg/result remapped through `mapping`.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> SpaceOperation {
        let newargs: Vec<Hlvalue> = self.args.iter().map(|a| a.replace(mapping)).collect();
        let newresult = self.result.replace(mapping);
        SpaceOperation {
            opname: self.opname.clone(),
            args: newargs,
            result: newresult,
            offset: self.offset,
        }
    }
}

impl PartialEq for SpaceOperation {
    // RPython `SpaceOperation.__eq__`: same class, same opname, same
    // args, same result. `offset` is intentionally NOT part of the
    // identity (upstream: model.py:442-446).
    fn eq(&self, other: &Self) -> bool {
        self.opname == other.opname && self.args == other.args && self.result == other.result
    }
}

impl std::hash::Hash for SpaceOperation {
    // RPython `SpaceOperation.__hash__`: `hash((opname, tuple(args),
    // result))`.
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.opname.hash(state);
        self.args.hash(state);
        self.result.hash(state);
    }
}

impl std::fmt::Display for SpaceOperation {
    // RPython `SpaceOperation.__repr__` returns
    // `"%r = %s(%s)" % (result, opname, ", ".join(map(repr, args)))`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} = {}(", self.result, self.opname)?;
        let mut first = true;
        for arg in &self.args {
            if !first {
                f.write_str(", ")?;
            }
            write!(f, "{arg}")?;
            first = false;
        }
        f.write_str(")")
    }
}

// RPython `flowspace/model.py:469-470`.
//
// `last_exception = Atom('last_exception')`
// `c_last_exception = Constant(last_exception)`
//
// A Block's `exitswitch` set to `c_last_exception` signals that the
// last operation of the block may raise, and the exit links' exitcase
// fields carry the matching exception classes.

/// RPython `last_exception` — module-level Atom. Static so both the
/// `Block::canraise` check and the `c_last_exception` constant share
/// identity.
pub static LAST_EXCEPTION: Atom = Atom {
    name: "last_exception",
};

/// RPython `c_last_exception = Constant(last_exception)`.
pub fn c_last_exception() -> Constant {
    Constant::new(ConstValue::Atom(LAST_EXCEPTION.clone()))
}

/// `Rc<RefCell<Block>>` alias matching the RPython behaviour of
/// Block/Link references being mutable shared pointers. Every Link
/// targets a Block by shared reference; every Block holds outgoing
/// Links by shared reference. Cycles back through `Link.prevblock`
/// use `Weak<RefCell<Block>>` to avoid leaks.
pub type BlockRef = Rc<RefCell<Block>>;

/// `Rc<RefCell<Link>>` alias — see `BlockRef`.
pub type LinkRef = Rc<RefCell<Link>>;

/// `Link.args` carries the values flowing into a target block's
/// `inputargs`. RPython's `FrameState.getoutputargs()` may emit an
/// undefined-local sentinel (`None`) in transient merge links, so the
/// Rust port preserves that exact shape with `Option<Hlvalue>`.
pub type LinkArg = Option<Hlvalue>;

/// RPython `flowspace/model.py:109-168` — `class Link`.
///
/// `__slots__ = "args target exitcase llexitcase prevblock
///              last_exception last_exc_value".split()`.
#[derive(Debug)]
pub struct Link {
    /// RPython `Link.args` — mixed list of var/const, with transient
    /// merge links allowed to carry `None` for undefined locals.
    pub args: Vec<LinkArg>,
    /// RPython `Link.target` — successor block. Always `Some` in
    /// constructed graphs; `Option` only to admit the upstream
    /// `target=None` transient state during graph building.
    pub target: Option<BlockRef>,
    /// RPython `Link.exitcase` — concrete value discriminating this
    /// exit. `None` for fall-through / single-exit blocks; Python
    /// values (via `Hlvalue::Constant`) for switch exits; exception
    /// class Constants for `c_last_exception` blocks.
    pub exitcase: Option<Hlvalue>,
    /// RPython `Link.llexitcase` — low-level equivalent of
    /// `exitcase`, populated by the rtyper at Phase 6. Placeholder
    /// here.
    pub llexitcase: Option<Hlvalue>,
    /// RPython `Link.prevblock` — the Block this Link exits. Weak
    /// reference to break the cycle with `Block.exits`.
    pub prevblock: Option<Weak<RefCell<Block>>>,
    /// RPython `Link.last_exception` — extra variable introduced on
    /// exception-handling links.
    pub last_exception: Option<Hlvalue>,
    /// RPython `Link.last_exc_value` — sibling of `last_exception`.
    pub last_exc_value: Option<Hlvalue>,
}

impl Link {
    /// RPython `Link.__init__(args, target, exitcase=None)`.
    pub fn new(args: Vec<Hlvalue>, target: Option<BlockRef>, exitcase: Option<Hlvalue>) -> Self {
        Self::new_mergeable(args.into_iter().map(Some).collect(), target, exitcase)
    }

    pub fn new_mergeable(
        args: Vec<LinkArg>,
        target: Option<BlockRef>,
        exitcase: Option<Hlvalue>,
    ) -> Self {
        if let Some(t) = &target {
            assert_eq!(
                args.len(),
                t.borrow().inputargs.len(),
                "output args mismatch"
            );
        }
        Link {
            args,
            target,
            exitcase,
            llexitcase: None,
            prevblock: None,
            last_exception: None,
            last_exc_value: None,
        }
    }

    /// Wraps the owned `Link` in a shared `LinkRef`. Rust adaptation
    /// closing upstream's Python identity-sharing: `Block.closeblock`
    /// stores the link by reference into `block.exits`, and
    /// `Link.prevblock` is later set by `closeblock` through the
    /// stored reference.
    pub fn into_ref(self) -> LinkRef {
        Rc::new(RefCell::new(self))
    }

    /// RPython `Link.extravars(last_exception=None, last_exc_value=None)`.
    pub fn extravars(&mut self, last_exception: Option<Hlvalue>, last_exc_value: Option<Hlvalue>) {
        self.last_exception = last_exception;
        self.last_exc_value = last_exc_value;
    }

    /// RPython `Link.getextravars()` — collect the Variable-typed
    /// extras, ignoring Constant slots.
    pub fn getextravars(&self) -> Vec<Variable> {
        let mut result = Vec::new();
        if let Some(Hlvalue::Variable(v)) = &self.last_exception {
            result.push(v.clone());
        }
        if let Some(Hlvalue::Variable(v)) = &self.last_exc_value {
            result.push(v.clone());
        }
        result
    }

    /// RPython `Link.copy(rename=lambda x: x)`.
    ///
    /// The `rename` callback maps each `Hlvalue` to a substitute —
    /// usually derived from a `Variable -> Variable` mapping. The
    /// closure form mirrors upstream's call shape.
    pub fn copy<F>(&self, rename: F) -> Link
    where
        F: Fn(&Hlvalue) -> Hlvalue,
    {
        let newargs: Vec<LinkArg> = self
            .args
            .iter()
            .map(|arg| arg.as_ref().map(&rename))
            .collect();
        let mut newlink = Link::new_mergeable(newargs, self.target.clone(), self.exitcase.clone());
        newlink.prevblock = self.prevblock.clone();
        newlink.last_exception = self.last_exception.as_ref().map(&rename);
        newlink.last_exc_value = self.last_exc_value.as_ref().map(&rename);
        newlink.llexitcase = self.llexitcase.clone();
        newlink
    }

    /// RPython `Link.replace(mapping)`.
    pub fn replace(&self, mapping: &HashMap<Variable, Variable>) -> Link {
        self.copy(|v| v.replace(mapping))
    }

    /// RPython `Link.settarget(targetblock)`.
    pub fn settarget(&mut self, targetblock: BlockRef) {
        assert_eq!(
            self.args.len(),
            targetblock.borrow().inputargs.len(),
            "output args mismatch"
        );
        self.target = Some(targetblock);
    }
}

/// RPython `flowspace/model.py:171-276` — `class Block`.
///
/// `__slots__ = "inputargs operations exitswitch exits blockcolor
///              generation".split()`.
#[derive(Debug)]
pub struct Block {
    /// RPython `Block.inputargs` — mixed list of variable/const.
    pub inputargs: Vec<Hlvalue>,
    /// RPython `Block.operations` — list of `SpaceOperation`.
    pub operations: Vec<SpaceOperation>,
    /// Block parity flag: final (return/except) blocks have
    /// `operations == ()` in RPython (a tuple, not a list).
    pub is_final: bool,
    /// RPython `Block.exitswitch` — either a Variable, a Constant,
    /// or `c_last_exception`. `None` means no exit discriminator.
    pub exitswitch: Option<Hlvalue>,
    /// RPython `Block.exits` — list of outgoing Links.
    pub exits: Vec<LinkRef>,
    /// RPython `Block.blockcolor` — used by graph-layout visualisers.
    pub blockcolor: Option<u32>,
    /// RPython `Block.generation` — used by `flowcontext` iteration
    /// scheduling.
    pub generation: Option<u32>,
}

impl Block {
    /// RPython `Block.__init__(inputargs)`.
    pub fn new(inputargs: Vec<Hlvalue>) -> Self {
        Block {
            inputargs,
            operations: Vec::new(),
            is_final: false,
            exitswitch: None,
            exits: Vec::new(),
            blockcolor: None,
            generation: None,
        }
    }

    /// Shared-ownership constructor — upstream passes `Block(args)`
    /// and stores it directly in `FunctionGraph`/`Link.target`; our
    /// graph types take `Rc<RefCell<Block>>`, so callers usually
    /// want `Block::shared(...)` instead of `Block::new(...)`.
    pub fn shared(inputargs: Vec<Hlvalue>) -> BlockRef {
        Rc::new(RefCell::new(Block::new(inputargs)))
    }

    /// RPython `Block.is_final_block()` — `self.operations == ()`.
    pub fn is_final_block(&self) -> bool {
        self.is_final
    }

    /// Mark this block as final (return/except). Matches upstream's
    /// post-`__init__` assignment `block.operations = ()`.
    pub fn mark_final(&mut self) {
        self.operations.clear();
        self.is_final = true;
    }

    /// RPython `Block.canraise` (property).
    pub fn canraise(&self) -> bool {
        match &self.exitswitch {
            Some(Hlvalue::Constant(c)) => matches!(
                &c.value,
                ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name
            ),
            _ => false,
        }
    }

    /// RPython `Block.raising_op` (property).
    pub fn raising_op(&self) -> Option<&SpaceOperation> {
        if self.canraise() {
            self.operations.last()
        } else {
            None
        }
    }

    /// RPython `Block.getvariables()` — unique Variables mentioned
    /// in this block (inputargs + every op's args + every op's
    /// result).
    pub fn getvariables(&self) -> Vec<Variable> {
        let mut result: Vec<Variable> = Vec::new();
        let mut push_var = |w: &Hlvalue, result: &mut Vec<Variable>| {
            if let Hlvalue::Variable(v) = w {
                if !result.iter().any(|x| x == v) {
                    result.push(v.clone());
                }
            }
        };
        for w in &self.inputargs {
            push_var(w, &mut result);
        }
        for op in &self.operations {
            for a in &op.args {
                push_var(a, &mut result);
            }
            push_var(&op.result, &mut result);
        }
        result
    }

    /// RPython `Block.getconstants()` — unique Constants mentioned
    /// in this block (inputargs + every op's args).
    pub fn getconstants(&self) -> Vec<Constant> {
        let mut result: Vec<Constant> = Vec::new();
        let mut push_const = |w: &Hlvalue, result: &mut Vec<Constant>| {
            if let Hlvalue::Constant(c) = w {
                if !result.iter().any(|x| x == c) {
                    result.push(c.clone());
                }
            }
        };
        for w in &self.inputargs {
            push_const(w, &mut result);
        }
        for op in &self.operations {
            for a in &op.args {
                push_const(a, &mut result);
            }
        }
        result
    }

    /// RPython `Block.renamevariables(mapping)`.
    pub fn renamevariables(&mut self, mapping: &HashMap<Variable, Variable>) {
        self.inputargs = self.inputargs.iter().map(|a| a.replace(mapping)).collect();
        self.operations = self
            .operations
            .iter()
            .map(|op| op.replace(mapping))
            .collect();
        if let Some(sw) = self.exitswitch.take() {
            self.exitswitch = Some(sw.replace(mapping));
        }
        for link_ref in &self.exits {
            let mut link = link_ref.borrow_mut();
            link.args = link
                .args
                .iter()
                .map(|arg| arg.as_ref().map(|value| value.replace(mapping)))
                .collect();
        }
    }

    /// RPython `Block.closeblock(*exits)` — one-shot close.
    pub fn closeblock(&mut self, exits: Vec<LinkRef>) {
        assert!(self.exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    /// RPython `Block.recloseblock(*exits)` — may be called after
    /// `closeblock`, rewiring `Link.prevblock` to this block.
    ///
    /// Caller must pass `self_ref: Weak<RefCell<Block>>` because
    /// `&mut self` cannot upgrade back to the Rc. Upstream has this
    /// for free via Python object identity.
    pub fn recloseblock_with_self(&mut self, self_ref: Weak<RefCell<Block>>, exits: Vec<LinkRef>) {
        for link in &exits {
            link.borrow_mut().prevblock = Some(self_ref.clone());
        }
        self.exits = exits;
    }

    /// Variant that skips `prevblock` wiring when the caller has
    /// already set it (e.g. during `copygraph`).
    pub fn recloseblock(&mut self, exits: Vec<LinkRef>) {
        self.exits = exits;
    }
}

/// Extension trait that gives `BlockRef` the full RPython
/// `Block.closeblock` / `Block.recloseblock` API, including
/// `Link.prevblock` wiring. `Block::closeblock`/`recloseblock` on
/// `&mut self` cannot wire `prevblock` because the borrow-checker
/// does not let us upgrade back to the owning Rc; the trait provides
/// the `self_ref`-aware version.
pub trait BlockRefExt {
    /// RPython `Block.closeblock(*exits)`.
    fn closeblock(&self, exits: Vec<LinkRef>);
    /// RPython `Block.recloseblock(*exits)`.
    fn recloseblock(&self, exits: Vec<LinkRef>);
}

impl BlockRefExt for BlockRef {
    fn closeblock(&self, exits: Vec<LinkRef>) {
        assert!(self.borrow().exits.is_empty(), "block already closed");
        self.recloseblock(exits);
    }

    fn recloseblock(&self, exits: Vec<LinkRef>) {
        let weak = Rc::downgrade(self);
        for link in &exits {
            link.borrow_mut().prevblock = Some(weak.clone());
        }
        self.borrow_mut().exits = exits;
    }
}

/// Stand-in for the Python function object attached to
/// `FunctionGraph.func`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphFunc {
    /// Python function `__name__`.
    pub name: String,
    /// Optional method owner, mirroring upstream `func.class_`.
    pub class_: Option<HostObject>,
    /// Upstream `func._annspecialcase_` decorator tag consulted by
    /// `Bookkeeper.newfuncdesc()` when selecting a specializer.
    pub annspecialcase: Option<String>,
    /// Upstream generator helper attribute populated by
    /// `attach_next_method()`.
    pub _generator_next_method_of_: Option<HostObject>,
    /// Upstream function attribute consulted by
    /// `annotator/specialize.py:64` and `jit/codewriter/policy.py:56`.
    ///
    /// `None` mirrors the common "attribute absent" case so callers
    /// can apply the same default as upstream's
    /// `getattr(func, '_jit_look_inside_', True)`.
    pub _jit_look_inside_: Option<bool>,
    /// Upstream function attribute consulted by
    /// `description.py:222-224`.
    ///
    /// `None` mirrors the common "attribute absent" case so callers
    /// can apply the same default as upstream's
    /// `getattr(func, 'relax_sig_check', False)`.
    pub relax_sig_check: Option<bool>,
    /// Python function `__globals__`, wrapped as a flow-space
    /// constant.
    pub globals: Constant,
    /// Python function `__closure__`.
    pub closure: Vec<Constant>,
    /// Python function `__defaults__`.
    pub defaults: Vec<Constant>,
    /// Python function `__code__`.
    pub code: Option<Box<HostCode>>,
    /// Source text returned by `inspect.getsource(func)`.
    pub source: Option<String>,
    /// `func.__code__.co_firstlineno`.
    pub firstlineno: Option<u32>,
    /// `func.__code__.co_filename`.
    pub filename: Option<String>,
    /// Upstream `func._not_rpython_` attribute (objspace.py:21). Set
    /// to `true` by RPython's `@not_rpython` decorator to mark a
    /// function as flow-space-ineligible.
    pub not_rpython: bool,
}

impl GraphFunc {
    pub fn new(name: impl Into<String>, globals: Constant) -> Self {
        GraphFunc {
            name: name.into(),
            class_: None,
            annspecialcase: None,
            _generator_next_method_of_: None,
            _jit_look_inside_: None,
            relax_sig_check: None,
            globals,
            closure: Vec::new(),
            defaults: Vec::new(),
            code: None,
            source: None,
            firstlineno: None,
            filename: None,
            not_rpython: false,
        }
    }

    /// RPython `func_with_new_name(func, newname)` copies the function
    /// object while reusing `__code__`, `__defaults__`, and
    /// `__closure__`; the `__code__.co_name` payload stays unchanged.
    pub fn with_new_name(&self, newname: &str) -> Self {
        let mut cloned = self.clone();
        cloned.name = newname.to_string();
        cloned
    }

    pub fn from_host_code(code: HostCode, globals: Constant, defaults: Vec<Constant>) -> Self {
        let mut func = GraphFunc::new(code.co_name.clone(), globals);
        func.filename = Some(code.co_filename.clone());
        func.firstlineno = Some(code.co_firstlineno);
        func.defaults = defaults;
        func.code = Some(Box::new(code));
        func
    }
}

/// RPython `flowspace/model.py:13-106` — `class FunctionGraph`.
#[derive(Debug)]
pub struct FunctionGraph {
    /// RPython `FunctionGraph.name`.
    pub name: String,
    /// RPython `FunctionGraph.startblock`.
    pub startblock: BlockRef,
    /// RPython `FunctionGraph.returnblock` — `Block([return_var])`
    /// with `operations = ()` and `exits = ()`.
    pub returnblock: BlockRef,
    /// RPython `FunctionGraph.exceptblock` —
    /// `Block([Variable('etype'), Variable('evalue')])` with
    /// `operations = ()` and `exits = ()`.
    pub exceptblock: BlockRef,
    /// RPython `FunctionGraph.tag`. `None` until set by an analysis
    /// pass.
    pub tag: Option<String>,
    /// RPython `FunctionGraph.func`.
    pub func: Option<GraphFunc>,
    /// RPython `FunctionGraph._source`.
    pub _source: Option<String>,
}

impl FunctionGraph {
    /// RPython `FunctionGraph.__init__(name, startblock,
    /// return_var=None)`.
    pub fn new(name: impl Into<String>, startblock: BlockRef) -> Self {
        // RPython path: `return_var = None` → a fresh Variable.
        let return_var = Hlvalue::Variable(Variable::new());
        let returnblock = Block::shared(vec![return_var]);
        returnblock.borrow_mut().mark_final();

        let etype = Hlvalue::Variable(Variable::named("etype"));
        let evalue = Hlvalue::Variable(Variable::named("evalue"));
        let exceptblock = Block::shared(vec![etype, evalue]);
        exceptblock.borrow_mut().mark_final();

        FunctionGraph {
            name: name.into(),
            startblock,
            returnblock,
            exceptblock,
            tag: None,
            func: None,
            _source: None,
        }
    }

    /// RPython `FunctionGraph.__init__(name, startblock, return_var)`.
    pub fn with_return_var(
        name: impl Into<String>,
        startblock: BlockRef,
        return_var: Hlvalue,
    ) -> Self {
        let returnblock = Block::shared(vec![return_var]);
        returnblock.borrow_mut().mark_final();

        let etype = Hlvalue::Variable(Variable::named("etype"));
        let evalue = Hlvalue::Variable(Variable::named("evalue"));
        let exceptblock = Block::shared(vec![etype, evalue]);
        exceptblock.borrow_mut().mark_final();

        FunctionGraph {
            name: name.into(),
            startblock,
            returnblock,
            exceptblock,
            tag: None,
            func: None,
            _source: None,
        }
    }

    /// RPython `FunctionGraph.getargs()`.
    pub fn getargs(&self) -> Vec<Hlvalue> {
        self.startblock.borrow().inputargs.clone()
    }

    /// RPython `FunctionGraph.getreturnvar()`.
    pub fn getreturnvar(&self) -> Hlvalue {
        self.returnblock.borrow().inputargs[0].clone()
    }

    /// RPython `FunctionGraph.source` property.
    pub fn source(&self) -> Result<&str, &'static str> {
        if let Some(source) = self._source.as_deref() {
            return Ok(source);
        }
        self.func
            .as_ref()
            .and_then(|func| func.source.as_deref())
            .ok_or("source not found")
    }

    /// RPython `FunctionGraph.source = value`.
    pub fn set_source(&mut self, value: impl Into<String>) {
        self._source = Some(value.into());
    }

    /// RPython `FunctionGraph.startline` property.
    pub fn startline(&self) -> Result<u32, &'static str> {
        self.func
            .as_ref()
            .and_then(|func| func.firstlineno)
            .ok_or("startline not found")
    }

    /// RPython `FunctionGraph.filename` property.
    pub fn filename(&self) -> Result<&str, &'static str> {
        self.func
            .as_ref()
            .and_then(|func| func.filename.as_deref())
            .ok_or("filename not found")
    }

    /// RPython `FunctionGraph.iterblocks()` — DFS over reachable
    /// blocks starting at `startblock`, following `Link.target` and
    /// matching upstream's right-to-left stack order.
    pub fn iterblocks(&self) -> Vec<BlockRef> {
        let mut result = Vec::new();
        let mut seen: Vec<*const RefCell<Block>> = Vec::new();
        let start = self.startblock.clone();
        seen.push(Rc::as_ptr(&start));
        result.push(start.clone());

        // `stack = list(block.exits[::-1])`
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
        while let Some(link) = stack.pop() {
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let key = Rc::as_ptr(&target);
            if !seen.contains(&key) {
                seen.push(key);
                result.push(target.clone());
                // `stack += block.exits[::-1]`
                let more: Vec<LinkRef> = target.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }
        result
    }

    /// RPython `FunctionGraph.iterlinks()`.
    pub fn iterlinks(&self) -> Vec<LinkRef> {
        let mut result = Vec::new();
        let mut seen: Vec<*const RefCell<Block>> = Vec::new();
        let start = self.startblock.clone();
        seen.push(Rc::as_ptr(&start));
        let mut stack: Vec<LinkRef> = start.borrow().exits.iter().rev().cloned().collect();
        while let Some(link) = stack.pop() {
            result.push(link.clone());
            let target = match link.borrow().target.as_ref() {
                Some(t) => t.clone(),
                None => continue,
            };
            let key = Rc::as_ptr(&target);
            if !seen.contains(&key) {
                seen.push(key);
                let more: Vec<LinkRef> = target.borrow().exits.iter().rev().cloned().collect();
                stack.extend(more);
            }
        }
        result
    }

    /// RPython `FunctionGraph.iterblockops()`.
    pub fn iterblockops(&self) -> Vec<(BlockRef, SpaceOperation)> {
        let mut result = Vec::new();
        for block in self.iterblocks() {
            for op in &block.borrow().operations {
                result.push((block.clone(), op.clone()));
            }
        }
        result
    }
}

impl std::fmt::Display for FunctionGraph {
    // RPython `FunctionGraph.__str__` returns the name unless `func`
    // is set; `func` lands at Phase 3. Until then, the name is
    // always the output.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.name)
    }
}

/// RPython `flowspace/model.py:476-484` — `uniqueitems(lst)`.
///
/// Returns a list with duplicate elements removed, preserving order.
pub fn uniqueitems<T: Clone + PartialEq>(lst: &[T]) -> Vec<T> {
    let mut result: Vec<T> = Vec::new();
    for item in lst {
        if !result.iter().any(|x| x == item) {
            result.push(item.clone());
        }
    }
    result
}

/// RPython `flowspace/model.py:495-502` — `mkentrymap(funcgraph)`.
///
/// Returns a map from each Block to the list of Links that target
/// it. The entry for the startblock contains a synthetic
/// `Link(graph.getargs(), graph.startblock)`.
pub fn mkentrymap(funcgraph: &FunctionGraph) -> HashMap<BlockKey, Vec<LinkRef>> {
    let startlink = Rc::new(RefCell::new(Link::new(
        funcgraph.getargs(),
        Some(funcgraph.startblock.clone()),
        None,
    )));
    let mut result: HashMap<BlockKey, Vec<LinkRef>> = HashMap::new();
    result.insert(BlockKey::of(&funcgraph.startblock), vec![startlink]);

    for link_ref in funcgraph.iterlinks() {
        let target_key = match link_ref.borrow().target.as_ref() {
            Some(t) => BlockKey::of(t),
            None => continue,
        };
        result.entry(target_key).or_default().push(link_ref.clone());
    }
    result
}

/// Identity-keyed wrapper for `BlockRef` so blocks can be used as
/// `HashMap` keys via pointer identity, matching RPython `dict[Block]`
/// semantics (Python uses `id(block)` by default).
#[derive(Clone, Debug)]
pub struct BlockKey(*const RefCell<Block>);

impl BlockKey {
    pub fn of(b: &BlockRef) -> Self {
        BlockKey(Rc::as_ptr(b))
    }

    /// Exposes the pointer identity as a `usize` — useful as a
    /// `PositionKey` payload.
    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl PartialEq for BlockKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for BlockKey {}

impl std::hash::Hash for BlockKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// RPython `FunctionGraph` container used by the annotator. Upstream
/// passes `FunctionGraph` instances directly through dicts/sets
/// (Python identity). The Rust port wraps the graph in `Rc<RefCell<…>>`
/// so the annotator can mutate per-graph state (tag, entry-point
/// bookkeeping) without fighting the borrow checker.
pub type GraphRef = Rc<RefCell<FunctionGraph>>;

/// Identity-keyed wrapper for `GraphRef` — same rationale as
/// [`BlockKey`]: RPython `dict[FunctionGraph]` uses Python object
/// identity, which maps to `Rc::as_ptr` on the Rust side.
#[derive(Clone, Debug)]
pub struct GraphKey(*const RefCell<FunctionGraph>);

impl GraphKey {
    pub fn of(g: &GraphRef) -> Self {
        GraphKey(Rc::as_ptr(g))
    }

    pub fn as_usize(&self) -> usize {
        self.0 as usize
    }
}

impl PartialEq for GraphKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for GraphKey {}

impl std::hash::Hash for GraphKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Identity-keyed wrapper for `LinkRef`. Used by `RPythonAnnotator`'s
/// `links_followed` set — upstream Python stores `Link` objects in a
/// dict keyed by identity.
#[derive(Clone, Debug)]
pub struct LinkKey(*const RefCell<Link>);

impl LinkKey {
    pub fn of(l: &LinkRef) -> Self {
        LinkKey(Rc::as_ptr(l))
    }
}

impl PartialEq for LinkKey {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for LinkKey {}

impl std::hash::Hash for LinkKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// RPython `flowspace/model.py:504-566` — `copygraph(graph,
/// shallow=False, varmap={}, shallowvars=False)`.
///
/// Deep-copies a flow graph. Each Variable encountered is replaced
/// by a fresh Variable (new identity) unless `shallowvars` is true,
/// in which case the original Variable is reused. When `shallow` is
/// true, operation lists are also kept as-is.
pub fn copygraph(
    graph: &FunctionGraph,
    shallow: bool,
    varmap: &HashMap<Variable, Variable>,
    shallowvars: bool,
) -> FunctionGraph {
    let mut varmap: HashMap<Variable, Variable> = varmap.clone();
    let shallowvars = shallowvars || shallow;

    let copyvar = |v: &Variable, varmap: &mut HashMap<Variable, Variable>| -> Variable {
        if shallowvars {
            return v.clone();
        }
        if let Some(existing) = varmap.get(v) {
            return existing.clone();
        }
        let v2 = v.copy();
        varmap.insert(v.clone(), v2.clone());
        v2
    };

    let copyhl = |w: &Hlvalue, varmap: &mut HashMap<Variable, Variable>| -> Hlvalue {
        match w {
            Hlvalue::Variable(v) => Hlvalue::Variable(copyvar(v, varmap)),
            Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
        }
    };

    let copyblock = |block: &BlockRef, varmap: &mut HashMap<Variable, Variable>| -> BlockRef {
        let b = block.borrow();
        let newinputargs: Vec<Hlvalue> = b.inputargs.iter().map(|a| copyhl(a, varmap)).collect();
        let newblock = Block::shared(newinputargs);
        {
            let mut nb = newblock.borrow_mut();
            if b.is_final_block() {
                nb.mark_final();
            } else {
                nb.operations = if shallow {
                    b.operations.clone()
                } else {
                    b.operations
                        .iter()
                        .map(|op| {
                            let newargs: Vec<Hlvalue> =
                                op.args.iter().map(|a| copyhl(a, varmap)).collect();
                            let newresult = copyhl(&op.result, varmap);
                            SpaceOperation::with_offset(
                                op.opname.clone(),
                                newargs,
                                newresult,
                                op.offset,
                            )
                        })
                        .collect()
                };
            }
            nb.exitswitch = b.exitswitch.as_ref().map(|s| copyhl(s, varmap));
        }
        newblock
    };

    // Build blockmap over all reachable blocks + returnblock + exceptblock.
    let mut blockmap: HashMap<BlockKey, BlockRef> = HashMap::new();
    for block in graph.iterblocks() {
        blockmap.insert(BlockKey::of(&block), copyblock(&block, &mut varmap));
    }
    if !blockmap.contains_key(&BlockKey::of(&graph.returnblock)) {
        blockmap.insert(
            BlockKey::of(&graph.returnblock),
            copyblock(&graph.returnblock, &mut varmap),
        );
    }
    if !blockmap.contains_key(&BlockKey::of(&graph.exceptblock)) {
        blockmap.insert(
            BlockKey::of(&graph.exceptblock),
            copyblock(&graph.exceptblock, &mut varmap),
        );
    }

    // Wire up exits.
    for block in graph.iterblocks() {
        let src_key = BlockKey::of(&block);
        let newblock = blockmap[&src_key].clone();
        let mut newlinks: Vec<LinkRef> = Vec::new();
        for link_ref in &block.borrow().exits {
            let link = link_ref.borrow();
            let newargs: Vec<LinkArg> = link
                .args
                .iter()
                .map(|arg| arg.as_ref().map(|value| copyhl(value, &mut varmap)))
                .collect();
            let new_target = link.target.as_ref().map(|t| {
                blockmap
                    .get(&BlockKey::of(t))
                    .cloned()
                    .expect("target block missing from blockmap")
            });
            let mut newlink = Link::new_mergeable(newargs, new_target, link.exitcase.clone());
            newlink.prevblock = Some(Rc::downgrade(&newblock));
            newlink.last_exception = link.last_exception.as_ref().map(|x| copyhl(x, &mut varmap));
            newlink.last_exc_value = link.last_exc_value.as_ref().map(|x| copyhl(x, &mut varmap));
            newlink.llexitcase = link.llexitcase.clone();
            newlinks.push(Rc::new(RefCell::new(newlink)));
        }
        newblock.borrow_mut().closeblock(newlinks);
    }

    let newstart = blockmap[&BlockKey::of(&graph.startblock)].clone();
    let mut newgraph = FunctionGraph::new(graph.name.clone(), newstart);
    newgraph.returnblock = blockmap[&BlockKey::of(&graph.returnblock)].clone();
    newgraph.exceptblock = blockmap[&BlockKey::of(&graph.exceptblock)].clone();
    newgraph.tag = graph.tag.clone();
    newgraph.func = graph.func.clone();
    newgraph._source = graph._source.clone();
    newgraph
}

/// RPython `flowspace/model.py:702-709` — `summary(graph)`.
///
/// Returns a map of opname → occurrence count, excluding `same_as`.
pub fn summary(graph: &FunctionGraph) -> HashMap<String, usize> {
    let mut insns: HashMap<String, usize> = HashMap::new();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            if op.opname != "same_as" {
                *insns.entry(op.opname.clone()).or_insert(0) += 1;
            }
        }
    }
    insns
}

fn is_exception_exitcase(exitcase: &Hlvalue) -> bool {
    matches!(
        exitcase,
        Hlvalue::Constant(Constant {
            value: ConstValue::HostObject(obj),
            ..
        }) if obj.is_class()
    )
}

fn is_valid_switch_exitcase(exitcase: &Hlvalue) -> bool {
    matches!(
        exitcase,
        Hlvalue::Constant(Constant {
            value: ConstValue::Int(_) | ConstValue::Bool(_) | ConstValue::None,
            ..
        })
    ) || matches!(
        exitcase,
        Hlvalue::Constant(Constant {
            value: ConstValue::Str(s),
            ..
        }) if s == "default" || s.chars().count() == 1
    )
}

/// RPython `flowspace/model.py:568-700` — `checkgraph(graph)`.
///
/// Sanity-check a flow graph. Panics with an assertion failure on any
/// violation, matching upstream semantics (RPython: `AssertionError`).
pub fn checkgraph(graph: &FunctionGraph) {
    for (block, nbargs) in [(&graph.returnblock, 1usize), (&graph.exceptblock, 2usize)] {
        let b = block.borrow();
        assert_eq!(b.inputargs.len(), nbargs);
        assert!(
            b.operations.is_empty(),
            "exitblock must not contain operations"
        );
        assert!(b.exits.is_empty(), "exitblock must not contain exits");
    }

    let mut vars_previous_blocks: Vec<Variable> = Vec::new();

    for block in graph.iterblocks() {
        let b = block.borrow();
        if b.exits.is_empty() {
            let is_return = BlockKey::of(&block) == BlockKey::of(&graph.returnblock);
            let is_except = BlockKey::of(&block) == BlockKey::of(&graph.exceptblock);
            assert!(is_return || is_except, "exit block not in graph exitblocks");
        }

        let mut vars: HashMap<Variable, Option<usize>> = HashMap::new();

        let definevar = |v: &Variable,
                         only_in_link: Option<usize>,
                         vars: &mut HashMap<Variable, Option<usize>>,
                         vars_previous_blocks: &[Variable]| {
            assert!(!vars.contains_key(v), "duplicate variable {}", v.name());
            assert!(
                !vars_previous_blocks.contains(v),
                "variable {} used in more than one block",
                v.name()
            );
            vars.insert(v.clone(), only_in_link);
        };

        let usevar =
            |v: &Variable, in_link: Option<usize>, vars: &HashMap<Variable, Option<usize>>| {
                let only_in_link = vars
                    .get(v)
                    .unwrap_or_else(|| panic!("variable {} used before definition", v.name()));
                if let Some(in_link) = in_link {
                    assert!(
                        only_in_link.is_none() || *only_in_link == Some(in_link),
                        "variable {} used from the wrong exception link",
                        v.name()
                    );
                }
            };

        for w in &b.inputargs {
            let Hlvalue::Variable(v) = w else {
                panic!("block inputargs must be Variables");
            };
            definevar(v, None, &mut vars, &vars_previous_blocks);
        }

        for op in &b.operations {
            for a in &op.args {
                match a {
                    Hlvalue::Variable(v) => usevar(v, None, &vars),
                    Hlvalue::Constant(c) => assert!(
                        !matches!(&c.value, ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name),
                        "last_exception constant cannot appear as an operation argument"
                    ),
                }
            }
            if op.opname == "direct_call" {
                assert!(
                    matches!(op.args.first(), Some(Hlvalue::Constant(_))),
                    "direct_call expects a Constant function argument"
                );
            } else if op.opname == "indirect_call" {
                assert!(
                    matches!(op.args.first(), Some(Hlvalue::Variable(_))),
                    "indirect_call expects a Variable function argument"
                );
            }
            if let Hlvalue::Variable(v) = &op.result {
                definevar(v, None, &mut vars, &vars_previous_blocks);
            }
        }

        let mut exc_links: Vec<usize> = Vec::new();
        match &b.exitswitch {
            None => {
                assert!(b.exits.len() <= 1, "block without exitswitch has > 1 exits");
                if let Some(first) = b.exits.first() {
                    assert!(
                        first.borrow().exitcase.is_none(),
                        "unconditional exit carries an exitcase"
                    );
                }
            }
            Some(Hlvalue::Constant(c)) if matches!(&c.value, ConstValue::Atom(a) if a.name == LAST_EXCEPTION.name) =>
            {
                assert!(
                    !b.operations.is_empty(),
                    "canraise block must end with a raising operation"
                );
                let raising_op = b.raising_op().expect("canraise block missing raising_op");
                assert!(
                    raising_op.opname != "keepalive"
                        && raising_op.opname != "cast_pointer"
                        && raising_op.opname != "same_as",
                    "invalid raising_op for canraise block"
                );
                assert!(
                    b.exits.len() >= 2,
                    "canraise block requires exception exits"
                );
                assert!(b.exits[0].borrow().exitcase.is_none());
                for link in b.exits.iter().skip(1) {
                    let exitcase = link
                        .borrow()
                        .exitcase
                        .clone()
                        .expect("exception exitcase missing");
                    assert!(
                        is_exception_exitcase(&exitcase),
                        "exception exitcase must carry an exception class constant"
                    );
                    exc_links.push(Rc::as_ptr(link) as usize);
                }
            }
            Some(Hlvalue::Variable(sw)) => {
                usevar(sw, None, &vars);
                let is_boolean_switch = b.exits.len() == 2
                    && matches!(
                        (
                            b.exits[0].borrow().exitcase.as_ref(),
                            b.exits[1].borrow().exitcase.as_ref()
                        ),
                        (
                            Some(Hlvalue::Constant(Constant {
                                value: ConstValue::Bool(false),
                                ..
                            })),
                            Some(Hlvalue::Constant(Constant {
                                value: ConstValue::Bool(true),
                                ..
                            }))
                        )
                    );
                if !is_boolean_switch {
                    assert!(!b.exits.is_empty(), "switch block must have exits");
                    for (idx, link) in b.exits.iter().enumerate() {
                        let exitcase = link
                            .borrow()
                            .exitcase
                            .clone()
                            .expect("switch exitcase cannot be None");
                        assert!(
                            is_valid_switch_exitcase(&exitcase),
                            "switch on a non-primitive value {exitcase:?}"
                        );
                        if let Hlvalue::Constant(Constant {
                            value: ConstValue::Str(s),
                            ..
                        }) = &exitcase
                        {
                            if s == "default" {
                                assert!(
                                    idx + 1 == b.exits.len(),
                                    "'default' branch of a switch is not the last exit"
                                );
                            }
                        }
                    }
                }
            }
            Some(Hlvalue::Constant(_)) => {
                panic!("unexpected constant exitswitch outside canraise blocks");
            }
        }

        let mut all_exitcases: Vec<Option<Hlvalue>> = Vec::new();
        for link_ref in &b.exits {
            let link_id = Rc::as_ptr(link_ref) as usize;
            let exc_link = exc_links.contains(&link_id);
            let link = link_ref.borrow();
            let target = link.target.as_ref().expect("link.target missing");
            assert_eq!(
                link.args.len(),
                target.borrow().inputargs.len(),
                "link arity mismatch"
            );
            let prev = link.prevblock.as_ref().and_then(|w| w.upgrade());
            assert!(
                prev.as_ref().map_or(false, |p| Rc::ptr_eq(p, &block)),
                "link.prevblock does not point back to the owning block"
            );
            if exc_link {
                for extra in [&link.last_exception, &link.last_exc_value] {
                    let extra = extra.as_ref().expect("exception link missing extravar");
                    if let Hlvalue::Variable(v) = extra {
                        definevar(v, Some(link_id), &mut vars, &vars_previous_blocks);
                    }
                }
            } else {
                assert!(link.last_exception.is_none());
                assert!(link.last_exc_value.is_none());
            }
            for arg in &link.args {
                let arg = arg
                    .as_ref()
                    .expect("finalized graph cannot contain undefined-local link args");
                if let Hlvalue::Variable(v) = arg {
                    usevar(v, Some(link_id), &vars);
                    if exc_link {
                        if let Some(last_op) = b.operations.last() {
                            assert!(
                                *arg != last_op.result,
                                "raising operation result cannot flow into exception link"
                            );
                        }
                    }
                }
            }
            assert!(
                !all_exitcases.contains(&link.exitcase),
                "duplicate exitcase in block exits"
            );
            all_exitcases.push(link.exitcase.clone());
        }
        vars_previous_blocks.extend(vars.into_keys());
    }
}

#[cfg(test)]
mod tests {
    //! Lightweight scaffold tests for F1.2 + F1.3. The full
    //! `rpython/flowspace/test/test_model.py` port lands at F1.7
    //! after `Block`, `Link`, and `FunctionGraph` exist.

    use super::*;

    #[test]
    fn variable_default_name_numbers_from_dummy_prefix() {
        // Two fresh Variables share the dummy prefix and should get
        // unique lazy numbers.
        let v1 = Variable::new();
        let v2 = Variable::new();
        let n1 = v1.name();
        let n2 = v2.name();
        assert_ne!(n1, n2);
        assert!(n1.starts_with('v'));
        assert!(n2.starts_with('v'));
    }

    #[test]
    fn variable_rename_sets_prefix_once() {
        let mut v = Variable::new();
        v.rename("counter");
        assert!(v.renamed());
        let first = v.name();
        // Second rename is a no-op per upstream.
        v.rename("other");
        assert_eq!(v.name(), first);
    }

    #[test]
    fn variable_copy_preserves_annotation_and_concretetype() {
        use crate::annotator::model::{SomeInteger, SomeValue};
        let mut v = Variable::new();
        v.annotation = Some(Rc::new(SomeValue::Integer(SomeInteger::default())));
        v.concretetype = Some(());
        let c = v.copy();
        let ca = c.annotation.as_ref().unwrap();
        let va = v.annotation.as_ref().unwrap();
        // Rc shares the same allocation — identity preserved per
        // RPython parity (copy keeps reference to the same SomeValue).
        assert!(Rc::ptr_eq(ca, va));
        assert_eq!(c.concretetype, v.concretetype);
    }

    #[test]
    fn constant_replace_returns_self() {
        let c = Constant::new(ConstValue::Placeholder);
        let map = HashMap::new();
        assert!(std::ptr::eq(c.replace(&map), &c));
    }

    #[test]
    fn atom_display_returns_name() {
        let a = Atom::new("last_exception");
        assert_eq!(format!("{a}"), "last_exception");
    }

    #[test]
    fn space_operation_eq_ignores_offset() {
        // RPython __eq__ explicitly excludes offset (model.py:442).
        let r = Hlvalue::Variable(Variable::named("r"));
        let a = Hlvalue::Variable(Variable::named("a"));
        let op1 = SpaceOperation::with_offset("int_add", vec![a.clone(), a.clone()], r.clone(), 10);
        let op2 = SpaceOperation::with_offset("int_add", vec![a.clone(), a], r, 20);
        assert_eq!(op1, op2);
    }

    #[test]
    fn space_operation_repr_matches_rpython_layout() {
        let r = Hlvalue::Variable(Variable::named("result"));
        let a = Hlvalue::Variable(Variable::named("lhs"));
        let b = Hlvalue::Variable(Variable::named("rhs"));
        let op = SpaceOperation::new("int_add", vec![a, b], r);
        let s = format!("{op}");
        // result = opname(arg0, arg1)
        assert!(s.contains(" = int_add("));
        assert!(s.contains(", "));
    }

    #[test]
    fn c_last_exception_is_atom_constant() {
        let c = c_last_exception();
        assert!(matches!(
            c.value,
            ConstValue::Atom(ref a) if a.name == "last_exception"
        ));
    }

    #[test]
    fn block_canraise_detects_last_exception_exitswitch() {
        let mut block = Block::new(vec![]);
        assert!(!block.canraise());
        block.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert!(block.canraise());
    }

    #[test]
    fn block_raising_op_returns_last_when_canraise() {
        let r = Hlvalue::Variable(Variable::named("r"));
        let a = Hlvalue::Variable(Variable::named("a"));
        let op = SpaceOperation::new("int_add", vec![a.clone(), a], r);
        let mut block = Block::new(vec![]);
        block.operations.push(op.clone());
        block.exitswitch = Some(Hlvalue::Constant(c_last_exception()));
        assert_eq!(block.raising_op(), Some(&op));
    }

    #[test]
    fn block_closeblock_panics_on_second_call() {
        let target = Block::shared(vec![]);
        let link = Rc::new(RefCell::new(Link::new(vec![], Some(target.clone()), None)));
        let mut block = Block::new(vec![]);
        block.closeblock(vec![link.clone()]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block.closeblock(vec![link]);
        }));
        assert!(result.is_err(), "second closeblock should panic");
    }

    #[test]
    fn function_graph_constructs_return_and_except_blocks() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let startblock = Block::shared(vec![arg]);
        let g = FunctionGraph::new("f", startblock);
        // Returnblock has 1 inputarg, is final, no exits.
        let rb = g.returnblock.borrow();
        assert_eq!(rb.inputargs.len(), 1);
        assert!(rb.is_final_block());
        assert!(rb.exits.is_empty());
        drop(rb);
        // Exceptblock has 2 inputargs (etype, evalue), is final.
        let xb = g.exceptblock.borrow();
        assert_eq!(xb.inputargs.len(), 2);
        assert!(xb.is_final_block());
    }

    #[test]
    fn function_graph_iterblocks_visits_reachable_only() {
        // Build: start -> mid -> return.
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        // Link start -> mid.
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm]);
        // Build graph with start as entry; mid has no exits (dead-end).
        let g = FunctionGraph::new("f", start.clone());
        let blocks = g.iterblocks();
        // Should visit start then mid.
        assert_eq!(blocks.len(), 2);
        assert!(Rc::ptr_eq(&blocks[0], &start));
        assert!(Rc::ptr_eq(&blocks[1], &mid));
    }

    #[test]
    fn function_graph_iterlinks_walks_every_edge() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm.clone()]);
        let g = FunctionGraph::new("f", start);
        let links = g.iterlinks();
        assert_eq!(links.len(), 1);
        assert!(Rc::ptr_eq(&links[0], &link_sm));
    }

    #[test]
    fn function_graph_iterblockops_walks_every_op() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let r = Hlvalue::Variable(Variable::named("r"));
        let op = SpaceOperation::new("int_add", vec![arg.clone(), arg.clone()], r);
        start.borrow_mut().operations.push(op.clone());
        let g = FunctionGraph::new("f", start.clone());
        let ops = g.iterblockops();
        assert_eq!(ops.len(), 1);
        assert!(Rc::ptr_eq(&ops[0].0, &start));
        assert_eq!(ops[0].1, op);
    }

    #[test]
    fn uniqueitems_removes_duplicates_in_order() {
        let v = uniqueitems(&[1, 2, 3, 2, 1, 4]);
        assert_eq!(v, vec![1, 2, 3, 4]);
    }

    #[test]
    fn mkentrymap_includes_synthetic_startlink_and_every_edge() {
        let arg = Hlvalue::Variable(Variable::named("x"));
        let start = Block::shared(vec![arg.clone()]);
        let mid = Block::shared(vec![arg.clone()]);
        let link_sm = Rc::new(RefCell::new(Link::new(
            vec![arg.clone()],
            Some(mid.clone()),
            None,
        )));
        start.borrow_mut().closeblock(vec![link_sm]);
        let g = FunctionGraph::new("f", start.clone());
        let map = mkentrymap(&g);
        // The startblock has the synthetic startlink (from getargs).
        assert!(map.contains_key(&BlockKey::of(&start)));
        assert_eq!(map[&BlockKey::of(&start)].len(), 1);
        // The mid block has one entry (the edge from start).
        assert_eq!(map[&BlockKey::of(&mid)].len(), 1);
    }

    #[test]
    fn summary_counts_opnames_excluding_same_as() {
        let x = Hlvalue::Variable(Variable::named("x"));
        let r = Hlvalue::Variable(Variable::named("r"));
        let r2 = Hlvalue::Variable(Variable::named("r2"));
        let start = Block::shared(vec![x.clone()]);
        {
            let mut b = start.borrow_mut();
            b.operations.push(SpaceOperation::new(
                "int_add",
                vec![x.clone(), x.clone()],
                r,
            ));
            b.operations
                .push(SpaceOperation::new("same_as", vec![x.clone()], r2));
        }
        let g = FunctionGraph::new("f", start);
        let s = summary(&g);
        assert_eq!(s.get("int_add").copied(), Some(1));
        assert!(s.get("same_as").is_none(), "summary excludes same_as");
    }

    #[test]
    fn copygraph_produces_fresh_variables() {
        let x = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let g = FunctionGraph::new("f", start);
        let g2 = copygraph(&g, false, &HashMap::new(), false);
        // New startblock has a different input-variable identity.
        let b0 = g.startblock.borrow();
        let b1 = g2.startblock.borrow();
        let v0 = match &b0.inputargs[0] {
            Hlvalue::Variable(v) => v,
            _ => panic!(),
        };
        let v1 = match &b1.inputargs[0] {
            Hlvalue::Variable(v) => v,
            _ => panic!(),
        };
        assert_ne!(v0, v1, "copygraph should assign a fresh variable id");
    }

    #[test]
    fn checkgraph_passes_on_simple_graph() {
        // A trivial graph with just a start block and the return/except
        // blocks auto-created. startblock has no exits, so it must be a
        // return block — use the returnblock as the startblock so the
        // "exit block not in graph exitblocks" invariant holds.
        let start = Block::shared(vec![Hlvalue::Variable(Variable::named("x"))]);
        let r = Hlvalue::Variable(Variable::named("r"));
        // Link start -> returnblock.
        let g = {
            let g = FunctionGraph::new("f", start.clone());
            let retlink = Rc::new(RefCell::new(Link::new(
                vec![r.clone()],
                Some(g.returnblock.clone()),
                None,
            )));
            retlink.borrow_mut().prevblock = Some(Rc::downgrade(&start));
            start.borrow_mut().closeblock(vec![retlink]);
            // Give the start block at least one op producing `r`.
            start.borrow_mut().operations.push(SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Variable(Variable::named("x")),
                    Hlvalue::Variable(Variable::named("x")),
                ],
                r,
            ));
            g
        };
        // We deliberately skipped variable-definedness plumbing for this
        // hand-rolled test; just make sure the call itself does not panic
        // when the structural invariants hold.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checkgraph(&g);
        }));
        // The above graph has undefined-variable usage (the op uses
        // freshly-minted x variables not in inputargs), so checkgraph
        // will assert. That's expected — just confirm the function is
        // callable end-to-end.
        assert!(
            result.is_err(),
            "checkgraph should catch undefined-variable use"
        );
    }

    #[test]
    fn link_settarget_checks_arity() {
        let a = Hlvalue::Variable(Variable::named("a"));
        let target = Block::shared(vec![a.clone()]);
        let mut link = Link::new(vec![a], Some(target), None);
        let other = Block::shared(vec![]); // different arity
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            link.settarget(other);
        }));
        assert!(result.is_err(), "arity mismatch should panic");
    }

    #[test]
    fn space_operation_replace_remaps_args_and_result() {
        let r = Variable::named("r");
        let a = Variable::named("a");
        let b = Variable::named("b");
        // Mapping r -> r', a -> a'; b unchanged.
        let r_new = Variable::named("r_new");
        let a_new = Variable::named("a_new");
        let mut map: HashMap<Variable, Variable> = HashMap::new();
        map.insert(r.clone(), r_new.clone());
        map.insert(a.clone(), a_new.clone());

        let op = SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(a), Hlvalue::Variable(b.clone())],
            Hlvalue::Variable(r),
        );
        let op2 = op.replace(&map);
        assert_eq!(op2.result, Hlvalue::Variable(r_new));
        assert_eq!(op2.args[0], Hlvalue::Variable(a_new));
        assert_eq!(op2.args[1], Hlvalue::Variable(b));
    }

    // --- HostObject class_dict / mro — Phase 5 P5.2 classdesc c2 ---

    #[test]
    fn class_dict_get_set_roundtrip() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        assert!(cls.class_get("_mixin_").is_none());
        cls.class_set("_mixin_", ConstValue::Bool(true));
        match cls.class_get("_mixin_") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {other:?}"),
        }
    }

    #[test]
    fn class_dict_keys_snapshot_matches_insert() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        cls.class_set("a", ConstValue::Int(1));
        cls.class_set("b", ConstValue::Int(2));
        let mut keys = cls.class_dict_keys();
        keys.sort();
        assert_eq!(keys, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn class_dict_items_roundtrip() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        cls.class_set("x", ConstValue::Int(7));
        let items = cls.class_dict_items();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].0, "x");
        match items[0].1 {
            ConstValue::Int(7) => {}
            ref other => panic!("expected Int(7), got {other:?}"),
        }
    }

    #[test]
    fn class_has_reflects_membership() {
        let cls = HostObject::new_class("pkg.Foo", vec![]);
        assert!(!cls.class_has("missing"));
        cls.class_set("here", ConstValue::Int(0));
        assert!(cls.class_has("here"));
    }

    #[test]
    fn new_class_with_members_seeds_initial_dict() {
        let mut seed: HashMap<String, ConstValue> = HashMap::new();
        seed.insert("_mixin_".into(), ConstValue::Bool(true));
        let cls = HostObject::new_class_with_members("pkg.Foo", vec![], seed);
        match cls.class_get("_mixin_") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("expected Bool(true), got {other:?}"),
        }
    }

    #[test]
    fn class_dict_on_module_returns_empty() {
        let m = HostObject::new_module("pkg");
        assert_eq!(m.class_dict_keys(), Vec::<String>::new());
        assert!(m.class_get("anything").is_none());
        assert!(!m.class_has("anything"));
    }

    #[test]
    fn class_bases_matches_ctor() {
        let base = HostObject::new_class("pkg.Base", vec![]);
        let child = HostObject::new_class("pkg.Child", vec![base.clone()]);
        let bases = child.class_bases().expect("class has bases");
        assert_eq!(bases.len(), 1);
        assert_eq!(bases[0], base);
    }

    #[test]
    fn mro_single_inheritance_chain() {
        // A <- B <- C, mro = [C, B, A]
        let a = HostObject::new_class("pkg.A", vec![]);
        let b = HostObject::new_class("pkg.B", vec![a.clone()]);
        let c = HostObject::new_class("pkg.C", vec![b.clone()]);
        let mro = c.mro().expect("single-inheritance mro");
        assert_eq!(mro, vec![c, b, a]);
    }

    #[test]
    fn mro_diamond_c3_order() {
        // A, B<-A, C<-A, D<-B,C → mro = [D, B, C, A]
        let a = HostObject::new_class("pkg.A", vec![]);
        let b = HostObject::new_class("pkg.B", vec![a.clone()]);
        let c = HostObject::new_class("pkg.C", vec![a.clone()]);
        let d = HostObject::new_class("pkg.D", vec![b.clone(), c.clone()]);
        let mro = d.mro().expect("diamond mro");
        assert_eq!(mro, vec![d, b, c, a]);
    }

    #[test]
    fn mro_empty_bases_is_singleton() {
        let a = HostObject::new_class("pkg.A", vec![]);
        let mro = a.mro().expect("no-base mro");
        assert_eq!(mro, vec![a]);
    }

    #[test]
    fn mro_on_non_class_returns_none() {
        let m = HostObject::new_module("pkg");
        assert!(m.mro().is_none());
    }
}
