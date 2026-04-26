//! Port slice of `rpython/translator/c/database.py`.
//!
//! This module starts with the constructor and the small methods needed by
//! `genc.py:87-138 CBuilder.build_database`. The full node factory and C
//! rendering machinery remain to be ported from `database.py`.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;

use crate::translator::rtyper::lltypesystem::lltype::{_ptr, _ptr_obj};
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// RPython `gc.name_to_gcpolicy` values consumed by `genc.py:161-167`.
#[derive(Clone, Debug)]
pub enum GcPolicyClass {
    Boehm,
    Refcounting,
    None,
    FrameworkShadowStack,
    /// Unrecognised name from `translation.gctransformer` config —
    /// upstream `name_to_gcpolicy` would have raised `KeyError` here;
    /// the local port surfaces the raw name so the failure is at the
    /// downstream consumer instead of `from_name`.
    UnknownName(String),
    /// Upstream `genc.py:161-167 get_gcpolicyclass`: when a custom
    /// `gcpolicy` was supplied to `CBuilder.__init__`, upstream
    /// returns the policy *object* itself. The Rust port keeps the
    /// `Rc<dyn Any>` so the policy's methods/state survive the call;
    /// dropping to a pointer-string was a NEW-DEVIATION that erased
    /// the live state.
    Custom(Rc<dyn Any>),
}

impl PartialEq for GcPolicyClass {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (GcPolicyClass::Boehm, GcPolicyClass::Boehm)
            | (GcPolicyClass::Refcounting, GcPolicyClass::Refcounting)
            | (GcPolicyClass::None, GcPolicyClass::None)
            | (GcPolicyClass::FrameworkShadowStack, GcPolicyClass::FrameworkShadowStack) => true,
            (GcPolicyClass::UnknownName(a), GcPolicyClass::UnknownName(b)) => a == b,
            // `Custom` carries an `Rc<dyn Any>` whose identity is its
            // pointer — match upstream's `is`-equality on the policy
            // object.
            (GcPolicyClass::Custom(a), GcPolicyClass::Custom(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for GcPolicyClass {}

impl GcPolicyClass {
    pub fn from_name(name: &str) -> Self {
        match name {
            "boehm" => GcPolicyClass::Boehm,
            "ref" => GcPolicyClass::Refcounting,
            "none" => GcPolicyClass::None,
            "framework+shadowstack" => GcPolicyClass::FrameworkShadowStack,
            other => GcPolicyClass::UnknownName(other.to_string()),
        }
    }
}

/// Small port of `rpython/translator/c/gc.py:9-41 BasicGcPolicy`.
#[derive(Clone, Debug)]
pub struct BasicGcPolicy {
    pub policy_class: GcPolicyClass,
    pub thread_enabled: bool,
}

impl BasicGcPolicy {
    pub fn new(policy_class: GcPolicyClass, thread_enabled: bool) -> Self {
        Self {
            policy_class,
            thread_enabled,
        }
    }

    /// RPython `BasicGcPolicy.gc_startup_code()` at `gc.py:40-41`.
    pub fn gc_startup_code(&self) -> Vec<Rc<dyn Any>> {
        Vec::new()
    }

    /// RPython `BasicGcPolicy.compilation_info()` at `gc.py:26-35`.
    /// GC-policy-specific ECI production has not landed yet, so this keeps
    /// the observable "maybe some compilation info" slot empty.
    pub fn compilation_info(&self) -> Option<Rc<dyn Any>> {
        None
    }
}

/// Port of `rpython/translator/c/database.py:27-71 LowLevelDatabase.__init__`.
#[derive(Clone)]
pub struct LowLevelDatabase {
    pub translator: Option<Rc<TranslationContext>>,
    pub standalone: bool,
    pub sandbox: bool,
    pub split_gc_address_space: bool,
    pub reverse_debugger: bool,
    pub gcpolicy: BasicGcPolicy,
    pub gchooks: Option<Rc<dyn Any>>,
    pub exctransformer: Option<Rc<dyn Any>>,

    /// RPython dict/list slots at `database.py:48-55`.
    pub structdefnodes: RefCell<HashMap<String, Rc<dyn Any>>>,
    pub pendingsetupnodes: RefCell<Vec<Rc<dyn Any>>>,
    pub containernodes: RefCell<HashMap<usize, Rc<dyn Any>>>,
    pub containerlist: RefCell<Vec<Rc<dyn Any>>>,
    pub idelayedfunctionnames: RefCell<HashMap<u64, String>>,
    pub delayedfunctionptrs: RefCell<Vec<Rc<dyn Any>>>,
    pub completedcontainers: Cell<usize>,
    pub containerstats: RefCell<HashMap<String, usize>>,

    /// RPython `late_initializations`, `completed`,
    /// `instrument_ncounter`, `all_field_names` at `database.py:57-71`.
    pub late_initializations: RefCell<Vec<Rc<dyn Any>>>,
    pub completed: Cell<bool>,
    pub instrument_ncounter: Cell<usize>,
    pub all_field_names: RefCell<Option<Vec<String>>>,
}

impl LowLevelDatabase {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        translator: Option<Rc<TranslationContext>>,
        standalone: bool,
        gcpolicyclass: GcPolicyClass,
        gchooks: Option<Rc<dyn Any>>,
        exctransformer: Option<Rc<dyn Any>>,
        thread_enabled: bool,
        sandbox: bool,
        split_gc_address_space: bool,
        reverse_debugger: bool,
        countfieldaccess: bool,
    ) -> Self {
        let all_field_names = if countfieldaccess {
            Some(Vec::new())
        } else {
            None
        };
        Self {
            translator,
            standalone,
            sandbox,
            split_gc_address_space,
            reverse_debugger,
            gcpolicy: BasicGcPolicy::new(gcpolicyclass, thread_enabled),
            gchooks,
            exctransformer,
            structdefnodes: RefCell::new(HashMap::new()),
            pendingsetupnodes: RefCell::new(Vec::new()),
            containernodes: RefCell::new(HashMap::new()),
            containerlist: RefCell::new(Vec::new()),
            idelayedfunctionnames: RefCell::new(HashMap::new()),
            delayedfunctionptrs: RefCell::new(Vec::new()),
            completedcontainers: Cell::new(0),
            containerstats: RefCell::new(HashMap::new()),
            late_initializations: RefCell::new(Vec::new()),
            completed: Cell::new(false),
            instrument_ncounter: Cell::new(0),
            all_field_names: RefCell::new(all_field_names),
        }
    }

    /// Port slice of `LowLevelDatabase.get(obj)` at `database.py:181-...`.
    /// For function pointers, RPython returns the C function name; the
    /// local `lltype.getfunctionptr` port stores that in `_func._name`.
    ///
    /// PRE-EXISTING-ADAPTATION: upstream `:194-:` walks every non-NULL
    /// pointer through `getcontainernode()` to register the underlying
    /// `_obj` as a containerlist root + bump the `containerstats[kind]`
    /// counter. The local port returns the name string but skips the
    /// node-factory side effect — `entrypoint.pf` is therefore NOT a
    /// DB root, so [`Self::complete`] currently has nothing to walk
    /// from. Convergence path = port the
    /// `database.py:34-71 getcontainernode` factory + the per-lltype
    /// node renderers (`node.py:Function/StructNode/ArrayNode`); only
    /// after that lands does the `containerlist`/`containerstats`
    /// invariant hold.
    pub fn get(&self, obj: Rc<dyn Any>) -> String {
        if let Some(ptr) = obj.as_ref().downcast_ref::<_ptr>() {
            ptr_name(ptr).unwrap_or_else(|| format!("ptr_{}", ptr._hashable_identity()))
        } else {
            format!("obj_{:p}", Rc::as_ptr(&obj))
        }
    }

    /// Port of `LowLevelDatabase.getcontainernode(obj)`'s observable
    /// registration surface at `database.py:157-179`, without node
    /// factory rendering.
    pub fn getcontainernode(&self, obj: Rc<dyn Any>) -> Rc<dyn Any> {
        let key = Rc::as_ptr(&obj) as *const () as usize;
        if let Some(existing) = self.containernodes.borrow().get(&key) {
            return existing.clone();
        }
        self.containernodes.borrow_mut().insert(key, obj.clone());
        self.containerlist.borrow_mut().push(obj.clone());
        obj
    }

    /// Port of `LowLevelDatabase.complete()` at `database.py:complete`.
    ///
    /// Upstream walks `pendingsetupnodes`, runs `late_initializations`,
    /// expands `containerstats` dependencies and populates every
    /// container node's `_implementationtypename`. None of those
    /// sub-passes are ported locally, so reporting completion would
    /// claim a structurally incomplete DB is ready. Surface a
    /// `TaskError` citing the upstream method until the missing pieces
    /// land. Callers that only need to flip the observable bit during
    /// unit tests can use [`Self::mark_complete_for_tests`].
    pub fn complete(&self) -> Result<(), TaskError> {
        Err(TaskError {
            message: "database.py LowLevelDatabase.complete — pendingsetupnodes / late_initializations / containerstats expansion not yet ported".to_string(),
        })
    }

    /// Test-only sibling of [`Self::complete`] that flips the
    /// `completed` flag without claiming the upstream completion
    /// pipeline ran. Intended for shell tests that exercise the
    /// observable flag in isolation.
    #[cfg(test)]
    pub fn mark_complete_for_tests(&self) {
        self.completed.set(true);
    }

    pub fn globalcontainers(&self) -> Vec<Rc<dyn Any>> {
        self.containerlist.borrow().clone()
    }

    pub fn getstructdeflist(&self) -> Vec<Rc<dyn Any>> {
        self.structdefnodes.borrow().values().cloned().collect()
    }
}

fn ptr_name(ptr: &_ptr) -> Option<String> {
    match &ptr._obj0 {
        Ok(Some(_ptr_obj::Func(func))) => Some(func._name.clone()),
        _ => None,
    }
}

impl std::fmt::Debug for LowLevelDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LowLevelDatabase")
            .field("standalone", &self.standalone)
            .field("sandbox", &self.sandbox)
            .field("split_gc_address_space", &self.split_gc_address_space)
            .field("reverse_debugger", &self.reverse_debugger)
            .field("gcpolicy", &self.gcpolicy)
            .field("completed", &self.completed.get())
            .finish_non_exhaustive()
    }
}
