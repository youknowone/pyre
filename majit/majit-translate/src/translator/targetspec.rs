//! Typed carrier for `rpython/translator/driver.py:573-602
//! TranslationDriver.from_targetspec`.
//!
//! Upstream receives a Python dict whose `"target"` value is callable as
//! `target(driver, args)` and returns either `(entry, inputtypes, policy)`,
//! `(entry, inputtypes)`, or just `entry`. Rust has no Python callable
//! object here, so this module keeps the same dict-shaped `extra` payload
//! while making the callable and return variants explicit.

use std::any::Any;
use std::collections::HashMap;
use std::rc::Rc;

use crate::annotator::policy::AnnotatorPolicy;
use crate::annotator::signature::AnnotationSpec;
use crate::flowspace::model::HostObject;
use crate::translator::driver::TranslationDriver;
use crate::translator::tool::taskengine::TaskError;

/// Result variants accepted by upstream `driver.py:587-595`.
#[derive(Clone, Debug)]
pub enum TargetSpecResult {
    /// Non-tuple return: `entry_point = spec; inputtypes = policy = None`.
    Entry(HostObject),
    /// Two-tuple return: `policy = None; entry_point, inputtypes = spec`.
    EntryInputTypes(HostObject, Vec<AnnotationSpec>),
    /// Three-tuple return: `entry_point, inputtypes, policy = spec`.
    EntryInputTypesPolicy(HostObject, Vec<AnnotationSpec>, AnnotatorPolicy),
}

impl TargetSpecResult {
    /// Convert to the argument shape consumed by
    /// `TranslationDriver::setup`, matching `driver.py:587-595`.
    pub fn into_setup_parts(
        self,
    ) -> (
        HostObject,
        Option<Vec<AnnotationSpec>>,
        Option<AnnotatorPolicy>,
    ) {
        match self {
            TargetSpecResult::Entry(entry) => (entry, None, None),
            TargetSpecResult::EntryInputTypes(entry, inputtypes) => (entry, Some(inputtypes), None),
            TargetSpecResult::EntryInputTypesPolicy(entry, inputtypes, policy) => {
                (entry, Some(inputtypes), Some(policy))
            }
        }
    }
}

/// Typed replacement for upstream's arbitrary Python `"target"` callable.
pub trait TargetSpecCallable {
    /// Upstream call site: `spec = target(driver, args)` at
    /// `rpython/translator/driver.py:584`.
    fn call(
        &self,
        driver: &Rc<TranslationDriver>,
        args: &[String],
    ) -> Result<TargetSpecResult, TaskError>;
}

impl<F> TargetSpecCallable for F
where
    F: Fn(&Rc<TranslationDriver>, &[String]) -> Result<TargetSpecResult, TaskError> + 'static,
{
    fn call(
        &self,
        driver: &Rc<TranslationDriver>,
        args: &[String],
    ) -> Result<TargetSpecResult, TaskError> {
        self(driver, args)
    }
}

/// Opaque `"target"` value stored in `driver.extra`.
///
/// Upstream passes the whole `targetspec_dic` to `setup(..., extra=...)`.
/// `driver.extra["target"]` is therefore still present after setup, but
/// Rust stores it inside this wrapper because `Rc<dyn TargetSpecCallable>`
/// cannot itself be downcast out of `Rc<dyn Any>`.
pub struct TargetSpecCallableSlot {
    pub target: Rc<dyn TargetSpecCallable>,
}

/// Dict-shaped target spec with a typed `"target"` key.
pub struct TargetSpecDict {
    target: Rc<dyn TargetSpecCallable>,
    extra: HashMap<String, Rc<dyn Any>>,
}

impl TargetSpecDict {
    /// Build a targetspec dict containing the required upstream
    /// `"target"` key.
    pub fn new(target: Rc<dyn TargetSpecCallable>) -> Self {
        let mut extra: HashMap<String, Rc<dyn Any>> = HashMap::new();
        extra.insert(
            "target".to_string(),
            Rc::new(TargetSpecCallableSlot {
                target: Rc::clone(&target),
            }) as Rc<dyn Any>,
        );
        Self { target, extra }
    }

    /// Add an arbitrary extra key, matching upstream's open dict.
    ///
    /// Upstream `driver.py:582 target = targetspec_dic['target']` reads the
    /// callable from the same dict it later passes through to
    /// `setup(extra=targetspec_dic)`, so the dict slot and the cached
    /// callable must stay in lockstep. Reject `"target"` here — overriding
    /// the callable is done by constructing a new [`TargetSpecDict`] with
    /// [`Self::new`].
    pub fn with_extra(mut self, key: impl Into<String>, value: Rc<dyn Any>) -> Self {
        let key = key.into();
        assert_ne!(
            key, "target",
            "TargetSpecDict::with_extra(\"target\", _) would desync the typed \
             callable slot from extra[\"target\"]; use TargetSpecDict::new \
             to swap the callable",
        );
        self.extra.insert(key, value);
        self
    }

    /// Read the typed `"target"` callable for `driver.py:582-584`.
    pub fn target(&self) -> Rc<dyn TargetSpecCallable> {
        Rc::clone(&self.target)
    }

    /// Consume into `TranslationDriver::setup(..., extra=targetspec_dic)`.
    pub fn into_extra(self) -> HashMap<String, Rc<dyn Any>> {
        self.extra
    }
}
