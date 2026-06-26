//! Port of `rpython/rtyper/extfunc.py`.

use crate::annotator::argument::ArgumentsForTranslation;
use crate::annotator::model::{
    AnnotatorError, KnownType, SomeBool, SomeFloat, SomeInteger, SomeNone, SomeObject,
    SomeObjectTrait, SomeTuple, SomeValue,
};
use crate::annotator::signature::SignatureError;
use crate::flowspace::model::{ConcretetypePlaceholder, ConstValue, Constant, HostObject};

use super::error::TyperError;
use super::lltypesystem::lltype::{self, _ptr, FuncType, LowLevelType};
use super::rmodel::{Repr, ReprState};

/// Send-able counterpart of `annotator.signature.annotation(...)` inputs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ExternalAnnotation {
    Float,
    Int,
    Bool,
    None,
    Tuple(Vec<ExternalAnnotation>),
}

impl ExternalAnnotation {
    pub(crate) fn annotation(&self) -> SomeValue {
        match self {
            ExternalAnnotation::Float => s_float(),
            ExternalAnnotation::Int => s_int(),
            ExternalAnnotation::Bool => s_bool(),
            ExternalAnnotation::None => SomeValue::None_(SomeNone::new()),
            ExternalAnnotation::Tuple(items) => {
                s_tuple(items.iter().map(ExternalAnnotation::annotation).collect())
            }
        }
    }
}

/// RPython `class SomeExternalFunction(SomeObject)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SomeExternalFunction {
    pub base: SomeObject,
    pub name: String,
    pub args_s: Vec<SomeValue>,
    pub s_result: Box<SomeValue>,
    pub needs_sandboxing: bool,
}

impl SomeExternalFunction {
    pub fn new(
        function: HostObject,
        name: impl Into<String>,
        args_s: Vec<SomeValue>,
        s_result: SomeValue,
        needs_sandboxing: bool,
    ) -> Self {
        let mut base = SomeObject::new(KnownType::BuiltinFunctionOrMethod, true);
        base.const_box = Some(Constant::new(ConstValue::HostObject(function)));
        SomeExternalFunction {
            base,
            name: name.into(),
            args_s,
            s_result: Box::new(s_result),
            needs_sandboxing,
        }
    }

    /// RPython `SomeExternalFunction.check_args`.
    pub fn check_args(&self, callspec: &ArgumentsForTranslation) -> Result<(), SignatureError> {
        let (args_s, kwargs) = callspec
            .unpack()
            .map_err(|err| SignatureError::new(err.getmsg()))?;
        if !kwargs.is_empty() {
            return Err(SignatureError::new(
                "External functions cannot be called with keyword arguments",
            ));
        }
        if args_s.len() != self.args_s.len() {
            return Err(SignatureError::new("Argument number mismatch"));
        }
        for (i, (arg, s_param)) in args_s.iter().zip(&self.args_s).enumerate() {
            let Some(arg) = arg else {
                continue;
            };
            let union = crate::annotator::model::union(arg, s_param)
                .map_err(|e| SignatureError::new(e.msg))?;
            if !s_param.contains(&union) {
                return Err(SignatureError::new(format!(
                    "In call to external function {:?}:\narg {} must be {:?},\n          got {:?}",
                    self.name,
                    i + 1,
                    s_param,
                    arg
                )));
            }
        }
        Ok(())
    }

    /// RPython `SomeExternalFunction.call`.
    pub fn call(
        &self,
        callspec: &ArgumentsForTranslation,
    ) -> Result<Option<SomeValue>, AnnotatorError> {
        self.check_args(callspec)?;
        Ok(Some((*self.s_result).clone()))
    }

    /// RPython `SomeExternalFunction.rtyper_makerepr`.
    pub fn rtyper_makerepr(
        &self,
        impl_ptr: Option<_ptr>,
        fakeimpl: Option<HostObject>,
    ) -> Result<ExternalFunctionRepr, TyperError> {
        if !self.is_constant() {
            return Err(TyperError::message("Non-constant external function!"));
        }
        Ok(ExternalFunctionRepr::new(self.clone(), impl_ptr, fakeimpl))
    }

    pub fn rtyper_makekey(&self) -> (&'static str, usize) {
        (
            "SomeExternalFunction",
            self.base
                .const_box
                .as_ref()
                .and_then(|c| match &c.value {
                    ConstValue::HostObject(obj) => Some(obj.identity_id()),
                    _ => None,
                })
                .unwrap_or(0),
        )
    }
}

impl SomeObjectTrait for SomeExternalFunction {
    fn knowntype(&self) -> KnownType {
        self.base.knowntype
    }

    fn immutable(&self) -> bool {
        true
    }

    fn is_constant(&self) -> bool {
        self.base.const_box.is_some()
    }

    fn can_be_none(&self) -> bool {
        false
    }
}

/// RPython `class ExternalFunctionRepr(Repr)`.
#[derive(Debug)]
pub struct ExternalFunctionRepr {
    pub s_func: SomeExternalFunction,
    pub impl_ptr: Option<_ptr>,
    pub fakeimpl: Option<HostObject>,
    lowleveltype: LowLevelType,
    state: ReprState,
}

impl ExternalFunctionRepr {
    pub fn new(
        s_func: SomeExternalFunction,
        impl_ptr: Option<_ptr>,
        fakeimpl: Option<HostObject>,
    ) -> Self {
        ExternalFunctionRepr {
            s_func,
            impl_ptr,
            fakeimpl,
            lowleveltype: LowLevelType::Void,
            state: ReprState::new(),
        }
    }

    /// RPython `ExternalFunctionRepr.get_funcptr`.
    pub fn get_funcptr(
        &self,
        args_ll: Vec<ConcretetypePlaceholder>,
        ll_result: ConcretetypePlaceholder,
    ) -> _ptr {
        if let Some(ptr) = &self.impl_ptr {
            return ptr.clone();
        }
        let callable = self.fakeimpl.as_ref().map(|obj| obj.qualname().to_string());
        lltype::functionptr_with_external_name(
            FuncType {
                args: args_ll,
                result: ll_result,
            },
            &self.s_func.name,
            callable,
        )
    }
}

impl Repr for ExternalFunctionRepr {
    fn lowleveltype(&self) -> &LowLevelType {
        &self.lowleveltype
    }

    fn state(&self) -> &ReprState {
        &self.state
    }

    fn class_name(&self) -> &'static str {
        "ExternalFunctionRepr"
    }
}

/// RPython `class ExtFuncEntry(ExtRegistryEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtFuncEntry {
    pub function: HostObject,
    pub safe_not_sandboxed: bool,
    pub(crate) signature_args: Vec<ExternalAnnotation>,
    pub(crate) signature_result: ExternalAnnotation,
    pub name: String,
    pub lltypeimpl: Option<_ptr>,
    pub lltypefakeimpl: Option<HostObject>,
}

impl ExtFuncEntry {
    pub fn compute_annotation(&self, sandbox: bool) -> SomeExternalFunction {
        SomeExternalFunction::new(
            self.function.clone(),
            self.name.clone(),
            self.signature_args
                .iter()
                .map(ExternalAnnotation::annotation)
                .collect(),
            self.signature_result.annotation(),
            sandbox && !self.safe_not_sandboxed,
        )
    }
}

/// RPython `register_external(function, args, result=None, ...)`.
pub fn register_external(
    function: HostObject,
    args: Vec<ExternalAnnotation>,
    result: Option<ExternalAnnotation>,
    export_name: Option<String>,
    llimpl: Option<_ptr>,
    llfakeimpl: Option<HostObject>,
    sandboxsafe: bool,
) -> Result<ExtFuncEntry, TyperError> {
    let name = export_name.unwrap_or_else(|| function.simple_name().to_string());
    let entry = ExtFuncEntry {
        function: function.clone(),
        safe_not_sandboxed: sandboxsafe,
        signature_args: args,
        signature_result: result.unwrap_or(ExternalAnnotation::None),
        name,
        lltypeimpl: llimpl,
        lltypefakeimpl: llfakeimpl,
    };
    super::extregistry::register_host_value(
        function,
        super::extregistry::ExtRegistryEntry::ExternalFunction(entry.clone()),
    )?;
    Ok(entry)
}

/// RPython `is_external(func)`.
pub fn is_external(func: &ConstValue) -> bool {
    match func {
        ConstValue::LLPtr(ptr) => match ptr._obj0_value() {
            Ok(Some(lltype::_ptr_obj::Func(func))) => func.attrs.contains_key("_external_name"),
            _ => false,
        },
        ConstValue::HostObject(_) => super::extregistry::lookup(func).is_some_and(|entry| {
            matches!(
                entry,
                super::extregistry::ExtRegistryEntry::ExternalFunction(_)
            )
        }),
        _ => false,
    }
}

fn s_float() -> SomeValue {
    SomeValue::Float(SomeFloat::new())
}

fn s_int() -> SomeValue {
    SomeValue::Integer(SomeInteger::new(false, false))
}

fn s_bool() -> SomeValue {
    SomeValue::Bool(SomeBool::new())
}

fn s_tuple(items: Vec<SomeValue>) -> SomeValue {
    SomeValue::Tuple(SomeTuple::new(items))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;

    #[test]
    fn some_external_function_rejects_keywords() {
        let function = HostObject::new_builtin_callable("pkg.extfunc.kw");
        let s = SomeExternalFunction::new(function, "pkg.extfunc.kw", vec![], s_int(), false);
        let mut callspec = ArgumentsForTranslation::new(vec![], None, None);
        callspec.keywords.insert("x".into(), Some(s_int()));
        let err = s.check_args(&callspec).expect_err("keywords are rejected");
        assert!(
            err.0
                .contains("External functions cannot be called with keyword arguments")
        );
    }

    #[test]
    fn register_external_surfaces_extregistry_entry() {
        let function = HostObject::new_builtin_callable("pkg.extfunc.registered");
        let entry = register_external(
            function.clone(),
            vec![ExternalAnnotation::Float],
            Some(ExternalAnnotation::Bool),
            Some("pkg.extfunc.exported".into()),
            None,
            None,
            true,
        )
        .expect("register_external");
        assert_eq!(entry.name, "pkg.extfunc.exported");
        let cv = ConstValue::HostObject(function);
        assert!(super::super::extregistry::is_registered(&cv));
        let s = super::super::extregistry::lookup(&cv)
            .expect("entry")
            .compute_annotation()
            .expect("annotation");
        assert!(matches!(s, SomeValue::ExternalFunction(_)));
    }

    #[test]
    fn external_function_repr_builds_external_name_funcptr() {
        let function = HostObject::new_builtin_callable("pkg.extfunc.ptr");
        let s = SomeExternalFunction::new(
            function,
            "pkg.extfunc.ptr",
            vec![s_float()],
            s_float(),
            false,
        );
        let repr = s.rtyper_makerepr(None, None).expect("makerepr");
        assert!(matches!(repr.lowleveltype(), LowLevelType::Void));
        let ptr = repr.get_funcptr(vec![LowLevelType::Float], LowLevelType::Float);
        assert!(is_external(&ConstValue::LLPtr(Box::new(ptr))));
    }
}
