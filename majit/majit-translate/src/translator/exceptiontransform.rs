//! RPython `rpython/translator/exceptiontransform.py`.
//!
//! Upstream rewrites graphs so calls that can raise update global exception
//! state and return a default error value. Pyre already lowers Rust
//! `Result`/`?` into exceptional graph exits before the C backend path, but
//! `translator.py` and `genc.py` still expect this module and a lazy
//! `ExceptionTransformer(self)` object. This file ports that public surface
//! and deterministic helpers first; graph-mutating methods return explicit
//! errors until the full block/link rewrite is ported.

#![allow(non_snake_case)]

use std::rc::Weak;

use crate::flowspace::model::{ConstValue, Constant, GraphRef, Hlvalue};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// RPython `PrimitiveErrorValue`.
pub fn PrimitiveErrorValue(TYPE: &LowLevelType) -> Option<ConstValue> {
    match TYPE {
        LowLevelType::Signed | LowLevelType::SignedLongLong | LowLevelType::SignedLongLongLong => {
            Some(ConstValue::Int(-1))
        }
        LowLevelType::Unsigned
        | LowLevelType::UnsignedLongLong
        | LowLevelType::UnsignedLongLongLong => Some(ConstValue::Int(-1)),
        LowLevelType::Float | LowLevelType::SingleFloat | LowLevelType::LongFloat => {
            Some(ConstValue::float(-1.0))
        }
        LowLevelType::Char => Some(ConstValue::ByteStr(vec![255])),
        LowLevelType::UniChar => Some(ConstValue::UniStr("\u{ffff}".to_string())),
        LowLevelType::Bool => Some(ConstValue::Bool(true)),
        LowLevelType::Address | LowLevelType::Void => Some(ConstValue::None),
        _ => None,
    }
}

/// RPython `default_error_value(T)`.
pub fn default_error_value(T: &LowLevelType) -> Result<ConstValue, TaskError> {
    if let Some(value) = PrimitiveErrorValue(T) {
        return Ok(value);
    }
    if matches!(T, LowLevelType::Ptr(_)) {
        return Ok(ConstValue::None);
    }
    Err(TaskError {
        message: format!("exceptiontransform.py: default_error_value({T:?}) not implemented yet"),
    })
}

/// RPython `has_llhelper_error_value(graph)`.
///
/// The Rust graph carrier does not yet expose function-object ad-hoc
/// attributes such as `_llhelper_error_value_`, so this predicate remains
/// false until that slot lands.
pub fn has_llhelper_error_value(_graph: &GraphRef) -> bool {
    false
}

/// RPython `error_value(graph)`.
pub fn error_value(graph: &GraphRef) -> Result<ConstValue, TaskError> {
    let graph_borrow = graph.borrow();
    let returnblock = graph_borrow.returnblock.borrow();
    let Some(v_return) = returnblock.inputargs.first() else {
        return Err(TaskError {
            message: "exceptiontransform.py: returnblock has no return value".to_string(),
        });
    };
    let Some(T) = hlvalue_concretetype(v_return) else {
        return Err(TaskError {
            message: "exceptiontransform.py: return value has no concretetype".to_string(),
        });
    };
    default_error_value(&T)
}

/// RPython `error_constant(graph)`.
pub fn error_constant(graph: &GraphRef) -> Result<Constant, TaskError> {
    let value = error_value(graph)?;
    let graph_borrow = graph.borrow();
    let returnblock = graph_borrow.returnblock.borrow();
    let T = returnblock
        .inputargs
        .first()
        .and_then(hlvalue_concretetype)
        .ok_or_else(|| TaskError {
            message: "exceptiontransform.py: return value has no concretetype".to_string(),
        })?;
    Ok(Constant::with_concretetype(value, T))
}

/// RPython `constant_value(llvalue)`.
pub fn constant_value(llvalue: ConstValue, concretetype: LowLevelType) -> Constant {
    Constant::with_concretetype(llvalue, concretetype)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TransformStats {
    pub n_need_exc_matching_blocks: usize,
    pub n_gen_exc_checks: usize,
}

/// RPython `class ExceptionTransformer`.
#[derive(Debug)]
pub struct ExceptionTransformer {
    pub translator: Weak<TranslationContext>,
    pub lltype_of_exception_value: Option<LowLevelType>,
    pub lltype_of_exception_type: Option<LowLevelType>,
}

impl ExceptionTransformer {
    /// RPython `ExceptionTransformer.__init__(translator)`.
    pub fn new(translator: Weak<TranslationContext>) -> Result<Self, TaskError> {
        let strong = translator.upgrade().ok_or_else(|| TaskError {
            message: "exceptiontransform.py: translator already dropped".to_string(),
        })?;
        let rtyper = strong.rtyper.borrow();
        let rtyper = rtyper.as_ref().ok_or_else(|| TaskError {
            message: "exceptiontransform.py: ExceptionTransformer requires translator.rtyper"
                .to_string(),
        })?;
        let exceptiondata = rtyper.exceptiondata.borrow();
        let lltype_of_exception_value = exceptiondata
            .as_ref()
            .map(|edata| edata.lltype_of_exception_value.clone());
        let lltype_of_exception_type = exceptiondata
            .as_ref()
            .map(|edata| edata.lltype_of_exception_type.clone());
        Ok(Self {
            translator,
            lltype_of_exception_value,
            lltype_of_exception_type,
        })
    }

    pub fn noinline<F>(&self, fn_: F) -> NoInline<F> {
        NoInline { fn_ }
    }

    pub fn transform_completely(&self) -> Result<Vec<TransformStats>, TaskError> {
        let translator = self.translator.upgrade().ok_or_else(|| TaskError {
            message: "exceptiontransform.py: translator already dropped".to_string(),
        })?;
        translator
            .graphs
            .borrow()
            .iter()
            .map(|graph| self.create_exception_handling(graph))
            .collect()
    }

    pub fn create_exception_handling(
        &self,
        _graph: &GraphRef,
    ) -> Result<TransformStats, TaskError> {
        Err(TaskError {
            message:
                "exceptiontransform.py: create_exception_handling graph rewrite is not ported yet"
                    .to_string(),
        })
    }

    pub fn transform_block(&self) -> Result<(), TaskError> {
        Err(TaskError {
            message: "exceptiontransform.py: transform_block is not ported yet".to_string(),
        })
    }

    pub fn build_extra_funcs(&self) -> Result<(), TaskError> {
        Err(TaskError {
            message: "exceptiontransform.py: build_extra_funcs is not ported yet".to_string(),
        })
    }

    pub fn same_obj<T>(&self, ptr1: *const T, ptr2: *const T) -> bool {
        ptr1 == ptr2
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoInline<F> {
    pub fn_: F,
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_error_value_matches_primitive_table() {
        assert_eq!(
            default_error_value(&LowLevelType::Signed).unwrap(),
            ConstValue::Int(-1)
        );
        assert_eq!(
            default_error_value(&LowLevelType::Bool).unwrap(),
            ConstValue::Bool(true)
        );
        assert_eq!(
            default_error_value(&LowLevelType::Void).unwrap(),
            ConstValue::None
        );
    }

    #[test]
    fn default_error_value_rejects_unimplemented_nonprimitive() {
        let err = default_error_value(&LowLevelType::Array(Box::new(
            crate::translator::rtyper::lltypesystem::lltype::ArrayType::new(LowLevelType::Signed),
        )))
        .expect_err("arrays are not default error values");

        assert!(err.message.contains("default_error_value"));
    }

    #[test]
    fn same_obj_uses_pointer_identity() {
        let ctx = std::rc::Rc::new(TranslationContext::new());
        let transformer = ExceptionTransformer {
            translator: std::rc::Rc::downgrade(&ctx),
            lltype_of_exception_value: None,
            lltype_of_exception_type: None,
        };
        let a = 1_i32;
        let b = 1_i32;

        assert!(transformer.same_obj(&a, &a));
        assert!(!transformer.same_obj(&a, &b));
    }
}
