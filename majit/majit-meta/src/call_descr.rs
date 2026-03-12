use std::sync::Arc;

use majit_ir::{CallDescr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, Type};

/// Generic CallDescr for function call operations.
///
/// Replaces per-interpreter boilerplate (IoCallDescr, etc.)
/// with a single reusable implementation.
#[derive(Debug)]
struct MetaCallDescr {
    arg_types: Vec<Type>,
    result_type: Type,
}

impl majit_ir::Descr for MetaCallDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for MetaCallDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        0
    }
    fn effect_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo {
            extra_effect: ExtraEffect::CannotRaise,
            oopspec_index: OopSpecIndex::None,
        };
        &INFO
    }
}

/// Create a CallDescr with the given argument types and result type.
pub fn make_call_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    Arc::new(MetaCallDescr {
        arg_types: arg_types.to_vec(),
        result_type,
    })
}
