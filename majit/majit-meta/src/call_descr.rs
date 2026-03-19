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

#[derive(Debug)]
struct MetaCallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: u64,
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
        static INFO: EffectInfo =
            EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);
        &INFO
    }
}

impl majit_ir::Descr for MetaCallAssemblerDescr {
    fn index(&self) -> u32 {
        u32::MAX
    }
    fn as_call_descr(&self) -> Option<&dyn CallDescr> {
        Some(self)
    }
}

impl CallDescr for MetaCallAssemblerDescr {
    fn arg_types(&self) -> &[Type] {
        &self.arg_types
    }
    fn result_type(&self) -> Type {
        self.result_type
    }
    fn result_size(&self) -> usize {
        8
    }
    fn call_target_token(&self) -> Option<u64> {
        Some(self.target_token)
    }
    fn effect_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
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

/// Create a CallDescr for `CALL_ASSEMBLER_*` with the given target token.
pub fn make_call_assembler_descr(
    target_token: u64,
    arg_types: &[Type],
    result_type: Type,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
    })
}
