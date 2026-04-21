use std::sync::Arc;

use majit_ir::{CallDescr, DescrRef, EffectInfo, ExtraEffect, OopSpecIndex, Type, VableExpansion};

/// Generic CallDescr for function call operations.
///
/// Stores per-call-site EffectInfo, matching RPython's
/// `effectinfo_from_writeanalyze` (call.py:320).
#[derive(Debug)]
struct MetaCallDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    effect_info: EffectInfo,
}

#[derive(Debug)]
struct MetaCallAssemblerDescr {
    arg_types: Vec<Type>,
    result_type: Type,
    target_token: u64,
    vable_expansion: Option<VableExpansion>,
    /// rewrite.py:684 `jd.index_of_virtualizable`: index of the
    /// virtualizable argument inside the callee's red-arg list.
    virtualizable_arg_index: Option<usize>,
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
    fn get_extra_info(&self) -> &EffectInfo {
        &self.effect_info
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
    fn call_virtualizable_index(&self) -> Option<usize> {
        self.virtualizable_arg_index
    }
    fn get_extra_info(&self) -> &EffectInfo {
        static INFO: EffectInfo = EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
        &INFO
    }
    fn vable_expansion(&self) -> Option<&VableExpansion> {
        self.vable_expansion.as_ref()
    }
}

/// Default EffectInfo for call descriptors that lack per-call-site
/// analysis. CannotRaise until `effectinfo_from_writeanalyze`
/// (call.py:320) is ported — the optimizer relies on this to keep
/// heap caches alive across calls.
const DEFAULT_EFFECT_INFO: EffectInfo =
    EffectInfo::const_new(ExtraEffect::CannotRaise, OopSpecIndex::None);

/// Create a CallDescr with the given argument types and result type.
pub fn make_call_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    make_call_descr_with_effect(arg_types, result_type, DEFAULT_EFFECT_INFO)
}

/// call.py:320 `effectinfo_from_writeanalyze` parity. Create a
/// CallDescr with explicit per-call-site EffectInfo.
pub fn make_call_descr_with_effect(
    arg_types: &[Type],
    result_type: Type,
    effect_info: EffectInfo,
) -> DescrRef {
    Arc::new(MetaCallDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        effect_info,
    })
}

/// Create a CallDescr for CALL_MAY_FORCE_* operations.
///
/// RPython treats these as may-raise calls guarded by GUARD_NOT_FORCED, not as
/// generic cannot-raise helpers.
pub fn make_call_may_force_descr(arg_types: &[Type], result_type: Type) -> DescrRef {
    #[derive(Debug)]
    struct MetaCallMayForceDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl majit_ir::Descr for MetaCallMayForceDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for MetaCallMayForceDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }
        fn result_type(&self) -> Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            0
        }
        fn get_extra_info(&self) -> &EffectInfo {
            static INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
            &INFO
        }
    }

    Arc::new(MetaCallMayForceDescr {
        arg_types: arg_types.to_vec(),
        result_type,
    })
}

/// Create a CallDescr for `CALL_ASSEMBLER_*` with the given target token.
pub fn make_call_assembler_descr(
    target_token: u64,
    arg_types: &[Type],
    result_type: Type,
    virtualizable_arg_index: Option<usize>,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: None,
        virtualizable_arg_index,
    })
}

/// rewrite.py:665-695 handle_call_assembler: create a CallDescr that carries
/// virtualizable expansion info. The backend reads fields from the frame
/// reference to populate the callee's full inputarg jitframe layout.
pub fn make_call_assembler_descr_with_vable(
    target_token: u64,
    arg_types: &[Type],
    result_type: Type,
    expansion: VableExpansion,
) -> DescrRef {
    Arc::new(MetaCallAssemblerDescr {
        arg_types: arg_types.to_vec(),
        result_type,
        target_token,
        vable_expansion: Some(expansion),
        virtualizable_arg_index: None,
    })
}
