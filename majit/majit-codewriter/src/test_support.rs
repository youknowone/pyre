#[path = "../../../pyre/pyre-jit-trace/src/call_spec.rs"]
mod call_spec;
#[path = "../../../pyre/pyre-jit-trace/src/virtualizable_spec.rs"]
mod virtualizable_spec;

pub(crate) fn pyre_pipeline_config() -> crate::PipelineConfig {
    crate::PipelineConfig {
        transform: crate::GraphTransformConfig {
            vable_fields: virtualizable_spec::PYFRAME_VABLE_FIELDS
                .iter()
                .map(|(name, idx)| {
                    crate::VirtualizableFieldDescriptor::new(
                        *name,
                        Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                        *idx,
                    )
                })
                .collect(),
            vable_arrays: virtualizable_spec::PYFRAME_VABLE_ARRAYS
                .iter()
                .map(|(name, idx)| {
                    crate::VirtualizableFieldDescriptor::new(
                        *name,
                        Some(virtualizable_spec::PYFRAME_VABLE_OWNER_ROOT.to_string()),
                        *idx,
                    )
                })
                .collect(),
            call_effects: call_spec::PYFRAME_CALL_EFFECTS
                .iter()
                .map(|spec| {
                    let target = match spec.target {
                        call_spec::CallTargetSpec::Method {
                            name,
                            receiver_root,
                        } => crate::CallTarget::method(name, Some(receiver_root.to_string())),
                        call_spec::CallTargetSpec::FunctionPath(segments) => {
                            crate::CallTarget::function_path(segments.iter().copied())
                        }
                    };
                    let effect = match spec.effect {
                        call_spec::CallEffectKind::Elidable => crate::CallEffectKind::Elidable,
                        call_spec::CallEffectKind::Residual => crate::CallEffectKind::Residual,
                    };
                    crate::CallEffectOverride::new(target, effect)
                })
                .collect(),
            ..Default::default()
        },
        classify: crate::ClassificationConfig {
            field_roles: virtualizable_spec::PYFRAME_FIELD_ROLES
                .iter()
                .map(|spec| {
                    let role = match spec.role {
                        virtualizable_spec::FieldPatternRole::LocalArray => {
                            crate::FieldPatternRole::LocalArray
                        }
                        virtualizable_spec::FieldPatternRole::InstructionPosition => {
                            crate::FieldPatternRole::InstructionPosition
                        }
                        virtualizable_spec::FieldPatternRole::ConstantPool => {
                            crate::FieldPatternRole::ConstantPool
                        }
                    };
                    crate::FieldRoleDescriptor::new(
                        spec.name,
                        Some(spec.owner_root.to_string()),
                        role,
                    )
                })
                .collect(),
            call_roles: call_spec::PYFRAME_CALL_EFFECTS
                .iter()
                .filter_map(|spec| {
                    let role = spec.role?;
                    let target = match spec.target {
                        call_spec::CallTargetSpec::Method {
                            name,
                            receiver_root,
                        } => crate::CallTarget::method(name, Some(receiver_root.to_string())),
                        call_spec::CallTargetSpec::FunctionPath(segments) => {
                            crate::CallTarget::function_path(segments.iter().copied())
                        }
                    };
                    let role = match role {
                        call_spec::CallPatternRole::IntArithmetic => {
                            crate::CallPatternRole::IntArithmetic
                        }
                        call_spec::CallPatternRole::FloatArithmetic => {
                            crate::CallPatternRole::FloatArithmetic
                        }
                        call_spec::CallPatternRole::LocalRead => crate::CallPatternRole::LocalRead,
                        call_spec::CallPatternRole::LocalWrite => {
                            crate::CallPatternRole::LocalWrite
                        }
                        call_spec::CallPatternRole::FunctionCall => {
                            crate::CallPatternRole::FunctionCall
                        }
                        call_spec::CallPatternRole::TruthCheck => {
                            crate::CallPatternRole::TruthCheck
                        }
                        call_spec::CallPatternRole::StackManip => {
                            crate::CallPatternRole::StackManip
                        }
                        call_spec::CallPatternRole::ConstLoad => crate::CallPatternRole::ConstLoad,
                        call_spec::CallPatternRole::NamespaceLoadLocal => {
                            crate::CallPatternRole::NamespaceLoadLocal
                        }
                        call_spec::CallPatternRole::NamespaceLoadGlobal => {
                            crate::CallPatternRole::NamespaceLoadGlobal
                        }
                        call_spec::CallPatternRole::NamespaceStoreLocal => {
                            crate::CallPatternRole::NamespaceStoreLocal
                        }
                        call_spec::CallPatternRole::NamespaceStoreGlobal => {
                            crate::CallPatternRole::NamespaceStoreGlobal
                        }
                        call_spec::CallPatternRole::RangeIterNext => {
                            crate::CallPatternRole::RangeIterNext
                        }
                        call_spec::CallPatternRole::IterCleanup => {
                            crate::CallPatternRole::IterCleanup
                        }
                        call_spec::CallPatternRole::Return => crate::CallPatternRole::Return,
                        call_spec::CallPatternRole::BuildList => crate::CallPatternRole::BuildList,
                        call_spec::CallPatternRole::BuildTuple => {
                            crate::CallPatternRole::BuildTuple
                        }
                        call_spec::CallPatternRole::UnpackSequence => {
                            crate::CallPatternRole::UnpackSequence
                        }
                        call_spec::CallPatternRole::SequenceSetitem => {
                            crate::CallPatternRole::SequenceSetitem
                        }
                        call_spec::CallPatternRole::CollectionAppend => {
                            crate::CallPatternRole::CollectionAppend
                        }
                    };
                    Some(crate::CallRoleDescriptor::new(target, role))
                })
                .collect(),
        },
    }
}

pub(crate) fn pyre_analyze_config() -> crate::AnalyzeConfig {
    crate::AnalyzeConfig {
        pipeline: pyre_pipeline_config(),
    }
}
