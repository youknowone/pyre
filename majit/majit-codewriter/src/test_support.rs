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
    }
}

pub(crate) fn pyre_analyze_config() -> crate::AnalyzeConfig {
    crate::AnalyzeConfig {
        pipeline: pyre_pipeline_config(),
    }
}
