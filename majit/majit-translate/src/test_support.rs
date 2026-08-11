#[path = "../../../pyre/pyre-jit-trace/src/call_spec.rs"]
mod call_spec;
#[path = "../../../pyre/pyre-jit-trace/src/virtualizable_spec.rs"]
mod virtualizable_spec;

/// Carry one declared `EF_*` from the pyre-owned spec to the translator's
/// own enum. Mirrors `pyre-jit-trace/build.rs`, which does the same walk for
/// the production pipeline; both enumerate the arms so a new member is a
/// compile error rather than a silent re-mapping.
fn declared_extra_effect(extra: call_spec::DeclaredExtraEffect) -> crate::DeclaredExtraEffect {
    use crate::DeclaredExtraEffect as Translate;
    use call_spec::DeclaredExtraEffect as Spec;
    match extra {
        Spec::ElidableCannotRaise => Translate::ElidableCannotRaise,
        Spec::LoopInvariant => Translate::LoopInvariant,
        Spec::CannotRaise => Translate::CannotRaise,
        Spec::ElidableOrMemoryError => Translate::ElidableOrMemoryError,
        Spec::ElidableCanRaise => Translate::ElidableCanRaise,
        Spec::CanRaise => Translate::CanRaise,
        Spec::ForcesVirtualOrVirtualizable => Translate::ForcesVirtualOrVirtualizable,
    }
}

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
                        call_spec::CallEffectKind::Declared(declared) => {
                            crate::CallEffectKind::Declared(crate::DeclaredCallEffects {
                                extra: declared_extra_effect(declared.extra),
                                can_collect: declared.can_collect,
                                can_invalidate: declared.can_invalidate,
                            })
                        }
                    };
                    crate::CallEffectOverride::new(target, effect)
                })
                .collect(),
            ..Default::default()
        },
        jit_drivers: vec![crate::JitDriverSpec {
            portal: crate::CallPath::from_segments(["eval", "eval_loop_jit"]),
            greens: vec![
                "next_instr".to_string(),
                "is_being_profiled".to_string(),
                "pycode".to_string(),
            ],
            reds: vec!["frame".to_string(), "ec".to_string()],
            autoreds: false,
            virtualizables: vec!["frame".to_string()],
            red_types: vec!["PyFrame".to_string(), "ExecutionContext".to_string()],
        }],
        register_trait_families: Vec::new(),
    }
}
