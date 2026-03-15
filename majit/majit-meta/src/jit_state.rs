use majit_ir::{GcRef, OpRef, Type, Value};

use crate::blackhole::ExceptionState;
use crate::resume::{
    MaterializedValue, MaterializedVirtual, ReconstructedState, ResolvedPendingFieldWrite,
    ResumeFrameLayoutSummary,
};
use crate::trace_ctx::JitDriverDescriptor;
use crate::virtualizable::{
    clear_vable_token, read_all_virtualizable_boxes, write_all_virtualizable_boxes,
    VirtualizableInfo,
};

/// Layout description for replaying a pending field or array write during deopt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingFieldWriteLayout {
    Field {
        offset: usize,
        value_type: Type,
    },
    ArrayItem {
        base_offset: usize,
        item_size: usize,
        item_type: Type,
    },
}

/// Interpreter-specific JIT state contract.
pub trait JitState: Sized {
    type Meta: Clone;
    type Sym;
    type Env: ?Sized;

    fn can_trace(&self) -> bool {
        true
    }

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta;

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64>;

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        self.extract_live(meta)
            .into_iter()
            .map(Value::Int)
            .collect()
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        self.extract_live_values(meta)
            .iter()
            .map(Value::get_type)
            .collect()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym;

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverDescriptor> {
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool;

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]);

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let ints: Vec<i64> = values
            .iter()
            .map(|value| match value {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect();
        self.restore(meta, &ints);
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &ExceptionState,
    ) -> bool {
        self.restore_values(meta, values);
        true
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        let Some(obj_ptr) = self.virtualizable_heap_ptr(meta, virtualizable, info) else {
            return true;
        };
        let Some(lengths) = self.virtualizable_array_lengths(meta, virtualizable, info) else {
            return true;
        };
        let (static_boxes, array_boxes) =
            unsafe { read_all_virtualizable_boxes(info, obj_ptr.cast_const(), &lengths) };
        self.import_virtualizable_boxes(meta, virtualizable, info, &static_boxes, &array_boxes)
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(obj_ptr) = self.virtualizable_heap_ptr(meta, virtualizable, info) else {
            return;
        };
        let Some((static_boxes, array_boxes)) =
            self.export_virtualizable_boxes(meta, virtualizable, info)
        else {
            return;
        };
        unsafe {
            write_all_virtualizable_boxes(info, obj_ptr, &static_boxes, &array_boxes);
            clear_vable_token(info, obj_ptr);
        }
    }

    fn sync_named_virtualizable_before_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) -> bool {
        match info {
            Some(info) => self.sync_virtualizable_before_jit(meta, virtualizable, info),
            None => true,
        }
    }

    fn sync_named_virtualizable_after_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) {
        if let Some(info) = info {
            self.sync_virtualizable_after_jit(meta, virtualizable, info);
        }
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        None
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        None
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        _static_boxes: &[i64],
        _array_boxes: &[Vec<i64>],
    ) -> bool {
        true
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        None
    }

    /// Called before a residual call (CALL_MAY_FORCE) during tracing.
    ///
    /// Writes virtualizable state to heap via SETFIELD_GC ops so the callee
    /// can observe up-to-date virtualizable fields. Interpreters that use
    /// virtualizable objects should override this to emit the necessary
    /// SETFIELD_GC operations for each tracked field.
    ///
    /// Default: no-op (interpreters without virtualizable fields).
    fn sync_virtualizable_before_residual_call(&self, _ctx: &mut crate::trace_ctx::TraceCtx) {}

    /// Called after a residual call (CALL_MAY_FORCE) during tracing.
    ///
    /// Re-reads virtualizable state from heap via GETFIELD_GC ops because
    /// the callee may have modified virtualizable fields. Returns a list
    /// of (field_index, new_opref) pairs so the caller can track the
    /// updated symbolic values.
    ///
    /// Default: no-op (interpreters without virtualizable fields).
    fn sync_virtualizable_after_residual_call(
        &self,
        _ctx: &mut crate::trace_ctx::TraceCtx,
    ) -> Vec<(u32, OpRef)> {
        Vec::new()
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef>;

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::collect_jump_args(sym)
            .into_iter()
            .map(|opref| (opref, Type::Int))
            .collect()
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool;

    fn validate_close_with_jump_args(
        sym: &Self::Sym,
        meta: &Self::Meta,
        _jump_args: &[OpRef],
    ) -> bool {
        Self::validate_close(sym, meta)
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        _materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        None
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        _materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_virtual_ref(meta, virtual_index, materialized)
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        _descr_index: u32,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        None
    }

    fn restore_reconstructed_frames(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        materialized_virtuals: &[MaterializedVirtual],
        exception: &ExceptionState,
    ) -> bool {
        let cache = self.materialize_virtual_refs(meta, materialized_virtuals);
        self.try_restore_reconstructed_frames_with_cache(
            meta,
            reconstructed_state,
            exception,
            &cache,
        )
    }

    fn restore_reconstructed_frames_with_resume_layout(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        exception: &ExceptionState,
    ) -> bool {
        let total_frames = reconstructed_state.frames.len();
        let can_use_generic_layout = total_frames == 1
            || reconstructed_state
                .frames
                .iter()
                .enumerate()
                .any(|(frame_index, frame)| {
                    frame.slot_types.as_ref().is_some()
                        || self
                            .reconstructed_frame_value_types_with_metadata(
                                meta,
                                frame_index,
                                total_frames,
                                frame,
                            )
                            .is_some()
                        || frame_layouts
                            .and_then(|layouts| layouts.get(frame_index))
                            .filter(|layout| {
                                layout.pc == frame.pc
                                    && layout.slot_layouts.len() == frame.values.len()
                            })
                            .and_then(|layout| layout.slot_types.as_ref())
                            .is_some()
                });

        if can_use_generic_layout {
            let cache = self.materialize_virtual_refs(meta, materialized_virtuals);
            if self.try_restore_reconstructed_frames_with_cache_and_layout(
                meta,
                reconstructed_state,
                exception,
                &cache,
                frame_layouts,
            ) {
                return true;
            }
        }

        self.restore_reconstructed_frames(
            meta,
            reconstructed_state,
            materialized_virtuals,
            exception,
        )
    }

    fn reconstructed_frame_value_types(
        &self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        None
    }

    fn reconstructed_frame_value_types_with_metadata(
        &self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
    ) -> Option<Vec<Type>> {
        self.reconstructed_frame_value_types(meta, frame_index, total_frames, frame.pc)
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        _values: &[Value],
        _exception: &ExceptionState,
    ) -> bool {
        false
    }

    fn restore_reconstructed_frame_values_with_metadata(
        &mut self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
        values: &[Value],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_reconstructed_frame_values(
            meta,
            frame_index,
            total_frames,
            frame.pc,
            values,
            exception,
        )
    }

    fn try_restore_reconstructed_frames_with_cache(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
    ) -> bool {
        self.try_restore_reconstructed_frames_with_cache_and_layout(
            meta,
            reconstructed_state,
            exception,
            materialized_refs,
            None,
        )
    }

    fn try_restore_reconstructed_frames_with_cache_and_layout(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
    ) -> bool {
        if reconstructed_state.frames.is_empty() {
            return false;
        }

        let total_frames = reconstructed_state.frames.len();
        let mut used_generic = false;
        for (frame_index, frame) in reconstructed_state.frames.iter().enumerate() {
            let layout_types = frame.slot_types.clone().or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.get(frame_index))
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            });
            if let Some(types) = self
                .reconstructed_frame_value_types_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                )
                .or(layout_types)
            {
                let Some(values) =
                    frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
                else {
                    return false;
                };
                if self.restore_reconstructed_frame_values_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                    &values,
                    exception,
                ) {
                    used_generic = true;
                    continue;
                }
                if total_frames == 1 && frame_index + 1 == total_frames {
                    self.restore_values(meta, &values);
                    return true;
                }
                return false;
            }
        }

        if used_generic {
            return true;
        }

        let frame = reconstructed_state.frames.last().expect("non-empty");
        let types = frame
            .slot_types
            .clone()
            .or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.last())
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            })
            .unwrap_or_else(|| self.live_value_types(meta));
        if types.len() != frame.values.len() {
            return false;
        }
        let Some(values) =
            frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
        else {
            return false;
        };
        self.restore_values(meta, &values);
        true
    }

    fn restore_guard_failure(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_guard_failure_with_resume_layout(
            meta,
            fail_values,
            reconstructed_state,
            None,
            materialized_virtuals,
            pending_field_writes,
            exception,
        )
    }

    fn restore_guard_failure_with_resume_layout(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
    ) -> bool {
        if let Some(reconstructed_state) = reconstructed_state {
            let materialized_refs = self.materialize_virtual_refs(meta, materialized_virtuals);
            let total_frames = reconstructed_state.frames.len();
            let can_use_generic_cache = total_frames == 1
                || reconstructed_state
                    .frames
                    .iter()
                    .enumerate()
                    .any(|(frame_index, frame)| {
                        frame.slot_types.as_ref().is_some()
                            || self
                                .reconstructed_frame_value_types_with_metadata(
                                    meta,
                                    frame_index,
                                    total_frames,
                                    frame,
                                )
                                .is_some()
                            || frame_layouts
                                .and_then(|layouts| layouts.get(frame_index))
                                .filter(|layout| {
                                    layout.pc == frame.pc
                                        && layout.slot_layouts.len() == frame.values.len()
                                })
                                .and_then(|layout| layout.slot_types.as_ref())
                                .is_some()
                    });

            let restored = if can_use_generic_cache {
                let restored = self.try_restore_reconstructed_frames_with_cache_and_layout(
                    meta,
                    reconstructed_state,
                    exception,
                    &materialized_refs,
                    frame_layouts,
                );
                if restored {
                    true
                } else {
                    self.restore_reconstructed_frames_with_resume_layout(
                        meta,
                        reconstructed_state,
                        frame_layouts,
                        materialized_virtuals,
                        exception,
                    )
                }
            } else {
                self.restore_reconstructed_frames_with_resume_layout(
                    meta,
                    reconstructed_state,
                    frame_layouts,
                    materialized_virtuals,
                    exception,
                )
            };

            if !restored {
                return false;
            }
            self.replay_pending_field_writes(meta, pending_field_writes, &materialized_refs);
            return true;
        }

        self.restore(meta, fail_values);
        true
    }

    fn materialize_virtual_refs(
        &mut self,
        meta: &Self::Meta,
        materialized_virtuals: &[MaterializedVirtual],
    ) -> Vec<Option<GcRef>> {
        let mut refs = vec![None; materialized_virtuals.len()];
        let mut progress = true;
        while progress {
            progress = false;
            for (virtual_index, virtual_value) in materialized_virtuals.iter().enumerate() {
                if refs[virtual_index].is_some() {
                    continue;
                }
                if let Some(materialized) = self.materialize_virtual_ref_with_refs(
                    meta,
                    virtual_index,
                    virtual_value,
                    &refs,
                ) {
                    refs[virtual_index] = Some(materialized);
                    progress = true;
                }
            }
        }
        refs
    }

    fn replay_pending_field_writes(
        &mut self,
        meta: &Self::Meta,
        pending_field_writes: &[ResolvedPendingFieldWrite],
        materialized_refs: &[Option<GcRef>],
    ) {
        for pending in pending_field_writes {
            let Some(layout) = self.pending_field_write_layout(
                meta,
                pending.descr_index,
                pending.item_index.is_some(),
            ) else {
                continue;
            };
            let Some(target) = materialized_value_to_i64(&pending.target, materialized_refs) else {
                continue;
            };
            let Some(value) = materialized_value_to_i64(&pending.value, materialized_refs) else {
                continue;
            };
            unsafe {
                apply_pending_field_write(layout, target as usize, pending.item_index, value)
            };
        }
    }
}

fn frame_values_from_reconstructed(
    frame_values: &[crate::resume::ReconstructedValue],
    types: &[Type],
    materialized_refs: &[Option<GcRef>],
) -> Option<Vec<Value>> {
    if frame_values.len() != types.len() {
        return None;
    }
    let mut result = Vec::with_capacity(frame_values.len());
    for (value, expected) in frame_values.iter().zip(types.iter().copied()) {
        let decoded = match (value, expected) {
            (crate::resume::ReconstructedValue::Value(raw), Type::Int) => Value::Int(*raw),
            (crate::resume::ReconstructedValue::Value(raw), Type::Ref) => {
                Value::Ref(GcRef(*raw as usize))
            }
            (crate::resume::ReconstructedValue::Value(raw), Type::Float) => {
                Value::Float(f64::from_bits(*raw as u64))
            }
            (crate::resume::ReconstructedValue::Value(_), Type::Void) => Value::Void,
            (crate::resume::ReconstructedValue::Virtual(index), Type::Ref) => {
                Value::Ref(materialized_refs.get(*index).copied().flatten()?)
            }
            (crate::resume::ReconstructedValue::Uninitialized, _) => Value::Void,
            (crate::resume::ReconstructedValue::Unavailable, _) => Value::Void,
            _ => return None,
        };
        result.push(decoded);
    }
    Some(result)
}

fn materialized_value_to_i64(
    value: &MaterializedValue,
    materialized_refs: &[Option<GcRef>],
) -> Option<i64> {
    match value {
        MaterializedValue::Value(value) => Some(*value),
        MaterializedValue::VirtualRef(index) => materialized_refs
            .get(*index)
            .copied()
            .flatten()
            .map(|gc_ref| gc_ref.as_usize() as i64),
    }
}

unsafe fn apply_pending_field_write(
    layout: PendingFieldWriteLayout,
    target: usize,
    item_index: Option<usize>,
    value: i64,
) {
    match layout {
        PendingFieldWriteLayout::Field { offset, value_type } => {
            let ptr = (target as *mut u8).add(offset);
            write_typed_value(ptr, value, value_type);
        }
        PendingFieldWriteLayout::ArrayItem {
            base_offset,
            item_size,
            item_type,
        } => {
            let index = item_index.unwrap_or(0);
            let ptr = (target as *mut u8).add(base_offset + index * item_size);
            write_typed_value(ptr, value, item_type);
        }
    }
}

unsafe fn write_typed_value(ptr: *mut u8, value: i64, value_type: Type) {
    match value_type {
        Type::Int => (ptr as *mut i64).write(value),
        Type::Ref => (ptr as *mut usize).write(value as usize),
        Type::Float => (ptr as *mut f64).write(f64::from_bits(value as u64)),
        Type::Void => {}
    }
}
