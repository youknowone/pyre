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

/// Session-scoped materialization cache that persists across guard failure
/// resumption within the same deopt session.
///
/// When GUARD_NOT_FORCED triggers resumption after CALL_MAY_FORCE, the same
/// virtuals may need to be materialized again. This cache ensures each virtual
/// is materialized at most once per deopt session and reused throughout.
#[derive(Debug, Clone, Default)]
pub struct DeoptMaterializationCache {
    /// Maps virtual indices to materialized GcRefs. `None` means not yet
    /// materialized; `Some(gcref)` means already materialized in this session.
    pub refs: Vec<Option<GcRef>>,
}

impl DeoptMaterializationCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self { refs: Vec::new() }
    }

    /// Create a cache pre-sized for the given number of virtuals.
    pub fn with_capacity(num_virtuals: usize) -> Self {
        Self {
            refs: vec![None; num_virtuals],
        }
    }

    /// Return the number of cached entries (including None slots).
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Return true if no slots have been allocated.
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Return the number of successfully materialized entries.
    pub fn materialized_count(&self) -> usize {
        self.refs.iter().filter(|r| r.is_some()).count()
    }

    /// Clear all cached entries, ending the deopt session.
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    /// Look up a cached materialization by virtual index.
    pub fn get(&self, virtual_index: usize) -> Option<GcRef> {
        self.refs.get(virtual_index).copied().flatten()
    }

    /// Ensure the cache has room for at least `count` virtual indices.
    pub fn ensure_capacity(&mut self, count: usize) {
        if self.refs.len() < count {
            self.refs.resize(count, None);
        }
    }

    /// Insert a materialized ref at the given virtual index.
    pub fn insert(&mut self, virtual_index: usize, gc_ref: GcRef) {
        self.ensure_capacity(virtual_index + 1);
        self.refs[virtual_index] = Some(gc_ref);
    }
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
        let raw = self.extract_live(meta);
        let types = self.live_value_types(meta);
        raw.into_iter()
            .zip(types)
            .map(|(v, t)| match t {
                Type::Float => Value::Float(f64::from_bits(v as u64)),
                Type::Ref => Value::Ref(GcRef(v as usize)),
                _ => Value::Int(v),
            })
            .collect()
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        self.extract_live(meta).iter().map(|_| Type::Int).collect()
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
        self.restore_guard_failure_with_session_cache(
            meta,
            fail_values,
            reconstructed_state,
            frame_layouts,
            materialized_virtuals,
            pending_field_writes,
            exception,
            None,
        )
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

    /// Materialize virtual refs into an existing session cache.
    ///
    /// Virtuals that are already present in the cache (from a previous deopt
    /// in the same session) are skipped. Only unmaterialized entries trigger
    /// a call to `materialize_virtual_ref_with_refs`.
    fn materialize_virtual_refs_into_cache(
        &mut self,
        meta: &Self::Meta,
        materialized_virtuals: &[MaterializedVirtual],
        cache: &mut DeoptMaterializationCache,
    ) {
        cache.ensure_capacity(materialized_virtuals.len());
        let mut progress = true;
        while progress {
            progress = false;
            for (idx, virt) in materialized_virtuals.iter().enumerate() {
                if cache.refs[idx].is_some() {
                    continue; // Already materialized (possibly from previous session)
                }
                if let Some(materialized) =
                    self.materialize_virtual_ref_with_refs(meta, idx, virt, &cache.refs)
                {
                    cache.refs[idx] = Some(materialized);
                    progress = true;
                }
            }
        }
    }

    /// Force a single virtual using the session cache. Returns the cached
    /// result if already materialized, otherwise materializes and inserts
    /// into the cache.
    fn force_virtual_with_cache(
        &mut self,
        meta: &Self::Meta,
        cache: &mut DeoptMaterializationCache,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        // Check cache first
        if let Some(cached) = cache.get(virtual_index) {
            return Some(cached);
        }
        // Materialize and cache
        cache.ensure_capacity(virtual_index + 1);
        let result =
            self.materialize_virtual_ref_with_refs(meta, virtual_index, materialized, &cache.refs)?;
        cache.refs[virtual_index] = Some(result);
        Some(result)
    }

    /// Restore guard failure state using a persistent session cache.
    ///
    /// This is the session-aware variant of `restore_guard_failure_with_resume_layout`.
    /// When `session_cache` is provided, materialized virtuals are stored in
    /// it and reused across subsequent deopts within the same session (e.g.,
    /// when GUARD_NOT_FORCED triggers resumption after CALL_MAY_FORCE).
    fn restore_guard_failure_with_session_cache(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
        session_cache: Option<&mut DeoptMaterializationCache>,
    ) -> bool {
        if let Some(reconstructed_state) = reconstructed_state {
            let materialized_refs = if let Some(cache) = session_cache {
                self.materialize_virtual_refs_into_cache(meta, materialized_virtuals, cache);
                cache.refs.clone()
            } else {
                self.materialize_virtual_refs(meta, materialized_virtuals)
            };
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
            (crate::resume::ReconstructedValue::Virtual(index), _) => {
                // Virtual objects always materialize as Ref, regardless of the
                // expected type. The interpreter's restore_values() knows to
                // treat these as object references.
                Value::Ref(materialized_refs.get(*index).copied().flatten()?)
            }
            (crate::resume::ReconstructedValue::Uninitialized, _) => Value::Void,
            (crate::resume::ReconstructedValue::Unavailable, _) => Value::Void,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::{
        MaterializedValue, MaterializedVirtual, ReconstructedFrame, ReconstructedState,
        ReconstructedValue, ResumeFrameLayoutSummary,
    };
    use majit_codegen::ExitFrameLayout;
    use std::cell::Cell;

    /// Test state that tracks how many times `materialize_virtual_ref` was
    /// called, so we can verify the session cache prevents re-materialization.
    #[derive(Default)]
    struct CachingTestState {
        materialize_call_count: Cell<usize>,
        restored_values: Vec<Value>,
        restored_frames: Vec<(usize, u64, Vec<Value>)>,
    }

    impl JitState for CachingTestState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _: usize, _: &()) -> () {}
        fn extract_live(&self, _: &()) -> Vec<i64> {
            Vec::new()
        }
        fn create_sym(_: &(), _: usize) -> () {}
        fn is_compatible(&self, _: &()) -> bool {
            true
        }
        fn restore(&mut self, _: &(), _: &[i64]) {}

        fn restore_values(&mut self, _: &(), values: &[Value]) {
            self.restored_values = values.to_vec();
        }

        fn restore_reconstructed_frame_values_with_metadata(
            &mut self,
            _meta: &(),
            frame_index: usize,
            _total_frames: usize,
            frame: &crate::resume::ReconstructedFrame,
            values: &[Value],
            _exception: &ExceptionState,
        ) -> bool {
            self.restored_frames
                .push((frame_index, frame.pc, values.to_vec()));
            true
        }

        fn materialize_virtual_ref(
            &mut self,
            _meta: &(),
            virtual_index: usize,
            _materialized: &MaterializedVirtual,
        ) -> Option<GcRef> {
            self.materialize_call_count
                .set(self.materialize_call_count.get() + 1);
            Some(GcRef(0xB000 + virtual_index))
        }

        fn collect_jump_args(_: &()) -> Vec<OpRef> {
            Vec::new()
        }
        fn validate_close(_: &(), _: &()) -> bool {
            true
        }
    }

    #[test]
    fn test_deopt_materialization_cache_basic() {
        let mut cache = DeoptMaterializationCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.materialized_count(), 0);

        cache.insert(0, GcRef(0x100));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(0), Some(GcRef(0x100)));
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.materialized_count(), 1);

        cache.insert(2, GcRef(0x200));
        assert_eq!(cache.len(), 3); // 0, 1(None), 2
        assert_eq!(cache.get(2), Some(GcRef(0x200)));
        assert_eq!(cache.materialized_count(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.materialized_count(), 0);
    }

    #[test]
    fn test_materialization_cache_persists_across_guard_not_forced() {
        let mut state = CachingTestState::default();

        let virtuals_session1 = vec![
            MaterializedVirtual::Obj {
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                type_id: 2,
                descr_index: 1,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
        ];

        // First deopt: materialize A and B into session cache
        let mut session_cache = DeoptMaterializationCache::new();
        state.materialize_virtual_refs_into_cache(&(), &virtuals_session1, &mut session_cache);

        let calls_after_first = state.materialize_call_count.get();
        assert_eq!(calls_after_first, 2);
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));

        // Second deopt (GUARD_NOT_FORCED resumption) in same session:
        // same virtuals should NOT be re-materialized
        state.materialize_virtual_refs_into_cache(&(), &virtuals_session1, &mut session_cache);

        let calls_after_second = state.materialize_call_count.get();
        assert_eq!(
            calls_after_second, calls_after_first,
            "virtuals should not be re-materialized from cache"
        );
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));
    }

    #[test]
    fn test_materialization_cache_extends_for_new_virtuals() {
        let mut state = CachingTestState::default();

        let virtuals_first = vec![
            MaterializedVirtual::Obj {
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                type_id: 2,
                descr_index: 1,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
        ];

        // First deopt: materialize A and B
        let mut session_cache = DeoptMaterializationCache::new();
        state.materialize_virtual_refs_into_cache(&(), &virtuals_first, &mut session_cache);
        assert_eq!(state.materialize_call_count.get(), 2);

        // Second deopt adds virtual C
        let virtuals_second = vec![
            MaterializedVirtual::Obj {
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                type_id: 2,
                descr_index: 1,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
            MaterializedVirtual::Obj {
                type_id: 3,
                descr_index: 2,
                fields: vec![(0, MaterializedValue::Value(300))],
            },
        ];

        state.materialize_virtual_refs_into_cache(&(), &virtuals_second, &mut session_cache);

        // Only 1 new call for virtual C (A and B were cached)
        assert_eq!(state.materialize_call_count.get(), 3);
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));
        assert_eq!(session_cache.get(2), Some(GcRef(0xB002)));
    }

    #[test]
    fn test_virtual_ref_forcing_uses_session_cache() {
        let mut state = CachingTestState::default();
        let mut cache = DeoptMaterializationCache::new();

        let virt = MaterializedVirtual::Obj {
            type_id: 1,
            descr_index: 0,
            fields: vec![(0, MaterializedValue::Value(42))],
        };

        // First call: materializes and caches
        let result1 = state.force_virtual_with_cache(&(), &mut cache, 0, &virt);
        assert_eq!(result1, Some(GcRef(0xB000)));
        assert_eq!(state.materialize_call_count.get(), 1);

        // Second call: returns cached value without materializing
        let result2 = state.force_virtual_with_cache(&(), &mut cache, 0, &virt);
        assert_eq!(result2, Some(GcRef(0xB000)));
        assert_eq!(
            state.materialize_call_count.get(),
            1,
            "should not re-materialize cached virtual"
        );
    }

    #[test]
    fn test_restore_guard_failure_with_session_cache_integration() {
        let mut state = CachingTestState::default();

        let reconstructed_state = ReconstructedState {
            frames: vec![ReconstructedFrame {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slot_types: None,
                values: vec![
                    ReconstructedValue::Value(42),
                    ReconstructedValue::Virtual(0),
                ],
            }],
            virtuals: vec![MaterializedVirtual::Obj {
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            }],
            pending_fields: Vec::new(),
        };

        let frame_layouts = vec![ResumeFrameLayoutSummary::from_exit_frame_layout(
            &ExitFrameLayout {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slots: vec![
                    majit_codegen::ExitValueSourceLayout::ExitValue(0),
                    majit_codegen::ExitValueSourceLayout::ExitValue(1),
                ],
                slot_types: Some(vec![Type::Int, Type::Ref]),
            },
        )];

        let mut session_cache = DeoptMaterializationCache::new();

        // First deopt with session cache
        let restored = state.restore_guard_failure_with_session_cache(
            &(),
            &[42, 0],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &reconstructed_state.virtuals,
            &[],
            &ExceptionState::default(),
            Some(&mut session_cache),
        );
        assert!(restored);
        assert_eq!(state.materialize_call_count.get(), 1);

        // Verify cache was populated
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));

        // Reset state for second deopt
        state.restored_frames.clear();

        // Second deopt reusing the same session cache
        let restored2 = state.restore_guard_failure_with_session_cache(
            &(),
            &[42, 0],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &reconstructed_state.virtuals,
            &[],
            &ExceptionState::default(),
            Some(&mut session_cache),
        );
        assert!(restored2);
        assert_eq!(
            state.materialize_call_count.get(),
            1,
            "session cache should prevent re-materialization"
        );
    }

    #[test]
    fn test_restore_guard_failure_without_session_cache_backward_compat() {
        let mut state = CachingTestState::default();

        let reconstructed_state = ReconstructedState {
            frames: vec![ReconstructedFrame {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slot_types: None,
                values: vec![
                    ReconstructedValue::Value(42),
                    ReconstructedValue::Virtual(0),
                ],
            }],
            virtuals: vec![MaterializedVirtual::Obj {
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            }],
            pending_fields: Vec::new(),
        };

        let frame_layouts = vec![ResumeFrameLayoutSummary::from_exit_frame_layout(
            &ExitFrameLayout {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slots: vec![
                    majit_codegen::ExitValueSourceLayout::ExitValue(0),
                    majit_codegen::ExitValueSourceLayout::ExitValue(1),
                ],
                slot_types: Some(vec![Type::Int, Type::Ref]),
            },
        )];

        // Using the old API (no session cache) should still work
        let restored = state.restore_guard_failure_with_resume_layout(
            &(),
            &[42, 0],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &reconstructed_state.virtuals,
            &[],
            &ExceptionState::default(),
        );
        assert!(restored);
        assert_eq!(state.materialize_call_count.get(), 1);
    }

    #[test]
    fn test_session_cache_with_multi_frame_restore() {
        let mut state = CachingTestState::default();

        let reconstructed_state = ReconstructedState {
            frames: vec![
                ReconstructedFrame {
                    trace_id: Some(100),
                    header_pc: Some(500),
                    source_guard: None,
                    pc: 10,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Value(1),
                        ReconstructedValue::Virtual(0),
                    ],
                },
                ReconstructedFrame {
                    trace_id: Some(200),
                    header_pc: Some(600),
                    source_guard: None,
                    pc: 20,
                    slot_types: None,
                    values: vec![
                        ReconstructedValue::Virtual(1),
                        ReconstructedValue::Value(99),
                    ],
                },
            ],
            virtuals: vec![
                MaterializedVirtual::Obj {
                    type_id: 1,
                    descr_index: 0,
                    fields: vec![(0, MaterializedValue::Value(100))],
                },
                MaterializedVirtual::Obj {
                    type_id: 2,
                    descr_index: 1,
                    fields: vec![(0, MaterializedValue::Value(200))],
                },
            ],
            pending_fields: Vec::new(),
        };

        let frame_layouts = vec![
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(100),
                header_pc: Some(500),
                source_guard: None,
                pc: 10,
                slots: vec![
                    majit_codegen::ExitValueSourceLayout::ExitValue(0),
                    majit_codegen::ExitValueSourceLayout::ExitValue(1),
                ],
                slot_types: Some(vec![Type::Int, Type::Ref]),
            }),
            ResumeFrameLayoutSummary::from_exit_frame_layout(&ExitFrameLayout {
                trace_id: Some(200),
                header_pc: Some(600),
                source_guard: None,
                pc: 20,
                slots: vec![
                    majit_codegen::ExitValueSourceLayout::ExitValue(2),
                    majit_codegen::ExitValueSourceLayout::ExitValue(3),
                ],
                slot_types: Some(vec![Type::Ref, Type::Int]),
            }),
        ];

        let mut session_cache = DeoptMaterializationCache::new();

        // First deopt
        let restored = state.restore_guard_failure_with_session_cache(
            &(),
            &[1, 0, 0, 99],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &reconstructed_state.virtuals,
            &[],
            &ExceptionState::default(),
            Some(&mut session_cache),
        );
        assert!(restored);
        assert_eq!(state.materialize_call_count.get(), 2);
        assert_eq!(session_cache.materialized_count(), 2);

        // Second deopt with same cache — no re-materialization
        state.restored_frames.clear();
        let restored2 = state.restore_guard_failure_with_session_cache(
            &(),
            &[1, 0, 0, 99],
            Some(&reconstructed_state),
            Some(&frame_layouts),
            &reconstructed_state.virtuals,
            &[],
            &ExceptionState::default(),
            Some(&mut session_cache),
        );
        assert!(restored2);
        assert_eq!(
            state.materialize_call_count.get(),
            2,
            "multi-frame session cache should prevent re-materialization"
        );
    }
}
