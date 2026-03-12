use majit_ir::{OpRef, Type, Value};

use crate::virtualizable::VirtualizableInfo;

/// Interpreter-specific JIT state contract.
///
/// Defines how the interpreter's live state maps to/from the JIT's
/// representation. Interpreters implement this trait once; [`JitDriver`]
/// handles all lifecycle management automatically.
///
/// # Associated Types
///
/// - `Meta`: stored alongside each compiled loop (e.g., storage layout)
/// - `Sym`: symbolic state during tracing (e.g., `HashMap<usize, SymbolicStack>`)
/// - `Env`: environment context passed to `build_meta` (e.g., `Program`)
pub trait JitState: Sized {
    /// Metadata stored alongside each compiled loop.
    type Meta: Clone;

    /// Symbolic state maintained during tracing.
    type Sym;

    /// Environment context for `build_meta` (e.g., the program bytecodes).
    type Env: ?Sized;

    /// Whether this back-edge location can be traced.
    ///
    /// Return `false` for unsupported states. For example, aheuijit
    /// returns `false` when Queue or Port storage is selected.
    fn can_trace(&self) -> bool {
        true
    }

    /// Build metadata describing the current live state layout.
    ///
    /// Called at every back-edge before `on_back_edge`. Should be
    /// cheap to compute since it runs on every backward jump.
    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta;

    /// Extract concrete i64 values from the interpreter state.
    ///
    /// Values must be in the order defined by `meta`. These become
    /// the InputArgs for the traced loop.
    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64>;

    /// Extract concrete typed values from the interpreter state.
    ///
    /// This is the typed equivalent of [`extract_live`](Self::extract_live).
    /// The default implementation preserves the legacy integer-only contract by
    /// wrapping each `i64` as `Value::Int`.
    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        self.extract_live(meta)
            .into_iter()
            .map(Value::Int)
            .collect()
    }

    /// Create the initial symbolic state from InputArgs.
    ///
    /// `OpRef(0), OpRef(1), ...` map to the values from `extract_live`.
    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym;

    /// Check whether the current interpreter state is compatible
    /// with a compiled loop's assumptions.
    ///
    /// Called before executing compiled code. Return `false` to skip
    /// execution and fall back to interpretation.
    fn is_compatible(&self, meta: &Self::Meta) -> bool;

    /// Restore interpreter state from compiled loop output values.
    ///
    /// `values` are in the same order as `extract_live`.
    fn restore(&mut self, meta: &Self::Meta, values: &[i64]);

    /// Restore interpreter state from typed compiled loop output values.
    ///
    /// The default implementation round-trips through the legacy integer-only
    /// surface. Interpreters carrying ref/float reds should override this.
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

    /// Synchronize the named virtualizable before tracing or entering JIT code.
    ///
    /// This is the runtime seam where interpreters can flush heap-backed
    /// virtualizable state into their red variables before live-value
    /// extraction. Return `false` to reject tracing/execution for the current
    /// state.
    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> bool {
        true
    }

    /// Synchronize the named virtualizable after leaving JIT code.
    ///
    /// This lets interpreters flush restored red variables back to their
    /// heap-backed virtualizable object before resuming the non-JIT path.
    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) {
    }

    /// Synchronize the named virtualizable before tracing or entering JIT code.
    ///
    /// This higher-level hook always runs when declarative virtualizable
    /// metadata is present. `info` is `Some(...)` when the runtime has a
    /// configured `VirtualizableInfo`, and `None` when only the declarative
    /// driver metadata is available.
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

    /// Synchronize the named virtualizable after leaving JIT code.
    ///
    /// This is the `Option`-aware counterpart of
    /// [`sync_virtualizable_after_jit`](Self::sync_virtualizable_after_jit).
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

    /// Collect symbolic values for the JUMP instruction at loop close.
    ///
    /// Must return OpRefs in the same order as `extract_live`.
    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef>;

    /// Collect typed symbolic values for the JUMP instruction at loop close.
    ///
    /// Default delegates to `collect_jump_args` with all types as `Type::Int`.
    /// Mixed-type interpreters should override this.
    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::collect_jump_args(sym)
            .into_iter()
            .map(|opref| (opref, Type::Int))
            .collect()
    }

    /// Validate that symbolic state depths match the initial layout.
    ///
    /// If `false`, the trace is aborted (depth mismatch means the
    /// loop body doesn't preserve the state shape).
    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool;
}
