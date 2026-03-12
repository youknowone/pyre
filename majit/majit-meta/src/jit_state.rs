use majit_ir::OpRef;

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

    /// Collect symbolic values for the JUMP instruction at loop close.
    ///
    /// Must return OpRefs in the same order as `extract_live`.
    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef>;

    /// Validate that symbolic state depths match the initial layout.
    ///
    /// If `false`, the trace is aborted (depth mismatch means the
    /// loop body doesn't preserve the state shape).
    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool;
}
