/// JIT optimization pipeline.
///
/// Translated from rpython/jit/metainterp/optimizeopt/.
///
/// The optimizer chains multiple passes, each implementing the OptimizationPass trait.
/// Operations flow through the chain: IntBounds → Rewrite → Virtualize → String →
/// Pure → Guard → Simplify → Heap (configurable).
pub mod bridgeopt;
pub mod earlyforce;
pub mod guard;
pub mod heap;
pub mod info;
pub mod intbounds;
pub mod intdiv;
pub mod intutils;
pub mod optimizer;
pub mod pure;
pub mod rewrite;
pub mod shortpreamble;
pub mod simplify;
pub mod unroll;
pub mod vector;
pub mod virtualize;
pub mod virtualstate;
pub mod vstring;
pub mod walkvirtual;

use majit_ir::{Op, OpRef, Value};

/// Result of an optimization pass processing an operation.
#[derive(Debug)]
pub enum PassResult {
    /// Emit this operation (possibly modified).
    Emit(Op),
    /// Replace with a different operation.
    Replace(Op),
    /// Remove the operation entirely.
    Remove,
    /// Pass the operation to the next pass unchanged.
    PassOn,
}

/// Context provided to optimization passes.
///
/// Holds the shared state that passes read from and write to.
pub struct OptContext {
    /// The output operation list being built.
    pub new_operations: Vec<Op>,
    /// Constants known at optimization time (op -> value).
    pub constants: Vec<Option<Value>>,
    /// Forwarding chain: maps old OpRef to replacement OpRef.
    pub forwarding: Vec<OpRef>,
}

impl OptContext {
    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            constants: Vec::new(),
            forwarding: Vec::new(),
        }
    }

    /// Emit an operation to the output.
    pub fn emit(&mut self, op: Op) -> OpRef {
        let idx = self.new_operations.len();
        let opref = OpRef(idx as u32);
        self.new_operations.push(op);
        opref
    }

    /// Record that `old` should be replaced by `new` wherever it appears.
    pub fn replace_op(&mut self, old: OpRef, new: OpRef) {
        if old == new {
            return; // avoid self-referencing forwarding loop
        }
        let idx = old.0 as usize;
        if idx >= self.forwarding.len() {
            self.forwarding.resize(idx + 1, OpRef::NONE);
        }
        self.forwarding[idx] = new;
    }

    /// Follow the forwarding chain to get the current replacement for `opref`.
    pub fn get_replacement(&self, mut opref: OpRef) -> OpRef {
        loop {
            let idx = opref.0 as usize;
            if idx >= self.forwarding.len() {
                return opref;
            }
            let next = self.forwarding[idx];
            if next.is_none() {
                return opref;
            }
            opref = next;
        }
    }

    /// Record that an operation produces a known constant value.
    pub fn make_constant(&mut self, opref: OpRef, value: Value) {
        let idx = opref.0 as usize;
        if idx >= self.constants.len() {
            self.constants.resize(idx + 1, None);
        }
        self.constants[idx] = Some(value);
    }

    /// Get the constant value for an operation, if known.
    pub fn get_constant(&self, opref: OpRef) -> Option<&Value> {
        let opref = self.get_replacement(opref);
        let idx = opref.0 as usize;
        self.constants.get(idx).and_then(|v| v.as_ref())
    }

    /// Get constant integer value, if known.
    pub fn get_constant_int(&self, opref: OpRef) -> Option<i64> {
        self.get_constant(opref).and_then(|v| match v {
            Value::Int(i) => Some(*i),
            _ => None,
        })
    }
}

/// An optimization pass.
///
/// Mirrors rpython/jit/metainterp/optimizeopt/optimizer.py Optimization.
pub trait OptimizationPass {
    /// Process an operation. Called for each operation in the trace.
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> PassResult;

    /// Called once before optimization starts.
    fn setup(&mut self) {}

    /// Called after all operations have been processed.
    fn flush(&mut self) {}

    /// Name of this pass (for debugging).
    fn name(&self) -> &'static str;
}
