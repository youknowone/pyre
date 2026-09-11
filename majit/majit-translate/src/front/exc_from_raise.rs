//! RPython `exc_from_raise` lowering for raise sites.
//!
//! ## Positioning
//!
//! The authoritative reference implementation is
//! `flowspace::flowcontext::FlowContext::exc_from_raise`
//! (`majit-translate/src/flowspace/flowcontext.rs`), which is a
//! line-by-line port of upstream `rpython/flowspace/flowcontext.py:600`:
//!
//! ```python
//! def exc_from_raise(self, w_arg1, w_arg2):
//!     check_not_none = False
//!     w_is_type = op.isinstance(w_arg1, const(type)).eval(self)
//!     if self.guessbool(w_is_type):
//!         if self.guessbool(op.is_(w_arg2, w_None).eval(self)):
//!             w_value = op.simple_call(w_arg1).eval(self)
//!         else:
//!             w_valuetype = op.type(w_arg2).eval(self)
//!             if self.guessbool(op.issubtype(w_valuetype, w_arg1).eval(self)):
//!                 w_value = w_arg2
//!                 check_not_none = True
//!             else:
//!                 w_value = op.simple_call(w_arg1, w_arg2).eval(self)
//!     else:
//!         ...
//!     if check_not_none:
//!         w_value = op.simple_call(const(ll_assert_not_none), w_value).eval(self)
//!     w_type = op.type(w_value).eval(self)
//!     return FSException(w_type, w_value)
//! ```
//!
//! `front::mir` lowering reaches the raise machinery through
//! Rust macros (`panic!`, `assert!`, `unreachable!`, …) whose adapter
//! always produces the *(Class, optional args)* shape — `w_arg1` is a
//! statically known exception class, and whatever the macro passes as
//! a message becomes the rest of `simple_call(w_arg1, *args)`. That
//! corresponds to the constant-foldable slice of the flowspace
//! implementation: `is_type_const=True` + either `arg2_is_none_const=True`
//! (no message) or no `issubtype` hit (message is not a pre-built
//! exception instance), which reduces to:
//!
//! ```text
//! evalue = op.simple_call(const(exc_class), *message_args)
//! graph.set_raise_values(block, evalue, evalue)
//! ```
//!
//! ### The `etype` link arg
//!
//! Upstream's tail line is `w_type = op.type(w_value)`, but that value never
//! reaches a jitcode. `make_bytecode_block` hands the `exceptblock`'s
//! `inputargs` to `make_return` (`flatten.py`), whose 2-arg arm emits
//! `-live-` + `raise self.getcolor(args[1])` and never touches `args[0]`
//! (`flatten.py:139-143`, mirrored at `flatten.rs`). So the slot the
//! `etype` link arg feeds has no consumer in any emitted bytecode.
//! `set_raise_values` therefore receives `evalue` for both slots instead of a
//! synthesised `type(evalue)` call, which would be a residual with a dead
//! result register on every raise tail.
//!
//! ### Operand shape
//!
//! Upstream encodes the exception class as a `Constant(class_obj)`
//! living in `SpaceOperation.args` — `flowspace/model.py` makes
//! `args` a mixed `Variable | Constant` list, and
//! `flowspace/operation.py SimpleCall.eval` reads
//! `w_callable, args_w = self.args[0], self.args[1:]`.  The helper
//! emits that list: `args[0]` is
//! `LinkArg::Const(HostObject(exc_class))` from
//! `HOST_ENV.lookup_builtin`, and the message Variables occupy
//! `args[1..]`.  The Call's `FunctionPath` is the op name
//! `["simple_call"]`; the adapter does not wrap a second callable.
//!
//! The Rust macro → RPython exception class mapping is an adapter
//! decision (mirroring upstream's per-bytecode adapter at
//! `flowcontext.py:2861 Instruction::RaiseVarargs`, which maps
//! `BareRaise` / `Raise` / `RaiseCause` to the canonical `(w_arg1,
//! w_arg2)` input of `exc_from_raise`).  Keeping the version- /
//! host-specific wiring out of this helper means `front::mir` lowering
//! and flowspace lowering both converge on the same inlined op sequence.
//!
//! ## What this helper is not
//!
//! - It is **not** a synthetic helper. The call target emitted here
//!   is the RPython op name itself (`simple_call`), so any downstream
//!   reader sees the same op namespace upstream uses.
//! - There are no `__pyre_exc_from_raise__` / `__pyre_exception_type_of__`
//!   opaque Call targets any more — that earlier deviation is removed
//!   by the same change that introduced this module.

use crate::flowspace::model::{ConstValue, HOST_ENV, Variable};
use crate::model::{BlockId, CallTarget, FunctionGraph, LinkArg, OpKind, ValueType};

/// Close `block` with an `(etype, evalue)` Link to `exceptblock`
/// whose value comes from the canonical RPython `exc_from_raise` op
/// sequence (`op.simple_call(const(exc_class), *args)`).  The `etype`
/// slot reuses `evalue` — see the module-level "etype link arg" note.
///
/// `exc_class_name` is the Python-layer exception class name
/// (`"AssertionError"`, `"ValueError"`, …).  It is resolved through
/// `HOST_ENV.lookup_builtin` and placed at `simple_call.args[0]` as
/// a `Constant`, matching `flowcontext.py exc_from_raise`
/// (`op.simple_call(w_arg1, …)` when `w_arg1` is a class Constant).
///
/// `message_args` is the pre-evaluated list of message `Variable`s
/// (side effects already on the graph from the caller's walk).
/// Empty for bare `panic!()` / `assert!(cond)`; single element for
/// `panic!(msg)` / `assert!(cond, msg)`; multiple for
/// `panic!("fmt", a, b)`.  They become `simple_call.args[1..]`.
/// `simple_call` is variadic upstream (`flowspace/operation.py
/// SimpleCall` / `CallOp.args = [callable, *args]`).
#[allow(dead_code)] // exercised by tests; raise-class sites call this helper
pub(crate) fn lower_exc_from_raise(
    graph: &mut FunctionGraph,
    block: BlockId,
    exc_class_name: &str,
    message_args: Vec<Variable>,
) {
    // `op.simple_call(const(exc_class), *args)` — `flowcontext.py`.
    let exc_class = HOST_ENV.lookup_builtin(exc_class_name).unwrap_or_else(|| {
        panic!("lower_exc_from_raise: {exc_class_name} is not a HOST_ENV builtin exception class")
    });
    let mut args = Vec::with_capacity(message_args.len() + 1);
    args.push(LinkArg::from(ConstValue::HostObject(exc_class)));
    args.extend(message_args.into_iter().map(LinkArg::from));
    let evalue_var = graph
        .push_op_var(
            block,
            OpKind::Call {
                target: CallTarget::function_path(["simple_call"]),
                args,
                result_ty: ValueType::Ref(None),
            },
            true,
        )
        .expect("op.simple_call(exc_class, ...) must produce a Ref exception instance");
    // `flowspace/flowcontext.py Raise.nomoreblocks` — close the block
    // with the `(etype, evalue)` Link to the graph's `exceptblock`.
    //
    // Upstream's `w_type = op.type(w_value)` (`flowcontext.py`) lives at
    // flow-space level only — see the module-level "etype link arg" note: the
    // 2-arg `make_return` arm emits `raise <args[1]>` and never reads
    // `args[0]`, and `make_exception_link` drops both for a direct `reraise`.
    // So the `etype` link arg is write-only in every emitted jitcode. The
    // emitted drain tail is `-live-` + `raise <evalue>` and nothing else.
    // Materialising it as a `type(evalue)` call left a residual whose
    // result register is dead by construction on every raise tail, and
    // made the raise arm unwalkable in the blackhole (canonical
    // `inline_call_*` on a callee with no runtime address). Reuse
    // `evalue`: same ref kind, no new op.
    graph.set_raise_values(block, evalue_var.clone(), evalue_var);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CallTarget, FunctionGraph, OpKind};

    #[test]
    fn lower_exc_from_raise_puts_class_constant_at_args_0() {
        let mut graph = FunctionGraph::new("exc_from_raise_shape");
        let start = graph.startblock;
        let msg = graph.alloc_value_var();
        lower_exc_from_raise(&mut graph, start, "ValueError", vec![msg.clone()]);
        let op = graph.blocks[start.0]
            .operations
            .last()
            .expect("simple_call must be emitted");
        let OpKind::Call { target, args, .. } = &op.kind else {
            panic!("expected Call, got {:?}", op.kind);
        };
        assert_eq!(
            target,
            &CallTarget::function_path(["simple_call"]),
            "FunctionPath is the op name, not the class"
        );
        assert_eq!(args.len(), 2);
        let LinkArg::Const(class) = &args[0] else {
            panic!("args[0] must be the class Constant");
        };
        let expected = HOST_ENV
            .lookup_builtin("ValueError")
            .expect("ValueError is a builtin exception");
        assert!(
            matches!(&class.value, ConstValue::HostObject(h) if *h == expected),
            "args[0] must be HOST_ENV ValueError"
        );
        assert_eq!(args[1], LinkArg::from(msg));
    }
}
