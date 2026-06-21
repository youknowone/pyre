//! Re-exports of the `majit_ir::box_ref` Box identity types
//! (`BoxRef` / `Forwarded` / `PtrInfoBorrowMut`) plus shared test helpers.
//! Box / BoxKind / BoxRef / Forwarded themselves live in
//! `majit_ir::box_ref`.

pub use majit_ir::box_ref::{BoxRef, Forwarded, PtrInfoBorrowMut};

#[cfg(test)]
pub(crate) mod test_support {
    //! Shared test helpers for constructing **bound** BoxRefs. Production
    //! recorder→TreeLoop handoff binds every `AbstractInputArg` /
    //! `AbstractResOp` BoxRef to its `InputArg` / `Op` identity, so tests
    //! that seed bound boxes directly must do the same to match
    //! `make_equal_to`'s `Forwarded::Op` / `Forwarded::InputArg` chain
    //! shape.
    use majit_ir::box_ref::BoxRef;
    use majit_ir::resoperation::{Op, OpCode, OpRc};
    use majit_ir::{InputArg, InputArgRc, OpRef, Type, Value};

    /// Bind a fresh `BoxRef::new_inputarg(tp, index)` to a fresh
    /// `InputArgRc` (`box_ref.rs:354 bind_inputarg`). The returned
    /// `InputArgRc` must outlive every read of `box.get_forwarded()`
    /// for the bound `Weak<InputArg>` upgrade to stay live.
    pub(crate) fn bound_inputarg_box(tp: Type, index: u32) -> (BoxRef, InputArgRc) {
        let b = BoxRef::new_inputarg(tp, index);
        let ia = std::rc::Rc::new(InputArg::from_type(tp, index));
        b.bind_inputarg(&ia);
        (b, ia)
    }

    /// Bind a fresh `BoxRef::new_resop(tp, position)` to a fresh
    /// `OpRc` (`box_ref.rs:322 bind_op`). The Op carries `SameAsI/F/R`
    /// (or `Jump` for `Type::Void`) so its `opcode.result_type()`
    /// matches `tp`; `pos` is seeded with the typed OpRef so
    /// `BoxRef::from_bound_op` materialises a transient terminal box
    /// of the correct type during chain walks.
    pub(crate) fn bound_resop_box(tp: Type, position: u32) -> (BoxRef, OpRc) {
        let b = BoxRef::new_resop(tp, position);
        let opcode = match tp {
            Type::Int => OpCode::SameAsI,
            Type::Float => OpCode::SameAsF,
            Type::Ref => OpCode::SameAsR,
            Type::Void => OpCode::Jump,
        };
        let op = std::rc::Rc::new(Op::new(opcode, &[]));
        op.pos.set(OpRef::op_typed(position, tp));
        b.bind_op(&op);
        (b, op)
    }

    thread_local! {
        /// Roots synthetic producer `Op` / `InputArg` Rcs for the test
        /// thread's lifetime so a single returned bound `BoxRef` stays bound
        /// (its `Weak` upgrades) without the caller threading the producer
        /// Rc. Test-only and never cleared — bounded by the fixtures one
        /// test thread builds.
        static PRODUCER_ROOTS: std::cell::RefCell<Vec<std::rc::Rc<dyn std::any::Any>>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Drop-in for `BoxRef::from_opref(OpRef::{int,ref,float}_op(N))` at op
    /// argument / fail-arg sites: a bound ResOp box (sheds to `Operand::Op`,
    /// `to_opref()`s to the same `(tp, position)`) whose synthetic producer
    /// is kept alive by the thread-local pool, so callers can use it inline
    /// in an `Op::new`/`setarg` arg list without retaining the `OpRc`.
    pub(crate) fn rooted_resop_box(tp: Type, position: u32) -> BoxRef {
        let (b, op) = bound_resop_box(tp, position);
        let rooted: std::rc::Rc<dyn std::any::Any> = op;
        PRODUCER_ROOTS.with(|p| p.borrow_mut().push(rooted));
        b
    }

    /// Drop-in for `BoxRef::from_opref(OpRef::input_arg_{int,ref,float}(N))`
    /// at op argument / fail-arg sites: a bound InputArg box (sheds to
    /// `Operand::InputArg`), producer rooted in the thread-local pool.
    pub(crate) fn rooted_inputarg_box(tp: Type, index: u32) -> BoxRef {
        let (b, ia) = bound_inputarg_box(tp, index);
        let rooted: std::rc::Rc<dyn std::any::Any> = ia;
        PRODUCER_ROOTS.with(|p| p.borrow_mut().push(rooted));
        b
    }

    /// oparser-faithful trace builder for optimizer unit tests
    /// (`rpython/jit/tool/oparser.py`). Each producer op is registered as a
    /// live `OpRc` and each consumer arg references the producing op's bound
    /// result box (`from_bound_op`) — mirroring oparser's `self.vars[name] =
    /// resop; args.append(self.vars[arg])` object-identity wiring. Every arg
    /// therefore sheds to `Operand::Op` / `Operand::InputArg` / `Operand::Const`
    /// at construction (never the position-only `Operand::Box`).
    ///
    /// Drive the optimizer with `optimize_with_constants_and_inputs_oprc` (the
    /// `input_ops_from_ops = true` entry) so the builder's `OpRc`s are threaded
    /// as the canonical producers `find_producer_op` resolves to; a forwarding
    /// write through a consumer arg then lands on the SAME `Op` the optimizer
    /// indexes, with no detached synthetic to diverge.
    pub(crate) struct TraceBuilder {
        ops: Vec<OpRc>,
        inputs: Vec<Type>,
        next_pos: u32,
    }

    impl TraceBuilder {
        pub(crate) fn new() -> Self {
            Self {
                ops: Vec::new(),
                inputs: Vec::new(),
                next_pos: 0,
            }
        }

        /// Header input var (oparser `[i0]`): a bound `InputArg` box at
        /// `index`, recorded so [`Self::build`] can emit a matching
        /// `trace_inputargs` slot. The producer `InputArgRc` is rooted in the
        /// thread-local pool so the box's `Weak` upgrade stays live.
        pub(crate) fn input(&mut self, tp: Type, index: u32) -> BoxRef {
            let idx = index as usize;
            if idx >= self.inputs.len() {
                self.inputs.resize(idx + 1, Type::Int);
            }
            self.inputs[idx] = tp;
            rooted_inputarg_box(tp, index)
        }

        /// Const literal arg (oparser numeric/`ConstInt` literal).
        pub(crate) fn const_int(&self, v: i64) -> BoxRef {
            BoxRef::new_const(Value::Int(v))
        }

        /// Append a producer op (oparser `iN = opcode(args)`), assigning it the
        /// next sequential result position. Returns the producing op's bound
        /// result box for use as a later consumer arg.
        pub(crate) fn op(&mut self, opcode: OpCode, args: &[BoxRef]) -> BoxRef {
            let op = std::rc::Rc::new(Op::new(opcode, args));
            op.pos
                .set(OpRef::op_typed(self.next_pos, opcode.result_type()));
            self.next_pos += 1;
            let result = BoxRef::from_bound_op(&op);
            self.ops.push(op);
            result
        }

        /// Append a producer op carrying a descr.
        pub(crate) fn op_with_descr(
            &mut self,
            opcode: OpCode,
            args: &[BoxRef],
            descr: majit_ir::DescrRef,
        ) -> BoxRef {
            let op = std::rc::Rc::new(Op::with_descr(opcode, args, descr));
            op.pos
                .set(OpRef::op_typed(self.next_pos, opcode.result_type()));
            self.next_pos += 1;
            let result = BoxRef::from_bound_op(&op);
            self.ops.push(op);
            result
        }

        /// Finish the trace: returns the canonical `OpRc` slice to pass to
        /// `optimize_with_constants_and_inputs_oprc`, plus the per-index input
        /// types to install as `opt.trace_inputargs` (via
        /// `OpRef::inputarg_refs`).
        pub(crate) fn build(self) -> (Vec<OpRc>, Vec<Type>) {
            (self.ops, self.inputs)
        }
    }
}
