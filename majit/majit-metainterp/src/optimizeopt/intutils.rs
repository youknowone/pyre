//! `IntBound` re-export + extension trait for optimizer-bound emission.
//!
//! The data type and its pure leaf methods live in `majit-ir::intbound`
//! so the `Forwarded`/`OpInfo` host types can reference `IntBound`
//! without a circular dep back into `majit-metainterp`. The one method
//! that materialises guard operations into the optimizer's pool stays
//! here, because it needs `Op` / `OptContext` from this crate.

pub use majit_ir::intbound::IntBound;
pub use majit_ir::optimize::InvalidLoop;

/// Extension trait that materialises the guards implied by an
/// `IntBound` into the optimizer's pending guard list.
///
/// Lives outside `IntBound` itself because it needs the metainterp
/// `Op` / `OptContext` types. Imported wherever `bound.make_guards(...)`
/// is invoked.
pub(crate) trait IntBoundMakeGuards {
    /// intutils.py `IntBound.make_guards(box, guards, optimizer)`
    /// (line-by-line port).
    ///
    /// ```python
    /// def make_guards(self, box, guards, optimizer):
    ///     if self.is_constant():
    ///         guards.append(ResOperation(rop.GUARD_VALUE,
    ///                                    [box, ConstInt(self.upper)]))
    ///         return
    ///     if self.lower > MININT:
    ///         bound = self.lower
    ///         op = ResOperation(rop.INT_GE, [box, ConstInt(bound)])
    ///         guards.append(op)
    ///         op = ResOperation(rop.GUARD_TRUE, [op])
    ///         guards.append(op)
    ///     if self.upper < MAXINT:
    ///         bound = self.upper
    ///         op = ResOperation(rop.INT_LE, [box, ConstInt(bound)])
    ///         guards.append(op)
    ///         op = ResOperation(rop.GUARD_TRUE, [op])
    ///         guards.append(op)
    ///     if not self._are_knownbits_implied():
    ///         op = ResOperation(rop.INT_AND, [box, ConstInt(intmask(~self.tmask))])
    ///         guards.append(op)
    ///         op = ResOperation(rop.GUARD_VALUE, [op, ConstInt(intmask(self.tvalue))])
    ///         guards.append(op)
    /// ```
    ///
    /// Each `INT_GE` / `INT_LE` / `INT_AND` is followed by a guard whose
    /// first argument is the *result* of that op, not `box`. The guard
    /// binds the producer's `OpRc` itself, so the chained guard pair names
    /// the object pushed into `guards`. The producer still takes a fresh
    /// Int position, which the positional short-preamble replay keys on.
    fn make_guards(
        &self,
        box_: &majit_ir::operand::Operand,
        guards: &mut Vec<majit_ir::OpRc>,
        ctx: &mut crate::optimizeopt::OptContext,
    );
}

impl IntBoundMakeGuards for IntBound {
    fn make_guards(
        &self,
        box_: &majit_ir::operand::Operand,
        guards: &mut Vec<majit_ir::OpRc>,
        ctx: &mut crate::optimizeopt::OptContext,
    ) {
        use crate::optimizeopt::Op;
        use majit_ir::operand::Operand;
        use majit_ir::{OpCode, OpRc, Type, Value};

        // history.py ConstInt: the value rides inline on the operand.
        let const_int = |v: i64| Operand::const_from_value(Value::Int(v));
        // `op = ResOperation(opnum, [box, ConstInt(c)]); guards.append(op)`
        let mut producer = |opcode: OpCode, c: i64, guards: &mut Vec<OpRc>| {
            let op = OpRc::new(Op::new(opcode, &[box_.clone(), const_int(c)]));
            op.pos().set(ctx.alloc_op_position_typed(Type::Int));
            guards.push(op.clone());
            Operand::from_bound_op(&op)
        };
        if self.is_constant() {
            guards.push(OpRc::new(Op::new(
                OpCode::GuardValue,
                &[box_.clone(), const_int(self.upper)],
            )));
            return;
        }
        if self.lower > i64::MIN {
            let op = producer(OpCode::IntGe, self.lower, guards);
            guards.push(OpRc::new(Op::new(OpCode::GuardTrue, &[op])));
        }
        if self.upper < i64::MAX {
            let op = producer(OpCode::IntLe, self.upper, guards);
            guards.push(OpRc::new(Op::new(OpCode::GuardTrue, &[op])));
        }
        if !self.are_knownbits_implied() {
            let op = producer(OpCode::IntAnd, !self.tmask as i64, guards);
            guards.push(OpRc::new(Op::new(
                OpCode::GuardValue,
                &[op, const_int(self.tvalue as i64)],
            )));
        }
    }
}
