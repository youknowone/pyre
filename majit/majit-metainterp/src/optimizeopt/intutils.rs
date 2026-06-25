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
    /// intutils.py:1264-1289 `IntBound.make_guards(box, guards, optimizer)`
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
    /// first argument is the *result* of that op, not `box`. The previous
    /// pyre signature returned `Vec<(OpCode, i64)>` and the caller emitted
    /// every guard against `box` directly, dropping the chained guard pair
    /// and leaving the bound silently unenforced.
    ///
    /// RPython relies on Python-object identity: `op = ResOperation(...)`
    /// and then `[op]` references the same object as the next guard's
    /// first argument (intutils.py:1275-1284). Pyre's flat-OpRef model
    /// has no implicit identity, so each `INT_GE` / `INT_LE` / `INT_AND`
    /// producer allocates a fresh Int OpRef into `op.pos` *before* the
    /// consumer guard is constructed. Constants land in the constant
    /// pool via `reserve_const_ref` + `seed_constant` (mirroring inline
    /// `ConstInt(bound)` in upstream).
    fn make_guards(
        &self,
        box_ref: majit_ir::OpRef,
        guards: &mut Vec<crate::optimizeopt::Op>,
        ctx: &mut crate::optimizeopt::OptContext,
    );
}

impl IntBoundMakeGuards for IntBound {
    fn make_guards(
        &self,
        box_ref: majit_ir::OpRef,
        guards: &mut Vec<crate::optimizeopt::Op>,
        ctx: &mut crate::optimizeopt::OptContext,
    ) {
        use crate::optimizeopt::Op;
        use majit_ir::operand::Operand;
        use majit_ir::{OpCode, Type, Value};

        // history.py:227/268/314 Const{Int,Float,Ptr}.value inline: the
        // value rides on the inline-Const OpRef variant tag itself, so no
        // `seed_constant` step is needed (its const arm is a no-op).
        // IntBound guards mint Int values only, so the match is total.
        let alloc_const = |value: Value| match value {
            Value::Int(v) => majit_ir::OpRef::const_int(v),
            Value::Float(v) => majit_ir::OpRef::const_float(v),
            Value::Ref(v) => majit_ir::OpRef::const_ptr(v),
            Value::Void => panic!("alloc_const: ConstVoid not allowed"),
        };
        if self.is_constant() {
            let c = alloc_const(Value::Int(self.upper));
            let arg_box = ctx.materialize_operand_at(box_ref);
            let arg_c = ctx.materialize_operand_at(c);
            guards.push(Op::new(
                OpCode::GuardValue,
                &[arg_box.clone(), arg_c.clone()],
            ));
            return;
        }
        if self.lower > i64::MIN {
            let bound = alloc_const(Value::Int(self.lower));
            let arg_box = ctx.materialize_operand_at(box_ref);
            let arg_bound = ctx.materialize_operand_at(bound);
            let mut op = Op::new(OpCode::IntGe, &[arg_box.clone(), arg_bound.clone()]);
            // intutils.py:1275 `op = ResOperation(rop.INT_GE, ...)` then
            // `[op]` — RPython uses the ResOperation object as identity.
            // pyre allocates a fresh Int OpRef into `op.pos` so the next
            // guard's arg vector captures the producer's result, not the
            // sentinel `OpRef::NONE` left over from `Op::new`.
            op.pos.set(ctx.alloc_op_position_typed(Type::Int));
            let op_pos = op.pos.get();
            guards.push(op);
            let arg_op = ctx.materialize_operand_at(op_pos);
            guards.push(Op::new(OpCode::GuardTrue, &[arg_op.clone()]));
        }
        if self.upper < i64::MAX {
            let bound = alloc_const(Value::Int(self.upper));
            let arg_box = ctx.materialize_operand_at(box_ref);
            let arg_bound = ctx.materialize_operand_at(bound);
            let mut op = Op::new(OpCode::IntLe, &[arg_box.clone(), arg_bound.clone()]);
            // intutils.py:1281 INT_LE producer identity — see comment above.
            op.pos.set(ctx.alloc_op_position_typed(Type::Int));
            let op_pos = op.pos.get();
            guards.push(op);
            let arg_op = ctx.materialize_operand_at(op_pos);
            guards.push(Op::new(OpCode::GuardTrue, &[arg_op.clone()]));
        }
        if !self.are_knownbits_implied() {
            let mask = alloc_const(Value::Int(!self.tmask as i64));
            let arg_box = ctx.materialize_operand_at(box_ref);
            let arg_mask = ctx.materialize_operand_at(mask);
            let mut op = Op::new(OpCode::IntAnd, &[arg_box.clone(), arg_mask.clone()]);
            // intutils.py:1286 INT_AND producer identity — see comment above.
            op.pos.set(ctx.alloc_op_position_typed(Type::Int));
            let op_pos = op.pos.get();
            guards.push(op);
            let value = alloc_const(Value::Int(self.tvalue as i64));
            let arg_op = ctx.materialize_operand_at(op_pos);
            let arg_value = ctx.materialize_operand_at(value);
            guards.push(Op::new(
                OpCode::GuardValue,
                &[arg_op.clone(), arg_value.clone()],
            ));
        }
    }
}
