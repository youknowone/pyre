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
    use majit_ir::{InputArg, InputArgRc, OpRef, Type};

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
}
