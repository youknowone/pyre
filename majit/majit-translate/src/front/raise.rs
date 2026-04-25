//! RPython `exc_from_raise` lowering for AST-sourced raise sites.
//!
//! ## Positioning
//!
//! The authoritative reference implementation is
//! `flowspace::flowcontext::FlowContext::exc_from_raise`
//! (`majit-translate/src/flowspace/flowcontext.rs:1189`), which is a
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
//! AST lowering (`front/ast.rs`) reaches the raise machinery through
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
//! etype  = op.type(evalue)
//! graph.set_raise_values(block, etype, evalue)
//! ```
//!
//! ### PRE-EXISTING-ADAPTATION — `Constant` SSA carrier shape
//!
//! Upstream RPython encodes the exception class operand as a
//! `Constant(class_obj)` SSA value living directly inside
//! `SpaceOperation.args` — `flowspace/model.py:354` defines
//! `Constant(value)`, `flowspace/model.py:436` makes `args` a mixed
//! `Variable | Constant` list, and `flowspace/operation.py:666
//! SimpleCall.eval` reads `w_callable, args_w = self.args[0],
//! self.args[1:]` — that is, `args[0]` is *itself* the Constant
//! carrier, with no producing op.
//!
//! Pyre's `FunctionGraph` currently represents every SSA value
//! through `ValueId(usize)` produced by a `SpaceOperation`; the
//! op-args slot is `Vec<ValueId>` (no `LinkArg`-style mixed enum).
//! The properly orthodox fix is to migrate `OpKind::Call.args` to
//! `Vec<LinkArg>` (RPython parity rule #1: structural equivalence)
//! and update every consumer (~80 sites across 14 files spanning
//! `front/`, `inline.rs`, `jit_codewriter/{call,jtransform,liveness,
//! regalloc,assembler,format,flatten,support}.rs`,
//! `translator/rtyper/rpbc.rs`, plus tests).  That is a multi-
//! session port and is **deferred**.
//!
//! Until the `Vec<LinkArg>` migration lands, this helper carries the
//! constant exception class as the *second segment* of the
//! `simple_call` Call's `FunctionPath` target — `["simple_call",
//! exc_class_name]` — instead of as `simple_call.args[0]`.  That
//! preserves the per-segment string identity of the class without
//! needing a real `Constant` SSA carrier, but is **not** RPython-
//! orthodox at the operand level (`args[0]` should be a Constant,
//! not a path segment).  Three earlier attempts to remove this
//! adaptation each ran into a different blocker:
//!
//! 1. **`["const", X]` Call producer** — emitted a 0-arg Call to
//!    materialise the class as a real `args[0]` Variable.  Reviewer
//!    rejected it because RPython has no `const` op at all (Constants
//!    are SSA values, not ops).
//! 2. **`FunctionGraph.constants` side table + `alloc_const`** — gave
//!    each constant a real `ValueId` with no producer op, mirroring
//!    upstream's "Constant has no producer".  Broke regalloc /
//!    assembler `lookup_coloring` because the downstream pipeline
//!    requires every `args` `ValueId` to have a regalloc coloring,
//!    which non-Variable Constants don't have.  Plumbing constants
//!    through liveness / regalloc / assembler is multi-session.
//! 3. **`OpKind::Const { value, result_ty }`** — a typed parallel of
//!    `OpKind::ConstInt(i64)`.  Reviewer would still reject because
//!    `ConstInt` is itself an existing PRE-EXISTING-ADAPTATION
//!    (RPython has no `const_int` op either), so adding more of the
//!    same does not improve parity.  Also `ConstValue` lacks
//!    `Serialize`/`Deserialize` derives, so wiring it into `OpKind`
//!    bytes-stable serialisation is its own deep change.
//!
//! Once the `Vec<LinkArg>` migration is on the roadmap, the helper
//! collapses to a `LinkArg::Const(ConstValue::HostObject(class))`
//! at `args[0]` and this comment block goes away.
//!
//! ### Operand-position parity (the non-class args)
//!
//! `simple_call`'s *non-class* args (the message values) are real
//! Variables and live at `args[1..]`, matching upstream
//! `flowspace/operation.py:666 SimpleCall.eval` reading
//! `args_w = self.args[1:]`.  This part is already orthodox — only
//! the operand-0 class slot is the deviation noted above.
//!
//! The Rust macro → RPython exception class mapping is an adapter
//! decision (mirroring upstream's per-bytecode adapter at
//! `flowcontext.py:2861 Instruction::RaiseVarargs`, which maps
//! `BareRaise` / `Raise` / `RaiseCause` to the canonical `(w_arg1,
//! w_arg2)` input of `exc_from_raise`).  Keeping the version- /
//! host-specific wiring out of this helper means AST lowering and
//! flowspace lowering both converge on the same inlined op sequence.
//!
//! ## What this helper is not
//!
//! - It is **not** a synthetic helper. The call targets emitted here
//!   are the RPython op names themselves (`simple_call`, `type`), so
//!   any downstream reader sees the same op namespace upstream uses.
//! - There are no `__pyre_exc_from_raise__` / `__pyre_exception_type_of__`
//!   opaque Call targets any more — that earlier deviation is removed
//!   by the same change that introduced this module.

use crate::model::{BlockId, CallTarget, FunctionGraph, OpKind, ValueId, ValueType};

/// Close `block` with an `(etype, evalue)` Link to `exceptblock`
/// whose values come from the canonical RPython `exc_from_raise`
/// op sequence (`op.simple_call(const(exc_class), *args)` followed
/// by `op.type(evalue)`).
///
/// `exc_class_name` is the Python-layer exception class name
/// (`"AssertionError"`, `"PanicError"`, …) carried as the second
/// segment of the `simple_call` target's `FunctionPath`.  See the
/// module-level "PRE-EXISTING-ADAPTATION" block for why the class
/// is not yet at `simple_call.args[0]` — the orthodox fix is the
/// `Vec<ValueId>` → `Vec<LinkArg>` migration tracked separately
/// (multi-session).
///
/// `message_args` is the pre-evaluated list of message ValueIds
/// (side effects already on the graph from the caller's walk).
/// Empty for bare `panic!()` / `assert!(cond)`; single element for
/// `panic!(msg)` / `assert!(cond, msg)`; multiple for
/// `panic!("fmt", a, b)`.  In every case they become the trailing
/// positional arguments of the `simple_call` op — `simple_call`
/// itself is variadic upstream (`flowspace/operation.py:663
/// SimpleCall(SingleDispatchMixin, CallOp)`, `CallOp.args =
/// [callable, *args]`), so the multi-arg shape is RPython-canonical.
pub fn lower_exc_from_raise(
    graph: &mut FunctionGraph,
    block: BlockId,
    exc_class_name: &str,
    message_args: Vec<ValueId>,
) {
    // `op.simple_call(const(exc_class), *args)` — upstream
    // `flowcontext.py:614/623`.  The class is carried as the second
    // path segment so the single op still models
    // `simple_call(const(exc_class), *args)`; downstream readers can
    // reconstruct `(op, const_class, args…)` from `(path[0], path[1],
    // op.args)`.  This is a documented PRE-EXISTING-ADAPTATION — see
    // the module-level docstring for why it stands until the
    // `Vec<ValueId>` → `Vec<LinkArg>` migration lands.
    let simple_call_target = CallTarget::function_path(["simple_call", exc_class_name]);
    let evalue = graph
        .push_op(
            block,
            OpKind::Call {
                target: simple_call_target,
                args: message_args,
                result_ty: ValueType::Ref,
            },
            true,
        )
        .expect("op.simple_call(exc_class, ...) must produce a Ref exception instance");
    // `op.type(evalue)` — upstream `flowcontext.py:600` tail line
    // (`w_type = op.type(w_value).eval(self)`).
    let type_target = CallTarget::function_path(["type"]);
    let etype = graph
        .push_op(
            block,
            OpKind::Call {
                target: type_target,
                args: vec![evalue],
                result_ty: ValueType::Ref,
            },
            true,
        )
        .expect("op.type(evalue) must produce a Ref type value");
    // `flowspace/flowcontext.py:1253 Raise.nomoreblocks` — close
    // the block with the `(etype, evalue)` Link to the graph's
    // `exceptblock`.
    graph.set_raise_values(block, etype, evalue);
}
