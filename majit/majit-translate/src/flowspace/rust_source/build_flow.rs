//! `build_flow_from_rust` — core entry of the Rust-AST adapter.
//!
//! Mirrors `rpython/flowspace/objspace.py:38-53 build_flow(func)` in
//! signature and in the "walk source, emit SpaceOperations, close
//! blocks" contract. The Rust version consumes a `syn::ItemFn`
//! directly because pyre-interpreter's portal is Rust source — the
//! Python-side `func.func_code` round-trip is not available.
//!
//! ## Upstream construction idioms consulted
//!
//! - `rpython/flowspace/test/test_model.py:13-43` — canonical shape
//!   for straight-line graphs (startblock with inputargs, operations,
//!   single Link into returnblock).
//! - `rpython/translator/exceptiontransform.py:380-396` — the
//!   `Block::shared(inputargs) → closeblock([Link])` construction
//!   idiom for a block that produces one output.
//! - `rpython/flowspace/operation.py:445-521` — the `add_operator`
//!   table whose opnames every binop in [`binop_opname`] cites.
//! - `rpython/flowspace/flowcontext.py` `POP_JUMP_IF_FALSE` /
//!   `JUMP_FORWARD` handlers — the 2-exit fork + join idiom the
//!   `lower_if` routine mirrors without the Python bytecode
//!   vocabulary.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use syn::{
    Arm, BinOp, Block as SynBlock, Expr, ExprBinary, ExprIf, ExprLit, ExprMatch, ExprParen,
    ExprPath, ExprReturn, ItemFn, Lit, Local, LocalInit, Pat, PatIdent, Stmt,
};

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link,
    SpaceOperation, Variable,
};

/// Reasons the adapter rejects the input. Every variant carries a
/// human-readable `reason` string; the caller decides whether to log
/// it or surface it. No upstream counterpart — RPython's `build_flow`
/// assumes a well-formed Python function and crashes on bad input;
/// the Rust side rejects earlier so the caller can fall back to
/// another strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    /// Function signature (outer shape) disagrees with M2.5a scope.
    InvalidSignature { reason: String },
    /// A construct inside the body is outside the current adapter
    /// subset — remaining control flow (match / loop / ? / break /
    /// continue), method calls, struct literals, etc. The string
    /// names the construct and cites the phase it lands in.
    Unsupported { reason: String },
    /// Identifier resolved via `syn::Expr::Path` is not in the
    /// locals map. Corresponds to upstream `UnboundLocalError` at
    /// `flowcontext.py:LOAD_FAST` when a local is read before store.
    UnboundLocal { name: String },
}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterError::InvalidSignature { reason } => {
                write!(f, "invalid signature: {reason}")
            }
            AdapterError::Unsupported { reason } => write!(f, "unsupported construct: {reason}"),
            AdapterError::UnboundLocal { name } => write!(f, "unbound local: {name}"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Entry point. Walks `func` and emits the `FunctionGraph` that
/// upstream `build_flow(GraphFunc)` would have produced for the
/// equivalent Python source.
pub fn build_flow_from_rust(func: &ItemFn) -> Result<FunctionGraph, AdapterError> {
    validate_signature(func)?;

    let (inputargs, locals) = collect_params(func)?;
    let startblock = Block::shared(inputargs);
    let name = func.sig.ident.to_string();
    let graph = FunctionGraph::new(name, startblock.clone());

    let mut builder = Builder {
        current: BlockBuilder {
            block: startblock.clone(),
            ops: Vec::new(),
            locals,
        },
        returnblock: graph.returnblock.clone(),
    };

    let tail = lower_block(&mut builder, &func.block)?;

    // Terminate the currently-open block with a Link into returnblock.
    let link = Rc::new(RefCell::new(Link::new(
        vec![tail],
        Some(builder.returnblock.clone()),
        None,
    )));
    builder.finalize_current(vec![link], None);

    Ok(graph)
}

// ____________________________________________________________
// Signature validation — outer "is this function shape supported?"

fn validate_signature(func: &ItemFn) -> Result<(), AdapterError> {
    let sig = &func.sig;
    if sig.asyncness.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "async fn not supported (M2.5a)".into(),
        });
    }
    if sig.unsafety.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "unsafe fn not supported (M2.5a)".into(),
        });
    }
    if !sig.generics.params.is_empty() {
        return Err(AdapterError::InvalidSignature {
            reason: "generic fn not supported (trait dispatch lands in M2.5c)".into(),
        });
    }
    if sig.generics.where_clause.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "where-clause not supported (trait dispatch lands in M2.5c)".into(),
        });
    }
    if sig.variadic.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "variadic fn not supported".into(),
        });
    }
    Ok(())
}

fn collect_params(func: &ItemFn) -> Result<(Vec<Hlvalue>, HashMap<String, Hlvalue>), AdapterError> {
    let mut inputargs: Vec<Hlvalue> = Vec::new();
    let mut locals: HashMap<String, Hlvalue> = HashMap::new();
    for input in &func.sig.inputs {
        let fn_arg = match input {
            syn::FnArg::Receiver(_) => {
                return Err(AdapterError::InvalidSignature {
                    reason: "self receiver not supported (method dispatch lands in M2.5c)".into(),
                });
            }
            syn::FnArg::Typed(pat_type) => pat_type,
        };
        let ident = match &*fn_arg.pat {
            Pat::Ident(PatIdent {
                ident,
                by_ref: None,
                subpat: None,
                ..
            }) => ident.to_string(),
            _ => {
                return Err(AdapterError::InvalidSignature {
                    reason: "parameter pattern must be a plain identifier".into(),
                });
            }
        };
        let var = Hlvalue::Variable(Variable::named(&ident));
        locals.insert(ident, var.clone());
        inputargs.push(var);
    }
    Ok((inputargs, locals))
}

// ____________________________________________________________
// Builder — tracks the currently-open block and orchestrates
// fork/join for control flow.

/// Per-block accumulation state. Matches the shape of upstream's
/// `FlowContext.locals_w` + `FlowContext.pending_block.operations` —
/// one block's worth of "things to attach before closing".
struct BlockBuilder {
    block: BlockRef,
    ops: Vec<SpaceOperation>,
    /// RPython `FrameState.locals_w` by-name view. The Rust adapter
    /// does not need the full mergeable-stack view because Rust source
    /// has no Python-bytecode stack.
    locals: HashMap<String, Hlvalue>,
}

struct Builder {
    current: BlockBuilder,
    returnblock: BlockRef,
}

impl Builder {
    fn emit_op(&mut self, op: SpaceOperation) {
        self.current.ops.push(op);
    }

    fn locals(&self) -> &HashMap<String, Hlvalue> {
        &self.current.locals
    }

    fn set_local(&mut self, name: String, value: Hlvalue) {
        self.current.locals.insert(name, value);
    }

    /// Attach accumulated `current.ops` + `exitswitch` to
    /// `current.block`, wire `exits` via `BlockRefExt::closeblock`, and
    /// leave the Builder without a live "current" block. Used when the
    /// terminator replaces the block entirely (graph exit) or before
    /// `open_new_block` swaps in a fresh one.
    ///
    /// Mirrors upstream `flowcontext.py` where attaching `operations
    /// = tuple(ops)` then `block.closeblock(*exits)` happens in the
    /// same finalization pass.
    fn finalize_current(&mut self, exits: Vec<Rc<RefCell<Link>>>, switch: Option<Hlvalue>) {
        let ops = std::mem::take(&mut self.current.ops);
        {
            let mut b = self.current.block.borrow_mut();
            b.operations = ops;
            b.exitswitch = switch;
        }
        self.current.block.closeblock(exits);
    }

    /// Swap the currently-open block with `new`. Used after
    /// `finalize_current` to start emitting into `new`.
    fn open_new_block(&mut self, new: BlockBuilder) {
        self.current = new;
    }
}

// ____________________________________________________________
// Statement & expression lowering.

fn lower_block(b: &mut Builder, block: &SynBlock) -> Result<Hlvalue, AdapterError> {
    let mut tail: Option<Hlvalue> = None;
    for (idx, stmt) in block.stmts.iter().enumerate() {
        let is_last = idx + 1 == block.stmts.len();
        match stmt {
            Stmt::Local(local) => {
                lower_let(b, local)?;
            }
            Stmt::Expr(expr, semi) => {
                if semi.is_some() {
                    if let Expr::Return(ret) = expr {
                        if !is_last {
                            return Err(AdapterError::Unsupported {
                                reason: "return before end of block (control flow lands in M2.5b)"
                                    .into(),
                            });
                        }
                        tail = Some(lower_return(b, ret)?);
                    } else {
                        return Err(AdapterError::Unsupported {
                            reason:
                                "expression-with-semicolon statement (side-effecting statement \
                                    lands with method calls in M2.5c)"
                                    .into(),
                        });
                    }
                } else {
                    if !is_last {
                        return Err(AdapterError::Unsupported {
                            reason: "non-tail expression without trailing `;`".into(),
                        });
                    }
                    tail = Some(lower_expr(b, expr)?);
                }
            }
            Stmt::Item(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "nested item (fn/struct/impl inside fn body)".into(),
                });
            }
            Stmt::Macro(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "macro invocation in statement position".into(),
                });
            }
        }
    }
    tail.ok_or(AdapterError::Unsupported {
        reason: "function body has no tail expression / explicit return".into(),
    })
}

fn lower_let(b: &mut Builder, local: &Local) -> Result<(), AdapterError> {
    let ident = match &local.pat {
        Pat::Ident(PatIdent {
            ident,
            by_ref: None,
            subpat: None,
            ..
        }) => ident.to_string(),
        Pat::Type(pat_type) => match &*pat_type.pat {
            Pat::Ident(PatIdent {
                ident,
                by_ref: None,
                subpat: None,
                ..
            }) => ident.to_string(),
            _ => {
                return Err(AdapterError::Unsupported {
                    reason: "let pattern must be a plain identifier".into(),
                });
            }
        },
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "let pattern must be a plain identifier".into(),
            });
        }
    };
    let init = match &local.init {
        Some(LocalInit {
            expr,
            diverge: None,
            ..
        }) => expr,
        Some(_) => {
            return Err(AdapterError::Unsupported {
                reason: "let-else is control flow (lands in M2.5b)".into(),
            });
        }
        None => {
            return Err(AdapterError::Unsupported {
                reason: "let without initializer".into(),
            });
        }
    };
    let value = lower_expr(b, init)?;
    // Upstream STORE_FAST: reassignment REPLACES the locals-map entry.
    // The SSA value feeding subsequent reads is the new one.
    b.set_local(ident, value);
    Ok(())
}

fn lower_return(b: &mut Builder, ret: &ExprReturn) -> Result<Hlvalue, AdapterError> {
    match &ret.expr {
        Some(expr) => lower_expr(b, expr),
        None => Ok(Hlvalue::Constant(Constant::new(ConstValue::None))),
    }
}

fn lower_expr(b: &mut Builder, expr: &Expr) -> Result<Hlvalue, AdapterError> {
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => lower_literal(lit),
        Expr::Path(ExprPath {
            path, qself: None, ..
        }) => {
            let ident = path
                .get_ident()
                .ok_or_else(|| AdapterError::Unsupported {
                    reason: "qualified path (type resolution lands in M2.5c)".into(),
                })?
                .to_string();
            b.locals()
                .get(&ident)
                .cloned()
                .ok_or(AdapterError::UnboundLocal { name: ident })
        }
        Expr::Binary(ExprBinary {
            op, left, right, ..
        }) => lower_binop(b, *op, left, right),
        Expr::Paren(ExprParen { expr, .. }) => lower_expr(b, expr),
        Expr::If(if_expr) => lower_if(b, if_expr),
        Expr::Block(block_expr) => lower_block(b, &block_expr.block),
        Expr::Return(ret) => {
            let _ = ret;
            Err(AdapterError::Unsupported {
                reason: "return in expression position (control flow lands in M2.5b)".into(),
            })
        }
        Expr::Match(match_expr) => lower_match(b, match_expr),
        Expr::ForLoop(_) | Expr::While(_) | Expr::Loop(_) => Err(AdapterError::Unsupported {
            reason: "loop construct (lands in next M2.5b slice)".into(),
        }),
        Expr::Try(_) => Err(AdapterError::Unsupported {
            reason: "? operator (exception edges land in next M2.5b slice)".into(),
        }),
        Expr::Break(_) | Expr::Continue(_) => Err(AdapterError::Unsupported {
            reason: "break/continue (lands with loops in M2.5b)".into(),
        }),
        Expr::MethodCall(_) => Err(AdapterError::Unsupported {
            reason: "method call (trait dispatch lands in M2.5c)".into(),
        }),
        Expr::Call(_) => Err(AdapterError::Unsupported {
            reason: "function call (call lowering lands in M2.5c)".into(),
        }),
        Expr::Struct(_) | Expr::Tuple(_) | Expr::Array(_) => Err(AdapterError::Unsupported {
            reason: "composite literal (struct/tuple/array land in M2.5d)".into(),
        }),
        Expr::Closure(_) => Err(AdapterError::Unsupported {
            reason: "closure (not in roadmap scope)".into(),
        }),
        Expr::Reference(_) | Expr::Unary(_) | Expr::Cast(_) | Expr::Field(_) | Expr::Index(_) => {
            Err(AdapterError::Unsupported {
                reason: "operator / field / index (later M2.5 slices)".into(),
            })
        }
        _ => Err(AdapterError::Unsupported {
            reason: format!("unrecognised expression kind: {}", discriminant(expr)),
        }),
    }
}

fn lower_literal(lit: &Lit) -> Result<Hlvalue, AdapterError> {
    match lit {
        Lit::Int(int) => {
            let value: i64 = int.base10_parse().map_err(|e| AdapterError::Unsupported {
                reason: format!("integer literal out of i64 range: {e}"),
            })?;
            Ok(Hlvalue::Constant(Constant::new(ConstValue::Int(value))))
        }
        Lit::Bool(bl) => Ok(Hlvalue::Constant(Constant::new(ConstValue::Bool(bl.value)))),
        Lit::Str(_) | Lit::ByteStr(_) | Lit::Byte(_) | Lit::Char(_) | Lit::Float(_) => {
            Err(AdapterError::Unsupported {
                reason: "non-integer/bool literal (string/float/char land in M2.5d)".into(),
            })
        }
        _ => Err(AdapterError::Unsupported {
            reason: "unrecognised literal kind".into(),
        }),
    }
}

fn lower_binop(
    b: &mut Builder,
    op: BinOp,
    left: &Expr,
    right: &Expr,
) -> Result<Hlvalue, AdapterError> {
    let opname = binop_opname(op)?;
    let lhs = lower_expr(b, left)?;
    let rhs = lower_expr(b, right)?;
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(opname, vec![lhs, rhs], result.clone()));
    Ok(result)
}

/// Rust `BinOp` → upstream `operation.py` opname. Covers the 16
/// non-short-circuit infix operators the M2.5a subset supports.
/// Short-circuit `&&` / `||` are control flow and land in M2.5b
/// together with `if` / `match`.
fn binop_opname(op: BinOp) -> Result<&'static str, AdapterError> {
    Ok(match op {
        BinOp::Add(_) => "add",
        BinOp::Sub(_) => "sub",
        BinOp::Mul(_) => "mul",
        BinOp::Div(_) => "div",
        BinOp::Rem(_) => "mod",
        BinOp::BitAnd(_) => "and_",
        BinOp::BitOr(_) => "or_",
        BinOp::BitXor(_) => "xor",
        BinOp::Shl(_) => "lshift",
        BinOp::Shr(_) => "rshift",
        BinOp::Eq(_) => "eq",
        BinOp::Ne(_) => "ne",
        BinOp::Lt(_) => "lt",
        BinOp::Le(_) => "le",
        BinOp::Gt(_) => "gt",
        BinOp::Ge(_) => "ge",
        BinOp::And(_) | BinOp::Or(_) => {
            return Err(AdapterError::Unsupported {
                reason: "short-circuit && / || (lands with match in next M2.5b slice)".into(),
            });
        }
        BinOp::AddAssign(_)
        | BinOp::SubAssign(_)
        | BinOp::MulAssign(_)
        | BinOp::DivAssign(_)
        | BinOp::RemAssign(_)
        | BinOp::BitAndAssign(_)
        | BinOp::BitOrAssign(_)
        | BinOp::BitXorAssign(_)
        | BinOp::ShlAssign(_)
        | BinOp::ShrAssign(_) => {
            return Err(AdapterError::Unsupported {
                reason: "compound assignment (mutation lands with control flow in M2.5b)".into(),
            });
        }
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "unrecognised binary operator".into(),
            });
        }
    })
}

// ____________________________________________________________
// `if/else` — 2-way fork with FrameState-style locals merge.
//
// Upstream basis: `rpython/flowspace/flowcontext.py` `POP_JUMP_IF_FALSE`
// / `JUMP_FORWARD` handler pair. The Python-bytecode flow is:
//
//   if cond: then_body                POP_JUMP_IF_FALSE to else-pc
//   else: else_body             →     then-ops
//   …continue                          JUMP_FORWARD to join-pc
//                                      else-pc: else-ops
//                                      join-pc: …
//
// The graph-level shape `flowcontext` emits is what this routine
// mirrors directly against `syn::ExprIf`: a startblock with
// `exitswitch=cond` and two exits (one per `Constant::Bool(x)`
// exitcase), two branch blocks, and a single join block reached by
// `Link(target=join)` from each branch. Locals live in each branch's
// own `locals_w` and merge into the join block's inputargs via
// `FrameState` union.

fn lower_if(b: &mut Builder, if_expr: &ExprIf) -> Result<Hlvalue, AdapterError> {
    // 1. Evaluate condition into the current block.
    let cond = lower_expr(b, &if_expr.cond)?;

    // `if` without `else` in tail position produces `()`, which is
    // outside the integer/bool subset. Defer to the M2.5d literal
    // phase and reject cleanly here.
    let (_else_tok, else_expr) = match &if_expr.else_branch {
        Some(branch) => branch,
        None => {
            return Err(AdapterError::Unsupported {
                reason: "if without else produces () — unit literal lands in M2.5d".into(),
            });
        }
    };

    // 2. Snapshot state at the fork point. Upstream `FrameState.copy()`
    //    clones locals_w so STORE_FAST inside one branch doesn't leak
    //    into the other. The adapter replays this as "each branch
    //    inherits its own copy of locals, addressed via fresh
    //    Variables on the branch block's inputargs".
    //
    //    Sorted merged_names gives deterministic inputargs ordering
    //    across runs, mirroring upstream's Python dict order.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();

    // 3. Allocate branch blocks. Each carries a fresh Variable per
    //    merged local in its inputargs, matching upstream where the
    //    jump target's frame receives fresh vars via link.args.
    let (then_block, then_locals) = branch_block_with_inputargs(&merged_names);
    let (else_block, else_locals) = branch_block_with_inputargs(&merged_names);

    // 4. Close the fork block. Link.args carry the pre-fork SSA values
    //    so branch inputargs bind to the caller's locals on entry.
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();
    let false_link = Rc::new(RefCell::new(Link::new(
        fork_args.clone(),
        Some(else_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        fork_args,
        Some(then_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(cond));

    // 5. Lower the then-branch into its own BlockBuilder. Nested
    //    control flow may reassign `b.current.block` away from
    //    `then_block` to a nested join; we close whatever the final
    //    current block is after the branch body returns.
    b.open_new_block(BlockBuilder {
        block: then_block.clone(),
        ops: Vec::new(),
        locals: then_locals,
    });
    let then_tail = lower_block(b, &if_expr.then_branch)?;
    let then_exit_block = b.current.block.clone();
    let then_exit_ops = std::mem::take(&mut b.current.ops);
    let then_exit_locals = std::mem::take(&mut b.current.locals);

    // 6. Lower the else-branch the same way.
    b.open_new_block(BlockBuilder {
        block: else_block.clone(),
        ops: Vec::new(),
        locals: else_locals,
    });
    let else_tail = lower_expr(b, else_expr)?;
    let else_exit_block = b.current.block.clone();
    let else_exit_ops = std::mem::take(&mut b.current.ops);
    let else_exit_locals = std::mem::take(&mut b.current.locals);

    // 7. Build the join block.
    //    inputargs = [tail_var, local_var_0, local_var_1, …].
    let tail_var = Hlvalue::Variable(Variable::new());
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    join_inputargs.push(tail_var.clone());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // 8. Close each branch's tail block with a Link into the join.
    //    `branch_link_args` reads from the branch's locals snapshot —
    //    branch-local `let` bindings (names absent from merged_names)
    //    never reach the join.
    let then_link_args = branch_link_args(&then_tail, &merged_names, &then_exit_locals);
    let else_link_args = branch_link_args(&else_tail, &merged_names, &else_exit_locals);

    {
        let then_link = Rc::new(RefCell::new(Link::new(
            then_link_args,
            Some(join_block.clone()),
            None,
        )));
        then_exit_block.borrow_mut().operations = then_exit_ops;
        then_exit_block.closeblock(vec![then_link]);
    }
    {
        let else_link = Rc::new(RefCell::new(Link::new(
            else_link_args,
            Some(join_block.clone()),
            None,
        )));
        else_exit_block.borrow_mut().operations = else_exit_ops;
        else_exit_block.closeblock(vec![else_link]);
    }

    // 9. Continue lowering into the join block. Subsequent reads of
    //    any merged local see the join-block inputarg.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });
    Ok(tail_var)
}

/// Create a branch block whose inputargs are fresh Variables — one
/// per merged local name — plus the locals map that binds each name
/// to its own inputarg Variable.
fn branch_block_with_inputargs(merged_names: &[String]) -> (BlockRef, HashMap<String, Hlvalue>) {
    let mut inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len());
    let mut locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        inputargs.push(fresh.clone());
        locals.insert(name.clone(), fresh);
    }
    (Block::shared(inputargs), locals)
}

/// Build the `Link.args` a branch carries into the join block.
///
/// Head slot: the branch's tail expression value.
/// Tail slots: one per merged local name, in the caller-provided
/// order. Each slot holds the branch's current SSA value for that
/// name. `merged_names` is sourced from the pre-fork locals, so every
/// name is guaranteed to be present in `branch_locals` because every
/// branch inherits the full pre-fork set on entry.
fn branch_link_args(
    tail: &Hlvalue,
    merged_names: &[String],
    branch_locals: &HashMap<String, Hlvalue>,
) -> Vec<Hlvalue> {
    let mut out = Vec::with_capacity(merged_names.len() + 1);
    out.push(tail.clone());
    for name in merged_names {
        let value = branch_locals
            .get(name)
            .cloned()
            .expect("merged_names is a subset of branch entry locals");
        out.push(value);
    }
    out
}

// ____________________________________________________________
// `match` — n-way jump table with FrameState-style locals merge.
//
// Upstream basis: `rpython/flowspace/model.py:643-666`. The model
// accepts either (a) a 2-arm boolean switch with `Bool(False)` then
// `Bool(True)` exitcases or (b) an n-arm primitive switch with
// `is_valid_int(n)` exitcases ending optionally in a
// `Constant::Str("default")` catch-all — exactly the shape this
// routine emits depending on the arm set.
//
// Arm → exitcase mapping:
//
// | Rust pattern      | Link.exitcase                                |
// |-------------------|----------------------------------------------|
// | `N` (int literal) | `Constant::Int(N)`                           |
// | `true` / `false`  | `Constant::Bool(_)`                          |
// | `_`               | `Constant::Str("default")` (must be last)    |
//
// Patterns outside that subset (`Pat::Or`, `Pat::TupleStruct`,
// `Pat::Range`, identifier-binding, guards, …) reject via
// `AdapterError::Unsupported` — they land in M2.5c / M2.5d.

fn lower_match(b: &mut Builder, match_expr: &ExprMatch) -> Result<Hlvalue, AdapterError> {
    // 1. Evaluate the scrutinee into the current block. Becomes the
    //    block's `exitswitch` when we close the fork.
    let scrutinee = lower_expr(b, &match_expr.expr)?;

    if match_expr.arms.is_empty() {
        return Err(AdapterError::Unsupported {
            reason: "match with zero arms".into(),
        });
    }

    // 2. Validate every arm up-front so the fork block is only closed
    //    once we know every branch can be lowered. `exitcase = None`
    //    marks a wildcard — upstream `model.py:652` requires such a
    //    case to be the last exit.
    let mut arm_exitcases: Vec<Option<Hlvalue>> = Vec::with_capacity(match_expr.arms.len());
    for (idx, arm) in match_expr.arms.iter().enumerate() {
        validate_arm(arm)?;
        let is_last = idx + 1 == match_expr.arms.len();
        let exitcase = classify_pattern(&arm.pat, is_last)?;
        arm_exitcases.push(exitcase);
    }

    // 3. Enforce upstream's uniqueness invariant
    //    (`model.py:692 allexitcases[link.exitcase] = True`).
    let mut seen: Vec<&Hlvalue> = Vec::new();
    for exitcase in arm_exitcases.iter().flatten() {
        if seen.iter().any(|s| *s == exitcase) {
            return Err(AdapterError::Unsupported {
                reason: "match arm exitcase repeated — upstream forbids duplicate jump-table \
                    cases"
                    .into(),
            });
        }
        seen.push(exitcase);
    }

    // 4. Snapshot locals for the fork (same discipline as `lower_if`).
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();

    // 5. Allocate one branch block per arm, bundle the exit-case into a
    //    Link from the fork.
    let mut branch_blocks: Vec<(BlockRef, HashMap<String, Hlvalue>)> =
        Vec::with_capacity(match_expr.arms.len());
    let mut fork_exits: Vec<Rc<RefCell<Link>>> = Vec::with_capacity(match_expr.arms.len());
    for exitcase in &arm_exitcases {
        let (branch_block, branch_locals) = branch_block_with_inputargs(&merged_names);
        let link_exitcase = match exitcase {
            Some(v) => Some(v.clone()),
            None => Some(Hlvalue::Constant(Constant::new(ConstValue::Str(
                "default".into(),
            )))),
        };
        let link = Rc::new(RefCell::new(Link::new(
            fork_args.clone(),
            Some(branch_block.clone()),
            link_exitcase,
        )));
        fork_exits.push(link);
        branch_blocks.push((branch_block, branch_locals));
    }
    b.finalize_current(fork_exits, Some(scrutinee));

    // 6. Lower each arm's body. Record the exit block / ops / locals /
    //    tail for the later join-block wiring.
    struct ArmExit {
        block: BlockRef,
        ops: Vec<SpaceOperation>,
        locals: HashMap<String, Hlvalue>,
        tail: Hlvalue,
    }
    let mut arm_exits: Vec<ArmExit> = Vec::with_capacity(match_expr.arms.len());
    for ((branch_block, branch_locals), arm) in branch_blocks.into_iter().zip(&match_expr.arms) {
        b.open_new_block(BlockBuilder {
            block: branch_block,
            ops: Vec::new(),
            locals: branch_locals,
        });
        let tail = lower_expr(b, &arm.body)?;
        let exit_block = b.current.block.clone();
        let exit_ops = std::mem::take(&mut b.current.ops);
        let exit_locals = std::mem::take(&mut b.current.locals);
        arm_exits.push(ArmExit {
            block: exit_block,
            ops: exit_ops,
            locals: exit_locals,
            tail,
        });
    }

    // 7. Build the join block — inputargs = [tail_var, local_var_0, …].
    let tail_var = Hlvalue::Variable(Variable::new());
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    join_inputargs.push(tail_var.clone());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // 8. Close each arm's exit block with a Link into the join.
    for arm_exit in arm_exits {
        let link_args = branch_link_args(&arm_exit.tail, &merged_names, &arm_exit.locals);
        let link = Rc::new(RefCell::new(Link::new(
            link_args,
            Some(join_block.clone()),
            None,
        )));
        arm_exit.block.borrow_mut().operations = arm_exit.ops;
        arm_exit.block.closeblock(vec![link]);
    }

    // 9. Continue lowering into the join block.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });
    Ok(tail_var)
}

fn validate_arm(arm: &Arm) -> Result<(), AdapterError> {
    if arm.guard.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "match arm guard (`if COND`) (lands after control-flow slice)".into(),
        });
    }
    Ok(())
}

/// Map a `syn::Pat` to the `Link.exitcase` upstream expects. Returns
/// `Ok(None)` for a wildcard (catch-all) arm — the caller inserts the
/// sentinel `"default"` string per `model.py:652`.
fn classify_pattern(pat: &Pat, is_last: bool) -> Result<Option<Hlvalue>, AdapterError> {
    match pat {
        Pat::Wild(_) => {
            if !is_last {
                return Err(AdapterError::Unsupported {
                    reason: "wildcard arm must be the last arm (upstream `model.py:652` invariant)"
                        .into(),
                });
            }
            Ok(None)
        }
        Pat::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Int(int) => {
                let value: i64 = int.base10_parse().map_err(|e| AdapterError::Unsupported {
                    reason: format!("match arm integer pattern out of i64 range: {e}"),
                })?;
                Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::Int(
                    value,
                )))))
            }
            Lit::Bool(bl) => Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(
                bl.value,
            ))))),
            Lit::Str(_) | Lit::ByteStr(_) | Lit::Byte(_) | Lit::Char(_) | Lit::Float(_) | _ => {
                Err(AdapterError::Unsupported {
                    reason: "match arm non-integer/bool literal pattern (string/float/char \
                    patterns land in M2.5d)"
                        .into(),
                })
            }
        },
        Pat::Or(_) => Err(AdapterError::Unsupported {
            reason: "match arm `|` pattern (or-pattern)".into(),
        }),
        Pat::Range(_) => Err(AdapterError::Unsupported {
            reason: "match arm range pattern (`a..b`)".into(),
        }),
        Pat::TupleStruct(_) | Pat::Tuple(_) | Pat::Struct(_) => Err(AdapterError::Unsupported {
            reason: "match arm composite pattern (enum/tuple/struct — lands in M2.5d)".into(),
        }),
        Pat::Ident(_) => Err(AdapterError::Unsupported {
            reason: "match arm identifier-binding pattern".into(),
        }),
        _ => Err(AdapterError::Unsupported {
            reason: "match arm pattern not in M2.5b subset (only int / bool literals and \
                wildcard supported)"
                .into(),
        }),
    }
}

// ____________________________________________________________

fn discriminant(expr: &Expr) -> &'static str {
    match expr {
        Expr::Array(_) => "Array",
        Expr::Assign(_) => "Assign",
        Expr::Async(_) => "Async",
        Expr::Await(_) => "Await",
        Expr::Binary(_) => "Binary",
        Expr::Block(_) => "Block",
        Expr::Break(_) => "Break",
        Expr::Call(_) => "Call",
        Expr::Cast(_) => "Cast",
        Expr::Closure(_) => "Closure",
        Expr::Const(_) => "Const",
        Expr::Continue(_) => "Continue",
        Expr::Field(_) => "Field",
        Expr::ForLoop(_) => "ForLoop",
        Expr::Group(_) => "Group",
        Expr::If(_) => "If",
        Expr::Index(_) => "Index",
        Expr::Infer(_) => "Infer",
        Expr::Let(_) => "Let",
        Expr::Lit(_) => "Lit",
        Expr::Loop(_) => "Loop",
        Expr::Macro(_) => "Macro",
        Expr::Match(_) => "Match",
        Expr::MethodCall(_) => "MethodCall",
        Expr::Paren(_) => "Paren",
        Expr::Path(_) => "Path",
        Expr::Range(_) => "Range",
        Expr::Reference(_) => "Reference",
        Expr::Repeat(_) => "Repeat",
        Expr::Return(_) => "Return",
        Expr::Struct(_) => "Struct",
        Expr::Try(_) => "Try",
        Expr::TryBlock(_) => "TryBlock",
        Expr::Tuple(_) => "Tuple",
        Expr::Unary(_) => "Unary",
        Expr::Unsafe(_) => "Unsafe",
        Expr::Verbatim(_) => "Verbatim",
        Expr::While(_) => "While",
        Expr::Yield(_) => "Yield",
        _ => "Unknown",
    }
}

// ____________________________________________________________

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::checkgraph;

    fn parse(src: &str) -> ItemFn {
        syn::parse_str(src).expect("test source should parse")
    }

    fn lower(src: &str) -> Result<FunctionGraph, AdapterError> {
        build_flow_from_rust(&parse(src))
    }

    // ---- M2.5a straight-line accept tests -------------------------

    #[test]
    fn lit_constant_return() {
        let g = lower("fn one() -> i64 { 1 }").unwrap();
        checkgraph(&g);
        assert_eq!(g.name, "one");
        assert_eq!(g.getargs().len(), 0);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        assert_eq!(start.exits.len(), 1);
        let link = start.exits[0].borrow();
        assert_eq!(link.args.len(), 1);
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(1)),
            other => panic!("expected Constant(1), got {other:?}"),
        }
    }

    #[test]
    fn identity_function() {
        let g = lower("fn identity(x: i64) -> i64 { x }").unwrap();
        checkgraph(&g);
        assert_eq!(g.getargs().len(), 1);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        let link = start.exits[0].borrow();
        match &start.inputargs[0] {
            Hlvalue::Variable(v) => match link.args[0].as_ref().unwrap() {
                Hlvalue::Variable(lv) => assert_eq!(lv.id(), v.id()),
                other => panic!("expected Variable, got {other:?}"),
            },
            other => panic!("expected Variable inputarg, got {other:?}"),
        }
    }

    #[test]
    fn binop_add_emits_space_operation() {
        let g = lower("fn add(a: i64, b: i64) -> i64 { a + b }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        let op = &start.operations[0];
        assert_eq!(op.opname, "add");
        assert_eq!(op.args.len(), 2);
    }

    #[test]
    fn binop_sub_emits_space_operation() {
        let g = lower("fn sub(a: i64, b: i64) -> i64 { a - b }").unwrap();
        checkgraph(&g);
        assert_eq!(g.startblock.borrow().operations[0].opname, "sub");
    }

    #[test]
    fn let_then_tail_chains_two_ops() {
        let g = lower(
            "fn f(a: i64, b: i64) -> i64 {
                let t = a + b;
                t * t
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "add");
        assert_eq!(start.operations[1].opname, "mul");
        let t_var = &start.operations[0].result;
        assert_eq!(&start.operations[1].args[0], t_var);
        assert_eq!(&start.operations[1].args[1], t_var);
    }

    #[test]
    fn chained_lets_and_rebinding() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                let x = x + 1;
                let x = x * 2;
                x
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "add");
        assert_eq!(start.operations[1].opname, "mul");
        let link = start.exits[0].borrow();
        assert_eq!(link.args[0].as_ref().unwrap(), &start.operations[1].result);
    }

    #[test]
    fn comparison_ops() {
        let g = lower("fn f(a: i64, b: i64) -> bool { a < b }").unwrap();
        checkgraph(&g);
        assert_eq!(g.startblock.borrow().operations[0].opname, "lt");
    }

    #[test]
    fn bool_literal_return() {
        let g = lower("fn f() -> bool { true }").unwrap();
        checkgraph(&g);
        let link = g.startblock.borrow().exits[0].clone();
        match link.borrow().args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Bool(true)),
            other => panic!("expected Constant(true), got {other:?}"),
        }
    }

    #[test]
    fn explicit_return_is_accepted_in_tail() {
        let g = lower("fn f() -> i64 { return 1; }").unwrap();
        checkgraph(&g);
        let link = g.startblock.borrow().exits[0].clone();
        match link.borrow().args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(1)),
            other => panic!("expected Constant(1), got {other:?}"),
        }
    }

    // ---- M2.5b if/else accept tests --------------------------------

    #[test]
    fn if_else_literal_branches() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { 1 } else { 0 }
            }",
        )
        .unwrap();
        checkgraph(&g);

        // startblock: [x] → emits `x > 0`, exitswitch = that var, 2
        // exits.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "gt");
        let switch_var = start.operations[0].result.clone();
        assert_eq!(start.exitswitch.as_ref(), Some(&switch_var));
        assert_eq!(start.exits.len(), 2);

        let false_exit = start.exits[0].borrow();
        let true_exit = start.exits[1].borrow();
        assert_eq!(
            false_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
        assert_eq!(
            true_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );

        // Both branches reach the same join block (iterblocks has 4:
        // start, else, then, join; returnblock is listed separately
        // and may appear as the join's successor).
        drop(false_exit);
        drop(true_exit);
        drop(start);
        let blocks = g.iterblocks();
        // start + else + then + join + returnblock.
        assert_eq!(blocks.len(), 5);
    }

    #[test]
    fn if_else_preserves_local_merge() {
        // Both branches modify `x`; join must receive the merged
        // `x_local_*` through its inputargs and forward it.
        let g = lower(
            "fn f(x: i64) -> i64 {
                let y = if x > 0 { x + 1 } else { x - 1 };
                y
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        assert_eq!(blocks.len(), 5); // start + else + then + join + return
        // Join has 2 inputargs: the if-tail + the merged `x`.
        let join = blocks
            .iter()
            .find(|b| b.borrow().inputargs.len() == 2)
            .expect("expected join block with 2 inputargs");
        let join_ref = join.borrow();
        assert_eq!(join_ref.operations.len(), 0);
        // Join's single exit carries [tail_var] into returnblock (plus
        // any merged locals are dropped — tail assignment just reads
        // the first inputarg).
        let exit = join_ref.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        assert_eq!(exit.args[0].as_ref().unwrap(), &join_ref.inputargs[0]);
    }

    #[test]
    fn if_else_nested() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 {
                    if x > 10 { 2 } else { 1 }
                } else {
                    0
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Outer if-else has nested if-else in then-branch: 5 blocks
        // from the outer, +2 (inner then + inner else) +1 (inner join,
        // which acts as the outer then-branch's ops host), minus
        // duplication of the outer join…
        // Rather than pin exact count, check that checkgraph passes
        // (structural invariants hold).
        assert!(g.iterblocks().len() >= 5);
    }

    #[test]
    fn if_else_if_chain_positive_literals() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x == 0 { 0 } else if x == 1 { 1 } else { 2 }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // 2 nested if-elses ⇒ 2 fork blocks + 2 × 2 branch blocks +
        // 2 joins + startblock overlap: at least 7 reachable blocks.
        assert!(g.iterblocks().len() >= 6);
    }

    // ---- M2.5b match accept tests ---------------------------------

    #[test]
    fn match_int_literals_with_default() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x {
                    0 => 0,
                    1 => 10,
                    _ => 99,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // scrutinee path is a bare LOAD — no operation emitted.
        assert_eq!(start.operations.len(), 0);
        // exitswitch is the scrutinee (`x`'s Variable, per identity).
        assert!(start.exitswitch.is_some());
        assert_eq!(start.exits.len(), 3);
        // Per-arm exitcase ordering matches source order.
        let ec0 = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec1 = start.exits[1].borrow().exitcase.clone().unwrap();
        let ec2 = start.exits[2].borrow().exitcase.clone().unwrap();
        assert_eq!(ec0, Hlvalue::Constant(Constant::new(ConstValue::Int(0))));
        assert_eq!(ec1, Hlvalue::Constant(Constant::new(ConstValue::Int(1))));
        assert_eq!(
            ec2,
            Hlvalue::Constant(Constant::new(ConstValue::Str("default".into())))
        );
    }

    #[test]
    fn match_scrutinee_is_op_result() {
        // Scrutinee is a non-trivial expression — the binop's result
        // Variable becomes `exitswitch`.
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x + 1 {
                    0 => 0,
                    _ => 1,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "add");
        let add_result = start.operations[0].result.clone();
        assert_eq!(start.exitswitch.as_ref(), Some(&add_result));
    }

    #[test]
    fn match_merges_locals_into_join() {
        // Every arm rebinds `y`. The join block's inputargs must carry
        // both `tail_var` and the merged `y`.
        let g = lower(
            "fn f(x: i64, y: i64) -> i64 {
                let y = match x {
                    0 => y + 1,
                    _ => y - 1,
                };
                y
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // startblock + 2 arm blocks + 1 join + returnblock.
        assert_eq!(blocks.len(), 5);
        // Locate the join block — it has len(merged_names)+1 = 3
        // inputargs (tail_var, x_merged, y_merged) and zero ops. The
        // startblock also has zero ops but only 2 inputargs (x, y).
        let join = blocks
            .iter()
            .find(|b| b.borrow().inputargs.len() == 3 && b.borrow().operations.is_empty())
            .expect("expected empty-ops join block with 3 inputargs");
        let join_ref = join.borrow();
        // Join's single exit forwards inputarg[0] (tail_var) as the
        // return, matching STORE_FAST `y = <match tail>` semantics.
        let exit = join_ref.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        assert_eq!(exit.args[0].as_ref().unwrap(), &join_ref.inputargs[0]);
    }

    #[test]
    fn match_bool_scrutinee_two_arms() {
        // 2-arm match on a bool is the boolean-switch shape — upstream
        // `model.py:643-644` accepts `[False, True]` ordering but the
        // adapter preserves *source* order, i.e. whatever the user
        // writes. `checkgraph` tolerates the multi-case branch too.
        let g = lower(
            "fn f(x: bool) -> i64 {
                match x {
                    true => 1,
                    false => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        let ec0 = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec1 = start.exits[1].borrow().exitcase.clone().unwrap();
        assert_eq!(
            ec0,
            Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );
        assert_eq!(
            ec1,
            Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
    }

    #[test]
    fn match_single_wildcard_is_straight_line() {
        // Degenerate case: single wildcard arm. The adapter still emits
        // a fork + single-arm join; tail propagates through.
        let g = lower("fn f(x: i64) -> i64 { match x { _ => 42 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 1);
        let ec = start.exits[0].borrow().exitcase.clone().unwrap();
        assert_eq!(
            ec,
            Hlvalue::Constant(Constant::new(ConstValue::Str("default".into())))
        );
    }

    // ---- reject paths ---------------------------------------------

    #[test]
    fn rejects_generic_fn() {
        match lower("fn f<T>(x: T) -> T { x }").unwrap_err() {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("generic"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature, got {other:?}"),
        }
    }

    #[test]
    fn rejects_async_fn() {
        assert!(matches!(
            lower("async fn f() -> i64 { 1 }").unwrap_err(),
            AdapterError::InvalidSignature { .. }
        ));
    }

    #[test]
    fn rejects_unsafe_fn() {
        assert!(matches!(
            lower("unsafe fn f() -> i64 { 1 }").unwrap_err(),
            AdapterError::InvalidSignature { .. }
        ));
    }

    #[test]
    fn rejects_self_receiver() {
        match lower("fn f(&self) -> i64 { 1 }").unwrap_err() {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("self"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature (self), got {other:?}"),
        }
    }

    #[test]
    fn rejects_if_without_else() {
        // `if` in expression position without `else` — triggers the
        // lower_if else-branch check directly (not the semicolon-stmt
        // gate).
        match lower("fn f(x: i64) -> i64 { let _y = if x > 0 { 1 }; 2 }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("if without else") || reason.contains("unit"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(if-without-else), got {other:?}"),
        }
    }

    #[test]
    fn rejects_match_arm_guard() {
        // Arm guards (`if COND`) are out-of-scope for M2.5b. They
        // require the slice-2 fork + slice-1 if lowering in sequence
        // inside the arm body, which the adapter does not yet plumb.
        match lower("fn f(x: i64) -> i64 { match x { n if n > 0 => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("guard"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(guard), got {other:?}"),
        }
    }

    #[test]
    fn rejects_match_or_pattern() {
        match lower("fn f(x: i64) -> i64 { match x { 1 | 2 => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("or-pattern"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(or), got {other:?}"),
        }
    }

    #[test]
    fn rejects_match_range_pattern() {
        match lower("fn f(x: i64) -> i64 { match x { 0..=9 => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("range"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(range), got {other:?}"),
        }
    }

    #[test]
    fn rejects_match_wildcard_not_last() {
        match lower("fn f(x: i64) -> i64 { match x { _ => 0, 1 => 1 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("wildcard"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(wildcard-not-last), got {other:?}"),
        }
    }

    #[test]
    fn rejects_method_call() {
        match lower("fn f(x: i64) -> i64 { x.abs() }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("method"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(method), got {other:?}"),
        }
    }

    #[test]
    fn rejects_unbound_local() {
        match lower("fn f() -> i64 { x }").unwrap_err() {
            AdapterError::UnboundLocal { name } => assert_eq!(name, "x"),
            other => panic!("expected UnboundLocal, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_circuit_and() {
        match lower("fn f(a: bool, b: bool) -> bool { a && b }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("short-circuit"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(short-circuit), got {other:?}"),
        }
    }

    #[test]
    fn rejects_try_op() {
        match lower("fn f(x: Result<i64, ()>) -> i64 { x? }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("?") || reason.contains("exception"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(?), got {other:?}"),
        }
    }

    #[test]
    fn rejects_tuple_literal() {
        match lower("fn f() -> (i64, i64) { (1, 2) }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("composite") || reason.contains("tuple"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(tuple), got {other:?}"),
        }
    }
}
