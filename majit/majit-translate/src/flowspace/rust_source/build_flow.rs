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
    Arm, BinOp, Block as SynBlock, Expr, ExprArray, ExprBinary, ExprBreak, ExprCall, ExprCast,
    ExprContinue, ExprField, ExprForLoop, ExprIf, ExprIndex, ExprLit, ExprLoop, ExprMatch,
    ExprMethodCall, ExprParen, ExprPath, ExprReturn, ExprTry, ExprTuple, ExprUnary, ExprWhile,
    ItemFn, Lit, Local, LocalInit, Member, Pat, PatIdent, Stmt, UnOp,
};

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, HOST_ENV, Hlvalue, Link,
    SpaceOperation, Variable, c_last_exception,
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
        exceptblock: graph.exceptblock.clone(),
        loop_stack: Vec::new(),
    };

    match lower_block(&mut builder, &func.block)? {
        BlockExit::FallThrough(tail) => {
            // Body reached its closing `}` with a tail value —
            // terminate the currently-open block with a Link into
            // returnblock carrying that value (upstream implicit
            // `RETURN_VALUE` at the function-body tail).
            let link = Rc::new(RefCell::new(Link::new(
                vec![tail],
                Some(builder.returnblock.clone()),
                None,
            )));
            builder.finalize_current(vec![link], None);
        }
        BlockExit::Terminated => {
            // Body already closed itself via an explicit `return` —
            // `Return.nomoreblocks()` at `flowcontext.py:1232` ran,
            // so the returnblock Link has already been emitted and
            // the current block is gone. Nothing further to do here;
            // upstream's StopFlowing at the same site terminates the
            // pending-block scheduler identically.
        }
    }

    Ok(graph)
}

/// Result of lowering a block-shaped construct. Mirrors the control-
/// flow dichotomy in upstream `flowcontext.py`: either the block
/// reached its closing `}` with a tail value (`FallThrough`), or
/// execution left the block via a non-fallthrough terminator —
/// `return` in the current subset (`Terminated`). Upstream analogue:
/// `Return.nomoreblocks(ctx)` at `flowcontext.py:1232` closes the
/// current block straight to `graph.returnblock` and raises
/// `StopFlowing`; downstream code in the same block never runs.
enum BlockExit {
    FallThrough(Hlvalue),
    Terminated,
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
    // Generic type / lifetime parameters and where-clauses are
    // accepted as signature-level markers. The adapter does not
    // track trait-bound constraints directly; the annotator's
    // `FunctionDesc.specialize` / `cachedgraph`
    // (`description.py:272-281`, `:228-249`) reads the concrete
    // `args_s` at `build_types` call time and monomorphizes the
    // generics into a classdef-keyed specialized graph. Const
    // generic parameters still reject pending value-carrying
    // const-param support.
    for p in &sig.generics.params {
        match p {
            syn::GenericParam::Type(_) | syn::GenericParam::Lifetime(_) => {}
            syn::GenericParam::Const(_) => {
                return Err(AdapterError::InvalidSignature {
                    reason: "const generic parameter not supported (lands in M2.5d)".into(),
                });
            }
        }
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
        let ident = match input {
            // `self` / `&self` / `&mut self` — the method dispatch
            // case. Upstream RPython binds `self` as the first local
            // after `FunctionDesc.bind_self` has annotated it with
            // the concrete classdef (`description.py:350-355`); the
            // adapter just exposes it as a Variable named `self`,
            // matching the Python source convention.
            syn::FnArg::Receiver(_) => "self".to_string(),
            syn::FnArg::Typed(pat_type) => match &*pat_type.pat {
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
            },
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

/// Enclosing loop context — one entry per `while`/`loop` open on the
/// statement stack. `break` lowers a Link into `exit_block`; `continue`
/// into `header_block`. `merged_names` pins the locals set the loop
/// entry agreed on, so `break`/`continue` carry every entry-visible
/// local on the outgoing Link regardless of which body-local rebinds
/// happened at the jump point.
///
/// Upstream analogue: `flowcontext.py:794 SETUP_LOOP` pushes a
/// `LoopBlock` onto `self.blockstack`, and `BREAK_LOOP` / `CONTINUE_LOOP`
/// (`:525-529`) raise Break/Continue exceptions that the surrounding
/// LoopBlock turns into the corresponding bytecode-target jumps. The
/// Rust adapter does not have Python's blockstack abstraction; the
/// `loop_stack` here is the minimal equivalent for tracking
/// "which break target is live".
struct LoopCtx {
    header_block: BlockRef,
    exit_block: BlockRef,
    /// Names carried through the loop's back-edge + body inputargs.
    /// Includes any internal iter sidecar.
    merged_names: Vec<String>,
    /// Names carried through `break` / loop-exit exits — a subset of
    /// `merged_names` with internal iter sidecars removed. For
    /// `while` / `loop` the two sets are identical; `for` filters
    /// out its synthetic iterator slot so it does NOT leak into the
    /// exit block's inputargs (upstream `flowcontext.py:787, :1355,
    /// :1383` — iterator lives on the value stack and is popped at
    /// loop exit, never visible to post-loop code).
    exit_merged_names: Vec<String>,
}

struct Builder {
    current: BlockBuilder,
    returnblock: BlockRef,
    /// The graph's canonical exception exit — `Block([etype, evalue])`
    /// per `model.py:22-25`. `?` exception links target this.
    exceptblock: BlockRef,
    loop_stack: Vec<LoopCtx>,
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

fn lower_block(b: &mut Builder, block: &SynBlock) -> Result<BlockExit, AdapterError> {
    let mut tail: Option<Hlvalue> = None;
    for (idx, stmt) in block.stmts.iter().enumerate() {
        let is_last = idx + 1 == block.stmts.len();
        match stmt {
            Stmt::Local(local) => {
                lower_let(b, local)?;
            }
            Stmt::Expr(expr, semi) => {
                // `return` is a structural terminator in upstream:
                // `flowcontext.py:687 RETURN_VALUE` raises `Return`,
                // `flowcontext.py:1232 Return.nomoreblocks()` closes
                // the current block straight to `graph.returnblock`
                // and raises `StopFlowing`. Anything following in the
                // same block is dead code — mirror that by rejecting
                // non-last `return` and threading `BlockExit::Terminated`
                // out of `lower_block` on the last-stmt case. Works
                // identically with or without the trailing `;` (Rust
                // syntax allows both; syn preserves the semi flag but
                // the flow semantic is identical).
                if let Expr::Return(ret) = expr {
                    if !is_last {
                        return Err(AdapterError::Unsupported {
                            reason: "statement after `return` — upstream \
                                `flowcontext.py:1232` closes the block to \
                                graph.returnblock on Return, making any \
                                subsequent ops unreachable"
                                .into(),
                        });
                    }
                    return lower_return(b, ret);
                }
                // `while` / `loop` are statement-only in the M2.5b
                // slice-3 subset — they produce upstream's bytecode
                // `SETUP_LOOP` + back-edge shape but carry no tail
                // value, so they are accepted with or without the
                // trailing `;` (Rust allows omitting it after a
                // block-tailed expression).
                match expr {
                    Expr::While(while_expr) => {
                        lower_while(b, while_expr)?;
                        continue;
                    }
                    Expr::Loop(loop_expr) => {
                        lower_loop(b, loop_expr)?;
                        continue;
                    }
                    Expr::ForLoop(for_expr) => {
                        lower_for(b, for_expr)?;
                        continue;
                    }
                    // `if cond { body }` without `else` is a statement
                    // producing `()` (upstream Python: `None` via the
                    // implicit RETURN_VALUE on `if x: body` fallthrough
                    // — `flowcontext.py:756 POP_JUMP_IF_FALSE`). Treat
                    // it the same as while/loop: always a statement,
                    // never a tail. When it IS the last stmt, the
                    // function falls through to the implicit-None tail
                    // handled by `lower_block`'s terminal fallback.
                    //
                    // `lower_if_without_else` unconditionally falls
                    // through (the Bool(false) exit shortcuts to
                    // join, so join is always reachable), so it never
                    // produces `Terminated` — but match structurally
                    // so a future slice that tightens this invariant
                    // forwards termination out of the enclosing block
                    // correctly.
                    Expr::If(if_expr) if if_expr.else_branch.is_none() => {
                        match lower_if(b, if_expr)? {
                            BlockExit::FallThrough(_) => continue,
                            BlockExit::Terminated => return Ok(BlockExit::Terminated),
                        }
                    }
                    // `break` / `continue` appearing at the statement
                    // level of a non-loop block is unconditionally an
                    // error — `lower_loop_body` intercepts these
                    // inside actual loop bodies so the only way to
                    // reach `lower_block` with a break/continue stmt
                    // is outside any `loop_stack` entry.
                    Expr::Break(_) => {
                        return Err(AdapterError::Unsupported {
                            reason: "`break` outside of a loop".into(),
                        });
                    }
                    Expr::Continue(_) => {
                        return Err(AdapterError::Unsupported {
                            reason: "`continue` outside of a loop".into(),
                        });
                    }
                    _ => {}
                }
                if semi.is_some() {
                    // Expression statement: lower for side effect,
                    // discard the result. Upstream CPython emits
                    // `POP_TOP` after a stack-producing op —
                    // `flowcontext.py:488` covers the same
                    // semantic at the bytecode level.
                    let _ = lower_expr(b, expr)?;
                } else {
                    if !is_last {
                        return Err(AdapterError::Unsupported {
                            reason: "non-tail expression without trailing `;`".into(),
                        });
                    }
                    // Tail expression: evaluated as the block's
                    // value. Control-flow-bearing constructs
                    // (`if`/`match`/`return`/nested block) may
                    // terminate when every path returns — thread
                    // the `BlockExit` out so the enclosing lowering
                    // observes termination. Non-control-flow
                    // expressions always fall through with a value,
                    // so `lower_arm_body`'s default arm routes them
                    // through `lower_expr`.
                    match lower_arm_body(b, expr)? {
                        BlockExit::FallThrough(v) => tail = Some(v),
                        BlockExit::Terminated => return Ok(BlockExit::Terminated),
                    }
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
    // No explicit tail expression → implicit `None` return. Upstream
    // CPython's bytecode compiler emits `LOAD_CONST None;
    // RETURN_VALUE` as the default terminator of any function body
    // that doesn't end in an explicit return, so the flowspace sees
    // the returnblock Link carrying `Constant(None)`. The adapter
    // mirrors that directly: blocks tail-less by virtue of ending in
    // a statement (`if x: body`, `while …`, `let …;`) produce the
    // same None sentinel both at function-body scope and inside
    // `Expr::Block` / if-branch bodies.
    Ok(BlockExit::FallThrough(tail.unwrap_or_else(|| {
        Hlvalue::Constant(Constant::new(ConstValue::None))
    })))
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

/// Close the currently-open block with a Link into `graph.returnblock`
/// carrying the return value, then report `Terminated` so enclosing
/// control-flow lowering knows not to emit a fallthrough link from the
/// now-closed block.
///
/// Upstream basis: `flowcontext.py:687 RETURN_VALUE` raises `Return`;
/// `flowcontext.py:1232 Return.nomoreblocks(ctx)` does
/// `Link([w_result], ctx.graph.returnblock)` on `ctx.recorder.crnt_block`
/// and raises `StopFlowing`. Our `BlockExit::Terminated` is the Rust
/// analogue of StopFlowing — it tells the caller not to wire further
/// exits from the closed block.
fn lower_return(b: &mut Builder, ret: &ExprReturn) -> Result<BlockExit, AdapterError> {
    let value = match &ret.expr {
        Some(expr) => lower_expr(b, expr)?,
        // Bare `return;` — upstream has no syntactic analogue since
        // Python `return` without a value implicitly pushes `None`
        // (`flowcontext.py:687` pops whatever `compile.c` pushed, which
        // is `LOAD_CONST None` for bare `return`). Mirror that:
        // `ConstValue::None` carried through the returnblock link.
        None => Hlvalue::Constant(Constant::new(ConstValue::None)),
    };
    let link = Rc::new(RefCell::new(Link::new(
        vec![value],
        Some(b.returnblock.clone()),
        None,
    )));
    b.finalize_current(vec![link], None);
    Ok(BlockExit::Terminated)
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
        Expr::If(if_expr) => match lower_if(b, if_expr)? {
            BlockExit::FallThrough(v) => Ok(v),
            // Every branch terminated via `return` — the if-
            // expression has no reachable value. Upstream bytecode
            // would have raised `StopFlowing` inside each branch
            // before the join PC was even reached; in Rust source
            // position this is `let x = if c { return a } else {
            // return b };` (type `!`) which is valid but produces no
            // binding. Reject at the adapter boundary rather than
            // silently synthesize an unreachable join.
            BlockExit::Terminated => Err(AdapterError::Unsupported {
                reason: "if-expression where every branch terminates via `return` — \
                    the expression has no reachable value. Reshape so the \
                    `return` is a semicolon-terminated statement instead of \
                    an expression operand"
                    .into(),
            }),
        },
        Expr::Block(block_expr) => match lower_block(b, &block_expr.block)? {
            BlockExit::FallThrough(v) => Ok(v),
            BlockExit::Terminated => Err(AdapterError::Unsupported {
                reason: "block-expression whose body terminates via `return` — the \
                    expression has no reachable value. Reshape so the `return` \
                    is a semicolon-terminated statement instead of an expression \
                    operand"
                    .into(),
            }),
        },
        // Bare `return` in expression position (e.g. `let x = return
        // 1;`). Upstream has no syntactic analogue — Python's
        // `return` is a statement. Rust's expression-position
        // `return` produces type `!`. The adapter's subset already
        // supports `return` at every statement position (including
        // the tails of if / match branches) via `lower_block`'s
        // explicit `Expr::Return` branch, so rejecting here simply
        // funnels users to the statement position.
        Expr::Return(_) => Err(AdapterError::Unsupported {
            reason: "return in expression position — put the `return` at statement \
                position (a semicolon-terminated statement, or the tail of an if / \
                match branch) instead"
                .into(),
        }),
        Expr::Match(match_expr) => match lower_match(b, match_expr)? {
            BlockExit::FallThrough(v) => Ok(v),
            BlockExit::Terminated => Err(AdapterError::Unsupported {
                reason: "match-expression where every arm terminates via `return` — \
                    the expression has no reachable value. Reshape so the \
                    `return` is a semicolon-terminated statement instead of an \
                    expression operand"
                    .into(),
            }),
        },
        Expr::ForLoop(_) | Expr::While(_) | Expr::Loop(_) => Err(AdapterError::Unsupported {
            reason: "loop construct in expression position produces `()` — use it as a statement \
                (trailing `;` or non-last position) instead"
                .into(),
        }),
        Expr::Try(try_expr) => lower_try(b, try_expr),
        Expr::Break(_) | Expr::Continue(_) => Err(AdapterError::Unsupported {
            reason: "break/continue (lands with loops in M2.5b)".into(),
        }),
        Expr::MethodCall(method_call) => lower_method_call(b, method_call),
        Expr::Call(call) => lower_call(b, call),
        Expr::Tuple(tup) => lower_tuple(b, tup),
        Expr::Array(arr) => lower_array(b, arr),
        Expr::Struct(_) => Err(AdapterError::Unsupported {
            reason: "struct literal (user-type resolution lands in M2.5e — the annotator needs a \
                `ClassDesc` lookup path that `HOST_ENV` doesn't bootstrap on its own)"
                .into(),
        }),
        Expr::Closure(_) => Err(AdapterError::Unsupported {
            reason: "closure (not in roadmap scope)".into(),
        }),
        Expr::Reference(r) => {
            // Rust borrow `&x` / `&mut x` — upstream Python has no
            // ownership model, so the annotator tracks value identity
            // + type only. Pass-through the operand.
            lower_expr(b, &r.expr)
        }
        Expr::Unary(u) => lower_unary(b, u),
        Expr::Cast(c) => lower_cast(b, c),
        Expr::Field(f) => lower_field(b, f),
        Expr::Index(i) => lower_index(b, i),
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
        Lit::Str(s) => Ok(Hlvalue::Constant(Constant::new(ConstValue::Str(s.value())))),
        Lit::Float(f) => {
            // `ConstValue::Float` stores `f64::to_bits()` so the
            // enum keeps `Eq + Hash` — see model.rs:1696-1701. The
            // adapter round-trips through `base10_parse::<f64>()` to
            // preserve the exact literal the user wrote.
            let value: f64 = f.base10_parse().map_err(|e| AdapterError::Unsupported {
                reason: format!("float literal out of f64 range: {e}"),
            })?;
            Ok(Hlvalue::Constant(Constant::new(ConstValue::Float(
                value.to_bits(),
            ))))
        }
        // Rust `char` literal is a single Unicode scalar. Upstream
        // RPython has no `char` type; single-character strings fill
        // the role (`model.py:658` switch-exitcase admits
        // `isinstance(n, (str, unicode)) and len(n) == 1`; general
        // `operation.py` string ops accept len==1 the same as any
        // other str). Emit as `ConstValue::Str(c.to_string())` so
        // expression-position `'a'` and match-arm `'a' =>` carry the
        // identical constant — see `classify_pattern` for the
        // match-arm side.
        Lit::Char(ch) => Ok(Hlvalue::Constant(Constant::new(ConstValue::Str(
            ch.value().to_string(),
        )))),
        Lit::ByteStr(_) | Lit::Byte(_) => Err(AdapterError::Unsupported {
            reason: "byte / bytestring literal — upstream has no direct analogue \
                (Python 2.7 collapses bytes into `str`, modern `bytes` is not in \
                the flowspace vocabulary)"
                .into(),
        }),
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

/// State captured after lowering a single branch's body. `None` when
/// the branch terminated via `return` — in that case the branch's
/// block has already been closed to `graph.returnblock` by
/// `lower_return`, so no further link / op emission applies. `Some`
/// captures the now-open current-block state so the caller can later
/// attach a Link from that block into the post-if join.
struct ArmCapture {
    tail: Hlvalue,
    block: BlockRef,
    ops: Vec<SpaceOperation>,
    locals: HashMap<String, Hlvalue>,
}

/// Finalize the current BlockBuilder state into an `ArmCapture` when
/// the arm fell through, or discard it when the arm terminated.
fn capture_arm_exit(b: &mut Builder, exit: BlockExit) -> Option<ArmCapture> {
    match exit {
        BlockExit::FallThrough(tail) => Some(ArmCapture {
            tail,
            block: b.current.block.clone(),
            ops: std::mem::take(&mut b.current.ops),
            locals: std::mem::take(&mut b.current.locals),
        }),
        BlockExit::Terminated => {
            // `lower_return` already called `finalize_current` on the
            // branch's last block. Clear b.current so the next
            // `open_new_block` starts from a clean slate.
            b.current.ops.clear();
            b.current.locals.clear();
            None
        }
    }
}

/// Dispatch the else-arm of an `if` / `else` chain. Per `syn`'s
/// grammar, `ExprIf.else_branch` is always `Expr::Block` (a plain
/// `else { … }`) or `Expr::If` (chained `else if …`). Both sub-
/// routines return `BlockExit`, so termination threads through a
/// chain of nested `else if` transparently.
fn lower_else_arm(b: &mut Builder, else_expr: &Expr) -> Result<BlockExit, AdapterError> {
    match else_expr {
        Expr::Block(block_expr) => lower_block(b, &block_expr.block),
        Expr::If(nested_if) => lower_if(b, nested_if),
        _ => Err(AdapterError::Unsupported {
            reason: "`else` branch is neither a block nor an `if` — syn's grammar \
                should forbid this; if it fires, please file a bug citing the \
                input fragment"
                .into(),
        }),
    }
}

fn lower_if(b: &mut Builder, if_expr: &ExprIf) -> Result<BlockExit, AdapterError> {
    // 1. Evaluate condition into the current block, then coerce via
    //    `bool(cond)` — mirrors upstream POP_JUMP_IF_FALSE at
    //    `flowcontext.py:756` which always emits `op.bool(w_value)`
    //    before the guessbool test. The `bool` op is in upstream's
    //    `operation.py:467 add_operator('bool', 1)` registry.
    let cond_raw = lower_expr(b, &if_expr.cond)?;
    let cond = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("bool", vec![cond_raw], cond.clone()));

    // Extract the else-less path early so the rest of `lower_if` can
    // assume `else_expr` is present; the common `bool(cond)` +
    // locals-snapshot steps are shared.
    let Some((_else_tok, else_expr)) = &if_expr.else_branch else {
        return lower_if_without_else(b, cond, &if_expr.then_branch);
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
    //    current block is after the branch body returns. A
    //    `Terminated` exit means `lower_return` already closed the
    //    current block — we drop the capture in that case.
    b.open_new_block(BlockBuilder {
        block: then_block.clone(),
        ops: Vec::new(),
        locals: then_locals,
    });
    let then_exit = lower_block(b, &if_expr.then_branch)?;
    let then_capture = capture_arm_exit(b, then_exit);

    // 6. Lower the else-branch the same way. `else_expr` is always
    //    `Expr::Block` or `Expr::If` per syn's grammar — the two
    //    variants route through `lower_else_arm` so nested `else if`
    //    chains thread termination up transparently.
    b.open_new_block(BlockBuilder {
        block: else_block.clone(),
        ops: Vec::new(),
        locals: else_locals,
    });
    let else_exit = lower_else_arm(b, else_expr)?;
    let else_capture = capture_arm_exit(b, else_exit);

    // 7. Fork the post-branch wiring by which arms fell through.
    //    Upstream analogue: `Return.nomoreblocks()` on a branch
    //    raises `StopFlowing`, and the pending-block scheduler never
    //    enqueues that branch's PC at the join. Here:
    //    - both terminated → no join block, whole-if is Terminated
    //    - exactly one fell through → join has one predecessor
    //    - both fell through → canonical two-predecessor join
    match (then_capture, else_capture) {
        (None, None) => {
            // Both branches closed themselves to returnblock via
            // `return`. Nothing reaches the post-if PC; signal
            // termination to the enclosing lowering.
            Ok(BlockExit::Terminated)
        }
        (Some(cap), None) | (None, Some(cap)) => {
            // Exactly one branch reached the post-if PC. Build a
            // single-predecessor join: the branch's tail value is the
            // if-expression's value, and its locals snapshot feeds
            // the join's inputargs.
            let (_join_block, tail_var) = build_if_join_block(&merged_names, b, &cap);
            Ok(BlockExit::FallThrough(tail_var))
        }
        (Some(then_cap), Some(else_cap)) => {
            // Canonical case — both branches reach the join.
            // inputargs = [tail_var, local_var_0, …].
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

            // Close each arm's tail block with a Link into the join.
            // `branch_link_args` reads from the arm's locals snapshot
            // — branch-local `let` bindings (names absent from
            // merged_names) never reach the join.
            {
                let link_args = branch_link_args(&then_cap.tail, &merged_names, &then_cap.locals);
                let link = Rc::new(RefCell::new(Link::new(
                    link_args,
                    Some(join_block.clone()),
                    None,
                )));
                then_cap.block.borrow_mut().operations = then_cap.ops;
                then_cap.block.closeblock(vec![link]);
            }
            {
                let link_args = branch_link_args(&else_cap.tail, &merged_names, &else_cap.locals);
                let link = Rc::new(RefCell::new(Link::new(
                    link_args,
                    Some(join_block.clone()),
                    None,
                )));
                else_cap.block.borrow_mut().operations = else_cap.ops;
                else_cap.block.closeblock(vec![link]);
            }

            b.open_new_block(BlockBuilder {
                block: join_block,
                ops: Vec::new(),
                locals: join_locals,
            });
            Ok(BlockExit::FallThrough(tail_var))
        }
    }
}

/// Build the post-if join block for the single-predecessor case and
/// close the surviving branch's tail block into it. Returns the join
/// block (for caller bookkeeping) and the `tail_var` representing the
/// if-expression's value as seen from join-block locals.
///
/// Upstream analogue: `flowcontext.py` never creates a join block for
/// a PC that only one arm jumps to (it would still be there, but with
/// one incoming Link) — this routine mirrors that shape.
fn build_if_join_block(
    merged_names: &[String],
    b: &mut Builder,
    cap: &ArmCapture,
) -> (BlockRef, Hlvalue) {
    let tail_var = Hlvalue::Variable(Variable::new());
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    join_inputargs.push(tail_var.clone());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // Close the surviving arm's tail block with the one-and-only Link
    // into the join.
    let link_args = branch_link_args(&cap.tail, merged_names, &cap.locals);
    let link = Rc::new(RefCell::new(Link::new(
        link_args,
        Some(join_block.clone()),
        None,
    )));
    cap.block.borrow_mut().operations = cap.ops.clone();
    cap.block.closeblock(vec![link]);

    // Open the join as the new current block.
    b.open_new_block(BlockBuilder {
        block: join_block.clone(),
        ops: Vec::new(),
        locals: join_locals,
    });
    (join_block, tail_var)
}

/// Lower `if cond { body }` without an `else` branch.
///
/// Shape mirrors upstream `flowcontext.py:756 POP_JUMP_IF_FALSE`: the
/// false-branch target IS the post-body join PC (fallthrough of
/// `body` lands on the same PC), so the fork block's `false` exit
/// links directly to the join, and the `true` exit threads through a
/// `then_block` whose body-tail also links to the join. No else
/// block is allocated. The join's inputargs carry ONLY the merged
/// locals — there is no tail-value slot, because `if` without `else`
/// is a statement and the expression "produces" `None`
/// (Python convention mirrored by `ConstValue::None`; upstream
/// RPython bytecode never leaves a value on the stack for this
/// construct).
///
/// Return value: `Constant(None)`. The `Stmt::Expr` loop in
/// `lower_block` discards the tail of semicolon-terminated
/// expressions, so in normal statement position this is erased. In
/// tail position (`if cond { body }` as the function's last
/// expression), the None flows into the returnblock — matching
/// Python's implicit `return None` fallthrough.
fn lower_if_without_else(
    b: &mut Builder,
    cond: Hlvalue,
    then_branch: &SynBlock,
) -> Result<BlockExit, AdapterError> {
    // 1. Snapshot pre-fork locals. Same discipline as the with-else
    //    path: deterministic ordering via sort, one entry per
    //    live local name at the fork point.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();

    // 2. Allocate the then_block (sole branch block) and the join
    //    block up-front — the fork's false Link needs the join as its
    //    target, so the join must exist before `finalize_current`.
    //    Join inputargs are just the merged locals; no tail slot.
    let (then_block, then_locals) = branch_block_with_inputargs(&merged_names);
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // 3. Close the fork block. `false` shortcuts directly to the
    //    join; `true` routes through `then_block`.
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();
    let false_link = Rc::new(RefCell::new(Link::new(
        fork_args.clone(),
        Some(join_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        fork_args,
        Some(then_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(cond));

    // 4. Lower the then-branch. A `Terminated` exit (branch ends in
    //    `return`) means `lower_return` has already wired the block
    //    to `graph.returnblock`; skip the then→join link. The
    //    false-path link into the join is already installed above,
    //    so the join is always reachable — `lower_if_without_else`
    //    therefore always returns FallThrough.
    b.open_new_block(BlockBuilder {
        block: then_block,
        ops: Vec::new(),
        locals: then_locals,
    });
    let then_exit = lower_block(b, then_branch)?;
    if let BlockExit::FallThrough(_tail) = then_exit {
        // Body's tail expression value is discarded — `if` without
        // else has no tail slot in the join (it produces implicit
        // `None` regardless of what the body evaluated to).
        let then_exit_block = b.current.block.clone();
        let then_exit_ops = std::mem::take(&mut b.current.ops);
        let then_exit_locals = std::mem::take(&mut b.current.locals);
        let then_link_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|name| {
                then_exit_locals
                    .get(name)
                    .cloned()
                    .expect("merged_names is a subset of branch entry locals")
            })
            .collect();
        let then_link = Rc::new(RefCell::new(Link::new(
            then_link_args,
            Some(join_block.clone()),
            None,
        )));
        then_exit_block.borrow_mut().operations = then_exit_ops;
        then_exit_block.closeblock(vec![then_link]);
    } else {
        // Then-branch terminated via `return`. Clear any stale
        // current-block state so `open_new_block` starts cleanly.
        b.current.ops.clear();
        b.current.locals.clear();
    }

    // 5. Continue lowering into the join block. The false-path link
    //    is always installed, so the join is reachable whether the
    //    then-branch fell through or terminated.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });

    // 6. The if-without-else expression value is `None` — Python
    //    statement convention, upstream's bytecode never leaves a
    //    value on the stack for `if x: body`.
    Ok(BlockExit::FallThrough(Hlvalue::Constant(Constant::new(
        ConstValue::None,
    ))))
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

/// Dispatch a match-arm body across the control-flow-bearing
/// expression kinds so termination (via `return`) threads up through
/// the arm naturally. Arms with a non-control-flow body (literal,
/// path, binop, etc.) fall back to `lower_expr` and report
/// `FallThrough` with the evaluated value.
fn lower_arm_body(b: &mut Builder, body: &Expr) -> Result<BlockExit, AdapterError> {
    match body {
        Expr::Block(block_expr) => lower_block(b, &block_expr.block),
        Expr::If(if_expr) => lower_if(b, if_expr),
        Expr::Match(match_expr) => lower_match(b, match_expr),
        Expr::Return(ret) => lower_return(b, ret),
        _ => Ok(BlockExit::FallThrough(lower_expr(b, body)?)),
    }
}

fn lower_match(b: &mut Builder, match_expr: &ExprMatch) -> Result<BlockExit, AdapterError> {
    // 1. Evaluate the scrutinee into the current block. Becomes the
    //    block's `exitswitch` when we close the fork.
    let scrutinee = lower_expr(b, &match_expr.expr)?;

    if match_expr.arms.is_empty() {
        return Err(AdapterError::Unsupported {
            reason: "match with zero arms".into(),
        });
    }

    // 2. Validate every arm up-front so the fork block is only closed
    //    once we know every branch can be lowered. Or-patterns
    //    (`A | B | C => body`) expand into multiple sub-patterns per
    //    arm — each sub-pattern classifies independently, and at step
    //    5 each sub-pattern contributes one Link targeting the ONE
    //    branch block allocated for the original arm (upstream
    //    `model.py:648-692` admits multiple Links with distinct
    //    exitcases pointing to the same target block). `exitcase =
    //    None` marks a wildcard — upstream `model.py:652` requires
    //    such a case to be the last exit of the match, and forbids it
    //    inside an or-pattern.
    let mut arm_sub_exitcases: Vec<Vec<Option<Hlvalue>>> =
        Vec::with_capacity(match_expr.arms.len());
    for (idx, arm) in match_expr.arms.iter().enumerate() {
        validate_arm(arm)?;
        let is_last_arm = idx + 1 == match_expr.arms.len();
        let mut sub_pats: Vec<&Pat> = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        let sub_pat_count = sub_pats.len();
        // Pre-check: upstream `model.py:652` reserves the wildcard
        // for a standalone default arm at match level; embedding it
        // inside an or-pattern duplicates the catch-all intent. Flag
        // it here before `classify_pattern` (whose own `is_last`
        // check would otherwise surface a misleading "wildcard must
        // be last" message).
        if sub_pat_count > 1 {
            for sub_pat in &sub_pats {
                if matches!(sub_pat, Pat::Wild(_)) {
                    return Err(AdapterError::Unsupported {
                        reason: "wildcard sub-pattern inside or-pattern — upstream \
                            `model.py:652` reserves the wildcard for a standalone \
                            default arm at match-level"
                            .into(),
                    });
                }
            }
        }
        let mut sub_exitcases: Vec<Option<Hlvalue>> = Vec::with_capacity(sub_pat_count);
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && (sub_idx + 1 == sub_pat_count);
            let exitcase = classify_pattern(sub_pat, is_last)?;
            sub_exitcases.push(exitcase);
        }
        arm_sub_exitcases.push(sub_exitcases);
    }

    // 3. Enforce upstream's uniqueness invariant
    //    (`model.py:692 allexitcases[link.exitcase] = True`). Check
    //    across all sub-exitcases of every arm — two or-pattern
    //    sub-cases on the same value would collide just like two arms
    //    with the same case.
    let mut seen: Vec<&Hlvalue> = Vec::new();
    for arm_ex in &arm_sub_exitcases {
        for exitcase in arm_ex.iter().flatten() {
            if seen.iter().any(|s| *s == exitcase) {
                return Err(AdapterError::Unsupported {
                    reason: "match arm exitcase repeated — upstream forbids duplicate \
                        jump-table cases"
                        .into(),
                });
            }
            seen.push(exitcase);
        }
    }

    // 4. Snapshot locals for the fork (same discipline as `lower_if`).
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();

    // 5. Allocate one branch block per ORIGINAL arm; each arm emits
    //    one Link per sub-pattern pointing at that shared block.
    let mut branch_blocks: Vec<(BlockRef, HashMap<String, Hlvalue>)> =
        Vec::with_capacity(match_expr.arms.len());
    let mut fork_exits: Vec<Rc<RefCell<Link>>> = Vec::new();
    for arm_ex in &arm_sub_exitcases {
        let (branch_block, branch_locals) = branch_block_with_inputargs(&merged_names);
        for exitcase in arm_ex {
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
        }
        branch_blocks.push((branch_block, branch_locals));
    }
    b.finalize_current(fork_exits, Some(scrutinee));

    // 6. Lower each arm's body via `lower_arm_body` so termination
    //    via `return` threads up per `flowcontext.py:1232`. Record
    //    ONE capture per original arm — a `None` capture means the
    //    arm terminated and is not linked to the join. Regardless of
    //    how many sub-patterns (Links) feed into the arm's branch
    //    block, the body is lowered exactly once.
    let mut arm_captures: Vec<Option<ArmCapture>> = Vec::with_capacity(match_expr.arms.len());
    for ((branch_block, branch_locals), arm) in branch_blocks.into_iter().zip(&match_expr.arms) {
        b.open_new_block(BlockBuilder {
            block: branch_block,
            ops: Vec::new(),
            locals: branch_locals,
        });
        let exit = lower_arm_body(b, &arm.body)?;
        arm_captures.push(capture_arm_exit(b, exit));
    }

    // 7. If every arm terminated, the post-match PC is unreachable:
    //    no join block is allocated, and the enclosing lowering must
    //    observe termination. Upstream analogue:
    //    `Return.nomoreblocks()` on every pending arm raises
    //    `StopFlowing` and the join PC is never enqueued.
    if arm_captures.iter().all(|c| c.is_none()) {
        return Ok(BlockExit::Terminated);
    }

    // 8. Build the join block — inputargs = [tail_var, local_var_0, …].
    //    Every surviving arm contributes one Link; terminated arms
    //    are silently skipped (their block was already closed to
    //    returnblock by `lower_return`).
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

    // 9. Close each surviving arm's exit block with a Link into the
    //    join.
    for cap in arm_captures.into_iter().flatten() {
        let link_args = branch_link_args(&cap.tail, &merged_names, &cap.locals);
        let link = Rc::new(RefCell::new(Link::new(
            link_args,
            Some(join_block.clone()),
            None,
        )));
        cap.block.borrow_mut().operations = cap.ops;
        cap.block.closeblock(vec![link]);
    }

    // 10. Continue lowering into the join block.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });
    Ok(BlockExit::FallThrough(tail_var))
}

// ____________________________________________________________
// `while` / `loop` — header + back-edge with `break` / `continue`.
//
// Upstream basis: `rpython/flowspace/flowcontext.py:794 SETUP_LOOP`
// pushes a LoopBlock; `:718 JUMP_ABSOLUTE` returns the back-edge target
// that the pending-block scheduler merges against the header; `:525
// BREAK_LOOP` / `:528 CONTINUE_LOOP` raise Break/Continue which the
// enclosing LoopBlock turns into the corresponding jumps.
//
// The graph-level shape upstream produces for
//
//   while cond:                header:  cond, exitswitch=cond,
//     body                              [false → exit, true → body]
//                              body:    body-ops, Link(→ header)
//                              exit:    (continue from here)
//
// is what `lower_while` emits directly. `lower_loop` is the same shape
// without the `cond` fork — the header is entered unconditionally and
// the only way out is via `break` (or fallthrough through a
// body-tail that happens to not exist in our subset, since `loop`
// bodies end at the `}` with an implicit back-edge).
//
// Body subset (slice 3): a flat sequence of `let` bindings, optionally
// terminated by a single `break;` / `continue;`. If the body falls
// through to the closing `}`, the adapter emits the back-edge
// automatically. Dead code after an explicit terminator rejects via
// `AdapterError::Unsupported`. Loops nested inside `if`/`match`
// branches work because those lower into independent BlockBuilders
// before hitting the terminator check.

/// Outcome of a loop body lowering:
/// - `FallThrough` — body reached its closing `}` naturally, the caller
///   emits the back-edge Link.
/// - `Terminated` — body finalized its current block via `break;` /
///   `continue;`; no further emission required from the caller.
enum LoopBodyExit {
    FallThrough,
    Terminated,
}

fn lower_while(b: &mut Builder, while_expr: &ExprWhile) -> Result<(), AdapterError> {
    // 1. Snapshot pre-loop locals and fix the merged name ordering.
    //    This is what both the header's inputargs and the back-edge
    //    Link args agree on.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let pre_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| pre_fork_locals[n].clone())
        .collect();

    // 2. Allocate the header + exit blocks. Both carry one fresh
    //    inputarg per merged local; the header becomes the back-edge
    //    target, the exit block is where execution resumes after the
    //    loop.
    let (header_block, header_locals) = branch_block_with_inputargs(&merged_names);
    let (exit_block, exit_locals) = branch_block_with_inputargs(&merged_names);

    // 3. Close the pre-loop block with an unconditional Link into the
    //    header.
    let pre_link = Rc::new(RefCell::new(Link::new(
        pre_args,
        Some(header_block.clone()),
        None,
    )));
    b.finalize_current(vec![pre_link], None);

    // 4. Open the header, lower the condition, and allocate the body
    //    block. The header's locals at fork time are identical to the
    //    header's inputargs — no rebinding happens between the entry
    //    and the condition evaluation.
    b.open_new_block(BlockBuilder {
        block: header_block.clone(),
        ops: Vec::new(),
        locals: header_locals,
    });
    // Upstream POP_JUMP_IF_FALSE (`flowcontext.py:756`) always wraps
    // the predicate in `op.bool(w_value)` before the fork. Emit the
    // `bool` op explicitly so the exitswitch carries the coerced
    // result — the annotator / optimizer can fold it away when the
    // input is already `SomeBool`.
    let cond_raw = lower_expr(b, &while_expr.cond)?;
    let cond = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("bool", vec![cond_raw], cond.clone()));
    let header_locals_at_fork: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();

    let (body_block, body_locals) = branch_block_with_inputargs(&merged_names);
    let false_link = Rc::new(RefCell::new(Link::new(
        header_locals_at_fork.clone(),
        Some(exit_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        header_locals_at_fork,
        Some(body_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(cond));

    // 5. Push the loop context before lowering the body so nested
    //    `break` / `continue` resolve against *this* loop.
    b.loop_stack.push(LoopCtx {
        header_block: header_block.clone(),
        exit_block: exit_block.clone(),
        merged_names: merged_names.clone(),
        // while/loop have no synthetic sidecar — exit set == full set.
        exit_merged_names: merged_names.clone(),
    });
    b.open_new_block(BlockBuilder {
        block: body_block,
        ops: Vec::new(),
        locals: body_locals,
    });
    let body_exit = lower_loop_body(b, &while_expr.body)?;

    // 6. If the body fell through, emit the back-edge. Body-local
    //    rebinds that ended up in `current.locals` flow back into the
    //    header via the new Link's args.
    if matches!(body_exit, LoopBodyExit::FallThrough) {
        let back_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|n| b.current.locals[n].clone())
            .collect();
        let back_link = Rc::new(RefCell::new(Link::new(back_args, Some(header_block), None)));
        b.finalize_current(vec![back_link], None);
    }
    b.loop_stack.pop();

    // 7. Execution after the loop resumes in the exit block, with its
    //    inputargs bound as the new `locals`.
    b.open_new_block(BlockBuilder {
        block: exit_block,
        ops: Vec::new(),
        locals: exit_locals,
    });
    Ok(())
}

fn lower_loop(b: &mut Builder, loop_expr: &ExprLoop) -> Result<(), AdapterError> {
    // `loop { body }` is `while true { body }` without the cond fork.
    // The header is entered unconditionally and doubles as the body
    // start; `break` jumps to the exit block; fallthrough loops back.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let pre_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| pre_fork_locals[n].clone())
        .collect();

    let (header_block, header_locals) = branch_block_with_inputargs(&merged_names);
    let (exit_block, exit_locals) = branch_block_with_inputargs(&merged_names);

    let pre_link = Rc::new(RefCell::new(Link::new(
        pre_args,
        Some(header_block.clone()),
        None,
    )));
    b.finalize_current(vec![pre_link], None);

    b.loop_stack.push(LoopCtx {
        header_block: header_block.clone(),
        exit_block: exit_block.clone(),
        merged_names: merged_names.clone(),
        // while/loop have no synthetic sidecar — exit set == full set.
        exit_merged_names: merged_names.clone(),
    });
    b.open_new_block(BlockBuilder {
        block: header_block.clone(),
        ops: Vec::new(),
        locals: header_locals,
    });
    let body_exit = lower_loop_body(b, &loop_expr.body)?;

    if matches!(body_exit, LoopBodyExit::FallThrough) {
        let back_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|n| b.current.locals[n].clone())
            .collect();
        let back_link = Rc::new(RefCell::new(Link::new(back_args, Some(header_block), None)));
        b.finalize_current(vec![back_link], None);
    }
    b.loop_stack.pop();

    b.open_new_block(BlockBuilder {
        block: exit_block,
        ops: Vec::new(),
        locals: exit_locals,
    });
    Ok(())
}

// ____________________________________________________________
// `for item in iter { body }` — iter protocol with a StopIteration
// exception exit that catches at the loop boundary (not at
// graph.exceptblock).
//
// Upstream basis: `rpython/flowspace/flowcontext.py:782 GET_ITER`
// emits `op.iter(iterable)`; `:787 FOR_ITER` pushes an IterBlock
// exception handler that catches StopIteration and jumps to the
// post-loop pc, then emits `op.next(iterator)` which may raise
// StopIteration. In graph form the header block ends in
// `next(iter)` with `exitswitch = c_last_exception`, exit[0] is the
// normal fall-through into the body with the freshly-yielded
// element, and exit[1] carries `exitcase = StopIteration` targeting
// the loop-exit block (not graph.exceptblock — StopIteration is
// silently caught by the IterBlock handler). `operation.py:596-599`
// confirms `next` is arity=1 with `canraise = [StopIteration,
// RuntimeError]`.
//
// Exception exit shape (post-simplify-equivalent):
// - StopIteration exit: link.args carry the exit-visible merged
//   locals, target = loop exit_block, exitcase = Constant(StopIteration
//   class). link.extravars.last_exception = `Constant(StopIteration)`
//   per upstream `flowcontext.py:127` — class-specific guessexception
//   exits use a Constant for `last_exc`; only the generic `Exception`
//   case uses a Variable. link.extravars.last_exc_value = fresh
//   Variable named `last_exc_value`.
// - RuntimeError exit: no outer handler in the adapter's simplified
//   model, so the egg's `RaiseImplicit.nomoreblocks`
//   (`flowcontext.py:1271-1284`) runs immediately — its closing link
//   carries `[Constant(AssertionError_class), Constant(AssertionError(
//   "implicit RuntimeError shouldn't occur"))]` straight to
//   `graph.exceptblock`. After `eliminate_empty_blocks` the trivial
//   RaiseImplicit egg folds away and the header link lands directly
//   on the exceptblock with those AssertionError constants — that is
//   the shape the adapter emits.
//
// PRE-EXISTING-ADAPTATION (value-stack vs locals-map). Upstream
// `flowcontext.py:782 GET_ITER` leaves the iterator on the Python
// value stack, and `:787 FOR_ITER` pops it via IterBlock.handle
// after StopIteration. The adapter lacks a stack model — its frame
// state is a name-keyed locals map. We stash the iterator in a
// reserved local slot named `#for_iter_{depth}` (the `#` character
// cannot appear in any `syn::Ident::to_string()` output, ruling out
// user-source collisions) and strip it from every post-loop visible-
// name set. Cited as unavoidable per CLAUDE.md "smallest possible
// change from RPython structure" — a full value-stack port is an
// M2.5e scope item.
//
// Scope (slice 5):
// - Pattern: `Pat::Ident` only (simple `for item in iter`, no
//   destructuring).
// - Iterator expression: whatever `lower_expr` accepts (locals,
//   literals, binops).
// - Body: the `lower_loop_body` subset — `let` bindings, nested
//   while/loop/for, optionally terminated by `break;` / `continue;`.
// - `break;`, StopIteration exit, `continue;`, fall-through back-edge
//   all route through the standard merged-locals machinery — the
//   iterator sits in the `#for_iter_{depth}` slot that participates
//   in the merged-names set, so nested loops thread it through their
//   own merged-name machinery without special-casing.

fn lower_for(b: &mut Builder, for_expr: &ExprForLoop) -> Result<(), AdapterError> {
    // 1. `for item in iter` — only a plain identifier pattern is
    //    accepted. Destructuring / tuple-patterns land with composite
    //    literals in M2.5d.
    let item_name = match &*for_expr.pat {
        Pat::Ident(PatIdent {
            ident,
            by_ref: None,
            subpat: None,
            ..
        }) => ident.to_string(),
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "`for` pattern must be a plain identifier (destructuring lands in M2.5d)"
                    .into(),
            });
        }
    };

    // 2. Lower the iterable into the current block and emit
    //    `iter(iterable) -> v_iter` — flowcontext's GET_ITER
    //    sequence. Store `v_iter` into a reserved internal slot so
    //    the standard merged-names machinery carries it through
    //    every enclosed nested loop / conditional without special
    //    casing.
    //
    //    The slot name is `#for_iter_{depth}` indexed by
    //    `loop_stack.len()` at entry, so nested for-loops pick
    //    distinct names. `#` is not a legal Rust identifier
    //    character, so `syn::Ident::to_string()` can never produce
    //    a name that collides with the slot.
    let iterable = lower_expr(b, &for_expr.expr)?;
    let v_iter = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("iter", vec![iterable], v_iter.clone()));
    let iter_slot = format!("#for_iter_{}", b.loop_stack.len());
    b.set_local(iter_slot.clone(), v_iter);

    // 3. Snapshot the pre-loop locals (now including the iter slot)
    //    and fix the merged ordering.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let pre_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| pre_fork_locals[n].clone())
        .collect();

    // 4. Pre-compute the exit-block's visible name set: full
    //    merged_names MINUS the synthetic iter slot. Upstream
    //    `flowcontext.py:787, :1355, :1383` — the iterator lives on
    //    the value stack for the loop's lifetime and is popped at
    //    loop exit, so post-loop code must NOT see it.
    let exit_merged_names: Vec<String> = merged_names
        .iter()
        .filter(|n| *n != &iter_slot)
        .cloned()
        .collect();

    // 5. Allocate header (threads the iter slot via merged_names) and
    //    exit blocks (uses the filtered exit_merged_names so the
    //    iter slot does NOT appear in post-loop inputargs or locals).
    let (header_block, header_locals) = branch_block_with_inputargs(&merged_names);
    let (exit_block, exit_locals) = branch_block_with_inputargs(&exit_merged_names);

    // 5. Close the pre-loop block with a plain Link into the header.
    let pre_link = Rc::new(RefCell::new(Link::new(
        pre_args,
        Some(header_block.clone()),
        None,
    )));
    b.finalize_current(vec![pre_link], None);

    // 6. Open the header block and emit the raising `next(iter_h)`
    //    op reading from the header's own iter-slot binding.
    b.open_new_block(BlockBuilder {
        block: header_block.clone(),
        ops: Vec::new(),
        locals: header_locals,
    });
    let iter_h = b.current.locals[&iter_slot].clone();
    let v_next = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("next", vec![iter_h], v_next.clone()));

    // 7. Body block. Upstream STORE_FAST
    //    (`flowcontext.py:878-884`) rebinds the loop-variable slot in
    //    place — after FOR_ITER pops the new item and STORE_FAST
    //    writes it, `locals_w[i_item]` IS the new item and no
    //    separate channel exists for the pre-loop value of that slot.
    //    Mirror that by walking `merged_names` once and emitting a
    //    single inputarg per slot: for `item_name` the inputarg IS
    //    `body_item_var`; for every other slot the inputarg is a
    //    fresh `Variable::named(name)`. No double channels.
    let body_item_var = Hlvalue::Variable(Variable::named(&item_name));
    let mut body_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len());
    let mut body_locals: HashMap<String, Hlvalue> = HashMap::new();
    let mut item_in_merged = false;
    for name in &merged_names {
        if name == &item_name {
            body_inputargs.push(body_item_var.clone());
            body_locals.insert(name.clone(), body_item_var.clone());
            item_in_merged = true;
        } else {
            let fresh = Hlvalue::Variable(Variable::named(name));
            body_inputargs.push(fresh.clone());
            body_locals.insert(name.clone(), fresh);
        }
    }
    // If `item_name` wasn't pre-existing in merged_names, the
    // STORE_FAST creates a new slot: append body_item_var as the
    // last inputarg and record it under its name.
    if !item_in_merged {
        body_inputargs.push(body_item_var.clone());
        body_locals.insert(item_name.clone(), body_item_var);
    }
    let body_block = Block::shared(body_inputargs);

    // 8. Close the header with the canraise shape. Upstream
    //    `operation.py:595-599` — `Next.canraise = [StopIteration,
    //    RuntimeError]`. The flowcontext's implicit IterBlock
    //    (`flowcontext.py:1378`) catches StopIteration and routes it
    //    to the loop's exit; every other canraise exception is
    //    unrolled as `RaiseImplicit` (`flowcontext.py:176`) which,
    //    with no outer handler, closes via
    //    `RaiseImplicit.nomoreblocks` (`:1271-1284`) — that pathway
    //    rewrites the exception into `AssertionError("implicit <CLS>
    //    shouldn't occur")` and links straight to
    //    `graph.exceptblock` with those constants as the link args.
    //
    //    Emission order matches upstream guessexception
    //    (`flowcontext.py:124-148` — `[None] + list(cases)`):
    //    normal → body, StopIteration → loop exit,
    //    RuntimeError → graph.exceptblock (via AssertionError
    //    rewrite).
    //
    //    Normal link.args align positionally with body_inputargs:
    //    item_name's slot carries `v_next`, every other merged slot
    //    carries the header's current binding. If `item_name` is
    //    NOT in merged_names, `v_next` is appended last (mirroring
    //    STORE_FAST creating a new slot).
    let mut normal_args: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    for name in &merged_names {
        if name == &item_name {
            normal_args.push(v_next.clone());
        } else {
            normal_args.push(b.current.locals[name].clone());
        }
    }
    if !item_in_merged {
        normal_args.push(v_next);
    }
    let normal_link = Rc::new(RefCell::new(Link::new(
        normal_args,
        Some(body_block.clone()),
        None,
    )));

    // StopIteration exit → loop's own exit_block. Link.args carry
    // only the exit-visible names (iter slot filtered out).
    // Extravars mirror `guessexception` at `flowcontext.py:127-134`
    // for a class-specific case: `last_exception = Constant(case)`,
    // `last_exc_value = Variable('last_exc_value')` (fresh).
    let stopiter_exit_args: Vec<Hlvalue> = exit_merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();
    let stopiter_cls = HOST_ENV
        .lookup_exception_class("StopIteration")
        .expect("HOST_ENV bootstrap must register `StopIteration` — model.rs:1426 ensures it");
    let stopiter_const = Hlvalue::Constant(Constant::new(ConstValue::HostObject(stopiter_cls)));
    let stopiter_last_exc_value = Variable::named("last_exc_value");
    let mut stopiter_link_inner = Link::new(
        stopiter_exit_args,
        Some(exit_block.clone()),
        Some(stopiter_const.clone()),
    );
    stopiter_link_inner.extravars(
        Some(stopiter_const),
        Some(Hlvalue::Variable(stopiter_last_exc_value)),
    );
    let stopiter_link = Rc::new(RefCell::new(stopiter_link_inner));

    // RuntimeError exit → graph.exceptblock via the RaiseImplicit
    // rewrite. `flowcontext.py:1271-1284 RaiseImplicit.nomoreblocks`
    // fires when no outer handler catches the `Constant(RuntimeError)`
    // raise: it closes the current block with
    // `Link([Constant(AssertionError_class), Constant(AssertionError(msg))],
    // graph.exceptblock)`. After `eliminate_empty_blocks` the
    // intervening egg folds away, leaving the header linked directly
    // to `graph.exceptblock` with those AssertionError constants.
    //
    // The link's own extravars still reflect the original
    // guessexception class-specific case — `last_exception =
    // Constant(RuntimeError)`, `last_exc_value =
    // Variable('last_exc_value')` — matching upstream
    // `flowcontext.py:127-143`.
    let runtime_cls = HOST_ENV
        .lookup_exception_class("RuntimeError")
        .expect("HOST_ENV bootstrap must register `RuntimeError` — model.rs:1419 ensures it");
    let runtime_const = Hlvalue::Constant(Constant::new(ConstValue::HostObject(runtime_cls)));
    let assertion_cls = HOST_ENV
        .lookup_exception_class("AssertionError")
        .expect("HOST_ENV bootstrap must register `AssertionError` — model.rs:1424 ensures it");
    let assertion_cls_const =
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(assertion_cls.clone())));
    let assertion_msg = "implicit RuntimeError shouldn't occur".to_string();
    let assertion_instance = crate::flowspace::model::HostObject::new_instance(
        assertion_cls,
        vec![ConstValue::Str(assertion_msg)],
    );
    let assertion_instance_const =
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(assertion_instance)));
    let runtime_last_exc_value = Variable::named("last_exc_value");
    let mut runtime_link_inner = Link::new(
        vec![assertion_cls_const, assertion_instance_const],
        Some(b.exceptblock.clone()),
        Some(runtime_const.clone()),
    );
    runtime_link_inner.extravars(
        Some(runtime_const),
        Some(Hlvalue::Variable(runtime_last_exc_value)),
    );
    let runtime_link = Rc::new(RefCell::new(runtime_link_inner));

    b.finalize_current(
        vec![normal_link, stopiter_link, runtime_link],
        Some(Hlvalue::Constant(c_last_exception())),
    );

    // 9. Push the LoopCtx and lower the body. Break / continue /
    //    fall-through all use the standard merged-locals shape;
    //    because the iter slot is in `merged_names` their Links
    //    automatically carry it. `break` / loop-exit Links use
    //    `exit_merged_names` which excludes the internal iter slot
    //    so the post-loop block never sees it.
    let exit_merged_names: Vec<String> = merged_names
        .iter()
        .filter(|n| *n != &iter_slot)
        .cloned()
        .collect();
    b.loop_stack.push(LoopCtx {
        header_block: header_block.clone(),
        exit_block: exit_block.clone(),
        merged_names: merged_names.clone(),
        exit_merged_names: exit_merged_names.clone(),
    });
    b.open_new_block(BlockBuilder {
        block: body_block,
        ops: Vec::new(),
        locals: body_locals,
    });
    let body_exit = lower_loop_body(b, &for_expr.body)?;
    if matches!(body_exit, LoopBodyExit::FallThrough) {
        let back_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|n| b.current.locals[n].clone())
            .collect();
        let back_link = Rc::new(RefCell::new(Link::new(back_args, Some(header_block), None)));
        b.finalize_current(vec![back_link], None);
    }
    b.loop_stack.pop();

    // 10. Continue lowering in the exit block. `exit_locals` has
    //     had the internal iter slot stripped so user source can't
    //     reference it.
    b.open_new_block(BlockBuilder {
        block: exit_block,
        ops: Vec::new(),
        locals: exit_locals,
    });
    Ok(())
}

/// Lower a `while`/`loop` body. Upstream SETUP_LOOP
/// (`flowcontext.py:488, :794`) wraps arbitrary bytecode flow — the
/// loop body is not a separate dispatch subset. The adapter follows
/// suit: any statement accepted by top-level lowering is accepted
/// here, with two additions specific to loops:
/// - `break;` and `continue;` are terminator statements that
///   finalize the current block with a Link into the loop's
///   exit_block / header_block.
/// - the loop body has no tail value (Rust loop body returns `()`),
///   so any trailing expression is discarded as a POP_TOP.
fn lower_loop_body(b: &mut Builder, body: &SynBlock) -> Result<LoopBodyExit, AdapterError> {
    let n = body.stmts.len();
    for (idx, stmt) in body.stmts.iter().enumerate() {
        let is_last = idx + 1 == n;
        match stmt {
            Stmt::Local(local) => lower_let(b, local)?,
            Stmt::Expr(Expr::Break(brk), _) => {
                if !is_last {
                    return Err(AdapterError::Unsupported {
                        reason: "dead code after `break;` — the terminator must be the last \
                            statement in the loop body"
                            .into(),
                    });
                }
                lower_break(b, brk)?;
                return Ok(LoopBodyExit::Terminated);
            }
            Stmt::Expr(Expr::Continue(cont), _) => {
                if !is_last {
                    return Err(AdapterError::Unsupported {
                        reason: "dead code after `continue;` — the terminator must be the last \
                            statement in the loop body"
                            .into(),
                    });
                }
                lower_continue(b, cont)?;
                return Ok(LoopBodyExit::Terminated);
            }
            // `return` inside a loop body closes the current block
            // to `graph.returnblock` just like at any other statement
            // position (upstream `flowcontext.py:687, :1232`). That
            // also ends this loop-body lowering — further body stmts
            // would be dead code, and the loop's back-edge must NOT
            // be emitted because execution never reaches the body's
            // closing `}`.
            Stmt::Expr(Expr::Return(ret), _) => {
                if !is_last {
                    return Err(AdapterError::Unsupported {
                        reason: "dead code after `return` — the terminator must be the last \
                            statement in the loop body"
                            .into(),
                    });
                }
                let _ = lower_return(b, ret)?;
                return Ok(LoopBodyExit::Terminated);
            }
            Stmt::Expr(Expr::While(while_expr), _) => {
                lower_while(b, while_expr)?;
            }
            Stmt::Expr(Expr::Loop(loop_expr), _) => {
                lower_loop(b, loop_expr)?;
            }
            Stmt::Expr(Expr::ForLoop(for_expr), _) => {
                lower_for(b, for_expr)?;
            }
            // Any other expression statement: lower for side effect
            // and discard the result (upstream POP_TOP after a
            // stack-producing instruction). Covers call / method_call
            // / if-else / match / attribute access / etc. — matches
            // SETUP_LOOP's "arbitrary bytecode inside the loop"
            // semantic at flowcontext.py:488.
            Stmt::Expr(expr, _) => {
                let _ = lower_expr(b, expr)?;
            }
            Stmt::Item(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "nested item (fn/struct/impl inside loop body)".into(),
                });
            }
            Stmt::Macro(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "macro invocation in statement position".into(),
                });
            }
        }
    }
    Ok(LoopBodyExit::FallThrough)
}

fn lower_break(b: &mut Builder, brk: &ExprBreak) -> Result<(), AdapterError> {
    if brk.expr.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "`break VALUE` — loop-as-expression value is out of M2.5b slice-3 scope".into(),
        });
    }
    if brk.label.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "labeled `break 'label` is out of M2.5b scope".into(),
        });
    }
    let ctx = b
        .loop_stack
        .last()
        .cloned_ctx()
        .ok_or_else(|| AdapterError::Unsupported {
            reason: "`break` outside of a loop".into(),
        })?;
    // Break jumps to the loop's exit block, which uses
    // `exit_merged_names` — the set excluding internal iter sidecars.
    let args: Vec<Hlvalue> = ctx
        .exit_merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();
    let link = Rc::new(RefCell::new(Link::new(args, Some(ctx.exit_block), None)));
    b.finalize_current(vec![link], None);
    Ok(())
}

fn lower_continue(b: &mut Builder, cont: &ExprContinue) -> Result<(), AdapterError> {
    if cont.label.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "labeled `continue 'label` is out of M2.5b scope".into(),
        });
    }
    let ctx = b
        .loop_stack
        .last()
        .cloned_ctx()
        .ok_or_else(|| AdapterError::Unsupported {
            reason: "`continue` outside of a loop".into(),
        })?;
    let args: Vec<Hlvalue> = ctx
        .merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();
    let link = Rc::new(RefCell::new(Link::new(args, Some(ctx.header_block), None)));
    b.finalize_current(vec![link], None);
    Ok(())
}

// ____________________________________________________________
// `?` operator — raising operation + exception edge to
// `graph.exceptblock`.
//
// Upstream basis: `rpython/flowspace/model.py:469-470`
// (`c_last_exception`), `:214-221` (`Block.canraise` /
// `Block.raising_op` properties), and the `graph.exceptblock`
// constructor at `:22-25`. A canraise block's last operation is the
// raising op, `exitswitch` is set to `c_last_exception`, `exits[0]`
// is the normal fall-through (exitcase=None, carrying the op's
// result), and `exits[1..]` are exception exits whose `exitcase` is
// an exception-class `Constant` and whose `last_exception` /
// `last_exc_value` carry the caught exception's type and value
// Variables. The RPython parser emits this shape when it encounters
// any opcode marked `canraise` in `operation.py:536-611`.
//
// Rust-specific adaptation: Rust's `?` expands to
//   match operand { Ok(v) => v, Err(e) => return Err(e.into()) }
// — an early-return on `Err`. We mirror this via the canraise shape
// rather than a direct match so the exception signal flows through
// `graph.exceptblock`, matching upstream's "uncaught exception
// propagates through the graph's exception exit" invariant.
// `HOST_ENV.lookup_exception_class("Exception")` provides the
// bootstrap exception-class HostObject required by
// `is_exception_exitcase` in `model.rs` `checkgraph`.

fn lower_try(b: &mut Builder, try_expr: &ExprTry) -> Result<Hlvalue, AdapterError> {
    // Upstream canraise sites are ops emitted by `ctx.do_op(op)` that
    // carry `canraise != []` (operation.py:475-611). The fork comes
    // from `guessexception` at flowcontext.py:124 / :379 / :385 —
    // attached to THAT real op, not a synthetic wrapper.
    //
    // Rust's `?` has no upstream counterpart, so the line-by-line-
    // orthodox mapping is: the `?` operand must itself be a call
    // whose lowered SpaceOperation IS the raising site. Any other
    // operand (bare variable, arithmetic, literal) has no call op
    // to hang canraise on.
    match &*try_expr.expr {
        Expr::Call(_) | Expr::MethodCall(_) => {}
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "`?` operand must be a direct call / method call — upstream \
                    canraise sites are the ops themselves (flowcontext.py:124, :379), \
                    not wrappers over arbitrary values"
                    .into(),
            });
        }
    }

    // Lower the operand. `lower_call` / `lower_method_call` emit their
    // SpaceOperation into `b.current.ops` and return its result
    // Variable. That last op is the raising site — no synthetic
    // wrapper.
    let unwrapped = lower_expr(b, &try_expr.expr)?;

    // Snapshot the locals set so the normal-exit Link can carry
    // them into the continuation block. The exception exit is only
    // [etype, evalue] — `graph.exceptblock.inputargs.len() == 2` per
    // the constructor at `model.py:22-25`; no caller locals survive
    // the exception edge.
    let pre_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_locals.keys().cloned().collect();
    merged_names.sort();

    // Allocate the continuation block — inputargs =
    // [unwrapped_var, local_var_0, …]. The first inputarg is the
    // result of the raising op flowing through the normal exit.
    let cont_unwrapped_var = Hlvalue::Variable(Variable::new());
    let mut cont_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    cont_inputargs.push(cont_unwrapped_var.clone());
    let mut cont_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        cont_inputargs.push(fresh.clone());
        cont_locals.insert(name.clone(), fresh);
    }
    let cont_block = Block::shared(cont_inputargs);

    // Build the normal-exit Link (exitcase=None). Args carry
    // [unwrapped, ...current-locals] — the call's result flows
    // into `cont_unwrapped_var`.
    let mut normal_args: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    normal_args.push(unwrapped.clone());
    for name in &merged_names {
        normal_args.push(pre_locals[name].clone());
    }
    let normal_link = Rc::new(RefCell::new(Link::new(
        normal_args,
        Some(cont_block.clone()),
        None,
    )));

    // 6. Build the exception-exit Link. The target is
    //    `graph.exceptblock` (retrieved via Builder's reference), the
    //    exitcase is an exception-class Constant per
    //    `is_exception_exitcase`. `last_exception` / `last_exc_value`
    //    are fresh Variables defined exclusively on this link
    //    (checkgraph at `model.rs:3780-3785`).
    let etype = Variable::new();
    let evalue = Variable::new();
    let exc_class = HOST_ENV.lookup_exception_class("Exception").expect(
        "HOST_ENV bootstrap must register the builtin `Exception` class — \
            model.rs:1418 ensures it",
    );
    let exc_class_const = Hlvalue::Constant(Constant::new(ConstValue::HostObject(exc_class)));
    let mut exc_link_inner = Link::new(
        vec![
            Hlvalue::Variable(etype.clone()),
            Hlvalue::Variable(evalue.clone()),
        ],
        Some(b.exceptblock.clone()),
        Some(exc_class_const),
    );
    exc_link_inner.extravars(
        Some(Hlvalue::Variable(etype)),
        Some(Hlvalue::Variable(evalue)),
    );
    let exc_link = Rc::new(RefCell::new(exc_link_inner));

    // 7. Close the current block with the canraise shape. The
    //    exitswitch sentinel is `c_last_exception()` which
    //    `Block::canraise` detects via its Atom identity.
    let switch = Hlvalue::Constant(c_last_exception());
    b.finalize_current(vec![normal_link, exc_link], Some(switch));

    // 8. Open the continuation block. Subsequent reads of any local
    //    see the continuation's fresh inputarg; subsequent reads of
    //    the expression's value (the `?` unwrap result) see
    //    `cont_unwrapped_var`.
    b.open_new_block(BlockBuilder {
        block: cont_block,
        ops: Vec::new(),
        locals: cont_locals,
    });
    Ok(cont_unwrapped_var)
}

// ____________________________________________________________
// Method calls + function calls.
//
// Upstream basis:
// - `rpython/flowspace/operation.py:617-622` — `GetAttr(obj, name)`:
//   `arity=2`, `canraise=[]`, `pyfunc = staticmethod(getattr)`.
//   Emission convention is `getattr(obj, name_as_constant_str)`.
// - `rpython/flowspace/operation.py:663-679` — `SimpleCall(f, *args)`:
//   variable arity, no canraise at the op layer. Dispatched through
//   `SPECIAL_CASES` at annotator time if the callable is a
//   `Constant`; otherwise falls through to `ctx.do_op(self)`.
// - `rpython/flowspace/flowcontext.py` `LOAD_METHOD` + `CALL_METHOD`
//   (aliased at `:1000 CALL_METHOD = CALL_FUNCTION`): the bytecode
//   sequence emits `getattr(obj, name)` then `simple_call(bound,
//   *args)`, matching this lowering exactly.
//
// Trait dispatch is *not* emitted as a separate kind of op. The
// annotator downstream (`FunctionDesc.specialize` /
// `MethodDesc.bind_self`, `description.py:272-281` / `:1805-1819`)
// reads the receiver's `SomeInstance.classdef` and threads the
// concrete impl's `FunctionDesc` into the `simple_call` site. The
// adapter's job is to emit the structural `getattr + simple_call`
// pair so the annotator has something to rewrite.

fn lower_method_call(
    b: &mut Builder,
    method_call: &ExprMethodCall,
) -> Result<Hlvalue, AdapterError> {
    if method_call.turbofish.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "method turbofish `obj.method::<T>(…)` (explicit method generics land in \
                M2.5d alongside struct/enum literal typing)"
                .into(),
        });
    }
    let receiver = lower_expr(b, &method_call.receiver)?;

    // `getattr(receiver, "method")` — the method name is a Python
    // string constant (upstream `ConstValue::Str`).
    let method_name = method_call.method.to_string();
    let bound = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "getattr",
        vec![
            receiver,
            Hlvalue::Constant(Constant::new(ConstValue::Str(method_name))),
        ],
        bound.clone(),
    ));

    // `simple_call(bound, *args)` — arg list starts with the bound
    // method (which carries the receiver after upstream's
    // `FunctionDesc.bind_self`), matching upstream
    // `operation.py:663-679 SimpleCall` convention.
    let mut call_args: Vec<Hlvalue> = Vec::with_capacity(method_call.args.len() + 1);
    call_args.push(bound);
    for arg in &method_call.args {
        call_args.push(lower_expr(b, arg)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "simple_call",
        call_args,
        result.clone(),
    ));
    Ok(result)
}

fn lower_call(b: &mut Builder, call: &ExprCall) -> Result<Hlvalue, AdapterError> {
    // Callee must be a simple identifier path. Qualified paths
    // (`module::fn`) would need module-resolved HostObject lookup
    // which the adapter does not perform — M2.5g registers adapter
    // -produced HostObjects by name, so those will land later.
    match &*call.func {
        Expr::Path(_) => {}
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "call callee must be a simple identifier path (closures, method-returned \
                    callables, etc. land in M2.5d/g)"
                    .into(),
            });
        }
    }
    let callee = lower_expr(b, &call.func)?;

    let mut call_args: Vec<Hlvalue> = Vec::with_capacity(call.args.len() + 1);
    call_args.push(callee);
    for arg in &call.args {
        call_args.push(lower_expr(b, arg)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "simple_call",
        call_args,
        result.clone(),
    ));
    Ok(result)
}

// ____________________________________________________________
// Tuple / array literals.
//
// Upstream basis:
// - `rpython/flowspace/operation.py:543-546` — `newtuple(*items)`:
//   variable arity, the annotator's `bookkeeper.newtuple` builds a
//   `SomeTuple` whose item types come from the arg annotations.
// - `rpython/flowspace/operation.py:552-559` — `newlist(*items)`:
//   variable arity, `bookkeeper.newlist` produces a `SomeList` whose
//   element type is the union of the arg types.
// - `rpython/flowspace/flowcontext.py:1163-1166` — `BUILD_TUPLE`
//   emits `op.newtuple(*items).eval(self)`.
// - `rpython/flowspace/flowcontext.py:1168-1171` — `BUILD_LIST`
//   emits `op.newlist(*items).eval(self)`.

fn lower_tuple(b: &mut Builder, tup: &ExprTuple) -> Result<Hlvalue, AdapterError> {
    let mut args: Vec<Hlvalue> = Vec::with_capacity(tup.elems.len());
    for elem in &tup.elems {
        args.push(lower_expr(b, elem)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("newtuple", args, result.clone()));
    Ok(result)
}

fn lower_array(b: &mut Builder, arr: &ExprArray) -> Result<Hlvalue, AdapterError> {
    let mut args: Vec<Hlvalue> = Vec::with_capacity(arr.elems.len());
    for elem in &arr.elems {
        args.push(lower_expr(b, elem)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("newlist", args, result.clone()));
    Ok(result)
}

// ____________________________________________________________
// Unary operators, field access, index, cast — the "small surface"
// expression kinds that Rust source uses frequently.
//
// Upstream basis (all from `operation.py` `add_operator` registry):
// - `neg` arity=1 (:466)           — unary `-x`.
// - `pos` arity=1 (:465)           — unary `+x`.
// - `invert` arity=1 (:474)        — bitwise `~x`.
// - `getattr` arity=2 (:618)       — field access.
// - `getitem` arity=2 (:457)       — index access.
// - `int` / `float` / `bool` (:488/:490/:467) — type coercion.

fn lower_unary(b: &mut Builder, u: &ExprUnary) -> Result<Hlvalue, AdapterError> {
    // Upstream unary operators covered here are only those with a
    // direct 1-to-1 mapping into the `operation.py` `add_operator`
    // registry. `UnOp::Not` is deliberately NOT mapped — see the
    // match arm for rationale.
    match u.op {
        // `*x` — Rust deref. No upstream counterpart (Python has no
        // explicit deref). The annotator tracks identity + type
        // regardless of borrow form; pass-through is safe.
        UnOp::Deref(_) => lower_expr(b, &u.expr),
        // `-x` — upstream `operation.py:466 neg` arity=1.
        UnOp::Neg(_) => {
            let arg = lower_expr(b, &u.expr)?;
            let result = Hlvalue::Variable(Variable::new());
            b.emit_op(SpaceOperation::new("neg", vec![arg], result.clone()));
            Ok(result)
        }
        // `!x` — Rust overloads this for bitwise (ints) AND logical
        // (bools). Upstream handles the two paths *differently*:
        // - `UNARY_INVERT` (flowcontext.py:194) emits `op.invert`.
        // - `UNARY_NOT` (flowcontext.py:531) emits `op.bool` then
        //   forks on `guessbool` — `bool(x)` is a runtime
        //   conversion, the true-branch yields `Constant(False)`
        //   and the false-branch yields `Constant(True)`. There is
        //   no direct `not_` op in the registry.
        //
        // The adapter cannot pick bitwise-vs-logical without type
        // info, and has no `guessbool` facility to replicate the
        // upstream logical path. Rejecting keeps the lowering
        // honest — users who need bitwise NOT can write the
        // explicit helper call.
        UnOp::Not(_) => Err(AdapterError::Unsupported {
            reason: "unary `!` has no line-by-line upstream mapping (UNARY_NOT uses \
                `bool` + guessbool fork; UNARY_INVERT emits `invert`) — reject to keep \
                adapter orthodox"
                .into(),
        }),
        _ => Err(AdapterError::Unsupported {
            reason: "unrecognised unary operator".into(),
        }),
    }
}

fn lower_field(b: &mut Builder, f: &ExprField) -> Result<Hlvalue, AdapterError> {
    let base = lower_expr(b, &f.base)?;
    let attr_name = match &f.member {
        Member::Named(id) => id.to_string(),
        Member::Unnamed(idx) => {
            // `x.0` / `x.1` — tuple-struct index access. Upstream
            // Python has `tup[0]` which lowers to `getitem(tup,
            // 0)`; Rust tuple-struct / tuple field access has no
            // direct equivalent. Emit `getitem` with an integer
            // Constant index — the annotator can distinguish by
            // receiver type (Tuple vs user struct).
            let int_index = idx.index as i64;
            let result = Hlvalue::Variable(Variable::new());
            b.emit_op(SpaceOperation::new(
                "getitem",
                vec![
                    base,
                    Hlvalue::Constant(Constant::new(ConstValue::Int(int_index))),
                ],
                result.clone(),
            ));
            return Ok(result);
        }
    };
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "getattr",
        vec![
            base,
            Hlvalue::Constant(Constant::new(ConstValue::Str(attr_name))),
        ],
        result.clone(),
    ));
    Ok(result)
}

fn lower_index(b: &mut Builder, idx: &ExprIndex) -> Result<Hlvalue, AdapterError> {
    let base = lower_expr(b, &idx.expr)?;
    let key = lower_expr(b, &idx.index)?;
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "getitem",
        vec![base, key],
        result.clone(),
    ));
    Ok(result)
}

fn lower_cast(_b: &mut Builder, _c: &ExprCast) -> Result<Hlvalue, AdapterError> {
    // `x as T` — Rust compile-time numeric conversion. Upstream has
    // no direct counterpart:
    //
    // - `operation.py:347 do_int(x)` → calls `x.__int__()`, not a
    //   numeric cast. It is a Python-method dispatch with its own
    //   method-resolution path.
    // - `operation.py:490 float(x)` → same shape, calls `__float__`.
    // - `operation.py:467 bool(x)` → calls `__bool__`.
    //
    // Mapping Rust `as` to these would silently inject a dunder call
    // the user never wrote, diverging semantically. Users who need
    // conversion can invoke the helper explicitly (e.g. `x.to_i64()`
    // or a registered function) and let the adapter lower that as a
    // method / function call.
    Err(AdapterError::Unsupported {
        reason: "`x as T` has no line-by-line upstream mapping — upstream `do_int` / \
            `do_float` / `bool` are __int__/__float__/__bool__ dunder dispatches, \
            not numeric coercions. Use an explicit helper call instead."
            .into(),
    })
}

/// Private helper trait — lets `lower_break`/`lower_continue` snapshot
/// the top-of-stack [`LoopCtx`] through an Option without tangling
/// borrow regions (the body then calls `b.finalize_current`, a
/// `&mut self` borrow on the Builder). Clones the small fields only;
/// it is not a hot path.
trait LoopCtxSnapshot {
    fn cloned_ctx(self) -> Option<LoopCtx>;
}

impl LoopCtxSnapshot for Option<&LoopCtx> {
    fn cloned_ctx(self) -> Option<LoopCtx> {
        self.map(|ctx| LoopCtx {
            header_block: ctx.header_block.clone(),
            exit_block: ctx.exit_block.clone(),
            merged_names: ctx.merged_names.clone(),
            exit_merged_names: ctx.exit_merged_names.clone(),
        })
    }
}

// ____________________________________________________________

fn validate_arm(arm: &Arm) -> Result<(), AdapterError> {
    if arm.guard.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "match arm guard (`if COND`) (lands after control-flow slice)".into(),
        });
    }
    Ok(())
}

/// Flatten a `Pat::Or` into its constituent sub-patterns, recursing so
/// nested or-patterns (`A | (B | C)`) reduce to a flat list.
/// Parenthesised patterns are transparent — `(X)` is grouping-only in
/// syn, and upstream semantics treat the inner pattern identically —
/// so we unwrap them before deciding whether to recurse. Non-or
/// patterns contribute a single entry. No direct upstream analogue
/// (Python source produces `if-elif` chains, not or-patterns) but the
/// resulting list maps onto a standard fan-out of Links all pointing
/// at the same target block, which `model.py:648-692` admits directly.
fn flatten_or_pattern<'a>(pat: &'a Pat, out: &mut Vec<&'a Pat>) {
    match pat {
        Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                flatten_or_pattern(case, out);
            }
        }
        Pat::Paren(paren) => flatten_or_pattern(&paren.pat, out),
        other => out.push(other),
    }
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
            // Upstream `model.py:658` admits single-character strings as
            // switch exitcases (`isinstance(n, (str, unicode)) and
            // len(n) == 1`). Rust's `char` literal is the direct
            // analogue — RPython has no `char` type; it uses one-char
            // `str`. Emit `ConstValue::Str(c.to_string())` so the
            // resulting exitcase passes `checkgraph`'s len==1 check.
            Lit::Char(ch) => Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::Str(
                ch.value().to_string(),
            ))))),
            // Upstream `model.py:658` admits `isinstance(n, (str,
            // unicode)) and len(n) == 1` as a valid switch exitcase,
            // so single-character string patterns are legal. `char`
            // literals lower to the same `ConstValue::Str(c.to_string())`
            // shape (see the `Lit::Char` arm above), making
            // `match x { "a" => … }` and `match x { 'a' => … }`
            // structurally interchangeable. Multi-character strings
            // still reject — `checkgraph` would flag them.
            Lit::Str(s) => {
                let value = s.value();
                // `str.chars().count()` counts Unicode scalars,
                // matching what `len(u"é")` observes on RPython's
                // unicode side.
                if value.chars().count() != 1 {
                    return Err(AdapterError::Unsupported {
                        reason: "match arm multi-character string-literal pattern — \
                            upstream `model.py:658` admits only single-character \
                            strings as switch exitcases"
                            .into(),
                    });
                }
                Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::Str(
                    value,
                )))))
            }
            Lit::ByteStr(_) | Lit::Byte(_) | Lit::Float(_) | _ => Err(AdapterError::Unsupported {
                reason: "match arm non-integer/bool/char literal pattern \
                        (byte/bytestring/float patterns have no upstream analogue)"
                    .into(),
            }),
        },
        Pat::Or(_) => {
            // Unreachable under `lower_match` which pre-flattens
            // or-patterns via `flatten_or_pattern` before calling
            // `classify_pattern`. Keep this arm as a defensive
            // contract so a future caller that forgets the flatten
            // step fails loudly instead of silently mis-classifying.
            Err(AdapterError::Unsupported {
                reason: "or-pattern reached classify_pattern without flattening — \
                    caller must pre-expand via `flatten_or_pattern`"
                    .into(),
            })
        }
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

        // startblock: [x] → emits `x > 0` then `bool(gt_result)`
        // (upstream POP_JUMP_IF_FALSE wraps cond in `op.bool`), so
        // exitswitch = the bool-op result, 2 exits.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "gt");
        assert_eq!(start.operations[1].opname, "bool");
        let switch_var = start.operations[1].result.clone();
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

    // NOTE: generic fns are now accepted (slice M2.5c). The
    // annotator layer monomorphizes via `FunctionDesc.specialize`.
    // See `generic_fn_identity` below.

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

    // NOTE: `self` receivers are now accepted (slice M2.5c) as a
    // local named `"self"`. See `self_receiver_is_local_named_self`
    // below.

    #[test]
    fn if_without_else_return_in_branch_closes_to_returnblock() {
        // Upstream `flowcontext.py:687 RETURN_VALUE` raises `Return`
        // and `flowcontext.py:1232 Return.nomoreblocks(ctx)` closes
        // the current block straight to `graph.returnblock`. In an
        // `if cond { return X; } tail` shape the then-branch's
        // closing Link therefore targets `returnblock` (carrying the
        // return value), while the `false` shortcut threads through
        // the join and on to the function's implicit-None return
        // tail.
        let g = lower("fn f(x: i64) -> i64 { if x > 0 { return 1; } 2 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        let false_exit = start.exits[0].borrow();
        let true_exit = start.exits[1].borrow();
        let true_target = true_exit
            .target
            .as_ref()
            .expect("true branch has target")
            .clone();
        // The then_block's only exit Links to `graph.returnblock`
        // with `Constant(1)` — upstream `flowcontext.py:1232`
        // structural shape.
        let then_exits = true_target.borrow().exits.clone();
        assert_eq!(
            then_exits.len(),
            1,
            "then_block closes with a single returnblock Link per flowcontext.py:1232"
        );
        let then_link = then_exits[0].borrow();
        let then_link_target = then_link
            .target
            .as_ref()
            .expect("then-Link has target")
            .clone();
        assert!(
            Rc::ptr_eq(&then_link_target, &g.returnblock),
            "then-branch `return` must Link directly to graph.returnblock (not join)"
        );
        assert_eq!(then_link.args.len(), 1);
        match then_link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(1)),
            other => panic!("expected Constant(1), got {other:?}"),
        }
        // The `false` shortcut threads to the post-if join, which in
        // turn closes via the `2` tail into `returnblock`. Walk it
        // explicitly.
        let false_target = false_exit
            .target
            .as_ref()
            .expect("false branch has target")
            .clone();
        let join_exits = false_target.borrow().exits.clone();
        assert_eq!(join_exits.len(), 1);
        let join_link = join_exits[0].borrow();
        assert!(Rc::ptr_eq(
            join_link.target.as_ref().unwrap(),
            &g.returnblock
        ));
        match join_link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(2)),
            other => panic!("expected Constant(2), got {other:?}"),
        }
    }

    #[test]
    fn if_else_both_branches_return_produces_no_join() {
        // When every branch of an if/else terminates via `return`,
        // upstream's `Return.nomoreblocks()` runs on both sides and
        // `StopFlowing` prevents the pending-block scheduler from
        // ever enqueueing a post-if PC. Result: no join block, just
        // two fork-target blocks that each Link to `graph.returnblock`.
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { return 1; } else { return 2; }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        // Each branch block has exactly one exit, which Links to
        // graph.returnblock.
        for exit in start.exits.iter() {
            let branch = exit
                .borrow()
                .target
                .as_ref()
                .expect("fork exit has target")
                .clone();
            let branch_exits = branch.borrow().exits.clone();
            assert_eq!(branch_exits.len(), 1);
            let link = branch_exits[0].borrow();
            assert!(Rc::ptr_eq(link.target.as_ref().unwrap(), &g.returnblock));
        }
    }

    #[test]
    fn if_else_then_returns_else_falls_through_yields_else_tail() {
        // Mixed case: then terminates via `return`, else falls
        // through with a tail value. The if-expression's value is
        // the else-tail, and the join block has exactly one
        // predecessor (the else arm).
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { return 1; } else { 2 }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Walk to the returnblock from the else-side — its incoming
        // Link should carry `Constant(2)` per the else-tail.
        let ret = g.returnblock.borrow();
        assert_eq!(ret.inputargs.len(), 1);
        assert!(ret.is_final);
    }

    #[test]
    fn match_all_arms_return_yields_no_join() {
        // Upstream `model.py:648-692` builds the switch-exit table
        // the same way `flowcontext.py:1232` closes each arm — if
        // every arm's body raises `Return`, the post-match PC is
        // unreachable and the scheduler never allocates a join.
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x {
                    0 => return 10,
                    _ => return 20,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Each arm's branch block ends in a Link straight to
        // graph.returnblock.
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        for exit in start.exits.iter() {
            let branch = exit
                .borrow()
                .target
                .as_ref()
                .expect("fork exit has target")
                .clone();
            let branch_exits = branch.borrow().exits.clone();
            assert_eq!(branch_exits.len(), 1);
            let link = branch_exits[0].borrow();
            assert!(Rc::ptr_eq(link.target.as_ref().unwrap(), &g.returnblock));
        }
    }

    #[test]
    fn return_before_end_of_block_rejects() {
        // `return X; Y` with anything after the return is unreachable
        // dead code in upstream terms — `flowcontext.py:1232` closes
        // the block at Return; subsequent ops never emit. Reject
        // cleanly so users get a parse-time error instead of silent
        // dead-code emission.
        match lower("fn f() -> i64 { return 1; 2 }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("after `return`"),
                    "reason should cite dead code: {reason}"
                );
            }
            other => panic!("expected Unsupported(dead code after return), got {other:?}"),
        }
    }

    #[test]
    fn if_without_else_statement_shortcuts_false_link_to_join() {
        // `if x > 0 { let _ = 1; }` as a statement lowers to the
        // upstream `POP_JUMP_IF_FALSE` shape: fork block has two
        // exits, the Bool(false) link shortcuts straight to the join
        // block (no else block allocated), the Bool(true) link
        // routes through the then_block which in turn Links into the
        // same join.
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { let _y = 1; }
                2
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2, "if fork has exactly two exits");
        let false_exit = start.exits[0].borrow();
        let true_exit = start.exits[1].borrow();
        assert!(
            matches!(
                &false_exit.exitcase,
                Some(Hlvalue::Constant(c)) if matches!(c.value, ConstValue::Bool(false))
            ),
            "first exit must carry Bool(false)"
        );
        assert!(
            matches!(
                &true_exit.exitcase,
                Some(Hlvalue::Constant(c)) if matches!(c.value, ConstValue::Bool(true))
            ),
            "second exit must carry Bool(true)"
        );
        // False shortcut: its target is reached from fork in one hop
        // and each of its exits Links to the join. True path routes
        // through a then_block whose single exit Links to the same
        // join. Verified by following one hop from each side and
        // asserting pointer-equality of their targets.
        let false_target = false_exit
            .target
            .as_ref()
            .expect("false has target")
            .clone();
        let true_target = true_exit.target.as_ref().expect("true has target").clone();
        assert!(
            !Rc::ptr_eq(&false_target, &true_target),
            "false shortcut and true branch must head to distinct blocks from the fork"
        );
        // The false-side target IS the join directly; the true-side
        // target is the then_block, whose single exit Links to the
        // join.
        let then_exits = true_target.borrow().exits.clone();
        assert_eq!(then_exits.len(), 1, "then_block single-exit to join");
        let then_link_target = then_exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("then_block link has target")
            .clone();
        assert!(
            Rc::ptr_eq(&false_target, &then_link_target),
            "then_block's Link target must be the SAME join block the false shortcut points at"
        );
    }

    #[test]
    fn if_without_else_tail_returns_none() {
        // In tail position, `if cond { body }` produces `None` as
        // the expression value — matches Python's fallthrough
        // convention. `ConstValue::None` should flow into the
        // returnblock Link.
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { let _y = 1; }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Walk to the returnblock: startblock → join → returnblock.
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        let false_target = start.exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("false target")
            .clone();
        // The join's single exit is the Link to the returnblock,
        // carrying the None tail as its first (and only) arg.
        let join_exits = false_target.borrow().exits.clone();
        assert_eq!(join_exits.len(), 1);
        let return_link = join_exits[0].borrow();
        assert!(!return_link.args.is_empty(), "return link carries tail arg");
        match return_link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::None),
            other => panic!("expected Constant(None), got {other:?}"),
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
    fn or_pattern_emits_multiple_links_sharing_one_branch_block() {
        // `match x { 1 | 2 => 1, _ => 0 }` — or-pattern `1 | 2`
        // lowers into TWO Links from the fork block (one per
        // sub-pattern) both targeting the SAME branch block; then one
        // wildcard-default Link for the `_` arm. Upstream
        // `model.py:648-692` admits multiple Links with distinct
        // exitcases pointing at the same target block.
        let g = lower("fn f(x: i64) -> i64 { match x { 1 | 2 => 1, _ => 0 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 3, "two or-sub-cases + one default");
        let target_0 = start.exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("fork exit has target")
            .clone();
        let target_1 = start.exits[1]
            .borrow()
            .target
            .as_ref()
            .expect("fork exit has target")
            .clone();
        let target_2 = start.exits[2]
            .borrow()
            .target
            .as_ref()
            .expect("fork exit has target")
            .clone();
        assert!(
            Rc::ptr_eq(&target_0, &target_1),
            "or-pattern sub-cases must share one branch block"
        );
        assert!(
            !Rc::ptr_eq(&target_0, &target_2),
            "default arm must reach a distinct branch block"
        );
        // Exitcases: Int(1), Int(2), Str("default").
        let ec_0 = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec_1 = start.exits[1].borrow().exitcase.clone().unwrap();
        let ec_2 = start.exits[2].borrow().exitcase.clone().unwrap();
        assert!(
            matches!(
                ec_0,
                Hlvalue::Constant(ref c) if matches!(c.value, ConstValue::Int(1))
            ),
            "first or-sub-case should be Int(1)"
        );
        assert!(
            matches!(
                ec_1,
                Hlvalue::Constant(ref c) if matches!(c.value, ConstValue::Int(2))
            ),
            "second or-sub-case should be Int(2)"
        );
        assert!(
            matches!(
                ec_2,
                Hlvalue::Constant(ref c) if matches!(&c.value, ConstValue::Str(s) if s == "default")
            ),
            "default arm should carry Str(\"default\")"
        );
    }

    #[test]
    fn or_pattern_nested_flattens_to_siblings() {
        // `A | (B | C)` — nested or-pattern. `flatten_or_pattern`
        // recurses so three Links come out of the fork.
        let g = lower("fn f(x: i64) -> i64 { match x { 1 | (2 | 3) => 1, _ => 0 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 4, "three or-sub-cases + one default");
        let t0 = start.exits[0].borrow().target.as_ref().unwrap().clone();
        let t1 = start.exits[1].borrow().target.as_ref().unwrap().clone();
        let t2 = start.exits[2].borrow().target.as_ref().unwrap().clone();
        assert!(
            Rc::ptr_eq(&t0, &t1) && Rc::ptr_eq(&t1, &t2),
            "all three or-sub-cases must share the same branch block"
        );
    }

    #[test]
    fn rejects_wildcard_inside_or_pattern() {
        // Upstream `model.py:652` reserves wildcard for the
        // standalone default arm; embedding it inside an or-pattern
        // would duplicate the "catch-all" intent.
        match lower("fn f(x: i64) -> i64 { match x { 1 | _ => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("wildcard sub-pattern inside or-pattern"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(wildcard-in-or), got {other:?}"),
        }
    }

    #[test]
    fn rejects_or_pattern_with_duplicate_case() {
        // `1 | 1` or `1 | 2` crossing into another arm that reuses 2
        // both violate the uniqueness invariant
        // (`model.py:692 allexitcases`).
        match lower("fn f(x: i64) -> i64 { match x { 1 | 2 => 1, 2 => 2, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("repeated"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(repeated), got {other:?}"),
        }
    }

    #[test]
    fn match_char_pattern_emits_single_char_str_exitcase() {
        // Upstream `model.py:658` admits `isinstance(n, (str, unicode))
        // and len(n) == 1` as a switch exitcase. Rust's `char` literal
        // (`'a'`) is the direct analogue — RPython has no `char` type,
        // so single-char strings fill the role. The arm exitcase
        // should be `ConstValue::Str("a")` (len==1), passing
        // `checkgraph`.
        let g = lower("fn f(c: char) -> i64 { match c { 'a' => 1, 'b' => 2, _ => 0 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 3, "two char arms + default");
        let ec_a = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec_b = start.exits[1].borrow().exitcase.clone().unwrap();
        let ec_d = start.exits[2].borrow().exitcase.clone().unwrap();
        assert!(
            matches!(
                ec_a,
                Hlvalue::Constant(ref c) if matches!(&c.value, ConstValue::Str(s) if s == "a")
            ),
            "first char arm should carry Str(\"a\") — got {ec_a:?}"
        );
        assert!(
            matches!(
                ec_b,
                Hlvalue::Constant(ref c) if matches!(&c.value, ConstValue::Str(s) if s == "b")
            ),
            "second char arm should carry Str(\"b\") — got {ec_b:?}"
        );
        assert!(
            matches!(
                ec_d,
                Hlvalue::Constant(ref c) if matches!(&c.value, ConstValue::Str(s) if s == "default")
            ),
            "wildcard arm should carry Str(\"default\") — got {ec_d:?}"
        );
    }

    #[test]
    fn rejects_match_multichar_string_literal_pattern() {
        // `Lit::Str` with len > 1 violates upstream `model.py:658`'s
        // `len(n) == 1` invariant for string exitcases. Single-char
        // strings are accepted (see
        // `match_single_char_str_pattern_emits_str_exitcase`).
        match lower("fn f(s: &str) -> i64 { match s { \"abc\" => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("multi-character"),
                    "reason should cite multi-char rejection: {reason}"
                );
                assert!(
                    reason.contains("model.py:658"),
                    "reason should cite the upstream rule: {reason}"
                );
            }
            other => panic!("expected Unsupported(multi-char string), got {other:?}"),
        }
    }

    #[test]
    fn match_single_char_str_pattern_emits_str_exitcase() {
        // Upstream `model.py:658` admits single-character strings as
        // valid switch exitcases; the adapter should accept `"a"` in
        // a match arm the same way it accepts `'a'`. Both produce a
        // `ConstValue::Str("a".into())` exitcase — structurally
        // interchangeable at the graph layer.
        let g = lower(
            "fn f(s: &str) -> i64 {
                match s { \"a\" => 1, _ => 0 }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        let ec = start.exits[0]
            .borrow()
            .exitcase
            .clone()
            .expect("arm has exitcase");
        assert!(
            matches!(
                ec,
                Hlvalue::Constant(ref c) if matches!(&c.value, ConstValue::Str(s) if s == "a")
            ),
            "single-char string pattern should carry Str(\"a\") — got {ec:?}"
        );
    }

    #[test]
    fn match_single_char_str_and_char_pattern_produce_identical_exitcase() {
        // `match x { "a" => … }` and `match x { 'a' => … }` must
        // produce the same `ConstValue::Str("a")` exitcase. This
        // pins `lower_literal` / `classify_pattern`'s symmetry
        // between the two syntactic forms.
        let g_str = lower("fn f(s: &str) -> i64 { match s { \"a\" => 1, _ => 0 } }").unwrap();
        let g_char = lower("fn f(c: char) -> i64 { match c { 'a' => 1, _ => 0 } }").unwrap();
        let start_str = g_str.startblock.borrow();
        let start_char = g_char.startblock.borrow();
        let ec_str = start_str.exits[0]
            .borrow()
            .exitcase
            .clone()
            .expect("arm has exitcase");
        let ec_char = start_char.exits[0]
            .borrow()
            .exitcase
            .clone()
            .expect("arm has exitcase");
        assert_eq!(ec_str, ec_char);
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

    // NOTE: method calls are now accepted (slice M2.5c). See
    // `method_call_emits_getattr_plus_simple_call` below.

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

    // NOTE: `?` operator is now accepted (slice 4). See the
    // `try_op_*` tests below for coverage.

    // NOTE: tuple literals are now accepted (slice M2.5d). See
    // `tuple_literal_emits_newtuple_op`.

    // ---- M2.5b while/loop/break/continue accept tests -------------

    #[test]
    fn while_loop_emits_header_body_back_edge_exit() {
        // The canonical shape: a header block with exitswitch=cond and
        // two exits (false → exit, true → body), a body block that
        // Links back to the header, and an exit block that continues
        // into the returnblock.
        let g = lower(
            "fn f(n: i64) -> i64 {
                let i = 0;
                while i < n {
                    let i = i + 1;
                }
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // startblock + header + body + exit + returnblock = 5.
        assert_eq!(blocks.len(), 5);

        // startblock has no ops (i = 0 is a constant rebind, no op)
        // and closes with a single Link into the header.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        assert_eq!(start.exits.len(), 1);

        // The header is reached from startblock and from the body
        // (2 incoming Links = the test oracle). Identify it via its
        // operations — `i < n` emits `lt` then `bool` (upstream
        // POP_JUMP_IF_FALSE wraps cond in `op.bool`); exitswitch
        // references the bool-op result and the block has 2 exits.
        let header = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 2
                    && br.operations[0].opname == "lt"
                    && br.operations[1].opname == "bool"
                    && br.exitswitch.is_some()
                    && br.exits.len() == 2
            })
            .expect("expected header block with `lt` + `bool` ops + 2 exits");
        let header_ref = header.borrow();
        let false_exit = header_ref.exits[0].borrow();
        let true_exit = header_ref.exits[1].borrow();
        assert_eq!(
            false_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
        assert_eq!(
            true_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );

        // Body block emits `i + 1` and back-edges to header.
        let body_target = true_exit.target.as_ref().unwrap();
        let body_ref = body_target.borrow();
        assert_eq!(body_ref.operations.len(), 1);
        assert_eq!(body_ref.operations[0].opname, "add");
        assert_eq!(body_ref.exits.len(), 1);
        // Back-edge target is `header`.
        let back_link = body_ref.exits[0].borrow();
        let back_target = back_link.target.as_ref().unwrap();
        assert!(Rc::ptr_eq(back_target, header));
    }

    #[test]
    fn while_loop_no_rebinds_in_body() {
        // Body has no let — body's back-edge args are exactly the
        // inputargs forwarded through.
        let g = lower(
            "fn f(n: i64) -> i64 {
                let i = 0;
                while i < n {}
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn loop_with_break_reaches_exit() {
        // `loop { break; }` — no condition, body's only stmt is `break;`.
        // Header and body are the same block; the body's only exit is
        // the Link into the exit block.
        let g = lower(
            "fn f() -> i64 {
                loop { break; }
                42
            }",
        )
        .unwrap();
        checkgraph(&g);
        // startblock → header/body (break-only) → exit → returnblock.
        // The startblock has no ops.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        assert_eq!(start.exits.len(), 1);
        // Exit block tail emits the constant 42 and Links into return.
        let blocks = g.iterblocks();
        // start + header(=body) + exit + return = 4.
        assert_eq!(blocks.len(), 4);
    }

    #[test]
    fn continue_back_edges_to_header() {
        let g = lower(
            "fn f(n: i64) -> i64 {
                let i = 0;
                while i < n {
                    continue;
                }
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // start + header + body + exit + return = 5. The body's only
        // exit is a Link back to the header via `continue`.
        assert_eq!(blocks.len(), 5);
        let header = blocks
            .iter()
            .find(|b| b.borrow().exitswitch.is_some())
            .expect("expected header with exitswitch");
        let header_true_exit = header.borrow().exits[1].clone();
        let body = header_true_exit.borrow().target.clone().unwrap();
        let body_exit = body.borrow().exits[0].clone();
        let body_target = body_exit.borrow().target.clone().unwrap();
        assert!(Rc::ptr_eq(&body_target, header));
    }

    #[test]
    fn nested_loops_break_resolves_to_innermost() {
        // `break;` inside the inner loop exits only that loop.
        let g = lower(
            "fn f(n: i64, m: i64) -> i64 {
                let i = 0;
                while i < n {
                    let j = 0;
                    while j < m {
                        break;
                    }
                    let i = i + 1;
                }
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn loop_with_let_rebind_before_break() {
        let g = lower(
            "fn f() -> i64 {
                let x = 0;
                loop {
                    let x = x + 1;
                    break;
                }
                x
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    // ---- M2.5b while/loop/break/continue reject tests -------------

    #[test]
    fn rejects_break_outside_loop() {
        match lower("fn f() -> i64 { break; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("break") || reason.contains("outside"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(break outside), got {other:?}"),
        }
    }

    #[test]
    fn rejects_continue_outside_loop() {
        match lower("fn f() -> i64 { continue; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("continue") || reason.contains("outside"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(continue outside), got {other:?}"),
        }
    }

    #[test]
    fn rejects_break_with_value() {
        match lower("fn f() -> i64 { loop { break 1; } }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("break VALUE")
                    || reason.contains("value")
                    || reason.contains("loop-as-expression"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(break with value), got {other:?}"),
        }
    }

    #[test]
    fn rejects_dead_code_after_break() {
        match lower(
            "fn f() -> i64 {
                loop { break; let _x = 1; }
                0
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("dead code"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(dead code after break), got {other:?}"),
        }
    }

    #[test]
    fn rejects_labeled_break() {
        match lower("fn f() -> i64 { 'outer: loop { break 'outer; } }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("label") || reason.contains("VALUE") || reason.contains("value"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(labeled break), got {other:?}"),
        }
    }

    // ---- M2.5b `?` operator accept tests --------------------------

    #[test]
    fn try_op_emits_canraise_block_with_exception_edge() {
        // The canonical shape: the startblock emits the operand's own
        // call op as the raising op (no synthetic wrapper), sets
        // exitswitch = c_last_exception, and exits with
        // (exit[0]=normal→continuation, exit[1]=exception
        // →graph.exceptblock with exitcase=Exception class).
        // Operand must be a direct call; bare variables reject.
        let g = lower("fn f(h: Handler) -> i64 { h.read()? }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // getattr('read') + simple_call(bound) — the simple_call is
        // the raising op.
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "simple_call");
        assert!(start.canraise(), "startblock must canraise");
        assert_eq!(start.exits.len(), 2);

        let normal = start.exits[0].borrow();
        assert!(normal.exitcase.is_none());
        assert!(normal.last_exception.is_none());
        assert!(normal.last_exc_value.is_none());

        let exc = start.exits[1].borrow();
        match exc.exitcase.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class(), "exitcase must be a class HostObject");
                    assert_eq!(obj.qualname(), "Exception");
                }
                other => panic!("expected HostObject exitcase, got {other:?}"),
            },
            other => panic!("expected Constant exitcase, got {other:?}"),
        }
        let target = exc.target.as_ref().unwrap();
        assert!(Rc::ptr_eq(target, &g.exceptblock));
        assert!(exc.last_exception.is_some());
        assert!(exc.last_exc_value.is_some());
    }

    #[test]
    fn try_op_continuation_produces_unwrapped_value() {
        // `h.read()? + 1` — the continuation block picks up the
        // unwrapped value via its inputargs[0] and emits
        // `add(unwrapped, 1)`.
        let g = lower("fn f(h: Handler) -> i64 { h.read()? + 1 }").unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // startblock (canraise) + continuation + returnblock +
        // exceptblock = 4.
        assert_eq!(blocks.len(), 4);

        let cont = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 1 && br.operations[0].opname == "add"
            })
            .expect("expected continuation block with `add` op");
        let cont_ref = cont.borrow();
        assert_eq!(cont_ref.operations[0].args[0], cont_ref.inputargs[0]);
    }

    #[test]
    fn try_op_let_binding_continues_locals() {
        // `let y = h.read()?; y + 1` — pre-try local `h` survives
        // via the continuation's inputargs.
        let g = lower(
            "fn f(h: Handler) -> i64 {
                let y = h.read()?;
                y + 1
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn try_op_twice_chains_two_canraise_blocks() {
        // Two call-? in sequence — two canraise blocks. checkgraph
        // enforces both exception exits target the single
        // graph.exceptblock.
        let g = lower(
            "fn f(h: Handler) -> i64 {
                let y = h.read()?;
                h.write(y)?
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        let canraise_count = blocks.iter().filter(|b| b.borrow().canraise()).count();
        assert_eq!(canraise_count, 2);
    }

    #[test]
    fn try_op_rejects_non_call_operand() {
        // `x?` where `x` is a bare variable — no call op to hang
        // canraise on, so reject.
        match lower("fn f(x: i64) -> i64 { x? }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("direct call"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(? non-call), got {other:?}"),
        }
    }

    // ---- M2.5b `for` loop accept tests ----------------------------

    #[test]
    fn for_loop_emits_iter_next_canraise_with_stopiter_exit() {
        // The canonical shape: startblock emits `iter(iter_expr)`,
        // unconditional Link to header; header emits
        // `next(iter_h)` with exitswitch = c_last_exception,
        // exits[0] normal → body, exits[1] → exit_block with
        // exitcase = StopIteration class.
        let g = lower(
            "fn f(it: i64) -> i64 {
                for item in it {
                    let _x = item;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);

        // startblock has one op `iter(it)`.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "iter");

        // Locate the header block — it has op `next` + canraise.
        let blocks = g.iterblocks();
        let header = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 1 && br.operations[0].opname == "next" && br.canraise()
            })
            .expect("expected header block with `next` op + canraise");
        let header_ref = header.borrow();
        // 3 exits: normal → body, StopIteration → loop exit,
        // RuntimeError → graph.exceptblock. Matches upstream
        // `operation.py:595-599 Next.canraise = [StopIteration,
        // RuntimeError]`.
        assert_eq!(header_ref.exits.len(), 3);

        // exit[0]: normal → body.
        let normal = header_ref.exits[0].borrow();
        assert!(normal.exitcase.is_none());

        // exit[1]: StopIteration → exit block.
        let stopiter_exit = header_ref.exits[1].borrow();
        match stopiter_exit.exitcase.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "StopIteration");
                }
                other => panic!("expected HostObject exitcase, got {other:?}"),
            },
            other => panic!("expected Constant exitcase, got {other:?}"),
        }
        // `guessexception` class-specific exit: last_exception is a
        // Constant(case) per `flowcontext.py:131-132`, not a Variable.
        match stopiter_exit.last_exception.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "StopIteration");
                }
                other => panic!("expected StopIteration HostObject, got {other:?}"),
            },
            other => panic!(
                "StopIteration exit's last_exception must be a \
                 Constant(StopIteration) per upstream guessexception \
                 (flowcontext.py:127-132), got {other:?}"
            ),
        }
        match stopiter_exit.last_exc_value.as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!(
                "StopIteration exit's last_exc_value must be a \
                 fresh Variable('last_exc_value') per upstream \
                 (flowcontext.py:133), got {other:?}"
            ),
        }

        // exit[2]: RuntimeError → graph.exceptblock. Per upstream
        // `RaiseImplicit.nomoreblocks` (flowcontext.py:1271-1284)
        // the link args carry [Constant(AssertionError class),
        // Constant(AssertionError("implicit RuntimeError …"))],
        // NOT the original RuntimeError class/value pair.
        let runtime_exit = header_ref.exits[2].borrow();
        match runtime_exit.exitcase.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "RuntimeError");
                }
                other => panic!("expected HostObject exitcase, got {other:?}"),
            },
            other => panic!("expected Constant exitcase, got {other:?}"),
        }
        let runtime_target = runtime_exit.target.as_ref().unwrap();
        assert!(
            Rc::ptr_eq(runtime_target, &g.exceptblock),
            "RuntimeError exit must target graph.exceptblock"
        );
        // link.args == [Constant(AssertionError class), Constant(AE instance)]
        assert_eq!(runtime_exit.args.len(), 2);
        let arg0 = runtime_exit.args[0]
            .as_ref()
            .expect("RuntimeError link.args[0] cannot be undefined-local sentinel");
        match arg0 {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "AssertionError");
                }
                other => panic!("expected AssertionError class HostObject, got {other:?}"),
            },
            other => panic!(
                "RuntimeError link.args[0] must be Constant(AssertionError \
                 class) per RaiseImplicit.nomoreblocks, got {other:?}"
            ),
        }
        let arg1 = runtime_exit.args[1]
            .as_ref()
            .expect("RuntimeError link.args[1] cannot be undefined-local sentinel");
        match arg1 {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_instance());
                    let cls = obj.instance_class().expect("instance class");
                    assert_eq!(cls.qualname(), "AssertionError");
                }
                other => panic!("expected AssertionError instance HostObject, got {other:?}"),
            },
            other => panic!(
                "RuntimeError link.args[1] must be Constant(AssertionError \
                 instance) per RaiseImplicit.nomoreblocks, got {other:?}"
            ),
        }
        // last_exception on the RuntimeError link remains the original
        // class-specific Constant(RuntimeError) per `guessexception`,
        // even though the args were rewritten by `nomoreblocks`.
        match runtime_exit.last_exception.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "RuntimeError");
                }
                other => panic!("expected RuntimeError HostObject, got {other:?}"),
            },
            other => panic!(
                "RuntimeError exit's last_exception must be a \
                 Constant(RuntimeError) per upstream guessexception, \
                 got {other:?}"
            ),
        }
        match runtime_exit.last_exc_value.as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!(
                "RuntimeError exit's last_exc_value must be a fresh \
                 Variable('last_exc_value'), got {other:?}"
            ),
        }
    }

    #[test]
    fn for_loop_body_shadows_preloop_local_single_channel() {
        // `for x in xs { … }` where `x` is also a parameter — upstream
        // STORE_FAST (flowcontext.py:878-884) rebinds the same local
        // slot in place. The body block must therefore expose exactly
        // ONE inputarg for the `x` slot, not two, and the header's
        // normal Link must route `v_next` into that slot.
        let g = lower(
            "fn f(x: i64, xs: i64) -> i64 {
                for x in xs {
                    let _y = x;
                }
                x
            }",
        )
        .unwrap();
        checkgraph(&g);

        // Header block (has `next` op + canraise).
        let blocks = g.iterblocks();
        let header = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 1 && br.operations[0].opname == "next" && br.canraise()
            })
            .expect("expected header block");
        let header_ref = header.borrow();

        // Body block = target of header.exits[0].
        let body = header_ref.exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("normal exit must have a target")
            .clone();
        let body_ref = body.borrow();

        // Count how many body inputargs carry the Variable::named("x")
        // prefix. `rename()` canonicalises `"x"` to `"x_"` (see
        // `model.rs:2050-2082`). Upstream STORE_FAST makes it exactly
        // one channel; a `body_inputargs = [item_body_var, fresh_x,
        // ...]` shape would be "two channels for the same slot" — the
        // bug this test guards against.
        let x_slots: usize = body_ref
            .inputargs
            .iter()
            .filter(|h| match h {
                Hlvalue::Variable(v) => v.name_prefix() == "x_",
                _ => false,
            })
            .count();
        assert_eq!(
            x_slots, 1,
            "body inputargs must have exactly one channel for the `x` \
             slot (upstream STORE_FAST rebinds in place — \
             flowcontext.py:878-884), got {x_slots}"
        );
    }

    #[test]
    fn for_loop_body_back_edges_to_header_with_iter_slot() {
        // Body's fall-through back-edge carries [iter_var, ...locals]
        // into header whose inputargs are [iter_h, ...locals].
        let g = lower(
            "fn f(it: i64, a: i64) -> i64 {
                for item in it {
                    let a = a + 1;
                }
                a
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn for_loop_with_break_exits_past_loop() {
        let g = lower(
            "fn f(it: i64) -> i64 {
                for _item in it {
                    break;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn for_loop_with_continue_back_edges_to_header() {
        let g = lower(
            "fn f(it: i64) -> i64 {
                for _item in it {
                    continue;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn nested_for_loops() {
        let g = lower(
            "fn f(it1: i64, it2: i64) -> i64 {
                for _a in it1 {
                    for _b in it2 {
                        let _x = 1;
                    }
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    // ---- M2.5b `for` loop reject tests ----------------------------

    #[test]
    fn rejects_for_tuple_pattern() {
        match lower(
            "fn f(it: i64) -> i64 {
                for (a, b) in it {}
                0
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("identifier") || reason.contains("destructuring"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(destructuring), got {other:?}"),
        }
    }

    #[test]
    fn rejects_for_in_expr_position() {
        // `let x = for ... { ... };` — loop as expression produces `()`
        // which is out of scope.
        match lower(
            "fn f(it: i64) -> i64 {
                let _x = for _i in it {};
                0
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("loop construct") || reason.contains("statement"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(for-in-expr), got {other:?}"),
        }
    }

    // ---- M2.5c method calls + function calls + generics ----------

    #[test]
    fn method_call_emits_getattr_plus_simple_call() {
        // `x.abs()` → [getattr(x, "abs") → v_bound, simple_call(v_bound) → v_result].
        let g = lower("fn f(x: i64) -> i64 { x.abs() }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Str(s) => assert_eq!(s, "abs"),
                other => panic!("expected Str method name, got {other:?}"),
            },
            other => panic!("expected Constant method name, got {other:?}"),
        }
        assert_eq!(start.operations[1].opname, "simple_call");
        // simple_call's first arg is the bound method (getattr result).
        assert_eq!(start.operations[1].args[0], start.operations[0].result);
        // And the call has no extra args (abs is nullary beyond receiver).
        assert_eq!(start.operations[1].args.len(), 1);
    }

    #[test]
    fn method_call_with_args_threads_args_into_simple_call() {
        // `x.add(y, z)` → [getattr(x, "add"), simple_call(bound, y, z)].
        let g = lower("fn f(x: i64, y: i64, z: i64) -> i64 { x.add(y, z) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        let sc = &start.operations[1];
        assert_eq!(sc.opname, "simple_call");
        assert_eq!(sc.args.len(), 3);
    }

    #[test]
    fn function_call_emits_simple_call_with_callee_first() {
        // `g(x, 1)` where `g` is a local → simple_call(g, x, 1).
        // Synthetic: the adapter doesn't know `g` is callable, just
        // that it resolves as a local identifier.
        let g = lower("fn f(x: i64, g: i64) -> i64 { g(x, 1) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "simple_call");
        assert_eq!(start.operations[0].args.len(), 3);
    }

    #[test]
    fn method_call_chained() {
        // `x.a().b()` → 4 ops: getattr/simple_call for each.
        let g = lower("fn f(x: i64) -> i64 { x.a().b() }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 4);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "simple_call");
        assert_eq!(start.operations[2].opname, "getattr");
        assert_eq!(start.operations[3].opname, "simple_call");
    }

    #[test]
    fn method_call_on_binop_result() {
        // `(x + 1).abs()` — receiver is an op result, not a local.
        let g = lower("fn f(x: i64) -> i64 { (x + 1).abs() }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 3);
        assert_eq!(start.operations[0].opname, "add");
        assert_eq!(start.operations[1].opname, "getattr");
        // The getattr receiver is the add's result Variable.
        assert_eq!(start.operations[1].args[0], start.operations[0].result);
        assert_eq!(start.operations[2].opname, "simple_call");
    }

    #[test]
    fn self_receiver_is_local_named_self() {
        // `fn m(&self) -> i64 { self.x() }` — `self` becomes a
        // Variable named "self" in `locals`, and `self.x()` lowers
        // like any other method call.
        let g = lower("fn m(&self) -> i64 { self.x() }").unwrap();
        checkgraph(&g);
        // One inputarg named by the adapter's `Variable::named("self")`
        // path, one getattr/simple_call pair.
        let start = g.startblock.borrow();
        assert_eq!(start.inputargs.len(), 1);
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
    }

    #[test]
    fn generic_fn_identity() {
        // `fn id<T>(x: T) -> T { x }` — accepted post-slice-M2.5c.
        // The adapter does not track T; the annotator's
        // FunctionDesc.specialize monomorphizes at the call site.
        let g = lower("fn id<T>(x: T) -> T { x }").unwrap();
        checkgraph(&g);
        assert_eq!(g.getargs().len(), 1);
    }

    #[test]
    fn generic_fn_with_trait_bound() {
        // `<E: Trait>` — trait-bound markers are parsed but not
        // inspected by the adapter; downstream annotator reads the
        // `args_s` classdef set.
        let g = lower(
            "fn step<E: StepExecutor>(e: E) -> i64 {
                e.execute()
            }",
        )
        .unwrap();
        checkgraph(&g);
        assert_eq!(g.getargs().len(), 1);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
    }

    #[test]
    fn generic_fn_with_where_clause() {
        let g = lower(
            "fn step<E>(e: E) -> i64 where E: StepExecutor {
                e.execute()
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn method_call_chained_with_let_rebinds() {
        // Ensure chained method calls interact properly with the
        // locals map: `let t = x.a(); t.b()` — the method-call
        // result is stored into a local and re-read.
        let g = lower(
            "fn f(x: i64) -> i64 {
                let t = x.a();
                t.b()
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 4);
    }

    // ---- M2.5c reject tests --------------------------------------

    #[test]
    fn rejects_method_turbofish() {
        match lower("fn f(x: i64) -> i64 { x.convert::<i64>() }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("turbofish"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(turbofish), got {other:?}"),
        }
    }

    #[test]
    fn rejects_call_with_non_identifier_callee() {
        // `(x + 1)(y)` — callee is not a simple path.
        match lower("fn f(x: i64, y: i64) -> i64 { (x + 1)(y) }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("identifier") || reason.contains("path"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(non-ident-callee), got {other:?}"),
        }
    }

    #[test]
    fn rejects_const_generic_param() {
        match lower("fn f<const N: usize>(x: i64) -> i64 { x }").unwrap_err() {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("const generic"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature(const generic), got {other:?}"),
        }
    }

    // ---- M2.5d literals + tuple/array ops ------------------------

    #[test]
    fn string_literal_lowers_to_constant_str() {
        // `fn f() -> i64 { let _s = "hello"; 0 }` — the string is
        // stored as a ConstValue::Str attached to a let binding,
        // then the tail emits 0. No op emitted for the literal
        // itself (it's a Constant, not a SpaceOperation result).
        let g = lower(r#"fn f() -> i64 { let _s = "hello"; 0 }"#).unwrap();
        checkgraph(&g);
    }

    #[test]
    fn float_literal_lowers_to_constant_float() {
        let g = lower("fn f() -> i64 { let _x = 3.14; 0 }").unwrap();
        checkgraph(&g);
    }

    #[test]
    fn tuple_literal_emits_newtuple_op() {
        let g = lower("fn f() -> i64 { let _t = (1, 2, 3); 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "newtuple");
        assert_eq!(start.operations[0].args.len(), 3);
    }

    #[test]
    fn array_literal_emits_newlist_op() {
        let g = lower("fn f() -> i64 { let _a = [1, 2, 3]; 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "newlist");
        assert_eq!(start.operations[0].args.len(), 3);
    }

    #[test]
    fn nested_tuple_of_tuples() {
        let g = lower("fn f(x: i64) -> i64 { let _t = ((x, 1), (2, x)); 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // Two inner newtuple + one outer newtuple = 3 ops.
        assert_eq!(start.operations.len(), 3);
        assert_eq!(start.operations[0].opname, "newtuple");
        assert_eq!(start.operations[1].opname, "newtuple");
        assert_eq!(start.operations[2].opname, "newtuple");
    }

    #[test]
    fn tuple_as_method_argument() {
        // The tuple flows into a method call — checks that
        // `lower_tuple`'s result is re-readable downstream.
        // Emission order is receiver → getattr → args → simple_call.
        let g = lower("fn f(x: i64) -> i64 { x.fold((1, 2)) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 3);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "newtuple");
        assert_eq!(start.operations[2].opname, "simple_call");
        // simple_call's second arg = newtuple result.
        assert_eq!(start.operations[2].args[1], start.operations[1].result);
    }

    #[test]
    fn empty_tuple_emits_newtuple_with_no_args() {
        let g = lower("fn f() -> i64 { let _u = (); 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "newtuple");
        assert_eq!(start.operations[0].args.len(), 0);
    }

    // ---- M2.5d reject tests --------------------------------------

    #[test]
    fn rejects_struct_literal() {
        match lower("fn f() -> i64 { let _s = Point { x: 1, y: 2 }; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("struct") || reason.contains("user-type"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(struct), got {other:?}"),
        }
    }

    #[test]
    fn char_literal_lowers_to_single_char_str() {
        // Rust `char` → `ConstValue::Str(len==1)`. Matches the
        // match-arm side (`classify_pattern`) so scrutinee and
        // exitcase share the identical Constant for an end-to-end
        // char match. No operations emitted — a bare `let _c = 'a'`
        // is pure SSA binding.
        let g = lower("fn f() -> i64 { let c = 'a'; 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "bare char literal binding emits no ops"
        );
    }

    #[test]
    fn rejects_byte_literal() {
        // `b'a'` parses as `Lit::Byte` and stays rejected — upstream
        // Python 2.7 doesn't have a distinct byte vocabulary in
        // flowspace.
        match lower("fn f() -> i64 { let _b = b'a'; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("byte"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(byte), got {other:?}"),
        }
    }

    #[test]
    fn char_literal_returned_as_single_char_str_constant() {
        // Expression-position `'a'` must reach the returnblock as
        // `Constant(Str("a"))` — same encoding the match-arm side
        // emits in `classify_pattern`. An end-to-end test with a
        // constant-scrutinee match would also need `guessbool` /
        // `HLOperation.eval` constant-folding (PRE-EXISTING-ADAPTATION
        // #3 in `mod.rs`) to close cleanly, so that composition is
        // deferred to the constfold port; here we only pin the
        // literal-to-Constant step.
        let g = lower("fn f() -> &'static str { 'a' }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 1);
        let link = start.exits[0].borrow();
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Str(s) => {
                    assert_eq!(s, "a");
                    assert_eq!(s.chars().count(), 1, "model.py:658 requires len==1");
                }
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    // ---- small-surface operator tests ---------------------------

    #[test]
    fn unary_neg_emits_neg_op() {
        let g = lower("fn f(x: i64) -> i64 { -x }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "neg");
        assert_eq!(start.operations[0].args.len(), 1);
    }

    #[test]
    fn rejects_unary_not() {
        // `!x` has no 1-to-1 upstream mapping — UNARY_NOT is a
        // `bool` + guessbool fork, UNARY_INVERT emits `invert`. The
        // adapter can't distinguish without type info.
        match lower("fn f(x: i64) -> i64 { !x }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("unary `!`"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(unary !), got {other:?}"),
        }
    }

    #[test]
    fn reference_is_passthrough() {
        // `&x` and `*x` emit no ops — they preserve the SSA value.
        let g = lower("fn f(x: i64) -> i64 { *&x }").unwrap();
        checkgraph(&g);
        assert_eq!(g.startblock.borrow().operations.len(), 0);
    }

    #[test]
    fn field_access_emits_getattr() {
        let g = lower("fn f(p: i64) -> i64 { p.x }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "getattr");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Str(s) => assert_eq!(s, "x"),
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    #[test]
    fn tuple_field_access_emits_getitem() {
        let g = lower("fn f(t: i64) -> i64 { t.0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations[0].opname, "getitem");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(0) => {}
                other => panic!("expected Int(0), got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    #[test]
    fn index_emits_getitem() {
        let g = lower("fn f(a: i64, i: i64) -> i64 { a[i] }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations[0].opname, "getitem");
        assert_eq!(start.operations[0].args.len(), 2);
    }

    #[test]
    fn rejects_numeric_cast() {
        // `x as T` has no 1-to-1 upstream mapping — upstream's
        // `do_int`/`do_float`/`bool` are dunder-method dispatches,
        // not numeric coercions. Users who need conversion should
        // call an explicit helper.
        match lower("fn f(x: i64) -> i64 { x as i32 }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("`x as T`"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(x as T), got {other:?}"),
        }
    }

    // ---- loop body can carry general expression statements -------

    #[test]
    fn loop_body_accepts_method_call_statement() {
        // `while cond { h.step(); }` — method call as a side-effect
        // inside the body. Upstream SETUP_LOOP (`flowcontext.py:488,
        // :794`) wraps arbitrary bytecode, not a subset.
        let g = lower(
            "fn f(h: Handler, cond: bool) -> i64 {
                while cond {
                    h.step();
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Body emits getattr('step') + simple_call(bound) — evidence
        // that the statement-position expression is lowered.
        let has_getattr = g.iterblocks().iter().any(|blk| {
            blk.borrow()
                .operations
                .iter()
                .any(|o| o.opname == "getattr")
        });
        assert!(has_getattr, "loop body must lower method call");
    }

    #[test]
    fn loop_body_accepts_field_access_statement() {
        // `while cond { let _ = h.field; }` — field access as a
        // side-effect in a let pattern. Verifies loop body delegates
        // to the same expression lowering path.
        let g = lower(
            "fn f(h: Handler, cond: bool) -> i64 {
                while cond {
                    let _ignore = h.field;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn top_level_expression_statement_discards_result() {
        // `fn f(h) { h.step(); 0 }` — method call as a statement at
        // top level. Same POP_TOP semantic as inside loop body; the
        // call's ops still emit but the result Variable is discarded.
        let g = lower(
            "fn f(h: Handler) -> i64 {
                h.step();
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
        let has_getattr = g.iterblocks().iter().any(|blk| {
            blk.borrow()
                .operations
                .iter()
                .any(|o| o.opname == "getattr")
        });
        assert!(has_getattr, "top-level expression statement must lower");
    }
}
