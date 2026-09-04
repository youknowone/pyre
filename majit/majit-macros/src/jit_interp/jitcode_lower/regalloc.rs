//! Proc-macro JitCode register allocation.
//!
//! RPython runs `tool/algo/regalloc.py::RegAllocator` on the graph before
//! `flatten.py` emits numbered registers (`codewriter.py::CodeWriter.make_jitcode`).
//! The proc-macro lowerer used to number every temporary monotonically and
//! bake those numbers straight into its future `JitCodeBuilder` calls.  This
//! pass colors the still-symbolic [`OpMeta`] control-flow graph, then rewrites
//! the not-yet-executed builder statements.  In particular, it runs before
//! liveness encoding and before bytecode flattening at macro-expansion time.

use super::*;
use majit_translate::tool::algo::color::DependencyGraph;
use quote::{ToTokens, quote};
use std::collections::{BTreeSet, HashMap};
use syn::visit_mut::VisitMut;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RegisterCounts {
    pub(super) ints: u16,
    pub(super) refs: u16,
    pub(super) floats: u16,
}

impl RegisterCounts {
    fn for_kind(self, kind: BindingKind) -> u16 {
        match kind {
            BindingKind::Int => self.ints,
            BindingKind::Ref => self.refs,
            BindingKind::Float => self.floats,
        }
    }

    fn observe(&mut self, reg: Register) {
        let end = u16::from(reg.index) + 1;
        match reg.kind {
            BindingKind::Int => self.ints = self.ints.max(end),
            BindingKind::Ref => self.refs = self.refs.max(end),
            BindingKind::Float => self.floats = self.floats.max(end),
        }
    }
}

/// RPython `tool/algo/unionfind.py::UnionFind`, narrowed to the symbolic
/// registers owned by this proc-macro lowering path.
///
/// `majit-translate` has the same private helper for real flowspace
/// Variables. Keeping this one local avoids making that implementation part
/// of the cross-crate API merely for the temporary adapter that disappears
/// once proc-macro helpers enter the normal codewriter graph.
struct UnionFind {
    parent: HashMap<Register, Register>,
    weight: HashMap<Register, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            weight: HashMap::new(),
        }
    }

    fn find_rep(&mut self, reg: Register) -> Register {
        if !self.parent.contains_key(&reg) {
            self.parent.insert(reg, reg);
            self.weight.insert(reg, 1);
            return reg;
        }
        let mut root = reg;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }
        let mut current = reg;
        while current != root {
            let next = self.parent[&current];
            self.parent.insert(current, root);
            current = next;
        }
        root
    }

    fn union(&mut self, left: Register, right: Register) -> Register {
        let left = self.find_rep(left);
        let right = self.find_rep(right);
        if left == right {
            return left;
        }
        let left_weight = self.weight[&left];
        let right_weight = self.weight[&right];
        let (winner, loser) = if left_weight >= right_weight {
            (left, right)
        } else {
            (right, left)
        };
        self.parent.insert(loser, winner);
        self.weight.remove(&loser);
        self.weight.insert(winner, left_weight + right_weight);
        winner
    }
}

fn successors(ops: &[OpMeta]) -> Vec<Vec<usize>> {
    let labels: HashMap<String, usize> = ops
        .iter()
        .enumerate()
        .filter_map(|(index, op)| match op.control {
            ControlFlowClass::LabelDef => op
                .target_label
                .as_ref()
                .map(|label| (label.to_string(), index)),
            _ => None,
        })
        .collect();
    ops.iter()
        .enumerate()
        .map(|(index, op)| {
            let fallthrough = || (index + 1 < ops.len()).then_some(index + 1);
            match op.control {
                ControlFlowClass::Terminal => Vec::new(),
                ControlFlowClass::UnconditionalJump => op
                    .target_label
                    .as_ref()
                    .and_then(|label| labels.get(&label.to_string()).copied())
                    .into_iter()
                    .collect(),
                ControlFlowClass::ConditionalGuard => {
                    let mut out: Vec<usize> = fallthrough().into_iter().collect();
                    if let Some(target) = op
                        .target_label
                        .as_ref()
                        .and_then(|label| labels.get(&label.to_string()).copied())
                        && !out.contains(&target)
                    {
                        out.push(target);
                    }
                    out
                }
                ControlFlowClass::Linear
                | ControlFlowClass::LiveMarker
                | ControlFlowClass::LabelDef => fallthrough().into_iter().collect(),
            }
        })
        .collect()
}

fn live_sets(ops: &[OpMeta]) -> (Vec<BTreeSet<Register>>, Vec<BTreeSet<Register>>) {
    let succ = successors(ops);
    let mut live_in = vec![BTreeSet::new(); ops.len()];
    let mut live_out = vec![BTreeSet::new(); ops.len()];
    loop {
        let mut changed = false;
        for index in (0..ops.len()).rev() {
            let mut out = BTreeSet::new();
            for &next in &succ[index] {
                out.extend(live_in[next].iter().copied());
            }
            let mut input = out.clone();
            for written in &ops[index].writes {
                input.remove(written);
            }
            input.extend(ops[index].reads.iter().copied());
            if out != live_out[index] || input != live_in[index] {
                live_out[index] = out;
                live_in[index] = input;
                changed = true;
            }
        }
        if !changed {
            return (live_in, live_out);
        }
    }
}

fn add_clique(graph: &mut DependencyGraph<Register>, regs: &BTreeSet<Register>) {
    for &reg in regs {
        graph.add_node(reg);
    }
    let regs: Vec<_> = regs.iter().copied().collect();
    for (index, &left) in regs.iter().enumerate() {
        for &right in &regs[index + 1..] {
            if left.kind == right.kind && !graph.has_edge(&left, &right) {
                graph.add_edge(left, right);
            }
        }
    }
}

fn coloring(
    ops: &[OpMeta],
    reserved: RegisterCounts,
    return_reg: Option<Register>,
) -> HashMap<Register, Register> {
    let mut owned_ops;
    let ops = if let Some(return_reg) = return_reg {
        owned_ops = ops.to_vec();
        owned_ops.push(OpMeta::terminal(vec![return_reg]));
        owned_ops.as_slice()
    } else {
        ops
    };
    let (live_in, live_out) = live_sets(ops);
    let mut graphs: HashMap<BindingKind, DependencyGraph<Register>> = HashMap::new();
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        graphs.insert(kind, DependencyGraph::new());
    }
    // `RegAllocator.make_dependencies` starts every block with all of its
    // inputargs live and makes that set a clique. The proc-macro ABI registers
    // are the entry block's inputargs; seed them explicitly so even an unused
    // parameter retains a distinct caller slot.
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        let inputs: BTreeSet<_> = (0..reserved.for_kind(kind))
            .map(|index| Register::new(kind, index))
            .collect();
        add_clique(graphs.get_mut(&kind).unwrap(), &inputs);
    }
    for op in ops {
        for &reg in op.reads.iter().chain(op.writes.iter()) {
            graphs.get_mut(&reg.kind).unwrap().add_node(reg);
        }
    }
    for set in live_in.iter().chain(live_out.iter()) {
        for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
            let bank: BTreeSet<_> = set.iter().filter(|reg| reg.kind == kind).copied().collect();
            add_clique(graphs.get_mut(&kind).unwrap(), &bank);
        }
    }
    for (op, out) in ops.iter().zip(&live_out) {
        for &written in &op.writes {
            let graph = graphs.get_mut(&written.kind).unwrap();
            for &alive in out {
                if alive.kind == written.kind
                    && alive != written
                    && !graph.has_edge(&written, &alive)
                {
                    graph.add_edge(written, alive);
                }
            }
        }
    }

    // RPython `tool/algo/regalloc.py::RegAllocator.coalesce_variables` walks
    // blocks from the end and coalesces each link argument with the matching
    // target inputarg before coloring. The proc-macro CFG has already
    // flattened those links into typed Move operations, so those moves are
    // exactly the source/target pairs to feed to the same algorithm. This was
    // the missing half of this adapter's claimed pre-flatten allocation: every
    // branch join survived as a runtime `int_copy` even when its Variables did
    // not interfere.
    let originals: BTreeSet<Register> = ops
        .iter()
        .flat_map(|op| op.reads.iter().chain(&op.writes))
        .copied()
        .chain(return_reg)
        .collect();
    let mut unionfind = UnionFind::new();
    for op in ops.iter().rev() {
        if !matches!(op.kind, OpKind::MoveI | OpKind::MoveR | OpKind::MoveF)
            || op.reads.len() != 1
            || op.writes.len() != 1
        {
            continue;
        }
        let source = unionfind.find_rep(op.reads[0]);
        let target = unionfind.find_rep(op.writes[0]);
        if source == target || source.kind != target.kind {
            continue;
        }
        let graph = graphs.get_mut(&source.kind).unwrap();
        if graph.has_edge(&source, &target) {
            continue;
        }
        let representative = unionfind.union(source, target);
        if representative == source {
            graph.coalesce(target, source);
        } else {
            graph.coalesce(source, target);
        }
    }

    let mut result = HashMap::new();
    for kind in [BindingKind::Int, BindingKind::Ref, BindingKind::Float] {
        let fixed = reserved.for_kind(kind);
        let graph = &graphs[&kind];
        let mut representative_colors = graph.find_node_coloring();

        // RPython `flatten.py::GraphFlattener.enforce_input_args` does not
        // reserve the ABI prefix while coloring. It swaps colors afterwards,
        // which lets a temporary reuse a dead input slot while still making
        // caller-visible inputargs dense at 0..N. The former adapter excluded
        // the entire prefix from every temporary.
        for input_index in 0..fixed {
            let input = Register::new(kind, input_index);
            let representative = unionfind.find_rep(input);
            let Some(current_color) = representative_colors.get(&representative).copied() else {
                continue;
            };
            let desired_color = usize::from(input_index);
            if current_color == desired_color {
                continue;
            }
            for color in representative_colors.values_mut() {
                if *color == current_color {
                    *color = desired_color;
                } else if *color == desired_color {
                    *color = current_color;
                }
            }
        }

        for original in originals.iter().filter(|reg| reg.kind == kind) {
            let representative = unionfind.find_rep(*original);
            if let Some(&color) = representative_colors.get(&representative) {
                result.insert(
                    *original,
                    Register {
                        kind,
                        index: u8::try_from(color).expect("JitCode register coloring exceeds u8"),
                    },
                );
            }
        }
    }
    result
}

fn is_builder_receiver(expr: &syn::Expr) -> bool {
    matches!(expr, syn::Expr::Path(path) if path.path.is_ident("__builder"))
}

fn is_aux_builder_method(name: &str) -> bool {
    name.starts_with("ensure_")
        || name.starts_with("register_")
        || name.starts_with("set_")
        || name.starts_with("add_")
        || matches!(
            name,
            "new_label" | "mark_label" | "finalize_liveness" | "finish"
        )
}

/// Rewrites one builder call's register literals, in argument order.
///
/// The pass works on tokens, so a register operand and an ordinary integer
/// constant are both `Lit::Int` and nothing in the syntax tells them apart.
/// What separates them is that `remaining` is seeded with exactly this op's
/// reads and writes and each match spends one: by the time a constant is
/// visited, the registers it could collide with are already spent. That holds
/// only because **every builder method lists its register operands before its
/// constants**. A method taking a constant first would let it consume the slot
/// and leave the real register literal unrewritten — with the counts still
/// balancing, so nothing here would report it. Keep new builder methods
/// registers-first.
struct LiteralRegisterRewriter<'a> {
    mapping: &'a HashMap<Register, Register>,
    remaining: HashMap<Register, usize>,
    /// Every register this rewriter recolored, accumulated across all the
    /// builder calls in one statement so [`rewrite_statement`] can check that
    /// the statement spelled each register the [`OpMeta`] declares.
    recolored: &'a mut BTreeSet<Register>,
}

impl VisitMut for LiteralRegisterRewriter<'_> {
    fn visit_expr_lit_mut(&mut self, expr: &mut syn::ExprLit) {
        let syn::Lit::Int(lit) = &expr.lit else {
            return;
        };
        let Ok(index) = lit.base10_parse::<u8>() else {
            return;
        };
        let candidates: Vec<_> = self
            .remaining
            .iter()
            .filter(|(reg, count)| reg.index == index && **count > 0)
            .map(|(reg, _)| *reg)
            .collect();
        if candidates.is_empty() {
            return;
        }
        let replacement = self.mapping[&candidates[0]];
        assert!(
            candidates
                .iter()
                .all(|candidate| self.mapping[candidate].index == replacement.index),
            "ambiguous cross-bank register literal {index} in one builder call"
        );
        *self.remaining.get_mut(&candidates[0]).unwrap() -= 1;
        self.recolored.insert(candidates[0]);
        expr.lit = syn::Lit::Int(syn::LitInt::new(
            &format!("{}u16", replacement.index),
            lit.span(),
        ));
    }
}

struct BuilderStatementRewriter<'a> {
    mapping: &'a HashMap<Register, Register>,
    expected: HashMap<Register, usize>,
    recolored: BTreeSet<Register>,
}

impl VisitMut for BuilderStatementRewriter<'_> {
    fn visit_expr_method_call_mut(&mut self, call: &mut syn::ExprMethodCall) {
        if is_builder_receiver(&call.receiver) && !is_aux_builder_method(&call.method.to_string()) {
            let mut rewrite = LiteralRegisterRewriter {
                mapping: self.mapping,
                remaining: self.expected.clone(),
                recolored: &mut self.recolored,
            };
            for arg in &mut call.args {
                rewrite.visit_expr_mut(arg);
            }
            return;
        }
        syn::visit_mut::visit_expr_method_call_mut(self, call);
    }
}

fn rewrite_statement(
    statement: &TokenStream,
    meta: &OpMeta,
    mapping: &HashMap<Register, Register>,
) -> TokenStream {
    let mut expected = HashMap::new();
    for &reg in meta.reads.iter().chain(meta.writes.iter()) {
        *expected.entry(reg).or_insert(0) += 1;
    }
    if expected.is_empty() {
        return statement.clone();
    }
    let mut block: syn::Block = syn::parse2(quote!({ #statement }))
        .expect("macro-generated JitCode statement must parse as a Rust block");
    let mut rewriter = BuilderStatementRewriter {
        mapping,
        expected: expected.clone(),
        recolored: BTreeSet::new(),
    };
    rewriter.visit_block_mut(&mut block);
    // Only the arguments of a non-aux `__builder` method call are recolored, so
    // a register an operation declares but spells anywhere else — bound to a
    // local first, say — would silently keep the number the lowerer handed out
    // before coloring, and the emitted call would then read a register nothing
    // ever writes. Every declared register must therefore appear as a literal
    // the rewriter reached.
    for reg in expected.keys() {
        assert!(
            rewriter.recolored.contains(reg),
            "JitCode statement declares {reg:?} but never spells it inside a \
             `__builder` call, so coloring cannot reach it: {statement}"
        );
    }
    block
        .stmts
        .into_iter()
        .map(|stmt| stmt.into_token_stream())
        .collect()
}

/// Color the proc-macro operation graph and rewrite its future builder calls.
/// Returns the compact per-bank register counts and the remapped `return_reg`.
pub(super) fn compact_registers(
    lowerer: &mut Lowerer<'_>,
    reserved: RegisterCounts,
    return_reg: Option<Register>,
) -> (RegisterCounts, Option<Register>) {
    debug_assert_eq!(lowerer.statements.len(), lowerer.op_metadata.len());
    let mapping = coloring(&lowerer.op_metadata, reserved, return_reg);
    for (statement, meta) in lowerer.statements.iter_mut().zip(&lowerer.op_metadata) {
        *statement = rewrite_statement(statement, meta, &mapping);
    }
    for meta in &mut lowerer.op_metadata {
        for reg in meta.reads.iter_mut().chain(meta.writes.iter_mut()) {
            if let Some(mapped) = mapping.get(reg) {
                *reg = *mapped;
            }
        }
    }
    // `flatten.py::GraphFlattener.insert_renamings` compares post-regalloc
    // source and destination colors and emits no `*_copy` when they match.
    // Our link moves were materialized before this adapter runs, so perform
    // the same check after rewriting and remove the now-empty links.
    let (statements, op_metadata): (Vec<_>, Vec<_>) = std::mem::take(&mut lowerer.statements)
        .into_iter()
        .zip(std::mem::take(&mut lowerer.op_metadata))
        .filter(|(_, meta)| {
            !matches!(meta.kind, OpKind::MoveI | OpKind::MoveR | OpKind::MoveF)
                || meta.reads != meta.writes
        })
        .unzip();
    lowerer.statements = statements;
    lowerer.op_metadata = op_metadata;
    let mut counts = reserved;
    for mapped in mapping.values() {
        counts.observe(*mapped);
    }
    let return_reg = return_reg.map(|reg| mapping.get(&reg).copied().unwrap_or(reg));
    (counts, return_reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_overlapping_temporaries_share_a_color_before_flattening() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.int_is_true(2u16, 1u16); },
        );
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(3)]),
            quote! { __builder.load_const_i_value(3u16, 7i64); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(3)]),
            quote! { __builder.int_return(3u16); },
        );

        let (counts, returned) = compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(3)),
        );
        assert!(counts.ints < 4);
        assert_eq!(u16::from(returned.unwrap().index) + 1, counts.ints);
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(!emitted.contains("3u16"));
    }

    /// `RegAllocator.coalesce_variables` makes a link source and its target
    /// inputarg one Variable when they do not interfere; `flatten.py`
    /// `GraphFlattener.insert_renamings` consequently emits no copy.
    #[test]
    fn noninterfering_link_move_is_coalesced_and_not_emitted() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.move_i(2u16, 1u16); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(2)]),
            quote! { __builder.int_return(2u16); },
        );

        let (counts, returned) = compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(2)),
        );

        assert_eq!(counts.ints, 1, "a dead ABI input slot is reusable");
        assert_eq!(returned, Some(Register::int(0)));
        assert!(
            !lowerer
                .op_metadata
                .iter()
                .any(|meta| meta.kind == OpKind::MoveI)
        );
        assert!(
            !lowerer
                .statements
                .iter()
                .map(ToString::to_string)
                .collect::<String>()
                .contains("move_i")
        );
    }

    /// A link source that stays live after the assignment interferes with the
    /// target. Upstream `_try_coalesce` leaves that renaming for flattening.
    #[test]
    fn interfering_link_move_remains_a_copy() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(1)],
                vec![Register::int(2)],
            ),
            quote! { __builder.move_i(2u16, 1u16); },
        );
        lowerer.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[1, 2]),
                vec![Register::int(3)],
            ),
            quote! { __builder.record_binop_i(3u16, majit_ir::OpCode::IntAdd, 1u16, 2u16); },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(3)]),
            quote! { __builder.int_return(3u16); },
        );

        compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(3)),
        );

        assert!(
            lowerer
                .op_metadata
                .iter()
                .any(|meta| meta.kind == OpKind::MoveI)
        );
    }

    /// The rewriter reaches only the arguments of a `__builder` call, so an
    /// operation that binds its arguments to a local first would keep its
    /// pre-coloring register numbers and call a register nothing writes.
    #[test]
    #[should_panic(expected = "never spells it inside a")]
    fn a_declared_register_spelled_outside_the_builder_call_is_rejected() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(1)]),
            quote! { __builder.load_const_i_value(1u16, 41i64); },
        );
        lowerer.emit_op(
            OpMeta::linear(OpKind::Call, vec![Register::int(1)], vec![Register::int(2)]),
            quote! {
                let __typed_args = &[majit_metainterp::JitCallArg::int(1u16)];
                __builder.residual_call_int_canonical_via_target(__fn_idx, __typed_args, 2u16);
            },
        );
        lowerer.emit_op(
            OpMeta::terminal(vec![Register::int(2)]),
            quote! { __builder.int_return(2u16); },
        );

        compact_registers(
            &mut lowerer,
            RegisterCounts {
                ints: 1,
                ..RegisterCounts::default()
            },
            Some(Register::int(2)),
        );
    }
}
