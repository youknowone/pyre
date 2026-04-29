//! Port of `rpython/translator/backendopt/stat.py`.
//!
//! Walks reachable graphs from a single entry, counting blocks, ops,
//! and mallocs. Used by `all.py:78` / `:122` to print pre/post-opt
//! statistics when `config.translation.backendopt.print_statistics`
//! is set.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use md5::{Digest, Md5};

use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{ConstValue, GraphKey, GraphRef, Hlvalue};
use crate::translator::simplify::get_graph_for_call;
use crate::translator::translator::TranslationContext;

/// RPython `get_statistics(graph, translator,
/// save_per_graph_details=None, ignore_stack_checks=False)` at
/// `stat.py:6-57`. Returns `(num_graphs, num_blocks, num_ops,
/// num_mallocs)` over the graphs reachable from `graph` via
/// `direct_call` / `indirect_call` operations.
///
/// When `save_per_graph_details` is `Some(path)`, mirror upstream
/// `stat.py:41-56`: build a `(hash, name, nblocks, nops, nmallocs)`
/// row per graph, sort, and write each row to `path`. Upstream
/// hashes `graph.func.__code__.co_code` via `md5(...).hexdigest()`,
/// falling back to the literal string `"None"` when `co_code` is
/// missing (`AttributeError`). pyre walks `GraphFunc.code` —
/// `Some(HostCode)` graphs hash their `co_code` byte representation
/// (CPython 3.14 `CodeUnits`, two bytes per instruction); `None`
/// graphs fall back to `md5("None")` exactly as upstream does for
/// the `AttributeError` branch. The byte representation differs
/// from upstream's CPython-2.7 bytecode layout, so the hex digests
/// will not be byte-identical with an upstream run; the structural
/// branching (`code present → md5(bytes)`, `code absent →
/// md5("None")`) matches `:44-48`, and the sort order behaves
/// identically — distinct graphs emit distinct hashes when their
/// bodies differ.
pub fn get_statistics(
    graph: &GraphRef,
    translator: &TranslationContext,
    save_per_graph_details: Option<&Path>,
    ignore_stack_checks: bool,
) -> Statistics {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut stack: Vec<GraphRef> = vec![graph.clone()];
    let mut stats = Statistics::default();
    // Upstream `stat.py:13` `per_graph = {}`. Insertion-order vector
    // keyed by `GraphKey` so the post-walk emission can mirror
    // upstream's per-graph delta tuple.
    let mut per_graph: Vec<PerGraphEntry> = Vec::new();

    while let Some(current) = stack.pop() {
        let key = GraphKey::of(&current).as_usize();
        if !seen.insert(key) {
            continue;
        }
        stats.num_graphs += 1;
        let old_num_blocks = stats.num_blocks;
        let old_num_ops = stats.num_ops;
        let old_num_mallocs = stats.num_mallocs;
        let blocks = current.borrow().iterblocks();
        for block in blocks {
            stats.num_blocks += 1;
            let b = block.borrow();
            for op in &b.operations {
                match op.opname.as_str() {
                    "direct_call" => {
                        // Upstream `:27-32`: pull the called graph out
                        // of `op.args[0]`. None for indirect/delayed
                        // pointers, the `ll_stack_check` filter when
                        // `ignore_stack_checks` is set.
                        if let Some(callee) = op
                            .args
                            .first()
                            .and_then(|arg| get_graph_for_call(arg, translator))
                        {
                            let stack_check = ignore_stack_checks
                                && callee.borrow().name.starts_with("ll_stack_check");
                            if !stack_check {
                                stack.push(callee);
                            } else {
                                // Skip: matches upstream's
                                // `continue` after the
                                // `ll_stack_check` filter.
                                continue;
                            }
                        }
                    }
                    "indirect_call" => {
                        // Upstream `:33-36`: trailing arg carries the
                        // `c_graphs` list (`Constant(graphs, Void)`).
                        if let Some(Hlvalue::Constant(c)) = op.args.last() {
                            if let ConstValue::Graphs(keys) = &c.value {
                                let trans_graphs = translator.graphs.borrow();
                                for k in keys {
                                    if let Some(g) = trans_graphs
                                        .iter()
                                        .find(|g| GraphKey::of(g).as_usize() == *k)
                                    {
                                        stack.push(g.clone());
                                    }
                                }
                            }
                        }
                    }
                    name if name.starts_with("malloc") => {
                        stats.num_mallocs += 1;
                    }
                    _ => {}
                }
                stats.num_ops += 1;
            }
        }
        let hash = graph_co_code_hash(&current);
        per_graph.push(PerGraphEntry {
            hash,
            name: current.borrow().name.clone(),
            nblocks: stats.num_blocks - old_num_blocks,
            nops: stats.num_ops - old_num_ops,
            nmallocs: stats.num_mallocs - old_num_mallocs,
        });
    }

    if let Some(path) = save_per_graph_details {
        write_per_graph_details(path, &per_graph);
    }
    stats
}

/// Compute the hash column for upstream
/// `stat.py:44-48 try: code = graph.func.__code__.co_code except
/// AttributeError: code = "None"; hash = md5(code).hexdigest()`.
///
/// pyre walks `GraphFunc.code: Option<HostCode>`:
///   * `None`        → `md5(b"None")` (upstream's AttributeError
///                     branch).
///   * `Some(host)`  → `md5(co_code_bytes(&host))`. The byte
///                     representation packs each `CodeUnit` as
///                     `(op, arg)` two-byte pairs, which matches
///                     CPython 3.14's `co_code` layout. The digest
///                     is therefore not byte-identical to a
///                     CPython-2.7 upstream run, but the
///                     stat.py-shaped branching is preserved.
fn graph_co_code_hash(graph: &GraphRef) -> String {
    let g = graph.borrow();
    match g.func.as_ref().and_then(|f| f.code.as_deref()) {
        Some(host) => md5_hex(&co_code_bytes(host)),
        None => md5_hex(b"None"),
    }
}

/// Pack `HostCode.co_code` into a flat byte sequence — two bytes per
/// `CodeUnit` (`op`, `arg`), matching CPython 3.14's `co_code` shape.
fn co_code_bytes(host: &HostCode) -> Vec<u8> {
    let units = &*host.co_code;
    let mut bytes = Vec::with_capacity(units.len() * 2);
    for unit in units {
        bytes.push(u8::from(unit.op));
        bytes.push(u8::from(unit.arg));
    }
    bytes
}

/// Compute `hashlib.md5(data).hexdigest()` — upstream's
/// `stat.py:48 md5(code).hexdigest()`.
fn md5_hex(data: &[u8]) -> String {
    let digest = Md5::digest(data);
    format!("{digest:x}")
}

#[derive(Clone, Debug)]
struct PerGraphEntry {
    hash: String,
    name: String,
    nblocks: usize,
    nops: usize,
    nmallocs: usize,
}

fn write_per_graph_details(path: &Path, per_graph: &[PerGraphEntry]) {
    // Upstream `stat.py:42-50`: build (hash, name, nblocks, nops,
    // nmallocs) tuples and sort. Tuple comparison cascades through
    // the columns left-to-right; identical hashes fall through to
    // `(name, nblocks, ...)` per Python tuple ordering.
    let mut rows: Vec<(&str, &str, usize, usize, usize)> = per_graph
        .iter()
        .map(|e| {
            (
                e.hash.as_str(),
                e.name.as_str(),
                e.nblocks,
                e.nops,
                e.nmallocs,
            )
        })
        .collect();
    rows.sort();
    // Upstream `:51-56`: open file, print one row per line, close.
    let file = File::create(path)
        .unwrap_or_else(|err| panic!("stat.py:51 open({}, 'w') failed: {err}", path.display(),));
    let mut writer = BufWriter::new(file);
    for (hash, name, nblocks, nops, nmallocs) in rows {
        writeln!(writer, "{hash} {name} {nblocks} {nops} {nmallocs}")
            .unwrap_or_else(|err| panic!("stat.py:54 write to {} failed: {err}", path.display(),));
    }
}

/// RPython `print_statistics(graph, translator,
/// save_per_graph_details=None, ignore_stack_checks=False)` at
/// `stat.py:59-67`. Mirrors upstream's signature; the
/// `save_per_graph_details` argument forwards into `get_statistics`
/// where the file-emission path lives.
pub fn print_statistics(
    graph: &GraphRef,
    translator: &TranslationContext,
    save_per_graph_details: Option<&Path>,
    ignore_stack_checks: bool,
) {
    let stats = get_statistics(
        graph,
        translator,
        save_per_graph_details,
        ignore_stack_checks,
    );
    println!(
        "Statistics:\nnumber of graphs {}\nnumber of blocks {}\nnumber of operations {}\nnumber of mallocs {}\n",
        stats.num_graphs, stats.num_blocks, stats.num_ops, stats.num_mallocs
    );
}

/// Tuple-equivalent return of upstream
/// `(num_graphs, num_blocks, num_ops, num_mallocs)`. Rust's tuples
/// have no field names; the explicit struct is the smallest
/// adaptation that keeps call sites self-documenting without
/// introducing accessors that upstream lacks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statistics {
    pub num_graphs: usize,
    pub num_blocks: usize,
    pub num_ops: usize,
    pub num_mallocs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, FunctionGraph, Link, SpaceOperation, Variable,
    };
    use crate::translator::translator::TranslationContext;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn empty_graph(name: &str) -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new(name.to_string(), start.clone());
        // Close startblock onto the auto-generated returnblock.
        let return_target = graph.returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    fn add_malloc_op(graph: &GraphRef) {
        let block = graph.borrow().startblock.clone();
        let op = SpaceOperation::new(
            "malloc".to_string(),
            vec![],
            Hlvalue::Variable(Variable::named("m")),
        );
        block.borrow_mut().operations.push(op);
    }

    #[test]
    fn get_statistics_counts_blocks_ops_and_mallocs_in_single_graph() {
        let graph = empty_graph("entry");
        add_malloc_op(&graph);
        let translator = TranslationContext::new();
        translator.graphs.borrow_mut().push(graph.clone());

        let stats = get_statistics(&graph, &translator, None, false);
        assert_eq!(stats.num_graphs, 1);
        // iterblocks BFS from startblock: startblock + returnblock.
        // exceptblock is excluded — no link reaches it.
        assert_eq!(stats.num_blocks, 2);
        // single malloc op.
        assert_eq!(stats.num_ops, 1);
        assert_eq!(stats.num_mallocs, 1);
    }

    #[test]
    fn get_statistics_dedups_repeat_walks() {
        let graph = empty_graph("entry");
        let translator = TranslationContext::new();
        translator.graphs.borrow_mut().push(graph.clone());

        let stats = get_statistics(&graph, &translator, None, false);
        // Re-walking from the same entry must give the same totals.
        let stats2 = get_statistics(&graph, &translator, None, false);
        assert_eq!(stats, stats2);
    }

    #[test]
    fn save_per_graph_details_writes_md5_name_counts_per_line() {
        // Mirror upstream `stat.py:42-56`: every graph yields one row
        // `<hash> <name> <nblocks> <nops> <nmallocs>`. With pyre's
        // `code = None` AttributeError branch, `hash` is `md5("None")`
        // verbatim — confirming the AttributeError branch lights up
        // for graphs whose `func.code` is absent.
        let graph = empty_graph("entry");
        add_malloc_op(&graph);
        let translator = TranslationContext::new();
        translator.graphs.borrow_mut().push(graph.clone());

        let dir = std::env::temp_dir();
        let path = dir.join(format!("majit-stat-test-{}.txt", std::process::id(),));
        let _ = std::fs::remove_file(&path);
        let stats = get_statistics(&graph, &translator, Some(&path), false);
        assert_eq!(stats.num_graphs, 1);
        let body = std::fs::read_to_string(&path).expect("file emitted");
        let expected_hash = md5_hex(b"None");
        assert_eq!(body, format!("{expected_hash} entry 2 1 1\n"),);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn md5_of_none_matches_python_hashlib() {
        // `hashlib.md5(b"None").hexdigest()` from CPython 3.x —
        // pinned so a regression in the digest path is visible at
        // unit-test time without running the file-emission test.
        assert_eq!(md5_hex(b"None"), "6adf97f83acf6453d4a6a4b1070f3754");
    }
}
