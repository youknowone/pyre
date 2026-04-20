//! RPython `rpython/tool/error.py` — annotator error formatting.
//!
//! Upstream:
//! * `SHOW_TRACEBACK = False` (error.py:13)
//! * `SHOW_ANNOTATIONS = True` (error.py:14)
//! * `SHOW_DEFAULT_LINES_OF_CODE = 0` (error.py:15)
//! * `source_lines1`, `source_lines`, `gather_error`, `format_annotations`,
//!   `format_blocked_annotation_error`, `format_simple_call`, `offset2lineno`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{Block, BlockRef, FunctionGraph, GraphRef, Hlvalue};

/// Upstream `SHOW_ANNOTATIONS = True` (error.py:14).
pub const SHOW_ANNOTATIONS: bool = true;

/// Upstream `SHOW_DEFAULT_LINES_OF_CODE = 0` (error.py:15).
pub const SHOW_DEFAULT_LINES_OF_CODE: usize = 0;

/// RPython `offset2lineno(c, stopat)` (error.py:160-171).
///
/// ```python
/// @jit.elidable
/// def offset2lineno(c, stopat):
///     tab = c.co_lnotab
///     line = c.co_firstlineno
///     addr = 0
///     for i in range(0, len(tab), 2):
///         addr = addr + ord(tab[i])
///         if addr > stopat:
///             break
///         line = line + ord(tab[i+1])
///     return line
/// ```
pub fn offset2lineno(code: &HostCode, stopat: i64) -> u32 {
    let tab = &code.co_lnotab;
    let mut line: u32 = code.co_firstlineno;
    let mut addr: i64 = 0;
    let mut i = 0;
    while i + 1 < tab.len() {
        addr += tab[i] as i64;
        if addr > stopat {
            break;
        }
        line = line.saturating_add(tab[i + 1] as u32);
        i += 2;
    }
    line
}

/// RPython `source_lines1(graph, block, operindex, offset=None,
///                          long=False, show_lines_of_code=SHOW_DEFAULT_LINES_OF_CODE)`
/// (error.py:18-61).
///
/// ```python
/// def source_lines1(graph, block, operindex=None, offset=None, long=False,
///                   show_lines_of_code=SHOW_DEFAULT_LINES_OF_CODE):
///     if block is not None:
///         if block is graph.returnblock:
///             return ['<return block>']
///     try:
///         source = graph.source
///     except AttributeError:
///         return ['no source!']
///     else:
///         graph_lines = source.split("\n")
///         if offset is not None:
///             linestart = offset2lineno(graph.func.__code__, offset)
///             linerange = (linestart, linestart)
///             here = None
///         else:
///             if block is None or not block.operations:
///                 return []
///             def toline(operindex):
///                 return offset2lineno(graph.func.__code__,
///                                      block.operations[operindex].offset)
///             if operindex is None:
///                 linerange = (toline(0), toline(-1))
///                 if not long: return ['?']
///                 here = None
///             else:
///                 operline = toline(operindex)
///                 if long:
///                     linerange = (toline(0), toline(-1))
///                     here = operline
///                 else:
///                     linerange = (operline, operline)
///                     here = None
///         lines = ["Happened at file %s line %d" % (graph.filename, here or linerange[0]), ""]
///         for n in range(max(0, linerange[0]-show_lines_of_code),
///                        min(linerange[1]+1+show_lines_of_code, len(graph_lines)+graph.startline)):
///             prefix = '==> ' if n == here else '    '
///             lines.append(prefix + graph_lines[n-graph.startline])
///         lines.append("")
///         return lines
/// ```
pub fn source_lines1(
    graph: &GraphRef,
    block: Option<&BlockRef>,
    operindex: Option<usize>,
    offset: Option<i64>,
    long: bool,
    show_lines_of_code: usize,
) -> Vec<String> {
    let g = graph.borrow();
    // upstream: `if block is graph.returnblock: return ['<return block>']`.
    if let Some(b) = block {
        if Rc::ptr_eq(b, &g.returnblock) {
            return vec!["<return block>".into()];
        }
    }
    // upstream: `source = graph.source`; attribute absent → ['no source!'].
    let func = match &g.func {
        Some(f) => f,
        None => return vec!["no source!".into()],
    };
    let source = match func.source.as_deref() {
        Some(s) => s,
        None => return vec!["no source!".into()],
    };
    let code = match func.code.as_deref() {
        Some(c) => c,
        None => return vec!["no source!".into()],
    };
    let startline: u32 = func.firstlineno.unwrap_or(1);
    let filename = func.filename.as_deref().unwrap_or("<unknown>");
    let graph_lines: Vec<&str> = source.split('\n').collect();

    // linerange computation branches on offset / operindex / block.
    let (linerange, here): ((u32, u32), Option<u32>) = if let Some(off) = offset {
        // upstream: offset provided — linerange = (linestart, linestart), here = None.
        let l = offset2lineno(code, off);
        ((l, l), None)
    } else {
        // upstream: `if block is None or not block.operations: return []`.
        let b = match block {
            Some(b) => b,
            None => return Vec::new(),
        };
        let block_borrow = b.borrow();
        if block_borrow.operations.is_empty() {
            return Vec::new();
        }
        let toline =
            |idx: usize| -> u32 { offset2lineno(code, block_borrow.operations[idx].offset) };
        let last_idx = block_borrow.operations.len() - 1;
        match operindex {
            None => {
                let range = (toline(0), toline(last_idx));
                if !long {
                    return vec!["?".into()];
                }
                (range, None)
            }
            Some(i) => {
                let operline = toline(i);
                if long {
                    ((toline(0), toline(last_idx)), Some(operline))
                } else {
                    ((operline, operline), None)
                }
            }
        }
    };

    // upstream: `lines = ["Happened at file %s line %d" % (graph.filename,
    //             here or linerange[0]), ""]`.
    let header_line = here.unwrap_or(linerange.0);
    let mut lines: Vec<String> = vec![
        format!("Happened at file {} line {}", filename, header_line),
        String::new(),
    ];
    // upstream: `for n in range(max(0, linerange[0]-show_lines_of_code),
    //             min(linerange[1]+1+show_lines_of_code, len(graph_lines)+graph.startline))`.
    let lo = linerange.0.saturating_sub(show_lines_of_code as u32);
    let hi_cap = (graph_lines.len() as u32).saturating_add(startline);
    let hi = std::cmp::min(linerange.1 + 1 + show_lines_of_code as u32, hi_cap);
    for n in lo..hi {
        let prefix = if Some(n) == here { "==> " } else { "    " };
        let idx = (n as i64 - startline as i64) as usize;
        if let Some(src) = graph_lines.get(idx) {
            lines.push(format!("{}{}", prefix, src));
        }
    }
    lines.push(String::new());
    lines
}

/// RPython `source_lines(graph, *args, **kwds)` (error.py:63-65).
///
/// ```python
/// def source_lines(graph, *args, **kwds):
///     lines = source_lines1(graph, *args, **kwds)
///     return ['In %r:' % (graph,)] + lines
/// ```
pub fn source_lines(
    graph: &GraphRef,
    block: Option<&BlockRef>,
    operindex: Option<usize>,
    offset: Option<i64>,
    long: bool,
    show_lines_of_code: usize,
) -> Vec<String> {
    let mut out = vec![format!("In {:?}:", graph.borrow().name)];
    out.extend(source_lines1(
        graph,
        block,
        operindex,
        offset,
        long,
        show_lines_of_code,
    ));
    out
}

/// RPython `format_annotations(annotator, oper)` (error.py:84-93).
///
/// ```python
/// def format_annotations(annotator, oper):
///     msg = []
///     msg.append("Known variable annotations:")
///     for arg in oper.args + [oper.result]:
///         if isinstance(arg, Variable):
///             try:
///                 msg.append(" " + str(arg) + " = " + str(annotator.binding(arg)))
///             except KeyError:
///                 pass
///     return msg
/// ```
pub fn format_annotations(
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    oper: &crate::flowspace::model::SpaceOperation,
) -> Vec<String> {
    let mut msg: Vec<String> = vec!["Known variable annotations:".into()];
    let mut args_and_result = oper.args.clone();
    args_and_result.push(oper.result.clone());
    for arg in &args_and_result {
        if let Hlvalue::Variable(v) = arg {
            if let Some(s) = annotator.annotation(arg) {
                msg.push(format!(" {} = {:?}", v.name(), s));
            }
        }
    }
    msg
}

/// RPython `gather_error(annotator, graph, block, operindex)`
/// (error.py:67-82).
///
/// ```python
/// def gather_error(annotator, graph, block, operindex):
///     msg = [""]
///     if operindex is not None:
///         oper = block.operations[operindex]
///         if oper.opname == 'simple_call':
///             format_simple_call(annotator, oper, msg)
///     else:
///         oper = None
///     msg.append("    %s\n" % str(oper))
///     msg += source_lines(graph, block, operindex, long=True)
///     if oper is not None:
///         if SHOW_ANNOTATIONS:
///             msg += format_annotations(annotator, oper)
///             msg += ['']
///     return "\n".join(msg)
/// ```
pub fn gather_error(
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    graph: &GraphRef,
    block: &BlockRef,
    operindex: Option<usize>,
) -> String {
    let mut msg: Vec<String> = vec![String::new()];
    // upstream: branch on `operindex is not None`.
    let oper = match operindex {
        Some(i) => {
            let b = block.borrow();
            let Some(op) = b.operations.get(i).cloned() else {
                msg.push(format!("    None  (operindex {} out of range)", i));
                return msg.join("\n");
            };
            // upstream: `if oper.opname == 'simple_call': format_simple_call(...)`.
            if op.opname == "simple_call" {
                format_simple_call(annotator, &op, &mut msg);
            }
            Some(op)
        }
        None => None,
    };
    // upstream: `msg.append("    %s\n" % str(oper))`.
    match &oper {
        Some(op) => msg.push(format!("    {:?}\n", op)),
        None => msg.push("    None\n".into()),
    }
    // upstream: `msg += source_lines(graph, block, operindex, long=True)`.
    msg.extend(source_lines(
        graph,
        Some(block),
        operindex,
        None,
        true,
        SHOW_DEFAULT_LINES_OF_CODE,
    ));
    // upstream: annotation dump.
    if let Some(op) = &oper {
        if SHOW_ANNOTATIONS {
            msg.extend(format_annotations(annotator, op));
            msg.push(String::new());
        }
    }
    msg.join("\n")
}

/// RPython `format_simple_call(annotator, oper, msg)` (error.py:102-112).
///
/// ```python
/// def format_simple_call(annotator, oper, msg):
///     msg.append("Occurred processing the following simple_call:")
///     try:
///         descs = annotator.binding(oper.args[0]).descriptions
///     except (KeyError, AttributeError) as e:
///         msg.append("      (%s getting at the binding!)" % (
///             e.__class__.__name__,))
///         return
///     for desc in list(descs):
///         ...
/// ```
///
/// The Rust port only surfaces the header + `(no callee descriptions)`
/// tail when the binding isn't a SomePBC — the full description-dump
/// (desc.pyobj / desc.name) requires HostObject introspection that the
/// callers haven't wired up yet.
pub fn format_simple_call(
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    oper: &crate::flowspace::model::SpaceOperation,
    msg: &mut Vec<String>,
) {
    msg.push("Occurred processing the following simple_call:".into());
    let Some(first) = oper.args.first() else {
        msg.push("      (simple_call with no arguments!)".into());
        return;
    };
    let Some(binding) = annotator.annotation(first) else {
        msg.push("      (KeyError getting at the binding!)".into());
        return;
    };
    match binding {
        crate::annotator::model::SomeValue::PBC(pbc) => {
            for desc in pbc.descriptions.values() {
                msg.push(format!("      {:?}", desc));
            }
        }
        other => {
            msg.push(format!("      (callee not a PBC: {:?})", other));
        }
    }
}

/// RPython `format_blocked_annotation_error(annotator, blocked_blocks)`
/// (error.py:95-100).
///
/// ```python
/// def format_blocked_annotation_error(annotator, blocked_blocks):
///     text = []
///     for block, (graph, index) in blocked_blocks.items():
///         text.append("Blocked block -- operation cannot succeed")
///         text.append(gather_error(annotator, graph, block, index))
///     return '\n'.join(text)
/// ```
pub fn format_blocked_annotation_error(
    annotator: &crate::annotator::annrpython::RPythonAnnotator,
    blocked_blocks: &[(GraphRef, BlockRef, Option<usize>)],
) -> String {
    let mut text: Vec<String> = Vec::new();
    for (graph, block, index) in blocked_blocks {
        text.push("Blocked block -- operation cannot succeed".into());
        text.push(gather_error(annotator, graph, block, *index));
    }
    text.join("\n")
}

// Imports kept alive even when format_annotations returns empty
// outside test builds.
#[allow(dead_code)]
fn _refs_hold(
    _block: Option<&RefCell<Block>>,
    _code: Option<&HostCode>,
    _graph: Option<&FunctionGraph>,
) {
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::bytecode::HostCode;
    use crate::flowspace::model::{Block, FunctionGraph, GraphFunc, Variable};

    fn mk_code() -> HostCode {
        use crate::flowspace::argument::Signature;
        HostCode {
            co_name: "f".into(),
            co_filename: "<test.py>".into(),
            co_firstlineno: 10,
            co_nlocals: 0,
            co_argcount: 0,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: Vec::new(),
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: vec![2, 1, 4, 1], // addr+2 → line+1 twice
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(Vec::new(), None, None),
        }
    }

    #[test]
    fn offset2lineno_walks_line_table() {
        let code = mk_code();
        // firstlineno = 10, lnotab = [2,1,4,1]:
        //   stopat=0..1 → line 10 (first addr=2, 2>stopat=1 → break, no inc)
        //   stopat=2..5 → line 11 (first addr=2 ≤ 2 → inc line, second addr=6 > stopat → break)
        //   stopat=6+  → line 12 (both pairs applied)
        assert_eq!(offset2lineno(&code, 0), 10);
        assert_eq!(offset2lineno(&code, 1), 10);
        assert_eq!(offset2lineno(&code, 2), 11);
        assert_eq!(offset2lineno(&code, 5), 11);
        assert_eq!(offset2lineno(&code, 6), 12);
        assert_eq!(offset2lineno(&code, 100), 12);
    }

    #[test]
    fn source_lines1_on_returnblock_returns_marker() {
        let startblock = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("f", startblock.clone());
        let mut func = GraphFunc::new(
            "f",
            crate::flowspace::model::Constant::new(crate::flowspace::model::ConstValue::Dict(
                Default::default(),
            )),
        );
        func.code = Some(Box::new(mk_code()));
        func.source = Some("def f():\n    return 1\n".into());
        func.firstlineno = Some(10);
        func.filename = Some("<test.py>".into());
        graph.func = Some(func);
        let returnblock = graph.returnblock.clone();
        let graph_ref: GraphRef = Rc::new(RefCell::new(graph));
        let out = source_lines1(&graph_ref, Some(&returnblock), None, None, false, 0);
        assert_eq!(out, vec!["<return block>".to_string()]);
    }

    #[test]
    fn gather_error_includes_location_header() {
        let startblock = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("f", startblock.clone());
        let mut func = GraphFunc::new(
            "f",
            crate::flowspace::model::Constant::new(crate::flowspace::model::ConstValue::Dict(
                Default::default(),
            )),
        );
        func.code = Some(Box::new(mk_code()));
        func.source = Some("def f():\n    return 1\n".into());
        func.firstlineno = Some(10);
        func.filename = Some("<test.py>".into());
        graph.func = Some(func);
        let graph_ref: GraphRef = Rc::new(RefCell::new(graph));
        let ann = RPythonAnnotator::new(None, None, None, false);
        // Add an operation to the startblock so source_lines has a
        // non-empty operations list.
        {
            let mut b = startblock.borrow_mut();
            b.operations
                .push(crate::flowspace::model::SpaceOperation::new(
                    "newtuple",
                    Vec::new(),
                    Hlvalue::Variable(Variable::new()),
                ));
        }
        let text = gather_error(&ann, &graph_ref, &startblock, Some(0));
        assert!(text.contains("Happened at file <test.py>"));
        assert!(text.contains("In \"f\":"));
    }
}
