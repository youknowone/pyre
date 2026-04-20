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

fn py_bool(value: bool) -> &'static str {
    if value { "True" } else { "False" }
}

fn render_classdef_repr(classdef: &Rc<RefCell<crate::annotator::classdesc::ClassDef>>) -> String {
    format!("<ClassDef '{}'>", classdef.borrow().name)
}

fn render_classdef_key_repr(
    bookkeeper: &crate::annotator::bookkeeper::Bookkeeper,
    key: crate::annotator::description::ClassDefKey,
) -> String {
    for classdef in bookkeeper.classdefs.borrow().iter() {
        if crate::annotator::description::ClassDefKey::from_classdef(classdef) == key {
            return render_classdef_repr(classdef);
        }
    }
    format!("<ClassDefKey 0x{:x}>", key.0)
}

fn render_desc_repr(desc: &crate::annotator::description::DescEntry) -> String {
    use crate::annotator::description::DescEntry;
    match desc {
        DescEntry::Function(fd) => {
            let fd = fd.borrow();
            match &fd.base.pyobj {
                Some(pyobj) => format!("<FunctionDesc for {:?}>", pyobj),
                None => "<FunctionDesc>".into(),
            }
        }
        DescEntry::Frozen(fd) => {
            let fd = fd.borrow();
            match &fd.base.pyobj {
                Some(pyobj) => format!("<FrozenDesc for {:?}>", pyobj),
                None => "<FrozenDesc>".into(),
            }
        }
        DescEntry::Class(cd) => {
            let cd = cd.borrow();
            format!("<ClassDesc for {:?}>", cd.pyobj)
        }
        DescEntry::Method(md) => {
            let md = md.borrow();
            let origin = render_classdef_key_repr(&md.base.bookkeeper, md.originclassdef);
            match md.selfclassdef {
                None => format!("<unbound MethodDesc {:?} of {}>", md.name, origin),
                Some(selfclassdef) => format!(
                    "<MethodDesc {:?} of {} bound to {} {:?}>",
                    md.name,
                    origin,
                    render_classdef_key_repr(&md.base.bookkeeper, selfclassdef),
                    md.flags
                ),
            }
        }
        DescEntry::MethodOfFrozen(mfd) => {
            let mfd = mfd.borrow();
            let funcdesc = crate::annotator::description::DescEntry::Function(mfd.funcdesc.clone());
            let frozendesc =
                crate::annotator::description::DescEntry::Frozen(mfd.frozendesc.clone());
            format!(
                "<MethodOfFrozenDesc {} of {}>",
                render_desc_repr(&funcdesc),
                render_desc_repr(&frozendesc)
            )
        }
    }
}

fn render_somevalue(value: &crate::annotator::model::SomeValue) -> String {
    use crate::annotator::model::SomeValue;
    match value {
        SomeValue::Impossible => "SomeImpossibleValue()".into(),
        SomeValue::Object(_) => "SomeObject()".into(),
        SomeValue::Type(_) => "SomeType()".into(),
        SomeValue::Float(_) => "SomeFloat()".into(),
        SomeValue::SingleFloat(_) => "SomeSingleFloat()".into(),
        SomeValue::LongFloat(_) => "SomeLongFloat()".into(),
        SomeValue::Integer(i) => {
            let mut args = Vec::new();
            if i.nonneg {
                args.push("nonneg=True".to_string());
            }
            if i.unsigned {
                args.push("unsigned=True".to_string());
            }
            format!("SomeInteger({})", args.join(", "))
        }
        SomeValue::Bool(_) => "SomeBool()".into(),
        SomeValue::String(s) => {
            let mut args = Vec::new();
            if s.inner.can_be_none {
                args.push("can_be_None=True".to_string());
            }
            if s.inner.no_nul {
                args.push("no_nul=True".to_string());
            }
            format!("SomeString({})", args.join(", "))
        }
        SomeValue::UnicodeString(s) => {
            let mut args = Vec::new();
            if s.inner.can_be_none {
                args.push("can_be_None=True".to_string());
            }
            if s.inner.no_nul {
                args.push("no_nul=True".to_string());
            }
            format!("SomeUnicodeString({})", args.join(", "))
        }
        SomeValue::ByteArray(s) => {
            let mut args = Vec::new();
            if s.inner.can_be_none {
                args.push("can_be_None=True".to_string());
            }
            format!("SomeByteArray({})", args.join(", "))
        }
        SomeValue::Char(s) => {
            let mut args = Vec::new();
            if s.inner.no_nul {
                args.push("no_nul=True".to_string());
            }
            format!("SomeChar({})", args.join(", "))
        }
        SomeValue::UnicodeCodePoint(s) => {
            let mut args = Vec::new();
            if s.inner.no_nul {
                args.push("no_nul=True".to_string());
            }
            format!("SomeUnicodeCodePoint({})", args.join(", "))
        }
        SomeValue::List(_) => "SomeList(...)".into(),
        SomeValue::Tuple(t) => format!("SomeTuple(items=[{}])", t.items.len()),
        SomeValue::Dict(_) => "SomeDict(...)".into(),
        SomeValue::Iterator(_) => "SomeIterator(...)".into(),
        SomeValue::Instance(inst) => {
            let classdef = inst
                .classdef
                .as_ref()
                .map(render_classdef_repr)
                .unwrap_or_else(|| "None".into());
            format!(
                "SomeInstance(classdef={}, can_be_None={}, flags={:?})",
                classdef,
                py_bool(inst.can_be_none),
                inst.flags
            )
        }
        SomeValue::Exception(exc) => format!(
            "SomeException(classdefs=[{}])",
            exc.classdefs
                .iter()
                .map(render_classdef_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        SomeValue::PBC(pbc) => format!(
            "SomePBC(descriptions={{{}}}, can_be_None={})",
            pbc.descriptions
                .values()
                .map(render_desc_repr)
                .collect::<Vec<_>>()
                .join(", "),
            py_bool(pbc.can_be_none)
        ),
        SomeValue::None_(_) => "SomeNone()".into(),
        SomeValue::Builtin(_) => "SomeBuiltin()".into(),
        SomeValue::BuiltinMethod(_) => "SomeBuiltinMethod()".into(),
        SomeValue::WeakRef(_) => "SomeWeakRef()".into(),
        SomeValue::TypeOf(_) => "SomeTypeOf(...)".into(),
    }
}

fn graph_repr(graph: &GraphRef) -> String {
    let g = graph.borrow();
    format!(
        "<FunctionGraph of {} at 0x{:x}>",
        &*g,
        Rc::as_ptr(graph) as usize
    )
}

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
    let source = match g.source() {
        Ok(s) => s,
        Err(_) => return vec!["no source!".into()],
    };
    let filename = match g.filename() {
        Ok(s) => s,
        Err(_) => return vec!["no source!".into()],
    };
    let startline = match g.startline() {
        Ok(n) => n,
        Err(_) => return vec!["no source!".into()],
    };
    let func = match &g.func {
        Some(f) => f,
        None => return vec!["no source!".into()],
    };
    let code = match func.code.as_deref() {
        Some(c) => c,
        None => return vec!["no source!".into()],
    };
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
    let mut out = vec![format!("In {}:", graph_repr(graph))];
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
            if annotator.annotation(arg).is_some() {
                let s = annotator.binding(arg);
                msg.push(format!(" {} = {}", v, render_somevalue(&s)));
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
            let op = b
                .operations
                .get(i)
                .cloned()
                .expect("gather_error: operindex out of range");
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
        Some(op) => msg.push(format!("    {}\n", op)),
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
    if annotator.annotation(first).is_none() {
        msg.push("      (KeyError getting at the binding!)".into());
        return;
    }
    let crate::annotator::model::SomeValue::PBC(pbc) = annotator.binding(first) else {
        msg.push("      (AttributeError getting at the binding!)".into());
        return;
    };
    for desc in pbc.descriptions.values() {
        let rendered = match desc.pyobj() {
            Some(pyobj) if pyobj.is_class() => {
                let class_name = pyobj
                    .qualname()
                    .rsplit('.')
                    .next()
                    .unwrap_or(pyobj.qualname());
                match pyobj.class_get("__init__") {
                    Some(crate::flowspace::model::ConstValue::HostObject(init_host)) => {
                        if let Some(func) = init_host.user_function() {
                            match (func.filename.as_deref(), func.firstlineno) {
                                (Some(filename), Some(firstlineno)) => Some(format!(
                                    "function {}.__init__ <{}, line {}>",
                                    class_name, filename, firstlineno
                                )),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            Some(pyobj) => match pyobj.user_function() {
                Some(func) => match (func.filename.as_deref(), func.firstlineno) {
                    (Some(filename), Some(firstlineno)) => Some(format!(
                        "function {} <{}, line {}>",
                        func.name, filename, firstlineno
                    )),
                    _ => None,
                },
                None => None,
            },
            None => None,
        };
        let rendered = rendered.unwrap_or_else(|| render_desc_repr(desc));
        msg.push(format!("  {} returning", rendered));
        msg.push(String::new());
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
    blocked_blocks: &std::collections::HashMap<
        crate::flowspace::model::BlockKey,
        (BlockRef, GraphRef, Option<usize>),
    >,
) -> String {
    let mut text: Vec<String> = Vec::new();
    for (block, graph, index) in blocked_blocks.values() {
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
    use crate::annotator::model::{SomePBC, SomeValue};
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

    fn mk_func_host(
        name: &str,
        filename: &str,
        firstlineno: u32,
    ) -> crate::flowspace::model::HostObject {
        let mut func = GraphFunc::new(
            name,
            crate::flowspace::model::Constant::new(crate::flowspace::model::ConstValue::Dict(
                Default::default(),
            )),
        );
        let mut code = mk_code();
        code.co_name = name.into();
        code.co_filename = filename.into();
        code.co_firstlineno = firstlineno;
        func.code = Some(Box::new(code));
        func.filename = Some(filename.into());
        func.firstlineno = Some(firstlineno);
        crate::flowspace::model::HostObject::new_user_function(func)
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
        assert!(text.contains("In <FunctionGraph of f at 0x"));
    }

    #[test]
    fn format_simple_call_renders_function_filename_and_line() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let func_host = mk_func_host("callee", "<callee.py>", 42);
        let desc = ann.bookkeeper.getdesc(&func_host).unwrap();
        let pbc = SomeValue::PBC(SomePBC::new(vec![desc], false));
        let mut v_func = Variable::named("v_func");
        v_func.annotation = Some(Rc::new(pbc));
        let oper = crate::flowspace::model::SpaceOperation::new(
            "simple_call",
            vec![Hlvalue::Variable(v_func)],
            Hlvalue::Variable(Variable::new()),
        );
        let mut msg = Vec::new();

        format_simple_call(&ann, &oper, &mut msg);

        assert!(
            msg.iter()
                .any(|line| line.contains("Occurred processing the following simple_call:"))
        );
        assert!(
            msg.iter()
                .any(|line| line.contains("function callee <<callee.py>, line 42> returning"))
        );
    }

    #[test]
    fn format_annotations_uses_upstream_style_somevalue_rendering() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let mut v = Variable::named("v0");
        v.annotation = Some(Rc::new(SomeValue::Integer(
            crate::annotator::model::SomeInteger::new(true, false),
        )));
        let oper = crate::flowspace::model::SpaceOperation::new(
            "same_as",
            vec![Hlvalue::Variable(v.clone())],
            Hlvalue::Variable(v),
        );

        let lines = format_annotations(&ann, &oper);

        assert!(
            lines
                .iter()
                .any(|line| line.contains("SomeInteger(nonneg=True)")),
            "got {lines:?}"
        );
        assert!(
            !lines
                .iter()
                .any(|line| line.contains("Integer(SomeInteger")),
            "got {lines:?}"
        );
    }
}
