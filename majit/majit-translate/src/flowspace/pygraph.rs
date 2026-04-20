//! Flow graphs for Python callables.
//!
//! RPython upstream: `rpython/flowspace/pygraph.py` (33 LOC).
//!
//! Rust adaptation (parity rule #1):
//!
//! * Upstream `class PyGraph(FunctionGraph)` uses Python single
//!   inheritance to add three fields (`func`, `signature`, `defaults`)
//!   and override `__init__`. Rust has no inheritance; the port
//!   composes `FunctionGraph` into `PyGraph` and exposes the fields
//!   directly. Every `FunctionGraph` method remains reachable via
//!   `pygraph.graph`.

use super::argument::Signature;
use super::bytecode::HostCode;
use super::flowcontext::SpamBlock;
use super::framestate::FrameState;
use super::model::{Constant, FunctionGraph, GraphFunc, Hlvalue, Variable};

/// RPython `rpython/flowspace/pygraph.py:7-33` — `class PyGraph(FunctionGraph)`.
///
/// A flow graph for a Python function. Carries the originating
/// `GraphFunc`, the code object's call `Signature`, and the Python
/// `func.__defaults__` tuple alongside the ordinary `FunctionGraph`
/// fields.
#[derive(Debug)]
pub struct PyGraph {
    /// Ported `FunctionGraph` base instance. Accessible fields:
    /// `name`, `startblock`, `returnblock`, `exceptblock`, etc.
    pub graph: FunctionGraph,
    /// RPython `PyGraph.func`.
    pub func: GraphFunc,
    /// RPython `PyGraph.signature = code.signature`.
    pub signature: Signature,
    /// RPython `PyGraph.defaults = func.__defaults__ or ()`.
    pub defaults: Vec<Constant>,
}

impl PyGraph {
    /// RPython `PyGraph.__init__(func, code)` (pygraph.py:12-22).
    pub fn new(func: GraphFunc, code: &HostCode) -> Self {
        // upstream: `locals = [None] * code.co_nlocals`.
        let mut locals: Vec<Option<Hlvalue>> = vec![None; code.co_nlocals as usize];
        // upstream: `for i in range(code.formalargcount):
        //                locals[i] = Variable(code.co_varnames[i])`.
        for i in 0..code.formalargcount() {
            locals[i] = Some(Hlvalue::Variable(Variable::named(&code.co_varnames[i])));
        }
        // upstream: `state = FrameState(locals, [], None, [], 0)`.
        let state = FrameState::new(locals, Vec::new(), None, Vec::new(), 0);
        // upstream: `initialblock = SpamBlock(state)`.
        let initialblock = SpamBlock::new(state);
        // upstream: `super().__init__(self._sanitize_funcname(func),
        //                             initialblock)`.
        let name = Self::sanitize_funcname(&func);
        let mut graph = FunctionGraph::new(name, initialblock.block.clone());
        // upstream: `self.func = func`. FunctionGraph already carries a
        // `func: Option<GraphFunc>` slot; mirror assignment there too so
        // downstream helpers (`FlowContext::new`) see the same object.
        graph.func = Some(func.clone());

        // upstream: `self.signature = code.signature` / `self.defaults = ...`.
        PyGraph {
            graph,
            signature: code.signature.clone(),
            defaults: func.defaults.clone(),
            func,
        }
    }

    /// RPython `PyGraph._sanitize_funcname(func)` (pygraph.py:24-33).
    ///
    /// Upstream folds `<`, `>`, `&`, `!` to `_` so the generated name
    /// is safe for identifier-shaped use downstream (graph dumps, C
    /// symbol generation). If `func.class_` is present, prefix the
    /// function name with `class_.__name__`.
    pub fn sanitize_funcname(func: &GraphFunc) -> String {
        let mut name = if let Some(class_) = &func.class_ {
            let class_name = class_
                .qualname()
                .rsplit('.')
                .next()
                .unwrap_or(class_.qualname());
            format!("{class_name}.{}", func.name)
        } else {
            func.name.clone()
        };
        for c in ['<', '>', '&', '!'] {
            name = name.replace(c, "_");
        }
        name
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::{ConstValue, Constant};
    use super::*;

    fn empty_globals() -> Constant {
        Constant::new(ConstValue::Dict(Default::default()))
    }

    fn make_host_code(name: &str, nlocals: u32, argcount: u32, varnames: &[&str]) -> HostCode {
        HostCode {
            co_name: name.to_string(),
            co_filename: "<test>".to_string(),
            co_firstlineno: 1,
            co_nlocals: nlocals,
            co_argcount: argcount,
            co_stacksize: 0,
            co_flags: 0,
            co_code: rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
            co_varnames: varnames.iter().map(|s| s.to_string()).collect(),
            co_freevars: Vec::new(),
            co_cellvars: Vec::new(),
            consts: Vec::new(),
            names: Vec::new(),
            co_lnotab: Vec::new(),
            exceptiontable: Vec::new().into_boxed_slice(),
            signature: Signature::new(varnames.iter().map(|s| s.to_string()).collect(), None, None),
        }
    }

    #[test]
    fn pygraph_initial_state_matches_upstream() {
        // upstream: `def f(a, b): return a + b`.
        let code = make_host_code("f", 2, 2, &["a", "b"]);
        let func = GraphFunc::new("f", empty_globals());
        let pygraph = PyGraph::new(func, &code);

        assert_eq!(pygraph.graph.name, "f");
        assert_eq!(pygraph.defaults.len(), 0);
        assert_eq!(pygraph.signature.num_argnames(), 2);
        // startblock.inputargs carries a Variable per formal arg
        // (no None slots leak through).
        let inputargs = pygraph.graph.startblock.borrow().inputargs.clone();
        assert_eq!(inputargs.len(), 2);
        for arg in inputargs {
            assert!(matches!(arg, Hlvalue::Variable(_)));
        }
    }

    #[test]
    fn pygraph_handles_extra_locals_as_none() {
        // upstream: `def f(a): x = 1; return x` → `co_nlocals = 2`,
        // formalargcount = 1.
        let code = make_host_code("f", 2, 1, &["a", "x"]);
        let func = GraphFunc::new("f", empty_globals());
        let _pygraph = PyGraph::new(func, &code);
        // No panics; PyGraph construction covers the None-filled tail
        // path. Deeper framestate assertions live in the objspace
        // integration tests.
    }

    #[test]
    fn sanitize_funcname_replaces_special_chars() {
        // upstream comment notes `<lambda>`-style names come in via
        // `CallableFactory.pycall`.
        let mut func = GraphFunc::new("<lambda>", empty_globals());
        func.name = "<lambda>".into();
        assert_eq!(PyGraph::sanitize_funcname(&func), "_lambda_");

        let func = GraphFunc::new("my&fn!", empty_globals());
        assert_eq!(PyGraph::sanitize_funcname(&func), "my_fn_");

        let func = GraphFunc::new("plain_name", empty_globals());
        assert_eq!(PyGraph::sanitize_funcname(&func), "plain_name");
    }

    #[test]
    fn sanitize_funcname_prefixes_method_owner_name() {
        let mut func = GraphFunc::new("method", empty_globals());
        func.class_ = Some(super::super::model::HostObject::new_class(
            "pkg.Owner",
            vec![],
        ));
        assert_eq!(PyGraph::sanitize_funcname(&func), "Owner.method");
    }
}
