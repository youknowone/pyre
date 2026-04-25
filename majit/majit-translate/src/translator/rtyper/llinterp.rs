//! Port of `rpython/rtyper/llinterp.py`.
//!
//! Upstream is 1477 LOC of two top-level types — `LLException` /
//! `LLFatalError` exceptions, plus the `LLInterpreter` and its
//! `LLFrame` opcode-dispatch loop. Driver task `task_llinterpret_lltype`
//! (`driver.py:543-555`) is the only consumer of `LLInterpreter` and
//! uses only:
//!
//! * Constructor: `LLInterpreter(translator.rtyper)` (`:67-82`).
//! * `eval_graph(graph, args)` (`:84-…`): walks a `FunctionGraph`,
//!   running each `SpaceOperation` through `op_<opname>` handlers.
//!
//! The full interpreter requires `rpython.rtyper.lltypesystem.{lltype,
//! llmemory, lloperation, llheap}` plus the `LLFrame` opcode table —
//! none of which are ported. This file ships the structural shell:
//! `LLException` / `LLFatalError` / `LLAssertFailure` types, the
//! `LLInterpreter::new(rtyper, tracing, exc_data_ptr)` constructor with
//! upstream slot layout, and `eval_graph` returning [`TaskError`]
//! citing the upstream line.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::translator::rtyper::rtyper::RPythonTyper;
use crate::translator::tool::taskengine::TaskError;

/// Port of upstream `class LLException(Exception)` at `:28-54`.
///
/// `error_value` defaults to upstream's `UNDEFINED_ERROR_VALUE`
/// sentinel — the local port encodes the sentinel as `None`.
#[derive(Debug, Clone)]
pub struct LLException {
    pub args: Vec<Rc<dyn Any>>,
    pub error_value: Option<Rc<dyn Any>>,
}

impl LLException {
    /// Upstream `__init__(*args, error_value=UNDEFINED_ERROR_VALUE)` at
    /// `:35-40`.
    pub fn new(args: Vec<Rc<dyn Any>>, error_value: Option<Rc<dyn Any>>) -> Self {
        Self { args, error_value }
    }
}

impl std::fmt::Display for LLException {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<LLException>")
    }
}

impl std::error::Error for LLException {}

/// Port of upstream `class LLFatalError(Exception)` at `:56-58`.
#[derive(Debug, Clone)]
pub struct LLFatalError(pub Vec<String>);

impl std::fmt::Display for LLFatalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.join(": "))
    }
}

impl std::error::Error for LLFatalError {}

/// Port of upstream `class LLAssertFailure(Exception)` at `:60-61`.
#[derive(Debug, Clone)]
pub struct LLAssertFailure;

impl std::fmt::Display for LLAssertFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("LLAssertFailure")
    }
}

impl std::error::Error for LLAssertFailure {}

/// Port of upstream `class LLInterpreter(object)` at `:67-…`.
///
/// Slot layout matches upstream's `__init__` (`:72-82`). Bodies are
/// stubbed; `eval_graph` returns [`TaskError`] until the opcode-handler
/// table lands.
pub struct LLInterpreter {
    /// Upstream `self.bindings = {}` at `:73`. Maps each Variable to
    /// its concrete value during a frame's `eval()`. The local port
    /// keeps the shape opaque (`Rc<dyn Any>`) until lltype lands.
    pub bindings: RefCell<Vec<(Rc<dyn Any>, Rc<dyn Any>)>>,
    /// Upstream `self.typer = typer` at `:74`.
    pub typer: Rc<RPythonTyper>,
    /// Upstream `self.exc_data_ptr = exc_data_ptr` at `:77`.
    pub exc_data_ptr: Option<Rc<dyn Any>>,
    /// Upstream `self.frame_stack = []` at `:78`.
    pub frame_stack: RefCell<Vec<Rc<dyn Any>>>,
    /// Upstream `self.tracer = None` (or `Tracer()`) at `:79-82`.
    pub tracer: Option<Rc<dyn Any>>,
}

impl LLInterpreter {
    /// Upstream `__init__(self, typer, tracing=True, exc_data_ptr=None)`
    /// at `:72-82`.
    pub fn new(typer: Rc<RPythonTyper>, tracing: bool, exc_data_ptr: Option<Rc<dyn Any>>) -> Self {
        Self {
            bindings: RefCell::new(Vec::new()),
            typer,
            exc_data_ptr,
            frame_stack: RefCell::new(Vec::new()),
            // Upstream `:81-82`: `if tracing: self.tracer = Tracer()`.
            // The Tracer port is not landed yet — keep the slot but
            // never populate it.
            tracer: if tracing { None } else { None },
        }
    }

    /// Upstream `eval_graph(self, graph, args=(), recursive=False)` at
    /// `:84-…`.
    pub fn eval_graph(
        &self,
        _graph: Rc<dyn Any>,
        _args: Vec<Rc<dyn Any>>,
        _recursive: bool,
    ) -> Result<Rc<dyn Any>, TaskError> {
        Err(TaskError {
            message: "llinterp.py:84 LLInterpreter.eval_graph — leaf LLFrame opcode-dispatch loop / lltype / llheap not yet ported".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::translator::rtyper::rtyper::RPythonTyper;

    fn fixture_typer() -> Rc<RPythonTyper> {
        let ann = RPythonAnnotator::new(None, None, None, false);
        Rc::new(RPythonTyper::new(&ann))
    }

    #[test]
    fn llinterpreter_init_mirrors_upstream_slots() {
        // Upstream `:72-82`: `__init__(typer, tracing=True,
        // exc_data_ptr=None)` writes `bindings = {}`, `typer = typer`,
        // `exc_data_ptr = exc_data_ptr`, `frame_stack = []`, `tracer
        // = None or Tracer()`.
        let typer = fixture_typer();
        let interp = LLInterpreter::new(typer, true, None);
        assert!(interp.bindings.borrow().is_empty());
        assert!(interp.exc_data_ptr.is_none());
        assert!(interp.frame_stack.borrow().is_empty());
    }

    #[test]
    fn eval_graph_returns_task_error_until_opcode_table_lands() {
        // Upstream `eval_graph` at `:84-…` walks a graph through
        // `LLFrame.eval()`. Until the opcode handler table lands, the
        // local port surfaces a TaskError citing `:84`.
        let interp = LLInterpreter::new(fixture_typer(), false, None);
        let err = interp
            .eval_graph(Rc::new(()) as Rc<dyn Any>, Vec::new(), false)
            .expect_err("must be DEFERRED");
        assert!(err.message.contains("llinterp.py:84"), "{}", err.message);
    }

    #[test]
    fn llexception_carries_upstream_argument_layout() {
        // Upstream `:28-54`: `LLException(*args, error_value=...)`. The
        // local port preserves `args` and `error_value` slots so tests
        // tracking `e.error_value` (e.g. test_exceptiontransform per
        // `:30-32`) can land verbatim later.
        let exc = LLException::new(vec![Rc::new("etype".to_string()) as Rc<dyn Any>], None);
        assert_eq!(exc.args.len(), 1);
        assert!(exc.error_value.is_none());
    }

    #[test]
    fn llfatalerror_renders_args_separated_by_colons() {
        // Upstream `:56-58`: `__str__ = ': '.join([str(x) for x in
        // self.args])`.
        let err = LLFatalError(vec!["one".to_string(), "two".to_string()]);
        assert_eq!(err.to_string(), "one: two");
    }
}
