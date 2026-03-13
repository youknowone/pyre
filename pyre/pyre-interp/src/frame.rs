//! PyFrame — execution frame for Python bytecode.
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::collections::HashMap;
use std::rc::Rc;

use pyre_bytecode::CodeObject;
use pyre_bytecode::ConstantData;
use pyre_object::pyobject::PyDisplay;
use pyre_object::*;

/// Execution frame for a single Python code block.
///
/// In Phase 3, `locals_w` and `value_stack_w` become virtualizable arrays
/// via `VirtualizableInfo::add_array_field`.
pub struct PyFrame {
    /// The code object being executed (shared via Rc for borrow-safe access).
    pub code: Rc<CodeObject>,
    /// Local variables (fast locals), indexed by varname slot.
    pub locals_w: Vec<PyObjectRef>,
    /// Operand stack for bytecode execution.
    pub value_stack_w: Vec<PyObjectRef>,
    /// Current stack depth (number of live values on value_stack_w).
    pub stack_depth: usize,
    /// Index of the next instruction to execute.
    pub next_instr: usize,
    /// Name-based local/global namespace (for module-level code).
    pub namespace: HashMap<String, PyObjectRef>,
}

impl PyFrame {
    /// Create a new frame for executing a code object.
    ///
    /// Pre-populates the namespace with built-in functions (`print`, `len`, etc.).
    pub fn new(code: CodeObject) -> Self {
        let num_locals = code.varnames.len();
        let max_stack = code.max_stackdepth as usize;
        let mut namespace = HashMap::new();

        // Install builtins into the namespace
        namespace.insert(
            "print".to_string(),
            w_builtin_func_new("print", builtin_print),
        );

        PyFrame {
            code: Rc::new(code),
            locals_w: vec![PY_NULL; num_locals],
            value_stack_w: vec![PY_NULL; max_stack],
            stack_depth: 0,
            next_instr: 0,
            namespace,
        }
    }

    // ── Stack operations ──────────────────────────────────────────────

    #[inline]
    pub fn push(&mut self, value: PyObjectRef) {
        self.value_stack_w[self.stack_depth] = value;
        self.stack_depth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        self.stack_depth -= 1;
        self.value_stack_w[self.stack_depth]
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        self.value_stack_w[self.stack_depth - 1]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        self.value_stack_w[self.stack_depth - 1 - depth]
    }

    // ── Constant loading ──────────────────────────────────────────────

    /// Convert a RustPython ConstantData to a PyObjectRef.
    pub fn load_const(constant: &ConstantData) -> PyObjectRef {
        match constant {
            ConstantData::Integer { value } => {
                use num_traits::ToPrimitive;
                let v = value.to_i64().unwrap_or(0);
                w_int_new(v)
            }
            ConstantData::Boolean { value } => w_bool_from(*value),
            ConstantData::None => w_none(),
            _ => {
                // Other constant types not yet supported
                w_none()
            }
        }
    }
}

// ── Built-in function implementations ─────────────────────────────────

/// `print(*args)` — write space-separated str representations to stdout.
fn builtin_print(args: &[PyObjectRef]) -> PyObjectRef {
    let parts: Vec<String> = args
        .iter()
        .map(|&obj| format!("{}", PyDisplay(obj)))
        .collect();
    println!("{}", parts.join(" "));
    w_none()
}
