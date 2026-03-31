//! pyre-interpreter — the Python interpreter.
//!
//! PyPy equivalent: pypy/interpreter/
//!
//! Contains the object space (baseobjspace.rs), bytecode evaluation (eval.rs),
//! frame management (pyframe.rs), function call dispatch (call.rs),
//! import machinery (importing.rs), builtin functions (builtins.rs),
//! type definitions (typedef.rs), and builtin modules (module/).

// ── Bytecode / compiler re-exports (was pyre-bytecode) ──
pub mod compile;
pub use compile::*;

// ── Core interpreter modules ──
pub mod baseobjspace;
pub mod builtins;
pub mod display;
pub mod error;
pub mod executioncontext;
pub mod frame_array;
pub mod function;
pub mod gateway;
pub mod opcode_ops;
pub mod pycode;
pub mod pyopcode;
pub mod runtime_ops;
pub mod shared_opcode;
pub mod type_methods;
pub mod typedef;

// ── Execution and import modules ──
pub mod call;
pub mod eval;
pub mod importing;
pub mod module;
pub mod pyframe;

// ── Re-exports ──
pub use baseobjspace::*;
pub use builtins::*;
pub use display::*;
pub use error::*;
pub use executioncontext::*;
pub use frame_array::*;
pub use function::*;
pub use gateway::{
    BUILTIN_CODE_TYPE, BuiltinCode, BuiltinCodeFn, builtin_code_get, builtin_code_name,
    builtin_code_new, is_builtin_code,
};
pub use malachite_bigint::BigInt as PyBigInt;
pub use opcode_ops::*;
pub use pycode::*;
pub use pyframe::*;
pub use pyopcode::*;
pub use runtime_ops::*;
pub use shared_opcode::*;

// baseobjspace call helpers are re-exported from `baseobjspace`.
