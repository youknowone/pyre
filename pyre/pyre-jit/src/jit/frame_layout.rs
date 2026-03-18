use majit_ir::Type;
use majit_meta::virtualizable::VirtualizableInfo;
use pyre_bytecode::CodeObject;
use pyre_runtime::{PyExecutionContext, PyNamespace, PyObjectArray};

/// Shared PyFrame layout contract used by the interpreter and tracer.
///
/// This mirrors `pyre-interp::frame::PyFrame` exactly so both crates can
/// compute the same virtualizable offsets without introducing a reverse
/// dependency from `pyre-jit` to `pyre-interp`.
#[repr(C)]
struct PyFrameLayout {
    #[allow(dead_code)]
    execution_context: *const PyExecutionContext,
    #[allow(dead_code)]
    code: *const CodeObject,
    #[allow(dead_code)]
    locals_w: PyObjectArray,
    #[allow(dead_code)]
    value_stack_w: PyObjectArray,
    #[allow(dead_code)]
    stack_depth: usize,
    #[allow(dead_code)]
    next_instr: usize,
    #[allow(dead_code)]
    namespace: *mut PyNamespace,
    #[allow(dead_code)]
    vable_token: usize,
}

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, vable_token);

/// Byte offset of `next_instr` in `PyFrame`.
pub const PYFRAME_NEXT_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, next_instr);

/// Byte offset of `stack_depth` in `PyFrame`.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, stack_depth);

/// Byte offset of `locals_w` in `PyFrame`.
pub const PYFRAME_LOCALS_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, locals_w);

/// Byte offset of `value_stack_w` in `PyFrame`.
pub const PYFRAME_STACK_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, value_stack_w);

/// Byte offset of `namespace` in `PyFrame`.
pub const PYFRAME_NAMESPACE_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, namespace);

/// Byte offset of `code` in `PyFrame`.
pub const PYFRAME_CODE_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, code);

/// Build the virtualizable layout description for `PyFrame`.
///
/// PyPy interp_jit.py:25-32 declares these virtualizable fields:
///   last_instr, pycode, valuestackdepth, locals_cells_stack_w[*],
///   debugdata, last_exception, lastblock, w_globals
///
/// Our subset (pyre doesn't have debugdata/last_exception/lastblock):
///   next_instr, stack_depth, locals_w[*], value_stack_w[*]
pub fn build_pyframe_virtualizable_info() -> VirtualizableInfo {
    let mut info = VirtualizableInfo::new(PYFRAME_VABLE_TOKEN_OFFSET);
    // PyPy: last_instr
    info.add_field("next_instr", Type::Int, PYFRAME_NEXT_INSTR_OFFSET);
    // PyPy: valuestackdepth
    info.add_field("stack_depth", Type::Int, PYFRAME_STACK_DEPTH_OFFSET);
    // PyPy: locals_cells_stack_w[*] (pyre splits into locals_w + value_stack_w)
    info.add_array_field("locals_w", Type::Ref, PYFRAME_LOCALS_OFFSET);
    info.add_array_field("value_stack_w", Type::Ref, PYFRAME_STACK_OFFSET);
    info
}
