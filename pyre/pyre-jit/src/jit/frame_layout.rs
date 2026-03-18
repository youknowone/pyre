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
    locals_cells_stack_w: PyObjectArray,
    #[allow(dead_code)]
    valuestackdepth: usize,
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

/// Byte offset of `valuestackdepth` in `PyFrame`.
pub const PYFRAME_VALUESTACKDEPTH_OFFSET: usize =
    std::mem::offset_of!(PyFrameLayout, valuestackdepth);

/// Byte offset of `locals_cells_stack_w` in `PyFrame`.
pub const PYFRAME_LOCALS_CELLS_STACK_OFFSET: usize =
    std::mem::offset_of!(PyFrameLayout, locals_cells_stack_w);

/// Byte offset of `namespace` in `PyFrame`.
pub const PYFRAME_NAMESPACE_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, namespace);

/// Byte offset of `code` in `PyFrame`.
pub const PYFRAME_CODE_OFFSET: usize = std::mem::offset_of!(PyFrameLayout, code);

// Backward-compat aliases used by JIT descriptor helpers.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = PYFRAME_VALUESTACKDEPTH_OFFSET;
pub const PYFRAME_LOCALS_OFFSET: usize = PYFRAME_LOCALS_CELLS_STACK_OFFSET;

/// Build the virtualizable layout description for `PyFrame`.
///
/// PyPy interp_jit.py:25-32 declares these virtualizable fields:
///   last_instr, pycode, valuestackdepth, locals_cells_stack_w[*],
///   debugdata, last_exception, lastblock, w_globals
///
/// Our subset (pyre doesn't have debugdata/last_exception/lastblock):
///   next_instr, valuestackdepth, locals_cells_stack_w[*]
pub fn build_pyframe_virtualizable_info() -> VirtualizableInfo {
    let mut info = VirtualizableInfo::new(PYFRAME_VABLE_TOKEN_OFFSET);
    // PyPy: last_instr
    info.add_field("next_instr", Type::Int, PYFRAME_NEXT_INSTR_OFFSET);
    // PyPy: valuestackdepth
    info.add_field(
        "valuestackdepth",
        Type::Int,
        PYFRAME_VALUESTACKDEPTH_OFFSET,
    );
    // PyPy: locals_cells_stack_w[*] — single unified array
    info.add_array_field(
        "locals_cells_stack_w",
        Type::Ref,
        PYFRAME_LOCALS_CELLS_STACK_OFFSET,
    );
    info
}
