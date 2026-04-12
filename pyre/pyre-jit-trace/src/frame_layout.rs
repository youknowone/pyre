use majit_metainterp::virtualizable::VirtualizableInfo;
use pyre_interpreter::CodeObject;
use pyre_interpreter::{PyExecutionContext, PyNamespace, PyObjectArray};

/// Shared PyFrame layout contract used by the interpreter and tracer.
///
/// This mirrors `pyre-interpreter::pyframe::PyFrame` exactly so both crates can
/// compute the same virtualizable offsets without introducing a reverse
/// dependency from `pyre-jit` to `pyre-interpreter`.
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
/// virtualizable.py:28, 71-79: uses `cpu.fielddescrof(VTYPE, name)` to
/// create canonical descriptors. Pyre equivalent: supply canonical
/// descriptors from `PYFRAME_DESCR_GROUP` via `set_canonical_descriptors`.
///
/// Static field order (from virtualizable! macro): next_instr, code,
/// valuestackdepth, namespace. Array: locals_cells_stack_w.
pub fn build_pyframe_virtualizable_info() -> VirtualizableInfo {
    use crate::descr;
    let mut info = crate::virtualizable_gen::build_virtualizable_info();
    // virtualizable.py:28, 71-79: cpu.fielddescrof(VTYPE, name) for each field.
    // Descriptor identity from PYFRAME_DESCR_GROUP matches any other code path
    // that resolves the same PyFrame field, so optimizer lookups see the same
    // descriptor everywhere (RPython's cpu.fielddescrof cache guarantee).
    info.set_canonical_descriptors(
        descr::pyframe_size_descr_canonical(),
        // virtualizable.py:28: self.vable_token_descr
        descr::pyframe_vable_token_descr(),
        // virtualizable.py:71-72: self.static_field_descrs
        // Order must match virtualizable! macro declaration:
        // [next_instr, code, valuestackdepth, namespace]
        vec![
            descr::pyframe_next_instr_descr(),
            descr::pyframe_code_descr(),
            descr::pyframe_stack_depth_descr(),
            descr::pyframe_namespace_descr(),
        ],
        // virtualizable.py:73-74: self.array_field_descrs
        // EmbeddedArray: descriptor offset = field_offset + ptr_offset
        // so GETFIELD_GC_R reads the data pointer, not the container.
        vec![descr::pyframe_locals_cells_stack_descr()],
        // virtualizable.py:58: self.array_descrs = [cpu.arraydescrof(...)]
        vec![descr::pyframe_locals_array_descr()],
    );
    info
}
