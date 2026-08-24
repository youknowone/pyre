//! The entry points whose bodies are C rather than Rust.
//!
//! `PyArg_ParseTuple`, `Py_BuildValue`, `PyErr_Format` and the rest of the
//! variadic surface take a `va_list`, so they are C functions with C bodies
//! (`src/getargs.c` and its peers, which `build.rs` compiles); upstream builds
//! the same ones the same way, in `pypy/module/cpyext/src/`.
//!
//! Nothing on this side calls one -- the signatures that matter are the
//! headers' -- so each is declared with no arguments and only ever has its
//! address taken.  That address is the whole point: an archive member is
//! pulled into the binary by a reference to a symbol it defines, and
//! [`ensure_linked`] is that reference, the same job the per-module
//! `ensure_linked` does for the entry points written in Rust.

unsafe extern "C" {
    // `src/getargs.c`
    fn PyArg_Parse();
    fn PyArg_ParseTuple();
    fn PyArg_ParseTupleAndKeywords();
    fn PyArg_UnpackTuple();
    fn _PyArg_CheckPositional();
    // `src/modsupport.c`
    fn Py_BuildValue();
    fn Py_VaBuildValue();
    // `src/abstract.c`
    fn PyObject_CallFunction();
    fn PyObject_CallFunctionObjArgs();
    fn PyObject_CallMethod();
    fn PyObject_CallMethodObjArgs();
    // `src/mysnprintf.c`
    fn PyOS_snprintf();
    fn PyOS_vsnprintf();
    // `src/unicodeobject.c`
    fn PyUnicode_FromFormat();
    fn PyUnicode_FromFormatV();
    // `src/pyerrors.c`
    fn PyErr_Format();
    fn PyErr_FormatUnraisable();
}

pub(super) fn ensure_linked() {
    for entry in [
        PyArg_Parse as *const (),
        PyArg_ParseTuple as *const (),
        PyArg_ParseTupleAndKeywords as *const (),
        PyArg_UnpackTuple as *const (),
        _PyArg_CheckPositional as *const (),
        Py_BuildValue as *const (),
        Py_VaBuildValue as *const (),
        PyObject_CallFunction as *const (),
        PyObject_CallFunctionObjArgs as *const (),
        PyObject_CallMethod as *const (),
        PyObject_CallMethodObjArgs as *const (),
        PyOS_snprintf as *const (),
        PyOS_vsnprintf as *const (),
        PyUnicode_FromFormat as *const (),
        PyUnicode_FromFormatV as *const (),
        PyErr_Format as *const (),
        PyErr_FormatUnraisable as *const (),
    ] {
        std::hint::black_box(entry);
    }
}
