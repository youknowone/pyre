//! The bound-method and code-object entry points -- PyPy
//! `cpyext/funcobject.py`.

use super::object::{argument, result};
use super::pyobject::{self, CPyObject};
use std::ffi::{c_char, c_int};

/// `funcobject.py PyMethod_New(func, self)` — bind `receiver` to
/// `function`, the way an attribute lookup on an instance does.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_New(
    function: *mut CPyObject,
    receiver: *mut CPyObject,
) -> *mut CPyObject {
    super::object::realize_all([function, receiver]);
    let (Some(function), Some(receiver)) = (argument(function), argument(receiver)) else {
        return std::ptr::null_mut();
    };
    result(Ok(pyre_object::w_method_new(
        function,
        receiver,
        pyre_object::PY_NULL,
    )))
}

/// The member `reader` names on `method`, borrowed, or NULL with a
/// `SystemError` when `method` is not a bound method.
///
/// Borrowed is what both readers answer with: the member is reachable through
/// the method for as long as the caller holds it.
unsafe fn method_member(
    method: *mut CPyObject,
    reader: unsafe fn(pyre_object::PyObjectRef) -> pyre_object::PyObjectRef,
) -> *mut CPyObject {
    let Some(object) = argument(method) else {
        return std::ptr::null_mut();
    };
    if !unsafe { pyre_object::function::is_method(object) } {
        unsafe { super::pyerrors::PyErr_BadInternalCall() };
        return std::ptr::null_mut();
    }
    pyobject::borrow_from(method, unsafe { reader(object) })
}

/// `funcobject.py PyMethod_Function(method)` — the callable the binding
/// wraps, borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_Function(method: *mut CPyObject) -> *mut CPyObject {
    unsafe { method_member(method, pyre_object::function::w_method_get_func) }
}

/// `funcobject.py PyMethod_Self(method)` — the receiver the binding
/// carries, borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyMethod_Self(method: *mut CPyObject) -> *mut CPyObject {
    unsafe { method_member(method, pyre_object::function::w_method_get_self) }
}

// ── code objects ────────────────────────────────────────────────────────

/// `RESUME 0; LOAD_COMMON_CONSTANT AssertionError; RAISE_VARARGS 1` — the
/// body `PyCode_NewEmpty` gives the code object it builds.
///
/// The opcodes are named rather than spelled as the numbers `codeobject.c
/// assert0` uses, so a renumbering is a build error rather than a body that
/// decodes to something else.
fn assert0() -> [u8; 6] {
    use crate::bytecode::{CommonConstant, Opcode, RaiseKind};
    [
        Opcode::Resume as u8,
        // `ResumeLocation::AtFuncStart`, which is not reachable by name here.
        0,
        Opcode::LoadCommonConstant as u8,
        CommonConstant::AssertionError as u8,
        Opcode::RaiseVarargs as u8,
        RaiseKind::Raise as u8,
    ]
}

/// One line-table entry covering all three units of [`assert0`], carrying the
/// line and no columns, as `codeobject.c linetable` spells it.
const ASSERT0_LINETABLE: [u8; 2] = [(1 << 7) | (13 << 3) | (3 - 1), 0];

/// Hand `code.__new__`'s argument list to the interpreter's constructor.
///
/// The C entry points name their arguments in a different order and let the
/// caller's `nlocals` disagree with `varnames`, which `codeobject.c` derives
/// rather than reads; both are reconciled here so the interpreter constructor
/// sees only what it accepts.
unsafe fn build_code(args: &[pyre_object::PyObjectRef]) -> *mut CPyObject {
    result(unsafe { crate::pycode::code_new(args) })
}

/// `funcobject.py:198 PyCode_NewEmpty(filename, funcname, firstlineno)` — a
/// code object with the given source location and a body that raises
/// `AssertionError`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyCode_NewEmpty(
    filename: *const c_char,
    funcname: *const c_char,
    firstlineno: c_int,
) -> *mut CPyObject {
    // The two names are decoded as `codeobject.c` decodes them: the function
    // name as UTF-8, the path through the filesystem codec, so a path this
    // runtime cannot spell as UTF-8 still names a file.
    let name = unsafe { super::unicodeobject::PyUnicode_FromString(funcname) };
    let path = unsafe { super::unicodeobject::PyUnicode_DecodeFSDefault(filename) };
    let code = match (argument(name), argument(path)) {
        (Some(name), Some(path)) => {
            let empty_tuple = pyre_object::w_tuple_new(Vec::new());
            let empty_bytes = pyre_object::bytesobject::w_bytes_from_bytes(&[]);
            // Nothing between here and the call collects: an int, a tuple, a
            // bytes and a str are all allocated outside the nursery.
            result(unsafe {
                crate::pycode::code_new(&[
                    pyre_object::PY_NULL,
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(0),
                    pyre_object::w_int_new(1),
                    pyre_object::w_int_new(0),
                    pyre_object::bytesobject::w_bytes_from_bytes(&assert0()),
                    empty_tuple,
                    empty_tuple,
                    empty_tuple,
                    path,
                    name,
                    name,
                    pyre_object::w_int_new(firstlineno as i64),
                    pyre_object::bytesobject::w_bytes_from_bytes(&ASSERT0_LINETABLE),
                    empty_bytes,
                ])
            })
        }
        _ => std::ptr::null_mut(),
    };
    for reference in [name, path] {
        if !reference.is_null() {
            unsafe { pyobject::decref(reference) };
        }
    }
    code
}

/// `funcobject.py:161 PyUnstable_Code_NewWithPosOnlyArgs(...)` — a code object
/// built from every field, as `code.__new__` takes them.
///
/// The C argument order puts the free and cell names before the filename;
/// the interpreter constructor takes them last, and derives `nlocals` from
/// `varnames` rather than reading the one the caller passed, as
/// `codeobject.c` does.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnstable_Code_NewWithPosOnlyArgs(
    argcount: c_int,
    posonlyargcount: c_int,
    kwonlyargcount: c_int,
    _nlocals: c_int,
    stacksize: c_int,
    flags: c_int,
    code: *mut CPyObject,
    consts: *mut CPyObject,
    names: *mut CPyObject,
    varnames: *mut CPyObject,
    freevars: *mut CPyObject,
    cellvars: *mut CPyObject,
    filename: *mut CPyObject,
    name: *mut CPyObject,
    qualname: *mut CPyObject,
    firstlineno: c_int,
    linetable: *mut CPyObject,
    exceptiontable: *mut CPyObject,
) -> *mut CPyObject {
    let objects = [
        code,
        consts,
        names,
        varnames,
        freevars,
        cellvars,
        filename,
        name,
        qualname,
        linetable,
        exceptiontable,
    ];
    super::object::realize_all(objects);
    let mut arguments = [pyre_object::PY_NULL; 11];
    for (slot, &raw) in arguments.iter_mut().zip(objects.iter()) {
        match argument(raw) {
            Some(object) => *slot = object,
            None => return std::ptr::null_mut(),
        }
    }
    let [
        code,
        consts,
        names,
        varnames,
        freevars,
        cellvars,
        filename,
        name,
        qualname,
        linetable,
        exceptiontable,
    ] = arguments;
    // A name list that is not a tuple is a caller error, which the C entry
    // point reports the way every other bad argument to one is reported.
    for list in [names, varnames, freevars, cellvars] {
        if !unsafe { pyre_object::is_tuple(list) } {
            unsafe { super::pyerrors::PyErr_BadInternalCall() };
            return std::ptr::null_mut();
        }
    }
    let nlocals = unsafe { pyre_object::tupleobject::w_tuple_len(varnames) };
    result(unsafe {
        crate::pycode::code_new(&[
            pyre_object::PY_NULL,
            pyre_object::w_int_new(argcount as i64),
            pyre_object::w_int_new(posonlyargcount as i64),
            pyre_object::w_int_new(kwonlyargcount as i64),
            pyre_object::w_int_new(nlocals as i64),
            pyre_object::w_int_new(stacksize as i64),
            pyre_object::w_int_new(flags as i64),
            code,
            consts,
            names,
            varnames,
            filename,
            name,
            qualname,
            pyre_object::w_int_new(firstlineno as i64),
            linetable,
            exceptiontable,
            freevars,
            cellvars,
        ])
    })
}

/// `funcobject.py:134 PyUnstable_Code_New(...)` — the same with no
/// positional-only arguments.
#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyUnstable_Code_New(
    argcount: c_int,
    kwonlyargcount: c_int,
    nlocals: c_int,
    stacksize: c_int,
    flags: c_int,
    code: *mut CPyObject,
    consts: *mut CPyObject,
    names: *mut CPyObject,
    varnames: *mut CPyObject,
    freevars: *mut CPyObject,
    cellvars: *mut CPyObject,
    filename: *mut CPyObject,
    name: *mut CPyObject,
    qualname: *mut CPyObject,
    firstlineno: c_int,
    linetable: *mut CPyObject,
    exceptiontable: *mut CPyObject,
) -> *mut CPyObject {
    unsafe {
        PyUnstable_Code_NewWithPosOnlyArgs(
            argcount,
            0,
            kwonlyargcount,
            nlocals,
            stacksize,
            flags,
            code,
            consts,
            names,
            varnames,
            freevars,
            cellvars,
            filename,
            name,
            qualname,
            firstlineno,
            linetable,
            exceptiontable,
        )
    }
}
