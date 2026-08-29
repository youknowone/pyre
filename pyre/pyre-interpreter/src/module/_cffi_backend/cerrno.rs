//! Saved foreign-call error state — PyPy:
//! `pypy/module/_cffi_backend/cerrno.py`.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::cell::Cell;

thread_local! {
    /// `rthread.tlfield_alt_errno`: error state belongs to the calling thread.
    static SAVED_ALT_ERRNO: Cell<i32> = const { Cell::new(0) };
}

/// `cerrno.py get_errno`.
pub fn get_errno(_args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    Ok(pyre_object::w_int_new(
        SAVED_ALT_ERRNO.with(Cell::get) as i64
    ))
}

/// `cerrno.py set_errno`.
pub fn set_errno(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let value = crate::baseobjspace::int_w(args[0])? as i32;
    SAVED_ALT_ERRNO.with(|saved| saved.set(value));
    Ok(pyre_object::w_none())
}

/// `rposix._errno_before(RFFI_ALT_ERRNO)`.
pub fn errno_before() {
    let value = SAVED_ALT_ERRNO.with(Cell::get);
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    rustpython_host_env::os::set_errno(value);
    #[cfg(not(all(feature = "host_env", not(feature = "sandbox"))))]
    let _ = value;
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    rustpython_host_env::windows::set_last_error(rustpython_host_env::ctypes::get_last_error());
}

/// `rposix._errno_after(RFFI_ALT_ERRNO)`.
pub fn errno_after() {
    // `_errno_after` reads LastError before errno because reading errno may
    // itself overwrite the Windows error state.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    let last_error = rustpython_host_env::windows::get_last_error();
    let value = crate::builtins::crt_errno();
    SAVED_ALT_ERRNO.with(|saved| saved.set(value));
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    let _ = rustpython_host_env::ctypes::set_last_error(last_error);
}

/// `cerrno.py getwinerror`.
#[cfg(windows)]
pub fn getwinerror(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let code = match args.first() {
        Some(&w_code) => crate::baseobjspace::int_w(w_code)? as i32,
        None => -1,
    };
    let code = if code == -1 {
        #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
        {
            rustpython_host_env::ctypes::get_last_error() as i32
        }
        #[cfg(not(all(feature = "host_env", not(feature = "sandbox"))))]
        {
            0
        }
    } else {
        code
    };
    let roots = pyre_object::gc_roots::push_roots();
    let code_slot = roots.base();
    let _ = roots.pin_root(pyre_object::w_int_new(code as i64));
    let message = crate::PyError::win32_strerror(code);
    // `w_tuple_new` allocates, so the message needs a root of its own rather
    // than only the Rust local.
    let _ = roots.pin_root(pyre_object::w_str_new(&message));
    Ok(pyre_object::w_tuple_new(vec![
        roots.get(code_slot),
        roots.get(code_slot + 1),
    ]))
}
