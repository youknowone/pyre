//! `_winapi.Overlapped` — the record an asynchronous `ConnectNamedPipe`,
//! `ReadFile` or `WriteFile` hands back, and the three calls that make one.
//!
//! Distinct from `_overlapped.Overlapped`, which `asyncio`'s proactor drives
//! through a completion port: this one owns a manual-reset event of its own
//! and is waited on directly, which is what `multiprocessing.connection`'s
//! `PipeConnection` does.
//!
//! The OVERLAPPED record and the transfer buffer have to keep one address for
//! as long as the operation is in flight, so both live in a `Box`ed native
//! state the object owns rather than on the moving heap.  The read buffer in
//! particular is a plain `Vec`, not the `bytes` upstream reads straight into;
//! `getbuffer` makes the `bytes` out of it once the read has completed.

use pyre_object::{PY_NULL, PyObject, PyObjectRef};
use rustpython_host_env::winapi as host_winapi;
use std::sync::{Mutex, OnceLock};
use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::IO::OVERLAPPED;

use super::win32_err;

const ERROR_IO_PENDING: u32 = 997;
const ERROR_MORE_DATA: u32 = 234;
const ERROR_OPERATION_ABORTED: u32 = 995;
const ERROR_IO_INCOMPLETE: u32 = 996;
const ERROR_PIPE_CONNECTED: u32 = 535;
const ERROR_NOT_FOUND: u32 = 1168;

/// What the buffer holds, which is also what `getbuffer` may answer with.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Transfer {
    /// `ConnectNamedPipe`, which moves no bytes.
    None,
    Read,
    Write,
}

struct NativeOverlapped {
    overlapped: OVERLAPPED,
    handle: HANDLE,
    /// The operation has been started and not yet seen to finish.
    pending: bool,
    /// `GetOverlappedResult` has seen it finish, which is what releases the
    /// read buffer to `getbuffer`.
    completed: bool,
    transfer: Transfer,
    /// The read destination or the write source.  Written by the OS while the
    /// operation is in flight, so it is never reallocated between the call
    /// that starts one and the result that ends it.
    buffer: Vec<u8>,
}

// Windows completes the operation on a thread of its own, writing into the
// record and the buffer this state owns.  Access from Python stays serialized
// through the per-object mutex.
unsafe impl Send for NativeOverlapped {}

impl Drop for NativeOverlapped {
    /// The event belongs to the record, so it is closed wherever the record
    /// goes away — including the call that failed to start its operation and
    /// drops the state it had already built.
    fn drop(&mut self) {
        if !self.overlapped.hEvent.is_null() {
            let _ = host_winapi::close_handle(self.overlapped.hEvent);
            self.overlapped.hEvent = std::ptr::null_mut();
        }
    }
}

#[crate::pyre_class("_winapi.Overlapped")]
#[derive(Default)]
pub struct W_Overlapped {
    backend: *mut Mutex<NativeOverlapped>,
}

fn this(obj: PyObjectRef) -> Result<&'static mut W_Overlapped, crate::PyError> {
    W_Overlapped::from_obj(obj)
        .ok_or_else(|| crate::PyError::type_error("expected _winapi.Overlapped object"))
}

fn native(obj: PyObjectRef) -> Result<&'static Mutex<NativeOverlapped>, crate::PyError> {
    let this = this(obj)?;
    if this.backend.is_null() {
        return Err(crate::PyError::value_error("invalid overlapped object"));
    }
    Ok(unsafe { &*this.backend })
}

fn arg(args: &[PyObjectRef], index: usize, name: &str) -> Result<PyObjectRef, crate::PyError> {
    args.get(index)
        .copied()
        .ok_or_else(|| crate::PyError::type_error(format!("{name}() missing required argument")))
}

/// A fresh record over `handle`, with the manual-reset event the operation
/// signals on.
///
/// Returned as the native state alone: the object around it is only built once
/// the call it belongs to has been started, so a failed start has nothing to
/// finalize.
fn new_native(handle: HANDLE) -> Result<Box<Mutex<NativeOverlapped>>, crate::PyError> {
    let event = host_winapi::create_event_w(true, false, None).map_err(win32_err)?;
    let mut overlapped: OVERLAPPED = unsafe { core::mem::zeroed() };
    overlapped.hEvent = event;
    Ok(Box::new(Mutex::new(NativeOverlapped {
        overlapped,
        handle,
        pending: false,
        completed: false,
        transfer: Transfer::None,
        buffer: Vec::new(),
    })))
}

/// Wrap a started operation's native state in the Python object that owns it.
fn w_overlapped(backend: Box<Mutex<NativeOverlapped>>) -> PyObjectRef {
    W_Overlapped::allocate_stable(W_Overlapped {
        ob: PyObject::default(),
        backend: Box::into_raw(backend),
    })
}

// ── methods ──

/// `Overlapped.GetOverlappedResult(wait)` -> `(transferred, error)`
///
/// The result is asked for unconditionally rather than only while the
/// operation is pending: a call that completed inside `ReadFile` still has its
/// transfer count here, and that is what the caller reads it for.
fn overlapped_get_result(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = arg(args, 0, "GetOverlappedResult")?;
    let wait = crate::baseobjspace::is_true(arg(args, 1, "GetOverlappedResult")?)?;
    let record = native(obj)?;
    // A waiting call runs with the lock dropped.  It blocks until the
    // operation ends, and `cancel` — the one call that can end it early —
    // takes this same lock, so holding it here would put the wait beyond
    // reach of the only thing that could wake it.  The record is owned by the
    // object and never moves, so its address stays good while the guard is
    // gone; the kernel is writing into it over the same period either way.
    let (handle, overlapped) = {
        let state = record.lock().unwrap();
        (state.handle, std::ptr::addr_of!(state.overlapped))
    };
    let mut transferred: u32 = 0;
    let completed = {
        let _blocked = crate::module::thread::before_external_block();
        unsafe {
            windows_sys::Win32::System::IO::GetOverlappedResult(
                handle,
                overlapped,
                &mut transferred,
                i32::from(wait),
            )
        }
    };
    let error = if completed == 0 {
        unsafe { windows_sys::Win32::Foundation::GetLastError() }
    } else {
        0
    };
    let mut state = record.lock().unwrap();
    match error {
        0 | ERROR_MORE_DATA | ERROR_OPERATION_ABORTED => {
            state.completed = true;
            state.pending = false;
        }
        ERROR_IO_INCOMPLETE => {}
        _ => {
            state.pending = false;
            return Err(super::win32_code(error));
        }
    }
    // The read asked for a fixed number of bytes and may have been given
    // fewer, so the buffer is cut down to what arrived.
    if state.completed && state.transfer == Transfer::Read {
        state.buffer.truncate(transferred as usize);
    }
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::w_int_new(i64::from(transferred)),
        pyre_object::w_int_new(i64::from(error)),
    ]))
}

/// `Overlapped.getbuffer()` -> what a completed read produced, or `None` when
/// the operation moved no bytes into one.
fn overlapped_getbuffer(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = arg(args, 0, "getbuffer")?;
    let state = native(obj)?.lock().unwrap();
    if !state.completed {
        return Err(crate::PyError::value_error(
            "can't get read buffer before GetOverlappedResult() signals the operation completed",
        ));
    }
    if state.transfer != Transfer::Read {
        return Ok(pyre_object::w_none());
    }
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&state.buffer))
}

/// `Overlapped.cancel()`
///
/// An operation that finished between the caller's decision to cancel and the
/// call itself answers `ERROR_NOT_FOUND`, which is not a failure — there is
/// simply nothing left to cancel.
fn overlapped_cancel(args: &[PyObjectRef]) -> crate::PyResult {
    let obj = arg(args, 0, "cancel")?;
    let state = native(obj)?.lock().unwrap();
    if !state.pending {
        return Ok(pyre_object::w_none());
    }
    let cancelled = {
        let _blocked = crate::module::thread::before_external_block();
        unsafe { windows_sys::Win32::System::IO::CancelIoEx(state.handle, &state.overlapped) }
    };
    if cancelled == 0 {
        let error = unsafe { windows_sys::Win32::Foundation::GetLastError() };
        if error != ERROR_NOT_FOUND {
            return Err(super::win32_code(error));
        }
    }
    Ok(pyre_object::w_none())
}

fn overlapped_new(_args: &[PyObjectRef]) -> crate::PyResult {
    Err(crate::PyError::type_error(
        "cannot create '_winapi.Overlapped' instances",
    ))
}

fn init_overlapped_type(ns: PyObjectRef) {
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "__new__",
            crate::typedef::make_new_descr(overlapped_new),
        )
    };
    for (name, arity, function) in [
        (
            "GetOverlappedResult",
            2,
            overlapped_get_result as crate::BuiltinCodeFn,
        ),
        ("getbuffer", 1, overlapped_getbuffer),
        ("cancel", 1, overlapped_cancel),
    ] {
        unsafe {
            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, function, arity),
            )
        };
    }
    // `event` is the record's own `hEvent`, which a caller waits on; it is
    // read-only because the operation was started against that handle.
    unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
            ns,
            "event",
            crate::typedef::make_getset_descriptor_named(
                crate::make_builtin_function_with_arity(
                    "event",
                    |args: &[PyObjectRef]| -> crate::PyResult {
                        let state = native(arg(args, 1, "event")?)?.lock().unwrap();
                        Ok(super::w_handle(state.overlapped.hEvent))
                    },
                    2,
                ),
                "event",
            ),
        )
    };
}

static OVERLAPPED_RUNTIME_TYPE: OnceLock<usize> = OnceLock::new();

pub fn overlapped_type() -> PyObjectRef {
    *OVERLAPPED_RUNTIME_TYPE.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_winapi.Overlapped",
            init_overlapped_type,
            crate::typedef::w_object(),
            <W_Overlapped as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_Overlapped as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        unsafe { pyre_object::w_type_set_acceptable_as_base_class(tp, false) };
        tp as usize
    }) as PyObjectRef
}

/// Sweep the native record: an operation still in flight is cancelled and
/// waited out before the buffer it writes into goes away, and dropping the
/// state then closes its event.  The thread's last-error value is restored so
/// a collection does not change what the running code would read from
/// `GetLastError`.
pub unsafe fn w_overlapped_dealloc(obj: PyObjectRef) {
    let Some(this) = W_Overlapped::from_obj(obj) else {
        return;
    };
    if this.backend.is_null() {
        return;
    }
    let backend = unsafe { Box::from_raw(this.backend) };
    this.backend = std::ptr::null_mut();
    let old_error = host_winapi::get_last_error();
    {
        let mut state = backend.lock().unwrap();
        if state.pending {
            let mut transferred: u32 = 0;
            unsafe {
                windows_sys::Win32::System::IO::CancelIoEx(state.handle, &state.overlapped);
                windows_sys::Win32::System::IO::GetOverlappedResult(
                    state.handle,
                    &state.overlapped,
                    &mut transferred,
                    1,
                );
            }
            state.pending = false;
        }
    }
    drop(backend);
    rustpython_host_env::windows::set_last_error(old_error);
}

// ── the three calls that produce one ──

/// `_winapi.ConnectNamedPipe(handle, overlapped=False)`
pub fn connect_named_pipe(
    handle: HANDLE,
    use_overlapped: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if !use_overlapped {
        let _blocked = crate::module::thread::before_external_block();
        host_winapi::connect_named_pipe(handle).map_err(win32_err)?;
        return Ok(pyre_object::w_none());
    }
    let backend = new_native(handle)?;
    let error = {
        let mut state = backend.lock().unwrap();
        // An overlapped `ConnectNamedPipe` never reports success directly; it
        // reports either the pending connection or one already made.
        let _blocked = crate::module::thread::before_external_block();
        unsafe {
            windows_sys::Win32::System::Pipes::ConnectNamedPipe(handle, &mut state.overlapped);
            windows_sys::Win32::Foundation::GetLastError()
        }
    };
    match error {
        ERROR_IO_PENDING => backend.lock().unwrap().pending = true,
        // Nothing is left to wait for, so the event is signalled by hand and
        // the caller's wait returns at once.
        ERROR_PIPE_CONNECTED => {
            let state = backend.lock().unwrap();
            host_winapi::set_event(state.overlapped.hEvent).map_err(win32_err)?;
        }
        _ => return Err(super::win32_code(error)),
    }
    Ok(w_overlapped(backend))
}

/// `_winapi.ReadFile(handle, size, overlapped=False)` -> `(data, error)`, and
/// `(overlapped, error)` once an `Overlapped` was asked for.
///
/// `ERROR_MORE_DATA` is not a failure: a message-mode pipe reports it for the
/// rest of a message that did not fit, and the caller reads on.
pub fn read_file(
    handle: HANDLE,
    size: u32,
    use_overlapped: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if !use_overlapped {
        let result = {
            let _blocked = crate::module::thread::before_external_block();
            host_winapi::read_file(handle, size).map_err(win32_err)?
        };
        return Ok(pyre_object::w_tuple_new(vec![
            pyre_object::bytesobject::w_bytes_from_bytes(&result.data),
            pyre_object::w_int_new(i64::from(result.error)),
        ]));
    }
    let backend = new_native(handle)?;
    let error = {
        let mut state = backend.lock().unwrap();
        state.transfer = Transfer::Read;
        state.buffer = vec![0u8; size as usize];
        let mut read: u32 = 0;
        let _blocked = crate::module::thread::before_external_block();
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::ReadFile(
                handle,
                state.buffer.as_mut_ptr().cast(),
                size,
                &mut read,
                &mut state.overlapped,
            )
        };
        if ok == 0 {
            unsafe { windows_sys::Win32::Foundation::GetLastError() }
        } else {
            0
        }
    };
    match error {
        0 | ERROR_MORE_DATA => {}
        ERROR_IO_PENDING => backend.lock().unwrap().pending = true,
        _ => return Err(super::win32_code(error)),
    }
    Ok(pyre_object::w_tuple_new(vec![
        w_overlapped(backend),
        pyre_object::w_int_new(i64::from(error)),
    ]))
}

/// `_winapi.WriteFile(handle, buffer, overlapped=False)` -> `(written, error)`,
/// and `(overlapped, error)` once an `Overlapped` was asked for.
///
/// The overlapped form copies what it is given: the OS reads from the source
/// for as long as the write is in flight, and only the state's own buffer is
/// guaranteed to stay put and unchanged for that long.
pub fn write_file(
    handle: HANDLE,
    data: &[u8],
    use_overlapped: bool,
) -> Result<PyObjectRef, crate::PyError> {
    if !use_overlapped {
        let result = {
            let _blocked = crate::module::thread::before_external_block();
            host_winapi::write_file(handle, data).map_err(win32_err)?
        };
        return Ok(pyre_object::w_tuple_new(vec![
            pyre_object::w_int_new(i64::from(result.written)),
            pyre_object::w_int_new(i64::from(result.error)),
        ]));
    }
    let backend = new_native(handle)?;
    let error = {
        let mut state = backend.lock().unwrap();
        state.transfer = Transfer::Write;
        state.buffer = data.to_vec();
        let length = state.buffer.len().min(u32::MAX as usize) as u32;
        let mut written: u32 = 0;
        let _blocked = crate::module::thread::before_external_block();
        let ok = unsafe {
            windows_sys::Win32::Storage::FileSystem::WriteFile(
                handle,
                state.buffer.as_ptr().cast(),
                length,
                &mut written,
                &mut state.overlapped,
            )
        };
        if ok == 0 {
            unsafe { windows_sys::Win32::Foundation::GetLastError() }
        } else {
            0
        }
    };
    match error {
        0 => {}
        ERROR_IO_PENDING => backend.lock().unwrap().pending = true,
        _ => return Err(super::win32_code(error)),
    }
    Ok(pyre_object::w_tuple_new(vec![
        w_overlapped(backend),
        pyre_object::w_int_new(i64::from(error)),
    ]))
}
