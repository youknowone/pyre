//! C-callable callback thunks for `_CFuncPtr` instances.

use pyre_object::PyObjectRef;

#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
))]
mod imp {
    use super::PyObjectRef;
    use crate::module::_ctypes::{cdata, funcptr};
    use core::ffi::c_void;
    use rustpython_host_env::ctypes as host_ctypes;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::{LazyLock, Mutex};

    struct ThunkUserdata {
        /// Address of the GC-rooted slot holding the `CFuncPtr` instance: the
        /// slot, not the instance. A minor collection moves the instance and
        /// rewrites the slot, so the live pointer is read through it at call
        /// time.
        slot: *mut usize,
    }

    struct StoredThunk {
        #[allow(dead_code)]
        slot: Box<usize>,
        #[allow(dead_code)]
        thunk: host_ctypes::CallbackThunk<ThunkUserdata>,
    }

    unsafe impl Send for StoredThunk {}

    static CALLBACK_THUNKS: LazyLock<Mutex<Vec<StoredThunk>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));

    pub(super) fn build_thunk(obj: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
        let argtypes = funcptr::resolve_argtypes(obj).unwrap_or_default();
        let ffi_arg_types = argtypes
            .iter()
            .copied()
            .map(ffi_type_for_arg)
            .collect::<Vec<_>>();
        let restype = funcptr::resolve_restype(obj).map_err(|_| invalid_callback_result_type())?;
        let ffi_res_type = ffi_type_for_ret(&restype)?;

        let mut slot = Box::new(obj as usize);
        let slot_ptr = (&mut *slot) as *mut usize;
        let root_slot = slot_ptr as *mut *mut u8;
        unsafe { pyre_object::gc_hook::try_gc_add_root(root_slot) };

        let thunk = host_ctypes::CallbackThunk::new(
            ffi_arg_types,
            ffi_res_type,
            Box::new(ThunkUserdata { slot: slot_ptr }),
            thunk_callback,
        );
        let code = thunk.code_ptr().0 as usize;

        // A foreign caller may retain a callback pointer indefinitely; pyre has
        // no proof point for unregistering it, so the rooted slot and closure
        // intentionally live for the process lifetime.
        CALLBACK_THUNKS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(StoredThunk { slot, thunk });
        Ok(Some(code))
    }

    fn ffi_type_for_arg(at: PyObjectRef) -> host_ctypes::FfiType {
        if funcptr::argtype_is_pointer_kind(at) {
            return host_ctypes::ffi_pointer_type();
        }
        let Some(tc) = cdata::type_code_of(at) else {
            return host_ctypes::ffi_pointer_type();
        };
        host_ctypes::ffi_type_from_code(&tc).unwrap_or_else(host_ctypes::ffi_pointer_type)
    }

    fn ffi_type_for_ret(ret: &funcptr::Ret) -> Result<host_ctypes::FfiType, crate::PyError> {
        match ret {
            funcptr::Ret::Void => Ok(host_ctypes::ffi_void_type()),
            funcptr::Ret::Code(c) => {
                host_ctypes::ffi_type_from_code(c).ok_or_else(invalid_callback_result_type)
            }
            funcptr::Ret::Pointer(_) | funcptr::Ret::Aggregate(_) => {
                Err(invalid_callback_result_type())
            }
        }
    }

    fn invalid_callback_result_type() -> crate::PyError {
        crate::PyError::type_error("invalid result type for callback function")
    }

    unsafe extern "C" fn thunk_callback(
        _cif: &host_ctypes::FfiCif,
        result: &mut c_void,
        args: *const *const c_void,
        userdata: &ThunkUserdata,
    ) {
        let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
            thunk_callback_inner(result, args, userdata)
        }));
    }

    unsafe fn thunk_callback_inner(
        result: &mut c_void,
        args: *const *const c_void,
        userdata: &ThunkUserdata,
    ) {
        crate::module::thread::ensure_runtime_thread();
        let _callback = crate::module::thread::enter_external_callback();
        let obj = unsafe { *userdata.slot } as PyObjectRef;
        let use_errno = funcptr::funcptr_flags(obj) & funcptr::FUNCFLAG_USE_ERRNO != 0;
        let call_result = match decode_args(obj, args) {
            Ok(decoded) => host_ctypes::with_callback_errno_preserved(use_errno, || {
                let callable = funcptr::instance_get(obj, funcptr::CALLABLE_KEY)
                    .ok_or_else(|| crate::PyError::type_error("callback has no callable"))?;
                crate::call::call_function_impl_result(callable, &decoded)
            }),
            Err(error) => Err(error),
        };
        // A callback has nowhere to raise: the frame above it is foreign.  The
        // call's own failure was already reported by `callback_result`, which
        // substitutes a zero result; what is left here is a result the declared
        // `restype` cannot represent.
        let written = funcptr::callback_result(obj, call_result)
            .and_then(|value| write_result(obj, value, result as *mut c_void));
        if let Err(mut error) = written {
            error.write_unraisable(
                pyre_object::w_none(),
                &rustpython_wtf8::Wtf8Buf::from_string(
                    "Exception ignored while converting result of ctypes callback function"
                        .to_string(),
                ),
                pyre_object::PY_NULL,
            );
        }
    }

    fn decode_args(
        obj: PyObjectRef,
        args: *const *const c_void,
    ) -> Result<Vec<PyObjectRef>, crate::PyError> {
        let argtypes = funcptr::resolve_argtypes(obj).unwrap_or_default();
        let mut decoded = Vec::with_capacity(argtypes.len());
        for (i, at) in argtypes.into_iter().enumerate() {
            let p = unsafe { *args.add(i) };
            match decode_arg(at, p) {
                Ok(value) => decoded.push(value),
                Err(error) => return Err(error),
            }
        }
        Ok(decoded)
    }

    fn decode_arg(at: PyObjectRef, p: *const c_void) -> Result<PyObjectRef, crate::PyError> {
        if let Some(tc) = cdata::type_code_of(at)
            && !funcptr::is_simple_subclass(at)
        {
            if tc == "O" {
                let address = unsafe { *(p as *const usize) };
                return Ok(if address == 0 {
                    pyre_object::w_none()
                } else {
                    address as PyObjectRef
                });
            }
            return Ok(cdata::decoded_to_pyobject(unsafe {
                host_ctypes::callback_arg_value(Some(&tc), p)
            }));
        }
        if let Some(size) = cdata::ctype_size_of(at) {
            let inst = crate::call::type_call_instantiate(at, &[])?;
            let bytes = unsafe { host_ctypes::borrow_memory(p.cast::<u8>(), size) };
            cdata::cdata_write(inst, 0, bytes);
            return Ok(inst);
        }
        Err(crate::PyError::type_error("cannot build parameter"))
    }

    fn write_result(
        obj: PyObjectRef,
        value: PyObjectRef,
        result: *mut c_void,
    ) -> Result<(), crate::PyError> {
        match funcptr::resolve_restype(obj)? {
            funcptr::Ret::Void => Ok(()),
            funcptr::Ret::Code(c) => {
                let bytes = cdata::encode_value_into(&c, value, obj, "result")?;
                // The return slot is one `ffi_arg` word: a narrow value has to
                // go into a zeroed word rather than beside whatever the closure
                // left there, and nothing wider than the word may be written.
                // `g` is the case that overruns — it encodes to a long double
                // while `ffi_type_from_code` maps it to the `f64` the slot was
                // sized for.
                let width = std::mem::size_of::<usize>();
                let n = bytes.len().min(width);
                unsafe {
                    std::ptr::write_bytes(result.cast::<u8>(), 0, width);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.cast::<u8>(), n);
                }
                Ok(())
            }
            funcptr::Ret::Pointer(_) | funcptr::Ret::Aggregate(_) => {
                Err(invalid_callback_result_type())
            }
        }
    }
}

#[cfg(not(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
)))]
mod imp {
    use super::PyObjectRef;

    pub(super) fn build_thunk(_obj: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
        Ok(None)
    }
}

/// Build a C-callable thunk for the callback `CFuncPtr` instance `obj` and
/// return its code address, or `None` when this platform has no libffi closure
/// support.
pub(super) fn build_thunk(obj: PyObjectRef) -> Result<Option<usize>, crate::PyError> {
    imp::build_thunk(obj)
}
