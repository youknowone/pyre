//! `_wmi` — CPython's private Windows Management Instrumentation query door.
//!
//! PyPy has no `_wmi` module.  This explicitly requested scope extension is a
//! structural port of CPython 3.14's `PC/_wmimodule.cpp` (`_query_thread`,
//! `wait_event`, and `_wmi_exec_query_impl`): each call owns one pipe, two
//! staged events and one apartment-threaded COM worker.  In particular, the
//! worker is not replaced with PowerShell or a process-global COM connection;
//! those would change the timeout, isolation and concurrent-call semantics.

use pyre_object::{PY_NULL, PyObjectRef};

#[cfg(not(feature = "host_env"))]
use fallback::execute_query;

#[cfg(feature = "host_env")]
fn execute_query(query: Vec<u16>) -> Result<Vec<u16>, crate::PyError> {
    use rustpython_host_env::wmi::{BUFFER_SIZE, ExecQueryError, exec_query_wide};
    let query = widestring::U16CString::from_vec(query)
        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
    let result = {
        let _blocked = crate::module::thread::before_external_block();
        exec_query_wide(&query)
    };
    result.map_err(|error| match error {
        ExecQueryError::MoreData => {
            crate::PyError::os_error(format!("Query returns more than {BUFFER_SIZE} characters"))
        }
        ExecQueryError::Code(code) => {
            crate::PyError::os_error_win32_syscall2(code as i32, PY_NULL, PY_NULL)
        }
    })
}

// Preserve builds that explicitly omit the host provider.
#[cfg(not(feature = "host_env"))]
mod fallback {
    use std::ffi::c_void;
    use std::ptr::{null, null_mut};

    use pyre_object::PY_NULL;
    use windows_sys::Win32::Foundation::{
        CloseHandle, E_FAIL, ERROR_BROKEN_PIPE, ERROR_MORE_DATA, ERROR_NOT_ENOUGH_MEMORY,
        GetLastError, HANDLE, RPC_E_TOO_LATE, SysAllocStringLen, SysFreeString, SysStringLen,
        WAIT_OBJECT_0, WAIT_TIMEOUT,
    };
    use windows_sys::Win32::Storage::FileSystem::{ReadFile, WriteFile};
    use windows_sys::Win32::System::Com::{
        CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED, CoCreateInstance, CoInitializeEx,
        CoInitializeSecurity, CoSetProxyBlanket, CoUninitialize, EOAC_NONE, RPC_C_AUTHN_LEVEL_CALL,
        RPC_C_AUTHN_LEVEL_DEFAULT, RPC_C_IMP_LEVEL_IMPERSONATE,
    };
    use windows_sys::Win32::System::Pipes::CreatePipe;
    use windows_sys::Win32::System::Rpc::{RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE};
    use windows_sys::Win32::System::Threading::{
        CreateEventW, CreateThread, GetExitCodeThread, SetEvent, WaitForSingleObject,
    };
    use windows_sys::Win32::System::Variant::{VARIANT, VariantClear, VariantToString};
    use windows_sys::Win32::System::Wmi::{
        WBEM_FLAG_FORWARD_ONLY, WBEM_FLAG_RETURN_IMMEDIATELY, WBEM_FLAVOR_MASK_ORIGIN,
        WBEM_FLAVOR_ORIGIN_SYSTEM, WBEM_INFINITE, WBEM_S_FALSE, WBEM_S_NO_MORE_DATA,
    };
    use windows_sys::core::{BSTR, GUID, HRESULT, IUnknown_Vtbl};

    const RESULT_UNITS: usize = 8192;
    const CLSID_WBEM_LOCATOR: GUID = GUID::from_u128(0x4590f811_1d3a_11d0_891f_00aa004b2e24);
    const IID_IWBEM_LOCATOR: GUID = GUID::from_u128(0xdc12a687_737f_11cf_884d_00aa004b2e24);

    // `windows-sys` deliberately omits COM interface wrappers.  These are the
    // literal vtable prefixes from wbemidl.h, ending at the last method this module
    // calls.  Unused entries stay pointer-sized slots so the called method keeps
    // its header-defined index without pulling the enormous generated `windows`
    // crate through Charon's LLBC extraction.
    #[repr(C)]
    #[allow(non_snake_case)]
    struct IWbemLocatorVtable {
        base: IUnknown_Vtbl,
        ConnectServer: unsafe extern "system" fn(
            this: *mut c_void,
            network_resource: BSTR,
            user: BSTR,
            password: BSTR,
            locale: BSTR,
            security_flags: i32,
            authority: BSTR,
            context: *mut c_void,
            services: *mut *mut c_void,
        ) -> HRESULT,
    }

    #[repr(C)]
    #[allow(non_snake_case)]
    struct IWbemServicesVtable {
        base: IUnknown_Vtbl,
        unused_before_exec_query: [usize; 17],
        ExecQuery: unsafe extern "system" fn(
            this: *mut c_void,
            language: BSTR,
            query: BSTR,
            flags: i32,
            context: *mut c_void,
            enumerator: *mut *mut c_void,
        ) -> HRESULT,
    }

    #[repr(C)]
    #[allow(non_snake_case)]
    struct IEnumWbemClassObjectVtable {
        base: IUnknown_Vtbl,
        Reset: usize,
        Next: unsafe extern "system" fn(
            this: *mut c_void,
            timeout: i32,
            count: u32,
            values: *mut *mut c_void,
            returned: *mut u32,
        ) -> HRESULT,
    }

    #[repr(C)]
    #[allow(non_snake_case)]
    struct IWbemClassObjectVtable {
        base: IUnknown_Vtbl,
        unused_before_begin_enumeration: [usize; 5],
        BeginEnumeration: unsafe extern "system" fn(this: *mut c_void, flags: i32) -> HRESULT,
        Next: unsafe extern "system" fn(
            this: *mut c_void,
            flags: i32,
            name: *mut BSTR,
            value: *mut VARIANT,
            value_type: *mut i32,
            flavor: *mut i32,
        ) -> HRESULT,
        EndEnumeration: unsafe extern "system" fn(this: *mut c_void) -> HRESULT,
    }

    /// One owned COM interface pointer.  Every WMI interface inherits IUnknown,
    /// so the first vtable prefix owns the common Release entry.
    struct ComPtr(*mut c_void);

    impl ComPtr {
        fn new(raw: *mut c_void) -> Result<Self, HRESULT> {
            if raw.is_null() {
                Err(E_FAIL)
            } else {
                Ok(Self(raw))
            }
        }

        unsafe fn vtable<T>(&self) -> &T {
            unsafe { &**(self.0.cast::<*const T>()) }
        }
    }

    impl Drop for ComPtr {
        fn drop(&mut self) {
            unsafe {
                let vtable = &**(self.0.cast::<*const IUnknown_Vtbl>());
                (vtable.Release)(self.0);
            }
        }
    }

    /// BSTR ownership for the query/constants and property names returned by WMI.
    struct OwnedBstr(BSTR);

    impl OwnedBstr {
        fn from_wide(units: &[u16]) -> Result<Self, HRESULT> {
            let len = u32::try_from(units.len())
                .map_err(|_| hresult_from_win32(ERROR_NOT_ENOUGH_MEMORY))?;
            let value = unsafe { SysAllocStringLen(units.as_ptr(), len) };
            if value.is_null() {
                Err(hresult_from_win32(ERROR_NOT_ENOUGH_MEMORY))
            } else {
                Ok(Self(value))
            }
        }

        fn from_str(value: &str) -> Result<Self, HRESULT> {
            Self::from_wide(&value.encode_utf16().collect::<Vec<_>>())
        }

        fn empty() -> Self {
            Self(null())
        }

        fn as_slice(&self) -> &[u16] {
            if self.0.is_null() {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(self.0, SysStringLen(self.0) as usize) }
            }
        }
    }

    impl Drop for OwnedBstr {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { SysFreeString(self.0) };
            }
        }
    }

    /// The data CPython's `_query_data` hands to `_query_thread`.
    ///
    /// The query is owned rather than borrowed.  This is the Rust spelling of
    /// gh-125315's immediate `SysAllocString(data.query)`: once the caller's staged
    /// wait gives up, a detached worker must not retain a pointer into a Python
    /// string that the caller can release.
    struct QueryData {
        query: Vec<u16>,
        write_pipe: HANDLE,
        init_event: HANDLE,
        connect_event: HANDLE,
    }

    /// `HRESULT_FROM_WIN32`, used where the C++ worker promotes `GetLastError()` to
    /// an HRESULT before returning it as the thread exit code.
    fn hresult_from_win32(code: u32) -> HRESULT {
        if code == 0 {
            0
        } else {
            ((code & 0xffff) | 0x8007_0000) as i32
        }
    }

    fn signal_event(event: HANDLE) -> Result<(), HRESULT> {
        if unsafe { SetEvent(event) } == 0 {
            Err(hresult_from_win32(unsafe { GetLastError() }))
        } else {
            Ok(())
        }
    }

    fn write_wide(pipe: HANDLE, units: &[u16]) -> Result<(), HRESULT> {
        let mut written = 0;
        let bytes = units.len().saturating_mul(size_of::<u16>());
        let bytes = u32::try_from(bytes).map_err(|_| hresult_from_win32(ERROR_MORE_DATA))?;
        if unsafe {
            WriteFile(
                pipe,
                units.as_ptr().cast::<u8>(),
                bytes,
                &mut written,
                null_mut(),
            )
        } == 0
        {
            Err(hresult_from_win32(unsafe { GetLastError() }))
        } else {
            Ok(())
        }
    }

    /// Balances every successful `CoInitializeEx`, including `S_FALSE`.
    struct ComApartment;

    impl Drop for ComApartment {
        fn drop(&mut self) {
            unsafe { CoUninitialize() };
        }
    }

    /// The body of CPython's `_query_thread`; each owned interface is released
    /// before the apartment guard calls `CoUninitialize`.
    fn query_thread_inner(data: &QueryData) -> Result<(), HRESULT> {
        let query = OwnedBstr::from_wide(&data.query)?;

        let initialized = unsafe { CoInitializeEx(null(), COINIT_APARTMENTTHREADED as u32) };
        if initialized < 0 {
            return Err(initialized);
        }
        let _apartment = ComApartment;

        let security = unsafe {
            CoInitializeSecurity(
                null_mut(),
                -1,
                null(),
                null(),
                RPC_C_AUTHN_LEVEL_DEFAULT,
                RPC_C_IMP_LEVEL_IMPERSONATE,
                null(),
                EOAC_NONE as u32,
                null(),
            )
        };
        if security < 0 {
            // Another component may already have installed process-wide COM
            // security.  CPython continues on RPC_E_TOO_LATE and lets the actual
            // WMI calls decide whether that policy is sufficient.
            if security != RPC_E_TOO_LATE {
                return Err(security);
            }
        }

        let mut locator = null_mut();
        let status = unsafe {
            CoCreateInstance(
                &CLSID_WBEM_LOCATOR,
                null_mut(),
                CLSCTX_INPROC_SERVER,
                &IID_IWBEM_LOCATOR,
                &mut locator,
            )
        };
        if status < 0 {
            return Err(status);
        }
        let locator = ComPtr::new(locator)?;
        signal_event(data.init_event)?;

        let empty = OwnedBstr::empty();
        let namespace = OwnedBstr::from_str("ROOT\\CIMV2")?;
        let mut services = null_mut();
        let status = unsafe {
            (locator.vtable::<IWbemLocatorVtable>().ConnectServer)(
                locator.0,
                namespace.0,
                empty.0,
                empty.0,
                empty.0,
                0,
                empty.0,
                null_mut(),
                &mut services,
            )
        };
        if status < 0 {
            return Err(status);
        }
        let services = ComPtr::new(services)?;
        signal_event(data.connect_event)?;

        let status = unsafe {
            CoSetProxyBlanket(
                services.0,
                RPC_C_AUTHN_WINNT,
                RPC_C_AUTHZ_NONE,
                null(),
                RPC_C_AUTHN_LEVEL_CALL,
                RPC_C_IMP_LEVEL_IMPERSONATE,
                null(),
                EOAC_NONE as u32,
            )
        };
        if status < 0 {
            return Err(status);
        }

        let language = OwnedBstr::from_str("WQL")?;
        let mut enumerator = null_mut();
        let status = unsafe {
            (services.vtable::<IWbemServicesVtable>().ExecQuery)(
                services.0,
                language.0,
                query.0,
                WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                null_mut(),
                &mut enumerator,
            )
        };
        if status < 0 {
            return Err(status);
        }
        let enumerator = ComPtr::new(enumerator)?;

        let mut start_of_enum = true;
        loop {
            let mut got = 0;
            let mut value = null_mut();
            let status = unsafe {
                (enumerator.vtable::<IEnumWbemClassObjectVtable>().Next)(
                    enumerator.0,
                    WBEM_INFINITE,
                    1,
                    &mut value,
                    &mut got,
                )
            };
            if status == WBEM_S_FALSE {
                break;
            }
            if status < 0 {
                return Err(status);
            }
            if got != 1 || value.is_null() {
                continue;
            }
            let value = ComPtr::new(value)?;

            if !start_of_enum {
                write_wide(data.write_pipe, &[0])?;
            }
            start_of_enum = false;

            let status =
                unsafe { (value.vtable::<IWbemClassObjectVtable>().BeginEnumeration)(value.0, 0) };
            if status < 0 {
                return Err(status);
            }
            loop {
                let mut prop_name: BSTR = null();
                let mut prop_value = VARIANT::default();
                let mut flavor = 0;
                let status = unsafe {
                    (value.vtable::<IWbemClassObjectVtable>().Next)(
                        value.0,
                        0,
                        &mut prop_name,
                        &mut prop_value,
                        null_mut(),
                        &mut flavor,
                    )
                };
                let prop_name = OwnedBstr(prop_name);
                if status == WBEM_S_NO_MORE_DATA {
                    break;
                }
                if status < 0 {
                    unsafe { VariantClear(&mut prop_value) };
                    unsafe { (value.vtable::<IWbemClassObjectVtable>().EndEnumeration)(value.0) };
                    return Err(status);
                }

                if flavor & WBEM_FLAVOR_MASK_ORIGIN != WBEM_FLAVOR_ORIGIN_SYSTEM {
                    let mut rendered = [0u16; RESULT_UNITS];
                    let converted = unsafe {
                        VariantToString(&prop_value, rendered.as_mut_ptr(), RESULT_UNITS as u32)
                    };
                    unsafe { VariantClear(&mut prop_value) };
                    if converted < 0 {
                        return Err(converted);
                    }
                    let rendered_len = rendered
                        .iter()
                        .position(|&unit| unit == 0)
                        .unwrap_or(RESULT_UNITS);
                    write_wide(data.write_pipe, prop_name.as_slice())?;
                    write_wide(data.write_pipe, &['=' as u16])?;
                    write_wide(data.write_pipe, &rendered[..rendered_len])?;
                    write_wide(data.write_pipe, &[0])?;
                } else {
                    unsafe { VariantClear(&mut prop_value) };
                }
            }
            // CPython ignores EndEnumeration's status after the Next loop has
            // already selected the result to report.
            unsafe { (value.vtable::<IWbemClassObjectVtable>().EndEnumeration)(value.0) };
        }

        Ok(())
    }

    /// Native `CreateThread` entry point.  Returning the HRESULT bit pattern makes
    /// `GetExitCodeThread` feed the same signed Win32 code to `OSError.winerror` as
    /// CPython's `(DWORD)hr` return.
    unsafe extern "system" fn query_thread(param: *mut c_void) -> u32 {
        let data = unsafe { Box::from_raw(param.cast::<QueryData>()) };
        let status =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| query_thread_inner(&data)))
                .unwrap_or(Err(E_FAIL));
        unsafe {
            CloseHandle(data.write_pipe);
        }
        match status {
            Ok(()) => 0,
            Err(error) => error as u32,
        }
    }

    /// CPython `wait_event`: an expired staged wait is surfaced as WinError 258,
    /// while every other wait failure uses the calling thread's last error.
    fn wait_event(event: HANDLE, timeout: u32) -> u32 {
        match unsafe { WaitForSingleObject(event, timeout) } {
            WAIT_OBJECT_0 => 0,
            WAIT_TIMEOUT => WAIT_TIMEOUT,
            _ => unsafe { GetLastError() },
        }
    }

    /// `_wmi_exec_query_impl` after the Python query has been converted to owned
    /// UTF-16.  Everything through the worker cleanup wait is one released region,
    /// matching `Py_BEGIN_ALLOW_THREADS` and allowing concurrent calls to progress.
    pub(super) fn execute_query(query: Vec<u16>) -> Result<Vec<u16>, crate::PyError> {
        let mut buffer = [0u16; RESULT_UNITS];
        let mut offset = 0usize;

        let err = {
            let _blocked = crate::module::thread::before_external_block();
            let mut err = 0u32;
            let init_event = unsafe { CreateEventW(null(), 1, 0, null()) };
            let connect_event = unsafe { CreateEventW(null(), 1, 0, null()) };
            let mut read_pipe: HANDLE = null_mut();
            let mut write_pipe: HANDLE = null_mut();
            let mut thread: HANDLE = null_mut();

            if init_event.is_null()
                || connect_event.is_null()
                || unsafe { CreatePipe(&mut read_pipe, &mut write_pipe, null(), 0) } == 0
            {
                err = unsafe { GetLastError() };
            } else {
                let data = Box::new(QueryData {
                    query,
                    write_pipe,
                    init_event,
                    connect_event,
                });
                let data = Box::into_raw(data);
                thread = unsafe {
                    CreateThread(
                        null(),
                        0,
                        Some(query_thread),
                        data.cast::<c_void>(),
                        0,
                        null_mut(),
                    )
                };
                if thread.is_null() {
                    err = unsafe { GetLastError() };
                    unsafe {
                        drop(Box::from_raw(data));
                        CloseHandle(write_pipe);
                    }
                }
            }

            // gh-112278: ConnectServer has no timeout.  CPython gives first-time
            // COM setup one second, then the WMI connection itself 100 ms.
            if err == 0 {
                err = wait_event(init_event, 1000);
                if err == 0 {
                    err = wait_event(connect_event, 100);
                }
            }

            while err == 0 {
                let mut bytes_read = 0;
                let capacity = size_of_val(&buffer) - offset;
                let read = unsafe {
                    ReadFile(
                        read_pipe,
                        (buffer.as_mut_ptr().cast::<u8>()).add(offset),
                        capacity as u32,
                        &mut bytes_read,
                        null_mut(),
                    )
                };
                if read != 0 {
                    offset += bytes_read as usize;
                    if offset >= size_of_val(&buffer) {
                        err = ERROR_MORE_DATA;
                    }
                } else {
                    err = unsafe { GetLastError() };
                }
            }

            if !read_pipe.is_null() {
                unsafe {
                    CloseHandle(read_pipe);
                }
            }

            if !thread.is_null() {
                let thread_err = match unsafe { WaitForSingleObject(thread, 100) } {
                    WAIT_OBJECT_0 => {
                        let mut code = 0;
                        if unsafe { GetExitCodeThread(thread, &mut code) } == 0 {
                            unsafe { GetLastError() }
                        } else {
                            code
                        }
                    }
                    WAIT_TIMEOUT => WAIT_TIMEOUT,
                    _ => unsafe { GetLastError() },
                };
                if err == 0 || err == ERROR_BROKEN_PIPE {
                    err = thread_err;
                }
                unsafe {
                    CloseHandle(thread);
                }
            }

            if !init_event.is_null() {
                unsafe {
                    CloseHandle(init_event);
                }
            }
            if !connect_event.is_null() {
                unsafe {
                    CloseHandle(connect_event);
                }
            }
            err
        };

        if err == ERROR_MORE_DATA {
            return Err(crate::PyError::os_error(format!(
                "Query returns more than {RESULT_UNITS} characters"
            )));
        }
        if err != 0 {
            return Err(crate::PyError::os_error_win32_syscall2(
                err as i32, PY_NULL, PY_NULL,
            ));
        }
        if offset == 0 {
            return Ok(Vec::new());
        }
        debug_assert_eq!(offset % size_of::<u16>(), 0);
        let units = offset / size_of::<u16>();
        Ok(buffer[..units.saturating_sub(1)].to_vec())
    }
}

fn is_select_query(query: &[u16]) -> bool {
    const SELECT: &[u8; 7] = b"select ";
    query.len() >= SELECT.len()
        && query[..SELECT.len()]
            .iter()
            .zip(SELECT)
            .all(|(&unit, &expected)| {
                unit == expected as u16
                    || (expected.is_ascii_lowercase()
                        && unit == expected.to_ascii_uppercase() as u16)
            })
}

/// `_wmi.exec_query($module, /, query)`.
fn exec_query(args: &[PyObjectRef]) -> crate::PyResult {
    let (positional, kwargs) = crate::builtins::split_builtin_kwargs(args);
    crate::builtins::clinic_arity(
        "exec_query",
        positional.len(),
        crate::builtins::real_kwarg_count(kwargs),
        1,
        1,
        0,
    )?;
    let query = positional
        .first()
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "query"))
        .ok_or_else(|| {
            crate::PyError::type_error("exec_query() missing required argument 'query' (pos 1)")
        })?;
    if !unsafe { pyre_object::is_str(query) } {
        return Err(crate::PyError::type_error(format!(
            "exec_query() argument 'query' must be str, not {}",
            crate::gateway::short_type_name(query)
        )));
    }

    // The clinic unicode converter has accepted the object, but the wide-char
    // copy and SELECT validation live after the audit in CPython's impl.
    let roots = pyre_object::gc_roots::push_roots();
    let query_slot = roots.base();
    let _ = roots.pin_root(query);
    crate::module::sys::vm::audit("_wmi.exec_query", &[roots.get(query_slot)])?;
    let query = roots.get(query_slot);
    let mut wide: Vec<u16> = unsafe { pyre_object::w_str_get_wtf8(query) }
        .encode_wide()
        .collect();
    if wide.contains(&0) {
        return Err(crate::PyError::value_error("embedded null character"));
    }
    if !is_select_query(&wide) {
        return Err(crate::PyError::value_error(
            "only SELECT queries are supported",
        ));
    }

    let result = execute_query(std::mem::take(&mut wide))?;
    Ok(pyre_object::w_str_from_wtf8(
        rustpython_wtf8::Wtf8Buf::from_wide(&result),
    ))
}

const EXEC_QUERY_DOC: &str = "Runs a WMI query against the local machine.\n\nThis returns a single string with 'name=value' pairs in a flat array separated\nby null characters.";

crate::py_module! {
    "_wmi",
    extra_init: |ns| {
        crate::module_ns_store(
            ns,
            "exec_query",
            crate::gateway::with_module(
                "_wmi",
                crate::make_module_builtin_function_with_doc(
                    "exec_query",
                    exec_query,
                    EXEC_QUERY_DOC,
                ),
            ),
        );
    },
}
