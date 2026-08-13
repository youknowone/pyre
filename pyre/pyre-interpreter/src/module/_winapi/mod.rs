//! _winapi module — partial port of `lib_pypy/_winapi.py`.
//!
//! The Windows build reports `sys.platform == "win32"` and installs the posix
//! module under `os.name == "nt"`, so both the stdlib's `os.name == "nt"` and
//! its `sys.platform == "win32"` branches are live.  The latter reach for
//! `_winapi`, and without the module `import shutil` — hence `tempfile`, and
//! everything downstream — fails outright.
//!
//! The one name a shutil call then goes on to need is
//! `NeedCurrentDirectoryForExePath`, which `shutil.which` invokes on every
//! executable lookup; the `CopyFile2` flag and error constants round out
//! that part of the module surface, though `shutil.copyfile` reads them
//! only behind a `hasattr(_winapi, "CopyFile2")` probe.  `CopyFile2` is
//! deliberately absent, so the probe fails and the generic read/write copy
//! runs, which is the path pyre wants.  Neither name appears in
//! `lib_pypy/_winapi.py`, which predates the stdlib revision pyre ships, so
//! both are defined against the Win32 headers instead.
//!
//! `subprocess` picks its Windows implementation on the presence of `msvcrt`
//! (now installed), so its module body imports the process/priority constants
//! and captures `CloseHandle`/`WaitForSingleObject`/`GetExitCodeProcess` as
//! default arguments.  The launch itself — `CreatePipe`, `DuplicateHandle`,
//! `GetStdHandle`, `CreateProcess`, `TerminateProcess`, `GetFileType` — is in
//! [`process`], which is what `Popen` actually spawns through.

/// Map the current thread's last OS error to an `OSError`
/// (`PyErr_SetFromWindowsErr`): the code is the one `winerror` reports.
fn last_os_error() -> crate::PyError {
    let code = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    crate::PyError::os_error_win32_syscall2(code, pyre_object::PY_NULL, pyre_object::PY_NULL)
}

/// The process-launch half of the module, which `subprocess.Popen` walks in
/// order: `CreatePipe` for each redirected stream, `DuplicateHandle` to make
/// the child's end of it inheritable, `CreateProcess`, then
/// `WaitForSingleObject` / `GetExitCodeProcess` / `TerminateProcess` on what
/// comes back.  Backed by `rustpython_host_env::winapi`, so it is compiled
/// only where that is — which is also the only build that has `msvcrt`, the
/// module `subprocess` picks its Windows implementation from.
#[cfg(feature = "host_env")]
mod process {
    use pyre_object::{PyObjectRef, w_int_new, w_none, w_tuple_new};
    use rustpython_host_env::winapi as host_winapi;
    use windows_sys::Win32::Foundation::HANDLE;

    /// The OSError a failed Win32 call reports (`PyErr_SetFromWindowsErr`).
    fn win32_err(error: std::io::Error) -> crate::PyError {
        crate::PyError::os_error_win32_syscall2(
            error.raw_os_error().unwrap_or(0),
            pyre_object::PY_NULL,
            pyre_object::PY_NULL,
        )
    }

    /// A handle as Python spells it — an integer, which is what every call
    /// here takes and gives back.
    fn handle_w(w_handle: PyObjectRef) -> Result<HANDLE, crate::PyError> {
        Ok(crate::baseobjspace::int_w(w_handle)? as isize as HANDLE)
    }

    fn w_handle(handle: HANDLE) -> PyObjectRef {
        w_int_new(handle as isize as i64)
    }

    fn arg(args: &[PyObjectRef], index: usize, name: &str) -> Result<PyObjectRef, crate::PyError> {
        args.get(index).copied().ok_or_else(|| {
            crate::PyError::type_error(format!("{name}() missing required argument"))
        })
    }

    /// `_winapi.GetStdHandle(std_handle)` — `None` for a stream the process
    /// does not have, which is what the caller tests for before making a pipe
    /// of its own.
    pub fn get_std_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let id = crate::baseobjspace::c_uint_w(arg(args, 0, "GetStdHandle")?)?;
        match host_winapi::get_std_handle(id) {
            Ok(Some(handle)) => Ok(w_handle(handle)),
            Ok(None) => Ok(w_none()),
            Err(e) => Err(win32_err(e)),
        }
    }

    /// `_winapi.GetCurrentProcess()` — the pseudo handle, which names this
    /// process to `DuplicateHandle` without being a handle to close.
    pub fn get_current_process(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(w_handle(host_winapi::get_current_process()))
    }

    /// `_winapi.GetFileType(handle)` — a console handle is the one kind that
    /// cannot be passed in an inherited handle list.
    pub fn get_file_type(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let handle = handle_w(arg(args, 0, "GetFileType")?)?;
        host_winapi::get_file_type(handle)
            .map(|file_type| w_int_new(file_type as i64))
            .map_err(win32_err)
    }

    /// `_winapi.GetLastError()`
    pub fn get_last_error(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(w_int_new(host_winapi::get_last_error() as i64))
    }

    /// `_winapi.GetModuleFileName(module_handle)` — the path the module was
    /// loaded from, or the executable's own path for handle 0.
    ///
    /// `sysconfig._init_non_posix` calls it on `sys.dllhandle` to locate the
    /// install prefix, so it is on the path of every `import sysconfig` here.
    ///
    /// The `MAX_PATH` buffer is the interface, not a shortcut: this call is
    /// specified to hand back one fixed-size buffer, so a longer path comes
    /// back truncated rather than retried in a growing loop.  `initpath.py:308-315
    /// _get_module_file_name` allocates exactly `_MAX_PATH` and gives up when
    /// the result does not fit, and the module this shadows declares
    /// `WCHAR filename[MAX_PATH]` and forces `filename[MAX_PATH - 1] = '\0'`
    /// before returning it.  Growing the buffer here would be the deviation.
    ///
    /// The length is then the NUL scan rather than the count the call reported,
    /// because on a truncation the call reports the whole buffer while the
    /// loader has already terminated the string inside it — taking the reported
    /// count would append that terminator to the path.
    pub fn get_module_file_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        const MAX_PATH: usize = 260;
        let module = handle_w(arg(args, 0, "GetModuleFileName")?)? as *mut core::ffi::c_void;
        let mut buffer = [0u16; MAX_PATH];
        let length = host_winapi::get_module_file_name(module, &mut buffer);
        if length == 0 {
            return Err(super::last_os_error());
        }
        buffer[MAX_PATH - 1] = 0;
        let filename = &buffer[..buffer.iter().position(|&u| u == 0).unwrap_or(MAX_PATH)];
        Ok(pyre_object::w_str_from_wtf8(
            rustpython_wtf8::Wtf8Buf::from_wide(filename),
        ))
    }

    /// `_winapi.TerminateProcess(handle, exit_code)`
    pub fn terminate_process(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let handle = handle_w(arg(args, 0, "TerminateProcess")?)?;
        let exit_code = crate::baseobjspace::c_uint_w(arg(args, 1, "TerminateProcess")?)?;
        if host_winapi::terminate_process(handle, exit_code) == 0 {
            return Err(super::last_os_error());
        }
        Ok(w_none())
    }

    /// `_winapi.CreatePipe(pipe_attrs, size)` -> (read, write).  The
    /// attributes argument is the security descriptor, which the call takes
    /// the default of; neither end is inheritable until one is duplicated
    /// into an inheritable copy.
    pub fn create_pipe(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let size = crate::baseobjspace::c_uint_w(arg(args, 1, "CreatePipe")?)?;
        let (read, write) = host_winapi::create_pipe(size).map_err(win32_err)?;
        Ok(w_tuple_new(vec![w_handle(read), w_handle(write)]))
    }

    /// `_winapi.DuplicateHandle(source_process, source, target_process,
    /// desired_access, inherit_handle, options=0)` -> handle
    pub fn duplicate_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let source_process = handle_w(arg(args, 0, "DuplicateHandle")?)?;
        let source = handle_w(arg(args, 1, "DuplicateHandle")?)?;
        let target_process = handle_w(arg(args, 2, "DuplicateHandle")?)?;
        let access = crate::baseobjspace::c_uint_w(arg(args, 3, "DuplicateHandle")?)?;
        let inherit = crate::baseobjspace::c_int_w(arg(args, 4, "DuplicateHandle")?)?;
        let options = match args.get(5) {
            Some(&w) => crate::baseobjspace::c_uint_w(w)?,
            None => 0,
        };
        host_winapi::duplicate_handle(
            source_process,
            source,
            target_process,
            access,
            inherit,
            options,
        )
        .map(w_handle)
        .map_err(win32_err)
    }

    /// The environment block `CreateProcess` takes: every `KEY=value` in
    /// order, each terminated, the whole thing terminated again
    /// (`getenvironment`).
    ///
    /// Read through the mapping protocol, which is what `getenvironment`
    /// takes: `subprocess` hands `os.environ` straight on, and that is a
    /// `MutableMapping` rather than a dict.  `keys()` names the variables and
    /// each one is subscripted for its value, so a mapping that computes its
    /// values gets to.
    fn environment_block(w_env: PyObjectRef) -> Result<Vec<u16>, crate::PyError> {
        // What a subscript cannot be taken from is no mapping.
        if unsafe { crate::baseobjspace::lookup(w_env, "__getitem__") }.is_none() {
            return Err(crate::PyError::type_error(
                "environment must be dictionary or None",
            ));
        }
        let w_keys = crate::baseobjspace::call_method(w_env, "keys", &[]);
        if w_keys.is_null() {
            return Err(crate::call::take_call_error().unwrap_or_else(|| {
                crate::PyError::type_error("environment must be dictionary or None")
            }));
        }
        let keys = crate::baseobjspace::unpackiterable(w_keys, -1)?;
        // `getitem` runs the mapping's own `__getitem__`, which allocates, so
        // the mapping and every key are published and read back per iteration
        // rather than kept in plain locals.
        let _env_roots = pyre_object::gc_roots::push_roots();
        let env_slot = pyre_object::gc_roots::pin_roots(&[w_env]);
        let keys_base = pyre_object::gc_roots::pin_roots(&keys);
        let mut entries = Vec::with_capacity(keys.len());
        for i in 0..keys.len() {
            let _entry_roots = pyre_object::gc_roots::push_roots();
            let w_value = crate::baseobjspace::getitem(
                pyre_object::gc_roots::shadow_stack_get(env_slot),
                pyre_object::gc_roots::shadow_stack_get(keys_base + i),
            )?;
            let value_slot = pyre_object::gc_roots::pin_roots(&[w_value]);
            let key = crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(
                keys_base + i,
            ))?
            .to_string();
            let value =
                crate::baseobjspace::text_w(pyre_object::gc_roots::shadow_stack_get(value_slot))?
                    .to_string();
            entries.push((key, value));
        }
        host_winapi::build_environment_block(entries).map_err(|e| {
            crate::PyError::value_error(match e {
                host_winapi::BuildEnvironmentBlockError::ContainsNul => "embedded null character",
                host_winapi::BuildEnvironmentBlockError::IllegalName => {
                    "illegal environment variable name"
                }
            })
        })
    }

    /// Read one of `STARTUPINFO`'s fields.  A field left as `None` is the
    /// zero the structure is built with (`getulong` / `gethandle`).
    fn startup_info_field(w_info: PyObjectRef, name: &str) -> Result<i64, crate::PyError> {
        let w_value = crate::baseobjspace::getattr_str(w_info, name)?;
        if unsafe { pyre_object::is_none(w_value) } {
            return Ok(0);
        }
        crate::baseobjspace::int_w(w_value)
    }

    /// The handles `lpAttributeList["handle_list"]` names, which are the ones
    /// the child inherits and the only ones it does.
    ///
    /// An empty list is no list: an attribute list has to carry at least one
    /// handle, and `getattributelist` drops it rather than building one that
    /// `CreateProcess` answers `ERROR_BAD_LENGTH` to.  `subprocess.STARTUPINFO`
    /// starts out with exactly that empty list.
    fn handle_list(w_info: PyObjectRef) -> Result<Option<Vec<usize>>, crate::PyError> {
        let w_attrs = crate::baseobjspace::getattr_str(w_info, "lpAttributeList")?;
        if w_attrs.is_null() || !unsafe { pyre_object::is_dict(w_attrs) } {
            return Ok(None);
        }
        let Some(w_handles) = (unsafe { pyre_object::w_dict_getitem_str(w_attrs, "handle_list") })
        else {
            return Ok(None);
        };
        let items = crate::baseobjspace::unpackiterable(w_handles, -1)?;
        if items.is_empty() {
            return Ok(None);
        }
        let mut handles = Vec::with_capacity(items.len());
        for w_handle in items {
            handles.push(crate::baseobjspace::int_w(w_handle)? as usize);
        }
        Ok(Some(handles))
    }

    /// `_winapi.CreateProcess(application_name, command_line, proc_attrs,
    /// thread_attrs, inherit_handles, creation_flags, env_mapping,
    /// current_directory, startup_info)` -> (process, thread, pid, tid)
    ///
    /// The two attribute arguments are security descriptors, which the call
    /// takes the defaults of.
    pub fn create_process(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let wide = |w: PyObjectRef| -> Result<Option<widestring::WideCString>, crate::PyError> {
            if w.is_null() || unsafe { pyre_object::is_none(w) } {
                return Ok(None);
            }
            let text = crate::baseobjspace::text_w(w)?;
            widestring::WideCString::from_str(text)
                .map(Some)
                .map_err(|_| crate::PyError::value_error("embedded null character"))
        };

        let application_name = wide(arg(args, 0, "CreateProcess")?)?;
        // `CreateProcessW` writes into the command line it is given, so it
        // takes a buffer rather than the string itself.
        let mut command_line =
            wide(arg(args, 1, "CreateProcess")?)?.map(|line| line.into_vec_with_nul());
        let inherit_handles = crate::baseobjspace::int_w(arg(args, 4, "CreateProcess")?)?;
        let creation_flags = crate::baseobjspace::c_uint_w(arg(args, 5, "CreateProcess")?)?;
        let w_env = arg(args, 6, "CreateProcess")?;
        let environment = if w_env.is_null() || unsafe { pyre_object::is_none(w_env) } {
            None
        } else {
            Some(environment_block(w_env)?)
        };
        let current_directory = wide(arg(args, 7, "CreateProcess")?)?;
        let w_startup_info = arg(args, 8, "CreateProcess")?;
        let startup_info = host_winapi::StartupInfoData {
            flags: startup_info_field(w_startup_info, "dwFlags")? as u32,
            show_window: startup_info_field(w_startup_info, "wShowWindow")? as u16,
            std_input: startup_info_field(w_startup_info, "hStdInput")? as isize as HANDLE,
            std_output: startup_info_field(w_startup_info, "hStdOutput")? as isize as HANDLE,
            std_error: startup_info_field(w_startup_info, "hStdError")? as isize as HANDLE,
        };
        let handles = handle_list(w_startup_info)?;

        let info = host_winapi::create_process(
            application_name.as_deref(),
            command_line.as_deref_mut(),
            inherit_handles as i32,
            creation_flags,
            environment.as_deref(),
            current_directory.as_deref(),
            startup_info,
            handles,
        )
        .map_err(win32_err)?;
        Ok(w_tuple_new(vec![
            w_handle(info.process),
            w_handle(info.thread),
            w_int_new(info.process_id as i64),
            w_int_new(info.thread_id as i64),
        ]))
    }
}

crate::py_module! {
    "_winapi",
    int_constants: {
        // CopyFileEx flags (winbase.h).
        "COPY_FILE_ALLOW_DECRYPTED_DESTINATION" => 0x0000_0008,
        "COPY_FILE_COPY_SYMLINK" => 0x0000_0800,
        // System error codes (winerror.h) a caller compares
        // `OSError.winerror` against to decide whether to retry.
        "ERROR_ACCESS_DENIED" => 5,
        "ERROR_PRIVILEGE_NOT_HELD" => 1314,
        // `subprocess` imports these at module load (its Windows branch, taken
        // once `msvcrt` exists) — GetStdHandle ids, ShowWindow/STARTUPINFO
        // flags, and CreateProcess creation/priority flags (winbase.h,
        // processthreadsapi.h).  Exposed as the unsigned DWORD values.
        "STD_INPUT_HANDLE" => 0xFFFF_FFF6u32,
        "STD_OUTPUT_HANDLE" => 0xFFFF_FFF5u32,
        "STD_ERROR_HANDLE" => 0xFFFF_FFF4u32,
        "SW_HIDE" => 0,
        "STARTF_USESHOWWINDOW" => 0x0000_0001,
        "STARTF_USESTDHANDLES" => 0x0000_0100,
        "STARTF_FORCEONFEEDBACK" => 0x0000_0040,
        "STARTF_FORCEOFFFEEDBACK" => 0x0000_0080,
        "CREATE_NEW_CONSOLE" => 0x0000_0010,
        "CREATE_NEW_PROCESS_GROUP" => 0x0000_0200,
        "CREATE_NO_WINDOW" => 0x0800_0000,
        "DETACHED_PROCESS" => 0x0000_0008,
        "CREATE_DEFAULT_ERROR_MODE" => 0x0400_0000,
        "CREATE_BREAKAWAY_FROM_JOB" => 0x0100_0000,
        "ABOVE_NORMAL_PRIORITY_CLASS" => 0x0000_8000,
        "BELOW_NORMAL_PRIORITY_CLASS" => 0x0000_4000,
        "HIGH_PRIORITY_CLASS" => 0x0000_0080,
        "IDLE_PRIORITY_CLASS" => 0x0000_0040,
        "NORMAL_PRIORITY_CLASS" => 0x0000_0020,
        "REALTIME_PRIORITY_CLASS" => 0x0000_0100,
        // WaitForSingleObject results / GetExitCodeProcess sentinel that
        // `subprocess.Popen._wait`/`poll` capture (winbase.h, ntstatus.h).
        "WAIT_OBJECT_0" => 0,
        "WAIT_ABANDONED_0" => 0x0000_0080,
        "WAIT_TIMEOUT" => 0x0000_0102,
        "INFINITE" => 0xFFFF_FFFFu32,
        "STILL_ACTIVE" => 259,
        // `DuplicateHandle` options and the handle sentinel (handleapi.h).
        "DUPLICATE_SAME_ACCESS" => 0x0000_0002,
        "DUPLICATE_CLOSE_SOURCE" => 0x0000_0001,
        "INVALID_HANDLE_VALUE" => -1,
        // `GetFileType` answers.  A console handle is the one a caller drops
        // from an inherited handle list (`Popen._filter_handle_list`).
        "FILE_TYPE_UNKNOWN" => 0x0000,
        "FILE_TYPE_DISK" => 0x0001,
        "FILE_TYPE_CHAR" => 0x0002,
        "FILE_TYPE_PIPE" => 0x0003,
        "FILE_TYPE_REMOTE" => 0x8000,
        // `OpenProcess`/`DuplicateHandle` access rights (processthreadsapi.h).
        "PROCESS_ALL_ACCESS" => 0x001F_FFFF,
        "PROCESS_DUP_HANDLE" => 0x0000_0040,
    },
    inline_functions: {
        fn NeedCurrentDirectoryForExePath(exe_name: &str) -> bool {
            unsafe extern "system" {
                fn NeedCurrentDirectoryForExePathW(exe_name: *const u16) -> i32;
            }
            let exe_name_w: Vec<u16> =
                exe_name.encode_utf16().chain(std::iter::once(0)).collect();
            unsafe { NeedCurrentDirectoryForExePathW(exe_name_w.as_ptr()) != 0 }
        }
        // `subprocess.Handle.Close` captures `_winapi.CloseHandle` as a default
        // argument at class-definition time, so the attribute must exist for
        // `import subprocess` to succeed.
        fn CloseHandle(handle: i64) -> Result<(), crate::PyError> {
            let ok = unsafe {
                windows_sys::Win32::Foundation::CloseHandle(handle as *mut _)
            };
            if ok == 0 {
                return Err(last_os_error());
            }
            Ok(())
        }
        // `subprocess.Popen._wait`/`poll` also capture these as default
        // arguments at import time, and reach them once a launch has a
        // process to wait on.
        fn WaitForSingleObject(handle: i64, milliseconds: i64) -> i64 {
            unsafe {
                windows_sys::Win32::System::Threading::WaitForSingleObject(
                    handle as *mut _,
                    milliseconds as u32,
                ) as i64
            }
        }
        fn GetExitCodeProcess(handle: i64) -> Result<i64, crate::PyError> {
            let mut code: u32 = 0;
            let ok = unsafe {
                windows_sys::Win32::System::Threading::GetExitCodeProcess(handle as *mut _, &mut code)
            };
            if ok == 0 {
                return Err(last_os_error());
            }
            Ok(code as i64)
        }
    },
    extra_init: |ns| {
        // The launch half, registered by hand: the module is built without
        // `host_env` too, and there it stops at the constants and the calls
        // above.
        #[cfg(feature = "host_env")]
        {
            for (name, arity, function) in [
                ("GetStdHandle", 1, process::get_std_handle as crate::BuiltinCodeFn),
                ("GetCurrentProcess", 0, process::get_current_process),
                ("GetFileType", 1, process::get_file_type),
                ("GetLastError", 0, process::get_last_error),
                ("GetModuleFileName", 1, process::get_module_file_name),
                ("TerminateProcess", 2, process::terminate_process),
                ("CreatePipe", 2, process::create_pipe),
            ] {
                crate::module_ns_store(
                    ns,
                    name,
                    crate::gateway::with_module(
                        "_winapi",
                        crate::make_module_builtin_function_with_arity(name, function, arity),
                    ),
                );
            }
            // Fixed-arity builtin fast paths only cover zero through four
            // arguments. CreateProcess still needs an exact-arity check, so
            // register it through the general call path with a checked body.
            crate::module_ns_store(
                ns,
                "CreateProcess",
                crate::gateway::with_module(
                    "_winapi",
                    crate::make_module_builtin_function(
                        "CreateProcess",
                        crate::py_checked_arity_fn!(
                            "CreateProcess",
                            9,
                            process::create_process
                        ),
                    ),
                ),
            );
            // `options` is the one argument with a default (`0`), so the
            // count is not fixed.
            crate::module_ns_store(
                ns,
                "DuplicateHandle",
                crate::gateway::with_module(
                    "_winapi",
                    crate::make_module_builtin_function(
                        "DuplicateHandle",
                        process::duplicate_handle,
                    ),
                ),
            );
        }
    },
}
