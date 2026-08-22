//! _winapi module — the Win32 calls the stdlib's Windows branches reach for.
//!
//! The Windows build reports `sys.platform == "win32"` and installs the posix
//! module under `os.name == "nt"`, so both the stdlib's `os.name == "nt"` and
//! its `sys.platform == "win32"` branches are live.  The latter reach for
//! `_winapi`, and without the module `import shutil` — hence `tempfile`, and
//! everything downstream — fails outright.
//!
//! `lib_pypy/_winapi.py` predates the stdlib revision pyre ships and stops
//! well short of it, so the shapes here follow `PC/_winapi.c` and the Win32
//! headers instead.  The module comes in three parts: the constants and the
//! few calls made straight against `windows-sys` are below; [`process`] is
//! the launch half `subprocess.Popen` spawns through — `CreatePipe`,
//! `DuplicateHandle`, `GetStdHandle`, `CreateProcess`, `TerminateProcess`,
//! `GetFileType` — and [`host`] is everything
//! `rustpython_host_env::winapi` backs, which is the named-pipe, event,
//! mutex, file-mapping, path and locale surface `multiprocessing`,
//! `ntpath.normcase`, `mimetypes` and `shutil.copy2` walk.
//!
//! `subprocess` picks its Windows implementation on the presence of `msvcrt`,
//! so its module body imports the process/priority constants and captures
//! `CloseHandle`/`WaitForSingleObject`/`GetExitCodeProcess` as default
//! arguments.
//!
//! [`overlapped`] holds the `Overlapped` object and the asynchronous form of
//! `ConnectNamedPipe`, `ReadFile` and `WriteFile` that produces one, which is
//! what `multiprocessing.connection`'s `PipeConnection` is written against.

use windows_sys::Win32::Foundation::HANDLE;

use crate::PyError;

/// Map the current thread's last OS error to an `OSError`
/// (`PyErr_SetFromWindowsErr`): the code is the one `winerror` reports.
fn last_os_error() -> crate::PyError {
    win32_err(std::io::Error::last_os_error())
}

/// The `OSError` a failed Win32 call reports (`PyErr_SetFromWindowsErr`).
fn win32_err(error: std::io::Error) -> crate::PyError {
    crate::PyError::os_error_win32_syscall2(
        error.raw_os_error().unwrap_or(0),
        pyre_object::PY_NULL,
        pyre_object::PY_NULL,
    )
}

/// [`win32_err`] for a call that reports its error code rather than setting
/// the thread's.
fn win32_code(code: u32) -> crate::PyError {
    crate::PyError::os_error_win32_syscall2(code as i32, pyre_object::PY_NULL, pyre_object::PY_NULL)
}

/// How a call names an integer argument in the `TypeError` it raises for one
/// that is not an integer at all.
enum IntArg<'a> {
    /// The only parameter of `function`.
    Only(&'a str),
    /// Parameter `position` of `function`, counted from one.
    At { function: &'a str, position: usize },
    /// An element of a sequence argument, which the message does not name.
    Element,
}

/// The value an integer parameter carries: read through `__index__`, then
/// taken modulo the width the call passes it in.  Both `_Py_PARSE_UINTPTR`
/// and the `DWORD` converter work that way, so a handle may be written as the
/// negative it is or as the unsigned value it prints as, and a flag word may
/// be written either way round too.
fn masked_int_w(w_value: pyre_object::PyObjectRef, argument: IntArg<'_>) -> Result<i64, PyError> {
    if !unsafe { pyre_object::pyobject::is_int_or_long(w_value) }
        && unsafe { crate::baseobjspace::lookup(w_value, "__index__") }.is_none()
    {
        let got = crate::gateway::short_type_name(w_value);
        return Err(PyError::type_error(match argument {
            IntArg::Only(function) => format!("{function}() argument must be int, not {got}"),
            IntArg::At { function, position } => {
                format!("{function}() argument {position} must be int, not {got}")
            }
            IntArg::Element => format!("argument must be int, not {got}"),
        }));
    }
    // `PyLong_AsNativeBytes` reads the value under
    // `Py_ASNATIVEBYTES_ALLOW_INDEX`, so `__index__` decides it and `__int__`
    // is never asked for one.
    crate::baseobjspace::truncatedint_w(crate::baseobjspace::space_index(w_value)?)
}

/// [`masked_int_w`] for a `HANDLE` parameter.
fn handle_w(w_handle: pyre_object::PyObjectRef, argument: IntArg<'_>) -> Result<HANDLE, PyError> {
    Ok(masked_int_w(w_handle, argument)? as isize as HANDLE)
}

/// [`masked_int_w`] for a `DWORD` parameter.
fn dword_w(w_value: pyre_object::PyObjectRef, argument: IntArg<'_>) -> Result<u32, PyError> {
    Ok(masked_int_w(w_value, argument)? as u32)
}

/// The integer a handle comes back as (`PyLong_FromVoidPtr`): the unsigned
/// value, which is how the pseudo handles print.
fn w_handle(handle: HANDLE) -> pyre_object::PyObjectRef {
    let value = handle as usize as u64;
    match i64::try_from(value) {
        Ok(fits) => pyre_object::w_int_new(fits),
        Err(_) => {
            pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(value as i128))
        }
    }
}

#[cfg(feature = "host_env")]
mod host;
// The asynchronous half reaches the host's overlapped I/O directly, so a
// sandbox build leaves it out along with `_overlapped`.
#[cfg(all(feature = "host_env", not(feature = "sandbox")))]
pub mod overlapped;

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

    use super::{IntArg, handle_w, w_handle, win32_err};

    fn arg(args: &[PyObjectRef], index: usize, name: &str) -> Result<PyObjectRef, crate::PyError> {
        args.get(index).copied().ok_or_else(|| {
            crate::PyError::type_error(format!("{name}() missing required argument"))
        })
    }

    /// `_winapi.GetStdHandle(std_handle)` — `None` for a stream the process
    /// does not have, which is what the caller tests for before making a pipe
    /// of its own.
    pub fn get_std_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let id = super::dword_w(arg(args, 0, "GetStdHandle")?, IntArg::Only("GetStdHandle"))?;
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
        let handle = handle_w(arg(args, 0, "GetFileType")?, IntArg::Only("GetFileType"))?;
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
        let module = handle_w(
            arg(args, 0, "GetModuleFileName")?,
            IntArg::Only("GetModuleFileName"),
        )? as *mut core::ffi::c_void;
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
        let handle = handle_w(
            arg(args, 0, "TerminateProcess")?,
            IntArg::At {
                function: "TerminateProcess",
                position: 1,
            },
        )?;
        let exit_code = super::dword_w(
            arg(args, 1, "TerminateProcess")?,
            IntArg::At {
                function: "TerminateProcess",
                position: 2,
            },
        )?;
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
        let size = super::dword_w(
            arg(args, 1, "CreatePipe")?,
            IntArg::At {
                function: "CreatePipe",
                position: 2,
            },
        )?;
        let (read, write) = host_winapi::create_pipe(size).map_err(win32_err)?;
        Ok(w_tuple_new(vec![w_handle(read), w_handle(write)]))
    }

    /// `_winapi.DuplicateHandle(source_process, source, target_process,
    /// desired_access, inherit_handle, options=0)` -> handle
    pub fn duplicate_handle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        const NAME: &str = "DuplicateHandle";
        let at = |position| IntArg::At {
            function: NAME,
            position,
        };
        let source_process = handle_w(arg(args, 0, NAME)?, at(1))?;
        let source = handle_w(arg(args, 1, NAME)?, at(2))?;
        let target_process = handle_w(arg(args, 2, NAME)?, at(3))?;
        let access = super::dword_w(arg(args, 3, NAME)?, at(4))?;
        let inherit = crate::baseobjspace::index_c_int_w(arg(args, 4, NAME)?)?;
        let options = match args.get(5) {
            Some(&w) => super::dword_w(w, at(6))?,
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
        // CopyFileEx / CopyFile2 flags (winbase.h).
        "COPY_FILE_ALLOW_DECRYPTED_DESTINATION" => 0x0000_0008,
        "COPY_FILE_COPY_SYMLINK" => 0x0000_0800,
        "COPY_FILE_DIRECTORY" => 0x0000_0080,
        "COPY_FILE_FAIL_IF_EXISTS" => 0x0000_0001,
        "COPY_FILE_NO_BUFFERING" => 0x0000_1000,
        "COPY_FILE_NO_OFFLOAD" => 0x0004_0000,
        "COPY_FILE_OPEN_SOURCE_FOR_WRITE" => 0x0000_0004,
        "COPY_FILE_REQUEST_COMPRESSED_TRAFFIC" => 0x1000_0000,
        "COPY_FILE_REQUEST_SECURITY_PRIVILEGES" => 0x0000_2000,
        "COPY_FILE_RESTARTABLE" => 0x0000_0002,
        "COPY_FILE_RESUME_FROM_PAUSE" => 0x0000_4000,
        // The reasons a `CopyFile2` progress routine is called and the
        // answers it may give (winbase.h).  Nothing here calls one — the
        // extended parameters leave the callback field unset — but the
        // constants are part of the module's surface.
        "COPYFILE2_CALLBACK_CHUNK_STARTED" => 1,
        "COPYFILE2_CALLBACK_CHUNK_FINISHED" => 2,
        "COPYFILE2_CALLBACK_STREAM_STARTED" => 3,
        "COPYFILE2_CALLBACK_STREAM_FINISHED" => 4,
        "COPYFILE2_CALLBACK_POLL_CONTINUE" => 5,
        "COPYFILE2_CALLBACK_ERROR" => 6,
        "COPYFILE2_PROGRESS_CONTINUE" => 0,
        "COPYFILE2_PROGRESS_CANCEL" => 1,
        "COPYFILE2_PROGRESS_STOP" => 2,
        "COPYFILE2_PROGRESS_QUIET" => 3,
        "COPYFILE2_PROGRESS_PAUSE" => 4,
        // System error codes (winerror.h) a caller compares
        // `OSError.winerror` against to decide whether to retry.
        "ERROR_ACCESS_DENIED" => 5,
        "ERROR_PRIVILEGE_NOT_HELD" => 1314,
        "ERROR_ALREADY_EXISTS" => 183,
        "ERROR_BROKEN_PIPE" => 109,
        "ERROR_IO_PENDING" => 997,
        "ERROR_MORE_DATA" => 234,
        "ERROR_NETNAME_DELETED" => 64,
        "ERROR_NO_DATA" => 232,
        "ERROR_NO_SYSTEM_RESOURCES" => 1450,
        "ERROR_OPERATION_ABORTED" => 995,
        "ERROR_PIPE_BUSY" => 231,
        "ERROR_PIPE_CONNECTED" => 535,
        "ERROR_SEM_TIMEOUT" => 121,
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
        "STARTF_USESIZE" => 0x0000_0002,
        "STARTF_USEPOSITION" => 0x0000_0004,
        "STARTF_USECOUNTCHARS" => 0x0000_0008,
        "STARTF_USEFILLATTRIBUTE" => 0x0000_0010,
        "STARTF_RUNFULLSCREEN" => 0x0000_0020,
        "STARTF_USEHOTKEY" => 0x0000_0200,
        "STARTF_TITLEISLINKNAME" => 0x0000_0800,
        "STARTF_TITLEISAPPID" => 0x0000_1000,
        "STARTF_PREVENTPINNING" => 0x0000_2000,
        "STARTF_UNTRUSTEDSOURCE" => 0x0000_8000,
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
        "NULL" => 0,
        // `GetFileType` answers.  A console handle is the one a caller drops
        // from an inherited handle list (`Popen._filter_handle_list`).
        "FILE_TYPE_UNKNOWN" => 0x0000,
        "FILE_TYPE_DISK" => 0x0001,
        "FILE_TYPE_CHAR" => 0x0002,
        "FILE_TYPE_PIPE" => 0x0003,
        "FILE_TYPE_REMOTE" => 0x8000,
        // `OpenProcess`/`DuplicateHandle` access rights (processthreadsapi.h),
        // and the one right every waitable object grants (winnt.h).
        "PROCESS_ALL_ACCESS" => 0x001F_FFFF,
        "PROCESS_DUP_HANDLE" => 0x0000_0040,
        "SYNCHRONIZE" => 0x0010_0000,
        // `CreateFile` access rights, share modes and creation dispositions
        // (winnt.h, fileapi.h).
        "GENERIC_READ" => 0x8000_0000u32,
        "GENERIC_WRITE" => 0x4000_0000,
        "FILE_GENERIC_READ" => 0x0012_0089,
        "FILE_GENERIC_WRITE" => 0x0012_0116,
        "OPEN_EXISTING" => 3,
        "FILE_FLAG_OVERLAPPED" => 0x4000_0000,
        "FILE_FLAG_FIRST_PIPE_INSTANCE" => 0x0008_0000,
        // Named pipe open modes, pipe modes and waits (winbase.h).
        "PIPE_ACCESS_INBOUND" => 0x0000_0001,
        "PIPE_ACCESS_DUPLEX" => 0x0000_0003,
        "PIPE_READMODE_MESSAGE" => 0x0000_0002,
        "PIPE_TYPE_MESSAGE" => 0x0000_0004,
        "PIPE_WAIT" => 0x0000_0000,
        "PIPE_UNLIMITED_INSTANCES" => 255,
        "NMPWAIT_WAIT_FOREVER" => 0xFFFF_FFFFu32,
        // File-mapping protections, view access rights, section attributes
        // and the region states `VirtualQuery` reports (winnt.h, memoryapi.h).
        "PAGE_NOACCESS" => 0x0000_0001,
        "PAGE_READONLY" => 0x0000_0002,
        "PAGE_READWRITE" => 0x0000_0004,
        "PAGE_WRITECOPY" => 0x0000_0008,
        "PAGE_EXECUTE" => 0x0000_0010,
        "PAGE_EXECUTE_READ" => 0x0000_0020,
        "PAGE_EXECUTE_READWRITE" => 0x0000_0040,
        "PAGE_EXECUTE_WRITECOPY" => 0x0000_0080,
        "PAGE_GUARD" => 0x0000_0100,
        "PAGE_NOCACHE" => 0x0000_0200,
        "PAGE_WRITECOMBINE" => 0x0000_0400,
        "FILE_MAP_COPY" => 0x0000_0001,
        "FILE_MAP_WRITE" => 0x0000_0002,
        "FILE_MAP_READ" => 0x0000_0004,
        "FILE_MAP_EXECUTE" => 0x0000_0020,
        "FILE_MAP_ALL_ACCESS" => 0x000F_001F,
        "SEC_IMAGE" => 0x0100_0000,
        "SEC_RESERVE" => 0x0400_0000,
        "SEC_COMMIT" => 0x0800_0000,
        "SEC_NOCACHE" => 0x1000_0000,
        "SEC_WRITECOMBINE" => 0x4000_0000,
        "SEC_LARGE_PAGES" => 0x8000_0000u32,
        "MEM_COMMIT" => 0x0000_1000,
        "MEM_RESERVE" => 0x0000_2000,
        "MEM_FREE" => 0x0001_0000,
        "MEM_PRIVATE" => 0x0002_0000,
        "MEM_MAPPED" => 0x0004_0000,
        "MEM_IMAGE" => 0x0100_0000,
        // `LCMapStringEx` transforms and the longest locale name it takes
        // (winnls.h).  The four that answer with a sort key or a hash rather
        // than text are not among them — `LCMapStringEx` rejects those.
        "LCMAP_LOWERCASE" => 0x0000_0100,
        "LCMAP_UPPERCASE" => 0x0000_0200,
        "LCMAP_TITLECASE" => 0x0000_0300,
        "LCMAP_HIRAGANA" => 0x0010_0000,
        "LCMAP_KATAKANA" => 0x0020_0000,
        "LCMAP_HALFWIDTH" => 0x0040_0000,
        "LCMAP_FULLWIDTH" => 0x0080_0000,
        "LCMAP_LINGUISTIC_CASING" => 0x0100_0000,
        "LCMAP_SIMPLIFIED_CHINESE" => 0x0200_0000,
        "LCMAP_TRADITIONAL_CHINESE" => 0x0400_0000,
        "LOCALE_NAME_MAX_LENGTH" => 85,
    },
    inline_functions: {
        fn NeedCurrentDirectoryForExePath(
            exe_name: &str,
        ) -> Result<bool, crate::PyError> {
            // The wide string ends at its first NUL, so a name carrying one
            // would be answered for a shorter name than was passed.
            let mut exe_name_w: Vec<u16> = exe_name.encode_utf16().collect();
            if exe_name_w.contains(&0) {
                return Err(crate::PyError::value_error("embedded null character"));
            }
            exe_name_w.push(0);
            Ok(unsafe {
                windows_sys::Win32::System::Environment::NeedCurrentDirectoryForExePathW(
                    exe_name_w.as_ptr(),
                ) != 0
            })
        }
        // `subprocess.Handle.Close` captures `_winapi.CloseHandle` as a default
        // argument at class-definition time, so the attribute must exist for
        // `import subprocess` to succeed.
        fn CloseHandle(handle: pyre_object::PyObjectRef) -> Result<(), crate::PyError> {
            let handle = handle_w(handle, IntArg::Only("CloseHandle"))?;
            let ok = unsafe {
                windows_sys::Win32::Foundation::CloseHandle(handle)
            };
            if ok == 0 {
                return Err(last_os_error());
            }
            Ok(())
        }
        // `subprocess.Popen._wait`/`poll` also capture these as default
        // arguments at import time, and reach them once a launch has a
        // process to wait on.
        fn WaitForSingleObject(
            handle: pyre_object::PyObjectRef,
            milliseconds: pyre_object::PyObjectRef,
        ) -> Result<i64, crate::PyError> {
            const NAME: &str = "WaitForSingleObject";
            let handle = handle_w(handle, IntArg::At { function: NAME, position: 1 })?;
            let milliseconds =
                dword_w(milliseconds, IntArg::At { function: NAME, position: 2 })?;
            // `rpython/rlib/rwin32.py` declares this through `winexternal`,
            // whose default `releasegil='auto'` runs the native call between
            // the rffi around-handlers.  In particular, an infinite wait must
            // let the Python thread which will signal the handle run.
            let result = {
                let _blocked = crate::module::thread::before_external_block();
                unsafe {
                    windows_sys::Win32::System::Threading::WaitForSingleObject(
                        handle,
                        milliseconds,
                    )
                }
            };
            if result == windows_sys::Win32::Foundation::WAIT_FAILED {
                return Err(last_os_error());
            }
            Ok(i64::from(result))
        }
        fn GetExitCodeProcess(handle: pyre_object::PyObjectRef) -> Result<i64, crate::PyError> {
            let handle = handle_w(handle, IntArg::Only("GetExitCodeProcess"))?;
            let mut code: u32 = 0;
            let ok = unsafe {
                windows_sys::Win32::System::Threading::GetExitCodeProcess(handle, &mut code)
            };
            if ok == 0 {
                return Err(last_os_error());
            }
            Ok(code as i64)
        }
    },
    extra_init: |ns| {
        // The handle sentinel is `(HANDLE)-1` (handleapi.h), which prints as
        // the unsigned value and so does not fit the `int_constants` table.
        crate::module_ns_store(
            ns,
            "INVALID_HANDLE_VALUE",
            w_handle(windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE),
        );
        // The launch half, registered by hand: the module is built without
        // `host_env` too, and there it stops at the constants and the calls
        // above.
        #[cfg(feature = "host_env")]
        {
            host::install(ns);
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
            // arguments (`gateway.py BuiltinCode0..BuiltinCode4`), and
            // CreateProcess takes nine. It still needs an exact-arity check, so
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
