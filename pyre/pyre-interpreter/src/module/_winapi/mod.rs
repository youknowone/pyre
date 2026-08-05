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
//! default arguments — those must exist here for `import subprocess` to
//! succeed.  The `CreateProcess`/pipe/`DuplicateHandle` half a real launch
//! needs is still absent, so an actual `Popen` spawn is a further follow-up.

/// Map the current thread's last OS error to an `OSError`
/// (`PyErr_SetFromWindowsErr`): the code is the one `winerror` reports.
fn last_os_error() -> crate::PyError {
    let code = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
    crate::PyError::os_error_win32_syscall2(code, pyre_object::PY_NULL, pyre_object::PY_NULL)
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
        // arguments at import time (a process launch, needing the missing
        // `CreateProcess`/pipe half, is what actually reaches them).
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
    }
}
