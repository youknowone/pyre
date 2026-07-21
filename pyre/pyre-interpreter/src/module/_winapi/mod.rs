//! _winapi module — partial port of `lib_pypy/_winapi.py`.
//!
//! The Windows build reports `sys.platform == "win32"` while `posix` is the
//! installed filesystem module, so `os.name` is `"posix"` and the stdlib's
//! `os.name == "nt"` branches stay dormant while its `sys.platform` ones do
//! not.  Those reach for `_winapi`, and without the module `import shutil`
//! — hence `tempfile`, and everything downstream — fails outright.
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
//! The process and handle half (`CreateProcess`, `DuplicateHandle`, the
//! pipe calls) has no reachable caller: `subprocess` picks its Windows
//! implementation on the presence of `msvcrt`, and `multiprocessing`'s
//! spawn/reduction paths only touch `_winapi` once a Windows process
//! launch is already underway.

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
    }
}
