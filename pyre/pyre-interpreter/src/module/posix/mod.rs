//! posix module — PyPy: pypy/module/posix/
//!
//! Provides the minimal surface that os.py module init needs to succeed
//! plus the host_env-backed implementations of the calls pyre actually
//! exercises.  The shared `stat_result` builtin type lives here too, since
//! both arms build one: `interp_posix` wraps the host's syscalls, and
//! `interp_posix_wasm` answers from the import machinery's source seam on a
//! target that has none.

#[cfg(not(target_arch = "wasm32"))]
crate::pyre_module_init!(interp_posix);

#[cfg(not(target_arch = "wasm32"))]
pub use interp_posix::{W_DirEntry, W_ScandirIterator};

// wasm32 has no operating system to wrap: the only filesystem the guest can
// see is the embedder's, reached through the import machinery's
// `SourceProvider`.  That leaves a `posix` too narrow to be a `cfg` pass over
// the syscall module, so it is its own arm.
#[cfg(target_arch = "wasm32")]
crate::pyre_module_init!(interp_posix_wasm);

use pyre_object::PyObjectRef;

/// `posix.stat_result` — a real structseq (tuple subclass) so `st[0]`,
/// `len(st)`, iteration and `isinstance(st, tuple)` all work, matching
/// `posixmodule.c` `stat_result_desc`.  The 10 sequence slots hold the
/// integer fields, with the integer-seconds times at 7..10 under the
/// hidden `_integer_atime`/`_integer_mtime`/`_integer_ctime` names; the
/// float `st_atime`/`st_mtime`/`st_ctime`, the `st_*_ns` integers, and the
/// `st_blksize`/`st_blocks`/`st_rdev` block-device fields are named-only
/// extras.
pub(crate) fn stat_result_seq_type() -> PyObjectRef {
    static STAT_RESULT_SEQ_TYPE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STAT_RESULT_SEQ_TYPE.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            // Dotted name → `__name__` "stat_result", repr "os.stat_result(...)".
            "os.stat_result",
            // `app_posix.py:20-37` — slots 7..10 are the hidden integer
            // timestamps; the float `st_atime`/`st_mtime`/`st_ctime` are
            // named-only extras, never indexable.
            &[
                "st_mode",
                "st_ino",
                "st_dev",
                "st_nlink",
                "st_uid",
                "st_gid",
                "st_size",
                "_integer_atime",
                "_integer_mtime",
                "_integer_ctime",
            ],
            // `app_posix.py:38-69` — named-only extras ordered by their
            // `structseqfield` index (11..13, 20..23, 40..42, 50..52).
            // `structseq_descr_new` fills surplus sequence items into
            // this list in order, so the list order must match PyPy's
            // index sort, not the build-time population order.
            &[
                // float times, indices 11..13.
                "st_atime",
                "st_mtime",
                "st_ctime",
                // `app_posix.py:45-52` — present where the platform's
                // `struct stat` carries them (every Unix target),
                // indices 20..23.
                #[cfg(unix)]
                "st_blksize",
                #[cfg(unix)]
                "st_blocks",
                #[cfg(unix)]
                "st_rdev",
                // `rposix_stat.py` exposes `st_flags` where the C
                // `struct stat` carries it (BSD family / macOS).
                #[cfg(target_os = "macos")]
                "st_flags",
                // `build_stat_result` (interp_posix.py) +
                // `rposix_stat.py STAT_FIELDS += ALL_STAT_FIELDS[-3:]`
                // — the sub-second nanosecond remainders, indices 40..42.
                "nsec_atime",
                "nsec_mtime",
                "nsec_ctime",
                // full nanosecond timestamps, indices 50..52.
                "st_atime_ns",
                "st_mtime_ns",
                "st_ctime_ns",
                // The Windows members of `stat_result_fields`, which sit
                // after every time and platform extra there too.
                #[cfg(windows)]
                "st_birthtime",
                #[cfg(windows)]
                "st_birthtime_ns",
                #[cfg(windows)]
                "st_file_attributes",
                #[cfg(windows)]
                "st_reparse_tag",
            ],
        ) as usize
    }) as PyObjectRef
}
