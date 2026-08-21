//! posix implementation — PyPy: pypy/module/posix/interp_posix.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! shared `stat_result_type` helper is carried in here too; `init_posix`
//! is renamed to `register_module`.

use crate::importing::host::{fs as host_fs, os as host_os};
use pyre_object::PyObjectRef;
use std::sync::{LazyLock, Mutex};
// Under sandbox, name libc through the seam facade so any direct syscall call
// in this module is a compile error (only types/constants/pure fns resolve).
#[cfg(feature = "sandbox")]
use crate::host_seam::sys as libc;

/// PyPy `ApplevelForkCallbacks`, cached on the object space.
///
/// The callback collections are RPython lists, so use insertion-ordered Vecs;
/// this is process/interpreter state, never TLS.
#[derive(Default)]
struct ApplevelForkCallbacks {
    before_w: Vec<usize>,
    parent_w: Vec<usize>,
    child_w: Vec<usize>,
}

/// `posix.DirEntry` — native layout `[PyObject | w_name | w_path | w_stat |
/// w_lstat | dir_fd | enum_ino | enum_type]`, matching `interp_scandir.py
/// W_DirEntry`: the name and full path plus the cached `stat`
/// (`follow_symlinks=True`) and `lstat` (`follow_symlinks=False`) results.
/// `w_stat`/`w_lstat` are `PY_NULL` until first requested, so `entry.stat()`
/// re-fetches once and then returns the same object, and `is_dir`/`is_file`
/// share the same on-demand stat.  `dir_fd` is the descriptor a `scandir(fd)`
/// handed the entry (`-1` for a name), which its own stat resolves the bare
/// `name` against — the native counterpart of
/// `self.scandir_iterator.orig_fd`.  `enum_ino` is the inode `readdir` reported
/// at enumeration (`descr_inode`'s `self.inode`), so `inode()` answers from it
/// without a stat; it is `-1` when unavailable (non-unix hosts), which falls
/// back to a stat.  `enum_type` is the `d_type` `readdir` reported (the
/// `known_type` half of `self.flags`), so `is_dir`/`is_file`/`is_symlink`
/// answer from it without a stat when it is not `DT_UNKNOWN`; it defaults to
/// `DT_UNKNOWN` (`0`) — the value for a host or filesystem that reports no
/// type — which falls through to the stat.
/// The layout carries no instance dict; `name`/`path` are read-only getset
/// descriptors, so the type is not instantiable and not acceptable as a base.
#[crate::pyre_class("posix.DirEntry")]
#[derive(Default)]
pub struct W_DirEntry {
    pub w_name: PyObjectRef,
    pub w_path: PyObjectRef,
    pub w_stat: PyObjectRef,
    pub w_lstat: PyObjectRef,
    pub dir_fd: i32,
    pub enum_ino: i64,
    pub enum_type: i32,
}

/// Native owner for `posix.ScandirIterator` entries and enumeration state.
/// PyPy's `interp_scandir.W_ScandirIterator` keeps the equivalent state on
/// `dirp`; its typedef exposes operations rather than these fields.
#[crate::pyre_class("posix.ScandirIterator")]
#[derive(Default)]
pub struct W_ScandirIterator {
    pub entries: PyObjectRef,
    pub index: i64,
    pub open: bool,
    /// `W_ScandirIterator._in_next` (interp_scandir.py).
    pub in_next: bool,
}

static APPLEVEL_FORK_CALLBACKS: LazyLock<Mutex<ApplevelForkCallbacks>> =
    LazyLock::new(|| Mutex::new(ApplevelForkCallbacks::default()));
// PyPy's GIL serializes concurrent fork entry.  Pyre is free-threaded, so the
// corresponding process operation has its own narrow serializer.
static FORK_SERIALIZER: Mutex<()> = Mutex::new(());

// `_in_next`'s test-and-set is indivisible under PyPy's GIL. Pyre is
// free-threaded, so every borrow of the native scandir iterator takes this
// narrow serializer. Claiming, taking, and releasing are separate serialized
// accesses, so a second thread arriving during a claimed step observes
// `_in_next` and is refused as interp_scandir.py:133-135 requires.
static SCANDIR_IN_NEXT_SERIALIZER: Mutex<()> = Mutex::new(());

fn require_env_mapping(
    mapping: PyObjectRef,
    function: &str,
    accepts_none: bool,
) -> Result<(), crate::PyError> {
    if crate::baseobjspace::py_mapping_check(mapping) {
        return Ok(());
    }
    let none_tail = if accepts_none { " or None" } else { "" };
    Err(crate::PyError::type_error(format!(
        "{function}: environment must be a mapping object{none_tail}"
    )))
}

/// The `key=value` byte entries an exec takes from `mapping`, in the order its
/// `keys()` and `values()` hold them.
///
/// Both sequences are snapshotted before any element is encoded, so a
/// `__fspath__` running during the encoding cannot make a later read observe a
/// mutation it performed.  How many variables there are is the mapping's own
/// `len()`, so a snapshot too short to cover it is an error rather than a
/// quietly shorter environment.
///
/// `function` names the caller in the errors, and `accepts_none` spells the
/// message for an entry point that also takes `None` — what `None` means is
/// decided before the call, never here.
fn collect_env_entries(
    mapping: PyObjectRef,
    function: &str,
    accepts_none: bool,
) -> Result<Vec<Vec<u8>>, crate::PyError> {
    let _env_roots = pyre_object::gc_roots::push_roots();
    let mapping_slot = pyre_object::gc_roots::pin_roots(&[mapping]);
    require_env_mapping(
        pyre_object::gc_roots::shadow_stack_get(mapping_slot),
        function,
        accepts_none,
    )?;
    let pair_count =
        crate::baseobjspace::len_w(pyre_object::gc_roots::shadow_stack_get(mapping_slot))? as usize;
    let mut bases = [0usize; 2];
    let mut lengths = [0usize; 2];
    for (i, method) in ["keys", "values"].into_iter().enumerate() {
        let sequence = crate::baseobjspace::call_method(
            pyre_object::gc_roots::shadow_stack_get(mapping_slot),
            method,
            &[],
        );
        if sequence.is_null() {
            return Err(crate::call::take_call_error().unwrap_or_else(|| {
                crate::PyError::type_error(format!("{function}: env must be a mapping"))
            }));
        }
        let items = crate::baseobjspace::unpackiterable(sequence, -1)?;
        bases[i] = pyre_object::gc_roots::pin_roots(&items);
        lengths[i] = items.len();
    }
    // Capacity follows the available snapshots, while iteration still uses the
    // mapping's reported length and rejects a snapshot too short to cover it.
    let mut env = Vec::with_capacity(pair_count.min(lengths[0]).min(lengths[1]));
    for i in 0..pair_count {
        if i >= lengths[0] || i >= lengths[1] {
            return Err(crate::PyError::index_error("list index out of range"));
        }
        let key = crate::gateway::fsencode_bytes_w(pyre_object::gc_roots::shadow_stack_get(
            bases[0] + i,
        ))?;
        let value = crate::gateway::fsencode_bytes_w(pyre_object::gc_roots::shadow_stack_get(
            bases[1] + i,
        ))?;
        // PyPy's `_env2interp` permits the Windows `=C:` form and rejects `=`
        // only after the first byte.
        if key.is_empty() || key.get(1..).is_some_and(|tail| tail.contains(&b'=')) {
            return Err(crate::PyError::value_error(
                "illegal environment variable name",
            ));
        }
        let mut entry = key;
        entry.push(b'=');
        entry.extend_from_slice(&value);
        env.push(entry);
    }
    Ok(env)
}

#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
fn sysconf_names() -> &'static [(&'static str, i32)] {
    &[
        ("SC_2_CHAR_TERM", libc::_SC_2_CHAR_TERM),
        ("SC_2_C_BIND", libc::_SC_2_C_BIND),
        ("SC_2_C_DEV", libc::_SC_2_C_DEV),
        ("SC_2_FORT_DEV", libc::_SC_2_FORT_DEV),
        ("SC_2_FORT_RUN", libc::_SC_2_FORT_RUN),
        ("SC_2_LOCALEDEF", libc::_SC_2_LOCALEDEF),
        ("SC_2_SW_DEV", libc::_SC_2_SW_DEV),
        ("SC_2_UPE", libc::_SC_2_UPE),
        ("SC_2_VERSION", libc::_SC_2_VERSION),
        ("SC_AIO_LISTIO_MAX", libc::_SC_AIO_LISTIO_MAX),
        ("SC_AIO_MAX", libc::_SC_AIO_MAX),
        ("SC_AIO_PRIO_DELTA_MAX", libc::_SC_AIO_PRIO_DELTA_MAX),
        ("SC_ARG_MAX", libc::_SC_ARG_MAX),
        ("SC_ASYNCHRONOUS_IO", libc::_SC_ASYNCHRONOUS_IO),
        ("SC_ATEXIT_MAX", libc::_SC_ATEXIT_MAX),
        ("SC_BC_BASE_MAX", libc::_SC_BC_BASE_MAX),
        ("SC_BC_DIM_MAX", libc::_SC_BC_DIM_MAX),
        ("SC_BC_SCALE_MAX", libc::_SC_BC_SCALE_MAX),
        ("SC_BC_STRING_MAX", libc::_SC_BC_STRING_MAX),
        ("SC_CHILD_MAX", libc::_SC_CHILD_MAX),
        ("SC_CLK_TCK", libc::_SC_CLK_TCK),
        ("SC_COLL_WEIGHTS_MAX", libc::_SC_COLL_WEIGHTS_MAX),
        ("SC_DELAYTIMER_MAX", libc::_SC_DELAYTIMER_MAX),
        ("SC_EXPR_NEST_MAX", libc::_SC_EXPR_NEST_MAX),
        ("SC_FSYNC", libc::_SC_FSYNC),
        ("SC_GETGR_R_SIZE_MAX", libc::_SC_GETGR_R_SIZE_MAX),
        ("SC_GETPW_R_SIZE_MAX", libc::_SC_GETPW_R_SIZE_MAX),
        ("SC_IOV_MAX", libc::_SC_IOV_MAX),
        ("SC_JOB_CONTROL", libc::_SC_JOB_CONTROL),
        ("SC_LINE_MAX", libc::_SC_LINE_MAX),
        ("SC_LOGIN_NAME_MAX", libc::_SC_LOGIN_NAME_MAX),
        ("SC_MAPPED_FILES", libc::_SC_MAPPED_FILES),
        ("SC_MEMLOCK", libc::_SC_MEMLOCK),
        ("SC_MEMLOCK_RANGE", libc::_SC_MEMLOCK_RANGE),
        ("SC_MEMORY_PROTECTION", libc::_SC_MEMORY_PROTECTION),
        ("SC_MESSAGE_PASSING", libc::_SC_MESSAGE_PASSING),
        ("SC_MQ_OPEN_MAX", libc::_SC_MQ_OPEN_MAX),
        ("SC_MQ_PRIO_MAX", libc::_SC_MQ_PRIO_MAX),
        ("SC_NGROUPS_MAX", libc::_SC_NGROUPS_MAX),
        ("SC_NPROCESSORS_CONF", libc::_SC_NPROCESSORS_CONF),
        ("SC_NPROCESSORS_ONLN", libc::_SC_NPROCESSORS_ONLN),
        ("SC_OPEN_MAX", libc::_SC_OPEN_MAX),
        ("SC_PAGE_SIZE", libc::_SC_PAGE_SIZE),
        ("SC_PAGESIZE", libc::_SC_PAGE_SIZE),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_PASS_MAX", libc::_SC_PASS_MAX),
        ("SC_PHYS_PAGES", libc::_SC_PHYS_PAGES),
        ("SC_PRIORITIZED_IO", libc::_SC_PRIORITIZED_IO),
        ("SC_PRIORITY_SCHEDULING", libc::_SC_PRIORITY_SCHEDULING),
        ("SC_REALTIME_SIGNALS", libc::_SC_REALTIME_SIGNALS),
        ("SC_RE_DUP_MAX", libc::_SC_RE_DUP_MAX),
        ("SC_RTSIG_MAX", libc::_SC_RTSIG_MAX),
        ("SC_SAVED_IDS", libc::_SC_SAVED_IDS),
        ("SC_SEMAPHORES", libc::_SC_SEMAPHORES),
        ("SC_SEM_NSEMS_MAX", libc::_SC_SEM_NSEMS_MAX),
        ("SC_SEM_VALUE_MAX", libc::_SC_SEM_VALUE_MAX),
        ("SC_SHARED_MEMORY_OBJECTS", libc::_SC_SHARED_MEMORY_OBJECTS),
        ("SC_SIGQUEUE_MAX", libc::_SC_SIGQUEUE_MAX),
        ("SC_STREAM_MAX", libc::_SC_STREAM_MAX),
        ("SC_SYNCHRONIZED_IO", libc::_SC_SYNCHRONIZED_IO),
        ("SC_THREADS", libc::_SC_THREADS),
        ("SC_THREAD_ATTR_STACKADDR", libc::_SC_THREAD_ATTR_STACKADDR),
        ("SC_THREAD_ATTR_STACKSIZE", libc::_SC_THREAD_ATTR_STACKSIZE),
        (
            "SC_THREAD_DESTRUCTOR_ITERATIONS",
            libc::_SC_THREAD_DESTRUCTOR_ITERATIONS,
        ),
        ("SC_THREAD_KEYS_MAX", libc::_SC_THREAD_KEYS_MAX),
        (
            "SC_THREAD_PRIORITY_SCHEDULING",
            libc::_SC_THREAD_PRIORITY_SCHEDULING,
        ),
        ("SC_THREAD_PRIO_INHERIT", libc::_SC_THREAD_PRIO_INHERIT),
        ("SC_THREAD_PRIO_PROTECT", libc::_SC_THREAD_PRIO_PROTECT),
        ("SC_THREAD_PROCESS_SHARED", libc::_SC_THREAD_PROCESS_SHARED),
        ("SC_THREAD_SAFE_FUNCTIONS", libc::_SC_THREAD_SAFE_FUNCTIONS),
        ("SC_THREAD_STACK_MIN", libc::_SC_THREAD_STACK_MIN),
        ("SC_THREAD_THREADS_MAX", libc::_SC_THREAD_THREADS_MAX),
        ("SC_TIMERS", libc::_SC_TIMERS),
        ("SC_TIMER_MAX", libc::_SC_TIMER_MAX),
        ("SC_TTY_NAME_MAX", libc::_SC_TTY_NAME_MAX),
        ("SC_TZNAME_MAX", libc::_SC_TZNAME_MAX),
        ("SC_VERSION", libc::_SC_VERSION),
        ("SC_XOPEN_CRYPT", libc::_SC_XOPEN_CRYPT),
        ("SC_XOPEN_ENH_I18N", libc::_SC_XOPEN_ENH_I18N),
        ("SC_XOPEN_LEGACY", libc::_SC_XOPEN_LEGACY),
        ("SC_XOPEN_REALTIME", libc::_SC_XOPEN_REALTIME),
        (
            "SC_XOPEN_REALTIME_THREADS",
            libc::_SC_XOPEN_REALTIME_THREADS,
        ),
        ("SC_XOPEN_SHM", libc::_SC_XOPEN_SHM),
        ("SC_XOPEN_UNIX", libc::_SC_XOPEN_UNIX),
        ("SC_XOPEN_VERSION", libc::_SC_XOPEN_VERSION),
        ("SC_XOPEN_XCU_VERSION", libc::_SC_XOPEN_XCU_VERSION),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_ILP32_OFF32", libc::_SC_XBS5_ILP32_OFF32),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_ILP32_OFFBIG", libc::_SC_XBS5_ILP32_OFFBIG),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_LP64_OFF64", libc::_SC_XBS5_LP64_OFF64),
        #[cfg(any(
            target_os = "linux",
            target_vendor = "apple",
            target_os = "netbsd",
            target_os = "fuchsia"
        ))]
        ("SC_XBS5_LPBIG_OFFBIG", libc::_SC_XBS5_LPBIG_OFFBIG),
    ]
}

pub(crate) fn walk_fork_callback_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    let mut callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
    let ApplevelForkCallbacks {
        before_w,
        parent_w,
        child_w,
    } = &mut *callbacks;
    for callbacks in [before_w, parent_w, child_w] {
        for callback in callbacks {
            visitor(unsafe { &mut *(callback as *mut usize as *mut PyObjectRef) });
        }
    }
}

fn run_fork_callbacks(kind: &str) {
    let reverse = kind == "before";
    let initial_len = {
        let callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
        match kind {
            "before" => callbacks.before_w.len(),
            "parent" => callbacks.parent_w.len(),
            "child" => callbacks.child_w.len(),
            _ => unreachable!(),
        }
    };
    let indices: Box<dyn Iterator<Item = usize>> = if reverse {
        Box::new((0..initial_len).rev())
    } else {
        Box::new(0..initial_len)
    };
    for index in indices {
        let callback = {
            let callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
            match kind {
                "before" => callbacks.before_w.get(index),
                "parent" => callbacks.parent_w.get(index),
                "child" => callbacks.child_w.get(index),
                _ => unreachable!(),
            }
            .copied()
        };
        let Some(callback) = callback else { continue };
        if let Err(mut error) = crate::call::call_function_impl_result(callback as PyObjectRef, &[])
        {
            error.write_unraisable(
                pyre_object::w_none(),
                rustpython_wtf8::Wtf8::new("fork hook"),
                callback as PyObjectRef,
            );
        }
    }
}

fn register_at_fork(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    if !pos.is_empty() {
        return Err(crate::PyError::type_error(
            "register_at_fork() takes no positional arguments",
        ));
    }
    crate::builtins::kwarg_reject_unknown(
        kwargs,
        &["before", "after_in_parent", "after_in_child"],
        "register_at_fork",
    )?;
    let before = crate::builtins::kwarg_get(kwargs, "before");
    let parent = crate::builtins::kwarg_get(kwargs, "after_in_parent");
    let child = crate::builtins::kwarg_get(kwargs, "after_in_child");
    if before.is_none() && parent.is_none() && child.is_none() {
        return Err(crate::PyError::type_error(
            "At least one argument is required.",
        ));
    }
    for (name, callback) in [
        ("before", before),
        ("after_in_parent", parent),
        ("after_in_child", child),
    ] {
        if callback.is_some_and(|callback| !crate::baseobjspace::callable_w(callback)) {
            return Err(crate::PyError::type_error(format!(
                "'{name}' must be callable",
            )));
        }
    }
    {
        let mut callbacks = APPLEVEL_FORK_CALLBACKS.lock().unwrap();
        if let Some(callback) = before {
            callbacks.before_w.push(callback as usize);
        }
        if let Some(callback) = parent {
            callbacks.parent_w.push(callback as usize);
        }
        if let Some(callback) = child {
            callbacks.child_w.push(callback as usize);
        }
    }
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    Ok(pyre_object::w_none())
}

/// `posix.stat_result` — a real structseq (tuple subclass) so `st[0]`,
/// `len(st)`, iteration and `isinstance(st, tuple)` all work, matching
/// `posixmodule.c` `stat_result_desc`.  The 10 sequence slots hold the
/// integer fields, with the integer-seconds times at 7..10 under the
/// hidden `_integer_atime`/`_integer_mtime`/`_integer_ctime` names; the
/// float `st_atime`/`st_mtime`/`st_ctime`, the `st_*_ns` integers, and the
/// `st_blksize`/`st_blocks`/`st_rdev` block-device fields are named-only
/// extras.
fn stat_result_seq_type() -> PyObjectRef {
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
            ],
        ) as usize
    }) as PyObjectRef
}

/// `os.terminal_size` structseq — `(columns, lines)`.
fn terminal_size_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq("os.terminal_size", &["columns", "lines"]) as usize
    }) as PyObjectRef
}

/// `os.uname_result` structseq — `(sysname, nodename, release, version,
/// machine)`; repr renders "posix.uname_result(...)".
#[cfg(unix)]
fn uname_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "posix.uname_result",
            &["sysname", "nodename", "release", "version", "machine"],
        ) as usize
    }) as PyObjectRef
}

/// `os.statvfs_result` structseq — 10 sequence slots with `f_fsid` as an
/// extra named field (`n_sequence_fields=10`, `n_fields=11`).
fn statvfs_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq_with_extra(
            "os.statvfs_result",
            &[
                "f_bsize",
                "f_frsize",
                "f_blocks",
                "f_bfree",
                "f_bavail",
                "f_files",
                "f_ffree",
                "f_favail",
                "f_flag",
                "f_namemax",
            ],
            &["f_fsid"],
        ) as usize
    }) as PyObjectRef
}

/// `posix.waitid_result` structseq — the five `siginfo_t` fields `waitid`
/// fills (`posixmodule.c waitid_result_fields`). The call is one
/// `interp_posix.py:1722` names and does not carry, so the shape here is the
/// one CPython 3.14 publishes.
#[cfg(all(unix, not(feature = "sandbox")))]
fn waitid_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            "posix.waitid_result",
            &["si_pid", "si_uid", "si_signo", "si_status", "si_code"],
        ) as usize
    }) as PyObjectRef
}

/// `posix.sched_param` structseq — the single field `app_posix.py`
/// declares.  `_structseq.py:102-107` already wraps the scalar a 1-field
/// structseq is handed, so `__new__` only has to name the argument;
/// `__reduce__` has to be replaced outright, because the generic one hands
/// back `(tuple(self), self.__dict__)` and this `__new__` takes one argument.
#[cfg(all(
    unix,
    any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    )
))]
fn sched_param_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        let _roots = pyre_object::gc_roots::push_roots();
        let ty = crate::_structseq::make_struct_seq("posix.sched_param", &["sched_priority"]);
        pyre_object::gc_roots::pin_root(ty);
        let ty_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        let new_descr = crate::typedef::make_new_descr_with_signature(
            crate::_structseq::structseq_descr_new,
            crate::gateway::Signature::new(vec!["cls", "sched_priority"], None, None, 0, 1),
        );
        pyre_object::gc_roots::pin_root(new_descr);
        let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let reduce = crate::make_builtin_function_with_arity("__reduce__", sched_param_reduce, 1);
        pyre_object::gc_roots::pin_root(reduce);
        let reduce_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        unsafe {
            // A store can resize the namespace and collect, which moves the
            // type and the function objects, so every one of them is read back
            // out of its slot and `ns` is re-derived per store.
            let ns = || {
                pyre_object::w_type_get_dict_ptr(pyre_object::gc_roots::shadow_stack_get(ty_slot))
                    as PyObjectRef
            };
            pyre_object::w_dict_setitem_str_no_proxy(
                ns(),
                "__new__",
                pyre_object::gc_roots::shadow_stack_get(new_slot),
            );
            pyre_object::w_dict_setitem_str_no_proxy(
                ns(),
                "__reduce__",
                pyre_object::gc_roots::shadow_stack_get(reduce_slot),
            );
            crate::baseobjspace::mutated(pyre_object::gc_roots::shadow_stack_get(ty_slot), None);
            pyre_object::gc_roots::shadow_stack_get(ty_slot) as usize
        }
    }) as PyObjectRef
}

/// `os_sched_param_reduce` — `(type(self), (self[0],))`, the one shape this
/// type's own `__new__` can be called back with.
#[cfg(all(
    unix,
    any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    )
))]
fn sched_param_reduce(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let Some(&inst) = args.first().filter(|inst| !inst.is_null()) else {
        return Err(crate::PyError::type_error(
            "sched_param.__reduce__ missing self",
        ));
    };
    let cls = unsafe { (*inst).w_class };
    let priority =
        unsafe { pyre_object::w_tuple_getitem(inst, 0) }.unwrap_or_else(pyre_object::w_none);
    // Both tuple allocations can collect, so the class and the element are
    // published first and each one is read back out of its slot at the point
    // it is stored.
    let _roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(cls);
    pyre_object::gc_roots::pin_root(priority);
    let inner = pyre_object::w_tuple_new(vec![pyre_object::gc_roots::shadow_stack_get(base + 1)]);
    pyre_object::gc_roots::pin_root(inner);
    Ok(pyre_object::w_tuple_new(vec![
        pyre_object::gc_roots::shadow_stack_get(base),
        pyre_object::gc_roots::shadow_stack_get(base + 2),
    ]))
}

/// The `w_param` argument `sched_setparam` and `sched_setscheduler` share.
/// `interp_posix.py:3086-3092` refuses anything that is not a `sched_param`,
/// reads field 0 through the sequence protocol, and refuses a priority the C
/// `int` cannot hold.
#[cfg(all(
    unix,
    not(target_env = "musl"),
    any(
        target_os = "android",
        target_os = "freebsd",
        target_os = "linux",
        target_os = "netbsd"
    )
))]
fn sched_priority_w(w_param: PyObjectRef) -> Result<i32, crate::PyError> {
    if !crate::baseobjspace::isinstance(w_param, sched_param_seq_type())? {
        return Err(crate::PyError::type_error("must have a sched_param object"));
    }
    let w_priority = crate::baseobjspace::getitem(w_param, pyre_object::w_int_new(0))?;
    let priority = crate::baseobjspace::int_w(w_priority)?;
    i32::try_from(priority)
        .map_err(|_| crate::PyError::overflow_error("sched_priority out of range"))
}

/// `rpy_cpu_count` — `rposix.py:2968-3006` splits the processor count three
/// ways; these are its two Unix arms, `sysconf(_SC_NPROCESSORS_ONLN)` on linux
/// and gnu and `sysctl(CTL_HW, HW_NCPU)` on the Apple and BSD targets. The
/// third is Windows' and stays with the Windows registration. Anywhere else
/// there is no answer and the count is 0, which `cpu_count` reports as None
/// (`interp_posix.py` `if count <= 0`).
///
/// This is deliberately not the thread count. `get_number_of_os_threads` reads
/// `/proc/self/stat`'s `num_threads` and the mach `task_threads` count, and
/// serves `warn_if_multi_threaded` in the fork path; answering `cpu_count` with
/// it reports how many threads happen to be alive, which moves under the
/// caller's feet.
#[cfg(not(feature = "sandbox"))]
fn host_cpu_count() -> i64 {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let ncpu = {
        let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        if n < 0 { 0 } else { n as i64 }
    };
    #[cfg(any(
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "ios",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd"
    ))]
    let ncpu = {
        let mut ncpu: libc::c_int = 0;
        let mut mib: [libc::c_int; 2] = [libc::CTL_HW, libc::HW_NCPU];
        let mut len = core::mem::size_of::<libc::c_int>();
        let rc = unsafe {
            libc::sysctl(
                mib.as_mut_ptr(),
                2,
                (&mut ncpu as *mut libc::c_int).cast(),
                &mut len,
                core::ptr::null_mut(),
                0,
            )
        };
        if rc != 0 { 0 } else { ncpu as i64 }
    };
    #[cfg(not(any(
        target_os = "android",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "ios",
        target_os = "linux",
        target_os = "macos",
        target_os = "netbsd",
        target_os = "openbsd"
    )))]
    let ncpu = 0i64;
    ncpu
}

/// `os.times_result` structseq — `(user, system, children_user,
/// children_system, elapsed)`; repr renders "posix.times_result(...)", or
/// "nt.times_result(...)" on the host whose module is spelled that way.  The
/// name is the one `pickle` imports to resolve the type, so it has to be the
/// module the host actually has.
fn times_result_seq_type() -> PyObjectRef {
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        crate::_structseq::make_struct_seq(
            if cfg!(windows) {
                "nt.times_result"
            } else {
                "posix.times_result"
            },
            &[
                "user",
                "system",
                "children_user",
                "children_system",
                "elapsed",
            ],
        ) as usize
    }) as PyObjectRef
}

/// Split `path` into the root and everything after it, the way
/// `ntpath.splitroot` splits it three ways with the drive and the root joined
/// back together.
///
/// `_bootstrap_external._path_join` maps this over its parts and classifies
/// each result: a root that starts or ends with a separator is absolute, one
/// ending in `:` is drive-relative, anything else is a plain relative part.
///
/// Both separators count, so the tests are taken on a copy with `/` rewritten
/// to `\`; that rewrite is code-point-for-code-point, so an offset into it
/// addresses the same code point of the original. The offsets are per code
/// point rather than per byte, which is what makes `"ä:\x"` take the drive
/// branch — at byte 1 it would be a continuation byte and take none.
///
/// Compiled everywhere so the tests run on every platform; only the Windows
/// build has a caller.
#[cfg_attr(not(windows), allow(dead_code))]
fn split_root(path: &rustpython_wtf8::Wtf8) -> (&rustpython_wtf8::Wtf8, &rustpython_wtf8::Wtf8) {
    use rustpython_wtf8::Wtf8;

    const SEP: u32 = '\\' as u32;
    const COLON: u32 = ':' as u32;
    const UNC_PREFIX: &str = "\\\\?\\UNC\\";

    // A path is a sequence of code points, and one of them may be a lone
    // surrogate that `str` cannot hold; the separators the split looks for are
    // all ASCII, so the scan runs over the code-point values.
    let norm: Vec<u32> = path
        .code_points()
        .map(|c| {
            if c.to_u32() == '/' as u32 {
                SEP
            } else {
                c.to_u32()
            }
        })
        .collect();
    let byte_at = |index: usize| {
        path.code_point_indices()
            .nth(index)
            .map_or(path.len(), |(offset, _)| offset)
    };
    let split_at = |offset: usize| {
        let bytes = path.as_bytes();
        // The offset comes from `code_point_indices`, so both halves are
        // code-point aligned and stay well-formed WTF-8.
        unsafe {
            (
                Wtf8::from_bytes_unchecked(&bytes[..offset]),
                Wtf8::from_bytes_unchecked(&bytes[offset..]),
            )
        }
    };
    let sep_from = |start: usize| {
        norm.get(start..)
            .and_then(|rest| rest.iter().position(|&c| c == SEP))
            .map(|offset| offset + start)
    };

    if norm.first() != Some(&SEP) {
        if norm.get(1) == Some(&COLON) {
            // `X:\Windows` keeps the separator in the root; `X:Windows` names
            // a location on the drive's own cursor and has no root at all.
            let split = if norm.get(2) == Some(&SEP) { 3 } else { 2 };
            return split_at(byte_at(split));
        }
        return (Wtf8::new(""), path);
    }
    if norm.get(1) != Some(&SEP) {
        // A path rooted on the current drive, e.g. `\Windows`.
        return split_at(byte_at(1));
    }
    // A UNC share (`\\server\share`, `\\?\UNC\server\share`) or a device
    // (`\\.\device`): the root runs to the separator after the share name,
    // and a path that never reaches a second separator is all root.
    let unc = norm.len() >= 8
        && norm[..8]
            .iter()
            .map(|&c| match u8::try_from(c) {
                Ok(b) => u32::from(b.to_ascii_uppercase()),
                Err(_) => c,
            })
            .eq(UNC_PREFIX.chars().map(u32::from));
    let start = if unc { 8 } else { 2 };
    match sep_from(start).and_then(|index| sep_from(index + 1)) {
        Some(index) => split_at(byte_at(index + 1)),
        None => (path, Wtf8::new("")),
    }
}

/// Windows-only `nt` calls — PyPy: pypy/module/posix/interp_nt.py and the
/// `if _WIN32` blocks of interp_posix.py. Registered under the `nt` module
/// name on Windows (moduledef.py `applevel_name = os.name`); `ntpath` reaches
/// for these through `from nt import _getfullpathname` and friends.
#[cfg(all(windows, feature = "host_env"))]
mod win_nt {
    use pyre_object::PyObjectRef;
    use rustpython_host_env::nt as host_nt;

    /// Wrap a host-layer `io::Error` as an OSError carrying the offending path.
    /// These calls are Win32 APIs, so the code they report is a Win32 error and
    /// lands in `.winerror` (`os_error_win32_syscall2`).
    fn io_err(error: &std::io::Error, path: &str) -> crate::PyError {
        let filename = if path.is_empty() {
            pyre_object::PY_NULL
        } else {
            pyre_object::w_str_new(path)
        };
        io_err_with_filename(error, filename)
    }

    fn io_err_with_filename(error: &std::io::Error, filename: PyObjectRef) -> crate::PyError {
        match error.raw_os_error() {
            Some(winerror) => {
                crate::PyError::os_error_win32_syscall2(winerror, filename, pyre_object::PY_NULL)
            }
            None => crate::PyError::os_error_syscall(
                crate::builtins::io_error_posix_errno(error, 0),
                filename,
            ),
        }
    }

    /// Read argument 0 as a filesystem path; the flag reports whether the
    /// input was bytes so the result can be encoded back to match.
    ///
    /// The conversion is the caller-less one. Every entry point reached through
    /// here is Windows-only, so what it should name itself and its argument is
    /// not something this host can measure, and a guessed wording would be
    /// worse than the one uniform gap. See the follow-up task.
    fn arg_path(
        args: &[PyObjectRef],
        func: &str,
    ) -> Result<(widestring::WideCString, bool, crate::gateway::FsEncodedPath), crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(format!(
                "{func}() missing required argument 'path'"
            )));
        };
        let resolved = crate::gateway::fsencode_path_w(arg)?;
        let as_bytes = unsafe { resolved.is_bytes() };
        // Windows names files in UTF-16, so the path reaches the host API as
        // code units rather than bytes. Going through a Rust `String` on the
        // way would replace an undecodable byte with U+FFFD, and the call
        // would then address a different file than the caller named --
        // `interp_posix.py:866-884` keeps the syscall spelling intact for the
        // same reason.
        let wide: Vec<u16> = crate::gateway::fsdecode_filename_wtf8(&resolved.as_bytes)
            .encode_wide()
            .collect();
        let path = widestring::WideCString::from_vec(wide)
            .map_err(|_| crate::PyError::value_error("embedded null character"))?;
        Ok((path, as_bytes, resolved))
    }

    fn wrap_path(s: &std::ffi::OsStr, as_bytes: bool) -> PyObjectRef {
        // One decode feeds both arms: the bytes form is the filesystem
        // encoding of the same text, not the UTF-8 of a lossy rendering of it.
        let text = crate::gateway::fsdecode_os_str_wtf8(s);
        if as_bytes {
            pyre_object::w_bytes_from_bytes(text.as_bytes())
        } else {
            pyre_object::w_str_from_wtf8(text)
        }
    }

    /// ntpath.abspath helper — resolves `.`/`..` and the drive without
    /// requiring the path to exist.
    pub fn _getfullpathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getfullpathname")?;
        match host_nt::getfullpathname(&path) {
            Ok(result) => Ok(wrap_path(&result, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// ntpath.realpath helper — the canonical `\\?\`-prefixed path, via a
    /// backup-semantics handle so directories open too.
    pub fn _getfinalpathname(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_getfinalpathname")?;
        match host_nt::getfinalpathname(&path) {
            Ok(result) => Ok(wrap_path(&result, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// ntpath.realpath helper — the final component's on-disk name, read out of
    /// the `cFileName` FindFirstFileW fills in. This is the only way to learn
    /// the long name behind an 8.3 alias (`C:\PROGRA~1` → `Program Files`) when
    /// the file cannot be opened, so `_getfinalpathname_nonstrict` falls back
    /// to it on the winerrors that mean "found it, but no handle for you"
    /// (`ntpath.py:648-655`). Only the leaf is returned; the caller splits the
    /// parent off itself.
    pub fn _findfirstfile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, as_bytes, resolved) = arg_path(args, "_findfirstfile")?;
        match host_nt::find_first_file_name(&path) {
            Ok(name) => Ok(wrap_path(&name, as_bytes)),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// os.stat helper for ntpath.samefile — (volume serial, file index high,
    /// file index low) uniquely identifies a file across handles. host_env has
    /// no wrapper for GetFileInformationByHandle, so call windows-sys directly.
    pub fn _getfileinformation(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "_getfileinformation() missing required argument 'fd'",
            ));
        };
        let fd = crate::baseobjspace::c_int_w(arg)?;
        let handle = host_nt::handle_from_fd(fd);
        let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(handle, &mut info) } == 0 {
            return Err(io_err(&std::io::Error::last_os_error(), ""));
        }
        Ok(pyre_object::w_tuple_new(vec![
            pyre_object::w_int_new(info.dwVolumeSerialNumber as i64),
            pyre_object::w_int_new(info.nFileIndexHigh as i64),
            pyre_object::w_int_new(info.nFileIndexLow as i64),
        ]))
    }

    /// shutil.disk_usage helper — (total, free) bytes. host_env::getdiskusage
    /// retries against the parent directory when the path names a file
    /// (ERROR_DIRECTORY).
    pub fn _getdiskusage(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (path, _, resolved) = arg_path(args, "_getdiskusage")?;
        match host_nt::getdiskusage(&path) {
            Ok((total, free)) => Ok(pyre_object::w_tuple_new(vec![
                pyre_object::w_int_new(total as i64),
                pyre_object::w_int_new(free as i64),
            ])),
            Err(error) => Err(io_err_with_filename(&error, resolved.w_path())),
        }
    }

    /// os.get_handle_inheritable — the argument is an OS handle value, not a
    /// CRT fd (rwin32.cast(HANDLE, fd)).
    pub fn get_handle_inheritable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "get_handle_inheritable() missing required argument 'handle'",
            ));
        };
        let handle = crate::baseobjspace::c_int_w(arg)? as libc::intptr_t;
        match host_nt::get_handle_inheritable(handle) {
            Ok(value) => Ok(pyre_object::w_bool_from(value)),
            Err(error) => Err(io_err(&error, "")),
        }
    }

    /// os.set_handle_inheritable.
    pub fn set_handle_inheritable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() < 2 {
            return Err(crate::PyError::type_error(
                "set_handle_inheritable() takes 2 arguments",
            ));
        }
        let handle = crate::baseobjspace::c_int_w(args[0])? as libc::intptr_t;
        let inheritable = crate::baseobjspace::is_true(args[1])?;
        match host_nt::set_handle_inheritable(handle, inheritable) {
            Ok(()) => Ok(pyre_object::w_none()),
            Err(error) => Err(io_err(&error, "")),
        }
    }

    /// os._add_dll_directory. PyPy hands back a W_DLLCapsule; os.py only
    /// round-trips the value into `_remove_dll_directory`, so the opaque
    /// DLL_DIRECTORY_COOKIE pointer is returned as an int instead. host_env has
    /// no AddDllDirectory wrapper, so call windows-sys directly.
    pub fn _add_dll_directory(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use windows_sys::Win32::System::LibraryLoader::AddDllDirectory;
        let (path, _, resolved) = arg_path(args, "_add_dll_directory")?;
        // `arg_path` keeps the code units it decoded the path into; going back
        // through a `str` would have no spelling for a lone surrogate and would
        // address a different directory.
        let cookie = unsafe { AddDllDirectory(path.as_ptr()) };
        if cookie.is_null() {
            return Err(io_err_with_filename(
                &std::io::Error::last_os_error(),
                resolved.w_path(),
            ));
        }
        Ok(pyre_object::w_int_new(cookie as usize as i64))
    }

    /// os._remove_dll_directory — takes the cookie returned above.
    pub fn _remove_dll_directory(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        use windows_sys::Win32::System::LibraryLoader::RemoveDllDirectory;
        let Some(&arg) = args.first() else {
            return Err(crate::PyError::type_error(
                "_remove_dll_directory() missing required argument 'cookie'",
            ));
        };
        let cookie = (crate::baseobjspace::int_w(arg)? as usize) as *mut std::ffi::c_void;
        let ok = unsafe { RemoveDllDirectory(cookie) };
        Ok(pyre_object::w_bool_from(ok != 0))
    }

    /// os._supports_virtual_terminal — whether stderr's console mode carries
    /// ENABLE_VIRTUAL_TERMINAL_PROCESSING.
    pub fn _supports_virtual_terminal(
        _args: &[PyObjectRef],
    ) -> Result<PyObjectRef, crate::PyError> {
        Ok(pyre_object::w_bool_from(
            host_nt::supports_virtual_terminal(),
        ))
    }
}

/// A fresh dict holding the current process environment.
///
/// PyPy equivalent: posix.State.startup → `_convertenviron`, which copies the
/// environment into `posix.environ` at interpreter startup. os.py seeds
/// `environ` from it at import time and re-reads it from `reload_environ()`.
fn create_environ() -> pyre_object::PyObjectRef {
    let _roots = pyre_object::gc_roots::push_roots();
    let dict_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
    // Both halves of an entry stay rooted across the store: either allocation
    // may move what the other one already produced.
    #[cfg_attr(
        not(any(feature = "sandbox", feature = "host_env")),
        expect(dead_code, reason = "no environment source is compiled in")
    )]
    fn store(
        dict_slot: usize,
        key: impl FnOnce() -> pyre_object::PyObjectRef,
        value: impl FnOnce() -> pyre_object::PyObjectRef,
    ) {
        let _entry = pyre_object::gc_roots::push_roots();
        let key_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(key());
        let value_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(value());
        unsafe {
            pyre_object::w_dict_store(
                pyre_object::gc_roots::shadow_stack_get(dict_slot),
                pyre_object::gc_roots::shadow_stack_get(key_slot),
                pyre_object::gc_roots::shadow_stack_get(value_slot),
            );
        }
    }
    #[cfg(feature = "sandbox")]
    {
        // The controller delivers the virtual environment as (bytes, bytes).
        if let Ok(items) = crate::host_seam::ops::envitems() {
            for (k_bytes, v_bytes) in items {
                store(
                    dict_slot,
                    || pyre_object::w_bytes_from_bytes(&k_bytes),
                    || pyre_object::w_bytes_from_bytes(&v_bytes),
                );
            }
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox"), not(windows)))]
    {
        // On POSIX, posix.environ stores bytes → bytes. os.py's
        // _create_environ_mapping wraps this dict in an _Environ object that
        // encodes/decodes via surrogateescape when accessed.
        // (_convertenviron: `space.newbytes(key), space.newbytes(value)`.)
        for (key, value) in host_os::vars_os() {
            store(
                dict_slot,
                || pyre_object::w_bytes_from_bytes(key.as_encoded_bytes()),
                || pyre_object::w_bytes_from_bytes(value.as_encoded_bytes()),
            );
        }
    }
    #[cfg(all(feature = "host_env", not(feature = "sandbox"), windows))]
    {
        // On Windows nt.environ stores str → str; os.py's nt branch of
        // _create_environ_mapping demands str keys/values and upper-cases the
        // keys itself. (_convertenviron: `space.newtext(key), newtext(value)`.)
        for (key, value) in host_os::vars_os() {
            // `_convertenviron`'s Windows arm reads `rwin32._wenviron_items()`,
            // the wide-char environment, and keeps those code units.
            // `fsdecode_os_str` carries them across with `from_wide`; a lossy
            // decode would fold an unpaired one to U+FFFD and stop
            // `os.environ` round-tripping.
            store(
                dict_slot,
                || crate::gateway::fsdecode_os_str(&key),
                || crate::gateway::fsdecode_os_str(&value),
            );
        }
    }
    pyre_object::gc_roots::shadow_stack_get(dict_slot)
}

/// `posix_fspath` / `PyOS_FSPath` — `str` and `bytes` pass through unchanged
/// (the protocol's identity case); any other object has `type(path).__fspath__`
/// bound before it is called.
pub(crate) fn fspath(
    arg: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    // `str` and `bytes` only — a `bytearray` is a readable buffer and not a
    // path, so it goes on to be rejected below.
    unsafe {
        if pyre_object::is_str(arg) || pyre_object::bytesobject::is_bytes(arg) {
            return Ok(arg);
        }
    }
    let roots = pyre_object::gc_roots::push_roots();
    let arg_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(arg);
    let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
    let path_type = crate::typedef::r#type(arg);
    if let Some(pt) = path_type
        && let Some(fspath_descr) =
            unsafe { crate::baseobjspace::lookup_in_type(pt.as_ptr(), "__fspath__") }
    {
        let fspath_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(fspath_descr);
        // PyPy's `interp_posix._fspath` binds `__fspath__` before calling it;
        // a non-descriptor is its own value.
        let fspath_fn = unsafe {
            crate::baseobjspace::get(
                pyre_object::gc_roots::shadow_stack_get(fspath_slot),
                pyre_object::gc_roots::shadow_stack_get(arg_slot),
                pt.as_ptr(),
            )?
        }
        .unwrap_or_else(|| pyre_object::gc_roots::shadow_stack_get(fspath_slot));
        let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
        // A `None` left on the type switches the protocol off the way
        // `__hash__ = None` does, so the object is turned away as not
        // path-like and named by its own type.  `_fspath` instead calls what
        // it found, which reports `NoneType` as not callable.
        if unsafe { pyre_object::is_none(fspath_fn) } {
            return Err(crate::PyError::type_error(format!(
                "expected str, bytes or os.PathLike object, not {}",
                crate::gateway::short_type_name(arg)
            )));
        }
        pyre_object::gc_roots::shadow_stack_set(fspath_slot, fspath_fn);
        let result = crate::call::call_function_impl_result(
            pyre_object::gc_roots::shadow_stack_get(fspath_slot),
            &[],
        )?;
        let result_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(result);
        let result = pyre_object::gc_roots::shadow_stack_get(result_slot);
        // The protocol is only satisfied by what a path can be, so an answer
        // that is neither names the object that gave it and the type it gave.
        if unsafe { pyre_object::is_str(result) || pyre_object::bytesobject::is_bytes(result) } {
            return Ok(result);
        }
        return Err(crate::PyError::type_error(format!(
            "expected {}.__fspath__() to return str or bytes, not {}",
            crate::gateway::short_type_name(pyre_object::gc_roots::shadow_stack_get(arg_slot)),
            crate::gateway::short_type_name(result)
        )));
    }
    let arg = pyre_object::gc_roots::shadow_stack_get(arg_slot);
    let error = crate::PyError::type_error(format!(
        "expected str, bytes or os.PathLike object, not {}",
        crate::gateway::short_type_name(arg)
    ));
    drop(roots);
    Err(error)
}

/// posix stub — PyPy: pypy/module/posix/ interp_posix.py
///
/// Provides the minimal surface that os.py module init needs to succeed.
/// Real posix calls are not implemented — they raise or return defaults.
pub fn register_module(ns: pyre_object::PyObjectRef) {
    crate::module_ns_store(ns, "environ", create_environ());
    crate::module_ns_store(
        ns,
        "_create_environ",
        crate::make_builtin_function_with_arity("_create_environ", |_args| Ok(create_environ()), 0),
    );

    // ── posix.putenv(name, value) / posix.unsetenv(name) ──
    // `os.environ.__setitem__` calls `putenv` before updating its own dict, so
    // without these the mapping and the real process environment drift apart:
    // child processes and any native reader of the environment keep seeing the
    // values captured at startup.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    {
        /// The environment block is a list of NUL-terminated `NAME=VALUE`
        /// strings, so neither half may embed a NUL.
        ///
        /// Bytes, like the path boundary: an entry the process was started with
        /// can hold a byte with no UTF-8 spelling, and `posix.environ` already
        /// hands those back as bytes, so writing one must not fold it first.
        fn env_bytes(arg: PyObjectRef) -> Result<Vec<u8>, crate::PyError> {
            let bytes = crate::gateway::fsencode_bytes_w(arg)?;
            if bytes.contains(&0) {
                return Err(crate::PyError::value_error("embedded null byte"));
            }
            Ok(bytes)
        }
        /// A name that is empty or contains `=` cannot be expressed in the
        /// environment block.
        fn illegal_name(name: &[u8]) -> bool {
            name.is_empty() || name.contains(&b'=')
        }
        crate::module_ns_store(
            ns,
            "putenv",
            crate::make_builtin_function_with_arity(
                "putenv",
                |args| {
                    // interp_posix.py putenv_impl rejects the name itself...
                    let name = env_bytes(args[0])?;
                    if illegal_name(&name) {
                        return Err(crate::PyError::value_error(
                            "illegal environment variable name",
                        ));
                    }
                    let value = env_bytes(args[1])?;
                    unsafe {
                        host_os::set_var(os_str_from_bytes(&name), os_str_from_bytes(&value))
                    };
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );
        crate::module_ns_store(
            ns,
            "unsetenv",
            crate::make_builtin_function_with_arity(
                "unsetenv",
                |args| {
                    let name = env_bytes(args[0])?;
                    // ...while unsetenv leaves the same rejection to the
                    // syscall, which reports EINVAL.
                    if illegal_name(&name) {
                        return Err(crate::PyError::os_error_syscall(
                            libc::EINVAL,
                            pyre_object::PY_NULL,
                        ));
                    }
                    unsafe { host_os::remove_var(os_str_from_bytes(&name)) };
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );
    }

    // _have_functions — list of HAVE_* macro names that were defined at
    // build time. os.py uses this to populate the supports_* capability sets
    // (supports_dir_fd / supports_effective_ids / supports_fd /
    // supports_follow_symlinks), which callers like shutil.rmtree consult to
    // choose between fd-relative and path-based implementations. Only the
    // macros whose functionality is actually implemented may be listed — the
    // entry beside each one below names the claim os.py reads out of it, so a
    // bit whose call is still missing has no entry at all rather than a
    // qualified one.
    //
    // Each bit is the same constant the entry point itself branches on, so the
    // advertisement cannot drift from the behaviour: a build where `chdir`
    // rejects a descriptor is a build that does not claim HAVE_FCHDIR. That is
    // what drops the whole fd-relative family under sandbox, where the host
    // probes/mutators are raising stubs, and on the hosts that carry no
    // `host_env::posix` at all.
    let have_functions: &[(&str, bool)] = &[
        // os.py:117,137,158 reads this as `access` honouring all three of its
        // modifiers, which is the one `faccessat` its body makes.
        ("HAVE_FACCESSAT", HAVE_FACCESSAT),
        ("HAVE_FCHDIR", HAVE_FCHDIR),
        ("HAVE_FCHMOD", HAVE_FCHMOD),
        ("HAVE_FCHOWN", HAVE_FCHOWN),
        // os.py:119,180 reads this as `chown` honouring both dir_fd and
        // follow_symlinks. HAVE_LCHOWN is not listed beside it: os.py:186
        // reads either one as the same follow_symlinks capability, and nothing
        // here calls `lchown` — os.lchown is `fchownat` with the flag.
        // os.py:118 reads this as `chmod` honouring dir_fd.
        ("HAVE_FCHMODAT", HAVE_FCHMODAT),
        ("HAVE_FCHOWNAT", HAVE_FCHOWNAT),
        // Do not advertise HAVE_FEXECVE until execve() accepts an open file
        // descriptor.  os.py uses this bit to add execve to supports_fd, and
        // test_posix then runs a fork+fexecve path whose child must never
        // return to the libregrtest worker.
        //
        // os.py reads this as `listdir` and `scandir` taking a
        // descriptor, which `fdlistdir` serves through `fdopendir`.
        ("HAVE_FDOPENDIR", HAVE_FDOPENDIR),
        ("HAVE_FPATHCONF", HAVE_FPATHCONF),
        // os.py:120-121 reads this as `stat` and `lstat` honouring dir_fd.
        ("HAVE_FSTATAT", HAVE_FSTATAT),
        // os.py:122 reads this as `link` honouring `src_dir_fd`/`dst_dir_fd`,
        // which its `linkat` call is.
        ("HAVE_LINKAT", HAVE_LINKAT),
        ("HAVE_FSTATVFS", HAVE_FSTATVFS),
        ("HAVE_FTRUNCATE", HAVE_FTRUNCATE),
        // os.py reads this as `chflags` honouring follow_symlinks, which
        // is its `lchflags` arm.
        ("HAVE_LCHFLAGS", HAVE_LCHFLAGS),
        // os.py:183 reads this as `chmod` honouring follow_symlinks; os.py:179
        // shows why HAVE_FCHMODAT is not read for that claim.
        ("HAVE_LCHMOD", HAVE_LCHMOD),
        // HAVE_FUTIMES is not listed beside it: nothing here calls `futimes`,
        // and os.py:150-151 reads either one as the same `utime` capability.
        ("HAVE_FUTIMENS", HAVE_FUTIMENS),
        ("HAVE_LSTAT", HAVE_LSTAT),
        // os.py reads these as `mkdir`, `mkfifo`, `mknod` and `open`
        // honouring dir_fd.
        ("HAVE_MKDIRAT", HAVE_MKDIRAT),
        ("HAVE_MKFIFOAT", HAVE_MKFIFOAT),
        ("HAVE_MKNODAT", HAVE_MKNODAT),
        ("HAVE_OPENAT", HAVE_OPENAT),
        // os.py:131-132 reads this as `unlink` and `rmdir` honouring dir_fd,
        // which is the one `unlinkat` both of them make. os.remove is not in
        // that set — os.py never names it — even though the call takes the
        // modifier all the same.
        ("HAVE_UNLINKAT", HAVE_UNLINKAT),
        // os.py:133,191 reads this as `utime` honouring both dir_fd and
        // follow_symlinks, which is the one `utimensat` the name form makes.
        // HAVE_LUTIMES is not listed beside it for the same reason as
        // HAVE_FUTIMES above: os.py:188 reads it as the same follow_symlinks
        // capability and nothing here calls `lutimes`.
        ("HAVE_UTIMENSAT", HAVE_UTIMENSAT),
        // `interp_posix.py:2854-2855` appends this after the HAVE_* loop, so
        // it keeps that position here too.
        ("MS_WINDOWS", MS_WINDOWS),
    ];
    crate::module_ns_store(
        ns,
        "_have_functions",
        pyre_object::w_list_new(
            have_functions
                .iter()
                .filter(|&&(_, have)| have)
                .map(|&(n, _)| pyre_object::w_str_new(n))
                .collect(),
        ),
    );
    // POSIX constants — real libc values (cross-platform subset).
    for (name, val) in [
        // F_OK/R_OK/W_OK/X_OK: Windows doesn't have them in libc crate,
        // define standard POSIX values directly.
        #[cfg(unix)]
        ("F_OK", libc::F_OK as i64),
        #[cfg(not(unix))]
        ("F_OK", 0i64),
        #[cfg(unix)]
        ("R_OK", libc::R_OK as i64),
        #[cfg(not(unix))]
        ("R_OK", 4i64),
        #[cfg(unix)]
        ("W_OK", libc::W_OK as i64),
        #[cfg(not(unix))]
        ("W_OK", 2i64),
        #[cfg(unix)]
        ("X_OK", libc::X_OK as i64),
        #[cfg(not(unix))]
        ("X_OK", 1i64),
        ("O_RDONLY", libc::O_RDONLY as i64),
        ("O_WRONLY", libc::O_WRONLY as i64),
        ("O_RDWR", libc::O_RDWR as i64),
        ("O_APPEND", libc::O_APPEND as i64),
        ("O_CREAT", libc::O_CREAT as i64),
        ("O_EXCL", libc::O_EXCL as i64),
        ("O_TRUNC", libc::O_TRUNC as i64),
        // O_NONBLOCK, O_DSYNC, O_SYNC are Unix-only, and `nt` does not carry
        // them -- a zero there is a flag that silently does nothing. The
        // targets that are neither keep the zero they were given.
        #[cfg(unix)]
        ("O_NONBLOCK", libc::O_NONBLOCK as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_NONBLOCK", 0i64),
        #[cfg(unix)]
        ("O_NDELAY", libc::O_NONBLOCK as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_NDELAY", 0i64),
        // `moduledef.py:264-266` publishes O_CLOEXEC by name wherever the host
        // has it, which is every Unix. `nt` has no such flag -- it spells the
        // same intent O_NOINHERIT -- so a zero there would be a flag that
        // silently leaves the descriptor inheritable.
        #[cfg(unix)]
        ("O_CLOEXEC", libc::O_CLOEXEC as i64),
        // The rest of the `<fcntl.h>` set. Each value is the host header's own,
        // and the split below is the hosts' own too: these six are on every
        // Unix, the next two groups are one platform's each. `nt` has none of
        // them and is left with the flags it does have.
        #[cfg(unix)]
        ("O_ACCMODE", libc::O_ACCMODE as i64),
        #[cfg(unix)]
        ("O_ASYNC", libc::O_ASYNC as i64),
        #[cfg(unix)]
        ("O_DIRECTORY", libc::O_DIRECTORY as i64),
        #[cfg(unix)]
        ("O_FSYNC", libc::O_FSYNC as i64),
        #[cfg(unix)]
        ("O_NOCTTY", libc::O_NOCTTY as i64),
        #[cfg(unix)]
        ("O_NOFOLLOW", libc::O_NOFOLLOW as i64),
        // Linux's own. O_LARGEFILE is 0 on the targets that are already 64-bit,
        // which is the header answering that there is nothing to widen.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_DIRECT", libc::O_DIRECT as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_LARGEFILE", libc::O_LARGEFILE as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_NOATIME", libc::O_NOATIME as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_PATH", libc::O_PATH as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_RSYNC", libc::O_RSYNC as i64),
        #[cfg(any(target_os = "linux", target_os = "android"))]
        ("O_TMPFILE", libc::O_TMPFILE as i64),
        // The Apple targets' own.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_EVTONLY", libc::O_EVTONLY as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_EXEC", libc::O_EXEC as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_EXLOCK", libc::O_EXLOCK as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_NOFOLLOW_ANY", libc::O_NOFOLLOW_ANY as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_SEARCH", libc::O_SEARCH as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_SHLOCK", libc::O_SHLOCK as i64),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        ("O_SYMLINK", libc::O_SYMLINK as i64),
        #[cfg(unix)]
        ("O_DSYNC", libc::O_DSYNC as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_DSYNC", 0i64),
        #[cfg(unix)]
        ("O_SYNC", libc::O_SYNC as i64),
        #[cfg(not(any(unix, windows)))]
        ("O_SYNC", 0i64),
        // SEEK_SET/SEEK_CUR/SEEK_END are os.py's own (`SEEK_SET = 0`,
        // os.py) on every platform, named in its own `__all__`, so a
        // binding here is counted a second time through the star-import.
        // Neither `posix` nor `nt` carries them; the other SEEK_* values are
        // the module's to publish.
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
    }
    // Windows-only open() mode flags (fcntl.h). os.py exposes these off `nt`,
    // and stdlib callers reach for them behind `hasattr(os, 'O_BINARY')`
    // probes (tempfile) that only succeed once the names are bound.
    #[cfg(windows)]
    for (name, val) in [
        ("O_BINARY", libc::O_BINARY as i64),
        ("O_TEXT", libc::O_TEXT as i64),
        ("O_NOINHERIT", libc::O_NOINHERIT as i64),
        ("O_TEMPORARY", libc::O_TEMPORARY as i64),
        ("O_SHORT_LIVED", libc::_O_SHORT_LIVED as i64),
        ("O_RANDOM", libc::O_RANDOM as i64),
        ("O_SEQUENTIAL", libc::O_SEQUENTIAL as i64),
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
    }
    // Placeholders the POSIX blocks further down overwrite with the real libc
    // values — the wait options beside the `W*` predicates, the `PRIO_*` trio
    // beside `getpriority`. A build that reaches neither keeps the zero.
    fn install_zero_constants(ns: PyObjectRef, names: &[&str]) {
        for &name in names {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(0));
        }
    }

    // The wait flags and the priority classes are `nt`'s absentees for the same
    // reason its calls are: they exist only where the platform defines them,
    // and code reads their presence to decide whether the facility is there at
    // all.
    #[cfg(unix)]
    install_zero_constants(
        ns,
        &[
            "WNOHANG",
            "WCONTINUED",
            "WUNTRACED",
            "PRIO_PROCESS",
            "PRIO_PGRP",
            "PRIO_USER",
        ],
    );

    // `nt`'s own constants. The spawn modes are the C runtime's `_P_*`
    // (process.h) and carry its values: registering the set at zero made
    // `P_NOWAIT` mean `P_WAIT`. On POSIX these are os.py's, not the module's —
    // it defines `P_WAIT = 0` and `P_NOWAIT = P_NOWAITO = 1` for itself in the
    // branch that has `fork`, which is why `P_NOWAITO` is 3 here and 1 there.
    //
    // `EX_OK` is the one member of the `<sysexits.h>` family Windows answers to
    // as well, and it carries the same 0 there.
    #[cfg(windows)]
    for (name, val) in [
        ("EX_OK", 0i64),
        ("P_WAIT", 0i64),
        ("P_NOWAIT", 1),
        ("P_OVERLAY", 2),
        ("P_NOWAITO", 3),
        ("P_DETACH", 4),
        // The number of names `tempfile` will try before giving up.
        ("TMP_MAX", 2_147_483_647),
        // LoadLibraryEx search flags, which `os.add_dll_directory` and
        // `ctypes` pass through.
        ("_LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR", 0x100),
        ("_LOAD_LIBRARY_SEARCH_APPLICATION_DIR", 0x200),
        ("_LOAD_LIBRARY_SEARCH_USER_DIRS", 0x400),
        ("_LOAD_LIBRARY_SEARCH_SYSTEM32", 0x800),
        ("_LOAD_LIBRARY_SEARCH_DEFAULT_DIRS", 0x1000),
    ] {
        crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
    }
    #[cfg(unix)]
    {
        // `<sysexits.h>`. The header is a verbatim descendant of the 4.3BSD one
        // wherever it is carried, so the values are the same on every host that
        // has it and the `libc` crate binds none of them.
        for (name, val) in [
            ("EX_OK", 0i64),
            ("EX_USAGE", 64),
            ("EX_DATAERR", 65),
            ("EX_NOINPUT", 66),
            ("EX_NOUSER", 67),
            ("EX_NOHOST", 68),
            ("EX_UNAVAILABLE", 69),
            ("EX_SOFTWARE", 70),
            ("EX_OSERR", 71),
            ("EX_OSFILE", 72),
            ("EX_CANTCREAT", 73),
            ("EX_IOERR", 74),
            ("EX_TEMPFAIL", 75),
            ("EX_PROTOCOL", 76),
            ("EX_NOPERM", 77),
            ("EX_CONFIG", 78),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
        // The `f_flag` bits `statvfs` answers with, which is the only reader
        // there is for them.
        for (name, val) in [
            ("ST_RDONLY", libc::ST_RDONLY as i64),
            ("ST_NOSUID", libc::ST_NOSUID as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
        // `<dlfcn.h>` — `rdynload.py:50-82` reads the same set, and
        // `sys.setdlopenflags` and `ctypes` hand them straight back to
        // `dlopen`, where a zero would ask for `RTLD_LOCAL | RTLD_LAZY`
        // whatever was named.
        for (name, val) in [
            ("RTLD_LAZY", libc::RTLD_LAZY as i64),
            ("RTLD_NOW", libc::RTLD_NOW as i64),
            ("RTLD_GLOBAL", libc::RTLD_GLOBAL as i64),
            ("RTLD_LOCAL", libc::RTLD_LOCAL as i64),
            ("RTLD_NODELETE", libc::RTLD_NODELETE as i64),
            ("RTLD_NOLOAD", libc::RTLD_NOLOAD as i64),
            // A glibc extension, absent from the header anywhere else.
            #[cfg(all(target_os = "linux", target_env = "gnu"))]
            ("RTLD_DEEPBIND", libc::RTLD_DEEPBIND as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
        // `<sched.h>`, read as `rposix.py` reads it: present where the
        // header defines it, and with the host's own numbering rather than a
        // shared one — the three disagree between Linux, the BSDs and Darwin.
        // The Apple targets declare them in `<pthread/pthread_impl.h>`, which
        // the `libc` crate does not mirror.
        for (name, val) in [
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            ("SCHED_OTHER", libc::SCHED_OTHER as i64),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ("SCHED_OTHER", 1i64),
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            ("SCHED_FIFO", libc::SCHED_FIFO as i64),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ("SCHED_FIFO", 4i64),
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            ("SCHED_RR", libc::SCHED_RR as i64),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            ("SCHED_RR", 2i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_BATCH", libc::SCHED_BATCH as i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_IDLE", libc::SCHED_IDLE as i64),
            // `<linux/sched.h>` names these two; `SCHED_RESET_ON_FORK` is a
            // flag OR-ed into a policy rather than a policy of its own.
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_NORMAL", libc::SCHED_NORMAL as i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_DEADLINE", libc::SCHED_DEADLINE as i64),
            #[cfg(any(target_os = "linux", target_os = "android"))]
            ("SCHED_RESET_ON_FORK", libc::SCHED_RESET_ON_FORK as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
        // The `cmd` `lockf` takes, which is the whole of its vocabulary and
        // which os.py neither writes nor names.
        for (name, val) in [
            ("F_ULOCK", libc::F_ULOCK as i64),
            ("F_LOCK", libc::F_LOCK as i64),
            ("F_TLOCK", libc::F_TLOCK as i64),
            ("F_TEST", libc::F_TEST as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
        // The two `whence` values beyond the three os.py fixes itself: they
        // seek to the next hole or the next data in a sparse file. A host that
        // cannot answer that question defines neither, and the OpenBSD/NetBSD
        // and AIX headers are among those — so the set is named rather than
        // excluded, and a host left out of it is one short of a name rather
        // than one carrying a wrong value.
        #[cfg(any(
            target_os = "macos",
            target_os = "ios",
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd",
            target_os = "dragonfly",
            target_os = "solaris",
            target_os = "illumos",
            target_os = "hurd",
        ))]
        for (name, val) in [
            ("SEEK_HOLE", libc::SEEK_HOLE as i64),
            ("SEEK_DATA", libc::SEEK_DATA as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
    }
    // Remaining noop stubs — functions os.py references at module level.
    // Functions with real implementations are registered individually below.
    fn install_noop_stubs(ns: PyObjectRef, names: &[&'static str]) {
        for &name in names {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, |_| Ok(pyre_object::w_none())),
            );
        }
    }

    // Names both `posix` and `nt` answer to.
    install_noop_stubs(
        ns,
        &[
            "dup",
            "dup2",
            "chdir",
            "link",
            "symlink",
            "chmod",
            "fchmod",
            "access",
            "execve",
            "execv",
            "waitpid",
            "truncate",
            "ftruncate",
            "getppid",
            "umask",
            "getlogin",
            "pipe",
            "fsync",
            "get_inheritable",
            "set_inheritable",
            // "get_terminal_size" — implemented below
            "cpu_count",
            "kill",
            "device_encoding",
            "waitstatus_to_exitcode",
            "_exit",
            "abort",
            // "spawnv"/"spawnve" — os.py builds the spawn family out of
            // fork+exec+waitpid, but only `if not _exists("spawnv")`, so a name
            // bound here is not a placeholder waiting to be overwritten: it is
            // what stops the real implementation from ever being defined. That
            // reading is the POSIX one: `nt` carries `_spawnv` itself and the
            // os.py block is behind `_exists("fork")`, so on Windows the name
            // is the module's or it is nowhere, and it is registered below.
            "system",
        ],
    );

    // The spawn entry points `nt` has of its own. There is no fork on Windows
    // for os.py:881 to write them over, so unbinding them here would not hand
    // the definition back to os.py — it would delete the name.
    #[cfg(windows)]
    install_noop_stubs(ns, &["spawnv", "spawnve"]);

    // The calls `nt` has not got. Each is probed for presence rather than
    // called blind — `os.py` gates `supports_fd` on `_exists`, `shutil` picks
    // its `disk_usage` implementation on `hasattr(os, 'statvfs')`, and
    // `multiprocessing` picks a start method on `hasattr(os, 'fork')` — so a
    // stub answering `None` here does not add a call, it wins the POSIX branch
    // on a host that cannot serve it. Registered where the platform is one that
    // can, which is the only place the name exists at all.
    #[cfg(unix)]
    install_noop_stubs(
        ns,
        &[
            // "fstatat"/"faccessat"/"futimens"/"futimes"/"fdopendir" — the `*at`
            // and `f*` C entry points the module calls to serve `dir_fd` and a
            // descriptor path. They are how the calls above are made, not calls
            // of their own, and `moduledef.py` publishes none of them.
            "statvfs",
            "fstatvfs",
            "fchdir",
            "fchown",
            "fork",
            "forkpty",
            "wait",
            "pathconf",
            "fpathconf",
            "setsid",
            "setpgid",
            "getgroups",
            "setgroups",
            "setpgrp",
            "nice",
            // "pipe2" — the flag-taking form of `pipe`, published below on the
            // hosts whose libc declares it. "dup3" is not a name `moduledef.py`
            // defines, nor one `os` publishes on any host, so there is nothing
            // here for it to stand in for.
            "fdatasync",
            "mkfifo",
            "getloadavg",
            "killpg",
            "getpriority",
            "setpriority",
            "sched_get_priority_max",
            "sched_get_priority_min",
            // "sched_getparam"/"sched_setparam"/"sched_getscheduler"/
            // "sched_setscheduler" — the policy calls, published below together
            // with the `sched_param` type they hand back and forth.
            // `moduledef.py:168-174` gates the five as one group.
            "sched_yield",
            // "confstr"/"confstr_names" — the host's string-valued configuration
            // table, published below where the host defines one. A build with no
            // `<unistd.h>` behind it has no `confstr` at all, which is what the
            // name being absent says.
            "sysconf",
            "sysconf_names",
            // "setenv" — the entry point is spelled `putenv`, and there is no
            // second name for it.
            "ttyname",
            "openpty",
            "login_tty",
            "tcgetpgrp",
            "tcsetpgrp",
            // "get_exec_path" — `os.py` writes it in Python and lists it in
            // its own `__all__`, so a name bound here is not overwritten by that
            // definition; it is counted a second time, through the star-import.
            "WIFEXITED",
            "WEXITSTATUS",
            "WIFSIGNALED",
            "WTERMSIG",
            "WIFSTOPPED",
            "WSTOPSIG",
            // "WEXITED"/"WNOWAIT"/"WSTOPPED" — `waitid`'s option flags, which
            // are numbers rather than calls; bound with the other wait options
            // below.
            "_cpu_count",
            // "spawnvp"/"spawnvpe" — the same os.py branch defines these,
            // for the same reason the two above are not bound.
            // "popen" — `os.py` writes it over `subprocess` and
            // appends it to its own `__all__`, with no guard on this name being
            // free.
        ],
    );

    // putenv/unsetenv are implemented above unless the host environment is out
    // of reach.
    #[cfg(any(not(feature = "host_env"), feature = "sandbox"))]
    install_noop_stubs(ns, &["unsetenv", "putenv"]);
    // There is no fork to register against on Windows, and `os.py` reaches for
    // the name to decide whether it has one.
    #[cfg(not(windows))]
    crate::module_ns_store(
        ns,
        "register_at_fork",
        crate::make_builtin_function("register_at_fork", register_at_fork),
    );

    // os.major(device) / os.minor(device) / os.makedev(major, minor)
    // (`interp_posix.py:2551-2563`) — how a device number is taken apart and
    // put back together, which is the host's own encoding and not arithmetic
    // that can be spelled portably. `tarfile` reads a node's pair out of
    // `st_rdev` to write a header (`tarfile.py`) and puts one back
    // together to recreate the node (`:2735`), so a `None` here writes a
    // header field that is not a number.
    //
    // No syscall, but the encoding is still the host's, and the sandbox build
    // reaches libc through a shim that carries no `dev_t` — so the names are
    // absent there rather than answering with another host's arithmetic.
    // `moduledef.py:152-157` registers each only where the host has it.
    #[cfg(all(unix, not(feature = "sandbox")))]
    {
        fn device_u64_w(value: PyObjectRef) -> Result<u64, crate::PyError> {
            let value = if unsafe {
                pyre_object::is_bool(value)
                    || pyre_object::is_int(value)
                    || pyre_object::is_long(value)
            } {
                value
            } else {
                crate::baseobjspace::space_index(value)?
            };
            crate::baseobjspace::uint_w(value)
        }
        fn device_value_w(value: PyObjectRef) -> Result<libc::dev_t, crate::PyError> {
            let indexed = crate::baseobjspace::space_index(value)?;
            // Reading the sentinel must not be fallible: a device number above
            // `i64::MAX` has no machine-word form, so propagating `int_w`'s
            // overflow here would refuse a value `uint_w` below accepts.
            #[cfg(all(target_os = "linux", not(target_env = "musl")))]
            if matches!(crate::baseobjspace::int_w(indexed), Ok(-1)) {
                return Ok(-1i64 as libc::dev_t);
            }
            let value = crate::baseobjspace::uint_w(indexed)?;
            // `dev_t` is signed on some targets and unsigned on others, so the
            // ceiling comes from the type rather than from its signedness.
            let max = u64::try_from(libc::dev_t::MAX).unwrap_or(u64::MAX);
            if value > max {
                return Err(crate::PyError::overflow_error(
                    "Python int too large to convert to C dev_t",
                ));
            }
            Ok(value as libc::dev_t)
        }
        fn device_w(args: &[PyObjectRef]) -> Result<libc::dev_t, crate::PyError> {
            let Some(&value) = args.first() else {
                return Err(crate::PyError::type_error("device is required"));
            };
            device_value_w(value)
        }
        fn major_minor_result(value: i64) -> PyObjectRef {
            #[cfg(all(target_os = "linux", not(target_env = "musl")))]
            if value == -1 || value == libc::c_uint::MAX as i64 {
                return pyre_object::w_int_new(-1);
            }
            pyre_object::w_int_new(value)
        }
        fn major_minor_arg(value: PyObjectRef) -> Result<libc::dev_t, crate::PyError> {
            // Where `NODEV` is spelled -1 that one value passes through, rather
            // than being rejected as out of range for an unsigned field.
            #[cfg(all(target_os = "linux", not(target_env = "musl")))]
            let value = {
                let indexed = crate::baseobjspace::space_index(value)?;
                if crate::baseobjspace::int_w(indexed)? == -1 {
                    return Ok(-1i64 as libc::dev_t);
                }
                crate::baseobjspace::uint_w(indexed)?
            };
            #[cfg(not(all(target_os = "linux", not(target_env = "musl"))))]
            let value = device_u64_w(value).map_err(|err| {
                if err.kind == crate::PyErrorKind::OverflowError {
                    crate::PyError::overflow_error(
                        "Python int too large to convert to C unsigned int",
                    )
                } else {
                    err
                }
            })?;
            if value > libc::c_uint::MAX as u64 {
                return Err(crate::PyError::overflow_error(
                    "Python int too large to convert to C unsigned int",
                ));
            }
            Ok(value as libc::dev_t)
        }
        crate::module_ns_store(
            ns,
            "major",
            crate::make_builtin_function_with_arity(
                "major",
                |args| Ok(major_minor_result(libc::major(device_w(args)?) as i64)),
                1,
            ),
        );
        crate::module_ns_store(
            ns,
            "minor",
            crate::make_builtin_function_with_arity(
                "minor",
                |args| Ok(major_minor_result(libc::minor(device_w(args)?) as i64)),
                1,
            ),
        );
        crate::module_ns_store(
            ns,
            "makedev",
            crate::make_builtin_function_with_arity(
                "makedev",
                |args| {
                    let (major, minor) = match args {
                        [major, minor, ..] => (*major, *minor),
                        _ => return Err(crate::PyError::type_error("makedev takes 2 arguments")),
                    };
                    let major = major_minor_arg(major)?;
                    let minor = major_minor_arg(minor)?;
                    Ok(pyre_object::w_int_new(
                        libc::makedev(major as _, minor as _) as i64,
                    ))
                },
                2,
            ),
        );
    }

    // PyPy `interp_posix.get_blocking/set_blocking` → rposix
    // `get_blocking/set_blocking`: inspect or update O_NONBLOCK with fcntl.
    fn get_blocking(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() != 1 {
            return Err(crate::PyError::type_error(format!(
                "get_blocking() takes exactly one argument ({} given)",
                args.len()
            )));
        }
        let fd = libc::c_int::try_from(crate::builtins::space_index_w(args[0])?)
            .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            // PyPy passes `eintr_retry=False`; CPython 3.14's
            // `_Py_get_blocking` likewise performs one F_GETFL call.
            let flags = unsafe { libc::fcntl(fd, libc::F_GETFL) };
            if flags < 0 {
                let error = std::io::Error::last_os_error();
                return Err(errno_err(error.raw_os_error().unwrap_or(libc::EIO), ""));
            }
            Ok(pyre_object::w_bool_from(flags & libc::O_NONBLOCK == 0))
        }
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::System::Pipes::{GetNamedPipeHandleStateW, PIPE_NOWAIT};

            // CPython 3.14 `_Py_get_blocking`: translate the CRT descriptor
            // through the host seam, then read PIPE_NOWAIT from the pipe's
            // current mode.  GetNamedPipeHandleStateW is one of the narrow nt
            // calls host_env does not wrap, like the other windows-sys calls
            // owned by this crate.
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(|error| {
                    crate::PyError::os_error_syscall(
                        crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                        pyre_object::PY_NULL,
                    )
                })?;
            let handle = rustpython_host_env::crt_fd::as_handle(borrowed).map_err(|error| {
                crate::PyError::os_error_syscall(
                    crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                    pyre_object::PY_NULL,
                )
            })?;
            let mut mode = 0;
            let success = unsafe {
                GetNamedPipeHandleStateW(
                    handle.as_raw_handle() as _,
                    &mut mode,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            };
            if success == 0 {
                return Err(crate::PyError::os_error_win32_syscall2(
                    rustpython_host_env::winapi::get_last_error() as i32,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ));
            }
            Ok(pyre_object::w_bool_from(mode & PIPE_NOWAIT == 0))
        }
        #[cfg(any(
            feature = "sandbox",
            all(not(unix), not(all(windows, feature = "host_env")))
        ))]
        {
            let _ = fd;
            Err(crate::PyError::not_implemented(
                "get_blocking is unavailable on this target",
            ))
        }
    }

    fn set_blocking(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        if args.len() != 2 {
            return Err(crate::PyError::type_error(format!(
                "set_blocking() takes exactly two arguments ({} given)",
                args.len()
            )));
        }
        let fd = libc::c_int::try_from(crate::builtins::space_index_w(args[0])?)
            .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
        // CPython 3.14's Argument Clinic declares this parameter `bool`, so
        // it truth-tests arbitrary objects.  This intentionally differs from
        // PyPy's older `@unwrap_spec(blocking=int)` gateway.
        let blocking = crate::baseobjspace::is_true(args[1])?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let mut flags = loop {
                let result = unsafe { libc::fcntl(fd, libc::F_GETFL) };
                if result >= 0 {
                    break result;
                }
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EINTR) {
                    return Err(errno_err(error.raw_os_error().unwrap_or(libc::EIO), ""));
                }
            };
            if blocking {
                flags &= !libc::O_NONBLOCK;
            } else {
                flags |= libc::O_NONBLOCK;
            }
            loop {
                let result = unsafe { libc::fcntl(fd, libc::F_SETFL, flags) };
                if result >= 0 {
                    break;
                }
                let error = std::io::Error::last_os_error();
                if error.raw_os_error() != Some(libc::EINTR) {
                    return Err(errno_err(error.raw_os_error().unwrap_or(libc::EIO), ""));
                }
            }
            Ok(pyre_object::w_none())
        }
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            use std::os::windows::io::AsRawHandle;
            use windows_sys::Win32::System::Pipes::{GetNamedPipeHandleStateW, PIPE_NOWAIT};

            // CPython 3.14 `_Py_set_blocking` maps a CRT descriptor back to
            // its pipe HANDLE and flips PIPE_NOWAIT with
            // SetNamedPipeHandleState.  rustpython-host_env owns both unsafe
            // host boundaries; keep the PyPy `set_blocking` entry point and
            // argument conversion around them.
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(|error| {
                    crate::PyError::os_error_syscall(
                        crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                        pyre_object::PY_NULL,
                    )
                })?;
            let handle = rustpython_host_env::crt_fd::as_handle(borrowed).map_err(|error| {
                crate::PyError::os_error_syscall(
                    crate::builtins::io_error_posix_errno(&error, libc::EBADF),
                    pyre_object::PY_NULL,
                )
            })?;
            let mut mode = 0;
            let success = unsafe {
                GetNamedPipeHandleStateW(
                    handle.as_raw_handle() as _,
                    &mut mode,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                )
            };
            if success == 0 {
                return Err(crate::PyError::os_error_win32_syscall2(
                    rustpython_host_env::winapi::get_last_error() as i32,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ));
            }
            if blocking {
                mode &= !PIPE_NOWAIT;
            } else {
                mode |= PIPE_NOWAIT;
            }
            rustpython_host_env::winapi::set_named_pipe_handle_state(
                handle.as_raw_handle(),
                Some(mode),
                None,
                None,
            )
            .map_err(|error| match error.raw_os_error() {
                Some(winerror) => crate::PyError::os_error_win32_syscall2(
                    winerror,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ),
                None => crate::PyError::os_error_syscall(
                    crate::builtins::io_error_posix_errno(&error, libc::EINVAL),
                    pyre_object::PY_NULL,
                ),
            })?;
            Ok(pyre_object::w_none())
        }
        #[cfg(any(
            feature = "sandbox",
            all(not(unix), not(all(windows, feature = "host_env")))
        ))]
        {
            let _ = (fd, blocking);
            Err(crate::PyError::not_implemented(
                "set_blocking is unavailable on this target",
            ))
        }
    }

    crate::module_ns_store(
        ns,
        "get_blocking",
        crate::make_builtin_function("get_blocking", get_blocking),
    );
    crate::module_ns_store(
        ns,
        "set_blocking",
        crate::make_builtin_function("set_blocking", set_blocking),
    );

    // `baseobjspace.py fsencode_w` returns filesystem bytes; syscall
    // boundaries must not pass through a Rust `String`.
    use crate::gateway::fsencode_bytes_w as extract_path;

    /// The descriptor an `fd` argument names, as the borrowed handle the host
    /// API takes one as. `-1` is the single value `BorrowedFd::borrow_raw`
    /// refuses — the standard library reserves it as the niche that makes
    /// `Option<BorrowedFd>` free — so a caller who names it gets the `EBADF`
    /// the call would have answered with rather than a handle built out of the
    /// one integer that may not become one.
    #[cfg(unix)]
    fn fd_borrow(fd: libc::c_int) -> Result<std::os::fd::BorrowedFd<'static>, crate::PyError> {
        if fd == -1 {
            return Err(errno_err(libc::EBADF, ""));
        }
        Ok(unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) })
    }

    /// The host-API view of OS bytes — a filename, or a half of an environment
    /// entry. Unix spells both in bytes and takes them back unchanged.
    fn os_str_from_bytes(bytes: &[u8]) -> std::borrow::Cow<'_, std::ffi::OsStr> {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStrExt;
            std::borrow::Cow::Borrowed(std::ffi::OsStr::from_bytes(bytes))
        }
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStringExt;
            // PEP 529 spells a name as UTF-8 over the host's UTF-16, so the
            // bytes decode back to the code units the API takes — including an
            // unpaired surrogate, which is a unit a name may carry.  A lossy
            // decode would substitute U+FFFD, addressing a different name and
            // making distinct entries alias onto one another.
            let units: Vec<u16> = crate::typedef::fsdecode_wtf8_total(bytes)
                .encode_wide()
                .collect();
            std::borrow::Cow::Owned(std::ffi::OsString::from_wide(&units))
        }
        #[cfg(not(any(unix, windows)))]
        {
            // This platform has no byte spelling, so the host API necessarily
            // receives the best text representation of these bytes.
            std::borrow::Cow::Owned(std::ffi::OsString::from(
                String::from_utf8_lossy(bytes).into_owned(),
            ))
        }
    }

    fn path_from_bytes(path: &[u8]) -> std::borrow::Cow<'_, std::path::Path> {
        match os_str_from_bytes(path) {
            std::borrow::Cow::Borrowed(s) => std::borrow::Cow::Borrowed(std::path::Path::new(s)),
            std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(std::path::PathBuf::from(s)),
        }
    }

    // ── Helper: convert std::io::Error → PyError (OSError) ──
    fn errno_err(errno: i32, path: &str) -> crate::PyError {
        let w_filename = if path.is_empty() {
            pyre_object::PY_NULL
        } else {
            pyre_object::w_str_new(path)
        };
        crate::PyError::os_error_syscall(errno, w_filename)
    }

    // `interp_posix.py` keeps the resolved `Path.w_path` as
    // `OSError.filename`.
    fn errno_err_with_filename(errno: i32, w_path: PyObjectRef) -> crate::PyError {
        crate::PyError::os_error_syscall(errno, w_path)
    }

    fn io_err(e: std::io::Error, path: &str) -> crate::PyError {
        errno_err(crate::builtins::io_error_posix_errno(&e, 0), path)
    }

    fn io_err_with_filename(e: std::io::Error, w_path: PyObjectRef) -> crate::PyError {
        errno_err_with_filename(crate::builtins::io_error_posix_errno(&e, 0), w_path)
    }

    /// The OSError for a failed *filesystem* call, whose error code Windows
    /// reports through `GetLastError` rather than `errno`: it becomes the
    /// `.winerror` attribute and picks up the system's message, the way
    /// `os.stat` and `os.mkdir` report `[WinError 3]`.  The descriptor calls
    /// (`open`, `read`, `write`, `close`, `lseek`) go through the C runtime and
    /// keep `io_err`'s errno form, which is what they report as well.
    ///
    /// `default_errno` stands in when the error carries no OS code at all.
    fn fs_err_with_filename2(
        e: std::io::Error,
        default_errno: i32,
        w_path: PyObjectRef,
        w_path2: PyObjectRef,
    ) -> crate::PyError {
        #[cfg(windows)]
        if let Some(winerror) = e.raw_os_error() {
            return crate::PyError::os_error_win32_syscall2(winerror, w_path, w_path2);
        }
        crate::PyError::os_error_syscall2(
            crate::builtins::io_error_posix_errno(&e, default_errno),
            w_path,
            w_path2,
        )
    }

    /// The width the platform's truncating call takes its length in.
    #[cfg(windows)]
    type TruncateLen = i64;
    #[cfg(not(windows))]
    type TruncateLen = libc::off_t;

    /// `space.int_w` over the `r_longlong` half of `interp_posix.py:404`.
    ///
    /// `Py_off_t_converter` names the C type it could not fit the value
    /// into, and that type is the platform's: a `long` where the converter
    /// is `PyLong_AsLong`, and nothing at all where it is
    /// `PyLong_AsLongLong`.
    fn truncate_length_w(obj: PyObjectRef) -> Result<TruncateLen, crate::PyError> {
        const TOO_BIG: &str = if cfg!(windows) {
            "int too big to convert"
        } else {
            "Python int too large to convert to C long"
        };
        let w_length = crate::baseobjspace::space_index(obj)?;
        let length = crate::baseobjspace::int_w(w_length).map_err(|err| {
            if err.kind == crate::PyErrorKind::OverflowError {
                crate::PyError::overflow_error(TOO_BIG)
            } else {
                err
            }
        })?;
        // `off_t` is the width the call takes the length in, and a value
        // above it is not a size the file can be given. An `as` cast would
        // wrap it into one the caller never asked for and truncate the file
        // to that instead.
        TruncateLen::try_from(length).map_err(|_| crate::PyError::overflow_error(TOO_BIG))
    }

    fn fs_err_with_filename(e: std::io::Error, w_path: PyObjectRef) -> crate::PyError {
        fs_err_with_filename2(e, 0, w_path, pyre_object::PY_NULL)
    }

    /// The wide (UTF-16) spelling of a path, for the Windows entry points that
    /// take one.  The narrow entry points re-encode through the ANSI code page,
    /// which has no spelling for most of what a filesystem name may hold, so
    /// every path call takes the `W` form.  A name holding an interior NUL is
    /// no more nameable there than it is to `CString`.
    #[cfg(all(windows, feature = "host_env"))]
    fn wide_path(bytes: &[u8]) -> Result<widestring::WideCString, crate::PyError> {
        let name = os_str_from_bytes(bytes);
        widestring::WideCString::from_os_str(&*name)
            .map_err(|_| crate::PyError::value_error("embedded null in path"))
    }

    /// A filesystem name reported back to the caller: `bytes` when the path
    /// argument was `bytes` (`posixmodule.c path_converter`), else `str`.
    ///
    /// The name arrives as raw bytes because `readdir`'s `d_name` is handed
    /// back unchanged. Both arms keep it openable: bytes mode returns it
    /// verbatim, and the `str` arm takes the filesystem decoding
    /// (`interp_posix.py:1121 space.newfilename(f)`), so a name like
    /// `b"bad_\xff"` still names the file on disk instead of carrying U+FFFD.
    fn fs_name_obj(bytes_mode: bool, name: &[u8]) -> PyObjectRef {
        if bytes_mode {
            pyre_object::bytesobject::w_bytes_from_bytes(name)
        } else {
            crate::gateway::fsdecode_filename_bytes(name)
        }
    }

    /// interp_posix.py `unwrap_fd`.
    ///
    /// ```python
    /// def unwrap_fd(space, w_value, allowed_types='integer'):
    ///     try:
    ///         result = space.c_int_w(w_value)
    ///     except OperationError as e:
    ///         if not e.match(space, space.w_OverflowError):
    ///             raise oefmt(space.w_TypeError,
    ///                 "argument should be %s, not %T", allowed_types, w_value)
    ///         else:
    ///             raise
    ///     if result == -1:
    ///         # -1 is used as sentinel value for not a fd
    ///         raise oefmt(space.w_OSError, "invalid file descriptor: -1")
    ///     return result
    /// ```
    ///
    /// `c_int_w` is the load-bearing part: it converts through `__index__`,
    /// so an `int` subclass — an `IntEnum` member, say — reaches the syscall,
    /// where an exact-type test would reject it and a raw payload read would
    /// interpret the instance's first word as the descriptor.
    fn unwrap_fd(value: PyObjectRef, allowed_types: &str) -> Result<i32, crate::PyError> {
        if unsafe { pyre_object::is_bool(value) } {
            crate::warn::warn_category("bool is used as a file descriptor", "RuntimeWarning", 1)?;
        }
        let result = crate::baseobjspace::c_int_w(value).map_err(|err| {
            if err.kind == crate::PyErrorKind::OverflowError {
                err
            } else {
                crate::PyError::type_error(format!(
                    "argument should be {allowed_types}, not {}",
                    crate::baseobjspace::object_functionstr_type_name(value)
                ))
            }
        })?;
        if result == -1 {
            return Err(crate::PyError::os_error("invalid file descriptor: -1"));
        }
        Ok(result)
    }

    /// The `*, dir_fd=None` tail, read the way `DirFD(available)` reads it
    /// (`interp_posix.py:274-292`): `None` and an absent argument are the same
    /// `DEFAULT_DIR_FD`, and the value is converted before the platform is
    /// reported, so a wrongly typed one is a TypeError even on a build that
    /// carries no `*at` call to honour it.
    fn dir_fd_kwarg(
        kwargs: Option<pyre_object::PyObjectRef>,
        have: bool,
    ) -> Result<Option<i32>, crate::PyError> {
        match crate::builtins::kwarg_get(kwargs, "dir_fd") {
            Some(v) if !unsafe { pyre_object::is_none(v) } => {
                let fd = unwrap_fd(v, "integer or None")?;
                if !have {
                    return Err(dir_fd_unavailable());
                }
                Ok(Some(fd))
            }
            _ => Ok(None),
        }
    }

    /// Bind an entry point whose positional parameters all sit before the
    /// clinic `/`. None of them binds by name, so the count is over
    /// positionals alone.
    ///
    /// `kwonly` decides which parser reports a bad count, and the two word it
    /// differently. With no keyword-capable parameter at all the call is
    /// parsed by `_PyArg_CheckPositional`, whose wording carries neither the
    /// trailing `()` nor a parenthesised count, and every keyword is refused
    /// against the module-qualified name. A keyword-only tail puts the call
    /// back on `_PyArg_UnpackKeywords`, which reports the positional count in
    /// the parenthesised form and accepts the named modifiers.
    fn bind_posonly_args(
        args: &[pyre_object::PyObjectRef],
        name: &str,
        qualname: &str,
        total: usize,
        required: usize,
        kwonly: &[&'static str],
    ) -> Result<
        (
            Vec<Option<pyre_object::PyObjectRef>>,
            Option<pyre_object::PyObjectRef>,
        ),
        crate::PyError,
    > {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        if kwonly.is_empty() && crate::builtins::real_kwarg_count(kwargs) > 0 {
            return Err(crate::PyError::type_error(format!(
                "{qualname}() takes no keyword arguments"
            )));
        }
        // The count is checked before the keyword names: a call that supplies
        // neither the positionals nor a recognised keyword is reported against
        // the positionals.
        if pos.len() < required || pos.len() > total {
            let plural = if total == 1 { "" } else { "s" };
            let text = if !kwonly.is_empty() {
                let limit = if required == total {
                    "exactly"
                } else {
                    "at most"
                };
                format!(
                    "{name}() takes {limit} {total} positional argument{plural} ({} given)",
                    pos.len()
                )
            } else if required == total {
                format!(
                    "{name} expected {total} argument{plural}, got {}",
                    pos.len()
                )
            } else {
                let bound = if pos.len() > total { total } else { required };
                let plural = if bound == 1 { "" } else { "s" };
                let at = if pos.len() > total {
                    "at most"
                } else {
                    "at least"
                };
                format!(
                    "{name} expected {at} {bound} argument{plural}, got {}",
                    pos.len()
                )
            };
            return Err(crate::PyError::type_error(text));
        }
        crate::builtins::kwarg_reject_unknown(kwargs, kwonly, name)?;
        Ok((
            (0..total).map(|index| pos.get(index).copied()).collect(),
            kwargs,
        ))
    }

    /// Bind the positional-or-keyword prefix of a path-taking entry point.
    /// `params` names that prefix in order, `path` first, and the leading
    /// `required` of them carry no default; the rest are reported absent as
    /// `None`. `kwonly` names the keyword-only tail, which is left in the
    /// returned kwargs dict — which `HAVE_*` bit each modifier answers to is
    /// the caller's business.
    ///
    /// A surplus argument is reported the way the entry point's own generated
    /// parser reports it, and the two forms differ. Where there is a
    /// keyword-only tail the count is over positionals alone, and a signature
    /// with no defaults says "exactly" where one with defaults says "at most";
    /// where there is none, every argument counts toward the one limit and it
    /// is always "at most" — which is why `os.lchflags(p, 0, follow_symlinks=1)`
    /// is a count error and not an unknown keyword.
    fn bind_path_args(
        args: &[pyre_object::PyObjectRef],
        name: &str,
        params: &[&'static str],
        required: usize,
        kwonly: &[&'static str],
    ) -> Result<
        (
            Vec<Option<pyre_object::PyObjectRef>>,
            Option<pyre_object::PyObjectRef>,
        ),
        crate::PyError,
    > {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        let count = params.len();
        let plural = if count == 1 { "" } else { "s" };
        if kwonly.is_empty() {
            let given = pos.len() + crate::builtins::real_kwarg_count(kwargs);
            if given > count {
                return Err(crate::PyError::type_error(format!(
                    "{name}() takes at most {count} argument{plural} ({given} given)"
                )));
            }
        } else if pos.len() > count {
            let limit = if required == count {
                "exactly"
            } else {
                "at most"
            };
            return Err(crate::PyError::type_error(format!(
                "{name}() takes {limit} {count} positional argument{plural} ({} given)",
                pos.len()
            )));
        }
        let mut allowed: Vec<&str> = params.to_vec();
        allowed.extend_from_slice(kwonly);
        crate::builtins::kwarg_reject_unknown(kwargs, &allowed, name)?;
        let mut bound = Vec::with_capacity(params.len());
        for (index, key) in params.iter().enumerate() {
            let value = crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)?;
            if value.is_none() && index < required {
                return Err(crate::PyError::type_error(format!(
                    "{name}() missing required argument '{key}' (pos {})",
                    index + 1
                )));
            }
            bound.push(value);
        }
        Ok((bound, kwargs))
    }

    // ── posix.open(path, flags, mode=0o777, *, dir_fd=None) → fd ──
    crate::module_ns_store(
        ns,
        "open",
        crate::make_builtin_function("open", |args| {
            let (bound, kwargs) =
                bind_path_args(args, "open", &["path", "flags", "mode"], 2, &["dir_fd"])?;
            let path = crate::gateway::fsencode_path_or_fd_w(
                bound[0].expect("path is required"),
                "open",
                false,
            )?;
            let flags =
                crate::baseobjspace::c_int_w(bound[1].expect("flags is required"))? as libc::c_int;
            let mode: u32 = match bound[2] {
                Some(value) => crate::baseobjspace::c_int_w(value)? as u32,
                None => 0o777,
            };
            // `open` types `dir_fd` as `DirFD(rposix.HAVE_OPENAT)`
            // (`interp_posix.py`). Only the `openat` arm below reads it;
            // every other build has already turned a descriptor away, because
            // `HAVE_OPENAT` is what those builds do not claim.
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_OPENAT)?;
            #[cfg(not(feature = "sandbox"))]
            let fd = {
                // Open the fd non-inheritable (PEP 446) so the descriptor does
                // not leak across exec into child processes: O_CLOEXEC on unix,
                // O_NOINHERIT on Windows (O_CLOEXEC is unix-only in libc). Moot
                // under sandbox, where the controller hands out virtual fds, so
                // it is applied only here.
                #[cfg(unix)]
                let flags = flags | libc::O_CLOEXEC;
                #[cfg(windows)]
                let flags = flags | libc::O_NOINHERIT;
                // `_wopen`, so the name reaches the filesystem intact.
                #[cfg(all(windows, feature = "host_env"))]
                let (fd, errno) = {
                    let wide = wide_path(&path.as_bytes)?;
                    crate::module::thread::call_external_function(|| {
                        rustpython_host_env::crt_fd::wopen(&wide, flags, mode as i32)
                            .map_or(-1, |owned| owned.into_raw())
                    })
                };
                #[cfg(not(all(windows, feature = "host_env")))]
                let (fd, errno) = {
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    // interp_posix.py `open`: the syscall sits inside the
                    // `eintr_retry=True` loop.  A FIFO opened without O_NONBLOCK
                    // is the case that makes this reachable — it waits for a peer,
                    // and an alarm arriving meanwhile must run its handler rather
                    // than surface as `InterruptedError`.
                    loop {
                        let (fd, errno) = crate::module::thread::call_external_function(|| {
                            // `openat` resolves the name against the descriptor;
                            // the plain `open` is what a name without one means
                            // (`interp_posix.py:325-329`).
                            #[cfg(unix)]
                            if let Some(dir_fd) = _dir_fd {
                                return unsafe {
                                    libc::openat(
                                        dir_fd,
                                        c_path.as_ptr(),
                                        flags,
                                        mode as libc::c_uint,
                                    )
                                };
                            }
                            unsafe { libc::open(c_path.as_ptr(), flags, mode as libc::c_uint) }
                        });
                        if fd >= 0 {
                            break (fd, errno);
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| {
                                errno_err_with_filename(
                                    e.raw_os_error().unwrap_or(0),
                                    path.w_path(),
                                )
                            },
                        )?;
                    }
                };
                if fd < 0 {
                    return Err(errno_err_with_filename(errno, path.w_path()));
                }
                fd
            };
            #[cfg(feature = "sandbox")]
            let fd = crate::host_seam::ops::open(&path.as_bytes, flags, mode)
                .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_int_new(fd as i64))
        }),
    );

    // ── posix.close(fd) ──
    crate::module_ns_store(
        ns,
        "close",
        crate::make_builtin_function_with_arity(
            "close",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("close() requires 1 argument"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                #[cfg(not(feature = "sandbox"))]
                {
                    let ret = crate::builtins::crt_call!(libc::close(fd));
                    if ret < 0 {
                        return Err(errno_err(crate::builtins::crt_errno(), ""));
                    }
                }
                #[cfg(feature = "sandbox")]
                crate::host_seam::ops::close(fd)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );

    // ── posix.closerange(fd_low, fd_high) ──
    // Half-open, and every failure is dropped: the point of the call is to shut
    // whatever is open in the range without first asking what that is, which is
    // how `subprocess` closes the parent's descriptors in the child. Closing one
    // the process does not have is the ordinary case, not an error — on Windows
    // it is what would take the C runtime's invalid-parameter handler and abort
    // the process, so the close goes through `crt_call!` like every other.
    crate::module_ns_store(
        ns,
        "closerange",
        crate::make_builtin_function_with_arity(
            "closerange",
            |args| {
                if args.len() != 2 {
                    return Err(crate::PyError::type_error(format!(
                        "closerange() takes exactly 2 arguments ({} given)",
                        args.len()
                    )));
                }
                let low = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                let high = crate::baseobjspace::c_int_w(args[1])? as libc::c_int;
                for fd in low..high {
                    #[cfg(not(feature = "sandbox"))]
                    let _ = crate::builtins::crt_call!(libc::close(fd));
                    #[cfg(feature = "sandbox")]
                    let _ = crate::host_seam::ops::close(fd);
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );

    // ── posix.strerror(code) ──
    // The C runtime's message table, which is the one `OSError.strerror`
    // already reports from — the two answer alike for the same errno.
    crate::module_ns_store(
        ns,
        "strerror",
        crate::make_builtin_function_with_arity(
            "strerror",
            |args| {
                if args.len() != 1 {
                    return Err(crate::PyError::type_error(format!(
                        "strerror() takes exactly 1 argument ({} given)",
                        args.len()
                    )));
                }
                let code = crate::baseobjspace::c_int_w(args[0])?;
                Ok(pyre_object::w_str_new(&crate::PyError::clean_strerror(
                    code,
                )))
            },
            1,
        ),
    );

    // ── posix.read(fd, n) → bytes ──
    crate::module_ns_store(
        ns,
        "read",
        crate::make_builtin_function_with_arity(
            "read",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("read() requires 2 arguments"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                let n_signed = crate::baseobjspace::int_w(args[1])?;
                // A negative size would wrap to a huge `usize` (and allocation);
                // os.read rejects it with EINVAL, matching the host read(2).
                if n_signed < 0 {
                    return Err(crate::PyError::os_error_with_errno(
                        libc::EINVAL,
                        "read: negative size",
                    ));
                }
                let n = n_signed as usize;
                #[cfg(not(feature = "sandbox"))]
                let buf = {
                    let mut buf = vec![0u8; n];
                    // interp_posix.py `read`: the syscall sits inside
                    // the `eintr_retry=True` loop, so an interrupted read runs
                    // the pending signal handlers and is re-issued rather than
                    // surfacing as `InterruptedError`.  The blocking guard is
                    // scoped to the syscall alone: `checksignals` runs Python.
                    loop {
                        let (ret, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, n as _)
                            });
                        if ret >= 0 {
                            buf.truncate(ret as usize);
                            break buf;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                };
                #[cfg(feature = "sandbox")]
                let buf = crate::host_seam::ops::read(fd, n as i64)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            2,
        ),
    );

    // Python 3.14 `os.readinto`: acquire one writable buffer export for the
    // complete `_Py_read(fd, buffer->buf, buffer->len)` call and return the
    // number of bytes transferred without allocating an intermediate bytes
    // object on the real-host path.
    crate::module_ns_store(
        ns,
        "readinto",
        crate::make_builtin_function_with_arity(
            "readinto",
            |args| {
                let fd_value = crate::baseobjspace::int_w(args[0])?;
                let fd = libc::c_int::try_from(fd_value)
                    .map_err(|_| crate::PyError::overflow_error("fd is greater than maximum"))?;
                let mut buffer = unsafe { crate::builtins::WritableBuffer::acquire(args[1]) }?;
                let target = unsafe { buffer.as_mut_slice() };
                #[cfg(not(feature = "sandbox"))]
                let result = loop {
                    #[cfg(all(windows, feature = "host_env"))]
                    let read_result = crate::builtins::fd_read_into(fd, &mut *target);
                    #[cfg(not(all(windows, feature = "host_env")))]
                    let (result, errno) =
                        crate::module::thread::call_external_function(|| unsafe {
                            libc::read(
                                fd,
                                target.as_mut_ptr() as *mut libc::c_void,
                                target.len() as _,
                            )
                        });
                    #[cfg(all(windows, feature = "host_env"))]
                    let (result, errno) = match read_result {
                        Ok(result) => (result as i64, 0),
                        Err(error) => (-1, error.raw_os_error().unwrap_or(libc::EIO)),
                    };
                    if result >= 0 {
                        break result as i64;
                    }
                    // The retry used to skip the handlers, so an interrupted
                    // read never let the handler that supplies the remaining
                    // bytes run.  Guard dropped first: `checksignals` runs
                    // Python.
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(errno),
                        |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                    )?;
                };
                #[cfg(feature = "sandbox")]
                let result = {
                    let data = crate::host_seam::ops::read(fd, target.len() as i64)
                        .map_err(|error| crate::host_seam::seam_os_err(error, ""))?;
                    let length = data.len().min(target.len());
                    target[..length].copy_from_slice(&data[..length]);
                    length as i64
                };
                Ok(pyre_object::w_int_new(result))
            },
            2,
        ),
    );

    // ── posix.write(fd, data) → nbytes ──
    crate::module_ns_store(
        ns,
        "write",
        crate::make_builtin_function_with_arity(
            "write",
            |args| {
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("write() requires 2 arguments"));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                // CPython `os_write_impl` receives a `Py_buffer`: text is not
                // accepted, while every contiguous readable exporter is.
                let data = unsafe { crate::builtins::file_write_buffer_bytes(args[1]) }
                    .map_err(|_| crate::PyError::type_error("write() arg 2 must be bytes-like"))?;
                #[cfg(not(feature = "sandbox"))]
                let ret = {
                    // interp_posix.py `write`: the syscall sits inside
                    // the `eintr_retry=True` loop, so an interrupted write runs
                    // the pending signal handlers and is re-issued rather than
                    // surfacing as `InterruptedError`.  The blocking guard is
                    // scoped to the syscall alone: `checksignals` runs Python.
                    loop {
                        let (ret, errno) = crate::builtins::crt_write_once(fd, &data);
                        if ret >= 0 {
                            break ret as i64;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                };
                #[cfg(feature = "sandbox")]
                let ret = crate::host_seam::ops::write(fd, &data)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_int_new(ret))
            },
            2,
        ),
    );

    // ── posix.lseek(fd, offset, whence) → position ──
    crate::module_ns_store(
        ns,
        "lseek",
        crate::make_builtin_function_with_arity(
            "lseek",
            |args| {
                if args.len() < 3 {
                    return Err(crate::PyError::type_error("lseek() requires 3 arguments"));
                }
                // interp_posix.py `@unwrap_spec(fd=c_int, position=r_longlong,
                // how=c_int)` — the position is a 64-bit offset, not a C int.
                let fd = crate::baseobjspace::c_int_w(args[0])? as libc::c_int;
                let offset = crate::baseobjspace::int_w(args[1])?;
                let whence = crate::baseobjspace::c_int_w(args[2])? as libc::c_int;
                #[cfg(not(feature = "sandbox"))]
                let ret = {
                    let ret = crate::builtins::crt_lseek(fd, offset, whence);
                    if ret < 0 {
                        return Err(errno_err(crate::builtins::crt_errno(), ""));
                    }
                    ret
                };
                #[cfg(feature = "sandbox")]
                let ret = crate::host_seam::ops::lseek(fd, offset, whence)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_int_new(ret))
            },
            3,
        ),
    );

    // ── posix.unlink(path, *, dir_fd=None) / posix.remove(path, *, dir_fd=None) ──
    // `remove` is `unlink` written out a second time under its own name
    // (`interp_posix.py:827-869`), so it reports itself by that name.
    fn posix_unlink(
        args: &[pyre_object::PyObjectRef],
        name: &str,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        let (bound, kwargs) = bind_path_args(args, name, &["path"], 1, &["dir_fd"])?;
        let path = crate::gateway::fsencode_path_or_fd_w(
            bound[0].expect("path is required"),
            name,
            false,
        )?;
        // Both take `DirFD(rposix.HAVE_UNLINKAT)` (`interp_posix.py`).
        let _dir_fd = dir_fd_kwarg(kwargs, HAVE_UNLINKAT)?;
        // `DeleteFileW`, except on a directory symlink, which `RemoveDirectoryW`
        // unlinks without following (`os_unlink_impl`).
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        rustpython_host_env::nt::remove(&wide_path(&path.as_bytes)?)
            .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
        #[cfg(all(not(all(windows, feature = "host_env")), not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            // `unlinkat` without `AT_REMOVEDIR` is the name form resolved
            // against a descriptor (`rposix.py:2717-2720`).
            #[cfg(unix)]
            let ret = match _dir_fd {
                Some(dir_fd) => unsafe { libc::unlinkat(dir_fd, c_path.as_ptr(), 0) },
                None => unsafe { libc::unlink(c_path.as_ptr()) },
            };
            #[cfg(not(unix))]
            let ret = unsafe { libc::unlink(c_path.as_ptr()) };
            if ret < 0 {
                return Err(fs_err_with_filename(
                    std::io::Error::last_os_error(),
                    path.w_path(),
                ));
            }
        }
        #[cfg(feature = "sandbox")]
        crate::host_seam::ops::unlink(&path.as_bytes)
            .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
        Ok(pyre_object::w_none())
    }
    crate::module_ns_store(
        ns,
        "unlink",
        crate::make_builtin_function("unlink", |args| posix_unlink(args, "unlink")),
    );
    crate::module_ns_store(
        ns,
        "remove",
        crate::make_builtin_function("remove", |args| posix_unlink(args, "remove")),
    );

    // ── posix.readlink(path, *, dir_fd=None) ──
    // Returns the symlink target; a non-symlink raises OSError(EINVAL), which
    // `posixpath.realpath` relies on to stop following links.
    // Under sandbox readlink is unavailable (the controller has no ll_os
    // readlink handler); the stub override loop registers a raising stub, so
    // keep the raw std::fs::read_link body out of the sandbox build.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "readlink",
        crate::make_builtin_function("readlink", |args| {
            let (bound, kwargs) = bind_path_args(args, "readlink", &["path"], 1, &["dir_fd"])?;
            // `readlink` types `dir_fd` as `DirFD(rposix.HAVE_READLINKAT)`.
            // This build resolves the name through `std::fs::read_link`, which
            // has no at-variant, so a descriptor is refused rather than
            // silently resolved against the working directory — matching what
            // `os.supports_dir_fd` advertises.
            let _dir_fd = dir_fd_kwarg(kwargs, false)?;
            let path = crate::gateway::fsencode_path_named_w(
                bound[0].expect("path is required"),
                "readlink",
                "path",
            )?;
            let bytes_mode = unsafe { path.is_bytes() };
            match std::fs::read_link(path_from_bytes(&path.as_bytes).as_ref()) {
                Ok(target) => {
                    let target = target.as_os_str().as_encoded_bytes();
                    Ok(fs_name_obj(bytes_mode, target))
                }
                Err(e) => Err(fs_err_with_filename(e, path.w_path())),
            }
        }),
    );

    // ── posix.mkdir(path, mode=0o777, *, dir_fd=None) ──
    crate::module_ns_store(
        ns,
        "mkdir",
        crate::make_builtin_function("mkdir", |args| {
            let (bound, kwargs) = bind_path_args(args, "mkdir", &["path", "mode"], 1, &["dir_fd"])?;
            let path = crate::gateway::fsencode_path_or_fd_w(
                bound[0].expect("path is required"),
                "mkdir",
                false,
            )?;
            let _mode: u32 = match bound[1] {
                Some(value) => crate::baseobjspace::c_int_w(value)? as u32,
                None => 0o777,
            };
            // `mkdir` types `dir_fd` as `DirFD(rposix.HAVE_MKDIRAT)`
            // (`interp_posix.py:921`).
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_MKDIRAT)?;
            // `CreateDirectoryW`; a mode of 0o700 is served by the security
            // descriptor that denies everyone but the owner (`os_mkdir_impl`).
            #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
            rustpython_host_env::nt::mkdir(&wide_path(&path.as_bytes)?, _mode as i32)
                .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
            #[cfg(all(not(all(windows, feature = "host_env")), not(feature = "sandbox")))]
            {
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `mkdirat` resolves the name against the descriptor
                // (`rposix.py:2708-2710`).
                #[cfg(unix)]
                let ret = match _dir_fd {
                    Some(dir_fd) => unsafe {
                        libc::mkdirat(dir_fd, c_path.as_ptr(), _mode as libc::mode_t)
                    },
                    None => unsafe { libc::mkdir(c_path.as_ptr(), _mode as libc::mode_t) },
                };
                #[cfg(windows)]
                let ret = unsafe { libc::mkdir(c_path.as_ptr()) };
                if ret < 0 {
                    return Err(fs_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
            }
            #[cfg(feature = "sandbox")]
            crate::host_seam::ops::mkdir(&path.as_bytes, _mode)
                .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.rmdir(path, *, dir_fd=None) ──
    // Mutates the host filesystem; stubbed under sandbox, so the real body
    // (and its libc call) is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "rmdir",
        crate::make_builtin_function("rmdir", |args| {
            let (bound, kwargs) = bind_path_args(args, "rmdir", &["path"], 1, &["dir_fd"])?;
            let path = crate::gateway::fsencode_path_or_fd_w(
                bound[0].expect("path is required"),
                "rmdir",
                false,
            )?;
            // Removing a directory is the same call as removing a file, so
            // `rmdir` reads the same bit: `DirFD(rposix.HAVE_UNLINKAT)`
            // (`interp_posix.py:942`).
            let _dir_fd = dir_fd_kwarg(kwargs, HAVE_UNLINKAT)?;
            // `RemoveDirectoryW`, which is what `std::fs::remove_dir` is on
            // Windows (`os_rmdir_impl`).
            #[cfg(windows)]
            std::fs::remove_dir(path_from_bytes(&path.as_bytes).as_ref())
                .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
            #[cfg(not(windows))]
            {
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `AT_REMOVEDIR` is what makes the one `unlinkat` a `rmdir`
                // (`rposix.py:2717-2720` `removedir=True`).
                let ret = match _dir_fd {
                    Some(dir_fd) => unsafe {
                        libc::unlinkat(dir_fd, c_path.as_ptr(), libc::AT_REMOVEDIR)
                    },
                    None => unsafe { libc::rmdir(c_path.as_ptr()) },
                };
                if ret < 0 {
                    return Err(fs_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
            }
            Ok(pyre_object::w_none())
        }),
    );

    // ── posix.rename / posix.replace(src, dst, *, src_dir_fd=None,
    //    dst_dir_fd=None) ──
    // A non-None `src_dir_fd` / `dst_dir_fd` resolves the path relative to the
    // open directory descriptor (`renameat`); the descriptors are only usable
    // where `renameat` exists (unix).
    //
    // The two entry points take the same arguments and differ only in what a
    // pre-existing `dst` does: `replace` overwrites it on every platform,
    // `rename` leaves that to the platform call. On Windows the host layer's
    // `replace` uses MoveFileExW(MOVEFILE_REPLACE_EXISTING); plain `rename`
    // deliberately omits that flag.
    fn rename_impl(
        args: &[PyObjectRef],
        name: &'static str,
    ) -> Result<PyObjectRef, crate::PyError> {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.len() < 2 {
            return Err(crate::PyError::type_error(format!(
                "{name}() requires 2 arguments"
            )));
        }
        if pos.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "{name}() takes exactly 2 positional arguments ({} given)",
                pos.len()
            )));
        }
        crate::builtins::kwarg_reject_unknown(kwargs, &["src_dir_fd", "dst_dir_fd"], name)?;
        // `rename` and `replace` are one body here and two argument-clinic
        // declarations there, so the rejected argument is named after whichever
        // of the two the caller reached.
        let src = crate::gateway::fsencode_path_named_w(pos[0], name, "src")?;
        let dst = crate::gateway::fsencode_path_named_w(pos[1], name, "dst")?;
        let dir_fd = |name: &str| -> Result<Option<i32>, crate::PyError> {
            match crate::builtins::kwarg_get(kwargs, name) {
                // interp_posix.py `_unwrap_dirfd` — a non-`None` value
                // goes through `unwrap_fd` with `allowed_types="integer or
                // None"`.
                Some(v) if !unsafe { pyre_object::is_none(v) } => {
                    Ok(Some(unwrap_fd(v, "integer or None")?))
                }
                _ => Ok(None),
            }
        };
        let src_fd = dir_fd("src_dir_fd")?;
        let dst_fd = dir_fd("dst_dir_fd")?;
        #[cfg(unix)]
        let (src_b, dst_b) = {
            use rustpython_host_env::crt_fd::Borrowed;
            (
                src_fd.map(|fd| unsafe { Borrowed::borrow_raw(fd) }),
                dst_fd.map(|fd| unsafe { Borrowed::borrow_raw(fd) }),
            )
        };
        #[cfg(not(unix))]
        let (src_b, dst_b) = {
            if src_fd.is_some() || dst_fd.is_some() {
                return Err(crate::PyError::not_implemented(
                    "dir_fd unavailable on this platform",
                ));
            }
            (None, None)
        };
        let src_path = path_from_bytes(&src.as_bytes);
        let dst_path = path_from_bytes(&dst.as_bytes);
        let result = if name == "replace" {
            host_os::replace(src_path.as_ref(), src_b, dst_path.as_ref(), dst_b)
        } else {
            host_os::rename(src_path.as_ref(), src_b, dst_path.as_ref(), dst_b)
        };
        // interp_posix.py hands both resolved `Path.w_path` objects to
        // `wrap_oserror2`.
        result.map_err(|e| fs_err_with_filename2(e, 0, src.w_path(), dst.w_path()))?;
        Ok(pyre_object::w_none())
    }
    crate::module_ns_store(
        ns,
        "rename",
        crate::make_builtin_function("rename", |args| rename_impl(args, "rename")),
    );
    crate::module_ns_store(
        ns,
        "replace",
        crate::make_builtin_function("replace", |args| rename_impl(args, "replace")),
    );

    // os.utime(path, times=None, *, ns=None, dir_fd=None, follow_symlinks=True)
    /// One of the two times `utime` writes, kept the way `rposix.futimens` and
    /// `rposix.utimensat` keep it (`rposix.py`): the seconds and the
    /// nanoseconds apart, both signed, so a time before the epoch is the
    /// negative second it names rather than a value with no representation.
    /// The nanoseconds are always the ones after that second — `1969-12-31
    /// 23:59:59.999999999` is `(-1, 999999999)`, which is the `ns=-1` a caller
    /// asked for.
    #[derive(Clone, Copy)]
    struct UTime {
        sec: i64,
        nsec: i64,
    }

    /// `interp_posix.py:1901-1904` answers a descriptor with `futimens`, which
    /// is the call HAVE_FUTIMENS names.
    fn utime_fd(
        fd: i32,
        now: bool,
        access: UTime,
        modified: UTime,
    ) -> Result<PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let times = [timespec_of(access, now), timespec_of(modified, now)];
            if unsafe { libc::futimens(fd, times.as_ptr()) } < 0 {
                return Err(io_err(std::io::Error::last_os_error(), ""));
            }
            return Ok(pyre_object::w_none());
        }
        #[allow(unreachable_code)]
        {
            let _ = (fd, now, access, modified);
            Err(crate::PyError::not_implemented(
                "utime: fd is unavailable on this platform",
            ))
        }
    }

    /// `do_utimens` (`interp_posix.py`) writes `UTIME_NOW` over both
    /// nanosecond fields when the caller named no time, rather than reading a
    /// time off its own clock and asking for that one. The two are different
    /// requests: `UTIME_NOW` on both stamps is granted to anyone the file is
    /// writable to, while naming a timestamp asks for ownership, so a writable
    /// descriptor onto someone else's file answers `utime(fd)` and refuses
    /// `utime(fd, ns=(now, now))` with EPERM.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn timespec_of(t: UTime, now: bool) -> libc::timespec {
        libc::timespec {
            tv_sec: t.sec as libc::time_t,
            tv_nsec: if now {
                libc::UTIME_NOW as _
            } else {
                t.nsec as _
            },
        }
    }

    // PyPy `interp_posix.utime` → rposix `utimensat`/`SetFileTime`.  `times` is a
    // `(atime, mtime)` pair in seconds; `ns` the same pair in integer
    // nanoseconds; the two are mutually exclusive.  Both `None` means "now".
    fn utime_impl(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        if pos.is_empty() {
            return Err(crate::PyError::type_error(
                "utime() missing required argument 'path' (pos 1)",
            ));
        }
        if pos.len() > 2 {
            return Err(crate::PyError::type_error(format!(
                "utime() takes from 1 to 2 positional arguments but {} were given",
                pos.len()
            )));
        }
        // `interp_posix.py` puts `__kwonly__` after `w_times`, so `times`
        // is the one argument here a caller may spell either way.
        crate::builtins::kwarg_reject_unknown(
            kwargs,
            &["times", "ns", "dir_fd", "follow_symlinks"],
            "utime",
        )?;
        let w_times = crate::builtins::kwarg_get(kwargs, "times");
        if w_times.is_some() && pos.len() > 1 {
            return Err(crate::PyError::type_error(
                "utime() got multiple values for argument 'times'",
            ));
        }
        // interp_posix.py `path_or_fd(allow_fd=rposix.HAVE_FUTIMENS or
        // rposix.HAVE_FUTIMES)`.
        let path = crate::gateway::fsencode_path_or_fd_w(pos[0], "utime", HAVE_FUTIMENS)?;

        let present = |v: PyObjectRef| (!unsafe { pyre_object::is_none(v) }).then_some(v);
        let times = pos.get(1).copied().or(w_times).and_then(present);
        let ns = crate::builtins::kwarg_get(kwargs, "ns").and_then(present);
        let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v)?,
            None => true,
        };
        let dir_fd = match crate::builtins::kwarg_get(kwargs, "dir_fd").and_then(present) {
            // interp_posix.py types `dir_fd` as `DirFD(...)`, whose
            // `unwrap` is `_unwrap_dirfd` (:274-278).
            Some(v) => Some(unwrap_fd(v, "integer or None")?),
            None => None,
        };

        let unpack_two =
            |obj: PyObjectRef, what: &str| -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
                if !unsafe { pyre_object::is_tuple(obj) }
                    || unsafe { pyre_object::w_tuple_len(obj) } != 2
                {
                    // `times` is the argument that also has a `None` spelling
                    // — it is the one whose default means "now" — and its
                    // message names that spelling. `ns` has no such form.
                    let shape = if what == "times" {
                        "either a tuple of two ints or None"
                    } else {
                        "a tuple of two ints"
                    };
                    return Err(crate::PyError::type_error(format!(
                        "utime: '{what}' must be {shape}"
                    )));
                }
                Ok((
                    unsafe { pyre_object::w_tuple_getitem(obj, 0) }.unwrap(),
                    unsafe { pyre_object::w_tuple_getitem(obj, 1) }.unwrap(),
                ))
            };
        /// The out-of-range answer for both spellings of a second.
        ///
        /// `_PyTime_ObjectToDenominator` refuses a value no `time_t` can hold
        /// by overflow rather than by value, and `_PyLong_AsTime_t` gives the
        /// same words for an integer too wide to be one — `(2**200, 0)` and
        /// `(1e30, 0)` answer alike.
        fn time_t_overflow() -> crate::PyError {
            crate::PyError::overflow_error("timestamp out of range for platform time_t")
        }
        // `_PyTime_ObjectToTimespec(..., _PyTime_ROUND_FLOOR)`: the seconds are
        // the floor of the value and the nanoseconds are what is left above
        // that floor, so they stay in `0..1_000_000_000` however negative the
        // time is. `utime(p, (-1.5, -2.5))` is `(-2, 500000000)` and
        // `(-3, 500000000)`, which reads back as `-1_500_000_000` and
        // `-2_500_000_000` nanoseconds.
        let time_from_secs = |v: PyObjectRef| -> Result<UTime, crate::PyError> {
            // An integer names its second exactly, so it is read as one
            // rather than through a float that would round the seconds it is
            // too wide to hold.
            if unsafe { pyre_object::is_int_or_long(v) } {
                let sec = crate::builtins::space_index_w(v).map_err(|_| time_t_overflow())?;
                return Ok(UTime { sec, nsec: 0 });
            }
            // Everything else has to name a float. What cannot is not a time
            // at all, and is refused by type: `utime(p, ('a', 'b'))` names the
            // type it was given rather than reporting a failed float parse.
            let f = crate::builtins::builtin_float(&[v]).map_err(|err| {
                if err.kind == crate::PyErrorKind::OverflowError {
                    time_t_overflow()
                } else {
                    crate::PyError::type_error(format!(
                        "argument must be int or float, not {}",
                        crate::type_methods::arg_type_name(v)
                    ))
                }
            })?;
            let secs = unsafe { pyre_object::w_float_get_value(f) };
            // A NaN has no floor to take, and it is the one non-finite value
            // answered by value rather than by range.
            if secs.is_nan() {
                return Err(crate::PyError::value_error(
                    "Invalid value NaN (not a number)",
                ));
            }
            let floor = secs.floor();
            // The floor of an infinite or too-large value is not a second any
            // clock names; `i64::MIN`/`MAX` are what an `as` cast would answer
            // for both, so the range is checked before the cast rather than
            // read back out of it.
            if !(floor >= -(2f64.powi(63)) && floor < 2f64.powi(63)) {
                return Err(time_t_overflow());
            }
            let mut sec = floor as i64;
            let mut nsec = ((secs - floor) * 1e9).floor() as i64;
            if nsec >= 1_000_000_000 {
                nsec -= 1_000_000_000;
                sec = sec.checked_add(1).ok_or_else(time_t_overflow)?;
            }
            Ok(UTime { sec, nsec })
        };
        let time_from_ns = |v: PyObjectRef| -> Result<UTime, crate::PyError> {
            // `split_py_long_to_s_and_ns` splits with `divmod` before it
            // narrows anything, so a count of nanoseconds too wide for a
            // `time_t` is only refused when the SECOND it names is — `ns=2**80`
            // is a second that fits. Dividing after the narrowing turned away
            // the whole range instead. `divmod` is also what answers for a
            // value that is not a number at all.
            let split =
                crate::builtins::builtin_divmod(&[v, pyre_object::w_int_new(1_000_000_000)])?;
            let (w_sec, w_nsec) = unsafe {
                (
                    pyre_object::w_tuple_getitem(split, 0),
                    pyre_object::w_tuple_getitem(split, 1),
                )
            };
            let (Some(w_sec), Some(w_nsec)) = (w_sec, w_nsec) else {
                return Err(crate::PyError::type_error(
                    "utime: divmod() returned a non-pair",
                ));
            };
            // Python's own `//` and `%`, so a negative count of nanoseconds
            // lands on the second below it with a positive remainder.
            //
            // Only an integer second can be out of a `time_t`'s range. A
            // quotient that is not one at all — `divmod` answers a float pair
            // for `ns=(1.5, 2.5)` — keeps the conversion's own refusal.
            let sec = crate::builtins::space_index_w(w_sec).map_err(|err| {
                if err.kind == crate::PyErrorKind::OverflowError {
                    time_t_overflow()
                } else {
                    err
                }
            })?;
            Ok(UTime {
                sec,
                nsec: crate::builtins::space_index_w(w_nsec)?,
            })
        };

        // `parse_utime_args` (`interp_posix.py`) answers a "now" flag
        // beside the pair and leaves the pair itself at zero when it is set;
        // each of the calls below is what turns that flag into its own spelling
        // of "now".
        let (now, access, modified) = match (times, ns) {
            (Some(_), Some(_)) => {
                return Err(crate::PyError::value_error(
                    "utime: you may specify either 'times' or 'ns' but not both",
                ));
            }
            (Some(t), None) => {
                let (a, m) = unpack_two(t, "times")?;
                (false, time_from_secs(a)?, time_from_secs(m)?)
            }
            (None, Some(n)) => {
                let (a, m) = unpack_two(n, "ns")?;
                (false, time_from_ns(a)?, time_from_ns(m)?)
            }
            (None, None) => (true, UTime { sec: 0, nsec: 0 }, UTime { sec: 0, nsec: 0 }),
        };

        if path.as_fd != -1 {
            // interp_posix.py:1893-1900 — both modifiers reinterpret a *name*,
            // and a descriptor is not one. 3.14, which the parity suite reads
            // as the oracle, words the first "can't specify dir_fd without
            // matching path" where `interp_posix.py:1895` says "can't specify
            // both dir_fd and fd".
            if dir_fd.is_some() {
                return Err(crate::PyError::value_error(
                    "utime: can't specify dir_fd without matching path",
                ));
            }
            if !follow_symlinks {
                return Err(crate::PyError::value_error(
                    "utime: cannot use fd and follow_symlinks together",
                ));
            }
            return utime_fd(path.as_fd, now, access, modified);
        }

        #[cfg(all(windows, feature = "host_env"))]
        {
            use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, INVALID_HANDLE_VALUE};
            use windows_sys::Win32::Storage::FileSystem::{
                CreateFileW, FILE_FLAG_BACKUP_SEMANTICS, FILE_WRITE_ATTRIBUTES, OPEN_EXISTING,
                SetFileTime,
            };
            if dir_fd.is_some() || !follow_symlinks {
                return Err(crate::PyError::not_implemented(
                    "utime: dir_fd and follow_symlinks=False are unavailable on this platform",
                ));
            }
            // `time_t_to_FILE_TIME` (`rwin32file.py`): a FILETIME
            // counts 100ns ticks from 1601-01-01, so shifting the epoch is what
            // makes a second before 1970 an ordinary positive tick count rather
            // than one with no representation. The sub-100ns of a nanosecond is
            // not a tick the filesystem holds and is floored, as the conversion
            // floors it — which is why `ns=-1` reads back as `-100`.
            //
            // The arithmetic wraps rather than checks: `r_longlong` there, and
            // the same `__int64` in the C the line is a transcription of. A
            // second this filesystem cannot hold writes the bits the
            // multiplication leaves rather than being diagnosed, and a value no
            // `time_t` can hold has already been refused by `time_from_secs`.
            const EPOCH_DIFF: i64 = 11_644_473_600;
            let to_filetime = |t: UTime| -> FILETIME {
                let ticks = t
                    .sec
                    .wrapping_add(EPOCH_DIFF)
                    .wrapping_mul(10_000_000)
                    .wrapping_add(t.nsec / 100) as u64;
                FILETIME {
                    dwLowDateTime: ticks as u32,
                    dwHighDateTime: (ticks >> 32) as u32,
                }
            };
            // `rposix.py:1568-1576` reads a clock here when the caller named no
            // time — `GetSystemTime` into both stamps — and reaches
            // `time_t_to_FILE_TIME` only for a named pair. `SetFileTime` has no
            // word for "now", which is what the `utimensat` arm below spells
            // `UTIME_NOW`; the pair arrives at zero while the flag carries the
            // meaning, so reading it here is what keeps `os.utime(path)` off
            // 1970.
            let (access, modified) = if now {
                let d = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or(std::time::Duration::ZERO);
                let t = UTime {
                    sec: d.as_secs() as i64,
                    nsec: d.subsec_nanos() as i64,
                };
                (t, t)
            } else {
                (access, modified)
            };
            let atime = to_filetime(access);
            let mtime = to_filetime(modified);
            let wide = wide_path(&path.as_bytes)?;
            // FILE_WRITE_ATTRIBUTES is the access `SetFileTime` takes;
            // FILE_FLAG_BACKUP_SEMANTICS lets the name open a directory too.
            let handle = unsafe {
                CreateFileW(
                    wide.as_ptr(),
                    FILE_WRITE_ATTRIBUTES,
                    0,
                    std::ptr::null(),
                    OPEN_EXISTING,
                    FILE_FLAG_BACKUP_SEMANTICS,
                    std::ptr::null_mut(),
                )
            };
            if handle == INVALID_HANDLE_VALUE {
                return Err(fs_err_with_filename(
                    std::io::Error::last_os_error(),
                    path.w_path(),
                ));
            }
            let wrote = unsafe { SetFileTime(handle, std::ptr::null(), &atime, &mtime) };
            let error = (wrote == 0).then(std::io::Error::last_os_error);
            unsafe { CloseHandle(handle) };
            if let Some(error) = error {
                return Err(fs_err_with_filename(error, path.w_path()));
            }
            return Ok(pyre_object::w_none());
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null character"))?;
            // `rposix.utimensat` (`rposix.py`) — the whole name form
            // is this one call: the descriptor the name resolves against is
            // `AT_FDCWD` when the caller named none, and `follow_symlinks=False`
            // is `AT_SYMLINK_NOFOLLOW`.
            let flag = if follow_symlinks {
                0
            } else {
                libc::AT_SYMLINK_NOFOLLOW
            };
            let times = [timespec_of(access, now), timespec_of(modified, now)];
            let error = unsafe {
                libc::utimensat(
                    dir_fd.unwrap_or(libc::AT_FDCWD),
                    c_path.as_ptr(),
                    times.as_ptr(),
                    flag,
                )
            };
            if error < 0 {
                return Err(io_err_with_filename(
                    std::io::Error::last_os_error(),
                    path.w_path(),
                ));
            }
            return Ok(pyre_object::w_none());
        }
        #[allow(unreachable_code)]
        {
            let _ = (now, access, modified, dir_fd, follow_symlinks, &path, pos);
            Err(crate::PyError::not_implemented(
                "utime is unavailable on this platform",
            ))
        }
    }
    crate::module_ns_store(
        ns,
        "utime",
        crate::make_builtin_function("utime", utime_impl),
    );

    // ── posix._path_splitroot(path) → (root, tail) ──
    // Registered only where `sys.platform` is `win32`, the same condition
    // `_bootstrap_external` gates its use on.
    #[cfg(windows)]
    crate::module_ns_store(
        ns,
        "_path_splitroot",
        crate::make_builtin_function_with_arity(
            "_path_splitroot",
            |args| {
                let Some(&arg) = args.first() else {
                    return Err(crate::PyError::type_error(
                        "_path_splitroot() missing required argument 'path'",
                    ));
                };
                // Windows-only, so what it should name itself with is not
                // measurable from a POSIX host; it keeps the caller-less
                // conversion meanwhile. See the follow-up task.
                let path = extract_path(arg)?;
                // Splitting a drive or UNC prefix is a text operation on a
                // Windows path, and both halves are handed back as `str`, so
                // this one stays in the text domain rather than the byte one.
                let path = crate::gateway::fsdecode_filename_wtf8(&path);
                let (root, tail) = split_root(&path);
                Ok(pyre_object::w_tuple_new(vec![
                    pyre_object::w_str_from_wtf8(root.to_wtf8_buf()),
                    pyre_object::w_str_from_wtf8(tail.to_wtf8_buf()),
                ]))
            },
            1,
        ),
    );

    // ── Windows-only nt calls — moduledef.py `if os.name == 'nt'` block ──
    // ntpath imports these from `nt` behind try/except ImportError, so a build
    // that omits one silently falls back to its pure-Python path implementation.
    #[cfg(all(windows, feature = "host_env"))]
    {
        for (name, func, arity) in [
            (
                "_getfullpathname",
                win_nt::_getfullpathname as crate::gateway::BuiltinCodeFn,
                1u16,
            ),
            ("_getfinalpathname", win_nt::_getfinalpathname, 1),
            ("_findfirstfile", win_nt::_findfirstfile, 1),
            ("_getfileinformation", win_nt::_getfileinformation, 1),
            ("_getdiskusage", win_nt::_getdiskusage, 1),
            ("get_handle_inheritable", win_nt::get_handle_inheritable, 1),
            ("set_handle_inheritable", win_nt::set_handle_inheritable, 2),
            ("_add_dll_directory", win_nt::_add_dll_directory, 1),
            ("_remove_dll_directory", win_nt::_remove_dll_directory, 1),
        ] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, func, arity),
            );
        }
        crate::module_ns_store(
            ns,
            "_supports_virtual_terminal",
            crate::make_builtin_function(
                "_supports_virtual_terminal",
                win_nt::_supports_virtual_terminal,
            ),
        );
    }

    /// Drive an open `DIR*` to its end, handing each real entry (`.` and `..`
    /// left out) to `f` as `(name, d_ino, d_type)` — the `get_name_bytes`,
    /// `get_inode`, and `get_known_type` a `nextentry` yields
    /// (`interp_scandir.py:148-153`).  Returns the errno at the end: `0` for a
    /// clean end, or the failure `readdir` reported.  Does not close `dirp`.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn readdir_collect(dirp: *mut libc::DIR, mut f: impl FnMut(&[u8], i64, u8)) -> i32 {
        loop {
            // `readdir` reports the end of the directory and a failure the
            // same way — a null return — so errno is cleared before the call
            // and read back after it (`rposix.py` RFFI_FULL_ERRNO_ZERO).
            rustpython_host_env::os::set_errno(0);
            let entry = unsafe { libc::readdir(dirp) };
            if entry.is_null() {
                return crate::builtins::crt_errno();
            }
            let name = unsafe { std::ffi::CStr::from_ptr((*entry).d_name.as_ptr()) };
            let name = name.to_bytes();
            if name != b"." && name != b".." {
                let ino = unsafe { (*entry).d_ino } as i64;
                let d_type = unsafe { (*entry).d_type };
                f(name, ino, d_type);
            }
        }
    }

    /// Read a directory descriptor's entries through `f`
    /// (`rposix.py` `_listdir`/`fdlistdir`).
    ///
    /// `fdopendir` takes the descriptor over and `closedir` closes it, so the
    /// caller's own is duplicated first — `interp_posix.py:1118` spells that
    /// `rposix.dup(fd, inheritable=False)`, which is `F_DUPFD_CLOEXEC`.  The
    /// duplicate shares its file description — and so its directory offset —
    /// with the caller's descriptor, which would be left at the end of the
    /// directory and read as empty next time; `_listdir`'s `rewind=True`
    /// (`rposix.py`) puts it back before the close.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn fd_readdir(fd: i32, f: impl FnMut(&[u8], i64, u8)) -> Result<(), i32> {
        let dup = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
        if dup < 0 {
            return Err(crate::builtins::crt_errno());
        }
        let dirp = unsafe { libc::fdopendir(dup) };
        if dirp.is_null() {
            let errno = crate::builtins::crt_errno();
            unsafe { libc::close(dup) };
            return Err(errno);
        }
        let errno = readdir_collect(dirp, f);
        unsafe { libc::rewinddir(dirp) };
        // `closedir` closes the duplicate, so nothing here outlives the call.
        unsafe { libc::closedir(dirp) };
        if errno != 0 {
            return Err(errno);
        }
        Ok(())
    }

    /// The names a directory descriptor holds, `.` and `..` left out.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn fdlistdir(fd: i32) -> Result<Vec<Vec<u8>>, i32> {
        let mut names = Vec::new();
        fd_readdir(fd, |name, _ino, _d_type| names.push(name.to_vec()))?;
        Ok(names)
    }

    // ── posix.listdir(path=".") → list of str ──
    crate::module_ns_store(
        ns,
        "listdir",
        crate::make_builtin_function("listdir", |args| {
            let (bound, _kwargs) = bind_path_args(args, "listdir", &["path"], 0, &[])?;
            // One resolution yields both the path and its bytes-ness, so
            // `__fspath__` runs exactly once. The omitted argument is the same
            // `None` the signature names, which resolves to `"."` there.
            let arg = bound[0].unwrap_or(pyre_object::w_none());
            let resolved =
                crate::gateway::fsencode_path_or_fd_nullable_w(arg, "listdir", HAVE_FDOPENDIR)?;
            // A descriptor names no directory to prefix and is not `bytes`, so
            // its names come back as `str` whatever the caller held
            // (`interp_posix.py:1112-1121`).
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            if resolved.as_fd != -1 {
                // The descriptor is what named the directory, so it is what
                // names the failure.
                let names = fdlistdir(resolved.as_fd)
                    .map_err(|errno| errno_err_with_filename(errno, resolved.w_path()))?;
                let items = names.iter().map(|n| fs_name_obj(false, n)).collect();
                return Ok(pyre_object::w_list_new(items));
            }
            let bytes_mode = unsafe { resolved.is_bytes() };
            let path = resolved.as_bytes.as_slice();
            let w_path = || resolved.w_path();
            #[cfg(feature = "sandbox")]
            {
                let names = crate::host_seam::ops::listdir(path)
                    .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, w_path()))?;
                let items = names
                    .into_iter()
                    .map(|n| fs_name_obj(bytes_mode, &n))
                    .collect();
                return Ok(pyre_object::w_list_new(items));
            }
            #[cfg(not(feature = "sandbox"))]
            {
                let entries = host_fs::read_dir(path_from_bytes(path).as_ref())
                    .map_err(|e| fs_err_with_filename(e, w_path()))?;
                let mut items = Vec::new();
                for entry in entries {
                    let entry = entry.map_err(|e| fs_err_with_filename(e, w_path()))?;
                    let name = entry.file_name();
                    items.push(fs_name_obj(bytes_mode, name.as_encoded_bytes()));
                }
                Ok(pyre_object::w_list_new(items))
            }
        }),
    );

    // ── posix.isatty(fd) → bool ──
    crate::module_ns_store(
        ns,
        "isatty",
        crate::make_builtin_function_with_arity(
            "isatty",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_bool_from(false));
                }
                let fd = crate::baseobjspace::c_int_w(args[0])?;
                #[cfg(feature = "sandbox")]
                {
                    return Ok(pyre_object::w_bool_from(
                        crate::host_seam::ops::isatty(fd).unwrap_or(false),
                    ));
                }
                #[cfg(not(feature = "sandbox"))]
                Ok(pyre_object::w_bool_from(host_os::isatty(fd)))
            },
            1,
        ),
    );

    // ── posix.urandom(n) → bytes ──
    crate::module_ns_store(
        ns,
        "urandom",
        crate::make_builtin_function_with_arity(
            "urandom",
            |args| {
                crate::gateway::check_declared_arity("urandom", 1, args.len())?;
                // `__index__` conversion, so a non-integer is a TypeError
                // instead of a raw field read that asks for an arbitrary
                // number of bytes.
                let n = crate::builtins::space_index_w(args[0])?;
                if n < 0 {
                    return Err(crate::PyError::value_error("negative argument not allowed"));
                }
                // A 32-bit target's `usize` is narrower than the `__index__`
                // result, so a size it cannot hold is reported rather than
                // wrapped to a smaller request.
                let n = usize::try_from(n)
                    .map_err(|_| crate::PyError::overflow_error("argument out of range"))?;
                #[cfg(not(feature = "sandbox"))]
                // Report entropy failures: absorbing one would return predictable bytes.
                let buf = host_os::urandom(n).map_err(|e| io_err(e, ""))?;
                // Route host entropy through the trusted controller instead of
                // reaching host getrandom directly.
                #[cfg(feature = "sandbox")]
                let buf = crate::host_seam::ops::urandom(n as i64)
                    .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                Ok(pyre_object::w_bytes_from_bytes(&buf))
            },
            1,
        ),
    );
    // os.terminal_size — structseq (columns, lines).
    fn make_terminal_size(cols: i64, lines: i64) -> pyre_object::PyObjectRef {
        crate::_structseq::new_instance(
            terminal_size_seq_type(),
            vec![pyre_object::w_int_new(cols), pyre_object::w_int_new(lines)],
        )
    }
    crate::module_ns_store(ns, "terminal_size", terminal_size_seq_type());
    crate::module_ns_store(ns, "statvfs_result", statvfs_result_seq_type());
    crate::module_ns_store(ns, "times_result", times_result_seq_type());
    // `uname_result` names `posix` as its module, which is what pickle imports
    // to resolve it; only the POSIX hosts have that module, and only they
    // register `uname` below.
    #[cfg(unix)]
    crate::module_ns_store(ns, "uname_result", uname_result_seq_type());

    // ── posix.get_terminal_size(fd=1) → os.terminal_size(columns, lines) ──
    // The size belongs to the terminal the descriptor names, read through
    // `ioctl(TIOCGWINSZ)` or `GetConsoleScreenBufferInfo`.  A descriptor that
    // names no terminal has no size, which is reported rather than answered
    // with a guess — `shutil.get_terminal_size` is where the guess lives, and
    // it is reached by catching this.  Stubbed under sandbox, so the real body
    // is compiled out.
    #[cfg(not(feature = "sandbox"))]
    crate::module_ns_store(
        ns,
        "get_terminal_size",
        crate::make_builtin_function("get_terminal_size", |args| {
            // `($module, fd=<unrepresentable>, /)` — the descriptor is
            // positional-only, so `fd=1` is a keyword this entry point does
            // not take rather than a binding.
            // A keyword is refused against the module-qualified name, and this
            // module answers to `nt` on Windows.
            let qualname = if cfg!(windows) {
                "nt.get_terminal_size"
            } else {
                "posix.get_terminal_size"
            };
            let (bound, _kwargs) =
                bind_posonly_args(args, "get_terminal_size", qualname, 1, 0, &[])?;
            let fd = match bound[0] {
                Some(w) => crate::baseobjspace::c_int_w(w)?,
                None => 1,
            };
            #[cfg(unix)]
            {
                let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                if crate::builtins::crt_call!(libc::ioctl(fd, libc::TIOCGWINSZ, &mut ws)) != 0 {
                    return Err(errno_err(crate::builtins::crt_errno(), ""));
                }
                Ok(make_terminal_size(ws.ws_col as i64, ws.ws_row as i64))
            }
            #[cfg(all(windows, feature = "host_env"))]
            {
                let handle = rustpython_host_env::nt::handle_from_fd(fd);
                let (columns, lines) = rustpython_host_env::nt::get_terminal_size_handle(handle)
                    .map_err(|e| fs_err_with_filename(e, pyre_object::PY_NULL))?;
                Ok(make_terminal_size(columns as i64, lines as i64))
            }
            // A target with neither call has no terminal to measure.
            #[cfg(not(any(unix, all(windows, feature = "host_env"))))]
            {
                let _ = fd;
                Ok(make_terminal_size(80, 24))
            }
        }),
    );
    // os.fspath() — posixmodule.c posix_fspath / PyOS_FSPath.
    crate::module_ns_store(
        ns,
        "fspath",
        crate::make_builtin_function_with_arity(
            "fspath",
            |args| fspath(args.first().copied().unwrap_or(pyre_object::w_none())),
            1,
        ),
    );
    // os.stat / os.lstat / os.fstat — return stat_result structseq.
    // PyPy: posixmodule.c posix_do_stat → build_stat_result.
    //
    // The returned object is a tuple subclass with named attributes
    // (st_mode, st_ino, ...). We expose it as a plain instance with
    // attributes so that both `os.stat(p).st_mode` and
    // `os.stat(p)[0]` work.
    // `st_flags` lives in the BSD/macOS `struct stat` but `std`'s
    // `Metadata`/`MetadataExt` does not surface it, so read it with a raw
    // `stat`/`lstat`/`fstat`; on failure default to 0 (the primary
    // metadata read already succeeded).
    // Under sandbox the stat path is mediated (st_flags arrives over the wire),
    // so this raw-libc helper is compiled out.
    #[cfg(all(target_os = "macos", not(feature = "sandbox")))]
    fn macos_path_st_flags(path: &[u8], follow: bool) -> u32 {
        let Ok(c) = std::ffi::CString::new(path) else {
            return 0;
        };
        unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            let rc = if follow {
                libc::stat(c.as_ptr(), &mut st)
            } else {
                libc::lstat(c.as_ptr(), &mut st)
            };
            if rc == 0 { st.st_flags } else { 0 }
        }
    }
    #[cfg(all(target_os = "macos", not(feature = "sandbox")))]
    fn macos_fd_st_flags(fd: i32) -> u32 {
        unsafe {
            let mut st: libc::stat = std::mem::zeroed();
            if libc::fstat(fd, &mut st) == 0 {
                st.st_flags
            } else {
                0
            }
        }
    }

    /// `st_flags` (macOS/BSD) is not surfaced by `std::fs::Metadata`, so
    /// the caller obtains it via a raw `stat`/`lstat`/`fstat` and passes
    /// it in; it is ignored (and unread) on platforms whose `struct stat`
    /// lacks the field.
    fn make_stat_result(meta: &std::fs::Metadata, st_flags: u32) -> pyre_object::PyObjectRef {
        // Extract stat fields in a cross-platform way.
        #[cfg(unix)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.mode() as i64,
                meta.ino() as i64,
                meta.dev() as i64,
                meta.nlink() as i64,
                meta.uid() as i64,
                meta.gid() as i64,
                meta.size() as i64,
                meta.atime(),
                meta.mtime(),
                meta.ctime(),
                whole_ns(meta.atime(), meta.atime_nsec()),
                whole_ns(meta.mtime(), meta.mtime_nsec()),
                whole_ns(meta.ctime(), meta.ctime_nsec()),
            )
        };
        #[cfg(windows)]
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = {
            use std::os::windows::fs::MetadataExt;
            let ft = meta.file_type();
            let attrs = meta.file_attributes();
            // `attributes_to_mode`: a directory carries the execute bits and a
            // read-only file drops the write ones, both read off the
            // attributes.  `_Py_attribute_data_to_stat` then replaces only the
            // format bits for a symlink, so a link to a directory keeps the
            // execute bits its own attributes carry.
            const FILE_ATTRIBUTE_READONLY: u32 = 0x1;
            const FILE_ATTRIBUTE_DIRECTORY: u32 = 0x10;
            let permissions = if attrs & FILE_ATTRIBUTE_READONLY != 0 {
                0o444
            } else {
                0o666
            };
            let format = if ft.is_symlink() {
                0o120000
            } else if ft.is_dir() {
                0o40000
            } else {
                0o100000
            };
            let executable = if attrs & FILE_ATTRIBUTE_DIRECTORY != 0 {
                0o111
            } else {
                0
            };
            let mode: i64 = format | executable | permissions;
            let size = meta.file_size() as i64;
            // Windows FILETIME is 100-ns intervals since 1601-01-01.
            // Convert to Unix epoch seconds.
            const EPOCH_DIFF: i64 = 11_644_473_600;
            let atime_secs = (meta.last_access_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let mtime_secs = (meta.last_write_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let ctime_secs = (meta.creation_time() as i64 / 10_000_000) - EPOCH_DIFF;
            let atime_ns = whole_ns(
                atime_secs,
                (meta.last_access_time() as i64 % 10_000_000) * 100,
            );
            let mtime_ns = whole_ns(
                mtime_secs,
                (meta.last_write_time() as i64 % 10_000_000) * 100,
            );
            let ctime_ns = whole_ns(ctime_secs, (meta.creation_time() as i64 % 10_000_000) * 100);
            (
                mode, 0i64, // st_ino — not available on Windows
                0i64, // st_dev
                1i64, // nlink — not easily available on stable Windows
                0i64, // st_uid
                0i64, // st_gid
                size, atime_secs, mtime_secs, ctime_secs, atime_ns, mtime_ns, ctime_ns,
            )
        };

        #[cfg(unix)]
        let (st_blksize, st_blocks, st_rdev) = {
            use std::os::unix::fs::MetadataExt;
            (
                meta.blksize() as i64,
                meta.blocks() as i64,
                meta.rdev() as i64,
            )
        };

        stat_result_from_fields(
            &StatFields {
                mode: st_mode,
                ino: st_ino,
                dev: st_dev,
                nlink: st_nlink,
                uid: st_uid,
                gid: st_gid,
                size: st_size,
                atime: st_atime,
                mtime: st_mtime,
                ctime: st_ctime,
                atime_ns: st_atime_ns,
                mtime_ns: st_mtime_ns,
                ctime_ns: st_ctime_ns,
                #[cfg(unix)]
                blksize: st_blksize,
                #[cfg(unix)]
                blocks: st_blocks,
                #[cfg(unix)]
                rdev: st_rdev,
            },
            st_flags,
        )
    }

    /// The `stat_result` fields, read out of whichever source produced them:
    /// `std::fs::Metadata` for the path and descriptor forms, `libc::stat`
    /// for the `fstatat` form a `dir_fd`-relative name takes.
    /// A whole timestamp in nanoseconds.
    ///
    /// `i64` nanoseconds run out in 2262, and a file can carry a later time
    /// than that — every Windows FILETIME up to the year 30828 is one. The
    /// product is taken wide so `st_mtime_ns` is the number it is rather than
    /// the wrap `sec * 1_000_000_000` would answer with, and
    /// [`w_time_ns`] hands back an int of whatever width it needs.
    fn whole_ns(sec: i64, nsec: i64) -> i128 {
        sec as i128 * 1_000_000_000 + nsec as i128
    }

    /// The `st_*_ns` field as a Python int, which has no width to run out of.
    fn w_time_ns(ns: i128) -> pyre_object::PyObjectRef {
        match i64::try_from(ns) {
            Ok(n) => pyre_object::w_int_new(n),
            Err(_) => pyre_object::longobject::w_long_new(majit_rlib::rbigint::RBigInt::from(ns)),
        }
    }

    struct StatFields {
        mode: i64,
        ino: i64,
        dev: i64,
        nlink: i64,
        uid: i64,
        gid: i64,
        size: i64,
        atime: i64,
        mtime: i64,
        ctime: i64,
        atime_ns: i128,
        mtime_ns: i128,
        ctime_ns: i128,
        #[cfg(unix)]
        blksize: i64,
        #[cfg(unix)]
        blocks: i64,
        #[cfg(unix)]
        rdev: i64,
    }

    fn stat_result_from_fields(f: &StatFields, st_flags: u32) -> pyre_object::PyObjectRef {
        let (
            st_mode,
            st_ino,
            st_dev,
            st_nlink,
            st_uid,
            st_gid,
            st_size,
            st_atime,
            st_mtime,
            st_ctime,
            st_atime_ns,
            st_mtime_ns,
            st_ctime_ns,
        ) = (
            f.mode, f.ino, f.dev, f.nlink, f.uid, f.gid, f.size, f.atime, f.mtime, f.ctime,
            f.atime_ns, f.mtime_ns, f.ctime_ns,
        );
        #[cfg(unix)]
        let (st_blksize, st_blocks, st_rdev) = (f.blksize, f.blocks, f.rdev);
        // The 10 sequence slots are the integer fields (integer-seconds
        // times at 7..10, named `_integer_*`); the float times, `st_*_ns`,
        // and the platform block/device extras are named-only fields.
        let seq = vec![
            pyre_object::w_int_new(st_mode),
            pyre_object::w_int_new(st_ino),
            pyre_object::w_int_new(st_dev),
            pyre_object::w_int_new(st_nlink),
            pyre_object::w_int_new(st_uid),
            pyre_object::w_int_new(st_gid),
            pyre_object::w_int_new(st_size),
            pyre_object::w_int_new(st_atime),
            pyre_object::w_int_new(st_mtime),
            pyre_object::w_int_new(st_ctime),
        ];
        // `_ll_get_st_atime` — float times keep sub-second precision:
        // `float(seconds) + 1e-9 * nanosecond_fraction`, where the
        // fraction is recovered from the full-nanosecond field.
        let st_atime_f = st_atime as f64 + 1e-9 * (st_atime_ns - whole_ns(st_atime, 0)) as f64;
        let st_mtime_f = st_mtime as f64 + 1e-9 * (st_mtime_ns - whole_ns(st_mtime, 0)) as f64;
        let st_ctime_f = st_ctime as f64 + 1e-9 * (st_ctime_ns - whole_ns(st_ctime, 0)) as f64;
        #[allow(unused_mut)]
        let mut extras = vec![
            ("st_atime", pyre_object::w_float_new(st_atime_f)),
            ("st_mtime", pyre_object::w_float_new(st_mtime_f)),
            ("st_ctime", pyre_object::w_float_new(st_ctime_f)),
            ("st_atime_ns", w_time_ns(st_atime_ns)),
            ("st_mtime_ns", w_time_ns(st_mtime_ns)),
            ("st_ctime_ns", w_time_ns(st_ctime_ns)),
            // `build_stat_result` (interp_posix.py): the
            // sub-second remainder of each full-nanosecond timestamp,
            // `value % 1_000_000_000` (non-negative for pre-1970 times).
            (
                "nsec_atime",
                pyre_object::w_int_new(st_atime_ns.rem_euclid(1_000_000_000) as i64),
            ),
            (
                "nsec_mtime",
                pyre_object::w_int_new(st_mtime_ns.rem_euclid(1_000_000_000) as i64),
            ),
            (
                "nsec_ctime",
                pyre_object::w_int_new(st_ctime_ns.rem_euclid(1_000_000_000) as i64),
            ),
        ];
        #[cfg(unix)]
        {
            extras.push(("st_blksize", pyre_object::w_int_new(st_blksize)));
            extras.push(("st_blocks", pyre_object::w_int_new(st_blocks)));
            extras.push(("st_rdev", pyre_object::w_int_new(st_rdev)));
        }
        #[cfg(target_os = "macos")]
        extras.push(("st_flags", pyre_object::w_int_new(st_flags as i64)));
        #[cfg(not(target_os = "macos"))]
        let _ = st_flags;
        crate::_structseq::new_instance_with_extra(stat_result_seq_type(), seq, extras)
    }
    /// Build a `stat_result` from the sandbox wire `StatBuf` (sandbox build
    /// only, hence unix-only): the controller delivers the 10 protocol fields
    /// plus integer atime/mtime/ctime; the sub-second and block/device extras
    /// are whatever `StatBuf` carries (zero over the wire). Mirrors the unix
    /// slot/extra layout of `make_stat_result`.
    #[cfg(feature = "sandbox")]
    fn make_stat_result_from_statbuf(st: &crate::host_seam::StatBuf) -> pyre_object::PyObjectRef {
        let st_atime = st.atime;
        let st_mtime = st.mtime;
        let st_ctime = st.ctime;
        let st_atime_ns = whole_ns(st.atime, st.atime_nsec);
        let st_mtime_ns = whole_ns(st.mtime, st.mtime_nsec);
        let st_ctime_ns = whole_ns(st.ctime, st.ctime_nsec);
        let seq = vec![
            pyre_object::w_int_new(st.mode as i64),
            pyre_object::w_int_new(st.ino as i64),
            pyre_object::w_int_new(st.dev as i64),
            pyre_object::w_int_new(st.nlink as i64),
            pyre_object::w_int_new(st.uid as i64),
            pyre_object::w_int_new(st.gid as i64),
            pyre_object::w_int_new(st.size as i64),
            pyre_object::w_int_new(st_atime),
            pyre_object::w_int_new(st_mtime),
            pyre_object::w_int_new(st_ctime),
        ];
        let st_atime_f = st_atime as f64 + 1e-9 * (st_atime_ns - whole_ns(st_atime, 0)) as f64;
        let st_mtime_f = st_mtime as f64 + 1e-9 * (st_mtime_ns - whole_ns(st_mtime, 0)) as f64;
        let st_ctime_f = st_ctime as f64 + 1e-9 * (st_ctime_ns - whole_ns(st_ctime, 0)) as f64;
        #[allow(unused_mut)]
        let mut extras = vec![
            ("st_atime", pyre_object::w_float_new(st_atime_f)),
            ("st_mtime", pyre_object::w_float_new(st_mtime_f)),
            ("st_ctime", pyre_object::w_float_new(st_ctime_f)),
            ("st_atime_ns", w_time_ns(st_atime_ns)),
            ("st_mtime_ns", w_time_ns(st_mtime_ns)),
            ("st_ctime_ns", w_time_ns(st_ctime_ns)),
            (
                "nsec_atime",
                pyre_object::w_int_new(st_atime_ns.rem_euclid(1_000_000_000) as i64),
            ),
            (
                "nsec_mtime",
                pyre_object::w_int_new(st_mtime_ns.rem_euclid(1_000_000_000) as i64),
            ),
            (
                "nsec_ctime",
                pyre_object::w_int_new(st_ctime_ns.rem_euclid(1_000_000_000) as i64),
            ),
            ("st_blksize", pyre_object::w_int_new(st.blksize as i64)),
            ("st_blocks", pyre_object::w_int_new(st.blocks as i64)),
            ("st_rdev", pyre_object::w_int_new(st.rdev as i64)),
        ];
        #[cfg(target_os = "macos")]
        extras.push(("st_flags", pyre_object::w_int_new(st.st_flags as i64)));
        crate::_structseq::new_instance_with_extra(stat_result_seq_type(), seq, extras)
    }
    /// `os.stat(path, *, dir_fd=None, follow_symlinks=True)` /
    /// `os.lstat(path, *, dir_fd=None)` — `follow_symlinks` is keyword-only,
    /// so `stat` cannot take the fixed-arity carrier that rejects keywords.
    /// The three argument forms are the three arms of `do_stat`
    /// (`interp_posix.py`): an open descriptor as `path` goes to
    /// `fstat`, a `dir_fd`-relative name to `fstatat`, and a bare name to
    /// `stat`/`lstat`.
    fn stat_entry(
        args: &[pyre_object::PyObjectRef],
        default_follow: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        let name = if default_follow { "stat" } else { "lstat" };
        let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        let allowed: &[&str] = if default_follow {
            &["path", "dir_fd", "follow_symlinks"]
        } else {
            &["path", "dir_fd"]
        };
        crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
        if pos.len() > 1 {
            return Err(crate::PyError::type_error(format!(
                "{name}() takes at most 1 positional argument ({} given)",
                pos.len()
            )));
        }
        let path = match crate::builtins::bind_pos_or_kw(pos, kwargs, 0, "path", name, 1)? {
            Some(path) => path,
            None => {
                return Err(crate::PyError::type_error(format!(
                    "{name}() missing required argument 'path' (pos 1)"
                )));
            }
        };
        // The three arguments are unwrapped in signature order — `path`,
        // `dir_fd`, `follow_symlinks` (`interp_posix.py:610-614`) — because
        // `gateway.py` applies the unwrap specs in that order and each can
        // both raise and run user code: `__fspath__` for `path`, `__index__`
        // for `dir_fd`, `__bool__` for `follow_symlinks`.
        //
        // interp_posix.py — `stat` takes `path_or_fd(allow_fd=True)`
        // and `lstat` takes `allow_fd=False`, which is also what makes their
        // type errors name different allowed types.
        let path = crate::gateway::fsencode_path_or_fd_w(path, name, default_follow)?;
        // `stat`/`lstat` type `dir_fd` as `DirFD(rposix.HAVE_FSTATAT)`
        // (`interp_posix.py`), whose `unwrap` is `_unwrap_dirfd`
        // (:274-278).
        let dir_fd = match crate::builtins::kwarg_get(kwargs, "dir_fd")
            .filter(|&v| !unsafe { pyre_object::is_none(v) })
        {
            Some(v) => {
                let fd = unwrap_fd(v, "integer or None")?;
                // `DirFD(available=False)` is `_DirFD_Unavailable`
                // (:285-292), which turns a non-default `dir_fd` away while
                // unwrapping — so where the platform has no `fstatat` the
                // answer is this, not the descriptor conflict `do_stat`
                // would reach first.
                if !HAVE_FSTATAT {
                    return Err(dir_fd_unavailable());
                }
                Some(fd)
            }
            None => None,
        };
        let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v)?,
            None => default_follow,
        };
        // interp_posix.py `do_stat` tests the descriptor first: with one
        // in hand neither other argument has anything to apply to. Only the
        // `follow_symlinks` rejection precedes the platform's dir_fd
        // availability, though — `_DirFD_Unavailable` (`interp_posix.py`,
        // the `!HAVE_FSTATAT` arm above) turns the argument away while
        // unwrapping it, a step earlier than this, so where `fstatat` does not
        // exist a descriptor passed with `dir_fd` reports the platform rather
        // than the conflict.
        if path.as_fd != -1 {
            if dir_fd.is_some() {
                // 3.14 words this "can't specify dir_fd without matching
                // path"; `interp_posix.py:639` says "can't specify both
                // dir_fd and fd". The parity suite's oracle is CPython.
                return Err(crate::PyError::value_error(format!(
                    "{name}: can't specify dir_fd without matching path"
                )));
            }
            if !follow_symlinks {
                return Err(crate::PyError::value_error(format!(
                    "{name}: cannot use fd and follow_symlinks together"
                )));
            }
            return fstat_fd(path.as_fd);
        }
        match dir_fd {
            Some(dir_fd) => stat_at(name, &path, dir_fd, follow_symlinks),
            None => stat_path(&path, follow_symlinks),
        }
    }

    /// `rposix_stat.build_stat_result` reads the same fields off the raw
    /// `struct stat` the `*at` calls fill in.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn stat_fields_from_libc(st: &libc::stat) -> StatFields {
        StatFields {
            mode: st.st_mode as i64,
            ino: st.st_ino as i64,
            dev: st.st_dev as i64,
            nlink: st.st_nlink as i64,
            uid: st.st_uid as i64,
            gid: st.st_gid as i64,
            size: st.st_size,
            atime: st.st_atime,
            mtime: st.st_mtime,
            ctime: st.st_ctime,
            atime_ns: whole_ns(st.st_atime, st.st_atime_nsec),
            mtime_ns: whole_ns(st.st_mtime, st.st_mtime_nsec),
            ctime_ns: whole_ns(st.st_ctime, st.st_ctime_nsec),
            blksize: st.st_blksize as i64,
            blocks: st.st_blocks,
            rdev: st.st_rdev as i64,
        }
    }

    /// The Windows counterpart of `stat_fields_from_libc`.  `win32_xstat` and
    /// `fstat` fill the same `StatStruct`, so routing both entry points
    /// through it is what lets a name and a descriptor for one file report one
    /// identity — `st_ino`/`st_dev` were previously reported as 0 from a path,
    /// which made every file look like every other one.
    ///
    /// `st_ctime` carries the creation time, which is the value the field has
    /// always held here; `StatStruct` keeps that under `st_birthtime` and
    /// leaves its own `st_ctime` at zero.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn stat_fields_from_statstruct(st: &rustpython_host_env::fileutils::StatStruct) -> StatFields {
        StatFields {
            mode: st.st_mode as i64,
            ino: st.st_ino as i64,
            dev: st.st_dev as i64,
            nlink: st.st_nlink as i64,
            uid: st.st_uid as i64,
            gid: st.st_gid as i64,
            size: st.st_size as i64,
            atime: st.st_atime as i64,
            mtime: st.st_mtime as i64,
            ctime: st.st_birthtime as i64,
            atime_ns: whole_ns(st.st_atime as i64, st.st_atime_nsec as i64),
            mtime_ns: whole_ns(st.st_mtime as i64, st.st_mtime_nsec as i64),
            ctime_ns: whole_ns(st.st_birthtime as i64, st.st_birthtime_nsec as i64),
        }
    }

    /// `win32_xstat` for a byte path.  `os.stat`, `DirEntry.stat` and
    /// `DirEntry.inode` all go through here so that one file has one identity
    /// however it was reached; reaching only some of them would leave
    /// `entry.inode()` and `os.stat(entry.path).st_ino` disagreeing.
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    fn win_stat_fields(path: &[u8], follow_symlinks: bool) -> std::io::Result<StatFields> {
        let wide = widestring::WideCString::from_os_str(&*os_str_from_bytes(path)).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "embedded null character in path",
            )
        })?;
        rustpython_host_env::nt::win32_xstat(&wide, follow_symlinks)
            .map(|st| stat_fields_from_statstruct(&st))
    }

    /// Where the `host_env::posix`-backed implementations below are compiled.
    /// Elsewhere those names are the noop placeholders registered near the top
    /// of this function, which cannot serve a descriptor.
    const HOST_POSIX: bool = cfg!(all(unix, feature = "host_env", not(feature = "sandbox")));
    /// The same, for the calls Windows serves through the C runtime.
    const HOST_WINDOWS_CRT: bool =
        cfg!(all(windows, feature = "host_env", not(feature = "sandbox")));

    /// The `HAVE_*` macros `_have_functions` advertises, each spelled as the
    /// condition under which the entry point it names really does take an open
    /// descriptor. `os.py:140-155` reads them into `supports_fd`, so a bit set
    /// where the call still rejects an integer hands the caller a capability it
    /// cannot use.
    /// `rposix.HAVE_FACCESSAT` — what `access` types its `dir_fd` as
    /// (`interp_posix.py:745`) and what its two flag modifiers are tested
    /// against (`:771-775`). All three of `access`'s modifiers are the one
    /// `faccessat` call, so the same bit carries them: `os.py:117,137,158` read
    /// it into `supports_dir_fd`, `supports_effective_ids` and
    /// `supports_follow_symlinks` alike, and it is the only bit any of those
    /// three reads for `access`.
    const HAVE_FACCESSAT: bool = HOST_POSIX;
    const HAVE_FCHDIR: bool = HOST_POSIX;
    const HAVE_FCHMOD: bool = HOST_POSIX;
    const HAVE_FCHOWN: bool = HOST_POSIX;
    /// `chown` resolves a name against `dir_fd` and honours
    /// `follow_symlinks=False` through the one `fchownat` call
    /// (`interp_posix.py`), which is also how `lchown` is spelled.
    const HAVE_FCHOWNAT: bool = HOST_POSIX;
    const HAVE_FPATHCONF: bool = HOST_POSIX;
    const HAVE_FSTATVFS: bool = HOST_POSIX;
    /// Windows serves this one too — `_chsize_s` behind `os.ftruncate` — so
    /// `os.truncate` belongs in `supports_fd` on both.
    const HAVE_FTRUNCATE: bool = HOST_POSIX || HOST_WINDOWS_CRT;
    /// `utime` reaches `futimens` and `utimensat` through `libc` rather than
    /// `host_env`, so it needs one less condition than the rest.
    const HAVE_FUTIMENS: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// The name form of `utime` is one `utimensat`, so the same bit carries
    /// both of its modifiers: `dir_fd` is the descriptor the name resolves
    /// against and `follow_symlinks=False` is `AT_SYMLINK_NOFOLLOW`.
    const HAVE_UTIMENSAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_FSTATAT` — what `DirFD` is parameterised on
    /// (`interp_posix.py`). `stat_at` calls `fstatat` through `libc`.
    const HAVE_FSTATAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_FCHMODAT` — what `chmod` types its `dir_fd` as
    /// (`interp_posix.py`), and what `_chmod_path` calls to honour either
    /// modifier. `os.py:118` reads it as `chmod` honouring `dir_fd`; `os.py:179`
    /// deliberately does *not* read it for `follow_symlinks`, because a host can
    /// have `fchmodat` and still not honour `AT_SYMLINK_NOFOLLOW`.
    const HAVE_FCHMODAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `os.py:183` reads this as `chmod` honouring `follow_symlinks`, which is
    /// the `AT_SYMLINK_NOFOLLOW` arm of that same `fchmodat`. It is a narrower
    /// claim than `HAVE_FCHMODAT`: `os.py:159-177` records that where a host's
    /// `lchmod` is a stub returning ENOTSUP, the flag does not work either, so
    /// only the hosts that carry a working `lchmod` may say it — and those are
    /// exactly the ones `os.lchmod` is registered on below.
    const HAVE_LCHMOD: bool = HOST_POSIX && BSD_FLAVOURED;
    /// `os.py` reads this as `chflags` honouring `follow_symlinks`, which
    /// is the `lchflags` arm of the pair. `chflags` is a BSD interface, so the
    /// two names exist on exactly the hosts this is true for — and where they
    /// do not, `shutil.copystat`'s `lookup("chflags")` finds nothing and skips
    /// the flags rather than believing a stub that copied none.
    const HAVE_LCHFLAGS: bool = HOST_POSIX && BSD_FLAVOURED;
    /// Where `lchmod` and `chflags` are the host's own calls rather than stubs
    /// that report ENOTSUP — the platform half of the two bits above.
    const BSD_FLAVOURED: bool = cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "dragonfly",
    ));
    /// `rposix.HAVE_MKNODAT` — what `mknod` types its `dir_fd` as
    /// (`interp_posix.py`). Registered beside `mkfifo`, so it carries the
    /// same condition.
    const HAVE_MKNODAT: bool = HOST_POSIX;
    /// `rposix.HAVE_OPENAT` — what `open` types its `dir_fd` as
    /// (`interp_posix.py`). `openat` is reached through `libc`, so it needs
    /// no `host_env`; the Windows arm serves the name through `_wopen` and
    /// resolves nothing against a descriptor.
    const HAVE_OPENAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_MKDIRAT` — `mkdir`'s (`interp_posix.py`).
    const HAVE_MKDIRAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_UNLINKAT`. `os.py:131-132` reads it twice, for `unlink` and
    /// for `rmdir`, which are the same `unlinkat` told apart by `AT_REMOVEDIR`
    /// (`rposix.py:2717-2720`) — so the two cannot be advertised apart.
    const HAVE_UNLINKAT: bool = cfg!(all(unix, not(feature = "sandbox")));
    /// `rposix.HAVE_MKFIFOAT` — `mkfifo`'s (`interp_posix.py`). Narrower
    /// than the three above only because `os.mkfifo` itself is registered on
    /// the `host_env` POSIX builds; elsewhere the name is a noop placeholder
    /// that resolves nothing.
    const HAVE_MKFIFOAT: bool = HOST_POSIX;
    /// `rposix.HAVE_LINKAT` — what `link` takes its `src_dir_fd`/`dst_dir_fd`
    /// from. Its `linkat` call sits with the rest of the `host_env`-backed
    /// entry points, so it carries their condition and not `HAVE_FSTATAT`'s.
    const HAVE_LINKAT: bool = HOST_POSIX;
    /// `os.py` reads this as `listdir` and `scandir` taking a
    /// descriptor, which `fdlistdir` serves through `fdopendir`. `readdir`
    /// reports the end of a directory and a failure alike, so reading it apart
    /// needs the errno seam the host layer wraps — hence `HOST_POSIX` and not
    /// `HAVE_FSTATAT`'s weaker condition, which `HOST_POSIX` implies anyway for
    /// the `fstatat` a descriptor's entries stat themselves with.
    const HAVE_FDOPENDIR: bool = HOST_POSIX;
    /// `os.py:118` reads this as `lstat` honouring `dir_fd`, which is the same
    /// `fstatat` `stat` resolves one with — so the two cannot be advertised
    /// apart. `os.py:189` reads it a second time as `stat` honouring
    /// `follow_symlinks`; where this bit is false, `MS_WINDOWS` carries that
    /// second claim instead (`os.py:192`).
    const HAVE_LSTAT: bool = HAVE_FSTATAT;
    /// `_WIN32` (`interp_posix.py`). `os.py` reads it as `chmod`
    /// taking a descriptor (`:143`), and as `chmod` and `stat` honouring
    /// `follow_symlinks` (`:184`, `:192`) — all three of which the C runtime
    /// block below serves, and the noop placeholders do not.
    const MS_WINDOWS: bool = HOST_WINDOWS_CRT;

    /// `_DirFD_Unavailable` (`interp_posix.py`) names the argument and
    /// not the call it was passed to, which is also how
    /// `dir_fd_unavailable` reports it.
    fn dir_fd_unavailable() -> crate::PyError {
        crate::PyError::not_implemented("dir_fd unavailable on this platform")
    }

    /// `argument_unavailable` (`interp_posix.py`) — a modifier this
    /// platform has no call to apply, named together with the entry point that
    /// was asked to apply it.
    #[cfg(feature = "host_env")]
    fn argument_unavailable(funcname: &str, arg: &str) -> crate::PyError {
        crate::PyError::not_implemented(format!("{funcname}: {arg} unavailable on this platform"))
    }

    /// `os.link` takes its two names positionally and everything else by
    /// keyword, so a third positional argument is a `src_dir_fd` that would
    /// otherwise be dropped on the floor.
    #[cfg(all(feature = "host_env", not(feature = "sandbox")))]
    fn link_positional(args: &[pyre_object::PyObjectRef]) -> Result<(), crate::PyError> {
        if args.len() == 2 {
            return Ok(());
        }
        Err(crate::PyError::type_error(format!(
            "link() takes exactly 2 positional arguments ({} given)",
            args.len()
        )))
    }

    /// `do_stat` (`interp_posix.py`) resolves a name against an open
    /// directory descriptor with `fstatat`, where `AT_SYMLINK_NOFOLLOW`
    /// carries `follow_symlinks=False`. An absolute name ignores `dir_fd`,
    /// which is why the caller does not have to test for one.
    fn stat_at(
        name: &str,
        path: &crate::gateway::FsEncodedPath,
        dir_fd: i32,
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null character"))?;
            let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
            let flags = if follow_symlinks {
                0
            } else {
                libc::AT_SYMLINK_NOFOLLOW
            };
            let ret = unsafe { libc::fstatat(dir_fd, c_path.as_ptr(), st.as_mut_ptr(), flags) };
            if ret != 0 {
                let err = std::io::Error::last_os_error();
                return Err(errno_err_with_filename(
                    crate::builtins::io_error_posix_errno(&err, libc::EBADF),
                    path.w_path(),
                ));
            }
            let st = unsafe { st.assume_init() };
            #[cfg(target_os = "macos")]
            let st_flags = st.st_flags;
            #[cfg(not(target_os = "macos"))]
            let st_flags = 0u32;
            return Ok(stat_result_from_fields(
                &stat_fields_from_libc(&st),
                st_flags,
            ));
        }
        // Unreachable in practice — `stat_entry` turns a `dir_fd` away at
        // unwrap time wherever `HAVE_FSTATAT` is false — but the arm has to
        // exist for those targets to compile.
        #[allow(unreachable_code)]
        {
            let _ = (path, dir_fd, follow_symlinks);
            Err(dir_fd_unavailable())
        }
    }

    fn stat_path(
        path: &crate::gateway::FsEncodedPath,
        follow_symlinks: bool,
    ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        #[cfg(feature = "sandbox")]
        {
            let buf = if follow_symlinks {
                crate::host_seam::ops::stat(&path.as_bytes)
            } else {
                crate::host_seam::ops::lstat(&path.as_bytes)
            }
            .map_err(|e| crate::host_seam::seam_os_err_with_filename(e, path.w_path()))?;
            return Ok(make_stat_result_from_statbuf(&buf));
        }
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            return match win_stat_fields(&path.as_bytes, follow_symlinks) {
                Ok(fields) => Ok(stat_result_from_fields(&fields, 0)),
                Err(e) => Err(fs_err_with_filename2(
                    e,
                    2,
                    path.w_path(),
                    pyre_object::PY_NULL,
                )),
            };
        }
        #[cfg(all(not(feature = "sandbox"), not(all(windows, feature = "host_env"))))]
        {
            let meta = if follow_symlinks {
                host_fs::metadata(path_from_bytes(&path.as_bytes).as_ref())
            } else {
                host_fs::symlink_metadata(path_from_bytes(&path.as_bytes).as_ref())
            };
            match meta {
                Ok(m) => {
                    #[cfg(target_os = "macos")]
                    let st_flags = macos_path_st_flags(&path.as_bytes, follow_symlinks);
                    #[cfg(not(target_os = "macos"))]
                    let st_flags = 0u32;
                    Ok(make_stat_result(&m, st_flags))
                }
                Err(e) => Err(fs_err_with_filename2(
                    e,
                    2,
                    path.w_path(),
                    pyre_object::PY_NULL,
                )),
            }
        }
    }

    // ── posix.scandir(path=".") → ScandirIterator of DirEntry ──
    // `posix_scandir` / `posixmodule.c` DirEntry + ScandirIterator. The
    // entries are read eagerly into a list backing a context-manager
    // iterator so `with os.scandir(p) as it:` and `for e in it:` both work.
    //
    // DirEntry holds `name`/`path` as instance attributes; the type carries
    // is_dir/is_file/is_symlink/is_junction/stat/inode/__fspath__ which stat
    // the stored path on demand.
    type PyObjectRef = pyre_object::PyObjectRef;

    fn dir_entry_path(self_obj: PyObjectRef) -> Result<(PyObjectRef, Vec<u8>), crate::PyError> {
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        let path = extract_path(de.w_path)?;
        Ok((de.w_path, path))
    }
    fn dir_entry_get_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // GetSetProperty get: `(descriptor, w_obj)` — the entry is `args[1]`.
        let de = W_DirEntry::from_obj(args[1])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.w_name)
    }
    fn dir_entry_get_path(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let de = W_DirEntry::from_obj(args[1])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.w_path)
    }
    /// The descriptor the `scandir` that produced this entry was handed, or
    /// `-1` where it was given a name instead. An entry from a descriptor
    /// carries no directory in its `path` — `interp_scandir.py:50` leaves the
    /// prefix empty — so its own stat calls have to resolve the name against
    /// that descriptor rather than against the process's working directory.
    fn dir_entry_dir_fd(self_obj: PyObjectRef) -> Result<i32, crate::PyError> {
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.dir_fd)
    }
    /// `rposix_stat.fstatat(name, dirfd, follow)` — the call
    /// `interp_scandir.py:252-254,297-299` makes for exactly that reason. The
    /// errno comes back raw because the entry's own `path` is what names the
    /// failure (`:328` `wrap_oserror2(…, self.fget_path(space))`), and only the
    /// callers that report one hold it.
    #[cfg(all(unix, not(feature = "sandbox")))]
    fn dir_entry_stat_at(
        name: &[u8],
        dir_fd: i32,
        follow_symlinks: bool,
    ) -> Result<libc::stat, i32> {
        let c_name = std::ffi::CString::new(name).map_err(|_| libc::EINVAL)?;
        let mut st = std::mem::MaybeUninit::<libc::stat>::uninit();
        let flags = if follow_symlinks {
            0
        } else {
            libc::AT_SYMLINK_NOFOLLOW
        };
        let ret = unsafe { libc::fstatat(dir_fd, c_name.as_ptr(), st.as_mut_ptr(), flags) };
        if ret != 0 {
            let err = std::io::Error::last_os_error();
            return Err(crate::builtins::io_error_posix_errno(&err, libc::EBADF));
        }
        Ok(unsafe { st.assume_init() })
    }
    fn dir_entry_follow(args: &[PyObjectRef]) -> Result<bool, crate::PyError> {
        let (_pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
        match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
            Some(v) => crate::baseobjspace::is_true(v),
            None => Ok(true),
        }
    }
    /// The `S_IFMT` half of a mode, which is all `is_dir`/`is_file`/
    /// `is_symlink` read. The values are the same on every POSIX host, and
    /// spelling them here keeps the two arms below comparable.
    const S_IFMT: u32 = 0o170_000;
    const S_IFDIR: u32 = 0o040_000;
    const S_IFREG: u32 = 0o100_000;
    const S_IFLNK: u32 = 0o120_000;

    /// The `d_type` byte `readdir` reports — the `known_type` half of
    /// `interp_scandir.py`'s `flags`.  `DT_UNKNOWN` (`0`) is the `W_DirEntry`
    /// default, so an entry whose type the host did not report (a non-unix
    /// host, or a filesystem that answers `DT_UNKNOWN`) falls through to the
    /// stat `dir_entry_kind` runs.  Only the three types the tests read need a
    /// name.
    const DT_UNKNOWN: u8 = 0;
    const DT_DIR: u8 = 4;
    const DT_REG: u8 = 8;
    const DT_LNK: u8 = 10;

    fn dir_entry_known_type(self_obj: PyObjectRef) -> u8 {
        W_DirEntry::from_obj(self_obj).map_or(DT_UNKNOWN, |de| de.enum_type as u8)
    }

    /// Answer `is_dir`/`is_file`/`is_symlink` from the enumeration `d_type`
    /// when it decides the question, else `None` to fall through to a stat
    /// (`interp_scandir.py:399-426`).  `target` is the `DT_*` the query wants.
    /// An unknown type never decides.  A symlink decides every query but a
    /// followed `is_dir`/`is_file`, which need the target's type instead.
    fn dir_entry_kind_from_type(known: u8, target: u8, follow: bool) -> Option<bool> {
        if known == DT_UNKNOWN {
            None
        } else if known == target {
            Some(true)
        } else if follow && known == DT_LNK {
            None
        } else {
            Some(false)
        }
    }

    /// The file type an entry's name resolves to, or `None` for a name that has
    /// gone away — `check_mode` (`interp_scandir.py`) answers "no, not
    /// this type" for `ENOENT` alone, on the reasoning that a vanished entry is
    /// better reported as not being of the asked-for kind than as an error.
    /// Every other failure is the caller's to see, named by the entry.
    fn dir_entry_kind(args: &[PyObjectRef], follow: bool) -> Result<Option<u32>, crate::PyError> {
        let (w_path, path) = dir_entry_path(args[0])?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let dir_fd = dir_entry_dir_fd(args[0])?;
            if dir_fd != -1 {
                return match dir_entry_stat_at(&path, dir_fd, follow) {
                    Ok(st) => Ok(Some(st.st_mode as u32 & S_IFMT)),
                    Err(errno) if errno == libc::ENOENT => Ok(None),
                    Err(errno) => Err(errno_err_with_filename(errno, w_path)),
                };
            }
        }
        let meta = if follow {
            host_fs::metadata(path_from_bytes(&path).as_ref())
        } else {
            host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
        };
        match meta {
            Ok(m) => {
                let ft = m.file_type();
                Ok(Some(if ft.is_dir() {
                    S_IFDIR
                } else if ft.is_symlink() {
                    S_IFLNK
                } else if ft.is_file() {
                    S_IFREG
                } else {
                    0
                }))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(fs_err_with_filename(e, w_path)),
        }
    }
    fn dir_entry_is_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let follow = dir_entry_follow(args)?;
        let ans = match dir_entry_kind_from_type(dir_entry_known_type(args[0]), DT_DIR, follow) {
            Some(b) => b,
            None => dir_entry_kind(args, follow)? == Some(S_IFDIR),
        };
        Ok(pyre_object::w_bool_from(ans))
    }
    fn dir_entry_is_file(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let follow = dir_entry_follow(args)?;
        let ans = match dir_entry_kind_from_type(dir_entry_known_type(args[0]), DT_REG, follow) {
            Some(b) => b,
            None => dir_entry_kind(args, follow)? == Some(S_IFREG),
        };
        Ok(pyre_object::w_bool_from(ans))
    }
    fn dir_entry_is_symlink(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `is_symlink` never follows, so a known non-`DT_LNK` type answers `false`
        // and `DT_LNK` answers `true`; only `DT_UNKNOWN` needs the lstat.
        let ans = match dir_entry_kind_from_type(dir_entry_known_type(args[0]), DT_LNK, false) {
            Some(b) => b,
            None => dir_entry_kind(args, false)? == Some(S_IFLNK),
        };
        Ok(pyre_object::w_bool_from(ans))
    }
    fn dir_entry_is_junction(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // POSIX has no junction points.
        Ok(pyre_object::w_bool_from(false))
    }
    fn dir_entry_inode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        // `descr_inode` returns the inode `readdir` reported at enumeration when
        // the entry carries one (every unix `scandir` path, name or descriptor),
        // with no stat; `-1` means it has none and the stat paths below answer
        // instead.
        if let Some(de) = W_DirEntry::from_obj(args[0])
            && de.enum_ino != -1
        {
            return Ok(pyre_object::w_int_new(de.enum_ino));
        }
        let (w_path, path) = dir_entry_path(args[0])?;
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            let dir_fd = dir_entry_dir_fd(args[0])?;
            if dir_fd != -1 {
                let st = dir_entry_stat_at(&path, dir_fd, false)
                    .map_err(|errno| errno_err_with_filename(errno, w_path))?;
                return Ok(pyre_object::w_int_new(st.st_ino as i64));
            }
        }
        // `posixmodule.c DirEntry_inode` reads the same file index `os.stat`
        // reports, so the 0 this used to answer with on Windows disagreed with
        // `os.stat(entry.path).st_ino`.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            let fields =
                win_stat_fields(&path, false).map_err(|e| fs_err_with_filename(e, w_path))?;
            return Ok(pyre_object::w_int_new(fields.ino));
        }
        #[cfg(not(all(windows, feature = "host_env", not(feature = "sandbox"))))]
        {
            let meta = host_fs::symlink_metadata(path_from_bytes(&path).as_ref())
                .map_err(|e| fs_err_with_filename(e, w_path))?;
            #[cfg(unix)]
            let ino = {
                use std::os::unix::fs::MetadataExt;
                meta.ino() as i64
            };
            #[cfg(not(unix))]
            let ino = {
                let _ = &meta;
                0i64
            };
            Ok(pyre_object::w_int_new(ino))
        }
    }
    /// Fetch a fresh `stat` (`follow=true`) / `lstat` (`follow=false`) result —
    /// the uncached path shared by `dir_entry_stat`'s cache miss.  `dir_fd` is
    /// the descriptor the entry's `scandir` was handed (or `-1`); a real one
    /// resolves the entry's bare `name` through `fstatat`.
    fn dir_entry_fetch_stat(
        w_path: PyObjectRef,
        path: &[u8],
        follow: bool,
        dir_fd: i32,
    ) -> Result<PyObjectRef, crate::PyError> {
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            if dir_fd != -1 {
                let st = dir_entry_stat_at(path, dir_fd, follow)
                    .map_err(|errno| errno_err_with_filename(errno, w_path))?;
                #[cfg(target_os = "macos")]
                let st_flags = st.st_flags;
                #[cfg(not(target_os = "macos"))]
                let st_flags = 0u32;
                return Ok(stat_result_from_fields(
                    &stat_fields_from_libc(&st),
                    st_flags,
                ));
            }
        }
        #[cfg(not(all(unix, not(feature = "sandbox"))))]
        let _ = dir_fd;
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            return match win_stat_fields(path, follow) {
                Ok(fields) => Ok(stat_result_from_fields(&fields, 0)),
                Err(e) => Err(fs_err_with_filename(e, w_path)),
            };
        }
        #[cfg(not(all(windows, feature = "host_env", not(feature = "sandbox"))))]
        {
            let meta = if follow {
                host_fs::metadata(path_from_bytes(path).as_ref())
            } else {
                host_fs::symlink_metadata(path_from_bytes(path).as_ref())
            };
            match meta {
                Ok(m) => {
                    #[cfg(all(target_os = "macos", not(feature = "sandbox")))]
                    let st_flags = macos_path_st_flags(path, follow);
                    #[cfg(not(all(target_os = "macos", not(feature = "sandbox"))))]
                    let st_flags = 0u32;
                    Ok(make_stat_result(&m, st_flags))
                }
                Err(e) => Err(fs_err_with_filename(e, w_path)),
            }
        }
    }
    /// `posixmodule.c DirEntry_get_stat` caches the built result and hands back
    /// the same object — `follow_symlinks=True` into `w_stat`, `False` into
    /// `w_lstat` — so `entry.stat() is entry.stat()`.  (`interp_scandir.py
    /// descr_stat` caches only the raw stat data and rebuilds a fresh
    /// `build_stat_result` on every call, so under it the result identity
    /// differs; the 3.14 behavior is to cache the object, which is what this
    /// does.)  Only a successful fetch is cached; an error re-raises on each
    /// call.  The entry never moves (`allocate_stable`), so the raw receiver
    /// stays valid across the fetch's allocation.
    fn dir_entry_stat(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = args[0];
        let follow = dir_entry_follow(args)?;
        {
            let de = W_DirEntry::from_obj(self_obj)
                .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
            let cached = if follow { de.w_stat } else { de.w_lstat };
            if !cached.is_null() {
                return Ok(cached);
            }
        }
        let dir_fd = dir_entry_dir_fd(self_obj)?;
        let (w_path, path) = dir_entry_path(self_obj)?;
        let result = dir_entry_fetch_stat(w_path, &path, follow, dir_fd)?;
        let de = W_DirEntry::from_obj(self_obj)
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        if follow {
            de.w_stat = result;
        } else {
            de.w_lstat = result;
        }
        // `result` may be a nursery object stored into the stable entry; join
        // the remembered set so the next minor collection forwards it.
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(self_obj as *mut u8) };
        Ok(result)
    }
    fn dir_entry_fspath(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let de = W_DirEntry::from_obj(args[0])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?;
        Ok(de.w_path)
    }
    /// `posixmodule.c DirEntry_repr` — `"<DirEntry %R>"`, so the name is
    /// rendered by its own `repr`.  That keeps `os.scandir(b'.')`'s bytes
    /// name spelled `b'…'` and lets a name whose bytes have no UTF-8 form
    /// keep the lone surrogate `fs_name_obj` decoded it to.
    fn dir_entry_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let name = W_DirEntry::from_obj(args[0])
            .ok_or_else(|| crate::PyError::type_error("expected a 'posix.DirEntry' object"))?
            .w_name;
        let mut out = rustpython_wtf8::Wtf8Buf::new();
        out.push_str("<DirEntry ");
        out.push_wtf8(&unsafe { crate::py_repr_wtf8(name)? });
        out.push_str(">");
        // interp_scandir.py:230 returns `space.newtext(...)`: this rendered
        // value is an ordinary collectable runtime string.
        Ok(pyre_object::w_str_from_wtf8_managed(out))
    }
    /// `interp_scandir.py descr_reduce_ex` — an entry names a live
    /// position in a directory listing, so it refuses to be pickled.  `%T` is
    /// `error.py:592-593 space.type(value).name`, the qualified typedef name
    /// (`interp_scandir.py:469 'posix.DirEntry'`), which is what
    /// `w_type_get_name` returns; the flag-driven `reduce_newobj` refusal in
    /// `reduce_protocol_app.py:13-14` spells it with the bare `__name__`.
    fn dir_entry_reduce_ex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let type_name = match crate::typedef::r#type(args[0]) {
            Some(tp) => unsafe { pyre_object::typeobject::w_type_get_name(tp.as_ptr()) },
            None => "posix.DirEntry",
        };
        Err(crate::PyError::type_error(format!(
            "cannot pickle '{type_name}' object"
        )))
    }
    fn dir_entry_type() -> PyObjectRef {
        static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CELL.get_or_init(|| {
            // `interp_scandir.py` names the typedef `'posix.DirEntry'`.
            // `typedef.rs`'s `new_typeobject_with_base_and_layout` turns the
            // leading component of a qualified
            // builtin name into a `__module__` entry, so `type(e).__module__`
            // reports `posix` and every type-name-bearing error message is
            // spelled the way the typedef spells it.
            // The native `W_DirEntry` layout backs the type; `name`/`path` are
            // read-only getset descriptors over the inline fields, so instances
            // carry no `__dict__` (matching a `W_DirEntry` typedef with no
            // `makedict`).
            let tp = crate::typedef::make_builtin_type_with_layout(
                "posix.DirEntry",
                |ns| {
                    for (name, getter) in [
                        ("name", dir_entry_get_name as crate::gateway::BuiltinCodeFn),
                        ("path", dir_entry_get_path),
                    ] {
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                ns,
                                name,
                                crate::typedef::make_getset_descriptor_named(
                                    crate::gateway::make_builtin_function_with_arity(
                                        name, getter, 2,
                                    ),
                                    name,
                                ),
                            )
                        };
                    }
                    for (name, f) in [
                        ("is_dir", dir_entry_is_dir as crate::gateway::BuiltinCodeFn),
                        ("is_file", dir_entry_is_file),
                        ("is_symlink", dir_entry_is_symlink),
                        ("is_junction", dir_entry_is_junction),
                        ("inode", dir_entry_inode),
                        ("stat", dir_entry_stat),
                        ("__fspath__", dir_entry_fspath),
                        ("__repr__", dir_entry_repr),
                        ("__reduce_ex__", dir_entry_reduce_ex),
                    ] {
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                ns,
                                name,
                                crate::make_builtin_function(name, f),
                            )
                        };
                    }
                    // CPython 3.14 Modules/posixmodule.c DirEntry_methods —
                    // Py_GenericAlias with METH_CLASS.
                    unsafe {
                        pyre_object::w_dict_setitem_str(
                            ns,
                            "__class_getitem__",
                            pyre_object::function::w_classmethod_new(crate::make_builtin_function(
                                "__class_getitem__",
                                crate::_pypy_generic_alias::generic_alias_class_getitem,
                            )),
                        )
                    };
                },
                crate::typedef::w_object(),
                <W_DirEntry as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            );
            pyre_object::pyobject::set_instantiate(
                unsafe { &*<W_DirEntry as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
                tp,
            );
            // `interp_scandir.py:468-487` declares no `__new__` on the typedef
            // and `:487` sets `acceptable_as_base_class = False`; `typedef.py:55
            // acceptable_as_base_class = '__new__' in rawdict` is the rule, and
            // `typedef.py:754 assert not PyFrame.typedef.acceptable_as_base_class
            // # no __new__` is the same shape `typedef.rs`'s `init_typeobjects`
            // already ports for the `frame` and `traceback` types.
            // `scandir_fn` below allocates entries with `W_DirEntry::
            // allocate_stable`, which never enters `type.__call__`, so the
            // producer is untouched by the instantiation gate.
            unsafe {
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp as usize
        }) as PyObjectRef
    }

    fn scandir_iter_self(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        Ok(args[0])
    }

    /// Every mutable borrow of the native iterator is derived, used and dropped
    /// inside the serializer, so no two callers ever hold overlapping
    /// references to the same `W_ScandirIterator`.
    fn with_scandir_iter<R>(
        self_obj: PyObjectRef,
        body: impl FnOnce(&mut W_ScandirIterator) -> R,
    ) -> Option<R> {
        let _serialized = SCANDIR_IN_NEXT_SERIALIZER
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        W_ScandirIterator::from_obj(self_obj).map(body)
    }

    fn scandir_iter_mark_closed(self_obj: PyObjectRef) {
        // `W_ScandirIterator._close` clears the state inspected by
        // `_finalize_`, whether closure is explicit or due to exhaustion.
        let _ = with_scandir_iter(self_obj, |iterator| {
            iterator.open = false;
        });
    }
    fn scandir_iter_close(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        scandir_iter_mark_closed(args[0]);
        Ok(pyre_object::w_none())
    }
    /// What a `next()` may do, decided in one serialized read of the
    /// iterator's flags.
    enum ScandirStep {
        /// Enumeration is over, either by `close()` or by exhaustion.
        Ended,
        /// Another step holds `_in_next` (interp_scandir.py:133-135).
        InProgress,
        /// This call owns the step and must release it.
        Claimed,
    }

    /// `_in_next` around one enumeration step: `true` on the way in, `false` on
    /// every way out (interp_scandir.py:136,158).  The open flag is read in the
    /// same serialized region, so the answer names a state no concurrent
    /// `close()` can be halfway through.
    fn scandir_iter_claim_next(self_obj: PyObjectRef) -> Option<ScandirStep> {
        with_scandir_iter(self_obj, |iterator| {
            // `W_ScandirIterator.next_w` ends enumeration after `close()`, without
            // yielding entries already buffered in the native owner.
            if !iterator.open {
                return ScandirStep::Ended;
            }
            if iterator.in_next {
                return ScandirStep::InProgress;
            }
            iterator.in_next = true;
            ScandirStep::Claimed
        })
    }

    fn scandir_iter_release_next(self_obj: PyObjectRef) {
        let _ = with_scandir_iter(self_obj, |iterator| {
            iterator.in_next = false;
        });
    }

    /// One enumeration step, with the step already claimed.
    fn scandir_iter_next_entry(self_obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        with_scandir_iter(self_obj, |iterator| {
            let idx = iterator.index;
            let entries = iterator.entries;
            let len = unsafe { pyre_object::w_list_len(entries) } as i64;
            if idx >= len {
                iterator.open = false;
                return Err(crate::PyError::stop_iteration());
            }
            let Some(item) = (unsafe { pyre_object::w_list_getitem(entries, idx) }) else {
                iterator.open = false;
                return Err(crate::PyError::stop_iteration());
            };
            iterator.index = idx + 1;
            Ok(item)
        })
        .unwrap_or_else(|| {
            Err(crate::PyError::type_error(
                "expected a 'posix.ScandirIterator' object",
            ))
        })
    }

    fn scandir_iter_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = args[0];
        let step = scandir_iter_claim_next(self_obj).ok_or_else(|| {
            crate::PyError::type_error("expected a 'posix.ScandirIterator' object")
        })?;
        match step {
            ScandirStep::Ended => return Err(crate::PyError::stop_iteration()),
            // interp_scandir.py:133-135 refuses a step taken while another is
            // in progress, and refuses it through `fail`, which closes the
            // iterator before raising.  Without this two steps read one `index`
            // and hand out the same entry twice.
            ScandirStep::InProgress => {
                scandir_iter_mark_closed(self_obj);
                return Err(crate::PyError::runtime_error(
                    "cannot use ScandirIterator from multiple threads concurrently",
                ));
            }
            ScandirStep::Claimed => {}
        }
        let result = scandir_iter_next_entry(self_obj);
        scandir_iter_release_next(self_obj);
        result
    }
    fn scandir_iter_is_open(self_obj: PyObjectRef) -> bool {
        with_scandir_iter(self_obj, |iterator| iterator.open).unwrap_or(false)
    }
    fn scandir_iter_del(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let self_obj = args[0];
        if !scandir_iter_is_open(self_obj) {
            return Ok(pyre_object::w_none());
        }

        let message = match unsafe { crate::display::py_repr_wtf8(self_obj) } {
            Ok(repr) => format!(
                "unclosed scandir iterator {}",
                repr.to_string_lossy()
            ),
            Err(_) => "unclosed scandir iterator".to_string(),
        };
        if let Err(mut error) = crate::warn::warn_category(&message, "ResourceWarning", 1) {
            // `W_ScandirIterator._finalize_` reports a warning promoted to an
            // error as unraisable because finalization cannot propagate it.
            error.write_unraisable(
                pyre_object::w_none(),
                rustpython_wtf8::Wtf8::new(""),
                self_obj,
            );
        }
        scandir_iter_mark_closed(self_obj);
        Ok(pyre_object::w_none())
    }
    fn scandir_iter_type() -> PyObjectRef {
        static CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CELL.get_or_init(|| {
            // `interp_scandir.py` names the typedef `'posix.ScandirIterator'`.
            let tp = crate::typedef::make_builtin_type_with_layout(
                "posix.ScandirIterator",
                |ns| {
                    for (name, f) in [
                        (
                            "__iter__",
                            scandir_iter_self as crate::gateway::BuiltinCodeFn,
                        ),
                        ("__next__", scandir_iter_next),
                        ("__enter__", scandir_iter_self),
                        ("__exit__", scandir_iter_close),
                        ("close", scandir_iter_close),
                        // `interp_scandir.py:172-180` keeps finalization on the
                        // RPython-internal `_finalize_` and publishes no
                        // `__del__`.  3.14 makes `__del__` a real entry in
                        // `posix.ScandirIterator`'s type dict, so it is
                        // published here, with the arity-1 binding below that
                        // makes it callable as an ordinary method.
                        ("__del__", scandir_iter_del),
                    ] {
                        let function = if name == "__del__" {
                            crate::make_builtin_function_with_arity(name, f, 1)
                        } else {
                            crate::make_builtin_function(name, f)
                        };
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(
                                ns, name, function,
                            )
                        };
                    }
                },
                crate::typedef::w_object(),
                <W_ScandirIterator as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
            );
            pyre_object::pyobject::set_instantiate(
                unsafe {
                    &*<W_ScandirIterator as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE
                },
                tp,
            );
            unsafe { pyre_object::w_type_set_hasuserdel(tp, true) };
            // PyPy's `W_ScandirIterator.typedef` has no `__new__` and disallows
            // subclassing; `scandir_fn` creates instances with
            // `W_ScandirIterator::allocate_stable`.
            unsafe {
                pyre_object::typeobject::w_type_set_disallow_instantiation(tp);
                pyre_object::typeobject::w_type_set_acceptable_as_base_class(tp, false);
            }
            tp as usize
        }) as PyObjectRef
    }

    /// Allocate one `W_DirEntry` and append it to `list` (pinned at `list_slot`
    /// on the shadow stack).  Each string is pinned before the next allocation
    /// so a moving collection during `allocate_stable` forwards it; the entry
    /// is stable but its young strings join the remembered set.  `dir_fd` is the
    /// descriptor the entry resolves its own `name` against, or `-1`.  `enum_ino`
    /// is the `readdir` inode (or `-1` when the enumeration did not carry one).
    /// Join a `scandir` path prefix to an entry name the way
    /// `interp_scandir.py:65-67` builds `w_path_prefix`: a separator goes
    /// between them unless the prefix is empty or already ends in one.
    #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
    fn join_dir_name(prefix: &[u8], name: &[u8]) -> Vec<u8> {
        let mut full = Vec::with_capacity(prefix.len() + 1 + name.len());
        full.extend_from_slice(prefix);
        if !prefix.is_empty() && full.last() != Some(&b'/') {
            full.push(b'/');
        }
        full.extend_from_slice(name);
        full
    }
    fn scandir_push_entry(
        list_slot: usize,
        bytes_mode: bool,
        name: &[u8],
        full: &[u8],
        dir_fd: i32,
        enum_ino: i64,
        enum_type: u8,
    ) {
        let _entry_scope = pyre_object::gc_roots::push_roots();
        let base = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(fs_name_obj(bytes_mode, name));
        pyre_object::gc_roots::pin_root(fs_name_obj(bytes_mode, full));
        pyre_object::gc_roots::pin_root(W_DirEntry::allocate_stable(W_DirEntry::default()));
        let obj = pyre_object::gc_roots::shadow_stack_get(base + 2);
        let de = W_DirEntry::from_obj(obj).expect("freshly allocated posix.DirEntry");
        de.w_name = pyre_object::gc_roots::shadow_stack_get(base);
        de.w_path = pyre_object::gc_roots::shadow_stack_get(base + 1);
        de.dir_fd = dir_fd;
        de.enum_ino = enum_ino;
        de.enum_type = enum_type as i32;
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8) };
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        unsafe { pyre_object::w_list_append(list, obj) };
    }
    fn scandir_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let (bound, _kwargs) = bind_path_args(args, "scandir", &["path"], 0, &[])?;
        // One resolution yields both the path and its bytes-ness, so
        // `__fspath__` runs exactly once. The omitted argument is the same
        // `None` the signature names, which resolves to `"."` there.
        let arg = bound[0].unwrap_or(pyre_object::w_none());
        let resolved =
            crate::gateway::fsencode_path_or_fd_nullable_w(arg, "scandir", HAVE_FDOPENDIR)?;
        let bytes_mode = unsafe { resolved.is_bytes() };
        let path = resolved.as_bytes.as_slice();
        let w_path = || resolved.w_path();
        // Initialise the DirEntry type (`set_instantiate`) before allocating
        // any entry of it.
        let _ = dir_entry_type();
        let list = pyre_object::w_list_new(Vec::new());
        // The entries are `allocate_stable` (non-nursery) objects, but a stable
        // allocation can still drive a moving collection over a large listing,
        // so `list` and each per-entry temporary live on the shadow stack.
        let _list_scope = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(list);
        let list_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // A `scandir(fd)` lists through the descriptor and every entry records
        // it (`-1` for the entries a name produced), so its own stat resolves
        // the bare name against that descriptor.
        #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
        let from_fd = Some(resolved.as_fd).filter(|&fd| fd != -1);
        #[cfg(not(all(unix, feature = "host_env", not(feature = "sandbox"))))]
        let from_fd: Option<i32> = None;
        match from_fd {
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            Some(fd) => {
                // `interp_scandir.py:50` leaves the path prefix empty for a
                // descriptor — there is no directory to join — so an entry's
                // `path` is its bare `name`, and a descriptor is not `bytes`,
                // so both come back as `str`. Every entry records the
                // descriptor so its own stat resolves the bare name against it.
                fd_readdir(fd, |name, ino, d_type| {
                    scandir_push_entry(list_slot, false, name, name, fd, ino, d_type);
                })
                .map_err(|errno| errno_err_with_filename(errno, w_path()))?;
            }
            // A name is enumerated through `opendir`/`readdir` so each entry
            // carries the `d_ino` and `d_type` the dirent reports
            // (`interp_scandir.py:148-153`): `inode()` answers from `d_ino`
            // and `is_dir`/`is_file`/`is_symlink` from `d_type`, both without
            // a stat.
            #[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
            _ => {
                let c_path = std::ffi::CString::new(path)
                    .map_err(|_| crate::PyError::value_error("embedded null byte"))?;
                let dirp = unsafe { libc::opendir(c_path.as_ptr()) };
                if dirp.is_null() {
                    return Err(errno_err_with_filename(
                        crate::builtins::crt_errno(),
                        w_path(),
                    ));
                }
                let errno = readdir_collect(dirp, |name, ino, d_type| {
                    let full = join_dir_name(path, name);
                    scandir_push_entry(list_slot, bytes_mode, name, &full, -1, ino, d_type);
                });
                unsafe { libc::closedir(dirp) };
                if errno != 0 {
                    return Err(errno_err_with_filename(errno, w_path()));
                }
            }
            // No raw `readdir` to read the dirent from (wasm, the sandbox seam,
            // or a build without `host_env`), so `d_type` is unknown and
            // `is_dir` stats; the inode is still free from the dirent on unix.
            #[cfg(not(all(unix, feature = "host_env", not(feature = "sandbox"))))]
            _ => {
                let entries = host_fs::read_dir(path_from_bytes(path).as_ref())
                    .map_err(|e| fs_err_with_filename(e, w_path()))?;
                for entry in entries {
                    let entry = entry.map_err(|e| fs_err_with_filename(e, w_path()))?;
                    let name = entry.file_name();
                    let full = entry.path().into_os_string();
                    #[cfg(unix)]
                    let enum_ino = {
                        use std::os::unix::fs::DirEntryExt;
                        entry.ino() as i64
                    };
                    #[cfg(not(unix))]
                    let enum_ino = -1i64;
                    scandir_push_entry(
                        list_slot,
                        bytes_mode,
                        name.as_encoded_bytes(),
                        full.as_encoded_bytes(),
                        -1,
                        enum_ino,
                        DT_UNKNOWN,
                    );
                }
            }
        }
        // Initialise the iterator type before allocating its native owner, then
        // pin that stable owner while connecting it to the entries list.
        let _ = scandir_iter_type();
        pyre_object::gc_roots::pin_root(W_ScandirIterator::allocate_stable(
            W_ScandirIterator::default(),
        ));
        let it_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let it = pyre_object::gc_roots::shadow_stack_get(it_slot);
        let list = pyre_object::gc_roots::shadow_stack_get(list_slot);
        let iterator = W_ScandirIterator::from_obj(it)
            .expect("freshly allocated posix.ScandirIterator");
        iterator.entries = list;
        iterator.open = true;
        unsafe { pyre_object::gc_hook::try_gc_write_barrier(it as *mut u8) };
        // `StdObjSpace.allocate_instance` immediately queues instances whose
        // type has `hasuserdel`. This native allocation bypasses that helper,
        // so it must register the new iterator explicitly.
        pyre_object::gc_hook::maybe_register_finalizer(it);
        let it = pyre_object::gc_roots::shadow_stack_get(it_slot);
        drop(_list_scope);
        Ok(it)
    }
    crate::module_ns_store(
        ns,
        "scandir",
        crate::make_builtin_function("scandir", scandir_fn),
    );
    crate::module_ns_store(ns, "DirEntry", dir_entry_type());

    // os.uname() — returns structseq (sysname, nodename, release, version, machine).
    // Routed through `host_env::posix::uname_info` when available so the
    // result reports the host's real POSIX strings ("Darwin", "Linux",
    // node hostname, kernel release, etc.) instead of Rust's compile-time
    // `std::env::consts::OS` ("macos"/"linux"/...).
    //
    // POSIX only, the way `HAVE_UNAME` gates it. Its callers read presence as
    // "the POSIX identification is available": `platform.uname` falls back to
    // `sys.platform` on AttributeError, and `sysconfig.get_platform` tests
    // `hasattr(os, 'uname')` directly. A build with no host strings to report
    // would answer both with the compile-time constants instead.
    #[cfg(unix)]
    crate::module_ns_store(
        ns,
        "uname",
        crate::make_builtin_function_with_arity(
            "uname",
            |_| {
                #[cfg(all(unix, feature = "host_env"))]
                let (sysname, nodename, release, version, machine) = {
                    let info = rustpython_host_env::posix::uname_info().unwrap_or(
                        rustpython_host_env::posix::UnameInfo {
                            sysname: String::new(),
                            nodename: String::new(),
                            release: String::new(),
                            version: String::new(),
                            machine: String::new(),
                        },
                    );
                    (
                        info.sysname,
                        info.nodename,
                        info.release,
                        info.version,
                        info.machine,
                    )
                };
                #[cfg(not(all(unix, feature = "host_env")))]
                let (sysname, nodename, release, version, machine) = (
                    std::env::consts::OS.to_string(),
                    String::new(),
                    String::new(),
                    String::new(),
                    std::env::consts::ARCH.to_string(),
                );
                Ok(crate::_structseq::new_instance(
                    uname_result_seq_type(),
                    vec![
                        pyre_object::w_str_new(&sysname),
                        pyre_object::w_str_new(&nodename),
                        pyre_object::w_str_new(&release),
                        pyre_object::w_str_new(&version),
                        pyre_object::w_str_new(&machine),
                    ],
                ))
            },
            0,
        ),
    );
    crate::module_ns_store(
        ns,
        "stat",
        crate::gateway::make_builtin_function_with_text_signature(
            "stat",
            |args| stat_entry(args, true),
            "(path, *, dir_fd=None, follow_symlinks=True)",
        ),
    );
    crate::module_ns_store(
        ns,
        "lstat",
        crate::make_builtin_function("lstat", |args| stat_entry(args, false)),
    );
    /// `rposix_stat.py fstat`: the descriptor form both `os.fstat` and
    /// `os.stat` with a descriptor answer through, so the two cannot drift.
    fn fstat_fd(fd: i32) -> Result<pyre_object::PyObjectRef, crate::PyError> {
        // `rposix_stat.py:fstat` passes the descriptor to libc, where
        // `-1` reports EBADF.  Rust's `OwnedFd::from_raw_fd(-1)`
        // asserts before `File::metadata` can produce that error.
        // Windows answers a descriptor it cannot name with the Win32 error
        // instead (`_Py_fstat_noraise` sets ERROR_INVALID_HANDLE itself), so
        // leave that arm to report it.
        #[cfg(not(all(windows, feature = "host_env", not(feature = "sandbox"))))]
        if fd == -1 {
            return Err(crate::PyError::os_error_with_errno(
                libc::EBADF,
                std::io::Error::from_raw_os_error(libc::EBADF).to_string(),
            ));
        }
        #[cfg(feature = "sandbox")]
        {
            let buf = crate::host_seam::ops::fstat(fd)
                .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
            Ok(make_stat_result_from_statbuf(&buf))
        }
        #[cfg(all(unix, not(feature = "sandbox")))]
        {
            use std::os::unix::io::FromRawFd;
            // interp_posix.py `fstat`: retry its syscall on EINTR.
            let meta = loop {
                let f = unsafe { std::fs::File::from_raw_fd(fd) };
                let meta = f.metadata();
                let _ = std::mem::ManuallyDrop::new(f); // don't close
                match meta {
                    Ok(m) => break m,
                    Err(e) => crate::builtins::eintr_retry_with(e, |e| {
                        crate::PyError::os_error_with_errno(
                            crate::builtins::io_error_posix_errno(&e, 9),
                            format!("{}", e),
                        )
                    })?,
                }
            };
            #[cfg(target_os = "macos")]
            let st_flags = macos_fd_st_flags(fd);
            #[cfg(not(target_os = "macos"))]
            let st_flags = 0u32;
            Ok(make_stat_result(&meta, st_flags))
        }
        // `_Py_fstat_noraise`: the descriptor's underlying handle, then
        // `GetFileInformationByHandle` — which is what `File::metadata`
        // is here.  A descriptor with no handle is reported as the
        // Win32 `ERROR_INVALID_HANDLE`, the errno spelling of which is
        // `EBADF`.
        #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
        {
            let invalid_handle = || {
                crate::PyError::os_error_win32_syscall2(
                    windows_sys::Win32::Foundation::ERROR_INVALID_HANDLE as i32,
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                )
            };
            let borrowed = unsafe { rustpython_host_env::crt_fd::Borrowed::borrow_raw(fd) };
            // Same `StatStruct` the path forms take, for the reason
            // [`win_stat_fields`] states: `std::fs::Metadata` carries no file
            // index or volume serial, so answering from it would report a
            // different identity for the very file `os.stat(path)` just named.
            // It also answers a character device or pipe with the format bits
            // `GetFileType` reports rather than a disk file's.
            match rustpython_host_env::fileutils::fstat(borrowed) {
                Ok(st) => Ok(stat_result_from_fields(
                    &stat_fields_from_statstruct(&st),
                    0,
                )),
                Err(e) => Err(match e.raw_os_error() {
                    Some(winerror) => crate::PyError::os_error_win32_syscall2(
                        winerror,
                        pyre_object::PY_NULL,
                        pyre_object::PY_NULL,
                    ),
                    None => invalid_handle(),
                }),
            }
        }
        #[cfg(not(any(unix, feature = "sandbox", all(windows, feature = "host_env"))))]
        Err(crate::PyError::os_error_with_errno(
            9,
            "fstat unsupported".to_string(),
        ))
    }

    crate::module_ns_store(
        ns,
        "fstat",
        crate::make_builtin_function_with_arity(
            "fstat",
            |args| {
                if args.is_empty() {
                    return Err(crate::PyError::type_error("fstat() missing argument"));
                }
                fstat_fd(crate::baseobjspace::c_int_w(args[0])?)
            },
            1,
        ),
    );
    // stat_result type — structseq (tuple subclass). Exported so that
    // `posix.stat_result` and `isinstance(os.stat(p), os.stat_result)` work.
    crate::module_ns_store(ns, "stat_result", stat_result_seq_type());
    // os.getcwd() — PyPy: posixmodule.c posix_getcwd.
    crate::module_ns_store(
        ns,
        "getcwd",
        crate::make_builtin_function_with_arity(
            "getcwd",
            |_| {
                // `interp_posix.py` is `space.fsdecode(getcwdb(space))`,
                // so the directory's bytes reach Python through the filesystem
                // decoder and a byte with no UTF-8 spelling survives as its
                // surrogate escape. A lossy decode would fold it to U+FFFD,
                // which breaks the `os.fsencode(os.getcwd())` round trip and
                // makes two different directories compare equal.
                #[cfg(feature = "sandbox")]
                {
                    let cwd = crate::host_seam::ops::getcwd()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    Ok(crate::gateway::fsdecode_filename_bytes(&cwd))
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    #[cfg(feature = "host_env")]
                    {
                        if let Ok(cwd) = host_os::current_dir() {
                            return Ok(crate::gateway::fsdecode_os_str(cwd.as_os_str()));
                        }
                    }
                    Ok(pyre_object::w_str_new(""))
                }
            },
            0,
        ),
    );
    // os.getcwdb() — bytes form of getcwd.
    crate::module_ns_store(
        ns,
        "getcwdb",
        crate::make_builtin_function_with_arity(
            "getcwdb",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let cwd = crate::host_seam::ops::getcwd()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    Ok(pyre_object::w_bytes_from_bytes(&cwd))
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    #[cfg(feature = "host_env")]
                    {
                        if let Ok(cwd) = host_os::current_dir() {
                            return Ok(pyre_object::w_bytes_from_bytes(
                                cwd.as_os_str().as_encoded_bytes(),
                            ));
                        }
                    }
                    Ok(pyre_object::w_bytes_from_bytes(b""))
                }
            },
            0,
        ),
    );
    // os.getuid / geteuid / getgid / getegid — real syscalls (each builtin's
    // `#[cfg(feature = "sandbox")]` arm routes through `host_seam::ops`
    // instead).
    #[cfg(all(unix, not(feature = "sandbox")))]
    unsafe extern "C" {
        fn getuid() -> u32;
        fn geteuid() -> u32;
        fn getgid() -> u32;
        fn getegid() -> u32;
    }
    // The user and group ids are POSIX's; `nt` has none of the four, and code
    // reads `hasattr(os, 'geteuid')` to decide whether an ownership check is
    // meaningful at all.
    #[cfg(not(windows))]
    crate::module_ns_store(
        ns,
        "getuid",
        crate::make_builtin_function_with_arity(
            "getuid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getuid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(getuid() as i64))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    #[cfg(not(windows))]
    crate::module_ns_store(
        ns,
        "geteuid",
        crate::make_builtin_function_with_arity(
            "geteuid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::geteuid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(geteuid() as i64))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    #[cfg(not(windows))]
    crate::module_ns_store(
        ns,
        "getgid",
        crate::make_builtin_function_with_arity(
            "getgid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getgid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(getgid() as i64))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    #[cfg(not(windows))]
    crate::module_ns_store(
        ns,
        "getegid",
        crate::make_builtin_function_with_arity(
            "getegid",
            |_| {
                #[cfg(feature = "sandbox")]
                {
                    let v = crate::host_seam::ops::getegid()
                        .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                    return Ok(pyre_object::w_int_new(v));
                }
                #[cfg(all(unix, not(feature = "sandbox")))]
                unsafe {
                    Ok(pyre_object::w_int_new(getegid() as i64))
                }
                #[cfg(not(any(unix, feature = "sandbox")))]
                Ok(pyre_object::w_int_new(0))
            },
            0,
        ),
    );
    // os.getpid — host_os::process_id (std::process::id).
    crate::module_ns_store(
        ns,
        "getpid",
        crate::make_builtin_function_with_arity(
            "getpid",
            |_| Ok(pyre_object::w_int_new(host_os::process_id() as i64)),
            0,
        ),
    );
    // `getenv` is not bound here. `os.py` writes it against `environ`
    // — the dict this module publishes and that os.py's `_Environ` wrapper
    // writes back into — and names it in its own `__all__`, so a binding is
    // both shadowed and counted twice through the star-import.
    // ── host_env::posix-backed real implementations (override the noop
    //    placeholders registered above) ───────────────────────────────
    #[cfg(all(unix, feature = "host_env"))]
    {
        use rustpython_host_env::posix as host_posix;

        fn exec_argv(
            w_argv: PyObjectRef,
            function: &str,
        ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
            // PyPy interp_posix.execv: `space.unpackiterable(w_argv)` followed
            // by `space.fsencode_w` for every argument.
            let items = crate::baseobjspace::unpackiterable(w_argv, -1).map_err(|error| {
                if error.kind == crate::PyErrorKind::TypeError {
                    crate::PyError::type_error(format!(
                        "{function}() arg 2 must be an iterable of strings"
                    ))
                } else {
                    error
                }
            })?;
            if items.is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 must not be empty"
                )));
            }
            let mut argv = Vec::with_capacity(items.len());
            for item in items {
                // An element is converted on the sequence's behalf, not as an
                // argument of the call, so the caller-less message is the one
                // it reports — measured, and the same for the environment
                // below and for `posix_spawn`'s file actions.
                let value = extract_path(item)?;
                argv.push(std::ffi::CString::new(value).map_err(|_| {
                    crate::PyError::value_error(format!(
                        "{function}() arg 2 contains an embedded null byte"
                    ))
                })?);
            }
            if argv[0].as_bytes().is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 first element cannot be empty"
                )));
            }
            Ok(argv)
        }

        fn exec_pointer_array(values: &[std::ffi::CString]) -> Vec<*const libc::c_char> {
            let mut pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
            pointers.push(std::ptr::null());
            pointers
        }

        // PyPy interp_posix.execv: this call replaces the current process and
        // returns only to translate the host errno into OSError.
        crate::module_ns_store(
            ns,
            "execv",
            crate::make_builtin_function_with_arity(
                "execv",
                |args| {
                    // The path names itself; the argv entries below do not,
                    // because each of those is converted on the sequence's
                    // behalf rather than as an argument of its own.
                    let command =
                        crate::gateway::fsencode_path_named_w(args[0], "execv", "path")?.as_bytes;
                    let command_c = std::ffi::CString::new(command).map_err(|_| {
                        crate::PyError::value_error("execv() path contains an embedded null byte")
                    })?;
                    let argv = exec_argv(args[1], "execv")?;
                    let argv_ptrs = exec_pointer_array(&argv);
                    let errno =
                        host_posix::exec_replace(&[command_c], argv_ptrs.as_ptr(), None) as i32;
                    // interp_posix.py:1814-1817 uses `wrap_oserror`, which does
                    // not attach the command path.
                    Err(errno_err(errno, ""))
                },
                2,
            ),
        );

        // PyPy interp_posix.execve/_env2interp: accept a mapping, fsencode
        // names and values, reject illegal names, then replace the process.
        crate::module_ns_store(
            ns,
            "execve",
            crate::make_builtin_function_with_arity(
                "execve",
                |args| {
                    let command =
                        crate::gateway::fsencode_path_named_w(args[0], "execve", "path")?.as_bytes;
                    let command_c = std::ffi::CString::new(command).map_err(|_| {
                        crate::PyError::value_error("execve() path contains an embedded null byte")
                    })?;
                    let argv = exec_argv(args[1], "execve")?;
                    let argv_ptrs = exec_pointer_array(&argv);

                    let env = collect_env_entries(args[2], "execve", false)?
                        .into_iter()
                        .map(|entry| {
                            std::ffi::CString::new(entry).map_err(|_| {
                                crate::PyError::value_error(
                                    "execve() environment contains an embedded null byte",
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let env_ptrs = exec_pointer_array(&env);
                    let errno = host_posix::exec_replace(
                        &[command_c],
                        argv_ptrs.as_ptr(),
                        Some(env_ptrs.as_ptr()),
                    ) as i32;
                    // `interp_posix.py:1812,1817` wraps the failure with
                    // `wrap_oserror`, which names no file, so the path stays
                    // out of the error the same way `execv` leaves it out.
                    Err(errno_err(errno, ""))
                },
                3,
            ),
        );

        // os.strerror(code) -> str
        crate::module_ns_store(
            ns,
            "strerror",
            crate::make_builtin_function_with_arity(
                "strerror",
                |args| {
                    let code = match args.first() {
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error(
                                "strerror() requires 1 argument",
                            ));
                        }
                    };
                    #[cfg(feature = "sandbox")]
                    {
                        let msg = crate::host_seam::ops::strerror(code)
                            .map_err(|e| crate::host_seam::seam_os_err(e, ""))?;
                        // The text comes from the C library in the current
                        // locale, so a byte with no UTF-8 spelling takes the
                        // surrogateescape rather than U+FFFD.
                        return Ok(crate::typedef::charp2uni(&msg));
                    }
                    #[cfg(not(feature = "sandbox"))]
                    Ok(pyre_object::w_str_new(
                        &rustpython_host_env::time::strerror(code),
                    ))
                },
                1,
            ),
        );

        // os.pipe() -> (r_fd, w_fd)
        crate::module_ns_store(
            ns,
            "pipe",
            crate::make_builtin_function_with_arity(
                "pipe",
                |_| match host_posix::pipe() {
                    Ok((rfd, wfd)) => {
                        use std::os::fd::IntoRawFd;
                        Ok(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(rfd.into_raw_fd() as i64),
                            pyre_object::w_int_new(wfd.into_raw_fd() as i64),
                        ]))
                    }
                    Err(e) => Err(io_err(e, "")),
                },
                0,
            ),
        );

        // os.pipe2(flags) -> (r_fd, w_fd)
        //
        // `interp_posix.py`, which — unlike `pipe` two blocks up —
        // forces no inheritance on the pair afterwards: the flags argument is
        // the whole of the caller's control over it.
        #[cfg(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        crate::module_ns_store(
            ns,
            "pipe2",
            crate::make_builtin_function_with_arity(
                "pipe2",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("pipe2() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(flags=c_int)`.
                    let flags = crate::baseobjspace::c_int_w(args[0])?;
                    match host_posix::pipe2(flags) {
                        Ok((rfd, wfd)) => {
                            use std::os::fd::IntoRawFd;
                            Ok(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new(rfd.into_raw_fd() as i64),
                                pyre_object::w_int_new(wfd.into_raw_fd() as i64),
                            ]))
                        }
                        Err(e) => Err(io_err(e, "")),
                    }
                },
                1,
            ),
        );

        // os.sched_yield()
        crate::module_ns_store(
            ns,
            "sched_yield",
            crate::make_builtin_function_with_arity(
                "sched_yield",
                |_| {
                    // interp_posix.py `sched_yield`: retry on EINTR.
                    loop {
                        match host_posix::sched_yield() {
                            Ok(()) => break,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    }
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );

        // os.nice(increment) -> new niceness
        crate::module_ns_store(
            ns,
            "nice",
            crate::make_builtin_function_with_arity(
                "nice",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("nice() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(increment=c_int)`.
                    let inc = crate::baseobjspace::c_int_w(args[0])?;
                    let n = host_posix::nice(inc).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.umask(mask) -> previous mask
        crate::module_ns_store(
            ns,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(mask=c_int)`.
                    let mask = crate::baseobjspace::c_int_w(args[0])? as libc::mode_t;
                    let prev = host_posix::umask(mask);
                    Ok(pyre_object::w_int_new(prev as i64))
                },
                1,
            ),
        );

        // os.getlogin() -> str
        crate::module_ns_store(
            ns,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| match host_posix::getlogin() {
                    // A login name is an OS string; decode it the way every
                    // other one is so an undecodable byte keeps its escape.
                    Some(name) => Ok(crate::gateway::fsdecode_filename_bytes(name.as_bytes())),
                    None => Err(crate::PyError::os_error_with_errno(
                        crate::builtins::io_error_posix_errno(&std::io::Error::last_os_error(), 0),
                        "getlogin",
                    )),
                },
                0,
            ),
        );

        // `getgroups(2)` reports at most `NGROUPS_MAX` entries, so a process in
        // more groups than that gets a truncated list. `<unistd.h>` aliases the
        // name to an unlimited variant under `_DARWIN_C_SOURCE`, which is the
        // one the reference implementation is compiled against; the `libc`
        // binding names the capped symbol, so the alias is declared here.
        #[cfg(target_vendor = "apple")]
        unsafe extern "C" {
            #[link_name = "getgroups$DARWIN_EXTSN"]
            fn getgroups_unlimited(
                gidsetsize: libc::c_int,
                grouplist: *mut libc::gid_t,
            ) -> libc::c_int;
        }

        /// The group list, sized by the count the kernel reports first.
        #[cfg(target_vendor = "apple")]
        fn host_getgroups() -> std::io::Result<Vec<libc::gid_t>> {
            let count = unsafe { getgroups_unlimited(0, std::ptr::null_mut()) };
            if count < 0 {
                return Err(std::io::Error::last_os_error());
            }
            let mut groups = Vec::<libc::gid_t>::with_capacity(count as usize);
            let filled = unsafe { getgroups_unlimited(count, groups.as_mut_ptr()) };
            if filled < 0 {
                return Err(std::io::Error::last_os_error());
            }
            // A `gidsetsize` of 0 asks for the count instead of the list, so
            // a process that was in no groups at the first call and is in
            // some by the second gets back a count with nothing written.
            let filled = (filled as usize).min(groups.capacity());
            unsafe { groups.set_len(filled) };
            Ok(groups)
        }

        /// Elsewhere one symbol answers the question and `host_env` names it.
        #[cfg(not(target_vendor = "apple"))]
        fn host_getgroups() -> std::io::Result<Vec<libc::gid_t>> {
            host_posix::getgroups()
        }

        /// Replace the supplementary group list.  The `host_env` binding for
        /// this call is gated off on the apple targets, which do have it.
        #[cfg(target_vendor = "apple")]
        fn host_setgroups(groups: &[libc::gid_t]) -> std::io::Result<()> {
            let ret = unsafe { libc::setgroups(groups.len() as _, groups.as_ptr()) };
            if ret != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        }

        #[cfg(not(target_vendor = "apple"))]
        fn host_setgroups(groups: &[libc::gid_t]) -> std::io::Result<()> {
            host_posix::setgroups_raw(groups)
        }

        // os.getgroups() -> list[int]
        crate::module_ns_store(
            ns,
            "getgroups",
            crate::make_builtin_function_with_arity(
                "getgroups",
                |_| {
                    let gs = host_getgroups().map_err(|e| io_err(e, ""))?;
                    let items: Vec<_> = gs
                        .into_iter()
                        .map(|g| pyre_object::w_int_new(g as i64))
                        .collect();
                    Ok(pyre_object::w_list_new(items))
                },
                0,
            ),
        );

        // os.setgroups(list) -> None
        crate::module_ns_store(
            ns,
            "setgroups",
            crate::make_builtin_function_with_arity(
                "setgroups",
                |args| {
                    let Some(&w_list) = args.first() else {
                        return Err(crate::PyError::type_error("setgroups() requires 1 argument"));
                    };
                    // interp_posix.py:1053-1064 — the list is unpacked as any
                    // iterable and each element read with `c_uid_t_w`, which is
                    // what lets -1 name `(gid_t)-1` instead of being refused.
                    let items = crate::builtins::collect_iterable(w_list)?;
                    // `c_uid_t_w` reaches `__index__` for a non-int entry, so
                    // converting one entry can collect and move the entries not
                    // yet converted -- `collect_iterable` hands back a plain
                    // vector, its own roots already dropped.  Publish the
                    // sequence once and read each entry back per iteration.
                    let _seq_roots = pyre_object::gc_roots::push_roots();
                    let items_base = pyre_object::gc_roots::pin_roots(&items);
                    let mut groups: Vec<libc::gid_t> = Vec::with_capacity(items.len());
                    for offset in 0..items.len() {
                        let w_gid = pyre_object::gc_roots::shadow_stack_get(items_base + offset);
                        groups.push(crate::baseobjspace::c_uid_t_w(w_gid)?);
                    }
                    host_setgroups(&groups).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.sched_get_priority_max(policy) -> int
        crate::module_ns_store(
            ns,
            "sched_get_priority_max",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_max",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_max() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:2977 `@unwrap_spec(policy=int)`.
                    let policy = crate::baseobjspace::int_w(args[0])? as i32;
                    // interp_posix.py `sched_get_priority_max`: retry on EINTR.
                    let m = loop {
                        match host_posix::sched_get_priority_max(policy) {
                            Ok(m) => break m,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    };
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // os.sched_get_priority_min(policy) -> int
        crate::module_ns_store(
            ns,
            "sched_get_priority_min",
            crate::make_builtin_function_with_arity(
                "sched_get_priority_min",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "sched_get_priority_min() requires 1 argument",
                        ));
                    }
                    // interp_posix.py:2991 `@unwrap_spec(policy=int)`.
                    let policy = crate::baseobjspace::int_w(args[0])? as i32;
                    // interp_posix.py `sched_get_priority_min`: retry on EINTR.
                    let m = loop {
                        match host_posix::sched_get_priority_min(policy) {
                            Ok(m) => break m,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    };
                    Ok(pyre_object::w_int_new(m as i64))
                },
                1,
            ),
        );

        // The scheduling-policy group `moduledef.py:168-174` publishes as one —
        // the two getters, the two setters and the `sched_param` type they
        // exchange — plus `sched_rr_get_interval`, which `moduledef.py:166-167`
        // gates on its own but which the same libcs carry. The setters are left
        // out where the libc is musl, which declares neither.
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd"
        ))]
        {
            crate::module_ns_store(ns, "sched_param", sched_param_seq_type());

            // os.sched_rr_get_interval(pid) -> seconds
            //
            // host_env wraps none of this one, so the call is made here — which
            // is why it is absent from a sandbox build: `host_seam::sys`
            // re-exports no syscall function, and the name is served there by
            // the raising stub registered at the end of this module instead.
            #[cfg(not(feature = "sandbox"))]
            crate::module_ns_store(
                ns,
                "sched_rr_get_interval",
                crate::make_builtin_function_with_arity(
                    "sched_rr_get_interval",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_rr_get_interval() requires 1 argument",
                            ));
                        }
                        // interp_posix.py:3061 `@unwrap_spec(pid=int)`; the
                        // timespec the call fills is answered as one float
                        // (`rposix.py:2525`).
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        let mut interval: libc::timespec =
                            unsafe { core::mem::zeroed::<libc::timespec>() };
                        // interp_posix.py `sched_rr_get_interval`: retry on EINTR.
                        loop {
                            let (ret, errno) =
                                crate::module::thread::call_external_function(|| unsafe {
                                    libc::sched_rr_get_interval(pid, &mut interval)
                                });
                            if ret >= 0 {
                                break;
                            }
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(errno),
                                |e| io_err(e, ""),
                            )?;
                        }
                        Ok(pyre_object::w_float_new(
                            interval.tv_sec as f64 + 1e-9 * interval.tv_nsec as f64,
                        ))
                    },
                    1,
                ),
            );

            // os.sched_getscheduler(pid) -> policy
            crate::module_ns_store(
                ns,
                "sched_getscheduler",
                crate::make_builtin_function_with_arity(
                    "sched_getscheduler",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_getscheduler() requires 1 argument",
                            ));
                        }
                        // interp_posix.py:3073 `@unwrap_spec(pid=int)`.
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        // interp_posix.py `sched_getscheduler`: retry on EINTR.
                        let policy = loop {
                            match host_posix::sched_getscheduler(pid) {
                                Ok(policy) => break policy,
                                Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                            }
                        };
                        Ok(pyre_object::w_int_new(policy as i64))
                    },
                    1,
                ),
            );

            // os.sched_getparam(pid) -> sched_param
            crate::module_ns_store(
                ns,
                "sched_getparam",
                crate::make_builtin_function_with_arity(
                    "sched_getparam",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_getparam() requires 1 argument",
                            ));
                        }
                        // interp_posix.py:3103 `@unwrap_spec(pid=int)`; the
                        // priority the call fills in is handed back wrapped in
                        // the type, not bare (`interp_posix.py:3113`).
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        // interp_posix.py `sched_getparam`: retry on EINTR.
                        let param = loop {
                            match host_posix::sched_getparam(pid) {
                                Ok(param) => break param,
                                Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                            }
                        };
                        Ok(crate::_structseq::new_instance(
                            sched_param_seq_type(),
                            vec![pyre_object::w_int_new(param.sched_priority as i64)],
                        ))
                    },
                    1,
                ),
            );

            // Both setters answer None. `interp_posix.py:3097`/`:3131` hand
            // back the raw `handle_posix_error` result instead, which is 0 on
            // every success and which `os.sched_setparam` does not publish.
            #[cfg(not(target_env = "musl"))]
            {
                // os.sched_setscheduler(pid, policy, param)
                crate::module_ns_store(
                    ns,
                    "sched_setscheduler",
                    crate::make_builtin_function_with_arity(
                        "sched_setscheduler",
                        |args| {
                            if args.len() < 3 {
                                return Err(crate::PyError::type_error(
                                    "sched_setscheduler() requires 3 arguments",
                                ));
                            }
                            // interp_posix.py `@unwrap_spec(pid=int, policy=int)`.
                            let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                            let policy = crate::baseobjspace::int_w(args[1])? as libc::c_int;
                            let priority = sched_priority_w(args[2])?;
                            let mut param: libc::sched_param =
                                unsafe { core::mem::zeroed::<libc::sched_param>() };
                            param.sched_priority = priority;
                            // interp_posix.py `sched_setscheduler`: retry on EINTR.
                            loop {
                                match host_posix::sched_setscheduler(pid, policy, &param) {
                                    Ok(_) => break,
                                    Err(e) => {
                                        crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?
                                    }
                                }
                            }
                            Ok(pyre_object::w_none())
                        },
                        3,
                    ),
                );

                // os.sched_setparam(pid, param)
                crate::module_ns_store(
                    ns,
                    "sched_setparam",
                    crate::make_builtin_function_with_arity(
                        "sched_setparam",
                        |args| {
                            if args.len() < 2 {
                                return Err(crate::PyError::type_error(
                                    "sched_setparam() requires 2 arguments",
                                ));
                            }
                            // interp_posix.py:3117 `@unwrap_spec(pid=int)`.
                            let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                            let priority = sched_priority_w(args[1])?;
                            let mut param: libc::sched_param =
                                unsafe { core::mem::zeroed::<libc::sched_param>() };
                            param.sched_priority = priority;
                            // interp_posix.py `sched_setparam`: retry on EINTR.
                            loop {
                                match host_posix::sched_setparam(pid, &param) {
                                    Ok(_) => break,
                                    Err(e) => {
                                        crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?
                                    }
                                }
                            }
                            Ok(pyre_object::w_none())
                        },
                        2,
                    ),
                );
            }
        }

        // os.sched_getaffinity(pid) / os.sched_setaffinity(pid, mask) — the CPU
        // mask, which `moduledef.py` does not publish and `rposix` does not
        // wrap, so both are written against the host header rather than ported.
        //
        // The mask is the fixed `cpu_set_t`, `CPU_SETSIZE` CPUs wide: the libc
        // crate exposes no `CPU_ALLOC`. A CPU number at or past that width is
        // refused with the EINVAL the kernel would answer for it, and a host
        // with more CPUs than that gets the kernel's own EINVAL out of
        // `sched_getaffinity` rather than a silently truncated mask.
        //
        // Both name libc directly, so neither is compiled into a sandbox build;
        // `host_seam::sys` re-exports no syscall and the stubs at the end of
        // this module serve the names there.
        #[cfg(all(
            not(feature = "sandbox"),
            any(target_os = "linux", target_os = "android")
        ))]
        {
            crate::module_ns_store(
                ns,
                "sched_getaffinity",
                crate::make_builtin_function_with_arity(
                    "sched_getaffinity",
                    |args| {
                        if args.is_empty() {
                            return Err(crate::PyError::type_error(
                                "sched_getaffinity() requires 1 argument",
                            ));
                        }
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        let mut mask: libc::cpu_set_t =
                            unsafe { core::mem::zeroed::<libc::cpu_set_t>() };
                        unsafe { libc::CPU_ZERO(&mut mask) };
                        let res = unsafe {
                            libc::sched_getaffinity(
                                pid,
                                core::mem::size_of::<libc::cpu_set_t>(),
                                &mut mask,
                            )
                        };
                        if res == -1 {
                            return Err(io_err(std::io::Error::last_os_error(), ""));
                        }
                        let items: Vec<_> = (0..libc::CPU_SETSIZE as usize)
                            .filter(|&cpu| unsafe { libc::CPU_ISSET(cpu, &mask) })
                            .map(|cpu| pyre_object::w_int_new(cpu as i64))
                            .collect();
                        Ok(pyre_object::w_set_from_items(&items))
                    },
                    1,
                ),
            );

            crate::module_ns_store(
                ns,
                "sched_setaffinity",
                crate::make_builtin_function_with_arity(
                    "sched_setaffinity",
                    |args| {
                        if args.len() < 2 {
                            return Err(crate::PyError::type_error(
                                "sched_setaffinity() requires 2 arguments",
                            ));
                        }
                        let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                        let items = crate::builtins::collect_iterable(args[1])?;
                        let int_type =
                            crate::typedef::gettypeobject(&pyre_object::pyobject::INT_TYPE);
                        let mut mask: libc::cpu_set_t =
                            unsafe { core::mem::zeroed::<libc::cpu_set_t>() };
                        unsafe { libc::CPU_ZERO(&mut mask) };
                        for item in items {
                            if !crate::baseobjspace::isinstance(item, int_type)? {
                                return Err(crate::PyError::type_error(format!(
                                    "expected an iterator of ints, but iterator yielded <class '{}'>",
                                    crate::type_methods::arg_type_name(item)
                                )));
                            }
                            let cpu = crate::baseobjspace::int_w(item)?;
                            if cpu < 0 {
                                return Err(crate::PyError::value_error("negative CPU number"));
                            }
                            if cpu > libc::c_int::MAX as i64 {
                                return Err(crate::PyError::overflow_error("CPU number too large"));
                            }
                            if cpu >= libc::CPU_SETSIZE as i64 {
                                return Err(io_err(
                                    std::io::Error::from_raw_os_error(libc::EINVAL),
                                    "",
                                ));
                            }
                            unsafe { libc::CPU_SET(cpu as usize, &mut mask) };
                        }
                        let res = unsafe {
                            libc::sched_setaffinity(
                                pid,
                                core::mem::size_of::<libc::cpu_set_t>(),
                                &mask,
                            )
                        };
                        if res == -1 {
                            return Err(io_err(std::io::Error::last_os_error(), ""));
                        }
                        Ok(pyre_object::w_none())
                    },
                    2,
                ),
            );
        }

        // os.sync()
        #[cfg(not(any(target_os = "redox", target_os = "android")))]
        crate::module_ns_store(
            ns,
            "sync",
            crate::make_builtin_function_with_arity(
                "sync",
                |_| {
                    host_posix::sync();
                    Ok(pyre_object::w_none())
                },
                0,
            ),
        );

        // os.chdir(path)
        crate::module_ns_store(
            ns,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    // interp_posix.py dispatches `rposix.chdir` with
                    // `allow_fd_fn=os.fchdir` when `rposix.HAVE_FCHDIR`.
                    let path =
                        crate::gateway::fsencode_path_or_fd_w(args[0], "chdir", HAVE_FCHDIR)?;
                    if path.as_fd != -1 {
                        host_posix::fchdir(path.as_fd).map_err(|e| io_err(e, ""))?;
                        return Ok(pyre_object::w_none());
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    host_posix::chdir(&c_path)
                        .map_err(|e| errno_err_with_filename(e as i32, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fchdir(fd)
        crate::module_ns_store(
            ns,
            "fchdir",
            crate::make_builtin_function_with_arity(
                "fchdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fchdir() requires 1 argument"));
                    }
                    // interp_posix.py `fchdir(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`, which takes an int or
                    // anything exposing `fileno()`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    // interp_posix.py `fchdir`: retry on EINTR.
                    loop {
                        match host_posix::fchdir(fd) {
                            Ok(()) => break,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // PyPy's `_run_forking_function` enters the callback lifecycle
        // immediately. CPython 3.14 checks finalization first, so a refused
        // fork takes no callback lock and signals no thread.
        fn guard_fork_finalization() -> Result<(), crate::PyError> {
            if !crate::module::thread::is_finalizing() {
                return Ok(());
            }
            Err(crate::builtins::finalization_error(Some(
                "can't fork at interpreter shutdown",
            )))
        }

        // os.fork() -> child pid in parent, 0 in child
        crate::module_ns_store(
            ns,
            "fork",
            crate::make_builtin_function_with_arity(
                "fork",
                |_| {
                    guard_fork_finalization()?;
                    if majit_gc::gc_sync::registered_threads() > 1 {
                        crate::warn::warn_deprecation(
                            "This process is multi-threaded, use of fork() may lead to deadlocks",
                        )?;
                    }
                    let blocked = crate::module::thread::before_external_block();
                    let fork_serial = FORK_SERIALIZER
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    drop(blocked);
                    run_fork_callbacks("before");
                    // pypy/module/imp/moduledef.py:45-47 registers the import
                    // lock's acquire/release/reinit trio in the same ordered
                    // interpreter-level hook lists that contain the app-level
                    // callback dispatcher.  Holding it across the host fork
                    // prevents a child from inheriting a partially initialized
                    // sys.modules entry from another thread.
                    crate::module::imp::interp_imp::before_fork();
                    // A free-threaded fork must snapshot native/Python lock
                    // state while every other mutator is parked.  Enter
                    // through the collector's full request path so no GC
                    // operation can overlap the STW window.
                    let mut fork_result = None;
                    majit_gc::gc_sync::request_stw(|_| {
                        fork_result = Some(host_posix::fork());
                    });
                    match fork_result.expect("fork STW closure must run") {
                        Ok(0) => {
                            crate::module::thread::after_fork_child();
                            run_fork_callbacks("child");
                            crate::module::imp::interp_imp::after_fork_child();
                            // CPython's refcounting drops the replaced
                            // `_MainThread` before os.fork() returns, so its
                            // weakref has disappeared from `_dangling` when
                            // child Python code next runs.  A tracing GC needs
                            // an explicit reachability pass for the same
                            // observable result.  Defer a non-moving old-gen
                            // pass to the next bytecode boundary: collecting
                            // here would run while this native builtin still
                            // owns unregistered Rust-stack temporaries, and a
                            // moving full collection would be unsafe.
                            pyre_object::gc_interp::request_oldgen_collection();
                            drop(fork_serial);
                            Ok(pyre_object::w_int_new(0))
                        }
                        Ok(pid) => {
                            run_fork_callbacks("parent");
                            crate::module::imp::interp_imp::after_fork_parent()?;
                            drop(fork_serial);
                            Ok(pyre_object::w_int_new(pid as i64))
                        }
                        Err(error) => {
                            run_fork_callbacks("parent");
                            // interp_posix.py:1570-1575 keeps the original
                            // fork OSError if a parent hook also fails.
                            let _ = crate::module::imp::interp_imp::after_fork_parent();
                            drop(fork_serial);
                            Err(io_err(error, ""))
                        }
                    }
                },
                0,
            ),
        );

        // PyPy interp_posix.py:1703-1706 delegates to
        // `_run_forking_function(space, "P")`, so forkpty must use the same
        // before/parent/child hook and thread-reinitialization lifecycle as
        // fork above while returning the master descriptor as its second item.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "forkpty",
            crate::make_builtin_function_with_arity(
                "forkpty",
                |_| {
                    guard_fork_finalization()?;
                    if majit_gc::gc_sync::registered_threads() > 1 {
                        crate::warn::warn_deprecation(
                            "This process is multi-threaded, use of forkpty() may lead to deadlocks",
                        )?;
                    }
                    let blocked = crate::module::thread::before_external_block();
                    let fork_serial = FORK_SERIALIZER
                        .lock()
                        .unwrap_or_else(|poison| poison.into_inner());
                    drop(blocked);
                    run_fork_callbacks("before");
                    let mut master_fd = -1;
                    let mut fork_result = None;
                    majit_gc::gc_sync::request_stw(|_| {
                        let pid = unsafe {
                            libc::forkpty(
                                &mut master_fd,
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                                std::ptr::null_mut(),
                            )
                        };
                        fork_result = Some(if pid == -1 {
                            Err(std::io::Error::last_os_error())
                        } else {
                            Ok(pid)
                        });
                    });
                    match fork_result.expect("forkpty STW closure must run") {
                        Ok(0) => {
                            crate::module::thread::after_fork_child();
                            run_fork_callbacks("child");
                            drop(fork_serial);
                            Ok(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new(0),
                                pyre_object::w_int_new(master_fd as i64),
                            ]))
                        }
                        Ok(pid) => {
                            run_fork_callbacks("parent");
                            drop(fork_serial);
                            Ok(pyre_object::w_tuple_new(vec![
                                pyre_object::w_int_new(pid as i64),
                                pyre_object::w_int_new(master_fd as i64),
                            ]))
                        }
                        Err(error) => {
                            run_fork_callbacks("parent");
                            drop(fork_serial);
                            Err(io_err(error, ""))
                        }
                    }
                },
                0,
            ),
        );

        // os.getppid() -> int
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| Ok(pyre_object::w_int_new(unsafe { libc::getppid() } as i64)),
                0,
            ),
        );

        // PyPy interp_posix.getsid(pid) -> rposix.getsid(pid).
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "getsid",
            crate::make_builtin_function_with_arity(
                "getsid",
                |args| {
                    let pid = match args.first() {
                        // interp_posix.py `@unwrap_spec(pid=c_int)`.
                        Some(&obj) => crate::baseobjspace::c_int_w(obj)? as libc::pid_t,
                        None => {
                            return Err(crate::PyError::type_error("getsid() requires 1 argument"));
                        }
                    };
                    let sid = unsafe { libc::getsid(pid) };
                    if sid == -1 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_int_new(sid as i64))
                },
                1,
            ),
        );

        // `interp_posix.py:2167-2172` — the caller's own group, which cannot
        // fail and so is not checked.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "getpgrp",
            crate::make_builtin_function_with_arity(
                "getpgrp",
                |_| Ok(pyre_object::w_int_new(unsafe { libc::getpgrp() } as i64)),
                0,
            ),
        );

        // `interp_posix.py:2201-2210` — another process's group, which can be
        // one this process may not ask about.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "getpgid",
            crate::make_builtin_function_with_arity(
                "getpgid",
                |args| {
                    let pid = match args.first() {
                        // interp_posix.py `@unwrap_spec(pid=c_int)`.
                        Some(&obj) => crate::baseobjspace::c_int_w(obj)? as libc::pid_t,
                        None => {
                            return Err(crate::PyError::type_error(
                                "getpgid() requires 1 argument",
                            ));
                        }
                    };
                    let pgid = unsafe { libc::getpgid(pid) };
                    if pgid == -1 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            ),
        );

        // PyPy's six user/group ID setters share the `c_uid_t` conversion and
        // report syscall errors without retrying.
        #[cfg(not(feature = "sandbox"))]
        fn set_one_id(
            args: &[PyObjectRef],
            name: &str,
            setter: fn(u32) -> libc::c_int,
        ) -> Result<PyObjectRef, crate::PyError> {
            let id = match args.first() {
                Some(&obj) => crate::baseobjspace::c_uid_t_w(obj)?,
                None => {
                    return Err(crate::PyError::type_error(format!(
                        "{name}() requires 1 argument"
                    )));
                }
            };
            if setter(id) == -1 {
                return Err(io_err(std::io::Error::last_os_error(), ""));
            }
            Ok(pyre_object::w_none())
        }

        #[cfg(not(feature = "sandbox"))]
        fn set_two_ids(
            args: &[PyObjectRef],
            name: &str,
            setter: fn(u32, u32) -> libc::c_int,
        ) -> Result<PyObjectRef, crate::PyError> {
            if args.len() < 2 {
                return Err(crate::PyError::type_error(format!(
                    "{name}() requires 2 arguments"
                )));
            }
            let first = crate::baseobjspace::c_uid_t_w(args[0])?;
            let second = crate::baseobjspace::c_uid_t_w(args[1])?;
            if setter(first, second) == -1 {
                return Err(io_err(std::io::Error::last_os_error(), ""));
            }
            Ok(pyre_object::w_none())
        }

        #[cfg(not(feature = "sandbox"))]
        fn setuid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "setuid", |uid| unsafe {
                libc::setuid(uid as libc::uid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn seteuid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "seteuid", |euid| unsafe {
                libc::seteuid(euid as libc::uid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setgid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "setgid", |gid| unsafe {
                libc::setgid(gid as libc::gid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setegid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_one_id(args, "setegid", |egid| unsafe {
                libc::setegid(egid as libc::gid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setreuid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_two_ids(args, "setreuid", |ruid, euid| unsafe {
                libc::setreuid(ruid as libc::uid_t, euid as libc::uid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        fn setregid(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            set_two_ids(args, "setregid", |rgid, egid| unsafe {
                libc::setregid(rgid as libc::gid_t, egid as libc::gid_t)
            })
        }

        #[cfg(not(feature = "sandbox"))]
        for (name, function, arity) in [
            ("setuid", setuid as crate::gateway::BuiltinCodeFn, 1),
            ("seteuid", seteuid as crate::gateway::BuiltinCodeFn, 1),
            ("setgid", setgid as crate::gateway::BuiltinCodeFn, 1),
            ("setegid", setegid as crate::gateway::BuiltinCodeFn, 1),
            ("setreuid", setreuid as crate::gateway::BuiltinCodeFn, 2),
            ("setregid", setregid as crate::gateway::BuiltinCodeFn, 2),
        ] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function_with_arity(name, function, arity),
            );
        }

        // `interp_posix.py:2603-2608` — the controlling terminal's name, which
        // `rposix.py:1724-1728` reads by handing the call a null pointer and
        // taking the static buffer it answers with. It is a filename, so it is
        // decoded the way every other name from the host is.
        #[cfg(not(feature = "sandbox"))]
        {
            // `<stdio.h>` declares `ctermid` on every POSIX host; the `libc`
            // crate carries it for a few of them.
            #[cfg(not(any(
                target_os = "linux",
                target_os = "aix",
                target_os = "haiku",
                target_os = "hurd",
                target_os = "nto",
            )))]
            unsafe extern "C" {
                fn ctermid(s: *mut libc::c_char) -> *mut libc::c_char;
            }
            #[cfg(any(
                target_os = "linux",
                target_os = "aix",
                target_os = "haiku",
                target_os = "hurd",
                target_os = "nto",
            ))]
            use libc::ctermid;

            crate::module_ns_store(
                ns,
                "ctermid",
                crate::make_builtin_function_with_arity(
                    "ctermid",
                    |_| {
                        let name = unsafe { ctermid(std::ptr::null_mut()) };
                        if name.is_null() {
                            return Err(io_err(std::io::Error::last_os_error(), ""));
                        }
                        let bytes = unsafe { std::ffi::CStr::from_ptr(name) };
                        Ok(crate::gateway::fsdecode_filename_bytes(bytes.to_bytes()))
                    },
                    0,
                ),
            );
        }

        // os.waitpid(pid, options) -> (pid, status)
        crate::module_ns_store(
            ns,
            "waitpid",
            crate::make_builtin_function_with_arity(
                "waitpid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("waitpid() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pid=c_int, options=c_int)`.
                    let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                    let options = crate::baseobjspace::c_int_w(args[1])?;
                    let mut status: i32 = 0;
                    // interp_posix.py `waitpid`: retry on EINTR.
                    let res = loop {
                        let result = {
                            let _blocked = crate::module::thread::before_external_block();
                            host_posix::waitpid(pid, &mut status, options)
                        };
                        match result {
                            Ok(res) => break res,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    };
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(res as i64),
                        pyre_object::w_int_new(status as i64),
                    ]))
                },
                2,
            ),
        );

        // os.wait() -> (pid, status)
        crate::module_ns_store(
            ns,
            "wait",
            crate::make_builtin_function_with_arity(
                "wait",
                |_| {
                    let mut status: i32 = 0;
                    // `app_posix.wait` is `posix.waitpid(-1, 0)`, so it inherits
                    // that call's `eintr_retry=True` rather than surfacing the
                    // interruption of its own.
                    let res = loop {
                        let result = {
                            let _blocked = crate::module::thread::before_external_block();
                            host_posix::waitpid(-1, &mut status, 0)
                        };
                        match result {
                            Ok(res) => break res,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    };
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(res as i64),
                        pyre_object::w_int_new(status as i64),
                    ]))
                },
                0,
            ),
        );

        // os._exit(code) — immediate process exit, no cleanup.
        crate::module_ns_store(
            ns,
            "_exit",
            crate::make_builtin_function_with_arity(
                "_exit",
                |args| {
                    let code = match args.first() {
                        // interp_posix.py `@unwrap_spec(status=c_int)`.
                        Some(&o) => crate::baseobjspace::c_int_w(o)?,
                        None => {
                            return Err(crate::PyError::type_error("_exit() requires 1 argument"));
                        }
                    };
                    rustpython_host_env::os::exit(code)
                },
                1,
            ),
        );

        // Wait-status decoding macros (WIFEXITED/WEXITSTATUS/...): override
        // the noop stubs registered above with the libc bit-math.
        macro_rules! reg_wstatus {
            ($name:literal, |$s:ident| $body:expr) => {
                crate::module_ns_store(
                    ns,
                    $name,
                    crate::make_builtin_function_with_arity(
                        $name,
                        |args| {
                            let $s = match args.first() {
                                // interp_posix.py `declare_new_w_star`
                                // types every wait macro `@unwrap_spec(status=c_int)`.
                                Some(&o) => crate::baseobjspace::c_int_w(o)?,
                                None => {
                                    return Err(crate::PyError::type_error(concat!(
                                        $name,
                                        "() requires 1 argument"
                                    )));
                                }
                            };
                            Ok($body)
                        },
                        1,
                    ),
                );
            };
        }
        reg_wstatus!("WIFEXITED", |s| pyre_object::w_bool_from(libc::WIFEXITED(
            s
        )));
        reg_wstatus!("WEXITSTATUS", |s| pyre_object::w_int_new(
            libc::WEXITSTATUS(s) as i64
        ));
        reg_wstatus!("WIFSIGNALED", |s| pyre_object::w_bool_from(
            libc::WIFSIGNALED(s)
        ));
        reg_wstatus!("WTERMSIG", |s| pyre_object::w_int_new(
            libc::WTERMSIG(s) as i64
        ));
        reg_wstatus!("WIFSTOPPED", |s| pyre_object::w_bool_from(
            libc::WIFSTOPPED(s)
        ));
        reg_wstatus!("WSTOPSIG", |s| pyre_object::w_int_new(
            libc::WSTOPSIG(s) as i64
        ));

        // Wait option flags — override the `0` placeholders registered above
        // with their real libc values (os.WNOHANG must be non-zero for
        // subprocess.poll()).
        crate::module_ns_store(ns, "WNOHANG", pyre_object::w_int_new(libc::WNOHANG as i64));
        crate::module_ns_store(
            ns,
            "WUNTRACED",
            pyre_object::w_int_new(libc::WUNTRACED as i64),
        );
        crate::module_ns_store(
            ns,
            "WCONTINUED",
            pyre_object::w_int_new(libc::WCONTINUED as i64),
        );
        // The states `waitid` is asked to report on. They were registered above
        // as calls answering `None`, which is neither the number nor a name a
        // caller can tell apart from one.
        for (name, val) in [
            ("WEXITED", libc::WEXITED as i64),
            ("WSTOPPED", libc::WSTOPPED as i64),
            ("WNOWAIT", libc::WNOWAIT as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }
        // Which process `waitid` is asked about, and what `si_code` says
        // happened to it once it answers.
        for (name, val) in [
            ("P_ALL", libc::P_ALL as i64),
            ("P_PID", libc::P_PID as i64),
            ("P_PGID", libc::P_PGID as i64),
            ("CLD_EXITED", libc::CLD_EXITED as i64),
            ("CLD_KILLED", libc::CLD_KILLED as i64),
            ("CLD_DUMPED", libc::CLD_DUMPED as i64),
            ("CLD_TRAPPED", libc::CLD_TRAPPED as i64),
            ("CLD_STOPPED", libc::CLD_STOPPED as i64),
            ("CLD_CONTINUED", libc::CLD_CONTINUED as i64),
        ] {
            crate::module_ns_store(ns, name, pyre_object::w_int_new(val));
        }

        // os.waitid(idtype, id, options) -> waitid_result | None
        //
        // `os_waitid_impl` — the call reports on a child without reaping it
        // when WNOWAIT is among the options, which is what separates it from
        // `waitpid`. A zero `si_pid` means the options asked about a state no
        // child is in (WNOHANG with nothing to report), and that is `None`
        // rather than a result whose every field is zero.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(ns, "waitid_result", waitid_result_seq_type());
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "waitid",
            crate::make_builtin_function_with_arity(
                "waitid",
                |args| {
                    if args.len() != 3 {
                        return Err(crate::PyError::type_error(format!(
                            "waitid expected 3 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let idtype = crate::baseobjspace::c_int_w(args[0])? as libc::idtype_t;
                    let id = crate::baseobjspace::int_w(crate::baseobjspace::space_index(args[1])?)
                        .map_err(|_| {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C long",
                            )
                        })? as libc::id_t;
                    let options = crate::baseobjspace::c_int_w(args[2])?;
                    // `si.si_pid = 0` before the call: the field is what the
                    // "nothing to report" answer is read out of, and a call
                    // that reports nothing does not write it.
                    let mut si: libc::siginfo_t = unsafe { std::mem::zeroed() };
                    loop {
                        let (ret, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::waitid(idtype, id, &mut si, options)
                            });
                        if ret >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    // The three that live in the union `siginfo_t` keeps its
                    // process fields in are read through the accessors that
                    // name which arm is meant; `si_signo` and `si_code` are
                    // outside it.
                    let (pid, uid, status) = unsafe { (si.si_pid(), si.si_uid(), si.si_status()) };
                    if pid == 0 {
                        return Ok(pyre_object::w_none());
                    }
                    Ok(crate::_structseq::new_instance(
                        waitid_result_seq_type(),
                        vec![
                            pyre_object::w_int_new(pid as i64),
                            pyre_object::w_int_new(uid as i64),
                            pyre_object::w_int_new(si.si_signo as i64),
                            pyre_object::w_int_new(status as i64),
                            pyre_object::w_int_new(si.si_code as i64),
                        ],
                    ))
                },
                3,
            ),
        );

        // os.dup(fd) -> new_fd
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    // `_Py_dup` makes the copy close-on-exec in the same call
                    // that makes it, so no fork in between inherits it.
                    let n = crate::builtins::crt_call!(libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0));
                    if n < 0 {
                        return Err(errno_err(crate::builtins::crt_errno(), ""));
                    }
                    Ok(pyre_object::w_int_new(n as i64))
                },
                1,
            ),
        );

        // os.dup2(fd, fd2, inheritable=True) -> fd2
        //
        // Carries a `Signature`, so `inheritable` binds by name.  Registered
        // raw it did not: the trailing `__pyre_kw__` marker dict was never
        // split off the argument slice, so it landed in the third positional
        // slot and read truthy, and `dup2(fd, fd2, inheritable=False)`
        // returned an *inheritable* descriptor that an exec would carry.
        //
        // The arguments stay `PyObjectRef` and are unwrapped in the body: the
        // macro's bare `i32` binding is a raw `w_int_get_value` cast, which
        // would read a non-int argument's payload instead of reporting it.
        #[cfg(not(feature = "sandbox"))]
        #[crate::pyre_function]
        fn dup2(
            fd: pyre_object::PyObjectRef,
            fd2: pyre_object::PyObjectRef,
            inheritable: Option<pyre_object::PyObjectRef>,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            // interp_posix.py `@unwrap_spec(fd=c_int, fd2=c_int, inheritable=bool)`.
            let fd = crate::baseobjspace::c_int_w(fd)?;
            let fd2 = crate::baseobjspace::c_int_w(fd2)?;
            let inheritable = match inheritable {
                Some(w) => crate::baseobjspace::is_true(w)?,
                None => true,
            };
            // `os_dup2_impl` asks for a non-inheritable duplicate through
            // `dup3` where it exists, so no window opens in which the new
            // descriptor is inheritable and an exec could carry it.  The
            // inheritable case keeps plain `dup2`, which is also the one
            // that tolerates `fd == fd2`.
            let n = if inheritable {
                crate::builtins::crt_call!(libc::dup2(fd, fd2))
            } else {
                #[cfg(any(target_os = "android", target_os = "linux", target_os = "freebsd"))]
                {
                    crate::builtins::crt_call!(libc::dup3(fd, fd2, libc::O_CLOEXEC))
                }
                #[cfg(not(any(
                    target_os = "android",
                    target_os = "linux",
                    target_os = "freebsd"
                )))]
                {
                    let n = crate::builtins::crt_call!(libc::dup2(fd, fd2));
                    if n >= 0 {
                        use std::os::fd::BorrowedFd;
                        let bfd = unsafe { BorrowedFd::borrow_raw(n) };
                        host_posix::set_inheritable(bfd, false).map_err(|e| io_err(e, ""))?;
                    }
                    n
                }
            };
            if n < 0 {
                return Err(errno_err(crate::builtins::crt_errno(), ""));
            }
            Ok(pyre_object::w_int_new(n as i64))
        }

        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "dup2",
            crate::make_builtin_function_with_arity_and_maybe_sig(
                "dup2",
                dup2,
                dup2_pyre_arity(),
                dup2_pyre_sig(),
            ),
        );

        // os.fsync(fd)
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    // interp_posix.py `fsync(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    // interp_posix.py `fsync`: retry on EINTR.
                    loop {
                        let (r, errno) = crate::module::thread::call_external_function(|| unsafe {
                            libc::fsync(fd)
                        });
                        if r >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.fdatasync(fd) — falls back to fsync on macOS, which has no
        // fdatasync syscall but exposes the same semantics through fsync.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "fdatasync",
            crate::make_builtin_function_with_arity(
                "fdatasync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "fdatasync() requires 1 argument",
                        ));
                    }
                    // interp_posix.py `fdatasync(space, w_fd)` unwraps
                    // through `space.c_filedescriptor_w`.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    // interp_posix.py `fdatasync`: retry on EINTR.
                    loop {
                        let (r, errno) = crate::module::thread::call_external_function(|| unsafe {
                            #[cfg(any(target_os = "linux", target_os = "android"))]
                            {
                                libc::fdatasync(fd)
                            }
                            #[cfg(not(any(target_os = "linux", target_os = "android")))]
                            {
                                libc::fsync(fd)
                            }
                        });
                        if r >= 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // interp_posix.py:407-412: retry EINTR, propagate every other OSError.
        // The retry is `eintr_retry=True`, which runs the pending Python signal
        // handlers before going back to the call — a handler that raises ends
        // the loop there, and one that disarms the timer stops the interruption
        // recurring. Retrying on the bare errno would spin without ever giving
        // that handler a turn.
        //
        // Which filename the caller then reports is its own: `os.ftruncate` was
        // given no name to report, while `os.truncate` names the one it opened.
        #[cfg(all(unix, not(feature = "sandbox")))]
        fn ftruncate_retry(
            fd: libc::c_int,
            length: libc::off_t,
            wrap: impl Fn(i32) -> crate::PyError,
        ) -> Result<(), crate::PyError> {
            loop {
                if crate::builtins::crt_call!(libc::ftruncate(fd, length)) == 0 {
                    return Ok(());
                }
                let errno = crate::builtins::crt_errno();
                crate::builtins::eintr_retry_with(std::io::Error::from_raw_os_error(errno), |e| {
                    wrap(e.raw_os_error().unwrap_or(0))
                })?;
            }
        }

        // os.truncate(path, length) -> None
        //
        // interp_posix.py:414-431 takes a descriptor as it stands and opens a
        // name write-only, truncates whichever it ended up with, and closes
        // only the one it opened itself. The descriptor form is what
        // HAVE_FTRUNCATE advertises through `os.py:149`.
        #[cfg(all(unix, not(feature = "sandbox")))]
        crate::module_ns_store(
            ns,
            "truncate",
            crate::make_builtin_function_with_arity(
                "truncate",
                |args| {
                    if args.len() != 2 {
                        return Err(crate::PyError::type_error(format!(
                            "truncate expected 2 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let path =
                        crate::gateway::fsencode_path_or_fd_w(args[0], "truncate", HAVE_FTRUNCATE)?;
                    let length = truncate_length_w(args[1])?;
                    if path.as_fd != -1 {
                        ftruncate_retry(path.as_fd, length, |e| errno_err(e, ""))?;
                        return Ok(pyre_object::w_none());
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    // `truncate` opens the name through the module's own `open`
                    // (`interp_posix.py:418`), so this is that call and not a
                    // bare syscall: `inheritable=False`, the interpreter
                    // released for the duration — a FIFO with no reader waits
                    // here until another thread opens the other end — and an
                    // interrupted open re-issued after the signal handler has
                    // run rather than reported as `InterruptedError`.
                    let fd = loop {
                        let (fd, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::open(c_path.as_ptr(), libc::O_WRONLY | libc::O_CLOEXEC)
                            });
                        if fd >= 0 {
                            break fd;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| {
                                errno_err_with_filename(
                                    e.raw_os_error().unwrap_or(0),
                                    path.w_path(),
                                )
                            },
                        )?;
                    };
                    let truncated =
                        ftruncate_retry(fd, length, |e| errno_err_with_filename(e, path.w_path()));
                    // `interp_posix.py:429-431` closes the descriptor it opened
                    // in a `finally`, through the module's own `close` — so a
                    // writeback error the close is the first to see is the
                    // caller's, not something `truncate` reports success over.
                    // The truncation's own failure is the one reported when
                    // both fail, which is the order the `finally` gives them.
                    let closed = unsafe { libc::close(fd) };
                    let close_errno = (closed < 0).then(crate::builtins::crt_errno);
                    truncated?;
                    if let Some(errno) = close_errno {
                        return Err(errno_err_with_filename(errno, path.w_path()));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // rpython/rlib/rposix.py `ftruncate(fd, length)` — this must be a
        // real fd mutation whenever HAVE_FTRUNCATE is advertised.  Shared
        // memory sizes its newly-created object through this call before
        // mapping it.
        #[cfg(all(unix, not(feature = "sandbox")))]
        crate::module_ns_store(
            ns,
            "ftruncate",
            crate::make_builtin_function_with_arity(
                "ftruncate",
                |args| {
                    // interp_posix.py `@unwrap_spec(fd=c_int, length=r_longlong)`.
                    // CPython 3.14's clinic gateway accepts `__index__` but not
                    // `__int__`; that newer coercion wins over PyPy's legacy
                    // `gateway_int_w` behavior.
                    if args.len() != 2 {
                        return Err(crate::PyError::type_error(format!(
                            "ftruncate expected 2 arguments, got {}",
                            args.len(),
                        )));
                    }
                    let w_fd = crate::baseobjspace::space_index(args[0])?;
                    let fd_value = crate::baseobjspace::int_w(w_fd).map_err(|err| {
                        if err.kind == crate::PyErrorKind::OverflowError {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C int",
                            )
                        } else {
                            err
                        }
                    })?;
                    let fd = libc::c_int::try_from(fd_value).map_err(|_| {
                        crate::PyError::overflow_error("Python int too large to convert to C int")
                    })?;
                    let length = truncate_length_w(args[1])?;
                    ftruncate_retry(fd, length, |e| errno_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.lockf(fd, cmd, len) -> None
        //
        // `interp_posix.py` — one `lockf` under the `eintr_retry`
        // loop, so a lock that waits and is interrupted goes back to waiting
        // after the signal handler has run rather than surfacing as
        // `InterruptedError`. F_LOCK blocks, so it is put through the call
        // gate the way every other waiting call here is.
        #[cfg(all(unix, not(feature = "sandbox")))]
        crate::module_ns_store(
            ns,
            "lockf",
            crate::make_builtin_function_with_arity(
                "lockf",
                |args| {
                    if args.len() != 3 {
                        return Err(crate::PyError::type_error(format!(
                            "lockf expected 3 arguments, got {}",
                            args.len(),
                        )));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, cmd=c_int,
                    // length=r_longlong)` — the length is an offset, so it is
                    // the same conversion `ftruncate` gives one.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let cmd = crate::baseobjspace::c_int_w(args[1])?;
                    let length = truncate_length_w(args[2])?;
                    loop {
                        let (ret, errno) =
                            crate::module::thread::call_external_function(|| unsafe {
                                libc::lockf(fd, cmd, length)
                            });
                        if ret == 0 {
                            break;
                        }
                        crate::builtins::eintr_retry_with(
                            std::io::Error::from_raw_os_error(errno),
                            |e| errno_err(e.raw_os_error().unwrap_or(0), ""),
                        )?;
                    }
                    // `os_lockf_impl` answers `None`. `interp_posix.py:3012`
                    // answers the `0` the call returns on success, which 3.14
                    // — the oracle the parity suite reads — does not carry.
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.mkfifo(path, mode=0o666, *, dir_fd=None) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "mkfifo",
            crate::make_builtin_function("mkfifo", |args| {
                let (bound, kwargs) =
                    bind_path_args(args, "mkfifo", &["path", "mode"], 1, &["dir_fd"])?;
                let path = crate::gateway::fsencode_path_or_fd_w(
                    bound[0].expect("path is required"),
                    "mkfifo",
                    false,
                )?;
                // interp_posix.py `@unwrap_spec(mode=c_int, ...)`.
                let mode = match bound[1] {
                    Some(value) => crate::baseobjspace::c_int_w(value)? as libc::mode_t,
                    None => 0o666,
                };
                // `mkfifo` types `dir_fd` as `DirFD(rposix.HAVE_MKFIFOAT)`
                // (`interp_posix.py:1322`).
                let dir_fd = dir_fd_kwarg(kwargs, HAVE_MKFIFOAT)?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `mkfifoat` resolves the name against the descriptor
                // (`rposix.py:2784-2786`).
                // interp_posix.py `mkfifo`: retry on EINTR.
                loop {
                    let (r, errno) =
                        crate::module::thread::call_external_function(|| match dir_fd {
                            Some(dir_fd) => unsafe {
                                libc::mkfifoat(dir_fd, c_path.as_ptr(), mode)
                            },
                            None => unsafe { libc::mkfifo(c_path.as_ptr(), mode) },
                        });
                    if r >= 0 {
                        break;
                    }
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(errno),
                        |e| io_err_with_filename(e, path.w_path()),
                    )?;
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.mknod(path, mode=0o600, device=0, *, dir_fd=None) -> None
        // The node's kind is carried in `mode` alongside its permissions, so
        // an unadorned `mode` asks for a regular file — which is why the plain
        // call is the one a non-root process cannot make. `moduledef.py:160`
        // registers this only where the host has `mknod` at all.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "mknod",
            crate::make_builtin_function("mknod", |args| {
                let (bound, kwargs) =
                    bind_path_args(args, "mknod", &["path", "mode", "device"], 1, &["dir_fd"])?;
                let path = crate::gateway::fsencode_path_or_fd_w(
                    bound[0].expect("path is required"),
                    "mknod",
                    false,
                )?;
                // interp_posix.py `@unwrap_spec(mode=c_int, device=c_int,
                // ...)`.
                let mode = match bound[1] {
                    Some(value) => crate::baseobjspace::c_int_w(value)? as libc::mode_t,
                    None => 0o600,
                };
                let device = match bound[2] {
                    Some(value) => crate::baseobjspace::c_int_w(value)? as libc::dev_t,
                    None => 0,
                };
                // `mknod` types `dir_fd` as `DirFD(rposix.HAVE_MKNODAT)`
                // (`interp_posix.py:1345`).
                let dir_fd = dir_fd_kwarg(kwargs, HAVE_MKNODAT)?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                // `mknodat` resolves the name against the descriptor
                // (`rposix.py:2793-2795`).
                // interp_posix.py `mknod`: retry on EINTR.
                loop {
                    let (r, errno) =
                        crate::module::thread::call_external_function(|| match dir_fd {
                            Some(dir_fd) => unsafe {
                                libc::mknodat(dir_fd, c_path.as_ptr(), mode, device)
                            },
                            None => unsafe { libc::mknod(c_path.as_ptr(), mode, device) },
                        });
                    if r >= 0 {
                        break;
                    }
                    crate::builtins::eintr_retry_with(
                        std::io::Error::from_raw_os_error(errno),
                        |e| io_err_with_filename(e, path.w_path()),
                    )?;
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.chflags(path, flags, follow_symlinks=True) -> None
        // os.lchflags(path, flags) -> None
        //
        // One call whose `follow_symlinks=False` arm is `lchflags` under its
        // own name, which is what `os.py` reads `HAVE_LCHFLAGS` as. Only
        // the hosts that carry the pair are given the names at all: `chflags`
        // is a BSD interface, and `shutil.copystat` (`shutil.py`) reaches
        // for it through `lookup("chflags")`, which answers `_nop` where the
        // name is absent — so a name that exists has to work.
        //
        // Neither takes a `dir_fd`, so neither has a keyword-only tail, and a
        // surplus argument is counted the way `bind_path_args` counts one
        // without: `os.lchflags(p, 0, follow_symlinks=False)` is over the limit
        // rather than an unknown keyword.
        #[cfg(all(
            not(feature = "sandbox"),
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly",
            )
        ))]
        {
            // `<sys/stat.h>` declares `lchflags` on the Apple targets, where
            // the `libc` crate carries only `chflags` and `fchflags`.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            unsafe extern "C" {
                fn lchflags(path: *const libc::c_char, flags: libc::c_uint) -> libc::c_int;
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            use libc::lchflags;

            fn chflags_entry(
                args: &[pyre_object::PyObjectRef],
                name: &str,
                default_follow: bool,
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                let params: &[&'static str] = if default_follow {
                    &["path", "flags", "follow_symlinks"]
                } else {
                    &["path", "flags"]
                };
                let (bound, _) = bind_path_args(args, name, params, 2, &[])?;
                let path = crate::gateway::fsencode_path_or_fd_w(
                    bound[0].expect("path is required"),
                    name,
                    false,
                )?;
                // The flag word is read as a bit pattern rather than a number:
                // `SF_SETTABLE` does not fit a C int, and a negative value is
                // the mask it spells rather than an error.
                let flags =
                    crate::baseobjspace::int_w(bound[1].expect("flags is required"))? as u64;
                let follow = match bound.get(2).copied().flatten() {
                    Some(value) => crate::baseobjspace::is_true(value)?,
                    None => default_follow,
                };
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                let r = if follow {
                    unsafe { libc::chflags(c_path.as_ptr(), flags as _) }
                } else {
                    unsafe { lchflags(c_path.as_ptr(), flags as _) }
                };
                if r < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }
            crate::module_ns_store(
                ns,
                "chflags",
                crate::make_builtin_function("chflags", |args| {
                    chflags_entry(args, "chflags", true)
                }),
            );
            crate::module_ns_store(
                ns,
                "lchflags",
                crate::make_builtin_function("lchflags", |args| {
                    chflags_entry(args, "lchflags", false)
                }),
            );
        }

        // os.kill(pid, sig) / os.killpg(pgid, sig)
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "kill",
            crate::make_builtin_function_with_arity(
                "kill",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("kill() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pid=c_int, signal=c_int)`.
                    // Both arguments go through `c_int_w`: a raw payload read
                    // would accept any object, and `is_int` is an exact-type
                    // check, so an `int` subclass instance — `signal.SIGHUP` is
                    // an `IntEnum` member — never reaches the checked path.
                    let pid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                    let sig = crate::baseobjspace::c_int_w(args[1])? as libc::c_int;
                    let r = unsafe { libc::kill(pid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "killpg",
            crate::make_builtin_function_with_arity(
                "killpg",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("killpg() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(pgid=c_int, signal=c_int)`.
                    let pgid = crate::baseobjspace::c_int_w(args[0])? as libc::pid_t;
                    let sig = crate::baseobjspace::c_int_w(args[1])? as libc::c_int;
                    let r = unsafe { libc::killpg(pgid, sig) };
                    if r < 0 {
                        return Err(io_err(std::io::Error::last_os_error(), ""));
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.statvfs(path) / os.fstatvfs(fd) -> statvfs_result
        #[cfg(not(target_os = "redox"))]
        crate::module_ns_store(ns, "statvfs_result", statvfs_result_seq_type());

        #[cfg(not(target_os = "redox"))]
        fn statvfs_to_obj(
            info: rustpython_host_env::posix::StatVfsInfo,
        ) -> pyre_object::PyObjectRef {
            let seq = vec![
                pyre_object::w_int_new(info.f_bsize as i64),
                pyre_object::w_int_new(info.f_frsize as i64),
                pyre_object::w_int_new(info.f_blocks as i64),
                pyre_object::w_int_new(info.f_bfree as i64),
                pyre_object::w_int_new(info.f_bavail as i64),
                pyre_object::w_int_new(info.f_files as i64),
                pyre_object::w_int_new(info.f_ffree as i64),
                pyre_object::w_int_new(info.f_favail as i64),
                pyre_object::w_int_new(info.f_flag as i64),
                pyre_object::w_int_new(info.f_namemax as i64),
            ];
            let extras = vec![("f_fsid", pyre_object::w_int_new(info.f_fsid as i64))];
            crate::_structseq::new_instance_with_extra(statvfs_result_seq_type(), seq, extras)
        }
        #[cfg(not(target_os = "redox"))]
        crate::module_ns_store(
            ns,
            "statvfs",
            crate::make_builtin_function_with_arity(
                "statvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("statvfs() requires 1 argument"));
                    }
                    // interp_posix.py dispatches `rposix_stat.statvfs`
                    // with `allow_fd_fn=rposix_stat.fstatvfs`.
                    let path =
                        crate::gateway::fsencode_path_or_fd_w(args[0], "statvfs", HAVE_FSTATVFS)?;
                    if path.as_fd != -1 {
                        let info = host_posix::statvfs_fd(path.as_fd).map_err(|e| io_err(e, ""))?;
                        return Ok(statvfs_to_obj(info));
                    }
                    let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
                    let info = host_posix::statvfs_path(&c_path)
                        .map_err(|e| io_err_with_filename(e, path.w_path()))?;
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );
        #[cfg(not(target_os = "redox"))]
        crate::module_ns_store(
            ns,
            "fstatvfs",
            crate::make_builtin_function_with_arity(
                "fstatvfs",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fstatvfs() requires 1 argument"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    // interp_posix.py `fstatvfs`: retry on EINTR.
                    let info = loop {
                        match host_posix::statvfs_fd(fd) {
                            Ok(info) => break info,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    };
                    Ok(statvfs_to_obj(info))
                },
                1,
            ),
        );

        // os.cpu_count() -> int | None — `interp_posix.py`, which
        // answers None for a count of 0 or less. Both names read the processor
        // count through `host_cpu_count`; the syscalls it makes are the reason
        // the pair is left to the sandbox stubs below.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "cpu_count",
            crate::make_builtin_function_with_arity(
                "cpu_count",
                |_| {
                    let n = host_cpu_count();
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n))
                    }
                },
                0,
            ),
        );
        // _cpu_count alias — newer CPython exposes both.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "_cpu_count",
            crate::make_builtin_function_with_arity(
                "_cpu_count",
                |_| {
                    let n = host_cpu_count();
                    if n <= 0 {
                        Ok(pyre_object::w_none())
                    } else {
                        Ok(pyre_object::w_int_new(n))
                    }
                },
                0,
            ),
        );

        // os.symlink(src, dst, target_is_directory=False) -> None
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                let (bound, kwargs) = bind_path_args(
                    args,
                    "symlink",
                    &["src", "dst", "target_is_directory"],
                    2,
                    &["dir_fd"],
                )?;
                // `target_is_directory` selects between the two Windows link
                // kinds and is ignored everywhere else (`os_symlink_impl`).
                // Bound rather than dropped so a fourth positional is the
                // `dir_fd` error it is, not a silently created link.
                let _target_is_directory = match bound[2] {
                    Some(value) => crate::baseobjspace::is_true(value)?,
                    None => false,
                };
                // `symlink` types `dir_fd` as `DirFD(rposix.HAVE_SYMLINKAT)`;
                // the body below calls `libc::symlink`, which has no
                // descriptor arm.
                let _dir_fd = dir_fd_kwarg(kwargs, false)?;
                let src = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("src is required"),
                    "symlink",
                    "src",
                )?;
                let dst = crate::gateway::fsencode_path_named_w(
                    bound[1].expect("dst is required"),
                    "symlink",
                    "dst",
                )?;
                let c_src = std::ffi::CString::new(src.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // host_env::posix only exposes symlinkat on non-redox unices;
                // call libc::symlink directly so we don't need an at-cwd dance.
                let ret = unsafe { libc::symlink(c_src.as_ptr(), c_dst.as_ptr()) };
                if ret < 0 {
                    return Err(io_err_with_filename(
                        std::io::Error::last_os_error(),
                        dst.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.link(src, dst) -> None — a second name for the file `src` names,
        // both of which the failure reports.
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "link",
            crate::make_builtin_function("link", |args| {
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["src_dir_fd", "dst_dir_fd", "follow_symlinks"],
                    "link",
                )?;
                link_positional(args)?;
                let src = crate::gateway::fsencode_path_named_w(args[0], "link", "src")?;
                let dst = crate::gateway::fsencode_path_named_w(args[1], "link", "dst")?;
                let c_src = std::ffi::CString::new(src.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in src"))?;
                let c_dst = std::ffi::CString::new(dst.as_bytes.as_slice())
                    .map_err(|_| crate::PyError::value_error("embedded null in dst"))?;
                // `link` takes `DirFD(rposix.HAVE_LINKAT)` for both ends: a
                // descriptor the platform can honour resolves the name against
                // it, and one it cannot is refused rather than silently
                // resolved against the process's own directory.
                let dir_fd = |name: &str| -> Result<i32, crate::PyError> {
                    match crate::builtins::kwarg_get(kwargs, name)
                        .filter(|&w| !unsafe { pyre_object::is_none(w) })
                    {
                        Some(w) => unwrap_fd(w, "integer or None"),
                        None => Ok(libc::AT_FDCWD),
                    }
                };
                let src_dir_fd = dir_fd("src_dir_fd")?;
                let dst_dir_fd = dir_fd("dst_dir_fd")?;
                // `os_link_impl` follows the final symlink of `src` by
                // default, which is `AT_SYMLINK_FOLLOW`.
                let follow = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                    Some(w) => crate::baseobjspace::is_true(w)?,
                    None => true,
                };
                // Whether plain `link` follows a source symlink is left to the
                // implementation and the hosts disagree, so both answers are
                // spelled out through `linkat` rather than taken from it.
                let flags = if follow { libc::AT_SYMLINK_FOLLOW } else { 0 };
                let ret = unsafe {
                    libc::linkat(
                        src_dir_fd,
                        c_src.as_ptr(),
                        dst_dir_fd,
                        c_dst.as_ptr(),
                        flags,
                    )
                };
                if ret < 0 {
                    return Err(fs_err_with_filename2(
                        std::io::Error::last_os_error(),
                        0,
                        src.w_path(),
                        dst.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.chmod(path, mode, *, dir_fd=None, follow_symlinks=True) -> None
        #[cfg(not(feature = "sandbox"))]
        fn chmod_entry(
            args: &[pyre_object::PyObjectRef],
            name: &str,
            default_follow: bool,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            use std::os::fd::BorrowedFd;
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            // `lchmod(path, mode)` is `chmod(path, mode,
            // follow_symlinks=False)` under another name and declares no
            // keyword of its own.
            let allowed: &[&str] = if default_follow {
                &["path", "mode", "dir_fd", "follow_symlinks"]
            } else {
                &["path", "mode"]
            };
            crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
            if pos.len() > 2 {
                let surplus = if default_follow {
                    format!(
                        "{name}() takes exactly 2 positional arguments ({} given)",
                        pos.len()
                    )
                } else {
                    format!("{name}() takes at most 2 arguments ({} given)", pos.len())
                };
                return Err(crate::PyError::type_error(surplus));
            }
            let arg = |index: usize, key: &'static str| -> Result<PyObjectRef, crate::PyError> {
                match crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)? {
                    Some(value) => Ok(value),
                    None => Err(crate::PyError::type_error(format!(
                        "{name}() missing required argument '{key}' (pos {})",
                        index + 1
                    ))),
                }
            };
            let (path_obj, mode_obj) = (arg(0, "path")?, arg(1, "mode")?);
            // interp_posix.py reads a `chmod` whose path did not
            // fsencode as a descriptor and answers it with `os.fchmod`.
            // `lchmod` names no descriptor form.
            let path = crate::gateway::fsencode_path_or_fd_w(
                path_obj,
                name,
                default_follow && HAVE_FCHMOD,
            )?;
            // `posix.chmod` unwraps `mode` as `c_int`, so a non-integer raises
            // TypeError instead of reinterpreting its layout.
            let mode = crate::baseobjspace::c_int_w(mode_obj)? as u32;
            // `chmod` types `dir_fd` as `DirFD(rposix.HAVE_FCHMODAT)`
            // (`interp_posix.py:1197`).
            let dir_fd = dir_fd_kwarg(kwargs, HAVE_FCHMODAT)?;
            let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                Some(v) => crate::baseobjspace::is_true(v)?,
                None => default_follow,
            };
            // interp_posix.py `chmod`: retry the selected syscall on EINTR.
            if path.as_fd != -1 {
                // A descriptor answers before either modifier is consulted
                // (`interp_posix.py:1233-1242`), so neither is an error here —
                // unlike `chown`, which turns both away (`:2481-2486`). The
                // descriptor already names the file, and `fchmod` is what
                // `os.chmod(fd, …)` means.
                let bfd = unsafe { BorrowedFd::borrow_raw(path.as_fd) };
                loop {
                    match host_posix::fchmod(bfd, mode) {
                        Ok(()) => break,
                        Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                    }
                }
                return Ok(pyre_object::w_none());
            }
            let c_path = std::ffi::CString::new(path.as_bytes.as_slice())
                .map_err(|_| crate::PyError::value_error("embedded null in path"))?;
            // `_chmod_path` (`interp_posix.py`) keeps the plain
            // `chmod` for the unmodified call and reaches for `fchmodat` only
            // where a name has to be resolved against something else or the
            // final symlink must not be followed (`rposix.py`).
            let syscall = || {
                if dir_fd.is_some() || !follow_symlinks {
                    let flag = if follow_symlinks {
                        0
                    } else {
                        libc::AT_SYMLINK_NOFOLLOW
                    };
                    unsafe {
                        libc::fchmodat(
                            dir_fd.unwrap_or(libc::AT_FDCWD),
                            c_path.as_ptr(),
                            mode as libc::mode_t,
                            flag,
                        )
                    }
                } else {
                    unsafe { libc::chmod(c_path.as_ptr(), mode as libc::mode_t) }
                }
            };
            loop {
                if syscall() >= 0 {
                    break;
                }
                let err = std::io::Error::last_os_error();
                // A host can accept `AT_SYMLINK_NOFOLLOW` and not implement it,
                // reporting so by refusing the call rather than by lacking
                // `fchmodat` — which is why `HAVE_LCHMOD` is a narrower bit than
                // `HAVE_FCHMODAT`. `interp_posix.py:1247-1251` reads that refusal
                // as the modifier being unavailable rather than as an OS error,
                // and reads it ahead of the retry, so an unimplemented modifier
                // is never mistaken for an interruption.
                if !follow_symlinks {
                    let errno = crate::builtins::io_error_posix_errno(&err, 0);
                    if errno == libc::ENOTSUP || errno == libc::EOPNOTSUPP {
                        return Err(argument_unavailable(name, "follow_symlinks"));
                    }
                }
                crate::builtins::eintr_retry_with(err, |e| io_err_with_filename(e, path.w_path()))?;
            }
            Ok(pyre_object::w_none())
        }
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "chmod",
            crate::make_builtin_function("chmod", |args| chmod_entry(args, "chmod", true)),
        );
        // `os.lchmod` exists only where the host has a working one — os.py:159
        // records that some platforms carry a stub returning ENOTSUP, and that
        // `fchmodat`'s `AT_SYMLINK_NOFOLLOW` does not work either where that is
        // so. It is the same call the `follow_symlinks=False` arm above makes.
        #[cfg(all(
            not(feature = "sandbox"),
            any(
                target_os = "macos",
                target_os = "ios",
                target_os = "freebsd",
                target_os = "netbsd",
                target_os = "openbsd",
                target_os = "dragonfly",
            )
        ))]
        crate::module_ns_store(
            ns,
            "lchmod",
            crate::make_builtin_function("lchmod", |args| chmod_entry(args, "lchmod", false)),
        );

        // os.fchmod(fd, mode) -> None
        crate::module_ns_store(
            ns,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, mode=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                    let bfd = fd_borrow(fd)?;
                    // interp_posix.py `fchmod`: retry on EINTR.
                    loop {
                        match host_posix::fchmod(bfd, mode) {
                            Ok(()) => break,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    }
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True) -> None
        // os.lchown(path, uid, gid) -> None
        // `uid`/`gid` of -1 means "leave unchanged", as for fchown.
        fn chown_entry(
            args: &[pyre_object::PyObjectRef],
            name: &str,
            default_follow: bool,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            use std::os::fd::BorrowedFd;
            let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
            let allowed: &[&str] = if default_follow {
                &["path", "uid", "gid", "dir_fd", "follow_symlinks"]
            } else {
                &["path", "uid", "gid"]
            };
            crate::builtins::kwarg_reject_unknown(kwargs, allowed, name)?;
            if pos.len() > 3 {
                // `chown` declares keyword-only `dir_fd`/`follow_symlinks`, so
                // its surplus-positional report is the "positional arguments"
                // form; `lchown` takes no keywords at all and reports the
                // plain "arguments" form.
                let surplus = if default_follow {
                    format!(
                        "{name}() takes exactly 3 positional arguments ({} given)",
                        pos.len()
                    )
                } else {
                    format!("{name}() takes at most 3 arguments ({} given)", pos.len())
                };
                return Err(crate::PyError::type_error(surplus));
            }
            // Every parameter is bound duplicate-aware, so a call that supplies
            // one both ways raises before the ownership syscall runs.
            let arg = |index: usize, key: &'static str| -> Result<PyObjectRef, crate::PyError> {
                match crate::builtins::bind_pos_or_kw(pos, kwargs, index, key, name, index + 1)? {
                    Some(value) => Ok(value),
                    None => Err(crate::PyError::type_error(format!(
                        "{name}() missing required argument '{key}' (pos {})",
                        index + 1
                    ))),
                }
            };
            let (path_obj, uid_obj, gid_obj) = (arg(0, "path")?, arg(1, "uid")?, arg(2, "gid")?);
            // `posixmodule.c path_converter` calls `__fspath__` and lets what it
            // raises out: a `RuntimeError` from a user `__fspath__` is that
            // object's error, not a statement that the argument was the wrong
            // type.  Rewriting every failure into a `TypeError` here would also
            // swallow the `UnicodeEncodeError` a lone surrogate produces.
            let path = crate::gateway::fsencode_path_or_fd_w(
                path_obj,
                name,
                // `lchown` is `path_t(allow_fd=0)` — only `chown` reads an
                // integer as a descriptor (`interp_posix.py:2475-2481`).
                default_follow && HAVE_FCHOWN,
            )?;
            // `_Py_Uid_Converter` / `_Py_Gid_Converter`: `uid_t` is unsigned, yet
            // -1 is always accepted as the "leave unchanged" sentinel.  Only
            // that one value means unchanged; every other id is judged by
            // round-tripping through `uid_t`, so nothing is silently wrapped —
            // 2**32 truncates to 0 and would otherwise request uid 0.
            //
            // The two range reports follow the C converter's own split: a value
            // that still fits a C long but fails the round trip is "less than
            // minimum" (including 2**32, whose truncation reads as underflow),
            // while one too wide for a long is "greater than maximum".
            let id_of =
                |w: pyre_object::PyObjectRef, what: &str| -> Result<Option<u32>, crate::PyError> {
                    if !unsafe { crate::builtins::index_check(w) } {
                        return Err(crate::PyError::type_error(format!(
                            "{what} should be integer, not {}",
                            crate::type_methods::arg_type_name(w)
                        )));
                    }
                    let w_index = crate::baseobjspace::space_index(w)?;
                    let raw = crate::baseobjspace::int_w(w_index).map_err(|_| {
                        crate::PyError::overflow_error(format!("{what} is greater than maximum"))
                    })?;
                    if raw == -1 {
                        return Ok(None);
                    }
                    let narrowed = raw as u32;
                    if i64::from(narrowed) != raw {
                        return Err(crate::PyError::overflow_error(format!(
                            "{what} is less than minimum"
                        )));
                    }
                    Ok(Some(narrowed))
                };
            let (uid, gid) = (id_of(uid_obj, "uid")?, id_of(gid_obj, "gid")?);
            // `chown` types `dir_fd` as `DirFD(rposix.HAVE_FCHOWNAT)`
            // (`interp_posix.py`); `lchown` declares no keyword at
            // all, so `allowed` above has already rejected it.
            let dir_fd = dir_fd_kwarg(kwargs, HAVE_FCHOWNAT)?;
            let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                Some(v) => crate::baseobjspace::is_true(v)?,
                None => default_follow,
            };
            // interp_posix.py `chown`: retry the selected syscall on EINTR.
            if path.as_fd != -1 {
                // interp_posix.py:2481-2486 — a descriptor already names the
                // file, so neither modifier, which each reinterpret a name, can
                // apply. Upstream spells the second "cannnot"; 3.14, which the
                // parity suite reads as the oracle, spells it "cannot".
                if dir_fd.is_some() {
                    return Err(crate::PyError::value_error(format!(
                        "{name}: can't specify both dir_fd and fd"
                    )));
                }
                if !follow_symlinks {
                    return Err(crate::PyError::value_error(format!(
                        "{name}: cannot use fd and follow_symlinks together"
                    )));
                }
                let bfd = unsafe { BorrowedFd::borrow_raw(path.as_fd) };
                loop {
                    match host_posix::fchown(bfd, uid, gid) {
                        Ok(()) => break,
                        Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                    }
                }
                return Ok(pyre_object::w_none());
            }
            // `fchownat` is the whole family (`interp_posix.py`): the
            // flagless call follows the final symlink and `AT_SYMLINK_NOFOLLOW`
            // does not, while the directory descriptor it resolves the name
            // against is `AT_FDCWD` when the caller named none.
            let at = fd_borrow(dir_fd.unwrap_or(libc::AT_FDCWD))?;
            if name == "chown" {
                loop {
                    match host_posix::fchownat(
                        at,
                        path_from_bytes(&path.as_bytes).as_os_str(),
                        uid,
                        gid,
                        follow_symlinks,
                    ) {
                        Ok(()) => break,
                        Err(e) => crate::builtins::eintr_retry_with(e, |e| {
                            io_err_with_filename(e, path.w_path())
                        })?,
                    }
                }
            } else {
                host_posix::fchownat(
                    at,
                    path_from_bytes(&path.as_bytes).as_os_str(),
                    uid,
                    gid,
                    follow_symlinks,
                )
                .map_err(|e| io_err_with_filename(e, path.w_path()))?;
            }
            Ok(pyre_object::w_none())
        }
        crate::module_ns_store(
            ns,
            "chown",
            crate::make_builtin_function("chown", |args| chown_entry(args, "chown", true)),
        );
        crate::module_ns_store(
            ns,
            "lchown",
            crate::make_builtin_function("lchown", |args| chown_entry(args, "lchown", false)),
        );

        // os.fchown(fd, uid, gid) -> None  (uid/gid of -1 means "leave unchanged")
        crate::module_ns_store(
            ns,
            "fchown",
            crate::make_builtin_function_with_arity(
                "fchown",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error("fchown() requires 3 arguments"));
                    }
                    // interp_posix.py `@unwrap_spec(uid=c_uid_t,
                    // gid=c_gid_t)` with the descriptor taken by
                    // `space.c_filedescriptor_w`. The spec is applied by the
                    // gateway before the body runs, so a bad uid/gid is
                    // reported ahead of a bad descriptor.
                    //
                    // `c_uid_t_w` (baseobjspace.py) is what turns -1
                    // into `UINT_MAX`, i.e. the `(uid_t)-1` "leave unchanged"
                    // sentinel that `host_posix::fchown` spells as `None`.
                    let unchanged = |value: u32| (value != u32::MAX).then_some(value);
                    let uid = unchanged(crate::baseobjspace::c_uid_t_w(args[1])?);
                    let gid = unchanged(crate::baseobjspace::c_uid_t_w(args[2])?);
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    let bfd = fd_borrow(fd)?;
                    // interp_posix.py `fchown`: retry on EINTR.
                    loop {
                        match host_posix::fchown(bfd, uid, gid) {
                            Ok(()) => break,
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    }
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.get_inheritable(fd) -> bool.  `_Py_get_inheritable`: a descriptor
        // is inheritable exactly when its close-on-exec flag is clear.
        crate::module_ns_store(
            ns,
            "get_inheritable",
            crate::make_builtin_function_with_arity(
                "get_inheritable",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "get_inheritable() requires 1 argument",
                        ));
                    }
                    use std::os::fd::BorrowedFd;
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let bfd = fd_borrow(fd)?;
                    let inheritable = rustpython_host_env::fcntl::get_inheritable(bfd)
                        .map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_bool_from(inheritable))
                },
                1,
            ),
        );

        // os.set_inheritable(fd, inheritable) -> None
        crate::module_ns_store(
            ns,
            "set_inheritable",
            crate::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "set_inheritable() requires 2 arguments",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, inheritable=int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let inherit = crate::baseobjspace::int_w(args[1])? != 0;
                    let bfd = fd_borrow(fd)?;
                    host_posix::set_inheritable(bfd, inherit).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.access(path, mode, *, dir_fd=None, effective_ids=False,
        //           follow_symlinks=True) -> bool
        crate::module_ns_store(
            ns,
            "access",
            crate::make_builtin_function("access", |args| {
                // `access` names three keyword-only modifiers, so a third
                // positional is an error rather than a `dir_fd`.
                let (bound, kwargs) = bind_path_args(
                    args,
                    "access",
                    &["path", "mode"],
                    2,
                    &["dir_fd", "effective_ids", "follow_symlinks"],
                )?;
                // The parameters convert in declaration order, and every one of
                // them can raise, so the order is observable: `path` reports
                // before `mode`, `mode` before `dir_fd`, and both before either
                // flag's `__bool__` is called at all.
                let path = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("path is required"),
                    "access",
                    "path",
                )?
                .as_bytes;
                // interp_posix.py `@unwrap_spec(mode=c_int, ...)`.
                let mode = crate::baseobjspace::c_int_w(bound[1].expect("mode is required"))?;
                // interp_posix.py:745 types `dir_fd` as
                // `DirFD(rposix.HAVE_FACCESSAT)`, so a host with no `faccessat`
                // turns the descriptor away instead of resolving the name
                // against the working directory as though none had been given.
                let dir_fd = dir_fd_kwarg(kwargs, HAVE_FACCESSAT)?;
                let effective_ids = match crate::builtins::kwarg_get(kwargs, "effective_ids") {
                    Some(v) => crate::baseobjspace::is_true(v)?,
                    None => false,
                };
                let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                    Some(v) => crate::baseobjspace::is_true(v)?,
                    None => true,
                };
                // interp_posix.py:771-775 — the two flag modifiers have no other
                // call to reach, so without `faccessat` they are refused rather
                // than answered as though they had been applied.
                if !HAVE_FACCESSAT {
                    if !follow_symlinks {
                        return Err(argument_unavailable("access", "follow_symlinks"));
                    }
                    if effective_ids {
                        return Err(argument_unavailable("access", "effective_ids"));
                    }
                }
                #[cfg(feature = "sandbox")]
                {
                    // `HAVE_FACCESSAT` is false here, so the three modifiers
                    // have already been turned away and only the plain form is
                    // left to serve.
                    let _ = dir_fd;
                    return Ok(pyre_object::w_bool_from(
                        crate::host_seam::ops::access(&path, mode).unwrap_or(false),
                    ));
                }
                #[cfg(not(feature = "sandbox"))]
                {
                    let c_path = std::ffi::CString::new(path.as_slice())
                        .map_err(|_| crate::PyError::value_error("embedded null character"))?;
                    // interp_posix.py keeps the plain `access` for the
                    // unmodified call and reaches for `faccessat` only where the
                    // name resolves against a descriptor, the final symlink must
                    // not be followed, or the effective ids are the ones to ask
                    // about. `rposix.py:2551-2560` is the flag mapping.
                    let ret = if dir_fd.is_some() || !follow_symlinks || effective_ids {
                        let mut flags = 0;
                        if !follow_symlinks {
                            flags |= libc::AT_SYMLINK_NOFOLLOW;
                        }
                        if effective_ids {
                            flags |= libc::AT_EACCESS;
                        }
                        unsafe {
                            libc::faccessat(
                                dir_fd.unwrap_or(libc::AT_FDCWD),
                                c_path.as_ptr(),
                                mode,
                                flags,
                            )
                        }
                    } else {
                        unsafe { libc::access(c_path.as_ptr(), mode) }
                    };
                    // `rposix.access` and `rposix.faccessat` both answer
                    // `error == 0` without `handle_posix_error`, so a refused
                    // call is False and not an `OSError` — including the EINVAL
                    // a mode outside `R_OK | W_OK | X_OK` can draw.
                    Ok(pyre_object::w_bool_from(ret == 0))
                }
            }),
        );

        // os.chroot(path) -> None
        crate::module_ns_store(
            ns,
            "chroot",
            crate::make_builtin_function_with_arity(
                "chroot",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chroot() requires 1 argument"));
                    }
                    let path = crate::gateway::fsencode_path_named_w(args[0], "chroot", "path")?;
                    host_posix::chroot(path_from_bytes(&path.as_bytes).as_ref())
                        .map_err(|e| io_err_with_filename(e, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.getloadavg() -> (1m, 5m, 15m)
        crate::module_ns_store(
            ns,
            "getloadavg",
            crate::make_builtin_function_with_arity(
                "getloadavg",
                |_| {
                    let [l1, l5, l15] =
                        rustpython_host_env::time::getloadavg().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_float_new(l1),
                        pyre_object::w_float_new(l5),
                        pyre_object::w_float_new(l15),
                    ]))
                },
                0,
            ),
        );

        // os.times() -> posix.times_result(user, system, children_user,
        //                                  children_system, elapsed)
        crate::module_ns_store(
            ns,
            "times",
            crate::make_builtin_function_with_arity(
                "times",
                |_| {
                    let t =
                        rustpython_host_env::time::process_times().map_err(|e| io_err(e, ""))?;
                    Ok(crate::_structseq::new_instance(
                        times_result_seq_type(),
                        vec![
                            pyre_object::w_float_new(t.user),
                            pyre_object::w_float_new(t.system),
                            pyre_object::w_float_new(t.children_user),
                            pyre_object::w_float_new(t.children_system),
                            pyre_object::w_float_new(t.elapsed),
                        ],
                    ))
                },
                0,
            ),
        );

        // os.waitstatus_to_exitcode(status) -> int
        crate::module_ns_store(
            ns,
            "waitstatus_to_exitcode",
            crate::make_builtin_function_with_arity(
                "waitstatus_to_exitcode",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "waitstatus_to_exitcode() requires 1 argument",
                        ));
                    }
                    // app_posix.py:149-176 is app-level and reaches the status
                    // through `posix.WIFEXITED`/`WEXITSTATUS`, each of which is
                    // `@unwrap_spec(status=c_int)`.
                    let status = crate::baseobjspace::c_int_w(args[0])?;
                    match rustpython_host_env::time::waitstatus_to_exitcode(status) {
                        Some(code) => Ok(pyre_object::w_int_new(code as i64)),
                        None => Err(crate::PyError::value_error(
                            "waitstatus_to_exitcode: invalid status",
                        )),
                    }
                },
                1,
            ),
        );

        // os.system(command) -> exit_status
        crate::module_ns_store(
            ns,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    // `interp_posix.py:815 command='fsencode'`, which
                    // `gateway.py:365` unwraps with `space.fsencode_w`: the
                    // shell gets the filesystem bytes, so a command naming a
                    // byte with no UTF-8 spelling survives instead of being
                    // refused, and `bytes` / `__fspath__` are accepted as the
                    // converter accepts them.
                    let cmd = crate::gateway::fsencode_bytes_w(args[0])?;
                    let c_cmd = std::ffi::CString::new(cmd)
                        .map_err(|_| crate::PyError::value_error("embedded null in command"))?;
                    let rc = rustpython_host_env::os::system(&c_cmd);
                    Ok(pyre_object::w_int_new(rc as i64))
                },
                1,
            ),
        );

        // os.sendfile(out_fd, in_fd, offset, count) -> bytes_sent
        //
        // Ported from pypy/module/posix/interp_posix.py:2932-2961:
        //   * 4 positional args: out_fd, in_fd (called "in_" in PyPy because
        //     "in" is reserved), offset, count.
        //   * offset == None: linux-only "no-offset" path (NULL pointer);
        //     non-linux raises TypeError("an integer is required (got None)")
        //     verbatim from PyPy.
        //   * offset == int: read as i64 (PyPy uses
        //     space.gateway_r_longlong_w) and routed through
        //     rustpython_host_env::posix::sendfile (linux) or the BSD-form
        //     wrapper (macos).
        //   * Returns bytes-sent as int (PyPy: space.newint(res)).
        //
        // Both arms of `interp_posix.py:2958-2974` sit in a
        // `while True: ... except OSError: wrap_oserror(..., eintr_retry=True)`,
        // so an interrupted transfer runs the pending Python signal handlers and
        // then goes back to the call. The three below do the same through
        // `builtins::eintr_retry_with`.
        //
        // The BSD arm discards a partial `sbytes` on EINTR rather than reporting
        // it: `rposix.py` rescues a partial transfer for `EAGAIN` and
        // `EBUSY` alone, and EINTR falls through to `handle_posix_error`, which
        // raises. The loop then re-runs the whole call with the same `offset` and
        // `count` — both are loop-invariant in `interp_posix.py`, and `rposix`
        // never sees the retry — so the transfer restarts from the range the
        // caller asked for, not from where it had got to.
        #[cfg(all(
            any(target_os = "linux", target_os = "macos"),
            not(feature = "sandbox")
        ))]
        crate::module_ns_store(
            ns,
            "sendfile",
            crate::make_builtin_function("sendfile", |args| {
                use std::os::fd::BorrowedFd;
                // Every parameter is positional-or-keyword. `headers`,
                // `trailers` and `flags` are the BSD `sendfile(2)` tail. The
                // macOS arm forwards both vectors; `flags` alone remains
                // unused because the host wrapper exposes no flags parameter.
                // Listing the BSD-only parameters makes unknown keywords fail
                // during argument binding on every platform.
                let (bound, _kwargs) = bind_path_args(
                    args,
                    "sendfile",
                    &[
                        "out_fd", "in_fd", "offset", "count", "headers", "trailers", "flags",
                    ],
                    4,
                    &[],
                )?;
                // interp_posix.py `@unwrap_spec(out_fd=c_int, count=int)`,
                // with `in_ = space.c_int_w(w_in_fd)` in the body (:2955). The
                // spec runs in the gateway, so the count is converted before
                // the descriptor argument that follows it here.
                let out_fd = crate::baseobjspace::c_int_w(bound[0].expect("out_fd is required"))?;
                let count_raw = crate::baseobjspace::int_w(bound[3].expect("count is required"))?;
                let in_fd = crate::baseobjspace::c_int_w(bound[1].expect("in_fd is required"))?;
                let w_offset = bound[2].expect("offset is required");
                if unsafe { pyre_object::is_none(w_offset) } {
                    // linux-only no-offset path; non-linux raises TypeError
                    // matching interp_posix.py:2946.
                    #[cfg(not(target_os = "linux"))]
                    {
                        let _ = (out_fd, in_fd, count_raw);
                        return Err(crate::PyError::type_error(
                            "an integer is required (got None)",
                        ));
                    }
                    #[cfg(target_os = "linux")]
                    {
                        // host_env doesn't expose a NULL-offset variant; call
                        // libc::sendfile directly with a null pointer, matching
                        // rposix.sendfile_no_offset (rposix.py).
                        let count = count_raw as libc::size_t;
                        loop {
                            let (res, errno) =
                                crate::module::thread::call_external_function(|| unsafe {
                                    libc::sendfile(out_fd, in_fd, core::ptr::null_mut(), count)
                                });
                            if res >= 0 {
                                return Ok(pyre_object::w_int_new(res as i64));
                            }
                            crate::builtins::eintr_retry_with(
                                std::io::Error::from_raw_os_error(errno),
                                |e| io_err(e, ""),
                            )?;
                        }
                    }
                }
                // interp_posix.py `space.gateway_r_longlong_w(w_offset)`.
                let offset_i64 = crate::baseobjspace::int_w(w_offset)?;
                let out_b = fd_borrow(out_fd)?;
                let in_b = fd_borrow(in_fd)?;
                #[cfg(target_os = "linux")]
                {
                    let count = count_raw as usize;
                    loop {
                        // Seeded from the caller's value on every attempt.
                        // `rposix.sendfile` (`rposix.py`) writes the
                        // offset into a fresh cell each call from the argument it
                        // was passed, and the retry sits above it holding that
                        // argument unchanged, so what a failed call left behind
                        // here is not what the next one starts from.
                        let mut offset: rustpython_host_env::crt_fd::Offset = offset_i64 as _;
                        let result = {
                            let _blocked = crate::module::thread::before_external_block();
                            host_posix::sendfile(out_b, in_b, &mut offset, count)
                        };
                        match result {
                            Ok(n) => return Ok(pyre_object::w_int_new(n as i64)),
                            Err(e) => crate::builtins::eintr_retry_with(e, |e| io_err(e, ""))?,
                        }
                    }
                }
                #[cfg(target_os = "macos")]
                {
                    // Both Python sequences and all of their buffer exports are
                    // consumed before entering the EINTR retry loop. The retry
                    // therefore reuses only Rust-owned bytes.
                    let (header_buffers, trailer_buffers) = {
                        let _roots = pyre_object::gc_roots::push_roots();
                        let header_slot = bound[4].map(|value| {
                            let slot = pyre_object::gc_roots::shadow_stack_len();
                            pyre_object::gc_roots::pin_root(value);
                            slot
                        });
                        let trailer_slot = bound[5].map(|value| {
                            let slot = pyre_object::gc_roots::shadow_stack_len();
                            pyre_object::gc_roots::pin_root(value);
                            slot
                        });
                        let collect_buffers = |slot: Option<usize>, name: &str| {
                            let Some(slot) = slot else {
                                return Ok(None);
                            };
                            let value = pyre_object::gc_roots::shadow_stack_get(slot);
                            if unsafe { pyre_object::is_none(value) } {
                                return Ok(None);
                            }
                            // Indexed header/trailer vectors require a sequence;
                            // consuming an iterator or mapping keys would change
                            // the accepted `sendfile` argument protocol.
                            if !crate::baseobjspace::issequence_w(value) {
                                return Err(crate::PyError::type_error(format!(
                                    "sendfile() {name} must be a sequence"
                                )));
                            }
                            let items = crate::baseobjspace::unpackiterable(value, -1)?;
                            let items_base = pyre_object::gc_roots::pin_roots(&items);
                            let mut buffers = Vec::with_capacity(items.len());
                            for index in 0..items.len() {
                                let item =
                                    pyre_object::gc_roots::shadow_stack_get(items_base + index);
                                let Some(buffer) =
                                    crate::baseobjspace::simple_buffer_bytes(item)?
                                else {
                                    return Err(crate::PyError::type_error(format!(
                                        "sendfile() {name} items must be bytes-like"
                                    )));
                                };
                                buffers.push(buffer.as_bytes().to_vec());
                                buffer.release();
                            }
                            if buffers.is_empty() {
                                Ok(None)
                            } else {
                                Ok(Some(buffers))
                            }
                        };
                        (
                            collect_buffers(header_slot, "headers")?,
                            collect_buffers(trailer_slot, "trailers")?,
                        )
                    };
                    // An empty sequence is indistinguishable from an absent
                    // one at the syscall boundary, independently for headers
                    // and trailers.
                    let header_slices = header_buffers.as_ref().map(|buffers| {
                        buffers
                            .iter()
                            .map(Vec::as_slice)
                            .collect::<Vec<&[u8]>>()
                    });
                    let trailer_slices = trailer_buffers.as_ref().map(|buffers| {
                        buffers
                            .iter()
                            .map(Vec::as_slice)
                            .collect::<Vec<&[u8]>>()
                    });
                    // `sendfile(2)` on this host spends the length cell on the
                    // header and the file together — "the value of len argument
                    // indicates the maximum number of bytes in the header
                    // and/or file to be sent" — so a caller asking for `count`
                    // bytes of the file has to be given room for its headers on
                    // top, or the headers eat into the range it asked for. The
                    // trailer is outside the budget and is always sent whole.
                    // A count of 0 already asks for everything and stays 0.
                    let count = match header_buffers.as_ref() {
                        Some(buffers) if count_raw != 0 => {
                            buffers.iter().try_fold(count_raw, |count, buffer| {
                                count.checked_add(buffer.len() as i64).ok_or_else(|| {
                                    crate::PyError::overflow_error("sendfile() count is too large")
                                })
                            })?
                        }
                        _ => count_raw,
                    };
                    loop {
                        let (res, written) = {
                            let _blocked = crate::module::thread::before_external_block();
                            host_posix::sendfile(
                                in_b,
                                out_b,
                                offset_i64 as rustpython_host_env::crt_fd::Offset,
                                count,
                                header_slices.as_deref(),
                                trailer_slices.as_deref(),
                            )
                        };
                        match res {
                            Ok(_) => return Ok(pyre_object::w_int_new(written)),
                            Err(error) => {
                                // rposix.py: BSD sendfile reports a
                                // partial transfer through sbytes even when the
                                // syscall result is EAGAIN/EBUSY. Return that
                                // progress so asyncio advances its file offset
                                // instead of resending the same range. EINTR is
                                // not in that set, so a partial transfer a signal
                                // interrupted goes to the retry below, which asks
                                // for the caller's original range again.
                                if written != 0
                                    && matches!(
                                        error.raw_os_error(),
                                        Some(libc::EAGAIN) | Some(libc::EBUSY)
                                    )
                                {
                                    return Ok(pyre_object::w_int_new(written));
                                }
                                crate::builtins::eintr_retry_with(error, |e| io_err(e, ""))?;
                            }
                        }
                    }
                }
            }),
        );

        // os.posix_spawn(path, argv, env, *, file_actions=None, setpgroup=None,
        // resetids=False, setsid=False, setsigmask=(), setsigdef=(),
        // scheduler=None) -> pid
        // os.posix_spawnp(file, argv, env, *, file_actions=None, ...) -> pid
        #[cfg(all(
            any(target_os = "linux", target_os = "freebsd", target_os = "macos"),
            not(feature = "sandbox")
        ))]
        {
            struct SpawnScheduler {
                #[allow(dead_code)]
                policy: Option<libc::c_int>,
                #[allow(dead_code)]
                param: libc::sched_param,
            }

            /// The `POSIX_SPAWN_SETSID` spawn attribute, or `None` where the
            /// platform has no such flag — which is what makes `setsid=True`
            /// report an unavailable argument rather than being ignored.
            #[cfg(target_os = "linux")]
            const POSIX_SPAWN_SETSID: Option<libc::c_int> = Some(libc::POSIX_SPAWN_SETSID);
            /// `<sys/spawn.h>` defines the flag, but the `libc` binding for
            /// this target does not export it.
            #[cfg(target_vendor = "apple")]
            const POSIX_SPAWN_SETSID: Option<libc::c_int> = Some(0x0400);
            #[cfg(not(any(target_os = "linux", target_vendor = "apple")))]
            const POSIX_SPAWN_SETSID: Option<libc::c_int> = None;

            fn build_posix_spawn(
                args: &[pyre_object::PyObjectRef],
                spawnp: bool,
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                // The two entry points share this body and have an argument
                // clinic declaration each, so the one the caller reached is the
                // name its rejected path reports.
                let func = if spawnp {
                    "posix_spawnp"
                } else {
                    "posix_spawn"
                };
                // `(path, argv, env, /, *, file_actions=(), ...)` — the three
                // names are positional-only and everything else is
                // keyword-only, so there is no positional-or-keyword slot at
                // all.
                let (bound, kwargs) = bind_posonly_args(
                    args,
                    func,
                    func,
                    3,
                    3,
                    &[
                        "file_actions",
                        "setpgroup",
                        "resetids",
                        "setsid",
                        "setsigmask",
                        "setsigdef",
                        "scheduler",
                    ],
                )?;
                // Every conversion below reaches app-level code -- `__fspath__`,
                // a mapping's `keys()`, `__index__` -- and each one can collect
                // and move the arguments that are still unread, including the
                // keyword dictionary the remaining options are looked up in.
                // Publish them once and read each back at its use.
                let _roots = pyre_object::gc_roots::push_roots();
                let positional_base = pyre_object::gc_roots::pin_roots(&[
                    bound[0].expect("path is required"),
                    bound[1].expect("argv is required"),
                    bound[2].expect("env is required"),
                ]);
                let kwargs_slot = kwargs.map(|kwargs| {
                    let slot = pyre_object::gc_roots::shadow_stack_len();
                    pyre_object::gc_roots::pin_root(kwargs);
                    slot
                });
                let positional =
                    |index: usize| pyre_object::gc_roots::shadow_stack_get(positional_base + index);
                let kwargs = || kwargs_slot.map(pyre_object::gc_roots::shadow_stack_get);
                let path = crate::gateway::fsencode_path_named_w(positional(0), func, "path")?;
                let c_path = std::ffi::CString::new(path.as_bytes.as_slice()).map_err(|_| {
                    crate::PyError::value_error("posix_spawn: embedded null in path")
                })?;
                let argv = collect_cstring_seq(positional(1), func, "argv")?;
                // posixmodule.c parses `env` through the same keys/values
                // snapshot used by execve, then filesystem-encodes paired
                // elements into `key=value`.
                let env = collect_spawn_env(positional(2))?;
                let setpgroup = match crate::builtins::kwarg_get(kwargs(), "setpgroup") {
                    Some(value) if !unsafe { pyre_object::is_none(value) } => {
                        let value = crate::baseobjspace::space_index(value)?;
                        let value = crate::baseobjspace::int_w(value)?;
                        Some(libc::pid_t::try_from(value).map_err(|_| {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C pid_t",
                            )
                        })?)
                    }
                    _ => None,
                };
                let resetids = crate::builtins::kwarg_get(kwargs(), "resetids")
                    .map(crate::baseobjspace::is_true)
                    .transpose()?
                    .unwrap_or(false);
                let setsid = crate::builtins::kwarg_get(kwargs(), "setsid")
                    .map(crate::baseobjspace::is_true)
                    .transpose()?
                    .unwrap_or(false);
                if setsid && POSIX_SPAWN_SETSID.is_none() {
                    return Err(argument_unavailable(func, "setsid"));
                }
                let setsigmask = match crate::builtins::kwarg_get(kwargs(), "setsigmask") {
                    Some(value) => Some(sigset_arg(value)?),
                    None => None,
                };
                let setsigdef = match crate::builtins::kwarg_get(kwargs(), "setsigdef") {
                    Some(value) => Some(sigset_arg(value)?),
                    None => None,
                };
                let scheduler = parse_spawn_scheduler(func, kwargs())?;
                let file_actions_obj = crate::builtins::kwarg_get(kwargs(), "file_actions");
                let actions: Vec<rustpython_host_env::posix::PosixSpawnFileAction> =
                    if let Some(fa) = file_actions_obj {
                        if unsafe { pyre_object::is_none(fa) } {
                            Vec::new()
                        } else {
                            decode_file_actions(fa)?
                        }
                    } else {
                        Vec::new()
                    };
                let config = LocalPosixSpawnConfig {
                    path: c_path.as_c_str(),
                    args: &argv,
                    env: &env,
                    file_actions: &actions,
                    setsigdef: setsigdef.as_deref(),
                    setpgroup,
                    resetids,
                    setsid,
                    setsigmask: setsigmask.as_deref(),
                    scheduler: scheduler.as_ref(),
                    spawnp,
                };
                let pid = local_posix_spawn(config)
                    .map_err(|e| io_err_with_filename(e, path.w_path()))?;
                Ok(pyre_object::w_int_new(pid as i64))
            }
            fn collect_spawn_env(
                mapping: pyre_object::PyObjectRef,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                // CPython's posix_spawn accepts None as "inherit environ";
                // subprocess._posix_spawn uses exactly this form when Popen
                // was called without an explicit env mapping.
                let entries = if unsafe { pyre_object::is_none(mapping) } {
                    host_os::vars_os()
                        .map(|(key, value)| {
                            let key = key.as_encoded_bytes();
                            let value = value.as_encoded_bytes();
                            let mut entry = Vec::with_capacity(key.len() + 1 + value.len());
                            entry.extend_from_slice(key);
                            entry.push(b'=');
                            entry.extend_from_slice(value);
                            entry
                        })
                        .collect()
                } else {
                    collect_env_entries(mapping, "posix_spawn", true)?
                };
                entries
                    .into_iter()
                    .map(|entry| {
                        std::ffi::CString::new(entry).map_err(|_| {
                            crate::PyError::value_error(
                                "posix_spawn() environment contains an embedded null byte",
                            )
                        })
                    })
                    .collect()
            }
            fn collect_cstring_seq(
                obj: pyre_object::PyObjectRef,
                fn_name: &str,
                arg_name: &str,
            ) -> Result<Vec<std::ffi::CString>, crate::PyError> {
                let items: Vec<pyre_object::PyObjectRef> = if unsafe { pyre_object::is_list(obj) } {
                    let n = unsafe { pyre_object::w_list_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i as i64) })
                        .collect()
                } else if unsafe { pyre_object::is_tuple(obj) } {
                    let n = unsafe { pyre_object::w_tuple_len(obj) };
                    (0..n)
                        .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i as i64) })
                        .collect()
                } else {
                    return Err(crate::PyError::type_error(format!(
                        "{fn_name}(): {arg_name} must be a list or tuple",
                    )));
                };
                // `interp_posix.py:1742 args = [space.fsencode_w(w_arg)
                // for w_arg in args_w]`: every argv/envp entry crosses
                // to the new process as filesystem bytes, and the same
                // converter decides what an entry may be.
                // `fsencode_bytes_w` reaches `__fspath__` for a non-str, non-bytes entry, so
                // encoding one entry can collect and move the entries not yet converted.
                // Publish the sequence once and read each entry back per iteration.
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut out = Vec::with_capacity(items.len());
                for i in 0..items.len() {
                    let bytes = crate::gateway::fsencode_bytes_w(
                        pyre_object::gc_roots::shadow_stack_get(items_base + i),
                    )?;
                    out.push(std::ffi::CString::new(bytes).map_err(|_| {
                        crate::PyError::value_error(format!(
                            "{fn_name}(): embedded null in {arg_name}",
                        ))
                    })?);
                }
                Ok(out)
            }
            fn decode_file_actions(
                obj: pyre_object::PyObjectRef,
            ) -> Result<Vec<rustpython_host_env::posix::PosixSpawnFileAction>, crate::PyError>
            {
                use rustpython_host_env::posix::PosixSpawnFileAction;
                let items =
                    crate::builtins::sequence_fast(obj, "file_actions must be a sequence or None")?;
                // Every field of a `file_actions` entry is an `int` argument of
                // `os.posix_spawn`, so it is converted rather than read as a
                // payload: the caller controls the tuple's contents, and an
                // `int` subclass or a plain non-int would otherwise be
                // reinterpreted as a descriptor, flag set or mode.
                //
                // That conversion reaches `__index__`, and an OPEN path reaches
                // `__fspath__`, so reading one field can collect and move the
                // entries not yet read. The sequence is published once and each
                // entry read back out of its slot at every use, the way
                // `collect_cstring_seq` above does.
                let field = |slot: usize, index: i64| -> Result<i32, crate::PyError> {
                    let entry = pyre_object::gc_roots::shadow_stack_get(slot);
                    let value =
                        unsafe { pyre_object::w_tuple_getitem(entry, index) }.ok_or_else(|| {
                            crate::PyError::type_error(
                                "Each file_actions element must be a non-empty tuple",
                            )
                        })?;
                    crate::baseobjspace::c_int_w(value)
                };
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut out = Vec::with_capacity(items.len());
                for offset in 0..items.len() {
                    let slot = items_base + offset;
                    let entry = pyre_object::gc_roots::shadow_stack_get(slot);
                    let tlen = if unsafe { pyre_object::is_tuple(entry) } {
                        unsafe { pyre_object::w_tuple_len(entry) }
                    } else {
                        return Err(crate::PyError::type_error(
                            "Each file_actions element must be a non-empty tuple",
                        ));
                    };
                    if tlen == 0 {
                        return Err(crate::PyError::type_error(
                            "Each file_actions element must be a non-empty tuple",
                        ));
                    }
                    let op = field(slot, 0)?;
                    match op {
                        0 => {
                            // POSIX_SPAWN_OPEN: (op, fd, path, flags, mode)
                            if tlen != 5 {
                                return Err(crate::PyError::type_error(
                                    "A open file_action tuple must have 5 elements",
                                ));
                            }
                            let fd = field(slot, 1)?;
                            let path_obj = unsafe {
                                pyre_object::w_tuple_getitem(
                                    pyre_object::gc_roots::shadow_stack_get(slot),
                                    2,
                                )
                                .unwrap()
                            };
                            let path = extract_path(path_obj)?;
                            let cpath = std::ffi::CString::new(path).map_err(|_| {
                                crate::PyError::value_error(
                                    "posix_spawn: embedded null in OPEN path",
                                )
                            })?;
                            let oflag = field(slot, 3)?;
                            let mode = field(slot, 4)? as u32;
                            out.push(PosixSpawnFileAction::Open {
                                fd,
                                path: cpath,
                                oflag,
                                mode,
                            });
                        }
                        1 => {
                            // POSIX_SPAWN_CLOSE: (op, fd)
                            if tlen != 2 {
                                return Err(crate::PyError::type_error(
                                    "A close file_action tuple must have 2 elements",
                                ));
                            }
                            let fd = field(slot, 1)?;
                            out.push(PosixSpawnFileAction::Close { fd });
                        }
                        2 => {
                            // POSIX_SPAWN_DUP2: (op, fd, newfd)
                            if tlen != 3 {
                                return Err(crate::PyError::type_error(
                                    "A dup2 file_action tuple must have 3 elements",
                                ));
                            }
                            let fd = field(slot, 1)?;
                            let newfd = field(slot, 2)?;
                            out.push(PosixSpawnFileAction::Dup2 { fd, newfd });
                        }
                        _ => {
                            return Err(crate::PyError::type_error(
                                "Unknown file_actions identifier",
                            ));
                        }
                    }
                }
                Ok(out)
            }

            fn sigset_arg(value: PyObjectRef) -> Result<Vec<i32>, crate::PyError> {
                let items = crate::builtins::collect_iterable(value)?;
                // `space_index` runs `__index__`, so reading one element can
                // collect and move the elements not yet read.  Publish the
                // sequence once and read each element back per iteration.
                let _seq_roots = pyre_object::gc_roots::push_roots();
                let items_base = pyre_object::gc_roots::pin_roots(&items);
                let mut sigs = Vec::with_capacity(items.len());
                for offset in 0..items.len() {
                    let item = pyre_object::gc_roots::shadow_stack_get(items_base + offset);
                    let item = crate::baseobjspace::space_index(item)?;
                    let signum = crate::baseobjspace::int_w(item)?;
                    if !(1..crate::module::signal::signalstate::NSIG as i64).contains(&signum) {
                        return Err(crate::PyError::value_error(format!(
                            "signal number {signum} out of range [1; {}]",
                            crate::module::signal::signalstate::NSIG - 1
                        )));
                    }
                    sigs.push(signum as i32);
                }
                Ok(sigs)
            }

            fn parse_spawn_scheduler(
                func: &str,
                kwargs: Option<PyObjectRef>,
            ) -> Result<Option<SpawnScheduler>, crate::PyError> {
                let Some(value) = crate::builtins::kwarg_get(kwargs, "scheduler") else {
                    return Ok(None);
                };
                if unsafe { pyre_object::is_none(value) } {
                    return Ok(None);
                }
                if unsafe { !pyre_object::is_tuple(value) } {
                    return Err(crate::PyError::type_error(format!(
                        "{func}: scheduler must be a tuple or None"
                    )));
                }
                if unsafe { pyre_object::w_tuple_len(value) } != 2 {
                    return Err(crate::PyError::type_error(
                        "A scheduler tuple must have two elements",
                    ));
                }

                #[cfg(all(target_os = "linux", not(target_env = "musl")))]
                {
                    // `sched_priority_w` reaches `__index__`, so the collection
                    // it can trigger forwards the tuple's own slots while a raw
                    // element read before it goes stale.  Root the tuple and
                    // take the policy out of it afterwards.
                    let _roots = pyre_object::gc_roots::push_roots();
                    let tuple_slot = pyre_object::gc_roots::shadow_stack_len();
                    pyre_object::gc_roots::pin_root(value);
                    let param_obj = unsafe { pyre_object::w_tuple_getitem(value, 1).unwrap() };
                    let priority = sched_priority_w(param_obj)?;
                    let value = pyre_object::gc_roots::shadow_stack_get(tuple_slot);
                    let policy_obj = unsafe { pyre_object::w_tuple_getitem(value, 0).unwrap() };
                    let mut param: libc::sched_param =
                        unsafe { core::mem::zeroed::<libc::sched_param>() };
                    param.sched_priority = priority;
                    let policy = if unsafe { pyre_object::is_none(policy_obj) } {
                        None
                    } else {
                        let policy = crate::baseobjspace::space_index(policy_obj)?;
                        let policy = crate::baseobjspace::int_w(policy)?;
                        Some(libc::c_int::try_from(policy).map_err(|_| {
                            crate::PyError::overflow_error(
                                "Python int too large to convert to C int",
                            )
                        })?)
                    };
                    Ok(Some(SpawnScheduler { policy, param }))
                }

                #[cfg(any(not(target_os = "linux"), target_env = "musl"))]
                {
                    Err(crate::PyError::not_implemented(
                        "The scheduler option is not supported in this system.",
                    ))
                }
            }

            struct LocalPosixSpawnConfig<'a> {
                path: &'a std::ffi::CStr,
                args: &'a [std::ffi::CString],
                env: &'a [std::ffi::CString],
                file_actions: &'a [rustpython_host_env::posix::PosixSpawnFileAction],
                setsigdef: Option<&'a [i32]>,
                setpgroup: Option<libc::pid_t>,
                resetids: bool,
                setsid: bool,
                setsigmask: Option<&'a [i32]>,
                scheduler: Option<&'a SpawnScheduler>,
                spawnp: bool,
            }

            fn errno_result(ret: libc::c_int) -> std::io::Result<()> {
                if ret == 0 {
                    Ok(())
                } else {
                    Err(std::io::Error::from_raw_os_error(ret))
                }
            }

            unsafe fn fill_sigset(set: *mut libc::sigset_t, sigs: &[i32]) -> std::io::Result<()> {
                if unsafe { libc::sigemptyset(set) } != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                for signum in sigs {
                    if unsafe { libc::sigaddset(set, *signum) } != 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                }
                Ok(())
            }

            fn build_spawn_file_actions(
                actions: &[rustpython_host_env::posix::PosixSpawnFileAction],
            ) -> std::io::Result<Option<libc::posix_spawn_file_actions_t>> {
                use rustpython_host_env::posix::PosixSpawnFileAction;
                if actions.is_empty() {
                    return Ok(None);
                }
                let mut raw = unsafe { core::mem::zeroed::<libc::posix_spawn_file_actions_t>() };
                errno_result(unsafe { libc::posix_spawn_file_actions_init(&mut raw) })?;
                for action in actions {
                    let result = match action {
                        PosixSpawnFileAction::Open {
                            fd,
                            path,
                            oflag,
                            mode,
                        } => unsafe {
                            libc::posix_spawn_file_actions_addopen(
                                &mut raw,
                                *fd,
                                path.as_ptr(),
                                *oflag,
                                *mode as libc::mode_t,
                            )
                        },
                        PosixSpawnFileAction::Close { fd } => unsafe {
                            libc::posix_spawn_file_actions_addclose(&mut raw, *fd)
                        },
                        PosixSpawnFileAction::Dup2 { fd, newfd } => unsafe {
                            libc::posix_spawn_file_actions_adddup2(&mut raw, *fd, *newfd)
                        },
                    };
                    if let Err(error) = errno_result(result) {
                        unsafe { libc::posix_spawn_file_actions_destroy(&mut raw) };
                        return Err(error);
                    }
                }
                Ok(Some(raw))
            }

            fn build_spawn_attrs(
                config: &LocalPosixSpawnConfig<'_>,
            ) -> std::io::Result<libc::posix_spawnattr_t> {
                let mut raw = unsafe { core::mem::zeroed::<libc::posix_spawnattr_t>() };
                errno_result(unsafe { libc::posix_spawnattr_init(&mut raw) })?;
                let mut flags = 0i32;
                if let Some(pgid) = config.setpgroup {
                    if let Err(error) =
                        errno_result(unsafe { libc::posix_spawnattr_setpgroup(&mut raw, pgid) })
                    {
                        unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                        return Err(error);
                    }
                    flags |= libc::POSIX_SPAWN_SETPGROUP as i32;
                }
                if config.resetids {
                    flags |= libc::POSIX_SPAWN_RESETIDS as i32;
                }
                if config.setsid {
                    let Some(setsid_flag) = POSIX_SPAWN_SETSID else {
                        unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Unsupported,
                            "posix_spawn: setsid unavailable on this platform",
                        ));
                    };
                    flags |= setsid_flag as i32;
                }
                if let Some(sigs) = config.setsigmask {
                    let mut set = unsafe { core::mem::zeroed::<libc::sigset_t>() };
                    if let Err(error) = unsafe { fill_sigset(&mut set, sigs) }.and_then(|_| {
                        errno_result(unsafe { libc::posix_spawnattr_setsigmask(&mut raw, &set) })
                    }) {
                        unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                        return Err(error);
                    }
                    flags |= libc::POSIX_SPAWN_SETSIGMASK as i32;
                }
                if let Some(sigs) = config.setsigdef {
                    let mut set = unsafe { core::mem::zeroed::<libc::sigset_t>() };
                    if let Err(error) = unsafe { fill_sigset(&mut set, sigs) }.and_then(|_| {
                        errno_result(unsafe { libc::posix_spawnattr_setsigdefault(&mut raw, &set) })
                    }) {
                        unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                        return Err(error);
                    }
                    flags |= libc::POSIX_SPAWN_SETSIGDEF as i32;
                }
                if let Some(scheduler) = config.scheduler {
                    #[cfg(target_os = "linux")]
                    {
                        if let Some(policy) = scheduler.policy {
                            if let Err(error) = errno_result(unsafe {
                                libc::posix_spawnattr_setschedpolicy(&mut raw, policy)
                            }) {
                                unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                                return Err(error);
                            }
                            flags |= libc::POSIX_SPAWN_SETSCHEDULER as i32;
                        }
                        if let Err(error) = errno_result(unsafe {
                            libc::posix_spawnattr_setschedparam(&mut raw, &scheduler.param)
                        }) {
                            unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                            return Err(error);
                        }
                        flags |= libc::POSIX_SPAWN_SETSCHEDPARAM as i32;
                    }
                }
                if let Err(error) =
                    errno_result(unsafe { libc::posix_spawnattr_setflags(&mut raw, flags as _) })
                {
                    unsafe { libc::posix_spawnattr_destroy(&mut raw) };
                    return Err(error);
                }
                Ok(raw)
            }

            fn local_posix_spawn(
                config: LocalPosixSpawnConfig<'_>,
            ) -> std::io::Result<libc::pid_t> {
                let mut actions = build_spawn_file_actions(config.file_actions)?;
                // `actions` is initialized C state, not a Rust value a drop
                // reclaims, so a later failure has to destroy it explicitly.
                let mut attrs = match build_spawn_attrs(&config) {
                    Ok(attrs) => attrs,
                    Err(error) => {
                        if let Some(actions) = actions.as_mut() {
                            unsafe { libc::posix_spawn_file_actions_destroy(actions) };
                        }
                        return Err(error);
                    }
                };
                let mut argv: Vec<*mut libc::c_char> = config
                    .args
                    .iter()
                    .map(|arg| arg.as_ptr() as *mut libc::c_char)
                    .collect();
                argv.push(std::ptr::null_mut());
                let mut env: Vec<*mut libc::c_char> = config
                    .env
                    .iter()
                    .map(|entry| entry.as_ptr() as *mut libc::c_char)
                    .collect();
                env.push(std::ptr::null_mut());
                let actionsp = actions
                    .as_mut()
                    .map_or(std::ptr::null(), |actions| actions as *mut _ as *const _);
                let mut pid: libc::pid_t = 0;
                let ret = crate::module::thread::call_external_function(|| unsafe {
                    if config.spawnp {
                        libc::posix_spawnp(
                            &mut pid,
                            config.path.as_ptr(),
                            actionsp,
                            &attrs,
                            argv.as_ptr(),
                            env.as_ptr(),
                        )
                    } else {
                        libc::posix_spawn(
                            &mut pid,
                            config.path.as_ptr(),
                            actionsp,
                            &attrs,
                            argv.as_ptr(),
                            env.as_ptr(),
                        )
                    }
                })
                .0;
                unsafe { libc::posix_spawnattr_destroy(&mut attrs) };
                if let Some(actions) = actions.as_mut() {
                    unsafe { libc::posix_spawn_file_actions_destroy(actions) };
                }
                errno_result(ret)?;
                Ok(pid)
            }
            crate::module_ns_store(
                ns,
                "posix_spawn",
                crate::make_builtin_function("posix_spawn", |args| build_posix_spawn(args, false)),
            );
            crate::module_ns_store(
                ns,
                "posix_spawnp",
                crate::make_builtin_function("posix_spawnp", |args| build_posix_spawn(args, true)),
            );
            crate::module_ns_store(ns, "POSIX_SPAWN_OPEN", pyre_object::w_int_new(0));
            crate::module_ns_store(ns, "POSIX_SPAWN_CLOSE", pyre_object::w_int_new(1));
            crate::module_ns_store(ns, "POSIX_SPAWN_DUP2", pyre_object::w_int_new(2));
        }

        // os.ttyname(fd) -> str
        crate::module_ns_store(
            ns,
            "ttyname",
            crate::make_builtin_function_with_arity(
                "ttyname",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("ttyname() requires fd"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let bfd = fd_borrow(fd)?;
                    let name = host_posix::ttyname(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(crate::gateway::fsdecode_os_str(name.as_os_str()))
                },
                1,
            ),
        );

        // os.tcgetpgrp(fd) -> pgid
        crate::module_ns_store(
            ns,
            "tcgetpgrp",
            crate::make_builtin_function_with_arity(
                "tcgetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("tcgetpgrp() requires fd"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let bfd = fd_borrow(fd)?;
                    let pgid = host_posix::tcgetpgrp(bfd).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(pgid as i64))
                },
                1,
            ),
        );

        // os.tcsetpgrp(fd, pgid) -> None
        crate::module_ns_store(
            ns,
            "tcsetpgrp",
            crate::make_builtin_function_with_arity(
                "tcsetpgrp",
                |args| {
                    use std::os::fd::BorrowedFd;
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("tcsetpgrp() requires fd, pgid"));
                    }
                    // interp_posix.py `@unwrap_spec(fd=c_int, pgid=c_gid_t)`.
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let pgid = crate::baseobjspace::c_uid_t_w(args[1])? as libc::pid_t;
                    let bfd = fd_borrow(fd)?;
                    host_posix::tcsetpgrp(bfd, pgid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.getpriority(which, who) -> int
        crate::module_ns_store(
            ns,
            "getpriority",
            crate::make_builtin_function_with_arity(
                "getpriority",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "getpriority() requires which, who",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(which=int, who=int)`.
                    let which =
                        crate::baseobjspace::int_w(args[0])? as host_posix::PriorityWhichType;
                    let who = crate::baseobjspace::int_w(args[1])? as host_posix::PriorityWhoType;
                    let prio = host_posix::getpriority(which, who).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(prio as i64))
                },
                2,
            ),
        );

        // os.setpriority(which, who, priority) -> None
        crate::module_ns_store(
            ns,
            "setpriority",
            crate::make_builtin_function_with_arity(
                "setpriority",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setpriority() requires which, who, priority",
                        ));
                    }
                    // interp_posix.py:2352 `@unwrap_spec(which=int, who=int,
                    // priority=int)`.
                    let which =
                        crate::baseobjspace::int_w(args[0])? as host_posix::PriorityWhichType;
                    let who = crate::baseobjspace::int_w(args[1])? as host_posix::PriorityWhoType;
                    let prio = crate::baseobjspace::int_w(args[2])? as i32;
                    host_posix::setpriority(which, who, prio).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        crate::module_ns_store(
            ns,
            "PRIO_PROCESS",
            pyre_object::w_int_new(libc::PRIO_PROCESS as i64),
        );
        crate::module_ns_store(
            ns,
            "PRIO_PGRP",
            pyre_object::w_int_new(libc::PRIO_PGRP as i64),
        );
        crate::module_ns_store(
            ns,
            "PRIO_USER",
            pyre_object::w_int_new(libc::PRIO_USER as i64),
        );

        // `posixmodule.c` `pathconf_names` — the `_PC_*` table
        // `conv_path_confname` resolves a string `name` argument through.
        // `libc` exports the constants only on the BSD family, so the glibc
        // values (`bits/confname.h`) are spelled out for Linux.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const PATHCONF_NAMES: &[(&str, i32)] = &[
            ("PC_ALLOC_SIZE_MIN", libc::_PC_ALLOC_SIZE_MIN),
            ("PC_ASYNC_IO", libc::_PC_ASYNC_IO),
            ("PC_CHOWN_RESTRICTED", libc::_PC_CHOWN_RESTRICTED),
            ("PC_FILESIZEBITS", libc::_PC_FILESIZEBITS),
            ("PC_LINK_MAX", libc::_PC_LINK_MAX),
            ("PC_MAX_CANON", libc::_PC_MAX_CANON),
            ("PC_MAX_INPUT", libc::_PC_MAX_INPUT),
            ("PC_MIN_HOLE_SIZE", libc::_PC_MIN_HOLE_SIZE),
            ("PC_NAME_MAX", libc::_PC_NAME_MAX),
            ("PC_NO_TRUNC", libc::_PC_NO_TRUNC),
            ("PC_PATH_MAX", libc::_PC_PATH_MAX),
            ("PC_PIPE_BUF", libc::_PC_PIPE_BUF),
            ("PC_PRIO_IO", libc::_PC_PRIO_IO),
            ("PC_REC_INCR_XFER_SIZE", libc::_PC_REC_INCR_XFER_SIZE),
            ("PC_REC_MAX_XFER_SIZE", libc::_PC_REC_MAX_XFER_SIZE),
            ("PC_REC_MIN_XFER_SIZE", libc::_PC_REC_MIN_XFER_SIZE),
            ("PC_REC_XFER_ALIGN", libc::_PC_REC_XFER_ALIGN),
            ("PC_SYMLINK_MAX", libc::_PC_SYMLINK_MAX),
            ("PC_SYNC_IO", libc::_PC_SYNC_IO),
            ("PC_VDISABLE", libc::_PC_VDISABLE),
        ];
        #[cfg(target_os = "linux")]
        const PATHCONF_NAMES: &[(&str, i32)] = &[
            ("PC_2_SYMLINKS", 20),
            ("PC_ALLOC_SIZE_MIN", 18),
            ("PC_ASYNC_IO", 10),
            ("PC_CHOWN_RESTRICTED", 6),
            ("PC_FILESIZEBITS", 13),
            ("PC_LINK_MAX", 0),
            ("PC_MAX_CANON", 1),
            ("PC_MAX_INPUT", 2),
            ("PC_NAME_MAX", 3),
            ("PC_NO_TRUNC", 7),
            ("PC_PATH_MAX", 4),
            ("PC_PIPE_BUF", 5),
            ("PC_PRIO_IO", 11),
            ("PC_REC_INCR_XFER_SIZE", 14),
            ("PC_REC_MAX_XFER_SIZE", 15),
            ("PC_REC_MIN_XFER_SIZE", 16),
            ("PC_REC_XFER_ALIGN", 17),
            ("PC_SOCK_MAXBUF", 12),
            ("PC_SYMLINK_MAX", 19),
            ("PC_SYNC_IO", 9),
            ("PC_VDISABLE", 8),
        ];
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
        const PATHCONF_NAMES: &[(&str, i32)] = &[];
        /// The dict a `conv_confname` table is published as — `pathconf_names`
        /// for `pathconf`, `confstr_names` for `confstr`.
        fn store_names_dict(ns: PyObjectRef, key: &str, table: &[(&str, i32)]) {
            let _names_roots = pyre_object::gc_roots::push_roots();
            let names_slot = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(pyre_object::w_dict_new());
            for (name, value) in table {
                // The value is allocated before the store, and the dict is
                // reloaded from its root slot every iteration because the
                // insert itself can grow — and so relocate — the dict.
                let w_value = pyre_object::w_int_new(*value as i64);
                unsafe {
                    pyre_object::w_dict_setitem_str(
                        pyre_object::gc_roots::shadow_stack_get(names_slot),
                        name,
                        w_value,
                    )
                };
            }
            crate::module_ns_store(ns, key, pyre_object::gc_roots::shadow_stack_get(names_slot));
        }
        store_names_dict(ns, "pathconf_names", PATHCONF_NAMES);

        /// A limit the host has no determinate answer for. `pathconf` reports
        /// it as `-1` with the errno left alone, which the host wrapper spells
        /// `None` — but `interp_posix.py` hands whatever `pathconf`
        /// returned straight to `space.newint`, so what the caller sees is the
        /// number `-1`. `PC_ASYNC_IO` and `PC_SYMLINK_MAX` answer this way on
        /// hosts that do not implement them, and `None` is neither the value
        /// nor the type the caller can compare against a limit.
        fn indeterminate_limit(limit: Option<libc::c_long>) -> i64 {
            limit.map_or(-1, |v| v)
        }

        /// `posixmodule.c conv_confname`: an `int` passes through, a `str` is
        /// resolved through the table the caller's entry point publishes.
        fn confname_arg(w: PyObjectRef, table: &[(&str, i32)]) -> Result<i32, crate::PyError> {
            if unsafe { pyre_object::is_str(w) } {
                // A str carrying a lone surrogate has no `&str` view.  It simply
                // matches no known name, which is the ValueError below — not an
                // interpreter abort, which is what reading the value unchecked
                // would produce.
                let name = unsafe { pyre_object::w_str_get_value_opt(w) };
                return name
                    .and_then(|name| {
                        table
                            .iter()
                            .find(|(known, _)| *known == name)
                            .map(|(_, value)| *value)
                    })
                    .ok_or_else(|| crate::PyError::value_error("unrecognized configuration name"));
            }
            // `conv_confname` gates on `PyIndex_Check` before converting, so an
            // object that is neither a str nor index-able is this TypeError,
            // while an `__index__` that raises propagates its own exception.
            if !unsafe { crate::builtins::index_check(w) } {
                return Err(crate::PyError::type_error(
                    "configuration names must be strings or integers",
                ));
            }
            // `conv_confname` narrows to a C `int` and reports a value that does
            // not fit rather than truncating it. A truncated name reaches the
            // syscall as an unrelated one — `2**40` narrows to 0 — and comes
            // back EINVAL, which reads as "no such configuration option" for a
            // name the caller never asked about. The object is an int here, so
            // an `int_w` that fails did so on width.
            let too_large =
                || crate::PyError::overflow_error("Python int too large to convert to C int");
            let value = crate::baseobjspace::int_w(crate::baseobjspace::space_index(w)?)
                .map_err(|_| too_large())?;
            i32::try_from(value).map_err(|_| too_large())
        }

        // os.pathconf(path, name) -> int | None
        crate::module_ns_store(
            ns,
            "pathconf",
            crate::make_builtin_function_with_arity(
                "pathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("pathconf() requires path, name"));
                    }
                    // interp_posix.py `path_or_fd(allow_fd=hasattr(os,
                    // 'fpathconf'))`, whose body converts the name before it
                    // reads `path.as_fd != -1`.
                    let path =
                        crate::gateway::fsencode_path_or_fd_w(args[0], "pathconf", HAVE_FPATHCONF)?;
                    let name = confname_arg(args[1], PATHCONF_NAMES)?;
                    let limit = if path.as_fd != -1 {
                        host_posix::fpathconf(path.as_fd, name).map_err(|e| io_err(e, ""))?
                    } else {
                        let cpath =
                            std::ffi::CString::new(path.as_bytes.as_slice()).map_err(|_| {
                                crate::PyError::value_error("pathconf: embedded null in path")
                            })?;
                        host_posix::pathconf(&cpath, name)
                            .map_err(|e| io_err_with_filename(e, path.w_path()))?
                    };
                    Ok(pyre_object::w_int_new(indeterminate_limit(limit)))
                },
                2,
            ),
        );

        // os.fpathconf(fd, name) -> int | None
        crate::module_ns_store(
            ns,
            "fpathconf",
            crate::make_builtin_function_with_arity(
                "fpathconf",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fpathconf() requires fd, name"));
                    }
                    // interp_posix.py:2411 descriptor argument: accept an int
                    // or a `fileno()` object through `space.c_filedescriptor_w`;
                    // that boundary also raises the bool file descriptor warning.
                    let fd = crate::baseobjspace::c_filedescriptor_w(args[0])?;
                    let name = confname_arg(args[1], PATHCONF_NAMES)?;
                    let limit = host_posix::fpathconf(fd, name).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(indeterminate_limit(limit)))
                },
                2,
            ),
        );

        // os.sysconf(name) -> int
        crate::module_ns_store(
            ns,
            "sysconf",
            crate::make_builtin_function_with_arity(
                "sysconf",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("sysconf() requires name"));
                    }
                    let name = confname_arg(args[0], sysconf_names())?;
                    let v = host_posix::sysconf(name).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_int_new(v as i64))
                },
                1,
            ),
        );
        let w_sysconf_names = pyre_object::w_dict_new();
        let _sysconf_names_root = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_sysconf_names);
        let names_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        for (name, value) in sysconf_names() {
            // Build the value first: call arguments evaluate left to right, so
            // reading the rooted dict inline would take its address before
            // `w_int_new` allocates, and a collection there would leave the
            // store writing through the pre-move address.
            let w_value = pyre_object::w_int_new(*value as i64);
            unsafe {
                pyre_object::w_dict_setitem_str(
                    pyre_object::gc_roots::shadow_stack_get(names_slot),
                    name,
                    w_value,
                );
            }
        }
        crate::module_ns_store(
            ns,
            "sysconf_names",
            pyre_object::gc_roots::shadow_stack_get(names_slot),
        );

        // `posixmodule.c` `posix_constants_confstr` — the `_CS_*` table
        // `conv_confstr_confname` resolves a string `name` argument through,
        // and the same candidate set `rposix.py:2248-2300` names. Every entry
        // there is `#ifdef`-guarded, so a host publishes exactly the names its
        // own `<unistd.h>` defines; `libc` carries `_CS_PATH` alone, and the
        // two numberings disagree from that first entry on — it is 1 on the
        // Apple targets and 0 in glibc's `bits/confname.h` enum. The ten names
        // the candidate set carries for the System V hosts (`CS_ARCHITECTURE`,
        // `CS_HOSTNAME`, `CS_HW_PROVIDER`, `CS_HW_SERIAL`, `CS_INITTAB_NAME`,
        // `CS_MACHINE`, `CS_RELEASE`, `CS_SRPC_DOMAIN`, `CS_SYSNAME`,
        // `CS_VERSION`) are defined by neither header, so neither table has
        // them.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        const CONFSTR_NAMES: &[(&str, i32)] = &[
            ("CS_PATH", 1),
            ("CS_XBS5_ILP32_OFF32_CFLAGS", 20),
            ("CS_XBS5_ILP32_OFF32_LDFLAGS", 21),
            ("CS_XBS5_ILP32_OFF32_LIBS", 22),
            ("CS_XBS5_ILP32_OFF32_LINTFLAGS", 23),
            ("CS_XBS5_ILP32_OFFBIG_CFLAGS", 24),
            ("CS_XBS5_ILP32_OFFBIG_LDFLAGS", 25),
            ("CS_XBS5_ILP32_OFFBIG_LIBS", 26),
            ("CS_XBS5_ILP32_OFFBIG_LINTFLAGS", 27),
            ("CS_XBS5_LP64_OFF64_CFLAGS", 28),
            ("CS_XBS5_LP64_OFF64_LDFLAGS", 29),
            ("CS_XBS5_LP64_OFF64_LIBS", 30),
            ("CS_XBS5_LP64_OFF64_LINTFLAGS", 31),
            ("CS_XBS5_LPBIG_OFFBIG_CFLAGS", 32),
            ("CS_XBS5_LPBIG_OFFBIG_LDFLAGS", 33),
            ("CS_XBS5_LPBIG_OFFBIG_LIBS", 34),
            ("CS_XBS5_LPBIG_OFFBIG_LINTFLAGS", 35),
        ];
        // glibc numbers the enum from zero and restarts it twice, at 1000 for
        // the large-file names and at 1100 for the XBS5 ones.
        #[cfg(target_os = "linux")]
        const CONFSTR_NAMES: &[(&str, i32)] = &[
            ("CS_PATH", 0),
            ("CS_GNU_LIBC_VERSION", 2),
            ("CS_GNU_LIBPTHREAD_VERSION", 3),
            ("CS_LFS_CFLAGS", 1000),
            ("CS_LFS_LDFLAGS", 1001),
            ("CS_LFS_LIBS", 1002),
            ("CS_LFS_LINTFLAGS", 1003),
            ("CS_LFS64_CFLAGS", 1004),
            ("CS_LFS64_LDFLAGS", 1005),
            ("CS_LFS64_LIBS", 1006),
            ("CS_LFS64_LINTFLAGS", 1007),
            ("CS_XBS5_ILP32_OFF32_CFLAGS", 1100),
            ("CS_XBS5_ILP32_OFF32_LDFLAGS", 1101),
            ("CS_XBS5_ILP32_OFF32_LIBS", 1102),
            ("CS_XBS5_ILP32_OFF32_LINTFLAGS", 1103),
            ("CS_XBS5_ILP32_OFFBIG_CFLAGS", 1104),
            ("CS_XBS5_ILP32_OFFBIG_LDFLAGS", 1105),
            ("CS_XBS5_ILP32_OFFBIG_LIBS", 1106),
            ("CS_XBS5_ILP32_OFFBIG_LINTFLAGS", 1107),
            ("CS_XBS5_LP64_OFF64_CFLAGS", 1108),
            ("CS_XBS5_LP64_OFF64_LDFLAGS", 1109),
            ("CS_XBS5_LP64_OFF64_LIBS", 1110),
            ("CS_XBS5_LP64_OFF64_LINTFLAGS", 1111),
            ("CS_XBS5_LPBIG_OFFBIG_CFLAGS", 1112),
            ("CS_XBS5_LPBIG_OFFBIG_LDFLAGS", 1113),
            ("CS_XBS5_LPBIG_OFFBIG_LIBS", 1114),
            ("CS_XBS5_LPBIG_OFFBIG_LINTFLAGS", 1115),
        ];
        #[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "linux")))]
        const CONFSTR_NAMES: &[(&str, i32)] = &[];
        store_names_dict(ns, "confstr_names", CONFSTR_NAMES);

        // os.confstr(name) -> str | None
        #[cfg(not(feature = "sandbox"))]
        crate::module_ns_store(
            ns,
            "confstr",
            crate::make_builtin_function_with_arity(
                "confstr",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("confstr() requires name"));
                    }
                    let name = confname_arg(args[0], CONFSTR_NAMES)?;
                    // `rposix.confstr` (`rposix.py`) asks for the
                    // length first and fills a buffer of exactly that size on
                    // the second call. A zero length is either a name this host
                    // has no string for, which is `None`, or a name it does not
                    // know at all, which is the errno it set — so errno is
                    // cleared before the question is put.
                    rustpython_host_env::os::clear_errno();
                    let len = unsafe { libc::confstr(name, std::ptr::null_mut(), 0) };
                    if len == 0 {
                        let errno = crate::builtins::crt_errno();
                        if errno != 0 {
                            return Err(errno_err(errno, ""));
                        }
                        return Ok(pyre_object::w_none());
                    }
                    let mut buf = vec![0u8; len];
                    unsafe { libc::confstr(name, buf.as_mut_ptr() as *mut libc::c_char, len) };
                    // The length counts the terminator, which is not part of
                    // the string — `os_confstr_impl` decodes `len - 1` bytes.
                    // (`rffi.charp2strn(buf, n)` keeps it, so upstream's
                    // `space.newtext` carries a trailing NUL.)
                    buf.truncate(len - 1);
                    // The value can be a search path, so it is decoded the way
                    // every other name from the host is.
                    Ok(crate::gateway::fsdecode_filename_bytes(&buf))
                },
                1,
            ),
        );

        // os.initgroups(username, gid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "initgroups",
            crate::make_builtin_function_with_arity(
                "initgroups",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "initgroups() requires username, gid",
                        ));
                    }
                    let user = unsafe {
                        if pyre_object::is_str(args[0]) {
                            crate::baseobjspace::str_utf8_w(args[0])?.to_string()
                        } else {
                            return Err(crate::PyError::type_error(
                                "initgroups(): username must be str",
                            ));
                        }
                    };
                    let cuser = std::ffi::CString::new(user.as_bytes()).map_err(|_| {
                        crate::PyError::value_error("initgroups: embedded null in username")
                    })?;
                    // interp_posix.py `@unwrap_spec(username='text', gid=c_gid_t)`.
                    let gid = crate::baseobjspace::c_uid_t_w(args[1])?;
                    host_posix::initgroups(&cuser, gid).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.openpty() -> (master_fd, slave_fd)
        crate::module_ns_store(
            ns,
            "openpty",
            crate::make_builtin_function_with_arity(
                "openpty",
                |_| {
                    use std::os::fd::IntoRawFd;
                    let (master, slave) = host_posix::openpty().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(master.into_raw_fd() as i64),
                        pyre_object::w_int_new(slave.into_raw_fd() as i64),
                    ]))
                },
                0,
            ),
        );

        // os.getresuid() -> (ruid, euid, suid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "getresuid",
            crate::make_builtin_function_with_arity(
                "getresuid",
                |_| {
                    let (r, e, s) = host_posix::getresuid().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(r as i64),
                        pyre_object::w_int_new(e as i64),
                        pyre_object::w_int_new(s as i64),
                    ]))
                },
                0,
            ),
        );

        // os.getresgid() -> (rgid, egid, sgid)
        #[cfg(any(target_os = "android", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "getresgid",
            crate::make_builtin_function_with_arity(
                "getresgid",
                |_| {
                    let (r, e, s) = host_posix::getresgid().map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(r as i64),
                        pyre_object::w_int_new(e as i64),
                        pyre_object::w_int_new(s as i64),
                    ]))
                },
                0,
            ),
        );

        // os.setresuid(ruid, euid, suid) -> None
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "openbsd"
        ))]
        crate::module_ns_store(
            ns,
            "setresuid",
            crate::make_builtin_function_with_arity(
                "setresuid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresuid() requires ruid, euid, suid",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(ruid=c_uid_t,
                    // euid=c_uid_t, suid=c_uid_t)`.
                    let r = crate::baseobjspace::c_uid_t_w(args[0])?;
                    let e = crate::baseobjspace::c_uid_t_w(args[1])?;
                    let s = crate::baseobjspace::c_uid_t_w(args[2])?;
                    host_posix::setresuid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );

        // os.setresgid(rgid, egid, sgid) -> None
        #[cfg(any(target_os = "freebsd", target_os = "linux", target_os = "openbsd"))]
        crate::module_ns_store(
            ns,
            "setresgid",
            crate::make_builtin_function_with_arity(
                "setresgid",
                |args| {
                    if args.len() < 3 {
                        return Err(crate::PyError::type_error(
                            "setresgid() requires rgid, egid, sgid",
                        ));
                    }
                    // interp_posix.py `@unwrap_spec(rgid=c_gid_t,
                    // egid=c_gid_t, sgid=c_gid_t)`.
                    let r = crate::baseobjspace::c_uid_t_w(args[0])?;
                    let e = crate::baseobjspace::c_uid_t_w(args[1])?;
                    let s = crate::baseobjspace::c_uid_t_w(args[2])?;
                    host_posix::setresgid(r, e, s).map_err(|e| io_err(e, ""))?;
                    Ok(pyre_object::w_none())
                },
                3,
            ),
        );
    }

    // ── the descriptor calls Windows serves through the C runtime (the same
    //    noop placeholders, overridden) ───────────────────────────────────
    #[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
    {
        use rustpython_host_env::os::ErrorExt;
        use rustpython_host_env::{crt_fd, nt as host_nt};

        /// A failed C runtime call carries its errno as the error's payload
        /// rather than as a raw OS code, which is what `posix_errno` reads.
        fn crt_errno_of(e: &std::io::Error) -> i32 {
            e.posix_errno()
        }

        /// The descriptor an fd argument names, for the runtime calls that
        /// take one. `-1` is the sentinel for "no descriptor", never a real
        /// one, so it is refused before the call sees it.
        fn borrowed_fd(w_fd: PyObjectRef) -> Result<crt_fd::Borrowed<'static>, crate::PyError> {
            let fd = crate::baseobjspace::c_int_w(w_fd)?;
            unsafe { crt_fd::Borrowed::try_borrow_raw(fd) }
                .map_err(|e| errno_err(crt_errno_of(&e), ""))
        }

        fn crt_result(result: std::io::Result<()>) -> Result<PyObjectRef, crate::PyError> {
            match result {
                Ok(()) => Ok(pyre_object::w_none()),
                Err(e) => Err(errno_err(crt_errno_of(&e), "")),
            }
        }

        /// The Win32 error a handle call failed with.  A descriptor that names
        /// no handle never reaches such a call, and `ERROR_INVALID_HANDLE` is
        /// what the caller reports in its place.
        fn handle_err(e: &std::io::Error) -> crate::PyError {
            let winerror = e
                .raw_os_error()
                .unwrap_or(windows_sys::Win32::Foundation::ERROR_INVALID_HANDLE as i32);
            crate::PyError::os_error_win32_syscall2(
                winerror,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
            )
        }

        /// The descriptor's handle, or `None` when it names none.
        fn fd_handle(fd: i32) -> Option<windows_sys::Win32::Foundation::HANDLE> {
            let handle = host_nt::handle_from_fd(fd);
            (!handle.is_null() && handle != windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE)
                .then_some(handle)
        }

        // os.dup(fd) -> new_fd.  `_Py_dup` makes the copy non-inheritable, so
        // it does not leak into a child the way the CRT's own copy would.
        crate::module_ns_store(
            ns,
            "dup",
            crate::make_builtin_function_with_arity(
                "dup",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("dup() requires 1 argument"));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    match host_nt::dup(fd) {
                        Ok(n) => Ok(pyre_object::w_int_new(n as i64)),
                        Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                    }
                },
                1,
            ),
        );

        // os.dup2(fd, fd2, inheritable=True) -> fd2 — the `Signature`-bearing
        // twin of the unix registration, and defective in the same way while
        // it was registered raw.
        #[crate::pyre_function]
        fn dup2(
            fd: pyre_object::PyObjectRef,
            fd2: pyre_object::PyObjectRef,
            inheritable: Option<pyre_object::PyObjectRef>,
        ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
            let fd = crate::baseobjspace::c_int_w(fd)?;
            let fd2 = crate::baseobjspace::c_int_w(fd2)?;
            let inheritable = match inheritable {
                Some(w) => crate::baseobjspace::is_true(w)?,
                None => true,
            };
            match host_nt::dup2(fd, fd2, inheritable) {
                Ok(n) => Ok(pyre_object::w_int_new(n as i64)),
                Err(e) => Err(errno_err(crt_errno_of(&e), "")),
            }
        }

        crate::module_ns_store(
            ns,
            "dup2",
            crate::make_builtin_function_with_arity_and_maybe_sig(
                "dup2",
                dup2,
                dup2_pyre_arity(),
                dup2_pyre_sig(),
            ),
        );

        // os.fsync(fd) — `_commit`, the runtime's flush-to-disk.
        crate::module_ns_store(
            ns,
            "fsync",
            crate::make_builtin_function_with_arity(
                "fsync",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("fsync() requires 1 argument"));
                    }
                    crt_result(crt_fd::fsync(borrowed_fd(args[0])?))
                },
                1,
            ),
        );

        // os.ftruncate(fd, length) — `_chsize_s`.
        crate::module_ns_store(
            ns,
            "ftruncate",
            crate::make_builtin_function_with_arity(
                "ftruncate",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "ftruncate() requires 2 arguments",
                        ));
                    }
                    let length = truncate_length_w(args[1])?;
                    crt_result(crt_fd::ftruncate(borrowed_fd(args[0])?, length))
                },
                2,
            ),
        );

        // os.truncate(path, length) — `_wopen` then the same `_chsize_s`
        // (`os_truncate_impl`).  Its path is `path_t(allow_fd=…)`, so an
        // integer names an open descriptor and the call is `ftruncate` on it,
        // with no name to report the failure with.
        crate::module_ns_store(
            ns,
            "truncate",
            crate::make_builtin_function_with_arity(
                "truncate",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "truncate() requires 2 arguments",
                        ));
                    }
                    let path = crate::gateway::fsencode_path_or_fd_w(args[0], "truncate", true)?;
                    let length = truncate_length_w(args[1])?;
                    if path.as_fd != -1 {
                        let bfd = unsafe { crt_fd::Borrowed::borrow_raw(path.as_fd) };
                        return crt_result(crt_fd::ftruncate(bfd, length));
                    }
                    let name = |e: &std::io::Error| {
                        errno_err_with_filename(crt_errno_of(e), path.w_path())
                    };
                    let wide = wide_path(&path.as_bytes)?;
                    let flags = libc::O_WRONLY | libc::O_BINARY | libc::O_NOINHERIT;
                    let fd = crt_fd::wopen(&wide, flags, 0).map_err(|e| name(&e))?;
                    let result = crt_fd::ftruncate(fd.borrow(), length);
                    let closed = crt_fd::close(fd);
                    result.and(closed).map_err(|e| name(&e))?;
                    Ok(pyre_object::w_none())
                },
                2,
            ),
        );

        // os.chdir(path) — `SetCurrentDirectoryW`, which is what
        // `std::env::set_current_dir` is here.
        crate::module_ns_store(
            ns,
            "chdir",
            crate::make_builtin_function_with_arity(
                "chdir",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("chdir() requires 1 argument"));
                    }
                    // The POSIX `chdir` names `integer` in its allowed types
                    // because it can `fchdir`; there is none here, so the list
                    // this one shows is the path-only one.
                    let path = crate::gateway::fsencode_path_named_w(args[0], "chdir", "path")?;
                    std::env::set_current_dir(path_from_bytes(&path.as_bytes).as_ref())
                        .map_err(|e| fs_err_with_filename(e, path.w_path()))?;
                    Ok(pyre_object::w_none())
                },
                1,
            ),
        );

        // os.access(path, mode, *, dir_fd=None, effective_ids=False,
        //           follow_symlinks=True) -> bool
        //
        // Windows has no permission bits to consult beyond the read-only
        // attribute, so `W_OK` is the only mode that can answer False for a
        // name that exists (`os_access_impl`).  None of the three modifiers
        // has a call to reach: `dir_fd` types as `dir_fd(requires='faccessat')`
        // and the other two are the pair `os_access_impl` turns away without
        // `faccessat`, so each is refused rather than answered as though it
        // had been applied.
        crate::module_ns_store(
            ns,
            "access",
            crate::make_builtin_function("access", |args| {
                // The three modifiers are keyword-only, so a third positional
                // is an error rather than a `dir_fd`.
                let (bound, kwargs) = bind_path_args(
                    args,
                    "access",
                    &["path", "mode"],
                    2,
                    &["dir_fd", "effective_ids", "follow_symlinks"],
                )?;
                // The parameters convert in declaration order and each of them
                // can raise, so the order is observable: `path` reports before
                // `mode`, and both before either flag's `__bool__` is called.
                let path = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("path is required"),
                    "access",
                    "path",
                )?;
                // Only `W_OK` is read, so the byte holding it is the whole of
                // the mode as far as the answer goes.
                let mode = crate::baseobjspace::c_int_w(bound[1].expect("mode is required"))? as u8;
                dir_fd_kwarg(kwargs, false)?;
                if let Some(v) = crate::builtins::kwarg_get(kwargs, "effective_ids")
                    && crate::baseobjspace::is_true(v)?
                {
                    return Err(argument_unavailable("access", "effective_ids"));
                }
                if let Some(v) = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                    && !crate::baseobjspace::is_true(v)?
                {
                    return Err(argument_unavailable("access", "follow_symlinks"));
                }
                Ok(pyre_object::w_bool_from(host_nt::access(
                    path_from_bytes(&path.as_bytes).as_ref(),
                    mode,
                )))
            }),
        );

        // os.execv(path, argv) / os.execve(path, argv, env)
        //
        // `_wexecv` / `_wexecve` are the wide forms `os_execv_impl` reaches
        // for; they return only on failure, because on success the calling
        // process is gone by the time they would.
        fn exec_argv_wide(
            w_argv: PyObjectRef,
            function: &str,
        ) -> Result<Vec<widestring::WideCString>, crate::PyError> {
            let items = crate::baseobjspace::unpackiterable(w_argv, -1).map_err(|error| {
                if error.kind == crate::PyErrorKind::TypeError {
                    crate::PyError::type_error(format!(
                        "{function}() arg 2 must be an iterable of strings"
                    ))
                } else {
                    error
                }
            })?;
            if items.is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 must not be empty"
                )));
            }
            let mut argv = Vec::with_capacity(items.len());
            for item in items {
                // An element is converted on the sequence's behalf, not as an
                // argument of the call, so the caller-less message is the one
                // it reports — the same for the environment below.
                let value = extract_path(item)?;
                argv.push(
                    widestring::WideCString::from_os_str(&*os_str_from_bytes(&value)).map_err(
                        |_| {
                            crate::PyError::value_error(format!(
                                "{function}() arg 2 contains an embedded null byte"
                            ))
                        },
                    )?,
                );
            }
            if argv[0].is_empty() {
                return Err(crate::PyError::value_error(format!(
                    "{function}() arg 2 first element cannot be empty"
                )));
            }
            Ok(argv)
        }

        fn exec_pointer_array_wide(values: &[widestring::WideCString]) -> Vec<*const u16> {
            let mut pointers: Vec<_> = values.iter().map(|value| value.as_ptr()).collect();
            pointers.push(std::ptr::null());
            pointers
        }

        crate::module_ns_store(
            ns,
            "execv",
            crate::make_builtin_function_with_arity(
                "execv",
                |args| {
                    // The path names itself; the argv entries do not, because
                    // each of those is converted on the sequence's behalf
                    // rather than as an argument of its own.
                    let command =
                        crate::gateway::fsencode_path_named_w(args[0], "execv", "path")?.as_bytes;
                    let command_w = wide_path(&command)?;
                    let argv = exec_argv_wide(args[1], "execv")?;
                    let argv_ptrs = exec_pointer_array_wide(&argv);
                    unsafe { libc::wexecv(command_w.as_ptr(), argv_ptrs.as_ptr()) };
                    // `wrap_oserror` names no file, so the path stays out of
                    // the error.
                    Err(io_err(std::io::Error::last_os_error(), ""))
                },
                2,
            ),
        );

        crate::module_ns_store(
            ns,
            "execve",
            crate::make_builtin_function_with_arity(
                "execve",
                |args| {
                    let command =
                        crate::gateway::fsencode_path_named_w(args[0], "execve", "path")?.as_bytes;
                    let command_w = wide_path(&command)?;
                    let argv = exec_argv_wide(args[1], "execve")?;
                    let argv_ptrs = exec_pointer_array_wide(&argv);

                    let env = collect_env_entries(args[2], "execve", false)?
                        .into_iter()
                        .map(|entry| {
                            widestring::WideCString::from_os_str(&*os_str_from_bytes(&entry))
                                .map_err(|_| {
                                    crate::PyError::value_error(
                                        "execve() environment contains an embedded null byte",
                                    )
                                })
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let env_ptrs = exec_pointer_array_wide(&env);
                    unsafe {
                        libc::wexecve(command_w.as_ptr(), argv_ptrs.as_ptr(), env_ptrs.as_ptr())
                    };
                    Err(io_err(std::io::Error::last_os_error(), ""))
                },
                3,
            ),
        );

        /// The mode bit Windows keeps: with the owner's write bit the
        /// read-only attribute comes off, without it goes on.
        const S_IWRITE: u32 = 0o200;

        // os.chmod(path, mode, *, dir_fd=None, follow_symlinks=True) -> None
        //
        // `MS_WINDOWS` is what puts this name in `supports_fd` (`os.py:143`)
        // and in `supports_follow_symlinks` (`os.py`), so all three forms
        // are served here: a descriptor through the handle it wraps, a name
        // through the file the link resolves to, and `follow_symlinks=False`
        // through the link's own attributes.  `dir_fd` is the one modifier
        // Windows cannot honour — `chmod` types it as
        // `dir_fd(requires='fchmodat')`, which is `_DirFD_Unavailable`.
        crate::module_ns_store(
            ns,
            "chmod",
            crate::make_builtin_function("chmod", |args| {
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["dir_fd", "follow_symlinks"],
                    "chmod",
                )?;
                if args.len() < 2 {
                    return Err(crate::PyError::type_error("chmod() requires 2 arguments"));
                }
                // Both modifiers are keyword-only.
                if args.len() > 2 {
                    return Err(crate::PyError::type_error(format!(
                        "chmod() takes exactly 2 positional arguments ({} given)",
                        args.len()
                    )));
                }
                // `_DirFD_Unavailable.unwrap` (`interp_posix.py`)
                // converts first and reports the platform second, so a wrongly
                // typed value is a TypeError here as well.
                if let Some(w) = crate::builtins::kwarg_get(kwargs, "dir_fd")
                    .filter(|&w| !unsafe { pyre_object::is_none(w) })
                {
                    unwrap_fd(w, "integer or None")?;
                    return Err(dir_fd_unavailable());
                }
                let path = crate::gateway::fsencode_path_or_fd_w(args[0], "chmod", MS_WINDOWS)?;
                // `posix.chmod` unwraps `mode` as `c_int`, so a non-integer
                // raises TypeError instead of reinterpreting its layout.
                let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                // The descriptor form has no name to resolve, so neither
                // modifier applies to it and it dispatches straight to
                // `os.fchmod` (`interp_posix.py`).
                if path.as_fd != -1 {
                    return match host_nt::fchmod(path.as_fd, mode, S_IWRITE) {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(e) => Err(handle_err(&e)),
                    };
                }
                let follow_symlinks = match crate::builtins::kwarg_get(kwargs, "follow_symlinks") {
                    Some(w) => crate::baseobjspace::is_true(w)?,
                    None => true,
                };
                let wide = wide_path(&path.as_bytes)?;
                let result = if follow_symlinks {
                    host_nt::chmod_follow(&wide, mode, S_IWRITE)
                } else {
                    // `SetFileAttributesW` on the name itself, which is what
                    // "modify the link rather than its target" means where the
                    // mode is one attribute bit.
                    host_nt::win32_lchmod(&wide, mode, S_IWRITE)
                };
                result.map_err(|e| fs_err_with_filename(e, path.w_path()))?;
                Ok(pyre_object::w_none())
            }),
        );

        // os.fchmod(fd, mode) -> None
        crate::module_ns_store(
            ns,
            "fchmod",
            crate::make_builtin_function_with_arity(
                "fchmod",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("fchmod() requires 2 arguments"));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let mode = crate::baseobjspace::c_int_w(args[1])? as u32;
                    // Every failure here is the handle call's, reported the
                    // Win32 way (`os_fchmod_impl`).
                    match host_nt::fchmod(fd, mode, S_IWRITE) {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(e) => Err(handle_err(&e)),
                    }
                },
                2,
            ),
        );

        // os.link(src, dst) -> None.  `CreateHardLinkW` names the new link
        // first and the file it points at second.
        crate::module_ns_store(
            ns,
            "link",
            crate::make_builtin_function("link", |args| {
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["src_dir_fd", "dst_dir_fd", "follow_symlinks"],
                    "link",
                )?;
                // No `linkat` here, so a descriptor to resolve either name
                // against is refused rather than ignored — and the message
                // names both arguments, the way `argument_unavailable_error`
                // spells this one.
                if ["src_dir_fd", "dst_dir_fd"].iter().any(|name| {
                    crate::builtins::kwarg_get(kwargs, name)
                        .is_some_and(|w| !unsafe { pyre_object::is_none(w) })
                }) {
                    return Err(crate::PyError::not_implemented(
                        "link: src_dir_fd and dst_dir_fd unavailable on this platform",
                    ));
                }
                // `CreateHardLinkW` links the symlink `src` names rather than
                // what it points at, so asking for the other behaviour is
                // refused.  Leaving the argument out is not asking: the
                // default is the unspecified one.
                if let Some(w) = crate::builtins::kwarg_get(kwargs, "follow_symlinks")
                    && crate::baseobjspace::is_true(w)?
                {
                    return Err(crate::PyError::not_implemented(
                        "link: follow_symlinks=True unavailable on this platform",
                    ));
                }
                link_positional(args)?;
                let src = crate::gateway::fsencode_path_named_w(args[0], "link", "src")?;
                let dst = crate::gateway::fsencode_path_named_w(args[1], "link", "dst")?;
                let (wide_src, wide_dst) = (wide_path(&src.as_bytes)?, wide_path(&dst.as_bytes)?);
                let ok = unsafe {
                    windows_sys::Win32::Storage::FileSystem::CreateHardLinkW(
                        wide_dst.as_ptr(),
                        wide_src.as_ptr(),
                        std::ptr::null(),
                    )
                };
                if ok == 0 {
                    return Err(fs_err_with_filename2(
                        std::io::Error::last_os_error(),
                        0,
                        src.w_path(),
                        dst.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.symlink(src, dst, target_is_directory=False) -> None.
        // `CreateSymbolicLinkW` names the link first and its target second,
        // and a link to a directory is a different kind of reparse point from
        // a link to a file — which is what `target_is_directory` picks, since
        // the target need not exist yet for the link to be made.
        crate::module_ns_store(
            ns,
            "symlink",
            crate::make_builtin_function("symlink", |args| {
                let (bound, kwargs) = bind_path_args(
                    args,
                    "symlink",
                    &["src", "dst", "target_is_directory"],
                    2,
                    &["dir_fd"],
                )?;
                // `symlink` types `dir_fd` as `DirFD(rposix.HAVE_SYMLINKAT)`,
                // and `CreateSymbolicLinkW` resolves a relative name against
                // the working directory alone.
                dir_fd_kwarg(kwargs, false)?;
                let src = crate::gateway::fsencode_path_named_w(
                    bound[0].expect("src is required"),
                    "symlink",
                    "src",
                )?;
                let dst = crate::gateway::fsencode_path_named_w(
                    bound[1].expect("dst is required"),
                    "symlink",
                    "dst",
                )?;
                let target_is_directory = match bound[2] {
                    Some(w) => crate::baseobjspace::is_true(w)?,
                    None => false,
                };
                let (wide_src, wide_dst) = (wide_path(&src.as_bytes)?, wide_path(&dst.as_bytes)?);
                // The kind is not read off `target_is_directory` alone.
                // `os_symlink_impl` also asks `_check_dirW`, which resolves a
                // relative `src` against `dirname(dst)` and reports whether
                // `GetFileAttributesExW` calls it a directory — so
                // `os.symlink(os.path.join('a', 'bcd'), 'sym3')` makes a link
                // that can be walked into rather than a file link nothing can
                // list.  Creating either is a privilege the account may not
                // hold, so the call asks for the developer-mode path and
                // retries without it on the Windows too old to know the flag.
                // The host wrapper carries both, including the
                // `windows_has_symlink_unprivileged_flag` latch that stops
                // paying for the first attempt once it is known to fail.
                rustpython_host_env::nt::symlink(
                    &path_from_bytes(&src.as_bytes),
                    &path_from_bytes(&dst.as_bytes),
                    &wide_src,
                    &wide_dst,
                    target_is_directory,
                )
                .map_err(|e| fs_err_with_filename2(e, 0, src.w_path(), dst.w_path()))?;
                Ok(pyre_object::w_none())
            }),
        );

        // os.umask(mask) -> previous mask
        crate::module_ns_store(
            ns,
            "umask",
            crate::make_builtin_function_with_arity(
                "umask",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("umask() requires 1 argument"));
                    }
                    let mask = crate::baseobjspace::c_int_w(args[0])?;
                    match host_nt::umask(mask) {
                        Ok(previous) => Ok(pyre_object::w_int_new(previous as i64)),
                        Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                    }
                },
                1,
            ),
        );

        // os.pipe() -> (read_fd, write_fd), both non-inheritable.
        crate::module_ns_store(
            ns,
            "pipe",
            crate::make_builtin_function_with_arity(
                "pipe",
                |_| match host_nt::pipe() {
                    Ok((read_fd, write_fd)) => Ok(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(read_fd as i64),
                        pyre_object::w_int_new(write_fd as i64),
                    ])),
                    Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                },
                0,
            ),
        );

        // os.getppid() — the parent recorded in the process's own entry, which
        // Windows only offers through a snapshot of the process list.
        crate::module_ns_store(
            ns,
            "getppid",
            crate::make_builtin_function_with_arity(
                "getppid",
                |_| Ok(pyre_object::w_int_new(host_nt::getppid() as i64)),
                0,
            ),
        );

        // os.getlogin() -> str
        crate::module_ns_store(
            ns,
            "getlogin",
            crate::make_builtin_function_with_arity(
                "getlogin",
                |_| match host_nt::getlogin() {
                    Ok(name) => Ok(pyre_object::w_str_new(&name)),
                    Err(e) => Err(fs_err_with_filename2(
                        e,
                        0,
                        pyre_object::PY_NULL,
                        pyre_object::PY_NULL,
                    )),
                },
                0,
            ),
        );

        // os.startfile(path, operation=None, arguments=None, cwd=None,
        // show_cmd=None) -> None.  `ShellExecuteW` hands the file to whatever
        // program is registered for it, and reports failure by returning 32 or
        // less rather than through a flag.
        crate::module_ns_store(
            ns,
            "startfile",
            crate::make_builtin_function("startfile", |args| {
                use windows_sys::Win32::UI::Shell::ShellExecuteW;
                use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

                // Every optional argument is positional-or-keyword, so the
                // four of them are looked up either way round.
                //
                // `filepath` and `cwd` convert through the caller-less form:
                // this entry point exists only here, so the wording it should
                // name itself with is unmeasured on this host. See the
                // follow-up task.
                let (args, kwargs) = crate::builtins::split_builtin_kwargs(args);
                crate::builtins::kwarg_reject_unknown(
                    kwargs,
                    &["operation", "arguments", "cwd", "show_cmd"],
                    "startfile",
                )?;
                if args.is_empty() {
                    return Err(crate::PyError::type_error(
                        "startfile() missing required argument 'filepath' (pos 1)",
                    ));
                }
                let path = crate::gateway::fsencode_path_w(args[0])?;
                let wide_file = wide_path(&path.as_bytes)?;
                let given = |index: usize, name: &str| -> Option<pyre_object::PyObjectRef> {
                    args.get(index)
                        .copied()
                        .or_else(|| crate::builtins::kwarg_get(kwargs, name))
                        .filter(|&w| !unsafe { pyre_object::is_none(w) })
                };
                // A missing optional argument is spelled `None`, which stands
                // for the null the call takes for "no operation", "no
                // arguments", "the process's own directory".
                let wide_arg = |index: usize, name: &str| -> Result<Option<_>, crate::PyError> {
                    match given(index, name) {
                        Some(w) => {
                            let text = crate::baseobjspace::text_w(w)?;
                            Ok(Some(widestring::WideCString::from_str(text).map_err(
                                |_| crate::PyError::value_error("embedded null character"),
                            )?))
                        }
                        None => Ok(None),
                    }
                };
                let operation = wide_arg(1, "operation")?;
                let arguments = wide_arg(2, "arguments")?;
                let cwd = match given(3, "cwd") {
                    Some(w) => Some(wide_path(&crate::gateway::fsencode_path_w(w)?.as_bytes)?),
                    None => None,
                };
                let show_cmd = match given(4, "show_cmd") {
                    Some(w) => crate::baseobjspace::c_int_w(w)?,
                    None => SW_SHOWNORMAL as i32,
                };
                let as_ptr = |wide: &Option<widestring::WideCString>| {
                    wide.as_ref().map_or(std::ptr::null(), |w| w.as_ptr())
                };
                let rc = unsafe {
                    ShellExecuteW(
                        std::ptr::null_mut(),
                        as_ptr(&operation),
                        wide_file.as_ptr(),
                        as_ptr(&arguments),
                        as_ptr(&cwd),
                        show_cmd,
                    )
                };
                if rc as isize <= 32 {
                    return Err(fs_err_with_filename(
                        std::io::Error::last_os_error(),
                        path.w_path(),
                    ));
                }
                Ok(pyre_object::w_none())
            }),
        );

        // os.cpu_count() -> int | None
        //
        // `rposix.py:2978-2986` reads `GetSystemInfo().dwNumberOfProcessors`
        // here, which counts the processors in the caller's processor group;
        // `available_parallelism` answers the process affinity mask instead, so
        // the two part company on a host that has restricted one. Left as it is
        // because no Windows oracle is reachable from this host to measure
        // which the surface should report — see the follow-up task.
        crate::module_ns_store(
            ns,
            "cpu_count",
            crate::make_builtin_function_with_arity(
                "cpu_count",
                |_| match std::thread::available_parallelism() {
                    Ok(n) => Ok(pyre_object::w_int_new(n.get() as i64)),
                    Err(_) => Ok(pyre_object::w_none()),
                },
                0,
            ),
        );

        // os.system(command) -> the command interpreter's exit status.  The
        // wide entry point is the one that can spell every command; the narrow
        // one re-encodes it through the ANSI code page.  Neither reports a
        // failure other than through the status, which `os_system_impl`
        // returns as it is.
        //
        // The POSIX `system` converts its command as a filesystem name and so
        // reports the caller-less message — measured. This one declares text
        // rather than a path, so the message it should report is a different
        // shape entirely and is unmeasured here; it keeps the same conversion
        // meanwhile. See the follow-up task.
        crate::module_ns_store(
            ns,
            "system",
            crate::make_builtin_function_with_arity(
                "system",
                |args| {
                    unsafe extern "C" {
                        fn _wsystem(command: *const u16) -> libc::c_int;
                    }
                    if args.is_empty() {
                        return Err(crate::PyError::type_error("system() requires 1 argument"));
                    }
                    let command = crate::gateway::fsencode_path_w(args[0])?;
                    let wide = wide_path(&command.as_bytes)?;
                    let status = crate::builtins::crt_call!(_wsystem(wide.as_ptr()));
                    Ok(pyre_object::w_int_new(status as i64))
                },
                1,
            ),
        );

        // os.waitpid(pid, options) -> (pid, status).  `_cwait` waits for one
        // process by handle; the status it reports is the exit code, which
        // `os_waitpid_impl` shifts into the byte a POSIX wait status keeps it
        // in.
        crate::module_ns_store(
            ns,
            "waitpid",
            crate::make_builtin_function_with_arity(
                "waitpid",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error("waitpid() requires 2 arguments"));
                    }
                    let pid = crate::baseobjspace::int_w(args[0])? as isize;
                    let options = crate::baseobjspace::c_int_w(args[1])?;
                    match host_nt::cwait(pid, options) {
                        Ok((pid, status)) => Ok(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(pid as i64),
                            pyre_object::w_int_new((status as i64) << 8),
                        ])),
                        Err(e) => Err(errno_err(crt_errno_of(&e), "")),
                    }
                },
                2,
            ),
        );

        // os.times() -> posix.times_result.  Windows keeps the process's own
        // user and kernel time and nothing else, so the three fields that
        // count a child's are zero (`os_times_impl`).
        crate::module_ns_store(
            ns,
            "times",
            crate::make_builtin_function_with_arity(
                "times",
                |_| {
                    let times =
                        rustpython_host_env::time::get_process_times_100ns().ok_or_else(|| {
                            fs_err_with_filename(
                                std::io::Error::last_os_error(),
                                pyre_object::PY_NULL,
                            )
                        })?;
                    // `GetProcessTimes` counts in hundreds of nanoseconds.
                    let seconds = |ticks: u64| pyre_object::w_float_new(ticks as f64 * 1e-7);
                    Ok(crate::_structseq::new_instance(
                        times_result_seq_type(),
                        vec![
                            seconds(times.user),
                            seconds(times.system),
                            pyre_object::w_float_new(0.0),
                            pyre_object::w_float_new(0.0),
                            pyre_object::w_float_new(0.0),
                        ],
                    ))
                },
                0,
            ),
        );

        // os.listdrives() / os.listvolumes() / os.listmounts(volume) — the
        // names Windows mounts its filesystems under.
        fn name_list(
            names: std::io::Result<Vec<std::ffi::OsString>>,
        ) -> Result<PyObjectRef, crate::PyError> {
            let names = names.map_err(|e| fs_err_with_filename(e, pyre_object::PY_NULL))?;
            Ok(pyre_object::w_list_new(
                names
                    .iter()
                    .map(|name| fs_name_obj(false, name.as_encoded_bytes()))
                    .collect(),
            ))
        }
        crate::module_ns_store(
            ns,
            "listdrives",
            crate::make_builtin_function_with_arity(
                "listdrives",
                |_| name_list(host_nt::listdrives()),
                0,
            ),
        );
        crate::module_ns_store(
            ns,
            "listvolumes",
            crate::make_builtin_function_with_arity(
                "listvolumes",
                |_| name_list(host_nt::listvolumes()),
                0,
            ),
        );
        // `volume` converts through the caller-less form: 3.14 added this entry
        // point on Windows alone, so what it names itself with is unmeasured on
        // this host. See the follow-up task.
        crate::module_ns_store(
            ns,
            "listmounts",
            crate::make_builtin_function_with_arity(
                "listmounts",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "listmounts() requires 1 argument",
                        ));
                    }
                    let volume = crate::gateway::fsencode_path_w(args[0])?;
                    name_list(host_nt::listmounts(&wide_path(&volume.as_bytes)?))
                },
                1,
            ),
        );

        // os.device_encoding(fd) -> str | None.  `_Py_device_encoding`: only a
        // terminal has one, and a process with no console attached has no code
        // page to name it with.
        crate::module_ns_store(
            ns,
            "device_encoding",
            crate::make_builtin_function_with_arity(
                "device_encoding",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "device_encoding() requires 1 argument",
                        ));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    if !host_os::isatty(fd) {
                        return Ok(pyre_object::w_none());
                    }
                    match host_os::device_encoding(fd) {
                        // `GetConsoleCP` answers 0 for a process with no console.
                        Some(name) if name != "cp0" => Ok(pyre_object::w_str_new(&name)),
                        _ => Ok(pyre_object::w_none()),
                    }
                },
                1,
            ),
        );

        // os.get_inheritable(fd) / os.set_inheritable(fd, inheritable) — the
        // flag lives on the descriptor's handle (`HANDLE_FLAG_INHERIT`).
        crate::module_ns_store(
            ns,
            "get_inheritable",
            crate::make_builtin_function_with_arity(
                "get_inheritable",
                |args| {
                    if args.is_empty() {
                        return Err(crate::PyError::type_error(
                            "get_inheritable() requires 1 argument",
                        ));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let handle = fd_handle(fd).ok_or_else(|| errno_err(libc::EBADF, ""))?;
                    match host_nt::get_handle_inheritable(handle as _) {
                        Ok(inheritable) => Ok(pyre_object::w_bool_from(inheritable)),
                        Err(e) => Err(handle_err(&e)),
                    }
                },
                1,
            ),
        );
        crate::module_ns_store(
            ns,
            "set_inheritable",
            crate::make_builtin_function_with_arity(
                "set_inheritable",
                |args| {
                    if args.len() < 2 {
                        return Err(crate::PyError::type_error(
                            "set_inheritable() requires 2 arguments",
                        ));
                    }
                    let fd = crate::baseobjspace::c_int_w(args[0])?;
                    let inherit = crate::baseobjspace::is_true(args[1])?;
                    let handle = fd_handle(fd).ok_or_else(|| errno_err(libc::EBADF, ""))?;
                    match host_nt::set_handle_inheritable(handle as _, inherit) {
                        Ok(()) => Ok(pyre_object::w_none()),
                        Err(e) => Err(handle_err(&e)),
                    }
                },
                2,
            ),
        );
    }

    // The trampoline only mediates the curated ll_os/ll_time surface, so the
    // real impls registered above for process control, fd duplication, host
    // filesystem mutation and privilege changes would otherwise reach libc
    // directly under sandbox.  Overwrite each with a raising stub, mirroring the
    // RPython sandbox where unsupported externals are simply unavailable.  The
    // mediated names (open/read/write/close/lseek/stat/access/getcwd/listdir/
    // getenv/isatty/strerror/get{u,g}id/unlink/mkdir) are intentionally absent
    // here — they stay live through host_seam.
    #[cfg(feature = "sandbox")]
    {
        fn sandbox_unavailable(_: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            Err(crate::host_seam::stub("this OS operation"))
        }
        for name in [
            // process creation / control
            "fork",
            "forkpty",
            "system",
            "execv",
            "execve",
            "execvp",
            "execvpe",
            // Neither the spawn family nor `popen` is an external here: os.py
            // writes both in Python, over fork+exec+waitpid (:881) and over
            // `subprocess` (:1020), and the stubs above already refuse what
            // they reach for. Binding those names would only stop os.py from
            // defining them — and with the spawn family, P_WAIT and P_NOWAIT.
            "posix_spawn",
            "posix_spawnp",
            "abort",
            "_exit",
            "register_at_fork",
            "wait",
            "waitpid",
            "kill",
            "killpg",
            // file-descriptor duplication / pipes / ttys / cross-fd copy +
            // inheritance control (set_inheritable would mutate a real fd).
            "dup",
            "dup2",
            "pipe",
            "openpty",
            "login_tty",
            "sendfile",
            "set_inheritable",
            // host filesystem mutation that bypasses the controller
            "chmod",
            "fchmod",
            "lchmod",
            "chown",
            "fchown",
            "lchown",
            "chroot",
            "chdir",
            "fchdir",
            "link",
            "symlink",
            "truncate",
            "ftruncate",
            "rename",
            "replace",
            "rmdir",
            "mkfifo",
            "mknod",
            // privilege / scheduling
            "setuid",
            "seteuid",
            "setgid",
            "setegid",
            "setreuid",
            "setregid",
            "setresuid",
            "setresgid",
            "setgroups",
            "initgroups",
            "setsid",
            "setpgid",
            "setpgrp",
            "nice",
            "setpriority",
            "sched_get_priority_max",
            "sched_get_priority_min",
            // durability + real process environment mutation
            "sync",
            "fsync",
            "fdatasync",
            "setenv",
            "unsetenv",
            "putenv",
            // host filesystem inspection that bypasses the controller VFS.
            // DirEntry is a type, but its is_dir/is_file/stat/inode methods stat
            // a guest-controlled `path` via host_fs, so neutralise it too (its
            // only producer, scandir, is already stubbed here).
            "readlink",
            "scandir",
            "DirEntry",
            "statvfs",
            "fstatvfs",
            // host process / environment information leaks
            "getpid",
            "getppid",
            "uname",
            "getlogin",
            "getloadavg",
            "getpriority",
            "times",
            "umask",
            "getgroups",
            "cpu_count",
            "_cpu_count",
            "getresuid",
            "getresgid",
            "getpgrp",
            "getpgid",
            // host system-configuration probes; pathconf consults a
            // guest-controlled path on the real filesystem, and confstr
            // answers with the host's own search path among other strings.
            "pathconf",
            "fpathconf",
            "sysconf",
            "confstr",
            // a lock on a descriptor the controller owns
            "lockf",
            // reports on a child of this process, which the sandbox has none of
            "waitid",
            // terminal / tty inspection + control
            "tcgetpgrp",
            "tcsetpgrp",
            "get_terminal_size",
            "ttyname",
            "ctermid",
        ] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, sandbox_unavailable),
            );
        }

        // The same, for the names only some hosts have. Listing them above
        // would not neutralise anything on a host that never registered them —
        // `module_ns_store` writes rather than overwrites, so it would publish
        // a `posix.pipe2` where there is no `pipe2` to refuse.
        #[cfg(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd",
            target_os = "openbsd"
        ))]
        crate::module_ns_store(
            ns,
            "pipe2",
            crate::make_builtin_function("pipe2", sandbox_unavailable),
        );
        // The policy calls reach the host scheduler; only the setters mutate,
        // but a policy read is a host-process leak in the same way `getpriority`
        // above is. `sched_param` is left alone — it carries no host access.
        #[cfg(any(
            target_os = "android",
            target_os = "freebsd",
            target_os = "linux",
            target_os = "netbsd"
        ))]
        for name in [
            "sched_getscheduler",
            "sched_getparam",
            "sched_rr_get_interval",
        ] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, sandbox_unavailable),
            );
        }
        // The affinity mask is the same kind of host-process leak, and carries
        // the narrower gate the pair is published under.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        for name in ["sched_getaffinity", "sched_setaffinity"] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, sandbox_unavailable),
            );
        }
        #[cfg(all(
            not(target_env = "musl"),
            any(
                target_os = "android",
                target_os = "freebsd",
                target_os = "linux",
                target_os = "netbsd"
            )
        ))]
        for name in ["sched_setscheduler", "sched_setparam"] {
            crate::module_ns_store(
                ns,
                name,
                crate::make_builtin_function(name, sandbox_unavailable),
            );
        }
    }

    crate::module_ns_store(ns, "error", crate::typedef::w_object());
}

#[cfg(test)]
mod split_root_tests {
    use super::split_root;
    use rustpython_wtf8::{CodePoint, Wtf8, Wtf8Buf};

    /// Expectations taken from `ntpath.splitroot`, whose three-way split
    /// joins drive and root into the root this returns.
    #[test]
    fn matches_ntpath_splitroot() {
        let cases: &[(&str, &str, &str)] = &[
            ("", "", ""),
            ("Windows", "", "Windows"),
            ("a/b", "", "a/b"),
            ("\\", "\\", ""),
            ("/", "/", ""),
            ("\\Windows", "\\", "Windows"),
            ("/Windows", "/", "Windows"),
            ("C:", "C:", ""),
            ("C:a", "C:", "a"),
            ("C:\\", "C:\\", ""),
            ("C:/Windows", "C:/", "Windows"),
            (":a", "", ":a"),
            ("\\\\server", "\\\\server", ""),
            ("\\\\server\\share", "\\\\server\\share", ""),
            ("\\\\server\\share\\", "\\\\server\\share\\", ""),
            ("\\\\server\\share\\dir", "\\\\server\\share\\", "dir"),
            ("//server/share/dir", "//server/share/", "dir"),
            ("\\\\.\\device\\x", "\\\\.\\device\\", "x"),
            ("\\\\?\\C:\\x", "\\\\?\\C:\\", "x"),
            (
                "\\\\?\\UNC\\server\\share\\dir",
                "\\\\?\\UNC\\server\\share\\",
                "dir",
            ),
            ("//?/unc/server/share/dir", "//?/unc/server/share/", "dir"),
            // No separator after the share name, so the whole path is root.
            ("\\\\\\a", "\\\\\\a", ""),
            // Offsets address characters: the drive branch is taken here.
            ("\u{e4}:\\x", "\u{e4}:\\", "x"),
            (
                "\\\\s\u{e4}rver\\share\\dir",
                "\\\\s\u{e4}rver\\share\\",
                "dir",
            ),
        ];
        for &(path, root, tail) in cases {
            let (got_root, got_tail) = split_root(Wtf8::new(path));
            assert_eq!(
                (got_root.as_str(), got_tail.as_str()),
                (Ok(root), Ok(tail)),
                "split_root({path:?})"
            );
            assert_eq!(
                format!("{root}{tail}"),
                path,
                "split_root({path:?}) lost characters"
            );
        }
    }

    /// A path carrying a lone surrogate — what `fsdecode` produces for an
    /// undecodable name — splits on the same boundary and keeps the code point.
    #[test]
    fn keeps_a_lone_surrogate() {
        let mut path = Wtf8Buf::from_string("C:\\".to_string());
        path.push(CodePoint::from_u32(0xdcff).unwrap());
        path.push_str("x");
        let (root, tail) = split_root(&path);
        assert_eq!(root.as_str(), Ok("C:\\"));
        assert_eq!(tail.code_points().next().map(|c| c.to_u32()), Some(0xdcff));
        assert_eq!(tail.len(), path.len() - root.len());
    }
}
