//! OS-level hardening: a seccomp-bpf syscall allowlist for the sandboxed child.
//!
//! The compile-out seam ([`host_seam`](../../pyre_interpreter/host_seam) in the
//! interpreter) routes every *intended* OS access through the marshalling pipe,
//! and the fails-closed `host_seam::sys` facade makes a stray direct `libc::`
//! syscall a compile error. This module is the backstop for everything those
//! source-level mechanisms cannot cover: the linked `host_env` crate, the Rust
//! std library and allocator, a future un-rerouted call site, or a syscall
//! reached by a memory-safety exploit. It is the analog of RPython's
//! `os_level_sandboxing`.
//!
//! [`install_runtime_filter`] installs a classic-BPF program that ALLOWS only a
//! curated set of host-neutral runtime syscalls (memory, signals, time, and I/O
//! on the already-open marshalling fds 0/1/2) and TRAPs anything else to a
//! SIGSYS handler that names the blocked syscall and exits — so
//! `open`/`openat`/`socket`/`connect`/`execve`/`fork`/`clone`/`ptrace` and the
//! rest of the host-affecting surface are simply unreachable. It is
//! installed in the child *after* interpreter startup (which legitimately opens
//! files, allocates, seeds hashing, …) and *before* the first byte of untrusted
//! code, so those startup syscalls run unfiltered while user code does not.
//!
//! Over-listing a benign syscall here cannot widen the escape surface (every
//! listed call is host-neutral); omitting one the runtime needs only
//! over-restricts and kills the child — i.e. it fails in the safe direction.
//!
//! Because pyre has no genc backend, this is the *only* whole-program guarantee
//! in the sandbox — the compile-time seam + clippy fence are selective, not a
//! translation-derived total proof. That makes the boundary Linux-bound: off
//! Linux (or with `PYRE_SANDBOX_NO_SECCOMP`) there is no kernel backstop and
//! containment falls back to seam + fence coverage alone. See the crate root's
//! "Structural constraint" note for the full guarantee model.
#![cfg(target_os = "linux")]

use std::io;

// Classic-BPF instruction encodings (`linux/bpf_common.h`; not in `libc`).
const BPF_LD: u16 = 0x00;
const BPF_W: u16 = 0x00;
const BPF_ABS: u16 = 0x20;
const BPF_JMP: u16 = 0x05;
const BPF_JEQ: u16 = 0x10;
const BPF_K: u16 = 0x00;
const BPF_RET: u16 = 0x06;

const LD_ABS_W: u16 = BPF_LD | BPF_W | BPF_ABS; // load a 32-bit word at an absolute offset
const JEQ_K: u16 = BPF_JMP | BPF_JEQ | BPF_K; // jump-if-equal against an immediate
const RET_K: u16 = BPF_RET | BPF_K; // return an immediate action

// Byte offsets into `struct seccomp_data` (the BPF input): `nr` then `arch`.
const SECCOMP_DATA_NR_OFFSET: u32 = 0;
const SECCOMP_DATA_ARCH_OFFSET: u32 = 4;

// AUDIT_ARCH_* (`linux/audit.h`) = EM_<arch> | __AUDIT_ARCH_64BIT | __AUDIT_ARCH_LE.
#[cfg(target_arch = "x86_64")]
const AUDIT_ARCH: u32 = 0xC000_003E;
#[cfg(target_arch = "aarch64")]
const AUDIT_ARCH: u32 = 0xC000_00B7;

// The SIGSYS branch of `siginfo_t` (linux, LP64). `libc::siginfo_t` does not
// expose `si_syscall`, so the handler reinterprets the leading fields.
#[repr(C)]
struct SigSysSiginfo {
    si_signo: libc::c_int,
    si_errno: libc::c_int,
    si_code: libc::c_int,
    _pad0: libc::c_int,
    si_call_addr: *mut libc::c_void,
    si_syscall: libc::c_int,
    si_arch: libc::c_uint,
}

/// SIGSYS handler for `SECCOMP_RET_TRAP`: write the blocked syscall number to
/// stderr, then exit. Deny still means die, but the kill is no longer silent —
/// the number names exactly which call to add to [`allowed_syscalls`] (or which
/// reroute is missing). Async-signal-safe: only `write(2)`/`_exit(2)`, both
/// allowlisted, and no allocation.
extern "C" fn report_blocked_syscall(
    _signo: libc::c_int,
    info: *mut libc::siginfo_t,
    _ctx: *mut libc::c_void,
) {
    let nr = if info.is_null() {
        -1
    } else {
        unsafe { (*(info as *const SigSysSiginfo)).si_syscall }
    };
    let prefix = b"pyre: sandbox seccomp blocked syscall ";
    let mut buf = [0u8; 64];
    let mut len = prefix.len();
    buf[..len].copy_from_slice(prefix);
    let mut n = nr as i64;
    if n < 0 {
        buf[len] = b'-';
        len += 1;
        n = -n;
    }
    let mut digits = [0u8; 20];
    let mut d = 0;
    loop {
        digits[d] = b'0' + (n % 10) as u8;
        d += 1;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    while d > 0 {
        d -= 1;
        buf[len] = digits[d];
        len += 1;
    }
    buf[len] = b'\n';
    len += 1;
    unsafe {
        libc::write(2, buf.as_ptr() as *const libc::c_void, len);
        libc::_exit(159);
    }
}

fn stmt(code: u16, k: u32) -> libc::sock_filter {
    libc::sock_filter {
        code,
        jt: 0,
        jf: 0,
        k,
    }
}

/// `if nr == syscall { pc += 1 + jt } else { pc += 1 }` — `jt` skips forward to
/// the trailing ALLOW terminator.
fn jeq(syscall: u32, jt: u8) -> libc::sock_filter {
    libc::sock_filter {
        code: JEQ_K,
        jt,
        jf: 0,
        k: syscall,
    }
}

/// Host-neutral syscalls the interpreter runtime, the system allocator and the
/// JIT legitimately issue while untrusted code runs. `libc::SYS_*` are the
/// numbers for THIS compile target; the arch guard in the filter refuses to run
/// it under any other syscall personality, so the numbers cannot be confused.
fn allowed_syscalls() -> Vec<u32> {
    let mut nums: Vec<libc::c_long> = vec![
        // I/O on the already-open marshalling pipe (0/1) + stderr (2); seek/stat/
        // fcntl/positional I/O only ever touch those fds (real file access is
        // marshalled), and dup just clones an already-open fd.
        libc::SYS_read,
        libc::SYS_write,
        libc::SYS_readv,
        libc::SYS_writev,
        libc::SYS_pread64,
        libc::SYS_pwrite64,
        libc::SYS_close,
        libc::SYS_lseek,
        libc::SYS_fstat,
        libc::SYS_fcntl,
        libc::SYS_getcwd,
        libc::SYS_dup,
        libc::SYS_dup3,
        libc::SYS_ppoll,
        // Memory: system malloc (brk/mmap/madvise) + the JIT's executable maps.
        libc::SYS_mmap,
        libc::SYS_munmap,
        libc::SYS_mremap,
        libc::SYS_mprotect,
        libc::SYS_madvise,
        libc::SYS_brk,
        libc::SYS_membarrier,
        // Signals: panic/abort delivery and the runtime's signal scaffolding.
        libc::SYS_rt_sigaction,
        libc::SYS_rt_sigprocmask,
        libc::SYS_rt_sigreturn,
        libc::SYS_rt_sigtimedwait,
        libc::SYS_sigaltstack,
        // Synchronisation, scheduling, hashing entropy.
        libc::SYS_futex,
        libc::SYS_sched_yield,
        libc::SYS_sched_getaffinity,
        libc::SYS_getrandom,
        libc::SYS_set_robust_list,
        libc::SYS_set_tid_address,
        libc::SYS_rseq,
        // Thread creation: the JIT driver spawns one background loop-invalidation
        // thread on first trace (majit `jitdriver.rs`), so `pthread_create` runs
        // after this filter is installed. glibc issues `clone3` on new kernels and
        // falls back to `clone`; allow both. A spawned thread inherits this same
        // filter, so it is confined identically — thread creation stays
        // host-neutral (its stack/sync syscalls are already listed above).
        libc::SYS_clone,
        libc::SYS_clone3,
        // Time (mostly served by the vDSO, but allow the syscall fallbacks).
        libc::SYS_clock_gettime,
        libc::SYS_clock_getres,
        libc::SYS_clock_nanosleep,
        libc::SYS_nanosleep,
        libc::SYS_gettimeofday,
        // Process self-info (read-only) + own resource limits/usage + abort path
        // + clean exit. tkill/tgkill only ever target this single-threaded child.
        libc::SYS_getpid,
        libc::SYS_gettid,
        libc::SYS_getrusage,
        libc::SYS_prlimit64,
        libc::SYS_tkill,
        libc::SYS_tgkill,
        libc::SYS_exit,
        libc::SYS_exit_group,
        libc::SYS_restart_syscall,
    ];
    // x86_64 sets the thread-pointer (TLS) base via arch_prctl; absent on the
    // generic syscall ABI (aarch64), which uses a register convention instead.
    #[cfg(target_arch = "x86_64")]
    nums.push(libc::SYS_arch_prctl);
    nums.iter().map(|&n| n as u32).collect()
}

/// Install the syscall allowlist on the current (single) thread/process. After
/// it returns `Ok`, any syscall outside [`allowed_syscalls`] traps to the
/// SIGSYS handler ([`report_blocked_syscall`]), which writes the blocked
/// syscall number to stderr and exits — deny still means die, just diagnosably.
///
/// Fails-closed: the caller MUST treat an `Err` as fatal and refuse to run
/// untrusted code, since a failed install means the backstop is absent.
pub fn install_runtime_filter() -> io::Result<()> {
    // `gmtime_r` is permitted as a pure calendar call (re-exported by
    // `host_seam::sys`), but glibc loads the timezone database on the first
    // conversion, opening `/etc/localtime` — a direct `openat` the filter below
    // would trap. Prime the cache now, while file opens are still allowed, by
    // running one `gmtime_r` through the exact path the runtime uses; the
    // post-lockdown conversions reuse the in-memory cache and issue no `openat`.
    // (`tzset` alone does not open the zone file when `TZ` is unset in the
    // controller-cleared environment, so drive the real call instead.)
    unsafe {
        let t: libc::time_t = 0;
        let mut tm: libc::tm = core::mem::zeroed();
        libc::gmtime_r(&t, &mut tm);
    }

    // Install the SIGSYS handler before the filter so a denied syscall is
    // reported and exits cleanly rather than dying silently.
    let mut sa: libc::sigaction = unsafe { core::mem::zeroed() };
    let handler: extern "C" fn(libc::c_int, *mut libc::siginfo_t, *mut libc::c_void) =
        report_blocked_syscall;
    sa.sa_sigaction = handler as usize;
    sa.sa_flags = libc::SA_SIGINFO;
    unsafe { libc::sigemptyset(&mut sa.sa_mask) };
    if unsafe { libc::sigaction(libc::SIGSYS, &sa, core::ptr::null_mut()) } != 0 {
        return Err(io::Error::last_os_error());
    }

    let syscalls = allowed_syscalls();
    let n = syscalls.len();
    // Layout (indices): 0 load-arch, 1 arch-check, 2 arch-mismatch kill,
    // 3 load-nr, 4..4+n per-syscall allow checks, 4+n default trap, 4+n+1 allow.
    let mut prog: Vec<libc::sock_filter> = Vec::with_capacity(n + 6);
    prog.push(stmt(LD_ABS_W, SECCOMP_DATA_ARCH_OFFSET));
    // arch == AUDIT_ARCH -> skip the next (kill) instruction; else fall into it.
    prog.push(jeq(AUDIT_ARCH, 1));
    prog.push(stmt(RET_K, libc::SECCOMP_RET_KILL_PROCESS));
    prog.push(stmt(LD_ABS_W, SECCOMP_DATA_NR_OFFSET));
    for (i, &sc) in syscalls.iter().enumerate() {
        // From the check at index 4+i, taking jt lands at 4+i+1+jt; the ALLOW
        // terminator is at 4+n+1, so jt = n - i.
        prog.push(jeq(sc, (n - i) as u8));
    }
    // Default deny: trap to the SIGSYS handler, which names the syscall and
    // exits. (The arch-mismatch path above stays an unconditional kill.)
    prog.push(stmt(RET_K, libc::SECCOMP_RET_TRAP));
    prog.push(stmt(RET_K, libc::SECCOMP_RET_ALLOW));

    // A non-privileged process may only install a filter after NO_NEW_PRIVS, so
    // the filter can never be used to gain privileges via a set-uid exec.
    if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
        return Err(io::Error::last_os_error());
    }
    let fprog = libc::sock_fprog {
        len: prog.len() as libc::c_ushort,
        filter: prog.as_mut_ptr(),
    };
    if unsafe {
        libc::prctl(
            libc::PR_SET_SECCOMP,
            libc::SECCOMP_MODE_FILTER as libc::c_ulong,
            &fprog as *const libc::sock_fprog as libc::c_ulong,
        )
    } != 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
