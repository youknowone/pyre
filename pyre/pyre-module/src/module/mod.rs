//! Builtin modules that are not part of interpreter bootstrap.

#[allow(non_snake_case)]
pub mod _abc;
#[allow(non_snake_case)]
pub mod _bisect;
#[allow(non_snake_case)]
pub mod _blake2;
#[allow(non_snake_case)]
pub mod _bz2;
#[allow(non_snake_case)]
#[cfg(all(
    feature = "host_env",
    not(feature = "sandbox"),
    not(target_arch = "wasm32")
))]
pub mod _cffi_backend;
#[allow(non_snake_case)]
pub mod _codecs_cn;
#[allow(non_snake_case)]
pub mod _codecs_hk;
#[allow(non_snake_case)]
pub mod _codecs_iso2022;
#[allow(non_snake_case)]
pub mod _codecs_jp;
#[allow(non_snake_case)]
pub mod _codecs_kr;
#[allow(non_snake_case)]
pub mod _codecs_tw;
#[allow(non_snake_case)]
pub mod _csv;
#[allow(non_snake_case)]
#[cfg(all(not(feature = "sandbox")))]
pub mod _ctypes;
#[allow(non_snake_case)]
pub mod _functools;
#[allow(non_snake_case)]
pub mod _hashlib;
#[allow(non_snake_case)]
pub mod _heapq;
#[allow(non_snake_case)]
pub mod _immutables_map;
#[allow(non_snake_case)]
pub mod _json;
#[allow(non_snake_case)]
pub mod _lsprof;
#[allow(non_snake_case)]
pub mod _lzma;
#[allow(non_snake_case)]
pub mod _multibytecodec;
#[allow(non_snake_case)]
#[cfg(not(feature = "sandbox"))]
pub mod _multiprocessing;
#[allow(non_snake_case)]
pub mod _opcode;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _overlapped;
#[allow(non_snake_case)]
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod _posixshmem;
#[allow(non_snake_case)]
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod _posixsubprocess;
#[allow(non_snake_case)]
pub mod _pypy_generic_alias;
#[allow(non_snake_case)]
pub mod _queue;
#[allow(non_snake_case)]
#[cfg(target_os = "macos")]
pub mod _scproxy;
#[allow(non_snake_case)]
#[cfg(all(not(feature = "sandbox")))]
pub mod _socket;
#[allow(non_snake_case)]
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod _ssl;
#[allow(non_snake_case)]
pub mod _stat;
#[allow(non_snake_case)]
pub mod _statistics;
#[allow(non_snake_case)]
pub mod _suggestions;
#[allow(non_snake_case)]
pub mod _symtable;
#[allow(non_snake_case)]
pub mod _template;
#[allow(non_snake_case)]
pub mod _tokenize;
#[allow(non_snake_case)]
pub mod _typing;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _uuid;
#[allow(non_snake_case)]
#[cfg(windows)]
pub mod _winapi;
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _wmi;
pub mod binascii;
pub mod cmath;
pub mod errno;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod fcntl;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod grp;
pub mod math;
#[cfg(all(
    not(target_arch = "wasm32"),
    feature = "host_env",
    not(feature = "sandbox")
))]
pub mod mmap;
#[cfg(all(windows, feature = "host_env"))]
pub mod msvcrt;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod pwd;
pub mod pyexpat;
pub mod pypyjit;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod resource;
#[cfg(all(not(feature = "sandbox")))]
pub mod select;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod syslog;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod termios;
pub mod unicodedata;
#[cfg(windows)]
pub mod winreg;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod winsound;
pub mod zlib;
