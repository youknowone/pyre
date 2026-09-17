//! Builtin modules that are not part of interpreter bootstrap.

#[allow(non_snake_case)]
pub mod _bisect;
#[allow(non_snake_case)]
pub mod _blake2;
#[allow(non_snake_case)]
pub mod _bz2;
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
pub mod _heapq;
#[allow(non_snake_case)]
pub mod _immutables_map;
#[allow(non_snake_case)]
pub mod _json;
#[allow(non_snake_case)]
pub mod _lsprof;
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
pub mod _queue;
#[allow(non_snake_case)]
#[cfg(target_os = "macos")]
pub mod _scproxy;
#[allow(non_snake_case)]
pub mod _statistics;
#[allow(non_snake_case)]
pub mod _suggestions;
#[allow(non_snake_case)]
pub mod _template;
#[allow(non_snake_case)]
pub mod _tokenize;
#[allow(non_snake_case)]
#[cfg(all(windows, feature = "host_env", not(feature = "sandbox")))]
pub mod _uuid;
pub mod binascii;
pub mod cmath;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod faulthandler;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod fcntl;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod grp;
pub mod math;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod pwd;
pub mod pyexpat;
#[cfg(all(unix, feature = "host_env", not(feature = "sandbox")))]
pub mod resource;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod syslog;
#[cfg(all(unix, not(feature = "sandbox")))]
pub mod termios;
pub mod zlib;
