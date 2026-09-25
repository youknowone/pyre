//! The call flags of `rpython/rtyper/lltypesystem/rffi.py` that a call
//! descr's `EffectInfo` carries into the JIT runtime.

/// RPython `RFFI_SAVE_ERRNO` and related bit flags (`rffi.py`).
pub const RFFI_SAVE_ERRNO: i64 = 1;
pub const RFFI_READSAVED_ERRNO: i64 = 2;
pub const RFFI_ZERO_ERRNO_BEFORE: i64 = 4;
pub const RFFI_FULL_ERRNO: i64 = RFFI_SAVE_ERRNO | RFFI_READSAVED_ERRNO;
pub const RFFI_FULL_ERRNO_ZERO: i64 = RFFI_SAVE_ERRNO | RFFI_ZERO_ERRNO_BEFORE;
pub const RFFI_SAVE_LASTERROR: i64 = 8;
pub const RFFI_READSAVED_LASTERROR: i64 = 16;
pub const RFFI_SAVE_WSALASTERROR: i64 = 32;
pub const RFFI_FULL_LASTERROR: i64 = RFFI_SAVE_LASTERROR | RFFI_READSAVED_LASTERROR;
pub const RFFI_ERR_NONE: i64 = 0;
pub const RFFI_ERR_ALL: i64 = RFFI_FULL_ERRNO | RFFI_FULL_LASTERROR;
pub const RFFI_ALT_ERRNO: i64 = 64;
