//! `_uuid` — the private extension `uuid` reads a stable node from.
//!
//! PyPy has no `_uuid` module.  `_uuidmodule.c` builds a libuuid door on POSIX
//! and an rpcrt4 one on Windows, and configure leaves the module out of the
//! build entirely where neither is available; `uuid.py` catches the resulting
//! `ImportError` and falls back to its Python getters.  Only the `MS_WINDOWS`
//! half is ported here, so every other target keeps that fallback.
//!
//! Without it `uuid.getnode()` cannot reach a MAC address on Windows at all:
//! `uuid.py` leaves `_OS_GETTERS` empty on win32, so `_windll_getnode` is the
//! only non-random getter and it declines unless `_uuid` reports a stable
//! extractable node.  The answer is then a fresh random multicast node on
//! every call.

use pyre_object::{PY_NULL, PyObjectRef};
use windows_sys::Win32::System::Rpc::{
    RPC_S_OK, RPC_S_UUID_LOCAL_ONLY, RPC_S_UUID_NO_ADDRESS, UuidCreateSequential,
};
use windows_sys::core::GUID;

/// One `UuidCreateSequential` call, answering the raw UUID and its status.
fn create_sequential() -> (GUID, i32) {
    let mut uuid = GUID::from_u128(0);
    // SAFETY: `uuid` is a live, aligned `GUID` the callee only writes into.
    let status = unsafe { UuidCreateSequential(&raw mut uuid) };
    (uuid, status)
}

/// `py_windows_has_stable_node`: only `RPC_S_OK` means the node came from a
/// network card.  The two local-only statuses report a random node, which is
/// no more stable than the one `uuid.py` makes for itself.
fn has_stable_node() -> bool {
    create_sequential().1 == RPC_S_OK
}

/// `py_UuidCreate($module, /)`.
fn uuid_create(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (uuid, status) = create_sequential();
    // The two local-only statuses are successes that say the node is random
    // rather than MAC-derived.  If the OS cannot tell, neither can we, so the
    // UUID is taken anyway.
    if !matches!(
        status,
        RPC_S_OK | RPC_S_UUID_LOCAL_ONLY | RPC_S_UUID_NO_ADDRESS
    ) {
        return Err(crate::PyError::os_error_win32_syscall2(
            status, PY_NULL, PY_NULL,
        ));
    }
    // `Py_BuildValue("y#", (const char *)&uuid, sizeof(uuid))` hands over the
    // struct's own memory, which is why `uuid.py` reads it back through
    // `UUID(bytes_le=...)`.  A `GUID` is four, two, two and eight bytes with
    // no padding, so spelling the three integer fields little-endian gives
    // exactly those sixteen bytes.
    let mut bytes = [0u8; 16];
    bytes[0..4].copy_from_slice(&uuid.data1.to_le_bytes());
    bytes[4..6].copy_from_slice(&uuid.data2.to_le_bytes());
    bytes[6..8].copy_from_slice(&uuid.data3.to_le_bytes());
    bytes[8..16].copy_from_slice(&uuid.data4);
    Ok(pyre_object::bytesobject::w_bytes_from_bytes(&bytes))
}

crate::py_module! {
    "_uuid",
    extra_init: |ns| {
        crate::module_ns_store(
            ns,
            "UuidCreate",
            crate::gateway::with_module(
                "_uuid",
                crate::make_module_builtin_function_with_arity("UuidCreate", uuid_create, 0),
            ),
        );
        // `generate_time_safe` is the libuuid entry point and is not in the
        // method table of a Windows build, so the flag that guards it is 0.
        crate::module_ns_store(ns, "has_uuid_generate_time_safe", pyre_object::w_int_new(0));
        crate::module_ns_store(
            ns,
            "has_stable_extractable_node",
            pyre_object::w_int_new(i64::from(has_stable_node())),
        );
    },
}
