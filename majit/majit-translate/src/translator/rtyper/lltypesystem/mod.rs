//! `translator/rtyper/lltypesystem/` — RPython-orthodox counterparts of
//! `rpython/rtyper/lltypesystem/`.

// `ll2ctypes` is intentionally absent and must NEVER be added. RPython's
// `ll2ctypes.py` runs lltype programs on CPython by simulating C memory
// through ctypes; pyre compiles to native code via Charon/LLBC and never
// simulates lltype, so the whole module is permanently unused by design.
pub mod ll_str;
pub mod llarena;
pub mod llgroup;
pub mod llheap;
pub mod llmemory;
pub mod lloperation;
pub mod lltype;
pub mod module;
pub mod opimpl;
pub mod rbuilder;
pub mod rbytearray;
pub mod rdict;
pub mod rffi;
pub mod rgcref;
pub mod rlist;
pub mod rordereddict;
pub mod rrange;
pub mod rstr;
pub mod rtagged;
