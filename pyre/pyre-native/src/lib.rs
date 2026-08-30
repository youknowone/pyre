//! Native library backends for pyre that must stay outside the Charon/LLBC
//! extraction. The interpreter reaches each backend through a non-inlined,
//! non-generic `pub fn`; `pyre/scripts/extract-llbc.py` explicitly marks this
//! crate opaque because Charon otherwise follows local workspace dependencies.
//! The declarations remain available as residual externs, while heavy native
//! code (crypto engines, codecs, large static tables) is never lowered into the
//! meta-traceable `.ullbc`.

pub mod binascii;
pub mod bz2;
#[cfg(all(not(target_arch = "wasm32"), not(feature = "sandbox")))]
pub mod cffi;
pub mod cjkcodecs;
pub mod hash;
pub mod inet_text;
pub mod json;
pub mod locale;
pub mod lzma;
#[cfg(not(target_arch = "wasm32"))]
pub mod ssl;
#[cfg(feature = "wasm_vfs")]
pub mod vfs;
pub mod zlib;
