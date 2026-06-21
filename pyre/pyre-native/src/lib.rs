//! Native library backends for pyre that must stay outside the Charon/LLBC
//! extraction. The interpreter reaches each backend through a non-inlined,
//! non-generic `pub fn` that Charon treats as a residual opaque extern, so the
//! heavy native code (crypto engines, codecs, large static tables) is never
//! lowered into the meta-traceable `.ullbc`.

pub mod hash;
