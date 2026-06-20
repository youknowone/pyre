// `extern crate self as pyre_object;` lets `#[pyre_class]`'s emitted
// `::pyre_object::lltype::*` / `::pyre_object::PyObject` paths resolve
// to *this* crate when the macro is consumed from inside pyre-object
// itself (e.g. `superobject.rs`, `propertyobject.rs`, …) instead of
// erroring with `unresolved module pyre_object`.
extern crate self as pyre_object;

/// Typed-payload binding macro: see `pyre/pyre-macros/src/lib.rs`.
pub use pyre_macros::pyre_class;

pub mod array_object;
pub mod boolobject;
pub mod bytearrayobject;
pub mod bytesobject;
pub mod callableiteratorobject;
pub mod celldict;
pub mod cellobject;
pub mod complexobject;
pub mod dict_eq_hook;
pub mod dictmultiobject;
pub mod dictproxyobject;
pub mod dictstrategy;
pub mod dictviewobject;
pub mod enumerateobject;
pub mod excobject;
pub mod filterobject;
pub mod float_array;
pub mod floatobject;
pub mod gc_hook;
pub mod gc_roots;
pub mod generatorobject;
pub mod genericaliasobject;
pub mod getsetproperty;
pub mod identitydict;
pub mod instanceobject;
pub mod int_array;
pub mod intobject;
pub mod itertoolsmodule;
pub mod kwargsdict;
pub mod listobject;
pub mod lltype;
pub mod longobject;
pub mod mapobject;
pub mod memberobject;
pub mod methodobject;
pub mod moduleobject;
pub mod noneobject;
pub mod object_array;
pub mod propertyobject;
pub mod pyobject;
pub mod rangeobject;
pub mod reversedobject;
pub mod setobject;
pub mod sliceobject;
pub mod specialisedtupleobject;
pub mod sreobject;
pub mod strobject;
pub mod superobject;
pub mod tupleobject;
pub mod typeobject;
pub mod unionobject;
pub mod weakref;
pub mod zipobject;

pub use boolobject::*;
pub use bytearrayobject::*;
pub use bytesobject::*;
pub use cellobject::*;
pub use complexobject::*;
pub use dictmultiobject::*;
pub use dictproxyobject::*;
pub use excobject::*;
pub use float_array::*;
pub use floatobject::*;
pub use gc_hook::*;
pub use generatorobject::*;
pub use genericaliasobject::*;
pub use instanceobject::*;
pub use int_array::*;
pub use intobject::*;
pub use itertoolsmodule::*;
pub use listobject::*;
pub use longobject::*;
pub use memberobject::*;
pub use methodobject::*;
pub use moduleobject::*;
pub use noneobject::*;
pub use object_array::*;
pub use propertyobject::*;
pub use pyobject::*;
pub use rangeobject::*;
pub use setobject::*;
pub use sliceobject::*;
pub use specialisedtupleobject::*;
pub use strobject::*;
pub use superobject::*;
pub use tupleobject::*;
pub use typeobject::*;
pub use unionobject::*;
