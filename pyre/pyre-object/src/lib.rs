// PyObjectRef intentionally carries the translated runtime's raw object
// handle. Public slot wrappers and `#[pyre_class]` expansions validate object
// kinds internally while preserving the safe interpreter-facing API; changing
// every generated wrapper to `unsafe` would alter that contract. The remaining
// shapes below mirror PyPy object APIs and line-by-line control flow.
#![allow(
    clippy::approx_constant,
    clippy::arc_with_non_send_sync,
    clippy::assertions_on_constants,
    clippy::doc_lazy_continuation,
    clippy::len_without_is_empty,
    clippy::missing_safety_doc,
    clippy::needless_range_loop,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::result_unit_err,
    clippy::too_many_arguments,
    clippy::while_let_loop
)]

// `extern crate self as pyre_object;` lets `#[pyre_class]`'s emitted
// `::pyre_object::lltype::*` / `::pyre_object::PyObject` paths resolve
// to *this* crate when the macro is consumed from inside pyre-object
// itself (e.g. `descriptor.rs`, `function.rs`, …) instead of
// erroring with `unresolved module pyre_object`.
extern crate self as pyre_object;

/// Typed-payload binding macro: see `pyre/pyre-macros/src/lib.rs`.
pub use pyre_macros::pyre_class;

pub mod _pypy_generic_alias;
pub mod boolobject;
pub mod buffer;
pub mod bufferview;
pub mod bytearrayobject;
pub mod bytesobject;
pub mod celldict;
pub mod complexobject;
pub mod descriptor;
pub mod dict_eq_hook;
pub mod dictmultiobject;
pub mod dictproxyobject;
pub mod float_array;
pub mod floatobject;
pub mod function;
pub mod functional;
pub mod gc_hook;
pub mod gc_interp;
pub mod gc_roots;
pub mod gc_storage;
pub mod generator;
pub mod identitydict;
pub mod int_array;
pub mod interp_array;
pub mod interp_exceptions;
pub mod interp_itertools;
pub mod interp_sre;
pub mod intobject;
pub mod iterobject;
pub mod kw_marker;
pub mod kwargsdict;
pub mod listobject;
pub mod lltype;
pub mod longobject;
pub mod memoryview;
pub mod module;
pub mod nestedscope;
pub mod noneobject;
pub mod object_array;
pub mod objectobject;
pub mod operation;
pub mod pyobject;
pub mod quasiimmut;
pub mod rutf8;
pub mod setobject;
pub mod sliceobject;
pub mod slots;
pub mod special;
pub mod specialisedtupleobject;
pub mod tagged_int;
pub mod tupleobject;
pub mod typedef;
pub mod typeobject;
pub mod unicodeobject;
pub mod weakref;

pub use _pypy_generic_alias::*;
pub use boolobject::*;
pub use bytearrayobject::*;
pub use bytesobject::*;
pub use complexobject::*;
pub use descriptor::*;
pub use dictmultiobject::*;
pub use dictproxyobject::*;
pub use float_array::*;
pub use floatobject::*;
pub use function::*;
pub use functional::*;
pub use gc_hook::*;
pub use generator::*;
pub use int_array::*;
pub use interp_exceptions::*;
pub use interp_itertools::*;
pub use intobject::*;
pub use iterobject::*;
pub use listobject::*;
pub use longobject::*;
pub use module::*;
pub use nestedscope::*;
pub use noneobject::*;
pub use object_array::*;
pub use objectobject::*;
pub use pyobject::*;
pub use setobject::*;
pub use sliceobject::*;
pub use special::*;
pub use specialisedtupleobject::*;
pub use tupleobject::*;
pub use typedef::*;
pub use typeobject::*;
pub use unicodeobject::*;
