//! select.kevent — PyPy: pypy/module/select/interp_kqueue.py W_Kevent.
//!
//! Each `#[pyre_class]` emits a module-scoped `type_object()`, so kevent
//! and kqueue live in separate files to avoid a name clash.

#![allow(dead_code)]

#[cfg(all(target_os = "macos", feature = "host_env"))]
use super::interp_select::filedescriptor_w;
#[cfg(all(target_os = "macos", feature = "host_env"))]
use pyre_object::PyObjectRef;

/// `select.kevent` object — PyPy: `interp_kqueue.py:265 class W_Kevent`.
///
/// Mirrors the platform `struct kevent`: a 64-bit `ident`, signed
/// 16-bit `filter`, 16-bit `flags`, 32-bit `fflags`, signed 64-bit
/// `data`, and an opaque pointer-sized `udata`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
#[crate::pyre_class("select.kevent")]
#[derive(Default)]
pub struct W_Kevent {
    pub ident: u64,
    pub filter: i16,
    pub flags: u16,
    pub fflags: u32,
    pub data: i64,
    pub udata: u64,
}

/// Lexicographic comparison of all six fields, matching
/// `interp_kqueue.py:288 _compare_all_fields`.  Returns an `Ordering`
/// the dunder wrappers turn into the requested relation.
#[cfg(all(target_os = "macos", feature = "host_env"))]
impl W_Kevent {
    fn cmp_key(&self) -> (u64, i64, u32, u32, i64, u64) {
        // `ident`/`udata` unsigned, `filter` widened signed, `flags`/
        // `fflags` unsigned, `data` signed — the field widths PyPy casts
        // to before comparing.
        (
            self.ident,
            self.filter as i64,
            self.flags as u32,
            self.fflags,
            self.data,
            self.udata,
        )
    }
}

#[cfg(all(target_os = "macos", feature = "host_env"))]
#[crate::pyre_methods(
    doc = "kevent(ident, filter=KQ_FILTER_READ, flags=KQ_EV_ADD, fflags=0, data=0, udata=0)"
)]
impl W_Kevent {
    /// `interp_kqueue.py:274 descr__init__`.  Mirrors
    /// `@unwrap_spec(filter=int, flags='c_uint', fflags='c_uint', data=int,
    /// udata=r_uint)`: `ident` is `uint_w` for an int else
    /// `c_filedescriptor_w`; `flags`/`fflags` are `c_uint` (reject
    /// negative / >0xffffffff); `udata` is `r_uint` (full unsigned word).
    fn __init__(
        &mut self,
        w_ident: PyObjectRef,
        #[default(pyre_object::w_int_new(libc::EVFILT_READ as i64))] w_filter: PyObjectRef,
        #[default(pyre_object::w_int_new(libc::EV_ADD as i64))] w_flags: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] w_fflags: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] w_data: PyObjectRef,
        #[default(pyre_object::w_int_new(0))] w_udata: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        let ident: u64 = if unsafe { pyre_object::is_int(w_ident) } {
            crate::baseobjspace::uint_w(w_ident)?
        } else {
            filedescriptor_w(w_ident)? as u64
        };
        let filter = crate::baseobjspace::int_w(w_filter)?;
        let flags = crate::baseobjspace::c_uint_w(w_flags)?;
        let fflags = crate::baseobjspace::c_uint_w(w_fflags)?;
        let data = crate::baseobjspace::int_w(w_data)?;
        let udata = crate::baseobjspace::uint_w(w_udata)?;
        self.ident = ident;
        self.filter = filter as i16;
        self.flags = flags as u16;
        self.fflags = fflags;
        self.data = data;
        self.udata = udata;
        Ok(())
    }

    #[getter]
    fn ident(&self) -> PyObjectRef {
        newint_from_u64(self.ident)
    }
    #[getter]
    fn filter(&self) -> i64 {
        self.filter as i64
    }
    #[getter]
    fn flags(&self) -> i64 {
        self.flags as i64
    }
    #[getter]
    fn fflags(&self) -> i64 {
        self.fflags as i64
    }
    #[getter]
    fn data(&self) -> i64 {
        self.data
    }
    #[getter]
    fn udata(&self) -> PyObjectRef {
        newint_from_u64(self.udata)
    }

    /// `interp_kqueue.py:352 descr__eq__` and friends — two kevents
    /// compare by all six fields lexicographically.  A non-kevent other
    /// yields `NotImplemented`.
    fn __eq__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, |o| o == std::cmp::Ordering::Equal)
    }
    fn __ne__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, |o| o != std::cmp::Ordering::Equal)
    }
    fn __lt__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, |o| o == std::cmp::Ordering::Less)
    }
    fn __le__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, |o| o != std::cmp::Ordering::Greater)
    }
    fn __gt__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, |o| o == std::cmp::Ordering::Greater)
    }
    fn __ge__(&self, w_other: PyObjectRef) -> PyObjectRef {
        kevent_compare(self, w_other, |o| o != std::cmp::Ordering::Less)
    }
}

/// Wrap a 64-bit unsigned word as a Python int, mirroring `space.newint`
/// of a `UINTPTR_T`: values in `i64` range become a plain int, larger ones
/// become a positive long instead of wrapping to a negative via `as i64`.
#[cfg(all(target_os = "macos", feature = "host_env"))]
fn newint_from_u64(v: u64) -> PyObjectRef {
    if v <= i64::MAX as u64 {
        pyre_object::w_int_new(v as i64)
    } else {
        pyre_object::w_long_new(malachite_bigint::BigInt::from(v))
    }
}

/// Shared body for the kevent rich-comparison dunders.
#[cfg(all(target_os = "macos", feature = "host_env"))]
fn kevent_compare(
    this: &W_Kevent,
    w_other: PyObjectRef,
    relation: impl Fn(std::cmp::Ordering) -> bool,
) -> PyObjectRef {
    match W_Kevent::from_obj(w_other) {
        Some(other) => {
            let ord = this.cmp_key().cmp(&other.cmp_key());
            pyre_object::w_bool_from(relation(ord))
        }
        None => pyre_object::w_not_implemented(),
    }
}
