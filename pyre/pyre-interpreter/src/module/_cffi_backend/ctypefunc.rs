//! Function pointers — PyPy: `pypy/module/_cffi_backend/ctypefunc.py`.
//!
//! A non-variadic function type owns one `CIF_DESCRIPTION` block, built when
//! the type is created and reused by every call through it
//! (`CifDescrBuilder.rawallocate`).  That block holds the `ffi_cif` itself,
//! the `ffi_type *` array libffi reads, the `ffi_type` records any by-value
//! struct argument needs, and the offsets of each argument inside the
//! exchange buffer a call fills.  A variadic function has no such block: its
//! signature is only known per call, so one is built and freed around the
//! call itself.

use crate::PyError;
use pyre_object::PyObjectRef;

use super::cdataobj::{self, W_CData};
use super::ctypeobj::{self, W_CType};

/// `ctypefunc.py set_mustfree_flag`'s values.  RPython's flags 3 to 6 name
/// the pinned and non-moving-buffer cases of an RPython string handed
/// straight to C; pyre copies such an argument instead, so only the first
/// three occur.
pub const MUSTFREE_NOTHING: u8 = 0;
/// The argument slot holds memory this call allocated and must free.
pub const MUSTFREE_FREE: u8 = 1;
/// The argument slot holds a `FILE *` the call borrowed.
pub const MUSTFREE_FILE: u8 = 2;

/// `ctypefunc.py get_mustfree_flag`.
///
/// # Safety
/// `data` must be an argument slot of an exchange buffer whose ctype is a
/// pointer, so the byte before it belongs to this argument.
pub unsafe fn get_mustfree_flag(data: *const u8) -> u8 {
    unsafe { data.sub(1).read() }
}

/// `ctypefunc.py set_mustfree_flag`.
///
/// # Safety
/// As [`get_mustfree_flag`].
pub unsafe fn set_mustfree_flag(data: *mut u8, flag: u8) {
    unsafe { data.sub(1).write(flag) }
}

/// `W_CTypeFunc._compute_extra_text`.
pub fn compute_extra_text(fargs: &[PyObjectRef], ellipsis: bool, abi: i64) -> (String, i64) {
    let mut xpos = 2i64;
    let mut out = if stdcall_abi() == Some(abi) {
        xpos += "__stdcall ".len() as i64;
        "(__stdcall *)(".to_string()
    } else {
        "(*)(".to_string()
    };
    for (i, &w_farg) in fargs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        if let Some(farg) = ctypeobj::ctype_at(w_farg) {
            out.push_str(farg.name());
        }
    }
    if ellipsis {
        if !fargs.is_empty() {
            out.push_str(", ");
        }
        out.push_str("...");
    }
    out.push(')');
    (out, xpos)
}

/// `moduledef.py has_stdcall` and `FFI_STDCALL` — a calling convention only
/// 32-bit Windows has.
#[cfg(all(windows, target_arch = "x86"))]
pub fn stdcall_abi() -> Option<i64> {
    Some(libffi::raw::ffi_abi_FFI_STDCALL as i64)
}

#[cfg(not(all(windows, target_arch = "x86")))]
pub fn stdcall_abi() -> Option<i64> {
    None
}

/// The argument ctypes of a function type, snapshotted before anything that
/// could allocate.  Every ctype is non-moving, so the addresses stay valid.
pub fn fargs_of(ct: &W_CType) -> Vec<PyObjectRef> {
    if ct.fargs.is_null() {
        return Vec::new();
    }
    unsafe { pyre_object::tupleobject::w_tuple_items_copy_as_vec(ct.fargs) }
}

/// `W_CTypeFunc.call` — the entry `W_CData.call` reaches.
pub fn call(
    ct: &W_CType,
    funcaddr: *mut u8,
    args_w: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    if funcaddr.is_null() {
        return Err(PyError::runtime_error(format!(
            "cannot call null function pointer from cdata '{}'",
            ct.name()
        )));
    }
    let fargs = fargs_of(ct);
    if !ct.cif_descr.is_null() {
        if args_w.len() != fargs.len() {
            return Err(PyError::type_error(format!(
                "'{}' expects {} arguments, got {}",
                ct.name(),
                fargs.len(),
                args_w.len()
            )));
        }
        return do_call(&fargs, ct.ctitem, ct.cif_descr, funcaddr, args_w);
    }
    // `W_CTypeFunc.call_varargs` — the cif depends on what was passed, so it
    // is built for this call and freed with it.
    if args_w.len() < fargs.len() {
        return Err(PyError::type_error(format!(
            "'{}' expects at least {} arguments, got {}",
            ct.name(),
            fargs.len(),
            args_w.len()
        )));
    }
    let fvarargs = complete_argtypes(&fargs, args_w)?;
    let cif = build_cif_descr(&fvarargs, ct.ctitem, ct.abi, Some(fargs.len()))?;
    let result = do_call(&fvarargs, ct.ctitem, cif, funcaddr, args_w);
    unsafe { free_cif_descr(cif) };
    result
}

/// `W_CTypeFunc.new_ctypefunc_completing_argtypes` — the declared argument
/// types followed by the promoted type of each variadic argument.
fn complete_argtypes(
    fargs: &[PyObjectRef],
    args_w: &[PyObjectRef],
) -> Result<Vec<PyObjectRef>, PyError> {
    let mut fvarargs = Vec::with_capacity(args_w.len());
    fvarargs.extend_from_slice(fargs);
    for (i, &w_obj) in args_w.iter().enumerate().skip(fargs.len()) {
        let Some(cdata) = W_CData::from_obj(w_obj) else {
            return Err(PyError::type_error(format!(
                "argument {} passed in the variadic part needs to be a cdata object (got {})",
                i + 1,
                crate::type_methods::arg_type_name(w_obj)
            )));
        };
        fvarargs.push(ctypeobj::get_vararg_type(cdata.ctype)?);
    }
    Ok(fvarargs)
}

/// `W_CTypeFunc._call` — fill the exchange buffer, call, read the result out.
fn do_call(
    fargs: &[PyObjectRef],
    w_fresult: PyObjectRef,
    cif: *mut u8,
    funcaddr: *mut u8,
    args_w: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let fresult = ctypeobj::ctype_arg(w_fresult)?;
    // A conversion can run arbitrary Python, and the argument array a builtin
    // receives is a native copy no collector rewrites, so the arguments are
    // read back out of the shadow stack rather than out of `args_w`.
    let roots = pyre_object::gc_roots::push_roots();
    let args_slot = roots.base();
    for &w_arg in args_w {
        let _ = roots.pin_root(w_arg);
    }
    let size = unsafe { exchange_size(cif) };
    let buffer = cdataobj::raw_alloc(size as i64, false)?;
    let mut mustfree_max_plus_1 = 0usize;
    let called = (|| -> Result<PyObjectRef, PyError> {
        for i in 0..args_w.len() {
            let data = unsafe { buffer.add(exchange_arg(cif, i)) };
            let argtype = ctypeobj::ctype_arg(fargs[i])?;
            if unsafe {
                ctypeobj::convert_argument_from_object(argtype, data, roots.get(args_slot + i))?
            } {
                mustfree_max_plus_1 = i + 1;
            }
        }
        // `clibffi.py jit_ffi_call` swaps the thread's alternate errno into
        // the C runtime around every foreign call.
        super::cerrno::errno_before();
        unsafe { invoke(cif, funcaddr, buffer, args_w.len()) };
        super::cerrno::errno_after();
        let resultdata = unsafe { buffer.add(exchange_result(cif)) };
        unsafe { ctypeobj::copy_and_convert_to_object(fresult, resultdata) }
    })();
    for i in 0..mustfree_max_plus_1 {
        let Some(argtype) = ctypeobj::ctype_at(fargs[i]) else {
            continue;
        };
        if argtype.kind != ctypeobj::KIND_POINTER {
            continue;
        }
        let data = unsafe { buffer.add(exchange_arg(cif, i)) };
        if unsafe { get_mustfree_flag(data) } == MUSTFREE_FREE {
            let raw = unsafe { data.cast::<*mut u8>().read_unaligned() };
            unsafe { libc::free(raw.cast::<libc::c_void>()) };
        }
    }
    unsafe { libc::free(buffer.cast::<libc::c_void>()) };
    called
}

// ── the CIF_DESCRIPTION block ───────────────────────────────────────────

#[cfg(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
))]
mod cif {
    use super::{PyError, PyObjectRef, W_CType, ctypeobj};
    use libffi::low::{CodePtr, ffi_abi, ffi_cif, ffi_type, type_tag, types};

    /// `jit_libffi.py CIF_DESCRIPTION`.  An array of `nargs` exchange offsets
    /// follows the record; `atypes` and any struct `ffi_type` follow that, all
    /// inside the one block this describes.
    #[repr(C)]
    pub struct CifDescription {
        pub cif: ffi_cif,
        pub abi: ffi_abi,
        pub nargs: usize,
        pub rtype: *mut ffi_type,
        pub atypes: *mut *mut ffi_type,
        pub exchange_size: usize,
        pub exchange_result: usize,
    }

    /// `jit_libffi.py SIZE_OF_FFI_ARG`.
    const SIZE_OF_FFI_ARG: usize = std::mem::size_of::<libffi::low::ffi_arg>();

    unsafe fn header(cif: *mut u8) -> &'static mut CifDescription {
        unsafe { &mut *cif.cast::<CifDescription>() }
    }

    unsafe fn exchange_args(cif: *mut u8) -> *mut usize {
        unsafe {
            cif.add(std::mem::size_of::<CifDescription>())
                .cast::<usize>()
        }
    }

    pub unsafe fn exchange_size(cif: *mut u8) -> usize {
        unsafe { header(cif).exchange_size }
    }

    pub unsafe fn exchange_result(cif: *mut u8) -> usize {
        unsafe { header(cif).exchange_result }
    }

    pub unsafe fn exchange_arg(cif: *mut u8, i: usize) -> usize {
        unsafe { exchange_args(cif).add(i).read() }
    }

    /// `jit_libffi.py jit_ffi_call_impl_any` — point the argument array at the
    /// slots the exchange buffer already holds and hand libffi the result slot.
    ///
    /// # Safety
    /// `buffer` must be an exchange buffer this `cif` describes, filled with
    /// `nargs` converted arguments, and `funcaddr` must be the entry point the
    /// cif was prepared for.
    pub unsafe fn invoke(cif: *mut u8, funcaddr: *mut u8, buffer: *mut u8, nargs: usize) {
        let argptrs = buffer.cast::<*mut std::ffi::c_void>();
        for i in 0..nargs {
            unsafe {
                argptrs
                    .add(i)
                    .write(buffer.add(exchange_arg(cif, i)).cast::<std::ffi::c_void>())
            };
        }
        let resultdata = unsafe { buffer.add(exchange_result(cif)) };
        let descr = unsafe { header(cif) };
        unsafe {
            libffi::low::call_return_into(
                &raw mut descr.cif,
                CodePtr::from_ptr(funcaddr.cast::<std::ffi::c_void>()),
                argptrs,
                resultdata.cast::<std::ffi::c_void>(),
            );
        }
    }

    /// `CifDescrBuilder` — the two-pass bump allocator that measures the block
    /// and then fills it.
    struct Builder<'a> {
        fargs: &'a [PyObjectRef],
        w_fresult: PyObjectRef,
        nb_bytes: usize,
        bufferp: *mut u8,
        atypes: *mut *mut ffi_type,
        rtype: *mut ffi_type,
    }

    impl Builder<'_> {
        /// `CifDescrBuilder.fb_alloc`.  Every request is a multiple of eight
        /// and the block itself comes from `malloc`, so each record inside it
        /// lands on its own alignment.
        fn alloc(&mut self, size: usize) -> *mut u8 {
            let size = size.next_multiple_of(8);
            if self.bufferp.is_null() {
                self.nb_bytes += size;
                return std::ptr::null_mut();
            }
            let result = self.bufferp;
            self.bufferp = unsafe { result.add(size) };
            result
        }

        /// `CifDescrBuilder.fb_build`.
        fn build(&mut self) -> Result<(), PyError> {
            let nargs = self.fargs.len();
            self.alloc(
                std::mem::size_of::<CifDescription>() + nargs * std::mem::size_of::<usize>(),
            );
            let atypes = self.alloc(nargs * std::mem::size_of::<*mut ffi_type>());
            self.atypes = atypes.cast::<*mut ffi_type>();
            self.rtype = self.fill_type(ctypeobj::ctype_arg(self.w_fresult)?, true)?;
            for i in 0..nargs {
                let farg = ctypeobj::ctype_arg(self.fargs[i])?;
                let atype = self.fill_type(farg, false)?;
                if !self.atypes.is_null() {
                    unsafe { self.atypes.add(i).write(atype) };
                }
            }
            Ok(())
        }

        /// `CifDescrBuilder.fb_fill_type` and the `_get_ffi_type` each ctype
        /// class is given at the bottom of `ctypefunc.py`.
        fn fill_type(&mut self, ct: &W_CType, is_result: bool) -> Result<*mut ffi_type, PyError> {
            let by_size = |signed: bool| -> Option<*mut ffi_type> {
                let t: *mut ffi_type = match (ct.size, signed) {
                    (1, true) => &raw mut types::sint8,
                    (2, true) => &raw mut types::sint16,
                    (4, true) => &raw mut types::sint32,
                    (8, true) => &raw mut types::sint64,
                    (1, false) => &raw mut types::uint8,
                    (2, false) => &raw mut types::uint16,
                    (4, false) => &raw mut types::uint32,
                    (8, false) => &raw mut types::uint64,
                    _ => return None,
                };
                Some(t)
            };
            match ct.kind {
                ctypeobj::KIND_STRUCT if ct.size >= 0 => self.struct_ffi_type(ct, is_result),
                // Only for a better error message: a completed union still
                // cannot be passed by value.
                ctypeobj::KIND_UNION if ct.size >= 0 => Err(unsupported(
                    ct,
                    is_result,
                    "libffi",
                    "Unions",
                    String::new(),
                )),
                ctypeobj::KIND_PRIM_SIGNED => {
                    by_size(true).ok_or_else(|| missing_ffi_type(ct, is_result))
                }
                ctypeobj::KIND_PRIM_UNSIGNED
                | ctypeobj::KIND_PRIM_BOOL
                | ctypeobj::KIND_PRIM_CHAR
                | ctypeobj::KIND_PRIM_UNICHAR => {
                    by_size(false).ok_or_else(|| missing_ffi_type(ct, is_result))
                }
                ctypeobj::KIND_PRIM_FLOAT => match ct.size {
                    4 => Ok(&raw mut types::float),
                    8 => Ok(&raw mut types::double),
                    _ => Err(missing_ffi_type(ct, is_result)),
                },
                ctypeobj::KIND_PRIM_LONGDOUBLE => Ok(long_double_ffi_type()),
                ctypeobj::KIND_PRIM_COMPLEX => Err(PyError::not_implemented(format!(
                    "ctype '{}' (size {}) not supported as {} (the support for complex types inside libffi is mostly missing at this point, so CFFI only supports complex types as arguments or return value in API-mode functions)",
                    ct.name(),
                    ct.size,
                    place(is_result)
                ))),
                // `W_CTypePtrBase._get_ffi_type` — a pointer or a function
                // pointer.  A plain array is not one of them.
                ctypeobj::KIND_POINTER | ctypeobj::KIND_FUNC => Ok(&raw mut types::pointer),
                ctypeobj::KIND_VOID if is_result => Ok(&raw mut types::void),
                _ => Err(missing_ffi_type(ct, is_result)),
            }
        }

        /// `CifDescrBuilder.fb_struct_ffi_type`.
        fn struct_ffi_type(
            &mut self,
            ct: &W_CType,
            is_result: bool,
        ) -> Result<*mut ffi_type, PyError> {
            ct.force_lazy_struct()?;
            // A struct completed from an incomplete declaration may be laid
            // out differently from the one the C compiler saw, and the calling
            // convention depends on the fields; so may an anonymous nested
            // struct, whose origin is no longer recorded here.
            if ct.has(ctypeobj::F_CUSTOM_FIELD_POS) {
                return Err(unsupported(
                    ct,
                    is_result,
                    "",
                    "Such structs",
                    "It is a struct declared with \"...;\", but the C calling convention may depend on the missing fields; or, it contains anonymous struct/unions".to_string(),
                ));
            }
            if ct.has(ctypeobj::F_WITH_PACKED_CHANGE) {
                return Err(unsupported(
                    ct,
                    is_result,
                    "",
                    "Such structs",
                    "It is a 'packed' structure, with a different layout than expected by libffi"
                        .to_string(),
                ));
            }
            let fields = super::super::ctypestruct::fields_list_of(ct)?;
            let mut nflat = 0usize;
            for &w_field in &fields {
                nflat += flatten(ct, w_field, is_result)?.1;
            }
            let elements = self
                .alloc((nflat + 1) * std::mem::size_of::<*mut ffi_type>())
                .cast::<*mut ffi_type>();
            let mut written = 0usize;
            for &w_field in &fields {
                let (base, flat) = flatten(ct, w_field, is_result)?;
                let sub = self.fill_type(base, false)?;
                if !elements.is_null() {
                    for _ in 0..flat {
                        unsafe { elements.add(written).write(sub) };
                        written += 1;
                    }
                }
            }
            if !elements.is_null() {
                unsafe { elements.add(written).write(std::ptr::null_mut()) };
            }
            let ffistruct = self
                .alloc(std::mem::size_of::<ffi_type>())
                .cast::<ffi_type>();
            if !ffistruct.is_null() {
                unsafe {
                    ffistruct.write(ffi_type {
                        size: ct.size as usize,
                        alignment: ct.alignof()? as u16,
                        type_: type_tag::STRUCT,
                        elements,
                    })
                };
            }
            Ok(ffistruct)
        }
    }

    /// One struct field expanded to the scalar it repeats and how many times.
    fn flatten(
        ct: &W_CType,
        w_field: PyObjectRef,
        is_result: bool,
    ) -> Result<(&'static mut W_CType, usize), PyError> {
        let field = super::super::ctypestruct::W_CField::from_obj(w_field)
            .ok_or_else(|| PyError::system_error("struct field list holds a non-field"))?;
        if field.is_bitfield() {
            return Err(unsupported(
                ct,
                is_result,
                "",
                "Such structs",
                "It is a struct with bit fields, which libffi does not support".to_string(),
            ));
        }
        let mut flat: i64 = 1;
        let mut item = ctypeobj::ctype_arg(field.ctype)?;
        while item.kind == ctypeobj::KIND_ARRAY {
            flat *= item.length;
            item = ctypeobj::ctype_arg(item.ctitem)?;
        }
        if flat <= 0 {
            return Err(unsupported(
                ct,
                is_result,
                "",
                "Such structs",
                "It is a struct with a zero-length array, which libffi does not support"
                    .to_string(),
            ));
        }
        Ok((item, flat as usize))
    }

    fn place(is_result: bool) -> &'static str {
        if is_result {
            "return value"
        } else {
            "argument"
        }
    }

    /// `ctypefunc.py _missing_ffi_type` and `_notimplemented_ffi_type`.
    /// The `ffi_type` for `long double`.
    ///
    /// libffi defines `ffi_type_longdouble` only where its configure saw
    /// `sizeof(long double) != sizeof(double)`; where it did not, the two are
    /// the same type and `ffi_type_double` describes both.  The build script
    /// asks the target's own compiler that question and sets the cfg, because
    /// naming the symbol on a target that has none is a link error rather than
    /// a call that fails.
    #[cfg(pyre_ffi_type_longdouble)]
    fn long_double_ffi_type() -> *mut ffi_type {
        &raw mut types::longdouble
    }

    #[cfg(not(pyre_ffi_type_longdouble))]
    fn long_double_ffi_type() -> *mut ffi_type {
        &raw mut types::double
    }

    fn missing_ffi_type(ct: &W_CType, is_result: bool) -> PyError {
        if ct.size < 0 {
            return PyError::type_error(format!("ctype '{}' has incomplete type", ct.name()));
        }
        PyError::not_implemented(format!(
            "ctype '{}' (size {}) not supported as {}",
            ct.name(),
            ct.size,
            place(is_result)
        ))
    }

    /// `CifDescrBuilder.fb_unsupported` and `fb_union_ffi_type`, which share
    /// `_SUPPORTED_IN_API_MODE`'s tail.
    fn unsupported(
        ct: &W_CType,
        is_result: bool,
        by: &str,
        subject: &str,
        detail: String,
    ) -> PyError {
        let place = place(is_result);
        let head = if by.is_empty() {
            format!(
                "ctype '{}' not supported as {place}.  {detail}.  ",
                ct.name()
            )
        } else {
            format!("ctype '{}' not supported as {place} by {by}.  ", ct.name())
        };
        PyError::not_implemented(format!(
            "{head}{subject} are only supported as {place} if the function is 'API mode' and non-variadic (i.e. declared inside ffibuilder.cdef()+ffibuilder.set_source() and not taking a final '...' argument)"
        ))
    }

    fn align_arg(n: usize) -> usize {
        (n + 7) & !7
    }

    fn align_to(n: usize, t: *mut ffi_type) -> usize {
        let a = unsafe { (*t).alignment } as usize - 1;
        (n + a) & !a
    }

    /// `CifDescrBuilder.rawallocate` — measure, allocate, fill, prepare.
    /// `CifDescrBuilder.rawallocate`.  `nfixedargs` is the count of declared
    /// arguments a variadic call passes before its `...`, and `None` says the
    /// function is not variadic.
    pub fn build_cif_descr(
        fargs: &[PyObjectRef],
        w_fresult: PyObjectRef,
        abi: i64,
        nfixedargs: Option<usize>,
    ) -> Result<*mut u8, PyError> {
        let mut builder = Builder {
            fargs,
            w_fresult,
            nb_bytes: 0,
            bufferp: std::ptr::null_mut(),
            atypes: std::ptr::null_mut(),
            rtype: std::ptr::null_mut(),
        };
        builder.build()?;
        let rawmem = super::cdataobj::raw_alloc(builder.nb_bytes as i64, true)?;
        builder.bufferp = rawmem;
        builder.nb_bytes = 0;
        if let Err(e) = builder.build() {
            unsafe { libc::free(rawmem.cast::<libc::c_void>()) };
            return Err(e);
        }

        // `CifDescrBuilder.fb_build_exchange`.
        let nargs = fargs.len();
        let mut offset = std::mem::size_of::<*mut u8>() * nargs;
        offset = align_arg(align_to(offset, builder.rtype));
        let exchange_result = offset;
        offset += unsafe { (*builder.rtype).size }.max(SIZE_OF_FFI_ARG);
        for i in 0..nargs {
            // Room for the must-free flag the pointer conversion writes just
            // before the slot.
            if ctypeobj::ctype_arg(fargs[i])?.kind == ctypeobj::KIND_POINTER {
                offset += 1;
            }
            let atype = unsafe { builder.atypes.add(i).read() };
            offset = align_arg(align_to(offset, atype));
            unsafe { exchange_args(rawmem).add(i).write(offset) };
            offset += unsafe { (*atype).size };
        }

        // `CifDescrBuilder.fb_extra_fields`, then `jit_ffi_prep_cif`.
        let descr = unsafe { header(rawmem) };
        descr.abi = abi as ffi_abi;
        descr.nargs = nargs;
        descr.rtype = builder.rtype;
        descr.atypes = builder.atypes;
        descr.exchange_result = exchange_result;
        descr.exchange_size = align_arg(offset);
        let (abi, rtype, atypes) = (descr.abi, descr.rtype, descr.atypes);
        // A target whose variadic arguments do not travel the way its fixed
        // ones do -- aarch64 Darwin passes them on the stack -- needs the
        // fixed count, which is what `clibffi.py:503` hands `ffi_prep_cif_var`
        // whenever `variadic_args > 0`.
        let cif = &raw mut descr.cif;
        let prepared = match nfixedargs {
            Some(nfixed) => unsafe {
                libffi::low::prep_cif_var(cif, abi, nfixed, nargs, rtype, atypes)
            },
            None => unsafe { libffi::low::prep_cif(cif, abi, nargs, rtype, atypes) },
        };
        if prepared.is_err() {
            unsafe { libc::free(rawmem.cast::<libc::c_void>()) };
            return Err(PyError::system_error(
                "libffi failed to build this function type",
            ));
        }
        Ok(rawmem)
    }

    /// `W_CTypeFunc.__del__` — the block is one allocation.
    ///
    /// # Safety
    /// `cif` must come from [`build_cif_descr`] and not be in use.
    pub unsafe fn free_cif_descr(cif: *mut u8) {
        unsafe { libc::free(cif.cast::<libc::c_void>()) };
    }
}

#[cfg(not(all(
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android"
    ),
    not(any(target_env = "musl", target_env = "sgx"))
)))]
mod cif {
    use super::{PyError, PyObjectRef};

    /// No libffi on this target, so no foreign call can be described.
    pub fn build_cif_descr(
        _fargs: &[PyObjectRef],
        _w_fresult: PyObjectRef,
        _abi: i64,
        _nfixedargs: Option<usize>,
    ) -> Result<*mut u8, PyError> {
        Err(PyError::not_implemented(
            "this platform has no libffi, so a C function cannot be called",
        ))
    }

    pub unsafe fn free_cif_descr(_cif: *mut u8) {}

    pub unsafe fn exchange_size(_cif: *mut u8) -> usize {
        0
    }

    pub unsafe fn exchange_result(_cif: *mut u8) -> usize {
        0
    }

    pub unsafe fn exchange_arg(_cif: *mut u8, _i: usize) -> usize {
        0
    }

    pub unsafe fn invoke(_cif: *mut u8, _funcaddr: *mut u8, _buffer: *mut u8, _nargs: usize) {}
}

pub use cif::{build_cif_descr, free_cif_descr};
use cif::{exchange_arg, exchange_result, exchange_size, invoke};
