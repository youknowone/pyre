//! `CFuncPtr` (imported as `_CFuncPtr`) — the foreign-function object.
//!
//! `__new__` resolves and stores a function-pointer address (`"_ptr"`) from a
//! `(name, dll)` pair, a bare integer address, or nothing (NULL).  `__call__`
//! marshals scalar Python arguments into libffi values and performs the call
//! through `host_env`, then decodes the scalar result.
//!
//! All host/FFI work is delegated to `rustpython_host_env::ctypes`.  Arguments
//! are marshalled into `CallArg`s and the return type into a `CallRet`, then the
//! call runs through the single `call` entry point, which performs the libffi
//! call and decodes the result.  By-reference arguments — `byref()` carriers,
//! `_Pointer`/`Array` instances, and pointer-typed cdata — lower to
//! `CallArg::Pointer(addr)`; by-value struct/union arguments and returns lower
//! to `CallArg::Aggregate` / `CallRet::Aggregate`; a pointer-typed `restype`
//! wraps the returned address in a fresh instance.

use super::cdata;
use super::stginfo;
use super::type_ns_store;
use pyre_object::PyObjectRef;
use rustpython_host_env::ctypes as host_ctypes;

/// `_flags_ & FUNCFLAG_USE_ERRNO` — swap the ctypes-local errno around the call.
const FUNCFLAG_USE_ERRNO: i64 = 0x8;

/// Reserved instance-dict keys.
const PTR_KEY: &str = "_ptr";
const RESTYPE_KEY: &str = "_restype";
const ARGTYPES_KEY: &str = "_argtypes";

thread_local! {
    static CFUNCPTR_TYPE_OBJ: std::cell::OnceCell<PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

/// The native `CFuncPtr` type object (cached, `hasdict=true`).
pub(super) fn cfuncptr_type() -> PyObjectRef {
    CFUNCPTR_TYPE_OBJ.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("CFuncPtr", init_cfuncptr_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

fn init_cfuncptr_type(ns: PyObjectRef) {
    type_ns_store(
        ns,
        "__new__",
        crate::make_builtin_function("__new__", cfuncptr_new),
    );
    type_ns_store(
        ns,
        "__call__",
        crate::make_builtin_function("__call__", cfuncptr_call),
    );
    // `restype` / `argtypes` — settable data descriptors with class-attr
    // fallback to `_restype_` / `_argtypes_`.
    type_ns_store(
        ns,
        "restype",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("restype", restype_getter, 2),
            crate::make_builtin_function_with_arity("restype", restype_setter, 3),
            pyre_object::PY_NULL,
            "restype",
        ),
    );
    type_ns_store(
        ns,
        "argtypes",
        crate::typedef::make_getset_property_named(
            crate::make_builtin_function_with_arity("argtypes", argtypes_getter, 2),
            crate::make_builtin_function_with_arity("argtypes", argtypes_setter, 3),
            pyre_object::PY_NULL,
            "argtypes",
        ),
    );
}

// ── construction ──────────────────────────────────────────────────────

/// `_CFuncPtr.__new__(cls, arg=None)`.
fn cfuncptr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() || !unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error(
            "CFuncPtr.__new__(): not enough arguments",
        ));
    }
    let cls = args[0];
    let (pos, _kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);
    let addr: usize = match pos.first().copied() {
        None => 0,
        Some(a) if unsafe { pyre_object::is_none(a) } => 0,
        Some(a) if unsafe { pyre_object::is_int(a) } => {
            (unsafe { pyre_object::w_int_get_value(a) }) as usize
        }
        Some(a) if unsafe { pyre_object::is_tuple(a) } => resolve_from_tuple(a)?,
        Some(_) => {
            return Err(crate::PyError::type_error(
                "argument must be callable or integer function address",
            ));
        }
    };
    let obj = pyre_object::w_instance_new(cls);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error(
            "CFuncPtr instance has no instance dict",
        ));
    }
    unsafe { pyre_object::w_dict_setitem_str(d, PTR_KEY, pyre_object::w_int_new(addr as i64)) };
    Ok(obj)
}

/// `(name, dll)` → resolved symbol address.  `dll._handle` is the integer
/// library handle; `name` is the symbol string/bytes.
fn resolve_from_tuple(t: PyObjectRef) -> Result<usize, crate::PyError> {
    let name_obj = unsafe { pyre_object::w_tuple_getitem(t, 0) };
    let dll_obj = unsafe { pyre_object::w_tuple_getitem(t, 1) };
    let (Some(name_obj), Some(dll_obj)) = (name_obj, dll_obj) else {
        return Err(crate::PyError::type_error(
            "CFuncPtr constructor requires a (name, dll) pair",
        ));
    };
    let handle_obj = crate::baseobjspace::getattr_str(dll_obj, "_handle")?;
    let handle = crate::baseobjspace::int_w(handle_obj)? as usize;
    let name_bytes: Vec<u8> = if unsafe { pyre_object::is_str(name_obj) } {
        unsafe { pyre_object::w_str_get_value(name_obj) }
            .as_bytes()
            .to_vec()
    } else if unsafe { pyre_object::is_bytes(name_obj) } {
        unsafe { pyre_object::bytesobject::w_bytes_data(name_obj) }.to_vec()
    } else {
        return Err(crate::PyError::type_error(
            "function name must be string or bytes (ordinals not supported)",
        ));
    };
    host_ctypes::lookup_function_symbol_addr(handle, &name_bytes).map_err(|e| {
        use host_ctypes::LookupSymbolError as L;
        let sym = String::from_utf8_lossy(&name_bytes);
        match e {
            L::LibraryNotFound => crate::PyError::value_error("library not found"),
            L::LibraryClosed => {
                crate::PyError::attribute_error(format!("function '{sym}' not found"))
            }
            L::Load(_) => crate::PyError::attribute_error(format!("function '{sym}' not found")),
        }
    })
}

// ── restype / argtypes descriptors ────────────────────────────────────

fn instance_get(obj: PyObjectRef, key: &str) -> Option<PyObjectRef> {
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return None;
    }
    unsafe { pyre_object::w_dict_getitem_str(d, key) }
}

fn instance_set(obj: PyObjectRef, key: &str, value: PyObjectRef) {
    let d = crate::baseobjspace::getdict(obj);
    if !d.is_null() {
        unsafe { pyre_object::w_dict_setitem_str(d, key, value) };
    }
}

fn restype_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[1];
    if let Some(v) = instance_get(obj, RESTYPE_KEY) {
        return Ok(v);
    }
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    Ok(
        unsafe { crate::baseobjspace::lookup_in_type(cls, "_restype_") }
            .unwrap_or_else(pyre_object::w_none),
    )
}

fn restype_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    instance_set(args[1], RESTYPE_KEY, args[2]);
    Ok(pyre_object::w_none())
}

fn argtypes_getter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[1];
    if let Some(v) = instance_get(obj, ARGTYPES_KEY) {
        return Ok(v);
    }
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    Ok(
        unsafe { crate::baseobjspace::lookup_in_type(cls, "_argtypes_") }
            .unwrap_or_else(pyre_object::w_none),
    )
}

fn argtypes_setter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    instance_set(args[1], ARGTYPES_KEY, args[2]);
    Ok(pyre_object::w_none())
}

// ── call ──────────────────────────────────────────────────────────────

/// Resolved return-type selector.
enum Ret {
    Void,
    Code(String),
    /// A pointer metaclass type (`POINTER(T)`): the result address is wrapped
    /// in a fresh instance of this type.
    Pointer(PyObjectRef),
    /// A by-value struct/union type: the returned aggregate bytes are copied
    /// into a fresh instance of this type.
    Aggregate(PyObjectRef),
}

fn resolve_restype(obj: PyObjectRef) -> Result<Ret, crate::PyError> {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let rt = instance_get(obj, RESTYPE_KEY)
        .or_else(|| unsafe { crate::baseobjspace::lookup_in_type(cls, "_restype_") });
    match rt {
        // CDLL functions default to c_int when no restype is set.
        None => Ok(Ret::Code("i".to_string())),
        Some(o) if unsafe { pyre_object::is_none(o) } => Ok(Ret::Void),
        Some(o) => {
            // A `_Pointer` subtype returns a live pointer instance; a
            // struct/union subtype returns a by-value aggregate instance.
            if let Some(info) = stginfo::stginfo_of(o) {
                match stginfo::stginfo_paramfunc(info).as_str() {
                    "pointer" => return Ok(Ret::Pointer(o)),
                    "struct" | "union" => return Ok(Ret::Aggregate(o)),
                    _ => {}
                }
            }
            let tc = cdata::type_code_of(o)
                .ok_or_else(|| crate::PyError::type_error("invalid restype"))?;
            Ok(Ret::Code(tc))
        }
    }
}

/// Wrap a returned pointer value `p` in a fresh instance of pointer type `rt`.
fn wrap_pointer_result(rt: PyObjectRef, p: usize) -> Result<PyObjectRef, crate::PyError> {
    let obj = pyre_object::w_instance_new(rt);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error("pointer instance has no dict"));
    }
    let psize = host_ctypes::pointer_size();
    let ba = pyre_object::w_bytearray_new(psize);
    let bytes = host_ctypes::simple_storage_value_to_bytes_endian(
        "P",
        host_ctypes::SimpleStorageValue::Pointer(p),
        false,
    );
    let n = bytes.len().min(psize);
    unsafe {
        pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
        pyre_object::w_dict_setitem_str(d, "_b_", ba);
    }
    Ok(obj)
}

/// The `_argtypes_` sequence as a Vec, or `None` when unset (ConvParam
/// defaults apply).
fn resolve_argtypes(obj: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    let at = instance_get(obj, ARGTYPES_KEY)
        .or_else(|| unsafe { crate::baseobjspace::lookup_in_type(cls, "_argtypes_") })?;
    if unsafe { pyre_object::is_none(at) } {
        return None;
    }
    seq_to_vec(at)
}

fn seq_to_vec(obj: PyObjectRef) -> Option<Vec<PyObjectRef>> {
    if unsafe { pyre_object::is_tuple(obj) } {
        let n = unsafe { pyre_object::w_tuple_len(obj) };
        Some(
            (0..n as i64)
                .filter_map(|i| unsafe { pyre_object::w_tuple_getitem(obj, i) })
                .collect(),
        )
    } else if unsafe { pyre_object::is_list(obj) } {
        let n = unsafe { pyre_object::w_list_len(obj) };
        Some(
            (0..n as i64)
                .filter_map(|i| unsafe { pyre_object::w_list_getitem(obj, i) })
                .collect(),
        )
    } else {
        None
    }
}

fn funcptr_flags(obj: PyObjectRef) -> i64 {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    match unsafe { crate::baseobjspace::lookup_in_type(cls, "_flags_") } {
        Some(o) if unsafe { pyre_object::is_int(o) } => unsafe { pyre_object::w_int_get_value(o) },
        _ => 0,
    }
}

fn funcptr_addr(obj: PyObjectRef) -> usize {
    instance_get(obj, PTR_KEY)
        .filter(|o| unsafe { pyre_object::is_int(*o) })
        .map(|o| unsafe { pyre_object::w_int_get_value(o) } as usize)
        .unwrap_or(0)
}

/// Owned argument data whose buffers must outlive the borrowed `CallArg`s
/// handed to `call`.
enum OwnedArg {
    Typed(String, Vec<u8>),
    Int(i32),
    Double(f64),
    Pointer(usize),
    /// A by-value aggregate: its recursive layout and a copy of its bytes.
    Aggregate(host_ctypes::CTypeLayout, Vec<u8>),
}

/// `_CFuncPtr.__call__(self, *args)`.
fn cfuncptr_call(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("__call__ requires self"));
    }
    let self_obj = args[0];
    let (call_args, _kwargs) = crate::builtins::split_builtin_kwargs(&args[1..]);

    // Marshal arguments into owned scalar data.  `keepalive` owns any
    // null-terminated `bytes` copies that pointer args address; `owned` owns
    // the typed buffers.  Both must outlive the borrowed `SimpleArg`s below.
    let mut owned: Vec<OwnedArg> = Vec::with_capacity(call_args.len());
    let mut keepalive: Vec<Vec<u8>> = Vec::new();

    match resolve_argtypes(self_obj) {
        Some(argtypes) => {
            for (i, at) in argtypes.iter().enumerate() {
                let arg = *call_args.get(i).ok_or_else(|| {
                    crate::PyError::type_error(format!(
                        "this function takes at least {} argument(s)",
                        argtypes.len()
                    ))
                })?;
                owned.push(marshal_typed_arg(arg, *at, &mut keepalive)?);
            }
        }
        None => {
            for &arg in call_args {
                owned.push(marshal_default_arg(arg, &mut keepalive)?);
            }
        }
    }

    let ret = resolve_restype(self_obj)?;
    // Build the aggregate return layout (if any) up front so it outlives the
    // borrowed `CallRet` handed to the call.
    let ret_layout = match &ret {
        Ret::Aggregate(ty) => Some(build_layout(*ty)?),
        _ => None,
    };
    let restype = match &ret {
        Ret::Void => host_ctypes::CallRet::Void,
        Ret::Code(c) => host_ctypes::CallRet::Code(c.as_str()),
        Ret::Pointer(_) => host_ctypes::CallRet::Code("P"),
        Ret::Aggregate(_) => host_ctypes::CallRet::Aggregate(
            ret_layout.as_ref().expect("aggregate restype has a layout"),
        ),
    };

    // Borrow the owned data as `CallArg`s; these borrows end with the call.
    let host_args: Vec<host_ctypes::CallArg> = owned
        .iter()
        .map(|o| match o {
            OwnedArg::Typed(code, buf) => host_ctypes::CallArg::Typed {
                code: code.as_str(),
                buffer: buf.as_slice(),
            },
            OwnedArg::Int(v) => host_ctypes::CallArg::Int(*v),
            OwnedArg::Double(v) => host_ctypes::CallArg::Double(*v),
            OwnedArg::Pointer(v) => host_ctypes::CallArg::Pointer(*v),
            OwnedArg::Aggregate(layout, buf) => host_ctypes::CallArg::Aggregate {
                layout,
                buffer: buf.as_slice(),
            },
        })
        .collect();

    let addr = funcptr_addr(self_obj);
    let use_errno = funcptr_flags(self_obj) & FUNCFLAG_USE_ERRNO != 0;
    let options = host_ctypes::CallOptions {
        use_errno,
        ..Default::default()
    };
    let result = host_ctypes::call(addr, &host_args, restype, options).map_err(|e| match e {
        host_ctypes::CallError::NullFunctionPointer => {
            crate::PyError::value_error("NULL function pointer")
        }
        host_ctypes::CallError::UnknownTypeCode(c) => {
            crate::PyError::type_error(format!("unsupported type code {c:?}"))
        }
        host_ctypes::CallError::BufferTooSmall { expected, got } => crate::PyError::value_error(
            format!("aggregate argument buffer too small: expected {expected}, got {got}"),
        ),
    });
    // `owned` / `keepalive` must outlive the call above.
    drop(keepalive);
    let result = result?;
    match ret {
        Ret::Pointer(rt) => {
            let p = match result {
                host_ctypes::CallValue::Pointer(p) => p,
                host_ctypes::CallValue::Scalar(b) => host_ctypes::read_pointer_from_buffer(&b),
                _ => 0,
            };
            wrap_pointer_result(rt, p)
        }
        Ret::Aggregate(ty) => {
            let bytes = match result {
                host_ctypes::CallValue::Aggregate(b) => b,
                _ => Vec::new(),
            };
            make_aggregate_instance(ty, &bytes)
        }
        Ret::Void => Ok(cdata::decoded_to_pyobject(host_ctypes::DecodedValue::None)),
        Ret::Code(c) => {
            // Reconstruct the raw result bytes and decode exactly as before:
            // a scalar carries its register image, a pointer-code result its
            // address bytes.
            let decoded = match result {
                host_ctypes::CallValue::Scalar(b) => host_ctypes::decode_type_code(&c, &b),
                host_ctypes::CallValue::Pointer(p) => {
                    host_ctypes::decode_type_code(&c, &p.to_ne_bytes())
                }
                host_ctypes::CallValue::Void => host_ctypes::DecodedValue::None,
                host_ctypes::CallValue::Aggregate(b) => host_ctypes::decode_type_code(&c, &b),
            };
            Ok(cdata::decoded_to_pyobject(decoded))
        }
    }
}

/// The `StgInfo.paramfunc` of a cdata instance's type ("simple"/"array"/
/// "pointer"/"struct"/"union"), or empty when it carries no `StgInfo`.
fn cdata_paramfunc(obj: PyObjectRef) -> String {
    let cls = unsafe { pyre_object::w_instance_get_type(obj) };
    stginfo::stginfo_of(cls)
        .map(stginfo::stginfo_paramfunc)
        .unwrap_or_default()
}

/// Whether argument type `at` lowers to a pointer (a pointer metaclass type,
/// an array type — which decays — or a simple pointer code like `P`/`z`/`Z`).
fn argtype_is_pointer_kind(at: PyObjectRef) -> bool {
    if let Some(info) = stginfo::stginfo_of(at) {
        if stginfo::stginfo_flags(info) & stginfo::TYPEFLAG_ISPOINTER != 0 {
            return true;
        }
        if stginfo::stginfo_paramfunc(info) == "array" {
            return true;
        }
    }
    matches!(cdata::type_code_of(at).as_deref(), Some(c) if cdata::is_pointer_code(c))
}

/// Whether type `t` is a by-value aggregate (struct or union).
fn is_aggregate_type(t: PyObjectRef) -> bool {
    stginfo::stginfo_of(t)
        .map(stginfo::stginfo_paramfunc)
        .is_some_and(|pf| pf == "struct" || pf == "union")
}

/// The ordered field types of a struct/union type's `_fields_` (2-tuples only;
/// bit fields are rejected at class-definition time).
fn struct_field_types(t: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let fields = unsafe { crate::baseobjspace::lookup_in_type(t, "_fields_") }
        .ok_or_else(|| crate::PyError::type_error("struct/union type has no '_fields_'"))?;
    let items = seq_to_vec(fields)
        .ok_or_else(|| crate::PyError::type_error("_fields_ must be a sequence"))?;
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        if !unsafe { pyre_object::is_tuple(it) } {
            return Err(crate::PyError::type_error(
                "_fields_ entries must be tuples",
            ));
        }
        let ft = unsafe { pyre_object::w_tuple_getitem(it, 1) }.unwrap_or(pyre_object::PY_NULL);
        if ft.is_null() || !unsafe { pyre_object::is_type(ft) } {
            return Err(crate::PyError::type_error(
                "field type must be a ctypes type",
            ));
        }
        out.push(ft);
    }
    Ok(out)
}

/// Build the recursive `CTypeLayout` of a ctypes type, driven by its `StgInfo`
/// `paramfunc`: simple → code, pointer → `Pointer`, array → element layout +
/// length, struct/union → per-field layouts from `_fields_`.
fn build_layout(t: PyObjectRef) -> Result<host_ctypes::CTypeLayout, crate::PyError> {
    use host_ctypes::CTypeLayout;
    let info = stginfo::stginfo_of(t)
        .ok_or_else(|| crate::PyError::type_error("type has no ctypes layout info"))?;
    let size = stginfo::stginfo_size(info);
    let paramfunc = stginfo::stginfo_paramfunc(info);
    match paramfunc.as_str() {
        "simple" => {
            let tc = cdata::type_code_of(t)
                .ok_or_else(|| crate::PyError::type_error("simple type has no '_type_'"))?;
            let ch = tc
                .chars()
                .next()
                .ok_or_else(|| crate::PyError::type_error("empty '_type_' code"))?;
            Ok(CTypeLayout::Simple(ch))
        }
        "pointer" => Ok(CTypeLayout::Pointer),
        "array" => {
            let element = stginfo::stginfo_proto(info)
                .ok_or_else(|| crate::PyError::type_error("array type has no element type"))?;
            Ok(CTypeLayout::Array {
                element: Box::new(build_layout(element)?),
                length: stginfo::stginfo_length(info),
                size,
            })
        }
        "struct" | "union" => {
            let mut fields = Vec::new();
            for ft in struct_field_types(t)? {
                fields.push(build_layout(ft)?);
            }
            if paramfunc == "union" {
                Ok(CTypeLayout::Union { fields, size })
            } else {
                Ok(CTypeLayout::Struct { fields, size })
            }
        }
        _ => Ok(CTypeLayout::Opaque { size }),
    }
}

/// Marshal a by-value aggregate argument `arg` of type `at`: build the layout
/// and snapshot the instance's buffer bytes (padded to the layout size).
fn marshal_aggregate_arg(arg: PyObjectRef, at: PyObjectRef) -> Result<OwnedArg, crate::PyError> {
    let layout = build_layout(at)?;
    let bytes = cdata::cdata_bytes(arg).ok_or_else(|| {
        crate::PyError::type_error("by-value aggregate argument is not a ctypes instance")
    })?;
    let buf = host_ctypes::copy_to_sized_bytes(bytes, layout.size());
    Ok(OwnedArg::Aggregate(layout, buf))
}

/// Create a fresh instance of aggregate type `ty` whose owned buffer holds the
/// returned `bytes`.
fn make_aggregate_instance(ty: PyObjectRef, bytes: &[u8]) -> Result<PyObjectRef, crate::PyError> {
    let size = stginfo::stginfo_of(ty)
        .map(stginfo::stginfo_size)
        .unwrap_or(bytes.len());
    let ba = pyre_object::w_bytearray_new(size);
    let n = bytes.len().min(size);
    unsafe {
        pyre_object::w_bytearray_data_mut(ba)[..n].copy_from_slice(&bytes[..n]);
    }
    let obj = pyre_object::w_instance_new(ty);
    let d = crate::baseobjspace::getdict(obj);
    if d.is_null() {
        return Err(crate::PyError::type_error(
            "aggregate instance has no instance dict",
        ));
    }
    unsafe { pyre_object::w_dict_setitem_str(d, "_b_", ba) };
    Ok(obj)
}

/// Marshal one argument that has an explicit `argtype` `at`.
fn marshal_typed_arg(
    arg: PyObjectRef,
    at: PyObjectRef,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<OwnedArg, crate::PyError> {
    if argtype_is_pointer_kind(at) {
        return Ok(OwnedArg::Pointer(resolve_pointer_addr(arg, keepalive)?));
    }
    // A by-value struct/union argtype.
    if is_aggregate_type(at) {
        return marshal_aggregate_arg(arg, at);
    }
    let tc = cdata::type_code_of(at)
        .ok_or_else(|| crate::PyError::type_error("argtype has no valid '_type_'"))?;
    let buf: Vec<u8> = if cdata::is_simplecdata_instance(arg) {
        cdata::cdata_bytes(arg).unwrap_or(&[]).to_vec()
    } else {
        cdata::encode_value(&tc, arg)?
    };
    Ok(OwnedArg::Typed(tc, buf))
}

/// Marshal one argument with no explicit `argtype` (ConvParam defaults).
fn marshal_default_arg(
    arg: PyObjectRef,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<OwnedArg, crate::PyError> {
    // `byref()` carrier → the address it wraps.
    if super::interp_ctypes::is_carg(arg) {
        return Ok(OwnedArg::Pointer(super::interp_ctypes::carg_ptr(arg)));
    }
    // A scalar cdata is passed by value.
    if cdata::is_simplecdata_instance(arg) {
        let cls = unsafe { pyre_object::w_instance_get_type(arg) };
        let tc = cdata::type_code_of(cls)
            .ok_or_else(|| crate::PyError::type_error("argument type has no '_type_'"))?;
        let buf = cdata::cdata_bytes(arg).unwrap_or(&[]).to_vec();
        return Ok(OwnedArg::Typed(tc, buf));
    }
    // Aggregate / pointer cdata: arrays and pointers decay to a pointer; a
    // struct/union with no `byref()` is passed by value.
    if cdata::is_cdata_instance(arg) {
        match cdata_paramfunc(arg).as_str() {
            "pointer" => {
                return Ok(OwnedArg::Pointer(host_ctypes::read_pointer_from_buffer(
                    cdata::cdata_bytes(arg).unwrap_or(&[]),
                )));
            }
            "array" => {
                return Ok(OwnedArg::Pointer(cdata::cdata_addr(arg).unwrap_or(0)));
            }
            "struct" | "union" => {
                let cls = unsafe { pyre_object::w_instance_get_type(arg) };
                return marshal_aggregate_arg(arg, cls);
            }
            _ => {}
        }
    }
    if unsafe { pyre_object::is_none(arg) } {
        Ok(OwnedArg::Pointer(0))
    } else if unsafe { pyre_object::is_bytes(arg) } {
        Ok(OwnedArg::Pointer(bytes_pointer_addr(arg, keepalive)))
    } else if unsafe { pyre_object::is_str(arg) } {
        Err(str_arg_unsupported())
    } else if unsafe { pyre_object::is_float(arg) } {
        Ok(OwnedArg::Double(crate::baseobjspace::float_w(arg)?))
    } else if unsafe { pyre_object::is_int(arg) } {
        Ok(OwnedArg::Int(crate::baseobjspace::int_w(arg)? as i32))
    } else {
        Err(crate::PyError::type_error(
            "Don't know how to convert parameter",
        ))
    }
}

/// Resolve the address a pointer-kind argument lowers to: `byref()` carriers,
/// `_Pointer`/`Array`/`Structure` instances, pointer-typed scalars, bytes, an
/// integer address, or `None`.
fn resolve_pointer_addr(
    arg: PyObjectRef,
    keepalive: &mut Vec<Vec<u8>>,
) -> Result<usize, crate::PyError> {
    if super::interp_ctypes::is_carg(arg) {
        return Ok(super::interp_ctypes::carg_ptr(arg));
    }
    if cdata::is_simplecdata_instance(arg) {
        // A pointer-typed scalar stores the target address in its buffer.
        return Ok(host_ctypes::read_pointer_from_buffer(
            cdata::cdata_bytes(arg).unwrap_or(&[]),
        ));
    }
    if cdata::is_cdata_instance(arg) {
        // `_Pointer` → stored address; `Array`/`Structure` → buffer address.
        return Ok(match cdata_paramfunc(arg).as_str() {
            "pointer" => {
                host_ctypes::read_pointer_from_buffer(cdata::cdata_bytes(arg).unwrap_or(&[]))
            }
            _ => cdata::cdata_addr(arg).unwrap_or(0),
        });
    }
    if unsafe { pyre_object::is_none(arg) } {
        Ok(0)
    } else if unsafe { pyre_object::is_int(arg) } {
        Ok(crate::baseobjspace::int_w(arg)? as usize)
    } else if unsafe { pyre_object::is_bytes(arg) } {
        Ok(bytes_pointer_addr(arg, keepalive))
    } else if unsafe { pyre_object::is_str(arg) } {
        Err(str_arg_unsupported())
    } else {
        Err(crate::PyError::type_error(
            "expected bytes, integer address, ctypes instance, or None",
        ))
    }
}

/// Null-terminate a `bytes` payload, keep the copy alive, and return the
/// address of the copy.
fn bytes_pointer_addr(arg: PyObjectRef, keepalive: &mut Vec<Vec<u8>>) -> usize {
    let raw = unsafe { pyre_object::bytesobject::w_bytes_data(arg) };
    keepalive.push(host_ctypes::null_terminated_bytes(raw));
    // The inner Vec's heap buffer is stable even if `keepalive` reallocates.
    keepalive.last().unwrap().as_ptr() as usize
}

fn str_arg_unsupported() -> crate::PyError {
    crate::PyError::type_error(
        "str argument marshalling (wchar_t*) is not implemented in this ctypes slice; \
         pass bytes for char* arguments",
    )
}
