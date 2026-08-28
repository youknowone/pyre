//! `_cffi_backend._CDataBase` — PyPy:
//! `pypy/module/_cffi_backend/cdataobj.py`.
//!
//! `W_CData.typedef.acceptable_as_base_class = False` and none of the
//! subclasses (`W_CDataMem`, `W_CDataNewStd`, `W_CDataSliced`, …) declares a
//! typedef of its own, so the family is one Python type here as well; the
//! subclass is carried as [`W_CData::flavor`].

use crate::PyError;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;

use super::ctypeobj::{self, W_CType};

// ── the subclass discriminant ───────────────────────────────────────────

/// `cdataobj.py W_CData` itself: a view that owns nothing.
pub const FLAVOR_STATIC: i64 = 0;
/// `W_CDataMem` — the buffer `cast()` allocates for a primitive value.
pub const FLAVOR_MEM: i64 = 1;
/// `W_CDataNewStd` — what `newp()` returns through the default allocator.
pub const FLAVOR_NEW_STD: i64 = 2;
/// `W_CDataSliced` — a slice of an array or pointer, with its own length.
pub const FLAVOR_SLICED: i64 = 3;
/// `W_CDataPtrToStructOrUnion` — what `newp()` returns for a `struct *`: it
/// owns nothing itself and co-owns the `W_CDataNewStd` holding the struct.
pub const FLAVOR_PTR_TO_STRUCT: i64 = 4;

/// `cdataobj.py W_CData` and the RPython subclasses sharing its typedef.
#[crate::pyre_class("_cffi_backend._CDataBase")]
#[derive(Default)]
pub struct W_CData {
    /// `W_CData.ctype`.
    pub ctype: PyObjectRef,
    /// `W_CData._ptr`.
    pub ptr: *mut u8,
    /// Which RPython subclass this is; one of the `FLAVOR_*`.
    pub flavor: i64,
    /// `W_CDataNewOwning.allocated_length` for an owning cdata, and
    /// `W_CDataSliced.length` for a slice.  -1 when neither applies.
    pub length: i64,
    /// `W_CDataNewStd.datasize`, which becomes -1 once the memory is freed.
    pub datasize: i64,
    /// Whatever the cdata must keep alive: `W_CDataPtrToStructOrUnion`'s
    /// `structobj`, `W_CDataFromBuffer`'s `w_keepalive`, `W_CDataGCP`'s
    /// `w_original_cdata`.
    pub w_keepalive: PyObjectRef,
    /// `W_CDataGCP.w_destructor`.
    pub w_destructor: PyObjectRef,
}

impl W_CData {
    fn ctype_ref(&self) -> Result<&'static mut W_CType, PyError> {
        ctypeobj::ctype_at(self.ctype).ok_or_else(|| PyError::system_error("cdata without a ctype"))
    }

    /// `W_CData._sizeof` and the overrides of it.
    fn sizeof(&self) -> Result<i64, PyError> {
        let ct = self.ctype_ref()?;
        match self.flavor {
            FLAVOR_NEW_STD if self.length >= 0 => {
                if ct.kind == ctypeobj::KIND_ARRAY {
                    let item = ctypeobj::ctype_at(ct.ctitem)
                        .ok_or_else(|| PyError::system_error("array without an item type"))?;
                    Ok(self.length * item.size)
                } else {
                    // A var-sized struct records its total size directly.
                    Ok(self.length)
                }
            }
            FLAVOR_SLICED => {
                let item = ctypeobj::ctype_at(ct.ctitem)
                    .ok_or_else(|| PyError::system_error("slice without an item type"))?;
                Ok(self.length * item.size)
            }
            _ => Ok(ct.size),
        }
    }

    /// `W_CData.get_array_length`.
    pub fn array_length(&self) -> Result<i64, PyError> {
        match self.flavor {
            FLAVOR_NEW_STD | FLAVOR_SLICED => Ok(self.length),
            _ => {
                let ct = self.ctype_ref()?;
                Ok(ct.length)
            }
        }
    }

    /// `W_CData._repr_extra` and the overrides of it.
    pub fn repr_extra(&self) -> Result<String, PyError> {
        let ct = self.ctype_ref()?;
        match self.flavor {
            FLAVOR_NEW_STD => {
                // `W_CData._repr_extra_owning`: a pointer reports the size of
                // what it points at, not the size of the pointer.
                let bytes = if ct.kind == ctypeobj::KIND_POINTER {
                    ctypeobj::ctype_at(ct.ctitem)
                        .ok_or_else(|| PyError::system_error("pointer without an item type"))?
                        .size
                } else {
                    self.sizeof()?
                };
                Ok(format!("owning {bytes} bytes"))
            }
            FLAVOR_SLICED => Ok(format!("sliced length {}", self.length)),
            // `W_CDataPtrToStructOrUnion._repr_extra`.
            FLAVOR_PTR_TO_STRUCT => match W_CData::from_obj(self.w_keepalive) {
                Some(structobj) => structobj.repr_extra(),
                None => Ok("NULL".to_string()),
            },
            _ => unsafe { ct.extra_repr(self.ptr) },
        }
    }
}

/// `@unwrap_spec(w_cdata=cdataobj.W_CData)`.
pub fn cdata_arg(w_cdata: PyObjectRef) -> Result<&'static mut W_CData, PyError> {
    W_CData::from_obj(w_cdata).ok_or_else(|| {
        PyError::type_error(format!(
            "expected a cdata object, got '{}'",
            crate::type_methods::arg_type_name(w_cdata)
        ))
    })
}

// ── construction ────────────────────────────────────────────────────────

/// `cdataobj.py W_CData(space, ptr, ctype)` — a view that owns no memory.
pub fn new_cdata(ptr: *mut u8, w_ctype: PyObjectRef) -> PyObjectRef {
    new_cdata_full(ptr, w_ctype, FLAVOR_STATIC, -1, -1, pyre_object::PY_NULL)
}

/// `cdataobj.py W_CDataSliced(space, ptr, ctype, length)`.
pub fn new_cdata_sliced(ptr: *mut u8, w_ctype: PyObjectRef, length: i64) -> PyObjectRef {
    new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_SLICED,
        length,
        -1,
        pyre_object::PY_NULL,
    )
}

/// `cdataobj.py W_CDataMem(space, ctype)` — `ctype.size` bytes of raw memory
/// this cdata owns and frees.
pub fn new_cdata_mem(w_ctype: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let size = ctypeobj::ctype_arg(w_ctype)?.size.max(0);
    let ptr = raw_alloc(size, false)?;
    Ok(new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_MEM,
        -1,
        size,
        pyre_object::PY_NULL,
    ))
}

/// `W_CDataPtrToStructOrUnion.__init__` — a pointer that co-owns the cdata
/// really holding the struct.
pub fn new_cdata_ptr_to_struct(
    ptr: *mut u8,
    w_ctype: PyObjectRef,
    w_structobj: PyObjectRef,
) -> PyObjectRef {
    new_cdata_full(ptr, w_ctype, FLAVOR_PTR_TO_STRUCT, -1, -1, w_structobj)
}

/// `W_CTypeStructOrUnion.copy_and_convert_to_object` — a `W_CDataNewStd`
/// holding a copy of what `source` points at.
///
/// # Safety
/// `source` must be readable for `size` bytes.
pub unsafe fn new_cdata_copy(
    w_ctype: PyObjectRef,
    source: *const u8,
    size: i64,
) -> Result<PyObjectRef, PyError> {
    let ptr = raw_alloc(size, false)?;
    unsafe { std::ptr::copy_nonoverlapping(source, ptr, size.max(0) as usize) };
    Ok(new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_NEW_STD,
        -1,
        size,
        pyre_object::PY_NULL,
    ))
}

/// `W_CData.get_structobj` and the two overrides of it — the owning cdata a
/// variable-length array reads its bound from.
pub fn structobj_of(w_cdata: PyObjectRef) -> Option<&'static mut W_CData> {
    let cdata = W_CData::from_obj(w_cdata)?;
    match cdata.flavor {
        FLAVOR_NEW_STD => Some(cdata),
        FLAVOR_PTR_TO_STRUCT => {
            W_CData::from_obj(cdata.w_keepalive).filter(|s| s.flavor == FLAVOR_NEW_STD)
        }
        _ => None,
    }
}

/// `allocator.py default_allocator.allocate` — zeroed memory owned by the
/// cdata it hands back.
pub fn new_cdata_owning(
    w_ctype: PyObjectRef,
    datasize: i64,
    length: i64,
) -> Result<PyObjectRef, PyError> {
    let ptr = raw_alloc(datasize, true)?;
    Ok(new_cdata_full(
        ptr,
        w_ctype,
        FLAVOR_NEW_STD,
        length,
        datasize,
        pyre_object::PY_NULL,
    ))
}

fn new_cdata_full(
    ptr: *mut u8,
    w_ctype: PyObjectRef,
    flavor: i64,
    length: i64,
    datasize: i64,
    w_keepalive: PyObjectRef,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let ctype_slot = roots.base();
    let _ = roots.pin_root(w_ctype);
    let keepalive_slot = ctype_slot + 1;
    let _ = roots.pin_root(w_keepalive);
    // The struct literal is built before `allocate_stable` runs, and that
    // allocation is itself a collection point, so the movable keepalive is
    // stored from its slot afterwards rather than from the literal.
    let obj = W_CData::allocate_stable(W_CData {
        ctype: roots.get(ctype_slot),
        ptr,
        flavor,
        length,
        datasize,
        ..Default::default()
    });
    W_CData::from_obj(obj)
        .expect("allocate_stable hands back this layout")
        .w_keepalive = roots.get(keepalive_slot);
    // The cdata is born old-gen; the keepalive it just took may be young, so
    // the barrier has to run again after this write.
    pyre_object::gc_hook::try_gc_write_barrier_managed(obj.cast::<u8>());
    obj
}

/// `lltype.malloc(rffi.CCHARP.TO, size, flavor='raw')`.  A zero-size request
/// still returns a distinct address, so `newp` on an empty array does not
/// hand back NULL.
pub fn raw_alloc(size: i64, zero: bool) -> Result<*mut u8, PyError> {
    if size < 0 {
        return Err(PyError::value_error("negative allocation size"));
    }
    let bytes = size.max(1) as usize;
    let ptr = unsafe {
        if zero {
            libc::calloc(bytes, 1)
        } else {
            libc::malloc(bytes)
        }
    };
    if ptr.is_null() {
        return Err(PyError::new(
            crate::PyErrorKind::MemoryError,
            "out of memory",
        ));
    }
    Ok(ptr.cast::<u8>())
}

/// `lltype.free(self._ptr, flavor='raw')` in the light finalizers of
/// `W_CDataMem` and `W_CDataNewStd`.
///
/// # Safety
/// `obj` must be a GC-dead `W_CData`.
pub unsafe fn w_cdata_dealloc(obj: PyObjectRef) {
    let cdata = unsafe { &mut *(obj as *mut W_CData) };
    if matches!(cdata.flavor, FLAVOR_MEM | FLAVOR_NEW_STD) && cdata.datasize >= 0 {
        unsafe { libc::free(cdata.ptr.cast::<libc::c_void>()) };
    }
    cdata.ptr = std::ptr::null_mut();
    cdata.datasize = -1;
}

// ── the Python type ─────────────────────────────────────────────────────

static CDATA_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// `_cffi_backend._CDataBase`.
pub fn cdata_type() -> PyObjectRef {
    *CDATA_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_cffi_backend._CDataBase",
            init_cdata_type,
            crate::typedef::w_object(),
            <W_CData as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_CData as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp as usize
    }) as PyObjectRef
}

const CDATA_DOC: &str = "The internal base type for CData objects.  Use FFI.CData to access it.  Always check with isinstance(): subtypes are sometimes returned on CPython, for performance reasons.";

fn init_cdata_type(ns: PyObjectRef) {
    let store = |name: &str, value: PyObjectRef| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value)
    };
    store("__doc__", pyre_object::w_str_new(CDATA_DOC));
    // Both are typedef entries in PyPy, so they answer on an instance too.
    store("__module__", pyre_object::w_str_new("_cffi_backend"));
    store("__name__", pyre_object::w_str_new("<cdata>"));
    store("__weakref__", crate::typedef::make_weakref_descr(ns));
    for (name, f, arity) in [
        (
            "__repr__",
            cdata_repr as crate::gateway::BuiltinCodeFn,
            1u16,
        ),
        ("__bool__", cdata_bool, 1),
        ("__int__", cdata_int, 1),
        ("__float__", cdata_float, 1),
        ("__complex__", cdata_complex, 1),
        ("__len__", cdata_len, 1),
        ("__hash__", cdata_hash, 1),
        ("__dir__", cdata_dir, 1),
        ("__enter__", cdata_enter, 1),
        ("__iter__", cdata_iter, 1),
        ("__lt__", cdata_lt, 2),
        ("__le__", cdata_le, 2),
        ("__eq__", cdata_eq, 2),
        ("__ne__", cdata_ne, 2),
        ("__gt__", cdata_gt, 2),
        ("__ge__", cdata_ge, 2),
        ("__getitem__", cdata_getitem, 2),
        ("__add__", cdata_add, 2),
        ("__radd__", cdata_add, 2),
        ("__sub__", cdata_sub, 2),
        ("__getattr__", cdata_getattr, 2),
        ("__setitem__", cdata_setitem, 3),
        ("__setattr__", cdata_setattr, 3),
    ] {
        store(
            name,
            crate::make_builtin_function_with_arity(name, f, arity),
        );
    }
    // `__call__` and `__exit__` take a variable number of arguments.
    store(
        "__call__",
        crate::make_builtin_function("__call__", cdata_call),
    );
    store(
        "__exit__",
        crate::make_builtin_function("__exit__", cdata_exit),
    );
}

/// `W_CData.repr`.
fn cdata_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    let extra = cdata.repr_extra()?;
    Ok(pyre_object::w_str_new(&format!(
        "<cdata '{}' {extra}>",
        ct.name()
    )))
}

/// `W_CData.bool`.
fn cdata_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    Ok(pyre_object::boolobject::w_bool_from(unsafe {
        ctypeobj::nonzero(ct, cdata.ptr)?
    }))
}

/// `W_CData.int`.
fn cdata_int(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    unsafe { ctypeobj::cast_to_int(ct, cdata.ptr) }
}

/// `W_CData.float`.
fn cdata_float(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    unsafe { ctypeobj::float(ct, cdata.ptr) }
}

/// `W_CData.complex`.
fn cdata_complex(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    unsafe { ctypeobj::complex(ct, cdata.ptr) }
}

/// `W_CData.len`.
fn cdata_len(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    if ct.kind != ctypeobj::KIND_ARRAY {
        return Err(PyError::type_error(format!(
            "cdata of type '{}' has no len()",
            ct.name()
        )));
    }
    Ok(pyre_object::w_int_new(cdata.array_length()?))
}

/// `W_CData.hash`.
fn cdata_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    if ct.is_primitive() {
        let w_ob = unsafe { ctypeobj::convert_to_object(ct, cdata.ptr)? };
        if W_CData::from_obj(w_ob).is_none() {
            return Ok(pyre_object::w_int_new(crate::baseobjspace::hash_w_strict(
                w_ob,
            )?));
        }
    }
    // Pointers are hashed by address, folded so the always-zero alignment
    // bits do not end up in every key's low bits.
    let h = cdata.ptr as i64;
    Ok(pyre_object::w_int_new(h ^ (h >> 4)))
}

/// `W_CData.dir` — the fields of what the cdata points at.
fn cdata_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let mut ct = cdata.ctype_ref()?;
    if ct.kind == ctypeobj::KIND_POINTER {
        ct = ctypeobj::ctype_at(ct.ctitem)
            .ok_or_else(|| PyError::system_error("pointer without an item type"))?;
    }
    // `W_CType.cdata_dir` is empty for every ctype a struct is not.
    let names = if ct.is_struct_or_union() {
        super::ctypestruct::cdata_dir(ct)?
    } else {
        Vec::new()
    };
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    for name in &names {
        let _ = roots.pin_root(pyre_object::w_str_new(name));
    }
    let items = (0..names.len()).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}

/// `W_CData.descr_enter`.
fn cdata_enter(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    enter_exit(args[0], false)?;
    Ok(args[0])
}

/// `W_CData.descr_exit`.
fn cdata_exit(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    enter_exit(args[0], true)?;
    Ok(pyre_object::w_none())
}

/// `W_CData.enter_exit` and the overrides of it.
pub fn enter_exit(w_cdata: PyObjectRef, exit_now: bool) -> Result<(), PyError> {
    let cdata = cdata_arg(w_cdata)?;
    // `W_CDataPtrToStructOrUnion.enter_exit` releases the struct it co-owns.
    if cdata.flavor == FLAVOR_PTR_TO_STRUCT {
        if exit_now && !cdata.w_keepalive.is_null() {
            return enter_exit(cdata.w_keepalive, true);
        }
        return Ok(());
    }
    let ct = cdata.ctype_ref()?;
    if cdata.flavor != FLAVOR_NEW_STD || !ct.is_ptr_or_array() {
        return Err(PyError::value_error(
            "only 'cdata' object from ffi.new(), ffi.gc(), ffi.from_buffer() or ffi.new_allocator()() can be used with the 'with' keyword or ffi.release()",
        ));
    }
    if exit_now && cdata.datasize >= 0 {
        // `W_CDataNewStd._do_exit`.
        cdata.datasize = -1;
        unsafe { libc::free(cdata.ptr.cast::<libc::c_void>()) };
        cdata.ptr = std::ptr::null_mut();
    }
    Ok(())
}

/// `W_CData.iter`.
fn cdata_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    if ct.kind != ctypeobj::KIND_ARRAY {
        return Err(PyError::type_error(format!(
            "cdata '{}' does not support iteration",
            ct.name()
        )));
    }
    super::ctypearray::new_cdata_iter(args[0])
}

/// `W_CData.call`.
fn cdata_call(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    if ct.kind == ctypeobj::KIND_FUNC {
        return super::ctypefunc::call(ct, cdata.ptr, &args[1..]);
    }
    Err(PyError::type_error(format!(
        "cdata '{}' is not callable",
        ct.name()
    )))
}

/// `W_CData.getattr`.
fn cdata_getattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    let field = ctypeobj::getcfield(ct, crate::baseobjspace::text_w(args[1])?, "read")?;
    unsafe { super::ctypestruct::read(field, cdata.ptr, args[0]) }
}

/// `W_CData.setattr`.
fn cdata_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    // The value outlives a conversion that runs arbitrary Python, so it is
    // read back out of its slot.
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(args[2]);
    let cdata = cdata_arg(args[0])?;
    let ct = cdata.ctype_ref()?;
    let field = ctypeobj::getcfield(ct, crate::baseobjspace::text_w(args[1])?, "write")?;
    unsafe { super::ctypestruct::write(field, cdata.ptr, roots.get(value_slot))? };
    Ok(pyre_object::w_none())
}

// ── comparison ──────────────────────────────────────────────────────────

/// `W_CData._compare_mode`'s three answers.
enum CompareMode {
    /// Both operands are pointer-like: compare the addresses.
    Addresses(usize, usize),
    /// Exactly one operand is pointer-like: the comparison is undefined.
    Incomparable,
    /// Neither is: compare what they hold.
    Objects(PyObjectRef, PyObjectRef),
}

fn compare_mode(w_self: PyObjectRef, w_other: PyObjectRef) -> Result<CompareMode, PyError> {
    let cdata = cdata_arg(w_self)?;
    let self_is_ptr = !cdata.ctype_ref()?.is_primitive();
    let other = W_CData::from_obj(w_other);
    let other_is_ptr = other
        .as_ref()
        .and_then(|o| ctypeobj::ctype_at(o.ctype))
        .is_some_and(|ct| !ct.is_primitive());
    if self_is_ptr && other_is_ptr {
        let other = other.expect("other_is_ptr implies a cdata");
        return Ok(CompareMode::Addresses(
            cdata.ptr as usize,
            other.ptr as usize,
        ));
    }
    if self_is_ptr || other_is_ptr {
        return Ok(CompareMode::Incomparable);
    }
    // Boxing this side's value allocates, so the other operand — which the
    // non-cdata arm hands straight back — has to survive it.
    let roots = pyre_object::gc_roots::push_roots();
    let other_slot = roots.base();
    let _ = roots.pin_root(w_other);
    let ob1_slot = other_slot + 1;
    let _ = roots.pin_root(unsafe { ctypeobj::convert_to_object(cdata.ctype_ref()?, cdata.ptr)? });
    let w_ob2 = match other {
        Some(other) => unsafe { ctypeobj::convert_to_object(other.ctype_ref()?, other.ptr)? },
        None => roots.get(other_slot),
    };
    Ok(CompareMode::Objects(roots.get(ob1_slot), w_ob2))
}

fn compare(
    args: &[PyObjectRef],
    op: crate::bytecode::ComparisonOperator,
    on_addresses: fn(usize, usize) -> bool,
) -> Result<PyObjectRef, PyError> {
    match compare_mode(args[0], args[1])? {
        CompareMode::Addresses(a, b) => {
            Ok(pyre_object::boolobject::w_bool_from(on_addresses(a, b)))
        }
        CompareMode::Incomparable => Ok(pyre_object::special::w_not_implemented()),
        CompareMode::Objects(a, b) => crate::opcode_ops::compare_value(a, b, op),
    }
}

macro_rules! comparison {
    ($name:ident, $op:ident, $addr:expr) => {
        fn $name(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
            compare(args, crate::bytecode::ComparisonOperator::$op, $addr)
        }
    };
}

comparison!(cdata_lt, Less, |a, b| a < b);
comparison!(cdata_le, LessOrEqual, |a, b| a <= b);
comparison!(cdata_eq, Equal, |a, b| a == b);
comparison!(cdata_ne, NotEqual, |a, b| a != b);
comparison!(cdata_gt, Greater, |a, b| a > b);
comparison!(cdata_ge, GreaterOrEqual, |a, b| a >= b);

// ── indexing ────────────────────────────────────────────────────────────

/// `W_CData.getitem`.
fn cdata_getitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_index = args[1];
    if unsafe { pyre_object::sliceobject::is_slice(w_index) } {
        return do_getslice(w_self, w_index);
    }
    let i = crate::baseobjspace::getindex_w(w_index)?;
    let cdata = cdata_arg(w_self)?;
    let ct = check_subscript_index(cdata, i)?;
    // `W_CDataPtrToStructOrUnion._do_getitem` — `p[0]` is the struct itself.
    if cdata.flavor == FLAVOR_PTR_TO_STRUCT {
        return Ok(cdata.w_keepalive);
    }
    let item = ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("indexed ctype without an item type"))?;
    unsafe { ctypeobj::convert_to_object(item, cdata.ptr.offset((i * item.size) as isize)) }
}

/// `W_CData.setitem`.
fn cdata_setitem(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_index = args[1];
    let w_value = args[2];
    if unsafe { pyre_object::sliceobject::is_slice(w_index) } {
        do_setslice(w_self, w_index, w_value)?;
        return Ok(pyre_object::w_none());
    }
    // The index's `__index__` is arbitrary Python, so the value being stored
    // has to be read back out of its slot afterwards.
    let roots = pyre_object::gc_roots::push_roots();
    let value_slot = roots.base();
    let _ = roots.pin_root(w_value);
    let i = crate::baseobjspace::getindex_w(w_index)?;
    let cdata = cdata_arg(w_self)?;
    let ct = check_subscript_index(cdata, i)?;
    let item = ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("indexed ctype without an item type"))?;
    unsafe {
        ctypeobj::convert_from_object(
            item,
            cdata.ptr.offset((i * item.size) as isize),
            roots.get(value_slot),
        )?;
    }
    Ok(pyre_object::w_none())
}

/// `W_CType._check_subscript_index` and the overrides of it, returning the
/// ctype whose `ctitem` names the element.
fn check_subscript_index(cdata: &W_CData, i: i64) -> Result<&'static mut W_CType, PyError> {
    let ct = cdata.ctype_ref()?;
    match ct.kind {
        // `W_CTypePointer._check_subscript_index`.
        ctypeobj::KIND_POINTER => {
            if cdata.flavor == FLAVOR_NEW_STD || cdata.flavor == FLAVOR_PTR_TO_STRUCT {
                if i != 0 {
                    return Err(PyError::index_error(format!(
                        "cdata '{}' can only be indexed by 0",
                        ct.name()
                    )));
                }
            } else if cdata.ptr.is_null() {
                return Err(PyError::runtime_error(format!(
                    "cannot dereference null pointer from cdata '{}'",
                    ct.name()
                )));
            }
            Ok(ct)
        }
        // `W_CTypeArray._check_subscript_index`.
        ctypeobj::KIND_ARRAY => {
            if i < 0 {
                return Err(PyError::index_error("negative index"));
            }
            let length = cdata.array_length()?;
            if i >= length {
                return Err(PyError::index_error(format!(
                    "index too large for cdata '{}' (expected {i} < {length})",
                    ct.name()
                )));
            }
            Ok(ct)
        }
        _ => Err(PyError::type_error(format!(
            "cdata of type '{}' cannot be indexed",
            ct.name()
        ))),
    }
}

/// `W_CData._do_getslicearg` — the pointer ctype the slice reads through,
/// the start index, and the length.
fn getslicearg(
    w_self: PyObjectRef,
    w_slice: PyObjectRef,
) -> Result<(PyObjectRef, i64, i64), PyError> {
    // Each bound's `__index__` is arbitrary Python, so all three components
    // are pinned before the first conversion runs.
    let roots = pyre_object::gc_roots::push_roots();
    let start_slot = roots.base();
    unsafe {
        let _ = roots.pin_root(pyre_object::sliceobject::w_slice_get_start(w_slice));
        let _ = roots.pin_root(pyre_object::sliceobject::w_slice_get_stop(w_slice));
        let _ = roots.pin_root(pyre_object::sliceobject::w_slice_get_step(w_slice));
    }
    let (stop_slot, step_slot) = (start_slot + 1, start_slot + 2);
    if unsafe { pyre_object::pyobject::is_none(roots.get(start_slot)) } {
        return Err(PyError::index_error("slice start must be specified"));
    }
    let start = crate::baseobjspace::int_w(roots.get(start_slot))?;
    if unsafe { pyre_object::pyobject::is_none(roots.get(stop_slot)) } {
        return Err(PyError::index_error("slice stop must be specified"));
    }
    let stop = crate::baseobjspace::int_w(roots.get(stop_slot))?;
    if !unsafe { pyre_object::pyobject::is_none(roots.get(step_slot)) } {
        return Err(PyError::index_error("slice with step not supported"));
    }
    if start > stop {
        return Err(PyError::index_error("slice start > stop"));
    }
    let cdata = cdata_arg(w_self)?;
    let ct = cdata.ctype_ref()?;
    // `W_CType._check_slice_index` and the overrides of it.
    let w_ctptr = match ct.kind {
        ctypeobj::KIND_POINTER => cdata.ctype,
        ctypeobj::KIND_ARRAY => {
            if start < 0 {
                return Err(PyError::index_error("negative index"));
            }
            let length = cdata.array_length()?;
            if stop > length {
                return Err(PyError::index_error(format!(
                    "index too large (expected {stop} <= {length})"
                )));
            }
            ct.ctptr
        }
        _ => {
            return Err(PyError::type_error(format!(
                "cdata of type '{}' cannot be indexed",
                ct.name()
            )));
        }
    };
    Ok((w_ctptr, start, stop - start))
}

/// `W_CData._do_getslice`.
fn do_getslice(w_self: PyObjectRef, w_slice: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let (w_ctptr, start, length) = getslicearg(w_self, w_slice)?;
    let roots = pyre_object::gc_roots::push_roots();
    let self_slot = roots.base();
    let _ = roots.pin_root(w_self);
    let w_ctarray = super::newtype::cached_unbounded_array_type(w_ctptr)?;
    let array_slot = self_slot + 1;
    let _ = roots.pin_root(w_ctarray);
    let item_size = ctypeobj::ctype_at(ctypeobj::ctype_arg(roots.get(array_slot))?.ctitem)
        .ok_or_else(|| PyError::system_error("array without an item type"))?
        .size;
    let cdata = cdata_arg(roots.get(self_slot))?;
    let ptr = unsafe { cdata.ptr.offset((start * item_size) as isize) };
    Ok(new_cdata_sliced(ptr, roots.get(array_slot), length))
}

/// `W_CData._do_setslice`.
fn do_setslice(
    w_self: PyObjectRef,
    w_slice: PyObjectRef,
    w_value: PyObjectRef,
) -> Result<(), PyError> {
    // The slice bounds run `__index__`, so the assigned value is pinned first
    // and read back out of its slot for every arm below.
    let value_roots = pyre_object::gc_roots::push_roots();
    let value_slot = value_roots.base();
    let _ = value_roots.pin_root(w_value);
    let (w_ctptr, start, length) = getslicearg(w_self, w_slice)?;
    let w_value = value_roots.get(value_slot);
    let ctptr = ctypeobj::ctype_arg(w_ctptr)?;
    let item = ctypeobj::ctype_at(ctptr.ctitem)
        .ok_or_else(|| PyError::system_error("pointer without an item type"))?;
    let item_size = item.size;
    let cdata = cdata_arg(w_self)?;
    let target = unsafe { cdata.ptr.offset((start * item_size) as isize) };

    // The fast path: copying from an array of exactly the item type.
    if let Some(source) = W_CData::from_obj(w_value)
        && let Some(source_ct) = ctypeobj::ctype_at(source.ctype)
        && source_ct.kind == ctypeobj::KIND_ARRAY
        && std::ptr::eq(
            ctypeobj::ctype_at(source_ct.ctitem).map_or(std::ptr::null(), |c| c as *const W_CType),
            item as *const W_CType,
        )
        && source.array_length()? == length
    {
        unsafe {
            std::ptr::copy_nonoverlapping(source.ptr, target, (item_size * length) as usize);
        }
        return Ok(());
    }

    // `<char[]>[0:N] = b"somestring"`.
    if item.is_primitive()
        && item_size == 1
        && unsafe { pyre_object::bytesobject::is_bytes(w_value) }
    {
        let value = unsafe { pyre_object::bytesobject::w_bytes_data(w_value) };
        if value.len() as i64 != length {
            return Err(PyError::value_error(format!(
                "need a string of length {length}, got {}",
                value.len()
            )));
        }
        unsafe { std::ptr::copy_nonoverlapping(value.as_ptr(), target, value.len()) };
        return Ok(());
    }

    // `W_CData._do_setslice_iterate`.
    let items = crate::baseobjspace::unpackiterable(w_value, -1)?;
    if items.len() as i64 != length {
        return Err(if (items.len() as i64) < length {
            PyError::value_error(format!(
                "need {length} values to unpack, got {}",
                items.len()
            ))
        } else {
            PyError::value_error(format!("got more than {length} values to unpack"))
        });
    }
    let roots = pyre_object::gc_roots::push_roots();
    let base = pyre_object::gc_roots::pin_roots(&items);
    for i in 0..items.len() {
        let element = unsafe { target.offset((i as i64 * item_size) as isize) };
        unsafe { ctypeobj::convert_from_object(item, element, roots.get(base + i))? };
    }
    Ok(())
}

// ── arithmetic ──────────────────────────────────────────────────────────

/// `W_CData.add`, which is also `__radd__`.
fn cdata_add(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    add_or_sub(args[0], args[1], 1)
}

/// `W_CData.sub`.
fn cdata_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let w_other = args[1];
    let Some(other) = W_CData::from_obj(w_other) else {
        return add_or_sub(w_self, w_other, -1);
    };
    let cdata = cdata_arg(w_self)?;
    let self_ct = cdata.ctype_ref()?;
    let other_ct = ctypeobj::ctype_at(other.ctype)
        .ok_or_else(|| PyError::system_error("cdata without a ctype"))?;
    // An array is compared through the pointer type it decays to.
    let other_ct = if other_ct.kind == ctypeobj::KIND_ARRAY {
        ctypeobj::ctype_at(other_ct.ctptr)
            .ok_or_else(|| PyError::system_error("array without a pointer type"))?
    } else {
        other_ct
    };
    let item_size = ctypeobj::ctype_at(other_ct.ctitem).map_or(-1, |it| it.size);
    if !std::ptr::eq(self_ct as *const W_CType, other_ct as *const W_CType)
        || other_ct.kind != ctypeobj::KIND_POINTER
        || (item_size <= 0 && !other_ct.has(ctypeobj::F_VOID_PTR))
    {
        return Err(PyError::type_error(format!(
            "cannot subtract cdata '{}' and cdata '{}'",
            self_ct.name(),
            other_ct.name()
        )));
    }
    let mut diff = cdata.ptr as i64 - other.ptr as i64;
    if item_size > 1 {
        if diff % item_size != 0 {
            return Err(PyError::value_error(
                "pointer subtraction: the distance between the two pointers is not a multiple of the item size",
            ));
        }
        diff /= item_size;
    }
    Ok(pyre_object::w_int_new(diff))
}

/// `W_CData._add_or_sub`.
fn add_or_sub(
    w_self: PyObjectRef,
    w_other: PyObjectRef,
    sign: i64,
) -> Result<PyObjectRef, PyError> {
    let i = sign * crate::baseobjspace::getindex_w(w_other)?;
    let cdata = cdata_arg(w_self)?;
    ctypeobj::add(cdata.ctype, cdata.ptr, i)
}

// ── module-level entry points that read a cdata ─────────────────────────

/// `func.py sizeof` on a cdata.
pub fn cdata_sizeof(w_cdata: PyObjectRef) -> Result<i64, PyError> {
    cdata_arg(w_cdata)?.sizeof()
}

/// `W_CData.unpack`.
pub fn unpack(w_cdata: PyObjectRef, length: i64) -> Result<PyObjectRef, PyError> {
    let cdata = cdata_arg(w_cdata)?;
    let ct = cdata.ctype_ref()?;
    if !ct.has(ctypeobj::F_NONFUNC_POINTER_OR_ARRAY) {
        return Err(PyError::type_error(format!(
            "expected a pointer or array, got '{}'",
            ct.name()
        )));
    }
    if length < 0 {
        return Err(PyError::value_error("'length' cannot be negative"));
    }
    if cdata.ptr.is_null() {
        let w_repr = crate::builtins::builtin_repr(&[w_cdata])?;
        return Err(PyError::runtime_error(format!(
            "cannot use unpack() on {}",
            unsafe { pyre_object::w_str_get_value(w_repr) }
        )));
    }
    let item = ctypeobj::ctype_at(ct.ctitem)
        .ok_or_else(|| PyError::system_error("pointer without an item type"))?;
    super::ctypeptr::unpack_ptr(ct, item, cdata.ptr, length)
}
