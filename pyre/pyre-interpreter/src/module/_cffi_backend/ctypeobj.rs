//! `_cffi_backend.CType` — PyPy: `pypy/module/_cffi_backend/ctypeobj.py`
//! and every RPython subclass of `W_CType` (`ctypevoid`, `ctypeprim`,
//! `ctypeptr`, `ctypearray`).
//!
//! `W_CType.typedef.acceptable_as_base_class = False` and none of the
//! subclasses declares a typedef of its own, so the whole family is one
//! Python type.  What the RPython hierarchy expresses as a class, this
//! module expresses as [`W_CType::kind`]: the subclass is data, and the
//! methods it overrides are the `match` arms of the free functions below.

use crate::PyError;
use pyre_object::PyObjectRef;
use std::sync::OnceLock;

use super::cdataobj::W_CData;

// ── the subclass discriminant ───────────────────────────────────────────

/// `ctypevoid.py W_CTypeVoid`.
pub const KIND_VOID: i64 = 0;
/// `ctypeprim.py W_CTypePrimitiveChar`.
pub const KIND_PRIM_CHAR: i64 = 1;
/// `ctypeprim.py W_CTypePrimitiveUniChar`.
pub const KIND_PRIM_UNICHAR: i64 = 2;
/// `ctypeprim.py W_CTypePrimitiveSigned`.
pub const KIND_PRIM_SIGNED: i64 = 3;
/// `ctypeprim.py W_CTypePrimitiveUnsigned`.
pub const KIND_PRIM_UNSIGNED: i64 = 4;
/// `ctypeprim.py W_CTypePrimitiveBool`.
pub const KIND_PRIM_BOOL: i64 = 5;
/// `ctypeprim.py W_CTypePrimitiveFloat`.
pub const KIND_PRIM_FLOAT: i64 = 6;
/// `ctypeprim.py W_CTypePrimitiveLongDouble`.
pub const KIND_PRIM_LONGDOUBLE: i64 = 7;
/// `ctypeprim.py W_CTypePrimitiveComplex`.
pub const KIND_PRIM_COMPLEX: i64 = 8;
/// `ctypeptr.py W_CTypePointer`.
pub const KIND_POINTER: i64 = 9;
/// `ctypearray.py W_CTypeArray`.
pub const KIND_ARRAY: i64 = 10;

// ── the boolean class attributes ────────────────────────────────────────

/// `W_CType.is_primitive_integer`.
pub const F_PRIMITIVE_INTEGER: i64 = 1 << 0;
/// `W_CType.is_nonfunc_pointer_or_array`.
pub const F_NONFUNC_POINTER_OR_ARRAY: i64 = 1 << 1;
/// `W_CTypePtrOrArray.accept_str`.
pub const F_ACCEPT_STR: i64 = 1 << 2;
/// `W_CTypePtrBase.is_void_ptr`.
pub const F_VOID_PTR: i64 = 1 << 3;
/// `W_CTypePtrBase.is_voidchar_ptr`.
pub const F_VOIDCHAR_PTR: i64 = 1 << 4;
/// `W_CTypePtrBase.is_onebyte_ptr`.
pub const F_ONEBYTE_PTR: i64 = 1 << 5;
/// `W_CTypePointer.is_file`.
pub const F_FILE_PTR: i64 = 1 << 6;
/// `W_CTypePrimitiveSigned.value_fits_long` /
/// `W_CTypePrimitiveUnsigned.value_fits_long`.
pub const F_VALUE_FITS_LONG: i64 = 1 << 7;
/// `W_CTypePrimitiveSigned.value_smaller_than_long`.
pub const F_VALUE_SMALLER_THAN_LONG: i64 = 1 << 8;
/// `W_CTypePrimitiveUnsigned.value_fits_ulong`.
pub const F_VALUE_FITS_ULONG: i64 = 1 << 9;
/// `W_CTypePrimitiveUniChar.is_signed_wchar`.
pub const F_SIGNED_WCHAR: i64 = 1 << 10;

/// `ctypeobj.py W_CType` and the RPython subclasses that share its typedef.
///
/// PyPy caches every ctype and reaches the derived ones through weak
/// references (`W_CType._pointer_type`, `W_CTypePointer._array_types`), so a
/// ctype dies once the last user does.  pyre holds them strongly and roots
/// them for the process instead: the memo has to survive because
/// `convert_from_object` decides pointer compatibility by object identity,
/// and a ctype is a small, bounded population that a program declares once.
#[crate::pyre_class("_cffi_backend.CType")]
pub struct W_CType {
    /// `W_CType.size` — the size of an instance, or -1 when unknown.
    pub size: i64,
    /// `W_CType.name`.  Interpreter-level, as upstream has it: `cname`
    /// wraps it on each read.  Every ctype is memoised and rooted for the
    /// process, so its name is leaked on the same terms.
    pub name: &'static str,
    /// `W_CType.name_position` — where `insert_name` splices into `name`.
    pub name_position: i64,
    /// Which RPython subclass of `W_CType` this is; one of the `KIND_*`.
    pub kind: i64,
    /// `W_CTypePrimitive.align`, or -1 for a type of unknown alignment.
    pub align: i64,
    /// `W_CTypePtrOrArray.ctitem` — what a pointer points to, what an array
    /// holds.  `PY_NULL` on every other kind.
    pub ctitem: PyObjectRef,
    /// `W_CType._pointer_type` — the pointer *to* this type, which
    /// `new_pointer_type` memoises here.  `PY_NULL` until one is asked for.
    pub pointer_type: PyObjectRef,
    /// `W_CTypeArray.ctptr` — the pointer this array was built from, whose
    /// `ctitem` is the array's item type.  `PY_NULL` on every other kind.
    /// It is not `pointer_type`: the pointer to an `int[5]` is `int(*)[5]`,
    /// while its `ctptr` is `int *`.
    pub ctptr: PyObjectRef,
    /// `W_CTypeArray.length` — -1 for `int[]` and for anything not an array.
    pub length: i64,
    /// The `F_*` class attributes above.
    pub flags: i64,
}

impl W_CType {
    /// `W_CType.name`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn has(&self, flag: i64) -> bool {
        self.flags & flag != 0
    }

    /// The ctype as the object it is; every ctype is allocated non-moving, so
    /// its address is stable for the process.
    pub fn as_object(&self) -> PyObjectRef {
        std::ptr::from_ref(self)
            .cast::<pyre_object::PyObject>()
            .cast_mut()
    }

    /// `isinstance(ct, W_CTypePrimitive)`.
    pub fn is_primitive(&self) -> bool {
        (KIND_PRIM_CHAR..=KIND_PRIM_COMPLEX).contains(&self.kind)
    }

    /// `isinstance(ct, W_CTypePrimitiveFloat)` — which
    /// `W_CTypePrimitiveLongDouble` also satisfies.
    pub fn is_float_family(&self) -> bool {
        self.kind == KIND_PRIM_FLOAT || self.kind == KIND_PRIM_LONGDOUBLE
    }

    /// `isinstance(ct, W_CTypePrimitiveCharOrUniChar)`.
    pub fn is_char_or_unichar(&self) -> bool {
        self.kind == KIND_PRIM_CHAR || self.kind == KIND_PRIM_UNICHAR
    }

    /// `isinstance(ct, W_CTypePtrOrArray)`.
    pub fn is_ptr_or_array(&self) -> bool {
        self.kind == KIND_POINTER || self.kind == KIND_ARRAY
    }

    /// `W_CTypePtrOrArray.is_unichar_ptr_or_array`.
    pub fn is_unichar_ptr_or_array(&self) -> bool {
        self.is_ptr_or_array()
            && ctype_at(self.ctitem).is_some_and(|it| it.kind == KIND_PRIM_UNICHAR)
    }

    /// `W_CTypePtrOrArray.is_char_or_unichar_ptr_or_array`.
    pub fn is_char_or_unichar_ptr_or_array(&self) -> bool {
        self.is_ptr_or_array() && ctype_at(self.ctitem).is_some_and(|it| it.is_char_or_unichar())
    }

    /// `W_CType._within_bounds`.
    pub fn within_bounds(&self, actual_length: i64) -> bool {
        self.length < 0 || actual_length <= self.length
    }

    /// `W_CType.insert_name` — splice `extra` into `name` at the recorded
    /// position, and report where the spliced name's own position is.
    pub fn insert_name(&self, extra: &str, extra_position: i64) -> (String, i64) {
        let name = self.name();
        let at = self.name_position as usize;
        let mut spliced = String::with_capacity(name.len() + extra.len());
        spliced.push_str(&name[..at]);
        spliced.push_str(extra);
        spliced.push_str(&name[at..]);
        (spliced, self.name_position + extra_position)
    }

    /// `W_CType.alignof` — `_alignof` raises for a type that has none.
    pub fn alignof(&self) -> Result<i64, PyError> {
        match self.kind {
            // `W_CTypePtrBase._alignof` — every pointer aligns as one.
            KIND_POINTER => Ok(align_of::<*const u8>() as i64),
            // `W_CTypeArray._alignof` — `self.ctitem.alignof()`.
            KIND_ARRAY => ctype_at(self.ctitem)
                .ok_or_else(|| self.unknown_alignment())?
                .alignof(),
            _ if self.align >= 0 => Ok(self.align),
            _ => Err(self.unknown_alignment()),
        }
    }

    fn unknown_alignment(&self) -> PyError {
        PyError::value_error(format!("ctype '{}' is of unknown alignment", self.name()))
    }

    /// `W_CType._convert_error` — the initializer-mismatch TypeError, with
    /// the two special cases a same-named cdata gets.
    pub fn convert_error(&self, expected: &str, w_got: PyObjectRef) -> PyError {
        let name = self.name();
        if let Some(got) = W_CData::from_obj(w_got)
            && let Some(got_ctype) = ctype_at(got.ctype)
        {
            if name == got_ctype.name() {
                if std::ptr::eq(self, got_ctype) {
                    return PyError::system_error(format!(
                        "initializer for ctype '{name}' is correct, but we get an internal mismatch--please report a bug"
                    ));
                }
                return PyError::type_error(format!(
                    "initializer for ctype '{name}' appears indeed to be '{}', but the types are different (check that you are not e.g. mixing up different ffi instances)",
                    got_ctype.name()
                ));
            }
            return PyError::type_error(format!(
                "initializer for ctype '{name}' must be a {expected}, not cdata '{}'",
                got_ctype.name()
            ));
        }
        PyError::type_error(format!(
            "initializer for ctype '{name}' must be a {expected}, not {}",
            crate::type_methods::arg_type_name(w_got)
        ))
    }

    /// `W_CType.extra_repr` — what `<cdata '...' HERE>` shows.
    ///
    /// # Safety
    /// `cdata` must be readable for this ctype's size when it is primitive.
    pub unsafe fn extra_repr(&self, cdata: *const u8) -> Result<String, PyError> {
        if self.kind == KIND_PRIM_LONGDOUBLE {
            return Ok(unsafe { super::misc::longdouble2str(cdata) });
        }
        if self.is_primitive() {
            // `W_CTypePrimitive.extra_repr` — `repr(convert_to_object(cdata))`.
            let roots = pyre_object::gc_roots::push_roots();
            let ob_slot = roots.base();
            let _ = roots.pin_root(unsafe { convert_to_object(self, cdata)? });
            let w_repr = crate::builtins::builtin_repr(&[roots.get(ob_slot)])?;
            return Ok(unsafe { pyre_object::w_str_get_value(w_repr) }.to_string());
        }
        Ok(if cdata.is_null() {
            "NULL".to_string()
        } else {
            format!("0x{:x}", cdata as usize)
        })
    }
}

/// Borrow the `W_CType` a `PyObjectRef` names, or `None` when it is not one.
pub fn ctype_at(w_ctype: PyObjectRef) -> Option<&'static mut W_CType> {
    if w_ctype.is_null() {
        return None;
    }
    W_CType::from_obj(w_ctype)
}

/// The ctype a cdata carries.
pub fn ctype_of(cdata: &W_CData) -> Option<&'static mut W_CType> {
    ctype_at(cdata.ctype)
}

/// `@unwrap_spec(w_ctype=ctypeobj.W_CType)`.
pub fn ctype_arg(w_ctype: PyObjectRef) -> Result<&'static mut W_CType, PyError> {
    ctype_at(w_ctype).ok_or_else(|| {
        PyError::type_error(format!(
            "expected a ctype object, got '{}'",
            crate::type_methods::arg_type_name(w_ctype)
        ))
    })
}

// ── construction ────────────────────────────────────────────────────────

/// Build a ctype and root it for the process.  Every ctype is memoised, so
/// none of them is ever garbage: rooting once at birth is what lets the
/// caches hold plain addresses.
pub fn new_ctype(
    kind: i64,
    size: i64,
    name: &str,
    name_position: i64,
    align: i64,
    ctitem: PyObjectRef,
    length: i64,
    flags: i64,
) -> PyObjectRef {
    let roots = pyre_object::gc_roots::push_roots();
    let ctitem_slot = roots.base();
    let _ = roots.pin_root(ctitem);
    let obj = W_CType::allocate_stable(W_CType {
        size,
        name: String::leak(name.to_string()),
        name_position,
        kind,
        align,
        ctitem: roots.get(ctitem_slot),
        pointer_type: pyre_object::PY_NULL,
        ctptr: pyre_object::PY_NULL,
        length,
        flags,
        ..Default::default()
    });
    root_forever(obj);
    obj
}

impl Default for W_CType {
    fn default() -> Self {
        Self {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: pyre_object::PY_NULL,
            },
            size: -1,
            name: "",
            name_position: 0,
            kind: KIND_VOID,
            align: -1,
            ctitem: pyre_object::PY_NULL,
            pointer_type: pyre_object::PY_NULL,
            ctptr: pyre_object::PY_NULL,
            length: -1,
            flags: 0,
        }
    }
}

/// The rooted slots the ctype caches name.  A ctype is born through
/// `allocate_stable`, so it never relocates and the cache can hold its plain
/// address; the slot exists so the collector sees the object as live.
static ROOTED_CTYPES: std::sync::Mutex<Vec<Box<usize>>> = std::sync::Mutex::new(Vec::new());

pub fn root_forever(obj: PyObjectRef) {
    let mut slot = Box::new(obj as usize);
    let root_slot = (&raw mut *slot) as *mut *mut u8;
    unsafe { pyre_object::gc_hook::try_gc_add_root(root_slot) };
    ROOTED_CTYPES
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(slot);
}

// ── the dispatchers the RPython hierarchy spells as overrides ───────────

/// `W_CType.convert_to_object`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn convert_to_object(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    match ct.kind {
        KIND_POINTER => Ok(super::ctypeptr::pointer_convert_to_object(ct, cdata)),
        KIND_ARRAY => Ok(super::ctypeptr::array_convert_to_object(ct, cdata)),
        _ if ct.is_primitive() => unsafe { super::ctypeprim::convert_to_object(ct, cdata) },
        _ => Err(PyError::type_error(format!(
            "cannot return a cdata '{}'",
            ct.name()
        ))),
    }
}

/// `W_CType.copy_and_convert_to_object` — `void` answers `None` rather than
/// reading anything.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes unless the ctype is `void`.
pub unsafe fn copy_and_convert_to_object(
    ct: &W_CType,
    cdata: *const u8,
) -> Result<PyObjectRef, PyError> {
    if ct.kind == KIND_VOID {
        return Ok(pyre_object::w_none());
    }
    unsafe { convert_to_object(ct, cdata) }
}

/// `W_CType.convert_from_object`.
///
/// # Safety
/// `cdata` must be writable for `ct.size` bytes.
pub unsafe fn convert_from_object(
    ct: &W_CType,
    cdata: *mut u8,
    w_ob: PyObjectRef,
) -> Result<(), PyError> {
    match ct.kind {
        KIND_POINTER => unsafe { super::ctypeptr::pointer_convert_from_object(ct, cdata, w_ob) },
        KIND_ARRAY => unsafe { super::ctypeptr::array_convert_from_object(ct, cdata, w_ob) },
        _ if ct.is_primitive() => unsafe { super::ctypeprim::convert_from_object(ct, cdata, w_ob) },
        _ => Err(PyError::type_error(format!(
            "cannot initialize cdata '{}'",
            ct.name()
        ))),
    }
}

/// `W_CType.cast`.
pub fn cast(w_ctype: PyObjectRef, w_ob: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_POINTER | KIND_ARRAY => super::ctypeptr::cast(w_ctype, w_ob),
        _ if ct.is_primitive() => super::ctypeprim::cast(w_ctype, w_ob),
        _ => Err(PyError::type_error(format!(
            "cannot cast to '{}'",
            ct.name()
        ))),
    }
}

/// `W_CType.newp`.
pub fn newp(w_ctype: PyObjectRef, w_init: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_POINTER => super::ctypeptr::pointer_newp(w_ctype, w_init),
        KIND_ARRAY => super::ctypeptr::array_newp(w_ctype, w_init),
        _ => Err(PyError::type_error(format!(
            "expected a pointer or array ctype, got '{}'",
            ct.name()
        ))),
    }
}

/// `W_CType.cast_to_int` — what `int(cdata)` reads.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn cast_to_int(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.is_primitive() {
        return unsafe { super::ctypeprim::cast_to_int(ct, cdata) };
    }
    Err(PyError::type_error(format!(
        "int() not supported on cdata '{}'",
        ct.name()
    )))
}

/// `W_CType.float`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn float(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.is_float_family() {
        return unsafe { super::ctypeprim::float(ct, cdata) };
    }
    Err(PyError::type_error(format!(
        "float() not supported on cdata '{}'",
        ct.name()
    )))
}

/// `W_CType.complex`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn complex(ct: &W_CType, cdata: *const u8) -> Result<PyObjectRef, PyError> {
    if ct.kind == KIND_PRIM_COMPLEX {
        return unsafe { super::ctypeprim::convert_to_object(ct, cdata) };
    }
    Err(PyError::type_error(format!(
        "complex() not supported on cdata '{}'",
        ct.name()
    )))
}

/// `W_CType.nonzero`.
///
/// # Safety
/// `cdata` must be readable for `ct.size` bytes.
pub unsafe fn nonzero(ct: &W_CType, cdata: *const u8) -> Result<bool, PyError> {
    if ct.is_primitive() {
        return unsafe { super::ctypeprim::nonzero(ct, cdata) };
    }
    Ok(!cdata.is_null())
}

/// `W_CType.add` — pointer arithmetic on a cdata.
pub fn add(w_ctype: PyObjectRef, cdata: *mut u8, i: i64) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_POINTER | KIND_ARRAY => super::ctypeptr::add(w_ctype, cdata, i),
        _ => Err(PyError::type_error(format!(
            "cannot add a cdata '{}' and a number",
            ct.name()
        ))),
    }
}

/// `W_CType.string`.
pub fn string(w_cdata: PyObjectRef, maxlen: i64) -> Result<PyObjectRef, PyError> {
    let cdata =
        W_CData::from_obj(w_cdata).ok_or_else(|| PyError::type_error("expected a cdata object"))?;
    let ct = ctype_of(cdata).ok_or_else(|| PyError::type_error("expected a cdata object"))?;
    match ct.kind {
        KIND_POINTER | KIND_ARRAY => super::ctypeptr::string(w_cdata, maxlen),
        _ if ct.is_primitive() => super::ctypeprim::string(w_cdata, maxlen),
        _ => Err(unexpected_string_argument(ct)),
    }
}

pub fn unexpected_string_argument(ct: &W_CType) -> PyError {
    PyError::type_error(format!(
        "string(): unexpected cdata '{}' argument",
        ct.name()
    ))
}

/// `W_CType.get_vararg_type` — the type an argument is promoted to when it
/// is passed through `...`.
pub fn get_vararg_type(w_ctype: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_ctype)?;
    match ct.kind {
        KIND_ARRAY => Ok(ct.ctptr),
        KIND_PRIM_CHAR | KIND_PRIM_UNICHAR => super::newtype::new_primitive_type("int"),
        KIND_PRIM_SIGNED | KIND_PRIM_UNSIGNED
            if ct.size < std::mem::size_of::<std::ffi::c_int>() as i64 =>
        {
            super::newtype::new_primitive_type("int")
        }
        _ => Ok(w_ctype),
    }
}

// ── the Python type ─────────────────────────────────────────────────────

static CTYPE_TYPE_OBJ: OnceLock<usize> = OnceLock::new();

/// `_cffi_backend.CType`.
pub fn ctype_type() -> PyObjectRef {
    *CTYPE_TYPE_OBJ.get_or_init(|| {
        let tp = crate::typedef::make_builtin_type_with_layout(
            "_cffi_backend.CType",
            init_ctype_type,
            crate::typedef::w_object(),
            <W_CType as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
        );
        pyre_object::pyobject::set_instantiate(
            unsafe { &*<W_CType as pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE },
            tp,
        );
        tp as usize
    }) as PyObjectRef
}

/// The names `W_CType.dir` reports — `typedef.rawdict` minus the dunders,
/// sorted, filtered by which of them the ctype actually answers.
const ATTRIBUTE_NAMES: [&str; 11] = [
    "abi",
    "args",
    "cname",
    "elements",
    "ellipsis",
    "fields",
    "item",
    "kind",
    "length",
    "relements",
    "result",
];

fn init_ctype_type(ns: PyObjectRef) {
    let store = |name: &str, value: PyObjectRef| unsafe {
        pyre_object::dictmultiobject::w_dict_setitem_str_no_proxy(ns, name, value)
    };
    store(
        "__repr__",
        crate::make_builtin_function_with_arity("__repr__", ctype_repr, 1),
    );
    store(
        "__dir__",
        crate::make_builtin_function_with_arity("__dir__", ctype_dir, 1),
    );
    store("__weakref__", crate::typedef::make_weakref_descr(ns));
    for (name, doc, attrchar) in [
        ("kind", "kind", 'k'),
        ("cname", "C name", 'c'),
        ("item", "pointer to, or array of", 'i'),
        ("length", "array length or None", 'l'),
        ("fields", "struct or union fields", 'f'),
        ("args", "function argument types", 'a'),
        ("result", "function result type", 'r'),
        ("ellipsis", "function has '...'", 'E'),
        ("abi", "function ABI", 'A'),
        ("elements", "enum elements", 'e'),
        ("relements", "enum elements, reversed", 'R'),
    ] {
        let getter = make_fget(name, attrchar);
        store(
            name,
            crate::typedef::make_getset_property_named_doc(
                getter,
                pyre_object::PY_NULL,
                pyre_object::PY_NULL,
                doc,
                name,
            ),
        );
    }
}

/// One `fget_*` per attribute character.  `W_CType._fget` is a single method
/// switching on that character, so the wrappers only differ in which one they
/// pass; a builtin function carries no closure, hence the explicit table.
fn make_fget(name: &'static str, attrchar: char) -> PyObjectRef {
    macro_rules! fget {
        ($c:literal) => {
            crate::make_builtin_function_with_arity(
                name,
                |args| {
                    // `typedef.py:361 self.fget(self, space, w_obj)` — the
                    // descriptor comes first and the instance second.
                    let w_self = args
                        .get(1)
                        .copied()
                        .ok_or_else(|| PyError::type_error("descriptor requires an instance"))?;
                    fget(w_self, $c)
                },
                2,
            )
        };
    }
    match attrchar {
        'k' => fget!('k'),
        'c' => fget!('c'),
        'i' => fget!('i'),
        'l' => fget!('l'),
        'f' => fget!('f'),
        'a' => fget!('a'),
        'r' => fget!('r'),
        'E' => fget!('E'),
        'A' => fget!('A'),
        'e' => fget!('e'),
        _ => fget!('R'),
    }
}

/// `W_CType._fget` and the subclass overrides of it.
fn fget(w_self: PyObjectRef, attrchar: char) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(w_self)?;
    match attrchar {
        // `W_CType._fget('k')` — a class attribute in PyPy.
        'k' => Ok(pyre_object::w_str_new(kind_name(ct.kind))),
        'c' => Ok(pyre_object::w_str_new(ct.name)),
        // `W_CTypePtrOrArray._fget('i')`.
        'i' if ct.is_ptr_or_array() => Ok(ct.ctitem),
        // `W_CTypeArray._fget('l')`.
        'l' if ct.kind == KIND_ARRAY => Ok(if ct.length >= 0 {
            pyre_object::w_int_new(ct.length)
        } else {
            pyre_object::w_none()
        }),
        _ => Err(PyError::attribute_error(format!(
            "ctype '{}' has no such attribute",
            ct.name()
        ))),
    }
}

/// `W_CType.kind`, the class attribute each subclass overrides.
pub fn kind_name(kind: i64) -> &'static str {
    match kind {
        KIND_VOID => "void",
        KIND_POINTER => "pointer",
        KIND_ARRAY => "array",
        _ => "primitive",
    }
}

/// `W_CType.repr`.
fn ctype_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let ct = ctype_arg(args[0])?;
    Ok(pyre_object::w_str_new(&format!("<ctype '{}'>", ct.name())))
}

/// `W_CType.dir` — every attribute above that this ctype answers.
fn ctype_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_self = args[0];
    let roots = pyre_object::gc_roots::push_roots();
    let base = roots.base();
    let mut count = 0;
    for name in ATTRIBUTE_NAMES {
        if crate::baseobjspace::getattr_str(w_self, name).is_ok() {
            let _ = roots.pin_root(pyre_object::w_str_new(name));
            count += 1;
        }
    }
    let items = (0..count).map(|i| roots.get(base + i)).collect();
    Ok(pyre_object::w_list_new(items))
}
