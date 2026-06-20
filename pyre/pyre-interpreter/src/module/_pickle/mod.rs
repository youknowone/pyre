//! `_pickle` — interp-level accelerator for the `pickle` module.
//!
//! Port of `pypy/module/_pickle/interp_pickle.py` (`W_Pickler` /
//! `W_Unpickler`). Targets the CPython 3.14 wire format.
//!
//! Scope: protocols 0-5 — atoms (None / bool / int / float / str / bytes),
//! the memo (PUT/GET families), containers (tuple / list / dict with
//! APPENDS / SETITEMS batching + recursion), set / frozenset / bytearray,
//! the reduce protocol (save_reduce / save_global / find_class), the legacy
//! protocol-0/1 text opcodes, persistent_id, `_compat_pickle` fix_imports,
//! and the multi-frame framer. Out-of-band proto-5 buffers are deferred
//! (see the note at the pickler dispatch site).
//!
//! The module exports all nine names `pickle.py` imports — `Pickler`,
//! `Unpickler`, `dump`, `dumps`, `load`, `loads`, `PickleError`,
//! `PicklingError`, `UnpicklingError` — so `from _pickle import (...)`
//! resolves and the accelerated path engages.

use malachite_bigint::{BigInt, Sign};
use pyre_object::PyObjectRef;

use crate::PyError;

mod pickler;
mod unpickler;

pub use pickler::{W_Pickler, W_PicklerMemoProxy};
pub use unpickler::{W_Unpickler, W_UnpicklerMemoProxy};

pub(crate) const HIGHEST_PROTOCOL: i64 = 5;
pub(crate) const DEFAULT_PROTOCOL: i64 = 5;
pub(crate) const FRAME_SIZE_MIN: usize = 4;
pub(crate) const FRAME_SIZE_TARGET: usize = 64 * 1024;

/// `interp_pickle.py Opcodes`.
pub(crate) mod op {
    pub const PROTO: u8 = 0x80;
    pub const FRAME: u8 = 0x95;
    pub const STOP: u8 = b'.';
    pub const NONE: u8 = b'N';
    pub const NEWTRUE: u8 = 0x88;
    pub const NEWFALSE: u8 = 0x89;
    pub const BININT: u8 = b'J';
    pub const BININT1: u8 = b'K';
    pub const BININT2: u8 = b'M';
    pub const LONG1: u8 = 0x8a;
    pub const LONG4: u8 = 0x8b;
    pub const BINFLOAT: u8 = b'G';
    pub const SHORT_BINUNICODE: u8 = 0x8c;
    pub const BINUNICODE: u8 = b'X';
    pub const BINUNICODE8: u8 = 0x8d;
    pub const SHORT_BINBYTES: u8 = b'C';
    pub const BINBYTES: u8 = b'B';
    pub const BINBYTES8: u8 = 0x8e;
    // memo
    pub const MEMOIZE: u8 = 0x94;
    pub const BINPUT: u8 = b'q';
    pub const LONG_BINPUT: u8 = b'r';
    pub const PUT: u8 = b'p';
    pub const GET: u8 = b'g';
    pub const BINGET: u8 = b'h';
    pub const LONG_BINGET: u8 = b'j';
    // stack
    pub const MARK: u8 = b'(';
    pub const POP: u8 = b'0';
    pub const POP_MARK: u8 = b'1';
    // tuple
    pub const EMPTY_TUPLE: u8 = b')';
    pub const TUPLE: u8 = b't';
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const TUPLE3: u8 = 0x87;
    // list
    pub const EMPTY_LIST: u8 = b']';
    pub const LIST: u8 = b'l';
    pub const APPEND: u8 = b'a';
    pub const APPENDS: u8 = b'e';
    // dict
    pub const EMPTY_DICT: u8 = b'}';
    pub const DICT: u8 = b'd';
    pub const SETITEM: u8 = b's';
    pub const SETITEMS: u8 = b'u';
    // set / frozenset
    pub const EMPTY_SET: u8 = 0x8f;
    pub const FROZENSET: u8 = 0x91;
    pub const ADDITEMS: u8 = 0x90;
    // bytearray
    pub const BYTEARRAY8: u8 = 0x96;
    // protocol 5 out-of-band buffers (see the deferral note in `pickler.rs`).
    pub const NEXT_BUFFER: u8 = 0x97;
    pub const READONLY_BUFFER: u8 = 0x98;
    // reduce / global
    pub const REDUCE: u8 = b'R';
    pub const BUILD: u8 = b'b';
    pub const GLOBAL: u8 = b'c';
    pub const STACK_GLOBAL: u8 = 0x93;
    pub const NEWOBJ: u8 = 0x81;
    pub const NEWOBJ_EX: u8 = 0x92;
    pub const EXT1: u8 = 0x82;
    pub const EXT2: u8 = 0x83;
    pub const EXT4: u8 = 0x84;
    // protocol 0 / 1 legacy text opcodes
    pub const INT: u8 = b'I';
    pub const LONG: u8 = b'L';
    pub const FLOAT: u8 = b'F';
    pub const STRING: u8 = b'S';
    pub const UNICODE: u8 = b'V';
    pub const BINSTRING: u8 = b'T';
    pub const SHORT_BINSTRING: u8 = b'U';
    pub const INST: u8 = b'i';
    pub const OBJ: u8 = b'o';
    pub const DUP: u8 = b'2';
    // persistent id
    pub const PERSID: u8 = b'P';
    pub const BINPERSID: u8 = b'Q';

    /// `_tuplesize2code` — TUPLE1/2/3 indexed by element count (1..=3).
    pub const TUPLESIZE2CODE: [u8; 4] = [EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3];
}

/// `interp_pickle.py W_Pickler._BATCHSIZE`.
pub(crate) const BATCHSIZE: usize = 1000;

// ── shared call helpers ──────────────────────────────────────────────
// `call_function` / `call_method` return PY_NULL on failure and stash the
// error through `call::set_call_error`; surface it as a Rust `Result`.

pub(crate) fn call_fn(callable: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let r = crate::baseobjspace::call_function(callable, args);
    if r.is_null() {
        return Err(
            crate::call::take_call_error().unwrap_or_else(|| PyError::runtime_error("call failed"))
        );
    }
    Ok(r)
}

pub(crate) fn call_meth(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let r = crate::baseobjspace::call_method(obj, name, args);
    if r.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("method call failed")));
    }
    Ok(r)
}

/// Build a `PyError` whose raised object is an instance of the named
/// `_pickle` exception class (registered by the `exceptions:` block), with
/// `msg` as the single argument. Falls back to a generic ValueError carrying
/// the same text if the class is somehow unavailable.
fn pickle_exc(class_name: &str, msg: String) -> PyError {
    let mut err = PyError::value_error(msg.clone());
    if let Some(cls) = crate::builtins::lookup_exc_class(class_name) {
        let args = [cls, pyre_object::w_str_new(&msg)];
        if let Ok(exc) = crate::builtins::exc_exception_new(&args) {
            err.exc_object = exc;
        }
    }
    err
}

pub(crate) fn unpickling_error(msg: &str) -> PyError {
    pickle_exc("_pickle.UnpicklingError", msg.to_string())
}

pub(crate) fn pickling_error(msg: impl Into<String>) -> PyError {
    pickle_exc("_pickle.PicklingError", msg.into())
}

// ── import / dotted attribute resolution (save_global / find_class) ───

/// Return the named module from `sys.modules`, importing it only if absent.
/// An already-loaded module is returned directly: re-running `importhook`
/// for a loaded module (notably `builtins`) can rebind the canonical module
/// object and corrupt name resolution elsewhere. The `sys.modules` entry
/// (not the `importhook` return) is authoritative.
pub(crate) fn import_module(name: &str) -> Result<PyObjectRef, PyError> {
    if let Some(m) = crate::importing::get_sys_module(name) {
        return Ok(m);
    }
    // The `builtins` module lives on the execution context, not in the
    // importable `sys.modules` cache; re-running `importhook` on it would
    // reinitialise builtin state (and orphan the live exception classes).
    if name == "builtins" {
        if let Some(b) = ec_builtins_module() {
            return Ok(b);
        }
    }
    crate::importing::importhook(
        name,
        pyre_object::w_none(),
        pyre_object::listobject::w_list_new(vec![pyre_object::w_str_new("*")]),
        0,
        crate::call::getexecutioncontext(),
    )?;
    crate::importing::get_sys_module(name)
        .ok_or_else(|| PyError::value_error(format!("Can't find module {name:?} in sys.modules")))
}

/// The live execution context reached via the current frame, or `None`
/// when no frame is on the stack.
fn current_ec() -> Option<*const crate::PyExecutionContext> {
    let frame = crate::eval::CURRENT_FRAME.with(|f| f.get());
    if frame.is_null() {
        return None;
    }
    let ec = unsafe { (*frame).execution_context };
    if ec.is_null() { None } else { Some(ec) }
}

/// The current execution context's `builtins` module, via the live frame.
fn ec_builtins_module() -> Option<PyObjectRef> {
    current_ec().map(|ec| unsafe { (*ec).get_builtin() })
}

/// Resolve a name in the `builtins` module through the execution context's
/// `lookup_builtin` (the `LOAD_GLOBAL` fallback path). This bypasses the
/// module-object `getattr` wrapper, whose hash table does not see builtins
/// installed on the underlying storage.
pub(crate) fn lookup_builtin(name: &str) -> Option<PyObjectRef> {
    current_ec().and_then(|ec| unsafe { (*ec).lookup_builtin(name) })
}

/// `_compat_pickle` import/name compatibility mapping applied at protocol
/// < 3. `reverse` picks the py3 → py2 direction used when dumping (the
/// REVERSE_* tables); the forward direction (py2 → py3) is used when
/// loading. Returns the mapped `(module, name)`, unchanged when no entry
/// matches.
pub(crate) fn compat_map(module: &str, name: &str, reverse: bool) -> (String, String) {
    let (name_map_attr, import_map_attr) = if reverse {
        ("REVERSE_NAME_MAPPING", "REVERSE_IMPORT_MAPPING")
    } else {
        ("NAME_MAPPING", "IMPORT_MAPPING")
    };
    let compat = match import_module("_compat_pickle") {
        Ok(m) => m,
        Err(_) => return (module.to_string(), name.to_string()),
    };
    // (module, name) entry takes precedence over a bare module remap.
    if let Ok(w_name_map) = crate::baseobjspace::getattr_str(compat, name_map_attr) {
        let key = pyre_object::tupleobject::w_tuple_new(vec![
            pyre_object::w_str_new(module),
            pyre_object::w_str_new(name),
        ]);
        if let Some(v) = unsafe { pyre_object::w_dict_lookup(w_name_map, key) } {
            let m = unsafe { pyre_object::tupleobject::w_tuple_getitem(v, 0) };
            let n = unsafe { pyre_object::tupleobject::w_tuple_getitem(v, 1) };
            if let (Some(m), Some(n)) = (m, n) {
                return (
                    unsafe { pyre_object::strobject::w_str_get_value(m) }.to_string(),
                    unsafe { pyre_object::strobject::w_str_get_value(n) }.to_string(),
                );
            }
        }
    }
    if let Ok(w_import_map) = crate::baseobjspace::getattr_str(compat, import_map_attr) {
        if let Some(v) =
            unsafe { pyre_object::w_dict_lookup(w_import_map, pyre_object::w_str_new(module)) }
        {
            return (
                unsafe { pyre_object::strobject::w_str_get_value(v) }.to_string(),
                name.to_string(),
            );
        }
    }
    (module.to_string(), name.to_string())
}

/// `interp_pickle.py _getattribute` — walk a dotted `qualname` from `obj`,
/// returning `(resolved, parent)`.
pub(crate) fn getattribute_dotted(
    obj: PyObjectRef,
    qualname: &str,
) -> Result<(PyObjectRef, PyObjectRef), PyError> {
    let mut cur = obj;
    let mut parent = obj;
    for sub in qualname.split('.') {
        if sub == "<locals>" {
            return Err(PyError::attribute_error(format!(
                "Can't get local attribute {qualname:?}"
            )));
        }
        parent = cur;
        cur = crate::baseobjspace::getattr_str(cur, sub)?;
    }
    Ok((cur, parent))
}

/// Resolve `module_name.name` to the live object, importing the module
/// first. `builtins` names resolve through the execution context (whose
/// `getattr` wrapper does not see builtins on the underlying storage); other
/// names resolve through `getattr`, component-by-component for a dotted name,
/// so a module-level dynamic attribute (`__getattr__`, PEP 562) resolves as it
/// does for the live object. Returns `Ok(None)` when the name is simply absent
/// (AttributeError) so the dump-time verification can report a precise
/// "not found" error; the load-side `find_class` maps `None` to an error. No
/// `_compat_pickle` remap — callers apply that first.
///
/// `allow_qualname` mirrors `getattribute(obj, name, allow_qualname)`: when set
/// (proto >= 4 STACK_GLOBAL qualnames and the dump-side `whichmodule`
/// verification) the name is walked as a qualname via `getattr` on each
/// component, like `_getattribute`; when clear (the unpickler at protocol < 4)
/// the name is a single `getattr`, so a dotted name resolves as one literal
/// attribute and fails, as in CPython.
pub(crate) fn try_resolve_global(
    module_name: &str,
    name: &str,
    allow_qualname: bool,
) -> Result<Option<PyObjectRef>, PyError> {
    if module_name == "builtins" && !name.contains('.') {
        if let Some(obj) = lookup_builtin(name) {
            return Ok(Some(obj));
        }
    }
    let module = import_module(module_name)?;
    if allow_qualname {
        // protocol >= 4: walk the qualname via `getattr` on each component (a
        // non-dotted name is a single `getattr`), mirroring `_getattribute`.
        return match getattribute_dotted(module, name) {
            Ok((obj, _)) => Ok(Some(obj)),
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(None),
            Err(e) => Err(e),
        };
    }
    // protocol < 4: a single `getattr`, never a qualname walk.
    match crate::baseobjspace::getattr_str(module, name) {
        Ok(obj) => Ok(Some(obj)),
        Err(e) if e.kind == crate::PyErrorKind::AttributeError => Ok(None),
        Err(e) => Err(e),
    }
}

// ── encode_long / decode_long — two's-complement little-endian ───────
// `interp_pickle.py encode_long` / CPython `pickle.encode_long`.

pub(crate) fn encode_long(big: &BigInt) -> Vec<u8> {
    let sign = big.sign();
    if sign == Sign::NoSign {
        return Vec::new(); // 0 -> b''
    }
    // magnitude, little-endian
    let (_s, digits) = big.to_u32_digits();
    let mut mag: Vec<u8> = Vec::with_capacity(digits.len() * 4);
    for d in &digits {
        mag.extend_from_slice(&d.to_le_bytes());
    }
    while mag.len() > 1 && *mag.last().unwrap() == 0 {
        mag.pop();
    }
    // reserve a byte for the sign bit when the top magnitude bit is set
    if mag.last().map_or(true, |&b| b & 0x80 != 0) {
        mag.push(0x00);
    }
    if sign == Sign::Plus {
        return mag;
    }
    // negative: two's complement (invert + 1)
    let mut carry: u16 = 1;
    for b in mag.iter_mut() {
        let v = (!*b as u16) + carry;
        *b = (v & 0xff) as u8;
        carry = v >> 8;
    }
    // trim a redundant 0xff sign byte (encode_long minimal form)
    while mag.len() > 1 && mag[mag.len() - 1] == 0xff && (mag[mag.len() - 2] & 0x80) != 0 {
        mag.pop();
    }
    mag
}

pub(crate) fn decode_long(data: &[u8]) -> PyObjectRef {
    if data.is_empty() {
        return pyre_object::w_int_new(0);
    }
    let negative = data[data.len() - 1] & 0x80 != 0;
    let unsigned = BigInt::from_bytes_le(Sign::Plus, data);
    let value = if negative {
        // subtract 2**(8*len): little-endian bytes are `len` zeros then 0x01
        let mut pow = vec![0u8; data.len()];
        pow.push(1);
        unsigned - BigInt::from_bytes_le(Sign::Plus, &pow)
    } else {
        unsigned
    };
    int_from_bigint(value)
}

/// Demote to a small int when it fits, mirroring the int/long unification.
pub(crate) fn int_from_bigint(value: BigInt) -> PyObjectRef {
    match i64::try_from(&value) {
        Ok(v) => pyre_object::w_int_new(v),
        Err(_) => pyre_object::w_long_new(value),
    }
}

/// Parse a decimal integer literal (INT / LONG text opcodes) into an int.
pub(crate) fn parse_int_text(s: &str) -> Result<PyObjectRef, PyError> {
    match BigInt::parse_bytes(s.trim().as_bytes(), 10) {
        Some(big) => Ok(int_from_bigint(big)),
        None => Err(unpickling_error("could not convert string to int")),
    }
}

pub(crate) fn read_int_le(data: &[u8]) -> i64 {
    let mut v: i64 = 0;
    for (i, &b) in data.iter().enumerate() {
        v |= (b as i64) << (8 * i);
    }
    v
}

pub(crate) fn str_from_utf8(data: &[u8]) -> Result<PyObjectRef, PyError> {
    let s = std::str::from_utf8(data).map_err(|_| unpickling_error("invalid utf-8 in pickle"))?;
    Ok(pyre_object::w_str_new(s))
}

crate::py_module! {
    "_pickle",
    interpleveldefs: {
        "Pickler" => pickler::type_object(),
        "Unpickler" => unpickler::type_object(),
        // Shared singleton with `__pypy__.PickleBuffer`; `pickle.py` does
        // `from _pickle import PickleBuffer` to set `_HAVE_PICKLE_BUFFER`.
        "PickleBuffer" => crate::module::__pypy__::pickle_buffer::type_object(),
    },
    exceptions: {
        "PickleError" => crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _pickle init"),
        "PicklingError" => crate::builtins::lookup_exc_class("_pickle.PickleError")
            .expect("_pickle.PickleError registered just above"),
        "UnpicklingError" => crate::builtins::lookup_exc_class("_pickle.PickleError")
            .expect("_pickle.PickleError registered just above"),
    },
    inline_functions: {
        // `pickle.dump` — write a pickle of `obj` to `file`.
        fn dump(
            obj: PyObjectRef,
            file: PyObjectRef,
            #[default(pyre_object::w_none())] protocol: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
            #[default(pyre_object::w_none())] buffer_callback: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // Pin every input before any Python-visible work: `normalize_protocol`
            // (`__index__`) and `is_true` (`__bool__`) run user code and can
            // relocate objects under the moving GC.
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(obj);
            let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(buffer_callback);
            let bc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(file);
            let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(protocol);
            let proto_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(fix_imports);
            let fix_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let proto = pickler::normalize_protocol(pyre_object::gc_roots::shadow_stack_get(
                proto_slot,
            ))?;
            pickler::check_buffer_callback(pyre_object::gc_roots::shadow_stack_get(bc_slot), proto)?;
            let fix =
                crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(fix_slot))?;
            // The dump-time `dispatch_table` (no per-pickler one here) — its
            // `copyreg` import can collect; pin it before the memo allocation.
            let dispatch_table = pickler::copyreg_dispatch_table();
            pyre_object::gc_roots::pin_root(dispatch_table);
            let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_memo = pyre_object::listobject::w_list_new(Vec::new());
            // Pass `file` so `pickle_core` streams the frames to it directly.
            pickler::pickle_core(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                pyre_object::gc_roots::shadow_stack_get(file_slot),
                proto,
                proto >= 1,
                proto >= 4,
                fix,
                pyre_object::PY_NULL,
                pyre_object::gc_roots::shadow_stack_get(bc_slot),
                w_memo,
                false,
                pyre_object::gc_roots::shadow_stack_get(dt_slot),
                pyre_object::PY_NULL,
            )?;
            Ok(pyre_object::w_none())
        }

        // `pickle.dumps` — return a pickle of `obj` as `bytes`.
        fn dumps(
            obj: PyObjectRef,
            #[default(pyre_object::w_none())] protocol: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
            #[default(pyre_object::w_none())] buffer_callback: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // Pin every input before any Python-visible work (`__index__` /
            // `__bool__` / `copyreg` import / memo alloc can relocate objects).
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(obj);
            let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(buffer_callback);
            let bc_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(protocol);
            let proto_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(fix_imports);
            let fix_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let proto = pickler::normalize_protocol(pyre_object::gc_roots::shadow_stack_get(
                proto_slot,
            ))?;
            pickler::check_buffer_callback(pyre_object::gc_roots::shadow_stack_get(bc_slot), proto)?;
            let fix =
                crate::baseobjspace::is_true(pyre_object::gc_roots::shadow_stack_get(fix_slot))?;
            let dispatch_table = pickler::copyreg_dispatch_table();
            pyre_object::gc_roots::pin_root(dispatch_table);
            let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_memo = pyre_object::listobject::w_list_new(Vec::new());
            // No file: `pickle_core` accumulates and returns the pickle bytes.
            pickler::pickle_core(
                pyre_object::gc_roots::shadow_stack_get(obj_slot),
                pyre_object::PY_NULL,
                proto,
                proto >= 1,
                proto >= 4,
                fix,
                pyre_object::PY_NULL,
                pyre_object::gc_roots::shadow_stack_get(bc_slot),
                w_memo,
                false,
                pyre_object::gc_roots::shadow_stack_get(dt_slot),
                pyre_object::PY_NULL,
            )
        }

        // `pickle.load` — read a pickle from `file`.
        fn load(
            file: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
            #[default(pyre_object::w_none())] encoding: PyObjectRef,
            #[default(pyre_object::w_none())] errors: PyObjectRef,
            #[default(pyre_object::w_none())] buffers: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            let unpickler = call_fn(
                unpickler::type_object(),
                &[file, fix_imports, encoding, errors, buffers],
            )?;
            call_meth(unpickler, "load", &[])
        }

        // `pickle.loads` — read a pickle from a `bytes` object.
        fn loads(
            data: PyObjectRef,
            #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
            #[default(pyre_object::w_none())] encoding: PyObjectRef,
            #[default(pyre_object::w_none())] errors: PyObjectRef,
            #[default(pyre_object::w_none())] buffers: PyObjectRef,
        ) -> Result<PyObjectRef, PyError> {
            // Pin every argument that outlives the `BytesIO` construction;
            // a minor collection there can relocate them.
            let _roots = pyre_object::gc_roots::push_roots();
            let base = pyre_object::gc_roots::shadow_stack_len();
            pyre_object::gc_roots::pin_root(data);
            pyre_object::gc_roots::pin_root(fix_imports);
            pyre_object::gc_roots::pin_root(encoding);
            pyre_object::gc_roots::pin_root(errors);
            pyre_object::gc_roots::pin_root(buffers);
            let io = import_module("io")?;
            let bytesio_cls = crate::baseobjspace::getattr_str(io, "BytesIO")?;
            let file = call_fn(
                bytesio_cls,
                &[pyre_object::gc_roots::shadow_stack_get(base)],
            )?;
            let unpickler = call_fn(
                unpickler::type_object(),
                &[
                    file,
                    pyre_object::gc_roots::shadow_stack_get(base + 1),
                    pyre_object::gc_roots::shadow_stack_get(base + 2),
                    pyre_object::gc_roots::shadow_stack_get(base + 3),
                    pyre_object::gc_roots::shadow_stack_get(base + 4),
                ],
            )?;
            call_meth(unpickler, "load", &[])
        }
    },
}
