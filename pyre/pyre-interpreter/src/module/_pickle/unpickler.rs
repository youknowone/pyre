//! `_pickle.Unpickler` — `interp_pickle.py W_Unpickler` (atom + container subset).

use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    HIGHEST_PROTOCOL, call_fn, call_meth, decode_long, import_module, op, parse_int_text,
    read_int_le, str_from_utf8, unpickling_error,
};

#[crate::pyre_class("_pickle.Unpickler")]
pub struct W_Unpickler {
    w_file_read: PyObjectRef,
    w_file_readline: PyObjectRef,
    /// Result stack — a Python `list` (GC-managed across `read` allocs).
    w_stack: PyObjectRef,
    /// Saved stacks for the MARK machinery — a Python `list` of lists.
    w_metastack: PyObjectRef,
    /// Memo — a position-indexed Python `list` (`interp_pickle.py:2016`). Unset
    /// slots hold `PY_NULL`; the Object strategy keeps stored values by pointer
    /// identity (a GET returns the exact memoized object).
    w_memo: PyObjectRef,
    /// Next free memo slot (`_memo_append` target); invariant `== len(w_memo)`.
    memo_index: i64,
    /// Active frame bytes (`bytes`) or None.
    w_frame: PyObjectRef,
    frame_index: i64,
    proto: i64,
    /// Apply the `_compat_pickle` py2→py3 name remap at protocol < 3.
    fix_imports: bool,
    /// Encoding for the legacy STRING / BINSTRING / SHORT_BINSTRING decode
    /// (`"ASCII"` by default; `"bytes"` returns the raw bytes object).
    encoding: String,
    /// Decode error handler for the above (`"strict"` by default).
    errors: String,
    /// Out-of-band `buffers` iterator (proto 5), or None.
    w_buffers: PyObjectRef,
    /// `persistent_load` callable set on the instance, or `PY_NULL` when unset
    /// (a subclass may instead override the `persistent_load` method).
    w_persistent_load: PyObjectRef,
}

#[crate::pyre_methods(doc = "Unpickler(file) -> unpickler reading from file.")]
impl W_Unpickler {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_Unpickler::allocate(W_Unpickler {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_file_read: pyre_object::w_none(),
            w_file_readline: pyre_object::w_none(),
            w_stack: pyre_object::w_none(),
            w_metastack: pyre_object::w_none(),
            w_memo: pyre_object::w_none(),
            memo_index: 0,
            w_frame: pyre_object::w_none(),
            frame_index: 0,
            proto: 0,
            fix_imports: true,
            encoding: String::from("ASCII"),
            errors: String::from("strict"),
            w_buffers: pyre_object::w_none(),
            w_persistent_load: pyre_object::PY_NULL,
        })
    }

    fn __init__(
        &mut self,
        file: PyObjectRef,
        #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
        #[default(pyre_object::w_none())] encoding: PyObjectRef,
        #[default(pyre_object::w_none())] errors: PyObjectRef,
        #[default(pyre_object::w_none())] buffers: PyObjectRef,
    ) -> Result<(), PyError> {
        // `encoding` / `errors` govern the legacy STRING / BINSTRING /
        // SHORT_BINSTRING decode (`_decode_string`); `None` falls back to the
        // `"ASCII"` / `"strict"` defaults. `fix_imports` gates the proto-< 3
        // py2→py3 name remap.
        // `interp_pickle.py:2858` declares `encoding="text"`, `errors="text"`:
        // a non-string (other than the `None` default) is a TypeError.
        self.encoding = if unsafe { pyre_object::is_none(encoding) } {
            String::from("ASCII")
        } else if unsafe { pyre_object::is_str(encoding) } {
            unsafe { pyre_object::strobject::w_str_get_value(encoding) }.to_string()
        } else {
            return Err(PyError::type_error(format!(
                "Unpickler() argument 'encoding' must be str, not {}",
                crate::baseobjspace::object_functionstr_type_name(encoding)
            )));
        };
        self.errors = if unsafe { pyre_object::is_none(errors) } {
            String::from("strict")
        } else if unsafe { pyre_object::is_str(errors) } {
            unsafe { pyre_object::strobject::w_str_get_value(errors) }.to_string()
        } else {
            return Err(PyError::type_error(format!(
                "Unpickler() argument 'errors' must be str, not {}",
                crate::baseobjspace::object_functionstr_type_name(errors)
            )));
        };
        self.fix_imports = crate::baseobjspace::is_true(fix_imports)?;
        // `file` must expose both `read` and `readline`; a missing either is a
        // TypeError, not the bare AttributeError a direct `getattr` surfaces
        // (`_Unpickler_SetInputStream`). Store `read` before resolving
        // `readline` so the first bound method is rooted across that allocation.
        self.w_file_read =
            crate::baseobjspace::findattr_result(file, "read")?.unwrap_or(pyre_object::PY_NULL);
        self.w_file_readline =
            crate::baseobjspace::findattr_result(file, "readline")?.unwrap_or(pyre_object::PY_NULL);
        if self.w_file_read.is_null() || self.w_file_readline.is_null() {
            return Err(PyError::type_error(
                "file must have 'read' and 'readline' attributes",
            ));
        }
        self.w_stack = pyre_object::w_none();
        self.w_metastack = pyre_object::w_none();
        // The memo persists across `load` calls (a multi-object stream may
        // back-reference an object memoized by an earlier load).
        self.w_memo = pyre_object::listobject::w_list_new(Vec::new());
        self.memo_index = 0;
        self.w_frame = pyre_object::w_none();
        self.frame_index = 0;
        self.proto = 0;
        // A non-None `buffers` is consumed as an iterator by NEXT_BUFFER.
        self.w_buffers = if unsafe { pyre_object::is_none(buffers) } {
            pyre_object::w_none()
        } else {
            crate::baseobjspace::iter(buffers)?
        };
        self.w_persistent_load = pyre_object::PY_NULL;
        Ok(())
    }

    fn load(&mut self) -> Result<PyObjectRef, PyError> {
        // Fresh stack each load; the memo persists across `load` calls so a
        // later object can back-reference one memoized by an earlier load
        // (lazily created when the unpickler was built only via `__new__`).
        self.w_stack = pyre_object::listobject::w_list_new(Vec::new());
        self.w_metastack = pyre_object::listobject::w_list_new(Vec::new());
        if unsafe { pyre_object::is_none(self.w_memo) } {
            self.w_memo = pyre_object::listobject::w_list_new(Vec::new());
            self.memo_index = 0;
        }
        self.w_frame = pyre_object::w_none();
        self.frame_index = 0;
        self.proto = 0;

        let self_ptr = self as *mut W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self_ptr);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        loop {
            let opcode = read1(slot)?;
            if opcode == op::STOP {
                let me = cur(slot);
                return unsafe { pyre_object::listobject::w_list_pop_end(me.w_stack) }
                    .ok_or_else(|| unpickling_error("STOP with empty stack"));
            }
            dispatch(slot, opcode)?;
        }
    }

    /// `Unpickler.find_class(module, name)` — import `module` and resolve
    /// `name` against it. A subclass may override this to control which
    /// globals the unpickler is allowed to import (the standard security
    /// hook). Emits the `pickle.find_class` audit event.
    fn find_class(
        &self,
        w_module: PyObjectRef,
        w_name: PyObjectRef,
    ) -> Result<PyObjectRef, PyError> {
        // `find_class` is public; reject non-str args before the unchecked
        // `w_str_get_value` reinterpret cast (a non-str would be UB).
        if !unsafe { pyre_object::is_str(w_module) } {
            return Err(PyError::type_error("module name must be a string"));
        }
        if !unsafe { pyre_object::is_str(w_name) } {
            return Err(PyError::type_error(format!(
                "attribute name must be string, not '{}'",
                crate::baseobjspace::object_functionstr_type_name(w_name)
            )));
        }
        let module = unsafe { pyre_object::strobject::w_str_get_value(w_module) }.to_string();
        let name = unsafe { pyre_object::strobject::w_str_get_value(w_name) }.to_string();
        audit_find_class(&module, &name)?;
        // protocol < 3 with `fix_imports` applies the py2 → py3 `_compat_pickle`
        // forward map before resolution; otherwise the name resolves literally.
        let (module, name) = if self.proto < 3 && self.fix_imports {
            crate::module::_pickle::compat_map(&module, &name, false)
        } else {
            (module, name)
        };
        // proto >= 4 walks dotted qualnames; proto < 4 is a single getattr.
        let allow_qualname = self.proto >= 4;
        crate::module::_pickle::try_resolve_global(&module, &name, allow_qualname)?.ok_or_else(
            || {
                PyError::attribute_error(format!(
                    "Can't get attribute {name:?} on module {module:?}"
                ))
            },
        )
    }

    /// `Unpickler.persistent_load` — the per-instance persistent-id resolver.
    /// Once set, a PERSID / BINPERSID opcode calls it with the persistent id.
    /// A subclass may instead define a `persistent_load` method (resolved when
    /// the opcode is read). Reading it while unset raises `AttributeError` (a
    /// readable no-op default is omitted: a shared callable would need a
    /// GC-stable singleton under the relocating nursery).
    #[getter]
    fn persistent_load(&self) -> Result<PyObjectRef, PyError> {
        if self.w_persistent_load.is_null() {
            return Err(PyError::attribute_error(
                "'_pickle.Unpickler' object has no attribute 'persistent_load'".to_string(),
            ));
        }
        Ok(self.w_persistent_load)
    }

    #[setter]
    fn set_persistent_load(&mut self, w_value: PyObjectRef) {
        self.w_persistent_load = w_value;
    }

    #[deleter("persistent_load")]
    fn del_persistent_load(&mut self) {
        self.w_persistent_load = pyre_object::PY_NULL;
    }

    /// `Unpickler.memo` — a fresh `UnpicklerMemoProxy` viewing this unpickler's
    /// memo (CPython hands back a new proxy on each access).
    #[getter]
    fn memo(&self) -> PyObjectRef {
        let self_obj = self as *const W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        memo_proxy::type_object();
        let proxy = W_UnpicklerMemoProxy::allocate(W_UnpicklerMemoProxy {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_unpickler: pyre_object::PY_NULL,
        });
        // `allocate` may have relocated the unpickler; wire the (young) proxy to
        // its post-collection address.
        if let Some(px) = W_UnpicklerMemoProxy::from_obj(proxy) {
            px.w_unpickler = pyre_object::gc_roots::shadow_stack_get(slot);
        }
        proxy
    }

    /// `Unpickler.memo` setter — an `UnpicklerMemoProxy` snapshots the source
    /// unpickler's memo (read out as a `{index: obj}` dict via `copy`) and
    /// rebuilds it into this one's position-indexed memo list, NULL-filling any
    /// gap. A plain dict assignment validates its keys (non-negative integers)
    /// and then leaves the memo empty: the entries are written into the memo
    /// that is replaced wholesale. Any other type is a `TypeError`.
    #[setter]
    fn set_memo(&mut self, w_value: PyObjectRef) -> Result<(), PyError> {
        let self_obj = self as *mut W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self_obj);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if W_UnpicklerMemoProxy::from_obj(w_value).is_some() {
            let w_dict = call_meth(w_value, "copy", &[])?;
            pyre_object::gc_roots::pin_root(w_dict);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let (w_list, next) =
                memo_list_from_dict(pyre_object::gc_roots::shadow_stack_get(dict_slot))?;
            // `memo_list_from_dict`'s `w_list_new` may have collected; re-read self.
            let me = unsafe {
                &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut W_Unpickler)
            };
            me.w_memo = w_list;
            me.memo_index = next;
        } else if unsafe { pyre_object::is_dict(w_value) } {
            // Validate keys, then discard: a dict assignment yields an empty memo.
            let items = unsafe { pyre_object::dictmultiobject::w_dict_items(w_value) };
            for (k, _) in &items {
                if !unsafe { pyre_object::is_int(*k) } {
                    return Err(PyError::type_error("memo key must be integers"));
                }
                if crate::baseobjspace::int_w(*k)? < 0 {
                    return Err(PyError::value_error("memo key must be positive integers."));
                }
            }
            let empty = pyre_object::listobject::w_list_new(Vec::new());
            let me = unsafe {
                &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut W_Unpickler)
            };
            me.w_memo = empty;
            me.memo_index = 0;
        } else {
            return Err(PyError::type_error(format!(
                "'memo' attribute must be an UnpicklerMemoProxy object or dict, not {}",
                crate::baseobjspace::object_functionstr_type_name(w_value),
            )));
        }
        Ok(())
    }

    /// `Unpickler.memo` is not deletable.
    #[deleter("memo")]
    fn del_memo(&self) -> Result<(), PyError> {
        Err(PyError::type_error("attribute deletion is not supported"))
    }
}

/// `interp_pickle.py UnpicklerMemoProxy` — a live view of an unpickler's
/// index→object memo (a Python `dict`). `copy` snapshots it; `clear` resets it.
///
/// Held in its own module so `#[pyre_methods]` emits a `type_object()` that
/// does not clash with `W_Unpickler`'s (each impl emits a module-scoped one).
pub use memo_proxy::W_UnpicklerMemoProxy;

mod memo_proxy {
    use super::*;

    #[crate::pyre_class("_pickle.UnpicklerMemoProxy")]
    pub struct W_UnpicklerMemoProxy {
        pub(super) w_unpickler: PyObjectRef,
    }

    #[crate::pyre_methods(doc = "Proxy for an Unpickler's memo.")]
    impl W_UnpicklerMemoProxy {
        /// `UnpicklerMemoProxy.copy` — a shallow `{index: obj}` copy of the memo,
        /// projecting the position-indexed memo list (NULL slots omitted).
        fn copy(&self) -> Result<PyObjectRef, PyError> {
            let w_unpickler = self.w_unpickler;
            let w_memo = unsafe { &*(w_unpickler as *const W_Unpickler) }.w_memo;
            let w_dict = pyre_object::dictmultiobject::w_dict_new();
            if unsafe { pyre_object::is_none(w_memo) } {
                return Ok(w_dict);
            }
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(w_memo);
            let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            pyre_object::gc_roots::pin_root(w_dict);
            let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let len = unsafe {
                pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(
                    memo_slot,
                ))
            } as i64;
            for i in 0..len {
                let v = unsafe {
                    pyre_object::listobject::w_list_getitem(
                        pyre_object::gc_roots::shadow_stack_get(memo_slot),
                        i,
                    )
                };
                if let Some(v) = v {
                    if !v.is_null() {
                        // `w_dict_setitem` boxes the int key (`w_int_new`), which
                        // may collect and move `v`; pin it across the store.
                        let _r = pyre_object::gc_roots::push_roots();
                        pyre_object::gc_roots::pin_root(v);
                        let v_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                        let w_dict = pyre_object::gc_roots::shadow_stack_get(dict_slot);
                        unsafe {
                            pyre_object::dictmultiobject::w_dict_setitem(
                                w_dict,
                                i,
                                pyre_object::gc_roots::shadow_stack_get(v_slot),
                            )
                        };
                    }
                }
            }
            Ok(pyre_object::gc_roots::shadow_stack_get(dict_slot))
        }

        /// `UnpicklerMemoProxy.clear` — empty the unpickler's memo.
        fn clear(&self) {
            let w_unpickler = self.w_unpickler;
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(w_unpickler);
            let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let empty = pyre_object::listobject::w_list_new(Vec::new());
            let u = unsafe {
                &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Unpickler)
            };
            u.w_memo = empty;
            u.memo_index = 0;
        }
    }
}

/// Re-read the (possibly relocated) unpickler from the pinned shadow slot.
#[inline]
fn cur(slot: usize) -> &'static mut W_Unpickler {
    unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Unpickler) }
}

// ── stack / metastack helpers ────────────────────────────────────────

fn push(slot: usize, obj: PyObjectRef) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_stack, obj) };
}

/// `data_pop` — pop the top of the current stack.
fn pop(slot: usize) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_pop_end(me.w_stack) }
        .ok_or_else(|| unpickling_error("unpickling stack underflow"))
}

/// `_stack_top` — the top of the current stack without removing it.
fn top(slot: usize, opcode_name: &str) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    let n = unsafe { pyre_object::listobject::w_list_len(me.w_stack) };
    if n < 1 {
        return Err(unpickling_error(&format!("stack empty in {opcode_name}")));
    }
    Ok(unsafe { pyre_object::listobject::w_list_getitem(me.w_stack, (n - 1) as i64).unwrap() })
}

/// `load_mark` — save the current stack and start a fresh one.
fn mark(slot: usize) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_metastack, me.w_stack) };
    let new_stack = pyre_object::listobject::w_list_new(Vec::new());
    cur(slot).w_stack = new_stack;
}

/// `pop_mark` — return the items pushed since the last MARK and restore the
/// previous stack.
fn pop_mark(slot: usize) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    let items = me.w_stack;
    let prev = unsafe { pyre_object::listobject::w_list_pop_end(me.w_metastack) }
        .ok_or_else(|| unpickling_error("no items on stack"))?;
    cur(slot).w_stack = prev;
    Ok(items)
}

// ── out-of-band buffers ──────────────────────────────────────────────

/// `load_next_buffer` (NEXT_BUFFER) — push the next buffer from the
/// `buffers` iterator given at construction.
fn load_next_buffer(slot: usize) -> Result<(), PyError> {
    let w_buffers = cur(slot).w_buffers;
    if unsafe { pyre_object::is_none(w_buffers) } {
        return Err(unpickling_error(
            "pickle stream refers to out-of-band data but no *buffers* argument was given",
        ));
    }
    let w_buf = match crate::baseobjspace::next(w_buffers) {
        Ok(b) => b,
        Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
            return Err(unpickling_error("not enough out-of-band buffers"));
        }
        Err(e) => return Err(e),
    };
    push(slot, w_buf);
    Ok(())
}

/// `load_readonly_buffer` (READONLY_BUFFER) — replace the top buffer with a
/// read-only memoryview onto it.
fn load_readonly_buffer(slot: usize) -> Result<(), PyError> {
    let w_buf = top(slot, "READONLY_BUFFER")?;
    let w_mv = call_fn(memoryview_type()?, &[w_buf])?;
    let w_readonly = call_meth(w_mv, "toreadonly", &[])?;
    // Replace the top of the stack (`stack[-1] = w_readonly`).
    pop(slot)?;
    push(slot, w_readonly);
    Ok(())
}

/// The `memoryview` builtin type via the live execution context.
fn memoryview_type() -> Result<PyObjectRef, PyError> {
    let frame = crate::eval::CURRENT_FRAME.with(|f| f.get());
    let ec = if frame.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*frame).execution_context }
    };
    if ec.is_null() {
        return Err(unpickling_error("memoryview type unavailable"));
    }
    unsafe { (*ec).lookup_builtin("memoryview") }
        .ok_or_else(|| unpickling_error("memoryview type unavailable"))
}

// ── memo helpers ─────────────────────────────────────────────────────

/// `_memo_put` — store `w_val` at index `i`, growing the list (NULL-filling any
/// gap) and advancing the next-free slot.
fn memo_put(slot: usize, i: i64, w_val: PyObjectRef) {
    let me = cur(slot);
    let len = unsafe { pyre_object::listobject::w_list_len(me.w_memo) } as i64;
    if i < len {
        // In-range overwrite: no allocation, so no relocation.
        unsafe { pyre_object::listobject::w_list_setitem(me.w_memo, i, w_val) };
    } else {
        // Grow. `w_list_append` may collect; pin the value and re-read self/list.
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_val);
        let val_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // NULL-fill through index `i` (inclusive). Appending `PY_NULL` forces the
        // Object strategy, so memoized values keep pointer identity on GET.
        while (unsafe { pyre_object::listobject::w_list_len(cur(slot).w_memo) } as i64) <= i {
            unsafe {
                pyre_object::listobject::w_list_append(cur(slot).w_memo, pyre_object::PY_NULL)
            };
        }
        unsafe {
            pyre_object::listobject::w_list_setitem(
                cur(slot).w_memo,
                i,
                pyre_object::gc_roots::shadow_stack_get(val_slot),
            )
        };
    }
    let me = cur(slot);
    if i >= me.memo_index {
        me.memo_index = i + 1;
    }
}

/// `_memo_append` — store `w_val` at the next free slot.
fn memo_append(slot: usize, w_val: PyObjectRef) {
    let i = cur(slot).memo_index;
    memo_put(slot, i, w_val);
}

fn memo_get(slot: usize, i: i64) -> Result<PyObjectRef, PyError> {
    // A negative index must not wrap (the list would index from the end); treat
    // it as absent, matching the prior dict lookup.
    if i < 0 {
        return Err(unpickling_error(&format!(
            "Memo value not found at index {i}"
        )));
    }
    let me = cur(slot);
    match unsafe { pyre_object::listobject::w_list_getitem(me.w_memo, i) } {
        Some(v) if !v.is_null() => Ok(v),
        _ => Err(unpickling_error(&format!(
            "Memo value not found at index {i}"
        ))),
    }
}

/// Build a position-indexed memo `list` from a `{index: obj}` dict (as produced
/// by `UnpicklerMemoProxy.copy`), NULL-filling any gap. Returns the list and the
/// next-free index (`max_index + 1`).
fn memo_list_from_dict(w_dict: PyObjectRef) -> Result<(PyObjectRef, i64), PyError> {
    let items = unsafe { pyre_object::dictmultiobject::w_dict_items(w_dict) };
    let mut max_idx: i64 = -1;
    for (k, _) in &items {
        let idx = crate::baseobjspace::int_w(*k)?;
        if idx > max_idx {
            max_idx = idx;
        }
    }
    let mut slots: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; (max_idx + 1) as usize];
    for (k, v) in &items {
        let idx = crate::baseobjspace::int_w(*k)? as usize;
        slots[idx] = *v;
    }
    Ok((
        pyre_object::listobject::w_list_new_object(slots),
        max_idx + 1,
    ))
}

// ── reading ──────────────────────────────────────────────────────────

/// Read one opcode byte (from the active frame, else the file).
fn read1(slot: usize) -> Result<u8, PyError> {
    let me = cur(slot);
    if !unsafe { pyre_object::is_none(me.w_frame) } {
        let frame = unsafe { pyre_object::bytesobject::w_bytes_data(me.w_frame) };
        let idx = me.frame_index as usize;
        if idx < frame.len() {
            me.frame_index += 1;
            return Ok(frame[idx]);
        }
    }
    let v = read(slot, 1)?;
    Ok(v[0])
}

/// Read `n` bytes (from the active frame, else the file). Returns an owned
/// copy so the result survives later allocations.
fn read(slot: usize, n: usize) -> Result<Vec<u8>, PyError> {
    let me = cur(slot);
    if !unsafe { pyre_object::is_none(me.w_frame) } {
        let frame = unsafe { pyre_object::bytesobject::w_bytes_data(me.w_frame) };
        let idx = me.frame_index as usize;
        if idx + n <= frame.len() {
            let out = frame[idx..idx + n].to_vec();
            me.frame_index += n as i64;
            return Ok(out);
        }
        // Frame exhausted — fall through to the file.
        me.w_frame = pyre_object::w_none();
        me.frame_index = 0;
    }
    let w_n = pyre_object::w_int_new(n as i64);
    let read_fn = cur(slot).w_file_read;
    let w_res = call_fn(read_fn, &[w_n])?;
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_res) };
    if data.len() < n {
        return Err(unpickling_error("pickle data was truncated"));
    }
    Ok(data[..n].to_vec())
}

fn dispatch(slot: usize, opcode: u8) -> Result<(), PyError> {
    match opcode {
        x if x == op::PROTO => {
            let p = read1(slot)? as i64;
            if !(0..=HIGHEST_PROTOCOL).contains(&p) {
                return Err(PyError::value_error("unsupported pickle protocol"));
            }
            cur(slot).proto = p;
        }
        x if x == op::FRAME => {
            let sz = read(slot, 8)?;
            let frame_size = read_int_le(&sz) as usize;
            // Load the frame body from the file.
            let w_n = pyre_object::w_int_new(frame_size as i64);
            let read_fn = cur(slot).w_file_read;
            let w_res = call_fn(read_fn, &[w_n])?;
            let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_res) };
            if data.len() < frame_size {
                return Err(unpickling_error("pickle data was truncated"));
            }
            let w_frame = pyre_object::w_bytes_from_bytes(&data[..frame_size]);
            let me = cur(slot);
            me.w_frame = w_frame;
            me.frame_index = 0;
        }
        x if x == op::NONE => push(slot, pyre_object::w_none()),
        x if x == op::NEWTRUE => push(slot, pyre_object::w_bool_from(true)),
        x if x == op::NEWFALSE => push(slot, pyre_object::w_bool_from(false)),
        x if x == op::BININT => {
            let d = read(slot, 4)?;
            let v = i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            push(slot, pyre_object::w_int_new(v));
        }
        x if x == op::BININT1 => {
            let d = read(slot, 1)?;
            push(slot, pyre_object::w_int_new(d[0] as i64));
        }
        x if x == op::BININT2 => {
            let d = read(slot, 2)?;
            push(
                slot,
                pyre_object::w_int_new(u16::from_le_bytes([d[0], d[1]]) as i64),
            );
        }
        x if x == op::LONG1 => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, decode_long(&d));
        }
        x if x == op::LONG4 => {
            let nb = read(slot, 4)?;
            let n = i32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]);
            if n < 0 {
                // Corrupt or hostile pickle -- we never write one like this.
                return Err(unpickling_error("LONG pickle has negative byte count"));
            }
            let d = read(slot, n as usize)?;
            push(slot, decode_long(&d));
        }
        x if x == op::BINFLOAT => {
            let d = read(slot, 8)?;
            let f = f64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
            push(slot, pyre_object::w_float_new(f));
        }
        x if x == op::INT => {
            let s = read_line(slot)?;
            let w = match s.as_str() {
                "00" => pyre_object::w_bool_from(false),
                "01" => pyre_object::w_bool_from(true),
                _ => parse_int_text(&s)?,
            };
            push(slot, w);
        }
        x if x == op::LONG => {
            let mut s = read_line(slot)?;
            // strip the Python 2 'L' suffix, if present.
            if s.ends_with('L') {
                s.pop();
            }
            push(slot, parse_int_text(&s)?);
        }
        x if x == op::FLOAT => {
            let s = read_line(slot)?;
            let f = s
                .trim()
                .parse::<f64>()
                .map_err(|_| PyError::value_error("could not convert string to float"))?;
            push(slot, pyre_object::w_float_new(f));
        }
        x if x == op::UNICODE => {
            // raw-unicode-escape over the line's raw bytes.
            let data = read_line_bytes(slot)?;
            let w_bytes = pyre_object::w_bytes_from_bytes(&data);
            let w_uni = call_meth(
                w_bytes,
                "decode",
                &[pyre_object::w_str_new("raw-unicode-escape")],
            )?;
            push(slot, w_uni);
        }
        x if x == op::SHORT_BINUNICODE => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::BINUNICODE => {
            let nb = read(slot, 4)?;
            let n = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::BINUNICODE8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::SHORT_BINBYTES => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        x if x == op::BINBYTES => {
            let nb = read(slot, 4)?;
            let n = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        x if x == op::BINBYTES8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        // ── legacy protocol-0/1 str ───────────────────────────────────
        x if x == op::STRING => {
            // A protocol-0 quoted py2 str: strip the matching outer quotes,
            // decode the bytes-literal escapes, then apply `encoding`/`errors`.
            let line = read_line_bytes(slot)?;
            let data = strip_string_quotes(&line)?;
            let raw = escape_decode(&data)?;
            let w = decode_string(slot, &raw)?;
            push(slot, w);
        }
        x if x == op::BINSTRING => {
            // Deprecated BINSTRING uses a signed 32-bit length.
            let nb = read(slot, 4)?;
            let n = i32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]);
            if n < 0 {
                return Err(unpickling_error("BINSTRING pickle has negative byte count"));
            }
            let d = read(slot, n as usize)?;
            let w = decode_string(slot, &d)?;
            push(slot, w);
        }
        x if x == op::SHORT_BINSTRING => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            let w = decode_string(slot, &d)?;
            push(slot, w);
        }
        // ── stack ────────────────────────────────────────────────────
        x if x == op::MARK => mark(slot),
        x if x == op::POP => {
            // Pop a stack item, or discard the topmost MARK group.
            let me = cur(slot);
            let n = unsafe { pyre_object::listobject::w_list_len(me.w_stack) };
            if n > 0 {
                pop(slot)?;
            } else {
                pop_mark(slot)?;
            }
        }
        x if x == op::POP_MARK => {
            pop_mark(slot)?;
        }
        x if x == op::DUP => {
            let v = top(slot, "DUP")?;
            push(slot, v);
        }
        // ── tuple ─────────────────────────────────────────────────────
        x if x == op::EMPTY_TUPLE => push(slot, pyre_object::tupleobject::w_tuple_new(Vec::new())),
        x if x == op::TUPLE => {
            let items = pop_mark(slot)?;
            push(slot, list_to_tuple(items));
        }
        x if x == op::TUPLE1 => {
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a]));
        }
        x if x == op::TUPLE2 => {
            let b = pop(slot)?;
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a, b]));
        }
        x if x == op::TUPLE3 => {
            let c = pop(slot)?;
            let b = pop(slot)?;
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a, b, c]));
        }
        // ── list ──────────────────────────────────────────────────────
        x if x == op::EMPTY_LIST => push(slot, pyre_object::listobject::w_list_new(Vec::new())),
        x if x == op::LIST => {
            let items = pop_mark(slot)?;
            push(slot, list_copy(items));
        }
        x if x == op::APPEND => {
            let value = pop(slot)?;
            let w_list = top(slot, "APPEND")?;
            call_meth(w_list, "append", &[value])?;
        }
        x if x == op::APPENDS => {
            let items = pop_mark(slot)?;
            // Pin `items` so the `extend`/`append` lookups (which may allocate
            // and collect) do not strand it; re-read the list top per call.
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(items);
            let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_list = top(slot, "APPENDS")?;
            match crate::baseobjspace::findattr(w_list, "extend") {
                Some(extend) if !unsafe { pyre_object::is_none(extend) } => {
                    call_fn(
                        extend,
                        &[pyre_object::gc_roots::shadow_stack_get(items_slot)],
                    )?;
                }
                _ => {
                    // PEP 307 requires extend(); fall back to append() for
                    // objects lacking it (backward compatibility).
                    let n = unsafe {
                        pyre_object::listobject::w_list_len(
                            pyre_object::gc_roots::shadow_stack_get(items_slot),
                        )
                    };
                    for i in 0..n {
                        let w_list = top(slot, "APPENDS")?;
                        let item = unsafe {
                            pyre_object::listobject::w_list_getitem(
                                pyre_object::gc_roots::shadow_stack_get(items_slot),
                                i as i64,
                            )
                            .unwrap()
                        };
                        call_meth(w_list, "append", &[item])?;
                    }
                }
            }
        }
        // ── dict ──────────────────────────────────────────────────────
        x if x == op::EMPTY_DICT => push(slot, pyre_object::dictmultiobject::w_dict_new()),
        x if x == op::DICT => {
            let items = pop_mark(slot)?;
            let w_dict = pyre_object::dictmultiobject::w_dict_new();
            dict_update_from_pairs(w_dict, items)?;
            push(slot, w_dict);
        }
        x if x == op::SETITEM => {
            let value = pop(slot)?;
            let key = pop(slot)?;
            let w_dict = top(slot, "SETITEM")?;
            crate::baseobjspace::setitem(w_dict, key, value)?;
        }
        x if x == op::SETITEMS => {
            let items = pop_mark(slot)?;
            let w_dict = top(slot, "SETITEMS")?;
            dict_update_from_pairs(w_dict, items)?;
        }
        // ── set / frozenset ───────────────────────────────────────────
        x if x == op::EMPTY_SET => push(slot, pyre_object::setobject::w_set_new()),
        x if x == op::FROZENSET => {
            let items = pop_mark(slot)?;
            push(slot, list_to_frozenset(items));
        }
        x if x == op::ADDITEMS => {
            let items = pop_mark(slot)?;
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(items);
            let items_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_set = top(slot, "ADDITEMS")?;
            let w_set_type = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
            if unsafe { crate::baseobjspace::isinstance_w(w_set, w_set_type) } {
                // `PySet_Check`: `set` and its subclasses (not `frozenset`)
                // take the `update` path; `set.update(items)` dispatches
                // through the (possibly overridden) method.
                call_meth(
                    w_set,
                    "update",
                    &[pyre_object::gc_roots::shadow_stack_get(items_slot)],
                )?;
            } else {
                let n = unsafe {
                    pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(
                        items_slot,
                    ))
                };
                for i in 0..n {
                    let w_set = top(slot, "ADDITEMS")?;
                    let item = unsafe {
                        pyre_object::listobject::w_list_getitem(
                            pyre_object::gc_roots::shadow_stack_get(items_slot),
                            i as i64,
                        )
                        .unwrap()
                    };
                    call_meth(w_set, "add", &[item])?;
                }
            }
        }
        // ── bytearray ─────────────────────────────────────────────────
        x if x == op::BYTEARRAY8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(
                slot,
                pyre_object::bytearrayobject::w_bytearray_from_bytes(&d),
            );
        }
        // ── proto-5 out-of-band buffers ───────────────────────────────
        x if x == op::NEXT_BUFFER => load_next_buffer(slot)?,
        x if x == op::READONLY_BUFFER => load_readonly_buffer(slot)?,
        // ── global / reduce / build ───────────────────────────────────
        x if x == op::GLOBAL => {
            let module = read_line(slot)?;
            let name = read_line(slot)?;
            let v = call_find_class_names(slot, &module, &name)?;
            push(slot, v);
        }
        x if x == op::STACK_GLOBAL => {
            let w_name = pop(slot)?;
            let w_module = pop(slot)?;
            if !unsafe { pyre_object::is_str(w_name) } || !unsafe { pyre_object::is_str(w_module) }
            {
                return Err(unpickling_error("STACK_GLOBAL requires str"));
            }
            let v = call_find_class(slot, w_module, w_name)?;
            push(slot, v);
        }
        x if x == op::EXT1 => {
            let code = read(slot, 1)?[0] as i64;
            get_extension(slot, code)?;
        }
        x if x == op::EXT2 => {
            let d = read(slot, 2)?;
            let code = u16::from_le_bytes([d[0], d[1]]) as i64;
            get_extension(slot, code)?;
        }
        x if x == op::EXT4 => {
            let d = read(slot, 4)?;
            let code = i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            get_extension(slot, code)?;
        }
        x if x == op::REDUCE => {
            let w_args = pop(slot)?;
            let w_func = pop(slot)?;
            let args = tuple_items(w_args);
            let w_obj = call_fn(w_func, &args)?;
            push(slot, w_obj);
        }
        x if x == op::NEWOBJ => {
            let w_args = pop(slot)?;
            let w_cls = pop(slot)?;
            let w_obj = new_instance(w_cls, &tuple_items(w_args))?;
            push(slot, w_obj);
        }
        x if x == op::NEWOBJ_EX => {
            let w_kwargs = pop(slot)?;
            let w_args = pop(slot)?;
            let w_cls = pop(slot)?;
            let kw_items = unsafe { pyre_object::dictmultiobject::w_dict_items(w_kwargs) };
            let args = tuple_items(w_args);
            let w_obj = if kw_items.is_empty() {
                new_instance(w_cls, &args)?
            } else {
                new_instance_kw(w_cls, &args, &kw_items)?
            };
            push(slot, w_obj);
        }
        x if x == op::BUILD => {
            let w_state = pop(slot)?;
            let w_inst = top(slot, "BUILD")?;
            build_instance(w_inst, w_state)?;
        }
        // ── memo ──────────────────────────────────────────────────────
        x if x == op::MEMOIZE => {
            let v = top(slot, "MEMOIZE")?;
            memo_append(slot, v);
        }
        x if x == op::PUT => {
            let i = read_line_int(slot)?;
            if i < 0 {
                return Err(PyError::value_error("negative PUT argument"));
            }
            let v = top(slot, "PUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::BINPUT => {
            let i = read(slot, 1)?[0] as i64;
            let v = top(slot, "BINPUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::LONG_BINPUT => {
            let d = read(slot, 4)?;
            let i = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            let v = top(slot, "LONG_BINPUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::GET => {
            let i = read_line_int(slot)?;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::BINGET => {
            let i = read(slot, 1)?[0] as i64;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::LONG_BINGET => {
            let d = read(slot, 4)?;
            let i = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::PERSID => {
            let pid = read_line_bytes(slot)?;
            if !pid.is_ascii() {
                return Err(unpickling_error(
                    "persistent IDs in protocol 0 must be ASCII strings",
                ));
            }
            let w_pid = str_from_utf8(&pid)?;
            let v = persistent_load(slot, w_pid)?;
            push(slot, v);
        }
        x if x == op::BINPERSID => {
            let w_pid = pop(slot)?;
            let v = persistent_load(slot, w_pid)?;
            push(slot, v);
        }
        x if x == op::INST => {
            let module = read_line(slot)?;
            let name = read_line(slot)?;
            let w_cls = call_find_class_names(slot, &module, &name)?;
            let w_args = pop_mark(slot)?;
            let v = instantiate(w_cls, w_args)?;
            push(slot, v);
        }
        x if x == op::OBJ => {
            let args = pop_mark(slot)?;
            let n = unsafe { pyre_object::listobject::w_list_len(args) };
            if n == 0 {
                return Err(unpickling_error("OBJ opcode with empty stack"));
            }
            let w_cls = unsafe { pyre_object::listobject::w_list_getitem(args, 0).unwrap() };
            let rest: Vec<PyObjectRef> = (1..n)
                .map(|i| unsafe {
                    pyre_object::listobject::w_list_getitem(args, i as i64).unwrap()
                })
                .collect();
            let v = instantiate(w_cls, pyre_object::listobject::w_list_new(rest))?;
            push(slot, v);
        }
        _ => {
            return Err(unpickling_error("unsupported opcode in this build"));
        }
    }
    Ok(())
}

/// Build a tuple from the items of a (popped) stack list.
fn list_to_tuple(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    let v: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() })
        .collect();
    pyre_object::tupleobject::w_tuple_new(v)
}

/// Build a frozenset from the items of a (popped) stack list.
fn list_to_frozenset(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    let v: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() })
        .collect();
    pyre_object::setobject::w_frozenset_from_items(&v)
}

/// Copy a (popped) stack list into a fresh list.
fn list_copy(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    let v: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() })
        .collect();
    pyre_object::listobject::w_list_new(v)
}

/// Set `dict[items[2k]] = items[2k+1]` for each pair in a (popped) stack list.
fn dict_update_from_pairs(w_dict: PyObjectRef, items: PyObjectRef) -> Result<(), PyError> {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    if n % 2 != 0 {
        return Err(unpickling_error("odd number of items for DICT"));
    }
    let mut i = 0;
    while i < n {
        let k = unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() };
        let v = unsafe { pyre_object::listobject::w_list_getitem(items, (i + 1) as i64).unwrap() };
        crate::baseobjspace::setitem(w_dict, k, v)?;
        i += 2;
    }
    Ok(())
}

/// Read a newline-terminated line (without the trailing newline).
fn read_line_bytes(slot: usize) -> Result<Vec<u8>, PyError> {
    let mut bytes: Vec<u8> = Vec::new();
    loop {
        let b = read1(slot)?;
        if b == b'\n' {
            break;
        }
        bytes.push(b);
    }
    Ok(bytes)
}

fn read_line(slot: usize) -> Result<String, PyError> {
    let bytes = read_line_bytes(slot)?;
    String::from_utf8(bytes).map_err(|_| unpickling_error("invalid utf-8 in pickle line"))
}

/// Read a newline-terminated decimal integer argument (GET / PUT in the
/// text protocols).
fn read_line_int(slot: usize) -> Result<i64, PyError> {
    let s = read_line(slot)?;
    s.trim()
        .parse::<i64>()
        .map_err(|_| PyError::value_error("invalid int literal"))
}

/// Dispatch to `self.find_class(module, name)` through the instance, so a
/// Python subclass override (the standard security hook) is honoured.
/// `call_find_class` for two fresh module/name strings: allocate and pin the
/// module string before allocating the name string, so the second `w_str_new`
/// cannot relocate the first before both are rooted.
fn call_find_class_names(slot: usize, module: &str, name: &str) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    let w_module = pyre_object::w_str_new(module);
    pyre_object::gc_roots::pin_root(w_module);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_name = pyre_object::w_str_new(name);
    call_find_class(
        slot,
        pyre_object::gc_roots::shadow_stack_get(module_slot),
        w_name,
    )
}

fn call_find_class(
    slot: usize,
    w_module: PyObjectRef,
    w_name: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    // Resolving the bound `find_class` allocates, which can move the nursery;
    // pin the arguments (and re-read `self`) across the lookup.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_module);
    let module_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(w_name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let method = crate::baseobjspace::getattr_str(
        pyre_object::gc_roots::shadow_stack_get(slot),
        "find_class",
    )?;
    call_fn(
        method,
        &[
            pyre_object::gc_roots::shadow_stack_get(module_slot),
            pyre_object::gc_roots::shadow_stack_get(name_slot),
        ],
    )
}

/// Emit the `pickle.find_class` audit event. `interp_pickle.py:2601` calls
/// `space.audit(...)` and lets a blocking audit hook's error propagate.
fn audit_find_class(module: &str, name: &str) -> Result<(), PyError> {
    if let Ok(sys) = import_module("sys") {
        if let Ok(audit) = crate::baseobjspace::getattr_str(sys, "audit") {
            call_fn(
                audit,
                &[
                    pyre_object::w_str_new("pickle.find_class"),
                    pyre_object::w_str_new(module),
                    pyre_object::w_str_new(name),
                ],
            )?;
        }
    }
    Ok(())
}

/// `get_extension` (EXT1 / EXT2 / EXT4) — resolve a `copyreg` extension code
/// to its registered global via `_extension_cache` / `_inverted_registry`.
fn get_extension(slot: usize, code: i64) -> Result<(), PyError> {
    let copyreg = import_module("copyreg")?;
    let cache = crate::baseobjspace::getattr_str(copyreg, "_extension_cache")?;
    if let Some(obj) = unsafe { pyre_object::w_dict_lookup(cache, pyre_object::w_int_new(code)) } {
        push(slot, obj);
        return Ok(());
    }
    let inverted = crate::baseobjspace::getattr_str(copyreg, "_inverted_registry")?;
    let key = match unsafe { pyre_object::w_dict_lookup(inverted, pyre_object::w_int_new(code)) } {
        Some(k) if !unsafe { pyre_object::is_none(k) } => k,
        _ => {
            if code <= 0 {
                // Corrupt or hostile pickle (0 is forbidden by add_extension).
                return Err(unpickling_error("EXT specifies code <= 0"));
            }
            return Err(PyError::value_error(format!(
                "unregistered extension code {code}"
            )));
        }
    };
    // `key` is the `(module, name)` tuple registered for this code.
    let w_module = unsafe { pyre_object::tupleobject::w_tuple_getitem(key, 0).unwrap() };
    let w_name = unsafe { pyre_object::tupleobject::w_tuple_getitem(key, 1).unwrap() };
    let obj = call_find_class(slot, w_module, w_name)?;
    // `_extension_cache[code] = obj`; pin `obj` across the dict lookups.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let copyreg = import_module("copyreg")?;
    let cache = crate::baseobjspace::getattr_str(copyreg, "_extension_cache")?;
    crate::baseobjspace::setitem(
        cache,
        pyre_object::w_int_new(code),
        pyre_object::gc_roots::shadow_stack_get(obj_slot),
    )?;
    push(slot, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(())
}

/// `_decode_string` — turn the raw bytes of a STRING / BINSTRING /
/// SHORT_BINSTRING into the object the unpickler pushes: the raw `bytes` when
/// `encoding == "bytes"`, otherwise `bytes.decode(encoding, errors)`.
fn decode_string(slot: usize, data: &[u8]) -> Result<PyObjectRef, PyError> {
    let (encoding, errors, as_bytes) = {
        let me = cur(slot);
        (
            me.encoding.clone(),
            me.errors.clone(),
            me.encoding == "bytes",
        )
    };
    if as_bytes {
        return Ok(pyre_object::w_bytes_from_bytes(data));
    }
    let _roots = pyre_object::gc_roots::push_roots();
    let w_bytes = pyre_object::w_bytes_from_bytes(data);
    pyre_object::gc_roots::pin_root(w_bytes);
    let b = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_encoding = pyre_object::w_str_new(&encoding);
    pyre_object::gc_roots::pin_root(w_encoding);
    let e = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_errors = pyre_object::w_str_new(&errors);
    call_meth(
        pyre_object::gc_roots::shadow_stack_get(b),
        "decode",
        &[pyre_object::gc_roots::shadow_stack_get(e), w_errors],
    )
}

/// Strip the matching outer quotes from a protocol-0 STRING argument.
fn strip_string_quotes(line: &[u8]) -> Result<Vec<u8>, PyError> {
    if line.len() >= 2 && line[0] == line[line.len() - 1] && (line[0] == b'"' || line[0] == b'\'') {
        Ok(line[1..line.len() - 1].to_vec())
    } else {
        Err(unpickling_error(
            "the STRING opcode argument must be quoted",
        ))
    }
}

/// `codecs.escape_decode` over a byte string — decode the Python bytes-literal
/// escapes (`PyBytes_DecodeEscape` semantics, `strict`). Unrecognised escapes
/// keep the backslash verbatim.
fn escape_decode(data: &[u8]) -> Result<Vec<u8>, PyError> {
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        let c = data[i];
        if c != b'\\' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        if i >= data.len() {
            // Trailing backslash — kept verbatim.
            out.push(b'\\');
            break;
        }
        let e = data[i];
        i += 1;
        match e {
            b'\n' => {} // line continuation
            b'\\' => out.push(b'\\'),
            b'\'' => out.push(b'\''),
            b'"' => out.push(b'"'),
            b'a' => out.push(0x07),
            b'b' => out.push(0x08),
            b'f' => out.push(0x0c),
            b'n' => out.push(b'\n'),
            b'r' => out.push(b'\r'),
            b't' => out.push(b'\t'),
            b'v' => out.push(0x0b),
            b'0'..=b'7' => {
                // up to three octal digits (the first already consumed).
                let mut val = (e - b'0') as u32;
                let mut k = 0;
                while k < 2 && i < data.len() && (b'0'..=b'7').contains(&data[i]) {
                    val = val * 8 + (data[i] - b'0') as u32;
                    i += 1;
                    k += 1;
                }
                out.push(val as u8);
            }
            b'x' => {
                let hi = data.get(i).and_then(|&b| (b as char).to_digit(16));
                let lo = data.get(i + 1).and_then(|&b| (b as char).to_digit(16));
                match (hi, lo) {
                    (Some(h), Some(l)) => {
                        out.push((h * 16 + l) as u8);
                        i += 2;
                    }
                    _ => return Err(unpickling_error("invalid \\x escape in STRING")),
                }
            }
            other => {
                // Unrecognised escape: keep the backslash and the character.
                out.push(b'\\');
                out.push(other);
            }
        }
    }
    Ok(out)
}

/// Resolve and invoke `self.persistent_load(pid)` (PERSID / BINPERSID).
fn persistent_load(slot: usize, w_pid: PyObjectRef) -> Result<PyObjectRef, PyError> {
    // `w_pid` was just popped from the stack / freshly built for the opcode and
    // is not in a GC-walked container; resolving the bound `persistent_load`
    // allocates and can move the nursery, so pin it across the lookup.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_pid);
    let pid_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let self_obj = pyre_object::gc_roots::shadow_stack_get(slot);
    // `findattr_result` propagates a descriptor's own error instead of panicking;
    // an explicit `persistent_load = None` is kept as the hook so `call_fn(None,
    // pid)` raises `TypeError: 'NoneType' object is not callable` (only an absent
    // attribute disables the hook).
    match crate::baseobjspace::findattr_result(self_obj, "persistent_load")? {
        Some(f) => call_fn(f, &[pyre_object::gc_roots::shadow_stack_get(pid_slot)]),
        None => Err(unpickling_error(
            "A load persistent id instruction was encountered, but no persistent_load function was specified.",
        )),
    }
}

/// `_instantiate` — build an old-style INST / OBJ instance. With args, or a
/// non-type class, or a `__getinitargs__`, call the class; otherwise build
/// via `__new__` without invoking `__init__`.
fn instantiate(w_cls: PyObjectRef, w_args: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let n = unsafe { pyre_object::listobject::w_list_len(w_args) };
    let has_getinitargs = crate::baseobjspace::findattr(w_cls, "__getinitargs__").is_some();
    let is_type = unsafe { pyre_object::typeobject::is_type(w_cls) };
    if n > 0 || !is_type || has_getinitargs {
        let args: Vec<PyObjectRef> = (0..n)
            .map(|i| unsafe { pyre_object::listobject::w_list_getitem(w_args, i as i64).unwrap() })
            .collect();
        call_fn(w_cls, &args)
    } else {
        new_instance(w_cls, &[])
    }
}

/// `cls.__new__(cls, *args)`.
fn new_instance(w_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_new = crate::baseobjspace::getattr_str(w_cls, "__new__")?;
    let mut call_args = vec![w_cls];
    call_args.extend_from_slice(args);
    call_fn(w_new, &call_args)
}

/// `cls.__new__(cls, *args, **kwargs)` — NEWOBJ_EX with the keyword
/// arguments returned by `__getnewargs_ex__`. Keyword delivery to a
/// user `__new__` needs the frame-based call path (`call_with_kwargs`);
/// the flat-slice path binds every argument positionally.
fn new_instance_kw(
    w_cls: PyObjectRef,
    args: &[PyObjectRef],
    kw_items: &[(PyObjectRef, PyObjectRef)],
) -> Result<PyObjectRef, PyError> {
    let w_new = crate::baseobjspace::getattr_str(w_cls, "__new__")?;
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(w_cls);
    call_args.extend_from_slice(args);
    let mut kwargs = Vec::with_capacity(kw_items.len());
    for &(k, v) in kw_items {
        if !unsafe { pyre_object::is_str(k) } {
            return Err(unpickling_error("keyword arguments must be strings"));
        }
        let name = unsafe { pyre_object::strobject::w_str_get_wtf8(k) }.to_owned();
        kwargs.push((name, v));
    }
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return Err(unpickling_error("no execution context for NEWOBJ_EX"));
    }
    let frame = unsafe { (*ec).gettopframe() };
    if frame.is_null() {
        return Err(unpickling_error("no frame for NEWOBJ_EX with kwargs"));
    }
    crate::call::call_with_kwargs(unsafe { &mut *frame }, w_new, &call_args, &kwargs)
}

/// `load_build` — apply pickled state to a freshly created instance.
fn build_instance(w_inst: PyObjectRef, w_state: PyObjectRef) -> Result<(), PyError> {
    // __setstate__ takes precedence.
    if let Some(setstate) = crate::baseobjspace::findattr(w_inst, "__setstate__") {
        if !unsafe { pyre_object::is_none(setstate) } {
            call_fn(setstate, &[w_state])?;
            return Ok(());
        }
    }

    // state may be a (dict-state, slot-state) pair.
    let (w_dict_state, w_slot_state) = if unsafe { pyre_object::is_tuple(w_state) }
        && unsafe { pyre_object::tupleobject::w_tuple_len(w_state) } == 2
    {
        (
            unsafe { pyre_object::tupleobject::w_tuple_getitem(w_state, 0).unwrap() },
            unsafe { pyre_object::tupleobject::w_tuple_getitem(w_state, 1).unwrap() },
        )
    } else {
        (w_state, pyre_object::w_none())
    };

    if !unsafe { pyre_object::is_none(w_dict_state) } {
        let w_inst_dict = crate::baseobjspace::getattr_str(w_inst, "__dict__")?;
        call_meth(w_inst_dict, "update", &[w_dict_state])?;
    }
    if !unsafe { pyre_object::is_none(w_slot_state) } {
        for (k, v) in unsafe { pyre_object::dictmultiobject::w_dict_items(w_slot_state) } {
            crate::baseobjspace::setattr(w_inst, k, v)?;
        }
    }
    Ok(())
}

fn tuple_items(w_tuple: PyObjectRef) -> Vec<PyObjectRef> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_tuple) };
    (0..n)
        .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, i as i64).unwrap() })
        .collect()
}
