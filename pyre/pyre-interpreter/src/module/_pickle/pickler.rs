//! `_pickle.Pickler` — `interp_pickle.py W_Pickler` (atom + container subset).

use std::collections::HashMap;

use malachite_bigint::BigInt;
use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    BATCHSIZE, DEFAULT_PROTOCOL, FRAME_SIZE_MIN, FRAME_SIZE_TARGET, HIGHEST_PROTOCOL, call_fn,
    call_meth, encode_long, getattribute_dotted, import_module, op, pickling_error,
};

#[crate::pyre_class("_pickle.Pickler")]
pub struct W_Pickler {
    /// Output file (has a `write` method).
    w_file: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
    /// Apply the `_compat_pickle` py3→py2 name remap at protocol < 3.
    fix_imports: bool,
    /// `buffer_callback` for proto-5 out-of-band buffers, or `None`.
    buffer_callback: PyObjectRef,
    /// Memo of saved objects — a Python `list` (GC-walked) persisted across
    /// `dump` calls until `clear_memo`, position = memo index.
    w_memo: PyObjectRef,
    /// `fast` mode — when non-zero, memoization is skipped. Stored as the
    /// assigned integer (`space.int_w`) so `Pickler.fast` round-trips.
    fast: i64,
    /// User-set `dispatch_table` mapping, or `PY_NULL` when unset; the dump
    /// path falls back to `copyreg.dispatch_table` when unset.
    w_dispatch_table: PyObjectRef,
    /// `persistent_id` callable set on the instance, or `PY_NULL` when unset
    /// (a subclass may instead override the `persistent_id` method).
    w_pers_func: PyObjectRef,
}

/// Per-`dump` pickling context.  The identity memo maps an already-saved
/// object to its memo index.  pyre's incminimark nursery relocates live
/// objects, so the memo cannot key on a raw address: the memoized objects
/// live in a pinned Python `list` (`memo_slot`) which the GC walks, so the
/// stored references follow every move, and `index` maps the move-stable
/// `gc_identity_hash` to the list positions sharing that hash, resolved by
/// pointer identity against a freshly-read list element.  The memo index
/// (the PUT/GET argument) is the object's position in that list.
struct PickleCtx {
    proto: i64,
    bin: bool,
    /// Apply the `_compat_pickle` py3→py2 name remap at protocol < 3.
    fix_imports: bool,
    /// Shadow-stack slot of the memo `list`; re-read on every access so a
    /// relocation of the list itself is observed.
    memo_slot: usize,
    /// `gc_identity_hash(obj)` → memo indices sharing that hash.
    index: HashMap<usize, Vec<usize>>,
    /// `persistent_id` callable resolved off the pickler (subclass override
    /// or set attribute), or `PY_NULL` when not defined.
    pers_func: PyObjectRef,
    /// `buffer_callback` for proto-5 out-of-band buffers, or `None`/`PY_NULL`.
    buffer_callback: PyObjectRef,
    /// `fast` mode — when set, memoization is skipped (no PUT/GET, no
    /// recursion guard).
    fast: bool,
    /// Effective `dispatch_table` (the pickler's, else `copyreg.dispatch_table`)
    /// consulted by `type` for the reduce of an otherwise-unhandled object;
    /// `None`/`PY_NULL` when unavailable.
    dispatch_table: PyObjectRef,
    /// `reducer_override` callable (a subclass hook) consulted for every
    /// object, or `PY_NULL` when not defined.
    reducer_override: PyObjectRef,
}

impl PickleCtx {
    /// The memo `list`, re-read from its pinned slot (it may have moved).
    fn memo_list(&self) -> PyObjectRef {
        pyre_object::gc_roots::shadow_stack_get(self.memo_slot)
    }

    fn memo_get(&self, w_obj: PyObjectRef) -> Option<usize> {
        let h = pyre_object::gc_hook::gc_identity_hash(w_obj as usize);
        let list = self.memo_list();
        for &idx in self.index.get(&h)? {
            let memoized =
                unsafe { pyre_object::listobject::w_list_getitem(list, idx as i64) }.unwrap();
            if memoized == w_obj {
                return Some(idx);
            }
        }
        None
    }
}

/// `interp_pickle.py _Framer` — emits output as frames and streams them to the
/// file's `write` as they are produced. Bytes are appended to the active frame;
/// once a frame reaches `FRAME_SIZE_TARGET` it is committed (FRAME opcode +
/// 8-byte little-endian length + body when the body is at least
/// `FRAME_SIZE_MIN` bytes) into `pending` and `pending` is flushed to the file.
/// Large payloads bypass the frame and are written directly
/// (`write_large_bytes`). When framing is off (protocol < 4) the active frame
/// is `None` and bytes accumulate in `pending` until the final flush.
///
/// `push` / `extend_from_slice` mirror the `Vec<u8>` methods the save routines
/// call. They never touch Python, so the deep save tree stays GC-inert; the
/// only nursery-relocation points are the `commit_frame` at each `save`
/// boundary, `write_large_bytes`, and the end flush — and those callers pin the
/// objects they still need across the `file.write` (arbitrary Python).
///
/// `file_slot` is the shadow-stack slot of the destination file (pinned by the
/// caller for the whole dump). When it is `None` (the `dumps` path) nothing is
/// flushed: `pending` accumulates the entire pickle and the caller takes it.
struct Framer {
    current_frame: Option<Vec<u8>>,
    pending: Vec<u8>,
    file_slot: Option<usize>,
}

impl Framer {
    fn new(file_slot: Option<usize>) -> Self {
        Framer {
            current_frame: None,
            pending: Vec::new(),
            file_slot,
        }
    }

    /// `_Framer.write` (single byte).
    fn push(&mut self, byte: u8) {
        match &mut self.current_frame {
            Some(f) => f.push(byte),
            None => self.pending.push(byte),
        }
    }

    /// `_Framer.write` (slice).
    fn extend_from_slice(&mut self, data: &[u8]) {
        match &mut self.current_frame {
            Some(f) => f.extend_from_slice(data),
            None => self.pending.extend_from_slice(data),
        }
    }

    /// `_Framer.start_framing`.
    fn start_framing(&mut self) {
        self.current_frame = Some(Vec::new());
    }

    /// Write the buffered `pending` bytes to the file in a single `write`. A
    /// no-op when there is no file (`dumps`) or nothing is pending. The file is
    /// re-read from its pin because `write` (arbitrary Python) can relocate it.
    fn flush(&mut self) -> Result<(), PyError> {
        if let Some(slot) = self.file_slot {
            if !self.pending.is_empty() {
                let w_bytes = pyre_object::w_bytes_from_bytes(&self.pending);
                // `call_meth` resolves `write` via getattr, which allocates a
                // bound method and can relocate the freshly-built `w_bytes`;
                // pin it and pass the re-read root (`w_file` is pinned at `slot`).
                let _roots = pyre_object::gc_roots::push_roots();
                pyre_object::gc_roots::pin_root(w_bytes);
                let bytes_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let w_file = pyre_object::gc_roots::shadow_stack_get(slot);
                call_meth(
                    w_file,
                    "write",
                    &[pyre_object::gc_roots::shadow_stack_get(bytes_slot)],
                )?;
                self.pending.clear();
            }
        }
        Ok(())
    }

    /// `_Framer.end_framing` — commit any remaining frame and stop framing.
    fn end_framing(&mut self) -> Result<(), PyError> {
        if matches!(&self.current_frame, Some(f) if !f.is_empty()) {
            self.commit_frame(true)?;
        }
        self.current_frame = None;
        Ok(())
    }

    /// `_Framer.commit_frame` — commit the active frame into `pending` when it
    /// has reached the target size (or `force`), then flush `pending` to the
    /// file. Only ever called at a `save` boundary or `end_framing`.
    fn commit_frame(&mut self, force: bool) -> Result<(), PyError> {
        let commit = match &self.current_frame {
            Some(f) => f.len() >= FRAME_SIZE_TARGET || force,
            None => false,
        };
        if !commit {
            return Ok(());
        }
        let data = std::mem::take(self.current_frame.as_mut().unwrap());
        if data.len() >= FRAME_SIZE_MIN {
            self.pending.push(op::FRAME);
            self.pending
                .extend_from_slice(&(data.len() as u64).to_le_bytes());
        }
        self.pending.extend_from_slice(&data);
        self.flush()
    }

    /// `_Framer.write_large_bytes` — terminate the active frame, then write the
    /// header (with the committed frame) and the payload as separate `write`s,
    /// avoiding buffering the large payload. The payload is copied to
    /// GC-stable storage first because the `write` calls below run arbitrary
    /// Python and can relocate the (managed) source the slice borrows from.
    fn write_large_bytes(&mut self, header: &[u8], payload: &[u8]) -> Result<(), PyError> {
        let owned = payload.to_vec();
        if matches!(&self.current_frame, Some(f) if !f.is_empty()) {
            self.commit_frame(true)?;
        }
        self.pending.extend_from_slice(header);
        match self.file_slot {
            Some(slot) => {
                self.flush()?;
                let w_payload = pyre_object::w_bytes_from_bytes(&owned);
                // Pin the freshly-built payload across `call_meth`'s `write`
                // getattr (which can allocate and relocate it).
                let _roots = pyre_object::gc_roots::push_roots();
                pyre_object::gc_roots::pin_root(w_payload);
                let payload_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let w_file = pyre_object::gc_roots::shadow_stack_get(slot);
                call_meth(
                    w_file,
                    "write",
                    &[pyre_object::gc_roots::shadow_stack_get(payload_slot)],
                )?;
            }
            None => self.pending.extend_from_slice(&owned),
        }
        Ok(())
    }
}

#[crate::pyre_methods(doc = "Pickler(file, protocol=None) -> pickler writing to file.")]
impl W_Pickler {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_Pickler::allocate(W_Pickler {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_file: pyre_object::w_none(),
            proto: 0,
            bin: false,
            framing: false,
            fix_imports: true,
            buffer_callback: pyre_object::w_none(),
            w_memo: pyre_object::listobject::w_list_new(Vec::new()),
            fast: 0,
            w_dispatch_table: pyre_object::PY_NULL,
            w_pers_func: pyre_object::PY_NULL,
        })
    }

    fn __init__(
        &mut self,
        file: PyObjectRef,
        #[default(pyre_object::w_none())] protocol: PyObjectRef,
        #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
        #[default(pyre_object::w_none())] buffer_callback: PyObjectRef,
    ) -> Result<(), PyError> {
        // `fix_imports` gates the `_compat_pickle` py3→py2 name remap that the
        // protocol-< 3 save path would otherwise always apply.
        let proto = normalize_protocol(protocol)?;
        // `file must have a 'write' attribute` (interp_pickle.py:557).
        if crate::baseobjspace::findattr(file, "write").is_none() {
            return Err(PyError::type_error("file must have a 'write' attribute"));
        }
        if !unsafe { pyre_object::is_none(buffer_callback) } && proto < 5 {
            return Err(PyError::value_error("buffer_callback needs protocol >= 5"));
        }
        self.w_file = file;
        self.proto = proto;
        self.bin = proto >= 1;
        self.framing = proto >= 4;
        self.fix_imports = crate::baseobjspace::is_true(fix_imports)?;
        self.buffer_callback = buffer_callback;
        self.w_memo = pyre_object::listobject::w_list_new(Vec::new());
        self.fast = 0;
        self.w_dispatch_table = pyre_object::PY_NULL;
        self.w_pers_func = pyre_object::PY_NULL;
        Ok(())
    }

    /// `Pickler.clear_memo` — reset the memo so the next `dump` starts fresh.
    fn clear_memo(&mut self) {
        self.w_memo = pyre_object::listobject::w_list_new(Vec::new());
    }

    fn dump(&mut self, w_obj: PyObjectRef) -> Result<(), PyError> {
        // Read every field before any allocation can relocate `self`.
        let proto = self.proto;
        let bin = self.bin;
        let framing = self.framing;
        let fix_imports = self.fix_imports;
        let fast = self.fast;
        let w_dispatch_table = self.w_dispatch_table;
        let w_file = self.w_file;
        let buffer_callback = self.buffer_callback;
        let w_memo = self.w_memo;
        let self_ptr = self as *mut W_Pickler as PyObjectRef;

        // Pin everything that must survive the `persistent_id` /
        // `reducer_override` / `dispatch_table` lookups below: each runs Python
        // (a property / `__getattr__` / `copyreg` import) and can relocate
        // objects under the moving GC.
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_obj);
        let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_object::gc_roots::pin_root(w_file);
        let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_object::gc_roots::pin_root(w_memo);
        let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_object::gc_roots::pin_root(buffer_callback);
        let cb_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        // `persistent_id` resolves to a subclass method override or the
        // instance value set through the getter (its getter raises while
        // unset, so a base pickler resolves to `PY_NULL`); `reducer_override`
        // is a subclass hook only.  An explicit `persistent_id = None` is kept
        // as the hook: `dump` then calls `None(obj)` and raises `TypeError`,
        // matching `_pickle` (only deleting/leaving it unset disables it).
        // `findattr_result` propagates a hook property's own error instead of
        // panicking; each resolved hook is pinned before the next lookup.
        let pers_func = crate::baseobjspace::findattr_result(self_ptr, "persistent_id")?
            .unwrap_or(pyre_object::PY_NULL);
        pyre_object::gc_roots::pin_root(pers_func);
        let pers_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let reducer_override = crate::baseobjspace::findattr_result(self_ptr, "reducer_override")?
            .filter(|&f| !unsafe { pyre_object::is_none(f) })
            .unwrap_or(pyre_object::PY_NULL);
        pyre_object::gc_roots::pin_root(reducer_override);
        let reducer_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // `interp_pickle.py:686-690` — the internal `dispatch_table` field,
        // else a dynamically-resolved `dispatch_table` attribute (a subclass
        // class attr / property), else the global `copyreg.dispatch_table`.  An
        // explicit `dispatch_table = None` is honored as the table (later
        // subscripting `None[type]` raises `TypeError`); only an absent
        // attribute falls back to `copyreg`.
        let dispatch_table = if !w_dispatch_table.is_null() {
            w_dispatch_table
        } else if let Some(dt) = crate::baseobjspace::findattr_result(self_ptr, "dispatch_table")? {
            dt
        } else {
            copyreg_dispatch_table()
        };
        pyre_object::gc_roots::pin_root(dispatch_table);
        let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        // Re-read every pinned value: the lookups / `copyreg` import above may
        // have collected. `pickle_core` streams the frames to `w_file`.
        let w_obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let w_memo = pyre_object::gc_roots::shadow_stack_get(memo_slot);
        let w_file = pyre_object::gc_roots::shadow_stack_get(file_slot);
        let buffer_callback = pyre_object::gc_roots::shadow_stack_get(cb_slot);
        let pers_func = pyre_object::gc_roots::shadow_stack_get(pers_slot);
        let reducer_override = pyre_object::gc_roots::shadow_stack_get(reducer_slot);
        let dispatch_table = pyre_object::gc_roots::shadow_stack_get(dt_slot);
        pickle_core(
            w_obj,
            w_file,
            proto,
            bin,
            framing,
            fix_imports,
            pers_func,
            buffer_callback,
            w_memo,
            fast != 0,
            dispatch_table,
            reducer_override,
        )?;
        Ok(())
    }

    /// `Pickler.bin` — whether binary opcodes are in use (protocol >= 1).
    #[getter]
    fn bin(&self) -> i64 {
        self.bin as i64
    }

    /// `Pickler.fast` — when non-zero, memoization is skipped.
    #[getter]
    fn fast(&self) -> i64 {
        self.fast
    }

    #[setter]
    fn set_fast(&mut self, w_value: PyObjectRef) -> Result<(), PyError> {
        // `interp_pickle.py:1219-1223` stores `space.int_w(w_val)`.
        self.fast = crate::baseobjspace::int_w(w_value)?;
        Ok(())
    }

    /// `Pickler.dispatch_table` — the per-pickler type→reduce mapping. Unset by
    /// default (reading it raises `AttributeError`); when unset the dump path
    /// falls back to `copyreg.dispatch_table`.
    #[getter]
    fn dispatch_table(&self) -> Result<PyObjectRef, PyError> {
        if self.w_dispatch_table.is_null() {
            return Err(PyError::attribute_error(
                "'_pickle.Pickler' object has no attribute 'dispatch_table'".to_string(),
            ));
        }
        Ok(self.w_dispatch_table)
    }

    #[setter]
    fn set_dispatch_table(&mut self, w_value: PyObjectRef) {
        self.w_dispatch_table = w_value;
    }

    /// `Pickler.persistent_id` — the per-instance persistent-id callable. Once
    /// set, every object is offered to it before the type dispatch; a non-None
    /// result is saved as a persistent reference. A subclass may instead define
    /// a `persistent_id` method (resolved at dump time). Reading it while unset
    /// raises `AttributeError` (a readable no-op default is omitted: a shared
    /// callable would need a GC-stable singleton under the relocating nursery).
    #[getter]
    fn persistent_id(&self) -> Result<PyObjectRef, PyError> {
        if self.w_pers_func.is_null() {
            return Err(PyError::attribute_error(
                "'_pickle.Pickler' object has no attribute 'persistent_id'".to_string(),
            ));
        }
        Ok(self.w_pers_func)
    }

    #[setter]
    fn set_persistent_id(&mut self, w_value: PyObjectRef) {
        self.w_pers_func = w_value;
    }

    #[deleter("persistent_id")]
    fn del_persistent_id(&mut self) {
        self.w_pers_func = pyre_object::PY_NULL;
    }

    /// `Pickler.memo` — a fresh `PicklerMemoProxy` viewing this pickler's memo
    /// (CPython hands back a new proxy on each access).
    #[getter]
    fn memo(&self) -> PyObjectRef {
        let self_obj = self as *const W_Pickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        memo_proxy::type_object();
        let proxy = W_PicklerMemoProxy::allocate(W_PicklerMemoProxy {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_pickler: pyre_object::PY_NULL,
        });
        // `allocate` may have relocated the pickler; wire the (young) proxy to
        // its post-collection address.
        if let Some(px) = W_PicklerMemoProxy::from_obj(proxy) {
            px.w_pickler = pyre_object::gc_roots::shadow_stack_get(slot);
        }
        proxy
    }

    /// `Pickler.memo` setter — replace the memo from a `PicklerMemoProxy` or an
    /// `{id: (index, obj)}` mapping; the object is restored at its `index`
    /// position (the position-indexed memo list, from which the dump rebuilds
    /// the identity index). Any other type is a `TypeError`.
    #[setter]
    fn set_memo(&mut self, w_value: PyObjectRef) -> Result<(), PyError> {
        let self_obj = self as *mut W_Pickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self_obj);
        let self_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_dict = if W_PicklerMemoProxy::from_obj(w_value).is_some() {
            call_meth(w_value, "copy", &[])?
        } else if unsafe { pyre_object::is_dict(w_value) } {
            unsafe { pyre_object::dictmultiobject::w_dict_copy(w_value) }
        } else {
            return Err(PyError::type_error(format!(
                "'memo' attribute must be a PicklerMemoProxy object or dict, not {}",
                crate::baseobjspace::object_functionstr_type_name(w_value),
            )));
        };
        pyre_object::gc_roots::pin_root(w_dict);
        let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        // No GC between reading the items and `w_list_new` (which re-pins them).
        let items = unsafe {
            pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
                dict_slot,
            ))
        };
        let mut max_idx: i64 = -1;
        for (_, tup) in &items {
            if !(unsafe { pyre_object::is_tuple(*tup) }
                && unsafe { pyre_object::tupleobject::w_tuple_len(*tup) } == 2)
            {
                return Err(PyError::type_error("'memo' values must be 2-item tuples"));
            }
            let idx = unsafe { pyre_object::tupleobject::w_tuple_getitem(*tup, 0) }.unwrap();
            let i = crate::baseobjspace::int_w(idx)?;
            if i < 0 {
                // The memo is a position-indexed list, so a negative slot index
                // has no representation (and would emit a malformed back-reference).
                return Err(PyError::value_error("memo index must be non-negative"));
            }
            if i > max_idx {
                max_idx = i;
            }
        }
        // Gaps in a sparse memo hold `PY_NULL`, not `None`: a `None` slot would
        // be indistinguishable from a genuinely memoized `None`, which `copy()`
        // must still expose. `w_list_new_object` keeps the `PY_NULL` holes by
        // pointer (Object strategy), mirroring the unpickler memo.
        let mut slots: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; (max_idx + 1) as usize];
        for (_, tup) in &items {
            let idx = unsafe { pyre_object::tupleobject::w_tuple_getitem(*tup, 0) }.unwrap();
            let i = crate::baseobjspace::int_w(idx)? as usize;
            slots[i] = unsafe { pyre_object::tupleobject::w_tuple_getitem(*tup, 1) }.unwrap();
        }
        let list = pyre_object::listobject::w_list_new_object(slots);
        let me =
            unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(self_slot) as *mut W_Pickler) };
        me.w_memo = list;
        Ok(())
    }

    /// `Pickler.memo` is not deletable.
    #[deleter("memo")]
    fn del_memo(&self) -> Result<(), PyError> {
        Err(PyError::type_error("attribute deletion is not supported"))
    }
}

/// `interp_pickle.py PicklerMemoProxy` — a live view of a pickler's identity
/// memo. `copy` snapshots it as `{id(obj): (memo_index, obj)}`; `clear` empties
/// it. The pickler stores the memo as a position-indexed Python `list` (see
/// `PickleCtx`), so `copy` derives the `id` keys from the listed objects.
///
/// Held in its own module so `#[pyre_methods]` emits a `type_object()` that
/// does not clash with `W_Pickler`'s (each impl emits a module-scoped one).
pub use memo_proxy::W_PicklerMemoProxy;

mod memo_proxy {
    use super::*;

    #[crate::pyre_class("_pickle.PicklerMemoProxy")]
    pub struct W_PicklerMemoProxy {
        pub(super) w_pickler: PyObjectRef,
    }

    #[crate::pyre_methods(doc = "Proxy for a Pickler's memo.")]
    impl W_PicklerMemoProxy {
        /// `PicklerMemoProxy.copy` — `{id(obj): (memo_index, obj)}`.
        fn copy(&self) -> Result<PyObjectRef, PyError> {
            let w_memo = unsafe { &*(self.w_pickler as *const W_Pickler) }.w_memo;
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(w_memo);
            let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let result = pyre_object::dictmultiobject::w_dict_new();
            pyre_object::gc_roots::pin_root(result);
            let res_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let n = unsafe {
                pyre_object::listobject::w_list_len(pyre_object::gc_roots::shadow_stack_get(
                    memo_slot,
                ))
            };
            for i in 0..n {
                let _it = pyre_object::gc_roots::push_roots();
                let obj = unsafe {
                    pyre_object::listobject::w_list_getitem(
                        pyre_object::gc_roots::shadow_stack_get(memo_slot),
                        i as i64,
                    )
                }
                .unwrap();
                // Gap slots from a sparse `set_memo` hold the `PY_NULL`
                // placeholder, not real memo entries; skip them so `copy()`
                // exposes only the memoized objects (an id-keyed, gap-free
                // snapshot) while still surfacing a genuinely memoized `None`.
                if obj.is_null() {
                    continue;
                }
                // `(index, obj)` — `w_tuple_new` pins its inputs across the malloc.
                let tup = pyre_object::tupleobject::w_tuple_new(vec![
                    pyre_object::w_int_new(i as i64),
                    obj,
                ]);
                pyre_object::gc_roots::pin_root(tup);
                let tup_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                // id(obj) read from the (relocated) tuple element.
                let cur_obj = unsafe {
                    pyre_object::tupleobject::w_tuple_getitem(
                        pyre_object::gc_roots::shadow_stack_get(tup_slot),
                        1,
                    )
                }
                .unwrap();
                let key = pyre_object::w_int_new(cur_obj as i64);
                pyre_object::gc_roots::pin_root(key);
                let key_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                // The IndexMap insert allocates via `std::alloc` (no collection), so
                // the freshly read addresses stay valid through it.
                unsafe {
                    pyre_object::dictmultiobject::w_dict_store(
                        pyre_object::gc_roots::shadow_stack_get(res_slot),
                        pyre_object::gc_roots::shadow_stack_get(key_slot),
                        pyre_object::gc_roots::shadow_stack_get(tup_slot),
                    );
                }
            }
            Ok(pyre_object::gc_roots::shadow_stack_get(res_slot))
        }

        /// `PicklerMemoProxy.clear` — empty the pickler's memo.
        fn clear(&self) {
            let w_pickler = self.w_pickler;
            let _roots = pyre_object::gc_roots::push_roots();
            pyre_object::gc_roots::pin_root(w_pickler);
            let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let empty = pyre_object::listobject::w_list_new(Vec::new());
            let p =
                unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Pickler) };
            p.w_memo = empty;
        }
    }
}

/// `copyreg.dispatch_table`, or `PY_NULL` when `copyreg` is unavailable.
pub(crate) fn copyreg_dispatch_table() -> PyObjectRef {
    match import_module("copyreg") {
        Ok(copyreg) => crate::baseobjspace::getattr_str(copyreg, "dispatch_table")
            .unwrap_or(pyre_object::PY_NULL),
        Err(_) => pyre_object::PY_NULL,
    }
}

/// `buffer_callback needs protocol >= 5` — reject a non-None callback under
/// an earlier protocol (interp_pickle.py:1818).
pub(crate) fn check_buffer_callback(
    buffer_callback: PyObjectRef,
    proto: i64,
) -> Result<(), PyError> {
    if !unsafe { pyre_object::is_none(buffer_callback) } && proto < 5 {
        return Err(PyError::value_error("buffer_callback needs protocol >= 5"));
    }
    Ok(())
}

/// `interp_pickle.py W_Pickler.__init__` protocol resolution: `None` →
/// `DEFAULT_PROTOCOL`, a negative value → `HIGHEST_PROTOCOL`, and anything
/// above `HIGHEST_PROTOCOL` is rejected.
pub(crate) fn normalize_protocol(w_protocol: PyObjectRef) -> Result<i64, PyError> {
    if unsafe { pyre_object::is_none(w_protocol) } {
        return Ok(DEFAULT_PROTOCOL);
    }
    let p = crate::baseobjspace::int_w(w_protocol)?;
    if p < 0 {
        Ok(HIGHEST_PROTOCOL)
    } else if p > HIGHEST_PROTOCOL {
        Err(PyError::value_error("pickle protocol must be <= 5"))
    } else {
        Ok(p)
    }
}

/// Pickle `w_obj`. When `w_file` is a file object the frames are streamed to
/// its `write` as they are produced and `PY_NULL` is returned; when `w_file` is
/// `PY_NULL` (the `dumps` path) the whole pickle is accumulated and returned as
/// a `bytes`. Shared by `W_Pickler.dump` and the module-level `dump` / `dumps`.
/// `pers_func` is the `persistent_id` callable or `PY_NULL`. PROTO is written
/// before framing begins (outside the frame); STOP is written while framing is
/// active (inside the last frame).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pickle_core(
    w_obj: PyObjectRef,
    w_file: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
    fix_imports: bool,
    pers_func: PyObjectRef,
    buffer_callback: PyObjectRef,
    w_memo: PyObjectRef,
    fast: bool,
    dispatch_table: PyObjectRef,
    reducer_override: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    if !pers_func.is_null() {
        pyre_object::gc_roots::pin_root(pers_func);
    }
    if !buffer_callback.is_null() && !unsafe { pyre_object::is_none(buffer_callback) } {
        pyre_object::gc_roots::pin_root(buffer_callback);
    }
    if !dispatch_table.is_null() && !unsafe { pyre_object::is_none(dispatch_table) } {
        pyre_object::gc_roots::pin_root(dispatch_table);
    }
    if !reducer_override.is_null() {
        pyre_object::gc_roots::pin_root(reducer_override);
    }
    // Pin the memo list and index its existing entries (a reused `Pickler`
    // carries memo state across `dump` calls until `clear_memo`).
    pyre_object::gc_roots::pin_root(w_memo);
    let memo_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let mut index: HashMap<usize, Vec<usize>> = HashMap::new();
    let n = unsafe { pyre_object::listobject::w_list_len(w_memo) };
    for i in 0..n {
        let o = unsafe { pyre_object::listobject::w_list_getitem(w_memo, i as i64) }.unwrap();
        // Gap slots from a sparse `set_memo` hold `PY_NULL`; they are not real
        // memo entries, so leave them out of the identity index.
        if o.is_null() {
            continue;
        }
        index
            .entry(pyre_object::gc_hook::gc_identity_hash(o as usize))
            .or_default()
            .push(i);
    }

    let mut ctx = PickleCtx {
        proto,
        bin,
        fix_imports,
        memo_slot,
        index,
        pers_func,
        buffer_callback,
        fast,
        dispatch_table,
        reducer_override,
    };
    // Pin the destination file so the framer can re-read it across each
    // streaming `write` (which can relocate it). `None` selects accumulate mode.
    let file_slot = if w_file.is_null() {
        None
    } else {
        pyre_object::gc_roots::pin_root(w_file);
        Some(pyre_object::gc_roots::shadow_stack_len() - 1)
    };
    let mut fr = Framer::new(file_slot);
    if proto >= 2 {
        fr.push(op::PROTO);
        fr.push(proto as u8);
    }
    if framing {
        fr.start_framing();
    }
    save(&mut ctx, &mut fr, w_obj)?;
    fr.push(op::STOP);
    fr.end_framing()?;
    // Flush the unframed residual (the whole pickle for protocol < 4, a no-op
    // after the last frame committed for protocol >= 4); a no-op without a file.
    fr.flush()?;

    if file_slot.is_some() {
        Ok(pyre_object::PY_NULL)
    } else {
        Ok(pyre_object::w_bytes_from_bytes(&fr.pending))
    }
}

/// `interp_pickle.py W_Pickler.save` with the persistent-id hook: every
/// object is first offered to `persistent_id`; a non-None result is saved
/// as a persistent reference instead of by value.
fn save(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // A frame boundary can only fall at the start of a `save`, never inside an
    // object; commit the active frame once it has grown past the target.
    // Committing streams to the file (arbitrary Python), and the `persistent_id`
    // hook below is too, so pin `w_obj` across both and read it from the pin.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    buf.commit_frame(false)?;
    if !ctx.pers_func.is_null() {
        let w_pid = call_fn(
            ctx.pers_func,
            &[pyre_object::gc_roots::shadow_stack_get(slot)],
        )?;
        if !unsafe { pyre_object::is_none(w_pid) } {
            return save_pers(ctx, buf, w_pid);
        }
        return save_object(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    }
    save_object(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot))
}

/// `interp_pickle.py save_pers` — emit a persistent reference. The
/// persistent id itself is saved by value (skipping the persistent-id
/// hook) in binary protocols, or as an ASCII line in protocol 0.
fn save_pers(ctx: &mut PickleCtx, buf: &mut Framer, w_pid: PyObjectRef) -> Result<(), PyError> {
    if ctx.bin {
        save_object(ctx, buf, w_pid)?;
        buf.push(op::BINPERSID);
        return Ok(());
    }
    let w_str = if unsafe { pyre_object::is_str(w_pid) } {
        w_pid
    } else {
        let str_fn = crate::module::_pickle::lookup_builtin("str")
            .ok_or_else(|| pickling_error("str builtin unavailable"))?;
        call_fn(str_fn, &[w_pid])?
    };
    let s = unsafe { pyre_object::strobject::w_str_get_value(w_str) };
    if !s.is_ascii() {
        return Err(pickling_error(
            "persistent IDs in protocol 0 must be ASCII strings",
        ));
    }
    buf.push(op::PERSID);
    buf.extend_from_slice(s.as_bytes());
    buf.push(b'\n');
    Ok(())
}

/// `interp_pickle.py W_Pickler.save` body after the persistent-id hook: the
/// identity memo (atoms are never memoized), then the type dispatch in
/// [`dispatch_save`].  The memo is consulted before `reducer_override` (a
/// repeated reference becomes a GET, matching CPython 3.14: a second
/// occurrence of the same object never re-enters the hook).
fn save_object(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // Only EXACT atoms skip the memo; an `int`/`float` subclass is a regular
    // object (memoized, saved through the reduce protocol).
    let is_atom = unsafe {
        pyre_object::is_none(w_obj)
            || pyre_object::is_bool(w_obj)
            || pyre_object::is_exact_type(w_obj, &pyre_object::INT_TYPE)
            || pyre_object::is_exact_type(w_obj, &pyre_object::FLOAT_TYPE)
    };
    // Identity memo — a repeated reference becomes a GET back-reference.
    if !is_atom {
        if let Some(idx) = ctx.memo_get(w_obj) {
            write_get(ctx, buf, idx);
            return Ok(());
        }
    }
    dispatch_save(ctx, buf, w_obj)
}

/// Exact-type dispatch via the `is_*` predicates (bool is checked before int
/// because a bool is not an int here, and `is_int_or_long` also covers big
/// integers). The memo has already been consulted by [`save_object`]. Classes
/// and functions are saved by reference; anything else goes through the
/// `dispatch_table` then the reduce protocol.
fn dispatch_save(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // Atoms — never memoized.
    if unsafe { pyre_object::is_none(w_obj) } {
        buf.push(op::NONE);
        return Ok(());
    }
    if unsafe { pyre_object::is_bool(w_obj) } {
        return save_bool(ctx, buf, w_obj);
    }
    // Built-in dispatch is by EXACT type (`type(obj) is X`): a subclass of a
    // built-in carries the same `ob_type`/layout but a retagged `w_class`, and
    // must fall through to `save_global_or_reduce` so its subclass identity is
    // preserved.  `is_exact_type` is correct for the specialised arity-2 tuples
    // (distinct `ob_type`, canonical `tuple` `w_class`) where the `ob_type`
    // predicate `is_exact_builtin_instance` is not.
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::INT_TYPE) } {
        save_long(ctx, buf, w_obj)?;
        return Ok(());
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::FLOAT_TYPE) } {
        save_float(ctx, buf, w_obj)?;
        return Ok(());
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::bytesobject::BYTES_TYPE) } {
        return save_bytes(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::STR_TYPE) } {
        return save_str(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::DICT_TYPE) } {
        return save_dict(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::setobject::SET_TYPE) } {
        return save_set(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::setobject::FROZENSET_TYPE) } {
        return save_frozenset(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::LIST_TYPE) } {
        return save_list(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::TUPLE_TYPE) } {
        return save_tuple(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_exact_type(w_obj, &pyre_object::bytearrayobject::BYTEARRAY_TYPE) } {
        return save_bytearray(ctx, buf, w_obj);
    }
    if crate::module::__pypy__::W_PickleBuffer::from_obj(w_obj).is_some() {
        return save_picklebuffer(ctx, buf, w_obj);
    }

    // `reducer_override` (a Pickler subclass hook) is offered every object not
    // dispatched above as a built-in atom/container, before save_global and
    // the reduce protocol. A result other than NotImplemented replaces the
    // default reduction. (interp_pickle.py:619-625 calls it earlier; CPython
    // 3.14 — the behaviour target — dispatches built-in types first, so the
    // hook never sees a list/dict/str and a repeated object hits the memo.)
    if !ctx.reducer_override.is_null() {
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        let w_rv = call_fn(
            ctx.reducer_override,
            &[pyre_object::gc_roots::shadow_stack_get(slot)],
        )?;
        if !unsafe { pyre_object::is_not_implemented(w_rv) } {
            return save_reduce_value(
                ctx,
                buf,
                pyre_object::gc_roots::shadow_stack_get(slot),
                w_rv,
            );
        }
        return save_global_or_reduce(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    }
    save_global_or_reduce(ctx, buf, w_obj)
}

/// The tail of [`dispatch_save`] after built-in dispatch and the
/// `reducer_override` hook: classes/functions by reference, then the
/// per-pickler `dispatch_table`, then the `__reduce_ex__` / `__reduce__`
/// protocol.
fn save_global_or_reduce(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    // Classes and functions are saved by reference.
    if unsafe { pyre_object::typeobject::is_type(w_obj) }
        || unsafe { crate::function::is_function(w_obj) }
    {
        return save_global(ctx, buf, w_obj, None);
    }

    // A `dispatch_table` reduce function registered for this exact type takes
    // precedence over `__reduce_ex__`.
    if let Some(w_rv) = dispatch_table_reduce(ctx, w_obj)? {
        return save_reduce_value(ctx, buf, w_obj, w_rv);
    }

    // Everything else goes through the reduce protocol.
    let w_rv = match crate::baseobjspace::findattr(w_obj, "__reduce_ex__") {
        Some(reduce_ex) => call_fn(reduce_ex, &[pyre_object::w_int_new(ctx.proto)])?,
        None => match crate::baseobjspace::findattr(w_obj, "__reduce__") {
            Some(reduce) => call_fn(reduce, &[])?,
            None => return Err(pickling_error("Can't pickle object: no __reduce_ex__")),
        },
    };
    save_reduce_value(ctx, buf, w_obj, w_rv)
}

/// Save the result of a reduce hook (`reducer_override` / `dispatch_table` /
/// `__reduce_ex__`): a `str` saves the object by global reference, a 2-to-6
/// element tuple drives `save_reduce`.
fn save_reduce_value(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
    w_rv: PyObjectRef,
) -> Result<(), PyError> {
    if unsafe { pyre_object::is_str(w_rv) } {
        return save_global(ctx, buf, w_obj, Some(w_rv));
    }
    if unsafe { pyre_object::is_tuple(w_rv) } {
        let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_rv) };
        if !(2..=6).contains(&n) {
            return Err(pickling_error(
                "Tuple returned by __reduce__ must have two to six elements",
            ));
        }
        let rv: Vec<PyObjectRef> = (0..n)
            .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_rv, i as i64).unwrap() })
            .collect();
        return save_reduce(ctx, buf, &rv, Some(w_obj));
    }
    Err(pickling_error("__reduce__ must return string or tuple"))
}

/// Look up `type(w_obj)` in the effective `dispatch_table` and, if registered,
/// call the reduce function. Returns its result, or `None` when no table or
/// no matching entry applies. The `copyreg.dispatch_table` fallback is a dict;
/// a user-set `dispatch_table` may be any mapping, so a non-dict is consulted
/// via `__getitem__` with `KeyError` meaning "no entry".
fn dispatch_table_reduce(
    ctx: &PickleCtx,
    w_obj: PyObjectRef,
) -> Result<Option<PyObjectRef>, PyError> {
    let dt = ctx.dispatch_table;
    if dt.is_null() {
        return Ok(None);
    }
    if unsafe { pyre_object::is_none(dt) } {
        // An explicit `dispatch_table = None` is still consulted by subscript;
        // `None[type(obj)]` raises, so surface that rather than silently
        // falling back to `__reduce_ex__`.
        return Err(PyError::type_error(
            "'NoneType' object is not subscriptable",
        ));
    }
    // The common case is an empty `copyreg.dispatch_table`; skip the lookup.
    if unsafe { pyre_object::is_dict(dt) }
        && unsafe { pyre_object::dictmultiobject::w_dict_len(dt) } == 0
    {
        return Ok(None);
    }
    let type_fn = crate::module::_pickle::lookup_builtin("type")
        .ok_or_else(|| pickling_error("type builtin unavailable"))?;
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(dt);
    let dt_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let w_type = call_fn(
        type_fn,
        &[pyre_object::gc_roots::shadow_stack_get(obj_slot)],
    )?;
    pyre_object::gc_roots::pin_root(w_type);
    let type_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let dt = pyre_object::gc_roots::shadow_stack_get(dt_slot);
    let w_type = pyre_object::gc_roots::shadow_stack_get(type_slot);
    let reduce_fn = if unsafe { pyre_object::is_dict(dt) } {
        unsafe { pyre_object::w_dict_lookup(dt, w_type) }
    } else {
        match crate::baseobjspace::getitem(dt, w_type) {
            Ok(reduce_fn) => Some(reduce_fn),
            Err(e) if e.kind == crate::PyErrorKind::KeyError => None,
            Err(e) => return Err(e),
        }
    };
    match reduce_fn {
        Some(reduce_fn) => Ok(Some(call_fn(
            reduce_fn,
            &[pyre_object::gc_roots::shadow_stack_get(obj_slot)],
        )?)),
        None => Ok(None),
    }
}

fn save_bool(ctx: &PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let truthy = crate::baseobjspace::is_true(w_obj)?;
    if ctx.proto >= 2 {
        buf.push(if truthy { op::NEWTRUE } else { op::NEWFALSE });
    } else {
        // I00\n / I01\n
        buf.extend_from_slice(if truthy { b"I01\n" } else { b"I00\n" });
    }
    Ok(())
}

fn save_long(ctx: &PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let small = crate::baseobjspace::int_w(w_obj).ok();
    let to_big = |v: Option<i64>| match v {
        Some(v) => BigInt::from(v),
        None => unsafe { crate::builtins::obj_to_bigint(w_obj) },
    };
    if ctx.bin {
        if let Some(v) = small {
            if v >= 0 {
                if v <= 0xff {
                    buf.push(op::BININT1);
                    buf.push(v as u8);
                    return Ok(());
                }
                if v <= 0xffff {
                    buf.push(op::BININT2);
                    buf.extend_from_slice(&(v as u16).to_le_bytes());
                    return Ok(());
                }
            }
            if (-0x8000_0000..=0x7fff_ffff).contains(&v) {
                buf.push(op::BININT);
                buf.extend_from_slice(&(v as i32).to_le_bytes());
                return Ok(());
            }
        }
    }
    if ctx.proto >= 2 {
        write_long(buf, &encode_long(&to_big(small)));
        return Ok(());
    }
    // protocol 0 / 1 text: INT for a signed 4-byte value, else LONG.
    if let Some(v) = small {
        if (-0x8000_0000..=0x7fff_ffff).contains(&v) {
            buf.push(op::INT);
            buf.extend_from_slice(v.to_string().as_bytes());
            buf.push(b'\n');
            return Ok(());
        }
    }
    buf.push(op::LONG);
    buf.extend_from_slice(to_big(small).to_string().as_bytes());
    buf.extend_from_slice(b"L\n");
    Ok(())
}

fn write_long(buf: &mut Framer, enc: &[u8]) {
    let n = enc.len();
    if n < 256 {
        buf.push(op::LONG1);
        buf.push(n as u8);
    } else {
        buf.push(op::LONG4);
        buf.extend_from_slice(&(n as i32).to_le_bytes());
    }
    buf.extend_from_slice(enc);
}

fn save_float(ctx: &PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.bin {
        let f = crate::baseobjspace::float_w(w_obj)?;
        // BINFLOAT — 8-byte big-endian IEEE 754.
        buf.push(op::BINFLOAT);
        buf.extend_from_slice(&f.to_be_bytes());
    } else {
        // proto 0: FLOAT + repr(obj) + '\n' (shortest round-trip text).
        let f = crate::baseobjspace::float_w(w_obj)?;
        buf.push(op::FLOAT);
        buf.extend_from_slice(crate::display::format_float_repr(f).as_bytes());
        buf.push(b'\n');
    }
    Ok(())
}

fn save_bytes(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // proto < 3 emits a `codecs.encode(s, 'latin1')` / `bytes()` reduce
    // instead of a BINBYTES opcode (interp_pickle.py:1349).
    if ctx.proto < 3 {
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) };
        if data.is_empty() {
            let w_bytes = crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
            let w_args = pyre_object::tupleobject::w_tuple_new(Vec::new());
            return save_reduce(ctx, buf, &[w_bytes, w_args], Some(w_obj));
        }
        let codecs = import_module("codecs")?;
        let w_encode = crate::baseobjspace::getattr_str(codecs, "encode")?;
        let w_decoded = call_meth(w_obj, "decode", &[pyre_object::w_str_new("latin1")])?;
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![
            w_decoded,
            pyre_object::w_str_new("latin1"),
        ]);
        return save_reduce(ctx, buf, &[w_encode, w_args], Some(w_obj));
    }
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) };
    let n = data.len();
    // A large payload streams via `file.write` (arbitrary Python); pin `w_obj`
    // so the trailing `memoize` reads it at its post-write address.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if n <= 0xff {
        buf.push(op::SHORT_BINBYTES);
        buf.push(n as u8);
        buf.extend_from_slice(data);
    } else if n > 0xffff_ffff && ctx.proto >= 4 {
        let mut header = vec![op::BINBYTES8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BINBYTES];
        header.extend_from_slice(&(n as u32).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else {
        buf.push(op::BINBYTES);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(data);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

fn save_str(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    // A large payload streams via `file.write` (arbitrary Python); pin `w_obj`
    // so the trailing `memoize` reads it at its post-write address.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if ctx.bin {
        let s = unsafe { pyre_object::strobject::w_str_get_value(w_obj) };
        let data = s.as_bytes();
        let n = data.len();
        if n <= 0xff && ctx.proto >= 4 {
            buf.push(op::SHORT_BINUNICODE);
            buf.push(n as u8);
            buf.extend_from_slice(data);
        } else if n > 0xffff_ffff && ctx.proto >= 4 {
            let mut header = vec![op::BINUNICODE8];
            header.extend_from_slice(&(n as u64).to_le_bytes());
            buf.write_large_bytes(&header, data)?;
        } else if n >= FRAME_SIZE_TARGET {
            let mut header = vec![op::BINUNICODE];
            header.extend_from_slice(&(n as u32).to_le_bytes());
            buf.write_large_bytes(&header, data)?;
        } else {
            buf.push(op::BINUNICODE);
            buf.extend_from_slice(&(n as u32).to_le_bytes());
            buf.extend_from_slice(data);
        }
    } else {
        // proto 0: UNICODE + raw-unicode-escape. The codec leaves
        // backslash / NUL / newline / CR / EOF-on-DOS literal, so escape
        // those first; the load side reverses with raw-unicode-escape.
        let mut w_tmp = w_obj;
        for (from, to) in [
            ("\\", "\\u005c"),
            ("\0", "\\u0000"),
            ("\n", "\\u000a"),
            ("\r", "\\u000d"),
            ("\u{1a}", "\\u001a"),
        ] {
            w_tmp = call_meth(
                w_tmp,
                "replace",
                &[pyre_object::w_str_new(from), pyre_object::w_str_new(to)],
            )?;
        }
        let w_enc = call_meth(
            w_tmp,
            "encode",
            &[pyre_object::w_str_new("raw-unicode-escape")],
        )?;
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_enc) };
        buf.push(op::UNICODE);
        buf.extend_from_slice(data);
        buf.push(b'\n');
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

/// `interp_pickle.py save_tuple`.
fn save_tuple(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_obj) };
    if n == 0 {
        if ctx.bin {
            buf.push(op::EMPTY_TUPLE);
        } else {
            buf.push(op::MARK);
            buf.push(op::TUPLE);
        }
        return Ok(());
    }

    // Pin the tuple; a recursive save below can relocate the elements, so
    // re-read each one (and the tuple itself for the memo) from the
    // GC-walked tuple right before it is used.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let elem = |i: usize| unsafe {
        pyre_object::tupleobject::w_tuple_getitem(
            pyre_object::gc_roots::shadow_stack_get(slot),
            i as i64,
        )
        .unwrap()
    };

    if n <= 3 && ctx.proto >= 2 {
        for i in 0..n {
            save(ctx, buf, elem(i))?;
        }
        // Subtle: saving the elements may have memoized this very tuple
        // (a recursive tuple). If so, discard the elements and GET it.
        if let Some(idx) = ctx.memo_get(pyre_object::gc_roots::shadow_stack_get(slot)) {
            for _ in 0..n {
                buf.push(op::POP);
            }
            write_get(ctx, buf, idx);
        } else {
            buf.push(op::TUPLESIZE2CODE[n]);
            memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
        }
        return Ok(());
    }

    buf.push(op::MARK);
    for i in 0..n {
        save(ctx, buf, elem(i))?;
    }
    if let Some(idx) = ctx.memo_get(pyre_object::gc_roots::shadow_stack_get(slot)) {
        // Recursive tuple: throw away the stack contents and GET it.
        if ctx.bin {
            buf.push(op::POP_MARK);
        } else {
            for _ in 0..(n + 1) {
                buf.push(op::POP);
            }
        }
        write_get(ctx, buf, idx);
        return Ok(());
    }
    buf.push(op::TUPLE);
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

/// `interp_pickle.py save_list`. The PyPy ascii/bytes-list fast paths are
/// gated on `pypy_extensions` (off here) so the generic path is used,
/// matching CPython's wire format.
fn save_list(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // The list itself is a GC-walked Python `list`; pin it and append by
    // re-reading each element, so a relocation during a recursive save is
    // observed instead of dereferencing a stale snapshot.
    pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if ctx.bin {
        buf.push(op::EMPTY_LIST);
    } else {
        buf.push(op::MARK);
        buf.push(op::LIST);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    batch_appends(ctx, buf, slot)
}

/// `interp_pickle.py save_dict`.
fn save_dict(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // Pin the dict (so `memoize` sees its current address) and, since a dict
    // has no stable index access, flatten its items into a pinned
    // `[k0, v0, …]` Python list (GC-walked), re-read per save.
    pyre_object::gc_roots::pin_root(w_obj);
    let dict_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let items = unsafe {
        pyre_object::dictmultiobject::w_dict_items(pyre_object::gc_roots::shadow_stack_get(
            dict_slot,
        ))
    };
    let mut flat = Vec::with_capacity(items.len() * 2);
    for (k, v) in items {
        flat.push(k);
        flat.push(v);
    }
    let items_slot = pin_items(flat);
    if ctx.bin {
        buf.push(op::EMPTY_DICT);
    } else {
        buf.push(op::MARK);
        buf.push(op::DICT);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(dict_slot));
    batch_setitems(ctx, buf, items_slot)
}

/// `interp_pickle.py save_set`. Sets are unordered, so the wire bytes are
/// not byte-identical to CPython, but the encoding round-trips. The
/// protocol < 4 reduce fallback arrives with `save_reduce`.
fn save_set(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.proto < 4 {
        // save_reduce(set, (list(obj),)).
        let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
        let w_list = pyre_object::listobject::w_list_new(items);
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![w_list]);
        let w_set_type = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
        return save_reduce(ctx, buf, &[w_set_type, w_args], Some(w_obj));
    }
    buf.push(op::EMPTY_SET);
    // Pin the set so `memoize` records its current address, then snapshot its
    // members into a pinned Python `list` re-read per save (a recursive save
    // can relocate them).
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let set_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(set_slot));

    let items = unsafe {
        pyre_object::setobject::w_set_items(pyre_object::gc_roots::shadow_stack_get(set_slot))
    };
    let slot = pin_items(items);
    let length = pinned_len(slot);
    if length == 0 {
        return Ok(());
    }
    buf.push(op::MARK);
    save(ctx, buf, pinned_get(slot, 0))?;
    let mut i = 1;
    while i + 1 < length {
        if i % BATCHSIZE == 0 {
            buf.push(op::ADDITEMS);
            buf.push(op::MARK);
        }
        save(ctx, buf, pinned_get(slot, i))?;
        i += 1;
    }
    if length > 1 {
        save(ctx, buf, pinned_get(slot, length - 1))?;
    }
    buf.push(op::ADDITEMS);
    Ok(())
}

/// `interp_pickle.py save_frozenset`. Protocol < 4 reduces to
/// `frozenset(list(obj))`; protocol >= 4 uses the FROZENSET opcode.
/// Unordered, so not byte-identical to CPython.
fn save_frozenset(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 4 {
        // save_reduce(frozenset, (list(obj),)).
        let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
        let w_list = pyre_object::listobject::w_list_new(items);
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![w_list]);
        let w_frozenset_type =
            crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE);
        return save_reduce(ctx, buf, &[w_frozenset_type, w_args], Some(w_obj));
    }
    // Pin the frozenset and snapshot its members into a pinned Python `list`
    // re-read per save (a recursive save can relocate them); the frozenset
    // itself is re-read for the memo check after the saves.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let fs_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let items = unsafe {
        pyre_object::setobject::w_set_items(pyre_object::gc_roots::shadow_stack_get(fs_slot))
    };
    let slot = pin_items(items);
    buf.push(op::MARK);
    let n = pinned_len(slot);
    for i in 0..n {
        save(ctx, buf, pinned_get(slot, i))?;
    }
    if let Some(idx) = ctx.memo_get(pyre_object::gc_roots::shadow_stack_get(fs_slot)) {
        buf.push(op::POP);
        write_get(ctx, buf, idx);
    } else {
        buf.push(op::FROZENSET);
        memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(fs_slot));
    }
    Ok(())
}

/// `interp_pickle.py save_bytearray` (proto >= 5 raw form; lower protocols
/// reach the generic reduce path).
fn save_bytearray(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 5 {
        // save_reduce(bytearray, ()) for empty, else save_reduce(bytearray, (bytes,)).
        let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(w_obj) };
        let w_bytearray_type =
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE);
        let w_args = if data.is_empty() {
            pyre_object::tupleobject::w_tuple_new(Vec::new())
        } else {
            let w_bytes = pyre_object::w_bytes_from_bytes(data);
            pyre_object::tupleobject::w_tuple_new(vec![w_bytes])
        };
        return save_reduce(ctx, buf, &[w_bytearray_type, w_args], Some(w_obj));
    }
    let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(w_obj) };
    let n = data.len();
    // A large payload streams via `file.write` (arbitrary Python); pin `w_obj`
    // so the trailing `memoize` reads it at its post-write address.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BYTEARRAY8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else {
        buf.push(op::BYTEARRAY8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    Ok(())
}

/// `interp_pickle.py save_picklebuffer` — serialize a `PickleBuffer`. With
/// no `buffer_callback`, or a callback returning a true value, the contents
/// are written in-band (BINBYTES for a read-only buffer, BYTEARRAY8 for a
/// mutable one). A callback returning a false value writes the data
/// out-of-band: NEXT_BUFFER, plus READONLY_BUFFER for a read-only buffer.
fn save_picklebuffer(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 5 {
        return Err(pickling_error(
            "PickleBuffer can only pickled with protocol >= 5",
        ));
    }
    // Read the wrapped object out of the buffer, then drop the borrow before
    // any allocation (the callback below) can relocate the wrapper.
    let wrapped = {
        let pb = crate::module::__pypy__::W_PickleBuffer::from_obj(w_obj)
            .ok_or_else(|| pickling_error("save_picklebuffer: not a PickleBuffer"))?;
        pb.wrapped()
    };
    if unsafe { pyre_object::is_none(wrapped) } {
        return Err(pickling_error(
            "PickleBuffer can not be pickled after release",
        ));
    }
    let (data, readonly) = crate::module::__pypy__::pickle_buffer::buffer_view(wrapped)?;
    let mut in_band = true;
    if !unsafe { pyre_object::is_none(ctx.buffer_callback) } {
        let w_ret = call_fn(ctx.buffer_callback, &[w_obj])?;
        in_band = crate::baseobjspace::is_true(w_ret)?;
    }
    if in_band {
        // In-band buffers memoize the wrapper (`_save_bytes_data` /
        // `_save_bytearray_data`), so a repeated reference becomes a GET. A
        // large payload streams via `file.write`; pin `w_obj` for the memoize.
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_obj);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        if readonly {
            save_raw_bytes(ctx, buf, &data)?;
        } else {
            save_raw_bytearray(buf, &data)?;
        }
        memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(slot));
    } else {
        buf.push(op::NEXT_BUFFER);
        if readonly {
            buf.push(op::READONLY_BUFFER);
        }
    }
    Ok(())
}

/// `interp_pickle.py save_raw_bytes` — emit raw bytes with the size-appropriate
/// BINBYTES opcode (no memoization). `data` is owned storage, so the streaming
/// `file.write` in `write_large_bytes` cannot dangle it; the caller pins any
/// object it still needs afterwards.
fn save_raw_bytes(ctx: &PickleCtx, buf: &mut Framer, data: &[u8]) -> Result<(), PyError> {
    let n = data.len();
    if n <= 0xff {
        buf.push(op::SHORT_BINBYTES);
        buf.push(n as u8);
        buf.extend_from_slice(data);
    } else if n > 0xffff_ffff && ctx.proto >= 4 {
        let mut header = vec![op::BINBYTES8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BINBYTES];
        header.extend_from_slice(&(n as u32).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else {
        buf.push(op::BINBYTES);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
        buf.extend_from_slice(data);
    }
    Ok(())
}

/// `interp_pickle.py save_raw_bytearray` — emit raw bytes with BYTEARRAY8
/// (no memoization).
fn save_raw_bytearray(buf: &mut Framer, data: &[u8]) -> Result<(), PyError> {
    let n = data.len();
    if n >= FRAME_SIZE_TARGET {
        let mut header = vec![op::BYTEARRAY8];
        header.extend_from_slice(&(n as u64).to_le_bytes());
        buf.write_large_bytes(&header, data)?;
    } else {
        buf.push(op::BYTEARRAY8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
        buf.extend_from_slice(data);
    }
    Ok(())
}

/// Build a Python `list` from `items` and pin it in the shadow stack,
/// returning its slot.  `w_list_new` pins each element across its own
/// allocation, so the snapshot is captured safely; thereafter the GC walks
/// the list and rewrites its entries, so `pinned_get` reads the relocated
/// element even after the recursive `save` calls below trigger collections.
fn pin_items(items: Vec<PyObjectRef>) -> usize {
    let w_list = pyre_object::listobject::w_list_new(items);
    pyre_object::gc_roots::pin_root(w_list);
    pyre_object::gc_roots::shadow_stack_len() - 1
}

/// Length of the pinned list at `slot`.
fn pinned_len(slot: usize) -> usize {
    let list = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { pyre_object::listobject::w_list_len(list) }
}

/// Element `i` of the pinned list at `slot`, re-read so a relocation of the
/// element (or the list) since the last access is observed.
fn pinned_get(slot: usize, i: usize) -> PyObjectRef {
    let list = pyre_object::gc_roots::shadow_stack_get(slot);
    unsafe { pyre_object::listobject::w_list_getitem(list, i as i64) }.unwrap()
}

/// `interp_pickle.py _batch_appends`. `slot` pins a Python `list` of the
/// items; each is re-read from the (GC-walked) list right before saving so a
/// mid-batch collection cannot leave a stale element behind.
fn batch_appends(ctx: &mut PickleCtx, buf: &mut Framer, slot: usize) -> Result<(), PyError> {
    let n = pinned_len(slot);
    if !ctx.bin {
        // proto 0 — no APPENDS, one APPEND per item.
        for i in 0..n {
            save(ctx, buf, pinned_get(slot, i))?;
            buf.push(op::APPEND);
        }
        return Ok(());
    }
    let mut i = 0;
    while i < n {
        if i + 1 == n {
            // Exactly one item left.
            save(ctx, buf, pinned_get(slot, i))?;
            buf.push(op::APPEND);
            return Ok(());
        }
        buf.push(op::MARK);
        let mut cnt = 0;
        while i < n && cnt < BATCHSIZE {
            save(ctx, buf, pinned_get(slot, i))?;
            i += 1;
            cnt += 1;
        }
        buf.push(op::APPENDS);
    }
    Ok(())
}

/// `interp_pickle.py _batch_setitems` (bin path). Single pair → SETITEM;
/// otherwise MARK … SETITEMS in batches of `BATCHSIZE`. `slot` pins a flat
/// `[k0, v0, k1, v1, …]` Python `list`, re-read per access (see
/// `batch_appends`).
fn batch_setitems(ctx: &mut PickleCtx, buf: &mut Framer, slot: usize) -> Result<(), PyError> {
    let npairs = pinned_len(slot) / 2;
    if !ctx.bin {
        // proto 0 — no SETITEMS, one SETITEM per pair.
        for p in 0..npairs {
            save(ctx, buf, pinned_get(slot, 2 * p))?;
            save(ctx, buf, pinned_get(slot, 2 * p + 1))?;
            buf.push(op::SETITEM);
        }
        return Ok(());
    }
    let mut p = 0;
    while p < npairs {
        if p + 1 == npairs {
            // Exactly one pair left.
            save(ctx, buf, pinned_get(slot, 2 * p))?;
            save(ctx, buf, pinned_get(slot, 2 * p + 1))?;
            buf.push(op::SETITEM);
            return Ok(());
        }
        buf.push(op::MARK);
        let mut cnt = 0;
        while p < npairs && cnt < BATCHSIZE {
            save(ctx, buf, pinned_get(slot, 2 * p))?;
            save(ctx, buf, pinned_get(slot, 2 * p + 1))?;
            p += 1;
            cnt += 1;
        }
        buf.push(op::SETITEMS);
    }
    Ok(())
}

/// `interp_pickle.py W_Pickler.write_get` — emit a GET back-reference.
fn write_get(ctx: &PickleCtx, buf: &mut Framer, idx: usize) {
    if ctx.bin {
        if idx < 256 {
            buf.push(op::BINGET);
            buf.push(idx as u8);
        } else {
            buf.push(op::LONG_BINGET);
            buf.extend_from_slice(&(idx as u32).to_le_bytes());
        }
    } else {
        buf.push(op::GET);
        buf.extend_from_slice(format!("{idx}\n").as_bytes());
    }
}

/// `interp_pickle.py memoize` — record the object's identity and write the
/// put opcode. A no-op in `fast` mode (no PUT/GET, no recursion guard).
fn memoize(ctx: &mut PickleCtx, buf: &mut Framer, w_obj: PyObjectRef) {
    if ctx.fast {
        return;
    }
    let list = ctx.memo_list();
    let idx = unsafe { pyre_object::listobject::w_list_len(list) };
    // Compute the move-stable hash before the append, whose growth could
    // relocate `w_obj` and leave the local stale.
    let h = pyre_object::gc_hook::gc_identity_hash(w_obj as usize);
    unsafe { pyre_object::listobject::w_list_append(list, w_obj) };
    ctx.index.entry(h).or_default().push(idx);
    if ctx.proto >= 4 {
        buf.push(op::MEMOIZE);
    } else if ctx.bin {
        if idx < 256 {
            buf.push(op::BINPUT);
            buf.push(idx as u8);
        } else {
            buf.push(op::LONG_BINPUT);
            buf.extend_from_slice(&(idx as u32).to_le_bytes());
        }
    } else {
        buf.push(op::PUT);
        buf.extend_from_slice(format!("{idx}\n").as_bytes());
    }
}

// ── reduce / global ──────────────────────────────────────────────────

/// `whichmodule` — the module an object belongs to. `__module__` takes
/// precedence; otherwise scan `sys.modules` for the module that exposes
/// `name` resolving back to `w_obj`, skipping `__main__` / `__mp_main__` /
/// `None`, and default to `"__main__"`. Then verify (importing the module)
/// that `module_name.name` resolves back to this exact object, raising a
/// PicklingError otherwise — the dump-time check that the wire reference is
/// actually loadable.
fn whichmodule(w_obj: PyObjectRef, name: &str) -> Result<PyObjectRef, PyError> {
    if name.split('.').any(|s| s == "<locals>") {
        return Err(pickling_error(format!("Can't pickle local object {name}")));
    }
    // `interp_pickle.py:1738-1742 whichmodule` returns any non-None
    // `__module__`. A non-string module name is invalid; reject it here with
    // `TypeError("module name must be a string")` — the same error it surfaces
    // once used, raised at resolution time rather than deferred to the import.
    let from_attr: Option<String> = match crate::baseobjspace::findattr(w_obj, "__module__") {
        Some(m) if !unsafe { pyre_object::is_none(m) } => {
            if !unsafe { pyre_object::is_str(m) } {
                return Err(PyError::type_error("module name must be a string"));
            }
            Some(unsafe { pyre_object::strobject::w_str_get_value(m) }.to_string())
        }
        _ => None,
    };
    let module_name = match from_attr {
        Some(mn) => mn,
        None => {
            // Scan sys.modules; a match here is already verified by identity.
            let modules = crate::importing::sys_modules_dict();
            let mut found: Option<String> = None;
            if !modules.is_null() {
                // `getattribute_dotted` runs Python and can relocate objects, so
                // pin `w_obj` and every candidate module up front (capturing each
                // name as an owned String — GC-independent), then scan via the
                // pinned slots. The snapshot loop itself triggers no collection.
                let _roots = pyre_object::gc_roots::push_roots();
                pyre_object::gc_roots::pin_root(w_obj);
                let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let mut candidates: Vec<(String, usize)> = Vec::new();
                for (w_modname, w_module) in
                    unsafe { pyre_object::dictmultiobject::w_dict_items(modules) }
                {
                    if !unsafe { pyre_object::is_str(w_modname) }
                        || unsafe { pyre_object::is_none(w_module) }
                    {
                        continue;
                    }
                    let modname =
                        unsafe { pyre_object::strobject::w_str_get_value(w_modname) }.to_string();
                    if modname == "__main__" || modname == "__mp_main__" {
                        continue;
                    }
                    pyre_object::gc_roots::pin_root(w_module);
                    candidates.push((modname, pyre_object::gc_roots::shadow_stack_len() - 1));
                }
                for (modname, mod_slot) in candidates {
                    let w_module = pyre_object::gc_roots::shadow_stack_get(mod_slot);
                    if let Ok((resolved, _)) = getattribute_dotted(w_module, name) {
                        if crate::baseobjspace::is_w(
                            resolved,
                            pyre_object::gc_roots::shadow_stack_get(obj_slot),
                        ) {
                            found = Some(modname);
                            break;
                        }
                    }
                }
            }
            match found {
                Some(mn) => return Ok(pyre_object::w_str_new(&mn)),
                None => String::from("__main__"),
            }
        }
    };
    // Verify `module_name.name` resolves back to `w_obj` (qualnames walked).
    match crate::module::_pickle::try_resolve_global(&module_name, name, true)? {
        Some(resolved) if crate::baseobjspace::is_w(resolved, w_obj) => {
            Ok(pyre_object::w_str_new(&module_name))
        }
        Some(_) => Err(pickling_error(format!(
            "Can't pickle object: it's not the same object as {module_name}.{name}"
        ))),
        None => Err(pickling_error(format!(
            "Can't pickle object: it's not found as {module_name}.{name}"
        ))),
    }
}

/// `interp_pickle.py save_global` / `save_global2` — save an object by
/// qualified reference. `w_name_opt` carries the name when a `__reduce__`
/// returned a string; otherwise it is derived from `__qualname__`.
fn save_global(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    w_obj: PyObjectRef,
    w_name_opt: Option<PyObjectRef>,
) -> Result<(), PyError> {
    let w_name = match w_name_opt {
        Some(n) => n,
        None => crate::baseobjspace::findattr(w_obj, "__qualname__")
            .or_else(|| crate::baseobjspace::findattr(w_obj, "__name__"))
            .ok_or_else(|| pickling_error("Can't pickle object: no __qualname__ / __name__"))?,
    };
    // `whichmodule` imports the home module (to verify the reference), so pin
    // `w_obj` / `w_name` and re-read them afterwards.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(w_obj);
    let obj_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    pyre_object::gc_roots::pin_root(w_name);
    let name_slot = pyre_object::gc_roots::shadow_stack_len() - 1;

    let name = unsafe {
        pyre_object::strobject::w_str_get_value(pyre_object::gc_roots::shadow_stack_get(name_slot))
    }
    .to_string();
    let w_module_name = whichmodule(pyre_object::gc_roots::shadow_stack_get(obj_slot), &name)?;
    let module_name = unsafe { pyre_object::strobject::w_str_get_value(w_module_name) }.to_string();

    // protocol >= 2: a `copyreg` extension code is emitted as EXT1/EXT2/EXT4
    // (and the object is not memoized — the reference is idempotent).
    if ctx.proto >= 2 {
        if let Some(code) = extension_code(&module_name, &name) {
            write_ext(buf, code)?;
            return Ok(());
        }
    }

    if ctx.proto >= 4 {
        save(ctx, buf, pyre_object::w_str_new(&module_name))?;
        save(ctx, buf, pyre_object::gc_roots::shadow_stack_get(name_slot))?;
        buf.push(op::STACK_GLOBAL);
    } else if name.contains('.') {
        // protocol < 4 nested: objects with a multi-part `__qualname__` are
        // represented as `getattr(getattr(..., attrname1), attrname2)`. The
        // top-level name is emitted by GLOBAL and each remaining dotted part
        // by one `getattr(<current>, attrname)` REDUCE. `getattr` and the
        // attrname strings memoize through `save`; the top-level GLOBAL and
        // the intermediate tuples are not memoized, and the final result is
        // memoized by the trailing `memoize(obj)`.
        let parts: Vec<&str> = name.split('.').collect();
        let rest = &parts[1..];
        for _ in rest {
            let w_getattr = builtin_attr("getattr")?;
            save(ctx, buf, w_getattr)?;
            if ctx.proto < 2 {
                buf.push(op::MARK);
            }
        }
        save_toplevel_by_name(ctx, buf, &module_name, parts[0])?;
        for attrname in rest {
            save(ctx, buf, pyre_object::w_str_new(attrname))?;
            if ctx.proto < 2 {
                buf.push(op::TUPLE);
            } else {
                buf.push(op::TUPLE2);
            }
            buf.push(op::REDUCE);
        }
    } else {
        save_toplevel_by_name(ctx, buf, &module_name, &name)?;
    }
    memoize(ctx, buf, pyre_object::gc_roots::shadow_stack_get(obj_slot));
    Ok(())
}

/// `_save_toplevel_by_name` — emit a GLOBAL opcode for a top-level name,
/// applying the protocol-< 3 `fix_imports` py3 → py2 reverse map.
fn save_toplevel_by_name(
    ctx: &PickleCtx,
    buf: &mut Framer,
    module_name: &str,
    name: &str,
) -> Result<(), PyError> {
    let (module_name, name) = if ctx.proto < 3 && ctx.fix_imports {
        crate::module::_pickle::compat_map(module_name, name, true)
    } else {
        (module_name.to_string(), name.to_string())
    };
    buf.push(op::GLOBAL);
    buf.extend_from_slice(module_name.as_bytes());
    buf.push(b'\n');
    buf.extend_from_slice(name.as_bytes());
    buf.push(b'\n');
    Ok(())
}

/// The `copyreg._extension_registry` code for `(module_name, name)`, if any.
fn extension_code(module_name: &str, name: &str) -> Option<i64> {
    let copyreg = import_module("copyreg").ok()?;
    let registry = crate::baseobjspace::getattr_str(copyreg, "_extension_registry").ok()?;
    let key = pyre_object::tupleobject::w_tuple_new(vec![
        pyre_object::w_str_new(module_name),
        pyre_object::w_str_new(name),
    ]);
    let code = unsafe { pyre_object::w_dict_lookup(registry, key) }?;
    crate::baseobjspace::int_w(code).ok()
}

/// Emit the size-appropriate EXT opcode for an extension code. `save_global`
/// requires `0 < code <= 0x7fffffff`; anything else is out of range.
fn write_ext(buf: &mut Framer, code: i64) -> Result<(), PyError> {
    if code <= 0 || code > 0x7fffffff {
        return Err(PyError::runtime_error(format!(
            "extension code {code} is out of range"
        )));
    }
    if code <= 0xff {
        buf.push(op::EXT1);
        buf.push(code as u8);
    } else if code <= 0xffff {
        buf.push(op::EXT2);
        buf.extend_from_slice(&(code as u16).to_le_bytes());
    } else {
        buf.push(op::EXT4);
        buf.extend_from_slice(&(code as i32).to_le_bytes());
    }
    Ok(())
}

/// `__builtins__.<name>` (e.g. `getattr`), via the execution context's
/// `lookup_builtin` (the `LOAD_GLOBAL` path). Only used on the rare nested
/// protocol < 4 path.
fn builtin_attr(name: &str) -> Result<PyObjectRef, PyError> {
    crate::module::_pickle::lookup_builtin(name)
        .ok_or_else(|| pickling_error(format!("Can't resolve builtin {name:?}")))
}

/// `interp_pickle.py save_reduce`. `rv` is the 2-to-6 element reduce tuple
/// `(func, args[, state[, listitems[, dictitems[, state_setter]]]])`.
fn save_reduce(
    ctx: &mut PickleCtx,
    buf: &mut Framer,
    rv: &[PyObjectRef],
    w_obj_opt: Option<PyObjectRef>,
) -> Result<(), PyError> {
    let _roots = pyre_object::gc_roots::push_roots();
    // Recursive saves (and the reduce callbacks they invoke) relocate young
    // objects, so pin the reduce values in a GC-walked `list` and re-read each
    // one immediately before it is consumed.
    let rv_len = rv.len();
    let rv_slot = pin_items(rv.to_vec());
    let w_obj_slot = match w_obj_opt {
        Some(o) => {
            pyre_object::gc_roots::pin_root(o);
            Some(pyre_object::gc_roots::shadow_stack_len() - 1)
        }
        None => None,
    };
    let rv_get = |i: usize| pinned_get(rv_slot, i);
    let present = |i: usize| i < rv_len && !unsafe { pyre_object::is_none(pinned_get(rv_slot, i)) };

    if !unsafe { pyre_object::is_tuple(rv_get(1)) } {
        return Err(pickling_error("args from save_reduce() must be a tuple"));
    }
    if !crate::baseobjspace::callable_w(rv_get(0)) {
        return Err(pickling_error("func from save_reduce() must be callable"));
    }

    let has_state = present(2);
    let has_listitems = present(3);
    let has_dictitems = present(4);
    let has_state_setter = present(5);

    let func_name = func_name_str(rv_get(0));

    // Pin the args tuple; its elements are re-read per save.
    pyre_object::gc_roots::pin_root(rv_get(1));
    let args_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    let args_get = |i: usize| unsafe {
        pyre_object::tupleobject::w_tuple_getitem(
            pyre_object::gc_roots::shadow_stack_get(args_slot),
            i as i64,
        )
        .unwrap()
    };
    let args_len = unsafe {
        pyre_object::tupleobject::w_tuple_len(pyre_object::gc_roots::shadow_stack_get(args_slot))
    };

    if ctx.proto >= 2 && func_name.as_deref() == Some("__newobj_ex__") {
        if args_len != 3 {
            return Err(pickling_error("__newobj_ex__ requires three args"));
        }
        if crate::baseobjspace::findattr(args_get(0), "__new__").is_none() {
            return Err(pickling_error(
                "args[0] from __newobj_ex__ args has no __new__",
            ));
        }
        if let Some(slot) = w_obj_slot {
            let w_class = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(slot),
                "__class__",
            )?;
            if !crate::baseobjspace::is_w(args_get(0), w_class) {
                pyre_object::gc_roots::pin_root(w_class);
                let class_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let cls_repr = unsafe { crate::display::py_repr(args_get(0))? };
                let obj_class_repr = unsafe {
                    crate::display::py_repr(pyre_object::gc_roots::shadow_stack_get(class_slot))?
                };
                return Err(pickling_error(format!(
                    "first argument to __newobj_ex__() must be {obj_class_repr}, not {cls_repr}"
                )));
            }
        }
        if ctx.proto >= 4 {
            save(ctx, buf, args_get(0))?;
            save(ctx, buf, args_get(1))?;
            save(ctx, buf, args_get(2))?;
            buf.push(op::NEWOBJ_EX);
        } else {
            // protocol 2/3: encode the constructor as
            // `partial(cls.__new__, cls, *args, **kwargs)`, then REDUCE with
            // an empty argument tuple.
            let functools = import_module("functools")?;
            let w_partial = crate::baseobjspace::getattr_str(functools, "partial")?;
            pyre_object::gc_roots::pin_root(w_partial);
            let partial_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            let w_new = crate::baseobjspace::getattr_str(args_get(0), "__new__")?;
            pyre_object::gc_roots::pin_root(w_new);
            let new_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
            // keyword arguments from the `kwargs` dict (re-read fresh; no GC
            // until the `partial` construction below).
            let kw_items = unsafe { pyre_object::dictmultiobject::w_dict_items(args_get(2)) };
            let mut kwargs = Vec::with_capacity(kw_items.len());
            for (k, v) in kw_items {
                if !unsafe { pyre_object::is_str(k) } {
                    return Err(pickling_error("__newobj_ex__ kwargs keys must be strings"));
                }
                kwargs.push((
                    unsafe { pyre_object::strobject::w_str_get_wtf8(k) }.to_owned(),
                    v,
                ));
            }
            // positional: (cls.__new__, cls, *args).
            let mut pos = vec![
                pyre_object::gc_roots::shadow_stack_get(new_slot),
                args_get(0),
            ];
            pos.extend(tuple_items(args_get(1)));
            let ec = crate::call::getexecutioncontext();
            if ec.is_null() {
                return Err(pickling_error("no execution context for __newobj_ex__"));
            }
            let frame = unsafe { (*ec).gettopframe() };
            if frame.is_null() {
                return Err(pickling_error("no frame for __newobj_ex__ at protocol < 4"));
            }
            let w_func = crate::call::call_with_kwargs(
                unsafe { &mut *frame },
                pyre_object::gc_roots::shadow_stack_get(partial_slot),
                &pos,
                &kwargs,
            )?;
            save(ctx, buf, w_func)?;
            save(ctx, buf, pyre_object::tupleobject::w_tuple_new(Vec::new()))?;
            buf.push(op::REDUCE);
        }
    } else if ctx.proto >= 2 && func_name.as_deref() == Some("__newobj__") {
        if args_len == 0 {
            return Err(pickling_error("__newobj__ requires at least one arg"));
        }
        if crate::baseobjspace::findattr(args_get(0), "__new__").is_none() {
            return Err(pickling_error(
                "args[0] from __newobj__ args has no __new__",
            ));
        }
        if let Some(slot) = w_obj_slot {
            let w_class = crate::baseobjspace::getattr_str(
                pyre_object::gc_roots::shadow_stack_get(slot),
                "__class__",
            )?;
            if !crate::baseobjspace::is_w(args_get(0), w_class) {
                pyre_object::gc_roots::pin_root(w_class);
                let class_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
                let cls_repr = unsafe { crate::display::py_repr(args_get(0))? };
                let obj_class_repr = unsafe {
                    crate::display::py_repr(pyre_object::gc_roots::shadow_stack_get(class_slot))?
                };
                return Err(pickling_error(format!(
                    "first argument to __newobj__() must be {obj_class_repr}, not {cls_repr}"
                )));
            }
        }
        let w_newargs =
            pyre_object::tupleobject::w_tuple_new((1..args_len).map(|i| args_get(i)).collect());
        pyre_object::gc_roots::pin_root(w_newargs);
        let newargs_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        save(ctx, buf, args_get(0))?;
        save(
            ctx,
            buf,
            pyre_object::gc_roots::shadow_stack_get(newargs_slot),
        )?;
        buf.push(op::NEWOBJ);
    } else {
        save(ctx, buf, rv_get(0))?;
        save(ctx, buf, rv_get(1))?;
        buf.push(op::REDUCE);
    }

    if let Some(slot) = w_obj_slot {
        let w_obj = pyre_object::gc_roots::shadow_stack_get(slot);
        if let Some(idx) = ctx.memo_get(w_obj) {
            buf.push(op::POP);
            write_get(ctx, buf, idx);
        } else {
            memoize(ctx, buf, w_obj);
        }
    }

    if has_listitems {
        let items_slot = drain_iter_pinned(rv_get(3))?;
        batch_appends(ctx, buf, items_slot)?;
    }
    if has_dictitems {
        let pairs_slot = drain_iter_pairs_pinned(rv_get(4))?;
        batch_setitems(ctx, buf, pairs_slot)?;
    }
    if has_state {
        if has_state_setter {
            save(ctx, buf, rv_get(5))?;
            save(
                ctx,
                buf,
                pyre_object::gc_roots::shadow_stack_get(w_obj_slot.unwrap()),
            )?;
            save(ctx, buf, rv_get(2))?;
            buf.push(op::TUPLE2);
            buf.push(op::REDUCE);
            buf.push(op::POP);
        } else {
            save(ctx, buf, rv_get(2))?;
            buf.push(op::BUILD);
        }
    }
    Ok(())
}

/// The `__name__` of a callable as an owned `String`, if it is a str.
fn func_name_str(w_func: PyObjectRef) -> Option<String> {
    let w_name = crate::baseobjspace::findattr(w_func, "__name__")?;
    if unsafe { pyre_object::is_str(w_name) } {
        Some(unsafe { pyre_object::strobject::w_str_get_value(w_name) }.to_string())
    } else {
        None
    }
}

fn tuple_items(w_tuple: PyObjectRef) -> Vec<PyObjectRef> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_tuple) };
    (0..n)
        .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, i as i64).unwrap() })
        .collect()
}

/// Drain an iterable into a freshly-pinned Python `list`, returning its
/// shadow-stack slot. Appending into a GC-walked `list` as iteration proceeds
/// keeps every already-yielded item reachable and relocation-tracked — unlike
/// a Rust `Vec`, whose elements a later `next()` could strand by relocating
/// them. The iterator object is pinned too, since `next` may collect.
fn drain_iter_pinned(w_iterable: PyObjectRef) -> Result<usize, PyError> {
    let slot = pin_items(Vec::new());
    let w_iter = crate::baseobjspace::iter(w_iterable)?;
    pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    loop {
        match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot)) {
            Ok(item) => unsafe {
                pyre_object::listobject::w_list_append(
                    pyre_object::gc_roots::shadow_stack_get(slot),
                    item,
                )
            },
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(slot)
}

/// Drain an iterable of `(key, value)` pairs into a freshly-pinned, flat
/// `[k0, v0, k1, v1, …]` Python `list` (see [`drain_iter_pinned`]), returning
/// its shadow-stack slot.
fn drain_iter_pairs_pinned(w_iterable: PyObjectRef) -> Result<usize, PyError> {
    let slot = pin_items(Vec::new());
    let w_iter = crate::baseobjspace::iter(w_iterable)?;
    pyre_object::gc_roots::pin_root(w_iter);
    let iter_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
    loop {
        let it = match crate::baseobjspace::next(pyre_object::gc_roots::shadow_stack_get(iter_slot))
        {
            Ok(it) => it,
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        };
        if !unsafe { pyre_object::is_tuple(it) }
            || unsafe { pyre_object::tupleobject::w_tuple_len(it) } != 2
        {
            return Err(pickling_error("dictitems must yield (key, value) pairs"));
        }
        // `w_list_append` does not collect, so `it`/`k`/`v` stay valid between
        // the reads and the two appends.
        let k = unsafe { pyre_object::tupleobject::w_tuple_getitem(it, 0).unwrap() };
        let v = unsafe { pyre_object::tupleobject::w_tuple_getitem(it, 1).unwrap() };
        unsafe {
            pyre_object::listobject::w_list_append(
                pyre_object::gc_roots::shadow_stack_get(slot),
                k,
            );
            pyre_object::listobject::w_list_append(
                pyre_object::gc_roots::shadow_stack_get(slot),
                v,
            );
        }
    }
    Ok(slot)
}
