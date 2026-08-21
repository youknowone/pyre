//! PyCode — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;
use pyre_object::{
    w_bool_from, w_bool_get_value, w_int_new, w_list_new, w_seq_iter_new, w_str_new, w_tuple_new,
};
use rustpython_compiler_core::SourceLocation;
use rustpython_compiler_core::bytecode::PyCodeLocationInfoKind;

const YIELDS_INSIDE_TRY_BIT: u16 = 0x8000;

/// Compatibility marker for malformed bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeCorruption;

impl From<BytecodeCorruption> for crate::PyError {
    fn from(_: BytecodeCorruption) -> Self {
        crate::PyError::new(
            crate::PyErrorKind::BytecodeCorruption,
            "bytecode corruption",
        )
    }
}

/// CPython 3.14 `_PyCodeAddressRange`, ported through RustPython's code
/// object implementation.  This reads the authoritative `co_linetable`;
/// `CodeObject.locations` is an execution-oriented expansion which cannot
/// represent a missing line or column.
struct PyCodeAddressRange<'a> {
    ar_start: i32,
    ar_end: i32,
    ar_line: i32,
    computed_line: i32,
    reader: LineTableReader<'a>,
}

impl<'a> PyCodeAddressRange<'a> {
    fn new(linetable: &'a [u8], first_line: i32) -> Self {
        Self {
            ar_start: 0,
            ar_end: 0,
            ar_line: -1,
            computed_line: first_line,
            reader: LineTableReader::new(linetable),
        }
    }

    fn advance(&mut self) -> bool {
        let Some(first_byte) = self.reader.read_byte() else {
            return false;
        };
        // A byte in header position is decoded as one, bit 7 or not: the
        // marker separates a header from the payload bytes the skip below
        // consumes, and is not consulted here. `code.replace()` stores an
        // arbitrary `co_linetable`, and stopping on the marker made
        // `co_linetable=b"\0"` report no ranges where one entry is decoded.
        let code = (first_byte >> 3) & 0x0f;
        let length = ((first_byte & 0x07) + 1) as i32;
        self.computed_line += self.get_line_delta(code);
        self.ar_line = if first_byte >> 3 == 0x1f {
            -1
        } else {
            self.computed_line
        };
        self.ar_start = self.ar_end;
        self.ar_end += length * 2;

        // Every payload byte has bit 7 clear; the next header has it set.
        while self.reader.peek_byte().is_some_and(|byte| byte & 0x80 == 0) {
            self.reader.read_byte();
        }
        true
    }

    fn get_line_delta(&mut self, code: u8) -> i32 {
        let Some(kind) = PyCodeLocationInfoKind::from_code(code) else {
            return 0;
        };
        match kind {
            PyCodeLocationInfoKind::None => 0,
            PyCodeLocationInfoKind::Long => {
                let delta = self.reader.read_signed_varint();
                self.reader.read_varint();
                self.reader.read_varint();
                self.reader.read_varint();
                delta
            }
            PyCodeLocationInfoKind::NoColumns => self.reader.read_signed_varint(),
            PyCodeLocationInfoKind::OneLine0
            | PyCodeLocationInfoKind::OneLine1
            | PyCodeLocationInfoKind::OneLine2 => {
                self.reader.read_byte();
                self.reader.read_byte();
                kind.one_line_delta().unwrap_or(0)
            }
            _ if kind.is_short() => {
                self.reader.read_byte();
                0
            }
            _ => 0,
        }
    }
}

/// RustPython `LineTableReader`, matching CPython's 6-bit little-endian
/// location-table varints.
struct LineTableReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> LineTableReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn read_byte(&mut self) -> Option<u8> {
        let byte = self.data.get(self.pos).copied()?;
        self.pos += 1;
        Some(byte)
    }

    fn peek_byte(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    /// Read one location-table varint: 6 bits per byte, **least
    /// significant group first**, bit 6 (0x40) marking continuation.
    ///
    /// This inverts `write_varint`, and deliberately differs from
    /// `decode_varint` below, which reads the exception table most
    /// significant group first to invert `parse_varint`. Both helpers sit
    /// a few lines apart in `pycore_code.h`: the byte order is a property
    /// of the table, not of the file. `_decode_varint` in PyPy's pycode.py
    /// is the exception-table reader despite its generic name, so reusing
    /// it here would decode every multi-byte line delta wrong.
    fn read_varint(&mut self) -> u32 {
        let Some(first) = self.read_byte() else {
            return 0;
        };
        let mut value = (first & 0x3f) as u32;
        let mut shift = 0;
        let mut byte = first;
        while byte & 0x40 != 0 {
            let Some(next) = self.read_byte() else {
                break;
            };
            shift += 6;
            // `code.replace(co_linetable=...)` stores arbitrary bytes, so the
            // continuation chain can run past what a u32 holds. Those groups
            // are unrepresentable either way; drop them rather than shift by
            // the full width, which panics in a debug build.
            if shift < u32::BITS {
                value |= ((next & 0x3f) as u32) << shift;
            }
            byte = next;
        }
        value
    }

    fn read_signed_varint(&mut self) -> i32 {
        let value = self.read_varint();
        if value & 1 != 0 {
            -((value >> 1) as i32)
        } else {
            (value >> 1) as i32
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.data.len()
    }
}

/// pycode.py:683-695 — decode one CPython-3.11 varint at `i`.
///
/// Returns `(value, new_i)`. Reads 6 bits per byte, MSB first. Bit 6
/// (0x40) is the continuation flag; bit 7 (0x80) is the start-of-entry
/// marker, ignored here and masked off along with the continuation bit
/// via `& 63`.
///
/// Mirrors `parse_varint`. The location table uses the opposite byte
/// order — see `LineTableReader::read_varint` — so this decoder must not
/// be reused for `co_linetable`.
#[inline]
pub fn decode_varint(table: &[u8], mut i: usize) -> (u32, usize) {
    let mut b = table[i] as u32;
    i += 1;
    let mut value = b & 63;
    while b & 64 != 0 {
        b = table[i] as u32;
        i += 1;
        value = (value << 6) | (b & 63);
    }
    (value, i)
}

/// Decoded exception-table entry. Byte offsets throughout.
///
/// Field shape mirrors PyPy's `(start, length, target, depth, lasti)`
/// per-entry varint sequence; `end = start + length` is precomputed for
/// callers that want a half-open `start..end` range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExceptionTableEntry {
    pub start: u32,
    pub end: u32,
    pub target: u32,
    pub depth: u32,
    pub lasti: bool,
}

/// pycode.py `lookup_exceptiontable`.
///
/// Search `table` for a handler covering `instr_offset` (byte offset
/// into `co_code`). Returns `Some((target, depth, lasti))` when found,
/// `None` otherwise.
///
/// **Last matching wins**: entries are scanned in encoding order; if
/// multiple entries cover `instr_offset`, the later one (innermost in
/// CPython's emission order) is returned. Scanning short-circuits when
/// `start > instr_offset`, since entries are emitted in ascending
/// `start` order.
pub fn lookup_exceptiontable(table: &[u8], instr_offset: u32) -> Option<(u32, u32, bool)> {
    let n = table.len();
    if n == 0 {
        return None;
    }
    let mut best: Option<(u32, u32, bool)> = None;
    let mut i = 0;
    while i < n {
        let (start_raw, ni) = decode_varint(table, i);
        let start = start_raw * 2;
        let (length_raw, ni) = decode_varint(table, ni);
        let length = length_raw * 2;
        let (target_raw, ni) = decode_varint(table, ni);
        let target = target_raw * 2;
        let (dl, ni) = decode_varint(table, ni);
        let depth = dl >> 1;
        let lasti = (dl & 1) != 0;
        i = ni;
        if start <= instr_offset && instr_offset < start + length {
            best = Some((target, depth, lasti));
        } else if start > instr_offset {
            break;
        }
    }
    best
}

/// PyPy `astcompiler/codegen.py:2825-2826` / `generator.py:24-27`:
/// compute the `CO_YIELD_INSIDE_TRY` property once while constructing the
/// interpreter-level code object.
///
/// Python 3.14 wraps every generator body in a depth-zero, `lasti` exception
/// entry which only converts an escaping `StopIteration`; that synthetic
/// entry must not make every generator finalizable. Entries emitted for an
/// actual `try` around a yield either omit `lasti` at depth zero (`try`) or
/// carry a non-zero unwind depth (`with`). `lookup_exceptiontable` selects the
/// innermost entry, matching the compiler's `has_yield_inside_try` question.
fn code_yields_inside_try(code: &crate::CodeObject) -> bool {
    let mut index = 0;
    while index < code.instructions.len() {
        if matches!(
            code.instructions[index].op,
            crate::bytecode::Instruction::YieldValue { .. }
        ) {
            let offset = (index * 2) as u32;
            if let Some((_target, depth, lasti)) =
                lookup_exceptiontable(&code.exceptiontable, offset)
                && (depth != 0 || !lasti)
            {
                return true;
            }
        }
        index += 1;
    }
    false
}

/// Iterator over all decoded entries in `table`.
///
/// Convenience for callers that want a structural view (JIT codewriter,
/// liveness, the PyPy-style `mark_stacks` handler-shape seeder). The
/// runtime `handle_operation_error` dispatch uses [`lookup_exceptiontable`]
/// directly.
pub fn decode_exceptiontable(table: &[u8]) -> ExceptionTableIter<'_> {
    ExceptionTableIter { table, i: 0 }
}

pub struct ExceptionTableIter<'a> {
    table: &'a [u8],
    i: usize,
}

impl Iterator for ExceptionTableIter<'_> {
    type Item = ExceptionTableEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.table.len() {
            return None;
        }
        let (start_raw, i) = decode_varint(self.table, self.i);
        let start = start_raw * 2;
        let (length_raw, i) = decode_varint(self.table, i);
        let length = length_raw * 2;
        let (target_raw, i) = decode_varint(self.table, i);
        let target = target_raw * 2;
        let (dl, i) = decode_varint(self.table, i);
        self.i = i;
        Some(ExceptionTableEntry {
            start,
            end: start + length,
            target,
            depth: dl >> 1,
            lasti: (dl & 1) != 0,
        })
    }
}

/// Compatibility container for code-hook caching state.
#[derive(Debug, Default)]
pub struct CodeHookCache {
    _code_hook: Option<PyObjectRef>,
}

/// Type descriptor for code objects.
pub static CODE_TYPE: PyType = pyre_object::pyobject::new_pytype("code");

/// Python code object wrapper.
///
/// Stores an opaque pointer to the bytecode CodeObject. A top-level body is
/// `Box::into_raw`'d; a nested body points into that permanently-live owner,
/// matching PyPy's one recursively-owned `co_consts_w` code graph without
/// cloning the child graph at each wrapping boundary.
#[repr(C)]
pub struct PyCode {
    pub ob_header: PyObject,
    /// Opaque pointer to a permanently-live `CodeObject`. Top-level bodies are
    /// owned via `Box::into_raw`; nested bodies borrow from one of those boxes.
    pub code_ptr: *const (),
    /// `pycode.py self.co_firstlineno = firstlineno`. RustPython's
    /// `CodeObject.first_line_number: Option<OneIndexed>` cannot represent
    /// the zero/negative values accepted by Python 3.14's CodeType
    /// constructor, so preserve the exact Python integer on the PyCode itself.
    pub co_firstlineno_raw: i32,
    /// `pycode.py self.co_filename = filename`: the byte-exact filesystem
    /// spelling owned by this `PyCode`. Null means that `code.source_path` is
    /// already the exact UTF-8 spelling, avoiding an allocation for compiled
    /// source paths and ordinary filename replacements.
    ///
    /// Do not add the derived `pycode.py:136 w_filename` cache without giving
    /// the slot everything `w_globals` gets. The filesystem decode hands back a
    /// movable managed string, so a retained slot has to be visited by a managed
    /// wrapper's own custom trace and forwarded by `eval::walk_raw_code_roots`
    /// for the bootstrap family; without both it holds a pre-move address after
    /// the first collection that moves the string.
    pub filename_bytes: *mut Vec<u8>,
    /// PyPy `pycode.py:132 self.co_code = co_code`: byte-exact public
    /// bytecode supplied to `CodeType` / `code.replace` when it contains an
    /// opcode that RustPython's enum cannot represent (or its explicit
    /// `Reserved` hole).  Ordinary compiler-produced code leaves this null
    /// and derives `co_code` from `CodeUnits`.
    ///
    /// The execution stream uses `Instruction::Reserved` as a placeholder,
    /// but getters, equality, hashing and marshal must retain the actual byte.
    /// Keeping the owner-local field mirrors PyPy's `PyCode.co_code`; an
    /// address-keyed side table would lose both lifetime and structural parity.
    pub co_code_bytes: *mut Vec<u8>,
    /// Whether a nested compiler constant selected by
    /// `importing.py update_code_filenames`' `oldname` guard inherits
    /// `filename_bytes` if a fallback slot ever has to be rebuilt. False for
    /// `pycode.py` constructor/replace filenames, which affect only the
    /// code object being constructed.
    pub filename_inherits_to_nested: bool,
    /// PyPy: `PyCode.w_globals` — the globals dict OBJECT (`W_DictMultiObject`,
    /// `pycode.py "w_globals?"`).  Module globals are `malloc_typed`-
    /// immortal, but `exec`/custom-globals dicts are `try_gc_alloc` movable.
    /// Managed wrappers trace this slot through `eval::walk_raw_code_roots`.
    /// Bootstrap wrappers outside the collector are registered in
    /// [`W_GLOBALS_STAMPED_CODES`] and forwarded from there. Null until first
    /// stamped by `frame_stores_global`.
    pub w_globals: PyObjectRef,
    /// `typedef.py make_weakref_descr(PyCode)` installs `_lifeline_` on
    /// the interpreter-level class.  Keep that owner-local field here rather
    /// than routing code objects through the fallback address-keyed table.
    /// The lifeline owns the cached weakrefs and their callbacks and is traced
    /// with the rest of this code object's managed children.
    pub w_weakreflifeline: PyObjectRef,
    /// PyPy: `PyCode.hidden_applevel` (`pycode.py, 147`). Set by
    /// `pycompiler.compile(hidden_applevel=True)` for PyPy gateway/
    /// app_main bridge code.  Pyre has no such call site yet, so this
    /// is always `false` on currently constructed instances; the
    /// field exists so that `frame.hide()` can read the canonical
    /// `pyframe.py return self.pycode.hidden_applevel`.
    pub hidden_applevel: bool,
    /// pycode.py `_compute_flatcall`. Cached arity descriptor:
    /// - 0-4: impossible (builtins only)
    /// - FLATPYCALL | co_argcount: simple user function
    /// - HOPELESS: has *args/**kwargs/kwonly/too many params
    /// The unused high bit caches `CO_YIELD_INSIDE_TRY`; accessors mask it
    /// away from the arity value.
    pub fast_natural_arity: u16,
    /// Cached [`crate::pyframe::npure_cellvars`] — the count of cellvars that
    /// are not also varnames.  Code-invariant, so computed once here instead
    /// of re-walking the O(cellvars × varnames) overlap check on every
    /// `PyFrame::ncells()` / stack-base query (a per-`pop_value` hot path).
    /// `u32::MAX` sentinel when `code_ptr` is null/unaligned (test stubs).
    pub npure_cellvars: u32,
    /// `pycode.py self._globals_caches = [None] * len(self.co_names_w)`.
    ///
    /// Per-name slot for `LOAD_GLOBAL_cached` / `STORE_GLOBAL_cached`
    /// (`celldict.py:292,321,335,353`).  Stores a weak reference to
    /// the `GlobalCache` resolved on the first miss, so subsequent
    /// hits bypass the `mstrategy.get_global_cache(varname)` string
    /// lookup.
    ///
    /// Owned via `Box::into_raw`; allocated once at construction sized
    /// to `code.names.len()`, never resized.  `null` when `code_ptr`
    /// is null or unaligned (test fixtures, gateway builtins).
    pub globals_caches: *mut std::sync::Mutex<
        Vec<Option<std::sync::Weak<std::sync::Mutex<pyre_object::celldict::GlobalCache>>>>,
    >,
    /// `mapdict.py self._mapdict_caches = [INVALID_CACHE_ENTRY] *
    /// len(co_names_w)`.
    ///
    /// Per-name slot for the `LOAD_ATTR_caching` / `STORE_ATTR_caching` inline
    /// attribute cache (`mapdict.py:1480/1574`).  A `None` slot is PyPy's
    /// `INVALID_CACHE_ENTRY` (mapdict.py); a `Some` holds the immortal map
    /// node + attribute node + `version_tag` last resolved for this slot, so a
    /// monomorphic re-read skips the type lookup + map walk.  The
    /// LOAD_METHOD fill additionally stores a movable `w_method`
    /// reference (mapdict.py:1418), forwarded during collection by
    /// `walk_mapdict_method_cache_gc`; the other fields are immortal
    /// node pointers and need no walking.
    ///
    /// Owned via `Box::into_raw`, sized to `code.names.len()` at construction,
    /// never resized; `null` when `code_ptr` is null or unaligned.
    pub mapdict_caches: *mut Vec<Option<crate::objspace::std::mapdict::MapdictCacheEntry>>,
    /// `pycode.py self.co_consts_w = consts` (`_immutable_fields_
    /// co_consts_w[*]`, pycode.py:97).  The realized constant objects indexed by
    /// constant index.  `getconstant_w(index)` (`pyopcode.py`) returns
    /// `co_consts_w[index]`, so every `LOAD_CONST` yields the one shared object
    /// stored here — repeated loads (including blackhole resume through the
    /// same virtualizable `pycode`) preserve object identity.
    ///
    /// PyPy receives this list already wrapped from the compiler. Pyre's
    /// compiler keeps `ConstantData` unwrapped, so the `PyCode` constructor
    /// wraps every slot before publishing the code object. This is observable:
    /// `gc.get_objects()` must not gain a permanent object the first time a
    /// `LOAD_CONST` executes. `eval::walk_raw_code_roots` traces every slot
    /// because value constants (notably `W_LongObject`) are GC-managed.
    ///
    /// Owned via `Box::into_raw`, sized to `code.constants.len()` at construction,
    /// never resized. A `null` slot is reserved for unreadable test stubs or a
    /// defensive fallback after construction. The whole pointer is `null` when
    /// `code_ptr` is null or unaligned (test fixtures, gateway builtins).
    pub co_consts_w: *mut Vec<std::sync::atomic::AtomicPtr<PyObject>>,
    /// `pycode.py:127-129 self.co_names_w = [space.new_interned_str(aname) for
    /// aname in names]` (`_immutable_fields_ co_names_w[*]`, pycode.py:100).
    /// The realized name objects indexed by name index.  `getname_w(index)`
    /// (`pyopcode.py`) returns `co_names_w[index]`, so every opcode
    /// needing a wrapped name hands back the one object this code object owns
    /// rather than minting a `W_UnicodeObject` per execution — the identity
    /// argument of `w_qualname` below, applied per name index.
    ///
    /// PyPy interns the whole list in the constructor.  Pyre realizes slots
    /// lazily at the same wrapped/unwrapped compiler boundary `co_consts_w`
    /// uses, so a name that never executes costs nothing.
    ///
    /// Slots hold `intern_str_value` results — `malloc_typed`-immortal, so a
    /// published pointer is fixed and the table needs no walking: there is
    /// nothing to forward and nothing whose liveness a trace could decide.
    /// Interning is also what keeps the immortality affordable: the canonical
    /// object is shared by every code object naming the same value, so a lost
    /// publish race abandons nothing — both racers hold the same object.
    ///
    /// Owned via `Box::into_raw`, sized to `code.names.len()` at construction,
    /// never resized; a `null` slot is unrealized.  The whole pointer is `null`
    /// when `code_ptr` is null or unaligned (test fixtures, gateway builtins).
    pub co_names_w: *mut Vec<std::sync::atomic::AtomicPtr<PyObject>>,
    /// `pycode.py self.co_qualname = qualname` realized as one shared
    /// wrapped object.
    ///
    /// `function.py self.qualname = qualname or self.name` copies the code
    /// object's interp-level string into every function built from it, so all
    /// of them name the same immutable string. Pyre wraps names as objects, so
    /// the code object owns the single realized instance and both
    /// `MAKE_FUNCTION` and the `co_qualname` getter hand back that object
    /// instead of allocating a fresh `W_UnicodeObject` per read.
    ///
    /// `PY_NULL` until first realized by [`w_code_qualname_obj`], which builds
    /// it with `w_str_new` — `malloc_typed`-immortal, so the stored pointer is
    /// fixed.  `eval::walk_raw_code_roots` forwards the slot anyway, for the
    /// same reason it forwards `w_globals`: the walker serves both the managed
    /// custom trace and the bootstrap prebuilt-root walk.
    pub w_qualname: PyObjectRef,
    /// `pycode.py self.co_name = name` realized as one shared wrapped
    /// object, the exact counterpart of `w_qualname` above.
    ///
    /// `function.py self.name = forcename or code.co_name` copies the code
    /// object's interp-level string into every function built from it, so the
    /// same "all of them name one immutable string" argument applies, and the
    /// same realize-once treatment follows.  Without it the getter ran
    /// `w_str_new` per read — two allocations, a `String` copy and a full
    /// `chars().count()` — which a traceback walk pays on every hop
    /// (`tb.tb_frame.f_code.co_name`).
    ///
    /// `PY_NULL` until first realized by [`w_code_name_obj`]; the storage,
    /// immortality and root-walk argument are `w_qualname`'s verbatim.
    pub w_name: PyObjectRef,
}

/// Field offset of `code_ptr` within `PyCode`.
pub const CODE_PTR_OFFSET: usize = std::mem::offset_of!(PyCode, code_ptr);
/// Field offset of `w_globals` within `PyCode`.
pub const CODE_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(PyCode, w_globals);
/// Field offset of `w_weakreflifeline` within `PyCode`.
pub const CODE_W_WEAKREFLIFELINE_OFFSET: usize = std::mem::offset_of!(PyCode, w_weakreflifeline);
/// Field offset of `w_qualname` within `PyCode`.
pub const CODE_W_QUALNAME_OFFSET: usize = std::mem::offset_of!(PyCode, w_qualname);
/// Field offset of `w_name` within `PyCode`.
pub const CODE_W_NAME_OFFSET: usize = std::mem::offset_of!(PyCode, w_name);
/// Field offset of `co_firstlineno_raw` within `PyCode`.
pub const CODE_CO_FIRSTLINENO_RAW_OFFSET: usize = std::mem::offset_of!(PyCode, co_firstlineno_raw);
/// Field offset of `hidden_applevel` within `PyCode`.
pub const CODE_HIDDEN_APPLEVEL_OFFSET: usize = std::mem::offset_of!(PyCode, hidden_applevel);

/// The `co_firstlineno` slot, exactly as [`code_get_field`] reads it.
///
/// # Safety
/// `w_code` must be a live `PyCode`.
pub unsafe fn w_code_firstlineno_raw(w_code: PyObjectRef) -> i32 {
    unsafe { (*(w_code as *const PyCode)).co_firstlineno_raw }
}

/// `make_weakref_descr(PyCode).getweakref` — read the owner-local lifeline.
///
/// # Safety
/// `w_code` must point to a live [`PyCode`].
pub unsafe fn w_code_getweakref(w_code: PyObjectRef) -> PyObjectRef {
    unsafe { (*(w_code as *const PyCode)).w_weakreflifeline }
}

/// `make_weakref_descr(PyCode).setweakref` — publish the lifeline and retain
/// the old-to-young edge when a stable-oldgen code object receives it.
///
/// # Safety
/// `w_code` must point to a live [`PyCode`].
pub unsafe fn w_code_setweakref(w_code: PyObjectRef, lifeline: PyObjectRef) {
    unsafe {
        (*(w_code as *mut PyCode)).w_weakreflifeline = lifeline;
        pyre_object::gc_hook::try_gc_write_barrier(w_code as *mut u8);
    }
}

/// `pycode.py self.co_qualname = qualname` — the shared wrapped qualified
/// name, realized on first demand and retained on the code object.
///
/// `function.py self.qualname = qualname or self.name` and the `co_qualname`
/// getter both hand out the code object's own string, so they name one
/// immutable value; realizing it once here reproduces that identity instead of
/// minting a `W_UnicodeObject` per `def` and per attribute read.  Returns
/// `PY_NULL` for a wrapper with no code body (test stubs, gateway builtins),
/// whose callers already handle the missing name.
///
/// # Safety
/// `w_code` must point to a valid `PyCode`.
pub unsafe fn w_code_qualname_obj(w_code: PyObjectRef) -> PyObjectRef {
    unsafe {
        let cached = (*(w_code as *const PyCode)).w_qualname;
        if !cached.is_null() {
            return cached;
        }
        let code_ptr = w_code_get_ptr(w_code) as *const crate::CodeObject;
        if code_ptr.is_null() {
            return pyre_object::PY_NULL;
        }
        // `w_str_new` is a collection point, but the wrapper is stable-address
        // and so never relocates; the fresh string is stored through the
        // still-valid `w_code` before any further allocation can sweep it.
        let w_qualname = w_str_new(&(*code_ptr).qualname);
        (*(w_code as *mut PyCode)).w_qualname = w_qualname;
        w_qualname
    }
}

/// `pycode.py self.co_name = name` — the shared wrapped name, realized on
/// first demand and retained on the code object.
///
/// The counterpart of [`w_code_qualname_obj`], for the same reason and with the
/// same lifetime argument: `function.py:51 self.name = forcename or
/// code.co_name` and the `co_name` getter both hand out the code object's own
/// string, so they name one immutable value.
///
/// # Safety
/// `w_code` must point to a valid `PyCode`.
pub unsafe fn w_code_name_obj(w_code: PyObjectRef) -> PyObjectRef {
    unsafe {
        let cached = (*(w_code as *const PyCode)).w_name;
        if !cached.is_null() {
            return cached;
        }
        let code_ptr = w_code_get_ptr(w_code) as *const crate::CodeObject;
        if code_ptr.is_null() {
            return pyre_object::PY_NULL;
        }
        // `w_str_new` is a collection point, but the wrapper is stable-address
        // and so never relocates; the fresh string is stored through the
        // still-valid `w_code` before any further allocation can sweep it.
        let w_name = w_str_new(&(*code_ptr).obj_name);
        (*(w_code as *mut PyCode)).w_name = w_name;
        w_name
    }
}

/// `pycode.py self.co_filename = filename` as an application-level object,
/// decoded with the filesystem encoding (`objspace.py newfilename`) so a
/// path byte with no UTF-8 spelling survives instead of folding to U+FFFD.
///
/// Unlike [`w_code_name_obj`] and [`w_code_qualname_obj`] this realizes a fresh
/// object per call rather than retaining one on the `PyCode`. Those two build
/// their string with `w_str_new`, whose address is fixed, so the slot they store
/// stays valid on its own; the filesystem decode returns a movable managed
/// string, which the `filename_bytes` field doc explains a retained slot cannot
/// hold unless it is traced and forwarded the way `w_globals` is.
///
/// # Safety
/// `w_code` must point to a valid `PyCode` whose `code_ptr` is live.
pub unsafe fn w_code_filename_obj(w_code: PyObjectRef) -> PyObjectRef {
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if pycode.filename_bytes.is_null() {
        let code = unsafe { &*(pycode.code_ptr as *const crate::CodeObject) };
        w_str_new(&code.source_path)
    } else {
        crate::gateway::fsdecode_filename_bytes(unsafe { &*pycode.filename_bytes })
    }
}

/// Bootstrap/prebuilt `PyCode` wrappers that own off-GC `w_globals` and
/// `co_consts_w` slots.
///
/// PyPy's `PyCode` is GC-managed, so ordinary graph tracing reaches these
/// fields even when a code object is stored directly in a module/container
/// without first becoming a function or frame. Only wrappers created before
/// the runtime GC hook is installed stay outside the collector; expose that
/// fallback family through one small insertion-ordered registry. Ordinary
/// runtime wrappers are managed stable-oldgen objects.
static PREBUILT_CODE_ROOTS: std::sync::OnceLock<std::sync::Mutex<Vec<usize>>> =
    std::sync::OnceLock::new();

fn register_prebuilt_code_root(code: PyObjectRef) {
    let roots = PREBUILT_CODE_ROOTS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut roots = roots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let identity = code as usize;
    if !roots.contains(&identity) {
        roots.push(identity);
    }
}

/// Retire a wrapper from the fallback prebuilt-root registry once its managed
/// old-generation allocation is reclaimed.  The registry also contains the
/// bootstrap `malloc_typed` family, which never reaches this destructor.
fn unregister_prebuilt_code_root(code: PyObjectRef) {
    let Some(roots) = PREBUILT_CODE_ROOTS.get() else {
        return;
    };
    let mut roots = roots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    roots.retain(|&identity| identity != code as usize);
}

/// Trace every bootstrap/prebuilt code wrapper exactly as PyPy traces every
/// live GC-managed `PyCode`. Every bootstrap wrapper is registered here, so
/// the raw walker reports direct fields just like a GC trace callback; it does
/// not need to recreate the collector's transitive mark walk.
pub(crate) fn walk_prebuilt_code_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    let Some(roots) = PREBUILT_CODE_ROOTS.get() else {
        return;
    };
    let roots = roots
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for &code in roots.iter() {
        unsafe { crate::eval::walk_raw_code_roots(code as PyObjectRef, visitor) };
    }
}

/// GC type id assigned to `PyCode`.
///
/// `PyCode` is a normal interpreter-level code object in PyPy
/// (`pycode.py class PyCode(eval.Code)`).  This tid is pinned by
/// a `debug_assert_eq!` in the pyre-jit type-registration sequence: the
/// `PyCode` `TypeInfo` is registered explicitly just before the
/// foreign-pytype loop, taking the slot directly after
/// the two `GcArray` tids (41 and 42 in that order).  Pre-registering it there (and
/// inserting `CODE_TYPE` into `pytype_to_tid`) makes the foreign loop
/// skip `CODE_TYPE`, so the net register-call count up to
/// `W_MODULE_DICT_GC_TYPE_ID = 48` is unchanged and no downstream tid
/// shifts.  The numeric value coincides with the dormant
/// `pytraceback::PYTRACEBACK_GC_TYPE_ID` constant, but `PyTraceback`
/// is still host-allocated and is never GC-registered, so tid 43 only
/// ever tags a `PyCode` at runtime and the two do not collide.
pub const W_CODE_GC_TYPE_ID: u32 = 43;

impl pyre_object::lltype::GcType for PyCode {
    fn type_id() -> u32 {
        W_CODE_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<PyCode>();
}

/// Compatibility helper for unpacking a tuple of strings.
pub fn unpack_text_tuple(_space: PyObjectRef, w_str_tuple: PyObjectRef) -> Vec<String> {
    let _ = (_space, w_str_tuple);
    Vec::new()
}

/// Compatibility API for building a signature-like object.
pub fn make_signature(_code: &PyCode) -> PyObjectRef {
    let _ = _code;
    pyre_object::w_none()
}

/// pycode.py _compute_args_as_cellvars
pub fn _compute_args_as_cellvars(
    varnames: &[String],
    cellvars: &[String],
    argcount: usize,
) -> Vec<isize> {
    let mut args_as_cellvars = Vec::new();
    for i in 0..cellvars.len() {
        let cellname = &cellvars[i];
        for j in 0..argcount {
            if *cellname == varnames[j] {
                while args_as_cellvars.len() < i {
                    args_as_cellvars.push(-1isize);
                }
                args_as_cellvars.push(j as isize);
            }
        }
    }
    args_as_cellvars
}

#[inline]
pub fn _code_const_eq(_space: PyObjectRef, w_a: PyObjectRef, w_b: PyObjectRef) -> bool {
    let _ = _space;
    std::ptr::eq(w_a, w_b)
}

#[inline]
pub fn _convert_const(_space: PyObjectRef, w_a: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    w_a
}

/// pypy/interpreter/pycode.py `PyCode.__init__`
/// (`hidden_applevel` field assignment, line 147).
///
/// ```python
/// def __init__(self, space, ..., hidden_applevel=False, magic=default_magic):
///     ...
///     self.hidden_applevel = hidden_applevel
/// ```
///
/// `w_code_new(code_ptr)` is the `hidden_applevel=False` default
/// shorthand; callers who need the flag set (mirroring PyPy's
/// `BuiltinCode` (gateway.py) / `ApplevelClass`
/// (gateway.py:1355) / `_continuation` entrypoint dummy
/// (interp_continuation.py:195)) construct via this entry point.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a permanently-live `CodeObject`,
/// either obtained via `Box::into_raw` or nested inside one such owner.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the body
/// boxes the `PyCode` through the prebuilt allocator and its per-name cache
/// tables through direct raw allocations. Residualise the whole
/// constructor — a `PyObjectRef` GCREF modelled by signature. Code objects are
/// built at import/compile time, never on a traced hot path.
///
/// `pycode.py` makes `PyCode` an ordinary GC object, so upstream traces
/// `w_globals` and the cache tables structurally through it. `malloc_typed`
/// here makes it immortal and non-moving instead, which nothing traces into —
/// so every GC-heap slot it owns needs an explicit root walker
/// ([`walk_w_globals_stamped_code_roots`], [`walk_mapdict_method_cache_gc`]).
/// Those registries retire when code objects become GC-managed.
#[majit_macros::dont_look_inside]
pub fn w_code_new_with_hidden_applevel(code_ptr: *const (), hidden_applevel: bool) -> PyObjectRef {
    // RPython pointer alignment idiom (`rpython/memory/gc/minimarkpage.py:159
    // ll_assert((nsize & (WORD-1)) == 0, "malloc: size is not aligned")`):
    // bitwise AND of `cast_ptr_to_int(p)` against `(power_of_two_align - 1)`
    // gives the misalignment residual.  `front::mir` lowers a `(Ref, Int)`
    // cast to the `cast_ptr_to_int` op, so casting through `i64` (not
    // `usize`) routes the pointer through the proper LL conversion.
    // `align_of::<T>()` is always a power of two — `& (align - 1)` is
    // equivalent to `% align` for power-of-two alignments and matches the
    // RPython pattern bit-for-bit.  The residual is computed once and shared
    // by every field initializer below.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    let code_ptr_aligned = !code_ptr.is_null() && (code_ptr as i64) & align_mask == 0;
    if code_ptr_aligned {
        // The expanded `locations` array is redundant with `linetable`; hand it
        // to [`code_locations`] to rebuild on the first reader that wants it.
        // `first_line_number` is the value the array was decoded against here;
        // a `co_firstlineno_raw` stamp that cannot be spelled as `OneIndexed`
        // corrects the record in `box_code_object_with_firstlineno`.
        let firstlineno_raw = unsafe { &*(code_ptr as *const crate::CodeObject) }
            .first_line_number
            .map(|line| line.get() as i32)
            .unwrap_or(1);
        release_code_locations(code_ptr as *mut crate::CodeObject, firstlineno_raw);
    }
    let mut fast_natural_arity = if !code_ptr_aligned {
        crate::gateway::HOPELESS
    } else {
        compute_flatcall(unsafe { &*(code_ptr as *const crate::CodeObject) })
    };
    if code_ptr_aligned
        && code_yields_inside_try(unsafe { &*(code_ptr as *const crate::CodeObject) })
    {
        fast_natural_arity |= YIELDS_INSIDE_TRY_BIT;
    }
    // `pycode.py self._globals_caches = [None] * len(self.co_names_w)`.
    let globals_caches = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<
            Option<std::sync::Weak<std::sync::Mutex<pyre_object::celldict::GlobalCache>>>,
        > = Vec::with_capacity(names_len);
        v.resize_with(names_len, || None);
        Box::into_raw(Box::new(std::sync::Mutex::new(v)))
    };
    // `mapdict.py self._mapdict_caches = [INVALID_CACHE_ENTRY] *
    // len(co_names_w)` — `None` is `INVALID_CACHE_ENTRY`.
    let mapdict_caches = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<Option<crate::objspace::std::mapdict::MapdictCacheEntry>> =
            Vec::with_capacity(names_len);
        v.resize_with(names_len, || None);
        Box::into_raw(Box::new(v))
    };
    // `pycode.py self.co_consts_w = consts` — allocate the wrapped-constant
    // table at the compiler/interpreter boundary. It is filled immediately
    // after the stable `PyCode` allocation below.
    let co_consts_w = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let consts_len = code_ref.constants.len();
        let mut v: Vec<std::sync::atomic::AtomicPtr<PyObject>> = Vec::with_capacity(consts_len);
        v.resize_with(consts_len, || {
            std::sync::atomic::AtomicPtr::new(std::ptr::null_mut())
        });
        Box::into_raw(Box::new(v))
    };
    // `pycode.py:127-129 self.co_names_w = [...]` — the realized-name table
    // sized to the name count, with slots filled lazily by `w_code_getname_w`.
    let co_names_w = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<std::sync::atomic::AtomicPtr<PyObject>> = Vec::with_capacity(names_len);
        v.resize_with(names_len, || {
            std::sync::atomic::AtomicPtr::new(std::ptr::null_mut())
        });
        Box::into_raw(Box::new(v))
    };
    let npure_cellvars = if !code_ptr_aligned {
        u32::MAX
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        crate::pyframe::npure_cellvars(code_ref) as u32
    };
    let co_firstlineno_raw = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        1
    } else {
        unsafe { &*(code_ptr as *const crate::CodeObject) }
            .first_line_number
            .map_or(1, |line| line.get() as i32)
    };
    let obj = PyCode {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(&CODE_TYPE),
        },
        code_ptr,
        co_firstlineno_raw,
        filename_bytes: std::ptr::null_mut(),
        co_code_bytes: std::ptr::null_mut(),
        filename_inherits_to_nested: false,
        w_globals: pyre_object::PY_NULL,
        w_weakreflifeline: pyre_object::PY_NULL,
        hidden_applevel,
        fast_natural_arity,
        npure_cellvars,
        globals_caches,
        mapdict_caches,
        co_consts_w,
        co_names_w,
        w_qualname: pyre_object::PY_NULL,
        w_name: pyre_object::PY_NULL,
    };
    // PyPy's `PyCode` is an ordinary GC object.  Keep the address stable for
    // the raw `code_ptr -> wrapper` JIT seam, but let the collector reclaim
    // the wrapper so its `w_globals` field participates in cycles instead of
    // making every `exec` namespace process-global.  Before GC installation,
    // `malloc_typed_stable` falls back to the prebuilt family and the explicit
    // root registry remains necessary.
    let obj = pyre_object::lltype::malloc_typed_stable(obj) as PyObjectRef;
    // PyPy's ast compiler has already wrapped every entry before PyCode.__init__
    // (`assemble.py:479-492`). Pin the freshly allocated stable wrapper while
    // recursive code constants and managed scalar constants allocate, then
    // publish one object in every co_consts_w slot before returning PyCode.
    let _roots = pyre_object::gc_roots::push_roots();
    let obj_slot = pyre_object::gc_roots::shadow_stack_len();
    pyre_object::gc_roots::pin_root(obj);
    // The shadow-stack root forwards the wrapper itself; only the raw walker
    // driven from this registry reaches its `co_consts_w` slots. Enrol an
    // off-GC wrapper before the fill loop, or a collection triggered by a
    // later constant reclaims the constants already published into it. A
    // managed wrapper is traced from its own allocation and stays out.
    if !pyre_object::gc_hook::try_gc_owns_object(obj as *mut u8) {
        register_prebuilt_code_root(obj);
    }
    if code_ptr_aligned {
        let consts_len = unsafe { &*(code_ptr as *const crate::CodeObject) }
            .constants
            .len();
        for index in 0..consts_len {
            unsafe { w_code_const(pyre_object::gc_roots::shadow_stack_get(obj_slot), index) };
        }
    }
    pyre_object::gc_roots::shadow_stack_get(obj_slot)
}

/// pypy/interpreter/pycode.py `PyCode.__init__` shorthand —
/// equivalent to PyPy `hidden_applevel=False` default
/// (pycode.py:111).  Most user-level pycode constructions take this
/// path; only the gateway / continuation / `__pypy__.hidden_applevel`
/// surfaces flip the flag to `True`.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a `CodeObject` that is never freed
/// and never moved. `pycode_destructor` releases the wrapper's side tables
/// but never `code_ptr`, so a leaked box and a constant reached through one
/// both qualify.
pub fn w_code_new(code_ptr: *const ()) -> PyObjectRef {
    w_code_new_with_hidden_applevel(code_ptr, false)
}

/// `generator.py:24-27` — read the code object's cached
/// `CO_YIELD_INSIDE_TRY` equivalent.
///
/// # Safety
/// `w_code` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_yields_inside_try(w_code: PyObjectRef) -> bool {
    unsafe { (*(w_code as *const PyCode)).fast_natural_arity & YIELDS_INSIDE_TRY_BIT != 0 }
}

/// Box a compiler code object the caller owns into a heap Python code wrapper.
///
/// PyPy's compiler constructs `PyCode` directly (`pycode.py`) and
/// therefore has no foreign compiler-object in the translated graph. Pyre's
/// compiler-core `CodeObject` is a dependency ADT that recursively owns boxed
/// slices and nested constants; it is solely the serialization/API seam used
/// to reach the interpreter-level `PyCode`. Residualize that foreign seam so
/// the translated graph retains PyPy's direct `PyCode` value shape instead of
/// inventing an RPython layout for the opaque Rust value. Keep Box ownership
/// transfer and wrapper publication in the one boundary.
///
/// The object is leaked on purpose: code wrappers are immortal and
/// `pycode_destructor` never frees `code_ptr`.
///
/// `#[dont_look_inside]` (`@jit.dont_look_inside`, `rlib/jit.py`): the body
/// `Box::into_raw`s a `CodeObject` (unlifted raw allocation) before forwarding
/// to the residualised `w_code_new`.
#[majit_macros::dont_look_inside]
pub fn box_code_object(code: crate::CodeObject) -> PyObjectRef {
    let code_ptr = Box::into_raw(Box::new(code)) as *const ();
    w_code_new(code_ptr)
}

/// [`box_code_object`] for a caller that only has a borrow, which has to copy.
/// A caller that owns its `CodeObject` should hand it over instead — the copy
/// is a whole recursive duplicate of the constants graph.
#[majit_macros::dont_look_inside]
pub fn box_code_constant(code: &crate::CodeObject) -> PyObjectRef {
    box_code_object(code.clone())
}

/// Publish a code wrapper over a `CodeObject` the caller keeps alive, rather
/// than over a copy of it.
///
/// [`box_code_constant`] copies because most callers hold a `CodeObject` on
/// the stack. A nested compiler constant is not one of those: it sits in its
/// enclosing `CodeObject`'s constants table behind its own `Box`, that table
/// is only ever edited in place, and the enclosing object is itself leaked.
/// So the constant already outlives every wrapper published for it.
///
/// # Safety
/// `code` must point to a `CodeObject` that is never freed and never moved.
unsafe fn box_code_constant_in_place(code: *const crate::CodeObject) -> PyObjectRef {
    w_code_new(code as *const ())
}

/// Wrap a nested compiler constant and inherit the enclosing `PyCode`'s raw
/// filename when it belongs to the set selected by
/// `importing.py update_code_filenames`' `oldname` guard.
///
/// # Safety
/// `code` must satisfy [`box_code_constant_in_place`], and `parent` must be
/// the `PyCode` whose constants table holds it.
unsafe fn box_code_constant_inheriting_filename(
    code: *const crate::CodeObject,
    parent: &PyCode,
) -> PyObjectRef {
    let obj = unsafe { box_code_constant_in_place(code) };
    if parent.filename_inherits_to_nested
        && unsafe { &*code }.source_path
            == unsafe { &*(parent.code_ptr as *const crate::CodeObject) }.source_path
        && !parent.filename_bytes.is_null()
    {
        let bytes = unsafe { (&*parent.filename_bytes).clone() };
        unsafe { set_filename_bytes(obj, Some(bytes)) };
        unsafe { (*(obj as *mut PyCode)).filename_inherits_to_nested = true };
    }
    obj
}

/// Attach the filesystem bytes a whole compilation unit was named with.
///
/// `compiling.py filename='fsencode'` names the unit, not one object, so
/// recurse through the already-wrapped constants exactly like
/// `importing.py update_code_filenames`. That is the difference from
/// the code constructor and `replace`, whose filename changes only the object
/// being built and leaves nested constants on the name they compiled under.
pub(crate) unsafe fn set_compilation_unit_filename_bytes(
    w_code: PyObjectRef,
    bytes: Option<Vec<u8>>,
) {
    let old_filename = unsafe { code_filename_bytes(w_code) };
    if let Some(bytes) = bytes.as_ref() {
        let pycode = unsafe { &*(w_code as *const PyCode) };
        if !pycode.co_consts_w.is_null() {
            for slot in unsafe { &*pycode.co_consts_w } {
                let nested = slot.load(std::sync::atomic::Ordering::Acquire);
                if !nested.is_null()
                    && unsafe { is_code(nested) }
                    && unsafe { code_filename_bytes(nested) } == old_filename
                {
                    unsafe { set_compilation_unit_filename_bytes(nested, Some(bytes.clone())) };
                }
            }
        }
    }
    let inherits = bytes.is_some();
    unsafe { set_filename_bytes(w_code, bytes) };
    unsafe { (*(w_code as *mut PyCode)).filename_inherits_to_nested = inherits };
}

/// Replace the owned raw filename allocation. Code wrappers are immortal, so
/// this is also the only point that retires an earlier spelling after a second
/// `_fix_co_filename` call.
unsafe fn set_filename_bytes(obj: PyObjectRef, bytes: Option<Vec<u8>>) {
    let slot = unsafe { &mut (*(obj as *mut PyCode)).filename_bytes };
    if let Some(bytes) = bytes {
        if slot.is_null() {
            *slot = Box::into_raw(Box::new(bytes));
        } else {
            unsafe { **slot = bytes };
        }
    } else if !slot.is_null() {
        unsafe { drop(Box::from_raw(*slot)) };
        *slot = std::ptr::null_mut();
    }
}

/// Replace the byte-exact `PyCode.co_code` fallback owned by `obj`.
///
/// A null slot means every opcode is representable by compiler-core and its
/// canonical `original_bytes()` is authoritative.
pub(crate) unsafe fn set_co_code_bytes(obj: PyObjectRef, bytes: Option<Vec<u8>>) {
    let slot = unsafe { &mut (*(obj as *mut PyCode)).co_code_bytes };
    if let Some(bytes) = bytes {
        if slot.is_null() {
            *slot = Box::into_raw(Box::new(bytes));
        } else {
            unsafe { **slot = bytes };
        }
    } else if !slot.is_null() {
        unsafe { drop(Box::from_raw(*slot)) };
        *slot = std::ptr::null_mut();
    }
}

/// Return the public, byte-exact `co_code` spelling for a live `PyCode`.
///
/// # Safety
/// `w_code` must point to a live [`PyCode`].
pub(crate) unsafe fn code_bytes(w_code: PyObjectRef) -> Vec<u8> {
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if pycode.co_code_bytes.is_null() {
        let code = unsafe { &*(pycode.code_ptr as *const crate::CodeObject) };
        code.instructions.original_bytes()
    } else {
        unsafe { (&*pycode.co_code_bytes).clone() }
    }
}

pub(crate) unsafe fn code_filename_bytes(w_code: PyObjectRef) -> Vec<u8> {
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if pycode.filename_bytes.is_null() {
        let code = unsafe { &*(pycode.code_ptr as *const crate::CodeObject) };
        code.source_path.as_bytes().to_vec()
    } else {
        unsafe { (&*pycode.filename_bytes).clone() }
    }
}

fn box_code_object_with_firstlineno(code: crate::CodeObject, firstlineno: i32) -> PyObjectRef {
    let obj = box_code_object(code);
    unsafe {
        (*(obj as *mut PyCode)).co_firstlineno_raw = firstlineno;
    }
    // `w_code_new` recorded `first_line_number`, which drops the zero and
    // negative values this stamp carries; re-record the exact one so the
    // released rows decode against what the array was built from.
    let code_ptr = unsafe { w_code_get_ptr(obj) } as *mut crate::CodeObject;
    if !code_ptr.is_null() {
        record_deferred_locations_firstlineno(code_ptr, firstlineno);
    }
    obj
}

/// Fill a newly-created code object's `co_consts_w` from the wrapped tuple
/// supplied to `CodeType.__new__` / `code.replace`.
///
/// PyPy passes this list straight into `PyCode.__init__`
/// (`pycode.py self.co_consts_w = consts`), preserving every wrapper's
/// identity. Pyre additionally serializes the values into compiler
/// `ConstantData` for bytecode decoding, but that representation must not
/// replace the interpreter-level owner.
unsafe fn w_code_fill_consts_from_tuple(obj: PyObjectRef, constants: PyObjectRef) {
    let code = unsafe { &*(obj as *const PyCode) };
    if code.co_consts_w.is_null() {
        return;
    }
    let slots = unsafe { &*code.co_consts_w };
    let count = slots.len().min(pyre_object::w_tuple_len(constants));
    let mut filled = false;
    for (index, slot) in slots.iter().take(count).enumerate() {
        if let Some(value) = unsafe { pyre_object::w_tuple_getitem(constants, index as i64) } {
            slot.store(value, std::sync::atomic::Ordering::Release);
            filled = true;
        }
    }
    if filled {
        publish_code_slot_store(obj);
    }
}

/// Record a store into one of the Rust-side tables a `PyCode` owns —
/// `co_consts_w`, `w_globals`, the mapdict method cache.
///
/// Those tables are not collector objects: only the wrapper's custom trace
/// reaches them, so a store there is a store into the wrapper. A prebuilt
/// wrapper is covered by the root walk, which clean minor collections skip,
/// hence `mark_prebuilt_roots_dirty`. A managed wrapper is not in that
/// registry at all — `w_code_new` registers a code object only when the
/// collector does not already own it — so a tenured wrapper that now points at
/// a nursery constant needs its remembered-set entry back, which is what the
/// write barrier restores. Without it the next minor collection never traces
/// the slot and leaves the wrapper holding a stale pointer.
#[inline]
fn publish_code_slot_store(obj: PyObjectRef) {
    if obj.is_null() {
        return;
    }
    pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8);
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
}

/// `w_code_fill_consts_from_tuple` for the marshal reader, whose `co_consts`
/// arrive as decoded objects rather than as a tuple. PyPy's marshal reader
/// passes the complete wrapped list to `PyCode.__init__`; replace every eager
/// compiler-boundary placeholder with that authoritative decoded object.
pub(crate) unsafe fn w_code_fill_wrapped_consts(obj: PyObjectRef, constants: &[PyObjectRef]) {
    let code = unsafe { &*(obj as *const PyCode) };
    if code.co_consts_w.is_null() {
        return;
    }
    let slots = unsafe { &*code.co_consts_w };
    let count = slots.len().min(constants.len());
    for index in 0..count {
        slots[index].store(constants[index], std::sync::atomic::Ordering::Release);
    }
    if count != 0 {
        publish_code_slot_store(obj);
    }
}

/// Preserve the existing wrapped constant array when `code.replace()` changes
/// fields other than `co_consts`. PyPy copies its already-wrapped list; pyre
/// copies those same eager slot identities into the newly built wrapper.
unsafe fn w_code_copy_const_slots(dst: PyObjectRef, src: PyObjectRef) {
    let dst_code = unsafe { &*(dst as *const PyCode) };
    let src_code = unsafe { &*(src as *const PyCode) };
    if dst_code.co_consts_w.is_null() || src_code.co_consts_w.is_null() {
        return;
    }
    let dst_slots = unsafe { &*dst_code.co_consts_w };
    let src_slots = unsafe { &*src_code.co_consts_w };
    let mut copied = false;
    for (dst_slot, src_slot) in dst_slots.iter().zip(src_slots.iter()) {
        let value = src_slot.load(std::sync::atomic::Ordering::Acquire);
        if !value.is_null() {
            dst_slot.store(value, std::sync::atomic::Ordering::Release);
            copied = true;
        }
    }
    if copied {
        publish_code_slot_store(dst);
    }
}

/// The keyword-only fields `code.replace` accepts, in the order
/// `pypy/interpreter/pycode.py:77-81` reconstructs the code object.
const REPLACE_KWARGS: [&str; 18] = [
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_firstlineno",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_filename",
    "co_name",
    "co_qualname",
    "co_linetable",
    "co_exceptiontable",
];

#[inline]
unsafe fn require_code(
    obj: PyObjectRef,
    descriptor: &str,
) -> Result<&'static crate::CodeObject, crate::PyError> {
    if obj.is_null() || !unsafe { is_code(obj) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{descriptor}' requires a 'code' object"
        )));
    }
    let ptr = unsafe { w_code_get_ptr(obj) } as *const crate::CodeObject;
    if ptr.is_null() || !(ptr as usize).is_multiple_of(std::mem::align_of::<crate::CodeObject>()) {
        return Err(crate::PyError::type_error("code object has no code body"));
    }
    Ok(unsafe { &*ptr })
}

/// The `co_names` / `co_varnames` / `co_freevars` / `co_cellvars` getters.
///
/// Interned for `pycode.py space.new_interned_str(aname)`' reason: the
/// entries name values, so every code object spelling one must hand back the
/// same object rather than a fresh immortal string per read.
fn names_tuple(names: &[String]) -> PyObjectRef {
    w_tuple_new(
        names
            .iter()
            .map(|name| pyre_object::unicodeobject::intern_str_value(name))
            .collect(),
    )
}

fn constants_tuple(obj: PyObjectRef, code: &crate::CodeObject) -> PyObjectRef {
    let mut constants = Vec::with_capacity(code.constants.len());
    for (index, constant) in crate::pyframe::code_constants(code).iter().enumerate() {
        let value = unsafe { w_code_const(obj, index) };
        constants.push(if value.is_null() {
            crate::pyframe::pyobject_from_constant(constant)
        } else {
            value
        });
    }
    w_tuple_new(constants)
}

fn legacy_lnotab(code: &crate::CodeObject, firstlineno: i64) -> Vec<u8> {
    fn encode_pair(mut address: usize, mut line: i64, out: &mut Vec<u8>) {
        while address > 255 {
            out.extend_from_slice(&[255, 0]);
            address -= 255;
        }
        while line < -128 {
            out.extend_from_slice(&[address as u8, 128]);
            line += 128;
            address = 0;
        }
        while line > 127 {
            out.extend_from_slice(&[address as u8, 127]);
            line -= 127;
            address = 0;
        }
        out.extend_from_slice(&[address as u8, line as i8 as u8]);
    }

    let mut out = Vec::new();
    let mut line = firstlineno;
    let mut start_offset = 0usize;
    let mut range = PyCodeAddressRange::new(
        &code.linetable,
        firstlineno.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
    );
    while range.advance() {
        // CPython's `decode_linetable` follows the computed line rather than
        // `ar_line`, so a NO_LOCATION range does not manufacture a -1 delta.
        let next_line = range.computed_line as i64;
        if next_line != line {
            let offset = range.ar_start as usize;
            encode_pair(offset - start_offset, next_line - line, &mut out);
            line = next_line;
            start_offset = offset;
        }
    }
    out
}

/// `PyCode.typedef` field getters. Each type-dict descriptor delegates here so
/// the object carries one authoritative compiler `CodeObject`, matching
/// `pycode.py`'s direct `co_*` attributes rather than a parallel side table.
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_get_field(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    if name == "co_lnotab" {
        // CPython 3.14 `code_getlnotab`: issue the warning before decoding or
        // allocating the bytes result, and propagate warnings-as-errors.
        unsafe { require_code(obj, name)? };
        // Warning dispatch can run arbitrary Python, so it is a collection
        // point.  `obj` arrives as a plain Rust local, which no root walker
        // scans: pin it on the shadow stack and read it back afterwards, then
        // reacquire the immutable CodeObject view rather than retaining a Rust
        // borrow across the re-entrant boundary.
        let _roots = pyre_object::gc_roots::push_roots();
        let obj_slot = pyre_object::gc_roots::shadow_stack_len();
        pyre_object::gc_roots::pin_root(obj);
        crate::warn::warn_deprecation("co_lnotab is deprecated, use co_lines instead.")?;
        let obj = pyre_object::gc_roots::shadow_stack_get(obj_slot);
        let code = unsafe { require_code(obj, name)? };
        return Ok(pyre_object::bytesobject::w_bytes_from_bytes(
            &legacy_lnotab(code, unsafe {
                (*(obj as *const PyCode)).co_firstlineno_raw as i64
            }),
        ));
    }
    let code = unsafe { require_code(obj, name)? };
    Ok(match name {
        "co_argcount" => w_int_new(code.arg_count as i64),
        "co_posonlyargcount" => w_int_new(code.posonlyarg_count as i64),
        "co_kwonlyargcount" => w_int_new(code.kwonlyarg_count as i64),
        "co_nlocals" => w_int_new(code.varnames.len() as i64),
        "co_stacksize" => w_int_new(code.max_stackdepth as i64),
        "co_flags" => w_int_new(code.flags.bits() as i64),
        "co_code" | "_co_code_adaptive" => {
            pyre_object::bytesobject::w_bytes_from_bytes(&unsafe { code_bytes(obj) })
        }
        "co_consts" => constants_tuple(obj, code),
        "co_names" => names_tuple(&code.names),
        "co_varnames" => names_tuple(&code.varnames),
        "co_freevars" => names_tuple(&code.freevars),
        "co_cellvars" => names_tuple(&code.cellvars),
        "co_filename" => unsafe { w_code_filename_obj(obj) },
        // Shared realized object, like `co_qualname` below.
        "co_name" => unsafe { w_code_name_obj(obj) },
        // The realized qualname is shared with every function built from this
        // code object (`function.py self.qualname = qualname or self.name`),
        // so the attribute yields the same object on each read.
        "co_qualname" => unsafe { w_code_qualname_obj(obj) },
        "co_firstlineno" => w_int_new((*(obj as *const PyCode)).co_firstlineno_raw as i64),
        "co_linetable" => pyre_object::bytesobject::w_bytes_from_bytes(&code.linetable),
        "co_exceptiontable" => pyre_object::bytesobject::w_bytes_from_bytes(&code.exceptiontable),
        _ => {
            return Err(crate::PyError::attribute_error(format!(
                "'code' object has no attribute '{name}'"
            )));
        }
    })
}

/// CPython 3.14 `code.__new__` positional-only constructor, with the PyPy
/// `descr_code__new__` validations and field order adjusted to 3.14 (the
/// exception table precedes freevars/cellvars).
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !(17..=19).contains(&args.len()) {
        return Err(crate::PyError::type_error(format!(
            "code expected at least 16 arguments, got {}",
            args.len().saturating_sub(1),
        )));
    }
    let argcount = unsafe { read_code_u32(args[1], "argcount")? };
    let posonly = unsafe { read_code_u32(args[2], "posonlyargcount")? };
    let kwonly = unsafe { read_code_u32(args[3], "kwonlyargcount")? };
    let nlocals = unsafe { read_code_u32(args[4], "nlocals")? };
    let stacksize_value = unsafe { read_code_c_int(args[5])? };
    let flags_value = unsafe { read_code_c_int(args[6])? };
    if stacksize_value < 0 || flags_value < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Objects/codeobject.c: bad argument to internal function",
        ));
    }
    let stacksize = stacksize_value as u32;
    let flags_bits = flags_value as u32;
    let (instructions, co_code_bytes) = unsafe { read_code_units(args[7])? };
    let constants = unsafe { read_code_consts(args[8])? };
    let names = unsafe { read_code_names(args[9], "names")? };
    let varnames = unsafe { read_code_names(args[10], "varnames")? };
    if varnames.len() != nlocals as usize {
        return Err(crate::PyError::value_error(
            "code: co_nlocals != len(co_varnames)".to_string(),
        ));
    }
    let (source_path, filename_bytes) = unsafe { read_code_filename(args[11], "filename", None)? };
    let obj_name = unsafe { read_code_str(args[12], "name")? };
    let qualname = unsafe { read_code_str(args[13], "qualname")? };
    let first_line = unsafe { read_code_c_int(args[14])? } as i64;
    let first_line_number = if first_line <= 0 {
        None
    } else {
        rustpython_compiler_core::OneIndexed::new(first_line as usize)
    };
    let linetable = unsafe { read_code_bytes(args[15], "linetable")? };
    let exceptiontable = unsafe { read_code_bytes(args[16], "exceptiontable")? };
    let freevars = if args.len() >= 18 {
        unsafe { read_code_names(args[17], "freevars")? }
    } else {
        Vec::<String>::new().into_boxed_slice()
    };
    let cellvars = if args.len() >= 19 {
        unsafe { read_code_names(args[18], "cellvars")? }
    } else {
        Vec::<String>::new().into_boxed_slice()
    };
    if argcount + kwonly > nlocals || posonly > argcount {
        return Err(crate::PyError::value_error("code: invalid argument count"));
    }

    // CPython's localsplus table stores cell aliases on the local slot and
    // appends only pure cells, followed by free variables.
    let mut localspluskinds = vec![crate::bytecode::CO_FAST_LOCAL; varnames.len()];
    for cell in cellvars.iter() {
        if let Some(index) = varnames.iter().position(|name| name == cell) {
            localspluskinds[index] |= crate::bytecode::CO_FAST_CELL;
        } else {
            localspluskinds.push(crate::bytecode::CO_FAST_CELL);
        }
    }
    localspluskinds.extend(std::iter::repeat_n(
        crate::bytecode::CO_FAST_FREE,
        freevars.len(),
    ));

    let locations = rustpython_compiler_core::marshal::linetable_to_locations(
        &linetable,
        first_line.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        instructions.len(),
    );
    let code = crate::CodeObject {
        instructions,
        locations,
        flags: crate::bytecode::CodeFlags::from_bits_retain(flags_bits),
        posonlyarg_count: posonly,
        arg_count: argcount,
        kwonlyarg_count: kwonly,
        source_path,
        first_line_number,
        max_stackdepth: stacksize,
        obj_name,
        qualname,
        constants,
        names,
        varnames,
        cellvars,
        freevars,
        localspluskinds: localspluskinds.into_boxed_slice(),
        linetable,
        exceptiontable,
    };
    let result = box_code_object_with_firstlineno(
        code,
        first_line.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
    );
    unsafe { set_filename_bytes(result, filename_bytes) };
    unsafe { set_co_code_bytes(result, co_code_bytes) };
    unsafe { w_code_fill_consts_from_tuple(result, args[8]) };
    Ok(result)
}

fn code_data_equal(a: &crate::CodeObject, b: &crate::CodeObject) -> bool {
    a.obj_name == b.obj_name
        && a.qualname == b.qualname
        && a.arg_count == b.arg_count
        && a.posonlyarg_count == b.posonlyarg_count
        && a.kwonlyarg_count == b.kwonlyarg_count
        && a.varnames.len() == b.varnames.len()
        && a.flags == b.flags
        && a.first_line_number == b.first_line_number
        && a.instructions.original_bytes() == b.instructions.original_bytes()
        && a.names.len() == b.names.len()
        && a.constants.len() == b.constants.len()
        && a.varnames == b.varnames
        && a.freevars == b.freevars
        && a.cellvars == b.cellvars
        && a.names == b.names
        && a.linetable == b.linetable
        && a.exceptiontable == b.exceptiontable
        && crate::pyframe::code_constants(a)
            .iter()
            .zip(crate::pyframe::code_constants(b).iter())
            .all(|(left, right)| constant_strong_equal(left, right))
}

fn constant_strong_equal(
    left: &crate::bytecode::ConstantData,
    right: &crate::bytecode::ConstantData,
) -> bool {
    use crate::bytecode::ConstantData;
    match (left, right) {
        (ConstantData::Code { code: a }, ConstantData::Code { code: b }) => code_data_equal(a, b),
        (ConstantData::Tuple { elements: a }, ConstantData::Tuple { elements: b }) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| constant_strong_equal(x, y))
        }
        (ConstantData::Slice { elements: a }, ConstantData::Slice { elements: b }) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| constant_strong_equal(x, y)),
        (ConstantData::Frozenset { elements: a }, ConstantData::Frozenset { elements: b }) => {
            if a.len() != b.len() {
                return false;
            }
            let mut matched = vec![false; b.len()];
            a.iter().all(|item| {
                b.iter()
                    .enumerate()
                    .find(|(index, candidate)| {
                        !matched[*index] && constant_strong_equal(item, candidate)
                    })
                    .map(|(index, _)| {
                        matched[index] = true;
                    })
                    .is_some()
            })
        }
        _ => left == right,
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_eq(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_code(other) } {
        return Ok(pyre_object::special::w_not_implemented());
    }
    let a = unsafe { require_code(this, "__eq__")? };
    let b = unsafe { require_code(other, "__eq__")? };
    if (*(this as *const PyCode)).co_firstlineno_raw
        != (*(other as *const PyCode)).co_firstlineno_raw
    {
        return Ok(w_bool_from(false));
    }
    if unsafe { code_bytes(this) } != unsafe { code_bytes(other) } {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(code_data_equal(a, b)))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_ne(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let equal = unsafe { code_eq(this, other)? };
    if unsafe { pyre_object::is_not_implemented(equal) } {
        Ok(equal)
    } else {
        Ok(w_bool_from(!unsafe { w_bool_get_value(equal) }))
    }
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_hash(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let code = unsafe { require_code(obj, "__hash__")? };
    #[inline]
    fn scramble(result: i64, value: i64) -> i64 {
        ((result as u64 ^ value as u64).wrapping_mul(1_000_003)) as i64
    }
    #[inline]
    fn add_obj(result: &mut i64, value: PyObjectRef) -> Result<(), crate::PyError> {
        *result = scramble(*result, crate::baseobjspace::hash_w_strict(value)?);
        Ok(())
    }
    let mut result = 20_250_211i64;
    add_obj(&mut result, w_str_new(&code.obj_name))?;
    add_obj(&mut result, w_str_new(&code.qualname))?;
    for value in [
        code.arg_count as i64,
        code.posonlyarg_count as i64,
        code.kwonlyarg_count as i64,
        code.varnames.len() as i64,
        code.flags.bits() as i64,
        (*(obj as *const PyCode)).co_firstlineno_raw as i64,
    ] {
        result = scramble(result, value);
    }
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&unsafe { code_bytes(obj) }),
    )?;
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.linetable),
    )?;
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.exceptiontable),
    )?;
    for names in [&code.varnames, &code.freevars, &code.cellvars, &code.names] {
        for name in names.iter() {
            add_obj(&mut result, w_str_new(name))?;
        }
    }
    for (index, constant) in crate::pyframe::code_constants(code).iter().enumerate() {
        let w_constant = unsafe { w_code_const(obj, index) };
        let w_constant = if w_constant.is_null() {
            crate::pyframe::pyobject_from_constant(constant)
        } else {
            w_constant
        };
        add_obj(&mut result, w_constant)?;
    }
    Ok(if result == -1 { -2 } else { result })
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_repr(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "__repr__")? };
    // pycode.py:570-572 represents the internal zero sentinel as line -1.
    let raw_line = (*(obj as *const PyCode)).co_firstlineno_raw as i64;
    let line = if raw_line == 0 { -1 } else { raw_line };
    let mut repr = rustpython_wtf8::Wtf8Buf::from_string(format!(
        "<code object {} at {}, file \"",
        code.obj_name,
        crate::display::repr_addr(obj as usize),
    ));
    let filename = crate::gateway::fsdecode_filename_wtf8(&unsafe { code_filename_bytes(obj) });
    repr.push_wtf8(&filename);
    repr.push_str(&format!("\", line {line}>"));
    Ok(pyre_object::w_str_from_wtf8_managed(repr))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_varname_from_oparg(
    obj: PyObjectRef,
    index: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "_varname_from_oparg")? };
    let mut index = unsafe { crate::builtins::space_index_w(index)? };
    if index >= 0 {
        // closure-free, Option-pattern-free `varnames.get` / `freevars.get`
        // rewrites — keep the bounds check a plain `lt + getitem`.
        let vi = index as usize;
        if vi < code.varnames.len() {
            return Ok(w_str_new(&code.varnames[vi]));
        }
        index -= code.varnames.len() as i64;
        let pure_cellvars = code
            .cellvars
            .iter()
            .filter(|cell| !code.varnames.iter().any(|var| var == *cell));
        let pure_cellvar_count = pure_cellvars.clone().count();
        if let Some(name) = pure_cellvars.skip(index as usize).next() {
            return Ok(w_str_new(name));
        }
        index -= pure_cellvar_count as i64;
        let fi = index as usize;
        if fi < code.freevars.len() {
            return Ok(w_str_new(&code.freevars[fi]));
        }
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::IndexError,
        "tuple index out of range",
    ))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_positions(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "co_positions")? };
    let mut rows = Vec::new();
    let mut reader = LineTableReader::new(&code.linetable);
    let mut line = unsafe { (*(obj as *const PyCode)).co_firstlineno_raw };

    while !reader.at_end() {
        let Some(first_byte) = reader.read_byte() else {
            break;
        };
        // Decoded as a header regardless of bit 7, for the reason given on
        // `PyCodeAddressRange::advance`.
        let code = (first_byte >> 3) & 0x0f;
        let length = ((first_byte & 0x07) + 1) as usize;
        let Some(kind) = PyCodeLocationInfoKind::from_code(code) else {
            break;
        };
        let (line_delta, end_line_delta, column, end_column) = match kind {
            PyCodeLocationInfoKind::None => (0, 0, None, None),
            PyCodeLocationInfoKind::Long => {
                let delta = reader.read_signed_varint();
                let end_line_delta = reader.read_varint() as i32;
                let column = match reader.read_varint() {
                    0 => None,
                    value => Some((value - 1) as i32),
                };
                let end_column = match reader.read_varint() {
                    0 => None,
                    value => Some((value - 1) as i32),
                };
                (delta, end_line_delta, column, end_column)
            }
            PyCodeLocationInfoKind::NoColumns => (reader.read_signed_varint(), 0, None, None),
            PyCodeLocationInfoKind::OneLine0
            | PyCodeLocationInfoKind::OneLine1
            | PyCodeLocationInfoKind::OneLine2 => {
                let column = reader.read_byte().unwrap_or(0) as i32;
                let end_column = reader.read_byte().unwrap_or(0) as i32;
                (
                    kind.one_line_delta().unwrap_or(0),
                    0,
                    Some(column),
                    Some(end_column),
                )
            }
            _ if kind.is_short() => {
                let column_data = reader.read_byte().unwrap_or(0);
                let column_group = kind.short_column_group().unwrap_or(0);
                let column = ((column_group as i32) << 3) | ((column_data >> 4) as i32);
                let end_column = column + (column_data & 0x0f) as i32;
                (0, 0, Some(column), Some(end_column))
            }
            _ => (0, 0, None, None),
        };
        line += line_delta;

        for _ in 0..length {
            let (line_obj, end_line_obj) = if kind == PyCodeLocationInfoKind::None {
                (pyre_object::w_none(), pyre_object::w_none())
            } else {
                (
                    w_int_new(line as i64),
                    w_int_new((line + end_line_delta) as i64),
                )
            };
            rows.push(w_tuple_new(vec![
                line_obj,
                end_line_obj,
                column.map_or_else(pyre_object::w_none, |value| w_int_new(value as i64)),
                end_column.map_or_else(pyre_object::w_none, |value| w_int_new(value as i64)),
            ]));
        }
    }
    let n = rows.len();
    Ok(w_seq_iter_new(w_list_new(rows), n))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_lines(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "co_lines")? };
    let mut rows = Vec::new();
    let first_line = unsafe { (*(obj as *const PyCode)).co_firstlineno_raw };
    let mut range = PyCodeAddressRange::new(&code.linetable, first_line);
    let mut pending: Option<(i32, i32, i32)> = None;

    while range.advance() {
        let start = range.ar_start;
        let end = range.ar_end;
        let line = range.ar_line;
        if let Some((previous_start, _, previous_line)) = pending {
            if previous_line == line {
                pending = Some((previous_start, end, previous_line));
            } else {
                rows.push(w_tuple_new(vec![
                    w_int_new(previous_start as i64),
                    w_int_new(start as i64),
                    if previous_line == -1 {
                        pyre_object::w_none()
                    } else {
                        w_int_new(previous_line as i64)
                    },
                ]));
                pending = Some((start, end, line));
            }
        } else {
            pending = Some((start, end, line));
        }
    }
    if let Some((start, end, line)) = pending {
        rows.push(w_tuple_new(vec![
            w_int_new(start as i64),
            w_int_new(end as i64),
            if line == -1 {
                pyre_object::w_none()
            } else {
                w_int_new(line as i64)
            },
        ]));
    }
    let n = rows.len();
    Ok(w_seq_iter_new(w_list_new(rows), n))
}

/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn code_branches(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "co_branches")? };
    let mut rows = Vec::new();
    let mut index = 0usize;
    let mut op_arg = 0usize;
    while index < code.instructions.len() {
        let op = code.instructions.read_op(index).deoptimize();
        let next = index + 1 + op.cache_entries();
        let arg = u8::from(code.instructions.read_arg(index)) as usize;
        match op {
            crate::bytecode::Instruction::ExtendedArg => {
                op_arg = (op_arg << 8) | arg;
            }
            crate::bytecode::Instruction::ForIter { .. } => {
                op_arg = (op_arg << 8) | arg;
                rows.push(w_tuple_new(vec![
                    w_int_new((index * 2) as i64),
                    w_int_new((next * 2) as i64),
                    w_int_new(((next + op_arg + 2) * 2) as i64),
                ]));
                op_arg = 0;
            }
            crate::bytecode::Instruction::PopJumpIfFalse { .. }
            | crate::bytecode::Instruction::PopJumpIfTrue { .. }
            | crate::bytecode::Instruction::PopJumpIfNone { .. }
            | crate::bytecode::Instruction::PopJumpIfNotNone { .. } => {
                op_arg = (op_arg << 8) | arg;
                // Python 3.14 inserts NOT_TAKEN at the fallthrough edge so
                // branch instrumentation can distinguish the untaken path.
                let not_taken = next + 1;
                rows.push(w_tuple_new(vec![
                    w_int_new((index * 2) as i64),
                    w_int_new((not_taken * 2) as i64),
                    w_int_new(((next + op_arg) * 2) as i64),
                ]));
                op_arg = 0;
            }
            crate::bytecode::Instruction::EndAsyncFor => {
                op_arg = (op_arg << 8) | arg;
                let source = next - op_arg;
                debug_assert!(matches!(
                    code.instructions.read_op(source).deoptimize(),
                    crate::bytecode::Instruction::EndSend
                ));
                debug_assert!(matches!(
                    code.instructions.read_op(source + 1).deoptimize(),
                    crate::bytecode::Instruction::NotTaken
                ));
                rows.push(w_tuple_new(vec![
                    w_int_new((source * 2) as i64),
                    w_int_new(((source + 2) * 2) as i64),
                    w_int_new((next * 2) as i64),
                ]));
                op_arg = 0;
            }
            _ => op_arg = 0,
        }
        index = next.max(index + 1);
    }
    let n = rows.len();
    Ok(w_seq_iter_new(w_list_new(rows), n))
}

/// `code.replace(**kwds)` — `pypy/interpreter/pycode.py` applevel
/// `replace`, which gathers every `co_*` attribute (taking the keyword
/// override where present) and reconstructs the code object through the
/// `CodeType` constructor.  pyre stores a compiler `CodeObject`, so the
/// equivalent is to clone it, override each supplied field, and re-box it.
///
/// # Safety
/// `args[0]` must be the receiver `code` object (verified).
pub unsafe fn code_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let w_self = pos.first().copied().unwrap_or(PY_NULL);
    if w_self.is_null() || !unsafe { is_code(w_self) } {
        return Err(crate::PyError::type_error(
            "descriptor 'replace' requires a 'code' object",
        ));
    }
    // `replace` is keyword-only (`__args__.topacked()` asserts no positional
    // args at pycode.py:548-549).
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(
            "replace() takes no positional arguments",
        ));
    }
    // pycode.py:86-87 `raise TypeError(f"{kwds.popitem()[0]!r} is an invalid
    // keyword argument for replace()")`.
    if let Some(dict) = kwargs {
        for (key, _) in unsafe { pyre_object::w_dict_str_entries(dict) } {
            if key == "__pyre_kw__" {
                continue;
            }
            if !REPLACE_KWARGS.contains(&key.as_str()) {
                return Err(crate::PyError::type_error(format!(
                    "replace() got an unexpected keyword argument '{key}'"
                )));
            }
        }
    }

    let code_ptr = unsafe { w_code_get_ptr(w_self) } as *const crate::CodeObject;
    if code_ptr.is_null() {
        return Err(crate::PyError::type_error(
            "cannot replace fields of a code object with no code body",
        ));
    }
    let mut code = unsafe { (*code_ptr).clone() };
    let mut firstlineno_raw = unsafe { (*(w_self as *const PyCode)).co_firstlineno_raw };
    let mut filename_bytes = unsafe {
        let ptr = (*(w_self as *const PyCode)).filename_bytes;
        if ptr.is_null() {
            None
        } else {
            Some((&*ptr).clone())
        }
    };
    let mut filename_inherits_to_nested =
        unsafe { (*(w_self as *const PyCode)).filename_inherits_to_nested };
    let mut co_code_bytes = unsafe {
        let ptr = (*(w_self as *const PyCode)).co_code_bytes;
        if ptr.is_null() {
            None
        } else {
            Some((&*ptr).clone())
        }
    };
    let get = |name: &str| crate::builtins::kwarg_get(kwargs, name);
    let rebuild_localspluskinds = get("co_varnames").is_some()
        || get("co_cellvars").is_some()
        || get("co_freevars").is_some();

    if let Some(v) = get("co_argcount") {
        code.arg_count = unsafe { read_code_u32(v, "co_argcount")? };
    }
    if let Some(v) = get("co_posonlyargcount") {
        code.posonlyarg_count = unsafe { read_code_u32(v, "co_posonlyargcount")? };
    }
    if let Some(v) = get("co_kwonlyargcount") {
        code.kwonlyarg_count = unsafe { read_code_u32(v, "co_kwonlyargcount")? };
    }
    let requested_nlocals = get("co_nlocals")
        .map(|v| unsafe { read_code_u32(v, "co_nlocals") })
        .transpose()?;
    if let Some(v) = get("co_stacksize") {
        code.max_stackdepth = unsafe { read_code_u32(v, "co_stacksize")? };
    }
    if let Some(v) = get("co_flags") {
        let value = unsafe { read_code_c_int(v)? };
        if value < 0 {
            return Err(crate::PyError::value_error(
                "co_flags must be a positive integer",
            ));
        }
        let bits = value as u32;
        code.flags = crate::bytecode::CodeFlags::from_bits_retain(bits);
    }
    if let Some(v) = get("co_firstlineno") {
        let n = unsafe { read_code_c_int(v)? } as i64;
        if n < 0 {
            return Err(crate::PyError::value_error(
                "co_firstlineno must be a positive integer",
            ));
        }
        firstlineno_raw = n.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        code.first_line_number = if n <= 0 {
            None
        } else {
            rustpython_compiler_core::OneIndexed::new(n as usize)
        };
    }
    if let Some(v) = get("co_name") {
        code.obj_name = unsafe { read_code_str(v, "co_name")? };
    }
    if let Some(v) = get("co_qualname") {
        code.qualname = unsafe { read_code_str(v, "co_qualname")? };
    }
    if let Some(v) = get("co_filename") {
        (code.source_path, filename_bytes) =
            unsafe { read_code_filename(v, "co_filename", Some(&code.source_path))? };
        filename_inherits_to_nested = false;
    }
    if let Some(v) = get("co_names") {
        code.names = unsafe { read_code_names(v, "co_names")? };
    }
    if let Some(v) = get("co_varnames") {
        code.varnames = unsafe { read_code_names(v, "co_varnames")? };
    }
    if let Some(v) = get("co_freevars") {
        code.freevars = unsafe { read_code_names(v, "co_freevars")? };
    }
    if let Some(v) = get("co_cellvars") {
        code.cellvars = unsafe { read_code_names(v, "co_cellvars")? };
    }
    if let Some(v) = get("co_linetable") {
        code.linetable = unsafe { read_code_bytes(v, "co_linetable")? };
    }
    if let Some(v) = get("co_exceptiontable") {
        code.exceptiontable = unsafe { read_code_bytes(v, "co_exceptiontable")? };
    }
    if let Some(v) = get("co_consts") {
        code.constants = unsafe { read_code_consts(v)? };
        filename_inherits_to_nested = false;
    }
    if let Some(v) = get("co_code") {
        (code.instructions, co_code_bytes) = unsafe { read_code_units(v)? };
    }

    if requested_nlocals.is_some_and(|n| n as usize != code.varnames.len()) {
        return Err(crate::PyError::value_error(
            "code: co_nlocals != len(co_varnames)",
        ));
    }
    if code.posonlyarg_count > code.arg_count {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Objects/codeobject.c: bad argument to internal function",
        ));
    }
    if code.arg_count as usize + code.kwonlyarg_count as usize > code.varnames.len() {
        return Err(crate::PyError::value_error(
            "code: co_varnames is too small",
        ));
    }

    // PyCode_Replace preserves the private locals-plus kind table when only
    // unrelated public fields change.  This matters for PEP 709's
    // CO_FAST_HIDDEN slots, which cannot be reconstructed from co_varnames,
    // co_cellvars and co_freevars.  Rebuild only when one of those public
    // layout fields was explicitly replaced.
    if rebuild_localspluskinds {
        let mut localspluskinds = vec![crate::bytecode::CO_FAST_LOCAL; code.varnames.len()];
        for cell in code.cellvars.iter() {
            if let Some(index) = code.varnames.iter().position(|name| name == cell) {
                localspluskinds[index] |= crate::bytecode::CO_FAST_CELL;
            } else {
                localspluskinds.push(crate::bytecode::CO_FAST_CELL);
            }
        }
        localspluskinds.extend(std::iter::repeat_n(
            crate::bytecode::CO_FAST_FREE,
            code.freevars.len(),
        ));
        code.localspluskinds = localspluskinds.into_boxed_slice();
    }
    code.locations = rustpython_compiler_core::marshal::linetable_to_locations(
        &code.linetable,
        firstlineno_raw,
        code.instructions.len(),
    );

    let result = box_code_object_with_firstlineno(code, firstlineno_raw);
    unsafe { set_filename_bytes(result, filename_bytes) };
    unsafe { set_co_code_bytes(result, co_code_bytes) };
    unsafe {
        (*(result as *mut PyCode)).filename_inherits_to_nested = filename_inherits_to_nested;
    }
    if let Some(constants) = get("co_consts") {
        unsafe { w_code_fill_consts_from_tuple(result, constants) };
    } else {
        unsafe { w_code_copy_const_slots(result, w_self) };
    }
    Ok(result)
}

/// A non-negative `co_*` count argument as `u32`.
unsafe fn read_code_u32(v: PyObjectRef, field: &str) -> Result<u32, crate::PyError> {
    let n = unsafe { read_code_c_int(v)? } as i64;
    if n < 0 {
        let message = if field.starts_with("co_") {
            format!("{field} must be a positive integer")
        } else {
            format!("code: {field} must not be negative")
        };
        return Err(crate::PyError::value_error(message));
    }
    Ok(n as u32)
}

/// Argument Clinic converts every public code-object integer through C `int`.
unsafe fn read_code_c_int(v: PyObjectRef) -> Result<i32, crate::PyError> {
    let n = unsafe { crate::builtins::space_index_w(v)? };
    i32::try_from(n).map_err(|_| {
        crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            "Python int too large to convert to C int",
        )
    })
}

/// A `str` `co_*` field as an owned `String` (the compiler `Name` type).
unsafe fn read_code_str(v: PyObjectRef, field: &str) -> Result<String, crate::PyError> {
    if !unsafe { pyre_object::is_str(v) } {
        return Err(crate::PyError::type_error(format!("{field} must be a str")));
    }
    Ok(unsafe { pyre_object::w_str_get_value(v) }.to_string())
}

/// `pycode.py filename='fsencode'`: retain filesystem bytes that the
/// compiler dependency's UTF-8 `source_path` cannot represent. A supplied
/// `fallback` is the last valid UTF-8 spelling used by interpreter/JIT readers;
/// a new code object uses a lossy compiler-only spelling until those readers
/// can consume the byte-exact field directly.
unsafe fn read_code_filename(
    v: PyObjectRef,
    field: &str,
    fallback: Option<&str>,
) -> Result<(String, Option<Vec<u8>>), crate::PyError> {
    if !unsafe { pyre_object::is_str(v) } {
        return Err(crate::PyError::type_error(format!("{field} must be a str")));
    }
    let bytes = crate::gateway::fsencode_bytes_w(v)?;
    Ok(split_code_filename_bytes(bytes, fallback))
}

/// Split the authoritative filesystem bytes from the compiler dependency's
/// UTF-8-only `source_path` spelling (`objspace.py newfilename`).
pub(crate) fn split_code_filename_bytes(
    bytes: Vec<u8>,
    fallback: Option<&str>,
) -> (String, Option<Vec<u8>>) {
    match String::from_utf8(bytes) {
        Ok(source_path) => (source_path, None),
        Err(error) => {
            let bytes = error.into_bytes();
            let source_path = fallback
                .map(str::to_owned)
                .unwrap_or_else(|| String::from_utf8_lossy(&bytes).into_owned());
            (source_path, Some(bytes))
        }
    }
}

/// A `tuple[str]` `co_*` field (names / varnames / freevars / cellvars).
unsafe fn read_code_names(v: PyObjectRef, field: &str) -> Result<Box<[String]>, crate::PyError> {
    if !unsafe { is_tuple(v) } {
        return Err(crate::PyError::type_error(format!(
            "{field} must be a tuple of strings"
        )));
    }
    let n = pyre_object::w_tuple_len(v);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = pyre_object::w_tuple_getitem(v, i as i64).unwrap_or_else(pyre_object::w_none);
        if !unsafe { pyre_object::is_str(e) } {
            return Err(crate::PyError::type_error(format!(
                "{field} must be a tuple of strings"
            )));
        }
        out.push(unsafe { pyre_object::w_str_get_value(e) }.to_string());
    }
    Ok(out.into_boxed_slice())
}

/// A `bytes` `co_*` field (linetable / exceptiontable) as raw bytes.
unsafe fn read_code_bytes(v: PyObjectRef, field: &str) -> Result<Box<[u8]>, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(v) } {
        return Err(crate::PyError::type_error(format!(
            "{field} must be a bytes object"
        )));
    }
    Ok(unsafe { pyre_object::bytesobject::bytes_like_data(v) }
        .to_vec()
        .into_boxed_slice())
}

/// `co_code` bytes → the decoded `CodeUnits` instruction stream.  The byte
/// form is the `original_bytes` layout: one `(opcode, arg)` pair per unit.
unsafe fn read_code_units(
    v: PyObjectRef,
) -> Result<(crate::bytecode::CodeUnits, Option<Vec<u8>>), crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(v) } {
        return Err(crate::PyError::type_error("co_code must be a bytes object"));
    }
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(v) };
    decode_code_units(bytes)
        .map_err(|()| crate::PyError::value_error("co_code length must be a multiple of 2"))
}

/// Decode public `co_code` bytes into compiler-core's execution storage while
/// retaining an exact fallback for opcode values its enum cannot represent.
///
/// The marshal runtime bag calls this same boundary, so `CodeType`,
/// `code.replace` and marshal loading cannot disagree about which bytes are
/// deferred to dispatch as `Instruction::Reserved`.
pub(crate) fn decode_code_units(
    bytes: &[u8],
) -> Result<(crate::bytecode::CodeUnits, Option<Vec<u8>>), ()> {
    if bytes.len() % 2 != 0 {
        return Err(());
    }
    let mut units = Vec::with_capacity(bytes.len() / 2);
    let mut preserve_raw = false;
    for pair in bytes.chunks_exact(2) {
        let (op, arg) = match crate::bytecode::Instruction::try_from(pair[0]) {
            Ok(crate::bytecode::Instruction::Reserved) | Err(_) => {
                preserve_raw = true;
                // `CodeUnit` cannot carry an arbitrary opcode byte.  Keep the
                // exact public stream in `PyCode.co_code_bytes`, and carry the
                // invalid opcode in the Reserved placeholder's otherwise
                // meaningless arg so shared interpreter/JIT dispatch can
                // report CPython's `unknown opcode N` at execution time.
                (
                    crate::bytecode::Instruction::Reserved,
                    crate::bytecode::OpArgByte::from(pair[0]),
                )
            }
            Ok(op) => (op, crate::bytecode::OpArgByte::from(pair[1])),
        };
        units.push(crate::bytecode::CodeUnit::new(op, arg));
    }
    Ok((
        crate::bytecode::CodeUnits::from(units),
        preserve_raw.then(|| bytes.to_vec()),
    ))
}

/// A `tuple` `co_consts` field → the compiler `Constants` backing table.
///
/// PyPy stores the supplied wrapped objects directly in `co_consts_w`
/// (`pycode.py:126`).  Pyre's compiler table still needs one entry per wrapped
/// object so bytecode indices remain valid, but it is not the semantic owner:
/// values the compiler enum cannot represent use `None` only as an unobserved
/// shape placeholder.  `w_code_fill_consts_from_tuple` immediately installs
/// every supplied object in the authoritative wrapped slots, and LOAD_CONST
/// plus marshal both read those slots.
unsafe fn read_code_consts(
    v: PyObjectRef,
) -> Result<crate::bytecode::Constants<crate::bytecode::ConstantData>, crate::PyError> {
    if !unsafe { is_tuple(v) } {
        return Err(crate::PyError::type_error("co_consts must be a tuple"));
    }
    let n = pyre_object::w_tuple_len(v);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = pyre_object::w_tuple_getitem(v, i as i64).unwrap_or_else(pyre_object::w_none);
        out.push(unsafe { obj_to_constant_data(e) }.unwrap_or(crate::bytecode::ConstantData::None));
    }
    Ok(out.into_iter().collect())
}

/// Convert a Python object into compiler `ConstantData` when that enum can
/// represent it.  Callers that need a literal compiler constant propagate the
/// `ValueError`; `read_code_consts` instead supplies a shape placeholder and
/// retains the arbitrary object in PyPy's authoritative `co_consts_w` slot.
pub(crate) unsafe fn obj_to_constant_data(
    obj: PyObjectRef,
) -> Result<crate::bytecode::ConstantData, crate::PyError> {
    use crate::bytecode::ConstantData;
    unsafe {
        if is_none(obj) {
            return Ok(ConstantData::None);
        }
        if is_ellipsis(obj) {
            return Ok(ConstantData::Ellipsis);
        }
        // bool is a subclass of int, so test it first.
        if is_bool(obj) {
            let value = crate::builtins::space_index_w(obj)? != 0;
            return Ok(ConstantData::Boolean { value });
        }
        if is_int_or_long(obj) {
            return Ok(ConstantData::Integer {
                value: crate::rbigint_to_compiler_bigint(&crate::builtins::obj_to_bigint(obj)),
            });
        }
        if is_float(obj) {
            return Ok(ConstantData::Float {
                value: pyre_object::w_float_get_value(obj),
            });
        }
        if pyre_object::is_complex(obj) {
            return Ok(ConstantData::Complex {
                value: num_complex::Complex64::new(
                    pyre_object::w_complex_get_real(obj),
                    pyre_object::w_complex_get_imag(obj),
                ),
            });
        }
        if pyre_object::is_str(obj) {
            return Ok(ConstantData::Str {
                value: pyre_object::w_str_get_wtf8(obj).to_owned(),
            });
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return Ok(ConstantData::Bytes {
                value: pyre_object::bytesobject::bytes_like_data(obj).to_vec(),
            });
        }
        if is_tuple(obj) {
            let n = pyre_object::w_tuple_len(obj);
            let mut elements = Vec::with_capacity(n);
            for i in 0..n {
                let e =
                    pyre_object::w_tuple_getitem(obj, i as i64).unwrap_or_else(pyre_object::w_none);
                elements.push(obj_to_constant_data(e)?);
            }
            return Ok(ConstantData::Tuple { elements });
        }
        if pyre_object::sliceobject::is_slice(obj) {
            return Ok(ConstantData::Slice {
                elements: Box::new([
                    obj_to_constant_data(pyre_object::sliceobject::w_slice_get_start(obj))?,
                    obj_to_constant_data(pyre_object::sliceobject::w_slice_get_stop(obj))?,
                    obj_to_constant_data(pyre_object::sliceobject::w_slice_get_step(obj))?,
                ]),
            });
        }
        if pyre_object::setobject::is_frozenset(obj) {
            let elements = pyre_object::w_set_items(obj)
                .into_iter()
                .map(|item| obj_to_constant_data(item))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(ConstantData::Frozenset { elements });
        }
        // The inverse of `pyframe.rs`'s `pyobject_from_constant`, which realizes
        // this constant as a slice object: a subscript with only literal bounds
        // (`p[:0]`) folds to one, so a module that has any is unconvertible
        // without this arm.
        if pyre_object::is_slice(obj) {
            let start = obj_to_constant_data(pyre_object::w_slice_get_start(obj))?;
            let stop = obj_to_constant_data(pyre_object::w_slice_get_stop(obj))?;
            let step = obj_to_constant_data(pyre_object::w_slice_get_step(obj))?;
            return Ok(ConstantData::Slice {
                elements: Box::new([start, stop, step]),
            });
        }
        if is_code(obj) {
            let ptr = w_code_get_ptr(obj) as *const crate::CodeObject;
            if !ptr.is_null() {
                return Ok(ConstantData::Code {
                    code: Box::new((*ptr).clone()),
                });
            }
        }
        Err(crate::PyError::value_error(
            "co_consts contains a value that is not a valid code constant",
        ))
    }
}

/// `pyopcode.py getconstant_w(index) -> co_consts_w[index]`: return the
/// one shared constant object the enclosing code holds at `index`. Normal
/// constructors filled the slot eagerly, matching `pycode.py`; realization
/// here is only a defensive fallback for a readable empty slot.
///
/// `w_code_obj` is the enclosing `PyCode` (`frame.pycode` for the interpreter,
/// the virtualizable `pycode` field for the blackhole), and `idx` is the
/// constant index. No side table is involved: the owner and storage shape are
/// the literal port of `PyCode.co_consts_w`.
///
/// Returns `PY_NULL` only when the enclosing code/slot cannot be resolved.
///
/// # Safety
/// `w_code_obj` must point to a valid `PyCode`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_code_const(w_code_obj: PyObjectRef, idx: usize) -> PyObjectRef {
    let w_code = unsafe { &*(w_code_obj as *const PyCode) };
    // Guard `code_ptr` before dereferencing it — the same null/alignment check
    // the lazy-cache initializers use. A null/misaligned pointer means the
    // nested code is unreadable, so return PY_NULL and let the caller realize
    // the constant from its own code object.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    if w_code.code_ptr.is_null() || (w_code.code_ptr as i64) & align_mask != 0 {
        return pyre_object::pyobject::PY_NULL;
    }
    let code = unsafe { &*(w_code.code_ptr as *const crate::CodeObject) };
    let constants = crate::pyframe::code_constants(code);
    // closure-free, Option-pattern-free `constants.get(idx)` rewrite — keep the
    // bounds check a plain `lt + getitem` ahead of the variant destructure.
    if idx >= constants.len() {
        return pyre_object::pyobject::PY_NULL;
    }
    if w_code.co_consts_w.is_null() {
        return crate::pyframe::pyobject_from_constant(&constants[idx]);
    }
    let slot_table = unsafe { &*w_code.co_consts_w };
    let Some(slot) = slot_table.get(idx) else {
        return pyre_object::pyobject::PY_NULL;
    };
    // Normal slots are already filled. Keep the fallback free-thread safe for
    // test stubs and alternate construction paths by retaining the AtomicPtr.
    let existing = slot.load(std::sync::atomic::Ordering::Acquire);
    if !existing.is_null() {
        return existing;
    }

    let mut realized = match &constants[idx] {
        crate::bytecode::ConstantData::Code { code } => unsafe {
            box_code_constant_inheriting_filename(&**code as *const crate::CodeObject, w_code)
        },
        constant => crate::pyframe::pyobject_from_constant(constant),
    };
    // Keep the losing or winning candidate live until the CAS has either
    // published it or selected the concurrently-published canonical object.
    let candidate_root = &mut realized as *mut PyObjectRef as *mut *mut u8;
    let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(candidate_root) };
    let published = match slot.compare_exchange(
        std::ptr::null_mut(),
        realized,
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
    ) {
        Ok(_) => {
            publish_code_slot_store(w_code_obj);
            realized
        }
        Err(winner) => winner,
    };
    if registered {
        pyre_object::gc_hook::try_gc_remove_root(candidate_root);
    }
    published
}

/// `pyopcode.py getname_w(index) -> self.getcode().co_names_w[index]`
/// — the one wrapped name this code object holds at `idx`.
///
/// Realized on first demand with `w_str_new`, whose result is
/// `malloc_typed`-immortal: the published pointer is fixed, so a slot is never
/// forwarded and a thread losing the publish race abandons its candidate rather
/// than freeing it.
///
/// Returns `PY_NULL` when the enclosing code or the slot cannot be resolved
/// (test fixtures and gateway builtins carry no name table); callers fall back
/// to wrapping the key themselves.
///
/// # Safety
/// `w_code_obj` must point to a valid `PyCode`.
#[majit_macros::dont_look_inside]
pub unsafe fn w_code_getname_w(w_code_obj: PyObjectRef, idx: usize) -> PyObjectRef {
    if w_code_obj.is_null() {
        return pyre_object::pyobject::PY_NULL;
    }
    let w_code = unsafe { &*(w_code_obj as *const PyCode) };
    if w_code.co_names_w.is_null() {
        return pyre_object::pyobject::PY_NULL;
    }
    let slot_table = unsafe { &*w_code.co_names_w };
    let Some(slot) = slot_table.get(idx) else {
        return pyre_object::pyobject::PY_NULL;
    };
    // PyPy's GIL serializes first access to its already-interned list. Pyre is
    // free-threaded and realizes this slot lazily, so every reader and writer
    // uses the AtomicPtr element stored in co_names_w.
    let existing = slot.load(std::sync::atomic::Ordering::Acquire);
    if !existing.is_null() {
        return existing;
    }
    // Guard `code_ptr` before dereferencing it — the same null/alignment check
    // the lazy-cache initializers use.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    if w_code.code_ptr.is_null() || (w_code.code_ptr as i64) & align_mask != 0 {
        return pyre_object::pyobject::PY_NULL;
    }
    let code = unsafe { &*(w_code.code_ptr as *const crate::CodeObject) };
    let Some(name) = code.names.get(idx) else {
        return pyre_object::pyobject::PY_NULL;
    };
    // `pycode.py space.new_interned_str(aname)` — one canonical object
    // per name value, not one per code object that names it.
    let realized = pyre_object::unicodeobject::intern_str_value(name);
    match slot.compare_exchange(
        std::ptr::null_mut(),
        realized,
        std::sync::atomic::Ordering::AcqRel,
        std::sync::atomic::Ordering::Acquire,
    ) {
        Ok(_) => realized,
        Err(winner) => winner,
    }
}

/// [`w_code_getname_w`] with the caller's own fallback folded in: a wrapper
/// carrying no name table answers `PY_NULL`, and the key is then minted the way
/// it was before `co_names_w` existed.
///
/// # Safety
/// `w_code_obj` must be null or point to a valid `PyCode`.
pub unsafe fn w_code_getname_w_or_new(
    w_code_obj: PyObjectRef,
    idx: usize,
    name: &str,
) -> PyObjectRef {
    let w_name = unsafe { w_code_getname_w(w_code_obj, idx) };
    if w_name.is_null() {
        // No slot to realize into, so nothing bounds how often this runs;
        // interning is what keeps an immortal string per execution from being
        // an immortal string per execution.
        return pyre_object::unicodeobject::intern_str_value(name);
    }
    w_name
}

/// pypy/module/__pypy__/interp_magic.py:79
/// `func.getcode().hidden_applevel = True` — explicit setter for the
/// `__pypy__.hidden_applevel(func)` builtin marker, plus the
/// `_continuation.entrypoint_pycode.hidden_applevel = True`
/// hand-edit (interp_continuation.py:195).  PyPy mutates the field
/// directly; pyre wraps the raw write because the field is private
/// to this module.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_set_hidden_applevel(obj: PyObjectRef, hidden_applevel: bool) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut PyCode)).hidden_applevel = hidden_applevel;
    }
}

/// Extract the opaque code pointer from a known PyCode.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_get_ptr(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const PyCode)).code_ptr }
}

/// `importing.py update_code_filenames`: set `source_path` on `code` and,
/// recursively, on every nested code constant whose filename still matches the
/// root's *original* name (leaving unrelated inlined filenames untouched).
///
/// The table is edited through `Constants`' `DerefMut`, so every constant keeps
/// its address. Nested constants are wrapped in place by
/// [`box_code_constant_in_place`], and a wrapper published before this runs
/// reads the new name out of the object it already points at — which is what
/// `update_code_filenames` mutating already-wrapped nested `PyCode` constants
/// amounts to.
fn fix_code_filenames(code: &mut crate::CodeObject, oldname: &str, newname: &str) {
    code.source_path = newname.to_owned();
    for constant in code.constants.iter_mut() {
        if let crate::bytecode::ConstantData::Code { code: nested } = constant
            && nested.source_path == oldname
        {
            fix_code_filenames(nested, oldname, newname);
        }
    }
}

/// `_imp._fix_co_filename(code, path)` (importing.py:158 fix_co_filename):
/// replace `co_filename` on the code object in place, recursing through its
/// nested code constants.  Only valid because the code object is owned
/// (`Box::into_raw`), single-threaded, and not executing when import machinery
/// calls this right after compilation.
///
/// # Safety
/// `w_code` must point to a valid `PyCode` whose body is not currently running.
pub unsafe fn fix_co_filename(w_code: PyObjectRef, newname: &[u8]) {
    let code_ptr = unsafe { w_code_get_ptr(w_code) } as *mut crate::CodeObject;
    if code_ptr.is_null() {
        return;
    }
    let old_filename = unsafe { code_filename_bytes(w_code) };

    // `importing.py update_code_filenames` mutates already-wrapped
    // nested `PyCode` constants, selected by the root's original filename.
    let pycode = unsafe { &*(w_code as *const PyCode) };
    if !pycode.co_consts_w.is_null() {
        for slot in unsafe { &*pycode.co_consts_w } {
            let nested = slot.load(std::sync::atomic::Ordering::Acquire);
            if !nested.is_null()
                && unsafe { is_code(nested) }
                && unsafe { code_filename_bytes(nested) } == old_filename
            {
                unsafe { fix_co_filename(nested, newname) };
            }
        }
    }

    let filename = match std::str::from_utf8(newname) {
        Ok(newname) => {
            let oldname = unsafe { (*code_ptr).source_path.clone() };
            unsafe { fix_code_filenames(&mut *code_ptr, &oldname, newname) };
            None
        }
        Err(_) => Some(newname.to_vec()),
    };
    unsafe { set_filename_bytes(w_code, filename) };
    unsafe {
        (*(w_code as *mut PyCode)).filename_inherits_to_nested =
            std::str::from_utf8(newname).is_err();
    }
}

/// Cached [`crate::pyframe::npure_cellvars`] for the code wrapper `obj`,
/// or `None` for a null/stub wrapper (sentinel `u32::MAX`) so the caller
/// falls back to recomputation.
///
/// # Safety
/// `obj` must be null or point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_npure_cellvars(obj: PyObjectRef) -> Option<usize> {
    if obj.is_null() {
        return None;
    }
    match unsafe { (*(obj as *const PyCode)).npure_cellvars } {
        u32::MAX => None,
        n => Some(n as usize),
    }
}

/// PyPy: `PyCode.hidden_applevel` (`pycode.py`). Reads the field
/// initialised by `w_code_new`.  `pyframe.py:521-522
/// hide(self): return self.pycode.hidden_applevel` is the sole caller
/// in the canonical interpreter; pyre routes through this accessor
/// from `pyframe.rs::PyFrame::hide`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_hidden_applevel(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe { (*(obj as *const PyCode)).hidden_applevel }
}

/// PyPy: `PyCode.w_globals` — the globals dict OBJECT. The JIT
/// codewriter/bridge read this to fold globals lookups without an off-GC
/// proxy.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_code_get_w_globals(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { (*(obj as *const PyCode)).w_globals }
}

/// PyPy: `PyCode.w_globals = w_globals`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_code_set_w_globals(obj: PyObjectRef, w_globals: PyObjectRef) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut PyCode)).w_globals = w_globals;
    }
    // A bootstrap code slot is reached only by the prebuilt root walk, which
    // clean minor collections may skip; record the store.
    publish_code_slot_store(obj);
    if !w_globals.is_null() {
        let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
        register_live_code_wrapper(code_ptr, obj);
        register_w_globals_stamped_code(obj);
    }
}

/// PyPy: `PyCode.frame_stores_global(w_globals)`.
#[inline]
/// # Safety
/// The caller must uphold every validity, runtime-type, aliasing, and lifetime
/// invariant required by the object and pointer arguments for the entire call.
pub unsafe fn w_code_frame_stores_global(obj: PyObjectRef, w_globals: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let code = unsafe { &mut *(obj as *mut PyCode) };
    if code.w_globals.is_null() {
        code.w_globals = w_globals;
        // Prebuilt-family store (see `w_code_set_w_globals`).
        publish_code_slot_store(obj);
        register_live_code_wrapper(code.code_ptr, obj);
        register_w_globals_stamped_code(obj);
        return false;
    }
    !std::ptr::eq(code.w_globals, w_globals)
}

/// The state of a code object's `co_positions` rows once its own `locations`
/// array has been released.
///
/// `Deferred` carries the first line number the rows must be decoded against.
/// `CodeObject.first_line_number` cannot stand in for it: it is an
/// `Option<OneIndexed>`, which cannot spell the zero and negative values
/// `CodeType(...)` accepts and `co_firstlineno_raw` preserves.
#[derive(Clone, Copy)]
enum CodeLocations {
    Deferred(i32),
    Decoded(&'static [(SourceLocation, SourceLocation)]),
}

/// Rows released from `CodeObject.locations`, keyed by code object address.
///
/// `locations` is not serialized: `marshal.rs:265,951` expand it out of
/// `linetable` while reading a code object, so a loaded code object carries the
/// same line information twice — compressed at about 1.5 bytes per instruction
/// and expanded at 32, since `SourceLocation` is a pair of `NonZeroUsize`.
/// Nothing but a traceback, a debugger line jump and the `co_positions` /
/// `co_lines` getters ever reads the expanded form, so [`w_code_new`] releases
/// it and [`code_locations`] rebuilds it on the first reader — the
/// realize-once treatment `co_consts_w` and `co_names_w` already get.
///
/// Decoded rows are leaked because the `CodeObject` describing them is itself
/// never released: `pycode_destructor` frees the side tables and leaves
/// `code_ptr` standing, so an entry can never outlive its key's validity and
/// never has to be revisited.
///
/// Process-global and keyed by `usize` for `LIVE_CODE_WRAPPERS`' reasons: the
/// rows belong to the shared code object, and a raw `PyObjectRef` is not
/// `Send`.
static CODE_LOCATIONS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, CodeLocations>>,
> = std::sync::OnceLock::new();

fn code_locations_cache()
-> &'static std::sync::Mutex<std::collections::HashMap<usize, CodeLocations>> {
    CODE_LOCATIONS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Release a freshly constructed code object's expanded `locations` array,
/// recording the first line number [`code_locations`] must decode it back
/// against.
///
/// Called before the wrapper is published, so no reader can be holding the
/// array that is dropped here, and a later call for the same code object (the
/// `co_firstlineno_raw` stamp that follows `CodeType(...)` and `code.replace`)
/// simply corrects the recorded line number.
fn release_code_locations(code_ptr: *mut crate::CodeObject, firstlineno_raw: i32) {
    // Nested constants are wrapped in place, so two threads realizing one
    // `co_consts_w` slot reach the same `CodeObject` before either wrapper is
    // published. The lock is taken and the vacant entry claimed before the
    // pointer is dereferenced, so only the thread that owns the record ever
    // forms a `&mut` to the object: the array is read and written under this
    // lock alone. A second wrapper over an already-released object must keep
    // the record the first made — re-deferring would abandon rows a reader has
    // since decoded.
    let mut cache = code_locations_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let std::collections::hash_map::Entry::Vacant(entry) = cache.entry(code_ptr as usize) else {
        return;
    };
    let code = unsafe { &mut *code_ptr };
    // A code object with no instructions has no rows to decode, and one that
    // holds no array has nothing to release.
    if code.instructions.is_empty() || code.locations.is_empty() {
        return;
    }
    entry.insert(CodeLocations::Deferred(firstlineno_raw));
    code.locations = Vec::new().into_boxed_slice();
}

/// Correct the first line number a released array is decoded against, leaving a
/// code object that still holds its own array alone.
fn record_deferred_locations_firstlineno(code_ptr: *mut crate::CodeObject, firstlineno_raw: i32) {
    let mut cache = code_locations_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(slot @ CodeLocations::Deferred(_)) = cache.get_mut(&(code_ptr as usize)) {
        *slot = CodeLocations::Deferred(firstlineno_raw);
    }
}

/// The `co_positions` rows of `code`, decoding them out of `linetable` on the
/// first reader when [`release_code_locations`] has taken the array away.
///
/// Returns the code object's own array untouched when it still holds one, so a
/// code object built outside [`w_code_new`] reads exactly as it always did.
///
/// An empty array on a code object that has instructions always means the rows
/// were released: every construction path fills `locations` through
/// `linetable_to_locations`, which returns one row per instruction. Decoding is
/// therefore the right answer whether or not the map still knows the object.
pub fn code_locations(code: &crate::CodeObject) -> &[(SourceLocation, SourceLocation)] {
    if !code.locations.is_empty() || code.instructions.is_empty() {
        return &code.locations;
    }
    let key = code as *const crate::CodeObject as usize;
    let mut cache = code_locations_cache()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let firstlineno_raw = match cache.get(&key) {
        Some(CodeLocations::Decoded(rows)) => return *rows,
        Some(CodeLocations::Deferred(firstlineno_raw)) => *firstlineno_raw,
        // A copy of a released code object: `w_code_fill_consts_from_tuple`
        // serializes a wrapped constant back into `ConstantData` by copying the
        // `CodeObject`, so the empty array travels to an address this map has
        // never seen. `linetable` is the serialized form and travels with it,
        // which is why it — not the map — is what makes the rows recoverable;
        // the record only exists to carry a first line number
        // `Option<OneIndexed>` cannot spell.
        None => code
            .first_line_number
            .map(|line| line.get() as i32)
            .unwrap_or(1),
    };
    let rows: &'static [(SourceLocation, SourceLocation)] =
        Box::leak(rustpython_compiler_core::marshal::linetable_to_locations(
            &code.linetable,
            firstlineno_raw,
            code.instructions.len(),
        ));
    cache.insert(key, CodeLocations::Decoded(rows));
    rows
}

/// Registry mapping a raw CodeObject pointer (`PyCode.code_ptr`) to the
/// live, globals-stamped `PyCode` wrapper. Populated where a frame stamps
/// the wrapper's `w_globals` — the only point both the raw pointer and the
/// live wrapper are in hand — and consumed by the JIT to recover the live
/// wrapper (and hence its `w_globals`) from a raw code pointer it already
/// holds, so the JIT need not carry the wrapper identity as a separate
/// `w_code` courier. First-write-wins, mirroring the first-store-wins
/// `PyCode.w_globals` semantics in `w_code_frame_stores_global`. Wrappers use
/// stable GC allocation; `MetaInterpStaticData.jitcodes` roots the ones it
/// retains and the collector destructor removes every other mapping before its
/// pointer can dangle.
///
/// Process-global, not per-thread: `pycode.py:159` keeps `w_globals` on the
/// shared `PyCode` instance, and a code object stamped on one thread must be
/// recoverable from every thread that later runs it — a thread-local map made
/// the JIT's `recover_inline_callee_globals` answer `PY_NULL` there and
/// decline the inline. Keys and values are `usize` because a raw
/// `PyObjectRef` is not `Send`, as in `interp_sre`'s pattern registry.
static LIVE_CODE_WRAPPERS: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<usize, usize>>,
> = std::sync::OnceLock::new();

fn live_code_wrappers() -> &'static std::sync::Mutex<std::collections::HashMap<usize, usize>> {
    LIVE_CODE_WRAPPERS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Record `wrapper` as the live wrapper for `code_ptr`, keeping the first one
/// stamped (later stores are ignored). No-op on null inputs.
pub fn register_live_code_wrapper(code_ptr: *const (), wrapper: PyObjectRef) {
    if code_ptr.is_null() || wrapper.is_null() {
        return;
    }
    live_code_wrappers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .entry(code_ptr as usize)
        .or_insert(wrapper as usize);
}

/// Recover the live wrapper previously registered for `code_ptr`, or `PY_NULL`
/// if none has been stamped.
pub fn live_code_wrapper(code_ptr: *const ()) -> PyObjectRef {
    if code_ptr.is_null() {
        return PY_NULL;
    }
    live_code_wrappers()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(code_ptr as usize))
        .map_or(PY_NULL, |&w| w as PyObjectRef)
}

/// Collector destructor for a managed `PyCode` wrapper.
///
/// PyPy lets the GC reclaim the `PyCode` and its list-valued cache fields
/// together.  Pyre's three cache vectors are raw Rust allocations, so release
/// them here and retire the compatibility registries that used to assume every
/// wrapper was immortal.  `code_ptr` deliberately remains allocated: compiled
/// JitCodes currently retain that raw compiler-body address independently of
/// the wrapper, and freeing it belongs to the later removal of that courier.
///
/// # Safety
/// `obj_addr` must be a collector-owned `PyCode` payload and this function must
/// run at most once for it.
pub unsafe fn pycode_destructor(obj_addr: usize) {
    let code = unsafe { &mut *(obj_addr as *mut PyCode) };
    let wrapper = obj_addr as PyObjectRef;
    unregister_prebuilt_code_root(wrapper);
    {
        let mut wrappers = live_code_wrappers()
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if wrappers.get(&(code.code_ptr as usize)).copied() == Some(wrapper as usize) {
            wrappers.remove(&(code.code_ptr as usize));
        }
    }
    w_globals_stamped_codes()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&obj_addr);
    mapdict_method_cache_codes()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&obj_addr);
    if !code.globals_caches.is_null() {
        drop(unsafe { Box::from_raw(code.globals_caches) });
        code.globals_caches = std::ptr::null_mut();
    }
    if !code.mapdict_caches.is_null() {
        drop(unsafe { Box::from_raw(code.mapdict_caches) });
        code.mapdict_caches = std::ptr::null_mut();
    }
    if !code.co_consts_w.is_null() {
        drop(unsafe { Box::from_raw(code.co_consts_w) });
        code.co_consts_w = std::ptr::null_mut();
    }
    if !code.co_names_w.is_null() {
        drop(unsafe { Box::from_raw(code.co_names_w) });
        code.co_names_w = std::ptr::null_mut();
    }
    if !code.filename_bytes.is_null() {
        drop(unsafe { Box::from_raw(code.filename_bytes) });
        code.filename_bytes = std::ptr::null_mut();
    }
    if !code.co_code_bytes.is_null() {
        drop(unsafe { Box::from_raw(code.co_code_bytes) });
        code.co_code_bytes = std::ptr::null_mut();
    }
}

/// pycode.py `_compute_flatcall`.
///
/// Returns FLATPYCALL | co_argcount for simple user functions (no *args,
/// **kwargs, keyword-only args). Returns HOPELESS otherwise.
fn compute_flatcall(code: &crate::CodeObject) -> u16 {
    use crate::CodeFlags;
    use crate::gateway::{FLATPYCALL, HOPELESS};
    if code
        .flags
        .intersects(CodeFlags::VARARGS | CodeFlags::VARKEYWORDS)
    {
        return HOPELESS;
    }
    if code.kwonlyarg_count > 0 {
        return HOPELESS;
    }
    if code.arg_count > 0xff {
        return HOPELESS;
    }
    // pycode.py:234 — disqualify if any arg is also a cellvar.
    // Pyre's CodeObject exposes cellvars; check for overlap.
    let argcount = code.arg_count as usize;
    if !code.cellvars.is_empty() && argcount > 0 {
        for cellname in &code.cellvars {
            for j in 0..argcount {
                if j < code.varnames.len() && *cellname == code.varnames[j] {
                    return HOPELESS;
                }
            }
        }
    }
    FLATPYCALL | (code.arg_count as u16)
}

/// eval.py:16-23 — read `fast_natural_arity` from a PyCode.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe { (*(obj as *const PyCode)).fast_natural_arity & !YIELDS_INSIDE_TRY_BIT }
}

/// Unified accessor: read `fast_natural_arity` from any code object
/// (BuiltinCode or PyCode).
///
/// # Safety
/// `obj` must point to a valid code object (either type).
#[inline]
pub unsafe fn code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe {
        if crate::gateway::is_builtin_code(obj) {
            crate::gateway::builtin_code_get_fast_natural_arity(obj)
        } else {
            w_code_get_fast_natural_arity(obj)
        }
    }
}

/// pycode.py `PyCode.lookup_exceptiontable`.
///
/// Search the wrapped code object's exception table for a handler
/// covering `instr_offset` (byte offset into `co_code`).  Returns
/// `Some((target, depth, lasti))` with byte-offset `target` when found.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_lookup_exceptiontable(
    obj: PyObjectRef,
    instr_offset: u32,
) -> Option<(u32, u32, bool)> {
    if obj.is_null() {
        return None;
    }
    let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
    if code_ptr.is_null() {
        return None;
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    crate::pycode::lookup_exceptiontable(&code.exceptiontable, instr_offset)
}

/// pycode.py `self.co_exceptiontable = exceptiontable` — copy the
/// varint-packed table bytes out of the wrapped `CodeObject`.
///
/// The bytes are owned by the inner `CodeObject` (`Box<[u8]>` field), so
/// returning a reference would tie the lifetime to the obj's heap
/// allocation.  Callers that need to hand the bytes to Python (where
/// they get copied into a `W_BytesObject`) take the owned `Vec<u8>`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_exceptiontable(obj: PyObjectRef) -> Vec<u8> {
    if obj.is_null() {
        return Vec::new();
    }
    let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
    if code_ptr.is_null() {
        return Vec::new();
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    code.exceptiontable.to_vec()
}

/// `celldict.py:292 cache_wref = pycode._globals_caches[nameindex]` —
/// read slot `nameindex` and upgrade the weakref to a strong
/// `Arc<Mutex<GlobalCache>>` (returning `None` when the slot is
/// unset, the weak target is gone, or `code_ptr` is invalid).
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<std::sync::Arc<std::sync::Mutex<pyre_object::celldict::GlobalCache>>> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return None;
    }
    let vec = unsafe { &*code.globals_caches }.lock().unwrap();
    vec.get(nameindex)
        .and_then(|slot| slot.as_ref())
        .and_then(|w| w.upgrade())
}

/// `celldict.py:321/353 pycode._globals_caches[nameindex] = cache.ref`
/// — store `Arc::downgrade(cache)` in slot `nameindex`.  No-op when
/// `code_ptr` is invalid or `nameindex` is out of range.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    cache: &std::sync::Arc<std::sync::Mutex<pyre_object::celldict::GlobalCache>>,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return;
    }
    let mut vec = unsafe { &*code.globals_caches }.lock().unwrap();
    if let Some(slot) = vec.get_mut(nameindex) {
        *slot = Some(std::sync::Arc::downgrade(cache));
    }
}

/// Number of `_globals_caches` slots — equals `len(co_names_w)` at
/// construction time.  Returns 0 for code objects built from null
/// or unaligned `code_ptr`.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return 0;
    }
    unsafe { (*code.globals_caches).lock().unwrap().len() }
}

/// `mapdict.py/1546/1575 entry = pycode._mapdict_caches[nameindex]` — read
/// slot `nameindex`, returning `None` (PyPy `INVALID_CACHE_ENTRY`) when the slot
/// is unset, out of range, or `code_ptr` is invalid.  The entry is `Copy`, so a
/// value is returned (no aliasing of the slot).
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<crate::objspace::std::mapdict::MapdictCacheEntry> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return None;
    }
    let vec = unsafe { &*code.mapdict_caches };
    vec.get(nameindex).copied().flatten()
}

/// `mapdict.py pycode._mapdict_caches[nameindex] = entry` — store the
/// filled entry in slot `nameindex`.  No-op when `code_ptr` is invalid or
/// `nameindex` is out of range.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    entry: crate::objspace::std::mapdict::MapdictCacheEntry,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return;
    }
    let vec = unsafe { &mut *code.mapdict_caches };
    if let Some(slot) = vec.get_mut(nameindex) {
        *slot = Some(entry);
        // The LOAD_METHOD fill (mapdict.py:1474) stores a movable
        // `w_method` reference; register this code object so
        // `walk_mapdict_method_cache_gc` forwards the slot.
        if !entry.w_method.is_null() {
            if !pyre_object::gc_hook::try_gc_owns_object(obj as *mut u8) {
                mapdict_method_cache_codes()
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(obj as usize);
            }
            // The slot is reached only by `walk_mapdict_method_cache_gc`,
            // skipped on clean minors.
            publish_code_slot_store(obj);
        }
    }
}

/// Code objects whose `_mapdict_caches` hold (or once held) a filled
/// `w_method` slot.  In PyPy `CacheEntry.w_method` (mapdict.py)
/// is traced through the GC-managed `PyCode`; a managed wrapper's custom
/// trace reaches the slot the same way, so only a bootstrap wrapper minted
/// before the collector exists enters this registry and the extra-root
/// walker forwards it from here (same family as `walk_method_cache_gc`).
static MAPDICT_METHOD_CACHE_CODES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<usize>>,
> = std::sync::OnceLock::new();

fn mapdict_method_cache_codes() -> &'static std::sync::Mutex<std::collections::HashSet<usize>> {
    MAPDICT_METHOD_CACHE_CODES
        .get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// Code objects whose `w_globals` has been stamped
/// (`pycode.py frame_stores_global`).  `w_globals` is a permanent
/// strong field: the first globals object a code object runs in is kept
/// for the code object's lifetime and never replaced.  Upstream that
/// field is traced through the GC-managed `PyCode`, and a managed wrapper's
/// custom trace does the same.  A bootstrap wrapper minted before the
/// collector exists is reached only by the opportunistic
/// `walk_raw_code_roots` calls on `frame.pycode` and `func.code`, which miss
/// any stamped code object off the frame chain and not held in a walked frame
/// slot at collection time; that leaves a pre-move nursery address in
/// `w_globals` once its dict is promoted, and the next call through that code
/// object forwards a dangling pointer.  This registry makes the slot a root of
/// its own for that family, as [`MAPDICT_METHOD_CACHE_CODES`] does.
static W_GLOBALS_STAMPED_CODES: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashSet<usize>>,
> = std::sync::OnceLock::new();

fn w_globals_stamped_codes() -> &'static std::sync::Mutex<std::collections::HashSet<usize>> {
    W_GLOBALS_STAMPED_CODES.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()))
}

/// Record `obj` as a code object holding a stamped `w_globals` slot.
fn register_w_globals_stamped_code(obj: PyObjectRef) {
    // A managed PyCode's own custom trace owns this edge.  Only bootstrap
    // wrappers outside the collector need the compatibility registry; making
    // every managed code's globals an independent root would turn
    // `code -> globals -> function/generator -> code` cycles immortal.
    if pyre_object::gc_hook::try_gc_owns_object(obj as *mut u8) {
        return;
    }
    w_globals_stamped_codes()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(obj as usize);
}

/// Forward the stamped `w_globals` slot of every registered code object.
///
/// A process-global walker, not a per-mutator root area: a code object is
/// immortal and `w_globals` is first-store-wins, so the slot outlives whichever
/// thread happened to stamp it.  Owning the root per-thread meant
/// `unregister_mutator` dropped that thread's area at thread exit while the
/// code object stayed live and callable, and the slot then went unforwarded.
/// Same ownership argument as `interp_sre`'s pattern registry.
///
/// Reached only from the collector's root walk, where every mutator is at a
/// safepoint — that is what makes handing out `&mut code.w_globals` sound.
#[majit_macros::dont_look_inside]
pub fn walk_w_globals_stamped_code_roots(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    let codes = w_globals_stamped_codes()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &code in codes.iter() {
        let code = unsafe { &mut *(code as *mut PyCode) };
        if code.w_globals.is_null() {
            continue;
        }
        forward(&mut code.w_globals);
    }
}

/// Forward every filled `entry.w_method` slot during collection — the
/// faithful equivalent of the GC tracing PyPy's `CacheEntry.w_method`
/// (mapdict.py:1418) gets through its GC-managed holder.  The cached
/// map/attr node pointers are immortal interned nodes and the
/// `version_tag` is a `u64`, so `w_method` is the entry's only movable
/// reference.
///
/// Process-global for the same reason as
/// [`walk_w_globals_stamped_code_roots`]: the holder is immortal and outlives
/// the thread that filled the slot.
#[majit_macros::dont_look_inside]
pub fn walk_mapdict_method_cache_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    let codes = mapdict_method_cache_codes()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    for &code in codes.iter() {
        let code = unsafe { &*(code as *const PyCode) };
        if code.mapdict_caches.is_null() {
            continue;
        }
        let vec = unsafe { &mut *code.mapdict_caches };
        for slot in vec.iter_mut() {
            if let Some(entry) = slot.as_mut()
                && !entry.w_method.is_null()
            {
                forward(&mut entry.w_method);
            }
        }
    }
}

/// Number of `_mapdict_caches` slots — equals `len(co_names_w)` at construction
/// time.  Returns 0 for code objects built from null or unaligned `code_ptr`.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return 0;
    }
    unsafe { (*code.mapdict_caches).len() }
}

/// Check if an object is a code object.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CODE_TYPE) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_exec;

    /// Build a minimal varint-encoded exception table from `(start, length,
    /// target, depth, lasti)` tuples, mirroring the encoding produced by
    /// `assemble.py::_encode_varint`. Values are passed as word offsets
    /// (the on-disk unit), not byte offsets.
    fn encode_table(entries: &[(u32, u32, u32, u32, bool)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (start, length, target, depth, lasti) in entries.iter().copied() {
            push_varint(&mut out, start, true);
            push_varint(&mut out, length, false);
            push_varint(&mut out, target, false);
            push_varint(&mut out, (depth << 1) | (lasti as u32), false);
        }
        out
    }

    fn push_varint(out: &mut Vec<u8>, mut value: u32, entry_start: bool) {
        let mut chunks = [0u8; 6];
        let mut n = 0;
        loop {
            chunks[n] = (value & 63) as u8;
            n += 1;
            value >>= 6;
            if value == 0 {
                break;
            }
        }
        for j in (0..n).rev() {
            let mut byte = chunks[j];
            if j != 0 {
                byte |= 0x40;
            }
            if j == n - 1 && entry_start {
                byte |= 0x80;
            }
            out.push(byte);
        }
    }

    /// The two tables disagree on byte order: `co_exceptiontable` puts the
    /// most significant 6-bit group first (`parse_varint`), `co_linetable`
    /// the least significant one (`write_varint`). 70 therefore encodes
    /// differently under each, and handing either reader the other's bytes
    /// yields 385 — the failure a shared decoder would produce on every
    /// multi-byte delta.
    #[test]
    fn the_two_varint_tables_use_opposite_byte_order() {
        let exception_bytes = [0x41u8, 0x06];
        let location_bytes = [0x46u8, 0x01];

        assert_eq!(decode_varint(&exception_bytes, 0), (70, 2));
        assert_eq!(LineTableReader::new(&location_bytes).read_varint(), 70);

        assert_eq!(decode_varint(&location_bytes, 0), (385, 2));
        assert_eq!(LineTableReader::new(&exception_bytes).read_varint(), 385);
    }

    /// `code.replace(co_linetable=...)` accepts arbitrary bytes, so the
    /// continuation chain can be longer than a u32 holds. Shifting by the full
    /// width panics in a debug build, so the groups past it are dropped.
    #[test]
    fn a_location_varint_chain_past_the_word_width_does_not_panic() {
        let overlong = [0x7fu8, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x00];
        let mut expected = 0x3fu32;
        for shift in [6u32, 12, 18, 24, 30] {
            expected |= 0x3fu32 << shift;
        }
        assert_eq!(LineTableReader::new(&overlong).read_varint(), expected);
    }

    #[test]
    fn empty_table_returns_none() {
        assert_eq!(lookup_exceptiontable(&[], 0), None);
    }

    #[test]
    fn lookup_returns_byte_offsets() {
        // entry: word offsets start=4 (byte 8), length=10 (byte 20),
        // target=20 (byte 40), depth=2, lasti=false
        let table = encode_table(&[(4, 10, 20, 2, false)]);
        assert_eq!(lookup_exceptiontable(&table, 8), Some((40, 2, false)));
        assert_eq!(lookup_exceptiontable(&table, 27), Some((40, 2, false)));
        assert_eq!(lookup_exceptiontable(&table, 28), None);
        assert_eq!(lookup_exceptiontable(&table, 7), None);
    }

    #[test]
    fn last_matching_wins() {
        // Two overlapping ranges; outer first, inner second
        // (CPython emission order).
        let table = encode_table(&[(0, 10, 20, 1, false), (3, 4, 30, 3, true)]);
        assert_eq!(lookup_exceptiontable(&table, 2), Some((40, 1, false)));
        // PC 8 (byte) is covered by both. PyPy returns the later entry.
        assert_eq!(lookup_exceptiontable(&table, 8), Some((60, 3, true)));
        assert_eq!(lookup_exceptiontable(&table, 14), Some((40, 1, false)));
    }

    #[test]
    fn lasti_low_bit() {
        let table = encode_table(&[(0, 2, 10, 5, true)]);
        let (target, depth, lasti) = lookup_exceptiontable(&table, 0).unwrap();
        assert_eq!((target, depth, lasti), (20, 5, true));
    }

    #[test]
    fn iter_matches_lookup_count() {
        let table = encode_table(&[
            (0, 4, 8, 1, false),
            (10, 6, 20, 2, true),
            (30, 2, 40, 0, false),
        ]);
        let entries: Vec<_> = decode_exceptiontable(&table).collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1].start, 20);
        assert_eq!(entries[1].end, 32);
        assert_eq!(entries[1].target, 40);
        assert_eq!(entries[1].depth, 2);
        assert!(entries[1].lasti);
    }

    #[test]
    fn early_break_when_start_past_offset() {
        let table = encode_table(&[(0, 2, 10, 1, false), (100, 2, 200, 2, false)]);
        assert_eq!(lookup_exceptiontable(&table, 50), None);
    }

    #[test]
    fn live_code_wrapper_round_trips_first_write() {
        let code = 0x1000usize as *const ();
        let w1 = 0x2000usize as PyObjectRef;
        let w2 = 0x3000usize as PyObjectRef;
        register_live_code_wrapper(code, w1);
        // First-write-wins: a later store for the same code is ignored.
        register_live_code_wrapper(code, w2);
        assert_eq!(live_code_wrapper(code), w1);
        // An unregistered code pointer recovers to PY_NULL.
        assert!(live_code_wrapper(0x9999usize as *const ()).is_null());
        // Null inputs are no-ops / recover to PY_NULL.
        register_live_code_wrapper(std::ptr::null(), w1);
        assert!(live_code_wrapper(std::ptr::null()).is_null());
    }

    #[test]
    fn w_code_const_null_code_ptr_returns_py_null() {
        // A `PyCode` built from a null `code_ptr` must not be
        // dereferenced; the guard returns PY_NULL so the caller falls back to
        // its own constant realization.
        let w_code = w_code_new(std::ptr::null());
        let result = unsafe { w_code_const(w_code, 0) };
        assert_eq!(result, pyre_object::pyobject::PY_NULL);
    }

    #[test]
    fn box_code_object_preserves_owned_storage() {
        let code = compile_exec("answer = 42\n").expect("compile failed");
        let source_storage = code.source_path.as_ptr();
        let instruction_storage = code.instructions.as_ptr();

        let w_code = box_code_object(code);
        let stored = unsafe { &*(w_code_get_ptr(w_code) as *const crate::CodeObject) };

        assert_eq!(stored.source_path.as_ptr(), source_storage);
        assert_eq!(stored.instructions.as_ptr(), instruction_storage);
    }

    #[test]
    fn w_code_const_shares_large_integer_and_root_walker_visits_slot() {
        let code =
            compile_exec("x = 123456789012345678901234567890123456789012345678901234567890\n")
                .expect("compile failed");
        let idx = code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("large integer constant");
        let w_code = box_code_constant(&code);

        let eager = unsafe {
            (&*(*(w_code as *const PyCode)).co_consts_w)[idx]
                .load(std::sync::atomic::Ordering::Acquire)
        };
        assert_ne!(
            eager,
            pyre_object::pyobject::PY_NULL,
            "PyCode construction must eagerly wrap every compiler constant"
        );
        let first = unsafe { w_code_const(w_code, idx) };
        let second = unsafe { w_code_const(w_code, idx) };
        assert_eq!(first, eager);
        assert_eq!(
            first, second,
            "getconstant_w must return the co_consts_w slot identity"
        );
        assert!(unsafe { pyre_object::is_long(first) });
        let exposed = unsafe { code_get_field(w_code, "co_consts") }.expect("co_consts descriptor");
        assert_eq!(
            unsafe { pyre_object::w_tuple_getitem(exposed, idx as i64) },
            Some(first),
            "code.co_consts must expose the same wrapped slot object"
        );

        let mut visited = false;
        unsafe {
            crate::eval::walk_raw_code_roots(w_code, &mut |root| {
                if root.0 == first as usize {
                    visited = true;
                }
            });
        }
        assert!(
            visited,
            "Box-immortal PyCode must expose managed co_consts_w values as roots"
        );

        let mut globally_visited = false;
        walk_prebuilt_code_roots(&mut |root| {
            if root.0 == first as usize {
                globally_visited = true;
            }
        });
        assert!(
            globally_visited,
            "a standalone PyCode must remain a root without a function/frame owner"
        );
    }

    #[test]
    fn copy_const_slots_preserves_eager_source_identities() {
        let code = compile_exec(
            "x = 12345678901234567890123456789012345678901234567890\n\
             y = 98765432109876543210987654321098765432109876543210\n",
        )
        .expect("compile failed");
        let integer_indices: Vec<usize> = code
            .constants
            .iter()
            .enumerate()
            .filter_map(|(index, constant)| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
                .then_some(index)
            })
            .collect();
        assert!(integer_indices.len() >= 2);
        let first_idx = integer_indices[0];
        let second_idx = integer_indices[1];
        let src = box_code_constant(&code);
        let dst = box_code_constant(&code);
        let first = unsafe { w_code_const(src, first_idx) };
        let second = unsafe { w_code_const(src, second_idx) };

        unsafe {
            w_code_copy_const_slots(dst, src);
            let dst_slots = &*(*(dst as *const PyCode)).co_consts_w;
            assert_eq!(
                dst_slots[first_idx].load(std::sync::atomic::Ordering::Acquire),
                first
            );
            assert_eq!(
                dst_slots[second_idx].load(std::sync::atomic::Ordering::Acquire),
                second,
                "code.replace must preserve every eager co_consts_w identity"
            );
        }
    }

    #[test]
    fn raw_code_root_walker_reports_one_gc_edge_at_a_time() {
        let code = compile_exec(
            "def outer():\n\
             \x20   def inner():\n\
             \x20       return 12345678901234567890123456789012345678901234567890\n\
             \x20   return inner\n",
        )
        .expect("compile failed");
        let (outer_idx, outer_code) = code
            .constants
            .iter()
            .enumerate()
            .find_map(|(index, constant)| match constant {
                crate::bytecode::ConstantData::Code { code } => Some((index, code.as_ref())),
                _ => None,
            })
            .expect("outer code constant");
        let (inner_idx, inner_code) = outer_code
            .constants
            .iter()
            .enumerate()
            .find_map(|(index, constant)| match constant {
                crate::bytecode::ConstantData::Code { code } => Some((index, code.as_ref())),
                _ => None,
            })
            .expect("inner code constant");
        let bigint_idx = inner_code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("nested large integer");

        let w_top = box_code_constant(&code);
        let w_outer = unsafe { w_code_const(w_top, outer_idx) };
        let w_inner = unsafe { w_code_const(w_outer, inner_idx) };
        let w_bigint = unsafe { w_code_const(w_inner, bigint_idx) };
        let mut top_reaches_outer = false;
        let mut top_reaches_deep_value = false;
        unsafe {
            crate::eval::walk_raw_code_roots(w_top, &mut |root| {
                if root.0 == w_outer as usize {
                    top_reaches_outer = true;
                }
                if root.0 == w_bigint as usize {
                    top_reaches_deep_value = true;
                }
            });
        }
        assert!(
            top_reaches_outer,
            "a PyCode trace must report its directly-held child code"
        );
        assert!(
            !top_reaches_deep_value,
            "a GC trace callback must leave transitive traversal to the mark worklist"
        );
        let mut inner_reaches_bigint = false;
        unsafe {
            crate::eval::walk_raw_code_roots(w_inner, &mut |root| {
                if root.0 == w_bigint as usize {
                    inner_reaches_bigint = true;
                }
            });
        }
        assert!(
            inner_reaches_bigint,
            "the nested object's direct edge was lost"
        );
    }

    #[test]
    fn w_code_const_reads_are_free_threaded_identity_safe() {
        let code =
            compile_exec("x = 314159265358979323846264338327950288419716939937510582097494\n")
                .expect("compile failed");
        let idx = code
            .constants
            .iter()
            .position(|constant| {
                matches!(
                    constant,
                    crate::bytecode::ConstantData::Integer { value }
                        if num_traits::ToPrimitive::to_i64(value).is_none()
                )
            })
            .expect("large integer constant");
        let w_code = box_code_constant(&code) as usize;
        // The constructor publishes every slot eagerly, so readers would all
        // observe that one store and never reach the path the identity
        // guarantee actually rests on. Clear the slot under test first: each
        // worker then realizes its own candidate and races to install it, and
        // the assertion below becomes a statement about the CAS rather than
        // about the constructor.
        let pycode = unsafe { &*(w_code as *const PyCode) };
        assert!(!pycode.co_consts_w.is_null(), "wrapped constant array");
        let slots = unsafe { &*pycode.co_consts_w };
        slots[idx].store(std::ptr::null_mut(), std::sync::atomic::Ordering::Release);
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(8));
        let mut workers = Vec::new();
        for _ in 0..8 {
            let barrier = barrier.clone();
            workers.push(std::thread::spawn(move || {
                barrier.wait();
                unsafe { w_code_const(w_code as PyObjectRef, idx) as usize }
            }));
        }
        let values: Vec<usize> = workers
            .into_iter()
            .map(|worker| worker.join().expect("constant worker panicked"))
            .collect();
        assert!(values[0] != 0);
        assert!(
            values.iter().all(|value| *value == values[0]),
            "all readers must observe one canonical co_consts_w wrapper"
        );
    }
}
