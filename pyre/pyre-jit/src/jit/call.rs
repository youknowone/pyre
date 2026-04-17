//! RPython: `rpython/jit/codewriter/call.py` `CallControl`.
//!
//! pyre-specific analog of RPython's `CallControl`. RPython's CallControl
//! owns the JIT's per-warmspot state: the `jitcodes` dict mapping graphs
//! to compiled JitCodes and the `unfinished_graphs` queue that
//! `CodeWriter.make_jitcodes()` drains at warmspot init (codewriter.py:74-89).
//!
//! pyre's "graph" is a `CodeObject` pointer; otherwise the field-by-field
//! port matches upstream:
//!
//! | RPython `call.py`        | pyre `call.rs`           |
//! |--------------------------|--------------------------|
//! | `self.jitcodes`          | `self.jitcodes`          |
//! | `self.unfinished_graphs` | `self.unfinished_graphs` |
//! | `get_jitcode()`          | `get_jitcode()`          |
//! | `grab_initial_jitcodes()`| `grab_initial_jitcodes()`|
//! | `enum_pending_graphs()`  | `enum_pending_graphs()`  |
//!
//! PRE-EXISTING-ADAPTATION: RPython's CallControl is owned by a single
//! `CodeWriter` instance (`warmspot.py:245`) for the lifetime of the JIT.
//! pyre mirrors that with a per-thread singleton accessed via
//! `CodeWriter::instance()`; interior mutability of the queue/cache is
//! expressed with `UnsafeCell` on the owning `CodeWriter`.

use std::collections::HashMap;

use pyre_interpreter::CodeObject;

use super::codewriter::{CodeWriter, PyJitCode};

/// RPython: `rpython/jit/codewriter/call.py:21` `class CallControl(object)`.
pub struct CallControl {
    /// call.py:29 `self.jitcodes = {}` — map `{graph: jitcode}`.
    ///
    /// pyre keys on the opaque `CodeObject*` (erased to `usize`) rather
    /// than a flow-graph reference because pyre's "graph" is a Python
    /// bytecode object; the identity semantics are the same.
    ///
    /// Values are `Box<PyJitCode>` so that the entry's heap address is
    /// stable across `HashMap` rehashes. `get_jitcode` returns a
    /// `&PyJitCode` elaborated into `'static` to match RPython's Python
    /// semantics (see the SAFETY comment there); without boxing, a
    /// later `insert` could rehash and invalidate a ref the caller
    /// still holds.
    pub jitcodes: HashMap<usize, Box<PyJitCode>>,

    /// call.py:30 `self.unfinished_graphs = []` — LIFO queue of graphs
    /// pending compilation. Populated by `get_jitcode()` when it sees a
    /// new graph and drained by `enum_pending_graphs()` inside
    /// `make_jitcodes()` (codewriter.py:79).
    ///
    /// pyre stores `(code_ptr, w_code)` pairs instead of bare graphs
    /// because the drain loop — pyre's analog of
    /// `assembler.finished(callinfocollection)` at codewriter.py:85 —
    /// needs `w_code` to pipe `set_majit_jitcode` into
    /// `MetaInterpStaticData`. RPython's Assembler reads this same
    /// identity off each `JitCode` directly; pyre's `state::JitCode`
    /// is keyed on `w_code` so we carry it through.
    pub unfinished_graphs: Vec<(*const CodeObject, *const ())>,
}

impl CallControl {
    /// RPython: `CallControl.__init__(cpu=None, jitdrivers_sd=[])`
    /// (call.py:25-47). pyre's cpu-equivalent lives on `CodeWriter`
    /// directly (Phase A); the jitdrivers_sd analog will land with
    /// `setup_jitdriver` in Phase D.
    pub fn new() -> Self {
        Self {
            jitcodes: HashMap::new(),
            unfinished_graphs: Vec::new(),
        }
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py:145-148).
    ///
    /// ```python
    /// for jd in self.jitdrivers_sd:
    ///     jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///     jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `jitdrivers_sd` list yet; the
    /// "portal graph" is whatever CodeObject the hot-loop heuristic
    /// decides to trace. Phase D will replace the pyre-specific
    /// `initial: &[...]` parameter with the RPython iteration over
    /// `jitdrivers_sd`; for now callers pass the initial codes (with an
    /// optional `merge_point_pc` refinement, pyre-only) explicitly.
    ///
    /// The third tuple element — `merge_point_pc` — is a pyre-only
    /// refinement: the first trace reveals the `MERGE_POINT` opcode's
    /// PC, which must be re-recorded on the `PyJitCode` even after the
    /// initial eager compile. RPython has no analog because RPython
    /// portals have fixed entry PCs at flow-graph time.
    pub fn grab_initial_jitcodes(
        &mut self,
        writer: &CodeWriter,
        initial: &[(&CodeObject, *const (), Option<usize>)],
    ) {
        for (code, w_code, merge_point_pc) in initial {
            // call.py:147 `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`.
            // Inserts the new graph into `jitcodes` and pushes it onto
            // `unfinished_graphs`. The `jitdriver_sd` assignment
            // (call.py:148) already happens inside
            // `CodeWriter::transform_graph_to_jitcode` at codewriter.rs:1654
            // when `is_portal` is true.
            let _ = self.get_jitcode(code, *w_code, writer, *merge_point_pc);
        }
    }

    /// Pure lookup of an already-compiled JitCode. Matches RPython's
    /// runtime pattern where callers read `self.jitcodes[graph]`
    /// directly after `make_jitcodes()` has populated the dict; no
    /// compilation side effect. Returns `None` if `code` has not been
    /// compiled yet.
    pub fn find_jitcode(&self, code: *const CodeObject) -> Option<&PyJitCode> {
        self.jitcodes.get(&(code as usize)).map(|b| &**b)
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py:150-153).
    ///
    /// ```python
    /// while self.unfinished_graphs:
    ///     graph = self.unfinished_graphs.pop()
    ///     yield graph, self.jitcodes[graph]
    /// ```
    ///
    /// pyre returns a single `(code_ptr, &PyJitCode)` tuple per call to
    /// keep the return type simple in the absence of a proper
    /// generator; callers loop until `None`.
    pub fn enum_pending_graphs(&mut self) -> Option<(*const CodeObject, *const (), &PyJitCode)> {
        let (code_ptr, w_code) = self.unfinished_graphs.pop()?;
        let key = code_ptr as usize;
        self.jitcodes.get(&key).map(|b| (code_ptr, w_code, &**b))
    }

    /// RPython: `CallControl.get_jitcode(self, graph, called_from=None)`
    /// (call.py:155-172).
    ///
    /// Returns the JitCode for `code`, compiling it if not cached.
    /// Matches the upstream contract: first lookup the jitcodes dict;
    /// on a miss construct a new JitCode, insert, and push the graph
    /// onto `unfinished_graphs`.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre deviates from upstream in two
    /// places.  `writer` is threaded through because
    /// `transform_graph_to_jitcode` lives on `CodeWriter` (pyre's
    /// eager-compile model does the flow-graph transform inline here,
    /// not in a later drain pass), and `merge_point_pc` can be refined
    /// after the first trace discovers the `MERGE_POINT` location so
    /// the cache entry is rebuilt when it changes — RPython has no
    /// such refinement because portal PCs are known at flow-graph
    /// time.
    pub fn get_jitcode(
        &mut self,
        code: &CodeObject,
        w_code: *const (),
        writer: &CodeWriter,
        merge_point_pc: Option<usize>,
    ) -> &'static PyJitCode {
        let key = code as *const CodeObject as usize;
        let needs_rebuild = if let Some(existing) = self.jitcodes.get(&key) {
            merge_point_pc.is_some() && existing.merge_point_pc != merge_point_pc
        } else {
            true
        };
        if needs_rebuild {
            // call.py:168-171 — create JitCode, insert, append to
            // unfinished_graphs. Piping to `MetaInterpStaticData` is
            // deferred to `CodeWriter::make_jitcodes` (the analog of
            // `assembler.finished(callinfocollection)` at
            // codewriter.py:85); draining pops each entry and pipes
            // then, matching the upstream "transform in drain, pipe in
            // finished" split.
            let pyjitcode = writer.transform_graph_to_jitcode(code, w_code, merge_point_pc);
            self.jitcodes.insert(key, Box::new(pyjitcode));
            // call.py:171 `self.unfinished_graphs.append(graph)`. pyre
            // also re-pushes on a merge_point_pc refinement so the
            // drain picks the refined entry up again for the pipe
            // — see `make_jitcodes` at codewriter.rs.
            self.unfinished_graphs
                .push((code as *const CodeObject, w_code));
        }
        let entry = self.jitcodes.get(&key).unwrap();
        // SAFETY: the per-thread `CodeWriter` singleton lives for the
        // thread's lifetime, and `PyJitCode` is owned through `Box` so
        // the heap address is stable across `HashMap` rehashes. The
        // `'static` promise mirrors RPython's Python semantics
        // (call.py returns the dict value directly and relies on
        // refcounting to keep it alive); in Rust the Box pointer is
        // the source of address stability.
        //
        // Caller contract: do not re-enter `get_jitcode` (or anything
        // else that can call `HashMap::insert` on `self.jitcodes`, such
        // as the `merge_point_pc` refinement path) while still holding
        // the returned reference. A refinement drops the previous Box
        // and dangles the ref.
        unsafe { &*(&**entry as *const PyJitCode) }
    }
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new()
    }
}
